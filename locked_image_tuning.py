import model
from constants import *
from data_preparation import *
from checkpoint import *
# from teacher import teacher_model_output
from losses import MaxSimLoss, HardTextQueueLoss, lora_orthogonality_loss, collect_trainable_lora_As, SupConLoss

import math
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from peft import get_peft_model, LoraConfig
from typing import List, Optional
import itertools

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PromptLearner(nn.Module):
    def __init__(self, 
                 text_tokenizer,
                 token_embedding,
                 num_prompt_tokens, 
                 embedding_dim, 
                 device,
                 class_names: List[str],
                 init_prompts: List[str] = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_prompt_tokens = num_prompt_tokens

        # Store tokenized class names
        self.class_names = class_names

        # --- Optional AI-generated prompts ---
        if init_prompts is not None:
            assert len(init_prompts) == len(class_names), \
                "init_prompts must match number of class names"
            self.init_prompts = init_prompts
        else:
            self.init_prompts = [f"a photo of a {name}" for name in class_names]
        # Note: In a full implementation, you'd handle tokenization carefully here.
        text_inputs = text_tokenizer(self.class_names, padding=True, return_tensors="pt").input_ids.to(device)
        ai_text_inputs = text_tokenizer(self.init_prompts, padding=True, return_tensors="pt").input_ids.to(device)

        self.register_buffer("text_inputs", text_inputs)
        self.register_buffer("ai_text_inputs", ai_text_inputs)

        # add positional embedding
        with torch.no_grad():
            class_embedding = token_embedding(text_inputs)
            ai_text_embedding = token_embedding(ai_text_inputs)[:, 1:, :]
        
        # Initialize the learnable prompt vectors
        init_prompt = torch.randn(len(class_names), num_prompt_tokens, embedding_dim, dtype=ai_text_embedding.dtype, device=device)
        nn.init.normal_(init_prompt, std=0.02)
        self.prompt = nn.Parameter(init_prompt)

        self.register_buffer("class_name_embedding", class_embedding)
        self.register_buffer("ai_text_embedding", ai_text_embedding)

    def forward(self, text_model, domain_ids):
        # --- Mix the AI-generated and learnable prompts ---
        # prompt: (N, P, D)
        # ai_prompt_embs: (N, L, D)
        # class_name_embs: (N, L', D)
        # Append the learnable prompt to the class name embeddings
        # [PROMPT, PROMPT, ..., CLASS_NAME]
        combined_embs = torch.cat([self.class_name_embedding, self.prompt], dim=1)
        
        prompted_text_features = text_model(inputs_embeds=combined_embs, domain_ids=domain_ids)
        
        return prompted_text_features


def tuning_vision_projection(dataset_names,
                             input_sizes,
                             device):
    base_model = model.load_weights(MODEL_NAME)
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    ai_prompts = []

    for dataset_name in dataset_names:
        ai_prompt = []
        with open(f"prompts_{dataset_name}_full.txt", "r", encoding="utf-8") as f:
            ai_prompt += [prompt.strip() for prompt in f.readlines()]
        token_ai_prompt = text_tokenizer(ai_prompt, padding=True, return_tensors="pt").input_ids.to(device)
        ai_prompts.append(token_ai_prompt)
    base_model.train()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        use_dora=True,
        init_lora_weights="pissa_niter_4"
    )
    base_model.vision_model = get_peft_model(base_model.vision_model, lora_config)
    for param in base_model.text_model.parameters():
        param.requires_grad = False
    for name, param in base_model.vision_model.named_parameters():
        if "dal" in name.lower():
            param.requires_grad = False
    base_model = base_model.to(device)
    all_trainable_params = list(filter(lambda p: p.requires_grad, base_model.parameters()))

    real_sup_con_loss = MaxSimLoss(device)
    info_nce_loss = SupConLoss(device)
    scaler = GradScaler(device)
    optimizer = torch.optim.Adam(all_trainable_params, lr=1e-3, weight_decay=1e-4)
    temperature_optimizer = torch.optim.Adam(list(real_sup_con_loss.parameters()) + list(info_nce_loss.parameters()), lr=1e-3)

    train_dataloaders = []
    n_clses = []
    for dataset_name, input_size in zip(dataset_names, input_sizes):
        train_dataloader, _, n_cls, _ = create_dataloader(dataset_name, input_size, "train", True, False)
        train_dataloaders.append(train_dataloader)
        n_clses.append(n_cls)
    
    frozen_text_features_list = []
    with torch.inference_mode():
        batch_size = 32
        frozen_text_feature_list = []
        for j, n_cls in enumerate(n_clses):
            for i in range(0, n_cls, batch_size):
                text_features = base_model.text_model(ai_prompts[j][i:i+batch_size])
                frozen_text_feature_list.append(text_features)
            frozen_text_features_list.append(torch.cat(frozen_text_feature_list, dim=0))
    
    # --- Training Loop ---
    num_batches = max(len(loader) for loader in train_dataloaders)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-3,
        total_steps=num_batches * N_EPOCHS_PRESTAGE,
        pct_start=0.2,
        final_div_factor=10
    )

    for epoch in range(N_EPOCHS_PRESTAGE):
        loss_by_epoch = 0
        dataloader_iters = [itertools.cycle(loader) for loader in train_dataloaders]
        round_robin_iter = itertools.cycle(enumerate(dataloader_iters))
        
        for batch_idx in range(num_batches):

            i, dataloader_iter = next(round_robin_iter)

            image_tensor, label_batch = next(dataloader_iter)[:2]

            optimizer.zero_grad()
            temperature_optimizer.zero_grad()
            total_loss = 0

            label_batch = label_batch.to(device)
            domain_batch = i % 1
            frozen_text_feature = frozen_text_features_list[i][label_batch]

            with autocast(device):
                image_features_batch = base_model.get_image_features(image_tensor.to(device), domain_ids=domain_batch)[0]
                
                image_contrastive_loss = real_sup_con_loss(image_features_batch, label_batch)
                image_text_loss = info_nce_loss(image_features_batch, frozen_text_feature.detach(), label_batch, label_batch)

                loss = image_contrastive_loss + 0.1 * image_text_loss 
                total_loss += loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.step(temperature_optimizer)
            scaler.update()
            loss_by_epoch += total_loss.item()
            scheduler.step()

        print("Epoch: {}, Avg loss: {}".format(epoch, loss_by_epoch / num_batches))
    base_model.vision_model = base_model.vision_model.merge_and_unload()
    save_checkpoint(base_model.eval(), None, real_sup_con_loss.temperature.exp().item(), N_EPOCHS_PRESTAGE, optimizer, scheduler)
    return base_model.eval()


def LoRA_tuning_variable_dataset(base_model,
                                 dataset_names,
                                 input_sizes,
                                 class_names_list,
                                 device):
    # --- Shared model and optimizer ---
    if os.path.exists(f"checkpoint_epoch_{N_EPOCHS_PRESTAGE}.pth"):
        base_model, _, _, _, _ = load_checkpoint(base_model, None, None, None, f"checkpoint_epoch_{N_EPOCHS_PRESTAGE}.pth", device)
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        use_dora=True,
        init_lora_weights="pissa_niter_4"
    )
    base_model.text_model = get_peft_model(base_model.text_model, lora_config)
    base_model.vision_model.eval()
    for param in base_model.vision_model.parameters():
        param.requires_grad = False
    lora_model = base_model.to(device)
    embedding_dim = lora_model.config.text_config.hidden_size

    # --- Dataloaders and Prompt Learners for each dataset ---
    train_dataloaders = []
    prompt_learners = []
    full_prompts = []
    n_clses = []
    text_encoder_params = list(filter(lambda p: p.requires_grad, lora_model.parameters()))
    prompt_learner_params = []
    lora_params = collect_trainable_lora_As(lora_model)

    for dataset_name, input_size, class_names in zip(dataset_names, input_sizes, class_names_list):
        train_dataloader, _, n_cls, ai_prompts = create_dataloader(dataset_name, input_size, "train", False, True)
        train_dataloaders.append(train_dataloader)
        n_clses.append(n_cls)
        full_prompts.append(text_tokenizer(ai_prompts, padding=True, return_tensors="pt").input_ids.to(device))

        if isinstance(class_names, str):
            class_names = [class_names] * n_cls

        prompt_learner = PromptLearner(
            text_tokenizer=text_tokenizer,
            token_embedding=lora_model.text_model.get_input_embeddings(),
            num_prompt_tokens=N_PROMPT_TOKEN,
            embedding_dim=embedding_dim,
            device=device,
            class_names=class_names
        )
        prompt_learners.append(prompt_learner)
        prompt_learner_params.extend(list(prompt_learner.parameters()))

    sup_con_loss = HardTextQueueLoss(embedding_dim).to(device)
    scaler = GradScaler(device)
    text_encoder_optimizer = torch.optim.Adam(
        text_encoder_params, lr=3.5e-3, weight_decay=1e-4)
    prompt_learner_optimizer = torch.optim.Adam(
        prompt_learner_params, lr=3.5e-3, weight_decay=1e-4
    )
    temperature_optimizer = torch.optim.Adam(list(sup_con_loss.parameters()), lr=3.5e-3)

    prompt_scheduler = torch.optim.lr_scheduler.SequentialLR(
        prompt_learner_optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                prompt_learner_optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=5
            ),
            torch.optim.lr_scheduler.LambdaLR(
                prompt_learner_optimizer,
                lr_lambda=lambda _: 1.0
            )
        ],
        milestones=[5]
    )
    
    # --- Pre-extract image features for all datasets ---
    image_features_lists = []
    image_label_lists = []
    with torch.inference_mode():
        for i, train_dataloader in enumerate(train_dataloaders):
            image_features_list = []
            image_label_list = []
            for batch in train_dataloader:
                image_tensor, label = batch[:2]
                image_tensor = image_tensor.to(device)
                label = label.to(device)
                domain_ids = i
                image_features = lora_model.get_image_features(image_tensor, domain_ids=domain_ids)[0]
                image_features_list.append(image_features)
                image_label_list.append(label)
            image_features_lists.append(torch.cat(image_features_list, dim=0))
            image_label_lists.append(torch.cat(image_label_list, dim=0))
        # teacher_model_outputs = teacher_model_output(train_dataloaders, ai_prompts)
    # --- Samplers for each dataset ---
    pk_samplers = [
        PKsamplerWithLabels(labels.cpu().tolist(), BATCH_SIZE // N_INSTANCE, N_INSTANCE)
        for labels in image_label_lists
    ]
    
    # --- Training Loop ---
    # Use itertools.cycle to handle datasets of different sizes
    num_batches = max(len(sampler) for sampler in pk_samplers)

    for epoch in range(N_EPOCHS_LoRA):
        loss_by_epoch = 0

        with torch.inference_mode():
            batch_size = 32
            frozen_text_features_list = []
            for j, n_cls in enumerate(n_clses):
                frozen_text_feature_list = []
                for i in range(0, n_cls, batch_size):
                    text_features = lora_model.text_model(full_prompts[j][i:i+batch_size])
                    frozen_text_feature_list.append(text_features)
                frozen_text_features_list.append(torch.cat(frozen_text_feature_list, dim=0))

        # Create cyclic iterators
        sampler_iters = [itertools.cycle(sampler) for sampler in pk_samplers]
        if epoch == 10:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                text_encoder_optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        text_encoder_optimizer,
                        start_factor=0.1,      
                        end_factor=1.0,
                        total_iters=10
                    ), 
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        text_encoder_optimizer, 
                        T_max=N_EPOCHS_LoRA-20, 
                        eta_min=1e-6)
                ],
                milestones=[10]
            )
        
        for step, (i, sampler_iter) in enumerate(itertools.cycle(enumerate(sampler_iters))):
            if step >= num_batches:  # stop after desired batches
                break
            if epoch >= 10:
                text_encoder_optimizer.zero_grad()
            prompt_learner_optimizer.zero_grad()
            temperature_optimizer.zero_grad()
            total_loss = 0

            indices_batch, label_batch = next(sampler_iter)
            
            image_features_batch = image_features_lists[i][indices_batch]
            label_batch = torch.tensor(label_batch, device=device)
            frozen_text_features_batch = frozen_text_features_list[i][label_batch]
            sup_con_loss.update_hard_text(frozen_text_features_batch)
            domain_batch = i % 1

            with autocast(device):
                modified_text_embeddings = prompt_learners[i](lora_model.text_model, domain_batch)
                text_features = modified_text_embeddings[label_batch]
                
                loss = sup_con_loss(image_features_batch, text_features, label_batch) + \
                        1e-4 * lora_orthogonality_loss(lora_params) + \
                        (1 - F.cosine_similarity(text_features, frozen_text_features_batch.detach(), dim=1)).mean()
                total_loss += loss

            scaler.scale(total_loss).backward()
            if epoch >= 10:
                scaler.step(text_encoder_optimizer)
            scaler.step(prompt_learner_optimizer)
            scaler.step(temperature_optimizer)
            scaler.update()
            loss_by_epoch += total_loss.item()
        if epoch >= 10:
            scheduler.step()
        prompt_scheduler.step()

        print("Epoch: {}, Avg loss: {}".format(epoch, loss_by_epoch / num_batches))
    
    base_model.text_model = lora_model.text_model.merge_and_unload()
    for prompt_learner in prompt_learners:
        prompt_learner.eval()
    save_checkpoint(base_model.eval(), prompt_learners, sup_con_loss.temperature.exp().item(), N_EPOCHS_LoRA, text_encoder_optimizer, scheduler)
    return base_model.eval(), prompt_learners, sup_con_loss.temperature.exp().item()

