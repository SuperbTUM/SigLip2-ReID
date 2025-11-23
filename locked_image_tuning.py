import model
from constants import *
from data_preparation import *
from checkpoint import *

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


class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0

    def forward(self, text_features, image_features, t_label, i_targets):
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device)

        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss

class MoCoInfoNCELoss(nn.Module):
    def __init__(self, 
                 feature_dim: int, 
                 queue_size: int = 32,
                 momentum: float = 0.999,
                 temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.queue_size = queue_size
        self.momentum = momentum

        # --- Queues ---
        self.register_buffer("image_queue", torch.randn(queue_size, feature_dim))
        self.register_buffer("text_queue", torch.randn(queue_size, feature_dim))
        self.register_buffer("queue_pids", torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # --- Optional momentum encoders ---
        self.image_encoder_m = None
        self.text_encoder_m = None

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_img, keys_txt, pids):
        batch_size = keys_txt.size(0)
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0
        self.image_queue[ptr:ptr+batch_size, :] = keys_img
        self.text_queue[ptr:ptr+batch_size, :] = keys_txt
        self.queue_pids[ptr:ptr+batch_size] = pids
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    @torch.no_grad()
    def _momentum_update(self, model, model_m):
        for param_q, param_k in zip(model.parameters(), model_m.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(
        self,
        image_features,             # precomputed image embeddings [B,D]
        text_features,              # precomputed text embeddings [B,D]
        pid_labels=None,            # batch IDs
        image_encoder=None,         # optional main image encoder for momentum
        image_inputs=None,          # optional raw image input for momentum
        text_encoder=None,          # optional main text encoder for momentum
        text_inputs=None            # optional raw text input for momentum
    ):
        # --- Temperature ---
        T = self.temperature.exp().clamp(0.01, 10.0)

        # --- In-batch logits (no normalization) ---
        logits_i2t = image_features @ text_features.t() * T
        logits_t2i = logits_i2t.t().clone()
        labels = torch.arange(logits_i2t.size(0), device=image_features.device)

        same_id_mask_inbatch = pid_labels.unsqueeze(1) == pid_labels.unsqueeze(0)
        same_id_mask_inbatch.fill_diagonal_(False)  # allow self-positive
        logits_i2t = logits_i2t.masked_fill(same_id_mask_inbatch, torch.finfo(logits_i2t.dtype).min)
        logits_t2i = logits_t2i.masked_fill(same_id_mask_inbatch, torch.finfo(logits_t2i.dtype).min)

        # --- Queue logits ---
        with torch.no_grad():
            queue_img = self.image_queue.clone().detach()
            queue_txt = self.text_queue.clone().detach()
            queue_pids = self.queue_pids.clone().detach()

        logits_i2t_queue = image_features @ queue_txt.t() * T
        logits_t2i_queue = text_features @ queue_img.t() * T

        # --- Mask false negatives ---
        if pid_labels is not None:
            same_id_mask = pid_labels.unsqueeze(1) == queue_pids.unsqueeze(0)
            logits_i2t_queue = logits_i2t_queue.masked_fill(same_id_mask, torch.finfo(logits_i2t_queue.dtype).min)
            logits_t2i_queue = logits_t2i_queue.masked_fill(same_id_mask, torch.finfo(logits_t2i_queue.dtype).min)

        # --- Concatenate logits ---
        logits_i2t = torch.cat([logits_i2t, logits_i2t_queue], dim=1)
        logits_t2i = torch.cat([logits_t2i, logits_t2i_queue], dim=1)

        # --- InfoNCE loss ---
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
        loss = 0.5 * (loss_i2t + loss_t2i)

        # --- Momentum encoder updates ---
        with torch.no_grad():
            # Image momentum
            if image_encoder is not None:
                if self.image_encoder_m is None:
                    self.image_encoder_m = copy.deepcopy(image_encoder)
                    for p in self.image_encoder_m.parameters():
                        p.requires_grad = False
                self._momentum_update(image_encoder, self.image_encoder_m)
                img_keys = self.image_encoder_m(image_inputs).detach() if image_inputs is not None else image_features
            else:
                img_keys = image_features  # frozen

            # Text momentum
            if text_encoder is not None:
                if self.text_encoder_m is None:
                    self.text_encoder_m = copy.deepcopy(text_encoder)
                    for p in self.text_encoder_m.parameters():
                        p.requires_grad = False
                self._momentum_update(text_encoder, self.text_encoder_m)
                txt_keys = self.text_encoder_m(text_inputs).detach() if text_inputs is not None else text_features
            else:
                txt_keys = text_features  # frozen

            # Enqueue
            self._dequeue_and_enqueue(img_keys, txt_keys, pid_labels)

        return loss
    
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
        self.prompt = nn.Parameter(ai_text_embedding).to(device)

        self.register_buffer("class_name_embedding", class_embedding)
        self.register_buffer("ai_text_embedding", ai_text_embedding)

    def forward(self, text_model, domain_ids):
        # --- Mix the AI-generated and learnable prompts ---
        # prompt: (N, P, D)
        # ai_prompt_embs: (N, L, D)
        # class_name_embs: (N, L', D)
        if isinstance(domain_ids, int):
            domain_ids = torch.ones((self.prompt.size(0),), dtype=torch.long, device=self.prompt.device) * domain_ids
        # Prepend the learnable prompt to the class name embeddings
        # [PROMPT, PROMPT, ..., CLASS_NAME]
        combined_embs = torch.cat([self.prompt, self.class_name_embedding], dim=1)
        
        # Pass the combined embeddings through the rest of the text encoder
        # This part requires a custom forward pass through the text model layers
        # For simplicity, we assume we can pass embeddings directly. 
        # In transformers, you pass it to the encoder layers.
        
        # A simplified representation of passing through the encoder:
        # Note: The actual `transformers` implementation requires passing embeddings
        # through `model.text_model.encoder` and then `model.text_model.final_layer_norm`.
        prompted_text_features, prompted_text_hidden_state = text_model(inputs_embeds=combined_embs, domain_ids=domain_ids)
        
        return prompted_text_features, prompted_text_hidden_state

def tuning_preparation(class_names, 
                       device):
    # --- Setup from before ---
    base_model = model.load_weights(MODEL_NAME)
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
    )
    base_model.text_model = get_peft_model(base_model.text_model, lora_config)
    lora_model = base_model.to(device)
    # --- End of previous setup ---


    # 1. Instantiate the Prompt Learner
    # We need to know the embedding dimension of our text model.
    embedding_dim = lora_model.config.text_config.hidden_size
    prompt_learner = PromptLearner(
        text_tokenizer=text_tokenizer,
        num_prompt_tokens=N_PROMPT_TOKEN, # hyper-parameter
        embedding_dim=embedding_dim,
        class_names=class_names
    ).to(device)

    # 2. Define the Optimizer to train BOTH sets of parameters
    # This is the crucial step: you combine the parameters from both the
    # prompt_learner and the LoRA-adapted model.
    trainable_params = list(prompt_learner.parameters()) + list(lora_model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=3.5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_LoRA, eta_min=1e-6)
    sup_con_loss = SupConLoss(device)
    scaler = GradScaler(device)

    return lora_model, prompt_learner, optimizer, scheduler, scaler, sup_con_loss


def LoRA_tuning(dataset_name, 
                input_size, 
                class_names,
                device):
    train_dataloader, _, n_cls = create_dataloader(dataset_name, input_size, "train", False)
    if isinstance(class_names, str):
        class_names = [class_names] * n_cls
    lora_model, prompt_learner, optimizer, scheduler, scaler, sup_con_loss = tuning_preparation(class_names, device)
    image_features_list = []
    image_label_list = []
    with torch.inference_mode():
        for batch in train_dataloader:
            image_tensor, label = batch[:2]
            image_tensor = image_tensor.to(device)
            label = label.to(device)
            image_features = lora_model.get_image_features(image_tensor)[0]
            image_features_list.append(image_features)
            image_label_list.append(label)
    
    image_features_list = torch.cat(image_features_list, dim=0)
    image_label_list = torch.cat(image_label_list, dim=0)
    pk_sampler = PKsamplerWithLabels(image_label_list.cpu().tolist(), BATCH_SIZE // N_INSTANCE, N_INSTANCE)

    for epoch in range(N_EPOCHS_LoRA):
        loss_by_epoch = 0
        for iters, (indices_batch, label_batch) in enumerate(pk_sampler):
            image_features_batch = image_features_list[indices_batch]
            label_batch = torch.tensor(label_batch, device=device)
            
            optimizer.zero_grad()

            with autocast(device):
                # 3. Use the Prompt Learner
                # Pass the initial embeddings through the prompt learner to get the combined embeddings.
                modified_text_embeddings = prompt_learner(lora_model.text_model)[0]

                # 4. Forward pass through the LoRA-adapted model
                # The model now receives the modified input.
                # Note: This is a simplified forward pass. The actual call might differ based on the model.
                # The key is that `modified_text_embeddings` is the input, not the original `token_embeddings`.
                text_features = modified_text_embeddings[label_batch]
                
                # Your loss calculation and backpropagation would follow...
                loss = sup_con_loss(text_features, image_features_batch, label_batch, label_batch) + \
                        sup_con_loss(image_features_batch, text_features, label_batch, label_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_by_epoch += loss.item()
        print("Avg loss at epoch {} is {}.".format(epoch, loss_by_epoch / iters))
    return lora_model.eval()


def LoRA_tuning_variable_dataset(dataset_names,
                                 input_sizes,
                                 class_names_list,
                                 device):
    # --- Shared model and optimizer ---
    base_model = model.load_weights(MODEL_NAME)
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        use_dora=True,
        init_lora_weights="pissa_niter_4"
    )
    base_model.text_model = get_peft_model(base_model.text_model, lora_config)
    lora_model = base_model.to(device)
    embedding_dim = lora_model.config.text_config.hidden_size

    # --- Dataloaders and Prompt Learners for each dataset ---
    train_dataloaders = []
    prompt_learners = []
    all_trainable_params = list(lora_model.parameters())
    for name, m in lora_model.text_model.named_modules():
        if "dal" in name.lower():
            for p in m.parameters():
                p.requires_grad = True

    for dataset_name, input_size, class_names in zip(dataset_names, input_sizes, class_names_list):
        train_dataloader, _, n_cls, ai_prompts = create_dataloader(dataset_name, input_size, "train", False, True)
        train_dataloaders.append(train_dataloader)
        
        if isinstance(class_names, str):
            class_names = [class_names] * n_cls

        prompt_learner = PromptLearner(
            text_tokenizer=text_tokenizer,
            token_embedding=lora_model.text_model.get_input_embeddings(),
            num_prompt_tokens=N_PROMPT_TOKEN,
            embedding_dim=embedding_dim,
            device=device,
            class_names=class_names,
            init_prompts=ai_prompts
        )
        prompt_learners.append(prompt_learner)
        all_trainable_params.extend(list(prompt_learner.parameters()))

    sup_con_loss = MoCoInfoNCELoss(embedding_dim).to(device)
    scaler = GradScaler(device)
    optimizer = torch.optim.Adam(
        all_trainable_params + list(sup_con_loss.parameters()), lr=3.5e-3, weight_decay=1e-4)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,      
        end_factor=1.0,
        total_iters=10
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_LoRA-10, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[10]
    )
    
    # --- Pre-extract image features for all datasets ---
    image_features_lists = []
    image_hidden_state_lists = []
    image_label_lists = []
    with torch.inference_mode():
        for i, train_dataloader in enumerate(train_dataloaders):
            image_features_list = []
            image_hidden_state_list = []
            image_label_list = []
            for batch in train_dataloader:
                image_tensor, label = batch[:2]
                image_tensor = image_tensor.to(device)
                label = label.to(device)
                # domain_ids = torch.ones_like(label) * i
                image_features, image_last_hidden_state = lora_model.get_image_features(image_tensor)
                image_features_list.append(image_features)
                image_hidden_state_list.append(image_last_hidden_state)
                image_label_list.append(label)
            image_features_lists.append(torch.cat(image_features_list, dim=0).to(device))
            image_hidden_state_lists.append(torch.cat(image_hidden_state_list, dim=0).to(device))
            image_label_lists.append(torch.cat(image_label_list, dim=0))

    # --- Samplers for each dataset ---
    pk_samplers = [
        PKsamplerWithLabels(labels.cpu().tolist(), BATCH_SIZE // N_INSTANCE, N_INSTANCE)
        for labels in image_label_lists
    ]
    
    # --- Training Loop ---
    # Use itertools.cycle to handle datasets of different sizes
    num_batches = max(len(sampler) for sampler in pk_samplers)
    # Create cyclic iterators
    sampler_iters = [itertools.cycle(sampler) for sampler in pk_samplers]

    for epoch in range(N_EPOCHS_LoRA):
        loss_by_epoch = 0
        
        for step, (i, sampler_iter) in enumerate(itertools.cycle(enumerate(sampler_iters))):
            if step >= num_batches:  # stop after desired batches
                break
            optimizer.zero_grad()
            total_loss = 0

            indices_batch, label_batch = next(sampler_iter)
            
            image_features_batch = image_features_lists[i][indices_batch]
            image_hidden_state_batch = image_hidden_state_lists[i][indices_batch]
            label_batch = torch.tensor(label_batch, device=device)
            domain_batch = i % 1

            with autocast(device):
                modified_text_embeddings, modified_text_hidden_states = prompt_learners[i](lora_model.text_model, domain_batch)
                text_features = modified_text_embeddings[label_batch]
                text_hidden_state = modified_text_hidden_states[label_batch]
                
                loss = sup_con_loss(image_features_batch, text_features, label_batch) + \
                        0.25 * sup_con_loss(image_hidden_state_batch, text_hidden_state, label_batch)
                total_loss += loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_by_epoch += total_loss.item()
        scheduler.step()

        print("Epoch: {}, Avg loss: {}".format(epoch, loss_by_epoch / num_batches))
    
    base_model.text_model = lora_model.text_model.merge_and_unload()
    for prompt_learner in prompt_learners:
        prompt_learner.eval()
    save_checkpoint(base_model.eval(), prompt_learners, sup_con_loss.temperature.exp().item(), N_EPOCHS_LoRA, optimizer, scheduler)
    return base_model.eval(), prompt_learners, sup_con_loss.temperature.exp().item()

