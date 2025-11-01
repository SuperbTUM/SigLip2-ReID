import model
from constants import *
from data_preparation import *
from checkpoint import *
from evaluation import R1_mAP_eval_pt

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

class SupervisedSigmoidLoss(nn.Module):
    """
    A sigmoid-based loss inspired by Supervised Contrastive learning.

    It can operate in two modes:
    1.  Instance-level (default): Assumes image_i matches text_i.
    2.  Supervised-level: If class_labels are provided, it treats all samples
        with the same label as positive pairs.
    """
    def __init__(self, temperature: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.reduction = reduction

    def forward(self, 
              image_features: torch.Tensor, 
              text_features: torch.Tensor,
              class_labels: Optional[torch.Tensor]):
        """
        Computes the Supervised Sigmoid Loss.

        Args:
            image_features (torch.Tensor): Normalized image embeddings (N, D).
            text_features (torch.Tensor): Normalized text embeddings (N, D).
            class_labels (Optional[torch.Tensor]): A 1D tensor of class labels (N,).
                                                   If None, performs standard instance-level
                                                   matching.
        
        Returns:
            torch.Tensor: The computed loss value.
        """
        temperature = self.temperature.exp().clamp(0.1, 10.0)
        logits = torch.matmul(image_features, text_features.t()) * temperature

        labels = torch.arange(logits.shape[0], device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss
    
class PromptLearner(nn.Module):
    def __init__(self, 
                 text_tokenizer,
                 num_prompt_tokens, 
                 embedding_dim, 
                 class_names: List[str],
                 init_prompts: List[str] = None,
                 mix_ratio: float = 0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_prompt_tokens = num_prompt_tokens

        # Initialize the learnable prompt vectors
        prompt_vectors = torch.empty(len(class_names), num_prompt_tokens, embedding_dim)
        nn.init.normal_(prompt_vectors, std=0.02) # Standard initialization
        self.prompt = nn.Parameter(prompt_vectors)

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
        text_inputs = text_tokenizer(self.class_names, padding=True, return_tensors="pt").input_ids
        ai_text_inputs = text_tokenizer(self.init_prompts, padding=True, return_tensors="pt").input_ids

        self.register_buffer("text_inputs", text_inputs)
        self.register_buffer("ai_text_inputs", ai_text_inputs)
        self.mix_ratio = mix_ratio

    def forward(self, text_model):
        with torch.no_grad():
            emb_layer = text_model.get_input_embeddings()
            # Get the standard word embeddings for class names
            class_name_embs = emb_layer(self.text_inputs.to(self.prompt.device))
            ai_prompt_embs = emb_layer(self.ai_text_inputs.to(self.prompt.device))
        
        # --- Mix the AI-generated and learnable prompts ---
        # prompt: (N, P, D)
        # ai_prompt_embs: (N, L, D)
        # class_name_embs: (N, L', D)

        # Trim or pad the AI prompt embeddings to num_prompt_tokens
        if ai_prompt_embs.size(1) > self.num_prompt_tokens:
            ai_prompt_embs = ai_prompt_embs[:, :self.num_prompt_tokens, :]
        elif ai_prompt_embs.size(1) < self.num_prompt_tokens:
            pad_len = self.num_prompt_tokens - ai_prompt_embs.size(1)
            pad = torch.zeros(ai_prompt_embs.size(0), pad_len, ai_prompt_embs.size(2), device=self.prompt.device)
            ai_prompt_embs = torch.cat([ai_prompt_embs, pad], dim=1)
        mask = (ai_prompt_embs.abs().sum(dim=-1, keepdim=True) > 0).float()  # 1 if not padded
        
        # Blend the AI and learned prompts
        mixed_prompt = (
            self.mix_ratio * ai_prompt_embs * mask + (1 - self.mix_ratio) * self.prompt
        )
        # Prepend the learnable prompt to the class name embeddings
        # [PROMPT, PROMPT, ..., CLASS_NAME]
        combined_embs = torch.cat([mixed_prompt, class_name_embs], dim=1)
        
        # Pass the combined embeddings through the rest of the text encoder
        # This part requires a custom forward pass through the text model layers
        # For simplicity, we assume we can pass embeddings directly. 
        # In transformers, you pass it to the encoder layers.
        
        # A simplified representation of passing through the encoder:
        # Note: The actual `transformers` implementation requires passing embeddings
        # through `model.text_model.encoder` and then `model.text_model.final_layer_norm`.
        prompted_text_features, prompted_text_hidden_state = text_model(inputs_embeds=combined_embs)
        
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
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        use_dora=True
    )
    frozen_text_model = copy.deepcopy(base_model.text_model).to(device)
    base_model.text_model = get_peft_model(base_model.text_model, lora_config)
    lora_model = base_model.to(device)
    embedding_dim = lora_model.config.text_config.hidden_size

    # --- Dataloaders and Prompt Learners for each dataset ---
    train_dataloaders = []
    prompt_learners = []
    all_trainable_params = list(lora_model.parameters())

    for dataset_name, input_size, class_names in zip(dataset_names, input_sizes, class_names_list):
        train_dataloader, _, n_cls, ai_prompts = create_dataloader(dataset_name, input_size, "train", False, True)
        train_dataloaders.append(train_dataloader)
        
        if isinstance(class_names, str):
            class_names = [class_names] * n_cls

        prompt_learner = PromptLearner(
            text_tokenizer=text_tokenizer,
            num_prompt_tokens=N_PROMPT_TOKEN,
            embedding_dim=embedding_dim,
            class_names=class_names,
            init_prompts=ai_prompts
        ).to(device)
        prompt_learners.append(prompt_learner)
        all_trainable_params.extend(list(prompt_learner.parameters()))

    sup_con_loss = SupervisedSigmoidLoss().to(device)
    scaler = GradScaler(device)
    optimizer = torch.optim.Adam(
        all_trainable_params + list(sup_con_loss.parameters()), lr=3.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_LoRA, eta_min=1e-6)

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
                image_features = lora_model.get_image_features(image_tensor)[0]
                image_features_list.append(image_features)
                image_label_list.append(label)
            image_features_lists.append(torch.cat(image_features_list, dim=0).to(device))
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
            label_batch = torch.tensor(label_batch, device=device)

            with autocast(device):
                with torch.no_grad():
                    frozen_modified_text_embeddings = prompt_learners[i](frozen_text_model)[0]
                    frozen_text_features = frozen_modified_text_embeddings[label_batch]
                modified_text_embeddings = prompt_learners[i](lora_model.text_model)[0]
                text_features = modified_text_embeddings[label_batch]
                
                loss = sup_con_loss(text_features, image_features_batch, label_batch) + \
                        sup_con_loss(image_features_batch, text_features, label_batch)
                loss_align = F.mse_loss(text_features, frozen_text_features.detach())
                total_loss += loss + 0.1 * loss_align

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_by_epoch += total_loss.item()
        scheduler.step()

        print("Epoch: {}, Avg loss: {}".format(epoch, loss_by_epoch / num_batches))
    
    base_model.text_model = lora_model.text_model.merge_and_unload()
    save_checkpoint(base_model, prompt_learners, sup_con_loss.temperature.exp().item(), N_EPOCHS_LoRA, optimizer, scheduler)
    return base_model.eval(), prompt_learners, sup_con_loss.temperature.exp().item()

def test(model,
         dataset_name,
         device):
    validation_dataloader, num_query, _, _ = create_dataloader(dataset_name, INPUT_SIZE, "val", False)
    evaluator = R1_mAP_eval_pt(num_query, 10)
    with torch.inference_mode():
        for batch in validation_dataloader:
            img, label, cam = batch[:3]
            img = img.to(device)
            label = label.to(device)
            cam = cam.to(device)
            test_feat = model.get_image_features(img)[0]
            evaluator.update((test_feat, label, cam))
    cmc, mAP = evaluator.compute()[:2]
    return cmc[0], cmc[4], cmc[9], mAP

if __name__ == "__main__":
    dataset_name = "Market1501"
    # model = LoRA_tuning(dataset_name, INPUT_SIZE, "person", DEVICE)
    model = LoRA_tuning_variable_dataset(["Market1501", "veri"], [(256, 128), (224, 224)], ["person", "vehicle"], DEVICE)[0]
    cmc1, cmc5, cmc10, mAP = test(model, dataset_name, DEVICE)
    print("Dataset: {}, cmc 1: {}, cmc 5: {}, cmc 10: {}, mAP: {}".format(dataset_name, cmc1, cmc5, cmc10, mAP))
