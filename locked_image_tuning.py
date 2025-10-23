import model
from constants import *
from data_preparation import *
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
    def __init__(self, temperature: float = 10.0, reduction: str = 'mean'):
        super().__init__()
        self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.reduction = reduction

    def forward(self, 
              image_features: torch.Tensor, 
              text_features: torch.Tensor,
              class_labels: Optional[torch.Tensor] = None):
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
        temperature = self.temperature.exp().clamp(1, 100)
        logits = torch.matmul(image_features, text_features.t()) * temperature

        if class_labels is None:
            # Standard instance-level matching (image_i matches text_i)
            labels = torch.eye(logits.shape[0], device=logits.device)
        else:
            # Supervised matching: pairs with the same class label are positives.
            # Use broadcasting to create a matrix of label equality.
            # Shape: (N, N)
            labels = (class_labels.unsqueeze(0) == class_labels.unsqueeze(1)).float()

        # We must ignore the loss for an item matched with itself if it's not
        # the designated text pair in the instance-level case. In the supervised
        # case, an item is always positive with itself.
        # For simplicity here, we assume the main diagonal should always be positive.
        if class_labels is not None:
             labels.fill_diagonal_(1)


        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction=self.reduction
        )

        return loss
    
class PromptLearner(nn.Module):
    def __init__(self, 
                 text_tokenizer,
                 num_prompt_tokens, 
                 embedding_dim, 
                 class_names: List[str]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_prompt_tokens = num_prompt_tokens

        # Initialize the learnable prompt vectors
        prompt_vectors = torch.empty(len(class_names), num_prompt_tokens, embedding_dim)
        nn.init.normal_(prompt_vectors, std=0.02) # Standard initialization
        self.prompt = nn.Parameter(prompt_vectors)

        # Store tokenized class names
        self.class_names = class_names
        # Note: In a full implementation, you'd handle tokenization carefully here.
        text_inputs = text_tokenizer(self.class_names, padding=True, return_tensors="pt").input_ids

        self.register_buffer("text_inputs", text_inputs)

    def forward(self, text_model):
        with torch.no_grad():
            # Get the standard word embeddings for class names
            class_name_embs = text_model.get_input_embeddings()(self.text_inputs.to(self.prompt.device))
        
        # Prepend the learnable prompt to the class name embeddings
        # [PROMPT, PROMPT, ..., CLASS_NAME]
        combined_embs = torch.cat([self.prompt, class_name_embs], dim=1)
        
        # Pass the combined embeddings through the rest of the text encoder
        # This part requires a custom forward pass through the text model layers
        # For simplicity, we assume we can pass embeddings directly. 
        # In transformers, you pass it to the encoder layers.
        
        # A simplified representation of passing through the encoder:
        # Note: The actual `transformers` implementation requires passing embeddings
        # through `model.text_model.encoder` and then `model.text_model.final_layer_norm`.
        prompted_text_features = text_model(inputs_embeds=combined_embs)
        
        return prompted_text_features

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
    with torch.no_grad():
        for batch in train_dataloader:
            image_tensor, label = batch[:2]
            image_tensor = image_tensor.to(device)
            label = label.to(device)
            image_features = lora_model.get_image_features(image_tensor)
            image_features_list.append(image_features)
            image_label_list.append(label)
    
    image_features_list = torch.cat(image_features_list, dim=0).to(device)
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
                modified_text_embeddings = prompt_learner(lora_model.text_model)

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
    base_model.text_model = get_peft_model(base_model.text_model, lora_config)
    lora_model = base_model.to(device)
    embedding_dim = lora_model.config.text_config.hidden_size

    # --- Dataloaders and Prompt Learners for each dataset ---
    train_dataloaders = []
    prompt_learners = []
    all_trainable_params = list(lora_model.parameters())

    for dataset_name, input_size, class_names in zip(dataset_names, input_sizes, class_names_list):
        train_dataloader, _, n_cls = create_dataloader(dataset_name, input_size, "train", False)
        train_dataloaders.append(train_dataloader)
        
        if isinstance(class_names, str):
            class_names = [class_names] * n_cls

        prompt_learner = PromptLearner(
            text_tokenizer=text_tokenizer,
            num_prompt_tokens=N_PROMPT_TOKEN,
            embedding_dim=embedding_dim,
            class_names=class_names
        ).to(device)
        prompt_learners.append(prompt_learner)
        all_trainable_params.extend(list(prompt_learner.parameters()))

    sup_con_loss = SupervisedSigmoidLoss().to(device)
    scaler = GradScaler(device)
    optimizer = torch.optim.Adam(
        all_trainable_params + list(sup_con_loss.parameters()), lr=3.5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, N_EPOCHS_LoRA // 2, eta_min=1e-6)

    # --- Pre-extract image features for all datasets ---
    # @torch.inference_mode()
    image_features_lists = []
    image_label_lists = []
    with torch.no_grad():
        for i, train_dataloader in enumerate(train_dataloaders):
            image_features_list = []
            image_label_list = []
            for batch in train_dataloader:
                image_tensor, label = batch[:2]
                image_tensor = image_tensor.to(device)
                label = label.to(device)
                image_features = lora_model.get_image_features(image_tensor)
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
                modified_text_embeddings = prompt_learners[i](lora_model.text_model)
                text_features = modified_text_embeddings[label_batch]
                
                loss = sup_con_loss(text_features, image_features_batch, label_batch) + \
                        sup_con_loss(image_features_batch, text_features, label_batch)
                total_loss += loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_by_epoch += total_loss.item()
        scheduler.step()

        print("Epoch: {}, Avg loss: {}".format(epoch, loss_by_epoch / num_batches))
    
    base_model.text_model = lora_model.text_model.merge_and_unload()

    return base_model.eval(), prompt_learners

def test(model,
         dataset_name,
         device):
    validation_dataloader, num_query, _ = create_dataloader(dataset_name, INPUT_SIZE, "val", False)
    evaluator = R1_mAP_eval_pt(num_query, 10)
    for batch in validation_dataloader:
        with torch.no_grad():
            img, label, cam = batch[:3]
            img = img.to(device)
            label = label.to(device)
            cam = cam.to(device)
            test_feat = model.get_image_features(img)
            evaluator.update((test_feat, label, cam))
    cmc, mAP = evaluator.compute()[:2]
    return cmc[0], cmc[4], cmc[9], mAP

if __name__ == "__main__":
    dataset_name = "Market1501"
    # model = LoRA_tuning(dataset_name, INPUT_SIZE, "person", DEVICE)
    model = LoRA_tuning_variable_dataset(["Market1501", "veri"], [(256, 128), (224, 224)], ["person", "vehicle"], DEVICE)[0]
    cmc1, cmc5, cmc10, mAP = test(model, dataset_name, DEVICE)
    print("Dataset: {}, cmc 1: {}, cmc 5: {}, cmc 10: {}, mAP: {}".format(dataset_name, cmc1, cmc5, cmc10, mAP))
