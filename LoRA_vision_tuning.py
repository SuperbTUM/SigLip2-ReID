import model
from constants import *
from data_preparation import *
from evaluation import R1_mAP_eval_pt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import List, Optional
import itertools

from peft import get_peft_model, LoraConfig
from locked_image_tuning import LoRA_tuning_variable_dataset

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        self.temperature = nn.Parameter(torch.tensor(temperature))
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

        logits = torch.matmul(image_features, text_features.t()) * self.temperature

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


def mine_hard_triplets(features, labels, margin=0.3):
    device = features.device
    loss = torch.tensor(0.0, device=device)
    num_triplets = 0

    for i in range(len(labels)):
        anchor_feature = features[i]
        anchor_label = labels[i]

        # Find hardest positive
        is_pos = (labels == anchor_label) & (torch.arange(len(labels), device=device) != i)
        if not torch.any(is_pos):
            continue
        
        pos_features = features[is_pos]
        pos_dists = torch.cdist(anchor_feature.unsqueeze(0), pos_features)
        hardest_pos_dist, hardest_pos_idx = torch.max(pos_dists, dim=1)

        # Find hardest negative
        is_neg = labels != anchor_label
        if not torch.any(is_neg):
            continue

        neg_features = features[is_neg]
        neg_dists = torch.cdist(anchor_feature.unsqueeze(0), neg_features)
        hardest_neg_dist, hardest_neg_idx = torch.min(neg_dists, dim=1)
        
        # Calculate triplet loss for this anchor
        triplet_loss = F.relu(hardest_pos_dist[0] - hardest_neg_dist[0] + margin)
        loss += triplet_loss
        num_triplets += 1
        
    
    if num_triplets == 0:
        return torch.tensor(0.0, device=device)

    return loss / num_triplets

def LoRA_vision_tuning(base_model,
                                 prompt_learners,
                                 dataset_names,
                                 input_sizes,
                                 device):

    # PEFT LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    vision_model = get_peft_model(base_model.vision_model, lora_config)
    vision_model.print_trainable_parameters()
    
    vision_model = vision_model.to(device).train()
    text_model = base_model.text_model.to(device)
    embedding_dim = base_model.config.vision_config.hidden_size

    # Freeze text model
    for param in text_model.parameters():
        param.requires_grad = False
    text_model.eval()

    # --- Dataloaders and Classifiers for each dataset ---
    train_dataloaders = []
    classifiers = []
    all_trainable_params = list(vision_model.parameters())

    for dataset_name, input_size in zip(dataset_names, input_sizes):
        train_dataloader, _, n_cls = create_dataloader(dataset_name, input_size, "train", True)
        train_dataloaders.append(train_dataloader)
        
        classifier = nn.Linear(embedding_dim, n_cls, bias=False, device=device)
        classifiers.append(classifier)
        all_trainable_params.extend(list(classifier.parameters()))

    optimizer = torch.optim.Adam(all_trainable_params, lr=5e-6, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    sup_con_loss = SupervisedSigmoidLoss().to(device)
    scaler = GradScaler(device)

    # --- Freeze prompt learners ---
    modified_text_embeddings = []
    for prompt_learner in prompt_learners:
        for param in prompt_learner.parameters():
            param.requires_grad = False
        prompt_learner.eval()
        with torch.no_grad():
            modified_text_embeddings.append(prompt_learner(text_model))

    # --- Training Loop (with Gradient Accumulation) ---
    accumulation_steps = 4  # Adjust as needed
    num_batches = max(len(loader) for loader in train_dataloaders)
    dataloader_iters = [itertools.cycle(loader) for loader in train_dataloaders]
    round_robin_iter = itertools.cycle(enumerate(dataloader_iters))
    
    for epoch in range(N_EPOCHS_VISION):
        loss_by_epoch = 0
        optimizer.zero_grad()

        for batch_idx in range(num_batches):
            total_loss = 0
            i, dataloader_iter = next(round_robin_iter)

            image_tensor, label = next(dataloader_iter)[:2]
            image_tensor = image_tensor.to(device)
            label = label.to(device)

            with autocast(device):
                image_features = vision_model(pixel_values=image_tensor, interpolate_pos_encoding=False)
                
                # Cross-entropy loss
                logits = classifiers[i](image_features)
                loss_ce = criterion(logits, label)
                
                # Sigmoid loss
                with torch.no_grad():
                    text_features = modified_text_embeddings[i][label]
                
                loss_sigmoid = sup_con_loss(image_features, text_features)

                # Triplet loss
                loss_triplet = mine_hard_triplets(image_features, label, margin=0.3)
                total_loss += loss_triplet + loss_ce + loss_sigmoid

                # Normalize loss for accumulation
                total_loss = total_loss / accumulation_steps
            
            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            loss_by_epoch += total_loss.item() * accumulation_steps

        scheduler.step()
        print(f"Avg loss at epoch {epoch} is {loss_by_epoch / num_batches}.")
        torch.cuda.empty_cache()

    vision_model = vision_model.merge_and_unload()
    base_model.vision_model = vision_model
    return base_model.eval()

def test(model,
         dataset_name,
         input_size,
         device):
    validation_dataloader, num_query, _ = create_dataloader(dataset_name, input_size, "val", False)
    evaluator = R1_mAP_eval_pt(num_query, 10)
    for batch in validation_dataloader:
        with torch.no_grad():
            img, label, cam = batch[:3]
            img = img.to(device)
            label = label.to(device)
            cam = cam.to(device)
            test_feat = model.vision_model(pixel_values=img, interpolate_pos_encoding=False)
            test_feat = test_feat / test_feat.norm(p=2, dim=-1, keepdim=True)
            evaluator.update((test_feat, label, cam))
    cmc, mAP = evaluator.compute()[:2]
    evaluator.reset()
    return cmc[0], cmc[4], cmc[9], mAP

if __name__ == "__main__":
    dataset_names = ["Market1501", "veri"]
    input_sizes = [(256, 128), (224, 224)]
    class_names_list = ["person", "vehicle"]
    
    # Get the pre-tuned base model and prompt learners from locked image tuning
    base_model, prompt_learners = LoRA_tuning_variable_dataset(dataset_names, input_sizes, class_names_list, DEVICE)

    # Run LoRA vision tuning
    model = LoRA_vision_tuning(base_model, prompt_learners, dataset_names, input_sizes, DEVICE)
    
    for i, dataset_name in enumerate(dataset_names):
        cmc1, cmc5, cmc10, mAP = test(model, dataset_name, input_sizes[i], DEVICE)
        print(f"Dataset: {dataset_name}, cmc 1: {cmc1}, cmc 5: {cmc5}, cmc 10: {cmc10}, mAP: {mAP}")
        torch.cuda.empty_cache()

