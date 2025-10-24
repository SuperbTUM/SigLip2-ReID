import model
from constants import *
from data_preparation import *
from evaluation import R1_mAP_eval_pt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
from typing import List, Optional
import itertools
from functools import partial

from peft import get_peft_model, AdaLoraConfig, LoraConfig
from locked_image_tuning import LoRA_tuning_variable_dataset

torch.backends.cuda.matmul.allow_tf32 = True  # on Ampere+
torch.backends.cudnn.benchmark = True

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


def mine_hard_triplets(features, labels, base_margin=0.3, adaptive_weight=0.5, reg_weight=0.1):
    """
    Hard triplet mining with flexible, regularized adaptive margin.
    
    Args:
        features: [B, D] tensor of embeddings.
        labels: [B] tensor of class indices.
        base_margin: base margin value (float).
        adaptive_weight: how strongly the margin adapts per sample.
        reg_weight: regularization weight pulling adaptive margin back to base_margin.
    """
    dist_matrix = torch.cdist(features, features, p=2)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)

    # Hardest positive and negative distances
    hardest_pos_dist = (dist_matrix * labels_eq.float()).max(dim=1)[0]
    hardest_neg_dist = (dist_matrix + 1e5 * labels_eq.float()).min(dim=1)[0]

    # --- Adaptive margin component ---
    # Example: use distance spread (neg - pos) as confidence
    dist_diff = (hardest_neg_dist - hardest_pos_dist).detach()
    adaptive_margin = base_margin + adaptive_weight * (0.3 - torch.tanh(dist_diff))
    # Smooth clamp to avoid exploding margin
    adaptive_margin = torch.clamp(adaptive_margin, 0.05, 0.6)

    # --- Triplet loss with adaptive margin ---
    triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + adaptive_margin)
    
    # --- Regularization to keep margins near base ---
    reg_loss = reg_weight * (adaptive_margin - base_margin).abs().mean()

    return triplet_loss.mean() + reg_loss


def LoRA_vision_tuning(
        base_model,
        prompt_learners,
        dataset_names,
        input_sizes,
        device,
        adalora=False
):

    # --- 1. Dataloaders and Classifiers for each dataset ---
    train_dataloaders = []
    classifiers = []
    classifier_params = [] # <-- Store classifier params here first
    embedding_dim = base_model.config.vision_config.hidden_size

    for dataset_name, input_size in zip(dataset_names, input_sizes):
        train_dataloader, _, n_cls = create_dataloader(dataset_name, input_size, "train", True)
        train_dataloaders.append(train_dataloader)
        
        # Create classifier and move to device
        # Classifier: BatchNorm + FC
        classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, n_cls, bias=False),
            
        ).to(device)
        classifiers.append(classifier)
        
        # Add its parameters to the list
        classifier_params.extend(list(classifier.parameters()))

    num_batches = max(len(loader) for loader in train_dataloaders)

    # --- 2. Freeze Text Model ---
    # (Do this early, it doesn't affect the optimizer)
    text_model = base_model.text_model.to(device)
    for param in text_model.parameters():
        param.requires_grad = False
    text_model.eval()
    text_model = text_model.to(device)

    # --- 3. Apply PEFT to the Vision Model ---
    base_model.vision_model = base_model.vision_model.train()
    if adalora:
        lora_config = AdaLoraConfig(
            r=8,
            init_r=12,
            tinit=num_batches * 5,
            tfinal=num_batches * 50,
            deltaT=num_batches,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            total_step=num_batches * N_EPOCHS_VISION,
        )
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            use_dora=True
        )
    # This creates the *new* trainable LoRA parameters
    base_model.vision_model = base_model.vision_model.train()
    vision_model = get_peft_model(base_model.vision_model, lora_config)
    vision_model.print_trainable_parameters()
    vision_model = vision_model.to(device)

    # --- 4. NOW Create the Optimizer ---
    # vision_model.parameters() will *only* return the trainable LoRA parameters
    sup_con_loss = SupervisedSigmoidLoss().to(device)
    scaler = GradScaler(device)
    optimizer = torch.optim.Adam(
        [
            {'params': list(vision_model.parameters()) + list(sup_con_loss.parameters()), 'lr': 5e-3, 'weight_decay': 1e-4},           # Group 1: LoRA params
            {'params': classifier_params, 'lr': 5e-3, 'weight_decay': 1e-4}    # Group 2: Classifier params
        ])

    # --- 5. Schedulers and Losses ---
    def warmup_lambda(epoch, warmup_epochs, start_lr, base_lr):
        if epoch >= warmup_epochs:
            return 1.0  # after warmup, multiplier = 1
        else:
            # linear from start_lr -> base_lr
            # since optimizer.lr is already base_lr, scale relative to that
            lr_factor = start_lr / base_lr + (1 - start_lr / base_lr) * (epoch / warmup_epochs)
            return lr_factor

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=partial(warmup_lambda, warmup_epochs=5, start_lr=5e-4, base_lr=5e-3))
    multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50])

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, multistep_scheduler],
        milestones=[5]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # --- Freeze prompt learners ---
    modified_text_embeddings = []
    for prompt_learner in prompt_learners:
        for param in prompt_learner.parameters():
            param.requires_grad = False
        prompt_learner.eval()
        with torch.no_grad():
            modified_text_embeddings.append(prompt_learner(text_model))

    # --- Training Loop (with Gradient Accumulation) ---
    accumulation_steps = 2  # Adjust as needed

    def fwd(x):
        return vision_model(pixel_values=x, interpolate_pos_encoding=False)
    
    for epoch in range(N_EPOCHS_VISION):
        loss_by_epoch = 0
        optimizer.zero_grad()

        dataloader_iters = [itertools.cycle(loader) for loader in train_dataloaders]
        round_robin_iter = itertools.cycle(enumerate(dataloader_iters))

        for batch_idx in range(num_batches):
            total_loss = 0
            i, dataloader_iter = next(round_robin_iter)

            image_tensor, label = next(dataloader_iter)[:2]
            image_tensor = image_tensor.to(device)
            label = label.to(device)

            with autocast(device, torch.float16):
                image_features, last_hidden_state = checkpoint(fwd, image_tensor, use_reentrant=False)
                
                # Cross-entropy loss
                logits = classifiers[i](image_features)
                loss_ce = criterion(logits, label)
                
                # Sigmoid loss
                with torch.no_grad():
                    text_features = modified_text_embeddings[i][label]
                
                loss_sigmoid = sup_con_loss(image_features, text_features, label) + \
                               sup_con_loss(text_features, image_features, label)

                # Triplet loss
                loss_triplet = mine_hard_triplets(image_features, label, base_margin=0.3) + \
                                mine_hard_triplets(last_hidden_state, label, base_margin=0.3)
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
    with torch.inference_mode():
        for batch in validation_dataloader:
            img, label, cam = batch[:3]
            img = img.to(device)
            label = label.to(device)
            cam = cam.to(device)
            test_feat = model.vision_model(pixel_values=img, interpolate_pos_encoding=False)[0]
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

