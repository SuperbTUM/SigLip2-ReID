import model
from constants import *
from data_preparation import *
from checkpoint import *
from evaluation import R1_mAP_eval_pt
from locked_image_tuning import LoRA_tuning_variable_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
from typing import List, Optional
import itertools
from functools import partial

from peft import get_peft_model, AdaLoraConfig, LoraConfig


torch.backends.cuda.matmul.allow_tf32 = True  # on Ampere+
torch.backends.cudnn.benchmark = True

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MoCoInfoNCELoss(nn.Module):
    def __init__(self, 
                 feature_dim: int, 
                 queue_size: int = 1024,
                 momentum: float = 0.999,
                 temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.queue_size = queue_size
        self.momentum = momentum

        # --- Memory queues for negatives ---
        self.register_buffer("image_queue", torch.randn(queue_size, feature_dim))
        self.register_buffer("text_queue",  torch.randn(queue_size, feature_dim))
        self.image_queue = F.normalize(self.image_queue, dim=1)
        self.text_queue  = F.normalize(self.text_queue,  dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # --- Momentum encoders (initialized later externally) ---
        self.image_encoder_m = None
        self.text_encoder_m = None

    @torch.no_grad()
    def _momentum_update(self, model, model_m):
        """Update momentum encoder parameters."""
        for param_q, param_k in zip(model.parameters(), model_m.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_img, keys_txt):
        """Update the queue with new keys."""
        batch_size = keys_img.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        self.image_queue[ptr:ptr + batch_size, :] = keys_img
        self.text_queue[ptr:ptr + batch_size, :] = keys_txt
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, image_features, text_features, 
                image_encoder=None, text_encoder=None):
        """
        image_features, text_features: outputs from *main* encoders
        image_encoder, text_encoder: optional main encoders for momentum update
        """
        T = self.temperature.exp().clamp(0.1, 10.0)

        # ----- Main (in-batch) contrastive logits -----
        logits_img2txt = image_features @ text_features.t() * T  # [B, B]
        labels = torch.arange(logits_img2txt.size(0), device=logits_img2txt.device)

        # ----- Extra negatives from the queue -----
        with torch.no_grad():
            queue_img = self.image_queue.clone().detach()
            queue_txt = self.text_queue.clone().detach()

        # Each image contrasts also with text negatives from queue
        logits_img_queue = image_features @ queue_txt.t() * T
        logits_txt_queue = text_features @ queue_img.t() * T

        # Concatenate in-batch + queue negatives
        logits_i2t = torch.cat([logits_img2txt, logits_img_queue], dim=1)
        logits_t2i = torch.cat([logits_img2txt.t(), logits_txt_queue], dim=1)

        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
        loss = 0.5 * (loss_i2t + loss_t2i)

        # ----- Momentum update + enqueue -----
        if (image_encoder is not None) and (text_encoder is not None):
            if self.image_encoder_m is None:
                # Initialize momentum encoders on first call
                import copy
                self.image_encoder_m = copy.deepcopy(image_encoder)
                self.text_encoder_m = copy.deepcopy(text_encoder)
                for p in self.image_encoder_m.parameters():
                    p.requires_grad = False
                for p in self.text_encoder_m.parameters():
                    p.requires_grad = False

            with torch.no_grad():
                self._momentum_update(image_encoder, self.image_encoder_m)
                self._momentum_update(text_encoder,  self.text_encoder_m)
                img_keys = self.image_encoder_m().detach()
                txt_keys = self.text_encoder_m().detach()
                self._dequeue_and_enqueue(img_keys, txt_keys)

        return loss


class GeMPooling(nn.Module):
    def __init__(self, p=1.0, eps=1e-6, trainable=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if trainable else p
        self.eps = eps

    def forward(self, x):
        # x: [B, N, D]
        p = self.p.clamp(1.0, 6.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=1).pow(1.0 / p)
        return x

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
        temperature,
        dataset_names,
        input_sizes,
        device,
        adalora=False
):
    if os.path.exists(f"checkpoint_epoch_{N_EPOCHS_LoRA}.pth"):
        base_model, prompt_learners, temperature, _, _ = load_checkpoint(base_model, prompt_learners, None, None, f"checkpoint_epoch_{N_EPOCHS_LoRA}.pth", device)
    # --- 1. Dataloaders and Classifiers for each dataset ---
    train_dataloaders = []
    classifiers = []
    classifier_params = [] # <-- Store classifier params here first
    embedding_dim = base_model.config.vision_config.hidden_size

    for dataset_name, input_size in zip(dataset_names, input_sizes):
        train_dataloader, _, n_cls, _ = create_dataloader(dataset_name, input_size, "train", True, dual_branch=True)
        train_dataloaders.append(train_dataloader)

        # Create classifier and move to device
        # Classifier: BatchNorm + FC
        classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, n_cls, bias=False),
            
        ).to(device)
        classifier[0].bias.requires_grad_(False)
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
    base_model.vision_model.vision_model.embeddings.domain_embedding.weight.data.normal_(mean=0.0, std=0.02)
    base_model.vision_model = base_model.vision_model.train()
    if adalora:
        lora_config = AdaLoraConfig(
            target_r=16,
            init_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=num_batches * 5,
            tfinal=num_batches * 45,
            deltaT=num_batches * 2,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            total_step=num_batches * N_EPOCHS_VISION,
            lora_dropout=0.05,
            init_lora_weights="eva"
        )
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "out_proj", "k_proj"], # Experiment
            lora_dropout=0.1,
            use_dora=True,
            init_lora_weights="eva",
            modules_to_save=["domain_embedding"]
        )
    # This creates the *new* trainable LoRA parameters
    vision_model = get_peft_model(base_model.vision_model, lora_config)
    vision_model.print_trainable_parameters()
    gem_pooling = GeMPooling().to(device)
    vision_model = vision_model.to(device)

    # --- 4. NOW Create the Optimizer ---
    # vision_model.parameters() will *only* return the trainable LoRA parameters
    sup_con_loss = MoCoInfoNCELoss(embedding_dim, temperature=temperature).to(device)
    scaler = GradScaler(device)
    optimizer = torch.optim.Adam(
        [
            {'params': list(vision_model.parameters()) + list(sup_con_loss.parameters()), 'lr': 5e-3, 'weight_decay': 1e-4},           # Group 1: LoRA params
            {'params': classifier_params + list(gem_pooling.parameters()), 'lr': 5e-3, 'weight_decay': 1e-4}    # Group 2: Classifier params
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
    criterion = [nn.CrossEntropyLoss(label_smoothing=0.05).to(device), nn.CrossEntropyLoss(label_smoothing=0.1).to(device)]

    # --- Freeze prompt learners ---
    modified_text_embeddings = []
    modified_text_hidden_states = []
    for prompt_learner in prompt_learners:
        for param in prompt_learner.parameters():
            param.requires_grad = False
        prompt_learner.eval()
        with torch.no_grad():
            modified_text_embedding, modified_text_hidden_state = prompt_learner(text_model)
            modified_text_embeddings.append(modified_text_embedding)
            modified_text_hidden_states.append(modified_text_hidden_state)

    # --- Training Loop (with Gradient Accumulation) ---
    accumulation_steps = 2  # Adjust as needed

    def fwd(x, y=None):
        image_features, last_hidden_state = vision_model(pixel_values=x, interpolate_pos_encoding=False, domain_id=y)
        last_hidden_state = gem_pooling(last_hidden_state)
        return image_features, last_hidden_state
    
    for epoch in range(N_EPOCHS_VISION):
        loss_by_epoch = 0
        optimizer.zero_grad()

        dataloader_iters = [itertools.cycle(loader) for loader in train_dataloaders]
        round_robin_iter = itertools.cycle(enumerate(dataloader_iters))

        for batch_idx in range(num_batches):
            total_loss = 0
            i, dataloader_iter = next(round_robin_iter)

            image_tensor, label, _, _, _, image_tensor_orig = next(dataloader_iter)
            image_tensor = image_tensor.to(device)
            image_tensor_orig = image_tensor_orig.to(device)
            label = label.to(device)

            with autocast(device, torch.float16):
                image_features, last_hidden_state = checkpoint(fwd, image_tensor, torch.zeros_like(label) if (i%1) else torch.ones_like(label), use_reentrant=False)
                
                # Cross-entropy loss
                logits = classifiers[i](image_features)
                loss_ce = criterion[i](logits, label)

                image_features_orig, last_hidden_state_orig = checkpoint(fwd, image_tensor_orig, torch.zeros_like(label) if (i%1) else torch.ones_like(label), use_reentrant=False)
                
                # Sigmoid loss
                with torch.no_grad():
                    text_features = modified_text_embeddings[i][label]
                    text_hidden_state = modified_text_hidden_states[i][label]
                
                loss_sigmoid = sup_con_loss(image_features_orig, text_features) + \
                               0.25 * sup_con_loss(last_hidden_state_orig, text_hidden_state)

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
    save_checkpoint(base_model, prompt_learners, sup_con_loss.temperature.exp().item(), N_EPOCHS_VISION, optimizer, scheduler)
    return base_model.eval()

def test(model,
         dataset_name,
         input_size,
         device):
    validation_dataloader, num_query, _, _ = create_dataloader(dataset_name, input_size, "val", False)
    evaluator = R1_mAP_eval_pt(num_query, 10)
    with torch.inference_mode():
        for batch in validation_dataloader:
            img, label, cam = batch[:3]
            img = img.to(device)
            label = label.to(device)
            cam = cam.to(device)
            test_feat = model.vision_model(pixel_values=img, interpolate_pos_encoding=False, domain_id=torch.zeros_like(label) if dataset_name == "Market1501" else torch.ones_like(label))[0]
            evaluator.update((test_feat, label, cam))
    cmc, mAP = evaluator.compute()[:2]
    evaluator.reset()
    return cmc[0], cmc[4], cmc[9], mAP

if __name__ == "__main__":
    dataset_names = ["Market1501", "veri"]
    input_sizes = [(256, 128), (224, 224)]
    class_names_list = ["person", "vehicle"]
    
    # Get the pre-tuned base model and prompt learners from locked image tuning
    base_model, prompt_learners, temperature = LoRA_tuning_variable_dataset(dataset_names, input_sizes, class_names_list, DEVICE)

    # Run LoRA vision tuning
    model = LoRA_vision_tuning(base_model, prompt_learners, temperature, dataset_names, input_sizes, DEVICE)
    
    for i, dataset_name in enumerate(dataset_names):
        cmc1, cmc5, cmc10, mAP = test(model, dataset_name, input_sizes[i], DEVICE)
        print(f"Dataset: {dataset_name}, cmc 1: {cmc1}, cmc 5: {cmc5}, cmc 10: {cmc10}, mAP: {mAP}")
        torch.cuda.empty_cache()

