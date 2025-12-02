import model
from constants import *
from data_preparation import *
from checkpoint import *
from evaluation import R1_mAP_eval_pt
from locked_image_tuning import tuning_vision_projection, LoRA_tuning_variable_dataset

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
        nn.init.normal_(classifier[1].weight, std=0.001)
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
            init_lora_weights="eva"
        )
    # This creates the *new* trainable LoRA parameters
    vision_model = get_peft_model(base_model.vision_model, lora_config)
    vision_model.print_trainable_parameters()
    vision_model = vision_model.to(device)

    # --- 4. NOW Create the Optimizer ---
    # vision_model.parameters() will *only* return the trainable LoRA parameters
    sup_con_loss = MoCoInfoNCELoss(embedding_dim, temperature=temperature).to(device)
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
    criterion = [nn.CrossEntropyLoss(label_smoothing=0.05).to(device), nn.CrossEntropyLoss(label_smoothing=0.1).to(device)]

    # --- Freeze prompt learners ---
    modified_text_embeddings = []
    modified_text_hidden_states = []
    for i, prompt_learner in enumerate(prompt_learners):
        for param in prompt_learner.parameters():
            param.requires_grad = False
        prompt_learner.eval()
        with torch.no_grad():
            modified_text_embedding, modified_text_hidden_state = prompt_learner(text_model, i % 1)
            modified_text_embeddings.append(modified_text_embedding)
            modified_text_hidden_states.append(modified_text_hidden_state)

    # --- Training Loop (with Gradient Accumulation) ---
    accumulation_steps = 2  # Adjust as needed

    def fwd(x):
        image_features, last_hidden_state = vision_model(pixel_values=x, interpolate_pos_encoding=False)
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
                image_features, last_hidden_state = checkpoint(fwd, image_tensor, use_reentrant=False)
                
                # Cross-entropy loss
                logits = classifiers[i](image_features)
                loss_ce = criterion[i](logits, label)

                image_features_orig, last_hidden_state_orig = checkpoint(fwd, image_tensor_orig, use_reentrant=False)
                
                # Sigmoid loss
                with torch.no_grad():
                    text_features = modified_text_embeddings[i][label]
                    text_hidden_state = modified_text_hidden_states[i][label]
                
                loss_sigmoid = sup_con_loss(image_features_orig, text_features, label) + \
                               0.25 * sup_con_loss(last_hidden_state_orig, text_hidden_state, label)

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
            domain_ids = torch.zeros_like(label) if dataset_name == "Market1501" else torch.ones_like(label)
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
    base_model = tuning_vision_projection(dataset_names, input_sizes, class_names_list, DEVICE)
    base_model, prompt_learners, temperature = LoRA_tuning_variable_dataset(base_model, dataset_names, input_sizes, class_names_list, DEVICE)

    # Run LoRA vision tuning
    model = LoRA_vision_tuning(base_model, prompt_learners, temperature, dataset_names, input_sizes, DEVICE)
    
    for i, dataset_name in enumerate(dataset_names):
        cmc1, cmc5, cmc10, mAP = test(model, dataset_name, input_sizes[i], DEVICE)
        print(f"Dataset: {dataset_name}, cmc 1: {cmc1}, cmc 5: {cmc5}, cmc 10: {cmc10}, mAP: {mAP}")
        torch.cuda.empty_cache()

