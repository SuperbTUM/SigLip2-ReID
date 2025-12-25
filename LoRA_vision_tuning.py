import model
from constants import *
from data_preparation import *
from checkpoint import *
from losses import HardTextQueueLoss, mine_hard_triplets
from evaluation import R1_mAP_eval_pt
from locked_image_tuning import tuning_vision_projection, LoRA_tuning_variable_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
from typing import List, Optional
import itertools
from functools import partial


torch.backends.cuda.matmul.allow_tf32 = True  # on Ampere+
torch.backends.cudnn.benchmark = True

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def LoRA_vision_tuning(
        base_model,
        prompt_learners,
        temperature,
        dataset_names,
        input_sizes,
        device
):
    if os.path.exists(f"checkpoint_epoch_120.pth"):
        base_model, prompt_learners, temperature, _, _ = load_checkpoint(base_model, prompt_learners, None, None, f"checkpoint_epoch_120.pth", device)
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
    
    # This creates the *new* trainable LoRA parameters
    if hasattr(base_model.vision_model, "peft_config"):
        del base_model.vision_model.peft_config
    vision_model = base_model.vision_model
    for name, param in vision_model.named_parameters():
        param.requires_grad = True
    vision_model = vision_model.to(device)

    # --- 4. NOW Create the Optimizer ---
    # vision_model.parameters() will *only* return the trainable LoRA parameters
    # sup_con_loss = HardTextQueueLoss(embedding_dim, temperature=temperature).to(device)
    scaler = GradScaler(device)
    optimizer = torch.optim.Adam(
        [
            {'params': list(vision_model.parameters()), 'lr': 5e-4, 'weight_decay': 1e-4},           # Group 1: LoRA params
            {'params': classifier_params, 'lr': 5e-4, 'weight_decay': 1e-4}    # Group 2: Classifier params
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

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=partial(warmup_lambda, warmup_epochs=5, start_lr=5e-5, base_lr=5e-4))
    multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50])

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, multistep_scheduler],
        milestones=[5]
    )
    criterion = [nn.CrossEntropyLoss(label_smoothing=0.05).to(device), nn.CrossEntropyLoss(label_smoothing=0.1).to(device)]

    # --- Freeze prompt learners ---
    modified_text_embeddings = []
    for i, prompt_learner in enumerate(prompt_learners):
        for param in prompt_learner.parameters():
            param.requires_grad = False
        prompt_learner.eval()
        with torch.no_grad():
            modified_text_embedding = prompt_learner(text_model, i % 1)
            modified_text_embeddings.append(modified_text_embedding)

    # --- Training Loop (with Gradient Accumulation) ---
    accumulation_steps = 2  # Adjust as needed
    
    for epoch in range(N_EPOCHS_VISION):
        loss_by_epoch = 0
        optimizer.zero_grad()

        dataloader_iters = [itertools.cycle(loader) for loader in train_dataloaders]
        round_robin_iter = itertools.cycle(enumerate(dataloader_iters))

        for batch_idx in range(num_batches):
            total_loss = 0
            i, dataloader_iter = next(round_robin_iter)

            image_tensor, label, _, _, _, _ = next(dataloader_iter)
            image_tensor = image_tensor.to(device)
            label = label.to(device)

            with autocast(device, torch.float16):
                image_features, last_hidden_state = vision_model(pixel_values=image_tensor, interpolate_pos_encoding=False)
                
                # Cross-entropy loss
                logits = classifiers[i](image_features)
                loss_ce = criterion[i](logits, label) + criterion[i](image_features @ modified_text_embeddings[i].t(), label)

                # Triplet loss
                loss_triplet = mine_hard_triplets(image_features, label, base_margin=0.3) + \
                                mine_hard_triplets(last_hidden_state, label, base_margin=0.3)
                total_loss += loss_triplet + loss_ce

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

    # vision_model = vision_model.merge_and_unload()
    base_model.vision_model = vision_model
    save_checkpoint(base_model, prompt_learners, None, N_EPOCHS_VISION, optimizer, scheduler)
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
    for i, dataset_name in enumerate(dataset_names):
        cmc1, cmc5, cmc10, mAP = test(base_model, dataset_name, input_sizes[i], DEVICE)
        print(f"Dataset: {dataset_name}, cmc 1: {cmc1}, cmc 5: {cmc5}, cmc 10: {cmc10}, mAP: {mAP}")
        torch.cuda.empty_cache()    
    base_model, prompt_learners, temperature = LoRA_tuning_variable_dataset(base_model, dataset_names, input_sizes, class_names_list, DEVICE)

    # Run LoRA vision tuning
    model = LoRA_vision_tuning(base_model, prompt_learners, temperature, dataset_names, input_sizes, DEVICE)
    
    for i, dataset_name in enumerate(dataset_names):
        cmc1, cmc5, cmc10, mAP = test(model, dataset_name, input_sizes[i], DEVICE)
        print(f"Dataset: {dataset_name}, cmc 1: {cmc1}, cmc 5: {cmc5}, cmc 10: {cmc10}, mAP: {mAP}")
        torch.cuda.empty_cache()

