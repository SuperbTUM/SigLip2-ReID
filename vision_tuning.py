import model
from constants import *
from data_preparation import *
from checkpoint import *
from losses import TokenMaxSimLoss, mine_hard_triplets, MMSupConAndProxyCE
from evaluation import R1_mAP_eval_pt
from locked_image_tuning import tuning_vision_projection, prompt_tuning_variable_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import itertools


torch.backends.cuda.matmul.allow_tf32 = True  # on Ampere+
torch.backends.cudnn.benchmark = True

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def vision_tuning(
        base_model,
        prompt_learners,
        classifiers,
        dataset_names,
        input_sizes,
        device
):
    if os.path.exists(f"checkpoint_epoch_{N_EPOCHS_PRESTAGE}.pth"):
        classifiers = load_checkpoint(None, None, classifiers, None, None, f"checkpoint_epoch_{N_EPOCHS_PRESTAGE}.pth", device)[2]
    if os.path.exists(f"checkpoint_epoch_{N_EPOCHS_LoRA}.pth"):
        base_model, prompt_learners, _, _, _, _ = load_checkpoint(base_model, prompt_learners, None, None, None, f"checkpoint_epoch_{N_EPOCHS_LoRA}.pth", device)
    # --- 1. Dataloaders and Classifiers for each dataset ---
    train_dataloaders = []
    classifier_params = []

    for dataset_name, input_size, classifier in zip(dataset_names, input_sizes, classifiers):
        train_dataloader, _, _, _ = create_dataloader(dataset_name, input_size, "train", True)
        train_dataloaders.append(train_dataloader)
        
        # Add its parameters to the list
        classifier_params.extend(list(classifier.parameters()))

    num_batches = max(len(loader) for loader in train_dataloaders) * len(train_dataloaders)

    # --- 2. Freeze Text Model ---
    # (Do this early, it doesn't affect the optimizer)
    text_model = base_model.text_model.to(device)
    for param in text_model.parameters():
        param.requires_grad = False
    text_model.eval()
    text_model = text_model.to(device)

    base_model.vision_model = base_model.vision_model.train()
    
    # # This creates the *new* trainable LoRA parameters
    # if hasattr(base_model.vision_model, "peft_config"):
    #     del base_model.vision_model.peft_config
    vision_model = base_model.vision_model
    params = []
    base_lr = 5e-5
    for name, param in vision_model.named_parameters():
        param.requires_grad = True
        if "bias" in name:
            params += [{'params': [param], "lr": base_lr * 2, "weight_decay": 1e-4}]
        else:
            params += [{'params': [param], "lr": base_lr, "weight_decay": 1e-4}]
    
    vision_model = vision_model.to(device)

    # --- 4. NOW Create the Optimizer ---
    # vision_model.parameters() will *only* return the trainable LoRA parameters
    scaler = GradScaler(device)
    sup_con_loss = MMSupConAndProxyCE(alpha_ce=1.0, alpha_rank=0.)
    token_max_sim_loss = TokenMaxSimLoss().to(device)
    criterion = [nn.CrossEntropyLoss(label_smoothing=0.1), nn.CrossEntropyLoss(label_smoothing=0.1)]
    params += [
        {'params': classifier_params, 'lr': base_lr * 2, 'weight_decay': 1e-4}    # Group 2: Classifier params
    ]
    temperature_params = [{'params': list(token_max_sim_loss.parameters()) + list(sup_con_loss.parameters()), 'lr': base_lr}]
    optimizer = torch.optim.Adam(params, lr=base_lr, weight_decay=1e-4)
    temperature_optimizer = torch.optim.Adam(temperature_params, lr=base_lr)

    # --- 5. Schedulers and Losses ---

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / 10, 1.0))
    multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50])

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, multistep_scheduler],
        milestones=[10]
    )

    # --- Freeze prompt learners ---
    modified_text_embeddings = []
    for i, prompt_learner in enumerate(prompt_learners):
        for param in prompt_learner.parameters():
            param.requires_grad = False
        prompt_learner.eval()
        with torch.no_grad():
            n_cls = prompt_learner.n_cls
            modified_text_embedding = prompt_learner(text_model, i % 1, torch.arange(n_cls, device=device))
            modified_text_embeddings.append(modified_text_embedding)

    # --- Training Loop (with Gradient Accumulation) ---
    accumulation_steps = 2  # Adjust as needed
    
    for epoch in range(N_EPOCHS_VISION):
        loss_by_epoch = 0
        optimizer.zero_grad()
        # if epoch <= 10:
        temperature_optimizer.zero_grad()
        # else:
        #     for param in token_max_sim_loss.parameters():
        #         param.requires_grad = False

        dataloader_iters = [itertools.cycle(loader) for loader in train_dataloaders]
        round_robin_iter = itertools.cycle(enumerate(dataloader_iters))

        for batch_idx in range(num_batches):
            total_loss = 0
            i, dataloader_iter = next(round_robin_iter)

            image_tensor, label, _, _, _ = next(dataloader_iter)
            image_tensor = image_tensor.to(device)
            label = label.to(device)

            with autocast(device, torch.float16):
                image_features, image_attention, last_hidden_state = vision_model(pixel_values=image_tensor, interpolate_pos_encoding=False, domain_ids=i)
            
            image_features = image_features.float()
            image_attention = image_attention.float()
            last_hidden_state = last_hidden_state.float()
            
            # Cross-entropy loss
            logits = classifiers[i](image_features)
            logits_preproj = classifiers[i+(len(classifiers)>>1)](image_attention)
            loss_ce = 0.25 * criterion[i](logits, label) + 0.25 * criterion[i](logits_preproj, label) + criterion[i](F.normalize(image_features) @ F.normalize(modified_text_embeddings[i]).t() / 0.07, label)

            loss_triplet = mine_hard_triplets(image_features, label) + mine_hard_triplets(image_attention, label)

            # MaxSim loss
            loss_maxsim = token_max_sim_loss(last_hidden_state, modified_text_embeddings[i], label)
            total_loss += loss_ce + loss_triplet + 0.5 * loss_maxsim

            # Normalize loss for accumulation
            total_loss = total_loss / accumulation_steps
            
            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                # if epoch <= 10:
                scaler.step(temperature_optimizer)
                scaler.update()
                optimizer.zero_grad()
                temperature_optimizer.zero_grad()
            
            loss_by_epoch += total_loss.item() * accumulation_steps

        scheduler.step()
        print(f"Avg loss at epoch {epoch} is {loss_by_epoch / num_batches}.")
        torch.cuda.empty_cache()

    # vision_model = vision_model.merge_and_unload()
    base_model.vision_model = vision_model
    save_checkpoint(base_model, prompt_learners, None, N_EPOCHS_VISION, None, optimizer, scheduler)
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
            test_feat, test_attention = model.vision_model(pixel_values=img, interpolate_pos_encoding=False, domain_ids=domain_ids)[:2]
            evaluator.update((torch.cat((test_feat, test_attention), dim=1), label, cam))
    cmc, mAP = evaluator.compute()[:2]
    evaluator.reset()
    return cmc[0], cmc[4], cmc[9], mAP

if __name__ == "__main__":
    dataset_names = ["Market1501", "veri"]
    input_sizes = [(256, 128), (224, 224)]
    class_names_list = ["person", "vehicle"]
    
    # Get the pre-tuned base model and prompt learners from locked image tuning
    base_model, classifiers = tuning_vision_projection(dataset_names, input_sizes, DEVICE)
    for i, dataset_name in enumerate(dataset_names):
        cmc1, cmc5, cmc10, mAP = test(base_model, dataset_name, input_sizes[i], DEVICE)
        print(f"Dataset: {dataset_name}, cmc 1: {cmc1}, cmc 5: {cmc5}, cmc 10: {cmc10}, mAP: {mAP}")
        torch.cuda.empty_cache()    
    base_model, prompt_learners = prompt_tuning_variable_dataset(base_model, dataset_names, input_sizes, class_names_list, DEVICE)

    # Run LoRA vision tuning
    model = vision_tuning(base_model, prompt_learners, classifiers, dataset_names, input_sizes, DEVICE)
    
    for i, dataset_name in enumerate(dataset_names):
        cmc1, cmc5, cmc10, mAP = test(model, dataset_name, input_sizes[i], DEVICE)
        print(f"Dataset: {dataset_name}, cmc 1: {cmc1}, cmc 5: {cmc5}, cmc 10: {cmc10}, mAP: {mAP}")
        torch.cuda.empty_cache()

