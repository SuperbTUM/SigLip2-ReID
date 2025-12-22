import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0, device=device)))

    def forward(self, text_features, image_features, t_label, i_targets, same_modality=False):
        temperature = self.temperature.exp().clamp(0.01, 10.0)

        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device)

        logits = torch.div(torch.matmul(text_features, image_features.T), temperature)
        if same_modality:
            logits_mask = torch.scatter(
                torch.ones_like(mask), 
                1, 
                torch.arange(batch_size).view(-1, 1).to(self.device), 
                0
            )
            mask = mask * logits_mask
        else:
            logits_mask = torch.ones_like(mask)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss

class HardTextQueueLoss(nn.Module):
    def __init__(self, 
                 feature_dim: int, 
                 queue_size: int = 32,
                 temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.queue_size = queue_size

        # --- Only Text Queue (for hard negatives) ---
        self.register_buffer("text_queue", torch.zeros(queue_size, feature_dim))
        self.register_buffer("queue_pids", torch.ones(queue_size, dtype=torch.long) * -1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_hard_text(self, hard_text_features):
        """Manually inject hard text negatives."""
        hard_text_features = F.normalize(hard_text_features, p=2, dim=1)
        batch_size = hard_text_features.size(0)
        ptr = int(self.queue_ptr)
        
        num_to_copy = min(batch_size, self.queue_size - ptr)
        self.text_queue[ptr:ptr+num_to_copy, :] = hard_text_features[:num_to_copy]
        self.queue_pids[ptr:ptr+num_to_copy] = -1
            
        self.queue_ptr[0] = (ptr + num_to_copy) % self.queue_size

    def forward(self, image_features, text_features, pid_labels):
        # 1. L2 Normalize
        T = self.temperature.exp().clamp(0.01, 10.0)

        # 2. In-batch similarity (Square matrix [B, B])
        logits_i2t = (image_features @ text_features.t()) * T
        logits_t2i = logits_i2t.t()
        
        # Labels for diagonal
        labels = torch.arange(image_features.size(0), device=image_features.device)
        
        # Mask for samples with same PID within the batch
        in_batch_mask = pid_labels.unsqueeze(1) == pid_labels.unsqueeze(0)
        in_batch_mask.fill_diagonal_(False)
        logits_i2t = logits_i2t.masked_fill(in_batch_mask, torch.finfo(logits_i2t.dtype).min)
        logits_t2i = logits_t2i.masked_fill(in_batch_mask.t(), torch.finfo(logits_i2t.dtype).min)

        # --- Image-to-Text (I2T) with Hard Queue ---
        if self.text_queue[0].any(): 
            logits_i2t_queue = (image_features @ self.text_queue.t()) * T
            
            # Original PID mask logic
            queue_mask = pid_labels.unsqueeze(1) == self.queue_pids.unsqueeze(0)
            logits_i2t_queue = logits_i2t_queue.masked_fill(queue_mask, torch.finfo(logits_i2t.dtype).min)
            
            # Concatenate for I2T
            logits_i2t = torch.cat([logits_i2t, logits_i2t_queue], dim=1)
        
        loss_i2t = F.cross_entropy(logits_i2t, labels)

        # --- Text-to-Image (T2I) ---
        # Note: We only have batch images, so this is standard InfoNCE 
        # across the transposed batch matrix.
        
        loss_t2i = F.cross_entropy(logits_t2i, labels)

        # 3. Final Symmetric Loss
        return (loss_i2t + loss_t2i) / 2

# def maxsim_similarity(image_tokens, text_tokens, temperature=1.0):
#     # L2-normalized embeddings
#     image_tokens = F.normalize(image_tokens, dim=-1)   # (B, I, D)
#     text_tokens  = F.normalize(text_tokens,  dim=-1)   # (B, T, D)

#     # 1. Compute similarity
#     sim = torch.einsum("b t d, b i d -> b t i", text_tokens, image_tokens)  # (B, T, I)

#     attn_weights = F.softmax(sim / temperature, dim=-1) # (B, T, I)
    
#     # 4. Compute weighted score
#     # This allows gradients to flow to ALL image tokens, proportional to their relevance.
#     # Even background tokens get a tiny gradient signal, preventing "dead neurons".
#     sim_smooth_max = (attn_weights * sim).sum(dim=-1) # (B, T)

#     # 5. Aggregate
#     score = sim_smooth_max.mean(dim=-1) # (B,)
    
#     return score

# class MaxSimInfoNCE(nn.Module):
#     def __init__(self, temperature=1.0):
#         super().__init__()
#         self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

#     def forward(self, image_tokens, text_tokens, labels=None):
#         # (B, B) similarity matrix
#         sim = maxsim_similarity(image_tokens, text_tokens)

#         temperature = self.temperature.exp().clamp(0.01, 10.0)
#         logits = sim * temperature

#         if labels is None:
#             targets = torch.arange(logits.size(0), device=logits.device)
#             loss_i2t = F.cross_entropy(logits, targets)
#             loss_t2i = F.cross_entropy(logits.t(), targets)
#         else:
#             labels = labels.view(-1, 1)  
#             pos_mask = (labels == labels.t()).float() 
#             pos_per_sample = pos_mask.sum(dim=1, keepdim=True)   # (B, 1)
#             pos_weights = pos_mask / pos_per_sample.clamp(min=1)

#             # log-softmax over candidates
#             log_probs = F.log_softmax(logits, dim=-1)            # (B, B)

#             # Multi-positive NCE: sum over positive targets
#             loss_i2t = -(pos_weights * log_probs).sum(dim=1).mean()

#             # Symmetric (text → image)
#             log_probs_t = F.log_softmax(logits.t(), dim=-1)
#             loss_t2i = -(pos_weights * log_probs_t).sum(dim=1).mean()

#         return 0.5 * (loss_i2t + loss_t2i)
    

def mine_hard_triplets(features, labels, base_margin=0.3):
    """
    Hard triplet mining (vectorized) with base margin.
    Only uses hardest positive and hardest negative per anchor.
    
    Args:
        features: [B, D] tensor of embeddings.
        labels: [B] tensor of class indices.
        base_margin: float, base margin value.
        
    Returns:
        Scalar loss.
    """
    # Pairwise distance matrix
    dist = torch.cdist(features, features, p=2)  # (B,B)

    # Masks
    labels = labels.unsqueeze(1)
    mask_positive = labels.eq(labels.T)
    mask_negative = ~mask_positive
    mask_positive.fill_diagonal_(False)  # remove self

    # Hardest positive: max distance among positives
    hardest_pos_dist = torch.where(mask_positive, dist, torch.tensor(float('-inf'), device=features.device))
    hardest_pos_dist, _ = hardest_pos_dist.max(dim=1)

    # Hardest negative: min distance among negatives
    hardest_neg_dist = torch.where(mask_negative, dist, torch.tensor(float('inf'), device=features.device))
    hardest_neg_dist, _ = hardest_neg_dist.min(dim=1)

    # Triplet loss
    loss = F.relu(hardest_pos_dist - hardest_neg_dist + base_margin)
    return loss.mean()

def collect_trainable_lora_As(model):
    lora_As = []

    for module in model.modules():
        if not hasattr(module, "lora_A"):
            continue

        for lora_A in module.lora_A.values():
            if lora_A.weight.requires_grad:
                lora_As.append(lora_A.weight)

    return lora_As

def lora_orthogonality_loss(lora_As):
    loss = 0.0

    for A in lora_As:
        A = A.float()     # AMP-safe
        r = A.size(0)

        # (A Aᵀ − I)
        gram = A @ A.T
        I = torch.eye(r, device=A.device)
        loss = loss + torch.norm(gram - I, p="fro") ** 2

    return loss / len(lora_As)