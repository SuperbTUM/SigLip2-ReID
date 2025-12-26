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
        batch_size = hard_text_features.size(0)
        ptr = int(self.queue_ptr)
        
        num_to_copy = min(batch_size, self.queue_size - ptr)
        self.text_queue[ptr:ptr+num_to_copy, :] = hard_text_features[:num_to_copy]
        self.queue_pids[ptr:ptr+num_to_copy] = -1
            
        self.queue_ptr[0] = (ptr + num_to_copy) % self.queue_size

    def forward(self, image_features, text_features, pid_labels):
        T = self.temperature.exp().clamp(0.01, 10.0)

        unique_pids, inv = torch.unique(pid_labels, return_inverse=True)
        P = unique_pids.size(0)

        class_texts = torch.zeros(
            P, text_features.size(1),
            device=text_features.device,
            dtype=text_features.dtype
        )
        class_texts.index_add_(0, inv, text_features)

        # targets: class index per image
        targets = inv  # [B], values in [0, P-1]

        # In-batch similarity (Square matrix [B, B])
        logits_i2t = (image_features @ class_texts.t()) * T
        logits_t2i = (class_texts @ image_features.t()) * T
        
        pos_mask = unique_pids.unsqueeze(1) == pid_labels.unsqueeze(0)  # [P, B]
        pos_logits = logits_t2i.masked_fill(~pos_mask, torch.finfo(logits_t2i.dtype).min)

        log_num = torch.logsumexp(pos_logits, dim=1)     # [P]
        log_den = torch.logsumexp(logits_t2i, dim=1)     # [P]

        # --- Image-to-Text (I2T) with Hard Queue ---
        if self.text_queue[0].any(): 
            logits_i2t_queue = (image_features @ self.text_queue.t()) * T
            
            # Original PID mask logic
            # queue_mask = pid_labels.unsqueeze(1) == self.queue_pids.unsqueeze(0)
            # logits_i2t_queue = logits_i2t_queue.masked_fill(queue_mask, torch.finfo(logits_i2t.dtype).min)
            
            # Concatenate for I2T
            logits_i2t = torch.cat([logits_i2t, logits_i2t_queue], dim=1)
        
        loss_i2t = F.cross_entropy(logits_i2t, targets)

        # --- Text-to-Image (T2I) ---
        # Note: We only have batch images, so this is standard InfoNCE 
        # across the transposed batch matrix.
        
        loss_t2i = -(log_num - log_den).mean()

        # 3. Final Symmetric Loss
        return (loss_i2t + loss_t2i) / 2

def maxsim_img_loss(image_features, pid_labels, temperature=1.0):
    """
    image_features: [B, D] normalized
    pid_labels: [B] long tensor
    """
    
    # similarity matrix
    sim_matrix = image_features @ image_features.t()  # [B, B]

    # masks
    pid_labels = pid_labels.to(sim_matrix.device)
    pos_mask = pid_labels.unsqueeze(0) == pid_labels.unsqueeze(1)
    eye = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
    pos_mask = pos_mask ^ eye.bool()  # remove diagonal

    # log-sum-exp over positives
    pos_sim = sim_matrix.masked_fill(~pos_mask, torch.finfo(image_features.dtype).min)
    log_num = torch.logsumexp(pos_sim / temperature, dim=1)

    # log-sum-exp over all except self (denominator)
    sim_matrix_no_diag = sim_matrix.masked_fill(eye.bool(), torch.finfo(image_features.dtype).min)
    log_den = torch.logsumexp(sim_matrix_no_diag / temperature, dim=1)

    # MaxSim loss
    loss = -(log_num - log_den).mean()
    return loss


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