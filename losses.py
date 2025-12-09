import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = nn.Parameter(torch.log(torch.tensor(0.07, device=device)))

    def forward(self, text_features, image_features, t_label, i_targets):
        temperature = self.temperature.exp().clamp(0.01, 10.0)

        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device)

        logits = torch.div(torch.matmul(text_features, image_features.T), temperature)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size).view(-1, 1).to(self.device), 
            0
        )
        mask = mask * logits_mask
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss

class MoCoInfoNCELoss(nn.Module):
    def __init__(self, 
                 feature_dim: int, 
                 queue_size: int = 1024,
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

def maxsim_similarity(image_tokens, text_tokens):

    # (B, N_txt, D) → (B, 1, N_txt, D)
    # (B, N_img, D) → (1, B, N_img, D)
    t = text_tokens.unsqueeze(1)     # (B, 1, N_txt, D)
    i = image_tokens.unsqueeze(0)    # (1, B, N_img, D)

    # similarity between all text tokens and all image tokens
    sim = torch.einsum("b1 t d, b2 i d -> b1 b2 t i", t, i)  # (B, B, N_txt, N_img)

    # max over image tokens
    sim_max = sim.max(dim=-1).values      # (B, B, N_txt)

    # average over text tokens
    sim_score = sim_max.mean(dim=-1)      # (B, B)

    return sim_score

class MaxSimInfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_tokens, text_tokens):
        # (B, B) similarity matrix
        sim = maxsim_similarity(image_tokens, text_tokens)

        logits = sim / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)

        # symmetric InfoNCE (optional, but recommended)
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)

        return (loss_i2t + loss_t2i) * 0.5
    

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