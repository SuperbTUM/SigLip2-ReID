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

class MMSupConAndProxyCE(nn.Module):
    """
    Two losses:
      (1) L_mm_supcon: supervised contrastive over anchors Z = [images ; class_text_proxies]
          positives = same class (image-image, image-proxy, proxy-image)
          + optional per-class near-miss negative in denominator

      (2) L_proxy_ce: image -> class proxy CE
          + optional per-class near-miss negative as an extra logit

    Inputs:
      image_features: [B, D]
      text_features:  [B, D]  (soft prompt embedding per image; duplicated per class with PK)
      pid_labels:     [B]
      near_miss_features: optional [B, D] (near-miss prompt embedding per image; duplicated per class)
    """

    def __init__(self, temperature: float = 0.07, alpha_proxy_ce: float = 0.1):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        self.alpha_proxy_ce = alpha_proxy_ce

    def _T(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.01, 1.0)  # logits = sim / T

    @staticmethod
    def _mean_by_class(features: torch.Tensor, pid_labels: torch.Tensor):
        """
        Mean-pool features per class in the batch.

        Returns:
          unique_pids: [P]
          class_feats: [P, D]
          inv:         [B] in 0..P-1
        """
        unique_pids, inv = torch.unique(pid_labels, return_inverse=True)
        P, D = unique_pids.size(0), features.size(1)

        class_feats = torch.zeros(P, D, device=features.device, dtype=features.dtype)
        class_feats.index_add_(0, inv, features)

        counts = torch.bincount(inv, minlength=P).to(features.device).clamp_min(1).unsqueeze(1)
        class_feats = class_feats / counts
        return unique_pids, class_feats, inv

    @staticmethod
    def _supcon_loss(logits: torch.Tensor, pos_mask: torch.Tensor, extra_den_logits: torch.Tensor | None = None):
        """
        logits: [N, N] anchor-to-anchor logits
        pos_mask: [N, N] True where positive (excluding self)
        extra_den_logits: [N, K] additional negatives-only logits appended to denominator
        """
        N = logits.size(0)
        self_mask = torch.eye(N, device=logits.device, dtype=torch.bool)

        # Denominator excludes self
        logits_den = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
        if extra_den_logits is not None:
            log_den = torch.logsumexp(torch.cat([logits_den, extra_den_logits], dim=1), dim=1)
        else:
            log_den = torch.logsumexp(logits_den, dim=1)

        # Numerator over positives only
        log_num = torch.logsumexp(logits.masked_fill(~pos_mask, torch.finfo(logits.dtype).min), dim=1)

        valid = pos_mask.sum(dim=1) > 0
        return -(log_num[valid] - log_den[valid]).mean()

    def loss_mm_supcon(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        pid_labels: torch.Tensor,
        near_miss_features: torch.Tensor | None = None,  # [B, D]
    ) -> torch.Tensor:
        T = self._T()

        img = F.normalize(image_features, dim=1)
        txt = F.normalize(text_features, dim=1)

        unique_pids, class_txt, inv = self._mean_by_class(txt, pid_labels)
        class_txt = F.normalize(class_txt, dim=1)

        # anchors: images + class proxies
        Z = torch.cat([img, class_txt], dim=0)  # [N, D]
        labels = torch.cat([pid_labels, unique_pids], dim=0)  # [N]
        N = Z.size(0)

        logits = (Z @ Z.t()) / T
        logits = logits - logits.max(dim=1, keepdim=True).values  # stability

        self_mask = torch.eye(N, device=Z.device, dtype=torch.bool)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)

        # Optional: add per-class near-miss negative in denominator for each anchor
        extra_den = None
        if near_miss_features is not None:
            nm = F.normalize(near_miss_features, dim=1)

            # Collapse duplicated [B, D] near-miss into per-class [P, D]
            _, class_nm, _ = self._mean_by_class(nm, pid_labels)  # aligns with unique_pids via same pid_labels
            class_nm = F.normalize(class_nm, dim=1)
            P = class_nm.size(0)

            # For each anchor, pick its class index in 0..P-1
            anchor_class_idx = torch.cat(
                [inv, torch.arange(P, device=Z.device, dtype=inv.dtype)],
                dim=0
            )  # [N]

            nm_for_anchor = class_nm[anchor_class_idx]  # [N, D]
            extra_den = (Z * nm_for_anchor).sum(dim=1, keepdim=True) / T  # [N, 1]
            extra_den = extra_den - extra_den.max(dim=1, keepdim=True).values

        return self._supcon_loss(logits, pos_mask, extra_den_logits=extra_den)

    def loss_proxy_ce(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        pid_labels: torch.Tensor,
        near_miss_features: torch.Tensor | None = None,  # [B, D]
    ) -> torch.Tensor:
        T = self._T()

        img = F.normalize(image_features, dim=1)
        txt = F.normalize(text_features, dim=1)

        unique_pids, class_txt, inv = self._mean_by_class(txt, pid_labels)
        class_txt = F.normalize(class_txt, dim=1)

        logits_proxy = (img @ class_txt.t()) / T  # [B, P]

        if near_miss_features is None:
            return F.cross_entropy(logits_proxy, inv)

        nm = F.normalize(near_miss_features, dim=1)

        # Collapse duplicated [B, D] near-miss into per-class [P, D]
        _, class_nm, _ = self._mean_by_class(nm, pid_labels)
        class_nm = F.normalize(class_nm, dim=1)

        # Each image gets its class's near-miss as one extra negative logit
        nm_for_img = class_nm[inv]  # [B, D]
        logits_nm = (img * nm_for_img).sum(dim=1, keepdim=True) / T  # [B, 1]

        logits = torch.cat([logits_proxy, logits_nm], dim=1)  # [B, P+1]
        return F.cross_entropy(logits, inv)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        pid_labels: torch.Tensor,
        near_miss_features: torch.Tensor | None = None,  # [B, D]
        return_separate: bool = True,
    ):
        l_mm = self.loss_mm_supcon(image_features, text_features, pid_labels, near_miss_features)
        l_ce = self.loss_proxy_ce(image_features, text_features, pid_labels, near_miss_features)
        l_total = l_mm + self.alpha_proxy_ce * l_ce

        if return_separate:
            return l_mm, l_ce, l_total
        return l_total

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


class TokenMaxSimLoss(nn.Module):
    def __init__(self, tau=0.07, k=4, label_smoothing=0.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.log(torch.tensor(tau)))
        self.k = k
        self.label_smoothing = label_smoothing

    def forward(self, image_tokens, text_features, pid_labels):
        """
        image_tokens: [B, N, D]
        text_features: [C, D]
        pid_labels: [B]
        """

        # Normalize
        image_tokens = F.normalize(image_tokens, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Token-level similarity: [B, N, C]
        sim = torch.einsum("bnd,cd->bnc", image_tokens, text_features) / self.temperature.exp().clamp(0.01, 10)

        # Max over image tokens → [B, C]
        logits = sim.topk(k=self.k, dim=1, largest=True, sorted=False).values.mean(dim=1)

        # Masks
        loss = F.cross_entropy(logits, pid_labels, label_smoothing=self.label_smoothing)
        return loss

def compute_centroids(features: torch.Tensor,
                      labels: torch.Tensor,
                      normalize: bool = True):
    """
    features: [B, D] tensor
    labels:   [B] long tensor
    normalize: whether to L2-normalize centroids

    returns:
        centroids: [C_batch, D]  (one per unique label in the batch)
        uniq_labels: [C_batch]   (labels corresponding to centroids)
    """
    device = features.device
    labels = labels.to(device)

    uniq_labels = labels.unique(sorted=True)
    centroids = []

    for lbl in uniq_labels:
        mask = labels == lbl
        if mask.any():
            c = features[mask].mean(dim=0)
            centroids.append(c)

    centroids = torch.stack(centroids, dim=0)

    if normalize:
        centroids = F.normalize(centroids, dim=-1)

    return centroids