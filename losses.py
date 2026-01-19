import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = nn.Parameter(torch.log(torch.tensor(0.07, device=device)))

    def forward(self, text_features, image_features, t_label, i_targets, same_modality=False):
        text_features = F.normalize(text_features)
        image_features = F.normalize(image_features)
        
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
          anchors = class text proxies (P)
          contrasts = images (B)
          positives per proxy = K images of that class

      (2) L_proxy_ce: image -> class proxy CE (batch-proxy CE over P classes)

      Add near-miss as a negative with a *pairwise margin/ranking* term:
          L_rank = mean ReLU(margin + sim(img, nm_y) - sim(img, soft_y))

      This is a clean signal when near_miss is a GPT distractor for the same class:
          enforce: sim(img, soft_prompt_y) > sim(img, near_miss_y) + margin
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alpha_ce: float = 0.1,
        alpha_rank: float = 0.1,
        rank_margin: float = 0.1,
        min_logT: float = -4.605170186,  # log(0.01)
        max_logT: float = 0.0,           # log(1.0)
    ):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        self.alpha_ce = float(alpha_ce)
        self.alpha_rank = float(alpha_rank)
        self.rank_margin = float(rank_margin)
        self.min_logT = float(min_logT)
        self.max_logT = float(max_logT)

    def _T(self) -> torch.Tensor:
        # Clamp in log-space so gradients don't die at the clamp boundaries
        logT = self.log_temperature.clamp(self.min_logT, self.max_logT)
        return logT.exp()  # logits = sim / T

    @staticmethod
    def _mean_by_class(features: torch.Tensor, pid_labels: torch.Tensor):
        unique_pids, inv = torch.unique(pid_labels, return_inverse=True)
        P, D = unique_pids.size(0), features.size(1)
        out = torch.zeros(P, D, device=features.device, dtype=features.dtype)
        out.index_add_(0, inv, features)
        counts = torch.bincount(inv, minlength=P).to(features.device).clamp_min(1).unsqueeze(1)
        out = out / counts
        return unique_pids, out, inv  # inv: [B] in 0..P-1

    @staticmethod
    def _supcon_from_logits(logits: torch.Tensor, pos_mask: torch.Tensor, mask_out: torch.Tensor | None = None):
        """
        logits:   [A, N]
        pos_mask: [A, N] boolean positives
        mask_out: [A, N] boolean where True means "exclude from denominator" (e.g., self)
        """
        dtype = logits.dtype
        NEG = torch.finfo(dtype).min

        if mask_out is None:
            logits_den = logits
        else:
            logits_den = logits.masked_fill(mask_out, NEG)

        # stability
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits_den = logits_den - logits_den.max(dim=1, keepdim=True).values.detach()

        log_den = torch.logsumexp(logits_den, dim=1)  # [A]
        log_num = torch.logsumexp(logits.masked_fill(~pos_mask, NEG), dim=1)  # [A]

        valid = pos_mask.sum(dim=1) > 0
        return -(log_num[valid] - log_den[valid]).mean()

    def loss_mm_proxy_to_image(self, image_features, text_features, pid_labels):
        """
        Cross-modal supervised contrastive:
          anchors = class text proxies (P)
          contrasts = images (B)
          positives per proxy = K images of that class
        """
        T = self._T().float()
        img = F.normalize(image_features.float(), dim=1, eps=1e-6)   # [B,D]
        txt = F.normalize(text_features.float(), dim=1, eps=1e-6)    # [B,D]

        unique_pids, class_txt, _ = self._mean_by_class(txt, pid_labels)  # [P,D]
        class_txt = F.normalize(class_txt, dim=1, eps=1e-6)

        logits = (class_txt @ img.t()) / T  # [P, B]
        pos_mask = (unique_pids.view(-1, 1) == pid_labels.view(1, -1))  # [P,B]
        return self._supcon_from_logits(logits, pos_mask)

    def loss_ce_image_to_proxy(self, image_features, text_features, pid_labels):
        """
        Batch-proxy CE: image -> class proxy among P batch classes.
        """
        T = self._T().float()
        img = F.normalize(image_features.float(), dim=1, eps=1e-6)
        txt = F.normalize(text_features.float(), dim=1, eps=1e-6)

        _, class_txt, inv = self._mean_by_class(txt, pid_labels)   # [P,D], inv:[B]
        class_txt = F.normalize(class_txt, dim=1, eps=1e-6)

        logits_proxy = (img @ class_txt.t()) / T  # [B,P]
        return F.cross_entropy(logits_proxy, inv)

    def loss_rank_soft_vs_nearmiss(self, image_features, text_features, pid_labels, near_miss_features):
        """
        Pairwise margin/ranking loss enforcing:
            sim(img, soft_proxy_y) >= sim(img, near_miss_y) + margin

        near_miss_features: [B,D] duplicated per class with PK; collapsed to [P,D].
        """
        T = self._T().float()
        img = F.normalize(image_features.float(), dim=1, eps=1e-6)
        txt = F.normalize(text_features.float(), dim=1, eps=1e-6)
        nm = F.normalize(near_miss_features.float(), dim=1, eps=1e-6)

        _, class_txt, inv = self._mean_by_class(txt, pid_labels)   # [P,D], inv:[B]
        class_txt = F.normalize(class_txt, dim=1, eps=1e-6)

        _, class_nm, _ = self._mean_by_class(nm, pid_labels)       # [P,D]
        class_nm = F.normalize(class_nm, dim=1, eps=1e-6)

        # per-sample sims to its class proxies
        s_soft = (img * class_txt[inv]).sum(dim=1) / T  # [B]
        s_nm = (img * class_nm[inv]).sum(dim=1) / T     # [B]

        # hinge ranking
        return F.relu(self.rank_margin + s_nm - s_soft).mean()

    def forward(self, image_features, text_features, pid_labels, near_miss_features=None, return_separate=True):
        l_mm = self.loss_mm_proxy_to_image(image_features, text_features, pid_labels)
        l_ce = self.loss_ce_image_to_proxy(image_features, text_features, pid_labels)

        l_rank = None
        if near_miss_features is not None and self.alpha_rank > 0:
            l_rank = self.loss_rank_soft_vs_nearmiss(image_features, text_features, pid_labels, near_miss_features)
            total = l_mm + self.alpha_ce * l_ce + self.alpha_rank * l_rank
        else:
            total = l_mm + self.alpha_ce * l_ce

        if return_separate:
            # return l_rank=0 when absent for easy logging
            return l_mm, l_ce, (l_rank if l_rank is not None else image_features.new_tensor(0.0)), total
        return total

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
    def __init__(self, tau=0.07, p=1.0, label_smoothing=0.0,
                 min_log_tau=-6.0, max_log_tau=2.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))
        self.p = nn.Parameter(torch.tensor(p))  # learnable sharpness
        self.label_smoothing = label_smoothing
        self.min_log_tau = min_log_tau
        self.max_log_tau = max_log_tau
        self.eps = 1e-6

    def gem_pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, D] -> pooled: [B, D]
        """
        p = self.p.clamp(1.0, 4.0)

        # Signed GeM over tokens: sign(mean) * GeM(|x|)
        mag = tokens.abs().clamp_min(self.eps)
        pooled_mag = mag.pow(p).mean(dim=1).clamp_min(self.eps).pow(1.0 / p)
        pooled_sign = tokens.mean(dim=1).sign()
        return pooled_sign * pooled_mag

    def forward(self, image_tokens, text_features, pid_labels):
        # Normalize tokens + text prototypes
        image_tokens = F.normalize(image_tokens, dim=-1)   # [B,N,D]
        text_features = F.normalize(text_features, dim=-1) # [C,D]

        # 1) GeM pool tokens -> [B,D]
        img_emb = self.gem_pool_tokens(image_tokens)
        img_emb = F.normalize(img_emb, dim=-1)

        # 2) CLIP-style logits over classes
        tau = self.log_tau.clamp(self.min_log_tau, self.max_log_tau).exp()
        logits = (img_emb @ text_features.t()) / tau       # [B,C]

        # 3) CE loss
        return F.cross_entropy(
            logits, pid_labels.long(), label_smoothing=self.label_smoothing
        )


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