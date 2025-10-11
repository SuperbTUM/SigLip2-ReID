import torch
import torch.nn.functional as F

# Note: The original numpy-based functions are removed for clarity.

def euclidean_distance_pt(qf, gf):
    """
    Calculates the euclidean distance between two tensors.
    Returns a PyTorch tensor on the same device as the input.
    """
    m, n = qf.size(0), gf.size(0)
    # Expanded formula: ||q - g||^2 = ||q||^2 + ||g||^2 - 2 * q @ g.T
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat

def cosine_distance_pt(qf, gf):
    """
    Calculates the cosine distance (angular distance) between two tensors.
    Returns a PyTorch tensor on the same device as the input.
    """
    # Normalize features to compute cosine similarity
    qf_norm = F.normalize(qf, p=2, dim=1)
    gf_norm = F.normalize(gf, p=2, dim=1)
    
    # Cosine similarity is the dot product of normalized vectors
    similarity_matrix = torch.mm(qf_norm, gf_norm.t())
    
    # Clamp for numerical stability
    similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
    
    # Convert similarity to angular distance
    dist_mat = torch.acos(similarity_matrix)
    return dist_mat

def eval_func_pt(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    Evaluation with market1501 metric, optimized for PyTorch.
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    device = distmat.device

    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")

    all_cmc = []
    all_AP = []
    num_valid_q = 0

    for i in range(0, num_q, 128): # Process in chunks of 128
        distmat_chunk = distmat[i:i+128]
        q_pids_chunk = q_pids[i:i+128]
        q_camids_chunk = q_camids[i:i+128]

        # --- Vectorized Junk Image Removal for chunk ---
        q_pids_view = q_pids_chunk.view(-1, 1).expand(-1, num_g)
        g_pids_view = g_pids.view(1, -1).expand(distmat_chunk.shape[0], -1)
        q_camids_view = q_camids_chunk.view(-1, 1).expand(-1, num_g)
        g_camids_view = g_camids.view(1, -1).expand(distmat_chunk.shape[0], -1)
        
        junk_mask = (q_pids_view == g_pids_view) & (q_camids_view == g_camids_view)
        distmat_chunk.masked_fill_(junk_mask, float('inf'))
        # ------------------------------------

        indices_chunk = torch.argsort(distmat_chunk, dim=1)
        matches_chunk = (g_pids[indices_chunk] == q_pids_chunk.view(-1, 1)).int()

        for q_idx in range(distmat_chunk.shape[0]):
            q_matches = matches_chunk[q_idx]

            if not torch.any(q_matches):
                continue

            num_valid_q += 1
            
            cmc = q_matches.cumsum(0)
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:max_rank])
            
            num_rel = q_matches.sum()
            positions = torch.nonzero(q_matches).squeeze()
            
            precision_at_k = (torch.arange(1, num_rel + 1, device=device)) / (positions + 1.0)
            AP = precision_at_k.mean()
            all_AP.append(AP)
            
    if num_valid_q == 0:
        raise RuntimeError("Error: all query identities do not appear in gallery")

    all_cmc = torch.stack(all_cmc).float()
    all_cmc = all_cmc.sum(0) / num_valid_q
    
    mAP = torch.tensor(all_AP).mean()

    return all_cmc, mAP


class R1_mAP_eval_pt():
    def __init__(self, num_query, max_rank=50, feat_norm=True):
        super(R1_mAP_eval_pt, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat) # Keep on original device
        # pid and camid might be on CPU, that's fine for now
        self.pids.append(pid)
        self.camids.append(camid)

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        pids = torch.cat(self.pids, dim=0)
        camids = torch.cat(self.camids, dim=0)
        
        device = feats.device # Get the device where feats are stored (e.g., 'cuda:0')

        if self.feat_norm:
            print("The test feature is normalized")
            feats = F.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = pids[:self.num_query].to(device)
        q_camids = camids[:self.num_query].to(device)
        # gallery
        gf = feats[self.num_query:]
        g_pids = pids[self.num_query:].to(device)
        g_camids = camids[self.num_query:].to(device)

        print('=> Computing DistMat with euclidean_distance_pt')
        distmat = euclidean_distance_pt(qf, gf)
        
        cmc, mAP = eval_func_pt(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)

        # Convert final results to CPU for logging/printing if needed
        return cmc.cpu().numpy(), mAP.item()
