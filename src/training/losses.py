from typing import Optional, List, Dict
import torch
import torch.nn.functional as F


def _label_to_proto_index(proto_keys: List[int]) -> Dict[int, int]:
    return {k: i for i, k in enumerate(proto_keys)}


def positional_regularization_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    proto_mat: Optional[torch.Tensor],
    proto_keys: Optional[List[int]],
) -> torch.Tensor:
    """
    Proxy for iNeMo's positional regularization:
    penalize distance of feature to its class centroid / prototype.
    Here: 1 - cosine_similarity(feature, prototype).
    """
    if proto_mat is None or proto_keys is None:
        return torch.tensor(0.0, device=feats.device)

    f = F.normalize(feats, dim=1)
    p = F.normalize(proto_mat, dim=1)
    key_to_idx = _label_to_proto_index(proto_keys)
    idx = torch.tensor(
        [key_to_idx.get(int(y.item()), -1) for y in labels],
        device=feats.device,
    )
    valid = idx >= 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=feats.device)

    fv = f[valid]
    iv = idx[valid]
    pos = (fv * p[iv]).sum(dim=1)  # cosine sim
    return (1.0 - pos).mean()


def margin_separation_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    proto_mat: Optional[torch.Tensor],
    proto_keys: Optional[List[int]],
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Encourage proto_y to be margin-better than hardest negative prototype.
    hinge: max(0, margin - (s_pos - s_neg))
    """
    if proto_mat is None or proto_keys is None or len(proto_keys) < 2:
        return torch.tensor(0.0, device=feats.device)

    f = F.normalize(feats, dim=1)
    p = F.normalize(proto_mat, dim=1)
    sims = f @ p.t()  # (B, C_protos)
    key_to_idx = _label_to_proto_index(proto_keys)
    idx = torch.tensor(
        [key_to_idx.get(int(y.item()), -1) for y in labels],
        device=feats.device,
    )
    valid = idx >= 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=feats.device)

    sims_v = sims[valid]
    idx_v = idx[valid]

    pos = sims_v[torch.arange(sims_v.size(0), device=feats.device), idx_v]
    mask = torch.ones_like(sims_v, dtype=torch.bool)
    mask[torch.arange(sims_v.size(0), device=feats.device), idx_v] = False
    neg = sims_v.masked_fill(~mask, float("-inf")).max(dim=1).values
    return torch.clamp(margin - (pos - neg), min=0.0).mean()


def latent_partition_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    proto_mat: Optional[torch.Tensor],
    proto_keys: Optional[List[int]],
    margin: float = 0.2,
) -> torch.Tensor:
    # Combine "positional" + "separation" terms
    pos = positional_regularization_loss(
        feats, labels, proto_mat, proto_keys
    )
    sep = margin_separation_loss(
        feats, labels, proto_mat, proto_keys, margin=margin
    )
    return torch.nan_to_num(pos + sep, nan=0.0, posinf=0.0, neginf=0.0)
