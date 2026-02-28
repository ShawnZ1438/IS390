from typing import Tuple
import torch.nn as nn
from torchvision.models import resnet18, resnet34

def build_backbone(name: str, feature_dim: int) -> Tuple[nn.Module, int]:
    if name == "resnet18":
        m = resnet18(weights=None)
    elif name == "resnet34":
        m = resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    
    in_dim = m.fc.in_features
    m.fc = nn.Identity()
    
    if feature_dim != in_dim:
        proj = nn.Linear(in_dim, feature_dim)
        model = nn.Sequential(m, proj)
        out_dim = feature_dim
    else:
        model = m
        out_dim = in_dim
    
    return model, out_dim
