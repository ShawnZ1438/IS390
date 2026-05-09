from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
@dataclass
class PrototypeConfig:
	momentum: float = 0.9


class PrototypeMemory:
	"""
	One prototype per class (EMA-updated feature centroid).
	Stored on CPU; moved to device on request.
	"""

	def __init__(self, feature_dim: int, cfg: PrototypeConfig):
		self.feature_dim = feature_dim
		self.cfg = cfg
		self.protos: Dict[int, torch.Tensor] = {}

	@torch.no_grad()
	def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
		feats = feats.detach().cpu()
		labels = labels.detach().cpu()
		m = float(self.cfg.momentum)
		for f, y in zip(feats, labels):
			y = int(y.item())
			if y not in self.protos:
				self.protos[y] = f.clone()
			else:
				self.protos[y] = m * self.protos[y] + (1.0 - m) * f

	def get_matrix(self, device: torch.device) -> Tuple[Optional[torch.Tensor],
														 Optional[List[int]]]:
		if not self.protos:
			return None, None
		keys = sorted(self.protos.keys())
		mat = torch.stack([self.protos[k] for k in keys], dim=0).to(device)
		return mat, keys