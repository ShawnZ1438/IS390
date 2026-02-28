from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, List
import random
import torch

Strategy = Literal["reservoir", "biased_reservoir", "biased", "fifo"]

@dataclass
class ReplayConfig:
    buffer_size: int = 2000
    strategy: Strategy = "reservoir"
    alpha: float = 1.0  # used for biased_reservoir
    per_class_cap: Optional[int] = 50
    seed: int = 42

class ReplayBuffer:
    """
    Stores (image_tensor, label) on CPU.
    Supports:
    - reservoir: uniform reservoir sampling
    - biased_reservoir / biased: replacement prob scaled by alpha (alpha>1 favors recency)
    - fifo: most-recent queue
    """
    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.n_seen = 0
        self.x: List[torch.Tensor] = []
        self.y: List[int] = []
        self.class_counts: Dict[int, int] = {}
        self.fifo_ptr = 0

    def __len__(self) -> int:
        return len(self.x)

    def _class_cap_allows(self, lab: int) -> bool:
        if self.cfg.per_class_cap is None:
            return True
        return self.class_counts.get(lab, 0) < self.cfg.per_class_cap

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        for i in range(images.size(0)):
            self._add_one(images[i], int(labels[i].item()))

    def _add_one(self, img: torch.Tensor, lab: int) -> None:
        self.n_seen += 1
        if not self._class_cap_allows(lab):
            return

        if len(self.x) < self.cfg.buffer_size:
            self.x.append(img)
            self.y.append(lab)
            self.class_counts[lab] = self.class_counts.get(lab, 0) + 1
            return

        # Buffer full
        if self.cfg.strategy == "fifo":
            j = self.fifo_ptr % self.cfg.buffer_size
            self.fifo_ptr += 1
            old = self.y[j]
            self.class_counts[old] -= 1
            self.x[j] = img
            self.y[j] = lab
            self.class_counts[lab] = self.class_counts.get(lab, 0) + 1
            return

        # Reservoir-like
        m = self.cfg.buffer_size
        p = m / float(self.n_seen)
        if self.cfg.strategy in ("biased_reservoir", "biased"):
            p = min(1.0, self.cfg.alpha * p)
        
        if self.rng.random() < p:
            j = self.rng.randrange(0, m)
            old = self.y[j]
            self.class_counts[old] -= 1
            self.x[j] = img
            self.y[j] = lab
            self.class_counts[lab] = self.class_counts.get(lab, 0) + 1

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.x) == 0:
            raise ValueError("ReplayBuffer is empty.")
        idx = [self.rng.randrange(0, len(self.x)) for _ in range(n)]
        bx = torch.stack([self.x[i] for i in idx], dim=0)
        by = torch.tensor([self.y[i] for i in idx], dtype=torch.long)
        return bx, by
