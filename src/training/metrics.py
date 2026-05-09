from dataclasses import dataclass, field
from typing import Dict
import torch

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def num_correct(logits: torch.Tensor, y: torch.Tensor) -> int:
    pred = torch.argmax(logits, dim=1)
    return int((pred == y).sum().item())

@dataclass
class StreamingLog:
    nda_by_step: Dict[int, float] = field(default_factory=dict)
    train_bucket_by_step: Dict[int, str] = field(default_factory=dict)
    test_bucket_by_step: Dict[int, str] = field(default_factory=dict)

@dataclass
class ForgettingLog:
    # For each bucket j, store best accuracy achieved so far on its shadow holdout
    best_by_bucket: Dict[int, float] = field(default_factory=dict)
    current_by_bucket: Dict[int, float] = field(default_factory=dict)

    def update_bucket(self, bucket_idx: int, acc: float) -> float:
        prev_best = self.best_by_bucket.get(bucket_idx, 0.0)
        new_best = max(prev_best, acc)
        self.best_by_bucket[bucket_idx] = new_best
        self.current_by_bucket[bucket_idx] = acc
        forgetting = max(0.0, new_best - acc)
        return forgetting
