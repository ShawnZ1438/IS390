from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from src.memory.replay_buffer import ReplayBuffer, ReplayConfig
from src.memory.prototypes import PrototypeMemory, PrototypeConfig
from src.training.losses import latent_partition_loss
from src.training.metrics import accuracy

@dataclass
class TrainConfig:
    device: str = "cuda"
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs_per_bucket: int = 1
    batch_size: int = 64
    log_every: int = 50

@dataclass
class InemoLikeConfig:
    enabled: bool = True
    prototypes_enabled: bool = True
    proto_momentum: float = 0.9
    latent_partition_enabled: bool = True
    partition_strength: float = 0.05
    partition_margin: float = 0.2

class StreamingTrainer:
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        feature_dim: int,
        num_classes: int,
        train_cfg: TrainConfig,
        replay_cfg: ReplayConfig,
        inemo_cfg: InemoLikeConfig,
        replay_ratio: float = 0.3,
    ):
        self.backbone = backbone
        self.head = head
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.cfg = train_cfg
        self.replay_ratio = float(replay_ratio)
        self.device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        self.head.to(self.device)
        self.optim = AdamW(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.crit = nn.CrossEntropyLoss()
        self.buffer = ReplayBuffer(replay_cfg)
        self.inemo = inemo_cfg
        self.proto_mem: Optional[PrototypeMemory] = None
        if self.inemo.enabled and self.inemo.prototypes_enabled:
            self.proto_mem = PrototypeMemory(
                feature_dim=feature_dim,
                cfg=PrototypeConfig(momentum=self.inemo.proto_momentum),
            )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return feats, logits

    def train_one_bucket(self, train_loader, bucket_idx: int) -> None:
        self.backbone.train()
        self.head.train()
        step = 0
        for epoch in range(self.cfg.epochs_per_bucket):
            pbar = tqdm(train_loader, desc=f"Train bucket {bucket_idx} | epoch {epoch+1}")
            for x, y in pbar:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                cur_bs = x.size(0)
                
                # Mix in replay
                if len(self.buffer) > 0 and self.replay_ratio > 0:
                    rb = int(round(cur_bs * self.replay_ratio))
                    rb = max(1, rb) if rb > 0 else 0
                    if rb > 0:
                        rx, ry = self.buffer.sample(rb)
                        rx = rx.to(self.device, non_blocking=True)
                        ry = ry.to(self.device, non_blocking=True)
                        x_mix = torch.cat([x, rx], dim=0)
                        y_mix = torch.cat([y, ry], dim=0)
                    else:
                        x_mix, y_mix = x, y
                else:
                    x_mix, y_mix = x, y
                
                feats, logits = self.forward(x_mix)
                loss = self.crit(logits, y_mix)
                
                # iNeMo-inspired: update prototypes using current (non-replay) slice
                if self.proto_mem is not None:
                    with torch.no_grad():
                        self.proto_mem.update(feats[:cur_bs], y[:cur_bs])
                
                # iNeMo-inspired: latent partition loss (positional + separation)
                if (
                    self.inemo.enabled
                    and self.inemo.latent_partition_enabled
                    and self.proto_mem is not None
                ):
                    proto_mat, proto_keys = self.proto_mem.get_matrix(self.device)
                    lp = latent_partition_loss(
                        feats, y_mix, proto_mat, proto_keys,
                        margin=self.inemo.partition_margin
                    )
                    if torch.isfinite(lp):
                        loss = loss + self.inemo.partition_strength * lp

                if not torch.isfinite(loss):
                    self.optim.zero_grad(set_to_none=True)
                    continue
                
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                self.optim.step()
                
                # Update replay buffer with current batch only
                self.buffer.add_batch(x, y)
                
                if step % self.cfg.log_every == 0:
                    acc = accuracy(logits.detach(), y_mix.detach())
                    pbar.set_postfix(loss=float(loss.item()), acc=float(acc), buf=len(self.buffer))
                step += 1

    @torch.no_grad()
    def eval_loader(self, loader) -> float:
        self.backbone.eval()
        self.head.eval()
        accs = []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            _, logits = self.forward(x)
            accs.append(accuracy(logits, y))
        return float(sum(accs) / max(1, len(accs)))
