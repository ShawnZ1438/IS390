import os
import time
import random
import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path