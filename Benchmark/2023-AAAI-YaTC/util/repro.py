"""可复现实验：统一随机种子与 DataLoader worker 种子。"""

import random
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed_deterministic(seed: int) -> torch.Generator:
    """
    设置 Python / NumPy / PyTorch / CUDA 的随机种子，并关闭 cudnn 非确定性算法。
    返回用于 DataLoader 的 torch.Generator（需传入 DataLoader(generator=...)）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 内再播种，与主进程 torch.initial_seed 对齐。"""
    worker_seed = (torch.initial_seed() % 2**32) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
