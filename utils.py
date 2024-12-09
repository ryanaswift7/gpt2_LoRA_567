import random
import numpy as np
import torch
import os

def enforce_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def use_single_cuda_device():
    if torch.cuda.device_count() > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
