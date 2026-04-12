import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    """Seed random/numpy/torch and enable full CUDA determinism (slower but bit-exact)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Required for deterministic CUBLAS on CUDA >= 10.2.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    print(f"Global seed set to {seed} with full determinism enabled")
