from typing import Optional

import torch


_RANDOM_STATE = 42
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
TORCH_DEVICE = torch.device(DEVICE)
TORCH_GENERATOR = torch.Generator().manual_seed(_RANDOM_STATE)

def random_state(state: Optional[int] = None):
    global _RANDOM_STATE

    if state is not None:
        _RANDOM_STATE = state

    return _RANDOM_STATE

