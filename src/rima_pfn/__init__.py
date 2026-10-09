from typing import Optional

_RANDOM_STATE = 42

def random_state(state: Optional[int] = None):
    global _RANDOM_STATE

    if state is not None:
        _RANDOM_STATE = state

    return _RANDOM_STATE

