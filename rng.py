"""
Random number generator with context management
"""
from contextlib import contextmanager
import numpy as np

_registered_rngs = {}
current_rng = np.random.RandomState(0)


def register_rng(name, rng=None):
    """Register a random number generator with a name"""
    global _registered_rngs
    if rng is None:
        rng = np.random.RandomState(0)
    _registered_rngs[name] = rng


def get_rng():
    """Get the current random number generator"""
    global current_rng
    return current_rng


@contextmanager
def rng_context(name):
    """Context manager for random number generation"""
    global _registered_rngs
    global current_rng
    prev_context = current_rng
    if name not in _registered_rngs:
        register_rng(name)
    current_rng = _registered_rngs[name]
    yield
    current_rng = prev_context

register_rng("expert")
register_rng("router")
register_rng("testing")
register_rng("expert_with_rank")
