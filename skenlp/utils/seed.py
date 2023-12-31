import random
from random import Random
import torch
import numpy as np


_DEFAULT_RANDOM_SEED: int = 123

_random_options = dict(_deterministic_mode=True, _default_random_seed=_DEFAULT_RANDOM_SEED)

_random_seed_generator: Random = Random(_DEFAULT_RANDOM_SEED)


def set_seeds(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def is_deterministic_mode():
    return _random_options['_deterministic_mode']


def set_deterministic_mode(value: bool):
    _random_options['_deterministic_mode'] = value


def get_default_random_seed():
    if is_deterministic_mode():
        return _random_options['_default_random_seed']
    else:
        return _random_seed_generator.randint(0, 1 << 64)


def set_default_random_seed(value: int):
    _random_options['_default_random_seed'] = value
