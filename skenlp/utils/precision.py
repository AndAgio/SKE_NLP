from math import log10

_DEFAULT_PRECISION: float = 1e-6

_precision_options: dict = {'precision': _DEFAULT_PRECISION}


def get_default_precision() -> float:
    return _precision_options['precision']


def get_int_precision() -> int:
    return -1 * int(log10(get_default_precision()))
