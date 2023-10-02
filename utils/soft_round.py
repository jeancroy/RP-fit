from types import SimpleNamespace
import numpy as np

soft_round_options = SimpleNamespace()
soft_round_options.alpha = 18
soft_round_options.exact = False


def soft_round(x, alpha=None):

    if alpha is None and soft_round_options.exact:
        return np.round(x)

    a = alpha if alpha is not None else soft_round_options.alpha
    m = np.floor(x) + 0.5
    r = x - m
    z = np.tanh(a / 2.0) * 2.0
    return m + np.tanh(a * r) / z


def soft_floor(x, alpha=None):

    if alpha is None and soft_round_options.exact:
        return np.floor(x)

    almost_half = 0.5 - 1e-12
    return soft_round(x - almost_half, alpha)
