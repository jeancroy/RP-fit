from types import SimpleNamespace
import numpy as np

soft_round_options = SimpleNamespace(
    alpha=18,
    exact=False,
)


def soft_round(x, alpha=None):
    if alpha is None and soft_round_options.exact:
        return np.round(x)

    a = alpha if alpha is not None else soft_round_options.alpha

    if a < 0.01:
        return x

    m = np.floor(x) + 0.5
    r = x - m
    z = np.tanh(a / 2.0) * 2.0
    return m + np.tanh(a * r) / z


def soft_round_inverse(y, alpha):
    m = np.floor(y) + .5
    s = (y - m) * (np.tanh(alpha / 2.0) * 2.0)
    r = np.arctanh(s) / alpha
    r = np.clip(r, -.5, .5)
    return m + r


def soft_floor(x, alpha=None):
    if alpha is None and soft_round_options.exact:
        return np.floor(x)

    a = alpha if alpha is not None else soft_round_options.alpha

    if a < 0.01:
        return x

    almost_half = 0.5 - 1e-12
    return soft_round(x - almost_half, alpha)


def soft_floor_top(x, alpha=None):
    if alpha is None and soft_round_options.exact:
        return np.floor(x)

    a = alpha if alpha is not None else soft_round_options.alpha

    if a < 0.01:
        return x

    offset = soft_round_inverse(1 - 0.25 * (0.97 ** a), a)
    return soft_round(x + offset - 1, alpha)
