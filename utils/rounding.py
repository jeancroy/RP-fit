from types import SimpleNamespace
from enum import Enum
import numpy as np

soft_round_options = SimpleNamespace(
    alpha=18,
    use_integer_bias=True,
    use_progressive_alpha=False,
    alpha_progression=[3, 6, 12, 18]
)


class RoundApprox(Enum):
    Pass = 0
    Exact = 1
    Soft = 2


class RoundDirection(Enum):
    Floor = -1
    Middle = 0
    Ceil = 1


def floor(value, resolution=1.0):
    return np.floor(value / resolution) * resolution


def optional_floor(value, approx_option, resolution=1.0):
    if approx_option == RoundApprox.Soft:
        return soft_floor(value / resolution) * resolution

    elif approx_option == RoundApprox.Exact:
        return floor(value, resolution)

    else:
        return value


def optional_round(value, approx_option, resolution=1.0):
    if approx_option == RoundApprox.Soft:
        return soft_round(value / resolution) * resolution

    elif approx_option == RoundApprox.Exact:
        return np.round(value / resolution) * resolution

    else:
        return value


# Differentiable approximation of round()
# alpha = 0 gives f(x) = x, then goes closer f(x) = round(x) as alpha goes to inf.
#
# Setting alpha instead of gamma is mostly a convenience for linear progression of effect.

def soft_round(t, alpha=None):
    if (alpha is None):
        alpha = soft_round_options.alpha

    if alpha <= 0.0:
        return t

    gamma = 1.0 / (1.0 - (0.85 ** alpha))
    s_2pi_t = np.sin(2.0 * np.pi * t)
    c_2pi_t = np.cos(2.0 * np.pi * t)
    return t - (1.0 / np.pi) * np.arctan(s_2pi_t / (c_2pi_t + gamma))


# Differentiable approximation of floor()
#
# Unfortunately soft_floor(integer) is always integer-0.5 regardless of alpha.
# This is correct unbiased behavior but may give a strange result when we search for integer values.
#
# The option int_bias shifts the data to give better behavior at integer positions.
# That offset corrects itself as alpha goes to inf, but at a slower pace than sr(x) goes to round(x),
# So the overall effect is that we converge to soft_floor(integer,int_bias=True) == integer.

def soft_floor(t, alpha=None, int_bias=None):
    if (alpha is None):

        if (soft_round_options.alpha is None):
            return np.floor(t)

        alpha = soft_round_options.alpha

    if (int_bias is None):
        int_bias = soft_round_options.use_integer_bias

    if not int_bias:
        return soft_round(t - 0.5, alpha)
    else:
        return soft_round(t + (0.25 * (0.92 ** alpha) - 0.5), alpha)


def progressive_soft_round_loop(x0, optim_func, alpha_progression=None):
    if (not soft_round_options.use_progressive_alpha):
        return optim_func(x0)

    if (alpha_progression is None):
        alpha_progression = soft_round_options.alpha_progression

    x = x0
    opt = {}

    for alpha in alpha_progression:
        soft_round_options.alpha = alpha
        opt = optim_func(x)
        x = opt.x

    return opt
