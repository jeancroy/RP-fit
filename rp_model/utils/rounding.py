import numpy as np

# Sometime the value 0.020 really is 0.019999999999999997,
# And that gets floored to 0.019 when asking for truncation at 3 decimals.
# And that is enough for ~10 RP difference
eps = np.finfo(np.float64).eps


def round_precise(values):
    return np.rint(np.nextafter(values, values + 1))


# Ie truncate(value, 2)
def truncate(value, digits):
    scale = 10 ** digits
    return np.floor(scale * (value + eps)) / scale


# Ie floor(value, 0.01)
def floor(value, resolution):
    return np.floor((value + eps) / resolution) * resolution
