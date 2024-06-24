import numpy as np
import pandas as pd
import numbers
from copy import copy
from collections import namedtuple

PackInfo = namedtuple("PackInfo", "type start size rescale source")
RangeInfo = namedtuple("RangeInfo", "low high")
ScaleInfo = namedtuple("ScaleInfo", "offset scale")

frozen_type = type("frozen")
subset_type = type("subset")


# The pack method is used to transform a dict name-> list of numbers into a 1D numpy vector.
# We also generate the metadata needed to do the opposite transform.
# We also support name -> scalar and nested dict.
#
# Range_info_dict is of the same shape as the source.
#  - Each leaf is either None or a RangeInfo tuple.
#  - Allows the optimizer to work with values
#    normalized to be in [-1, 1] most of the time.
#  - Those are not hard bound, just scaling.
#
# Frozen keys allow removing a variable from the 1D vector that the optimizer sees,
# without changing the code that consumes the unpacked dict.
# It'll also be used for non-numeric keys.
#
# Subset keys (todo) are the same idea as frozen key.
# But a small subset is allowed to change.
#  - extracted from the vector when packing
#  - copied over the frozen base when upacking.

def pack(source, range_info_dict=None, append_to=None, frozen_keys=None):
    if frozen_keys is None:
        frozen_keys = []

    output = [] if append_to is None else append_to
    unpack_info = {}

    for key, value in source.items():

        # This is the position of the value in output vector
        start = len(output)

        # Frozen keys are not packed in the output vector (position=none)
        # But they will be copied as-is to the unpacked dict
        if key in frozen_keys or isinstance(value, str):
            unpack_info[key] = PackInfo(frozen_type, None, None, None, copy(value))

        elif isinstance(value, (int, float, complex)):
            scaled_value, scale_info = pack_rescale(value, key, range_info_dict)
            output.append(scaled_value)
            unpack_info[key] = PackInfo(type(value), start, 1, scale_info, None)

        elif isinstance(value, np.ndarray) and value.size > 0:
            scaled_value, scale_info = pack_rescale(value, key, range_info_dict)

            if value.ndim > 0:
                output.extend(scaled_value)
            else:
                output.append(scaled_value)  # scalar ndarray

            unpack_info[key] = PackInfo(type(value), start, value.size, scale_info, copy(value))

        elif isinstance(value, pd.Series) and value.size > 0:
            scaled_value, scale_info = pack_rescale(value, key, range_info_dict)
            output.extend(scaled_value)
            unpack_info[key] = PackInfo(type(value), start, value.size, scale_info, copy(value))

        elif isinstance(value, (list, tuple)):

            # Empty list or non-numeric list treated the same as frozen
            if len(value) == 0 or not isinstance(value[0], numbers.Number):
                unpack_info[key] = PackInfo(frozen_type, None, None, None, copy(value))

            else:
                scaled_value, scale_info = pack_rescale(np.array(value), key, range_info_dict)
                output.extend(scaled_value)
                unpack_info[key] = PackInfo(type(value), start, len(value), scale_info, copy(value))

        elif isinstance(value, dict):
            # for dict, we basically recurse.
            _, nfo = pack(value, range_info_dict, append_to=output)
            unpack_info[key] = PackInfo(type(value), start, 1, None, nfo)

    return np.array(output), unpack_info


def unpack(x, unpack_info):
    unpacked = {}

    for name, packed_value in unpack_info.items():

        type_info, start, size, scale_info, source = packed_value
        value = unpack_rescale(x, start, size, scale_info)

        if type_info is frozen_type:
            unpacked[name] = source

        elif type_info in (int, float, complex, numbers.Number):
            unpacked[name] = value

        elif type_info is np.ndarray:
            unpacked[name] = np.array(value).reshape(source.shape)

        elif type_info is pd.Series:
            unpacked[name] = pd.Series(data=value, index=source.index)

        elif type_info is list:
            unpacked[name] = list(value)

        elif type_info is tuple:
            unpacked[name] = tuple(value)

        elif type_info is dict:
            unpacked[name] = unpack(x, source)

    return unpacked


def pack_rescale(value, key, range_info_dict):
    if range_info_dict is None or key not in range_info_dict:
        return value, None

    range_info = range_info_dict[key]

    if range_info is None:
        return value, None

    low, high = range_info
    offset = (high + low) * 0.5
    scale = np.abs(high - low) * 0.5
    rescaled = (value - offset) / scale
    return rescaled, ScaleInfo(offset, scale)


def unpack_rescale(x, start, size, scale_info):
    if start is None:
        return None

    value = x[start] if (size is None or size < 2) else x[start: start + size]

    if scale_info is None:
        return value

    offset, scale = scale_info
    return value * scale + offset
