import numpy as np
import pandas as pd
import numbers
from copy import copy
from collections import namedtuple

PackInfo = namedtuple("PackInfo", "type start size metadata rescale")
RangeInfo = namedtuple("RangeInfo", "low high")
ScaleInfo = namedtuple("ScaleInfo", "offset scale")


# The pack method is used to transform a dict name-> list of numbers into a 1D numpy vector.
# We also generate the metadata needed to do the opposite transform.
# We also support name -> scalar and nested dict.
#
# Frozen keys allow removing a variable from the 1D vector without changing the code
# that consumes the unpacked dict. It'll also be used for non-numeric keys.
#
# Range_info_dict is of the same shape as the source.
# But each leaf is either None or a RangeInfo tuple.
# The goal is to allow the optimizer to work with values
# normalized to be in the [-1, 1] range most of the time.

def pack(source, range_info_dict=None, append_to=None, frozen_keys=None):
    if frozen_keys is None:
        frozen_keys = []

    output = [] if append_to is None else append_to
    unpack_info = {}

    for key, value in source.items():

        start = len(output)

        # Frozen keys are not packed in the output vector
        # But they will be copied as is to the unpacked dict
        if key in frozen_keys or isinstance(value, str):
            unpack_info[key] = PackInfo(type(value), None, None, copy(value), None)

        elif isinstance(value, (int, float, complex)):
            scaled, scale_info = pack_rescale(value, key, range_info_dict)
            output.append(scaled)
            unpack_info[key] = PackInfo(type(value), start, 1, None, scale_info)

        elif isinstance(value, np.ndarray) and value.size > 0:
            scaled, scale_info = pack_rescale(value, key, range_info_dict)

            if value.ndim > 0:
                output.extend(scaled)
            else:
                output.append(scaled)  # scalar ndarray

            unpack_info[key] = PackInfo(type(value), start, value.size, value.shape, scale_info)

        elif isinstance(value, pd.Series) and value.size > 0:
            scaled, scale_info = pack_rescale(value, key, range_info_dict)

            output.extend(scaled)
            unpack_info[key] = PackInfo(type(value), start, value.size, value.index, scale_info)

        elif isinstance(value, (list, tuple)):

            # Empty list or non-numeric treated the same as frozen
            if len(value) == 0 or not isinstance(value[0], numbers.Number):
                unpack_info[key] = PackInfo(type(value), None, None, copy(value), None)

            else:
                scaled, scale_info = pack_rescale(np.array(value), key, range_info_dict)
                output.extend(scaled)
                unpack_info[key] = PackInfo(type(value), start, len(value), type(value[0]), scale_info)

        elif isinstance(value, dict):
            _, nfo = pack(value, range_info_dict, append_to=output)
            unpack_info[key] = PackInfo(type(value), start, 1, nfo, None)

    return np.array(output), unpack_info


def unpack(x, unpack_info):
    unpacked = {}

    for k, v in unpack_info.items():

        type_info, start, size, metadata, scale_info = v
        value = unpack_rescale(x, start, size, scale_info)

        if start is None:
            unpacked[k] = metadata

        elif type_info in (int, float, complex, numbers.Number):
            unpacked[k] = value

        elif type_info is np.ndarray:
            unpacked[k] = np.array(value).reshape(metadata)

        elif type_info is pd.Series:
            unpacked[k] = pd.Series(data=value, index=metadata)

        elif type_info is list:
            unpacked[k] = list(value)

        elif type_info is tuple:
            unpacked[k] = tuple(value)

        elif type_info is dict:
            unpacked[k] = unpack(x, metadata)

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


def simplify_opt_result(opt):
    # remove some stuff we don't need to save.
    if "jac" in opt:
        del opt.jac
    if "active_mask" in opt:
        del opt.active_mask
    if "fun" in opt:
        del opt.fun
    if "final_simplex" in opt:
        del opt.final_simplex

    return opt


def lookup_table(serie1, table2, table2_lookup_column, table2_return_column):
    return serie1.map(table2.set_index(table2_lookup_column)[table2_return_column])
