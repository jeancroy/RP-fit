import numpy as np
import pandas as pd
import numbers
from copy import copy, deepcopy


# The pack method is used to transform a dict name-> list of numbers into a 1D numpy vector.
# We also generate the metadata needed to do the opposite transform.
# We also support name -> scalar and nested dict.
#
# Frozen keys allow removing a variable from the 1D vector without changing the code
# that consumes the unpacked dict. It'll also be used for non-numeric keys.

def pack(source, append_to=None, frozen_keys=None):
    if frozen_keys is None:
        frozen_keys = []

    output = [] if append_to is None else append_to
    unpack_info = {}

    for k, v in source.items():

        start = len(output)

        # Frozen keys are not packed in the output vector
        # But they will be copied as is to the unpacked dict
        if (k in frozen_keys or isinstance(v, str)):
            unpack_info[k] = (type(v), None, None, copy(v))

        elif isinstance(v, numbers.Number):
            output.append(v)
            unpack_info[k] = (type(v), start, 1, None)

        elif isinstance(v, np.ndarray) and v.size > 0:
            if v.ndim > 0:
                output.extend(v)
            else:
                output.append(v)  # scalar ndarray

            unpack_info[k] = (type(v), start, v.size, v.shape)

        elif isinstance(v, pd.Series) and v.size > 0:
            output.extend(v)
            unpack_info[k] = (type(v), start, v.size, v.index)

        elif isinstance(v, (list, tuple)):

            if (len(v) == 0 or not isinstance(v[0], numbers.Number)):
                unpack_info[k] = (type(v), None, None, copy(v))

            else:
                output.extend(v)
                unpack_info[k] = (type(v), start, len(v), type(v[0]))

        elif isinstance(v, dict):
            _, nfo = pack(v, append_to=output)
            unpack_info[k] = (type(v), start, 1, nfo)

    return np.array(output), unpack_info


def unpack(x, unpack_info):
    unpacked = {}

    for k, v in unpack_info.items():

        type_info, start, size, metadata = v

        if (start is None):
            unpacked[k] = metadata

        elif (type_info in (int, float, complex, numbers.Number)):
            unpacked[k] = x[start]

        elif (type_info is np.ndarray):
            unpacked[k] = np.array(x[start: start + size]).reshape(metadata)

        elif (type_info is pd.Series):
            unpacked[k] = pd.Series(data=x[start: start + size], index=metadata)

        elif (type_info is list):
            unpacked[k] = list(x[start: start + size])

        elif (type_info is tuple):
            unpacked[k] = tuple(x[start: start + size])

        elif (type_info is dict):
            unpacked[k] = unpack(x, metadata)

    return unpacked


def lookup_table(serie1, table2, table2_lookup_column, table2_return_column):
    return serie1.map(table2.set_index(table2_lookup_column)[table2_return_column])
