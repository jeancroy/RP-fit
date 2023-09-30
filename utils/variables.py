import numpy as np
import pandas as pd
import numbers


# The pack method is used to transform a dict name-> list of numbers into a 1D vector.
# We also generate the metadata needed to do the opposite transform.
# We also support name -> scalar and nested dict.

def pack(source, append_to=None):
    output = [] if append_to is None else append_to
    unpack_info = {}

    for k, v in source.items():

        start = len(output)

        if isinstance(v, numbers.Number):
            output.append(v)
            unpack_info[k] = (start, type(v), 1, None)

        elif isinstance(v, np.ndarray) and v.size > 0:
            if v.ndim > 0:
                output.extend(v)
            else:
                output.append(v) # scalar ndarray

            unpack_info[k] = (start, type(v), v.size, v.shape)

        elif isinstance(v, pd.Series) and len(v) > 0:
            output.extend(v)
            unpack_info[k] = (start, type(v), v.size, v.index)

        elif isinstance(v, (list, tuple)) and len(v) > 0:
            output.extend(v)
            unpack_info[k] = (start, type(v), len(v), None)

        elif isinstance(v, dict):
            _, nfo = pack(v, append_to=output)
            unpack_info[k] = (start, type(v), 1, nfo)

    return np.array(output), unpack_info


def unpack(x, unpack_info):
    unpacked = {}

    for k, v in unpack_info.items():

        start, type_info, size, metadata = v

        if (type_info in (int, float, complex, numbers.Number)):
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
