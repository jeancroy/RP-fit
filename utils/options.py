from enum import Enum
import numpy as np


def merge_dict_like(destination, source):
    if source is None:
        return destination

    write_to = destination if isinstance(destination, dict) else destination.__dict__
    read_from = source if isinstance(source, dict) else source.__dict__

    for key, value in read_from.items():

        if key.startswith('_') or callable(value):
            continue

        if key not in write_to:
            write_to[key] = value

        elif isinstance(value, (bool, str, int, float, type(None), list, tuple, Enum, np.ndarray)):
            write_to[key] = value

        else:
            merge_dict_like(write_to[key], value)

    return destination
