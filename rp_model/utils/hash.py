import datetime
import numbers
from typing import Callable

import numpy as np
import pandas as pd
import xxhash

from .display import list_members


def digest(*argv):
    x = xxhash.xxh64(seed=0)

    for arg in argv:
        _update_hash(arg, lambda b: x.update(b))

    return x.hexdigest()


def _update_by_np_array(val: np.ndarray, update: Callable[[bytes], None]):
    val.sort()

    if val.dtype != np.dtype("object"):
        # Not object type for the data type, get bytes from numpy
        update(val.data.tobytes())
        return

    for x in val:
        _update_hash(x, update)


# Relies on numpy to extract byte representation and feed those to the hash.
def _update_hash(val, update: Callable[[bytes], None]):
    # Mark the typing of the value
    update(bytes(str(type(val)), "utf8"))

    if isinstance(val, str):
        update(bytes(val, "utf8"))
        return

    if isinstance(val, (bytes, bytearray)):
        update(val)
        return

    if isinstance(val, (numbers.Number, datetime.date, bool)):
        # Use numpy scalar to get bytes
        _update_by_np_array(np.array(val), update)
        return

    if val is None:
        update(b"")
        return

    if isinstance(val, pd.Series):
        # Convert to numpy then do the numpy logic
        _update_by_np_array(val.to_numpy(), update)
        return

    if isinstance(val, (np.ndarray, np.generic)):
        # If we have a uniform primitive type, get bytes from numpy.
        # Otherwise, recurse _update_hash on each member.
        _update_by_np_array(val, update)
        return

    if isinstance(val, (pd.DataFrame, pd.Index)):
        # hash_pandas_object return one hash per row
        # we only use it when we have multiple column
        # or a situation where a single to_numpy is not good enough.
        _update_by_np_array(pd.util.hash_pandas_object(val).to_numpy(), update)
        return

    if isinstance(val, (list, set, tuple)):
        _update_by_np_array(np.array(val), update)
        return

    # use reflexion to list the thing as key value pairs
    members = sorted(list_members(val))
    if len(members) > 0:
        for x in members:
            _update_hash(x[0], update)
            _update_hash(x[1], update)
        return

    # Enumeration failed. Try to convert to a string and hash that.
    update(bytes(str(val), "utf8"))
    return
