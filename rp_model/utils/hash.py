import datetime
import numbers

import numpy as np
import pandas as pd
import xxhash

from .display import list_members


def digest(*argv):
    x = xxhash.xxh64(seed=0)

    for arg in argv:
        _update_hash(arg, lambda b: x.update(b))

    return x.hexdigest()


# We mostly rely on numpy to extract byte representation
# And feed those to the hash.

def _update_hash(val, update):
    # always update by the type, then by the value.
    # this is to differentiate empty string, empty list, None, 0, False etc.
    update(bytes(str(type(val)), "utf8"))

    if isinstance(val, str):
        update(bytes(val, "utf8"))
        return

    if isinstance(val, (bytes, bytearray)):
        update(val)
        return

    if isinstance(val, (numbers.Number, datetime.date, bool)):
        # Use numpy scalar to get bytes
        update(np.array(val).data.tobytes())
        return

    if val is None:
        update(b"")

    if isinstance(val, pd.Series):
        # Convert to numpy then do the numpy logic
        val = val.to_numpy()

    if isinstance(val, (np.ndarray, np.generic)):
        # If we have a uniform basic type, get bytes from numpy, otherwise recurse. 
        # Uses .item() to convert from a numpy object to python
        if val.dtype is not np.dtype("O"):
            update(val.data.tobytes())
        else:
            for x in val:
                _update_hash(x.item(), update)
        return

    if isinstance(val, (pd.DataFrame, pd.Index)):
        # hash_pandas_object return one hash per row, so we only use it on complex objects
        update(pd.util.hash_pandas_object(val).to_numpy().data.tobytes())
        return

    if isinstance(val, (list, set, tuple)):
        # If we have a uniform basic type, get bytes from numpy, otherwise recurse.
        try:
            arr = np.array(val)
            if arr.dtype is not np.dtype('O'):
                update(arr.data.tobytes())
            else:
                for x in val:
                    _update_hash(x, update)
            return

        except (ValueError, TypeError):
            for x in val:
                _update_hash(x, update)
            return
    else:
        members = sorted(list_members(val))

        # for objects and dict, we list (members, values) and hash those
        if len(members) > 0:
            for x in members:
                _update_hash(x[0], update)
                _update_hash(x[1], update)

        else:
            # if properties are not iterables, use string representation
            update(bytes(str(val), "utf8"))
            return
