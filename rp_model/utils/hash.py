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
        # If we have a uniform primitive type, get bytes from numpy.
        # Otherwise, recurse _update_hash on each member.
        if val.dtype is not np.dtype("O"):
            update(val.data.tobytes())
        else:
            for x in val:
                # Using x.item() convert from a numpy object to a python object
                _update_hash(x.item(), update)
        return

    if isinstance(val, (pd.DataFrame, pd.Index)):
        # hash_pandas_object return one hash per row
        # we only use it when we have multiple column
        # or a situation where a single to_numpy is not good enough.
        update(pd.util.hash_pandas_object(val).to_numpy().data.tobytes())
        return

    if isinstance(val, (list, set, tuple)):
        # If we have a uniform primitive type, make a numpy array and get bytes from numpy.
        # Otherwise, recurse _update_hash on each member.
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

        # use reflexion to enumerate the thing as key value pairs
        members = sorted(list_members(val))

        if len(members) > 0:
            for x in members:
                _update_hash(x[0], update)
                _update_hash(x[1], update)

        else:
            # Enumeration failed. Try to convert to a string and hash that.
            update(bytes(str(val), "utf8"))
            return
