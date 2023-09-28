import pandas as pd
import numpy as np

import xxhash
import numbers
import datetime


def digest(*argv):
    x = xxhash.xxh64(seed=0)

    for arg in argv:
        _update_hash(arg, lambda b: x.update(b))

    return x.hexdigest()


# We mostly rely on numpy to extract byte representation
# And feed those to the hash.

def _update_hash(v, update):
    # always update by the type, then by the value.
    # this is to differentiate empty string, empty list, None, 0, False etc.

    update(bytes(str(type(v)), "utf8"))

    if isinstance(v, str):
        update(bytes(v, "utf8"))
        return

    if isinstance(v, (bytes, bytearray)):
        update(v)
        return

    if isinstance(v, (numbers.Number, datetime.date, bool)):
        # Use numpy scalar to get bytes
        update(np.array(v).data.tobytes())
        return

    if v is None:
        update(b'')

    if isinstance(v, pd.Series):
        # Convert to numpy then do the numpy logic
        v = v.to_numpy()

    if isinstance(v, (np.ndarray, np.generic)):

        # If we have a uniform basic type, get bytes from numpy, otherwise recurse. 
        # Uses .item() to convert from a numpy object to python

        if v.dtype is not np.dtype('O'):
            update(v.data.tobytes())
        else:
            for x in v:
                _update_hash(x.item(), update)
        return

    if isinstance(v, (pd.DataFrame, pd.Index)):
        # hash_pandas_object return one hash per row, so we only use it on complex objects
        update(pd.util.hash_pandas_object(v).to_numpy().data.tobytes())
        return

    if isinstance(v, (list, set, tuple)):

        # If we have a uniform basic type, get bytes from numpy, otherwise recurse. 

        try:
            arr = np.array(v)
            if arr.dtype is not np.dtype('O'):
                update(arr.data.tobytes())
            else:
                for x in v:
                    _update_hash(x, update)
            return

        except (ValueError, TypeError):
            for x in v:
                _update_hash(x, update)
            return

    else:

        members = sorted(_list_members(v))

        # for objects and dict, we list (members, values) and hash those
        if (len(members) > 0):
            for x in members:
                _update_hash(x[0], update)
                _update_hash(x[1], update)

        else:
            # if properties are not iterables, use string representation
            update(bytes(str(v), "utf8"))
            return


def _list_members(obj):
    if isinstance(obj, dict):
        return list(obj.items())

    return ([(a, getattr(obj, a))
             for a in dir(obj) if not a.startswith('_') and not callable(getattr(obj, a))
             ])
