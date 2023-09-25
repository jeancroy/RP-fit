import numpy as np
import pandas as pd
import xxhash
import numbers
import pickle
from tabulate import tabulate
from IPython.display import display


def save(filepath, data):
    with open(filepath,"xb") as handle:
        pickle.dump(data, handle)

def load(filepath):
    with open(filepath,"rb") as handle:
        return pickle.load(handle)

def pack(source):

    x =  []
    unpack_info = {}

    for k,v in source.items():
    
        start = len(x)
            
        if isinstance(v, (list, np.ndarray, pd.Series) ) and len(v) > 0:        
            x.extend(v)
            unpack_info[k] = (start, len(v))
        
        elif isinstance(v, numbers.Number):
            x.append(v)
            unpack_info[k] = (start,0)
    
    return np.array(x), unpack_info


def unpack(x, unpack_info):

    unpacked = {}

    for k,v in unpack_info.items():

        start, length = v

        if(length==0):
            unpacked[k] = x[start]
        else:
            unpacked[k] = np.array(x[start:start+length])
    
    return unpacked


def table(data): 
    display(tabulate(data,tablefmt='html'))
    
def tabledict(dictionary): 
    
    table([( [k]+ (
                [v] if isinstance(v, (numbers.Number, str) ) else
                [v] if len(v) < 2 else
                [np.array2string(v, threshold=10)]
            )
        ) for k,v in dictionary.items() ])

def tableobj(obj): 
    tabledict(dict(list_members(obj)))
    
def list_members(obj): 
    
    if isinstance(obj, dict):
        return list(obj.items())
    
    return ([ (a, getattr(obj, a) )
              for a in dir(obj) if not a.startswith('_') and not callable(getattr(obj, a))
            ])

# We mostly rely on numpy to extract byte representation
# And feed those to the hash.

def _update_hash(v, update):

    if isinstance(v, str):
        update( bytes(v, "utf8") )

    elif isinstance(v, (np.ndarray, np.generic) ):
        update( v.data.tobytes() )

    elif isinstance(v, pd.Series):
        update( v.to_numpy().data.tobytes() )

    elif isinstance(v, pd.DataFrame):
        for col in sorted(v.columns):
            update( bytes(col, "utf8") )
            update( v[col].to_numpy().data.tobytes() )

    elif isinstance(v, (list,set)):
        update( np.array(v).data.tobytes() )

    elif isinstance(v, numbers.Number):

        # it is surprisingly hard to convert a single number to bytes in python.
        # so again we use numpy. This is slow but large quantity of numbers
        # should be in one of the other data types.
        np.array([v]).data.tobytes()

    elif isinstance(v, tuple):

        # tuple can store mixed types so we do recursion.
        for x in v:
            _update_hash(x, update)

    else:

        # for objects and dict, list members and values then hash them
        for x in sorted(list_members(v)):
            _update_hash(x, update)
    
    
def digest(v):
    x = xxhash.xxh64(seed=0)
    _update_hash(v, lambda b: x.update(b))
    return x.hexdigest()