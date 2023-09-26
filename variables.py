import numpy as np
import pandas as pd
import numbers

    
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

        if(length == 0):
            unpacked[k] = x[start]
        else:
            unpacked[k] = np.array(x[start:start+length])
    
    return unpacked   