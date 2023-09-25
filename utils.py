import pandas as pd

import pickle
import requests
from io import StringIO, BytesIO
from os import path, listdir
import re
import numbers

from tabulate import tabulate
from IPython.display import display

USE_AUTOGRAD = False

if USE_AUTOGRAD:
    import autograd.numpy as np 
    from autograd import grad, jacobian
    from autograd.builtins import isinstance, tuple
    
else:
    import numpy as np

    
def download_sheet(file_id, sheet_id):
    r = requests.get(f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&id={file_id}&gid={sheet_id}')
    df = pd.read_csv(BytesIO(r.content), thousands=',')
    return df

# Variables vector management 

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

# Display anything as a table

def table(data): 
    
    if( isinstance(data,(list,pd.Series,pd.DataFrame,pd.Index)) ):  
        display(tabulate(data,tablefmt='html'))
        
    else:
        table([( [k]+ (
                [v] if isinstance(v, (numbers.Number, str) ) else
                [v] if len(v) < 2 else
                [np.array2string(v, threshold=10)]
            )
        ) for k,v in list_members(data) ])
        
def list_members(obj): 

    if isinstance(obj, dict):
        return list(obj.items())

    return ([ (a, getattr(obj, a) )
              for a in dir(obj) if not a.startswith('_') and not callable(getattr(obj, a))
            ])

# Pickle file management

def save(filepath, data):
    with open(filepath,"xb") as handle:
        pickle.dump(data, handle)

def load(filepath):
    with open(filepath,"rb") as handle:
        return pickle.load(handle)
    
def last_modified_file_with_pattern(search_path, pattern_str):
    pattern = re.compile(pattern_str)
    files = filter(os.path.isfile, os.listdir(search_path))
    files = [os.path.join(searchdir, f) for f in files if pattern.search(f)]
    if(len(files)==0): return None
    if(len(files)==1): return files[0]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]