import numpy as np
import pandas as pd

import pickle
import requests
from io import StringIO, BytesIO
from os import path, listdir
import re
import numbers

from tabulate import tabulate
from IPython.display import display
    
def download_sheet(file_id, sheet_id):
    r = requests.get(f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&id={file_id}&gid={sheet_id}')
    df = pd.read_csv(BytesIO(r.content), thousands=',')
    return df


def lookup_table( serie1, table2, table2_lookup_column, table2_return_column ):
    return serie1.map(table2.set_index(table2_lookup_column)[table2_return_column])

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

