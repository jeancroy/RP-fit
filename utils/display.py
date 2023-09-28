import numpy as np
import pandas as pd
import numbers

from tabulate import tabulate
from IPython.display import display

# Display anything as a table
def table(data):
    if (isinstance(data, (list, pd.Series, pd.DataFrame, pd.Index))):
        display(tabulate(data, tablefmt='html'))

    else:
        table([([k] + (
            [v] if isinstance(v, (numbers.Number, str)) else
            [v] if len(v) < 2 else
            [np.array2string(v, threshold=10)]
        )
                ) for k, v in _list_members(data)])


def _list_members(obj):
    if isinstance(obj, dict):
        return list(obj.items())

    return ([(a, getattr(obj, a))
             for a in dir(obj) if not a.startswith('_') and not callable(getattr(obj, a))
             ])
