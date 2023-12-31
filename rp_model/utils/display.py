import sys

import numpy as np
import pandas as pd
from IPython.display import display
from tabulate import tabulate


# Display anything as a table
def table(data, **kwargs):
    if isinstance(data, (list, pd.Series, pd.DataFrame, pd.Index)):
        if in_notebook():
            display(tabulate(data, tablefmt='html', **kwargs))
        else:
            print(tabulate(data, tablefmt='plain', **kwargs))
    else:
        table(
            [
                ([k] + ([np.array2string(v, threshold=10)] if isinstance(v, np.ndarray) else [str(v)]))
                for k, v in list_members(data)
            ],
            **kwargs
        )


def in_notebook():
    return 'ipykernel' in sys.modules


def list_members(obj):
    if isinstance(obj, dict):
        return list(obj.items())

    return [
        (a, getattr(obj, a)) for a in dir(obj)
        if not a.startswith("_") and not callable(getattr(obj, a))
    ]
