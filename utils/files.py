import pickle
import requests
from io import BytesIO
import os
import re

import pandas as pd


def download_sheet(file_id, sheet_id):
    r = requests.get(f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&id={file_id}&gid={sheet_id}')
    df = pd.read_csv(BytesIO(r.content), thousands=',')
    return df


# Pickle file management

def save(filepath, data):
    with open(filepath, "wb") as handle:
        pickle.dump(data, handle)


def load(filepath):
    with open(filepath, "rb") as handle:
        return pickle.load(handle)


def isfile(filepath):
    return os.path.isfile(filepath)


def last_modified_file_with_pattern(search_path, pattern_str):
    pattern = re.compile(pattern_str)
    files = filter(os.path.isfile, os.listdir(search_path))
    files = [os.path.join(search_path, f) for f in files if pattern.search(f)]
    if (len(files) == 0):
        return None
    if (len(files) == 1):
        return files[0]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]
