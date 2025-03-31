import os

from calc.rp_model.rp_model.env import RP_MODEL_FILE_PATH


def from_files_directory(path):
    return os.path.join(RP_MODEL_FILE_PATH, path)
