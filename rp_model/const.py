import os

_MODEL_FILE_PATH = os.path.abspath(os.environ.get("RP_MODEL_FILE_PATH", "./files"))

print(f"RP model file path: {_MODEL_FILE_PATH}")


def get_files_directory():
    return _MODEL_FILE_PATH
