import os


class PathOptions:
    FilePathEnvKey = "RP_MODEL_FILE_PATH"
    FilePath = "./files"


def refresh_files_directory():
    PathOptions.FilePath = os.path.abspath(
        os.environ.get(PathOptions.FilePathEnvKey, PathOptions.FilePath)
    )


def set_files_directory(path):
    os.environ.setdefault(PathOptions.FilePathEnvKey, path)
    PathOptions.FilePath = path


def get_files_directory():
    return PathOptions.FilePath


refresh_files_directory()
