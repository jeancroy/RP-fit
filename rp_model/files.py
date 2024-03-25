import os


class PathOptions:
    FilePathEnvKey = "RP_MODEL_FILE_PATH"
    FilePath = os.path.abspath("./files")


def refresh_files_directory():
    PathOptions.FilePath = os.path.abspath(
        os.environ.get(PathOptions.FilePathEnvKey, PathOptions.FilePath)
    )


def from_files_directory(path):
    return os.path.join(PathOptions.FilePath, path)


def get_files_directory():
    return PathOptions.FilePath


def set_files_directory(path):
    path = os.path.abspath(path)
    os.environ.setdefault(PathOptions.FilePathEnvKey, path)
    PathOptions.FilePath = path


refresh_files_directory()
