import os
import warnings

_MODEL_FILE_PATH = None


def set_files_directory(path: str, ignore_empty_path: bool = False):
    global _MODEL_FILE_PATH

    if not path:
        if not ignore_empty_path:
            warnings.warn(f"Path provided in `set_files_directory()` is falsy, not setting files path ({path})")
        return

    _MODEL_FILE_PATH = os.path.abspath(path)
    print(f"Configured files path to: {_MODEL_FILE_PATH}")


set_files_directory(os.environ.get("RP_MODEL_FILE_PATH"), ignore_empty_path=True)


def get_files_directory():
    global _MODEL_FILE_PATH

    if not _MODEL_FILE_PATH:
        set_files_directory(os.environ.get("RP_MODEL_FILE_PATH"))

    if not _MODEL_FILE_PATH:
        raise RuntimeError(
            "Call `set_files_directory()` in `rp_model.const` to set the files directory first, "
            "or set the environment variable `RP_MODEL_FILE_PATH` as the pickle files parent path"
        )

    return _MODEL_FILE_PATH
