from model.fit_options import fit_options
from model.data import download_data, refresh_pokedex
from model.initial_guess import make_initial_guess

from utils.options import merge_dict_like
from utils.files import load, save, isfile
from utils.hash import digest
from utils.variables import pack

import os.path as path
from ipynb_convert import ipynb_to_py


# This is a work in progress
# The idea is to be able to launch the fit without jupyter
# and store result at an arbitrary location

class FitResultStorage:

    def __init__(self, options):
        self.get_fit_location = options.get_result_file

    # Read from the final storage location
    def try_get(self, hashid):
        location = self.get_fit_location(hashid)
        return None if not isfile(location) else load(location)

    # Write to final storage location
    def put(self, hashid, data):
        save(self.get_fit_location(hashid), data)

    # We get return value from the script as a file.
    def get_script_output(self, hashid):
        location = self.get_fit_location(hashid)
        return None if not isfile(location) else load(location)


def launch_fit(fit_storage=None, options=None):
    opt = merge_dict_like(fit_options, options)

    if (fit_storage is None):
        fit_storage = FitResultStorage(opt)

    refresh_pokedex()  # todo
    data = download_data()
    x0, unpack_info = pack(*make_initial_guess())
    hashid = digest(data, x0)

    fit = fit_storage.try_get(hashid)

    if (fit is None):

        launch_fit_as_script()
        fit = fit_storage.get_script_output(hashid)
        fit_storage.put(hashid, fit)

        # in our sample api usage,
        # merging the settings does most of the heavy lifting.


    else:
        print("Loaded from cache")

    return fit


def launch_fit_as_script():
    # This seems to be the recommended way to run another script??
    import converted.rp_fit
    pass


def path_from_this_file(p):
    return path.join(path.dirname(__file__), p)


def main():
    ipynb_to_py("RP fit.ipynb", "converted/rp_fit.py")

    overwrites = dict(
        result_file_pattern=path_from_this_file("./converted/saves/some-filename-{hash_id}.pickle"),
        rp_file_id="1kBrPl0pdAO8gjOf_NrTgAPseFtqQA27fdfEbMBBeAhs",
        rp_sheet_ids=({
            "data_1_9": "1682088244",
            "data_10_49": "1691041080",
        }),
    )

    launch_fit(options=overwrites)


if __name__ == "__main__":
    main()
