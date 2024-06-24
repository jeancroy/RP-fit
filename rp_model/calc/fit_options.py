from types import SimpleNamespace

from ..files import get_files_directory, from_files_directory

# print(f"RP model file path: {get_files_directory()}")


class FitOptions:
    data_file = from_files_directory("data/rp-data.pickle")
    result_file = from_files_directory("results/least-squares-fit.pickle")
    boostrap_file = from_files_directory("results/bootstrap-fit.pickle")

    rp_file_id = "1kBrPl0pdAO8gjOf_NrTgAPseFtqQA27fdfEbMBBeAhs"

    rp_sheet_ids = {
        "data_1_9": "1682088244",
        "data_10_49": "1691041080",
        "data_50_74": "161092121",
        "pokedex": "513440248",
        "main_skill": "1395455629",
    }

    least_squares_kwargs = dict(loss="huber", xtol=None, verbose=2, max_nfev=200, f_scale=10)

