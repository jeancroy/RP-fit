import os.path
from types import SimpleNamespace

from ..const import get_files_directory
from ..utils import RoundApprox, soft_round_options

print(f"RP model file path: {get_files_directory()}")

class FitOptions:
    data_file = f"{get_files_directory()}/data/rp-data.pickle"
    result_file_pattern = "results/least-squares-fit-{hash_id}.pickle"

    @classmethod
    def get_result_file(cls, hash_id):
        return os.path.join(get_files_directory(), cls.result_file_pattern.replace('{hash_id}', hash_id))

    rp_file_id = "1kBrPl0pdAO8gjOf_NrTgAPseFtqQA27fdfEbMBBeAhs"

    rp_sheet_ids = {
        "data_1_9": "1682088244",
        "data_10_49": "1691041080",
        "data_50_74": "161092121",
        "pokedex": "513440248",
    }

    least_squares_kwargs = dict(loss="huber", xtol=None, verbose=2, max_nfev=200, f_scale=10)

    rounding = SimpleNamespace(
        bonus=RoundApprox.Exact,
        period=RoundApprox.Pass,
        components=RoundApprox.Pass,
        final_rp=RoundApprox.Pass,
    )

    soft_round = soft_round_options
