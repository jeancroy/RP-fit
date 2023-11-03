from types import SimpleNamespace
from utils.rounding import soft_round_options, RoundApprox
from os import path


def path_from_this_file(p):
    return path.join(path.dirname(__file__), p)


class FitOptions():

    data_file = path_from_this_file("../data/rp-data.pickle")
    result_file_pattern = path_from_this_file("../results/least-squares-fit-{hash_id}.pickle")

    def get_result_file(self, hash_id):
        return self.result_file_pattern.replace("{hash_id}", hash_id)

    rp_file_id = "1kBrPl0pdAO8gjOf_NrTgAPseFtqQA27fdfEbMBBeAhs"

    rp_sheet_ids = {
        "data_1_9": "1682088244",
        "data_10_49": "1691041080",
    }

    least_squares_kwargs = dict(loss='huber', xtol=None, verbose=2, max_nfev=200, f_scale=10)

    rounding = SimpleNamespace(
        bonus=RoundApprox.Exact,
        period=RoundApprox.Pass,
        components=RoundApprox.Pass,
        final_rp=RoundApprox.Pass,
    )

    soft_round = soft_round_options


# Main export
fit_options = FitOptions()