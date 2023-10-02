from types import SimpleNamespace
from utils.soft_round import soft_round_options

fit_options = SimpleNamespace()
fit_options.result_file = lambda hash_id: f"./results/least-squares-fit-{hash_id}.pickle"
fit_options.data_file = "./data/rp-data.pickle"
fit_options.least_squares_kwargs = dict(loss="huber", xtol=None, verbose=2, max_nfev=200, f_scale=10)
fit_options.soft_round = soft_round_options
