from types import SimpleNamespace

fit_options = SimpleNamespace()
fit_options.result_file = lambda hash_id: f"./results/least-squares-fit-{hash_id}.pickle"
fit_options.data_file = "./data/rp-data.pickle"
fit_options.least_squares_kwargs = dict(loss="soft_l1", xtol=None, verbose=2)

