from enum import Enum
from types import SimpleNamespace
from utils.soft_round import soft_round_options


class RoundDirection(Enum):
    Floor = -1
    Middle = 0
    Ceil = 1


class RoundApprox(Enum):
    Pass = 0
    Exact = 1
    Soft = 2


fit_options = SimpleNamespace(
    data_file="./data/rp-data.pickle",
    result_file=lambda hash_id: f"./results/least-squares-fit-{hash_id}.pickle",
    least_squares_kwargs=dict(loss="huber", xtol=None, verbose=2, max_nfev=200, f_scale=10),
    soft_round=soft_round_options,
    progressive_soft_round_alpha=False,
    bonus_rounding=RoundApprox.Exact,
    rp_rounding=RoundApprox.Pass
)

# Option 1
# progressive_soft_round_alpha = True,
# bonus_rounding = RoundApprox.Soft,
# rp_rounding = RoundApprox.Soft

# Option 2
# progressive_soft_round_alpha = False,
# bonus_rounding = RoundApprox.Exact,
# rp_rounding = RoundApprox.Pass

