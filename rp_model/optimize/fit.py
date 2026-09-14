from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

from .typedef import OptimizerFitContext
from ..calc import compute_rp
from ..enum import RpFitResult
from ..type import LastFitData
from ..utils import remove_nan


def get_rp_fit_result(
    rp_diff_clean: npt.NDArray[np.float64], /, *, optimal_fit_count: int,
) -> RpFitResult:
    if not rp_diff_clean.any():
        return RpFitResult.PERFECT if optimal_fit_count == 1 else RpFitResult.SUBOPTIMAL

    if rp_diff_clean.min() >= -1 and rp_diff_clean.max() <= 1:
        avg = abs(rp_diff_clean).mean()

        # Count sparse rounding errors as suboptimal.
        if avg < 1 / 30:
            return RpFitResult.SUBOPTIMAL

    return RpFitResult.FAILED


def get_rp_diff(
    context: OptimizerFitContext,
    reference_rp: npt.NDArray[np.float64],
    computed: SimpleNamespace,
    fit_data: LastFitData,
) -> npt.NDArray[np.float64]:
    return remove_nan(reference_rp - compute_rp(
        context.x0,
        context.pokemon_data_of_group,
        computed,
        context.unpack_info,
        fit=fit_data,
    ))
