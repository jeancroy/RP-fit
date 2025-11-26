from types import SimpleNamespace

import numpy as np

from .typedef import OptimizerFitContext
from ..calc import compute_rp
from ..type import LastFitData
from ..utils import remove_nan


def print_imperfect_fit_details(
    context: OptimizerFitContext,
    fit_data: LastFitData,
    reference_rp: np.ndarray,
    computed: SimpleNamespace,
    /,
    initiator: str = "Validate",
):
    # Recompute full computed RP for alignment with the original data
    computed_rp = remove_nan(compute_rp(
        context.x0,
        context.pokemon_data_of_group,
        computed,
        context.unpack_info,
        fit=fit_data,
    ))
    reference_rp = remove_nan(reference_rp)

    mismatched = reference_rp != computed_rp

    mismatch_count = np.count_nonzero(mismatched)
    total_count = reference_rp.size

    if not mismatch_count:
        return

    df = context.pokemon_data_of_group
    id_values = df["ID"].to_numpy()

    context.print_func(
        f"{initiator} - {context.pokemon_name:<25} using fit "
        f"[Ing {fit_data.ing:>6.2%} / Skl {fit_data.skl:>6.2%}]: "
        f"{mismatch_count} of {total_count} failed"
    )

    prefix = " " * (len(initiator) + 3)
    for row_id, ref_single, calc_single in zip(
        id_values[mismatched],
        reference_rp[mismatched],
        computed_rp[mismatched],
    ):
        context.print_func(f"{prefix}ID [{row_id}]: Ref RP {ref_single:>8} | Calc {calc_single:>8}")
