import numpy as np

from .fit import get_rp_diff, get_rp_fit_result
from .grid.main import solve_rp_grid
from .typedef import OptimizerFitContext, OptimizerSolvedDataEntry
from .validate import print_imperfect_fit_details
from ..calc import make_precomputed_columns
from ..enum import RpFitResult
from ..type import LastFitData


def _format_fit(fit: LastFitData) -> str:
    return f"[Ing {fit.ing:>6.2%} / Skl {fit.skl:>6.2%}]"


def process_pokemon(
    last_fit: LastFitData | None,
    context: OptimizerFitContext,
) -> OptimizerSolvedDataEntry:
    last_fit_of_pokemon = last_fit or LastFitData.default()
    computed = make_precomputed_columns(context.pokemon_data_of_group)
    reference_rp = context.pokemon_data_of_group["RP"].to_numpy(dtype=np.float64)
    validation_result = get_rp_fit_result(
        get_rp_diff(context, reference_rp, computed, last_fit_of_pokemon),
        lax=True,
    )
    if validation_result != RpFitResult.PERFECT:
        print_imperfect_fit_details(
            context,
            last_fit_of_pokemon,
            reference_rp,
            computed,
            initiator="Imperfect",
        )

    result = solve_rp_grid(context)
    if result.has_multiple_solutions:
        examples = " / ".join(_format_fit(fit) for fit in result.optimal_fits[:5])
        suffix = " / ..." if result.optimal_fit_count > 5 else ""
        context.print_func(
            f"Solve - {context.pokemon_name:<25} has {result.optimal_fit_count} global fits: "
            f"{examples}{suffix}"
        )
    if not result.is_exact:
        context.print_func(
            f"Solve - {context.pokemon_name:<25} has no exact fit; "
            f"using global minimum MSE {result.loss:g}"
        )
    context.print_func(
        f"Solve - [{result.fit_result.name}] RP fit of {context.pokemon_name:<25} found at: "
        f"Ing {result.fit.ing:>6.2%} / Skl {result.fit.skl:>6.2%}"
    )
    return OptimizerSolvedDataEntry(
        fit=result.fit,
        result=result.fit_result,
        pokemon=context.pokemon_name,
        data_count=context.pokemon_data_count,
        optimal_fit_count=result.optimal_fit_count,
        minimum_loss=result.loss,
        is_exact=result.is_exact,
    )
