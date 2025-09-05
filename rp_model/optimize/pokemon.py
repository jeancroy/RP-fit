from numpy import float64

from .fit import get_rate_combo_fit_result
from .traverse.bfs import traverse_last_fit_bfs
from .traverse.dfs import traverse_last_fit_dfs
from .typedef import OptimizerFitContext, OptimizerSingleFitResult, OptimizerSolvedDataEntry
from ..calc import make_precomputed_columns
from ..enum import RpFitResult
from ..env import RP_MODEL_IS_GLOBAL_CHECK
from ..type import LastFitData


def get_solution_of_pokemon(
    context: OptimizerFitContext,
    single_mon_fit_results: set[OptimizerSingleFitResult],
):
    # Variable use shortcut
    print_func = context.print_func
    pokemon_name = context.pokemon_name

    solutions_found: list[OptimizerSingleFitResult] = list(filter(
        lambda x: x.result.is_possible_fit,
        single_mon_fit_results
    ))
    solutions_formatted = [f"{solution.fit} ({solution.result.name[:1]})" for solution in solutions_found]
    print_func(f"{pokemon_name:>25} - {len(solutions_found)} solutions found - {" / ".join(solutions_formatted)}")
    if len(solutions_found) > 1:
        print_func(f"WARNING - Multiple solutions found for [{pokemon_name}]")

    try:
        fit_result_to_use_for_mon = sorted(single_mon_fit_results, key=lambda x: x.result.value, reverse=True)[0]

        if fit_result_to_use_for_mon.result != RpFitResult.PERFECT:
            imperfect = fit_result_to_use_for_mon

            print_func(
                f"WARNING - Imperfect solution used for [{pokemon_name}] - "
                f"Ingredient: {imperfect.fit.ing:>6.2%} / Skill: {imperfect.fit.skl:>6.2%}"
            )

        return OptimizerSolvedDataEntry(
            fit=fit_result_to_use_for_mon.fit,
            result=RpFitResult.SUBOPTIMAL if len(solutions_found) > 1 else fit_result_to_use_for_mon.result,
            pokemon=pokemon_name,
            data_count=context.pokemon_data_count,
        )
    except IndexError:
        print_func(f"WARNING - No solution found for [{pokemon_name}], default is used")

        return OptimizerSolvedDataEntry(
            fit=LastFitData.default(),
            result=RpFitResult.FAILED,
            pokemon=pokemon_name,
            data_count=context.pokemon_data_count,
        )


def process_pokemon(
    last_fit: LastFitData | None,
    context: OptimizerFitContext,
) -> OptimizerSolvedDataEntry:
    last_fit_of_pokemon = last_fit or LastFitData.default()

    computed = make_precomputed_columns(context.pokemon_data_of_group)
    reference_rp = context.pokemon_data_of_group["RP"].astype(float64).to_numpy()

    single_mon_fit_results: set[OptimizerSingleFitResult] = set()

    # DFS with multiple starting points spawned by BFS
    for centroid in traverse_last_fit_bfs(last_fit_of_pokemon, point_gap=0.015):
        single_fit_result = traverse_last_fit_dfs(centroid, context, reference_rp, computed)
        single_mon_fit_results.add(single_fit_result)

        if RP_MODEL_IS_GLOBAL_CHECK or single_fit_result.result != RpFitResult.PERFECT:
            # Keep recording fits if the result is not failed or is checking globally
            continue

        # Not global check OR Got a single perfect result, only store it and break the loop
        single_mon_fit_results = {single_fit_result}
        break

    # Search with BFS, only if DFS not finding anything or no perfect result
    if not any(result.result == RpFitResult.PERFECT for result in single_mon_fit_results):
        context.print_func(
            f"{context.pokemon_name:<25} - "
            f"DFS not finding any solution, switch to BFS... (Global: {RP_MODEL_IS_GLOBAL_CHECK})"
        )
        for idx, fit_data in enumerate(traverse_last_fit_bfs(last_fit_of_pokemon)):
            single_fit_result = get_rate_combo_fit_result(
                context,
                fit_data,
                reference_rp,
                computed,
                initiator="Solve",
                idx=idx,
            )

            single_mon_fit_results.add(single_fit_result)
            if RP_MODEL_IS_GLOBAL_CHECK or single_fit_result.result != RpFitResult.PERFECT:
                # Keep recording fits if the result is not failed or is checking globally
                continue

            # Not global check OR Got a single perfect result, only store it and break the loop
            single_mon_fit_results = {single_fit_result}
            break

    return get_solution_of_pokemon(context, single_mon_fit_results)
