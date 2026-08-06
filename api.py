import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from pandas import DataFrame

from .rp_model.calc import (
    FitOptions, download_data, game, make_initial_guess, refresh_ing_growth,
    refresh_pokedex,
)
from .rp_model.enum import RpFitResult
from .rp_model.env import RP_MODEL_IS_GLOBAL_CHECK, is_pokemon_included_for_rp_model
from .rp_model.optimize.check import is_last_fit_perfect
from .rp_model.optimize.pokemon import process_pokemon
from .rp_model.optimize.typedef import OptimizerFitContext, OptimizerSolvedDataEntry
from .rp_model.type import LastFitData
from .rp_model.utils import DataStore, pack, table
from .rp_model.utils.thread_safe_print import safe_print, thread_safe_print


@dataclass
class RpModelFitResult:
    raw_data: DataFrame
    fit_result: DataFrame


def print_final_result_entry(entry: tuple[bool, OptimizerSolvedDataEntry]):
    is_cached, solution = entry

    safe_print(
        f"[{"C" if is_cached else "N"}] {solution.pokemon:>25} ({solution.data_count:>3}) - "
        f"{solution.fit} ({solution.result.name})"
    )


def print_final_results(entries: list[tuple[bool, OptimizerSolvedDataEntry]]):
    sorted_entries = sorted(entries, key=lambda x: x[1].pokemon)

    safe_print(f"{"=" * 25} Final Results {"=" * 25}")
    for entry in sorted_entries:
        print_final_result_entry(entry)

    imperfect_entries = [x for x in sorted_entries if x[1].result != RpFitResult.PERFECT]
    if imperfect_entries:
        safe_print(f"{"=" * 25} Imperfect Results {"=" * 25}")
        for entry in imperfect_entries:
            print_final_result_entry(entry)

    recalc_entries = [x for x in sorted_entries if not x[0]]
    if recalc_entries:
        safe_print(f"{"=" * 25} Recalculated Results {"=" * 25}")
        for entry in recalc_entries:
            print_final_result_entry(entry)


# The return `bool` indicates whether the result was cached or recalculated.
def use_or_refresh_solved_pokemon(
    pokemon_name: str,
    data_of_pokemon: DataFrame,
    last_fit: LastFitData | None,
    x0,
    unpack_info,
) -> tuple[bool, OptimizerSolvedDataEntry]:
    cache_path = FitOptions.get_pokemon_result_file(pokemon_name)

    store = (
        DataStore(cache_path)
        # Cache by IDs and last fit result
        .with_dependency_on(data_of_pokemon["ID"], last_fit)
        .try_read_and_validate()
    )

    is_valid_store = store.is_valid()
    is_last_fit_passed = is_last_fit_perfect(pokemon_name, data_of_pokemon, last_fit, x0, unpack_info)

    if not is_last_fit_passed:
        if is_valid_store:
            thread_safe_print(f"{pokemon_name:>25} - No data update, but the last fit was not perfect")
        else:
            thread_safe_print(f"{pokemon_name:>25} - Recalculating, data updated while last fit failed")

    if is_valid_store and not RP_MODEL_IS_GLOBAL_CHECK:
        return True, store.data()

    solved_data = process_pokemon(
        last_fit,
        OptimizerFitContext(
            x0=x0,
            unpack_info=unpack_info,
            pokemon_name=pokemon_name,
            pokemon_data_of_group=data_of_pokemon,
            print_func=thread_safe_print,
        )
    )
    store.use_data(solved_data).save_to_path()

    return False, solved_data


def process_pokemon_batch(
    pokemon_batch: list[tuple[str, DataFrame]],
    last_fit: dict,
    x0,
    unpack_info
) -> list[tuple[bool, OptimizerSolvedDataEntry]]:
    results = []
    for name, group in pokemon_batch:
        if not is_pokemon_included_for_rp_model(name):
            continue

        result = use_or_refresh_solved_pokemon(name, group, last_fit.get(name), x0, unpack_info)
        results.append(result)

    return results


def update_fit_cached() -> RpModelFitResult:
    refresh_pokedex()
    refresh_ing_growth()

    data = download_data()
    data.to_pickle(FitOptions.data_file)

    initial_guess, range_info, last_fit = make_initial_guess(include_last_fit_dict=True)
    x0, unpack_info = pack(initial_guess, range_info)

    # Prepare Pokemon groups for parallel processing
    pokemon_groups = [(name, df_group) for name, df_group in data.groupby("Pokemon")]

    num_workers = min(os.cpu_count() or 1, len(pokemon_groups))
    batch_size = max(1, len(pokemon_groups) // num_workers)

    pokemon_batches = []
    for i in range(0, len(pokemon_groups), batch_size):
        batch = pokemon_groups[i:i + batch_size]
        pokemon_batches.append(batch)

    solved: list[tuple[bool, OptimizerSolvedDataEntry]] = []

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(process_pokemon_batch, batch, last_fit, x0, unpack_info): batch
            for batch in pokemon_batches
        }

        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            solved.extend(batch_results)

    print_final_results(solved)

    solution_records = []
    for _, item in solved:
        record = asdict(item)
        record.pop("fit")

        record["ing"] = item.fit.ing
        record["skl"] = item.fit.skl
        solution_records.append(record)

    solution = DataFrame.from_records(solution_records, index="pokemon")
    result = (DataFrame({"pokemon": game.data.pokedex["Pokemon"], "pokemonId": game.data.pokedex["Pokemon ID"]})
              .join(solution, on="pokemon")
              .rename(columns={"ing": "ingredientSplit", "skl": "skillValue", "result": "fitResult"}))

    # Merge with result count
    result = result.set_index("pokemon")
    result["dataCount"] = data.groupby(["Pokemon"]).size()

    # If there's no count, Panda uses NaN, and therefore casts everything to double.
    # So we now undo that.
    result["dataCount"] = result["dataCount"].fillna(0).astype(int)

    return RpModelFitResult(raw_data=data, fit_result=result)


def get_rp_model_result() -> RpModelFitResult:
    return update_fit_cached()


def main():
    sol = update_fit_cached().fit_result
    table(sol)


if __name__ == "__main__":
    main()
