import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Hashable

from pandas import DataFrame

from .pokemon import process_pokemon
from .typedef import OptimizerFitContext, OptimizerSolvedDataEntry
from ..env import is_pokemon_included_for_rp_model
from ..type import LastFitData

# Global print lock
print_lock = threading.Lock()


def thread_safe_print(message: str) -> None:
    with print_lock:
        print(message)


def run_optimizer(
    data: DataFrame,
    last_fit_dict: dict[Hashable, LastFitData],
    x0,
    unpack_info,
) -> set[OptimizerSolvedDataEntry]:
    # Group data by Pokemon and filter out Pokemon that should be included
    pokemon_groups = [
        (name, group) for name, group in data.groupby("Pokemon")
        if is_pokemon_included_for_rp_model(name)
    ]

    solved_data: set[OptimizerSolvedDataEntry] = set()

    with ProcessPoolExecutor() as executor:
        future_to_pokemon = {
            executor.submit(
                process_pokemon,
                last_fit_dict,
                OptimizerFitContext(
                    x0=x0,
                    unpack_info=unpack_info,
                    pokemon_name=pokemon_name,
                    pokemon_data_of_group=pokemon_data_of_group,
                    print_func=thread_safe_print,
                ),
            )
            for pokemon_name, pokemon_data_of_group in pokemon_groups
        }

        for future in as_completed(future_to_pokemon):
            solved_data.add(future.result())

    print(f"{"=" * 25} Final Results {"=" * 25}")
    for solution in sorted(solved_data, key=lambda x: x.pokemon):
        print(f"{solution.pokemon:>25} - {solution.fit} ({solution.result.name})")

    return solved_data
