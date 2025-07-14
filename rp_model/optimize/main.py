import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from pandas import DataFrame

from .pokemon import process_pokemon
from .typedef import OptimizerSolvedDataEntry
from ..env import is_pokemon_included_for_rp_model
from ..type import LastFitData

# Global print lock
print_lock = threading.Lock()


def thread_safe_print(message: str) -> None:
    with print_lock:
        print(message)


def run_optimizer(
    data: DataFrame,
    last_fit_dict: dict[str, LastFitData],
    x0,
    unpack_info,
) -> list[OptimizerSolvedDataEntry]:
    # Group data by Pokemon and filter out Pokemon that should be included
    pokemon_groups = [
        (name, group) for name, group in data.groupby("Pokemon")
        if is_pokemon_included_for_rp_model(name)
    ]

    solved_data: list[OptimizerSolvedDataEntry] = []

    with ThreadPoolExecutor() as executor:
        future_to_pokemon = {
            executor.submit(
                process_pokemon,
                last_fit_dict,
                x0,
                unpack_info,
                pokemon_name,
                pokemon_data_of_group,
                thread_safe_print
            )
            for pokemon_name, pokemon_data_of_group in pokemon_groups
        }

        for future in as_completed(future_to_pokemon):
            solved_data.append(future.result())

    print(f"{"=" * 25} Final Results {"=" * 25}")
    for solution in sorted(solved_data, key=lambda x: x["pokemon"]):
        print(
            f"{solution["pokemon"]:>25} - "
            f"[Ing] {solution["ing"]:6.2%} [Skl] {solution["skl"]:6.2%} ({solution["result"].name})"
        )

    return solved_data
