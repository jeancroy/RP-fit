from dataclasses import dataclass

from pandas import DataFrame

from .rp_model.calc import (
    FitOptions, download_data, game, make_initial_guess, refresh_pokedex,
)
from .rp_model.files import from_files_directory
from .rp_model.optimize import run_optimizer, is_all_last_fit_valid
from .rp_model.utils import DataStore, pack, table


@dataclass
class RpModelFitResult:
    raw_data: DataFrame
    fit_result: DataFrame


def update_fit_cached() -> RpModelFitResult:
    refresh_pokedex()
    # refresh_main_skill()

    data = download_data()
    data.to_pickle(FitOptions.data_file)

    # `compute_rp()` not supporting 3rd ingredient for now, therefore skipping mons with level >= 60
    data = data[data["Level"] < 60]

    initial_guess, range_info, last_fit = make_initial_guess(include_last_fit_dict=True)
    x0, unpack_info = pack(initial_guess, range_info)

    store = (DataStore(FitOptions.result_file)
             .with_dependency_on(data, x0)
             .try_read_and_validate())

    if not store.is_valid() or not is_all_last_fit_valid(data, last_fit, x0, unpack_info):
        print(f"RP model pickle hash mismatch, generating new file...")
        solved_data = run_optimizer(data, last_fit, x0, unpack_info)
        store.use_data(solved_data).save_to_path()
    else:
        print("RP model pickle loaded from cache")
        solved_data = store.data()

    solution = DataFrame.from_records(solved_data, index="pokemon")
    result = (DataFrame({"pokemon": game.data.pokedex["Pokemon"], "pokemonId": game.data.pokedex["Pokemon ID"]})
              .join(solution, on="pokemon")
              .rename(columns={"ing": "ingredientSplit", "skl": "skillValue"}))

    # Merge with result count
    result = result.set_index("pokemon")
    result["dataCount"] = data.groupby(["Pokemon"]).size()

    # If there's no count, Panda uses NaN, and therefore casts everything to double.
    # So we now undo that.
    result["dataCount"] = result["dataCount"].fillna(0).astype(int)

    return RpModelFitResult(raw_data=data, fit_result=result)


def get_rp_model_result(result_file: str) -> RpModelFitResult:
    """
    The only method that is called by the scraper.

    Example ``file_pickle_pattern``: ``"results/least-squares-fit.pickle"``.

    :param result_file: The path pattern to the result pickle file.
    :return: The resulting ``pd.DataFrame``.
    """

    FitOptions.result_file = from_files_directory(result_file)

    return update_fit_cached()


def main():
    sol = update_fit_cached().fit_result
    table(sol)


if __name__ == "__main__":
    main()
