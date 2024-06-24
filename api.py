import scipy
import pandas as pd
from pandas import DataFrame
from dataclasses import dataclass
from dfols import solve as dfols_solve

from .rp_model.calc import (
    FitOptions, compute_rp, download_data, game, make_initial_guess, make_precomputed_columns,
    refresh_pokedex,
)
from .rp_model.files import from_files_directory
from .rp_model.utils import DataStore, pack, table, unpack


@dataclass
class RpModelFitResult:
    raw_data: DataFrame
    fit_result: DataFrame


def update_fit_cached() -> RpModelFitResult:
    refresh_pokedex()
    # refresh_main_skill()

    data = download_data()
    data.to_pickle(FitOptions.data_file)

    x0, unpack_info = pack(*make_initial_guess())

    store = (DataStore(FitOptions.result_file)
             .with_dependency_on(data, x0)
             .try_read_and_validate()
             )

    if not store.is_valid():
        print(f"RP model pickle hash mismatch, generating new file...")
        opt = run_optimizer(data, x0, unpack_info)
        store.use_data(opt).save_to_path()

    else:
        opt = store.data()
        print("RP Model pickle loaded from cache")

    sol = unpack(opt.x, unpack_info)

    result = pd.DataFrame({
        "pokemon": game.data.pokedex["Pokemon"],
        "pokemonId": game.data.pokedex["Pokemon ID"],
        "ingredientSplit": sol["Pokemons ing fractions"],
        "skillValue": sol["Pokemons skill chances"],
    })

    # Merge with result count

    result = result.set_index("pokemon")
    result["dataCount"] = data.groupby(["Pokemon"]).size()

    # If there's no count, Panda uses NaN, and therefore casts everything to double.
    # So we now undo that.
    result["dataCount"] = result["dataCount"].fillna(0).astype(int)

    return RpModelFitResult(raw_data=data, fit_result=result)


def run_optimizer(data, x0, unpack_info):
    computed = make_precomputed_columns(data)
    reference_rp = data["RP"].to_numpy()

    def residual(x):
        return reference_rp - compute_rp(x, data, computed, unpack_info)

    opt = dfols_solve(residual, x0, print_progress=True, maxfun=1500, rhoend=1.1e-4)

    return opt


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
