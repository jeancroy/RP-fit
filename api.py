import pandas as pd
import scipy

from .rp_model.calc import (
    FitOptions, compute_rp, download_data, game, make_initial_guess, make_precomputed_columns,
    refresh_pokedex,
)
from .rp_model.files import from_files_directory
from .rp_model.utils import DataStore, pack, simplify_opt_result, table, unpack


def update_fit_cached():
    refresh_pokedex()
    # refresh_main_skill()

    data = download_data()
    data.to_pickle(FitOptions.data_file)

    x0, unpack_info = pack(*make_initial_guess())

    store = (DataStore()
             .with_dependency_on(data, x0)
             .try_read_and_validate(FitOptions.result_file)
             )

    if not store.is_valid():

        print(f"RP model pickle hash mismatch, generating new file...")
        opt = run_optimizer(data, x0, unpack_info)
        store.use_data(opt).save_to(FitOptions.result_file)

    else:
        opt = store.data()
        print("RP Model pickle loaded from cache")

    sol = unpack(opt.x, unpack_info)

    result = pd.DataFrame({
        "pokemon": game.data.pokedex["Pokemon"],
        "pokemonId": game.data.pokedex["Pokemon ID"],
        "ingredientSplit": sol["Pokemons ing fractions"],
        "skillValue": sol["Pokemons skill chances"]
        # TODO
        # `error`: A python dict with the keys of ingredient and skill. Sample value: 0.1022 (conf (xxx)*) in the
        # bootstrap one
        # `skillPercent`: Only needed if there's a field for the calc to consume skill value and calculate skill %.
        # Sample value: 6.512
        # > `skillPercent` can be skipped if the model doesn't take anything else or do computations.
        # > The scraper can handle it with single value of "skill value"
    })

    # Merge with result count

    result = result.set_index("pokemon")
    result["dataCount"] = data.groupby(["Pokemon"]).size()

    # If there's no count, Panda uses NaN, and therefore casts everything to double.
    # So we now undo that.
    result["dataCount"] = result["dataCount"].fillna(0).astype(int)

    return result


def run_optimizer(data, x0, unpack_info):
    computed = make_precomputed_columns(data)
    reference_rp = data["RP"].to_numpy()

    def residual(x):
        return reference_rp - compute_rp(x, data, computed, unpack_info)

    opt = scipy.optimize.least_squares(residual, x0, **FitOptions.least_squares_kwargs)

    return simplify_opt_result(opt)


def get_rp_model_result(result_file: str):
    """
    The only method that is called by the scraper.

    Example ``file_pickle_pattern``: ``"results/least-squares-fit.pickle"``.

    :param result_file: The path pattern to the result pickle file.
    :return: The resulting ``pd.DataFrame``.
    """

    FitOptions.result_file = from_files_directory(result_file)

    return update_fit_cached()


def main():

    sol = update_fit_cached()
    table(sol)


if __name__ == "__main__":
    main()
