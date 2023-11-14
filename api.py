import pandas as pd
import scipy

from .rp_model.const import set_files_directory

set_files_directory("./files")

from .rp_model.calc import (  # noqa: E402
    compute_rp, download_data, FitOptions, game, make_initial_guess, make_precomputed_columns, refresh_pokedex,
)
from .rp_model.utils import digest, pack, save, simplify_opt_result, try_load, unpack, table  # noqa: E402


def update_fit_cached():
    refresh_pokedex()
    data = download_data()
    data.to_pickle(FitOptions.data_file)

    x0, unpack_info = pack(*make_initial_guess())

    hashid = digest(data, x0)
    filepath = FitOptions.get_result_file(hashid)
    opt = try_load(filepath)

    if opt is None:
        opt = run_optimizer(data, x0, unpack_info)
        save(filepath, simplify_opt_result(opt))
    else:
        print("RP Model loaded from cache")

    sol = unpack(opt.x, unpack_info)

    result = pd.DataFrame({
        "pokemon": game.data.pokedex["Pokemon"],
        "pokemonId": game.data.pokedex["Pokemon ID"],
        "ingredientSplit": sol["Pokemons ing fractions"],
        "skillValue": sol["Pokemons skill products"]
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

    return opt


def get_rp_model_result(
    file_dir: str,
    result_pattern: str,
):
    """
    The only method that is being called by the scraper.

    ``file_pickle_pattern`` should contain ``{hash_id}`` for the hash ID of the result file.

    Example ``file_pickle_pattern``: ``"results/least-squares-fit-{hash_id}.pickle"``.

    :param file_dir: The directory that stores all data files.
    :param result_pattern: The path pattern to the result pickle file.
    :return: The resulting ``pd.DataFrame``.
    """
    set_files_directory(file_dir)

    FitOptions.result_file_pattern = result_pattern

    sol = update_fit_cached()
    table(sol)


def main():
    FitOptions.result_file_pattern = "results/least-squares-fit-{hash_id}.pickle"
    FitOptions.rp_file_id = "1kBrPl0pdAO8gjOf_NrTgAPseFtqQA27fdfEbMBBeAhs"

    sol = update_fit_cached()
    table(sol)


if __name__ == "__main__":
    main()
