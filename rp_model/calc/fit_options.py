from ..files import from_files_directory


class FitOptions:
    data_file = from_files_directory("data/rp-data.pickle")
    boostrap_file = from_files_directory("results/bootstrap-fit.pickle")

    rp_file_id = "1dSsvWe49DzjK6dDsAMRB-okcw9DtJrdAXQ1dHBv_UNI"

    rp_sheet_ids = {
        "data_1_9": "1682088244",
        "data_10_49": "1691041080",
        "data_50_74": "161092121",
        "legacy": "2047819558",
        "pokedex": "513440248",
        "main_skill": "1395455629",
    }

    @staticmethod
    def get_pokemon_result_file(pokemon_name: str):
        return from_files_directory(f"results/pokemon/{pokemon_name}.pickle")
