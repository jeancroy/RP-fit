import pandas as pd

from .fit_options import FitOptions
from .game import game
from ..utils import download_sheet


# %%
def download_data():
    data_1_9 = download_sheet(FitOptions.rp_file_id, FitOptions.rp_sheet_ids["data_1_9"])
    data_10_49 = download_sheet(FitOptions.rp_file_id, FitOptions.rp_sheet_ids["data_10_49"])
    data_50_74 = download_sheet(FitOptions.rp_file_id, FitOptions.rp_sheet_ids["data_50_74"])

    # ugly patch, sheet 1-9 miss that column, because there's no skill
    data_1_9["MiscMult"] = data_1_9["NrgNat"]

    df = pd.concat([data_1_9, data_10_49, data_50_74], axis=0)
    df.dropna(subset=["Pokemon", "Level", "RP", "Nature", "MS lvl"], inplace=True)
    df.fillna(
        {'Amnt': 0, 'Ing2P': 0, 'Help skill bonus': 1, 'RP Multiplier': 1, 'ModelRP': -1, 'Difference': -1},
        inplace=True
    )
    df.fillna(
        {'Sub Skill 1': '', 'Sub Skill 2': '', 'Sub Skill 3': '', 'Ingredient 2': '', 'Source': ''},
        inplace=True
    )
    df.fillna(
        {'MiscMult': 1.0, 'HelpSub': 1.0, 'IngrSub': 1.0, 'SkillSub': 1.0},
        inplace=True
    )

    # data above 30 requires a 2nd ingredient to be valid.
    df.drop(df.index[(df["Level"] >= 30) & (df["Amnt"] == 0.0)], inplace=True)

    # data below 30 we clear 2nd ingredient
    df.loc[df["Level"] < 30, "Amnt"] = 0.0
    df.loc[df["Level"] < 30, "Ing2P"] = 0.0
    df.loc[df["Level"] < 30, "Ingredient 2"] = ""

    # data below 50 we clear 3rd skill.
    # data below 25 we clear 2nd skill.
    # data below 10 we clear the 1st.
    df.loc[df["Level"] < 50, "Sub Skill 3"] = ""
    df.loc[df["Level"] < 25, "Sub Skill 2"] = ""
    df.loc[df["Level"] < 10, "Sub Skill 1"] = ""

    # avoid a bug in RP of freshly caught mon with skill up unlocked.
    # (We now trust the bugged data is quarantined so we can use the valid data)
    # data = data[ ~( (data["Sub Skill 1"] == "Skill Level Up S") & (data["MS lvl"] == 2) )]
    # data = data[ ~( (data["Sub Skill 1"] == "Skill Level Up M") & (data["MS lvl"] == 3) )]

    # only use data known to the Pokedex (we'll update Pokedex as needed)
    df.drop(df.index[~df["Pokemon"].isin(game.data.pokedex["Pokemon"])], inplace=True)

    return df


def refresh_pokedex():
    pokedex = download_sheet(FitOptions.rp_file_id, FitOptions.rp_sheet_ids["pokedex"])

    pokedex = pokedex.fillna(0)
    pokedex.to_pickle(game.data_files.pokedex)
    game.refresh_loaded_data()


# def refresh_main_skill():
#     main_skills = download_sheet(FitOptions.rp_file_id, FitOptions.rp_sheet_ids["main_skill"])
#
#     main_skills = main_skills[["Main skill", "Value lvl 1"]]
#     main_skills = main_skills.rename(columns={
#         "Main skill": "Skill",
#         "Value lvl 1": "Value",
#     })
#
#     main_skills.to_pickle(game.data_files.mainskills)
#     game.refresh_loaded_data()
