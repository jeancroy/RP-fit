import pandas as pd
from utils.files import download_sheet
from model.game import game
from model.fit_options import fit_options


# %%
def download_data():
    data_1_9 = download_sheet(fit_options.rp_file_id, fit_options.rp_sheet_ids["data_1_9"])
    data_10_49 = download_sheet(fit_options.rp_file_id, fit_options.rp_sheet_ids["data_10_49"])

    df = pd.concat([data_1_9, data_10_49], axis=0)
    df.dropna(subset=["Pokemon", "Level", "RP", "Nature", "MS lvl"], inplace=True)
    df.fillna({'Amnt': 0, 'Ing2P': 0, 'Help skill bonus': 1, 'RP Multiplier': 1, 'ModelRP': -1, 'Difference': -1},
              inplace=True)
    df.fillna({'Sub Skill 1': '', 'Sub Skill 2': '', 'Ingredient 2': '', 'Source': ''}, inplace=True)

    # data above 30 requires a 2nd ingredient to be valid.
    df.drop(df.index[(df["Level"] >= 30) & (df["Amnt"] == 0.0)], inplace=True)

    # data below 30 we clear 2nd ingredient
    df.loc[df["Level"] < 30, "Amnt"] = 0.0
    df.loc[df["Level"] < 30, "Ing2P"] = 0.0
    df.loc[df["Level"] < 30, "Ingredient 2"] = ""

    # data below 25 we clear 2nd skill, and below 10 we clear the 1st
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

    pokedex = download_sheet(fit_options.rp_file_id, fit_options.rp_sheet_ids["pokedex"])

    pokedex = pokedex.fillna(0)
    pokedex.to_pickle(game.data_files.pokedex)
    game.refresh_loaded_data()



