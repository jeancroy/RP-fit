from types import SimpleNamespace

import pandas as pd

from rp_model.const import get_files_directory


class GameData:
    natures = SimpleNamespace(
        soh_effect=0.1,
        ing_effect=0.2,
        msc_effect=0.2,
        energy_effect=0.08,
    )

    subskills = SimpleNamespace(
        help_s_effect=0.07,
        help_m_effect=0.14,
        ing_s_effect=0.18,
        ing_m_effect=0.36,
        trigger_s_effect=0.18,
        trigger_m_effect=0.36,
    )

    data_files = SimpleNamespace(
        natures=f"{get_files_directory()}/data/natures.pickle",
        subskills=f"{get_files_directory()}/data/subskills.pickle",
        mainskills=f"{get_files_directory()}/data/mainskills.pickle",
        pokedex=f"{get_files_directory()}/data/pokedex.pickle",
    )

    data = SimpleNamespace()

    def refresh_loaded_data(self):
        # read all the data files here
        self.data = SimpleNamespace(
            **dict([(k, pd.read_pickle(v)) for k, v in vars(self.data_files).items()])
        )

        # find which subskills are of the bonus type
        self.subskills.bonus_names = \
            self.data.subskills["Subskill"][self.data.subskills["RP Bonus Estimate"] > 0].tolist()


# Main export
game = GameData()
game.refresh_loaded_data()
