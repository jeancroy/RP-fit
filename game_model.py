import pandas as pd
from types import SimpleNamespace

game = SimpleNamespace()

game.natures = SimpleNamespace(
    soh_effect=0.1,
    ing_effect=0.2,
    msc_effect=0.2,
    energy_effect=0.08,
)

game.subskills = SimpleNamespace(
    help_s_effect=0.07,
    help_m_effect=0.14,
    ing_s_effect=0.18,
    ing_m_effect=0.36,
    trigger_s_effect=0.18,
    trigger_m_effect=0.36,
)

game.data_files = SimpleNamespace(
    natures='./data/natures.pickle',
    subskills='./data/subskills.pickle',
    mainskills='./data/mainskills.pickle',
    pokedex='./data/pokedex.pickle',
)

# read all the data files here
game.data = SimpleNamespace(**dict([(k, pd.read_pickle(v)) for k, v in vars(game.data_files).items()]))

game.subskills.bonus_names = game.data.subskills["Subskill"][game.data.subskills["RP Bonus Estimate"] > 0].tolist()
