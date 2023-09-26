import pandas as pd
from types import SimpleNamespace 

game = SimpleNamespace()
game.natures    = SimpleNamespace()
game.mainskills = SimpleNamespace()
game.subskills  = SimpleNamespace()
game.pokedex    = SimpleNamespace()

game.natures.soh_effect = 0.1
game.natures.ing_effect = 0.2
game.natures.msc_effect = 0.2
game.natures.energy_effect = 0.08

game.subskills.help_s_effect = 0.07
game.subskills.help_m_effect = 0.14
game.subskills.ing_s_effect = 0.18
game.subskills.ing_m_effect = 0.36
game.subskills.trigger_s_effect = 0.18
game.subskills.trigger_m_effect = 0.36

game.data_files = {
    "natures" :   './data/natures.pickle',
    "subskills":  './data/subskills.pickle',
    "mainskills": './data/mainskills.pickle',
    "pokedex":    './data/pokedex.pickle',
}

game.natures.data    = pd.read_pickle(game.data_files["natures"])
game.mainskills.data = pd.read_pickle(game.data_files["mainskills"])
game.subskills.data  = pd.read_pickle(game.data_files["subskills"])
game.pokedex.data    = pd.read_pickle(game.data_files["pokedex"])

game.subskills.bonus_names = game.subskills.data["Subskill"][ game.subskills.data["RP Bonus Estimate"] >0 ].tolist()