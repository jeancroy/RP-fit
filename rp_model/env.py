import os

from environs import Env

env = Env()
env.read_env()

RP_MODEL_SELECTED_POKEMON_LIST: list[str] | None = env.list("RP_MODEL_POKEMON", None)

RP_MODEL_IS_GLOBAL_CHECK: bool = env.bool("RP_MODEL_IS_GLOBAL_CHECK", False)

RP_MODEL_FILE_PATH: str = os.path.abspath(env.str("RP_MODEL_FILE_PATH", "./files"))

def is_pokemon_included_for_rp_model(pokemon_en_name: str) -> bool:
    if RP_MODEL_SELECTED_POKEMON_LIST is None:
        return True

    return pokemon_en_name in RP_MODEL_SELECTED_POKEMON_LIST
