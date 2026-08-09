from dataclasses import dataclass
from typing import Any, Callable

from pandas import DataFrame

from ..enum import RpFitResult
from ..type import LastFitData


@dataclass
class OptimizerSolvedDataEntry:
    fit: LastFitData
    result: RpFitResult
    pokemon: str
    data_count: int
    optimal_fit_count: int = 1
    minimum_loss: float = 0.0
    is_exact: bool = True


@dataclass
class OptimizerFitContext:
    x0: Any
    unpack_info: Any
    pokemon_name: str
    pokemon_data_of_group: DataFrame
    print_func: Callable[[str], None]

    @property
    def pokemon_data_count(self) -> int:
        return len(self.pokemon_data_of_group)
