from typing import TypedDict

from ..enum import RpFitResult


class OptimizerSingleFitResult(TypedDict):
    ing: float
    skl: float
    result: RpFitResult


class OptimizerSolvedDataEntry(TypedDict, OptimizerSingleFitResult):
    pokemon: str
