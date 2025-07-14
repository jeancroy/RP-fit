from typing import NamedTuple

from .const import ING_MAX, ING_MIN, SKILL_MAX, SKILL_MIN


class LastFitData(NamedTuple):
    ing: float
    skl: float

    def is_in_boundary(self) -> bool:
        return ING_MIN <= self.ing <= ING_MAX and SKILL_MIN <= self.skl <= SKILL_MAX

    @staticmethod
    def default():
        return LastFitData(ing=0.2, skl=0.02)

    def __str__(self) -> str:
        return f"[Ing] {self.ing:>6.1%} [Skl] {self.skl:>6.1%}"
