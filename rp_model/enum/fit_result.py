from enum import Enum, auto


class RpFitResult(Enum):
    FAILED = auto()
    SUBOPTIMAL = auto()
    PERFECT = auto()

    @property
    def is_possible_fit(self) -> bool:
        return self == RpFitResult.SUBOPTIMAL or self == RpFitResult.PERFECT
