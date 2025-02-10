from enum import Enum, auto


class RpFitResult(Enum):
    # Value represents priority - fit result with the highest priority should be used
    FAILED = 1
    SUBOPTIMAL = 2
    PERFECT = 3

    @property
    def is_possible_fit(self) -> bool:
        return self == RpFitResult.SUBOPTIMAL or self == RpFitResult.PERFECT
