import numpy as np
import numpy.typing as npt


def remove_nan(array: npt.NDArray) -> npt.NDArray:
    return array[~np.isnan(array)]
