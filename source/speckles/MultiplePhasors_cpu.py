import numpy as np
from Phasors_cpu import Phasors
import typing


class MultiplePhasors:

    def __init__(self, phasors: typing.Sequence[Phasors], where: typing.Sequence[np.ndarray]):
        if len(phasors) != len(where):
            raise ValueError("There should be as many phasors as group of indices in `where`.")
        self.__R = len(phasors)
        previous_shape = phasors[0].shape
        for p in phasors[1:]:
            current_shape = p.shape
            if p.shape != previous_shape:
                raise ValueError("All phasors should have the same shape.")
            previous_shape = current_shape
        self.__phasors = phasors
        self.__where = where
        self.__merged = None
        self.__shape = (self.__R,) + previous_shape

    def merge(self, dtype: np.dtype = None):
        merged = np.empty(self.__shape[1:], dtype=dtype)
        for i, w in self.__where:
            p = self.__phasors[i]
