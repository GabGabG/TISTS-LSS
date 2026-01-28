import numpy as np
import typing


class Phasors:

    def __init__(self, complex_amplitudes: np.ndarray):
        # complex_amplitudes should be of shape (M, N, N) where M is the number of time steps.
        self.__amps = complex_amplitudes

    @property
    def amps(self):
        return self.__amps

    @classmethod
    def from_correlation(cls, correlation_matrix: np.ndarray, N: int, rng_gen=None, seed=None, real_dtype=np.float32,
                         complex_dtype=np.complex64) -> "Phasors":
        if rng_gen is None:
            rng_gen = np.random.default_rng(seed)
        re = np.real(correlation_matrix)
        im = np.imag(correlation_matrix)
        stack_1 = np.hstack((re, -im))
        stack_2 = np.hstack((im, re))
        full_corr = np.vstack((stack_1, stack_2), dtype=correlation_matrix.dtype)
        n_ind = len(re)
        means = np.zeros(len(full_corr))
        random_parts = rng_gen.multivariate_normal(means, full_corr, (N, N)).astype(real_dtype)
        reals = random_parts[..., n_ind:]
        imags = random_parts[..., n_ind:]
        amps = (reals + 1j * imags).transpose((2, 0, 1)).astype(complex_dtype)  # We want shape (M, N, N) where M is
        # number of time steps.
        return cls(amps)

    @staticmethod
    def check_types(other: typing.Union["Phasors", np.generic, float, int]):
        return isinstance(other, (Phasors, np.generic, float, int))

    def __add__(self, other: typing.Union["Phasors", np.generic, float, int]):
        if not self.check_types(other):
            raise TypeError("Addition of `Phasors` should be with `Phasors` objects or scalars.")
        return self.__amps + other

    def __iadd__(self, other: typing.Union["Phasors", np.generic, float, int]):
        if not self.check_types(other):
            raise TypeError("Addition of `Phasors` should be with `Phasors` objects or scalars.")
        self.__amps += other
        return self.__amps

    def __mul__(self, other: typing.Union["Phasors", np.generic, float, int]):
        if not self.check_types(other):
            raise TypeError("Multiplication of `Phasors` should be with `Phasors` objects or scalars.")
        return self.__amps * other

    def __imul__(self, other: typing.Union["Phasors", np.generic, float, int]):
        if not self.check_types(other):
            raise TypeError("Multiplication of `Phasors` should be with `Phasors` objects or scalars.")
        self.__amps *= other
        return self.__amps

