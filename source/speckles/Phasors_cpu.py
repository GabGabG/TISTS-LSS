import numpy as np
import typing


class Phasors:

    def __init__(self, complex_amplitudes: np.ndarray):
        # complex_amplitudes should be of shape (M, N, N) where M is the number of time steps.
        shape = complex_amplitudes.shape
        self.__amps = complex_amplitudes
        self.__M, self.__N1, self.__N2 = shape
        self.__shape = (self.__M, self.__N1, self.__N2)

    @property
    def complex_amps(self) -> np.ndarray:
        return self.__amps

    @property
    def M(self) -> int:
        return self.__M

    @property
    def N1(self) -> int:
        return self.__N1

    @property
    def N2(self) -> int:
        return self.__N2

    @property
    def shape(self) -> typing.Tuple[int, int, int]:
        return self.__shape

    @classmethod
    def from_correlation(cls, correlation_matrix: np.ndarray, N: int, rng_gen: np.random.Generator = None,
                         seed: int = None, real_dtype: np.dtype = np.float32,
                         complex_dtype: np.dtype = np.complex64) -> "Phasors":
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
    def check_types(other: typing.Union["Phasors", np.generic, float, int]) -> bool:
        return isinstance(other, (Phasors, np.generic, float, int))

    def __add__(self, other: typing.Union["Phasors", np.generic, float, int]) -> "Phasors":
        if not self.check_types(other):
            raise TypeError("Addition of `Phasors` should be with `Phasors` objects or scalars.")
        amps = self.__amps + other
        new_phasors = Phasors(amps)
        return new_phasors

    def __iadd__(self, other: typing.Union["Phasors", np.generic, float, int]) -> "Phasors":
        if not self.check_types(other):
            raise TypeError("Addition of `Phasors` should be with `Phasors` objects or scalars.")
        self.__amps += other
        return self

    def __mul__(self, other: typing.Union["Phasors", np.generic, float, int]) -> "Phasors":
        if not self.check_types(other):
            raise TypeError("Multiplication of `Phasors` should be with `Phasors` objects or scalars.")
        amps = self.__amps * other
        new_phasors = Phasors(amps)
        return new_phasors

    def __imul__(self, other: typing.Union["Phasors", np.generic, float, int]) -> "Phasors":
        if not self.check_types(other):
            raise TypeError("Multiplication of `Phasors` should be with `Phasors` objects or scalars.")
        self.__amps *= other
        return self

    def propagate(self, shift: bool = True, inverse: bool = False, dtype: np.dtype = None) -> "Phasors":
        if inverse:
            self.__amps = np.fft.ifft2(self.__amps, axes=(-1, -2)).astype(dtype)
        else:
            self.__amps = np.fft.fft2(self.__amps, axes=(-1, -2)).astype(dtype)
        if shift:
            if inverse:
                self.__amps = np.fft.ifftshift(self.__amps, axes=(-1, -2))
            else:
                self.__amps = np.fft.fftshift(self.__amps, axes=(-1, -2))
        return self

    def apply_mask(self, mask: np.ndarray, dtype: np.dtype = None) -> "Phasors":
        N1 = self.N1
        N2 = self.N2
        if mask.shape != (N1, N2):
            raise ValueError(f"The mask should be of shape `({N1}, {N2})`.")
        np.multiply(self.__amps, mask, out=self.__amps, dtype=dtype)
        return self

    def phases(self, dtype: np.dtype = None) -> np.ndarray:
        return np.angle(self.__amps).astype(dtype)

    def real_amps(self, dtype: np.dtype = None) -> np.ndarray:
        return np.abs(self.__amps).astype(dtype)

    def intensities(self, dtype: np.dtype = None) -> np.ndarray:
        real_amps = self.real_amps(dtype)
        return real_amps * real_amps
