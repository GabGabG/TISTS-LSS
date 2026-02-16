import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import cupy as cp
from cupyx.scipy.linalg import toeplitz as toeplitz_gpu
toto = 0

def broadcast_cov_matrix(cov_matrix_base: cp.ndarray) -> cp.ndarray:
    # base_shape = cov_matrix_base.shape
    # if cov_matrix_base.ndim == 1:
    #     new_shape = (2, 2)
    # else:
    #     new_shape = (base_shape[0] * 2, base_shape[1] * 2)
    re = cp.real(cov_matrix_base)
    im = cp.imag(cov_matrix_base)
    first_stack = cp.hstack((re, -im))
    second_stack = cp.hstack((im, re))
    stack = cp.vstack((first_stack, second_stack)).astype(cov_matrix_base.dtype)
    stack += 1e-10 * cp.eye(stack.shape[0])  # Régularisation, permet d'avoir une matrice positive semi-définie
    return stack


def create_circular_mask(sim_width: int, circle_radius: float) -> cp.ndarray:
    x_coords = cp.arange(sim_width) - sim_width / 2
    x_coords_squared = x_coords * x_coords
    y_coords_squared = x_coords_squared[..., None]  # Transposes
    return (x_coords_squared + y_coords_squared) <= (circle_radius * circle_radius)


def create_random_phases(sim_width: int, cov_matrix: cp.ndarray) -> cp.ndarray:
    if not cov_matrix.shape[0] == cov_matrix.shape[1] and cov_matrix.ndim != 2:
        raise ValueError('Covariance matrix must be square.')
    full_cov = broadcast_cov_matrix(cov_matrix)
    n_ind = len(cov_matrix)
    means = cp.zeros(len(full_cov))
    random_parts = cp.random.multivariate_normal(means, full_cov, (sim_width, sim_width)).astype(np.float32)
    reals = random_parts[..., :n_ind]
    imags = random_parts[..., n_ind:]
    amps = (reals + 1j * imags).transpose((2, 0, 1))  # We want shape (M, N, N) where M is the time dimension and N is
    # the width of each simulation.
    return amps


def _inner_time_integrated_generation(sim_width: int, eigenvals: cp.ndarray, mask_radius: float,
                                      cov_matrix: cp.ndarray) -> (cp.ndarray, cp.ndarray):
    M = len(cov_matrix)
    ell = len(eigenvals)
    intensity_matrix = cp.zeros((M, sim_width, sim_width), dtype=np.float32)
    base_amplitudes = cp.zeros((M, sim_width, sim_width), dtype=np.complex64)
    mask = create_circular_mask(sim_width, mask_radius)
    cp.clip(eigenvals, 0, None, eigenvals)  # Nécessaire si valeurs propres très petites mais négatives
    for i, eig in enumerate(eigenvals):
        current_complex_amps = create_random_phases(sim_width, cov_matrix)
        if i == 0: print(current_complex_amps)
        propagation = cp.fft.fftshift(cp.fft.fft2(current_complex_amps, axes=(1, 2)) * cp.sqrt(eig), axes=(1, 2))
        del current_complex_amps
        cp.multiply(propagation, mask, out=propagation)
        propagation = cp.fft.ifftshift(propagation, axes=(1, 2))
        current_final_amplitude_base = cp.fft.ifft2(propagation, axes=(1, 2))
        del propagation
        base_amplitudes += current_final_amplitude_base
        current_final_amplitude = cp.abs(current_final_amplitude_base)
        del current_final_amplitude_base
        current_final_intensity = current_final_amplitude * current_final_amplitude
        del current_final_amplitude
        intensity_matrix += current_final_intensity
        del current_final_intensity
        print(f"\rStep {i + 1} / {ell} done", end="")
    return intensity_matrix, base_amplitudes


def generate_time_integrated_speckles(sim_width: int, mask_radius: float, n_time_sampling: int, T: float,
                                      corrfunc: callable, n_corrfunc_sampling: int, *corrfunc_args,
                                      **corrfunc_kwargs) -> (cp.ndarray, cp.ndarray, cp.ndarray):
    # n_time_sampling : number of time steps, T : integration time
    t1 = cp.linspace(0, T, n_corrfunc_sampling)
    t2 = t1[..., None]
    t = cp.abs(t1 - t2)
    g1_mat = corrfunc(t, *corrfunc_args, **corrfunc_kwargs)
    eigenvals = cp.linalg.eigvalsh(g1_mat) / n_corrfunc_sampling
    first_line = cp.array([corrfunc(T * i, *corrfunc_args, **corrfunc_kwargs) for i in range(n_time_sampling)])
    cov_matrix = toeplitz_gpu(first_line.conj())
    specks, amps = _inner_time_integrated_generation(sim_width, eigenvals, mask_radius, cov_matrix)
    return cov_matrix, specks, amps


def g1_expon(tau, tau_c):
    return cp.exp(-tau / tau_c)

def g1_gauss(tau, tau_c):
    arg = tau / tau_c
    return cp.exp(-arg * arg)


def correlation(time_series):
    # Suppose shape (t, N, M) where t = time step
    t0 = time_series[0]
    m0 = np.mean(t0)
    s0 = np.std(t0, ddof=1)
    corrs = []
    for i in range(len(time_series)):
        ti = time_series[i]
        mi = np.mean(ti)
        si = np.std(ti, ddof=1)
        corr = np.mean((t0 - m0) * (ti - mi)) / (s0 * si)
        corrs.append(corr)
    return corrs


if __name__ == '__main__':
    N = 256  # Image linear size (NxN)
    radius = 200 * 1 / 6  # Related to speckle size with N
    M = 100  # Number of time steps
    ell = 100  # Sampling of eigenvalues
    T = 0.03  # Integration time
    tau_c = 1  # Correlation time
    g1 = g1_expon
    cov, s, a = generate_time_integrated_speckles(N, radius, M, T, g1, ell, tau_c=tau_c)
