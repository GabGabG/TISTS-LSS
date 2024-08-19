import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz


def broadcast_cov_matrix(cov_matrix_base: np.ndarray) -> np.ndarray:
    # base_shape = cov_matrix_base.shape
    # if cov_matrix_base.ndim == 1:
    #     new_shape = (2, 2)
    # else:
    #     new_shape = (base_shape[0] * 2, base_shape[1] * 2)
    re = np.real(cov_matrix_base)
    im = np.imag(cov_matrix_base)
    first_stack = np.hstack((re, -im))
    second_stack = np.hstack((im, re))
    stack = np.vstack((first_stack, second_stack), dtype=cov_matrix_base.dtype)
    return stack


def create_circular_mask(sim_width: int, circle_radius: float) -> np.ndarray:
    x_coords = np.arange(sim_width) - sim_width / 2
    x_coords_squared = x_coords * x_coords
    y_coords_squared = x_coords_squared[..., None]  # Transposes
    return (x_coords_squared + y_coords_squared) <= (circle_radius * circle_radius)


def create_random_phases(sim_width: int, cov_matrix: np.ndarray) -> np.ndarray:
    if not cov_matrix.shape[0] == cov_matrix.shape[1] and cov_matrix.ndim != 2:
        raise ValueError('Covariance matrix must be square.')
    full_cov = broadcast_cov_matrix(cov_matrix)
    n_ind = len(cov_matrix)
    means = np.zeros(len(full_cov))
    random_parts = np.random.multivariate_normal(means, full_cov, (sim_width, sim_width)).astype(np.float32)
    reals = random_parts[..., :n_ind]
    imags = random_parts[..., n_ind:]
    amps = (reals + 1j * imags).transpose((2, 0, 1))  # We want shape (M, N, N) where M is the time dimension and N is
    # the width of each simulation.
    return amps


def _inner_time_integrated_generation(sim_width: int, eigenvals: np.ndarray, mask_radius: float,
                                      cov_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    M = len(cov_matrix)
    ell = len(eigenvals)
    intensity_matrix = np.zeros((M, sim_width, sim_width), dtype=np.float32)
    base_amplitudes = np.zeros((M, sim_width, sim_width), dtype=np.complex64)
    mask = create_circular_mask(sim_width, mask_radius)
    for i, eig in enumerate(eigenvals):
        current_complex_amps = create_random_phases(sim_width, cov_matrix)
        propagation = np.fft.fftshift(np.fft.fft2(current_complex_amps, axes=(1, 2)) * np.sqrt(eig), axes=(1, 2))
        del current_complex_amps
        np.multiply(propagation, mask, out=propagation)
        propagation = np.fft.ifftshift(propagation, axes=(1, 2))
        current_final_amplitude_base = np.fft.ifft2(propagation, axes=(1, 2))
        del propagation
        base_amplitudes += current_final_amplitude_base
        current_final_amplitude = np.abs(current_final_amplitude_base)
        del current_final_amplitude_base
        current_final_intensity = current_final_amplitude * current_final_amplitude
        del current_final_amplitude
        intensity_matrix += current_final_intensity
        del current_final_intensity
        print(f"\rStep {i + 1} / {ell} done", end="")
    return intensity_matrix, base_amplitudes


def generate_time_integrated_speckles(sim_width: int, mask_radius: float, n_time_sampling: int, T: float,
                                      corrfunc: callable, n_corrfunc_sampling: int, *corrfunc_args,
                                      **corrfunc_kwargs) -> (np.ndarray, np.ndarray):
    # n_time_sampling : number of time steps, T : integration time
    t1 = np.linspace(0, T, n_corrfunc_sampling)
    t2 = t1[..., None]
    t = np.abs(t1 - t2)
    g1_mat = corrfunc(t, *corrfunc_args, **corrfunc_kwargs)
    eigenvals = np.linalg.eigvalsh(g1_mat) / n_corrfunc_sampling
    first_line = np.array([corrfunc(T * i, *corrfunc_args, **corrfunc_kwargs) for i in range(n_time_sampling)])
    cov_matrix = toeplitz(first_line.conj())
    specks, amps = _inner_time_integrated_generation(sim_width, eigenvals, mask_radius, cov_matrix)
    return cov_matrix, specks, amps


def g1(tau, tau_c):
    return np.exp(-tau / tau_c)


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
    N = 250  # Image linear size (NxN)
    radius = 200 * 5 / 6  # Related to speckle size with N
    M = 200  # Number of time steps
    ell = 100  # Sampling of eigenvalues
    T = 1 / 100  # Integration time
    tau_c = 1  # Correlation time
    cov, s, a = generate_time_integrated_speckles(N, radius, M, T, g1, ell, tau_c=tau_c)
    plt.imshow(cov)
    plt.colorbar()
    plt.show()
    plt.plot(correlation(s))
    plt.plot(cov[0] ** 2)
    plt.show()
