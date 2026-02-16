import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cupy as cp
from cupyx.scipy.linalg import toeplitz
import time
import itertools
import random
import os
import pandas as pd


def generate_multiple_time_series(n_time_series: int, possible_corrfuncs: list, possible_Ts: list, possible_tau_c: list,
                                  n_repeat: int = 1, root: str = "", *time_series_generation_args,
                                  **time_series_generation_kwargs):
    combinaisons = list(itertools.product(possible_corrfuncs, possible_Ts, possible_tau_c))
    combinaisons_choisies = random.sample(combinaisons, n_time_series)
    repo_data = []
    cols = ["File path", "Correlation function", "Integration time", "Correlation time"]
    for i, (g1, T, tau_c) in enumerate(combinaisons_choisies):
        for j in range(n_repeat):
            savename = f"speckles_{i}_v{j + 1}.npy"
            savename = os.path.join(root, savename)
            current_info = [savename, g1.__name__, T, tau_c]
            repo_data.append(current_info)
            intensities = time_series_generation(T=T, corrfunc=g1, tau_c=tau_c, *time_series_generation_args,
                                                 **time_series_generation_kwargs)
            cp.save(savename, intensities)
            print(f"{current_info} done")
    df = pd.DataFrame(data=repo_data, columns=cols)
    return df


def time_series_generation(sim_width: int, mask_radius: float, n_time_sampling: int, T: float,
                           corrfunc: callable, n_corrfunc_sampling: int, *corrfunc_args,
                           **corrfunc_kwargs) -> cp.ndarray:
    """
    Algorithme pour générer une série temporelle d'images de tavelures temporellement intégrées.
    :param sim_width: int, largeur de l'image. Pour l'instant, la hauteur est aussi cette valeur.
    :param mask_radius: float, largeur du masque binaire circulaire permettant de former des tavelures de taille finie.
    :param n_time_sampling: int, nombre de pas de temps de la série à générer.
    :param T: float, temps d'intégration. TODO: Peut-être 0?
    :param corrfunc: callable, fonction de corrélation. TODO: Peut-être None pour séquence indépendante?
    :param n_corrfunc_sampling: int, nombre de valeurs propres à considérer pour l'algorithme.
    :param corrfunc_args: arguments à fournir à la fonction de corrélation.
    :param corrfunc_kwargs: arguments-clés à fournir à la fonction de corrélation.
    :return: Intensité finale (ndarray (n_time_sampling, sim_width, sim_width))
    """
    t1 = cp.linspace(0, T, n_corrfunc_sampling)
    t2 = t1[..., None]
    t = cp.abs(t1 - t2)
    g1_mat = corrfunc(t, *corrfunc_args, **corrfunc_kwargs)
    del t1, t2, t
    eigenvals = cp.linalg.eigvalsh(g1_mat) / n_corrfunc_sampling
    del g1_mat
    first_line = cp.array([corrfunc(T * i, *corrfunc_args, **corrfunc_kwargs) for i in range(n_time_sampling)],
                          dtype=cp.float64)
    cov = toeplitz(first_line)
    cov += 1e-10 * cp.eye(cov.shape[0])
    del first_line
    M = len(cov)
    ell = len(eigenvals)
    intensity_matrix = cp.zeros((M, sim_width, sim_width), dtype=cp.float32)
    X = (cp.arange(sim_width) - sim_width / 2)
    Y = X[..., None]
    mask = ((X * X + Y * Y) <= mask_radius * mask_radius).astype(cp.float32)
    del X, Y
    cp.clip(eigenvals, 0, None, eigenvals)  # Nécessaire si valeurs propres très petites mais négatives
    sqrt_eigenvals = cp.sqrt(eigenvals, dtype=cp.float32)
    del eigenvals
    L = cp.linalg.cholesky(cov).astype(cp.float32)
    LT = L.copy().T

    del L, cov
    for i, s_eig in enumerate(sqrt_eigenvals):
        reals = cp.random.standard_normal((sim_width, sim_width, M), dtype=cp.float32)
        reals @= LT
        imags = cp.random.standard_normal((sim_width, sim_width, M), dtype=cp.float32)
        imags @= LT
        # We want shape (M, N, N) where M is the time dimension and N is the width of each simulation.
        amps = (reals + 1j * imags).transpose((2, 0, 1))
        del reals, imags
        propagation = cp.fft.fftshift(cp.fft.fft2(amps, axes=(1, 2)) * s_eig, axes=(1, 2))
        del amps
        propagation *= mask
        propagation = cp.fft.ifftshift(propagation, axes=(1, 2))
        current_final_amplitude_base = cp.fft.ifft2(propagation, axes=(1, 2))
        del propagation
        current_final_amplitude = cp.abs(current_final_amplitude_base, dtype=cp.float32)
        del current_final_amplitude_base
        current_final_intensity = current_final_amplitude * current_final_amplitude
        del current_final_amplitude
        intensity_matrix += current_final_intensity
        del current_final_intensity
        # cp.get_default_memory_pool().free_all_blocks()

        print(f"\rStep {i + 1} / {ell} done", end="")
    print("\nDone!")
    return intensity_matrix


def g1_expon(tau, tau_c):
    return cp.exp(-tau / tau_c)


def g1_gauss(tau, tau_c):
    return cp.exp(-(tau / tau_c) ** 2)


def correlation(time_series):
    # shape (T, N, N)

    t0 = time_series[0]
    m0 = t0.mean()
    s0 = t0.std(ddof=1)

    means = time_series.mean(axis=(1, 2))
    stds = time_series.std(axis=(1, 2), ddof=1)

    centered_0 = t0 - m0
    centered_all = time_series - means[:, None, None]

    numerators = (centered_all * centered_0).mean(axis=(1, 2))

    corrs = numerators / (s0 * stds)

    return corrs


if __name__ == '__main__':
    Ts = np.linspace(1e-3, 2, 1000)
    tau_c = np.linspace(1e-5, 1, 1000)

    N = 256  # Image linear size (NxN)
    radius = 200 * 1 / 6  # Related to speckle size with N
    M = 5_00  # Number of time steps
    ell = 100  # Sampling of eigenvalues
    T = 0.03  # Integration time
    tau_c = 1  # Correlation time
    df = generate_multiple_time_series(2, [g1_expon, g1_gauss], Ts, np.linspace(1e-5, 1, 1000), 1, "./data_toto", N,
                                       radius, M, n_corrfunc_sampling=ell)
    print(df)
    exit()
    intensities = time_series_generation(N, radius, M, T, g1_expon, ell, tau_c=tau_c)
    # exit()
    intensities = intensities.get()
    fig = plt.figure()
    im = plt.imshow(intensities[0], cmap="gray")


    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(intensities[j])
        # return the artists set
        return [im]


    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=len(intensities),
                                  interval=50, blit=True, repeat=False)
    plt.show()
    # cp.save("test_saved.npy", intensities)
    print("Retour au main")
    # intensities_reshape = intensities.reshape(intensities.shape[0], -1)
    # print(cp.corrcoef(intensities_reshape))
    # exit()
    # corr = correlation(intensities)
    # print(corr)
    # plt.plot(corr.get())
    # plt.plot(cov_mat[0].get() ** 2)
    # plt.show()
