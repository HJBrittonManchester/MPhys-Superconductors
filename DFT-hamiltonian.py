# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:12:32 2024

@author: hbrit
"""
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
import scipy.constants
from phase_diagram import H_0, a
from matplotlib import colors


ham_file = "MoS2/MoS2_hr_new.dat"

# constants
MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]
k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]

# system variables
DEBYE_ENERGY = 0.022  # 0.022  # eV
FERMI_ENERGY = -0.96  # eV

# Default field allignment - x direction
PHI_DEFAULT = 0
THETA_DEFAULT = np.pi / 2

# Simulation settings
RESOLUTION = 300
NUM_FREQ = 500

# k - path settings
DEFAULT_PATH = ['K', 'G', 'K']

# Full BZ settings
PRESELECTION_BOXSIZE = 0.22  # set to -1 to use full area but, 0.22 works well


# Bracket settings
BRACKET_TOLERANCE = 1e-6
MAX_BRACKET_STEPS = 25
TEMP_START = 6.4
TEMP_STOP = 2
TEMP_STEPS = 10
H_U_START = 60
H_L_START = -1


#########################################################
# get_k_XXXX methods


def get_k_block(res=RESOLUTION, size_of_box=PRESELECTION_BOXSIZE):

    if size_of_box == -1:  # use full space

        k = np.zeros((3, res*res))

        for xi in range(res):
            for yi in range(res):
                k[0, xi + res * yi] = 2 * np.pi * xi / (res)
                k[1, xi + res * yi] = 2 * np.pi * yi / (res)
        return k
    else:

        reduced_res = int(size_of_box * res)

        # box around 1/3, 1/3

        k = np.zeros((3, 2 * reduced_res * reduced_res))

        for xi in range(reduced_res):
            for yi in range(reduced_res):
                k[0, xi + reduced_res * yi] = 2 * \
                    np.pi * (xi / res + 1/3 - size_of_box/2)
                k[1, xi + reduced_res * yi] = 2 * \
                    np.pi * (yi / res + 1/3 - size_of_box/2)

        # box around 2/3, 2/3

        for xi in range(reduced_res):
            for yi in range(reduced_res):
                k[0, xi + reduced_res * yi + reduced_res * reduced_res] = 2 * \
                    np.pi * (xi / res + 2/3 - size_of_box/2)
                k[1, xi + reduced_res * yi + reduced_res * reduced_res] = 2 * \
                    np.pi * (yi / res + 2/3 - size_of_box/2)

        return k


def get_k_path(path=DEFAULT_PATH,  res=RESOLUTION):
    name_points = {'G': np.array([0, 0, 0]), 'K': np.array(
        [1/3, 1/3, 0]), 'M': np.array([0, 1/2, 0])}

    path_lengths = np.zeros(len(path)-1)

    for i in range(len(path)-1):
        path_lengths[i] = np.linalg.norm(
            (name_points[path[i+1]] - name_points[path[i]]))

    path_size = np.array(path_lengths * RESOLUTION /
                         np.sum(path_lengths), dtype=int)

    kpath = []

    for i in range(len(path)-1):
        for j in range(RESOLUTION):
            kpath.append(j / RESOLUTION * (name_points[path[i+1]] -
                         name_points[path[i]]) + name_points[path[i]])

    return np.array(kpath).T * 2 * np.pi


def get_k_path_spacing(path):
    spacing = np.zeros(len(path[0]))

    for i in range(1, len(path[0])):

        spacing[i] = np.linalg.norm(path[:, i] - path[:, i-1]) + spacing[i-1]

        # print(spacing[i] - spacing[i-1])

    return spacing / spacing[-1]

################################################################
# eigen value operations


def epsilon(hamk):
    eigs = np.array([np.real_if_close(LA.eig(hamk[:, :, i])[0])
                    for i in range(hamk.shape[2])], dtype=float)
    return eigs


def projection_z(hamk, band=0):
    eigvecs = np.array([LA.eig(hamk[:, :, i])[1]
                        for i in range(hamk.shape[2])], dtype=complex)
    # print(eigvecs.shape)

    proj = eigvecs[:, 0] * eigvecs[:, 0].conj() - \
        eigvecs[:,  1] * eigvecs[:, 1].conj()

    # print(proj[:, 0] - proj[:, 1])

    return proj


def get_eig_vec(hamk):
    return np.array([LA.eig(hamk[:, :, i])[1]
                     for i in range(hamk.shape[2])], dtype=complex)


def projection_x(hamk, band=0):
    eigvecs = np.array([LA.eig(hamk[:, :, i])[1]
                        for i in range(hamk.shape[2])], dtype=complex)

    proj = eigvecs[:, :, 0] * eigvecs[:, :, 1].conj() + \
        eigvecs[:, :, 1] * eigvecs[:, :, 0].conj()

    return proj


def projection_y(hamk, band=0):
    eigvecs = np.array([np.linalg.eig(hamk[:, :, i])[1]
                        for i in range(hamk.shape[2])], dtype=complex)

    proj = 1j * eigvecs[:, :, 0] * eigvecs[:, :, 1].conj() - \
        eigvecs[:, :, 1] * eigvecs[:, :, 0].conj() * 1j

    return proj

#############################################################


def get_hamr():

    ndeg = np.array([])

    with open(ham_file) as f:
        f.readline()  # skip the metadata line

        nb = int(f.readline())  # number of bands
        nr = int(f.readline())  # number of lattice points to consider

        rvec = np.zeros((3, nr), dtype=float)
        hamr = np.zeros((2, 2, nr), dtype=np.complex128)

        for step in range(7):  # should calculate this number from nr
            ndeg = np.append(ndeg, np.array(
                f.readline().strip().split("    "), dtype=int))

        for ri in range(nr):
            for xi in range(nb):
                for yi in range(nb):
                    temp_data = f.readline().strip().split()

                    index_1, index_2 = int(
                        temp_data[3]) - 1,  int(temp_data[4]) - 1

                    rvec[:, ri] = np.array(temp_data[:3], dtype=float)
                    hamr[index_1, index_2, ri] = complex(float(
                        temp_data[5]), float(temp_data[6]))

    return hamr, ndeg, rvec


def find_hamk(k, hamr, ndeg, rvec):
    ham = np.zeros((2, 2, k.shape[1]), dtype=np.complex128)
    for i in range(k.shape[1]):
        for j in range(hamr.shape[2]):

            if np.linalg.norm(rvec[:, j]) > 9999:  # for debug purposes
                continue

            # Compute the phase factor
            # Ensure the correct sign in the phase
            phase = np.dot(k[:, i], rvec[:, j])

            # Add the contribution to the Hamiltonian in k-space
            ham[:, :, i] += hamr[:, :, j] * \
                complex(np.cos(phase), -np.sin(phase)) / ndeg[j]

    return ham


def get_toy_ham(b):

    kx = (b[0] + 2 * b[1]) / (a * np.sqrt(3))
    ky = b[0] / a

    H = np.zeros((2, 2, len(kx)), dtype=complex)

    for i in range(len(kx)):
        H[:, :, i] = H_0(kx[i], ky[i])

    return H


def get_bands_on_path(path=DEFAULT_PATH):

    hamr, ndeg, rvec = get_hamr()  # Read in the real-space hamiltonian

    hamk = find_hamk(get_k_path(path), hamr, ndeg, rvec)  # FT the hamiltonian

    hamk_pert = vary_ham(hamk)  # Adjust the fermi level

    hamk_pert_toy = get_toy_ham(get_k_path())

    # Find the energy eigen values
    energy_toy = epsilon(hamk_pert_toy)
    energy_real = epsilon(hamk_pert)

    return energy_real, energy_toy


def vary_ham(ham, ef=FERMI_ENERGY, H=0, theta=THETA_DEFAULT, phi=PHI_DEFAULT):

    new_ham = np.zeros_like(ham, dtype=complex)
    new_ham[0, 0, :] = ham[0, 0, :] - ef - H * MU_B * np.cos(theta)
    new_ham[1, 1, :] = ham[1, 1, :] - ef + H * MU_B * np.cos(theta)

    new_ham[0, 1, :] = ham[0, 1, :] - H * MU_B * \
        complex(np.cos(phi), np.sin(phi)) * np.sin(theta)
    new_ham[1, 0, :] = ham[1, 0, :] - H * MU_B * \
        complex(np.cos(phi), -np.sin(phi)) * np.sin(theta)

    # print(new_ham[0, 1, 10])

    # print(H * MU_B * complex(np.cos(phi), np.sin(phi)) * np.sin(theta))

    return new_ham


def get_greens_function(ham, freq):
    greens = np.zeros_like(ham, dtype=complex)
    det = (1j*freq - ham[0, 0, :])*(1j*freq -
                                    ham[1, 1, :]) - (-ham[0, 1, :]*-ham[1, 0, :])

    greens[0, 0, :] = (1j*freq - ham[1, 1, :])
    greens[1, 1, :] = (1j*freq - ham[0, 0, :])
    greens[0, 1, :] = ham[0, 1, :]
    greens[1, 0, :] = ham[1, 0, :]

    greens = greens / det
    return greens


def matsubara_frequency(T, m):
    return (2*m+1)*k_B * T * np.pi


def susc(ham_N, ham_P, T, for_plot=False):
    chi_0 = np.zeros(ham_N.shape[2], dtype=complex)

    for m in range(-NUM_FREQ, NUM_FREQ):
        current_freq = matsubara_frequency(T, m)
        greens_N = get_greens_function(ham_N, -current_freq)
        greens_P = get_greens_function(ham_P, current_freq)

        chi_0 -= greens_P[0, 0, :] * greens_N[1, 1, :] - \
            greens_P[0, 1, :]*greens_N[1, 0, :]

    if for_plot:
        return np.real_if_close(k_B * T / (RESOLUTION**2) * chi_0, 1e-4)

    return np.real_if_close(k_B * T / (RESOLUTION**2) * np.sum(chi_0), 1e-4)


def find_v():

    hamr, ndeg, rvec = get_hamr()  # Read in the real-space hamiltonian

    hamk_P = find_hamk(get_k_block(), hamr, ndeg, rvec)  # FT the hamiltonian

    # hamk_P = get_toy_ham()

    hamk_pert_P = vary_ham(hamk_P)  # Adjust the fermi level

    # Find the mean of energy eigen values
    energy = epsilon(hamk_pert_P).mean(axis=1)

    # Find the points within the Debye energy of fermi surface
    significant_kpoints_indices = np.where(abs(energy) < DEBYE_ENERGY)

    # Find -ve ham to significant k points
    significant_kpoints = get_k_block(
    )[:, significant_kpoints_indices][:, 0, :]

    hamk_N = find_hamk(-significant_kpoints, hamr, ndeg, rvec)
    # hamk_N = get_toy_ham(-1)[:, :, significant_kpoints_indices][:, :, 0]

    print(hamk_N.shape)

    # correct +ve ham to significant k points
    hamk_P = hamk_P[:, :, significant_kpoints_indices][:, :, 0]
    hamk_pert_P = vary_ham(hamk_P)  # reset +ve
    hamk_pert_N = vary_ham(hamk_N)

    v = 1/susc(hamk_pert_N, hamk_pert_P, 6.5)

    return hamk_P, hamk_N, v


def delta(T,  v, hamk_P, hamk_N, H=0, phi=PHI_DEFAULT, theta=THETA_DEFAULT):

    hamk_pert_N = vary_ham(hamk_N, FERMI_ENERGY, H, theta=theta, phi=phi)

    hamk_pert_P = vary_ham(hamk_P, FERMI_ENERGY, H, theta=theta, phi=phi)

    return 1 - v * susc(hamk_pert_N, hamk_pert_P, T)


def braket(ham_P, ham_N, T, v, start_H_U=H_U_START, start_H_L=H_L_START,
           theta=THETA_DEFAULT, phi=PHI_DEFAULT, tol=BRACKET_TOLERANCE):

    # Larger H => +ve Delta
    # Smaller H => -ve Delta
    iterations = 0

    #theta = T
    #T = 6.4

    current_H_U = start_H_U
    current_H_L = start_H_L

    current_delta_U = delta(T, v,  ham_P, ham_N,
                            H=current_H_U, theta=theta, phi=phi)
    current_delta_L = delta(T, v,  ham_P, ham_N,
                            H=current_H_L, theta=theta, phi=phi)

    # print("Δ_max = {}, Δ_min = {}".format(
    #    current_delta_U, current_delta_L))

    if current_delta_U < 0:
        print("Upper H too low")
        return start_H_L
    elif current_delta_L > 0:
        print("Lower H too high")
        return 0

    old_H_U = 100
    old_H_L = 0

    while abs(current_delta_L) > tol and abs(current_delta_U) > tol and iterations < MAX_BRACKET_STEPS:
        # print("Δ_max = {}, Δ_min = {}".format(
        #    current_delta_U, current_delta_L))

        if current_delta_L > 0 and current_delta_U > 0:
            print("both +ve sign")

            current_H_U = current_H_L
            current_H_L = old_H_L

            # reset upper
            current_delta_U = current_delta_L
            # recalculate lower
            delta(T, v,  ham_P, ham_N, H=current_H_L, theta=theta, phi=phi)

        elif current_delta_L < 0 and current_delta_U < 0:
            print("both -ve sign")

            current_H_L = current_H_U
            current_H_U = old_H_U

            # reset lower
            current_delta_L = current_delta_U
            # recalculate Upper
            current_delta_U = delta(
                T, v,  ham_P, ham_N, H=current_H_U, theta=theta, phi=phi)

        elif abs(current_delta_L) > abs(current_delta_U):
            old_H_L = current_H_L
            current_H_L = (current_H_L + current_H_U) / 2
            current_delta_L = delta(
                T, v,  ham_P, ham_N, H=current_H_L, theta=theta, phi=phi)

        else:
            old_H_U = current_H_U
            current_H_U = (current_H_L + current_H_U) / 2
            current_delta_U = delta(
                T, v,  ham_P, ham_N, H=current_H_U, theta=theta, phi=phi)

        iterations += 1

    if iterations == MAX_BRACKET_STEPS-1:
        print("Reached max iterations")

    if abs(current_delta_L) < tol:
        # print(current_delta_L)
        return current_H_L
    else:
        # print(current_delta_U)
        return current_H_U


def bracketing(ham_P, ham_N, v):
    T_array = []
    H_array = []
    for t_index in range(TEMP_STEPS):
        temp_T = TEMP_START - (TEMP_START - TEMP_STOP) / \
            (TEMP_STEPS-1) * t_index
        T_array.append(temp_T)

        H_array.append(braket(ham_P.copy(), ham_N.copy(), temp_T, v))

        print(H_array[-1])

    print(T_array)
    print(H_array)
    plt.plot(T_array, H_array)
    plt.title(r"res: {}, $N_f$ = {}".format(RESOLUTION, NUM_FREQ))


def main():
    '''
    hamk_P, hamk_N, v = find_v()
    print(v)

    bracketing(hamk_P, hamk_N, v)
    '''
    hamr_obs = get_hamr()
    k = get_k_path()
    hamk = find_hamk(k, *hamr_obs)
    hamk = vary_ham(hamk, H=9, theta=0)

    p_z = np.real(projection_z(hamk))
    p_x = np.real(projection_x(hamk))
    p_y = np.real(projection_y(hamk))

    # total_proj = p_z**2 + p_x**2 + p_y**2

    print((p_z).mean(axis=0))

    # e_dft, e_t = get_bands_on_path()
    e_dft = epsilon(hamk)

    xk = get_k_path_spacing(k)

    fig, axs = plt.subplots(3, figsize=(10, 15), dpi=100, sharex=True)

    norm = colors.Normalize(-1, 1)
    projs = [p_x, p_y, p_z]
    titles = ['x', 'y', 'z']

    for j in range(len(axs)):

        for i in range(len(e_dft[0])):

            scatter = axs[j].scatter(xk, e_dft[:, i], c=projs[j][:, i],
                                     cmap='bwr', norm=norm, linewidths=1)
            axs[j].set_title(titles[j])
            # plt.plot(xk, e_t)
            # plt.hlines(0, 0, 1)
            # plt.hlines(0.022, 0, 1)
            # plt.hlines(-0.022, 0, 1)

            # Add colorbar to indicate the values of z

            # plt.xlabel("Distance along k-path")
            # plt.ylabel("Energy / eV")

            # plt.title("Energy bands for No field ")
    # plt.xlim(0.3, .4)
    #plt.colorbar(scatter, label='y value')
    print(e_dft.min())


main()
