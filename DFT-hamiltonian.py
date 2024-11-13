# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:12:32 2024

@author: hbrit
"""
import numpy as np
import scipy.constants
import matplotlib.pyplot as plt

ham_file = "MoS2/MoS2_hr.dat"

# constants
MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]
k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]

# system variables
DEBYE_ENERGY = 0.022  # eV
FERMI_ENERGY = 0.915  # eV


# Simulation settings
RESOLUTION = 100
NUM_FREQ = 500
PRESELECTION_BOXSIZE = 0.22  # set to -1 to use full area but, 0.22 works well


def get_hamr():

    ndeg = np.array([])

    with open(ham_file) as f:
        f.readline()  # skip the metadata line

        nb = int(f.readline())  # number of bands
        nr = int(f.readline())  # number of lattice points to consider

        rvec = np.zeros((3, nr), dtype=float)
        hamr = np.zeros((2, 2, nr), dtype=np.complex128)

        for step in range(11):  # should calculate this number from nr
            ndeg = np.append(ndeg, np.array(
                f.readline().strip().split("    "), dtype=int))

        for ri in range(nr):
            for xi in range(nb):
                for yi in range(nb):
                    temp_data = f.readline().strip().split("   ")

                    rvec[:, ri] = np.array(temp_data[:3], dtype=float)
                    hamr[xi, yi, ri] = complex(float(
                        temp_data[-2]), float(temp_data[-1]))

    return hamr, ndeg, rvec


def get_k(res=RESOLUTION, size_of_box=PRESELECTION_BOXSIZE):

    if size_of_box == -1:  # use full space

        k = np.zeros((3, res*res))

        for xi in range(res):
            for yi in range(res):
                k[0, xi + res * yi] = 2 * np.pi * xi / (res-1)
                k[1, xi + res * yi] = 2 * np.pi * yi / (res-1)
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


def find_hamk(k, hamr, ndeg, rvec):
    ham = np.zeros((2, 2, k.shape[1]), dtype=np.complex128)
    for i in range(k.shape[1]):
        for j in range(hamr.shape[2]):
            phase = 0
            phase = np.dot(k[:, i], rvec[:, j])
            ham[:, :, i] += hamr[:, :, j] * \
                complex(np.cos(phase), -np.sin(phase)) / float(ndeg[j])

    return ham


def epsilon(hamk):
    return np.array([np.real_if_close(np.linalg.eig(hamk[:, :, i])[0]) for i in range(hamk.shape[2])], dtype=float)


def vary_ham(ham, ef=FERMI_ENERGY, H=0, theta=0, phi=np.pi / 2):

    new_ham = np.zeros_like(ham, dtype=np.complex128)
    new_ham[0, 0, :] = ham[0, 0, :] - ef - H * MU_B * np.cos(phi)
    new_ham[1, 1, :] = ham[1, 1, :] - ef + H * MU_B * np.cos(phi)

    new_ham[0, 1, :] = ham[0, 1, :] - H * MU_B * \
        complex(np.cos(theta), np.sin(theta)) * np.sin(phi)
    new_ham[1, 0, :] = ham[1, 0, :] - H * MU_B * \
        complex(np.cos(theta), -np.sin(theta)) * np.sin(phi)

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


def susc(ham_N, ham_P, T, plot=False):
    if plot:
        chi_0 = np.zeros(ham_N.shape[2], dtype=complex)
    else:
        chi_0 = 0

    for m in range(-NUM_FREQ, NUM_FREQ):
        current_freq = matsubara_frequency(T, m)
        greens_N = get_greens_function(ham_N, -current_freq)
        greens_P = get_greens_function(ham_P, current_freq)

        if plot:
            chi_0 -= greens_P[0, 0, :] * greens_N[1, 1, :] - \
                greens_N[0, 1, :]*greens_P[1, 0, :]
        else:
            chi_0 -= np.sum(greens_P[0, 0, :] * greens_N[1, 1, :] -
                            greens_N[0, 1, :]*greens_P[1, 0, :])

    if plot:
        plot_linearised_map(-abs(chi_0) * T * k_B)

        return np.real_if_close(k_B * T / (RESOLUTION**2) * np.sum(chi_0), 1e-4)
    else:
        return np.real_if_close(k_B * T / (RESOLUTION**2) * chi_0, 1e-4)


def plot_linearised_map(l_map):
    square_map = l_map.reshape(
        int(np.sqrt(l_map.size)), int(np.sqrt(l_map.size)))
    c = plt.pcolor(square_map, shading='auto')
    plt.colorbar(c)
    plt.contourf(square_map,
                 levels=[-0.022, 0.022], colors='red', alpha=0.5)
    plt.show


hamr, ndeg, rvec = get_hamr()  # Read in the real-space hamiltonian

hamk_P = find_hamk(get_k(), hamr, ndeg, rvec)  # FT the hamiltonian

hamk_pert_P = vary_ham(hamk_P)  # Adjust the fermi level


# Find the mean of energy eigen values
energy = epsilon(hamk_pert_P).mean(axis=1)


# plot_linearised_map(energy)


# Find the points within the Debye energy of fermi surface
significant_kpoints_indices = np.where(abs(energy) < 0.022)


# Find -ve ham to significant k points
significant_kpoints = get_k()[:, significant_kpoints_indices][:, 0, :]
hamk_N = find_hamk(-significant_kpoints, hamr, ndeg, rvec)


# correct +ve ham to significant k points
hamk_P = hamk_P[:, :, significant_kpoints_indices][:, :, 0]
hamk_pert_P = vary_ham(hamk_P)  # reset +ve
hamk_pert_N = vary_ham(hamk_N)


v = 1/susc(hamk_pert_N, hamk_pert_P, 6.5)
print(v)

critical_field = []
for t in range(10):
    temp = 6.2 - t * 0.6
    H_array = []
    delta_array = []
    for h_index in range(200):
        # print(t*np.pi/20)
        hamk_pert_N = vary_ham(hamk_N, FERMI_ENERGY,
                               h_index,  np.pi * 15 / 20)

        hamk_pert_P = vary_ham(hamk_P, FERMI_ENERGY,
                               h_index, t * np.pi * 15 / 20)

        delta_array.append(1 - v * susc(hamk_pert_N, hamk_pert_P, temp))
        # print(delta_array[-1])
        H_array.append(h_index)
    # print(min(abs(np.array(delta_array))))
    #print(t * np.pi / 20)
    critical_field.append(
        np.where(abs(np.array(delta_array)) == min(abs(np.array(delta_array))))[0][0])
plt.plot(critical_field)
