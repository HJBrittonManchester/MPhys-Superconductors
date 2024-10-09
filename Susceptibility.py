# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:18:35 2024

@author: hbrit
"""
import numpy as np
from phase_diagram import H_0

import matplotlib.pyplot as plt

import scipy.constants

MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]

k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]

EXTERNAL_FIELD_STRENGTH = 0  # tesla

EXTERNAL_FIELD = np.array([EXTERNAL_FIELD_STRENGTH, 0, 0])

Fermi_radius = 0.1771  # A^-1

# simulation params
FREQUENCY_LIMIT = 40

IntegralRes = [10, 100]

H_0_vec = np.vectorize(H_0, otypes=[np.matrix])


a = 3.25  # https://www.researchgate.net/publication/279633132_A_tight-binding_model_for_MoS_2_monolayers
b = 4 * np.pi / (np.sqrt(3) * a)

PAULI = np.array([np.matrix([[0, 1], [1, 0]]), np.matrix(
    [[0, -1j], [1j, 0]]), np.matrix([[1, 0], [0, -1]])])


def H_P(magnetic_field_vector):
    # Define perturbing Hamiltonian for Zeeman interaction with external Field H
    # np.einsum implementation allows for any direction of H-field
    return np.matrix(- MU_B * np.einsum("i,ijk->jk", magnetic_field_vector, PAULI))


def Matsubara_Green_Function(frequency, kx, ky):

    inverse_greens_function = np.matrix(1j * frequency * np.identity(2) -
                                        H_0(kx, ky) - H_P(EXTERNAL_FIELD))
    # print(inverse_greens_function)

    return np.linalg.inv(inverse_greens_function)


def matsubara_frequency(T, m):
    return (2*m+1)*k_B * T * np.pi


def Susceptibility_at_T(T):
    N = 40
    chi_0 = 0

    for m in range(-FREQUENCY_LIMIT, FREQUENCY_LIMIT):
        frequency = matsubara_frequency(T, m)
        ks = np.array([[0, 4*np.pi/3 / a]])
        for i in range(N):
            for j in range(N):

                kx = i / (N) * 2 * np.pi / a
                ky = (j - i / np.sqrt(3)) / (N) * 2 * np.pi / a

                GF_plus = Matsubara_Green_Function(frequency, kx, ky)
                GF_minus = Matsubara_Green_Function(-frequency, -kx, -ky)

                chi_0 += (GF_plus[0, 0]*GF_minus[1, 1] - GF_plus[0, 1]
                          * GF_minus[0, 1])

                ks = np.vstack((ks, [kx, ky]))

        #plt.plot(ks[:, 0], ks[:, 1], 'o')
    return chi_0 * T / N**2 / FREQUENCY_LIMIT / 2


def plot_2d(T):
    N_points = 50
    ky_range = np.linspace(0, 1, N_points)
    kx_range = np.linspace(0, 1, N_points)

    X, Y = np.meshgrid(kx_range, ky_range)

    KX = X * 2*np.pi / a
    KY = Y * 2*np.pi / a - X * 2*np.pi / a / np.sqrt(3)

    H_0_mesh = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            if KX[i, j]**2 + (KY[i, j] - 4 * np.pi / 3 / a)**2 < 0.3**2:
                pass
            elif (KX[i, j]-1.15)**2 + (KY[i, j] - .7)**2 < 0.3**2:
                pass
            elif (KX[i, j]-1.15)**2 + (KY[i, j] + 0.5)**2 < 0.3**2:
                pass
            else:
                H_0_mesh[i, j] = 0
                continue

            chi_0 = 0
            for m in range(-FREQUENCY_LIMIT, FREQUENCY_LIMIT):

                frequency = matsubara_frequency(T, m)

                GF_plus = Matsubara_Green_Function(
                    frequency, KX[i, j], KY[i, j])
                GF_minus = Matsubara_Green_Function(
                    -frequency, -KX[i, j], -KY[i, j])

                chi_0 -= np.real((GF_plus[0, 0]*GF_minus[1, 1] - GF_plus[0, 1]
                                  * GF_minus[0, 1])) * T * k_B

            H_0_mesh[i, j] = chi_0 / N_points**2

    #np.linalg.eig(H_0(X[i, j], Y[i, j]))[0][0]

    print(H_0_mesh.sum())

    plt.pcolor(KX, KY, H_0_mesh, shading="auto")
    plt.colorbar()
    return 1 + H_0_mesh.sum() * 0.0828


def test():
    N_points = 50
    ky_range = np.linspace(0, 1, N_points)
    kx_range = np.linspace(0, 1, N_points)

    X, Y = np.meshgrid(kx_range, ky_range)

    KX = X * 2*np.pi / a
    KY = Y * 2*np.pi / a - X * 2*np.pi / a / np.sqrt(3)

    freq = matsubara_frequency(6.5, 1) * 0

    energies = H_0_vec(KX, KY)

    return (1j * freq * np.identity(2) * np.ones_like(energies)) - energies


def plot_crit():
    critical_values = np.array([[0, 6.5], [10, 6.44], [20, 6.05], [
        30, 5.33], [40, 3.96]])
    plt.plot(critical_values[:, 1], critical_values[:, 0])
    plt.xlim((0, 7))


def test_sus():
    v = -0.0828
    tol = 0.005
    temp = 3.5
    current_gap_value = plot_2d(temp)
    while abs(current_gap_value) > tol:
        if current_gap_value < 0:
            temp += 0.01
        else:
            temp -= 0.01
        current_gap_value = plot_2d(temp)
    print(temp)
