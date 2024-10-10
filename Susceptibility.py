# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:18:35 2024

@author: hbrit
"""
import numpy as np
from phase_diagram import epsilon, g_rashba, g_zeeman, alpha_rashba, alpha_zeeman, beta, a
import matplotlib.pyplot as plt
import scipy.constants
import time

MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]

k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]

# system parameters
V = -0.4823435925215936  # attractive potential

# simulation params
FREQUENCY_LIMIT = 1000


def Det(kx, ky, ohmega, H):
    return -ohmega**2 + epsilon(kx, ky)**2 - (alpha_zeeman *
                                              g_zeeman(kx, ky, beta)[2])**2 - 2j * ohmega * epsilon(kx, ky) - \
        ((MU_B * H + alpha_rashba * g_rashba(kx, ky, beta)[0])**2 +
         (alpha_rashba*g_rashba(kx, ky, beta)[1])**2)


def GF_up_up(kx, ky, ohmega, H):
    return 1/Det(kx, ky, ohmega, H) * (1j * ohmega - epsilon(kx, ky) + alpha_zeeman * g_zeeman(kx, ky, beta)[2])


def GF_down_down(kx, ky, ohmega, H):
    return 1/Det(kx, ky, ohmega, H) * (1j * ohmega - epsilon(kx, ky) - alpha_zeeman * g_zeeman(kx, ky, beta)[2])


def GF_up_down(kx, ky, ohmega, H):
    return 1/Det(kx, ky, ohmega, H) * (alpha_rashba * (g_rashba(kx, ky, beta)[0] - 1j * g_rashba(kx, ky, beta)[1]) + MU_B * H)


def GF_down_up(kx, ky, ohmega, H):
    return 1/Det(kx, ky, ohmega, H) * (alpha_rashba * (g_rashba(kx, ky, beta)[0] + 1j * g_rashba(kx, ky, beta)[1]) + MU_B * H)


def matsubara_frequency(T, m):
    return (2*m+1)*k_B * T * np.pi


def Susceptibility(T, H, plot=False, N_points=100):

    ky_range = np.linspace(0, .75, 3 * N_points//4)
    kx_range = np.hstack((np.linspace(0, .25, N_points//4),
                         np.linspace(.45, .7, N_points//4)))

    X, Y = np.meshgrid(kx_range, ky_range)

    KX = X * 2*np.pi / a
    KY = Y * 4*np.pi / a / np.sqrt(3) - X * 2*np.pi / a / np.sqrt(3)

    chi_0 = np.zeros((KY.shape[0], KY.shape[1], 2*FREQUENCY_LIMIT),
                     dtype=np.complex128)
    freq = np.zeros(2 * FREQUENCY_LIMIT)
    for m in range(-FREQUENCY_LIMIT, FREQUENCY_LIMIT):
        freq[m+FREQUENCY_LIMIT] = matsubara_frequency(T, m)

    chi_0 -= (GF_up_up(KX[:, :, None], KY[:, :, None], freq[None, None, :], H) * GF_down_down(-KX[:, :, None], -KY[:, :, None], -freq[None, None, :],
              H) - GF_up_down(KX[:, :, None], KY[:, :, None], freq[None, None, :], H) * GF_down_up(-KX[:, :, None], -KY[:, :, None], -freq[None, None, :], H))

    if plot:
        plt.pcolor(KX, KY, np.real(chi_0.sum(axis=2)), shading="auto")

    return np.real(chi_0.sum() * T * k_B / (N_points**2))


def delta(T, H):
    return 1 - Susceptibility(T, H) * V


def braket(start_T_U, start_T_L, H, tol=0.001):
    # Larger T => +ve Delta
    # Smaller T => -ve Delta
    MAX_ITERATIONS = 10
    iterations = 0

    current_T_U = start_T_U
    current_T_L = start_T_L

    current_delta_U = delta(current_T_U, H)
    current_delta_L = delta(current_T_L, H)
    # print("Δ_max = {}, Δ_min = {}".format(
    #    current_delta_U, current_delta_L))

    while abs(current_delta_L) > tol and abs(current_delta_U) > tol and iterations < MAX_ITERATIONS:
        # print("Δ_max = {}, Δ_min = {}".format(
        #    current_delta_U, current_delta_L))
        if abs(current_delta_L) > abs(current_delta_U):
            current_T_L = (current_T_L + current_T_U) / 2
            current_delta_L = delta(current_T_L, H)

        else:
            current_T_U = (current_T_L + current_T_U) / 2
            current_delta_U = delta(current_T_U, H)
        iterations += 1

    if abs(current_delta_L) < tol:
        # print(current_delta_L)
        return current_T_L
    else:
        # print(current_delta_U)
        return current_T_U


def plot_critical_field():
    critical = np.array(
        [[0.0, 6.496874999999999],
         [3.25, 6.496874999999999],
            [6.5, 6.3953613281249995],
            [9.75, 6.345397567749023],
            [13.0, 6.246250730752944],
            [16.25, 6.051055395416915],
            [19.5, 5.9092337845868315],
            [22.75, 5.724570228818493],
            [26.0, 5.456230999342626],
            [29.25, 5.2004701712484405],
            [32.5, 4.875440785545413],
            [35.75, 4.570725736448825],
            [39.0, 4.213637788288761],
            [42.25, 3.884447336078701],
            [45.5, 3.459585908695093],
            [48.75, 3.0271376701082064],
            [52.0, 2.6487454613446806],
            [55.25, 2.152105687342553],
            [58.5, 1.647705916871642],
            [61.75, 0.9783253881425373], ])

    plt.plot(critical[:, 1], critical[:, 0])
    plt.ylabel(r"$H_{c2}$ (T)")
    plt.xlabel(r"$T_c$ (K)")
    plt.xlim((0, 7))


def test_freq_convergence(start, stop, points=10):
    global FREQUENCY_LIMIT
    L = []
    chi = []
    for i in range(points+1):
        FREQUENCY_LIMIT = int(i * (stop - start) / (points) + start)
        chi.append(Susceptibility(6.5, 0))
        L.append(FREQUENCY_LIMIT * 2)

        print("χ = {} at {} matsubara frequencies".format(chi[-1], L[-1]))

    plt.plot(L, chi)
    plt.xlabel("Number of matsubara frequencies used")
    plt.ylabel(r"$\chi^0$")


time_0 = time.time()
#print(braket(.7, 1.35, 70))
# print(delta(6.5, 0))

'''
temp = 6.6
for H_index in range(20):
    H = 65 * H_index / 20
    temp = braket(temp, 0, H)
    print("[{}, {}],".format(H, temp))
'''
print(Susceptibility(6.5, 0))
# plot_critical_field()
#test_freq_convergence(500, 2000, 5)

print("Runtime: {} s".format(time.time() - time_0))
