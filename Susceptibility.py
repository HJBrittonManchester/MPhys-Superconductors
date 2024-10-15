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
# FOR 1000 freq and 100 N -> -0.34864337758262043  # attractive potential
V = -0.3578945281365615
V_for3 = -0.3213018026628682
# simulation params
FREQUENCY_LIMIT = 2000


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


def Susceptibility(T, H, plot=False, N_points=150, useBoundingBoxes=False):
    chi_total = 0
    bounding_boxes = np.array(
        [[[0, .25], [.3, .9]], [[.3, .9], [.3, .9]], [[.3, .9], [0, .25]]])

    if plot and useBoundingBoxes:
        fig, axs = plt.subplots(1, 3, figsize=(15, 15))

    for i in range(3 if useBoundingBoxes else 1):
        # create top left range

        if useBoundingBoxes:
            box = bounding_boxes[i]
        else:
            box = np.array([[0, 1], [0, 1]])

        ky_range = np.linspace(box[1, 0], box[1, 1], int(
            N_points * (box[1, 1] - box[1, 0])))
        kx_range = np.linspace(box[0, 0], box[0, 1], int(
            N_points * (box[0, 1] - box[0, 0])))

        X, Y = np.meshgrid(kx_range, ky_range)

        KX = X * 2*np.pi / a
        KY = Y * 4*np.pi / a / np.sqrt(3) - X * 2*np.pi / a / np.sqrt(3)

        chi_0 = np.zeros((KY.shape[0], KY.shape[1]),
                         dtype=np.complex128)

        block_size = FREQUENCY_LIMIT // 10
        for block_step in range(10):
            # print("block {}".format(block_step))
            freq_block = np.zeros(2 * block_size)
            for m in range(0, block_size):
                freq_block[2 *
                           m] = matsubara_frequency(T, m + block_step * block_size)
                if m == 0:
                    pass
                freq_block[2*m +
                           1] = matsubara_frequency(T, -(m + block_step * block_size))

            chi_0 -= (GF_up_up(KX[:, :, None], KY[:, :, None], freq_block[None, None, :], H) * GF_down_down(-KX[:, :, None], -KY[:, :, None], -freq_block[None, None, :],
                                                                                                            H) - GF_up_down(KX[:, :, None], KY[:, :, None], freq_block[None, None, :], H) * GF_down_up(-KX[:, :, None], -KY[:, :, None], -freq_block[None, None, :], H)).sum(axis=2)

        chi_total += chi_0.sum()
        if plot:
            if useBoundingBoxes:
                axs[i].pcolor(KX, KY, np.real(chi_0),
                              shading="auto", vmin=-(N_points**2), vmax=0)
            else:
                plt.pcolor(KX, KY, np.real(chi_0),
                           shading="auto", vmin=-(N_points**2), vmax=0)

            # axs[i].set_aspect('equal')

    return np.real(chi_total * T * k_B / (N_points**2))


def delta(T, H):
    return 1 - Susceptibility(T, H) * V


def braket(start_T_U, start_T_L, H, tol=0.0001):
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
    critical_old_for_lab_book = np.array(
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

    critical = np.array([[0.0, 6.5],
                         [8.333333333333334, 6.37109375],
                         [16.666666666666668, 6.0283203125],
                         [25.0, 5.523221015930176],
                         [33.333333333333336, 4.892415761947632],
                         [41.666666666666664, 4.2371028158813715],
                         [50.0, 3.542377527355711],
                         [51.666666666666664, 3.40234375],
                         [53.333333333333336, 3.2615814208984375],
                         [55.0, 3.1290668845176697],
                         [56.666666666666664, 2.9960002042353153],
                         [58.333333333333336, 2.863453315672814],
                         [58.333333333333336, 2.8670584966518504],
                         [60.0, 2.739708368928916],
                         [61.666666666666664, 2.6241808600547305],
                         [63.333333333333336, 2.509980643332132],
                         [65.0, 2.4038101293478418],
                         [65.0, 2.400390625],

                         [66.66666666666667, 2.2981891109532397],
                         [66.66666666666667, 2.2970527648925785],
                         [68.33333333333333, 2.19835703521967],
                         [70.0, 2.099996549193748],
                         [71.66666666666667, 2.0023404676901473],
                         [73.33333333333333, 1.9094527495412146],
                         [75.0, 1.8175664771035749],
                         [76.66666666666667, 1.7404101705149375],
                         [78.33333333333333, 1.6731277221149106],
                         [80.0, 1.6055323903052856],
                         ])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(critical[:, 1], critical[:, 0])
    ax.set_ylabel(r"$H_{c2}$ (T)")
    ax.set_xlabel(r"$T_c$ (K)")
    ax.set_xlim((0, 7))
    ax.set_ylim((0, 85))

    ax.set_aspect("auto")


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
# print(braket(.7, 1.35, 70))
# print(delta(6.5, 0))

lower_bound_estimate = [6, 4, 2, 0.1]

temp = 6.6
for H_index in range(15):
    H = 75 * H_index / 14
    temp = braket(temp, 0.1, H)
    print("[{}, {}],".format(H, temp))

#print(Susceptibility(6.5, 0, False, 100))
# plot_critical_field()
# test_freq_convergence(500, 2000, 5)

print("Runtime: {} s".format(time.time() - time_0))
