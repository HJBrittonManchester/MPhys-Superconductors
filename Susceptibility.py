# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:18:35 2024

@author: hbrit
"""
import numpy as np
from phase_diagram import epsilon, g_rashba, g_zeeman, alpha_rashba, alpha_zeeman, beta, a
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy.constants
import time

beta = 1.00730995e+01  # for 3 mev band gap at fermi surface

#alpha_rashba *= 5
#alpha_zeeman = 0

MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]

k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]

# system parameters
# FOR 1000 freq and 100 N -> -0.34864337758262043  # attractive potential
V = -0.34864337758262043

# simulation params
FREQUENCY_LIMIT = 10000
DEFAULT_N_POINTS = 1000


def Det(kx, ky, omega, H):
    return (1j * omega - epsilon(kx, ky))**2 - (alpha_zeeman *
                                                g_zeeman(kx, ky, beta)[2])**2 - \
        ((MU_B * H - alpha_rashba * g_rashba(kx, ky, beta)[0])**2 +
         (alpha_rashba*g_rashba(kx, ky, beta)[1])**2)


def GF_up_up(kx, ky, omega, H):
    return 1/Det(kx, ky, omega, H) * (1j * omega - epsilon(kx, ky) + alpha_zeeman * g_zeeman(kx, ky, beta)[2])


def GF_down_down(kx, ky, omega, H):
    return 1/Det(kx, ky, omega, H) * (1j * omega - epsilon(kx, ky) - alpha_zeeman * g_zeeman(kx, ky, beta)[2])


def GF_up_down(kx, ky, omega, H):
    return 1/Det(kx, ky, omega, H) * (-alpha_rashba * (g_rashba(kx, ky, beta)[0] - 1j * g_rashba(kx, ky, beta)[1]) - MU_B * H)


def GF_down_up(kx, ky, omega, H):
    return 1/Det(kx, ky, omega, H) * (-alpha_rashba * (g_rashba(kx, ky, beta)[0] + 1j * g_rashba(kx, ky, beta)[1]) - MU_B * H)


def matsubara_frequency(T, m):
    return (2*m+1)*k_B * T * np.pi


def Susceptibility(T, H_mag, theta=np.pi/2, phi=0, plot=False, N_points=DEFAULT_N_POINTS, threshold=.01):
    b2_range = np.linspace(0, 1, N_points)
    b1_range = np.linspace(0, 1, N_points)

    B1, B2 = np.meshgrid(b1_range, b2_range)

    #
    KY = B1 * np.pi / a * 2
    KX = B2 * 4 / np.sqrt(3) * np.pi / a - 2 * np.pi/a / np.sqrt(3) * B1

    # select region near fermi level
    Z_valid = np.where(np.abs(epsilon(KX, KY)) < threshold, [KX, KY], 0)
    Z_invalid = np.where(np.abs(epsilon(KX, KY)) >= threshold, [KX, KY], 0)
    # Z_invalid is the points which don't pass the threshold but this is used
    # for plotting

    Valid_k_points = Z_valid[:, ~(Z_valid == 0).all(0)]
    Invalid_k_points = Z_invalid[:, ~(Z_invalid == 0).all(0)]

    # print(Valid_k_points.shape)

    valid_kx, valid_ky = Valid_k_points
    invalid_kx, invalid_ky = Invalid_k_points

    # to plot energy
    if plot:
        # define your scale, with white at zero
        #norm = colors.CenteredNorm()
        #colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        c = plt.pcolormesh(KX, KY, epsilon(KX, KY),
                           shading='auto', cmap='RdYlBu')
        cbar = plt.colorbar(c)
        cbar.set_label("Energy")
        plt.scatter(valid_kx, valid_ky, color="k",
                    label="excluded k-points")
        plt.ylabel(r"$k_y$")
        plt.xlabel(r"$k_x$")
        plt.plot(1/3 * (4 / np.sqrt(3) * np.pi / a + 2 /
                 np.sqrt(3) * np.pi/a), 1/3 * np.pi / a * 2, 'xk')
        plt.text(1/3 * (4 / np.sqrt(3) * np.pi / a + 2 / np.sqrt(3) *
                 np.pi/a) + .04, 1/3 * np.pi / a * 2 + .04, r"$K^{\prime}$")

        plt.plot(0, 4/3 * np.pi / a, 'xk')
        plt.text(.04,  4/3 * np.pi / a + .04, r"$K$")
        print(valid_kx.shape)

    # print(valid_kx.shape)

    GF_part = np.zeros_like(valid_kx)
    block_size = FREQUENCY_LIMIT // 5
    for block_step in range(5):
        # print("block {}".format(block_step))
        freq_block = np.zeros(2 * block_size)
        for m in range(0, block_size):
            freq_block[2 *
                       m] = matsubara_frequency(T, m + block_step * block_size)
            if m == 0:
                pass

            freq_block[2*m +
                       1] = matsubara_frequency(T, -(m + block_step * block_size))
        GF_part -= np.real((GF_up_up(valid_kx[:, None], valid_ky[:, None], freq_block[None, :], H_mag) * GF_down_down(-valid_kx[:, None], -valid_ky[:, None], -freq_block[None, :], H_mag) - GF_up_down(
            valid_kx[:, None], valid_ky[:, None], freq_block[None, :], H_mag) * GF_down_up(-valid_kx[:, None], -valid_ky[:, None], -freq_block[None, :], H_mag)).sum(axis=1))

    chi_0 = GF_part * T * k_B / (N_points**2)

    # to plot susc
    #trip = plt.tripcolor(valid_kx, valid_ky, chi_0)
    # plt.colorbar(trip)

    return chi_0.sum()


def delta(T, H_mag, theta=np.pi/2, phi=0., N_points=DEFAULT_N_POINTS):
    return 1 - Susceptibility(T, H_mag, theta, phi, N_points=N_points) * V


def braket(start_H_U, start_H_L, T, theta=np.pi/2, phi=0., tol=0.0001, N_points=DEFAULT_N_POINTS):

    # Larger H => +ve Delta
    # Smaller H => -ve Delta
    MAX_ITERATIONS = 20
    iterations = 0

    current_H_U = start_H_U
    current_H_L = start_H_L

    current_delta_U = delta(T, current_H_U, theta, phi, N_points=N_points)
    current_delta_L = delta(T, current_H_L, theta, phi, N_points=N_points)

    # print("Δ_max = {}, Δ_min = {}".format(
    #     current_delta_U, current_delta_L))

    if current_delta_U < 0:
        print("Upper H too low")
        return start_H_L
    elif current_delta_L > 0:
        print("Lower H too high")
        return 0

    old_H_U = 100
    old_H_L = 0

    while abs(current_delta_L) > tol and abs(current_delta_U) > tol and iterations < MAX_ITERATIONS:
        # print("Δ_max = {}, Δ_min = {}".format(
        #     current_delta_U, current_delta_L))

        if current_delta_L > 0 and current_delta_U > 0:
            print("both +ve sign")

            current_H_U = current_H_L
            current_H_L = old_H_L

            # reset upper
            current_delta_U = current_delta_L
            # recalculate lower
            current_delta_L = delta(
                T, current_H_L, theta, phi, N_points=N_points)

        elif current_delta_L < 0 and current_delta_U < 0:
            print("both -ve sign")

            current_H_L = current_H_U
            current_H_U = old_H_U

            # reset lower
            current_delta_L = current_delta_U
            # recalculate Upper
            current_delta_U = delta(
                T, current_H_U, theta, phi, N_points=N_points)

        elif abs(current_delta_L) > abs(current_delta_U):
            old_H_L = current_H_L
            current_H_L = (current_H_L + current_H_U) / 2
            current_delta_L = delta(
                T, current_H_L, theta, phi, N_points=N_points)

        else:
            old_H_U = current_H_U
            current_H_U = (current_H_L + current_H_U) / 2
            current_delta_U = delta(
                T, current_H_U, theta, phi, N_points=N_points)

        iterations += 1

    if iterations == MAX_ITERATIONS:
        print("Reached max iterations")

    if abs(current_delta_L) < tol:
        # print(current_delta_L)
        return current_H_L
    else:
        # print(current_delta_U)
        return current_H_U


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

    crit_N_100_F_1000 = np.array([[0.0, 6.4984375],
                                  [5.357142857142857, 6.44844970703125],
                                  [10.714285714285714, 6.299657917022705],
                                  [16.071428571428573, 6.057483779639005],
                                  [21.428571428571427, 5.743319595947105],
                                  [26.785714285714285, 5.379590012614576],
                                  [32.142857142857146, 4.982589591744144],
                                  [37.5, 4.5677602026018205],
                                  [42.857142857142854, 4.140181589462194],
                                  [48.214285714285715, 3.690395748447848],
                                  [53.57142857142857, 3.2415962798918665],
                                  [58.92857142857143, 2.8182170937345647],
                                  [64.28571428571429, 2.446585850450542],
                                  [69.64285714285714, 2.121180390720096],
                                  [75.0, 1.8172138085219567], ])

    crit_N_150_F_1000 = np.array([[0.0, 6.4984375],
                                  [5.357142857142857, 6.44844970703125],
                                  [10.714285714285714, 6.274859285354614],
                                  [16.071428571428573, 5.985412756353616],
                                  [21.428571428571427, 5.571594671922503],
                                  [26.785714285714285, 5.047945963086169],
                                  [32.142857142857146, 4.371468975945483],
                                  [37.5, 3.4203997117701217],
                                  [42.857142857142854, 2.003393194149474],
                                  [48.214285714285715, 2.003393194149474],
                                  [53.57142857142857, 2.003393194149474],
                                  [58.92857142857143, 2.003393194149474],
                                  [64.28571428571429, 2.003393194149474], ])

    crit_new_method = np.array([[0, 6.5],
                                [21.141128540039062, 5.8500000000000005],
                                [31.016577695554588, 5.2],
                                [39.23510592132524, 4.55],
                                [46.78323742503433, 3.9],
                                [54.70678554109935, 3.25],
                                [62.55796839113147, 2.6],
                                [73.04718026339746, 1.9500000000000002], ])

    crit_100_new_new = np.array([[0, 6.5],
                                 [25.845243644714348, 5.8500000000000005],
                                 [35.59437083898155, 5.2],
                                 [43.24682465153819, 4.55],
                                 [49.42421926044948, 3.9],
                                 [53.861323807441295, 3.25],
                                 [56.91373549186723, 2.6],
                                 [58.132149591504, 1.9500000000000002],
                                 [69.63749999999999, 0.6499999999999999], ])

    critical_fix_100_1500 = np.array([[0.,       6.5],
                                      [13.7303657,  6.2],
                                      [19.69572128,  5.9],
                                      [24.76885251,  5.6],
                                      [29.36361165,  5.3],
                                      [33.61417474,  5.],
                                      [37.72122555,  4.7],
                                      [41.80940548,  4.4],
                                      [45.74662311,  4.1],
                                      [49.6864702,  3.8],
                                      [53.66382129,  3.5],
                                      [57.81020357,  3.2],
                                      [62.16114903,  2.9],
                                      [66.90295563,  2.6],
                                      [71.88749668,  2.3],
                                      [76.97818632,  2.],
                                      [82.05615926,  1.7],
                                      [85.29672722,  1.4],
                                      [86.97646389,  1.1],
                                      [87.68514181,  0.8]])

    critical_fix_120_1500 = np.array([[0.0, 6.5],
                                      [12.205383899494905, 6.199999999999999],
                                      [17.329622133252244, 5.9],
                                      [21.28487622172021, 5.6],
                                      [24.779553301081776, 5.300000000000001],
                                      [28.014810591945555, 5.0],
                                      [30.999661909568232, 4.699999999999999],
                                      [33.996905476951945, 4.4],
                                      [37.15287311318057, 4.1],
                                      [40.50678270016597, 3.8000000000000003],
                                      [44.484133786865925, 3.5],
                                      [49.118797317951945, 3.1999999999999997],
                                      [54.250992779668024, 2.9000000000000004],
                                      [58.74865875273787, 2.5999999999999996],
                                      [62.121871683357355, 2.3000000000000003],
                                      [64.5636355374076, 2.0],
                                      [66.22363973352938, 1.6999999999999997],
                                      [67.26694205982807, 1.4000000000000001],
                                      [67.74428615148838, 1.0999999999999999],
                                      [67.6953125, 0.8000000000000003]])

    critical_fix_150_2000 = np.array([[0.0, 6.5], [11.445632330469719, 6.199999999999999], [16.427123980146007, 5.9], [20.480767268591535, 5.6], [24.146817856418785, 5.300000000000001], [27.701089888820555, 5.0], [31.332913862693232, 4.699999999999999], [35.263995320701945, 4.4], [39.72977741005557, 4.1], [44.85736863766597, 3.8000000000000003], [50.198245114990925, 3.5], [
                                     54.879905716389445, 3.1999999999999997], [58.56022739880865, 2.9000000000000004], [61.35455474883162, 2.5999999999999996], [63.443282816169855, 2.3000000000000003], [64.96204251982948, 2.0], [65.9900276729825, 1.6999999999999997], [66.56602958424213, 1.4000000000000001], [66.69287929113682, 1.0999999999999999], [66.32325674279522, 0.8000000000000003]])

    critical_fix_200_2000 = np.array([[0.0, 6.5], [14.0674279736905, 6.199999999999999], [20.003552034557174, 5.9], [24.60781252638091, 5.6], [28.492642218926242, 5.300000000000001], [31.919839888820555, 5.0], [35.00234745644323, 4.699999999999999], [37.801837117576945, 4.4], [40.37797076943057, 4.1], [42.75897996579097, 3.8000000000000003], [44.979739255615925, 3.5], [
                                     47.057640091389445, 3.1999999999999997], [49.033707623418024, 2.9000000000000004], [50.93676910430037, 2.5999999999999996], [52.808517191169855, 2.3000000000000003], [54.696692178032606, 2.0], [56.66675496790438, 1.6999999999999997], [58.74033073170307, 1.4000000000000001], [60.60439357336338, 1.0999999999999999], [61.290488493283505, 0.8000000000000003]])

    critical_selected_100_4000 = np.array([[0.0, 6.5], [18.079091186811446, 6.199999999999999], [25.64575405423891, 5.9], [31.533195788232355, 5.6], [36.47454575910953, 5.300000000000001], [40.807779341945555, 5.0], [44.63735722206823, 4.699999999999999], [48.117999226951945, 4.4], [51.25443561318057, 4.1], [54.12982957516597, 3.8000000000000003], [56.779055661865925, 3.5], [
                                          59.301902786701945, 3.1999999999999997], [61.843766217168024, 2.9000000000000004], [64.94983062773787, 2.5999999999999996], [72.02909824585736, 2.3000000000000003], [79.2333984375, 2.0], [82.42969442102938, 1.6999999999999997], [84.26706413014057, 1.4000000000000001], [85.29341945226963, 1.0999999999999999], [85.7021095870335, 0.8000000000000003]])

    critical_selected_500_10000 = np.array([[[0.0, 6.5], [18.074282956859548, 5.9], [25.73009557642778, 5.300000000000001], [31.761380659568232, 4.699999999999999], [37.01615436318057, 4.1], [
                                           41.925540036865925, 3.5], [46.946305279668024, 2.9000000000000004], [52.956832620857355, 2.3000000000000003], [62.36865926477938, 1.6999999999999997], [69.42641505773838, 1.0999999999999999]]])

    critical_selected_1000_1000 = np.array([[0.0, 6.5], [13.978626647124734, 6.185], [19.9859555318214, 5.87], [24.743507928266354, 5.555], [28.875728532420524, 5.24], [32.66873980530046, 4.925], [36.22660664836119, 4.609999999999999], [39.598139966039966, 4.295], [42.87931912640704, 3.98], [46.113174357638755, 3.6650000000000005], [
                                           49.27330168930612, 3.35], [52.465412504079296, 3.0349999999999997], [55.71091120640372, 2.72], [59.038376025125416, 2.405], [62.46622874464707, 2.0900000000000003], [65.96556544884676, 1.775], [69.53687008730391, 1.4599999999999997], [73.41531901223806, 1.145], [79.56635589509291, 0.8299999999999998], [85.06498950028254, 0.5150000000000003]])

    crit_selected_1000_5000 = np.array([[0.0, 6.5], [11.37174469824445, 6.29], [16.250100230922534, 6.08], [19.9859555318214, 5.87], [23.275395774600604, 5.66], [26.181072285375002, 5.45], [28.919673844920524, 5.24], [31.439747844127837, 5.029999999999999], [33.881668229907156, 4.82], [36.22660664836119, 4.609999999999999], [38.515948445701945, 4.4], [40.743604452325435, 4.1899999999999995], [42.92326443890704, 3.98], [45.067311769810324, 3.77], [47.20943594241787, 3.56], [49.33921965805612, 3.35], [51.46849677747653, 3.14], [
                                       53.5878311541133, 2.93], [55.77682917515372, 2.72], [57.98492853853642, 2.5100000000000002], [60.251754495857355, 2.3000000000000003], [62.55411936964707, 2.0900000000000003], [64.89499190626333, 1.8800000000000003], [67.26104144416645, 1.6699999999999995], [69.65771969667891, 1.4599999999999997], [72.20027561938744, 1.2499999999999998], [75.12950264719204, 1.0399999999999998], [79.91791839509291, 0.8299999999999998], [83.92096716961888, 0.6199999999999999], [99.93546429987978, 0.4099999999999999]])

    crit_3mev_1000_10000 = np.array([[0.0, 6.5], [4.1191906763829085, 6.29], [5.752961596953853, 6.08], [7.004882465535328, 5.87], [8.038183027680649, 5.66], [8.933586124973727, 5.45], [9.736284510332395, 5.24], [10.458990760915285, 5.029999999999999], [11.141458296112368, 4.82], [11.769575682937743, 4.609999999999999], [12.365083901350232, 4.4], [12.953421877831287, 4.1899999999999995], [13.508424620226469, 3.98], [14.05706960044147, 3.77], [14.608009112735171, 3.56], [15.155541650330196, 3.35], [15.72481223345521, 3.14], [
                                    16.295007118639838, 2.93], [16.884845869915296, 2.72], [17.51569941479852, 2.5100000000000002], [18.166446710573926, 2.3000000000000003], [18.86963481394187, 2.0900000000000003], [19.633552614123673, 1.8800000000000003], [20.46819518926661, 1.6699999999999995], [21.41739414918938, 1.4599999999999997], [22.544166479224714, 1.2499999999999998], [23.924240941212418, 1.0399999999999998], [25.555550730741075, 0.8299999999999998], [27.20555187201036, 0.6199999999999999], [28.576892488627614, 0.4099999999999999]])

    fig, ax = plt.subplots(figsize=(5, 5))
    #ax.plot(crit_N_150_F_1000[:, 1], crit_N_150_F_1000[:, 0])
    # ax.plot(crit_N_100_F_1000[:, 1],
    #        crit_N_100_F_1000[:, 0], 'r', label="N = 100")
    #ax.plot(crit_new_method[:, 1], crit_new_method[:, 0], 'g', label="N = 100")
    ax.plot(critical_fix_100_1500[:, 1],
            critical_fix_100_1500[:, 0], 'g', label="N = 100")
    ax.plot(critical_fix_120_1500[:, 1],
            critical_fix_120_1500[:, 0], 'r', label="N = 120")
    ax.plot(critical_fix_150_2000[:, 1],
            critical_fix_150_2000[:, 0], label="N = 150")
    ax.plot(critical_fix_200_2000[:, 1],
            critical_fix_200_2000[:, 0], label="N = 200")

    ax.set_ylabel(r"$H_{c2}$ (T)")
    ax.set_xlabel(r"$T_c$ (K)")
    ax.set_xlim((0, 7))
    ax.set_ylim((0, 85))
    plt.legend()
    ax.set_aspect("auto")


def test_freq_convergence(start, stop, points=10, N_points=100):
    global FREQUENCY_LIMIT
    global V
    L = []
    chi = []
    for i in range(points+1):
        FREQUENCY_LIMIT = int(i * (stop - start) / (points) + start)

        V = find_V()
        chi.append(braket(60, 20, 4))
        L.append(FREQUENCY_LIMIT * 2)

        print("χ = {} at {} matsubara frequencies".format(chi[-1], L[-1]))

    plt.plot(L, chi)
    plt.xlabel("Number of matsubara frequencies used")
    plt.ylabel(r"$\chi^0$")


def test_N_convergence(T, N_values=[100, 101, 102, 103], FL=1000):
    global FREQUENCY_LIMIT
    global V
    H = []
    temp_freq = FREQUENCY_LIMIT
    FREQUENCY_LIMIT = FL

    for n in N_values:
        print("starting n = " + str(n))
        V = find_V()
        H.append(braket(100, 30, T, N_points=n))

    plt.plot(N_values, H)

    # return FL to value before test
    FREQUENCY_LIMIT = temp_freq


def find_V(N_points=DEFAULT_N_POINTS):
    return 1 / Susceptibility(6.5, 0, N_points=N_points)


def range_guesser(T):
    return 40 * np.sqrt((-T + 6.5) / (T+1)**(2/3))


def find_phase_diagram(steps=25):
    values = []
    H = 5
    for T_index in range(steps):
        T = 6.3 * (1-(T_index) / steps) + .2
        H_upper_estimate = range_guesser(T) + 20
        H_lower_estimate = np.clip(range_guesser(T) - 25, -1, 100)

        H = braket(H_upper_estimate, 0.5,  T)
        values.append([H, T])
        print("[{}, {}],".format(H, T))
    return np.array(values)


time_0 = time.time()
# print(delta(6.5, 0))

V = find_V()
print(V)

# #print(braket(65, 40, 3))

# test_N_convergence(3)

r = find_phase_diagram(30)

plt.plot(r[:, 1], r[:, 0])
print(r.tolist())


#print(Susceptibility(6.5, 0, plot=True))
# plot_critical_field()
#test_freq_convergence(1500, 4000, 5)

print("Runtime: {} s".format(time.time() - time_0))
