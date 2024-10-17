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
V = -0.34864337758262043
# -0.3578945281365615
V_for3 = -0.3213018026628682
# simulation params
FREQUENCY_LIMIT = 1000


def H(H_mag, theta, phi):
    # spherical coordinates - theta is angle from z axis, phi is azimuthal angle
    return H_mag * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def Det(kx, ky, omega, H_mag, theta, phi):
    return -omega**2 + epsilon(kx, ky)**2 - (MU_B * H(H_mag, theta, phi)[2])**2 - (alpha_zeeman * g_zeeman(kx, ky, beta)[2])**2 - \
        2j * omega * epsilon(kx, ky) - 2 * alpha_zeeman * g_zeeman(kx, ky, beta)[2] * MU_B * H(H_mag, theta, phi)[2] - \
        (alpha_rashba * g_rashba(kx, ky, beta)[0] + MU_B * H(H_mag, theta, phi)[0])**2 - (
            alpha_rashba * g_rashba(kx, ky, beta)[1] + MU_B * H(H_mag, theta, phi)[1])**2


def GF_up_up(kx, ky, omega, H_mag, theta, phi):
    return 1/Det(kx, ky, omega, H_mag, theta, phi) * (1j * omega - epsilon(kx, ky) + alpha_zeeman * g_zeeman(kx, ky, beta)[2] + MU_B * H(H_mag, theta, phi)[2])


def GF_down_down(kx, ky, omega, H_mag, theta, phi):
    return 1/Det(kx, ky, omega, H_mag, theta, phi) * (1j * omega - epsilon(kx, ky) - alpha_zeeman * g_zeeman(kx, ky, beta)[2] - MU_B * H(H_mag, theta, phi)[2])


def GF_up_down(kx, ky, omega, H_mag, theta, phi):
    return 1/Det(kx, ky, omega, H_mag, theta, phi) * (alpha_rashba * (g_rashba(kx, ky, beta)[0] - 1j * g_rashba(kx, ky, beta)[1]) + (MU_B * (H(H_mag, theta, phi)[0] - 1j * H(H_mag, theta, phi)[1])))


def GF_down_up(kx, ky, omega, H_mag, theta, phi):
    return 1/Det(kx, ky, omega, H_mag, theta, phi) * (alpha_rashba * (g_rashba(kx, ky, beta)[0] + 1j * g_rashba(kx, ky, beta)[1]) + (MU_B * (H(H_mag, theta, phi)[0] + 1j * H(H_mag, theta, phi)[1])))


def matsubara_frequency(T, m):
    return (2*m+1)*k_B * T * np.pi


def Susceptibility(T, H_mag, theta=np.pi/2, phi=0, plot=False, N_points=100, threshold=2):
    b2_range = np.linspace(0, 1, N_points)
    b1_range = np.linspace(0, 1, N_points)

    B1, B2 = np.meshgrid(b1_range, b2_range)

    KX = B1 * 4 / np.sqrt(3) * np.pi / a + np.pi/a * B2
    KY = B2 * np.pi / a * 2

    # select region near fermi level
    if threshold != -1:
        Z = np.where(np.abs(epsilon(KX, KY)) < threshold, [KX, KY], 0)
    else:
        Z = np.where(np.isreal(KX), [KX, KY], 0)
    Valid_k_points = Z[:, ~(Z == 0).all(0)]

    valid_kx, valid_ky = Valid_k_points

    if plot:
        plt.pcolor(KX, KY, epsilon(KX, KY), shading='auto')
        plt.scatter(valid_kx, valid_ky, color="k")

    GF_part = 0
    block_size = FREQUENCY_LIMIT // 10
    for block_step in range(10):
        # print("block {}".format(block_step))
        freq_block = np.zeros(2 * block_size)
        for m in range(0, block_size):
            freq_block[2 *
                       m] = matsubara_frequency(T, m + block_step * block_size)

            freq_block[2*m +
                       1] = matsubara_frequency(T, -(m + block_step * block_size))

        GF_part -= (GF_down_down(valid_kx[:, None], valid_ky[:, None], freq_block[None, :], H_mag, theta, phi) *
                    GF_up_up(-valid_kx[:, None], -valid_ky[:, None], -freq_block[None, :], H_mag, theta, phi) -
                    GF_down_up(valid_kx[:, None], valid_ky[:, None], freq_block[None, :], H_mag, theta, phi) *
                    GF_up_down(-valid_kx[:, None], -valid_ky[:, None], -freq_block[None, :], H_mag, theta, phi)).sum()

    chi_0 = np.real(T * k_B * GF_part / (N_points**2))
    return chi_0


def delta(T, H_mag, theta=np.pi/2, phi=0.):
    return 1 - Susceptibility(T, H_mag, theta, phi) * V


def braket(start_H_U, start_H_L, T, theta=np.pi/2, phi=0., tol=0.0001):

    # Larger H => +ve Delta
    # Smaller H => -ve Delta
    MAX_ITERATIONS = 10
    iterations = 0

    current_H_U = start_H_U
    current_H_L = start_H_L

    current_delta_U = delta(T, current_H_U, theta, phi)
    current_delta_L = delta(T, current_H_L, theta, phi)

    # print("Δ_max = {}, Δ_min = {}".format(
    #    current_delta_U, current_delta_L))

    if current_delta_U < 0:
        print("Upper H too low")
        return start_H_L
    elif current_delta_L > 0:
        print("Lower H too high")
        return 0

    while abs(current_delta_L) > tol and abs(current_delta_U) > tol and iterations < MAX_ITERATIONS:
        # print("Δ_max = {}, Δ_min = {}".format(
        #     current_delta_U, current_delta_L))
        if abs(current_delta_L) > abs(current_delta_U):

            current_H_L = (3*current_H_L + current_H_U) / 4
            current_delta_L = delta(T, current_H_L, theta, phi)

        else:
            current_H_U = (current_H_L + 3*current_H_U) / 4
            current_delta_U = delta(T, current_H_U, theta, phi)

        iterations += 1

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

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(crit_N_150_F_1000[:, 1], crit_N_150_F_1000[:, 0])
    ax.plot(crit_N_100_F_1000[:, 1],
            crit_N_100_F_1000[:, 0], 'r', label="N = 100")
    ax.plot(crit_new_method[:, 1], crit_new_method[:, 0], 'g', label="N = 100")

    ax.set_ylabel(r"$H_{c2}$ (T)")
    ax.set_xlabel(r"$T_c$ (K)")
    ax.set_xlim((0, 7))
    ax.set_ylim((0, 85))

    ax.set_aspect("auto")


def test_freq_convergence(start, stop, points=10, N_points=100):
    global FREQUENCY_LIMIT
    L = []
    chi = []
    for i in range(points+1):
        FREQUENCY_LIMIT = int(i * (stop - start) / (points) + start)
        chi.append(Susceptibility(6.5, 0, N_points=N_points))
        L.append(FREQUENCY_LIMIT * 2)

        print("χ = {} at {} matsubara frequencies".format(chi[-1], L[-1]))

    plt.plot(L, chi)
    plt.xlabel("Number of matsubara frequencies used")
    plt.ylabel(r"$\chi^0$")


def find_V():
    return 1 / Susceptibility(6.5, 0)


def find_phase_diagram(steps=15):
    H = 0
    for T_index in range(steps):
        T = 6.5 * (1-(T_index) / steps)
        H_upper_estimate = -11 * T + 100

        H = braket(H_upper_estimate, H,  T)
        print("[{}, {}],".format(H, T))


time_0 = time.time()
# print(braket(.7, 1.35, 70))
# print(delta(6.5, 0))

V = find_V()


print(V)
find_phase_diagram(10)


#print(Susceptibility(6.5, 0, plot=True, N_points=400, threshold=1))
# plot_critical_field()
#test_freq_convergence(500, 2000, 5, 200)

print("Runtime: {} s".format(time.time() - time_0))
