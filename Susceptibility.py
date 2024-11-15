# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:18:35 2024

@author: hbrit
"""
# beta = 1.00730995e+01  # for 3 mev band gap at fermi surface

# alpha_rashba *= 5
# alpha_zeeman = 0

from phase_diagram import epsilon, GF_up_up, GF_down_down, GF_up_down, GF_down_up,  a, b
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants
from scipy.optimize import curve_fit
import time

MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]
k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]
h = scipy.constants.h
e = scipy.constants.elementary_charge

# h = scipy.constants["Planck"][0]

# e = scipy.constants.physical_constants["elementary_charge"][0]

# print(h)
# print(e)

# system parameters
# FOR 1000 freq and 100 N -> -0.34864337758262043  # attractive potential
V_VAL = -0.34864337758262043

# simulation params
FREQUENCY_LIMIT = 500
DEFAULT_N_POINTS = 1000

en = 0


def matsubara_frequency(T, m):
    return (2*m+1)*k_B * T * np.pi


def Susceptibility(T, H_mag, theta, phi, plot=False, N_points=DEFAULT_N_POINTS, threshold=.022):
    global en
    b2_range = np.linspace(0, 1, N_points)
    b1_range = np.linspace(0, 1, N_points)

    B1, B2 = np.meshgrid(b1_range, b2_range)

    #
    KY = B1 * np.pi / a * 2
    KX = B2 * 4 / np.sqrt(3) * np.pi / a + 2 * np.pi/a / np.sqrt(3) * B1

    # select region near fermi level
    Z_valid = np.where(np.abs(epsilon(KX, KY)) < threshold, [KX, KY], 0)
    Z_invalid = np.where(np.abs(epsilon(KX, KY)) >= threshold, [KX, KY], 0)
    # Z_invalid is the points which don't pass the threshold but this is used
    # for plotting

    # en = epsilon(KX, KY)
    # np.savetxt('.\MoS2\en.txt', en, delimiter=',')

    Valid_k_points = Z_valid[:, ~(Z_valid == 0).all(0)]
    Invalid_k_points = Z_invalid[:, ~(Z_invalid == 0).all(0)]

    # print(Valid_k_points.shape)

    valid_kx, valid_ky = Valid_k_points
    invalid_kx, invalid_ky = Invalid_k_points

    # to plot energy
    if plot:
        pass
        # define your scale, with white at zero
        # norm = colors.CenteredNorm()
        # colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        # c = plt.pcolormesh(KX, KY, epsilon(KX, KY),
        #                  shading='auto', cmap='RdYlBu')
        # cbar = plt.colorbar(c)
        # cbar.set_label("Energy (normalised to $E_{F}$)")
        # plt.scatter(valid_kx, valid_ky, color="k",
        #          label="Included k-values")
        # plt.ylabel(r"$k_{y}$")
        # plt.xlabel(r"$k_{x}$")
        # plt.plot(1/3 * (4 / np.sqrt(3) * np.pi / a + 2 /
        #        np.sqrt(3) * np.pi/a), 1/3 * np.pi / a * 2, 'xk')
        # plt.text(1/3 * (4 / np.sqrt(3) * np.pi / a + 2 / np.sqrt(3) *
        #        np.pi/a) + .04, 1/3 * np.pi / a * 2 + .04, r"$K^{\prime}$")

        # plt.plot(2/3 * (4 / np.sqrt(3) * np.pi / a + 2 /
        #      np.sqrt(3) * np.pi/a), 4/3 * np.pi / a, 'xk')
        # plt.text(2/3 * (4 / np.sqrt(3) * np.pi / a + 2 /
        #        np.sqrt(3) * np.pi/a),  4/3 * np.pi / a + .04, r"$K$")
        # print(valid_kx.shape)
        # plt.legend(loc='upper left')
        # plt.savefig("energy_spectrum_modelham.png", dpi=400)

    GF_part = np.zeros_like(valid_kx)
    block_size = FREQUENCY_LIMIT // 5
    count = 0
    for block_step in range(5):
        # print("block {}".format(block_step))
        freq_block = np.zeros(2 * block_size)
        for m in range(0, block_size):
            count += 2
            freq_block[2 *
                       m] = matsubara_frequency(T, m + block_step * block_size)
            if m == 0:
                pass

            freq_block[2*m +
                       1] = matsubara_frequency(T, -(m + block_step * block_size))
        GF_part -= np.real((GF_up_up(valid_kx[:, None], valid_ky[:, None], freq_block[None, :], H_mag, theta, phi) * GF_down_down(-valid_kx[:, None], -valid_ky[:, None], -freq_block[None, :], H_mag, theta, phi) - GF_up_down(
            valid_kx[:, None], valid_ky[:, None], freq_block[None, :], H_mag, theta, phi) * GF_down_up(-valid_kx[:, None], -valid_ky[:, None], -freq_block[None, :], H_mag, theta, phi)).sum(axis=1))
    # print(count)
    chi_0 = GF_part * T * k_B / (N_points**2)

    # to plot susc
    # trip = plt.tripcolor(valid_kx, valid_ky, chi_0)
    # plt.colorbar(trip)

    return chi_0.sum()


def delta(T, H_mag, theta, phi, N_points=DEFAULT_N_POINTS):
    global V
    return 1 - Susceptibility(T, H_mag, theta, phi, N_points) * V


def braket(start_H_U, start_H_L, T, theta=np.pi/2, phi=0., tol=0.0001, N_points=DEFAULT_N_POINTS):

    # Larger H => +ve Delta
    # Smaller H => -ve Delta
    MAX_ITERATIONS = 20
    iterations = 0

    current_H_U = start_H_U
    current_H_L = start_H_L

    current_delta_U = delta(T, current_H_U, theta, phi, N_points)
    current_delta_L = delta(T, current_H_L, theta, phi, N_points)

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


def find_V(N_points=DEFAULT_N_POINTS, theta=np.pi/2, phi=0.):
    return 1 / Susceptibility(6.5, 0, theta=theta, phi=phi)


def range_guesser(T):
    return 40 * np.sqrt((-T + 6.5) / (T+1)**(2/3))


def find_phase_diagram(steps=25, theta=np.pi/2):
    values = []
    H = 5
    for T_index in range(steps):
        T = 6.3 * (1-(T_index) / steps) + .2
        H_upper_estimate = range_guesser(T) + 20
        H_lower_estimate = np.clip(range_guesser(T) - 25, -1, 100)

        H = braket(H_upper_estimate, 0.5,  T, theta=theta)
        values.append([H, T])
        print("[{}, {}],".format(H, T))
    return np.array(values)


def H_angle(steps, T, angle_range):
    values = []

    H_upper_estimate = 20
    H_lower_estimate = 0.001

    # theta = np.pi/2  # going from -5 -> 5 degrees

    theta_linspace = np.linspace(
        np.deg2rad(angle_range[0]), np.deg2rad(angle_range[1]), steps)

    for theta_i in theta_linspace:
        # print(theta_i)
        H = braket(H_upper_estimate, H_lower_estimate,
                   T, theta=theta_i, tol=0.0001)
        theta_i = np.rad2deg(theta_i)
        values.append([H, theta_i])
        print("[{}, {}],".format(H, theta_i))

    # print(values)

    return np.array(values)


def gl_angle_model(theta, Hc2_par, Hc2_perp):

    gamma = Hc2_par/Hc2_perp

    return Hc2_par / np.sqrt(gamma**2*(np.cos(theta))**2 + (np.sin(theta))**2)


def tinkham_angle_model(theta, Hc2_par, Hc2_perp):

    gamma = Hc2_par/Hc2_perp

    first_term = - gamma*Hc2_par * \
        abs(np.cos(theta)) / 2 / (np.sin(theta))**2

    second_term = Hc2_par*np.sqrt((2/gamma*np.tan(theta)*np.sin(theta)) **
                                  (-2) + (np.sin(theta))**(-2))

    return first_term + second_term


def plot_H_angle(t, Hc2_par=0, Hc2_perp=0, plot_fit=False):

    fig, ax = plt.subplots(dpi=400)

    ax.plot(t[:, 1], t[:, 0], 'k-', label="Upper Critical Field")
    ax.set_ylabel(r"$H$ (T)")
    ax.set_xlabel("Polar Angle " r"$\theta$ (" r"$\phi$" " = 0)")
    # ax.set_xlim((0, 7))
    # ax.set_ylim((0, 85))
    plt.legend()
    ax.set_aspect("auto")

    if plot_fit:

        theta = np.linspace(t[0, 1], t[-1, 1], 1000)

        ax.plot(theta, gl_angle_model(theta, Hc2_par, Hc2_perp),
                'm--', label="Fit to G-L Model")
        ax.plot(theta, tinkham_angle_model(theta, Hc2_par, Hc2_perp),
                'c--', label="Fit to Tinkham Model")

    return None


def perp_gl_model(T):

    flux_quantum = h/(2*e)
    gl_length = 4.8e-9 # from hc2 perp fitting
    # d = 1.5e-9
    Tc = 6.5

    factor_perp = flux_quantum / (2*np.pi*gl_length**2)
    # print(factor_perp)
    return factor_perp * (1-T/Tc)


def par_gl_model(T, a):  # d):  # will be optimised

    # flux_quantum = h/(2*e)  # in the normal model
    # gl_length = 8e-9
    # d = 1.5e-9 # from the paper, but we can optimise this
    Tc = 6.5

    # factor_par = flux_quantum*np.sqrt(12) / (2*np.pi*gl_length*d)
    # print(factor_par)
    return a * np.sqrt(1-T/Tc)


def plot_phase_diagram_fitted(r, plot_fit=False, fit_range=2):

    fig, ax = plt.subplots(figsize=(5, 5), dpi=400)

    ax.plot(r[:, 1], r[:, 0], 'k-',
            label="Phase Diagram, " r"$\Delta_{Z}$ = 13 meV")

    ax.set_ylabel(r"$H$ (T)")
    ax.set_xlabel(r"$T$ (K)")
    ax.set_xlim((0, 7))
    ax.set_ylim((0, 85))
    plt.legend(loc="upper right", fontsize=8)

    if plot_fit:

        r_red = r[:len(r)//fit_range, :len(r)//fit_range]

        params = curve_fit(par_gl_model, r_red[:, 1], r_red[:, 0])
        print(params)

        x = np.linspace(0, 6.5, 1000)

        ax.plot(x, par_gl_model(x, params[0][0]),
                'm--', label="In-Plane G-L Model")
        ax.plot(x, perp_gl_model(x), 'c--',
                label="Out-of-Plane G-L Model")

        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 6.5, 7])
        ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6, r"$T_{c}$", 7])
        # Hc2 values from optimised parameter:
        ax.set_yticks([0, params[0][0], 20, 40, 60,
                      par_gl_model(0, params[0][0]), 80])
        ax.set_yticklabels(
            [0, r"$H_{c2}^{⟂}$", 20, 40, 60, r"$H_{c2}^{∥}$", 80])

        return params

    return None


time_0 = time.time()


V = find_V()
print(V)


# print(delta(6.3, 0, np.pi/2, 0.))

# Susceptibility(6.5, 0, 0, 0, plot=True) # what is wrong with the plot?

# print(braket(65, 40, 3))

# test_N_convergence(3)
"""
theta_linspace = np.linspace(np.pi/2 - np.pi/180, np.pi/2 + np.pi/180, 10)
Hc2_vals = []
count = 1

for theta_i in theta_linspace:
    r_i = find_phase_diagram(10, theta=theta_i)
    Hc2_i = plot_phase_diagram_fitted(r_i, True, fit_range=3)[0][0]
    print("Value", count, "out of", len(theta_linspace), [theta_i, Hc2_i])
    Hc2_vals.append(Hc2_i)
    count += 1

Hc2_vals = [63.16972060803493,
 63.16972060803493,
 63.16972060803493,
 63.310322077935226,
 63.310322077935226,
 63.310322077935226,
 63.310322077935226,
 63.16972060803493,
 63.16972060803493,
 63.16972060803493]
print(Hc2_vals)

fig, ax = plt.subplots(dpi=400)

ax.plot(theta_linspace, Hc2_vals, 'k-')

Hc2_perp = perp_gl_model(0)
# Hc2_par =

ax.plot(theta_linspace, gl_angle_model(theta_linspace, Hc2_vals[len(Hc2_vals)//2], Hc2_perp), 'm--')
ax.plot(theta_linspace, tinkham_angle_model(theta_linspace, Hc2_vals[len(Hc2_vals)//2], Hc2_perp), 'c--')

for i in range(len(Hc2_vals)):
    print(Hc2_vals[i])
    ax.plot(theta_linspace, gl_angle_model(
        theta_linspace, Hc2_vals[i], Hc2_perp), 'm--')
    ax.plot(theta_linspace, tinkham_angle_model(
        theta_linspace, Hc2_vals[i], Hc2_perp), 'c--')
"""
# r = find_phase_diagram(20, theta=np.pi/2 - np.pi/36)
# print(r)


# the best plot we got
# r = np.array([[0.0, 6.5], [11.37174469824445, 6.29], [16.250100230922534, 6.08], [19.9859555318214, 5.87], [23.275395774600604, 5.66], [26.181072285375002, 5.45], [28.919673844920524, 5.24], [31.439747844127837, 5.029999999999999], [33.881668229907156, 4.82], [36.22660664836119, 4.609999999999999], [38.515948445701945, 4.4], [40.743604452325435, 4.1899999999999995], [42.92326443890704, 3.98], [45.067311769810324, 3.77], [47.20943594241787, 3.56], [49.33921965805612, 3.35], [51.46849677747653, 3.14], [
#   53.5878311541133, 2.93], [55.77682917515372, 2.72], [57.98492853853642, 2.5100000000000002], [60.251754495857355, 2.3000000000000003], [62.55411936964707, 2.0900000000000003], [64.89499190626333, 1.8800000000000003], [67.26104144416645, 1.6699999999999995], [69.65771969667891, 1.4599999999999997], [72.20027561938744, 1.2499999999999998], [75.12950264719204, 1.0399999999999998], [79.91791839509291, 0.8299999999999998], [83.92096716961888, 0.6199999999999999], [99.93546429987978, 0.4099999999999999]])

#Hc2_par = plot_phase_diagram_fitted(r, True)[0][0]

# d = plot_phase_diagram_fitted(r, True)[0][0]
#Hc2_perp = perp_gl_model(0)
#Hc2_par = par_gl_model(0, d)

t = H_angle(40, 6.4, [0, 180])  # same shape, peak is a little above 90 ?
"""
t = np.array([[2.17652344,   1.],
              [2.17652344,   5.54166667],
              [2.23507812,  10.08333333],
              [2.23507812,  14.625],
              [2.29363281,  19.16666667],
              [2.3521875,  23.70833333],
              [2.46929687,  28.25],
              [2.58640625,  32.79166667],
              [2.70351562,  37.33333333],
              [2.820625,  41.875],
              [3.05484375,  46.41666667],
              [3.2890625,  50.95833333],
              [3.58183594,  55.5],
              [3.99171875,  60.04166667],
              [4.46015625,  64.58333333],
              [4.92859375,  69.125],
              [5.63125,  73.66666667],
              [6.568125,  78.20833333],
              [7.27078125,  82.75],
              [7.9734375,  87.29166667],
              [7.9734375,  91.83333333],
              [7.505,  96.375],
              [6.68523438, 100.91666667],
              [5.86546875, 105.45833333],
              [5.1628125, 110.]])
"""
print(t)
#Hc2_par = 8.125
plot_H_angle(t)

"""
# superconducting thickness = 2.10556751e-09


# print(Susceptibility(6.5, 0, plot=True))
# plot_critical_field()
# test_freq_convergence(1500, 4000, 5)

print("Runtime: {} s".format(time.time() - time_0))

# intro to superconductivity textbook pg 321. look at model for Hc2(theta)
# derive the susceptibility using prb paper for out-of-plane H field
"""
