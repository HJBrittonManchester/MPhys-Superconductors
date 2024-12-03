# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:33:42 2024

@author: w10372hb
"""

import numpy as np
import scipy.constants
import matplotlib.pyplot as plt

h = scipy.constants.h
e = scipy.constants.elementary_charge


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


def perp_field_gl_model(T, a):

    flux_quantum = h/(2*e)
    gl_length = 4.8e-9  # from hc2 perp fitting
    # d = 1.5e-9
    Tc = 6.5

    factor_perp = flux_quantum / (2*np.pi*gl_length**2)
    # print(factor_perp)
    return a * (1-T/Tc)


def par_field_gl_model(T, a):  # d):  # will be optimised

    # flux_quantum = h/(2*e)  # in the normal model
    # gl_length = 8e-9
    # d = 1.5e-9 # from the paper, but we can optimise this
    Tc = 6.5

    # factor_par = flux_quantum*np.sqrt(12) / (2*np.pi*gl_length*d)
    # print(factor_par)
    return a * np.sqrt(1-T/Tc)


def plot_H_angle(t, t_l, t_u, Hc2_par=0, Hc2_perp=0, plot_fit=False):

    fig, ax = plt.subplots(figsize=(5, 5), dpi=400)
    errors = t.copy()
    errors[:, 0] = t_u[:, 0] - t_l[:, 0]
    # print(errors)
    #ax.plot(t[:, 1], t[:, 0], 'k-', label="Simulation")
    ax.errorbar(t[:, 1], t[:, 0], errors[:, 0], fmt='k-', label="Simulation")
    ax.set_ylabel(r"$\mu_{0}$ $H_{c2}$ (T)")
    ax.set_xlabel("Polar Angle " r"$\theta$ (°) (" r"$\phi$" " = 0)")
    ax.set_xlim((89.9, 90.1))
    ax.set_xticks([89.9, 89.95, 90, 90.05, 90.1])
    #ax.set_ylim((68, 75))

    if plot_fit:

        theta = np.linspace(t[0, 1], t[-1, 1], 1000)

        ax.plot(theta, gl_angle_model(np.deg2rad(theta), Hc2_par, Hc2_perp),
                'm--', label="Ginzburg-Landau Model")
        # ax.plot(theta, tinkham_angle_model(np.deg2rad(theta), Hc2_par, Hc2_perp),
        #       'c--', label="Tinkham Model")

    plt.legend(loc="upper left", fontsize=7)

    return None


def plot_phase_diagram_fitted(r, r_l, r_u, r_perp=0, plot_fit=False, fit_range=2):

    fig, ax = plt.subplots(figsize=(5, 5), dpi=400)

    errors = r.copy()
    errors[:, 0] = abs(r_u[:, 0] - r_l[:, 0])

    ax.errorbar(r[:, 1], r[:, 0], errors[:, 0], fmt='r-',
                label="Phase Diagram, In-Plane H-Field ")
    # ax.plot(r_perp[:, 1], r_perp[:, 0], 'b-',
    #       label="Phase Diagram, Out-of-Plane H-Field ")

    ax.set_ylabel(r"$\mu_{0} H_{c2}$ (T)")
    ax.set_xlabel(r"$T$ (K)")
    ax.set_xlim((0, 7))
    ax.set_ylim((0, 85))
    """
    if plot_fit:

        r_red = r[:len(r)//fit_range, :len(r)//fit_range]

        params = curve_fit(par_gl_model, r_red[:, 1], r_red[:, 0])
        params_perp = curve_fit(par_gl_model, r_perp[:, 1], r_perp[:, 0])
        print(params, params_perp)

        x = np.linspace(0, 6.5, 1000)

        Hc2_perp = 8

        ax.plot(x, par_gl_model(x, params[0][0]),
                'm--', label="In-Plane G-L Model")
        ax.plot(x, perp_gl_model(x, params_perp[0][0]), 'c--',
                label="Out-of-Plane G-L Model")

        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 6.5, 7])
        ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6, r"$T_{c}$", 7])
        # Hc2 values from optimised parameter:
        ax.set_yticks([0, params_perp[0][0], 20, 40, 60, params[0][0], 80])
        ax.set_yticklabels(
            [0, r"$H_{c2}^{⟂}$", 20, 40, 60, r"$H_{c2}^{∥}$", 80])

        plt.legend(loc="upper right", fontsize=7)

        return params
    """
    plt.legend(loc="upper right", fontsize=7)

    return None


"""
par = 70
perp = 5
rnge = np.pi/36

fig, ax = plt.subplots(figsize=(6, 5), dpi=400)
x = np.linspace(np.pi/2 - rnge, np.pi/2 + rnge, 1000)
gl_3d = gl_angle_model(x, par, perp)
gl_2d = tinkham_angle_model(x, par, perp)

ax.plot(x, gl_3d, 'r-', label="3D Ginzburg-Landau Model")
ax.plot(x, gl_2d, 'b-', label="2D Tinkham Model")

ax.set_xlabel("Angle " r"${\theta}$" " (°)")
ax.set_ylabel("$\mu_{0} H_{c2} " r"(\theta)$ (T)")

ax.set_xlim(np.pi/2 - rnge, np.pi/2 + rnge)
ax.set_xticks([np.pi/2 - rnge, np.pi/2 - rnge/2, np.pi /
              2, np.pi/2 + rnge/2, np.pi/2 + rnge])
ax.set_xticklabels(np.rad2deg([np.pi/2 - rnge, np.pi/2 - rnge/2, np.pi /
                               2, np.pi/2 + rnge/2, np.pi/2 + rnge]))
ax.set_yticks([70])
ax.set_yticklabels([r"$H_{c2}^{∥}$"])

plt.legend(loc="upper left", fontsize=7)
"""
