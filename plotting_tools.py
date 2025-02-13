# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:58:31 2025

@author: zumai
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.optimize import curve_fit

from Main import vary_ham
from DFT_tools import get_hamr, find_hamk
from GF_tools import Kubo_susceptibility_unnorm
from eig_tools import epsilon
from k_tools import get_k_path, get_k_path_spacing, get_k_block, get_better_k_square
from eig_tools import projection_x, projection_y, projection_z

DFT_FILES = ["Data/MoS2_hr.dat", "Data/DFT_H0_300_P.npy",
             "Data/DFT_H0_300_N.npy", "Data/hamk_1000.npy"]
VEL_FILES = ["Data/velocity_x_1000.npy", "Data/velocity_y_1000.npy"]
DOS_FILE = "Data/DOS_data_300_DFT.npy"
TC_FILE = "Data/TC_DFT_1000_FINAL.npy"

RESOLUTION = 1000
DEFAULT_PATH = ['G', 'M', 'K', 'G']


def plot_bands_on_path(path=DEFAULT_PATH, ef=-0.96, H=0., theta=np.pi/2, phi=0.):

    k_points = get_k_path(path, RESOLUTION)
    x = get_k_path_spacing(k_points)

    hamr, ndeg, rvec = get_hamr(DFT_FILES[0])
    hamk = find_hamk(k_points, hamr, ndeg,
                     rvec)

    hamk_pert = vary_ham(hamk, ef=ef, H=H, theta=theta, phi=phi)
    energies = epsilon(hamk_pert)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

    ax.plot(x, energies[:, 1], 'midnightblue')
    ax.plot(x, energies[:, 0], 'r')

    ax.set_ylabel("Energy (eV)")
    ax.set_xlabel("Path in Brillouin Zone")
    ax.set_xticks([x[i*(RESOLUTION-1)] for i in range(len(path))],
                  labels=path)  # not technically right but it's ok
    plt.legend(loc="upper right")

    return None


def plot_susceptibility(ef=-0.96, H=0., theta=np.pi/2, phi=0., variable_field=False):
    ham = np.load(DFT_FILES[3])
    vel_x = np.load(VEL_FILES[0])
    vel_y = np.load(VEL_FILES[1])

    k_points, res = get_better_k_square(region_res=1000)
    cut_indices = np.where(
        np.sqrt((k_points[0]-2*np.pi/3)**2 + (k_points[1]-2*np.pi/3)**2) < 0.6)[0]
    # print(len(cut_indices))

    vel_x_cut = vel_x[:, :, cut_indices]
    vel_y_cut = vel_y[:, :, cut_indices]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

    if variable_field:

        n = 25

        susc_array = np.zeros(n)
        H_array = np.linspace(0., 150, n)

        for i in range(n):

            ham_pert = vary_ham(ham, ef=ef, H=H_array[i], theta=theta, phi=phi)
            ham_pert_cut = ham_pert[:, :, cut_indices]

            susc_array[i] = Kubo_susceptibility_unnorm(
                ham_pert_cut, vel_x_cut, vel_y_cut, 6.3, 20, PLOT=False)
            print("{}".format(i))

        ax.plot(H_array, susc_array, marker='x', markersize=5)

        for j in range(n):
            print("[[{}, {}],]".format(H_array[j], susc_array[j]))

        return None

    ham_pert = vary_ham(ham, ef=ef, H=H, theta=theta, phi=phi)
    ham_pert_cut = ham_pert[:, :, cut_indices]

    susc_array = Kubo_susceptibility_unnorm(
        ham_pert_cut, vel_x_cut, vel_y_cut, 6.3, 20, PLOT=True)

    k_points_cut = k_points[:, cut_indices]
    cax = ax.scatter(k_points_cut[0], k_points_cut[1],
                     c=np.real(susc_array), cmap='bwr', norm=colors.CenteredNorm(0))
    fig.colorbar(cax, ax=ax)

    return None


def plot_energy_spectrum(cutoff=0.022, ef=-0.96, H=0., theta=np.pi/2, phi=0., plot_selected=False):
    #ham_P = np.load(DFT_FILES[1])
    #ham_N = np.load(DFT_FILES[2])

    ham = np.load(DFT_FILES[3])

    k_points = get_k_block(RESOLUTION, size_of_box=-1)

    ham_pert = vary_ham(ham, ef=ef, H=H, theta=theta, phi=phi)
    energy = epsilon(ham_pert).mean(axis=1).reshape(RESOLUTION, RESOLUTION)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=400)

    alpha = beta = np.linspace(0, 1, RESOLUTION)
    alpha, beta = np.meshgrid(alpha, beta)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

    cax = ax.pcolor(alpha, beta, np.real(energy))
    fig.colorbar(cax, label="Energy (eV)")

    sig_points = np.array([[0, 0, 1/3, 2/3], [0, 1/2, 1/3, 2/3],
                           [r"$\Gamma$", "$M$", "$K$", "$K'$"]])

    for i in range(4):
        ax.plot(float(sig_points[0, i]), float(
            sig_points[1, i]), 'ko', markersize=6)
        ax.text(float(sig_points[0, i]) + 0.01, float(
            sig_points[1, i]) + 0.01, str(sig_points[2, i]), fontsize=20)

    if plot_selected:
        significant_kpoints_indices = np.where(abs(energy) < cutoff)

        # Find -ve ham to significant k points
        significant_kpoints = k_points[:, significant_kpoints_indices][:, 0, :]

        ax.contourf(alpha, beta, energy, levels=[
            -cutoff, cutoff], colors="r", label="Valid K Points")

        plt.legend(loc="upper left")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")

    plt.show()

    return None


def plot_projections(path=DEFAULT_PATH):

    k_points = get_k_path(path, RESOLUTION)
    x = get_k_path_spacing(k_points)

    hamr, ndeg, rvec = get_hamr(DFT_FILES[0])
    hamk = find_hamk(k_points, hamr, ndeg,
                     rvec)

    hamk_pert = vary_ham(hamk, ef=-0.96, H=0., theta=np.pi/2, phi=0.)

    p_z = np.real(projection_z(hamk_pert))
    p_x = np.real(projection_x(hamk_pert))
    p_y = np.real(projection_y(hamk_pert))
    energies = epsilon(hamk_pert)

    # total_proj = p_z**2 + p_x**2 + p_y**2

    fig, axs = plt.subplots(3, figsize=(6, 9), dpi=400, sharex=True)

    norm = colors.Normalize(-1, 1)
    projs = [p_x, p_y, p_z]
    titles = ['X', 'Y', 'Z']

    for j in range(len(axs)):
        for i in range(len(energies[0])):
            axs[j].scatter(x, energies[:, i], c=projs[j][:, i],
                           cmap='bwr', norm=norm, linewidths=1)
        axs[j].set_title(titles[j])
        #plt.hlines(0, 0, 1)
        #plt.hlines(0.022, 0, 1)
        #plt.hlines(-0.022, 0, 1)

        axs[j].set_ylabel("Energy (eV)")
        plt.xlabel("Path in Brillouin Zone")
        plt.xticks([x[i*(RESOLUTION-1)] for i in range(len(path))],
                   labels=path)  # not technically right but it's ok

        #plt.colorbar(label='y value')

    return None


def BCS_critical_T(dos, v):

    return 1.134 * 262.3 * np.exp(-2/(dos * -v))


def plot_tc(ef=-0.96, v=-1.19):
    dos = np.load(DOS_FILE)
    tc = np.load(TC_FILE)

    e = tc[:, 0] - ef
    t = tc[:, 1]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

    ax.plot(e, t, 'kx', ms=5, label="Data")
    ax.set_xlabel("Fermi Energy (eV)")
    ax.set_ylabel("Critical Temperature $T_{c}$ (K)")

    tc_bcs = BCS_critical_T(dos[:, 1], v)
    tc_bcs_upper = BCS_critical_T(dos[:, 1], v+0.01)
    tc_bcs_lower = BCS_critical_T(dos[:, 1], v-0.01)

    ax.plot(dos[:, 0], tc_bcs, label='Fit to BCS Model')
    plt.fill_between(dos[:, 0], tc_bcs_upper, tc_bcs_lower, color='b',
                     alpha=0.2, label="Uncertainty in Fit")

    # smaller range
    #ax.set_xlim(-0.17, 0.12)
    #ax.set_ylim(-0.5, 15)

    plt.legend(loc="upper left", fontsize=8)

    return None


def plot_dos():
    dos = np.load(DOS_FILE)
    e = dos[:, 0]
    d = dos[:, 1]
    c = dos[:, 2]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

    ax.plot(e, d, color='tab:blue', ls='-', label="Density of States")

    plt.legend(loc="upper left", fontsize=8)

    ax1 = ax.twinx()
    ax1.plot(e, c, 'r-', label="Carrier Density")
    ax.set_ylim(0, 5.5)

    ax.set_xlabel("Fermi Energy (eV)")
    ax.set_ylabel("Density of States, $N(E)$ (eV$^{-1}$ m$^{-3}$)")
    ax1.set_ylabel(
        "Carrier Density, n(E) = $\int_{-\infty}^{E} N(E') dE'$ (m$^{-3}$)")
    ax1.set_ylim(0, 2.3)

    plt.legend(loc="upper right", fontsize=8)

    return None


def gl_model(T, a):
    Tc = 6.5
    return a * np.sqrt(1 - T/Tc)


def plot_phase_diagram(r, plot_fit=False, fit_range=2):

    fig, ax = plt.subplots(figsize=(5, 5), dpi=400)

    ax.plot(r[:, 0], r[:, 1], 'r-',
            label="Phase Diagram, In-Plane H-Field ")

    ax.set_ylabel(r"$\mu_{0} H_{c2}$ (T)")
    ax.set_xlabel(r"$T$ (K)")
    ax.set_xlim((0, 7))
    ax.set_ylim((0, 65))
    plt.legend(loc="upper right", fontsize=7)

    if plot_fit:

        r_red = r[:len(r)//fit_range, :len(r)//fit_range]

        params = curve_fit(gl_model, r_red[:, 0], r_red[:, 1])

        x = np.linspace(0, 6.5, 100)
        ax.plot(x, gl_model(x, params[0][0]),
                color='orange', linestyle='--', label="In-Plane G-L Model")

        return params

    return None
