# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:58:31 2025

@author: zumai
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.optimize import curve_fit
import time

from Main import vary_ham, find_v, delta
from DFT_tools import get_hamr, find_hamk
from GF_tools import Kubo_susceptibility_unnorm, susc
from eig_tools import diagonalise
from k_tools import get_k_path, get_k_path_spacing, get_k_block, get_better_k_square, get_close_k_points
from eig_tools import projection_x, projection_y, projection_z

# DFT_FILES = ["Data/MoS2_hr.dat", "Data/DFT_H0_300_P.npy",
#            "Data/DFT_H0_300_N.npy", "Data/hamk_1000.npy"]
VEL_FILES = ["Data/velocity_x_1000.npy", "Data/velocity_y_1000.npy"]
DFT_FILES = ["Data/MoS2_hr.dat", "Data/DFT_H0_300_P.npy",
             "Data/DFT_H0_300_N.npy", "Data/ham_h_2k_1000.npy", "Data/hamk_1000.npy"]
#VEL_FILES = ["Data/vx_2k_1000.npy", "Data/vy_2k_1000.npy"]
CUT_FILES = ["temp/ham_h_9689.npy", "temp/vx_9689.npy", "temp/vy_9689.npy"]
DOS_FILE = "Data/DOS_data_300_DFT.npy"
TC_FILE = "Data/TC_DFT_1000_FINAL.npy"

RESOLUTION = 1000
NUM_FREQ = 20
DEFAULT_PATH = ['G', 'M', 'K', 'G']


def plot_bands_on_path(path=DEFAULT_PATH, ef=-0.96, H=0., theta=np.pi/2, phi=0.):

    k_points = get_k_path(path, RESOLUTION)
    x = get_k_path_spacing(k_points)

    #hamr, ndeg, rvec = get_hamr(np.load(DFT_FILES[0]))
    # hamk = find_hamk(k_points, hamr, ndeg,
    #                rvec)

    hamk = np.load(DFT_FILES[4])

    hamk_pert = vary_ham(hamk, ef=ef, H=H, theta=theta, phi=phi)
    energies = diagonalise(hamk_pert)[0]
    print(energies.shape)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

    ax.plot(x, energies[1, :], 'midnightblue')
    ax.plot(x, energies[0, :], 'r')

    ax.set_ylabel("Energy (eV)")
    ax.set_xlabel("Path in Brillouin Zone")
    ax.set_xticks([x[i*(RESOLUTION-1)] for i in range(len(path))],
                  labels=path)  # not technically right but it's ok
    plt.legend(loc="upper right")

    return None


def vary_susceptibility(n, m, H_U, H_L, T_U, T_L, ef=0., H=0., theta=np.pi/2, phi=0.):
    # n = number of temperatures
    # m = number of H fields
    n = int(n)
    m = int(m)

    # these files are already corrected for fermi energy
    ham = np.load(CUT_FILES[0])
    #hamk_p = np.load(DFT_FILES[1])
    #hamk_n = np.load(DFT_FILES[2])
    vel_x = np.load(CUT_FILES[1])
    vel_y = np.load(CUT_FILES[2])

    # chi = Kubo_susceptibility_unnorm(
    #   ham, vel_x, vel_y, 6.5, NUM_FREQ, PLOT=False) / RESOLUTION**2
    #v = 1/np.real(chi)
    v = 3.184727158924242e-08  # num freq = 20
    # v = 3.175099392550576e-08  # num freq = 500
    # print(v)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

    T_array = np.linspace(T_U, T_L, int(n))
    H_array = np.linspace(H_L, H_U, int(m))
    susc_array = np.zeros((n, m))

    for i in range(n):
        for j in range(m):

            ham_pert = vary_ham(
                ham, ef=ef, H=H_array[j], theta=theta, phi=phi)

            susc_array[i][j] = np.real(Kubo_susceptibility_unnorm(
                ham_pert, vel_x, vel_y, T_array[i], NUM_FREQ, PLOT=False)) / RESOLUTION**2
            print("index: ({},{})".format(i, j))
            print("index: ({},{})".format(i, j))

    for i in range(n):
        print("\nT = {}".format(T_array[i]))
        for j in range(m):
            print("[{}, {}],".format(H_array[j], susc_array[i][j]))

        ax.plot(H_array, 1-v*susc_array[i], marker='x',
                markersize=5, label="T = {:.2f} K".format(T_array[i]))

    ax.set_xlabel("Magnetic Field (T)")
    ax.set_ylabel(r"$\Delta = 1 - V \chi$")

    ax.axhline(ls='--', c='k')

    #ax.set_xlim(0, 102)
    plt.legend(bbox_to_anchor=(1, 1), fontsize=8)

    return susc_array


def plot_susceptibility(ef=0, H=0., theta=np.pi/2, phi=0.):
    ham = np.load(DFT_FILES[3])
    vel_x = np.load(VEL_FILES[0])
    vel_y = np.load(VEL_FILES[1])

    k_points = get_close_k_points(RESOLUTION, thresholds=(0.68, 0.8))
    k_prime_points = get_close_k_points(RESOLUTION, centre=(2/3, 2/3),
                                        thresholds=(0.68, 0.8))
    k = np.hstack((k_points, k_prime_points))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

    ham_pert = vary_ham(ham, ef=ef, H=H, theta=theta, phi=phi)

    susc_array = Kubo_susceptibility_unnorm(
        ham_pert, vel_x, vel_y, 6.3, NUM_FREQ, PLOT=True)

    cax = ax.scatter(k[0], k[1],
                     c=np.real(susc_array), cmap='bwr', norm=colors.CenteredNorm(0))
    fig.colorbar(cax, ax=ax, label="Susceptibility")

    ax.set_xlabel("kx")
    ax.set_ylabel("ky")

    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)

    return None


def plot_energy_spectrum(cutoff=0.022, ef=-0.96, H=0., theta=np.pi/2, phi=0., plot_selected=False):
    #ham_P = np.load(DFT_FILES[1])
    #ham_N = np.load(DFT_FILES[2])

    ham = np.load(DFT_FILES[3])

    vx = np.load(VEL_FILES[0])

    ham_pert = vary_ham(ham, ef=ef, H=H, theta=theta, phi=phi)
    # energy = diagonalise(ham_pert)[0].mean(
    #   axis=0).reshape((RESOLUTION, RESOLUTION))

    vel = diagonalise(vx)[0][0].reshape((RESOLUTION, RESOLUTION))

    fig, ax = plt.subplots(figsize=(7, 6), dpi=400)

    alpha = beta = np.linspace(0, 1, RESOLUTION)
    alpha, beta = np.meshgrid(alpha, beta)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

    cax = ax.pcolor(alpha, beta, np.real(vel))
    fig.colorbar(cax)  # , label="Energy (eV)")

    sig_points = np.array([[0, 0, 1/3, 2/3], [0, 1/2, 1/3, 2/3],
                           [r"$\Gamma$", "$M$", "$K$", "$K'$"]])

    for i in range(4):
        ax.plot(float(sig_points[0, i]), float(
            sig_points[1, i]), 'ko', markersize=6)
        ax.text(float(sig_points[0, i]) + 0.01, float(
            sig_points[1, i]) + 0.01, str(sig_points[2, i]), fontsize=20)

    if plot_selected:
        #significant_kpoints_indices = np.where(abs(energy) < cutoff)

        # Find -ve ham to significant k points
        #significant_kpoints = k_points[:, significant_kpoints_indices][:, 0, :]

        # ax.contourf(alpha, beta, energy, levels=[
        #   -cutoff, cutoff], colors="r", label="Valid K Points")

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
    energies = diagonalise(hamk_pert)[0]

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


t_0 = time.time()

#s = vary_susceptibility(n=3, m=25, T_U=2, T_L=1, H_L=0, H_U=50, theta=0.)
#np.save("Data/susceptibility_T_5_65_oldsuscformula.npy", s)

# plot_energy_spectrum()

t = time.time()
print("\nruntime: {} seconds".format(t - t_0))
