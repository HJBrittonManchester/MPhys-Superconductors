# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:12:32 2024

@author: hbrit
"""
import numpy as np
import scipy.linalg as LA
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.constants
import time


# our files
from k_tools import get_k_path, get_k_block, get_k_path_spacing
from phase_diagram import H_0, a
from GL_models import par_field_gl_model


ham_file = "Data/MoS2_hr.dat"

# constants
MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]
k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]

# system variables
DEBYE_ENERGY = 0.022  # 0.022  # eV
FERMI_ENERGY = -0.96  # -0.96  # eV
START_T_L = 0.5
START_T_U = 6.5
FERMI_ENERGY_RANGE = [-0.77, -0.76, -0.75, -0.74, -0.73]  # in increasing order

# Default field allignment - x direction
PHI_DEFAULT = 0
THETA_DEFAULT = np.pi / 2

# Simulation settings
RESOLUTION = 100
NUM_FREQ = 100

# k - path settings
DEFAULT_PATH = ['K', 'G', 'K']

# Full BZ settings
PRESELECTION_BOXSIZE = 0.22  # set to -1 to use full area but, 0.22 works well


# Bracket settings
BRACKET_TOLERANCE = 1e-5
MAX_BRACKET_STEPS = 30
TEMP_START = 6.5
TEMP_STOP = 0.5
TEMP_STEPS = 15
H_U_START = 100
H_L_START = .1


################################################################
# eigen value operations


def epsilon(hamk):
    eigs = np.array([np.real_if_close(LA.eig(hamk[:, :, i])[0])
                    for i in range(hamk.shape[2])], dtype=float)
    return eigs


def get_eig_vec(hamk):
    return np.array([LA.eig(hamk[:, :, i])[1]
                     for i in range(hamk.shape[2])], dtype=complex)


def projection_z(hamk, band=0):
    eigvecs = np.array([LA.eig(hamk[:, :, i])[1]
                        for i in range(hamk.shape[2])], dtype=complex)
    # print(eigvecs.shape)

    proj = eigvecs[:, 0] * eigvecs[:, 0].conj() - \
        eigvecs[:,  1] * eigvecs[:, 1].conj()

    # print(proj[:, 0] - proj[:, 1])

    return proj


def projection_x(hamk, band=0):
    eigvecs = np.array([LA.eig(hamk[:, :, i])[1]
                        for i in range(hamk.shape[2])], dtype=complex)

    proj = eigvecs[:, :, 0] * eigvecs[:, :, 1].conj() + \
        eigvecs[:, :, 1] * eigvecs[:, :, 0].conj()

    return proj


def projection_y(hamk, band=0):
    eigvecs = np.array([np.linalg.eig(hamk[:, :, i])[1]
                        for i in range(hamk.shape[2])], dtype=complex)

    proj = 1j * eigvecs[:, :, 0] * eigvecs[:, :, 1].conj() - \
        eigvecs[:, :, 1] * eigvecs[:, :, 0].conj() * 1j

    return proj

#############################################################
# hamiltonian related functions


def get_hamr():

    ndeg = np.array([])

    with open(ham_file) as f:
        f.readline()  # skip the metadata line

        nb = int(f.readline())  # number of bands
        nr = int(f.readline())  # number of lattice points to consider

        rvec = np.zeros((3, nr), dtype=float)
        hamr = np.zeros((2, 2, nr), dtype=np.complex128)

        for step in range(7):  # should calculate this number from nr
            ndeg = np.append(ndeg, np.array(
                f.readline().strip().split("    "), dtype=int))

        for ri in range(nr):
            for xi in range(nb):
                for yi in range(nb):
                    temp_data = f.readline().strip().split()

                    index_1, index_2 = int(
                        temp_data[3]) - 1,  int(temp_data[4]) - 1

                    rvec[:, ri] = np.array(temp_data[:3], dtype=float)
                    hamr[index_1, index_2, ri] = complex(float(
                        temp_data[5]), float(temp_data[6]))

    return hamr, ndeg, rvec


def find_hamk(k, hamr, ndeg, rvec):
    ham = np.zeros((2, 2, k.shape[1]), dtype=np.complex128)
    for i in range(k.shape[1]):
        for j in range(hamr.shape[2]):

            if np.linalg.norm(rvec[:, j]) > 9999:  # for debug purposes
                continue

            # Compute the phase factor
            # Ensure the correct sign in the phase
            phase = np.dot(k[:, i], rvec[:, j])

            # Add the contribution to the Hamiltonian in k-space
            ham[:, :, i] += hamr[:, :, j] * \
                complex(np.cos(phase), -np.sin(phase)) / ndeg[j]

    return ham


def get_toy_ham(b):

    kx = (b[0] + 2 * b[1]) / (a * np.sqrt(3))
    ky = b[0] / a

    ham = np.zeros((2, 2, len(kx)), dtype=complex)

    for i in range(len(kx)):
        ham[:, :, i] = H_0(kx[i], ky[i])

    return ham


def get_bands_on_path(path=DEFAULT_PATH):

    hamr, ndeg, rvec = get_hamr()  # Read in the real-space hamiltonian

    hamk = find_hamk(get_k_path(path, RESOLUTION), hamr,
                     ndeg, rvec)  # FT the hamiltonian

    hamk_pert = vary_ham(hamk)  # Adjust the fermi level

    hamk_pert_toy = get_toy_ham(get_k_path(path, RESOLUTION))

    # Find the energy eigen values
    energy_toy = epsilon(hamk_pert_toy)
    energy_real = epsilon(hamk_pert)

    return energy_real, energy_toy


def vary_ham(ham, ef=FERMI_ENERGY, H=0, theta=THETA_DEFAULT, phi=PHI_DEFAULT):

    new_ham = np.zeros_like(ham, dtype=complex)
    new_ham[0, 0, :] = ham[0, 0, :] - ef - H*MU_B*np.cos(theta)
    new_ham[1, 1, :] = ham[1, 1, :] - ef + H*MU_B*np.cos(theta)

    new_ham[0, 1, :] = ham[0, 1, :] - H*MU_B * \
        complex(np.cos(phi), -np.sin(phi)) * np.sin(theta)
    new_ham[1, 0, :] = ham[1, 0, :] - H*MU_B * \
        complex(np.cos(phi), np.sin(phi)) * np.sin(theta)

    return new_ham

##############################################################################
# Susceptibility functions


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


def susc(ham_N, ham_P, T, for_plot=False):
    chi_0 = np.zeros(ham_N.shape[2], dtype=complex)

    for m in range(-NUM_FREQ, NUM_FREQ):
        current_freq = matsubara_frequency(T, m)
        greens_N = get_greens_function(ham_N, -current_freq)
        greens_P = get_greens_function(ham_P, current_freq)

        chi_0 -= greens_P[0, 0, :] * greens_N[1, 1, :] - \
            greens_P[0, 1, :]*greens_N[1, 0, :]

    if for_plot:
        return np.real_if_close(k_B * T / (RESOLUTION**2) * chi_0, 1e-4)

    return np.real_if_close(k_B * T / (RESOLUTION**2) * np.sum(chi_0), 1e-4)


##############################################################################
# DOS functions

def DOS(E, hamk, res=RESOLUTION):

    ham = vary_ham(hamk)

    E_k = epsilon(ham).flatten()

    print(E_k)

    sigma = 1e-3

    deltas = (np.exp(-(E - E_k)**2 / (2 * sigma**2)) /
              (sigma * np.sqrt(2*np.pi))).sum()

    return deltas / (RESOLUTION * RESOLUTION)


##############################################################################
# Main Utilities


def delta(T, v, hamk_P, hamk_N, ef=FERMI_ENERGY, H=0, phi=PHI_DEFAULT, theta=THETA_DEFAULT):

    hamk_pert_N = vary_ham(hamk_N, ef=ef, H=H, theta=theta, phi=phi)
    hamk_pert_P = vary_ham(hamk_P, ef=ef, H=H, theta=theta, phi=phi)

    return 1 - v * susc(hamk_pert_N, hamk_pert_P, T)


def braket(ham_P, ham_N, T, v, ef=FERMI_ENERGY, start_H_U=H_U_START, start_H_L=H_L_START,
           theta=THETA_DEFAULT, phi=PHI_DEFAULT, tol=BRACKET_TOLERANCE):

    # Larger H => +ve Delta
    # Smaller H => -ve Delta
    iterations = 0

    #theta = T
    #T = 6.4

    current_H_U = start_H_U
    current_H_L = start_H_L

    current_delta_U = delta(T, v,  ham_P, ham_N, ef=ef,
                            H=current_H_U, theta=theta, phi=phi)
    current_delta_L = delta(T, v,  ham_P, ham_N, ef=ef,
                            H=current_H_L, theta=theta, phi=phi)

    # print("Î”_max = {}, Î”_min = {}".format(
    #    current_delta_U, current_delta_L))

    if current_delta_U < 0:
        print("Upper H too low")
        return [start_H_L, start_H_U]
    elif current_delta_L > 0:
        print("Lower H too high")
        return [0, 0]

    old_H_U = 100
    old_H_L = 0

    while abs(current_delta_L) > tol and abs(current_delta_U) > tol and iterations < MAX_BRACKET_STEPS:
        # print("Î”_max = {}, Î”_min = {}".format(
        #    current_delta_U, current_delta_L))

        if current_delta_L > 0 and current_delta_U > 0:
            print("both +ve sign")

            current_H_U = current_H_L
            current_H_L = old_H_L

            # reset upper
            current_delta_U = current_delta_L
            # recalculate lower
            current_delta_L = delta(T, v,  ham_P, ham_N, ef=ef,
                                    H=current_H_L, theta=theta, phi=phi)

        elif current_delta_L < 0 and current_delta_U < 0:
            print("both -ve sign")

            current_H_L = current_H_U
            current_H_U = old_H_U

            # reset lower
            current_delta_L = current_delta_U
            # recalculate upper
            current_delta_U = delta(
                T, v,  ham_P, ham_N, ef=ef, H=current_H_U, theta=theta, phi=phi)

        elif abs(current_delta_L) > abs(current_delta_U):
            old_H_L = current_H_L
            current_H_L = (current_H_L + current_H_U) / 2
            current_delta_L = delta(
                T, v,  ham_P, ham_N, ef=ef, H=current_H_L, theta=theta, phi=phi)

        else:
            old_H_U = current_H_U
            current_H_U = (current_H_L + current_H_U) / 2
            current_delta_U = delta(
                T, v,  ham_P, ham_N, ef=ef, H=current_H_U, theta=theta, phi=phi)

        iterations += 1

    if iterations == MAX_BRACKET_STEPS-1:
        print("Reached max iterations")

    return [current_H_L, current_H_U]


def bracketing(ham_P, ham_N, v, ef=FERMI_ENERGY):
    # print(ef)
    #T_smallarray = np.linspace(7, 5, TEMP_STEPS)
    #T_largearray = np.linspace(TEMP_START, TEMP_STOP, TEMP_STEPS)
    #T_array = np.append(T_smallarray, T_largearray)
    T_array = np.linspace(TEMP_START, TEMP_STOP, TEMP_STEPS)
    H_array = []
    lower_bounds = []
    upper_bounds = []

    for t_index in range(TEMP_STEPS):
        temp_T = T_array[t_index]

        bounds = braket(ham_P.copy(), ham_N.copy(), temp_T, v, ef=ef)

        mean_H = np.mean(bounds)

        H_array.append([mean_H])
        lower_bounds.append([bounds[0]])
        upper_bounds.append([bounds[1]])

        print("[{}, {}], bounds: [{}, {}],".format(
            mean_H, temp_T, bounds[0], bounds[1]))

    return T_array, np.array(H_array), np.array(lower_bounds), np.array(upper_bounds)


def fermi_level_bracketing(ham_P, ham_N, v, ef=FERMI_ENERGY, start_T_L=START_T_L, start_T_U=START_T_U,
                           theta=THETA_DEFAULT, phi=PHI_DEFAULT, tol=BRACKET_TOLERANCE):
    iterations = 0

    current_T_L = start_T_L
    current_T_U = start_T_U
    current_delta_L = delta(current_T_L, v, ham_P, ham_N, ef=ef,
                            H=0, theta=theta, phi=phi)
    current_delta_U = delta(current_T_U, v, ham_P, ham_N, ef=ef,
                            H=0, theta=theta, phi=phi)
    old_T_L = START_T_L - .1
    old_T_U = START_T_U - .1

    while abs(current_delta_L) > tol and abs(current_delta_U) > tol and iterations < MAX_BRACKET_STEPS:

        if current_delta_L > 0 and current_delta_U > 0:
            print("both +ve sign")

            current_T_U = current_T_L
            current_T_L = old_T_L

            # reset upper
            current_delta_U = current_delta_L
            # recalculate lower
            current_delta_L = delta(current_T_L, v, ham_P, ham_N, ef=ef,
                                    H=0, theta=theta, phi=phi)

        elif current_delta_L < 0 and current_delta_U < 0:
            print("both -ve sign")

            current_T_L = current_T_U
            current_T_U = old_T_U

            # reset lower
            current_delta_L = current_delta_U
            # recalculate upper
            current_delta_U = delta(current_T_U, v, ham_P, ham_N, ef=ef,
                                    H=0, theta=theta, phi=phi)

        elif abs(current_delta_L) > abs(current_delta_U):
            old_T_L = current_T_L
            current_T_L = (current_T_L + current_T_U) / 2
            current_delta_L = delta(
                current_T_L, v, ham_P, ham_N, ef=ef, H=0, theta=theta, phi=phi)

        else:
            old_T_U = current_T_U
            current_T_U = (current_T_L + current_T_U) / 2
            current_delta_U = delta(
                current_T_U, v, ham_P, ham_N, ef=ef, H=0, theta=theta, phi=phi)

        # print("[{}, {}]".format(
         #   current_delta_L, current_delta_U))

        iterations += 1

    if iterations == MAX_BRACKET_STEPS-1:
        print("Reached max iterations")

    return [current_T_L, current_T_U]


def find_v(ef=FERMI_ENERGY, useToy=False):

    preselected_kpoints = get_k_block(RESOLUTION, PRESELECTION_BOXSIZE)

    if useToy:
        hamk_P = get_toy_ham(preselected_kpoints)
    else:
        hamr, ndeg, rvec = get_hamr()  # Read in the real-space hamiltonian
        hamk_P = find_hamk(preselected_kpoints, hamr, ndeg,
                           rvec)  # FT the hamiltonian

    hamk_pert_P = vary_ham(hamk_P, ef=ef)  # Adjust the parameters

    # Find the mean of energy eigen values
    energy = epsilon(hamk_pert_P).mean(axis=1)

    # Find the points within the Debye energy of fermi surface
    significant_kpoints_indices = np.where(abs(energy) < DEBYE_ENERGY)

    # Find -ve ham to significant k points
    significant_kpoints = preselected_kpoints[:,
                                              significant_kpoints_indices][:, 0, :]

    if useToy:
        hamk_N = get_toy_ham(-significant_kpoints)
    else:
        hamk_N = find_hamk(-significant_kpoints, hamr, ndeg, rvec)

    # correct +ve ham to significant k points
    hamk_P = hamk_P[:, :, significant_kpoints_indices][:, :, 0]
    hamk_pert_P = vary_ham(hamk_P, ef=ef)  # reset +ve
    hamk_pert_N = vary_ham(hamk_N, ef=ef)

    v = 1/susc(hamk_pert_N, hamk_pert_P, 6.5)

    return hamk_P, hamk_N, v


##############################################################################
# For plotting

def plot_phase_diagram(T, H, H_L, H_U, plot_fit=False, fit_range=2):

    fig, ax = plt.subplots(figsize=(5, 5), dpi=400)

    errors = H.copy()
    errors[:, 0] = abs(H_U[:, 0] - H_L[:, 0])

    ax.errorbar(T, H[:, 0], errors[:, 0], fmt='r-',
                label="Phase Diagram, In-Plane H-Field ")

    ax.set_ylabel(r"$\mu_{0} H_{c2}$ (T)")
    ax.set_xlabel(r"$T$ (K)")
    ax.set_xlim((0, 7))
    ax.set_ylim((0, 85))

    if plot_fit:

        T_red = T[:len(T)//fit_range]
        H_red = H[:len(H)//fit_range]

        params = curve_fit(par_field_gl_model, T_red, H_red)
        print(params)

        x = np.linspace(0, T[0], 1000)

        ax.plot(x, par_field_gl_model(x, params[0][0]),
                'm--', label="In-Plane G-L Model")

        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 6.5, 7])
        ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6, r"$T_{c}$", 7])
        # Hc2 values from optimised parameter:
        ax.set_yticks([0, 20, 40, 60, params[0][0], 80])
        ax.set_yticklabels(
            [0, 20, 40, 60, r"$H_{c2}^{âˆ¥}$", 80])

        plt.legend(loc="upper right", fontsize=7)

        return params

    plt.legend(loc="upper right", fontsize=7)

    return None


def plot_projections(path=DEFAULT_PATH, res=RESOLUTION):
    hamr_obs = get_hamr()
    k = get_k_path(path, res)
    hamk = find_hamk(k, *hamr_obs)
    hamk = vary_ham(hamk, H=0, theta=0)

    p_z = np.real(projection_z(hamk))
    p_x = np.real(projection_x(hamk))
    p_y = np.real(projection_y(hamk))

    # total_proj = p_z**2 + p_x**2 + p_y**2

    print((p_z).mean(axis=0))

    # e_dft, e_t = get_bands_on_path()
    e_dft = epsilon(hamk)

    xk = get_k_path_spacing(k)

    fig, axs = plt.subplots(3, figsize=(10, 15), dpi=100, sharex=True)

    norm = colors.Normalize(-1, 1)
    projs = [p_x, p_y, p_z]
    titles = ['x', 'y', 'z']

    for j in range(len(axs)):

        for i in range(len(e_dft[0])):

            axs[j].scatter(xk, e_dft[:, i], c=projs[j][:, i],
                           cmap='bwr', norm=norm, linewidths=1)
            axs[j].set_title(titles[j])
            # plt.plot(xk, e_t)
            # plt.hlines(0, 0, 1)
            # plt.hlines(0.022, 0, 1)
            # plt.hlines(-0.022, 0, 1)

            # Add colorbar to indicate the values of z

            # plt.xlabel("Distance along k-path")
            # plt.ylabel("Energy / eV")

            # plt.title("Energy bands for No field ")
    # plt.xlim(0.3, .4)
    #plt.colorbar(scatter, label='y value')
    print(e_dft.min())


"""
def main():

    hamk_P, hamk_N, v = find_v()
    print(v)

    T, H, HL, HU = bracketing(hamk_P, hamk_N, v)
    print(T)

    return None

main()
"""

time_0 = time.time()

hamk_P, hamk_N, v = find_v()
#eig_P = epsilon(hamk_P)
#eig_N = epsilon(hamk_N)
# print(eig_P.shape)
#k = get_k_path(DEFAULT_PATH, RESOLUTION)
#plt.plot(k, eig_P)
#plt.plot(k, eig_N)
# plt.plot(eig_P)
#v = -0.8375575503135118
print(v)
# v = -0.8280025754528594  # N = 1000, Nfreq = 1000
# v = -0.8148490911693423  # N = 200, Nfreq = 500
# v = -0.8307493217674475  # N = 1200, Nfreq = 1000
"""
T_bounds = fermi_level_bracketing(
    hamk_P, hamk_N, v, start_T_L=START_T_L, start_T_U=START_T_U)
T_crit = np.mean(T_bounds)
print("[{}, {}], ".format(FERMI_ENERGY, T_crit))
"""
"""
r = np.zeros((len(FERMI_ENERGY_RANGE), TEMP_STEPS, 2))
fig, ax = plt.subplots(figsize=(5, 5), dpi=400)

for i in range(len(FERMI_ENERGY_RANGE)):
    print(str(i+1) + " out of " + str(len(FERMI_ENERGY_RANGE)))
    # = FERMI_ENERGY_RANGE[i]
    hamk_P, hamk_N, v = find_v(ef=FERMI_ENERGY_RANGE[i])
    v = -0.814849091169342  # fix v
    # , ef=FERMI_ENERGY_RANGE[i])
    values_i = bracketing(hamk_P, hamk_N, v, ef=FERMI_ENERGY_RANGE[i])

    for j in range(TEMP_STEPS):
        r[i][j, :] = [values_i[0][j], values_i[1][j][0]]
        print("[{}, {}],".format(r[i][j, 0], r[i][j, 1]))

    ax.plot(r[i][:, 0], r[i][:, 1],
            label=r"$E_{F}$ = " + str(0.01*i-max(FERMI_ENERGY_RANGE)) + " eV")

    ax.set_ylabel(r"$\mu_{0} H_{c2}$ (T)")
    ax.set_xlabel(r"$T$ (K)")
    ax.set_xlim((0, 7))
    ax.set_ylim((0, 85))
    plt.legend(loc="upper right", fontsize=8)
"""

"""
values = bracketing(hamk_P, hamk_N, v)
plot_phase_diagram(*values)

for i in range(len(values[0])):
    print("[{}, {}],".format(
        values[0][i], values[1][i][0]))
"""
print("Runtime: {} s".format(time.time() - time_0))
