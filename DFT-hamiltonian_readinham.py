# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:12:32 2024

@author: hbrit
"""
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
import scipy.constants
from matplotlib import colors
import time


# our files
from k_tools import get_k_path, get_k_block, get_k_path_spacing
from phase_diagram import H_0, a


ham_file = "Data/MoS2_hr.dat"
H0_P_file = "Data/DFT_H0_1000_P.npy"
H0_N_file = "Data/DFT_H0_1000_N.npy"

# constants
MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]
k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]

# system variables
DEBYE_ENERGY = 0.022  # 99990.022  # eV
FERMI_ENERGY = -0.96  # eV
START_T_L = 10
START_T_U = 50

# Default field allignment - x direction
PHI_DEFAULT = 0
THETA_DEFAULT = np.pi / 2

# Simulation settings
RESOLUTION = 1000
NUM_FREQ = 1000

# k - path settings
DEFAULT_PATH = ['G', 'K', 'G']

# Full BZ settings
PRESELECTION_BOXSIZE = -1  # 0.22  # set to -1 to use full area but, 0.22 works well


# Bracket settings
BRACKET_TOLERANCE = 1e-5
MAX_BRACKET_STEPS = 25
TEMP_START = 6.5
TEMP_STOP = 0.5
TEMP_STEPS = 15
H_U_START = 100
H_L_START = .1


fermi_levels = np.array([[-0.89, 2.5956954956054688],
                         [-0.88, 4.06298828125],
                         [-0.87, 4.653564453125],
                         [-0.86, 4.81103515625],
                         [-0.85, 4.923828125],
                         [-0.84, 5.064453125],
                         [-0.83, 5.21923828125],
                         [-0.82, 5.369824218749999],
                         [-0.81, 5.5166259765625005],
                         [-0.8, 5.6259765625],
                         [-0.79, 5.8251953125],
                         [-0.78, 5.97705078125],
                         [-0.77, 6.137939453125],
                         [-0.76, 6.3369140625],
                         [-0.75, 6.5],
                         [-0.74, 6.6009765625],
                         [-0.73, 6.7426391601562505],
                         [-0.72, 6.886572265625],
                         [-0.71, 7.02021484375],
                         [-0.7, 7.1708984375],
                         [-0.69, 7.42236328125],
                         [-0.68, 7.579833984375],
                         [-0.67, 7.7490234375],
                         [-0.66, 7.9560546875],
                         [-0.65, 7.987060546875002]])

fermi_levels_dft = np.array([[-0.84, 10.6968994140625],
                             [-0.85, 10.307025909423828],
                             [-0.86, 9.020751953125],
                             [-0.87, 8.8232421875],
                             [-0.88, 8.550195312500001],
                             [-0.89, 8.232421875],
                             [-0.9, 7.9638671875],
                             [-0.91, 7.7119140625],
                             [-0.92, 7.46826171875],
                             [-0.93, 7.08740234375],
                             [-0.94, 6.8017578125],
                             [-0.95, 6.771630859375],
                             [-0.96, 6.499957275390624],
                             [-0.97, 6.364331054687499],
                             [-0.98, 6.15185546875],
                             [-0.99, 5.97890625],
                             [-1, 5.855468749999999],
                             [-1.1, 0.07465265274047853]])

fermi_levels_dft_2 = np.array([[-0.96, 6.499],
                               [-0.95, 24.66552734375],
                               [-0.94, 55.15625],
                               [-0.93, 102.7587890625],
                               [-0.92, 143.212890625], ])


################################################################
# eigen value operations


def epsilon(hamk):
    eigs = np.array([np.real_if_close(LA.eig(hamk[:, :, i])[0])
                    for i in range(hamk.shape[2])], dtype=float)
    return eigs


def projection_z(hamk, band=0):
    eigvecs = np.array([LA.eig(hamk[:, :, i])[1]
                        for i in range(hamk.shape[2])], dtype=complex)
    # print(eigvecs.shape)

    proj = eigvecs[:, 0] * eigvecs[:, 0].conj() - \
        eigvecs[:,  1] * eigvecs[:, 1].conj()

    # print(proj[:, 0] - proj[:, 1])

    return proj


def get_eig_vec(hamk):
    return np.array([LA.eig(hamk[:, :, i])[1]
                     for i in range(hamk.shape[2])], dtype=complex)


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

    H = np.zeros((2, 2, len(kx)), dtype=complex)

    for i in range(len(kx)):
        H[:, :, i] = H_0(kx[i], ky[i])

    return H


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
    new_ham[0, 0, :] = ham[0, 0, :] - ef - H * MU_B * np.cos(theta)
    new_ham[1, 1, :] = ham[1, 1, :] - ef + H * MU_B * np.cos(theta)

    new_ham[0, 1, :] = ham[0, 1, :] - H * MU_B * \
        complex(np.cos(phi), np.sin(phi)) * np.sin(theta)
    new_ham[1, 0, :] = ham[1, 0, :] - H * MU_B * \
        complex(np.cos(phi), -np.sin(phi)) * np.sin(theta)

    # print(new_ham[0, 1, 10])

    # print(H * MU_B * complex(np.cos(phi), np.sin(phi)) * np.sin(theta))

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


def DOS(E, hamk, res=RESOLUTION, sigma=1e-2):

    ham = vary_ham(hamk)

    E_k = epsilon(ham).flatten()

    deltas = (np.exp(-(E - E_k)**2 / (2 * sigma**2)) /
              (sigma * np.sqrt(2*np.pi))).sum()

    return deltas / (RESOLUTION * RESOLUTION)


def BCS_critical_T(dos, v, E_D=DEBYE_ENERGY):

    return 1.134 * E_D * np.exp(-1/(dos * -v)) / k_B


def BCS_v(dos, td=262.3, tc=6.5):
    return 1 / (dos * np.log(tc / (1.134 * td)))


##############################################################################
# Main Utilities


def delta(T,  v,   hamk_P, hamk_N, fermi_energy=FERMI_ENERGY, H=0, phi=PHI_DEFAULT, theta=THETA_DEFAULT):

    hamk_pert_N = vary_ham(hamk_N, fermi_energy, H, theta=theta, phi=phi)

    hamk_pert_P = vary_ham(hamk_P, fermi_energy, H, theta=theta, phi=phi)

    return 1 - v * susc(hamk_pert_N, hamk_pert_P, T)


def braket(ham_P, ham_N, T, v, fermi_energy=FERMI_ENERGY, start_H_U=H_U_START, start_H_L=H_L_START,
           theta=THETA_DEFAULT, phi=PHI_DEFAULT, tol=BRACKET_TOLERANCE):

    # Larger H => +ve Delta
    # Smaller H => -ve Delta
    iterations = 0

    #theta = T
    #T = 6.4

    current_H_U = start_H_U
    current_H_L = start_H_L

    current_delta_U = delta(T, v,  ham_P, ham_N, fermi_energy,
                            H=current_H_U, theta=theta, phi=phi)
    current_delta_L = delta(T, v,  ham_P, ham_N, fermi_energy,
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
            current_delta_L = delta(
                T, v,  ham_P, ham_N, fermi_energy, H=current_H_L, theta=theta, phi=phi)

        elif current_delta_L < 0 and current_delta_U < 0:
            print("both -ve sign")

            current_H_L = current_H_U
            current_H_U = old_H_U

            # reset lower
            current_delta_L = current_delta_U
            # recalculate Upper
            current_delta_U = delta(
                T, v,  ham_P, ham_N, fermi_energy, H=current_H_U, theta=theta, phi=phi)

        elif abs(current_delta_L) > abs(current_delta_U):
            old_H_L = current_H_L
            current_H_L = (current_H_L + current_H_U) / 2
            current_delta_L = delta(
                T, v,  ham_P, ham_N, fermi_energy, H=current_H_L, theta=theta, phi=phi)

        else:
            old_H_U = current_H_U
            current_H_U = (current_H_L + current_H_U) / 2
            current_delta_U = delta(
                T, v,  ham_P, ham_N, fermi_energy, H=current_H_U, theta=theta, phi=phi)

        iterations += 1

    if iterations == MAX_BRACKET_STEPS-1:
        print("Reached max iterations")

    return [current_H_L, current_H_U]


def bracketing(ham_P, ham_N, v):
    T_array = np.linspace(TEMP_START, TEMP_STOP, TEMP_STEPS)
    H_array = []

    lower_bounds = []
    upper_bounds = []

    for t_index in range(TEMP_STEPS):
        temp_T = T_array[t_index]

        bounds = braket(ham_P.copy(), ham_N.copy(), temp_T, v)

        mean_H = (bounds[0] + bounds[1])/2

        H_array.append(mean_H)
        lower_bounds.append(bounds[0])
        upper_bounds.append(bounds[1])

        print("[{}, {}], bounds: [{}, {}],".format(
            mean_H, temp_T, bounds[0], bounds[1]))

    return T_array, np.array(H_array), np.array(lower_bounds), np.array(upper_bounds)


def fermi_level_bracketing(ham_P, ham_N, v, fermi_energy=FERMI_ENERGY, start_T_L=START_T_L, start_T_U=START_T_U,
                           theta=THETA_DEFAULT, phi=PHI_DEFAULT, tol=BRACKET_TOLERANCE):
    iterations = 0

    current_T_L = start_T_L
    current_T_U = start_T_U
    current_delta_L = delta(current_T_L, v, ham_P, ham_N,
                            H=0, theta=theta, phi=phi)
    current_delta_U = delta(current_T_U, v, ham_P, ham_N, fermi_energy=fermi_energy,
                            H=0, theta=theta, phi=phi)

    while current_delta_U < 0:
        print("Upper T too low")
        current_T_U = current_T_U + 5
        current_delta_U = delta(current_T_U, v, ham_P, ham_N, fermi_energy=fermi_energy,
                                H=0, theta=theta, phi=phi)
    while current_delta_L > 0:
        print("Lower T too high")
        current_T_L = current_T_L - 5
        current_delta_L = delta(current_T_L, v, ham_P, ham_N, fermi_energy=fermi_energy,
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
            current_delta_L = delta(current_T_L, v, ham_P, ham_N, fermi_energy=fermi_energy,
                                    H=0, theta=theta, phi=phi)

        elif current_delta_L < 0 and current_delta_U < 0:
            print("both -ve sign")

            current_T_L = current_T_U
            current_T_U = old_T_U

            # reset lower
            current_delta_L = current_delta_U
            # recalculate upper
            current_delta_U = delta(current_T_U, v, ham_P, ham_N, fermi_energy=fermi_energy,
                                    H=0, theta=theta, phi=phi)

        elif abs(current_delta_L) > abs(current_delta_U):
            old_T_L = current_T_L
            current_T_L = (current_T_L + current_T_U) / 2
            current_delta_L = delta(
                current_T_L, v, ham_P, ham_N, fermi_energy=fermi_energy, H=0, theta=theta, phi=phi)

        else:
            old_T_U = current_T_U
            current_T_U = (current_T_L + current_T_U) / 2
            current_delta_U = delta(
                current_T_U, v, ham_P, ham_N, fermi_energy=fermi_energy, H=0, theta=theta, phi=phi)

        print("[{}, {}]".format(
            current_delta_L, current_delta_U))

        iterations += 1

    if iterations == MAX_BRACKET_STEPS-1:
        print("Reached max iterations")

    return [current_T_L, current_T_U]


def find_v(useToy=False):

    preselected_kpoints = get_k_block(RESOLUTION, PRESELECTION_BOXSIZE)

    if useToy:
        hamk_P = get_toy_ham(preselected_kpoints)
    else:
        hamr, ndeg, rvec = get_hamr()  # Read in the real-space hamiltonian
        hamk_P = find_hamk(preselected_kpoints, hamr, ndeg,
                           rvec)  # FT the hamiltonian

    hamk_pert_P = vary_ham(hamk_P)  # Adjust the fermi level

    # Find the mean of energy eigen values
    energy = epsilon(hamk_pert_P).mean(axis=1)

    # Find the points within the Debye energy of fermi surface
    significant_kpoints_indices = np.where(abs(energy) < DEBYE_ENERGY)

    # Find -ve ham to significant k points
    significant_kpoints = preselected_kpoints[:,
                                              significant_kpoints_indices][:, 0, :]
    print(significant_kpoints.shape)

    if useToy:
        hamk_N = get_toy_ham(-significant_kpoints)
    else:
        hamk_N = find_hamk(-significant_kpoints, hamr, ndeg, rvec)

    # correct +ve ham to significant k points
    hamk_P = hamk_P[:, :, significant_kpoints_indices][:, :, 0]
    hamk_pert_P = vary_ham(hamk_P)  # reset +ve
    hamk_pert_N = vary_ham(hamk_N)

    v = 1/susc(hamk_pert_N, hamk_pert_P, 6.5)

    return hamk_P, hamk_N, v


def load_v(ham_P_file=H0_P_file, ham_N_file=H0_N_file):

    ham_P = np.load(ham_P_file)
    ham_N = np.load(ham_N_file)

    ham_pert_P = vary_ham(ham_P)  # Adjust the fermi level
    bz = get_k_block(RESOLUTION, PRESELECTION_BOXSIZE)

    # Find the mean of energy eigen values
    energy = epsilon(ham_pert_P).mean(axis=1)

    # Find the points within the Debye energy of fermi surface
    significant_kpoints_indices = np.where(abs(energy) < DEBYE_ENERGY)

    significant_kpoints = bz[:, significant_kpoints_indices][:, 0, :]
    print(significant_kpoints.shape)

    # correct +ve ham to significant k points
    ham_P = ham_P[:, :, significant_kpoints_indices][:, :, 0]
    ham_N = ham_N[:, :, significant_kpoints_indices][:, :, 0]
    ham_pert_P = vary_ham(ham_P)  # reset +ve
    ham_pert_N = vary_ham(ham_N)

    v = 1/susc(ham_pert_N, ham_pert_P, 6.5)

    # v = -0.3788176484109553 # full space
    # v = -0.6945318028905759  # ED = 0.122

    return ham_P, ham_N, v


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


def get_DOS(ham, energy_range=(-0.15, 1.5), energy_steps=165, save_data=False):

    density_of_state_array = []

    energies = np.linspace(energy_range[0], energy_range[1], energy_steps)
    energy_spacing = (energies[1]-energies[0])
    for e in energies:
        temp_dos = DOS(e, ham, sigma=1e-2)

        print("energy: \t {:.3e}, \t DOS: \t {:.3e}".format(e, temp_dos))

        density_of_state_array.append(temp_dos)

    density_of_state_array = np.array(density_of_state_array)
    carrier_density = np.cumsum(density_of_state_array * energy_spacing)

    if save_data:
        temp_data = np.dstack(
            (energies, density_of_state_array, carrier_density))[0]

        np.save("Data/DOS_data_{}.csv".format(RESOLUTION), temp_data)

    return energies, density_of_state_array, carrier_density


def main():

    #hamk_P, hamk_N, v = find_v(useToy=False)

    #k = get_k_path(DEFAULT_PATH, RESOLUTION)

    #hamr_obs =get_hamr()

    #hamk_P = find_hamk(k, *hamr_obs)

    # print(hamk_P.shape)
    # print(v)

    #T, H, HL, HU = bracketing(hamk_P, hamk_N, v)
    """
    fig, ax1 = plt.subplots()


    ax1.plot(energies, doss)
    ax1.set_xlabel("energy / eV")
    ax1.set_ylabel("DOS")

    ax2 = ax1.twinx()

    ax2.set_ylabel("Carrier density")
    ax2.plot(energies, carrier_density,'orange')
    #plt.vlines(0.04,0.4,0.5,label="0.04", colors='red')
    #plt.vlines(-0.15,0.0,0.1,label="-0.15", colors='purple')
    #plt.vlines(0.12,0.45,.6,label="0.12", colors='green')
    #plt.legend()
    """


def plot_phase_diagram_fitted(r, r_l, r_u, r_perp=0, plot_fit=False, fit_range=2):

    fig, ax = plt.subplots(figsize=(5, 5), dpi=400)

    errors = r.copy()
    errors = abs(r_u - r_l)

    ax.errorbar(r[:, 1], r[:, 0], errors, fmt='r-',
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
            [0, r"$H_{c2}^{âŸ‚}$", 20, 40, 60, r"$H_{c2}^{âˆ¥}$", 80])

        plt.legend(loc="upper right", fontsize=7)

        return params
    """
    plt.legend(loc="upper right", fontsize=7)

    return None


"""
r_DFT = np.array([[6.5, 0.0],
[6.071428571428571, 8.79490966796875],
[5.642857142857143, 12.48079528808594],
[5.214285714285714, 15.370968627929688],
[4.785714285714286, 17.91968688964844],
[4.357142857142858, 20.32740249633789],
[3.928571428571429, 22.712252807617187],
[3.5, 25.1832290649414],
[3.0714285714285716, 27.802067565917966],
[2.6428571428571432, 30.642699432373046],
[2.2142857142857144, 33.78362884521484],
[1.7857142857142856, 37.37805328369141],
[1.3571428571428577, 41.60355987548829],
[0.9285714285714288, 46.60267562866211],
[0.5, 54.244257354736334]])
"""

time_0 = time.time()

hamk_P, hamk_N, v = load_v()
print(v)

values = bracketing(hamk_P, hamk_N, v)

for i in range(len(values[0])):
    print("[{}, {}],".format(values[0][i], values[1][i]))

# plot_phase_diagram_fitted(*values)

# v = -1.1911438470045455 # N = 1000, Nfreq = 1000
# v = -1.256164818696094 # N = 300, Nf = 500

#T_bounds = fermi_level_bracketing(hamk_P, hamk_N, v)
#Tc = np.mean(T_bounds)
#print("[{}, {}], ".format(FERMI_ENERGY, Tc))

print("Runtime: {} s".format(time.time() - time_0))
