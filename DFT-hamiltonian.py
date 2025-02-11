# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:12:32 2024

@author: hbrit
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from matplotlib import colors

import time


## our files
from k_tools import get_k_path, get_k_block, get_k_path_spacing, get_better_k_square
from eig_tools import epsilon, projection_z, projection_x, projection_y , get_eig_vec
from DFT_tools import get_hamr, find_hamk, find_hamk_a, find_Berry_connection, find_pos_operator
from GF_tools import get_greens_function, matsubara_frequency, susc, Kubo_susceptibility_unnorm
from phase_diagram import H_0, a

from braket import range_of_brakets


DFT_HAM_FILE_NAME = r"Data/MoS2_hr.dat"
DFT_POS_FILE_NAME = r"Data/pos_wr.dat"

# constants
MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]
k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]

# system variables
DEBYE_TEMP = 262.3 # K
DEBYE_ENERGY = 0.022  # eV
FERMI_ENERGY = -0.96  # eV

# Default field allignment - x direction
PHI_DEFAULT = 0
THETA_DEFAULT = np.pi / 2

# Simulation settings
RESOLUTION = 10
NUM_FREQ = 20

# k - path settings
DEFAULT_PATH = ['G', 'M', 'K','G']

# Full BZ settings
PRESELECTION_BOXSIZE = -1 # 0.32  # set to -1 to use full area but, 0.22 works well
FULL_RES = 0

# Bracket settings
BRACKET_TOLERANCE = 1e-6
MAX_BRACKET_STEPS = 25
TEMP_START = 6.5
TEMP_STOP = .5
TEMP_STEPS = 20
H_U_START = 100
H_L_START = -1



#############################################################
# hamiltonian related functions


def get_toy_ham(b):

    kx = (b[0] + 2 * b[1]) / (a * np.sqrt(3))
    ky = b[0] / a

    H = np.zeros((2, 2, len(kx)), dtype=complex)

    for i in range(len(kx)):
        H[:, :, i] = H_0(kx[i], ky[i])

    return H


def get_bands_on_path(path=DEFAULT_PATH):

    hamr, ndeg, rvec = get_hamr()  # Read in the real-space hamiltonian

    hamk = find_hamk(get_k_path(path, RESOLUTION), hamr, ndeg, rvec)  # FT the hamiltonian

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
# DOS functions

def DOS(E, hamk, res=RESOLUTION, sigma = 1e-2):

    ham = vary_ham(hamk)

    E_k = epsilon(ham).flatten()


    deltas = (np.exp(-(E - E_k)**2 /(2* sigma**2) ) / (sigma * np.sqrt(2*np.pi))).sum()


    return deltas / (RESOLUTION * RESOLUTION)

def BCS_critical_T(dos, v, td = DEBYE_TEMP):

    return 1.134 * td * np.exp(-1/(dos * -v))

def BCS_v(dos, td = DEBYE_TEMP, tc=6.5):
    return 1 / (dos * np.log(tc / (1.134 * td)))




##############################################################################
# Main Utilities


def delta(T,  v,   hamk_P, hamk_N, fermi_energy = FERMI_ENERGY, H=0, phi=PHI_DEFAULT, theta=THETA_DEFAULT):
    hamk_pert_N = vary_ham(hamk_N, fermi_energy, H, theta=theta, phi=phi)

    hamk_pert_P = vary_ham(hamk_P, fermi_energy, H, theta=theta, phi=phi)

    return 1 - v * np.real_if_close(susc(hamk_pert_N, hamk_pert_P, T, NUM_FREQ).sum() / RESOLUTION**2,)


def delta_kubo(T,  v,   ham, v_x, v_y, fermi_energy = FERMI_ENERGY, H=0, phi=PHI_DEFAULT, theta=THETA_DEFAULT):

    ham_pert = vary_ham(ham, fermi_energy, H, theta=theta, phi=phi)

    return 1- v* Kubo_susceptibility_unnorm(ham_pert, v_x, v_y, T, NUM_FREQ) / RESOLUTION**2



def find_v(useToy=False):
    global FULL_RES
    #hamk_P = np.load("Data/DFT_H0_300_P.npy")


    preselected_kpoints, FULL_RES = get_better_k_square(RESOLUTION, (1/3,1/3), 0.21) #get_k_block(RESOLUTION, PRESELECTION_BOXSIZE)

    if useToy:
        hamk_P = get_toy_ham(preselected_kpoints)
    else:
        hamr, ndeg, rvec, nb, nr = get_hamr(DFT_HAM_FILE_NAME)  # Read in the real-space hamiltonian
        hamk_P = find_hamk(preselected_kpoints, hamr, ndeg, rvec)  # FT the hamiltonian


    hamk_pert_P = vary_ham(hamk_P)  # Adjust the fermi level

    # Find the mean of energy eigen values
    energy = epsilon(hamk_pert_P).mean(axis=1)

    # Find the points within the Debye energy of fermi surface
    significant_kpoints_indices = np.where(abs(energy) < DEBYE_ENERGY)

    # Find -ve ham to significant k points
    significant_kpoints = preselected_kpoints[:, significant_kpoints_indices][:, 0, :]



    if useToy:
        hamk_N = get_toy_ham(-significant_kpoints)
    else:
        hamk_N = find_hamk(-significant_kpoints, hamr, ndeg, rvec)


    #hamk_N = np.load("Data/DFT_H0_300_N.npy")
    #hamk_N = hamk_N[:, :, significant_kpoints_indices][:, :, 0]


    # correct +ve ham to significant k points
    hamk_P = hamk_P[:, :, significant_kpoints_indices][:, :, 0]

    hamk_pert_P = vary_ham(hamk_P)  # reset +ve
    hamk_pert_N = vary_ham(hamk_N)

    print(hamk_pert_N.shape)

    v = 1/(susc(hamk_pert_N, hamk_pert_P, 6.5, NUM_FREQ).sum() / FULL_RES**2)

    return hamk_P, hamk_N, v

def plot_projections(path = DEFAULT_PATH, res =RESOLUTION):
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


def get_DOS(ham, energy_range = (-0.15,1.5), energy_steps = 165, save_data=False):

    density_of_state_array = []


    energies = np.linspace(energy_range[0], energy_range[1], energy_steps)
    energy_spacing = (energies[1]-energies[0])
    for e in energies:
        temp_dos = DOS(e, ham, sigma = 1e-2)

        print("energy: \t {:.3e}, \t DOS: \t {:.3e}".format(e,temp_dos))

        density_of_state_array.append(temp_dos)

    density_of_state_array = np.array(density_of_state_array)
    carrier_density = np.cumsum(density_of_state_array * energy_spacing)

    if save_data:
        temp_data = np.dstack((energies, density_of_state_array, carrier_density))[0]

        np.save("Data/DOS_data_{}_{}".format(RESOLUTION, "DFT" if FERMI_ENERGY == -0.96 else "TOY"), temp_data)

    return energies, density_of_state_array, carrier_density

def velocity_path():
    #hamk_P, hamk_N, v = find_v(useToy=False)

    hamr, ndeg, rvec, nb, nr = get_hamr(DFT_HAM_FILE_NAME)  # Read in the real-space hamiltonian

    hamk_x = find_hamk_a(get_k_path(DEFAULT_PATH, RESOLUTION), hamr, ndeg, rvec, 0)  # FT the velocity x
    hamk_y = find_hamk_a(get_k_path(DEFAULT_PATH, RESOLUTION), hamr, ndeg, rvec, 1)  # FT the velocity y

    hamk = find_hamk(get_k_path(DEFAULT_PATH, RESOLUTION), hamr, ndeg, rvec)  # FT the hamiltonian


    #hamk_pert = vary_ham(hamk)  # Adjust the fermi level


    # Find the energy eigen values
    e = epsilon(hamk)

    #find velocity eigen values
    vx = epsilon(hamk_x)
    vy = epsilon(hamk_y)

    fig, ax = plt.subplots(1, dpi = 200)

    ax.plot(e[:,0], '--b', label=r"$E_1(k) $")
    ax.plot(vy[:,0], 'b', label=r"$v_{y1}(k)$")
    ax.plot(e[:,1], '--r', label=r"$E_2(k) $")
    ax.plot(vy[:,1], 'r', label=r"$v_{y2}(k)$")
    ax.legend()

    ax.set_xticks([RESOLUTION * i for i in range(len(DEFAULT_PATH))], DEFAULT_PATH)
    ax.set_xlim(0, (len(DEFAULT_PATH) -1) *RESOLUTION )
    ax.hlines(0,color="k", linestyle="--", xmin=0, xmax=(len(DEFAULT_PATH) -1) *RESOLUTION)

    plt.show()



def main():



    hamr, ndeg, rvec, nb, nr = get_hamr(DFT_HAM_FILE_NAME)  # Read in the real-space hamiltonian

    k = get_better_k_square(RESOLUTION)

    hamk = find_hamk(k, hamr, ndeg, rvec)  # FT the hamiltonian

    r_x, r_y, r_z = find_pos_operator(DFT_POS_FILE_NAME, nb, nr)


    # get unitary rotations in shape (nb,nb,nk)
    U = np.transpose(get_eig_vec(hamk), (1,2,0))
    U_dagger = np.conj(np.transpose(np.array([np.linalg.inv(U[:,:,i]) for i in range(U.shape[2])]),(1,2,0)))

    # get eigenvalues
    E = epsilon(hamk)


    E_H = np.zeros((2,2,E.shape[0]))
    E_H[0,1,:] = E[:, 0] - E[:, 1]
    E_H[1,0,:] =  E[:,1] - E[:,0]


    # get Berry Connection for x and y
    b_con_x = find_Berry_connection(k, r_x, ndeg, rvec)
    b_con_y = find_Berry_connection(k, r_y, ndeg, rvec)



    # get dH/dk_a
    h_x = find_hamk_a(k, hamr, ndeg, rvec, 0)  # FT the ix . H
    h_y = find_hamk_a(k, hamr, ndeg, rvec, 1)  # FT the iy . H


    ## transform all variables into BAR - (H) basis

    h_x_bar = np.einsum("ij... , jk..., kl... -> il...", U_dagger, h_x, U)
    h_y_bar = np.einsum("ij... , jk..., kl... -> il...", U_dagger, h_y, U)
    b_con_x_bar = np.einsum("ij... , jk..., kl... -> il...", U_dagger, b_con_x, U)
    b_con_y_bar = np.einsum("ij... , jk..., kl... -> il...", U_dagger, b_con_y, U)


    v_x =   h_x_bar - 1j * np.einsum("ij..., ij... -> ij...", E_H, b_con_x_bar)
    v_y =   h_y_bar - 1j * np.einsum("ij..., ij... -> ij...", E_H, b_con_y_bar)

    np.save("Data/velocity_x_{}".format(RESOLUTION),v_x )
    np.save("Data/velocity_y_{}".format(RESOLUTION),v_y )

    np.save("Data/hamk_{}".format(RESOLUTION),hamk )

    '''

    k, full_res = get_better_k_square(RESOLUTION,scale= 1)

    hamk = np.load("Data/hamk_{}.npy".format(RESOLUTION))
    v_x = np.load("Data/velocity_x_{}.npy".format(RESOLUTION))
    v_y = np.load("Data/velocity_y_{}.npy".format(RESOLUTION))
    print(hamk.shape)

    e = epsilon(hamk)

    #
    T = 6.5
    pert_hamk = vary_ham(hamk, H = 0)

    e = epsilon(pert_hamk)



    significant_kpoints_indices = np.where(abs(e[:,0]) < DEBYE_ENERGY)

    #print(significant_kpoints_indices)

    sig_k = k[:,significant_kpoints_indices][:,0]

    sig_ham = hamk[:,:,significant_kpoints_indices][:,:,0]

    sig_v_x = v_x[:,:,significant_kpoints_indices][:,:,0]
    sig_v_y = v_y[:,:,significant_kpoints_indices][:,:,0]


    print(sig_ham.shape)



    #sig_k /= (2 * np.pi)
    #e = e.reshape((RESOLUTION,RESOLUTION,2))
    #k = k.reshape((3,RESOLUTION, RESOLUTION)) / (2 * np.pi)
    #col = plt.pcolormesh(k[0],k[1],e[:,:,0])
    #plt.scatter(sig_k[0],sig_k[1],color="black")
    #plt.colorbar(col)


    pert_hamk = vary_ham(sig_ham)


    chi = Kubo_susceptibility_unnorm(pert_hamk, sig_v_x, sig_v_y, T, NUM_FREQ) / (RESOLUTION**2)
    v = 1/chi

    func = lambda T, H : delta_kubo(T,  v,   sig_ham, sig_v_x, sig_v_y, fermi_energy = FERMI_ENERGY, H=H, phi=PHI_DEFAULT, theta=THETA_DEFAULT)

    range_of_brakets(func, H_L_START, H_U_START, TEMP_START, TEMP_STOP, TEMP_STEPS)
    '''

    '''


    #find velocity eigen values
    vx = epsilon(v_x)
    vy = epsilon(v_y)

    vx = vx.reshape((RESOLUTION,RESOLUTION,2))
    vy = vy.reshape((RESOLUTION,RESOLUTION,2))

    # start plotting
    fig, axs = plt.subplots(ncols=2, dpi = 200, figsize=(10,8/3))


    k = k.reshape((3,RESOLUTION, RESOLUTION)) / (2 * np.pi)

    norm_1 = colors.CenteredNorm(0)

    image = axs[0].pcolor(k[0], k[1],chi.real.reshape((RESOLUTION,RESOLUTION)), cmap="bwr", norm = norm_1)
    axs[0].set_aspect("equal")
    axs[0].set_xlim((k[0].min(), k[0].max()))
    axs[0].set_ylim((k[1].min(), k[1].max()))
    axs[0].set_title(r"$Re[\chi]$")
    axs[0].set_xlabel(r"B1")
    axs[0].set_ylabel(r"B2")


    fig.colorbar(image)
    fig.suptitle(r"Susceptibility $\chi $" + "\n")

    norm_2 = colors.CenteredNorm(0)

    image = axs[1].pcolor(k[0], k[1],chi.imag.reshape((RESOLUTION,RESOLUTION)), cmap="bwr", norm= norm_2)
    axs[1].set_aspect("equal")
    axs[1].set_xlim((k[0].min(), k[0].max()))
    axs[1].set_ylim((k[1].min(), k[1].max()))
    axs[1].set_title(r"$Im[\chi]$")
    axs[1].set_xlabel(r"B1")
    axs[1].set_ylabel(r"B2")


    fig.colorbar(image)

    path_points = np.array([[0,0,1/3, 2/3],[0,1/2,1/3, 2/3]])
    axs[0].scatter(path_points[0],path_points[1],color="black",marker="x")
    axs[1].scatter(path_points[0],path_points[1],color="black",marker="x")


    '''



if __name__ == "__main__":
    main()





