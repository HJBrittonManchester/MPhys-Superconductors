# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:12:32 2024

@author: hbrit
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from matplotlib import colors


## our files
from k_tools import get_k_path, get_k_block, get_k_path_spacing
from eig_tools import epsilon, projection_z, projection_x, projection_y , get_eig_vec
from DFT_tools import get_hamr, find_hamk, find_hamk_a
from GF_tools import get_greens_function, matsubara_frequency, susc
from phase_diagram import H_0, a


DFT_FILE_NAME = "Data/MoS2_hr.dat"

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
RESOLUTION = 50
NUM_FREQ = 500

# k - path settings
DEFAULT_PATH = ['G', 'M', 'K','G']  

# Full BZ settings
PRESELECTION_BOXSIZE = -1 # 0.32  # set to -1 to use full area but, 0.22 works well


# Bracket settings
BRACKET_TOLERANCE = 1e-6
MAX_BRACKET_STEPS = 25
TEMP_START = 9
TEMP_STOP = 6.5
TEMP_STEPS = 50
H_U_START = 60
H_L_START = -1


fermi_levels = np.array([[-0.85, 4.923828125],
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

    return 1 - v * susc(hamk_pert_N, hamk_pert_P, T, NUM_FREQ)


def braket(ham_P, ham_N, T, v, fermi_energy = FERMI_ENERGY, start_H_U=H_U_START, start_H_L=H_L_START,
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

    # print("Δ_max = {}, Δ_min = {}".format(
    #    current_delta_U, current_delta_L))

    if current_delta_U < 0:
        print("Upper H too low")
        return [start_H_L,start_H_L]
    elif current_delta_L > 0:
        print("Lower H too high")
        return [0,0]

    old_H_U = 100
    old_H_L = 0

    while abs(current_delta_L) > tol and abs(current_delta_U) > tol and iterations < MAX_BRACKET_STEPS:
        # print("Δ_max = {}, Δ_min = {}".format(
        #    current_delta_U, current_delta_L))

        if current_delta_L > 0 and current_delta_U > 0:
            print("both +ve sign")

            current_H_U = current_H_L
            current_H_L = old_H_L

            # reset upper
            current_delta_U = current_delta_L
            # recalculate lower
            delta(T, v,  ham_P, ham_N, fermi_energy, H=current_H_L, theta=theta, phi=phi)

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


def bracketing(ham_P, ham_N, v, fermi_energy = FERMI_ENERGY):
    T_array = np.linspace(TEMP_START, TEMP_STOP, TEMP_STEPS)
    H_array = []
    
    lower_bounds = []
    upper_bounds = []
    
    for t_index in range(TEMP_STEPS):
        temp_T = T_array[t_index]
        
        
        bounds = braket(ham_P.copy(), ham_N.copy(), temp_T, v, fermi_energy)
        
        mean_H = (bounds[0] + bounds[1])/2
        
        H_array.append(mean_H)
        lower_bounds.append(bounds[0])
        upper_bounds.append(bounds[1])

        print("[{}, {}], bounds: [{}, {}],".format(mean_H, temp_T, bounds[0], bounds[1]))
        
    return T_array, np.array(H_array), np.array(lower_bounds), np.array(upper_bounds)


def fermi_level_bracketing(ham_P, ham_N, v, ef=FERMI_ENERGY, start_T_L=6, start_T_U=9.3,
                           theta=THETA_DEFAULT, phi=PHI_DEFAULT, tol=BRACKET_TOLERANCE):
    iterations = 0

    current_T_L = start_T_L
    current_T_U = start_T_U
    current_delta_L = delta(current_T_L, v, ham_P, ham_N, fermi_energy=ef,
                            H=0, theta=theta, phi=phi)
    current_delta_U = delta(current_T_U, v, ham_P, ham_N, fermi_energy=ef,
                            H=0, theta=theta, phi=phi)
    old_T_L = 5
    old_T_U = 7

    while abs(current_delta_L) > tol and abs(current_delta_U) > tol and iterations < MAX_BRACKET_STEPS:

        if current_delta_L > 0 and current_delta_U > 0:
            print("both +ve sign")

            current_T_U = current_T_L
            current_T_L = old_T_L

            # reset upper
            current_delta_U = current_delta_L
            # recalculate lower
            current_delta_L = delta(current_T_L, v, ham_P, ham_N, fermi_energy=ef,
                                    H=0, theta=theta, phi=phi)

        elif current_delta_L < 0 and current_delta_U < 0:
            print("both -ve sign")

            current_T_L = current_T_U
            current_T_U = old_T_U

            # reset lower
            current_delta_L = current_delta_U
            # recalculate upper
            current_delta_U = delta(current_T_U, v, ham_P, ham_N, fermi_energy=ef,
                                    H=0, theta=theta, phi=phi)

        elif abs(current_delta_L) > abs(current_delta_U):
            old_T_L = current_T_L
            current_T_L = (current_T_L + current_T_U) / 2
            current_delta_L = delta(
                current_T_L, v, ham_P, ham_N, fermi_energy=ef, H=0, theta=theta, phi=phi)

        else:
            old_T_U = current_T_U
            current_T_U = (current_T_L + current_T_U) / 2
            current_delta_U = delta(
                current_T_U, v, ham_P, ham_N, fermi_energy=ef, H=0, theta=theta, phi=phi)

        # print("[{}, {}]".format(
         #   current_delta_L, current_delta_U))

        iterations += 1

    if iterations == MAX_BRACKET_STEPS-1:
        print("Reached max iterations")

    return [current_T_L, current_T_U]

def find_v(useToy=False):
    
    preselected_kpoints = get_k_block(RESOLUTION, PRESELECTION_BOXSIZE)

    if useToy:
        hamk_P = get_toy_ham(preselected_kpoints)
    else:
        hamr, ndeg, rvec = get_hamr(DFT_FILE_NAME)  # Read in the real-space hamiltonian
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


    # correct +ve ham to significant k points
    hamk_P = hamk_P[:, :, significant_kpoints_indices][:, :, 0]
    hamk_pert_P = vary_ham(hamk_P)  # reset +ve
    hamk_pert_N = vary_ham(hamk_N)

    return hamk_P, hamk_N

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
    
    hamr, ndeg, rvec = get_hamr(DFT_FILE_NAME)  # Read in the real-space hamiltonian

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
   
    hamr, ndeg, rvec = get_hamr(DFT_FILE_NAME)  # Read in the real-space hamiltonian

    hamk_x = find_hamk_a(get_k_block(RESOLUTION), hamr, ndeg, rvec, 0)  # FT the velocity x
    hamk_y = find_hamk_a(get_k_block(RESOLUTION), hamr, ndeg, rvec, 1)  # FT the velocity y
    
    #find velocity eigen values
    vx = epsilon(hamk_x)
    vy = epsilon(hamk_y)
    
    vx = vx.reshape((RESOLUTION,RESOLUTION,2))
    vy = vy.reshape((RESOLUTION,RESOLUTION,2))

    
    fig, ax = plt.subplots(1, dpi = 200)
    
    col = np.zeros((RESOLUTION,RESOLUTION,3))
    
    
    
    col[:,:,0] = vx[:,:,0]
    col[:,:,1] = vy[:,:,1]
    
    #max_velocity = col.max()
    #min_velocity = col.min()
    
    #col = (col - min_velocity) / (max_velocity - min_velocity)
    
    #print(max_velocity)
    #(min_velocity)
    
    image = ax.imshow(col[:,:,0], cmap="magma")
    
    
    fig.colorbar(image)
    
    path_points = np.array([[0,0,1/3],[0,1/2,1/3]]) * RESOLUTION
    ax.scatter(path_points[0],path_points[1],color="white",marker="x")
    
    
    
    

if __name__ == "__main__":
    main()

