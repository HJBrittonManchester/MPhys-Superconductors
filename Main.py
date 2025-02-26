# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:12:32 2024

@author: hbrit
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from matplotlib import colors

import threading

import time


## our files
from k_tools import get_k_path, get_k_block, get_k_path_spacing, get_better_k_square, get_close_k_points
from eig_tools import diagonalise, projection_z, projection_x, projection_y , get_eig_vec
from DFT_tools import get_hamr, find_hamk, find_hamk_a, find_Berry_connection, find_pos_operator
from GF_tools import get_greens_function, matsubara_frequency, susc, Kubo_susceptibility_unnorm, test_susc
from phase_diagram import H_0, a

from bracket import range_of_brackets


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
THETA_DEFAULT =  0 # np.pi / 2

# Simulation settings
RESOLUTION = 600
NUM_FREQ = 2000

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

THREADING = True


#############
# profiler class
class profiler():

    def __init__(self, just_summary = False):
        self.times = [time.time()]
        self.labels = ["start"]
        self.just_summary = just_summary

    def Next(self, new_label):
        self.times.append(time.time())
        self.labels.append(new_label)

        if not self.just_summary:
            time_step = self.times[-1] - self.times[-2]
            print("* {} took: {:.5g}s".format(self.labels[-1], time_step))

    def Summary(self):

        print("-"*25)
        print("|\tProfiling Summary\t|")
        print("-"*25)
        print()
        print("Total runtime: {:.5g}s".format(self.times[-1] - self.times[0]))

        for i in range(1, len(self.times)):
            time_step = self.times[i] - self.times[i-1]

            print("\t* {} took: {:.5g}s".format(self.labels[i], time_step))


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
    energy_toy = diagonalise(hamk_pert_toy)[0]
    energy_real = diagonalise(hamk_pert)[0]

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



    #print(H * MU_B * np.cos(theta))

    return new_ham


##############################################################################
# DOS functions

def DOS(E, hamk, res=RESOLUTION, sigma = 1e-2):

    ham = vary_ham(hamk)

    E_k = diagonalise(ham)[0].flatten()


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


def delta_kubo(T,  v,   hamk_W, v_H, U, fermi_energy = FERMI_ENERGY, H=0, phi=PHI_DEFAULT, theta=THETA_DEFAULT):

    hamk_W_pert = vary_ham(hamk_W, fermi_energy, H, theta=theta, phi=phi)

    return 1- v*Kubo_susceptibility_unnorm(hamk_W_pert, v_H, U, T, NUM_FREQ).real / RESOLUTION**2


def find_v(useToy=False):
    hamk_P = np.load("Data/DFT_H0_300_P.npy")


    #preselected_kpoints, FULL_RES = get_better_k_square(RESOLUTION, (1/3,1/3), 0.21) #get_k_block(RESOLUTION, PRESELECTION_BOXSIZE)

    '''
    if useToy:
        hamk_P = get_toy_ham(preselected_kpoints)
    else:
        hamr, ndeg, rvec, nb, nr = get_hamr(DFT_HAM_FILE_NAME)  # Read in the real-space hamiltonian
        hamk_P = find_hamk(preselected_kpoints, hamr, ndeg, rvec)  # FT the hamiltonian
        '''

    hamk_pert_P = vary_ham(hamk_P)  # Adjust the fermi level

    # Find the mean of energy eigen values
    energy = diagonalise(hamk_pert_P)[0].mean(axis=1)

    # Find the points within the Debye energy of fermi surface
    significant_kpoints_indices = np.where(abs(energy) < DEBYE_ENERGY)

    # Find -ve ham to significant k points
    #significant_kpoints = preselected_kpoints[:, significant_kpoints_indices][:, 0, :]


    '''
    if useToy:
        hamk_N = get_toy_ham(-significant_kpoints)
    else:
        hamk_N = find_hamk(-significant_kpoints, hamr, ndeg, rvec)
    '''

    hamk_N = np.load("Data/DFT_H0_300_N.npy")
    hamk_N = hamk_N[:, :, significant_kpoints_indices][:, :, 0]


    # correct +ve ham to significant k points
    hamk_P = hamk_P[:, :, significant_kpoints_indices][:, :, 0]

    hamk_pert_P = vary_ham(hamk_P)  # reset +ve
    hamk_pert_N = vary_ham(hamk_N)

    print(hamk_pert_N.shape)

    v = 1/(susc(hamk_pert_N, hamk_pert_P, 6.5, NUM_FREQ).sum() / 300**2)

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
    e_dft = diagonalise(hamk)[0]

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

    # Find the energy eigen values
    e = diagonalise(hamk)[0]

    #find velocity eigen values
    vx = diagonalise(hamk_x)[0]
    vy = diagonalise(hamk_y)[0]

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


def calculate_full_bz_variables(resolution = RESOLUTION, save = True):
    # Read in the real-space hamiltonian and  the position matrix
    hamr, ndeg, rvec, nb, nr = get_hamr(DFT_HAM_FILE_NAME)
    r = find_pos_operator(DFT_POS_FILE_NAME, nb, nr)

    # generate k vector of reduced area of BZ centred around the K and K' points
    k = get_close_k_points(resolution) # K point
    k = np.hstack((k, get_close_k_points(resolution,centre=(2/3,2/3)))) # K' point


    # FT the hamiltonian to get the bloch Hamiltonian in Wannier (band) basis
    hamk_W = find_hamk(k, hamr, ndeg, rvec)
    hamk_W_n = find_hamk(-k, hamr, ndeg, rvec)

    ## DEBUG: test effect of Field on the gauge transform
    # hamk_W = vary_ham(hamk_W, H =0)

    # get eigenvalues E and unitary U rotations in shapes (nb,nk), and (nb,nb,nk)
    E, U = diagonalise(hamk_W)

    # get Berry Connection vector
    A_W = find_Berry_connection(k, r, ndeg, rvec)

    # get dH/dk vector
    dh_dka_W = find_hamk_a(k, hamr, ndeg, rvec)  # FT the ix_a . H

    # kmag = np.sqrt((k[0]-2*np.pi/3)**2 + (k[1]-2*np.pi/3)**2)
    # vmag = np.sqrt(dh_dka_W[0,0,0] ** 2 + dh_dka_W[0,0,1]**2)
    # c = plt.scatter(k[0]-2*np.pi/3, k[1]-2*np.pi/3 ,c=vmag)
    # plt.colorbar(c)

    ## transform all variables into BAR - (H) basis
    dh_dka_bar_H = np.einsum("jim , jkam, klm -> ilam", U.conj(), dh_dka_W, U)
    A_bar_H = np.einsum("jim , jkam, klm -> ilam", U.conj(), A_W, U)

    # calculate (E_n - E_m)A_nm_a  term
    EA_H = np.zeros_like(A_bar_H)
    EA_H[0,1] = np.real_if_close(E[0] - E[1]) * A_bar_H[0,1]
    EA_H[1,0] = np.real_if_close(E[1] - E[0]) * A_bar_H[1,0]

    # find velocity matrix in Hamiltonian Gauge
    v_H = dh_dka_bar_H - 1j * EA_H

    if save:
        np.save("Data/hamk_W_2k_{}".format(resolution), hamk_W)
        np.save("Data/velocity_H_2k_{}".format(resolution), v_H)
        np.save("Data/U_2k_{}".format(resolution), U)
        np.save("Data/hamk_W_n_2k_{}".format(resolution), hamk_W_n)


    return hamk_W, v_H, U, hamk_W_n

def find_sig_points(resolution = RESOLUTION, save = True, print_stats=True):
    hamk_W = np.load("Data/hamk_W_2k_{}.npy".format(resolution))
    hamk_W_n = np.load("Data/hamk_W_n_2k_{}.npy".format(resolution))
    v_H  = np.load("Data/velocity_H_2k_{}.npy".format(resolution))
    U = np.load("Data/U_2k_{}.npy".format(resolution))

    original_nk = hamk_W.shape[-1]

    # adjust fermi-energy and diagonalise
    pert_hamk = vary_ham(hamk_W)
    E = diagonalise(pert_hamk)[0]


    significant_kpoints_indices = np.where(abs(E[0]) < DEBYE_ENERGY)


    sig_hamk_W =    hamk_W[:,:,significant_kpoints_indices][:, :, 0]
    sig_hamk_W_n =  hamk_W_n[:,:,significant_kpoints_indices][:, :, 0]
    sig_v_H =       v_H[:,:, :, significant_kpoints_indices][:, :, :, 0]
    sig_U =         U[:,:,significant_kpoints_indices][:, :, 0]


    final_nk = sig_hamk_W.shape[-1]

    if print_stats:
        print("Selected {} significant points from {} total"
              .format(final_nk, original_nk))

    if save:
        np.save("Temp/hamk_W_2k_{}_sig".format(resolution), sig_hamk_W)
        np.save("Temp/hamk_W_n_2k_{}_sig".format(resolution), sig_hamk_W_n)

        np.save("Temp/velocity_H_2k_{}_sig".format(resolution), sig_v_H)
        np.save("Temp/U_2k_{}_sig".format(resolution), sig_U)

    return sig_hamk_W, sig_v_H, sig_U



def load_data(directory, suffix):


    hamk_W = np.load("{}/hamk_W_{}.npy".format(directory, suffix))
    hamk_W_n = np.load("{}/hamk_W_n_{}.npy".format(directory, suffix))
    v_H  = np.load("{}/velocity_H_{}.npy".format(directory, suffix))
    U = np.load("{}/U_{}.npy".format(directory, suffix))

    return hamk_W, v_H, U, hamk_W_n



def main():

    prof = profiler(True)

    hamk_W, v_H, U, hamk_W_n = calculate_full_bz_variables(RESOLUTION,True)


    prof.Next("Generate data")

    #find_sig_points()
    #prof.Next("Found sig and saved")

    hamk_W, v_H, U, hamk_W_n = load_data("Data", "{}".format(RESOLUTION))

    prof.Next("Loading files")




    hamk_W_pert = vary_ham(hamk_W)
    hamk_W_n_pert = vary_ham(hamk_W_n)




    chi = test_susc(hamk_W_pert, hamk_W_n_pert, 6.5, NUM_FREQ) / (RESOLUTION**2)
    #Kubo_susceptibility_unnorm(hamk_W_pert, v_H, U, 6.5, NUM_FREQ) / (RESOLUTION**2)

    v =  1/np.real(chi)
    print(v)
    print(1- v * np.real(chi))

    prof.Next("Finding V")


    x_range = np.linspace(0.,6.5, 2)
    y_range = np.linspace(8, 11, 1)
    X, Y = np.meshgrid(x_range, y_range)
    Y += 0# np.sqrt(1-X/ 6.5) * 125

    X = X.flatten()
    Y = Y.flatten()
    d = np.zeros_like(X)

    if THREADING:

        threads = []

        def d_func(i):
            hp = vary_ham(hamk_W, H=Y[i])
            hn = vary_ham(hamk_W_n, H=Y[i])

            d[i] =  1 - v * test_susc(hp, hn, X[i], NUM_FREQ).real /(RESOLUTION**2)
            #delta_kubo(X[i], v, hamk_W, v_H, U, H=Y[i])

            return None

        for i in range(len(X)):
            t = threading.Thread(target=d_func, args=(i,))
            t.start()
            threads.append(t)
            #print(i)

        print("all threads started")



        for i in range(len(threads)):
            threads[i].join()
    else:
        for i in range(len(X)):
            d[i] = delta_kubo(X[i], v, hamk_W, v_H, U, H=Y[i])


    prof.Next("bracketing")


    # #X, Y, d = np.load("Temp/tempFieldRes.npy")

    print(X.tolist())
    print()
    print(Y.tolist())
    print()
    print(d.tolist())

    # res = np.vstack((X,Y,d))
    # np.save("Temp/tempFieldRes", res)


    fig, ax = plt.subplots(dpi = 400)

    clipping= 1e-10
    #c = ax.scatter(X,Y,c=(X**2 *d), cmap="bwr",     norm = colors.TwoSlopeNorm(vcenter=0, vmin=-clipping, vmax=clipping))
    for t in np.unique(Y):
        hfield = X[np.where(Y==t)]
        delta = d[np.where(Y==t)]
        ax.plot(hfield, delta, label=str(t) + "T")

    #ax.set_facecolor("black")
    ax.legend()
    ax.set_title("Z-directed external field")
    #fig.colorbar(c, ax=ax)
    #ax.set_xlabel("T / Kelvin")
    ax.set_xlabel("T / K")
    ax.set_ylabel(r"$\Delta$")
    ax.hlines(0,X.min(), X.max(), color="black", linestyle="dashed")
    ax.set_xlim((X.min(), X.max()))
    #ax.set_ylim((-.01, .01))


    prof.Next("Plotting")


    prof.Summary()




if __name__ == "__main__":
    main()





