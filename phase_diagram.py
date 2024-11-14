import numpy as np
import matplotlib.pyplot as plt
import scipy.constants


MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]

# set up global variables and parameters (parameters will be optimised later)
tolerance = 0.002  # set to ensure fermi momenta constraint gives two points
resolution = 1000

# lattice constants
a = 3.25  # https://www.researchgate.net/publication/279633132_A_tight-binding_model_for_MoS_2_monolayers
b = 4 * np.pi / (np.sqrt(3) * a)

# tight binding parameters. optimised to fit the required constraints:
alpha_zeeman = 5.77350269e-04
alpha_rashba = alpha_zeeman * 9.03566289e-05
beta = 2.75681159e+01  # 1.0073e1
#
chemical_potential = -0.75

# hopping parameters
t1 = 0.146
t2 = -0.4 * t1
t3 = 0.25 * t1
hopping_parameters = np.array([t1, t2, t3])


# create coordinates and paths of interest:
gamma = np.array([0, 0])
k_point = np.array([0, b/np.sqrt(3)])
m_point = np.array([-b/4, np.sqrt(3)*b/4])

gamma_to_k_point_path = np.zeros((2, resolution))
gamma_to_k_point_path[0, :] = 0
gamma_to_k_point_path[1, :] = np.linspace(0, k_point[1], resolution)

k_to_m_path = np.zeros((2, resolution))
k_to_m_path[0, :] = np.linspace(k_point[0], m_point[0], resolution)
k_to_m_path[1, :] = np.linspace(k_point[1], m_point[1], resolution)


def magnitude(vector):
    result = 0.
    for i in range(len(vector)):
        result += vector[i]**2
    return np.sqrt(result)


# dispersion relation equation from tight binding model.
# this comes from considering 1st, 2nd, 3rd nearest neighbours between Mo atoms
# in hexagonal lattice
def dispersion_relation(kx, ky, hps):
    first_term = np.cos(ky*a) + 2*np.cos(np.sqrt(3)/2 * kx*a)*np.cos(ky*a/2)
    second_term = np.cos(np.sqrt(3) * kx*a) + 2 * \
        np.cos(np.sqrt(3)/2 * kx*a)*np.cos(3*ky*a/2)
    third_term = np.cos(2*ky*a) + 2*np.cos(np.sqrt(3) * kx*a)*np.cos(ky*a)

    return 2*(hps[0]*first_term + hps[1]*second_term + hps[2]*third_term)


def epsilon(kx, ky): return dispersion_relation(
    kx, ky, hopping_parameters) - chemical_potential

# define spin-splitting functions and zeeman/rashba interactions:


def f(kx, ky):
    global a
    return abs(np.sin(ky*a) - 2 * np.cos(np.sqrt(3) * kx*a / 2) * np.sin(ky*a / 2))


def F(kx, ky, beta):
    global k_point
    global f
    return beta * np.tanh(f(k_point[0], k_point[1]) - f(kx, ky)) - 1

# find zeeman/rashba corrections:


def g_zeeman(kx, ky, beta):
    global F
    z_term = np.sin(ky*a) - 2 * np.cos(np.sqrt(3)*kx*a/2) * np.sin(ky*a/2)
    return np.array([0, 0, F(kx, ky, beta) * z_term], dtype=object)


def g_rashba(kx, ky, beta):
    global F
    x_term = -np.sin(ky*a) - np.cos(np.sqrt(3)*kx*a/2)*np.sin(ky*a/2)
    y_term = np.sqrt(3)*np.sin(np.sqrt(3)*kx*a/2)*np.cos(ky*a/2)
    return np.array([F(kx, ky, beta) * x_term, F(kx, ky, beta) * y_term, 0], dtype=object)


def H_0(kx, ky):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)

    # add kinetic term

    hamiltonian[0, 0] += dispersion_relation(kx,
                                             ky, hopping_parameters) - chemical_potential
    hamiltonian[1, 1] += dispersion_relation(kx,
                                             ky, hopping_parameters) - chemical_potential

    # add zeeman

    hamiltonian[0, 0] += alpha_zeeman * g_zeeman(kx, ky, beta)[2]
    hamiltonian[1, 1] -= alpha_zeeman * g_zeeman(kx, ky, beta)[2]

    # add Rashba

    hamiltonian[0, 1] += alpha_rashba * \
        (g_rashba(kx, ky, beta)[0] - 1j * g_rashba(kx, ky, beta)[1])
    hamiltonian[1, 0] += alpha_rashba * \
        (g_rashba(kx, ky, beta)[0] + 1j * g_rashba(kx, ky, beta)[1])

    return np.matrix(hamiltonian)

# calculate the regular and spin-split energies along the paths:


def find_energies():
    gamma_to_k_point_dispersion = np.zeros(resolution)
    k_to_m_dispersion = np.zeros(resolution)
    k = np.zeros(2*resolution)

    for i in range(resolution):

        gamma_to_k_point_dispersion[i] = dispersion_relation(
            gamma_to_k_point_path[0, i], gamma_to_k_point_path[1, i], hopping_parameters) - chemical_potential
        k_to_m_dispersion[i] = dispersion_relation(
            k_to_m_path[0, i], k_to_m_path[1, i], hopping_parameters) - chemical_potential

        # 'reflect' the gamme->K path about zero, and make relative to K
        k[i] = -magnitude(gamma_to_k_point_path[:, i] - k_point)
        k[i+resolution] = magnitude(k_to_m_path[:, i] - k_point)

    energy = np.zeros(2*resolution)
    energy = np.append(gamma_to_k_point_dispersion, k_to_m_dispersion)

    g_zeeman_dispersion = np.zeros(2*resolution)
    g_rashba_dispersion_x = np.zeros(2*resolution)
    g_rashba_dispersion_y = np.zeros(2*resolution)

    for i in range(resolution):
        g_zeeman_dispersion[i] = g_zeeman(
            gamma_to_k_point_path[0, i], gamma_to_k_point_path[1, i], beta)[2]
        g_zeeman_dispersion[i+resolution] = g_zeeman(
            k_to_m_path[0, i], k_to_m_path[1, i], beta)[2]

        g_rashba_dispersion_x[i] = g_rashba(
            gamma_to_k_point_path[0, i], gamma_to_k_point_path[1, i], beta)[0]
        g_rashba_dispersion_x[i+resolution] = g_rashba(
            k_to_m_path[0, i], k_to_m_path[1, i], beta)[0]

        g_rashba_dispersion_y[i] = g_rashba(
            gamma_to_k_point_path[0, i], gamma_to_k_point_path[1, i], beta)[1]
        g_rashba_dispersion_y[i+resolution] = g_rashba(
            k_to_m_path[0, i], k_to_m_path[1, i], beta)[1]

    # create hamiltonian matrix and extract eigenvalues:
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    e_vals_positive = np.zeros(2*resolution)
    e_vals_negative = np.zeros(2*resolution)

    for i in range(2*resolution):
        hamiltonian[0][0] = energy[i] + alpha_zeeman*g_zeeman_dispersion[i]
        hamiltonian[0][1] = alpha_rashba * \
            (g_rashba_dispersion_x[i] - g_rashba_dispersion_y[i]*1j)
        hamiltonian[1][0] = alpha_rashba * \
            (g_rashba_dispersion_x[i] + g_rashba_dispersion_y[i]*1j)
        hamiltonian[1][1] = energy[i] - alpha_zeeman*g_zeeman_dispersion[i]

        e_vals_positive[i] = np.linalg.eigh(hamiltonian)[0][0]
        e_vals_negative[i] = np.linalg.eigh(hamiltonian)[0][1]

    return k, e_vals_positive, e_vals_negative, energy


# create plots:
def plot_dispersion_relation():
    k, energy = find_energies()[0], find_energies()[3]
    fig, ax = plt.subplots(1, figsize=(3, 4), dpi=400)

    ax.plot(k, energy, 'k-')
    ax.grid(True, linestyle='--')
    ax.set_title("$\Gamma$ $\leftarrow$ $K$" r"$\rightarrow$ $M$")
    ax.set_xlabel("$k$ $(\AA^{-1})$")
    ax.set_ylabel("$E - E_F$ $(eV)$")
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.05)
    plt.show()
    return None


def plot_tight_binding():
    k, e_vals_positive, e_vals_negative = find_energies()[0:3]
    fig, ax = plt.subplots(1, figsize=(3, 4), dpi=400)

    ax.plot(k, e_vals_positive, 'b-')
    ax.plot(k, e_vals_negative, 'r-')
    ax.grid(True, linestyle='--')
    ax.set_title(r"$\Gamma$" r"$\leftarrow$" "$K$" r"$\rightarrow$" "$M$")
    ax.set_xlabel(r"$k$ $(\AA^{-1})$")
    ax.set_ylabel(r"$E - E_F$ $(eV)$")
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.05)
    plt.show()
    return None


# optimise the parameters alpha_zeeman, alpha_rashba, beta. this can be tuned based
# on the desired spin-orbit ratios between zeeman and rashba effects:
def optimise_parameters(soi_ratio):
    k, e_vals_positive, e_vals_negative = find_energies()[0:3]
    # find the fermi momentum for use in optimising parameters using definition:
    fermi_momenta_indices = np.array(
        np.where(abs(e_vals_positive + e_vals_negative) < tolerance))[0]

    # calculate direction of fermi vectors from the magnitudes
    fermi_momenta = np.zeros((2, 2))
    fermi_momenta[:, 0] = [0, -k[fermi_momenta_indices[0]]]
    fermi_momenta[:, 1] = [k[fermi_momenta_indices[1]]
                           * np.sqrt(3)/2, k[fermi_momenta_indices[1]]*1/2]

    # coordinates relative to gamma point, not K:
    fermi_momenta[0][1] = k_point[0] - fermi_momenta[0][1]
    fermi_momenta[1][1] = k_point[1] - fermi_momenta[1][1]

    # alpha_zeeman found directly from substitution:
    alpha_zeeman_opt = 1e-3 / np.sqrt(3)

    # find coupling g-terms at the fermi momentum:
    gz_at_fermi_momentum = abs(np.sin(fermi_momenta[1][1]*a) - 2 * np.cos(np.sqrt(3)*fermi_momenta[0][1]*a/2) *
                               np.sin(fermi_momenta[1][1]*a/2))
    gr_at_fermi_momentum = magnitude([-np.sin(fermi_momenta[1][1]*a) - np.cos(np.sqrt(3)*fermi_momenta[0][1]*a/2) * np.sin(fermi_momenta[1][1]*a/2),
                                      np.sqrt(3)*np.sin(np.sqrt(3)*fermi_momenta[0][1]*a/2)*np.cos(fermi_momenta[1][1]*a/2)])

    # from constraint on soi ratios:
    alpha_rashba_opt = soi_ratio * alpha_zeeman_opt * \
        (gz_at_fermi_momentum/gr_at_fermi_momentum)

    # final constraint to find beta analytically:
    beta_opt_numerator = 1 + (13e-3/2) / np.sqrt(alpha_rashba_opt**2 *
                                                 gr_at_fermi_momentum**2 + alpha_zeeman_opt**2 * gz_at_fermi_momentum**2)
    beta_opt_denominator = np.tanh(
        f(k_point[0], k_point[1]) - f(fermi_momenta[0][1], fermi_momenta[1][1]))

    beta_opt = beta_opt_numerator / beta_opt_denominator

    optimised_parameters = np.array(
        [alpha_zeeman_opt, alpha_rashba_opt, beta_opt])

    return optimised_parameters


def Det(kx, ky, omega, hx, hy, hz):
    

    
    return (1j * omega - epsilon(kx, ky))**2 - (alpha_zeeman *
                                                g_zeeman(kx, ky, beta)[2] + MU_B* hz)**2 - \
        (( - alpha_rashba * g_rashba(kx, ky, beta)[0] + MU_B* hx)**2 +
         (-alpha_rashba*g_rashba(kx, ky, beta)[1] + MU_B* hy)**2)


def GF_up_up(kx, ky, omega, hx, hy, hz):
    

    return (1j * omega - epsilon(kx, ky) + alpha_zeeman * g_zeeman(kx, ky, beta)[2] - hz * MU_B)


def GF_down_down(kx, ky, omega,  hx, hy, hz):
    return  (1j * omega - epsilon(kx, ky) - alpha_zeeman * g_zeeman(kx, ky, beta)[2] + hz * MU_B)


def GF_up_down(kx, ky, omega,  hx, hy, hz):
    return  (-alpha_rashba * (g_rashba(kx, ky, beta)[0] - 1j * g_rashba(kx, ky, beta)[1]) - MU_B * complex(hx, -hy))


def GF_down_up(kx, ky, omega,  hx, hy, hz):
    return (-alpha_rashba * (g_rashba(kx, ky, beta)[0] + 1j * g_rashba(kx, ky, beta)[1]) - MU_B * complex(hx, hy))

def GF_susc(kx,ky,omega, H, theta=np.pi/2, phi = 0):
    
    hx = H * np.sin(theta) * np.cos(phi)
    hy = H * np.sin(theta) * np.sin(phi)
    hz = H * np.cos(theta)
    
    chi = GF_up_up(kx,ky,omega,hx,hy,hz) * GF_down_down(-kx, -ky, -omega, hx, hy, hz) - \
        GF_up_down(kx,ky,omega,hx,hy,hz) * GF_down_up(-kx, -ky, -omega, hx, hy, hz)
        
    chi = -chi /(Det(kx,ky,omega,hx,hy,hz) * Det(-kx, -ky, -omega, hx, hy, hz))
    
    return chi
    
    
    
