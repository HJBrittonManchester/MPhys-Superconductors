import numpy as np
import matplotlib.pyplot as plt


# set up global variables and parameters (parameters will be optimised later)
tolerance = 0.002  # set to ensure fermi momenta constraint gives two points
resolution = 1000

# lattice constants
a = 3.16  # https://www.researchgate.net/publication/279633132_A_tight-binding_model_for_MoS_2_monolayers
b = 4 * np.pi / (np.sqrt(3) * a)

# tight binding parameters
alpha_zeeman = 0.0005
alpha_rashba = alpha_zeeman * 0.15
beta = 27.
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


# calculate the energies along the paths:
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


# create plot:
def plot_dispersion_relation():
    global k
    global energy
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
    return F(kx, ky, beta) * np.array([0, 0, z_term])


def g_rashba(kx, ky, beta):
    global F
    x_term = -np.sin(ky*a) - np.cos(np.sqrt(3)*kx*a/2)*np.sin(ky*a/2)
    y_term = np.sqrt(3)*np.sin(np.sqrt(3)*kx*a/2)*np.cos(ky*a/2)
    return F(kx, ky, beta) * np.array([x_term, y_term, 0])


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


# create plot
def plot_tight_binding():
    global k
    global e_vals_positive
    global e_vals_negative
    fig, ax = plt.subplots(1, figsize=(3, 4), dpi=400)

    ax.plot(k, e_vals_positive, 'm-')
    ax.plot(k, e_vals_negative, 'c-')
    ax.grid(True, linestyle='--')
    ax.set_title(r"$\Gamma$" r"$\leftarrow$" "$K$" r"$\rightarrow$" "$M$")
    ax.set_xlabel(r"$k$ $(\AA^{-1})$")
    ax.set_ylabel(r"$E - E_F$ $(eV)$")
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.05)
    plt.show()
    return None


# find the fermi momentum for use in optimising parameters using definition:
fermi_momenta_indices = np.array(
    np.where(abs(e_vals_positive + e_vals_negative) < tolerance))[0]

# calculate direction of fermi vector from the magnitudes
fermi_momenta = np.zeros((2, 2))
fermi_momenta[:, 0] = [0, -k[fermi_momenta_indices[0]]]
fermi_momenta[:, 1] = [k[fermi_momenta_indices[1]]
                       * np.sqrt(3)/2, k[fermi_momenta_indices[1]]*1/2]

fermi_momenta[0][1] = k_point[0] - fermi_momenta[0][1]
fermi_momenta[1][1] = k_point[1] - fermi_momenta[1][1]

# print(k[fermi_momenta_indices])


# find the optimal parameters:
# found analytically from direct subsitution:
alpha_zeeman_opt = 1e-3 / np.sqrt(3)

# from ratio of soi's:
soi_ratio = 0.02  # can be tuned to find different results

# find soc terms at the fermi momentum:
gz_at_fermi_momentum = abs(
    np.sin(fermi_momenta[1][1]*a) - 2 * np.cos(np.sqrt(3)*fermi_momenta[0][1]*a/2) * np.sin(fermi_momenta[1][1]*a/2))
gr_at_fermi_momentum = magnitude([-np.sin(fermi_momenta[1][1]*a) - np.cos(np.sqrt(3)*fermi_momenta[0][1]*a/2) * np.sin(fermi_momenta[1][1]*a/2),
                                  np.sqrt(3)*np.sin(np.sqrt(3)*fermi_momenta[0][1]*a/2)*np.cos(fermi_momenta[1][1]*a/2)])

print(gz_at_fermi_momentum)
print(gr_at_fermi_momentum)

alpha_rashba_opt = soi_ratio * alpha_zeeman_opt * \
    (gz_at_fermi_momentum/gr_at_fermi_momentum)

print(alpha_rashba_opt)


beta_opt_numerator = 1 + (13e-3/2) / np.sqrt(alpha_rashba_opt**2 *
                                             gr_at_fermi_momentum**2 + alpha_zeeman_opt**2 * gz_at_fermi_momentum**2)
beta_opt_denominator = np.tanh(
    f(k_point[0], k_point[1]) - f(fermi_momenta[0][1], fermi_momenta[1][1]))

beta_opt = beta_opt_numerator / beta_opt_denominator

print(beta_opt)


print(f(k_point[0], k_point[1]))
print(f(fermi_momenta[0][0], fermi_momenta[1][0]))
