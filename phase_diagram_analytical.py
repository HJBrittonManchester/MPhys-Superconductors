# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:53:25 2025

@author: zumai
"""

# repeat over a smaller range eg 5 -> 7 K

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.constants
from scipy.integrate import quad, tplquad
from scipy.special import digamma
from scipy.optimize import curve_fit
import time


kB = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]
MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]
h_bar = scipy.constants.physical_constants["Planck constant in eV/Hz"][0]
c_light = scipy.constants.speed_of_light
e_charge = scipy.constants.elementary_charge


ED = 0.022
Tc = 6.5
m_xy = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0] *\
    1e6/c_light**2  # electron mass
m_z = m_xy / 1000  # anisotropy factor


def energy(kx, ky, kz, m_xy, m_z):
    return h_bar**2 * ((kx**2 + ky**2)/m_xy + kz**2/m_z)


def K_term_1(m_xy, m_z, T, H, is_x=True):

    result = tplquad(lambda x, y, z: S1(energy(x, y, z, m_xy, m_z), T, H),
                     0.001, np.sqrt(m_xy*ED)/h_bar,
                     0.001, lambda x: np.sqrt(m_xy*ED/h_bar**2 - x**2),
                     0.001, lambda x, y: np.sqrt(
                         m_z*ED/h_bar**2 - m_z/m_xy * (x**2 + y**2)),
                     epsrel=1e-1)
    if is_x:
        return 1/4 * 2*h_bar**2/m_xy * 8*result[0]  # even function in 3D space
    else:
        return 1/4 * 2*h_bar**2/m_z * 8*result[0]


def K_term_2(m_xy, m_z, T, H, is_x=True):
    if is_x:
        result = tplquad(lambda x, y, z: x**2 * S2(energy(x, y, z, m_xy, m_z), T, H),
                         0.001, np.sqrt(m_xy*ED)/h_bar,
                         0.001, lambda x: np.sqrt(m_xy*ED/h_bar**2 - x**2),
                         0.001, lambda x, y: np.sqrt(
                             m_z*ED/h_bar**2 - m_z/m_xy * (x**2 + y**2)),
                         epsrel=1e-1)

        return 1/4 * 4*h_bar**4/m_xy**2 * 8*result[0]

    else:
        result = tplquad(lambda x, y, z: z**2 * S2(energy(x, y, z, m_xy, m_z), T, H),
                         0.001, np.sqrt(m_xy*ED)/h_bar,
                         0.001, lambda x: np.sqrt(m_xy*ED/h_bar**2 - x**2),
                         0.001, lambda x, y: np.sqrt(
                             m_z*ED/h_bar**2 - m_z/m_xy * (x**2 + y**2)),
                         epsrel=1e-1)

        return 1/4 * 4*h_bar**4/m_z**2 * 8*result[0]


def susc_2_direct(m_xy, m_z, T, H):
    K_term_1_vec = np.vectorize(K_term_1)
    K_term_2_vec = np.vectorize(K_term_2)
    K1 = K_term_1_vec(m_xy, m_z, T, H) + K_term_2_vec(m_xy, m_z, T, H)
    K2 = K_term_1_vec(m_xy, m_z, T, H, is_x=False) + \
        K_term_2_vec(m_xy, m_z, T, H, is_x=False)
    return 2*e_charge*H/(h_bar*c_light) * np.sqrt(K1*K2)
    # return 2*e_charge*H/(h_bar*c_light) * K1


def susc_2_angular(m_xy, m_z, T, H, theta):  # angle in x-z plane from x axis
    K_term_1_vec = np.vectorize(K_term_1)
    K_term_2_vec = np.vectorize(K_term_2)
    K1 = K_term_1_vec(m_xy, m_z, T, H) + K_term_2_vec(m_xy, m_z, T, H)
    K2 = K_term_1_vec(m_xy, m_z, T, H, is_x=False) + \
        K_term_2_vec(m_xy, m_z, T, H, is_x=False)
    return -2*e_charge*H/(h_bar*c_light) * \
        np.sqrt(K1*K2*np.cos(theta)**2 + (K1*np.sin(theta))**2)


"""
# general energy integral for even functions
def integral(func, T, H, limit=0.01, is_even=True):
    if is_even:
        return 2*quad(func, 0.001, limit, args=(T, H))[0]
    else:
        return quad(func, 0.001, limit, args=(T, H))[0]


# isotropic susceptibility functions
def susc_0_H_term(e, T, H):
    e_red = e / (2*kB*T)
    return (MU_B*H)**2/(4*kB*T) * (1/(e*np.cosh(e_red))**2 - 2*kB*T*np.tanh(e_red)/e**3)


def susc_0_enh(T, H):
    bcs_term = np.log((2*np.exp(np.euler_gamma)*ED)/(np.pi*kB*T))
    # digamma_term = digamma(1/2) - digamma(1/2 - 1j*MU_B*H/(kB*T))
    # return bcs_term + 1/2 * np.real(digamma_term)
    integral_vec = np.vectorize(integral)
    return bcs_term + integral_vec(susc_0_H_term, T, H)

"""
# anisotropic functions


def S1(e, T, H):
    e_red = e / (2*kB*T)
    if e_red < 700:
        return 1/(8*kB*T*e**2) * (e / np.cosh(e_red) / np.cosh(e_red) - 2*kB*T*np.tanh(e_red))
    return -1/(4*e**2)  # large cosh, tanh limits


def S2(e, T, H):
    e_red = e / (2*kB*T)
    if e_red < 700:
        return -1/(4*e*(kB*T)**2) * np.tanh(e_red) / np.cosh(e_red) / np.cosh(e_red)
    return 0


def susc_0_sqrt(T, H):
    result = quad(lambda x: 1/x * (np.tanh((x+MU_B*H)/(2*kB*T)) + np.tanh((x-MU_B*H)/(2*kB*T))),
                  0.0001, ED)
    return 1/4 * result[0]  # even function so *2 from 0->ED


"""
def susc_2_1(T, H):
    # mass term (mass anisotropy terms added in K1, K2 functions)
    result = quad(lambda r: 1/(np.cosh(r**2/(2*kB*T)))**2 - 2*kB*T/r**2 * np.tanh(r**2/(2*kB*T)),
                  0.0001, np.sqrt(ED))
    return np.pi**2/4 * 1/(h_bar*kB*T) * result[0]


def susc_2_2(T, H):
    # velocity term
    result = quad(lambda r: r**2 * np.tanh(r**2/(2*kB*T)) / np.cosh(r**2/(2*kB*T))**2,
                  0.0001, np.sqrt(ED))
    # -np.pi/3 * 1/(h_bar*(kB*T)**2) * result[0]
    return -np.pi/3 * 1/(h_bar*(kB*T)**2) * result[0]


def K(m_xy, m_z, T, H):
    # Kxx = 1, Kzz = K2. array[0] = K1, array[1] = K2
    susc_2_1_vec = np.vectorize(susc_2_1)
    susc_2_2_vec = np.vectorize(susc_2_2)
    s = susc_2_1_vec(T, H) + susc_2_2_vec(T, H)

    return np.array([np.sqrt(m_z)*s, m_xy/np.sqrt(m_z)*s])


def susc_2(m_xy, m_z, T, H):
    K_array = K(m_xy, m_z, T, H)
    return -2*e_charge/(h_bar*c_light) * np.sqrt(K_array[0]*K_array[1]) * H
    # this minus sign shouldnt be here but it makes it work
   # return 2*e_charge/(h_bar*c_light) * K_array[0] * H

"""


def delta(N0V, susc_func):
    # general delta function for plotting
    return 1/N0V - susc_func


def plot_phase_diagram(single_plot=True, d=None, d_masses=None):
    global N0V_fitted
    theta = 0.05

    X, Y = np.meshgrid(x, y)
    susc_0_sqrt_vec = np.vectorize(susc_0_sqrt)

    if single_plot:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=400)

        if d is None:  # work out explicitly
            data = np.zeros((len(X), len(Y)))
            # data[:, :] = delta(N0V_fitted, susc_0_sqrt_vec(
            #   X, Y) + susc_2_direct(m_xy, m_z, X, Y))
            data[:, :] = delta(N0V_fitted, susc_0_sqrt_vec(
                X, Y) + susc_2_angular(m_xy, m_z, X, Y, theta=theta))

            # cax = ax.pcolormesh(X, Y, data[:, :], cmap='bwr',
            #                   vmin=-0.02, vmax=0.02)
            cax = ax.scatter(X, Y, c=data[:, :], cmap='bwr',
                             s=75, vmin=-0.02, vmax=0.02)

        else:  # input own data with matching dimensions
            cax = ax.pcolormesh(X, Y, d[:, :], cmap='bwr',
                                vmin=-0.02, vmax=0.02)

        fig.colorbar(cax, label=r'$\Delta \; (T,H)$')
        ax.set_xlabel(r'$T$ (K)')
        ax.set_ylabel(r'$\mu_0 H_{c2}$ (T)')
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_title(
            r"$m_\perp / m_\parallel = {}, \theta = {:.2f}$ degrees".format(m_z/m_xy, np.rad2deg(theta)))

    else:

        if d is None:  # work out explicitly
            m_array = [0.01, 0.1, 1]
            data = np.zeros((len(X), len(Y), len(m_array)))

            fig, axs = plt.subplots(
                1, len(m_array), figsize=(6*len(m_array), 5), dpi=400)
            for i in range(len(m_array)):
                print(i)
                print("current time: {:.2f} s".format(time.time() - time_0))
                data[:, :, i] = delta(N0V_fitted, susc_0_sqrt_vec(
                    X, Y) + susc_2_direct(m_xy, m_xy*m_array[i], X, Y))

        else:  # input own data

            fig, axs = plt.subplots(
                1, len(d_masses), figsize=(6*len(d_masses), 5), dpi=400)
            for i in range(len(d_masses)):
                cax = axs[i].pcolormesh(X, Y, d[:, :, i], cmap='bwr',
                                        vmin=-0.02, vmax=0.02)
                axs[i].set_xlabel(r'$T$ (K)')
                axs[0].set_ylabel(r'$\mu_0 H_{c2}$ (T)')
                axs[i].set_xlim(x[0], x[-1])
                axs[i].set_ylim(y[0], y[-1])
                axs[i].set_title(
                    r"$m_\perp / m_\parallel = {}$".format(d_masses[i]))
            fig.colorbar(cax, label=r'$\Delta \; (T,H)$')

    return None


def gl_model(T, a, b):
    global Tc
    return a*(1-T/Tc)**b


def plot_data(d, mass_ratio, below_Tc_cutoff=3.5, above_Tc_cutoff=6.5, plot_fit=False):

    # convert into index in format of data. won't work generally
    m = int(np.log10(100*mass_ratio))

    transition_indices = np.argmin(np.abs(d), axis=0)[:, m]
    transition_fields = y[transition_indices]

    red_indices = np.where(np.logical_and(
        x >= below_Tc_cutoff, x <= above_Tc_cutoff))
    x_red = x[red_indices]
    transition_fields = transition_fields[red_indices]
    field_err = np.zeros_like(transition_fields)
    field_err[:] = abs(y[1] - y[0])/2  # set error to half of pixel width

    fig, ax = plt.subplots(figsize=(7, 5), dpi=400)
    ax.errorbar(x_red, transition_fields,  field_err, fmt='kx',
                label='Data')

    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$\mu_0 H_{c2}$ (T)')
    ax.set_xlim(below_Tc_cutoff, above_Tc_cutoff)
    # ax.set_ylim(0, transition_fields.max())

    if plot_fit:
        params = curve_fit(gl_model, x_red, transition_fields,
                           sigma=field_err, maxfev=1000)
        print(
            r"mass ratio = {}: H_c2(T=0) = {:.2f} T, critical exponent = {:.2f} ± {:.2f}".format(
                mass_ratio, *(params[0]), np.sqrt(np.diag(params[1]))[1]))
        ax.plot(x_red, gl_model(x_red, *(params[0])),
                c='r', label="Fit to G-L Model")
        plt.legend(loc="upper right", fontsize=8)
        return params

    plt.legend(loc="upper right", fontsize=8)
    return None


time_0 = time.time()

N0V_fitted = 1/(susc_0_sqrt(6.5, 0.))

angles = np.linspace(0, 45, 5)  # in degrees
H_values = np.linspace(0., 2, 5)
X, Y = np.meshgrid(angles, H_values)

susc_0_sqrt_vec = np.vectorize(susc_0_sqrt)
T = 6.4


data = np.zeros(len(H_values))
for i, angle in enumerate(angles):
    delta_values = delta(N0V_fitted, susc_0_sqrt_vec(T, H_values) +
                         susc_2_angular(m_xy, m_z, T, H_values, np.deg2rad(angle)))
    print(delta_values)

    H_index = np.argmin(np.abs(delta_values), axis=0)
    H = H_values[H_index]
    print("angle = {:.2f} degrees".format(angle))
    print("Hc2 = {:.2f} T\n".format(H))
    data[i] = H

plt.plot(angles, data)


"""
N0V_fitted = 1/(susc_0_sqrt(6.5, 0.))

# X, Y = np.meshgrid(x, y)
susc_0_sqrt_vec = np.vectorize(susc_0_sqrt)

fig, ax = plt.subplots(figsize=(6, 5), dpi=400)

data = np.zeros((len(X), len(Y)))
data[:, :] = delta(N0V_fitted, susc_0_sqrt_vec(
    T, Y) + susc_2_angular(m_xy, m_z, T, Y, np.deg2rad(X)))

cax = ax.scatter(X, Y, c=data[:, :], cmap='bwr', s=75,
                 vmin=-0.02, vmax=0.02)

fig.colorbar(cax, label=r'$\Delta \; (T,H)$')
ax.set_xlabel(r'$\theta$ (degrees)')
ax.set_ylabel(r'$\mu_0 H_{c2}$ (T)')
# ax.set_xlim(x[0], x[-1])
# ax.set_ylim(y[0], y[-1])
ax.set_title(r"$m_\perp / m_\parallel = {}$".format(m_z/m_xy))
"""

"""
# to work out data

# phase space
x = np.linspace(5.5, 7., 5)
y = np.linspace(0., 10., 5)

N0V_fitted = 1/(susc_0_sqrt(6.5, 0.))  # + susc_2_direct(m_xy, m_z, 6.5, 0.))

plot_phase_diagram()

#np.save("phase_diagram_analytical_highres.npy", data)
"""

"""
# to load data
d = np.load("phase_diagram_analytical_highres.npy")

# plot_phase_diagram(single_plot=True, d=d)


for m in [0.01, 0.1, 1]:
    plot_data(d, m, below_Tc_cutoff=6.2,
              above_Tc_cutoff=6.5, plot_fit=True)

"""
print("time taken: {:.2f} s".format(time.time() - time_0))
