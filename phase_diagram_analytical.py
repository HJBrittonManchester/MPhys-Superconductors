# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:53:25 2025

@author: zumai
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.constants
from scipy.integrate import quad, tplquad
from scipy.special import digamma
import time


kB = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]
MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]
h_bar = scipy.constants.physical_constants["Planck constant in eV/Hz"][0]
c_light = scipy.constants.speed_of_light
e_charge = scipy.constants.elementary_charge

# changing m_xy (order 1e-2 -> 1) gives dimensional crossover, but m_z dependence cancels
# out in K1*K2 so that it has no overall impact. weird
ED = 0.022
Tc = 6.5
m_xy = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0] *\
    1e6/c_light**2  # electron mass
m_z = m_xy / 100  # anisotropy factor


# trying to do 3d integral directly
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
                         # -np.sqrt(m_xy*ED)/h_bar, np.sqrt(m_xy*ED)/h_bar,
                         # lambda x: -np.sqrt(m_xy*ED/h_bar**2 - x**2),
                         # lambda x: np.sqrt(m_xy*ED/h_bar**2 - x**2),
                         # lambda x, y: -np.sqrt(m_z*ED/h_bar **
                         #                       2 - m_z/m_xy * (x**2 + y**2)),
                         # lambda x, y: np.sqrt(m_z*ED/h_bar**2 - m_z/m_xy * (x**2 + y**2)))
                         0.001, np.sqrt(m_xy*ED)/h_bar,
                         0.001, lambda x: np.sqrt(m_xy*ED/h_bar**2 - x**2),
                         0.001, lambda x, y: np.sqrt(
                             m_z*ED/h_bar**2 - m_z/m_xy * (x**2 + y**2)),
                         epsrel=1e-1)

        return 1/4 * 4*h_bar**4/m_xy**2 * 8*result[0]

    else:
        result = tplquad(lambda x, y, z: z**2 * S2(energy(x, y, z, m_xy, m_z), T, H),
                         # -np.sqrt(m_xy*ED)/h_bar, np.sqrt(m_xy*ED)/h_bar,
                         # lambda x: -np.sqrt(m_xy*ED/h_bar**2 - x**2),
                         # lambda x: np.sqrt(m_xy*ED/h_bar**2 - x**2),
                         # lambda x, y: -np.sqrt(m_z*ED/h_bar **
                         #                       2 - m_z/m_xy * (x**2 + y**2)),
                         # lambda x, y: np.sqrt(m_z*ED/h_bar**2 - m_z/m_xy * (x**2 + y**2)))
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
    return -2*e_charge*H/(h_bar*c_light) * np.sqrt(K1*K2)


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
    #digamma_term = digamma(1/2) - digamma(1/2 - 1j*MU_B*H/(kB*T))
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


def plot(single_plot=True, plot_fit=True):
    global N0V_fitted
    # phase space
    x = np.linspace(3.5, 7., 3)
    y = np.linspace(0., 15., 3)
    X, Y = np.meshgrid(x, y)
    susc_0_sqrt_vec = np.vectorize(susc_0_sqrt)

    if single_plot:
        data = np.zeros((len(X), len(Y)))
        data[:, :] = delta(N0V_fitted, susc_0_sqrt_vec(
            X, Y) + susc_2_direct(m_xy, m_z, X, Y))
        fig, ax = plt.subplots(figsize=(6, 5), dpi=400)
        cax = ax.scatter(X, Y, c=data[:, :], cmap='bwr',
                         s=30, vmin=-0.05, vmax=0.05)
        fig.colorbar(cax, label=r'$\Delta \; (T,H)$')

        ax.set_xlabel(r'$T$ (K)')
        ax.set_ylabel(r'$\mu_0 H_{c2}$ (T)')
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_title(r"$m_\perp / m_\parallel = {}$".format(m_z/m_xy))

    else:
        m_array = [1, 0.01]
        data = np.zeros((len(X), len(Y), len(m_array)))

        fig, axs = plt.subplots(1, len(m_array), figsize=(12, 5), dpi=400)
        for i in range(len(m_array)):
            print(i)
            data[:, :, i] = delta(N0V_fitted, susc_0_sqrt_vec(
                X, Y) + susc_2_direct(m_xy, m_xy*m_array[i], X, Y))
            cax = axs[i].scatter(X, Y, c=data[:, :, i], cmap='bwr',
                                 s=30, vmin=-0.05, vmax=0.05)
            axs[i].set_xlabel(r'$T$ (K)')
            axs[0].set_ylabel(r'$\mu_0 H_{c2}$ (T)')
            axs[i].set_xlim(x[0], x[-1])
            axs[i].set_ylim(y[0], y[-1])
            axs[i].set_title(
                r"$m_\perp / m_\parallel = {}$".format(m_array[i]))
            # axs[i].annotate(r"$m_\perp / m_\parallel = {}$".format(m_array[i]), xy=(3.575, 12.75), xycoords='data',
            #               size=9, ha='left', va='top', bbox=dict(boxstyle='round', fc='w'))
        fig.colorbar(cax, label=r'$\Delta \; (T,H)$')

    return data


def gl_model_2D(T, a, is_2D):
    global Tc
    if is_2D:
        return a*np.sqrt(1-T/Tc)
    else:
        return a*(1-T/Tc)


time_0 = time.time()
"""
N0V_fitted = 1/(susc_0_sqrt(6.5, 0.))  # + susc_2_direct(m_xy, m_z, 6.5, 0.))

data = plot()
print(data)

# #np.save("phase_diagram_analytical.npy", data)
"""
d = np.load("phase_diagram_analytical.npy")
print(d.shape)

m_array = [0.01, 0.1, 1]
x = np.linspace(3.5, 7., 25)
y = np.linspace(0., 15., 25)
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(1, len(m_array), figsize=(6*len(m_array), 5), dpi=600)
for i in range(len(m_array)):
    cax = axs[i].scatter(X, Y, c=d[:, :, i], cmap='bwr',
                         s=100, vmin=-0.02, vmax=0.02, marker='o')
    axs[i].set_xlabel(r'$T$ (K)')
    axs[0].set_ylabel(r'$\mu_0 H_{c2}$ (T)')
    axs[i].set_xlim(x[0], x[-1])
    axs[i].set_ylim(y[0], y[-1])
    axs[i].set_title(
        r"$m_\perp / m_\parallel = {}$".format(m_array[i]))
    # axs[i].annotate(r"$m_\perp / m_\parallel = {}$".format(m_array[i]), xy=(3.575, 12.75), xycoords='data',
    #               size=9, ha='left', va='top', bbox=dict(boxstyle='round', fc='w'))
fig.colorbar(cax, label=r'$\Delta \; (T,H)$')

print("time taken: {:.2f} s".format(time.time() - time_0))

# data = delta(N0V_fitted, susc_0_sqrt_vec(X, Y) + susc_2(m_xy, m_z, X, Y))
# phase_points = np.zeros((len(X), len(Y)))
# phase_indices = np.zeros(len(X))
# for i, row in enumerate(data):
#     phase_indices[i] = int(np.argmin(row))

# print(phase_indices)
# phase_points = data[phase_indices]
# plt.scatter(X, Y, c=phase_points)
