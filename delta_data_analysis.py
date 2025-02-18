# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:53:12 2025

@author: zumai
"""

import numpy as np
import matplotlib.pyplot as plt

susc_array = np.load("Data/susceptibility_T_65_5.npy")
susc_fixedt = np.genfromtxt("Data/susc_field_data_t6_3.txt", delimiter="\t")


fig, ax = plt.subplots(figsize=(10, 6), dpi=400)

H_array = np.linspace(0., 100, 25)
T_array = np.linspace(6.5, 5, 25)
v = -2.0798017848698338e-14  # fitted for this system


def susc_fit(susc_array, n):
    x = np.linspace(susc_array[0, 0], susc_array[-1, 0], 100)
    susc_fit_params = np.polyfit(
        susc_fixedt[:, 0], 1-v*susc_fixedt[:, 1], int(n))
    print(susc_fit_params)

    polynomial = np.zeros(len(x))

    for i in range(n):
        polynomial += susc_fit_params[n-i]*x**i

    ax.plot(x, polynomial, 'k--')

    return None


susc_fixedt_red = susc_fixedt[:]

ax.plot(susc_fixedt_red[:, 0], 1-v *
        susc_fixedt_red[:, 1], marker='x', color='r')
susc_fit(susc_fixedt_red, 6)

# ax.set_xlim(0, 150)
ax.set_ylim(-2, 2)


for i in range(5):
    print("\nT = {}".format(T_array[i]))
    for j in range(25):
        print("[{}, {}],".format(H_array[j], susc_array[i][j]))

    ax.plot(H_array, 1 - v*susc_array[i], marker='x',
            markersize=5, label="T = {:.2f} K".format(T_array[i]))


ax.set_xlabel("In-Plane Magnetic Field (T)")
ax.set_ylabel(r"$\Delta = 1 - V \chi$")
#ax.axhline(ls='--', c='k')
ax.set_xlim(0, 102)

plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
