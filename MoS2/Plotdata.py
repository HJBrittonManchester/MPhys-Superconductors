# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:50:19 2024

@author: hbrit
"""
import numpy as np
import matplotlib.pyplot as plt

energyFile = "energy.dat"
kpointFile = "kpoints.dat"
bandFile = "band.dat"
enFile = "en.txt"

energydata = np.genfromtxt(energyFile, dtype=float)
endata = np.genfromtxt(enFile, delimiter = ",", dtype=float)
banddata = np.genfromtxt(bandFile, delimiter="   ", dtype=float)

kpointdata = np.genfromtxt(kpointFile, dtype=float)

resolution = energydata.shape[0]

#kpoints = kpointdata.reshape((resolution, resolution, 3))



def plot_color():
    alpha_range = np.linspace(0, 1, resolution)
    beta_range = np.linspace(0, 1, resolution)

    A, B = np.meshgrid(alpha_range, beta_range)

    plt.figure()

    c = plt.pcolormesh(A, B, energydata -endata, shading='auto')
    plt.contourf(A, B, energydata , levels=[-0.022, 0.022], colors='red', alpha=0.5)
    plt.contourf(A, B, endata , levels=[-0.022, 0.022], colors='black', alpha=0.5)
    
    plt.colorbar(c)
    plt.plot(1/3, 1/3, 'wx')
    plt.plot(2 * 1/3, 2 * 1/3, 'wx')

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\alpha$")
    plt.show()


# for ploting band from band.dat
def plot_band():
    plt.plot(banddata[:, 0], banddata[:, 1])
