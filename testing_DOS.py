# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:54:45 2024

@author: w10372hb
"""

import numpy as np
import matplotlib.pyplot as plt

def energy(k):
    
    return 7 * (k- 0.5)**2 - 0.15

def g(E):
    return  2/(14*np.sqrt((E+0.15)/7))

def DOS(E, k, sigma = 1e-3):
    
    return (np.exp(-(E - energy(k))**2 /(2* sigma**2) ) / (sigma * np.sqrt(2*np.pi))).sum() / k.size


test_k = np.linspace(0,1,1000)

test_e = np.linspace(-0.12, 0.12)

for i in range(1,4):
    sigma = 10**(-i)

    density_of_state = [DOS(e,test_k,sigma) for e in test_e]
    
    plt.plot(test_e,density_of_state,'x', label=r"$\log \sigma = {}$".format(np.log10(sigma)))


plt.plot(test_e, g(test_e),'r', label = "analytic" )
plt.legend()
plt.xlabel("Energy / eV")
plt.ylabel("Density of State")
plt.title(r"$E = A(k-k_0)^2 + E_0$")