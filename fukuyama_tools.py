# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:19:12 2025

@author: hbrit
"""
import numpy as np
import matplotlib.pyplot as plt

from GF_tools import get_greens_function, matsubara_frequency

def H(k, u=.5, v = 1):
    
    hamiltonian_arr = np.zeros((len(k),2,2),dtype=np.complex128)
    hamiltonian_arr[:,0,1] = u + v * np.exp(-1j * k)
    hamiltonian_arr[:,1,0] = u + v * np.exp(1j * k)

    
    return hamiltonian_arr

def energy_spectrum(res = 1000):
    k_arr = []
    en_pos_arr = []
    en_neg_arr = []
    for i in range(res):
        k = i / (res-1) * 2* np.pi
        en, vec = np.linalg.eig(H(k))
        en = np.real_if_close(en)

        k_arr.append(k)
        en_pos_arr.append(en[0])
        en_neg_arr.append(en[1])
    return k_arr, en_pos_arr, en_neg_arr


def v(ham, k_spacing):
    


k = np.linspace(0, np.pi * 2)


hamk = H(k)

eigVal, eigVec = np.linalg.eig(hamk)

## find velocity





