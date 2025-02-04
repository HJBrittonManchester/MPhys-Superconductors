# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:31:52 2025

@author: w10372hb
"""

import numpy as np
import scipy.constants


k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]


##############################################################################
# Susceptibility functions

def get_greens_function(ham, freq):
    greens = np.zeros_like(ham, dtype=complex)
    det = (1j*freq - ham[0, 0, :])*(1j*freq -
                                    ham[1, 1, :]) - (-ham[0, 1, :]*-ham[1, 0, :])

    greens[0, 0, :] = (1j*freq - ham[1, 1, :])
    greens[1, 1, :] = (1j*freq - ham[0, 0, :])
    greens[0, 1, :] = ham[0, 1, :]
    greens[1, 0, :] = ham[1, 0, :]

    greens = greens / det
    return greens


def matsubara_frequency(T, m):
    return (2*m+1)*k_B * T * np.pi


    

def susc(ham_N, ham_P, T, n_freq):
    '''
    
    Parameters
    ----------
    ham_N : TYPE
        Negative momentum hamiltonian.
    ham_P : TYPE
        Positive momentum hamiltonian.
    T : float
        Temperature.
    n_freq : int
        The number of frequencies to use.

    Returns
    -------
    TYPE
        Un-normalized suseptibility contributions for each given k-point.

    '''
    chi_0 = np.zeros(ham_N.shape[2], dtype=complex)

    for m in range(-n_freq, n_freq):
        current_freq = matsubara_frequency(T, m)
        greens_N = get_greens_function(ham_N, -current_freq)
        greens_P = get_greens_function(ham_P, current_freq)

        chi_0 -= greens_P[0, 0, :] * greens_N[1, 1, :] - \
            greens_P[0, 1, :]*greens_N[1, 0, :]

    return np.real_if_close(k_B * T * chi_0, 1e-4)
