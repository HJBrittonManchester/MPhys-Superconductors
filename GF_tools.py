# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:31:52 2025

@author: w10372hb
"""

import numpy as np
import scipy.constants


k_B = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]
MU_B = scipy.constants.physical_constants["Bohr magneton in eV/T"][0]



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


def Kubo_susceptibility_unnorm(hamk_W, v_H, U, T, n_freq, PLOT=False):

    v_x = v_H[:,:,0]
    v_y = v_H[:,:,1]


    gf_W = np.zeros((2,2,hamk_W.shape[-1], 2* n_freq),dtype=complex)
    I = np.array([[1,0],[0,1]])

    for m in range(2*n_freq):
        #print(m)

        gf_W[:,:,:,m] = get_greens_function(hamk_W, matsubara_frequency(T,m - n_freq))



    gf_H = np.einsum("jim , jkmn, klm -> ilmn", U.conj(), gf_W, U)


    mat_mul =np.einsum("abmn, bcm, cdmn, dem, efmn, fgm, ghmn, ham -> m", gf_H, v_x, gf_H, v_y, gf_H, v_x, gf_H, v_y)

    #np.einsum("abmn, bcmn, cdmn, demn-> m", gf_H, gf_H,gf_H, gf_H)


    if PLOT:
        return T * mat_mul

    return  -T * mat_mul.sum() * MU_B**2


def test_susc(hamk_W_p, hamk_W_n, T, n_freq, PLOT=False):
    chi = 0
    
    for m in range(2*n_freq):
        #print(m)
        
        gf_p = get_greens_function(hamk_W_p, matsubara_frequency(T,m - n_freq))
        gf_n = get_greens_function(hamk_W_n, -matsubara_frequency(T,m - n_freq))

        chi += (gf_p[0,0] * gf_n[1,1] - gf_p[0,1] * gf_n[1,0]).sum()
        
    return - T * chi


    