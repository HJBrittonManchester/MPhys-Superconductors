# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:22:09 2025

@author: w10372hb
"""

import numpy as np
import scipy.linalg as LA

def epsilon(hamk):
    eigs = np.array([np.real_if_close(LA.eig(hamk[:, :, i])[0])
                    for i in range(hamk.shape[2])], dtype=float)
    return eigs


def get_eig_vec(hamk):
    return np.array([LA.eig(hamk[:, :, i])[1]
                     for i in range(hamk.shape[2])], dtype=complex)

def projection_z(hamk, band=0):
    eigvecs = np.array([LA.eig(hamk[:, :, i])[1]
                        for i in range(hamk.shape[2])], dtype=complex)

    proj = eigvecs[:, 0] * eigvecs[:, 0].conj() - \
        eigvecs[:,  1] * eigvecs[:, 1].conj()

    return proj


def projection_x(hamk, band=0):
    eigvecs = np.array([LA.eig(hamk[:, :, i])[1]
                        for i in range(hamk.shape[2])], dtype=complex)

    proj = eigvecs[:, :, 0] * eigvecs[:, :, 1].conj() + \
        eigvecs[:, :, 1] * eigvecs[:, :, 0].conj()

    return proj


def projection_y(hamk, band=0):
    eigvecs = np.array([np.linalg.eig(hamk[:, :, i])[1]
                        for i in range(hamk.shape[2])], dtype=complex)

    proj = 1j * eigvecs[:, :, 0] * eigvecs[:, :, 1].conj() - \
        eigvecs[:, :, 1] * eigvecs[:, :, 0].conj() * 1j

    return proj