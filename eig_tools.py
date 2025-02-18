# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:22:09 2025

@author: w10372hb
"""

import numpy as np

def diagonalise(mat_arr):
    eigenvalues = np.zeros((mat_arr.shape[0], mat_arr.shape[-1]), dtype=complex)
    eigenvectors = np.zeros((mat_arr.shape[0],mat_arr.shape[1], mat_arr.shape[-1]), dtype=complex)

    for i in range(mat_arr.shape[-1]):
        eigvals, eigvecs = np.linalg.eig(mat_arr[:, :, i])
        eigenvalues[:, i] = eigvals
        eigenvectors[:, :, i] = eigvecs

    return eigenvalues, eigenvectors



def get_eig_vec(mat_arr):
    return diagonalise(mat_arr)[1]

def projection_z(mat_arr, band=0):
    eigvecs = get_eig_vec(mat_arr)

    proj = eigvecs[:, 0] * eigvecs[:, 0].conj() - \
        eigvecs[:,  1] * eigvecs[:, 1].conj()

    return proj


def projection_x(mat_arr, band=0):
    eigvecs =  get_eig_vec(mat_arr)

    proj = eigvecs[:, :, 0] * eigvecs[:, :, 1].conj() + \
        eigvecs[:, :, 1] * eigvecs[:, :, 0].conj()

    return proj


def projection_y(mat_arr, band=0):
    eigvecs = get_eig_vec(mat_arr)

    proj = 1j * eigvecs[:, :, 0] * eigvecs[:, :, 1].conj() - \
        eigvecs[:, :, 1] * eigvecs[:, :, 0].conj() * 1j

    return proj