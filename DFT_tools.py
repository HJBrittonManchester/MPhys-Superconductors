# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:56:19 2024

@author: hbrit
"""
import numpy as np


def get_hamr(file):

    ndeg = np.array([])

    with open(file) as f:
        f.readline()  # skip the metadata line

        nb = int(f.readline())  # number of bands
        nr = int(f.readline())  # number of lattice points to consider

        rvec = np.zeros((3, nr), dtype=float)
        hamr = np.zeros((nb, nb, nr), dtype=np.complex128)

        for step in range(7):  # should calculate this number from nr
            ndeg = np.append(ndeg, np.array(
                f.readline().strip().split("    "), dtype=int))

        for ri in range(nr):
            for xi in range(nb):
                for yi in range(nb):
                    temp_data = f.readline().strip().split()

                    index_1, index_2 = int(
                        temp_data[3]) - 1,  int(temp_data[4]) - 1

                    rvec[:, ri] = np.array(temp_data[:3], dtype=float)
                    hamr[index_1, index_2, ri] = complex(float(
                        temp_data[5]), float(temp_data[6]))

    return hamr, ndeg, rvec


def find_hamk(k, hamr, ndeg, rvec):
    ham = np.zeros((2, 2, k.shape[1]), dtype=np.complex128)
    for i in range(k.shape[1]):
        for j in range(hamr.shape[2]):

            if np.linalg.norm(rvec[:, j]) > 9999:  # for debug purposes
                continue

            # Compute the phase factor
            # Ensure the correct sign in the phase
            phase = np.dot(k[:, i], rvec[:, j])

            # Add the contribution to the Hamiltonian in k-space
            ham[:, :, i] += hamr[:, :, j] * \
                complex(np.cos(phase), -np.sin(phase)) / ndeg[j]

    return ham

def find_hamk_a(k,hamr, ndeg, rvec, a):
    
    
    