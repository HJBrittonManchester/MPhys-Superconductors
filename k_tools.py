# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:39:46 2024

@author: w10372hb
"""

import numpy as np


#########################################################
# get_k_XXXX methods


def get_k_block(res, size_of_box):

    if size_of_box == -1:  # use full space

        k = np.zeros((3, res*res))

        for xi in range(res):
            for yi in range(res):
                k[0, xi + res * yi] = 2 * np.pi * xi / (res)
                k[1, xi + res * yi] = 2 * np.pi * yi / (res)
        return k
    else:

        reduced_res = int(size_of_box * res)

        # box around 1/3, 1/3

        k = np.zeros((3, 2 * reduced_res * reduced_res))

        for xi in range(reduced_res):
            for yi in range(reduced_res):
                k[0, xi + reduced_res * yi] = 2 * \
                    np.pi * (xi / res + 1/3 - size_of_box/2)
                k[1, xi + reduced_res * yi] = 2 * \
                    np.pi * (yi / res + 1/3 - size_of_box/2)

        # box around 2/3, 2/3

        for xi in range(reduced_res):
            for yi in range(reduced_res):
                k[0, xi + reduced_res * yi + reduced_res * reduced_res] = 2 * \
                    np.pi * (xi / res + 2/3 - size_of_box/2)
                k[1, xi + reduced_res * yi + reduced_res * reduced_res] = 2 * \
                    np.pi * (yi / res + 2/3 - size_of_box/2)

        return k


def get_k_path(path,  res):
    name_points = {'G': np.array([0, 0, 0]), 'K': np.array(
        [1/3, 1/3, 0]), 'M': np.array([0, 1/2, 0])}

    path_lengths = np.zeros(len(path)-1)

    for i in range(len(path)-1):
        path_lengths[i] = np.linalg.norm(
            (name_points[path[i+1]] - name_points[path[i]]))

    path_size = np.array(path_lengths * res /
                         np.sum(path_lengths), dtype=int)

    kpath = []

    for i in range(len(path)-1):
        for j in range(res):
            kpath.append(j / res * (name_points[path[i+1]] -
                         name_points[path[i]]) + name_points[path[i]])

    return np.array(kpath).T * 2 * np.pi


def get_k_path_spacing(path):
    spacing = np.zeros(len(path[0]))

    for i in range(1, len(path[0])):

        spacing[i] = np.linalg.norm(path[:, i] - path[:, i-1]) + spacing[i-1]

        # print(spacing[i] - spacing[i-1])

    return spacing / spacing[-1]