# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:39:46 2024

@author: w10372hb
"""

import numpy as np


#########################################################
# get_k_XXXX methods


def get_better_k_square(region_res, centre =(1/2,1/2), scale=1):
    full_BZ_res = int(region_res / scale**2)

    k = np.zeros((3, region_res*region_res))

    for xi in range(region_res):
        for yi in range(region_res):
            k[0, xi + region_res * yi] = 2 * np.pi * ((xi / (region_res) - 1/2) * scale + centre[0])
            k[1, xi + region_res * yi] = 2 * np.pi * ((yi / (region_res) - 1/2) * scale + centre[1])
    return k, full_BZ_res


def dist(x,y, c = (0,0), oblique=.8):
    return np.sqrt((x- c[0])**2 + (y-c[1])**2 +oblique*(x- c[0]) * (y-c[1]))


def get_close_k_points(full_bz_res, centre = (1/3,1/3), width = .2, thresholds=(.68,.8), strength=5.5, print_stats=False):

    box_size = int(full_bz_res * width)

    threshold_L = thresholds[0]
    threshold_U = thresholds[1]


    kx_ = np.linspace(centre[0] - width/2, centre[0] + width/2, box_size)
    ky_ = np.linspace(centre[1] - width/2, centre[1] + width/2, box_size)
    kx,ky = np.meshgrid(kx_,ky_)



    potential = 1 / (strength * dist(kx,ky,centre) + 1) #+ 1 / (strength * dist(kx,ky,(2/3,2/3))+1)

    z = np.where(np.logical_and(potential > threshold_L,potential < threshold_U) , 1,0)

    kx *= 2 * np.pi
    ky *= 2 * np.pi


    num_sig_kpoints = kx[np.logical_and(potential > threshold_L,potential < threshold_U)].shape[0]

    if print_stats:
        print("{} kpoints chosen out of {} across BZ. BZ yield of {:.2g}% ".format(
            num_sig_kpoints,full_bz_res**2, 100 * num_sig_kpoints/(full_bz_res**2)))

        print("{} kpoints chosen out of {} across cropped area. cropped yield of {:.2g}% ".format(
            num_sig_kpoints,box_size**2, 100 * num_sig_kpoints/(box_size**2)))


    sig_kx = kx[np.logical_and(potential > threshold_L,potential < threshold_U)]
    sig_ky = ky[np.logical_and(potential > threshold_L,potential < threshold_U)]
    sig_z = z[np.logical_and(potential > threshold_L,potential < threshold_U)]


    return np.vstack((sig_kx,sig_ky, np.zeros_like(sig_ky)))


def get_k_block(res, size_of_box = -1):

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