# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:55:09 2025

@author: w10372hb
"""

import numpy as np


def delta(GF,T):

    return GF**5-T


def bracket(xu, xl, func, tol=1e-5, ms = 100,  debug=True):

    du = func(xu)
    dl = func(xl)

    if debug:
        print("du = "+str(du))
        print("dl = "+str(dl))

    xf = 0

    if du * dl > 0: # check if both have same sign
        print("poor choice of bounds")
        ms = 0

    # set up "old" params to go back to these if the 0 point is skipped over
    xuo = xu
    xlo = xl
    duo = du
    dlo = dl

    while ms > 0: # loop until max steps or acceptable delta is found


        if abs(du) < tol:

            xf = xu
            ms = 0

        elif abs(dl) < tol:

            xf = xl
            ms = 0

        elif du < 0:

            if debug:
                print("Both values are negative")

            dl = du
            xl = xu

            du = duo
            xu = xuo


        elif dl > 0:

            if debug:
                print("Both values are positive")

            du = dl
            xu = xl

            dl = dlo
            xl = xlo

        elif abs(du) > abs(dl):

            if debug:
                print(".")

            duo = du
            xuo = xu

            xu = (xu + xl) / 2
            du = func(xu)



        elif abs(du) <= abs(dl):

            if debug:
                print(".")

            dlo = dl
            xlo = xl

            xl = (xu + xl) / 2
            dl = func(xl)

        if debug:
            print("du = "+str(du))
            print("dl = "+str(dl))

        ms -= 1


    return xf

def range_of_brackets(func, xl, xu, y0, y1, ny , debug=True):


    y_arr = np.linspace(y0, y1, ny)
    x_arr = np.zeros_like(y_arr)
    for i, y in enumerate(y_arr):

        temp_func = lambda x: func(y,x)

        bracket_result = bracket(xu, xl, temp_func, debug = True)

        if debug:
            print("for y= {}, x= {}".format(y,bracket_result))

        x_arr[i] = bracket_result

    return y_arr, x_arr


def main():

    f = lambda H, T: delta( H, T)

    print(range_of_brackets(f, 0,5,0,5,10))


if __name__ == "__main__":
    main()