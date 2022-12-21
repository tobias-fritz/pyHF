#!/usr/bin/env python3
#===============================================================================================
# Script for computing contracted Gaussian type orbitals
# Date: 30.05.2022
# Author: Tobias Fritz
# Summary:
# Gives coefficients and exponents of the contracted gaussian that are used in the HF method 
# implemented through using numerical integration and restricted to diatomic molecules at this 
# point. 
#=============================================================================================== 

import numpy as np

#=============================================================================================== 

def STO3G(zeta, c, alpha, r):
    ''' Slater type molecular orbital, a contraction (sum) of three gaussians (= STO-3G)

        STO = N*r^(n-1) * exp(-ζr) * Y_lm,

        with Y_lm the sphirical part. The contracted gaussian cGTO is defined through

        cGTO = c * ((2 * alpha * ζ^2) / pi )^3/4 * exp(-alpha * ζ^2 * r^2).

        with the molecular orbital coefficient c and the exponent alpha * ζ^2 .

        param:
            zeta    :      zeta value
            c       :      coefficients of the cointracted gaussian
            alpha   :      exponent of the gaussian 
            r       :      interatomic distance 

        returns the contracted gaussian as a tupel of the  orbital coeeficients 
        and exponents.
    '''

    alpha = alpha * zeta**2
    c = c * (( 2 * alpha)/ np.pi)**(3/4)

    cGTO = (alpha, c)

    return cGTO
