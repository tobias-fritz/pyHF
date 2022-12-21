#!/usr/bin/env python3
#===============================================================================================
# Script for computing energy terms of the HF equation
# Date: 21.12.2022
# Author: Tobias Fritz
# Summary:
# Compute energy terms
#=============================================================================================== 

import numpy as np

#===============================================================================================

def T_e(A,B,r):
    ''' Kinetic energy term defined as
    
        T_AB = ∫ φ_A(r) ∇^2 φ_B(r) dr,
        
        with φ_A(r) and φ_B(r) two contracted gaussian type orbitals, through numerical integration.
        
        param:
            A:      first gaussian
            B:      second gaussian
            r:      interatomic distance
    '''

    c_A = A[0]  # coefficents of cGTO A
    c_B = B[0]  # coefficents of cGTO B
    d_A = A[1]  # exponents of cGTO A
    d_B = B[1]  # exponents of cGTO B

    ke = 0.0

    for i in range(3):          # hardcoded 3 for STO3G
        for j in range(3):      # hardcoded 3 for STO3G

            a = c_A[i]  # ith coefficent of gaussian A 
            b = c_B[j]  # jth coefficent of gaussian B 
            
            # reduced exponent
            red_exp = (a * b / (a+b)) 
            
            #
            ke += red_exp * (3 - 2 *  r * red_exp) * (np.pi/( a + b))**(3/2) * np.exp(- r * red_exp ) * d_A[i] * d_B[j]
    
    return  ke


def test_T_e():

    # check T11 of the H2 molecule
    c = np.array([0.444635,0.535328,0.154329])
    alpha = np.array([0.109818,0.405771,2.22766])
    zeta = 1.24

    A = STO3G(zeta,c,alpha,1)
    B = STO3G(zeta,c,alpha,1)

    assert pytest.approx(T_ee(A,B, 0), 0.0001) == 0.7600329435650853
