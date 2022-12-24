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

    # calculate reduced exponent
    k = lambda a,b: (a * b / (a+b)) 
    
    # expression for the kinetic energy
    kinetic_energy = lambda c_A, c_B, d_A, d_B: k(c_A,c_B) * (3 - 2 *  r * k(c_A,c_B)) * (np.pi/( c_A + c_B))**(3/2) * np.exp(- r * k(c_A,c_B) ) * d_A * d_B
    
    # calculate the kinetic energy for all permutations (this performed better than np.meshgrid and similar to itertools.product)
    T = [kinetic_energy(c_A,c_B,d_A,d_B) for c_A, d_A in zip(A[0],A[1]) for c_B, d_B in zip(B[0],B[1])]
    
    return  sum(T)

def test_T_e():

    # check T11 of the H2 molecule
    c = np.array([0.444635,0.535328,0.154329])
    alpha = np.array([0.109818,0.405771,2.22766])
    zeta = 1.24

    A = STO3G(zeta,c,alpha,1)
    B = STO3G(zeta,c,alpha,1)

    assert pytest.approx(T_ee(A,B, 0), 0.0001) == 0.7600329435650853
