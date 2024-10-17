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

def overlap_integral(a: float, b: float, r: float) -> float:
    ''' Overlap integral between two contracted gaussians

    S_AB = ∫ φ_A(r) φ_B(r) dr 

    args:
        a : float : exponent of the first gaussian
        b : float : exponent of the second gaussian
        r : float : interatomic distance

    returns
        float : the overlap integral
    '''

    return np.exp(-(a * b * r / (a + b))) * (np.pi / (a+ b))**(3/2)

def kinetic_integral(a: float, b: float, r: float) -> float:
    ''' Kinetic energy integral between two contracted gaussians

    T_AB = ∫ φ_A(r) (-1/2 ∇^2) φ_B(r) dr

    args:
        a : float : exponent of the first gaussian
        b : float : exponent of the second gaussian
        r : float : interatomic distance

    returns:
        float : the kinetic energy integral
    '''

    return (a * b / (a + b)) * (3 - 2 * a * b * r / (a + b)) * \
           (np.pi / (a + b))**(3/2) * np.exp(-a * b * r / (a + b))

def nuclear_attraction_integral(a: float, b: float, r1: float, r2: float, Z: float) -> float:
    ''' Nuclear attraction integral between two contracted gaussians

    V_AB = ∫ φ_A(r1) φ_B(r2) V(r1) dr1 dr2

    args:
        a : float : exponent of the first gaussian
        b : float : exponent of the second gaussian
        r1 : float : interatomic distance
        r2 : float : interatomic distance
        Z : float : nuclear charge

    returns:
        float : the nuclear attraction integral
    '''

    F0 = lambda x: (1.0 - x / 3.0) if x < 1e-6 else (0.5 * (((np.pi / x) ** 0.5) * erf(x ** 0.5))) # Boys function

    return -Z * 2 * np.pi / (a + b) * F0((a + b) * r2) * np.exp(-a * b * r1 / (a + b))

def two_electron_integral(a: float, b: float, c: float, d: float, r1: float, r2: float, r3: float) -> float:
    ''' Two electron integral between four contracted gaussians

    (AB|CD) = ∫ φ_A(r1) φ_B(r2) φ_C(r3) φ_D(r4) dr1 dr2 dr3 dr4

    args:
        a : float : exponent of the first gaussian
        b : float : exponent of the second gaussian
        c : float : exponent of the third gaussian
        d : float : exponent of the fourth gaussian
        r1 : float : interatomic distance between the first and second gaussian
        r2 : float : interatomic distance between the third and fourth gaussian
        r3 : float : interatomic distance between the first and third gaussian

    returns:
        float : the two electron integral
    '''
    F0 = lambda x: (1.0 - x / 3.0) if x < 1e-6 else (0.5 * (((np.pi / x) ** 0.5) * erf(x ** 0.5)))


    return 2 * (np.pi**2.5) / ((a + b) * (c + d) * (a + b + c + d)**0.5) * \
           F0(((a + b) * (c + d) * r3 / (a + b + c + d))) * \
           np.exp(-a * b * r1 / (a + b) - c * d * r2 / (c + d))

"""
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

"""
