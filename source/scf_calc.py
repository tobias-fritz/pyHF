#!/usr/bin/env python3
#===============================================================================================
# Script for computing SCF energy of a diatomic molecule in python 3
#
#=============================================================================================== 

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

from basis import STO3G
from energy_terms import kinetic_integral, nuclear_attraction_integral, two_electron_integral, overlap_integral 

#=============================================================================================== 

def HF_SCF(r: float, 
           ZA: float, 
           ZB: float, 
           zeta1: float, 
           zeta2: float, 
           thresh: float, 
           iterations: int) -> float:
    ''' HartrE_elec-Fock algorithm for diatomic molecules with minimal basis sets

    args:
        r : float : internuclear distance
        ZA : float : nuclear charge A
        ZB : float : nuclear charge B
        zeta1 : float : zeta value A
        zeta2 : float : zeta value B
        thresh : float : input threshold for P matrix convergence
        iteration : int : number of iterations

    returns:
        float : the electronic energy
    '''
    r2 = r**2

    # STO-3G basis set
    coef = np.array((0.444635, 0.535328, 0.154329)) # the coefficients of the contracted gaussian
    alpha = np.array((0.109818, 0.405771, 2.22766)) # the exponents of the gaussian
    A = STO3G(zeta1, coef, alpha) # the STO-3G basis set for atom A with a1, d1
    B = STO3G(zeta2, coef, alpha) # the STO-3G basis set for atom B with a2, d2
    N = A[0].shape[0]

    # Kinetic energy integrals
    T11 = np.sum(kinetic_integral(A[0][:, None], A[0][None, :], 0.0) * A[1][:, None] * A[1][None, :])
    T12 = np.sum(kinetic_integral(A[0][:, None], B[0][None, :], r2) * A[1][:, None] * B[1][None, :])
    T21 = np.sum(kinetic_integral(B[0][:, None], A[0][None, :], r2) * B[1][:, None] * A[1][None, :])
    T22 = np.sum(kinetic_integral(B[0][:, None], B[0][None, :], 0.0) * B[1][:, None] * B[1][None, :])

    # Nuclear attraction integrals
    V11A = np.sum([nuclear_attraction_integral(A[0][i], A[0][j], 0.0, 0.0, ZA) * A[1][i] * A[1][j] for i in range(N) for j in range(N)])
    V12A = np.sum([nuclear_attraction_integral(A[0][i], B[0][j], r2, (B[0][j] * r / (A[0][i] + B[0][j]))**2, ZA) * A[1][i] * B[1][j] for i in range(N) for j in range(N)])
    V22A = np.sum([nuclear_attraction_integral(B[0][i], B[0][j], 0.0, r2, ZA) * B[1][i] * B[1][j] for i in range(N) for j in range(N)])
    V11B = np.sum([nuclear_attraction_integral(A[0][i], A[0][j], 0.0, r2, ZB) * A[1][i] * A[1][j] for i in range(N) for j in range(N)])
    V12B = np.sum([nuclear_attraction_integral(A[0][i], B[0][j], r2, (r - B[0][j] * r / (A[0][i] + B[0][j]))**2, ZB) * A[1][i] * B[1][j] for i in range(N) for j in range(N)])
    V22B = np.sum([nuclear_attraction_integral(B[0][i], B[0][j], 0.0, 0.0, ZB) * B[1][i] * B[1][j] for i in range(N) for j in range(N)])

    # Two-electron integrals
    V1111 = np.sum([two_electron_integral(A[0][i], A[0][j], A[0][k], A[0][l], 0.0, 0.0, 0.0) * A[1][i] * A[1][j] * A[1][k] * A[1][l] for i in range(N) for j in range(N) for k in range(N) for l in range(N)])
    V2111 = np.sum([two_electron_integral(B[0][i], A[0][j], A[0][k], A[0][l], r2, 0.0, (B[0][i] * r / (A[0][j] + B[0][i]))**2) * B[1][i] * A[1][j] * A[1][k] * A[1][l] for i in range(N) for j in range(N) for k in range(N) for l in range(N)])
    V2121 = np.sum([two_electron_integral(B[0][i], A[0][j], B[0][k], A[0][l], r2, r2, (B[0][i] * r / (A[0][j] + B[0][i]))**2) * B[1][i] * A[1][j] * B[1][k] * A[1][l] for i in range(N) for j in range(N) for k in range(N) for l in range(N)])
    V2211 = np.sum([two_electron_integral(B[0][i], B[0][j], A[0][k], A[0][l], 0.0, 0.0, r2) * B[1][i] * B[1][j] * A[1][k] * A[1][l] for i in range(N) for j in range(N) for k in range(N) for l in range(N)])
    V2221 = np.sum([two_electron_integral(B[0][i], B[0][j], B[0][k], A[0][l], 0.0, r2, (r - B[0][j] * r / (A[0][i] + B[0][j]))**2) * B[1][i] * B[1][j] * B[1][k] * A[1][l] for i in range(N) for j in range(N) for k in range(N) for l in range(N)])
    V2222 = np.sum([two_electron_integral(B[0][i], B[0][j], B[0][k], B[0][l], 0.0, 0.0, 0.0) * B[1][i] * B[1][j] * B[1][k] * B[1][l] for i in range(N) for j in range(N) for k in range(N) for l in range(N)])

    # Form overlap matrix
    S = np.array([[1.0, np.sum(overlap_integral(A[0][:, None], B[0][None, :], r2) * A[1][:, None] * B[1][None, :])], 
                  [np.sum(overlap_integral(A[0][:, None], B[0][None, :], r2) * A[1][:, None] * B[1][None, :]), 1.0]])

    # Hamiltonian matrix
    H = np.array([[T11 + V11A + V11B, T12 + V12A + V12B],
                  [T12 + V12A + V12B, T22 + V22A + V22B]])

    # Form the orthogonalization matrix
    X = np.array([[1.0 / np.sqrt(2 * (1 + S[0, 1])), 1.0 / np.sqrt(2 * (1 - S[0, 1]))],
                  [1.0 / np.sqrt(2 * (1 + S[0, 1])), -1.0 / np.sqrt(2 * (1 - S[0, 1]))]])

    # Form the two-electron integrals tensor
    T = np.array([[[[V1111, V2111], [V2111, V2211]], [[V2111, V2121], [V2121, V2221]]],
                  [[[V2111, V2121], [V2121, V2221]], [[V2211, V2221], [V2221, V2222]]]])

    result = []
    for iteration in range(iterations):
        print('iteration:', iteration)

        # Initialize P, the density matrix for the first iteration
        if iteration == 0:
            P = np.zeros((2, 2))

        if iteration == 0: # Initialize G, the G matrix for the first iteration
            G = np.zeros((2, 2))
        else:
            G = np.einsum('kl,ijkl->ij', P, T) - 0.5 * np.einsum('kl,ilkj->ij', P, T)

        # Calculate Fock matrix
        F = H + G

        # Calculate electronic energy E_elec = ½ * Σᵢⱼ Pᵢⱼ (Hᵢⱼ + Fᵢⱼ) + (Zₐ * Zᵦ) / r
        E_elec = 0.5 * np.einsum('ij,ij->', P, H + F) + ZA * ZB / r

        # Calculate F prime, this is the Fock matrix in the orthogonalized basis
        F_pr = np.dot(X.T, np.dot(F, X))

        # Diagonalize F prime for E, C prime, these are the eigenvalues and eigenvectors of the Fock matrix
        E, C_pr = np.linalg.eigh(F_pr)
        
        # Transform the eigenvectors back to the original basis
        C = np.dot(X, C_pr)

        # Copy P to OLDP
        P_old = P.copy()

        # Update P with the new eigenvectors
        P = 2.0 * np.outer(C[:, 0], C[:, 0])

        # Calculate delta
        delta = np.sqrt(np.mean((P - P_old)**2))
        result.append(E_elec)
    

        if delta < thresh:
            print(f'SCF converged after {iteration} iterations to {E_elec} a.u.')
            return E_elec
        
        else:
            continue

