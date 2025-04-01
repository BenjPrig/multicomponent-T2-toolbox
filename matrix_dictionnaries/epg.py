#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Robust myelin water imaging from multi-echo T2 data using second-order Tikhonov regularization with control points
# ISMRM 2019, Montreal, Canada. Abstract ID: 4686
# ------------------------------------------------------------------------------
# Developers:
#
# Erick Jorge Canales-Rodríguez (EPFL, CHUV, Lausanne, Switzerland; FIDMAG Research Foundation, CIBERSAM, Barcelona, Spain)
# Marco Pizzolato               (EPFL)
# Gian Franco Piredda           (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
# Tom Hilbert                   (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
# Tobias Kober                  (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
# Jean-Philippe Thiran          (EPFL, UNIL, CHUV, Switzerland)
# Alessandro Daducci            (Computer Science Department, University of Verona, Italy)
# Date: 2020
#===============================================================================

from __future__ import division
import os
import numpy as np
import math
import numba as nb

# ------------------------------------------------------------------------------
# Functions to generate the Dictionary of multi-echo T2 signals using the exponential model

def create_met2_design_matrix(m_TEs, m_T2s):
    '''
    Creates the Multi Echo T2 (spectrum) design matrix.
    Given a grid of echo times (numpy vector TEs) and a grid of T2 times
    (numpy vector T2s), it returns the deign matrix to perform the inversion.
    '''
    M = len(m_TEs)
    N = len(m_T2s)
    design_matrix = np.zeros((M,N))
    for row in range(M):
        for col in range(N):
            exponent = -(m_TEs[row] / m_T2s[col])
            design_matrix[row,col] = np.exp(exponent)
        # end for col
    # end for row
    return design_matrix
#end fun

# Functions to generate the Dictionary of multi-echo T2 signals using the EPG model
def create_met2_design_matrix_epg(m_Npc, m_T2s, m_T1s, m_nEchoes, m_Tau, m_FlipAngle, m_TR):
    '''
    Creates the Multi Echo T2 (spectrum) design matrix.
    Given a grid of echo times (numpy vector TEs) and a grid of T2 times
    (numpy vector T2s), it returns the deign matrix to perform the inversion.
    *** Here we use the epg model to simulate signal artifacts
    '''
    design_matrix = np.zeros((m_nEchoes, m_Npc))
    rad           = np.pi/180.0  # constant to convert degrees to radians
    for cols in range(m_Npc):
        signal = (1.0 - np.exp(-m_TR/m_T1s[cols])) * epg_signal(m_nEchoes, m_Tau, np.array([1.0/m_T1s[cols]]), np.array([1.0/m_T2s[cols]]), m_FlipAngle * rad, m_FlipAngle/2.0 * rad)
        #signal = (1.0 - np.exp(-m_TR/m_T1s[cols])) * epg_signal(m_nEchoes, m_Tau, np.array([1.0/m_T1s[cols]]), np.array([1.0/m_T2s[cols]]), m_FlipAngle * rad, 90.0 * rad)
        design_matrix[:, cols] = signal.flatten()
        # end for row
    return design_matrix
#end fun

def epg_signal(m_n, m_Tau, m_R1Vec, m_R2Vec, m_Alpha, m_AlphaExc):
    '''Function that do this

    Parameters:

    
    '''
    nRates = m_R2Vec.shape[0]
    m_Tau = m_Tau/2.0

    # defining signal matrix
    H = np.zeros((m_n, nRates))

    # RF mixing matrix
    T = fill_T(m_n, m_Alpha)

    # Selection matrix to move all traverse states up one coherence level
    S = fill_S(m_n)

    for iRate in range(nRates):
        # Relaxation matrix
        R2 = m_R2Vec[iRate]
        R1 = m_R1Vec[iRate]

        R0      = np.zeros((3,3))
        R0[0,0] = np.exp(-m_Tau*R2)
        R0[1,1] = np.exp(-m_Tau*R2)
        R0[2,2] = np.exp(-m_Tau*R1)

        R = fill_R(m_n, m_Tau, R0, R2)
        # Precession and relaxation matrix
        P = np.dot(R,S)
        # Matrix representing the inter-echo duration
        E = np.dot(np.dot(P,T),P)
        H = fill_H(R, m_n, E, H, iRate, m_AlphaExc)
        # end
    return H
#end fun

def fill_S(m_n):
    '''Function that do this

    Parameters:

    
    '''
    the_size = 3*m_n + 1
    S = np.zeros((the_size,the_size))
    S[0,2]=1.0
    S[1,0]=1.0
    S[2,5]=1.0
    S[3,3]=1.0
    for o in range(2,m_n+1):
        offset1=( (o-1) - 1)*3 + 2
        offset2=( (o+1) - 1)*3 + 3
        if offset1<=(3*m_n+1):
            S[3*o-2,offset1-1] = 1.0  # F_k <- F_{k-1}
        # end
        if offset2<=(3*m_n+1):
            S[3*o-1,offset2-1] = 1.0  # F_-k <- F_{-k-1}
        # end
        S[3*o,3*o] = 1.0              # Z_order
    # end for
    return S
#end fun

def fill_T(m_n, m_Alpha):
    '''Function that do this

    Parameters:

    
    '''
    T0      = np.zeros((3,3))
    T0[0,:] = [math.cos(m_Alpha/2.0)**2, math.sin(m_Alpha/2.0)**2,  math.sin(m_Alpha)]
    T0[1,:] = [math.sin(m_Alpha/2.0)**2, math.cos(m_Alpha/2.0)**2, -math.sin(m_Alpha)]
    T0[2,:] = [-0.5*math.sin(m_Alpha),   0.5*math.sin(m_Alpha),     math.cos(m_Alpha)]

    T = np.zeros((3*m_n + 1, 3*m_n + 1))
    T[0,0] = 1.0
    T[1:3+1, 1:3+1] = T0
    for itn in range(m_n-1):
        T[(itn+1)*3+1:(itn+2)*3+1,(itn+1)*3+1:(itn+2)*3+1] = T0
    # end
    return T
#end fun

def fill_R(m_n, m_Tau, m_R0, m_R2):
    '''Function that do this

    Parameters:

    
    '''
    R  = np.zeros((3*m_n + 1, 3*m_n + 1))
    R[0,0] = np.exp(-m_Tau*m_R2)
    R[1:3+1, 1:3+1] = m_R0
    for itn in range(m_n-1):
        R[(itn+1)*3+1:(itn+2)*3+1,(itn+1)*3+1:(itn+2)*3+1] = m_R0
    # end
    return R
#end fun

def fill_H(m_R, m_n, m_E, m_H, m_iRate, m_AlphaExc):
    '''Function that do this

    Parameters:

    
    '''
    x    = np.zeros((m_R.shape[0],1))
    x[0] = math.sin(m_AlphaExc)
    x[1] = 0.0
    x[2] = math.cos(m_AlphaExc)
    for iEcho in range(m_n):
        x = np.dot(m_E,x)
        m_H[iEcho, m_iRate] = x[0]
    #end for IEcho
    return m_H
#end fun

def create_Dic_3D(m_Npc, m_T2s, m_T1s, m_nEchoes, m_Tau, m_AlphaValues, m_TR):
    '''Function that do this

    Parameters:

    
    '''
    dim3   = len(m_AlphaValues)
    Dic_3D = np.zeros((m_nEchoes, m_Npc, dim3))
    for iter in range(dim3):
        Dic_3D[:,:,iter] = create_met2_design_matrix_epg(m_Npc, m_T2s, m_T1s, m_nEchoes, m_Tau, m_AlphaValues[iter], m_TR)
    #end for
    return Dic_3D
#end fun

def main_Dic3D(m_FaMethod,m_Npc,m_T2s,m_T1s,m_nEchoes,m_Tau,m_TR):
    '''Main function to execute the creation of data dict depending of the FA choose

    Parameters:

    
    '''
    match m_FaMethod:

        case 'spline':
            N_alphas     = 91*3 # (steps = 0.333 degrees, from 90 to 180)
            #N_alphas     = 91*2 # (steps = 0.5 degrees, from 90 to 180)
            #N_alphas     = 91 # (steps = 1.0 degrees, from 90 to 180)
            AlphaValues = np.linspace(90.0,  180.0,  N_alphas)
            dic3D       = create_Dic_3D(m_Npc, m_T2s, m_T1s, m_nEchoes, m_Tau, AlphaValues, m_TR)
            #m_AlphaValuesSpline = np.round( np.linspace(90.0, 180.0, 8) )
            AlphaValuesSpline = np.linspace(90.0, 180.0, 15)
            dic3DLR    = create_Dic_3D(m_Npc, m_T2s, m_T1s, m_nEchoes, m_Tau, AlphaValuesSpline, m_TR)
            return dic3D, dic3DLR, AlphaValues,AlphaValuesSpline
        
        case 'brute-force':
            dic3DLR = None
            N_alphas     = 91 # (steps = 1.0 degrees, from 90 to 180)
            AlphaValues = np.linspace(90.0,  180.0,  N_alphas)
            dic3D       = create_Dic_3D(m_Npc, m_T2s, m_T1s, m_nEchoes, m_Tau, AlphaValues, m_TR)
            return dic3D,dic3DLR, AlphaValues,None