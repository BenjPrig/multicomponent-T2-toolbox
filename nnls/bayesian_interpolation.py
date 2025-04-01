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
# Date: 11/02/2019
#===============================================================================

from __future__ import division

import scipy
from   scipy.optimize import minimize_scalar, fminbound, minimize
from   scipy.special import erf
from   scipy.linalg import cholesky, det, inv


import numpy as np
import numba as nb

import inspect
import sys
import os
#sys.path.insert(1, os.path.dirname(inspect.getfile(scipy.optimize)))
#import _nnls
from scipy.optimize import _nnls,__nnls

#===============================================================================
#                                FUNCTIONS
#===============================================================================

# Standard NNLS python in scipy.
# The default number of iterations was increased from 3n to 5n to improve
# the estimation of smooth solutions.
# The "too many iterations" error was removed.

def nnls(m_A, m_b):
    '''Function that do this

    Parameters:

    
    '''
    m_A, m_b = map(np.asarray_chkfinite, (m_A, m_b))

    #if len(m_A.shape) != 2:
    #    raise ValueError("expected matrix")
    #if len(m_b.shape) != 1:
    #    raise ValueError("expected vector")

    m, n = m_A.shape

    #if m != m_b.shape[0]:
    #    raise ValueError("incompatible dimensions")

    #maxiter = -1 if maxiter is None else int(maxiter)
    maxiter = -1
    #maxiter = int(5*n)

    w     = np.zeros((n,), dtype=np.double)
    zz    = np.zeros((m,), dtype=np.double)
    index = np.zeros((n,), dtype=int)

    #x, rnorm, mode = _nnls.nnls(m_A, m, n, m_b, w, zz, index, maxiter)
    x, rnorm, mode = __nnls.nnls(m_A, m, n, m_b, w, zz, index, maxiter)

    #if mode != 1:
    #    raise RuntimeError("too many iterations")
    return x, rnorm
#end

# ------------------------------------------------------------------------------
#                           BayesReg
#              THIS IS THE FUNCTION USED IN THE PAPER
# ------------------------------------------------------------------------------
# For more details see:
# Eq 2.21 (Bayesian Interpolation - Mackay): -log(P): http://www.inference.org.uk/mackay/thesis.pdf
# Alternatively, see section 4.2: https://authors.library.caltech.edu/13792/1/MACnc92a.pdf
# This code can be accelerated by using a grid of alpha-values and by precomputing the related cholesky decompositions and determinants.
# ------------------------------------------------------------------------------
def BayesReg_nnls(m_Dic_i, m_M, m_L):
    '''Function that do this

    Parameters:

    
    '''
    m,n         = m_Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((m_M, Zerosm))
    # ---------------------
    # Estimate beta = 1/sigma**2
    x0, kk          = nnls(m_Dic_i, m_M)
    num_non_zeros   = np.sum(x0 > 0)
    degress_of_fred = np.max([m - num_non_zeros, 1.0]) # avoid negative values by error
    sigma           = np.sqrt( np.sum( (m_M - np.dot(m_Dic_i, x0))**2 ) / degress_of_fred )
    beta            = 1./sigma**2
    # ---------------------
    # New definition, here I wrote alpha as a function of beta: alpha = beta*k, where k=x[1]
    # This definition allows to simplify a bit the evaluation
    B           = np.matmul(m_Dic_i.T, m_Dic_i)
    K           = np.matmul(m_L.T, m_L)
    det_L       = det(m_L)
    reg_sol     = fminbound(obj_BayesReg_nnls, 1e-8, 2.0, args=(m_Dic_i, m_L, M_aug, m_M, m, n, B, det_L, beta, K), xtol=1e-05, maxfun=200, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((m_Dic_i, np.sqrt(reg_sol) * m_L)), M_aug )
    return f, reg_sol
#end fun

def obj_BayesReg_nnls(m_x, m_D, m_L, m_SignalAug, m_Signal, m_m, m_n, m_B, m_DetL, m_Beta, m_K):
    '''Function that do this

    Parameters:

    
    '''
    Daux        = np.concatenate((m_D, np.sqrt(m_x) * m_L))
    f, kk       = nnls( Daux, m_SignalAug )
    ED          = 0.5 * np.sum( (np.dot(m_D, f) - m_Signal)**2 )
    EW          = 0.5 * np.sum ( np.dot(m_L, f)**2 )
    A           = m_Beta*m_B + (m_Beta*m_x)*m_K
    # -----------------------
    # A=U.T*U
    U           = cholesky(A, lower=False, overwrite_a=True, check_finite=False) # faster evaluation with these input options
    #det_U       = det(U)
    det_U       = np.prod(np.diag(U)) # faster evaluation of the determinant

    error_term1  = 1.0 + erf( ( 1./np.sqrt(2.) ) * np.dot(U, f)  )
    series_prod1 = np.sum( np.log( error_term1 ) )

    cost_fun1   = m_Beta*ED + m_Beta*m_x*EW + np.log(det_U) - (m_n/2.) * np.log(np.pi/2.) - series_prod1
    cost_fun2   = (m_m/2.) * np.log(2.*np.pi) - (m_m/2.) * np.log(m_Beta) + (m_n/2.) * np.log(np.pi) - (m_n/2.) * np.log(2*m_Beta*m_x)  - np.log(m_DetL)
    cost_fun    = cost_fun1 + cost_fun2
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
#   *** OTHER PREVIOUS IMPLEMENTATIONS AND VARIANTS TO TEST IN THE FUTURE ***
# ------------------------------------------------------------------------------
# Original equations by Mackay: Bayesian Interpolation
# This method uses the original equations but replacing the regularized LS solution by the regularized NNLS
def compute_f_alpha_RNNLS_I_evidencelog(m_Dic_i, m_M):
    '''Function that do this

    Parameters:

    
    '''
    m,n         = m_Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((m_M, Zerosm))
    In          = np.eye(n)
    # ----------------------
    # SNR = m_M[0]/sigma, where m_M[0]=1 after normalization
    # SNR_min = 50
    # SNR_max = 1000

    # beta = 1/sigma**2
    inv_sigma_min = 20.0
    inv_sigma_max = 1000.0
    inv_sigma0    = 200.0

    beta_min      = inv_sigma_min**2
    beta_max      = inv_sigma_max**2
    beta0         = inv_sigma0**2
    # ----------------------
    x0          = [beta0, 0.1*beta0] # initial estimate
    bnds        = ((beta_min, beta_max),(1e-5*beta_min, 10.0*beta_max)) # bounds
    B           = np.matmul(m_Dic_i.T, m_Dic_i)
    res         = minimize(NNLSreg_obj_evidencelog, x0, method = 'L-BFGS-B', options={'gtol': 1e-20, 'disp': False, 'maxiter': 300}, bounds = bnds, args=(m_Dic_i, In, M_aug, m_M, m, n, B))
    reg_sol     = res.x
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((m_Dic_i, np.sqrt(reg_sol[1]/reg_sol[0]) * In)), M_aug )
    return f, reg_sol[0], reg_sol[1]
#end fun

def NNLSreg_obj_evidencelog(m_x, m_D, In, m_SignalAug, m_Signal, m_m, m_n, m_B):
    '''Function that do this

    Parameters:

    
    '''
    Daux       = np.concatenate((m_D, np.sqrt(m_x[1]/m_x[0]) * In))
    f, kk      = nnls( Daux, m_SignalAug )
    ED         = 0.5 * np.sum( (np.dot(m_D, f) - m_Signal)**2 )
    EW         = 0.5 * np.sum ( f**2 )
    ratio_dem  = np.linalg.det( m_x[0] * m_B + m_x[1] * In)
    # ------------------------
    cost_fun   = (m_x[0] * ED + m_x[1] * EW) + 0.5 * np.log(ratio_dem) - (m_m/2.) * np.log(m_x[0]) - (m_n/2.) *np.log(m_x[1]) + (m_m/2.) * np.log(2.*np.pi)
    # Eq 2.21 (Bayesian Interpolation - Mackay): -log(P): http://www.inference.org.uk/mackay/thesis.pdf
    # Alternatively, see section 4.2: https://authors.library.caltech.edu/13792/1/MACnc92a.pdf
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
# Modified equations by Mackay: Bayesian Interpolation.
# This method uses the original equations but replacing the regularized LS solution by the regularized NNLS
# Moreover, here we assume we know beta
def compute_f_alpha_RNNLS_I_evidencelog_fast(m_Dic_i, m_M):
    '''Function that do this

    Parameters:

    
    '''
    m,n         = m_Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((m_M, Zerosm))
    In          = np.eye(n)
    # ---------------------
    # Estimate beta = 1/sigma**2
    x0, kk      = nnls(m_Dic_i, m_M)
    sigma       = np.sqrt( np.sum( (m_M - np.dot(m_Dic_i, x0))**2 ) / (m - 1.0) )
    beta        = 1./sigma**2
    # ---------------------
    # New definition, here I wrote alpha as a function of beta: alpha = beta*x
    # This definition allows to simplify a bit the evaluation
    B           = np.matmul(m_Dic_i.T, m_Dic_i)
    reg_sol     = fminbound(NNLSreg_obj_evidencelog_fast, 1e-8, 10.0, args=(m_Dic_i, In, M_aug, m_M, m, n, B, beta), xtol=1e-05, maxfun=200, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((m_Dic_i, np.sqrt(reg_sol) * In)), M_aug )
    return f, reg_sol
#end fun

def NNLSreg_obj_evidencelog_fast(m_x, m_D, In, m_SignalAug, m_Signal, m_m, m_n, m_B, m_Beta):
    '''Function that do this

    Parameters:

    
    '''
    Daux       = np.concatenate((m_D, np.sqrt(m_x) * In))
    f, kk      = nnls( Daux, m_SignalAug )
    ED         = 0.5 * np.sum( (np.dot(m_D, f) - m_Signal)**2 )
    #EW         = 0.5 * np.sum ( np.dot(In, f)**2 )
    EW         = 0.5 * np.sum ( f**2 )
    ratio_dem  = np.linalg.det( m_B + m_x*In )
    # ------------------------
    cost_fun   = m_Beta * (ED + m_x*EW) + 0.5 * np.log(ratio_dem)  - (m_m/2.) * np.log(m_Beta) - (m_n/2.) * np.log(m_x) + (m_m/2.) * np.log(2.*np.pi)
    # Eq 2.21 (Bayesian Interpolation - Mackay): -log(P): http://www.inference.org.uk/mackay/thesis.pdf
    # Alternatively, see section 4.2: https://authors.library.caltech.edu/13792/1/MACnc92a.pdf
    return cost_fun
# end fun


# ------------------------------------------------------------------------------
# Modified equations by Mackay: Bayesian Interpolation
# This method uses the original equations but replacing the regularized LS solution by the regularized NNLS
# Here we assume we know beta and we are using an arbitrary matrix L, instead of I
def compute_f_alpha_RNNLS_I_evidencelog_mod(m_Dic_i, m_M, m_L):
    '''Function that do this

    Parameters:

    
    '''
    m,n         = m_Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((m_M, Zerosm))
    # ---------------------
    # Estimate beta = 1/sigma**2
    x0, kk          = nnls(m_Dic_i, m_M)
    num_non_zeros   = np.sum(x0 > 0)
    degress_of_fred = np.max([m - num_non_zeros, 1.0]) # avoid negative values by error
    sigma           = np.sqrt( np.sum( (m_M - np.dot(m_Dic_i, x0))**2 ) / degress_of_fred )
    beta            = 1./sigma**2
    # ---------------------
    # New definition, here I wrote alpha as a function of beta: alpha = beta*k, where k=x[1]
    # This definition allows to simplify a bit the evaluation
    B           = np.matmul(m_Dic_i.T, m_Dic_i)
    K           = np.matmul(m_L.T, m_L)
    #det_Linv    = np.linalg.det(np.linalg.inv(m_L))
    det_Linv    = np.linalg.det(np.linalg.inv(K))
    reg_sol     = fminbound(NNLSreg_obj_evidencelog_mod, 1e-8, 10.0, args=(m_Dic_i, m_L, M_aug, m_M, m, n, B, det_Linv, beta, K), xtol=1e-05, maxfun=200, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((m_Dic_i, np.sqrt(reg_sol) * m_L)), M_aug )
    return f, reg_sol
#end fun

def NNLSreg_obj_evidencelog_mod(m_x, m_D, m_L, m_SignalAug, m_Signal, m_m, m_n, m_B, det_Linv, m_Beta, m_K):
    '''Function that do this

    Parameters:

    
    '''
    Daux       = np.concatenate((m_D, np.sqrt(m_x) * m_L))
    f, kk      = nnls( Daux, m_SignalAug )
    ED         = 0.5 * np.sum( (np.dot(m_D, f) - m_Signal)**2 )
    EW         = 0.5 * np.sum ( np.dot(m_L, f)**2 )
    ratio_dem  = np.linalg.det( m_B + m_x*m_K )
    # ------------------------
    cost_fun   = m_Beta * (ED + m_x*EW) + 0.5 * ( np.log(ratio_dem) + np.log(det_Linv) ) - (m_m/2.) * np.log(m_Beta) - (m_n/2.) * np.log(m_x) + (m_m/2.) * np.log(2.*np.pi)

    # Eq 2.21 (Bayesian Interpolation - Mackay): -log(P): http://www.inference.org.uk/mackay/thesis.pdf
    # Alternatively, see section 4.2: https://authors.library.caltech.edu/13792/1/MACnc92a.pdf
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
# Here we assume we know beta and we are using an arbitrary matrix L, instead of I
# Moreover, we did some approximations to consider that f >= 0
# For more details see:
# Eq 2.21 (Bayesian Interpolation - Mackay): -log(P): http://www.inference.org.uk/mackay/thesis.pdf
# Alternatively, see section 4.2: https://authors.library.caltech.edu/13792/1/MACnc92a.pdf
# IT IS FASTER THAN THE ORIGINAL VERSION BUT THE PERFORMANCE IS WORSE
def compute_f_alpha_RNNLS_L_evidencelog_nn_fast(m_Dic_i, m_M, m_L):
    '''Function that do this

    Parameters:

    
    '''
    m,n         = m_Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((m_M, Zerosm))
    # ---------------------
    # Estimate beta = 1/sigma**2
    x0, kk          = nnls(m_Dic_i, m_M)
    num_non_zeros   = np.sum(x0 > 0)
    degress_of_fred = np.max([m - num_non_zeros, 1.0]) # avoid negative values by error
    sigma           = np.sqrt( np.sum( (m_M - np.dot(m_Dic_i, x0))**2 ) / degress_of_fred )
    beta            = 1./sigma**2
    # ---------------------
    # New definition, here I wrote alpha as a function of beta: alpha = beta*k, where k=x[1]
    # This definition allows to simplify a bit the evaluation
    B           = np.matmul(m_Dic_i.T, m_Dic_i)
    K           = np.matmul(m_L.T, m_L)
    det_L       = det(m_L)
    reg_0       = 1e-5
    UH          = cholesky(B + reg_0*K, lower=False, overwrite_a=True, check_finite=False) # faster evaluation with these input options
    diag_UH     = np.diag(UH)
    reg_sol     = fminbound(NNLSreg_obj_evidencelog_nn_fast, 1e-8, 2.0, args=(m_Dic_i, m_L, M_aug, m_M, m, n, B, det_L, beta, K, UH, diag_UH, reg_0), xtol=1e-05, maxfun=100, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((m_Dic_i, np.sqrt(reg_sol) * m_L)), M_aug )
    return f, reg_sol
#end fun

def NNLSreg_obj_evidencelog_nn_fast(m_x, m_D, m_L, m_SignalAug, m_Signal, m_m, m_n, m_B, m_DetL, m_Beta, m_K, m_UH, m_DiagUH, m_reg0):
    '''Function that do this

    Parameters:

    
    '''
    Daux        = np.concatenate((m_D, np.sqrt(m_x) * m_L))
    f, kk       = nnls( Daux, m_SignalAug )
    ED          = 0.5 * np.sum( (np.dot(m_D, f) - m_Signal)**2 )
    EW          = 0.5 * np.sum ( np.dot(m_L, f)**2 )
    # -----------------------
    #A           = m_Beta*m_B + (m_Beta*m_x)*m_K
    # Cholesky decomposition: A=U.T*U
    #U           = cholesky(A, lower=False, overwrite_a=True, check_finite=False) # faster evaluation with these input options
    diag_U  = np.sqrt(m_DiagUH**2 + (m_x-m_reg0)*np.diag(m_K)) - m_DiagUH
    U_mod   = m_UH + np.diag(diag_U)
    U       = np.sqrt(m_Beta)*U_mod
    #det_U       = det(U)
    det_U       = np.prod(np.diag(U)) # faster evaluation of the determinant

    error_term1  = 1.0 + erf( ( 1./np.sqrt(2.) ) * np.dot(U, f)  )
    series_prod1 = np.sum( np.log( error_term1 ) )

    cost_fun1   = m_Beta*ED + m_Beta*m_x*EW + np.log(det_U) - (m_n/2.) * np.log(np.pi/2.) - series_prod1
    cost_fun2   = (m_m/2.) * np.log(2.*np.pi) - (m_m/2.) * np.log(m_Beta) + (m_n/2.) * np.log(np.pi) - (m_n/2.) * np.log(2*m_Beta*m_x)  - np.log(m_DetL)
    cost_fun    = cost_fun1 + cost_fun2
    return cost_fun
# end fun

# The same that in the previous function, but also considering the measured signal is non-negative
def compute_f_alpha_RNNLS_L_evidencelog_nn_nnLikelihood(m_Dic_i, m_M, m_L):
    '''Function that do this

    Parameters:

    
    '''
    m,n         = m_Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((m_M, Zerosm))
    # ---------------------
    # Estimate beta = 1/sigma**2
    x0, kk          = nnls(m_Dic_i, m_M)
    num_non_zeros   = np.sum(x0 > 0)
    degress_of_fred = np.max([m - num_non_zeros, 1.0]) # avoid negative values by error
    sigma           = np.sqrt( np.sum( (m_M - np.dot(m_Dic_i, x0))**2 ) / degress_of_fred )
    beta            = 1./sigma**2
    # ---------------------
    # New definition, here I wrote alpha as a function of beta: alpha = beta*k, where k=x[1]
    # This definition allows to simplify a bit the evaluation
    B           = np.matmul(m_Dic_i.T, m_Dic_i)
    K           = np.matmul(m_L.T, m_L)
    det_L       = det(m_L)
    reg_sol     = fminbound(NNLSreg_obj_evidencelog_nnLikelihood, 1e-8, 2.0, args=(m_Dic_i, m_L, M_aug, m_M, m, n, B, det_L, beta, K), xtol=1e-05, maxfun=100, full_output=0, disp=0)
    # --------- Estimation
    f, rnorm_f  = nnls( np.concatenate((m_Dic_i, np.sqrt(reg_sol) * m_L)), M_aug )
    return f, reg_sol
#end fun

def NNLSreg_obj_evidencelog_nnLikelihood(m_x, m_D, m_L, m_SignalAug, m_Signal, m_m, m_n, m_B, m_DetL, m_Beta, m_K):
    '''Function that do this

    Parameters:

    
    '''
    Daux        = np.concatenate((m_D, np.sqrt(m_x) * m_L))
    f, kk       = nnls( Daux, m_SignalAug )
    ED          = 0.5 * np.sum( (np.dot(m_D, f) - m_Signal)**2 )
    EW          = 0.5 * np.sum ( np.dot(m_L, f)**2 )
    A           = m_Beta*m_B + (m_Beta*m_x)*m_K
    #det_A       = np.linalg.det(A)
    # -----------------------
    U           = cholesky(A, lower=False) # A=U.T*U
    det_U       = det(U)

    error_term1  = 1.0 + erf( ( 1./np.sqrt(2) ) * np.dot(U, f)  )
    series_prod1 = np.sum( np.log( error_term1 ) )

    error_term2  = 1.0 + erf( np.dot(m_D, f) * np.sqrt(m_Beta/2.) )
    series_prod2 = np.sum( np.log( error_term2 ) )

    cost_fun1   = m_Beta*ED + m_Beta*m_x*EW + np.log(det_U) - (m_n/2.) * np.log(np.pi/2.) - series_prod1
    cost_fun2   = (m_m/2.) * np.log(np.pi/2.) - (m_m/2.) * np.log(m_Beta) + series_prod2 + (m_n/2.) * np.log(np.pi) - (m_n/2.) * np.log(2*m_Beta*m_x)  - np.log(m_DetL)
    cost_fun    = cost_fun1 + cost_fun2
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
def compute_f_alpha_RNNLS_I_evidencelog_fullbayes(m_Dic3D, m_M):
    '''Function that do this

    Parameters:

    
    '''
    m,n,p       = m_Dic3D.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((m_M, Zerosm))
    In          = np.eye(n)
    # ----------------------
    SNR_min     = 10.0
    SNR_max     = 100.0
    SNR0        = 40.0
    beta_min    = SNR_min**2
    beta_max    = SNR_max**2
    beta0       = SNR0**2
    # ----------------------
    x0          = [beta0, 100.0] # initial estimate
    bnds        = ((beta_min, beta_max),(1e-5*beta_min, 2e3*beta_max)) # bounds
    # ----------------------
    x_sol       = np.zeros((n,p))
    beta_sol    = np.zeros(p)
    alpha_sol   = np.zeros(p)
    Mprop       = np.zeros(p)
    for iter in range(p):
        #print iter + 1
        Dic_i   = m_Dic3D[:,:,iter]
        B       = np.matmul(Dic_i.T, Dic_i)
        # --------- Regularization parameters
        #res     = minimize(NNLSreg_obj_evidencelog, x0, method = 'L-BFGS-B', options={'gtol': 1e-20, 'disp': False, 'maxiter': 500}, bounds = bnds, args=(Dic_i, In, M_aug, m_M, m, n, B))
        res     = minimize(NNLSreg_obj_evidencelog, x0, method = 'L-BFGS-B', options={'disp': False}, bounds = bnds, args=(Dic_i, In, M_aug, m_M, m, n, B))
        reg_sol = res.x
        # --------- Estimation
        f, rnorm_f  = nnls( np.concatenate((Dic_i, np.sqrt(reg_sol[1]/reg_sol[0]) * In)), M_aug )
        x_sol[:,iter]   = f
        beta_sol[iter]  = reg_sol[0]
        alpha_sol[iter] = reg_sol[1]
        Mprop[iter]     = Prob_model(reg_sol, f, Dic_i, m_M, In, B, m, n)
    #end for
    Mprop     = Mprop/np.sum(Mprop) # normalization
    return Mprop, x_sol, beta_sol, alpha_sol
#end

def fullbayes_max(m_Mprop, m_Xsol, m_BetaSol, m_AlphaSol):
    '''Function that do this

    Parameters:

    
    '''
    # Model comparison: select the best model
    ind_max   = np.argmax(m_Mprop)
    f_opt     = m_Xsol[:,ind_max]
    beta_opt  = m_BetaSol[ind_max]
    alpha_opt = m_AlphaSol[ind_max]
    return f_opt, beta_opt, alpha_opt, ind_max
#end

def fullbayes_BMA(m_Mprop, m_Xsol, m_BetaSol, m_AlphaSol, m_FaAngles):
    '''Function that do this

    Parameters:

    
    '''
    # Bayesian model averaging
    n,p       = m_Xsol.shape
    f_opt     = 0.0
    beta_opt  = 0.0
    alpha_opt = 0.0
    FA_opt    = 0.0
    for iter in range(p):
        f_opt     = f_opt     + m_Mprop[iter] * m_Xsol[:,iter]
        beta_opt  = beta_opt  + m_Mprop[iter] * m_BetaSol[iter]
        alpha_opt = alpha_opt + m_Mprop[iter] * m_AlphaSol[iter]
        FA_opt    = FA_opt    + m_Mprop[iter] * m_FaAngles[iter]
    #end for
    return f_opt, beta_opt, alpha_opt, FA_opt
#end fun

def Prob_model(m_x, m_f, m_D, m_Signal, In, m_B, m_m, m_n):
    '''Function that do this

    Parameters:

    
    '''
    ED          = 0.5 * np.sum( (np.dot(m_D, m_f) - m_Signal)**2 )
    EW          = 0.5 * np.sum ( m_f**2 )
    (sign, logdet) = np.linalg.slogdet(m_x[1]*In + m_x[0]*m_B)
    log_prob    = m_x[0]*ED + m_x[1]*EW + 0.5*(sign*logdet) - (m_n/2.0)*np.log(m_x[1]) - (m_m/2.0)*np.log(m_x[0]) - (m_m/2.0)*np.log(2*np.pi)
    gamma_opt   = 2.0 * m_x[1] * EW
    prob_model  = np.exp(-1.0*log_prob) * np.sqrt(2./gamma_opt) * np.sqrt(2./(m_m - gamma_opt))
    return prob_model
# end fun
