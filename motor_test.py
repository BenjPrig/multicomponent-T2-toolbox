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
# Alessandro Daducci            (Computer Science Department, University of Verona, Italy)
# Tobias Kober                  (CHUV, EPFL, Advanced Clinical Imaging Technology, Siemens Healthcare AG, Switzerland)
# Jean-Philippe Thiran          (EPFL, UNIL, CHUV, Switzerland)

# Date: 11/02/2019
#===============================================================================

from __future__ import division

import sys
import numpy as np

import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex']=True

import multiprocessing
import warnings

import denoise_signal as dnois
import compute_fa_signal.estimate_spectrum_metrics as est
import initiate_Values as value
import compute_fa_signal.fa_estimation as fa
import matrix_dictionnaries.epg as epg
import matrix_dictionnaries.create_LaplacianMatrix as laplac
import plot.plot_results as plot
import plot.plot_spectrum as spectrum
import save_load_img as sl

sys.path.append("..")
#===============================================================================
class obj5:
    def __init__(self, value):
        self.float = value
    def __repr__(self):
        return "%.5f" %(self.float)
    #end
#end

class obj1:
    def __init__(self, value):
        self.float = value
    def __repr__(self):
        return "%.1f" %(self.float)
    #end
#end

class obj0:
    def __init__(self, value):
        self.float = value
    def __repr__(self):
        return "%.0f" %(self.float)
    #end
#end

warnings.filterwarnings("ignore",category=FutureWarning)

#_______________________________________________________________________________
def motor_recon_met2(m_TEarray, m_PathToData, m_PathToMask, m_PathToSaveData, 
                     m_RegMethod, m_RegMatrix, m_Denoise, m_TR,
                     m_FAMethod, m_FASmooth, m_MyelinT2, m_NumCores):
    # Load Data and Mask
    data = sl.load_niftiFloatImage(m_PathToData)
    mask = sl.load_niftiIntImage(m_PathToMask)

    nx,ny,nz,nt = sl.get_shape(data,'Data')
    sl.get_shape(mask,'Mask')

    match nz:
        case 1: # 2D version
            mask = np.squeeze(mask,-1)
            for c in range(nt):
                data[:,:,:,c] = data[:,:,:,c] * mask
        case _: # 3D version
            for c in range(nt):
                data[:,:,:,c] = np.squeeze(data[:,:,:,c]) * mask

    sl.save_nibDatasets(data,m_PathToSaveData,'dataMasked')
    nEchoes   = m_TEarray.shape[0]
    tau       = m_TEarray[1] - m_TEarray[0]

    fM        = np.zeros((nx, ny, nz))
    fIE       = fM.copy()
    fnT       = fM.copy()
    fCSF      = fM.copy()
    T2m       = fM.copy()
    T2IE      = fM.copy()
    T2nT      = fM.copy()
    Ktotal    = fM.copy()
    FA        = fM.copy()
    FA_index  = fM.copy()
    reg_param = fM.copy()
    NITERS    = fM.copy()

    # ==============================================================================
    # Inital values for the dictionary
    
    Npc = value.choose_numberCompartment(m_RegMethod)
    T2s, ind_m, ind_t, ind_csf, T1s = value.intitiate_t2values(m_MyelinT2,Npc)

    # Create multi-dimensional dictionary with multiples flip_angles

    Dic_3D, Dic_3D_LR, alphaValues, alphaValuesSpine = epg.main_Dic3D(m_FAMethod,Npc,
                                                                      T2s,T1s,
                                                                      nEchoes,tau,m_TR)

    # Define regularization vectors for the L-curve method
    num_l_laplac   = 50
    lambda_reg     = np.zeros((num_l_laplac))
    # lambda_reg[1:] = np.logspace(math.log10(1e-8), math.log10(100.0), num=num_l_laplac-1, endpoint=True, base=10.0)
    lambda_reg[1:] = np.logspace(math.log10(1e-8), math.log10(10.0), num=num_l_laplac-1, 
                                 endpoint=True, base=10.0)

    # --------------------------------------------------------------------------
    laplace_matrix = laplac.create_RegMatrix(Npc,T2s,m_RegMatrix)

    data[data<0.0]  = 0.0 # correct artifacts
    number_of_cores = multiprocessing.cpu_count()
    if m_NumCores == -1:
        m_NumCores = number_of_cores
        print('Using all CPUs: ', number_of_cores)
    else:
        print('Using ', m_NumCores, ' CPUs from ', number_of_cores)
    #end if

    #_______________________________________________________________________________
    #_______________________________ ESTIMATION ____________________________________
    #_______________________________________________________________________________

    data = dnois.execute_mainDenoise(nx,ny,nz,nt,data,mask,m_Denoise,m_PathToSaveData)
    # sl.save_nibDatasets(data,r'/mnt/c/data/','testdata')

    print('Step #2: Estimation of flip angles:')
    
    mean_T2_dist, FA_index,data_1D = fa.compute_meanT2dist(nx, ny, nz, nt, data,mask,
                                                   m_FASmooth,FA,m_FAMethod,FA_index,
                                                   Ktotal, Dic_3D, Dic_3D_LR,
                                                   alphaValues,alphaValuesSpine, m_NumCores )

    # TO DO: (1) Estimate also the standard deviation of the spectrum and plot it
    #        (2) Estimate a different mean spectrum for each tissue type (using a segmentation from a T1, or any strategy to segment the raw MET2 data)
    

    dist_T2_mean1, dist_T2_mean2 = fa.compute_signalKernel(nx,ny,nz,Npc,FA_index,Dic_3D,data,mask,m_PathToSaveData)
    
    # # Save mean_T2_dist, which is the initial value for RUMBA
    spectrum.plot_mean_spectrum(T2s,mean_T2_dist,dist_T2_mean1,dist_T2_mean2,m_PathToSaveData)
    # # --------------------------------------------------------------------------

    print('Step #3: Estimation of T2 spectra:')

    # create 4D images
    f_sol_4D = np.zeros((nx, ny, nz, T2s.shape[0]))
    s_sol_4D = np.zeros((nx, ny, nz, nEchoes))
    f_sol_4D, s_sol_4D, reg_param = est.estimate_t2_spectra(nx, ny, nz, T2s, 
                                                            data, mask, FA_index,
                                                            f_sol_4D, s_sol_4D,reg_param,
                                                            Dic_3D, lambda_reg, nEchoes,
                                                            m_RegMethod, laplace_matrix,
                                                            m_NumCores )

    print('Step #4: Estimation of quantitative metrics')
    est.estimate_quantitativeMetrics(nx, ny, nz, T2s, data, mask, f_sol_4D,
                                     fM, fIE, fCSF, T2m, T2IE, Ktotal, ind_m,
                                     ind_t, ind_csf)
    # -------------------------- Save all datasets -----------------------------

    sl.save_nibDatasets(fM,m_PathToSaveData,'MWF')
    sl.save_nibDatasets(fIE,m_PathToSaveData,'IEWF')
    sl.save_nibDatasets(fCSF,m_PathToSaveData,'FWF')
    sl.save_nibDatasets(T2m,m_PathToSaveData,'T2_M')
    sl.save_nibDatasets(T2IE,m_PathToSaveData,'T2_IE')
    sl.save_nibDatasets(Ktotal,m_PathToSaveData,'TWC')
    sl.save_nibDatasets(FA,m_PathToSaveData,'FA')
    sl.save_nibDatasets(f_sol_4D,m_PathToSaveData,'fsol_4D')
    sl.save_nibDatasets(s_sol_4D,m_PathToSaveData,'Est_Signal')
    sl.save_nibDatasets(reg_param,m_PathToSaveData,'reg_param')


    print('Done!')
#end main function

    return None