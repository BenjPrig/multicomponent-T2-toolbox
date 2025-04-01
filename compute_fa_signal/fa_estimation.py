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
import sys
import numpy as np
import joblib as job
import scipy.optimize as opt
import scipy.ndimage as ndi
import scipy.interpolate as intr
import progressbar
import multiprocessing
import matrix_dictionnaries.epg as epg

sys.path.append("..")

import plot.plot_results as plot
import nnls.make_nnlsGCV as gcv
import nnls.make_nnlsX2 as x2
import nnls.make_nnlsLcurve as lcurv
import nnls.make_nnlsTikh as nTikh
import nnls.bayesian_interpolation as bayes
import save_load_img as sl
# import make_nnlsBase as nnls

#===============================================================================
#                                FUNCTIONS
#===============================================================================

# ------------------- Spline-based interpolation ------------------------------#
def fitting_slice_FA_spline_method(m_Nx, m_Data1D, m_Mask1D, m_Dic3D, m_Dic3DLR, m_AlphaValues, m_AlphaValuesSpline):
    '''Function that do this

    Parameters:

    
    '''
    tmp_FA            = np.zeros((m_Nx))
    tmp_FA_index      = np.zeros((m_Nx))
    totVoxels_sclices = np.count_nonzero(m_Mask1D)
    tmp_KM            = np.zeros((m_Nx))
    tmp_Fsol          = 0.0
    # -------------------------------------------
    dim3              = m_Dic3DLR.shape[2]
    if totVoxels_sclices > 0:
        for voxelx in range(0, m_Nx):
            if (m_Mask1D[voxelx] > 0.0) & (np.sum(m_Data1D[voxelx, :]) > 0.0):
                m_M        = m_Data1D[voxelx, :]
                # m_M        = m_M/m_M[0]
                residual = np.zeros((dim3))
                for i in range(dim3):
                    Dic_i        = np.ascontiguousarray(m_Dic3DLR[:,:,i])
                    f, rnorm_f   = opt.nnls( Dic_i, m_M, maxiter=-1 )
                    residual[i]  = rnorm_f
                #end for iter
                f2  = intr.interp1d(m_AlphaValuesSpline, residual, kind='cubic')
                res = opt.minimize_scalar(f2, method='Bounded', bounds=(90., 180.))
                # Find FA closest to the predefined grid m_AlphaValues
                indexFA = np.argmin( np.abs(m_AlphaValues - res.x) )
                tmp_FA_index[voxelx] = indexFA
                tmp_FA[voxelx]       = m_AlphaValues[indexFA]
                # ------- estimate PD and T2 distribution
                Dic_i          = np.ascontiguousarray(m_Dic3D[:,:,indexFA])
                fsol, f_sqrtn  = opt.nnls( Dic_i, m_M , maxiter=-1)
                km_i           = np.sum(fsol)
                tmp_KM[voxelx] = km_i
                tmp_Fsol       = tmp_Fsol + fsol
            #end if mask
        #end for x
    #end if
    return tmp_FA, tmp_FA_index, tmp_KM, tmp_Fsol
#end fun

def fitting_slice_FA_brute_force( m_Nx, m_Data1D, m_Mask1D, m_Dic3D, m_AlphaValues):
    '''Function that do this

    Parameters:

    
    '''
    tmp_FA         = np.zeros((m_Nx))
    tmp_FA_index   = np.zeros((m_Nx))
    tmp_KM         = np.zeros((m_Nx))
    tmp_Fsol       = 0.0
    totVoxels_sclices = np.count_nonzero(m_Mask1D)
    if totVoxels_sclices > 0:
        # for voxelx in xrange(m_Nx): For python 2 ? Generator ? 
        for voxelx in range(m_Nx):
            if (m_Mask1D[voxelx] > 0.0) and (np.sum(m_Data1D[voxelx, :])) > 0.0:
                m_M      = m_Data1D[voxelx, :]
                # compute the flip angle (alpha_mean) and the proton density (km_i)
                index_i, alpha_mean, km_i, SSE, fsol = compute_optimal_FA(m_M, m_Dic3D, m_AlphaValues)
                tmp_FA[voxelx]         = alpha_mean
                tmp_FA_index[voxelx]   = index_i
                tmp_KM[voxelx]         = km_i
                tmp_Fsol               = tmp_Fsol + fsol
            #end if mask
        #end for x
    #end if
    return tmp_FA, tmp_FA_index, tmp_KM, tmp_Fsol
#end function

def compute_optimal_FA(m_Dic3D, m_M, m_AlphaValues):
    '''Function that do this

    Parameters:

    
    '''
    dim3      = m_Dic3D.shape[2]
    residual  = np.zeros((dim3))
    m_M = np.ascontiguousarray(m_M)
    for iter in range(dim3):
        Dic_i          = np.ascontiguousarray(m_Dic3D[:,:,iter])
        f, rnorm_f     = opt.nnls( Dic_i, m_M, maxiter=-1 )
        residual[iter] = rnorm_f
    #end for
    index       = np.argmin(residual)
    Dic_i       = np.ascontiguousarray(m_Dic3D[:,:,index])
    f, f_sqrtn  = opt.nnls( Dic_i, m_M, maxiter=-1 )
    km          = np.sum(f)
    alpha       = m_AlphaValues[index]
    SSE         = np.sum( (np.dot(Dic_i, f) - m_M)**2 )
    return index, alpha, km, SSE, f
#end function


def chose_FaMethod(m_Nx, m_Ny, m_DataSlice, m_MaskSlice, m_Dic3D, m_Dic3DLR, m_FaMethod, m_AlphaValues, m_AlphaValuesSpline, m_NumCores):
    '''Function that do this

    Parameters:

    
    '''
    match m_FaMethod:
        case 'brute-force':
            FA_Par = job.Parallel(n_jobs=m_NumCores, backend='loky')(
                job.delayed(fitting_slice_FA_brute_force)(
                    m_Nx, m_DataSlice[:,voxely,:], m_MaskSlice[:, voxely],
                    m_Dic3D, m_AlphaValues) for voxely in range(m_Ny))
            return FA_Par
        case 'spline':
            FA_Par = job.Parallel(n_jobs=m_NumCores, backend='loky')(
                job.delayed(fitting_slice_FA_spline_method)(
                    m_Nx, m_DataSlice[:,voxely,:], m_MaskSlice[:, voxely],
                    m_Dic3D, m_Dic3DLR, m_AlphaValues, m_AlphaValuesSpline) for voxely in range(m_Ny))
            return FA_Par
        case _:
            sys.exit('Error: Wrong FA Method')
   

def compute_signalKernel(m_Nx,m_Ny,m_Nz,m_Npc,m_FaIndex,m_Dic3D,m_Data,m_Mask,m_PathToSaveData):
    '''Function that do this

    Parameters:

    
    '''
    total_signal = 0
    total_Kernel = 0
    nv           = 0
    for voxelx in range(m_Nx):
        for voxely in range(m_Ny):
            for voxelz in range(m_Nz):
                if m_Mask[voxelx, voxely,voxelz] == 1:
                    total_signal = total_signal + m_Data[voxelx,voxely,voxelz, :]
                    ind_xyz      = np.int_(m_FaIndex[voxelx,voxely,voxelz])
                    total_Kernel = total_Kernel + m_Dic3D[:,:,ind_xyz]
                    nv = nv + 1.0
            #end vz
        #end vy
    #end vx
    total_Kernel     = total_Kernel/nv
    total_signal     = total_signal/nv
    fmean1, SSE      = opt.nnls(total_Kernel, total_signal, maxiter=-1)
    dist_T2_mean1    = fmean1/np.sum(fmean1)
    # plot.plot_nnlsArgs(total_signal,total_Kernel,0,m_PathToSaveData,'Signal_Kernel')

    
    factor           = 1.01 # smaller than 1.02 due to the low level of noise
    order            = 0
    Id               = np.eye(m_Npc)
    fmean2, reg_opt2, k_est = x2.nnls_x2(total_Kernel, total_signal, Id, factor)
    # plot.plot_nnlsArgs(total_signal,k_est,0,m_PathToSaveData,'fmean')
    dist_T2_mean2    = fmean2/np.sum(fmean2)

    return dist_T2_mean1, dist_T2_mean2


def compute_meanT2dist(m_Nx, m_Ny, m_Nz, m_Nt, m_Data, m_Mask, 
                       m_FaSmooth, m_Fa, m_FaMethod, m_FaIndex, m_Ktotal, 
                       m_Dic3D, m_Dic3DLR, m_AlphaValues, m_AlphaValuesSpline, m_NumCores):
    '''Main function to execute the estimation of Flip Angle

    Parameters:

    
    '''
    data_smooth = chose_SmoothFa(m_Nx, m_Ny, m_Nz, m_Nt, m_FaSmooth,m_Data)
    mean_T2_dist = 0
    for voxelz in progressbar.progressbar(range(m_Nz),redirect_stdout=True):
    # for voxelz in range(m_Nz):
       
        print(voxelz+1, ' slice processed')
        
        # Parallelization by rows: this is more efficient for computing a single or a few slices
        m_MaskSlice = m_Mask[:,:,voxelz]
        data_slice = data_smooth[:,:,voxelz,:]

        FA_Par = chose_FaMethod(m_Nx, m_Ny, data_slice, m_MaskSlice, 
                                m_Dic3D, m_Dic3DLR, m_FaMethod, m_AlphaValues, 
                                m_AlphaValuesSpline, m_NumCores)
        
        for voxely in range(m_Ny):
            m_Fa[:,voxely,voxelz]       = FA_Par[voxely][0]
            m_FaIndex[:,voxely,voxelz] = FA_Par[voxely][1]
            m_Ktotal[:, voxely,voxelz]  = FA_Par[voxely][2]
            mean_T2_dist              = mean_T2_dist + FA_Par[voxely][3]
    
    mean_T2_dist = mean_T2_dist/np.sum(mean_T2_dist)
    return mean_T2_dist, m_FaIndex,data_slice
        

def chose_SmoothFa(m_Nx,m_Ny,m_Nz,m_Nt,m_FaSmooth,m_Data):
    '''Function that do this

    Parameters:

    
    '''
    match m_FaSmooth:
        case 'yes':
            data_smooth = np.zeros((m_Nx,m_Ny,m_Nz,m_Nt))
            sig_g = 2.0
            match m_Nz:
                case 1:
                    for c in range(m_Nt):
                        data_smooth[:,:,:,c] = ndi.gaussian_filter(m_Data[:,:,:,c], sig_g, 0)
                    return data_smooth
                case _:
                    for c in range(m_Nt):
                        data_smooth[:,:,:,c] = ndi.gaussian_filter(np.squeeze(m_Data[:,:,:,c]), sig_g, 0)
                    return data_smooth
        case _:
            return m_Data
        

def fitting_slice_T2(m_Nx, m_Data1D, m_Mask1D, m_Dic3D, FA_index_1d,
                     m_T2dim, m_nEchoes, m_LambdaReg, m_RegMethod, m_Laplac):
    '''Function that do this

    Parameters:

    
    '''
    # --------------------------------------
    tmp_f_sol_4D      = np.zeros((m_Nx, m_T2dim))
    tmp_signal        = np.zeros((m_Nx, m_nEchoes))
    tmp_Reg           = np.zeros((m_Nx))
    totVoxels_sclices = np.count_nonzero(m_Mask1D)
    if totVoxels_sclices > 0 :
        # ----------------------------------------------------------------------
        #                            Voxelwise estimation
        # ----------------------------------------------------------------------
        for voxelx in range(0, m_Nx):
            if (m_Mask1D[voxelx] > 0.0) & (np.sum(m_Data1D[voxelx, :]) > 0.0):
                # ==================== Reconstruction
                M       = np.ascontiguousarray(m_Data1D[voxelx, :])
                index_i = np.int_(FA_index_1d[voxelx])
                Kernel  = np.ascontiguousarray(m_Dic3D[:,:,index_i])
                km_i    = M[0]
                #pdb.set_trace()
                if km_i > 0:  # only if there is signal
                    M = M/km_i
                    x_sol,reg_opt = check_nnls(m_RegMethod,Kernel,m_Laplac,M,m_LambdaReg)
                    # ----------------------------------------------------------
                    # -------------
                    tmp_Reg[voxelx]        = reg_opt  # Regularization parameter
                    tmp_f_sol_4D[voxelx,:] = x_sol * km_i
                    tmp_signal[voxelx,:]   = np.dot(Kernel, x_sol) * km_i
                #end if ki
                # ---------------------------------------------------------#
            #end if mask
        #end for x
    #end if
    return tmp_f_sol_4D, tmp_signal, tmp_Reg
#end main function



def check_nnls(m_RegMethod,m_Kernel,m_Laplac,m_M, m_LambdaReg):
    '''Function that do this

    Parameters:
    
    '''
    match m_RegMethod:

        case 'NNLS':
            x_sol, rnorm_fkk = opt.nnls( m_Kernel, m_M, maxiter=-1 )
            reg_opt = 0
            return [x_sol, reg_opt]
        
        case 'T2SPARC':
            reg_opt= 1.8
            x_sol = nTikh.nnls_tik(m_Kernel, m_M, m_Laplac, reg_opt)
            return [x_sol, reg_opt]
        
        case 'X2':
            k = 1.02
            x_sol, reg_opt, k_est = x2.nnls_x2(m_Kernel, m_M, m_Laplac, k)
            reg_opt    = k_est
            return [x_sol, reg_opt]
        
        case 'L_curve':
            reg_opt = lcurv.nnls_lcurve_wrapper(m_Kernel, m_M, m_Laplac, m_LambdaReg)
            x_sol = nTikh.nnls_tik(m_Kernel, m_M, m_Laplac, reg_opt)
            return [x_sol, reg_opt]
        
        case 'GCV':
            x_sol, reg_opt = gcv.nnls_gcv(m_Kernel, m_M, m_Laplac)
            return [x_sol, reg_opt]
        
        case 'BayerReg':
            x_sol, reg_opt = bayes.BayesReg_nnls(m_Kernel, m_M, m_Laplac)
            return [x_sol, reg_opt]
        
        case _:
            sys.exit('The regularization method does not exist')