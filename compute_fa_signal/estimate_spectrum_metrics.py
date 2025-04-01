import progressbar
import numpy as np
import joblib as job
import compute_fa_signal.fa_estimation as fa

def estimate_t2_spectra(m_Nx, m_Ny, m_Nz, m_T2s, m_Data, m_Mask,
                        m_FaIndex, m_Fsol4D, m_Ssol4D, m_RegParam,
                        m_Dic3D, m_LambdaReg, m_nEchoes,
                        m_RegMethod, m_Laplac,
                        m_NumCores):
    '''Function that do this

    Parameters:

    
    '''    
    for voxelz in progressbar.progressbar(range(m_Nz), redirect_stdout=True):
        print(voxelz+1, ' slices processed')
        # Parallelization by rows: this is more efficient for computing a single or a few slices
        mask_slice = m_Mask[:,:,voxelz]
        data_slice = m_Data[:,:,voxelz,:]
        FA_index_slice = m_FaIndex[:,:,voxelz]
        #T2_par = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(fitting_slice_T2)(mask_slice[:, voxely], data_slice[:,voxely,:], FA_index_slice[:, voxely], nx, Dic_3D, lambda_reg, alpha_values, T2s.shape[0], nEchoes, num_l_laplac, N_alphas, reg_method, Laplac1, Laplac2, Is, Laplac_mod, mean_T2_dist, Laplac2_cp_var, W_inv_deltaT2) for voxely in range(ny))
        
        T2_par = job.Parallel(n_jobs=m_NumCores, backend='multiprocessing')(job.delayed(fa.fitting_slice_T2)(
            m_Nx, data_slice[:,voxely,:], mask_slice[:, voxely], m_Dic3D, 
            FA_index_slice[:, voxely], m_T2s.shape[0], m_nEchoes, m_LambdaReg,  
            m_RegMethod, m_Laplac) for voxely in range(m_Ny))
        
        for voxely in range(m_Ny):
            m_Fsol4D[:,voxely,voxelz,:] = T2_par[voxely][0]
            m_Ssol4D[:,voxely,voxelz,:] = T2_par[voxely][1]
            m_RegParam[:,voxely,voxelz]  = T2_par[voxely][2]
        #end voxely
    #end voxelx

    return m_Fsol4D, m_Ssol4D, m_RegParam

def estimate_quantitativeMetrics(m_Nx, m_Ny, m_Nz, m_T2s,
                                 m_Data, m_Mask, m_Fsol4D,
                                 m_fM, m_fIE, m_fCSF, m_T2m,
                                 m_T2IE, m_Ktotal, m_IndM, m_IndT,
                                 m_IndCsf, m_Epsilon=1.0e-16,):
    '''Function that do this

    Parameters:

    
    '''
    logT2 = np.log(m_T2s)
    for voxelx in range(m_Nx):
        for voxely in range(m_Ny):
            for voxelz in range(m_Nz):
                if m_Mask[voxelx, voxely, voxelz] > 0.0:
                    M     = m_Data[voxelx, voxely, voxelz, :]
                    x_sol = m_Fsol4D[voxelx, voxely, voxelz,:]
                    vt    = np.sum(x_sol) + m_Epsilon
                    x_sol = x_sol/vt
                    # fill matrices
                    # pdb.set_trace()
                    m_fM  [voxelx, voxely, voxelz] = np.sum(x_sol[m_IndM])
                    m_fIE [voxelx, voxely, voxelz] = np.sum(x_sol[m_IndT])
                    m_fCSF[voxelx, voxely, voxelz] = np.sum(x_sol[m_IndCsf])
                    # ------ T2m
                    # Aritmetic mean
                    # T2m [voxelx, voxely, voxelz] = np.sum(x_sol[ind_m] * T2s[ind_m])/(np.sum(x_sol[ind_m])   + epsilon)
                    # Geometric mean: see Bjarnason TA. Proof that gmT2 is the reciprocal of gmR2. Concepts Magn Reson 2011; 38A: 128– 131.
                    m_T2m[voxelx, voxely, voxelz] = np.exp(np.sum(x_sol[m_IndM] * logT2[m_IndM])/(np.sum(x_sol[m_IndM])   + m_Epsilon))
                    # ------ T2IE0
                    # Aritmetic mean
                    # T2IE[voxelx, voxely, voxelz] = np.sum(x_sol[ind_t] * T2s[ind_t])/(np.sum(x_sol[ind_t])   + epsilon)
                    # Geometric mean: see Bjarnason TA. Proof that gmT2 is the reciprocal of gmR2. Concepts Magn Reson 2011; 38A: 128– 131.
                    m_T2IE[voxelx, voxely, voxelz] = np.exp(np.sum(x_sol[m_IndT] * logT2[m_IndT])/(np.sum(x_sol[m_IndT])   + m_Epsilon))
                    m_Ktotal[voxelx, voxely, voxelz] = vt
                # end if
            #end for z
        # end for y
    # end for x

    return m_fM,m_fIE,m_fCSF,m_T2m,m_T2IE,m_Ktotal