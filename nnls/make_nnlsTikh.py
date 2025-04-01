import numpy as np
import scipy.optimize as opt
import plot.plot_results as plot

# ------------------------------------------------------------------------------
# Tikhonov regularization using a Laplacian matrix and a fixed reg. parameter
# ------------------------------------------------------------------------------
def nnls_tik(m_Dic_i, m_M, m_Laplac, m_RegOpt):
    '''Function that do this

    Parameters:

    
    '''
    pathToSaveData = r'C:\data\recon\derivatives\sub-011181\testFactored'+'\\'
    # figlcurv = plot.plot_nnlsArgs(m_M,m_Dic_i,0,pathToSaveData,'LCurve')
    m, n         = m_Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug       = np.concatenate((m_M, Zerosm))
    # figlcurv = plot.plot_nnlsArgs(M_aug,np.concatenate((m_Dic_i, np.sqrt(m_RegOpt)*m_Laplac)),0,pathToSaveData,'LCurveRegu')
    # --------- Estimation
    f, rnorm_f  = opt.nnls( np.concatenate((m_Dic_i, np.sqrt(m_RegOpt)*m_Laplac)), M_aug )
    return f
#end fun

