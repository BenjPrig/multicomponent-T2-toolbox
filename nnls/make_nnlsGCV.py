import numpy as np
import scipy.optimize as opt



# ------------------------------------------------------------------------------
#                                 GCV
# Modified GCV that selects the subset of columns in the dictionary with
# corresponding positive coefficients
# ------------------------------------------------------------------------------
def nnls_gcv(m_Dic_i, m_M, m_L):
    '''Function that do this

    Parameters:

    
    '''
    m,n         = m_Dic_i.shape
    M_aug       = np.concatenate( (m_M, np.zeros((n))) )
    m_Im          = np.eye(m)
    m_RegOpt     = opt.fminbound(obj_nnls_gcv, 1e-8, 10.0, args=(m_Dic_i, m_L, M_aug, m, m_Im), xtol=1e-05, maxfun=300, full_output=0, disp=0)
    f, rnorm_f  = opt.nnls( np.concatenate((m_Dic_i, np.sqrt(m_RegOpt)*m_L)), M_aug,maxiter=-1 )
    return f, m_RegOpt
#end fun

def obj_nnls_gcv(m_x, m_D, m_L, m_Signal, m_m, m_Im):
    '''Function that do this

    Parameters:

    
    '''
    Daux     = np.concatenate((m_D, np.sqrt(m_x)*m_L))
    f, SSEr  = opt.nnls( Daux, m_Signal,maxiter=-1 )
    Dr       = m_D[:, f>0]
    Lr       = m_L[f>0, f>0]
    DTD      = np.matmul(Dr.T, Dr)
    LTL      = np.matmul(Lr.T, Lr)
    #A        = np.matmul(Dr, np.matmul( inv( DTD + m_x*LTL ), Dr.T) )
    A        = np.matmul(Dr, np.linalg.lstsq(DTD  + m_x*LTL, Dr.T, rcond=None)[0] )
    cost_fun = ( (1.0/m_m)*(SSEr**2.0) ) / ((1.0/m_m) * np.trace(m_Im - A) )**2.0
    return np.log(cost_fun)
# end fun
