import numpy as np
import scipy.optimize as opt
import plot.plot_results as plot

# ------------------------------------------------------------------------------
#                        X2: conventional method of Mackay
# ------------------------------------------------------------------------------

def nnls_x2(m_Dic_i, m_M, m_Laplac, m_Factor):
    '''Function that do this

    Parameters:

    
    '''
    f0, kk      = opt.nnls( m_Dic_i, m_M ,maxiter=-1)
    m_Sse         = np.sum( (np.dot(m_Dic_i, f0) - m_M)**2 )
    # -----------------------
    m,n         = m_Dic_i.shape
    Zerosm      = np.zeros((n))
    M_aug1      = np.concatenate((m_M, Zerosm))
    # m_Factor      = 1.02
    m_RegOpt     = opt.fminbound(obj_nnls_x2, 0.0, 10.0, args=(m_Dic_i, m_Laplac, M_aug1, m_Sse, m_Factor, m_M), xtol=1e-05, maxfun=300, full_output=0, disp=0)
    f, rnorm_f  = opt.nnls( np.concatenate((m_Dic_i, np.sqrt(m_RegOpt)*m_Laplac)), M_aug1,maxiter=-1 )
    #return f, m_RegOpt
    k_est       = np.sum( (np.dot(m_Dic_i, f) - m_M)**2 )/m_Sse
    return f, m_RegOpt, k_est
#end fun

def obj_nnls_x2(m_x, m_D, m_L, m_Signal, m_Sse, m_Factor, m_M):
    '''Function that do this

    Parameters:

    
    '''
    Daux     = np.concatenate((m_D, np.sqrt(m_x)*m_L))
    f, kk    = opt.nnls( Daux, m_Signal,maxiter=-1 )
    #SSEr     = np.sum( (np.dot(Daux, f) - m_Signal)**2 )
    SSEr     = np.sum( (np.dot(m_D, f) - m_M)**2 )
    cost_fun = np.abs(SSEr - m_Factor*m_Sse)/m_Sse
    return cost_fun
# end fun

# ------------------------------------------------------------------------------
#                        X2 using an apriori estimate
# ------------------------------------------------------------------------------
def nnls_x2_prior(m_Dic_i, m_M, m_x0, m_Factor):
    '''Function that do this

    Parameters:

    
    '''
    f0, kk      = opt.nnls( m_Dic_i, m_M,maxiter=-1 )
    m_Sse         = np.sum( (np.dot(m_Dic_i, f0) - m_M)**2 )
    # -----------------------
    m_M,n         = m_Dic_i.shape
    m_Laplac      = np.eye(n)
    #m_Factor      = 1.02
    m_RegOpt     = opt.fminbound(obj_nnls_x2_prior, 0.0, 100.0, args=(m_Dic_i, m_Laplac, m_M, m_Sse, m_Factor, m_x0), xtol=1e-05, maxfun=300, full_output=0, disp=0)
    f, rnorm_f  = opt.nnls( np.concatenate( (m_Dic_i, np.sqrt(m_RegOpt)*m_Laplac) ), np.concatenate( (m_M, np.sqrt(m_RegOpt)*m_x0) ),maxiter=-1 )
    return f, m_RegOpt
#end fun

def obj_nnls_x2_prior(m_x, m_D, m_L, m_M, m_Sse, m_Factor, m_x0):
    '''Function that do this

    Parameters:

    
    '''
    Daux     = np.concatenate( (m_D, np.sqrt(m_x) * m_L ) )
    m_Signal   = np.concatenate( (m_M, np.sqrt(m_x) * m_x0) )
    f, kk    = opt.nnls( Daux, m_Signal,maxiter=-1 )
    SSEr     = np.sum( (np.dot(Daux, f) - m_Signal)**2 )
    cost_fun = np.abs(SSEr - m_Factor*m_Sse)/m_Sse
    return cost_fun
# end fun
