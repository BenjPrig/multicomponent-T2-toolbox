import numpy as np
import scipy.optimize as opt


# ------------------------------------------------------------------------------
#                                  m_L-CURVE
# ------------------------------------------------------------------------------
# Wrapper to the augmented nnls algorithm for many different lambdas

def nnls_lcurve_wrapper(m_D, m_y, m_LaplacMod, m_LambdaReg):
    '''Function that do this

    Parameters:

    
    '''
    # This script is using the fact that L_mod.T*L_mod = m_L.T*m_L+ Is.T*Is,
    # where L_mod is found via the cholesky decomposition
    m, n = m_D.shape
    # ------------------------------------
    # Define variables
    b            = np.concatenate((m_y, np.zeros((n))))
    num_l_laplac = len(m_LambdaReg)
    Log_error    = np.zeros((num_l_laplac))
    Log_norms    = np.zeros((num_l_laplac))
    # -------------------------------------
    for i_laplac in range(0, num_l_laplac):
        lambda_reg_i = m_LambdaReg[i_laplac]
        A  = np.concatenate( (m_D, np.sqrt(lambda_reg_i)*m_LaplacMod) )
        # ---------------------  Standard NNLS - scipy -------------------------
        m_x, rnorm = opt.nnls(A, b,maxiter=-1)
        # ----------------------------------------------------------------------
        # Variables for the m_L-curve Method
        Log_error[i_laplac] = np.log( np.sum( ( np.dot(m_D, m_x) - m_y  )**2.0 )     + 1e-200)
        Log_norms[i_laplac] = np.log( np.sum( ( np.dot(m_LaplacMod, m_x) )**2.0 ) + 1e-200)
        # ---------------------------------
    #end for
    corner   = select_corner(Log_error, Log_norms)
    m_RegOpt  = m_LambdaReg[corner]
    return m_RegOpt
#end fun

def nnls_lcurve_wrapper_prior(m_D, m_y, m_L, m_x0, m_LambdaReg):
    '''Function that do this

    Parameters:

    
    '''
    m, n = m_D.shape
    # ------------------------------------
    # Define variables
    num_l_laplac = len(m_LambdaReg)
    Log_error    = np.zeros((num_l_laplac))
    Log_norms    = np.zeros((num_l_laplac))
    # -------------------------------------
    for i_laplac in range(0, num_l_laplac):
        lambda_reg_i = m_LambdaReg[i_laplac]
        A  = np.concatenate( (m_D, np.sqrt(lambda_reg_i) * m_L ) )
        b  = np.concatenate( (m_y, np.sqrt(lambda_reg_i) * m_x0) )
        # ---------------------  Standard NNLS - scipy -------------------------
        m_x, rnorm = opt.nnls(A, b,maxiter=-1)
        # ----------------------------------------------------------------------
        # Variables for the m_L-curve Method
        Log_error[i_laplac] = np.log( np.sum( ( np.dot(m_D, m_x) - m_y  )**2.0 ) + 1e-200)
        Log_norms[i_laplac] = np.log( np.sum( ( np.dot(m_L, (m_x-m_x0)) )**2.0 ) + 1e-200)
        # ---------------------------------
    #end for
    # Smoothing
    Log_error = opt.savgol_filter(Log_error, 9, 3)
    Log_norms = opt.savgol_filter(Log_norms, 9, 3)

    corner   = select_corner(Log_error, Log_norms)
    m_RegOpt  = m_LambdaReg[corner]
    # ----------------------------------------
    #       Compute final solution
    # ----------------------------------------
    A  = np.concatenate( (m_D, np.sqrt(m_RegOpt) * m_L ) )
    b  = np.concatenate( (m_y, np.sqrt(m_RegOpt) * m_x0) )
    x_sol, rnorm = opt.nnls(A, b)
    return x_sol, m_RegOpt
#end fun

def select_corner(m_x,m_y):
    '''
    Select the corner value of the m_L-curve formed inversion results.
    References:
    Castellanos, J. m_L., S. Gomez, and V. Guerra (2002), The triangle method
    for finding the corner of the m_L-curve, Applied Numerical Mathematics,
    43(4), 359-373, doi:10.1016/S0168-9274(01)00179-9.

    http://www.fatiando.org/v0.5/_modules/fatiando/inversion/hyper_param.html
    '''
    m_x, m_y = scale_curve(m_x,m_y)
    n = len(m_x)
    corner = n - 1

    def dist(p1, p2):
        'Return the geometric distance between p1 and p2'
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    #end

    cte = 7. * np.pi / 8.
    angmin = None
    c = [m_x[-1], m_y[-1]]
    for k in range(0, n - 2):
        b = [m_x[k], m_y[k]]
        for j in range(k + 1, n - 1):
            a = [m_x[j], m_y[j]]
            ab = dist(a, b)
            ac = dist(a, c)
            bc = dist(b, c)
            cosa = (ab ** 2 + ac ** 2 - bc ** 2) / (2. * ab * ac)
            cosa = max(-1.0, min(cosa, 1.0)) # valid range: [-1, 1]
            ang  = np.arccos(cosa)
            area = 0.5 * ((b[0] - a[0]) * (a[1] - c[1]) - (a[0] - c[0]) * (b[1] - a[1]))
            # area is > 0 because in the paper C is index 0
            if area > 0 and (ang < cte and (angmin is None or ang < angmin)):
                corner = j
                angmin = ang
            #end if
        #end for j
    #end for k
    return corner
#end fun

def scale_curve(m_x,m_y):
    '''
    Puts the data-misfit and regularizing function values in the range
    [-10, 10].

    http://www.fatiando.org/v0.5/_modules/fatiando/inversion/hyper_param.html
    '''
    def scale(a):
        vmin, vmax = a.min(), a.max()
        m_L, u = -10, 10
        return (((u - m_L) / (vmax - vmin)) * (a - (u * vmin - m_L * vmax) / (u - m_L)))
    #end fun
    return scale(m_x), scale(m_y)
#end fun