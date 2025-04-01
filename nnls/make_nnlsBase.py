import numpy as np
import scipy.optimize as opt

# Standard NNLS python in scipy.
# The default number of iterations was increased from 3n to 5n to improve
# the estimation of smooth solutions.
# The 'too many iterations' error was removed.

def nnls(m_A, m_b):
    m_A, m_b = map(np.asarray_chkfinite, (m_A, m_b))

    #if len(m_A.shape) != 2:
    #    raise ValueError('expected matrix')
    #if len(m_b.shape) != 1:
    #    raise ValueError('expected vector')

    m, n = m_A.shape

    #if m_M != m_b.shape[0]:
    #    raise ValueError('incompatible dimensions')

    #maxiter = -1 if maxiter is None else int(maxiter)
    maxiter = -1
    #maxiter = int(5*n)

    w     = np.zeros((n,), dtype=np.double)
    zz    = np.zeros((m,), dtype=np.double)
    index = np.zeros((n,), dtype=int)
    
    #m_x, rnorm, mode = _nnls.nnls(m_A, m, n, m_b, w, zz, index, maxiter)
    m_x, rnorm, mode = opt.nnls(m_A, m, n, m_b, w, zz, index, maxiter)

    #if mode != 1:
    #    raise RuntimeError('too many iterations')
    return m_x, rnorm
#end