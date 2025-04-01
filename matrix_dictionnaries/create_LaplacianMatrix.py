import numpy as np
import scipy as sci
import sys

def create_MatrixL2(m_Npc):
    '''Function that do this

    Parameters:

    
    '''
    main_diag = np.ones(m_Npc,     dtype=np.double)*(2.0)
    side_diag = -1.0 * np.ones(m_Npc-1, dtype=np.double)
    diagonals = [main_diag, side_diag, side_diag]
    laplacian = sci.sparse.diags(diagonals, [0, -1, 1], format='csr')
    Laplac    = laplacian.toarray()
    # Newman boundary conditions
    Laplac [0,0]        = 1.0
    Laplac [-1,-1]      = 1.0
    return Laplac

def create_MatrixL1(m_Npc):
    '''Function that do this

    Parameters:

    
    '''
    main_diag = np.ones(m_Npc,     dtype=np.double)*(1.0)
    side_diag = -1.0 * np.ones(m_Npc-1, dtype=np.double)
    diagonals = [main_diag, side_diag]
    laplacian = sci.sparse.diags(diagonals, [0, -1], format='csr')
    Laplac    = laplacian.toarray()
    return Laplac

def create_MatrixInvT2(m_T2s):
    '''Function that do this

    Parameters:

    
    '''
    # Regularization matrix for method: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3568216/
    #: Junyu Guo et al. 2014. Multi-slice Myelin Water Imaging for Practical Clinical Applications at 3.0 T.
    T2s_mod = np.concatenate( (np.array([m_T2s[0] - 1.0]), m_T2s[:-1]) ) # add 0.0 and remove the last one
    deltaT2 = m_T2s - T2s_mod
    deltaT2[0] = deltaT2[1]
    Laplac  = np.diag(1./deltaT2)
    return Laplac

def create_RegMatrix(m_Npc,m_T2s,m_RegMatrix):
    '''Function that do this

    Parameters:

    
    '''
    match m_RegMatrix:
        case 'I':
            return np.eye(m_Npc)
        case 'L1':
            return create_MatrixL1(m_Npc)
        case 'L2':
            return create_MatrixL2(m_Npc)
        case 'InvT2':
            return create_MatrixInvT2(m_T2s)
        case _:
            sys.exit('Error: Wrong reg_matrix option')

