import numpy as np 
import math

def intitiate_t2values(m_MyelinThershold,m_Npc):
    '''Function that do this

    Parameters:

    
    '''
    T2m0   = 10.0
    T2mf   = m_MyelinThershold
    T2tf   = 200.0
    T2csf  = 2000.0

    T2s     = np.logspace(math.log10(T2m0), math.log10(T2csf), num=m_Npc, endpoint=True, base=10.0)

    ind_m   = T2s <= T2mf              # myelin
    ind_t   = (T2s>T2mf)&(T2s<=T2tf)   # intra+extra
    ind_csf = T2s >= T2tf              # quasi free-water and csf

    T1s     = 1000.0*np.ones_like(T2s) # a constant T1=1000 is assumed for all compartments

    return T2s,ind_m,ind_t,ind_csf,T1s

def choose_numberCompartment(m_RegMethod):
    '''Function that do this

    Parameters:

    
    '''
    match m_RegMethod:
        case 'T2SPARC':
            # Regularization matrix for method: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3568216/
            #: Junyu Guo et al. 2014. Multi-slice Myelin Water Imaging for Practical Clinical Applications at 3.0 T.
            return 96
        case _:
            return 60