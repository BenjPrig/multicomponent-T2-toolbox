#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ============================================= #
# factored tools for ejcr 
# B. Prigent
# v 1, 20-Fev-2025
# ============================================= #
'''
    Script to help factorized ejcr toolbox
'''

import sys
import nibabel as nib
import numpy as np

def get_shape(m_Data,m_nameDataset):
    '''Function that do this

    Parameters:

    
    '''
    print(f'--------- {m_nameDataset} shape -----------------')
    print(m_Data.shape)
    print('--------------------------------------')
    return m_Data.shape


def load_niftiFloatImage(m_ImagePath,):
    ''' Load a path and return a np array of float

    Parameters:
        m_ImagePath -- str

    Returns:
        a np.float64 of the path gave
    
    '''
    img      = nib.load(m_ImagePath)
    data     = img.get_fdata()
    data     = data.astype(np.float64, copy=False)

    return data

def load_niftiIntImage(m_ImagePath,):
    ''' Load a path and return a np array of float

    Parameters:
        m_ImagePath -- str

    Returns:
        a np.float64 of the path gave
    
    '''
    img      = nib.load(m_ImagePath)
    data     = img.get_fdata()
    data     = data.astype(np.int64, copy=False)

    return data

def save_nibDatasets(m_ArrayToConvert,m_PathToSaveData,m_NameFile):
    '''Function that do this

    Parameters:

    
    '''
    imgSave = nib.Nifti1Image(m_ArrayToConvert,np.eye(4))
    nib.save(imgSave,m_PathToSaveData+m_NameFile+'.nii.gz')
    return None

def convert_zero2nan(m_Dataset):
    '''Function that do this

    Parameters:

    
    '''
    m_Dataset[m_Dataset == 0] = np.nan

    return m_Dataset