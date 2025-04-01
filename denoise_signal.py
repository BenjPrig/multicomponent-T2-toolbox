import numpy as np
import scipy.optimize as opt
import progressbar
import skimage as ski 
import nibabel as nib
import sys

def denoise_signal_tv(m_Nt,m_Data,m_PathToSaveData):
    '''Function that do this

    Parameters:

    
    '''
    print('Step #1 : Denoising using total variation')
    for voxelt in progressbar.progressbar(range(m_Nt), redirect_stdout=True):
        print(voxelt+1, ' volumes processed')
        data_vol  = np.squeeze(m_Data[:,:,:,voxelt])
        sigma_est = np.mean(ski.restoration.estimate_sigma(data_vol, channel_axis=None))
        #data[:,:,:,voxelt] = denoise_tv_chambolle(data_vol, weight=1.0*sigma_est, eps=0.0002, n_iter_max=200, multichannel=False)
        m_Data[:,:,:,voxelt] = ski.restoration.denoise_tv_chambolle(data_vol, weight=2.0*sigma_est, eps=0.0002, max_num_iter=200, channel_axis=None)
    #end for
    outImg = nib.Nifti1Image(m_Data, opt.img.affine)
    nib.save(outImg, m_PathToSaveData + 'Data_denoised.nii.gz')
    return m_Data

def denoise_signal_nesma(m_Nx,m_Ny,m_Nz,m_Nt,m_Data,m_Mask):
    '''Function that do this

    Parameters:

    
    '''
    data_den  = np.zeros_like(m_Data)
    path_size = [6,6,6] # real-size = 2*path_size + 1
    print('Step #1: Denoising using the NESMA filter:')
    for voxelx in progressbar.progressbar(range(m_Nx), redirect_stdout=True):
        print(voxelx+1, ' slices processed')
        min_x = np.max([voxelx - path_size[0], 0])
        max_x = np.min([voxelx + path_size[0], m_Nx])
        for voxely in range(m_Ny):
            min_y = np.max([voxely - path_size[1], 0])
            max_y = np.min([voxely + path_size[1], m_Ny])
            for voxelz in range(m_Nz):
                if m_Mask[voxelx, voxely,voxelz] == 1:
                    min_z = np.max([voxelz - path_size[2], 0])
                    max_z = np.min([voxelz + path_size[2], m_Nz])
                    # -----------------------------------------
                    signal_path   = m_Data[min_x:max_x, min_y:max_y, min_z:max_z, :]
                    dim           = signal_path.shape
                    signal_path2D = signal_path.reshape((np.prod(dim[0:3]), m_Nt))
                    signal_xyz    = m_Data[voxelx, voxely,voxelz]
                    RE            = 100 * np.sum(np.abs(signal_path2D - signal_xyz), axis=1)/np.sum(signal_xyz)
                    ind_valid     = RE < 2.5 # (percent %)
                    data_den[voxelx, voxely, voxelz] = np.mean(signal_path2D[ind_valid,:], axis=0)
    return data_den



def execute_mainDenoise(m_Nx,m_Ny,m_Nz,m_Nt,m_Data,m_Mask,m_Denoise,m_PathToSaveData):
    '''Function that do this

    Parameters:

    
    '''
    match m_Denoise:
        case 'TV':
            return denoise_signal_tv(m_Nt,m_Data,m_PathToSaveData)
        case 'NESMA':
            return denoise_signal_nesma(m_Nx,m_Ny,m_Nz,m_Nt,m_Data,m_Mask)
        case 'None':
            return m_Data
        case _:
            sys.exit('Bad denoised declared')
        

if __name__ == '__main__':
    sys.exit()