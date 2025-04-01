import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import save_load_img as sl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_nnlsArgs(m_TotalSignal, m_TotalKernel, m_Slice, m_PathToSaveData,m_Method):

    #end

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,8.0), constrained_layout=True)

    ax0, ax1 = axes.flatten()

    ax0.plot(m_TotalSignal)
    ax0.set_title('Signal distribution')
    
    ax1.plot(m_TotalKernel)
    ax1.set_title('Dictionnary distribution')

    # ax2.plot(m_Signal[])

    plt.savefig(m_PathToSaveData + m_Method + '_distribution.png')
    return None

def plot_signal_dict(m_Signal,m_Mask,m_SignalDict,m_Slice):
    ''' Plot the comparison between the signal and the estimated signal.

    Parameters:
        m_ImagePath -- str

    Returns:
        a np.float64 of the path gave
    '''
    nx,ny,nz,nt = sl.get_shape(m_Signal,'Data')
    total_signal = 0
    total_dict = 0
    nv = 0
    for voxelx in range(nx):
        for voxely in range(ny):
            for voxelz in range(nz):
                total_signal = total_signal + m_Signal[voxelx,voxely,voxelz, :]
                total_dict = total_dict + m_SignalDict[voxelx,voxely,voxelz, :]
                nv = nv+1
    total_signal = total_signal / nv
    total_dict = total_dict / nv
    fig = plt.figure('Distribution', figsize=(6,6), constrained_layout=True)
    fig = plt.plot(total_signal ,color='r', label='Signal')
    fig = plt.plot(total_dict ,color='b', label='Dictionnary')
    fig = plt.legend(loc='best')
    # plt.show()
    return fig