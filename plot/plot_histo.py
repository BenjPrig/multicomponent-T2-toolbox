import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from plot.tool_plot import colorbar

def plot_result_histo(m_Slice,m_FA,m_fM,m_T2m,m_T2IE,m_DataType='invivo'):
    epsilon = 1.0e-16
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18,8.0), constrained_layout=True)
    ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes.flatten()

    im1 = ax0.imshow(m_FA[:,:,m_Slice].T, cmap='gray', origin='upper', clim=(90,180))
    ax0.set_title('Flip Angle')
    colorbar(im1)

    x = m_FA[:,:,m_Slice].flatten()
    # x = x[x>0.0]
    x = x[x>0.01]
    ax4.hist(x, 50, density=1, facecolor='lime', alpha=1.0)
    ax4.set_title('Histogram of FA')
    ax4.set_xlabel('FA')
    ax4.set_ylabel('Probability')
    ax4.grid(False)

    im1 = ax1.imshow(m_fM[:,:,m_Slice].T, cmap='gray', origin='upper', clim=(0.01,0.25))
    ax1.set_title('MWF')
    colorbar(im1)

    x = m_fM[:,:,m_Slice].flatten()
    # x = x[x>0.0]
    x = x[x>0.01]
    ax5.hist(x, 50, density=1, facecolor='SkyBlue', alpha=1.0, range=[0.01, 0.4])
    ax5.set_title('Histogram of MWF')
    ax5.set_xlabel('MWF')
    ax5.set_ylabel('Probability')
    ax5.grid(False)

    match m_DataType:
        case 'invivo':
            im1 = ax2.imshow(m_T2m[:,:,m_Slice].T, cmap='gray', origin='upper', clim=(10,40))
            ax2.set_title('T2m')
            colorbar(im1)

            x = m_T2m[:,:,m_Slice].flatten()
            # x = x[x>0.0]
            x = x[x>0.01]
            ax6.hist(x, 50, density=1, facecolor='IndianRed', alpha=1.0, range=[10, 40])
            ax6.set_title('Histogram of T2m')
            ax6.set_xlabel('T2m')
            ax6.set_ylabel('Probability')
            ax6.grid(False)

            im1 = ax3.imshow(m_T2IE[:,:,m_Slice].T, cmap='gray', origin='upper', clim=(50,100))
            ax3.set_title('T2IE')
            colorbar(im1)

            x = m_T2IE[:,:,m_Slice].flatten()
            # x = x[x>0.0]
            x = x[x>0.01]
            ax7.hist(x, 50, density=1, facecolor='tan', alpha=1.0, range=[40, 110])
            ax7.set_title('Histogram of T2IE')
            ax7.set_xlabel('T2IE')
            ax7.set_ylabel('Probability')
            ax7.grid(False)
    
        case 'exvivo' :
            im1 = ax2.imshow(m_T2m[:,:,m_Slice].T, cmap='gray', origin='upper', clim=(5,25))
            ax2.set_title('T2m')
            colorbar(im1)

            x = m_T2m[:,:,m_Slice].flatten()
            # x = x[x>0.0]
            x = x[x>0.01]
            ax6.hist(x, 50, density=1, facecolor='IndianRed', alpha=1.0)
            ax6.set_title('Histogram of T2m')
            ax6.set_xlabel('T2m')
            ax6.set_ylabel('Probability')
            ax6.grid(False)

            im1 = ax3.imshow(m_T2IE[:,:,m_Slice].T, cmap='gray', origin='upper', clim=(30,60))
            ax3.set_title('T2IE')
            colorbar(im1)

            x = m_T2IE[:,:,m_Slice].flatten()
            # x = x[x>0.0]
            x = x[x>0.01]
            ax7.hist(x, 50, density=1, facecolor='tan', alpha=1.0)
            ax7.set_title('Histogram of T2IE')
            ax7.set_xlabel('T2IE')
            ax7.set_ylabel('Probability')
            ax7.grid(False)

    return fig

def plot_mwfHisto(m_fM,m_Data,m_Slice):
    """ 
    Plot a colormap for mwf of a brain m_Slice and an histogram about frequency of
    mwf
    """
    epsilon = 1.0e-16
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,8.0), constrained_layout=True)
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.8)
    cmap = mpl.colormaps.get_cmap('inferno')
    cmap.set_bad(color='black')
    ax0, ax1 = axes.flatten()

    im1 = ax0.imshow(np.rot90(m_fM[:,:,m_Slice]), cmap=cmap, origin='upper', clim=(0,0.25))
    im2 = ax0.imshow(np.rot90(m_Data[:,:,m_Slice,0]),alpha=0.3)
    # im2 = ax0.imshow(np.rot90(m_Data[:,:,m_Slice]),alpha=0.3)
    
    ax0.set_title('Carte de la MWF')
    ax0.set_xticks([])
    ax0.set_yticks([])
    colorbar(im1)
    
    x = m_fM[:,:,m_Slice].flatten()
    # x = x[x>0.0]
    x = x[x>0.0]
    
    med = np.median(x)
    mean = np.mean(x)
    stats = (
        f"moyenne = {mean:.2f}\n"
        f"médiane = {med:.2f}"
        )
    ax1.hist(x, 50, density=1, facecolor='SkyBlue', alpha=1.0, range=[0.0, 0.4])
    ax1.set_title('Histogramme de la MWF')
    ax1.set_xlabel('MWF')
    ax1.set_ylabel('Fréquence')
    ax1.text(0.6, 0.8, stats, transform=ax1.transAxes,bbox=bbox)
    ax1.grid(False)

    return fig

if __name__ == '__main__':
    sys.exit('Done')