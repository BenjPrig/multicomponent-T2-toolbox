import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import save_load_img as sl


def plot_mean_spectrum(m_T2s,m_MeanT2Dist,m_DistT2Mean1,m_DistT2Mean2,m_PathToSaveData):
    fig  = plt.figure('Showing results', figsize=(8,8))
    ax0  = fig.add_subplot(1, 1, 1)
    im0  = plt.plot(m_T2s, m_MeanT2Dist,  color='b', label='Mean T2-dist from all voxels: NNLS')
    im1  = plt.plot(m_T2s, m_DistT2Mean1, color='g', label='T2-dist from mean signals: NNLS')
    im2  = plt.plot(m_T2s, m_DistT2Mean2, color='r', label='T2-dist from mean signals: NNLS-X2-I')

    ax0.set_xscale('log')
    plt.axvline(x=40.0, color='k', linestyle='--', ymin=0)
    plt.title('Mean spectrum', fontsize=18)
    plt.xlabel('T2', fontsize=18)
    plt.ylabel('Intensity', fontsize=18)
    ax0.set_xlim(m_T2s[0], m_T2s[-1])
    ax0.set_ylim(0, np.max(m_MeanT2Dist)*1.2)
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.tick_params(axis='both', which='minor', labelsize=14)
    ax0.set_yticks([])
    plt.legend()
    plt.savefig(m_PathToSaveData + 'Mean_spectrum_unitial_iter.png', dpi=600)
    plt.close('all')
    return None

def plot_mean_spectrum_slices(m_Fsol4D, m_Mask, m_Slice):
    print('Plotting T2 spectra')

    loc  = 1
    
    nx,ny,nz,nt = sl.get_shape(m_Fsol4D,'f_Sol_4D')
    match nz:
        case 1:
            m_Mask = np.squeeze(m_Mask)
            mask2 = np.zeros_like(m_Mask)
            mask2[:,:] = m_Mask[:,:]
            m_Mask = mask2
        case _:
            mask2 = np.zeros_like(m_Mask)
            mask2[:,:,m_Slice] = m_Mask[:,:,m_Slice]
            m_Mask = mask2
    
    ind_mask = m_Mask == 1

    print ('Plotting:', np.sum(ind_mask), 'T2 distributions')

    fsol_2D        = np.zeros((np.sum(ind_mask), nt))
    T2s            = np.logspace(math.log10(10), math.log10(2000), num=nt, endpoint=True, base=10.0)

    for nti in range(nt):
        match nz:
            case 1:
                data_i = m_Fsol4D[:, :, 0, nti] # Pour la 2D
                fsol_2D[:, nti] = data_i[ind_mask]
            case _:
                data_i  = m_Fsol4D[:,:,:,nti]
                fsol_2D[:, nti] = data_i[ind_mask]
    #end

    mean_Spectrum = np.mean(fsol_2D, axis=0)
    std_Spectrum  = np.std(fsol_2D, axis=0)

    Total = np.sum(mean_Spectrum)
    mean_Spectrum = mean_Spectrum/Total
    std_Spectrum  = std_Spectrum/Total

    fsol_2D = fsol_2D/Total

    #ymax = 0.3
    #ymax = np.max(mean_Spectrum[T2s<=40])
    #ymax = np.max(fsol_2D[:,T2s<=40])
    # ------------------------------------------------------------------------------

    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(8,8), constrained_layout=True)
    # ax0 = axes.flatten()
    ax0.set_title('Showing results')

    ax0.plot(T2s, fsol_2D.T[:,:], alpha=0.2)
    #plt.plot(T2s, 1.5*mean_Spectrum, color='k')

    ax0.set_title('Spectra', fontsize=18)
    ax0.set_xlabel('T2', fontsize=18)
    ax0.set_ylabel('Intensity', fontsize=18)
    ax0.set_xscale('log')

    ax0.set_xlim(T2s[0], T2s[-1])
    ymax_total = np.max(fsol_2D)*1.05
    ymax = ymax_total/3.
    ax0.set_ylim(0, ymax_total)

    zoom = 1.5
    #zoom1 = ymax_total/(1.5*ymax)
    #zoom2 = np.log(T2s[-1])/(1.5*np.log(50.0))
    #zoom  = np.min([zoom1, zoom2])

    #if zoom2 > zoom1:
    #    zoom  = (1.5)*zoom
    #end if

    plt.axvline(x=40.0, ymin=0, color='k', linestyle='--')

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax0, zoom, loc=loc) # zoom-factor: 2.5, location: upper-left
    axins.plot(T2s, fsol_2D.T[:,:], alpha=0.5)
    #axins.plot(T2s, 2.*mean_Spectrum, color='k')

    x1, x2, y1, y2 = 10, 50, 0, ymax # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits
    #plt.axvline(x=40.0, color='k', linestyle='--', ymin=0, ymax=ymax)
    plt.axvline(x=40.0, color='k', linestyle='--', ymin=0)

    plt.yticks(visible=False)
    plt.yticks([])

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax0, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax0.set_yticks([])
    return fig
#end function

if __name__ == '__main__':
    sys.exit('Done')