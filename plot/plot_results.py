import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

import save_load_img as sl
import plot.plot_histo as histo
import plot.plot_brain_collage as brain
import plot.plot_spectrum as spectrum
import plot.plot_signal_dict as signal

def plot_results(m_PathToData, m_PathToMask, m_Slice, m_Method, m_PathToSaveData):
    
    params = {
    'text.latex.preamble': r'\usepackage{gensymb}',
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 20, # fontsize for x and y labels (was 10)
    'axes.titlesize': 20,
    'font.size': 20, # was 10
    'legend.fontsize': 20, # was 10
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': True,
    'font.family': 'serif',
    }

    mpl.rcParams.update(params)

    data = sl.load_niftiFloatImage(m_PathToData)
    FA = sl.load_niftiFloatImage(m_PathToSaveData + 'FA.nii.gz')
    fM = sl.load_niftiFloatImage(m_PathToSaveData + 'MWF.nii.gz')
    fIE = sl.load_niftiFloatImage(m_PathToSaveData + 'IEWF.nii.gz')
    fCSF = sl.load_niftiFloatImage(m_PathToSaveData + 'FWF.nii.gz')
    T2m = sl.load_niftiFloatImage(m_PathToSaveData + 'T2_M.nii.gz')
    T2IE = sl.load_niftiFloatImage(m_PathToSaveData + 'T2_IE.nii.gz')
    Ktotal = sl.load_niftiFloatImage(m_PathToSaveData + 'TWC.nii.gz')
    data = sl.load_niftiFloatImage(m_PathToSaveData + 'dataMasked.nii.gz')
    fsol_4D = sl.load_niftiFloatImage(m_PathToSaveData + 'fsol_4D.nii.gz')
    mask = sl.load_niftiIntImage(m_PathToMask)

    estimated_Sig = sl.load_niftiFloatImage(m_PathToSaveData + 'Est_Signal.nii.gz')
    # total_Signal = sl.load_niftiFloatImage(m_PathToSaveData + 'total_Signal.nii.gz')

    fig1 = brain.plot_result_brain(data,m_Slice,FA,fM,fIE,fCSF,T2m,T2IE,Ktotal)
    plt.savefig(m_PathToSaveData + 'MET2_'  + m_Method + '.png', dpi=600)

    fig2 = histo.plot_result_histo(m_Slice,FA,fM,T2m,T2IE)
    plt.savefig(m_PathToSaveData + 'MET2_histograms_' + m_Method + '.png', dpi=600)

    fig3 = histo.plot_mwfHisto(fM,data,m_Slice)
    plt.savefig(m_PathToSaveData + 'MWF_histo_' + m_Method + '.png',dpi=600)

    fig4 = spectrum.plot_mean_spectrum_slices(fsol_4D,mask,m_Slice)
    plt.savefig(m_PathToSaveData + m_Method + '_spectrums_new.png')
    plt.close('all')

    fig5 = signal.plot_signal_dict(data, mask, estimated_Sig, 0)
    plt.savefig(m_PathToSaveData + m_Method + '_signalDict.png')

    plt.close('all')
    return None

if __name__ == '__main__':    
    sys.exit('Done')