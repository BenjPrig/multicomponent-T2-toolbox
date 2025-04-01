import sys
import matplotlib.pyplot as plt


from plot.tool_plot import colorbar

def plot_result_brain(m_Data,m_Slice,m_FA,m_fM,m_fIE,m_fCSF,m_T2m,m_T2IE,m_Ktotal,m_DataType='invivo'):

    fig = plt.figure('Showing all results', figsize=(11,10), constrained_layout=True)

    plt.subplot(3,3,1).set_axis_off()
    im0 = plt.imshow(m_Data[:,:,m_Slice,0].T, cmap='gray', origin='upper')
    plt.title('Signal(TE=10ls)')
    colorbar(im0)
    
    plt.subplot(3, 3, 2).set_axis_off()
    im1 = plt.imshow(m_FA[:,:,m_Slice].T, cmap='plasma', origin='upper', clim=(90,180))
    plt.title('Flip Angle (degrees)')
    colorbar(im1)

    plt.subplot(3, 3, 4).set_axis_off()
    #im1 = plt.imshow(fM[:,:,Slice].T, cmap='gray', origin='lower', clim=(0,0.25))
    im1 = plt.imshow(m_fM[:,:,m_Slice].T, cmap='afmhot', origin='upper', clim=(0,0.25))
    plt.title('Myelin Water Fraction')
    colorbar(im1)

    plt.subplot(3, 3, 5).set_axis_off()
    #im2 = plt.imshow(fIE[:,:,Slice].T, cmap='gray', origin='lower', clim=(0,1))
    im2 = plt.imshow(m_fIE[:,:,m_Slice].T, cmap='magma', origin='upper', clim=(0,1))
    plt.title('Intra/Extra Water Fraction')
    colorbar(im2)

    plt.subplot(3, 3, 6).set_axis_off()
    im3 = plt.imshow(m_fCSF[:,:,m_Slice].T, cmap='hot', origin='upper', clim=(0,1))
    plt.title('Free Water Fraction')
    colorbar(im3)

    match m_DataType:
        case 'invivo':
            plt.subplot(3, 3, 7).set_axis_off()
            im4 = plt.imshow(m_T2m[:,:,m_Slice].T, cmap='gnuplot2', origin='upper', clim=(9,40))
            plt.title('T2-Myelin (ms)')
            colorbar(im4)

            plt.subplot(3, 3, 8).set_axis_off()
            #im5 = plt.imshow(T2IE[:,:,Slice].T, origin='lower', clim=(50,100))
            im5 = plt.imshow(m_T2IE[:,:,m_Slice].T, cmap='gnuplot2', origin='upper', clim=(50,90))
            plt.title('T2-Intra/Extra (ms)')
            colorbar(im5)
        case 'exvivo':
            plt.subplot(3, 3, 7).set_axis_off()
            im4 = plt.imshow(m_T2m[:,:,m_Slice].T, origin='upper', clim=(5,25))
            plt.title('T2-Myelin (ms)')
            colorbar(im4)

            plt.subplot(3, 3, 8).set_axis_off()
            im5 = plt.imshow(m_T2IE[:,:,m_Slice].T, origin='upper', clim=(30,60))
            plt.title('T2-Intra/Extra (ms)')
            colorbar(im5)

    plt.subplot(3, 3, 9).set_axis_off()
    im6 = plt.imshow(m_Ktotal[:,:,m_Slice].T, cmap='gray', origin='upper')
    plt.title('Total Water Content')
    colorbar(im6)

    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.07, wspace=-0.3)

    return fig

if __name__ == '__main__':
    sys.exit('Done')