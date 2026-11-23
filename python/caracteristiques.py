#on cherche différentes caractéristiques des spikes qui pourrait nous servir à faire une ACP après 
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences

#amplitude 

def amplitude_spikes (signal, seuil):
    peaks_index, properties = find_peaks(signal[0], height=seuil)
    return (properties["peak_heights"])

#largeur 

def largeur_spikes (signal, peaks_index):
    largeur=peak_widths(signal[0], peaks_index)[0]
    return (largeur)

#amplitude crête à crête 

def amp_cac (signal, peaks_index):
    amp=peak_prominences(signal[0], peaks_index)[0]
    return(amp)

#énergie

def energie_spike(mat):
    d=len(mat)
    l=len(mat[0])
    energie=[]
    for i in range(0,d):
        enr=0
        for j in range (0,l):
            enr+=mat[i][j]**2
        energie.append(enr)
    return(energie)

#pente partie descendante

def slopes_spike (mat1, Fs): #k le nb de points après le pic entre lesquels on calcule la pente 
    mat=np.array(mat1)
    d=len(mat)
    slopedown=[]
    slopeup=[]
    
    for i in range (0,d):   #parcours de tous les pikes 
        spike=mat[i]
        l=len(mat[0])//2
        spike_up=spike[:l]
        spike_down=spike[l:]
        
        max_up=np.argmax(spike_up)
        max_down=np.argmax(spike_down)
        
        min_up=np.argmin(spike_up)
        min_down=np.argmin(spike_down)
        
        slopeup.append((spike_up[max_up]-spike_up[min_up])/((max_up-min_up)*1/Fs))
        slopedown.append((spike_down[min_down]-spike_down[max_down])/((min_down-max_down)*1/Fs))
        
    return(slopeup, slopedown)