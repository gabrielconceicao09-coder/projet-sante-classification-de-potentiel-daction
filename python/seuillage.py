#calcul de sigma(b)
import numpy as np
from scipy.signal import find_peaks

def sigma_b (signal_filtre):
    med=np.median(signal_filtre)
    signal_filtre_centre_pos=[np.abs(signal_filtre[i]-med) for i in range (0,len(signal_filtre))]
    sigma_b=np.median(signal_filtre_centre_pos)/0.6745
    return(sigma_b)

def mat_spike (signal, seuil, l):       #approche naïve
    mat=[]
    d=len(signal[0])
    i=0
    while i<(d-1):
        spike=[]
        
        if signal[0][i] > seuil :
            for j in range(0,l-1):      #on remplit spike[] avec les valeurs de du spike détecté
                spike.append(signal[0][i+j])
            mat.append(spike) #on ajoute une ligne à la matrice
            i+=l
        
        else :      #sinon on passe au candidat suivant
            i+=1
            
    return(mat)  
        
    
def mat_spike2 (signal, seuil, l):
    
    peaks_index, properties = find_peaks(signal[0], height=seuil)  #indices des max
    d=len(peaks_index)
    mat=[]
    
    for j in range(0, d):
        k=peaks_index[j]
        spike=[]
        for i in range(0,2*l):
            spike.append(signal[0][k+(i-l)])
        mat.append(spike)
    
    return(mat,peaks_index)            