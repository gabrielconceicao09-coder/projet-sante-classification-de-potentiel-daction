import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssig
import scipy.io as sio
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import pywt
import statistics as stat

def apply_highpass(signal, order, fc, fs_):
    hpf_sos = ssig.butter(order, fc, 'highpass', output='sos', fs=fs_)
    return ssig.sosfilt(hpf_sos, signal)

def seuil_DJ(signal, k):
    return k*(np.median(np.abs(signal-np.median(signal))))/0.6745

def make_spike_col(signal_filt, spike_idx, centered_time_window, fs): #Extraire les spikes du signal à partir des pics détectés
    i_window = int(fs*centered_time_window/2)
    spike_col =[]
    spike_left_bounds = []
    for i in spike_idx:
        spike_col.append(signal_filt[i-i_window:i+i_window])
        spike_left_bounds.append(i-i_window)
    return spike_col, spike_left_bounds

def wavelet_decomp_df(signal_mat, wavelet_ = 'db4'): #signal_mat liste de signaux, wavelet le type d'ondelettes
    coeff_array = []
    coeff_display_array = []
    for signal in signal_mat:
        decomp_result = pywt.wavedec(signal, wavelet=wavelet_)
        coeff_list = []
        for C in decomp_result:
            coeff_list = np.append(coeff_list, C)
        coeff_array.append(coeff_list)
        coeff_display_array.append(decomp_result)
    return pd.DataFrame(coeff_array).add_prefix("C"), coeff_display_array

def sort_cols_var(df): #Ordonner les colonnes en fonction de la variance. sorted_cols est une liste des noms des colonnes ordonnées
    #columns = list(df.columns)
    sorted_cols = []
    sorted_cols_var = []
    '''for col in columns:
        var = stat.variance(df[col])
        var_len = len(sorted_cols_var)
        if var_len == 0:
            sorted_cols_var.append(var)
            sorted_cols.append(col)
        elif var_len >0:
            for i in range(var_len):
                if var > sorted_cols_var[i]:
                    sorted_cols_var.insert(i, var)
                    sorted_cols.insert(i, col)
                    break
                elif var == sorted_cols_var[i]:
                    sorted_cols_var.insert(i, var)
                    sorted_cols.insert(i, col)
                    break
                elif i == var_len-1:
                    sorted_cols_var.append(var)
                    sorted_cols.append(col)
                    break'''
    var_series = df.var(axis=0,ddof=1).sort_values(ascending=False)
    sorted_cols = var_series.index.tolist()
    sorted_cols_var = var_series.values.tolist()
    return sorted_cols, sorted_cols_var


mat_contents = sio.loadmat("/home/gabriel/Documents/Projet 2A - Classification de potentiel d'action/transfer_13590691_files_765200d6/SignauxSimulés/simulation_4.mat")
gt = sio.loadmat("/home/gabriel/Documents/Projet 2A - Classification de potentiel d'action/transfer_13590691_files_765200d6/SignauxSimulés/ground_truth.mat")
# Extraction du signal et des instantes des spikes
signal = mat_contents["data"][0]
idx = gt['spike_first_sample'][0][0] # premier échantillon de chaque spike
label = gt['spike_classes'][0][0] # classe de chaque spike
Fs= 24000 #fréquence d'echantillonnage
duree = 100  # affichage de la premiere seconde (sur 10 minutes)

signal_cut = signal[:int(Fs*duree)]
signal_filt = apply_highpass(signal_cut, 2, 300, Fs) #filtrage


seuil = seuil_DJ(signal_filt, k=5)
spike_indexes, peak_prop = ssig.find_peaks(signal_filt, height=seuil)

mat = {}
mat["signal"], mat["spikes_left_bounds"] = make_spike_col(signal_filt, spike_indexes, 0.003, Fs) 
mat["peak_og_index"] = spike_indexes


coeff_df, coeff_display = wavelet_decomp_df(mat["signal"])
print(coeff_df)

sorted_cols, sorted_cols_var = sort_cols_var(coeff_df)

wavelet_df = pd.DataFrame()
ncoeffs = 10
for i in range(ncoeffs):
    col = sorted_cols[i]
    wavelet_df = pd.concat([wavelet_df, coeff_df[col]], axis = 1)

print(wavelet_df)



t = np.linspace(0, len(mat["signal"][0])/Fs, len(mat["signal"][0]))
def plot_wavelet_decomp(og_signal, time, coeffs, n_coef_plots): #og_signal le signal décomposé, time l'array d'instants pour le signal, coeffs les coéfficients resultants, n_coef_plots le nombre de coéfficients de détail qu'on veut visualiser
    level = len(coeffs)-1
    if n_coef_plots <= level:
        fig, axes = plt.subplots(n_coef_plots + 2, 1, figsize=(10,10), sharex=False)
        axes[0].plot(time, og_signal, "k")
        axes[0].set_title("Spike")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(coeffs[0], 'b')
        axes[1].set_title(f"Coéfficients d'Approximation (cA{level})")
        axes[1].set_ylabel("Amplitude")

        for i in range(1, n_coef_plots+1):
            axes[i+1].plot(coeffs[i], "r")
            axes[i+1].set_title(f"Coéfficients de Détail (cD{n_coef_plots-i+1})")
            axes[i+1].set_ylabel("Amplitude")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Nombre de coéfficients demandé ({n_coef_plots}) est plus grand que le nombre de niveaux de la décomposition ({level})")

plot_wavelet_decomp(mat["signal"][0], t, coeff_display[0], 3)