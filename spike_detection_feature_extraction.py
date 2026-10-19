import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssig
import scipy.io as sio
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

def apply_highpass(signal, order, fc, fs_):
    hpf_sos = ssig.butter(order, fc, 'highpass', output='sos', fs=fs_)
    return ssig.sosfilt(hpf_sos, signal)

def seuil_DJ(signal, k):
    return k*(np.median(np.abs(signal-np.median(signal))))/0.6745

def make_spike_col(signal_filt, spike_idx, centered_time_window, fs):
    i_window = int(fs*centered_time_window/2)
    spike_col =[]
    spike_left_bounds = []
    for i in spike_idx:
        spike_col.append(signal_filt[i-i_window:i+i_window])
        spike_left_bounds.append(i-i_window)
    return spike_col, spike_left_bounds

def extract_spike_caracteristics(spike_col, centered_time_window, fs):
    max_peak_amplitude = []
    max_peak_prominence = []
    max_spike_rel_width = []
    spike_energy_ = []
    spike_refract_height_ = []
    spike_trigger_height_ = []
    spike_halfpeak_width = []
    spike_trigger_to_peak_inclination = []
    spike_peak_to_refractory_inclination = []

    for spike in spike_col:
        spike_peaks, spike_peaks_properties = ssig.find_peaks(spike)
        spike_minima = ssig.argrelmin(spike) 
        spike_prominences, spike_lbases, spike_rbases = ssig.peak_prominences(spike, spike_peaks)
        spike_widths, spike_width_heights, spike_l_ips, spike_r_ips = ssig.peak_widths(spike, spike_peaks, prominence_data=[spike_prominences, spike_lbases, spike_rbases])
        spike_energy = 0
        for x in spike:
            spike_energy += x**2 
        spike_mainpeak_id = int(centered_time_window*fs/2) #centré
        for i in range(len(spike_minima[0])):
            index = spike_minima[0][i]
            if index > spike_mainpeak_id:
                spike_refract_height = spike[index] #Amplitudes des minima après et avant le pic principal du spike
                spike_trigger_height = spike[spike_minima[0][i-1]]
                left_id_dist = spike_mainpeak_id-spike_minima[0][i-1]
                right_id_dist = index - spike_mainpeak_id
                halfpeak_width = (left_id_dist+right_id_dist)/fs
                trigg_peak_incl = (spike[spike_mainpeak_id]-spike[spike_minima[0][i-1]])/((left_id_dist)/fs)
                peak_refr_incl = (spike[index]-spike[spike_mainpeak_id])/((right_id_dist)/fs)
                break
        max_peak_amplitude.append(spike[spike_mainpeak_id])
        max_peak_prominence.append(np.max(spike_prominences))
        max_spike_rel_width.append(np.max(spike_widths))
        spike_energy_.append(spike_energy)
        spike_refract_height_.append(spike_refract_height)
        spike_trigger_height_.append(spike_trigger_height)
        spike_halfpeak_width.append(halfpeak_width)
        spike_trigger_to_peak_inclination.append(trigg_peak_incl)
        spike_peak_to_refractory_inclination.append(peak_refr_incl)
    return max_peak_amplitude, max_peak_prominence, max_spike_rel_width, spike_energy_, spike_refract_height_, spike_trigger_height_, spike_halfpeak_width, spike_trigger_to_peak_inclination, spike_peak_to_refractory_inclination


mat_contents = sio.loadmat("/home/gabriel/Documents/Projet 2A - Classification de potentiel d'action/transfer_13590691_files_765200d6/SignauxSimulés/simulation_4.mat")
gt = sio.loadmat("/home/gabriel/Documents/Projet 2A - Classification de potentiel d'action/transfer_13590691_files_765200d6/SignauxSimulés/ground_truth.mat")
# Extraction du signal et des instantes des spikes
signal = mat_contents["data"][0]
idx = gt['spike_first_sample'][0][0] # premier échantillon de chaque spike
label = gt['spike_classes'][0][0] # classe de chaque spike
Fs= 24000 #fréquence d'echantillonnage
duree = 10  # affichage de la premiere seconde (sur 10 minutes)

signal_cut = signal[:int(Fs*duree)]
signal_filt = apply_highpass(signal_cut, 2, 300, Fs)

seuil = seuil_DJ(signal_filt, k=5)
spike_indexes, peak_prop = ssig.find_peaks(signal_filt, height=seuil)

mat = {}
mat["signal"], mat["spikes_left_bounds"] = make_spike_col(signal_filt, spike_indexes, 0.002, Fs)
mat["peak_og_index"] = spike_indexes
mat["peak_amplitude"], mat["peak_prominence"], mat["rel_width"], mat["energy"], mat["refract_height"], mat["trigger_height"], mat["halfpeak_width"], mat["trigger_inclination"], mat["refractory_inclination"] = extract_spike_caracteristics(mat["signal"], 0.002, Fs)

df = pd.DataFrame(mat)

X = df.drop(columns=["signal",'peak_og_index', 'peak_prominence', 'rel_width'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

spike_acp = PCA(n_components=3)
spike_projection = spike_acp.fit_transform(X_scaled)

acp_df = pd.DataFrame(spike_projection, columns=["DIM1", "DIM2", "DIM3"])
print(f"Explained variance ratio per axis: {spike_acp.explained_variance_ratio_}")

fig0 = plt.figure()
t = np.linspace(0, len(mat["signal"][13])/Fs, len(mat["signal"][13]))
plt.plot(t, mat["signal"][13])

print(f"Peak amplitude: {mat["peak_amplitude"][13]}")
print(f"Peak prominence: {mat["peak_prominence"][13]}")
print(f"Halfpeak width: {mat["halfpeak_width"][13]}")
print(f"Spike energy: {mat["energy"][13]}")
print(f"Refractory period height: {mat["refract_height"][13]}")
print(f"Trigger height: {mat["trigger_height"][13]}")
print(f"Trigger Inclination: {mat["trigger_inclination"][13]}")
print(f"Refractory inclination: {mat['refractory_inclination'][13]}")

fig = plt.figure(figsize=(15,6))

ax = fig.add_subplot(projection='3d')
ax.scatter3D(acp_df['DIM1'], acp_df['DIM2'], acp_df['DIM3'])
ax.set_title("Projection des caracteristiques des spikes")
ax.set_xlabel("DIM1")
ax.set_ylabel("DIM2")
ax.set_zlabel("DIM3")

fig1 = plt.figure(figsize=(6,13))

ax1 = fig1.add_subplot(212)
ax1.scatter(acp_df['DIM1'], acp_df['DIM2'])
ax1.set_xlabel('DIM1')
ax1.set_ylabel('DIM2')

ax2 = fig1.add_subplot(211)
ax2.scatter(acp_df['DIM1'], acp_df['DIM3'])
ax2.set_xlabel('DIM1')
ax2.set_ylabel('DIM3')

plt.show()
