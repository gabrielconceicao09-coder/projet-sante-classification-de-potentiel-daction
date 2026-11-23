import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, convolve, find_peaks
import scipy.io as sio

# Chargement du premier exemple
mat_contents = sio.loadmat('./SignauxSimulés/simulation_1.mat')
gt = sio.loadmat('./SignauxSimulés/ground_truth.mat')
# Extraction du signal et des instantes des spikes
xb = mat_contents['data']
idx = gt['spike_first_sample'][0][0] # premier échantillon de chaque spike
label = gt['spike_classes'][0][0] # classe de chaque spike
Fs= 24000 #fréquence d'echantillonnage
duree = 1  # affichage de la premiere seconde (sur 10 minutes)


# Création de l'axe des temps (en secondes)
t = np.linspace(0, duree, int(Fs * duree), endpoint=False)
# Tracé du graphique
plt.figure(figsize=(10, 4))
plt.plot(t, xb[0][:int(Fs * duree)], color="b", label="Signal x(b)")
plt.plot((idx[idx<duree*Fs])/Fs, xb[0][idx[idx<duree*Fs]], 'r*', label="1st spk sample") # marque le debut de chaque spike

# Personnalisation des axes et du titre
plt.title("Affichage du signal x(b) sur 1 minute")
plt.xlabel("Temps (secondes)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Affichage
plt.tight_layout()
plt.show()


