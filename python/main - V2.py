import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#def répertoire de travail et import des fonctions 
os.chdir('C:\\Users\\bourd\\Documents\\Alexis\\Mines\\2A\\Spike sorting\\python')
import filtre_passe_bande as flt
import seuillage as seuil
import ACP as acp
import clustering as clst 
import erreur as err
#%%

# Chargement du premier exemple
mat_contents = sio.loadmat('./SignauxSimulés/simulation_4.mat')
gt = sio.loadmat('./SignauxSimulés/ground_truth.mat')
# Extraction du signal et des instantes des spikes
xb = mat_contents['data']
idx = gt['spike_first_sample'][0][3] # premier échantillon de chaque spike
label = gt['spike_classes'][0][3] # classe de chaque spike
Fs= 24000 #fréquence d'echantillonnage
duree = 1  # affichage de la premiere seconde (sur 10 minutes)

#filtrage 
hf=3000
bf=300
ordre=2
signal_filtre = flt.bp_filter(Fs, hf, bf, ordre, xb)

#seuillage 
sigma=seuil.sigma_b(signal_filtre)
k=3.5
amplitude_seuil = k*sigma
print(f"seuil={amplitude_seuil}")

#récupération des spikes 
l=40 #moitié du nombre de pts souhaité, centrage autours du max
print(f"tps={2*1/Fs*l}") #temps d'échitillonage équivalent 
mat_spikes, peak_index =seuil.mat_spike2(signal_filtre, amplitude_seuil, l)



#%%

# Création de l'axe des temps (en secondes)
t = np.linspace(0, duree, int(Fs * duree), endpoint=False)
tspike = np.linspace(0,2*l*1/Fs,2*l)


# Tracé des graphiques

plt.figure(1,figsize=(10, 4))
plt.plot(t, xb[0][:int(Fs * duree)], color="b", label="Signal x(b)")
plt.title("Affichage du signal x(b) sur 1 minute")
plt.xlabel("Temps (secondes)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()



plt.figure(2,figsize=(10,4))
plt.plot(t, signal_filtre[0][:int(Fs * duree)], color="b", label="Signal x(b) filtré")
plt.title("Affichage du signal x(b) sur 1 minute")
plt.xlabel("Temps (secondes)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# représentation d'un spike 
plt.figure(3,figsize=(10,4))
plt.plot(tspike, mat_spikes[0], color="b", label="affichage d'un spike")
plt.xlabel("Temps (secondes)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Affichage
plt.tight_layout()
plt.show()

#%% création du Dataset

dim=len(mat_spikes)
dataset=acp.Dataset2(dim, mat_spikes) #création du dataset pour l'ACP


#%% ACP

pre_ACP=acp.pre_ACP(dataset)
dim=10
ACP=acp.ACP(dataset, dim)



#%% clustering

#méthode des kmoyennes coudée 
kmax=10
clst.elbow(ACP, kmax)

#méthode du silhouette score
sil=clst.silhouette(ACP,kmax)

#méthode BIC
clst.BIC(ACP, kmax)

#nombre de cluster retenu 
nclust=6

Kmoy_final,_=clst.Kmoyennes(ACP, nclust)

#%% calcul de l'erreur

# sur la détection

erreur_detection, faux_neg, faux_pos, spike_detected=err.erreur_detect(idx[0], peak_index) 

print(f"l'erreur de détection (sans prise en compte des faux positifs) est : {round(erreur_detection*100,3)}%, le nombre de faux positifs est : {faux_pos}, le nombre de faux négatifs est : {faux_neg}")


# sur la classification 

clust, Tclust=err.regroupement_cluster(spike_detected, idx[0], peak_index, label[0], Kmoy_final)

err_clust, correspondance=err.erreur_cluster(Tclust, clust)

print(f"l'erreur de classification vaut, pour chaque classe : {err_clust} (exprimée en pourcentage)")