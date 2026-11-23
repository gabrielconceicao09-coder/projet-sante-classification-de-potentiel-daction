import numpy as np


def erreur_detect(Tspike_first_sample, spike_peak):  #T pr toutes les grandeurs réelles 
    
    faux_neg=0
    Tnb_spike=len(Tspike_first_sample)
    nb_spike=len(spike_peak)
    
    spike_detected={}  #dict qui fait correspondre l'indice du spike du fichier ground truth à l'indice du spike dans les spikes détectés
    detect=0
    tolerance=36 #36 pts soit 1.5ms
    
    
    for i in range (Tnb_spike):
        
        booldetect=0
        
        for j in range(nb_spike):
            
            if np.abs(Tspike_first_sample[i]-spike_peak[j])<=tolerance : #on vérifie que le spike soit détecté
                detect+=1
                booldetect=1
                spike_detected[i]=j
                break
       
        if booldetect==0 : 
            faux_neg+=1
            spike_detected[i]=-1     #spike_detected[i]=-1 si le spike n'est pas détecté 
            
    erreur_detection=1-detect/Tnb_spike
    
    faux_pos=nb_spike-Tnb_spike+faux_neg
    
    return(erreur_detection, faux_neg, faux_pos, spike_detected)



def regroupement_cluster(spike_detected, Tspike_first_sample, spike_peak, Tlabels, labels): 
    
    Td=max(Tlabels+1)            #construction de deux dictionnaires clés=cluster, attribus=liste des spikes, sans prise en compte des faux positifs ni des faux négatifs
    d=max(labels+1)
    l=len(spike_detected)
    
    Tclusters = [[] for _ in range(int(Td))]
    clusters = [[] for _ in range(int(d))]   
    
    for i in range(l):
        
        if spike_detected[i]!=-1:           #si le spike n'est pas un faux négatif 
             
            Tclusters[Tlabels[i]].append(Tspike_first_sample[i])                #ajout du pts dans la liste du vrai label correspondant 
            
            clusters[labels[spike_detected[i]]].append(Tspike_first_sample[i])      #ajout du pts dans la liste du label calculé par les Kmoy correspondant

        
    return (clusters, Tclusters)



def proportion_commune(Tclust, clust):
    sb = set(clust)
    return 100 * sum(x in sb for x in Tclust) / len(Tclust)


def erreur_cluster (Tclust, clust): #on veut comparer chaque cluster entre l'expérience et le ground truth, mais ils n'ont pas forcément le même indice 
    L=len(Tclust)
    l=len(clust)
    correspondance=[-1 for _ in range(L)]  #liste qui garde en mémoire les clusters déjà attribués 
    err=[-1 for _ in range(l)]
    
    for i in range(l): #on parcourt les cluster expérimentaux
        ressemblance=[-1 for _ in range(L)] # on garde la ressemblance entre le cluster expérimental i et tous les vrais clusters puis on détermine le max 
        
        for j in range(L):
            if correspondance[j]==-1:  #si le cluster j n'est pas déjà attribué
                ressemblance[j]=proportion_commune(Tclust[j], clust[i])

       
        err[i]=max(ressemblance)
        correspondance[np.argmax(ressemblance)]=i
    
    for i in range(l) : #on effectue 100-proportion_commune pour avoir la bonne déf de l'erreur
        if err[i]!=-1:
            err[i]=100-err[i]
    
    return(err, correspondance)
    