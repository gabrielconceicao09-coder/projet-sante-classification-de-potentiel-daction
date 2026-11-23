from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import numpy as np

def Kmoyennes (coordonnees, nb_clusters):                                       #algo des k moyennes
    kmeans = KMeans(n_clusters=nb_clusters, n_init="auto")
    kmeans.fit(coordonnees)
    
    return(kmeans.labels_, kmeans.inertia_)

def silhouette (pts, kmax):
    silhouette_scores = []
    K_range = range(2, kmax)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pts)
        score = silhouette_score(pts, labels)
        silhouette_scores.append(score)

    plt.figure()
    plt.grid()
    plt.xticks(K_range)
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Score de silhouette')
    plt.title('Méthode de la silhouette')
    plt.show()
    
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"Meilleur k d'après silouhette = {best_k}")
    
    return silhouette_scores
    
def elbow (pts, kmax) :
    inertie=[]  
    ech=[]  
    i=1                      #attention inertie[0] correspond à k=0, impossible d'ou le choix d'une convention
    for k in range(1,kmax):               #on cherche le nb de cluster optimal
        Kmoy=Kmoyennes(pts, k) 
        inertie.append(Kmoy[1])
        ech.append(i) 
        i+=1 

    plt.figure(5,figsize=(10, 4))
    plt.plot(ech, inertie, color="b",marker='o', linestyle='-')
    plt.title("inertie en fct de nb_cluster")
    plt.xlabel("nb_clusters")
    plt.ylabel("inertie")
    plt.grid(True)



def BIC(pts, kmax):
    
    ks = range(1, kmax)
    bics = []
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=10, random_state=42).fit(pts)
        bics.append(gmm.bic(pts))

    plt.plot(ks, bics, "o-")
    plt.xlabel("k"); plt.ylabel("BIC (à minimiser)")
    plt.show()

    k_opt = ks[int(np.argmin(bics))]
    print("k optimal d'après méthode BIC' :", k_opt)