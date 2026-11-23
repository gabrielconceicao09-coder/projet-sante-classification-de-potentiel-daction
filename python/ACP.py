import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def Dataset(names,mat):
    d=len(names)
    data={}
    for i in range (0,d):
        data[names[i]]=mat[i]  #names liste de str 
    table=pd.DataFrame(data)
    return(table)

def Dataset2(dim,mat):
    data=[]
    for i in range (0,dim):
        data.append(np.array(mat[i]))   
    table=pd.DataFrame(data)
    return(table)


def pre_ACP (table_caracteristiques):
    
    table_normee = StandardScaler().fit_transform(table_caracteristiques)

    pca = PCA()
    res_acp = pca.fit_transform(table_normee)

    eigval = pca.explained_variance_      # Valeurs propres
    variance_ratio = pca.explained_variance_ratio_ * 100 # Pourcentage de variance
    varcumule=[]
    k=0
    for i in range (0,len(variance_ratio)) :
        k+=variance_ratio[i]
        varcumule.append(k)
    
    print("Valeurs propres :", eigval)
    print("Variance expliquée par axe (%) :", variance_ratio)
    print("Variance expliquée cumulée par axe (%) :", varcumule)
    
    return(res_acp)



def ACP (table_caracteristiques, dim):
    
    table_normee = StandardScaler().fit_transform(table_caracteristiques)
    pca = PCA(n_components=dim)
    res_acp = pca.fit_transform(table_normee)
    
    return(res_acp)