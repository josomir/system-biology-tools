from numpy import split
from numpy.random import sample
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import cluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pdb
import scipy
from sklearn.cluster import OPTICS


file = './CosmicExpressionsOvary.csv'
df = pd.read_csv(file, sep=',')
cell_lines = df['SAMPLE_NAME'].unique().tolist()
genes = df['GENE_NAME'].unique().tolist()
sample_gene_matrix = df.pivot_table(index='SAMPLE_NAME', values='Z_SCORE', columns='GENE_NAME', aggfunc='mean')
sample_gene_matrix.to_csv('SampleGeneMatrix.csv')
print ('Gene-Sample Matrix size: ', sample_gene_matrix.shape)


from sklearn.cluster import DBSCAN
from sklearn import metrics

#db = DBSCAN(eps=150, min_samples=2).fit(sample_gene_matrix)
db = KMeans(n_clusters=10).fit(sample_gene_matrix)

labels = db.labels_
print (labels)
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(sample_gene_matrix, labels))


# Dimensionality reduction using PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(sample_gene_matrix.values)

pca_matrix = pd.DataFrame()
pca_matrix['sample-name'] = sample_gene_matrix.index
pca_matrix['pca-one'] = pca_result[:,0]
pca_matrix['pca-two'] = pca_result[:,1] 
pca_matrix['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#print (pca_matrix)
#print (sample_gene_matrix.columns)

# Plot result
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors

#fig, ax = plt.subplots()
#fig = plt.figure(figsize=(16,10))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(
    xs=pca_matrix["pca-one"], 
    ys=pca_matrix["pca-two"], 
    zs=pca_matrix["pca-three"], 
    cmap='tab10',
    label=pca_matrix['sample-name']
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')

# Displaying label on mouse hover
mplcursors.cursor().connect(
    "add", lambda sel: sel.annotation.set_text(pca_matrix['sample-name'][sel.target.index]))

#import pickle
#pickle.dump(ax, open('FigureObject.fig.pickle', 'wb'))
#pdb.set_trace()
import mpld3
#fig = plt.gcf()
#print (mpld3.fig_to_html(fig))

#plt.show()


def calculatePCAsimilartiyMatrix(pca_matrix):
    ary = scipy.spatial.distance.cdist(pca_matrix[["pca-one", "pca-two", "pca-three"]], 
                                        pca_matrix[["pca-one", "pca-two", "pca-three"]], 
                                        metric='euclidean')
    similarity_matrix = pd.DataFrame(ary, index= pca_matrix['sample-name'], columns=pca_matrix['sample-name'])
    similarity_matrix.to_csv('cell_line_similarity_matrix.csv')
    #print ("Similarity matrix: \n")
    #print (similarity_matrix)
    return similarity_matrix
    

def findClosestNeighbours(similarity_matrix):
    similarity_matrix = similarity_matrix.replace({'0':np.nan, 0:np.nan})
    #print (similarity_matrix.min(),  similarity_matrix.idxmin())
    neighbours = pd.DataFrame( data ={ 'similarity_score':similarity_matrix.min(),
                                        'sample_name': similarity_matrix.idxmin()},
                                index= similarity_matrix.index.values)
    

    # look for lowest values in matrix and print their indexes (sample names)
    print ("Closest neighbours: ")
    print (neighbours)
    return neighbours

def clustering(pca_matrix):
    data = pca_matrix[["pca-one", "pca-two", "pca-three"]]
    clust = OPTICS(min_samples=2, xi=.05, min_cluster_size=.05)
    clust.fit(data)
    clusters = pd.DataFrame(data = {'cluster-labels': clust.labels_}, index = similarity_matrix.index.values)
    print (clusters.sort_values('cluster-labels'))
    return clusters


    



similarity_matrix = calculatePCAsimilartiyMatrix(pca_matrix)
findClosestNeighbours(similarity_matrix)
clustering(pca_matrix)


