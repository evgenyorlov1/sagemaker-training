import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# load data
iris = load_iris()
data = iris.data

# scale data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# apply PCA
pca_3d = PCA(n_components=3)
data = pca_3d.fit_transform(data)

# apply kmeans
kmeans = KMeans(n_clusters=3, random_state=77)
kmeans.fit(data)
cluster_points = kmeans.predict(data)
centroids = kmeans.cluster_centers_

# plot result
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'orange', 'purple']
markers = ['o', 's', 'D']

for cluster in np.unique(cluster_points):
    ax.scatter(
        data[cluster_points == cluster, 0],
        data[cluster_points == cluster, 1],
        data[cluster_points == cluster, 2],
        c=colors[cluster],
        marker=markers[cluster],
        label=f'Cluster {cluster+1}',
        edgecolor='black'
    )

for i, centroid in enumerate(centroids):
    ax.scatter(
        centroid[0], 
        centroid[1], 
        centroid[2],
        s=300,
        c='yellow',
        marker='*',
        edgecolor='black',
        label=f'Centroid {i+1}',
    )
    ax.text(
        centroid[0], 
        centroid[1], 
        centroid[2],
        f'C{i+1}',
        color='black',
        fontsize=12,
        ha='center',
        va='center'
    )

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

ax.legend()
ax.set_title('3D K-Means Clustering of Iris via PCA')
plt.show()
