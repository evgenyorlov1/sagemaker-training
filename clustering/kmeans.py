import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# load data
iris = load_iris()
data = iris.data[:, [0, 2, 3]]
labels = iris.target
feature_names = ['sepal length', 'petal length', 'petal width']

# normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# k-means clustering
kmeans = KMeans(n_clusters=3, random_state=77)
kmeans.fit(scaled_data)
clusters = kmeans.labels_

# plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'orange', 'purple']
markers = ['o', 's', 'D']

for cluster in np.unique(clusters):
    ax.scatter(
        scaled_data[clusters == cluster, 0],
        scaled_data[clusters == cluster, 1],
        scaled_data[clusters == cluster, 2],
        c=colors[cluster],
        marker=markers[cluster],
        label=f'Cluster {cluster+1}',
        edgecolor='black',
    )

centroids = kmeans.cluster_centers_
for i, centroid in enumerate(centroids):
    ax.scatter(
        centroid[0],
        centroid[1],
        centroid[2],
        s=300,
        c='yellow',
        marker='*',
        edgecolor='black',
        linewidth=2,
        label=f'Centroid {i+1}'
    )

ax.set_xlabel('Normalized Sepal length')
ax.set_ylabel('Normalized Petal length')
ax.set_zlabel('Normalized Petal width')
ax.legend()
ax.set_title('K-Means clustering of iris data')
plt.show()
