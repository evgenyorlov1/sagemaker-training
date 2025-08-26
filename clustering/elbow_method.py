import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# load data
iris = load_iris()
data = iris.data

# normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# calculate wcss
wcss = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=15)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# plot wcss
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), wcss, marker='o')
plt.title('Elbow Method for Determining Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()