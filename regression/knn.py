import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap


# load data
iris = load_iris()
x = iris.data[:, :2]
y = iris.target

# split data into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

# tran knn
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

# plot result
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1), 
    np.arange(y_min, y_max, 0.1)
)

z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, z, alpha=0.3, cmap=ListedColormap(('gray', 'lightgray', 'darkgray')))

training_markers = ['o', 's', '^']
test_markers = ['o', 's', '^']
colors = ['black', 'black', 'black']
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


# plot training data points
for i, (color, marker) in enumerate(zip(colors, training_markers)):
    mask = y_train == i
    plt.scatter(
        x_train[mask, 0], 
        x_train[mask, 1], 
        c=color, 
        label=f'Training {labels[i]}', 
        marker=marker, 
        edgecolor='k'
    )

# plot testing data
for i, marker in enumerate(test_markers):
    mask = y_test == i
    plt.scatter(
        x_test[mask, 0], 
        x_test[mask, 1], 
        facecolors='none',
        label=f'Test {labels[i]}', 
        marker=marker, 
        edgecolor='k'
    )

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title(f'k-NN classification (k={k})')
plt.legend()
plt.show()
