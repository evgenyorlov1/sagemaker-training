import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# load iris data
iris = datasets.load_iris()
x = iris.data[:, :2]
y = (iris.target == 2).astype(int)

# split data into train and test chunks
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2, 
    random_state=44
)

# train model
model = LogisticRegression()
model.fit(x_train, y_train)

# regression function
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
z = z.reshape(xx.shape)

# plot
contour = plt.contourf(
    xx, 
    yy, 
    z, 
    alpha=0.8,
    cmap=plt.cm.Greys,
    levels=np.linspace(0, 1, 11)
)
plt.scatter(x[y==0, 0], x[y==0, 1], c="white", edgecolors="k", marker="o", s=40, label="Not Iris Virginica")
plt.scatter(x[y==1, 0], x[y==1, 1], c="white", edgecolors="k", marker="s", s=40, label="Iris Virginica")
plt.colorbar(contour, label="Probability of Iris Virginica")

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Logistic Regression sigmoid function visualization")

plt.legend()
plt.show()