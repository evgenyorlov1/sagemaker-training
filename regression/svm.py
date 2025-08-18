import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC


# Generate toy dataset (two interleaving half circles - not linearly separable)
X, y = datasets.make_moons(n_samples=100, noise=0.15, random_state=42)

# Train linear SVM
clf_linear = SVC(kernel="linear", C=1.0)
clf_linear.fit(X, y)

# Train nonlinear SVM (RBF kernel)
clf_rbf = SVC(kernel="rbf", C=1.0, gamma=1.0)
clf_rbf.fit(X, y)

# Helper function to plot decision boundaries
def plot_svm(clf, X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=40, edgecolors="k")

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                         np.linspace(ylim[0], ylim[1], 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Decision boundary + margins
    ax.contour(xx, yy, Z, colors="k", levels=[-1, 0, 1],
               linestyles=["--", "-", "--"])

    # Support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=120, facecolors="none", edgecolors="k")

    plt.title(title)
    plt.show()

# Plot results
plot_svm(clf_linear, X, y, "Linear SVM (fails on nonlinear data)")
plot_svm(clf_rbf, X, y, "Nonlinear SVM with RBF kernel (fits curved boundary)")
