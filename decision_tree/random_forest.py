from sklearn import tree
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
import numpy as np
import matplotlib.pyplot as plt


# prepare data
x, y = make_moons(n_samples=500, noise=0.25, random_state=7)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=17, stratify=y
)

# train model
n_estimators = 100
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, random_state=77)
model.fit(x_train, y_train)

# evaluate model
y_predicted = model.predict(x_test)
y_probability = model.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, y_predicted)
print(f"accuracy: {accuracy}")

print("classification report:")
print(classification_report(y_test, y_predicted))

print("confusion matrix:")
print(confusion_matrix(y_test, y_predicted))

auc = roc_auc_score(y_test, y_probability)
print(f"ROC AUC: {auc:.3f}")

# plot
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict probabilities on the grid
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, probs, levels=20)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolor="k", s=40)
plt.title("Random Forest decision probability (class=1)")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.tight_layout()
plt.show()

importances = model.feature_importances_
feat_names = [f"f{i}" for i in range(x.shape[1])]

plt.figure(figsize=(6, 4))
plt.bar(np.arange(len(importances)), importances)
plt.xticks(np.arange(len(importances)), feat_names)
plt.title("Feature importances (Random Forest)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_probability)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1])
plt.title("ROC curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
