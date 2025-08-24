import numpy as np
import matplotlib.pyplot as plt
import xgboost
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load data
iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# model setup
d_train = xgboost.DMatrix(x_train, label=y_train)
d_test = xgboost.DMatrix(x_test, label=y_test)

params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'multi:softprob',
    'num_class': 3, 
}

num_round = 50
bst = xgboost.train(params, d_train, num_round)

predictions = bst.predict(d_test)
predictions = [np.argmax(pred) for pred in predictions]

# plot
class_names = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica',
}
for i, (data_point, pred) in enumerate(zip(x_test, predictions)):
    label = class_names[pred]
    print(f'Prediction for datapoint {i+1} ({data_point}): {label}. Actual class: {class_names[y_test[i]]}')

print(f'Accuracy: {accuracy_score(y_test, predictions)}')

figure, ax = plt.subplots(figsize=(10, 6))
xgboost.plot_importance(bst, ax=ax)
features = ['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']
feature_labels = {f'f{i}': feature for i, feature in enumerate(features)}

handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=f'{key}: {value}', markersize=10, markerfacecolor='gray') 
    for key, value in feature_labels.items()
]

ax.legend(handles=handles, title='Feature Legend', loc='upper right')
plt.show()
