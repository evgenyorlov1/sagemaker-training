import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# generate data
np.random.seed(17)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# train linear regression model
model = LinearRegression()
model.fit(x, y)
y_predicted = model.predict(x)

# plot result
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, y_predicted, color="red", label="Regression Line")
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Linear regression")
plt.legend()
plt.show()

# MSE
mse = np.mean((y - y_predicted) ** 2)
print(mse)