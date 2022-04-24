"""
===============
Curve fitting
===============

Demos a simple curve fitting
"""

############################################################
# First generate some data
import numpy as np

# Seed the random number generator for reproducibility
np.random.seed(0)

from fitter_data import y_data, x_data

y_data = np.array(y_data)
x_data = np.array(x_data)

# And plot it
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data)

############################################################
# Now fit a simple sine function to the data
from scipy import optimize


def test_func(x, a, b, c, d):
    return a * np.sin(b * x + c) + d


# params, params_covariance = optimize.curve_fit(
#     test_func, x_data, y_data, p0=[20, 0.2, 0, 40]
# )
params, params_covariance = optimize.curve_fit(test_func, x_data, y_data)

print(params)

############################################################
# And plot the resulting curve on the data

plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data, label="Data")
plt.plot(
    x_data,
    test_func(x_data, params[0], params[1], params[2], params[3]),
    label="Fitted function",
)
plt.title(f"{params=}")

plt.legend(loc="best")

plt.show()
