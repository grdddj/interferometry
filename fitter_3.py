import numpy as np

seed = 12345512
np.random.seed(seed)

n = 100
x_data = np.linspace(-5, 5, num=n)
y_data = 10 + 5 * np.cos(3 * x_data + 2) + 1.5 * np.random.normal(size=n)

# random split the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)


from hyperopt import hp, tpe, Trials, fmin
from IPython.display import display, Math


def objective(a0, a1, w, f):
    """Objective function to minimize"""
    return np.mean((a0 + a1 * np.cos(w * X_train + f) - y_train) ** 2)


def objective2(args):
    return objective(*args)


space = [
    hp.uniform("a0", 5, 15),
    hp.uniform("a1", 0, 10),
    hp.uniform("w", 0, 10),
    hp.uniform("f", -np.pi, np.pi),
]

tpe_algo = tpe.suggest
tpe_trials = Trials()

tpe_best = fmin(
    fn=objective2,
    space=space,
    algo=tpe_algo,
    trials=tpe_trials,
    max_evals=500,
    rstate=np.random.RandomState(seed),
)


# from HOBIT import RegressionForTrigonometric

# trig_reg = RegressionForTrigonometric()
# trig_reg.fit_cos(X_train, y_train, max_evals=500, rstate=np.random.RandomState(seed))


# from scipy import optimize
# from IPython.display import display, Math


# def test_func(x, dist, amp, omega, phi):
#     return dist + amp * np.cos(omega * x + phi)


# params, params_covariance = optimize.curve_fit(
#     test_func, x_data, y_data, p0=[1, 1, 2, 1]
# )
# print("params", params)

# print("Fitted parameters:")
# display(Math("a_0={:.2f}, a_1={:.2f}, \\omega={:.2f}, \\phi={:.2f}".format(*params)))
# print("Original parameters:")
# display(
#     Math(
#         "a_0={:.2f}, a_1={:.2f}, \\omega={:.2f}, \\phi={:.2f}".format(
#             *[10.0, 5.0, 3.0, 2.0]
#         )
#     )
# )
