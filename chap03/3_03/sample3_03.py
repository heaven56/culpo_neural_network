import numpy as np
import matplotlib.pyplot as plt

def standardsize(x):
    x_mean = x.mean(axis=0)
    std = x.std(axis=0)
    return (x - x_mean) / std


def create_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, 1])

def sigmoid(X, parameter):
    return 1 / (1 + np.exp(-np.dot(X, parameter)))

def logistic_regression(X, t):
    parameter = np.random.rand(3)
    LNR = 1e-3
    loop = 2000
    count = 1

    for i in range(loop):
        parameter = parameter - LNR * np.dot(
            sigmoid(X, parameter) - t,
            X
            )
        if (count == 1 or count % 100 == 0):
            print("{}回: parameter = {}".format(count, parameter))
        count += 1
    return parameter

data = np.loadtxt(
    "sample_zukai_NeuralNetwork/chap03/3_03/MaleFemale2.csv",
    dtype="int",
    delimiter=",",
    skiprows=1
    )
x = data[:,0:2]
t = data[:,2]
x_std = standardsize(x)
X = create_matrix(x_std)

parameter = logistic_regression(X, t)

x0 = np.linspace(start=-2,
                stop=2,
                num=1100
                )

plt.plot(x_std[t == 1, 0],
         x_std[t == 1,1], "x")

plt.plot(x_std[t == 0, 0],
         x_std[t == 0, 1], "o")

plt.plot(x0,
         -(parameter[0] + parameter[1] * x0) / parameter[2],
         linestyle="dashed"
        )

plt.show()