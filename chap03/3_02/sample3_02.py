import numpy as np
import matplotlib.pyplot as plt

def classify(x, w):
    if np.dot(w,x) >= 0:
        return 1
    else:
        return -1

def learn_weights(x, t):
    w = np.random.rand(2)
    loop = 5
    count = 1

    for i in range(loop):
        for element_x, element_t in zip(x, t):
            if classify(element_x, w) != element_t:
                w = w + element_t * element_x
                print("更新後のw = ", w)

        print("[{}回目]: w = {}***".format(count, w))
        count += 1
    return w

data = np.loadtxt(
    "sample_zukai_NeuralNetwork/chap03/3_02/MaleFemale.csv",
    delimiter=",",
    skiprows=1
    )
x = data[:,0:2]
t = data[:,2]

w = learn_weights(x, t)

x1 = np.arange(0, 600)
# 分類ラベルが1のデータをドットでプロット
plt.plot(
    x[t ==  1, 0], x[t ==  1, 1], 'x'
    )
# 分類ラベルが－1のデータをxでプロット
plt.plot(
    x[t == -1, 0], x[t == -1, 1], 'o'
    )
# 境界線をプロット
plt.plot(
    x1, -w[0] / w[1] * x1, linestyle='solid'
    )
plt.show()