from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

# 注意到X的每一行是一个长度为2的向量，而y的每一行是一个长度为1的向量（标量）。
print(X[0], y[0])

plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
plt.show()
