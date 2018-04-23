# 我们使用高维线性回归为例来引入一个过拟合问题。
# 具体来说我们使用如下的线性函数来生成每一个数据样本
# 这里噪音服从均值0和标准差为0.01的正态分布。
# 需要注意的是，我们用以上相同的数据生成函数来生成训练数据集和测试数据集。为了观察过拟合，
# 我们特意把训练数据样本数设低，例如n=20，同时把维度升高，例如p=200。

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import mxnet as mx

num_train = 20
num_test = 100
num_inputs = 200

true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05


X = nd.random.normal(shape=(num_train + num_test, num_inputs))
y = nd.dot(X, true_w) + true_b
y += .01 * nd.random.normal(shape=y.shape)

X_train, X_test = X[:num_train, :], X[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]

# 当我们开始训练神经网络的时候，我们需要不断读取数据块。这里我们定义一个函数它每次返回batch_size个随机的样本和对应的目标。
# 我们通过python的yield来构造一个迭代器。
import random
batch_size = 1


def data_iter(num_examples):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield X.take(j), y.take(j)


# 初始化模型参数
# 下面我们随机初始化模型参数。之后训练时我们需要对这些参数求导来更新它们的值，所以我们需要创建它们的梯度。
def init_params():
    w = nd.random_normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1, ))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params


# L2范数正则化
# 这里我们引入L2范数正则化。不同于在训练时仅仅最小化损失函数(Loss)，我们在训练时其实在最小化
# L = loss+λ∑p∈params||p||22。
# 直观上，L2范数正则化试图惩罚较大绝对值的参数值。下面我们定义L2正则化。注意有些时候大家对偏移加罚，有时候不加罚。
# 通常结果上两者区别不大。这里我们演示对偏移也加罚的情况：
def L2_penalty(w, b):
    return ((w**2).sum() + b**2) / 2


# 定义训练和测试
# 下面我们定义剩下的所需要的函数。这个跟之前的教程大致一样，主要是区别在于计算loss的时候我们加上了L2正则化，
# 以及我们将训练和测试损失都画了出来。
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt
import numpy as np

def net(X, w, b):
    return nd.dot(X, w) + b

def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def test(net, params, X, y):
    return square_loss(net(X, *params), y).mean().asscalar()
    #return np.mean(square_loss(net(X, *params), y).asnumpy())

def train(lambd):
    epochs = 10
    learning_rate = 0.005
    w, b = params = init_params()
    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in data_iter(num_train):
            with autograd.record():
                output = net(data, *params)
                loss = square_loss(
                    output, label) + lambd * L2_penalty(*params)
            loss.backward()
            sgd(params, learning_rate, batch_size)
        train_loss.append(test(net, params, X_train, y_train))
        test_loss.append(test(net, params, X_test, y_test))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()
    return 'learned w[:10]:', w[:10].T, 'learned b:', b


if __name__ == '__main__':
    print(train(5))






