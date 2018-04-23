import mxnet.ndarray as nd
import mxnet.autograd as ag

# 假设我们想对函数 f=2×x^2 求关于 x 的导数。我们先创建变量x，并赋初值。
x = nd.array([[1, 2], [3, 4]])

# 当进行求导的时候，我们需要一个地方来存x的导数，这个可以通过NDArray的方法attach_grad()来要求系统申请对应的空间。
x.attach_grad()

# 下面定义f。默认条件下，MXNet不会自动记录和构建用于求导的计算图，
# 我们需要使用autograd里的record()函数来显式的要求MXNet记录我们需要求导的程序。
with ag.record():
    z = x * x * 2

# 接下来我们可以通过z.backward()来进行求导。如果z不是一个标量，那么z.backward()等价于nd.sum(z).backward()
z.backward()

# 现在我们来看求出来的导数是不是正确的。注意到y = x*2和z = x*y，所以z等价于2*x*x。它的导数那么就是 dz/dx = 4×x
print('x.grad: ', x.grad)
# x.grad == 4*x


# ======================================================================================================================
# 对控制流求导
# 命令式的编程的一个便利之处是几乎可以对任意的可导程序进行求导，即使里面包含了Python的控制流。考虑下面程序，里面包含控制流for和if，
# 但循环迭代的次数和判断语句的执行都是取决于输入的值。不同的输入会导致这个程序的执行不一样。（对于计算图框架来说，这个对应于动态图，
# 就是图的结构会根据输入数据不同而改变）。
def f(a):
    b = a * 2
    print('a', a)
    print('nd.norm(a).asscalar()', nd.norm(a).asscalar())
    print('nd.norm(b).asscalar()', nd.norm(b).asscalar())
    while nd.norm(b).asscalar() < 1000:
        b = b * 2
    if nd.sum(b).asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c


# 我们可以跟之前一样使用record记录和backward求导。
a = nd.random_normal(shape=3)
a.attach_grad()
with ag.record():
    c = f(a)
c.backward()

# 注意到给定输入a，其输出 f(a)=xa，x 的值取决于输入a。所以有 df/da=x，我们可以很简单地评估自动求导的导数：
print('a.grad: ', a.grad)

# ======================================================================================================================
# 头梯度和链式法则
# 注意：读者可以跳过这一小节，不会影响阅读之后的章节
# 当我们在一个NDArray上调用backward方法时，例如y.backward()，此处y是一个关于x的函数，我们将求得y关于x的导数。数学家们会把这个求导
# 写成 dy(x)dx 。还有些更复杂的情况，比如z是关于y的函数，且y是关于x的函数，我们想对z关于x求导，也就是求 ddxz(y(x)) 的结果。回想一
# 下链式法则，我们可以得到ddxz(y(x))=dz(y)dydy(x)dx。当y是一个更大的z函数的一部分，并且我们希望求得 dzdx 保存在x.grad中时，我们
# 可以传入头梯度（head gradient） dzdy 的值作为backward()方法的输入参数，系统会自动应用链式法则进行计算。这个参数的默认值是
# nd.ones_like(y)。关于链式法则的详细解释，请参阅Wikipedia。
with ag.record():
    y = x * 2
    z = y * x

head_gradient = nd.array([[10, 1.], [.1, .01]])
z.backward(head_gradient)
print('x.grad: ', x.grad)