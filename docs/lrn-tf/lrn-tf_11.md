# 附录 A. 模型构建和使用 TensorFlow Serving 的提示

# 模型结构化和定制化

在这个简短的部分中，我们将专注于两个主题，这些主题延续并扩展了前几章——如何构建一个合适的模型，以及如何定制模型的实体。我们首先描述如何通过使用封装来有效地重构我们的代码，并允许其变量被共享和重复使用。在本节的第二部分，我们将讨论如何定制我们自己的损失函数和操作，并将它们用于优化。

## 模型结构化

最终，我们希望设计我们的 TensorFlow 代码高效，以便可以重用于多个任务，并且易于跟踪和传递。使事情更清晰的一种方法是使用可用的 TensorFlow 扩展库之一，这些库在第七章中已经讨论过。然而，虽然它们非常适合用于典型的网络，但有时我们希望实现的具有新组件的模型可能需要较低级别 TensorFlow 的完全灵活性。

让我们再次看一下前一章的优化代码：

```py
import tensorflow as tf

NUM_STEPS = 10

g = tf.Graph()
wb_ = []
with g.as_default():
  x = tf.placeholder(tf.float32,shape=[None,3])
  y_true = tf.placeholder(tf.float32,shape=None)

  with tf.name_scope('inference') as scope:
    w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
    b = tf.Variable(0,dtype=tf.float32,name='bias')
    y_pred = tf.matmul(w,tf.transpose(x)) + b

  with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.square(y_true-y_pred))

  with tf.name_scope('train') as scope:
    learning_rate = 0.5
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)   
    for step in range(NUM_STEPS):
      sess.run(train,{x: x_data, y_true: y_data})
      if (step % 5 == 0):
        print(step, sess.run([w,b]))
        wb_.append(sess.run([w,b]))

    print(10, sess.run([w,b]))

```

我们得到：

```py
(0, [array([[ 0.30149955,  0.49303722,  0.11409992]], 
                                        dtype=float32), -0.18563795])
(5, [array([[ 0.30094019,  0.49846715,  0.09822173]], 
                                        dtype=float32), -0.19780949])
(10, [array([[ 0.30094025,  0.49846718,  0.09822182]], 
                                        dtype=float32), -0.19780946])

```

这里的整个代码只是简单地一行一行堆叠。对于简单和专注的示例来说，这是可以的。然而，这种编码方式有其局限性——当代码变得更加复杂时，它既不可重用也不太可读。

让我们放大视野，思考一下我们的基础设施应该具有哪些特征。首先，我们希望封装模型，以便可以用于各种任务，如训练、评估和形成预测。此外，以模块化的方式构建模型可能更有效，使我们能够对其子组件具有特定控制，并增加可读性。这将是接下来几节的重点。

### 模块化设计

一个很好的开始是将代码分成捕捉学习模型中不同元素的函数。我们可以这样做：

```py
def predict(x,y_true,w,b):
  y_pred = tf.matmul(w,tf.transpose(x)) + b
  return y_pred

def get_loss(y_pred,y_true):
  loss = tf.reduce_mean(tf.square(y_true-y_pred))
  return loss

def get_optimizer(y_pred,y_true):
  loss = get_loss(y_pred,y_true)
  optimizer = tf.train.GradientDescentOptimizer(0.5)
  train = optimizer.minimize(loss)
  return train

def run_model(x_data,y_data):
    wb_ = []
    # Define placeholders and variables
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)
    w = tf.Variable([[0,0,0]],dtype=tf.float32)
    b = tf.Variable(0,dtype=tf.float32)
    print(b.name)

    # Form predictions
    y_pred = predict(x,y_true,w,b)

    # Create optimizer
    train = get_optimizer(y_pred,y_data)

    # Run session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)   
      for step in range(10):
        sess.run(train,{x: x_data, y_true: y_data})
          if (step % 5 == 0):
          print(step, sess.run([w,b]))
          wb_.append(sess.run([w,b]))

run_model(x_data,y_data)
run_model(x_data,y_data)
```

这里是结果：

```py
Variable_9:0 Variable_8:0
0 [array([[ 0.27383861,  0.48421991,  0.09082422]], 
                                      dtype=float32), -0.20805186]
4 [array([[ 0.29868397,  0.49840903,  0.10026278]], 
                                      dtype=float32), -0.20003076]
9 [array([[ 0.29868546,  0.49840906,  0.10026464]], 
                                      dtype=float32), -0.20003042]

Variable_11:0 Variable_10:0
0 [array([[ 0.27383861,  0.48421991,  0.09082422]], 
                                      dtype=float32), -0.20805186]
4 [array([[ 0.29868397,  0.49840903,  0.10026278]], 
                                      dtype=float32), -0.20003076]
9 [array([[ 0.29868546,  0.49840906,  0.10026464]], 
                                      dtype=float32), -0.20003042]

```

现在我们可以重复使用具有不同输入的代码，这种划分使其更易于阅读，特别是在变得更加复杂时。

在这个例子中，我们两次调用了主函数并打印了创建的变量。请注意，每次调用都会创建不同的变量集，从而创建了四个变量。例如，假设我们希望构建一个具有多个输入的模型，比如两个不同的图像。假设我们希望将相同的卷积滤波器应用于两个输入图像。将创建新的变量。为了避免这种情况，我们可以“共享”滤波器变量，在两个图像上使用相同的变量。

### 变量共享

通过使用`tf.get_variable()`而不是`tf.Variable()`，可以重复使用相同的变量。我们使用方式与`tf.Variable()`非常相似，只是需要将初始化器作为参数传递：

```py
w = tf.get_variable('w',[1,3],initializer=tf.zeros_initializer())
b = tf.get_variable('b',[1,1],initializer=tf.zeros_initializer())

```

在这里，我们使用了`tf.zeros_initializer()`。这个初始化器与`tf.zeros()`非常相似，只是它不会将形状作为参数，而是根据`tf.get_variable()`指定的形状排列值。

在这个例子中，变量`w`将被初始化为`[0,0,0]`，如给定的形状`[1,3]`所指定。

使用`get_variable()`，我们可以重复使用具有相同名称的变量（包括作用域前缀，可以通过`tf.variable_scope()`设置）。但首先，我们需要通过使用`tf.variable_scope.reuse_variable()`或设置`reuse`标志（`tf.variable.scope(reuse=True)`）来表明这种意图。下面的代码示例展示了如何共享变量。

# 标志误用的注意事项

每当一个变量与另一个变量具有完全相同的名称时，在未设置`reuse`标志时会抛出异常。相反的情况也是如此——期望重用的名称不匹配的变量（当`reuse=True`时）也会导致异常。

使用这些方法，并将作用域前缀设置为`Regression`，通过打印它们的名称，我们可以看到相同的变量被重复使用：

```py
def run_model(x_data,y_data):
    wb_ = []
    # Define placeholders and variables
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)

    w = tf.get_variable('w',[1,3],initializer=tf.zeros_initializer())
    b = tf.get_variable('b',[1,1],initializer=tf.zeros_initializer())

    print(b.name,w.name)

    # Form predictions
    y_pred = predict(x,y_true,w,b)

    # Create optimizer
    train = get_optimizer(y_pred,y_data)

    # Run session
    init = tf.global_variables_initializer()
    sess.run(init)   
    for step in range(10):
      sess.run(train,{x: x_data, y_true: y_data})
      if (step % 5 == 4) or (step == 0):
        print(step, sess.run([w,b]))
        wb_.append(sess.run([w,b]))

sess = tf.Session()

with tf.variable_scope("Regression") as scope:
  run_model(x_data,y_data)
  scope.reuse_variables()
  run_model(x_data,y_data)
sess.close()

```

输出如下所示：

```py
Regression/b:0 Regression/w:0
0 [array([[ 0.27383861,  0.48421991,  0.09082422]], 
              dtype=float32), array([[-0.20805186]], dtype=float32)]
4 [array([[ 0.29868397,  0.49840903,  0.10026278]], 
              dtype=float32), array([[-0.20003076]], dtype=float32)]
9 [array([[ 0.29868546,  0.49840906,  0.10026464]], 
              dtype=float32), array([[-0.20003042]], dtype=float32)]

Regression/b:0 Regression/w:0
0 [array([[ 0.27383861,  0.48421991,  0.09082422]], 
              dtype=float32), array([[-0.20805186]], dtype=float32)]
4 [array([[ 0.29868397,  0.49840903,  0.10026278]], 
              dtype=float32), array([[-0.20003076]], dtype=float32)]
9 [array([[ 0.29868546,  0.49840906,  0.10026464]], 
              dtype=float32), array([[-0.20003042]], dtype=float32)]

```

`tf.get_variables()`是一个简洁、轻量级的共享变量的方法。另一种方法是将我们的模型封装为一个类，并在那里管理变量。这种方法有许多其他好处，如下一节所述

### 类封装

与任何其他程序一样，当事情变得更加复杂，代码行数增加时，将我们的 TensorFlow 代码放在一个类中变得非常方便，这样我们就可以快速访问属于同一模型的方法和属性。类封装允许我们维护变量的状态，然后执行各种训练后任务，如形成预测、模型评估、进一步训练、保存和恢复权重，以及与我们的模型解决的特定问题相关的任何其他任务。

在下一批代码中，我们看到一个简单的类包装器示例。当实例化时创建模型，并通过调用`fit()`方法执行训练过程。

# @property 和 Python 装饰器

这段代码使用了`@property`装饰器。*装饰器*只是一个以另一个函数作为输入的函数，对其进行一些操作（比如添加一些功能），然后返回它。在 Python 中，装饰器用`@`符号定义。

`@property`是一个用于处理类属性访问的装饰器。

我们的类包装器如下：

```py
class Model:
  def __init__(self):

    # Model
    self.x = tf.placeholder(tf.float32,shape=[None,3])
    self.y_true = tf.placeholder(tf.float32,shape=None)
    self.w = tf.Variable([[0,0,0]],dtype=tf.float32)
    self.b = tf.Variable(0,dtype=tf.float32)

    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init)

    self._output = None
    self._optimizer = None
    self._loss = None

  def fit(self,x_data,y_data):
    print(self.b.name)

    for step in range(10):
      self.sess.run(self.optimizer,{self.x: x_data, self.y_true: y_data})
      if (step % 5 == 4) or (step == 0):
        print(step, self.sess.run([self.w,self.b]))

  @property
  def output(self):
    if not self._output:
      y_pred = tf.matmul(self.w,tf.transpose(self.x)) + self.b
      self._output = y_pred
    return self._output

  @property
  def loss(self):
    if not self._loss:
      error = tf.reduce_mean(tf.square(self.y_true-self.output))
      self._loss= error
    return self._loss

  @property
  def optimizer(self):
    if not self._optimizer:
      opt = tf.train.GradientDescentOptimizer(0.5)
      opt = opt.minimize(self.loss)
      self._optimizer = opt
    return self._optimizer

lin_reg = Model()
lin_reg.fit(x_data,y_data)
lin_reg.fit(x_data,y_data)

```

然后我们得到这个：

```py
Variable_89:0
0 [array([[ 0.32110521,  0.4908163 ,  0.09833425]], 
                                       dtype=float32), -0.18784374]
4 [array([[ 0.30250472,  0.49442694,  0.10041162]], 
                                       dtype=float32), -0.1999902]
9 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                       dtype=float32), -0.19999036]

Variable_89:0
0 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                       dtype=float32), -0.19999038]
4 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                       dtype=float32), -0.19999038]
9 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                       dtype=float32), -0.19999036]

```

将代码拆分为函数在某种程度上是多余的，因为相同的代码行在每次调用时都会重新计算。一个简单的解决方案是在每个函数的开头添加一个条件。在下一个代码迭代中，我们将看到一个更好的解决方法。

在这种情况下，不需要使用变量共享，因为变量被保留为模型对象的属性。此外，在调用两次训练方法`model.fit()`后，我们看到变量保持了它们的当前状态。

在本节的最后一批代码中，我们添加了另一个增强功能，创建一个自定义装饰器，自动检查函数是否已被调用。

我们可以做的另一个改进是将所有变量保存在字典中。这将使我们能够在每次操作后跟踪我们的变量，就像我们在第十章中看到的那样，当我们查看保存权重和模型时。

最后，添加了用于获取损失函数值和权重的额外函数：

```py
class Model:
  def __init__(self):

    # Model
    self.x = tf.placeholder(tf.float32,shape=[None,3])
    self.y_true = tf.placeholder(tf.float32,shape=None)

    self.params = self._initialize_weights()

    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init)

    self.output
    self.optimizer
    self.loss

  def _initialize_weights(self):
    params = dict()
    params['w'] = tf.Variable([[0,0,0]],dtype=tf.float32)
    params['b'] = tf.Variable(0,dtype=tf.float32)
    return params

  def fit(self,x_data,y_data):
    print(self.params['b'].name)

    for step in range(10):
      self.sess.run(self.optimizer,{self.x: x_data, self.y_true: y_data})
      if (step % 5 == 4) or (step == 0):
        print(step, 
                self.sess.run([self.params['w'],self.params['b']]))

  def evaluate(self,x_data,y_data):
    print(self.params['b'].name)

    MSE = self.sess.run(self.loss,{self.x: x_data, self.y_true: y_data})
    return MSE

  def getWeights(self):
    return self.sess.run([self.params['b']])

  @property_with_check
  def output(self):
    y_pred = tf.matmul(self.params['w'],tf.transpose(self.x)) + \
        self.params['b']
    return y_pred

  @property_with_check
  def loss(self):
    error = tf.reduce_mean(tf.square(self.y_true-self.output))
    return error

  @property_with_check
  def optimizer(self):
    opt = tf.train.GradientDescentOptimizer(0.5)
    opt = opt.minimize(self.loss)
    return opt

lin_reg = Model()
lin_reg.fit(x_data,y_data)
MSE = lin_reg.evaluate(x_data,y_data)
print(MSE)

print(lin_reg.getWeights())

```

以下是输出：

```py
Variable_87:0
0 [array([[ 0.32110521,  0.4908163 ,  0.09833425]], 
                                        dtype=float32), -0.18784374]
4 [array([[ 0.30250472,  0.49442694,  0.10041162]], 
                                        dtype=float32), -0.1999902]
9 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                        dtype=float32), -0.19999036]

Variable_87:0
0 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                        dtype=float32), -0.19999038]
4 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                        dtype=float32), -0.19999038]
9 [array([[ 0.30250433,  0.49442688,  0.10041161]], 
                                        dtype=float32), -0.19999036]
Variable_87:0
0.0102189
[-0.19999036] 
```

自定义装饰器检查属性是否存在，如果不存在，则根据输入函数设置它。否则，返回属性。使用`functools.wrap()`，这样我们就可以引用函数的名称：

```py
import functools

def property_with_check(input_fn):
  attribute = '_cache_' + input_fn.__name__

  @property
  @functools.wraps(input_fn)
  def check_attr(self):
    if not hasattr(self, attribute):
      setattr(self, attribute, input_fn(self))
    return getattr(self, attribute)

  return check_attr

```

这是一个相当基本的示例，展示了我们如何改进模型的整体代码。这种优化可能对我们简单的线性回归示例来说有些过度，但对于具有大量层、变量和特征的复杂模型来说，这绝对是值得的努力。

## 定制

到目前为止，我们使用了两个损失函数。在第二章中的分类示例中，我们使用了交叉熵损失，定义如下：

```py
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
```

相比之下，在前一节的回归示例中，我们使用了平方误差损失，定义如下：

```py
loss = tf.reduce_mean(tf.square(y_true-y_pred))
```

这些是目前在机器学习和深度学习中最常用的损失函数。本节的目的是双重的。首先，我们想指出 TensorFlow 在利用自定义损失函数方面的更一般能力。其次，我们将讨论正则化作为任何损失函数的扩展形式，以实现特定目标，而不考虑使用的基本损失函数。

### 自制损失函数

本书（以及我们的读者）以深度学习为重点来看待 TensorFlow。然而，TensorFlow 的范围更广泛，大多数机器学习问题都可以以一种 TensorFlow 可以解决的方式来表述。此外，任何可以在计算图框架中表述的计算都是从 TensorFlow 中受益的好候选。

主要特例是无约束优化问题类。这些问题在科学（和算法）计算中非常常见，对于这些问题，TensorFlow 尤为有用。这些问题突出的原因是，TensorFlow 提供了计算梯度的自动机制，这为解决这类问题的开发时间提供了巨大的加速。

一般来说，对于任意损失函数的优化将采用以下形式

```py
def my_loss_function(key-variables...):
loss = ...
return loss

my_loss = my_loss_function(key-variables...)
gd_step = tf.train.GradientDescentOptimizer().minimize(my_loss)
```

任何优化器都可以用于替代 `GradientDescentOptimizer`。

### 正则化

正则化是通过对解决方案的复杂性施加惩罚来限制优化问题（有关更多详细信息，请参见第四章中的注释）。在本节中，我们将看一下特定情况下，惩罚直接添加到基本损失函数中的附加形式。

例如，基于第二章中 softmax 示例，我们有以下内容：

```py
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

total_loss = cross_entropy + LAMBDA * tf.nn.l2_loss(W)

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(total_loss)
```

与第二章中的原始版本的区别在于，我们将 `LAMBDA * tf.nn.l2_loss(W)` 添加到我们正在优化的损失中。在这种情况下，使用较小的权衡参数 `LAMBDA` 值对最终准确性的影响很小（较大的值会有害）。在大型网络中，过拟合是一个严重问题，这种正则化通常可以拯救一命。

这种类型的正则化可以针对模型的权重进行，如前面的示例所示（也称为*权重衰减*，因为它会使权重值变小），也可以针对特定层或所有层的激活进行。

另一个因素是我们使用的函数——我们可以使用 `l1` 而不是 `l2` 正则化，或者两者的组合。所有这些正则化器的组合都是有效的，并在各种情境中使用。

许多抽象层使正则化的应用变得简单，只需指定滤波器数量或激活函数即可。例如，在 Keras（在第七章中审查的非常流行的扩展）中，我们提供了适用于所有标准层的正则化器，列在表 A-1 中。

表 A-1. 使用 Keras 进行正则化

| 正则化器 | 作用 | 示例 |
| --- | --- | --- |
| `l1` | `l1` 正则化权重 |

```py
Dense(100, W_regularizer=l1(0.01))
```

|

| `l2` | `l2` 正则化权重 |
| --- | --- |

```py
Dense(100, W_regularizer=l2(0.01))
```

|

| `l1l2` | 组合 `l1 + l2` 正则化权重 |
| --- | --- |

```py
Dense(100, W_regularizer=l1l2(0.01))
```

|

| `activity_l1` | `l1` 正则化激活 |
| --- | --- |

```py
Dense(100, activity_regularizer=activity_l1(0.01))
```

|

| `activity_l2` | `l2` 正则化激活 |
| --- | --- |

```py
Dense(100, activity_regularizer=activity_l2(0.01))
```

|

| `activity_l1l2` | 组合 `l1 + l2` 正则化激活 |
| --- | --- |

```py
Dense(100, activity_regularizer=activity_l1l2(0.01))
```

|

在模型过拟合时，使用这些快捷方式可以轻松测试不同的正则化方案。

### 编写自己的操作

TensorFlow 预装了大量本地操作，从标准算术和逻辑操作到矩阵操作、深度学习特定函数等等。当这些操作不够时，可以通过创建新操作来扩展系统。有两种方法可以实现这一点：

+   编写一个“从头开始”的 C++ 版本的操作

+   编写结合现有操作和 Python 代码创建新操作的 Python 代码

我们将在本节的其余部分讨论第二个选项。

构建 Python op 的主要原因是在 TensorFlow 计算图的上下文中利用 NumPy 功能。为了说明，我们将使用 NumPy 乘法函数构建前一节中的正则化示例，而不是 TensorFlow op：

```py
import numpy as np

LAMBDA = 1e-5

def mul_lambda(val):
    return np.multiply(val, LAMBDA).astype(np.float32)
```

请注意，这是为了说明的目的，没有特别的原因让任何人想要使用这个而不是原生的 TensorFlow op。我们使用这个过度简化的例子是为了将焦点转移到机制的细节而不是计算上。

为了在 TensorFlow 内部使用我们的新创建，我们使用`py_func()`功能：

```py
tf.py_func(my_python_function, [input], [output_types])

```

在我们的情况下，这意味着我们计算总损失如下：

```py
total_loss = cross_entropy + \
        tf.py_func(mul_lambda, [tf.nn.l2_loss(W)], [tf.float32])[0]
```

然而，这样做还不够。回想一下，TensorFlow 会跟踪每个 op 的梯度，以便对我们整体模型进行基于梯度的训练。为了使这个与新的基于 Python 的 op 一起工作，我们必须手动指定梯度。这分为两个步骤。

首先，我们创建并注册梯度：

```py
@tf.RegisterGradient("PyMulLambda")
def grad_mul_lambda(op, grad):
    return LAMBDA*grad

```

接下来，在使用函数时，我们将这个函数指定为 op 的梯度。这是使用在上一步中注册的字符串完成的：

```py
with tf.get_default_graph().gradient_override_map({"PyFunc": "PyMulLambda"}):
    total_loss = cross_entropy + \
            tf.py_func(mul_lambda, [tf.nn.l2_loss(W)], [tf.float32])[0]
```

将所有内容放在一起，通过我们基于新 Python op 的正则化 softmax 模型的代码现在是：

```py
import numpy as np
import tensorflow as tf

LAMBDA = 1e-5

def mul_lambda(val):
    return np.multiply(val, LAMBDA).astype(np.float32)

@tf.RegisterGradient("PyMulLambda")
def grad_mul_lambda(op, grad):
    return LAMBDA*grad

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = 
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
                (logits=y_pred, labels=y_true))

with tf.get_default_graph().gradient_override_map({"PyFunc": "PyMulLambda"}):
    total_loss = cross_entropy + \
            tf.py_func(mul_lambda, [tf.nn.l2_loss(W)], [tf.float32])[0]

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(total_loss)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
```

现在可以使用与第二章中首次介绍该模型时相同的代码进行训练。

# 在计算梯度时使用输入

在我们刚刚展示的简单示例中，梯度仅取决于相对于输入的梯度，而不是输入本身。在一般情况下，我们还需要访问输入。这很容易做到，使用`op.input`s 字段：

```py
x = op.inputs[0]
```

其他输入（如果存在）以相同的方式访问。

# TensorFlow Serving 所需和推荐的组件

在本节中，我们添加了一些在第十章中涵盖的材料的细节，并更深入地审查了 TensorFlow Serving 背后使用的一些技术组件。

在第十章中，我们使用 Docker 来运行 TensorFlow Serving。那些喜欢避免使用 Docker 容器的人需要安装以下内容：

Bazel

Bazel 是谷歌自己的构建工具，最近才公开。当我们使用术语*构建*时，我们指的是使用一堆规则从源代码中创建输出软件，以非常高效和可靠的方式。构建过程还可以用来引用构建输出所需的外部依赖项。除了其他语言，Bazel 还可以用于构建 C++应用程序，我们利用这一点来构建用 C++编写的 TensorFlow Serving 程序。Bazel 构建的源代码基于一个工作区目录，其中包含一系列包含相关源文件的嵌套层次结构的包。每个软件包包含三种类型的文件：人工编写的源文件称为*targets*，从源文件创建的*生成文件*，以及指定从输入派生输出的步骤的*规则*。

每个软件包都有一个*BUILD*文件，指定从该软件包内的文件构建的输出。我们使用基本的 Bazel 命令，比如`bazel build`来从目标构建生成的文件，以及`bazel run`来执行构建规则。当我们想要指定包含构建输出的目录时，我们使用`-bin`标志。

下载和安装说明可以在[Bazel 网站](https://bazel.build/versions/master/docs/install.html)上找到。

gRPC

远程过程调用（RPC）是一种客户端（调用者）-服务器（执行者）交互的形式；程序可以请求在另一台计算机上执行的过程（例如，一个方法）（通常在共享网络中）。gRPC 是由 Google 开发的开源框架。与任何其他 RPC 框架一样，gRPC 允许您直接调用其他机器上的方法，从而更容易地分发应用程序的计算。gRPC 的伟大之处在于它如何处理序列化，使用快速高效的协议缓冲区而不是 XML 或其他方法。

下载和安装说明可以在[GitHub](https://github.com/grpc/grpc/tree/master/src/python/grpcio)上找到。

接下来，您需要确保使用以下命令安装了 Serving 所需的依赖项：

```py
sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev
```

最后，克隆 Serving：

```py
git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving  
```

如第十章所示，另一个选择是使用 Docker 容器，实现简单干净的安装。

## 什么是 Docker 容器，为什么我们要使用它？

Docker 本质上解决了与 VirtualBox 的 Vagrant 相同的问题，即确保我们的代码在其他机器上能够顺利运行。不同的机器可能具有不同的操作系统以及不同的工具集（安装的软件、配置、权限等）。通过复制相同的环境——也许是为了生产目的，也许只是与他人分享——我们保证我们的代码在其他地方与在原始开发机器上运行的方式完全相同。

Docker 的独特之处在于，与其他类似用途的工具不同，它不会创建一个完全操作的虚拟机来构建环境，而是在现有系统（例如 Ubuntu）之上创建一个*容器*，在某种意义上充当虚拟机，并利用我们现有的操作系统资源。这些容器是从本地 Docker *镜像*创建的，该镜像是从*dockerfile*构建的，并封装了我们需要的一切（依赖安装、项目代码等）。从该镜像中，我们可以创建任意数量的容器（当然，直到内存用尽为止）。这使得 Docker 成为一个非常酷的工具，我们可以轻松地创建包含我们的代码的完整多个环境副本，并在任何地方运行它们（对于集群计算非常有用）。

## 一些基本的 Docker 命令

为了让您更加熟悉使用 Docker，这里简要介绍一些有用的命令，以最简化的形式编写。假设我们已经准备好一个 dockerfile，我们可以使用`docker build <*dockerfile*>`来构建一个镜像。然后，我们可以使用`docker run <*image*>`命令创建一个新的容器。该命令还将自动运行容器并打开一个终端（输入`exit`关闭终端）。要运行、停止和删除现有容器，我们分别使用`docker start <*container id*>`、`docker stop <*container id*>`和`docker rm <*container id*>`命令。要查看所有实例的列表，包括正在运行和空闲的实例，我们输入`docker ps -a`。

当我们运行一个实例时，我们可以添加`-p`标志，后面跟一个端口供 Docker 暴露，以及`-v`标志，后面跟一个要挂载的主目录，这将使我们能够在本地工作（主目录通过容器中的`/mnt/home`路径进行访问）。
