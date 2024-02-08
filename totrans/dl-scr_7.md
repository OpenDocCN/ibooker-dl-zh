# 第7章。PyTorch

在第[6](ch06.html#recurrent)章和第[5](ch05.html#convolution)章中，您学习了如何通过从头开始实现卷积和循环神经网络来使它们工作。然而，了解它们如何工作是必要的，但仅凭这些知识无法使它们在真实世界的问题上工作；为此，您需要能够在高性能库中实现它们。我们可以致力于构建一个高性能神经网络库的整本书，但那将是一本非常不同（或者只是更长）的书，面向一个非常不同的受众。相反，我们将把这最后一章献给介绍PyTorch，这是一个越来越受欢迎的基于自动微分的神经网络框架，我们在[第6章](ch06.html#recurrent)的开头介绍过。

与本书的其余部分一样，我们将以与神经网络工作方式相匹配的方式编写代码，编写`Layer`、`Trainer`等类。这样做的同时，我们不会按照常见的PyTorch实践编写代码，但我们将在[本书的GitHub存储库](https://oreil.ly/2N4H8jz)中包含链接，让您了解更多关于如何表达神经网络的信息，以PyTorch设计的方式来表达。在我们开始之前，让我们从学习PyTorch的核心数据类型开始，这种数据类型使其具有自动微分的能力，从而使其能够清晰地表达神经网络训练：`Tensor`。

# PyTorch张量

在上一章中，我们展示了一个简单的`NumberWithGrad`通过跟踪对其执行的操作来累积梯度。这意味着如果我们写：

```py
a = NumberWithGrad(3)

b = a * 4
c = b + 3
d = (a + 2)
e = c * d
e.backward()
```

然后`a.grad`将等于`35`，这实际上是`e`相对于`a`的偏导数。

PyTorch的`Tensor`类就像一个"`ndarrayWithGrad`"：它类似于`NumberWithGrad`，只是使用数组（如`numpy`）而不仅仅是`float`和`int`。让我们使用PyTorch的`Tensor`重新编写前面的示例。首先，我们将手动初始化一个`Tensor`：

```py
a = torch.Tensor([[3., 3.,],
                  [3., 3.]], requires_grad=True)
```

这里注意几点：

1.  我们可以通过简单地将其中包含的数据包装在`torch.Tensor`中来初始化一个`Tensor`，就像我们用`ndarray`做的那样。

1.  当以这种方式初始化`Tensor`时，我们必须传入参数`requires_grad=True`，以告诉`Tensor`累积梯度。

一旦我们这样做了，我们就可以像以前一样执行计算：

```py
b = a * 4
c = b + 3
d = (a + 2)
e = c * d
e_sum = e.sum()
e_sum.backward()
```

您可以看到与`NumberWithGrad`示例相比，这里有一个额外的步骤：在调用其总和之前，我们必须*对`e`进行求和*，然后调用`backward`。这是因为，正如我们在第一章中所讨论的，想象“一个数字相对于一个数组的导数”是没有意义的：但是，我们可以推断出`e_sum`相对于`a`的每个元素的偏导数是什么，并且，我们看到答案与我们在之前的章节中发现的是一致的：

```py
print(a.grad)
```

```py
tensor([[35., 35.],
        [35., 35.]], dtype=torch.float64)
```

PyTorch的这个特性使我们能够通过定义前向传播、计算损失并在损失上调用`.backward`来简单地定义模型，并自动计算每个`参数`相对于该损失的导数。特别是，我们不必担心在前向传递中多次重复使用相同的数量（这是我们在前几章中使用的`Operation`框架的限制）；正如这个简单的例子所示，一旦我们在我们的计算输出上调用`backward`，梯度将自动正确计算。

在接下来的几节中，我们将展示如何使用PyTorch的数据类型实现我们在本书中提出的训练框架。

# 使用PyTorch进行深度学习

正如我们所看到的，深度学习模型有几个元素共同工作以产生一个经过训练的模型：

+   一个包含`Layer`的`Model`

+   一个`Optimizer`

+   一个`Loss`

+   一个`Trainer`

事实证明，使用PyTorch，`Optimizer`和`Loss`都是一行代码，`Model`和`Layer`也很简单。让我们依次介绍这些元素。

## PyTorch元素：模型、层、优化器和损失

PyTorch的一个关键特性是能够将模型和层定义为易于使用的对象，通过继承`torch.nn.Module`类处理梯度向后传播和自动存储参数。稍后在本章中您将看到这些部分如何结合在一起；现在只需知道`PyTorchLayer`可以这样编写：

```py
from torch import nn, Tensor

class PyTorchLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor,
                inference: bool = False) -> Tensor:
        raise NotImplementedError()
```

`PyTorchModel`也可以这样编写：

```py
class PyTorchModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor,
                inference: bool = False) -> Tensor:
        raise NotImplementedError()
```

换句话说，`PyTorchLayer`或`PyTorchModel`的每个子类只需要实现`__init__`和`forward`方法，这将使我们能够以直观的方式使用它们。

### 推断标志

正如我们在[第4章](ch04.html#extensions)中看到的，由于dropout，我们需要根据我们是在训练模式还是推断模式下运行模型的能力来改变模型的行为。在PyTorch中，我们可以通过在模型或层（从`nn.Module`继承的任何对象）上运行`m.eval`将模型或层从训练模式（其默认行为）切换到推断模式。此外，PyTorch有一种优雅的方式可以快速更改层的所有子类的行为，即使用`apply`函数。如果我们定义：

```py
def inference_mode(m: nn.Module):
    m.eval()
```

然后我们可以包括：

```py
if inference:
    self.apply(inference_mode)
```

在我们定义的每个`PyTorchModel`或`PyTorchLayer`子类的`forward`方法中，从而获得我们想要的标志。

让我们看看这是如何结合在一起的。

## 使用PyTorch实现神经网络构建块：DenseLayer

现在我们已经具备开始使用PyTorch操作实现之前看到的`Layer`的所有先决条件，一个`DenseLayer`层将被写成如下：

```py
class DenseLayer(PyTorchLayer):
    def __init__(self,
                 input_size: int,
                 neurons: int,
                 dropout: float = 1.0,
                 activation: nn.Module = None) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, neurons)
        self.activation = activation
        if dropout < 1.0:
            self.dropout = nn.Dropout(1 - dropout)

    def forward(self, x: Tensor,
                inference: bool = False) -> Tensor:
        if inference:
            self.apply(inference_mode)

        x = self.linear(x) # does weight multiplication + bias
        if self.activation:
            x = self.activation(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        return x
```

在这里，通过`nn.Linear`，我们看到了PyTorch操作的第一个示例，它会自动处理反向传播。这个对象不仅在前向传播时处理权重乘法和偏置项的加法，还会导致`x`的梯度累积，以便在反向传播时计算出参数相对于损失的正确导数。还要注意，由于所有PyTorch操作都继承自`nn.Module`，我们可以像数学函数一样调用它们：在前面的情况下，例如，我们写`self.linear(x)`而不是`self.linear.forward(x)`。当我们在即将到来的模型中使用`DenseLayer`时，这也适用于`DenseLayer`本身。

## 示例：PyTorch中的波士顿房价模型

使用这个`Layer`作为构建块，我们可以实现在第[2](ch02.html#fundamentals)章和第[3](ch03.html#deep_learning_from_scratch)章中看到的熟悉的房价模型。回想一下，这个模型只有一个带有`sigmoid`激活函数的隐藏层；在[第3章](ch03.html#deep_learning_from_scratch)中，我们在面向对象的框架中实现了这一点，该框架具有`Layer`的类和一个模型，其`layers`属性是长度为2的列表。类似地，我们可以定义一个`HousePricesModel`类，它继承自`PyTorchModel`，如下所示：

```py
class HousePricesModel(PyTorchModel):

    def __init__(self,
                 hidden_size: int = 13,
                 hidden_dropout: float = 1.0):
        super().__init__()
        self.dense1 = DenseLayer(13, hidden_size,
                                 activation=nn.Sigmoid(),
                                 dropout = hidden_dropout)
        self.dense2 = DenseLayer(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:

        assert_dim(x, 2)

        assert x.shape[1] == 13

        x = self.dense1(x)
        return self.dense2(x)
```

然后我们可以通过以下方式实例化：

```py
pytorch_boston_model = HousePricesModel(hidden_size=13)
```

请注意，为PyTorch模型编写单独的`Layer`类并不是传统的做法；更常见的做法是简单地根据正在发生的各个操作定义模型，使用类似以下的方式：

```py
class HousePricesModel(PyTorchModel):

    def __init__(self,
                 hidden_size: int = 13):
        super().__init__()
        self.fc1 = nn.Linear(13, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:

        assert_dim(x, 2)

        assert x.shape[1] == 13

        x = self.fc1(x)
        x = torch.sigmoid(x)
        return self.fc2(x)
```

在未来构建自己的PyTorch模型时，您可能希望以这种方式编写代码，而不是创建一个单独的`Layer`类——当*阅读*他人的代码时，您几乎总是会看到类似于前面代码的东西。

`Layer`和`Model`比`Optimizer`和`Loss`更复杂，我们将在下一节中介绍。

## PyTorch元素：优化器和损失

`Optimizer`和`Loss`在PyTorch中实现为一行代码。例如，我们在[第4章](ch04.html#extensions)中介绍的`SGDMomentum`损失可以写成：

```py
import torch.optim as optim

optimizer = optim.SGD(pytorch_boston_model.parameters(), lr=0.001)
```

###### 注意

在PyTorch中，模型作为参数传递给`Optimizer`；这确保了优化器“指向”正确的模型参数，以便在每次迭代中知道要更新什么（我们之前使用`Trainer`类做过这个操作）。

此外，我们在[第2章]中看到的均方误差损失和我们在[第4章]中讨论的`SoftmaxCrossEntropyLoss`可以简单地写为：

```py
mean_squared_error_loss = nn.MSELoss()
softmax_cross_entropy_loss = nn.CrossEntropyLoss()
```

与之前的`Layer`一样，这些都继承自`nn.Module`，因此可以像调用`Layer`一样调用它们。

###### 注意

请注意，即使`nn.CrossEntropyLoss`类的名称中没有*softmax*一词，也确实对输入执行了softmax操作，因此我们可以传入神经网络的“原始输出”，而不是已经通过softmax函数的输出，就像我们以前做的那样。

这些`Loss`也继承自`nn.Module`，就像之前的`Layer`一样，因此可以使用相同的方式调用，例如使用`loss(x)`而不是`loss.forward(x)`。

## PyTorch元素：Trainer

`Trainer`将所有这些元素汇集在一起。让我们考虑`Trainer`的要求。我们知道它必须实现我们在本书中多次看到的训练神经网络的一般模式：

1.  通过模型传递一批输入。

1.  将输出和目标输入到损失函数中以计算损失值。

1.  计算损失相对于所有参数的梯度。

1.  使用`Optimizer`根据某种规则更新参数。

使用PyTorch，这一切都是一样的，只是有两个小的实现注意事项：

+   默认情况下，`Optimizer`将在每次参数更新迭代后保留参数的梯度（在本书中我们称之为`param_grads`）。在下一次参数更新之前清除这些梯度，我们将调用`self.optim.zero_grad`。

+   正如在简单的自动微分示例中所示，为了启动反向传播，我们在计算损失值后必须调用`loss.backward`。

这导致了在PyTorch训练循环中看到的以下代码序列，实际上将在`PyTorchTrainer`类中使用。与之前章节中的`Trainer`类一样，`PyTorchTrainer`将接收一个`Optimizer`，一个`PyTorchModel`和一个`Loss`（可以是`nn.MSELoss`或`nn.CrossEntropyLoss`）用于数据批次`(X_batch, y_batch)`；将这些对象放置为`self.optim`，`self.model`和`self.loss`，以下五行代码训练模型：

```py
# First, zero the gradients
self.optim.zero_grad()

# feed X_batch through the model
output = self.model(X_batch)

# Compute the loss
loss = self.loss(output, y_batch)

# Call backward on the loss to kick off backpropagation
loss.backward()

# Call self.optim.step() (as before) to update the parameters
self.optim.step()
```

这些是最重要的行；但是，这是`PyTorchTrainer`的其余代码，其中许多与我们在之前章节中看到的`Trainer`的代码相似：

```py
class PyTorchTrainer(object):
    def __init__(self,
                 model: PyTorchModel,
                 optim: Optimizer,
                 criterion: _Loss):
        self.model = model
        self.optim = optim
        self.loss = criterion
        self._check_optim_net_aligned()

    def _check_optim_net_aligned(self):
        assert self.optim.param_groups[0]['params']\
        == list(self.model.parameters())

    def _generate_batches(self,
                          X: Tensor,
                          y: Tensor,
                          size: int = 32) -> Tuple[Tensor]:

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch

    def fit(self, X_train: Tensor, y_train: Tensor,
            X_test: Tensor, y_test: Tensor,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32):

        for e in range(epochs):
            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self._generate_batches(X_train, y_train,
                                                     batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.optim.zero_grad()
                output = self.model(X_batch)
                loss = self.loss(output, y_batch)
                loss.backward()
                self.optim.step()

            output = self.model(X_test)
            loss = self.loss(output, y_test)
            print(e, loss)
```

###### 注意

由于我们将`Model`，`Optimizer`和`Loss`传入`Trainer`，我们需要检查`Optimizer`引用的参数实际上是否与模型的参数相同；`_check_optim_net_aligned`会执行此操作。

现在训练模型就像这样简单：

```py
net = HousePricesModel()
optimizer = optim.SGD(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

trainer = PyTorchTrainer(net, optimizer, criterion)

trainer.fit(X_train, y_train, X_test, y_test,
            epochs=10,
            eval_every=1)
```

这段代码几乎与我们在前三章中使用的训练模型的代码完全相同。无论您使用PyTorch、TensorFlow还是Theano作为底层，训练深度学习模型的要素都是相同的！

接下来，我们将通过展示如何实现我们在[第4章]中看到的改进训练的技巧来探索PyTorch的更多特性。

## 在PyTorch中优化学习的技巧

我们学到了四个加速学习的技巧[第4章]：

+   动量

+   Dropout

+   权重初始化

+   学习率衰减

这些在PyTorch中都很容易实现。例如，要在我们的优化器中包含动量，我们可以简单地在`SGD`中包含一个`momentum`关键字，使得优化器变为：

```py
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Dropout同样很容易。就像PyTorch有一个内置的`Module` `nn.Linear(n_in, n_out)`来计算之前的`Dense`层的操作一样，`Module` `nn.Dropout(dropout_prob)`实现了`Dropout`操作，但传入的概率默认情况下是*丢弃*给定神经元的概率，而不是像之前我们的实现中那样保留它。

我们根本不需要担心权重初始化：PyTorch中大多数涉及参数的操作，包括`nn.Linear`，其权重会根据层的大小自动缩放。

最后，PyTorch有一个`lr_scheduler`类，可以用来在各个epoch中衰减学习率。你需要开始的关键导入是`from torch.optim import lr_scheduler`。现在你可以轻松地在任何未来的深度学习项目中使用我们从头开始介绍的这些技术！

# PyTorch中的卷积神经网络

在[第5章](ch05.html#convolution)中，我们系统地介绍了卷积神经网络的工作原理，特别关注多通道卷积操作。我们看到该操作将输入图像的像素转换为组织成特征图的神经元层，其中每个神经元表示图像中该位置是否存在给定的视觉特征（由卷积滤波器定义）。多通道卷积操作对其两个输入和输出具有以下形状：

+   数据输入形状`[batch_size, in_channels, image_height, image_width]`

+   参数输入形状`[in_channels, out_channels, filter_size, filter_size]`

+   输出形状`[batch_size, out_channels, image_height, image_width]`

根据这种表示法，PyTorch中的多通道卷积操作是：

```py
nn.Conv2d(in_channels, out_channels, filter_size)
```

有了这个定义，将`ConvLayer`包装在这个操作周围就很简单了：

```py
class ConvLayer(PyTorchLayer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 filter_size: int,
                 activation: nn.Module = None,
                 flatten: bool = False,
                 dropout: float = 1.0) -> None:
        super().__init__()

        # the main operation of the layer
        self.conv = nn.Conv2d(in_channels, out_channels, filter_size,
                              padding=filter_size // 2)

        # the same "activation" and "flatten" operations from before
        self.activation = activation
        self.flatten = flatten
        if dropout < 1.0:
            self.dropout = nn.Dropout(1 - dropout)

    def forward(self, x: Tensor) -> Tensor:

        # always apply the convolution operation
        x = self.conv(x)

        # optionally apply the convolution operation
        if self.activation:
            x = self.activation(x)
        if self.flatten:
            x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        return x
```

###### 注意

在[第5章](ch05.html#convolution)中，我们根据滤波器大小自动填充输出，以保持输出图像与输入图像相同大小。PyTorch不会这样做；为了实现之前的相同行为，我们在`nn.Conv2d`操作中添加一个参数`padding = filter_size // 2`。

然后，我们只需在`__init__`函数中定义一个`PyTorchModel`及其操作，并在`forward`函数中定义操作序列，即可开始训练。下面是一个简单的架构，我们可以在第[4](ch04.html#extensions)章和第[5](ch05.html#convolution)章中看到的MNIST数据集上使用：

+   一个将输入从1个“通道”转换为16个通道的卷积层

+   另一个层，将这16个通道转换为8个（每个通道仍然包含28×28个神经元）

+   两个全连接层

几个卷积层后跟少量全连接层的模式对于卷积架构是常见的；在这里，我们只使用了两个：

```py
class MNIST_ConvNet(PyTorchModel):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(1, 16, 5, activation=nn.Tanh(),
                               dropout=0.8)
        self.conv2 = ConvLayer(16, 8, 5, activation=nn.Tanh(), flatten=True,
                               dropout=0.8)
        self.dense1 = DenseLayer(28 * 28 * 8, 32, activation=nn.Tanh(),
                                 dropout=0.8)
        self.dense2 = DenseLayer(32, 10)

    def forward(self, x: Tensor) -> Tensor:
        assert_dim(x, 4)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.dense1(x)
        x = self.dense2(x)
        return x
```

然后我们可以像训练`HousePricesModel`一样训练这个模型：

```py
model = MNIST_ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

trainer = PyTorchTrainer(model, optimizer, criterion)

trainer.fit(X_train, y_train,
            X_test, y_test,
            epochs=5,
            eval_every=1)
```

与`nn.CrossEntropyLoss`类相关的一个重要注意事项。回想一下，在前几章的自定义框架中，我们的`Loss`类期望输入与目标的形状相同。为了实现这一点，我们对MNIST数据中目标的10个不同值进行了独热编码，以便对于每批数据，目标的形状为`[batch_size, 10]`。

使用PyTorch的`nn.CrossEntropyLoss`类——其与我们之前的`SoftmaxCrossEntropyLoss`完全相同——我们不需要这样做。这个损失函数期望两个`Tensor`：

+   一个大小为`[batch_size, num_classes]`的预测`Tensor`，就像我们的`SoftmaxCrossEntropyLoss`类之前所做的那样

+   一个大小为`[batch_size]`的目标`Tensor`，具有`num_classes`个不同的值

因此，在前面的示例中，`y_train`只是一个大小为`[60000]`的数组（MNIST训练集中的观测数量），而`y_test`只是大小为`[10000]`的数组（测试集中的观测数量）。

现在我们正在处理更大的数据集，我们应该涵盖另一个最佳实践。将整个训练和测试集加载到内存中训练模型显然非常低效，就像我们现在使用的`X_train`、`y_train`、`X_test`和`y_test`一样。PyTorch有一种解决方法：`DataLoader`类。

## DataLoader和Transforms

回想一下，在[第2章](ch02.html#fundamentals)中的MNIST建模中，我们对MNIST数据应用了一个简单的预处理步骤，减去全局均值并除以全局标准差，以粗略“规范化”数据：

```py
X_train, X_test = X_train - X_train.mean(), X_test - X_train.mean()
X_train, X_test = X_train / X_train.std(), X_test / X_train.std()
```

然而，这要求我们首先完全将这两个数组读入内存；在将批次馈送到神经网络时，执行此预处理将更加高效。PyTorch具有内置函数来执行此操作，特别是在处理图像数据时经常使用——通过`transforms`模块进行转换，以及通过`torch.utils.data`进行`DataLoader`：

```py
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
```

之前，我们通过以下方式将整个训练集读入`X_train`中：

```py
mnist_trainset = MNIST(root="../data/", train=True)
X_train = mnist_trainset.train_data
```

然后，我们对`X_train`执行转换，使其准备好进行建模。

PyTorch有一些方便的函数，允许我们将许多转换组合到每个读入的数据批次中；这使我们既可以避免将整个数据集读入内存，又可以使用PyTorch的转换。

我们首先定义要对读入的每批数据执行的转换列表。例如，以下转换将每个MNIST图像转换为`Tensor`（大多数PyTorch数据集默认为“PIL图像”，因此`transforms.ToTensor()`通常是列表中的第一个转换），然后使用整体MNIST均值和标准差`0.1305`和`0.3081`对数据集进行“规范化”——先减去均值，然后除以标准差：

```py
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1305,), (0.3081,))
])
```

###### 注意

`Normalize`实际上是从输入图像的每个通道中减去均值和标准差。因此，当处理具有三个输入通道的彩色图像时，通常会有一个`Normalize`转换，其中包含两个三个数字的元组，例如`transforms.Normalize((0.1, 0.3, 0.6), (0.4, 0.2, 0.5))`，这将告诉`DataLoader`：

+   使用均值为0.1和标准差为0.4来规范化第一个通道

+   使用均值为0.3和标准差为0.2来规范化第二个通道

+   使用均值为0.6和标准差为0.5来规范化第三个通道

其次，一旦应用了这些转换，我们将其应用于读入的批次的`dataset`：

```py
dataset = MNIST("../mnist_data/", transform=img_transforms)
```

最后，我们可以定义一个`DataLoader`，它接收这个数据集并定义了连续生成数据批次的规则：

```py
dataloader = DataLoader(dataset, batch_size=60, shuffle=True)
```

然后，我们可以修改`Trainer`以使用`dataloader`生成用于训练网络的批次，而不是将整个数据集加载到内存中，然后手动使用`batch_generator`函数生成它们，就像我们之前做的那样。在[书的网站](https://oreil.ly/2N4H8jz)上，我展示了使用这些`DataLoader`训练卷积神经网络的示例。`Trainer`中的主要变化只是改变了这一行：

```py
for X_batch, y_batch in enumerate(batch_generator):
```

到：

```py
for X_batch, y_batch in enumerate(train_dataloader):
```

此外，我们现在不再将整个训练集馈送到`fit`函数中，而是馈送`DataLoader`：

```py
trainer.fit(train_dataloader = train_loader,
            test_dataloader = test_loader,
            epochs=1,
            eval_every=1)
```

使用这种架构并调用`fit`方法，就像我们刚刚做的那样，在一个epoch后我们可以达到大约97%的MNIST准确率。然而，比准确率更重要的是，您已经看到了如何将我们从第一原则推理出的概念实现到高性能框架中。现在您既了解了基本概念又了解了框架，我鼓励您修改[书的GitHub存储库](https://oreil.ly/2N4H8jz)中的代码，并尝试其他卷积架构、其他数据集等。

CNN是我们在本书中之前介绍的两种高级架构之一；现在让我们转向另一种，并展示如何在PyTorch中实现我们介绍过的最先进的RNN变体之一，即LSTMs。

## PyTorch中的LSTMs

在上一章中，我们看到了如何从头开始编写LSTMs。我们编写了一个`LSTMLayer`，接受大小为[`batch_size`, `sequence_length`, `feature_size`]的输入`ndarray`，并输出大小为[`batch_size`, `sequence_length`, `feature_size`]的`ndarray`。此外，每个层接受一个隐藏状态和一个单元状态，每个状态初始化为形状`[1, hidden_size]`，当传入一个批次时，扩展为形状`[batch_size, hidden_size]`，然后在迭代完成后缩小回`[1, hidden_size]`。

基于这一点，我们为我们的`LSTMLayer`定义`__init__`方法如下：

```py
class LSTMLayer(PyTorchLayer):
    def __init__(self,
                 sequence_length: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.h_init = torch.zeros((1, hidden_size))
        self.c_init = torch.zeros((1, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = DenseLayer(hidden_size, output_size)
```

与卷积层一样，PyTorch有一个`nn.lstm`操作用于实现LSTMs。请注意，在我们自定义的`LSTMLayer`中，我们将一个`DenseLayer`存储在`self.fc`属性中。您可能还记得上一章中，LSTM单元的最后一步是将最终隐藏状态通过`Dense`层的操作（权重乘法和偏置加法）转换为每个操作的维度`output_size`。PyTorch的做法有点不同：`nn.lstm`操作只是简单地输出每个时间步的隐藏状态。因此，为了使我们的`LSTMLayer`能够输出与其输入不同的维度 - 正如我们希望所有的神经网络层都能够做到的那样 - 我们在最后添加一个`DenseLayer`来将隐藏状态转换为维度`output_size`。

通过这种修改，`forward`函数现在变得简单明了，看起来与[第6章](ch06.html#recurrent)中的`LSTMLayer`的`forward`函数相似：

```py
def forward(self, x: Tensor) -> Tensor:

    batch_size = x.shape[0]

    h_layer = self._transform_hidden_batch(self.h_init,
                                           batch_size,
                                           before_layer=True)
    c_layer = self._transform_hidden_batch(self.c_init,
                                           batch_size,
                                           before_layer=True)

    x, (h_out, c_out) = self.lstm(x, (h_layer, c_layer))

    self.h_init, self.c_init = (
        self._transform_hidden_batch(h_out,
                                     batch_size,
                                     before_layer=False).detach(),
        self._transform_hidden_batch(c_out,
                                     batch_size,
                                     before_layer=False).detach()
                                    )

    x = self.fc(x)

    return x
```

这里的关键一行，应该看起来很熟悉，因为我们在[第6章](ch06.html#recurrent)中实现了LSTMs：

```py
x, (h_out, c_out) = self.lstm(x, (h_layer, c_layer))
```

除此之外，在`self.lstm`函数之前和之后对隐藏状态和单元状态进行一些重塑，通过一个辅助函数`self._transform_hidden_batch`。您可以在[书的GitHub存储库](https://oreil.ly/2N4H8jz)中看到完整的函数。

最后，将模型包装起来很容易：

```py
class NextCharacterModel(PyTorchModel):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int = 256,
                 sequence_length: int = 25):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        # In this model, we have only one layer,
        # with the same output size as input_size
        self.lstm = LSTMLayer(self.sequence_length,
                              self.vocab_size,
                              hidden_size,
                              self.vocab_size)

    def forward(self,
                inputs: Tensor):
        assert_dim(inputs, 3) # batch_size, sequence_length, vocab_size

        out = self.lstm(inputs)

        return out.permute(0, 2, 1)
```

###### 注意

`nn.CrossEntropyLoss`函数期望前两个维度是`batch_size`和类别分布；然而，我们一直在实现LSTMs时，类别分布作为最后一个维度（`vocab_size`）从`LSTMLayer`中出来。因此，为了准备最终模型输出以输入损失，我们使用`out.permute(0, 2, 1)`将包含字母分布的维度移动到第二维度。

最后，在[书的GitHub存储库](https://oreil.ly/2N4H8jz)中，我展示了如何编写一个从`PyTorchTrainer`继承的`LSTMTrainer`类，并使用它来训练`NextCharacterModel`生成文本。我们使用了与[第6章](ch06.html#recurrent)中相同的文本预处理：选择文本序列，对字母进行独热编码，将独热编码的文本序列分组成批次。

这就是如何将本书中看到的三种用于监督学习的神经网络架构 - 全连接神经网络、卷积神经网络和循环神经网络 - 转换为PyTorch。最后，我们将简要介绍神经网络如何用于机器学习的另一半：*非*监督学习。

# 附言：通过自动编码器进行无监督学习

在本书中，我们一直专注于深度学习模型如何用于解决*监督*学习问题。当然，机器学习还有另一面：无监督学习；这通常被描述为“在没有标签的数据中找到结构”; 我更喜欢将其看作是在数据中找到尚未被测量的特征之间的关系，而监督学习涉及在数据中找到已经被测量的特征之间的关系。

假设你有一个没有标签的图像数据集。你对这些图像了解不多——例如，你不确定是否有10个不同的数字，或者5个，或者20个（这些图像可能来自一个奇怪的字母表）——你想知道这样的问题的答案：

+   有多少个不同的数字？

+   哪些数字在视觉上相似？

+   是否有与其他图像明显*不*相似的“异常”图像？

要理解深度学习如何帮助解决这个问题，我们需要快速退后一步，从概念上思考深度学习模型试图做什么。

## 表示学习

我们已经看到深度学习模型可以学习进行准确的预测。它们通过将接收到的输入转换为逐渐更抽象且更直接用于解决相关问题的表示来实现这一点。特别是，在网络的最终层，直接在具有预测本身的层之前（对于回归问题只有一个神经元，对于分类问题有*`num_classes`*个神经元），网络试图创建一个对于预测任务尽可能有用的输入数据表示。这在[图7-1](#fig_07_01)中显示。

![](assets/dlfs_0701.png)

###### 图7-1\. 神经网络的最终层，在预测之前，表示网络对于预测任务发现的输入的表示

一旦训练完成，模型不仅可以为新数据点做出预测，*还可以生成这些数据点的表示*。这些表示可以用于聚类、相似性分析或异常检测，除了预测。

## 一种没有任何标签的情况下的方法

这种整体方法的一个限制是*需要标签来训练模型首先生成表示*。问题是：如何训练模型生成“有用”的表示而没有任何标签？如果没有标签，我们需要使用唯一拥有的东西——训练数据本身——来生成数据的表示。这就是一类被称为自动编码器的神经网络架构的理念，它涉及训练神经网络*重建*训练数据，迫使网络学习对于这种重建最有帮助的每个数据点的表示。

### 图表

[图7-2](#fig_07_02)展示了自动编码器的高级概述：

1.  一组层将数据转换为数据的压缩表示。

1.  另一组层将这个表示转换为与原始数据相同大小和形状的输出。

![](assets/dlfs_0702.png)

###### 图7-2\. 自动编码器有一组层（可以被视为“编码器”网络），将输入映射到一个低维表示，另一组层（可以被视为“解码器”网络）将低维表示映射回输入；这种结构迫使网络学习一个对于重建输入最有用的低维表示

实现这样的架构展示了一些我们还没有机会介绍的PyTorch特性。

## 在PyTorch中实现自动编码器

我们现在将展示一个简单的自动编码器，它接收输入图像，通过两个卷积层然后一个`Dense`层生成一个表示，然后将这个表示再通过一个`Dense`层和两个卷积层传递回来，生成与输入相同大小的输出。我们将使用这个示例来说明在PyTorch中实现更高级架构时的两种常见做法。首先，我们可以将`PyTorchModel`作为另一个`PyTorchModel`的属性包含，就像我们之前将`PyTorchLayer`定义为这些模型的属性一样。在以下示例中，我们将实现我们的自动编码器，将两个`PyTorchModel`作为属性：一个`Encoder`和一个`Decoder`。一旦我们训练模型，我们将能够使用训练好的`Encoder`作为自己的模型来生成表示。

我们将`Encoder`定义为：

```py
class Encoder(PyTorchModel):
    def __init__(self,
                 hidden_dim: int = 28):
        super(Encoder, self).__init__()
        self.conv1 = ConvLayer(1, 14, activation=nn.Tanh())
        self.conv2 = ConvLayer(14, 7, activation=nn.Tanh(), flatten=True)

        self.dense1 = DenseLayer(7 * 28 * 28, hidden_dim, activation=nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        assert_dim(x, 4)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense1(x)

        return x
```

我们将`Decoder`定义为：

```py
class Decoder(PyTorchModel):
    def __init__(self,
                 hidden_dim: int = 28):
        super(Decoder, self).__init__()
        self.dense1 = DenseLayer(hidden_dim, 7 * 28 * 28, activation=nn.Tanh())

        self.conv1 = ConvLayer(7, 14, activation=nn.Tanh())
        self.conv2 = ConvLayer(14, 1, activation=nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        assert_dim(x, 2)

        x = self.dense1(x)

        x = x.view(-1, 7, 28, 28)
        x = self.conv1(x)
        x = self.conv2(x)

        return x
```

###### 注意

如果我们使用的步幅大于1，我们将无法简单地使用常规卷积将编码转换为输出，而是必须使用*转置卷积*，其中操作的输出图像大小将大于输入图像的大小。有关更多信息，请参阅[PyTorch文档](https://oreil.ly/306qiV7)中的`nn.ConvTranspose2d`操作。

然后`Autoencoder`本身可以包裹这些并变成：

```py
class Autoencoder(PyTorchModel):
    def __init__(self,
                 hidden_dim: int = 28):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(hidden_dim)

        self.decoder = Decoder(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        assert_dim(x, 4)

        encoding = self.encoder(x)
        x = self.decoder(encoding)

        return x, encoding
```

`Autoencoder`的`forward`方法展示了PyTorch中的第二种常见做法：由于我们最终想要看到模型生成的隐藏表示，`forward`方法返回*两个*元素：这个“编码”，`encoding`，以及将用于训练网络的输出，`x`。

当然，我们需要修改我们的`Trainer`类以适应这一点；具体来说，`PyTorchModel`目前只从其`forward`方法中输出单个`Tensor`。事实证明，将其修改为默认返回`Tensor`的`Tuple`，即使该`Tuple`只有长度为1，将会非常有用——使我们能够轻松编写像`Autoencoder`这样的模型，并且不难。我们只需要做三件小事：首先，将我们的基本`PyTorchModel`类的`forward`方法的函数签名修改为：

```py
def forward(self, x: Tensor) -> Tuple[Tensor]:
```

然后，在任何继承自`PyTorchModel`基类的模型的`forward`方法末尾，我们将写`return x,`而不是之前的`return x`。

其次，我们将修改我们的`Trainer`，始终将模型返回的第一个元素作为输出：

```py
output = self.model(X_batch)[0]
...
output = self.model(X_test)[0]
```

`Autoencoder`模型的另一个显著特点是：我们对最后一层应用了`Tanh`激活函数，这意味着模型输出将在-1和1之间。对于任何模型，模型输出应该与其进行比较的目标在相同的尺度上，这里，目标是我们的输入本身。因此，我们应该将我们的输入缩放到-1的最小值和1的最大值，如下面的代码所示：

```py
X_train_auto = (X_train - X_train.min())
                / (X_train.max() - X_train.min()) * 2 - 1
X_test_auto = (X_test - X_train.min())
                / (X_train.max() - X_train.min()) * 2 - 1
```

最后，我们可以使用训练代码训练我们的模型，到目前为止应该看起来很熟悉（我们将28任意地用作编码输出的维度）：

```py
model = Autoencoder(hidden_dim=28)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

trainer = PyTorchTrainer(model, optimizer, criterion)

trainer.fit(X_train_auto, X_train_auto,
            X_test_auto, X_test_auto,
            epochs=1,
            batch_size=60)
```

一旦我们运行这段代码并训练模型，我们可以通过将`X_test_auto`通过模型（因为`forward`方法被定义为返回两个量）来查看重建图像和图像表示：

```py
reconstructed_images, image_representations = model(X_test_auto)
```

`reconstructed_images`的每个元素是一个`[1, 28, 28]`的`Tensor`，表示神经网络尝试重建对应原始图像后的最佳结果，通过将其通过一个具有较低维度的层的自动编码器架构。[图7-3](#fig_07_03)显示了随机选择的重建图像与原始图像并排。

![](assets/dlfs_0703.png)

###### 图7-3。来自MNIST测试集的图像以及通过自动编码器传递后的图像的重建

从视觉上看，这些图像看起来相似，告诉我们神经网络确实似乎已经将原始图像（784像素）映射到了一个较低维度的空间——具体来说是28——这样大部分关于784像素图像的信息都被编码在这个长度为28的向量中。我们如何检查整个数据集，以查看神经网络是否确实学习了图像数据的结构而没有看到标签呢？嗯，“数据的结构”在这里意味着底层数据实际上是10个不同手写数字的图像。因此，在新的28维空间中，接近给定图像的图像理想情况下应该是相同的数字，或者至少在视觉上非常相似，因为视觉相似性是我们作为人类区分不同图像的方式。我们可以通过应用Laurens van der Maaten在Geoffrey Hinton的指导下作为研究生时发明的降维技术*t-分布随机邻居嵌入*（t-SNE）来测试是否符合这种情况。t-SNE以类似于神经网络训练的方式进行降维：它从一个初始的低维表示开始，然后更新它，以便随着时间的推移，它接近具有这样的属性的解决方案，即在高维空间中“靠在一起”的点在低维空间中也是“靠在一起”，反之亦然。

我们将尝试以下操作：

+   将这10,000个图像通过t-SNE并将维度降低到2。

+   将生成的两维空间可视化，通过它们的*实际*标签对不同的点进行着色（自动编码器没有看到）。

[图7-4](#fig_07_04)显示了结果。

![](assets/dlfs_0704.png)

###### 图7-4。在自动编码器的28维学习空间上运行t-SNE的结果

似乎每个数字的图像主要被分组在自己的独立簇中；这表明训练我们的自动编码器架构学习仅从较低维度表示中重建原始图像确实使其能够发现这些图像的底层结构的大部分，而不需要看到任何标签。不仅10个数字被表示为不同的簇，而且视觉上相似的数字也更接近：在顶部稍微向右，我们有数字3、5和8的簇，底部我们看到4和9紧密聚集在一起，7也不远。最后，最不同的数字—0、1和6—形成最不同的簇。

## 无监督学习的更强测试，以及解决方案

我们刚刚看到的是一个相当弱的测试，用于检查我们的模型是否已经学习了输入图像空间的底层结构——到这一点，一个卷积神经网络可以学习图像的表示，使得视觉上相似的图像具有相似的表示，这应该不会太令人惊讶。一个更强的测试是检查神经网络是否发现了一个“平滑”的底层空间：一个空间，其中*任何*长度为28的向量，而不仅仅是通过编码器网络传递真实数字得到的向量，都可以映射到一个看起来像真实数字的图像。事实证明，我们的自动编码器无法做到这一点；[图7-5](#fig_07_05)显示了生成五个长度为28的随机向量并通过解码器网络传递它们的结果，利用了`Autoencoder`包含`Decoder`作为属性的事实：

```py
test_encodings = np.random.uniform(low=-1.0, high=1.0, size=(5, 28))
test_imgs = model.decoder(Tensor(test_encodings))
```

![](assets/dlfs_0705.png)

###### 图7-5。通过解码器传递五个随机生成的向量的结果

您可以看到生成的图像看起来不像数字；因此，虽然我们的自动编码器可以以合理的方式将数据映射到较低维度空间，但似乎无法学习一个像前面描述的“平滑”空间。

解决问题，即训练神经网络学习在训练集中表示图像的“平滑”基础空间，是*生成对抗网络*（GANs）的主要成就之一。GANs于2014年发明，最广为人知的是通过同时训练两个神经网络的训练过程，使神经网络能够生成看起来逼真的图像。然而，真正推动GANs的发展是在2015年，当研究人员将它们与深度卷积架构一起使用时，不仅生成了看起来逼真的64×64彩色卧室图像，还从随机生成的100维向量中生成了大量这样的图像样本。这表明神经网络确实已经学会了这些未标记图像的“空间”的基本表示。GANs值得有一本专门的书来介绍，所以我们不会详细介绍它们。

# 结论

现在，您对一些最流行的先进深度学习架构的机制有了深入的了解，以及如何在最流行的高性能深度学习框架之一中实现这些架构。阻止您使用深度学习模型解决实际问题的唯一障碍是实践。幸运的是，阅读他人的代码并迅速掌握使某些模型架构在某些问题上起作用的细节和实现技巧从未如此简单。推荐的下一步列表在书的GitHub存储库中列出（https://oreil.ly/2N4H8jz）。

继续前进！

以这种方式编写`Layer`和`Model`在PyTorch中并不是最常见或推荐的用法；我们在这里展示它，因为它最接近我们迄今为止涵盖的概念。要查看使用PyTorch构建神经网络构建块的更常见方法，请参阅官方文档中的这个入门教程（https://oreil.ly/SKB_V）。

在书的GitHub存储库中，您可以找到一个实现指数学习率衰减的代码示例，作为`PyTorchTrainer`的一部分。在那里使用的`ExponentialLR`类的文档可以在PyTorch网站上找到（https://oreil.ly/2Mj9IhH）。

查看“使用PyTorch的CNNs”部分。

2008年的原始论文是“使用t-SNE可视化数据”（https://oreil.ly/2KIAaOt），由Laurens van der Maaten和Geoffrey Hinton撰写。

此外，我们做到这一点并没有费多少力气：这里的架构非常简单，我们没有使用我们讨论过的训练神经网络的任何技巧，比如学习率衰减，因为我们只训练了一个周期。这说明使用类似自动编码器的架构来学习数据集的结构而不使用标签的基本想法是一个好主意，而不仅仅是在这里“碰巧起作用”。

查看DCGAN论文，“使用深度卷积生成对抗网络进行无监督表示学习”（https://arxiv.org/abs/1511.06434）由Alec Radford等人撰写，以及这个PyTorch文档（https://oreil.ly/2TEspgG）。
