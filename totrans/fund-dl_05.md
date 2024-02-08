# 第五章：在 PyTorch 中实现神经网络

# PyTorch 简介

在本章中，您将学习 PyTorch 的基础知识，这是当今最流行的深度学习框架之一。PyTorch 由 Facebook 的人工智能研究实验室于 2016 年推出，并在随后的几年中迅速获得用户，无论是在工业界还是研究领域。PyTorch 被广泛采用的一个原因是其直观的、Pythonic 的感觉，这与深度学习从业者遵循的现有工作流程和编码范式自然契合。

特别是，本章将讨论 PyTorch 使用的数据结构，如何在 PyTorch 中定义神经模型，以及如何将数据与模型连接起来进行训练和测试。最后，我们在 PyTorch 中实现一个实际的示例——一个用于 MNIST 数字数据集的分类器，包括用于训练和测试分类器的代码。

# 安装 PyTorch

安装与 CPU 兼容的 PyTorch 版本相对简单。PyTorch 文档建议使用 conda，一个包管理系统。在 conda 中，您可以创建多个环境，其中环境是封装所有包安装的上下文。包的访问权限不会跨环境传递，这允许用户通过在各个环境中下载包来实现不同上下文的清晰分离。我们建议您为深度学习目的创建一个 conda 环境，以便在必要时切换到该环境。我们建议您参考 conda 文档，了解如何下载 conda 以及有关环境的进一步说明。

一旦您安装了 conda，创建了深度学习环境并切换到该环境，PyTorch 文档建议从命令行运行以下代码来下载 macOS 上与 CPU 兼容的 PyTorch 版本：

```py
conda install pytorch torchvision torchaudio -c pytorch

```

请注意，随此安装而来的还有 torchvision 和 torchaudio，它们分别是用于处理图像数据和音频数据的专门包。如果您使用的是 Linux 系统，文档建议从命令行中运行以下代码：

```py
conda install pytorch torchvision torchaudio cpuonly -c pytorch

```

现在，您可以转到 Python shell（仍然在您的深度学习环境中），以下命令应该可以无问题运行：

```py
import torch

```

在继续运行以下部分的代码之前，重要的是在 Python shell 中确保此命令无错误运行，因为它们都需要能够导入 PyTorch 包。

# PyTorch 张量

张量是 PyTorch 存储和操作数值信息的主要数据结构。张量可以被看作是数组和矩阵的泛化，我们在第一章中详细介绍了数组和矩阵。具体来说，张量作为 2D 矩阵和 1D 数组的泛化，可以存储多维数据，例如批量的三通道图像。请注意，这需要 4D 数据存储，因为每个图像是 3D 的（包括通道维度），还需要第四个维度来索引每个单独的图像。张量甚至可以表示超过 4D 空间的维度，尽管在实践中使用这样的张量是不常见的。

在 PyTorch 中，张量被广泛使用。它们用于表示模型的输入，模型内部的权重层以及模型的输出。张量上可以运行所有标准的线性代数操作，如转置、加法、乘法、求逆等。

## 张量初始化

我们如何初始化张量？我们可以从各种数据类型初始化张量。一些示例是 Python 列表和 Python 数值原语：

```py
arr = [1,2]
tensor = torch.tensor(arr)
val = 2.0
tensor = torch.tensor(val)

```

张量也可以从 NumPy 数组中初始化，这使得 PyTorch 可以轻松地集成到现有的数据科学和机器学习工作流中：

```py
import numpy as np
np_arr = np.array([1,2])
x_t = torch.from_numpy(np_arr)

```

此外，张量还可以通过一些常见的 PyTorch API 端点来形成：

```py
zeros_t = torch.zeros((2,3)) # Returns 2x3 tensor of zeros
ones_t = torch.ones((2,3)) # Returns 2x3 tensor of ones
rand_t = torch.randn((2,3)) # Returns 2x3 tensor of random numbers

```

## 张量属性

在我们刚刚看到的例子中，我们将一个元组作为每个函数调用的参数传递。元组中的索引数量是要创建的张量的维度，而每个索引处的数字表示该特定维度的期望大小。要访问张量的维度，我们可以调用其形状属性：

```py
zeros_t.shape # Returns torch.Size([2, 3])

```

在任何先前示例中调用形状属性应该返回与输入参数相同的元组，假设张量在此期间没有被显着修改。

张量还有哪些其他属性？除了维度，张量还存储有关存储的数据类型的信息：浮点数、复数、整数和布尔值。在每个类别中存在子类型，但我们不会在这里讨论每个子类型之间的区别。还要注意的是，张量不能包含各种数据类型的混合和匹配-单个张量中的所有数据必须是相同的数据类型。要访问张量的数据类型，我们可以调用其`dtype`属性：

```py
x_t = torch.tensor(2.0)
x_t.dtype # Returns torch.float32

```

此外，尽管我们尚未展示这一点，但我们可以在初始化期间设置张量的数据类型。扩展我们先前的一个示例：

```py
arr = [1,2]
x_t = torch.tensor(arr, dtype=torch.float32)

```

除了张量的数据类型和形状，我们还可以了解张量所分配的设备。这些设备包括著名的 CPU，它是任何计算机的标准设备，也是任何张量的默认存储设备，以及 GPU，即图形处理单元，它是专门用于图像空间的数据处理单元。 GPU 通过在数百个小型专用核心上进行并行处理，大大加快了许多常见张量操作，例如乘法，因此使它们在大多数深度学习应用中非常有用。要访问张量设备，我们可以调用其设备属性：

```py
x_t.device # Returns device(type='cpu') by default

```

与数据类型类似，我们可以在初始化时设置张量的设备：

```py
# PyTorch will use GPU if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
arr = [1,2]
x_t = torch.tensor(arr, dtype=torch.float32, device=device)

```

这是通过代码检查 GPU 是否可用并在可用时使用 GPU 的常见方法。如果 GPU 不可用，它将在没有错误的情况下使用 CPU。

如果您已经定义了具有一组特定属性的张量，并希望修改这些属性，可以使用`to`函数：

```py
x_t = x_t.to(device, dtype=torch.int)

```

最后，正如我们将在“PyTorch 中的梯度”中介绍的那样，PyTorch 张量可以使用参数`requires_grad`进行初始化，当设置为`True`时，将张量的梯度存储在名为`grad`的属性中。

## 张量操作

PyTorch API 为我们提供了许多可能的张量操作，从张量算术到张量索引。在本节中，我们将介绍一些更有用的张量操作-这些操作在深度学习应用中经常使用。

最基本的操作之一是将张量乘以某个标量`c`。这可以通过以下代码实现：

```py
c = 10
x_t = x_t*c

```

这将导致标量与张量条目的逐元素乘积。张量加法和减法是最基本的张量操作之一。要做到这一点，我们可以通过`+`简单地添加张量。减法直接来自能够进行加法并将第二个张量乘以标量-1：

```py
x1_t = torch.zeros((1,2))
x2_t = torch.ones((1,2))
x1_t + x2_t
# returns tensor([[1., 1.]])

```

结果是两个张量的逐元素求和。这可以看作是对于任何维度的矩阵加法的直接推广。请注意，这种直接推广隐含地假设了我们之前讨论过的矩阵加法的相同约束：被求和的两个张量具有相同的维度。类似地，PyTorch 将接受任何两个可广播的输入而不会出现问题，其中广播是一种通过将两个输入解析为共同形状的过程，可广播指的是这两个输入是否能够解析为共同形状。如果两个张量已经具有相同的形状，则不需要广播。有关 API 如何确定两个输入是否可广播以及在这种情况下如何执行广播的更多信息，请参阅[PyTorch 文档](https://oreil.ly/rHEdO)。

张量乘法是另一个有用的操作，值得熟悉。当每个张量的维度小于或等于 2 时，张量乘法与矩阵和向量乘法相同。然而，张量乘法也适用于任意高维度的张量，只要这两个张量是兼容的。我们可以将高维度的张量乘法看作是批量矩阵乘法：想象我们有两个张量，第一个的形状为(2,1,2)，第二个的形状为(2,2,2)。我们可以进一步将第一个张量表示为长度为 2 的 1×2 矩阵列表，而第二个是长度为 2 的 2×2 矩阵列表。它们的乘积是一个长度为 2 的列表，其中产品的索引*i*是第一个张量的索引*i*和第二个张量的索引*i*的矩阵乘积，如图 5-1 所示。

![](img/fdl2_0501.png)

###### 图 5-1。为了帮助可视化一般张量乘法方法，这个图显示了重新堆叠之前发生的矩阵乘法。

将结果列表重新堆叠成一个 3D 张量，我们看到产品的形状为(2,1,2)。现在，我们可以将这个推广到四维，其中我们不再想象有一个矩阵列表，而是将每个 4D 张量表示为矩阵的网格，产品的*(i,j)*索引是两个 4D 输入张量的*(i,j)*索引的矩阵乘积。我们用数学方式表示这一点：

<math alttext="upper P Subscript i comma j comma x comma z Baseline equals sigma-summation Underscript y Endscripts upper A Subscript i comma j comma x comma y Baseline asterisk upper B Subscript i comma j comma y comma z"><mrow><msub><mi>P</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi><mo>,</mo><mi>x</mi><mo>,</mo><mi>z</mi></mrow></msub> <mo>=</mo> <msub><mo>∑</mo> <mi>y</mi></msub> <msub><mi>A</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi><mo>,</mo><mi>x</mi><mo>,</mo><mi>y</mi></mrow></msub> <mo>*</mo> <msub><mi>B</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi></mrow></msub></mrow></math>

这个过程可以推广到任何维度，假设两个输入张量遵循矩阵乘法的约束。与张量加法一样，也存在涉及广播的例外情况，但我们在这里不会详细介绍。有关广播的详细信息，请参阅 PyTorch 文档。要在 PyTorch 中对两个张量进行乘法，您可以使用`torch matmul`函数：

```py
x1_t = torch.tensor([[1,2],[3,4]])
x2_t = torch.tensor([[1,2,3],[4,5,6]])
torch.matmul(x1_t, x2_t) # Returns tensor([[9,12,15],[19,26,33]])

```

除了张量上的算术运算，我们还可以对张量进行索引和切片。如果您之前有使用 NumPy 的经验，您会注意到 PyTorch 的索引非常相似，并且基于线性代数基础。如果您有一个 3D 张量，您可以通过以下代码访问位置*(i,j,k)*处的值：

```py
i,j,k = 0,1,1
x3_t = torch.tensor([[[3,7,9],[2,4,5]],[[8,6,2],[3,9,1]]])
print(x3_t)
# out:
# tensor([[[3, 7, 9],
#          [2, 4, 5]],
#         [[8, 6, 2],
#          [3, 9, 1]]])

x3_t[i,j,k]
# out:
# tensor(4)

```

要访问张量的更大切片，比如在 3D 张量中位置为 0 的矩阵，您可以使用以下代码：

```py
x3_t[0] # Returns the matrix at position 0 in tensor
x3_t[0,:,:] # Also returns the matrix at position 0 in tensor!
# out:
# tensor([[3, 7, 9],
#         [2, 4, 1]])

```

这两行代码被 PyTorch API 解释为等价的。这是因为使用单个索引器，比如`x3_t[0]`，隐含地假设用户想要访问满足条件*i* = 0 的所有索引*(i,j,k)*（即原始 3D 张量中矩阵堆栈中的顶部矩阵）。使用`:`符号使这种隐含假设清晰明了，告诉 PyTorch 直接，用户不想在该维度上对数据进行子集操作。我们还可以使用`:`符号来对数据进行子集操作，例如：

```py
x3_t[0,1:3,:]
# returns tensor([[2, 4, 5]])

```

其中代码的最后一行被解释为：找到所有索引*(i,j,k)*，使得*i* = 0，*j*大于或等于 1，并且*j*小于 3（`:`遵循标准的 Python 列表索引约定，在定义范围的开始处是包含的，结束处是不包含的）。简单来说，我们想要访问原始 3D 张量中堆叠的矩阵的顶部矩阵的第二行和第三行。请注意，这种对`:`的使用与标准的 Python 列表索引一致。

除了访问张量的索引或切片之外，我们还可以将这些索引和切片设置为新值。在单索引的情况下，这很简单：

```py
x3_t[0,1,2] = 1

# out:
# tensor([[[3, 7, 9],
#          [2, 4, 1]],

#         [[8, 6, 2],
#          [3, 9, 1]]])

```

要设置张量的较大切片，最直接的方法是定义一个与切片具有相同维度的张量，并使用以下代码：

```py
x_t = torch.randn(2,3,4)
sub_tensor = torch.randn(2,4)
x_t[0,1:3,:] = sub_tensor

```

此外，通过广播，我们可以做如下操作：

```py
x_t[0,1:3,:] = 1
sub_tensor = torch.randn(1,4)
x_t[0,1:3,:] = sub_tensor

```

第一行将这两行的全部设置为 1，第二行将切片的两行都设置为传入的单行`sub_tensor`。在下一节中，我们将展示如何在 PyTorch 中计算函数的梯度，以及如何访问这些梯度的值。

# PyTorch 中的梯度

作为回顾，让我们回顾一下微积分中的导数和偏导数。函数的偏导数，可以是简单的多变量多项式函数，也可以是复杂的神经网络，相对于函数的一个输入表示函数输出随着该输入值稍微变化的变化速率。因此，大幅度的导数表示输出在输入稍微变化时非常不稳定（当*x*是适度大小时，考虑*f(x) = x¹⁰*），而小幅度的导数表示输出在输入稍微变化时相对稳定（考虑*f(x) = x/10*）。如果函数接受多个输入，则梯度是由所有这些偏导数组成的向量：

<math alttext="f left-parenthesis x comma y comma z right-parenthesis equals x squared plus y squared plus z squared"><mrow><mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>=</mo> <msup><mi>x</mi> <mn>2</mn></msup> <mo>+</mo> <msup><mi>y</mi> <mn>2</mn></msup> <mo>+</mo> <msup><mi>z</mi> <mn>2</mn></msup></mrow></math>

<math alttext="StartFraction normal partial-differential f Over normal partial-differential x EndFraction equals normal nabla Subscript x Baseline f left-parenthesis x comma y comma z right-parenthesis equals 2 x"><mrow><mfrac><mrow><mi>∂</mi><mi>f</mi></mrow> <mrow><mi>∂</mi><mi>x</mi></mrow></mfrac> <mo>=</mo> <msub><mi>∇</mi> <mi>x</mi></msub> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>=</mo> <mn>2</mn> <mi>x</mi></mrow></math>

<math alttext="normal nabla f equals left-bracket 2 x Baseline 2 y Baseline 2 z right-bracket"><mrow><mi>∇</mi> <mi>f</mi> <mo>=</mo> <mo>[</mo> <mn>2</mn> <mi>x</mi> <mn>2</mn> <mi>y</mi> <mn>2</mn> <mi>z</mi> <mo>]</mo></mrow></math>

继续这个例子，我们如何在 PyTorch 中表示这个呢？我们可以使用以下代码：

```py
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(1.5, requires_grad=True)
f = x**2+y**2+z**2
f.backward()
x.grad, y.grad, z.grad
# out:
# (tensor(4.), tensor(6.), tensor(3.))

```

调用`backward()`计算输出*f*相对于每个输入变量的偏导数。我们应该期望`x.grad`，`y.grad`和`z.grad`的值分别为 4.0，6.0 和 3.0。在神经网络的情况下，我们可以将神经网络表示为*f(x,θ)*，其中*f*是神经网络，*x*是表示输入的向量，*θ*是*f*的参数。与前面的例子中计算*f*输出相对于*x*的梯度不同，我们计算*f*输出的损失相对于*θ*的梯度。通过梯度调整*θ*最终将导致*θ*的设置使得训练数据的损失很小，并且希望泛化到*f*之前没有见过的数据。在下一节中，我们将介绍神经网络的构建模块。

# PyTorch nn 模块

PyTorch 的`nn`模块提供了定义、训练和测试模型所需的所有基本功能。要导入`nn`模块，您只需要运行以下代码行：

```py
 import torch.nn as nn

```

在本节中，我们将介绍`nn`模块的一些最常见用法。例如，要初始化前馈神经网络所需的权重矩阵，可以使用以下代码：

```py
in_dim, out_dim = 256, 10
vec = torch.randn(256)
layer = nn.Linear(in_dim, out_dim, bias=True)
out = layer(vec)

```

这定义了一个带有偏置的前馈神经网络中的单层，它是一个权重矩阵，接受维度为 256 的向量作为输入，并输出维度为 10 的向量。代码的最后一行演示了如何将这一层轻松应用于输入向量，并将输出存储在一个新的张量中。如果我们想要仅使用我们之前章节的知识来做同样的事情，我们需要手动定义一个权重矩阵`W`和偏置向量`b`，通过`torch`.tensor 并显式计算：

```py
W = torch.rand(10,256)
b = torch.zeros(10,1)
out = torch.matmul(W, vec) + b

```

`nn`模块的 Linear 层允许我们将这些手动操作抽象化，以便编写干净、简洁的代码。

前馈神经网络可以简单地被视为这些层的组合，例如：

```py
in_dim, feature_dim, out_dim = 784, 256, 10
vec = torch.randn(784)
layer1 = nn.Linear(in_dim, feature_dim, bias=True)
layer2 = nn.Linear(feature_dim, out_dim, bias=True)
out = layer2(layer1(vec))

```

这段代码表示一个神经网络，即函数组合`layer2(layer1(vec))`，或者数学上：<math alttext="upper W 2 left-parenthesis upper W 1 asterisk x plus b 1 right-parenthesis plus b 2"><mrow><msub><mi>W</mi> <mn>2</mn></msub> <mrow><mo>(</mo> <msub><mi>W</mi> <mn>1</mn></msub> <mo>*</mo> <mi>x</mi> <mo>+</mo> <msub><mi>b</mi> <mn>1</mn></msub> <mo>)</mo></mrow> <mo>+</mo> <msub><mi>b</mi> <mn>2</mn></msub></mrow></math>。为了表示更复杂、非线性的函数，`nn`模块还提供了诸如 ReLU（通过`nn.ReLU`访问）和 tanh（通过`nn.Tanh`访问）等非线性。这些非线性应用在层之间，如下所示：

```py
relu = nn.ReLU()
out  = layer2(relu(layer1(vec)))

```

我们已经几乎涵盖了在 PyTorch 中定义模型所需的一切。最后要介绍的是`nn.Module`类——所有神经网络在 PyTorch 中都是从这个基类继承的。

`nn.Module`类有一个重要的方法，你特定模型的子类将覆盖这个方法。这个方法是前向方法，它定义了在模型的构造函数中初始化的层如何与输入交互以生成模型的输出。这里是一个可以用来封装我们刚刚定义的简单两层神经网络的代码示例：

```py
class BaseClassifier(nn.Module):
  def __init__(self, in_dim, feature_dim, out_dim):
    super(BaseClassifier, self).__init__()
    self.layer1 = nn.Linear(in_dim, feature_dim, bias=True)
    self.layer2 = nn.Linear(feature_dim, out_dim, bias=True)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.layer1(x)
    x = self.relu(x)
    out = self.layer2(x)
    return out

```

我们已经在 PyTorch 中编写了我们的第一个神经网络！`BaseClassifier`是一个无 bug 的模型类，可以在定义`in_dim`、`feature_dim`和`out_dim`之后实例化。构造函数将这三个变量作为参数传入构造函数，这使得模型在层大小方面更加灵活。这种模型可以有效地用作诸如 MNIST 之类的数据集的第一次分类器，正如我们将在“在 PyTorch 中构建 MNIST 分类器”中演示的那样。要在某些输入上生成模型的输出，我们可以按照以下方式使用模型：

```py
no_examples = 10
in_dim, feature_dim, out_dim = 784, 256, 10
x = torch.randn((no_examples, in_dim))
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
out = classifier(x)

```

请注意，在最后一行中将分类器模型作为函数使用时，我们隐式调用了前向函数。与最初手动定义每个层参数并通过`matmul`操作计算输出的方法相比，这是一种更加清晰、模块化和可重用的定义神经网络的方法。

除了能够定义模型、实例化它并通过数据运行它之外，我们还必须能够训练和测试模型。要训练（和测试）模型，我们需要一个损失度量来评估模型。在训练过程中，一旦计算了这个损失度量，我们可以利用上一节的知识，并在计算的损失上调用`backward()`。这将把每个参数 p 的梯度存储在`grad`属性中。由于我们已经定义了一个分类器模型，我们可以使用 PyTorch `nn`中的交叉熵`loss`度量：

```py
loss = nn.CrossEntropyLoss()
target = torch.tensor([0,3,2,8,2,9,3,7,1,6])
computed_loss = loss(out, target)
computed_loss.backward()

```

在上述代码中，`target`是一个形状为(`no_examples`)的张量，每个索引表示与该索引对应的输入的地面真实类。现在我们已经计算出了相对于分类器中所有参数的小批量示例的损失的梯度，我们可以执行梯度下降步骤。当将神经网络定义为`nn.Module`的子类时，我们可以通过`parameters()`函数访问其所有参数——这是 PyTorch API 提供的另一个便利。要查看神经网络中每个参数的形状，您可以运行以下代码：

```py
for p in classifier.parameters():
  print(p.shape)

# out:
# torch.Size([256, 784])
# torch.Size([256])
# torch.Size([10, 256])
# torch.Size([10])

```

正如我们所看到的，第一层有 256×784 个权重和长度为 256 的偏置向量。最后一层有 10×256 个权重和长度为 10 的偏置向量。

在梯度下降过程中，我们需要根据梯度调整参数。我们可以手动执行此操作，但 PyTorch 将此功能抽象为`torch.optim`模块。该模块提供了确定优化器的功能，这可能比经典梯度下降更复杂，并更新模型的参数。您可以按照以下方式定义优化器：

```py
from torch import optim

lr = 1e-3
optimizer = optim.SGD(classifier.parameters(), lr=lr) 

```

这段代码创建了一个优化器，它将在每个小批量结束时通过 SGD 更新分类器的参数。要实际执行此更新，您可以使用以下代码：

```py
optimizer.step() # Updates parameters via SGD
optimizer.zero_grad() # Zeroes out gradients between minibatches

```

在`BaseClassifier`中定义的前馈网络的简单情况下，这种网络的测试模式与训练模式相同——我们只需在测试集中的任何小批量上调用`classifier(test_x)`来评估模型。然而，正如我们将在后面讨论的那样，这并不适用于所有神经网络架构。

这段代码适用于单个小批量——在整个数据集上进行训练需要在每个 epoch 手动洗牌数据集并将数据集分成可以迭代的小批量。幸运的是，PyTorch 还将这个过程抽象成了所谓的 PyTorch 数据集和数据加载器。在下一节中，我们将详细介绍这些模块。

# PyTorch 数据集和数据加载器

PyTorch 的`Dataset`是一个基类，可用于访问您的特定数据。在实践中，您可以通过覆盖两个重要方法`__len__()`和`__getitem__()`来子类化`Dataset`类。第一个方法，从其名称中您可能已经猜到，是指数据集的长度——即模型将在其上进行训练或测试的示例数量。如果我们将数据集视为示例列表，则第二个方法接受一个索引并返回该索引处的示例。每个示例包括数据点（例如图像）和标签（例如 MNIST 中从 0 到 9 的值）。以下是一个数据集的示例代码：

```py
import os
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
  def __init__(self, img_dir, label_file):
    super(ImageDataset, self).__init__()
    self.img_dir = img_dir
    self.labels = torch.tensor(np.load(label_file, allow_pickle=True))
    self.transforms = transforms.ToTensor()

  def __getitem__(self, idx):
    img_pth = os.path.join(self.img_dir, "img_{}.jpg".format(idx))
    img = Image.open(img_pth)
    img = self.transforms(img).flatten()
    label = self.labels[idx]
    return {"data":img, "label":label}

  def __len__(self):
    return len(self.labels)

```

在这个例子中，我们假设包含数据集的目录包含遵循命名约定*img-idx.png*的图像，其中*idx*是图像的索引。此外，我们假设我们的地面真实标签存储在一个保存的 NumPy 数组中，可以使用*idx*加载和索引以找到每个图像对应的标签。

PyTorch 中的`DataLoader`类以数据集实例作为输入，并将加载数据集所需的所有繁重工作抽象化，通过小批量加载数据集并在不同 epoch 之间对数据集进行洗牌。虽然我们不会深入了解背后的细节，`DataLoader`类确实利用了 Python 的内置多进程模块来并行高效地加载小批量数据。以下是将所有内容整合在一起的一些示例代码：

```py
train_dataset = ImageDataset(img_dir='./data/train/',
                             label_file='./data/train/labels.npy')

train_loader = DataLoader(train_dataset,
                          batch_size=4,
                          shuffle=True)

```

要遍历这些数据加载器，请使用以下代码作为模板：

```py
for minibatch in train_loader:
  data, labels = minibatch['data'], minibatch['label']
  print(data)
  print(labels)

```

返回的数据是形状为(64,784)的张量，返回的标签是形状为(64,)的张量。正如您所看到的，数据加载器还会将所有示例堆叠到一个可以简单通过网络运行的单个张量中：

```py
for minibatch in train_loader:
  data, labels = minibatch['data'], minibatch['label']
  out = classifier(data) # to be completed in the next section!

```

其中`out`在 MNIST 的情况下是形状为（64,10）。在下一节中，我们将汇总所有的学习成果，构建一个可以在 MNIST 数据集上进行训练和测试的神经架构，提供训练和测试模型的代码示例，通过在本节的工作基础上构建，并展示示例训练和测试损失曲线。

# 在 PyTorch 中构建 MNIST 分类器

现在是在 PyTorch 中构建一个 MNIST 分类器的时候了。在很大程度上，我们可以重用之前介绍和解释的大部分代码：

```py
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

class BaseClassifier(nn.Module):
  def __init__(self, in_dim, feature_dim, out_dim):
    super(BaseClassifier, self).__init__()
    self.classifier = nn.Sequential(
        nn.Linear(in_dim, feature_dim, bias=True),
        nn.ReLU(),
        nn.Linear(feature_dim, out_dim, bias=True)
    )

  def forward(self, x):
    return self.classifier(x)

# Load in MNIST dataset from PyTorch
train_dataset = MNIST(".", train=True,
                      download=True, transform=ToTensor())
test_dataset = MNIST(".", train=False,
                     download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset,
                          batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=64, shuffle=False)

```

请注意，默认情况下，小批量张量和模型参数都在 CPU 上，因此不需要在每个上调用`to`函数来更改设备。此外，PyTorch 提供的 MNIST 数据集不幸地没有提供验证集，因此我们将尽力仅使用训练损失曲线的见解来为测试集的最终超参数决策提供信息：

```py
# Instantiate model, optimizer, and hyperparameter(s)
in_dim, feature_dim, out_dim = 784, 256, 10
lr=1e-3
loss_fn = nn.CrossEntropyLoss()
epochs=40
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
optimizer = optim.SGD(classifier.parameters(), lr=lr)

def train(classifier=classifier,
          optimizer=optimizer,
          epochs=epochs,
          loss_fn=loss_fn):

  classifier.train()
  loss_lt = []
  for epoch in range(epochs):
    running_loss = 0.0
    for minibatch in train_loader:
      data, target = minibatch
      data = data.flatten(start_dim=1)
      out = classifier(data)
      computed_loss = loss_fn(out, target)
      computed_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      # Keep track of sum of loss of each minibatch
      running_loss += computed_loss.item()
    loss_lt.append(running_loss/len(train_loader))
    print("Epoch: {} train loss: {}".format(epoch+1, 
          running_loss/len(train_loader)))

  plt.plot([i for i in range(1,epochs+1)], loss_lt)
  plt.xlabel("Epoch")
  plt.ylabel("Training Loss")
  plt.title(
      "MNIST Training Loss: optimizer {}, lr {}".format("SGD", lr))
  plt.show()

  # Save state to file as checkpoint
  torch.save(classifier.state_dict(), 'mnist.pt')

def test(classifier=classifier,
          loss_fn = loss_fn):
  classifier.eval()
  accuracy = 0.0
  computed_loss = 0.0

  with torch.no_grad():
      for data, target in test_loader:
          data = data.flatten(start_dim=1)
          out = classifier(data)
          _, preds = out.max(dim=1)

          # Get loss and accuracy
          computed_loss += loss_fn(out, target)
          accuracy += torch.sum(preds==target)

      print("Test loss: {}, test accuracy: {}".format(
          computed_loss.item()/(len(test_loader)*64), 
          accuracy*100.0/(len(test_loader)*64)))

```

此外，请注意，在训练和测试函数的开始分别调用`classifier.train()`和`classifier.eval()`。对这些函数的调用向 PyTorch 后端传达了模型是处于训练模式还是推理模式。您可能会想知道为什么我们需要调用`classifier.train()`和`classifier.eval()`，如果在训练和测试时间神经网络的行为没有区别。尽管在我们的第一次尝试中是这样，但其他神经架构的训练和测试模式并不一定相同。例如，如果在模型架构中添加了 dropout 层，则在测试阶段需要忽略 dropout 层。我们在这里添加`train()`和`eval()`的调用，因为通常认为这样做是一个好习惯。

作为第一步，我们需要为模型训练设置一些起始超参数。我们从稍微保守的学习率`1e-4`开始，并在 40 个 epochs 或整个数据集的迭代后检查训练损失曲线和测试准确率。图 5-2 显示了通过 epochs 的训练损失曲线的图表。

![](img/fdl2_0502.png)

###### 图 5-2。我们看到欠拟合的迹象，因为模型在训练集上的表现未能平稳下来，这意味着我们尚未进入局部最优解

我们可以看到这个损失曲线在训练结束时并没有接近平稳，我们希望看到的是一个以足够学习率训练的模型开始出现的情况。虽然我们没有验证集来确认我们的怀疑，但我们有充分的理由怀疑更高的学习率会有所帮助。将学习率设置为稍微更积极的`1e-3`后，我们观察到训练损失曲线更符合我们所期望看到的情况（图 5-3）。

![](img/fdl2_0503.png)

###### 图 5-3。损失曲线的平稳是我们对于问题的合适学习率所期望看到的更多的情况

损失曲线仅在训练结束时开始平稳下来。这种趋势表明模型很可能处于欠拟合训练数据和过拟合训练数据之间的最佳状态，就像我们之前的尝试一样。在测试集上对 40 个 epochs 训练的模型进行评估，准确率达到 91％！虽然这与今天 MNIST 的顶尖表现者相去甚远，后者主要使用卷积神经分类器，但这是一个很好的开始。我们建议您尝试对代码进行一些扩展，例如增加隐藏层的数量并替换更复杂的优化器。

# 总结

在这一章中，我们涵盖了 PyTorch 及其功能的基础知识。具体来说，我们学习了 PyTorch 中张量的概念，以及这些张量如何存储数值信息。此外，我们还学习了如何通过张量操作来操作张量，访问张量中的数据，并设置一些重要的属性。我们还讨论了 PyTorch 中的梯度以及它们如何存储在张量中。我们通过 PyTorch `nn`模块部分的标准`nn`功能构建了我们的第一个神经网络。将基于`nn`的方法与仅使用 PyTorch 张量的方法进行比较，显示了`nn`模块提供的有效抽象，使其易于使用。最后，在最后一节中，我们将所有学到的知识结合起来，在 PyTorch 提供的测试集上训练和测试了一个 MNIST 手写数字前馈神经分类器，准确率达到了 91%。虽然我们涵盖了许多基础知识，并为您提供了您需要动手实践的知识，但我们只是触及了 PyTorch API 所提供的所有内容的表面—我们鼓励您进一步探索并改进我们在本节中构建的模型。我们建议您访问 PyTorch 文档以了解更多信息，并构建自己的神经网络，包括尝试其他架构，应用于各种在线数据集，如 CIFAR-10 图像识别数据集。在下一节中，我们将涵盖神经网络实现，这是当今另一个最流行的深度学习框架之一。
