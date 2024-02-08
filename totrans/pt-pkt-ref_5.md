# 第五章：自定义 PyTorch

到目前为止，您一直在使用内置的 PyTorch 类、函数和库来设计和训练各种预定义模型、模型层和激活函数。但是，如果您有一个新颖的想法或正在进行前沿的深度学习研究怎么办？也许您发明了一个全新的层架构或激活函数。也许您开发了一个新的优化算法或一个以前从未见过的特殊损失函数。

在本章中，我将向您展示如何在 PyTorch 中创建自定义的深度学习组件和算法。我们将首先探讨如何创建自定义层和激活函数，然后看看如何将这些组件组合成自定义模型架构。接下来，我将向您展示如何创建自定义的损失函数和优化算法。最后，我们将看看如何创建用于训练、验证和测试的自定义循环。

PyTorch 提供了灵活性：您可以扩展现有库，也可以将自定义内容组合到自己的库或包中。通过创建自定义组件，您可以解决新的深度学习问题，加快训练速度，并发现执行深度学习的创新方法。

让我们开始创建一些自定义深度学习层和激活函数。

# 自定义层和激活

PyTorch 提供了一套广泛的内置层和激活函数。然而，PyTorch 如此受欢迎，尤其是在研究社区中，是因为创建自定义层和激活如此简单。这样做的能力可以促进实验并加速您的研究。

如果我们查看 PyTorch 源代码，我们会看到层和激活是使用功能定义和类实现创建的。*功能定义*指定基于输入创建输出的方式。它在`nn.functional`模块中定义。*类实现*用于创建调用此函数的对象，但它还包括从`nn.Module`类派生的附加功能。

例如，让我们看看全连接的`nn.Linear`层是如何实现的。以下代码显示了功能定义`nn.functional.linear()`的简化版本：

```py
import torch

def linear(input, weight, bias=None):

    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret
```

`linear()`函数将输入张量乘以权重矩阵，可选择添加偏置向量，并将结果返回为张量。您可以看到代码针对性能进行了优化。当输入具有两个维度且没有偏置时，应使用融合矩阵加函数`torch.addmm()`，因为在这种情况下速度更快。

将数学计算保留在单独的功能定义中有一个好处，即将优化与层`nn.Module`分开。功能定义也可以在一般编写代码时作为独立函数使用。

然而，我们通常会使用`nn.Module`类来对我们的神经网络进行子类化。当我们创建一个`nn.Module`子类时，我们获得了`nn.Module`对象的所有内置优势。在这种情况下，我们从`nn.Module`派生`nn.Linear`类，如下面的代码所示：

```py
import torch.nn as nn
from torch import Tensor

class Linear(nn.Module):

    def __init__(self, in_features,
                 out_features, bias): ![1](Images/1.png)
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.Tensor(out_features,
                         in_features))
        if bias:
            self.bias = Parameter(
                torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,
                              a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = \
              init._calculate_fan_in_and_fan_out(
                  self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor: ![2](Images/2.png)
        return F.linear(input,
                        self.weight,
                        self.bias) ![3](Images/3.png)
```

①

初始化输入和输出大小、权重和偏置。

②

定义前向传递。

③

使用`linear()`的功能定义。

`nn.Linear`代码包括任何`nn.Module`子类所需的两种方法。一种是`__init__()`，它初始化类属性，即在这种情况下的输入、输出、权重和偏置。另一种是`forward()`方法，它定义了前向传递期间的处理。

如前面的代码所示，`forward()`方法经常调用与层相关的`nn.functional`定义。这种约定在 PyTorch 代码中经常使用于层。

创建自定义层的约定是首先创建一个实现数学运算的函数，然后创建一个`nn.Module`子类，该子类使用这个函数来实现层类。使用这种方法可以很容易地在 PyTorch 模型开发中尝试新的层设计。

## 自定义层示例（复杂线性）

接下来，我们将看看如何创建一个自定义层。在这个例子中，我们将为一种特殊类型的数字——*复数*创建自己的线性层。复数经常在物理学和信号处理中使用，由一对数字组成——一个“实”部分和一个“虚”部分。这两个部分都是浮点数。

PyTorch 正在添加对复杂数据类型的支持；然而，在撰写本书时，它们仍处于测试阶段。因此，我们将使用两个浮点张量来实现它们，一个用于实部，一个用于虚部。

在这种情况下，输入、权重、偏置和输出都将是复数，并且将由两个张量组成，而不是一个。复数乘法得到以下方程（其中 *j* 是复数 <math alttext="StartRoot 1 EndRoot"><msqrt><mn>1</mn></msqrt></math>）：

<math><mrow><mrow><mo>(</mo> <mi>i</mi> <msub><mi>n</mi> <mi>r</mi></msub> <mo>+</mo> <mi>i</mi> <msub><mi>n</mi> <mi>i</mi></msub> <mo>*</mo> <mi>j</mi> <mo>)</mo></mrow> <mo>*</mo> <mrow><mo>(</mo> <msub><mi>w</mi> <mi>r</mi></msub> <mo>+</mo> <msub><mi>w</mi> <mi>i</mi></msub> <mo>*</mo> <mi>j</mi> <mo>)</mo></mrow> <mo>+</mo> <mrow><mo>(</mo> <msub><mi>b</mi> <mi>r</mi></msub> <mo>+</mo> <msub><mi>b</mi> <mi>i</mi></msub> <mo>*</mo> <mi>j</mi> <mo>)</mo></mrow> <mo>=</mo> <mrow><mo>(</mo> <mi>i</mi> <msub><mi>n</mi> <mi>r</mi></msub> <mo>*</mo> <msub><mi>w</mi> <mi>r</mi></msub> <mo>-</mo> <mi>i</mi> <msub><mi>n</mi> <mi>i</mi></msub> <mo>*</mo> <msub><mi>w</mi> <mi>i</mi></msub> <mo>+</mo> <msub><mi>b</mi> <mi>r</mi></msub> <mo>)</mo></mrow> <mo>+</mo> <mrow><mo>(</mo> <mi>i</mi> <msub><mi>n</mi> <mi>r</mi></msub> <mo>*</mo> <msub><mi>w</mi> <mi>i</mi></msub> <mo>+</mo> <mi>i</mi> <msub><mi>n</mi> <mi>i</mi></msub> <mo>*</mo> <msub><mi>w</mi> <mi>r</mi></msub> <mo>+</mo> <msub><mi>b</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>*</mo> <mi>j</mi></mrow></math>

首先，我们将创建一个复杂线性层的函数版本，如下面的代码所示：

```py
def complex_linear(in_r, in_i, w_r, w_i, b_i, b_r):
    out_r = (in_r.matmul(w_r.t())
              - in_i.matmul(w_i.t()) + b_r)
    out_i = (in_r.matmul(w_i.t())
              - in_i.matmul(w_r.t()) + b_i)

    return out_r, out_i
```

如你所见，该函数将复杂乘法公式应用于张量数组。接下来，我们根据`nn.Module`创建`ComplexLinear`的类版本，如下面的代码所示：

```py
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = \
          Parameter(torch.randn(out_features,
                                in_features))
        self.weight_i = \
          Parameter(torch.randn(out_features,
                                in_features))
        self.bias_r = Parameter(
                        torch.randn(out_features))
        self.bias_i = Parameter(
                        torch.randn(out_features))

    def forward(self, in_r, in_i):
        return F.complex_linear(in_r, in_i,
                 self.weight_r, self.weight_i,
                 self.bias_r, self.bias_i)
```

在我们的类中，我们在`__init__()`函数中为实部和虚部定义了单独的权重和偏置。请注意，`in_features`和`out_features`的选项数量不会改变，因为实部和虚部的数量是相同的。我们的`forward()`函数只是调用我们的复杂乘法和加法操作的函数定义。

请注意，我们也可以使用 PyTorch 现有的`nn.Linear`层来构建我们的层，如下面的代码所示：

```py
class ComplexLinearSimple(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinearSimple, self).__init__()
        self.fc_r = Linear(in_features,
                           out_features)
        self.fc_i = Linear(in_features,
                           out_features)

    def forward(self,in_r, in_i):
        return (self.fc_r(in_r) - self.fc_i(in_i),
               self.fc_r(in_i)+self.fc_i(in_r))
```

在这段代码中，我们可以免费获得`nn.Linear`的所有附加好处，而无需实现新的函数定义。当你创建自己的自定义层时，检查 PyTorch 的内置层，看看是否可以重用现有的类。

即使这个例子非常简单，你也可以使用相同的方法来创建更复杂的层。此外，相同的方法也可以用来创建自定义激活函数。

激活函数与 NN 层非常相似，它们通过对一组输入执行数学运算来返回输出。它们的不同之处在于，操作是逐元素执行的，并且不包括在训练过程中调整的权重和偏置等参数。因此，激活函数可以仅使用函数版本执行。

例如，让我们来看看 ReLU 激活函数。ReLU 函数对于负值为零，对于正值为线性：

```py
def my_relu(input, thresh=0.0):
    return torch.where(
              input > thresh,
              input,
              torch.zeros_like(input))
```

当激活函数具有可配置参数时，通常会创建一个类版本。我们可以通过创建一个 ReLU 类来添加调整 ReLU 函数的阈值和值的功能，如下所示：

```py
class MyReLU(nn.Module):
  def __init__(self, thresh = 0.0):
      super(MyReLU, self).__init__()
      self.thresh = thresh

  def forward(self, input):
      return my_relu(input, self.thresh)
```

在构建 NN 时，通常使用激活函数的函数版本，但如果有的话也可以使用类版本。以下代码片段展示了如何使用`torch.nn`中包含的 ReLU 激活的两个版本。

这是函数版本：

```py
import torch.nn.functional as F ![1](Images/1.png)

class SimpleNet(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(SimpleNet, self).__init__()
    self.fc1 = nn.Linear(D_in, H)
    self.fc2 = nn.Linear(H, D_out)

  def forward(self, x):
    x = F.relu(self.fc1(x)) ![2](Images/2.png)
    return self.fc2(x)
```

①

导入函数包的常见方式。

②

这里使用了 ReLU 的函数版本。

这是类版本：

```py
class SimpleNet(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(SimpleNet, self).__init__()
    self.net = nn.Sequential( ![1](Images/1.png)
        nn.Linear(D_in, H),
        nn.ReLU(), ![2](Images/2.png)
        nn.Linear(H, D_out)
    )

  def forward(self, x):
    return self.net(x)
```

①

我们使用`nn.Sequential()`因为所有组件都是类。

②

我们正在使用 ReLU 的类版本。

## 自定义激活函数示例（Complex ReLU）

我们可以创建自己的自定义 ComplexReLU 激活函数来处理我们之前创建的`ComplexLinear`层中的复数值。以下代码展示了函数版本和类版本：

```py
def complex_relu(in_r, in_i): ![1](Images/1.png)
    return (F.relu(in_r), F.relu(in_i))

class ComplexReLU(nn.Module): ![2](Images/2.png)
  def __init__(self):
      super(ComplexReLU, self).__init__()

  def forward(self, in_r, in_i):
      return complex_relu(in_r, in_i)
```

①

函数版本

②

类版本

现在您已经学会了如何创建自己的层和激活函数，让我们看看如何创建自己的自定义模型架构。

# 自定义模型架构

在第二章和第三章中，我们使用了内置模型并从内置 PyTorch 层创建了自己的模型。在本节中，我们将探讨如何创建类似于`torchvision.models`的模型库，并构建灵活的模型类，根据用户提供的配置参数调整架构。

`torchvision.models`包提供了一个`AlexNet`模型类和一个`alexnet()`便利函数来方便其使用。让我们先看看`AlexNet`类：

```py
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11,
                      stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

与所有层、激活函数和模型一样，`AlexNet`类派生自`nn.Module`类。`AlexNet`类是如何创建和组合子模块成为 NN 的一个很好的例子。

该库定义了三个子网络——`features`、`avgpool`和`classifier`。每个子网络由 PyTorch 层和激活函数组成，并按顺序连接。AlexNet 的`forward()`函数描述了前向传播；即输入如何被处理以形成输出。

在这种情况下，PyTorch 的`torchvision.models`代码提供了一个方便的函数`alexnet()`来实例化或创建模型并提供一些选项。这里的选项是`pretrained`和`progress`；它们确定是否加载具有预训练参数的模型以及是否显示进度条：

```py
from torch.hub import load_state_dict_from_url
model_urls = {
    'alexnet':
    'https://pytorch.tips/alexnet-download',
}

def alexnet(pretrained=False,
            progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
              model_urls['alexnet'],
              progress=progress)
        model.load_state_dict(state_dict)
    return model
```

`**kwargs`参数允许您向 AlexNet 模型传递其他选项。在这种情况下，您可以使用`alexnet(n_classes = 10)`将类别数更改为 10。该函数将使用`n_classes = 10`实例化 AlexNet 模型并返回模型对象。如果`pretrained`为`True`，函数将从指定的 URL 加载权重。

通过采用类似的方法，您可以创建自己的模型架构。创建一个从`nn.Module`派生的顶级模型。定义您的`__init__()`和`forward()`函数，并根据子网络、层和激活函数实现您的 NN。您的子网络、层和激活函数甚至可以是您自己创建的自定义的。

如您所见，`nn.Module`类使创建自定义模型变得容易。除了`Module`类外，`torch.nn`包还包括内置的损失函数。让我们看看如何创建自己的损失函数。

# 自定义损失函数

如果您回忆一下第三章，在训练 NN 模型之前，我们需要定义损失函数。损失函数或成本函数定义了我们在训练过程中希望通过调整模型权重来最小化的度量。

起初，损失函数可能看起来只是一个函数定义，但请记住，损失函数是 NN 模块参数的函数。

因此，损失函数实际上就像是一个额外的层，将 NN 的输出作为输入，并产生一个度量作为其输出。当我们进行反向传播时，我们是在损失函数上进行反向传播，而不是在 NN 上。

这使我们能够直接调用该类来计算给定 NN 输出和真实值的损失。然后我们可以一次计算所有 NN 参数的梯度，即进行反向传播。以下代码展示了如何在代码中实现这一点：

```py
loss_fcn = nn.MSELoss() ![1](Images/1.png)
loss = loss_fcn(outputs, targets)
loss.backward()
```

①

有时称为`criterion`

首先我们实例化损失函数本身，然后调用该函数，传入输出（来自我们的模型）和目标值（来自我们的数据）。最后，我们调用`backward()`方法进行反向传播，并计算所有模型参数相对于损失的梯度。

与之前讨论的层类似，损失函数使用功能定义和从`nn.Module`类派生的类实现来实现。

`mse_loss`的功能定义和类实现的简化版本如下所示：

```py
def mse_loss(input, target):
    return ((input-target)**2).mean()

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target):
        return F.mse_loss(input, target)
```

让我们创建自己的损失函数，复数的 MSE 损失。为了创建自定义损失函数，我们首先定义一个数学上描述损失函数的功能定义。然后我们将创建损失函数类，如下所示：

```py
def complex_mse_loss(input_r, input_i,
                     target_r, target_i):
  return (((input_r-target_r)**2).mean(),
          ((input_i-target_i)**2).mean())

class ComplexMSELoss(nn.Module):
    def __init__(self, real_only=False):
        super(ComplexMSELoss, self).__init__()
        self.real_only = real_only

    def forward(self, input_r, input_i,
                target_r, target_i):
        if (self.real_only):
          return F.mse_loss(input_r, target_r)
        else:
          return complex_mse_loss(
              input_r, input_i,
              target_r, target_i)
```

这次，我们在类中创建了一个名为`real_only`的可选设置。当我们使用`real_only = True`实例化损失函数时，将使用`mse_loss()`函数而不是`complex_mse_loss()`函数。

正如您所看到的，PyTorch 在构建自定义模型架构和损失函数方面提供了出色的灵活性。在进行训练之前，还有一个函数可以自定义：优化器。让我们看看如何创建自定义优化器。

# 自定义优化器算法

优化器在训练 NN 模型中起着重要作用。优化器是在训练过程中更新模型参数的算法。当我们使用`loss.backward()`进行反向传播时，我们确定参数应该增加还是减少以最小化损失。优化器使用梯度来确定在每一步中参数应该改变多少并进行相应的更改。

PyTorch 有自己的子模块称为`torch.optim`，其中包含许多内置的优化器算法，正如我们在第三章中所看到的。要创建一个优化器，我们传入我们模型的参数和任何特定于优化器的选项。例如，以下代码创建了一个学习率为 0.01 和动量值为 0.9 的 SGD 优化器：

```py
from torch import optim

optimizer = optim.SGD(model.parameters(),
                      lr=0.01, momentum=0.9)
```

在 PyTorch 中，我们还可以为不同的参数指定不同的选项。当您想要为模型的不同层指定不同的学习率时，这是很有用的。每组参数称为参数组。我们可以使用字典指定不同的选项，如下所示：

```py
optim.SGD([
        {'params':
          model.features.parameters()},
        {'params':
          model.classifier.parameters(),
          'lr': 1e-3}
    ], lr=1e-2, momentum=0.9)
```

假设我们正在使用 AlexNet 模型，上述代码将分类器层的学习率设置为`1e-3`，并使用默认学习率`1e-2`来训练特征层。

PyTorch 提供了一个`torch.optim.Optimizer`基类，以便轻松创建自定义优化器。以下是`Optimizer`基类的简化版本：

```py
from collections import defaultdict

class Optimizer(object):

    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict) ![1](Images/1.png)
        self.param_groups = [] ![2](Images/2.png)

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError(
                """optimizer got an
                empty parameter list""")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def zero_grad(self): ![3](Images/3.png)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step(self, closure): ![4](Images/4.png)
        raise NotImplementedError
```

①

根据需要定义`state`。

②

根据需要定义 `param_groups`。

③

根据需要定义 `zero_grad()`。

④

您需要编写自己的 `step()`。

优化器有两个主要属性或组件：`state` 和 `param_groups`。`state` 属性是一个字典，可以在不同的优化器之间变化。它主要用于在每次调用 `step()` 函数之间维护值。`param_groups` 属性也是一个字典。它包含参数本身以及每个组的相关选项。

`Optimizer` 基类中的重要方法是 `zero_grad()` 和 `step()`。`zero_grad()` 方法用于在每次训练迭代期间将梯度归零或重置。`step()` 方法用于执行优化器算法，计算每个参数的变化，并更新模型对象中的参数。`zero_grad()` 方法已经为您实现。但是，当创建自定义优化器时，您必须创建自己的 `step()` 方法。

让我们通过创建我们自己简单版本的 SGD 来演示这个过程。我们的 SDG 优化器将有一个选项——学习率（LR）。在每个优化器步骤中，我们将梯度乘以 LR，并将其添加到参数中（即，调整模型的权重）：

```py
from torch.optim import Optimizer

class SimpleSGD(Optimizer):

    def __init__(self, params, lr='required'):
        if lr is not 'required' and lr < 0.0:
          raise ValueError(
            "Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(SimpleSGD, self).__init__(
            params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                p.add_(d_p, alpha=-group['lr'])

        return
```

`__init__()` 函数设置默认选项值，并根据输入参数初始化参数组。请注意，我们不必编写任何代码来执行此操作，因为 `super(SGD, self).init(params, defaults)` 调用基类初始化方法。我们真正需要做的是编写 `step()` 方法。对于每个参数组，我们首先将参数乘以组的 LR，然后从参数本身减去该乘积。这通过调用 `p.add_(d_p, alpha=-group['lr'])` 完成。

以下是我们如何使用新优化器的示例：

```py
optimizer = SimpleSGD(model.parameters(),
                      lr=0.001)
```

我们还可以使用以下代码为模型的不同层定义不同的学习率。在这里，我们假设再次使用 AlexNet 作为模型，其中包含名为 `feature` 和 `classifier` 的层：

```py
optimizer = SimpleSGD([
                {'params':
                 model.features.parameters()},
                {'params':
                 model.classifier.parameters(),
                 'lr': 1e-3}
            ], lr=1e-2)
```

现在您可以为训练模型创建自己的优化器了，让我们看看如何创建自己的自定义训练、验证和测试循环。

# 自定义训练、验证和测试循环

在整本书中，我们一直在使用自定义训练、验证和测试循环。这是因为在 PyTorch 中，所有训练、验证和测试循环都是由程序员手动创建的。

与 Keras 不同，没有 `fit()` 或 `eval()` 方法来执行循环。相反，PyTorch 要求您编写自己的循环。在许多情况下，这实际上是一个好处，因为您希望控制训练过程中发生的事情。

实际上，在“生成学习—使用 DCGAN 生成 Fashion-MNIST 图像”中的参考设计演示了如何创建更复杂的训练循环。

在本节中，我们将探讨编写循环的传统方式，并讨论开发人员定制循环的常见方式。让我们回顾一些用于训练、验证和测试循环的常用代码：

```py
for epoch in range(n_epochs):

    # Training
    for data in train_dataloader:
        input, targets = data
        optimizer.zero_grad()
        output = model(input)
        train_loss = criterion(output, targets)
        train_loss.backward()
        optimizer.step()

    # Validation
    with torch.no_grad():
      for input, targets in val_dataloader:
          output = model(input)
          val_loss = criterion(output, targets)

# Testing
with torch.no_grad():
  for input, targets in test_dataloader:
      output = model(input)
      test_loss = criterion(output, targets)
```

这段代码应该看起来很熟悉，因为我们在整本书中经常使用它。我们假设 `n_epochs`、`model`、`criterion`、`optimizer`、`train_`、`val_` 和 `test_dataloader` 已经定义。对于每个时期，我们执行训练和验证循环。训练循环逐批次处理每个批次，将批次输入通过模型，并计算损失。然后我们执行反向传播来计算梯度，并执行优化器来更新模型的参数。

验证循环禁用梯度计算，并逐批次将验证数据通过网络传递。测试循环逐批次将测试数据通过模型传递，并计算测试数据的损失。

让我们为我们的循环添加一些额外的功能。可能性是无限的，但这个示例将演示一些简单的任务，比如打印信息、重新配置模型以及在训练过程中调整超参数。让我们走一遍以下代码，看看如何实现这一点：

```py
for epoch in range(n_epochs):
    total_train_loss = 0.0 ![1](Images/1.png)
    total_val_loss = 0.0  ![1](Images/1.png)

    if (epoch == epoch//2):
      optimizer = optim.SGD(model.parameters(),
                            lr=0.001) ![2](Images/2.png)
    # Training
    model.train() ![3](Images/3.png)
    for data in train_dataloader:
        input, targets = data
        optimizer.zero_grad()
        output = model(input)
        train_loss = criterion(output, targets)
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss ![1](Images/1.png)

    # Validation
    model.eval() ![3](Images/3.png)
    with torch.no_grad():
      for input, targets in val_dataloader:
          output = model(input)
          val_loss = criterion(output, targets)
          total_val_loss += val_loss ![1](Images/1.png)

    print("""Epoch: {}
 Train Loss: {}
 Val Loss {}""".format(
         epoch, total_train_loss,
         total_val_loss)) ![1](Images/1.png)

# Testing
model.eval()
with torch.no_grad():
  for input, targets in test_dataloader:
      output = model(input)
      test_loss = criterion(output, targets)
```

①

打印 epoch、训练和验证损失的示例

②

重新配置模型的示例（最佳实践）

③

修改训练过程中的超参数示例

在上述代码中，我们添加了一些变量来跟踪运行的训练和验证损失，并在每个 epoch 打印它们。接下来，我们使用 `train()` 或 `eval()` 方法来配置模型进行训练或评估。这仅适用于模型的 `forward()` 函数在训练和评估时表现不同的情况。

例如，一些模型可能在训练过程中使用 dropout，但在验证或测试过程中不应用 dropout。在这种情况下，我们可以通过调用 `model.train()` 或 `model.eval()` 来重新配置模型，然后执行它。

最后，我们在训练过程中修改了优化器中的学习率。这使我们能够在一半的 epoch 训练后以更快的速度训练，同时在微调参数更新之后。

这个示例是如何自定义您的循环的简单演示。训练、验证和测试循环可能会更加复杂，因为您同时训练多个网络、使用多模态数据，或设计更复杂的网络，甚至可以训练其他网络。PyTorch 提供了设计用于训练、验证和测试的特殊和创新过程的灵活性。

###### 提示

PyTorch Lightning 是一个第三方 PyTorch 包，提供了用于训练、验证和测试循环的样板模板。该包提供了一个框架，允许您创建自定义循环，而无需为每个模型实现重复输入样板代码。我们将在第八章中讨论 PyTorch Lightning。您也可以在[PyTorch Lightning 网站](http://pytorch.tips/pytorch-lightning)上找到更多信息。

在本章中，您学习了如何为在 PyTorch 中开发深度学习模型创建自定义组件。随着您的模型变得越来越复杂，您可能会发现您需要训练模型的时间可能会变得相当长——也许是几天甚至几周。在下一章中，您将看到如何利用内置的 PyTorch 能力来加速和优化您的训练过程，从而显著减少整体模型开发时间。
