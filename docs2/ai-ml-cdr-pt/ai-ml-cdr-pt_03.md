# 第二章\. 计算机视觉简介

第一章介绍了机器学习工作的基础知识。您了解了如何使用神经网络编程入门，将数据与标签匹配，并从中看到如何推断出区分物品的规则。

在本章中，我们将考虑下一个逻辑步骤，即将这些概念应用于计算机视觉。在这个过程中，模型学习如何在图片中识别内容，以便它能够“看到”其中的内容。您将使用一个流行的衣物数据集，并构建一个能够区分它们并“看到”不同类型衣物之间差异的模型。

# 计算机视觉是如何工作的

*计算机视觉*是计算机识别物品的能力，而不仅仅是存储它们的像素。例如，考虑一下可能像图 2-1 中的物品。它们非常复杂，有很多不同种类的相同物品。看看这两双鞋——它们非常不同，但它们仍然是鞋！

![](img/aiml_0201.png)

###### 图 2-1\. 衣物示例

这里有许多可识别的衣物。您理解衬衫、外套和连衣裙之间的区别，并且基本知道每种物品是什么——但您如何向从未见过衣物的人解释这一切？鞋子呢？这张图片中有两双鞋，但鉴于它们之间的主要差异，您如何向某人解释使它们成为鞋子的共同点？这是另一个我们曾在第一章中提到的基于规则的编程可能失效的领域。有时，用规则描述某物是不切实际的。

当然，计算机视觉也不例外。但考虑一下你是如何学会识别所有这些物品的——通过看到很多不同的例子，并积累它们使用的经验。计算机能否以同样的方式学习？答案是肯定的，但有一定的限制。在本章的剩余部分，我们将通过一个例子来了解如何使用一个名为 Fashion MNIST 的知名数据集来教计算机识别衣物。

# The Fashion MNIST Database

学习和基准测试算法的基础数据集之一是修改后的国家标准与技术研究院（MNIST）数据库，该数据库由 Yann LeCun、Corinna Cortes 和 Christopher Burges 创建。这个数据集包含从 0 到 9 的 70,000 个手写数字图像，图像为 28 × 28 灰度图。

[Fashion MNIST](https://oreil.ly/f-mnist)被设计成 MNIST 的一个直接替代品，它具有相同的记录数、相同的图像尺寸和相同的类别数。与 0 到 9 的数字图像不同，Fashion MNIST 包含 10 种不同类型衣物的图像。

您可以在图 2-2 中看到数据集内容的示例，其中每件服装类型都分配了三行。

![](img/aiml_0202.png)

###### 图 2-2\. 探索 Fashion MNIST 数据集

Fashion MNIST 拥有丰富的服装种类，包括衬衫、裤子、连衣裙和许多类型的鞋子！此外，如您所注意到的，它是单色的，因此每张图片都由一定数量的像素组成，像素值介于 0 到 255 之间。这使得数据集更容易管理。

您可以在图 2-3 中看到数据集中某个图像的特写。

![](img/aiml_0203.png)

###### 图 2-3\. Fashion MNIST 数据集中图像的特写

就像任何图像一样，这个图像是一个像素的矩形网格。在这种情况下，网格大小是 28 × 28，每个像素的值介于 0 到 255 之间，因此它由一个灰度平方表示。为了使其更容易看到，我已经将其扩展，使其看起来像像素化。

现在，让我们看看如何使用我们之前看到的函数来使用这些像素值。

# 视觉神经元

在第一章中，您看到了一个非常简单的场景，其中一台机器被赋予了一组*x*和*y*值，并且它学会了它们之间的关系是*y* = 2*x* – 1。这是使用一个非常简单的只有一个层和一个神经元的神经网络完成的。如果您要直观地绘制它，它可能看起来像图 2-4。

![](img/aiml_0204.png)

###### 图 2-4\. 单个神经元学习线性关系

我们的每张图像是一组 784 个值（28 × 28）介于 0 到 255 之间。它们可以是我们的*x*。我们还知道在我们的数据集中有 10 种不同的图像类型，所以让我们考虑它们是我们的*y*。现在，我们想要了解*y*作为*x*的函数的函数看起来像什么。

由于每张图像有 784 个*x*值，而我们的*y*将在 0 到 9 之间，一个简单的方程*y* = *mx* + *c*将不足以解决这个问题。这是因为存在大量可能的值，而方程只能在一条线上绘制值。

但我们*可以*做的是让几个神经元共同工作。每个神经元将学习*参数*，当我们有所有这些参数共同作用的组合函数时，我们可以看到我们是否可以将该模式与我们的期望答案相匹配（参见图 2-5）。

![](img/aiml_0205.png)

###### 图 2-5\. 扩展我们的模式以进行更复杂的示例

此图顶部的灰色方框可以被认为是图像中的像素，它们是我们的*X*值。当我们训练神经网络时，我们将像素加载到一个神经元的层中——图 2-5 显示了它们被加载到第一个神经元中，但值只加载到每个神经元中。此外，考虑每个神经元的权重和偏差（*w*和*b*）是随机初始化的。然后，当我们对每个神经元的输出值求和时，我们将得到一个值。我们将对输出层中的每个神经元都这样做，因此神经元 0 将包含像素加起来等于标签 0 的概率值，神经元 1 将包含像素加起来等于标签 1 的概率值，等等。

随着时间的推移，我们希望将这个值匹配到期望的输出——对于这张图片来说，是数字 9，这也是图 2-3 中展示的踝靴的标签。换句话说，这个神经元应该拥有所有输出神经元中最大的值。

由于有 10 个标签，随机初始化应该有大约 10%的时间能够得到正确答案。基于这一点，损失函数和优化器可以逐个 epoch 调整每个神经元的内部参数，以改善这 10%的准确性。因此，随着时间的推移，计算机将学会“看到”什么使鞋子成为鞋子，连衣裙成为连衣裙。当你运行代码并看到你的神经网络有效地学会区分不同的物品时，你会看到这个过程不断改进。

# 设计神经网络

让我们以我们刚刚讨论的例子为例，探索它在代码中的样子。首先，我们将查看图 2-5 中展示的神经网络设计：

```py
self.linear_relu_stack = nn.Sequential(
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
)
```

如果你记得，在第一章中，我们有一个`Sequential`模型来指定我们有很多层。在那个例子中，我们只有一个层，但现在我们使用它来定义多个层。

第一层，一个`Linear`层，是一层学习其输入和输出之间线性关系的神经元。正如之前一样，当使用`Linear`时，你给出两个参数：输入形状和输出形状。方便的是，输出形状实际上是这个层中你想要的神经元数量，我们指定我们想要 128 个。*输入*形状定义为(28 × 28)，这是进入网络的数据大小，正如你之前看到的，这是 Fashion MNIST 图像的维度。

输入在 图 2-5 中显示为中间层，你经常会听到这样的层被描述为 *隐藏层*。术语 *隐藏* 只是意味着没有直接接口到该层。这需要一点时间来适应——中间层是你 *定义* 的第一个层，在一个像 图 2-5 这样的图中，你可以看到它在图的中间。这是因为我们还绘制了数据“进入”这个层。另一件需要注意的事情是，来自 Fashion MNIST 等数据集的图像数据通常是矩形的，但层不会识别这一点，因此它需要被“展平”成一个 1-D 数组，如 图 2-5 的顶部所示。你很快就会看到相关的代码。

使用这个第一个 `Linear`，我们要求有 128 个神经元，它们的内部参数随机初始化。通常，在这个时候我会被问到“为什么是 128？” 这完全是随机的——没有使用神经元数量的固定规则。当你设计层时，你想要选择适当数量的值，以便你的模型能够真正学习。更多的神经元意味着它将运行得更慢，因为它必须学习更多的参数。更多的神经元也可能导致一个网络在识别训练数据方面做得很好，但在识别它之前没有见过的数据方面做得不好。（这被称为 *过拟合*，我们将在本章后面讨论）。另一方面，更少的神经元意味着模型可能没有足够的参数来学习。

你将需要探索学习速度和学习精度之间的权衡，并在一段时间内进行实验以选择正确的值。这个过程通常被称为 *超参数调整*。在机器学习中，*超参数* 是用于控制训练的值，而不是经过训练/学习的神经元的内部值，这些内部值被称为 *参数*。

当你使用 PyTorch 定义神经网络并使用 `Sequential` 时，你不仅定义了网络的层以及它们可能使用的神经元类型。你还可以定义在数据在神经网络层之间流动时执行的功能。这些通常被称为 *激活函数*，激活函数是你在代码中看到的下一个指定为 `nn.ReLU()` 的东西。激活函数是将在层中的每个神经元上执行代码。PyTorch 支持许多开箱即用的激活函数，其中在中层中非常常见的一个是 `ReLU`，它代表 *修正线性单元*。这是一个简单的函数，只有当它的值大于 0 时才返回值。在这种情况下，我们不希望负值传递到下一层，从而可能影响求和函数，因此，我们不必编写大量的 `if-then` 代码，而可以简单地使用 `ReLU` 激活层。

最后，还有一个`Linear`层，它将是*输出层*。如果你看看定义的形状（128, 10），并通过“输入大小，输出大小”框架来思考，你会发现它有 128 个“输入”（即上一层中的神经元数量）和 10 个“输出”。这 10 个是什么？回想一下，Fashion MNIST 有 10 种服装类别。这些神经元中的每一个实际上被分配了一个类别，并且它最终会得到一个概率，即输入像素与该类别的匹配概率，因此我们的任务是确定哪一个具有最高的值。你可能想知道这些分配是如何发生的：代码在哪里说一个神经元代表鞋子，另一个代表衬衫？为了回答这个问题，回想一下第一章中的*y* = 2*x* ‒ 1 示例，在那里我们有一组输入数据和一组已知的正确答案，有时被称为*地面真相*。Fashion MNIST 将以相同的方式工作。在训练网络时，我们提供输入图像及其已知答案作为一组我们希望输出神经元看起来像的东西。因此，网络将“学习”到当它看到鞋子时，不表示该鞋子的输出神经元应该为零值，而表示该鞋子的神经元应该有“1”值。

我们*也可以*遍历输出神经元以找到最大值，但`LogSoftmax`激活函数已经为我们做了这件事。

因此，现在当我们训练我们的神经网络时，我们有两个目标。我们希望能够输入一个 28 × 28 像素的数组，并且我们希望中间层的神经元具有权重和偏差（*w* 和 *B* 值），当它们结合在一起时，能够匹配这些像素到 10 个输出值中的某一个。

# 完整代码

现在我们已经探索了神经网络的架构，让我们来看看使用 Fashion MNIST 数据训练模型的完整代码。

下面是完整的代码：

```py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                             download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                             download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, 
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, 
                          shuffle=False)

# Define the model
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = FashionMNISTModel()

# Define the loss function and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  
                    `[{``current``:``>``5``d``}``/``{``size``:``>``5``d``}]``")` ``` `# 训练过程` `epochs` `=` `5` `for` `t` `in` `range``(``epochs``):`     `print``(``f``"Epoch` `{``t``+``1``}``\n``-------------------------------"``)`     `train``(``train_loader``,` `model``,` `loss_function``,` `optimizer``)` `print``(``"Done!"``)` ```py
```

```py`` ````让我们逐个分析这个例子。首先，让我们考虑数据来自哪里。在 torchvision 库中，有一个数据集集合，我们可以从中加载 Fashion MNIST，如下所示：    ```py datasets.FashionMNIST ```    因此，在我们的第一个代码块中，你会看到这些：    ```py train_dataset = datasets.FashionMNIST(root='./data', train=True,                               download=True, transform=transform) test_dataset = datasets.FashionMNIST(root='./data', train=False,                               download=True, transform=transform) ```    现在，你可能想知道为什么我们使用 *两个* 数据集。很简单：一个用于训练，一个用于测试。这里的想法也很简单：如果你在一个数据集上训练一个神经网络，它可以成为该数据集的专家，但它可能无法有效地理解或分类它之前没有见过的其他数据。在 Fashion MNIST 的情况下，它可能非常擅长理解鞋子和衬衫子集之间的差异，但当新的数据呈现给它时，它可能会做得很差。因此，保留一部分数据而不用它来训练神经网络是一种很好的做法。在这种情况下，Fashion MNIST 有 70,000 项数据，但只有 60,000 项用于训练网络，其余 10,000 项用于测试。如果你仔细查看前面的代码，你会看到这两行之间的区别在于 `train=` 参数。对于第一个，训练集的参数设置为 True。对于另一个，它设置为 False。    你还会在数据集中看到 `transform` 参数。它指定要应用于数据的转换，其定义如下：    ```py transform = transforms.Compose([transforms.ToTensor()]) ```    神经网络通常使用 *归一化* 值（即介于 0 和 1 之间的值）。然而，我们的图像像素在 0–255 的范围内，这些值表示它们的颜色深度，其中 0 是黑色，255 是白色，介于两者之间的是灰色。为了准备神经网络的数据，我们应该将这些灰色映射到 0 和 1 之间的值。前面的代码将自动为你这样做，所以当你加载代码时应用这个 `transform` 参数会将像素值从 [0, 255] 的整数范围映射到 [0, 1] 的浮点范围，并将它们加载到适合神经网络的数组中（即张量）。    我们的工作将是在类似我们将 *y* 与 *x* 在 第一章 中拟合的方式，将训练图像拟合到训练标签。    关于 [为什么归一化数据对训练神经网络更好](https://oreil.ly/6d_Po) 的数学超出了本书的范围，但请记住，当你使用 PyTorch 训练神经网络时，归一化会提高性能。通常，当处理非归一化数据时，你的网络可能无法学习，并且会有巨大的错误。你可能会回想起 第一章 中的 *y* = 2*x* – 1 例子不需要归一化数据，因为它非常简单，但为了好玩，尝试用不同的 *x* 和 *y* 值训练它，其中 *x* 非常大——你会看到它很快就会失败！    接下来，我们定义构成我们模型的神经网络，正如之前讨论的那样，但我们将用更多细节来完善它——包括展平层以及我们希望在模型中如何实现“前向”传递。    这是代码：    ```py # Define the model class FashionMNISTModel(nn.Module):     def __init__(self):         super(FashionMNISTModel, self).__init__()         self.flatten = nn.Flatten()         self.linear_relu_stack = nn.Sequential(             nn.Linear(28*28, 128),             nn.ReLU(),             nn.Linear(128, 10),             nn.LogSoftmax(dim=1)         )       def forward(self, x):         x = self.flatten(x)         logits = self.linear_relu_stack(x)         return logits   model = FashionMNISTModel() ```    这里需要注意的一些关键点是，`FashionMNISTModel` 类是 `nn.Module` 的子类，这给了你覆盖其 `forward` 方法的权限。我们使用此方法在数据通过网络传递时。记得在 第一章 中我们看到 `loss.backward()` 调用执行反向传播并更改网络参数吗？当使用 PyTorch 训练模型时，你经常会遇到相同的模式。你将定义在数据通过网络传递时执行的功能，然后定义其他在从损失计算中得到的梯度通过网络反向传递时执行的功能。    因此，如果我们查看类的 `init`，我们定义了两个方法：`flatten`，设置为 `nn.FLatten()`（一个将 2D 图像展平到 1D 的内置函数），以及 `linear_relu_stack`，设置为定义网络行为的层和操作的序列（通常缩写为 *ops*）。    在 `forward` 中，我们然后简单地定义这些是如何工作的。首先，我们通过调用 `self.flatten` 展平我们的数据 `x`，然后结果将传递到 `linear_relu_stack` 以获取结果。结果被称为 *logits*，这是由 `LogSoftmax` 定义的指示模型对每个类别是正确分类的置信度的对数概率。    为了从我们的数据中学习，我们需要一个损失函数来计算我们的当前“猜测”有多好或有多坏，我们还需要一个优化器来确定改进猜测的下一组参数。    下面是如何定义这两个的示例：    ```py # Define the loss function and optimizer loss_function = nn.NLLLoss() optimizer = optim.Adam(model.parameters()) ```    首先，让我们看看损失函数。它定义为 `nn.NLLLoss()`，代表“负对数似然损失”。别担心——没有人期望你在这个阶段就理解它的意思！最终，随着你学习如何进行机器学习，你会了解不同的损失函数，并尝试哪些在特定场景中效果最好。在这种情况下，鉴于输出 logits 是对数概率，我选择了这个损失函数，因为它特别适合这个场景。如前所述，随着时间的推移，你会更多地了解损失函数库，并能够为你的场景选择最佳的损失函数。但就目前而言，只需顺其自然，使用这个即可！    对于优化器，我选择使用 `Adam` 优化算法。它与我们在 第一章 中用于 *y* = 2*x* – 1 模型的随机梯度下降类似，但通常更快、更准确。与损失函数一样，随着时间的推移，你会更多地了解优化算法，并能够从最适合你场景的优化器菜单中进行选择。这里的一个重要事项是请注意，我已经将 `model.parameters()` 作为参数传递给这个。这个参数将模型中所有可训练的参数传递给优化器，以便它可以调整它们以帮助最小化损失函数计算的损失。    现在，让我们具体探讨一下，看看我们用于训练网络的代码是什么样的：    ```py # Train the model def train(dataloader, model, loss_fn, optimizer):     size = len(dataloader.dataset)     model.train()     for batch, (X, y) in enumerate(dataloader):         # Compute prediction and loss         pred = model(X)         loss = loss_fn(pred, y)           # Backpropagation         optimizer.zero_grad()         loss.backward()         optimizer.step()           if batch % 100 == 0:             loss, current = loss.item(), batch * len(X)             print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]") ```    虽然其中一些看起来很熟悉，因为它是基于 第一章 中的简单神经网络构建的，但由于我们使用了更多的数据，这里有一些新的概念。首先，你会看到我们获取数据集的 `size`。我们只是简单地使用它来报告进度，如最后一行所示。    然后，我们调用 `model.train` 来明确将模型设置为训练模式。PyTorch 在训练期间有一些优化超出了本章的范围。（为了利用这些优化，你将在训练和推理模式之间切换模型。）请注意，这更多的是模型的一个属性，而不是一个方法，但方法语法是存在的。如果有点令人困惑，请见谅！    接下来是这一有趣的行：    ```py     for batch, (X, y) in enumerate(dataloader): ```    让我们更详细地探讨一下。我们通过使用数据加载器使 Fashion MNIST 数据集可用于我们的代码。有 60,000 条记录可用于训练，每条记录有 784 个像素。这是一大批数据，你不必一次将所有数据都加载到内存中。`batch` 的想法是取数据的一部分——默认情况下是 64 个项目——并对其进行处理。枚举数据加载器给我们这个，所以我们将用 938 个批次进行训练，937 个批次为 64，最后一个批次为 32，因为 60,000 不能被 64 整除！    现在，对于每个批次，我们将进行与上一个例子中相同的循环。我们将从模型获取预测，计算损失，从损失函数中反向传播梯度，并使用新参数进行优化。    我们还将使用 *epoch* 术语来表示使用所有数据（即每个批次）的训练周期。然后我们可以每 100 个批次输出一次训练状态，以免输出控制台过载！    因此，为了训练网络五个 epoch，我们可以使用如下代码：    ```py # Training process epochs = 5 for t in range(epochs):     print(f"Epoch {t+1}\n-------------------------------")     train(train_loader, model, loss_function, optimizer) print("Done!") ```    这将简单地调用我们指定的 train 函数五次——通过计算预测、确定损失、优化参数并重复五次来使网络通过训练循环。    ```py` `````  ```py```````py ``````py``` # Training the Neural Network    Once you’ve executed the code, you’ll see the network train epoch by epoch. Then, after running the training, you’ll see something at the end that looks like this:    ``` Epoch 5 ------------------------------- loss: 0.429329  [    0/60000] loss: 0.348756  [ 6400/60000] loss: 0.237481  [12800/60000] loss: 0.336960  [19200/60000] loss: 0.435592  [25600/60000] loss: 0.272769  [32000/60000] loss: 0.362881  [38400/60000] loss: 0.202799  [44800/60000] loss: 0.354268  [51200/60000] loss: 0.205381  [57600/60000] Done! ```py    You can see here that over time, the loss has gone down. For example, in my case, the loss value at the end of the first epoch was .345, and by the end of the fifth epoch, it was .205\. This data shows us that the network is learning.    But how can we tell how *accurately* it’s learning? Note that loss and accuracy, while related, don’t have a direct linear relationship—for example, we can’t say that if loss is 20%, then accuracy is 80%. So, we need to go a little deeper.    Recall that when we were getting the data, we got *two* datasets: one for training and one for testing. Here’s a great place where we can write code to pass the test data through our network and evaluate how accurate the network is at predicting answers. We already know the correct answers, so we could do inference on all 10,000 test records, get the answers that the model predicts, and then check them against the ground truth for accuracy.    Here’s the code:    ``` # Function to test the model def test(dataloader, model):     size = len(dataloader.dataset)     num_batches = len(dataloader)     model.eval()  # Set the model to evaluation mode     test_loss, correct = 0, 0     with torch.no_grad():         for X, y in dataloader:             pred = model(X)             test_loss += loss_function(pred, y).item()             correct += (pred.argmax(1) ==                          y).type(torch.float).sum().item()     test_loss /= num_batches     correct /= size     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%,              `Avg` `loss``:` `{``test_loss``:``>``8``f``}` \`n``")` ```py `# Evaluate the model` `test``(``test_loader``,` `model``)` ``` ```py   ``````py` ``````py There are a few things to note in this code. First is the `model.eval()` line, which indicates that we are switching the model from training mode to inference mode. Similarly, `torch.no_grad()`will turn off gradient calculation in PyTorch to speed up inference. We’re no longer *training* the model, so we don’t need to do all the loss function backpropagation and optimization. We can just turn that off.    Then, as it does during training, the network just goes through every item in the data loader, gets the prediction for that item, and checks its correctness with this line:    ``` correct += (pred.argmax(1) ==  y).type(torch.float).sum().item() ```py    That’s a bit of a mouthful, so let’s break it down.    First, the `pred` value will give us the prediction from the network. The network outputs 10 values, each of which includes the probability of the class it represents being the correct one. Calling `argmax` on this will give us which one had the biggest value (i.e., the one with the probability closest to 1). The *y* value is the correct answer. For example, if we get a prediction, the neuron with the highest value is the sixth one, and *y* = 6, so we know we have a correct answer. Also, because we’re dealing in batches, we want to count each time `pred.argmax(1) == y` for this batch, hence, the `sum()`.    Therefore, our accuracy value will be the sum of correct items divided by the total number of items. So, when you run this code after training the model, you should see output like this:    ``` Test Error:  Accuracy: 86.9%, Avg loss: 0.366243 ```py    Remarkably, after running the neural network for only five epochs, we can see that it is 86.9% accurate on data it hadn’t previously seen!    At this point, you may be thinking that it’s really nice to see the accuracy of the model on the test set, but you may also be asking why we’ve only reported loss on the training—why not also report accuracy there? It seems silly to finish training the model by only looking at minimizing loss and *then* to figure out the accuracy. And you’d be right!    Fortunately, updating the model training code to *also* report on accuracy is pretty easy to do. Here’s a function called `get_accuracy()` that you can use during training:    ``` # Function to calculate accuracy def get_accuracy(pred, labels):     _, predictions = torch.max(pred, 1)     correct = (predictions == labels).float().sum()     accuracy = correct / labels.shape[0]     return accuracy ```py    Then, in your training loop, you can simply call this function after the loss function call like this:    ``` for batch, (X, y) in enumerate(dataloader):     # Compute prediction and loss     pred = model(X)     loss = loss_fn(pred, y)     accuracy = get_accuracy(pred, y)       # Backpropagation ```py    And when you’re reporting on the output of the training, you can use the accuracy metric like this:    ``` if batch % 100 == 0:     current = batch * len(X)     avg_loss = total_loss / (batch + 1)     avg_accuracy = total_accuracy / (batch + 1) * 100     print(f"Batch {batch}, Loss: {avg_loss:>7f},              `Accuracy``:` `{``avg_accuracy``:``>``0.2``f``}``%`                        `[{``current``:``>``5``d``}``/``{``size``:``>``5``d``}]``")` ```py   ````` ```py`Running this will give you output a bit like this:    ``` Epoch 5 ------------------------------- Batch 0, Loss: 0.177518, Accuracy: 95.31% [    0/60000] Batch 100, Loss: 0.304973, Accuracy: 88.89% [ 6400/60000] Batch 200, Loss: 0.311628, Accuracy: 88.51% [12800/60000] Batch 300, Loss: 0.307373, Accuracy: 88.63% [19200/60000] Batch 400, Loss: 0.309722, Accuracy: 88.67% [25600/60000] Batch 500, Loss: 0.310240, Accuracy: 88.60% [32000/60000] Batch 600, Loss: 0.306988, Accuracy: 88.70% [38400/60000] Batch 700, Loss: 0.308556, Accuracy: 88.64% [44800/60000] Batch 800, Loss: 0.309518, Accuracy: 88.67% [51200/60000] Batch 900, Loss: 0.311487, Accuracy: 88.59% [57600/60000] Done! ```py    Now, you’re probably wondering why the accuracy for the test data (86.9%) is *lower* than the accuracy for the training data (88.59%). This is very common, and when you think about it, it makes sense: the neural network only really knows how to match the inputs it has been trained on with the outputs for those values. Our hope is that given enough data, the network will be able to generalize from the examples it has seen and thus “learn” what a shoe or a dress looks like. But there will always be examples of items that it hasn’t seen that are also different enough from what it has seen to confuse it.    For example, if you grew up only ever seeing sneakers, then that’s what a shoe looks like to you. So, when you first see a high-heeled shoe, you might be a little confused. From your experience, it’s probably a shoe, but you don’t know for sure. That’s exactly what a neural network “thinks” when it “sees” inputs that are different enough from what it’s been trained on.```` ```py`` ``````py ``````py`  ``````py`` ``````py` ``````py # Exploring the Model Output    Now that we’ve trained the model and gotten a good gauge of its accuracy by using the test set, let’s explore it a little. Here’s a function we can use to predict a single image:    ```  import matplotlib.pyplot as plt   def predict_single_image(image, label, model):     # Set the model to evaluation mode     model.eval()   # Unsqueeze image as the model expects a batch dimension     image = image.unsqueeze(0)       with torch.no_grad():         prediction = model(image)         print(prediction)         predicted_label = prediction.argmax(1).item()       # Display the image and predictions     plt.imshow(image.squeeze(), cmap='gray')     plt.title(f'Predicted: {predicted_label}, Actual: {label}')     plt.show()       return predicted_label   # Choose an image from the test set image, label = test_dataset[0]  # Change index to test different images   # Predict the class for the chosen image predicted_label = predict_single_image(image, label, model) print(f"The model predicted {predicted_label}, and the actual label is {label}.") ```py    Let’s start with this code, which should look familiar to you now that you’ve seen the previous accuracy calculation code:    ```     with torch.no_grad():         prediction = model(image)         print(prediction)         predicted_label = prediction.argmax(1).item() ```py    Here, we get the `image`, send it to the `model`, get back a `prediction`, and `print` it out. Then, we get the `argmax` of that to show the label. Here’s an example output of the `prediction`:    ``` tensor([[–12.4290, –16.0639, –14.3148, –16.2861, –13.1672,  –4.5377, –13.6284,           –1.3124,  –8.9946,  –0.3285]]) ```py    These numbers may seem vague, but ultimately, our goal is simply to look for the biggest one! The `Softmax` function gets the `log()` of the value, where `log(1)` is zero and the log of any value less than one is a negative value. As you look through the list, you’ll notice that the value closest to 0 (–0.3285) is the very last one. This indicates that the function believes the class for this image should be class number 9\. (There are 10 classes in Fashion MNIST, which are numbered 0 through 9.)    Fashion MNIST’s class number 9 is “Ankle Boot,” so I’ve also included the code to render the image in Figure 2-6.    Also, as we can see, this is an example of where the model got the prediction right. The ground truth was that it’s label 9, and the prediction was for number 9\. Drawing the image so that we mere humans can compare the two also gives us an ankle boot!  ![](img/aiml_0206.png)  ###### Figure 2-6\. Exploring the output of the predictive model    Now, try a few different values for yourself and see if you can find anywhere the model gets it wrong.    # Overfitting    In the last example, we trained for only five epochs. That is, we went through the entire training loop of having the neurons randomly initialized and checked against their labels, then that performance was measured by the loss function and updated by the optimizer five times. And the results we got were pretty good: 88.59% accuracy on the training set and 86.5% on the test set. So what happens if we train for longer?    Next, try updating it to train for 50 epochs instead of 5\. In my case, I got the following accuracy figures on the training set:    ``` Epoch 50 ------------------------------- Batch 0, Loss: 0.077159, Accuracy: 96.88% [    0/60000] Batch 100, Loss: 0.094825, Accuracy: 96.57% [ 6400/60000] Batch 200, Loss: 0.093598, Accuracy: 96.67% [12800/60000] Batch 300, Loss: 0.095906, Accuracy: 96.54% [19200/60000] Batch 400, Loss: 0.096683, Accuracy: 96.48% [25600/60000] Batch 500, Loss: 0.101872, Accuracy: 96.31% [32000/60000] Batch 600, Loss: 0.103130, Accuracy: 96.22% [38400/60000] Batch 700, Loss: 0.103901, Accuracy: 96.17% [44800/60000] Batch 800, Loss: 0.104216, Accuracy: 96.15% [51200/60000] Batch 900, Loss: 0.104010, Accuracy: 96.15% [57600/60000] Done! ```py    This is particularly exciting because we’re doing much better: we’re getting 96.15% accuracy!    However, for the test set, accuracy reached 89.2%:    ``` Test Error:  Accuracy: 89.2%, Avg loss: 0.433885 ```py    So, we got a big improvement over the training set and a smaller one over the test set. This might suggest that training our network for much longer would lead to much better results—but that’s not always the case. The network is doing much better with the training data, but the model is not necessarily a better model. In fact, the divergence in the accuracy numbers shows that the model might have become overspecialized to the training data, in a process that’s often called *overfitting*. As you build more neural networks, this problem is something to watch out for—and as you go through this book, you’ll learn a number of techniques to avoid it!    # Early Stopping    In each of the cases so far, we’ve hardcoded the number of epochs we’re training for. While that works, we might want to train until we reach the desired accuracy instead of constantly trying different numbers of epochs and training and retraining until we get to our desired value. So, for example, if we want to train until the model is at 95% accuracy on the training set, and if we want to do it without knowing in advance how many epochs it will take. . .how can we do it?    Given that we’ve updated our code to check the accuracy as the model trained and to print it out, now, all we have to do is check that accuracy and end the training if it’s above a certain amount—such as 95% (or 0.95 when normalized). For example, we can do this:    ``` if batch % 100 == 0:     current = batch * len(X)     avg_loss = total_loss / (batch + 1)     avg_accuracy = total_accuracy / (batch + 1) * 100     print(f"Batch {batch}, Loss: {avg_loss:>7f},             `Accuracy``:` `{``avg_accuracy``:``>``0.2``f``}``%` `[{``current``:``>``5``d``}``/``{``size``:``>``5``d``}]``")` ```py `# Early stopping condition` `if` `avg_accuracy` `>=` `95``:`     `print``(``"Reached 95``% a``ccuracy, stopping training."``)`     `return` `True`  `# Stop training` ``` ```py   ````` ```py`Note that if we use this code inside the `if batch % 100 == 0` block, we can break the training loop before all batches in a particular epoch have been processed. It’s better to do this check at the end of the epoch, so we need to be sure to place the `if avg_accuracy >= 95` in the right place!    Now, when we’re training, at the end of every epoch, the average accuracy for the epoch will be calculated—and if it hits 95%, the training will stop. Previously, I had trained the model for 50 epochs to get 96.15% accuracy, but with this early stopping, where I’ve defined 95% as “good enough,” you can see that the model stopped training after only 37 epochs. Interestingly, accuracy was 94.99% for a couple of epochs before that, so I might have been able to stop even earlier!    This process of *early stopping* is very powerful in helping you save time as you evaluate different model architectures for solving specific problems. It helps you train your model until it’s “good enough,” instead of having a fixed training loop. For example, the process can look like this:    ``` Epoch 36 ------------------------------- Batch 0, Loss: 0.098307, Accuracy: 96.88% [    0/60000
