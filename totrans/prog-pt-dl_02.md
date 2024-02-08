# 第二章：使用 PyTorch 进行图像分类

在设置 PyTorch 之后，深度学习教材通常会在做任何有趣的事情之前向你抛出一堆行话。我尽量将其减少到最低限度，并通过一个例子来解释，尽管这个例子可以在你更熟悉使用 PyTorch 的过程中轻松扩展。我们在整本书中使用这个例子来演示如何调试模型(第七章)或将其部署到生产环境(第八章)。

从现在开始直到第四章结束，我们将构建一个*图像分类器*。神经网络通常用作图像分类器；网络被给予一张图片，并被问到对我们来说是一个简单的问题：“这是什么？”

让我们开始构建我们的 PyTorch 应用程序。

# 我们的分类问题

在这里，我们构建一个简单的分类器，可以区分鱼和猫之间的区别。我们将不断迭代设计和构建模型的过程，使其变得更加准确。

图 2-1 和 2-2 展示了一条鱼和一只猫的全貌。我不确定这条鱼是否有名字，但这只猫叫 Helvetica。

让我们从讨论传统分类中涉及的挑战开始。

![一条鱼的图片](img/ppdl_0201.png)

###### 图 2-1\. 一条鱼！

![一个黑猫在盒子里的图片](img/ppdl_0202.png)

###### 图 2-2\. 盒子里的 Helvetica

# 传统挑战

你会如何编写一个程序来区分鱼和猫？也许你会编写一组规则，描述猫有尾巴，或者鱼有鳞片，并将这些规则应用于图像以确定你看到的是什么。但这需要时间、精力和技能。另外，如果你遇到像曼克斯猫这样的东西会发生什么；虽然它显然是一只猫，但它没有尾巴。

你可以看到这些规则只会变得越来越复杂，以描述所有可能的情况。此外，我承认我在图形编程方面非常糟糕，所以不得不手动编写所有这些规则的想法让我感到恐惧。

我们追求的是一个函数，给定一张图片的输入，返回*猫*或*鱼*。对于我们来说，通过详细列出所有标准来构建这个函数是困难的。但深度学习基本上让计算机做所有那些我们刚刚谈到的规则的艰苦工作——只要我们创建一个结构，给网络大量数据，并让它找出是否得到了正确答案的方法。这就是我们要做的。在这个过程中，你将学习如何使用 PyTorch 的一些关键概念。

## 但首先，数据

首先，我们需要数据。需要多少数据？这取决于情况。对于任何深度学习技术都需要大量数据来训练神经网络的想法并不一定正确，正如你将在第四章中看到的那样。然而，现在我们将从头开始训练，这通常需要大量数据。我们需要很多鱼和猫的图片。

现在，我们可以花一些时间从 Google 图像搜索等地方下载许多图片，但在这种情况下，我们有一个捷径：一个用于训练神经网络的标准图像集合，称为*ImageNet*。它包含超过 1400 万张图片和 20000 个图像类别。这是所有图像分类器用来评判自己的标准。所以我从那里获取图片，但如果你愿意，可以自行下载其他图片。

除了数据，PyTorch 还需要一种确定什么是猫和什么是鱼的方法。这对我们来说很容易，但对计算机来说有点困难（这也是我们首次构建程序的原因！）。我们使用附加到数据的*标签*，以这种方式进行训练称为*监督学习*。（当您无法访问任何标签时，您必须使用*无监督学习*方法进行训练，这可能并不令人惊讶。）

现在，如果我们使用 ImageNet 数据，它的标签对我们来说并不是那么有用，因为它们包含了对我们来说*太多*的信息。*tabby cat*或*trout*这样的标签，在计算机看来，与*cat*或*fish*是分开的。我们需要重新标记这些。因为 ImageNet 是如此庞大的图像集合，我已经整理了一份[图像 URL 和标签的列表](https://oreil.ly/NbtEU)供鱼类和猫类使用。

您可以在该目录中运行*download.py*脚本，它将从 URL 下载图像并将其放置在适当的位置进行训练。*重新标记*很简单；脚本将猫的图片存储在*train/cat*目录中，将鱼的图片存储在*train/fish*目录中。如果您不想使用下载脚本，只需创建这些目录并将适当的图片放在正确的位置。现在我们有了数据，但我们需要将其转换为 PyTorch 可以理解的格式。

## PyTorch 和数据加载器

加载和转换数据为训练准备的格式通常会成为数据科学中吸收我们太多时间的领域之一。PyTorch 已经发展了与数据交互的标准约定，使得与之一起工作变得相当一致，无论您是在处理图像、文本还是音频。

与数据交互的两个主要约定是*数据集*和*数据加载器*。*数据集*是一个 Python 类，允许我们访问我们提供给神经网络的数据。*数据加载器*是将数据从数据集传送到网络的工具。（这可能包括信息，例如，*有多少个工作进程正在将数据传送到网络中？*或*我们一次传入多少张图片？*）

让我们先看看数据集。无论数据集包含图像、音频、文本、3D 景观、股市信息还是其他任何内容，只要满足这个抽象的 Python 类，就可以与 PyTorch 进行交互：

```py
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
```

这是相当直接的：我们必须实现一个返回数据集大小的方法（`len`），并实现一个可以检索数据集中项目的方法，返回一个（`*label*`，`*tensor*`）对。这是由数据加载器调用的，因为它正在将数据推送到神经网络进行训练。因此，我们必须编写一个`getitem`的主体，它可以获取图像并将其转换为张量，然后返回该张量和标签，以便 PyTorch 可以对其进行操作。这很好，但你可以想象到这种情况经常发生，所以也许 PyTorch 可以让事情变得更容易？

## 构建训练数据集

`torchvision`包含一个名为`ImageFolder`的类，几乎为我们做了一切，只要我们的图像结构中每个目录都是一个标签（例如，所有猫都在一个名为*cat*的目录中）。对于我们的猫和鱼的示例，这是您需要的：

```py
import torchvision
from torchvision import transforms

train_data_path = "./train/"

transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder
(root=train_data_path,transform=transforms)
```

这里发生了更多的事情，因为`torchvision`还允许您指定一系列将应用于图像的转换，然后将其馈送到神经网络之前。默认转换是将图像数据转换为张量（在前面的代码中看到的`transforms.ToTensor()`方法），但我们还在做一些其他可能不太明显的事情。

首先，GPU 被设计为快速执行标准大小的计算。但我们可能有许多分辨率的图像。为了提高我们的处理性能，我们通过`Resize(64)`转换将每个传入的图像缩放到相同的分辨率 64×64。然后我们将图像转换为张量，最后我们将张量归一化到一组特定的均值和标准差点周围。

归一化很重要，因为当输入通过神经网络的层时会发生大量的乘法运算；保持输入值在 0 和 1 之间可以防止值在训练阶段变得过大（称为*梯度爆炸*问题）。这种神奇的化身只是 ImageNet 数据集作为整体的均值和标准差。你可以专门为这个猫和鱼子集计算它，但这些值已经足够好了。（如果你在完全不同的数据集上工作，你将不得不计算那个均值和偏差，尽管许多人只是使用这些 ImageNet 常数并报告可接受的结果。）

可组合的转换还允许我们轻松地进行图像旋转和扭曲以进行数据增强，我们将在第四章中回到这个话题。

###### 注意

在这个例子中，我们将图像调整为 64×64。我做出了这个任意选择，以便使我们即将到来的第一个网络的计算变得快速。大多数现有的架构在第三章中使用 224×224 或 299×299 作为图像输入。一般来说，输入尺寸越大，网络学习的数据就越多。另一方面，你通常可以将更小的图像批次适应到 GPU 的内存中。

我们对数据集还没有完成。但是为什么我们需要不止一个训练数据集呢？

## 构建验证和测试数据集

我们的训练数据已经设置好了，但我们需要为我们的*验证*数据重复相同的步骤。这里有什么区别？深度学习（实际上所有机器学习）的一个危险是*过拟合*的概念：你的模型在训练过的内容上表现得非常好，但无法推广到它没有见过的例子。所以它看到一张猫的图片，除非所有其他猫的图片都与那张图片非常相似，否则模型不认为它是一只猫，尽管它显然是一只猫。为了防止我们的网络这样做，我们在*download.py*中下载了一个*验证集*，其中包含一系列不在训练集中出现的猫和鱼的图片。在每个训练周期（也称为*epoch*）结束时，我们会与这个集合进行比较，以确保我们的网络没有出错。但不用担心，这段代码非常简单，因为它只是稍微更改了一些变量名的早期代码：

```py
val_data_path = "./val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path,
                                            transform=transforms)
```

我们只是重新使用了`transforms`链，而不必再次定义它。

除了验证集，我们还应该创建一个*测试集*。这用于在所有训练完成后测试模型：

```py
test_data_path = "./test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,
                                             transform=transforms)
```

区分数据集类型可能有点困惑，所以我编制了一张表来指示哪个数据集用于模型训练的哪个部分；请参见表 2-1。

表 2-1. 数据集类型

| 训练集 | 用于训练过程中更新模型的数据集 |
| --- | --- |
| 验证集 | 用于评估模型在问题领域中的泛化能力，而不是适应训练数据；不直接用于更新模型 |
| 测试集 | 在训练完成后提供最终评估模型性能的最终数据集 |

然后我们可以用几行 Python 代码构建我们的数据加载器：

```py
batch_size=64
train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader  = data.DataLoader(val_data, batch_size=batch_size)
test_data_loader  = data.DataLoader(test_data, batch_size=batch_size)
```

从这段代码中需要注意的新内容是`batch_size`。这告诉我们在训练和更新之前有多少图像会通过网络。理论上，我们可以将`batch_size`设置为测试集和训练集中图像的数量，以便网络在更新之前看到每个图像。实际上，我们通常不这样做，因为较小的批次（在文献中更常被称为*小批量*）需要比存储数据集中*每个*图像的所有信息更少的内存，并且较小的批次大小会使训练更快，因为我们更快地更新我们的网络。

默认情况下，PyTorch 的数据加载器设置为`batch_size`为 1。你几乎肯定会想要更改这个值。虽然我在这里选择了 64，但你可能想要尝试一下，看看你可以使用多大的小批量而不会耗尽 GPU 的内存。你可能还想尝试一些额外的参数：你可以指定数据集如何被采样，是否在每次运行时对整个集合进行洗牌，以及使用多少个工作进程来从数据集中提取数据。所有这些都可以在[PyTorch 文档](https://oreil.ly/XORs1)中找到。

这涵盖了将数据导入 PyTorch，所以现在让我们介绍一个简单的神经网络来开始对我们的图像进行分类。

# 最后，一个神经网络！

我们将从最简单的深度学习网络开始：一个输入层，用于处理输入张量（我们的图像）；一个输出层，其大小将是输出类别数量（2）的大小；以及它们之间的一个隐藏层。在我们的第一个示例中，我们将使用全连接层。图 2-3 展示了一个具有三个节点的输入层，三个节点的隐藏层和两个节点输出的样子。

![一个简单的神经网络](img/ppdl_0203.png)

###### 图 2-3\. 一个简单的神经网络

正如你所看到的，在这个全连接的例子中，每一层中的每个节点都会影响到下一层中的每个节点，并且每个连接都有一个*权重*，它决定了从该节点传入下一层的信号的强度。当我们训练网络时，这些权重通常会从随机初始化中更新。当一个输入通过网络时，我们（或 PyTorch）可以简单地将该层的权重和偏置进行矩阵乘法，然后将结果传递到下一个函数中，该结果会经过一个*激活函数*，这只是一种在我们的系统中插入非线性的方法。

## 激活函数

激活函数听起来很复杂，但你在文献中最常见的激活函数是`ReLU`，或者*修正线性单元*。这再次听起来很复杂！但事实证明，它只是实现*max(0,x)*的函数，所以如果输入是负数，则结果为 0，如果*x*是正数，则结果就是输入（*x*）。简单！

你可能会遇到的另一个激活函数是*softmax*，在数学上稍微复杂一些。基本上它会产生一组介于 0 和 1 之间的值，加起来等于 1（概率！），并且加权这些值以夸大差异——也就是说，它会在向量中产生一个比其他所有值都高的结果。你经常会看到它被用在分类网络的末尾，以确保网络对输入属于哪个类别做出明确的预测。

有了所有这些构建块，我们可以开始构建我们的第一个神经网络。

## 创建一个网络

在 PyTorch 中创建一个网络是一个非常 Pythonic 的事情。我们从一个名为`torch.nn.Network`的类继承，并填写`__init__`和`forward`方法：

```py
class SimpleNet(nn.Module):

def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(12288, 84)
    self.fc2 = nn.Linear(84, 50)
    self.fc3 = nn.Linear(50,2)

def forward(self):
    x = x.view(-1, 12288)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.softmax(self.fc3(x))
    return x

simplenet = SimpleNet()
```

同样，这并不太复杂。我们在`init()`中进行任何所需的设置，这种情况下调用我们的超类构造函数和三个全连接层（在 PyTorch 中称为`Linear`，而不是 Keras 中的`Dense`）。`forward()`方法描述了数据如何在网络中流动，无论是在训练还是进行预测（*推理*）。首先，我们必须将图像中的 3D 张量（*x*和*y*加上三通道的颜色信息—红色、绿色、蓝色）转换为 1D 张量，以便将其馈送到第一个`Linear`层中，我们使用`view()`来实现这一点。从那里，您可以看到我们按顺序应用层和激活函数，最后返回`softmax`输出以给出我们对该图像的预测。

隐藏层中的数字有些是任意的，除了最终层的输出是 2，与我们的两类猫或鱼相匹配。一般来说，您希望在层中的数据在向下堆栈时*压缩*。如果一个层要将 50 个输入传递到 100 个输出，那么网络可能会通过简单地将 50 个连接传递给 100 个输出中的 50 个来*学习*，并认为其工作完成。通过减小输出相对于输入的大小，我们迫使网络的这部分学习使用更少的资源来学习原始输入的表示，这希望意味着它提取了一些对我们要解决的问题重要的图像特征；例如，学习识别鳍或尾巴。

我们有一个预测，我们可以将其与原始图像的实际标签进行比较，以查看预测是否正确。但是我们需要一种让 PyTorch 能够量化预测是正确还是错误，以及有多错误或正确的方法。这由损失函数处理。

## 损失函数

*损失函数*是有效深度学习解决方案的关键组成部分之一。PyTorch 使用损失函数来确定如何更新网络以达到期望的结果。

损失函数可以是您想要的复杂或简单。PyTorch 配备了一个全面的损失函数集合，涵盖了您可能会遇到的大多数应用程序，当然，如果您有一个非常自定义的领域，您也可以编写自己的损失函数。在我们的情况下，我们将使用一个名为`CrossEntropyLoss`的内置损失函数，这是推荐用于多类别分类任务的，就像我们在这里所做的那样。您可能会遇到的另一个损失函数是`MSELoss`，这是一个标准的均方损失，您可能在进行数值预测时使用。

要注意的一件事是，`CrossEntropyLoss`还将`softmax()`作为其操作的一部分，因此我们的`forward()`方法变为以下内容：

```py
def forward(self):
    # Convert to 1D vector
    x = x.view(-1, 12288)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

现在让我们看看在训练循环期间神经网络的层如何更新。

## 优化

训练网络涉及通过网络传递数据，使用损失函数确定预测和实际标签之间的差异，然后使用该信息来更新网络的权重，以尽可能使损失函数返回尽可能小的损失。为了对神经网络进行更新，我们使用一个*优化器*。

如果我们只有一个权重，我们可以绘制损失值与权重值的图表，它可能看起来像图 2-4。

![损失的二维图](img/ppdl_0204.png)

###### 图 2-4。损失的二维图

如果我们从一个随机位置开始，用 X 标记，将我们的权重值放在 x 轴上，损失函数放在 y 轴上，我们需要到曲线的最低点找到我们的最佳解决方案。我们可以通过改变权重的值来移动，这将给我们一个新的损失函数值。要知道我们正在做出的移动有多好，我们可以根据曲线的梯度进行检查。可视化优化器的一种常见方法是像滚动大理石一样，试图找到一系列山谷中的最低点（或*最小值*）。如果我们将视图扩展到两个参数，创建一个如图 2-5 所示的 3D 图，这可能更清晰。

![损失的 3D 图](img/ppdl_0205.png)

###### 图 2-5。损失的 3D 图

在这种情况下，我们可以在每个点检查所有潜在移动的梯度，并选择使我们在山下移动最多的那个。

但是，您需要注意一些问题。首先是陷入*局部最小值*的危险，这些区域看起来像是损失曲线最浅的部分，如果我们检查梯度，但实际上在其他地方存在更浅的区域。如果我们回到图 2-4 中的 1D 曲线，我们可以看到如果通过短跳下陷入左侧的最小值，我们永远不会有离开该位置的理由。如果我们采取巨大的跳跃，我们可能会发现自己进入通往实际最低点的路径，但由于我们一直跳得太大，我们一直在到处弹跳。

我们的跳跃大小被称为*学习率*，通常是需要调整的*关键*参数，以便使您的网络学习正确和高效。您将在第四章中看到确定良好学习率的方法，但现在，您将尝试不同的值：尝试从 0.001 开始。正如刚才提到的，较大的学习率会导致网络在训练过程中到处反弹，并且不会*收敛*到一组良好的权重上。

至于局部最小值问题，我们对获取所有可能梯度进行了轻微修改，并在批处理期间指示样本随机梯度。称为*随机梯度下降*（SGD），这是优化神经网络和其他机器学习技术的传统方法。但是还有其他优化器可用，事实上对于深度学习来说更可取。PyTorch 提供了 SGD 和其他优化器，如 AdaGrad 和 RMSProp，以及 Adam，我们将在本书的大部分内容中使用的优化器。

Adam 的一个关键改进（RMSProp 和 AdaGrad 也是如此）是它为每个参数使用一个学习率，并根据这些参数的变化速率调整该学习率。它保持梯度和这些梯度的平方的指数衰减列表，并使用这些来缩放 Adam 正在使用的全局学习率。经验表明，Adam 在深度学习网络中优于大多数其他优化器，但您可以将 Adam 替换为 SGD 或 RMSProp 或另一个优化器，以查看是否使用不同的技术能够为您的特定应用程序提供更快更好的训练。

创建基于 Adam 的优化器很简单。我们调用`optim.Adam()`并传入网络的权重（通过`simplenet.parameters()`获得）和我们示例的学习率 0.001：

```py
import torch.optim as optim
optimizer = optim.Adam(simplenet.parameters(), lr=0.001)
```

优化器是拼图的最后一块，所以我们终于可以开始训练我们的网络了。

# 训练

这是我们完整的训练循环，将迄今为止看到的所有内容结合起来训练网络。我们将其编写为一个函数，以便可以将诸如损失函数和优化器之类的部分作为参数传递。目前看起来相当通用：

```py
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input, target = batch
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
```

这是相当简单的，但你应该注意几点。我们在循环的每次迭代中从训练集中取一个批次，这由我们的数据加载器处理。然后我们通过模型运行这些数据，并计算出期望输出的损失。为了计算梯度，我们在模型上调用`backward()`方法。`optimizer.step()`方法随后使用这些梯度来执行我们在前一节中讨论过的权重调整。

然而，`zero_grad()`调用是在做什么呢？事实证明，默认情况下计算的梯度会累积，这意味着如果我们在批次迭代结束时不将梯度清零，下一个批次将不得不处理这个批次的梯度以及自己的梯度，接下来的批次将不得不处理前两个批次的梯度，依此类推。这并不有用，因为我们希望在每次迭代中只查看当前批次的梯度进行优化。我们使用`zero_grad()`确保在我们完成循环后将它们重置为零。

这是训练循环的抽象版本，但在写完我们的完整函数之前，我们还需要解决一些问题。

## 使其在 GPU 上运行

到目前为止，如果你运行了任何代码，你可能已经注意到它并不那么快。那么那块闪亮的 GPU 呢，它就坐在我们云端实例上（或者我们在桌面上组装的非常昂贵的机器上）？PyTorch 默认使用 CPU 进行计算。为了利用 GPU，我们需要通过显式地使用`to()`方法将输入张量和模型本身移动到 GPU 上。这里有一个将`SimpleNet`复制到 GPU 的示例：

```py
if torch.cuda.is_available():
        device = torch.device("cuda")
else
    device = torch.device("cpu")

model.to(device)
```

在这里，如果 PyTorch 报告有 GPU 可用，我们将模型复制到 GPU 上，否则保持模型在 CPU 上。通过使用这种构造，我们可以确定 GPU 是否在我们的代码开始时可用，并在程序的其余部分中使用`tensor|model.to(device)`，确信它会到达正确的位置。

###### 注意

在早期版本的 PyTorch 中，你会使用`cuda()`方法将数据复制到 GPU 上。如果在查看其他人的代码时遇到这个方法，只需注意它与`to()`做的是相同的事情！

这就是训练所需的所有步骤。我们快要完成了！

# 将所有内容整合在一起

在本章中，你已经看到了许多不同的代码片段，让我们整合它们。我们将它们放在一起，创建一个通用的训练方法，接受一个模型，以及训练和验证数据，还有学习率和批次大小选项，并对该模型进行训练。我们将在本书的其余部分中使用这段代码：

```py
def train(model, optimizer, loss_fn, train_loader, val_loader,
epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, target = batch
            inputs = inputs.to(device)
            target = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_iterator)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1],
							   target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(valid_iterator)

        print('Epoch: {}, Training Loss: {:.2f},
        Validation Loss: {:.2f},
        accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))
```

这是我们的训练函数，我们可以通过传入所需的参数来启动训练：

```py
train(simplenet, optimizer, torch.nn.CrossEntropyLoss(),
      train_data_loader, test_data_loader,device)
```

网络将训练 20 个 epochs（你可以通过向`train()`传入一个值来调整这个值），并且你应该在每个 epoch 结束时得到模型在验证集上的准确性打印输出。

你已经训练了你的第一个神经网络——恭喜！现在你可以用它进行预测，让我们看看如何做到这一点。

## 进行预测

在本章的开头，我说过我们将制作一个神经网络，可以对图像进行分类，判断是猫还是鱼。我们现在已经训练了一个可以做到这一点的网络，但是我们如何使用它来为单个图像生成预测呢？这里有一段快速的 Python 代码，它将从文件系统加载一张图像，并打印出我们的网络是说“猫”还是“鱼”：

```py
from PIL import Image

labels = ['cat','fish']

img = Image.open(FILENAME)
img = transforms(img)
img = img.unsqueeze(0)

prediction = simplenet(img)
prediction = prediction.argmax()
print(labels[prediction])
```

大部分代码都很简单；我们重用了之前制作的转换流水线，将图像转换为神经网络所需的正确形式。然而，因为我们的网络使用批次，实际上它期望一个 4D 张量，第一个维度表示批次中的不同图像。我们没有批次，但我们可以通过使用`unsqueeze(0)`创建一个长度为 1 的批次，这会在我们的张量前面添加一个新的维度。

获取预测就像将我们的*batch*传递到模型中一样简单。然后我们必须找出具有更高概率的类别。在这种情况下，我们可以简单地将张量转换为数组并比较两个元素，但通常情况下不止这两个元素。幸运的是，PyTorch 提供了`argmax()`函数，它返回张量中最高值的索引。然后我们使用该索引来索引我们的标签数组并打印出我们的预测。作为练习，使用前面的代码作为基础，在本章开头创建的测试集上进行预测。您不需要使用`unsqueeze()`，因为您从`test_data_loader`中获取批次。

这就是您现在需要了解的有关进行预测的全部内容；在第八章中，我们将为生产使用加固事项时再次回顾这一点。

除了进行预测，我们可能希望能够在将来的任何时间点重新加载模型，使用我们训练好的参数，因此让我们看看如何在 PyTorch 中完成这个任务。

## 模型保存

如果您对模型的性能感到满意或因任何原因需要停止，您可以使用`torch.save()`方法将模型的当前状态保存为 Python 的*pickle*格式。相反，您可以使用`torch.load()`方法加载先前保存的模型迭代。

因此，保存我们当前的参数和模型结构将像这样工作：

```py
torch.save(simplenet, "/tmp/simplenet")
```

我们可以按以下方式重新加载：

```py
simplenet = torch.load("/tmp/simplenet")
```

这将模型的参数和结构都存储到文件中。如果以后更改模型的结构，这可能会成为一个问题。因此，更常见的做法是保存模型的`state_dict`。这是一个标准的 Python`dict`，其中包含模型中每个层的参数映射。保存`state_dict`看起来像这样：

```py
torch.save(model.state_dict(), PATH)
```

要恢复，首先创建模型的一个实例，然后使用`load_state_dict`。对于`SimpleNet`：

```py
simplenet = SimpleNet()
simplenet_state_dict = torch.load("/tmp/simplenet")
simplenet.load_state_dict(simplenet_state_dict)
```

这里的好处是，如果以某种方式扩展了模型，可以向`load_state_dict`提供一个`strict=False`参数，该参数将参数分配给模型中存在的层，但如果加载的`state_dict`中的层缺失或添加到模型的当前结构中，则不会失败。因为它只是一个普通的 Python`dict`，您可以更改键名称以适应您的模型，如果您从完全不同的模型中提取参数，这可能会很方便。

在训练运行期间可以将模型保存到磁盘，并在另一个时间点重新加载，以便可以在离开的地方继续训练。当使用像 Google Colab 这样的工具时，这非常有用，它让您在大约 12 小时内持续访问 GPU。通过跟踪时间，您可以在截止日期之前保存模型，并在新的 12 小时会话中继续训练。

# 结论

您已经快速浏览了神经网络的基础知识，并学会了如何使用 PyTorch 对其进行训练，对其他图像进行预测，并将模型保存/恢复到磁盘。

在阅读下一章之前，尝试一下我们在这里创建的`SimpleNet`架构。调整`Linear`层中的参数数量，也许添加一两个额外的层。查看 PyTorch 中提供的各种激活函数，并将`ReLU`替换为其他函数。看看如果调整学习率或将优化器从 Adam 切换到其他选项（也许尝试普通的 SGD），训练会发生什么变化。也许改变批量大小和图像在前向传递开始时被转换为 1D 张量的初始大小。许多深度学习工作仍处于手工调整阶段；学习率是手动调整的，直到网络被适当训练，因此了解所有移动部件如何相互作用是很重要的。

你可能对`SimpleNet`架构的准确性有些失望，但不用担心！第三章将引入卷积神经网络，带来明显的改进，取代我们目前使用的非常简单的网络。

# 进一步阅读

+   [PyTorch 文档](https://oreil.ly/x6pO7)

+   《Adam：一种随机优化方法》（2014）作者 Diederik P. Kingma 和 Jimmy Ba

+   《梯度下降优化算法概述》（2016）作者 Sebstian Ruder