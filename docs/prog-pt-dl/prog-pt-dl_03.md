# 第三章：卷积神经网络

在第二章中尝试使用全连接神经网络后，您可能注意到了一些问题。如果您尝试添加更多层或大幅增加参数数量，您几乎肯定会在 GPU 上耗尽内存。此外，训练时间很长，准确率也不尽如人意，尤其考虑到深度学习的炒作。到底发生了什么呢？

确实，全连接或(*前馈*)网络可以作为通用逼近器，但理论并没有说明训练它成为您真正想要的函数逼近器需要多长时间。但我们可以做得更好，尤其是对于图像。在本章中，您将了解*卷积神经网络*（CNNs）以及它们如何构成当今最准确的图像分类器的基础（我们会详细看一些）。我们为我们的鱼与猫应用程序构建了一个基于卷积的新架构，并展示它比我们在上一章中所做的更快速*和*更准确。让我们开始吧！

# 我们的第一个卷积模型

这一次，我将首先分享最终的模型架构，然后讨论所有新的部分。正如我在第二章中提到的，我们创建的训练方法与模型无关，因此您可以先测试这个模型，然后再回来了解解释！

```py
class CNNNet(nn.Module):

    def __init__(self, num_classes=2):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
      x = self.features(x)
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)
      return x
```

第一件要注意的事情是使用`nn.Sequential()`。这使我们能够创建一系列层。当我们在`forward()`中使用这些链中的一个时，输入会依次通过每个层的数组元素。您可以使用这个方法将模型分解为更合理的安排。在这个网络中，我们有两个链：`features`块和`classifier`。让我们看看我们正在引入的新层，从`Conv2d`开始。

## 卷积

`Conv2d`层是*2D 卷积*。如果我们有一个灰度图像，它由一个数组组成，*x*像素宽，*y*像素高，每个条目的值表示它是黑色、白色还是介于两者之间（我们假设是 8 位图像，因此每个值可以从 0 到 255 变化）。对于这个例子，我们看一个 4 像素高宽的小方形图像：

<math display="block"><mfenced close="]" open="["><mtable><mtr><mtd><mn>10</mn></mtd> <mtd><mn>11</mn></mtd> <mtd><mn>9</mn></mtd> <mtd><mn>3</mn></mtd></mtr> <mtr><mtd><mn>2</mn></mtd> <mtd><mn>123</mn></mtd> <mtd><mn>4</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>45</mn></mtd> <mtd><mn>237</mn></mtd> <mtd><mn>23</mn></mtd> <mtd><mn>99</mn></mtd></mtr> <mtr><mtd><mn>20</mn></mtd> <mtd><mn>67</mn></mtd> <mtd><mn>22</mn></mtd> <mtd><mn>255</mn></mtd></mtr></mtable></mfenced></math>

接下来我们介绍一种叫做*filter*或*卷积核*的东西。这是另一个矩阵，很可能更小，我们将它拖过我们的图像。这是我们的 2×2 filter：

<math display="block"><mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced></math>

为了产生我们的输出，我们取较小的 filter 并将其传递到原始输入上，就像放大镜放在一张纸上一样。从左上角开始，我们的第一个计算如下：

<math display="block"><mrow><mfenced close="]" open="["><mtable><mtr><mtd><mn>10</mn></mtd> <mtd><mn>11</mn></mtd></mtr> <mtr><mtd><mn>2</mn></mtd> <mtd><mn>123</mn></mtd></mtr></mtable></mfenced> <mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mrow></math>

我们所做的就是将矩阵中的每个元素与另一个矩阵中的对应成员相乘，并求和结果：(`10` × `1`) + (`11` × `0`) + (`2` × `1`) + (`123` × `0`) = `12`。做完这个之后，我们将滤波器移动并重新开始。但是我们应该移动滤波器多少？在这种情况下，我们将滤波器移动 2 个单位，这意味着我们的第二次计算是：

<math display="block"><mrow><mfenced close="]" open="["><mtable><mtr><mtd><mn>9</mn></mtd> <mtd><mn>3</mn></mtd></tr> <mtr><mtd><mn>4</mn></mtd> <mtd><mn>0</mn></mtd></tr></mtable></mfenced> <mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd> <mtd><mn>0</m></mtd></tr> <mtr><mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd></tr></mtable></mfenced></mrow></math>

这给我们一个输出为 13。现在我们将滤波器向下移动并向左移动，重复这个过程，给出这个最终结果（或*特征图*）：

<math display="block"><mfenced close="]" open=""><mtable><mtr><mtd><mn>12</mn></mtd> <mtd><mn>13</mn></mtd></mtr> <mtr><mtd><mn>65</mn></mtd> <mtd><mn>45</mn></m></tr></mtable></mfenced></math>

在图 3-1 中，您可以看到这是如何以图形方式工作的，一个 3×3 卷积核被拖动到一个 4×4 张量上，并产生一个 2×2 的输出（尽管每个部分基于九个元素而不是我们第一个示例中的四个）。

![3x3 卷积核在 4x4 输入上的操作

###### 图 3-1。3×3 卷积核在 4×4 输入上的操作

卷积层将有许多这样的滤波器，这些滤波器的值是由网络的训练填充的，该层中的所有滤波器共享相同的偏置值。让我们回到如何调用`Conv2d`层并看看我们可以设置的其他选项：

```py
nn.Conv2d(in_channels,out_channels, kernel_size, stride, padding)
```

`in_channels`是我们在该层接收的输入通道数。在网络的开始，我们将 RGB 图像作为输入，因此输入通道数为三。`out_channels`是输出通道数，对应于卷积层中的滤波器数量。接下来是`kernel_size`，描述了滤波器的高度和宽度。¹ 这可以是一个指定正方形的单个标量（例如，在第一个卷积层中，我们设置了一个 11×11 的滤波器），或者您可以使用一个元组（例如(3,5)表示一个 3×5 的滤波器）。

接下来的两个参数似乎无害，但它们可能对网络的下游层产生重大影响，甚至影响该特定层最终查看的内容。`stride`表示我们在调整滤波器到新位置时在输入上移动多少步。在我们的示例中，我们最终得到步幅为 2，这使得特征图的大小是输入的一半。但我们也可以使用步幅为 1 移动，这将给我们一个 4×4 的特征图输出，与输入的大小相同。我们还可以传入一个元组*(a,b)*，允许我们在每一步上移动*a*个单位横向和*b*个单位纵向。现在，您可能想知道，当它到达末尾时会发生什么。让我们看看。如果我们以步幅 1 拖动我们的滤波器，最终会到达这一点：

<math display="block"><mfenced close="]" open="["><mtable><mtr><mtd><mn>3</mn></mtd> <mtd><mo>?</mo></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mo>?</mo></mtd></mtr></mtable></mfenced></math>

我们的输入中没有足够的元素来进行完整的卷积。那么会发生什么？这就是`padding`参数发挥作用的地方。如果我们给出`padding`值为 1，我们的输入看起来有点像这样：

<math display="block"><mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>10</mn></mtd> <mtd><mn>11</mn></mtd> <mtd><mn>9</mn></mtd> <mtd><mn>3</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>2</mn></mtd> <mtd><mn>123</mn></mtd> <mtd><mn>4</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>45</mn></mtd> <mtd><mn>237</mn></mtd> <mtd><mn>23</mn></mtd> <mtd><mn>99</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>20</mn></mtd> <mtd><mn>67</mn></mtd> <mtd><mn>22</mn></mtd> <mtd><mn>255</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced></math>

现在当我们到达边缘时，我们的过滤器覆盖的值如下：

<math display="block"><mfenced close="]" open="["><mtable><mtr><mtd><mn>3</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced></math>

如果不设置填充，PyTorch 在输入的最后几列中遇到的任何边缘情况都会被简单地丢弃。您需要适当设置填充。与`stride`和`kernel_size`一样，您也可以传入一个`height`×`weight`填充的元组，而不是填充相同的单个数字。

这就是我们模型中的`Conv2d`层在做的事情。但是那些`MaxPool2d`层呢？

## 池化

与卷积层一起，您经常会看到*池化*层。这些层将网络的分辨率从前一个输入层降低，这使得我们在较低层中有更少的参数。这种压缩导致计算速度更快，有助于防止网络过拟合。

在我们的模型中，我们使用了一个核大小为 3，步长为 2 的`MaxPool2d`。让我们通过一个示例来看看它是如何工作的。这是一个 5×3 的输入：

<math display="block"><mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd> <mtd><mn>2</mn></mtd> <mtd><mn>1</mn></mtd> <mtd><mn>4</mn></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>5</mn></mtd> <mtd><mn>6</mn></mtd> <mtd><mn>1</mn></mtd> <mtd><mn>2</mn></mtd> <mtd><mn>5</mn></mtd></mtr> <mtr><mtd><mn>5</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>9</mn></mtd> <mtd><mn>6</mn></mtd></mtr></mtable></mfenced></math>

使用 3×3 的核大小和步长 2，我们从池化中得到两个 3×3 的张量：

<math display="block"><mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd> <mtd><mn>2</mn></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>5</mn></mtd> <mtd><mn>6</mn></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>5</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced></math><math display="block"><mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd> <mtd><mn>4</mn></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd> <mtd><mn>2</mn></mtd> <mtd><mn>5</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>9</mn></mtd> <mtd><mn>6</mn></mtd></mtr></mtable></mfenced></math>

在`MaxPool`中，我们从这些张量中取最大值，得到一个输出张量为[6,9]。就像在卷积层中一样，`MaxPool`有一个`padding`选项，可以在张量周围创建一个零值边界，以防步长超出张量窗口。

正如你可以想象的那样，除了从内核中取最大值之外，你还可以使用其他函数进行池化。一个流行的替代方法是取张量值的平均值，这样允许所有张量数据都参与到池中，而不仅仅是`max`情况下的一个值（如果你考虑一幅图像，你可以想象你可能想要考虑像素的最近邻）。此外，PyTorch 提供了`AdaptiveMaxPool`和`AdaptiveAvgPool`层，它们独立于传入输入张量的维度工作（例如，我们的模型中有一个`AdaptiveAvgPool`）。我建议在构建模型架构时使用这些，而不是标准的`MaxPool`或`AvgPool`层，因为它们允许你创建可以处理不同输入维度的架构；在处理不同数据集时这很方便。

我们还有一个新组件要讨论，这个组件非常简单但对训练非常重要。

## Dropout

神经网络的一个经常出现的问题是它们倾向于过拟合训练数据，深度学习领域正在进行大量工作，以确定允许网络学习和泛化到非训练数据的方法，而不仅仅是学习如何对训练输入做出响应。`Dropout` 层是一个极其简单但重要的方法，它易于理解且有效：如果我们在训练周期内不训练网络中的一组随机节点会怎样？因为它们不会被更新，它们就不会有机会过拟合输入数据，而且因为是随机的，每个训练周期将忽略不同的输入选择，这应该进一步帮助泛化。

在我们示例 CNN 网络中，默认情况下，`Dropout` 层的初始化为`0.5`，意味着输入张量的 50%会被随机置零。如果你想将其更改为 20%，请在初始化调用中添加`p`参数：`Dropout(p=0.2)`。

###### 注意

`Dropout` 应该只在训练期间发生。如果在推理时发生，你会失去网络推理能力的一部分，这不是我们想要的！幸运的是，PyTorch 的`Dropout`实现会根据你运行的模式来确定，并在推理时通过`Dropout`层传递所有数据。

在查看了我们的小型 CNN 模型并深入研究了层类型之后，让我们看看过去十年中制作的其他模型。

# CNN 架构的历史

尽管 CNN 模型已经存在几十年了（例如，LeNet-5 在 1990 年代末用于支票上的数字识别），但直到 GPU 变得广泛可用，深度 CNN 网络才变得实用。即使是在那时，深度学习网络开始压倒所有其他现有方法在图像分类中的应用也仅有七年。在本节中，我们将回顾过去几年的一些 CNN 学习里程碑，并探讨一些新技术。

## AlexNet

*AlexNet* 在许多方面改变了一切。它于 2012 年发布，并在当年的 ImageNet 竞赛中以 15.3%的前五错误率摧毁了所有其他参赛作品（第二名的前五错误率为 26.2%，这让你了解了它比其他最先进方法好多少）。AlexNet 是最早引入`MaxPool`和`Dropout`概念的架构之一，甚至推广了当时不太知名的`ReLU`激活函数。它是最早证明许多层次在 GPU 上训练是可能且高效的架构之一。虽然它不再是最先进的，但仍然是深度学习历史上的重要里程碑。

AlexNet 架构是什么样的？啊哈，是时候让你知道一个小秘密了。我们在本章中迄今为止一直在使用的网络？就是 AlexNet。惊喜！这就是为什么我们使用标准的`MaxPool2d`而不是`AdaptiveMaxPool2d`，以匹配原始的 AlexNet 定义。

## Inception/GoogLeNet

让我们直接跳到 2014 年 ImageNet 比赛的获胜者。GoogLeNet 架构引入了*Inception*模块，解决了 AlexNet 的一些缺陷。在该网络中，卷积层的卷积核被固定在某个分辨率上。我们可能期望图像在宏观和微观尺度上都有重要的细节。使用较大的卷积核可能更容易确定一个对象是否是汽车，但要确定它是 SUV 还是掀背车可能需要一个较小的卷积核。而要确定车型，我们可能需要一个更小的卷积核来识别标志和徽标等细节。

Inception 网络代替了在同一输入上运行一系列不同尺寸的卷积，并将所有滤波器连接在一起传递到下一层。不过，在执行任何操作之前，它会进行一个 1×1 的卷积作为*瓶颈*，压缩输入张量，这意味着 3×3 和 5×5 的卷积核操作的过滤器数量比如果没有 1×1 卷积存在时要少。你可以在图 3-2 中看到一个 Inception 模块的示例。

![一个 Inception 模块的图表](img/ppdl_0302.png)

###### 图 3-2。一个 Inception 模块

原始的 GoogLeNet 架构使用了九个这样的模块堆叠在一起，形成一个深度网络。尽管深度较大，但总体参数比 AlexNet 少，同时提供了一个 6.67%的前五名错误率，接近人类的表现。

## VGG

2014 年 ImageNet 的第二名是来自牛津大学的 Visual Geometry Group（VGG）网络。与 GoogLeNet 相比，VGG 是一个更简单的卷积层堆叠。在最终分类层之前，它展示了简单深度架构的强大之处（在 VGG-16 配置中获得了 8.8%的前五名错误率）。图 3-3 展示了 VGG-16 从头到尾的层。

VGG 方法的缺点是最终的全连接层使网络膨胀到一个庞大的尺寸，与 GoogLeNet 的 700 万参数相比，达到了 1.38 亿参数。尽管如此，VGG 网络在深度学习领域仍然非常受欢迎，因为它的构造更简单，训练权重早期可用。你经常会看到它在样式转移应用中使用（例如，将照片转换为梵高的画作），因为它的卷积滤波器的组合似乎捕捉到了这种信息，这种信息比更复杂的网络更容易观察。

![VGG-16 的图表](img/ppdl_0303.png)

###### 图 3-3。VGG-16

## ResNet

一年后，微软的 ResNet 架构在 ImageNet 2015 比赛中获得了 ResNet-152 变体的 4.49%和集成模型的 3.57%的前五名得分（在这一点上基本超越了人类的能力）。ResNet 带来的创新是改进了 Inception 风格的层叠层次结构方法，其中每个层叠执行通常的 CNN 操作，但还将传入的输入添加到块的输出中，如图 3-4 所示。

这种设置的优势在于每个块将原始输入传递到下一层，允许训练数据的“信号”在比 VGG 或 Inception 更深的网络中传递。（在深度网络中的权重变化的损失被称为*梯度消失*，因为在训练过程中反向传播的梯度变化趋于零。）

![一个 ResNet 块的图表](img/ppdl_0304.png)

###### 图 3-4。一个 ResNet 块

## 其他架构也是可用的！

自 2015 年以来，许多其他架构已经逐步提高了在 ImageNet 上的准确性，例如 DenseNet（ResNet 思想的延伸，允许构建 1,000 层的庞大架构），但也有很多工作致力于创建像 SqueezeNet 和 MobileNet 这样的架构，它们提供了合理的准确性，但与 VGG、ResNet 或 Inception 等架构相比，它们要小得多。

另一个重要的研究领域是让神经网络开始设计神经网络。到目前为止，最成功的尝试当然来自 Google，他们的 AutoML 系统生成了一个名为*NASNet*的架构，在 ImageNet 上的前五错误率为 3.8%，这是我在 2019 年初写这篇文章时的最新技术水平（还有另一个来自 Google 的自动生成架构称为*PNAS*）。事实上，ImageNet 比赛的组织者已经决定停止在这个领域进行进一步的比赛，因为这些架构已经超越了人类的能力水平。

这将我们带到了这本书出版时的最新技术水平，所以让我们看看我们如何可以使用这些模型而不是定义我们自己的。

# 在 PyTorch 中使用预训练模型

显然，每次想使用一个模型都要定义一个模型将是一件麻烦事，特别是一旦你远离 AlexNet，所以 PyTorch 在`torchvision`库中默认提供了许多最受欢迎的模型。对于 AlexNet，你只需要这样做：

```py
import torchvision.models as models
alexnet = models.alexnet(num_classes=2)
```

VGG、ResNet、Inception、DenseNet 和 SqueezeNet 变体的定义也是可用的。这给了你模型的定义，但你也可以进一步调用`models.alexnet(pretrained=True)`来下载 AlexNet 的预训练权重，让你可以立即用它进行分类，无需额外的训练。（但正如你将在下一章中看到的那样，你可能需要进行一些额外的训练来提高你特定数据集上的准确性。）

话虽如此，至少建立自己的模型一次是有必要的，这样你就能感受到它们如何组合在一起。这是一个很好的练习，在 PyTorch 中构建模型架构的方法，当然你也可以与提供的模型进行比较，以确保你所构建的与实际定义相匹配。但是你如何找出那个结构是什么呢？

## 检查模型的结构

如果你对其中一个模型是如何构建的感到好奇，有一个简单的方法可以让 PyTorch 帮助你。例如，这里是整个 ResNet-18 架构的一个示例，我们只需调用以下内容：

```py
print(model)

ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
  bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
track_running_stats=True)
  (relu): ReLU(inplace)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1,
  dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
      padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
       track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
      padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
       track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
       padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
       track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
      padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
       track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2),
       padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
       track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1),
       padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
       track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2),
         bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
         track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1),
       padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
       track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1),
       padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
      track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2),
       padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
      track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),
      padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
      track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2),
        bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
        track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),
      padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
      track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),
      padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
      track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2),
      padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
      track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
      padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
      track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2),
        bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
        track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
      padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
       track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),
      padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
      track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
```

在这一章中，你几乎没有看到什么新东西，除了`BatchNorm2d`。让我们看看其中一个层中的作用。

## BatchNorm

*BatchNorm*，即*批量归一化*，是一个简单的层，它的生活中只有一个任务：使用两个学习参数（意味着它将与网络的其余部分一起训练）来尝试确保通过网络的每个小批量具有以零为中心的均值和方差为 1。你可能会问为什么我们需要这样做，当我们已经通过使用第二章中的变换链对输入进行了归一化。对于较小的网络，`BatchNorm`确实不太有用，但随着它们变得更大，任何一层对另一层的影响，比如说 20 层之后，可能会很大，因为重复的乘法，你可能会得到消失或爆炸的梯度，这两者对训练过程都是致命的。`BatchNorm`层确保即使你使用像 ResNet-152 这样的模型，你网络内部的乘法也不会失控。

您可能会想：如果我们的网络中有`BatchNorm`，为什么在训练循环的转换链中还要对输入进行归一化呢？毕竟，`BatchNorm`不应该为我们做这项工作吗？答案是是的，您可以这样做！但网络将需要更长的时间来学习如何控制输入，因为它们将不得不自己发现初始转换，这将使训练时间更长。

我建议您实例化我们到目前为止讨论过的所有架构，并使用`print(model)`来查看它们使用的层以及操作发生的顺序。之后，还有另一个关键问题：*我应该使用这些架构中的哪一个？*

## 您应该使用哪个模型？

没有帮助的答案是，自然是哪个对您最有效！但让我们深入一点。首先，尽管我建议您目前尝试 NASNet 和 PNAS 架构，但我不会全力推荐它们，尽管它们在 ImageNet 上取得了令人印象深刻的结果。它们在操作中可能会消耗大量内存，并且*迁移学习*技术（您将在第四章中了解到）与人工构建的架构（包括 ResNet）相比并不那么有效。

我建议您在[Kaggle](https://www.kaggle.com)上浏览基于图像的比赛，这是一个举办数百个数据科学比赛的网站，看看获胜作品在使用什么。很可能您会看到一堆基于 ResNet 的集成模型。就我个人而言，我喜欢并使用 ResNet 架构，因为它们提供了良好的准确性，并且很容易从 ResNet-34 模型开始尝试实验，然后转向更大的 ResNet（更现实地说，使用不同 ResNet 架构的集成模型，就像微软在 2015 年 ImageNet 比赛中使用的那样），一旦我觉得有所希望。

在结束本章之前，我有一些关于下载预训练模型的最新消息。

# 模型一站式购物：PyTorch Hub

PyTorch 世界最近的一项公告提供了另一种获取模型的途径：*PyTorch Hub*。这将成为未来获取任何已发布模型的中心位置，无论是用于处理图像、文本、音频、视频还是其他任何类型的数据。要以这种方式获取模型，您可以使用`torch.hub`模块：

```py
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
```

第一个参数指向一个 GitHub 所有者和存储库（字符串中还可以包含可选的*标签/分支*标识符）；第二个是请求的模型（在本例中为`resnet50`）；最后一个指示是否下载预训练权重。您还可以使用`torch.hub.list('pytorch/vision')`来发现该存储库中可供下载的所有模型。

PyTorch Hub 是 2019 年中新推出的，所以在我写这篇文章时可用的模型数量并不多，但我预计到年底它将成为一个流行的模型分发和下载方式。本章中的所有模型都可以通过 PytorchHub 中的`pytorch/vision`存储库加载，所以可以随意使用这种加载过程，而不是`torchvision.models`。

# 结论

在这一章中，您已经快速了解了基于 CNN 的神经网络是如何工作的，包括`Dropout`、`MaxPool`和`BatchNorm`等特性。您还看了当今工业中最流行的架构。在继续下一章之前，尝试一下我们讨论过的架构，看看它们之间的比较。（不要忘记，您不需要训练它们！只需下载权重并测试模型。）

我们将通过使用这些预训练模型作为我们猫对鱼问题的自定义解决方案的起点来结束我们对计算机视觉的探讨，这将使用*迁移学习*。

# 进一步阅读

+   [AlexNet: “使用深度卷积神经网络进行 ImageNet 分类”](https://oreil.ly/CsoFv) 作者：Alex Krizhevsky 等人（2012 年）

+   [VGG: “用于大规模图像识别的非常深的卷积网络”](https://arxiv.org/abs/1409.1556) 作者：Karen Simonyan 和 Andrew Zisserman（2014 年）

+   [Inception: “使用卷积进行更深层次的研究”](https://arxiv.org/abs/1409.4842) 作者：Christian Szegedy 等人（2014 年）

+   [ResNet: “用于图像识别的深度残差学习”](https://arxiv.org/abs/1512.03385) 作者：Kaiming He 等人（2015 年）

+   [NASNet: “学习可迁移的架构以实现可扩展的图像识别”](https://arxiv.org/abs/1707.07012) 作者：Barret Zoph 等人（2017 年）

¹ 在文献中，核和滤波器往往可以互换使用。如果您有图形处理经验，核可能更熟悉，但我更喜欢滤波器。
