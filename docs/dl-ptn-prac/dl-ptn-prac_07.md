# 5.过程设计模式

本章涵盖

+   介绍卷积神经网络的过程设计模式

+   将过程设计模式的架构分解为宏观和微观组件

+   使用过程设计模式编码前 SOTA 模型

在 2017 年之前，大多数神经网络模型的实现都是用批处理脚本风格编写的。随着人工智能研究人员和经验丰富的软件工程师越来越多地参与研究和设计，我们开始看到模型编码的转向，这反映了软件工程原则的可重用性和设计模式。

使用设计模式为神经网络模型编写代码的最早版本之一是使用可重用的过程风格。设计模式意味着存在一种当前的最佳实践，用于构建和编码一个模型，该模型可以在广泛的案例中重新应用，例如图像分类、目标检测和跟踪、面部识别、图像分割、超分辨率和风格迁移。

那么，设计模式的引入是如何帮助 CNN（以及其他架构，如 NLP 中的 transformer）的进步的呢？首先，它帮助其他研究人员理解和复制一个模型的架构。将模型分解为其可重用组件或模式，为其他从业者提供了观察、理解和然后进行高效设备实验的手段。

我们可以看到，早在 AlexNet 向 VGG 的过渡时期，这种情况就已经发生了。AlexNet 的作者（[`mng.bz/1ApV`](http://mng.bz/1ApV)）没有足够的资源在单个 GPU 上运行 AlexNet 模型。他们设计了一个可以在两个 GPU 上并行运行的 CNN 架构。为了解决这个问题，他们提出了一个具有两个镜像卷积路径的设计，这个设计赢得了 2012 年 ILSVRC 图像分类竞赛。很快，其他研究人员抓住了拥有重复卷积模式的想法，他们开始研究卷积模式的影响，除了分析整体性能。2014 年，GoogLeNet（[`arxiv.org/abs/1409.4842`](https://arxiv.org/abs/1409.4842)）和 VGG（[`arxiv.org/pdf/1409.1556.pdf`](https://arxiv.org/pdf/1409.1556.pdf)）基于模型和相应的研究论文，使用了模型中重复的卷积模式；这些创新分别成为了 2014 年 ILSVRC 竞赛的冠军和亚军。

理解过程设计模式的架构对于你打算将其应用于你构建的任何模型至关重要。在本章中，我将首先向你展示如何构建这个模式，通过将其分解为其宏观架构组件，然后是微观架构组和块。一旦你看到各个部分如何单独和共同工作，你就可以开始使用构建这些部分的代码了。

为了展示过程设计模式如何使模型组件的再现更容易，我们将将其应用于几个以前 SOTA 模型：VGG、ResNet、ResNeXt、Inception、DenseNet 和 SqueezeNet。这应该让您对这些模型的工作原理有更深入的理解，以及实际再现它们的经验。这些架构的一些显著特点如下：

+   *VGG*—2014 年 ImageNet ILSVRC 挑战赛图像分类的获胜者

+   *ResNet*—2015 年 ImageNet ILSVRC 挑战赛图像分类的获胜者

+   *ResNeXt*—2016 年，作者通过引入宽卷积层提高了准确性

+   *Inception*—2014 年 ImageNet ILSVRC 挑战赛物体检测的获胜者

+   *DenseNet*—2017 年，作者引入了特征图重用

+   *SqueezeNet*—2016 年，作者引入了可配置组件的概念

我们将简要介绍基于 Idiomatic 设计模式的一个基于过程的 CNN 模型设计模式。

## 5.1 基本神经网络架构

*Idiomatic 设计模式*将模型视为由一个整体宏观架构模式组成，然后每个宏观组件依次由一个微观架构设计组成。模型宏观和微观架构的概念是在 2016 年 SqueezeNet 的研究论文中引入的([`arxiv.org/abs/1602.07360`](https://arxiv.org/abs/1602.07360))。对于 CNN，宏观架构遵循由三个宏观组件组成的惯例：主干、学习器和任务，如图 5.1 所示。

![](img/CH05_F01_Ferlitsch.png)

图 5.1 CNN 宏观架构由三个组件组成：主干、学习器和任务。

如您所见，主干组件接收输入（图像）并执行初始的粗略级别特征提取，这成为学习组件的输入。在这个例子中，主干包括一个预主干组，它执行数据预处理，以及一个主干卷积组，它执行粗略级别的特征提取。

然后，由任意数量的卷积组组成的*学习器*从提取的粗略特征中进行详细的特征提取和表示学习。学习组件的输出被称为*潜在空间*。

*任务*组件从潜在空间中输入表示学习任务（例如分类）。

虽然这本书主要关注 CNN，但这种主干、学习器和任务组件的宏观架构可以应用于其他神经网络架构，例如在自然语言处理中具有注意力机制的 Transformer 网络。

通过使用功能 API 查看 Idiomatic 设计模式的骨架模板，您可以在高层次上看到组件之间的数据流。我们将使用这个模板（在以下代码块中），并在使用 Idiomatic 设计模式的章节中在此基础上构建。该骨架由两个主要组件组成：

+   主要组件（过程）的输入/输出定义：主干、学习器和任务

+   输入（张量）流经主要组件

这是骨架模板：

```
def stem(input_shape):                            ❶
    ''' stem layers 
        Input_shape : the shape of the input tensor
    '''
    return outputs

def learner(inputs):                              ❷
    ''' leaner layers 
        inputs : the input tensors (feature maps)
    '''
    return outputs

def task(inputs, n_classes):                      ❸
    ''' classifier layers 
        inputs    : the input tensors (feature maps)
        n_classes : the number of output classes
    '''
    return outputs

inputs = Input(input_shape=(224, 224, 3))         ❹
outputs = stem(inputs)
outputs = learner(outputs)
outputs = task(x, n_classes=1000)
model = Model(inputs, outputs)                    ❺
```

❶ 构建主干组件

❷ 构建学习组件

❸ 为分类器构建任务组件

❹ 定义输入张量

❺ 组装模型

在这个例子中，`Input`类定义了模型的输入张量；对于 CNN 来说，它由图像的形状组成。元组（224，224，3）指的是一个 224 × 224 RGB（三通道）图像。`Model`类是使用 TF.Keras 功能 API 编码神经网络时的最后一步。这一步是模型的最终构建步骤（称为`compile()`方法）。`Model`类的参数是模型输入（s）张量和输出（s）张量。在我们的例子中，我们有一个输入张量和输出张量。图 5.2 描述了这些步骤。

![](img/CH05_F02_Ferlitsch.png)

图 5.2 构建 CNN 模型的步骤：定义输入，构建组件，编译成图

现在让我们更详细地看看三个宏观组件。

## 5.2 主干组件

**主干组件**是神经网络的人口点。其主要目的是执行第一层（粗粒度）特征提取，同时将特征图减少到适合学习组件的大小。主干组件输出的特征图数量和大小是通过同时平衡两个标准来设计的：

+   最大化粗粒度特征的特征提取。这里的目的是给模型提供足够的信息来学习更细粒度的特征，同时不超过模型的能力。

+   最小化下游学习组件中的参数数量。理想情况下，你希望最小化特征图的大小和训练模型所需的时间，但又不影响模型的表现。

这个初始任务由**主干卷积组**执行。现在让我们看看一些来自知名 CNN 模型（VGG、ResNet、ResNeXt 和 Inception）的主干组的变体。

### 5.2.1 VGG

VGG 架构赢得了 2014 年 ImageNet ILSVRC 图像分类竞赛，被认为是现代 CNN 之父，而 AlexNet 被认为是祖父。VGG 通过使用模式将 CNN 构建成组件和组的形式，正式化了这一概念。在 VGG 之前，CNN 被构建为 ConvNets，其用途并未超出学术上的新奇之处。

VGG 是第一个在生产中具有实际应用的。在其开发后的几年里，研究人员继续将更现代的 SOTA 架构发展与 VGG 进行比较，并使用 VGG 作为早期 SOTA 目标检测模型的分类骨干。

VGG，连同 Inception，正式化了拥有一个进行粗略级别特征提取的第一卷积组的概念，我们现在称之为*stem 组件*。随后的卷积组将进行更细级别的特征提取和学习，我们现在称之为*表征学习*，因此这个第二主要组件被称为*学习者*。

研究人员最终发现 VGG stem 的一个缺点：它在提取的粗略特征图中保留了输入大小（224 × 224），导致进入学习者的参数数量过多。参数数量的增加不仅增加了内存占用，还降低了训练和预测的性能。研究人员随后在后续的 SOTA 模型中通过在 stem 组件中添加池化来解决此问题，减少了粗略级别特征图的输出大小。这种变化减少了内存占用，同时提高了性能，而没有损失精度。

输出 64 个粗略级别特征图的惯例至今仍然存在，尽管一些现代 CNN 的 stem 可能输出 32 个特征图。

图 5.3 中所示的 VGG *stem 组件*被设计为以 224 × 224 × 3 图像作为输入，并输出 64 个特征图，每个特征图大小为 224 × 224。换句话说，VGG stem 组没有对特征图进行任何尺寸缩减。

![图片](img/CH05_F03_Ferlitsch.png)

图 5.3 VGG stem 组使用 3 × 3 滤波器进行粗略级别特征提取。

现在看看一个代码示例，用于在 Idiomatic 设计模式中编码 VGG stem 组件，该模式由一个单一的卷积层（`Conv2D`）组成。这个层使用 3 × 3 滤波器对 64 个滤波器进行粗略级别特征提取，不进行特征图尺寸的缩减。对于（224, 224, 3）图像输入（ImageNet 数据集），这个 stem 组的输出将是（224, 224, 64）：

```
def stem(inputs):
    """ Construct the Stem Convolutional Group
        inputs : the input tensor
    """
    outputs = Conv2D(64, (3, 3), strides=(1, 1), padding="same",
                     activation="relu")(inputs)
    return outputs
```

使用 Idiomatic 过程重用设计模式为 VGG 编写的完整代码可以在 GitHub 上找到（[`mng.bz/qe4w`](https://shortener.manning.com/qe4w)）。

### 5.2.2 ResNet

ResNet 架构赢得了 2015 年 ImageNet ILSVRC 竞赛的图像分类奖项，它是第一个结合了最大化粗略级别特征提取和通过特征图减少最小化参数的常规步骤的架构。当将他们的模型与 VGG 进行比较时，ResNet 的作者发现他们可以将 stem 组件中提取的特征图大小减少 94%，从而减少内存占用并提高模型性能，而不影响精度。

注意：将较新模型与之前的 SOTA 模型进行比较的过程被称为*消融研究*，这在机器学习领域是一种常见的做法。基本上，研究人员会复制之前模型的研究，然后为新模型使用相同的配置（例如，图像增强或学习率）。这使得他们能够与早期模型进行直接的苹果对苹果的比较。

ResNet 的作者还选择使用一个极大的 7 × 7 粗滤波器，覆盖了 49 个像素的区域。他们的理由是模型需要一个非常大的滤波器才能有效。缺点是在基组件中矩阵乘法或 matmul 操作的大量增加。最终，研究人员在后来的 SOTA 模型中发现 5 × 5 滤波器同样有效且更高效。在传统的 CNN 中，5 × 5 滤波器通常被两个 3 × 3 滤波器的堆叠所取代，第一个卷积是无步长的（没有池化），第二个卷积是步长的（带有特征池化）。

几年来，ResNet v1 和改进的 v2 成为了图像分类生产中实际使用的默认架构，以及在目标检测模型中的骨干。除了其改进的性能和准确性之外，公开的预训练 ResNets 版本在图像分类、目标检测和图像分割任务中广泛可用，因此这种架构成为了迁移学习的标准。即使今天，在高调的模型动物园中，如 TensorFlow Hub，预训练的 ResNet v2 仍然作为图像分类的骨干而高度流行。然而，今天更现代的预训练图像分类惯例是更小、更快、更准确的 EfficientNet。图 5.4 展示了 ResNet 基组件中的层。

![图像](img/CH05_F04_Ferlitsch.png)

图 5.4 ResNet 基组件通过步长卷积和最大池化积极减少特征图的大小。

在 ResNet 中，基组件由一个用于粗略特征提取的卷积层组成。该模型使用 7 × 7 的滤波器大小，在更宽的窗口上获取粗略特征，根据理论，这将提取更大的特征。7 × 7 的滤波器覆盖 49 个像素（相比之下，3 × 3 的滤波器覆盖 9 个像素）。使用更大的滤波器大小也显著增加了每个滤波器步骤（因为滤波器在图像上滑动）的计算量（矩阵乘法）。以每个像素为基础，3 × 3 有 9 次矩阵乘法，而 7 × 7 有 49 次。在 ResNet 之后，使用 7 × 7 来获取更大粗略级特征的传统不再被追求。

注意，VGG 和 ResNet 的基组件都输出 64 个初始特征图。这继续成为研究人员通过试错学习到的一个相当常见的惯例。

对于特征图降维，ResNet 基组件同时进行特征池化步骤（步长卷积）和下采样（最大池化）。

卷积层在滑动滤波器穿过图像时不会使用填充。因此，当滤波器到达图像边缘时，它会停止。由于边缘前的最后几个像素没有自己的滑动，输出尺寸小于输入尺寸，如图 5.5 所示。结果是输入和输出特征图的尺寸没有得到保留。例如，在步长为 1、滤波器大小为 3×3、输入特征图大小为 32×32 的卷积中，输出特征图将是 30×30。计算尺寸损失是直接的。如果滤波器大小是*N* × *N*，则尺寸损失将是*N* – 1 个像素。在 TF.Keras 中，这是通过将关键字参数`padding='valid'`指定给`Conv2D`层来实现的。

![图片](img/CH05_F05_Ferlitsch.png)

图 5.5 填充和不填充的选项导致滤波器的不同停止位置。

或者，我们可以将滤波器滑动到边缘，直到最后一行和最后一列都被覆盖。但滤波器的一部分会悬停在虚拟像素上。这样，边缘前的最后几个像素将有自己的滑动，输出特征图的尺寸得到保留。

存在几种填充虚拟像素的策略。如今最常用的惯例是在边缘使用相同的像素值填充虚拟像素，如图 5.5 所示。在 TF.Keras 中，这是通过将关键字参数`padding='same'`指定给`Conv2D`层来实现的。

ResNet 在遵循这一惯例之前就已经存在，并且用零值填充了虚拟像素；这就是为什么你在主干组中看到`ZeroPadding2D`层，其中在图像周围放置了零填充。如今，我们通常使用相同的填充来填充图像，并将特征图尺寸的减少推迟到池化或特征池化。通过反复试验，研究人员发现这种方法在保持图像边缘特征提取信息方面效果更好。

图 5.6 展示了在大小为*H* × *W* × 3（RGB 的三个通道）的图像上使用填充的卷积。使用单个滤波器，我们将输出一个大小为*H* × *W* × 1 的特征图。

![图片](img/CH05_F06_Ferlitsch.png)

图 5.6 使用单个滤波器的填充卷积产生特征提取的最小变异性。

图 5.7 展示了在大小为*H* × *W* × 3（RGB 的三个通道）的图像上使用多个滤波器*C*的卷积。在这里，我们将输出一个大小为*H* × *W* × *C*的特征图。

![图片](img/CH05_F07_Ferlitsch.png)

图 5.7 使用多个滤波器的填充卷积按比例增加了特征提取的变异性。

你是否曾见过如图 5.6 所示的单个输出特征图的主干卷积？答案是：没有。这是因为单个滤波器只能学习提取单个粗略特征。这对于图像来说是不行的！即使我们的图像是简单的平行线序列（一个特征）并且我们只想计数线条，这仍然是不行的：我们无法控制滤波器学习提取哪个特征。在这个过程中仍然存在一定程度的随机性，因此我们需要一些冗余来保证足够的滤波器能够学习提取重要特征。

你是否曾在 CNN 的某个地方输出单个特征图？答案是：是的。这将是通过 1 × 1 瓶颈卷积进行的一种激进减少。1 × 1 瓶颈卷积通常用于 CNN 中不同卷积之间的特征重用。

再次强调，这涉及到权衡。一方面，你希望结合 CNN 中某处特征提取/学习的优势与另一处的优势（特征重用）。问题是，重用整个先前的特征图，在数量和大小上，可能会在参数上造成潜在的爆炸。这种增加的内存占用和速度降低抵消了好处。ResNet 的作者选择了特征减少的量，这是在准确度、大小和性能之间最佳权衡的结果。

接下来，看看使用 Idiomatic 设计模式编码 ResNet 主干组件的示例。该代码演示了通过图 5.3 中先前展示的层进行顺序流：

+   `Conv2D`层使用 7 × 7 滤波器大小进行粗略级特征提取，并使用`strides=(2, 2)`进行特征池化。

+   `MaxPooling`层执行下采样以进一步减少特征图。

值得注意的是，ResNet 是第一个使用批归一化（`BatchNormalization`）约定的模型之一。早期的约定，现在称为 Conv-BN-RE，批归一化位于卷积和密集层之后。为了提醒你，批归一化通过将层的输出重新分配到正态分布来稳定神经网络。这允许神经网络在更深层次上运行而不会出现梯度消失或爆炸。更多详情，请参阅 Sergey Ioffe 和 Christian Szegedy 的论文“Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”（[`arxiv.org/abs/1502.03167`](https://arxiv.org/abs/1502.03167)）。

```
def stem(inputs):
    """ Construct the Stem Convolutional Group
        inputs : the input vector
    """

    outputs = ZeroPadding2D(padding=(3, 3))(inputs)                         ❶

    outputs = Conv2D(64, (7, 7), strides=(2, 2), padding='valid')(outputs)  ❷
    outputs = BatchNormalization()(outputs)
    outputs = ReLU()(outputs)

    outputs = ZeroPadding2D(padding=(1, 1))(outputs)                        ❸
    outputs = MaxPooling2D((3, 3), strides=(2, 2))(outputs)
    return outputs
```

❶ 224 × 224 的图像在第一卷积之前被零填充（黑色——无信号）以成为 230 × 230 的图像。

❷ 第一卷积层，使用大（粗略）滤波器

❸ 使用 2 × 2 步长，池化后的特征图将减少 75%。

使用 Idiomatic 程序重用设计模式为 ResNet 编写的完整代码版本可在 GitHub 上找到（[`mng.bz/7jK9`](https://shortener.manning.com/7jK9)）。

### 5.2.3 ResNeXt

在 ResNet 之后出现的模型使用了相同填充的惯例，这减少了层到单个步进卷积（特征池化）和步进最大池化（下采样），同时保持了相同的计算复杂度。Facebook AI Research 的 ResNeXt 模型 ([`arxiv.org/abs/1512.03385`](https://arxiv.org/abs/1512.03385))，以及谷歌公司的 Inception，引入了在学习者组件中使用宽残差块。如果你不知道宽和深残差块的影响，我在第六章中会解释。在这里，我只是想让你知道，卷积中的填充出现在早期的 SOTA 宽残差模型中。至于生产中的使用，ResNeXt 架构和其他宽 CNN 很少出现在内存受限的设备之外；后续在尺寸、速度和准确性方面的改进更为突出。

![](img/CH05_F08_Ferlitsch.png)

图 5.8 ResNeXt 基础组件通过组合特征和最大池化进行积极的特征图减少。

注意，由于使用了相同填充的惯例，因此没有必要使用 `ZeroPadding` 层来保持特征图大小。

以下是一个在 Idiomatic 设计模式中编码 ResNeXt 基础组件的代码示例。在这个例子中，你可以看到与 ResNet 基础组件的对比；`ZeroPadding` 层不存在，而是用 `padding='same'` 替换了 `Conv2D` 和 `MaxPooling` 层：

```
def stem(inputs):
    """ Construct the Stem Convolution Group
        inputs : input vector
    """
    outputs = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)    ❶
    outputs = BatchNormalization()(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(outputs) ❶
    return outputs
```

❶ 使用 padding='same' 而不是 VGG 中的 ZeroPadding2D

在后续的模型中，7 × 7 滤波器大小被替换为较小的 5 × 5 滤波器，它具有更低的计算复杂度。今天的常见惯例是将 5 × 5 滤波器重构为两个 3 × 3 滤波器，它们具有相同的表示能力，但计算复杂度更低。

在 GitHub ([`mng.bz/my6r`](https://shortener.manning.com/my6r)) 上有使用 Idiomatic 程序重用设计模式为 ResNeXt 编写的完整代码示例。

### 5.2.4 Xception

当前惯例是用两个 3 × 3 卷积层替换一个单独的 5 × 5 卷积层。图 5.9 中显示的 Xception ([`arxiv.org/abs/1610.02357`](https://arxiv.org/abs/1610.02357)) 基础组件是一个例子。第一个 3 × 3 卷积是步进（特征池化）的，并产生 32 个过滤器，第二个 3 × 3 卷积没有步进，将输出特征图的数量加倍到 64。然而，尽管在学术上具有新颖性，Xception 的架构并未在生产中得到采用，也没有被后续研究人员进一步发展。

![](img/CH05_F09_Ferlitsch.png)

图 5.9 Xception 基础组件

在这个例子中，对于在 Idiomatic 设计模式中编码 Xception 基础组件，你可以看到两个 3 × 3 卷积（重构的 5 × 5），第一个卷积是步进的（特征池化）。这两个卷积都紧跟着 Conv-BN-RE 形式的批量归一化：

```
def stem(inputs):
        """ Create the stem entry into the neural network
            inputs : input tensor to neural network
        """

        outputs = Conv2D(32, (3, 3), strides=(2, 2))(inputs)     ❶
        outputs = BatchNormalization()(outputs)
        outputs = ReLU()(outputs)

        outputs = Conv2D(64, (3, 3), strides=(1, 1))(outputs)    ❶
        outputs = BatchNormalization()(outputs)
        outputs = ReLU()(outputs)
        return outputs
```

❶ 将 5 × 5 卷积重构为两个 3 × 3 卷积

在 GitHub 上使用 Idiomatic 程序重用设计模式对 Xception 进行完整代码实现（[`mng.bz/5WzB`](https://shortener.manning.com/5WzB)）。

## 5.3 预前缀

2019 年，我们开始看到在基组件中添加一个*预前缀组*的兴起。预前缀的目的在于将一些或所有上游执行的数据预处理移动到图（模型）中。在预前缀组件开发之前，数据预处理发生在单独的模块中，然后在模型部署到未来示例上进行推理（预测）时必须重复执行。通常，这是在 CPU 上完成的。然而，许多数据预处理步骤可以被图操作所替代，然后在通常部署模型的 GPU 上更有效地执行。

预前缀也是即插即用的，它们可以被添加或从现有模型中移除并重用。我将在稍后介绍预前缀的技术细节。在这里，我只是想提供一个预前缀组通常执行的功能的摘要：

+   预处理

    +   将模型适应不同的输入大小

    +   正规化

+   增强处理

    +   调整大小和裁剪

    +   平移和尺度不变性

图 5.10 描述了如何向现有模型添加预前缀组。要附加预前缀，你需要创建一个新的空包装器模型，添加预前缀，然后添加现有模型。在后一个步骤中，预前缀组的输出形状必须与现有模型基组件的输入形状相匹配。

![](img/CH05_F10_Ferlitsch.png)

图 5.10 向现有模型添加预前缀，形成新的包装器模型

以下是将预前缀组添加到现有模型的典型方法示例。在这段代码中，实例化了一个空的`Sequential`包装器模型。然后添加预前缀组，接着添加现有模型。只要输出张量与模型的输入张量匹配（例如，(224, 224, 3)），这将有效。

```
from tf.keras.layers.experimental.preprocessing import Normalization

def prestem(input_shape):
    ''' pre-stem layers '''
    outputs = Normalization(input_shape=input_shape)
    return outputs

wrapper_model = Sequential()                            ❶

wrapper_model.add(prestem(input_shape=(224, 224, 3))    ❷

wrapper_model.add(model)                                ❸
```

❶ 创建一个空包装器模型

❷ 使用预前缀启动包装器模型

❸ 将现有模型添加到包装器模型

接下来我们将解释学习组件的设计，该组件将连接到基组件。

## 5.4 学习组件

*学习组件*是我们通常通过更详细的特征提取执行特征学习的地方。这个过程也被称为*表示性*或*转换性*学习（因为转换性学习依赖于任务）。学习组件由一个或多个卷积组组成，每个组由一个或多个卷积块组成。

根据共同的模型配置属性，卷积块被组装成组。在传统的 CNN 中，卷积组的常见属性是输入或输出滤波器的数量，或输入或输出特征图的大小。例如，在 ResNet 中，组的可配置属性是卷积块的数量，以及每个块的滤波器数量。

图 5.11 展示了一个可配置的卷积组。如图所示，卷积块对应于组中块数量的元参数。大多数 SOTA 架构中除了最后一个组之外的所有组具有相同数量的输出特征图，这对应于输入滤波器数量的元参数。最后一个块可能会改变组输出的特征图数量（例如，加倍），这对应于输出滤波器数量的元参数。最外层（在图像中标有*[特征]池化块*）指的是具有延迟下采样的组，这对应于池化类型的元参数。

![](img/CH05_F11_Ferlitsch.png)

图 5.11 卷积组元参数：输入/输出滤波器数量和输出特征图大小

以下代码是编码学习组件的骨架模板（和示例）。在这个例子中，组的配置属性作为字典值列表传递，每个组一个值。`learner()` 函数遍历组配置属性列表；每次迭代对应于相应组的组参数（`group_params`）。

相应地，`group()` 函数遍历组中每个块的块参数（`block_params`）。然后，`block()` 函数根据传递给它的特定于块的配置参数构建块。

如图 5.11 所示，传递给 `block()` 方法的可配置属性作为关键字参数列表将是输入滤波器数量（`in_filters`）、输出滤波器数量（`out_filters`）和卷积层数量（`n_layers`）。如果输入和输出滤波器数量相同，通常使用单个关键字参数（`n_filters`）：

```
def learner(inputs, groups):
    ''' leaner layers 
        inputs : the input tensors (feature maps)
        groups : the block parameters for each group
    '''
    outputs = inputs
    for group_parmas in groups:                        ❶
        outputs = group(outputs, **group_params)
    return outputs

def group(inputs, **blocks):
    ''' group layers
        inputs : the input tensors (feature maps)
        blocks : the block parameters for each block
    '''
    outputs = inputs
    for block_parmas in blocks:                        ❷
        outputs = block(**block_params)
    return outputs

def block(inputs, **params):
    ''' block layers
        inputs : the input tensors (feature maps)
        params : the block parameters for the block
    '''
    ...
    return outputs

outputs = learner(outputs, [ {'n_filters: 128'},       ❸
                 {'n_filters: 128'},  
                 {'n_filters: 256'} ]   
```

❶ 遍历每个组属性的字典值

❷ 遍历每个块属性的字典值

❸ 通过指定组数和每个组的滤波器数量来组装学习组件

### 5.4.1 ResNet

在 ResNet50、101 和 151 中，学习组件由四个卷积组组成。第一个组使用非步进卷积层作为第一个卷积块的投影捷径，该块从主干组件接收输入。其他三个卷积组在第一个卷积块的投影捷径中使用步进卷积层（特征池化）。图 5.12 展示了这种配置。

![](img/CH05_F12_Ferlitsch.png)

图 5.12 在 ResNet 学习器组件中，第一个组以一个非步长的投影快捷方式块开始。

现在我们将查看一个使用 ResNet50 学习器组件的骨架模板的示例应用。请注意，在`learner()`函数中，我们移除了第一个组的配置属性。在这个应用中，我们这样做是因为第一个组以一个非步长的投影快捷方式残差块开始，而所有剩余的组使用步长投影快捷方式。或者，我们也可以使用配置属性来指示第一个残差块是否步长，并消除特殊情况（编码单独的块构建）。

```
def learner(inputs, groups):
    """ Construct the Learner
        inputs: input to the learner
        groups: group parameters per group
    """
    outputs = inputs

    group_params = groups.pop(0)                                  ❶
    outputs = group(outputs, **group_params, strides=(1, 1))

    for group_params in groups:                                   ❷
        outputs = group(outputs, **group_params, strides=(2, 2))
    return outputs
```

❶ 首个残差组没有步长。

❷ 剩余的残差组使用步长卷积。

尽管 ResNets 今天仍然被用作图像分类骨干的标准模型，但如图 5.13 所示的 50 层 ResNet50 是标准。在 50 层时，模型在合理的大小和性能下提供了高精度。更大的 101 层和 151 层的 ResNet 在精度上只有轻微的增加，但大小显著增加，性能降低。

![](img/CH05_F13_Ferlitsch.png)

图 5.13 在 ResNet 组宏架构中，第一个块使用投影快捷方式，而剩余的块使用身份链接。

每个组以一个具有线性投影快捷方式的残差块开始，后面跟着一个或多个具有身份快捷方式的残差块。组中的所有残差块具有相同数量的输出滤波器。每个组依次将输出滤波器的数量加倍，具有线性投影快捷方式的残差块将输入到组中的滤波器数量加倍。

ResNets（例如，50、101、152）由四个卷积组组成；四个组的输出滤波器遵循加倍惯例，从 64 开始，然后是 128、256，最后是 512。数字惯例（50）指的是卷积层的数量，这决定了每个卷积组中的卷积块数量。

以下是一个使用 ResNet50 卷积组骨架模板的示例应用。对于`group()`函数，我们移除了第一个块的配置属性，我们知道对于 ResNet 来说这是一个投影块，然后迭代剩余的块作为身份块：

```
def group(inputs, blocks, strides=(2, 2)):
    """ Construct a Residual Group
        inputs    : input into the group
        blocks    : block parameters for each block
        strides   : whether the projection block is a strided convolution
    """
    outputs = inputs

    block_params = blocks.pop(0)                           ❶
    outputs = projection_block(outputs, strides=strides, **block_params)

    for block_params in blocks:                            ❷
        outputs = identity_block(outputs, **block_params)
    return outputs
```

❶ 残差组中的第一个块使用线性投影快捷链接。

❷ 剩余的块使用身份快捷链接。

在 GitHub 上有一个使用 Idiomatic procedure reuse 设计模式为 ResNet 编写的完整代码示例 ([`mng.bz/7jK9`](https://shortener.manning.com/7jK9))。

### 5.4.2 DenseNet

DenseNet 中的学习组件[(https://arxiv.org/abs/1608.06993)](https://arxiv.org/abs/1608.06993)由四个卷积组组成，如图 5.14 所示。除了最后一个组外，每个组都将池化延迟到组末尾，这被称为*过渡块*。最后一个卷积组没有过渡块，因为没有后续的组。特征图将由任务组件进行池化和展平，因此不需要（冗余）在组末进行池化。这种将最终池化延迟到最后一个组到任务组件的模式，至今仍是一个常见的惯例。

![图片](img/CH05_F14_Ferlitsch.png)

图 5.14 DenseNet 学习组件由四个具有延迟池化的卷积组组成。

以下是一个使用骨架模板编码 DenseNet 学习组件的示例实现。请注意，我们在遍历组之前移除最后一个组配置属性。我们将最后一个组视为一个特殊情况，因为该组不以过渡块结束。或者，我们也可以使用配置参数来指示一个组是否包含过渡块，从而消除特殊情况（即编写单独的块构造）。参数`reduction`指定了延迟池化期间特征图大小减少的量：

```
def learner(inputs, groups, reduction):
    """ Construct the Learner
        inputs    : input to the learner
        groups    : set of number of blocks per group
        reduction : the amount to reduce (compress) feature maps by
    """
    outputs = inputs

    last = groups.pop()                                 ❶

    for group_params in groups:                         ❷
        outputs = group(outputs, reduction, **group_params)

    outputs = group(outputs, last, reduction=None)      ❸
    return outputs
```

❶ 移除最后一个密集组参数并保存以供结尾使用

❷ 使用中间过渡块构建除最后一个密集组之外的所有密集组

❸ 在没有过渡块的最后一个密集组中添加

让我们看看 DenseNet 中的一个卷积组（图 5.15）。它仅由两种类型的卷积块组成。第一个块是用于特征学习的 DenseNet 块，最后一个块是一个过渡块，用于在下一个组之前减小特征图的大小，这被称为*压缩因子*。

![图片](img/CH05_F15_Ferlitsch.png)

图 5.15 DenseNet 组由一系列密集块和一个用于在输出特征图中进行降维的最终过渡块组成。

DenseNet 块本质上是一个残差块，除了在输出处不添加（矩阵加法操作）恒等链接，而是进行连接。在 ResNet 中，先前输入的信息只向前传递一个块。使用连接，特征图的信息累积，每个块将所有累积的信息向前传递给所有后续块。

这种特征图的连接会导致随着层数的加深，特征图的大小和相应的参数持续增长。为了控制（减少）增长，每个卷积块末尾的过渡块压缩（减小）了连接的特征图的尺寸。否则，如果没有缩减，随着层数的加深，需要学习的参数数量将显著增加，导致训练时间延长，而准确率没有提高。

以下是一个编码 DenseNet 卷积组的示例实现：

```
def group(inputs, reduction=None, **blocks):
    """ Construct a Dense Group
        inputs    : input tensor to the group
        reduction : amount to reduce feature maps by
        blocks    : parameters for each dense block in the group
    """
    outputs = inputs 

    for block_params in blocks:                            ❶
        outputs = residual_block(outputs, **block_params)

    if reduction is not None:                              ❷
        outputs = trans_block(outputs, reduction)
    return outputs
```

❶ 构建一组密集连接的残差块

❷ 构建中间过渡块

在 GitHub 上使用 Idiomatic 程序重用设计模式对 DenseNet 进行完整代码实现的示例是 ([`mng.bz/6N0o`](https://shortener.manning.com/6N0o))。接下来，我们将解释任务组件的设计，学习组件将连接到该组件。

## 5.5 任务组件

*任务组件*是我们通常执行任务学习的地方。在用于图像分类的大规模传统 CNN 中，这个组件通常由两层组成：

+   *瓶颈层*——将最终特征图的维度缩减到潜在空间

+   *分类层*——执行模型正在学习的任务

学习组件的输出是特征图的最终减小尺寸（例如，4 × 4 像素）。瓶颈层执行最终特征图的维度缩减，然后输入到分类层进行分类。

在本节的剩余部分，我们将以图像分类器为例描述任务组件；我们将其称为 *分类组件*。

### 5.5.1 ResNet

对于 ResNet50，特征图的数目是 2048。分类组件的第一个层既将特征图展平成 1D 向量，又使用 `GlobalAveragePooling2D` 等方法减小尺寸。这个展平/缩减层也被称为瓶颈层，如前所述。瓶颈层之后是一个 `Dense` 层，用于分类。

图 5.16 描述了 ResNet50 分类器。分类组件的输入来自学习组件的最终特征图（潜在空间），然后通过 `GlobalAveragePooling2D`，将每个特征图的尺寸减小到单个像素，并将其展平成一个 1D 向量（瓶颈）。从这个瓶颈层输出的内容通过 `Dense` 层，其中节点的数量对应于类别的数量。输出是所有类别的概率分布，通过 softmax 激活函数压缩，使其总和达到 100%。

![图片](img/CH05_F16_Ferlitsch.png)

图 5.16 ResNet 分类组

以下是将此方法编码为分类组件的示例，包括用于展平和维度缩减的 `GlobalAveragePooling2D`，然后是用于分类的 `Dense` 层：

```
def classifier(inputs, n_classes):
    """ The output classifier
        inputs    : input tensor to the classifier
        n_classes : number of output classes
    """
    outputs = GlobalAveragePooling2D()(inputs)                   ❶

    outputs = Dense(n_classes, activation='softmax')(outputs)    ❷
    return outputs
```

❶ 使用全局平均池化将特征图（潜在空间）减少并展平成一个 1D 特征向量（瓶颈层）

❷ 用于输入最终分类的完全连接的 Dense 层

使用 Idiomatic 过程重用设计模式为 ResNet 提供的完整代码版本可在 GitHub 上找到([`mng.bz/7jK9`](https://shortener.manning.com/7jK9))。

### 5.5.2 多层输出

在早期部署的机器学习生产系统中，模型被视为独立的算法，我们只对最终输出（预测）感兴趣。今天，我们构建的不是模型，而是由模型混合或组合而成的应用程序。因此，我们不再将任务组件视为单个输出。

相反，我们认为它有四个输出，这取决于模型如何连接到应用程序中的其他模型。这些输出如下：

+   特征提取

    +   高维度（编码）

    +   低维度（嵌入）—特征向量

+   预测

    +   预测预激活（概率）—软目标

    +   激活后（输出）—硬目标

后续章节将介绍这些输出的目的（第九章关于自编码器，第十一章关于迁移学习，第十四章关于训练管道中的预训练任务），你将看到分类器中的每一层都有两个并行输出。在图 5.17 中描述的传统分类器的多输出中，你可以看到任务组件的输入也是模型的独立输出，被称为*编码*。编码随后通过全局平均池化进行降维，进一步减小学习组件提取的特征的大小。全局平均池化的输出也是模型的独立输出，被称为*嵌入*。

![图片](img/CH05_F17_Ferlitsch.png)

图 5.17 多输出分类器组具有四个输出—两个用于特征提取共享，两个用于概率分布

嵌入随后传递到一个预激活的密集层（在 softmax 激活之前）。预激活层的输出也是模型的独立输出，被称为*预激活概率分布*。这个概率分布随后通过 softmax 得到激活后的概率分布，成为模型的第四个独立输出。所有这些输出都可以被下游任务使用。

让我们描述一个简单的现实世界示例，即使用多输出任务组件：从车辆照片中估计维修成本。我们希望对两个类别进行估计：轻微损坏（如凹痕和划痕）的成本，以及重大损坏（如碰撞损坏）的成本。我们可能会尝试在一个任务组件中完成这项工作，该组件作为回归器输出一个实值（美元值），但我们在训练期间实际上是在过度加载任务组件，因为它在学习很小的值（轻微损坏）和大的值（重大损坏）。在训练期间，值的广泛分布可能会阻止模型收敛。

这种方法是将这个问题解决为两个独立的任务组件：一个用于轻微损坏，一个用于重大损坏。轻微损坏的任务组件将只学习很小的值，而重大损坏的任务组件将只学习大的值——因此，两个任务组件应该在训练过程中收敛。

接下来，我们考虑与两个任务共享哪个输出级别。对于轻微损坏，我们关注的是微小的物体。虽然我们没有涵盖物体检测，但历史上在小型物体上进行物体分类的问题在于，在池化后的裁剪特征图中包含的空间信息太少。解决方案是从更早的卷积层中的特征图进行物体分类；这样，特征图就会足够大，当裁剪出微小的物体时，仍然保留足够的空间信息以进行物体分类。

在我们的例子中存在一个类似的问题。对于轻微的损坏，物体（每个凹痕）将会非常小，我们需要更大的特征图来检测它们。因此，为了这个目的，我们在平均和池化之前将高维编码连接到执行轻微损坏估计的任务。另一方面，重大的碰撞损坏不需要很多细节。例如，如果保险杠有凹痕，无论凹痕的大小或位置如何，都必须更换。因此，为了这个目的，我们在平均和池化之后将低维嵌入连接到执行重大损坏估计的任务。图 5.18 展示了这个例子。

![图片](img/CH05_F18_Ferlitsch.png)

图 5.18 展示了使用共享模型顶部的多输出从多任务组件估计车辆维修成本

以下是将多输出编码到分类组件的示例实现。特征提取和预测输出是通过捕获每一层的张量输入来实现的。在分类器的末尾，我们将返回单个输出替换为返回所有四个输出的元组：

```
def classifier(inputs, n_classes):
    """ The output classifier
        inputs    : input tensor to the classifier
        n_classes : number of output classes
    """
    encoding = inputs                                       ❶

    embeddings = GlobalAveragePooling2D()(inputs)           ❷

    probabilities = Dense(n_classes)(embeddings)            ❸

    outputs = Activation('softmax')(outputs)                ❹

    return encoding, embeddings, probabilities, outputs     ❺
```

❶ 高维特征提取（编码）

❷ 低维特征提取（嵌入）

❸ 预激活概率（软标签）

❹ 后激活概率（硬标签）

❺ 返回所有四个输出的元组

### 5.5.3 SqueezeNet

在紧凑模型中，尤其是对于移动设备，`GlobalAveraging2D`后面跟着一个`Dense`层被使用 softmax 激活的`Conv2D`所取代。`Conv2D`中的滤波器数量设置为类别的数量，然后是`GlobalAveraging2D`以展平到类别的数量。"SqueezeNet"论文由 Forrest Iandola 等人撰写([`arxiv.org/pdf/1602.07360.pdf`](https://arxiv.org/pdf/1602.07360.pdf))，解释了用卷积层替换密集层的理由："注意 SqueezeNet 中没有全连接层；这个设计选择受到了 NiN（Lin et al., 2013）架构的启发。"

图 5.19 是一个使用此方法对分类组件进行编码的 SqueezeNet 示例。SqueezeNet 于 2016 年由 DeepScale、加州大学伯克利分校和斯坦福大学为移动设备开发，当时是 SOTA。

![](img/CH05_F19_Ferlitsch.png)

图 5.19 SqueezeNet 分类组件

您可以看到，它使用的是 1 × 1 卷积而不是密集层，其中滤波器的数量对应于类别的数量（*C*）。这样，1 × 1 卷积正在学习类别的概率分布，而不是输入特征图的投影。然后得到的（*C*）个特征图被每个减少到一个单一的真实值用于概率分布，并展平成一个 1D 输出向量。例如，如果 1 × 1 卷积输出的每个特征图大小为 3 × 3（9 像素），则选择值最高的像素作为对应类别的概率。然后 1D 向量通过 softmax 激活函数进行压缩，使得所有概率之和为 1。

让我们将其与我们之前在大规模 SOTA 模型中讨论的全局平均池化和密集层方法进行对比。假设最终特征图的大小为 3 × 3（9 像素）。然后我们将 9 个像素平均到一个值，并根据每个特征图的单个平均值进行概率分布。在 SqueezeNet 使用的方法中，执行概率分布的卷积层看到的是 9 像素的特征图（而不是平均的单个像素），并且有更多的像素来学习概率分布。这可能是 SqueezeNet 的作者为了补偿较小模型底部的较少特征提取/学习而做出的选择。

以下是对 SqueezeNet 分类组件进行编码的示例。在这个例子中，`Conv2D`的滤波器数量是类别的数量（`n_classes`），然后是`GlobalAveragePooling2D`。由于这个层是一个静态层（未学习），它没有激活参数，因此我们必须明确地跟随一个 softmax 激活层：

```
def classifier(inputs, n_classes):
    ''' Construct the Classifier
        inputs   : input tensor to the classifier
        n_classes: number of output classes
    '''
    encoding = Conv2D(n_classes, (1, 1), strides=1,               ❶
                     activation='relu', padding='same')(inputs)

    embedding = GlobalAveragePooling2D()(outputs)                 ❷
    outputs = Activation('softmax')(outputs)                      ❸
    return outputs
```

❶ 将滤波器的数量设置为输出类别的数量

❷ 将每个特征图（类别）减少到一个单一值（软标签）

❸ 使用 softmax 将所有类别概率压缩，使其总和达到 100%

使用 Idiomatic 过程重用设计模式对 SqueezeNet 的完整代码实现位于 GitHub 上([`mng.bz/XYmv`](https://shortener.manning.com/XYmv))。

## 5.6 超越计算机视觉：NLP

如第一章所述，我在计算机视觉的背景下解释的设计模式在自然语言处理和结构化数据中也有类似的原则和模式。为了了解过程设计模式如何应用于 NLP，让我们看看自然语言理解（NLU）这类 NLP 的一个例子。

### 5.6.1 自然语言理解

让我们从查看图 5.20 中 NLU 的一般模型架构开始。在 NLU 中，模型学习理解文本，并基于这种理解执行任务。任务的例子包括对文本进行分类、情感分析和实体提取。

![图片](img/CH05_F20_Ferlitsch.png)

图 5.20 与所有深度学习模型一样，NLU 模型由词干、学习者和任务组件组成。不同之处在于每个组件内部。

我们可能会根据类型对医疗文档进行分类；例如，识别每个文档是处方、医生笔记、索赔提交或其他文档。对于情感分析，任务可能是确定评论是正面还是负面（二分类）或从负面到正面的排名（多分类）。对于实体提取，我们的任务可能是从实验室结果和医生/护士笔记中提取健康指标。

一个 NLU 模型被分解成所有深度学习模型都包含的相同组件：词干、学习者和任务。不同之处在于每个组件中发生的事情。

在一个 NLU 模型中，词干由一个编码器组成。其目的是将文本的字符串表示转换为基于数字的向量，称为*嵌入*。这个嵌入的维度比字符串输入更高，并包含关于单词、字符或句子的更丰富的上下文信息。词干编码器实际上是一个已经预训练的另一个模型。将词干编码器想象成一个字典。对于每个单词，它输出所有可能的含义，从低维度到高维度。嵌入的一个常见例子是*N*维度的向量，其中每个元素代表另一个单词，其值表示这个单词与其他单词的相关程度。

接下来，嵌入被传递到学习器组件。在一个 NLU 模型中，学习器由一个或多个编码器组组成，这些组又由一个或多个编码器块组成。每个这些块都基于一个设计模式，例如在转换器模型中的注意力块，而块和组的组装基于编码器模式的设计原则。

你可能已经注意到，茎和学习者都指的是编码器。*它们在每个组件中不是同一种类型的编码器。* 两个不同的事物使用相同的名称可能会有些令人困惑，所以我将进行澄清。当我们谈论生成嵌入的编码器时，我们将称之为*茎编码器*；否则，我们指的是学习者中的编码器。

学习者中编码器的目的是将嵌入转换为文本意义的低维表示，这被称为*中间表示*。这与在 CNN 中学习图像的基本特征相似。

任务组件与计算机视觉的对应组件非常相似。中间表示被展平成一个一维向量并进行了池化。对于分类和语义分析，池化表示被传递到一个 softmax 密集层，以预测跨类或语义排名的概率分布。

至于实体提取，任务组件与对象检测模型的任务组件相当；你正在学习两个任务：对提取的实体进行分类，以及在提取实体的文本中微调位置边界。

### 5.6.2 Transformer 架构

现在我们来看一下现代（NLU）模型的一个方面，它与计算机视觉中的 SOTA 相当。如第一章所述，NLU 的一个重大变化是在 2017 年谷歌大脑引入 Transformer 模型架构以及相应的论文“Attention is All You Need”由 Ashishh Vaswani 等人发表([`arxiv.org/abs/1706.03762`](https://arxiv.org/abs/1706.03762))。Transformer 架构解决了 NLU 中的一个难题：如何处理本质上类似于时间序列的文本序列——即，意义依赖于单词的序列顺序。在 Transformer 架构之前，NLU 模型被实现为循环神经网络（RNNs），这些网络会保留文本的序列顺序并学习单词的重要性（长记忆）或非重要性（短记忆）。

Transformer 模型所做的是引入了一种新的机制，称为*注意力*，它将 NLU 模型从时间序列转换为空间模型。我们不再将单词、字符或句子视为序列，而是将一组单词作为一个块来表示，就像图像一样。模型学习提取关键上下文——特征。注意力机制在残差网络中的身份链接作用相似。它为更重要的上下文添加了注意力——权重。

图 5.21 展示了在转换器架构中的一个注意力块。该块输入是一组来自前一个块的上下文图，类似于特征图。注意力机制为对上下文理解更为重要的上下文块部分添加权重（表示在此处需要关注）。然后，注意力上下文图被传递到一个前馈层，该层输出下一组上下文图。

![图片](img/CH05_F21_Ferlitsch.png)

图 5.21 一个注意力块为对文本理解更为重要的上下文部分添加权重。

在下一章中，我们将介绍宽卷积神经网络，这是一种关注较宽层而不是较深层的架构模式。

## 摘要

+   使用设计模式来设计和编码卷积神经网络可以使模型更易于理解，节省时间，确保模型代表了最佳 SOTA 实践，并且易于他人复现。

+   程序设计模式使用了软件工程中的重用原则，这是软件工程师广泛实践的原则。

+   宏架构由主干、学习者和任务组件组成，这些组件定义了模型中的流程以及在哪里/进行何种类型的学习。

+   微架构由定义模型如何执行学习的组和块设计模式组成。

+   预处理组的目的在于扩展现有的（预训练）模型以用于上游数据预处理、图像增强以及对其他部署环境的适应。将预处理器作为即插即用模块实现，为机器学习操作提供了部署模型而不需要附带上游代码的能力。

+   任务组件的目的是从潜在空间中学习一个特定于模型的任务，编码在特征提取和表示学习过程中的学习。

+   多层输出的目的是以最有效的方式扩展模型之间的互连性，同时保持性能目标。

+   在转换器中的注意力机制提供了以类似于计算机视觉的方式按顺序学习关键特征的方法，而不需要循环网络。
