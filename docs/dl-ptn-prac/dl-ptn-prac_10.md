# 8 个移动卷积神经网络

本章涵盖了

+   理解移动卷积网络的设计原则和独特要求

+   检查 MobileNet v1 和 v2、SqueezeNet 和 ShuffleNet 的设计模式

+   使用过程设计模式对这些模型的编码示例

+   通过量化模型并在 TensorFlow Lite（TF Lite）中执行它们来使模型更加紧凑

您现在已经学习了几个大型模型的无内存约束关键设计模式。现在让我们转向设计模式，例如来自 Facebook 的流行应用 FaceApp，这些模式针对内存受限设备，如手机和物联网设备进行了优化。

与它们的 PC 或云等价物相比，紧凑模型面临一个特殊挑战：它们需要在显著更少的内存中运行，因此不能从使用过容量以实现高精度中受益。为了适应这些受限的内存大小，模型在推理或预测时需要显著减少参数。紧凑模型的架构依赖于精度和延迟之间的权衡。模型占用的设备内存越多，精度越高，但响应时间延迟越长。

在早期的 SOTA 移动卷积模型中，研究人员找到了通过方法来解决这种权衡，这些方法在大幅减少参数和计算复杂性的同时，保持了最小程度的精度损失。这些方法依赖于对卷积的进一步重构，例如深度可分离卷积（MobileNet）和点卷积组卷积（ShuffleNet）。这些重构技术提供了增加容量以提高精度的手段，否则这些更极端的重构方法会导致精度损失。

本章介绍了两种重构方法，这些方法用于两种不同的模型：MobileNet 和 SqueezeNet。我们还将探讨第三种模型 ShuffleNet 中针对内存受限设备的另一种新颖方法。我们将在本章结束时探讨其他策略，以进一步减少内存占用，例如参数压缩和量化，以使模型更加紧凑。

在我们开始研究这三个模型的细节之前，让我简要比较一下它们处理有限内存的方法。MobileNet 的研究人员探索了通过调整模型以适应各种内存大小和延迟要求，以及这种调整对精度的影响。

SqueezeNet 研究人员提出了一种称为“fire 模块”的块模式，该模式在模型大小减少高达 90%后仍能保持精度。fire 模块使用深度压缩。这种压缩神经网络大小的方法在 Song Han 等人撰写的《深度压缩》一文中被介绍，该文于 2015 年国际学习表示会议（ICLR）上提出（[`arxiv.org/abs/1510.00149`](https://arxiv.org/abs/1510.00149)）。

同时，ShuffleNet 研究人员专注于为部署在极低功耗计算设备上的模型（例如，10 到 150 MFLOPs）增加表示能力。他们提出了两种方法：在一个高度分解的组内进行通道洗牌，以及点卷积。

现在，我们可以深入了解每个细节。

## 8.1 MobileNet v1

*MobileNet v1*是谷歌在 2017 年推出的一种架构，用于生成更小的网络，可以适应移动和物联网设备，同时保持与较大网络近似的准确性。在 Andrew G. Howard 等人撰写的“MobileNets”一文中（[`arxiv.org/abs/1704.04861`](https://arxiv.org/abs/1704.04861)），MobileNet v1 架构用深度可分离卷积替换了常规卷积，以进一步降低计算复杂度。（如您所记得，我们在第七章讨论 Xception 模型时，已经介绍了将常规卷积重构为深度可分离卷积的理论。）让我们看看 MobileNet 如何将这种方法应用于紧凑的模型。

### 8.1.1 架构

MobileNet v1 架构结合了几个针对受限内存设备的设计原则：

+   茎卷积组引入了一个额外的参数，称为**分辨率乘数**，用于更激进地减少输入到学习组件的特征图**大小**。（这在图 8.1 中标记为**A**。）

+   同样，学习组件为学习组件内部的特征图**数量**增加了一个**宽度乘数**参数，以实现更激进的减少。

+   该模型使用深度卷积（如 Xception 中所示）来降低计算复杂度，同时保持表示等价性（C）。

+   分类组件使用卷积层代替密集层进行最终分类（D）。

你可以在图 8.1 的宏架构中看到这些创新的应用，字母 A、B、C 和 D 标记了相应的特征。

![](img/CH08_F01_Ferlitsch.png)

图 8.1 MobileNet v1 宏架构在茎和学习者（A 和 B）中使用了元参数，在学习者中使用了深度卷积（C），在分类器中使用了卷积层而不是密集层（D）。

与我们之前讨论的模型不同，MobileNets 是根据其输入分辨率进行分类的。例如，MobileNet-224 的输入为（224, 224, 3）。卷积组遵循从上一个组加倍滤波器数量的惯例。

让我们先看看两个新的超参数，宽度乘数和分辨率乘数，看看它们如何以及在哪里帮助细化网络。然后我们将逐步介绍茎、学习者和分类组件。

### 8.1.2 宽度乘数

引入的第一个超参数是**宽度乘数**α（alpha），它在每一层均匀地细化网络。让我们快速看一下细化网络的优势和劣势。

我们知道，通过减薄，我们正在减少层之间的参数数量，并且指数级减少矩阵乘法操作的次数。例如，使用密集层，如果在减薄之前，一个密集层的输出和相应的输入各有 100 个参数，那么我们将有 10,000 次矩阵乘法（matmul）操作。换句话说，两个完全连接的 100 节点密集层，每通过一个 1D 向量就会进行 100 x 100 次矩阵乘法操作。

现在我们将其减薄一半。即 50 个输出参数和 50 个输入参数。现在我们已经将矩阵乘法操作的次数减少到 2500 次。结果是内存大小减少 50%，计算（延迟）减少 75%。缺点是我们进一步减少了过容量以提高准确性，并将需要探索其他策略来补偿这一点。

在每一层，输入通道的数量是 *αM*，输出通道的数量是 *αN*，其中 *M* 和 *N* 是未减薄的 MobileNet 的通道数（特征图）。现在让我们看看如何通过减薄网络层来计算参数的减少。*α*（alpha）的值从 0 到 1，并且通过 *α*²（参数数量）减少 MobileNet 的计算复杂度。*α* < 1 的值被称为*减薄 MobileNet*。通常，这些值是 0.25（6% 的未减薄）、0.50（25%）和 0.75（56%）。让我们继续进行计算。如果 *α* 因子是 0.25，那么得到的复杂度是 0.25 × 0.25，计算结果是 0.0625。

在论文中报告的测试结果中，未减薄的 MobileNet-224 在 ImageNet 上有 70.6% 的准确性，参数数量为 420 万，矩阵乘加操作为 5.69 亿，而 0.25（宽度乘数）MobileNet-224 有 50.6% 的准确性，参数数量为 50 万，矩阵乘加操作为 4100 万。这些结果表明，通过激进减薄导致的过容量损失并没有被模型设计有效地抵消。因此，研究人员转向减少分辨率，结果证明这更有利于保持准确性。

### 8.1.3 分辨率乘数

第二个引入的超参数是*分辨率乘数* *ρ*（rho），它减少了输入形状以及每个层的特征图大小。

当我们在不改变主干组件的情况下降低输入分辨率时，进入学习组件的特征图大小相应减少。例如，如果输入图像的高度和宽度减少一半，输入像素的数量将减少 75%。如果我们保持相同的粗略级滤波器和滤波器数量，输出的特征图将减少 75%。由于特征图减少，这将导致每个卷积（模型大小）和矩阵乘法操作（延迟）的数量减少。请注意，这与宽度细化不同，宽度细化会减少特征图的数量，同时保持其大小。

缺点是，如果我们过于激进地减少，当我们到达瓶颈时特征图的大小可能变为 1 × 1 像素，本质上失去了空间关系。我们可以通过减少中间层的数量来补偿这一点，使特征图大于 1 × 1，但这样我们会为了精度而移除更多的冗余。

在论文中报告的测试结果中，一个 0.25（分辨率乘数）的 MobileNet-224 在 4.2 百万个参数和 1.86 亿次矩阵乘加操作下达到了 64.4%的准确率。鉴于*ρ*（rho）的值在 0 到 1 之间，并且将 MobileNet 的计算复杂度降低到*ρ*²。如果*ρ*因子为 0.25，则结果复杂度为 0.25 × 0.25，计算结果为 0.0625。

以下是一个 MobileNet-224 的骨架模板。请注意，使用参数`alpha`和`rho`作为宽度和分辨率乘数：

```
def stem(inputs, alpha):                             ❶
    """ Construct the stem group
        inputs : input tensor
        alpha  : with multiplier
    """
                                                     ❷
    return outputs

def learner(inputs, alpha):                          ❶
    """ Construct the learner group
        inputs : input to the learner
        alpha  : with multiplier
    """
                                                     ❷
    return outputs

def classifier(inputs, alpha, dropout, n_classes):   ❶
    """ Construct the classifier group
        inputs : input to the classifier
        alpha  : with multiplier
        Dropout: percent of dropout
        n_classes: number of output classes
    """
                                                     ❷
    return outputs

inputs = Input((224*rho, 224*rho, 3))                ❸
outputs = stem(inputs, alpha)
outputs = learner(outputs, alpha)
outputs = classifier(outputs, alpha, dropout, n_classes)
model = Model(inputs, outputs)
```

❶ 模型中所有层使用的宽度乘数

❷ 为了简洁性移除的代码

❸ 仅在输入张量上使用的分辨率乘数

### 8.1.4 主干

主干组件由一个步进的 3 × 3 卷积（用于特征池化）和一个 64 个滤波器的单个深度可分离块组成。步进卷积和深度可分离块中的滤波器数量进一步通过超参数*α*（alpha）减少。通过超参数*ρ*（rho）减少输入大小不是在模型中完成，而是在输入预处理函数的上游完成。

让我们讨论一下这与当时大型模型的传统主干有何不同。通常，第一个卷积层会从粗略的 7 × 7、5 × 5 或重构的两个 3 × 3 卷积层（64 个滤波器）开始。粗略卷积会进行步进以减少特征图的大小，然后跟随一个最大池化层以进一步减少特征图的大小。

在 MobileNet v1 主干中，继续使用 64 个滤波器和两个 3 × 3 卷积层的传统，但有三个显著的变化：

+   第一个卷积输出的特征图数量是第二个卷积的一半（32）。这起到瓶颈的作用，在双 3 × 3 堆叠中减少计算复杂度。

+   第二个卷积被替换为深度可分离卷积，进一步降低了茎中的计算复杂度。

+   没有最大池化，只有第一个步长卷积导致一个特征图大小的减少。

这里的权衡是保持特征图的大小更大——是*H* × *W*的两倍。这抵消了在第一级粗略特征提取中激进减少计算复杂度所带来的表示损失。

图 8.2 展示了茎组件，它由两个 3 × 3 卷积的堆叠组成。第一个是一个正常的卷积，它执行特征池化（步长）。第二个是一个深度卷积，它保持特征图的大小（非步长）。步长 3 × 3 卷积没有使用填充。为了保持特征图减少 75%（0.5*H* × 0.5*W*），在卷积之前对输入添加了零填充。注意在输入大小上使用元参数*ρ*进行分辨率降低，以及在 3 × 3 卷积的双重堆叠上使用*α*进行网络细化。

![](img/CH08_F02_Ferlitsch.png)

图 8.2 展示了 MobileNet 茎组在 3 × 3 卷积堆叠中细化网络。

以下是一个茎组件的实现示例。正如您所看到的，卷积层使用了后激活批量归一化（Conv-BN-RE），因此模型没有使用预激活批量归一化的好处，这被发现可以将准确率从 0.5 提升到 2%：

```
def stem(inputs, alpha):
    """ Construct the stem group
        inputs : input tensor
        alpha  : with multiplier
    """
     x = ZeroPadding2D(padding=((0, 1), (0, 1)))(inputs)                 ❶
     x = Conv2D(32 * alpha, (3, 3), strides=(2, 2), padding='valid')(x)
     x = BatchNormalization()(x)
     x = ReLU(6.0)(x)

     x = depthwise_block(x, 64, alpha, (1, 1))                           ❷
     return x
```

❶ 输入特征图零填充的卷积块

❷ 深度可分离卷积块

注意，在这个例子中，`ReLU`有一个可选参数，其值为 6.0。这是`ReLU`的`max_value`参数，默认为`None`。它的目的是剪辑任何高于`max_value`的值。因此，在前面的例子中，所有输出都将位于 0 到 6.0 的范围内。在移动网络中，如果权重后来被量化，通常会将`ReLU`的输出进行剪辑。

在这个上下文中，“量化”是指使用较低位表示的计算；我将在第 8.5.1 节中解释这个过程的细节。研究发现，当`ReLU`的输出有一个约束范围时，量化模型可以保持更好的准确率。一般做法是将其设置为 6.0。

让我们简要讨论一下选择 6 这个值的理由。这个概念是在 Alex Krizhevsky 2010 年的论文“CIFAR-10 上的卷积深度信念网络”中提出的（[www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf](https://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf)）。Krizhevsky 将其作为解决深层网络梯度爆炸问题的解决方案。

当激活的输出变得非常大时，它可能会主导周围激活的输出。结果，该网络区域会表现出对称性，这意味着它会减少，就像只有一个节点一样。通过实验，Krizhevsky 发现 6 这个值是最好的。

记住，这在我们意识到批归一化的好处之前。批归一化会在每个连续的深度处压缩激活，因此不再需要截断。

在量化引入时截断 ReLU 返回值的概念。简而言之，当权重被量化时，我们正在减少表示值的位数。如果我们将权重映射到，比如说，一个 8 位整数范围，我们必须根据实际输出值的分布将整个输出范围“分桶”到 256 个桶中。范围越长，浮点值映射到桶中的拉伸就越薄，使得每个桶的特征就越不显著。

这里的理论是，那些 98%、99%和 99.5%置信度的值本质上是一样的，而较低值则更加独特——也就是说，输出有 70%的置信度。但是，通过截断，我们将所有高于 6 的值视为本质上 100%，并且仅对 0 到 6 之间的分布进行分桶，这些值对于推理更有意义。

### 8.1.5 学习者

MobileNet-224 中的学习组件由四个组组成，每个组包含两个或更多的卷积块。每个组将前一个组的过滤器数量翻倍，并且每个组中的第一个块使用步长卷积（特征池化）将特征图大小减少 75%。

构建 MobileNet 组遵循与大型卷积网络组相同的原理。两者通常具有以下特点：

1.  每组过滤器数量的进展，例如将过滤器数量翻倍

1.  通过使用步长卷积或延迟最大池化来减少输出的特征图大小

您可以在图 8.3 中看到，MobileNet 组在第一个块中使用步长卷积来减少特征图（原则 2）。尽管图中没有显示，但学习器中的每个组从 128 开始将过滤器数量翻倍（原则 1）。

![](img/CH08_F03_Ferlitsch.png)

图 8.3 在 MobileNet v1 的学习组件中，每个组是一系列深度卷积块。

图 8.4 放大了学习组中的深度卷积块。在 v1 中，模型的作者使用了卷积块设计而不是残差块设计；没有恒等连接。每个块本质上是一个单深度可分离卷积，由两个独立的卷积层构建。第一层是一个 3×3 的深度卷积，后面跟着一个 1×1 的点卷积。当结合时，这些形成深度可分离卷积。过滤器的数量，即对应于特征图的数量，可以通过元参数*α*进一步减少以进行网络细化。

![](img/CH08_F04_Ferlitsch.png)

图 8.4 MobileNet v1 卷积块

接下来是一个深度可分离卷积块的示例实现。第一步是计算在应用宽度乘数 `alpha` 后网络变薄的 `filters` 数量。对于组中的第一个块，使用步长卷积 (`strides=(2, 2)`) 对特征图大小进行减少（特征池化）。这对应于之前提到的卷积组设计原则 2，其中组中的第一个块通常对输入特征图的大小进行维度降低。

```
def depthwise_block(x, n_filters, alpha, strides):
    """ Construct a Depthwise Separable Convolution block
        x         : input to the block
        n_filters : number of filters
        alpha     : width multiplier
        strides   : strides
    """
    filters = int(n_filters * alpha)                                ❶

    if strides == (2, 2):                                           ❷
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        padding = 'valid'
    else:
        padding = 'same'

    x = DepthwiseConv2D((3, 3), strides, padding=padding)(x)        ❸
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)  ❹
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    return x
```

❶ 将宽度滤波器应用于特征图数量

❷ 在进行步长卷积时添加零填充，以匹配滤波器的数量

❸ 深度卷积

❹ 点卷积

### 8.1.6 分类器

分类组件与大型模型的传统分类器不同，它在分类步骤中使用卷积层代替密集层。像其他当时的分类器一样，为了防止记忆化，它在分类之前添加了一个 dropout 层进行正则化。

你可以在图 8.5 中看到，分类组件包含一个 `GlobalAveragePooling2D` 层，用于展平特征图并将高维编码降低到低维编码（每个特征图 1 个像素）。然后使用 softmax 激活函数，其中滤波器的数量是类别的数量，通过一个 `Reshape` 层将 1D 向量重塑为 2D 向量。然后是另一个 `Reshape` 层，将输出重塑回 1D 向量（每个类别一个元素）。在 2D 卷积之前是用于正则化的 `Dropout` 层。

![](img/CH08_F05_Ferlitsch.png)

图 8.5 MobileNet v1 分类组使用卷积层进行分类

下面的示例实现是分类组件。第一个 `Reshape` 层将 `GlobalAveragePooling2D` 的 1D 向量重塑为大小为 1 × 1 的 2D 向量。第二个 `Reshape` 层将 `Conv2D` 的 2D 1 × 1 输出重塑为用于 softmax 概率分布（分类）的 1D 向量：

```
def classifier(x, alpha, dropout, n_classes):
    """ Construct the classifier group
        x         : input to the classifier
        alpha     : width multiplier
        dropout   : dropout percentage
        n_classes : number of output classes
    """
    x = GlobalAveragePooling2D()(x)                                         ❶

    shape = (1, 1, int(1024 * alpha))                                       ❷
    x = Reshape(shape)(x)

    x = Dropout(dropout)(x)                                                 ❸

    x = Conv2D(n_classes, (1, 1), padding='same', activation='softmax')(x)  ❹

    x = Reshape((n_classes, ))(x)                                           ❺
    return x
```

❶ 将特征图展平为 1D 特征图 (*α*, N)

❷ 将展平的特征图重塑为 (*α*, 1, 1, 1024)

❸ 执行 dropout 以防止过拟合

❹ 使用卷积进行分类（模拟全连接层）

❺ 将结果输出重塑为包含类别数量的 1D 向量

使用 Idiomatic procedure reuse 设计模式对 MobileNet v1 进行完整代码实现的示例位于 GitHub 上 ([`mng.bz/Q2rG`](https://shortener.manning.com/Q2rG)).

## 8.2 MobileNet v2

在改进版本 1 之后，谷歌在 2018 年 Mark Sandler 等人撰写的“MobileNetV2: Inverted Residuals and Linear Bottlenecks”一文中引入了 *MobileNet v2* ([`arxiv.org/abs/1801.04381`](https://arxiv.org/abs/1801.04381))。新的架构用倒残差块替换了卷积块以提高性能。该论文总结了倒残差块的好处：

+   显著减少操作数量，同时保持与卷积块相同的准确度

+   显著减少推理所需的内存占用

### 8.2.1 架构

MobileNet v2 架构结合了几个针对受限内存设备的设计原则：

+   它继续使用超参数（alpha）作为宽度乘数，如 v1 中所述，在根部和学习者组件中进行网络稀疏化。

+   继续使用深度可分离卷积代替常规卷积，如 v1 中所述，以显著降低计算复杂度（延迟），同时保持几乎相当的表现力。

+   用残差块替换卷积块，允许更深的层以获得更高的准确度。

+   引入了一种新的残差块设计，作者们称之为*倒残差块*。

+   用 1 × 1 非线性卷积替换 1 × 1 线性卷积。

根据作者的说法，最后修改的原因，使用 1 × 1 线性卷积： “此外，我们发现，为了保持表现力，移除狭窄层中的非线性是很重要的。” 在他们的消融研究中，他们比较了使用 1 × 1 非线性卷积（带有`ReLU`）和使用 1 × 1 线性卷积（不带`ReLU`），通过移除 ReLU 在 ImageNet 上获得了 1%的 top-1 准确度提升。

作者们将他们的主要贡献描述为一种新颖的层模块：线性瓶颈的倒残差。我在第 8.2.3 节中详细描述了倒残差块。

图 8.6 展示了 MobileNet v2 架构。在宏观架构中，学习者组件由四个倒残差组组成，随后是一个最终的 1 × 1 线性卷积，这意味着激活函数是线性的。每个倒残差组将前一个组的滤波器数量增加。每个组的滤波器数量通过元参数宽度乘数*α*（alpha）进行稀疏化。最终的 1 × 1 卷积进行线性投影，将最终的特征图数量增加到四倍，达到 2048。

![](img/CH08_F06_Ferlitsch.png)

图 8.6 MobileNet v2 宏观架构

### 8.2.2 根部

根部组件与 v1 相似，除了在初始 3 × 3 卷积层之后，它不跟随 v1 中的深度卷积块（图 8.7）。因此，粗粒度特征提取的表现力将低于 v1 中的 3 × 3 双栈。作者们没有说明为什么表现力的降低没有影响模型，该模型在准确度上优于 v1。

![](img/CH08_F07_Ferlitsch.png)

图 8.7 MobileNet v2 根部组

### 8.2.3 学习者

学习组件由七个倒置残差组组成，随后是一个 1 × 1 的线性卷积。每个倒置残差组包含两个或更多的倒置残差块。每个组逐渐增加滤波器的数量，也称为*输出通道*。每个组从步长卷积开始，随着每个组逐渐增加特征图（通道）的数量，减少特征图的大小（通道）。

图 8.8 描述了一个 MobileNet v2 组，其中第一个倒置残差块进行了步长操作以减少特征图的大小，以抵消每个组中特征图数量逐渐增加的趋势。如图表所示，只有第 2、3、4 和 6 组以步长倒置残差块开始。换句话说，第 1、5 和 7 组以非步长残差块开始。此外，每个非步长块都有一个恒等连接，而步长块没有恒等连接。

![图片](img/CH08_F08_Ferlitsch.png)

图 8.8 MobileNet v2 组微架构

以下是一个 MobileNet v2 组的示例实现。该组遵循以下惯例：第一个块执行降维以减少特征图的大小。在这种情况下，第一个倒置块是步长的（特征池化），其余块不是步长的（无特征池化）。

```
def group(x, n_filters, n_blocks, alpha, expansion=6, strides=(2, 2)):
    """ Construct an Inverted Residual Group
        x         : input to the group
        n_filters : number of filters
        n_blocks  : number of blocks in the group
        alpha     : width multiplier
        expansion : multiplier for expanding the number of filters
        strides   : whether the first inverted residual block is strided.
    """ 
    x = inverted_block(x, n_filters, alpha, expansion, strides=strides)    ❶

    for _ in range(n_blocks - 1):                                          ❷
        x = inverted_block(x, n_filters, alpha, expansion, strides=(1, 1))
    return x
```

❶ 组中的第一个倒置残差块可能是步长的。

❷ 构建剩余的块

该块被称为*倒置残差块*，因为它反转（倒置）了围绕中间卷积层的降维和扩展关系，这与传统的残差块不同，例如在 ResNet50 中。它不是从 1 × 1 的瓶颈卷积开始进行降维，以 1 × 1 的线性投影卷积结束以恢复维度，而是顺序相反。一个倒置块从 1 × 1 的投影卷积开始进行维度扩展，并以 1 × 1 的瓶颈卷积结束以恢复维度（图 8.9）。

![图片](img/CH08_F09_Ferlitsch.png)

图 8.9 残差瓶颈块与倒置残差块之间的概念差异

在他们比较 MobileNet v1 中的瓶颈残差块设计与 v2 中的倒置残差块设计的消融研究中，作者在 ImageNet 上实现了 1.4%的 top-1 准确率提升。倒置残差块设计也更加高效，将总参数数量从 420 万减少到 340 万，并将 matmul 操作的数量从 5.75 亿减少到 3 亿。

接下来，我们将更深入地探讨反演背后的机制。MobileNet v2 引入了一种新的元参数扩展，用于初始的 1 × 1 投影卷积。1 × 1 投影卷积执行维度扩展，而元参数指定了扩展滤波器数量的量。换句话说，1 × 1 投影卷积将特征图数量扩展到高维空间。

中间卷积是一个 3 × 3 深度卷积。这随后是一个线性点卷积，它减少了特征图（也称为 *通道*），将它们恢复到原始数量。请注意，恢复卷积使用线性激活而不是非线性（ReLU）。作者发现，为了保持表示能力，移除狭窄层中的非线性很重要。

作者还发现，ReLU 激活在低维空间中会丢失信息，但当有大量滤波器时可以弥补这一点。这里的假设是，块的输入处于低维空间，但扩展了滤波器的数量，因此保持使用 ReLU 激活在第一个 1 × 1 卷积中的原因。

MobileNet v2 研究人员将扩展量称为块的 *表达能力*。在他们主要的实验中，他们尝试了 5 到 10 之间的扩展因子，并观察到准确率几乎没有差异。由于扩展的增加会导致参数数量的增加，而准确率的提升却很小，因此作者们在消融研究中使用了 6 的扩展比率。

图 8.10 展示了反演残差块。你可以看到其设计在减少内存占用同时保持准确性的基础上又迈出了新的一步。

![](img/CH08_F10_Ferlitsch.png)

图 8.10 带有恒等快捷连接的反演残差块反转了 v1 中 1 × 1 卷积的关系。

下面的示例实现了一个反演残差块。为了理解上下文，请记住，反演残差块的输入是来自先前块或低维空间中的主干组的输出。然后，通过 1 × 1 投影卷积将输入投影到更高维的空间，其中执行 3 × 3 深度卷积。然后，点卷积的 1 × 1 线性卷积将输出恢复到输入的较低维度。

这里有一些显著的步骤：

+   宽度因子应用于块的输出滤波器数量：`filters = int(n_filters` `*` `alpha)`。

+   输入通道（特征图）的数量由 `n_channels` `= int(x.shape[-1])` 确定。

+   当 `expansion` 因子大于 1 时，应用 1 × 1 线性投影。

+   在第一个组的第一块之外，每个块都执行 `Add()` 操作：`if` `n_channels` `==` `filters` `and` `strides` `==` `(1,` `1)`。

```
def inverted_block(x, n_filters, alpha, expansion=6, strides=(1, 1)):
    """ Construct an Inverted Residual Block
        x         : input to the block
        n_filters : number of filters
        alpha     : width multiplier
        expansion : multiplier for expanding number of filters
        strides   : strides
    """
    shortcut = x  # Remember input

    filters = int(n_filters * alpha)                                  ❶

    n_channels = int(x.shape[-1])

    if expansion > 1:                                                 ❷
        # 1x1 linear convolution
        x = Conv2D(expansion * n_channels, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)

    if strides == (2, 2):                                             ❸
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        padding = 'valid'
    else:
        padding = 'same'

    x = DepthwiseConv2D((3, 3), strides, padding=padding)(x)          ❹
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)    ❺
    x = BatchNormalization()(x)

    if n_channels == filters and strides == (1, 1):                   ❻
        x = Add()([shortcut, x])
    return x
```

❶ 将宽度乘数应用于点卷积的特征图数量

❷ 当不是组中的第一个块时，进行维度扩展（dimensionality expansion）

❸ 在步进卷积（特征池化）时向特征图添加零填充（zero padding）

❹ 3 × 3 深度卷积

❺ 1 × 1 线性点卷积

❻ 当输入滤波器数量与输出滤波器数量相匹配时，向输出添加身份链接（identity link）

### 8.2.4 分类器

在 v2 版本中，研究人员采用了传统的`GlobalAveragePooling2D`层后跟`Dense`层的方法，这在第五章第 5.4 节中已经介绍过。早期的卷积神经网络，如 AlexNet、ZFNet 和 VGG，会将瓶颈层（最终特征图）进行扁平化，然后接一个或多个隐藏密集层，最后是用于分类的最终密集层。例如，VGG 在最终密集层之前使用了两个包含 4096 个节点的层。

随着表示学习（representational learning）的改进，从 ResNet 和 Inception 开始，分类器中隐藏层的需求变得不再必要，同样，不需要将数据降维到瓶颈层（bottleneck layer）的扁平化层。MobileNet v2 沿袭了这一做法，当潜在空间（latent space）具有足够的表示信息时，我们可以进一步将其降低到低维空间——瓶颈层。在高表示信息下，模型可以将低维度的数据，也称为*嵌入*或*特征向量*，直接传递到分类器的密集层，而不需要中间的隐藏密集层。图 8.11 展示了分类器组件。

![图片](img/CH08_F11_Ferlitsch.png)

图 8.11 MobileNet v2 分类器组

在作者的消融研究中，他们比较了 MobileNet v1 和 v2 在 ImageNet 分类任务上的表现。MobileNet v2 实现了 72% 的 top-1 准确率，而 v1 实现了 70.6%。使用 Idiomatic procedure reuse 设计模式对 MobileNet v2 的完整代码实现可在 GitHub 上找到 ([`mng.bz/Q2rG`](http://mng.bz/Q2rG))。

接下来，我们将介绍 SqueezeNet，它引入了 fire 模块以及用于配置微架构属性的宏观架构和元参数的术语。虽然当时其他研究人员也在探索这个概念，但 SqueezeNet 的作者为这个创新性的里程碑式进展创造了术语，为后来宏观架构搜索、机器设计和模型融合的进步奠定了基础。对我个人而言，当我第一次阅读他们的论文和这些概念时，感觉就像一个灯泡突然亮了起来。

## 8.3 SqueezeNet

SqueezeNet 是由 DeepScale、加州大学伯克利分校和斯坦福大学于 2016 年共同研究引入的架构。在相应的"SqueezeNet"论文（"SqueezeNet: AlexNet-Level Accuracy with 50x Fewer Parameters and <0.5MB Model Size"; [`arxiv.org/abs/1602.07360`](https://arxiv.org/abs/1602.07360)）中，Forrest N. Iandola 等人介绍了一种新型模块，即"fire 模块"，以及微架构、宏架构和超参数的术语。作者们的目标是找到一个参数更少但与知名 AlexNet 模型具有相当准确性的 CNN 架构。

火模块的设计基于他们对微架构的研究以实现这一目标。"微架构"是模块或组的设计，而"宏架构"则是模块或组之间的连接方式。引入"超参数"这一术语有助于更好地区分什么是超参数（在第十章中详细讨论）。

通常，在训练过程中学习的权重和偏差是模型参数。"超参数"这个术语可能会令人困惑。一些研究人员/实践者使用这个术语来指代用于训练模型的可调参数，而其他人则使用这个术语来包括模型架构（例如，层和宽度）。在 SqueezeNet 论文中，作者们使用"元参数"来指代可配置的模型架构结构——例如，每组中的块数，每个块中卷积层的滤波器数量，以及组末端的维度缩减量。

作者们在他们的论文中解决了几个问题。首先，他们想要展示一种 CNN 架构设计，这种设计可以在移动设备上运行，同时仍然保持与 ImageNet 2012 数据集上 AlexNet 相当的准确性。在这方面，作者们在参数数量减少了 50 倍的情况下，通过经验实证达到了与 AlexNet 相同的结果。

其次，他们想要展示一种小型 CNN 架构，在压缩后仍能保持准确性。在这里，作者们在使用深度压缩算法压缩后，没有压缩的情况下达到了相同的结果，将模型的大小从 4.8 MB 减少到 0.47 MB。将模型大小降低到 0.5 MB 以下，同时保持 AlexNet 的准确性，证明了在极端内存受限的物联网设备（如微控制器）上放置模型的实用性。

在他们的 SqueezeNet 论文中，作者们将实现目标的设计原则称为策略 1、2 和 3：

+   *策略 1*——主要使用 1 × 1 滤波器，这比更常见的 3 × 3 滤波器减少了 9 倍的参数数量。SqueezeNet 的 v1.0 版本使用了 1 × 1 到 3 × 3 滤波器的 2:1 比例。

+   *策略 2*—减少 3 × 3 层的输入滤波器数量以进一步减少参数数量。他们将火模块的这个部分称为 *挤压层*。

+   *策略 3*—尽可能晚地延迟特征图的下采样。这与早期下采样以保持精度的传统做法相反。作者在早期卷积层使用了步长为 1，而延迟使用步长为 2。

作者陈述了他们策略的以下理由：

*策略 1 和 2 是关于在尝试保持精度的同时，巧妙地减少 CNN 中的参数数量。策略 3 是关于在有限的参数预算内最大化精度*。

作者将他们的架构命名为其 fire 块的设计，该设计使用了一个挤压操作后跟一个扩展操作。

### 8.3.1 架构

SqueezeNet 架构由一个主干组、三个包含总共八个 fire 块（在论文中称为 *模块*）的 fire 组和一个分类器组组成。作者没有明确说明他们为什么选择三个 fire 组和八个 fire 块，但描述了一种宏观架构探索，该探索展示了一种成本效益高的方法，即通过训练每组中块的数量和输入到输出的滤波器大小的不同组合来设计针对特定内存足迹和精度范围的模型。

图 8.12 展示了架构。在宏观架构视图中，你可以看到三个 fire 组。特征学习在主干组和前两个 fire 组中进行。最后一个 fire 组与分类组重叠，进行特征学习和分类学习。

![](img/CH08_F12_Ferlitsch.png)

图 8.12 SqueezeNet 宏观架构

前两个 fire 组将输入到输出的特征图数量翻倍，从 16 开始，翻倍到 32，然后再次翻倍到 64。第一和第二 fire 组都延迟了维度减少到组的末尾。最后一个 fire 组不进行特征图数量的加倍或维度减少，但在组的末尾添加了一个 dropout 以进行正则化。这一步骤与当时的传统做法不同，当时 dropout 层本应放置在瓶颈层（特征图减少并展平为 1D 向量）之后的分类器组中。

### 8.3.2 主干

主干组件使用了一个粗略级别的 7 × 7 卷积层，这与当时使用 5 × 5 或重构的两个 3 × 3 卷积层的传统做法相反。主干执行了激进的特性图减少，这继续是现在的传统做法。

粗略的 7 × 7 卷积进行了步长（特征池化）以实现 75% 的减少，随后是一个最大池化层以进一步减少 75%，结果得到特征图的大小仅为输入通道的 6%。图 8.13 描述了主干组件。

![](img/CH08_F13_Ferlitsch.png)

图 8.13 SqueezeNet 主干组

### 8.3.3 学习者

学习者由三个火组组成。第一个火组的输入为 16 个滤波器（通道），输出为 32 个滤波器（通道）。回想一下，主干输出 96 个通道，因此第一个火组通过减少到 16 个滤波器对输入进行降维。第二个火组将这个数量加倍，输入为 32 个滤波器（通道），输出为 64 个滤波器（通道）。

第一个和第二个火组都由多个火块组成。除了最后一个火块外，所有火块使用相同数量的输入滤波器。最后一个火块将输出滤波器的数量加倍。两个火组都使用`MaxPooling2D`层将特征图的下采样延迟到组的末尾。

第三个火组由一个 64 个滤波器的单个火块组成，随后是一个用于正则化的 dropout 层，在分类组之前。这与当时的惯例略有不同，因为 SqueezeNet 的 dropout 层出现在分类器的瓶颈层之前，而不是之后。图 8.14 描述了一个火组。

![](img/CH08_F14_Ferlitsch.png)

图 8.14 在 SqueezeNet 组微架构中，最后一个火组使用 dropout 而不是 max pooling。

以下是对第一个和第二个火组的示例实现。请注意，参数`filters`是一个列表，其中每个元素对应一个火块，其值是该块的滤波器数量。例如，考虑第一个火组，它由三个火块组成；输入为 16 个滤波器，输出为 32 个滤波器。参数`filters`将是列表[16, 16, 32]。

在为组添加所有火块之后，添加一个`MaxPooling2D`层以进行延迟下采样：

```
def group(x, filters):
    ''' Construct a Fire Group
        x     : input to the group
        filters: list of number of filters per fire block (module)
    '''
    for n_filters in filters:                      ❶
        x = fire_block(x, n_filters)

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)    ❷
    return x
```

❶ 为组添加火块（模块）

❷ 在组的末尾添加延迟下采样

图 8.15 说明了火块，它由两个卷积层组成。第一层是挤压层，第二层是扩展层。*挤压层*通过使用 1 × 1 瓶颈卷积将输入通道的数量减少到更低的维度，同时保持足够的信息供扩展层中的后续卷积使用。挤压操作显著减少了参数数量和相应的矩阵乘法操作。换句话说，1 × 1 瓶颈卷积学习最大化挤压特征图数量到更少的特征图的最佳方式，同时仍然能够在后续的扩展*层*中进行特征提取。

![](img/CH08_F15_Ferlitsch.png)

图 8.15 SqueezeNet 火块

*扩展层*是两个卷积的分支：一个 1 × 1 的线性投影卷积和一个 3 × 3 卷积，特征提取发生在其中。卷积的输出（feature maps）随后被连接。扩展层通过 8 倍因子扩展了 feature maps 的数量。

让我们举一个例子。来自主干的第一个 fire 块的输入是 96 个特征图（通道），squeeze 层将其减少到 16 个特征图。然后扩展层将其扩展 8 倍，因此输出再次是 96 个特征图。下一个（第二个）fire 块再次将其压缩到 16 个特征图，以此类推。

下面的示例是一个 fire 块的实现。该块从 squeeze 层的 1 × 1 瓶颈卷积开始。squeeze 层的输出`squeeze`分支到两个并行扩展卷积`expand1x1`和`expand3x3`。最后，两个扩展卷积的输出被连接在一起。

```
def fire_block(x, n_filters):
    ''' Construct a Fire Block
        x        : input to the block
        n_filters: number of filters
    '''
    squeeze = Conv2D(n_filters, (1, 1), strides=1, activation='relu', 
                     padding='same')(x)                                     ❶

    expand1x1 = Conv2D(n_filters * 4, (1, 1), strides=1, activation='relu',
                       padding='same')(squeeze)                             ❷
    expand3x3 = Conv2D(n_filters * 4, (3, 3), strides=1, activation='relu', ❷
                       padding='same')(squeeze)                             ❷

    x = Concatenate()([expand1x1, expand3x3])                               ❸
    return x
```

❶ 带有 1 × 1 瓶颈卷积的 squeeze 层

❷ 扩展层分支为 1 × 1 和 3 × 3 卷积，并加倍了滤波器的数量。

❸ 从激励层输出的分支被连接在一起。

### 8.3.4 分类器

分类器不遵循传统的`GlobalAveragingPooling2D`层后跟一个`Dense`层（输出节点的数量等于类别的数量）的做法。相反，它使用一个卷积层，滤波器的数量等于类别的数量，然后跟一个`GlobalAveragingPooling2D`层。这种安排将每个先前的滤波器（类别）减少到单个值。然后，`GlobalAveragingPooling2D`层的输出通过 softmax 激活，得到所有类别的概率分布。

让我们重新审视一个传统的分类器。在传统的分类器中，最终的 feature maps 在瓶颈层被减少并展平到更低的维度，通常使用`GlobalAveragingPooling2D`。现在，每个 feature map 将有一个像素作为 1D 向量（嵌入）。这个 1D 向量随后被传递到一个密集层，其中节点的数量等于输出类别的数量。

图 8.16 显示了分类器组件。在 SqueezeNet 中，最终的 feature maps 通过一个 1 × 1 的线性投影，该投影学习将最终的 feature maps 投影到一个新的集合，该集合正好等于输出类别的数量。现在，这些投影的 feature maps，每个对应一个类别，被减少到每个 feature map 的单个像素，并展平，成为一个长度正好等于输出类别数量的 1D 向量。这个 1D 向量随后通过 softmax 进行预测。

![图片](img/CH08_F16_Ferlitsch.png)

图 8.16 使用卷积而不是密集层进行分类的 SqueezeNet 分类器组

基本区别是什么？在传统的分类器中，密集层学习分类。在这个移动版本中，1 × 1 线性投影学习分类。

以下是对分类器的一个示例实现。在这个例子中，输入是最终的特征图，通过一个`Conv2D`层进行 1 × 1 线性投影到输出类别数量。随后，特征图通过`GlobalAveragePooling2D`被减少到一个单像素的 1D 向量：

```
def classifier(x, n_classes):
    ''' Construct the Classifier
        x        : input to the classifier
        n_classes: number of output classes
    '''
    x = Conv2D(n_classes, (1, 1), strides=1, activation='relu', 
               padding='same')(x)        ❶

    x = GlobalAveragePooling2D()(x)      ❷
    x = Activation('softmax')(x)         ❷
    return x
```

❶ 将过滤器数量设置为类别数量

❷ 将每个过滤器（类别）简化为单个值用于分类

接下来，让我们通过构建传统的大规模 SOTA 模型的传统方法来更深入地了解分类器设计。图 8.17 展示了传统方法。最终的特征图被全局池化成一个 1 × 1 矩阵（一个值）。然后矩阵被展平成一个长度等于特征图数量的 1D 向量（例如 ResNet 中的 2048）。然后 1D 向量通过一个具有 softmax 激活的密集层，输出每个类别的概率。

![图片](img/CH08_F17_Ferlitsch.png)

图 8.17 传统的大规模 SOTA 分类器中的特征图处理

图 8.18 展示了 SqueezeNet 中的方法。特征图通过一个 1 × 1 瓶颈卷积进行处理，将特征图的数量减少到类别数量。本质上，这是类别预测步骤——除了我们没有单个值，而是一个*N* × *N*矩阵。然后*N* × *N*矩阵的预测被全局池化成 1 × 1 矩阵，这些矩阵随后被展平成一个 1D 向量，其中每个元素是对应类别的概率。

![图片](img/CH08_F18_Ferlitsch.png)

图 8.18 使用卷积而不是密集层进行分类

### 8.3.5 绕过连接

在他们的消融研究中，作者使用 ResNet 中引入的恒等连接对块进行微架构搜索，他们将这种连接称为*绕过连接*。他们在论文中说，SqueezeNet 位于“CNN 架构的广泛且大部分未探索的设计空间中。”他们探索的一部分包括他们所说的*微架构设计空间*。他们指出，他们受到了 ResNet 作者在 ResNet34 上带有和不带有绕过连接的 A/B 比较的启发，并通过绕过连接获得了 2%的性能提升。

作者尝试了他们所说的简单绕过和复杂绕过。在*简单绕过*中，他们在 ImageNet 上获得了 2.9%的 top-1 准确率提升和 2.2%的 top-5 准确率提升，而没有增加计算复杂度。因此，他们的改进与 ResNet 作者观察到的改进相当。

在*复杂旁路*中，他们观察到较小的改进，准确率仅提高了 1.3%，模型大小从 4.8 MB 增加到 7.7 MB。在简单旁路中，模型大小没有增加。作者得出结论，简单旁路是足够的。

简单旁路

在简单旁路中，恒等链接仅在第一个 fire 块（组入口）和过滤器加倍之前的 fire 块中出现。图 8.19 说明了具有简单旁路连接的 fire 组。组中的第一个 fire 块具有旁路连接（恒等链接），然后是 fire 块，它将输出通道数（特征图）加倍。

![图片](img/CH08_F19_Ferlitsch.png)

图 8.19 SqueezeNet 组带有简单旁路块

现在我们来近距离观察一个具有简单旁路（恒等链接）连接的 fire 块。这如图 8.20 所示。请注意，块输入被添加到连接操作的输出中。

![图片](img/CH08_F20_Ferlitsch.png)

图 8.20 SqueezeNet fire 块带有恒等链接

让我们一步步来看。首先，我们知道，使用矩阵加法操作，输入上的特征图数量必须与连接操作的输出数量相匹配。对于许多 fire 块来说，这是正确的。例如，从 stem 组中，我们有 96 个特征图作为输入，在 squeeze 层中减少到 16，然后通过 expand 层扩展 8 倍（回到 96）。由于输入上的特征图数量等于输出，我们可以添加一个恒等链接。但并非所有 fire 块都是这样，这就是为什么只有一部分具有旁路连接。

以下是一个具有简单旁路连接（恒等链接）的 fire 块的示例实现。在这个实现中，我们传递额外的参数`bypass`。如果它是真的，我们在块的末尾添加一个最终层，该层对来自连接操作的输出执行矩阵加法（`Add()`）：

```
def fire_block(x, n_filters, bypass=False):
    ''' Construct a Fire Block
        x        : input to the block
        n_filters: number of filters in the block
        bypass   : whether block has an identity shortcut
    '''
    shortcut = x

    squeeze = Conv2D(n_filters, (1, 1), strides=1, activation='relu', 
                     padding='same')(x)

    expand1x1 = Conv2D(n_filters * 4, (1, 1), strides=1, activation='relu',
                       padding='same')(squeeze)
    expand3x3 = Conv2D(n_filters * 4, (3, 3), strides=1, activation='relu',
                       padding='same')(squeeze)

    x = Concatenate()([expand1x1, expand3x3])

    if bypass:                      ❶
        x = Add()([x, shortcut])

    return x
```

❶ 当 bypass 为 True 时，输入（快捷方式）会矩阵加到 fire 块的输出上。

复杂旁路

在作者接下来的微架构搜索中，他们探索了在不使用恒等链接（简单旁路）的情况下向剩余的 fire 块添加线性投影。线性投影会将输入特征的数量投影到连接操作后等于输出特征图数量的数量。他们将这称为*复杂旁路*。

目的是看看这是否会进一步提高 top-1/top-5 准确率，尽管会增加模型大小。正如我之前提到的，他们的实验表明使用复杂旁路对目标是有害的。图 8.21 描绘了一个 fire 组，其中剩余的没有简单旁路（恒等链接）的 fire 块具有复杂旁路（线性投影链接）。

![图片](img/CH08_F21_Ferlitsch.png)

图 8.21 SqueezeNet 组带有投影快捷 fire 块（复杂旁路）

现在让我们更详细地看看图 8.22 中所示的带复杂旁路的 fire 块。请注意，在身份链接上的 1 × 1 线性投影将滤波器（通道）的数量增加了 8。这是为了匹配分支 1 × 1 和 3 × 3 输出连接的大小，两者都增加了输出大小 4（4 + 4 = 8）。在身份链接上使用 1 × 1 线性投影是将复杂旁路与简单旁路区分开来的关键。

![图片](img/CH08_F22_Ferlitsch.png)

图 8.22 SqueezeNet 带投影快捷方式的 fire 块（复杂旁路）

在消融研究中，使用简单的旁路将 ImageNet 上 vanilla SqueezeNet 的准确率从 57.5%提高到 60.4%。对于复杂旁路，准确率仅提高到 58.8%。作者没有对为什么会出现这种情况做出结论，除了说这很有趣。使用 Idiomatic procedure reuse 设计模式对 SqueezeNet 的完整代码实现可在 GitHub 上找到（[`mng.bz/XYmv`](https://shortener.manning.com/XYmv)）。

接下来，我们将介绍 ShuffleNet，它引入了点卷积和通道洗牌（转置）操作，在不增加计算复杂性和尺寸的情况下增加特征图的数量。

## 8.4 ShuffleNet v1

大型网络的一个挑战是它们需要许多特征图，通常有数千个，这意味着它们有很高的计算成本。因此，在 2017 年，Face++的 Xiangyu Zhang 等人提出了一种在大幅降低计算成本的同时拥有大量特征图的方法。这种新的架构称为*ShuffleNet v1* ([`arxiv.org/abs/1707.01083`](https://arxiv.org/abs/1707.01083))，专门为通常在手机、无人机和机器人上发现的低计算设备设计。

该架构引入了新的层操作：分组点卷积和通道洗牌。与 MobileNet 相比，作者发现 ShuffleNet 通过显著的优势实现了更好的性能：在 40 MFLOPs 的水平上，ImageNet top-1 错误率绝对降低了 7.8%。虽然作者报告了在 MobileNet 对应版本上的准确率提升，但 MobileNets 仍然在生产中受到青睐，尽管它们现在正被 EfficientNets 所取代。

### 8.4.1 架构

ShuffleNet 架构由三个洗牌组组成，论文中将其称为*阶段*。该架构遵循传统做法，每个组将前一个组的输出通道或特征图数量翻倍。图 8.23 展示了 ShuffleNet 架构。

![图片](img/CH08_F23_Ferlitsch.png)

图 8.23 在 ShuffleNet v1 宏架构中，每个组将输出特征图的数量翻倍。

### 8.4.2 茎

与当时其他移动端 SOTA 模型相比，主干组件使用了更精细的 3 × 3 卷积层，而其他模型通常使用 7 × 7 或两个 3 × 3 卷积层的堆叠。主干组件，如图 8.24 所示，执行了激进的特征图降维，这至今仍是一种惯例。3 × 3 卷积层采用步长（特征池化）以实现 75%的降维，随后通过一个最大池化层进一步降维 75%，结果得到的特征图大小仅为输入通道的 6%。从输入到 6%的通道尺寸降低一直是一种传统做法。

![图像](img/CH08_F24_Ferlitsch.png)

图 8.24 ShuffleNet 主干组件通过结合特征和最大池化来降低输出特征图的大小，降至输入大小的 6%。

### 8.4.3 学习组件

学习组件中的每个组由一个步长洗牌块（在论文中称为“单元”）组成，后面跟一个或多个洗牌块。步长洗牌块将输出通道数加倍，同时将每个通道的大小减少 75%。每个特征中过滤器数量和输出特征图的逐步加倍，当时是一种惯例，并且至今仍保持。同时，当一组将输出特征图的数量加倍时，它们的尺寸也会减少，以防止在层中深入时参数增长爆炸。

组

与 MobileNet v1/v2 类似，ShuffleNet 组在组的开始处进行特征图降维，使用步长洗牌块。这与 SqueezeNet 和大型 SOTA 模型将特征图降维推迟到组末的做法形成对比。通过在组开始处减小尺寸，参数数量和矩阵乘法操作的数量显著减少，但代价是减少了表示能力。

图 8.25 说明了洗牌组。组从步长洗牌块开始，它在组的开始处进行特征图尺寸的降低，然后跟一个或多个洗牌块。步长和随后的洗牌块将前一个组的过滤器数量加倍。例如，如果前一个组有 144 个过滤器，当前组将加倍到 288 个。

![图像](img/CH08_F25_Ferlitsch.png)

图 8.25 ShuffleNet 组微观架构

下面的示例实现了一个洗牌组。参数`n_blocks`是组中的块数，`n_filters`是每个块的过滤器数量。参数`reduction`是洗牌块中维度降低的元参数（随后讨论），参数`n_partitions`是用于通道洗牌的分区元参数（随后讨论）。第一个块是一个步长洗牌块，其余块不是步长的：`for _ in range(n_blocks-1)`。

```
def group(x, n_partitions, n_blocks, n_filters, reduction):
    ''' Construct a Shuffle Group
        x            : input to the group
        n_partitions : number of groups to partition feature maps (channels) 
        ➥ into.
        n_blocks     : number of shuffle blocks for this group
        n_filters    : number of output filters
        reduction    : dimensionality reduction
    '''
    x = strided_shuffle_block(x, n_partitions, n_filters, reduction)   ❶

    for _ in range(n_blocks-1):                                        ❷
        x = shuffle_block(x, n_partitions, n_filters, reduction)
    return x
```

❶ 组中的第一个块是一个步长洗牌块。

❷ 添加剩余的非步长洗牌块

块

Shuffle 块基于 B(1, 3, 1)残差块，其中 3 × 3 卷积是深度卷积（如 MobileNet）。作者指出，由于昂贵的密集 1 × 1 卷积，Xception 和 ResNeXt 等架构在极小的网络中效率降低。为了解决这个问题，他们用逐点组卷积替换了 1 × 1 逐点卷积，以降低计算复杂度。图 8.26 显示了设计上的差异。

![](img/CH08_F26_Ferlitsch.png)

图 8.26 比较 ResNet 和 ShuffleNet B(1,3,1)设计

当参数`reduction`小于 1 时，第一个逐点组卷积也会对块输入的滤波器数量进行降维（`reduction` `*` `n_filters`），然后在第二个逐点组卷积的输出通道中恢复，以匹配矩阵加法操作的输入残差。

他们还偏离了 Xception 中在深度卷积后使用 ReLU 的惯例，转而使用线性激活。他们对此变化的原因并不明确，使用线性激活的优势也不清楚。论文仅陈述：“批归一化（BN）和非线性的使用与[ResNet, ResNeXt]类似，只是我们没有像[Xception]建议的那样在深度卷积后使用 ReLU。”在第一个逐点组卷积和深度卷积之间是通道洗牌操作，这两个操作将在后面讨论。

图 8.27 展示了 shuffle 块。你可以看到通道洗牌是如何在 3 × 3 深度卷积之前插入到 B(1,3,1)残差块设计中的，特征提取就在这里发生。B(1, 3, 1)残差块是一个与 MobileNet v1 相当的瓶颈设计，其中第一个 1 × 1 卷积进行降维，第二个 1 × 1 卷积进行升维。该块继续遵循 MobileNet 中的惯例，将 3 × 3 深度卷积与 1 × 1 逐点组卷积配对，形成一个深度可分离卷积。尽管如此，它与 MobileNet v1 的不同之处在于，将第一个 1 × 1 瓶颈卷积改为 1 × 1 瓶颈逐点组卷积。

![](img/CH08_F27_Ferlitsch.png)

图 8.27 ShuffleNet 块使用惯用设计

以下是一个 shuffle 块的示例实现。该块从函数`pw_group_conv``()`中定义的逐点 1 × 1 组卷积开始，其中参数值`int(reduction * n_filters)`指定了降维。接下来是函数`channel_shuffle()`中定义的通道洗牌，然后是深度卷积（`DepthwiseConv2D`）。接下来是最终的逐点组 1 × 1 卷积，它恢复了维度。最后，将块的输入通过矩阵加法（`Add()`）与逐点组卷积的输出相加。

```
def shuffle_block(x, n_partitions, n_filters, reduction):
    ''' Construct a shuffle Shuffle block
        x           : input to the block
        n_partitions: number of groups to partition feature maps (channels) into.
        n_filters   : number of filters
        reduction   : dimensionality reduction factor (e.g, 0.25)
    '''

    shortcut = x

    x = pw_group_conv(x, n_partitions, int(reduction * n_filters))    ❶
    x = ReLU()(x)

    x = channel_shuffle(x, n_partitions)                              ❷

    x = DepthwiseConv2D((3, 3), strides=1, padding='s                 ❸
    x = BatchNormalization()(x)

    x = pw_group_conv(x, n_partitions, n_filters)                     ❹

    x = Add()([shortcut, x])                                          ❺
    x = ReLU()(x)
    return x
```

❶ 第一个逐点组卷积操作进行了一次降维。

❷ 通道洗牌

❸ 3 × 3 深度卷积

❹ 第二组卷积进行维度恢复。

❺ 将输入（快捷连接）添加到块的输出中

点卷积组

以下是一个点卷积组卷积的示例实现。函数首先确定输入通道的数量 `(in_filters = x.shape[-1])`。接下来，通过将输入通道数除以组数 (`n_partitions`) 来确定每个组的通道数。然后，特征图按比例分配到各个组中 (`lambda`)，每个组通过一个单独的 1 × 1 点卷积。最后，将组卷积的输出连接在一起并通过批量归一化层。

```
def pw_group_conv(x, n_partitions, n_filters):
    ''' A Pointwise Group Convolution
        x        : input tensor
        n_groups : number of groups to partition feature maps (channels) into.
        n_filers : number of filters
    '''
    in_filters = x.shape[-1]                                         ❶

    grp_in_filters  = in_filters // n_partitions                     ❷
    grp_out_filters = int(n_filters / n_partitions + 0.5)            ❷

    groups = []                                                      ❸
    for i in range(n_partitions):
        group = Lambda(lambda x: x[:, :, :, grp_in_filters * i: 
                                   grp_in_filters * (i + 1)])(x)     ❹

        conv = Conv2D(grp_out_filters, (1,1), padding='same', strides=1)(group)

        groups.append(conv)                                          ❺

    x = Concatenate()(groups)                                        ❻
    x = BatchNormalization()(x)                                      ❼
    return x
```

❶ 计算输入特征图（通道）的数量

❷ 计算每个组的输入和输出滤波器（通道）数量。注意向上取整。

❸ 在每个通道组上执行 1 × 1 线性点卷积

❹ 沿通道组切片特征图

❺ 在列表中保持组点卷积。

❻ 将组点卷积的输出连接在一起

❼ 对连接的组输出（特征图）进行批量归一化

步长洗牌块

步长洗牌块与以下不同：

+   短路连接（块输入）的维度通过 3 × 3 平均池化操作减少。

+   在非步长洗牌块中，使用矩阵加法而不是连接残差和快捷特征图。

关于使用连接，作者推理出“用通道连接替换逐元素加法，这样可以在很少的计算成本下轻松扩大通道维度。”

图 8.28 描述了一个步长洗牌块。你可以看到与非步长洗牌块的两个不同之处。在快捷连接上添加了一个平均池化，通过将特征图减少到 0.5*H* × 0.5*W* 来进行维度减少。这是为了匹配步长 3 × 3 深度卷积所做的特征池化大小，这样它们就可以连接在一起——而不是在非步长洗牌块中的矩阵加法。

![](img/CH08_F28_Ferlitsch.png)

图 8.28 步长洗牌块

以下是一个步长洗牌块的示例实现。参数 `n_filters` 是块中卷积层的滤波器数量。参数 `reduction` 是用于进一步细化网络的元参数，参数 `n_partitions` 指定了将特征图划分成多少组进行点卷积组。

函数首先创建投影快捷连接。输入通过一个步长的 `AveragePooling2D` 层，将投影快捷连接中的特征图大小减少到 0.5*H* × 0.5*W*。

输入随后通过 1 × 1 点卷积组卷积(`pw_ group_conv()`)。请注意，网络细化发生在第一个点卷积组卷积(`int(reduction` `*` `n_filters)`)。输入经过通道混洗(`channel_ shuffle()`),然后通过 3 × 3 步长深度卷积，进行特征提取和特征池化；注意这里没有 ReLU 激活。

`DepthwiseConv2D()`的输出随后通过第二个 1 × 1 点卷积组卷积，其输出随后与投影快捷连接。

```
def strided_shuffle_block(x, n_partitions, n_filters, reduction):
    ''' Construct a Strided Shuffle Block
        x           : input to the block
        n_partitions: number of groups to partition feature maps (channels) 
        ➥ into.
        n_filters   : number of filters
        reduction   : dimensionality reduction factor (e.g, 0.25)
    '''
    # projection shortcut
    shortcut = x
    shortcut = AveragePooling2D((3, 3), strides=2, padding='same')(shortcut) ❶
    n_filters -= int(x.shape[-1])                                           ❷

    x = pw_group_conv(x, n_partitions, int(reduction * n_filters))
    x = ReLU()(x)

    x = channel_shuffle(x, n_partitions)

    x = DepthwiseConv2D((3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = pw_group_conv(x, n_partitions, n_filters)

    x = Concatenate()([shortcut, x])                                        ❸
    x = ReLU()(x)
    return x
```

❶ 使用平均池化进行瓶颈快捷连接

❷ 在第一个块中，入口点卷积组卷积的输出滤波器数量调整为与出口点卷积组卷积匹配。

❸ 将投影快捷连接到块的输出

通道混洗

*通道混洗*被设计用来克服组卷积的副作用，从而帮助信息在输出通道中流动。组卷积通过确保每个卷积只操作对应的输入通道组，显著降低了计算成本。正如作者所指出的，如果将多个组卷积堆叠在一起，会产生一个副作用：某些通道的输出仅来自输入通道的一小部分。换句话说，每个组卷积仅限于根据单个特征图（通道）学习其滤波器的下一个特征提取级别，而不是所有或部分输入特征图。

图 8.29 展示了将通道分成组并随后混洗通道的过程。本质上，混洗是通过构建新的通道来实现的，因为每个混洗通道都包含来自其他每个通道的部分——从而增加了输出通道之间的信息流。

![](img/CH08_F29_Ferlitsch.png)

图 8.29 通道混洗

让我们更仔细地看看这个过程。我们从一个输入通道组开始，我在图中用灰色阴影表示，以表明它们是不同的通道（不是副本）。接下来，根据分区设置，通道被分成相等大小的分区，我们称之为*组*。在我们的表示中，每个组有三个独立的通道。我们构建了三个通道的混洗版本。通过灰色阴影，我们表示每个混洗通道是由每个未混洗通道的部分组成的，并且每个混洗通道的部分是不同的。

例如，第一个混洗通道是由三个未混洗通道的特征图的前三分之一构建的。第二个混洗通道是由三个未混洗通道的特征图的前三分之一构建的，以此类推。

以下是一个通道洗牌的示例实现。参数 `n_partitions` 指定了将输入特征图 `x` 分成多少组。我们使用输入的形状来确定 *B* × *H* × *W* × *C*（其中 *C* 是通道），然后计算每个组的通道数（`grp_in_channels`）。

接下来的三个 Lambda 操作执行以下操作：

1.  将输入从 *B* × *H* × *W* × *C* 重塑为 *B* × *W* × *W* × *G* × *Cg*。添加了一个第五维 *G*（组），并将 *C* 重塑为 *G* × *Cg*，其中 *Cg* 是每个组中通道的子集。

1.  `k.permute_dimensions()` 执行了图 5.27 中展示的通道洗牌操作。

1.  第二次重塑将洗牌后的通道重新构建为形状 *B* × *H* × *W* × *C*。

```
def channel_shuffle(x, n_partitions):
    ''' Implements the channel shuffle layer
        x            : input tensor
        n_partitions : number of groups to partition feature maps (channels) into.
    '''
    batch, height, width, n_channels = x.shape                              ❶

    grp_in_channels  = n_channels // n_partitions                           ❷

    x = Lambda(lambda z: K.reshape(z, [-1, height, width, n_partitions, 
                                   grp_in_channels]))(x)                    ❸

    x = Lambda(lambda z: K.permute_dimensions(z, (0, 1, 2, 4, 3)))(x)       ❹

    x = Lambda(lambda z: K.reshape(z, [-1, height, width, n_channels]))(x)  ❺
    return x
```

❶ 获取输入张量的维度

❷ 推导每个组中输入滤波器（通道）的数量

❸ 分离通道组

❹ 交换通道组的顺序（洗牌）（即，3, 4 => 4, 3）

❺ 恢复输出形状

在他们的消融研究中，作者发现，在复杂性和准确性之间的最佳权衡是在减少因子为 1（无减少）的情况下，并将组分区数量设置为 8。使用 Idiomatic 过程重用设计模式为 ShuffleNet 编写的完整代码可以在 GitHub 上找到（[`mng.bz/oGop`](http://mng.bz/oGop)）。

接下来，我们将介绍如何使用量化缩小内存受限设备的模型大小，并使用 TensorFlow Lite Python 包进行转换/预测，以部署移动模型。

## 8.5 部署

我们将通过介绍部署移动卷积模型的基本知识来结束本章。我们首先将探讨量化，它减少了参数大小，从而降低了内存占用。量化在部署模型之前进行。接下来，我们将了解如何使用 TF Lite 在内存受限的设备上执行模型。在我们的示例中，我们使用 Python 环境作为代理。我们不会深入探讨与 Android 或 iOS 相关的具体细节。

### 8.5.1 量化

*量化* 是一个减少表示数字的位数的过程。对于内存受限的设备，我们希望在不会显著损失准确性的情况下，以较低的位表示存储权重。

由于神经网络对计算中的小错误具有一定的鲁棒性，因此在推理时不需要像训练时那样高的精度。这为在移动神经网络中降低权重的精度提供了机会。传统的减少方法是将 32 位浮点权重值替换为 8 位整数的离散近似。主要优势是，从 32 位到 8 位的减少只需要模型四分之一的内存空间。

在推理（预测）过程中，权重被缩放回其大约 32 位浮点值，以便进行矩阵运算，然后通过激活函数。现代硬件加速器已被设计用于优化此缩放操作，以实现名义上的计算开销。

在传统的缩减中，32 位浮点权重被分成整数范围内的桶（桶）。对于 8 位值，这将会有 256 个桶，如图 8.30 所示。

![](img/CH08_F30_Ferlitsch.png)

图 8.30 量化将浮点数范围分类为一系列由整数类型表示的固定桶。

在此示例中执行量化时，首先确定权重的浮点数范围，我们将其称为`[rmin, rmax]`，即最小值和最大值。然后，该范围按桶的数量（在 8 位整数的情况下为 256）进行线性划分。

根据硬件加速器，我们可能还会在 CPU（和 TPU）上看到从两次到三次的执行速度提升。GPU 不支持整数运算。

对于原生支持 float16（半精度）的 GPU，量化是通过将 float32 值转换为 float16 来完成的。这将模型的内存占用减半，并且通常将执行速度提高四倍。

此外，当权重的浮点数范围受到限制（缩小）时，量化效果最佳。对于移动模型，当前惯例是使用 ReLU 的`max_value`为 6.0。

我们应该小心量化非常小的模型。大型模型受益于权重的冗余，并且在量化为 8 位整数时对精度损失具有免疫力。最先进的移动模型已被设计为在量化时限制精度损失的数量。如果我们设计较小的模型并对其进行量化，它们在精度上可能会显著下降。

接下来，我们将介绍 TF Lite 在内存受限设备上执行模型。

### 8.5.2 TF Lite 转换和预测

TF Lite 是内存受限设备上 TensorFlow 模型的执行环境。与原生的 TensorFlow 运行时环境不同，TF Lite 运行时环境要小得多，更容易适应内存受限设备。虽然针对此目的进行了优化，但它也带来了一些权衡。例如，一些 TF 图操作不受支持，一些操作需要额外的步骤。我们不会涵盖不受支持的图操作，但我们会介绍所需的额外步骤。

以下代码演示了使用 TensorFlow Lite 对现有模型进行量化，其中模型是训练好的 TF.Keras 模型。第一步是将 SavedModel 格式的模型转换为 TF Lite 模型格式。这是通过实例化一个`TFLiteConverter`并将内存中或磁盘上的 SavedModel 格式模型传递给它来完成的，然后调用`convert()`方法：

```
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(model)     ❶

tflite_model = converter.convert()                              ❷
```

❶ 为 TF.Keras（SavedModel 格式）模型创建转换器实例

❷ 将模型转换为 TF Lite 格式

该模型的 TF Lite 版本不是 TensorFlow SavedModel 格式。您不能直接使用 `predict()` 等方法。相反，我们使用 TF Lite 解释器。您必须首先按照以下方式设置 TF Lite 模型的解释器：

1.  为 TF Lite 模型实例化一个 TF Lite 解释器。

1.  指示解释器为模型分配输入和输出张量。

1.  获取模型输入和输出张量的详细信息，这些信息将需要在预测时了解。

以下代码演示了这些步骤：

```
interpreter = tf.lite.Interpreter(model_content=tflite_model)    ❶
interpreter.allocate_tensors()                                   ❷

input_details = interpreter.get_input_details()                  ❸
output_details = interpreter.get_output_details()                ❸
input_shape = input_details[0]['shape']
```

❶ 实例化 TF Lite 模型的解释器

❷ 为模型分配输入和输出张量

❸ 获取预测所需的输入和输出张量详情

`input_details` 和 `output_details` 作为列表返回；元素的数量分别对应输入和输出张量的数量。例如，具有单个输入（例如图像）和单个输出（多类分类器）的模型将分别为输入和输出张量有一个元素。

每个元素都包含一个包含相应详细信息的字典。在输入张量的情况下，键 `shape` 返回一个元组，表示输入的形状。例如，如果模型以 (32, 32, 3) 的图像（例如 CIFAR-10）作为输入，则键将返回 (32, 32, 3)。

要进行单个预测，我们执行以下操作：

1.  准备输入以形成一个大小为 1 的批次。对于我们的 CIFAR-10 示例，这将是一个 (1, 32, 32, 3)。

1.  将批次分配给输入张量。

1.  调用解释器以执行预测。

1.  从模型获取输出张量（例如，多类模型中的 softmax 输出）。

以下代码演示了这些步骤：

```
import numpy as np

data = np.expand_dims(x_test[1], axis=0)                          ❶

interpreter.set_tensor(input_details[0]['index'], data)           ❷

interpreter.invoke()                                              ❸

softmax = interpreter.get_tensor(output_details[0]['index'])      ❹

label = np.argmax(softmax)                                        ❺
```

❶ 将单个输入转换为大小为 1 的批次

❷ 将批次分配给输入张量

❸ 执行（调用）解释器以执行预测

❹ 从模型获取输出

❺ 多类示例，确定从 softmax 输出预测的标签

对于批量预测，我们需要修改（调整大小）解释器的输入和输出张量以适应批次大小。以下代码在分配张量之前将解释器的批次大小调整为 128，对于 (32, 32, 3) 的输入（CIFAR-10）：

```
interpreter = tf.lite.Interpreter(model_content=tflite_model)              ❶

interpreter.resize_tensor_input(input_details[0]['index'], (128, 32, 32, 3))    
interpreter.resize_tensor_input(output_details[0]['index'], (128, 10))     ❷

interpreter.allocate_tensors()                                             ❸
```

❶ 实例化 TF Lite 模型的解释器

❷ 对 128 批次的输入和输出张量进行调整大小

❸ 为模型分配输入和输出张量

## 摘要

+   使用深度卷积和网络细化在 MobileNet v1 中的重构展示了在内存受限设备上运行模型并达到 AlexNet 准确率的能力。

+   将 MobileNet v2 中的残差块重新设计为倒残差块进一步减少了内存占用并提高了准确率。

+   SqueezeNet 引入了使用元参数配置组和块属性的计算效率宏架构搜索的概念。

+   ShuffleNet v1 中的重构和通道洗牌展示了在极端内存受限的设备上运行模型的能力，例如微控制器。

+   量化技术提供了一种方法，通过减少内存占用，可以将内存占用降低 75%，同时几乎不会损失推理精度。

+   使用 TF Lite 将 SavedModel 格式转换为量化 TF Lite 格式，并在内存受限的设备上进行预测部署。
