# 3 卷积和残差神经网络

本章涵盖了

+   理解卷积神经网络的结构

+   构建 ConvNet 模型

+   设计和构建 VGG 模型

+   设计和构建残差网络模型

第二章介绍了深度神经网络（DNN）背后的基本原理，这是一种基于密集层的网络架构。我们还演示了如何使用密集层制作一个简单的图像分类器，并讨论了尝试将 DNN 扩展到更大图像尺寸时的局限性。使用卷积层进行特征提取和学习的神经网络，称为“卷积神经网络”（*CNN*），使得将图像分类器扩展到实际应用成为可能。

本章涵盖了早期 SOTA 卷积神经网络的架构设计模式和设计模式的发展。本章按其演变的顺序，介绍了三个设计模式：

+   ConvNet

+   VGG

+   残差网络

这些设计模式中的每一个都对当今现代 CNN 设计做出了贡献。ConvNet，以 AlexNet 作为早期例子，引入了通过池化交替进行特征提取和降维的模式，并随着层深度的增加而逐次增加滤波器的数量。VGG 将卷积分组到由一个或多个卷积组成的块中，并在块的末尾延迟降维操作。残差网络进一步将分组块分组，将降维延迟到组末，并使用特征池化以及池化进行降维，以及分支路径——即恒等链接——的概念，用于块之间的特征重用。

## 3.1 卷积神经网络

早期卷积神经网络是一种可以看作由两部分组成的神经网络，即前端和后端。后端是一个深度神经网络（DNN），我们之前已经讨论过。名称“卷积神经网络”来源于前端，称为“卷积层”。前端充当预处理器的角色。DNN 后端执行“分类学习”。CNN 前端将图像数据预处理成 DNN 可以学习的计算上实用的形式。CNN 前端执行“特征学习”。

图 3.1 描述了一个 CNN，其中卷积层作为前端从图像中学习特征，然后将这些特征传递给后端 DNN 进行基于特征的分类。

![图片](img/CH03_F01_Ferlitsch.png)

图 3.1 卷积作为前端，用于从图像中学习特征，然后将这些特征传递给后端 DNN 进行分类。

本节涵盖了组装这些早期卷积神经网络的基本步骤和组件。虽然我们没有特别介绍 AlexNet，但其作为 2012 年 ILSVRC 图像分类冠军的成功，可以被视为研究人员探索和发展卷积设计的催化剂。AlexNet 前端的组装和设计原则被纳入了最早的设计模式，即 ConvNet，以供实际应用。

### 3.1.1 为什么我们使用 CNN 而不是 DNN 进行图像模型

一旦我们处理到更大的图像尺寸，对于深度神经网络（DNN）来说，像素的数量在计算上变得过于昂贵，以至于无法实现。假设你有一个 1MB 的图像，其中每个像素由一个字节（0..255 值）表示。在 1MB 中，你有 100 万个像素。这将需要一个包含 100 万个元素的输入向量。假设输入层有 1024 个节点。仅在输入层就需要更新和学习的权重数量就超过 10 亿（100 万 × 1024）！天哪。回到超级计算机和一生的计算能力。

让我们将其与我们第二章的 MNIST 示例进行对比，输入层有 784 像素 × 512 节点。这意味着有 40 万个权重需要学习，这比 10 亿小得多。你可以在你的笔记本电脑上完成前者，但不要尝试后者。

在接下来的小节中，我们将探讨 CNN 网络组件是如何解决原本可能是一个计算上不切实际的权重数量（也称为参数）的问题，这对于图像分类来说。

### 3.1.2 下采样（调整大小）

为了解决参数过多的问题，一种方法是通过称为下采样的过程降低图像的分辨率。但如果我们过度降低图像分辨率，在某个点上，我们可能失去清晰区分图像中内容的能力；它变得模糊，或者有伪影。因此，第一步是将分辨率降低到我们仍然有足够细节的水平。

对于日常计算机视觉来说，一个常见的约定是 224 × 224 像素。我们通过调整大小来实现这一点。即使在这个较低的分辨率和三个通道的颜色图像，以及 1024 节点的输入层，我们仍然有 1.54 亿个权重需要更新和学习（224 × 224 × 3 × 1024）；参见图 3.2。

![图片](img/CH03_F02_Ferlitsch.png)

图 3.2 输入层在调整大小前后的参数数量（图片来源：Pixabay，Stockvault）

因此，在引入使用卷积层之前，使用神经网络在真实世界图像上进行训练是不可能的。首先，卷积层是神经网络的前端，它将图像从基于高维像素的图像转换为基于显著低维度的特征图像。这些显著低维度的特征可以成为 DNN 的输入向量。因此，卷积前端是图像数据和 DNN 之间的前端。

但假设我们拥有足够的计算能力，只使用深度神经网络（DNN）并在输入层学习 1.54 亿个权重，就像我们前面的例子一样。嗯，像素在输入层的位置非常依赖。所以，我们学会了识别图片左边的猫。但然后我们把猫移到图片中间。现在我们必须学会从一组新的像素位置识别猫——哇！现在把它移到右边，加上躺着的猫、在空中跳跃等等。

从各种角度学习识别图像被称为*平移不变性*。对于基本的二维渲染，如数字和字母，这是可行的（暴力法），但对于其他所有东西，这都不行。早期的研究表明，当你将初始图像展平成 1D 向量时，你失去了构成被分类对象的特征的空间关系，比如猫。即使你成功地训练了一个 DNN，比如基于像素在图片中间识别猫，那么如果这个对象在图像中移动了位置，这个 DNN 不太可能识别出该对象。

接下来，我们将讨论卷积如何学习特征而不是像素，同时保留二维形状以进行空间关系，从而解决了这个问题。

### 3.1.3 特征检测

对于这些更高分辨率和更复杂的图像，我们通过检测和分类*特征*来进行识别，而不是对像素位置进行分类。可视化一张图像，问问自己是什么让你识别出那里的东西？超越问“那是一个人、一只猫还是一栋建筑？”这样的高级问题，去问为什么你能区分站在建筑物前面的人，或者从他们手中分离出一只猫。你的眼睛正在识别低级特征，如边缘、模糊和对比度。

如图 3.3 所示，这些低级特征被构建成轮廓，然后是空间关系。突然之间，眼睛/大脑有了识别鼻子、耳朵、眼睛的能力——感知到那是一只猫脸，或者那是一个人脸。

![图片](img/CH03_F03_Ferlitsch.png)

图 3.3 人眼识别低级特征到高级特征的流程

在计算机中，*卷积层*负责在图像中进行特征检测。每个卷积由一组过滤器组成。这些过滤器是*N* × *M*的值矩阵，用于检测特征可能存在的情况。把它们想象成小窗口。它们在图像上滑动，并在每个位置，将过滤器与该位置的像素值进行比较。这种比较是通过矩阵点积完成的，但在这里我们将跳过统计。重要的是，这个操作的结果将生成一个值，表示在图像的该位置检测到特征的程度有多强。例如，4 的值表示特征的检测比 1 的值更强。

在神经网络之前，成像科学家手动设计这些过滤器。今天，过滤器以及神经网络中的权重都是*学习*得到的。在卷积层中，我们指定过滤器的尺寸和过滤器的数量。典型的过滤器尺寸是 3 × 3 和 5 × 5，其中 3 × 3 是最常见的。过滤器的数量变化更多，但它们通常是 16 的倍数，例如浅层 CNN 的 16、32 或 64，以及深层 CNN 的 256、512 和 1024。

此外，我们指定一个*步长*，这是过滤器在图像上滑动的速率。例如，如果步长是 1，则过滤器每次前进 1 个像素；因此，过滤器在 3 × 3 的过滤器中会部分重叠前一步（并且因此步长为 2 的过滤器也是如此）。步长为 3 没有重叠。最常见的方法是使用步长 1 和 2。每个*学习*到的过滤器都会产生一个特征图，这是一个映射，表示在图像的特定位置检测到特征的程度，如图 3.4 所示。

![](img/CH03_F04_Ferlitsch.png)

图 3.4 过滤器在图像上滑动以产生检测到的特征的特征图。

过滤器可以在到达图像边缘时停止，或者继续直到覆盖最后一列，如图 3.5 所示。前者称为*无填充*。后者称为*填充*。当过滤器部分超出边缘时，我们想要为这些虚拟像素提供一个值。典型值是零或相同——与最后一列相同。

![](img/CH03_F05_Ferlitsch.png)

图 3.5 过滤器停止的位置取决于填充。

当你有多个卷积层时，一个常见的做法是在深层层中保持相同的过滤器数量或增加过滤器数量，并在第一层使用步长 1，在深层层使用步长 2。过滤器数量的增加提供了从粗略检测特征到在粗略特征内进行更详细检测的手段。步长的增加抵消了保留数据大小的增加；这个过程被称为*特征池化*，其中特征图被下采样。

CNNs 使用两种类型的下采样：池化和特征池化。在*池化*中，使用一个固定的算法来下采样图像数据的大小。在*特征池化*中，学习特定数据集的最佳下采样算法：

更多的过滤器 => 更多的数据

更大的步长 => 更少的数据

我们将在下一节更详细地研究池化。我们将在 3.2 节深入研究特征池化。

### 3.1.4 池化

尽管生成的每个特征图通常与图像大小相等或更小，但由于我们生成了多个特征图（例如，16 个），总数据量会增加。哎呀！下一步是减少总数据量，同时保留检测到的特征及其对应的空间关系。

正如我所说的，这一步被称为*池化*，这与*下采样*（或*子采样*）相同。在这个过程中，特征图通过在特征图内部使用最大值（下采样）或平均像素平均值（子采样）调整到更小的维度。在池化中，如图 3.6 所示，我们将要池化的区域大小设置为*N* × *M*矩阵以及步长。常见的做法是 2 × 2 的池化大小和 2 的步长。这将导致像素数据的 75%减少，同时仍然保留足够的分辨率，以确保检测到的特征不会丢失。

![图片](img/CH03_F06_Ferlitsch.png)

图 3.6 池化将特征图调整到更小的维度。

另一种看待池化的方式是在信息增益的背景下。通过减少不需要或不那么有信息的像素（例如，背景中的像素），我们正在减少熵，并使剩余的像素更有信息量。

### 3.1.5 展平

记住，深度神经网络以向量作为输入——数字的一维数组。在池化图的情况下，我们有一个 2D 矩阵的列表（复数），因此我们需要将它们转换成一个单一的 1D 向量，然后它成为 DNN 的输入向量。这个过程被称为*展平*：我们将 2D 矩阵的列表展平成一个单一的 1D 向量。

这相当直接。我们以第一个池化图的第一行为 1D 向量的开始。然后我们取第二行并将其附加到末尾，接着是第三行，以此类推。然后我们继续到第二个池化图，并执行相同的操作，持续地将每一行附加到我们完成最后一个池化图。只要我们通过池化图遵循相同的顺序，检测到的特征之间的空间关系将在训练和推理（预测）过程中保持一致，如图 3.7 所示。

例如，如果我们有 16 个大小为 20 × 20 的池化图，每个池化图有 3 个通道（例如，彩色图像中的 RGB 通道），我们的 1D 向量大小将是 16 × 20 × 20 × 3 = 19,200 个元素。

![图片](img/CH03_F07_Ferlitsch.png)

图 3.7 当池化图被展平时，空间关系得以保持。

## 3.2 CNN 的 ConvNet 设计

现在我们开始使用 TF.Keras。让我们假设一个假设但与现实世界相似的情况。贵公司的应用程序支持人机界面，并且目前可以通过语音激活访问。你被分配了一个开发概念证明的任务，以展示将手语纳入人机界面，以符合联邦无障碍法律。相关的法律，1973 年康复法案的第五百零三部分，“禁止联邦承包商和分包商在就业中歧视残疾人，并要求雇主采取积极行动招募、雇佣、晋升和留住这些个人”([`www.dol.gov/agencies/ofccp/section-503`](https://www.dol.gov/agencies/ofccp/section-503))。

你不应该假设你可以通过使用任意标记的手语图像和图像增强来训练模型。数据、其准备和模型的设计必须与实际的“野外”部署相匹配。否则，除了导致令人失望的准确率外，模型可能会学习噪声，使其暴露于可能导致意外后果的假阳性，并且容易受到黑客攻击。第十二章将更详细地介绍这一点。

对于我们的概念验证，我们将仅展示识别英文字母（从 A 到 Z）的手势。此外，我们假设个人将直接在摄像头前从正面的角度进行手势。我们不希望模型学习，例如，手势者的种族。因此，出于这个和其他原因，颜色并不重要。

为了让我们的模型不学习颜色（噪声），我们将以灰度模式对其进行训练。我们将设计模型以在灰度下学习和预测，这个过程也被称为*推理*。我们希望模型学习的是手的轮廓。我们将设计模型为两部分，即卷积前端和 DNN 后端，如图 3.8 所示。

![图片](img/CH03_F08_Ferlitsch.png)

图 3.8 带有卷积前端和 DNN 后端的 ConvNet

以下代码示例是用顺序 API 方法编写的，并且是长格式；激活函数是通过相应的方法指定的（而不是在添加相应层时将其作为参数指定）。

我们首先通过使用`Conv2D`类对象添加一个 16 个滤波器的卷积层作为第一层。回想一下，滤波器的数量等于将要生成的特征图的数量（在这种情况下，16）。每个滤波器的大小将是 3 × 3，这是通过`kernel_size`参数指定的，步长为 2，由`strides`参数指定。

注意，对于`strides`，指定了一个`(2, 2)`的元组而不是单个值 2。第一个数字是水平步长（横跨），第二个数字是垂直步长（向下）。这些水平和垂直值通常是相同的，因此我们通常说“步长为 2”而不是“2 × 2 步长”。

你可能会问，`Conv2D`这个名字中的 2D 部分是什么意思？2D 表示卷积层的输入将是一堆矩阵（二维数组）。对于本章，我们将坚持使用 2D 卷积，这是计算机视觉中的常见做法。

让我们计算从这个层输出的尺寸将会是多少。如您所回忆的，在步长为 1 的情况下，每个输出特征图的大小将与图像相同。有 16 个滤波器，那将是输入的 16 倍。但由于我们使用了步长为 2（特征池化），每个特征图将减少 75%，因此总输出大小将是输入的 4 倍。

卷积层的输出随后通过 ReLU 激活函数，然后传递给最大池化层，使用`MaxPool2D`类对象。池化区域的大小将是 2 × 2，由参数`pool_size`指定，步长为 2，由参数`strides`指定。池化层将特征图减少 75%到池化特征图。

让我们计算池化层之后的输出大小。我们知道输入的大小是输入大小的 4 倍。再额外减少 75%，输出大小与输入相同。那么我们在这里得到了什么？首先，我们训练了一组滤波器来学习第一组粗糙特征（从而获得信息增益），消除了非必要的像素信息（减少熵），并学会了下采样特征图的最佳方法。嗯，看起来我们得到了很多。

然后将池化特征图展平，使用`Flatten`类对象，形成一个 1D 向量，用于输入到深度神经网络（DNN）。我们将简要介绍一下参数`padding`。对于我们的目的来说，可以说在几乎所有情况下，你都会使用值`same`；只是默认值是`valid`，因此你需要明确地添加它。

最后，我们为我们的图像选择一个输入大小。我们希望尽可能减小大小，同时不丢失识别手部轮廓所需的特征检测。在这种情况下，我们选择 128 × 128。`Conv2D`类有一个特性：它总是要求指定通道数，而不是默认为灰度图的 1；因此我们将其指定为(128, 128, 1)而不是(128, 128)。

下面是代码：

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding="same", 
                 input_shape=(128, 128, 1)))                  ❶
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))     ❷
model.add(Flatten())                                          ❸

model.add(Dense(512))
model.add(ReLU())
model.add(Dense(26))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
```

❶ 图像数据输入到一个卷积层。

❷ 通过池化减少了特征图的大小。

❸ 在输出层之前，2D 特征图被展平成一个 1D 向量。

通过使用`summary()`方法，让我们来看看我们模型中各层的详细信息：

```
model.summary()
Layer (type)                     Output Shape            Param # 
=================================================================
conv2d_1 (Conv2D)                (None, 64, 64, 16)      160         ❶
_________________________________________________________________
re_lu_1 (ReLU)                   (None, 64, 64, 16)      0  
_________________________________________________________________
max_pooling2d_1 (MaxPooling2     (None, 32, 32, 16)      0           ❷
_________________________________________________________________
flatten_1 (Flatten)              (None, 16384)           0
_________________________________________________________________
dense_1 (Dense)                  (None, 512)             8389120     ❸
_________________________________________________________________
re_lu_2 (ReLU)                   (None, 512)             0     
_________________________________________________________________
dense_2 (Dense)                  (None, 26)              13338       ❹
_________________________________________________________________
activation_1 (Activation)        (None, 26)              0     
=================================================================
Total params: 8,402,618
Trainable params: 8,402,618
Non-trainable params: 0
```

❶ 卷积层的输出是 16 个 2D 大小为 64 × 64 的特征图。

❷ 池化层的输出将特征图大小减少到 32 × 32。

❸ 512 节点密集层的参数数量超过 800 万；展平层中的每个节点都与密集层中的每个节点相连。

❹ 最终的密集层有 26 个节点，每个节点对应英文字母表中的一个字母。

下面是如何读取输出形状列。对于`Conv2D`输入层，输出形状显示为（None, 64, 64, 16）。元组中的第一个值是单个前向传递中将通过的示例（批大小）数量。由于这是在训练时确定的，因此设置为`None`以表示当模型被喂数据时将绑定。最后一个数字是过滤器的数量，我们将其设置为 16。中间的两个数字（64, 64）是特征图的输出大小——在这种情况下，每个为 64 × 64 像素（总共 16）。输出大小由过滤器大小（3 × 3）、步长（2 × 2）和填充（same）决定。我们指定的组合将使高度和宽度减半，总大小减少 75%。

对于`MaxPooling2D`层，池化特征图的输出大小将是 32 × 32。通过指定 2 × 2 的池化区域和 2 的步长，池化特征图的高度和宽度将减半，总大小减少 75%。

从池化特征图得到的展平输出是一个大小为 16,384 的 1D 向量，计算方式为 16 × (32 × 32)。让我们看看这加起来是否等于我们之前计算的，即特征图的输出大小应该与输入大小相同。我们的输入是 128 × 128，也就是 16,384，这与`Flatten`层的输出大小相匹配。

展平后的池化特征图中的每个元素（像素）随后被输入到 DNN 输入层的每个节点中，该层有 512 个节点。因此，展平层和输入层之间的连接数是 16,384 × 512 = ~8.4 百万。这就是该层需要学习的权重数量，并且大部分计算将（压倒性地）发生在这里。

现在我们以序列方法风格的变体来展示相同的代码示例。在这里，激活方法是通过在每个层的实例化中使用参数`activation`来指定的（例如`Conv2D(), Dense()`）：

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding="same", 
                 activation='relu', input_shape=(128,128, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(26, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
```

现在我们以第三种方式展示相同的代码示例，使用功能 API 方法。在这种方法中，我们分别定义每个层，从输入向量开始，到输出层结束。在每一层，我们使用多态来调用实例化的类（层）对象作为可调用对象，并传入前一个层的对象来连接它。

例如，对于第一个`Dense`层，当作为可调用对象调用时，我们将`Flatten`层的层对象作为参数传递。作为一个可调用对象，这将导致`Flatten`层和第一个`Dense`层完全连接（`Flatten`层的每个节点将连接到`Dense`层的每个节点）：

```
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

inputs  = Input(shape=(128, 128, 1))                                       ❶
layer   = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding="same",
                 activation='relu')(inputs)                                ❷
layer   = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)            ❸
layer   = Flatten()(layer)
layer   = Dense(512, activation='relu')(layer)
outputs = Dense(26, activation='softmax')(layer)

model = Model(inputs, outputs)
```

❶ 对于卷积层，需要指定通道数。

❷ 构建卷积层

❸ 通过池化减少特征图的大小

## 3.3 VGG 网络

*VGG* 类型的卷积神经网络是由牛津大学的视觉几何组设计的。它是为了在国际 ILSVRC 图像识别竞赛中竞争 1000 类图像而设计的。2014 年的 VGGNet 在图像定位任务中获得了第一名，在图像分类任务中获得了第二名。

虽然 AlexNet（及其相应的卷积网络设计模式）被认为是卷积网络的鼻祖，但 VGGNet（及其相应的 VGG 设计模式）被认为是基于卷积组正规化设计模式的之父。像它的 AlexNet 先辈一样，它继续将卷积层视为前端，并保留一个大的 DNN 后端用于分类任务。VGG 设计模式背后的基本原理如下：

+   将多个卷积分组为具有相同数量的过滤器的块

+   在块之间逐步加倍过滤器数量

+   将池化延迟到块的末尾

当在当今的背景下讨论 VGG 设计模式时，可能会对术语 *group* 和 *block* 产生初始混淆。在为 VGGNet 进行研究时，作者使用了术语 *卷积组*。随后，研究人员将分组模式细化成由卷积块组成的卷积组。在今天的命名法中，VGG 组会被称为 *块*。

它是使用一些易于学习的原则设计的。卷积前端由一系列相同大小的卷积对（后来是三对）组成，随后是最大池化。最大池化层将生成的特征图下采样 75%，然后下一对（或三对）卷积层将学习到的过滤器数量加倍。卷积设计背后的原理是，早期层学习粗略特征，后续层通过增加过滤器，学习越来越精细的特征，而最大池化用于层之间以最小化特征图大小的增长（以及随后学习的参数）。最后，深度神经网络（DNN）后端由两个大小相同、每个有 4096 个节点的密集隐藏层和一个用于分类的 1000 个节点的最终密集输出层组成。图 3.9 描述了 VGG 架构中的第一个卷积组。

![](img/CH03_F09_Ferlitsch.png)

图 3.9 在 VGG 架构中，卷积被分组，池化被延迟到组末尾。

最著名的版本是 VGG16 和 VGG19。在竞赛中使用的 VGG16 和 VGG19，以及它们的竞赛训练权重，都已公开发布。由于它们在迁移学习中经常被使用，其他人保留了 ImageNet 预训练的 VGG16 或 VGG19 的卷积前端和相应的权重，并附加了一个新的 DNN 后端，用于重新训练新的图像类别。图 3.10 是 VGG16 的架构描述。

![](img/CH03_F10_Ferlitsch.png)

图 3.10 VGG16 架构由 VGG 组的卷积前端组成，后面跟着 DNN 后端。

因此，让我们继续用两种编码风格来编写 VGG16：第一种是顺序流，第二种是使用*重用*函数来复制层的公共块，并指定它们特定设置的参数。我们还将更改指定`kernel_size`和`pool_size`的方式，将它们作为关键字参数指定，而不是位置参数：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu", input_shape=(224, 224, 3)))     ❶
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same",
                 activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2))) 

model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu"))                                ❷
model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same",
                 activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2))) 

model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu"))                                ❸
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu"))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2))) 

model.add(Conv2D(512, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu"))                                ❹
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu"))
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2))) 

model.add(Conv2D(512, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu"))                                ❺
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu"))
model.add(Conv2D(512, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu"))
model.add(MaxPooling2D((2, 2), strides=(2, 2))) 

model.add(Flatten())                                                ❻
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))

model.add(Dense(1000, activation='softmax'))                        ❼

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
```

❶ 第一个卷积块

❷ 第二个卷积块——过滤器数量加倍

❸ 第三个卷积块——过滤器数量加倍

❹ 第四个卷积块——过滤器数量加倍

❺ 第五（最终）个卷积块

❻ DNN 后端

❼ 用于分类的输出层（1000 个类别）

你刚刚编写了一个 VGG16——不错。现在让我们用过程重用风格来编写相同的代码。在这个例子中，我们创建了一个过程（函数）`conv_block``()`，它构建卷积块，并接受块中层数（2 或 3）和过滤器数量（64、128、256 或 512）作为参数。注意，我们将第一个卷积层放在`conv_block`之外。第一层需要`input_shape`参数。我们本可以将它编码为`conv_block`的标志，但由于它只会出现一次，这不是重用。所以我们将其内联：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def conv_block(n_layers, n_filters):                ❶
    """
        n_layers : number of convolutional layers
        n_filters: number of filters
    """
    for n in range(n_layers):
        model.add(Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
                         activation="relu"))
    model.add(MaxPooling2D(2, strides=2))

model = Sequential()      
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same", 
                 activation="relu", 
                 input_shape=(224, 224, 3)))        ❷
conv_block(1, 64)                                   ❸
conv_block(2, 128)                                  ❹
conv_block(3, 256)                                  ❹
conv_block(3, 512)                                  ❹
conv_block(3, 512)                                  ❹

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))

model.add(Dense(1000, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
```

❶ 卷积块作为过程实现

❷ 第一个卷积块单独指定，因为它需要`input_shape`参数

❸ 第一个卷积块的剩余部分

❹ 第二至第五个卷积块

尝试在两个示例上运行 `model.summary()`，你会发现输出是相同的。

## 3.4 ResNet 网络

微软研究院设计的*ResNet*类型 CNN 是为了在国际 ILSVRC 竞赛中竞争。2015 年的比赛中，ResNet 在 ImageNet 和 Common Objects in Context (COCO)竞赛的所有类别中都获得了第一名。

在上一节中介绍的 VGGNet 设计模式在模型架构的层数深度上存在局限性，在遭受梯度消失和梯度爆炸之前，模型架构的深度是有限的。此外，不同层以不同速率收敛可能导致训练过程中的发散。

对于残差网络中残差块设计模式组件的研究人员提出了一个他们称之为*恒等链接*的新颖层连接。恒等链接引入了特征重用的最早概念。在恒等链接之前，每个卷积块都对前一个卷积输出进行特征提取，而不保留任何先前输出的知识。恒等链接可以看作是当前和先前卷积输出之间的耦合，以重用从早期提取中获得的特征信息。同时，与 ResNet 一起，其他研究人员——例如谷歌的 Inception v1（GoogLeNet）——进一步将卷积设计模式细化成组和块。与这些设计改进并行的是批量归一化的引入。

使用身份链接以及批量归一化在层之间提供了更多的稳定性，减少了梯度消失和爆炸以及层间的发散，使得模型架构可以在层中更深，从而提高预测的准确性。

### 3.4.1 架构

ResNet 以及这个类别中的其他架构使用不同的层到层连接模式。我们之前讨论的模式（ConvNet 和 VGG）使用的是全连接层到层模式。

ResNet34 引入了新的块层和层连接模式，分别是残差块和恒等连接。ResNet34 中的残差块由两个没有池化层的相同卷积层组成。每个块都有一个恒等连接，它创建了一个在残差块的输入和输出之间的并行路径，如图 3.11 所示。与 VGG 一样，每个后续块将滤波器的数量加倍。在块序列的末尾进行池化。

![图像](img/CH03_F11_Ferlitsch.png)

图 3.11 一个残差块，其中输入是一个矩阵加到卷积的输出上

神经网络的一个问题是，当我们增加更深的层（在假设增加准确性的前提下），它们的性能可能会下降。情况可能会变得更糟，而不是更好。这有几个原因。当我们深入时，我们正在添加更多的参数（权重）。参数越多，每个训练数据中的输入适合过多参数的地方就越多。不是泛化，神经网络将简单地学习每个训练示例（死记硬背）。另一个问题是*协变量偏移*：随着我们深入，权重的分布会变宽（进一步分散），这使得神经网络收敛变得更加困难。前者导致测试（保留）数据上的性能下降，后者在训练数据上也是如此，以及梯度消失或爆炸。

残差块允许神经网络构建更深的层，而不会降低测试数据上的性能。一个 ResNet 块可以看作是一个添加了恒等连接的 VGG 块。虽然块的 VGG 风格执行特征检测，但恒等连接保留了输入给下一个后续块，因此下一个块的输入包括前一个特征检测和输入。

通过保留过去（前一个输入）的信息，这种块设计使得神经网络可以比 VGG 对应物更深，同时提高准确性。从数学上讲，我们可以将 VGG 和 ResNet 表示如下。对于这两种情况，我们希望学习一个*h*(*x*)的公式，它是测试数据的分布（例如，标签）。对于 VGG，我们学习一个函数*f(x*, {*W*})，其中{*W*}代表权重。对于 ResNet，我们通过添加“+ *x*”这一项来修改方程，其中*x*是恒等：

VGG: *h*(*x*) = *f*(*x*, {*W*})

ResNet: *h*(*x*) = *f*(*x*, {*W*}) + *x*

以下代码片段展示了如何通过使用功能 API 方法在 TF.Keras 中编码一个残差块。变量`x`代表一个层的输出，它是下一层的输入。在块的开始，我们保留前一个块/层的输出作为变量`shortcut`。然后我们将前一个块/层的输出(`x`)通过两个卷积层，每次都将前一个层的输出作为下一层的输入。最后，块的最后一个输出（保留在变量`x`中）与原始的`x`值（快捷方式）相加（矩阵加法）。这是恒等连接，通常被称为*快捷方式*：

```
shortcut = x                          ❶
x = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
x = layers.ReLU()(x)                  ❷
x = layers.add([shortcut, x])         ❸
```

❶ 记住块的输入。

❷ 卷积序列的输出

❸ 输入到输出的矩阵加法

现在我们使用过程式风格将整个网络组合起来。此外，我们还需要添加 ResNet 的入口卷积层，然后是 DNN 分类器。

正如我们在 VGG 示例中所做的那样，我们定义了一个生成残差块模式的程序（函数），遵循我们在前面的代码片段中使用的模式。对于我们的`residual_block``()`过程，我们传入块的滤波器数量和输入层（前一个层的输出）。

ResNet 架构将一个(224, 224, 3)向量作为输入——一个 224（高度）× 224（宽度）像素的 RGB 图像（3 个通道）。第一层是一个基本的卷积层，使用一个相当大的 7 × 7 的滤波器。然后通过一个最大池化层减小输出（特征图）的大小。

在初始卷积层之后是一系列残差块组。每个后续组将过滤器数量加倍（类似于 VGG）。然而，与 VGG 不同，组之间没有池化层来减少特征图的大小。现在，如果我们直接将这些块连接起来，我们会遇到问题。下一个块的输入形状基于前一个块的过滤器大小（让我们称它为*X*）。下一个块通过加倍过滤器，将导致该残差块的输出大小加倍（让我们称它为 2*X*）。恒等链接将尝试将输入矩阵(*X*)和输出矩阵(2*X*)相加。哎呀——我们得到一个错误，表示我们无法广播（对于加法操作）不同大小的矩阵。

对于 ResNet，这是通过在每个“加倍”的残差块组之间添加一个卷积块来解决的。如图 3.12 所示，卷积块将过滤器加倍以改变大小，并将步长加倍以将特征图大小减少 75%（执行特征池化）。

![图片](img/CH03_F12_Ferlitsch.png)

图 3.12 卷积块执行池化并将特征图数量加倍，为下一个卷积组做准备。

最后一个残差块组的输出传递到一个池化和展平层(`GlobalAveragePooling2D`)，然后传递到一个有 1000 个节点的单个`Dense`层（类别数量）：

```
from tensorflow.keras import Model
import tensorflow.keras.layers as layers

def residual_block(n_filters, x):                                           ❶
    """ Create a Residual Block of Convolutions
        n_filters: number of filters
        x        : input into the block
    """
    shortcut = x
    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", 
                      activation="relu")(x)
    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", 
                      activation="relu")(x)
    x = layers.add([shortcut, x])
    return x

def conv_block(n_filters, x):                                               ❷
    """ Create Block of Convolutions without Pooling
        n_filters: number of filters
        x        : input into the block
    """
    x = layers.Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same", 
                      activation="relu")(x)
    x = layers.Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same", 
                      activation="relu")(x)
    return x

inputs = layers.Input(shape=(224, 224, 3))                                  ❸

x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', 
                  activation='relu')(inputs)                                ❹
x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)   ❹

for _ in range(2):                                                          ❺
    x = residual_block(64, x)                                               ❺

x = conv_block(128, x)                                                      ❻

for _ in range(3):
    x = residual_block(128, x)

x = conv_block(256, x)

for _ in range(5):
    x = residual_block(256, x)

x = conv_block(512, x)

    x = residual_block(512, x)

x = layers.GlobalAveragePooling2D()(x)

outputs = layers.Dense(1000, activation='softmax')(x)

model = Model(inputs, outputs)
```

❶ 残差块作为过程

❷ 卷积块作为过程

❸ 输入张量

❹ 首个卷积层，其中池化特征图将减少 75%

❺ 64 个过滤器的第一个残差块组

❻ 将过滤器大小加倍并减少特征图 75%（步长 s = 2, 2）以适应下一个残差组

现在运行`model.summary()`。我们看到需要学习的总参数数量是 2100 万。这与拥有 1.38 亿参数的 VGG16 形成对比。所以 ResNet 架构在计算上快了六倍。这种减少主要是由残差块的结构实现的。注意，深度神经网络后端只是一个单独的输出`Dense`层。实际上，没有后端。早期的残差块组充当 CNN 前端进行特征检测，而后面的残差块执行分类。这样做时，与 VGG 不同，不需要几个全连接的密集层，这会大大增加参数数量。

与之前示例中的池化不同，在池化中每个特征图的尺寸根据步长的大小而减小，`GlobalAveragePooling2D`就像一个超级充电版的池化：每个特征图被一个单一值所替代，在这种情况下是相应特征图中所有值的平均值。例如，如果输入是 256 个特征图，输出将是一个大小为 256 的 1D 向量。在 ResNet 之后，使用`GlobalAveragePooling2D`在最后一个池化阶段成为深度卷积神经网络的通用实践，这显著减少了进入分类器的参数数量，而没有在表示能力上造成重大损失。

另一个优点是恒等连接，它提供了在不降低性能的情况下添加更深层的功能，以实现更高的准确率。

ResNet50 **引入**了一种称为*瓶颈残差块*的残差块变体。在这个版本中，两个 3 × 3 卷积层组被一组 1 × 1、然后 3 × 3、最后 1 × 1 卷积层组所取代。第一个 1 × 1 卷积执行降维操作，降低计算复杂度，最后一个卷积恢复维度，通过 4 倍增加滤波器的数量。中间的 3 × 3 卷积被称为*瓶颈卷积*，就像瓶子的颈部。如图 3.13 所示的瓶颈残差块允许构建更深层的神经网络，而不降低性能，并进一步降低计算复杂度。

![](img/CH03_F13_Ferlitsch.png)

图 3.13 瓶颈设计使用 1 × 1 卷积进行降维和扩展。

这里是一个将瓶颈残差块作为可重用函数编写的代码片段：

```
def bottleneck_block(n_filters, x):
    """ Create a Bottleneck Residual Block of Convolutions
        n_filters: number of filters
        x        : input into the block
    """
    shortcut = x
    x = layers.Conv2D(n_filters, (1, 1), strides=(1, 1), padding="same", 
                      activation="relu")(x)                                 ❶
    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", 
                      activation="relu")(x)                                 ❷
    x = layers.Conv2D(n_filters * 4, (1, 1), strides=(1, 1), padding="same",
                      activation="relu")(x)                                 ❸
    x = layers.add([shortcut, x])                                           ❹
    return x
```

❶ 用于降维的 1 × 1 瓶颈卷积

❷ 用于特征提取的 3 × 3 卷积

❸ 用于降维的 1 × 1 投影卷积

❹ 输入到输出的矩阵加法

残差块引入了表示能力和表示等价性的概念。*表示能力*是衡量一个块作为特征提取器强大程度的一个指标。*表示等价性*是指一个块可以被分解成具有更低计算复杂度的形式，同时保持其表示能力。残差瓶颈块的设计被证明可以保持 ResNet34 块的表示能力，同时降低计算复杂度。

### 3.4.2 批标准化

在神经网络中添加更深层的问题还包括*梯度消失*问题。这实际上是关于计算机硬件的。在训练过程中（反向传播和梯度下降的过程），在每一层，权重都会乘以非常小的数字——具体来说，是小于 1 的数字。正如你所知，两个小于 1 的数字相乘会得到一个更小的数字。当这些微小的值通过更深层传播时，它们会持续变小。在某个点上，计算机硬件无法再表示这个值——因此，出现了*梯度消失*。

如果我们尝试使用半精度浮点数（16 位浮点数）进行矩阵运算，而不是单精度浮点数（32 位浮点数），问题会进一步加剧。前者的优势在于权重（和数据）存储的空间减少了一半——按照一般经验，将计算大小减半，我们可以在每个计算周期内执行四倍的指令。当然，问题是，即使精度更小，我们也会更早地遇到*梯度消失*问题。

*批归一化*是一种应用于层输出（在激活函数之前或之后）的技术。不深入统计学方面，它在训练过程中对权重的偏移进行归一化。这有几个优点：它平滑了（在整个批次中）变化量，从而减缓了得到一个无法由硬件表示的极小数字的可能性。此外，通过缩小权重之间的偏移量，可以使用更高的学习率并减少总的训练时间，从而更快地收敛。在 TF.Keras 中，使用 `BatchNormalization` 类将批归一化添加到层中。

在早期的实现中，批归一化是在激活函数之后实现的。批归一化发生在卷积和密集层之后。当时，人们争论批归一化应该在激活函数之前还是之后。此代码示例在卷积和密集层中，在激活函数之前和之后都使用了后激活批归一化：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, Flatten
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', 
                 input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(ReLU())                  ❶

model.add(Flatten())

model.add(Dense(4096))
model.add(ReLU())

model.add(BatchNormalization())    ❷
```

❶ 在激活函数之前添加批归一化

❷ 在激活函数之后添加批归一化

### 3.4.3 ResNet50

*ResNet50* 是一个广为人知的模型，通常被用作通用模型，例如用于迁移学习、作为目标检测中的共享层，以及用于性能基准测试。该模型有三个版本：v1、v1.5 和 v2。

*ResNet50 v1* 正式化了*卷积组*的概念。这是一个共享相同配置（如滤波器数量）的卷积块集合。在 v1 中，神经网络被分解成组，每个组将前一个组的滤波器数量翻倍。

此外，移除了单独的卷积块以将滤波器数量加倍的概念，并替换为使用*线性投影*的残差块。每个组从使用线性投影在标识连接上进行的残差块开始，以加倍滤波器数量，而其余的残差块直接将输入传递到输出以进行矩阵加法操作。此外，具有线性投影的残差块中的第一个 1 × 1 卷积使用步长为 2（特征池化），这也称为*带步长的卷积*，如图 3.14 所示，减少了特征图大小 75%。

![图片](img/CH03_F14_Ferlitsch.png)

图 3.14 标识连接被替换为 1 × 1 投影以匹配卷积输出上的特征图数量，以便进行矩阵加法操作。

以下是对 ResNet50 v1 使用瓶颈块与批量归一化相结合的实现：

```
from tensorflow.keras import Model
import tensorflow.keras.layers as layers

def identity_block(x, n_filters):
    """ Create a Bottleneck Residual Block of Convolutions
        n_filters: number of filters
        x        : input into the block
    """
    shortcut = x

    x = layers.Conv2D(n_filters, (1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(n_filters * 4, (1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)

    return x

def projection_block(x, n_filters, strides=(2,2)):                        ❶
    """ Create Block of Convolutions with feature pooling
        Increase the number of filters by 4X
        x        : input into the block
        n_filters: number of filters
    """
    shortcut = layers.Conv2D(4 * n_filters, (1, 1), strides=strides)(x)   ❷
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(n_filters, (1, 1), strides=strides)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(4 * n_filters, (1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)

    return x

inputs = layers.Input(shape=(224, 224, 3))

x = layers.ZeroPadding2D(padding=(3, 3))(inputs)
x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.ZeroPadding2D(padding=(1, 1))(x)
x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

x = projection_block(64, x, strides=(1,1))                                ❸

for _ in range(2):
    x = identity_block(64, x)

x = projection_block(128, x)

for _ in range(3):
    x = identity_block(128, x)

x = projection_block(256, x)

for _ in range(5):
    x = identity_block(256, x)

x = projection_block(512, x)

for _ in range(2):
    x = identity_block(512, x)

x = layers.GlobalAveragePooling2D()(x)

outputs = layers.Dense(1000, activation='softmax')(x)

model = Model(inputs, outputs)
```

❶ 投影块作为过程

❷ 在快捷连接上进行 1 × 1 投影卷积以匹配输出大小

❸ 首个组之后的每个卷积组都以投影块开始。

如图 3.15 所示，v1.5 引入了对瓶颈设计的重构，进一步降低了计算复杂度，同时保持了表示能力。在具有线性投影的残差块中的特征池化（步长=2）从第一个 1 × 1 卷积移动到 3 × 3 卷积，降低了计算复杂度，并在 ImageNet 上提高了 0.5%的结果。

![图片](img/CH03_F15_Ferlitsch.png)

图 3.15 维度降低从 1 × 1 卷积移动到 3 × 3 卷积。

以下是对具有投影连接的 ResNet50 v1 残差块的实现：

```
def projection_block(x, n_filters, strides=(2,2)):
    """ Create Block of Convolutions with feature pooling
        Increase the number of filters by 4X
        x        : input into the block
        n_filters: number of filters
    """
    shortcut = layers.Conv2D(4 * n_filters, (1, 1), strides=strides)(x)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(n_filters, (1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same')(x) ❶
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(4 * n_filters, (1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x
```

❶ 使用步长为 2 将瓶颈移动到 3 × 3 卷积

*ResNet50 v2*引入了*预激活批量归一化*（*BN-RE-Conv*），其中批量归一化和激活函数被放置在相应的卷积或密集层之前（而不是之后）。现在这已成为一种常见做法，如图中所示，v2 中实现了具有标识连接的残差块：

```
def identity_block(x, n_filters):
      """ Create a Bottleneck Residual Block of Convolutions
            n_filters: number of filters
            x            : input into the block
      """
      shortcut = x

      x = layers.BatchNormalization()(x)                         ❶
      x = layers.ReLU()(x)                                       ❶
      x = layers.Conv2D(n_filters, (1, 1), strides=(1, 1))(x)

      x = layers.BatchNormalization()(x)                         ❶
      x = layers.ReLU()(x)
      x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same")(x)

      x = layers.BatchNormalization()(x)                         ❶
      x = layers.ReLU()(x)
      x = layers.Conv2D(n_filters * 4, (1, 1), strides=(1, 1))(x)

      x = layers.add([shortcut, x])
      return x
```

❶ 在卷积之前进行批量归一化

## 摘要

+   卷积神经网络可以描述为向深度神经网络添加前端。

+   CNN 前端的目的是将高维像素输入降低到低维特征表示。

+   特征表示的较低维度使得使用真实世界图像进行深度学习变得实用。

+   使用图像缩放和池化来减少模型中的参数数量，而不损失信息。

+   使用一系列级联的滤波器来检测特征与人类眼睛有相似之处。

+   VGG 形式化了重复的卷积模式的概念。

+   残差网络引入了特征重用的概念，并证明了在相同层数的情况下，与 VGG 相比可以获得更高的准确率，并且可以加深层数以获得更高的准确率。

+   批标准化允许模型在暴露于梯度消失或梯度爆炸之前，在层中更深入地学习以获得更高的准确性。
