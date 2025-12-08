# 图像分类

> [`deeplearningwithpython.io/chapters/chapter08_image-classification`](https://deeplearningwithpython.io/chapters/chapter08_image-classification)

计算机视觉是深度学习第一个重大的成功故事。它导致了 2011 年至 2015 年间深度学习的初步兴起。一种名为**卷积神经网络**的深度学习类型在那个时期开始在图像分类竞赛中取得显著的好成绩，最初是丹·西雷桑赢得了两个小众竞赛（2011 年 ICDAR 汉字识别竞赛和 2011 年 IJCNN 德国交通标志识别竞赛），然后，更为显著的是，在 2012 年秋季，辛顿团队赢得了备受瞩目的 ImageNet 大规模视觉识别挑战赛。在其他计算机视觉任务中，更多有希望的结果也迅速涌现。

有趣的是，这些早期的成功并不足以使深度学习在当时成为主流——这需要几年时间。计算机视觉研究界已经投入了多年时间研究神经网络以外的其他方法，并且并不因为新来者的出现就准备放弃它们。2013 年和 2014 年，深度学习仍然面临着许多资深计算机视觉研究者的强烈怀疑。直到 2016 年，它才最终成为主流。一位作者记得在 2014 年 2 月敦促一位前教授转向深度学习。“这是下一个大趋势！”他会说。“嗯，也许这只是个潮流，”教授会回答。到 2016 年，他整个实验室都在做深度学习。一个时代到来了，任何阻止它的想法都是徒劳的。

现在，你不断地与基于深度学习的视觉模型互动——通过 Google Photos、Google 图片搜索、你手机上的相机、YouTube、OCR 软件等等。这些模型也是自动驾驶、机器人、AI 辅助医疗诊断、自主零售结账系统，甚至自主农业等尖端研究的核心。

本章介绍了卷积神经网络，也称为**ConvNets**或**CNNs**，这是一种大多数计算机视觉应用所使用的深度学习模型。你将学习如何将卷积神经网络应用于图像分类问题——特别是那些涉及小型训练数据集的问题，如果你不是大型科技公司，这将是最常见的用例。

## 卷积神经网络简介

我们即将深入探讨卷积神经网络（ConvNets）的理论，以及它们为何在计算机视觉任务中取得了如此巨大的成功。但首先，让我们先从实际的角度来看一个简单的卷积神经网络示例。它使用卷积神经网络来分类 MNIST 数字，这是我们第二章中使用密集连接网络（那时我们的测试准确率为 97.8%）所执行的任务。尽管这个卷积神经网络将非常基础，但它的准确率将远远超过第二章中密集连接模型的水平。

以下代码行展示了基本卷积神经网络（ConvNet）的外观。它是由一系列的`Conv2D`和`MaxPooling2D`层堆叠而成。你将在下一分钟内确切地看到它们的作用。我们将使用在上一章中介绍的功能 API 来构建模型。

```py
import keras
from keras import layers

inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs) 
```

代码列表 8.1：实例化一个小型卷积神经网络

重要的是，卷积神经网络（ConvNet）接受形状为`(image_height, image_width, image_channels)`的张量作为输入（不包括批处理维度）。在这种情况下，我们将配置卷积神经网络（ConvNet）以处理大小为`(28, 28, 1)`的输入，这是 MNIST 图像的格式。

让我们展示我们的卷积神经网络（ConvNet）的架构。

```py
>>> model.summary()
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)          │ (None, 28, 28, 1)        │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d (Conv2D)                   │ (None, 26, 26, 64)       │           640 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)      │ (None, 13, 13, 64)       │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)                 │ (None, 11, 11, 128)      │        73,856 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)    │ (None, 5, 5, 128)        │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)                 │ (None, 3, 3, 256)        │       295,168 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ global_average_pooling2d          │ (None, 256)              │             0 │
│ (GlobalAveragePooling2D)          │                          │               │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense (Dense)                     │ (None, 10)               │         2,570 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 372,234 (1.42 MB)
 Trainable params: 372,234 (1.42 MB)
 Non-trainable params: 0 (0.00 B)
```

代码列表 8.2：显示模型的摘要

你可以看到每个`Conv2D`和`MaxPooling2D`层的输出都是一个形状为`(height, width, channels)`的三维张量。随着你在模型中深入，宽度和高度维度往往会缩小。通道数由传递给`Conv2D`层的第一个参数控制（64、128 或 256）。

在最后一个`Conv2D`层之后，我们得到一个形状为`(3, 3, 256)`的输出——一个 256 通道的 3×3 特征图。下一步是将这个输出输入到一个密集连接的分类器中，就像你已经熟悉的那些：一系列的`Dense`层。这些分类器处理向量，它们是一维的，而当前的输出是一个三阶张量。为了弥合这个差距，我们在添加`Dense`层之前，使用一个`GlobalAveragePooling2D`层将 3D 输出展平到一维。这个层将张量形状为`(3, 3, 256)`中的每个 3×3 特征图的平均值取出来，结果得到一个形状为`(256,)`的输出向量。最后，我们将进行 10 种分类，所以我们的最后一层有 10 个输出和一个 softmax 激活函数。

现在，让我们在 MNIST 数字上训练卷积神经网络（ConvNet）。我们将重用第二章中 MNIST 示例中的大量代码。因为我们正在进行带有 softmax 输出的 10 种分类，所以我们将使用分类交叉熵损失，因为我们标签是整数，我们将使用稀疏版本，`sparse_categorical_crossentropy`。

```py
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_images, train_labels, epochs=5, batch_size=64) 
```

代码列表 8.3：在 MNIST 图像上训练卷积神经网络

让我们在测试数据上评估模型。

```py
>>> test_loss, test_acc = model.evaluate(test_images, test_labels)
>>> print(f"Test accuracy: {test_acc:.3f}")
Test accuracy: 0.991
```

代码列表 8.4：评估卷积神经网络

与第二章中的密集连接模型相比，测试准确率为 97.8%，而基本的卷积神经网络（ConvNet）的测试准确率为 99.1%：我们降低了大约 60%（相对）的错误率。还不错！

但为什么与密集连接模型相比，这个简单的卷积神经网络（ConvNet）表现得如此出色？为了回答这个问题，让我们深入了解`Conv2D`和`MaxPooling2D`层的作用。

### 卷积操作

密集连接层和卷积层之间的基本区别是这样的：`Dense`层在其输入特征空间中学习全局模式（例如，对于一个 MNIST 数字，涉及所有像素的模式），而卷积层学习局部模式（见图 8.1）：在图像的情况下，在输入的小 2D 窗口中找到的模式。在先前的例子中，这些窗口都是 3 × 3。

![图片](img/1d29271b09d5f37759936df42a9f5aca.png)

图 8.1：图像可以被分解成局部模式，如边缘、纹理等。

这种关键特性给卷积神经网络带来了两个有趣的特性：

+   *它们学习的模式是平移不变的*。在图片的右下角学习到一定模式后，卷积神经网络可以在任何地方识别它：例如，在左上角。如果密集连接模型在新的位置出现，它必须重新学习该模式。这使得卷积神经网络在处理图像时数据效率高——因为*视觉世界在本质上具有平移不变性*。它们需要更少的训练样本来学习具有泛化能力的表示。

+   *它们可以学习模式的空间层次结构（见图 8.2）*。第一层卷积层将学习小的局部模式，如边缘，第二层卷积层将学习由第一层特征组成的大模式，依此类推。这使得卷积神经网络能够高效地学习越来越复杂和抽象的视觉概念——因为*视觉世界在本质上具有空间层次性*。

![图片](img/fd7bec6c7a9246339ceec605eca1d67b.png)

图 8.2：视觉世界形成了一个视觉模块的空间层次结构：基本的线条或纹理组合成简单的物体，如眼睛或耳朵，这些物体再组合成高级概念，如“猫”。

卷积操作在称为*特征图*的三阶张量上操作，具有两个空间轴（*高度*和*宽度*）以及一个*深度*轴（也称为*通道*轴）。对于 RGB 图像，深度轴的维度是 3，因为图像有三个颜色通道：红色、绿色和蓝色。对于像 MNIST 数字这样的黑白图片，深度是 1（灰度级别）。卷积操作从其输入特征图中提取补丁，并将相同的变换应用于所有这些补丁，产生一个*输出特征图*。这个输出特征图仍然是一个三阶张量：它有宽度和高度。其深度可以是任意的，因为输出深度是层的参数，在该深度轴上的不同通道不再像 RGB 输入那样代表特定的颜色；相反，它们代表*过滤器*。过滤器编码输入数据的特定方面：在较高层次上，单个过滤器可以编码“输入中存在面部”的概念，例如。

在 MNIST 示例中，第一个卷积层接受大小为`(28, 28, 1)`的特征图，并输出大小为`(26, 26, 64)`的特征图：它在其输入上计算 64 个过滤器。这些 64 个输出通道中每个都包含一个 26 × 26 的值网格，这是过滤器在输入上的*响应图*，指示该过滤器模式在输入的不同位置的反应（参见图 8.3）。这就是术语*特征图*的含义：深度轴上的每个维度都是一个特征（或过滤器），而秩为 2 的张量`output[:, :, n]`是此过滤器在输入上的 2D 空间*图*。 

![图片](img/d506560758d6cee6f0fc59966eb67536.png)

图 8.3：响应图的概念：一个 2D 图，表示在输入的不同位置上模式的呈现

卷积由两个关键参数定义：

+   *从输入中提取的补丁的大小* — 这些通常是 3 × 3 或 5 × 5。在示例中，它们是 3 × 3，这是一个常见的选择。

+   *输出特征图的深度* — 通过卷积计算出的过滤器数量。示例从 32 个深度开始，以 64 个深度结束。

在 Keras 的`Conv2D`层中，这些参数是传递给层的第一个参数：`Conv2D(output_depth, (window_height, window_width))`。

卷积通过*滑动*大小为 3 × 3 或 5 × 5 的窗口在 3D 输入特征图上，在每个可能的位置停止，并提取形状为`(window_height, window_width, input_depth)`的周围特征 3D 补丁。然后，每个这样的 3D 补丁被转换成一个形状为`(output_depth,)`的 1D 向量，这是通过与一个称为*卷积核*的学习权重矩阵的张量积来完成的——相同的核在每一个补丁上被重复使用。然后，所有这些向量（每个补丁一个）在空间上重新组装成一个形状为`(height, width, output_depth)`的 3D 输出图。输出特征图中的每个空间位置对应于输入特征图中的相同位置（例如，输出图的右下角包含有关输入右下角的信息）。例如，使用 3 × 3 窗口时，向量`output[i, j, :]`来自 3D 补丁`input[i-1:i+1, j-1:j+1, :]`。整个过程在图 8.4 中详细说明。

![图片](img/934c8e82aac0247164ceb48739a78d9d.png)

图 8.4：卷积的工作原理

注意，输出宽度和高度可能与输入宽度和高度不同。它们可能因为以下两个原因而不同：

+   边界效应，可以通过填充输入特征图来抵消

+   使用*步长*，我们将在下一节定义

让我们更深入地探讨这些概念。

#### 理解边界效应和填充

考虑一个 5 × 5 特征图（总共 25 个瓦片）。你只能在 9 个瓦片周围放置一个 3 × 3 窗口，形成一个 3 × 3 网格（见图 8.5）。因此，输出特征图将是 3 × 3。它略微缩小：在每个维度上精确地缩小两个瓦片。你可以在前面的例子中看到这个边界效应的实际应用：你从 28 × 28 的输入开始，经过第一层卷积后变成 26 × 26。

![图片](img/c33e84c0fad9d7d5fd595d8177a06c46.png)

图 8.5：在 5 × 5 输入特征图中 3 × 3 补丁的有效位置

如果你想得到与输入具有相同空间维度的输出特征图，你可以使用*填充*。填充包括在输入特征图的每一边添加适当数量的行和列，以便在每个输入瓦片周围放置中心卷积窗口。对于 3 × 3 窗口，你需要在右边添加一列，左边添加一列，顶部添加一行，底部添加一行。对于 5 × 5 窗口，你需要在顶部添加两行（见图 8.6）。

![图片](img/c875426d9fe089ca3ef09266ff1bdd92.png)

图 8.6：填充 5 × 5 输入以提取 25 个 3 × 3 补丁

在`Conv2D`层中，填充可以通过`padding`参数进行配置，该参数接受两个值：`"valid"`，表示没有填充（只使用有效的窗口位置）；和`"same"`，表示“以这种方式填充，以便输出具有与输入相同的宽度和高度。”`padding`参数的默认值为`"valid"`。

#### 理解卷积步长

影响输出大小的另一个因素是*步长*的概念。到目前为止的卷积描述假设卷积窗口的中心瓦片都是连续的。但是，两个连续窗口之间的距离是卷积的一个参数，称为其*步长*，默认值为 1。可以有*步长卷积*：步长大于 1 的卷积。在图 8.7 中，你可以看到通过在 5 × 5 输入上使用步长为 2 的 3 × 3 卷积提取的补丁。

![图片](img/fa3a3d514d7e28b55abc09e59d9e9dfd.png)

图 8.7：步长为 2 的 3 × 3 卷积补丁

使用步长 2 意味着特征图的宽度和高度以 2 的倍数下采样（除了任何由边界效应引起的改变）。步长卷积在分类模型中很少使用，但在某些类型的模型中很有用，你将在下一章中了解到。

在分类模型中，我们倾向于使用*最大池化*操作来下采样特征图——你可以在我们的第一个卷积神经网络示例中看到它的实际应用。让我们更深入地了解一下。

### 最大池化操作

在卷积神经网络（ConvNet）的示例中，你可能已经注意到，在每一个`MaxPooling2D`层之后，特征图的大小都会减半。例如，在第一个`MaxPooling2D`层之前，特征图的大小是 26 × 26，但最大池化操作将其减半到 13 × 13。这就是最大池化的作用：积极地对特征图进行下采样，就像步长卷积一样。

最大池化包括从输入特征图中提取窗口，并输出每个通道的最大值。从概念上讲，它与卷积相似，除了不是通过学习到的线性变换（卷积核）来转换局部块，而是通过硬编码的`max`张量操作来转换。与卷积的一个重大区别是，最大池化通常使用 2 × 2 窗口和步长 2 来将特征图下采样 2 倍。另一方面，卷积通常使用 3 × 3 窗口而没有步长（步长 1）。

为什么要以这种方式下采样特征图？为什么不移除最大池化层，而保持相当大的特征图直到最后？让我们看看这个选项。那么，我们的模型将看起来像这样。

```py
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model_no_max_pool = keras.Model(inputs=inputs, outputs=outputs) 
```

列表 8.5：缺少最大池化层的错误结构化的 ConvNet

这里是对模型的总结：

```py
>>> model_no_max_pool.summary()
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)        │ (None, 28, 28, 1)        │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)                 │ (None, 26, 26, 64)       │           640 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_4 (Conv2D)                 │ (None, 24, 24, 128)      │        73,856 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_5 (Conv2D)                 │ (None, 22, 22, 256)      │       295,168 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ global_average_pooling2d_1        │ (None, 256)              │             0 │
│ (GlobalAveragePooling2D)          │                          │               │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_1 (Dense)                   │ (None, 10)               │         2,570 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 372,234 (1.42 MB)
 Trainable params: 372,234 (1.42 MB)
 Non-trainable params: 0 (0.00 B)
```

这个设置有什么问题？两点：

+   这不利于学习特征的空间层次。第三层的 3 × 3 窗口将只包含来自初始输入中 7 × 7 窗口的信息。卷积神经网络学习的高级模式与初始输入相比仍然非常小，这可能不足以学习对数字进行分类（尝试通过只有 7 × 7 像素的窗口来识别数字！）。我们需要最后卷积层的特征包含关于输入整体的信息。

+   最终的特征图尺寸为 22 × 22。这非常大——当你对每个 22 × 22 的特征图取平均值时，与你的特征图只有 3 × 3 时相比，你将丢失大量的信息。

简而言之，使用下采样的原因是为了减小特征图的大小，使得它们包含的信息在空间上的分布越来越不均匀，并且越来越多地包含在通道中，同时通过使连续的卷积层“观察”到越来越大的窗口（从原始输入图像覆盖的分数来看）来诱导空间滤波层次。

注意，最大池化并不是实现这种下采样的唯一方法。正如你所知，你还可以在先前的卷积层中使用步长。你还可以使用平均池化而不是最大池化，其中每个局部输入块通过取块中每个通道的平均值来转换，而不是取最大值。但最大池化通常比这些替代方案更有效。简而言之，原因在于特征倾向于在特征图的不同块中编码某种模式或概念的时空存在（因此得名*特征图*），观察不同特征的*最大存在*比观察它们的*平均存在*更有信息量。因此，最合理的下采样策略是首先生成特征（通过无步长的卷积）的密集图，然后观察特征在小型块上的最大激活，而不是观察输入的稀疏窗口（通过步长卷积）或平均输入块，这可能会导致你错过或稀释特征存在信息。

到目前为止，你应该已经了解了卷积神经网络的基本知识——特征图、卷积和最大池化——并且你知道如何构建一个小型卷积神经网络来解决玩具问题，例如 MNIST 数字分类。现在让我们继续探讨更有用、更实际的应用。

## 在小型数据集上从头开始训练卷积神经网络

必须使用非常少的数据来训练图像分类模型是一种常见情况，如果你在专业环境中进行计算机视觉，你可能会在实践中遇到这种情况。所谓的“少量”样本可能意味着从几百到几万张图片。作为一个实际例子，我们将专注于将图像分类为狗或猫。我们将使用一个包含 5,000 张猫和狗图片的数据集（2,500 只猫，2,500 只狗），这些图片来自原始的 Kaggle 数据集。我们将使用 2,000 张图片进行训练，1,000 张进行验证，2,000 张进行测试。

在本节中，我们将回顾一种基本策略来解决这个问题：从头开始训练一个新模型，使用我们拥有的少量数据。我们将首先天真地在一个包含 2,000 个训练样本的小型卷积神经网络（ConvNet）上训练，没有任何正则化，以设定一个基准，看看能实现什么效果。这将使我们的分类准确率达到大约 80%。在那个点上，主要问题将是过拟合。然后我们将介绍**数据增强**，这是一种在计算机视觉中减轻过拟合的强大技术。通过使用数据增强，我们将提高模型，使其测试准确率达到大约 84%。

在下一节中，我们将回顾两个将深度学习应用于小数据集的必要技术：*使用预训练模型进行特征提取*和*微调预训练模型*（这将使我们的最终准确率达到 98.5%）。这三个策略——从头开始训练小模型、使用预训练模型进行特征提取和微调预训练模型——将构成您未来解决使用小数据集进行图像分类问题的工具箱。

### 深度学习对于小数据集问题的相关性

“足够的样本”来训练一个模型是相对的——首先，相对于您试图训练的模型的大小和深度。不可能只用几十个样本来训练一个卷积神经网络来解决复杂问题，但如果模型很小且很好地正则化，任务简单，那么几百个样本可能就足够了。由于卷积神经网络学习局部、平移不变的特征，它们在感知问题上的数据效率很高。在非常小的图像数据集上从头开始训练卷积神经网络，尽管数据相对较少，但仍然可以产生合理的结果，而无需任何定制的特征工程。您将在本节中看到这一点。

此外，深度学习模型在本质上具有很强的可重用性：您可以从大型数据集上训练的图像分类或语音到文本模型中提取，并在一个显著不同的问题上仅进行少量修改后重用。具体来说，在计算机视觉领域，许多预训练的分类模型可供公开下载，并可用于从非常少的数据中启动强大的视觉模型。这是深度学习的一个最大优势：特征重用。您将在下一节中探索这一点。

让我们先着手获取数据。

### 下载数据

我们将要使用的 Dogs vs. Cats 数据集并不包含在 Keras 中。它是由 Kaggle 在 2013 年底作为计算机视觉竞赛的一部分提供的，那时卷积神经网络还不是主流。您可以从`www.kaggle.com/c/dogs-vs-cats/data`下载原始数据集（如果您还没有 Kaggle 账户，需要创建一个——别担心，这个过程很简单）。您还可以使用 Kaggle API 在 Colab 中下载数据集。

我们数据集中的图片是中等分辨率的彩色 JPEG。图 8.8 展示了几个示例。

![图片](img/f90c51c5a2e790835b3e0c4525ac4835.png)

图 8.8：Dogs vs. Cats 数据集的样本。大小未修改：样本有不同的尺寸、颜色、背景等。

不足为奇的是，原始的 2013 年狗与猫 Kaggle 竞赛，所有参赛者都使用了卷积神经网络。最佳参赛者实现了高达 95%的准确率。在这个例子中，我们将接近这个准确率（在下一节中），尽管我们将模型训练在竞争对手可用的数据不到 10%的情况下。

这个数据集包含 25,000 张狗和猫的图像（每个类别 12,500 张）和 543 MB（压缩）。在下载并解压缩数据后，我们将创建一个新的数据集，包含三个子集：一个包含每个类别 1,000 个样本的训练集，一个包含每个类别 500 个样本的验证集，以及一个包含每个类别 1,000 个样本的测试集。为什么要这样做？因为你在职业生涯中遇到的大多数图像数据集只包含几千个样本，而不是成千上万。有更多的数据可用会使问题更容易——因此，使用小数据集进行学习是一个好的实践。

我们将要工作的子采样数据集将具有以下目录结构：

```py
dogs_vs_cats_small/
...train/
# Contains 1,000 cat images
......cat/
# Contains 1,000 dog images
......dog/
...validation/
# Contains 500 cat images
......cat/
# Contains 500 dog images
......dog/
...test/
# Contains 1,000 cat images
......cat/
# Contains 1,000 dog images
......dog/ 
```

让我们在`shutil`库的几次调用中实现它，这是一个用于运行类似 shell 命令的 Python 库。

```py
import os, shutil, pathlib

# Path to the directory where the original dataset was uncompressed
original_dir = pathlib.Path("train")
# Directory where we will store our smaller dataset
new_base_dir = pathlib.Path("dogs_vs_cats_small")

# Utility function to copy cat (respectively, dog) images from index
# `start_index` to index `end_index` to the subdirectory
# `new_base_dir/{subset_name}/cat` (respectively, dog). "subset_name"
# will be either "train," "validation," or "test."
def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / fname, dst=dir / fname)

# Creates the training subset with the first 1,000 images of each
# category
make_subset("train", start_index=0, end_index=1000)
# Creates the validation subset with the next 500 images of each
# category
make_subset("validation", start_index=1000, end_index=1500)
# Creates the test subset with the next 1,000 images of each category
make_subset("test", start_index=1500, end_index=2500) 
```

列表 8.6：将图像复制到训练、验证和测试目录

我们现在有 2,000 个训练图像，1,000 个验证图像和 2,000 个测试图像。每个分割包含每个类别的样本数量相同：这是一个平衡的二分类问题，这意味着分类准确率将是一个适当的成功衡量标准。

### 构建你的模型

我们将重用你在第一个示例中看到的相同的一般模型结构：卷积神经网络将是一系列交替的`Conv2D`（带有`relu`激活）和`MaxPooling2D`层。

但因为我们处理的是更大的图像和更复杂的问题，我们将使我们的模型更大，相应地：它将有两个额外的`Conv2D` + `MaxPooling2D`阶段。这既增加了模型的容量，也进一步减小了特征图的大小，以便在达到池化层时它们不会过大。在这里，因为我们从 180 × 180 像素的输入开始（这是一个有些任意的选择），我们最终在`GlobalAveragePooling2D`层之前得到 7 × 7 大小的特征图。

由于我们正在查看一个二分类问题，我们将以一个单元（大小为 1 的`Dense`层）和一个`sigmoid`激活函数结束模型。这个单元将编码模型正在查看一个类别还是另一个类别的概率。

最后一个小差异：我们将从`Rescaling`层开始构建模型，该层将重缩放图像输入（其值最初在[0, 255]范围内）到[0, 1]范围。

```py
import keras
from keras import layers

# The model expects RGB images of size 180 x 180.
inputs = keras.Input(shape=(180, 180, 3))
# Rescales inputs to the [0, 1] range by dividing them by 255
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=512, kernel_size=3, activation="relu")(x)
# Flattens the 3D activations with shape (height, width, 512) into 1D
# activations with shape (512,) by averaging them over spatial
# dimensions
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs) 
```

列表 8.7：实例化一个小型卷积神经网络进行狗与猫分类

让我们看看特征图维度是如何随着每一层连续变化的：

```py
>>> model.summary()
Model: "functional_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_2 (InputLayer)        │ (None, 180, 180, 3)      │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ rescaling (Rescaling)             │ (None, 180, 180, 3)      │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_6 (Conv2D)                 │ (None, 178, 178, 32)     │           896 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)    │ (None, 89, 89, 32)       │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_7 (Conv2D)                 │ (None, 87, 87, 64)       │        18,496 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)    │ (None, 43, 43, 64)       │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_8 (Conv2D)                 │ (None, 41, 41, 128)      │        73,856 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ max_pooling2d_4 (MaxPooling2D)    │ (None, 20, 20, 128)      │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_9 (Conv2D)                 │ (None, 18, 18, 256)      │       295,168 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ max_pooling2d_5 (MaxPooling2D)    │ (None, 9, 9, 256)        │             0 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ conv2d_10 (Conv2D)                │ (None, 7, 7, 512)        │     1,180,160 │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ global_average_pooling2d_2        │ (None, 512)              │             0 │
│ (GlobalAveragePooling2D)          │                          │               │
├───────────────────────────────────┼──────────────────────────┼───────────────┤
│ dense_2 (Dense)                   │ (None, 1)                │           513 │
└───────────────────────────────────┴──────────────────────────┴───────────────┘
 Total params: 1,569,089 (5.99 MB)
 Trainable params: 1,569,089 (5.99 MB)
 Non-trainable params: 0 (0.00 B)
```

对于编译步骤，您将像往常一样使用 `adam` 优化器。因为您以单个 sigmoid 单元结束模型，所以您将使用二元交叉熵作为损失（作为提醒，请查看第六章中的表 6.1，以获取各种情况下使用损失函数的速查表）。

```py
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
) 
```

列表 8.8：配置模型以进行训练

### 数据预处理

如您所知，在输入模型之前，数据应格式化为适当的预处理的浮点张量。目前，数据以 JPEG 文件的形式存储在驱动器上，因此将数据输入模型的大致步骤如下：

1.  读取图片文件。

1.  将 JPEG 内容解码为像素的 RGB 网格。

1.  将这些转换为浮点张量。

1.  将它们调整到共享大小（我们将使用 180 x 180）。

1.  将它们打包成批次（我们将使用 32 张图像的批次）。

这可能看起来有点令人畏惧，但幸运的是，Keras 提供了自动处理这些步骤的工具。特别是，Keras 特有的实用函数 `image_dataset_from_directory` 允许您快速设置一个数据管道，该管道可以自动将磁盘上的图像文件转换为预处理的张量批次。这就是您在这里要使用的。

调用 `image_dataset_from_directory(directory)` 将首先列出 `directory` 的子目录，并假设每个子目录包含您的一个类别的图像。然后，它将索引每个子目录中的图像文件。最后，它将创建并返回一个配置为读取这些文件的 `tf.data.Dataset` 对象，对它们进行洗牌，将它们解码为张量，将它们调整到共享大小，并将它们打包成批次。

```py
from keras.utils import image_dataset_from_directory

batch_size = 64
image_size = (180, 180)
train_dataset = image_dataset_from_directory(
    new_base_dir / "train", image_size=image_size, batch_size=batch_size
)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation", image_size=image_size, batch_size=batch_size
)
test_dataset = image_dataset_from_directory(
    new_base_dir / "test", image_size=image_size, batch_size=batch_size
) 
```

列表 8.9：使用 `image_dataset_from_directory` 从目录中读取图像

#### 理解 TensorFlow 数据集对象

TensorFlow 提供了 `tf.data` API 来创建机器学习模型的效率输入管道。其核心类是 `tf.data.Dataset`。

`Dataset` 类可用于任何框架中的数据加载和预处理，而不仅仅是 TensorFlow。您可以使用它与 JAX 或 PyTorch 一起使用。当您与 Keras 模型一起使用时，它的工作方式相同，独立于您当前使用的后端。

`Dataset` 对象是一个迭代器：您可以在 `for` 循环中使用它。它通常会返回输入数据和标签的批次。您可以直接将 `Dataset` 对象传递给 Keras 模型的 `fit()` 方法。

`Dataset` 类处理了许多关键特性，否则您可能难以自行实现，特别是跨多个 CPU 核心的预处理逻辑并行化，以及异步数据预取（在处理前一个批次的同时预处理下一个批次的数据，这可以保持执行流程不间断）。

`Dataset` 类还公开了一个功能式 API，用于修改数据集。这里有一个快速示例：让我们从一个随机数的 NumPy 数组创建一个 `Dataset` 实例。我们将考虑 1,000 个样本，其中每个样本是一个大小为 16 的向量。

```py
import numpy as np
import tensorflow as tf

random_numbers = np.random.normal(size=(1000, 16))
# The from_tensor_slices() class method can be used to create a Dataset
# from a NumPy array or a tuple or dict of NumPy arrays.
dataset = tf.data.Dataset.from_tensor_slices(random_numbers) 
```

列表 8.10：从 NumPy 数组实例化`Dataset`

起初，我们的数据集只产生单个样本。

```py
>>> for i, element in enumerate(dataset):
>>>     print(element.shape)
>>>     if i >= 2:
>>>         break
(16,)
(16,)
(16,)
```

列表 8.11：迭代数据集

您可以使用`.batch()`方法对数据进行分批。

```py
>>> batched_dataset = dataset.batch(32)
>>> for i, element in enumerate(batched_dataset):
>>>     print(element.shape)
>>>     if i >= 2:
>>>         break
(32, 16)
(32, 16)
(32, 16)
```

列表 8.12：批处理数据集

更广泛地说，您可以使用一系列有用的数据集方法，例如这些：

+   `.shuffle(buffer_size)`将在缓冲区内部进行元素洗牌。

+   `.prefetch(buffer_size)`将预取 GPU 内存中的元素缓冲区，以实现更好的设备利用率。

+   `.map(callable)`将对数据集的每个元素应用任意转换（期望接受数据集产生的单个元素作为输入的函数`callable`）。

`.map(function, num_parallel_calls)`方法尤其是一个您会经常使用的方法。这里有一个例子：让我们用它来将我们的玩具数据集中的元素从形状`(16,)`重塑为形状`(4, 4)`。

```py
>>> reshaped_dataset = dataset.map(
...     lambda x: tf.reshape(x, (4, 4)),
...     num_parallel_calls=8)
>>> for i, element in enumerate(reshaped_dataset):
...     print(element.shape)
...     if i >= 2:
...         break
(4, 4)
(4, 4)
(4, 4)
```

列表 8.13：使用`map()`对`Dataset`元素应用转换

在接下来的章节中，您将看到更多的`map()`操作。

#### 拟合模型

让我们查看这些`Dataset`对象之一的输出：它产生 180 × 180 RGB 图像的批次（形状`(32, 180, 180, 3)`）和整数标签（形状`(32,)`）。每个批次中有 32 个样本（批次大小）。

```py
>>> for data_batch, labels_batch in train_dataset:
>>>     print("data batch shape:", data_batch.shape)
>>>     print("labels batch shape:", labels_batch.shape)
>>>     break
data batch shape: (32, 180, 180, 3)
labels batch shape: (32,)
```

列表 8.14：显示`Dataset`产生的形状

让我们在我们的数据集上拟合模型。我们在`fit()`中使用`validation_data`参数来监控单独的`Dataset`对象上的验证指标。

注意，我们还在每个 epoch 后使用一个`ModelCheckpoint`回调来保存模型。我们配置它以保存文件的路径，以及参数`save_best_only=True`和`monitor="val_loss"`：它们告诉回调只有在当前`val_loss`指标值低于训练过程中任何先前时间时才保存新文件（覆盖任何先前的文件）。这保证了您保存的文件将始终包含对应于其表现最佳训练 epoch 的模型状态，从其在验证数据上的性能来看。因此，如果我们开始过拟合，我们不需要重新训练一个具有较少 epoch 数的新模型：我们只需重新加载我们的保存文件即可。

```py
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks,
) 
```

列表 8.15：使用`Dataset`拟合模型

让我们在训练过程中绘制模型在训练和验证数据上的损失和准确率（见图 8.9）。

```py
import matplotlib.pyplot as plt

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, "r--", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show() 
```

列表 8.16：显示训练期间的损失和准确率曲线

![](img/e154cf5f9475f3c0d64a49453174de12.png) ![](img/19d507efdd59c5b2077f28641a60ec3a.png)

图 8.9：简单卷积神经网络的训练和验证指标

这些图是过拟合的特征。训练准确率随时间线性增加，直到接近 100%，而验证准确率在 80% 左右达到峰值。验证损失在第 10 个 epoch 后达到最低点，然后停滞，而训练损失随着训练的进行而线性下降。

让我们检查测试准确率。我们将从其保存的文件中重新加载模型，以评估它在开始过拟合之前的状态。

```py
test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") 
```

列表 8.17：在测试集上评估模型

我们得到了 78.6% 的测试准确率（由于神经网络初始化的随机性，你可能会得到相差几个百分点的数字）。

由于你的训练样本相对较少（2,000 个），过拟合将成为你的首要关注点。你已经知道一些可以帮助减轻过拟合的技术，例如 dropout 和权重衰减（L2 正则化）。我们现在将使用一个新的技术，专门针对计算机视觉，并且在用深度学习模型处理图像时几乎被普遍使用：*数据增强*。

### 使用数据增强

过拟合是由于样本太少，无法学习，导致你无法训练一个可以泛化到新数据的模型。给定无限的数据，你的模型将接触到数据分布的每一个可能方面：你永远不会过拟合。数据增强通过通过一系列随机变换来增强样本，从而生成看起来可信的图像，从而从现有的训练样本中生成更多的训练数据。目标是，在训练时，你的模型永远不会看到完全相同的图片两次。这有助于使模型接触到数据的更多方面，并更好地泛化。

在 Keras 中，这可以通过 *数据增强层* 来实现。这些层可以通过两种方式之一添加：

+   *在模型的开始处* — *在模型内部*。在我们的例子中，层将直接位于 `Rescaling` 层之前。

+   *在数据管道内部* — *在模型外部*。在我们的例子中，我们将通过 `map()` 调用将它们应用于我们的 `Dataset`。

这两种选项之间的主要区别在于，在模型内部进行的数据增强将在 GPU 上运行，就像模型的其他部分一样。同时，在数据管道中进行的数据增强将在 CPU 上运行，通常在多个 CPU 核心上并行运行。有时，进行前者可能会有性能上的好处，但后者通常是更好的选择。所以我们就这么做吧！

```py
# Defines the transformations to apply as a list
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
]

# Creates a function that applies them sequentially
def data_augmentation(images, targets):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images, targets

# Maps this function into the dataset
augmented_train_dataset = train_dataset.map(
    data_augmentation, num_parallel_calls=8
)
# Enables prefetching of batches on GPU memory; important for best
# performance
augmented_train_dataset = augmented_train_dataset.prefetch(tf.data.AUTOTUNE) 
```

列表 8.18：定义数据增强阶段

这些只是可用的几层（更多内容请参阅 Keras 文档）。让我们快速浏览一下这段代码：

+   `RandomFlip("horizontal")` 将将水平翻转应用于通过它的随机 50% 的图像。

+   `RandomRotation(0.1)`将随机旋转输入图像，旋转角度在[–10%，+10%]范围内（这些是完整圆周的分数——以度为单位，范围将是[–36 度，+36 度]）。

+   `RandomZoom(0.2)`将根据随机因子在[–20%，+20%]范围内放大或缩小图像。

让我们看看增强后的图像（见图 8.10）。

```py
plt.figure(figsize=(10, 10))
# You can use take(N) to only sample N batches from the dataset. This
# is equivalent to inserting a break in the loop after the Nth batch.
for image_batch, _ in train_dataset.take(1):
    image = image_batch[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image, _ = data_augmentation(image, None)
        augmented_image = keras.ops.convert_to_numpy(augmented_image)
        # Displays the first image in the output batch. For each of the
        # nine iterations, this is a different augmentation of the same
        # image.
        plt.imshow(augmented_image.astype("uint8"))
        plt.axis("off") 
```

列表 8.19：显示一些随机增强的训练图像

![](img/c23ec10d204d38425efe734fd98e79ec.png)

图 8.10：通过随机数据增强生成一个非常好的男孩的变体

如果你使用这个数据增强配置训练一个新的模型，模型将永远不会看到相同的输入两次。但是它看到的输入仍然高度相关，因为它们来自少量原始图像——你无法产生新信息；你只能重新混合现有信息。因此，这可能不足以完全消除过拟合。为了进一步对抗过拟合，你还将向你的模型中添加一个`Dropout`层，就在密集连接分类器之前。

```py
inputs = keras.Input(shape=(180, 180, 3))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=512, kernel_size=3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
) 
```

列表 8.20：定义一个新的包含`dropout`的卷积神经网络

让我们使用数据增强和`dropout`来训练模型。因为我们预计过拟合将在训练后期发生，我们将训练两倍的周期数——100 个。请注意，我们将在未增强的图像上进行评估——数据增强通常仅在训练时执行，因为它是一种正则化技术。

```py
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    augmented_train_dataset,
    # Since we expect the model to overfit slower, we train for more
    # epochs.
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks,
) 
```

列表 8.21：在增强图像上训练正则化卷积神经网络

让我们再次绘制结果；见图 8.11。多亏了数据增强和`dropout`，我们开始过拟合的时间大大推迟，大约在 60-70 个周期（与原始模型的第 10 个周期相比）。验证准确率最终达到 85% 以上——比我们第一次尝试有了很大的改进。

![](img/b2977f5c12049346adc8bb619d2010b7.png) ![](img/7377fd180ae66d4f52ab60a6e968fd80.png)

图 8.11：数据增强后的训练和验证指标

让我们检查测试准确率。

```py
test_model = keras.models.load_model(
    "convnet_from_scratch_with_augmentation.keras"
)
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") 
```

列表 8.22：在测试集上评估模型

我们得到了 83.9% 的测试准确率。它开始看起来不错了！如果你使用 Colab，请确保下载保存的文件（`convnet_from_scratch_with_augmentation.keras`），因为我们将用它进行下一章的一些实验。

通过进一步调整模型的配置（例如每个卷积层的滤波器数量或模型中的层数），你可能能够获得更高的准确率，可能高达 90%。但是，仅通过从头开始训练自己的卷积神经网络来提高这个问题的准确率将非常困难，因为你可用的数据非常少。为了提高这个问题的准确率，下一步你将不得不使用预训练模型，这是下一两个章节的重点。

## 使用预训练模型

在小型图像数据集上进行深度学习的一种常见且高度有效的方法是使用预训练模型。*预训练模型*是在大型数据集上预先训练的模型，通常是在大规模图像分类任务上。如果这个原始数据集足够大且足够通用，那么预训练模型学习到的特征的空间层次可以有效地作为视觉世界的通用模型，因此其特征可以证明对许多不同的计算机视觉问题是有用的，即使这些新问题可能涉及与原始任务完全不同的类别。例如，你可以在 ImageNet（其中类别主要是动物和日常物品）上训练一个模型，然后将其重新用于识别图像中的家具项目等遥远的应用。这种在不同问题之间学习特征的便携性是深度学习相对于许多较老、较浅学习方法的显著优势，并且使得深度学习在小型数据问题上非常有效。

在这个例子中，让我们考虑一个在 ImageNet 数据集（1.4 百万个标记图像和 1,000 个不同类别）上训练的大型卷积神经网络（ConvNet）。ImageNet 包含许多动物类别，包括不同种类的猫和狗，因此你可以期待它在狗与猫的分类问题上表现良好。

我们将使用 Xception 架构。这可能是你第一次遇到这些可爱的模型名称之一——Xception、ResNet、EfficientNet 等等；如果你继续进行计算机视觉的深度学习，你会习惯它们的，因为它们会经常出现。你将在下一章中了解 Xception 的架构细节。

使用预训练模型有两种方式：*特征提取*和*微调*。我们将介绍这两种方法。让我们从特征提取开始。

### 使用预训练模型进行特征提取

特征提取包括使用先前训练的模型学习到的表示来从新的样本中提取有趣的特征。然后，这些特征将通过一个新的分类器进行处理，该分类器是从零开始训练的。

正如你之前看到的，用于图像分类的卷积神经网络由两部分组成：它们从一系列池化和卷积层开始，并以一个密集连接的分类器结束。第一部分被称为模型的*卷积基*或*骨干*。在卷积神经网络的情况下，特征提取包括从先前训练的网络中提取卷积基，将新数据通过它运行，并在输出之上训练一个新的分类器（见图 8.12）。

![图片](img/6ae9c33c2b3199361221a86298064605.png)

图 8.12：在保持相同的卷积基的同时交换分类器

为什么只重用卷积基？难道密集连接分类器也不能重用吗？通常，这样做应该避免。原因是卷积基学习到的表示可能更通用，因此更可重用：卷积神经网络的特征图是在图片上对通用概念的呈现图，这可能在处理任何计算机视觉问题时都很有用。但分类器学习到的表示必然是特定于模型训练的类别集的——它们只包含关于整个图片中某个或某个类别的存在概率的信息。此外，密集连接层中找到的表示不再包含任何关于输入图像中对象位置的信息：这些层消除了空间的概念，而对象位置仍然由卷积特征图描述。对于对象位置重要的问题，密集连接特征在很大程度上是无用的。

注意，通过特定卷积层提取的表示的通用性（以及因此的可重用性）取决于该层在模型中的深度。模型中较早的层提取的是局部、高度通用的特征图（例如视觉边缘、颜色和纹理），而较上面的层则提取更抽象的概念（例如“猫耳”或“狗眼”）。因此，如果你的新数据集与原始模型训练的数据集差异很大，你最好只使用模型的前几层来进行特征提取，而不是使用整个卷积基。

在这种情况下，由于 ImageNet 类别集包含多个狗和猫类别，重新使用原始模型密集连接层中的信息可能是有益的。但我们将选择不这样做，以便涵盖新问题的类别集与原始模型的类别集不重叠的更一般情况。让我们通过使用预训练模型的卷积基从猫和狗图像中提取有趣的特征，然后在这些特征之上训练一个猫狗分类器来将这一点付诸实践。

我们将使用 *KerasHub* 库来创建本书中使用的所有预训练模型。KerasHub 包含了流行预训练模型架构的 Keras 实现，并配以可以下载到您机器上的预训练权重。它包含许多卷积神经网络，如 Xception、ResNet、EfficientNet 和 MobileNet，以及我们将在本书后续章节中使用的更大型的生成模型。让我们尝试使用它来实例化在 ImageNet 数据集上训练的 Xception 模型。

```py
import keras_hub

conv_base = keras_hub.models.Backbone.from_preset("xception_41_imagenet") 
```

列表 8.23：实例化 Xception 卷积基

你会注意到几个问题。首先，KerasHub 使用术语 *backbone* 来指代没有分类头的底层特征提取网络（比“卷积基础”容易输入一些）。它还使用一个特殊的构造函数 `from_preset()`，该函数将下载 Xception 模型的配置和权重。

我们使用的模型名称中的“41”是什么意思？按照惯例，预训练的卷积神经网络通常根据它们的“深度”来命名。在这种情况下，41 表示我们的 Xception 模型有 41 个可训练层（卷积和密集层）堆叠在一起。这是我们迄今为止在书中使用的“最深层”模型，差距很大。

在我们能够使用这个模型之前，还有一个缺失的部分需要补充。每个预训练的卷积神经网络（ConvNet）在预训练之前都会对图像进行一些缩放和调整大小。确保我们的输入图像*匹配*是很重要的；否则，我们的模型将需要重新学习如何从具有完全不同输入范围的图像中提取特征。与其跟踪哪些预训练模型使用 `[0, 1]` 的像素值输入范围，哪些使用 `[-1, 1]` 的范围，我们不如使用一个名为 `ImageConverter` 的 KerasHub 层，它将把我们的图像缩放到与我们的预训练检查点匹配。它具有与骨干类相同的特殊 `from_preset()` 构造函数。

```py
preprocessor = keras_hub.layers.ImageConverter.from_preset(
    "xception_41_imagenet",
    image_size=(180, 180),
) 
```

列表 8.24：实例化与 Xception 模型配合使用的预处理

在这个阶段，你可以选择两种方法继续：

+   在你的数据集上运行卷积基础，将其输出记录到磁盘上的 NumPy 数组中，然后使用这些数据作为独立、密集连接的分类器的输入，类似于你在第四章和第五章中看到的。这个解决方案运行速度快且成本低，因为它只需要为每个输入图像运行一次卷积基础，而卷积基础是整个流程中最昂贵的部分。但出于同样的原因，这种技术不允许你使用数据增强。

+   通过在输入数据上添加 `Dense` 层来扩展你现有的模型（`conv_base`），并从头到尾运行整个模型。这将允许你使用数据增强，因为每次模型看到输入图像时，每个输入图像都会通过卷积基础。但出于同样的原因，这种技术比第一种技术昂贵得多。

我们将介绍这两种技术。让我们通过设置第一个所需的代码来逐步进行：记录 `conv_base` 在你的数据上的输出，并使用这些输出作为新模型的输入。

#### 无数据增强的快速特征提取

我们将首先通过调用 `conv_base` 模型的 `predict()` 方法来提取特征，作为我们的训练、验证和测试数据集。让我们遍历我们的数据集以提取预训练模型的特征。

```py
def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = preprocessor(images)
        features = conv_base.predict(preprocessed_images, verbose=0)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

train_features, train_labels = get_features_and_labels(train_dataset)
val_features, val_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset) 
```

列表 8.25：提取图像特征和相应的标签

重要的是，`predict()` 只期望图像，而不是标签，但我们的当前数据集生成的批次包含图像及其标签。

提取的特征当前形状为 `(samples, 6, 6, 2048)`:

```py
>>> train_features.shape
(2000, 6, 6, 2048)
```

在这一点上，你可以定义你的密集连接分类器（注意正则化时使用 dropout），并在你刚刚记录的数据和标签上对其进行训练。

```py
inputs = keras.Input(shape=(6, 6, 2048))
# Averages spatial dimensions to flatten the feature map
x = layers.GlobalAveragePooling2D()(inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    train_features,
    train_labels,
    epochs=10,
    validation_data=(val_features, val_labels),
    callbacks=callbacks,
) 
```

代码列表 8.26：定义和训练密集连接分类器

训练非常快，因为你只需要处理两个 `Dense` 层——即使是在 CPU 上，一个 epoch 也只需不到 1 秒。

让我们看看训练期间的损失和准确率曲线（见图 8.13）。

```py
import matplotlib.pyplot as plt

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "r--", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show() 
```

代码列表 8.27：绘制结果

![](img/605acfe8452fb67a1808d09146951f69.png) ![](img/df8ababe79bc755a8d10cdfc5323b506.png)

图 8.13：平面特征提取的训练和验证指标

你达到了略高于 98% 的验证准确率——比你在上一节中使用从头开始训练的小模型所达到的准确率要好得多。然而，这种比较有点不公平，因为 ImageNet 包含许多狗和猫的实例，这意味着我们的预训练模型已经具备了完成任务所需的精确知识。当你使用预训练特征时，这种情况并不总是如此。

然而，这些图表也表明你几乎从一开始就过度拟合——尽管使用了相当大的 dropout 率。这是因为这项技术没有使用数据增强，这对于防止使用小型图像数据集时的过度拟合是至关重要的。

让我们检查测试准确率：

```py
test_model = keras.models.load_model("feature_extraction.keras")
test_loss, test_acc = test_model.evaluate(test_features, test_labels)
print(f"Test accuracy: {test_acc:.3f}") 
```

我们得到了 98.1% 的测试准确率——与从头开始训练模型相比，这是一个非常好的改进！

#### 特征提取与数据增强相结合

现在，让我们回顾我们提到的第二个用于特征提取的技术，它速度较慢且成本更高，但允许你在训练期间使用数据增强：创建一个将 `conv_base` 与新的密集分类器链式连接的模型，并在输入上从头到尾进行训练。

要做到这一点，我们首先冻结卷积基。*冻结*一个层或一组层意味着防止它们在训练期间更新权重。在这里，如果你不这样做，那么卷积基之前学习到的表示将在训练期间被修改。因为顶部的 `Dense` 层是随机初始化的，非常大的权重更新将通过网络传播，实际上破坏了之前学习到的表示。

在 Keras 中，你可以通过将层的 `trainable` 属性设置为 `False` 来冻结一个层或模型。

```py
import keras_hub

conv_base = keras_hub.models.Backbone.from_preset(
    "xception_41_imagenet",
    trainable=False,
) 
```

代码列表 8.28：创建冻结的卷积基

将 `trainable` 设置为 `False` 将清空层或模型的可训练权重列表。

```py
>>> conv_base.trainable = True
>>> # The number of trainable weights before freezing the conv base
>>> len(conv_base.trainable_weights)
154
>>> conv_base.trainable = False
>>> # The number of trainable weights after freezing the conv base
>>> len(conv_base.trainable_weights)
0
```

代码列表 8.29：冻结前后的可训练权重列表

现在，我们只需创建一个新的模型，将冻结的卷积基础和密集分类器链接在一起，如下所示：

```py
inputs = keras.Input(shape=(180, 180, 3))
x = preprocessor(inputs)
x = conv_base(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
) 
```

使用这种设置，只有你添加的两个`Dense`层的权重将被训练。总共是四个权重张量：每个层两个（主权重矩阵和偏置向量）。请注意，为了使这些更改生效，你必须首先编译模型。如果你在编译后修改权重的可训练性，那么你应该重新编译模型，否则这些更改将被忽略。

让我们训练我们的模型。我们将重用我们的增强数据集`augmented_train_dataset`。多亏了数据增强，模型开始过度拟合需要更长的时间，因此我们可以训练更多的轮次——让我们做 30 轮：

```py
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction_with_data_augmentation.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    augmented_train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks,
) 
```

让我们再次绘制结果（见图 8.14）。这个模型达到了 98.2%的验证准确率。

![图 8.14](img/8e3a8a5c36838e7499ef2d4ccb9849c0.png) ![图 8.15](img/ae553ac8e7101ab0653eeff0e1edba20.png)

图 8.14：使用数据增强进行特征提取的训练和验证指标

让我们检查测试准确率。

```py
test_model = keras.models.load_model(
    "feature_extraction_with_data_augmentation.keras"
)
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") 
```

列表 8.30：在测试集上评估模型

我们得到了 98.4%的测试准确率。这并没有比之前的模型有改进，这有点令人失望。这可能表明我们的数据增强配置并不完全符合测试数据的分布。让我们看看我们能否在我们的最新尝试中做得更好。

### 微调预训练模型

另一种广泛使用的模型重用技术，与特征提取互补的是*微调*（见图 8.15）。微调包括解冻用于特征提取的冻结模型基础，并联合训练模型的新添加部分（在这种情况下，全连接分类器）和基础模型。这被称为*微调*，因为它略微调整了被重用的模型的更抽象的表示，使其更相关于当前的问题。

我们之前提到，为了能够在顶部训练一个随机初始化的分类器，首先需要冻结预训练的卷积基础。出于同样的原因，只有在顶部的分类器已经训练好的情况下，才能微调卷积基础。如果分类器尚未训练，那么在训练过程中通过网络传播的错误信号将太大，并且之前由正在微调的层学到的表示将被破坏。因此，微调网络的步骤如下：

1.  在已经训练好的基础网络上添加你的自定义网络。

1.  解冻基础网络。

1.  训练你添加的部分。

1.  解冻基础网络。

1.  联合训练这两个层以及你添加的部分。

注意，你不应该解冻“批量归一化”层（`BatchNormalization`）。批量归一化及其对微调的影响将在下一章中解释。

当进行特征提取时，你已经完成了前三个步骤。让我们继续进行第 4 步：你将解冻你的`conv_base`。

让我们以一个非常低的学习率开始微调模型。使用低学习率的原因是，你希望限制你对正在微调的层的表示所做的修改的幅度。过大的更新可能会损害这些表示。

```py
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="fine_tuning.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model.fit(
    augmented_train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks,
) 
```

列表 8.31：微调模型

你现在可以最终在测试数据上评估这个模型了（见图 8.15）：

```py
model = keras.models.load_model("fine_tuning.keras")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") 
```

![图片 1](img/a4eed2da3b1ba70dcee533725f83fcc8.png) ![图片 2](img/6549d0428cc9b3aa20c7b92abef1c8e1.png)

![图 8.15](img/#figure-8-15)：微调的训练和验证指标

在这里，你得到了 98.6%的测试准确率（再次强调，你的结果可能在半个百分点之内）。在围绕这个数据集的原始 Kaggle 竞赛中，这将是一份顶尖的结果。然而，这种比较并不完全公平，因为你使用了已经包含有关猫和狗先前知识的预训练特征，而当时的竞争对手无法使用这些特征。

在积极的一面，通过使用现代深度学习技术，你只用到了竞赛中可用训练数据的一小部分（大约 10%）就达到了这个结果。与训练 2,000 个样本相比，能够在 20,000 个样本上进行训练有着巨大的差异！

现在你已经拥有了一套处理图像分类问题（特别是小型数据集）的强大工具。

## 摘要

+   卷积神经网络在计算机视觉任务中表现出色。即使是在非常小的数据集上，从头开始训练一个卷积神经网络也是可能的，并且可以得到相当不错的结果。

+   卷积神经网络通过学习一系列模块化的模式和概念来表示视觉世界，从而工作。

+   在小型数据集上，过拟合将是主要问题。数据增强是当你处理图像数据时对抗过拟合的一种强大方式。

+   通过特征提取，在新的数据集上重用现有的卷积神经网络（ConvNet）非常容易。这对于处理小型图像数据集来说是一种非常有价值的技巧。

+   作为特征提取的补充，你可以使用微调，它适应于新问题，并调整了现有模型之前学习的一些表示。这进一步提升了性能。
