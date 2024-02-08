# 第2章。深度学习

让我们从深度学习的基本定义开始：

> 深度学习是一类机器学习算法，它使用*多个堆叠层的处理单元*从*非结构化*数据中学习高级表示。

要充分理解深度学习，我们需要进一步探讨这个定义。首先，我们将看一下深度学习可以用来建模的不同类型的非结构化数据，然后我们将深入研究构建多个堆叠层的处理单元来解决分类任务的机制。这将为我们未来关注生成任务的章节奠定基础。

# 深度学习数据

许多类型的机器学习算法需要*结构化*的表格数据作为输入，这些数据被排列成描述每个观察结果的特征列。例如，一个人的年龄、收入和上个月的网站访问次数都是可以帮助预测这个人是否会在接下来的一个月订阅特定在线服务的特征。我们可以使用这些特征的结构化表格来训练逻辑回归、随机森林或XGBoost模型来预测二元响应变量——这个人是否订阅了（1）还是没有订阅（0）？在这里，每个单独的特征都包含了关于观察结果的信息，模型将学习这些特征如何相互作用以影响响应。

*非结构化*数据指的是任何不自然地排列成特征列的数据，例如图像、音频和文本。当然，图像具有空间结构，录音或文本段落具有时间结构，视频数据既具有空间结构又具有时间结构，但由于数据不是以特征列的形式到达，因此被认为是非结构化的，如[图2-1](#structured_unstructured)所示。

![](Images/gdl2_0201.png)

###### 图2-1。结构化数据和非结构化数据之间的区别

当我们的数据是非结构化的时候，单个像素、频率或字符几乎完全没有信息性。例如，知道图像的第234个像素是一种泥泞的褐色并不能真正帮助识别图像是房子还是狗，知道句子的第24个字符是一个*e*并不能帮助预测文本是关于足球还是政治。

像素或字符实际上只是画布上的凹痕，其中嵌入了高级信息性特征，例如烟囱的图像或*前锋*这个词。如果图像中的烟囱被放在房子的另一侧，图像仍然包含烟囱，但这些信息现在由完全不同的像素传递。如果文本中的*前锋*出现在稍早或稍晚的位置，文本仍然是关于足球的，但不同的字符位置会提供这些信息。数据的粒度与高度的空间依赖性破坏了像素或字符作为独立信息特征的概念。

因此，如果我们在原始像素值上训练逻辑回归、随机森林或XGBoost模型，那么训练好的模型通常只会在最简单的分类任务中表现不佳。这些模型依赖于输入特征具有信息性且不具有空间依赖性。另一方面，深度学习模型可以直接从非结构化数据中学习如何构建高级信息性特征。

深度学习可以应用于结构化数据，但其真正的力量，特别是在生成建模方面，来自于其处理非结构化数据的能力。通常，我们希望生成非结构化数据，例如新图像或原始文本字符串，这就是为什么深度学习对生成建模领域产生如此深远影响的原因。

# 深度神经网络

大多数深度学习系统是*人工神经网络*（ANNs，或简称*神经网络*）具有多个堆叠的隐藏层。因此，*深度学习*现在几乎已经成为*深度神经网络*的同义词。然而，任何使用多层学习输入数据的高级表示的系统也是一种深度学习形式（例如，深度信念网络）。

让我们首先详细解释一下神经网络的含义，然后看看它们如何用于从非结构化数据中学习高级特征。

## 什么是神经网络？

神经网络由一系列堆叠的*层*组成。每一层包含通过一组*权重*连接到前一层单元的*单元*。正如我们将看到的，有许多不同类型的层，但其中最常见的是*全连接*（或*密集*）层，它将该层中的所有单元直接连接到前一层的每个单元。

所有相邻层都是全连接的神经网络称为*多层感知器*（MLPs）。这是我们将要学习的第一种神经网络。[图2-2](#deep_learning_diagram)中显示了一个MLP的示例。

![](Images/gdl2_0202.png)

###### 图2-2。一个预测脸部是否微笑的多层感知器的示例

输入（例如，一张图像）依次通过网络中的每一层进行转换，直到达到输出层，这被称为网络的*前向传递*。具体来说，每个单元对其输入的加权和应用非线性变换，并将输出传递到后续层。最终的输出层是这个过程的结尾，单个单元输出一个概率，表明原始输入属于特定类别（例如，*微笑*）。

深度神经网络的魔力在于找到每一层的权重集，以获得最准确的预测。找到这些权重的过程就是我们所说的*训练*网络。

在训练过程中，一批图像通过网络传递，并将预测输出与真实值进行比较。例如，网络可能为一个真正微笑的人的图像输出80%的概率，为一个真正不微笑的人的图像输出23%的概率。对于这些示例，完美的预测将输出100%和0%，因此存在一定的误差。然后，预测中的误差通过网络向后传播，调整每组权重，使其朝着最显著改善预测的方向微调。这个过程被适当地称为*反向传播*。逐渐地，每个单元变得擅长识别一个特定的特征，最终帮助网络做出更好的预测。

## 学习高级特征

使神经网络如此强大的关键属性是它们能够从输入数据中学习特征，而无需人类指导。换句话说，我们不需要进行任何特征工程，这就是为什么神经网络如此有用！我们可以让模型决定如何安排其权重，只受其希望最小化预测误差的影响。

例如，让我们来解释一下[图2-2](#deep_learning_diagram)中所示的网络，假设它已经被训练得可以准确预测给定输入脸部是否微笑：

1.  单元A接收输入像素的单个通道的值。

1.  单元B组合其输入值，使得当存在特定的低级特征，例如边缘时，它发射最强。

1.  单元C组合低级特征，使得当图像中看到高级特征，例如*牙齿*时，它发射最强。

1.  单元D结合高级特征，使得当原始图像中的人在微笑时它发射最强。

每个后续层中的单元能够通过结合来自前一层的低级特征来表示原始输入的越来越复杂的方面。令人惊讶的是，这是训练过程中自然产生的——我们不需要*告诉*每个单元要寻找什么，或者它应该寻找高级特征还是低级特征。

输入层和输出层之间的层被称为*隐藏*层。虽然我们的例子只有两个隐藏层，但深度神经网络可以有更多层。堆叠大量层允许神经网络逐渐构建信息，从先前层中的低级特征逐渐构建出更高级别的特征。例如，用于图像识别的ResNet包含152层。

接下来，我们将直接深入深度学习的实践方面，并使用TensorFlow和Keras进行设置，以便您可以开始构建自己的深度神经网络。

## TensorFlow和Keras

[*TensorFlow*](https://www.tensorflow.org)是由谷歌开发的用于机器学习的开源Python库。TensorFlow是构建机器学习解决方案中最常用的框架之一，特别强调张量的操作（因此得名）。它提供了训练神经网络所需的低级功能，例如计算任意可微表达式的梯度和高效执行张量操作。

[*Keras*](https://keras.io)是一个用于构建神经网络的高级API，构建在TensorFlow之上（[图2-3](#tf_keras_logos)）。它非常灵活和用户友好，是开始深度学习的理想选择。此外，Keras提供了许多有用的构建模块，可以通过其功能API组合在一起，创建高度复杂的深度学习架构。

![](Images/gdl2_0203.png)

###### 图2-3\. TensorFlow和Keras是构建深度学习解决方案的优秀工具

如果您刚开始学习深度学习，我强烈推荐使用TensorFlow和Keras。这个设置将允许您在生产环境中构建任何您能想到的网络，同时还提供易于学习的API，可以快速开发新的想法和概念。让我们从看看使用Keras构建多层感知器有多容易开始。

# 多层感知器（MLP）

在本节中，我们将使用*监督学习*训练一个MLP来对给定的图像进行分类。监督学习是一种机器学习算法，计算机在标记的数据集上进行训练。换句话说，用于训练的数据集包括带有相应输出标签的输入数据。算法的目标是学习输入数据和输出标签之间的映射，以便它可以对新的、未见过的数据进行预测。

MLP是一种判别模型（而不是生成模型），但在本书后面的章节中，监督学习仍将在许多类型的生成模型中发挥作用，因此这是我们旅程的一个好起点。

# 运行此示例的代码

这个例子的代码可以在位于书籍存储库中的Jupyter笔记本中找到，位置为*notebooks/02_deeplearning/01_mlp/mlp.ipynb*。

## 准备数据

在这个例子中，我们将使用[CIFAR-10](https://oreil.ly/cNbFG)数据集，这是一个包含60,000个32×32像素彩色图像的集合，与Keras捆绑在一起。每个图像被分类为10个类别中的一个，如[图2-4](#cifar)所示。

![](Images/gdl2_0204.png)

###### 图2-4\. CIFAR-10数据集中的示例图像（来源：[Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf))^([2](ch02.xhtml#idm45387033163216))

默认情况下，图像数据由每个像素通道的0到255之间的整数组成。我们首先需要通过将这些值缩放到0到1之间来预处理图像，因为当每个输入的绝对值小于1时，神经网络的效果最好。

我们还需要将图像的整数标签更改为独热编码向量，因为神经网络的输出将是图像属于每个类的概率。如果图像的类整数标签是<math alttext="i"><mi>i</mi></math>，那么它的独热编码是一个长度为10的向量（类的数量），除了第<math alttext="i"><mi>i</mi></math>个元素为1之外，其他元素都为0。这些步骤在[示例2-1](#preprocessing-cifar-10)中显示。

##### 示例2-1。预处理CIFAR-10数据集

```py
import numpy as np
from tensorflow.keras import datasets, utils

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data() ![1](Images/1.png)

NUM_CLASSES = 10

x_train = x_train.astype('float32') / 255.0 ![2](Images/2.png)
x_test = x_test.astype('float32') / 255.0

y_train = utils.to_categorical(y_train, NUM_CLASSES) ![3](Images/3.png)
y_test = utils.to_categorical(y_test, NUM_CLASSES)
```

[![1](Images/1.png)](#co_deep_learning_CO1-1)

加载CIFAR-10数据集。`x_train`和`x_test`分别是形状为`[50000, 32, 32, 3]`和`[10000, 32, 32, 3]`的`numpy`数组。`y_train`和`y_test`分别是形状为`[50000, 1]`和`[10000, 1]`的`numpy`数组，包含每个图像类的范围为0到9的整数标签。

[![2](Images/2.png)](#co_deep_learning_CO1-2)

缩放每个图像，使像素通道值介于0和1之间。

[![3](Images/3.png)](#co_deep_learning_CO1-3)

对标签进行独热编码——`y_train`和`y_test`的新形状分别为`[50000, 10]`和`[10000, 10]`。

我们可以看到训练图像数据（`x_train`）存储在形状为`[50000, 32, 32, 3]`的*张量*中。在这个数据集中没有*列*或*行*；相反，这是一个具有四个维度的张量。张量只是一个多维数组——它是矩阵向超过两个维度的自然扩展。这个张量的第一个维度引用数据集中图像的索引，第二和第三个维度与图像的大小有关，最后一个是通道（即红色、绿色或蓝色，因为这些是RGB图像）。

例如，[示例2-2](#pixel-value)展示了如何找到图像中特定像素的通道值。

##### 示例2-2。图像54中位置为（12,13）的像素的绿色通道（1）值

```py
x_train[54, 12, 13, 1]
# 0.36862746
```

## 构建模型

在Keras中，您可以将神经网络的结构定义为`Sequential`模型或使用功能API。

`Sequential`模型适用于快速定义一系列层的线性堆叠（即一个层直接跟在前一个层后面，没有任何分支）。我们可以使用`Sequential`类来定义我们的MLP模型，如[示例2-3](#sequential_functional)所示。

##### 示例2-3。使用`Sequential`模型构建我们的MLP

```py
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(200, activation = 'relu'),
    layers.Dense(150, activation = 'relu'),
    layers.Dense(10, activation = 'softmax'),
])
```

本书中的许多模型要求从一层输出传递到多个后续层，或者反过来，一层接收来自多个前面层的输入。对于这些模型，`Sequential`类不适用，我们需要使用功能API，这样更加灵活。

###### 提示

我建议即使您刚开始使用Keras构建线性模型，也应该使用功能API而不是`Sequential`模型，因为随着您的神经网络变得更加复杂，功能API将在长远中为您提供更好的服务。功能API将为您提供对深度神经网络设计的完全自由。

[示例2-4](#sequential_functional-2)展示了使用功能API编码的相同MLP。在使用功能API时，我们使用`Model`类来定义模型的整体输入和输出层。

##### 示例2-4。使用功能API构建我们的MLP

```py
from tensorflow.keras import layers, models

input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(input_layer)
x = layers.Dense(units=200, activation = 'relu')(x)
x = layers.Dense(units=150, activation = 'relu')(x)
output_layer = layers.Dense(units=10, activation = 'softmax')(x)
model = models.Model(input_layer, output_layer)
```

这两种方法提供相同的模型——架构的图表显示在[图2-5](#cifar_nn)中。

![](Images/gdl2_0205.png)

###### 图2-5。MLP架构的图表

现在让我们更详细地看一下MLP中使用的不同层和激活函数。

### 层

为构建我们的MLP，我们使用了三种不同类型的层：`Input`、`Flatten`和`Dense`。

`Input`层是网络的入口点。我们告诉网络每个数据元素的形状应该是一个元组。请注意，我们不指定批量大小；这是不必要的，因为我们可以同时将任意数量的图像传递到`Input`层中。我们不需要在`Input`层定义中明确指定批量大小。

接下来，我们将这个输入展平成一个向量，使用`Flatten`层。这将导致一个长度为3072的向量（= 32 × 32 × 3）。我们这样做的原因是因为后续的`Dense`层要求其输入是平坦的，而不是多维数组。正如我们将在后面看到的，其他类型的层需要多维数组作为输入，因此您需要了解每种层类型所需的输入和输出形状，以便了解何时需要使用`Flatten`。

`Dense`层是神经网络中最基本的构建块之一。它包含一定数量的单元，这些单元与前一层密切连接，也就是说，层中的每个单元都与前一层中的每个单元连接，通过一个携带权重的单一连接（可以是正数或负数）。给定单元的输出是它从前一层接收的输入的加权和，然后通过非线性*激活函数*传递到下一层。激活函数对于确保神经网络能够学习复杂函数并且不仅仅输出其输入的线性组合至关重要。

### 激活函数

有许多种激活函数，但其中最重要的三种是ReLU、sigmoid和softmax。

*ReLU*（修正线性单元）激活函数被定义为如果输入为负数则为0，否则等于输入。*LeakyReLU*激活函数与ReLU非常相似，但有一个关键区别：ReLU激活函数对于小于0的输入值返回0，而LeakyReLU函数返回与输入成比例的一个小负数。如果ReLU单元总是输出0，有时会出现死亡现象，因为存在对负值预激活的大偏差。在这种情况下，梯度为0，因此没有错误通过该单元向后传播。LeakyReLU激活通过始终确保梯度为非零来解决这个问题。基于ReLU的函数是在深度网络的层之间使用的最可靠的激活函数之一，以鼓励稳定的训练。

如果您希望从该层输出的结果在0和1之间缩放，那么*sigmoid*激活函数是有用的，例如，对于具有一个输出单元的二元分类问题或多标签分类问题，其中每个观察结果可以属于多个类。[图2-6](#activations)显示了ReLU、LeakyReLU和sigmoid激活函数并排进行比较。

![](Images/gdl2_0206.png)

###### 图2-6。ReLU、LeakyReLU和sigmoid激活函数

如果您希望从该层输出的总和等于1，则*softmax*激活函数是有用的；例如，对于每个观察结果只属于一个类的多类分类问题。它被定义为：

<math alttext="y Subscript i Baseline equals StartFraction e Superscript x Super Subscript i Superscript Baseline Over sigma-summation Underscript j equals 1 Overscript upper J Endscripts e Superscript x Super Subscript j Superscript Baseline EndFraction" display="block"><mrow><msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo> <mfrac><msup><mi>e</mi> <msub><mi>x</mi> <mi>i</mi></msub></msup> <mrow><munderover><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>J</mi></munderover> <msup><mi>e</mi> <msub><mi>x</mi> <mi>j</mi></msub></msup></mrow></mfrac></mrow></math>

在这里，*J*是层中单元的总数。在我们的神经网络中，我们在最后一层使用softmax激活，以确保输出是一组总和为1的10个概率，这可以被解释为图像属于每个类的可能性。

在Keras中，激活函数可以在层内定义（[示例2-5](#activation-function-together)）或作为单独的层定义（[示例2-6](#activation-function-separate)）。

##### 示例2-5。作为`Dense`层的一部分定义的ReLU激活函数

```py
x = layers.Dense(units=200, activation = 'relu')(x)
```

##### 示例2-6。作为自己的层定义的ReLU激活函数

```py
x = layers.Dense(units=200)(x)
x = layers.Activation('relu')(x)
```

在我们的示例中，我们通过两个`Dense`层传递输入，第一个有200个单元，第二个有150个，两者都带有ReLU激活函数。

### 检查模型

我们可以使用`model.summary()`方法来检查每一层网络的形状，如[表2-1](#first_nn_shape)所示。

表2-1. `model.summary()`方法的输出

| 层（类型） | 输出形状 | 参数 # |
| --- | --- | --- |
| InputLayer | (None, 32, 32, 3) | 0 |
| 展平 | (None, 3072) | 0 |
| Dense | (None, 200) | 614,600 |
| Dense | (None, 150) | 30,150 |
| Dense | (None, 10) | 1,510 |
| 总参数 | 646,260 |
| 可训练参数 | 646,260 |
| 不可训练参数 | 0 |

注意我们的`Input`层的形状与`x_train`的形状匹配，而我们的`Dense`输出层的形状与`y_train`的形状匹配。Keras使用`None`作为第一维的标记，以显示它尚不知道将传递到网络中的观测数量。实际上，它不需要知道；我们可以一次通过1个观测或1000个观测通过网络。这是因为张量操作是使用线性代数同时在所有观测上进行的—这是由TensorFlow处理的部分。这也是为什么在GPU上训练深度神经网络而不是在CPU上时性能会提高的原因：GPU针对大型张量操作进行了优化，因为这些计算对于复杂的图形处理也是必要的。

`summary`方法还会给出每一层将被训练的参数（权重）的数量。如果你发现你的模型训练速度太慢，检查摘要看看是否有任何包含大量权重的层。如果有的话，你应该考虑是否可以减少该层中的单元数量以加快训练速度。

###### 提示

确保你理解每一层中参数是如何计算的！重要的是要记住，默认情况下，给定层中的每个单元也连接到一个额外的*偏置*单元，它总是输出1。这确保了即使来自前一层的所有输入为0，单元的输出仍然可以是非零的。

因此，200单元`Dense`层中的参数数量为200 * (3,072 + 1) = 614,600。

## 编译模型

在这一步中，我们使用一个优化器和一个损失函数来编译模型，如[示例2-7](#optimizer-loss)所示。

##### 示例2-7. 定义优化器和损失函数

```py
from tensorflow.keras import optimizers

opt = optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])
```

现在让我们更详细地看一下我们所说的损失函数和优化器。

### 损失函数

*损失函数*被神经网络用来比较其预测输出与实际情况的差异。它为每个观测返回一个单一数字；这个数字越大，网络在这个观测中的表现就越差。

Keras提供了许多内置的损失函数可供选择，或者你可以创建自己的损失函数。最常用的三个是均方误差、分类交叉熵和二元交叉熵。重要的是要理解何时适合使用每种损失函数。

如果你的神经网络旨在解决回归问题（即输出是连续的），那么你可能会使用*均方误差*损失。这是每个输出单元的实际值<math alttext="y Subscript i"><msub><mi>y</mi> <mi>i</mi></msub></math>和预测值<math alttext="p Subscript i"><msub><mi>p</mi> <mi>i</mi></msub></math>之间的平方差的平均值，其中平均值是在所有<math alttext="n"><mi>n</mi></math>个输出单元上取得的：

<math alttext="upper M upper S upper E equals StartFraction 1 Over n EndFraction sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis y Subscript i Baseline minus p Subscript i Baseline right-parenthesis squared" display="block"><mstyle scriptlevel="0" displaystyle="true"><mrow><mo form="prefix">MSE</mo> <mo>=</mo> <mfrac><mn>1</mn> <mi>n</mi></mfrac> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <msup><mrow><mo>(</mo><msub><mi>y</mi> <mi>i</mi></msub> <mo>-</mo><msub><mi>p</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mn>2</mn></msup></mrow></mstyle></math>

如果你正在处理一个分类问题，其中每个观测只属于一个类，那么*分类交叉熵*是正确的损失函数。它定义如下：

<math alttext="minus sigma-summation Underscript i equals 1 Overscript n Endscripts y Subscript i Baseline log left-parenthesis p Subscript i Baseline right-parenthesis" display="block"><mrow><mo>-</mo> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <msub><mi>y</mi> <mi>i</mi></msub> <mo form="prefix">log</mo> <mrow><mo>(</mo> <msub><mi>p</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></math>

最后，如果你正在处理一个具有一个输出单元的二元分类问题，或者一个每个观测可以同时属于多个类的多标签问题，你应该使用*二元交叉熵*：

<math alttext="minus StartFraction 1 Over n EndFraction sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis y Subscript i Baseline log left-parenthesis p Subscript i Baseline right-parenthesis plus left-parenthesis 1 minus y Subscript i Baseline right-parenthesis log left-parenthesis 1 minus p Subscript i Baseline right-parenthesis right-parenthesis" display="block"><mrow><mo>-</mo> <mfrac><mn>1</mn> <mi>n</mi></mfrac> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <mrow><mo>(</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo form="prefix">log</mo> <mrow><mo>(</mo> <msub><mi>p</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>+</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo form="prefix">log</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>p</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>

### 优化器

*优化器* 是基于损失函数的梯度更新神经网络权重的算法。最常用和稳定的优化器之一是 *Adam*（自适应矩估计）。^([3](ch02.xhtml#idm45387032147088)) 在大多数情况下，您不需要调整Adam优化器的默认参数，除了 *学习率*。学习率越大，每个训练步骤中权重的变化就越大。虽然初始时使用较大的学习率训练速度更快，但缺点是可能导致训练不稳定，无法找到损失函数的全局最小值。这是您可能需要在训练过程中调整的参数。

另一个您可能遇到的常见优化器是 *RMSProp*（均方根传播）。同样，您不需要太多调整这个优化器的参数，但值得阅读[Keras文档](https://keras.io/optimizers)以了解每个参数的作用。

我们将损失函数和优化器一起传递给模型的 `compile` 方法，还有一个 `metrics` 参数，我们可以在训练过程中指定任何额外的指标，如准确率。

## 训练模型

到目前为止，我们还没有向模型展示任何数据。我们只是设置了架构并使用损失函数和优化器编译了模型。

要针对数据训练模型，我们只需调用 `fit` 方法，如[示例2-8](#training-mlp)所示。

##### 示例2-8\. 调用 `fit` 方法来训练模型

```py
model.fit(x_train ![1](Images/1.png)
          , y_train ![2](Images/2.png)
          , batch_size = 32 ![3](Images/3.png)
          , epochs = 10 ![4](Images/4.png)
          , shuffle = True ![5](Images/5.png)
          )
```

[![1](Images/1.png)](#co_deep_learning_CO2-1)

原始图像数据。

[![2](Images/2.png)](#co_deep_learning_CO2-2)

独热编码的类标签。

[![3](Images/3.png)](#co_deep_learning_CO2-3)

`batch_size` 确定每个训练步骤将传递给网络多少观察值。

[![4](Images/4.png)](#co_deep_learning_CO2-4)

`epochs` 确定网络将被展示完整训练数据的次数。

[![5](Images/5.png)](#co_deep_learning_CO2-5)

如果 `shuffle = True`，每个训练步骤将从训练数据中随机抽取批次而不重复。

这将开始训练一个深度神经网络，以预测来自CIFAR-10数据集的图像的类别。训练过程如下。

首先，网络的权重被初始化为小的随机值。然后网络执行一系列训练步骤。在每个训练步骤中，通过网络传递一个 *batch* 图像，并将错误反向传播以更新权重。`batch_size` 确定每个训练步骤批次中有多少图像。批量大小越大，梯度计算越稳定，但每个训练步骤越慢。

###### 提示

使用整个数据集在每个训练步骤中计算梯度将耗费太多时间和计算资源，因此通常使用32到256之间的批量大小。现在推荐的做法是随着训练的进行增加批量大小。^([4](ch02.xhtml#idm45387032068928))

这将持续到数据集中的所有观察值都被看到一次。这完成了第一个 *epoch*。然后数据再次以批次的形式通过网络，作为第二个epoch的一部分。这个过程重复，直到指定的epoch数已经过去。

在训练过程中，Keras会输出过程的进展，如[图2-7](#first_nn_fit)所示。我们可以看到训练数据集已经被分成了1,563批次（每批包含32张图片），并且已经被展示给网络10次（即10个epochs），每批大约需要2毫秒的时间。分类交叉熵损失从1.8377下降到1.3696，导致准确率从第一个epoch后的33.69%增加到第十个epoch后的51.67%。

![](Images/gdl2_0207.png)

###### 图2-7\. `fit` 方法的输出

## 评估模型

我们知道模型在训练集上的准确率为51.9%，但它在从未见过的数据上表现如何？

为了回答这个问题，我们可以使用Keras提供的`evaluate`方法，如[示例2-9](#evaluate-mlp)所示。

##### 示例2-9。在测试集上评估模型性能

```py
model.evaluate(x_test, y_test)
```

[图2-8](#first_nn_evaluate)显示了这种方法的输出。

![](Images/gdl2_0208.png)

###### 图2-8。`evaluate`方法的输出

输出是我们正在监控的指标列表：分类交叉熵和准确率。我们可以看到，即使在它从未见过的图像上，模型的准确率仍然是49.0%。请注意，如果模型是随机猜测的，它将达到大约10%的准确率（因为有10个类别），因此49.0%是一个很好的结果，考虑到我们使用了一个非常基本的神经网络。

我们可以使用`predict`方法查看测试集上的一些预测，如[示例2-10](#predict-mlp)所示。

##### 示例2-10。使用`predict`方法查看测试集上的预测

```py
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog'
                   , 'frog', 'horse', 'ship', 'truck'])

preds = model.predict(x_test) ![1](Images/1.png)
preds_single = CLASSES[np.argmax(preds, axis = -1)] ![2](Images/2.png)
actual_single = CLASSES[np.argmax(y_test, axis = -1)]
```

[![1](Images/1.png)](#co_deep_learning_CO3-1)

`preds`是一个形状为`[10000, 10]`的数组，即每个观测的10个类别概率的向量。

[![2](Images/2.png)](#co_deep_learning_CO3-2)

我们将这个概率数组转换回一个单一的预测，使用`numpy`的`argmax`函数。这里，`axis = -1`告诉函数将数组折叠到最后一个维度（类别维度），因此`preds_single`的形状为`[10000, 1]`。

我们可以使用[示例2-11](#display-mlp)中的代码查看一些图像以及它们的标签和预测。如预期的那样，大约一半是正确的。

##### 示例2-11。显示MLP的预测与实际标签

```py
import matplotlib.pyplot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10
       , ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10
        , ha='center', transform=ax.transAxes)
    ax.imshow(img)
```

[图2-9](#first_nn_preds)显示了模型随机选择的一些预测，以及真实标签。

![](Images/gdl2_0209.png)

###### 图2-9。模型进行的一些预测，以及实际标签

恭喜！您刚刚使用Keras构建了一个多层感知器，并用它对新数据进行了预测。即使这是一个监督学习问题，但当我们在未来的章节中构建生成模型时，本章的许多核心思想（如损失函数、激活函数和理解层形状）仍然非常重要。接下来，我们将探讨通过引入一些新的层类型来改进这个模型的方法。

# 卷积神经网络（CNN）

我们的网络尚未表现得像它可能表现得那样好的原因之一是网络中没有考虑输入图像的空间结构。事实上，我们的第一步是将图像展平为一个单一向量，以便我们可以将其传递给第一个`Dense`层！

为了实现这一点，我们需要使用*卷积层*。

## 卷积层

首先，我们需要了解在深度学习背景下*卷积*的含义。

[图2-10](#simple_conv)显示了一个灰度图像的两个不同的3×3×1部分，与一个3×3×1*滤波器*（或*核心*）进行卷积。卷积是通过将滤波器逐像素地与图像部分相乘，并将结果求和来执行的。当图像部分与滤波器紧密匹配时，输出更为正向，当图像部分与滤波器的反向匹配时，输出更为负向。顶部示例与滤波器强烈共振，因此产生一个较大的正值。底部示例与滤波器的共振不大，因此产生一个接近零的值。

![](Images/gdl2_0210.png)

###### 图2-10。应用于灰度图像两个部分的3×3卷积滤波器

如果我们将滤波器从左到右和从上到下移动到整个图像上，并记录卷积输出，我们将获得一个新的数组，根据滤波器中的值选择输入的特定特征。例如，图2-11显示了突出显示水平和垂直边缘的两个不同滤波器。

# 运行此示例的代码

您可以在位于书籍存储库中的*notebooks/02_deeplearning/02_cnn/convolutions.ipynb*的Jupyter笔记本中手动查看这个卷积过程。

![](Images/gdl2_0211.png)

###### 图2-11。应用于灰度图像的两个卷积滤波器

卷积层只是一组滤波器，其中存储在滤波器中的值是通过训练的神经网络学习的权重。最初这些是随机的，但逐渐滤波器调整它们的权重以开始选择有趣的特征，如边缘或特定的颜色组合。

在Keras中，`Conv2D`层将卷积应用于具有两个空间维度（如图像）的输入张量。例如，[示例2-12](#conv-layer)中显示的代码构建了一个具有两个滤波器的卷积层，以匹配[图2-11](#conv_layer_2d)中的示例。

##### 示例2-12。应用于灰度输入图像的`Conv2D`层

```py
from tensorflow.keras import layers

input_layer = layers.Input(shape=(64,64,1))
conv_layer_1 = layers.Conv2D(
    filters = 2
    , kernel_size = (3,3)
    , strides = 1
    , padding = "same"
    )(input_layer)
```

接下来，让我们更详细地看一下`Conv2D`层的两个参数——`strides`和`padding`。

### 步幅

`strides`参数是层用来在输入上移动滤波器的步长。增加步长会减小输出张量的大小。例如，当`strides = 2`时，输出张量的高度和宽度将是输入张量大小的一半。这对于通过网络传递时减小张量的空间大小，同时增加通道数量是有用的。

### 填充

`padding = "same"`输入参数使用零填充输入数据，以便当`strides = 1`时，从层的输出大小与输入大小完全相同。

图2-12显示了一个3×3的卷积核在一个5×5的输入图像上进行传递，其中`padding = "same"`和`strides = 1`。这个卷积层的输出大小也将是5×5，因为填充允许卷积核延伸到图像的边缘，使其在两个方向上都适合五次。没有填充，卷积核只能在每个方向上适合三次，从而给出一个3×3的输出大小。

![](Images/gdl2_0212.png)

###### 图2-12。一个3×3×1的卷积核（灰色）在一个5×5×1的输入图像（蓝色）上进行传递，其中`padding = "same"`和`strides = 1`，生成5×5×1的输出（绿色）（来源：Dumoulin和Visin，2018）

设置`padding = "same"`是一种确保您能够轻松跟踪张量大小的好方法，因为它通过许多卷积层时。具有`padding = "same"`的卷积层的输出形状是：

<math alttext="left-parenthesis StartFraction i n p u t h e i g h t Over s t r i d e EndFraction comma StartFraction i n p u t w i d t h Over s t r i d e EndFraction comma f i l t e r s right-parenthesis" display="block"><mrow><mo>(</mo> <mfrac><mrow><mi>i</mi><mi>n</mi><mi>p</mi><mi>u</mi><mi>t</mi><mi>h</mi><mi>e</mi><mi>i</mi><mi>g</mi><mi>h</mi><mi>t</mi></mrow> <mrow><mi>s</mi><mi>t</mi><mi>r</mi><mi>i</mi><mi>d</mi><mi>e</mi></mrow></mfrac> <mo>,</mo> <mfrac><mrow><mi>i</mi><mi>n</mi><mi>p</mi><mi>u</mi><mi>t</mi><mi>w</mi><mi>i</mi><mi>d</mi><mi>t</mi><mi>h</mi></mrow> <mrow><mi>s</mi><mi>t</mi><mi>r</mi><mi>i</mi><mi>d</mi><mi>e</mi></mrow></mfrac> <mo>,</mo> <mi>f</mi> <mi>i</mi> <mi>l</mi> <mi>t</mi> <mi>e</mi> <mi>r</mi> <mi>s</mi> <mo>)</mo></mrow></math>

### 堆叠卷积层

`Conv2D`层的输出是另一个四维张量，现在的形状是`(batch_size, height, width, filters)`，因此我们可以将`Conv2D`层堆叠在一起，以增加神经网络的深度并使其更强大。为了演示这一点，让我们想象我们正在将`Conv2D`层应用于CIFAR-10数据集，并希望预测给定图像的标签。请注意，这一次，我们不是一个输入通道（灰度），而是三个（红色、绿色和蓝色）。

[示例2-13](#conv-network)展示了如何构建一个简单的卷积神经网络，我们可以训练它成功完成这项任务。

##### 示例2-13。使用Keras构建卷积神经网络模型的代码

```py
from tensorflow.keras import layers, models

input_layer = layers.Input(shape=(32,32,3))
conv_layer_1 = layers.Conv2D(
    filters = 10
    , kernel_size = (4,4)
    , strides = 2
    , padding = 'same'
    )(input_layer)
conv_layer_2 = layers.Conv2D(
    filters = 20
    , kernel_size = (3,3)
    , strides = 2
    , padding = 'same'
    )(conv_layer_1)
flatten_layer = layers.Flatten()(conv_layer_2)
output_layer = layers.Dense(units=10, activation = 'softmax')(flatten_layer)
model = models.Model(input_layer, output_layer)
```

这段代码对应于[图2-13](#conv_2d_complex)中显示的图表。

![](Images/gdl2_0213.png)

###### 图2-13。卷积神经网络的图表

请注意，现在我们正在处理彩色图像，第一个卷积层中的每个滤波器的深度为3，而不是1（即每个滤波器的形状为4×4×3，而不是4×4×1）。这是为了匹配输入图像的三个通道（红色、绿色、蓝色）。同样的想法也适用于第二个卷积层中的深度为10的滤波器，以匹配第一个卷积层输出的10个通道。

###### 提示

一般来说，层中滤波器的深度总是等于前一层输出的通道数。

### 检查模型

从一个卷积层到下一个卷积层，数据流经过时张量形状如何变化真的很有启发性。我们可以使用`model.summary()`方法检查张量在网络中传递时的形状（[表2-2](#conv_net_example_summary)）。

表2-2\. CNN模型摘要

| 层（类型） | 输出形状 | 参数数量 |
| --- | --- | --- |
| 输入层 | (None, 32, 32, 3) | 0 |
| Conv2D | (None, 16, 16, 10) | 490 |
| Conv2D | (None, 8, 8, 20) | 1,820 |
| Flatten | (None, 1280) | 0 |
| Dense | (None, 10) | 12,810 |
| 总参数 | 15,120 |
| 可训练参数 | 15,120 |
| 不可训练参数 | 0 |

让我们逐层走过我们的网络，注意张量的形状：

1.  输入形状为`(None, 32, 32, 3)`—Keras使用`None`表示我们可以同时通过网络传递任意数量的图像。由于网络只是执行张量代数运算，我们不需要单独通过网络传递图像，而是可以一起作为批次传递它们。

1.  第一个卷积层中每个滤波器的形状是4×4×3。这是因为我们选择每个滤波器的高度和宽度为4（`kernel_size=(4,4)`），并且在前一层中有三个通道（红色、绿色和蓝色）。因此，该层中的参数（或权重）数量为（4×4×3+1）×10=490，其中+1是由于每个滤波器附加了一个偏置项。每个滤波器的输出将是滤波器权重和它所覆盖的图像的4×4×3部分的逐像素乘积。由于`strides=2`和`padding="same"`，输出的宽度和高度都减半为16，由于有10个滤波器，第一层的输出是一批张量，每个张量的形状为`[16,16,10]`。

1.  在第二个卷积层中，我们选择滤波器为3×3，它们现在的深度为10，以匹配前一层中的通道数。由于这一层中有20个滤波器，这给出了总参数（权重）数量为（3×3×10+1）×20=1,820。同样，我们使用`strides=2`和`padding="same"`，所以宽度和高度都减半。这给出了一个总体输出形状为`(None, 8, 8, 20)`。

1.  现在我们使用Keras的`Flatten`层展平张量。这会产生一组8×8×20=1,280个单元。请注意，在`Flatten`层中没有需要学习的参数，因为该操作只是对张量进行重组。

1.  最后，我们将这些单元连接到一个具有softmax激活函数的10单元`Dense`层，表示10类分类任务中每个类别的概率。这会创建额外的1,280×10=12,810个参数（权重）需要学习。

这个例子演示了如何将卷积层链接在一起创建卷积神经网络。在我们看到这与我们密集连接的神经网络在准确性上的比较之前，我们将研究另外两种也可以提高性能的技术：批量归一化和dropout。

## 批量归一化

训练深度神经网络时的一个常见问题是确保网络的权重保持在合理范围内的数值范围内 - 如果它们开始变得过大，这表明您的网络正在遭受所谓的*梯度爆炸*问题。当错误向后传播通过网络时，早期层中梯度的计算有时可能会呈指数增长，导致权重值出现剧烈波动。

###### 警告

如果您的损失函数开始返回`NaN`，那么很有可能是您的权重已经变得足够大，导致溢出错误。

这并不一定会立即发生在您开始训练网络时。有时候，它可能在几个小时内愉快地训练，突然损失函数返回`NaN`，您的网络就爆炸了。这可能非常恼人。为了防止这种情况发生，您需要了解梯度爆炸问题的根本原因。

### 协变量转移

将输入数据缩放到神经网络的一个原因是确保在前几次迭代中稳定地开始训练。由于网络的权重最初是随机化的，未缩放的输入可能会导致立即产生激活值过大，从而导致梯度爆炸。例如，我们通常将像素值从0-255传递到输入层，而不是将这些值缩放到-1到1之间。

因为输入被缩放，自然地期望未来所有层的激活也相对缩放。最初可能是正确的，但随着网络训练和权重远离其随机初始值，这个假设可能开始破裂。这种现象被称为*协变量转移*。

# 协变量转移类比

想象一下，你正拿着一摞高高的书，突然被一阵风吹袭。你将书向与风相反的方向移动以补偿，但在这样做的过程中，一些书会移动，使得整个塔比以前稍微不稳定。最初，这没关系，但随着每阵风，这摞书变得越来越不稳定，直到最终书移动得太多，整摞书倒塌。这就是协变量转移。

将这与神经网络联系起来，每一层就像堆叠中的一本书。为了保持稳定，当网络更新权重时，每一层都隐含地假设其来自下一层的输入分布在迭代中大致保持一致。然而，由于没有任何东西可以阻止任何激活分布在某个方向上发生显着变化，这有时会导致权重值失控和网络整体崩溃。

### 使用批量归一化进行训练

*批量归一化*是一种极大地减少这个问题的技术。解决方案出奇地简单。在训练期间，批量归一化层计算每个输入通道在批处理中的均值和标准差，并通过减去均值并除以标准差来进行归一化。然后，每个通道有两个学习参数，即缩放（gamma）和移位（beta）。输出只是归一化的输入，由gamma缩放并由beta移位。[图2-14](#batch_norm)展示了整个过程。

![](Images/gdl2_0214.png)

###### 图2-14。批量归一化过程（来源：[Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167)）^([6](ch02.xhtml#idm45387025136368))

我们可以在密集层或卷积层之后放置批量归一化层来归一化输出。

###### 提示

参考我们之前的例子，这有点像用一小组可调节弹簧连接书层，以确保它们的位置随时间不会发生明显的整体移动。

### 使用批量归一化进行预测

您可能想知道这个层在预测时是如何工作的。在预测时，我们可能只想预测单个观测值，因此没有*批次*可以计算平均值和标准差。为了解决这个问题，在训练期间，批归一化层还会计算每个通道的平均值和标准差的移动平均值，并将这个值作为该层的一部分存储起来，以便在测试时使用。

批归一化层中包含多少参数？对于前一层中的每个通道，需要学习两个权重：比例（gamma）和偏移（beta）。这些是*可训练*参数。移动平均值和标准差也需要针对每个通道进行计算，但由于它们是从通过该层的数据派生而来，而不是通过反向传播进行训练，因此被称为*不可训练*参数。总共，这为前一层中的每个通道提供了四个参数，其中两个是可训练的，两个是不可训练的。

在Keras中，`BatchNormalization`层实现了批归一化功能，如[例2-14](#batchnorm-layer)所示。

##### 例2-14\. Keras中的`BatchNormalization`层

```py
from tensorflow.keras import layers
layers.BatchNormalization(momentum = 0.9)
```

在计算移动平均值和移动标准差时，`momentum`参数是给予先前值的权重。

## Dropout

在备考考试时，学生通常会使用过去的试卷和样题来提高对学科材料的了解。一些学生试图记住这些问题的答案，但在考试中却因为没有真正理解学科内容而失败。最好的学生利用练习材料来进一步提高他们对学科的整体理解，这样当面对以前没有见过的新问题时，他们仍然能够正确回答。

相同的原则适用于机器学习。任何成功的机器学习算法必须确保它能泛化到未见过的数据，而不仅仅是*记住*训练数据集。如果一个算法在训练数据集上表现良好，但在测试数据集上表现不佳，我们称其为*过拟合*。为了解决这个问题，我们使用*正则化*技术，确保模型在开始过拟合时受到惩罚。

有许多方法可以对机器学习算法进行正则化，但对于深度学习来说，最常见的一种方法是使用*dropout*层。这个想法是由Hinton等人在2012年提出的^([7](ch02.xhtml#idm45387025089232))，并在2014年由Srivastava等人在一篇论文中提出^([8](ch02.xhtml#idm45387025086976))

Dropout层非常简单。在训练期间，每个dropout层从前一层中选择一组随机单元，并将它们的输出设置为0，如[图2-15](#dropout)所示。

令人难以置信的是，这个简单的添加通过确保网络不会过度依赖某些单元或单元组而大大减少了过拟合，这些单元或单元组实际上只是记住了训练集中的观察结果。如果我们使用dropout层，网络就不能太依赖任何一个单元，因此知识更均匀地分布在整个网络中。

![](Images/gdl2_0215.png)

###### 图2-15\. 一个dropout层

这使得模型在泛化到未见过的数据时更加出色，因为网络已经经过训练，即使在由于丢弃随机单元引起的陌生条件下，也能产生准确的预测。在dropout层内没有需要学习的权重，因为要丢弃的单元是随机决定的。在预测时，dropout层不会丢弃任何单元，因此整个网络用于进行预测。

# Dropout类比

回到我们的类比，这有点像数学学生练习过去试卷，其中随机选择了公式书中缺失的关键公式。通过这种方式，他们学会了通过对核心原则的理解来回答问题，而不是总是在书中相同的地方查找公式。当考试时，他们会发现更容易回答以前从未见过的问题，因为他们能够超越训练材料进行泛化。

Keras中的`Dropout`层实现了这种功能，`rate`参数指定了要从前一层中丢弃的单元的比例，如[示例2-15](#dropout-layer)所示。

##### 示例2-15\. Keras中的`Dropout`层

```py
from tensorflow.keras import layers
layers.Dropout(rate = 0.25)
```

由于密集层的权重数量较高，最容易过拟合，因此通常在密集层之后使用Dropout层，尽管也可以在卷积层之后使用。

###### 提示

批量归一化也被证明可以减少过拟合，因此许多现代深度学习架构根本不使用dropout，完全依赖批量归一化进行正则化。与大多数深度学习原则一样，在每种情况下都没有适用的黄金法则，唯一确定最佳方法的方式是测试不同的架构，看看哪种在保留数据集上表现最好。

## 构建CNN

您现在已经看到了三种新的Keras层类型：`Conv2D`、`BatchNormalization`和`Dropout`。让我们将这些部分组合成一个CNN模型，并看看它在CIFAR-10数据集上的表现。

# 运行此示例的代码

您可以在书籍存储库中名为*notebooks/02_deeplearning/02_cnn/cnn.ipynb*的Jupyter笔记本中运行以下示例。

我们将测试的模型架构显示在[示例2-16](#conv-network-2)中。

##### 示例2-16\. 使用Keras构建CNN模型的代码

```py
from tensorflow.keras import layers, models

input_layer = layers.Input((32,32,3))

x = layers.Conv2D(filters = 32, kernel_size = 3
	, strides = 1, padding = 'same')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Flatten()(x)

x = layers.Dense(128)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(rate = 0.5)(x)

output_layer = layers.Dense(10, activation = 'softmax')(x)

model = models.Model(input_layer, output_layer)
```

我们使用四个堆叠的`Conv2D`层，每个后面跟一个`BatchNormalization`和一个`LeakyReLU`层。在展平结果张量后，我们通过一个大小为128的`Dense`层，再次跟一个`BatchNormalization`和一个`LeakyReLU`层。紧接着是一个用于正则化的`Dropout`层，网络最后是一个大小为10的输出`Dense`层。

###### 提示

使用批量归一化和激活层的顺序是个人偏好的问题。通常情况下，批量归一化层放在激活层之前，但一些成功的架构会反过来使用这些层。如果选择在激活之前使用批量归一化，可以使用缩写 *BAD*（批量归一化，激活，然后是dropout）来记住顺序！

模型摘要显示在[表2-3](#cnn_model_summary)中。

表2-3\. CIFAR-10的CNN模型摘要

| 层（类型） | 输出形状 | 参数 # |
| --- | --- | --- |
| InputLayer | (None, 32, 32, 3) | 0 |
| Conv2D | (None, 32, 32, 32) | 896 |
| BatchNormalization | (None, 32, 32, 32) | 128 |
| LeakyReLU | (None, 32, 32, 32) | 0 |
| Conv2D | (None, 16, 16, 32) | 9,248 |
| BatchNormalization | (None, 16, 16, 32) | 128 |
| LeakyReLU | (None, 16, 16, 32) | 0 |
| Conv2D | (None, 16, 16, 64) | 18,496 |
| BatchNormalization | (None, 16, 16, 64) | 256 |
| LeakyReLU | (None, 16, 16, 64) | 0 |
| Conv2D | (None, 8, 8, 64) | 36,928 |
| BatchNormalization | (None, 8, 8, 64) | 256 |
| LeakyReLU | (None, 8, 8, 64) | 0 |
| Flatten | (None, 4096) | 0 |
| Dense | (None, 128) | 524,416 |
| BatchNormalization | (None, 128) | 512 |
| LeakyReLU | (None, 128) | 0 |
| Dropout | (None, 128) | 0 |
| Dense | (None, 10) | 1290 |
| 总参数 | 592,554 |
| 可训练参数 | 591,914 |
| 不可训练参数 | 640 |

###### 提示

在继续之前，请确保您能够手工计算每一层的输出形状和参数数量。这是一个很好的练习，可以证明您已经完全理解了每一层是如何构建的，以及它是如何与前一层连接的！不要忘记包括作为`Conv2D`和`Dense`层的一部分包含的偏置权重。

## 训练和评估CNN

我们编译和训练模型的方式与之前完全相同，并调用`evaluate`方法来确定其在留存集上的准确率（[图2-16](#cnn_model_evaluate)）。

![](Images/gdl2_0216.png)

###### 图2-16。CNN性能

正如您所看到的，这个模型现在的准确率达到了71.5%，比之前的49.0%有所提高。好多了！[图2-17](#cnn_preds)展示了我们新卷积模型的一些预测。

通过简单地改变模型的架构来包括卷积、批量归一化和丢弃层，已经实现了这一改进。请注意，我们新模型中的参数数量实际上比之前的模型更少，尽管层数要多得多。这表明了对模型设计进行实验和熟悉不同层类型如何利用优势的重要性。在构建生成模型时，更加重要的是要理解模型的内部工作原理，因为您最感兴趣的是网络的中间层，这些层捕捉了高级特征。

![](Images/gdl2_0217.png)

###### 图2-17。CNN预测

# 总结

本章介绍了构建深度生成模型所需的核心深度学习概念。我们首先使用Keras构建了一个多层感知器（MLP），并训练模型来预测来自CIFAR-10数据集的给定图像的类别。然后，我们通过引入卷积、批量归一化和丢弃层来改进这个架构，创建了一个卷积神经网络（CNN）。

从本章中需要牢记的一个非常重要的观点是，深度神经网络在设计上是完全灵活的，在模型架构方面实际上没有固定的规则。有指导方针和最佳实践，但您应该随意尝试不同层和它们出现的顺序。不要感到受限于仅使用您在本书或其他地方阅读过的架构！就像一个拥有一套积木的孩子一样，您的神经网络的设计仅受您自己想象力的限制。

在下一章中，我们将看到如何使用这些积木来设计一个可以生成图像的网络。

^([1](ch02.xhtml#idm45387028957520-marker)) Kaiming He等人，“用于图像识别的深度残差学习”，2015年12月10日，[*https://arxiv.org/abs/1512.03385*](https://arxiv.org/abs/1512.03385)。

^([2](ch02.xhtml#idm45387033163216-marker)) Alex Krizhevsky，“从微小图像中学习多层特征”，2009年4月8日，[*https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf*](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)。

^([3](ch02.xhtml#idm45387032147088-marker)) Diederik Kingma和Jimmy Ba，“Adam：一种随机优化方法”，2014年12月22日，[*https://arxiv.org/abs/1412.6980v8*](https://arxiv.org/abs/1412.6980v8)。

^([4](ch02.xhtml#idm45387032068928-marker)) Samuel L. Smith等人，“不要降低学习率，增加批量大小”，2017年11月1日，[*https://arxiv.org/abs/1711.00489*](https://arxiv.org/abs/1711.00489)。

^([5](ch02.xhtml#idm45387031545152-marker)) Vincent Dumoulin和Francesco Visin，“深度学习卷积算术指南”，2018年1月12日，[*https://arxiv.org/abs/1603.07285*](https://arxiv.org/abs/1603.07285)。

^([6](ch02.xhtml#idm45387025136368-marker)) Sergey Ioffe和Christian Szegedy，“批量归一化：通过减少内部协变量转移加速深度网络训练”，2015年2月11日，[*https://arxiv.org/abs/1502.03167*](https://arxiv.org/abs/1502.03167)。

^([7](ch02.xhtml#idm45387025089232-marker)) Hinton等人，“通过防止特征探测器的共适应来构建网络”，2012年7月3日，[*https://arxiv.org/abs/1207.0580*](https://arxiv.org/abs/1207.0580)。

^([8](ch02.xhtml#idm45387025086976-marker)) Nitish Srivastava等人，“Dropout：防止神经网络过拟合的简单方法”，*机器学习研究杂志* 15 (2014): 1929–1958，[*http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf*](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)。
