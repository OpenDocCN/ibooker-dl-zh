# 4 训练基础

本章涵盖了

+   前向传播和反向传播

+   分割数据集和预处理数据

+   使用验证数据监控过拟合

+   使用检查点和提前停止以实现更经济的训练

+   使用超参数与模型参数

+   训练对位置和尺度的不变性

+   组装和访问磁盘上的数据集

+   保存并恢复训练好的模型

本章涵盖了训练模型的基础知识。在 2019 年之前，大多数模型都是根据这一套基本步骤进行训练的。可以将本章视为一个基础。

在本章中，我们将介绍随着时间的推移通过实验和试错开发的方法、技术和最佳实践。我们将从回顾前向传播和反向传播开始。虽然这些概念和实践在深度学习之前就已经存在，但多年的改进使模型训练变得实用——特别是，在数据分割、喂入以及使用梯度下降在反向传播期间更新权重的方式。这些技术改进提供了训练模型到收敛的手段，即模型预测准确率达到平台期的点。

在数据预处理和增强中开发的其他训练技术旨在推动收敛到更高的平台，并帮助模型更好地泛化到模型未训练过的数据。进一步的改进继续通过超参数搜索和调整、检查点和提前停止，以及更高效地从磁盘存储中抽取数据格式和方法，使训练更加经济。所有这些技术相结合，使得深度学习在现实世界中的应用在计算和经济上都具有可行性。

## 4.1 前向传播和反向传播

让我们从监督训练的概述开始。在训练模型时，你将数据前向通过模型，并计算预测结果的错误程度——*损失*。然后，将损失反向传播以更新模型的参数，这就是模型正在学习的内容——参数的值。

在训练模型时，你从代表模型将部署的目标环境的训练数据开始。换句话说，这些数据是人群分布的采样分布。训练数据由示例组成。每个示例有两个部分：特征，也称为*独立变量*；以及相应的标签，也称为*因变量*。

标签也被称为*真实值*（“正确答案”）。我们的目标是训练一个模型，一旦部署并给出来自人群（模型之前从未见过的例子）的无标签示例，模型就能泛化到足以准确预测标签（“正确答案”）——监督学习。这一步被称为*推理*。

在训练过程中，我们通过输入层（也称为模型的**底部**）将训练数据的**批次**（也称为**样本**）输入到模型中。随着数据向前移动，经过模型各层的参数（权重和偏置）的转换，训练数据被转化为输出节点（也称为模型的**顶部**）。在输出节点，我们测量我们与“正确”答案的距离，这被称为**损失**。然后，我们将损失反向传播通过模型的各层，并更新参数以更接近下一个批次得到正确答案。

我们继续重复这个过程，直到达到**收敛**，这可以描述为“在这个训练运行中，我们已经达到了尽可能高的准确性。”

### 4.1.1 输入

**输入**是从训练数据中采样批次并通过模型前馈批次的过程，然后在输出处计算损失。一个批次可以是随机选择的一个或多个训练数据示例。

批次的大小通常是恒定的，这被称为（迷你）**批次大小**。所有训练数据被分成批次，通常每个示例只会出现在一个批次中。

所有训练数据被多次输入到模型中。每次我们输入整个训练数据，都称为一个**epoch**。每个 epoch 是批次的不同随机排列——也就是说，没有两个 epoch 具有相同的示例顺序——如图 4.1 所示。

![](img/CH04_F01_Ferlitsch.png)

图 4.1 在训练过程中，训练数据的迷你批次通过神经网络前馈。

### 4.1.2 反向传播

在本节中，我们探讨反向传播发现的的重要性以及它今天的应用。

背景

让我们回顾历史，了解反向传播对深度学习成功的重要性。在早期的神经网络中，如感知器和单层神经元，学术研究人员尝试了各种更新权重的方法以获得正确答案。

当他们只使用少量神经元和简单问题时，逻辑上的第一次尝试只是随机更新。最终，令人惊讶的是，随机猜测竟然有效。然而，这种方法并不适用于大量神经元（比如数千个）和实际应用；正确的随机猜测可能需要数百万年。

下一个逻辑步骤是使随机值与预测的偏差成正比。换句话说，偏差越大，随机值的范围就越大；偏差越小，随机值的范围就越小。不错——现在我们可能只需要数千年就能在实际应用中猜测正确的随机值。

最终，学术研究人员尝试了多层感知器（MLPs），但将随机值与它们偏离正确答案的程度（损失）成比例的技术并没有奏效。他们发现，当你有多个层次时，这种技术会产生左手（一层）抵消右手（另一层）工作的效果。

这些研究人员发现，虽然输出层权重的更新与预测中的损失相关，但早期层中权重的更新与下一层的更新相关。因此，形成了反向传播的概念。在此阶段，学术研究人员超越了仅使用随机分布来计算更新的方法。尝试了许多方法，但都没有改进，直到开发出一种更新权重的方法，不是基于下一层的变化量，而是基于变化率——因此发现了梯度下降技术并进行了发展。

基于批次的反向传播

在将每个训练数据批次正向通过模型并计算损失后，损失将通过模型进行反向传播。我们逐层更新模型的参数（权重和参数），从顶层（输出）开始，移动到底层（输入）。参数如何更新是损失、当前参数的值以及前一层所做的更新的组合。

实现这一般方法的通用方法是基于*梯度下降*。优化器是梯度下降的一种实现，其任务是更新参数以最小化后续批次上的损失（最大化接近正确答案的程度）。图 4.2 展示了这一过程。

![图片](img/CH04_F02_Ferlitsch.png)

图 4.2 从迷你批次计算出的损失通过反向传播；优化器更新权重以最小化下一批次的损失。

## 4.2 数据集划分

*数据集*是一组足够大且多样化的示例，足以代表所建模的总体（采样分布）。当一个数据集符合此定义，并且经过清理（无噪声），以及以适合机器学习训练的格式准备时，我们称之为*精心制作的数据集*。本书不涉及数据集清理的细节，因为它是一个庞大且多样化的主题，可以成为一本单独的书籍。我们在本书中涉及数据清理的各个方面，当相关时。

对于学术和研究目的，有许多精心制作的数据库可供使用。其中一些用于图像分类的知名数据库包括 MNIST（在第二章中介绍）、CIFAR-10/100、SVHN、Flowers 和 Cats vs. Dogs。MNIST 和 CIFAR-10/100（加拿大高级研究研究所）已内置到 TF.Keras 框架中。SVHN（街景房屋号码）、Flowers 和 Cats vs. Dogs 可通过 TensorFlow Datasets（TFDS）获得。在本节中，我们将使用这些数据集进行教程演示。

一旦你拥有了一个精心挑选的数据集，下一步就是将其分割成用于训练的示例和用于测试（也称为*评估*或*保留*）的示例。我们使用数据集中作为训练数据的部分来训练模型。如果我们假设训练数据是一个好的采样分布（代表总体分布），那么训练数据的准确性应该反映在将模型部署到现实世界中对模型在训练期间未见过的总体中的示例进行预测时的准确性。

但在我们部署模型之前，我们如何知道这是否正确呢？因此，测试（保留）数据的目的。我们在模型训练完成后，留出一部分数据集来测试，看看我们是否可以得到可比的准确性。

例如，假设我们完成训练后，在训练数据上达到了 99%的准确性，但在测试数据上只有 70%的准确性。出了些问题（例如，过拟合）。那么我们为训练和测试预留多少呢？从历史上看，经验法则一直是 80/20：80%用于训练，20%用于测试。但这已经改变了，但我们将从这一经验法则开始，并在后面的章节中讨论现代更新。

### 4.2.1 训练和测试集

重要的是，我们能够假设我们的数据集足够大，以至于如果我们将其分成 80%和 20%，并且示例是随机选择的，以便两个数据集都将成为代表总体分布的好的采样分布，那么模型在部署后将会做出预测（推理）。图 4.3 说明了这个过程。

![](img/CH04_F03_Ferlitsch.png)

图 4.3 在分割成训练和测试数据之前，训练数据首先被随机打乱。

让我们从使用精心挑选的数据集进行训练的逐步过程开始。作为第一步，我们导入精心挑选的 TF.Keras 内置 MNIST 数据集，如下面的代码所示。TF.Keras 内置数据集有一个`load_data()`方法。此方法将数据集加载到内存中，该数据集已经随机打乱并预先分割成训练和测试数据。训练和测试数据进一步被分成特征（在这种情况下是图像数据）和相应的标签（代表每个数字的数值 0 到 9）。将训练和测试的特征和标签分别称为`(x_train, y_train)`和`(x_test, y_test)`是一种常见的约定：

```
from tensorflow.keras.datasets import mnist                   ❶

(x_train, y_train), (x_test, y_test) = mnist.load_data()      ❷
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)yp
```

❶ MNIST 是该框架中的一个内置数据集。

❷ 内置数据集会自动随机打乱并预先分割成训练和测试数据。

MNIST 数据集包含 60,000 个训练样本和 10,000 个测试样本，十个数字 0 到 9 的分布均匀（平衡）。每个示例由一个 28×28 像素的灰度图像（单通道）组成。从以下输出中，你可以看到训练数据`(x_train, y_train)`由 60,000 个大小为 28×28 的图像和相应的 60,000 个标签组成，而测试数据`(x_test, y_test)`由 10,000 个示例和标签组成：

```
(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)
```

### 4.2.2 独热编码

让我们构建一个简单的 DNN 来训练我们的精选数据集。在下一个代码示例中，我们首先通过使用`Flatten`层将 28×28 图像输入展平为 1D 向量，然后是两个各有 512 个节点的隐藏`Dense()`层，每个层使用`relu`激活函数的约定。最后，输出层是一个有 10 个节点的`Dense`层，每个数字一个节点。由于这是一个多类分类器，输出层的激活函数是`softmax`。

接下来，我们通过使用`categorical_crossentropy`作为损失函数和`adam`作为优化器来编译模型，以符合多类分类器的约定：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))         ❶
model.add(Dense(512, activation='relu'))         ❷
model.add(Dense(512, activation='relu'))         ❸
model.add(Dense(10, activation='softmax'))       ❹
model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['acc'])
```

❶ 将 2D 灰度图像展平为 1D 向量以供 DNN 使用

❷ DNN 的实际输入层，一旦图像被展平

❸ 一个隐藏层

❹ DNN 的输出层

使用此数据集训练此模型的最基本方法是使用`fit()`方法。我们将传递训练数据`(x_train, y_train)`作为参数。我们将剩余的关键字参数设置为它们的默认值：

```
model.fit(x_train, y_train)
```

当你运行前面的代码时，你会看到一个错误信息：

```
ValueError: You are passing a target array of shape (60000, 1) while using
as loss 'categorical_crossentropy'. 'categorical_crossentropy' expects 
targets to be binary matrices (1s and 0s) of shape (samples, classes).
```

发生了什么问题？这是与我们选择的损失函数有关的问题。它将比较每个输出节点与相应的输出期望之间的差异。例如，如果答案是数字 3，我们需要一个 10 个元素的向量（每个数字一个元素），在 3 的索引处有一个 1（100%的概率），在其余索引处有 0（0%的概率）。在这种情况下，我们需要将标量值标签转换为在相应索引处有 1 的 10 个元素的向量。这被称为*独热编码*，如图 4.4 所示。

![](img/CH04_F04_Ferlitsch.png)

图 4.4 独热编码标签的大小与输出类别的数量相同。

让我们通过首先从 TF.Keras 导入`to_categorical()`函数，然后使用它将标量值标签转换为独热编码标签来修复我们的示例。注意，我们将值 10 传递给`to_categorical()`，以指示独热编码标签的大小（类别数量）：

```
from tensorflow.keras.utils import to_categorical    ❶
y_train = to_categorical(y_train, 10)                ❷
y_test = to_categorical(y_test, 10)                  ❷

model.fit(x_train, y_train)
```

❶ 使用方法进行独热编码

❷ 对训练和测试标签进行独热编码

现在你运行这个，你的输出将看起来像这样：

```
60000/60000 [==============================] - 5s 81us/sample - loss: 
1.3920 - acc: 0.9078                                                    ❶
```

❶ 训练数据上的准确率刚好超过 90%。

这样做是有效的，我们在训练数据上达到了 90%的准确率——但我们还可以简化这一步骤。`compile()`方法内置了 one-hot 编码。要启用它，我们只需将损失函数从`categorical_crossentropy`更改为`sparse_categorical_crossentropy`。在此模式下，损失函数将接收标签作为标量值，并在执行交叉熵损失计算之前动态地将它们转换为 one-hot 编码的标签。

我们在以下示例中这样做，并且还设置关键字参数`epoch`为 10，以便将整个训练数据输入模型 10 次：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()     ❶

model.fit(x_train, y_train, epochs=10)                       ❷
```

❶ 将 MNIST 数据集加载到内存中

❷ 对 MNIST 模型进行 10 个 epoch 的训练

在第 10 个 epoch 后，你应该看到训练数据上的准确率大约为 97%：

```
Epoch 10/10
60000/60000 [==============================] - 5s 83us/sample - loss: 
0.0924 - acc: 0.9776
```

## 4.3 数据正则化

我们可以进一步改进这一点。通过`mnist()`模块加载的图像数据是原始格式；每个图像是一个 28 × 28 的整数值矩阵，范围从 0 到 255。如果你检查训练模型内的参数（权重和偏差），它们是非常小的数字，通常从-1 到 1。通常，当数据通过层前馈并通过一层参数矩阵乘以下一层的参数时，结果是一个非常小的数字。

我们前面示例的问题在于输入值相当大（高达 255），这将在通过层乘法时产生很大的初始数值。这将导致参数学习它们的最佳值需要更长的时间——如果它们能学习的话。

### 4.3.1 正则化

我们可以通过将输入值压缩到更小的范围来增加参数学习最优值的速度，并提高我们的收敛概率（随后讨论）。一种简单的方法是将它们按比例压缩到 0 到 1 的范围。我们可以通过将每个值除以 255 来实现这一点。

在下面的代码中，我们添加了通过将每个像素值除以 255 来规范化输入数据的步骤。`load_data()`函数以 NumPy 格式将数据集加载到内存中。*NumPy*是一个用 C 语言编写并带有 Python 包装器（CPython）的高性能数组处理模块，在模型训练期间，当整个训练数据集都在内存中时，它非常高效。第十三章涵盖了当训练数据集太大而无法放入内存时的方法和格式。

一个*NumPy 数组*是一个实现算术运算符多态性的类对象。在我们的例子中，我们展示了单个除法操作`(x_train / 255.0)`。除法运算符被 NumPy 数组覆盖，并实现了广播操作——这意味着数组中的每个元素都将除以 255.0。

默认情况下，NumPy 使用双精度（64 位）进行浮点运算。默认情况下，TF.Keras 模型中的参数是单精度浮点数（32 位）。为了提高效率，作为最后一步，我们使用 NumPy 的`astype()`方法将广播除法的结果转换为 32 位。如果我们没有进行转换，从输入层到输入层的初始矩阵乘法将需要双倍的机器周期（64×32 而不是 32×32）：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255.0).astype(np.float32)              ❶
x_test  = (x_test  / 255.0).astype(np.float32)              ❶

model.fit(x_train, y_train, epochs=10)
```

❶ 将像素数据从 0 标准化到 1

下面的输出是运行前面代码的结果。让我们将输出与先前非标准化输入的标准化输入进行比较。在先前的输入中，我们在第 10 个 epoch 后达到了 97%的准确率。在我们的标准化输入中，我们只需第二个 epoch 就达到了相同的准确率，在第 10 个 epoch 后几乎达到了 99.5%的准确率。因此，当我们标准化输入数据时，我们学得更快，更准确：

```
...
Epoch 2/10
60000/60000 [==============================] - 5s 84us/sample - loss: 
0.0808 - acc: 0.9744
...
Epoch 10/10
60000/60000 [==============================] - 5s 81us/sample - loss: 
0.0187 - acc: 0.9943
```

现在，让我们通过在测试（保留）数据上使用`evaluate()`方法来评估我们的模型，看看模型在训练期间从未见过的数据上的表现如何。`evaluate()`方法在推理模式下操作：测试数据被正向传递通过模型进行预测，但没有反向传播。模型的参数不会被更新。最后，`evaluate()`将输出损失和整体准确率：

```
model.evaluate(x_test, y_test)
```

在以下输出中，我们看到准确率约为 98%，与训练准确率 99.5%相比。这是预期的。在训练过程中总会发生一些过拟合。我们寻找的是训练和测试之间非常小的差异，在这种情况下大约是 1.5%：

```
10000/10000 [==============================] - 0s 23us/sample - loss:
0.0949 - acc: 0.9790
```

### 4.3.2 标准化

除了前面示例中使用的归一化之外，还有许多方法可以压缩输入数据。例如，一些机器学习从业者更喜欢将输入值压缩在-1 和 1 之间（而不是 0 和 1），这样值就集中在 0。以下代码是一个示例实现，它将每个元素除以最大值的一半（在这个例子中是 127.5），然后从结果中减去 1：

```
x_train = ((x_train / 127.5) - 1).astype(np.float32)
```

将值压缩在-1 和 1 之间是否比压缩在 0 和 1 之间产生更好的结果？我在研究文献或我的个人经验中都没有看到任何表明有差异的内容。

这种方法和之前的方法不需要对输入数据进行任何预分析，除了知道最大值。另一种称为*标准化*的技术被认为可以产生更好的结果。然而，它需要对整个输入数据进行预分析（扫描）以找到其平均值和标准差。然后，你将数据中心化在输入数据全分布的平均值处，并将值压缩在+/-一个标准差之间。以下代码实现了当输入数据作为 NumPy 多维数组存储在内存中时的标准化，使用了 NumPy 方法`np.mean()`和`np.std()`：

```
import numpy as np
mean = np.mean(x_train)                                 ❶
std = np.std(x_train)                                   ❷
x_train = ((x_train - mean) / std).astype(np.float32)   ❸
```

❶ 计算像素数据的平均值

❷ 计算像素数据的标准差

❸ 使用均值和标准差对像素数据进行标准化

## 4.4 验证和过拟合

本节演示了一个过拟合的案例，然后展示了如何在训练过程中检测过拟合以及我们可能如何解决这个问题。让我们重新回顾一下*过拟合*的含义。通常，为了获得更高的准确率，我们会构建更大和更大的模型。一个后果是模型可以死记硬背一些或所有示例。模型学习的是示例，而不是从示例中学习泛化，以准确预测训练过程中从未见过的示例。在极端情况下，一个模型可以达到 100%的训练准确率，但在测试（对于 10 个类别，那就是 10%的准确率）时具有随机准确率。

### 4.4.1 验证

假设训练模型需要几个小时。你真的想等到训练结束再在测试数据上测试，以了解模型是否过拟合吗？当然不想。相反，我们留出一小部分训练数据，我们称之为*验证数据*。

我们不使用验证数据来训练模型。相反，在每个 epoch 之后，我们使用验证数据来估计测试数据的可能结果。像测试数据一样，验证数据通过模型前馈（推理模式）而不更新模型的参数，我们测量损失和准确率。图 4.5 描述了此过程。

![](img/CH04_F05_Ferlitsch.png)

图 4.5 在每个 epoch，使用验证数据来估计测试数据的可能准确率。

如果数据集非常小，并且使用更少的数据进行训练会产生负面影响，我们可以使用*交叉验证*。不是一开始就留出一部分模型永远不会训练的训练数据，而是在每个 epoch 进行随机分割。在每个 epoch 的开始，随机选择验证示例，并在此 epoch 中不用于训练，而是用于验证测试。但由于选择是随机的，一些或所有示例将出现在其他 epoch 的训练数据中。今天的 datasets 都很大，所以很少需要这种技术。图 4.6 说明了数据集的交叉验证分割。

![](img/CH04_F06_Ferlitsch.png)

图 4.6 在每个 epoch，随机选择一个折作为验证数据。

接下来，我们将训练一个简单的 CNN 来对 CIFAR-10 数据集中的图像进行分类。我们的数据集是此小型图像数据集的一个子集，大小为 32 × 32 × 3。它包含 60,000 个训练图像和 10,000 个测试图像，涵盖了 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。

在我们的简单 CNN 中，我们有一个 32 个滤波器的 3 × 3 核大小的卷积层，后面跟着一个步长最大池化层。然后输出被展平并传递到最后一个输出密集层。图 4.7 说明了这个过程。

![](img/CH04_F07_Ferlitsch.png)

图 4.7 用于分类 CIFAR-10 图像的简单卷积神经网络

这是训练我们的简单 CNN 的代码：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import numpy as np

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test  / 255.0).astype(np.float32)

model.fit(x_train, y_train, epochs=15, validation_split=0.1)    ❶
```

❶ 使用 10%的训练数据用于验证——未进行训练

在这里，我们已将关键字参数 `validation_split=0.1` 添加到 `fit()` 方法中，以便在每个 epoch 之后为验证测试保留 10%的训练数据。

以下是在运行 15 个 epochs 后的输出。你可以看到，在第 4 个 epoch 之后，训练和评估准确率基本上是相同的。但在第 5 个 epoch 之后，我们开始看到它们开始分散（65%对 61%）。到第 15 个 epoch 时，分散非常大（74%对 63%）。我们的模型显然在第 5 个 epoch 开始过拟合：

```
Train on 45000 samples, validate on 5000 samples
... 
Epoch 4/15
45000/45000 [==============================] - 8s 184us/sample - loss: 1.0444 
➥ - acc: 0.6386 - val_loss: 1.0749 - val_acc: 0.6374                      ❶
Epoch 5/15
45000/45000 [==============================] - 9s 192us/sample - loss: 0.9923 
➥ - acc: 0.6587 - val_loss: 1.1099 - val_acc: 0.6182                      ❷
...
Epoch 15/15
45000/45000 [==============================] - 8s 180us/sample - loss: 0.7256 
➥ - acc: 0.7498 - val_loss: 1.1019 - val_acc: 0.6382                      ❸
```

❶ 在第 4 个 epoch 之后，训练数据和验证数据的准确率大致相同。

❷ 在第 5 个 epoch 之后，训练数据和验证数据之间的准确率开始分散。

❸ 在第 15 个 epoch 之后，训练数据和验证数据之间的准确率相差甚远。

现在我们来让模型不要过度拟合示例，而是从它们中泛化。正如前面章节所讨论的，我们希望在训练期间添加一些正则化——一些噪声——这样模型就不能死记硬背训练示例。在这个代码示例中，我们通过在最终密集层之前添加 50%的 dropout 来修改我们的模型。由于 dropout 会减慢我们的学习（因为遗忘），我们将 epochs 的数量增加到 20：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout
import numpy as np

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten(input_shape=(28, 28)))
model.add(Dropout(0.5))                          ❶
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test  / 255.0).astype(np.float32)

model.fit(x_train, y_train, epochs=20, validation_split=0.1)
```

❶ 向训练中添加噪声以防止过拟合

从以下输出中我们可以看到，虽然达到可比的训练准确率需要更多的 epochs，但训练和测试准确率是可比的。因此，模型正在学习泛化而不是死记硬背训练示例：

```
Epoch 18/20
45000/45000 [==============================] - 18s 391us/sample - loss: 
➥ 1.0029 - acc: 0.6532 - val_loss: 1.0069 - val_acc: 0.6600              ❶
Epoch 19/20
45000/45000 [==============================] - 17s 377us/sample - loss: 
➥ 0.9975 - acc: 0.6538 - val_loss: 1.0388 - val_acc: 0.6478              ❶
Epoch 20/20
45000/45000 [==============================] - 17s 381us/sample - loss: 
➥ 0.9891 - acc: 0.6568 - val_loss: 1.0562 - val_acc: 0.6502              ❶
```

❶ 通过使用 dropout 添加噪声，可以保持训练和验证准确率不偏离。

### 4.4.2 损失监控

到目前为止，我们一直专注于准确率。你看到的另一个输出指标是训练和验证数据批次的平均损失。理想情况下，我们希望看到每个 epoch 准确率的持续增加。但我们也可能看到一系列的 epochs，其中准确率持平或甚至波动±一小部分。

重要的是我们看到损失持续下降。在这种情况下，平台期或波动发生是因为我们接近或悬浮在线性分离的线上，或者还没有完全越过一条线，但随着损失的下降，我们正在接近。

让我们从另一个角度来观察。假设你正在构建一个用于区分狗和猫的分类器。在分类层上，你有两个输出节点：一个用于猫，一个用于狗。假设在某个特定的批次中，当模型错误地将狗分类为猫时，输出值（置信度）为猫 0.6，狗 0.4。在随后的批次中，当模型再次将狗错误分类为猫时，输出值变为 0.55（猫）和 0.45（狗）。这些值现在更接近真实值，因此损失正在减少，但它们仍未通过 0.5 的阈值，因此精度尚未改变。然后假设在另一个随后的批次中，狗图像的输出值为 0.49（猫）和 0.51（狗）；损失进一步减少，因为我们越过了 0.5 的阈值，精度有所上升。

### 4.4.3 使用层深入探索

如前几章所述，仅仅通过增加层数来深入探索，可能会导致模型不稳定，而没有解决像身份链接和批量归一化等技术问题。例如，我们进行矩阵乘法的许多值都是小于 1 的小数。乘以两个小于 1 的数，你会得到一个更小的数。在某个点上，数值变得如此之小，以至于硬件无法表示该值，这被称为*梯度消失*。在其他情况下，参数可能过于接近以至于无法区分——或者相反，分布得太远，这被称为*梯度爆炸*。

以下代码示例通过使用一个没有采用防止数值不稳定性方法（如每个密集层后的批量归一化）的 40 层深度神经网络（DNN）来演示这一点：

```
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(28, 28))
for _ in range(40):                                              ❶
    model.add(Dense(64, activation='relu'))                      ❶
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test  / 255.0).astype(np.float32)

model.fit(x_train, y_train, epochs=10, validation_split=0.1)
```

❶ 构建一个包含 40 个隐藏层的模型

在以下输出中，你可以看到在前三个 epoch 中，训练和评估数据上的精度持续增加，相应的损失持续减少。但之后，精度变得不稳定；模型数值不稳定：

```
Train on 54000 samples, validate on 6000 samples
Epoch 1/10
54000/54000 [==============================] - 9s 161us/sample - loss: 1.4461 
➥ - acc: 0.4367 - val_loss: 0.8802 - val_acc: 0.7223
Epoch 2/10
54000/54000 [==============================] - 7s 134us/sample - loss: 0.8054 
➥ - acc: 0.7202 - val_loss: 0.7419 - val_acc: 0.7727
Epoch 3/10
54000/54000 [==============================] - 7s 136us/sample - loss: 0.8606 
➥ - acc: 0.7530 - val_loss: 0.6923 - val_acc: 0.8352                       ❶
Epoch 4/10
54000/54000 [==============================] - 8s 139us/sample - loss: 0.8743 
➥ - acc: 0.7472 - val_loss: 0.7726 - val_acc: 0.7617
Epoch 5/10
54000/54000 [==============================] - 8s 139us/sample - loss: 0.7491 
➥ - acc: 0.7863 - val_loss: 0.9322 - val_acc: 0.7165                       ❷
Epoch 6/10
54000/54000 [==============================] - 7s 134us/sample - loss: 0.9151 
➥ - acc: 0.7087 - val_loss: 0.8160 - val_acc: 0.7573                       ❷
Epoch 7/10
54000/54000 [==============================] - 7s 135us/sample - loss: 0.9764 
➥ - acc: 0.6836 - val_loss: 0.7796 - val_acc: 0.7555                       ❷
Epoch 8/10
54000/54000 [==============================] - 7s 134us/sample - loss: 0.8836 
➥ - acc: 0.7202 - val_loss: 0.8348 - val_acc: 0.7382
Epoch 9/10
54000/54000 [==============================] - 8s 140us/sample - loss: 0.7975 
➥ - acc: 0.7626 - val_loss: 0.7838 - val_acc: 0.7760
Epoch 10/10
54000/54000 [==============================] - 8s 140us/sample - loss: 0.7317 
➥ - acc: 0.7719 - val_loss: 0.5664 - val_acc: 0.8282
```

❶ 模型精度在训练和评估数据上稳定提升。

❷ 模型精度在训练和评估数据上变得不稳定。

## 4.5 收敛

在训练阶段早期的假设是，你将训练数据喂给模型的次数越多，精度就越好。但我们发现，尤其是在更大、更复杂的网络中，在某个点上，精度会下降。今天，我们根据模型在应用中的使用方式，寻找在可接受的局部最优解上的收敛。如果我们过度训练神经网络，以下情况可能会发生：

+   神经网络对训练数据过度拟合，显示在训练数据上的精度增加，但在测试数据上的精度下降。

+   在更深的神经网络中，层将以非均匀的方式学习，并具有不同的收敛速率。因此，当一些层正在向收敛迈进时，其他层可能已经收敛并开始发散。

+   继续训练可能会导致神经网络跳出局部最优，并开始收敛到另一个更不准确的局部最优。

图 4.8 显示了在训练模型时理想情况下我们希望看到的收敛情况。你开始时在早期 epoch 中损失减少得相当快，随着训练逐渐接近（近）最优解，减少的速度减慢，然后最终停滞——这时，你达到了收敛。

![](img/CH04_F08_Ferlitsch.png)

图 4.8 当损失停滞时发生收敛。

让我们从使用 CIFAR-10 数据集在 TF.Keras 中构建一个简单的 ConvNet 模型开始，以演示收敛和发散的概念。在这段代码中，我故意省略了防止过拟合的方法，如 dropout 或批量归一化：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

height = x_train.shape[1]                              ❶
width  = x_train.shape[2]                              ❶

x_train = (x_train / 255.0).astype(np.float32)         ❷
x_test  = (x_test  / 255.0).astype(np.float32)         ❷

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(height, width, 3)))      ❸
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, validation_split=0.1)
```

❶ 计算数据集中图像的高度和宽度

❷ 标准化输入数据

❸ 将模型的输入形状设置为数据集中图像的高度和宽度

前六个 epoch 的统计数据如下。你可以看到每次通过时损失都在稳步减少，这意味着神经网络正在接近拟合数据。此外，训练数据的准确率从 52.35%上升到 87.4%，验证数据的准确率从 63.46%上升到 67.14%：

```
Train on 45000 samples, validate on 5000 samples
Epoch 1/20
45000/45000 [==============================] - 53s 1ms/sample - loss: 1.3348 
➥ - acc: 0.5235 - val_loss: 1.0552 - val_acc: 0.6346                       ❶
Epoch 2/20
45000/45000 [==============================] - 52s 1ms/sample - loss: 0.9527 
➥ - acc: 0.6667 - val_loss: 0.9452 - val_acc: 0.6726
Epoch 3/20
45000/45000 [==============================] - 52s 1ms/sample - loss: 0.7789 
➥ - acc: 0.7252 - val_loss: 0.9277 - val_acc: 0.6882
Epoch 4/20
45000/45000 [==============================] - 419s 9ms/sample - loss: 0.6328 
➥ - acc: 0.7785 - val_loss: 0.9324 - val_acc: 0.6964
Epoch 5/20
45000/45000 [==============================] - 53s 1ms/sample - loss: 0.4855 
➥ - acc: 0.8303 - val_loss: 1.0453 - val_acc: 0.6860
Epoch 6/20
45000/45000 [==============================] - 51s 1ms/sample - loss: 0.3575 
➥ - acc: 0.8746 - val_loss: 1.2903 - val_acc: 0.6714                       ❷
```

❶ 训练数据的初始损失

❷ 训练数据损失稳步下降，但在验证数据损失上有拟合数据的迹象

现在我们来看第 11 到 20 个 epoch。你可以看到我们在训练数据上达到了 98.46%，这意味着我们非常紧密地拟合了它。另一方面，我们的验证数据准确率在 66.58%处停滞。因此，经过六个 epoch 后，继续训练没有提供任何改进，我们可以得出结论，在第 7 个 epoch 时，模型已经过拟合到训练数据：

```
Epoch 11/20
45000/45000 [==============================] - 52s 1ms/sample - loss: 0.0966 
➥ - acc: 0.9669 - val_loss: 2.1891 - val_acc: 0.6694                     ❶
Epoch 12/20
45000/45000 [==============================] - 50s 1ms/sample - loss: 0.0845 
➥ - acc: 0.9712 - val_loss: 2.3046 - val_acc: 0.6666
.....
Epoch 20/20
45000/45000 [==============================] - 1683s 37ms/sample - loss: 
➥ 0.0463 - acc: 0.9848 - val_loss: 3.1512 - val_acc: 0.6658              ❷
```

❶ 验证损失持续上升，而模型变得非常过拟合训练数据。

❷ 验证损失非常高，模型高度拟合训练数据。

训练数据和验证数据的损失函数值也表明模型正在过拟合。训练数据在第 11 到 20 个 epoch 之间的损失函数值持续减小，但对于相应的验证数据，它停滞并变得更差（发散）。

## 4.6 检查点和提前停止

本节介绍了两种使训练更经济的技巧：检查点和提前停止。当模型过拟合并发散时，检查点非常有用，我们希望在收敛点恢复模型权重，而不需要额外的重新训练成本。你可以将提前停止视为检查点的扩展。我们有一个监控系统在最早的时刻检测到发散，然后停止训练，在发散点恢复检查点时节省额外的成本。

### 4.6.1 检查点

*检查点*是指在训练过程中定期保存学习到的模型参数和当前超参数值。这样做有两个原因：

+   为了能够从上次停止的地方恢复模型的训练，而不是从头开始重新训练

+   为了识别训练中模型给出最佳结果的过去某个点

在第一种情况下，我们可能希望将训练分散到不同的会话中，作为管理资源的一种方式。例如，我们可能每天预留（或被授权）一个小时用于训练。每天一小时的训练结束后，训练将进行检查点保存。第二天，训练将通过从检查点恢复来继续。例如，你可能在一家有固定计算费用预算的研究机构工作，你的团队正在尝试训练一个计算成本较高的模型。为了管理预算，你的团队可能被分配了每日计算费用的限额。

为什么仅仅保存模型的权重和偏差就不够呢？在神经网络中，一些超参数值会动态变化，例如学习率和衰减。我们希望在训练暂停时的相同超参数值下继续。

在另一种场景中，我们可能将连续学习作为持续集成和持续交付（CI/CD）过程的一部分来实现。在这种情况下，新的标记图像会持续添加到训练数据中，我们只想增量地重新训练模型，而不是在每个集成周期从头开始重新训练。

在第二种情况下，我们可能希望在模型训练超过最佳最优值并开始发散和/或过拟合后找到最佳结果。我们不想从更少的 epoch（或其他超参数变化）开始重新训练，而是识别出达到最佳结果的 epoch，并将学习到的模型参数恢复（设置）为该 epoch 结束时检查点的参数。

检查点在每个 epoch 结束时发生，但我们是否应该在每个 epoch 后都进行检查点保存？可能不是。这可能会在空间上变得昂贵。让我们假设模型有 2500 万个参数（例如，ResNet50），每个参数是一个 32 位的浮点值（4 字节）。那么每个检查点就需要 100 MB 来保存。经过 10 个 epoch 后，这已经需要 1 GB 的磁盘空间了。

我们通常只在模型参数数量较小和/或 epoch 数量较小的情况下在每个 epoch 后进行检查点保存。在下面的代码示例中，使用`ModelCheckpoint`类实例化了一个检查点。参数`filepath`表示检查点的文件路径。文件路径可以是完整的文件路径或格式化的文件路径。在前者的情况下，检查点文件每次都会被覆盖。

在下面的代码中，我们使用格式语法`epoch:02d`为每个检查点生成一个唯一的文件，基于 epoch 编号。例如，如果是第三个 epoch，文件将是 mymodel-03.ckpt：

```
from tensorflow.keras.callbacks import ModelCheckpoint               ❶

filepath = "mymodel-{epoch:02d}.ckpt"                                ❷

checkpoint = ModelCheckpoint(filepath)                               ❸

model.fit(x_train, y_train, epochs=epochs, callbacks=[checkpoint])   ❹
```

❶ 导入 ModelCheckpoint 类

❷ 为每个 epoch 设置唯一的文件路径名

❸ 创建一个 ModelCheckpoint 对象

❹ 训练模型并使用回调参数启用检查点

然后，可以使用 `load_model()` 方法从检查点恢复模型：

```
from tensorflow.keras.models import load_model    ❶

model = load_model('mymodel-03.ckpt')             ❷
```

❶ 导入 load_model 方法

❷ 从保存的检查点恢复模型

对于具有更多参数和/或 epoch 数的模型，我们可以选择使用参数 `period` 在每 *n* 个 epoch 上保存一个检查点。在这个例子中，每四个 epoch 保存一个检查点：

```
from tensorflow.keras.callbacks import ModelCheckpoint

filepath = "mymodel-{epoch:02d}.ckpt"

checkpoint = ModelCheckpoint(filepath, period=4)       ❶

model.fit(x_train, y_train, epochs=epochs, callbacks=[checkpoint])
```

❶ 每四个 epoch 创建一个检查点

或者，我们可以使用参数 `save_best_only=True` 和参数 `monitor` 来保存当前最佳检查点，以基于测量结果做出决策。例如，如果参数 `monitor` 设置为 `val_acc`，则只有在验证准确率高于上次保存的检查点时才会写入检查点。如果参数设置为 `val_loss`，则只有在验证损失低于上次保存的检查点时才会写入检查点：

```
from tensorflow.keras.callbacks import ModelCheckpoint

filepath = "mymodel-best.ckpt"                                   ❶
checkpoint = ModelCheckpoint(filepath, save_best_only=True, 
➥ monitor='val_acc')                                            ❷

model.fit(x_train, y_train, epochs=epochs, callbacks=[checkpoint])
```

❶ 保存最佳检查点的文件路径

❷ 仅当验证损失小于上次检查点时保存检查点

### 4.6.2 提前停止

*提前停止* 是设置一个条件，使得训练在设定的限制（例如，epoch 数）之前提前终止。这通常是为了在达到目标目标时（例如，准确率水平或评估损失的收敛）节省资源或防止过拟合而设置的。例如，我们可能会设置 20 个 epoch 的训练，每个 epoch 平均 30 分钟，总共 10 小时。但如果目标在 8 个 epoch 后达成，提前终止训练将节省 6 小时资源。

提前停止的指定方式与检查点类似。实例化一个 `EarlyStopping` 对象，并配置一个目标目标，然后将其传递给 `fit()` 方法的 `callbacks` 参数。在这个例子中，只有在验证损失停止减少时，训练才会提前停止：

```
from tensorflow.keras.callbacks import EarlyStopping                 ❶

earlystop = EarlyStopping(monitor='val_loss')                        ❷

model.fit(x_train, y_train, epochs=epochs, callbacks=[earlystop])    ❸
```

❶ 导入 EarlyStopping 类

❷ 当验证损失停止减少时设置提前停止

❸ 训练模型并使用提前停止，如果验证损失停止减少则提前停止训练

除了监控验证损失以实现提前停止外，我们还可以通过参数设置 `monitor="val_acc"` 监控验证准确率。存在一些额外的参数用于微调，以防止意外提前停止；例如，在损失曲线上陷入鞍点（一个损失曲线的平坦区域）时，更多的训练将克服这种情况。参数 `patience` 指定了在提前停止之前没有改进的最小 epoch 数，而 `min_delta` 指定了确定模型是否改进的最小阈值。在这个例子中，如果在三个 epoch 后验证损失没有改进，训练将提前停止：

```
from tensorflow.keras.callbacks import EarlyStopping

earlystop = EarlyStopping(monitor='val_loss', patience=3)     ❶

model.fit(x_train, y_train, epochs=epochs, callbacks=[earlystop])
```

❶ 当验证损失停止减少三个 epoch 时设置提前停止

## 4.7 超参数

让我们先解释一下学习参数和超参数之间的区别。*学习参数*，权重和偏差，是在训练过程中学习的。对于神经网络来说，这些通常是每个神经网络连接上的权重和每个节点的偏差。对于卷积神经网络（CNNs），学习参数是每个卷积层中的过滤器。这些学习参数在模型完成训练后仍作为模型的一部分。

*超参数*是用于训练模型的参数，但它们本身不是训练模型的一部分。训练后，超参数不再存在。超参数通过回答诸如这些问题来提高模型的训练效果：

+   训练模型需要多长时间？

+   模型收敛有多快？

+   它是否找到了全局最优解？

+   模型的准确度如何？

+   模型过拟合的程度如何？

超参数的另一个视角是，它们是衡量模型开发成本和质量的一种手段。随着我们在第十章进一步探讨超参数，我们将深入研究这些问题和其他问题。

### 4.7.1 Epochs

最基本的超参数是 epoch 的数量，尽管现在这更常见地被步骤所取代。*epoch 超参数*是在训练过程中你将整个训练数据通过神经网络的次数。

训练在计算时间上非常昂贵。它包括正向传播将训练数据传递过去和反向传播来更新（训练）模型的参数。例如，如果一次完整的数据传递（epoch）需要 15 分钟，而我们运行 100 个 epoch，训练时间将需要 25 小时。

### 4.7.2 步骤

提高准确度和减少训练时间的另一种方法是改变训练数据集的抽样分布。对于 epoch，我们考虑从我们的训练数据中按顺序抽取批次。尽管我们在每个 epoch 的开始时随机打乱训练数据，但抽样分布仍然是相同的。

让我们现在考虑我们想要识别的主题的整个群体。在统计学中，我们称这为*总体分布*（图 4.9）。

![](img/CH04_F09_Ferlitsch.png)

图 4.9 总体分布与人群中的随机样本之间的区别

但我们永远不会有一个数据集是实际的整个总体分布。相反，我们有样本，我们称之为*总体分布的抽样分布*（图 4.10）。

![](img/CH04_F10_Ferlitsch.png)

图 4.10 由人群中的随机样本组成的抽样分布。

提高我们的模型的另一种方法是学习训练模型的最佳抽样分布。尽管我们的数据集可能是固定的，但我们可以使用几种技术来改变分布，从而学习最适合训练模型的抽样分布。这些方法包括以下内容：

+   正则化/丢弃

+   批标准化

+   数据增强

从这个角度来看，我们不再将神经网络视为对训练数据的顺序遍历，而是将其视为从训练数据的采样分布中进行随机抽取。在这种情况下，*步骤*指的是我们将从训练数据的采样分布中抽取的批次（抽取）的数量。

当我们在神经网络中添加 dropout 层时，我们是在每个样本的基础上随机丢弃激活。除了减少神经网络的过拟合，我们还改变了分布。

在批量归一化中，我们最小化训练数据批次（样本）之间的协方差偏移。正如我们在输入上使用标准化一样，激活也被使用标准化重新缩放（我们减去批次均值并除以批次标准差）。这种归一化减少了模型参数更新中的波动；这个过程被称为*增加更多稳定性*到训练中。此外，这种归一化模仿从更具有代表性的总体分布的采样分布中抽取的过程。

通过数据增强（在第十三章中讨论），我们在一组参数内修改现有示例来创建新的示例。然后我们随机选择修改，这也有助于改变分布。

在批量归一化、正则化/dropout 和数据增强的情况下，没有两个 epoch 会有相同的采样分布。在这种情况下，现在的做法是限制从每个新的采样分布中随机抽取（步骤）的数量，进一步改变分布。例如，如果步骤设置为 1000，那么每个 epoch 中，只有 1000 个随机批次将被选中并输入到神经网络中进行训练。

在 TF.Keras 中，我们可以将`epochs`和`steps_per_epoch`参数指定为`fit()`方法的参数，作为参数：

```
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          steps_per_epoch=1000)
```

### 4.7.3 批量大小

要了解如何设置批量大小，你应该对三种梯度下降算法有基本的了解：随机梯度下降、批量梯度下降和迷你批量梯度下降。算法是模型参数在训练期间更新的（学习）手段。

随机梯度下降

在*随机梯度下降*（*SGD*）中，模型在训练过程中每个示例被输入后更新。由于每个示例都是随机选择的，因此示例之间的方差可能导致梯度的大幅波动。

一个好处是，在训练过程中，我们不太可能收敛到局部（即较差的）最优解，而更有可能找到全局最优解并收敛。另一个好处是，损失变化的速率可以实时监控，这可能有助于自动超参数调整的算法。缺点是这每轮计算成本更高。

批量梯度下降

在*批量梯度下降*中，每个示例的错误损失是在训练过程中每个示例被输入时计算的，但模型的更新是在每个 epoch 结束时（在所有训练数据通过之后）进行的。因此，由于它是基于所有示例的损失来计算的，而不是单个示例，所以梯度被平滑了。

这种方法的优点是每个 epoch 的计算成本更低，训练过程更可靠地收敛。缺点是模型可能收敛到一个不太准确的局部最优解，并且需要运行整个 epoch 来监控性能数据。

小批量梯度下降

*小批量梯度下降*方法是在随机梯度下降和批量梯度下降之间的折中。不是单个示例或所有示例，神经网络被输入到小批量中，这些小批量是整个训练数据的一个子集。小批量的大小越小，训练越类似于随机梯度下降，而较大的批量大小则更类似于批量梯度下降。

对于某些模型和数据集，随机梯度下降（SGD）效果最好。通常，使用小批量梯度下降的折中方案是一种常见做法。超参数`batch_size`表示小批量的大小。由于硬件架构，最节省时间/空间的批量大小是 8 的倍数，例如 8、16、32 和 64。首先尝试的批量大小通常是 32，然后是 128。对于在高端硬件（HW）加速器（如 GPU 和 TPU）上的极大数据集，常见的批量大小是 256 和 512。在 TF.Keras 中，你可以在模型的`fit()`方法中指定`batch_size`：

```
model.fit(x_train, y_train, batch_size=32)
```

### 4.7.4 学习率

*学习率*通常是超参数中最有影响力的。它可以对训练神经网络所需的时间长度以及神经网络是否收敛到局部（较差）最优解，以及是否收敛到全局（最佳）最优解产生重大影响。

在反向传播过程中更新模型参数时，梯度下降算法被用来从损失函数中导出一个值，以添加/减去到模型参数中。这些添加和减去可能会导致参数值的大幅波动。如果一个模型有并且继续有参数值的大幅波动，那么模型的参数将会“四处乱飞”并且永远不会收敛。

如果你观察到损失和/或准确性的波动很大，那么你的模型训练并没有收敛。如果训练没有收敛，那么运行多少个 epoch 都没有关系；模型永远不会完成训练。

学习率为我们提供了一种控制模型参数更新程度的方法。在基本方法中，学习率是 0 到 1 之间的一个固定系数，它乘以要添加/减去的值，以减少添加或减去的量。这些较小的增量在训练期间增加了稳定性，并增加了收敛的可能性。

小学习率与大脑学习率

如果我们使用一个非常小的学习率，如 0.001，我们将在更新模型参数时消除大的摆动。这通常可以保证训练将收敛到一个局部最优解。但有一个缺点。首先，我们使增量越小，需要的训练数据（周期）遍历次数越多，以最小化损失。这意味着需要更多的时间来训练。其次，增量越小，训练探索其他局部最优解的可能性就越小，这些局部最优解可能比训练收敛到的更准确；相反，它可能收敛到一个较差的局部最优解或卡在鞍点上。

一个大的学习率，如 0.1，在更新模型参数时很可能会引起大的跳跃。在某些情况下，它可能最初导致更快的收敛（更少的周期）。缺点是，即使你最初收敛得很快，跳跃可能会超过并开始导致收敛来回摆动，或者跳到不同的局部最优解。在非常高的学习率下，训练可能开始发散（损失增加）。

许多因素有助于确定在训练过程中的不同时间点最佳学习率是什么。在最佳实践中，该速率将在 10e-5 到 0.1 之间。

这是一个基本的公式，通过将学习率乘以计算出的添加/减去的量（梯度）来调整权重：

```
weight += -learning_rate * gradient
```

衰减

一种常见的做法是开始时使用稍微大一点的学习率，然后逐渐减小它，这也被称为*学习率衰减*。较大的学习率最初会探索不同的局部最优解，以收敛到并使初始深度摆动到相应的局部最优解。收敛速率和最小化损失函数的初始更新可以用来聚焦于最佳（好的）局部最优解。

从那个点开始，学习率逐渐衰减。随着学习率的衰减，出现偏离良好局部最优解的摆动可能性降低，稳定下降的学习率将调整收敛，使其接近最小点（尽管，越来越小的学习率会增加训练时间）。因此，衰减成为在最终精度的小幅度提升和整体训练时间之间的权衡。

以下是一个基本公式，在更新权重计算中添加衰减：在每次更新中，学习率通过衰减量（称为*固定衰减*）减少：

```
weight += -learning_rate * gradient
learning_rate -= decay
```

在实践中，衰减公式通常是基于时间、步长或余弦衰减。这些公式可以用简化的术语表达，迭代可以是批量或周期。默认情况下，TF.Keras 优化器使用基于时间的衰减。公式如下：

+   基于时间的衰减

    ```
    learning_rate *= (1 / (1 + decay * iteration))
    ```

+   步长衰减

    ```
    learning_rate = initial_learning_rate * decay**iteration
    ```

+   余弦衰减

    ```
    learning_rate = c * (1 + cos(pi * (steps_per_epoch * interaction)/epochs))
    # where c is typically in range 0.45 to 0.55
    ```

动量

另一种常见的做法是根据先前变化来加速或减速变化率。如果我们有大的收敛跳跃，我们可能会跳出局部最优，所以我们可能希望减速学习率。如果我们有小的或没有收敛变化，我们可能希望加速学习率以跳过一个鞍点。通常，动量的值在 0.5 到 0.99 之间：

```
velocity = (momentum * velocity) - (learning_rate * gradient)
weight += velocity
```

自适应学习率

许多流行的算法会动态调整学习率：

+   Adadelta

+   Adagrad

+   Adam

+   AdaMax

+   AMSGrad

+   动量

+   Nadam

+   Nesterov

+   RMSprop

这些算法的解释超出了本节的范围。有关这些和其他优化器的更多信息，请参阅`tf.keras.optimizers`的文档（[`mng.bz/Par9`](http://mng.bz/Par9)）。对于 TF.Keras，这些学习率算法是在定义优化器以最小化损失函数时指定的：

```
from tensorflow.keras import optimizers

optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)  ❶
model.compile(loss='mean_squared_error', optimizer=optimizer)               ❷
```

❶ 指定优化器的学习率和衰减

❷ 编译模型，指定损失函数和优化器

## 4.8 不变性

那么，*不变性*是什么意思？在神经网络的情况下，这意味着当输入被转换时，结果（预测）保持不变。在训练图像分类器的情况下，可以使用图像增强来训练模型，使其能够识别图像中无论对象的大小和位置如何的对象，而无需额外的训练数据。

让我们考虑一个图像分类器 CNN（这个类比也可以应用于目标检测）。我们希望被分类的对象无论其在图像中的位置如何都能被正确识别。如果我们变换输入，使得对象在图像中移动到新的位置，我们希望结果（预测）保持不变。

对于 CNN 和一般成像，我们希望模型支持的主要不变性类型是*平移*和*尺度*不变性。在 2019 年之前，平移和尺度不变性是通过在模型训练之前使用图像增强预处理来处理的，即在 CPU 上对图像数据进行预处理，同时在 GPU 上训练时提供数据。我们将在本节中讨论这些传统技术。

训练平移/尺度不变性的一个方法是为每个类别（每个对象）提供足够的图像，使得对象在图像中的位置、旋转、尺度以及视角都不同。嗯，这可能不太实际收集。

结果表明，使用图像增强预处理自动生成平移/尺度不变图像的方法非常直接，这种预处理通过矩阵操作高效执行。基于矩阵的变换可以通过各种 Python 包执行，例如 TF.Keras `ImageDataGenerator`类、TensorFlow `tf.image`模块或 OpenCV。

图 4.11 描述了在向模型提供训练数据时的典型图像增强流程。对于每个抽取的批次，从批次中的图像中选择一个随机子集进行增强（例如，50%）。然后，根据某些约束（例如，从-30 到 30 度的随机旋转值）随机变换这个随机选择的图像子集。然后，将修改后的批次（原始图像加上增强图像）输入模型进行训练。

![图片](img/CH04_F11_Ferlitsch.png)

图 4.11 在图像增强过程中，随机选择批次中的图像子集进行增强。

### 4.8.1 平移不变性

本小节将介绍如何在训练数据集中手动增强图像，以便模型学会识别图像中的对象，而不管其在图像中的位置如何。例如，我们希望模型能够识别出无论马在图像中朝哪个方向，都能识别出马，或者无论苹果在背景中的位置如何，都能识别出苹果。

*平移不变性*在图像输入的上下文中包括以下内容：

+   垂直/水平位置（对象可以在图片的任何位置）

+   旋转（对象可以处于任何旋转角度）

垂直/水平变换通常是通过矩阵滚动操作或裁剪来执行的。一个方向（例如，镜像）通常是通过矩阵翻转来实现的。旋转通常是通过矩阵转置来处理的。

翻转

*矩阵翻转*通过在垂直或水平轴上翻转图像来变换图像。由于图像数据表示为 2D 矩阵的堆叠（每个通道一个），翻转可以通过矩阵转置函数高效执行，而无需更改像素数据（例如插值）。图 4.12 比较了图像的原始版本和翻转版本。

![图片](img/CH04_F12_Ferlitsch.png)

图 4.12 比较了一个苹果：原始图像、垂直轴翻转和水平轴翻转（图片来源：malerapaso，iStock）

让我们从展示如何使用 Python 中流行的图像库来翻转图像开始。以下代码演示了如何使用 Python 的 PIL 图像库中的矩阵转置方法来垂直（镜像）和水平翻转图像：

```
from PIL import Image

image = Image.open('apple.jpg')                  ❶

image.show()                                     ❷

flip = image.transpose(Image.FLIP_LEFT_RIGHT)    ❸
flip.show()                                      ❸

flip = image.transpose(Image.FLIP_TOP_BOTTOM)    ❹
flip.show()                                      ❹
```

❶ 将图像读入内存

❷ 显示图像的原始视角

❸ 在垂直轴上翻转图像（镜像）

❹ 在水平轴上翻转图像（上下颠倒）

或者，可以使用 PIL 类`ImageOps`模块来执行翻转，如下所示：

```
from PIL import Image, ImageOps

image = Image.open('apple.jpg')     ❶

flip = ImageOps.mirror(image)       ❷
flip.show()                         ❷

flip = ImageOps.flip(image)v        ❸
flip.show()                         ❸
```

❶ 读取图像

❷ 在垂直轴上翻转图像（镜像）

❸ 在水平轴上翻转图像（上下颠倒）

以下代码演示了如何使用 OpenCV 中的矩阵转置方法垂直（镜像）和水平翻转图像：

```
import cv2
from matplotlib import pyplot as plt 

image = cv2.imread('apple.jpg')

plt.imshow(image)             ❶

flip = cv2.flip(image, 1)     ❷
plt.imshow(flip)              ❷

flip = cv2.flip(image, 0)     ❸
plt.imshow(flip)              ❸
```

❶ 以原始视角显示图像

❷ 在垂直轴上翻转图像（镜像）

❸ 在水平轴上翻转图像（上下颠倒）

以下代码演示了如何使用 NumPy 中的矩阵转置方法翻转图像垂直（镜像）和水平翻转：

```
import numpy as np
import cv2
from matplotlib import pyplot as plt 

image = cv2.imread('apple.jpg')
plt.imshow(image)
flip = np.flip(image, 1)     ❶
plt.imshow(flip)             ❶

flip = np.flip(image, 0)     ❷
plt.imshow(flip)             ❷
```

❶ 在垂直轴上翻转图像（镜像）

❷ 在水平轴上翻转图像（上下颠倒）

旋转 90/180/270

除了翻转之外，可以使用*矩阵转置*操作旋转图像 90 度（左转）、180 度和 270 度（右转）。与翻转一样，该操作效率高，不需要插值像素，并且没有裁剪的副作用。图 4.13 比较了原始图像和 90 度旋转版本。

![](img/CH04_F13_Ferlitsch.png)

图 4.13 苹果的 90 度、180 度和 270 度旋转比较

以下代码演示了如何使用 Python 的 PIL 图像库中的矩阵转置方法旋转图像 90 度、180 度和 270 度：

```
from PIL import Image

image = Image.open('apple.jpg')

rotate = image.transpose(Image.ROTATE_90)     ❶
rotate.show()                                 ❶

rotate = image.transpose(Image.ROTATE_180)    ❷
rotate.show()                                 ❷

rotate = image.transpose(Image.ROTATE_270)    ❸
rotate.show()                                 ❸
```

❶ 旋转图像 90 度

❷ 旋转图像 180 度

❸ 旋转图像 270 度

OpenCV 没有 90 度或 270 度的转置方法；您可以使用带有-1 值的翻转方法来执行 180 度翻转。（使用`imutils`模块演示了 OpenCV 中其他所有旋转方法，见下文小节。）

```
import cv2
from matplotlib import pyplot as plt 

image = cv2.imread('apple.jpg')

rotate = cv2.flip(image, -1)     ❶
plt.imshow(rotate)               ❶
```

❶ 旋转图像 180 度

下一个示例演示了如何使用 NumPy 的`rot90()`方法旋转图像 90 度、180 度和 270 度，其中第一个参数是要旋转 90 度的图像，第二个参数（`k`）是旋转的次数：

```
import numpy as np
import cv2
from matplotlib import pyplot as plt 

image = cv2.imread('apple.jpg')

rotate = np.rot90(image, 1)     ❶
plt.imshow(rotate)              ❶

rotate = np.rot90(image, 2)     ❷
plt.imshow(rotate)              ❷

rotate = np.rot90(image, 3)     ❸
plt.imshow(rotate)              ❸
```

❶ 旋转图像 90 度

❷ 旋转图像 180 度

❸ 旋转图像 270 度

当翻转图像 90 度或 270 度时，您正在改变图像的方向，如果图像的高度和宽度相同，则这不是问题。如果不相同，旋转后的图像高度和宽度将互换，并且不会与神经网络的输入向量匹配。在这种情况下，您应使用`imutils`模块或其他方法调整图像大小。

旋转

*旋转*通过在-180 度和 180 度之间旋转图像来变换图像。通常，旋转的角度是随机选择的。您可能还想限制旋转的范围以匹配模型部署的环境。以下是一些常见的做法：

+   如果图像将直接对齐，请使用-15 到 15 度的范围。

+   如果图像可能处于倾斜状态，请使用-30 度到 30 度的范围。

+   对于小型物体，如包裹或货币，请使用-180 度到 180 度的完整范围。

旋转的另一个问题是，如果您在相同大小的边界内旋转图像，除了 90 度、180 度或 270 度之外，图像的边缘部分将最终超出边界（裁剪）。

图 4.14 展示了使用 PIL 方法`rotate()`将苹果图像旋转 45 度的例子。你可以看到苹果底部的一部分和叶子被裁剪掉了。

![](img/CH04_F14_Ferlitsch.png)

图 4.14 旋转非 90 度倍数时的图像裁剪示例

正确处理旋转的方法是在更大的边界区域内旋转，这样图像的任何部分都不会被裁剪，然后将旋转后的图像调整回原始大小。为此，我推荐使用`imutils`模块（由 Adrian Rosebrock 创建，[`mng.bz/JvR0`](https://shortener.manning.com/JvR0)），它包含了一组针对 OpenCV 的便利方法：

```
import cv2, imutils
from matplotlib import pyplot as plt 

image = cv2.imread('apple.jpg')

shape = (image.shape[0], image.shape[1])                            ❶

rotate = imutils.rotate_bound(image, 45)                            ❷

rotate = cv2.resize(rotate, shape, interpolation=cv2.INTER_AREA)    ❸
plt.imshow(rotate)
```

❶ 记录原始高度和宽度

❷ 旋转图像

❸ 将图像调整回原始形状

移动

移动会将图像中的像素数据在垂直（高度）或水平（宽度）轴上移动+/-。这将改变被分类对象在图像中的位置。图 4.15 显示了苹果图像向下移动 10%和向上移动 10%的情况。

![](img/CH04_F15_Ferlitsch.png)

图 4.15 苹果的比较：原始图像，向下移动 10%，向上移动 10%

以下代码演示了使用 NumPy 的`np.roll()`方法垂直和水平移动图像+/- 10%：

```
import cv2
import numpy as np
from matplotlib import pyplot as plt 

image = cv2.imread('apple.jpg')

height = image.shape[0]                            ❶
Width  = image.shape[1]                            ❶

roll = np.roll(image, height // 10, axis=0)        ❷
plt.imshow(roll)

roll = np.roll(image, -(height // 10), axis=0)     ❸
plt.imshow(roll)

roll = np.roll(image, width // 10, axis=1)         ❹
plt.imshow(roll)

roll = np.roll(image, -(width // 10), axis=1)      ❺
plt.imshow(roll)
```

❶ 获取图像的高度和宽度

❷ 向下移动图像 10%

❸ 向上移动图像 10%

❹ 向右移动图像 10%

❺ 向左移动图像 10%

移动是高效的，因为它作为矩阵的滚动操作实现；行（高度）或列（宽度）被移动。因此，移出末尾的像素被添加到开始处。

如果移动太大，图像可以分裂成两块，每块都与另一块相对。图 4.16 显示了苹果垂直移动了 50%，导致其破碎。

![](img/CH04_F16_Ferlitsch.png)

图 4.16 当图像移动过多时，它变得破碎。

为了避免破碎，通常将图像的移动限制在不超过 20%。或者，我们可以裁剪图像，并用黑色填充裁剪的空间，如下所示使用 OpenCV：

```
import cv2
from matplotlib import pyplot as plt 

image = cv2.imread('apple.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height = image.shape[0]                                           ❶
width  = image.shape[1]                                           ❶

image = image[0: height//2,:,:]                                   ❷

image = cv2.copyMakeBorder(image, (height//4), (height//4), 0, 0, 
                           cv2.BORDER_CONSTANT, 0)                ❸

plt.imshow(image)
```

❶ 获取图像的高度

❷ 删除图像底部（50%）

❸ 添加黑色边框以重新调整图像大小

此代码生成了图 4.17 的输出。

![](img/CH04_F17_Ferlitsch.png)

图 4.17 使用裁剪和填充避免图像破碎

### 4.8.2 尺度不变性

本小节介绍了如何在训练数据集中手动增强图像，以便模型学会识别图像中的对象，无论对象的大小如何。例如，我们希望模型能够识别出苹果，无论它占据图像的大部分还是作为图像背景上的小部分。

在图像输入的上下文中，尺度不变性包括以下内容：

+   缩放（对象可以是图像中的任何大小）

+   放射性（对象可以从任何角度观看）

缩放

*缩放*通过从图像中心放大图像来转换图像，这是通过调整大小和裁剪操作完成的。你找到图像的中心，计算围绕中心的裁剪边界框，然后裁剪图像。图 4.18 是放大 2 倍的苹果图像。

![图片](img/CH04_F18_Ferlitsch.png)

图 4.18 在放大后裁剪图像以保持相同的图像大小。

当使用`Image.resize()`放大图像时，`Image.BICUBIC`插值通常提供最佳结果。此代码演示了如何使用 Python 的 PIL 图像库放大图像：

```
from PIL import Image
image = Image.open('apple.jpg')

zoom = 2 
height, width = image.size                                                 ❶

image = image.resize( (int(height*zoom), int(width*zoom)), Image.BICUBIC)  ❷

center = (image.size[0]//2, image.size[1]//2)                              ❸

crop = (int(center[0]//zoom), int(center[1]//zoom))                        ❹

box = ( crop[0], crop[1], (center[0] + crop[0]), (center[1] + crop[1]) )   ❺

image = image.crop( box )                                                  ❻
image.show()
```

❶ 记录图像的原始高度和宽度

❷ 通过缩放（按比例）调整图像大小

❸ 找到缩放图像的中心

❹ 计算裁剪的左上角

❺ 计算裁剪边界框

❻ 裁剪图像

下一个代码示例演示了如何使用 OpenCV 图像库放大图像。当使用`cv2.resize()`插值放大图像时，`cv2.INTER_CUBIC`通常提供最佳结果。插值`cv2.INTER_LINEAR`更快，并提供几乎相当的结果。插值`cv2.INTER_AREA`通常用于减小图像。

```
import cv2
from matplotlib import pyplot as plt 

zoom = 2 

height, width = image.shape[:2]                                             ❶

center = (image.shape[0]//2, image.shape[1]//2)                             ❷
z_height = int(height // zoom)                                              ❷
z_width  = int(width  // zoom)                                              ❷

image = image[(center[0] - z_height//2):(center[0] + z_height//2), center[1] -
              z_width//2:(center[1] + z_width//2)]                          ❸

image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)   ❹

plt.imshow(image)
```

❶ 记录图像的原始高度和宽度

❷ 找到缩放图像的中心

❸ 通过形成裁剪边界框来切割缩放图像

❹ 将裁剪的图像重新调整回原始大小

### 4.8.3 TF.Keras ImageDataGenerator

TF.Keras 图像预处理模块支持使用`ImageDataGenerator`类进行多种图像增强。这个类创建一个生成器，用于生成增强图像的批次。类的初始化器接受零个或多个参数，用于指定增强的类型。以下是一些参数，我们将在本节中介绍：

+   `horizontal_flip=True|False`

+   `vertical_flip=True|False`

+   `rotation_range=degrees`

+   `zoom_range=(lower, upper)`

+   `width_shift_range=percent`

+   `height_shift_range=percent`

+   `brightness_range=(lower, upper)`

翻转

在以下代码示例中，我们执行以下操作：

1.  读取一个苹果的单张图像。

1.  创建一个包含一个图像（苹果）的批次。

1.  实例化一个`ImageDataGenerator`对象。

1.  使用我们的增强选项（在这种情况下，水平和垂直翻转）初始化`ImageDataGenerator`。

1.  使用`ImageDataGenerator`的`flow()`方法创建一个批次生成器。

1.  通过生成器迭代六次，每次返回一个包含一个图像的`x`批次。

    +   生成器将每次迭代随机选择一个增强（包括无增强）。

    +   变换（增强）后，像素值类型将是 32 位浮点数。

    +   将像素的数据类型改回 8 位整数，以便使用 Matplotlib 显示。

下面是代码：

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('apple.jpg')                                          ❶
batch = np.asarray([image])                                              ❶

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)   ❷

step=0                                                                   ❸
for x in datagen.flow(batch, batch_size=1):                              ❸
        step += 1
        if step > 6: break
        plt.figure()
        plt.imshow(x[0].astype(np.uint8))                                ❹
```

❶ 制作一个包含一个图像（苹果）的批次

❷ 创建一个用于增强数据的生成器

❸ 运行生成器，其中每个图像都是随机增强

❹ 增强操作将像素数据转换为浮点数，然后将其转换回 uint8 以显示图像。

旋转

在以下代码中，我们使用`rotation_range`参数设置介于-60 度和 60 度之间的随机旋转。请注意，旋转操作不执行边界检查和调整大小（如`imutils.rotate_bound()`），因此图像的一部分可能最终被裁剪：

```
datagen = ImageDataGenerator(rotation_range=60)
```

缩放

在此代码中，我们使用`zoom_range`参数设置从 0.5（缩小）到 2（放大）的随机值。该值可以是两个元素的元组或列表：

```
datagen = ImageDataGenerator(zoom_range=(0.5, 2))
```

平移

在此代码中，我们使用`width_shift_range`和`height_shift_range`设置从 0 到 20%的随机值以水平或垂直移动：

```
datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
```

亮度

在以下代码中，我们使用`brightness_range`参数设置从 0.5（较暗）到 2（较亮）的随机值。该值可以是两个元素的元组或列表：

```
datagen = ImageDataGenerator(brightness_range=(0.5, 2))
```

作为最后的注意事项，像亮度这样的变换，在像素值上添加一个固定量，是在归一化或标准化之后完成的。如果在此之前完成，归一化和标准化会将值压缩到相同的原始范围，从而撤销变换。

## 4.9 原始（磁盘）数据集

到目前为止，我们已经讨论了直接从内存中存储和访问的图像的训练技术。这对于小型数据集有效，例如那些包含微小图像的数据集，或者对于包含少于 50,000 个图像的大型图像数据集。但是，一旦我们开始使用较大尺寸的图像和大量图像进行训练，例如几十万个图像，您的数据集很可能存储在磁盘上。本小节涵盖了在磁盘上存储图像和访问它们进行训练的常见约定。

除了用于学术/研究目的的精选数据集之外，我们在生产中使用的数据集很可能存储在磁盘上（如果是结构化数据，则为数据库）。在图像数据的情况下，我们需要执行以下操作：

1.  从磁盘读取图像及其对应的标签到内存中（假设图像数据适合内存）。

1.  将图像调整大小以匹配 CNN 的输入向量。

接下来，我们将介绍几种在磁盘上布局图像数据集的常用方法。

### 4.9.1 目录结构

将图像放置在本地磁盘上的目录文件夹结构中是最常见的布局之一。在此布局中，如图 4.19 所示，根（父）文件夹是数据集的容器。在根级别以下有一个或多个子目录。每个子目录对应一个类别（标签）并包含对应类别的图像。

使用我们的猫和狗示例，我们可能有一个名为 cats_n_dogs 的父目录，其中包含两个子目录，一个名为 cats，另一个名为 dogs。在每个子目录中会有对应类别的图像。

![](img/CH04_F19_Ferlitsch.png)

图 4.19 按类别目录文件夹布局

或者，如果数据集已经被分割成训练和测试数据，我们首先按训练/测试分组数据，然后按猫和狗的两个类别分组数据，如图 4.20 所示。

![图像](img/CH04_F20_Ferlitsch.png)

图 4.20 按训练和测试数据分割的目录文件夹布局

当数据集是分层标记时，每个顶级类别（标签）子文件夹根据类别（标签）层次进一步划分为子文件夹。以我们的猫和狗为例，每个图像根据是否是猫或狗（物种）进行分层标记，然后按品种划分。见图 4.21。

![图像](img/CH04_F21_Ferlitsch.png)

图 4.21 分层目录文件夹布局用于分层标记

### 4.9.2 CSV 文件

另一种常见的布局是使用逗号分隔值（CSV）文件来标识每个图像的位置和类别（标签）。在这种情况下，CSV 文件中的每一行都是一个单独的图像，CSV 文件至少包含两列，一列用于图像的位置，另一列用于图像的类别（标签）。位置可能是一个本地路径、远程位置，或者作为位置值的嵌入像素数据：

+   本地路径示例：

    ```
            label,location
            'cat', cats_n_dogs/cat/1.jpg
            'dog',cats_n_dogs/dog/2.jpg
    ```

    ...

+   远程路径示例：

    ```
            label,location
            'cat','http://mysite.com/cats_n_dogs/cat/1.jpg'
             'dog','http://mysite.com/cats_n_dogs/dog/2.jpg'
    ```

    ...

+   嵌入数据示例：

    ```
            label,location
             'cat',[[...],[...],[...]]
             'dog',[[...], [...], [...]]
    ```

### 4.9.3 JSON 文件

另一种常见的布局是使用 JavaScript 对象表示法（JSON）文件来标识每个图像的位置和类别（标签）。在这种情况下，JSON 文件是一个对象数组；每个对象是一个单独的图像，每个对象至少有两个键，一个用于图像的位置，另一个用于图像的类别（标签）。

位置可能是一个本地路径、远程位置，或者作为位置值的嵌入像素数据。以下是一个本地路径示例：

```
 [
            {'label': 'cat', 'location': 'cats_n_dogs/cat/1.jpg' },
            {'label': 'dog', 'location': 'cats_n_dogs/dog/2.jpg'}
            ...
]
```

### 4.9.4 读取图像

在磁盘数据集上训练时，第一步是从磁盘读取图像到内存。磁盘上的图像将以 JPG、PNG 或 TIF 等图像格式存在。这些格式定义了图像的编码和压缩方式以便存储。可以通过使用 PIL 的`Image.open()`方法将图像读取到内存中：

```
from PIL import Image
image = Image.open('myimage.jpg')
```

在实践中，你可能需要读取很多图像。假设你想要读取子目录（例如，猫）下的所有图像。在下面的代码中，我们扫描（获取子目录中所有文件的列表），将每个文件作为图像读取，并将读取的图像列表作为列表维护：

```
from PIL import Image
import os

def loadImages(subdir):                                 ❶
        images = []

        files = os.scandir(subdir)                      ❷
        for file in files:                              ❸
                 images.append(Image.open(file.path))   ❸
        return images

loadImages('cats')                                      ❹
```

❶ 读取单个类别标签下子文件夹中所有图像的步骤

❷ 获取子目录 cats 中所有文件的列表

❸ 读取每个图像并将其内存中的图像追加到列表中

❹ 读取子目录 cats 中的所有图像

注意，`os.scandir()`是在 Python 3.5 中添加的。如果你使用 Python 2.7 或更早版本的 Python 3，你可以使用`pip install scandir`来获取兼容版本。

让我们扩展前面的例子并假设图像数据集是按目录结构布局的；每个子目录是一个类别（标签）。在这种情况下，我们希望分别扫描每个子目录并记录类别的子目录名称：

```
import os

def loadDirectory(parent):                               ❶
        classes = {}
        dataset = []

        for subdir in os.scandir(parent):                ❷
                if not subdir.is_dir():                  ❸
                        continue

                classes[subdir.name] = len(dataset)      ❹

                dataset.append(loadImages(subdir.path))

                print("Processed:", subdir.name, "# Images", 
                      len(dataset[len(dataset)-1]))

        return dataset, classes                          ❺

loadDirectory('cats_n_dogs')                             ❻
```

❶ 读取数据集所有图像的类别的步骤

❷ 获取数据集父目录（根目录）下的所有子目录列表

❸ 忽略任何非子目录的条目（例如，许可证文件）

❹ 维护类别（子目录名称）到标签（索引）的映射

❺ 返回数据集图像和类别映射

❻ 读取数据集 cats_n_dogs 中所有图像的类别

现在我们尝试一个例子，其中图像的位置是远程的（非本地）并且由 URL 指定。在这种情况下，我们需要对 URL 指定的资源（图像）的内容发出 HTTP 请求，然后将响应解码成二进制字节流：

```
from PIL import Image 
import requests                                                  ❶
from io import BytesIO                                           ❷

def remoteImage(url):
        try:
                response = requests.get(url)                     ❸
                return Image.open(BytesIO(response.content))     ❹
        except:
                return None
```

❶ Python 的 HTTP 请求包

❷ Python 的反序列化 I/O 到字节流的包

❸ 请求指定 URL 的图像内容

❹ 将反序列化的内容读取到内存中作为图像

在读取训练图像后，您需要设置通道数以匹配卷积神经网络的输入形状，例如灰度图像的单通道或 RGB 图像的三个通道。

通道数是您图像中的颜色平面的数量。例如，灰度图像将有一个颜色通道。RGB 颜色图像将有三个颜色通道，每个通道分别对应红色、绿色和蓝色。在大多数情况下，这将是一个单通道（灰度）或三个通道（RGB），如图 4.22 所示。

![图片](img/CH04_F22_Ferlitsch.png)

图 4.22 灰度图像有一个通道，RGB 图像有三个通道。

`Image.open()` 方法将根据磁盘上存储的图像的通道数读取图像。因此，如果是一个灰度图像，该方法将作为一个通道读取它；如果是 RGB 图像，它将作为三个通道读取；如果是 RGBA（+alpha 通道），它将作为四个通道读取。

通常，当处理 RGBA 图像时，可以丢弃 alpha 通道。它是设置图像中每个像素透明度的掩码，因此不包含有助于图像识别的其他信息。

一旦图像被读入内存，下一步是将图像转换为与您的神经网络输入形状匹配的通道数。因此，如果神经网络接受灰度图像（单通道），我们希望将其转换为灰度；或者如果神经网络接受 RGB 图像（三个通道），我们希望将其转换为 RGB。`convert()` 方法执行通道转换。参数值 `L` 转换为单通道（灰度），RGB 转换为三个通道（RGB 颜色）。在这里，我们已经更新了 `loadImages()` 函数以包括通道转换：

```
from PIL import Image 
import os

def loadImages(subdir, channels):
        images = []

        files = os.scandir(subdir)
        for file in files:
                   image = Image.open(file.path)
                   if channels == 1:                   ❶
                        image = image.convert('L')     ❶
                   else:                               ❷
                        image = image.convert('RGB')   ❷
                   images.append(image)
        return images

loadImages('cats', 3)                                  ❸
```

❶ 转换为灰度图

❷ 转换为 RGB

❸ 指定转换为 RGB

### 4.9.5 调整大小

到目前为止，您已经看到了如何从磁盘读取图像，获取标签，然后设置通道数以匹配 CNN 输入形状中的通道数。接下来，我们需要调整图像的高度和宽度以最终匹配训练期间输入图像的形状。

例如，一个二维卷积神经网络将具有形式为（高度，宽度，通道）的形状。我们已经处理了通道部分，所以接下来我们需要调整每个图像的像素高度和宽度以匹配输入形状。例如，如果输入形状是（128，128，3），我们希望将每个图像的高度和宽度调整到（128，128）。`resize()` 方法将执行调整大小。

在大多数情况下，您将减小每个图像的大小（下采样）。例如，一个 1024 × 768 的图像将大小为 3 MB。这比神经网络所需的分辨率要高得多（更多细节请见第三章）。当图像被下采样时，一些分辨率（细节）将会丢失。为了最小化下采样时的影响，一个常见的做法是在 PIL 中使用反走样算法。最后，我们将然后将我们的 PIL 图像列表转换为多维数组：

```
from PIL import Image 
import os
import numpy as np

def loadImages(subdir, channels, shape):
        images = []

        files = os.scandir(subdir)
        for file in files:
                image = Image.open(file.path)
                if channels == 1:
                    image = image.convert('L')
                else:
                    image = image.convert('RGB')

                images.append(image.resize(shape, Image.ANTIALIAS))    ❶

        return np.asarray(images)                                      ❷

loadImages('cats', 3, (128, 128))                                      ❸
```

❶ 将图像调整为目标输入形状

❷ 在单次调用中将所有 PIL 图像转换为 NumPy 数组

❸ 指定目标输入大小为 128 × 128。

让我们现在使用 OpenCV 重复前面的步骤。通过使用 `cv2.imread()` 方法将图像读入内存。我发现这个方法的一个优点是输出已经是一个多维 NumPy 数据类型：

```
import cv2

image = cv2.imread('myimage.jpg')
```

OpenCV 相对于 PIL 的另一个优点是您可以在读取图像时进行通道转换，而不是第二步。默认情况下，`cv2.imread()` 将图像转换为三通道 RGB 图像。您可以指定一个第二个参数，以指示要使用的通道转换。在以下示例中，我们在读取图像时进行通道转换：

```
if channel == 1:                                                  ❶
        image = cv2.imread('myimage.jpg', cv2.IMREAD_GRAYSCALE)
else:                                                             ❷
        image = cv2.imread('myimage.jpg', cv2.IMREAD_COLOR)
```

❶ 以单通道（灰度）图像读取图像

❷ 以三通道（彩色）图像读取图像

在下一个示例中，我们从远程位置（`url`）读取图像，并在同一时间进行通道转换。在这种情况下，我们使用 `cv2.imdecode()` 方法： 

```
try:
        response = requests.get(url)
        if channel == 1:
               return cv2.imdecode(BytesIO(response.content),
                                   cv2.IMREAD_GRAYSCALE)
        else:
               return cv2.imdecode(BytesIO(response.content),
                                   cv2.IMREAD_COLOR)
except:
       return None
```

使用 `cv2.resize()` 方法调整图像大小。第二个参数是一个包含调整后图像高度和宽度的元组。可选的（关键字）第三个参数是在调整大小时使用的插值算法。由于在大多数情况下您将进行下采样，一个常见的做法是使用 `cv2.INTER_AREA` 算法以在保留信息和最小化下采样图像时的伪影方面获得最佳结果：

```
image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
```

让我们现在使用 OpenCV 重写 `loadImages()` 函数：

```
import cv2
import os
import numpy as np

def loadImages(subdir, channels, shape):
        images = []

        files = os.scandir(subdir)
        for file in files:
                   if channels == 1:
                       image = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
                   else:
                       image = cv2.imread(file.path, cv2.IMREAD_COLOR)

                   images.append(cv2.resize(image, shape, cv2.INTER_AREA))  ❶
        return np.asarray(images)

loadImages('cats', 3, (128, 128))                                           ❷
```

❶ 将图像调整为目标输入形状

❷ 指定目标输入形状为 128 × 128

## 4.10 模型保存/恢复

在本小节中，我们将介绍训练后的内容：现在你已经训练了一个模型，接下来你该做什么？嗯，你可能会想要保存模型架构和相应的学习权重和偏差（参数），然后随后恢复模型以进行部署。

### 4.10.1 保存

在 TF.Keras 中，我们可以保存模型和训练好的参数（权重和偏差）。模型和权重可以分别保存或一起保存。`save()`方法将权重/偏差和模型保存到 TensorFlow SavedModel 格式的指定文件夹中。以下是一个示例：

```
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)    ❶

model.save('mymodel')                                                ❷
```

❶ 训练模型

❷ 保存模型和训练好的权重和偏差

训练好的权重/偏差和模型可以分别保存。`save_weights()`方法将模型的参数仅保存到 TensorFlow Checkpoint 格式的指定文件夹中。以下是一个示例：

```
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

model.save_weights('myweights')       ❶
```

❶ 仅保存训练好的权重和偏差

### 4.10.2 恢复

在 TF.Keras 中，我们可以恢复模型架构和/或模型参数（权重和偏差）。通常，恢复模型架构是为了加载预构建模型，而同时加载模型架构和模型参数通常用于迁移学习（在第十一章中讨论）。

注意，加载模型和模型参数与检查点不同，因为我们不是在恢复超参数的当前状态。因此，这种方法不应用于持续学习：

```
from tensorflow.keras.models import load_model

model = load_model('mymodel')      ❶
```

❶ 加载预训练模型

在下一个代码示例中，使用`load_weights()`方法将模型的训练好的权重/偏差加载到相应的预构建模型中：

```
from tensorflow.keras.models import load_weights

model = load_model('mymodel')      ❶
model.load_weights('myweights')    ❷
```

❶ 加载预构建模型

❷ 加载模型的预训练权重

## 摘要

+   当一批图像被前向传递时，预测值与真实值之间的差异是损失。优化器使用损失来确定在反向传播中如何更新权重。

+   将一小部分数据集保留为测试数据，不进行训练。训练完成后，使用测试数据来观察模型泛化能力与记忆数据示例之间的差异。

+   在每个 epoch 之后使用验证数据来检测模型过拟合。

+   与归一化相比，像素数据的标准化更受欢迎，因为它有助于略微提高收敛速度。

+   当训练过程中损失值达到平台期时，会发生收敛。

+   超参数用于改进模型的训练，但不是模型的一部分。

+   增强允许使用更少的原始图像进行训练以实现不变性。

+   检查点用于在训练发散后恢复一个好的 epoch，而无需重新启动训练。

+   提前停止通过检测模型不会随着进一步训练而改进来节省训练时间和成本。

+   小数据集可以从内存存储和访问中进行训练，但大数据集是从磁盘存储和访问中进行训练的。

+   训练完成后，保存模型架构和学习的参数，然后随后恢复模型以进行部署。
