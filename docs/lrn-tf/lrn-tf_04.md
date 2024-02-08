# 第四章：卷积神经网络

在本章中，我们介绍卷积神经网络（CNNs）以及与之相关的构建块和方法。我们从对 MNIST 数据集进行分类的简单模型开始，然后介绍 CIFAR10 对象识别数据集，并将几个 CNN 模型应用于其中。尽管小巧快速，但本章介绍的 CNN 在实践中被广泛使用，以获得物体识别任务中的最新结果。

# CNN 简介

在过去几年中，卷积神经网络作为一种特别有前途的深度学习形式获得了特殊地位。根植于图像处理，卷积层已经在几乎所有深度学习的子领域中找到了应用，并且在大多数情况下非常成功。

*全连接*和*卷积*神经网络之间的根本区别在于连续层之间连接的模式。在全连接的情况下，正如名称所示，每个单元都连接到前一层的所有单元。我们在第二章中看到了一个例子，其中 10 个输出单元连接到所有输入图像像素。

另一方面，在神经网络的卷积层中，每个单元连接到前一层中附近的（通常很少）几个单元。此外，所有单元以相同的方式连接到前一层，具有相同的权重和结构。这导致了一种称为*卷积*的操作，给这种架构命名（请参见图 4-1 以了解这个想法的示例）。在下一节中，我们将更详细地介绍卷积操作，但简而言之，对我们来说，这意味着在图像上应用一小部分“窗口”权重（也称为*滤波器*），如稍后的图 4-2 所示。

![](img/letf_0401.png)

###### 图 4-1。在全连接层（左侧），每个单元都连接到前一层的所有单元。在卷积层（右侧），每个单元连接到前一层的一个局部区域中的固定数量的单元。此外，在卷积层中，所有单元共享这些连接的权重，如共享的线型所示。

有一些常常被引用为导致 CNN 方法的动机，来自不同的思想流派。第一个角度是所谓的模型背后的神经科学启发。第二个涉及对图像性质的洞察，第三个与学习理论有关。在我们深入了解实际机制之前，我们将简要介绍这些内容。

通常将神经网络总体描述为计算的生物学启发模型，特别是卷积神经网络。有时，有人声称这些模型“模仿大脑执行计算的方式”。尽管直接理解时会产生误导，但生物类比具有一定的兴趣。

诺贝尔奖获得者神经生理学家 Hubel 和 Wiesel 早在 1960 年代就发现，大脑中视觉处理的第一阶段包括将相同的局部滤波器（例如，边缘检测器）应用于视野的所有部分。神经科学界目前的理解是，随着视觉处理的进行，信息从输入的越来越广泛的部分集成，这是按层次进行的。

卷积神经网络遵循相同的模式。随着我们深入网络，每个卷积层查看图像的越来越大的部分。最常见的情况是，这将被全连接层跟随，这些全连接层在生物启发的类比中充当处理全局信息的更高级别的视觉处理层。

第二个角度，更加注重硬性事实工程方面，源于图像及其内容的性质。当在图像中寻找一个对象，比如一只猫的脸时，我们通常希望能够无论其在图像中的位置如何都能检测到它。这反映了自然图像的性质，即相同的内容可能在图像的不同位置找到。这种性质被称为*不变性*——这种类型的不变性也可以预期在（小）旋转、光照变化等方面存在。

因此，在构建一个对象识别系统时，应该对平移具有不变性（并且，根据情况，可能还对旋转和各种变形具有不变性，但这是另一回事）。简而言之，因此在图像的不同部分执行完全相同的计算是有意义的。从这个角度来看，卷积神经网络层在所有空间区域上计算图像的相同特征。

最后，卷积结构可以被看作是一种正则化机制。从这个角度来看，卷积层就像全连接层，但是我们不是在完整的矩阵空间中寻找权重，而是将搜索限制在描述固定大小卷积的矩阵中，将自由度的数量减少到卷积的大小，这通常非常小。

##### 正则化

术语*正则化*在本书中被广泛使用。在机器学习和统计学中，正则化主要用于指的是通过对解的复杂性施加惩罚来限制优化问题，以防止对给定示例的过度拟合。

过拟合发生在规则（例如，分类器）以解释训练集的方式计算时，但对未见数据的泛化能力较差。

正则化通常通过添加关于期望结果的隐式信息来实现（这可能采取的形式是说在搜索函数空间时我们更希望有一个更平滑的函数）。在卷积神经网络的情况下，我们明确表示我们正在寻找相对低维子空间中的权重，这些权重对应于固定大小的卷积。

在本章中，我们涵盖了与卷积神经网络相关的层和操作类型。我们首先重新审视 MNIST 数据集，这次应用一个准确率约为 99%的模型。接下来，我们将转向更有趣的对象识别 CIFAR10 数据集。

# MNIST：第二次

在本节中，我们再次查看 MNIST 数据集，这次将一个小型卷积神经网络应用作为我们的分类器。在这样做之前，有几个元素和操作我们必须熟悉。

## 卷积

卷积操作，正如你可能从架构的名称中期待的那样，是卷积神经网络中连接层的基本手段。我们使用内置的 TensorFlow `conv2d()`：

```py
tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

```

在这里，`x`是数据——输入图像，或者是在网络中进一步应用之前的卷积层后获得的下游特征图。正如之前讨论的，在典型的 CNN 模型中，我们按层次堆叠卷积层，并且*特征图*只是一个常用术语，指的是每个这样的层的输出。查看这些层的输出的另一种方式是*处理后的图像*，是应用滤波器和其他操作的结果。在这里，这个滤波器由`W`参数化，表示我们网络中学习的卷积滤波器的权重。这只是我们在图 4-2 中看到的小“滑动窗口”中的一组权重。

![](img/letf_0402.png)

###### 图 4-2。相同的卷积滤波器——一个“滑动窗口”——应用于图像之上。

这个操作的输出将取决于`x`和`W`的形状，在我们的情况下是四维的。图像数据`x`的形状将是：

```py
[None, 28, 28, 1]

```

这意味着我们有未知数量的图像，每个图像为 28×28 像素，具有一个颜色通道（因为这些是灰度图像）。我们使用的权重`W`的形状将是：

```py
[5, 5, 1, 32]

```

初始的 5×5×1 表示在图像中要进行卷积的小“窗口”的大小，在我们的情况下是一个 5×5 的区域。在具有多个颜色通道的图像中（RGB，如第一章中简要讨论的），我们将每个图像视为 RGB 值的三维张量，但在这个单通道数据中它们只是二维的，卷积滤波器应用于二维区域。稍后，当我们处理 CIFAR10 数据时，我们将看到多通道图像的示例以及如何相应地设置权重`W`的大小。

最终的 32 是特征图的数量。换句话说，我们有卷积层的多组权重——在这种情况下有 32 组。回想一下，卷积层的概念是沿着图像计算相同的特征；我们希望计算许多这样的特征，因此使用多组卷积滤波器。

`strides`参数控制滤波器`W`在图像（或特征图）`x`上的空间移动。

值`[1, 1, 1, 1]`表示滤波器在每个维度上以一个像素间隔应用于输入，对应于“全”卷积。此参数的其他设置允许我们在应用滤波器时引入跳跃—这是我们稍后会应用的常见做法—从而使得生成的特征图更小。

最后，将`padding`设置为`'SAME'`意味着填充`x`的边界，使得操作的结果大小与`x`的大小相同。

# 激活函数

在线性层之后，无论是卷积还是全连接，常见的做法是应用非线性*激活函数*（参见图 4-3 中的一些示例）。激活函数的一个实际方面是，连续的线性操作可以被单个操作替代，因此深度不会为模型的表达能力做出贡献，除非我们在线性层之间使用非线性激活。

![](img/letf_0403.png)

###### 图 4-3。常见的激活函数：逻辑函数（左）、双曲正切函数（中）、修正线性单元（右）

## 池化

在卷积层后跟随输出的池化是常见的。技术上，*池化*意味着使用某种本地聚合函数减少数据的大小，通常在每个特征图内部。

这背后的原因既是技术性的，也是更理论性的。技术方面是，池化会减少下游处理的数据量。这可以极大地减少模型中的总参数数量，特别是在卷积层之后使用全连接层的情况下。

应用池化的更理论的原因是，我们希望我们计算的特征不受图像中位置的微小变化的影响。例如，一个在图像右上部寻找眼睛的特征，如果我们稍微向右移动相机拍摄图片，将眼睛略微移动到图像中心，这个特征不应该有太大变化。在空间上聚合“眼睛检测器特征”使模型能够克服图像之间的这种空间变化，捕捉本章开头讨论的某种不变性形式。

在我们的示例中，我们对每个特征图的 2×2 块应用最大池化操作：

```py
tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

最大池化输出预定义大小的每个区域中的输入的最大值（这里是 2×2）。`ksize`参数控制池化的大小（2×2），`strides`参数控制我们在`x`上“滑动”池化网格的幅度，就像在卷积层的情况下一样。将其设置为 2×2 网格意味着池化的输出将恰好是原始高度和宽度的一半，总共是原始大小的四分之一。

## Dropout

我们模型中最后需要的元素是*dropout*。这是一种正则化技巧，用于强制网络将学习的表示分布到所有神经元中。在训练期间，dropout 会“关闭”一定比例的层中的单位，通过将它们的值设置为零。这些被丢弃的神经元是随机的，每次计算都不同，迫使网络学习一个即使在丢失后仍能正常工作的表示。这个过程通常被认为是训练多个网络的“集成”，从而增加泛化能力。在测试时使用网络作为分类器时（“推断”），不会进行 dropout，而是直接使用完整的网络。

在我们的示例中，除了我们希望应用 dropout 的层之外，唯一的参数是`keep_prob`，即每一步保持工作的神经元的比例：

```py
tf.nn.dropout(layer, keep_prob=keep_prob)

```

为了能够更改此值（我们必须这样做，因为对于测试，我们希望这个值为`1.0`，表示根本没有丢失），我们将使用`tf.placeholder`并传递一个值用于训练（`.5`）和另一个用于测试（`1.0`）。

## 模型

首先，我们定义了一些辅助函数，这些函数将在本章中广泛使用，用于创建我们的层。这样做可以使实际模型简短易读（在本书的后面，我们将看到存在几种框架，用于更抽象地定义深度学习构建块，这样我们可以专注于快速设计我们的网络，而不是定义所有必要的元素的繁琐工作）。我们的辅助函数有：

```py
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

```

让我们更仔细地看看这些：

`weight_variable()`

这指定了网络的全连接或卷积层的权重。它们使用截断正态分布进行随机初始化，标准差为 0.1。这种使用截断在尾部的随机正态分布初始化是相当常见的，通常会产生良好的结果（请参见即将介绍的随机初始化的注释）。

`bias_variable()`

这定义了全连接或卷积层中的偏置元素。它们都使用常数值`.1`进行初始化。

`conv2d()`

这指定了我们通常会使用的卷积。一个完整的卷积（没有跳过），输出与输入大小相同。

`max_pool_2×2`

这将将最大池设置为高度/宽度维度的一半大小，并且总体上是特征图大小的四分之一。

`conv_layer()`

这是我们将使用的实际层。线性卷积如`conv2d`中定义的，带有偏置，然后是 ReLU 非线性。

`full_layer()`

带有偏置的标准全连接层。请注意，这里我们没有添加 ReLU。这使我们可以在最终输出时使用相同的层，我们不需要非线性部分。

定义了这些层后，我们准备设置我们的模型（请参见图 4-4 中的可视化）：

```py
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)

```

![CNN](img/letf_0404.png)

###### 图 4-4. 所使用的 CNN 架构的可视化。

# 随机初始化

在前一章中，我们讨论了几种类型的初始化器，包括此处用于卷积层权重的随机初始化器：

```py
initial = tf.truncated_normal(shape, stddev=0.1)
```

关于深度学习模型训练中初始化的重要性已经说了很多。简而言之，糟糕的初始化可能会使训练过程“卡住”，或者由于数值问题完全失败。使用随机初始化而不是常数初始化有助于打破学习特征之间的对称性，使模型能够学习多样化和丰富的表示。使用边界值有助于控制梯度的幅度，使网络更有效地收敛，等等。

我们首先为图像和正确标签定义占位符`x`和`y_`。接下来，我们将图像数据重新整形为尺寸为 28×28×1 的 2D 图像格式。回想一下，我们在之前的 MNIST 模型中不需要数据的空间方面，因为所有像素都是独立处理的，但在卷积神经网络框架中，考虑图像时利用这种空间含义是一个重要的优势。

接下来我们有两个连续的卷积和池化层，每个层都有 5×5 的卷积和 32 个特征图，然后是一个具有 1,024 个单元的单个全连接层。在应用全连接层之前，我们将图像展平为单个向量形式，因为全连接层不再需要空间方面。

请注意，在两个卷积和池化层之后，图像的尺寸为 7×7×64。原始的 28×28 像素图像首先缩小到 14×14，然后在两个池化操作中缩小到 7×7。64 是我们在第二个卷积层中创建的特征图的数量。在考虑模型中学习参数的总数时，大部分将在全连接层中（从 7×7×64 到 1,024 的转换给我们提供了 3.2 百万个参数）。如果我们没有使用最大池化，这个数字将是原来的 16 倍（即 28×28×64×1,024，大约为 51 百万）。

最后，输出是一个具有 10 个单元的全连接层，对应数据集中的标签数量（回想一下 MNIST 是一个手写数字数据集，因此可能的标签数量是 10）。

其余部分与第二章中第一个 MNIST 模型中的内容相同，只有一些细微的变化：

`train_accuracy`

我们在每 100 步打印模型在用于训练的批次上的准确率。这是在训练步骤之前完成的，因此是对模型在训练集上当前性能的良好估计。

`test_accuracy`

我们将测试过程分为 10 个包含 1,000 张图像的块。对于更大的数据集，这样做非常重要。

以下是完整的代码：

```py
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=
                                                           y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        batch = mnist.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], 
                                                           y_: batch[1],
                                                           keep_prob: 1.0})
            print "step {}, training accuracy {}".format(i, train_accuracy)

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],
                                        keep_prob: 0.5})

    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean([sess.run(accuracy, 
                            feed_dict={x:X[i], y_:Y[i],keep_prob:1.0}) 
                            for i in range(10)])

print "test accuracy: {}".format(test_accuracy)

```

这个模型的性能已经相当不错，仅经过 5 个周期，准确率就超过了 99%，¹这相当于 5,000 步，每个步骤的迷你批次大小为 50。

有关多年来使用该数据集的模型列表以及如何进一步改进结果的一些想法，请查看[*http://yann.lecun.com/exdb/mnist/*](http://yann.lecun.com/exdb/mnist/)。

# CIFAR10

*CIFAR10*是另一个在计算机视觉和机器学习领域有着悠久历史的数据集。与 MNIST 类似，它是一个常见的基准，各种方法都会被测试。[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)是一个包含 60,000 张尺寸为 32×32 像素的彩色图像的数据集，每张图像属于以下十个类别之一：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。

针对这个数据集的最先进的深度学习方法在分类这些图像方面与人类一样出色。在本节中，我们首先使用相对较简单的方法，这些方法将运行相对较快。然后，我们简要讨论这些方法与最先进方法之间的差距。

## 加载 CIFAR10 数据集

在本节中，我们构建了一个类似于用于 MNIST 的内置`input_data.read_data_sets()`的 CIFAR10 数据管理器。²

首先，[下载数据集的 Python 版本](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)并将文件提取到本地目录中。现在应该有以下文件：

+   *data_batch_1*, *data_batch_2*, *data_batch_3*, *data_batch_4*, *data_batch_5*

+   *test_batch*

+   *batches_meta*

+   *readme.html*

*data_batch_X*文件是包含训练数据的序列化数据文件，*test_batch*是一个类似的包含测试数据的序列化文件。*batches_meta*文件包含从数字到语义标签的映射。*.html*文件是 CIFAR-10 数据集网页的副本。

由于这是一个相对较小的数据集，我们将所有数据加载到内存中：

```py
class CifarLoader(object):
def __init__(self, source_files):
self._source = source_files
self._i = 0
self.images = None
self.labels = None

def load(self):
data = [unpickle(f) for f in self._source]
images = np.vstack([d["data"] for d in data])
n = len(images)
self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)\
.astype(float) / 255
self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
return self

def next_batch(self, batch_size):
x, y = self.images[self._i:self._i+batch_size], 
self.labels[self._i:self._i+batch_size]
self._i = (self._i + batch_size) % len(self.images)
return x, y

```

在这里我们使用以下实用函数：

```py
DATA_PATH="*`/path/to/CIFAR10`*"defunpickle(file):withopen(os.path.join(DATA_PATH,file),'rb')asfo:dict=cPickle.load(fo)returndictdefone_hot(vec,vals=10):n=len(vec)out=np.zeros((n,vals))out[range(n),vec]=1returnout
```

`unpickle()`函数返回一个带有`data`和`labels`字段的`dict`，分别包含图像数据和标签。`one_hot()`将标签从整数（范围为 0 到 9）重新编码为长度为 10 的向量，其中除了标签位置上的 1 之外，所有位置都是 0。

最后，我们创建一个包含训练和测试数据的数据管理器：

```py
class CifarDataManager(object):
def __init__(self):
self.train = CifarLoader(["data_batch_{}".format(i) 
for i in range(1, 6)])
.load()
self.test = CifarLoader(["test_batch"]).load()

```

使用 Matplotlib，我们现在可以使用数据管理器来显示一些 CIFAR10 图像，并更好地了解这个数据集中的内容：

```py
def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()

d = CifarDataManager()
print "Number of train images: {}".format(len(d.train.images))
print "Number of train labels: {}".format(len(d.train.labels))
print "Number of test images: {}".format(len(d.test.images))
print "Number of test images: {}".format(len(d.test.labels))
images = d.train.images
display_cifar(images, 10)

```

# Matplotlib

*Matplotlib*是一个用于绘图的有用的 Python 库，设计得看起来和行为类似于 MATLAB 绘图。这通常是快速绘制和可视化数据集的最简单方法。

`display_cifar()`函数的参数是`images`（包含图像的可迭代对象）和`size`（我们想要显示的图像数量），并构建并显示一个`size×size`的图像网格。这是通过垂直和水平连接实际图像来形成一个大图像。

在显示图像网格之前，我们首先打印训练/测试集的大小。CIFAR10 包含 50K 个训练图像和 10K 个测试图像：

```py
Number of train images: 50000
Number of train labels: 50000
Number of test images: 10000
Number of test images: 10000

```

在图 4-5 中生成并显示的图像旨在让人了解 CIFAR10 图像实际上是什么样子的。值得注意的是，这些小的 32×32 像素图像每个都包含一个完整的单个对象，该对象位于中心位置，即使在这种分辨率下也基本上是可识别的。

![CIFAR10](img/letf_0405.png)

###### 图 4-5。100 个随机的 CIFAR10 图像。

## 简单的 CIFAR10 模型

我们将从先前成功用于 MNIST 数据集的模型开始。回想一下，MNIST 数据集由 28×28 像素的灰度图像组成，而 CIFAR10 图像是带有 32×32 像素的彩色图像。这将需要对计算图的设置进行轻微调整：

```py
cifar = CifarDataManager()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x, shape=[5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, 
                                                               y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def test(sess):
X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
Y = cifar.test.labels.reshape(10, 1000, 10)
acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], 
                                             keep_prob: 1.0})
for i in range(10)])
print "Accuracy: {:.4}%".format(acc * 100)

with tf.Session() as sess:
sess.run(tf.global_variables_initializer())

for i in range(STEPS):
batch = cifar.train.next_batch(BATCH_SIZE)
sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], 
                                keep_prob: 0.5})

test(sess)

```

这第一次尝试将在几分钟内达到大约 70%的准确度（使用批量大小为 100，自然取决于硬件和配置）。这好吗？截至目前，最先进的深度学习方法在这个数据集上实现了超过 95%的准确度，但是使用更大的模型并且通常需要许多小时的训练。

这与之前介绍的类似 MNIST 模型之间存在一些差异。首先，输入由大小为 32×32×3 的图像组成，第三维是三个颜色通道：

```py
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

```

同样，在两次池化操作之后，我们这次剩下的是大小为 8×8 的 64 个特征图：

```py
conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

```

最后，为了方便起见，我们将测试过程分组到一个名为`test()`的单独函数中，并且我们不打印训练准确度值（可以使用与 MNIST 模型中相同代码添加回来）。

一旦我们有了一些可接受的基线准确度的模型（无论是从简单的 MNIST 模型还是从其他数据集的最先进模型中派生的），一个常见的做法是通过一系列的适应和更改来尝试改进它，直到达到我们的目的所需的内容。

在这种情况下，保持其他所有内容不变，我们将添加一个具有 128 个特征图和 dropout 的第三个卷积层。我们还将把完全连接层中的单元数从 1,024 减少到 512：

```py
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x, shape=[5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

conv3 = conv_layer(conv2_pool, shape=[5, 5, 64, 128])
conv3_pool = max_pool_2x2(conv3)
conv3_flat = tf.reshape(conv3_pool, [-1, 4 * 4 * 128])
conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

full_1 = tf.nn.relu(full_layer(conv3_drop, 512))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)
```

这个模型将需要稍长一点的时间来运行（但即使没有复杂的硬件，也不会超过一个小时），并且可以达到大约 75%的准确度。

这仍然与最佳已知方法之间存在相当大的差距。有几个独立适用的元素可以帮助缩小这个差距：

模型大小

对于这种数据集和类似数据集，大多数成功的方法使用更深的网络和更多可调参数。

其他类型的层和方法

通常与这里介绍的层一起使用的是其他类型的流行层，比如局部响应归一化。

优化技巧

更多关于这个的内容以后再说！

领域知识

利用领域知识进行预处理通常是很有帮助的。在这种情况下，这将是传统的图像处理。

数据增强

基于现有数据集添加训练数据可能会有所帮助。例如，如果一张狗的图片水平翻转，那么显然仍然是一张狗的图片（但垂直翻转呢？）。小的位移和旋转也经常被使用。

重用成功的方法和架构

和大多数工程领域一样，从一个经过时间验证的方法开始，并根据自己的需求进行调整通常是正确的方式。在深度学习领域，这经常通过微调预训练模型来实现。

我们将在本章中介绍的最终模型是实际为这个数据集产生出色结果的模型类型的缩小版本。这个模型仍然紧凑快速，在大约 150 个 epochs 后达到约 83%的准确率：

```py
C1, C2, C3 = 30, 50, 80
F1 = 500

conv1_1 = conv_layer(x, shape=[3, 3, 3, C1])
conv1_2 = conv_layer(conv1_1, shape=[3, 3, C1, C1])
conv1_3 = conv_layer(conv1_2, shape=[3, 3, C1, C1])
conv1_pool = max_pool_2x2(conv1_3)
conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=keep_prob)

conv2_1 = conv_layer(conv1_drop, shape=[3, 3, C1, C2])
conv2_2 = conv_layer(conv2_1, shape=[3, 3, C2, C2])
conv2_3 = conv_layer(conv2_2, shape=[3, 3, C2, C2])
conv2_pool = max_pool_2x2(conv2_3)
conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)

conv3_1 = conv_layer(conv2_drop, shape=[3, 3, C2, C3])
conv3_2 = conv_layer(conv3_1, shape=[3, 3, C3, C3])
conv3_3 = conv_layer(conv3_2, shape=[3, 3, C3, C3])
conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], 
                           padding='SAME')
conv3_flat = tf.reshape(conv3_pool, [-1, C3])
conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

full1 = tf.nn.relu(full_layer(conv3_drop, F1))
full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)

```

这个模型由三个卷积层块组成，接着是我们之前已经见过几次的全连接和输出层。每个卷积层块包含三个连续的卷积层，然后是一个池化层和 dropout。

常数`C1`、`C2`和`C3`控制每个卷积块中每个层的特征图数量，常数`F1`控制全连接层中的单元数量。

在第三个卷积层之后，我们使用了一个 8×8 的最大池层：

```py
conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], 
                           padding='SAME')

```

由于在这一点上特征图的大小为 8×8（在前两个池化层之后，每个轴上都将 32×32 的图片减半），这样全局池化每个特征图并保留最大值。第三个块的特征图数量设置为 80，所以在这一点上（在最大池化之后），表示被减少到只有 80 个数字。这使得模型的整体大小保持较小，因为在过渡到全连接层时参数的数量保持在 80×500。

# 总结

在本章中，我们介绍了卷积神经网络及其通常由各种构建模块组成。一旦你能够正确运行小型模型，请尝试运行更大更深的模型，遵循相同的原则。虽然你可以随时查看最新的文献并了解哪些方法有效，但通过试错和自己摸索也能学到很多。在接下来的章节中，我们将看到如何处理文本和序列数据，以及如何使用 TensorFlow 抽象来轻松构建 CNN 模型。

¹ 在机器学习和特别是深度学习中，*epoch*指的是对所有训练数据的一次完整遍历；即，当学习模型已经看到每个训练示例一次时。

² 这主要是为了说明的目的。已经存在包含这种数据包装器的开源库，适用于许多流行的数据集。例如，查看 Keras 中的数据集模块（`keras.datasets`），特别是`keras.datasets.cifar10`。

³ 参见[谁在 CIFAR-10 中表现最好？](http://bit.ly/2srV5OO)以获取方法列表和相关论文。
