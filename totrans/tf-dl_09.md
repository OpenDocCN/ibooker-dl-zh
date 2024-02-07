# 第九章。训练大型深度网络

到目前为止，您已经看到了如何训练可以完全在一台好的笔记本电脑上训练的小模型。所有这些模型都可以在配备 GPU 的硬件上运行，并获得显著的速度提升（除了在前一章讨论的强化学习模型的显著例外）。然而，训练更大的模型仍然需要相当高的技术含量。在本章中，我们将讨论可以用于训练深度网络的各种硬件类型，包括图形处理单元（GPU）、张量处理单元（TPU）和神经形态芯片。我们还将简要介绍用于更大深度学习模型的分布式训练原则。我们将以一个深入的案例研究结束本章，该案例改编自 TensorFlow 教程之一，演示如何在具有多个 GPU 的服务器上训练 CIFAR-10 卷积神经网络。我们建议您尝试自己运行此代码，但也承认获得多 GPU 服务器的访问权限比找到一台好的笔记本电脑更加困难。幸运的是，云上的多 GPU 服务器访问正在变得可能，并且很可能是 TensorFlow 工业用户寻求训练大型模型的最佳解决方案。

# 深度网络的定制硬件

正如您在整本书中所看到的，深度网络训练需要在数据的小批量上重复执行张量操作链。张量操作通常通过软件转换为矩阵乘法操作，因此深度网络的快速训练基本上取决于能够快速执行矩阵乘法操作的能力。虽然 CPU 完全可以实现矩阵乘法，但 CPU 硬件的通用性意味着很多精力将被浪费在数学运算不需要的开销上。

硬件工程师多年来已经注意到这一事实，存在多种用于处理深度网络的替代硬件。这种硬件可以广泛分为*仅推断*或*训练和推断*。仅推断硬件不能用于训练新的深度网络，但可以用于在生产中部署经过训练的模型，从而可能实现性能的数量级增加。训练和推断硬件允许原生地训练模型。目前，由于 Nvidia 团队在软件和推广方面的重大投资，Nvidia 的 GPU 硬件在训练和推断市场上占据主导地位，但许多其他竞争对手也在紧追 GPU 的脚步。在本节中，我们将简要介绍一些这些新型硬件的替代方案。除了 GPU 和 CPU 外，这些替代形式的硬件大多尚未广泛可用，因此本节的大部分内容是前瞻性的。

# CPU 训练

尽管 CPU 训练绝不是训练深度网络的最先进技术，但对于较小的模型通常表现良好（正如您在本书中亲身见证的）。对于强化学习问题，多核 CPU 机器甚至可以胜过 GPU 训练。

CPU 也广泛用于深度网络的推断应用。大多数公司都在大力投资开发主要基于英特尔服务器盒的云服务器。很可能，广泛部署的第一代深度网络（在科技公司之外）将主要部署在这样的英特尔服务器上。虽然基于 CPU 的部署对于学习模型的重度部署来说并不足够，但对于第一批客户需求通常是足够的。图 9-1 展示了一个标准的英特尔 CPU。

![cpu_pic.jpg](img/tfdl_0901.png)

###### 图 9-1. 英特尔的 CPU。 CPU 仍然是计算机硬件的主导形式，并且存在于所有现代笔记本电脑、台式机、服务器和手机中。大多数软件都是编写为在 CPU 上执行的。数值计算（如神经网络训练）可以在 CPU 上执行，但可能比针对数值方法进行优化的定制硬件慢。

## GPU 训练

GPU 首先是为了执行图形社区所需的计算而开发的。在一个幸运的巧合中，事实证明用于定义图形着色器的基本功能可以重新用于执行深度学习。在它们的数学核心中，图形和机器学习都严重依赖于矩阵乘法。从经验上看，GPU 矩阵乘法比 CPU 实现快一个数量级或两个数量级。 GPU 是如何成功地做到这一点的呢？诀窍在于 GPU 利用了成千上万个相同的线程。聪明的黑客已经成功地将矩阵乘法分解为大规模并行操作，可以提供显著的加速。图 9-2 说明了 GPU 的架构。

尽管有许多 GPU 供应商，但 Nvidia 目前主导着 GPU 市场。 Nvidia 的 GPU 的许多强大功能源自其自定义库 CUDA（计算统一设备架构），该库提供了使编写 GPU 程序更容易的基本功能。 Nvidia 提供了一个 CUDA 扩展，CUDNN，用于加速深度网络（图 9-2）。 TensorFlow 内置了 CUDNN 支持，因此您可以通过 TensorFlow 也利用 CUDNN 来加速您的网络。

![GPU_architecture.jpg](img/tfdl_0902.png)

###### 图 9-2. Nvidia 的 GPU 架构。 GPU 比 CPU 拥有更多的核心，非常适合执行数值线性代数，这对图形和机器学习计算都很有用。 GPU 已经成为训练深度网络的主要硬件平台。

# 晶体管尺寸有多重要？

多年来，半导体行业一直通过观察晶体管尺寸的进展来跟踪芯片速度的发展。随着晶体管变小，更多的晶体管可以被打包到标准芯片上，算法可以运行得更快。在撰写本书时，英特尔目前正在使用 10 纳米晶体管，并正在努力过渡到 7 纳米。近年来，晶体管尺寸的缩小速度已经显著放缓，因为在这些尺度上会出现严重的散热问题。

Nvidia 的 GPU 在某种程度上打破了这一趋势。它们倾向于使用比英特尔最好的晶体管尺寸落后一两代的尺寸，并专注于解决架构和软件瓶颈，而不是晶体管工程。到目前为止，Nvidia 的策略已经取得了成功，并且该公司在机器学习芯片领域实现了市场主导地位。

目前尚不清楚架构和软件优化能走多远。 GPU 优化是否很快会遇到与 CPU 相同的摩尔定律障碍？还是聪明的架构创新将使 GPU 更快数年？只有时间能告诉我们。

## 张量处理单元

张量处理单元（TPU）是由谷歌设计的定制 ASIC（特定应用集成电路），旨在加速在 TensorFlow 中设计的深度学习工作负载。与 GPU 不同，TPU 被简化并且仅实现了在芯片上执行必要矩阵乘法所需的最低限度。与 GPU 不同，TPU 依赖于相邻的 CPU 来完成大部分预处理工作。这种精简的方法使 TPU 能够以更低的能源成本实现比 GPU 更高的速度。

第一个版本的 TPU 只允许对经过训练的模型进行推断，但最新版本（TPU2）也允许对（某些）深度网络进行训练。然而，谷歌并没有发布关于 TPU 的许多细节，访问权限仅限于谷歌的合作伙伴，计划通过谷歌云实现 TPU 访问。英伟达正在从 TPU 中汲取经验，未来的英伟达 GPU 很可能会变得类似于 TPU，因此终端用户可能会从谷歌的创新中受益，无论是谷歌还是英伟达赢得了消费者深度学习市场。图 9-3 展示了 TPU 架构设计。

![TPU_architecture.jpg](img/tfdl_0903.png)

###### 图 9-3. 谷歌的张量处理单元（TPU）架构。TPU 是由谷歌设计的专用芯片，旨在加速深度学习工作负载。TPU 是一个协处理器，而不是一个独立的硬件部件。

# 什么是 ASICs？

CPU 和 GPU 都是通用芯片。CPU 通常支持汇编指令集，并设计为通用。为了实现广泛的应用，需要小心设计。GPU 不太通用，但仍允许通过诸如 CUDA 等语言实现广泛的算法。

应用特定集成电路（ASICs）试图摆脱通用性，而是专注于特定应用的需求。从历史上看，ASICs 只在市场上取得了有限的渗透率。摩尔定律的持续发展意味着通用 CPU 仅仅落后于定制 ASICs 一两步，因此硬件设计开销通常不值得付出努力。

在过去几年中，这种状况已经开始发生变化。晶体管尺寸的减小放缓了 ASIC 的使用。例如，比特币挖掘完全依赖于实现专门密码操作的定制 ASICs。

## 现场可编程门阵列

现场可编程门阵列（FPGAs）是一种“现场可编程”的 ASIC 类型。标准 FPGAs 通常可以通过硬件描述语言（如 Verilog）重新配置，以动态实现新的 ASIC 设计。虽然 FPGAs 通常不如定制 ASICs 高效，但它们可以比 CPU 实现提供显著的速度提升。特别是微软已经使用 FPGAs 执行深度学习推断，并声称在部署中取得了显著的加速。然而，这种方法在微软之外尚未广泛流行。

## 神经形态芯片

深度网络中的“神经元”在数学上模拟了 1940 年代对神经生物学的理解。不用说，自那时以来，对神经元行为的生物学理解已经取得了巨大进展。首先，现在已知深度网络中使用的非线性激活并不是神经元非线性的准确模型。“脉冲列”是一个更好的模型（见图 9-4），在这个模型中，神经元以短暂的脉冲（脉冲）激活，但大部分时间处于背景状态。

![spike_trains.jpg](img/tfdl_0904.png)

###### 图 9-4. 神经元经常以短暂的脉冲列（A）激活。神经形态芯片试图在计算硬件中模拟脉冲行为。生物神经元是复杂的实体（B），因此这些模型仍然只是近似。

硬件工程师花费了大量精力探索是否可以基于脉冲列而不是现有电路技术（CPU、GPU、ASIC）创建芯片设计。这些设计师认为，今天的芯片设计受到基本功耗限制的影响；大脑消耗的能量比计算机芯片少几个数量级，智能设计应该从大脑的结构中学习。

许多项目已经构建了大型脉冲列车芯片，试图扩展这一核心论点。IBM 的 TrueNorth 项目成功地构建了具有数百万“神经元”的脉冲列车处理器，并证明了这种硬件可以以比现有芯片设计低得多的功耗要求执行基本图像识别。然而，尽管取得了这些成功，但如何将现代深度架构转换为脉冲列车芯片尚不清楚。如果无法将 TensorFlow 模型“编译”到脉冲列车硬件上，这些项目在不久的将来可能不会被广泛采用。

# 分布式深度网络训练

在前一节中，我们调查了训练深度网络的各种硬件选项。然而，大多数组织可能只能访问 CPU 和可能是 GPU。幸运的是，可以对深度网络进行*分布式训练*，其中多个 CPU 或 GPU 用于更快更有效地训练模型。图 9-5 说明了使用多个 CPU/GPU 训练深度网络的两种主要范式，即数据并行和模型并行训练。您将在接下来的两节中更详细地了解这些方法。

![parallelism_modes.jpg](img/tfdl_0905.png)

###### 图 9-5。数据并行和模型并行是深度架构的分布式训练的两种主要模式。数据并行训练将大型数据集分割到多个计算节点上，而模型并行训练将大型模型分割到多个节点上。接下来的两节将更深入地介绍这两种方法。

## 数据并行 ism

数据并行 ism 是最常见的多节点深度网络训练类型。数据并行模型将大型数据集分割到不同的机器上。大多数节点是工作节点，并且可以访问用于训练网络的总数据的一部分。每个工作节点都有一个正在训练的模型的完整副本。一个节点被指定为监督员，定期从工作节点收集更新的权重，并将平均版本的权重推送到工作节点。请注意，您在本书中已经看到了一个数据并行的示例；第八章中介绍的 A3C 实现是数据并行深度网络训练的一个简单示例。

作为历史注记，Google 的 TensorFlow 前身 DistBelief 是基于 CPU 服务器上的数据并行训练。该系统能够实现与 GPU 训练速度相匹配或超过的分布式 CPU 速度（使用 32-128 个节点）。图 9-6 说明了 Dist⁠Belief 实现的数据并行训练方法。然而，像 DistBelief 这样的系统的成功往往取决于具有高吞吐量网络互连的存在，这可以实现快速的模型参数共享。许多组织缺乏能够实现有效的多节点数据并行 CPU 训练的网络基础设施。然而，正如 A3C 示例所示，可以在单个节点上使用不同的 CPU 核心执行数据并行训练。对于现代服务器，还可以在单个服务器内使用多个 GPU 执行数据并行训练，我们稍后将向您展示。

![downpour_sgd.png](img/tfdl_0906.png)

###### 图 9-6。Downpour 随机梯度下降（SGD）方法维护模型的多个副本，并在数据集的不同子集上对其进行训练。这些碎片的学习权重定期同步到存储在参数服务器上的全局权重。

## 模型并行 ism

人类大脑是唯一已知的智能硬件的例子，因此自然会对深度网络的复杂性和大脑的复杂性进行比较。简单的论点指出，大脑大约有 1000 亿个神经元；构建具有这么多“神经元”的深度网络是否足以实现普遍智能？不幸的是，这种论点忽略了生物神经元比“数学神经元”复杂得多的事实。因此，简单的比较价值有限。尽管如此，近几年来，构建更大的深度网络一直是主要的研究重点。

训练非常大的深度网络的主要困难在于 GPU 的内存通常有限（通常为几十吉字节）。即使进行仔细编码，具有数亿个参数的神经网络也无法在单个 GPU 上训练，因为内存要求太高。模型并行训练算法试图通过将大型深度网络存储在多个 GPU 的内存中来规避这一限制。一些团队已成功在 GPU 阵列上实现了这些想法，以训练具有数十亿参数的深度网络。不幸的是，迄今为止，这些模型尚未显示出通过额外困难来证明性能改进的效果。目前看来，使用较小模型增加实验便利性的好处超过了模型并行 ism 的收益。

# 硬件内存互连

启用模型并行 ism 需要在计算节点之间具有非常高的带宽连接，因为每次梯度更新都需要节点间通信。请注意，虽然数据并行 ism 需要强大的互连，但同步操作只需要在多次本地梯度更新后偶尔执行。

一些团队使用 InfiniBand 互连（InfiniBand 是一种高吞吐量、低延迟的网络标准）或 Nvidia 的专有 NVLink 互连来尝试构建这样的大型模型。然而，迄今为止，这些实验的结果并不一致，而且这些系统的硬件要求往往昂贵。

# 在 Cifar10 上使用多个 GPU 进行数据并行训练

在本节中，我们将深入介绍如何在 Cifar10 基准集上训练数据并行卷积网络。Cifar10 由尺寸为 32×32 的 60,000 张图像组成。Cifar10 数据集经常用于评估卷积架构。图 9-7 显示了 Cifar10 数据集中的样本图像。

![cifar10.png](img/tfdl_0907.png)

###### 图 9-7。Cifar10 数据集包含来自 10 个类别的 60,000 张图像。这里显示了各种类别的一些样本图像。

本节中将使用的架构在不同的 GPU 上加载模型架构的单独副本，并定期同步跨核心学习的权重，如图 9-8 所示。

![cifar_parallelism.png](img/tfdl_0908.png)

###### 图 9-8。本章将训练的数据并行架构。

## 下载和加载数据

`read_cifar10()`方法读取和解析 Cifar10 原始数据文件。示例 9-1 使用`tf.FixedLengthRecordReader`从 Cifar10 文件中读取原始数据。

##### 示例 9-1。此函数从 Cifar10 原始数据文件中读取和解析数据

```py
def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

 Recommendation: if you want N-way read parallelism, call this function
 N times.  This will give you N independent Readers reading different
 files & positions within those files, which will give better mixing of
 examples.

 Args:
 filename_queue: A queue of strings with the filenames to read from.

 Returns:
 An object representing a single example, with the following fields:
 height: number of rows in the result (32)
 width: number of columns in the result (32)
 depth: number of color channels in the result (3)
 key: a scalar string Tensor describing the filename & record number
 for this example.
 label: an int32 Tensor with the label in the range 0..9.
 uint8image:: a [height, width, depth] uint8 Tensor with the image data
 """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result
```

## 深入探讨架构

网络的架构是一个标准的多层卷积网络，类似于您在第六章中看到的 LeNet5 架构的更复杂版本。`inference()`方法构建了架构（示例 9-2）。这个卷积架构遵循一个相对标准的架构，其中卷积层与本地归一化层交替出现。

##### 示例 9-2。此函数构建 Cifar10 架构

```py
def inference(images):
  """Build the CIFAR10 model.

 Args:
 images: Images returned from distorted_inputs() or inputs().

 Returns:
 Logits.
 """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, cifar10.NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [cifar10.NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear
```

# 缺少对象定位？

将此架构中呈现的模型代码与先前架构中的策略代码进行对比。注意介绍`Layer`对象如何使代码大大简化，同时提高可读性。这种明显的可读性改进是大多数开发人员在实践中更喜欢在 TensorFlow 之上使用面向对象的覆盖的原因之一。

也就是说，在本章中，我们使用原始的 TensorFlow，因为使类似`TensorGraph`这样的类与多个 GPU 一起工作将需要额外的开销。一般来说，原始的 TensorFlow 代码提供了最大的灵活性，但面向对象提供了便利。选择适合手头问题的抽象。

## 在多个 GPU 上训练

我们在每个 GPU 上实例化模型和架构的单独版本。然后，我们使用 CPU 来平均各个 GPU 节点的权重(示例 9-3)。

##### 示例 9-3。此函数训练 Cifar10 模型

```py
def train():
  """Train CIFAR10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * FLAGS.num_gpus)
```

示例 9-4 中的代码执行了基本的多 GPU 训练。注意每个 GPU 为不同批次出队，但通过`tf.get_variable_score().reuse_variables()`实现的权重共享使训练能够正确进行。

##### 示例 9-4。此代码片段实现了多 GPU 训练

```py
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch = batch_queue.dequeue()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope, image_batch, label_batch)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
```

最后，我们通过在示例 9-5 中需要时应用联合训练操作并编写摘要检查点来结束。

##### 示例 9-5。此代码片段将来自各个 GPU 的更新分组并根据需要编写摘要检查点。

```py
    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
      # Save the model checkpoint periodically.

      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
```

## 读者的挑战

您现在拥有实践中训练此模型所需的所有要素。尝试在适合的 GPU 服务器上运行它！您可能需要使用`nvidia-smi`等工具来确保所有 GPU 实际上都在使用。

# 回顾

在本章中，您了解了常用于训练深度架构的各种类型硬件。您还了解了在多个 CPU 或 GPU 上训练深度架构的数据并行和模型并行设计。我们通过一个案例研究来结束本章，介绍如何在 TensorFlow 中实现卷积网络的数据并行训练。

在第十章中，我们将讨论深度学习的未来以及如何有效和道德地运用您在本书中学到的技能。
