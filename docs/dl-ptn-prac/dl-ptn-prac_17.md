# 14 训练和部署流程

本章涵盖

+   在生产环境中为模型训练提供数据

+   为持续重新训练进行调度

+   使用版本控制和部署前后的模型评估

+   在单体和分布式部署中部署模型以处理大规模的按需和批量请求

在上一章中，我们探讨了端到端生产机器学习流程中的数据流程部分。在这里，在本书的最后一章，我们将涵盖端到端流程的最后一部分：训练、部署和服务。

为了用视觉方式提醒你，图 14.1 显示了整个流程，它来自第十三章。我已经圈出了本章我们将要解决的问题的部分系统。

![](img/CH14_F01_Ferlitsch.png)

图 14.1 本章重点介绍的端到端生产流程

你可能会问，究竟什么是流程，为什么我们使用它，无论是用于机器学习生产还是任何由编排管理的程序化生产操作？通常，当工作，如训练或其他由编排处理的操作，有多个按顺序发生的步骤时，你会使用流程：执行步骤 A，执行步骤 B，等等。

将这些步骤放入机器学习生产流程中可以带来多重好处。首先，该流程可以重复用于后续的训练和部署工作。其次，该流程可以被容器化，因此可以作为异步批量作业运行。第三，流程可以在多个计算实例之间分布，其中流程内的不同任务在不同的计算实例上执行，或者同一任务的各个部分可以在不同的计算实例上并行执行。最后，所有与流程执行相关的任务都可以被跟踪，其状态/结果可以保存为历史记录。

本章首先介绍了在生产环境中为训练提供模型的程序，包括顺序和分布式系统，以及使用`tf.data`和 TensorFlow Extended (TFX)的示例实现。然后我们学习如何安排训练和提供计算资源。我们将从介绍可重复使用的流程开始，讨论如何使用元数据将流程集成到生产环境中，以及历史和版本控制用于跟踪和审计。

接下来我们将看到模型是如何在投入生产环境前进行评估的。如今，我们不仅仅是将测试（保留）数据中的指标与模型前一个版本的测试指标进行比较。相反，我们识别出在生产环境中看到的不同子群体和分布，并构建额外的评估数据，这些数据通常被称为*评估切片*。然后，模型在一个模拟的生产环境中进行评估，通常称为*沙盒*，以查看它在响应时间和扩展性方面的表现如何。我包括了一些在沙盒环境中评估候选模型的 TFX 实现示例。

然后我们将转向将模型部署到生产环境并服务于按需和批量预测的过程。您将找到针对当前流量需求的扩展和负载均衡方法。您还将了解服务平台的配置情况。最后，我们讨论了在将模型部署到生产环境后，如何使用 A/B 测试方法进一步评估模型与之前版本的区别，以及如何使用持续评估方法在生产过程中获得洞察后进行后续重新训练。

## 14.1 模型喂养

图 14.2 是训练管道中模型喂养过程的概述。在前端是数据管道，它执行提取和准备训练数据的任务（图中的步骤 1）。由于今天我们在生产环境中处理的数据量非常大，我们将假设数据是从磁盘按需抽取的。因此，模型喂养器充当生成器，并执行以下操作：

+   向数据管道请求示例（步骤 2）

+   从数据管道接收这些示例（步骤 3）

+   将接收到的示例组装成用于训练的批次格式（步骤 4）

模型喂养器将每个批次交给训练方法，该训练方法依次向前馈送每个批次（步骤 5）到模型，计算前馈结束时的损失（步骤 6），并通过反向传播更新权重（步骤 7）。

![图片](img/CH14_F02_Ferlitsch.png)

图 14.2 数据管道与训练方法之间模型喂养过程的交互

模型喂养器位于数据管道和训练函数之间，可能在训练过程中成为 I/O 瓶颈，因此考虑其实施方式，以便喂养器能够以训练方法可以消耗的速度生成批次非常重要。例如，如果模型喂养器作为一个单 CPU 线程运行，而数据管道是一个多 CPU 或 GPU，训练过程是一个多 GPU，那么很可能会导致喂养器无法以接收示例的速度处理它们，或者以训练 GPU 可以消耗的速度生成批次。

由于模型喂养器与训练方法的关系，模型喂养器必须在训练方法消耗当前批次之前或同时，在内存中准备好下一个批次。生产环境中的模型喂养器通常是一个多线程过程，在多个 CPU 核心上运行。在训练过程中向模型提供训练示例有两种方式：顺序和分布式。

顺序训练模型喂养器

图 14.3 展示了一个顺序模型喂养器。我们从一个共享内存区域开始，然后经过以下四个步骤：

+   为模型喂养器保留的共享内存区域，用于在内存中保留两个或更多批次（步骤 1）。

+   在共享内存中实现了一个先进先出（FIFO）队列（步骤 1）。

+   第一个异步过程将准备好的批次放入队列中（步骤 2 和 3）。

+   当训练方法请求时（步骤 3 和 4），第二个异步过程从队列中拉取下一个批次。

通常，顺序方法在计算资源方面是最经济的，当完成训练的时间周期在您的训练时间要求内时，会使用这种方法。其好处是直接的：没有计算开销，就像分布式系统那样，CPU/GPUs 可以全速运行。

![图片](img/CH14_F03_Ferlitsch.png)

图 14.3 顺序训练的模型馈送器

分布式训练的模型馈送器

在分布式训练中，例如在多个 GPU 上，模型馈送器处的 I/O 瓶颈的影响可能会变得更加严重。如图 14.4 所示，它与单实例、非分布式顺序方法不同，因为多个异步提交过程正在从队列中拉取批次，以并行训练多个模型实例。

![图片](img/CH14_F04_Ferlitsch.png)

![图片](img/CH14_F04_Ferlitsch.png)

虽然分布式方法会引入一些计算效率低下，但在您的框架不允许顺序方法完成训练时，会使用它。通常，时间要求是基于业务需求，而未能满足业务需求比计算效率低下有更高的成本。

在分布式训练中，第一个异步过程必须以等于或大于其他多个异步过程拉取批次的速率（步骤 2）将多个批次提交到队列中。每个分布式训练节点都有一个异步过程用于从队列中拉取批次。最后，第三个异步过程协调从队列中拉取批次并等待完成（步骤 3）。在这种分布式训练形式中，每个分布式训练节点都有一个第二个异步过程（步骤 2），其中节点可以是

+   网络连接在一起的独立计算实例

+   同一计算实例上的独立硬件加速器（如 GPU）

+   多核计算实例（如 CPU）上的独立线程

你可能会问，当每个实例只能看到批次的一个子集时，模型是如何进行训练的？这是一个好问题。在这种分布式方法中，我们使用权重的批次平滑。

这样想：每个模型实例从训练数据的子采样分布中学习，我们需要一种方法来合并每个子采样分布中学习到的权重。每个节点在完成一个批次后，将向其他节点发送其权重更新。当接收节点收到权重更新时，它将与自己的批次中的权重更新进行平均——这就是权重批次平滑的原因。

有两种常见的网络方法用于发送权重。一种是在所有节点都连接到的子网上广播权重。另一种是使用环形网络，其中每个节点将其权重更新发送给下一个连接的节点。

这种分布式训练形式有两个后果，无论是广播还是环形。首先，有所有的网络活动。其次，你不知道权重更新的消息何时会出现。它是完全无协调和临时的。因此，权重批平滑固有的低效性会导致与顺序方法相比需要更多的训练轮次。

带有参数服务器的模型提供者

另一种分布式训练版本使用参数服务器。参数服务器通常运行在另一个节点上，通常是 CPU。例如，在谷歌的 TPU pods 中，每组四个 TPU 都有一个基于 CPU 的参数服务器。其目的是克服异步更新批平滑权重的低效性。

在这种分布式训练形式中，权重更新的批平滑是同步发生的。如图 14.5 所示的参数服务器将不同的批次分发给每个训练节点，然后等待每个节点完成其对应批次的消耗（步骤 1），并将损失计算发送回参数服务器（步骤 2）。参数服务器接收到每个训练节点的损失计算后，平均损失并在参数服务器维护的主副本上更新权重，然后将更新的权重发送给每个训练节点（步骤 3）。然后参数服务器向模型提供者发出信号，以分发下一组并行批次（步骤 4）。

这种同步方法的优点是，与上述异步方法相比，它不需要那么多的训练轮次。但缺点是，每个训练节点必须等待参数服务器发出接收下一批次的信号，因此训练节点可能运行在 GPU 或其他计算能力以下。

![图片](img/CH14_F05_Ferlitsch.png)

图 14.5 分布式训练中的参数服务器

有几点需要指出。对于每一轮，每个分布式训练节点都会从另一个节点接收不同的批次。因为训练节点之间的损失可能会有很大差异，以及等待参数更新权重的开销，分布式训练通常使用更大的批次大小。较大的批次可以平滑或减少并行批次之间的差异，以及在训练过程中的 I/O 瓶颈。

### 14.1.1 使用 tf.data.Dataset 进行模型提供

在第十三章中，我们看到了如何使用`tf.data.Dataset`构建数据管道。它可以作为模型提供的机制。本质上，`tf.data.Dataset`的一个实例是一个生成器。它可以集成到顺序和分布式训练中。然而，在分布式提供者中，该实例不充当参数服务器，因为该功能由底层分布式系统执行。

`tf.data.Dataset` 的主要优点包括设置批量大小、对数据进行随机打乱以及并行预取当前批次的下一个批次。

以下代码是使用 `tf.data.Dataset` 在训练期间为模型提供数据的示例，使用了一个虚拟模型——一个没有参数的单层 (`Flatten`) `Sequential` 模型进行训练。为了演示，我们使用了 TF.Keras 内置数据集的 CIFAR-10 数据。

由于本例中的 CIFAR-10 数据将已经在内存中，当通过 `cifar.load_data()` 加载时，我们将创建一个生成器，该生成器将从内存源提供批次。第一步是创建我们的内存数据集的生成器。我们使用 `from_tensor_slices()` 来完成这个任务，它接受一个参数，即内存中的训练示例和相应的标签的元组 `(x_train, y_train)`。注意，此方法不会复制训练数据。相反，它构建了一个指向训练数据源的索引，并使用该索引来打乱、迭代和获取示例：

```
from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test  / 255.0).astype(np.float32)

model = Sequential([ Flatten(input_shape=(32, 32, 3))] )
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))    ❶
dataset = dataset.batch(32).shuffle(1000).repeat().prefetch(2)      ❷

model.fit(dataset, epochs=10, steps_per_epoch=len(x_train//32))     ❸
```

❶ 创建一个 tf.data.Dataset 作为 CIFAR-10 训练数据的模型喂养生成器

❷ 设置模型喂养属性

❸ 在训练时使用生成器作为模型喂养器

现在我们已经有了前面代码示例中的生成器，我们将添加一些属性来将其完整地作为模型喂养器：

+   我们将批量大小设置为 32 (`batch(32)`）。

+   我们在内存中随机打乱每次 1000 个示例（`shuffle(1000)`）。

+   我们反复遍历整个训练数据（`repeat()`）。如果没有 `repeat()`，生成器只会对训练数据进行单次遍历。

+   在提供批量的同时，在喂养器队列中预取最多两个批次（`prefetch(2)`）。

接下来，我们可以将生成器作为训练输入源传递给 `fit(dataset, epochs=10, steps_per_epoch=len(x_train//32))` 命令进行训练。此命令将生成器视为迭代器，并且对于每次交互，生成器将执行模型喂养任务。

由于我们使用生成器进行模型喂养，并且 `repeat()` 将导致生成器无限迭代，因此 `fit()` 方法不知道它何时已经消耗了一个 epoch 的全部训练数据。因此，我们需要告诉 `fit()` 方法一个 epoch 由多少个批次组成，我们使用关键字参数 `steps_per_epoch` 来设置。

动态更新批量大小

在第十章中，我们讨论了批量大小与学习率成反比的关系。在训练期间，这种反比关系意味着传统的模型喂养技术将按比例增加批量以适应学习率的降低。虽然 TF.Keras 有一个内置的 `LearningRateScheduler` 回调来动态更新学习率，但它目前还没有同样的能力来更新批量大小。相反，我将向您展示在降低学习率的同时动态更新批量大小的 DIY 版本。

我将在描述实现它的代码时解释 DIY 过程。在这种情况下，我们添加一个外层训练循环以动态更新批量大小。回想一下，在`fit()`方法中，批量大小被指定为一个参数。因此，要更新批量大小，我们将划分 epoch 并多次调用`fit()`。在循环内部，我们将对模型进行指定数量的 epoch 的训练。至于循环，每次迭代时，我们将更新学习率和批量大小，并在循环中设置要训练的 epoch 数量。在`for`循环中，我们使用一个元组的列表，每个元组将指定学习率（`lr`）、批量大小（`bs`）和 epoch 数量（`epochs`）；例如，`(0.01, 32, 10)`。

在循环中重置`epochs`的数量很简单，因为我们可以将其指定为`fit()`方法的参数。对于学习率，我们通过（重新）编译模型并在指定优化器参数时重置学习率来重置它——`Adam(lr=lr)`。在训练过程中重新编译模型是可以的，因为它不会影响模型的权重。换句话说，重新编译不会撤销之前的训练。

重置`tf.data.Dataset`的批量大小并不简单，因为一旦设置，就无法重置。相反，我们将在每次循环迭代中为训练数据创建一个新的生成器，其中我们将使用`batch()`方法指定当前的批量大小。

```
from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test  / 255.0).astype(np.float32)

model = Sequential([ Conv2D(16, (3, 3), activation='relu', 
                            input_shape=(32, 32, 3)),
                     Conv2D(32, (3, 3), strides=(2, 2), activation='relu'),
                     MaxPooling2D((2, 2), strides=2),
                     Flatten(),
                     Dense(10, activation='softmax')
                   ])

for lr, bs, epochs in [ (0.01, 32, 10), (0.005, 64, 10), (0.0025, 128, 10) ]:  ❶
    print("hyperparams: lr", lr, "bs", bs, "epochs", epochs)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))           ❷
    dataset = dataset.shuffle(1000).repeat().batch(bs).prefetch(2)             ❷

    model.compile(loss='sparse_categorical_crossentropy', 
     optimizer=Adam(lr=lr), 
                  metrics=['acc'])                                             ❸
    model.fit(dataset, epochs=epochs, steps_per_epoch=200, verbose=1)          ❹
```

❶ 外层循环用于在训练期间动态重置超参数

❷ 创建一个新的生成器以重置批量大小

❸ 重新编译模型以重置学习率

❹ 使用重置的 epoch 数量训练模型

让我们看看运行我们的 DIY 版本动态重置训练中超参数的简略输出。您可以看到在外层循环的第一次迭代中，第 10 个 epoch 的训练准确率为 51%。在第二次迭代中，学习率减半，批量大小加倍，第 10 个 epoch 的训练准确率为 58%，在第三次迭代中达到 61%。如您从输出中观察到的，我们在三个迭代中能够保持损失持续减少和准确率增加，因为我们逐渐缩小到损失空间。

```
hyperparams: lr 0.01 bs 32 epochs 10
Epoch 1/10
200/200 [==============================] - 1s 3ms/step - loss: 1.9392 - acc: 0.2973
Epoch 2/10
200/200 [==============================] - 1s 3ms/step - loss: 1.6730 - acc: 0.4130
...
Epoch 10/10
200/200 [==============================] - 1s 3ms/step - loss: 1.3809 - acc: 0.5170

hyperparams: lr 0.005 bs 64 epochs 10
Epoch 1/10
200/200 [==============================] - 1s 3ms/step - loss: 1.2248 - acc: 0.5704
Epoch 2/10
200/200 [==============================] - 1s 3ms/step - loss: 1.2740 - acc: 0.5510
...
Epoch 10/10
200/200 [==============================] - 1s 3ms/step - loss: 1.1876 - acc: 0.5853

hyperparams: lr 0.0025 bs 128 epochs 10
Epoch 1/10
200/200 [==============================] - 1s 4ms/step - loss: 1.1186 - acc: 0.6063
Epoch 2/10
200/200 [==============================] - 1s 3ms/step - loss: 1.1434 - acc: 0.5997
...
Epoch 10/10
200/200 [==============================] - 1s 3ms/step - loss: 1.1156 - acc: 0.6129
```

### 14.1.2 使用 tf.Strategy 进行分布式喂养

TensorFlow 模块`tf.distribute.Strategy`提供了一个方便且封装的接口，为你完成所有工作，用于在同一个计算实例上的多个 GPU 之间或多个 TPU 之间进行分布式训练。它实现了本章前面描述的同步参数服务器。此 TensorFlow 模块针对 TensorFlow 模型的分布式训练以及并行 Google TPUs 上的分布式训练进行了优化。

当在单个计算实例上使用多个 GPU 进行训练时，你使用 `tf .distribute.MirrorStrategy`，而当在 TPU 上进行训练时，你使用 `tf.distribute .TPUStrategy`。在本章中，除了指出你将使用 `tf.distribute.experimental.ParameterServerStrategy` 以实现跨网络的异步参数服务器之外，我们不会涵盖跨机器的分布式训练。跨多个机器的分布式训练设置相对复杂，可能需要单独的一章。如果你在构建 TensorFlow 模型，并且训练过程中需要大量的并行处理以满足业务目标，我建议使用这种方法，并学习 TensorFlow 文档。

这是我们在单机上设置分布式训练运行的步骤，该机器具有多个 CPU 或 GPU：

1.  实例化一个分布策略。

1.  在分布策略的作用域内

    +   创建模型。

    +   编译模型。

1.  训练模型。

这些步骤可能看起来有些不合常理，因为我们是在构建和编译模型时设置分布策略，而不是在训练时。在 TensorFlow 中，模型构建需要知道它将使用分布式训练策略进行训练。截至本文撰写时，TensorFlow 团队最近发布了一个新的实验版本，其中分布策略可以在不编译模型的情况下设置。

以下是实现前面三个步骤和两个子步骤的代码，这些步骤在此处描述：

1.  我们定义函数 `create_model()` 来创建用于训练的模型实例。

1.  我们实例化分布策略：`strategy = tf.distribute.MirrorStrategy()`.

1.  我们设置分布上下文：`with strategy.scope()`.

1.  在分布上下文中，我们创建模型的实例：`model = create_model()`。然后我们编译它：`model.compile()`。

1.  最后，我们训练模型。

```
def create_model():                                               ❶
    model = Sequential([ Conv2D(16, (3, 3), activation='relu', 
                                input_shape=(32, 32, 3)),
                         Conv2D(32, (3, 3), strides=(2, 2), 
                                activation='relu'),
                         MaxPooling2D((2, 2), strides=2),
                         Flatten(),
                         Dense(10, activation='softmax')
                       ])
    return model

strategy = tf.distribute.MirroredStrategy()                       ❷

with strategy.scope():  bbbb                                      ❸
    model = create_model()                                        ❸
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')    

model.fit(dataset, epochs=10, steps_per_epoch=200)                ❹
```

❶ 创建模型实例的函数

❷ 实例化分布策略

❸ 在分布策略的作用域内创建和编译模型

❹ 训练模型

你可能会问，我能否使用已经构建好的模型？答案是：不可以；你必须在分布策略的作用域内构建模型。例如，以下代码将导致错误，提示模型未在分布策略的作用域内构建：

```
model = create_model()          ❶
with strategy.scope():
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

❶ 模型未在分布策略的作用域内构建

再次，你可能还会问：我已经有一个预先构建或预训练的模型，它不是为分布策略构建的；我还能进行分布式训练吗？这里的答案是：可以。如果你有一个保存到磁盘的 TF.Keras 模型，当你使用 `load_model()` 将其加载回内存时，它将隐式地构建模型。以下是从预训练模型设置分布策略的示例实现：

```
with strategy.scope():
    model = tf.keras.models.load_model('my_model')                          ❶
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

❶ 从磁盘加载时模型会隐式重建

同样，当从模型存储库加载预构建模型时，存在隐式的加载和相应的隐式构建。以下代码序列是加载`tf.keras.applications`内置模型存储库中模型的示例，其中模型被隐式重建：

```
with strategy.scope():
   model = tf.keras.applications.ResNet50()                                ❶
   model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

❶ 从存储库加载时模型被隐式重建

默认情况下，镜像策略将使用计算实例上的所有 GPU。您可以使用`num_replicas_in_sync`属性获取将要使用的 GPU 或 CPU 核心数。您还可以明确设置要使用的 GPU 或核心。在以下代码示例中，我们将分布策略设置为使用两个 GPU：

```
strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
print("GPUs:", strategy.num_replicas_in_sync)
```

以下代码示例生成了以下输出：

```
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1')
GPUs: 2
```

### 14.1.3 使用 TFX 进行模型喂养

第十三章涵盖了 TFX 端到端生产管道的数据管道部分。本节涵盖了训练管道组件的相应 TFX 模型喂养方面，作为另一种实现。图 14.6 描述了训练管道组件及其与数据管道的关系。训练管道由以下组件组成：

+   *训练器*—训练模型

+   *调优器*—调整超参数（例如，学习率）

+   *评估器*—评估模型的客观指标，例如准确性，并将结果与基线（例如，上一个版本）进行比较

+   *基础设施评估器*—在部署前在沙盒服务环境中测试模型

![图片](img/CH14_F06_Ferlitsch.png)

图 14.6 TFX 训练管道由调优器、训练器、评估器和基础设施评估器组件组成。

编排

让我们回顾一下 TFX 和管道的一般好处。如果我们单独执行训练/部署模型中的每个步骤，我们将其称为*任务感知架构*。每个组件都了解自己，但不知道连接的组件或之前执行的历史。

TFX 实现了*编排*。在编排中，一个管理接口监督每个组件的执行，记住过去组件的执行情况，并维护历史记录。如第十三章所述，每个组件的输出是工件；这些是执行的结果和历史。在编排中，这些工件或对其的引用被存储为元数据。对于 TFX，元数据以关系格式存储，因此可以通过 SQL 数据库进行存储和访问。

让我们更深入地探讨编排的好处，然后我们将介绍 TFX 中模型喂养的工作方式。通过编排，如图 14.7 所示，我们可以做以下事情：

+   在另一个组件（或多个组件）完成后执行组件的调度。例如，我们可以在从训练数据生成特征模式后调度数据转换的执行。

+   当组件的执行相互不依赖时，并行调度组件的执行。例如，在数据转换完成后，我们可以并行调度超参数调整和训练。

+   如果组件的执行没有变化（缓存），则重用组件先前执行中的工件。例如，如果训练数据没有变化，转换组件的缓存工件（即转换图）可以在不重新执行的情况下重用。

+   为每个组件提供不同的计算引擎实例。例如，数据管道组件可能配置在 CPU 计算实例上，而训练组件配置在 GPU 计算实例上。

+   如果任务支持分布，例如调整和训练，则可以将任务分布到多个计算实例上。

+   将组件的工件与组件先前执行的工件进行比较。例如，评估组件可以将模型的指标（例如，准确率）与先前训练的模型版本进行比较。

+   通过能够向前和向后移动通过生成的工件来调试和审计管道的执行。

![图片](img/CH14_F07_Ferlitsch.png)

图 14.7 编排器摄取表示为图的管道，并配置实例和调度任务。

训练组件

`Trainer`组件支持训练 TensorFlow 估计器、TF.Keras 模型和其他自定义训练循环。由于 TensorFlow 2.*x*建议逐步淘汰估计器，我们将仅关注配置 TF.Keras 模型的训练组件，并向其提供数据。训练组件需要以下最小参数：

+   `module_file`—这是用于自定义训练模型的 Python 脚本。它必须包含一个`run_fn()`函数作为训练的入口点。

+   `examples`—用于训练模型的示例，这些示例来自`ExampleGen`组件的输出，`example_gen.outputs['examples']`。

+   `schema`—数据集模式，它来自`SchemaGen`组件的输出，`schema_gen['schema']`。

+   `custom_executor_spec`—自定义训练的执行器，它将在`module_file`中调用`run_fn()`函数。

```
from tfx.components import Trainer
from tfx.components.base import executor_spec                              ❶
from tfx.components.trainer import GenericExecutor                         ❶

trainer = Trainer(
    module_file=module_file,                                               ❷
    examples=example_gen.outputs['examples'],                              ❸
    schema=schema_gen.outputs['schema'],                                   ❹
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor)  ❺
)
```

❶ 自定义训练的导入

❷ 自定义训练的 Python 脚本

❸ 在训练过程中向模型提供数据的训练数据源

❹ 从数据集中推断出的模式

❺ 自定义训练的自定义执行器

如果训练数据需要由`Transform`组件进行预处理，我们需要设置以下两个参数：

+   `transformed_examples`—设置为`Transform`组件的输出，`transform.outputs['transformed_examples']`。

+   `transform_graph`—由`Transform`组件产生的静态转换图，`transform.outputs['transformed_graph']`。

```
trainer = Trainer(
    module_file=module_file,
    transformed_examples=transform.outputs['transformed_examples'],    ❶
    transform_graph=transform.outputs['transform_graph'],              ❶
    schema=schema_gen.outputs['schema'],
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor)
)
```

❶ 训练数据从`Transform`组件馈送到静态转换图。

通常，我们希望将其他超参数传递到训练模块中。这些可以通过将 `train_args` 和 `eval_args` 作为附加参数传递给 `Trainer` 组件。这些参数被设置为键/值对列表，并转换为 Google 的 protobuf 格式。以下代码传递了训练和评估的步骤数：

```
from tfx.proto import trainer_pb2                                          ❶
trainer = Trainer(
    module_file=module_file,
    transformed_examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],   
    schema=schema_gen.outputs['schema'],
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    train_args=trainer_pb2.TrainArgs(num_steps=10000),                     ❷
    eval_args=trainer_pb2.EvalArgs(num_steps=5000)                         ❷
)
```

❶ 导入用于传递超参数的 TFX protobuf 格式

❷ 将超参数作为 protobuf 消息传递到 Trainer 组件中

现在，让我们看看自定义 Python 脚本中 `run_fn()` 函数的基本要求。`run_fn()` 的参数由传递给 `Trainer` 组件的参数构建，并且作为属性访问。在以下示例实现中，我们执行以下操作：

+   提取训练的总步骤数：`training_args.train_steps`。

+   提取每个 epoch 后的验证步骤数：`training_args .eval_steps`。

+   获取训练和评估数据的 TFRecord 文件路径：`training_args .train_files`。注意，`ExampleGen` 不会提供内存中的 `tf.Examples`，而是提供包含 `tf.Examples` 的磁盘上的 TFRecords。

+   获取转换图，`training_args.transform_output`，并构建转换执行函数，`tft.TFTransformOutput()`。

+   调用内部函数 `_input_fn()` 来创建训练和验证数据集的迭代器。

+   使用内部函数 `_build_model()` 构建或加载 TF.Keras 模型。

+   使用 `fit()` 方法训练模型。

+   获取存储训练模型的托管目录，`training_args.output`，该目录作为可选参数 `output` 传递给 `Trainer` 组件。

+   将训练好的模型保存到指定的服务输出位置，`model.save (serving_dir)`。

```
from tfx.components.trainer.executor import TrainerFnArgs
import tensorflow_transform as tft

BATCH_SIZE = 64                                                                  ❶
STEPS_PER_EPOCH = 250                                                            ❶

def run_fn(training_args: TrainerFnArgs):
    train_steps = training_args.train_steps                                      ❷
    eval_steps  = training_args.eval_steps                                       ❷

    train_files = training_args.train_files                                      ❸
    eval_files  = training_args.eval_files                                       ❸

    tf_transform_output = tft.TFTransformOutput(training_args.transform_output)  ❹
    train_dataset = _input_fn(train_files, tf_transform_output, BATCH_SIZE)      ❹
    eval_dataset  = _input_fn(eval_files, tf_transform_output, BATCH_SIZE)       ❹

    model = _build_model()                                                       ❺

    epochs = train_steps // STEPS_PER_EPOCH                                      ❻

    model.fit(train_dataset, epochs=epochs, validation_data=eval_dataset,
              validation_steps=eval_steps)                                       ❼

    serving_dir = training_args.output                                           ❽
    model.save(serving_dir)                                                      ❽
```

❶ 将超参数设置为常量

❷ 将训练/验证步骤作为参数传递给 Trainer 组件

❸ 将训练/验证数据作为参数传递给 Trainer 组件

❹ 为训练和验证数据创建数据集迭代器

❺ 构建或加载用于训练的模型

❻ 计算 epoch 数量

❼ 训练模型

❽ 将模型以 SavedModel 格式保存到指定的服务目录

在构建自定义 Python 训练脚本时，有很多细微之处和多种方向可以选择。有关更多详细信息和建议，我们建议查看 TFX 的 `Trainer` 组件指南（[www.tensorflow.org/tfx/guide/trainer](https://www.tensorflow.org/tfx/guide/trainer)）。

Tuner 组件

`Tuner` 组件是训练流程中的可选任务。您可以在自定义 Python 训练脚本中硬编码训练的超参数，或者使用 Tuner 来找到超参数的最佳值。

`Tuner` 的参数与 `Trainer` 非常相似。也就是说，`Tuner` 将进行短训练运行以找到最佳超参数。但与返回训练模型的 `Trainer` 不同，`Tuner` 的输出是调优的超参数值。通常不同的两个参数是 `train_args` 和 `eval_args`。由于这些将是较短的训练运行，因此调优器的步骤数通常是完整训练的 20% 或更少。

另一个要求是自定义的 Python 训练脚本 `module_file` 包含函数入口 `tuner_fn()`。典型的做法是使用一个包含 `run_fn()` 和 `tuner_fn()` 函数的单个 Python 训练脚本。

```
tuner = Tuner(
    module_file=module_file,
    transformed_examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=2000),    ❶
    eval_args=trainer_pb2.EvalArgs(num_steps=1000)       ❶
)
```

❶ 调优时较短的训练运行步骤数

接下来，我们将查看 `tuner_fn()` 的一个示例实现。我们将使用 KerasTuner 进行超参数调优，但你可以使用与你的模型框架兼容的任何调优器。我们之前在第十章中介绍了使用 KerasTuner。它是一个独立的包，因此你需要按照以下方式安装它：

```
pip install keras-tuner
```

与 `Trainer` 组件类似，将 `Tuner` 组件的参数和默认值作为 `tuner_args` 参数的属性传递给 `tuner_fn()`。请注意，函数的起始部分与 `run_fn()` 相同，但在到达训练步骤时有所不同。我们不是调用 `fit()` 方法并保存训练好的模型，而是这样做：

1.  实例化一个 KerasTuner：

    +   我们使用 `build_model()` 作为超参数模型参数。

    +   调用内部函数 `_get_hyperparameters()` 来指定超参数搜索空间。

    +   将最大试验次数设置为 6。

    +   设置选择最佳超参数值的指标。在这种情况下，是验证准确率。

1.  将调优器和剩余的训练参数传递给 `TunerFnResult()` 实例，该实例将执行调优。

1.  返回调优试验的结果。

```
import kerastuner

def tuner_fn(tuner_args: FnArgs) -> TunerFnResult:                          ❶
    train_steps = tuner_args.train_steps
    eval_steps  = tuner_args.eval_steps 

    train_files = tuner_args.train_files
    eval_files  = tuner_args.eval_files 

    tf_transform_output = tft.TFTransformOutput(tuner_args.transform_output)
    train_dataset = _input_fn(train_files, tf_transform_output, BATCH_SIZE)
    eval_dataset  = _input_fn(eval_files, tf_transform_output, BATCH_SIZE)

    tuner = kerastuner.RandomSearch(_build_model(),                         ❷
                                    max_trails=6, 
                                    hyperparameters=_get_hyperparameters(), ❸
                                    objective='val_accuracy'
                                   )

    result = TunerFnResult(tuner=tuner,                                     ❹
                           fit_kwargs={                                     ❺
                               'x': train_dataset,                          ❺
                               'validation_data': eval_dataset,             ❺
                               'steps_per_epoch': train_steps,              ❺
                               'validation_steps': eval_steps               ❺
                           })
    return result
```

❶ 超参数调优的入口点函数

❷ 为随机搜索实例化 KerasTuner

❸ 获取超参数搜索空间

❹ 使用指定的调优实例实例化和执行调优试验

❺ 调优期间短训练运行的训练参数

现在，让我们看看 `Tuner` 和 `Trainer` 组件是如何串联在一起形成一个可执行管道的。在下面的示例实现中，我们对 `Trainer` 组件的实例化进行了一次修改，添加了可选参数 `hyperparameters` 并将输入连接到 `Tuner` 组件的输出。现在，当我们使用 `context.run()` 执行 `Trainer` 实例时，协调器将看到对 `Tuner` 的依赖，并将它的执行安排在 `Trainer` 组件进行完整训练之前：

```
tuner = Tuner(
    module_file=module_file,
    transformed_examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=2000),
    eval_args=trainer_pb2.EvalArgs(num_steps=1000)   
)
trainer = Trainer(
    module_file=module_file,
    transformed_examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],   
    schema=schema_gen.outputs['schema'],
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    hyperparameters=tuner.outputs['best_hyperparameters'],     ❶
    train_args=trainer_pb2.TrainArgs(num_steps=10000),   
    eval_args=trainer_pb2.EvalArgs(num_steps=5000)   
)

context.run(trainer)                                           ❷
```

❶ 从 Tuner 组件获取调优的超参数

❷ 执行 Tuner/Trainer 管道

与训练器一样，Python 超参数调整脚本可以自定义。请参阅 TFX 的指南了解`Tuner`组件([www.tensorflow.org/tfx/guide/tuner](https://www.tensorflow.org/tfx/guide/tuner))。

## 14.2 训练调度器

在研究或开发环境中，训练管道通常是手动启动的；管道中的每个任务都是手动启动的，这样如果需要，每个任务都可以被观察和调试。在生产方面，它们是自动化的；自动化使得执行管道更加高效，劳动强度更低，并且更具可扩展性。在本节中，我们将了解生产环境中调度的工作方式，因为大量的训练作业可以排队进行训练，并且/或者模型会持续重新训练。

生产环境的需求与研究和开发不同，如下所示：

+   在生产环境中，计算和网络 I/O 的数量可能会有很大变化，因为在生产环境中，大量模型可能会并行持续重新训练。

+   训练作业可能有不同的优先级，因为它们必须在部署的交付时间表内完成。

+   训练工作可能需要按需资源，例如可能按使用情况配置的特殊硬件，如云实例。

+   训练作业的长度可能会因重启和超参数调整而变化。

图 14.8 描述了一个适用于具有上述需求的大规模生产环境的端到端生产管道作业调度器。我们使用一个概念视图，即生产环境中的作业调度尚未得到开源机器学习框架的充分支持，但由付费机器学习服务（如云提供商）以不同程度的支持。

![](img/CH14_F08_Ferlitsch.png)

图 14.8 大规模生产环境中管道作业调度

让我们深入探讨图 14.8 中所示的生产环境的一些假设，这在企业环境中是典型的：

+   没有定制工作。尽管在模型开发过程中可能存在定制工作，但一旦模型进入生产阶段，它将使用预定义的、受版本控制的管道进行训练和部署。

+   管道有定义好的依赖关系。例如，一个用于图像输入模型的训练管道将有一个只能与特定于图像数据的数据管道结合的依赖关系。

+   管道可能有可配置的属性。例如，数据管道的源输入和输出形状是可配置的。

+   如果对管道进行了升级，它将成为下一个版本。保留上一个版本和执行历史。

+   一个作业请求指定了管道需求。这些需求可以通过引用特定的管道和版本来指定，或者通过属性来指定，调度器将确定最佳匹配的管道。需求还可以指定可配置的属性，例如数据管道的输出形状。

+   作业请求指定了执行需求。例如，如果使用类似 AutoML 的服务，它可能指定计算时间内的最大训练预算。在另一个例子中，它可能指定提前停止条件，或者重新启动训练作业的条件。

+   作业请求指定了计算需求。例如，如果进行分布式训练，它可能指定计算实例的数量和类型。需求通常包括操作系统和软件要求。

+   作业请求指定了优先级需求。通常，这要么是按需，要么是批量。按需作业通常在计算资源可用于配置时调度。批量请求通常在满足一定条件后才会延迟。例如，它可能指定执行的时间窗口，或者等待计算实例最经济的时候。

+   按需工作可以设置一个可选的优先级条件。如果没有设置，通常以先进先出（FIFO）的方式调度。指定了优先级的作业可能会改变其在 FIFO 调度队列中的位置。例如，一个估计时长为*X*并在时间*Y*完成的作业可能会被提升到队列中以满足需求。

+   作业从队列中调度后，其管道组装、执行和计算需求将转交给调度器。

### 14.2.1 管道版本控制

在生产环境中，管道是受版本控制的。除了版本控制之外，每个版本的管道都将包含用于跟踪的元数据。这些元数据可能包括以下内容：

+   管道创建和最后更新的时间

+   管道上次使用的时间和使用的作业

+   虚拟机（VM）依赖项

+   平均执行时间

+   故障率

图 14.9 描述了一个数据管道和相应可重用组件的存储库，这些组件都处于版本控制之下。在这个例子中，我们有两个可重用组件的存储库：

+   *磁盘上的图像迭代器*—用于构建特定于数据集存储格式的数据集迭代器的组件

+   *内存中转换*—用于数据预处理和不变性转换的组件

![](img/CH14_F09_Ferlitsch.png)

图 14.9 展示了使用不同内存中可重用组件的相同数据管道的不同版本

在这个例子中，我们展示了一个具有两个版本的单个数据管道；v2 配置为使用标准化代替 v1 中的缩放。此外，v2 的历史记录比 v1 的历史记录有更好的训练结果。数据管道由磁盘上的图像迭代器和内存中可重用转换组件以及针对管道的特定代码组成。版本 v1 使用缩放进行数据归一化。假设后来我们发现标准化在训练图像数据集时给出了更好的结果，例如验证准确率。因此，我们将缩放替换为标准化，从而创建了管道的新版本 v2。

![](img/CH14_F10_Ferlitsch.png)

图 14.10 使用版本清单来识别特定版本管道中可重用组件的版本

现在我们来看一个不太明显的版本控制系统（图 14.10）。我们将继续使用现有的数据管道示例，但这次磁盘上的 TFRecords 迭代器已更新到 v2，v2 比 v1 有 5%的性能提升。

由于这是数据管道的可配置属性，为什么我们要更新相应数据管道的版本号，而该数据管道本身并没有改变？如果我们想重现或审计使用该管道的训练作业，我们需要知道作业完成时的可重用组件的版本号。在我们的例子中，我们这样做如下：

+   为可重用组件和相应的版本号创建清单。

+   更新数据管道以包含更新的清单。

+   更新数据管道上的版本号。

### 14.2.2 元数据

现在我们来讨论如何使用管道和其他资源（如数据集）存储元数据，以及它如何影响训练作业的组装、执行和调度。那么什么是元数据，它与工件和历史记录有何不同？*历史记录*是关于保留管道*执行*的信息，而*元数据*是关于保留管道*状态*的信息。*工件*是历史记录和元数据的组合。

参考我们的示例数据管道，假设我们正在使用版本 v3，但我们使用一个新的数据集资源。当时，我们在数据集资源上只有数据集中的示例数量这个统计数据。我们不知道的是示例的平均值和标准差。因此，当 v3 数据管道与新数据集组装时，底层的管道管理会查询数据集的平均值和标准差的状态。由于它们是未知的，管道管理会在标准化组件之前添加一个组件来计算标准化组件所需的值。图 14.11 的上半部分描述了当平均值和标准差的状态未知时管道的构建。

现在，假设我们在没有任何更改数据集的情况下再次运行此管道。我们会重新计算平均值和标准差吗？不，当管道管理查询数据集并发现值已知时，它将添加一个组件来使用缓存的值。图 14.11 的下半部分描述了当平均值和标准差的状态未知时管道的构建。

![](img/CH14_F11_Ferlitsch.png)

图 14.11 管理管道选择根据数据集的状态信息计算平均值/标准差或使用缓存值

现在，让我们通过添加一些新示例来更新数据集，我们将这个数据集版本称为 v2。由于示例已被更新，这使之前的均值和标准差计算无效，因此更新将此统计信息恢复为“未知”。

由于统计信息恢复为未知状态，下一次使用更新后的 v2 数据集的 v3 数据管道版本时，管道管理将再次添加计算均值和标准差的组件。图 14.12 展示了这一数据管道的重构过程。

![图片](img/CH14_F12_Ferlitsch.png)

图 14.12 管道管理在数据集中添加新示例后，将重新计算均值和标准差添加到管道中。

### 14.2.3 历史

*历史*指的是管道实例执行的输出结果。例如，考虑一个在模型完整训练之前进行超参数搜索的训练管道。超参数搜索空间和搜索中选定的值成为管道执行历史的一部分。

图 14.13 展示了管道实例的执行过程，它包括以下内容：

+   管道组件的版本，v1

+   训练数据和对应状态，统计信息

+   训练模型资源及其对应状态，指标

+   执行实例的版本，v1.1，以及对应的历史，超参数

![图片](img/CH14_F13_Ferlitsch.png)

图 14.13 展示了管道实例的执行历史，其中工件包括状态、历史和资源

现在，我们如何将历史数据整合到相同管道的后续执行实例中？图 14.14 展示了与图 14.13 相同的管道配置，但使用了新的数据集版本 v2。v2 数据集与 v1 的不同之处在于包含少量新的示例；这些新示例的数量远小于示例总数。

![图片](img/CH14_F14_Ferlitsch.png)

图 14.14 管道管理在新增示例数量显著较少时重用之前执行历史中选定的超参数

在管道实例的组装过程中，管道管理可以使用之前执行实例的历史数据。在我们的示例中，新增示例的数量足够低，以至于管道管理可以重用之前执行历史中选定的超参数值，从而消除了重新执行超参数搜索的开销。

图 14.15 展示了我们示例中管道管理的另一种方法。在这种替代方案中，管道管理继续在第二个执行实例中配置执行超参数搜索的任务，但有所不同：

+   假设第二次执行实例的新超参数值将位于第一次执行中选定的值附近

+   将搜索空间缩小到第一个执行实例历史中选定参数周围的小ε范围内

到目前为止，我们已经涵盖了端到端生产管道和调度的数据与训练部分。下一节将介绍在将模型部署到生产环境之前如何评估模型。

![图片](img/CH14_F15_Ferlitsch.png)

图 14.15 当新示例数量显著较少时，管道管理将超参数搜索空间缩小到先前执行历史附近的区域。

## 14.3 模型评估

在生产环境中，模型评估的目的是在部署到生产之前确定其相对于基线的性能。如果是第一次部署模型，基线由生产团队指定，通常被称为*机器学习操作*。否则，基线是当前部署的生产模型，通常被称为*受祝福模型*。与基线进行比较的模型称为*候选模型*。

### 14.3.1 候选模型与受祝福模型比较

之前，我们在实验和开发的环境中介绍了模型评估，其中的评估基于测试（保留）数据集的客观指标。然而，在生产中，评估基于一系列更广泛的因素，例如资源消耗、扩展以及在生产中受祝福模型看到的样本集（这些不是测试数据集的一部分）。

例如，假设我们想要评估生产模型的下一个候选版本。我们想要进行苹果对苹果的比较。为此，我们将评估受祝福模型和候选模型对相同的测试数据，确保测试数据具有与用于训练的数据集相同的采样分布。我们还想用相同的生产请求子集测试这两个模型；这些请求应该具有与受祝福模型在生产中实际看到的相同的采样分布。为了使候选模型取代受祝福模型并成为下一个部署的版本，测试和生产样本的指标值（例如，分类的准确率）必须在两者上都更好。在图 14.16 中，你可以看到我们如何设置这个测试。

所以，你可能会问，为什么我们不直接用与受祝福模型相同的测试数据来评估候选模型呢？嗯，现实情况是，一旦模型部署，它在预测的例子中的分布很可能与训练时的分布不同。我们还想评估模型在部署后可能看到的分布。接下来，我们将介绍从训练和生成环境中分布变化的两种类型：服务偏差和数据漂移。

![图片](img/CH14_F16_Ferlitsch.png)

图 14.16 候选模型的评估包括训练和生产数据的数据分布。

服务偏差

现在我们深入探讨一下为什么我们要将候选模型与生产数据进行评估。在第十二章中，我们讨论了你的训练可能是一个子群体的抽样分布，而不是整个群体。首先，我们假设部署模型的预测请求来自相同的子群体。例如，假设模型被训练来识别 10 种水果，并且所有部署模型的预测请求都是这 10 种水果——相同的子群体。

但现在假设我们没有相同的抽样分布。生产模型看到的每个类的频率与训练数据不同。例如，假设训练数据完美平衡，每种水果有 10%的示例，测试数据上的整体分类准确率为 97%。但对于 10 个类别中的一个（比如桃子），准确率为 75%。现在假设 40%的预测请求是针对部署的幸运模型的桃子。在这种情况下，子群体保持不变，但训练数据和生产请求之间的抽样分布发生了变化。这被称为*服务偏差*。

那么，我们如何做到这一点呢？首先，我们必须配置一个系统来捕获预测的随机选择及其对应的结果。假设你想要收集所有预测的 5%。你可以创建一个介于 1 到 20 之间的整数均匀随机分布，并为每个预测从分布中抽取一个值。如果抽取的值是 1，你保存预测及其对应的结果。在采样周期结束后，你手动检查保存的预测/结果，并确定每个预测的正确真实值。然后，你将手动标记的真实值与预测结果进行比较，以确定部署生产模型上的指标。

然后你使用相同生产样本的手动标记版本评估候选模型。

数据漂移

现在假设生产抽样分布不是来自与训练数据相同的子群体，而是不同的子群体。继续我们的 10 种水果的例子，并假设训练数据包括新鲜成熟的果实。但我们的模型部署在果园的拖拉机上，那里的果实可以处于各种成熟阶段：绿色、成熟、腐烂。这些绿色和腐烂的果实是训练数据中的不同子群体。在这种情况下，抽样分布保持不变，但训练数据和生产请求之间的子群体发生了变化。这被称为*数据漂移*。

在这种情况下，我们想要将生产样本分离并划分为两部分：一部分与训练数据的子群体相同（例如，成熟的果实），另一部分与训练数据的子群体不同（例如，绿色和腐烂的果实）。然后，我们对生产样本的每个部分进行单独评估。

总的来说，测试、服务偏差和数据漂移样本分别被称为*评估切片*，如图 14.17 所示。一个组织可能对其生产有自定义的评估切片定义，而这一组测试、服务偏差和数据漂移是一般规则。

![](img/CH14_F17_Ferlitsch.png)

图 14.17 生产中的评估切片，包括来自训练数据的样本、服务偏差和生产请求的数据漂移

扩展

现在假设我们的候选模型在所有评估切片中至少在一个指标上与受祝福模型相等或更好。我们现在可以版本化候选模型并将其作为受祝福模型的替代品部署吗？还不行。我们还没有了解候选模型与受祝福模型相比在计算性能上的表现。也许候选模型需要更多的内存，或者也许候选模型的延迟更长。

在我们做出最终决定之前，我们应该将模型部署到一个沙盒环境中，该环境复制了已部署的受祝福模型的计算环境。我们还想确保，在评估期间，生产环境中的预测请求实时复制，并发送到生产环境和沙盒环境。我们的目标是收集沙盒模型的利用率指标，例如消耗的计算和内存资源以及预测结果的延迟时间。你可以在图 14.18 中看到这个沙盒设置。

![](img/CH14_F18_Ferlitsch.png)

图 14.18 在部署之前，最后一步是在沙盒环境中运行候选模型，使用与受祝福模型相同的预测请求。

你可能会问，为什么我们需要在沙盒环境中测试候选模型。我们想知道新模型在服务性能上是否继续满足业务需求。也许候选模型在矩阵乘法操作上有显著增加，以至于返回预测的延迟时间更长，不符合业务需求。也许内存占用增加，使得模型在高服务负载下开始进行内存到页面缓存。

现在让我们考虑一些场景。首先，你可能会说，即使内存占用或计算扩展更大，或者延迟更长，我们也可以简单地添加更多的计算和/或内存资源。但是，有许多原因你可能无法仅仅添加更多资源。如果模型部署在受限环境中，例如移动设备，例如，你无法更改内存或计算设备。或者也许环境拥有极好的资源，但不能进一步修改，例如已经发射到太空的航天器。或者也许模型被一个学区使用，该学区有固定的计算成本预算。

无论原因如何，都必须进行最终的缩放评估，以确定其使用的资源。对于受限制的环境，例如手机或物联网设备，您希望了解候选模型是否会继续满足已部署模型的运行要求。如果您的工作环境不受限制，例如自动扩展的云计算实例，您需要知道新模型是否符合 ROI 的成本要求。

### 14.3.2 TFX 评估

现在我们来看看如何使用 TFX 评估当前训练的模型，以便我们可以决定它是否会成为下一个批准的模型。本质上，我们使用`Evaluator`和`InfraValidator`组件。

Evaluator

在`Trainer`组件完成后执行的`Evaluator`组件评估模型与基线。我们将来自`ExampleGen`组件的评估数据集以及来自`Trainer`组件的训练模型输入到`Evaluator`中。

如果存在之前批准的模型，我们也会将其输入。如果没有之前批准的模型，则跳过与批准模型基线的比较。

`Evaluator`组件使用 TensorFlow 模型分析度量库，除了 TFX 之外还需要导入，如下所示：

```
from tfx.components import Evaluator, ResolverNode
import tensorflow_model_analysis as tfma
```

下面的代码示例演示了将`Evaluator`组件构建到 TFX 管道中的最小要求，这些参数包括：

+   `examples`—`ExampleGen`的输出，它生成用于评估的示例批次

+   `model`—用于评估的`Trainer`训练模型的输出

```
evaluator = Evaluator(examples=example_gen.output['examples'],    ❶
                      model=trainer.output['model'],              ❶
                      baseline_model=None,                        ❷
                      eval_config=None                            ❸
                     )
```

❶ 参数的最小要求

❷ 没有用于比较的基线模型

❸ 用于评估的默认数据集切片

在前面的示例中，参数`eval_config`设置为`None`。在这种情况下，`Evaluator`将使用整个数据集进行评估，并使用在模型训练时指定的度量，例如分类模型的准确率。

当指定`eval_config`参数时，它接受一个`tfma.EvalConfig`实例，该实例接受三个参数：

+   `model_specs`—模型输入和输出的规范。默认情况下，假设输入是默认的服务签名。

+   `metrics_specs`—用于评估的一个或多个度量的规范。如果没有指定，则使用在模型训练时指定的度量。

+   `slicing_specs`—用于评估的数据集的一个或多个切片的规范。如果没有指定，则使用整个数据集。

```
eval_config = tfma.EvalConfig(model_specs=[],
                              metrics_specs=[],
                              slicing_specs=[]
                             )
```

`EvalConfig`的参数差异很大，我建议阅读 TensorFlow TFX 教程中的`Evaluator`组件([www.tensorflow.org/tfx/guide/evaluator](https://www.tensorflow.org/tfx/guide/evaluator))，以获得比我在这里涵盖的范围更深入的理解。

如果存在用于比较的先前批准的模型，则`baseline_model`参数设置为 TFX 组件`ResolverNode`的实例。

下面的代码示例是 `ResolverNode` 的最小规范，其中参数如下：

+   `instance_name`—这是分配给下一个祝福模型的名称，该模型作为元数据存储。

+   `resolver_class`—这是要使用的解析器对象的实例类型。在这种情况下，我们指定了实例类型以祝福最新的模型。

+   `model`—这指定了要祝福的模型类型。在这种情况下，`Channel (type=Model)` 可以是 TensorFlow 估计器或 TF.Keras 模型。

+   `model_blessing`—这指定了如何在元数据中存储祝福的模型。

```
from tfx.dsl.experimental.lastest_blessed_model_resolver import LatestBlessedModelResolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

baseline_model = ResolverNode(instance_name='blessed_model', 
                              resolver_class=LatestBlessedModelResolver,
                              model=Channel(type=Model),
                              model_blessing=Channel(type=ModelBlessing)
                             )
```

在前面的代码示例中，如果这是第一次为该模型调用 `ResolverNode()` 实例，则当前模型成为祝福的模型，并以 `blessed_model` 的实例名称存储在元数据中作为祝福模型。

否则，当前模型将与之前祝福的模型进行比较，该模型被标识为 `blessed_model`，并从元数据存储中相应地检索。在这种情况下，两个模型将与相同的评估片段进行比较，并比较它们相应的指标。如果新模型在指标上有所改进，它将成为下一个版本的 `blessed_model` 实例。

InfraValidator

在管道中的下一个组件是 `InfraValidator`。*Infra* 指的是 *基础设施*。只有当当前训练模型成为新的祝福模型时，才会调用此组件。此组件的目的是确定模型是否可以在模拟生产环境的沙盒环境中加载和查询。用户负责定义沙盒环境。换句话说，用户决定沙盒环境与生产环境的接近程度，因此决定了 *InfraValidator* 测试的准确性。

下一个代码示例展示了 `InfraValidator` 的最小参数要求：

+   `model`—训练模型（在此示例中，来自 `Trainer` 组件的当前训练模型）

+   `serving_spec`—沙盒环境的规范

```
from tfx.components import Evaluator, ResolverNode

infra_validator = InfraValidator(model=trainer.outputs['model'],     ❶
                                 serving_spec=serving_spec           ❷
                                )
```

❶ 部署到沙盒环境的训练模型

❷ 沙盒环境的规范

服务规范由两部分组成：

+   服务二进制的类型。截至 TFX 版本 0.22，仅支持 TensorFlow Serving。

+   服务平台的类型，可以是

    +   Kubernetes

    +   本地 Docker 容器

此示例展示了使用 TensorFlow Serving 和 Kubernetes 集群指定服务规范的最小要求：

```
from tfx.proto.infra_validator_pb2 import ServingSpec

serving_spec = ServingSpec(tensorflow_serving=TensorflowServing(tags=['latest']),
                           kubernetes=KubernetesConfig()
                          )
```

TFX 关于 ServingSpec 的文档目前很少，并将您重定向到 GitHub 存储库中的 protobuf 定义([`mng.bz/6NqA`](http://mng.bz/6NqA))以获取更多信息。

## 14.4 提供预测服务

现在我们有了新的祝福模型，我们将探讨如何将模型部署到生产环境中以提供预测服务。生产模型通常用于按需（实时）或批量预测。

批量预测与从部署的模型中进行的按需（实时）预测有何不同？有一个关键的区别，但除此之外，它们在结果上基本上是相同的：

+   *按需（实时）*——为整个实例集（一个或多个数据项）进行按需预测，并实时返回结果

+   *批量预测服务*——在后台为整个实例集进行排队（批量）预测，并在准备好时将结果存储在云存储桶中

### 14.4.1 按需（实时）服务

对于按需预测，例如通过交互式网站进行的在线请求，模型被部署到一个或多个计算实例上，并接收作为 HTTP 请求的预测请求。一个预测请求可以包含一个或多个单个预测；每个预测通常被称为*实例*。您可以有单实例请求，其中用户只想对一张图像进行分类，或者多实例请求，其中模型将为多张图像返回预测。

假设模型接收单实例请求：用户提交一个图像并希望得到一个预测，如分类或图像标题。这些都是通过互联网实时到达的按需请求。例如，它们可能来自用户浏览器中运行的 Web 应用程序，或者服务器上的后端应用程序作为微服务获取预测。

图 14.19 展示了这个过程。在这个描述中，模型包含在一个服务二进制文件中，该文件由一个 Web 服务器、服务功能和受祝福的模型组成。Web 服务器接收预测请求作为一个 HTTP 请求包，提取请求内容，并将内容传递给服务功能。服务功能随后将内容预处理成受祝福模型输入层期望的格式和形状，然后输入到受祝福模型中。受祝福模型将预测返回给服务功能，服务功能执行任何后处理以进行最终交付，然后将其返回给 Web 服务器，Web 服务器将后处理的预测作为 HTTP 响应包返回。

![图片](img/CH14_F19_Ferlitsch.png)

图 14.19 一个在生产二进制文件上的模型通过互联网接收按需预测请求

如您在图 14.19 中看到的，在客户端，一个或多个预测请求被传递给一个 Web 客户端。Web 客户端随后将创建一个单实例或多实例的预测 HTTP 请求包。预测请求被编码，通常为 base64，以确保通过互联网安全传输，并放置在 HTTP 请求包的内容部分。

Web 服务器接收 HTTP 请求，解码内容部分，并将单个或多个预测请求传递给服务功能。

现在我们来深入探讨服务函数的目的和构建方式。通常，在客户端，内容（如图片、视频、文本和结构化数据）以原始格式发送到服务二进制文件，而不进行任何预处理。当网络服务器接收到请求后，它会从请求包中提取内容并将其传递给服务函数。在传递给服务函数之前，内容可能需要进行解码，例如 base64 解码。

假设内容是一个包含单个实例请求的内容，例如 JPG 或 PNG 格式的压缩图像。假设模型的输入层是不压缩的图像字节，格式为多维数组，例如 TensorFlow 张量或 NumPy 数组。至少，服务函数必须执行模型之外的任何预处理（例如，预茎）。假设模型没有预茎，服务函数需要执行以下操作：

+   确定图像数据的压缩格式，例如从 MIME 类型。

+   将图像解压缩为原始字节。

+   将原始字节重塑为高度 × 宽度 × 通道（例如，RGB）。

+   将图像调整大小以匹配模型的输入形状。

+   对像素数据进行缩放，以进行归一化或标准化。

接下来是一个图像分类模型的服务函数示例实现，其中图像数据的预处理发生在模型上游，没有预茎。在这个例子中，`serving_fn()`方法通过将方法分配为模型的签名`serving_default`在服务二进制文件中的网络服务器上注册。我们在服务函数中添加了装饰器`@tf.function`，该装饰器指示 AutoGraph 编译器将 Python 代码转换为静态图，然后可以在 GPU 上与模型一起运行。在这个例子中，假设网络服务器将提取的预测请求内容（在这种情况下，JPG 压缩字节）作为 TensorFlow 字符串传递。对`tf.saved_model.save()`的调用将服务函数保存到与模型相同的存储位置，该位置由参数`export_path`指定。

现在我们来看看这个服务函数的主体。在下面的代码示例中，我们假设服务二进制文件中的网络服务器从 HTTP 请求包中提取内容，解码 base64 编码，并将内容（压缩的 JPG 图像字节）作为 TensorFlow 字符串数据类型`tf.string`传递。然后服务函数执行以下操作：

+   调用一个预处理函数`preprocess_fn()`，将 JPG 图像解码为原始字节，并调整大小和缩放以匹配底层模型的输入层，作为一个多维 TensorFlow 数组。

+   将多维 TensorFlow 数组传递给底层模型`m_call()`。

+   将底层模型返回的预测`prob`返回给网络服务器。

+   服务二进制文件中的 Web 服务器将预测结果打包到 HTTP 响应数据包中，返回给 Web 客户端。

```
@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def serving_fn(bytes_inputs):                                     ❶
    images = preprocess_fn(bytes_inputs)                          ❷
    prob = m_call(**images)                                       ❸
    return prob                                                   ❹

tf.saved_model.save(model, export_path, signatures={              ❺
    'serving_default': serving_fn,                                ❺
})
```

❶ 定义接收通过服务二进制文件的 Web 服务器内容的服务函数

❷ 将内容转换为与底层模型输入层匹配的方法

❸ 将预处理数据传递给底层模型进行预测。

❹ 预测结果被返回到服务二进制文件的 Web 服务器，作为 HTTP 响应返回。

❺ 将服务函数作为静态图与底层模型保存

以下是一个服务函数预处理步骤的示例实现。在这个例子中，函数`preprocess_fn()`接收来自 Web 服务器的 base64 解码后的 TensorFlow 字符串，并执行以下操作：

+   调用 TensorFlow 静态图操作`tf.io.decode_jpeg()`将输入解压缩为一个解压缩图像，作为多维 TensorFlow 数组。

+   调用 TensorFlow 静态图操作`tf.image.convert_image_dtype()`将整数像素值转换为 32 位浮点值，并将值缩放到 0 到 1 的范围（归一化）。

+   调用 TensorFlow 静态图操作`tf.image.resize()`将图像调整大小以适应模型的输入形状。在这个例子中，那将是(192, 192, 3)，其中值 3 是通道数。

+   将预处理后的图像数据传递给底层模型的输入层，该输入层由层的签名`numpy_inputs`指定。

```
def _preprocess(bytes_input):
    decoded = tf.io.decode_jpeg(bytes_input, channels=3)             ❶
    decoded = tf.image.convert_image_dtype(decoded, tf.float32)      ❷
    resized = tf.image.resize(decoded, size=(192, 192))              ❸
    return resized

@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def preprocess_fn(bytes_inputs):
    with tf.device("cpu:0"):
        decoded_images = tf.map_fn(_preprocess, bytes_inputs, 
        dtype=tf.float32)                                           ❹
    return {"numpy_inputs": decoded_images}                         ❺
```

❶ 将 TensorFlow 字符串解码为编码的 JPG，并将其转换为 TensorFlow 多维解压缩图像原始字节数据。

❷ 将像素转换为 32 位浮点值并缩放

❸ 将图像调整大小以适应底层模型的输入形状

❹ 预处理请求中的每个图像

❺ 将预处理图像传递给底层模型的输入层

以下是对底层模型调用的示例实现，此处进行了描述：

+   参数`model`是编译后的 TF.Keras 模型，其中`call()`方法是模型的前向馈预测方法。

+   `get_concrete_function()`方法构建了一个围绕底层模型的包装器，用于执行。包装器提供了从服务函数中的静态图切换到底层模型中的动态图的接口。

```
m_call = tf.function(model.call).get_concrete_function([tf.TensorSpec(shape=
            [None, 192, 192, 3], dtype=tf.float32, name="numpy_inputs")])
```

### 14.4.2 批量预测

*批量* *预测*与部署模型进行按需预测不同。在按需预测中，你创建一个服务二进制文件和服务平台来部署模型；我们称之为*端点*。然后你将模型部署到该端点。最后，用户向端点发出按需（实时）预测请求。

相比之下，批量预测从创建一个用于预测的批量作业开始。作业服务随后为批量预测请求分配资源，并将结果返回给调用者。然后作业服务释放请求的资源。

批量预测通常用于不需要立即响应的情况，因此响应可以延迟；需要处理的预测数量巨大（数百万）；并且只需要为处理批量分配计算资源。

例如，考虑一家金融机构，在银行日结束时有一百万笔交易，并且它有一个模型可以预测未来 10 天的存款和现金余额。由于预测是时间序列的，逐笔发送交易到实时预测服务是没有意义且效率低下的。相反，在银行日结束时，交易数据被提取（例如，从 SQL 数据库中）并作为一个单独的批量作业提交。然后为服务二进制文件和平台分配计算资源，处理作业，并释放服务二进制文件和平台（释放资源）。

图 14.20 展示了一个批量预测服务。此过程有五个主要步骤：

1.  累积的数据被提取并打包成批量请求，例如从 SQL 数据库中提取。

1.  批量请求已排队，队列管理器确定计算资源需求和优先级。

1.  当批量作业准备就绪时，它将从队列中出队到调度器。

1.  调度器为服务二进制文件和平台分配资源，然后提交批量作业。

1.  批量作业完成后，结果被存储，调度器释放分配的计算资源。

![](img/CH14_F20_Ferlitsch.png)

图 14.20 一个队列和调度器根据每个作业协调服务二进制文件和平台的分配和释放。

接下来，我们将介绍如何在 TFX 中部署模型以进行按需和批量预测。

### 14.4.3 TFX 部署管道组件

在 TFX 中，部署管道由组件 `Pusher` 和 `Bulk Inference` 以及一个服务二进制文件和平台组成。服务平台可以是基于云的、本地化的、边缘设备或基于浏览器的。对于基于云的模型，推荐的服务平台是 TensorFlow Serving。

图 14.21 展示了 TFX 部署管道的组件。`Pusher` 组件用于部署模型以进行按需预测或批量预测。`Bulk Inference` 组件处理批量预测。

![](img/CH14_F21_Ferlitsch.png)

图 14.21 一个 TFX 部署管道可以部署模型以进行按需服务和/或批量预测。

Pusher

以下是一个示例实现，展示了实例化 `Pusher` 组件以部署模型到服务二进制文件所需的最小要求：

+   `model`—要部署到服务二进制文件和平台的训练模型（在这种情况下，来自 `Trainer` 组件的当前训练模型实例）

+   `push_destination`—在服务二进制文件中安装模型的目录位置

```
from tfx.components import Pusher
from tfx.proto import pusher_pb2

pusher = Pusher(model=trainer.outputs['model'],                  ❶
                push_destination=pusher_pb2.PushDestination(     ❷

filesystem=pusher_pb2.PushDestination.FileSystem(

                base_directory=serving_model_dir                 ❸
                                                           )
               )
```

❶ 要部署的训练模型

❷ 部署模型的二进制文件目标

❸ 在服务二进制文件中的目录位置安装模型

在生产环境中，我们通常将其纳入部署管道中，只有当它是新的受祝福模型时才部署模型。以下是一个示例实现，其中只有当模型是新的受祝福模型时才部署模型：

+   `model`—来自`Trainer`组件的当前训练的模型

+   `model_blessing`—来自`Evaluator`组件的当前受祝福的模型

在这个例子中，只有在模型和受祝福的模型是相同的模型实例时，才会部署模型：

```
pusher = Pusher(model=trainer.outputs['model'],                  ❶
                model_blessing=evaluator.outputs['blessing'],    ❷
                push_destination=pusher_pb2.PushDestination( 

                filesystem=pusher_pb2.PushDestination.FileSystem(

                base_directory=serving_model_dir  
                                                            )
                                                           )
               )
```

❶ 当前训练的模型

❷ 当前受祝福的模型实例

接下来，我们将介绍在 TFX 中进行批量预测。

批量推理器

`BulkInferrer`组件执行批量预测服务，TFX 文档将其称为*批量推理*。以下代码是使用当前训练的模型进行批量预测的最小参数的示例实现：

+   `examples`—用于进行预测的示例。在这种情况下，它们来自`ExampleGen`组件的一个实例。

+   `model`—用于批量预测的模型（在这种情况下，当前训练的模型）。

+   `inference_result`—存储批量预测结果的位置。

```
from tfx.components import BulkInferrer

bulk_inferrer = BulkInferrer(examples=examples_gen.outputs['examples'],  ❶
                             model=trainer.outputs['model'],             ❷
                             inference_result=location                   ❸
                            )
```

❶ 批量预测的示例

❷ 用于批量预测的模型

❸ 存储预测结果的位置

以下是一个示例实现，仅当当前训练的模型是受祝福的模型时，使用当前训练的模型进行批量预测的最小参数。在这个例子中，只有在当前训练的模型和受祝福的模型实例相同时，才会执行批量预测。

```
from tfx.components import BulkInferrer

bulk_inferrer = BulkInferrer(examples=examples_gen.outputs['examples'],   
                             model=trainer.outputs['model'],  
                             model_blessing=evaluator.outputs['blessing'],  ❶
                             inference_result=location
                            )
```

❶ 当前受祝福的模型实例

### 14.4.4 A/B 测试

我们现在已经完成了对新训练的模型的两次测试，以查看它是否准备好成为下一个生产版本，即受祝福的模型。我们使用预定的评估数据在两个模型之间进行了模型指标的直接比较。我们还测试了候选模型在沙盒模拟的生产环境中。

尽管没有实际部署候选模型，但我们仍然不确定它是否是更好的模型。我们需要在*实时生产环境*中评估候选模型的性能。为此，我们向候选模型提供实时预测的子集，并测量候选模型和当前生产模型之间每个预测的结果。然后我们分析测量的数据或指标，以查看候选模型是否实际上是一个更好的模型。

这是在机器学习生产环境中的 A/B 测试。图 14.22 展示了这个过程。如图所示，两个模型都部署到了同一个实时生产环境中，预测流量在当前受祝福的（A）和候选受祝福的（B）之间分配。每个模型都看到基于百分比的随机选择的预测选择。

![图片](img/CH14_F22_Ferlitsch.png)

图 14.22 在实时生产环境中对当前模型和候选模型进行 A/B 测试，其中候选模型获得一小部分实时预测

如果候选模型不如当前模型好，我们不希望在生产环境中得到一个糟糕的结果。因此，我们通常将流量百分比保持尽可能小，但足以测量两者之间的差异。一个典型的分配是候选模型 5%，生产模型 95%。

接下来的问题是，你要测量什么？你已经测量并比较了模型的客观指标，所以重复这些指标的价值不大，尤其是如果评估切片包括服务倾斜和数据漂移。你在这里想要测量的是业务目标的结果有多好。

例如，假设你的模型是一个部署到制造装配线上的图像分类模型，用于寻找缺陷。对于每个模型，你有两个桶：一个用于好零件，一个用于缺陷。在指定的时间段后，你的 QA 人员会手动检查来自生产模型和候选模型桶中的采样分布，然后比较两者。特别是，他们想要回答两个问题：

+   候选模型检测到的缺陷数量是否与生产模型检测到的数量相等或更多？这些都是真阳性。

+   候选模型检测到的非缺陷数量是否与缺陷数量相等或更少？这些都是假阳性。

正如这个例子所示，你必须确定业务目标：增加真阳性，减少假阳性，或者两者都要。

让我们再考虑一个例子。假设我们正在为一个电子商务网站上的语言模型工作，这个模型执行的任务包括图像标题生成、为交易问题提供聊天机器人，以及为聊天机器人对用户的响应进行语言翻译。在这种情况下，我们可能测量的指标可能是完成交易的总数或每笔交易的平均收入。换句话说，候选模型是否能触及更广泛的受众并/或创造更多的收入？

### 14.4.5 负载均衡

一旦模型部署到按需生产环境，预测请求的量随时间可能会大幅变化。理想情况下，模型应在延迟约束内满足最高峰的需求，同时也要最小化计算成本。

如果模型是单体模型，即作为一个单一模型实例部署，我们可以简单地通过增加计算资源或 GPU 数量来满足第一个要求。但这样做会损害第二个要求，即最小化计算成本。

就像其他当代云应用一样，当请求流量有显著变化时，我们使用自动扩展和负载均衡来进行请求的分布式处理。让我们看看自动扩展在机器学习中的应用是如何工作的。

术语*自动扩展*和*负载均衡*可能看起来可以互换使用。但实际上，它们是两个独立的过程，协同工作。在*自动扩展*中，过程是响应整体当前预测请求负载进行提供（添加）和取消提供（删除）计算实例。在*负载均衡*中，过程是确定如何将当前预测请求负载分配到现有的已提供计算实例，并确定何时指令自动扩展过程提供或取消提供计算实例。

图 14.23 描述了一个适用于机器学习生产环境的负载均衡场景。本质上，负载均衡计算节点接收预测请求，然后将它们重定向到服务二进制，该服务二进制接收预测响应并将它们返回给客户端调用者。

![](img/CH14_F23_Ferlitsch.png)

图 14.23 一个负载均衡器将请求分配给由自动扩展节点动态提供和取消提供的多个服务二进制。

让我们更深入地看看图 14.23。负载均衡器监控流量负载，例如单位时间内的预测请求频率、网络流量的进出量以及返回预测请求响应的延迟时间。

这监控数据实时输入到自动扩展节点。自动扩展节点由 MLOps 人员配置以满足性能标准。如果性能低于预设的阈值和时长，自动扩展器将动态地提供一个或多个新的服务二进制副本实例。同样，如果性能高于预设的阈值和时长，自动扩展器将动态地取消提供一个或多个现有的服务二进制副本实例。

随着自动扩展器添加服务二进制，它将服务二进制注册到负载均衡器。同样，随着它移除服务二进制，它将服务二进制注销到负载均衡器。这告诉负载均衡器哪些服务二进制是活跃的，以便负载均衡器可以分配预测结果。

通常，负载均衡器配置了健康监控器来监控每个服务二进制的健康状态。如果确定服务二进制不健康，健康监控器将指令自动扩展节点取消提供该服务二进制并提供一个新的服务二进制作为替代。

### 14.4.6 持续评估

持续评估（CE）是软件开发过程中的持续集成（CI）和持续部署（CD）的机器学习生产扩展。这个扩展通常表示为 CI/CD/CE。*持续评估*意味着我们在模型部署到生产后，监控模型接收到的预测请求和响应，并对预测响应进行评估。这与使用现有的测试、服务偏差和数据漂移切片评估模型所做的工作类似。这样做是为了检测由于生产中预测请求随时间变化而导致的模型性能下降。

持续评估的典型过程如下：

+   将预配置的百分比（例如，2%）的预测请求和响应保存以供手动评估。

+   保存的预测请求和响应是随机选择的。

+   在某个周期性基础上，保存的预测请求和响应会手动审查并评估与模型的目标指标。

+   如果评估确定模型在目标指标上的表现低于模型部署前的评估，手动评估人员将识别出由于服务偏差、数据漂移和任何未预见到的情况导致的性能不佳的示例。这些都是异常情况。

+   确定的示例会手动标记并添加到训练数据集中，其中一部分被保留为相应的评估切片。

+   模型要么是增量重新训练，要么是全面重新训练。

图 14.24 描述了将部署的生产模型中的持续评估集成到模型开发过程中的 CI/CD/CE 方法。

![](img/CH14_F24_Ferlitsch.png)

图 14.24 一个生产部署的模型会持续评估以识别表现不佳的示例，然后这些示例将被添加到数据集中以重新训练模型。

## 14.5 生产管道设计演变

让我们以对机器学习从研究到全面生产的概念和必要性如何演变的简要讨论来结束这本书。你可能对模型融合的部分特别感兴趣，因为它是深度学习下一个前沿领域之一。

机器学习方法的演变是如何影响我们实际进行机器学习的方式的？深度学习模型的发展从实验室的实验到在全面的生产环境中部署和服务的演变。

### 14.5.1 机器学习作为管道

你可能之前见过这种情况。一个成功的机器学习工程师需要将机器学习解决方案分解为以下步骤：

1.  确定问题的模型类型。

1.  设计模型。

1.  准备模型的数据。

1.  训练模型。

1.  部署模型。

机器学习工程师将这些步骤组织成一个两阶段端到端管道。第一个端到端管道包括前三个步骤，如图 14.25 所示为建模、数据工程和训练。一旦机器学习工程师在这个阶段取得成功，它将与部署步骤相结合，形成一个第二个端到端管道。通常，模型被部署到容器环境中，并通过基于 REST 或微服务接口访问。

![图片](img/CH14_F25_Ferlitsch.png)

图 14.25 2017 年端到端机器学习管道的流行实践

那是 2017 年的流行做法。我将其称为*发现阶段*。组成部分是什么以及它们是如何相互配合的？

### 14.5.2 将机器学习作为 CI/CD 生产过程

在 2018 年，企业正在正式化 CI/CD 生产过程，我将其称为*探索阶段*。图 14.26 是我 2018 年末在谷歌演示中向商业决策者展示的幻灯片，捕捉了那时的我们所在的位置。这不仅仅是一个技术过程，还包括了计划和质量管理。数据工程变得更加明确，包括提取、分析、转换、管理和服务步骤。模型设计和训练包括特征工程，部署扩展到包括持续学习。

![图片](img/CH14_F26_Ferlitsch.png)

图 14.26 到 2018 年，谷歌和其他大型企业已经开始正式化生产过程，包括计划和质量管理阶段以及技术过程。

### 14.5.3 生产中的模型合并

今天的生产模型没有单一的输出层。相反，它们有多个输出层，从基本特征提取（常见层）、表示空间、潜在空间（特征向量、编码）和概率分布空间（软标签和硬标签）。现在的模型是整个应用；没有后端。 

这些模型学习最佳的接口和数据通信方式。2021 年的企业机器学习工程师现在正在指导模型合并中的搜索空间，其一个通用示例在图 14.27 中有所描述。

![图片](img/CH14_F27_Ferlitsch.png)

图 14.27 模型合并——当模型成为整个应用时！

让我们分解这个通用示例。在左侧是合并的输入。输入通过一组常见的卷积层进行处理，形成所谓的*共享模型底部*。在这个描述中，共享模型底部的输出有四个学习到的输出表示：1) 高维潜在空间，2) 低维潜在空间，3) 预激活条件概率分布，和 4) 后激活独立概率分布。

这些学习到的输出表示被专门的下游学习任务重复使用，这些任务执行某些操作（例如，状态转换变化或转换）。图中的每个任务（1、2、3 和 4）都重复使用对任务目标最优化（大小、速度、准确性）的输出表示。这些个别任务可能随后产生多个学习到的输出表示，或者将来自多个任务的学习表示（密集嵌入）组合起来，以供进一步的下游任务使用，正如你在第一章的体育广播示例中看到的。

不仅服务管道能够实现这些类型的解决方案，而且管道内的组件可以进行版本控制和重新配置。这使得这些组件可重用，这是现代软件工程的基本原则之一。

## 摘要

+   训练管道的基本组件包括模型喂养、模型评估和训练调度器。

+   模型每个实例的目标指标被保存为元数据。当模型实例的目标指标优于当前受祝福的模型时，该模型实例被祝福。

+   每个受祝福的模型都在模型存储库中进行跟踪和版本控制。

+   当模型用于分布式训练时，批次大小会增加，以平滑不同批次之间并行馈送时的差异。

+   在编排中，管理接口监督每个组件的执行，记住过去组件的执行情况，并维护历史记录。

+   评估切片包括与训练数据相同分布的示例，以及在生产中看到的分布外示例。这包括服务偏差和数据漂移。

+   部署管道的基本组件包括部署、服务、扩展和持续评估。

+   在实时生产环境中使用 A/B 测试来确定候选模型是否优于当前生产模型，例如，如果发生意外情况，不要干扰生产。

+   在实时生产环境中使用持续评估来识别服务偏差、数据漂移和异常，从而可以向数据集添加新的标记数据，并对模型进行进一步的重训练。
