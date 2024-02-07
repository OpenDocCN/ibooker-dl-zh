# 第8章。分布式训练

训练一个机器学习模型可能需要很长时间，特别是如果您的训练数据集很大或者您正在使用单台机器进行训练。即使您有一个GPU卡可供使用，训练一个复杂的模型（如ResNet50，一个具有50个卷积层的计算机视觉模型，用于将对象分类为一千个类别）仍可能需要几周的时间。

减少模型训练时间需要采用不同的方法。您已经看到了一些可用的选项：例如，在[第5章](ch05.xhtml#data_pipelines_for_streaming_ingestion)中，您学习了如何利用数据管道中的数据集。然后还有更强大的加速器，如GPU和TPU（这些加速器仅在Google Cloud中提供）。

本章将介绍一种不同的模型训练方式，称为*分布式训练*。分布式训练在一组设备（如CPU、GPU和TPU）上并行运行模型训练过程，以加快训练过程。 （在本章中，为了简洁起见，我将把GPU、CPU和TPU等硬件加速器称为*工作节点*或*设备*。）阅读完本章后，您将了解如何为分布式训练重构您的单节点训练例程。（到目前为止，您在本书中看到的每个示例都是单节点：也就是说，它们都使用一台带有一个CPU的机器来训练模型。）

在分布式训练中，您的模型由多个独立的进程进行训练。您可以将每个进程视为一个独立的训练尝试。每个进程在单独的设备上运行训练例程，使用训练数据的一个子集（称为*片*）。这意味着每个进程使用不同的训练数据。当每个进程完成一个训练周期时，它将结果发送回*主例程*，该主例程收集和聚合结果，然后向所有进程发出更新。然后，每个进程使用更新后的权重和偏差继续训练。

在我们深入实现代码之前，让我们更仔细地看一看分布式ML模型训练的核心。我们将从数据并行性的概念开始。

# 数据并行性

关于分布式训练，您需要了解的第一件事是如何处理训练数据。分布式训练中的主要架构被称为*数据并行性*。在这种架构中，您在每个工作节点上运行相同的模型和计算逻辑。每个工作节点使用不同于其他工作节点的数据片段计算损失和梯度，然后使用这些梯度来更新模型参数。然后，每个单独的工作节点中更新的模型将在下一轮计算中使用。这个概念在[图8-1](#data_parallelism_architecture_left_paren)中有所说明。

设计用于使用这些梯度更新模型的两种常见方法：异步参数服务器和同步allreduce。我们将依次查看每种方法。

![数据并行性架构（改编自Google I/O 2018视频中的分布式TensorFlow训练）](Images/t2pr_0801.png)

###### 图8-1。数据并行性架构（改编自[分布式TensorFlow训练](https://oreil.ly/beSob)中的Google I/O 2018视频）

## 异步参数服务器

让我们首先看一下*异步参数服务器*方法，如[图8-2](#distributed_training_using_asynchronous)所示。

![使用异步参数服务器进行分布式训练（改编自Google I/O 2018视频中的分布式TensorFlow训练）](Images/t2pr_0802.png)

###### 图8-2。使用异步参数服务器进行分布式训练。 （改编自[分布式TensorFlow训练](https://oreil.ly/beSob)中的Google I/O 2018视频）

在[图8-2](#distributed_training_using_asynchronous)中标记为PS0和PS1的设备是*参数服务器*；这些服务器保存模型的参数。其他设备被指定为工人，如[图8-2](#distributed_training_using_asynchronous)中标记的那样。

工人们承担了大部分计算工作。每个工人从服务器获取参数，计算损失和梯度，然后将梯度发送回参数服务器，参数服务器使用这些梯度来更新模型的参数。每个工人都独立完成这个过程，因此这种方法可以扩展到使用大量工人。这里的优势在于，如果训练工人被高优先级的生产工作抢占，如果工人之间存在不对称，或者如果一台机器因维护而宕机，都不会影响你的扩展，因为工人们不需要等待彼此。

然而，存在一个缺点：工人可能会失去同步。这可能导致在过时的参数值上计算梯度，从而延迟模型收敛，因此延迟朝着最佳模型的训练。随着硬件加速器的普及和流行，这种方法比同步全局归约实现得更少，接下来我们将讨论同步全局归约。

## 同步全局归约

随着GPU和TPU等快速硬件加速器变得更加普遍，*同步全局归约*方法变得更加常见。

在同步全局归约架构中（如[图8-3](#distributed_training_using_a_synchronous)所示），每个工人在自己的内存中保存模型的参数副本。没有参数服务器。相反，每个工人根据一部分训练样本计算损失和梯度。一旦计算完成，工人们相互通信以传播梯度并更新模型参数。所有工人都是同步的，这意味着下一轮计算仅在每个工人接收到新梯度并相应地更新模型参数后才开始。

![使用同步全局归约架构进行分布式训练（改编自Google I/O 2018视频中的分布式TensorFlow培训）](Images/t2pr_0803.png)

###### 图8-3. 使用同步全局归约架构的分布式训练（改编自[分布式TensorFlow培训](https://oreil.ly/beSob)在Google I/O 2018视频中）

在一个连接的集群中，工人之间的处理时间差异不是问题。因此，这种方法通常比异步参数服务器架构更快地收敛到最佳模型。

*全局归约*是一种算法，它将不同工人的梯度合并在一起。这种算法通过汇总不同工人的梯度值，例如将它们求和，然后将它们复制到不同的工人中。它的实现可以非常高效，因为它减少了同步梯度所涉及的开销。根据工人之间可用的通信类型和架构的拓扑结构，有许多全局归约算法的实现。这种算法的常见实现被称为*环形全局归约*，如[图8-4](#ring-allreduce_implementation_left_paren)所示。

在环形全局归约实现中，每个工人将其梯度发送给环上的后继者，并从前任者接收梯度。最终，每个工人都会收到合并梯度的副本。环形全局归约能够最优地利用网络带宽，因为它同时使用每个工人的上传和下载带宽。无论是在单台机器上与多个工人一起工作，还是在少量机器上工作，都能快速完成。

![环形全局归约实现（改编自Google I/O 2018视频中的分布式TensorFlow培训）](Images/t2pr_0804.png)

###### 图8-4. 环形全局归约实现（改编自[分布式TensorFlow培训](https://oreil.ly/beSob)在Google I/O 2018视频中）

现在让我们看看如何在TensorFlow中完成所有这些。我们将专注于使用同步allreduce架构扩展到多个GPU。您将看到将单节点训练代码重构为allreduce有多么容易。这是因为这些高级API在幕后处理了许多数据并行性的复杂性和细微差别。

# 使用`tf.distribute.MirroredStrategy`类

实现分布式训练的最简单方法是使用TensorFlow提供的`tf.distribute.MirroredStrategy`类。（有关TensorFlow支持的各种分布式训练策略的详细信息，请参见[TensorFlow文档中的“策略类型”](https://oreil.ly/0jQed)）。正如您将看到的，实现此类仅需要在源代码中进行最小更改，您仍然可以在单节点模式下运行，因此您无需担心向后兼容性。它还负责为您更新权重和偏差、指标和模型检查点。此外，您无需担心如何将训练数据分割为每个设备的碎片。您无需编写代码来处理从每个设备检索或更新参数。您也无需担心如何确保跨设备聚合梯度和损失。分发策略会为您处理所有这些。

我们将简要查看一些代码片段，演示在一个机器上使用多个设备时需要对训练代码进行的更改：

1.  创建一个对象来处理分布式训练。您可以在源代码的开头执行此操作：

    ```py
    strategy = tf.distribute.MirroredStrategy()
    ```

    `strategy`对象包含一个属性，其中包含机器上可用设备的数量。您可以使用此命令显示您可以使用多少个GPU或TPU：

    ```py
    print('Number of devices: {}'.format(
     strategy.num_replicas_in_sync))
    ```

    如果您正在使用GPU集群，例如通过Databricks或云提供商的环境，您将看到您可以访问的GPU数量：

    ```py
    Number of devices: 2
    ```

    请注意，Google Colab仅为每个用户提供一个GPU。

1.  将您的模型定义和损失函数包装在`strategy`范围内。您只需确保模型定义和编译，包括您选择的损失函数，封装在特定范围内：

    ```py
    with strategy.scope():
      model = tf.keras.Sequential([
      …
      ])
      model.build(…)
      model.compile(
        loss=tf.keras.losses…,
        optimizer=…,
        metrics=…)
    ```

    这是您需要进行代码更改的唯一两个地方。

`tf.distribute.MirroredStrategy`类是幕后的工作马。正如您所见，我们创建的`strategy`对象知道有多少设备可用。这些信息使其能够将训练数据分成不同的碎片，并将每个碎片馈送到特定设备中。由于模型架构包装在此对象的范围内，因此它也保存在每个设备的内存中。这使得每个设备可以在相同的模型架构上运行训练例程，最小化相同的损失函数，并根据其特定的训练数据碎片更新梯度。模型架构和参数被复制，或*镜像*，在所有设备上。`MirroredStrategy`类还在幕后实现了环形allreduce算法，因此您无需担心从每个设备聚合所有梯度。

该类知道您的硬件设置及其潜力进行分布式训练，因此您无需更改`model.fit`训练例程或数据摄入方法。保存模型检查点和模型摘要的方式与单节点训练中相同，正如我们在[“ModelCheckpoint”](ch07.xhtml#modelcheckpoint)中看到的那样。

## 设置分布式训练

要尝试本章中的分布式训练示例，您需要访问多个GPU或TPU。为简单起见，考虑使用提供GPU集群的各种商业平台之一，例如[Databricks](https://databricks.com)和[Paperspace](https://www.paperspace.com)。其他选择包括主要的云供应商，它们提供各种平台，从托管服务到容器。为了简单起见和易于获取，本章中的示例是在Databricks中完成的，这是一个基于云的计算供应商。它允许您设置一个分布式计算集群，可以是GPU或CPU，以运行单节点机器无法处理的重型工作负载。

虽然Databricks提供免费的“社区版”，但它不提供访问GPU集群的权限；为此，您需要[创建一个付费账户](https://oreil.ly/byE1d)。然后，您可以将Databricks与您选择的云供应商关联，并使用[图8-5](#setting_up_a_databricks_gpu_cluster)中显示的配置创建GPU集群。我的建议是：完成工作后，下载您的笔记本并删除您创建的集群。

![设置Databricks GPU集群](Images/t2pr_0805.png)

###### 图8-5。设置Databricks GPU集群

您可能会注意到[图8-5](#setting_up_a_databricks_gpu_cluster)中有自动驾驶选项。启用自动缩放选项将根据工作负载的需要自动扩展到更多的工作节点。为了节约成本，我还设置了此集群在120分钟不活动后自动终止的选项。（请注意，终止集群并不意味着您已经删除它。它将继续存在并在您的账户中产生一小笔费用，直到您删除它。）完成配置后，点击顶部的“创建集群”按钮。通常需要大约10分钟来完成整个过程。

接下来，创建一个笔记本（[图8-6](#creating_a_notebook_in_the_databricks_en)）。

![在Databricks环境中创建笔记本](Images/t2pr_0806.png)

###### 图8-6。在Databricks环境中创建笔记本

给您的笔记本命名，确保默认语言设置为Python，并选择您刚刚创建的集群（[图8-7](#setting_up_your_notebook)）。点击“创建”按钮生成一个空白笔记本。

![设置您的笔记本](Images/t2pr_0807.png)

###### 图8-7。设置您的笔记本

现在确保您的笔记本已连接到GPU集群（[图8-8](#attaching_your_notebook_to_an_active_clu)）。

![将您的笔记本附加到活动集群](Images/t2pr_0808.png)

###### 图8-8。将您的笔记本附加到活动集群

现在继续启动GPU集群。然后转到“库”选项卡，点击“安装新”按钮，如[图8-9](#installing_libraries_in_a_databricks_clu)所示。

![在Databricks集群中安装库](Images/t2pr_0809.png)

###### 图8-9。在Databricks集群中安装库

当出现“安装库”提示时（[图8-10](#installing_the_tensorflow-datasets_libra)），选择PyPl作为库源，在“包”字段中输入`**tensorflow-datasets**`，然后点击“安装”按钮。

完成后，您将能够使用TensorFlow的数据集API来完成本章中的示例。在下一节中，您将看到如何使用Databricks笔记本来尝试使用您刚刚创建的GPU集群进行分布式训练。

![在Databricks集群中安装tensorflow-datasets库](Images/t2pr_0810.png)

###### 图8-10。在Databricks集群中安装tensorflow-datasets库

## 使用tf.distribute.MirroredStrategy的GPU集群

在[第7章](ch07.xhtml#monitoring_the_training_process-id00010)中，您使用CIFAR-10图像数据集构建了一个单节点训练的图像分类器。在这个例子中，您将使用分布式训练方法来训练分类器。

像往常一样，您需要做的第一件事是导入必要的库：

```py
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import os
from datetime import datetime
```

现在是创建`MirroredStrategy`对象以处理分布式训练的好时机：

```py
strategy = tf.distribute.MirroredStrategy()
```

您的输出应该看起来像这样：

```py
INFO:tensorflow:Using MirroredStrategy with devices 
('/job:localhost/replica:0/task:0/device:GPU:0', 
'/job:localhost/replica:0/task:0/device:GPU:1')
```

这表明有两个GPU。您可以使用以下语句确认这一点，就像我们之前做过的那样：

```py
print('Number of devices: {}'.format(
strategy.num_replicas_in_sync))
Number of devices: 2
```

现在加载训练数据并将每个图像像素范围归一化为0到1之间：

```py
(train_images, train_labels), (test_images, test_labels) = 
datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, 
test_images / 255.0
```

您可以在列表中定义纯文本标签：

```py
# Plain-text name in alphabetical order. See
https://www.cs.toronto.edu/~kriz/cifar.html
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 
               'deer','dog', 'frog', 'horse', 'ship', 'truck']
```

这些纯文本标签来自[CIFAR-10数据集](https://oreil.ly/fCvCX)，按字母顺序排列：“airplane”映射到`train_labels`中的值0，而“truck”映射到9。

由于`test_images`有一个单独的分区，从`test_images`中提取前500个图像用作验证图像，同时保留其余部分用于测试。此外，为了更有效地使用计算资源，将这些图像和标签从其原生NumPy数组格式转换为数据集格式：

```py
validation_dataset = tf.data.Dataset.from_tensor_slices(
(test_images[:500], 
 test_labels[:500]))

test_dataset = tf.data.Dataset.from_tensor_slices(
(test_images[500:], 
 test_labels[500:]))

train_dataset = tf.data.Dataset.from_tensor_slices(
(train_images,
 train_labels))
```

执行这些命令后，您将拥有训练数据集、验证数据集和测试数据集的所有图像格式。

了解数据集的大小将是很好的。要找出TensorFlow数据集的样本大小，将其转换为列表，然后使用`len`函数找到列表的长度：

```py
train_dataset_size = len(list(train_dataset.as_numpy_iterator()))
print('Training data sample size: ', train_dataset_size)

validation_dataset_size = len(list(validation_dataset.
as_numpy_iterator()))
print('Validation data sample size: ', validation_dataset_size)

test_dataset_size = len(list(test_dataset.as_numpy_iterator()))
print('Test data sample size: ', test_dataset_size)
```

您可以期待这些结果：

```py
Training data sample size:  50000
Validation data sample size:  500
Test data sample size:  9500
```

接下来，对三个数据集进行洗牌和分批处理：

```py
TRAIN_BATCH_SIZE = 128
train_dataset = train_dataset.shuffle(50000).batch(
TRAIN_BATCH_SIZE, drop_remainder=True)

validation_dataset = validation_dataset.batch(validation_dataset_size)
test_dataset = test_dataset.batch(test_dataset_size)
```

请注意，`train_dataset`将被分成多个批次，每个批次包含`TRAIN_BATCH_SIZE`个样本。每个训练批次在训练过程中被馈送到模型中，以实现对权重和偏差的增量更新。无需为验证和测试创建多个批次：这些将作为一个批次使用，仅用于指标记录和测试。

接下来，指定权重更新和验证应该发生的频率：

```py
STEPS_PER_EPOCH = train_dataset_size // TRAIN_BATCH_SIZE
VALIDATION_STEPS = 1
```

前面的代码意味着在模型看到`STEPS_PER_EPOCH`批次的训练数据后，是时候用作一个批次的验证数据集进行测试了。

现在您需要将模型定义、模型编译和损失函数包含在策略范围内：

```py
with strategy.scope():
  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
activation='relu', name = 'conv_1',
      kernel_initializer='glorot_uniform',
padding='same', input_shape = (32,32,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
activation='relu', name = 'conv_2',
      kernel_initializer='glorot_uniform', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(name = 'flat_1'),
    tf.keras.layers.Dense(256, activation='relu',
kernel_initializer='glorot_uniform', name = 'dense_64'),
    tf.keras.layers.Dense(10, activation='softmax',
name = 'custom_class')
  ])
  model.build([None, 32, 32, 3])

  model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])
```

其余部分与您在[第7章](ch07.xhtml#monitoring_the_training_process-id00010)中所做的相同。您可以定义目录名称模式以在训练例程中检查点模型：

```py
MODEL_NAME = 'myCIFAR10-{}'.format(datetime.now().strftime(
"%Y%m%d-%H%M%S"))

checkpoint_dir = './' + MODEL_NAME
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-{epoch}")
print(checkpoint_prefix)
```

前面的命令指定目录路径类似于*./ myCIFAR10-20210302-014804/ckpt-{epoch}*。

一旦定义了检查点目录，只需将定义传递给`ModelCheckpoint`。为简单起见，我们只会在训练时代提高模型在验证数据上的准确性时保存检查点：

```py
myCheckPoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    monitor='val_accuracy',
    mode='max',
    save_weights_only=True,
    save_best_only = True)
```

然后将定义包装在一个列表中：

```py
myCallbacks = [
    myCheckPoint
]
```

现在使用`fit`函数启动训练例程，就像您在本书中的其他示例中所做的那样：

```py
hist = model.fit(
    train_dataset,
    epochs=12,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=VALIDATION_STEPS,
    callbacks=myCallbacks).history
```

在前面的命令中，`hist`对象以Python字典格式包含有关训练结果的信息。这个字典中感兴趣的属性是`val_accuracy`项：

```py
hist['val_accuracy']
```

这将显示从训练的第一个到最后一个时代的验证准确性。从这个列表中，我们可以确定在评分验证数据时具有最高准确性的时代。这就是您要用于评分的模型：

```py
max_value = max(hist['val_accuracy'])
max_index = hist['val_accuracy'].index(max_value)
print('Best epoch: ', max_index + 1)
```

由于您设置了检查点以保存最佳模型而不是每个时代，一个更简单的替代方法是加载最新的时代：

```py
tf.train.latest_checkpoint(checkpoint_dir)
```

这将为您提供`checkpoint_dir`下的最新检查点。从该检查点加载具有该检查点权重的模型如下：

```py
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
```

使用加载了最佳权重的模型对测试数据进行评分：

```py
eval_loss, eval_acc = model.evaluate(test_dataset)
print('Eval loss: {}, Eval Accuracy: {}'.format(
eval_loss, eval_acc))
```

典型的结果看起来像这样：

```py
1/1 [==============================] - 0s 726us/step
 loss: 1.7533
 accuracy: 0.7069 
Eval loss: 1.753335952758789, 
 Eval Accuracy: 0.706947386264801
```

## 摘要

您所看到的是使用分布式TensorFlow模型训练的最简单方法。您学会了如何从商业供应商的平台创建一个GPU集群，以及如何将单节点训练代码重构为分布式训练例程。此外，您还了解了分布式机器学习的基本知识以及不同的系统架构。

在下一节中，您将学习另一种实现分布式训练的方法。您将使用由Uber创建的开源库Horovod，该库在其核心也利用了环形全局归约算法。虽然这个库需要更多的重构，但如果您想比较训练时间差异，它可能会作为另一个选项为您提供服务。

# Horovod API

在上一节中，您学习了allreduce的工作原理。您还看到了`tf.distribute` API如何在幕后自动化分布式训练的各个方面，因此您只需要创建一个分布式训练对象，并将训练代码包装在对象的范围内。在本节中，您将了解Horovod，这是一个较旧的分布式训练API，需要您在代码中处理这些分布式训练的方面。由于`tf.distribute`很受欢迎且易于使用，Horovod API通常不是程序员的首选。在这里介绍它的目的是为您提供另一个分布式训练的选项。

与上一节一样，我们将使用Databricks作为学习分布式模型训练Horovod基础知识的平台。如果您按照我的说明操作，您将拥有一个由两个GPU组成的集群。

要了解Horovod API的工作原理，您需要了解两个关键参数：每个GPU的标识和用于并行训练的进程数。每个参数都分配给一个Horovod环境变量，该变量将在您的代码中使用。在这种特殊情况下，您应该有两个GPU。每个GPU将在一个数据片段上进行训练，因此将有两个训练进程。您可以使用以下函数检索Horovod环境变量：

等级

等级表示GPU的标识。如果有两个GPU，则一个GPU将被指定为等级值0，另一个GPU将被指定为等级值1。如果有更多GPU，则指定的等级值将是2、3等。

大小

大小表示GPU的总数。如果有两个GPU，则Horovod的范围大小为2，训练数据将被分成两个片段。同样，如果有四个GPU，则该值将为4，数据将被分成四个片段。

您将经常看到这两个函数被使用。您可以参考[Horovod文档](https://oreil.ly/KC893)获取更多详细信息。

## 实现Horovod API的代码模式

在我展示在Databricks中运行Horovod的完整源代码之前，让我们看一下如何运行Horovod训练作业。在Databricks中使用Horovod进行分布式训练的一般模式是：

```py
from sparkdl import HorovodRunner
hr = HorovodRunner(np=2)
hr.run(train_hvd, checkpoint_path=CHECKPOINT_PATH, 
learning_rate=LEARNING_RATE)
```

使用Databricks，您需要根据上述模式运行Horovod分布式训练。基本上，您创建一个名为`hr`的`HorovodRunner`对象，该对象分配两个GPU。然后，该对象执行一个`run`函数，该函数将`train_hvd`函数分发到每个GPU。`train_hvd`函数负责在每个GPU上执行数据摄入和训练例程。此外，`checkpoint_path`用于在每个训练时代保存模型，`learning_rate`用于训练过程的反向传播步骤中使用。

随着每个时代的训练进行，模型的权重和偏差被聚合、更新并存储在GPU 0上。`learning_rate`是由Databricks驱动程序指定的另一个参数，并传播到每个GPU。然而，在使用上述模式之前，您需要组织和实现几个函数，接下来我们将详细介绍。

## 封装模型架构

Databricks的主驱动程序的工作是将训练数据和模型架构蓝图分发到每个GPU。因此，您需要将模型架构包装在一个函数中。当执行`hr.run`时，`train_hvd`函数将在每个GPU上执行。在`train_hvd`中，将调用一个类似于这样的模型架构包装函数：

```py
def get_model(num_classes):
  from tensorflow.keras import models
  from tensorflow.keras import layers

  model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 
    kernel_size=(3, 3), 
    activation='relu', 
    name = 'conv_1',
    kernel_initializer='glorot_uniform', 
    padding='same', 
    input_shape = (32,32,3)),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(64, 
    kernel_size=(3, 3), 
    activation='relu', 
    name = 'conv_2',
    kernel_initializer='glorot_uniform', 
    padding='same'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(name = 'flat_1'),
  tf.keras.layers.Dense(256, 
    activation='relu', 
    kernel_initializer='glorot_uniform', 
    name = 'dense_64'),
  tf.keras.layers.Dense(num_classes, 
    activation='softmax', 
    name = 'custom_class')
  ])
  model.build([None, 32, 32, 3])
  return model
```

正如您所看到的，这是您在前一节中使用的相同模型架构，只是包装为一个函数。该函数将在每个GPU中将模型对象返回给执行过程。

## 封装数据分离和分片过程

为了确保每个GPU接收到一部分训练数据，您还需要将数据处理步骤封装为一个函数，该函数可以传递到每个GPU中，就像模型架构一样。

举个例子，让我们使用相同的数据集CIFAR-10来说明如何确保每个GPU获得不同的训练数据片段。看一下以下函数：

```py
def get_dataset(num_classes, rank=0, size=1):
  from tensorflow.keras import backend as K
  from tensorflow.keras import datasets, layers, models
  from tensorflow.keras.models import Sequential
  import tensorflow as tf
  from tensorflow import keras
  import horovod.tensorflow.keras as hvd
  import numpy as np

  (train_images, train_labels), (test_images, test_labels) =
      datasets.cifar10.load_data()

  #50000 train samples, 10000 test samples.
  train_images = train_images[rank::size]
  train_labels = train_labels[rank::size]

  test_images = test_images[rank::size]
  test_labels = test_labels[rank::size]

  # Normalize pixel values to be between 0 and 1
  train_images, test_images = train_images / 255.0, 
    test_images / 255.0

  return train_images, train_labels, test_images, test_labels
```

注意函数签名中的输入参数`rank`和`size`。`rank`默认为0，`size`默认为1，因此与单节点训练兼容。在具有多个GPU的分布式训练中，每个GPU将`hvd.rank`和`hvd.size`作为输入传递到此函数中。由于每个GPU的身份由`hvd.rank`通过双冒号(::)表示，图像和标签根据从一个记录到下一个跳过多少步来切片和分片。因此，此函数返回的数组`train_images`、`train_labels`、`test_images`和`test_labels`对于每个GPU都是不同的，取决于其`hvd.rank`。（有关NumPy数组跳过和切片的详细解释，请参见[此Colab笔记本](https://oreil.ly/23bmZ)。）

## 参数同步在工作节点之间

在开始训练之前，重要的是初始化和同步所有工作节点（设备）之间的权重和偏置的初始状态。这是通过一个回调函数完成的：

```py
hvd.callbacks.BroadcastGlobalVariablesCallback(0)
```

这实际上是将变量状态从排名为0的GPU广播到所有其他GPU。

所有工作节点的错误指标需要在每个训练步骤之间进行平均。这是通过另一个回调函数完成的：

```py
hvd.callbacks.MetricAverageCallback()
```

这也在训练期间传递到回调函数列表中。

最好在早期使用低学习率，然后在前5个时期之后切换到首选学习率，您可以通过以下代码指定热身时期的数量来实现：

```py
hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5)
```

此外，在训练过程中，当模型指标停止改善时，包括一种减小学习率的方法：

```py
tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor = 0.2)
```

在这个例子中，如果模型指标在10个时期后没有改善，您将开始将学习率降低0.2倍。

为了简化事情，我建议将所有这些回调函数放在一个列表中：

```py
callbacks = [
   hvd.callbacks.BroadcastGlobalVariablesCallback(0),
   hvd.callbacks.MetricAverageCallback(),
   hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, 
     verbose=1),
   tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)
]
```

## 模型检查点作为回调

如前所述，当所有工作节点完成一个时期的训练后，模型参数将作为检查点保存在排名为0的工作节点中。这是使用以下代码片段完成的：

```py
if hvd.rank() == 0:
   callbacks.append(keras.callbacks.ModelCheckpoint(
   filepath=checkpoint_path,
   monitor='val_accuracy',
   mode='max',
   save_best_only = True
   ))
```

这是为了防止工作节点之间的冲突，确保在模型性能和验证指标方面只有一个真相版本。如前面的代码所示，当`save_best_only`设置为True时，只有在该时期的验证指标优于上一个时期时，模型和训练参数才会被保存。因此，并非所有时期都会导致模型被保存，您可以确保最新的检查点是最佳模型。

## 梯度聚合的分布式优化器

梯度计算也是分布式的，因为每个工作节点都会执行自己的训练例程并单独计算梯度。您需要聚合然后平均来自不同工作节点的所有梯度，然后将平均值应用于所有工作节点，用于下一步训练。这是通过以下代码片段实现的：

```py
optimizer = tf.keras.optimizers.Adadelta(
lr=learning_rate * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)
```

在这里，`hvd.DistributedOptimizer`将单节点优化器的签名包装在Horovod的范围内。

## 使用Horovod API进行分布式训练

现在让我们看一下在Databricks中使用Horovod API进行分布式训练的完整实现。此实现使用与[“使用类tf.distribute.MirroredStrategy”](#using_the_tfdotdistributedotmirroredstra)中看到的相同数据集（CIFAR-10）和模型架构：

```py
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import os
import time

def get_dataset(num_classes, rank=0, size=1):
  from tensorflow.keras import backend as K
  from tensorflow.keras import datasets, layers, models
  from tensorflow.keras.models import Sequential
  import tensorflow as tf
  from tensorflow import keras
  import horovod.tensorflow.keras as hvd
  import numpy as np

  (train_images, train_labels), (test_images, test_labels) = 
datasets.cifar10.load_data()
  #50000 train samples, 10000 test samples.
  train_images = train_images[rank::size]
  train_labels = train_labels[rank::size]

  test_images = test_images[rank::size]
  test_labels = test_labels[rank::size]

  # Normalize pixel values to be between 0 and 1
  train_images, test_images = train_images / 255.0, 
    test_images / 255.0

  return train_images, train_labels, test_images, test_labels
```

前面的代码将在每个工作人员上执行。每个工作人员都会收到自己的`train_images`、`train_labels`、`test_images`和`test_labels`。

以下代码是一个包装模型架构的函数；它将构建到每个工作人员中：

```py
def get_model(num_classes):
  from tensorflow.keras import models
  from tensorflow.keras import layers

  model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 
    kernel_size=(3, 3), 
    activation='relu', 
    name = 'conv_1',
    kernel_initializer='glorot_uniform', 
    padding='same', 
    input_shape = (32,32,3)),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(64, kernel_size=(3, 3), 
    activation='relu', 
    name = 'conv_2',
    kernel_initializer='glorot_uniform', 
    padding='same'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(name = 'flat_1'),
  tf.keras.layers.Dense(256, activation='relu', 
     kernel_initializer='glorot_uniform', 
     name = 'dense_64'),
  tf.keras.layers.Dense(num_classes, 
     activation='softmax', 
     name = 'custom_class')
  ])
  model.build([None, 32, 32, 3])
  return model
```

接下来是主要的训练函数`train_hvd`，它调用了刚刚展示的两个函数。这个函数相当冗长，所以我会分块解释它。

在`train_hvd`内部，使用命令`hvd.init`创建并初始化了一个Horovod对象。此函数将`checkpoint_path`和`learning_rate`作为输入，用于存储每个时代的模型并在反向传播过程中设置梯度下降的速率。一开始，导入所有库：

```py
def train_hvd(checkpoint_path, learning_rate=1.0):

  # Import tensorflow modules to each worker
  from tensorflow.keras import backend as K
  from tensorflow.keras.models import Sequential
  import tensorflow as tf
  from tensorflow import keras
  import horovod.tensorflow.keras as hvd
  import numpy as np
```

然后，创建并初始化一个Horovod对象，并使用它来访问您的工作人员的配置，以便稍后可以正确分片数据：

```py

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
# These steps are skipped on a CPU cluster
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(
  gpus[hvd.local_rank()], 'GPU')

print(' Horovod size (processes): ', hvd.size())
```

现在您已经创建了`hvd`对象，使用它来提供工作人员标识（`hvd.rank`）和并行进程数（`hvd.size`）给`get_dataset`函数，该函数将以分片返回训练和验证数据。

一旦您有了这些分片，将它们转换为数据集，以便您可以像在[“使用tf.distribute.MirroredStrategy的GPU集群”](#using_a_gpu_cluster_with_tfdotdistribute)中那样流式传输训练数据：

```py

  # Call the get_dataset function you created, this time with the
    Horovod rank and size
  num_classes = 10
  train_images, train_labels, test_images, test_labels = get_dataset(
num_classes, hvd.rank(), hvd.size())

  validation_dataset = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels))
  train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels))
```

对训练和验证数据集进行洗牌和分批处理：

```py
  NUM_CLASSES = len(np.unique(train_labels))
  BUFFER_SIZE = 10000
  BATCH_SIZE_PER_REPLICA = 64
  validation_dataset_size = len(test_labels)
  BATCH_SIZE = BATCH_SIZE_PER_REPLICA * hvd.size()
  train_dataset = train_dataset.repeat().shuffle(BUFFER_SIZE).
    batch(BATCH_SIZE
  validation_dataset = validation_dataset.
  repeat().shuffle(BUFFER_SIZE).
batch(BATCH_SIZE, drop_remainder = True)

  train_dataset_size = len(train_labels)

  print('Training data sample size: ', train_dataset_size)

  validation_dataset_size = len(test_labels)
  print('Validation data sample size: ', validation_dataset_size)
```

现在定义批量大小、训练步数和训练时代：

```py
TRAIN_DATASET_SIZE = len(train_labels)
STEPS_PER_EPOCH = TRAIN_DATASET_SIZE // BATCH_SIZE_PER_REPLICA
VALIDATION_STEPS = validation_dataset_size // 
  BATCH_SIZE_PER_REPLICA
EPOCHS = 20
```

使用`get_model`函数创建一个模型，设置优化器，指定学习率，然后使用适合此分类任务的正确损失函数编译模型。请注意，优化器被`DistributedOptimizer`包装以进行分布式训练：

```py
model = get_model(10)

# Adjust learning rate based on number of GPUs
optimizer = tf.keras.optimizers.Adadelta(
             lr=learning_rate * hvd.size())

# Use the Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
            metrics=['accuracy'])
```

在这里，您将创建一个回调列表，以在工作人员之间同步变量，对梯度进行聚合和平均以进行同步更新，并根据时代或训练性能调整学习率：

```py
# Create a callback to broadcast the initial variable states from
  rank 0 to all other processes.
# This is required to ensure consistent initialization of all 
# workers when training is started with random weights or 
# restored from a checkpoint.
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, 
      verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(patience=10, 
      verbose=1)
  ]
```

最后，这是每个时代模型检查点的回调。此回调仅在0级工作人员（`hvd.rank() == 0`）中执行：

```py
# Save checkpoints only on worker 0 to prevent conflicts between 
# workers
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only = True
    ))
```

现在是最终的`fit`函数，将启动模型训练例程：

```py
model.fit(train_dataset,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=validation_dataset,
            validation_steps=VALIDATION_STEPS,
           verbose=1)
print('DISTRIBUTED TRAINING DONE')
```

这结束了`train_hvd`函数。

在您的Databricks笔记本的下一个单元格中，为每个训练时代指定一个检查点目录：

```py
# Create directory
checkpoint_dir = '/dbfs/ml/CIFAR10DistributedDemo/train/{}/'.
format(time.time())
os.makedirs(checkpoint_dir)
print(checkpoint_dir)
```

`checkpoint_dir`将看起来像*/dbfs/ml/CIFAR10DistributedDemo/train/1615074200.2146788/*。

在下一个单元格中，继续启动分布式训练例程：

```py
from sparkdl import HorovodRunner

checkpoint_path = checkpoint_dir + '/checkpoint-{epoch}.ckpt'
learning_rate = 0.1
hr = HorovodRunner(np=2)
hr.run(train_hvd, checkpoint_path=checkpoint_path, 
       learning_rate=learning_rate)
```

在运行器定义中，`HorovodRunner(np=2)`，将进程数指定为每个设置的两个（请参阅[“设置分布式训练”](#setting_up_distributed_training)），这将设置两个Standard_NC12工作人员GPU。

训练例程完成后，请使用以下命令查看检查点目录：

```py
ls -lrt  /dbfs/ml/CIFAR10DistributedDemo/train/1615074200.2146788/
```

您应该看到类似于这样的内容：

```py
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-9.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-8.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-7.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-6.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-5.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-4.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-3.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-2.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-20.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-1.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-19.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-17.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-16.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-15.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-14.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-13.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-11.ckpt/
drwxrwxrwx 2 root root 4096 Mar 6 23:18 checkpoint-10.ckpt/
```

如果没有模型在先前的时代中有所改进，则会跳过一些检查点。最新的检查点代表具有最佳验证指标的模型。

# 总结

在本章中，您了解了在具有多个工作人员的环境中使分布式模型训练正常工作所需的内容。在数据并行框架中，有两种主要的分布式训练模式：异步参数服务器和同步allreduce。如今，由于高性能加速器的普遍可用性，同步allreduce更受欢迎。

通过学习如何使用Databricks GPU集群执行两种类型的同步allreduce API：TensorFlow自己的`tf.distribute` API和Uber的Horovod API。TensorFlow选项提供了最优雅和方便的使用方式，并且需要最少的代码重构，而Horovod API需要用户手动处理数据分片、分发管道、梯度聚合和平均以及模型检查点。这两种选项通过确保每个工作节点执行自己的训练，然后在每个训练步骤结束时，在所有工作节点之间同步和一致地更新梯度来执行分布式训练。这是分布式训练的标志。

恭喜！通过学习本章内容，您学会了如何使用云中的一组GPU训练具有分布式数据管道和分布式训练例程的深度学习模型。在下一章中，您将学习如何为推理服务一个TensorFlow模型。
