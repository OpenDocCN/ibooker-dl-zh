# 第七章。监控训练过程

在上一章中，您学习了如何启动模型训练过程。在本章中，我们将介绍过程本身。

在本书中，我使用了相当直接的例子来帮助您理解每个概念。然而，在 TensorFlow 中运行真实的训练过程时，事情可能会更加复杂。例如，当出现问题时，您需要考虑如何确定您的模型是否*过拟合*训练数据（当模型学习并记忆训练数据和训练数据中的噪声以至于负面影响其学习新数据时就会发生过拟合）。如果是，您需要设置交叉验证。如果不是，您可以采取措施防止过拟合。

在训练过程中经常出现的其他问题包括：

+   在训练过程中应该多久保存一次模型？

+   在过拟合发生之前，我应该如何确定哪个时期给出了最佳模型？

+   我如何跟踪模型性能？

+   如果模型没有改进或出现过拟合，我可以停止训练吗？

+   有没有一种方法可以可视化模型训练过程？

TensorFlow 提供了一种非常简单的方法来解决这些问题：回调函数。在本章中，您将学习如何快速使用回调函数来监视训练过程。本章的前半部分讨论了`ModelCheckpoint`和`EarlyStopping`，而后半部分侧重于 TensorBoard，并向您展示了几种调用 TensorBoard 和使用它进行可视化的技巧。

# 回调对象

TensorFlow 的*回调对象*是一个可以执行由`tf.keras`提供的一组内置函数的对象。当训练过程中发生某些事件时，回调对象将执行特定的代码或函数。

使用回调是可选的，因此您不需要实现任何回调对象来训练模型。我们将看一下最常用的三个类：`ModelCheckpoint`、`EarlyStopping`和 TensorBoard。¹

## ModelCheckpoint

`ModelCheckpoint`类使您能够在训练过程中定期保存模型。默认情况下，在每个训练时期结束时，模型的权重和偏差会被最终确定并保存为权重文件。通常，当您启动训练过程时，模型会从该时期的训练数据中学习并更新权重和偏差，这些权重和偏差会保存在您在开始训练过程之前指定的目录中。然而，有时您只想在模型从上一个时期改进时保存模型，以便最后保存的模型始终是最佳模型。为此，您可以使用`ModelCheckpoint`类。在本节中，您将看到如何在模型训练过程中利用这个类。

让我们尝试在第六章中使用的 CIFAR-10 图像分类数据集中进行。像往常一样，我们首先导入必要的库，然后读取 CIFAR-10 数据：

```py
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pylab as plt
import os
from datetime import datetime

(train_images, train_labels), (test_images, test_labels) = 
datasets.cifar10.load_data()
```

首先，将图像中的像素值归一化为 0 到 1 的范围：

```py
train_images, test_images = train_images / 255.0, 
test_images / 255.0
```

该数据集中的图像标签由整数组成。使用 NumPy 命令验证这一点：

```py
np.unique(train_labels)
```

这显示的值为：

```py
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
```

现在您可以将这些整数映射到纯文本标签。这里提供的标签（由 Alex Krizhevsky、Vinod Nair 和 Geoffrey Hinton 提供）按字母顺序排列。因此，`airplane`在`train_labels`中映射为 0，而`truck`映射为 9：

```py
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

由于`test_images`有一个单独的分区，从`test_images`中提取前 500 张图像用于交叉验证，并将其命名为`validation_images`。剩下的图像将用于测试。

为了更有效地利用计算资源，将`test_images`的图像和标签从其原生 NumPy 数组格式转换为数据集格式：

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

执行这些命令后，您应该在三个数据集中拥有所有图像：一个训练数据集（`train_dataset`）、一个验证数据集（`validation_dataset`）和一个测试数据集（`test_dataset`）。

知道这些数据集的大小会很有帮助。要找到 TensorFlow 数据集的样本大小，将其转换为列表，然后使用`len`函数找到列表的长度：

```py
train_dataset_size = len(list(train_dataset.as_numpy_iterator()))
print('Training data sample size: ', train_dataset_size)

validation_dataset_size = len(list(validation_dataset.
as_numpy_iterator()))
print('Validation data sample size: ', 
validation_dataset_size)

test_dataset_size = len(list(test_dataset.as_numpy_iterator()))
print('Test data sample size: ', test_dataset_size)
```

您可以期待以下结果：

```py
Training data sample size:  50000
Validation data sample size:  500
Test data sample size:  9500
```

接下来，对这三个数据集进行洗牌和分批处理：

```py
TRAIN_BATCH_SIZE = 128
train_dataset = train_dataset.shuffle(50000).batch(
TRAIN_BATCH_SIZE, 
drop_remainder=True)

validation_dataset = validation_dataset.batch(
 validation_dataset_size)
test_dataset = test_dataset.batch(test_dataset_size)
```

请注意，`train_dataset`将被分成多个批次。每个批次将包含`TRAIN_BATCH_SIZE`个样本（在本例中为 128）。每个训练批次在训练过程中被馈送到模型中，以实现对权重和偏差的增量更新。对于验证和测试，不需要创建多个批次。它们将作为一个批次使用，但仅用于记录指标和测试。

接下来，指定多久更新权重和验证一次：

```py
STEPS_PER_EPOCH = train_dataset_size // TRAIN_BATCH_SIZE
VALIDATION_STEPS = 1
```

前面的代码意味着在模型看到由`STEPS_PER_EPOCH`指定的训练数据批次数量后，是时候使用验证数据集（作为一个批次）测试模型了。

为此，您首先要定义模型架构：

```py
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), 
      activation='relu',
      kernel_initializer='glorot_uniform', padding='same', 
      input_shape = (32,32,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), 
     activation='relu',
      kernel_initializer='glorot_uniform', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, 
     activation='relu', kernel_initializer='glorot_uniform'),
    tf.keras.layers.Dense(10, activation='softmax', 
    name = 'custom_class')
])
model.build([None, 32, 32, 3])
```

现在，编译模型以确保它设置正确：

```py
model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(
               from_logits=True),
          optimizer='adam',
          metrics=['accuracy'])
```

接下来，命名 TensorFlow 应在每个检查点保存模型的文件夹。通常，您会多次重新运行训练例程，可能会觉得每次创建一个唯一的文件夹名称很烦琐。一个简单且经常使用的方法是在模型名称后附加一个时间戳：

```py
MODEL_NAME = 'myCIFAR10-{}'.format(datetime.now().strftime(
"%Y%m%d-%H%M%S"))
print(MODEL_NAME)
```

前面的命令会产生一个名为*myCIFAR10-20210123-212138*的名称。您可以将此名称用于检查点目录：

```py
checkpoint_dir = './' + MODEL_NAME
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-{epoch}")
print(checkpoint_prefix)
```

前面的命令指定了目录路径为*./myCIFAR10-20210123-212138/ckpt-{epoch}*。该目录位于当前目录的下一级。*{epoch}*将在训练期间用 epoch 号进行编码。现在定义`myCheckPoint`对象：

```py
myCheckPoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    monitor='val_accuracy',
    mode='max')
```

在这里，您指定了 TensorFlow 将在每个 epoch 保存模型的文件路径。您还设置了监视验证准确性。

当您使用回调启动训练过程时，回调将期望一个 Python 列表。因此，让我们将`myCheckPoint`对象放入 Python 列表中：

```py
myCallbacks = [
    myCheckPoint
]
```

现在启动训练过程。此命令将整个模型训练历史分配给对象`hist`，这是一个 Python 字典：

```py
hist = model.fit(
    train_dataset,
    epochs=12,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=VALIDATION_STEPS,
    callbacks=myCallbacks).history
```

您可以使用命令`hist['val_accuracy']`查看从训练的第一个 epoch 到最后一个 epoch 的交叉验证准确性。显示应该类似于这样：

```py
[0.47200000286102295,
 0.5680000185966492,
 0.6000000238418579,
 0.5899999737739563,
 0.6119999885559082,
 0.6019999980926514,
 0.6100000143051147,
 0.6380000114440918,
 0.6100000143051147,
 0.5699999928474426,
 0.5619999766349792,
 0.5960000157356262]
```

在这种情况下，交叉验证准确性在一些 epoch 中有所提高，然后逐渐下降。这种下降是过拟合的典型迹象。这里最好的模型是具有最高验证准确性（数组中最高值）的模型。要确定其在数组中的位置（或索引），请使用此代码：

```py
max_value = max(hist['val_accuracy'])
max_index = hist['val_accuracy'].index(max_value)
print('Best epoch: ', max_index + 1)
```

请记住将`max_index`加 1，因为 epoch 从 1 开始，而不是 0（与 NumPy 数组索引不同）。输出是：

```py
Best epoch:  8
```

接下来，通过在 Jupyter Notebook 单元格中运行以下 Linux 命令来查看检查点目录：

```py
!ls -lrt ./cifar10_training_checkpoints
```

您将看到此目录的内容（如图 7-1 所示）。

![在每个检查点保存的模型](img/t2pr_0701.png)

###### 图 7-1。在每个检查点保存的模型

您可以重新运行此命令并指定特定目录，以查看特定 epoch 构建的模型（如图 7-2 所示）：

```py
!ls -lrt ./cifar10_training_checkpoints/ckpt_8
```

![在检查点 8 保存的模型文件](img/t2pr_0702.png)

###### 图 7-2。在检查点 8 保存的模型文件

到目前为止，您已经看到如何使用`CheckPoint`在每个 epoch 保存模型。如果您只希望保存最佳模型，请指定`save_best_only = True`：

```py
best_only_checkpoint_dir = 
 './best_only_cifar10_training_checkpoints'
best_only_checkpoint_prefix = os.path.join(
best_only_checkpoint_dir, 
"ckpt_{epoch}")

bestCheckPoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_only_checkpoint_prefix,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
```

然后将`bestCheckPoint`放入回调列表中：

```py
    bestCallbacks = [
    bestCheckPoint
]
```

之后，您可以启动训练过程：

```py
best_hist = model.fit(
    train_dataset,
    epochs=12,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=VALIDATION_STEPS,
    callbacks=bestCallbacks).history
```

在这个训练中，而不是保存所有检查点，`bestCallbacks`只有在验证准确率比上一个 epoch 更好时才保存模型。`save_best_only`选项允许您在第一个 epoch 之后*仅*在模型指标有增量改进时保存检查点（使用`monitor`指定），因此最后保存的检查点是最佳模型。

要查看您保存的内容，请在 Jupyter Notebook 单元格中运行以下命令：

```py
 !ls -lrt ./best_only_cifar10_training_checkpoints
```

在图 7-3 中显示了验证准确率逐渐提高的保存模型。

![设置 save_best_only 为 True 保存的模型](img/t2pr_0703.png)

###### 图 7-3。设置 save_best_only 为 True 的模型

第一个 epoch 的模型始终会被保存。在第三个 epoch，模型在验证准确率上有所改进，因此第三个检查点模型被保存。训练继续进行。第九个 epoch 中验证准确率提高，因此第九个检查点模型是最后一个被保存的目录。训练持续到第 12 个 epoch，没有进一步增加验证准确率的改进。这意味着第九个检查点目录包含了最佳验证准确率的模型。

现在您熟悉了`ModelCheckpoint`，让我们来看看另一个回调对象：`EarlyStopping`。

## EarlyStopping

`EarlyStopping`回调对象使您能够在达到最终 epoch 之前停止训练过程。通常，如果模型没有改进，您会这样做以节省训练时间。

该对象允许您指定一个模型指标，例如验证准确率，通过所有 epoch 进行监视。如果指定的指标在一定数量的 epoch 后没有改进，训练将停止。

要定义一个`EarlyStopping`对象，请使用以下命令：

```py
myEarlyStop = tf.keras.callbacks.EarlyStopping(
monitor='val_accuracy',
patience=4)
```

在这种情况下，您在每个 epoch 监视验证准确率。您将`patience`参数设置为 4，这意味着如果验证准确率在四个 epoch 内没有改进，训练将停止。

###### 提示

要了解更多自定义提前停止的方法，请参阅[TensorFlow 2 文档](https://oreil.ly/UDpaA)。

要在回调中使用`ModelCheckpoint`对象实现提前停止，需要将其放入列表中：

```py
myCallbacks = [
    myCheckPoint,
    myEarlyStop
]
```

训练过程是相同的，但您指定了`callbacks=myCallbacks`：

```py
hist = model.fit(
    train_dataset,
    epochs=20,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=VALIDATION_STEPS,
    callbacks=myCallbacks).history
```

一旦您启动了前面的训练命令，输出应该类似于图 7-4。

![训练过程中的提前停止](img/t2pr_0704.png)

###### 图 7-4。训练过程中的提前停止

在图 7-4 中显示的训练中，最佳验证准确率出现在第 15 个 epoch，值为 0.7220。再经过四个 epoch，验证准确率没有超过该值，因此在第 19 个 epoch 后停止训练。

## 总结

`ModelCheckpoint`类允许您设置条件或频率以在训练期间保存模型，而`EarlyStopping`类允许您在模型没有改进到您选择的指标时提前终止训练。这些类一起在 Python 列表中指定，并将此列表作为回调传递到训练例程中。

许多其他用于监控训练进度的功能可用（请参阅[tf.keras.callbacks.Callback](https://oreil.ly/1nIE6)和[Keras Callbacks API](https://oreil.ly/BeJBW)），但`ModelCheckpoint`和`EarlyStopping`是最常用的两个。

本章的其余部分将深入探讨被称为`TensorBoard`的流行回调类，它提供了您的训练进度和结果的可视化表示。

# TensorBoard

如果您希望可视化您的模型和训练过程，TensorBoard 是您的工具。TensorBoard 提供了一个视觉表示，展示了您的模型参数和指标在训练过程中如何演变。它经常用于跟踪训练周期中的模型准确性。它还可以让您看到每个模型层中的权重和偏差如何演变。就像`ModelCheckpoint`和`EarlyStopping`一样，TensorBoard 通过回调模块应用于训练过程。您创建一个代表`Tensorboard`的对象，然后将该对象作为回调列表的成员传递。

让我们尝试构建一个对 CIFAR-10 图像进行分类的模型。像往常一样，首先导入库，加载 CIFAR-10 图像，并将像素值归一化到 0 到 1 的范围内：

```py
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pylab as plt
import os
from datetime import datetime

(train_images, train_labels), (test_images, test_labels) = 
datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, 
 test_images / 255.0
```

定义您的纯文本标签：

```py
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat',
               'deer','dog', 'frog', 'horse', 'ship', 'truck']
```

现在将图像转换为数据集：

```py
validation_dataset = tf.data.Dataset.from_tensor_slices(
(test_images[:500], test_labels[:500]))

test_dataset = tf.data.Dataset.from_tensor_slices(
(test_images[500:], test_labels[500:]))

train_dataset = tf.data.Dataset.from_tensor_slices(
(train_images, train_labels))
```

然后确定用于训练、验证和测试分区的数据大小：

```py
train_dataset_size = len(list(train_dataset.as_numpy_iterator()))
print('Training data sample size: ', train_dataset_size)

validation_dataset_size = len(list(validation_dataset.
as_numpy_iterator()))
print('Validation data sample size: ', 
 validation_dataset_size)

test_dataset_size = len(list(test_dataset.as_numpy_iterator()))
print('Test data sample size: ', test_dataset_size)
```

您的结果应该如下所示：

```py
Training data sample size:  50000
Validation data sample size:  500
Test data sample size:  9500
```

现在您可以对数据进行洗牌和分批处理：

```py
TRAIN_BATCH_SIZE = 128
train_dataset = train_dataset.shuffle(50000).batch(
TRAIN_BATCH_SIZE, 
drop_remainder=True)

validation_dataset = validation_dataset.batch(
validation_dataset_size)
test_dataset = test_dataset.batch(test_dataset_size)
```

然后指定参数以设置更新模型权重的节奏：

```py
STEPS_PER_EPOCH = train_dataset_size // TRAIN_BATCH_SIZE
VALIDATION_STEPS = 1
```

`STEPS_PER_EPOCH`是一个整数，从`train_dataset_size`和`TRAIN_BATCH_SIZE`之间的除法向下取整得到。（双斜杠表示除法并向下取整到最接近的整数。）

我们将重用我们在“ModelCheckpoint”中构建的模型架构：

```py
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 
     kernel_size=(3, 3), 
     activation='relu', 
     name = 'conv_1',
     kernel_initializer='glorot_uniform', 
     padding='same', input_shape = (32,32,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), 
     activation='relu', name = 'conv_2',
      kernel_initializer='glorot_uniform', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, 
     kernel_size=(3, 3), 
     activation='relu', 
     name = 'conv_3',
      kernel_initializer='glorot_uniform', padding='same'),
    tf.keras.layers.Flatten(name = 'flat_1',),
    tf.keras.layers.Dense(64, activation='relu',   
     kernel_initializer='glorot_uniform', 
     name = 'dense_64'),
    tf.keras.layers.Dense(10, 
     activation='softmax', 
     name = 'custom_class')
])
model.build([None, 32, 32, 3])
```

请注意，这次每个层都有一个名称。为每个层指定一个名称有助于您知道您正在检查哪个层。这不是必需的，但对于在 TensorBoard 中进行可视化是一个好的实践。

现在编译模型以确保模型架构有效，并指定损失函数：

```py
model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(
               from_logits=True),
          optimizer='adam',
          metrics=['accuracy'])
```

设置模型名称将在以后有所帮助，当 TensorBoard 让您选择一个模型（或多个模型）并检查其训练结果的可视化时。您可以像在使用`ModelCheckpoint`时那样在模型名称后附加一个时间戳：

```py
MODEL_NAME =
'myCIFAR10-{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))

print(MODEL_NAME)
```

在这个例子中，`MODEL_NAME`是`myCIFAR10-20210124-135804`。您的将类似。

接下来，设置检查点目录：

```py
checkpoint_dir = './' + MODEL_NAME
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-{epoch}")
print(checkpoint_prefix)
```

*./myCIFAR10-20210124-135804/ckpt-{epoch}*是这个检查点目录的名称。

定义模型检查点：

```py
myCheckPoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    monitor='val_accuracy',
mode='max')
```

接下来，您将定义一个 TensorBoard，然后我们将更仔细地查看这段代码：

```py
myTensorBoard = tf.keras.callbacks.TensorBoard(
log_dir='./tensorboardlogs/{}'.format(MODEL_NAME),
write_graph=True,
write_images=True,
histogram_freq=1)
```

这里指定的第一个参数是`log_dir`。这是您想要保存训练日志的路径。如指示，它在当前级别下的一个名为*tensorboardlogs*的目录中，后面跟着一个名为`MODEL_NAME`的子目录。随着训练的进行，您的日志将在这里生成和存储，以便 TensorBoard 可以解析它们进行可视化。

参数`write_graph`设置为 True，这样模型图将被可视化。另一个参数`write_images`也设置为 True。这确保模型权重将被写入日志，这样您可以可视化它们在训练过程中的变化。

最后，`histogram_freq`设置为 1。这告诉 TensorBoard 何时按 epoch 创建可视化：1 表示每个 epoch 创建一个可视化。有关更多参数，请参阅 TensorBoard 的[文档](https://oreil.ly/k1Pd2)。

最后，您有两个回调对象要设置：`myCheckPoint`和`myTensorBoard`。要将两者放入 Python 列表中，您只需执行以下操作：

```py
myCallbacks = [
    myCheckPoint,
    myTensorBoard
]
```

然后将您的`myCallbacks`列表传递到训练例程中：

```py
hist = model.fit(
    train_dataset,
    epochs=30,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=VALIDATION_STEPS,
    callbacks=myCallbacks).history
```

一旦训练过程完成，有三种方法可以调用 TensorBoard。您可以在您自己的计算机上的 Jupyter Notebook 中的下一个单元格中运行它，也可以在您自己计算机上的命令终端中运行，或者在 Google Colab 中运行。我们将依次查看这些选项。

## 通过本地 Jupyter Notebook 调用 TensorBoard

如果您选择使用您的 Jupyter Notebook，在下一个单元格中运行以下命令：

```py
!tensorboard --logdir='./tensorboardlogs/'
```

请注意，在这种情况下，当您指定路径以查找训练日志时，参数是`logdir`，而不是在定义`myTensorBoard`时的`log_dir`。

运行上述命令后，您将看到以下内容：

```py
Serving TensorBoard on localhost; to expose to the network, use a
proxy or pass --bind_all
TensorBoard 2.3.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

正如您所看到的，TensorBoard 正在您当前的计算实例（localhost）上以端口号 6006 运行。

现在打开浏览器，导航至*http://localhost:6006*，您将看到 TensorBoard 正在运行，如图 7-5 所示。

![TensorBoard 可视化](img/t2pr_0705.png)

###### 图 7-5\. TensorBoard 可视化

正如您所看到的，通过每个时代，准确性和损失都在图表中追踪。训练数据显示为浅灰色，验证数据显示为深灰色。

使用 Jupyter Notebook 单元格的主要优势是方便。缺点是运行`!tensorboard`命令的单元格将保持活动状态，直到您停止 TensorBoard，您将无法使用此笔记本。

## 通过本地命令终端调用 TensorBoard

您的第二个选项是在本地环境的命令终端中启动 TensorBoard。如图 7-6 所示，等效命令是：

```py
tensorboard --logdir='./tensorboardlogs/'
```

![从命令终端调用 TensorBoard](img/t2pr_0706.png)

###### 图 7-6\. 从命令终端调用 TensorBoard

请记住，`logdir`是由训练回调 API 创建的训练日志的目录路径。前面代码中的命令使用相对路径表示法；如果您愿意，可以使用完整路径。

输出与在 Jupyter Notebook 中看到的完全相同：URL（*http://localhost:6006*）。使用浏览器打开此 URL 以显示 TensorBoard。

## 通过 Colab 笔记本调用 TensorBoard

现在是我们的第三个选项。如果您在此练习中使用 Google Colab 笔记本，那么调用 TensorBoard 将与您迄今为止看到的有所不同。您将无法在本地计算机上打开浏览器指向 Colab 笔记本，因为它将在 Google 的云环境中运行。因此，您需要安装 TensorBoard 笔记本扩展。这可以在第一个单元格中完成，当您导入所有库时。只需添加此命令并在第一个 Colab 单元格中运行：

```py
%load_ext tensorboard
```

完成后，每当您准备调用 TensorBoard（例如在训练完成后），请使用此命令：

```py
%tensorboard --logdir ./tensorboardlogs/
```

您将看到输出在您的 Colab 笔记本中运行，看起来如图 7-5 所示。

## 使用 TensorBoard 可视化模型过拟合

当您将 TensorBoard 用作模型训练的回调时，您将获得从第一个时代到最后一个时代的模型准确性和损失的图表。

例如，对于我们的 CIFAR-10 图像分类模型，您将看到类似于图 7-5 中所示的输出。在该特定训练运行中，尽管训练和验证准确性都在提高，损失在减少，但这两个趋势开始趋于平缓，表明进一步的训练时期可能只会带来较小的收益。

还要注意，在此运行中，验证准确性低于训练准确性，而验证损失高于训练损失。这是有道理的，因为模型在训练数据上的表现比在交叉验证数据上测试时更好。

您还可以在 TensorBoard 的 Scalars 选项卡中获得图表，就像图 7-7 中所示的那样。

在图 7-7 中，较暗的线表示验证指标，而较浅灰色的线表示训练指标。训练数据中的模型准确性远高于交叉验证数据，而训练数据中的损失远低于交叉验证数据。

您可能还注意到，交叉验证准确率在第 10 个时期达到峰值，略高于 0.7。之后，验证数据准确率开始下降，而损失开始增加。这是模型过拟合的明显迹象。这些图表告诉您的是，在第 10 个时期之后，您的模型开始记忆训练数据的模式。当遇到新的、以前未见过的数据（如交叉验证图像）时，这并没有帮助。事实上，模型在交叉验证中的表现（准确率和损失）将开始变差。

![在 TensorBoard 中显示的模型过拟合](img/t2pr_0707.png)

###### 图 7-7. 在 TensorBoard 中显示的模型过拟合

一旦您检查了这些图表，您将知道哪个时期提供了训练过程中最佳的模型。您还将了解模型何时开始过拟合并记忆其训练数据。

如果您的模型仍有改进的空间，就像 图 7-5 中的那个，您可能决定增加训练时期，并在过拟合模式开始出现之前继续寻找最佳模型（参见 图 7-7）。

## 使用 TensorBoard 可视化学习过程

TensorBoard 中的另一个很酷的功能是权重和偏差分布的直方图。这些显示为训练结果的每个时期。通过可视化这些参数是如何分布的，以及它们的分布随时间如何变化，您可以深入了解训练过程的影响。

让我们看看如何使用 TensorBoard 来检查模型的权重和偏差分布。这些信息将在 TensorBoard 的直方图选项卡中（图 7-8）。

![TensorBoard 中的权重和偏差直方图](img/t2pr_0708.png)

###### 图 7-8. TensorBoard 中的权重和偏差直方图

左侧是训练过的所有模型的面板。请注意有两个模型被选中。右侧是它们的权重（表示为 `kernel_0`）和偏差在每个训练时期的分布。每一行的图表示模型中的特定层。第一层被命名为 `conv_1`，这是您在设置模型架构时给这一层取的名字。

让我们更仔细地检查这些图表。我们将从 conv_1 层开始，如 图 7-9 所示。

![conv_1 层中的偏差分布经过训练](img/t2pr_0709.png)

###### 图 7-9. conv_1 层中的偏差分布经过训练

在两个模型中，`conv_1` 层中偏差值的分布从第一个时期（背景）到最后一个时期（前景）肯定发生了变化。方框表明随着训练的进行，这一层的所有节点中开始出现某种偏差分布模式。新值远离零，或整体分布的中心。

让我们也看一看权重的分布。这次，让我们只关注一个模型和一个层：conv_3。这在 图 7-10 中显示。

![conv_3 层中的权重分布经过训练](img/t2pr_0710.png)

###### 图 7-10. conv_3 层中的权重分布经过训练

值得注意的是，随着训练的进行，分布变得更广泛、更平坦。这可以从直方图从第一个到最后一个时期的峰值计数中看出，从 1.22e+4 到 7.0e+3。这意味着直方图逐渐变得更广泛，有更多的权重被更新为远离零的值（直方图的中心）。

使用 TensorBoard，您可以检查不同层和模型训练运行的组合，看看它们如何受训练过程或模型架构的变化影响。这就是为什么 TensorBoard 经常用于直观检查模型训练过程。

# 总结

在本章中，您看到了一些用于跟踪模型训练过程的最流行方法。本章介绍了模型检查点的概念，并提供了两种重要的方法来帮助您管理如何在训练过程中保存模型：在每个周期保存模型，或者仅在模型指标有增量改进时保存模型。您还了解到，交叉验证中的模型准确度决定了模型何时开始过拟合训练数据。

在本章中，您了解到的另一个重要工具是 TensorBoard，它可以用来可视化训练过程。TensorBoard 通过训练周期展示基本指标（准确度和损失）的趋势的可视图像。它还允许您检查每个层的权重和偏差分布。所有这些技术都可以通过回调函数轻松实现在训练过程中。

在下一章中，您将看到如何在 TensorFlow 中实现分布式训练，利用诸如 GPU 之类的高性能计算单元，以提供更短的训练时间。

¹ 这里没有涵盖的另外两个常见且有用的功能是[LearningRateScheduler](https://oreil.ly/CyuGs)和[CSVLogger](https://oreil.ly/vmeaY)。
