# 第五章。流摄入的数据管道

数据摄入是你工作流程中的一个重要部分。在原始数据达到模型期望的正确输入格式之前，需要执行几个步骤。这些步骤被称为 *数据管道*。数据管道中的步骤很重要，因为它们也将应用于生产数据，这是模型在部署时使用的数据。无论你是在构建和调试模型还是准备部署模型，你都需要为模型的消费格式化原始数据。

在模型构建过程中使用与部署规划相同的一系列步骤是很重要的，这样测试数据就会与训练数据以相同的方式处理。

在第三章中，你学习了 Python 生成器的工作原理，在第四章中，你学习了如何使用 `flow_from_directory` 方法进行迁移学习。在本章中，你将看到 TensorFlow 提供的更多处理其他数据类型（如文本和数值数组）的工具。你还将学习如何处理另一种图像文件结构。当处理文本或图像进行模型训练时，文件组织变得尤为重要，因为通常会使用目录名称作为标签。本章将在构建和训练文本或图像分类模型时推荐一种目录组织实践。

# 使用 `text_dataset_from_directory` 函数流式文本文件

只要正确组织目录结构，你几乎可以在管道中流式传输任何文件。在本节中，我们将看一个使用文本文件的示例，这在文本分类和情感分析等用例中会很有用。这里我们感兴趣的是 `text_dataset_from_directory` 函数，它的工作方式类似于我们用于流式传输图像的 `flow_from_directory` 方法。

为了将这个函数用于文本分类问题，你必须按照本节中描述的目录组织。在你当前的工作目录中，你必须有与文本标签或类名匹配的子目录。例如，如果你正在进行文本分类模型训练，你必须将训练文本组织成积极和消极。这是训练数据标记的过程；必须这样做以设置数据，让模型学习积极或消极评论的样子。如果文本是被分类为积极或消极的电影评论语料库，那么子目录的名称可能是 *pos* 和 *neg*。在每个子目录中，你有该类别的所有文本文件。因此，你的目录结构将类似于这样：

```py
Current working directory
    pos
        p1.txt
        p2.txt
    neg
        n1.txt
        n2.txt
```

举个例子，让我们尝试使用来自互联网电影数据库（IMDB）的电影评论语料库构建一个文本数据摄入管道。

## 下载文本数据并设置目录

你将在本节中使用的文本数据是[大型电影评论数据集](https://oreil.ly/EabEP)。你可以直接下载它，也可以使用 `get_file` 函数来下载。让我们首先导入必要的库，然后下载文件：

```py
import io
import os
import re
import shutil
import string
import tensorflow as tf

url = "https://ai.stanford.edu/~amaas/data/sentiment/
       aclImdb_v1.tar.gz"

ds = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')
```

通过传递 `untar=True`，`get_file` 函数也会解压文件。这将在当前目录中创建一个名为 *aclImdb* 的目录。让我们将这个文件路径编码为一个变量以供将来参考：

```py
ds_dir = os.path.join(os.path.dirname(ds), 'aclImdb')
```

列出这个目录以查看里面有什么：

```py
train_dir = os.path.join(ds_dir, 'train')
os.listdir(train_dir)

['neg',
 'unsup',
 'urls_neg.txt',
 'urls_unsup.txt',
 'pos',
 'urls_pos.txt',
 'unsupBow.feat',
 'labeledBow.feat']
```

有一个目录（*unsup*）没有在使用中，所以你需要将其删除：

```py
unused_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(unused_dir)
```

现在看一下训练目录中的内容：

```py
!ls -lrt ./aclImdb/train
-rw-r--r-- 1 7297 1000  2450000 Apr 12  2011 urls_unsup.txt
drwxr-xr-x 2 7297 1000   364544 Apr 12  2011 pos
drwxr-xr-x 2 7297 1000   356352 Apr 12  2011 neg
-rw-r--r-- 1 7297 1000   612500 Apr 12  2011 urls_pos.txt
-rw-r--r-- 1 7297 1000   612500 Apr 12  2011 urls_neg.txt
-rw-r--r-- 1 7297 1000 21021197 Apr 12  2011 labeledBow.feat
-rw-r--r-- 1 7297 1000 41348699 Apr 12  2011 unsupBow.feat
```

这两个目录是 *pos* 和 *neg*。这些名称将在文本分类任务中被编码为分类变量。

清理子目录并确保所有目录都包含用于分类训练的文本非常重要。如果我们没有删除那个未使用的目录，它的名称将成为一个分类变量，这绝不是我们的意图。那里的其他文件都很好，不会影响这里的结果。再次提醒，目录名称用作标签，因此请确保*只有*用于模型学习和映射到标签的目录。

## 创建数据流水线

现在您的文件已经正确组织，可以开始创建数据流水线了。让我们设置一些变量：

```py
batch_size = 1024
seed = 123
```

批量大小告诉生成器在训练的一个迭代中使用多少样本。还可以分配一个种子，以便每次执行生成器时，它以相同的顺序流式传输文件。如果不分配种子，生成器将以随机顺序输出文件。

然后使用`test_dataset_from_directory`函数定义一个流水线。它将返回一个数据集对象：

```py
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
```

在这种情况下，包含子目录的目录是*aclImdb/train*。此流水线定义用于 80%的训练数据集，由`subset='training'`指定。其他 20%用于交叉验证。

对于交叉验证数据，您将以类似的方式定义流水线：

```py
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)
```

一旦您在上述代码中执行了这两个流水线，这就是预期的输出：

```py
Found 25000 files belonging to 2 classes.
Using 20000 files for training.
Found 25000 files belonging to 2 classes.
Using 5000 files for validation.
```

因为*aclImdb/train*中有两个子目录，生成器将其识别为类。由于 20%的拆分，有 5,000 个文件用于交叉验证。

## 检查数据集

有了生成器，让我们来查看这些文件的内容。检查 TensorFlow 数据集的方法是遍历它并选择一些样本。以下代码片段获取第一批样本，然后随机选择五行电影评论：

```py
import random
idx = random.sample(range(1, batch_size), 5)
for text_batch, label_batch in train_ds.take(1):
  for i in idx:
    print(label_batch[i].numpy(), text_batch.numpy()[i])
```

在这里，`idx`是一个列表，其中包含在`batch_size`范围内生成的五个随机整数。然后，`idx`被用作索引，从数据集中选择文本和标签。

数据集将产生一个元组，包含`text_batch`和`label_batch`；元组在这里很有用，因为它跟踪文本及其标签（类）。这是五个随机选择的文本行和相应的标签：

```py
1 b'Very Slight Spoiler<br /><br /> This movie (despite being….
1 b"Not to mention easily Pierce Brosnan's best performance….
0 b'Bah. Another tired, desultory reworking of an out of copyright…
0 b'All the funny things happening in this sitcom is based on the…
0 b'This is another North East Florida production, filmed mainly…

```

前两个是正面评价（由数字 1 表示），最后三个是负面评价（由 0 表示）。这种方法称为*按类分组*。

## 总结

在本节中，您学习了如何流式传输文本数据集。该方法类似于图像的流式传输，唯一的区别是使用`text_dataset_from_directory`函数。您学习了按类分组以及数据的推荐目录组织方式，这很重要，因为目录名称用作模型训练过程中的标签。在图像和文本分类中，您看到目录名称被用作标签。

# 使用 flow_from_dataframe 方法流式传输图像文件列表

数据的组织方式影响您处理数据摄入流水线的方式。这在处理图像数据时尤为重要。在第四章中的图像分类任务中，您看到不同类型的花卉是如何组织到与每种花卉类型对应的目录中的。

按类分组不是您在现实世界中会遇到的唯一文件组织方法。在另一种常见风格中，如图 5-1 所示，所有图像都被放入一个目录中（这意味着您命名目录的方式并不重要）。

另一种存储图像文件的目录结构

###### 图 5-1。另一种存储图像文件的目录结构

在这个组织中，您会看到与包含所有图像的目录*flowers*在同一级别的位置，有一个名为*all_labels.csv*的 CSV 文件。该文件包含两列：一个包含所有文件名，另一个包含这些文件的标签：

```py
file_name,label
7176723954_e41618edc1_n.jpg,sunflowers
2788276815_8f730bd942.jpg,roses
6103898045_e066cdeedf_n.jpg,dandelion
1441939151_b271408c8d_n.jpg,daisy
2491600761_7e9d6776e8_m.jpg,roses
```

要使用以这种格式存储的图像文件，您需要使用*all_labels.csv*来训练模型以识别每个图像的标签。这就是`flow_from_dataframe`方法的用武之地。

## 下载图像并设置目录

让我们从一个示例开始，其中图像组织在一个单独的目录中。[下载文件](https://oreil.ly/WtKvA) *flower_photos.zip*，解压缩后，您将看到图 5-1 中显示的目录结构：

或者，如果您在 Jupyter Notebook 环境中工作，请运行 Linux 命令`wget`来下载*flower_photos.zip*。以下是 Jupyter Notebook 单元格的命令：

```py
!wget https://data.mendeley.com/public-files/datasets/jxmfrvhpyz/
files/283004ff-e529-4c3c-a1ee-4fb90024dc94/file_downloaded \
--output-document flower_photos.zip
```

前面的命令下载文件并将其放在当前目录中。使用此 Linux 命令解压缩文件：

```py
!unzip -q flower_photos.zip
```

这将创建一个与 ZIP 文件同名的目录：

```py
drwxr-xr-x 3 root root      4096 Nov  9 03:24 flower_photos
-rw-r--r-- 1 root root 228396554 Nov  9 20:14 flower_photos.zip
```

如您所见，有一个名为*flower_photos*的目录。使用以下命令列出其内容，您将看到与图 5-1 中显示的内容完全相同：

```py
!ls -alt flower_photos
```

现在您已经有了目录结构和图像文件，可以开始构建数据流水线，将这些图像馈送到用于训练的图像分类模型中。为了简化操作，您将使用 ResNet 特征向量，这是 TensorFlow Hub 中的一个预构建模型，因此您无需设计模型。您将使用`ImageDataGenerator`将这些图像流式传输到训练过程中。

## 创建数据摄入管道

通常，首先要做的是导入必要的库：

```py
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

请注意，在此示例中您需要 pandas 库。此库用于将标签文件解析为数据框。以下是如何将标签文件读取到 pandas DataFrame 中：

```py
traindf=pd.read_csv('flower_photos/all_labels.csv',dtype=str)
```

如果您查看数据框`traindf`，您将看到以下内容。

|   | **文件名** | **标签** |
| --- | --- | --- |
| 0 | 7176723954_e41618edc1_n.jpg | 向日葵 |
| 1 | 2788276815_8f730bd942.jpg | 玫瑰 |
| 2 | 6103898045_e066cdeedf_n.jpg | 蒲公英 |
| 3 | 1441939151_b271408c8d_n.jpg | 雏菊 |
| 4 | 2491600761_7e9d6776e8_m.jpg | 玫瑰 |
| ... | ... | ... |
| 3615 | 9558628596_722c29ec60_m.jpg | 向日葵 |
| 3616 | 4580206494_9386c81ed8_n.jpg | 郁金香 |

接下来，您需要创建一些变量来保存稍后使用的参数：

```py
data_root = 'flower_photos/flowers'
IMAGE_SIZE = (224, 224)
TRAINING_DATA_DIR = str(data_root)
BATCH_SIZE = 32
```

另外，请记住，当我们使用 ResNet 特征向量时，我们必须将图像像素强度重新缩放到[0, 1]的范围内，这意味着对于每个图像像素，强度必须除以 255。此外，我们需要保留一部分图像用于交叉验证，比如 20%。因此，让我们在一个字典中定义这些标准，我们可以将其用作`ImageDataGenerator`定义的输入：

```py
datagen_kwargs = dict(rescale=1./255, validation_split=.20)
```

另一个字典将保存一些其他参数。ResNet 特征向量期望图像具有 224×224 的像素尺寸，我们还需要指定批处理大小和重采样算法：

```py
dataflow_kwargs = dict(target_size=IMAGE_SIZE, 
batch_size=BATCH_SIZE,
interpolation="bilinear")
```

这个字典将作为数据流定义的输入。

用于训练图像的生成器定义如下：

```py
train_datagen = tf.keras.preprocessing.image.
                ImageDataGenerator(**datagen_kwargs)
```

请注意，我们将`datagen_kwargs`传递给`ImageDataGenerator`实例。接下来，我们使用`flow_from_dataframe`方法创建数据流水线：

```py
train_generator=train_datagen.flow_from_dataframe(
dataframe=traindf,
directory=data_root,
x_col="file_name",
y_col="label",
subset="training",
seed=10,
shuffle=True,
class_mode="categorical",
**dataflow_kwargs)
```

我们定义的`train_datagen`是用来调用`flow_from_dataframe`方法的。让我们看一下输入参数。第一个参数是`dataframe`，被指定为`traindf`。然后`directory`指定了在目录路径中可以找到图像的位置。`x_col`和`y_col`是`traindf`中的标题：`x_col`对应于在*all_labels.csv*中定义的列“file_name”，而`y_col`是列“label”。现在我们的生成器知道如何将图像与它们的标签匹配。

接下来，它指定了一个要进行`training`的子集，因为这是训练图像生成器。提供了种子以便批次的可重现性。图像被洗牌，图像类别被指定为分类。最后，`dataflow_kwargs`被传递到这个`flow_from_dataframe`方法中，以便将原始图像从其原始分辨率重新采样为 224×224 像素。

这个过程对验证图像生成器也是重复的：

```py
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
**datagen_kwargs)
valid_generator=valid_datagen.flow_from_dataframe(
dataframe=traindf,
directory=data_root,
x_col="file_name",
y_col="label",
subset="validation",
seed=10,
shuffle=True,
class_mode="categorical",
**dataflow_kwargs)
```

## 检查数据集

现在，检查 TensorFlow 数据集内容的唯一方法是通过迭代：

```py
image_batch, label_batch = next(iter(train_generator))
fig, axes = plt.subplots(8, 4, figsize=(20, 40))
axes = axes.flatten()
for img, lbl, ax in zip(image_batch, label_batch, axes):
    ax.imshow(img)
    label_ = np.argmax(lbl)
    label = idx_labels[label_]
    ax.set_title(label)
    ax.axis('off')
plt.show()
```

前面的代码片段从`train_generator`中获取了第一批图像，其输出是一个由`image_batch`和`label_batch`组成的元组。

您将看到 32 张图像（这是批处理大小）。有些看起来像图 5-2。

![数据集中一些花的图像](img/t2pr_0502.png)

###### 图 5-2\. 数据集中一些花的图像

现在数据摄入管道已经设置好，您可以在训练过程中使用它了。

## 构建和训练 tf.keras 模型

以下分类模型是如何在 TensorFlow Hub 中使用预构建模型的示例：

```py
mdl = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
                 hub.KerasLayer(
"https://tfhub.dev/tensorflow/resnet_50/feature_vector/1", 
trainable=False),

tf.keras.layers.Dense(5, activation='softmax', 
name = 'custom_class')
])
mdl.build([None, 224, 224, 3])
```

一旦模型架构准备好，就编译它：

```py
mdl.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
  loss=tf.keras.losses.CategoricalCrossentropy(
  from_logits=True, 
  label_smoothing=0.1),
  metrics=['accuracy'])
```

然后启动训练过程：

```py
steps_per_epoch = train_generator.samples // 
train_generator.batch_size
validation_steps = valid_generator.samples // 
valid_generator.batch_size

mdl.fit(
    train_generator,
    epochs=13, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps)
```

请注意，`train_generator`和`valid_generator`被传递到我们的`fit`函数中。这些将在训练过程中生成图像样本，直到所有时代完成。您应该期望看到类似于这样的输出：

```py
Epoch 10/13
90/90 [==============================] - 17s 194ms/step
loss: 1.0338 - accuracy: 0.9602 - val_loss: 1.0779
val_accuracy: 0.9020
Epoch 11/13
90/90 [==============================] - 17s 194ms/step
loss: 1.0311 - accuracy: 0.9623 - val_loss: 1.0750
val_accuracy: 0.9077
Epoch 12/13
90/90 [==============================] - 17s 193ms/step
loss: 1.0289 - accuracy: 0.9672 - val_loss: 1.0741
val_accuracy: 0.9091
Epoch 13/13
90/90 [==============================] - 17s 192ms/step
loss: 1.0266 - accuracy: 0.9693 - val_loss: 1.0728
val_accuracy: 0.9034
```

这表明您已成功将训练图像生成器和验证图像生成器传递到训练过程中，并且两个生成器都可以在训练时摄入数据。验证数据准确性`val_accuracy`的结果表明，我们选择的 ResNet 特征向量对于我们的用于分类花卉图像的用例效果很好。

# 使用 from_tensor_slices 方法流式传输 NumPy 数组

您还可以创建一个流式传输 NumPy 数组的数据管道。您*可以*直接将 NumPy 数组传递到模型训练过程中，但为了有效利用 RAM 和其他系统资源，最好建立一个数据管道。此外，一旦您对模型满意并准备好将其扩展以处理更大量的数据以供生产使用，您将需要一个数据管道。因此，建立一个数据管道是一个好主意，即使是像 NumPy 数组这样简单的数据结构也是如此。

Python 的 NumPy 数组是一种多功能的数据结构。它可以用来表示数值向量和表格数据，也可以用来表示原始图像。在本节中，您将学习如何使用`from_tensor_slices`方法将 NumPy 数据流式传输为数据集。

您将在本节中使用的示例 NumPy 数据是[Fashion-MNIST 数据集](https://oreil.ly/CaUbq)，其中包含 10 种服装类型的灰度图像。这些图像使用 NumPy 结构表示，而不是典型的图像格式，如 JPEG 或 PNG。总共有 70,000 张图像。该数据集在 TensorFlow 的分发中可用，并且可以使用`tf.keras`API 轻松加载。

## 加载示例数据和库

首先，让我们加载必要的库和 Fashion-MNIST 数据：

```py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), 
(test_images, test_labels) = fashion_mnist.load_data()
```

这些数据是使用`tf.keras`API 中的`load_data`函数加载的。数据被分成两个元组。每个元组包含两个 NumPy 数组，图像和标签，如下面的命令所确认的：

```py
print(type(train_images), type(train_labels))

<class 'numpy.ndarray'> <class 'numpy.ndarray'>
```

这确认了数据类型。了解数组维度很重要，您可以使用`shape`命令显示：

```py
print(train_images.shape, train_labels.shape)

(60000, 28, 28) (60000,)
```

正如您所见，`train_images`由 60,000 条记录组成，每条记录都是一个 28×28 的 NumPy 数组，而`train_labels`是一个 60,000 条记录的标签索引。TensorFlow 提供了一个[有用的教程](https://oreil.ly/7d85v)，介绍了这些索引如何映射到类名，但这里是一个快速查看。

| **标签** | **类别** |
| --- | --- |
| 0 | T 恤/上衣 |
| 1 | 裤子 |
| 2 | 套衫 |
| 3 | 连衣裙 |
| 4 | 外套 |
| 5 | 凉鞋 |
| 6 | 衬衫 |
| 7 | 运动鞋 |
| 8 | 包 |
| 9 | 短靴 |

## 检查 NumPy 数组

接下来，检查其中一条记录，看看图像。要将 NumPy 数组显示为颜色刻度，您需要使用之前导入的`matplotlib`库。对象`plt`代表这个库：

```py
plt.figure()
plt.imshow(train_images[5])
plt.colorbar()
plt.grid(False)
plt.show()
```

图 5-3 显示了`train_images[5]`的 NumPy 数组。

![来自 Fashion-MNIST 数据集的示例记录](img/t2pr_0503.png)

###### 图 5-3。来自 Fashion-MNIST 数据集的示例记录

与 JPEG 格式中包含三个独立通道（RGB）的彩色图像不同，Fashion-MNIST 数据集中的每个图像都表示为一个扁平的、二维的 28×28 像素结构。请注意，像素值介于 0 和 255 之间；我们需要将它们归一化为[0, 1]。

## 为 NumPy 数据构建输入管道

现在您已经准备好构建一个流水线。首先，您需要将图像中的每个像素归一化到范围[0, 1]内：

```py
train_images = train_images/255
```

现在数据值是正确的，并且准备传递给`from_tensor_slices`方法：

```py
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, 
train_labels))
```

接下来，将此数据集拆分为训练集和验证集。在以下代码片段中，我指定验证集为 10,000 张图像，剩下的 50,000 张图像进入训练集：

```py
SHUFFLE_BUFFER_SIZE = 10000
TRAIN_BATCH_SIZE = 50
VALIDATION_BATCH_SIZE = 10000

validation_ds = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).
take(VALIDATION_SAMPLE_SIZE).
batch(VALIDATION_BATCH_SIZE)

train_ds = train_dataset.skip(VALIDATION_BATCH_SIZE).
batch(TRAIN_BATCH_SIZE).repeat()
```

当交叉验证是训练过程的一部分时，您还需要定义一些参数，以便模型知道何时停止并在训练迭代期间评估交叉验证数据：

```py
steps_per_epoch = 50000 // TRAIN_BATCH_SIZE
validation_steps = 10000 // VALIDATION_BATCH_SIZE
```

以下是一个小型分类模型：

```py
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(
  from_logits=True),
  metrics=['sparse_categorical_accuracy'])
```

现在您可以开始训练：

```py
model.fit(
    train_ds,
    epochs=13, steps_per_epoch=steps_per_epoch,
    validation_data=validation_ds,
    validation_steps=validation_steps)
```

您的输出应该类似于这样：

```py
…
Epoch 10/13
1562/1562 [==============================] - 4s 3ms/step
loss: 0.2982 - sparse_categorical_accuracy: 0.8931
val_loss: 0.3476 - val_sparse_categorical_accuracy: 0.8778
Epoch 11/13
1562/1562 [==============================] - 4s 3ms/step
loss: 0.2923 - sparse_categorical_accuracy: 0.8954
val_loss: 0.3431 - val_sparse_categorical_accuracy: 0.8831
Epoch 12/13
1562/1562 [==============================] - 4s 3ms/step
loss: 0.2867 - sparse_categorical_accuracy: 0.8990
val_loss: 0.3385 - val_sparse_categorical_accuracy: 0.8854
Epoch 13/13
1562/1562 [==============================] - 4s 3ms/step
loss: 0.2826 - sparse_categorical_accuracy: 0.8997
val_loss: 0.3553 - val_sparse_categorical_accuracy: 0.8811

```

请注意，您可以直接将 train_ds 和 validation_ds 传递给 fit 函数。这正是您在第四章中学到的方法，当时您构建了一个图像生成器并训练了图像分类模型以对五种类型的花进行分类。

# 总结

在本章中，您学习了如何为文本、数值数组和图像构建数据流水线。正如您所见，数据和目录结构在应用不同的 API 将数据摄入模型之前是很重要的。我们从一个文本数据示例开始，使用了 TensorFlow 提供的`text_dataset_from_directory`函数来处理文本文件。您还学到了`flow_from_dataframe`方法是专门为按类别分组的图像文件设计的，这是与您在第四章中看到的完全不同的文件结构。最后，对于 NumPy 数组结构中的数值数组，您学会了使用`from_tensor_slices`方法构建用于流式传输的数据集。当构建数据摄入管道时，您必须了解文件结构以及数据类型，以便使用正确的方法。

现在您已经看到如何构建数据流水线，接下来将在下一章中学习更多关于构建模型的内容。
