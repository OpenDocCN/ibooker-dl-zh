# 第3章。猫与狗：使用Keras中的30行进行迁移学习

想象一下，我们想学习如何演奏口琴，这是一种手持键盘形式的吹奏乐器。如果没有音乐背景，口琴是我们的第一件乐器，可能需要我们几个月的时间才能熟练演奏。相比之下，如果我们已经擅长演奏另一种乐器，比如钢琴，可能只需要几天的时间，因为这两种乐器非常相似。将一个任务的经验应用到另一个类似任务上进行微调是我们在现实生活中经常做的事情（如[图3-1](part0005.html#transfer_learning_in_real_life)所示）。两个任务越相似，将一个任务的经验应用到另一个任务上就越容易。

我们可以将现实生活中的这种现象应用到深度学习的世界中。使用预训练模型开始一个深度学习项目可能会相对快速，因为它重新利用了在训练过程中学到的知识，并将其适应到手头的任务中。这个过程被称为*迁移学习*。

在这一章中，我们使用迁移学习来通过在几分钟内使用Keras训练我们自己的分类器来修改现有模型。到本章结束时，我们将拥有几种工具来创建任何任务的高准确度图像分类器。

![现实生活中的迁移学习](../images/00169.jpeg)

###### 图3-1。现实生活中的迁移学习

# 将预训练模型适应新任务

在讨论迁移学习的过程之前，让我们快速回顾一下深度学习蓬勃发展的主要原因：

+   像ImageNet这样更大更高质量的数据集的可用性

+   更好的计算资源可用；即更快速和更便宜的GPU

+   更好的算法（模型架构、优化器和训练过程）

+   可重复使用的预训练模型，它们经过数月的训练，但可以快速重复使用

最后一点可能是普及深度学习的最重要原因之一。如果每个训练任务都需要一个月的时间，只有少数资金雄厚的研究人员才会在这个领域工作。由于迁移学习，训练模型的被低估的英雄，我们现在可以在几分钟内修改现有模型以适应我们的任务。

例如，我们在[第2章](part0004.html#3Q283-13fa565533764549a6f0ab7f11eed62b)中看到，预训练的ResNet-50模型，它在ImageNet上训练，可以预测猫和狗的品种，以及其他成千上万个类别。因此，如果我们只想在高级别的“猫”和“狗”类别之间进行分类（而不是低级别的品种），我们可以从ResNet-50模型开始，快速重新训练此模型以分类猫和狗。我们只需要在训练期间向其展示包含这两个类别的数据集，这应该需要几分钟到几小时不等。相比之下，如果我们不使用预训练模型来训练猫与狗的模型，可能需要几个小时到几天的时间。

## 对卷积神经网络的浅层探讨

我们一直使用术语“模型”来指代AI中用于做出预测的部分。在计算机视觉的深度学习中，该模型通常是一种称为CNN的特殊类型的神经网络。尽管我们稍后会更详细地探讨CNN，但在这里我们简要地看一下如何通过迁移学习训练它们。

在机器学习中，我们需要将数据转换为一组可识别的特征，然后添加一个分类算法对它们进行分类。CNN也是如此。它们由两部分组成：卷积层和全连接层。卷积层的工作是将图像的大量像素转换为一个更小的表示；即特征。全连接层将这些特征转换为概率。全连接层实际上是一个具有隐藏层的神经网络，正如我们在[第1章](part0003.html#2RHM3-13fa565533764549a6f0ab7f11eed62b)中看到的那样。总之，卷积层充当特征提取器，而全连接层充当分类器。[图3-2](part0005.html#a_high-level_overview_of_a_convolutional)显示了CNN的高级概述。

![卷积神经网络的高级概述](../images/00082.jpeg)

###### 图3-2. CNN的高级概述

想象一下，我们想要检测一个人脸。我们可能想要使用CNN对图像进行分类，并确定其中是否包含人脸。这样的CNN由几个层连接在一起组成。这些层代表数学运算。一个层的输出是下一个层的输入。第一个（或最底层）是输入层，输入图像被馈送到这里。最后一个（或最顶层）是输出层，给出预测。

它的工作方式是将图像馈送到CNN中，并通过一系列层，每个层执行数学运算并将结果传递给下一个层。最终的输出是一个对象类别列表及其概率。例如，类别如球—65%，草—20%，等等。如果图像的输出包含一个“人脸”类别，概率为70%，我们可以得出结论，图像中包含人脸的可能性为70%。

###### 注意

看待CNN的一种直观（和过于简化的）方式是将它们视为一系列滤波器。正如“滤波器”一词所暗示的，每个层都充当信息的筛子，只有在识别到信息时才“通过”。（如果你听说过电子学中的高通和低通滤波器，这可能会很熟悉。）我们说该层对该信息“激活”。每个层对类似猫、狗、汽车等部分的视觉模式被激活。如果一个层没有识别信息（由于训练时学到的内容），其输出接近于零。CNN是深度学习世界的“保安”！

在人脸检测示例中，较低级别的层（[图3-3](part0005.html#left_parenthesisaright_parenthesis_lower) a; 靠近输入图像的层）被“激活”以获取更简单的形状；例如，边缘和曲线。因为这些层仅对基本形状激活，所以它们可以很容易地被重新用于不同于人脸识别的目的，比如检测汽车（毕竟每个图像都由边缘和曲线组成）。中级别的层（[图3-3](part0005.html#left_parenthesisaright_parenthesis_lower) b）被激活以获取更复杂的形状，比如眼睛、鼻子和嘴唇。这些层不像较低级别的层那样容易被重复使用。它们可能不太适用于检测汽车，但可能仍然适用于检测动物。更高级别的层（[图3-3](part0005.html#left_parenthesisaright_parenthesis_lower) c）被激活以获取更复杂的形状，例如大部分人脸。这些层往往更具任务特定性，因此在其他图像分类问题中最不可重复使用。

（a）较低级别的激活，接着是（b）中级别的激活和（c）上层的激活（图片来源：Lee等人的《用于可扩展无监督学习的分层表示的卷积深度信念网络》，ICML 2009）

###### 图3-3. (a) 低层激活，接着是(b) 中层激活和(c) 上层激活（图片来源：Convolutional Deep Belief Networks for Scalable Unsupervised Learning of Hierarchical Representations, Lee et al., ICML 2009）

随着我们接近最后的层，一层可以识别的复杂性和能力增加。相反，随着我们接近输出，一层的可重用性减少。当我们看到这些层学习的内容时，这很快就会变得明显。

## 迁移学习

如果我们想要从一个模型转移知识到另一个模型，我们希望重复使用更多*通用*层（靠近输入）和更少*任务特定*层（靠近输出）。换句话说，我们想要移除最后几层（通常是全连接层），以便我们可以利用更通用的层，并添加针对我们特定分类任务的层。一旦训练开始，通用层（构成我们新模型的大部分）将保持冻结（即，它们是不可修改的），而新添加的任务特定层将被允许修改。这就是迁移学习如何帮助快速训练新模型的方式。[图3-4](part0005.html#an_overview_of_transfer_learning)说明了这个过程，即针对任务X训练的预训练模型如何适应任务Y。

![迁移学习概述](../images/00284.jpeg)

###### 图3-4. 迁移学习概述

## 微调

基本的迁移学习只能带我们走这么远。我们通常在通用层之后只添加两到三个全连接层来构建新的分类器模型。如果我们想要更高的准确性，我们必须允许更多的层被训练。这意味着解冻一些在迁移学习中本来会被冻结的层。这被称为*微调*。[图3-5](part0005.html#fine-tuning_a_convolutional_neural_netwo)展示了一个例子，其中一些接近头部/顶部的卷积层被解冻并针对手头的任务进行训练。

![微调卷积神经网络](../images/00243.jpeg)

###### 图3-5. 微调卷积神经网络

显然，与基本的迁移学习相比，在微调过程中会调整更多的层到我们的数据集中。因为与迁移学习相比，更多的层已经适应了我们的任务，我们可以为我们的任务实现更高的准确性。微调多少层的决定取决于手头的数据量以及目标任务与预训练模型训练的原始数据集的相似性。

我们经常听到数据科学家说，“我微调了模型”，这意味着他们拿了一个预训练模型，移除了任务特定层并添加了新的层，冻结了较低层，然后在新数据集上训练网络的上部分。

###### 注意

在日常用语中，迁移学习和微调是可以互换使用的。在口语中，迁移学习更多地被用作一个概念，而微调则被称为其实施。

## 微调多少

我们应该微调卷积神经网络的多少层？这可以由以下两个因素来指导：

我们有多少数据？

如果我们有几百张标记的图像，从头开始训练和测试一个全新定义的模型（即，定义一个具有随机种子权重的模型架构）将会很困难，因为我们需要更多的数据。用这么少的数据进行训练的危险是这些强大的网络可能会潜在地记住它，导致不良的过拟合（我们将在本章后面探讨）。相反，我们将借用一个预训练的网络并微调最后几层。但如果我们有一百万张标记的图像，微调网络的所有层是可行的，如果必要，可以从头开始训练。因此，任务特定数据的数量决定了我们是否可以微调，以及微调多少。

数据有多相似？

如果任务特定数据与预训练网络使用的数据相似，我们可以调整最后几层。但是，如果我们的任务是在X射线图像中识别不同的骨骼，并且我们想要从ImageNet训练的网络开始，那么常规ImageNet图像和X射线图像之间的高差异将要求几乎所有层都进行训练。

总之，[表3-1](part0005.html#cheatsheet_for_when_and_how_to_fine_tune)提供了一个易于遵循的备忘单。

表3-1。调整的时间和方式的备忘单

|   | **数据集之间相似度高** | **数据集之间相似度低** |
| --- | --- | --- |
| **大量训练数据** | 调整所有层 | 从头开始训练，或者调整所有层 |
| **少量训练数据** | 调整最后几层 | 运气不佳！使用较小的网络进行训练，进行大量数据增强，或以某种方式获取更多数据 |

足够的理论，让我们看看实际操作。

# 在Keras中使用迁移学习构建自定义分类器

正如承诺的那样，现在是用30行或更少行构建我们的最先进分类器的时候了。在高层次上，我们将使用以下步骤：

1.  组织数据。下载标记的猫和狗的图像，然后将图像分成训练和验证文件夹。

1.  构建数据管道。定义一个用于读取数据的管道，包括对图像进行预处理（例如，调整大小）和将多个图像组合成批次。

1.  增加数据。在缺少大量训练图像的情况下，进行小的更改（增强），如旋转、缩放等，以增加训练数据的变化。

1.  定义模型。采用预训练模型，删除最后几个任务特定层，并附加一个新的分类器层。冻结原始层的权重（即，使它们不可修改）。选择一个优化算法和一个要跟踪的指标（如准确性）。

1.  训练和测试。训练几次迭代，直到我们的验证准确率很高。保存模型，以便最终加载到任何应用程序中进行预测。

这一切很快就会变得清晰起来。让我们详细探讨这个过程。

# 组织数据

理解训练、验证和测试数据之间的区别至关重要。让我们看一个学生为标准化考试（例如美国的SAT、中国的高考、印度的JEE、韩国的CSAT等）做准备的现实类比。课堂教学和家庭作业类比于训练过程。学校中的小测验、期中考试和其他测试相当于验证，学生可以经常参加这些测试，评估表现，并在学习计划中做出改进。他们最终是为了在只有一次机会的最终标准化考试中表现最佳。期末考试相当于测试集，学生在这里没有机会改进（忽略重考的能力）。这是他们展示所学内容的唯一机会。

同样，我们的目标是在现实世界中提供最佳预测。为此，我们将数据分为三部分：训练、验证和测试。典型的分布为80%用于训练，10%用于验证，10%用于测试。请注意，我们随机将数据分成这三组，以确保可能潜入的最少*偏见*。模型的最终准确性由*测试集*上的准确性确定，就像学生的分数仅由他们在标准化考试中的表现确定一样。

模型从训练数据中学习，并使用验证集来评估其性能。机器学习从业者将这种性能作为反馈，以找到持续改进模型的机会，类似于学生如何通过小测验改进他们的准备工作。我们可以调整几个旋钮来提高性能；例如，要训练的层数。

在许多研究竞赛中（包括*[Kaggle.com](http://Kaggle.com)*），参赛者会收到一个与用于构建模型的数据分开的测试集。这确保了在报告准确性时竞赛中的一致性。参赛者需要将可用数据划分为训练和验证集。同样，在本书的实验中，我们将继续将数据划分为这两个集合，记住测试数据集仍然是报告现实世界数字的必要条件。

那么为什么要使用验证集呢？有时数据很难获取，为什么不使用所有可用样本进行训练，然后在其上报告准确性呢？当模型开始学习时，它将逐渐在训练数据集上给出更高准确性的预测（称为训练准确性）。但由于它们非常强大，深度神经网络有可能记住训练数据，有时甚至在训练数据上达到100%的准确性。然而，其在现实世界中的表现将非常糟糕。这就好像学生在考试之前就知道会出现的问题一样。这就是为什么验证集，不用于训练模型，可以给出模型性能的真实评估。即使我们可能将10-15%的数据分配为验证集，它将在很大程度上指导我们了解我们的模型到底有多好。

对于训练过程，我们需要将数据集存储在正确的文件夹结构中。我们将图像分为两组：训练和验证。对于图像文件，Keras将根据其父文件夹名称自动分配*类*（类别）的名称。[图3-6](part0005.html#example_directory_structure_of_the_train)展示了重新创建的理想结构。

![不同类别的训练和验证数据的示例目录结构](../images/00204.jpeg)

###### 图3-6\. 不同类别的训练和验证数据的示例目录结构

以下一系列命令可以帮助下载数据并实现这个目录结构：

```py
$ wget https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/ 
download/train.zip 
$ unzip train.zip
$ mv train data
$ cd data
$ mkdir train val
$ mkdir train/cat train/dog
$ mkdir val/cat val/dog
```

数据文件夹中的25,000个文件以“cat”和“dog”为前缀。现在，将文件移动到它们各自的目录中。为了保持我们最初的实验简短，我们每类选择250个随机文件，并将它们放入训练和验证文件夹中。我们可以随时增加/减少这个数字，以尝试在准确性和速度之间取得平衡：

```py
$ ls | grep cat | sort -R | head -250 | xargs -I {} mv {} train/cat/
$ ls | grep dog | sort -R | head -250 | xargs -I {} mv {} train/dog/
$ ls | grep cat | sort -R | head -250 | xargs -I {} mv {} val/cat/
$ ls | grep dog | sort -R | head -250 | xargs -I {} mv {} val/dog/
```

# 构建数据管道

要开始我们的Python程序，我们首先导入必要的包：

```py
import tensorflow as tf
from tf.keras.preprocessing.image import ImageDataGenerator
from tf.keras.models import Model
from tf.keras.layers import Input, Flatten, Dense, Dropout,
GlobalAveragePooling2D
from tf.keras.applications.mobilenet import MobileNet, preprocess_input
import math
```

将以下配置行放在导入语句之后，我们可以根据我们的数据集进行修改：

```py
TRAIN_DATA_DIR = 'data/train_data/'
VALIDATION_DATA_DIR = 'data/val_data/'
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64
```

## 类别数

有两个类别需要区分，我们可以将这个问题视为以下之一：

+   一个二元分类任务

+   多类分类任务

### 二元分类

作为二元分类任务，重要的是要注意，“猫与狗”实际上是“猫与非猫”。狗会被分类为“非猫”，就像桌子或球一样。对于给定的图像，模型将给出一个与“cat”类对应的单个概率值，因此“非猫”的概率为1 - *P(cat)*。如果概率高于0.5，我们将预测为“cat”；否则为“非猫”。为了简化问题，我们假设测试集中只包含猫或狗的图像。因为“猫与非猫”是一个二元分类任务，我们将类别数设置为1；即“cat”。任何无法分类为“cat”的内容将被分类为“非猫”。

###### 提示

Keras按照文件夹名称的字母顺序处理输入数据。因为按字母顺序，“cat”在“dog”之前，我们的第一个预测类别是“cat”。对于多类任务，我们可以应用相同的概念，并根据文件夹排序顺序推断每个类别标识符（索引）。请注意，类别索引从第一个类别开始为0。

### 多类分类

在一个假设的世界中，只有猫和狗，没有其他东西，一个“非猫”总是狗。因此，“非猫”标签可以简单地替换为“狗”标签。然而，在现实世界中，我们有超过两种类型的物体。如前所述，球或沙发也会被分类为“狗”，这是不正确的。因此，在真实场景中，将其视为多类别分类任务而不是二元分类任务要更有用。作为多类别分类任务，我们为每个类别预测单独的概率值，最高的概率值是我们的赢家。在“猫与狗”案例中，我们将类别数设置为两。为了使我们的代码可重用于未来的任务，我们将把这视为多类别任务。

## 批量大小

在高层次上，训练过程包括以下步骤：

1.  对图像进行预测（*前向传递*）。

1.  确定哪些预测是错误的，并将预测与真实值之间的差异传播回去（*反向传播*）。

1.  反复进行，直到预测变得足够准确。

初始迭代很可能准确率接近0%。然而，重复这个过程几次可能会产生一个高度准确的模型（>90%）。

批量大小定义了模型一次看到多少张图片。重要的是，每个批次中有来自不同类别的各种图片，以防止在迭代之间的准确度指标出现大幅波动。为此，需要一个足够大的批量大小。然而，重要的是不要设置批量大小过大；太大的批量可能无法适应GPU内存，导致“内存不足”崩溃。通常，批量大小设置为2的幂。对于大多数问题，一个好的起点是64，我们可以通过增加或减少数量来调整。

# 数据增强

通常，当我们听到深度学习时，我们会将其与数百万张图片联系在一起。因此，我们拥有的500张图片可能对于真实世界的训练来说是一个较低的数字。虽然这些深度神经网络非常强大，但对于少量数据来说可能过于强大，训练图像集数量有限的危险在于神经网络可能会记住我们的训练数据，并在训练集上表现出色，但在验证集上准确度较低。换句话说，模型已经过度训练，无法泛化到以前未见过的图像。我们绝对不希望出现这种情况。

###### 提示

通常，当我们尝试在少量数据上训练神经网络时，结果是模型在训练数据上表现非常好，但在之前未见过的数据上做出相当糟糕的预测。这样的模型将被描述为*过度拟合*模型，问题本身被称为*过度拟合*。

[图3-7](part0005.html#underfittingcomma_overfittingcomma_and_i)说明了这种现象，即接近正弦曲线的点的分布（几乎没有噪音）。点代表我们的网络可见的训练数据，叉代表在训练期间未见过的测试数据。在一个极端情况下（欠拟合），一个简单的模型，如线性预测器，将无法很好地表示基础分布，导致训练数据和测试数据上的高错误率。在另一个极端情况下（过拟合），一个强大的模型（如深度神经网络）可能有能力记住训练数据，这将导致训练数据上的错误率非常低，但在测试数据上仍然有很高的错误率。我们希望的是在训练错误和测试错误都相对较低的愉快中间位置，这理想情况下确保我们的模型在真实世界中的表现与训练期间一样好。

![对于接近正弦曲线的点的欠拟合、过拟合和理想拟合](../images/00170.jpeg)

###### 图3-7。欠拟合、过拟合和理想拟合对于接近正弦曲线的点

伟大的力量伴随着伟大的责任。我们有责任确保我们强大的深度神经网络不会在我们的数据上过拟合。当我们有少量训练数据时，过拟合是常见的。我们可以通过几种不同的方式减少这种可能性：

+   以某种方式获取更多数据

+   大幅度增强现有数据

+   微调更少的层

通常存在数据不足的情况。也许我们正在处理一个小众问题，数据很难获得。但我们可以通过一些方式人为地增加我们的分类数据集：

旋转

在我们的例子中，我们可能希望随机将这500张图像旋转20度，向任一方向，从而产生多达20,000个可能的独特图像。

随机移动

将图像稍微向左或向右移动。

缩放

稍微放大或缩小图像。

通过结合旋转、移动和缩放，程序可以生成几乎无限数量的独特图像。这一重要步骤称为*数据增强*。数据增强不仅有助于添加更多数据，还有助于为真实场景训练更健壮的模型。例如，并非所有图像都将猫正确地放在中间或以完美的0度角。Keras提供了`ImageDataGenerator`函数，该函数在从目录加载数据时增强数据。为了说明图像增强的效果，[图3-8](part0005.html#possible_image_augmentations_generated_f)展示了由[imgaug](https://oreil.ly/KYA9O)库为一个示例图像生成的增强示例。（请注意，我们实际训练时不会使用imgaug。）

![从单个图像生成的可能的图像增强](../images/00123.jpeg)

###### 图3-8。从单个图像生成的可能的图像增强

彩色图像通常有三个通道：红色、绿色和蓝色。每个通道的强度值范围从0到255。为了将其归一化（即将值缩小到0到1之间），我们使用`preprocess_input`函数（其中，除了其他操作外，还将每个像素除以255）：

```py
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
```

###### 提示

有时，知道训练图像的标签可以帮助确定适当的增强方式。例如，在训练数字识别器时，您可能会同意对数字“8”的图像进行垂直翻转增强，但不适用于“6”和“9”。

与我们的训练集不同，我们不希望增强我们的验证集。原因是，使用动态增强，验证集将在每次迭代中保持变化，导致的准确度指标将不一致且难以在其他迭代中进行比较。

现在是从目录加载数据的时候了。一次训练一张图像可能效率不高，所以我们可以将它们分成组。为了在训练过程中引入更多的随机性，我们将在每个批次中保持图像的随机顺序。为了在同一程序的多次运行中保持可重现性，我们将为随机数生成器提供一个种子值：

```py
train_generator = train_datagen.flow_from_directory(
                        TRAIN_DATA_DIR,
                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        seed=12345,
                        class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
                        VALIDATION_DATA_DIR,
                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        class_mode='categorical')
```

# 模型定义

现在数据已经处理好了，我们来到我们训练过程中最关键的组件：模型。在接下来的代码中，我们重复使用了之前在ImageNet数据集上训练过的CNN（在我们的案例中是MobileNet），丢弃了最后几层，称为全连接层（即ImageNet特定的分类器层），并用适合当前任务的自己的分类器替换它们。

对于迁移学习，我们“冻结”原始模型的权重；也就是说，将这些层设置为不可修改，因此只有新分类器的层（我们将添加的）可以被修改。我们在这里使用MobileNet来保持速度，但这种方法对于任何神经网络都同样有效。以下几行包括一些术语，如`Dense`、`Dropout`等。虽然我们在本章不会探讨它们，但你可以在[附录A](part0021.html#K0RQ3-13fa565533764549a6f0ab7f11eed62b)中找到解释。

```py
def model_maker():
    base_model = MobileNet(include_top=False, input_shape =
(IMG_WIDTH,IMG_HEIGHT,3))
    for layer in base_model.layers[:]:
        layer.trainable = False *`# Freeze the layers`*
    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=predictions)
```

# 训练模型

## 设置训练参数

数据和模型都准备好了，我们唯一需要做的就是训练模型。这也被称为*将模型拟合到数据*。为了训练一个模型，我们需要选择和修改一些不同的训练参数。

损失函数

`loss`函数是我们在训练过程中对模型进行惩罚的方式。我们希望*最小化*这个函数的值。例如，在预测房价的任务中，`loss`函数可以是均方根误差。

优化器

这是一个帮助最小化`loss`函数的算法。我们使用Adam，这是目前最快的优化器之一。

学习率

学习是渐进的。学习率告诉优化器向解决方案迈出多大的步伐；换句话说，最小化损失的位置。迈出太大的步伐，我们最终会过度摆动并超过目标。迈出太小的步伐，可能需要很长时间才能最终到达目标损失值。设置一个最佳学习率是很重要的，以确保我们在合理的时间内达到学习目标。在我们的例子中，我们将学习率设置为0.001。

度量

选择一个度量标准来评估训练模型的性能。准确率是一个很好的可解释度量标准，特别是当类别不平衡时（即每个类别的数据量大致相等）。请注意，这个度量标准与`loss`函数无关，主要用于报告，而不是作为模型的反馈。

在下面的代码片段中，我们使用之前编写的`model_maker`函数创建自定义模型。我们使用这里描述的参数进一步定制这个模型，以适应我们的猫狗任务：

```py
model = model_maker()
model.compile(loss='categorical_crossentropy',
              optimizer= tf.train.Adam(lr=0.001),
              metrics=['acc'])
num_steps = math.ceil(float(TRAIN_SAMPLES)/BATCH_SIZE)              
model.fit_generator(train_generator,
                    steps_per_epoch = num_steps,
                    epochs=10,
                    validation_data = validation_generator,
                    validation_steps = num_steps)
```

###### 注意

你可能已经注意到了前面代码中的时代这个术语。一个时代代表了一个完整的训练步骤，网络已经遍历整个数据集。一个时代可能包含几个小批次。

## 开始训练

运行这个程序，让魔法开始吧。如果你没有GPU，在等待时可以煮杯咖啡——可能需要5到10分钟。或者为什么要等待，当你可以在Colab上免费使用GPU运行本章的笔记本时呢？

完成后，请注意有四个统计数据：训练数据和验证数据上的`loss`和`acc`。我们期待`val_acc:`

```py
> Epoch 1/100 7/7 [====] - 5s - 
loss: 0.6888 - acc: 0.6756 - val_loss: 0.2786 - val_acc: 0.9018
> Epoch 2/100 7/7 [====] - 5s - 
loss: 0.2915 - acc: 0.9019 - val_loss: 0.2022 - val_acc: 0.9220
> Epoch 3/100 7/7 [====] - 4s - 
loss: 0.1851 - acc: 0.9158 - val_loss: 0.1356 - val_acc: 0.9427
> Epoch 4/100 7/7 [====] - 4s - 
loss: 0.1509 - acc: 0.9341 - val_loss: 0.1451 - val_acc: 0.9404
> Epoch 5/100 7/7 [====] - 4s - 
loss: 0.1455 - acc: 0.9464 - val_loss: 0.1637 - val_acc: 0.9381
> Epoch 6/100 7/7 [====] - 4s - 
loss: 0.1366 - acc: 0.9431 - val_loss: 0.2319 - val_acc: 0.9151
> Epoch 7/100 7/7 [====] - 4s - 
loss: 0.0983 - acc: 0.9606 - val_loss: 0.1420 - val_acc: 0.9495
> Epoch 8/100 7/7 [====] - 4s - 
loss: 0.0841 - acc: 0.9731 - val_loss: 0.1423 - val_acc: 0.9518
> Epoch 9/100 7/7 [====] - 4s - 
loss: 0.0714 - acc: 0.9839 - val_loss: 0.1564 - val_acc: 0.9509
> Epoch 10/100 7/7 [====] - 5s - 
loss: 0.0848 - acc: 0.9677 - val_loss: 0.0882 - val_acc: 0.9702
```

在第一个时代仅用了5秒就在验证集上达到了90%的准确率，仅用了500张训练图片。不错！到第10步时，我们观察到约97%的*验证准确率*。这就是迁移学习的力量。

让我们花点时间欣赏这里发生的事情。仅仅用500张图片，我们就能在几秒钟内达到高水平的准确性，而且代码量很少。相比之下，如果我们没有一个在ImageNet上预先训练过的模型，要获得一个准确的模型可能需要几个小时到几天的训练时间，以及更多的数据。

这就是我们需要在任何问题上训练一流分类器的所有代码。将数据放入以类名命名的文件夹中，并更改配置变量中的相应值。如果我们的任务有两个以上的类别，我们应该使用`categorical_crossentropy`作为`loss`函数，并将最后一层的`activation`函数替换为`softmax`。[表3-2](part0005.html#deciding_the_loss_and_activation_type_ba)说明了这一点。

表3-2\. 根据任务决定损失和激活类型

| **分类类型** | **类模式** | **损失** | **最后一层的激活** |
| --- | --- | --- | --- |
| 1或2个类 | 二进制 | 二元交叉熵 | sigmoid |
| 多类别，单标签 | 分类 | 分类交叉熵 | softmax |
| 多类别，多标签 | 分类 | 二元交叉熵 | sigmoid |

在我们忘记之前，保存刚刚训练的模型，以便稍后使用：

```py
model.save('model.h5')
```

# 测试模型

现在我们有了一个经过训练的模型，我们可能最终想要稍后在我们的应用程序中使用它。我们现在可以随时加载这个模型并对图像进行分类。`load_model`，顾名思义，加载模型：

```py
from tf.keras.models import load_model
model = load_model('model.h5')
```

现在让我们尝试加载我们的原始样本图像，看看我们得到什么结果：

```py
img_path = '../../sample_images/dog.jpg'
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array) *`# Preprocess the image`*
prediction = model.predict(preprocessed_img)
print(prediction)
print(validation_generator.class_indices)
[[0.9967706]]
{'dog': 1, 'cat': 0}
```

打印概率值，我们看到它是0.996。这是给定图像属于类别“1”（狗）的概率。因为概率大于0.5，所以图像被预测为狗。

这就是我们训练自己分类器所需要的全部内容。在本书中，您可以期望在进行最少修改的情况下重复使用此代码进行训练。您也可以在自己的项目中重复使用此代码。尝试调整时代和图像的数量，并观察它如何影响准确性。此外，我们应该尝试使用我们可以在网上找到的任何其他数据。没有比这更容易的了！

# 分析结果

通过我们训练过的模型，我们可以分析它在验证数据集上的表现。除了更直接的准确性指标之外，查看误判的实际图像应该能让我们直观地了解这个例子是否真正具有挑战性，或者我们的模型还不够复杂。

有三个问题我们想要为每个类别（猫、狗）回答：

+   哪些图像我们最有信心是猫/狗？

+   哪些图像我们最不确定是猫/狗？

+   哪些图像尽管非常自信，但预测错误？

在进行这之前，让我们对整个验证数据集进行预测。首先，我们正确设置管道配置：

```py
*`# VARIABLES`*
IMG_WIDTH, IMG_HEIGHT = 224, 224
VALIDATION_DATA_DIR = 'data/val_data/'
VALIDATION_BATCH_SIZE = 64

*`# DATA GENERATORS`*
validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        class_mode='categorical')
ground_truth = validation_generator.classes
```

然后，我们进行预测：

```py
predictions = model.predict_generator(validation_generator)
```

为了使我们的分析更容易，我们创建一个字典，存储每个图像的索引到预测和实际值（预期预测）：

```py
*`# prediction_table is a dict with index, prediction, ground truth`*
prediction_table = {}
for index, val in enumerate(predictions):
    *`# get argmax index`*
    index_of_highest_probability = np.argmax(val)
    value_of_highest_probability = val[index_of_highest_probability]
    prediction_table[index] = [value_of_highest_probability,
index_of_highest_probability, ground_truth[index]]
assert len(predictions) == len(ground_truth) == len(prediction_table)
```

对于接下来的两个代码块，我们提供了常用的样板代码，我们在整本书中经常重复使用。

以下是我们将使用的辅助函数的签名，用于查找具有给定类别的最高/最低概率值的图像。此外，我们将使用另一个辅助函数，- `display`()，将图像以网格形式输出到屏幕上：

```py
def display(sorted_indices, message):
    similar_image_paths = []
    distances = []
    for name, value in sorted_indices:
        [probability, predicted_index, gt] = value
        similar_image_paths.append(VALIDATION_DATA_DIR + fnames[name])
        distances.append(probability)
    plot_images(similar_image_paths, distances, message)
```

此函数在本书的Github网站上定义（请参阅[*http://PracticalDeepLearning.ai*](http://PracticalDeepLearning.ai)，位于*code/chapter-3*）。

现在开始有趣的部分！哪些图像我们最有信心包含狗？让我们找到预测概率最高的图像（即最接近1.0；参见[图3-9](part0005.html#images_with_the_highest_probability_of_c)中概率最高的图像）与预测类别狗（即1）：

```py
*`# Most confident predictions of 'dog'`*
indices = get_images_with_sorted_probabilities(prediction_table,
get_highest_probability=True, label=1, number_of_items=10,
only_false_predictions=False)
message = 'Images with the highest probability of containing dogs'
display(indices[:10], message)
```

![概率最高的包含狗的图像](../images/00083.jpeg)

###### 图3-9\. 概率最高的包含狗的图像

这些图像确实非常像狗。概率如此之高的原因之一可能是图像中包含了多只狗，以及清晰、明确的视图。现在让我们尝试找出我们最不确定包含狗的图像（请参见[图3-10](part0005.html#images_with_the_lowest_probability_of_co)）：

```py
*`# Least confident predictions of 'dog'`*
indices = get_images_with_sorted_probabilities(prediction_table,
get_highest_probability=False, label=1, number_of_items=10,
only_false_predictions=False)
message = 'Images with the lowest probability of containing dogs'
display(indices[:10], message)
```

![概率最低的包含狗的图像](../images/00040.jpeg)

###### 图3-10\. 概率最低的包含狗的图像

重申一下，这些是我们的分类器最不确定包含狗的图像。这些预测大多处于临界点（即0.5概率）成为主要预测。请记住，作为猫的概率只是略小一些，大约在0.49左右。与之前一组图像相比，这些图像中出现的动物通常更小，更不清晰。这些图像通常导致错误预测——只有10张图像中的2张被正确预测。在这里做得更好的一种可能方法是使用更大的图像集进行训练。

如果您担心这些错误分类，不用担心。提高分类准确性的一个简单技巧是对接受分类器结果的阈值设定更高，比如0.75。如果分类器对图像类别不确定，其结果将被保留。在[第5章](part0007.html#6LJU3-13fa565533764549a6f0ab7f11eed62b)中，我们将看看如何找到最佳阈值。

说到错误预测，当分类器信心较低时（即两类问题的概率接近0.5时），显然会出现错误预测。但我们不希望的是在我们的分类器对其预测非常确信时出现错误预测。让我们看看分类器确信包含狗的图像，尽管它们实际上是猫（参见[图3-11](part0005.html#images_of_cats_with_the_highest_probabil)）：

```py
*`# Incorrect predictions of 'dog'`*
indices = get_images_with_sorted_probabilities(prediction_table,
get_highest_probability=True, label=1, number_of_items=10,
only_false_predictions=True)
message = 'Images of cats with the highest probability of containing dogs'
display(indices[:10], message)
```

![具有最高概率包含狗的猫的图像](../images/00001.jpeg)

###### 图3-11。具有最高概率包含狗的猫的图像

嗯...结果是这些图像中有一半包含猫和狗，我们的分类器正确地预测了狗类别，因为在这些图像中它们的大小更大。因此，这里不是分类器有问题，而是数据有问题。这在大型数据集中经常发生。另一半通常包含不清晰和相对较小的对象（但理想情况下，我们希望对这些难以识别的图像具有较低的置信度）。

对猫类重复相同一组问题，哪些图像更像猫（参见[图3-12](part0005.html#images_with_the_highest_probab_c-id00002)）？

```py
*`# Most confident predictions of 'cat'`*
indices = get_images_with_sorted_probabilities(prediction_table,
get_highest_probability=True, label=0, number_of_items=10,
only_false_predictions=False)
message = 'Images with the highest probability of containing cats'
display(indices[:10], message)
```

![具有最高概率包含猫的图像](../images/00288.jpeg)

###### 图3-12。具有最高概率包含猫的图像

有趣的是，其中许多图像中有多只猫。这证实了我们之前的假设，即多个清晰、明确的猫的视图可以给出更高的概率。另一方面，哪些图像我们对包含猫最不确定（参见[图3-13](part0005.html#images_with_the_lowest_probabili-id00001)）？

```py
*`# Least confident predictions of 'cat'`*
indices = get_images_with_sorted_probabilities(prediction_table,
get_highest_probability=False, label=0, number_of_items=10,
only_false_predictions=False)
message = 'Images with the lowest probability of containing cats'
display(indices[:10], message)
```

![具有最低概率包含猫的图像](../images/00248.jpeg)

###### 图3-13。具有最低概率包含猫的图像

正如之前所见，关键对象的大小较小，有些图像相当不清晰，这意味着在某些情况下对比度太高，或者对象太亮，这与大多数训练图像不符。例如，第八张（dog.6680）和第十张（dog.1625）图像中的相机闪光灯使得狗难以识别。第六张图像包含一只狗站在同色的沙发前。两张图像包含笼子。

最后，我们的分类器错误地确信包含猫的图像是哪些（参见[图3-14](part0005.html#images_of_dogs_with_highest_probability)）？

```py
*`# Incorrect predictions of 'cat'`*
indices = get_images_with_sorted_probabilities(prediction_table,
get_highest_probability=True, label=0, number_of_items=10,
only_false_predictions=True)
message = 'Images of dogs with the highest probability of containing cats'
display(indices[:10], message)
```

![具有最高概率包含猫的狗的图像](../images/00138.jpeg)

###### 图3-14。具有最高概率包含猫的狗的图像

这些错误预测是我们想要减少的。其中一些显然是错误的，而另一些则是令人困惑的图像。在[图3-14](part0005.html#images_of_dogs_with_highest_probability)中的第六张图（dog.4334）似乎被错误标记为狗。第七和第十张图在背景上难以区分。第一和第十张图内部缺乏足够的纹理，无法给分类器足够的识别能力。而一些狗太小，比如第二和第四张。

通过各种分析，我们可以总结出，错误预测可能是由于低照明、不清晰、难以区分的背景、缺乏纹理以及相对于图像的较小占用区域而引起的。

分析我们的预测是了解我们的模型学到了什么以及它擅长什么的好方法，并突出了增强其预测能力的机会。增加训练示例的大小和更强大的数据增强将有助于提高分类的准确性。还要注意，向我们的模型展示真实世界的图像（看起来类似于我们的应用程序将要使用的场景）将极大地提高其准确性。在[第5章](part0007.html#6LJU3-13fa565533764549a6f0ab7f11eed62b)中，我们使分类器更加健壮。

# 进一步阅读

为了更好地理解神经网络和CNN，[我们的网站](http://PracticalDeepLearning.ai)提供了一个学习指南，其中包括推荐的资源，如视频讲座、博客，以及更有趣的是，交互式可视化工具，让您可以在浏览器中玩不同的场景，而无需安装任何软件包。如果您是深度学习的初学者，我们强烈推荐这个指南，以加强您的基础知识。它涵盖了您需要建立直觉以解决未来问题的理论。我们使用Google的TensorFlow Playground（[图3-15](part0005.html#building_a_neural_network_in_tensorflow)）进行神经网络和Andrej Karpathy的ConvNetJS（[图3-16](part0005.html#defining_a_cnn_and_visualizing_output_of)）进行CNN。

![在TensorFlow Playground中构建神经网络](../images/00238.jpeg)

###### 图3-15. 在TensorFlow Playground中构建神经网络

![定义CNN并在ConvNetJS中训练期间可视化每个层的输出。](../images/00130.jpeg)

###### 图3-16. 定义CNN并在ConvNetJS中训练期间可视化每个层的输出

此外，我们在[附录A](part0021.html#K0RQ3-13fa565533764549a6f0ab7f11eed62b)中还有一个简短的指南，总结了卷积神经网络，作为一个方便的参考。

# 总结

在本章中，我们介绍了迁移学习的概念。我们重用了一个预训练模型，在不到30行代码和几乎500张图片的情况下构建了自己的猫狗分类器，在几分钟内达到了最先进的准确性。通过编写这段代码，我们也揭穿了一个神话，即我们需要数百万张图片和强大的GPU来训练我们的分类器（尽管它们有帮助）。

希望通过这些技能，您可能最终能够回答一个古老的问题，即谁放出了狗。

在接下来的几章中，我们将利用这些知识更深入地理解CNN，并将模型准确性提升到更高水平。
