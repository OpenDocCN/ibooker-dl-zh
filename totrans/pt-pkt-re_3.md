# 第三章. 使用 PyTorch 进行深度学习开发

现在您的开发环境已经运行，并且对张量及其操作有了很好的理解，我们可以开始使用 PyTorch 开发和部署深度学习模型。本章提供了基本 NN 开发过程和执行所需的 PyTorch 代码的快速参考。

首先我们将回顾整体过程，然后深入每个阶段，查看一些实现每个功能的示例 PyTorch 代码。我们将在第二章学到的基础上，将数据加载到张量中，并应用数据转换，将张量转换为模型的合适输入。

您将构建一个深度学习模型，并使用常见的训练循环结构对模型进行训练。然后，您将测试模型的性能，并调整超参数以改善结果和训练速度。最后，我们将探讨将模型部署到原型系统或生产环境的方法。在每个阶段，我将提供常用的 PyTorch 代码作为您开发自己的深度学习模型的参考。

本书的未来章节将提供更多示例，并涵盖更高级的主题，如定制、优化、加速、分布式训练和高级部署。现在，我们将专注于基本 NN 开发过程。

# 整体过程

尽管每个人构建深度学习模型的方式都不同，但整个过程基本上是相同的。无论您是使用带标签数据进行监督学习，使用无标签数据进行无监督学习，还是使用两者混合的半监督学习，都会使用基本的流程来训练、测试和部署您的深度学习模型。我假设您对深度学习模型开发有一定了解，但在开始之前，让我们回顾一下基本的深度学习训练过程。然后我将展示如何在 PyTorch 中实现这个过程。

图 3-1 展示了深度学习开发中最常见的任务。第一阶段是数据准备阶段，在这个阶段，我们将从外部来源加载数据，并将其转换为适合模型训练的格式。这些数据可以是图像、视频、语音录音、音频文件、文本、一般的表格数据，或者它们的任意组合。

首先，我们加载这些数据，并将其转换为张量形式的数值。这些张量将在模型训练阶段作为输入；然而，在传入之前，这些张量通常会通过转换进行预处理，并分组成批次以提高训练性能。因此，数据准备阶段将通用数据转换为可以传入 NN 模型的张量批次。

接下来，在模型实验和开发阶段，我们将设计一个 NN 模型，使用训练数据训练模型，测试其性能，并优化我们的超参数以提高性能到期望水平。为此，我们将将数据集分为三部分：一部分用于训练，一部分用于验证，一部分用于测试。我们将设计一个 NN 模型，并使用训练数据训练其参数。PyTorch 在`torch.nn`模块中提供了优雅设计的模块和类，帮助您创建和训练您的 NN。我们将从众多内置的 PyTorch 函数中定义损失函数和优化器。然后，我们将执行反向传播，并在训练循环中更新模型参数。

![“基本深度学习开发过程”](img/ptpr_0301.PNG)

###### 图 3-1. 基本深度学习开发过程

在每个 epoch 内，我们还将通过传入验证数据来验证我们的模型，衡量性能，并可能调整超参数。最后，我们将通过传入测试数据来测试我们的模型，并根据未知数据的性能来衡量模型的表现。在实践中，验证和测试循环可能是可选的，但我们在这里展示它们以确保完整性。

深度学习模型开发的最后阶段是模型部署阶段。在这个阶段，我们有一个完全训练好的模型——那么我们该怎么办呢？如果您是进行实验的深度学习研究科学家，您可能只想将模型保存到文件中，以便进一步研究和实验，或者您可能希望通过 PyTorch Hub 等存储库提供对其的访问。您还可以将其部署到边缘设备或本地服务器，以演示原型或概念验证。

另一方面，如果您是软件开发人员或系统工程师，您可能希望将模型部署到产品或服务中。在这种情况下，您可以将模型部署到云服务器上的生产环境，或将其部署到边缘设备或手机上。在部署经过训练的模型时，模型通常需要额外的后处理。例如，您可能要对一批图像进行分类，但只想报告最有信心的结果。模型部署阶段还处理从模型的输出值到最终解决方案所需的任何后处理。

现在我们已经探讨了整个开发过程，让我们深入每个部分，展示 PyTorch 如何帮助您开发深度学习模型。

# 数据准备

深度学习开发的第一阶段始于数据准备。在这个阶段，我们获取数据来训练和测试我们的 NN 模型，并将其转换为数字张量，以便我们的 PyTorch 模型可以处理。数据集的大小和数据本身对于开发良好的模型很重要；然而，生成良好的数据集超出了本书的范围。

在本节中，我将假设您已经确定数据是好的，因此我将重点介绍如何使用 PyTorch 的内置功能加载数据、应用转换并对数据进行批处理。首先我将展示如何使用`torchvision`包准备图像数据，然后我们将探索 PyTorch 资源以准备其他类型的数据。

## 数据加载

PyTorch 提供了强大的内置类和实用程序，如`Dataset`、`DataLoader`和`Sampler`类，用于加载各种类型的数据。`Dataset`类定义了如何从文件或数据源访问和预处理数据。`Sampler`类定义了如何从数据集中采样数据以创建批次，而`DataLoader`类将数据集与采样器结合在一起，允许您迭代一组批次。

PyTorch 库如 Torchvision 和 Torchtext 还提供支持专门数据的类，如计算机视觉和自然语言数据。`torchvision.datasets`模块是如何利用内置类加载数据的一个很好的例子。`torchvision.datasets`模块提供了许多子类来从流行的学术数据集加载图像数据。

其中一个流行的数据集是 CIFAR-10。CIFAR-10 数据集是由 Alex Krizhevsky、Vinod Nair 和 Geoffrey Hinton 在为加拿大高级研究所（CIFAR）进行研究时收集的。它包含 50,000 个训练图像和 10,000 个测试图像，涵盖了 10 种可能的对象：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。以下代码展示了如何使用 CIFAR-10 创建一个训练数据集：

```py
from torchvision.datasets import CIFAR10

train_data = CIFAR10(root="./train/",
                     train=True,
                     download=True)
```

`train`参数确定我们加载训练数据还是测试数据，将`download`设置为`True`将为我们下载数据（如果我们还没有）。

让我们探索`train_data`数据集对象。我们可以使用其方法和属性访问有关数据集的信息，如下面的代码所示：

```py

print(train_data)![1](img/1.png)# out:# Dataset CIFAR10#     Number of datapoints: 50000#     Root location: ./train/#     Split: Trainprint(len(train_data))![2](img/2.png)# out: 50000print(train_data.data.shape)# ndarray ![3](img/3.png)# out: (50000, 32, 32, 3)print(train_data.targets)# list ![4](img/4.png)# out: [6, 9, ...,  1, 1]print(train_data.classes)![5](img/5.png)# out: ['airplane', 'automobile', 'bird',#       'cat', 'deer', 'dog', 'frog',#       'horse', 'ship', 'truck']print(train_data.class_to_idx)![6](img/6.png)# out:# {'airplane': 0, 'automobile': 1, 'bird': 2,#  'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,#  'horse': 7, 'ship': 8, 'truck': 9}
```

![1](img/#comarker11)

打印对象会返回其一般信息。

![2](img/#comarker22)

使用`len()`检查数据样本的数量。

![3](img/#comarker33)

数据是一个包含 50,000 个 32×32 像素彩色图像的 NumPy 数组。

![4](img/#comarker44)

目标是一个包含 50,000 个数据标签的列表。

![5](img/#comarker55)

你可以使用`classes`将数值标签映射到类名。

![6](img/#comarker66)

你可以使用`class_to_idx`将类名映射到索引值。

让我们仔细看看`train_data`数据集的数据和标签。我们可以使用索引访问数据样本，如下面的代码所示：

```py
print(type(train_data[0]))
# out: <class 'tuple'>

print(len(train_data[0]))
# out: 2

data, label = train_data[0]
```

如代码中所示，`train_data[0]`返回一个包含两个元素的元组——数据和标签。让我们先检查数据：

```py
print(type(data))
# out: <class 'PIL.Image.Image'>

print(data)
# out:
# <PIL.Image.Image image mode=RGB
#       size=32x32 at 0x7FA61-D6F1748>
```

数据由一个 PIL 图像对象组成。PIL 是一种常见的图像格式，使用 Pillow 库以高度×宽度×通道的格式存储图像像素值。彩色图像有三个通道（RGB）分别为红色、绿色和蓝色。了解数据格式很重要，因为如果模型期望不同的格式，我们可能需要转换这种格式（稍后会详细介绍）。

图 3-2 显示了 PIL 图像。由于分辨率只有 32×32，所以有点模糊，但你能猜出是什么吗？

![“示例图像”](img/ptpr_0302.png)

###### 图 3-2\. 示例图像

让我们检查标签：

```py
print(type(label))
# out: <class 'int'>

print(label)
# out: 6

print(train_data.classes[label])
# out: frog
```

在代码中，`label`是一个表示图像类别的整数值（例如，飞机、狗等）。我们可以使用`classes`属性查看索引 6 对应于青蛙。

我们还可以将测试数据加载到另一个名为`test_data`的数据集对象中。更改根文件夹并将`train`标志设置为`False`即可，如下面的代码所示：

```py
test_data = CIFAR10(root="./test/",
                    train=False,
                    download=True)

print(test_data)
# out:
# Dataset CIFAR10
#     Number of datapoints: 10000
#     Root location: ./test/
#     Split: Test

print(len(test_data))
# out: 10000

print(test_data.data.shape) # ndarray
# out: (10000, 32, 32, 3)
```

`test_data`数据集与`train_data`数据集类似。但是测试数据集中只有 10,000 张图像。尝试访问数据集类的一些方法和`test_data`数据集上的属性。

## 数据转换

在数据加载步骤中，我们从数据源中提取数据并创建包含有关数据集和数据本身信息的数据集对象。但是，在将数据传递到 NN 模型进行训练和测试之前，数据可能需要进行调整。例如，数据值可能需要归一化以帮助训练，进行增强以创建更大的数据集，或者从一种对象类型转换为张量。

这些调整是通过应用*transforms*来完成的。在 PyTorch 中使用 transforms 的美妙之处在于你可以定义一系列 transforms 并在访问数据时应用它。稍后在第五章中，你将看到如何在 CPU 上并行应用 transforms，同时在 GPU 上进行训练。

在下面的代码示例中，我们将定义我们的 transforms 并使用这些 transforms 创建我们的`train_data`数据集：

```py
fromtorchvisionimporttransformstrain_transforms=transforms.Compose(transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=(0.4914,0.4822,0.4465),![1std=(0.2023,0.1994,0.2010))])train_data=CIFAR10(root="./train/",train=True,download=True,transform=train_transforms)![2](img/2.png)
```

![1](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO1-1)

这里的均值和标准差值是根据数据集本身预先确定的。

![2](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO1-2)

创建数据集时设置`transform`参数。

我们使用`transforms.Compose()`类定义一组 transforms。这个类接受一个 transforms 列表并按顺序应用它们。这里我们随机裁剪和翻转图像，将它们转换为张量，并将张量值归一化为预定的均值和标准差。

transforms 在实例化数据集类时传递，并成为数据集对象的一部分。每当访问数据集对象时都会应用 transforms，返回一个由转换后的数据组成的新结果。

我们可以通过打印数据集或其`transforms`属性来查看 transforms，如下面的代码所示：

```py
print(train_data)
# out:
# Dataset CIFAR10
#     Number of datapoints: 50000
#     Root location: ./train/
#     Split: Train
#     StandardTransform
# Transform: Compose(
#                RandomCrop(size=(32, 32),
#                  padding=4)
#                RandomHorizontalFlip(p=0.5)
#                ToTensor()
#                Normalize(
#                  mean=(0.4914, 0.4822, 0.4465),
#                  std=(0.2023, 0.1994, 0.201))
#            )

print(train_data.transforms)
# out:
# StandardTransform
# Transform: Compose(
#                RandomCrop(size=(32, 32),
#                  padding=4)
#                RandomHorizontalFlip(p=0.5)
#                ToTensor()
#                Normalize(
#                  mean=(0.4914, 0.4822, 0.4465),
#                  std=(0.2023, 0.1994, 0.201))
```

我们可以使用索引访问数据，如下一个代码块所示。PyTorch 在访问数据时会自动应用 transforms，因此输出数据将与之前看到的不同：

```py
data, label = train_data[0]

print(type(data))
# out: <class 'torch.Tensor'>

print(data.size())
# out: torch.Size([3, 32, 32])

print(data)
# out:
# tensor([[[-0.1416,  ..., -2.4291],
#          [-0.0060,  ..., -2.4291],
#          [-0.7426,  ..., -2.4291],
#          ...,
#          [ 0.5100, ..., -2.2214],
#          [-2.2214, ..., -2.2214],
#          [-2.2214, ..., -2.2214]]])
```

如你所见，数据输出现在是一个大小为 3×32×32 的张量。它也已经被随机裁剪、水平翻转和归一化。图 3-3 显示了应用 transforms 后的图像。

![“变换后的图像”](img/ptpr_0303.png)

###### 图 3-3。变换后的图像

颜色可能看起来奇怪是因为归一化，但这实际上有助于神经网络模型更好地对图像进行分类。

我们可以为测试定义不同的变换集，并将其应用于我们的测试数据。在测试数据的情况下，我们不希望裁剪或翻转图像，但我们确实需要将图像转换为张量并对张量值进行归一化，如下所示：

```py
test_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
      (0.4914, 0.4822, 0.4465),
      (0.2023, 0.1994, 0.2010))])

test_data = torchvision.datasets.CIFAR10(
      root="./test/",
      train=False,
      transform=test_transforms)

print(test_data)
# out:
# Dataset CIFAR10
#     Number of datapoints: 10000
#     Root location: ./test/
#     Split: Test
#     StandardTransform
# Transform: Compose(
#     ToTensor()
#     Normalize(
#       mean=(0.4914, 0.4822, 0.4465),
#       std=(0.2023, 0.1994, 0.201)))
```

## 数据批处理

现在我们已经定义了变换并创建了数据集，我们可以逐个访问数据样本。然而，当训练模型时，您将希望在每次迭代中传递小批量的数据，正如我们将在“模型开发”中看到的。将数据分批不仅可以实现更高效的训练，还可以利用 GPU 的并行性加速训练。

批处理可以很容易地使用`torch.utils.data.DataLoader`类实现。让我们从 Torchvision 如何使用这个类的示例开始，然后我们将更详细地介绍它。

在下面的代码中，我们为`train_data`创建一个数据加载器，可以用来加载一批样本并应用我们的变换：

```py
trainloader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=16,
                    shuffle=True)
```

我们使用批量大小为 16 个样本，并对数据集进行洗牌，以便数据加载器检索数据的随机抽样。

数据加载器对象结合了数据集和采样器，并为给定数据集提供了一个可迭代的对象。换句话说，您的训练循环可以使用此对象对数据集进行抽样，并一次一个批次地应用变换，而不是一次性地对整个数据集应用变换。这在训练和测试模型时显著提高了效率和速度。

以下代码显示了如何从`trainloader`中检索一批样本：

```py
data_batch, labels_batch = next(iter(trainloader))
print(data_batch.size())
# out: torch.Size([16, 3, 32, 32])

print(labels_batch.size())
# out: torch.Size([16])
```

我们需要使用`iter()`将`trainloader`转换为迭代器，然后使用`next()`再次迭代数据。这仅在访问一个批次时才是必要的。正如我们将在后面看到的，我们的训练循环将直接访问数据加载器，而无需使用`iter()`和`next()`。检查数据和标签的大小后，我们看到它们返回大小为 16 的批次。

我们可以为我们的`test_data`数据集创建一个数据加载器，如下所示：

```py
testloader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=16,
                    shuffle=False)
```

在这里，我们将`shuffle`设置为`False`，因为通常不需要对测试数据进行洗牌，研究人员希望看到可重复的测试结果。

## 通用数据准备（torch.utils.data）

到目前为止，我已经向您展示了如何使用 Torchvision 加载、转换和批处理图像数据。然而，您也可以使用 PyTorch 准备其他类型的数据。PyTorch 库如 Torchtext 和 Torchaudio 为文本和音频数据提供了数据集和数据加载器类，新的外部库也在不断开发中。

PyTorch 还提供了一个名为`torch.utils.data`的子模块，您可以使用它来创建自己的数据集和数据加载器类，就像您在 Torchvision 中看到的那样。它包括`Dataset`、`Sampler`和`DataLoader`类。

### 数据集类

PyTorch 支持映射和可迭代样式的数据集类。*映射样式数据集*源自抽象类`torch.utils.data.Dataset`。它实现了`getitem()`和`len()`函数，并表示从（可能是非整数）索引/键到数据样本的映射。例如，当使用`dataset[idx]`访问这样的数据集时，可以从磁盘上的文件夹中读取第 idx 个图像及其对应的标签。映射样式数据集比可迭代样式数据集更常用，所有表示由键或数据样本制成的映射的数据集都应该使用这个子类。

###### 提示

创建自己的数据集类的最简单方法是子类化映射样式的`torch.utils.data.Dataset`类，并使用自己的代码重写`getitem()`和`len()`函数。

所有子类都应该重写`getitem()`，它为给定键获取数据样本。子类也可以选择重写`len()`，它返回数据集的大小，由许多`Sampler`实现和`DataLoader`的默认选项使用。

另一方面，*可迭代样式数据集*派生自`torch.utils.data.IterableDataset`抽象类。它实现了`iter()`协议，并表示数据样本的可迭代。当从数据库或远程服务器读取数据以及实时生成数据时，通常使用这种类型的数据集。当随机读取昂贵或不确定时，以及批次大小取决于获取的数据时，可迭代数据集非常有用。

PyTorch 的`torch.utils.data`子模块还提供了数据集操作，用于转换、组合或拆分数据集对象。这些操作包括以下内容：

`TensorDataset(*tensors*)`

从张量创建数据集对象

`ConcatDataset(*datasets*)`

从多个数据集创建数据集

`ChainDataset(*datasets*)`

多个`IterableDatasets`链接

`Subset(*dataset*, *indices*)`

从指定索引创建数据集的子集

### 采样器类

除了数据集类，PyTorch 还提供了采样器类，它们提供了一种迭代数据集样本索引的方法。采样器派生自`torch.utils.data.Sampler`基类。

每个`Sampler`子类都需要实现一个`iter()`方法，以提供迭代数据元素索引的方法，以及一个返回迭代器长度的`len()`方法。表 3-1 提供了可用采样器的列表供参考。

表 3-1\. 数据集采样器（`torch.utils.data`）

| Sampler | 描述 |
| --- | --- |
| `SequentialSampler(`*`data_source`*`)` | 按顺序采样数据 |
| `RandomSampler(`*`data_source, replacement=False,`* *`num_samples=None,`* *`generator=None`*`)` | 随机采样数据 |
| `SubsetRandomSampler(`*`indices,`* *`generator=None`*`)` | 从数据集的子集中随机采样数据 |
| `WeightedRandomSampler(`*`weights,`* *`num_samples,`* *`replacement=True,`* *`generator=None`*`)` | 从加权分布中随机采样 |
| `BatchSampler(`*`sampler, batch_size, drop_last`*`)` | 返回一批样本 |
| `distributed.DistributedSampler(`*`dataset,`* *`num_replicas=None,`* *`rank=None,`* *`shuffle=True,`* *`seed=0`*`)` | 在分布式数据集上采样 |

通常不直接使用采样器。它们通常传递给数据加载器，以定义数据加载器对数据集进行采样的方式。

### DataLoader 类

`Dataset`类返回一个包含数据和数据信息的数据集对象。`Sampler`类以指定或随机的方式返回实际数据本身。`DataLoader`类将数据集与采样器结合起来，并返回一个可迭代对象。

数据集和采样器对象不是可迭代的，这意味着您不能在它们上运行`for`循环。数据加载器对象解决了这个问题。我们在本章前面的 CIFAR-10 示例中使用`DataLoader`类构建了一个数据加载器对象。以下是`DataLoader`的原型：

```py
torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                sampler=None,
                batch_sampler=None,
                num_workers=0,
                collate_fn=None,
                pin_memory=False,
                drop_last=False,
                timeout=0,
                worker_init_fn=None,
                multiprocessing_context=None,
                generator=None)
```

`dataset`、`batch_size`、`shuffle`和`sampler`参数是最常用的。`num_workers`参数通常用于增加生成批次的 CPU 进程数量。其余参数仅用于高级情况。

如果您编写自己的数据集类，您只需要调用内置的`DataLoader`来为您的数据生成一个可迭代对象。无需从头开始创建数据加载器类。

本节提供了 PyTorch 数据准备功能的快速参考。现在您了解了如何使用 PyTorch 加载、转换和批处理数据，可以开始使用您的数据来开发和训练深度学习。

# 模型开发

大多数研究和开发都集中在开发新颖的深度学习模型上。模型开发过程包括几个步骤。在这一点上，我假设您已经创建了良好的数据集，并已经准备好让模型处理。

过程中的第一步是模型设计，您将设计一个或多个模型架构并初始化模型的参数（例如权重和偏差）。通常的做法是从现有设计开始，然后修改它或创建自己的设计。我将在本节中向您展示如何做这两种操作。

下一步是训练。在训练过程中，您将通过模型传递训练数据，测量误差或损失，并调整参数以改善结果。

在验证过程中，您将测量模型在未在训练中使用的验证数据上的性能。这有助于防止*过拟合*，即模型在训练数据上表现良好，但不能泛化到其他输入数据。

最后，模型开发过程通常以测试结束。测试是指您测量经过训练的模型在之前未见数据上的性能。本节提供了如何在 PyTorch 中完成模型开发的步骤和子步骤的快速参考。

## 模型设计

在过去的十年中，模型设计研究在所有行业和领域都有了显著的扩展。每年都会有成千上万篇论文涉及计算机视觉、自然语言处理、语音识别和音频处理等领域，以解决早期癌症检测等问题，并创新出自动驾驶汽车等新技术。因此，根据您要解决的问题，可以选择许多不同类型的模型架构。您甚至可以创建一些自己的模型！

### 使用现有和预训练模型

大多数用户开始模型开发时会选择一个现有的模型。也许您想要从现有设计开始，进行轻微修改或尝试小的改进，然后再设计自己的架构。您还可以使用已经用大量数据训练过的现有模型或模型部分。

PyTorch 提供了许多资源来利用现有的模型设计和预训练的神经网络。一个示例资源是基于 PyTorch 的`torchvision`库，用于计算机视觉。`torchvision.models`子包含有不同任务的模型定义，包括图像分类、像素级语义分割、目标检测、实例分割、人体关键点检测和视频分类。

假设我们想要在设计中使用著名的 VGG16 模型。VGG16（也称为 OxfordNet）是一种卷积神经网络架构，以牛津大学的视觉几何组命名，他们开发了这个模型。它在 2014 年提交到大规模视觉识别挑战，并在 ImageNet 上取得了 92.7%的前 5 测试准确率，ImageNet 是一个包含 1400 万手工注释图像的非常庞大的数据集。

我们可以轻松地创建一个预训练的 VGG16 模型，如下面的代码所示：

```py
from torchvision import models

vgg16 = models.vgg16(pretrained=True)
```

默认情况下，模型将是未经训练的，并且具有随机初始化的权重。但是，在我们的情况下，我们希望使用预训练模型，因此我们设置`pretrained = True`。这将下载在 ImageNet 数据集上预训练的权重，并使用这些值初始化我们模型的权重。

您可以通过打印模型来查看 VGG16 模型中包含的层序列。VGG16 模型由三部分组成：`features`、`avgpool`和`classifier`。这里无法打印所有层，所以我们只打印`classifier`部分：

```py
print(vgg16.classifier)

# out:
# Sequential(
#   (0): Linear(in_features=25088,
#               out_features=4096, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=4096,
#               out_features=4096, bias=True)
#   (4): ReLU(inplace=True)
#   (5): Dropout(p=0.5, inplace=False)
#   (6): Linear(in_features=4096,
#               out_features=1000, bias=True)
# )
```

`Linear`、`ReLU`和`Dropout`是`torch.nn`模块。`torch.nn`用于创建神经网络层、激活函数、损失函数和其他神经网络组件。现在不要太担心它；我们将在下一节中详细介绍。

有许多著名的未经训练和经过预训练的模型可用，包括 AlexNet、VGG、ResNet、Inception 和 MobileNet 等。请参考[Torchvision 模型文档](https://pytorch.tips/torchvision-models)获取完整的模型列表以及有关它们使用的详细信息。

PyTorch Hub 是另一个用于现有和预训练 PyTorch 模型的优秀资源。您可以使用`torch.hub.load()`API 从另一个存储库加载模型。以下代码显示了如何从 PyTorch Hub 加载模型：

```py
waveglow = torch.hub.load(
    'nvidia/DeepLearningExamples:torchhub',
    'nvidia_waveglow')
```

在这里，我们加载一个名为 WaveGlow 的模型，用于从 NVIDIA DeepLearningExamples 存储库生成语音。

您可以在[主 PyTorch Hub 网站](https://pytorch.tips/pytorch-hub)找到 PyTorch Hub 存储库的列表。要探索特定存储库的所有可用 API 端点，您可以在存储库上使用`torch.hub.list()`函数，如下面的代码所示：

```py
torch.hub.list(
      'nvidia/DeepLearningExamples:torchhub')

# out:
# ['checkpoint_from_distributed',
#  'nvidia_ncf',
#  'nvidia_ssd',
#  'nvidia_ssd_processing_utils',
#  'nvidia_tacotron2',
#  'nvidia_waveglow',
#  'unwrap_distributed']
```

这列出了*nvidia/DeepLearningExamples:torchhub*存储库中所有可用的模型，包括 WaveGlow、Tacotron 2、SSD 等。尝试在支持 PyTorch Hub 的其他存储库上使用`hub.list()`，看看您可以找到哪些其他现有模型。

从 Python 库（如 Torchvision）和通过 PyTorch Hub 从存储库加载现有和预训练模型，可以让您在自己的工作中建立在以前的研究基础上。在本章后面，我将向您展示如何将您的模型部署到包和存储库中，以便其他人可以访问或基于您自己的研究和开发。

### PyTorch NN 模块（torch.nn）

PyTorch 最强大的功能之一是其 Python 模块`torch.nn`，它使得设计和尝试新模型变得容易。以下代码说明了如何使用`torch.nn`创建一个简单模型。在这个例子中，我们将创建一个名为 SimpleNet 的全连接模型。它包括一个输入层、一个隐藏层和一个输出层，接收 2,048 个输入值并返回 2 个用于分类的输出值：

```py
importtorch.nnasnnimporttorch.nn.functionalasFclassSimpleNet(nn.Module):def__init__(self):![1](img/1.png)super(SimpleNet,self).__init__()![2](img/2.png)self.fc1=nn.Linear(2048,256)self.fc2=nn.Linear(256,64)self.fc3=nn.Linear(64,2)defforward(self,x):![3](img/3.png)x=x.view(-1,2048)x=F.relu(self.fc1(x))x=F.relu(self.fc2(x))x=F.softmax(self.fc3(x),dim=1)returnx
```

![1](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO2-1)

通常将层创建为类属性

![2](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO2-2)

调用基类的`__init__()`函数来初始化参数

![3](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO2-3)

需要定义模型如何处理数据

在 PyTorch 中创建模型被认为是非常“Pythonic”的，意味着它以首选的 Python 方式创建对象。我们首先创建一个名为`SimpleNet`的新子类，它继承自`nn.Module`类，然后我们定义`__init__()`和`forward()`方法。`__init__()`函数初始化模型参数，而`forward()`函数定义了数据如何通过我们的模型传递。

在`__init__()`中，我们调用`super()`函数来执行父`nn.Module`类的`__init__()`方法以初始化类参数。然后我们使用`nn.Linear`模块定义一些层。

`forward()`函数定义了数据如何通过网络传递。在`forward()`函数中，我们首先使用`view()`将输入重塑为一个包含 2,048 个元素的向量，然后我们通过每一层处理输入并应用`relu()`激活函数。最后，我们应用`softmax()`函数并返回输出。

###### 警告

PyTorch 使用术语*module*来描述 NN 层或块。Python 使用这个术语来描述一个可以导入的库包。在本书中，我将坚持使用 PyTorch 的用法，并使用术语*Python 模块*来描述 Python 库模块。

到目前为止，我们已经定义了 SimpleNet 模型中包含的层或模块，它们是如何连接的，以及参数是如何初始化的（通过`super().init()`）。

以下代码显示了如何通过实例化名为`simplenet`的模型对象来创建模型：

```py
simplenet=SimpleNet()![1](img/1.png)print(simplenet)# out:# SimpleNet(#   (fc1): Linear(in_features=2048,#                 out_features=256, bias=True)#   (fc2): Linear(in_features=256,#                 out_features=64, bias=True)#   (fc3): Linear(in_features=64,#                 out_features=2, bias=True)# )input=torch.rand(2048)output=simplenet(input)![2](img/2.png)
```

![1](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO3-1)

实例化或创建模型。

![2](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO3-2)

通过模型运行数据（前向传递）。

如果我们打印模型，我们可以看到它的结构。执行我们的模型就像调用模型对象作为函数一样简单。我们传入输入，模型运行前向传递并返回输出。

这个简单的模型展示了在模型设计过程中需要做出的以下决策：

模块定义

您将如何定义您的 NN 的层？您将如何将这些层组合成构建块？在这个例子中，我们选择了三个线性或全连接层。

激活函数

您将在每个层或模块的末尾使用哪些激活函数？在这个例子中，我们选择在输入和隐藏层使用`relu`激活，在输出层使用`softmax`。

模块连接

您的模块将如何连接在一起？在这个例子中，我们选择简单地按顺序连接每个线性层。

输出选择

将返回什么输出值和格式？在这个例子中，我们从`softmax()`函数返回两个值。

这种范式的简单性、灵活性和 Python 风格是 PyTorch 在深度学习研究中如此受欢迎的原因。PyTorch 的`torch.nn` Python 模块包括用于创建 NN 模型设计所需的构建块、层和激活函数的类。让我们来看看 PyTorch 中可用的不同类型的构建块。

表 3-2 提供了一个*NN 容器*列表。您可以使用容器类来创建更高级别的构建块集。例如，您可以使用`Sequential`在一个块中创建一系列层。

表 3-2\. PyTorch NN 容器

| 类 | 描述 |
| --- | --- |
| `Module` | 所有 NN 模块的基类 |
| `Sequential` | 一个顺序容器 |
| `ModuleList` | 一个以列表形式保存子模块的容器 |
| `ModuleDict` | 一个以字典形式保存子模块的容器 |
| `ParameterList` | 一个以列表形式保存参数的容器 |
| `ParameterDict` | 一个以字典形式保存参数的容器 |

###### 注意

`nn.Module`是所有 NN 构建块的基类。您的 NN 可能由单个模块或包含其他模块的多个模块组成，这些模块也可能包含模块，从而创建构建块的层次结构。

表 3-3 列出了`torch.nn`支持的一些*线性层*。`Linear`通常用于全连接层。

表 3-3\. PyTorch NN 线性层

| 类 | 描述 |
| --- | --- |
| `nn.Identity` | 一个占位符身份运算符，不受参数影响 |
| `nn.Linear` | 将线性变换应用于传入数据的层 |
| `nn.Bilinear` | 将双线性变换应用于传入数据的层 |

表 3-4 列出了`torch.nn`支持的几种*卷积层*。卷积层在深度学习中经常用于在各个阶段对数据应用滤波器。正如您在表中看到的，PyTorch 内置支持 1D、2D 和 3D 卷积以及转置和折叠变体。

表 3-4\. PyTorch NN 卷积层

| 类 | 描述 |
| --- | --- |
| `nn.Conv1d` | 在由多个输入平面组成的输入信号上应用 1D 卷积 |
| `nn.Conv2d` | 在由多个输入平面组成的输入信号上应用 2D 卷积 |
| `nn.Conv3d` | 在由多个输入平面组成的输入信号上应用 3D 卷积 |
| `nn.ConvTranspose1d` | 在由多个输入平面组成的输入图像上应用 1D 转置卷积运算符 |
| `nn.ConvTranspose2d` | 在由多个输入平面组成的输入图像上应用 2D 转置卷积运算符 |
| `nn.ConvTranspose3d` | 对由多个输入平面组成的输入图像应用 3D 转置卷积运算符 |
| `nn.Unfold` | 从批量输入张量中提取滑动本地块 |
| `nn.Fold` | 将滑动本地块的数组组合成一个大的包含张量 |

表 3-5 显示了`torch.nn`中可用的*池化层*。池化通常用于下采样或减少输出层的复杂性。PyTorch 支持 1D、2D 和 3D 池化以及最大或平均池化方法，包括它们的自适应变体。

表 3-5. PyTorch NN 池化层

| 类 | 描述 |
| --- | --- |
| `nn.MaxPool1d` | 对由多个输入平面组成的输入信号应用 1D 最大池化 |
| `nn.MaxPool2d` | 对由多个输入平面组成的输入信号应用 2D 最大池化 |
| `nn.MaxPool3d` | 对由多个输入平面组成的输入信号应用 3D 最大池化 |
| `nn.MaxUnpool1d` | 计算`MaxPool1d`的部分逆操作 |
| `nn.MaxUnpool2d` | 计算`MaxPool2d`的部分逆操作 |
| `nn.MaxUnpool3d` | 计算`MaxPool3d`的部分逆操作 |
| `nn.AvgPool1d` | 对由多个输入平面组成的输入信号应用 1D 平均池化 |
| `nn.AvgPool2d` | 对由多个输入平面组成的输入信号应用 2D 平均池化 |
| `nn.AvgPool3d` | 对由多个输入平面组成的输入信号应用 3D 平均池化 |
| `nn.FractionalMaxPool2d` | 对由多个输入平面组成的输入信号应用 2D 分数最大池化 |
| `nn.LPPool1d` | 对由多个输入平面组成的输入信号应用 1D 幂平均池化 |
| `nn.LPPool2d` | 对由多个输入平面组成的输入信号应用 2D 幂平均池化 |
| `nn.AdaptiveMaxPool1d` | 对由多个输入平面组成的输入信号应用 1D 自适应最大池化 |
| `nn.AdaptiveMaxPool2d` | 对由多个输入平面组成的输入信号应用 2D 自适应最大池化 |
| `nn.AdaptiveMaxPool3d` | 对由多个输入平面组成的输入信号应用 3D 自适应最大池化 |
| `nn.AdaptiveAvgPool1d` | 对由多个输入平面组成的输入信号应用 1D 自适应平均池化 |
| `nn.AdaptiveAvgPool2d` | 对由多个输入平面组成的输入信号应用 2D 自适应平均池化 |
| `nn.AdaptiveAvgPool3d` | 对由多个输入平面组成的输入信号应用 3D 自适应平均池化 |

表 3-6 列出了可用的*填充层*。填充在图层输出增加尺寸时填充缺失数据。PyTorch 支持 1D、2D 和 3D 填充，并可以使用反射、复制、零或常数填充数据。

表 3-6. PyTorch NN 填充层

| 类 | 描述 |
| --- | --- |
| `nn.ReflectionPad1d` | 使用输入边界的反射填充输入张量 |
| `nn.ReflectionPad2d` | 使用输入边界的反射填充输入张量的 2D 输入 |
| `nn.ReplicationPad1d` | 使用输入边界的复制填充输入张量 |
| `nn.ReplicationPad2d` | 使用输入边界的复制填充输入张量的 2D 输入 |
| `nn.ReplicationPad3d` | 使用输入边界的复制填充输入张量的 3D 输入 |
| `nn.ZeroPad2d` | 使用零填充输入张量的边界 |
| `nn.ConstantPad1d` | 使用常数值填充输入张量的边界 |
| `nn.ConstantPad2d` | 使用常数值填充 2D 输入的边界 |
| `nn.ConstantPad3d` | 使用常数值填充 3D 输入的边界 |

表 3-7 列出了*dropout*的可用层。Dropout 通常用于减少复杂性、加快训练速度，并引入一些正则化以防止过拟合。PyTorch 支持 1D、2D 和 3D 层的 dropout，并提供对 alpha dropout 的支持。

表 3-7\. PyTorch NN dropout 层

| 类 | 描述 |
| --- | --- |
| `nn.Dropout` | 在训练期间，使用来自伯努利分布的样本，以概率*p*随机将输入张量的一些元素归零 |
| `nn.Dropout2d` | 对 2D 输入随机归零整个通道 |
| `nn.Dropout3d` | 对 3D 输入随机归零整个通道 |
| `nn.AlphaDropout` | 对输入应用 alpha dropout |

表 3-8 提供了支持*归一化*的类列表。在某些层之间执行归一化以防止梯度消失或爆炸，通过保持中间层输入在一定范围内来实现。它还可以帮助加快训练过程。PyTorch 支持 1D、2D 和 3D 输入的归一化，并提供批次、实例、组和同步归一化等归一化方法。

表 3-8\. PyTorch NN 归一化层

| 类 | 描述 |
| --- | --- |
| `nn.BatchNorm1d` | 对 2D 或 3D 输入（带有可选额外通道维度的 1D 输入的小批量）应用批次归一化，如论文“通过减少内部协变量转移加速深度网络训练的批次归一化”中所述 |
| `nn.BatchNorm2d` | 对 4D 输入（带有额外通道维度的 2D 输入的小批量）应用批次归一化，如论文“批次归一化”中所述 |
| `nn.BatchNorm3d` | 对 5D 输入（带有额外通道维度的 3D 输入的小批量）应用批次归一化，如论文“批次归一化”中所述 |
| `nn.GroupNorm` | 对输入的小批量应用组归一化，如论文“组归一化”中所述 |
| `nn.SyncBatchNorm` | 对*n*维输入（带有额外通道维度的[*n*–2]D 输入的小批量）应用批次归一化，如论文“批次归一化”中所述 |
| `nn.InstanceNorm1d` | 对 3D 输入（带有可选额外通道维度的 1D 输入的小批量）应用实例归一化，如论文“实例归一化：快速风格化的缺失成分”中所述 |
| `nn.InstanceNorm2d` | 对 4D 输入（带有额外通道维度的 2D 输入的小批量）应用实例归一化，如论文“实例归一化”中所述 |
| `nn.InstanceNorm3d` | 对 5D 输入（带有额外通道维度的 3D 输入的小批量）应用实例归一化，如论文“实例归一化”中所述 |
| `nn.LayerNorm` | 对输入的小批量应用层归一化，如论文“层归一化”中所述 |
| `nn.LocalResponseNorm` | 对由多个输入平面组成的输入信号应用局部响应归一化，其中通道占据第二维 |

表 3-9 显示了用于循环神经网络（RNN）的*循环层*。RNN 经常用于处理时间序列或基于序列的数据。PyTorch 内置支持 RNN、长短期记忆（LSTM）和门控循环单元（GRU）层，以及用于 RNN、LSTM 和 GRU 单元的类。

表 3-9\. PyTorch NN 循环层

| 类 | 描述 |
| --- | --- |
| `nn.RNNBase` | RNN 基类 |
| `nn.RNN` | 应用多层 Elman RNN（使用\Tanh 或 ReLU 非线性）到输入序列的层 |
| `nn.LSTM` | 应用多层 LSTM RNN 到输入序列的层 |
| `nn.GRU` | 应用多层 GRU RNN 到输入序列的层 |
| `nn.RNNCell` | 具有 tanh 或 ReLU 非线性的 Elman RNN 单元 |
| `nn.LSTMCell` | 一个 LSTM 单元 |
| `nn.GRUCell` | 一个 GRU 单元 |

表 3-10 列出了用于变压器网络的*变压器层*。变压器网络通常被认为是处理序列数据的最先进技术。PyTorch 支持完整的`Transformer`模型类，还提供了以堆栈和层格式提供的`Encoder`和`Decoder`子模块。

表 3-10. PyTorch NN 变压器层

| 类 | 描述 |
| --- | --- |
| `nn.Transformer` | 一个变压器模型 |
| `nn.TransformerEncoder` | *N*个编码器层的堆叠 |
| `nn.TransformerDecoder` | *N*个解码器层的堆叠 |
| `nn.TransformerEncoderLayer` | 由自我注意（attn）和前馈网络组成的层 |
| `nn.TransformerDecoderLayer` | 由自我注意、多头注意和前馈网络组成的层 |

表 3-11 包含了一系列*稀疏层*。PyTorch 提供了对文本数据嵌入的内置支持，以及用于余弦相似度和两两距离的稀疏层，这些在推荐引擎算法中经常使用。

表 3-11. PyTorch NN 稀疏层和距离函数

| 类 | 描述 |
| --- | --- |
| `nn.Embedding` | 存储固定字典和大小的嵌入 |
| `nn.EmbeddingBag` | 计算“包”嵌入的和或平均值，而不实例化中间嵌入 |
| `nn.CosineSimilarity` | 返回沿一个维度计算的*x*[1]和*x*[2]之间的余弦相似度 |
| `nn.PairwiseDistance` | 使用*p*-范数计算向量*v*[1]和*v*[2]之间的批次两两距离 |

表 3-12 包含了支持计算机视觉的*视觉层*列表。它们包括用于洗牌像素和执行多种上采样算法的层。

表 3-12. PyTorch NN 视觉层

| 类 | 描述 |
| --- | --- |
| `nn.PixelShuffle` | 将形状为(∗, <math alttext="upper C times r squared"><mrow><mi>C</mi> <mo>×</mo> <msup><mi>r</mi> <mn>2</mn></msup></mrow></math> , *H*, *W*)的张量重新排列为形状为(∗, *C*, <math alttext="upper H times r"><mrow><mi>H</mi> <mo>×</mo> <mi>r</mi></mrow></math> , <math alttext="upper W times r"><mrow><mi>W</mi> <mo>×</mo> <mi>r</mi></mrow></math> )的张量 |
| `nn.Upsample` | 上采样给定的多通道 1D（时间）、2D（空间）或 3D（体积）数据 |
| `nn.UpsamplingNearest2d` | 对由多个输入通道组成的输入信号应用 2D 最近邻上采样 |
| `nn.UpsamplingBilinear2d` | 对由多个输入通道组成的输入信号应用 2D 双线性上采样 |

表 3-13 提供了`torch.nn`中所有*激活函数*的列表。激活函数通常应用于层输出，以引入模型中的非线性。PyTorch 支持传统的激活函数，如 sigmoid、tanh、softmax 和 ReLU，以及最近的函数，如 leaky ReLU。随着研究人员设计和应用新的激活函数，更多的函数正在被添加到其中。

表 3-13. PyTorch NN 非线性激活

| 类 | 描述 |
| --- | --- |
| `nn.ELU` | 逐元素应用指数线性单元函数 |
| `nn.Hardshrink` | 逐元素应用硬收缩函数 |
| `nn.Hardsigmoid` | 逐元素应用硬 sigmoid 函数 |
| `nn.Hardtanh` | 逐元素应用 hardtanh 函数 |
| `nn.Hardswish` | 逐元素应用 hardswish 函数 |
| `nn.LeakyReLU` | 逐元素应用泄漏修正线性单元函数 |
| `nn.LogSigmoid` | 逐元素应用对数 sigmoid 函数 |
| `nn.MultiheadAttention` | 允许模型同时关注来自不同表示子空间的信息 |
| `nn.PReLU` | 逐元素应用参数化修正线性单元函数 |
| `nn.ReLU` | 逐元素应用修正线性单元函数 |
| `nn.ReLU6` | 应用带有最大值的修正线性单元函数 |
| `nn.RReLU` | 逐元素应用随机泄漏修正线性单元函数 |
| `nn.SELU` | 逐元素应用缩放指数线性单元函数 |
| `nn.CELU` | 逐元素应用连续可微指数线性单元函数 |
| `nn.GELU` | 应用高斯误差线性单元函数 |
| `nn.Sigmoid` | 逐元素应用 sigmoid 函数 |
| `nn.Softplus` | 逐元素应用 softplus 函数 |
| `nn.Softshrink` | 逐元素应用软收缩函数 |
| `nn.Softsign` | 逐元素应用 softsign 函数 |
| `nn.Tanh` | 逐元素应用双曲正切函数 |
| `nn.Tanhshrink` | 逐元素应用带收缩的双曲正切函数 |
| `nn.Threshold` | 设定输入张量的每个元素的阈值 |
| `nn.Softmin` | 将 softmin 函数应用于*n*维输入张量，以便将*n*维输出张量的元素重新缩放到[0,1]范围，并总和为 1 |
| `nn.Softmax` | 将 softmax 函数应用于*n*维输入张量，以便将*n*维输出张量的元素重新缩放到[0,1]范围，并总和为 1 |
| `nn.Softmax2d` | 将 softmax 函数应用于每个空间位置的特征 |
| `nn.LogSoftmax` | 将 log(softmax(*x*))函数应用于*n*维输入张量 |
| `nn.AdaptiveLogSoftmaxWithLoss` | 提供了一个高效的 softmax 近似，如 Edouard Grave 等人在“Efficient Softmax Approximation for GPUs”中描述的 |

正如您所看到的，PyTorch 的`torch.nn`模块支持一组强大的 NN 层和激活函数。您可以使用其类来创建从简单的顺序模型到复杂的多层次网络、生成对抗网络（GANs）、变换器网络、RNN 等各种模型。

现在您已经知道如何设计您的模型，让我们探讨如何使用 PyTorch 训练和测试您自己的 NN 模型设计。

## 训练

在模型设计过程中，您定义了 NN 模块、它们的参数以及它们之间的连接方式。在 PyTorch 中，您的模型设计被实现为一个从`torch.nn.Module`类派生的模型对象。您可以调用该对象将数据传递到模型中，并根据模型架构和其参数的当前值生成输出。

模型开发的下一步是使用训练数据训练您的模型。训练模型仅涉及估计模型的参数、传递数据和调整参数以获得对数据一般建模更准确的表示。

换句话说，您设置参数为某些值，通过数据，然后将模型的输出与真实输出进行比较以测量误差。目标是改变参数并重复该过程，直到误差最小化且模型的输出与真实输出相同。

### 基本训练循环

PyTorch 相对于其他机器学习框架的一个关键优势是其灵活性，特别是在创建自定义训练循环时。在本章中，我们将探讨一个常用于监督学习的基本训练循环。

在这个例子中，我们将使用本章前面使用过的 CIFAR-10 数据集训练 LeNet5 模型。LeNet5 模型是由 Yann LeCun 及其团队在 1990 年代在贝尔实验室开发的一个简单的卷积 NN，用于分类手写数字。（当时我并不知道，我实际上在新泽西州霍尔姆德尔的同一栋建筑物中为贝尔实验室工作，而这项工作正在进行中。）

可以使用以下代码创建现代化的 LeNet5 模型版本：

```py
fromtorchimportnnimporttorch.nn.functionalasFclassLeNet5(nn.Module):![1](img/1.png)def__init__(self):super(LeNet5,self).__init__()self.conv1=nn.Conv2d(3,6,5)self.conv2=nn.Conv2d(6,16,5)self.fc1=nn.Linear(16*5*5,120)self.fc2=nn.Linear(120,84)self.fc3=nn.Linear(84,10)defforward(self,x):x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))x=F.max_pool2d(F.relu(self.conv2(x)),2)x=x.view(-1,int(x.nelement()/x.shape[0]))x=F.relu(self.fc1(x))x=F.relu(self.fc2(x))x=self.fc3(x)returnxdevice=('cuda'iftorch.cuda.is_available()else'cpu')![2](img/2.png)model=LeNet5().to(device=device)![3](img/3.png)
```

![1](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO4-1)

定义模型类。

![2](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO4-2)

如果有 GPU 可用，请使用。

![3](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO4-3)

创建模型并将其移动到 GPU（如果可用）。

如前面的代码所示，我们的 LeNet5 模型使用了两个卷积层和三个全连接或线性层。它已经通过最大池化和 ReLU 激活进行了现代化。在这个例子中，我们还将利用 GPU 进行训练，以加快训练速度。在这里，我们创建了名为`model`的模型对象。

接下来，我们需要定义损失函数（也称为*标准*）和优化器算法。损失函数确定我们如何衡量模型的性能，并计算预测与真相之间的损失或错误。我们将尝试通过调整模型参数来最小化损失。优化器定义了我们在训练过程中如何更新模型参数。

为了定义损失函数和优化器，我们使用`torch.optim`和`torch.nn`包，如下面的代码所示：

```py
fromtorchimportoptimfromtorchimportnncriterion=nn.CrossEntropyLoss()optimizer=optim.SGD(model.parameters(),![1](img/1.png)lr=0.001,momentum=0.9)
```

![1](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO5-1)

确保传入`model.parameters()`作为您的模型。

在这个例子中，我们使用`CrossEntropyLoss()`函数和随机梯度下降（SGD）优化器。交叉熵损失经常用于分类问题。SGD 算法也常用作优化器函数。选择损失函数和优化器超出了本书的范围；但是，我们将在本章后面检查许多内置的 PyTorch 损失函数和优化器。

###### 警告

PyTorch 优化器要求您使用`parameters()`方法传入模型参数（即`model.parameters()`）。忘记`()`是一个常见错误。

以下 PyTorch 代码演示了基本的训练循环：

```py
N_EPOCHS=10forepochinrange(N_EPOCHS):![1](img/1.png)epoch_loss=0.0forinputs,labelsintrainloader:inputs=inputs.to(device)![2](img/2.png)labels=labels.to(device)optimizer.zero_grad()![3](img/3.png)outputs=model(inputs)![4](img/4.png)loss=criterion(outputs,labels)![5](img/5.png)loss.backward()![6](img/6.png)optimizer.step()![7](img/7.png)epoch_loss+=loss.item()![8](img/8.png)print("Epoch: {} Loss: {}".format(epoch,epoch_loss/len(trainloader)))# out: (results will vary and make take minutes)# Epoch: 0 Loss: 1.8982970092773437# Epoch: 1 Loss: 1.6062103009033204# Epoch: 2 Loss: 1.484384165763855# Epoch: 3 Loss: 1.3944422281837463# Epoch: 4 Loss: 1.334191104450226# Epoch: 5 Loss: 1.2834235876464843# Epoch: 6 Loss: 1.2407222446250916# Epoch: 7 Loss: 1.2081411465930938# Epoch: 8 Loss: 1.1832368299865723# Epoch: 9 Loss: 1.1534993273162841
```

![1](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO6-1)

外部训练循环；循环 10 个 epochs。

![2](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO6-2)

如果可用，将输入和标签移动到 GPU。

![3](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO6-3)

在每次反向传播之前将梯度清零，否则它们会累积。

![4](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO6-4)

执行前向传播。

![5](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO6-5)

计算损失。

![6](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO6-6)

执行反向传播；计算梯度。

![7](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO6-7)

根据梯度调整参数。

![8](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO6-8)

累积批量损失，以便我们可以在整个 epoch 上进行平均。

训练循环由两个循环组成。在外部循环中，我们将在每次迭代或 epoch 中处理整个训练数据集。然而，我们不会等到处理完整个数据集后再更新模型参数，而是一次处理一个较小的数据批次。内部循环遍历每个批次。

###### 警告

默认情况下，PyTorch 在每次调用`loss.backward()`（即反向传播）时累积梯度。这在训练某些类型的 NNs（如 RNNs）时很方便；然而，对于卷积神经网络（CNNs）来说并不理想。在大多数情况下，您需要调用`optimizer.zero_grad()`来将梯度归零，以便在进行反向传播之前优化器正确更新模型参数。

对于每个批次，我们将批次（称为`inputs`）传递到模型中。它运行前向传播并返回计算的输出。接下来，我们使用`criterion()`将模型输出（称为`outputs`）与训练数据集中的真实值（称为`labels`）进行比较，以计算错误或损失。

接下来，我们调整模型参数（即 NN 的权重和偏差）以减少损失。为此，我们首先使用`loss.backward()`执行反向传播来计算梯度，然后使用`optimizer.step()`运行优化器来根据计算的梯度更新参数。

这是训练 NN 模型使用的基本过程。实现可能有所不同，但您可以在创建自己的训练循环时使用此示例作为快速参考。在设计训练循环时，您需要决定如何处理或分批数据，使用什么损失函数以及运行什么优化算法。

您可以使用 PyTorch 内置的损失函数和优化算法之一，也可以创建自己的。

### 损失函数

PyTorch 在`torch.nn` Python 模块中包含许多内置的损失函数。表 3-14 提供了可用损失函数的列表。

表 3-14. 损失函数

| 损失函数 | 描述 |
| --- | --- |
| `nn.L1Loss()` | 创建一个标准，用于测量输入*x*和目标*y*中每个元素的平均绝对误差（MAE） |
| `nn.MSELoss()` | 创建一个标准，测量输入*x*和目标*y*中每个元素的均方误差（平方 L2 范数） |
| `nn.CrossEntropyLoss()` | 将`nn.LogSoftmax()`和`nn.NLLLoss()`结合在一个类中 |
| `nn.CTCLoss()` | 计算连接主义时间分类损失 |
| `nn.NLLLoss()` | 计算负对数似然损失 |
| `nn.PoissonNLLLoss()` | 使用泊松分布计算目标的负对数似然损失 |
| `nn.KLDivLoss()` | 用于测量 Kullback-Leibler 散度损失 |
| `nn.BCELoss()` | 创建一个标准，测量目标和输出之间的二元交叉熵 |
| `nn.BCEWithLogitsLoss()` | 将 sigmoid 层和`nn.BCELoss()`结合在一个类中 |
| `nn.MarginRankingLoss()` | 创建一个标准，用于在给定输入*x*¹、*x*²（两个 1D 小批量张量）和标签 1D 小批量张量*y*（包含 1 或-1）时测量损失 |
| `nn.HingeEmbeddingLoss()` | 在给定输入张量*x*和标签张量*y*（包含 1 或-1）时测量损失 |
| `nn.MultiLabelMarginLoss()` | 创建一个标准，用于优化多类分类的铰链损失（即基于边界的损失），输入为*x*（一个 2D 小批量张量）和输出*y*（一个目标类别索引的 2D 张量） |
| `nn.SmoothL1Loss()` | 创建一个标准，如果绝对元素误差低于 1，则使用平方项，否则使用 L1 项 |
| `nn.SoftMarginLoss()` | 创建一个标准，优化输入张量*x*和目标张量*y*（包含 1 或-1）之间的两类分类逻辑损失 |
| `nn.MultiLabelSoftMarginLoss()` | 创建一个标准，基于最大熵优化多标签一对所有损失 |
| `nn.CosineEmbeddingLoss()` | 创建一个标准，给定输入张量*x*¹、*x*²和标记为 1 或-1 的张量*y*时测量损失 |
| `nn.MultiMarginLoss()` | 创建一个标准，优化多类分类的铰链损失 |
| `nn.TripletMarginLoss()` | 创建一个标准，给定输入张量*x*¹、*x*²、*x*³和大于 0 的边界值时测量三元组损失 |

###### 警告

`CrossEntropyLoss()`函数包括 softmax 计算，通常在 NN 分类器模型的最后一步执行。在使用`CrossEntropyLoss()`时，不要在模型定义的输出层中包含`Softmax()`。

### 优化算法

PyTorch 还在`torch.optim` Python 子模块中包含许多内置的优化器算法。表 3-15 列出了可用的优化器算法及其描述。

表 3-15. 优化器算法

| Algorithm | 描述 |
| --- | --- |
| `Adadelta()` | 自适应学习率方法 |
| `Adagrad()` | 自适应梯度算法 |
| `Adam()` | 随机优化方法 |
| `AdamW()` | 一种 Adam 变体，提出于[“解耦权重衰减正则化”](https://arxiv.org/abs/1711.05101) |
| `SparseAdam()` | 适用于稀疏张量的 Adam 版本 |
| `Adamax()` | 基于无穷范数的 Adam 变体 |
| `ASGD()` | 平均随机梯度下降 |
| `LBFGS()` | BFGS 算法的有限内存实现，受[minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)启发 |
| `RMSprop()` | 均方根传播 |
| `Rprop()` | 弹性反向传播 |
| `SGD()` | 随机梯度下降 |

`torch.optim` Python 子模块支持大多数常用算法。接口足够通用，因此将来也可以轻松集成新算法。访问[torch.optim 文档](https://pytorch.org/docs/stable/optim.html)以获取有关如何配置算法和调整学习率的更多详细信息。

## 验证

现在我们已经训练了我们的模型并尝试最小化损失，我们如何评估其性能？我们如何知道我们的模型将泛化并与以前未见过的数据一起工作？

模型开发通常包括验证和测试循环，以确保不会发生过拟合，并且模型将针对未见数据表现良好。让我们首先讨论验证。在这里，我将为您提供一个如何使用 PyTorch 将验证添加到训练循环中的快速参考。

通常，我们会保留一部分训练数据用于验证。验证数据不会用于训练 NN；相反，我们将在每个时代结束时使用它来测试模型的性能。

在训练模型时进行验证是一个好的实践。在调整超参数时通常会执行验证。例如，也许我们想在五个时代后降低学习率。

在执行验证之前，我们需要将训练数据集分成训练数据集和验证数据集，如下所示：

```py
from torch.utils.data import random_split

train_set, val_set = random_split(
                      train_data,
                      [40000, 10000])

trainloader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=16,
                    shuffle=True)

valloader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=16,
                    shuffle=True)

print(len(trainloader))
# out: 2500
print(len(valloader))
# out: 625
```

我们使用`torch.utils.data`中的`random_split()`函数，将我们的 50,000 个训练图像中的 10,000 个保留用于验证。一旦创建了`train_set`和`val_set`，我们为每个创建数据加载器。

然后我们定义我们的模型、损失函数（或标准）和优化器，如下所示：

```py
from torch import optim
from torch import nn

model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.001,
                      momentum=0.9)
```

以下代码显示了先前的基本训练示例，并添加了验证：

```py
N_EPOCHS=10forepochinrange(N_EPOCHS):# Trainingtrain_loss=0.0model.train()![1](img/1.png)forinputs,labelsintrainloader:inputs=inputs.to(device)labels=labels.to(device)optimizer.zero_grad()outputs=model(inputs)loss=criterion(outputs,labels)loss.backward()optimizer.step()train_loss+=loss.item()# Validationval_loss=0.0model.eval()![2](img/2.png)forinputs,labelsinvalloader:inputs=inputs.to(device)labels=labels.to(device)outputs=model(inputs)loss=criterion(outputs,labels)val_loss+=loss.item()print("Epoch: {} Train Loss: {} Val Loss: {}".format(epoch,train_loss/len(trainloader),val_loss/len(valloader)))
```

![1](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO7-1)

为训练配置模型。

![2](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO7-2)

为测试配置模型。

在处理训练数据后，每个时代都会进行验证。在验证期间，模型会传递尚未用于训练且尚未被模型看到的数据。我们只在验证期间执行前向传递。

###### 注意

在模型对象上运行`.train()`或`.eval()`方法会将模型分别置于训练或测试模式。只有在模型在训练和评估时操作不同的情况下才需要调用这些方法。例如，训练中使用了 dropout 和批量归一化，但在验证或测试中没有使用。在循环中调用`.train()`和`.eval()`是一个好的实践。

如果验证数据的损失减少，那么模型表现良好。然而，如果训练损失减少而验证损失没有减少，那么模型很可能出现过拟合。查看前一个训练循环的结果。您应该有类似以下结果：

```py
# out: (results may vary and take a few minutes)
# Epoch: 0 Train Loss: 1.987607608 Val Loss: 1.740786979
# Epoch: 1 Train Loss: 1.649753892 Val Loss: 1.587019552
# Epoch: 2 Train Loss: 1.511723689 Val Loss: 1.435539366
# Epoch: 3 Train Loss: 1.408525426 Val Loss: 1.361453659
# Epoch: 4 Train Loss: 1.339505518 Val Loss: 1.293459154
# Epoch: 5 Train Loss: 1.290560259 Val Loss: 1.245048282
# Epoch: 6 Train Loss: 1.259268565 Val Loss: 1.285989610
# Epoch: 7 Train Loss: 1.235161985 Val Loss: 1.253840940
# Epoch: 8 Train Loss: 1.207051850 Val Loss: 1.215700019
# Epoch: 9 Train Loss: 1.189215132 Val Loss: 1.183332257
```

正如您所看到的，我们的模型训练良好，似乎没有过拟合，因为训练损失和验证损失都在减少。如果我们训练模型更多的 epochs，我们可能会获得更好的结果。

尽管如此，我们还没有完成。我们的模型可能仍然存在过拟合的问题。我们可能只是在选择超参数时运气好，导致验证结果良好。为了进一步测试是否存在过拟合，我们将一些测试数据通过我们的模型运行。

模型在训练期间从未见过测试数据，测试数据也没有对超参数产生任何影响。让我们看看我们在测试数据集上的表现如何。

## 测试

CIFAR-10 提供了自己的测试数据集，我们在本章前面创建了`test_data`和一个 testloader。让我们通过我们的测试循环运行测试数据，如下所示的代码：

```py
num_correct=0.0forx_test_batch,y_test_batchintestloader:model.eval()![1](img/1.png)y_test_batch=y_test_batch.to(device)x_test_batch=x_test_batch.to(device)y_pred_batch=model(x_test_batch)![2](img/2.png)_,predicted=torch.max(y_pred_batch,1)![3](img/3.png)num_correct+=(predicted==y_test_batch).float().sum()![4](img/4.png)accuracy=num_correct/(len(testloader)\
*testloader.batch_size)![5](img/5.png)print(len(testloader),testloader.batch_size)# out: 625 16print("Test Accuracy: {}".format(accuracy))# out: Test Accuracy: 0.6322000026702881
```

![1](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO8-1)

将模型设置为测试模式。

![2](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO8-2)

预测每个批次的结果。

![3](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO8-3)

选择具有最高概率的类索引。

![4](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO8-4)

将预测与真实标签进行比较并计算正确预测的数量。

![5](img/#co_deep_learning_development___span_class__keep_together__with_pytorch__span__CO8-5)

计算正确预测的百分比（准确率）。

我们在训练 10 个 epochs 后的初始测试结果显示，在测试数据上的准确率为 63%。这不是一个坏的开始；看看您是否可以通过训练更多 epochs 来提高准确率。

现在您知道如何使用 PyTorch 创建训练、验证和测试循环。在创建自己的循环时，可以随时使用此代码作为参考。

现在您已经有了一个完全训练好的模型，让我们探索在模型部署阶段可以做些什么。

# 模型部署

根据您的目标，有许多选项可用于保存或部署您的训练模型。如果您正在进行深度学习研究，您可能希望以一种可以重复实验或稍后访问以进行演示和发表论文的方式保存模型。您还可以希望将模型发布为像 Torchvision 这样的 Python 软件包的一部分，或将其发布到 PyTorch Hub 这样的存储库，以便其他研究人员可以访问您的工作。

在开发方面，您可能希望将训练好的 NN 模型部署到生产环境中，或将模型与产品或服务集成。这可能是一个原型系统、边缘设备或移动设备。您还可以将其部署到本地生产服务器或提供系统可以使用的 API 端点的云服务器。无论您的目标是什么，PyTorch 都提供了能力来帮助您按照您的意愿部署模型。

## 保存模型

最简单的事情之一是保存训练好的模型以备将来使用。当您想对新输入运行模型时，您只需加载它并使用新值调用模型。

以下代码演示了保存和加载训练模型的推荐方法。它使用`state_dict()`方法，该方法创建一个字典对象，将每个层映射到其参数张量。换句话说，我们只需要保存模型的学习参数。我们已经在模型类中定义了模型的设计，因此不需要保存架构。当我们加载模型时，我们使用构造函数创建一个“空白模型”，然后使用`load_state_dict()`为每个层设置参数：

```py
torch.save(model.state_dict(), "./lenet5_model.pt")

model = LeNet5().to(device)
model.load_state_dict(torch.load("./lenet5_model.pt"))
```

请注意，`load_state_dict()`需要一个字典对象，而不是一个保存的`state_dict`对象的路径。在将其传递给`load_state_dict()`之前，您必须使用`torch.load()`对保存的*state_dict*文件进行反序列化。

###### 注意

一个常见的 PyTorch 约定是使用*.pt*或*.pth*文件扩展名保存模型。

您也可以使用`torch.save(`*`PATH`*`)`和`model = torch.load(`*`PATH`*`)`保存和加载整个模型。尽管这更直观，但不建议这样做，因为序列化过程与用于定义模型类的确切文件路径和目录结构绑定。如果您重构类代码并尝试在其他项目中加载模型，您的代码可能会出现问题。相反，保存和加载`state_dict`对象将为您提供更多灵活性，以便稍后恢复模型。

## 部署到 PyTorch Hub

PyTorch Hub 是一个预训练模型存储库，旨在促进研究的可重复性。在本章的前面，我向您展示了如何从 PyTorch Hub 加载预先存在的或预训练的模型。现在，我将向您展示如何通过添加一个简单的*hubconf.py*文件将您的预训练模型（包括模型定义和预训练权重）发布到 GitHub 存储库。*hubconf.py*文件定义了代码依赖关系，并为 PyTorch API 提供一个或多个端点。

在大多数情况下，只需导入正确的函数就足够了，但您也可以明确定义入口点。以下代码显示了如何使用 VGG16 端点从 PyTorch Hub 加载模型：

```py
import torch
vgg16 = torch.hub.load('pytorch/vision',
  'vgg16', pretrained=True)
```

现在，如果您已经创建了 VGG16 并希望将其部署到 PyTorch Hub，您只需要在存储库的根目录中包含以下*hubconf.py*文件。*hubconf.py*配置文件将`torch`设置为依赖项。此文件中定义的任何函数都将充当端点，因此只需导入 VGG16 函数即可完成任务：

```py
dependencies = ['torch']
from torchvision.models.vgg import vgg16
```

如果您想明确定义端点，可以编写如下代码中的函数：

```py
dependencies = ['torch']
from torchvision.models.vgg import vgg16 as _vgg16

# vgg16 is the name of the entrypoint
def vgg16(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help():
 VGG16 model
 pretrained (bool): kwargs,
 load pretrained weights into the model
 """
    # Call the model; load pretrained weights
    model = _vgg16(pretrained=pretrained, **kwargs)
    return model
```

就是这样！全世界的研究人员将因为能够轻松从 PyTorch Hub 加载您的预训练模型而感到高兴。

## 部署到生产环境

将模型保存到文件和存储库可能在进行研究时是可以的；然而，为了解决大多数问题，我们必须将我们的模型集成到产品和服务中。这通常被称为“部署到生产环境”。有许多方法可以做到这一点，PyTorch 具有内置功能来支持它们。部署到生产环境是一个全面的主题，将在第七章中深入讨论。

本章涵盖了很多内容，探讨了深度学习开发过程，并提供了一个快速参考，介绍了 PyTorch 在实现每个步骤时的能力。下一章将介绍更多的参考设计，您可以在涉及迁移学习、情感分析和生成学习的项目中使用。
