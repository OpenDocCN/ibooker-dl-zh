# 13 数据管道

本章涵盖

+   理解训练数据集的常见数据格式和存储类型

+   使用 TensorFlow TFRecord 格式和 tf.data 进行数据集表示和转换

+   构建用于训练期间向模型提供数据的数据管道

+   使用 TF.Keras 预处理层、层子类化和 TFX 组件进行预处理

+   使用数据增强来训练具有平移、缩放和视口不变性的模型

您已经构建了模型，根据需要使用了可组合模型。您已经训练和重新训练了它，并进行了测试和重新测试。现在您准备将其发布。在接下来的两章中，您将学习如何发布一个模型。更具体地说，您将使用 TensorFlow 2.*x*生态系统和 TensorFlow Extended (TFX)将模型从准备和探索阶段迁移到生产环境。

在生产环境中，训练和部署等操作作为管道执行。*管道*具有可配置、可重用、版本控制和保留历史记录的优势。由于生产管道的广泛性，我们需要两章来涵盖它。本章重点介绍数据管道组件，这些组件构成了生产管道的前端。下一章将涵盖训练和部署组件。

让我们从一张图开始，这样您可以看到从开始到结束的过程。图 13.1 显示了基本端到端（e2e）生产管道的整体视图。

![图片](img/CH13_F01_Ferlitsch.png)

图 13.1 基本端到端（e2e）生产管道从数据管道开始，然后移动到训练和部署管道。

现代基本机器学习端到端（e2e）管道从数据仓库开始，这是所有生产模型训练数据的存储库。企业规模的公司正在训练的模型数量各不相同。但以下是我 2019 年经验中的一些例子。生产模型（新版本）重新训练的时间间隔已从每月减少到每周，在某些情况下甚至每天。我的雇主谷歌每天重新训练超过 4000 个模型。在这个规模上的数据仓库是一项巨大的任务。

从数据仓库中，我们需要高效地为模型训练提供数据，并确保数据及时提供，且没有输入/输出（I/O）瓶颈。编译和组装训练批次的上游过程必须在实时中足够快，以免阻碍 GPU/CPU 训练硬件。

在企业规模上，数据仓库通常分布在大量或庞大的计算实例上，无论是在本地、云端还是混合部署，这使得在训练过程中高效地提供数据变得更加具有挑战性。

现在是模型训练阶段。但我们不是训练单个模型的实例。我们并行训练多个实例以找到最佳版本，并且我们在多个阶段进行，从数据收集、准备、增强和预训练到完整训练的超参数搜索。

如果我们将时钟倒退几年，这个管道过程始于领域专家做出有根据的猜测并自动化它们。今天，这些阶段正在变得自我学习。我们已经从自动化（由专家设定规则）发展到自动学习，机器持续地从专家的人类指导中自我学习以不断改进。这就是为什么训练多个模型实例的原因：为了自动学习哪个模型实例将成为最佳训练模型。

然后是版本控制。我们需要一种方法来评估新的训练实例与过去版本，以回答新实例是否比上一个实例更好的问题。如果是这样，就进行版本控制；如果不是，就重复这个过程。对于新版本，将模型部署到生产使用。

在本章中，我们介绍了将数据从存储中移动、预处理数据和将其分批以在训练期间提供给模型实例的过程。在下一章中，我们将介绍训练、重新训练和持续训练、候选模型验证、版本控制、部署以及部署后的测试。

## 13.1 数据格式和存储

我们将首先查看用于存储机器学习图像数据的各种格式。从历史上看，图像数据存储在以下之一：

+   压缩图像格式（例如，JPG）

+   未压缩的原始图像格式（例如，BMP）

+   高维格式（例如，HDF5 或 DICOM）

使用 TensorFlow 2.*x*，我们以 TFRecord 格式存储图像数据。让我们更详细地看看这四种格式中的每一种。

### 13.1.1 压缩和原始图像格式

当深度学习首次在计算机视觉中变得流行时，我们通常直接从原始图像数据中训练，在图像数据解压缩之后。有两种基本方法来准备这种图像数据：从磁盘绘制 JPG、PNG 或其他压缩格式的训练批次；以及在 RAM 中绘制压缩图像的批次。

从磁盘抽取批次

在第一种方法中，当我们构建训练批次时，我们从磁盘以压缩格式（如 JPG 或 PNG）读取图像批次。然后我们在内存中解压缩它们，调整大小，并进行图像预处理，如归一化像素数据。

图 13.2 描述了此过程。在这个例子中，根据批次大小指定的 JPG 图像子集被读入内存，然后解压缩，最后调整大小到模型的输入形状以进行训练。

![图片](img/CH13_F02_Ferlitsch.png)

从磁盘绘制压缩图像很容易做，但持续重新处理以进行训练的成本很高。

让我们看看这种方法的一些优缺点。首先，它非常容易做。以下是步骤：

1.  创建磁盘上所有图像路径及其对应标签的索引（例如，CSV 索引文件）。

1.  将索引读入内存并随机打乱索引。

1.  通过使用打乱的索引文件，将一批图像及其对应标签绘制到内存中。

1.  解压图像。

1.  将解压后的图像调整到模型的输入形状以进行训练。

最大的缺点是，在训练模型时，你必须为每个 epoch 重复前面的步骤。可能成为问题的一步是从磁盘获取数据。这一步可能成为 I/O 瓶颈，并且具体依赖于磁盘存储的类型和数据的位置。理想情况下，我们希望数据存储在尽可能快的读取访问磁盘操作中，并且尽可能靠近（通过限制网络带宽）进行训练的计算设备。

为了比较磁盘和内存之间的这种权衡，假设你正在使用一个具有 ImageNet 输入形状(224, 224, 3)的 SOTA 模型。这个大小对于一般图像分类来说是典型的，而对于图像目标检测或分割，则使用像(512, 512, 3)这样的大尺寸。

形状为(224, 224, 3)的图像需要 150,000 字节的内存（224 × 224 × 3 = 150,000）。为了在 ImageNet 输入形状中连续存储 50,000 个训练图像，你需要 8 GB（50,000 × 150,000）的 RAM——这超过了操作系统、后台应用程序和模型训练所需的内存。现在假设你有 100,000 个训练图像。那么你需要 16 GB 的 RAM。如果你有一百万个图像，你需要 160 GB 的 RAM。

这将需要大量的内存，并且通常只有对于较小的数据集，将所有图像以未压缩格式存储在内存中才是实用的。对于学术和其他教程目的，训练数据集通常足够小，可以将解压和调整大小的图像完全存储在内存中。但在生产环境中，由于数据集太大而无法完全存储在内存中，我们需要使用一种策略，该策略结合了从磁盘获取图像。

从 RAM 中的压缩图像中抽取批次

在这种第二种策略中，我们消除了磁盘 I/O，但每次图像出现在批次中时，仍然在内存中解压缩和调整大小。通过消除磁盘 I/O，我们防止了 I/O 瓶颈，否则会减慢训练速度。例如，如果训练包括 100 个 epoch，每个图像将被解压缩和调整大小 100 次——但所有压缩图像都保持在内存中。

平均 JPEG 压缩约为 10:1。压缩图像的大小将取决于图像来源。例如，如果图像来自 350 万像素的手机（350 万像素），则压缩图像约为 350,000 字节。如果我们的图像是为浏览器加载进行优化的网页图像，则未压缩图像通常在 150,000 到 200,000 字节之间。

假设你有 100,000 张经过优化的训练图像，2 GB 的 RAM 就足够了（100K × 15K = 1.5 GB）。如果你有一百万张训练图像，16 GB 的 RAM 就足够了（1M × 15K = 15 GB）。

图 13.3 说明了这里概述的第二个方法：

1.  将所有压缩图像及其相应的标签读取到内存中作为一个列表。

1.  为内存中所有图像列表索引及其相应的标签创建一个索引。

1.  随机打乱索引。

1.  通过使用打乱的索引文件从内存中抽取一批图像及其相应的标签。

1.  解压缩图像。

1.  调整解压缩图像的大小。

![图片](img/CH13_F03_Ferlitsch.png)

图 13.3 从 RAM 中抽取压缩图像消除了磁盘 I/O，从而加快了过程。

这种方法对于中等大小的数据集来说通常是一个合理的折衷方案。假设我们拥有 200,000 张大小经过优化的网页图像。我们只需要 4 GB 的内存来在内存中存储所有压缩图像，而无需反复从磁盘读取。即使是大批量的图像（比如说，1024 张经过优化的网页图像），我们也只需要额外的 150 MB 内存来存储解压缩的图像——平均每张图像 150,000 字节。

下面是我的常规做法：

1.  如果我的训练数据的解压缩大小小于或等于我的 RAM，我将使用内存中的解压缩图像进行训练。这是最快的选项。

1.  如果我的训练数据的压缩大小小于或等于我的 RAM，我将使用内存中的压缩图像进行训练。这是下一个最快的选项。

1.  否则，我将使用从磁盘抽取的图像进行训练，或者使用我接下来要讨论的混合方法。

混合方法

接下来，让我们考虑从磁盘和内存中喂食训练图像的混合方法。我们为什么要这样做呢？我们想要在可用内存空间和不断从磁盘重新读取图像的 I/O 受限之间找到一个最佳平衡点。

要做到这一点，我们将回顾第十二章中关于采样分布的概念，它近似了总体分布。想象一下，你有 16 GB 的内存来存储数据，预处理后的数据集在调整大小后是 64 GB。在混合喂食中，我们一次取一个大的预处理数据段（在我们的例子中是 8 GB），这些数据段已经被*分层*（示例与训练数据类别分布相匹配）。然后我们反复将相同的段作为 epochs 输入到神经网络中。但每次，我们都会进行图像增强，使得每个 epoch 都是整个预处理图像数据集的唯一采样分布。

我建议在极大数据集上使用这种方法，比如一百万张图像。有了 16 GB 的内存，你可以存储非常大的子分布，并且能够与反复从磁盘读取相比，在可比的训练批次中获得收敛，同时减少训练时间或计算实例需求。

下面是进行混合内存/磁盘喂食的步骤。你还可以在图 13.4 中看到这个过程：

1.  在磁盘上创建一个对预处理的图像数据的分层索引。

1.  根据可用的内存将分层索引划分为存储一个段在内存中的分区。

1.  对于每个段，重复指定数量的 epoch：

    +   在每个 epoch 中随机打乱段。

    +   在每个 epoch 中随机应用图像增强以创建独特的采样分布。

    +   将 mini-batch 输送到神经网络。

![图像](img/CH13_F04_Ferlitsch.png)

图 13.4 从磁盘混合绘制图像作为训练数据的采样分布

### 13.1.2 HDF5 格式

*层次数据格式* 5 (*HDF5*) 已经是存储高维数据（如高分辨率卫星图像）的长期通用格式。因此，你可能想知道什么是 *高维性*？我们将此术语与信息非常密集的单维数据相关联，并且/或具有许多维度（我们称之为 *多维数据*）。正如之前关于 TFRecords 的讨论，这些格式本身并没有实质性地减少存储所需的磁盘空间。相反，它们的目的在于快速读取访问以减少 I/O 开销。

HDF5 是一种用于存储和访问大量多维数据（如图像）的高效格式。规范可以在 HDF5 for Python 网站找到（[www.h5py.org/](https://www.h5py.org/））。该格式支持数据集和组对象，以及每个对象的属性（元数据）。

使用 HDF5 存储图像训练数据的优点包括以下内容：

+   具有广泛的科学用途，例如 NASA 使用的卫星图像（见 [` mng.bz/qevJ`](http://mng.bz/qevJ)）

+   优化了高速数据切片访问

+   NumPy 是否与 NumPy 语法兼容，允许从磁盘访问，就像在内存中一样

+   具有对多维表示、属性和分类的分层访问

Python 的 HDF5 包可以按照以下方式安装：

```
pip install h5py
```

让我们从创建一个包含最基本 HDF5 表示的 dataset 开始，这个表示由原始（解压缩）图像数据和相应的整数标签数据组成。在这个表示中，我们创建了两个 dataset 对象，一个用于图像数据，另一个用于相应的标签：

```
dataset['images'] : [...]
dataset['labels'] : [...]
```

以下代码是一个示例实现。训练数据和标签都是 NumPy 格式。我们打开一个 HDF5 文件以进行写入访问，并创建两个数据集，一个用于图像，一个用于标签：

```
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

with h5py.File('myfile.h5', 'w') as hf:          ❶
    hf.create_dataset("images", data=x_train)    ❷
    hf.create_dataset("labels", data=y_train)    ❸
```

❶ 打开 HDF5 文件以进行写入访问

❷ 将训练图像存储为名为“images”的数据集

❸ 将训练标签存储为名为“labels”的数据集

现在，当我们想要读取图像和标签时，我们首先打开 HDF5 以进行读取访问。然后我们为数据集的图像和标签创建一个迭代器。HDF5 文件句柄是一个字典对象，我们通过数据集名称作为键来引用我们的命名数据集。

接下来，我们重新打开 HDF5 文件进行读取访问，然后为数据集的图像和标签创建 HDF5 迭代器，使用键 `images` 和 `labels`。在这里，`x_train` 和 `y_train` 是 HDF5 迭代器的别名。数据实际上尚未在内存中：

```
hf = h5py.File('myfile.h5', 'r')    ❶
x_train = hf['images']              ❷
y_train = hf['labels']              ❸
```

❶ 打开 HDF5 文件进行读取访问

❷ 为图像数据集创建 HDF5 迭代器

❸ 为标签数据集创建 HDF5 迭代器

由于 HDF5 迭代器使用 NumPy 语法，我们可以通过使用 NumPy 数组切片直接访问数据，这将从磁盘获取数据并将其加载到内存中的 NumPy 数组。在以下代码中，我们通过图像的数组切片（`x_batch`）和相应的标签（`y_batch`）获取单个批次：

```
x_batch = x_train[0:100]     ❶
y_batch = y_train[0:100]     ❷
```

❶ 前面的 100 张图像现在作为 NumPy 数组存储在内存中。

❷ 前面的 100 个标签现在作为 NumPy 数组存储在内存中。

接下来，我们将整个数据集作为批次迭代，并将每个批次输入到模型中进行训练。假设在我们的 HDF5 数据集中存储了 50,000 张图像（例如，在 CIFAR-10 数据集中）。

我们以 50 个批次的尺寸遍历 HDF5 数据集。每次，我们引用下一个顺序数组切片。迭代器将每次从磁盘中抽取 50 张图像并将它们加载到内存中的 `x_batch` 数组。我们同样为相应的标签执行相同的操作，这些标签被加载到 `y_batch` 数组中。然后，我们将图像批次和相应的标签传递给 TF.Keras 方法 `train_on_batch()`，该方法对模型执行单个批次的更新：

```
examples = 50000
batch_size = 50
batches = examples / batch_size
for batch in range(batches):    
    x_batch = x_train[batch*batch_size:(batch+1)*batch_size]     ❶
    y_batch = y_train[batch*batch_size:(batch+1)*batch_size]     ❶
    model.train_on_batch(x_batch, y_batch)                       ❷
```

❶ 从 HDF5 文件中抽取下一个批次作为内存中的 NumPy 切片

❷ 更新批次的模型

HDF5 组

接下来，我们将查看使用组存储 HDF5 格式数据集的另一种存储表示。这种方法具有更高效的存储，消除了存储标签的需求，并且可以存储分层数据集。在这种表示中，我们将为每个类别（标签）及其对应的数据集创建一个单独的组。

以下示例描述了这种表示。我们有两个类别，`cats` 和 `dogs`，并为每个类别创建一个组。在两个组中，我们为相应的图像创建一个数据集。请注意，我们不再需要存储标签数组，因为它们由组名称隐含表示：

```
Group['cats']
    Dataset['images']: [...]
Group['dogs']
    Dataset['images']: [...]
```

以下代码是一个示例实现，其中 `x_cats` 和 `x_dogs` 是猫和狗图像对应的内存中 NumPy 数组：

```
with h5py.File('myfile.h5', 'w') as hf:
    cats = hf.create_group('cats')                 ❶
    cats.create_dataset('images', data=x_cats)     ❶
    dogs = hf.create_group('dogs')                 ❷
    dogs.create_dataset('images', data=x_dogs)     ❷
```

❶ 在组中创建猫类别及其对应的用于存储猫图像的数据集

❷ 在组中创建狗类别及其对应的用于存储狗图像的数据集

然后我们从猫和狗的组版本中读取一批数据。在这个例子中，我们打开 HDF5 组句柄到猫和狗的组。然后使用字典语法引用 HDF5 组句柄。例如，要获取猫图片的迭代器，我们将其引用为 `cats['images']`。接下来，我们使用 NumPy 数组切片从猫数据集中抽取 25 张图片和从狗数据集中抽取 25 张图片，将它们作为 `x_batch` 存入内存。最后一步，我们在 `y_batch` 中生成相应的整数标签。我们将 0 分配给 `cats`，将 1 分配给 `dogs`：

```
hf = h5py.File('myfile.h5', 'r')
cats = hf['cats']                                                        ❶
dogs = hf['dogs']                                                        ❶
x_batch = np.concatenate([cats['images'][0:25], dogs['images'][0:25]])   ❷
y_batch = np.concatenate([np.full((25), 0), np.full((25), 1)])           ❸
```

❶ 为猫和狗的组打开 HDF5 组句柄

❷ 在相应的组内从猫和狗数据集中抽取一批数据

❸ 创建相应的标签

该格式支持对图像进行分层存储，当图像具有分层标签时。如果图像具有分层标签，每个组将进一步划分为子组的层次结构，如下所示。此外，我们使用`Group`属性显式地为相应的标签分配一个唯一的整数值：

```
Group['cats']
        Attribute: {label: 0}
        Group['persian']:
                Attribute: {label: 100}
                Dataset['images']: [...]
        Group['siamese']:
                Attribute: {label: 101}
                Dataset['images']: [...]
Group['dogs']
        Attribute: {label: 1}
        Group['poodles']:
                Attribute: {label: 200}
                Dataset['images']: [...]
        Group['beagle']:
                Attribute: {label: 201}
                Dataset['images']: [...]
```

要实现这种分层存储，我们创建顶级组和子组。在这个例子中，我们为猫创建一个顶级组。然后，使用猫的 HDF5 组句柄，我们为每个品种创建子组，例如`persian`和`siamese`。然后，对于每个品种的子组，我们为相应的图片创建一个数据集。此外，我们使用`attrs`属性显式地为唯一的标签值分配：

```
with h5py.File('myfile.h5', 'w') as hf:
    cats = hf.create_group('cats')                             ❶
    cats.attrs['label'] = 0                                    ❶

    breed = cats.create_group('persian')                       ❷
    breed.attrs['label'] = 100                                 ❷
    breed.create_dataset('images', data=x_cats['persian'])     ❷
    breed = cats.create_group('siamese')                       ❸
      breed.attrs['label'] = 101                               ❸
    breed.create_dataset('images', data=x_cats['siamese'])     ❸
```

❶ 为猫创建顶级组并分配标签 0 作为属性

❷ 在猫组下创建一个二级子组用于波斯猫品种，分配标签，并添加波斯猫的图片

❸ 在猫组下为品种暹罗猫创建一个二级子组，分配标签，并添加暹罗猫的图片

总结来说，HDF5 组功能是访问分层标记数据的简单高效存储方法，尤其是对于具有分层关系的多标签数据集。另一个常见的多标签分层示例是产品。在分层结构的顶部，你有两个类别：`水果`和`蔬菜`。在这两个类别下面是类型（例如，苹果、香蕉、橙子），在类型下面是品种（例如，格兰尼史密斯、加拉、金色美味）。

### 13.1.3 DICOM 格式

虽然 HDF5 格式在卫星图像中广泛使用，但*医学数字成像和通信*（*DICOM*）格式在医学成像中使用。实际上，DICOM 是存储和访问医学成像数据（如 CT 扫描和 X 射线）以及患者信息的 ISO 12052 国际标准。这种格式比 HDF5 更早，专门用于医学研究和医疗保健系统，广泛使用，拥有大量的公开去标识的健康成像数据集。如果你正在处理医学成像数据，你需要熟悉这种格式。

在这里，我将介绍一些使用该格式的基本指南，以及一个演示示例。但如果你是，或者计划成为医学影像方面的专家，我建议你查看 DICOM 网站上的 DICOM 规范和培训教程（[www.dicomstandard.org/](https://www.dicomstandard.org/))。

可以按照以下方式安装 Python 的 DICOM 包：

```
pip install pydicom
```

通常，DICOM 数据集非常大，达到数百个吉字节。这是因为该格式仅用于医学成像，通常包含用于分割的极高分辨率图像，并且可能还包含每张图像的 3D 切片层。

Pydicom 是一个 Python 开源包，用于处理 DICOM 格式的医学图像，它提供了一个用于演示的小数据集。我们将使用这个数据集进行我们的编码示例。让我们首先导入 Pydicom 包并获取测试数据集`CT_small.dcm`：

```
import pydicom
from pydicom.data import get_testdata_files

dcm_file = get_testdata_files('CT_small.dcm')[0]     ❶
```

❶ 此 Pydicom 方法返回演示数据集的文件名列表。

在 DICOM 中，标记的数据还包含表格数据，如患者信息。图像、标签和表格数据可以用于训练一个*多模态模型*（具有两个或更多输入层的模型），每个输入层具有不同的数据类型（例如，图像或数值）。

让我们看看如何从 DICOM 文件格式中读取图像和标签。我们将读取我们的演示数据集，该数据集模拟了患者医学影像数据的真实世界示例，并首先获取一些关于数据集的基本信息。每个数据集包含大量患者信息，可以作为一个字典访问。这个例子只展示了其中的一些字段，其中大部分已经被去标识化。研究日期表示图像拍摄的时间，而模态是成像类型（在这种情况下，为 CT 扫描）：

```
dataset = pydicom.dcmread(dcm_file)
for key in ['PatientID', 'PatientName', 'PatientAge', 'PatientBirthDate', 
            'PatientSex', 'PatientWeight', 'StudyDate', 'Modality']:
    print(key, dataset[key])

PatientID (0010, 0020) Patient ID                          LO: '1CT1'
PatientName (0010, 0010) Patient's Name                      PN: 'CompressedSamples^CT1'
PatientAge (0010, 1010) Patient's Age                      AS: '000Y'
PatientBirthDate (0010, 0030) Patient's Birth Date         DA: ''
PatientSex (0010, 0040) Patient's Sex                      CS: 'O'
PatientWeight (0010, 1030) Patient's Weight                DS: "0.0"
StudyDate (0008, 0020) Study Date                          DA: '20040119'
Modality (0008, 0060) Modality                             CS: 'CT'
```

最后，我们将提取图像数据并按此处和图 13.5 所示进行显示：

```
rows = int(dataset.Rows)
cols = int(dataset.Columns)
print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
        rows=rows, cols=cols, size=len(dataset.PixelData)))

plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
plt.show()
```

![](img/CH13_F05_Ferlitsch.png)

图 13.5 从 DICOM 文件中提取的图像

关于访问和解析 DICOM 图像的更多详细信息可以在规范以及 Pydicom 教程中找到（[`pydicom.github.io/`](https://pydicom.github.io/))。

### 13.1.4 TFRecord 格式

*TFRecord*是 TensorFlow 用于存储和访问用于 TensorFlow 训练的数据集的标准格式。这种二进制格式最初是为了使用 Google 的协议缓冲区定义高效地序列化结构化数据而设计的，但后来由 TensorFlow 团队进一步开发，用于高效地序列化非结构化数据，如图像、视频和文本。除了是 TensorFlow 组织推荐的训练数据格式外，该格式已无缝集成到 TF 生态系统，包括`tf.data`和 TFX。

在这里，我们再次只是了解一下如何使用该格式为训练 CNN 的图像。对于详细信息和标准信息，请查看教程 ([www.tensorflow.org/tutorials/load_data/tfrecord](http://www.tensorflow.org/tutorials/load_data/tfrecord))。

图 13.6 是使用 TFRecords 作为 tf.data 表示的分层关系图。以下是三个步骤：

1.  在最高层是 `tf.data.Dataset`。这是训练数据集的内存表示。

1.  下一个级别是一系列一个或多个 TFRecords。这些是数据集的磁盘存储。

1.  在最底层是 `tf.Example` 记录；每个记录包含一个单一的数据示例。

![](img/CH13_F06_Ferlitsch.png)

图 13.6 `tf.data`、TFRecords 和 `tf.Example` 的分层关系

现在我们从底层开始描述这种关系。我们将把训练数据中的每一个数据示例转换为 `tf.Example` 对象。例如，如果我们有 50,000 个训练图像，我们就有 50,000 个 `tf.Example` 记录。接下来，我们将这些记录序列化，以便它们在磁盘上作为 TFRecord 文件具有快速的读取访问。这些文件是为了顺序访问而设计的，不是随机访问，以最小化读取访问，因为它们将只写入一次但将被多次读取。

对于大量数据，记录通常被分割成多个 TFRecord 文件，以进一步最小化特定于存储设备的读取访问时间。虽然每个序列化的 `tf.Example` 条目的大小、示例数量、存储设备类型和分布将最好地决定分区大小；TensorFlow 团队建议每个分区的大小为 100 到 200 MB 作为一般规则。

tf.Example: Features

`tf.Example` 的格式与 Python 字典和 JSON 对象都有相似之处。一个示例（例如，一个图像）及其相应的元数据（例如，标签）封装在 `tf.Example` 类对象中。此对象由一个或多个 `tf.train.Feature` 条目列表组成。每个特征条目可以是以下数据类型之一：

+   `tf.train.ByteList`

+   `tf.train.FloatList`

+   `tf.train.Int64List`

`tf.train.ByteList` 类型用于字节序列或字符串。字节的例子可以是图像的编码或原始字节，字符串的例子可以是 NLP 模型的文本字符串或标签的类名。

`tf.train.FloatList` 类型用于 32 位（单精度）或 64 位（双精度）浮点数。结构化数据集中某一列的连续实值是一个例子。

`tf.train.Int64List` 类型用于 32 位和 64 位的有符号和无符号整数以及布尔值。对于整数，此类型用于结构化数据集中某一列的分类值，或标签的标量值，例如。

使用 `tf.Example` 格式编码图像数据时，采用了一些常见的做法：

+   用于编码图像数据的特征条目

+   为图像形状（用于重建）创建一个特征条目

+   为相应的标签创建一个特征条目

以下是一个定义 `tf.train.Example` 以编码图像的通用示例；`/entries here/` 是图像数据和相应元数据的字典条目的占位符，我们将在后面讨论。请注意，TensorFlow 将此格式称为 `tf.Example`，数据类型称为 `tf.train.Example`。这可能会让人一开始感到困惑。

```
example = tf.train.Example(features = { /entries here/ })
```

tf.Example：压缩图像

在下一个示例中，我们创建一个未解码的图像 `tf.train.Example` 对象（图像以压缩的磁盘格式存储）。这种方法的好处是，当作为 TFRecord 的一部分存储时，占用的磁盘空间最少。缺点是，在训练过程中，每次从磁盘读取 TFRecord 并向神经网络提供数据时，都必须解压缩图像数据；这是时间和空间之间的权衡。

在下面的代码示例中，我们定义了一个函数，用于将磁盘上的图像文件（参数 `path`）和相应的标签（参数 `label`）转换为以下形式：

+   首先使用 OpenCV 方法 `cv2.imread()` 读取磁盘上的图像，并将其解压缩为原始位图，以获取图像的形状（行数、列数、通道数）。

+   使用 `tf.io.gfile.GFile()` 再次从磁盘读取图像，格式保持原始压缩状态。注意，`tf.io.gfile.GFile()` 等同于文件 `open()` 函数，但如果图像存储在 GCS 存储桶中，该方法针对 I/O 读写性能进行了优化。

+   使用三个字典条目为特征对象实例化一个 `tf.train.Example()` 实例：

    +   `image` — 一个 `BytesList`，用于存储未压缩（原始磁盘数据）的图像数据

    +   `label` — 一个表示标签值的 `Int64List`

    +   `shape` — 一个表示图像形状（行数、高度、通道数）的 `Int64List` 元组

在我们的示例中，如果我们假设磁盘上图像的大小为 24,000 字节，那么 TFRecord 文件中 `tf.train.Example` 条目的大小大约为 25,000 字节。

```
import tensorflow as tf
import numpy as np
import sys
import cv2

def TFExampleImage(path, label):
        ''' The original compressed version of the image '''

        image = cv2.imread(path)                                            ❶
        shape = image.shape                                                 ❶

        with tf.io.gfile.GFile(path, 'rb') as f:                            ❷
            disk_image = f.read()                                           ❷

        return tf.train.Example(features = tf.train.Features(feature = {
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value =
                                  [disk_image])),                           ❸
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value =
                                  [label])),                                ❹
        'shape': tf.train.Feature(int64_list = tf.train.Int64List(value =
                                  [shape[0], shape[1], shape[2]]))          ❺
        }))

example = TFExampleImage('example.jpg', 0)
print(example.ByteSize())
```

❶ 使用 OpenCV 获取图像的形状

❷ 使用 TensorFlow 从 GCS 存储桶中读取压缩图像

❸ 为压缩图像的字节数据创建一个特征条目

❹ 为相应的标签创建一个特征条目

❺ 为相应的形状（H × W × C）创建一个特征条目

tf.Example：未压缩图像

在接下来的代码示例中，我们创建一个 `tf.train.Example` 条目来存储图像的未压缩版本到 TFRecord 中。这样做的好处是只需从磁盘读取一次图像，并且在训练过程中从磁盘上的 TFRecord 读取条目时不需要解压缩。

缺点是条目的大小将显著大于图像的磁盘版本。在前面的示例中，假设 95% 的 JPEG 压缩率，TFRecord 中的条目大小将是 500,000 字节。注意，在图像数据的 `BytesList` 编码中，保留了 `np.uint8` 数据格式。

```
def TFExampleImageUncompressed(path, label):
        ''' The uncompressed version of the image '''

        image = cv2.imread(path)                                            ❶
        shape = image.shape                                                 ❶

        return tf.train.Example(features = tf.train.Features(feature = {
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = 
                                  [image.tostring()])),                     ❷
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = 
                                  [label])),                                ❸
        'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = 
                                  [shape[0], shape[1], shape[2]]))          ❹
        }))

example = TFExampleImageUncompressed('example.jpg', 0)
print(example.ByteSize())
```

❶ 使用 OpenCV 读取未压缩的图像

❷ 为未压缩的图像字节数据创建一个特征条目

❸ 为相应的标签创建一个特征条目

❹ 为相应的形状（H × W × C）创建一个特征条目

tf.Example：机器学习就绪

在我们的最后一个代码示例中，我们首先对像素数据进行归一化（通过除以 255）并存储归一化的图像数据。这种方法的优势在于，在训练过程中每次从磁盘上的 TFRecord 读取条目时，我们不需要对像素数据进行归一化。缺点是现在像素数据以 `np.float32` 存储的，比相应的 `np.uint8` 大四倍。假设相同的图像示例，现在 TFRecord 的大小将是 200 万字节。

```
def TFExampleImageNormalized(path, label):
        ''' The normalized version of the image '''

        image = (cv2.imread(path) / 255.0).astype(np.float32)               ❶
        shape = image.shape                                                 ❶

        return tf.train.Example(features = tf.train.Features(feature = {
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value =
                                  [image.tostring()])),                     ❷
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value =
                                  [label])),                                ❸
        'shape': tf.train.Feature(int64_list = tf.train.Int64List(value =
                                  [shape[0], shape[1], shape[2]]))          ❹
        }))

example = TFExampleImageNormalized('example.jpg', 0)
print(example.ByteSize())
```

❶ 使用 OpenCV 读取未压缩的图像并对像素数据进行归一化

❷ 为未压缩的图像字节数据创建一个特征条目

❸ 为相应的标签创建一个特征条目

❹ 为相应的形状（H × W × C）创建一个特征条目

TFRecord：写入记录

现在我们已经在内存中构建了一个 `tf.train.Example` 条目，下一步是将它写入磁盘上的 TFRecord 文件。我们将这样做是为了在训练模型时从磁盘上喂入训练数据。

为了最大化写入和从磁盘存储读取的效率，记录被序列化为字符串格式以存储在 Google 的协议缓冲区格式中。在下面的代码中，`tf.io.TFRecordWriter` 是一个函数，它将序列化的记录写入到这种格式的文件中。在将 TFRecord 写入磁盘时，使用文件名后缀 `.tfrecord` 是一个常见的约定。

```
 with tf.io.TFRecordWriter('example.tfrecord') as writer:     ❶
        writer.write(example.SerializeToString())             ❷
```

❶ 创建一个 TFRecord 文件写入对象

❷ 将单个序列化的 tf.train.Example 条目写入文件

磁盘上的 TFRecord 文件可能包含多个 `tf.train.Example` 条目。以下代码将多个序列化的 `tf.train.Example` 条目写入 TFRecord 文件：

```
with tf.io.TFRecordWriter('example.tfrecord') as writer:      ❶
        for example in examples:                              ❷
            writer.write(example.SerializeToString())         ❷
```

❶ 创建一个 TFRecord 文件写入对象

❷ 将每个 tf.train.Example 条目顺序写入 TFRecord 文件

TFRecord：读取记录

下一个代码示例演示了如何按顺序从 TFRecord 文件中读取每个 `tf.train.Example` 条目。我们假设文件 `example.tfrecord` 包含多个序列化的 `tf.train.Example` 条目。

`tf.compat.v1.io.record_interator()` 创建了一个迭代器对象，当在 `for` 语句中使用时，将按顺序读取内存中的每个序列化的 `tf.train.Example`。`ParseFromString()` 方法用于将数据反序列化为内存中的 `tf.train.Example` 格式。

```
iterator = tf.compat.v1.io.tf_record_iterator('example.tfrecord')  ❶
for entry in iterator:                                             ❷
        example = tf.train.Example()                               ❷
        example.ParseFromString(entry)                             ❷
```

❶ 创建一个迭代器以按顺序遍历 tf.train.Example 条目

❷ 遍历每个条目并将序列化字符串转换为 tf.train.Example

或者，我们可以通过使用 `tf.data.TFRecordDataset` 类来读取和迭代来自 TFRecord 文件的一组 `tf.train.Example` 条目。在下一个代码示例中，我们执行以下操作：

+   实例化一个 `tf.data.TFRecordDataset` 对象作为磁盘记录的迭代器

+   定义字典 `feature_description` 以指定如何反序列化序列化的 `tf.train.Example` 条目

+   定义辅助函数 `_parse_function()` 以接受一个序列化的 `tf.train .Example` (`proto`) 并使用字典 `feature_description` 进行反序列化

+   使用 `map()` 方法迭代反序列化每个 `tf.train.Example` 条目

```
dataset = tf.data.TFRecordDataset('example.tfrecord')                   ❶

feature_description = {                                                 ❷
    'image': tf.io.FixedLenFeature([],  tf.string),                     ❷
    'label': tf.io.FixedLenFeature([],  tf.int64),                      ❷
    'shape': tf.io.FixedLenFeature([3], tf.int64),                      ❷
}

def _parse_function(proto):                                             ❸
    ''' parse the next serialized tf.train.Example using the feature    ❸
    description '''                                                     ❸
    return tf.io.parse_single_example(proto, feature_description)       ❸

parsed_dataset = dataset.map(_parse_function)                           ❹
```

❶ 为磁盘上的数据集创建一个迭代器

❷ 创建一个字典描述以反序列化 `tf.train.Example`

❸ 用于 tf.train.Example 的顺序解析函数

❹ 使用 `map()` 函数解析数据集的每个条目

如果我们打印 `parsed_dataset`，输出应该如下所示：

```
<MapDataset shapes: {image: (), shape: (), label: ()}, 
types: {image: tf.string, shape: tf.int64, label: tf.int64}>
```

## 13.2 数据馈送

在上一节中，我们讨论了数据在内存和磁盘上的结构和存储方式，用于训练。本节介绍使用 `tf.data` 将数据摄入到管道中，`tf.data` 是 TensorFlow 模块，用于构建数据集管道。它可以从各种来源构建管道，例如内存中的 NumPy 和 TensorFlow 张量以及磁盘上的 TFRecords。

数据集管道是通过 `tf.data.Dataset` 类创建的生成器。因此，`tf.data` 指的是 Python 模块，而 `tf.data.Dataset` 指的是数据集管道。数据管道用于预处理和为训练模型提供数据。

首先，我们将从内存中的 NumPy 数据构建数据管道，然后随后将构建一个从磁盘上的 TFRecords 构建的数据管道。

### 13.2.1 NumPy

要从 NumPy 数据创建一个内存中的数据集生成器，我们使用 `tf.data .Dataset` 方法 `from_tensor_slices``()`。该方法将训练数据作为参数，指定为一个元组：`(images, labels)`。

在以下代码中，我们使用 CIFAR-10 NumPy 数据创建一个 `tf.data.Dataset`，我们将其指定为参数值 `(x_train`, `y_train)`：

```
from tensorflow.data import Dataset
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

dataset = Dataset.from_tensor_slices((x_train, y_train))     ❶
```

❶ 为内存中的 NumPy 训练数据创建一个数据集生成器

注意，`dataset` 是一个生成器；因此它不可索引。你不能执行 `dataset[0]` 并期望获取第一个元素。这将抛出一个异常。

接下来，我们将遍历数据集。但我们要分批进行，就像我们在使用 TF.Keras 中的 `fit()` 方法馈送数据时指定批大小一样。在下一个代码示例中，我们使用 `batch()` 方法将数据集的批大小设置为 128。请注意，`batch` 不是一个属性。它不会改变现有数据集的状态，而是创建一个新的生成器。这就是为什么我们将 `dataset .batch(128)` 赋值回原始的 `dataset` 变量。TensorFlow 将这些类型的数据集方法称为 *数据集转换*。

接下来，我们遍历数据集，并对每个批次 `(x_batch,` `y_batch)` 打印其形状。对于每个批次，这将输出图像数据的形状为 (128, 32, 32, 3) 以及相应的标签的形状为 (128, 1)：

```
dataset = dataset.batch(128)                ❶
for x_batch, y_batch in dataset:            ❷
    print(x_batch.shape, y_batch.shape)     ❷
```

❶ 将数据集转换为以 128 个批次的迭代

❷ 以 128 个批次的批量遍历数据集

如果我们第二次重复相同的 `for` 循环迭代，我们将不会得到任何输出。为什么？发生了什么？默认情况下，数据集生成器只遍历一次数据集。为了连续重复，就像我们有多个时代一样，我们使用 `repeat()` 方法作为另一个数据集转换。由于我们希望每个时代都能看到不同随机顺序的批次，我们使用 `shuffle()` 方法作为另一个数据集转换。这里展示了数据集转换的顺序：

```
dataset = dataset.shuffle(1024)
dataset = dataset.repeat()
dataset = dataset.batch(128)
```

数据集转换方法也是可链式的。通常可以看到它们被链在一起。这一行与前面的三行序列相同：

```
dataset = dataset.shuffle(1024).repeat().batch(128)
```

应用转换的顺序很重要。如果我们首先使用 `repeat()` 然后是 `shuffle()` 转换，那么在第一个时代，批次将不会被随机化。

还要注意，我们为 `shuffle()` 转换指定了一个值。这个值表示每次从数据集中拉取到内存中并混洗的示例数量。例如，如果我们有足够的内存来存储整个数据集，我们将此值设置为训练数据中的示例总数（例如，CIFAR-10 的 50000）。这将一次性混洗整个数据集——一个完整的混洗。如果我们没有足够的内存，我们需要计算我们可以节省多少内存，并将其除以内存中每个示例的大小。假设我们有 2 GB 的空闲内存，每个内存中的示例是 200,000 字节。在这种情况下，我们将大小设置为 10,000（2 GB / 200K）。

在下一个代码示例中，我们使用 `tf.data.Dataset` 作为数据管道，用 CIFAR-10 数据训练一个简单的卷积神经网络。`fit()` 方法与 `tf.data.Dataset` 生成器兼容。我们不是传递原始图像数据和相应的标签，而是传递由变量 `dataset` 指定的数据集生成器。

因为它是一个生成器，`fit()` 方法不知道一个时代中会有多少批次。因此，我们需要额外指定 `steps_per_epoch` 并将其设置为训练数据中的批次数量。在我们的例子中，我们将其计算为训练数据中的示例数量除以批次大小 `(50000 // 128)`：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, Activation, 
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten

model = Sequential()
model.add(Conv2D(16, (3,3), strides=1, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(32, (3,3), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])

batches = 50000 // 128                                              ❶
model.fit(dataset, steps_per_epoch=batches, epochs=5, verbose=1)    ❷
```

❶ 计算数据集中的批次数量

❷ 使用 fit() 方法通过数据集生成器进行训练

在本节中，我们介绍了从内存中的数据源构建数据管道，例如在 NumPy 或 TensorFlow 张量格式中。接下来，我们将介绍使用 TFRecords 从磁盘上的数据源构建数据管道。

### 13.2.2 TFRecord

要从 TFRecord 文件创建磁盘上的数据集生成器，我们使用 `tf.data` 方法 `TFRecordDataset()`。该方法接受单个 TFRecord 文件的路径或多个 TFRecord 文件路径列表作为参数。如前所述，每个 TFRecord 文件可能包含一个或多个训练示例，例如图像，并且为了 I/O 性能，训练数据可能跨越多个 TFRecord 文件。

此代码为单个 TFRecord 文件创建数据集生成器：

```
dataset = tf.data.TFRecordDataset('example.tfrecord')
```

此代码示例为多个 TFRecord 文件创建数据集生成器，当数据集跨越多个 TFRecord 文件时：

```
dataset = tf.data.TFRecordDataset(['example1.tfrecord', 'example2.tfrecord'])
```

接下来，我们必须告诉数据集生成器如何解析 TFRecord 文件中的每个序列化条目。我们使用 `map()` 方法，它允许我们定义一个用于解析 TFRecord 特定示例的函数，该函数将在每次从磁盘读取示例时应用（映射）到每个示例。

在以下示例中，我们首先定义 `feature_description` 来描述如何解析 TFRecord 特定的条目。使用前面的示例，我们假设条目的布局是一个字节编码的图像键/值，一个整数标签键/值，以及一个三个元素的整数形状键/值。然后我们使用 `tf.io.parse_single_example()` 方法根据特征描述解析 TFRecord 文件中的序列化示例：

```
feature_description = {                                                ❶
    'image': tf.io.FixedLenFeature([], tf.string),                     ❶
    'label': tf.io.FixedLenFeature([], tf.int64),                      ❶
    'shape': tf.io.FixedLenFeature([3], tf.int64),                     ❶
}

def _parse_function(proto):                                            ❷
    ''' parse the next serialized tf.train.Example using the feature description '''
    return tf.io.parse_single_example(proto, feature_description)      ❸

dataset = dataset.map(_parse_function)  
```

为反序列化 `tf.train.Example` 创建字典描述

用于 `tf.train.Example` 的顺序解析函数

使用 `map()` 函数解析数据集中的每个条目

让我们现在进行一些更多的数据集转换，然后看看我们迭代磁盘上的 TFRecord 时会看到什么。在这个代码示例中，我们应用转换以打乱顺序并将批大小设置为 2。然后我们以两个示例为一批迭代数据集，并显示相应的 `label` 和 `shape` 键/值：

```
dataset = dataset.shuffle(4).batch(2)         ❶
for entry in dataset:                         ❷
    print(entry['label'], entry['shape'])     ❷
```

为磁盘上的数据库创建迭代器

以两个示例为一批迭代磁盘上的 TFRecord

以下输出显示每个批次包含两个示例，第一批的标签是 0 和 1，第二批的标签是 1 和 0，所有图像的大小为 (512, 512, 3)：

```
tf.Tensor([0 1], shape=(2,), dtype=int64) tf.Tensor(
[[512 512   3]
 [512 512   3]], shape=(2, 3), dtype=int64)
tf.Tensor([1 0], shape=(2,), dtype=int64) tf.Tensor(
[[512 512   3]
 [512 512   3]], shape=(2, 3), dtype=int64)
```

TFRecord：压缩图像

到目前为止，我们还没有解决序列化图像数据编码的格式。通常，图像以压缩格式（如 JPEG）或未压缩格式（原始）编码。在下一个代码示例中，我们在 `_parse_function()` 中添加一个额外的步骤，使用 `tf.io.decode_jpg()` 将图像数据从压缩格式（JPEG）解码为未压缩格式。因此，随着每个示例从磁盘读取并反序列化，现在图像数据已解码：

```
dataset = tf.data.TFRecordDataset(['example.tfrecord'])   

feature_description = {   
    'image': tf.io.FixedLenFeature([], tf.string),  
    'label': tf.io.FixedLenFeature([], tf.int64),  
    'shape': tf.io.FixedLenFeature([3], tf.int64), 
}

def _parse_function(proto):   
    ''' parse the next serialized tf.train.Example 
        using the feature description 
    '''
    example = tf.io.parse_single_example(proto, feature_description)
    example['image'] = tf.io.decode_jpg(example['image'])              ❶
    return example

dataset = dataset.map(_parse_function)
```

解码压缩的 JPEG 图像

TFRecord：未压缩图像

在下一个代码示例中，编码的图像数据以未压缩格式存储在 TFRecord 文件中。因此，我们不需要解压缩它，但仍然需要使用 `tf.io.decode_raw()` 将编码的字节数组解码为原始位图格式。

在此阶段，原始解码数据是一个一维数组，因此我们需要将其重新调整回其原始形状。在获取原始解码数据后，我们从`shape`键/值中获取原始形状，然后使用`tf.reshape()`调整原始图像数据：

```
dataset = tf.data.TFRecordDataset(['tfrec/example.tfrecord'])

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64), 
    'shape': tf.io.FixedLenFeature([3], tf.int64),
}

def _parse_function(proto): 
    ''' parse the next serialized tf.train.Example using the 
        feature description 
    '''
    example = tf.io.parse_single_example(proto, feature_description)  
    example['image'] = tf.io.decode_raw(example['image'], tf.uint8)    ❶
    shape = example['shape']                                           ❷
    example['image'] = tf.reshape(example['image'], shape)             ❸
    return example

dataset = dataset.map(_parse_function)
```

❶ 将图像数据解码为未压缩的原始格式

❷ 获取原始图像形状

❸ 将解码的图像重新调整回原始形状

## 13.3 数据预处理

到目前为止，我们已经涵盖了数据格式、存储以及从内存或磁盘读取训练数据，以及一些数据预处理。在本节中，我们将更详细地介绍预处理。首先，我们将探讨如何将预处理从上游数据管道移出，并移动到预处理器模型组件中，然后我们将探讨如何使用 TFX 设置预处理管道。

### 13.3.1 使用预处理的预处理

您应该记得，当 TensorFlow 2.0 发布时，其中一个建议是将预处理移动到图中。我们可以采取两种方法。首先，我们可以将其硬编码到图中。其次，我们可以使预处理独立于模型，但实现即插即用，这样预处理就会在图中进行，并且可以互换。这种即插即用预处理的优点如下：

+   在训练和部署管道中的可重用和可互换组件

+   在图中运行，而不是在 CPU 的上游运行，从而在为训练提供模型时消除潜在的 I/O 绑定

图 13.7 展示了使用即插即用预处理器进行预处理。这个展示显示了在训练或部署模型时可以选择的即插即用预处理器组件集合。连接预处理器的要求是它的输出形状必须与模型的输入形状匹配。

![](img/CH13_F07_Ferlitsch.png)

图 13.7 即插即用预处理器在训练和部署期间可以互换。

预处理器要实现与现有模型（已训练和未训练）的即插即用，有两个要求：

+   预处理器的输出必须与模型的输入匹配。例如，如果模型以输入（224, 224, 3）为输入——例如标准的 ResNet50——那么预处理器的输出也必须是（224, 224, 3）。

+   预处理输入的形状必须与输入源匹配，无论是用于训练还是预测。例如，输入源的大小可能与模型训练时的大小不同，而预处理器已经被训练来学习调整图像大小的最佳方法。

即插即用预处理器通常分为两种类型：

+   在部署后与模型一起用于预测。例如，预处理器处理输入源的调整大小和归一化，当预测请求由未压缩图像的原始字节组成时。

+   仅在训练期间使用，部署后不使用。例如，预前缀在训练期间对图像进行随机增强，以学习平移和尺度不变性，从而消除配置数据管道进行图像增强的需要。

我们将介绍两种构建预前缀的方法，以将数据预处理移动到图中。第一种方法为 TF.Keras 2.*x* 添加了层，用于此目的，第二种方法使用子类化来创建自己的自定义预处理层。

TF.Keras 预处理层

为了进一步帮助和鼓励将预处理移动到图中，TF.Keras 2.2 及后续版本引入了新的预处理层。这消除了使用子类化构建常见预处理步骤的需要。本节涵盖了这三个层：`Rescaling`、`Resizing` 和 `CenterCrop`。对于完整列表，请参阅 TF.Keras 文档 ([`mng.bz/7jqe`](https://shortener.manning.com/7jqe))。

图 13.8 展示了通过包装技术将即插即用的预前缀附加到现有模型的过程。在这里，创建了一个第二个模型实例，我们称之为 *包装模型*。使用顺序 API，例如，包装模型由两个组件组成：首先添加预前缀，然后添加现有模型。为了将现有模型连接到预前缀，预前缀的输出形状必须匹配现有模型的输入形状。

![](img/CH13_F08_Ferlitsch.png)

图 13.8 一个包装模型将一个预前缀附加到现有模型上。

下一个代码示例实现了一个即插即用的预前缀，我们在训练现有模型之前添加它。首先，我们创建了一个未训练的 ConvNet，包含两个 16 和 32 个过滤器的卷积 (`Conv2D`) 层。然后我们将特征图展平 (`Flatten`) 成一个 1D 向量，不进行降维，作为瓶颈层和最终的 `Dense` 层进行分类。我们将使用这个 ConvNet 模型作为我们想要训练和部署的模型。

接下来，我们实例化另一个空模型，我们将称之为 `wrapper` 模型。包装模型将包含两个部分：预前缀和未训练的 ConvNet 模型。对于预前缀，我们添加预处理层 `Rescaling` 以将整数像素数据归一化到浮点值 0 和 1 之间。由于预前缀将是包装模型中的输入层，我们添加参数 `(input_shape=(32, 32, 3))` 以指定输入形状。由于 `Rescaling` 不改变输入的大小，预前缀的输出与模型输入相匹配。

最后，我们训练包装模型并使用包装模型进行预测。因此，对于训练和预测，整数像素数据的归一化现在成为包装模型的一部分，在图上执行，而不是在 CPU 上上游执行。

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, Activation, 
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling    ❶

model = Sequential()                                                        ❷
model.add(Conv2D(16, (3,3), strides=1, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(32, (3,3), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

wrapper = Sequential()                                                      ❸
wrapper.add(Rescaling(scale=1.0/255, input_shape=(32, 32, 3)))              ❸

wrapper.add(model)                                                          ❹
wrapper.compile(loss='sparse_categorical_crossentropy', optimizer='adam',   ❹
                metrics=['acc'])                                            ❹

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
wrapper.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)           ❺
wrapper.evaluate(x_test, y_test)                                            ❺
```

❶ 导入缩放预处理层

❷ 构建一个简单的 ConvNet

❸ 使用缩放构建预前缀

❹ 将预前缀添加到 ConvNet

❺ 使用预预处理对 ConvNet 进行训练和测试

可即插即用的预预处理可以包含多个预处理层，如图 13.9 所示，例如图像输入的调整大小，随后是像素数据的重新缩放。在此表示中，由于缩放不会改变输出形状，因此前一个调整大小层的输出形状必须与茎组输入形状相匹配。

![](img/CH13_F09_Ferlitsch.png)

图 13.9 带有两个预处理层的预预处理

以下代码实现了一个可插拔的预预处理，它执行两个功能：调整输入大小并归一化像素数据。我们首先创建与上一个示例相同的 ConvNet。接下来，我们创建一个包含两个预处理层的包装模型：一个用于图像调整大小（`Resizing`）和一个用于归一化（`Rescaling`）。在此示例中，ConvNet 的输入形状为（28, 28, 3）。我们使用预预处理将输入从（32, 32, 3）调整大小到（28, 28, 3）以匹配 ConvNet 并归一化像素数据：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, 
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Resizing

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = Sequential()
model.add(Conv2D(16, (3,3), strides=1, padding='same', input_shape=(28, 28, 3)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(32, (3,3), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

wrapper = Sequential()                                                 ❶
wrapper.add(Resizing(height=28, width=28, input_shape=(32, 32, 3)))    ❷
wrapper.add(Rescaling(scale=1.0/255))                                  ❷
wrapper.add(model)                                                     ❸
wrapper.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
metrics=['acc'])

wrapper.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)
wrapper.evaluate(x_test, y_test)
```

❶ 创建包装模型

❷ 将预预处理添加到包装模型中

❸ 将 ConvNet 添加到模型中并进行训练

现在我们已经训练了模型，我们可以移除预预处理并使用模型进行推理。在下一个示例中，我们假设图像测试数据已经调整为（28, 28, 3）以匹配我们的 ConvNet，并且我们在模型上游对像素数据进行归一化。我们知道包装模型的前两层是预预处理，这意味着我们的底层训练模型从第三层开始；因此我们将 `model` 设置为 `wrapper.layers[2]`。现在我们可以使用不带预处理的底层模型进行推理：

```
x_test = (x_test / 255.0).astype(np.float32)      ❶
model = wrapper.layers[2]                         ❷
model.evaluate(x_test, y_test)                    ❸
```

❶ 数据预处理在 CPU 上上游进行

❷ 获取不带预处理的底层模型

❸ 使用底层模型进行评估（预测）

预处理链式连接

图 13.10 描述了预预处理链式连接；一个预预处理将与模型部署一起保留，另一个将在模型部署时移除。在此，我们创建了两个包装模型：一个内部包装和一个外部包装。内部包装包含一个将在模型部署时保留的预处理预预处理，而外部包装包含一个将在模型部署时从模型中移除的图像增强预预处理。对于训练，我们训练外部包装模型，对于部署，我们部署内部包装模型。

![](img/CH13_F10_Ferlitsch.png)

图 13.10 预预处理链式连接——内部预预处理在部署后与模型一起保留，外部预预处理被移除。

在我们的最终示例中，我们将两个预处理层连接在一起。第一个预处理层用于训练，然后在推理时移除，第二个层则保留在模型中。在第一个（内部）预处理层中，我们对整数像素数据进行归一化（`Rescaling`）。在第二个（外部）预处理层中，我们对输入图像进行中心裁剪（`CenterCrop`）以进行训练。我们还设置第二个预处理层的输入大小为任意高度和宽度：（`None, None, 3`）。因此，我们可以在训练期间将不同大小的图像输入到第二个预处理层，它将它们裁剪为`（32, 32, 3）`，然后将其作为输入传递给第一个预处理层，该层执行归一化。

最后，当训练完成后，我们移除第二个（外部）预处理层，并在没有中心裁剪的情况下进行推理：

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization,
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling,
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = Sequential()                                                        ❶
model.add(Conv2D(16, (3,3), strides=1, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(32, (3,3), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])

wrapper1 = Sequential()                                                     ❷
wrapper1.add(Rescaling(scale=1.0/255, input_shape=(32, 32, 3)))             ❷
wrapper1.add(model)                                                         ❷
wrapper1.compile(loss='sparse_categorical_crossentropy',                    ❷
                 optimizer='adam', metrics=['acc'])                         ❷

wrapper2 = Sequential()                                                     ❸
wrapper2.add(CenterCrop(height=32, width=32, input_shape=(None, None, 3)))  ❸
wrapper2.add(wrapper1)                                                      ❸
wrapper2.compile(loss='sparse_categorical_crossentropy',                    ❸
                 optimizer='adam', metrics=['acc'])                         ❸

wrapper2.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)          ❹

wrapper2.layers[1].evaluate(x_test, y_test)                                 ❺
```

❶ 构建 ConvNet 模型

❷ 将第一个预处理层附加到图像数据在训练和推理期间的归一化

❸ 在训练期间将第二个预处理层附加到图像数据以进行中心裁剪

❹ 使用第一个和第二个预处理层训练模型

❺ 仅使用第一个预处理层进行推理

TF.Keras 子类化层

作为使用 TF.Keras 内置预处理层的替代方案，我们可以通过层子类化创建自己的自定义预处理层。当你需要一个不是预构建在 TF.Keras 预处理层中的自定义预处理步骤时，这很有用。

TF.Keras 中所有预定义层都是`TF.Keras.Layer`类的子类。要创建自己的自定义层，你需要执行以下操作：

1.  创建一个继承（继承自）`TF.Keras.Layer`类的类。

1.  覆盖`__init__()`、`build()`和`call()`方法。

让我们现在通过子类化来构建我们自己的预处理层`Rescaling`版本。在下一个示例代码实现中，我们定义了`Rescaling`类，它继承自`TF.Keras.Layer`。接下来，我们覆盖了初始化器`__init__()`。在底层的`Layer`类中，初始化器接受两个参数：

+   `input_shape`—当作为模型中的第一个层使用时，模型输入的形状

+   `name`—为这个层实例定义的用户可定义名称

我们通过`super()`调用将这些两个参数传递到底层的`Layer`初始化器。

任何剩余的`__init__()`参数都是层特定的（自定义）参数。对于`Rescaling`，我们添加了`scale`参数并将其值保存在类对象中。

接下来，我们覆盖了`build()`方法。当使用`compile()`编译模型或使用功能 API 将一个层绑定到另一个层时，会调用此方法。底层方法接受`input_shape`参数，该参数指定了层的输入形状。底层参数`self.kernel`设置了层的内核形状；内核形状指定了参数的数量。如果我们有可学习的参数，我们将设置内核形状及其初始化方式。由于`Rescaling`没有可学习的参数，我们将其设置为`None`。

最后，我们覆盖了 `call()` 方法。当图在训练或推理时执行时，将调用此方法。底层方法将 `inputs` 作为参数，这是层的输入张量，并返回输出张量。在我们的情况下，我们将输入张量中的每个像素值乘以在层初始化时设置的 `scale` 因子，并输出缩放后的张量。

我们添加装饰器 `@tf.function` 来告诉 TensorFlow AutoGraph ([www.tensorflow.org/api_docs/python/tf/autograph](https://www.tensorflow.org/api_docs/python/tf/autograph)) 将此方法中的 Python 代码转换为模型中的图操作。AutoGraph 是 TensorFlow 2.0 中引入的一个工具，它是一个预编译器，可以将各种 Python 操作转换为静态图操作。这允许可以将转换为静态图操作的 Python 代码从在 CPU 上执行转移到图中的执行。虽然支持许多 Python 构造进行转换，但转换仅限于非 eager 张量的图操作。

```
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, Activation, 
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten
from tensorflow.keras.layers import Layer

class Rescaling(Layer):                                                     ❶
    """ Custom Layer for Preprocessing Input """
    def __init__(self, scale, input_shape=None, name=None):                 ❷
        """ Constructor """
        super(Rescaling, self).__init__(input_shape=input_shape, name=name)
        self.scale = scale                                                  ❸

    def build(self, input_shape):                                           ❹
        """ Handler for building the layer """
        self.kernel = None                                                  ❺

    @tf.function                                                            ❻
    def call(self, inputs):                                                 ❼
        """ Handler for layer object is callable """
        inputs = inputs * self.scale                                        ❽
        return inputs
```

❶ 使用层子类化定义自定义层

❷ 覆盖初始化器并添加输入参数 scale

❸ 在层对象实例中保存缩放因子

❹ 覆盖 build() 方法

❺ 没有可学习的（可训练）参数。

❻ 告诉 AutoGraph 将方法转换为放入模型中的图操作

❼ 覆盖 call() 方法

❽ 对输入张量中的每个像素（元素）进行缩放

关于 `Layer` 和 `Model` 子类化的详细信息，请参阅 TensorFlow 团队提供的各种教程和笔记本示例，例如“通过子类化创建新的层和模型” ([`mng.bz/my54`](https://shortener.manning.com/my54))。

### 13.3.2 使用 TF Extended 进行预处理

到目前为止，我们已经讨论了从底层组件构建数据管道。在这里，我们看到如何使用更高层次的组件，这些组件封装了更多的步骤，使用 TensorFlow Extended 来构建数据管道。

*TensorFlow Extended* (*TFX*) 是一个端到端的生产管道。本节涵盖了 TFX 的数据管道部分，如图 13.11 所示。

![](img/CH13_F11_Ferlitsch.png)

图 13.11 TFX 数据管道

在高层次上，`ExampleGen` 组件从数据集源中摄取数据。`StatisticsGen` 组件分析数据集中的示例，并生成数据集分布的统计数据。`SchemaGen` 组件通常用于结构化数据，从数据集统计数据中推导出数据模式。例如，它可能推断特征类型，如分类或数值，数据类型，范围，并设置数据策略，例如如何处理缺失数据。`ExampleValidator` 组件根据数据模式监控训练和提供数据中的异常。这四个组件共同构成了 TFX 数据验证库。

`Transform` 组件执行数据转换，例如特征工程、数据预处理和数据增强。该组件由 TFX Transform 库组成。

TFX 包不是 TensorFlow 2.*x* 版本的组成部分，因此您需要单独安装它，如下所示：

```
pip install tfx
```

本小节的其余部分仅从高层次概述这些组件。有关详细参考和教程，请参阅 TensorFlow 文档中的 TFX ([www.tensorflow.org/tfx](https://www.tensorflow.org/tfx))。

接下来，让我们创建一个代码片段，用于导入我们将在所有后续代码示例中使用的模块和类：

```
from tfx.utils.dsl_utils import external_input                              ❶
from tfx.components import ImportExampleGen                                 ❷
from tfx.components import StatisticsGen, SchemaGen, ExampleValidator       ❸
from tfx.components import Transform                                        ❸
from tfx.orchestration.experimental.interactive.interactive_context import  ❹
InteractiveContext                               
```

❶ 导入 util 以从外部源读取数据集

❷ 导入 ExampleGen 组件实例用于 TFRecords

❸ 导入剩余的 TFX 数据管道组件

❹ 导入 TFX 管道编排

在后续的代码示例中，我们将使用 TFX 管道编排模块进行交互式演示。这些代码序列设置了一个管道，但在编排执行管道之前，没有任何操作。

```
context = InteractiveContext()       ❶
```

❶ 实例化交互式管道编排

ExampleGen

`ExampleGen` 组件是 TFX 数据管道的入口点。其目的是从一个数据集中抽取示例批次。它支持多种数据集格式，包括 CSV 文件、TFRecords 和 Google BigQuery。`ExampleGen` 的输出是 `tf.Example` 记录。

下一个代码示例实例化了用于磁盘上 TFRecord 格式数据集（例如，图像）的 `ExampleGen` 组件。它包括两个步骤。

让我们从第二个步骤开始。我们将 `ExampleGen` 组件实例化为子类 `ImportExampleGen`，其中初始化器将示例输入源（input=examples）作为参数。

现在让我们退一步，定义一个连接到输入源的连接器。由于输入源是 TFRecords，我们使用 TFX 工具方法 `external_input()` 将连接器映射到磁盘上的 TFRecords 和我们的 `ImportExampleGen` 实例之间：

```
examples = external_input('tfrec')                ❶
example_gen = ImportExampleGen(input=examples)    ❶
context.run(example_gen)                          ❷
```

❶ 实例化一个以 TFRecords 为输入源的 ExampleGen

❷ 执行管道

统计生成器

`StatisticsGen` 组件从示例输入源生成数据集统计信息。这些示例可以是训练/评估数据或服务数据（后者在此未涉及）。在下一个代码示例中，我们为训练/评估数据生成数据集统计信息。我们实例化一个 `StatisticsGen()` 实例，并将示例源传递给初始化器。在这里，示例的源是前一个代码示例中我们的 `example_gen` 实例的输出。输出通过 `ExampleGen` 属性 `outputs` 指定，它是一个字典，键/值对为 `examples`：

```
statistics_gen = StatisticsGen(
      examples=example_gen.outputs['examples'])        ❶
context.run(statistics_gen)                            ❷
statistics_gen.outputs['statistics']._artifacts[0]     ❸
```

❶ 使用 ExampleGen 输出创建一个 StatisticsGen 实例

❷ 执行管道

❸ 显示统计信息的交互式输出

最后一条代码输出的结果将类似于以下内容。`uri` 属性是一个本地目录，用于存储统计信息。`split_names` 属性表示两组统计信息，一组用于训练，另一组用于评估：

```
Artifact of type 'ExampleStatistics' (uri: /tmp/tfx-interactive-2020-05-28T19_02_20.322858-8g1v59q7/StatisticsGen/statistics/2) at 0x7f9c7a1414d0
```

| `.type` | `<class 'tfx.types.standard_artifacts.ExampleStatistics'>` |
| --- | --- |
| `.uri` | `/tmp/tfx-interactive-2020-05-28T19_02_20.322858-8g1v59q7/StatisticsGen/statistics/2` |
| `.span` | `0` |
| `.split_names` | `["train", "eval"]` |

SchemaGen

`SchemaGen` 组件从数据集统计信息中生成模式。在下一个代码示例中，我们为训练/评估数据生成数据集统计信息中的模式。我们实例化了一个 `SchemaGen()` 对象，并将数据集统计信息的来源传递给初始化器。在我们的例子中，统计信息的来源是前一个代码示例中 `statistics_gen` 实例的输出。输出通过 `StatisticsGen` 属性 `outputs` 指定，该属性是一个字典，键/值对为 `statistics`：

```
schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'])    ❶
context.run(schema_gen)
schema_gen.outputs['schema']._artifacts[0]              ❷
```

❶ 使用 ExampleGen 的输出实例化一个 SchemaGen

❷ 显示模式的交互式输出

最后一条代码输出的结果将类似于以下内容。`uri` 属性是一个本地目录，用于存储模式。模式的文件名将是 schema.pbtxt。

```
Artifact of type 'Schema' (uri: /tmp/tfx-interactive-2020-05-28T19_02_20
➥ .322858-8g1v59q7/SchemaGen/schema/4) at 0x7f9c500d1790
```

| `.type` | `<class 'tfx.types.standard_artifacts.Schema'>` |
| --- | --- |
| `.uri` | `/tmp/tfx-interactive-2020-05-28T19_02_20.322858-8g1v59q7/SchemaGen/schema/4` |

对于我们的例子，schema.pbtxt 的内容将类似于以下内容：

```
feature {
  name: "image"
  value_count {
    min: 1
    max: 1
  }
  type: BYTES
  presence {
    min_fraction: 1.0
    min_count: 1
  }
}
feature {
  name: "label"
  value_count {
    min: 0
    max: 1
  }
  type: INT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
}
feature {
  name: "shape"
  value_count {
    min: 3
    max: 3
  }
  type: INT
  presence {
    min_fraction: 1.0
    min_count: 1
  }
}
```

示例验证器

`ExampleValidator` 组件通过使用数据集统计信息和模式作为输入来识别数据集中的异常。在下一个代码示例中，我们识别了训练/评估数据的数据集统计信息和模式中的异常。我们实例化了一个 `ExampleValidator()` 对象，并将数据集统计信息的来源和模式传递给初始化器。在我们的例子中，统计信息和模式的来源分别是前一个代码示例中 `statistics_gen` 实例和 `schema_gen` 实例的输出。

```
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])                  ❶
context.run(example_validator)
example_validator.outputs['anomalies']._artifacts[0]      ❷
```

❶ 实例化一个 ExampleValidator

❷ 显示异常的交互式输出

最后一条代码输出的结果将类似于以下内容。`uri` 属性是一个本地目录，用于存储异常信息（如果有存储的话）。

```
Artifact of type 'ExampleAnomalies' (uri: ) at 0x7f9c780cbdd0
```

| `.type` | `<class 'tfx.types.standard_artifacts.ExampleAnomalies'>` |
| --- | --- |
| `.uri` |  |
| `.span` | `0` |

转换

`Transform` 组件在训练或推理过程中将数据集转换作为示例被绘制到批次中执行。数据集转换通常是结构化数据的特征工程和数据预处理。

在下面的代码示例中，我们将数据集的示例批次进行转换。我们实例化了一个`Transform()`实例。初始化器接受三个参数：要转换的`examples`的输入源、数据`schema`以及一个自定义 Python 脚本来执行转换（例如，`my_preprocessing_fn.py`）。我们不会介绍如何编写用于转换的自定义 Python 脚本；更多详情，请参阅 TensorFlow 教程中的 TFX 组件部分([`mng.bz/5Wqa`](http://mng.bz/5Wqa))。

```
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file='my_preprocessing_fn.py')  
context.run(transform)
```

下一节将介绍如何将图像增强集成到现有的数据管道中，例如使用`tf.data`和/或使用预茎构建的数据管道。

## 13.4 数据增强

*图像* *(数据)增强* 在过去几年中有着各种各样的目的。最初，它被视为通过在现有图像上执行一些随机变换，将更多图像添加到现有数据集以进行训练的手段。随后，研究人员了解到某些类型的增强可以扩展模型的检测能力，例如对于不变性和遮挡。

本节展示了如何将图像增强添加到现有的数据管道中。我们将从图像增强背后的基本概念及其如何帮助模型泛化到未训练的示例开始。然后，我们将转向将方法集成到`tf.data`管道中的方法。最后，我们将看到如何通过在训练期间附加到模型的预茎中并随后断开连接的预处理层来集成它。

本节重点介绍常见的增强技术和实现，以扩展模型的不变性检测能力。接下来，我们将描述不变性是什么以及为什么它很重要。

### 13.4.1 不变性

今天，我们不再将图像增强的目的视为仅仅是为了向训练集中添加更多示例。相反，它是一种通过具有特定目的来生成现有图像的额外图像，以训练模型实现平移、缩放和视口不变性的手段。

好吧，这一切意味着什么呢？这意味着我们希望无论图像中的位置（平移）、对象的大小（缩放）和观看角度（视口），都能识别图像中的对象（或视频帧中的对象）。图像增强使我们能够训练模型实现不变性，而无需额外的真实世界人工标注数据。

图像增强通过随机变换训练数据中的图像来实现，以实现不同的平移、缩放和视口。在研究论文中，执行以下四种图像增强类型是一种常见的做法：

+   随机中心裁剪

+   随机翻转

+   随机旋转

+   随机平移

让我们详细看看这四种类型。

随机中心裁剪

在一个 *裁剪* 中，我们取图像的一部分。通常，裁剪是矩形的。中心裁剪是正方形，并且位于原始图像的中心（图 13.12）。裁剪的大小随机变化，因此在某些情况下，它只是图像的一小部分，而在其他情况下，则是一大部分。然后，裁剪的图像被调整大小以适应模型的输入大小。

这种变换有助于训练模型以实现尺度不变性，因为我们正在随机放大图像中物体的大小。你可能想知道这些随机裁剪是否可能会裁剪掉所有或过多的感兴趣物体，导致图像无用。通常，这不会发生，以下是一些原因：

+   前景物体（感兴趣的对象）往往出现在图片的中心或附近。

+   我们为裁剪设置一个最小尺寸，防止裁剪太小以至于不包含可用的数据。

+   物体的边缘被裁剪出来有助于训练模型进行遮挡，其中其他物体遮挡了感兴趣物体的一部分。

![](img/CH13_F12_Ferlitsch.png)

图 13.12 随机中心裁剪

随机翻转

在一个 *翻转* 中，我们在水平或垂直轴上翻转图像。如果我们沿垂直轴翻转，我们得到一个镜像图像。如果我们沿水平轴翻转，我们得到一个颠倒的图像。这种变换有助于训练模型以实现视口不变性。

你可能会想，在某些情况下，镜像或颠倒的图像在现实世界的应用中可能没有意义。例如，你可能会说停车标志的镜像没有意义，或者一辆颠倒的卡车。也许它确实有意义。也许停车标志是通过后视镜看到的？也许你的车翻了，卡车从你的视角看确实是颠倒的。

随机翻转还有助于学习物体的基本特征，这些特征与背景分离——无论模型在现实世界预测中部署时的实际视口如何。

随机旋转

在一个 *旋转* 中，我们沿着中心点旋转图像。我们可以旋转最多 360 度，但由于随机变换的常见做法是将它们串联起来，因此与随机翻转结合时，+/- 30 度的范围就足够了。这种变换有助于训练模型以实现视口不变性。

图 13.13 是两个串联的随机变换的例子。第一个是随机旋转，然后是随机中心裁剪。

![](img/CH13_F13_Ferlitsch.png)

图 13.13 随机变换链

随机平移

在一个*随机平移*中，我们垂直或水平地移动图像。如果我们水平移动，我们就会从左侧或右侧丢弃像素，并用相同数量的黑色像素（无信号）在对面替换它们。如果我们垂直移动，我们就会从顶部或底部丢弃像素，并用相同数量的黑色像素（无信号）在对面替换它们。一个一般规则是，将平移限制在图像宽度/高度的+/-20%以内，以防止裁剪掉太多感兴趣的对象。这种变换有助于训练模型以实现平移不变性。

除了这里提到的四种之外，还有大量的其他变换技术可以用于实现不变性。

### 13.4.2 使用 tf.data 进行增强

可以通过使用`map()`方法将图像变换添加到`tf.data.Dataset`管道中。在这种情况下，我们将变换编码为一个 Python 函数，该函数以图像为输入并输出变换后的图像。然后我们将该函数指定为`map()`方法的参数，该参数将应用于批处理中的每个元素。

在下一个示例中，我们定义了一个`flip()`函数，该函数将对数据集中的每个图像执行随机翻转平移，每次图像被绘制到批处理中时。在示例中，我们从一个 NumPy 图像训练数据元组及其相应的标签`(x_train, y_train)`创建`tf.data.Dataset`。然后我们将`flip()`函数应用于数据集，即`dataset.map(flip)`。由于批处理中的每个图像都是一个图像和标签的元组，因此变换函数需要两个参数：`(image, label)`。同样，我们需要返回相应的元组，但用变换后的输入图像替换：`(transform, label)`：

```
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def flip(image, label):                                             ❶
    transform = tf.image.random_flip_left_right(image)              ❷
    transform = tf.image.random_flip_up_down(transform)             ❷

    return transform, label                                         ❸

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.map(flip)                                         ❹
```

❶ 执行图像变换的函数，输入图像和相应的标签

❷ 随机翻转输入图像

❸ 返回变换后的图像和相应的标签

❹ 将翻转变换函数应用于每个图像/标签对

接下来，我们将对`tf.data.Dataset`进行多个变换。在下面的代码示例中，我们添加第二个变换函数以进行随机裁剪。请注意，`tf.image.random_crop()`方法不是一个中心裁剪。与总是居中的随机大小不同，这个 TensorFlow 方法设置一个固定的大小，由`shape`指定，但图像中的裁剪位置是随机的。然后我们将两个变换链在一起，首先进行随机翻转，然后进行随机裁剪：`dataset.map(flip).map(crop)`。

```
def crop(image, label):                                                ❶
    shape = (int(image.shape[0] * 0.8), int(image.shape[1] * 0.8), 
             image.shape[2])                                           ❷
    transform = tf.image.random_crop(image, shape)                     ❸

    return transform, label

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.map(flip).map(crop)                                  ❹
```

❶ 执行图像变换的函数，输入图像和相应的标签

❷ 根据原始图像大小的 80%选择裁剪大小

❸ 随机裁剪输入图像

❹ 应用一系列变换

### 13.4.3 预处理

`TF.Keras.layers.experimental.preprocessing`模块提供了几个预处理层，这些层提供了在模型中作为预前缀组件执行图像增强的手段。因此，这些操作将在 GPU（或等效）上发生，而不是在 CPU 上游发生。由于预前缀是即插即用的，在训练完成后，可以在将模型部署到生产之前断开这个预前缀组件。

在 TensorFlow 2.2 中，支持平移、缩放和视口不变性的预处理层如下：

+   `CenterCrop`

+   `RandomCrop`

+   `RandomRotation`

+   `随机翻译`

+   `RandomFlip`

在以下示例中，我们将两个预处理层`RandomFlip()`和`RandomTranslation()`作为即插即用的预前缀进行组合，以实现不变性：我们创建一个空的`wrapper`模型，添加即插即用的预前缀，然后添加`model`。对于部署，我们像本章前面所演示的那样，断开即插即用的预前缀。

```
wrapper = Sequential()                                                   ❶
wrapper.add(RandomFlip())                                                ❷
wrapper.add(RandomTranslation(fill_mode='constant', height_factor=0.2,   ❷
                              width_factor=0.2))                         ❷

wrapper.add(model)                                                       ❸
```

❶ 创建包装模型

❷ 添加不变性预前缀

❸ 添加基础模型

## 摘要

+   数据管道的基本组件包括数据存储、数据检索、数据预处理和数据馈送。

+   为了获得最佳的 I/O 性能，如果整个数据集可以适合内存，则在训练期间使用内存中的数据馈送；否则，使用磁盘上的数据馈送。

+   根据数据是否在磁盘上以压缩或未压缩的形式存储，存在额外的空间和时间性能权衡。你可能可以通过基于子群体采样分布的混合方法来平衡这些权衡。

+   如果你处理卫星数据，你需要了解 HDF5 格式。如果你处理医学影像数据，你需要了解 DICOM。

+   图像增强的主要目的是训练一个模型，使其对平移、缩放和视口不变性有更好的泛化能力，以便更好地泛化到训练期间未见过的示例。

+   可以通过使用`tf.data`或 TFX 在模型上游构建数据管道。

+   可以通过使用子类化的 TF.Keras 预处理层在模型下游构建数据管道。

+   预前缀可以被设计为预处理即插即用组件，并在训练和提供期间保持附加。

+   预前缀可以被设计为增强即插即用插件，并在训练期间附加，在推理期间断开。
