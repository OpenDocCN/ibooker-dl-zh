# 第 13 章。使用 TensorFlow 加载和预处理数据

在[第 2 章](ch02.html#project_chapter)中，您看到加载和预处理数据是任何机器学习项目的重要部分。您使用 Pandas 加载和探索（修改后的）加利福尼亚房屋数据集——该数据集存储在 CSV 文件中——并应用 Scikit-Learn 的转换器进行预处理。这些工具非常方便，您可能会经常使用它们，特别是在探索和实验数据时。

然而，在大型数据集上训练 TensorFlow 模型时，您可能更喜欢使用 TensorFlow 自己的数据加载和预处理 API，称为*tf.data*。它能够非常高效地加载和预处理数据，使用多线程和排队从多个文件中并行读取数据，对样本进行洗牌和分批处理等。此外，它可以实时执行所有这些操作——在 GPU 或 TPU 正在训练当前批次数据时，它会在多个 CPU 核心上加载和预处理下一批数据。

tf.data API 允许您处理无法放入内存的数据集，并充分利用硬件资源，从而加快训练速度。tf.data API 可以直接从文本文件（如 CSV 文件）、具有固定大小记录的二进制文件以及使用 TensorFlow 的 TFRecord 格式的二进制文件中读取数据。

TFRecord 是一种灵活高效的二进制格式，通常包含协议缓冲区（一种开源二进制格式）。tf.data API 还支持从 SQL 数据库中读取数据。此外，许多开源扩展可用于从各种数据源中读取数据，例如 Google 的 BigQuery 服务（请参阅[*https://tensorflow.org/io*](https://tensorflow.org/io)）。

Keras 还提供了强大而易于使用的预处理层，可以嵌入到您的模型中：这样，当您将模型部署到生产环境时，它将能够直接摄取原始数据，而无需您添加任何额外的预处理代码。这消除了训练期间使用的预处理代码与生产中使用的预处理代码之间不匹配的风险，这可能会导致*训练/服务偏差*。如果您将模型部署在使用不同编程语言编写的多个应用程序中，您不必多次重新实现相同的预处理代码，这也减少了不匹配的风险。

正如您将看到的，这两个 API 可以联合使用——例如，从 tf.data 提供的高效数据加载和 Keras 预处理层的便利性中受益。

在本章中，我们将首先介绍 tf.data API 和 TFRecord 格式。然后我们将探索 Keras 预处理层以及如何将它们与 tf.data API 一起使用。最后，我们将快速查看一些相关的库，您可能会发现它们在加载和预处理数据时很有用，例如 TensorFlow Datasets 和 TensorFlow Hub。所以，让我们开始吧！

# tf.data API

整个 tf.data API 围绕着 `tf.data.Dataset` 的概念展开：这代表了一系列数据项。通常，您会使用逐渐从磁盘读取数据的数据集，但为了简单起见，让我们使用 `tf.data.Dataset.from_tensor_slices()` 从一个简单的数据张量创建数据集：

```py
>>> import tensorflow as tf
>>> X = tf.range(10)  # any data tensor
>>> dataset = tf.data.Dataset.from_tensor_slices(X)
>>> dataset
<TensorSliceDataset shapes: (), types: tf.int32>
```

`from_tensor_slices()` 函数接受一个张量，并创建一个 `tf.data.Dataset`，其中的元素是沿着第一维度的所有 `X` 的切片，因此这个数据集包含 10 个项目：张量 0、1、2、…​、9。在这种情况下，如果我们使用 `tf.data.Dataset.range(10)`，我们将获得相同的数据集（除了元素将是 64 位整数而不是 32 位整数）。

您可以简单地迭代数据集的项目，如下所示：

```py
>>> for item in dataset:
...     print(item)
...
tf.Tensor(0, shape=(), dtype=int32)
tf.Tensor(1, shape=(), dtype=int32)
[...]
tf.Tensor(9, shape=(), dtype=int32)
```

###### 注意

tf.data API 是一个流式 API：您可以非常高效地迭代数据集的项目，但该 API 不适用于索引或切片。

数据集还可以包含张量的元组，或名称/张量对的字典，甚至是张量的嵌套元组和字典。在对元组、字典或嵌套结构进行切片时，数据集将仅切片它包含的张量，同时保留元组/字典结构。例如：

```py
>>> X_nested = {"a": ([1, 2, 3], [4, 5, 6]), "b": [7, 8, 9]}
>>> dataset = tf.data.Dataset.from_tensor_slices(X_nested)
>>> for item in dataset:
...     print(item)
...
{'a': (<tf.Tensor: [...]=1>, <tf.Tensor: [...]=4>), 'b': <tf.Tensor: [...]=7>}
{'a': (<tf.Tensor: [...]=2>, <tf.Tensor: [...]=5>), 'b': <tf.Tensor: [...]=8>}
{'a': (<tf.Tensor: [...]=3>, <tf.Tensor: [...]=6>), 'b': <tf.Tensor: [...]=9>}
```

## 链接转换

一旦您有了数据集，您可以通过调用其转换方法对其应用各种转换。每个方法都会返回一个新的数据集，因此您可以像这样链接转换（此链在[图13-1](#chaining_transformations_diagram)中有示例）：

```py
>>> dataset = tf.data.Dataset.from_tensor_slices(tf.range(10))
>>> dataset = dataset.repeat(3).batch(7)
>>> for item in dataset:
...     print(item)
...
tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32)
tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int32)
tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int32)
tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
tf.Tensor([8 9], shape=(2,), dtype=int32)
```

在这个例子中，我们首先在原始数据集上调用`repeat()`方法，它返回一个将原始数据集的项目重复三次的新数据集。当然，这不会将所有数据在内存中复制三次！如果您调用此方法而没有参数，新数据集将永远重复源数据集，因此迭代数据集的代码将不得不决定何时停止。

然后我们在这个新数据集上调用`batch()`方法，再次创建一个新数据集。这个新数据集将把前一个数据集的项目分组成七个项目一组的批次。

![mls3 1301](assets/mls3_1301.png)

###### 图13-1\. 链接数据集转换

最后，我们迭代这个最终数据集的项目。`batch()`方法必须输出一个大小为两而不是七的最终批次，但是如果您希望删除这个最终批次，使所有批次具有完全相同的大小，可以调用`batch()`并使用`drop_remainder=True`。

###### 警告

数据集方法*不会*修改数据集，它们会创建新的数据集。因此，请确保保留对这些新数据集的引用（例如，使用`dataset = ...`），否则什么也不会发生。

您还可以通过调用`map()`方法来转换项目。例如，这将创建一个所有批次乘以二的新数据集：

```py
>>> dataset = dataset.map(lambda x: x * 2)  # x is a batch
>>> for item in dataset:
...     print(item)
...
tf.Tensor([ 0  2  4  6  8 10 12], shape=(7,), dtype=int32)
tf.Tensor([14 16 18  0  2  4  6], shape=(7,), dtype=int32)
[...]
```

这个`map()`方法是您将调用的方法，用于对数据进行任何预处理。有时这将包括一些可能相当密集的计算，比如重塑或旋转图像，因此您通常会希望启动多个线程以加快速度。这可以通过将`num_parallel_calls`参数设置为要运行的线程数，或者设置为`tf.data.AUTOTUNE`来完成。请注意，您传递给`map()`方法的函数必须可以转换为TF函数（请参阅[第12章](ch12.html#tensorflow_chapter)）。

还可以使用`filter()`方法简单地过滤数据集。例如，此代码创建一个仅包含总和大于50的批次的数据集：

```py
>>> dataset = dataset.filter(lambda x: tf.reduce_sum(x) > 50)
>>> for item in dataset:
...     print(item)
...
tf.Tensor([14 16 18  0  2  4  6], shape=(7,), dtype=int32)
tf.Tensor([ 8 10 12 14 16 18  0], shape=(7,), dtype=int32)
tf.Tensor([ 2  4  6  8 10 12 14], shape=(7,), dtype=int32)
```

您经常会想查看数据集中的一些项目。您可以使用`take()`方法来实现：

```py
>>> for item in dataset.take(2):
...     print(item)
...
tf.Tensor([14 16 18  0  2  4  6], shape=(7,), dtype=int32)
tf.Tensor([ 8 10 12 14 16 18  0], shape=(7,), dtype=int32)
```

## 数据洗牌

正如我们在[第4章](ch04.html#linear_models_chapter)中讨论的，梯度下降在训练集中的实例是独立且同分布（IID）时效果最好。确保这一点的一个简单方法是对实例进行洗牌，使用`shuffle()`方法。它将创建一个新数据集，首先用源数据集的前几个项目填充缓冲区。然后，每当需要一个项目时，它将从缓冲区随机取出一个项目，并用源数据集中的新项目替换它，直到完全迭代源数据集。在这一点上，它将继续从缓冲区随机取出项目，直到缓冲区为空。您必须指定缓冲区大小，并且很重要的是要足够大，否则洗牌效果不会很好。⁠^([1](ch13.html#idm45720190533488)) 只是不要超出您拥有的RAM量，尽管即使您有很多RAM，也没有必要超出数据集的大小。如果您希望每次运行程序时都获得相同的随机顺序，可以提供一个随机种子。例如，以下代码创建并显示一个包含0到9的整数，重复两次，使用大小为4的缓冲区和随机种子42进行洗牌，并使用批次大小为7进行批处理的数据集：

```py
>>> dataset = tf.data.Dataset.range(10).repeat(2)
>>> dataset = dataset.shuffle(buffer_size=4, seed=42).batch(7)
>>> for item in dataset:
...     print(item)
...
tf.Tensor([3 0 1 6 2 5 7], shape=(7,), dtype=int64)
tf.Tensor([8 4 1 9 4 2 3], shape=(7,), dtype=int64)
tf.Tensor([7 5 0 8 9 6], shape=(6,), dtype=int64)
```

###### 提示

如果在打乱的数据集上调用`repeat()`，默认情况下它将在每次迭代时生成一个新的顺序。这通常是个好主意，但是如果您希望在每次迭代中重复使用相同的顺序（例如，用于测试或调试），可以在调用`shuffle()`时设置`reshuffle_each_​itera⁠tion=False`。

对于一个无法放入内存的大型数据集，这种简单的打乱缓冲区方法可能不够，因为缓冲区相对于数据集来说很小。一个解决方案是对源数据本身进行打乱（例如，在Linux上可以使用`shuf`命令对文本文件进行打乱）。这将显著改善打乱效果！即使源数据已经被打乱，通常也会希望再次打乱，否则每个时期将重复相同的顺序，模型可能会出现偏差（例如，由于源数据顺序中偶然存在的一些虚假模式）。为了进一步打乱实例，一个常见的方法是将源数据拆分为多个文件，然后在训练过程中以随机顺序读取它们。然而，位于同一文件中的实例仍然会相互靠近。为了避免这种情况，您可以随机选择多个文件并同时读取它们，交错它们的记录。然后在此基础上使用`shuffle()`方法添加一个打乱缓冲区。如果这听起来很费力，不用担心：tf.data API可以在几行代码中实现所有这些。让我们看看您可以如何做到这一点。

## 从多个文件中交错行

首先，假设您已经加载了加利福尼亚房屋数据集，对其进行了打乱（除非已经打乱），并将其分为训练集、验证集和测试集。然后将每个集合分成许多CSV文件，每个文件看起来像这样（每行包含八个输入特征加上目标中位房价）：

```py
MedInc,HouseAge,AveRooms,AveBedrms,Popul…,AveOccup,Lat…,Long…,MedianHouseValue
3.5214,15.0,3.050,1.107,1447.0,1.606,37.63,-122.43,1.442
5.3275,5.0,6.490,0.991,3464.0,3.443,33.69,-117.39,1.687
3.1,29.0,7.542,1.592,1328.0,2.251,38.44,-122.98,1.621
[...]
```

假设`train_filepaths`包含训练文件路径列表（您还有`valid_filepaths`和`test_filepaths`）：

```py
>>> train_filepaths
['datasets/housing/my_train_00.csv', 'datasets/housing/my_train_01.csv', ...]
```

或者，您可以使用文件模式；例如，`train_filepaths =` `"datasets/housing/my_train_*.csv"`。现在让我们创建一个仅包含这些文件路径的数据集：

```py
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
```

默认情况下，`list_files()`函数返回一个打乱文件路径的数据集。一般来说这是件好事，但是如果出于某种原因不想要这样，可以设置`shuffle=False`。

接下来，您可以调用`interleave()`方法一次从五个文件中读取并交错它们的行。您还可以使用`skip()`方法跳过每个文件的第一行（即标题行）：

```py
n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers)
```

`interleave()`方法将创建一个数据集，从`filepath_dataset`中提取五个文件路径，对于每个文件路径，它将调用您提供的函数（在本例中是lambda函数）来创建一个新的数据集（在本例中是`TextLineDataset`）。清楚地说，在这个阶段总共会有七个数据集：文件路径数据集、交错数据集以及交错数据集内部创建的五个`TextLineDataset`。当您迭代交错数据集时，它将循环遍历这五个`TextLineDataset`，从每个数据集中逐行读取，直到所有数据集都用完。然后它将从`filepath_dataset`中获取下一个五个文件路径，并以相同的方式交错它们，依此类推，直到文件路径用完。为了使交错效果最佳，最好拥有相同长度的文件；否则最长文件的末尾将不会被交错。

默认情况下，`interleave()`不使用并行处理；它只是顺序地从每个文件中一次读取一行。如果您希望实际并行读取文件，可以将`interleave()`方法的`num_parallel_calls`参数设置为您想要的线程数（请记住，`map()`方法也有这个参数）。甚至可以将其设置为`tf.data.AUTOTUNE`，让TensorFlow根据可用的CPU动态选择正确的线程数。现在让我们看看数据集现在包含什么：

```py
>>> for line in dataset.take(5):
...     print(line)
...
tf.Tensor(b'4.5909,16.0,[...],33.63,-117.71,2.418', shape=(), dtype=string)
tf.Tensor(b'2.4792,24.0,[...],34.18,-118.38,2.0', shape=(), dtype=string)
tf.Tensor(b'4.2708,45.0,[...],37.48,-122.19,2.67', shape=(), dtype=string)
tf.Tensor(b'2.1856,41.0,[...],32.76,-117.12,1.205', shape=(), dtype=string)
tf.Tensor(b'4.1812,52.0,[...],33.73,-118.31,3.215', shape=(), dtype=string)
```

这些是随机选择的五个 CSV 文件的第一行（忽略标题行）。看起来不错！

###### 注意

可以将文件路径列表传递给 `TextLineDataset` 构造函数：它将按顺序遍历每个文件的每一行。如果还将 `num_parallel_reads` 参数设置为大于一的数字，那么数据集将并行读取该数量的文件，并交错它们的行（无需调用 `interleave()` 方法）。但是，它不会对文件进行洗牌，也不会跳过标题行。

## 数据预处理

现在我们有一个返回每个实例的住房数据集，其中包含一个字节字符串的张量，我们需要进行一些预处理，包括解析字符串和缩放数据。让我们实现一些自定义函数来执行这些预处理：

```py
X_mean, X_std = [...]  # mean and scale of each feature in the training set
n_inputs = 8

def parse_csv_line(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    return tf.stack(fields[:-1]), tf.stack(fields[-1:])

def preprocess(line):
    x, y = parse_csv_line(line)
    return (x - X_mean) / X_std, y
```

让我们逐步解释这段代码：

+   首先，代码假设我们已经预先计算了训练集中每个特征的均值和标准差。`X_mean` 和 `X_std` 只是包含八个浮点数的 1D 张量（或 NumPy 数组），每个输入特征一个。可以使用 Scikit-Learn 的 `StandardScaler` 在数据集的足够大的随机样本上完成这个操作。在本章的后面，我们将使用 Keras 预处理层来代替。

+   `parse_csv_line()` 函数接受一个 CSV 行并对其进行解析。为了帮助实现这一点，它使用 `tf.io.decode_csv()` 函数，该函数接受两个参数：第一个是要解析的行，第二个是包含 CSV 文件中每列的默认值的数组。这个数组（`defs`）告诉 TensorFlow 不仅每列的默认值是什么，还告诉它列的数量和类型。在这个例子中，我们告诉它所有特征列都是浮点数，缺失值应默认为零，但我们为最后一列（目标）提供了一个空的 `tf.float32` 类型的默认值数组：该数组告诉 TensorFlow 这一列包含浮点数，但没有默认值，因此如果遇到缺失值，它将引发异常。

+   `tf.io.decode_csv()` 函数返回一个标量张量列表（每列一个），但我们需要返回一个 1D 张量数组。因此，我们对除最后一个（目标）之外的所有张量调用 `tf.stack()`：这将这些张量堆叠成一个 1D 数组。然后我们对目标值做同样的操作：这将使其成为一个包含单个值的 1D 张量数组，而不是标量张量。`tf.io.decode_csv()` 函数完成后，它将返回输入特征和目标。

+   最后，自定义的 `preprocess()` 函数只调用 `parse_csv_line()` 函数，通过减去特征均值然后除以特征标准差来缩放输入特征，并返回一个包含缩放特征和目标的元组。

让我们测试这个预处理函数：

```py
>>> preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782')
(<tf.Tensor: shape=(8,), dtype=float32, numpy=
 array([ 0.16579159,  1.216324  , -0.05204564, -0.39215982, -0.5277444 ,
 -0.2633488 ,  0.8543046 , -1.3072058 ], dtype=float32)>,
 <tf.Tensor: shape=(1,), dtype=float32, numpy=array([2.782], dtype=float32)>)
```

看起来不错！`preprocess()` 函数可以将一个实例从字节字符串转换为一个漂亮的缩放张量，带有相应的标签。我们现在可以使用数据集的 `map()` 方法将 `preprocess()` 函数应用于数据集中的每个样本。

## 将所有内容放在一起

为了使代码更具重用性，让我们将迄今为止讨论的所有内容放在另一个辅助函数中；它将创建并返回一个数据集，该数据集将高效地从多个 CSV 文件中加载加利福尼亚房屋数据，对其进行预处理、洗牌和分批处理（参见[图 13-2](#input_pipeline_diagram)）：

```py
def csv_reader_dataset(filepaths, n_readers=5, n_read_threads=None,
                       n_parse_threads=5, shuffle_buffer_size=10_000, seed=42,
                       batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths, seed=seed)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.batch(batch_size).prefetch(1)
```

请注意，我们在最后一行使用了 `prefetch()` 方法。这对性能很重要，你现在会看到。

![mls3 1302](assets/mls3_1302.png)

###### 图 13-2\. 从多个 CSV 文件加载和预处理数据

## 预取

通过在自定义`csv_reader_dataset()`函数末尾调用`prefetch(1)`，我们正在创建一个数据集，该数据集将尽力始终领先一个批次。换句话说，当我们的训练算法在处理一个批次时，数据集将已经在并行工作，准备好获取下一个批次（例如，从磁盘读取数据并对其进行预处理）。这可以显著提高性能，如[图13-3](#prefetching_diagram)所示。

如果我们还确保加载和预处理是多线程的（通过在调用`interleave()`和`map()`时设置`num_parallel_calls`），我们可以利用多个CPU核心，希望准备一个数据批次的时间比在GPU上运行训练步骤要短：这样GPU将几乎100%利用（除了从CPU到GPU的数据传输时间）[3]，训练将运行得更快。

![mls3 1303](assets/mls3_1303.png)

###### 图13-3。通过预取，CPU和GPU并行工作：当GPU处理一个批次时，CPU处理下一个批次

###### 提示

如果您计划购买GPU卡，其处理能力和内存大小当然非常重要（特别是对于大型计算机视觉或自然语言处理模型，大量的RAM至关重要）。对于良好性能同样重要的是GPU的*内存带宽*；这是它每秒可以将多少千兆字节的数据进出其RAM。

如果数据集足够小，可以放入内存，您可以通过使用数据集的`cache()`方法将其内容缓存到RAM来显着加快训练速度。通常应在加载和预处理数据之后，但在洗牌、重复、批处理和预取之前执行此操作。这样，每个实例只会被读取和预处理一次（而不是每个时期一次），但数据仍然会在每个时期以不同的方式洗牌，下一批数据仍然会提前准备好。

您现在已经学会了如何构建高效的输入管道，从多个文本文件加载和预处理数据。我们已经讨论了最常见的数据集方法，但还有一些您可能想看看的方法，例如`concatenate()`、`zip()`、`window()`、`reduce()`、`shard()`、`flat_map()`、`apply()`、`unbatch()`和`padded_batch()`。还有一些更多的类方法，例如`from_generator()`和`from_tensors()`，它们分别从Python生成器或张量列表创建新数据集。请查看API文档以获取更多详细信息。还请注意，`tf.data.experimental`中提供了一些实验性功能，其中许多功能可能会在未来的版本中成为核心API的一部分（例如，请查看`CsvDataset`类，以及`make_csv_dataset()`方法，该方法负责推断每列的类型）。

## 使用数据集与Keras

现在，我们可以使用我们之前编写的自定义`csv_reader_dataset()`函数为训练集、验证集和测试集创建数据集。训练集将在每个时期进行洗牌（请注意，验证集和测试集也将进行洗牌，尽管我们实际上并不需要）：

```py
train_set = csv_reader_dataset(train_filepaths)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)
```

现在，您可以简单地使用这些数据集构建和训练Keras模型。当您调用模型的`fit()`方法时，您传递`train_set`而不是`X_train, y_train`，并传递`validation_data=valid_set`而不是`validation_data=(X_valid, y_valid)`。`fit()`方法将负责每个时期重复训练数据集，每个时期使用不同的随机顺序：

```py
model = tf.keras.Sequential([...])
model.compile(loss="mse", optimizer="sgd")
model.fit(train_set, validation_data=valid_set, epochs=5)
```

同样，您可以将数据集传递给`evaluate()`和`predict()`方法：

```py
test_mse = model.evaluate(test_set)
new_set = test_set.take(3)  # pretend we have 3 new samples
y_pred = model.predict(new_set)  # or you could just pass a NumPy array
```

与其他数据集不同，`new_set`通常不包含标签。如果包含标签，就像这里一样，Keras会忽略它们。请注意，在所有这些情况下，您仍然可以使用NumPy数组而不是数据集（但当然它们需要先加载和预处理）。

如果您想构建自己的自定义训练循环（如[第12章](ch12.html#tensorflow_chapter)中讨论的），您可以很自然地遍历训练集：

```py
n_epochs = 5
for epoch in range(n_epochs):
    for X_batch, y_batch in train_set:
        [...]  # perform one gradient descent step
```

实际上，甚至可以创建一个TF函数（参见[第12章](ch12.html#tensorflow_chapter)），用于整个时期训练模型。这可以真正加快训练速度：

```py
@tf.function
def train_one_epoch(model, optimizer, loss_fn, train_set):
    for X_batch, y_batch in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.mean_squared_error
for epoch in range(n_epochs):
    print("\rEpoch {}/{}".format(epoch + 1, n_epochs), end="")
    train_one_epoch(model, optimizer, loss_fn, train_set)
```

在Keras中，`compile()`方法的`steps_per_execution`参数允许您定义`fit()`方法在每次调用用于训练的`tf.function`时将处理的批次数。默认值只是1，因此如果将其设置为50，您通常会看到显着的性能改进。但是，Keras回调的`on_batch_*()`方法只会在每50批次时调用一次。

恭喜，您现在知道如何使用tf.data API构建强大的输入管道！然而，到目前为止，我们一直在使用常见、简单和方便但不是真正高效的CSV文件，并且不太支持大型或复杂的数据结构（如图像或音频）。因此，让我们看看如何改用TFRecords。

###### 提示

如果您对CSV文件（或者您正在使用的其他格式）感到满意，您不一定*必须*使用TFRecords。俗话说，如果它没有坏，就不要修理！当训练过程中的瓶颈是加载和解析数据时，TFRecords非常有用。

# TFRecord格式

TFRecord格式是TensorFlow存储大量数据并高效读取的首选格式。它是一个非常简单的二进制格式，只包含一系列大小不同的二进制记录（每个记录由长度、用于检查长度是否损坏的CRC校验和、实际数据，最后是数据的CRC校验和组成）。您可以使用`tf.io.TFRecordWriter`类轻松创建TFRecord文件：

```py
with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")
```

然后，您可以使用`tf.data.TFRecordDataset`来读取一个或多个TFRecord文件：

```py
filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)
```

这将输出：

```py
tf.Tensor(b'This is the first record', shape=(), dtype=string)
tf.Tensor(b'And this is the second record', shape=(), dtype=string)
```

###### 提示

默认情况下，`TFRecordDataset`将逐个读取文件，但您可以使其并行读取多个文件，并通过传递文件路径列表给构造函数并将`num_parallel_reads`设置为大于1的数字来交错它们的记录。或者，您可以通过使用`list_files()`和`interleave()`来获得与我们之前读取多个CSV文件相同的结果。

## 压缩的TFRecord文件

有时将TFRecord文件压缩可能很有用，特别是如果它们需要通过网络连接加载。您可以通过设置`options`参数创建一个压缩的TFRecord文件：

```py
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
    f.write(b"Compress, compress, compress!")
```

在读取压缩的TFRecord文件时，您需要指定压缩类型：

```py
dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"],
                                  compression_type="GZIP")
```

## 协议缓冲区简介

尽管每个记录可以使用您想要的任何二进制格式，但TFRecord文件通常包含序列化的协议缓冲区（也称为*protobufs*）。这是一个在2001年由谷歌开发的便携式、可扩展和高效的二进制格式，并于2008年开源；protobufs现在被广泛使用，特别是在[grpc](https://grpc.io)中，谷歌的远程过程调用系统。它们使用一个看起来像这样的简单语言进行定义：

```py
syntax = "proto3";
message Person {
    string name = 1;
    int32 id = 2;
    repeated string email = 3;
}
```

这个protobuf定义表示我们正在使用protobuf格式的第3版，并且指定每个`Person`对象（可选）可能具有一个字符串类型的`name`、一个int32类型的`id`，以及零个或多个字符串类型的`email`字段。数字`1`、`2`和`3`是字段标识符：它们将在每个记录的二进制表示中使用。一旦你在*.proto*文件中有了一个定义，你就可以编译它。这需要使用protobuf编译器`protoc`在Python（或其他语言）中生成访问类。请注意，你通常在TensorFlow中使用的protobuf定义已经为你编译好了，并且它们的Python类是TensorFlow库的一部分，因此你不需要使用`protoc`。你只需要知道如何在Python中*使用*protobuf访问类。为了说明基础知识，让我们看一个简单的示例，使用为`Person`protobuf生成的访问类（代码在注释中有解释）：

```py
>>> from person_pb2 import Person  # import the generated access class
>>> person = Person(name="Al", id=123, email=["a@b.com"])  # create a Person
>>> print(person)  # display the Person
name: "Al"
id: 123
email: "a@b.com"
>>> person.name  # read a field
'Al'
>>> person.name = "Alice"  # modify a field
>>> person.email[0]  # repeated fields can be accessed like arrays
'a@b.com'
>>> person.email.append("c@d.com")  # add an email address
>>> serialized = person.SerializeToString()  # serialize person to a byte string
>>> serialized
b'\n\x05Alice\x10{\x1a\x07a@b.com\x1a\x07c@d.com'
>>> person2 = Person()  # create a new Person
>>> person2.ParseFromString(serialized)  # parse the byte string (27 bytes long)
27
>>> person == person2  # now they are equal
True
```

简而言之，我们导入由`protoc`生成的`Person`类，创建一个实例并对其进行操作，可视化它并读取和写入一些字段，然后使用`SerializeToString()`方法对其进行序列化。这是准备保存或通过网络传输的二进制数据。当读取或接收这些二进制数据时，我们可以使用`ParseFromString()`方法进行解析，并获得被序列化的对象的副本。

你可以将序列化的`Person`对象保存到TFRecord文件中，然后加载和解析它：一切都会正常工作。然而，`ParseFromString()`不是一个TensorFlow操作，所以你不能在tf.data管道中的预处理函数中使用它（除非将其包装在`tf.py_function()`操作中，这会使代码变慢且不太可移植，正如你在[第12章](ch12.html#tensorflow_chapter)中看到的）。然而，你可以使用`tf.io.decode_proto()`函数，它可以解析任何你想要的protobuf，只要你提供protobuf定义（请参考笔记本中的示例）。也就是说，在实践中，你通常会希望使用TensorFlow提供的专用解析操作的预定义protobuf。现在让我们来看看这些预定义的protobuf。

## TensorFlow Protobufs

TFRecord文件中通常使用的主要protobuf是`Example`protobuf，它表示数据集中的一个实例。它包含一个命名特征列表，其中每个特征可以是一个字节字符串列表、一个浮点数列表或一个整数列表。以下是protobuf定义（来自TensorFlow源代码）：

```py
syntax = "proto3";
message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true]; }
message Int64List { repeated int64 value = 1 [packed = true]; }
message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
message Features { map<string, Feature> feature = 1; };
message Example { Features features = 1; };
```

`BytesList`、`FloatList`和`Int64List`的定义足够简单明了。请注意，对于重复的数值字段，使用`[packed = true]`进行更有效的编码。`Feature`包含一个`BytesList`、一个`FloatList`或一个`Int64List`。一个`Features`（带有`s`）包含一个将特征名称映射到相应特征值的字典。最后，一个`Example`只包含一个`Features`对象。

###### 注意

为什么会定义`Example`，因为它只包含一个`Features`对象？嗯，TensorFlow的开发人员可能有一天决定向其中添加更多字段。只要新的`Example`定义仍然包含相同ID的`features`字段，它就是向后兼容的。这种可扩展性是protobuf的一个伟大特性。

这是你如何创建一个代表同一个人的`tf.train.Example`：

```py
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com",
                                                          b"c@d.com"]))
        }))
```

这段代码有点冗长和重复，但你可以很容易地将其包装在一个小的辅助函数中。现在我们有了一个`Example` protobuf，我们可以通过调用其`SerializeToString()`方法将其序列化，然后将生成的数据写入TFRecord文件。让我们假装写入五次，以假装我们有几个联系人：

```py
with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    for _ in range(5):
        f.write(person_example.SerializeToString())
```

通常，您会写比五个`Example`更多的内容！通常情况下，您会创建一个转换脚本，从当前格式（比如CSV文件）读取数据，为每个实例创建一个`Example` protobuf，将它们序列化，并保存到几个TFRecord文件中，最好在此过程中对它们进行洗牌。这需要一些工作，所以再次确保这确实是必要的（也许您的流水线使用CSV文件运行良好）。

现在我们有一个包含多个序列化`Example`的漂亮TFRecord文件，让我们尝试加载它。

## 加载和解析示例

为了加载序列化的`Example` protobufs，我们将再次使用`tf.data.TFRecordDataset`，并使用`tf.io.parse_single_example()`解析每个`Example`。它至少需要两个参数：包含序列化数据的字符串标量张量，以及每个特征的描述。描述是一个字典，将每个特征名称映射到`tf.io.FixedLenFeature`描述符，指示特征的形状、类型和默认值，或者`tf.io.VarLenFeature`描述符，仅指示特征列表的长度可能变化的类型（例如`"emails"`特征）。

以下代码定义了一个描述字典，然后创建了一个`TFRecordDataset`，并对其应用了一个自定义预处理函数，以解析该数据集包含的每个序列化`Example` protobuf：

```py
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

def parse(serialized_example):
    return tf.io.parse_single_example(serialized_example, feature_description)

dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).map(parse)
for parsed_example in dataset:
    print(parsed_example)
```

固定长度的特征被解析为常规张量，但变长特征被解析为稀疏张量。您可以使用`tf.sparse.to_dense()`将稀疏张量转换为密集张量，但在这种情况下，更简单的方法是直接访问其值：

```py
>>> tf.sparse.to_dense(parsed_example["emails"], default_value=b"")
<tf.Tensor: [...] dtype=string, numpy=array([b'a@b.com', b'c@d.com'], [...])>
>>> parsed_example["emails"].values
<tf.Tensor: [...] dtype=string, numpy=array([b'a@b.com', b'c@d.com'], [...])>
```

您可以使用`tf.io.parse_example()`批量解析示例，而不是使用`tf.io.parse_single_example()`逐个解析它们：

```py
def parse(serialized_examples):
    return tf.io.parse_example(serialized_examples, feature_description)

dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).batch(2).map(parse)
for parsed_examples in dataset:
    print(parsed_examples)  # two examples at a time
```

最后，`BytesList`可以包含您想要的任何二进制数据，包括任何序列化对象。例如，您可以使用`tf.io.encode_jpeg()`使用JPEG格式对图像进行编码，并将这些二进制数据放入`BytesList`中。稍后，当您的代码读取TFRecord时，它将从解析`Example`开始，然后需要调用`tf.io.decode_jpeg()`来解析数据并获取原始图像（或者您可以使用`tf.io.decode_image()`，它可以解码任何BMP、GIF、JPEG或PNG图像）。您还可以通过使用`tf.io.serialize_tensor()`对张量进行序列化，然后将生成的字节字符串放入`BytesList`特征中，将任何您想要的张量存储在`BytesList`中。稍后，当您解析TFRecord时，您可以使用`tf.io.parse_tensor()`解析这些数据。请参阅本章的笔记本[*https://homl.info/colab3*](https://homl.info/colab3) ，了解在TFRecord文件中存储图像和张量的示例。

正如您所看到的，`Example` protobuf非常灵活，因此对于大多数用例来说可能已经足够了。但是，当您处理列表列表时，可能会有些繁琐。例如，假设您想对文本文档进行分类。每个文档可以表示为一个句子列表，其中每个句子表示为一个单词列表。也许每个文档还有一个评论列表，其中每个评论表示为一个单词列表。还可能有一些上下文数据，比如文档的作者、标题和发布日期。TensorFlow的`SequenceExample` protobuf就是为这种用例而设计的。

## 使用SequenceExample Protobuf处理列表列表

这是`SequenceExample` protobuf的定义：

```py
message FeatureList { repeated Feature feature = 1; };
message FeatureLists { map<string, FeatureList> feature_list = 1; };
message SequenceExample {
    Features context = 1;
    FeatureLists feature_lists = 2;
};
```

`SequenceExample`包含一个`Features`对象用于上下文数据和一个包含一个或多个命名`FeatureList`对象（例如，一个名为`"content"`的`FeatureList`和另一个名为`"comments"`的`FeatureList`）的`FeatureLists`对象。每个`FeatureList`包含一个`Feature`对象列表，每个`Feature`对象可能是字节字符串列表、64位整数列表或浮点数列表（在此示例中，每个`Feature`可能代表一个句子或评论，可能以单词标识符列表的形式）。构建`SequenceExample`、序列化它并解析它类似于构建、序列化和解析`Example`，但您必须使用`tf.io.parse_single_sequence_example()`来解析单个`SequenceExample`或`tf.io.parse_sequence_example()`来解析批处理。这两个函数返回一个包含上下文特征（作为字典）和特征列表（也作为字典）的元组。如果特征列表包含不同大小的序列（如前面的示例），您可能希望使用`tf.RaggedTensor.from_sparse()`将它们转换为不规则张量（请参阅完整代码的笔记本）：

```py
parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
    serialized_sequence_example, context_feature_descriptions,
    sequence_feature_descriptions)
parsed_content = tf.RaggedTensor.from_sparse(parsed_feature_lists["content"])
```

现在您已经知道如何使用tf.data API、TFRecords和protobufs高效存储、加载、解析和预处理数据，是时候将注意力转向Keras预处理层了。

# Keras预处理层

为神经网络准备数据通常需要对数值特征进行归一化、对分类特征和文本进行编码、裁剪和调整图像等。有几种选项：

+   预处理可以提前在准备训练数据文件时完成，使用您喜欢的任何工具，如NumPy、Pandas或Scikit-Learn。您需要在生产中应用完全相同的预处理步骤，以确保您的生产模型接收到与训练时相似的预处理输入。

+   或者，您可以在加载数据时使用tf.data进行即时预处理，通过使用该数据集的`map()`方法对数据集的每个元素应用预处理函数，就像本章前面所做的那样。同样，您需要在生产中应用相同的预处理步骤。

+   最后一种方法是直接在模型内部包含预处理层，这样它可以在训练期间即时预处理所有输入数据，然后在生产中使用相同的预处理层。本章的其余部分将讨论这种最后一种方法。

Keras提供了许多预处理层，您可以将其包含在模型中：它们可以应用于数值特征、分类特征、图像和文本。我们将在接下来的部分中讨论数值和分类特征，以及基本文本预处理，我们将在[第14章](ch14.html#cnn_chapter)中涵盖图像预处理，以及在[第16章](ch16.html#nlp_chapter)中涵盖更高级的文本预处理。

## 归一化层

正如我们在[第10章](ch10.html#ann_chapter)中看到的，Keras提供了一个`Normalization`层，我们可以用来标准化输入特征。我们可以在创建层时指定每个特征的均值和方差，或者更简单地在拟合模型之前将训练集传递给该层的`adapt()`方法，以便该层可以在训练之前自行测量特征的均值和方差：

```py
norm_layer = tf.keras.layers.Normalization()
model = tf.keras.models.Sequential([
    norm_layer,
    tf.keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=2e-3))
norm_layer.adapt(X_train)  # computes the mean and variance of every feature
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5)
```

###### 提示

传递给`adapt()`方法的数据样本必须足够大，以代表您的数据集，但不必是完整的训练集：对于`Normalization`层，从训练集中随机抽取的几百个实例通常足以获得特征均值和方差的良好估计。

由于我们在模型中包含了`Normalization`层，现在我们可以将这个模型部署到生产环境中，而不必再担心归一化的问题：模型会自动处理（参见[图13-4](#preprocessing_in_model_diagram)）。太棒了！这种方法完全消除了预处理不匹配的风险，当人们尝试为训练和生产维护不同的预处理代码，但更新其中一个并忘记更新另一个时，就会发生这种情况。生产模型最终会接收到以其不期望的方式预处理的数据。如果他们幸运的话，会得到一个明显的错误。如果不幸的话，模型的准确性会悄悄下降。

![mls3 1304](assets/mls3_1304.png)

###### 图13-4。在模型中包含预处理层

直接在模型中包含预处理层很简单明了，但会减慢训练速度（在`Normalization`层的情况下只会稍微减慢）：实际上，由于预处理是在训练过程中实时进行的，每个时期只会发生一次。我们可以通过在训练之前仅对整个训练集进行一次归一化来做得更好。为此，我们可以像使用Scikit-Learn的`StandardScaler`一样单独使用`Normalization`层：

```py
norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_train)
X_train_scaled = norm_layer(X_train)
X_valid_scaled = norm_layer(X_valid)
```

现在我们可以在经过缩放的数据上训练模型，这次不需要`Normalization`层：

```py
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=2e-3))
model.fit(X_train_scaled, y_train, epochs=5,
          validation_data=(X_valid_scaled, y_valid))
```

很好！这应该会加快训练速度。但是现在当我们将模型部署到生产环境时，模型不会对其输入进行预处理。为了解决这个问题，我们只需要创建一个新模型，将适应的`Normalization`层和刚刚训练的模型包装在一起。然后我们可以将这个最终模型部署到生产环境中，它将负责对其输入进行预处理和进行预测（参见[图13-5](#optimized_preprocessing_in_model_diagram)）：

```py
final_model = tf.keras.Sequential([norm_layer, model])
X_new = X_test[:3]  # pretend we have a few new instances (unscaled)
y_pred = final_model(X_new)  # preprocesses the data and makes predictions
```

![mls3 1305](assets/mls3_1305.png)

###### 图13-5。在训练之前仅对数据进行一次预处理，然后将这些层部署到最终模型中

现在我们拥有了最佳的两种方式：训练很快，因为我们只在训练开始前对数据进行一次预处理，而最终模型可以在运行时对其输入进行预处理，而不会有任何预处理不匹配的风险。

此外，Keras预处理层与tf.data API很好地配合。例如，可以将`tf.data.Dataset`传递给预处理层的`adapt()`方法。还可以使用数据集的`map()`方法将Keras预处理层应用于`tf.data.Dataset`。例如，以下是如何将适应的`Normalization`层应用于数据集中每个批次的输入特征的方法：

```py
dataset = dataset.map(lambda X, y: (norm_layer(X), y))
```

最后，如果您需要比Keras预处理层提供的更多特性，您可以随时编写自己的Keras层，就像我们在[第12章](ch12.html#tensorflow_chapter)中讨论的那样。例如，如果`Normalization`层不存在，您可以使用以下自定义层获得类似的结果：

```py
import numpy as np

class MyNormalization(tf.keras.layers.Layer):
    def adapt(self, X):
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True)

    def call(self, inputs):
        eps = tf.keras.backend.epsilon()  # a small smoothing term
        return (inputs - self.mean_) / (self.std_ + eps)
```

接下来，让我们看看另一个用于数值特征的Keras预处理层：`Discretization`层。

## Discretization层

`Discretization`层的目标是通过将值范围（称为箱）映射到类别，将数值特征转换为分类特征。这对于具有多峰分布的特征或与目标具有高度非线性关系的特征有时是有用的。例如，以下代码将数值`age`特征映射到三个类别，小于18岁，18到50岁（不包括），50岁或以上：

```py
>>> age = tf.constant([[10.], [93.], [57.], [18.], [37.], [5.]])
>>> discretize_layer = tf.keras.layers.Discretization(bin_boundaries=[18., 50.])
>>> age_categories = discretize_layer(age)
>>> age_categories
<tf.Tensor: shape=(6, 1), dtype=int64, numpy=array([[0],[2],[2],[1],[1],[0]])>
```

在这个例子中，我们提供了期望的分箱边界。如果你愿意，你可以提供你想要的箱数，然后调用层的`adapt()`方法，让它根据值的百分位数找到合适的箱边界。例如，如果我们设置`num_bins=3`，那么箱边界将位于第33和第66百分位数之下的值（在这个例子中，值为10和37）：

```py
>>> discretize_layer = tf.keras.layers.Discretization(num_bins=3)
>>> discretize_layer.adapt(age)
>>> age_categories = discretize_layer(age)
>>> age_categories
<tf.Tensor: shape=(6, 1), dtype=int64, numpy=array([[1],[2],[2],[1],[2],[0]])>
```

通常不应将诸如此类的类别标识符直接传递给神经网络，因为它们的值无法有意义地进行比较。相反，它们应该被编码，例如使用独热编码。现在让我们看看如何做到这一点。

## CategoryEncoding层

当只有少量类别（例如，少于十几个或二十个）时，独热编码通常是一个不错的选择（如[第2章](ch02.html#project_chapter)中讨论的）。为此，Keras提供了`CategoryEncoding`层。例如，让我们对刚刚创建的`age_categories`特征进行独热编码：

```py
>>> onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3)
>>> onehot_layer(age_categories)
<tf.Tensor: shape=(6, 3), dtype=float32, numpy=
array([[0., 1., 0.],
 [0., 0., 1.],
 [0., 0., 1.],
 [0., 1., 0.],
 [0., 0., 1.],
 [1., 0., 0.]], dtype=float32)>
```

如果尝试一次对多个分类特征进行编码（只有当它们都使用相同的类别时才有意义），`CategoryEncoding`类将默认执行*多热编码*：输出张量将包含每个输入特征中存在的每个类别的1。例如：

```py
>>> two_age_categories = np.array([[1, 0], [2, 2], [2, 0]])
>>> onehot_layer(two_age_categories)
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[1., 1., 0.],
 [0., 0., 1.],
 [1., 0., 1.]], dtype=float32)>
```

如果您认为知道每个类别出现的次数是有用的，可以在创建`CategoryEncoding`层时设置`output_mode="count"`，在这种情况下，输出张量将包含每个类别的出现次数。在前面的示例中，输出将与之前相同，只是第二行将变为`[0., 0., 2.]`。

请注意，多热编码和计数编码都会丢失信息，因为无法知道每个活动类别来自哪个特征。例如，`[0, 1]`和`[1, 0]`都被编码为`[1., 1., 0.]`。如果要避免这种情况，那么您需要分别对每个特征进行独热编码，然后连接输出。这样，`[0, 1]`将被编码为`[1., 0., 0., 0., 1., 0.]`，`[1, 0]`将被编码为`[0., 1., 0., 1., 0., 0.]`。您可以通过调整类别标识符来获得相同的结果，以便它们不重叠。例如：

```py
>>> onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3 + 3)
>>> onehot_layer(two_age_categories + [0, 3])  # adds 3 to the second feature
<tf.Tensor: shape=(3, 6), dtype=float32, numpy=
array([[0., 1., 0., 1., 0., 0.],
 [0., 0., 1., 0., 0., 1.],
 [0., 0., 1., 1., 0., 0.]], dtype=float32)>
```

在此输出中，前三列对应于第一个特征，最后三列对应于第二个特征。这使模型能够区分这两个特征。但是，这也增加了馈送到模型的特征数量，因此需要更多的模型参数。很难事先知道单个多热编码还是每个特征的独热编码哪个效果最好：这取决于任务，您可能需要测试两种选项。

现在您可以使用独热编码或多热编码对分类整数特征进行编码。但是对于分类文本特征呢？为此，您可以使用`StringLookup`层。

## StringLookup层

让我们使用Keras的`StringLookup`层对`cities`特征进行独热编码：

```py
>>> cities = ["Auckland", "Paris", "Paris", "San Francisco"]
>>> str_lookup_layer = tf.keras.layers.StringLookup()
>>> str_lookup_layer.adapt(cities)
>>> str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])
<tf.Tensor: shape=(4, 1), dtype=int64, numpy=array([[1], [3], [3], [0]])>
```

我们首先创建一个`StringLookup`层，然后将其适应到数据：它发现有三个不同的类别。然后我们使用该层对一些城市进行编码。默认情况下，它们被编码为整数。未知类别被映射为0，就像在这个例子中的“Montreal”一样。已知类别从最常见的类别开始编号，从最常见到最不常见。

方便的是，当创建`StringLookup`层时设置`output_mode="one_hot"`，它将为每个类别输出一个独热向量，而不是一个整数：

```py
>>> str_lookup_layer = tf.keras.layers.StringLookup(output_mode="one_hot")
>>> str_lookup_layer.adapt(cities)
>>> str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])
<tf.Tensor: shape=(4, 4), dtype=float32, numpy=
array([[0., 1., 0., 0.],
 [0., 0., 0., 1.],
 [0., 0., 0., 1.],
 [1., 0., 0., 0.]], dtype=float32)>
```

###### 提示

Keras还包括一个`IntegerLookup`层，其功能类似于`StringLookup`层，但输入为整数，而不是字符串。

如果训练集非常大，可能会方便地将层适应于训练集的随机子集。在这种情况下，层的`adapt()`方法可能会错过一些较少见的类别。默认情况下，它会将它们全部映射到类别0，使它们在模型中无法区分。为了减少这种风险（同时仅在训练集的子集上调整层），您可以将`num_oov_indices`设置为大于1的整数。这是要使用的未知词汇（OOV）桶的数量：每个未知类别将使用哈希函数对OOV桶的数量取模，伪随机地映射到其中一个OOV桶。这将使模型能够区分至少一些罕见的类别。例如：

```py
>>> str_lookup_layer = tf.keras.layers.StringLookup(num_oov_indices=5)
>>> str_lookup_layer.adapt(cities)
>>> str_lookup_layer([["Paris"], ["Auckland"], ["Foo"], ["Bar"], ["Baz"]])
<tf.Tensor: shape=(4, 1), dtype=int64, numpy=array([[5], [7], [4], [3], [4]])>
```

由于有五个OOV桶，第一个已知类别的ID现在是5（“巴黎”）。但是，`"Foo"`、`"Bar"`和`"Baz"`是未知的，因此它们各自被映射到OOV桶中的一个。 `"Bar"`有自己的专用桶（ID为3），但不幸的是，`"Foo"`和`"Baz"`被映射到相同的桶中（ID为4），因此它们在模型中保持不可区分。这被称为*哈希碰撞*。减少碰撞风险的唯一方法是增加OOV桶的数量。但是，这也会增加总类别数，这将需要更多的RAM和额外的模型参数，一旦类别被独热编码。因此，不要将该数字增加得太多。

将类别伪随机映射到桶中的这种想法称为*哈希技巧*。Keras提供了一个专用的层，就是`Hashing`层。

## 哈希层

对于每个类别，Keras的`Hashing`层计算一个哈希值，取模于桶（或“bin”）的数量。映射完全是伪随机的，但在运行和平台之间是稳定的（即，只要桶的数量不变，相同的类别将始终被映射到相同的整数）。例如，让我们使用`Hashing`层来编码一些城市：

```py
>>> hashing_layer = tf.keras.layers.Hashing(num_bins=10)
>>> hashing_layer([["Paris"], ["Tokyo"], ["Auckland"], ["Montreal"]])
<tf.Tensor: shape=(4, 1), dtype=int64, numpy=array([[0], [1], [9], [1]])>
```

这个层的好处是它根本不需要适应，这有时可能很有用，特别是在核外设置中（当数据集太大而无法放入内存时）。然而，我们再次遇到了哈希碰撞：“东京”和“蒙特利尔”被映射到相同的ID，使它们在模型中无法区分。因此，通常最好坚持使用`StringLookup`层。

现在让我们看另一种编码类别的方法：可训练的嵌入。

## 使用嵌入编码分类特征

嵌入是一种高维数据（例如类别或词汇中的单词）的密集表示。如果有50,000个可能的类别，那么独热编码将产生一个50,000维的稀疏向量（即，大部分为零）。相比之下，嵌入将是一个相对较小的密集向量；例如，只有100个维度。

在深度学习中，嵌入通常是随机初始化的，然后通过梯度下降与其他模型参数一起训练。例如，在加利福尼亚住房数据集中，`"NEAR BAY"`类别最初可以由一个随机向量表示，例如`[0.131, 0.890]`，而`"NEAR OCEAN"`类别可能由另一个随机向量表示，例如`[0.631, 0.791]`。在这个例子中，我们使用了2D嵌入，但维度的数量是一个可以调整的超参数。

由于这些嵌入是可训练的，它们在训练过程中会逐渐改进；由于它们在这种情况下代表的是相当相似的类别，梯度下降肯定会使它们彼此更接近，同时也会使它们远离`"INLAND"`类别的嵌入（参见[图13-6](#embedding_diagram)）。实际上，表示得越好，神经网络就越容易做出准确的预测，因此训练倾向于使嵌入成为类别的有用表示。这被称为*表示学习*（您将在[第17章](ch17.html#autoencoders_chapter)中看到其他类型的表示学习）。

![mls3 1306](assets/mls3_1306.png)

###### 图13-6。嵌入将在训练过程中逐渐改进

Keras提供了一个`Embedding`层，它包装了一个*嵌入矩阵*：这个矩阵每行对应一个类别，每列对应一个嵌入维度。默认情况下，它是随机初始化的。要将类别ID转换为嵌入，`Embedding`层只需查找并返回对应于该类别的行。就是这样！例如，让我们用五行和2D嵌入初始化一个`Embedding`层，并用它来编码一些类别：

```py
>>> tf.random.set_seed(42)
>>> embedding_layer = tf.keras.layers.Embedding(input_dim=5, output_dim=2)
>>> embedding_layer(np.array([2, 4, 2]))
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[-0.04663396,  0.01846724],
 [-0.02736737, -0.02768031],
 [-0.04663396,  0.01846724]], dtype=float32)>
```

正如您所看到的，类别2被编码（两次）为2D向量`[-0.04663396, 0.01846724]`，而类别4被编码为`[-0.02736737, -0.02768031]`。由于该层尚未训练，这些编码只是随机的。

###### 警告

`Embedding`层是随机初始化的，因此除非使用预训练权重初始化，否则在模型之外作为独立的预处理层使用它是没有意义的。

如果要嵌入一个分类文本属性，您可以简单地将`StringLookup`层和`Embedding`层连接起来，就像这样：

```py
>>> tf.random.set_seed(42)
>>> ocean_prox = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
>>> str_lookup_layer = tf.keras.layers.StringLookup()
>>> str_lookup_layer.adapt(ocean_prox)
>>> lookup_and_embed = tf.keras.Sequential([
...     str_lookup_layer,
...     tf.keras.layers.Embedding(input_dim=str_lookup_layer.vocabulary_size(),
...                               output_dim=2)
... ])
...
>>> lookup_and_embed(np.array([["<1H OCEAN"], ["ISLAND"], ["<1H OCEAN"]]))
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[-0.01896119,  0.02223358],
 [ 0.02401174,  0.03724445],
 [-0.01896119,  0.02223358]], dtype=float32)>
```

请注意，嵌入矩阵中的行数需要等于词汇量的大小：这是总类别数，包括已知类别和OOV桶（默认只有一个）。`StringLookup`类的`vocabulary_size()`方法方便地返回这个数字。

###### 提示

在这个例子中，我们使用了2D嵌入，但一般来说，嵌入通常有10到300个维度，取决于任务、词汇量和训练集的大小。您将需要调整这个超参数。

将所有内容放在一起，现在我们可以创建一个Keras模型，可以处理分类文本特征以及常规数值特征，并为每个类别（以及每个OOV桶）学习一个嵌入：

```py
X_train_num, X_train_cat, y_train = [...]  # load the training set
X_valid_num, X_valid_cat, y_valid = [...]  # and the validation set

num_input = tf.keras.layers.Input(shape=[8], name="num")
cat_input = tf.keras.layers.Input(shape=[], dtype=tf.string, name="cat")
cat_embeddings = lookup_and_embed(cat_input)
encoded_inputs = tf.keras.layers.concatenate([num_input, cat_embeddings])
outputs = tf.keras.layers.Dense(1)(encoded_inputs)
model = tf.keras.models.Model(inputs=[num_input, cat_input], outputs=[outputs])
model.compile(loss="mse", optimizer="sgd")
history = model.fit((X_train_num, X_train_cat), y_train, epochs=5,
                    validation_data=((X_valid_num, X_valid_cat), y_valid))
```

这个模型有两个输入：`num_input`，每个实例包含八个数值特征，以及`cat_input`，每个实例包含一个分类文本输入。该模型使用我们之前创建的`lookup_and_embed`模型来将每个海洋接近类别编码为相应的可训练嵌入。接下来，它使用`concatenate()`函数将数值输入和嵌入连接起来，生成完整的编码输入，准备输入神经网络。在这一点上，我们可以添加任何类型的神经网络，但为了简单起见，我们只添加一个单一的密集输出层，然后我们创建Keras`Model`，使用我们刚刚定义的输入和输出。接下来，我们编译模型并训练它，传递数值和分类输入。

正如您在[第10章](ch10.html#ann_chapter)中看到的，由于`Input`层的名称是`"num"`和`"cat"`，我们也可以将训练数据传递给`fit()`方法，使用字典而不是元组：`{"num": X_train_num, "cat": X_train_cat}`。或者，我们可以传递一个包含批次的`tf.data.Dataset`，每个批次表示为`((X_batch_num, X_batch_cat), y_batch)`或者`({"num": X_batch_num, "cat": X_batch_cat}, y_batch)`。当然，验证数据也是一样的。

###### 注意

先进行独热编码，然后通过一个没有激活函数和偏置的`Dense`层等同于一个`Embedding`层。然而，`Embedding`层使用的计算量要少得多，因为它避免了许多零乘法——当嵌入矩阵的大小增长时，性能差异变得明显。`Dense`层的权重矩阵起到了嵌入矩阵的作用。例如，使用大小为20的独热向量和一个具有10个单元的`Dense`层等同于使用一个`input_dim=20`和`output_dim=10`的`Embedding`层。因此，在`Embedding`层后面的层中使用的嵌入维度不应该超过单元数。

好了，现在您已经学会了如何对分类特征进行编码，是时候将注意力转向文本预处理了。

## 文本预处理

Keras为基本文本预处理提供了一个`TextVectorization`层。与`StringLookup`层类似，您必须在创建时传递一个词汇表，或者使用`adapt()`方法从一些训练数据中学习词汇表。让我们看一个例子：

```py
>>> train_data = ["To be", "!(to be)", "That's the question", "Be, be, be."]
>>> text_vec_layer = tf.keras.layers.TextVectorization()
>>> text_vec_layer.adapt(train_data)
>>> text_vec_layer(["Be good!", "Question: be or be?"])
<tf.Tensor: shape=(2, 4), dtype=int64, numpy=
array([[2, 1, 0, 0],
 [6, 2, 1, 2]])>
```

两个句子“Be good!”和“Question: be or be?”分别被编码为`[2, 1, 0, 0]`和`[6, 2, 1, 2]`。词汇表是从训练数据中的四个句子中学习的：“be” = 2，“to” = 3，等等。为构建词汇表，`adapt()`方法首先将训练句子转换为小写并去除标点，这就是为什么“Be”、“be”和“be?”都被编码为“be” = 2。接下来，句子被按空格拆分，生成的单词按降序频率排序，产生最终的词汇表。在编码句子时，未知单词被编码为1。最后，由于第一个句子比第二个句子短，因此用0进行了填充。

###### 提示

`TextVectorization`层有许多选项。例如，您可以通过设置`standardize=None`来保留大小写和标点，或者您可以将任何标准化函数作为`standardize`参数传递。您可以通过设置`split=None`来防止拆分，或者您可以传递自己的拆分函数。您可以设置`output_sequence_length`参数以确保输出序列都被裁剪或填充到所需的长度，或者您可以设置`ragged=True`以获得一个不规则张量而不是常规张量。请查看文档以获取更多选项。

单词ID必须进行编码，通常使用`Embedding`层：我们将在[第16章](ch16.html#nlp_chapter)中进行这样做。或者，您可以将`TextVectorization`层的`output_mode`参数设置为`"multi_hot"`或`"count"`以获得相应的编码。然而，简单地计算单词通常不是理想的：像“to”和“the”这样的单词非常频繁，几乎没有影响，而“basketball”等更稀有的单词则更具信息量。因此，通常最好将`output_mode`设置为`"tf_idf"`，它代表*词频* × *逆文档频率*（TF-IDF）。这类似于计数编码，但在训练数据中频繁出现的单词被降权，反之，稀有单词被升权。例如：

```py
>>> text_vec_layer = tf.keras.layers.TextVectorization(output_mode="tf_idf")
>>> text_vec_layer.adapt(train_data)
>>> text_vec_layer(["Be good!", "Question: be or be?"])
<tf.Tensor: shape=(2, 6), dtype=float32, numpy=
array([[0.96725637, 0.6931472 , 0\. , 0\. , 0\. , 0\.        ],
 [0.96725637, 1.3862944 , 0\. , 0\. , 0\. , 1.0986123 ]], dtype=float32)>
```

TF-IDF的变体有很多种，但`TextVectorization`层实现的方式是将每个单词的计数乘以一个权重，该权重等于log(1 + *d* / (*f* + 1))，其中*d*是训练数据中的句子总数（也称为文档），*f*表示这些训练句子中包含给定单词的数量。例如，在这种情况下，训练数据中有*d* = 4个句子，单词“be”出现在*f* = 3个句子中。由于单词“be”在句子“Question: be or be?”中出现了两次，它被编码为2 × log(1 + 4 / (1 + 3)) ≈ 1.3862944。单词“question”只出现一次，但由于它是一个不太常见的单词，它的编码几乎一样高：1 × log(1 + 4 / (1 + 1)) ≈ 1.0986123。请注意，对于未知单词，使用平均权重。

这种文本编码方法易于使用，并且对于基本的自然语言处理任务可以得到相当不错的结果，但它有几个重要的局限性：它只适用于用空格分隔单词的语言，它不区分同音异义词（例如“to bear”与“teddy bear”），它不提示您的模型单词“evolution”和“evolutionary”之间的关系等。如果使用多热编码、计数或TF-IDF编码，则单词的顺序会丢失。那么还有哪些其他选项呢？

一种选择是使用[TensorFlow Text库](https://tensorflow.org/text)，它提供比`TextVectorization`层更高级的文本预处理功能。例如，它包括几种子词标记器，能够将文本分割成比单词更小的标记，这使得模型更容易检测到“evolution”和“evolutionary”之间有一些共同之处（有关子词标记化的更多信息，请参阅[第16章](ch16.html#nlp_chapter)）。

另一个选择是使用预训练的语言模型组件。现在让我们来看看这个。

## 使用预训练语言模型组件

[TensorFlow Hub库](https://tensorflow.org/hub)使得在您自己的模型中重用预训练模型组件变得容易，用于文本、图像、音频等。这些模型组件称为*模块*。只需浏览[TF Hub存储库](https://tfhub.dev)，找到您需要的模块，将代码示例复制到您的项目中，模块将自动下载并捆绑到一个Keras层中，您可以直接包含在您的模型中。模块通常包含预处理代码和预训练权重，并且通常不需要额外的训练（但当然，您的模型的其余部分肯定需要训练）。

例如，一些强大的预训练语言模型是可用的。最强大的模型非常庞大（几个千兆字节），因此为了快速示例，让我们使用`nnlm-en-dim50`模块，版本2，这是一个相当基本的模块，它将原始文本作为输入并输出50维句子嵌入。我们将导入TensorFlow Hub并使用它来加载模块，然后使用该模块将两个句子编码为向量：

```py
>>> import tensorflow_hub as hub
>>> hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2")
>>> sentence_embeddings = hub_layer(tf.constant(["To be", "Not to be"]))
>>> sentence_embeddings.numpy().round(2)
array([[-0.25,  0.28,  0.01,  0.1 ,  [...] ,  0.05,  0.31],
 [-0.2 ,  0.2 , -0.08,  0.02,  [...] , -0.04,  0.15]], dtype=float32)
```

`hub.KerasLayer`层从给定的URL下载模块。这个特定的模块是一个*句子编码器*：它将字符串作为输入，并将每个字符串编码为单个向量（在本例中是一个50维向量）。在内部，它解析字符串（在空格上拆分单词）并使用在一个巨大的语料库上预训练的嵌入矩阵嵌入每个单词：Google News 7B语料库（七十亿字长！）。然后计算所有单词嵌入的平均值，结果就是句子嵌入。

您只需要在您的模型中包含这个`hub_layer`，然后就可以开始了。请注意，这个特定的语言模型是在英语上训练的，但许多其他语言也可用，以及多语言模型。

最后，由Hugging Face提供的优秀开源[Transformers库](https://huggingface.co/docs/transformers)也使得在您自己的模型中包含强大的语言模型组件变得容易。您可以浏览[Hugging Face Hub](https://huggingface.co/models)，选择您想要的模型，并使用提供的代码示例开始。它以前只包含语言模型，但现在已扩展到包括图像模型等。

我们将在[第16章](ch16.html#nlp_chapter)中更深入地讨论自然语言处理。现在让我们看一下Keras的图像预处理层。

## 图像预处理层

Keras预处理API包括三个图像预处理层：

+   `tf.keras.layers.Resizing`将输入图像调整为所需大小。例如，`Resizing(height=100, width=200)`将每个图像调整为100×200，可能会扭曲图像。如果设置`crop_to_aspect_ratio=True`，则图像将被裁剪到目标图像比例，以避免扭曲。

+   `tf.keras.layers.Rescaling`重新缩放像素值。例如，`Rescaling(scale=2/255, offset=-1)`将值从0 → 255缩放到-1 → 1。

+   `tf.keras.layers.CenterCrop`裁剪图像，保留所需高度和宽度的中心区域。

例如，让我们加载一些示例图像并对它们进行中心裁剪。为此，我们将使用Scikit-Learn的`load_sample_images()`函数；这将加载两个彩色图像，一个是中国寺庙的图像，另一个是花朵的图像（这需要Pillow库，如果您正在使用Colab或者按照安装说明进行操作，应该已经安装）：

```py
from sklearn.datasets import load_sample_images

images = load_sample_images()["images"]
crop_image_layer = tf.keras.layers.CenterCrop(height=100, width=100)
cropped_images = crop_image_layer(images)
```

Keras还包括几个用于数据增强的层，如`RandomCrop`、`RandomFlip`、`RandomTranslation`、`RandomRotation`、`RandomZoom`、`RandomHeight`、`RandomWidth`和`RandomContrast`。这些层仅在训练期间激活，并随机对输入图像应用一些转换（它们的名称是不言自明的）。数据增强将人为增加训练集的大小，通常会导致性能提升，只要转换后的图像看起来像真实的（非增强的）图像。我们将在下一章更详细地介绍图像处理。

###### 注意

在幕后，Keras预处理层基于TensorFlow的低级API。例如，`Normalization`层使用`tf.nn.moments()`来计算均值和方差，`Discretization`层使用`tf.raw_ops.Bucketize()`，`CategoricalEncoding`使用`tf.math.bincount()`，`IntegerLookup`和`StringLookup`使用`tf.lookup`包，`Hashing`和`TextVectorization`使用`tf.strings`包中的几个操作，`Embedding`使用`tf.nn.embedding_lookup()`，图像预处理层使用`tf.image`包中的操作。如果Keras预处理API不满足您的需求，您可能偶尔需要直接使用TensorFlow的低级API。

现在让我们看看在TensorFlow中另一种轻松高效地加载数据的方法。

# TensorFlow数据集项目

[TensorFlow数据集（TFDS）](https://tensorflow.org/datasets)项目使加载常见数据集变得非常容易，从小型数据集如MNIST或Fashion MNIST到像ImageNet这样的大型数据集（您将需要相当大的磁盘空间！）。列表包括图像数据集、文本数据集（包括翻译数据集）、音频和视频数据集、时间序列等等。您可以访问[*https://homl.info/tfds*](https://homl.info/tfds)查看完整列表，以及每个数据集的描述。您还可以查看[了解您的数据](https://knowyourdata.withgoogle.com)，这是一个用于探索和理解TFDS提供的许多数据集的工具。

TFDS并未与TensorFlow捆绑在一起，但如果您在Colab上运行或者按照[*https://homl.info/install*](https://homl.info/install)的安装说明进行安装，那么它已经安装好了。然后您可以导入`tensorflow_datasets`，通常为`tfds`，然后调用`tfds.load()`函数，它将下载您想要的数据（除非之前已经下载过），并将数据作为数据集字典返回（通常一个用于训练，一个用于测试，但这取决于您选择的数据集）。例如，让我们下载MNIST：

```py
import tensorflow_datasets as tfds

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]
```

然后您可以应用任何您想要的转换（通常是洗牌、批处理和预取），然后准备训练您的模型。这里是一个简单的示例：

```py
for batch in mnist_train.shuffle(10_000, seed=42).batch(32).prefetch(1):
    images = batch["image"]
    labels = batch["label"]
    # [...] do something with the images and labels
```

###### 提示

`load()`函数可以对其下载的文件进行洗牌：只需设置`shuffle_files=True`。但是这可能不够，最好对训练数据进行更多的洗牌。

请注意，数据集中的每个项目都是一个包含特征和标签的字典。但是Keras期望每个项目是一个包含两个元素的元组（再次，特征和标签）。您可以使用`map()`方法转换数据集，就像这样：

```py
mnist_train = mnist_train.shuffle(buffer_size=10_000, seed=42).batch(32)
mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))
mnist_train = mnist_train.prefetch(1)
```

但是通过设置`as_supervised=True`，让`load()`函数为您执行此操作会更简单（显然，这仅适用于带标签的数据集）。

最后，TFDS提供了一种方便的方法来使用`split`参数拆分数据。例如，如果您想要使用训练集的前90%进行训练，剩余的10%进行验证，整个测试集进行测试，那么您可以设置`split=["train[:90%]", "train[90%:]", "test"]`。`load()`函数将返回所有三个集合。这里是一个完整的示例，使用TFDS加载和拆分MNIST数据集，然后使用这些集合来训练和评估一个简单的Keras模型：

```py
train_set, valid_set, test_set = tfds.load(
    name="mnist",
    split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True
)
train_set = train_set.shuffle(buffer_size=10_000, seed=42).batch(32).prefetch(1)
valid_set = valid_set.batch(32).cache()
test_set = test_set.batch(32).cache()
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=5)
test_loss, test_accuracy = model.evaluate(test_set)
```

恭喜，您已经到达了这个相当技术性的章节的结尾！您可能会觉得它与神经网络的抽象美有些远，但事实是深度学习通常涉及大量数据，知道如何高效加载、解析和预处理数据是一项至关重要的技能。在下一章中，我们将看一下卷积神经网络，这是图像处理和许多其他应用中最成功的神经网络架构之一。

# 练习

1.  为什么要使用tf.data API？

1.  将大型数据集拆分为多个文件的好处是什么？

1.  在训练过程中，如何判断您的输入管道是瓶颈？您可以做些什么来解决它？

1.  您可以将任何二进制数据保存到TFRecord文件中吗，还是只能序列化协议缓冲区？

1.  为什么要费心将所有数据转换为`Example`协议缓冲区格式？为什么不使用自己的协议缓冲区定义？

1.  在使用TFRecords时，何时应该激活压缩？为什么不系统地这样做？

1.  数据可以在编写数据文件时直接进行预处理，或者在tf.data管道中进行，或者在模型内的预处理层中进行。您能列出每个选项的一些优缺点吗？

1.  列举一些常见的编码分类整数特征的方法。文本呢？

1.  加载时尚MNIST数据集（在[第10章](ch10.html#ann_chapter)中介绍）；将其分为训练集、验证集和测试集；对训练集进行洗牌；并将每个数据集保存到多个TFRecord文件中。每个记录应该是一个序列化的`Example`协议缓冲区，具有两个特征：序列化图像（使用`tf.io.serialize_tensor()`来序列化每个图像），和标签。然后使用tf.data为每个集创建一个高效的数据集。最后，使用Keras模型来训练这些数据集，包括一个预处理层来标准化每个输入特征。尝试使输入管道尽可能高效，使用TensorBoard来可视化分析数据。

1.  在这个练习中，您将下载一个数据集，将其拆分，创建一个`tf.data.Dataset`来高效加载和预处理数据，然后构建和训练一个包含`Embedding`层的二元分类模型：

    1.  下载[大型电影评论数据集](https://homl.info/imdb)，其中包含来自[互联网电影数据库（IMDb）](https://imdb.com)的50,000条电影评论。数据组织在两个目录中，*train*和*test*，每个目录包含一个*pos*子目录，其中包含12,500条正面评论，以及一个*neg*子目录，其中包含12,500条负面评论。每个评论存储在单独的文本文件中。还有其他文件和文件夹（包括预处理的词袋版本），但在这个练习中我们将忽略它们。

    1.  将测试集分为验证集（15,000）和测试集（10,000）。

    1.  使用tf.data为每个集创建一个高效的数据集。

    1.  创建一个二元分类模型，使用`TextVectorization`层来预处理每个评论。

    1.  添加一个`Embedding`层，并计算每个评论的平均嵌入，乘以单词数量的平方根（参见[第16章](ch16.html#nlp_chapter)）。然后将这个重新缩放的平均嵌入传递给您模型的其余部分。

    1.  训练模型并查看您获得的准确性。尝试优化您的管道，使训练尽可能快。

    1.  使用TFDS更轻松地加载相同的数据集：`tfds.load("imdb_reviews")`。

这些练习的解决方案可以在本章笔记本的末尾找到，网址为[*https://homl.info/colab3*](https://homl.info/colab3)。

^([1](ch13.html#idm45720190533488-marker)) 想象一副排好序的扑克牌在您的左边：假设您只拿出前三张牌并洗牌，然后随机选取一张放在右边，将另外两张留在手中。再从左边拿一张牌，在手中的三张牌中洗牌，随机选取一张放在右边。当您像这样处理完所有的牌后，您的右边将有一副扑克牌：您认为它会被完美洗牌吗？

^([2](ch13.html#idm45720189926208-marker)) 一般来说，只预取一个批次就可以了，但在某些情况下，您可能需要预取更多。或者，您可以通过将`tf.data.AUTOTUNE`传递给`prefetch()`，让TensorFlow自动决定。

^([3](ch13.html#idm45720189876688-marker)) 但是请查看实验性的`tf.data.experimental.prefetch_to_device()`函数，它可以直接将数据预取到GPU。任何带有`experimental`的TensorFlow函数或类的名称可能会在未来版本中发生更改而没有警告。如果实验性函数失败，请尝试删除`experimental`一词：它可能已经移至核心API。如果没有，请查看笔记本，我会确保其中包含最新的代码。

^([4](ch13.html#idm45720189164144-marker)) 由于protobuf对象旨在被序列化和传输，它们被称为*消息*。

^([5](ch13.html#idm45720189071424-marker)) 本章包含了您使用TFRecords所需了解的最基本知识。要了解更多关于protobufs的信息，请访问[*https://homl.info/protobuf*](https://homl.info/protobuf)。

^([6](ch13.html#idm45720187048176-marker)) Tomáš Mikolov等人，“单词和短语的分布式表示及其组合性”，*第26届国际神经信息处理系统会议论文集* 2（2013）：3111–3119。

^([7](ch13.html#idm45720187037120-marker)) Malvina Nissim等人，“公平比耸人听闻更好：男人对医生，女人对医生”，arXiv预印本arXiv:1905.09866（2019）。

^([8](ch13.html#idm45720186424416-marker)) TensorFlow Hub没有与TensorFlow捆绑在一起，但如果您在Colab上运行或者按照[*https://homl.info/install*](https://homl.info/install)的安装说明进行安装，那么它已经安装好了。

^([9](ch13.html#idm45720186382240-marker)) 要精确，句子嵌入等于句子中单词嵌入的平均值乘以句子中单词数的平方根。这是为了弥补随着*n*增长，*n*个随机向量的平均值会变短的事实。

^([10](ch13.html#idm45720185796112-marker)) 对于大图像，您可以使用`tf.io.encode_jpeg()`。这将节省大量空间，但会损失一些图像质量。
