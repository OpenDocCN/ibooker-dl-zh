# 第二章 数据存储和摄入

要想设想如何设置一个 ML 模型来解决问题，您必须开始考虑数据结构模式。在本章中，我们将看一些存储、数据格式和数据摄入中的一般模式。通常，一旦您理解了您的业务问题并将其设置为数据科学问题，您必须考虑如何将数据转换为模型训练过程可以使用的格式或结构。在训练过程中进行数据摄入基本上是一个数据转换管道。没有这种转换，您将无法在企业驱动或用例驱动的环境中交付和提供模型；它将仍然只是一个探索工具，无法扩展以处理大量数据。

本章将向您展示如何为两种常见的数据结构（表格和图像）设计数据摄入管道。您将学习如何使用 TensorFlow 的 API 使管道可扩展。

*数据流式处理*是模型在训练过程中以小批量摄入数据的方式。在 Python 中进行数据流式处理并不是一个新概念。然而，理解它对于理解 TensorFlow 中更高级 API 的工作方式是基本的。因此，本章将从 Python 生成器开始。然后我们将看一下如何存储表格数据，包括如何指示和跟踪特征和标签。然后我们将转向设计您的数据结构，并讨论如何将数据摄入到模型进行训练以及如何流式传输表格数据。本章的其余部分涵盖了如何为图像分类组织图像数据以及如何流式传输图像数据。

# 使用 Python 生成器进行数据流式处理

有时候 Python 运行时的内存不足以处理整个数据集的加载。当发生这种情况时，建议的做法是将数据分批加载。因此，在训练过程中，数据被流式传输到模型中。

以小批量发送数据还有许多其他优点。其中之一是对每个批次应用梯度下降算法来计算误差（即模型输出与实际值之间的差异），并逐渐更新模型的权重和偏差以使这个误差尽可能小。这让我们可以并行计算梯度，因为一个批次的误差计算（也称为*损失计算*）不依赖于其他批次。这被称为*小批量梯度下降*。在每个时代结束时，当整个训练数据集经过模型时，所有批次的梯度被求和并更新权重。然后，使用新更新的权重和偏差重新开始训练下一个时代，并计算误差。这个过程根据用户定义的参数重复进行，这个参数被称为*训练时代数*。

*Python 生成器*是返回可迭代对象的迭代器。以下是它的工作示例。让我们从一个 NumPy 库开始，进行这个简单的 Python 生成器演示。我创建了一个名为`my_generator`的函数，它接受一个 NumPy 数组，并以数组中的两条记录为一组进行迭代：

```py
import numpy as np

def my_generator(my_array):
    i = 0
    while True:
        yield my_array[i:i+2, :] # output two elements at a time
        i += 1
```

这是我创建的测试数组，将被传递给`my_generator`：

```py
test_array = np.array([[10.0, 2.0],
                       [15, 6.0],
                       [3.2, -1.5],
                       [-3, -2]], np.float32)
```

这个 NumPy 数组有四条记录，每条记录由两个浮点值组成。然后我将这个数组传递给`my_generator`：

```py
output = my_generator(test_array)
```

要获得输出，请使用：

```py
next(output)
```

输出应该是：

```py
array([[10.,  2.],
       [15.,  6.]], dtype=float32)
```

如果您再次运行`next(output)`命令，输出将不同：

```py
array([[15\. ,  6\. ],
       [ 3.2, -1.5]], dtype=float32)
```

如果您再次运行它，输出将再次不同：

```py
array([[ 3.2, -1.5],
       [-3\. , -2\. ]], dtype=float32)
```

如果您第四次运行它，输出现在是：

```py
array([[-3., -2.]], dtype=float32)
```

现在最后一条记录已经显示，您已经完成了对这些数据的流式处理。如果您再次运行它，它将返回一个空数组：

```py
array([], shape=(0, 2), dtype=float32)
```

正如您所看到的，`my_generator` 函数每次运行时都会以 NumPy 数组的形式流式传输两条记录。生成器函数的独特之处在于使用 `yield` 语句而不是 `return` 语句。与 `return` 不同，`yield` 会生成一个值序列，而不会将整个序列存储在 Python 运行时内存中。`yield` 在我们调用 `next` 函数时每次生成一个序列，直到到达数组的末尾。

此示例演示了如何通过生成器函数生成数据子集。但是，在此示例中，NumPy 数组是即时创建的，因此保存在 Python 运行时内存中。让我们看看如何迭代存储为文件的数据集。

# 使用生成器流式传输文件内容

要理解如何流式传输存储中的文件，您可能会发现使用 CSV 文件作为示例更容易。我在这里使用的文件是 Pima 印第安人糖尿病数据集，这是一个[可供下载的开源数据集](https://oreil.ly/enlwY)。下载并将其存储在本地机器上。

这个文件没有包含标题，因此您还需要[下载](https://oreil.ly/NxIKk)此数据集的列名和描述。

简而言之，该文件中的列是：

```py
['Pregnancies', 'Glucose', 'BloodPressure',
 'SkinThickness', 'Insulin', 'BMI',
 'DiabetesPedigree', 'Age', 'Outcome']
```

让我们用以下代码行查看这个文件：

```py
import csv
import pandas as pd

file_path = 'working_data/'
file_name = 'pima-indians-diabetes.data.csv'

col_name = ['Pregnancies', 'Glucose', 'BloodPressure',
            'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigree', 'Age', 'Outcome']
pd.read_csv(file_path + file_name, names = col_name)
```

文件的前几行显示在 图 2-1 中。

![Pima Indians Diabetes Dataset](img/t2pr_0201.png)

###### 图 2-1\. Pima 印第安人糖尿病数据集

由于我们想要流式处理这个数据集，更方便的方法是将其作为 CSV 文件读取，并使用生成器输出行，就像我们在前一节中对 NumPy 数组所做的那样。实现这一点的方法是通过以下代码：

```py
import csv
file_path = 'working_data/'
file_name = 'pima-indians-diabetes.data.csv'

with open(file_path + file_name, newline='\n') as csvfile:
    f = csv.reader(csvfile, delimiter=',')
    for row in f:
        print(','.join(row))
```

让我们仔细看看这段代码。我们使用 `with open` 命令创建一个文件句柄对象 `csvfile`，该对象知道文件存储在哪里。下一步是将其传递给 CSV 库中的 `reader` 函数：

```py
f = csv.reader(csvfile, delimiter=',')
```

`f` 是 Python 运行时内存中的整个文件。要检查文件，执行这段简短的 `for` 循环代码：

```py
for row in f:
        print(','.join(row))
```

前几行的输出看起来像 图 2-2。

![Pima 印第安人糖尿病数据集 CSV 输出](img/t2pr_0202.png)

###### 图 2-2\. Pima 印第安人糖尿病数据集 CSV 输出

现在您已经了解了如何使用文件句柄，让我们重构前面的代码，以便可以在函数中使用 `yield`，有效地创建一个生成器来流式传输文件的内容：

```py
def stream_file(file_handle):
    holder = []
    for row in file_handle:
        holder.append(row.rstrip("\n"))
        yield holder
        holder = []

with open(file_path + file_name, newline = '\n') as handle:
    for part in stream_file(handle):
        print(part)
```

回想一下，Python 生成器是一个使用 `yield` 来迭代通过 `iterable` 对象的函数。您可以像往常一样使用 `with open` 获取文件句柄。然后我们将 `handle` 传递给一个包含 `for` 循环的生成器函数 `stream_file`，该循环逐行遍历 `handle` 中的文件，删除换行符 `\n`，然后填充一个 `holder`。每行通过 `yield` 从生成器传回到主线程的 `print` 函数。输出显示在 图 2-3 中。

![由 Python 生成器输出的 Pima 印第安人糖尿病数据集](img/t2pr_0203.png)

###### 图 2-3\. 由 Python 生成器输出的 Pima 印第安人糖尿病数据集

现在您已经清楚了数据集如何进行流式处理，让我们看看如何在 TensorFlow 中应用这一点。事实证明，TensorFlow 利用这种方法构建了一个用于数据摄入的框架。流式处理通常是摄入大量数据（例如一个表中的数十万行，或分布在多个表中）的最佳方式。

# JSON 数据结构

表格数据是用于对 ML 模型训练的特征和标签进行编码的常见和便捷格式，CSV 可能是最常见的表格数据格式。您可以将逗号分隔的每个字段视为一列。每列都定义了一个数据类型，例如数字（整数或浮点数）或字符串。

表格数据不是唯一的结构化数据格式，我指的是每个记录遵循相同约定，每个记录中字段的顺序相同。另一个常见的数据结构是 JSON。JSON（JavaScript 对象表示）是一个由嵌套、分层键值对构建的结构。您可以将键视为列名，将值视为该样本中数据的实际值。JSON 可以转换为 CSV，反之亦然。有时原始数据是以 JSON 格式存在，需要将其转换为 CSV，这样更容易显示和检查。

这里是一个示例 JSON 记录，显示了键值对：

```py
{
   "id": 1,
   "name": {
      "first": "Dan",
      "last": "Jones"
   },
   "rating": [
      8,
      7,
      9
   ]
},
```

请注意，“rating”键与数组[8, 7, 9]的值相关联。

有很多例子使用 CSV 文件或表作为训练数据，并将其纳入 TensorFlow 模型训练过程。通常，数据被读入一个 pandas DataFrame。然而，这种策略只有在所有数据都能适应 Python 运行时内存时才有效。您可以使用流处理数据，而不受 Python 运行时限制内存分配。由于您在前一节学习了 Python 生成器的工作原理，现在您可以看一下 TensorFlow 的 API，它遵循与 Python 生成器相同的原理，并学习如何使用 TensorFlow 采用 Python 生成器框架。

# 设置文件名的模式

在处理一组文件时，您会遇到文件命名约定中的模式。为了模拟一个不断生成和存储新数据的企业环境，我们将使用一个开源 CSV 文件，按行数拆分成多个部分，然后使用固定前缀重命名每个部分。这种方法类似于 Hadoop 分布式文件系统（HDFS）如何命名文件的部分。

如果您有一个方便的 CSV 文件，可以随时使用您自己的 CSV 文件。如果没有，您可以为本示例下载建议的[CSV 文件](https://oreil.ly/8uGKL)（一个 COVID-19 数据集）。（如果您愿意，您可以克隆这个存储库。）

现在，你只需要*owid-covid-data.csv*。一旦下载完成，检查文件并确定行数：

```py
wc -l owid-covid-data.csv
```

输出表明有超过 32,000 行：

```py
32788 owid-covid-data.csv
```

接下来，检查 CSV 文件的前三行，看看是否有标题：

```py
head -3 owid-covid-data.csv
iso_code,continent,location,date,total_cases,new_cases,
total_deaths,new_deaths,total_cases_per_million,
new_cases_per_million,total_deaths_per_million,
new_deaths_per_million,new_tests,total_tests,
total_tests_per_thousand,new_tests_per_thousand,
new_tests_smoothed,new_tests_smoothed_per_thousand,tests_units,
stringency_index,population,population_density,median_age,
aged_65_older,aged_70_older,gdp_per_capita,extreme_poverty,
cardiovasc_death_rate,diabetes_prevalence,female_smokers,
male_smokers,handwashing_facilities,hospital_beds_per_thousand,
life_expectancy
AFG,Asia,Afghanistan,2019-12-31,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,,,,,,,,,38928341.0,
54.422,18.6,2.581,1.337,1803.987,,597.029,9.59,,,37.746,0.5,64.8

```

由于这个文件包含一个标题，你会在每个部分文件中看到标题。你也可以查看一些数据行，看看它们实际上是什么样子的。

# 将单个 CSV 文件拆分为多个 CSV 文件

现在让我们将这个文件分割成多个 CSV 文件，每个文件有 330 行。你应该最终得到 100 个 CSV 文件，每个文件都有标题。如果你使用 Linux 或 macOS，请使用以下命令：

```py
cat owid-covid-data.csv| parallel --header : --pipe -N330 
'cat >owid-covid-data-
part00{#}.csv'
```

对于 macOS，您可能需要先安装`parallel`命令：

```py
brew install parallel
```

这里是一些已创建的文件：

```py
-rw-r--r--  1 mbp16  staff    54026 Jul 26 16:45 
owid-covid-data-part0096.csv
-rw-r--r--  1 mbp16  staff    54246 Jul 26 16:45 
owid-covid-data-part0097.csv
-rw-r--r--  1 mbp16  staff    51278 Jul 26 16:45 
owid-covid-data-part0098.csv
-rw-r--r--  1 mbp16  staff    62622 Jul 26 16:45 
owid-covid-data-part0099.csv
-rw-r--r--  1 mbp16  staff    15320 Jul 26 16:45 
owid-covid-data-part00100.csv
```

这个模式代表了多个 CSV 格式的标准存储安排。命名约定有一个明显的模式：要么所有文件都有相同的标题，要么没有一个文件有标题。

保持文件命名模式是个好主意，无论您有几十个还是几百个文件都会很方便。当您的命名模式可以很容易地用通配符表示时，更容易创建一个指向存储中所有数据的引用或文件模式对象。

在下一节中，我们将看看如何使用 TensorFlow API 创建一个文件模式对象，我们将使用它来为这个数据集创建一个流对象。

# 使用 tf.io 创建文件模式对象

TensorFlow 的`tf.io` API 用于引用包含具有共同命名模式的文件的分布式数据集。这并不意味着您要读取分布式数据集：您想要的是一个包含您想要读取的所有数据集文件的文件路径和名称列表。这并不是一个新的想法。例如，在 Python 中，[glob 库](https://oreil.ly/JLtGm)是检索类似列表的常用选择。`tf.io` API 简单地利用 glob 库生成符合模式对象的文件名列表：

```py
import tensorflow as tf

base_pattern = 'dataset'
file_pattern = 'owid-covid-data-part*'
files = tf.io.gfile.glob(base_pattern + '/' + file_pattern)
```

`files`是一个包含所有原始 CSV 文件名的列表，没有特定顺序：

```py
['dataset/owid-covid-data-part0091.csv',
 'dataset/owid-covid-data-part0085.csv',
 'dataset/owid-covid-data-part0052.csv',
 'dataset/owid-covid-data-part0046.csv',
 'dataset/owid-covid-data-part0047.csv',
…]
```

这个列表将成为下一步的输入，即基于 Python 生成器创建一个流数据集对象。

# 创建一个流数据集对象

现在您已经准备好文件列表，您可以将其用作输入来创建一个流数据集对象。请注意，此代码*仅*旨在演示如何将 CSV 文件列表转换为 TensorFlow 数据集对象。如果您真的要使用这些数据来训练一个监督式 ML 模型，您还将执行数据清洗、归一化和聚合等操作，所有这些我们将在第八章中介绍。在本例中，“new_deaths”被选为目标列：

```py
csv_dataset = tf.data.experimental.make_csv_dataset(files,
              header = True,
              batch_size = 5,
              label_name = 'new_deaths',
              num_epochs = 1,
              ignore_errors = True)
```

前面的代码指定`files`中的每个文件都包含一个标题。为了方便起见，我们将批量大小设置为 5。我们还使用`label_name`指定一个目标列，就好像我们要使用这些数据来训练一个监督式 ML 模型。`num_epochs`用于指定您希望对整个数据集进行多少次流式传输。

要查看实际数据，您需要使用`csv_dataset`对象迭代数据：

```py
for features, target in csv_dataset.take(1):
    print("'Target': {}".format(target))
    print("'Features:'")
    for k, v in features.items():
        print("  {!r:20s}: {}".format(k, v))
```

这段代码使用数据集的第一个批次（`take(1)`），其中包含五个样本。

由于您指定了`label_name`作为目标列，其他列都被视为特征。在数据集中，内容被格式化为键值对。前面代码的输出将类似于这样：

```py
'Target': [ 0\.  0\. 16\.  0\.  0.]
'Features:'
  'iso_code'          : [b'SWZ' b'ESP' b'ECU' b'ISL' b'FRO']
  'continent'         : 
[b'Africa' b'Europe' b'South America' b'Europe' b'Europe']
  'location'          : 
[b'Swaziland' b'Spain' b'Ecuador' b'Iceland' b'Faeroe Islands']
  'date'              : 
[b'2020-04-04' b'2020-02-07' b'2020-07-13' b'2020-04-01' 
  b'2020-06-11']
  'total_cases'       : [9.000e+00 1.000e+00 6.787e+04 
1.135e+03 1.870e+02]
  'new_cases'         : [  0\.   0\. 661\.  49\.   0.]
  'total_deaths'      : [0.000e+00 0.000e+00 5.047e+03 
2.000e+00 0.000e+00]
  'total_cases_per_million': 
              [7.758000e+00 2.100000e-02 3.846838e+03 
3.326007e+03 3.826870e+03]
  'new_cases_per_million': [  0\.      0\.     37.465 
143.59    0\.   ]
  'total_deaths_per_million': [  0\.      0\.    286.061   
5.861   0\.   ]
  'new_deaths_per_million': 
[0\.    0\.    0.907 0\.    0\.   ]
  'new_tests'         : 
[b'' b'' b'1331.0' b'1414.0' b'']
  'total_tests'       : 
[b'' b'' b'140602.0' b'20889.0' b'']
  'total_tests_per_thousand': 
[b'' b'' b'7.969' b'61.213' b'']
  'new_tests_per_thousand': 
[b'' b'' b'0.075' b'4.144' b'']
  'new_tests_smoothed': 
[b'' b'' b'1986.0' b'1188.0' b'']
  'new_tests_smoothed_per_thousand': 
[b'' b'' b'0.113' b'3.481' b'']
  'tests_units'       : 
[b'' b'' b'units unclear' b'tests performed' b'']
  'stringency_index'  : 
[89.81 11.11 82.41 53.7   0\.  ]
  'population'        : 
[ 1160164\. 46754784\. 17643060\.   341250\.    48865.]
  'population_density': 
[79.492 93.105 66.939  3.404 35.308]
  'median_age'        : 
[21.5 45.5 28.1 37.3  0\. ]
  'aged_65_older'     : 
[ 3.163 19.436  7.104 14.431  0\.   ]
  'aged_70_older'     :
[ 1.845 13.799  4.458  9.207  0\.   ]
  'gdp_per_capita'    : 
[ 7738.975 34272.36  10581.936 46482.957     0\.   ]
  'extreme_poverty'   : [b'' b'1.0' b'3.6' b'0.2' b'']
  'cardiovasc_death_rate': 
[333.436  99.403 140.448 117.992   0\.   ]
  'diabetes_prevalence': [3.94 7.17 5.55 5.31 0\.  ]
  'female_smokers'    : 
[b'1.7' b'27.4' b'2.0' b'14.3' b'']
  'male_smokers'      : 
[b'16.5' b'31.4' b'12.3' b'15.2' b'']
  'handwashing_facilities': 
[24.097  0\.    80.635  0\.     0\.   ]
  'hospital_beds_per_thousand': 
[2.1  2.97 1.5  2.91 0\.  ]
  'life_expectancy'   : 
[60.19 83.56 77.01 82.99 80.67]
```

此数据在运行时检索（延迟执行）。根据批量大小，每列包含五条记录。接下来，让我们讨论如何流式传输这个数据集。

# 流式传输 CSV 数据集

现在已经创建了一个 CSV 数据集对象，您可以使用这行代码轻松地按批次迭代它，该代码使用`iter`函数从 CSV 数据集创建一个迭代器，并使用`next`函数返回迭代器中的下一个项目：

```py
features, label = next(iter(csv_dataset))
```

请记住，在这个数据集中有两种类型的元素：`features`和`label`。这些元素作为*元组*返回（类似于对象列表，不同之处在于对象的顺序和值不能被更改或重新分配）。您可以通过将元组元素分配给变量来解压元组。

如果您检查标签，您将看到第一个批次的内容：

```py
<tf.Tensor: shape=(5,), dtype=float32, 
numpy=array([ 0.,  0.,  1., 33., 29.], dtype=float32)>
```

如果再次执行相同的命令，您将看到第二个批次：

```py
features, label = next(iter(csv_dataset))
```

让我们看一下`label`：

```py
<tf.Tensor: shape=(5,), dtype=float32, 
numpy=array([ 7., 15.,  1.,  0.,  6.], dtype=float32)>
```

确实，这是第二批观察结果；它包含与第一批不同的值。这就是在数据摄入管道中生成流式传输 CSV 数据集的方式。当每个批次被发送到模型进行训练时，模型通过*前向传递*计算预测，该计算通过将输入值与神经网络中每个节点的当前权重和偏差相乘来计算输出。然后，它将预测与标签进行比较并计算损失函数。接下来是*反向传递*，模型计算与预期输出的变化，并向网络中的每个节点后退以更新权重和偏差。然后模型重新计算并更新梯度。将新的数据批次发送到模型进行训练，过程重复。

接下来我们将看看如何为存储组织图像数据，并像我们流式传输结构化数据一样流式传输它。

# 组织图像数据

图像分类任务需要以特定方式组织图像，因为与 CSV 或表格数据不同，将标签附加到图像需要特殊技术。组织图像文件的一种直接和常见模式是使用以下目录结构：

```py
<PROJECT_NAME>
       train
           class_1
                <FILENAME>.jpg
                <FILENAME>.jpg
                …
           class_n
                <FILENAME>.jpg
                <FILENAME>.jpg
                …
       validation
           class_1
               <FILENAME>.jpg
               <FILENAME>.jpg
               …
           class_n
               <FILENAME>.jpg
               <FILENAME>.jpg
       test
           class_1
               <FILENAME>.jpg
               <FILENAME>.jpg
               …
           class_n
               <FILENAME>.jpg
               <FILENAME>.jpg
               …
```

*<PROJECT_NAME>*是基本目录。它的下一级包含训练、验证和测试目录。在每个目录中，都有以图像标签命名的子目录（`class_1`、`class_2`等，在下面的示例中是花卉类型），每个子目录包含原始图像文件。如图 2-4 所示。

这种结构很常见，因为它可以方便地跟踪标签及其相应的图像，但这绝不是组织图像数据的唯一方式。让我们看看另一种组织图像的结构。这与之前的结构非常相似，只是训练、测试和验证都是分开的。在*<PROJECT_NAME>*目录的正下方是不同图像类别的目录，如图 2-5 所示。

![用于图像分类和训练工作的文件组织](img/t2pr_0204.png)

###### 图 2-4\. 用于图像分类和训练工作的文件组织

![基于标签的图像文件组织](img/t2pr_0205.png)

###### 图 2-5\. 基于标签的图像文件组织

# 使用 TensorFlow 图像生成器

现在让我们看看如何处理图像。除了文件组织的细微差别外，处理图像还需要一些步骤来标准化和归一化图像文件。模型架构需要所有图像具有固定形状（固定尺寸）。在像素级别，值通常被归一化为[0, 1]的范围（将像素值除以 255）。

在这个例子中，您将使用一个包含五种不同类型的花朵的开源图像集（或者随意使用您自己的图像集）。假设图像应该是 224×224 像素，其中尺寸对应高度和宽度。如果您想要使用预训练的残差神经网络（ResNet）作为图像分类器，这些是输入图像的预期尺寸。

首先让我们下载这些图像。以下代码下载五种不同尺寸的花朵，并将它们放入稍后在图 2-6 中显示的文件结构中：

```py
import tensorflow as tf

data_dir = tf.keras.utils.get_file(
    'flower_photos',
'https://storage.googleapis.com/download.tensorflow.org/
example_images/flower_photos.tgz', untar=True)
```

我们将`data_dir`称为基本目录。它应该类似于：

```py
'/Users/XXXXX/.keras/datasets/flower_photos'
```

如果列出基本目录中的内容，您将看到：

```py
-rw-r-----    1 mbp16  staff  418049 Feb  8  2016 LICENSE.txt
drwx------  801 mbp16  staff   25632 Feb 10  2016 tulips
drwx------  701 mbp16  staff   22432 Feb 10  2016 sunflowers
drwx------  643 mbp16  staff   20576 Feb 10  2016 roses
drwx------  900 mbp16  staff   28800 Feb 10  2016 dandelion
drwx------  635 mbp16  staff   20320 Feb 10  2016 daisy
```

流式传输图像有三个步骤。让我们更仔细地看一下：

1.  创建一个`ImageDataGenerator`对象并指定归一化参数。使用`rescale`参数指示归一化比例，使用`validation_split`参数指定将 20%的数据留出用于交叉验证：

    ```py
    train_datagen = tf.keras.preprocessing.image.
        ImageDataGenerator(
        rescale = 1./255, 
        validation_split = 0.20)
    ```

    可选地，您可以将`rescale`和`validation_split`包装为一个包含键值对的字典：

    ```py
    datagen_kwargs = dict(rescale=1./255, 
                          validation_split=0.20)

    train_datagen = tf.keras.preprocessing.image.
        ImageDataGenerator(**datagen_kwargs)
    ```

    这是一种方便的方式，可以重复使用相同的参数并将多个输入参数包装在一起。 （将字典数据结构传递给函数是一种称为*字典解包*的 Python 技术。）

1.  将`ImageDataGenerator`对象连接到数据源，并指定参数将图像调整为固定尺寸：

    ```py
    IMAGE_SIZE = (224, 224) # Image height and width 
    BATCH_SIZE = 32             
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, 
                          batch_size=BATCH_SIZE, 
                          interpolation="bilinear")

    train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, 
    **dataflow_kwargs)
    ```

1.  为索引标签准备一个映射。在这一步中，您检索生成器为每个标签分配的索引，并创建一个将其映射到实际标签名称的字典。TensorFlow 生成器内部会跟踪`data_dir`下目录名称的标签。它们可以通过`train_generator.class_indices`检索，返回标签和索引的键值对。您可以利用这一点并将其反转以部署模型进行评分。模型将输出索引。要实现这种反向查找，只需反转由`train_generator.class_indices`生成的标签字典：

    ```py
    labels_idx = (train_generator.class_indices)
    idx_labels = dict((v,k) for k,v in labels_idx.items())
    ```

    这些是`idx_labels`：

    ```py
    {0: 'daisy', 1: 'dandelion', 2: 'roses', 
      3: 'sunflowers', 4: 'tulips'}
    ```

    现在您可以检查`train_generator`生成的项目的形状：

    ```py
    for image_batch, labels_batch in train_generator:
      print(image_batch.shape)
      print(labels_batch.shape)
      break
    ```

    预计通过迭代基本目录生成器产生的第一批数据将会看到以下内容：

    ```py
    (32, 224, 224, 3)
    (32, 5)
    ```

    第一个元组表示 32 张图片的批量大小，每张图片的尺寸为 224×224×3（高度×宽度×深度，深度代表三个颜色通道 RGB）。第二个元组表示 32 个标签，每个标签对应五种花的其中一种。它是根据`idx_labels`进行独热编码的。

# 流式交叉验证图片

回想一下，在创建用于流式训练数据的生成器时，您使用了`validation_split`参数，值为 0.2。如果不这样做，`validation_split`默认为 0。如果`validation_split`设置为非零小数，则在调用`flow_from_directory`方法时，还必须指定`subset`为`training`或`validation`。在前面的示例中，它是`subset="training"`。

您可能想知道如何知道哪些图片属于我们之前创建的训练生成器的`training`子集。好消息是，如果您重新分配和重复使用训练生成器，您就不需要知道这一点：

```py
valid_datagen = train_datagen

valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, 
    **dataflow_kwargs)
```

正如您所看到的，TensorFlow 生成器知道并跟踪`training`和`validation`子集，因此您可以重复使用相同的生成器来流式传输不同的子集。`dataflow_kwargs`字典也被重用。这是 TensorFlow 生成器提供的一个便利功能。

因为您重复使用`train_datagen`，所以可以确保图像重新缩放的方式与图像训练的方式相同。在`valid_datagen.flow_from_directory`方法中，您将传入相同的`dataflow_kwargs`字典，以设置交叉验证的图像大小与训练图像相同。

如果您希望自行组织图片到训练、验证和测试中，之前学到的内容仍然适用，但有两个例外。首先，您的`data_dir`位于训练、验证或测试目录的级别。其次，在`ImageDataGenerator`和`flow_from_directory`中不需要指定`validation_split`和`subset`。

# 检查调整大小的图片

现在让我们检查生成器生成的调整大小的图片。以下是通过生成器流式传输的数据批次进行迭代的代码片段：

```py
import matplotlib.pyplot as plt
import numpy as np

image_batch, label_batch = next(iter(train_generator))

fig, axes = plt.subplots(8, 4, figsize=(10, 20))
axes = axes.flatten()
for img, lbl, ax in zip(image_batch, label_batch, axes):
    ax.imshow(img)
    label_ = np.argmax(lbl)
    label = idx_labels[label_]
    ax.set_title(label)
    ax.axis('off')
plt.show()
```

这段代码将从生成器中产生的第一批数据中生成 32 张图片（参见图 2-6）。

![一批调整大小的图片](img/t2pr_0206.png)

###### 图 2-6. 一批调整大小的图片

让我们来检查代码：

```py
image_batch, label_batch = next(iter(train_generator))
```

这将通过生成器迭代基本目录。它将`iter`函数应用于生成器，并利用`next`函数将图像批次和标签批次输出为 NumPy 数组：

```py
fig, axes = plt.subplots(8, 4, figsize=(10, 20))
```

这行代码设置了您期望的子图数量，即 32，即批量大小：

```py
axes = axes.flatten()
for img, lbl, ax in zip(image_batch, label_batch, axes):
    ax.imshow(img)
    label_ = np.argmax(lbl)
    label = idx_labels[label_]
    ax.set_title(label)
    ax.axis('off')
plt.show()
```

然后您设置图形轴，使用`for`循环将 NumPy 数组显示为图片和标签。如图 2-6 所示，所有图片都被调整为 224×224 像素的正方形。尽管子图容器是一个尺寸为`(10, 20)`的矩形，您可以看到所有图片都是正方形的。这意味着您在生成器工作流中调整大小和归一化图片的代码按预期工作。

# 总结

在本章中，你学会了使用 Python 处理流数据的基础知识。这是在处理大型分布式数据集时的一种重要技术。你还看到了一些常见的表格和图像数据的文件组织模式。

在表格数据部分，你学会了如何选择一个良好的文件命名约定，可以更容易地构建对所有文件的引用，无论有多少个文件。这意味着你现在知道如何构建一个可扩展的流水线，可以将所需的数据输入到 Python 运行时中，用于任何用途（在本例中，用于 TensorFlow 创建数据集）。

你还学会了图像文件通常如何在文件存储中组织，以及如何将图像与标签关联起来。在下一章中，你将利用你在这里学到的关于数据组织和流式处理的知识，将其与模型训练过程整合起来。
