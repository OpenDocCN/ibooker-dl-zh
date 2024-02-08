# 第三章：数据预处理

在本章中，您将学习如何为训练准备和设置数据。机器学习工作中最常见的数据格式是表格、图像和文本。与每种数据格式相关的常见实践技术，尽管如何设置数据工程流水线当然取决于您的问题陈述是什么以及您试图预测什么。

我将详细查看所有三种格式，使用具体示例来引导您了解这些技术。所有数据都可以直接读入 Python 运行时内存；然而，这并不是最有效的使用计算资源的方式。当讨论文本数据时，我将特别关注标记和字典。通过本章结束时，您将学会如何准备表格、图像和文本数据进行训练。

# 为训练准备表格数据

在表格数据集中，重要的是要确定哪些列被视为分类列，因为您必须将它们的值编码为类或类的二进制表示（独热编码），而不是数值值。表格数据集的另一个方面是多个特征之间的相互作用的潜力。本节还将查看 TensorFlow 提供的 API，以便更容易地建模列之间的交互。

通常会遇到作为 CSV 文件的表格数据集，或者仅仅作为数据库查询结果的结构化输出。在这个例子中，我们将从已经在 pandas DataFrame 中的数据集开始，并学习如何转换它并为模型训练设置它。我们将使用*泰坦尼克*数据集，这是一个开源的表格数据集，通常用于教学，因为其可管理的大小和可用性。该数据集包含每位乘客的属性，如年龄、性别、舱位等级，以及他们是否幸存。我们将尝试根据他们的属性或特征来预测每位乘客的生存概率。请注意，这是一个用于教学和学习目的的小数据集。实际上，您的数据集可能会更大。您可能会对一些输入参数的默认值做出不同的决定，并选择不同的默认值，所以不要对这个例子过于字面理解。

让我们从加载所有必要的库开始：

```py
import functools
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
```

从谷歌的公共存储中加载数据：

```py
TRAIN_DATA_URL = "https://storage.googleapis.com/
tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/
tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", 
TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

```

现在看看`train_file_path`：

```py
print(train_file_path)

/root/.keras/datasets/train.csv
```

这个文件路径指向一个 CSV 文件，我们将其读取为一个 pandas DataFrame：

```py
titanic_df = pd.read_csv(train_file_path, header='infer')
```

图 3-1 展示了`titanic_df`作为 pandas DataFrame 的样子。

![泰坦尼克数据集作为 pandas DataFrame](img/t2pr_0301.png)

###### 图 3-1。*泰坦尼克*数据集作为 pandas DataFrame

## 标记列

如您在图 3-1 中所见，这些数据中既有数值列，也有分类列。目标列，或者用于预测的列，是“survived”列。您需要将其标记为目标，并将其余列标记为特征。

###### 提示

在 TensorFlow 中的最佳实践是将您的表格转换为流式数据集。这种做法确保数据的大小不会影响内存消耗。

为了做到这一点，TensorFlow 提供了函数`tf.data.experimental.make_csv_dataset`：

```py
LABEL_COLUMN = 'survived'
LABELS = [0, 1]

train_ds = tf.data.experimental.make_csv_dataset(
      train_file_path,
      batch_size=3,
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)

test_ds = tf.data.experimental.make_csv_dataset(
      test_file_path,
      batch_size=3,
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
```

在前面的函数签名中，您指定要生成数据集对象的文件路径。`batch_size`被任意设置为一个较小的值（在本例中为 3），以便方便检查数据。我们还将`label_name`设置为“survived”列。对于数据质量，如果任何单元格中指定了问号（?），您希望将其解释为“NA”（不适用）。对于训练，将`num_epochs`设置为对数据集进行一次迭代。您可以忽略任何解析错误或空行。

接下来，检查数据：

```py
for batch, label in train_ds.take(1):
  print(label)
  for key, value in batch.items():
    print("{}: {}".format(key,value.numpy()))
```

它看起来类似于图 3-2。

![一个批次的泰坦尼克数据集](img/t2pr_0302.png)

###### 图 3-2. *泰坦尼克*数据集的一个批次

以下是训练范式消耗训练数据集的主要步骤：

1.  按特征类型指定列。

1.  决定是否嵌入或交叉列。

1.  选择感兴趣的列，可能作为一个实验。

1.  为训练范式创建一个“特征层”以供使用。

现在您已经将数据设置为数据集，可以根据其特征类型指定每列，例如数字或分类，如果需要的话可以进行分桶。如果唯一类别太多且降维会有帮助，还可以嵌入列。

让我们继续进行第 1 步。有四个数字列：`age`、`n_siblings_spouses`、`parch`和`fare`。五列是分类的：`sex`、`class`、`deck`、`embark_town`和`alone`。完成后，您将创建一个`feature_columns`列表来保存所有特征列。

以下是如何根据实际数字值严格指定数字列的方法，而不进行任何转换：

```py
feature_columns = []

# numeric cols
for header in ['age', 'n_siblings_spouses', 'parch', 'fare']:
feature_columns.append(feature_column.numeric_column(header))
```

请注意，除了使用`age`本身，您还可以将`age`分箱，例如按年龄分布的四分位数。但是分箱边界（四分位数）是什么？您可以检查 pandas DataFrame 中数字列的一般统计信息：

```py
titanic_df.describe()
```

图 3-3 显示了输出。

![泰坦尼克号数据集中数字列的统计](img/t2pr_0303.png)

###### 图 3-3。*泰坦尼克号*数据集中的数字列统计

让我们尝试为年龄设置三个分箱边界：23、28 和 35。这意味着乘客年龄将被分组为第一四分位数、第二四分位数和第三四分位数（如图 3-3 所示）：

```py
age = feature_column.numeric_column('age')
age_buckets = feature_column.
bucketized_column(age, boundaries=[23, 28, 35])
```

因此，除了“age”之外，您还生成了另一个列“age_bucket”。

为了了解每个分类列的性质，了解其中的不同值将是有帮助的。您需要使用每列中的唯一条目对词汇表进行编码。对于分类列，这意味着您需要确定哪些条目是唯一的：

```py
h = {}
for col in titanic_df:
  if col in ['sex', 'class', 'deck', 'embark_town', 'alone']:
    print(col, ':', titanic_df[col].unique())
    h[col] = titanic_df[col].unique()
```

结果显示在图 3-4 中。

![泰坦尼克号数据集中每个分类列中的唯一值](img/t2pr_0304.png)

###### 图 3-4。数据集中每个分类列的唯一值

您需要以字典格式跟踪这些唯一值，以便模型进行映射和查找。因此，您将对“sex”列中的唯一分类值进行编码：

```py
sex_type = feature_column.categorical_column_with_vocabulary_list(
      'Type', ['male' 'female'])
sex_type_one_hot = feature_column.indicator_column(sex_type)
```

然而，如果列表很长，逐个写出会变得不方便。因此，当您遍历分类列时，可以将每列的唯一值保存在 Python 字典数据结构`h`中以供将来查找。然后，您可以将唯一值作为列表传递到这些词汇表中：

```py
sex_type = feature_column.
categorical_column_with_vocabulary_list(
      'Type', h.get('sex').tolist())
sex_type_one_hot = feature_column.
indicator_column(sex_type)

class_type = feature_column.
categorical_column_with_vocabulary_list(
      'Type', h.get('class').tolist())
class_type_one_hot = feature_column.
indicator_column(class_type)

deck_type = feature_column.
categorical_column_with_vocabulary_list(
      'Type', h.get('deck').tolist())
deck_type_one_hot = feature_column.
indicator_column(deck_type)

embark_town_type = feature_column.
categorical_column_with_vocabulary_list(
      'Type', h.get('embark_town').tolist())
embark_town_type_one_hot = feature_column.
indicator_column(embark_town_type)

alone_type = feature_column.
categorical_column_with_vocabulary_list(
      'Type', h.get('alone').tolist())
alone_one_hot = feature_column.
indicator_column(alone_type)

```

您还可以嵌入“deck”列，因为有八个唯一值，比任何其他分类列都多。将其维度减少到 3：

```py
deck = feature_column.
categorical_column_with_vocabulary_list(
      'deck', titanic_df.deck.unique())
deck_embedding = feature_column.
embedding_column(deck, dimension=3)
```

减少分类列维度的另一种方法是使用*哈希特征列*。该方法根据输入数据计算哈希值，然后为数据指定一个哈希桶。以下代码将“class”列的维度减少到 4：

```py
class_hashed = feature_column.categorical_column_with_hash_bucket(
      'class', hash_bucket_size=4)
```

## 将列交互编码为可能的特征

现在来到最有趣的部分：您将找到不同特征之间的交互（这被称为*交叉列*），并将这些交互编码为可能的特征。这也是您的直觉和领域知识可以有益于您的特征工程努力的地方。例如，基于*泰坦尼克号*灾难的历史背景，一个问题是：一等舱的女性是否比二等或三等舱的女性更有可能生存？为了将这个问题重新表述为一个数据科学问题，您需要考虑乘客的性别和舱位等级之间的交互。然后，您需要选择一个起始维度大小来表示数据的变化性。假设您任意决定将变化性分成五个维度（`hash_bucket_size`）：

```py

cross_type_feature = feature_column.
crossed_column(['sex', 'class'], hash_bucket_size=5)

```

现在您已经创建了所有特征，需要将它们组合在一起，并可能进行实验以决定在训练过程中包含哪些特征。为此，您首先要创建一个列表来保存您想要使用的所有特征：

```py
feature_columns = []

```

然后，您将每个感兴趣的特征附加到列表中：

```py
# append numeric columns
for header in ['age', 'n_siblings_spouses', 'parch', 'fare']:
  feature_columns.append(feature_column.numeric_column(header))

# append bucketized columns
age = feature_column.numeric_column('age')
age_buckets = feature_column.
bucketized_column(age, boundaries=[23, 28, 35])
feature_columns.append(age_buckets)

# append categorical columns
indicator_column_names = 
['sex', 'class', 'deck', 'embark_town', 'alone']
for col_name in indicator_column_names:
  categorical_column = feature_column.
  categorical_column_with_vocabulary_list(
      col_name, titanic_df[col_name].unique())
  indicator_column = feature_column.

indicator_column(categorical_column)
  feature_columns.append(indicator_column)

# append embedding columns
deck = feature_column.categorical_column_with_vocabulary_list(
      'deck', titanic_df.deck.unique())
deck_embedding = feature_column.
embedding_column(deck, dimension=3)
feature_columns.append(deck_embedding)

# append crossed columns
feature_columns.
append(feature_column.indicator_column(cross_type_feature))

```

现在创建一个特征层：

```py
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
```

这一层将作为您即将构建和训练的模型的第一（输入）层。这是您为模型的训练过程提供所有特征工程框架的方式。

## 创建一个交叉验证数据集

在开始训练之前，您需要为交叉验证目的创建一个小数据集。由于一开始只有两个分区（训练和测试），生成一个交叉验证数据集的一种方法是简单地将其中一个分区细分：

```py
val_df, test_df = train_test_split(test_df, test_size=0.4)
```

在这里，原始`test_df`分区的 40%被随机保留为`test_df`，剩下的 60%现在是`val_df`。通常，测试数据集是三个数据集中最小的，因为它们仅用于最终评估，而不用于模型训练。

现在您已经处理了特征工程和数据分区，还有最后一件事要做：使用数据集将数据流入训练过程。您将把三个 DataFrame（训练、验证和测试）分别转换为自己的数据集：

```py
batch_size = 32
labels = train_df.pop('survived')
working_ds = tf.data.Dataset.
from_tensor_slices((dict(train_df), labels))
working_ds = working_ds.shuffle(buffer_size=len(train_df))
train_ds = working_ds.batch(batch_size)
```

如前面的代码所示，首先您将任意决定要包含在一个批次中的样本数量（`batch_size`）。然后，您需要设置一个标签指定（`survived`）。`tf.data.Dataset.from_tensor_slices`方法接受一个元组作为参数。在这个元组中，有两个元素：特征列和标签列。

第一个元素是`dict(train_df)`。这个`dict`操作实质上将 DataFrame 转换为键值对，其中每个键代表一个列名，相应的值是该列中的值数组。另一个元素是`labels`。

最后，我们对数据集进行洗牌和分批处理。由于这种转换将应用于所有三个数据集，将这些步骤合并到一个辅助函数中以减少重复会很方便：

```py
def pandas_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('survived')
  ds = tf.data.Dataset.
from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
```

现在您可以将此函数应用于验证和测试数据：

```py
val_ds = pandas_to_dataset(val_df, shuffle=False, 
batch_size=batch_size)
test_ds = pandas_to_dataset(test_df, shuffle=False, 
batch_size=batch_size)
```

## 开始模型训练过程

现在，您已经准备好开始模型训练过程。从技术上讲，这并不是预处理的一部分，但通过这个简短的部分，您可以看到您所做的工作如何融入到模型训练过程中。

您将从构建模型架构开始：

```py
model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.1),
  layers.Dense(1)
])
```

为了演示目的，您将构建一个简单的两层深度学习感知器模型，这是一个前馈神经网络的基本配置。请注意，由于这是一个多层感知器模型，您将使用顺序 API。在这个 API 中，第一层是`feature_layer`，它代表所有特征工程逻辑和派生特征，例如年龄分段和交叉，用于建模特征交互。

编译模型并为二元分类设置损失函数：

```py
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(
from_logits=True),
              metrics=['accuracy'])
```

然后您可以开始训练。您只会训练 10 个 epochs：

```py
model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)
```

您可以期望得到与图 3-5 中所示类似的结果。

![泰坦尼克数据集中生存预测的示例训练结果](img/t2pr_0305.png)

###### 图 3-5\. *泰坦尼克*数据集中生存预测的示例训练结果

## 总结

在本节中，您看到了如何处理由多种数据类型组成的表格数据。您还看到了 TensorFlow 提供的`feature_column`API，它可以正确转换数据类型、处理分类数据，并为潜在交互提供特征交叉。这个 API 在简化数据和特征工程任务方面非常有帮助。

# 为处理图像数据做准备

对于图像，您需要将所有图像重塑或重新采样为相同的像素计数；这被称为*标准化*。您还需要确保所有像素值在相同的颜色范围内，以便它们落在每个像素的 RGB 值的有限范围内。

图像数据具有不同的文件扩展名，例如*.jpg*、*.tiff*和*.bmp*。这些并不是真正的问题，因为 Python 和 TensorFlow 中有可以读取和解析任何文件扩展名的图像的 API。关于图像数据的棘手部分在于捕获其维度——高度、宽度和深度——由像素计数来衡量。（如果是用 RGB 编码的彩色图像，这些会显示为三个独立的通道。）

如果您的数据集中的所有图像（包括训练、验证以及测试或部署时的所有图像）都预期具有相同的尺寸*并且*您将构建自己的模型，那么处理图像数据并不是太大的问题。然而，如果您希望利用预构建的模型如 ResNet 或 Inception，那么您必须符合它们的图像要求。例如，ResNet 要求每个输入图像为 224 × 224 × 3 像素，并呈现为 NumPy 多维数组。这意味着在预处理过程中，您必须重新采样您的图像以符合这些尺寸。

另一个需要重新采样的情况是当您无法合理地期望所有图像，特别是在部署时，具有相同的尺寸。在这种情况下，您需要在构建模型时考虑适当的图像尺寸，然后设置预处理程序以确保重新采样正确进行。

在本节中，您将使用 TensorFlow 提供的花卉数据集。它包含五种类型的花卉和不同的图像尺寸。这是一个方便的数据集，因为所有图像都已经是 JPEG 格式。您将处理这些图像数据，训练一个模型来解析每个图像并将其分类为五类花卉之一。

像往常一样，导入所有必要的库：

```py
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import pathlib
```

现在从以下网址下载花卉数据集：

```py
data_dir = tf.keras.utils.get_file(
    'flower_photos',
'https://storage.googleapis.com/download.tensorflow.org/
example_images/flower_photos.tgz',
    untar=True)
```

这个文件是一个压缩的 TAR 存档文件。因此，您需要设置`untar=True`。

使用`tf.keras.utils.get_file`时，默认情况下会在`~/.keras/datasets`目录中找到下载的数据。

在 Mac 或 Linux 系统的 Jupyter Notebook 单元格中执行：

```py
!ls -lrt ~/.keras/datasets/flower_photos
```

您将找到如图 3-6 所示的花卉数据集。

![花卉数据集文件夹](img/t2pr_0306.png)

###### 图 3-6\. 花卉数据集文件夹

现在让我们看看其中一种花卉：

```py
!ls -lrt ~/.keras/datasets/flower_photos/roses | head -10
```

您应该看到前九个图像，如图 3-7 所示。

![玫瑰目录中的十个示例图像文件](img/t2pr_0307.png)

###### 图 3-7\. 玫瑰目录中的九个示例图像文件

这些图像都是不同的尺寸。您可以通过检查几张图像来验证这一点。以下是一个您可以利用的辅助函数，用于显示原始尺寸的图像：¹

```py
def display_image_in_actual_size(im_path):

    dpi = 100
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape
    # What size does the figure need to be in inches to fit 
    # the image?
    figsize = width / float(dpi), height / float(dpi)
    # Create a figure of the right size with one axis that 
    # takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    ax.axis('off')
    # Display the image.
    ax.imshow(im_data, cmap='gray')
    plt.show()
```

让我们用它来显示一张图像（如图 3-8 所示）：

```py
IMAGE_PATH = "/root/.keras/datasets/flower_photos/roses/
7409458444_0bfc9a0682_n.jpg"
display_image_in_actual_size(IMAGE_PATH)
```

![玫瑰图像样本 1](img/t2pr_0308.png)

###### 图 3-8。玫瑰图像示例 1

现在尝试不同的图像（如图 3-9 所示）：

```py
IMAGE_PATH = "/root/.keras/datasets/flower_photos/roses/
5736328472_8f25e6f6e7.jpg"
display_image_in_actual_size(IMAGE_PATH)
```

![玫瑰图像示例 2](img/t2pr_0309.png)

###### 图 3-9。玫瑰图像示例 2

显然，这些图像的尺寸和长宽比是不同的。

## 将图像转换为固定规格

现在您已经准备好将这些图像转换为固定规格。在这个特定的示例中，您将使用 ResNet 输入图像规格，即 224×224，带有三个颜色通道（RGB）。此外，尽可能使用数据流。因此，您的目标是将这些彩色图像转换为 224×224 像素的形状，并从中构建一个数据集，以便流式传输到训练范式中。

为了实现这一点，您将使用`ImageDataGenerator`类和`flow_from_directory`方法。

`ImageDataGenerator`负责创建一个生成器对象，该对象从由`flow_from_directory`指定的目录中生成流式数据。

一般来说，编码模式是：

```py
my_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
my_generator = my_datagen.flow_from_directory(
data_dir, **dataflow_kwargs)
```

在这两种情况下，关键字参数选项或`kwargs`为您的代码提供了很大的灵活性。（关键字参数在 Python 中经常见到。）这些参数使您能够将可选参数传递给函数。事实证明，在`ImageDataGenerator`中，有两个与您需求相关的参数：`rescale`和`validation_split`。`rescale`参数用于将像素值归一化为有限范围，`validation_split`允许您将数据的一个分区细分，例如用于交叉验证。

在`flow_from_directory`中，有三个对于本示例有用的参数：`target_size`、`batch_size`和`interpolation`。`target_size`参数帮助您指定每个图像的期望尺寸，`batch_size`用于指定批量图像中的样本数。至于`interpolation`，请记住您需要对每个图像进行插值或重新采样，以达到用`target_size`指定的规定尺寸？插值的支持方法有`nearest`、`bilinear`和`bicubic`。对于本示例，首先尝试`bilinear`。

您可以将这些关键字参数定义如下。稍后将把它们传递给它们的函数调用：

```py
pixels =224
BATCH_SIZE = 32
IMAGE_SIZE = (pixels, pixels)

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, 
batch_size=BATCH_SIZE,
interpolation="bilinear")
```

创建一个生成器对象：

```py
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
**datagen_kwargs)
```

现在您可以指定此生成器将从中流式传输数据的源目录。此生成器将仅流式传输 20%的数据，并将其指定为验证数据集：

```py
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, 
    **dataflow_kwargs)
```

您可以使用相同的生成器对象进行训练数据：

```py
train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
data_dir, subset="training", shuffle=True, **dataflow_kwargs)
```

检查生成器的输出：

```py
for image_batch, labels_batch in train_generator:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

(32, 224, 224, 3)
(32, 5)
```

输出表示为 NumPy 数组。对于一批图像，样本大小为 32，高度和宽度为 224 像素，三个通道表示 RGB 颜色空间。对于标签批次，同样有 32 个样本。每行都是独热编码，表示属于五类中的哪一类。

另一个重要的事情是检索标签的查找字典。在推断期间，模型将输出每个五类中的概率。唯一的解码方式是使用标签的预测查找字典来确定哪个类具有最高的概率：

```py
labels_idx = (train_generator.class_indices)
idx_labels = dict((v,k) for k,v in labels_idx.items())
print(idx_labels)

{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 
4: 'tulips'}
```

我们分类模型的典型输出将类似于以下的 NumPy 数组：

```py
(0.7, 0.1, 0.1, 0.05, 0.05)
```

具有最高概率值的位置是第一个元素。将此索引映射到`idx_labels`中的第一个键 - 在本例中为`daisy`。这是您捕获预测结果的方法。保存`idx_labels`字典：

```py
import pickle
with open('prediction_lookup.pickle', 'wb') as handle:
    pickle.dump(idx_labels, handle, 
    protocol=pickle.HIGHEST_PROTOCOL)
```

这是如何加载它的方法：

```py
with open('prediction_lookup.pickle', 'rb') as handle:
    lookup = pickle.load(handle)
```

## 训练模型

最后，对于训练，您将使用从预训练的 ResNet 特征向量构建的模型。这种技术称为*迁移学习*。TensorFlow Hub 免费提供许多预训练模型。这是在模型构建过程中访问它的方法：

```py
import tensorflow_hub as hub
NUM_CLASSES = 5
mdl = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
         hub.KerasLayer("https://tfhub.dev/google/imagenet/
resnet_v1_101/feature_vector/4", trainable=False),
tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', 
 name = 'custom_class')
])
mdl.build([None, 224, 224, 3])
```

第一层是`InputLayer`。请记住，预期输入为 224×224×3 像素。您将使用元组添加技巧将额外的维度附加到`IMAGE_SIZE`：

```py
IMAGE_SIZE + (3,)
```

现在您有了（224, 224, 3），这是一个表示图像维度的 NumPy 数组的元组。

下一层是由指向 TensorFlow Hub 的预训练 ResNet 特征向量引用的层。让我们直接使用它，这样我们就不必重新训练它。

接下来是具有五个输出节点的`Dense`层。每个输出是图像属于该类的概率。然后，您将构建模型骨架，第一个维度为`None`。这意味着第一个维度，代表批处理的样本大小，在运行时尚未确定。这是如何处理批输入的方法。

检查模型摘要以确保它符合您的预期：

```py
mdl.summary()
```

输出显示在图 3-10 中。

![图像分类模型摘要](img/t2pr_0310.png)

###### 图 3-10\. 图像分类模型摘要

使用`optimizers`和相应的`losses`函数编译模型：

```py
mdl.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
  loss=tf.keras.losses.CategoricalCrossentropy(
from_logits=True, 
label_smoothing=0.1),
  metrics=['accuracy'])
```

然后对其进行训练：

```py
steps_per_epoch = train_generator.samples // 
train_generator.batch_size
validation_steps = valid_generator.samples // 
valid_generator.batch_size
mdl.fit(
    train_generator,
    epochs=5, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps)
```

您可能会看到类似于图 3-11 的输出。

![训练图像分类模型的输出](img/t2pr_0311.png)

###### 图 3-11\. 训练图像分类模型的输出

## 摘要

在本节中，您学习了如何处理图像文件。具体来说，在设计模型之前，有必要确保您已经设置了一个预定的图像大小要求。一旦这个标准被接受，下一步就是将图像重新采样到该大小，并将像素的值归一化为更小的动态范围。这些例程几乎是通用的。此外，将图像流式传输到训练工作流程是最有效的方法和最佳实践，特别是在您的工作样本大小接近 Python 运行时内存的情况下。

# 为处理文本数据做准备

对于文本数据，每个单词或字符都需要表示为一个数字整数。这个过程被称为*标记化*。此外，如果目标是分类，那么目标需要被编码为*类别*。如果目标是更复杂的，比如翻译，那么训练数据中的目标语言（比如英语到法语翻译中的法语）也需要自己的标记化过程。这是因为目标本质上是一个长字符串的文本，就像输入文本一样。同样，您还需要考虑是在单词级别还是字符级别对目标进行标记化。

文本数据可以以许多不同的格式呈现。从内容组织的角度来看，它可以被存储和组织为一个表格，其中一列包含文本的主体或字符串，另一列包含标签，例如二进制情感指示器。它可能是一个自由格式的文件，每行长度不同，每行末尾有一个换行符。它可能是一份手稿，其中文本块由段落或部分定义。

有许多方法可以确定要使用的处理技术和逻辑，当您设置自然语言处理（NLP）机器学习问题时；本节将涵盖一些最常用的技术。

这个例子将使用威廉·莎士比亚的悲剧《科里奥兰纳斯》中的文本，这是一个简单的公共领域示例，托管在谷歌上。您将构建一个*文本生成模型*，该模型将学习如何以莎士比亚的风格写作。

## 对文本进行标记化

文本由字符字符串表示。这些字符需要转换为整数以进行建模任务。这个例子是*科里奥兰纳斯*的原始文本字符串。

让我们导入必要的库并下载文本文件：

```py
import tensorflow as tf
import numpy as np
import os
import time

FILE_URL = 'https://storage.googleapis.com/download.tensorflow.org/
data/shakespeare.txt'
FILE_NAME = 'shakespeare.txt'
path_to_file = tf.keras.utils.get_file('shakespeare.txt', FILE_URL)
```

打开它并输出几行示例文本：

```py
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print ('Length of text: {} characters'.format(len(text)))
```

通过打印前 400 个字符来检查这段文本：

```py
print(text[:400])
```

输出显示在图 3-12 中。

为了对该文件中的每个字符进行标记化，简单的`set`操作就足够了。这个操作将创建在文本字符串中找到的字符的一个唯一集合：

```py
vocabulary = sorted(set(text))
print ('There are {} unique characters'.format(len(vocabulary)))

There are 65 unique characters
```

图 3-13 展示了`vocabulary`列表的一瞥。

![威廉·莎士比亚《科里奥兰纳斯》的样本](img/t2pr_0312.png)

###### 图 3-12. 威廉·莎士比亚《科里奥兰纳斯》的示例

![《科里奥兰纳斯》词汇列表的一部分](img/t2pr_0313.png)

###### 图 3-13.《科里奥兰纳斯》词汇列表的一部分

这些标记包括标点符号，以及大写和小写字符。不一定需要同时包含大写和小写字符；如果不想要，可以在执行`set`操作之前将每个字符转换为小写。由于您对标记列表进行了排序，您可以看到特殊字符也被标记化了。在某些情况下，这是不必要的；这些标记可以手动删除。以下代码将所有字符转换为小写，然后执行`set`操作：

```py
vocabulary = sorted(set(text.lower()))
print ('There are {} unique characters'.format(len(vocabulary)))

There are 39 unique characters
```

您可能会想知道是否将文本标记化为单词级而不是字符级是否合理。毕竟，单词是对文本字符串的语义理解的基本单位。尽管这种推理是合理的并且有一定的逻辑性，但实际上它会增加更多的工作和问题，而并没有真正为训练过程增加价值或为模型的准确性增加价值。为了说明这一点，让我们尝试按单词对文本字符串进行标记化。首先要认识到的是单词是由空格分隔的。因此，您需要在空格上拆分文本字符串：

```py
vocabulary_word = sorted(set(text.lower().split(' ')))
print ('There are {} unique words'.format(len(vocabulary_word)))

There are 41623 unique words
```

检查`vocabulary_word`列表，如图 3-14 所示。

由于每个单词标记中嵌入了特殊字符和换行符，这个列表几乎无法使用。需要通过正则表达式或更复杂的逻辑来清理它。在某些情况下，标点符号附加在单词上。此外，单词标记列表比字符级标记列表要大得多。这使得模型更难学习文本中的模式。出于这些原因和缺乏已证明的好处，将文本标记化为单词级并不是一种常见做法。如果您希望使用单词级标记化，则通常会执行单词嵌入操作以减少话语表示的变异性和维度。

![标记化单词示例](img/t2pr_0314.png)

###### 图 3-14. 标记化单词的示例

## 创建字典和反向字典

一旦您有包含所选字符的标记列表，您将需要将每个标记映射到一个整数。这被称为*字典*。同样，您需要创建一个*反向字典*，将整数映射回标记。

使用`enumerate`函数很容易生成一个整数。这个函数以列表作为输入，并返回与列表中每个唯一元素对应的整数。在这种情况下，列表包含标记：

```py
for i, u in enumerate(vocabulary):
  print(i, u)
```

您可以在图 3-15 中看到这个结果的示例。

![标记列表的示例枚举输出](img/t2pr_0315.png)

###### 图 3-15. 标记列表的示例枚举输出

接下来，您需要将其制作成一个字典。字典实际上是一组键值对，用作查找表：当您给出一个键时，它会返回与该键对应的值。构建字典的表示法，键是标记，值是整数：

```py
char_to_index = {u:i for i, u in enumerate(vocabulary)}
```

输出将类似于图 3-16。

这个字典用于将文本转换为整数。在推理时，模型输出也是整数格式。因此，如果希望输出为文本，则需要一个反向字典将整数映射回字符。要做到这一点，只需颠倒`i`和`u`的顺序：

```py
index_to_char = {i:u for i, u in enumerate(vocabulary)}
```

![字符到索引字典的示例](img/t2pr_0316.png)

###### 图 3-16. 字符到索引字典的示例

标记化是大多数自然语言处理问题中最基本和必要的步骤。文本生成模型不会生成纯文本作为输出；它会生成一系列整数作为输出。为了使这一系列索引映射到字母（标记），您需要一个查找表。`index_to_char`就是专门为此目的构建的。使用`index_to_char`，您可以通过键查找每个字符（标记），其中键是模型输出的索引。没有`index_to_char`，您将无法将模型输出映射回可读的纯文本格式。

# 总结

在本章中，您学习了如何处理一些最常见的数据结构：表格、图像和文本。表格数据集（结构化的、类似 CSV 的数据）非常常见，通常从数据库查询返回，并经常用作训练数据。您学会了如何处理这些结构中不同数据类型的列，以及如何通过交叉感兴趣的列来建模特征交互。

对于图像数据，您学会了在使用整个图像集训练模型之前需要标准化图像大小和像素值，以及需要跟踪图像标签。

文本数据在格式和用途方面是最多样化的数据类型。然而，无论数据是用于文本分类、翻译还是问答模型，标记化和字典构建过程都非常常见。本章描述的方法和方法并不是详尽或全面的；相反，它们代表了处理这些数据类型时的“基本要求”。

¹ 用户 Joe Kington 在[StackOverflow](https://oreil.ly/1iVv1)上的回答，2016 年 1 月 13 日，2020 年 10 月 23 日访问。
