# 第四章：可重用的模型元素

开发一个 ML 模型可能是一项艰巨的任务。除了任务的数据工程方面，您还需要了解如何构建模型。在 ML 的早期阶段，基于树的模型（如随机森林）是将直接应用于表格数据集的分类或回归任务的王者，模型架构是由与模型初始化相关的参数确定的。这些参数，称为超参数，包括森林中的决策树数量以及在拆分节点时每棵树考虑的特征数量。然而，将某些类型的数据，如图像或文本，转换为表格形式并不是直截了当的：图像可能具有不同的尺寸，文本长度也不同。这就是为什么深度学习已经成为图像和文本分类的事实标准模型架构的原因。

随着深度学习架构的流行，围绕它形成了一个社区。创建者为学术和 Kaggle 挑战构建和测试了不同的模型结构。许多人已经将他们的模型开源，以便进行迁移学习-任何人都可以将它们用于自己的目的。

例如，ResNet 是在 ImageNet 数据集上训练的图像分类模型，该数据集约为 150GB，包含超过一百万张图像。这些数据中的标签包括植物、地质形态、自然物体、体育、人物和动物。那么您如何重用 ResNet 模型来对您自己的图像集进行分类，即使具有不同的类别或标签？

像 ResNet 这样的开源模型具有非常复杂的结构。虽然源代码可以在 GitHub 等网站上供任何人访问，但下载源代码并不是复制或重用这些模型的最用户友好的方式。通常还有其他依赖项需要克服才能编译或运行源代码。那么我们如何使这些模型对非专家可用和可用？

TensorFlow Hub（TFH）旨在解决这个问题。它通过将各种 ML 模型作为库或 Web API 调用免费提供，从而实现迁移学习。任何人都可以写一行代码来加载模型。所有模型都可以通过简单的 Web 调用调用，然后整个模型将下载到您的源代码运行时。您不需要自己构建模型。

这绝对节省了开发和训练时间，并增加了可访问性。它还允许用户尝试不同的模型并更快地构建自己的应用程序。迁移学习的另一个好处是，由于您不是从头开始重新训练整个模型，因此您可能不需要高性能的 GPU 或 TPU 即可开始。

在本章中，我们将看一看如何轻松利用 TensorFlow Hub。所以让我们从 TFH 的组织方式开始。然后您将下载 TFH 预训练的图像分类模型之一，并看看如何将其用于您自己的图像。

# 基本的 TensorFlow Hub 工作流程

[TensorFlow Hub](https://oreil.ly/dQxxy)（图 4-1）是由 Google 策划的预训练模型的存储库。用户可以将任何模型下载到自己的运行时，并使用自己的数据进行微调和训练。

![TensorFlow Hub 主页](img/t2pr_0401.png)

###### 图 4-1\. TensorFlow Hub 主页

要使用 TFH，您必须通过熟悉的 Pythonic `pip install`命令在您的 Python 单元格或终端中安装它：

```py
pip install --upgrade tensorflow_hub
```

然后您可以通过导入它在您的源代码中开始使用它：

```py
import tensorflow_hub as hub
```

首先，调用模型：

```py
model = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim128/2")
```

这是一个预训练的文本嵌入模型。*文本嵌入*是将文本字符串映射到数字表示的多维向量的过程。您可以给这个模型四个文本字符串：

```py
embeddings = model(["The rain in Spain.", "falls",
                      "mainly", "In the plain!"])
```

在查看结果之前，请检查模型输出的形状：

```py
print(embeddings.shape)
```

应该是：

```py
(4, 128)
```

有四个输出，每个输出长度为 128 个单位。图 4-2 显示其中一个输出：

```py
print(embeddings[0])
```

![文本嵌入输出](img/t2pr_0402.png)

###### 图 4-2. 文本嵌入输出

正如在这个简单示例中所示，您没有训练这个模型。您只是加载它并用它来处理您自己的数据。这个预训练模型简单地将每个文本字符串转换为一个 128 维的向量表示。

在 TensorFlow Hub 首页，点击“Models”选项卡。如您所见，TensorFlow Hub 将其预训练模型分类为四个问题领域：图像、文本、视频和音频。

图 4-3 展示了迁移学习模型的一般模式。

从图 4-3 中，您可以看到预训练模型（来自 TensorFlow Hub）被夹在输入层和输出层之间，输出层之前可能还有一些可选层。

![迁移学习的一般模式](img/t2pr_0403.png)

###### 图 4-3. 迁移学习的一般模式

要使用任何模型，您需要解决一些重要的考虑因素，例如输入和输出：

输入层

输入数据必须被正确格式化（或“塑造”），因此请特别注意每个模型的输入要求（在描述各个模型的网页上的“Usage”部分中找到）。以[ResNet 特征向量](https://oreil.ly/6xGeP)为例：Usage 部分说明了输入图像所需的大小和颜色值，以及输出是一批特征向量。如果您的数据不符合要求，您需要应用一些数据转换技术，这些技术可以在“为处理准备图像数据”中学到。

输出层

另一个重要且必要的元素是输出层。如果您希望使用自己的数据重新训练模型，这是必须的。在之前展示的简单嵌入示例中，我们没有重新训练模型；我们只是输入了一些文本字符串来查看模型的输出。输出层的作用是将模型的输出映射到最可能的标签，如果问题是分类问题的话。如果是回归问题，那么它的作用是将模型的输出映射到一个数值。典型的输出层称为“密集层”，可以是一个节点（用于回归或二元分类）或多个节点（例如用于多类分类）。

可选层

可选地，您可以在输出层之前添加一个或多个层以提高模型性能。这些层可以帮助您提取更多特征以提高模型准确性，例如卷积层（Conv1D、Conv2D）。它们还可以帮助防止或减少模型过拟合。例如，通过随机将输出设置为零，dropout 可以减少过拟合。如果一个节点输出一个数组，例如[0.5, 0.1, 2.1, 0.9]，并且您设置了 0.25 的 dropout 比率，那么在训练过程中，根据随机机会，数组中的四个值中的一个将被设置为零；例如，[0.5, 0, 2.1, 0.9]。再次强调，这是可选的。您的训练不需要它，但它可能有助于提高模型的准确性。

# 通过迁移学习进行图像分类

我们将通过一个使用迁移学习的图像分类示例来进行讲解。在这个示例中，您的图像数据包括五类花。您将使用 ResNet 特征向量作为预训练模型。我们将解决以下常见任务：

+   模型要求

+   数据转换和输入处理

+   TFH 模型实现

+   输出定义

+   将输出映射到纯文本格式

## 模型要求

让我们看看[ResNet v1_101 特征向量](https://oreil.ly/70grM)模型。这个网页包含了一个概述、一个下载 URL、说明以及最重要的是您需要使用该模型的代码。

在使用部分中，您可以看到要加载模型，您只需要将 URL 传递给`hub.KerasLayer`。使用部分还包括模型要求。默认情况下，它期望输入图像，写为形状数组[高度，宽度，深度]，为[224, 224, 3]。像素值应在范围[0, 1]内。作为输出，它提供了具有节点数的`Dense`层，反映了训练图像中类别的数量。

## 数据转换和输入处理

您的任务是将图像转换为所需的形状，并将像素比例标准化到所需范围内。正如我们所见，图像通常具有不同的大小和像素值。每个 RGB 通道的典型彩色 JPEG 图像像素值可能在 0 到 225 之间。因此，我们需要操作来将图像大小标准化为[224, 224, 3]，并将像素值标准化为[0, 1]范围。如果我们在 TensorFlow 中使用`ImageDataGenerator`，这些操作将作为输入标志提供。以下是如何加载图像并创建生成器：

1.  首先加载库：

    ```py
    import tensorflow as tf
    import tensorflow_hub as hub
    import numpy as np
    import matplotlib.pylab as plt
    ```

1.  加载所需的数据。在这个例子中，让我们使用 TensorFlow 提供的花卉图像：

    ```py
    data_dir = tf.keras.utils.get_file(
        'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/
    example_images/flower_photos.tgz',
        untar=True)
    ```

1.  打开`data_dir`并找到图像。您可以在文件路径中看到文件结构：

    ```py
    !ls -lrt /root/.keras/datasets/flower_photos
    ```

    这是将显示的内容：

    ```py
    total 620
    -rw-r----- 1 270850 5000 418049 Feb  9  2016 LICENSE.txt
    drwx------ 2 270850 5000  45056 Feb 10  2016 tulips
    drwx------ 2 270850 5000  40960 Feb 10  2016 sunflowers
    drwx------ 2 270850 5000  36864 Feb 10  2016 roses
    drwx------ 2 270850 5000  53248 Feb 10  2016 dandelion
    drwx------ 2 270850 5000  36864 Feb 10  2016 daisy
    ```

    有五类花卉。每个类对应一个目录。

1.  定义一些全局变量来存储像素值和*批量大小*（训练图像批次中的样本数）。目前您只需要图像的高度和宽度，不需要图像的第三个维度：

    ```py
    pixels =224
    BATCH_SIZE = 32
    IMAGE_SIZE = (pixels, pixels)
    NUM_CLASSES = 5
    ```

1.  指定图像标准化和用于交叉验证的数据分数。将一部分训练数据保留用于交叉验证是一个好主意，这是通过每个时代评估模型训练过程的一种方法。在每个训练时代结束时，模型包含一组经过训练的权重和偏差。此时，用于交叉验证的数据，模型从未见过，可以用作模型准确性的测试：

    ```py
    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, 
    batch_size=BATCH_SIZE,
    interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.
    ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=False, 
        **dataflow_kwargs)
    ```

    `ImageDataGenerator`定义和生成器实例都以字典格式接受我们的参数。重新缩放因子和验证分数进入生成器定义，而标准化图像大小和批量大小进入生成器实例。

    `插值`参数表示生成器需要将图像数据重新采样到`target_size`，即 224×224 像素。

    现在，对训练数据生成器执行相同操作：

    ```py
    train_datagen = valid_datagen
    train_generator = train_datagen.flow_from_directory(
        data_dir, subset="training", shuffle=True, 
        **dataflow_kwargs)
    ```

1.  识别类索引到类名的映射。由于花卉类别被编码在索引中，您需要一个映射来恢复花卉类别名称：

    ```py
    labels_idx = (train_generator.class_indices)
    idx_labels = dict((v,k) for k,v in labels_idx.items())
    ```

    您可以显示`idx_labels`以查看这些类是如何映射的：

    ```py
    idx_labels

    {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers',
    4: 'tulips'}
    ```

现在您已经对图像数据进行了标准化和标准化。图像生成器已被定义并实例化用于训练和验证数据。您还具有标签查找来解码模型预测，并且已准备好使用 TFH 实现模型。

## 使用 TensorFlow Hub 实现模型

正如您在图 4-3 中看到的，预训练模型被夹在输入层和输出层之间。您可以相应地定义这个模型结构：

```py
model = tf.keras.Sequential([
     tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v1_101/
feature_vector/4", trainable=False),
     tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', 
name = 'flower_class') 
])

model.build([None, 224, 224, 3]) !!C04!!
```

注意这里有几点：

+   有一个输入层，定义图像的输入形状为[224, 224, 3]。

+   当调用`InputLayer`时，`trainable`应设置为 False。这表示您希望重用预训练模型的当前值。

+   有一个名为`Dense`的输出层提供模型输出（这在摘要页面的使用部分中有描述）。

构建模型后，您可以开始训练。首先，指定损失函数并选择优化器：

```py
model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
loss=tf.keras.losses.CategoricalCrossentropy(
from_logits=True, 
label_smoothing=0.1),
metrics=['accuracy'])
```

然后指定用于训练数据和交叉验证数据的批次数：

```py
steps_per_epoch = train_generator.samples // 
train_generator.batch_size
validation_steps = valid_generator.samples // 
valid_generator.batch_size
```

然后开始训练过程：

```py
model.fit(
    train_generator,
    epochs=5, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps)
```

在经过指定的所有时代运行训练过程后，模型已经训练完成。

## 定义输出

根据使用指南，输出层`Dense`由一定数量的节点组成，反映了预期图像中有多少类别。这意味着每个节点为该类别输出一个概率。您的任务是找到这些概率中哪一个最高，并使用`idx_labels`将该节点映射到花卉类别。回想一下，`idx_labels`字典如下所示：

```py
{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 
4: 'tulips'}
```

`Dense`层的输出由五个节点以完全相同的顺序组成。您需要编写几行代码将具有最高概率的位置映射到相应的花卉类别。

## 将输出映射到纯文本格式

让我们使用验证图像来更好地了解如何将模型预测输出映射到每个图像的实际类别。您将使用`predict`函数对这些验证图像进行评分。检索第一批次的 NumPy 数组：

```py
sample_test_images, ground_truth_labels = next(valid_generator)

prediction = model.predict(sample_test_images)
```

在交叉验证数据中有 731 张图像和 5 个对应的类别。因此，输出形状为[731, 5]：

```py
array([[9.9994004e-01, 9.4704428e-06, 3.8405190e-10, 5.0486942e-05,
        1.0701914e-08],
       [5.9500107e-06, 3.1842374e-06, 3.5622744e-08, 9.9999082e-01,
        3.0683900e-08],
       [9.9994218e-01, 5.9974178e-07, 5.8693445e-10, 5.7049790e-05,
        9.6709634e-08],
       ...,
       [3.1268091e-06, 9.9986601e-01, 1.5343730e-06, 1.2935932e-04,
        2.7383029e-09],
       [4.8439368e-05, 1.9247003e-05, 1.8034354e-01, 1.6394027e-02,
        8.0319476e-01],
       [4.9799957e-07, 9.9232978e-01, 3.5823192e-08, 7.6697678e-03,
        1.7666844e-09]], dtype=float32)
```

每行代表了图像类别的概率分布。对于第一张图像，最高概率为 1.0701914e-08（在上述代码中突出显示），位于该行的最后位置，对应于该行的索引 4（请记住，索引的编号从 0 开始）。

现在，您需要使用以下代码找到每行中最高概率出现的位置：

```py
predicted_idx = tf.math.argmax(prediction, axis = -1)
```

如果您使用`print`命令显示结果，您将看到以下内容：

```py
print (predicted_idx)

<tf.Tensor: shape=(731,), dtype=int64, numpy=
array([0, 3, 0, 1, 0, 4, 4, 1, 2, 3, 4, 1, 4, 0, 4, 3, 1, 4, 4, 0,
       …
       3, 2, 1, 4, 1])>
```

现在，对该数组中的每个元素应用`idx_labels`的查找。对于每个元素，使用一个函数：

```py
def find_label(idx):
    return idx_labels[idx]
```

要将函数应用于 NumPy 数组的每个元素，您需要对函数进行矢量化：

```py
find_label_batch = np.vectorize(find_label)
```

然后将此矢量化函数应用于数组中的每个元素：

```py
result = find_label_batch(predicted_idx)
```

最后，将结果与图像文件夹和文件名并排输出，以便保存以供报告或进一步调查。您可以使用 Python pandas DataFrame 操作来实现这一点：

```py
import pandas as pd
predicted_label = result_class.tolist()
file_name = valid_generator.filenames

results=pd.DataFrame({"File":file_name,
                      "Prediction":predicted_label})
```

让我们看看`results`数据框，它是 731 行×2 列。

|   | 文件 | 预测 |
| --- | --- | --- |
| 0 | daisy/100080576_f52e8ee070_n.jpg | daisy | 雏菊 |
| 1 | daisy/10140303196_b88d3d6cec.jpg | sunflowers | 向日葵 |
| 2 | daisy/10172379554_b296050f82_n.jpg | daisy | 雏菊 |
| 3 | daisy/10172567486_2748826a8b.jpg | dandelion | 玛丽金花 |
| 4 | daisy/10172636503_21bededa75_n.jpg | daisy | 雏菊 |
| ... | ... | ... |
| 726 | tulips/14068200854_5c13668df9_m.jpg | sunflowers | 向日葵 |
| 727 | tulips/14068295074_cd8b85bffa.jpg | roses | 玫瑰 |
| 728 | tulips/14068348874_7b36c99f6a.jpg | dandelion | 郁金香 |
| 729 | tulips/14068378204_7b26baa30d_n.jpg | tulips | 郁金香 |
| 730 | tulips/14071516088_b526946e17_n.jpg | dandelion | 玛丽金花 |

## 评估：创建混淆矩阵

混淆矩阵通过比较模型输出和实际情况来评估分类结果，是了解模型表现的最简单方法。让我们看看如何创建混淆矩阵。

您将使用 pandas Series 作为构建混淆矩阵的数据结构：

```py
y_actual = pd.Series(valid_generator.classes)
y_predicted = pd.Series(predicted_idx)
```

然后，您将再次利用 pandas 生成矩阵：

```py
pd.crosstab(y_actual, y_predicted, rownames = ['Actual'],
colnames=['Predicted'], margins=True)
```

图 4-4 显示了混淆矩阵。每行代表了实际花标签的分布情况。例如，看第一行，您会注意到总共有 126 个样本实际上是类别 0，即雏菊。模型正确地将这些图像中的 118 个预测为类别 0；四个被错误分类为类别 1，即蒲公英；一个被错误分类为类别 2，即玫瑰；三个被错误分类为类别 3，即向日葵；没有被错误分类为类别 4，即郁金香。

![花卉图像分类的混淆矩阵](img/t2pr_0404.png)

###### 图 4-4\. 花卉图像分类的混淆矩阵

接下来，使用`sklearn`库为每个图像类别提供统计报告：

```py
from sklearn.metrics import classification_report
report = classification_report(truth, predicted_results)
print(report)
              precision    recall  f1-score   support

           0       0.90      0.94      0.92       126
           1       0.93      0.87      0.90       179
           2       0.85      0.86      0.85       128
           3       0.85      0.88      0.86       139
           4       0.86      0.86      0.86       159

    accuracy                           0.88       731
   macro avg       0.88      0.88      0.88       731
weighted avg       0.88      0.88      0.88       731
```

这个结果表明，当对雏菊（类别 0）进行分类时，该模型的性能最佳，f1 分数为 0.92。在对玫瑰（类别 2）进行分类时，其性能最差，f1 分数为 0.85。“支持”列显示了每个类别中的样本量。

## 总结

您刚刚完成了一个使用来自 TensorFlow Hub 的预训练模型的示例项目。您添加了必要的输入层，执行了数据归一化和标准化，训练了模型，并对一批图像进行了评分。

这个经验表明了满足模型的输入和输出要求的重要性。同样重要的是，要密切关注预训练模型的输出格式。（这些信息都可以在 TensorFlow Hub 网站上的模型文档页面找到。）最后，您还需要创建一个函数，将模型的输出映射到纯文本，以使其具有意义并可解释。

# 使用 tf.keras.applications 模块进行预训练模型

另一个为您自己使用找到预训练模型的地方是 `tf.keras.applications` 模块（请参阅[可用模型列表](https://oreil.ly/HQJBl)）。当 Keras API 在 TensorFlow 中可用时，该模块成为 TensorFlow 生态系统的一部分。

每个模型都带有预训练的权重，使用它们和使用 TensorFlow Hub 一样简单。Keras 提供了方便地微调模型所需的灵活性。通过使模型中的每一层可访问，`tf.keras.applications` 让您可以指定哪些层要重新训练，哪些层保持不变。

## 使用 tf.keras.applications 实现模型

与 TensorFlow Hub 一样，您只需要一行代码从 Keras 模块加载一个预训练模型：

```py
base_model = tf.keras.applications.ResNet101V2(
input_shape = (224, 224, 3), 
include_top = False, 
weights = 'imagenet')
```

注意 `include_top` 输入参数。请记住，您需要为自己的数据添加一个输出层。通过将 `include_top` 设置为 False，您可以为分类输出添加自己的 `Dense` 层。您还将从 `imagenet` 初始化模型权重。

然后将 `base_model` 放入一个顺序架构中，就像您在 TensorFlow Hub 示例中所做的那样：

```py
model2 = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(NUM_CLASSES, 
  activation = 'softmax', 
  name = 'flower_class')
])
```

添加 `GlobalAveragePooling2D`，将输出数组平均为一个数值，然后将其发送到最终的 `Dense` 层进行预测。

现在编译模型并像往常一样启动训练过程：

```py
model2.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
  loss=tf.keras.losses.CategoricalCrossentropy(
  from_logits=True, label_smoothing=0.1),
  metrics=['accuracy']
)

model2.fit(
    train_generator,
    epochs=5, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps)
```

要对图像数据进行评分，请按照您在 “将输出映射到纯文本格式” 中所做的步骤进行。

## 从 tf.keras.applications 微调模型

如果您希望通过释放一些基础模型的层进行训练来尝试您的训练例程，您可以轻松地这样做。首先，您需要找出基础模型中有多少层，并将基础模型指定为可训练的：

```py
print("Number of layers in the base model: ", 
       len(base_model.layers))
base_model.trainable = True

Number of layers in the base model:  377
```

如所示，在这个版本的 ResNet 模型中，有 377 层。通常我们从模型末尾附近的层开始重新训练过程。在这种情况下，将第 370 层指定为微调的起始层，同时保持在第 300 层之前的权重不变：

```py
fine_tune_at = 370

for layer in base_model.layers[: fine_tune_at]:
  layer.trainable = False
```

然后使用 `Sequential` 类将模型组合起来：

```py
model3 = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(NUM_CLASSES, 
  activation = 'softmax', 
  name = 'flower_class')
])
```

###### 提示

您可以尝试使用 `tf.keras.layers.Flatten()` 而不是 `tf.keras.layers.GlobalAveragePooling2D()`，看看哪一个给您一个更好的模型。

编译模型，指定优化器和损失函数，就像您在 TensorFlow Hub 中所做的那样：

```py
model3.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
  loss=tf.keras.losses.CategoricalCrossentropy(
  from_logits=True, 
  label_smoothing=0.1),
  metrics=['accuracy']
)
```

启动训练过程：

```py
fine_tune_epochs = 5
steps_per_epoch = train_generator.samples // 
train_generator.batch_size
validation_steps = valid_generator.samples // 
valid_generator.batch_size
model3.fit(
    train_generator,
    epochs=fine_tune_epochs, 
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps)
```

由于您已经释放了更多基础模型的层进行重新训练，这个训练可能需要更长时间。训练完成后，对测试数据进行评分，并按照 “将输出映射到纯文本格式” 和 “评估：创建混淆矩阵” 中描述的方式比较结果。

# 结束

在这一章中，您学习了如何使用预训练的深度学习模型进行迁移学习。有两种方便的方式可以访问预训练模型：TensorFlow Hub 和 `tf.keras.applications` 模块。两者都简单易用，具有优雅的 API 和风格，可以快速开发模型。然而，用户需要正确地塑造他们的输入数据，并提供一个最终的 `Dense` 层来处理模型输出。

有大量免费可访问的预训练模型，具有丰富的库存，您可以使用它们来处理自己的数据。利用迁移学习来利用它们，让您花费更少的时间来构建、训练和调试模型。
