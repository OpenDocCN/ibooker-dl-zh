# 第十章。改善建模体验：公平性评估和超参数调整

使 ML 模型运行良好是一个迭代过程。它需要多轮调整模型参数、架构和训练持续时间。当然，您必须使用可用的数据，但理想情况下，您希望训练数据是平衡的。换句话说，它应该包含相等数量的类或跨范围的均匀分布。

为什么这种平衡很重要？因为如果数据中的任何特征存在偏差，那么训练的模型将重现该偏差。这被称为*模型偏差*。

想象一下，您正在训练一个识别狗的模型。如果您的训练图像中有 99 个负例和 1 个正例，即只有一张实际的狗图像，那么模型将每次都简单地预测负结果，有 99%的准确率。模型学会了在训练过程中最小化错误，最简单的方法是产生一个负预测，简而言之，每次都猜“不是狗”。这被称为数据不平衡问题，在现实世界中很普遍；这也是一个复杂的主题，我无法在这里充分阐述。它需要许多不同的方法，包括通过一种称为*数据增强*的技术添加合成数据。

在本章中，我将向您介绍公平性指标，这是一个用于评估模型偏差的新工具（截至撰写本文）。它是 TensorFlow Model Analysis 库的一部分，并可用于 Jupyter 笔记本。

您还将学习如何执行*超参数调整*。*超参数*是模型架构和模型训练过程中的变量。有时您想要尝试不同的值或实现选择，但不知道哪个对您的模型最好。为了找出，您需要评估许多超参数组合的模型性能。我将向您展示如何使用 Keras Tuner 库进行超参数调整的新方法。该库与 TensorFlow 2 的 Keras API 无缝配合。它非常灵活，易于设置为训练过程的一部分。我们将在下一节开始设置公平性指标。

###### 提示

模型偏差及其现实生活后果是众所周知的。模型偏差最显著的例子之一是 COMPAS（Correctional Offender Management Profiling for Alternative Sanctions）框架，该框架曾用于美国法院系统预测累犯。由于训练数据，该模型对黑人被告的[假阳性预测是白人被告的两倍](https://oreil.ly/1FdFy)。如果您对公平性有兴趣，可以查看 Aileen Nielsen 的《*Practical Fairness*》（O’Reilly，2020）和 Trisha Mahoney、Kush R. Varshney 和 Michael Hind 的《*AI Fairness*》（O’Reilly，2020）。

# 模型公平性

您需要安装 TensorFlow Model Analysis 库，该库不是 TensorFlow 2.4.1 常规发行版的一部分。您可以通过`pip install`命令下载并安装它：

```py
pip install tensorflow-model-analysis
```

您还需要安装`protobuf`库来解析您选择的模型指标：

```py
pip install protobuf
```

该库使您能够在测试数据上显示和审查模型统计信息，以便检测模型预测中的任何偏差。

为了说明这一点，我们将再次使用*Titanic*数据集。在第三章中，您使用此数据集构建了一个模型来预测乘客的生存。这个小数据集包含有关每位乘客的几个特征，是一个很好的起点。

我们将生存视为一个离散的结果：某人要么幸存，要么不幸存。然而，对于模型来说，我们真正意味着的是基于乘客的给定特征的生存概率。回想一下，您构建的模型是一个逻辑回归模型，输出是一个二元结果（幸存或未幸存）的概率。一门[Google 课程](https://oreil.ly/8ciQJ)这样表达：

> 为了将逻辑回归值映射到一个二进制类别，您必须定义一个*分类阈值*（也称为*决策阈值*）。高于该阈值的值表示[积极]；低于该值表示[消极]。人们很容易假设分类阈值应该始终为 0.5，但阈值是依赖于问题的，并且是您必须调整的值。

决定阈值是用户的责任。理解这一点的直观方式是，生存概率为 0.51 并不保证生存；它仍然意味着 49%的*不*幸存的机会。同样，生存概率为 0.49 并不是零。一个好的阈值是能够最小化两个方向的误分类。因此，阈值是一个用户确定的参数。通常，在您的模型训练和测试过程中，您将尝试几个不同的阈值，并查看哪个在测试数据中给出了最正确的分类。对于这个数据集，您可以从不同的阈值列表开始，例如 0.1、0.5 和 0.9。对于每个阈值，一个积极的结果——即，一个预测概率高于阈值的预测——表示这个个体幸存的预测。

回想一下，*Titanic*数据集看起来像图 10-1 所示。

![用于训练的 Titanic 数据集](img/t2pr_1001.png)

###### 图 10-1\. *用于训练的 Titanic*数据集

图 10-1 中的每一行代表一个乘客和几个相应的特征。模型的目标是根据这些特征预测“survived”列中的值。在训练数据中，这一列是二进制的，1 表示乘客幸存，0 表示乘客未幸存。测试数据也是由*Titanic*数据集提供的一个单独的分区。

## 模型训练和评分

让我们从“为训练准备表格数据”之后继续，您已经完成了模型训练。在继续之前再次运行这些导入语句：

```py
import functools
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pprint
import tensorflow_model_analysis as tfma
from google.protobuf import text_format
```

一旦模型训练完成（您在第三章中完成了），您可以使用它来预测测试数据集中每个乘客的生存概率：

```py
prediction_raw = model.predict(test_ds)
```

这个命令`prediction_raw`会生成一个 NumPy 数组，其中包含每个乘客的概率值：

```py
array([[0.6699142 ],
       [0.6239286 ],
       [0.06013593],
…..
       [0.10424912]], dtype=float32)
```

现在让我们将这个数组转换成一个 Python 列表，并将其附加为测试数据集的新列：

```py
prediction_list = prediction_raw.squeeze().tolist()
test_df['predicted'] = prediction_list
```

新列名为“predicted”，是最后一列。为了可见性，您可能希望通过将这最后一列移动到第一列旁边的“survived”列来重新排列列。

```py
# Put predicted as first col, next to survived.
cols = list(test_df.columns)
cols = [cols[-1]] + cols[:-1]
test_df = test_df[cols]
```

现在`test_df`看起来像图 10-2，我们可以轻松地将模型的预测与真实结果进行比较。

![附加了预测的 Titanic 测试数据集](img/t2pr_1002.png)

###### 图 10-2\. *附加了预测的 Titanic*测试数据集

## 公平性评估

要调查模型预测的公平性，您需要对您的用例和用于训练的数据有一个很好的理解。仅仅看数据本身可能不会给您足够的背景、上下文或情境意识来调查公平性或模型偏差。因此，您必须了解用例、模型的目的、谁在使用它以及如果模型预测错误可能产生的潜在现实影响，这是至关重要的。

*Titanic*有三个船舱等级：头等舱是最昂贵的，二等舱位于中间，而三等舱，或者称为船舱，是最便宜的并位于下层甲板。有充分的证据表明，大多数幸存者是女性，并且在头等舱。我们也知道性别和等级在上救生艇的选择过程中起着重要作用。该选择过程优先考虑了妇女和儿童，而不是男性。由于这个背景是如此出名，这个数据集适合作为一个教学示例来调查模型偏差。

在进行预测时，正如我在介绍中提到的，模型不可避免地重现训练数据中的任何偏见或不平衡。因此，我们调查的一个有趣问题是：模型在不同性别和等级的乘客生存预测方面表现如何？

让我们从以下代码块开始`eval_config`，定义我们的调查：

```py
eval_config = text_format.Parse("""    ①
  model_specs {                        ②
    prediction_key: 'predicted',
    label_key: 'survived'
  }
  metrics_specs {                      ③
    metrics {class_name: "AUC"}
    metrics {
      class_name: "FairnessIndicators"
      config: '{"thresholds": [0.1, 0.50, 0.90]}'
    }
    metrics { class_name: "ExampleCount" }
  }

  slicing_specs {                      ④
    feature_keys: ['sex', 'class']
  }
  slicing_specs {}
  """, tfma.EvalConfig())              ⑤

```

①

`eval_config`对象必须格式化为协议缓冲区数据类型，这就是为什么您需要导入`text_format`来解析定义。

②

指定`model_specs`以记录要比较的两列：“predicted”和“survived”。

③

定义分类准确性的指标以及基于生存概率分配生存状态的三个阈值。

④

这是您声明要用来调查模型偏差的特征的地方。`slicing_specs`中的`feature_keys`保存要检查偏差的特征列表：在我们的案例中是“sex”和“class”。由于性别有两个唯一值，而不同的班级有三个不同的类别，公平性指标将评估六个不同交互组的模型偏差。如果只列出一个特征，公平性指标将仅评估该特征上的偏差。

⑤

所有这些信息都包含在`“““ ”””`三重双引号中，这使得它成为一个纯文本表示。这个文本字符串被合并到`tfma.EvalConfig`消息中。

现在定义一个输出路径来存储公平性指标的结果：

```py
OUTPUT_PATH = './titanic-fairness'
```

然后运行模型分析例程：

```py
eval_result = tfma.analyze_raw_data(
  data= test_df,
  eval_config=eval_config,
  output_path=OUTPUT_PATH)
```

从前面的代码中，您可以看到`test_df`是测试数据，其中添加了预测。您将使用`tfma.analyze_raw_data`函数执行公平性分析。

###### 提示

如果您在本地 Jupyter Notebook 中运行此示例，则需要启用 Jupyter Notebook 以显示公平性指标 GUI。在下一个单元格中，输入以下命令：

```py
!jupyter nbextension enable tensorflow_model_analysis 
--user –py

```

如果您正在使用 Google Colab 笔记本，则此步骤是不必要的。

## 渲染公平性指标

现在您已经准备好查看您的`eval_result`。运行此命令：

```py
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
 eval_result)
```

您将在笔记本中看到公平性指标的交互式 GUI（参见图 10-3）。

![公平性指标交互式 GUI](img/t2pr_1003.png)

###### 图 10-3. 公平性指标交互式 GUI

在这个图中，选择了假阳性率指标。在*Titanic*数据集的背景下，*假阳性*表示模型预测乘客会生存，但实际上他们没有。

让我们看看这个 GUI。在左侧面板（如图 10-3 所示），您可以看到所有可用的指标。在右侧面板中，您会看到所选指标的条形图。在`slice_specs`中指示特征组合（性别和等级）。

在阈值为 0.5（概率大于 0.5 被视为正面，即生存）时，头等舱男性乘客和二三等舱女性乘客的假阳性率很高。

在条形图下方的表格中，您可以更详细地查看实际指标和样本大小（参见图 10-4）。 

![显示假阳性摘要的公平性指标仪表板](img/t2pr_1004.png)

###### 图 10-4\. 显示假阳性摘要的公平性指标仪表板

毫不奇怪，模型正确预测了所有头等舱女性乘客的幸存，您可以从实际情况（“survived”）列中看到。

那么为什么二等和三等舱的男性乘客产生了 0 的假阳性率？让我们查看`test_df`中的实际结果以了解原因。执行以下命令选择二等舱住宿的男性乘客：

```py
sel_df = test_df[(test_df['sex'] == 'male') & (test_df['class'] == 
'Second')]
```

然后显示`sel_df`：

```py
sel_df
```

图 10-5 显示了所有二等舱男性乘客的结果列表。让我们用它来寻找假阳性。在这个组中，是否有任何乘客没有幸存（即，“survived”列显示 0），但模型预测的概率大于阈值（0.5）？没有。因此，公平性指标表明假阳性率为 0。

![泰坦尼克号测试数据集的二等舱男性乘客](img/t2pr_1005.png)

###### 图 10-5\. *泰坦尼克号*测试数据集的二等舱男性乘客

您可能会注意到，一些二等舱男性乘客确实幸存了，但模型预测他们的生存概率低于 0.5 的阈值。这些被称为*假阴性*案例：他们实际上幸存了，但模型预测他们没有。简而言之，他们打破了规律！因此，让我们取消假阳性指标并检查我们的假阴性指标，看看在性别和阶级组合中它是什么样子（参见图 10-6）。

正如您在图 10-6 中所看到的，在当前阈值 0.5 的情况下，二等和三等舱的男性和男孩有很高的假阴性率。

![泰坦尼克号测试数据集的假阴性指标](img/t2pr_1006.png)

###### 图 10-6\. *泰坦尼克号*测试数据集的假阴性指标

这个练习清楚地表明，模型在性别和阶级方面存在偏见。该模型对头等舱女性乘客表现最佳，这反映了数据中明显的偏斜：几乎所有头等舱女性乘客都幸存了，而其他人的机会则不那么明确。您可以通过以下语句检查训练数据以确认这种偏斜，该语句将训练数据按“sex”，“class”和“survived”列分组：

```py
 TRAIN_DATA_URL = 
https://storage.googleapis.com/tf-datasets/titanic/train.csv
train_file_path = tf.keras.utils.get_file("train.csv", 
TRAIN_DATA_URL)
train_df = pd.read_csv(train_file_path, 
header='infer')
train_df.groupby(['sex', 'class', 'survived' ]).size().
reset_index(name='counts')
```

这将产生图 10-7 中显示的摘要。

如您所见，69 名头等舱女性乘客中只有两人死亡。您根据这些数据训练了您的模型，因此模型很容易学会，只要阶级和性别指示为头等和女性，它就应该预测高概率的生存。您还可以看到，第三等舱的女性乘客中，52 人幸存，41 人死亡，没有享受到同样有利的结果。这是一个典型的数据标签不平衡（“survived”列）导致模型偏见的案例。在所有可能的性别-阶级组合中，没有其他组合的胜算像头等舱女性乘客那样好。这使得模型更难正确预测。

![泰坦尼克号训练数据集中乘客组的生存计数摘要](img/t2pr_1007.png)

###### 图 10-7\. *泰坦尼克号*训练数据集中乘客组的生存计数摘要

使用公平性指标时，您可以切换阈值下拉框以在同一条形图中显示三个不同阈值的指标，如图 10-8 所示。

![展示所有阈值的公平性指标指标](img/t2pr_1008.png)

###### 图 10-8\. 展示所有阈值的公平性指标指标

为了易读性，每个阈值都有颜色编码。您可以将鼠标悬停在柱状图上以确定颜色分配。这有助于您解释不同阈值对每个指标的影响。

从这个例子中，我们可以得出以下结论：

+   乘客的特征，如性别和舱位等级（作为社会阶级的替代品），主要决定了结果。

+   对于头等舱和二等舱的女性乘客，模型的准确性要高得多，这强烈表明模型偏见是由性别和阶级驱动的。

+   生存概率偏向于某些性别和某些阶级。这被称为*数据不平衡*，与历史记载的发生一致。

+   特征之间的交互作用，比如性别和阶级（称为*特征交叉*），揭示了模型偏见的更深层次。在最低（第三）阶级的妇女和女孩在训练数据和模型准确性中都没有获得有利的生存结果，更不用说在现实生活中了。

当您看到模型偏见时，您可能会尝试调整模型架构或训练策略。但这里的根本问题是训练数据中的不平衡，这反映了基于性别和阶级的现实不平等结果。如果没有为其他性别和阶级带来更公平的结果，修复模型架构或训练策略将无法创建一个公平的模型。

这个例子相对简单：这里的假设是检查性别和阶级可能会揭示模型偏见，因此值得调查。由于这是一个被广泛讨论的历史事件，大多数人至少对导致*泰坦尼克号*悲剧的背景和事件有一定的了解。然而，在许多情况下，制定一个假设来调查数据中潜在偏见可能并不简单。您的数据可能不包含个人属性或个人可识别信息（PII），您可能没有域知识或对训练数据的收集方式和影响结果的了解。数据科学家和 ML 模型开发人员应该与主题专家合作，以使他们的训练数据具有背景。有时，如果模型公平性是主要关注点，删除相关特征可能是最好和最明智的选择。

# 超参数调整

*超参数*是指定用于控制模型训练过程的值或属性。想法是您希望使用这些值的组合来训练模型，并确定最佳组合。知道哪种组合最好的唯一方法是尝试这些值，因此您希望以有效的方式迭代所有组合并将选择范围缩小到最佳组合。超参数通常应用于模型架构，比如深度学习神经网络中密集层中的节点数。超参数可以应用于训练例程，比如执行反向传播的优化器（您在第八章中学到的）在训练过程中。它们还可以应用于*学习率*，指定您希望使用增量更改来更新模型的权重和偏差的程度，从而确定反向传播在训练过程中更新权重和偏差的步长有多大。综合起来，超参数可以是数值（节点数、学习率）或非数值（优化器名称）。

截至 TensorFlow 分发 2.4.1 版本，Keras Tuner 库尚未成为标准分发的一部分。这意味着您需要安装它。您可以在命令终端或 Google Colab 笔记本中运行此安装语句：

```py
pip install keras-tuner
```

如果在 Jupyter Notebook 单元格中运行它，请使用此版本：

```py
!pip install keras-tuner
```

感叹号（！）告诉笔记本单元格将其解释为命令而不是 Python 代码。

安装完成后，您将像往常一样导入它。请注意，当您安装库时，库名称是带连字符的，但在导入语句中不使用连字符：

```py
import kerastuner as kt
```

从 Keras Tuner 的角度来看，超参数有三种数据类型：整数、项目选择（一组离散值或对象）和浮点数。

## 整数列表作为超参数

通过示例更容易看到如何使用 Keras Tuner，所以假设您想尝试在密集层中使用不同数量的节点。首先定义可能的数字，然后将该列表传递给密集层的定义：

```py
hp_node_count = hp.Int('units', min_value = 16, max_value = 32, 
step = 8)
tf.keras.layers.Dense(units = hp_node_count, activation = 'relu')
```

在上述代码中，`hp.Int`定义了一个别名：`hp_node_count`。这个别名包含一个整数列表（16、24 和 32），您将其作为`units`传递给`Dense`层。您的目标是看哪个数字效果最好。

## 项目选择作为超参数

设置超参数的另一种方法是将所有选择放在一个列表中作为离散项或选择。这可以通过`hp.Choice`函数实现：

```py
hp_units = hp.Choice('units', values = [16, 18, 21])
```

以下是通过名称指定激活函数的示例：

```py
hp_activation = hp.Choice('dense_activation', 
                values=['relu', 'tanh', 'sigmoid'])
```

## 浮点值作为超参数

在许多情况下，您可能希望在训练例程中尝试不同的小数值。如果您想为优化器选择学习率，这是非常常见的。要这样做，请使用以下命令：

```py
hp_learning_rate = hp.Float('learning_rate', 
 min_value = 1e-4, 
 max_value = 1e-2, 
 step = 1e-3)
optimizer=tf.keras.optimizers.SGD(
 lr=hp_learning_rate, 
 momentum=0.5)
```

# 端到端超参数调整

超参数调整是一个耗时的过程。它涉及尝试多种组合，每种组合都经过相同数量的 epochs 训练。对于“蛮力”方法，您需要循环遍历每个超参数组合，*然后*启动训练例程。

使用 Keras Tuner 的优势在于其提前停止实现：如果特定组合似乎没有改善结果，它将终止训练例程并转移到下一个组合。这有助于减少总体训练时间。

接下来，您将看到如何使用一种称为*超带搜索*的策略执行和优化超参数搜索。超带搜索利用训练过程中的逐渐减少原则。在每次循环中，算法会对所有超参数组合的模型表现进行排名，并丢弃参数组合中较差的一半。下一轮中，较好的组合将获得更多的处理器核心和内存。这将一直持续到最后一种组合保留下来，消除所有但最佳超参数组合。

这有点像体育比赛中的[淘汰赛](https://oreil.ly/jq8Lb)：每一轮和每一场比赛都会淘汰排名较低的球队。然而，在超带搜索中，失败的球队在比赛完成之前就被宣布失败了。这个过程会一直持续到冠军赛，冠军赛中排名第一的球队最终成为冠军。这种策略比蛮力方法要节约得多，因为每种组合都会被训练到完整的 epochs，这会消耗大量的训练资源和时间。

让我们将您在上一章中使用的 CIFAR-10 图像分类数据集中学到的知识应用起来。

## 导入库和加载数据

我建议使用 Google Colab 笔记本电脑和 GPU 来运行此示例中的代码。截至 TensorFlow 2.4.1，Keras Tuner 尚未成为 TensorFlow 分发的一部分，因此在 Google Colab 环境中，您需要运行`pip install`命令：

```py
pip install -q -U keras-tuner
```

安装完成后，将其与所有其他库一起导入：

```py
import tensorflow as tf
import kerastuner as kt
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pylab as plt
import os
from datetime import datetime
print(tf.__version__)
```

它将显示当前版本的 TensorFlow——在本例中为 2.4.1。然后在一个单元格中加载和归一化图像：

```py
(train_images, train_labels), (test_images, test_labels) = 
datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, 
test_images / 255.0
```

现在提供一个标签列表：

```py
# Plain-text name in alphabetical order. 
https://www.cs.toronto.edu/~kriz/cifar.html
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat',
               'deer','dog', 'frog', 'horse', 'ship', 'truck']
```

接下来，通过将图像和标签合并为张量将图像转换为数据集。将测试数据集分为两组——前 500 张图像（用于训练期间的验证）和其他所有图像（用于测试）：

```py
validation_dataset = tf.data.Dataset.from_tensor_slices(
 (test_images[:500], test_labels[:500]))

test_dataset = tf.data.Dataset.from_tensor_slices(
 (test_images[500:], test_labels[500:]))

train_dataset = tf.data.Dataset.from_tensor_slices(
 (train_images, train_labels))
```

要确定每个数据集的样本大小，请执行以下命令：

```py
train_dataset_size = len(list(train_dataset.as_numpy_iterator()))
print('Training data sample size: ', train_dataset_size)

validation_dataset_size = len(list(validation_dataset.
as_numpy_iterator()))
print('Validation data sample size: ', validation_dataset_size)

test_dataset_size = len(list(test_dataset.as_numpy_iterator()))
print('Test data sample size: ', test_dataset_size)
```

您应该会得到类似以下输出：

```py
Training data sample size:  50000
Validation data sample size:  500
Test data sample size:  9500
```

接下来，为了利用分布式训练，定义一个`MirroredStrategy`对象来处理分布式训练：

```py
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(
strategy.num_replicas_in_sync))
```

在您的 Colab 笔记本中，您应该看到以下输出：

```py
Number of devices: 1
```

现在设置样本批处理参数：

```py
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
STEPS_PER_EPOCH = train_dataset_size // BATCH_SIZE_PER_REPLICA
VALIDATION_STEPS = 1
```

对所有数据集进行洗牌和批处理：

```py
train_dataset = train_dataset.repeat().shuffle(BUFFER_SIZE).batch(
BATCH_SIZE)
validation_dataset = validation_dataset.shuffle(BUFFER_SIZE).batch(
validation_dataset_size)
test_dataset = test_dataset.batch(test_dataset_size)
```

现在您可以创建一个函数来包装模型架构：

```py
def build_model(hp):
  model = tf.keras.Sequential()
  # Node count for next layer as hyperparameter
  hp_node_count = hp.Int('units', min_value=16, max_value=32, 
      step=8)
  model.add(tf.keras.layers.Conv2D(filters = hp_node_count,
      kernel_size=(3, 3),
      activation='relu',
      name = 'conv_1',
      kernel_initializer='glorot_uniform',
      padding='same', input_shape = (32,32,3)))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Flatten(name = 'flat_1'))
  # Activation function for next layer as hyperparameter
  hp_AF = hp.Choice('dense_activation', 
      values = ['relu', 'tanh'])
  model.add(tf.keras.layers.Dense(256, activation=hp_AF,
      kernel_initializer='glorot_uniform',
      name = 'dense_1'))
  model.add(tf.keras.layers.Dense(10, 
      activation='softmax',
      name = 'custom_class'))

  model.build([None, 32, 32, 3])
  # Compile model with optimizer
  # Learning rate as hyperparameter
  hp_LR = hp.Float('learning_rate', 1e-2, 1e-4)

  model.compile(
     loss=tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True),
       optimizer=tf.keras.optimizers.Adam(
          learning_rate=hp_LR),
       metrics=['accuracy'])

  return model
```

这个函数与您在第九章中看到的函数之间有一些主要区别。该函数现在期望一个输入对象`hp`。这意味着该函数将由名为`hp`的超参数调整对象调用。

在模型架构中，第一层`conv_1`的节点计数通过使用`hp_node_count`进行超参数搜索。层`dense_1`的激活函数也通过使用`hp_AF`进行超参数搜索进行声明。最后，`optimizer`中的学习率通过`hp_LR`进行超参数搜索进行声明。此函数返回具有声明的超参数的模型。

接下来，使用`kt.Hyperband`定义一个对象（`tuner`），该对象将`build_model`函数作为输入：

```py
tuner = kt.Hyperband(hypermodel = build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='hp_dir',
                     project_name='hp_kt')
```

您传递以下输入以定义`tuner`对象：

`超模型`

定义模型架构和优化器的函数。

`目标`

用于评估模型性能的训练指标。

`max_epochs`

模型训练的最大时期数。

`因子`

每个框架中的时期和模型数量的减少因子。排名在前 1/因子的模型被选中并进入下一轮训练。如果因子是 2，那么前一半将进入下一轮。如果因子是 4，那么前一四分之一将进入下一轮。

`目录`

将结果写入的目标目录，例如每个模型的检查点。

`project_name`

在目标目录中保存的所有文件的前缀。

在这里，您可以定义一个提前停止，如果验证准确性在五个时期内没有改善，则停止训练：

```py
early_stop = tf.keras.callbacks.EarlyStopping(
 monitor='val_accuracy', 
 patience=5)
```

现在您已准备好通过 Hyperband 算法启动搜索。当搜索完成时，它将打印出最佳超参数：

```py
tuner.search(train_dataset,
             steps_per_epoch = STEPS_PER_EPOCH,
             validation_data = validation_dataset,
             validation_steps = VALIDATION_STEPS,
             epochs = 15,
             callbacks = [early_stop]
             )
# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units 
in conv_1 layer is {best_hps.get('units')} and the optimal 
learning rate for the optimizer is {best_hps.get('learning_rate')} 
and the optimal activation for dense_1 layer
is {best_hps.get('dense_activation')}.
""")
```

正如您所看到的，搜索后，`best_hps`保存了有关最佳超参数值的所有信息。

当您在带有 GPU 的 Colab 笔记本中运行此示例时，通常需要大约 10 分钟才能完成。期望看到类似以下的输出：

```py
Trial 42 Complete [00h 01m 14s]
val_accuracy: 0.593999981880188

Best val_accuracy So Far: 0.6579999923706055
Total elapsed time: 00h 28m 53s
INFO:tensorflow:Oracle triggered exit

The hyperparameter search is complete. The optimal number of units 
in conv_1 layer is 24 and the optimal learning rate for the 
optimizer is 0.0013005004751682134 and the optimal activation 
for dense_1 layer is relu.
```

此输出告诉我们最佳超参数如下：

+   `conv_1`层的最佳节点计数为 24。

+   `optimizer`的最佳学习率为 0.0013005004751682134。

+   `dense_1`的最佳激活函数选择为“relu”。

现在您已经获得了最佳超参数，需要使用这些值正式训练模型。Keras Tuner 具有一个称为`hypermodel.build`的高级函数，使这成为一个单一命令过程：

```py
best_hp_model = tuner.hypermodel.build(best_hps)
```

之后，设置检查点目录，方式与您在第九章中所做的相同：

```py
MODEL_NAME = 'myCIFAR10-{}'.format(datetime.datetime.now().
strftime("%Y%m%d-%H%M%S"))
print(MODEL_NAME)
checkpoint_dir = './' + MODEL_NAME
checkpoint_prefix = os.path.join(
checkpoint_dir, "ckpt-{epoch}")
print(checkpoint_prefix)
```

您还将设置检查点，方式与您在第九章中所做的相同：

```py
myCheckPoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    monitor='val_accuracy',
    mode='max',
    save_weights_only = True,
    save_best_only = True
    )
```

现在是时候使用`best_hp_model`对象启动模型训练过程：

```py
best_hp_model.fit(train_dataset,
             steps_per_epoch = STEPS_PER_EPOCH,
             validation_data = validation_dataset,
             validation_steps = VALIDATION_STEPS,
             epochs = 15,
             callbacks = [early_stop, myCheckPoint])
```

训练完成后，加载具有最高验证准确性的模型。将`save_best_only`设置为 True，最佳模型将是最新检查点中的模型：

```py
best_hp_model.load_weights(tf.train.latest_checkpoint(
checkpoint_dir))
```

现在`best_hp_model`已准备好用于服务。它是使用最佳超参数训练的，并且权重和偏差从产生最高验证准确性的最佳训练时期加载。

# 总结

在这一章中，您学会了如何改进模型构建和质量保证流程。

机器学习模型中最常见和重要的质量保证问题之一是公平性。公平性指标是一个工具，可以帮助你调查模型在许多不同特征交互和组合中的偏见。在评估模型公平性时，你必须查找训练数据中的模型偏见。你还需要依赖主题专家的背景知识，以便在调查任何模型偏见时制定假设。在《泰坦尼克号》的例子中，这个过程相当直接，因为很明显性别和阶级在决定每个人生存机会中起着重要作用。然而，在实践中，还有许多其他因素使事情复杂化，包括数据是如何收集的，以及数据收集的背景或条件是否偏向于样本来源中的某一组。

在模型构建过程中，超参数调整是耗时的。过去，你必须迭代每个潜在超参数值的组合，以搜索最佳组合。然而，使用 Keras Tuner 库，一个相对先进的搜索算法称为 Hyperband 可以高效地进行搜索，使用一种类似锦标赛的框架。在这个框架中，基于弱超参数训练的模型会在训练周期完成之前被终止和移除。这减少了总体搜索时间，最佳超参数会脱颖而出。你只需要用获胜组合正式训练相同的模型。

有了这些知识，你现在可以将你的 TensorFlow 模型开发和测试技能提升到下一个水平。

如果你需要对分类指标进行复习，我建议你查看谷歌的《机器学习速成课程》中包含的简洁而有用的复习部分。
