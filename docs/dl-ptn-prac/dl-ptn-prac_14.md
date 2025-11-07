# 11 迁移学习

本章涵盖

+   使用 TF.Keras 和 TensorFlow Hub 中的预构建和预训练模型

+   在类似和不同领域之间执行任务迁移学习

+   使用特定领域权重初始化迁移学习模型

+   确定何时重用高维或低维潜在空间

TensorFlow 和 TF.Keras 支持广泛的预构建和预训练模型。*预训练*模型可以直接使用，而*预构建*模型则可以从零开始训练。通过替换任务组，预训练模型也可以重新配置以执行任何数量的任务。用重新训练替换或重新配置任务组的过程称为*迁移学习*。

从本质上讲，迁移学习意味着将解决一个任务的知识迁移到解决另一个任务。与从头开始训练模型相比，迁移学习的优势是新的任务可以更快地训练，并且需要的数据更少。把它看作是一种重用：我们正在重用带有其学习权重的模型。

你可能会问，我能否将一个模型架构学习到的权重重用于另一个模型？不，两个模型必须是相同的架构，例如 ResNet50 到 ResNet50。另一个常见的问题是：我能否将学习到的权重重用于*任何*不同的任务？你可以，但结果将取决于预训练模型的领域和新数据集之间的相似程度。所以我们真正所说的学习权重是指学习到的基本特征、相应的特征提取和潜在空间表示——表示学习。

让我们看看几个例子，看看迁移学习是否会产生期望的结果。假设我们有一个针对水果种类和品种的预训练模型，我们还有一个针对蔬菜种类和品种的新数据集。高度可能的是，水果的学习表示可以用于蔬菜，我们只需要训练任务组。但如果我们的新数据集包括卡车和面包车的型号和制造商。在这种情况下，数据集领域之间的差异非常大，水果学习到的表示不太可能用于卡车和面包车。在类似领域的情况下，我们希望新模型执行的任务在领域上与原始模型训练的数据相似。

另一种学习表示的方法是使用在大量不同图像类别上训练的模型。许多 AI 公司提供这种类型的迁移学习服务。通常，他们的预训练模型是在数万个图像类别上训练的。这里的假设是，由于这种广泛的多样性，学习到的表示中的一部分可以在任何任意新的数据集上重用。缺点是，为了覆盖如此广泛的多样性，潜在空间必须非常大——因此你最终得到的是一个在任务组中非常大的模型（过参数化）。

第三种方法是在参数高效、窄域训练模型和大规模训练模型之间找到一个合适的平衡点。例如，ResNet50 和更近期的 EffcientNet-B7 都是使用包含 1000 个不同类别图像的 ImageNet 数据集进行预训练的。DIY 迁移学习项目通常使用这些模型。例如，ResNet50 具有合理高效的潜在空间，但足够大，可以在任务组件之前用于迁移学习到各种图像分类数据集；潜在空间由 2048 个 4×4 特征图组成。

让我们总结这三种方法：

+   相似领域迁移：

    +   参数高效、窄域预训练模型

    +   重新训练新的任务组件

+   不同领域迁移：

    +   参数过剩、窄域预训练模型

    +   使用其他组件的微调重新训练新的任务组件

+   通用迁移

    +   参数过剩、通用领域预训练模型

    +   重新训练新的任务组件

预训练模型也可以在迁移学习中重复使用，以从预训练模型学习不同类型的任务。例如，假设我们有一个预训练模型，它可以从房屋前外部的图片中分类建筑风格。现在假设我们想要学习预测房屋的售价。很可能，基本特征、特征提取和潜在空间会转移到不同类型的任务上，例如回归器——一个输出单个实数的模型（例如，房屋的售价）。如果其他任务类型也可以使用原始数据集进行训练，那么这种将迁移学习应用于其他任务类型通常是可能的。

本章介绍了从公共资源中获取预构建和预训练的 SOTA 模型：TF.Keras 和 TensorFlow Hub。然后我将向您展示如何直接使用这些模型。最后，您将学习各种使用预训练模型进行迁移学习的方法。

## 11.1 TF.Keras 预构建模型

TF.Keras 框架附带预构建模型，您可以使用它们直接训练新模型，或者修改和/或微调以进行迁移学习。这些模型基于图像分类的最佳模型，在 ImageNet 等竞赛中获奖的模型，这些模型在深度学习研究论文中被频繁引用。

预构建 Keras 模型的文档可以在 Keras 网站上找到（[`keras.io/api/applications/`](https://keras.io/api/applications/)）。表 11.1 列出了 Keras 预构建模型架构。

表 11.1 Keras 预构建模型

| 模型类型 | SOTA 模型架构 |
| --- | --- |
| 顺序 CNN | VGG16, VGG19 |
| 残差 CNN | ResNet, ResNet v2 |
| 宽残差 CNN | ResNeXt, Inception v3, InceptionResNet v2 |
| 交替连接的 CNN | DenseNet, Xception, NASNet |
| 移动 CNN | MobileNet, MobileNet v2 |

预构建的 Keras 模型是从`keras.applications`模块导入的。以下是可以导入的预构建 SOTA 模型的示例。例如，如果您想使用 VGG16，只需将 VGG19 替换为 VGG16 即可。一些模型架构可以选择不同数量的层，例如 VGG、ResNet、ResNeXt 和 DenseNet。

```
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import MobileNet
```

### 11.1.1 基础模型

默认情况下，TF.Keras 预构建模型是完整的但未训练的，这意味着权重和偏差是随机初始化的。每个未训练的预构建 CNN 模型都针对特定的输入形状（见文档）和输出类数量进行配置。在大多数情况下，输入形状是(224, 224, 3)或(299, 299, 3)。模型还将以通道优先的格式接收输入，例如(3, 224, 224)和(3, 299, 299)。输出类数量通常是 1000，这意味着模型可以识别 1000 个常见的图像标签。这些预构建但未训练的模型本身可能对您不太有用，因为您必须在一个具有相同数量标签（1000）的数据集上完全训练它们。了解这些预构建模型的内容很重要，这样您就可以使用预训练的权重、新的任务组件或两者结合来重新配置。我们将在本章中涵盖所有三种后续的重新配置。

图 11.1 展示了预构建 CNN 模型的架构。该架构包括为输入形状预设的茎卷积组、一个用于更多卷积组（学习者）的预设、瓶颈层以及预设为 1000 个类别的分类器层。

![图片](img/CH11_F01_Ferlitsch.png)

列表 11.1 以深灰色显示任务组层的预构建 CNN 模型架构

预构建模型没有分配损失函数和优化器。在使用它们之前，我们必须发出`compile()`方法来分配损失、优化器和性能度量。在下面的代码示例中，我们首先导入并实例化一个 ResNet50 预构建模型，然后编译模型：

```
from tensorflow.keras.applications import ResNet50

model = ResNet50()                                               ❶

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])                              ❷
```

❶ 获取一个完整且未训练的预构建 ResNet50 模型

❷ 将模型编译为用于数据集的分类器

以这种方式使用预构建模型相当有限，不仅因为输入大小固定，而且分类器的类别数量也是固定的，即 1000。您需要完成的任何任务很可能不会使用默认配置。接下来，我们将探讨配置预构建模型以执行各种任务的方法。

### 11.1.2 用于预测的预训练 ImageNet 模型

所有预构建的模型都附带从*ImageNet 2012*数据集预训练的权重和偏差，该数据集包含 1000 个类别中的 120 万张图像。如果你的需求仅仅是预测图像是否在 ImageNet 数据集的 1000 个类别中，你可以直接使用预训练的预构建模型。标签标识符到类名的映射可以在 GitHub 上找到（[`gist.github.com/yrevar/942d3a0ac09ec9e5eb3a`](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)）。类别的例子包括秃鹰、卫生纸、草莓和气球等。

让我们使用预训练的 ResNet 模型，该模型使用 ImageNet 权重进行预训练，来对大象的图像进行分类（或预测）。以下是步骤，一步一步来：

1.  `preprocess_input()`方法将根据预构建的 ResNet 模型使用的方法对图像进行预处理。

1.  `decode_predictions()`方法将标签标识符映射回类名。

1.  使用 ImageNet 权重实例化预构建的 ResNet 模型。

1.  使用 OpenCV 读取大象的图像，并将其调整大小为（224, 224）以适应模型的输入形状。

1.  然后使用模型的`preprocessed_input()`方法对图像进行预处理。

1.  然后将图像重塑为一批。

1.  然后使用`predict()`方法通过模型对图像进行分类。

1.  然后使用`decode_predictions()`将前三个预测标签映射到其类名，并打印出来。在这个例子中，我们可能会看到非洲象作为最高预测。

图 11.2 展示了 TF.Keras 预训练模型及其伴随的预处理输入和后处理输出函数。

![图片](img/CH11_F02_Ferlitsch.png)

列表 11.2 TF.Keras 预训练模型及其伴随的预处理输入和后处理输出特定函数

现在我们来看看如何编写这个过程：

```
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input, 
                                                 decode_predictions

model = ResNet50(weights='imagenet')                        ❶

image = cv2.imread('elephant.jpg', cv2.IMREAD_COLOR)        ❷

image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR)     ❸

image = preprocess_input(image)                             ❹

image = image.reshape((-1, 224, 224, 3))                    ❺

predictions = model.predict(image)                          ❻

print(decode_predictions(predictions, top=3))               ❼
```

❶ 获取在 ImageNet 上预训练的 ResNet50 模型

❷ 读取图像并将其作为 NumPy 数组预测到内存中

❸ 将图像调整大小以适应预训练模型的输入形状

❹ 使用与预训练模型相同的图像处理方法对图像进行预处理

❺ 将单个图像形状（224, 224, 3）重塑为单个图像的一批（1, 224, 224, 3）以供 predict()方法使用

❻ 调用 predict()方法对图像进行分类

❽ 使用预训练模型的解码函数根据预测标签显示类名

### 11.1.3 新的分类器

在所有预构建模型中，可以移除最终的分类器层并替换为新的分类器——以及另一个任务，如回归器。然后，可以使用新的分类器来训练预构建模型以适应新的数据集和类别集。例如，如果你有一个包含 20 种面条菜肴的数据集，你只需移除现有的分类器层，用新的 20 节点分类器层替换它，编译模型，并用面条菜肴数据集进行训练。

在所有预构建的模型中，分类器层被称为*顶层*。对于 TF.Keras 预构建模型，输入形状默认为(224, 224, 3)，输出层的类别数为 1000。当你实例化一个 TF.Keras 预构建模型时，你会设置参数`include_top`为`False`以获取一个不带分类器层的模型实例。另外，当`include_top=False`时，我们可以使用参数`input_shape`指定模型的不同输入形状。

现在我们来描述这个流程及其在我们 20 种面条菜品分类器中的应用。假设你拥有一家面条餐厅，厨师们不断地将各种新鲜烹制的面条菜品放在点餐柜台上。顾客可以挑选任何菜品，为了简化起见，让我们假设所有面条菜品的价格相同。收银员只需要计算面条菜品的数量。但你仍然有一些问题需要解决。有时你的厨师准备过多的一种或多种菜品，这些菜品变凉后不得不丢弃，因此你损失了收入。其他时候，你的厨师准备得太少的一种或多种菜品，顾客因为他们的菜品不可用而去了另一家餐厅——这是一个机会损失的情况。

为了解决这两个问题，你计划在结账处放置一个摄像头，并在丢弃冷面条菜品的烹饪区域放置另一个摄像头。你希望摄像头能够实时分类购买的面条菜品和丢弃的菜品，并将这些信息显示给厨师，以便他们更好地估计需要准备哪些菜品。

让我们开始实施你的计划。首先，因为你是一家现有的面条餐厅，你雇佣了一个人来拍摄放在点餐柜台上的菜品照片。当拍照时，厨师会喊出菜品的名字，这个名字会与照片一起记录。假设在一天的业务结束时，你的面条菜品数量为 500 种。假设菜品的分布相当均匀，这将给你平均每种面条菜品 25 张照片。这可能看起来每个类别的数量很少，但既然它们是你的菜品，背景总是相同的，这可能是足够的。现在你只需要从音频录音中标记照片。

现在你已经准备好进行训练了。你从 TF.Keras 获取一个预构建的模型，并指定`include_top=False`以删除 1000 类分类器的密集层——你将随后用 20 节点的密集层替换它。因为你移动很多面条菜品，所以你希望模型预测速度快，因此你想要减少参数数量，同时不影响模型的准确性。你不再从(224, 224, 3)大小的 ImageNet 进行预测，而是指定`input_shape=(100, 100, 3)`以改变模型的输入向量大小为(100, 100, 3)。

我们也可以在预构建模型中删除最终的展平/池化层（瓶颈层），通过设置参数 `pooling=None` 来替换成你自己的。

图 11.3 描述了一个可重构的预构建 CNN 模型架构。它由一个可配置输入大小的茎卷积组、一个或多个卷积组（学习器）以及可选的可配置瓶颈层组成。

![](img/CH11_F03_Ferlitsch.png)

列表 11.3 在这个没有分类器层的可重构预构建模型架构中，保留池化层是可选的。

至于输入形状，预构建模型的文档对最小输入形状大小有限制。对于大多数模型，这是 (32, 32, 3)。我通常不建议以这种方式使用预构建模型，因为对于这些架构中的大多数，全局平均池化层（瓶颈层）之前的最终特征图将是 1 × 1（单像素）特征图——本质上丢失了所有空间关系。然而，研究人员发现，当与 CIFAR-10 和 CIFAR-100（32, 32, 3）图像一起使用时，他们能够在进入竞赛级（如 ImageNet）图像数据集（224, 224, 3）之前找到良好的超参数设置。

在下面的代码中，我们实例化了一个预构建的 ResNet50 模型，并用一个新的分类器替换了它，用于我们的 20 种面条菜肴示例：

1.  我们使用参数 `include_top=False` 移除了现有的 1000 个节点的分类器。

1.  我们使用参数 `input_shape` 将输入形状设置为 (100, 100, 3)，以适应较小的输入尺寸。

1.  我们决定保留最终的池化/展平层（瓶颈层），将其作为全局平均池化层，参数为 `pooling`。

1.  我们添加了一个替换的密集层，包含 20 个节点，对应于面条菜肴的数量，以及一个 softmax 激活函数作为顶层。

    +   预构建 ResNet50 模型的最后一个（输出）层是 `model.output`。这对应于瓶颈层，因为我们删除了默认的分类器。

    +   我们将预构建 ResNet50 的 `model.output` 绑定为替换密集层的输入。

1.  我们构建了模型。输入是 ResNet 模型的输入，即 `models.input`。

1.  最后，我们编译模型以进行训练，并将损失函数设置为 `categorical_crossentropy`，优化器设置为 `adam`，这是图像分类模型的最佳实践。

```
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense

model = ResNet50(include_top=False, input_shape=(100, 100, 3), pooling='avg') ❶

outputs = Dense(20, activation='softmax')(model.output)                       ❷
model = Model(model.input, outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])                                           ❸
```

❶ 获取输入形状为 (100,100,3) 且没有最终分类器的预构建模型

❷ 添加了 20 个类别的分类器

❸ 编译模型以进行训练

对于大多数 TF.Keras 预构建模型，瓶颈层是一个全局平均池化层。这个层既作为特征图的最终池化层，又作为一个展平操作，将特征图转换为 1D 向量。在某些情况下，我们可能想用我们自己的自定义最终池化/展平层替换这个层。在这种情况下，我们要么指定参数 `pooling=None`，要么不指定它，这是默认设置。那么我们为什么要这样做呢？

为了回答这个问题，让我们回到我们的面条菜肴。假设当您训练模型时，您得到了 92%的准确率，并希望做得更好。首先，您决定添加图像增强。嗯，我们可能不会考虑水平翻转，因为面条菜肴永远不会被倒着看到！同样，垂直翻转可能也不会有帮助，因为面条碗相当均匀（没有镜像）。我们可以跳过旋转，因为面条碗相当均匀，我们跳过缩放，因为相机到菜肴的位置是固定的。嗯，所以您问，还有什么？

关于移动碗的位置怎么样，因为碗在结账和扔掉柜台时都会移动？您这样做并得到了 94%的准确率。但您希望更高的准确率。凭直觉，我们推测可能特征信息保留得不够，当每个最终特征图通过默认的 `GlobalAveragePooling2D` 池化减少到一个像素，然后展平成一个 1D 向量时。您查看您的模型摘要，看到最终特征图的大小是 4 × 4。因此，您决定取消默认池化，并用步长为 2 的 `MaxPooling2D` 替换它，这样每个特征图将减少到 2 × 2，4 个像素而不是一个像素，然后进行展平成一个 1D 向量。

在这个代码示例中，我们用最大池化 (`outputs = MaxPooling2D(model.outputs)`) 和展平 (`outputs = Flatten(outputs)`) 替换了瓶颈层，用于我们的 20 种面条菜肴分类器：

```
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

model = ResNet50(include_top=False, input_shape=(100, 100, 3), pooling=None) ❶

outputs = MaxPooling2D(model.output)                                         ❷
outputs = Flatten()(ouputs)                                                  ❷

outputs = Dense(20, activation='softmax')(outputs)                           ❸

model = Model(model.input, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
```

❶ 获取输入形状为 (100,100,3) 且不带分类器组的预建模型

❷ 将特征图池化并展平成一个 1D 向量

❸ 添加了一个 20 类别的分类器

在本节中，我们介绍了 TF.Keras 的预建模型和预训练模型。总结一下，预建模型是一个现有的模型，通常基于 SOTA 架构，其输入形状和任务组是可重新配置的，且权重未经过训练。预建模型通常用于从头开始训练模型，具有可重用性和可重新配置以适应您的数据集和任务的优势。缺点是架构可能没有针对您的数据集/任务进行调整，因此最终得到的模型在尺寸和准确性方面可能都不够高效。

预训练模型本质上与预建模型相同，只是权重已经使用另一个数据集（如 ImageNet 数据集）进行了预训练。预训练模型用于即插即用预测或迁移学习，具有通过代表性学习重用快速训练新数据集/任务并减少数据量的优势。缺点是预训练的代表性学习可能不适合您的数据集/任务领域。

在下一节中，我们将使用来自 TensorFlow Hub 存储库的预建模型介绍相同的概念。

## 11.2 TF Hub 预建模型

*TensorFlow Hub*，或*TF Hub*，是一个开源公共仓库，包含预构建和预训练模型，比 TF.Keras 更为广泛。TF.Keras 的预构建/预训练模型适合学习和练习迁移学习，但在生产目的上提供的选项过于有限。TF Hub 包含大量预构建的 SOTA 架构、广泛的任务类别、特定领域的预训练权重以及超出 TensorFlow 组织直接提供的模型之外的公共提交。 

本节涵盖了图像分类的预构建模型。TF Hub 为每个模型提供两个版本，具体描述如下：

+   用于特定类别的图像分类的模块。这个过程与预训练模型相同。

+   用于提取图像特征向量（瓶颈值）的模块，用于在自定义图像分类器中使用。这些分类器与 TF.Keras 中描述的新分类器相同。

我们将使用两个预构建模型，一个用于开箱即用的分类，另一个用于迁移学习。我们将从 TensorFlow Hub 的预构建模型开源仓库中下载这些模型，该仓库位于[www.tensorflow.org/hub](https://www.tensorflow.org/hub)。

要使用 TF Hub，您首先需要安装`tensorflow_hub` Python 模块：

```
pip install tensorflow_hub
```

在您的 Python 脚本中，通过导入`tensorflow_hub`模块来访问 TF Hub：

```
import tensorflow_hub as hub
```

您现在已设置好下载我们两个模型。

### 11.2.1 使用 TF Hub 预训练模型

与 TF.Keras 相比，TF Hub 在可加载的模型格式类型方面非常灵活：

+   *TF2.x SavedModel*—在本地、REST 或云上的微服务、桌面/笔记本电脑或工作站中使用。

+   *TF Lite*—在移动或内存受限的 IoT 设备上的应用程序服务中使用。

+   *TF.js*—在客户端浏览器应用程序中使用。

+   *Coral*—优化用于在 Coral Edge/IoT 设备上作为应用程序服务使用。

本节将仅涵盖 TF 2.x 的 SavedFormat 模型。要加载一个模型，您需要执行以下操作：

1.  获取 TF Hub 仓库中图像分类器模型的 URL。

1.  使用`hub.KerasLayer()`从指定的 URL 指定的仓库中检索模型数据。

1.  通过使用 TF.Keras sequential API 从模型数据构建一个 TF.Keras SavedModel。

1.  将输入形状指定为(224, 224, 3)，这与预训练模型在 ImageNet 数据库上训练的输入形状相匹配。

```
model_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4" ❶

model = tf.keras.Sequential([hub.KerasLayer(model_url,
                             input_shape=(224,224,3))])                       ❷
```

❶ TF Hub 仓库中 ResNet50 v2 模型数据的存储位置

❷ 从模型数据检索并构建 SavedModel 格式的模型

当您执行`model.summary()`时，输出将如下所示：

```
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer_7 (KerasLayer)   (None, 1001)              25615849  
=================================================================
Total params: 25,615,849
Trainable params: 0
Non-trainable params: 25,615,849
```

现在，您可以使用该模型进行预测，这被称为*推理*。图 11.4 描述了使用 TF Hub ImageNet 预训练模型进行预测的以下步骤：

1.  获取 ImageNet 的标签（类别名称）信息，以便我们将预测的标签（数字索引）转换为类别名称。

1.  预处理图像以预测以下内容：

    +   将图像输入调整大小以匹配模型的输入：(224, 224, 3)。

    +   标准化图像数据：除以 255。

1.  对图像调用 `predict()`。

1.  使用 `np.argmax()` 返回最高概率的标签索引。

1.  将预测的标签索引转换为相应的类名。

![](img/CH11_F04_Ferlitsch.png)

列表 11.4 使用 TF Hub 的 ImageNet 预训练模型预测标签，然后使用 ImageNet 映射显示预测的类名

这里是这些五个步骤的一个示例实现。

```
path = tf.keras.utils.get_file('ImageNetLabels.txt',
'https://storage.googleapis.com/download.tensorflow.org/data/
ImageNetLabels.txt')           
imagenet_labels = np.array(open(path).read().splitlines())    ❶

import cv2
import numpy as np
data = cv2.imread('apple.png')                                ❷
data = cv2.resize(data, (224, 224))                           ❷
data = (data / 255.0).astype(np.float32)                      ❷

p = model.predict(np.asarray([data]))                         ❸
y = np.argmax(p)                                              ❸

print(imagenet_labels[y])                                     ❹
```

❶ 获取从 ImageNet 标签索引到类名的转换

❷ 预处理图像以进行预测

❸ 使用模型进行预测

❹ 将预测的标签索引转换为类名

### 11.2.2 新的分类器

对于为预训练模型构建新的分类器，我们加载相应的模型 URL，表示为模型的*特征向量*版本。这个版本加载了预训练模型，但没有模型顶部或分类器。这允许你添加自己的顶部或任务组。模型的输出是输出层。我们还可以指定一个与 TF Hub 模型默认输入形状不同的新输入形状。

以下是一个加载预训练 ResNet50 v2 模型特征向量版本的示例实现，我们将添加自己的任务组件以训练 CIFAR-10 模型。由于我们的 CIFAR-10 输入大小与 TF Hub 的 ResNet50 v2 版本不同，其大小为(224, 224, 3)，因此我们还可以选择指定输入形状：

1.  获取 TF Hub 存储库中图像分类器模型的 URL。

1.  使用 `hub.KerasLayer()` 从由 URL 指定的存储库中检索模型数据。

1.  为 CIFAR-10 数据集指定新的输入形状为(32, 32, 3)。

```
f_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"  ❶

f_layer = hub.KerasLayer(f_url, input_shape=(32,32,3))                     ❷
```

❶ TF Hub 存储库中 ResNet50 v2 特征向量版本模型数据的存储位置

❷ 将模型数据作为 TF.Keras 层检索并设置输入形状

这里是构建 CIFAR-10 新分类器的一个示例实现，格式为 SavedModel：

1.  使用顺序 API 创建 SavedModel。

    +   将预训练的 ResNet v2 的特征向量版本指定为模型底部。

    +   指定一个有 10 个节点（每个 CIFAR-10 类别一个）的密集层作为模型顶部。

1.  编译模型。

```
model = tf.keras.Sequential([
                             f_layer,
                             Dense(10, activation='softmax')
                            ])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
```

当你执行 `model.summary()` 时，输出将如下所示：

```
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer_4 (KerasLayer)   (None, 2048)              23561152  
_________________________________________________________________
dense_2 (Dense)              (None, 10)                20490     
=================================================================
Total params: 23,581,642
Trainable params: 20,490
Non-trainable params: 23,561,152
```

到目前为止，我们已经涵盖了使用预训练模型进行即插即用预测和使用可重新配置的预构建模型进行更方便的新模型训练。接下来，我们将介绍如何使用和重新配置预训练模型以实现更高效的训练并减少新任务所需的数据。

## 11.3 领域间的迁移学习

在迁移学习中，我们使用预训练模型完成一个任务，并重新训练分类器和/或微调层以完成新任务。这个过程与我们刚刚在预构建模型上构建新分类器类似，但除此之外，模型是从头开始完全训练的。

迁移学习有两种一般方法：

+   *相似任务*——预训练数据集和新数据集来自相似的域（例如水果到蔬菜）。

+   *不同任务*——预训练数据集和新数据集来自不同的域（例如水果和卡车/面包车）。

### 11.3.1 相似任务

如本章前面所讨论的，在决定方法时，我们查看源（预训练）图像域和目标（新）域的相似性。越相似，我们可以重用更多现有底层而无需重新训练。例如，如果我们有一个在水果上训练的模型，那么预训练模型的底层所有层很可能可以重用而无需重新训练来构建一个用于识别蔬菜的新模型。

我们假设在底层学习到的粗略和详细特征对于新分类器将是相同的，并且可以在进入最顶层（的）分类之前直接重用。让我们考虑一些我们可以推测水果和蔬菜来自非常相似域的原因。两者都是天然食品。虽然水果通常在地面上生长，而蔬菜在地下生长，但它们在形状和质地上有相似的物理特性，以及如茎和叶等装饰。

当源域和目标域具有这种高水平相似性时，我们通常可以用新的分类器层替换现有的最顶层分类器层，冻结底层层，并仅训练分类器层。由于我们不需要学习其他层的权重/偏差，因此我们可以用大量更少的数据和更少的周期来训练新域的模型。

虽然拥有更多数据总是更好的，但相似源域和目标域之间的迁移学习提供了使用大量更小数据集进行训练的能力。关于数据集最小尺寸的两个最佳实践如下：

+   每个类别（标签）的大小是源数据集的 10%。

+   每个类别（标签）至少有 100 张图片。

与*新分类器*所示的方法相反，我们在训练之前修改代码以冻结所有位于最顶层分类器层之前的层。冻结可以防止这些层（的）权重/偏差在分类器（最顶层）层的训练期间被更新（重新训练）。在 TF.Keras 中，每个层都有`trainable`属性，默认为`True`。

图 11.5 描述了预训练模型分类器层的重新训练；以下是步骤：

1.  使用具有预训练权重/偏差的预构建模型（ImageNet 2012），

1.  从预构建模型中删除现有的分类器（最顶层）。

1.  冻结剩余的层。

1.  添加一个新的分类器层。

1.  通过迁移学习训练模型。

![图片](img/CH11_F05_Ferlitsch.png)

列表 11.5 当源域和目标域相似时，只有分类器权重被重新训练，而剩余模型底层的权重被冻结。

这里是一个示例实现：

```
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

model = ResNet50(include_top=False, pooling='avg', weights='imagenet')   ❶

for layer in model.layers:                                               ❷
    layer.trainable = False                                              ❷

output = Dense(20, activation='softmax')(model.output)                   ❸

model = Model(model.input, output)                                       ❹
model.compile(loss='categorical_crossentropy', optimizer='adam',         ❹
              metrics=['accuracy'])                                      ❹
```

❶ 获取不带分类器的预训练模型并保留全局平均池化层

❷ 冻结剩余层的权重

❸ 添加一个 20 个类别的分类器

❹ 编译模型以进行训练

注意，在这个代码示例中，我们保留了原始输入形状（224, 224, 3）。在实际操作中，如果我们更改输入形状，现有的训练权重/偏差将不会匹配它们训练的特征提取分辨率。在这种情况下，最好将其作为一个独立任务案例处理。

### 11.3.2 独立任务

当图像数据集的源域和目标域不同，例如我们例子中的水果和卡车/面包车时，我们开始与之前相似任务方法中的相同步骤，然后继续微调底部层。步骤，如图 11.6 所示，通常如下：

1.  添加一个新的分类器层并冻结剩余的底部层。

1.  训练新的分类器层以达到目标周期数。

1.  重复进行微调：

    +   解冻下一个最底部的卷积组（从顶部到底部的方向）。

    +   训练几个周期以进行微调。

1.  在卷积组微调后：

    +   解冻卷积主干组。

    +   训练几个周期以进行微调。

在图 11.6 中，你可以看到步骤 2 到 4 的训练周期：在周期 1 中重新训练分类器，在周期 2 到 4 中按顺序微调卷积组，在周期 5 中微调主干。请注意，这与源域和目标域相似且我们只微调分类器的情况不同。

![图像](img/CH11_F06_Ferlitsch.png)

列表 11.6 在这种独特的源到目标迁移学习中，卷积组逐步微调。

以下是一个示例实现，演示了对新分类器级别（周期 1）的粗粒度训练，然后是每个卷积组（周期 2 到 4）的微调，最后是主干卷积组（周期 5）。步骤如下：

1.  模型底部的层被冻结（`layer.trainable = False`）。

1.  在模型顶部添加一个 20 个类别的分类器层。

1.  分类器层使用 50 个周期进行训练：

```
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

model = ResNet50(include_top=False, pooling='avg', weights='imagenet')

for layer in model.layers:                                                 ❶
    layer.trainable = False                                                ❶

output = Dense(20, activation='softmax')(model.output)                     ❷

model = Model(model.input, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])                                        ❸

model.fit(x_data, y_data, batch_size=32, epochs=50, validation_split=0.2)  ❹
```

❶ 冻结所有预训练层的权重

❷ 添加一个未训练的分类器

❸ 编译模型以进行训练

❹ 粗粒度训练新的分类器

在分类器训练后，模型进行微调（周期 2 到 4）：

1.  从底部到顶部遍历层，识别主干卷积和每个 ResNet 组的结束，这通过一个`Add()`层检测到。

1.  对于每个卷积组，构建该组中每个卷积层的列表。

1.  以相反的顺序构建组列表（`groups.insert(0, conv2d)`）: 从顶部到底部。

1.  从顶部到底部遍历卷积组，并逐步训练每个组和其前驱，共五个周期。

以下是对这四个步骤的示例实现。

```
stem = None
groups = []
conv2d = []

first_conv2d = True
for layer in model.layers:
        if type(layer) == layers.convolutional.Conv2D:
            if first_conv2d == True:                                ❶
                stem = layer
                first_conv2d = False
            else:                                                   ❷
                conv2d.append(layer)
        elif type(layer) == layers.merge.Add:                       ❸
                groups.insert(0, conv2d)                            ❹
                conv2d = []

for i in range(1, len(groups)):                                     ❺
        for layer in groups[i]:                                     ❺
            layer.trainable = True                                  ❺

        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics=['accuracy'])                         ❻
                                                                    ❻
        model.fit(x_data, y_data, batch_size=32, epochs=5)          ❻
```

❶ 在 ResNet50 中，第一个 Conv2D 是主干卷积层。

❷ 为每个卷积组保持卷积层的列表

❸ 残差网络中的每个卷积组都以一个 Add()层结束。

❹ 以相反的顺序维护列表（最上面的卷积组是列表的顶部）

❺ 一次解冻一个卷积组（从上到下）

❻ 微调（训练）该层

最后，主干卷积以及整个模型额外训练了五个周期（周期 5）。以下是最后一步的示例实现：

```
stem.trainable = True                                                      ❶
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
model.fit(x_data, y_data, batch_size=32, epochs=5, validation_split=0.2)   ❷
```

❶ 解冻主干卷积

❷ 进行最终微调

在此示例中，当解冻层进行微调时，必须在发出下一个训练会话之前重新编译模型。

### 11.3.3 特定领域权重

在之前的迁移学习示例中，我们使用从 ImageNet 2012 数据集学习到的权重初始化了模型的冻结层。但让我们假设你想要使用除 ImageNet 2012 之外特定领域的预训练权重，就像我们关于水果的例子一样。

例如，如果你正在构建一个植物领域的域迁移模型，你可能需要树木、灌木、花朵、杂草、叶子、树枝、水果、蔬菜和种子的图像。但我们不需要每种可能的植物类型——只需要足够的来学习基本特征和特征提取，这些可以推广到更具体和更全面的植物领域。你也可能考虑你想要推广的背景。例如，目标领域可能是室内植物，因此你有家庭室内背景，或者它可能是产品，因此你想要一个货架背景。你应该在源域中有一定数量的这些背景，这样源模型就学会了从潜在空间中过滤掉它们。

在下一个代码示例中，我们首先为特定领域（在这种情况下，是水果产品）训练一个预构建的 ResNet50 架构；然后，我们使用预训练的、特定领域的权重和初始化来训练另一个在类似领域（例如，蔬菜）中的 ResNet50 模型。

图 11.7 描述了将特定领域的权重从水果迁移到类似领域（蔬菜）并进行微调的过程如下：

1.  实例化一个未初始化的 ResNet50 模型，不带分类器和池化层，我们将其指定为基础模型。

1.  保存基础模型架构以供以后在迁移学习中重复使用（`produce-model`）。

1.  添加一个分类器（`Flatten`和`Dense`层）并针对特定的（源）领域（例如，产品）进行训练。

1.  保存训练模型的权重（`produce-weights`）。

1.  加载基础模型架构（`model-produce`），它不包含分类器层。

1.  使用源域的预训练权重初始化基础模型架构（`model-produce`）。

1.  为新类似领域添加一个分类器。

1.  训练新类似领域的模型/分类器。

![](img/CH11_F07_Ferlitsch.png)

列表 11.7：与源域类似领域的预训练模型之间的迁移学习

这里是一个将特定领域权重从水果迁移到类似领域蔬菜的迁移学习的示例实现：

```
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model

model = ResNet50(include_top=False, pooling=None, input_shape=(100, 100, 3))

model.save('produce-model')                           ❶

output = Flatten(name='bottleneck')(model.output)     ❷
output = Dense(20, activation='softmax')(output)      ❷

model.save_weights('produce-weights')                 ❸

model = load_model('produce-model')                   ❹
model.load_weights('produce-weights')                 ❹

output = Flatten(name='bottleneck')(model.output)     ❺
output = Dense(20, activation='softmax')(output)      ❺

model = Model(model.input, output)                    ❻
model.compile(loss='categorical_crossentropy', optimizer='adam',  
              metrics=['accuracy'])                   ❼
```

❶ 保存基础模型

❷ 添加分类器

❸ 保存训练好的模型权重

❹ 训练模型

❺ 重新使用基础模型和训练好的权重

❻ 添加分类器

❼ 编译并训练新数据集的新模型

### 11.3.4 领域迁移权重初始化

另一种迁移学习的形式是将特定领域权重迁移到作为我们将重新训练的模型的权重初始化。在这种情况下，我们试图改进基于随机权重分布算法（例如，对于 ReLU 激活函数的 He-normal）的初始化器，而不是使用彩票假设或数值稳定性。让我们再次看看我们的产品示例，并假设我们已经为数据集实例（如水果）完全训练了一个模型。我们不是从完全训练的模型实例中迁移权重，而是使用一个更早的检查点，其中我们已经建立了数值稳定性。我们将重用这个更早的检查点作为重新训练领域相似数据集（如蔬菜）的初始化器。

转移特定领域权重是一种一次性权重初始化方法。假设是生成一组足够泛化的权重初始化，以便模型训练将导致最佳局部（或全局）最优解。理想情况下，在初始训练期间，模型的权重将执行以下操作：

+   指向收敛的一般正确方向

+   防止过度泛化以避免陷入任意局部最优解

+   作为单次（一次性）训练会话的初始化权重，该会话将收敛到最佳局部最优解

图 11.8 描述了权重初始化的领域迁移。

![图片](img/CH11_F08_Ferlitsch.png)

列表 11.8 使用类似领域的早期检查点作为新模型完全重新训练的权重初始化

这种权重初始化的预训练步骤如下：

1.  实例化一个 ResNet50 模型，具有随机权重分布（例如，Xavier 或 He-normal）。

1.  使用高水平的正则化（l2(0.001)）以防止拟合数据和小学习率。

1.  运行几个时代（未展示）。

1.  使用模型方法 `save_weights()` 保存权重。

```
from tensorflow.keras.regularizers import l2

model = ResNet50(include_top=False, pooling='avg', input_shape=(100, 100, 3)) ❶

model.save('base_model')                                                      ❷

output = layers.Dropout(0.75)(model.output)                                   ❸
output = layers.Dense(20, activation='softmax',                               ❸
                      kernel_regularizer=l2(0.001))(output)                   ❸
model  = Model(model.input, output)                                           ❸

model.save_weights('weights-init')                                            ❹
```

❶ 使用默认权重初始化（He-normal）实例化基础模型

❷ 保存模型

❸ 在基础 ResNet 模型中添加 dropout 层和分类器，并使用激进的正则化级别

❹ 预训练后保存模型和权重

在下一个代码示例中，我们使用保存的预训练权重开始一个完整的训练会话。首先我们加载未初始化的基础模型(`base_model`)，它不包括最顶层。然后我们将保存的预训练权重(`weights-init`)加载到模型中。接下来，我们添加一个新的最顶层，它是一个有 20 个节点的密集层，用于 20 个类别。我们构建新的模型，编译，然后开始完整的训练。

```
model = load_model('base_model')                                  ❶

model.load_weights('weights-init')                                ❷

output = Dense(20, activation='softmax')(model.output)            ❸

model = Model(model.input, output)                                ❹
model.compile(loss='categorical_crossentropy', optimizer='adam',  ❹
              metrics=['accuracy'])                               ❹
```

❶ 重新加载基础模型

❷ 使用域迁移权重初始化来初始化权重

❸ 添加不带 dropout 的分类器

❹ 编译并训练新模型

### 11.3.5 负迁移

在某些情况下，我们会发现迁移学习的结果比从头开始训练的准确度低：当使用预训练模型来训练新模型时，训练过程中的整体准确度低于如果没有预训练模型时的准确度。这被称为*负迁移*。

在这种情况下，源域和目标域非常不同，以至于源域的学习权重不能在目标域上重用。此外，当权重被重用时，模型将不会收敛，甚至可能会发散。一般来说，我们通常可以在五到十个 epoch 内发现负迁移。

## 11.4 超越计算机视觉

本章讨论的用于计算机视觉的迁移学习方法也适用于 NLU 模型。除了某些术语外，过程是相同的。在 NLU 模型中，移除顶层有时被称为*移除头部*。

在这两种情况下，你都是在移除所有或部分的任务组件，并用新的任务替换它。你所依赖的是类似于计算机视觉中的潜在空间；中间表示具有学习新任务所必需的上下文（特征）。对于相似任务和不同任务的方 法，在计算机视觉和 NLU 中是相同的。

然而，对于结构化数据来说，情况并非如此。实际上，跨域（数据集）的预训练模型之间不可能进行迁移学习。你可以在同一个数据集上学习不同类型的工作（例如，回归与分类），但你不能在不同特征的数据集之间重用学习到的权重。至少目前还没有一个概念——即具有可跨不同领域（列）的数据集重用基本特征的潜在空间。

## 摘要

+   来自 TF.Keras 和 TF Hub 模型存储库的预构建和预训练模型可以用于直接用于预测的重用，或者用于迁移学习新的分类器。

+   预训练模型的分类器组可以被替换，无论是通用的还是与类似域的，并且可以在更少的训练时间和更小的数据集上重新训练以适应新域。

+   在迁移学习中，如果新域与之前训练的域相似，则冻结所有层除了新的任务层，并进行微调训练。

+   在迁移学习中，如果新领域与之前训练的领域不同，你需要在重新训练时按顺序冻结和解冻层，从模型底部开始，逐步向上移动。

+   在领域迁移权重中，你使用训练模型的权重作为初始权重，并完全训练一个新的模型。
