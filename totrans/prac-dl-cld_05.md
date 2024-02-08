# 第五章。从新手到大师预测者：最大化卷积神经网络准确性

在第一章中，我们探讨了负责任的人工智能开发的重要性。我们讨论的一个方面是我们模型的稳健性的重要性。用户只有在能够确信他们在日常生活中遇到的人工智能是准确可靠的情况下，才能信任我们构建的内容。显然，应用的背景非常重要。食物分类器偶尔将意大利面误分类为面包可能没问题。但是对于自动驾驶汽车将行人误认为街道车道就很危险。因此，本章的主要目标是构建更准确的模型。

在本章中，您将培养一种直觉，以识别下次开始训练时改进模型准确性的机会。我们首先看一下确保您不会盲目进行的工具。之后，在本章的大部分时间里，我们采取了一种非常实验性的方法，建立基线，隔离要调整的单个参数，并观察它们对模型性能和训练速度的影响。本章中使用的许多代码都汇总在一个 Jupyter Notebook 中，还附有一个可交互的示例清单。如果您选择将其纳入下次训练脚本中，它应该是非常可重用的。

我们探讨了在模型训练过程中经常出现的几个问题：

+   我不确定是使用迁移学习还是从头开始构建来训练自己的网络。对于我的情况，哪种方法更好？

+   我可以提供的最少数据量是多少，以获得可接受的结果？

+   我想确保模型正在学习正确的内容，而不是获取虚假相关性。我如何能够看到这一点？

+   如何确保我（或其他人）每次运行实验时都能获得相同的结果？换句话说，如何确保我的实验可重复性？

+   改变输入图像的长宽比是否会影响预测结果？

+   减少输入图像大小是否会对预测结果产生显著影响？

+   如果我使用迁移学习，应该微调多少层才能实现我偏好的训练时间与准确性的平衡？

+   或者，如果我从头开始训练，我的模型应该有多少层？

+   在模型训练期间提供适当的“学习率”是多少？

+   有太多事情需要记住。有没有一种方法可以自动化所有这些工作？

我们将尝试通过对几个数据集进行实验来逐一回答这些问题。理想情况下，您应该能够查看结果，阅读要点，并对实验所测试的概念有所了解。如果您感到更有冒险精神，您可以选择使用 Jupyter Notebook 自行进行实验。

# 行业工具

本章的主要重点之一是在试图获得高准确性的同时，减少实验过程中涉及的代码和工作量。存在一系列工具可以帮助我们使这个过程更加愉快：

TensorFlow 数据集

快速便捷地访问大约 100 个数据集，性能良好。所有知名数据集都可用，从最小的 MNIST（几兆字节）到最大的 MS COCO、ImageNet 和 Open Images（数百吉字节）。此外，还提供医学数据集，如结直肠组织学和糖尿病视网膜病变。

TensorBoard

近 20 种易于使用的方法，可视化训练的许多方面，包括可视化图形、跟踪实验以及检查通过网络的图像、文本和音频数据。

What-If 工具

在单独的模型上并行运行实验，并通过比较它们在特定数据点上的性能来揭示它们之间的差异。编辑单个数据点以查看它如何影响模型训练。

tf-explain

分析网络所做的决策，以识别数据集中的偏见和不准确性。此外，使用热图可视化网络在图像的哪些部分激活。

Keras 调谐器

一个为`tf.keras`构建的库，可以在 TensorFlow 2.0 中自动调整超参数。

AutoKeras

自动化神经架构搜索（NAS），跨不同任务进行图像、文本和音频分类以及图像检测。

自动增强

利用强化学习来改进现有训练数据集中的数据量和多样性，从而提高准确性。

现在让我们更详细地探索这些工具。

## TensorFlow 数据集

TensorFlow 数据集是一个近 100 个准备就绪数据集的集合，可以快速帮助构建用于训练 TensorFlow 模型的高性能输入数据管道。TensorFlow 数据集标准化了数据格式，使得很容易用另一个数据集替换一个数据集，通常只需更改一行代码。正如您将在后面看到的，将数据集分解为训练、验证和测试也只需一行代码。我们还将在下一章从性能的角度探索 TensorFlow 数据集。

您可以使用以下命令列出所有可用数据集（为了节省空间，此示例中仅显示了完整输出的一小部分）：

```py
import tensorflow_datasets as tfds
print(tfds.list_builders())
```

```py
['amazon_us_reviews', 'bair_robot_pushing_small', 'bigearthnet', 'caltech101',
'cats_vs_dogs', 'celeb_a', 'imagenet2012', … , 'open_images_v4',
'oxford_flowers102', 'stanford_dogs','voc2007', 'wikipedia', 'wmt_translate',
'xnli']
```

让我们看看加载数据集有多简单。稍后我们将把这个插入到一个完整的工作流程中：

```py
# Import necessary packages
import tensorflow_datasets as tfds

# Downloading and loading the dataset
dataset = tfds.load(name="cats_vs_dogs", split=tfds.Split.TRAIN)

# Building a performance data pipeline
dataset = dataset.map(preprocess).cache().repeat().shuffle(1024).batch(32).
prefetch(tf.data.experimental.AUTOTUNE)

model.fit(dataset, ...)
```

###### 提示

`tfds`生成了很多进度条，它们占用了很多屏幕空间——使用`tfds.disable_progress_bar()`可能是一个好主意。

## TensorBoard

TensorBoard 是您可视化需求的一站式服务，提供近 20 种工具来理解、检查和改进模型的训练。

传统上，为了跟踪实验进展，我们保存每个时代的损失和准确性值，然后在完成时使用`matplotlib`绘制。这种方法的缺点是它不是实时的。我们通常的选择是观察文本中的训练进度。此外，在训练完成后，我们需要编写额外的代码来在`matplotlib`中制作图表。TensorBoard 通过提供实时仪表板（图 5-1）来解决这些问题以及更多紧迫问题，帮助我们可视化所有日志（如训练/验证准确性和损失），以帮助理解训练的进展。它提供的另一个好处是能够比较当前实验的进展与上一个实验，这样我们就可以看到参数的变化如何影响我们的整体准确性。

![TensorBoard 默认视图展示实时训练指标（浅色线表示上一次运行的准确性）](img/00226.jpeg)

###### 图 5-1。TensorBoard 默认视图展示实时训练指标（浅色线表示上一次运行的准确性）

为了使 TensorBoard 能够可视化我们的训练和模型，我们需要使用摘要写入器记录有关我们的训练的信息：

```py
summary_writer = tf.summary.FileWriter('./logs')
```

要实时跟踪我们的训练，我们需要在模型训练开始之前加载 TensorBoard。我们可以使用以下命令加载 TensorBoard：

```py
# Get TensorBoard to run
%load_ext tensorboard

# Start TensorBoard
%tensorboard --logdir ./log
```

随着更多 TensorFlow 组件需要可视化用户界面，它们通过成为可嵌入插件在 TensorBoard 中重用 TensorBoard。您会注意到 TensorBoard 上的非活动下拉菜单；那里您可以看到 TensorFlow 提供的所有不同配置文件或工具。表 5-1 展示了各种可用工具中的一小部分。

表 5-1\. TensorBoard 的插件

| **TensorBoard 插件名称** | **描述** |
| --- | --- |
| 默认标量 | 可视化标量值，如分类准确度。 |
| 自定义标量 | 可视化用户定义的自定义指标。例如，不同类别的不同权重，这可能不是一个现成的指标。 |
| 图像 | 通过点击图像选项卡查看每个层的输出。 |
| 音频 | 可视化音频数据。 |
| 调试工具 | 允许可视化调试并设置条件断点（例如，张量包含 NaN 或无穷大）。 |
| 图表 | 以图形方式显示模型架构。 |
| 直方图 | 显示模型各层的权重分布随训练进展的变化。这对于检查使用量化压缩模型的效果特别有用。 |
| Projector | 使用 t-SNE、PCA 等可视化投影。 |
| 文本 | 可视化文本数据。 |
| PR 曲线 | 绘制精确率-召回率曲线。 |
| 概要 | 对模型中所有操作和层的速度进行基准测试。 |
| Beholder | 实时训练过程中可视化模型的梯度和激活。它允许按滤波器查看它们，并允许将它们导出为图像甚至视频。 |
| What-If 工具 | 通过切片和切块数据以及检查其性能来调查模型。特别有助于发现偏见。 |
| HParams | 查找哪些参数以及以什么值最重要，允许记录整个参数服务器（在本章中详细讨论）。 |
| 网格 | 可视化 3D 数据（包括点云）。 |

值得注意的是，TensorBoard 并不是特定于 TensorFlow 的，可以与其他框架如 PyTorch、scikit-learn 等一起使用，具体取决于所使用的插件。要使插件工作，我们需要编写要可视化的特定元数据。例如，TensorBoard 将 TensorFlow Projector 工具嵌入其中，以使用 t-SNE 对图像、文本或音频进行聚类（我们在第四章中详细讨论过）。除了调用 TensorBoard 外，我们还需要编写像图像的特征嵌入这样的元数据，以便 TensorFlow Projector 可以使用它来进行聚类，如图 5-2 中所示。

![TensorFlow 嵌入项目展示数据聚类，可以作为 TensorBoard 插件运行](img/00190.jpeg)

###### 图 5-2\. TensorFlow 嵌入项目展示数据聚类（可以作为 TensorBoard 插件运行）

## What-If 工具

如果我们能够通过可视化来检查我们的 AI 模型的预测结果会怎么样？如果我们能够找到最佳阈值来最大化精确度和召回率会怎么样？如果我们能够通过切片和切块数据以及我们的模型所做的预测来看到它擅长的地方以及有哪些改进机会会怎么样？如果我们能够比较两个模型以找出哪个确实更好会怎么样？如果我们能够通过在浏览器中点击几下就能做到所有这些以及更多呢？听起来肯定很吸引人！Google 的 People + AI Research（PAIR）倡议中的 What-If 工具（图 5-3 和图 5-4）帮助打开 AI 模型的黑匣子，实现模型和数据的可解释性。

![What-If 工具的数据点编辑器使根据数据集的注释和分类器的标签对数据进行过滤和可视化成为可能](img/00152.jpeg)

###### 图 5-3\. What-If 工具的数据点编辑器使根据数据集的注释和分类器的标签对数据进行过滤和可视化成为可能

![在 What-If 工具的性能和公平性部分中的 PR 曲线帮助交互式选择最佳阈值以最大化精确度和召回率](img/00018.jpeg)

###### 图 5-4。在 What-If 工具的性能和公平性部分中的 PR 曲线帮助交互式地选择最佳阈值以最大化精度和召回率

要使用 What-If 工具，我们需要数据集和一个模型。正如我们刚才看到的，TensorFlow Datasets 使得下载和加载数据（以`tfrecord`格式）相对容易。我们只需要定位数据文件即可。此外，我们希望将模型保存在同一目录中：

```py
# Save model for What If Tool
tf.saved_model.save(model, "/tmp/model/1/")
```

最好在本地系统中执行以下代码行，而不是在 Colab 笔记本中，因为 Colab 和 What-If 工具之间的集成仍在不断发展。

让我们开始 TensorBoard：

```py
$ mkdir tensorboard
$ tensorboard --logdir ./log --alsologtostderr
```

现在，在一个新的终端中，让我们为所有的 What-If 工具实验创建一个目录：

```py
$ mkdir what-if-stuff
```

将训练好的模型和 TFRecord 数据移动到这里。整体目录结构看起来像这样：

```py
$ tree .
├── colo
│   └── model
│         └── 1
│         ├── assets
│         ├── saved_model.pb
│         └── variables
```

我们将在新创建的目录中使用 Docker 来提供模型：

```py
$ sudo docker run -p 8500:8500 \
--mount type=bind,source=/home/{*`your_username`*}/what-if-stuff/colo/model/,
 target=/models/colo \
-e MODEL_NAME=colo -t tensorflow/serving
```

一句警告：端口必须是`8500`，所有参数必须与前面的示例中显示的完全相同。

接下来，在最右侧，点击设置按钮（灰色的齿轮图标），并添加表 5-2 中列出的值。

表 5-2。What-If 工具的配置

| **参数** | **值** |
| --- | --- |
| 推断地址 | `ip_addr:8500` |
| 模型名称 | `/models/colo` |
| 模型类型 | 分类 |
| 示例路径 | */home/{*`your_username`*}/what_if_stuff/colo/models/colo.tfrec*（注意：这必须是绝对路径） |

我们现在可以在 TensorBoard 中的浏览器中打开 What-If 工具，如图 5-5 所示。

![What-If 工具的设置窗口](img/00067.jpeg)

###### 图 5-5。What-If 工具的设置窗口

What-If 工具还可以根据不同的分组对数据集进行可视化，如图 5-6 所示。我们还可以使用该工具通过`set_compare_estimator_and_feature_spec`函数确定在同一数据集上多个模型中表现更好的模型。

```py
from witwidget.notebook.visualization import WitConfigBuilder

*`# features are the test examples that we want to load into the tool`*
models = [model2, model3, model4]
config_builder =
WitConfigBuilder(test_examples).set_estimator_and_feature_spec(model1, features)

for each_model in models:
    config_builder =
 config_builder.set_compare_estimator_and_feature_spec(each_model, features)
```

![What-If 工具可以使用多个指标、数据可视化等等](img/00025.jpeg)

###### 图 5-6。What-If 工具可以使用多个指标、数据可视化等等

现在，我们可以加载 TensorBoard，然后在可视化部分选择我们想要比较的模型，如图 5-7 所示。这个工具有很多功能可以探索！

![使用 What-If 工具选择要比较的模型](img/00313.jpeg)

###### 图 5-7。使用 What-If 工具选择要比较的模型

## tf-explain

传统上，深度学习模型一直是黑匣子，直到现在，我们通常通过观察类别概率和验证准确性来了解它们的性能。为了使这些模型更具可解释性和可解释性，热图应运而生。通过显示导致预测的图像区域的强度更高，热图可以帮助可视化它们的学习过程。例如，经常在雪地中看到的动物可能会得到高准确度的预测，但如果数据集中只有那种动物和雪作为背景，模型可能只会关注雪作为与动物不同的模式，而不是动物本身。这样的数据集展示了偏见，使得当分类器置于现实世界中时，预测不够稳健（甚至可能危险！）。热图可以特别有用，以探索这种偏见，因为如果数据集没有经过仔细筛选，往往会渗入虚假相关性。

`tf-explain`（由 Raphael Meudec 开发）通过这样的可视化帮助理解神经网络的结果和内部工作，揭示数据集中的偏见。我们可以在训练时添加多种类型的回调，或者使用其核心 API 生成后续可以加载到 TensorBoard 中的 TensorFlow 事件。对于推断，我们只需要传递一张图像、其 ImageNet 对象 ID 以及一个模型到 tf-explain 的函数中。您必须提供对象 ID，因为`tf.explain`需要知道为该特定类别激活了什么。`tf.explain`提供了几种不同的可视化方法：

Grad CAM

梯度加权类激活映射（Grad CAM）通过查看激活图来可视化图像的部分如何影响神经网络的输出。基于最后一个卷积层的对象 ID 的梯度生成热图（在图 5-8 中有示例）。Grad CAM 在很大程度上是一个广谱热图生成器，因为它对噪声具有鲁棒性，并且可以用于各种 CNN 模型。

遮挡敏感性

通过遮挡图像的一部分（使用随机放置的小方块补丁）来确定网络的稳健性。如果预测仍然正确，那么网络平均上是稳健的。图像中最温暖（即红色）的区域在遮挡时对预测的影响最大。

激活

可视化卷积层的激活。

![使用 MobileNet 和 tf-explain 在图像上进行可视化](img/00272.jpeg)

###### 图 5-8。使用 MobileNet 和 tf-explain 在图像上进行可视化

如下面的代码示例所示，这样的可视化可以用很少的代码构建。通过拍摄视频，生成单独的帧，并使用 Grad CAM 运行 tf-explain 并将它们组合在一起，我们可以详细了解这些神经网络如何对移动摄像机角度做出反应。

```py
from tf_explain.core.grad_cam import GradCAM
From tf.keras.applications.MobileNet import MobileNet

model = MobileNet(weights='imagenet', include_top=True)

# Set Grad CAM System
explainer = GradCAM()

# Image Processing
IMAGE_PATH = 'dog.jpg'
dog_index = 263
img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)
data = ([img], None)

# Passing the image through Grad CAM
grid = explainer.explain(data, model, 'conv1', index)
name = IMAGE_PATH.split(".jpg")[0]
explainer.save(grid, '/tmp', name + '_grad_cam.png')
```

# 机器学习实验的常见技术

前几章着重于训练模型。然而，接下来的几节包含一些在运行训练实验时要牢记的事项。

## 数据检查

数据检查的第一个最大障碍是确定数据的结构。TensorFlow Datasets 使得这一步相对容易，因为所有可用的数据集都具有相同的格式和结构，并且可以以高效的方式使用。我们只需要将数据集加载到 What-If 工具中，并使用已经存在的各种选项来检查数据。例如，在 SMILE 数据集上，我们可以根据其注释可视化数据集，例如戴眼镜和不戴眼镜的人的图像，如图 5-9 所示。我们观察到数据集中有更广泛的分布是没有戴眼镜的人的图像，从而揭示了由于数据不平衡而导致的数据偏见。这可以通过通过工具修改指标的权重来解决。

![根据预测和真实类别切分和分割数据](img/00231.jpeg)

###### 图 5-9。根据预测和真实类别切分和分割数据

## 数据分割：训练、验证、测试

将数据集分割为训练、验证和测试集非常重要，因为我们希望在分类器（即测试数据集）上报告结果。TensorFlow Datasets 使得下载、加载和将数据集分割为这三部分变得容易。一些数据集已经带有三个默认的分割。另外，数据可以按百分比进行分割。以下代码展示了使用默认分割：

```py
dataset_name = "cats_vs_dogs"
train, info_train = tfds.load(dataset_name, split=tfds.Split.TRAIN,
                    with_info=True)
```

在`tfds`中的猫狗数据集只有预定义的训练集分割。与此类似，TensorFlow 数据集中的一些数据集没有`validation`分割。对于这些数据集，我们从预定义的`training`集中取一小部分样本，并将其视为`validation`集。总而言之，使用`weighted_splits`来拆分数据集可以处理在拆分之间随机化和洗牌数据：

```py
# Load the dataset
dataset_name = "cats_vs_dogs"

# Dividing data into train (80), val (10) and test (10)
split_train, split_val, split_test = tfds.Split.TRAIN.subsplit(weighted=[80, 10,
                                     10])
train, info_train = tfds.load(dataset_name, split=split_train , with_info=True)
val, info_val = tfds.load(dataset_name, split=split_val, with_info=True)
test, info_test = tfds.load(dataset_name, split=split_test, with_info=True)
```

## 早停

早停有助于避免网络过度训练，通过监视显示有限改进的时期的数量。假设一个模型被设置为训练 1,000 个时期，在第 10 个时期达到 90%的准确率，并在接下来的 10 个时期内不再有进一步的改进，那么继续训练可能是一种资源浪费。如果时期数超过了一个名为`patience`的预定义阈值，即使可能还有更多的时期可以训练，训练也会停止。换句话说，早停决定了训练不再有用的时刻，并停止训练。我们可以使用`monitor`参数更改指标，并将早停添加到模型的回调列表中：

```py
# Define Early Stopping callback
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
					 min_delta=0.0001, patience=10)

# Add to the training model
model.fit_generator(... callbacks=[earlystop_callback])
```

## 可重现的实验

训练一次网络。然后，再次训练，而不更改任何代码或参数。您可能会注意到，即使在代码中没有进行任何更改，两次连续运行的准确性也略有不同。这是由于随机变量造成的。为了使实验在不同运行中可重现，我们希望控制这种随机化。模型权重的初始化、数据的随机洗牌等都利用了随机化算法。我们知道，通过初始化种子，可以使随机数生成器可重现，这正是我们要做的。各种框架都有设置随机种子的方法，其中一些如下所示：

```py
# Seed for Tensorflow
tf.random.set_seed(1234)

# Seed for Numpy
import numpy as np
np.random.seed(1234)

# Seed for Keras
seed = 1234
fit(train_data, augment=True, seed=seed)
flow_from_dataframe(train_dataframe, shuffle=True, seed=seed)
```

###### 注意

在所有正在使用的框架和子框架中设置种子是必要的，因为种子在框架之间不可转移。

# 端到端深度学习示例管道

让我们结合几个工具，构建一个骨干框架，这将作为我们的管道，在其中我们将添加和删除参数、层、功能和各种其他附加组件，以真正理解发生了什么。按照书籍 GitHub 网站上的代码（参见[*http://PracticalDeepLearning.ai*](http://PracticalDeepLearning.ai)），您可以在 Colab 中的浏览器中交互式运行此代码，针对 100 多个数据集进行修改。此外，您可以将其修改为大多数分类任务。

## 基本迁移学习管道

首先，让我们为迁移学习构建这个端到端示例。

```py
# Import necessary packages
import tensorflow as tf
import tensorflow_datasets as tfds

# tfds makes a lot of progress bars, which takes up a lot of screen space, hence
# disabling them
tfds.disable_progress_bar()

tf.random.set_seed(1234)

# Variables
BATCH_SIZE = 32
NUM_EPOCHS= 20
IMG_H = IMG_W = 224
IMG_SIZE = 224
LOG_DIR = './log'
SHUFFLE_BUFFER_SIZE = 1024
IMG_CHANNELS = 3

dataset_name = "oxford_flowers102"

def preprocess(ds):
  x = tf.image.resize_with_pad(ds['image'], IMG_SIZE, IMG_SIZE)
  x = tf.cast(x, tf.float32)
  x = (x/127.5) - 1
  return x, ds['label']

def augmentation(image,label):
  image = tf.image.random_brightness(image, .1)
  image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
  image = tf.image.random_flip_left_right(image)
  return image, label

def get_dataset(dataset_name):
  split_train, split_val = tfds.Split.TRAIN.subsplit(weighted=[9,1])
  train, info_train = tfds.load(dataset_name, split=split_train , with_info=True)
  val, info_val = tfds.load(dataset_name, split=split_val, with_info=True)
  NUM_CLASSES = info_train.features['label'].num_classes
  assert NUM_CLASSES >= info_val.features['label'].num_classes
  NUM_EXAMPLES = info_train.splits['train'].num_examples * 0.9
  IMG_H, IMG_W, IMG_CHANNELS = info_train.features['image'].shape
  train = train.map(preprocess).cache().
          repeat().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
  train = train.map(augmentation)
  train = train.prefetch(tf.data.experimental.AUTOTUNE)
  val = val.map(preprocess).cache().repeat().batch(BATCH_SIZE)
  val = val.prefetch(tf.data.experimental.AUTOTUNE)
  return train, info_train, val, info_val, IMG_H, IMG_W, IMG_CHANNELS,
         NUM_CLASSES, NUM_EXAMPLES

train, info_train, val, info_val, IMG_H, IMG_W, IMG_CHANNELS, NUM_CLASSES,
NUM_EXAMPLES = get_dataset(dataset_name)

# Allow TensorBoard callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_DIR,
                                                      histogram_freq=1,
                                                      write_graph=True,
                                                      write_grads=True,
                                                      batch_size=BATCH_SIZE,
                                                      write_images=True)

def transfer_learn(train, val, unfreeze_percentage, learning_rate):
   mobile_net = tf.keras.applications.ResNet50(input_shape=(IMG_SIZE, IMG_SIZE,
                IMG_CHANNELS), include_top=False)
   mobile_net.trainable=False
   # Unfreeze some of the layers according to the dataset being used
   num_layers = len(mobile_net.layers)
   for layer_index in range(int(num_layers - unfreeze_percentage*num_layers),
                             num_layers ):
   		mobile_net.layers[layer_index].trainable = True
   model_with_transfer_learning = tf.keras.Sequential([mobile_net,
                          tf.keras.layers.GlobalAveragePooling2D(),
                          tf.keras.layers.Flatten(),
                          tf.keras.layers.Dense(64),
                          tf.keras.layers.Dropout(0.3),
                          tf.keras.layers.Dense(NUM_CLASSES, 
                                                activation='softmax')],)
  model_with_transfer_learning.compile(
               optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                   loss='sparse_categorical_crossentropy',
                   metrics=["accuracy"])
  model_with_transfer_learning.summary()
  earlystop_callback = tf.keras.callbacks.EarlyStopping(
                                   monitor='val_accuracy', 
                                   min_delta=0.0001, 
                                   patience=5)
  model_with_transfer_learning.fit(train,
                                   epochs=NUM_EPOCHS,
                                   steps_per_epoch=int(NUM_EXAMPLES/BATCH_SIZE),
                                   validation_data=val,
                                   validation_steps=1,
                                   validation_freq=1,
                                   callbacks=[tensorboard_callback,
                                              earlystop_callback])
  return model_with_transfer_learning

# Start TensorBoard
%tensorboard --logdir ./log

# Select the last % layers to be trained while using the transfer learning
# technique. These layers are the closest to the output layers.
unfreeze_percentage = .33
learning_rate = 0.001

model = transfer_learn(train, val, unfreeze_percentage, learning_rate)
```

## 基本自定义网络管道

除了在最先进的模型上进行迁移学习外，我们还可以通过构建自己的自定义网络来进行实验和开发更好的直觉。只需在先前定义的迁移学习代码中交换模型即可：

```py
def create_model():
  model = tf.keras.Sequential([
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
          input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)),
     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
     tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
     tf.keras.layers.Dropout(rate=0.3),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dropout(rate=0.3),
     tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
  ])
  return model 

def scratch(train, val, learning_rate):
  model = create_model()
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

  earlystop_callback = tf.keras.callbacks.EarlyStopping(
                 monitor='val_accuracy', 
                 min_delta=0.0001, 
                 patience=5)

  model.fit(train,
           epochs=NUM_EPOCHS,
           steps_per_epoch=int(NUM_EXAMPLES/BATCH_SIZE),
           validation_data=val, 
           validation_steps=1,
           validation_freq=1,
           callbacks=[tensorboard_callback, earlystop_callback])
  return model
```

现在，是时候利用我们的管道进行各种实验了。

# 超参数如何影响准确性

在本节中，我们旨在逐一修改深度学习管道的各种参数——从微调的层数到使用的激活函数的选择——主要看其对验证准确性的影响。此外，当相关时，我们还观察其对训练速度和达到最佳准确性的时间（即收敛）的影响。

我们的实验设置如下：

+   为了减少实验时间，本章中我们使用了一个更快的架构——MobileNet。

+   我们将输入图像分辨率降低到 128 x 128 像素以进一步加快训练速度。一般来说，我们建议在生产系统中使用更高的分辨率（至少 224 x 224）。

+   如果实验连续 10 个时期准确率不增加，将应用早停。

+   对于使用迁移学习进行训练，通常解冻最后 33%的层。

+   学习率设置为 0.001，使用 Adam 优化器。

+   除非另有说明，我们主要使用牛津花卉 102 数据集进行测试。我们选择这个数据集是因为它相对难以训练，包含了大量类别（102 个）以及许多类别之间的相似之处，这迫使网络对特征进行细粒度理解以取得良好的效果。

+   为了进行苹果与苹果的比较，我们取特定实验中的最大准确性值，并将该实验中的所有其他准确性值相对于该最大值进行归一化。

基于这些和其他实验，我们总结了一份可操作的提示清单，可在下一个模型训练冒险中实施。这些内容可以在本书的 GitHub（参见[*http://PracticalDeepLearning.ai*](http://PracticalDeepLearning.ai)）上找到，还有交互式可视化。如果您有更多提示，请随时在推特上发表[@PracticalDLBook](https://twitter.com/PracticalDLBook)或提交拉取请求。

## 迁移学习与从头开始训练

实验设置

训练两个模型：一个使用迁移学习，一个从头开始在相同数据集上训练。

使用的数据集

牛津花卉 102，结肠组织学

使用的架构

预训练的 MobileNet，自定义模型

图 5-10 显示了结果。

![比较在不同数据集上进行迁移学习和训练自定义模型](img/00194.jpeg)

###### 图 5-10。比较在不同数据集上进行迁移学习和训练自定义模型

以下是关键要点：

+   通过重复使用先前学习的特征，迁移学习可以使训练期间的准确性迅速提高。

+   尽管预计基于 ImageNet 上预训练模型的迁移学习在目标数据集也是自然图像时会起作用，但网络在早期层学习的模式对超出 ImageNet 范围的数据集也能奇妙地奏效。这并不一定意味着它会产生最佳结果，但可以接近。当图像与模型预训练的更多真实世界图像匹配时，我们可以相对快速地提高准确性。

## 迁移学习中微调层数的影响

实验设置

将可训练层的百分比从 0 变化到 100%

使用的数据集

牛津花卉 102

使用的架构

预训练的 MobileNet

图 5-11 显示了结果。

![微调的层数对模型准确性的影响](img/00159.jpeg)

###### 图 5-11。微调的层数对模型准确性的影响

以下是关键要点：

+   微调的层数越多，达到收敛所需的纪元越少，准确性越高。

+   微调的层数越多，每个纪元的训练时间就越长，因为涉及更多的计算和更新。

+   对于需要对图像进行细粒度理解的数据集，通过解冻更多层使其更具任务特定性是获得更好模型的关键。

## 数据大小对迁移学习的影响

实验设置

每次添加一个类别的图像

使用的数据集

猫与狗

使用的架构

预训练的 MobileNet

图 5-12 显示了结果。

![每个类别数据量对模型准确性的影响](img/00108.jpeg)

###### 图 5-12。每个类别数据量对模型准确性的影响

以下是关键要点：

+   即使每个类别只有三张图像，模型也能够以接近 90%的准确性进行预测。这显示了迁移学习在减少数据需求方面的强大作用。

+   由于 ImageNet 有几个猫和狗，所以在 ImageNet 上预训练的网络更容易适应我们的数据集。像牛津花卉 102 这样更困难的数据集可能需要更多的图像才能达到类似的准确性。

## 学习率的影响

实验设置

在 0.1、0.01、0.001 和 0.0001 之间变化学习率

使用的数据集

牛津花卉 102

使用的架构

预训练的 MobileNet

图 5-13 显示了结果。

![学习率对模型准确性和收敛速度的影响](img/00071.jpeg)

###### 图 5-13\. 学习率对模型准确性和收敛速度的影响

以下是关键要点：

+   学习率过高，模型可能永远无法收敛。

+   学习率过低会导致收敛所需时间过长。

+   在快速训练中找到合适的平衡至关重要。

## 优化器的影响

实验设置

尝试可用的优化器，包括 AdaDelta、AdaGrad、Adam、梯度下降、动量和 RMSProp

使用的数据集

牛津花卉 102

使用的架构

预训练的 MobileNet

图 5-14 显示了结果。

![不同优化器对收敛速度的影响](img/00030.jpeg)

###### 图 5-14\. 不同优化器对收敛速度的影响

以下是关键要点：

+   Adam 是更快收敛到高准确性的不错选择。

+   RMSProp 通常更适用于 RNN 任务。

## 批量大小的影响

实验设置

以 2 的幂变化批量大小

使用的数据集

牛津花卉 102

使用的架构

预训练

图 5-15 显示了结果。

![批量大小对准确性和收敛速度的影响](img/00317.jpeg)

###### 图 5-15\. 批量大小对准确性和收敛速度的影响

以下是关键要点：

+   批量大小越大，结果从一个时期到另一个时期的不稳定性就越大，波动也越大。但更高的准确性也会导致更高效的 GPU 利用率，因此每个时期的速度更快。

+   批量大小过低会减缓准确性的提升。

+   16/32/64 是很好的起始批量大小。

## 调整大小的影响

实验设置

将图像大小改为 128x128、224x224

使用的数据集

牛津花卉 102

使用的架构

预训练

图 5-16 显示了结果。

![图像大小对准确性的影响](img/00275.jpeg)

###### 图 5-16\. 图像大小对准确性的影响

以下是关键要点：

+   即使像素只有三分之一，验证准确性也没有显著差异。这一方面显示了 CNN 的稳健性。这可能部分是因为牛津花卉 102 数据集中有花朵的特写可见。对于对象在图像中占比较小的数据集，结果可能较低。

## 宽高比变化对迁移学习的影响

实验设置

拍摄具有不同宽高比（宽：高比）的图像，并将它们调整为正方形（1:1 宽高比）。

使用的数据集

猫与狗

使用的架构

预训练

图 5-17 显示了结果。

![图像中宽高比和对应准确性的分布](img/00168.jpeg)

###### 图 5-17\. 图像中宽高比和对应准确性的分布

以下是关键要点：

+   最常见的宽高比是 4:3，即 1.33，而我们的神经网络通常在 1:1 的比例下进行训练。

+   神经网络对由调整为正方形形状引起的宽高比的轻微修改相对稳健。即使达到 2.0 的比例也能得到不错的结果。

# 自动调整工具以获得最大准确性

正如我们自 19 世纪以来所看到的，自动化总是导致生产力的提高。在本节中，我们研究可以帮助我们自动搜索最佳模型的工具。

## Keras 调谐器

由于有许多潜在的超参数组合需要调整，找到最佳模型可能是一个繁琐的过程。通常，两个或更多参数可能会对收敛速度和验证准确性产生相关影响，因此逐个调整可能不会导致最佳模型。如果好奇心占了上风，我们可能想要同时对所有超参数进行实验。

Keras Tuner 用于自动化超参数搜索。我们定义了一个搜索算法，每个参数可以取的潜在值（例如，离散值或范围），我们要最大化的目标对象（例如，验证准确性），然后坐下来看程序开始训练。Keras Tuner 代表我们进行多次实验，改变参数，存储最佳模型的元数据。以下代码示例改编自 Keras Tuner 文档，展示了通过不同的模型架构进行搜索（在 2 到 10 层之间变化），以及调整学习率（在 0.1 到 0.001 之间）：

```py
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

# Input data
(x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
x = x.astype('float32') / 255.
val_x = val_x.astype('float32') / 255.

# Defining hyper parameters
hp = HyperParameters()
hp.Choice('learning_rate', [0.1, 0.001])
hp.Int('num_layers', 2, 10)

# Defining model with expandable number of layers
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for _ in range(hp.get('num_layers')):
        model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

hypermodel = RandomSearch(
             build_model,
             max_trials=20, # Number of combinations allowed
             hyperparameters=hp,
             allow_new_entries=False,
             objective='val_accuracy')

hypermodel.search(x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))

# Show summary of overall best model
hypermodel.results_summary()
```

每次实验都会显示类似这样的数值：

```py
 > Hp values:
  |-learning_rate: 0.001
  |-num_layers: 6
┌──────────────┬────────────┬───────────────┐
│ Name         │ Best model │ Current model │
├──────────────┼────────────┼───────────────┤
│ accuracy     │ 0.9911     │ 0.9911        │
│ loss         │ 0.0292     │ 0.0292        │
│ val_loss     │ 0.227      │ 0.227         │
│ val_accuracy │ 0.9406     │ 0.9406        │
└──────────────┴────────────┴───────────────┘
```

在实验结束时，结果摘要提供了迄今为止进行的实验的快照，并保存了更多元数据。

```py
Hypertuning complete - results in ./untitled_project
[Results summary]
 |-Results in ./untitled_project
 |-Ran 20 trials
 |-Ran 20 executions (1 per trial)
 |-Best val_accuracy: 0.9406
```

另一个重要的好处是能够实时在线跟踪实验并通过访问[*http://keras-tuner.appspot.com*](http://keras-tuner.appspot.com)获取进展通知，获取 API 密钥（来自 Google App Engine），并在我们的 Python 程序中输入以下行以及真实的 API 密钥：

```py
tuner.enable_cloud(api_key=api_key)
```

由于潜在的组合空间可能很大，随机搜索比网格搜索更受青睐，因为这是一种更实际的方式来在有限的实验预算上找到一个好的解决方案。但也有更快的方法，包括 Hyperband（Lisha Li 等人）的实现也在 Keras Tuner 中可用。

对于计算机视觉问题，Keras Tuner 包括可调整的应用程序，如 HyperResNet。

## AutoAugment

另一个示例超参数是增强。要使用哪些增强？增强的幅度有多大？组合太多会使情况变得更糟吗？我们可以让人工智能来决定，而不是将这些决定留给人类。AutoAugment 利用强化学习来找到最佳的增强组合（如平移、旋转、剪切）以及应用的概率和幅度，以最大化验证准确性。（该方法由 Ekin D. Cubuk 等人应用，以得出新的 ImageNet 验证数据的最新技术成果。）通过在 ImageNet 上学习最佳的增强参数组合，我们可以轻松地将其应用到我们的问题上。

应用从 ImageNet 预先学习的增强策略非常简单：

```py
from PIL import Image
from autoaugment import ImageNetPolicy
img = Image.open("cat.jpg")
policy = ImageNetPolicy()
imgs = [policy(img) for _ in range(8) ]
```

图 5-18 显示了结果。

![通过在 ImageNet 数据集上学习的增强策略的输出](img/00140.jpeg)

###### 图 5-18。通过在 ImageNet 数据集上学习的增强策略的输出

## AutoKeras

随着人工智能自动化越来越多的工作，不足为奇它最终也可以自动设计人工智能架构。NAS 方法利用强化学习将小型架构块连接在一起，直到它们能够最大化目标函数；换句话说，我们的验证准确性。当前最先进的网络都基于 NAS，使人类设计的架构相形见绌。这一领域的研究始于 2017 年，2018 年更加注重使训练更快。现在有了 AutoKeras（Haifeng Jin 等人），我们也可以以相对容易的方式在我们的特定数据集上应用这种最先进的技术。

使用 AutoKeras 生成新的模型架构只需提供我们的图像和相关标签，以及一个完成作业的时间限制。在内部，它实现了几种优化算法，包括贝叶斯优化方法来搜索最佳架构：

```py
!pip3 install autokeras
!pip3 install graphviz
from keras.datasets import mnist
from autokeras.image.image_supervised import ImageClassifier

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

clf = ImageClassifier(path=".",verbose=True, augment=False)
clf.fit(x_train, y_train, time_limit= 30 * 60) # 30 minutes
clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
y = clf.evaluate(x_test, y_test)
print(y)

*`# Save the model as a pickle file`*
clf.export_autokeras_model("model.pkl")

visualize('.')
```

训练后，我们都渴望了解新模型架构的样子。与我们通常看到的大多数干净的图像不同，这个看起来相当难以理解或打印出来。但我们相信的是，它能产生高准确性。

# 总结

在本章中，我们看到了一系列工具和技术，帮助探索改进 CNN 准确性的机会。通过建立迭代实验的案例，您了解到调整超参数如何带来最佳性能。由于有这么多超参数可供选择，我们随后看了自动化方法，包括 AutoKeras、AutoAugment 和 Keras Tuner。最重要的是，本章的核心代码结合了多个工具，存储在书的 GitHub 上（请参阅[*http://PracticalDeepLearning.ai*](http://PracticalDeepLearning.ai)），可以通过一行更改轻松调整到 100 多个数据集，并在浏览器中在线运行。此外，我们编制了一份可操作提示清单，以及在线托管的交互式实验，帮助您的模型获得一点额外优势。我们希望本章涵盖的内容能够使您的模型更加健壮，减少偏见，使其更易解释，并最终有助于负责任地发展人工智能。
