# 第十章。人物检测：训练模型

在第九章中，我们展示了如何部署一个用于识别图像中人物的预训练模型，但我们没有解释该模型来自何处。如果您的产品有不同的要求，您将希望能够训练自己的版本，本章将解释如何做到这一点。

# 选择一台机器

训练这个图像模型需要比我们之前的示例更多的计算资源，因此如果您希望训练在合理的时间内完成，您需要使用一台配备高端图形处理单元（GPU）的机器。除非您期望运行大量训练作业，我们建议您首先租用云实例而不是购买特殊的机器。不幸的是，我们在之前章节中用于较小模型的免费 Colaboratory 服务将不起作用，您需要支付访问机器的费用。有许多优秀的提供商可供选择，但我们的说明将假定您正在使用谷歌云平台，因为这是我们最熟悉的服务。如果您已经在使用亚马逊网络服务（AWS）或微软 Azure，它们也支持 TensorFlow，并且训练说明应该是相同的，但您需要按照它们的教程设置机器。

# 设置谷歌云平台实例

您可以从谷歌云平台租用一个预安装了 TensorFlow 和 NVIDIA 驱动程序的虚拟机，并支持 Jupyter Notebook 网络界面，这可能非常方便。不过，设置这一点可能有点复杂。截至 2019 年 9 月，以下是您需要执行的步骤来创建一台机器：

1.  登录[*console.cloud.google.com*](https://oreil.ly/Of6oo)。如果您还没有谷歌账号，您需要创建一个，您还需要设置计费来支付您创建的实例。如果您还没有项目，您需要创建一个。

1.  在屏幕的左上角，打开汉堡菜单（带有三条水平线的主菜单图标，如#ai_platform_menu 所示），向下滚动直到找到人工智能部分。

1.  在此部分中，选择 AI 平台→笔记本，如#ai_platform_menu 所示。

    ![AI 平台菜单](img/timl_1001.png)

    ###### 图 10-1。AI 平台菜单

1.  您可能会看到一个提示，要求您启用计算引擎 API 以继续，如#api_enable 所示；请继续批准。这可能需要几分钟的时间。

    ![计算引擎 API](img/timl_1002.png)

    ###### 图 10-2。计算引擎 API 界面

1.  将打开一个“笔记本实例”屏幕。在顶部的菜单栏中，选择 NEW INSTANCE。在打开的子菜单中，选择“自定义实例”，如#new_instance 所示。

    ![实例创建菜单](img/timl_1003.png)

    ###### 图 10-3。实例创建菜单

1.  在“新笔记本实例”页面上，在“实例名称”框中，为您的机器命名，如#instance_naming 所示，然后向下滚动设置环境。

    ![实例命名界面](img/timl_1004.png)

    ###### 图 10-4。命名界面

1.  截至 2019 年 9 月，选择的正确 TensorFlow 版本是 TensorFlow 1.14。推荐的版本可能在您阅读本文时已经增加到 2.0 或更高，但可能存在一些不兼容性，因此如果可能的话，请从选择 1.14 或 1.x 分支的其他版本开始。

1.  在“机器配置”部分，选择至少 4 个 CPU 和 15GB 的 RAM，如#system_setup 所示。

    ![CPU 和版本界面](img/timl_1005.png)

    ###### 图 10-5。CPU 和版本界面

1.  选择正确的 GPU 将在训练速度上产生最大的差异。这可能有些棘手，因为并非所有区域都提供相同类型的硬件。在我们的情况下，我们使用“us-west1（俄勒冈州）”作为地区，“us-west-1b”作为区域，因为我们知道它们目前提供高端 GPU。您可以使用[Google Cloud Platform 的定价计算器](https://oreil.ly/t2XO0)获取详细的定价信息，但在本例中，我们选择了一块 NVIDIA Tesla V100 GPU，如图 10-6 所示。这每月花费 1300 美元，但可以让我们在大约一天内训练人员检测器模型，因此模型训练成本约为 45 美元。

    ![GPU 选择界面](img/timl_1006.png)

    ###### 图 10-6。GPU 选择界面

    ###### 提示

    这些高端机器的运行成本很高，因此请确保在不使用训练时停止实例。否则，您将为一个空闲的机器付费。

1.  自动安装 GPU 驱动程序会让生活变得更轻松，因此请确保选择该选项，如图 10-7 所示。

    ![GPU 驱动程序界面](img/timl_1007.png)

    ###### 图 10-7。GPU 驱动程序界面

1.  因为您将在这台机器上下载数据集，我们建议将引导磁盘大小调整为比默认的 100GB 大一些；也许大到 500GB，如图 10-8 所示。

    ![引导磁盘大小](img/timl_1008.png)

    ###### 图 10-8。增加引导磁盘大小

1.  当您设置好所有这些选项后，在页面底部点击 CREATE 按钮，这将返回到“笔记本实例”屏幕。列表中应该有一个新的实例，名称与您给机器的名称相同。在实例设置完成时，列表旁边会有旋转器几分钟。设置完成后，点击打开 JUPYTERLAB 链接，如图 10-9 所示。

    ![实例屏幕](img/timl_1009.png)

    ###### 图 10-9。实例屏幕

1.  在打开的屏幕中，选择创建一个 Python 3 笔记本（参见图 10-10）。

    ![笔记本选择屏幕](img/timl_1010.png)

    ###### 图 10-10。笔记本选择屏幕

    这为您提供了一个连接到实例的 Jupyter 笔记本。如果您不熟悉 Jupyter，它为您提供了一个漂亮的 Web 界面，用于运行在计算机上的 Python 解释器，并将命令和结果存储在一个可以共享的笔记本中。要开始使用它，在右侧面板中键入`**print("Hello World!")**`，然后按 Shift+Return。您应该会看到“Hello World！”打印在下方，如图 10-11 所示。如果是这样，您已成功设置了机器实例。我们将使用这个笔记本作为本教程其余部分输入命令的地方。

![hello world 示例](img/timl_1011.png)

###### 图 10-11。"hello world"示例

接下来的许多命令假定您是从 Jupyter 笔记本中运行的，因此它们以`!`开头，表示它们应该作为 shell 命令而不是 Python 语句运行。如果您直接从终端运行（例如，在打开安全外壳连接后与实例通信），可以删除初始的`!`。

# 训练框架选择

Keras 是在 TensorFlow 中构建模型的推荐接口，但在创建人员检测模型时，它还不支持我们需要的所有功能。因此，我们向您展示如何使用*tf.slim*训练模型，这是一个较旧的接口。它仍然被广泛使用，但已被弃用，因此未来版本的 TensorFlow 可能不支持这种方法。我们希望将来在线发布 Keras 说明；请查看[tinymlbook.com/persondetector](https://oreil.ly/sxP6q)获取更新。

Slim 的模型定义是[TensorFlow 模型存储库](https://oreil.ly/iamdB)的一部分，因此要开始，您需要从 GitHub 下载它：

```py
! cd ~
! git clone https://github.com/tensorflow/models.git
```

###### 注意

以下指南假定您已经从您的主目录中完成了这些操作，因此模型存储库代码位于*~/models*，并且除非另有说明，否则所有命令都是从主目录运行的。您可以将存储库放在其他位置，但您需要更新所有对它的引用。

要使用 Slim，您需要确保 Python 可以找到其模块并安装一个依赖项。以下是如何在 iPython 笔记本中执行此操作：

```py
! pip install contextlib2
import os
new_python_path = (os.environ.get("PYTHONPATH") or '') + ":models/research/slim"
%env PYTHONPATH=$new_python_path
```

通过像这样的`EXPORT`语句更新`PYTHONPATH`仅适用于当前的 Jupyter 会话，因此如果您直接使用 bash，您应该将其添加到持久性启动脚本中，运行类似于这样的内容：

```py
echo 'export PYTHONPATH=$PYTHONPATH:models/research/slim' >> ~/.bashrc
source ~/.bashrc
```

如果在运行 Slim 脚本时看到导入错误，请确保`PYTHONPATH`已正确设置并且已安装`contextlib2`。您可以在[存储库的 README](https://oreil.ly/azuvk)中找到有关*tf.slim*的更多一般信息。

# 构建数据集

为了训练我们的人员检测模型，我们需要一个大量的图像集，这些图像根据它们是否有人员进行了标记。用于训练图像分类器的 ImageNet 1,000 类数据集不包括人员的标签，但幸运的是[COCO 数据集](http://cocodataset.org/#home)包括这些标签。

数据集设计用于训练本地化模型，因此图像没有用“人”、“非人”类别标记，我们希望对其进行训练。相反，每个图像都附带一个包含其包含的所有对象的边界框列表。“人”是这些对象类别之一，因此为了获得我们想要的分类标签，我们需要寻找带有人边界框的图像。为了确保它们不会太小而无法识别，我们还需要排除非常小的边界框。Slim 包含一个方便的脚本，可以同时下载数据并将边界框转换为标签：

```py
! python download_and_convert_data.py \
  --dataset_name=visualwakewords \
  --dataset_dir=data/visualwakewords
```

这是一个大型下载，大约 40 GB，所以需要一段时间，您需要确保您的驱动器上至少有 100 GB 的空闲空间以供解压和进一步处理。如果整个过程需要大约 20 分钟才能完成，请不要感到惊讶。完成后，您将在*data/visualwakewords*中拥有一组 TFRecords，其中包含带有标记的图像信息。此数据集由 Aakanksha Chowdhery 创建，被称为[Visual Wake Words 数据集](https://oreil.ly/EC6nd)。它旨在用于基准测试和测试嵌入式计算机视觉，因为它代表了我们需要在严格的资源约束下完成的非常常见的任务。我们希望看到它推动更好的模型用于这个和类似的任务。

# 训练模型

使用*tf.slim*处理训练的好处之一是我们通常需要修改的参数可用作命令行参数，因此我们只需调用标准的*train_image_classifier.py*脚本来训练我们的模型。您可以使用此命令构建我们在示例中使用的模型：

```py
! python models/research/slim/train_image_classifier.py \
    --train_dir=vww_96_grayscale \
    --dataset_name=visualwakewords \
    --dataset_split_name=train \
    --dataset_dir=data/visualwakewords \
    --model_name=mobilenet_v1_025 \
    --preprocessing_name=mobilenet_v1 \
    --train_image_size=96 \
    --use_grayscale=True \
    --save_summaries_secs=300 \
    --learning_rate=0.045 \
    --label_smoothing=0.1 \
    --learning_rate_decay_factor=0.98 \
    --num_epochs_per_decay=2.5 \
    --moving_average_decay=0.9999 \
    --batch_size=96 \
    --max_number_of_steps=1000000
```

在单 GPU V100 实例上完成所有一百万步骤需要几天的时间，但如果您想要尽早进行实验，几个小时后您应该能够获得一个相当准确的模型。以下是一些额外的考虑事项：

+   检查点和摘要将保存在`--train_dir`参数中给定的文件夹中。这是您需要查看结果的地方。

+   `--dataset_dir`参数应该与您从 Visual Wake Words 构建脚本中保存 TFRecords 的目录匹配。

+   我们使用的架构由`--model_name`参数定义。`mobilenet_v1`前缀指示脚本使用 MobileNet 的第一个版本。我们尝试过后续版本，但这些版本对于其中间激活缓冲区使用了更多的 RAM，因此目前我们仍然坚持使用原始版本。`025`是要使用的深度乘数，这主要影响权重参数的数量；这个低设置确保模型适合在 250 KB 的闪存内。

+   `--preprocessing_name` 控制输入图像在被馈送到模型之前如何修改。`mobilenet_v1` 版本将图像的宽度和高度缩小到 `--train_image_size` 中给定的大小（在我们的例子中为 96 像素，因为我们想要减少计算需求）。它还将像素值从 0 到 255 的整数缩放为范围为 -1.0 到 +1.0 的浮点数（尽管我们将在训练后对其进行量化）。

+   我们在 SparkFun Edge 板上使用的 [HM01B0 相机](https://oreil.ly/RGciN) 是单色的，因此为了获得最佳结果，我们需要对黑白图像进行模型训练。我们传入 `--use_grayscale` 标志以启用该预处理。

+   `--learning_rate`、`--label_smoothing`、`--learning_rate_decay_factor`、`--num_epochs_per_decay`、`--moving_average_decay` 和 `--batch_size` 参数都控制训练过程中权重如何更新。训练深度网络仍然是一种黑暗的艺术，所以这些确切的值是通过对这个特定模型进行实验找到的。您可以尝试调整它们以加快训练速度或在准确性上获得一点提升，但我们无法为如何进行这些更改提供太多指导，而且很容易出现训练准确性永远不收敛的组合。

+   `--max_number_of_steps` 定义了训练应该持续多久。没有好的方法来提前确定这个阈值；您需要进行实验来确定模型的准确性何时不再提高，以知道何时停止。在我们的情况下，我们默认为一百万步，因为对于这个特定模型，我们知道这是一个停止的好时机。

启动脚本后，您应该看到类似以下的输出：

```py
INFO:tensorflow:global step 4670: loss = 0.7112 (0.251 sec/step)
  I0928 00:16:21.774756 140518023943616 learning.py:507] global step 4670: loss
  = 0.7112 (0.251 sec/step)
INFO:tensorflow:global step 4680: loss = 0.6596 (0.227 sec/step)
  I0928 00:16:24.365901 140518023943616 learning.py:507] global step 4680: loss
  = 0.6596 (0.227 sec/step)
```

不要担心行重复：这只是 TensorFlow 日志打印与 Python 交互的副作用。每行都包含关于训练过程的两个关键信息。全局步骤是我们进行训练的进度计数。因为我们将限制设置为一百万步，所以在这种情况下我们已经完成了近 5%。连同每秒步数的估计，这是有用的，因为您可以用它来估计整个训练过程的大致持续时间。在这种情况下，我们每秒完成大约 4 步，所以一百万步将需要大约 70 小时，或者 3 天。另一个关键信息是损失。这是部分训练模型的预测与正确值之间的接近程度的度量，较低的值更好。这将显示很多变化，但如果模型正在学习，平均情况下应该在训练过程中减少。因为它非常嘈杂，这些数量会在短时间内反弹很多次，但如果事情进展顺利，您应该在等待一个小时左右并返回检查时看到明显的下降。这种变化在图表中更容易看到，这也是尝试 TensorBoard 的主要原因之一。

# TensorBoard

TensorBoard 是一个 Web 应用程序，允许您查看来自 TensorFlow 训练会话的数据可视化，它默认包含在大多数云实例中。如果您正在使用 Google Cloud AI Platform，您可以通过在笔记本界面的左侧选项卡中打开命令面板，然后向下滚动选择“创建一个新的 TensorBoard”来启动一个新的 TensorBoard 会话。然后会提示您输入摘要日志的位置。输入您在训练脚本中用于 `--train_dir` 的路径——在上一个示例中，文件夹名称是 *vww_96_grayscale*。要注意的一个常见错误是在路径末尾添加斜杠，这将导致 TensorBoard 无法找到目录。

如果您在不同环境中从命令行启动 TensorBoard，您需要将此路径作为 `--logdir` 参数传递给 TensorBoard 命令行工具，并将浏览器指向 [*http://localhost:6006*](http://localhost:6006)（或者您正在运行它的机器的地址）。

在导航到 TensorBoard 地址或通过 Google Cloud 打开会话后，您应该看到一个类似于图 10-12 的页面。鉴于脚本仅每五分钟保存一次摘要，可能需要一段时间才能在图表中找到有用的内容。图 10-12 显示了训练一天以上后的结果。最重要的图表称为“clone_loss”；它显示了与日志输出中显示的相同损失值的进展。正如您在此示例中所看到的，它波动很大，但随着时间的推移总体趋势是向下的。如果在训练几个小时后没有看到这种进展，这表明您的模型没有收敛到一个好的解决方案，您可能需要调试数据集或训练参数出现的问题。

TensorBoard 在打开时默认显示 SCALARS 选项卡，但在训练期间可能有用的另一个部分是 IMAGES(图 10-13)。这显示了模型当前正在训练的图片的随机选择，包括任何扭曲和其他预处理。在图中，您可以看到图像已被翻转，并且在馈送到模型之前已被转换为灰度。这些信息并不像损失图表那样重要，但可以用来确保数据集符合您的期望，并且在训练进行时看到示例更新是有趣的。

![Tensorboard 中的训练图表](img/timl_1012.png)

###### 图 10-12. TensorBoard 中的图表

![Tensorboard 中的训练图片](img/timl_1013.png)

###### 图 10-13. TensorBoard 中的图片

# 评估模型

损失函数与模型训练的好坏相关，但它不是一个直接可理解的度量标准。我们真正关心的是模型正确检测到多少人，但要让它计算这一点，我们需要运行一个单独的脚本。您不需要等到模型完全训练，可以检查`--train_dir`文件夹中任何检查点的准确性。要执行此操作，请运行以下命令：

```py
! python models/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=vww_96_grayscale/model.ckpt-698580 \
    --dataset_dir=data/visualwakewords \
    --dataset_name=visualwakewords \
    --dataset_split_name=val \
    --model_name=mobilenet_v1_025 \
    --preprocessing_name=mobilenet_v1 \
    --use_grayscale=True \
    --train_image_size=96
```

您需要确保`--checkpoint_path`指向有效的检查点数据集。检查点存储在三个单独的文件中，因此值应为它们的公共前缀。例如，如果您有一个名为*model.ckpt-5179.data-00000-of-00001*的检查点文件，则前缀将是*model.ckpt-5179*。脚本应该生成类似于以下内容的输出：

```py
INFO:tensorflow:Evaluation [406/406]
I0929 22:52:59.936022 140225887045056 evaluation.py:167] Evaluation [406/406]
eval/Accuracy[0.717438412]eval/Recall_5[1]
```

这里的重要数字是准确率。它显示了被正确分类的图片的比例，在这种情况下是 72%，转换为百分比后。如果按照示例脚本进行，您应该期望完全训练的模型在一百万步后达到约 84%的准确率，并显示约 0.4 的损失。

# 将模型导出到 TensorFlow Lite

当模型训练到您满意的准确度时，您需要将结果从 TensorFlow 训练环境转换为可以在嵌入式设备上运行的形式。正如我们在之前的章节中看到的，这可能是一个复杂的过程，而*tf.slim*也添加了一些自己的特点。

## 导出到 GraphDef Protobuf 文件

Slim 每次运行其脚本时都会从`model_name`生成架构，因此要在 Slim 之外使用模型，需要将其保存为通用格式。我们将使用 GraphDef protobuf 序列化格式，因为 Slim 和 TensorFlow 的其余部分都能理解它：

```py
! python models/research/slim/export_inference_graph.py \
    --alsologtostderr \
    --dataset_name=visualwakewords \
    --model_name=mobilenet_v1_025 \
    --image_size=96 \
    --use_grayscale=True \
    --output_file=vww_96_grayscale_graph.pb
```

如果成功，您应该在主目录中有一个新的*vww_96_grayscale_graph.pb*文件。这包含了模型中操作的布局，但尚未包含任何权重数据。

## 冻结权重

将训练好的权重与操作图一起存储的过程称为*冻结*。这将所有图中的变量转换为常量，加载它们的值后从检查点文件中。接下来的命令使用了百万次训练步骤的检查点，但您可以提供任何有效的检查点路径。图冻结脚本存储在主 TensorFlow 存储库中，因此在运行此命令之前，您需要从 GitHub 下载这个脚本：

```py
! git clone https://github.com/tensorflow/tensorflow
! python tensorflow/tensorflow/python/tools/freeze_graph.py \
    --input_graph=vww_96_grayscale_graph.pb \
    --input_checkpoint=vww_96_grayscale/model.ckpt-1000000 \
    --input_binary=true --output_graph=vww_96_grayscale_frozen.pb \
    --output_node_names=MobilenetV1/Predictions/Reshape_1
```

之后，您应该看到一个名为*vww_96_grayscale_frozen.pb*的文件。

## 量化和转换为 TensorFlow Lite

量化是一个棘手而复杂的过程，仍然是一个活跃的研究领域，因此将我们迄今为止训练的浮点图转换为 8 位实体需要相当多的代码。您可以在第十五章中找到更多关于量化是什么以及它是如何工作的解释，但在这里我们将向您展示如何在我们训练的模型中使用它。大部分代码是准备示例图像以馈送到训练网络中，以便测量典型使用中激活层的范围。我们依赖`TFLiteConverter`类来处理量化并将其转换为我们需要用于推理引擎的 TensorFlow Lite FlatBuffer 文件：

```py
import tensorflow as tf
import io
import PIL
import numpy as np

def representative_dataset_gen():

  record_iterator = tf.python_io.tf_record_iterator
      (path='data/visualwakewords/val.record-00000-of-00010')

  count = 0
  for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    image_stream = io.BytesIO
        (example.features.feature['image/encoded'].bytes_list.value[0])
    image = PIL.Image.open(image_stream)
    image = image.resize((96, 96))
    image = image.convert('L')
    array = np.array(image)
    array = np.expand_dims(array, axis=2)
    array = np.expand_dims(array, axis=0)
    array = ((array / 127.5) - 1.0).astype(np.float32)
    yield([array])
    count += 1
    if count > 300:
        break

converter = tf.lite.TFLiteConverter.from_frozen_graph \
    ('vww_96_grayscale_frozen.pb', ['input'],  ['MobilenetV1/Predictions/ \
 Reshape_1'])
converter.inference_input_type = tf.lite.constants.INT8
converter.inference_output_type = tf.lite.constants.INT8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

tflite_quant_model = converter.convert()
open("vww_96_grayscale_quantized.tflite", "wb").write(tflite_quant_model)
```

## 转换为 C 源文件

转换器会写出一个文件，但大多数嵌入式设备没有文件系统。为了从我们的程序中访问序列化数据，我们必须将其编译到可执行文件中并存储在闪存中。最简单的方法是将文件转换为 C 数据数组，就像我们在之前的章节中所做的那样：

```py
# Install xxd if it is not available
! apt-get -qq install xxd
# Save the file as a C source file
! xxd -i vww_96_grayscale_quantized.tflite > person_detect_model_data.cc
```

现在，您可以用您训练过的版本替换现有的*person_detect_model_data.cc*文件，并能够在嵌入式设备上运行您自己的模型。

# 为其他类别训练

COCO 数据集中有 60 多种不同的对象类型，因此自定义模型的一种简单方法是在构建训练数据集时选择其中一种而不是`person`。以下是一个查找汽车的示例：

```py
! python models/research/slim/datasets/build_visualwakewords_data.py \
   --logtostderr \
   --train_image_dir=coco/raw-data/train2014 \
   --val_image_dir=coco/raw-data/val2014 \
   --train_annotations_file=coco/raw-data/annotations/instances_train2014.json \
   --val_annotations_file=coco/raw-data/annotations/instances_val2014.json \
   --output_dir=coco/processed_cars \
   --small_object_area_threshold=0.005 \
   --foreground_class_of_interest='car'
```

您应该能够按照与人员检测器相同的步骤进行操作，只需在以前的`data/visualwakewords`路径处用新的`coco/processed_cars`路径替换。

如果您感兴趣的对象类型在 COCO 中不存在，您可能可以使用迁移学习来帮助您在您收集的自定义数据集上进行训练，即使它很小。虽然我们还没有分享这方面的示例，但您可以查看[*tinymlbook.com*](http://tinymlbook.com)以获取有关这种方法的更新。

# 理解架构

[MobileNets](https://oreil.ly/tK57G)是一系列旨在提供尽可能少的权重参数和算术运算的良好准确性的架构。现在有多个版本，但在我们的情况下，我们使用原始的 v1，因为它在运行时需要的 RAM 最少。该架构背后的核心概念是*深度可分离卷积*。这是经典 2D 卷积的一种变体，以更高效的方式工作，而几乎不损失准确性。常规卷积根据在输入的所有通道上应用特定大小的滤波器来计算输出值。这意味着每个输出中涉及的计算数量是滤波器的宽度乘以高度，再乘以输入通道的数量。深度卷积将这个大计算分解为不同的部分。首先，每个输入通道通过一个或多个矩形滤波器进行滤波，以产生中间值。然后使用逐点卷积来组合这些值。这显著减少了所需的计算量，并且在实践中产生了与常规卷积类似的结果。

MobileNet v1 是由 14 个这些深度可分离卷积层堆叠而成，其中包括一个平均池化层，然后是一个全连接层，最后是一个 softmax 层。我们指定了一个*宽度乘数*为 0.25，这样可以将每次推断的计算量减少到约 6000 万次，通过将每个激活层中的通道数量缩减 75%与标准模型相比。本质上，它在操作上与普通的卷积神经网络非常相似，每一层都在学习输入中的模式。较早的层更像是边缘识别滤波器，识别图像中的低级结构，而较后的层将这些信息合成为更抽象的模式，有助于最终的对象分类。

# 总结

使用机器学习进行图像识别需要大量的数据和大量的处理能力。在本章中，您学习了如何从头开始训练模型，只提供数据集，并将该模型转换为适用于嵌入式设备的形式。

这种经验应该为您解决产品所需解决的机器视觉问题奠定了良好的基础。计算机能够看到并理解周围世界仍然有些神奇，所以我们迫不及待地想看看您会有什么创意！
