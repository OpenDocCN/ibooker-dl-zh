# 第八章：唤醒词检测：训练模型

在第七章中，我们围绕一个训练有能力识别“是”和“否”的模型构建了一个应用程序。在本章中，我们将训练一个*新*模型，可以识别不同的单词。

我们的应用代码相当通用。它只是捕获和处理音频，将其输入到 TensorFlow Lite 模型中，并根据输出执行某些操作。它大多数情况下不关心模型正在寻找哪些单词。这意味着如果我们训练一个新模型，我们可以直接将其放入我们的应用程序中，它应该立即运行。

在训练新模型时，我们需要考虑以下事项：

输入

新模型必须在与我们的应用代码相同形状和格式的输入数据上进行训练，具有相同的预处理。

输出

新模型的输出必须采用相同的格式：每个类别一个概率张量。

训练数据

无论我们选择哪些新单词，我们都需要很多人说这些单词的录音，这样我们才能训练我们的新模型。

优化

模型必须经过优化，以在内存有限的微控制器上高效运行。

幸运的是，我们现有的模型是使用由 TensorFlow 团队发布的公开可用脚本进行训练的，我们可以使用这个脚本来训练一个新模型。我们还可以访问一个免费的口语音频数据集，可以用作训练数据。

在下一节中，我们将详细介绍使用此脚本训练模型的过程。然后，在“在我们的项目中使用模型”中，我们将把新模型整合到我们现有的应用程序代码中。之后，在“模型的工作原理”中，您将了解模型的实际工作原理。最后，在“使用您自己的数据进行训练”中，您将看到如何使用您自己的数据集训练模型。

# 训练我们的新模型

我们正在使用的模型是使用 TensorFlow [Simple Audio Recognition](https://oreil.ly/E292V)脚本进行训练的，这是一个示例脚本，旨在演示如何使用 TensorFlow 构建和训练用于音频识别的模型。

该脚本使训练音频识别模型变得非常容易。除其他事项外，它还允许我们执行以下操作：

+   下载一个包含 20 个口语单词的音频数据集。

+   选择要训练模型的单词子集。

+   指定在音频上使用哪种类型的预处理。

+   选择几种不同类型的模型架构。

+   使用量化将模型优化为微控制器。

当我们运行脚本时，它会下载数据集，训练模型，并输出代表训练模型的文件。然后我们使用其他工具将此文件转换为适合 TensorFlow Lite 的正确形式。

###### 注意

模型作者通常会创建这些类型的训练脚本。这使他们能够轻松地尝试不同变体的模型架构和超参数，并与他人分享他们的工作。

运行训练脚本的最简单方法是在 Colaboratory（Colab）笔记本中进行，我们将在下一节中进行。

## 在 Colab 中训练

Google Colab 是一个很好的训练模型的地方。它提供了云中强大的计算资源，并且设置了我们可以用来监视训练过程的工具。而且它完全免费。

在本节中，我们将使用 Colab 笔记本来训练我们的新模型。我们使用的笔记本可以在 TensorFlow 存储库中找到。

[打开笔记本](https://oreil.ly/0Z2ra)并单击“在 Google Colab 中运行”按钮，如图 8-1 所示。

![在 Google Colab 中运行按钮](img/timl_0403.png)

###### 图 8-1。在 Google Colab 中运行按钮

###### 提示

截至目前，GitHub 存在一个错误，导致在显示 Jupyter 笔记本时出现间歇性错误消息。如果尝试访问笔记本时看到消息“抱歉，出了点问题。重新加载？”，请按照“构建我们的模型”中的说明操作。

本笔记本将指导我们完成训练模型的过程。它将按照以下步骤进行：

+   配置参数

+   安装正确的依赖项

+   使用称为 TensorBoard 的工具监视训练

+   运行训练脚本

+   将训练输出转换为我们可以使用的模型

### 启用 GPU 训练

在第四章中，我们在少量数据上训练了一个非常简单的模型。我们现在正在训练的模型要复杂得多，具有更大的数据集，并且需要更长时间来训练。在一台普通的现代计算机 CPU 上，训练它需要三到四个小时。

为了缩短训练模型所需的时间，我们可以使用一种称为*GPU 加速*的东西。*GPU*，即图形处理单元。它是一种旨在帮助计算机快速处理图像数据的硬件部件，使其能够流畅地渲染用户界面和视频游戏等内容。大多数计算机都有一个。

图像处理涉及并行运行许多任务，训练深度学习网络也是如此。这意味着可以使用 GPU 硬件加速深度学习训练。通常情况下，使用 GPU 运行训练比使用 CPU 快 5 到 10 倍是很常见的。

我们训练过程中需要的音频预处理意味着我们不会看到如此巨大的加速，但我们的模型在 GPU 上仍然会训练得更快 - 大约需要一到两个小时。

幸运的是，Colab 支持通过 GPU 进行训练。默认情况下未启用，但很容易打开。要这样做，请转到 Colab 的运行时菜单，然后单击“更改运行时类型”，如图 8-2 所示。

![在 Colab 中的“更改运行时类型”选项](img/timl_0802.png)

###### 图 8-2. 在 Colab 中的“更改运行时类型”选项

选择此选项后，将打开图 8-3 中显示的“笔记本设置”框。

![“笔记本设置”框](img/timl_0803.png)

###### 图 8-3. “笔记本设置”框

从“硬件加速器”下拉列表中选择 GPU，如图 8-4 所示，然后单击保存。

![在 Colab 中的“硬件加速器”下拉菜单](img/timl_0804.png)

###### 图 8-4. “硬件加速器”下拉列表

Colab 现在将在具有 GPU 的后端计算机（称为*运行时*）上运行其 Python。

下一步是配置笔记本，以包含我们想要训练的单词。

### 配置训练

训练脚本通过一系列命令行标志进行配置，这些标志控制从模型架构到将被训练分类的单词等所有内容。

为了更容易运行脚本，笔记本的第一个单元格将一些重要值存储在环境变量中。当运行这些脚本时，这些值将被替换为脚本的命令行标志。

第一个是`WANTED_WORDS`，允许我们选择要训练模型的单词：

```py
os.environ["WANTED_WORDS"] = "yes,no"
```

默认情况下，选定的单词是“yes”和“no”，但我们可以提供以下单词的任何组合，这些单词都出现在我们的数据集中：

+   常见命令：*yes*、*no*、*up*、*down*、*left*、*right*、*on*、*off*、*stop*、*go*、*backward*、*forward*、*follow*、*learn*

+   数字零到九：*zero*、*one*、*two*、*three*、*four*、*five*、*six*、*seven*、*eight*、*nine*

+   随机单词：*bed*、*bird*、*cat*、*dog*、*happy*、*house*、*Marvin*、*Sheila*、*tree*、*wow*

要选择单词，我们只需将它们包含在逗号分隔的列表中。让我们选择单词“on”和“off”来训练我们的新模型：

```py
os.environ["WANTED_WORDS"] = "on,off"
```

在训练模型时，未包含在列表中的任何单词将在模型训练时归为“未知”类别。

###### 注意

在这里选择超过两个单词是可以的；我们只需要稍微调整应用代码。我们提供了在“在我们的项目中使用模型”中执行此操作的说明。

还要注意`TRAINING_STEPS`和`LEARNING_RATE`变量：

```py
os.environ["TRAINING_STEPS"]="15000,3000"
os.environ["LEARNING_RATE"]="0.001,0.0001"
```

在第三章中，我们了解到模型的权重和偏差会逐渐调整，以便随着时间的推移，模型的输出越来越接近所期望的值。`TRAINING_STEPS`指的是训练数据批次通过网络运行的次数，以及其权重和偏差的更新次数。`LEARNING_RATE`设置调整速率。

使用高学习率，权重和偏差在每次迭代中调整更多，意味着收敛速度快。然而，这些大幅跳跃意味着更难以达到理想值，因为我们可能会一直跳过它们。使用较低的学习率，跳跃较小。需要更多步骤才能收敛，但最终结果可能更好。对于给定模型的最佳学习率是通过试错确定的。

在上述变量中，训练步骤和学习率被定义为逗号分隔的列表，定义了每个训练阶段的学习率。根据我们刚刚查看的值，模型将进行 15,000 步的训练，学习率为 0.001，然后进行 3,000 步的训练，学习率为 0.0001。总步数将为 18,000。

这意味着我们将使用高学习率进行一系列迭代，使网络快速收敛。然后我们将使用低学习率进行较少的迭代，微调权重和偏差。

现在，我们将保持这些值不变，但知道它们是什么是很好的。运行单元格。您将看到以下输出打印：

```py
Training these words: on,off
Training steps in each stage: 15000,3000
Learning rate in each stage: 0.001,0.0001
Total number of training steps: 18000
```

这提供了我们的模型将如何训练的摘要。

### 安装依赖项

接下来，我们获取一些运行脚本所必需的依赖项。

运行下面的两个单元格来执行以下操作：

+   安装包含训练所需操作的特定版本的 TensorFlow `pip`软件包。

+   克隆 TensorFlow GitHub 存储库的相应版本，以便我们可以访问训练脚本。

### 加载 TensorBoard

为了监视训练过程，我们使用[TensorBoard](https://oreil.ly/wginD)。这是一个用户界面，可以向我们显示图表、统计数据和其他关于训练进展的见解。

当训练完成时，它将看起来像图 8-5 中的截图。您将在本章后面了解所有这些图表的含义。

![TensorBoard 的截图](img/timl_0805.png)

###### 图 8-5。训练完成后的 TensorBoard 截图

运行下一个单元格以加载 TensorBoard。它将出现在 Colab 中，但在我们开始训练之前不会显示任何有趣的内容。

### 开始训练

以下单元格运行开始训练的脚本。您可以看到它有很多命令行参数：

```py
!python tensorflow/tensorflow/examples/speech_commands/train.py \
--model_architecture=tiny_conv --window_stride=20 --preprocess=micro \
--wanted_words=${WANTED_WORDS} --silence_percentage=25 --unknown_percentage=25 \
--quantize=1 --verbosity=WARN --how_many_training_steps=${TRAINING_STEPS} \
--learning_rate=${LEARNING_RATE} --summaries_dir=/content/retrain_logs \
--data_dir=/content/speech_dataset --train_dir=/content/speech_commands_train
```

其中一些，如`--wanted_words=${WANTED_WORDS}`，使用我们之前定义的环境变量来配置我们正在创建的模型。其他设置脚本的输出，例如`--train_dir=/content/speech_commands_train`，定义了训练模型将保存的位置。

保持参数不变，运行单元格。您将开始看到一些输出流过。在下载语音命令数据集时，它将暂停一段时间：

```py
>> Downloading speech_commands_v0.02.tar.gz 18.1%
```

完成后，会出现更多输出。可能会有一些警告，只要单元格继续运行，您可以忽略它们。此时，您应该向上滚动到 TensorBoard，希望它看起来像图 8-6。如果您看不到任何图表，请单击 SCALARS 选项卡。

![TensorBoard 的截图](img/timl_0806.png)

###### 图 8-6。训练开始时的 TensorBoard 截图

万岁！这意味着训练已经开始。您刚刚运行的单元将继续执行，训练将需要最多两个小时才能完成。该单元将不会输出更多日志，但有关训练运行的数据将出现在 TensorBoard 中。

您可以看到 TensorBoard 显示了两个图形，“准确度”和“交叉熵”，如图 8-7 所示。两个图形都显示了 x 轴上的当前步骤。“准确度”图显示了模型在 y 轴上的准确度，这表明它能够正确检测单词的时间有多少。“交叉熵”图显示了模型的损失，量化了模型预测与正确值之间的差距。

！[TensorBoard 中图形的屏幕截图]（Images/timl_0807.png）

###### 图 8-7。"准确度"和"交叉熵"图

###### 注意

交叉熵是衡量机器学习模型损失的常见方法，用于执行分类，目标是预测输入属于哪个类别。

图形上的锯齿线对应于训练数据集上的性能，而直线反映了验证数据集上的性能。验证定期进行，因此图上的验证数据点较少。

新数据将随着时间的推移出现在图形中，但要显示它，您需要调整它们的比例以适应。您可以通过单击每个图形下面的最右边的按钮来实现这一点，如图 8-8 所示。

！[TensorBoard 中图标的屏幕截图]（Images/timl_0808.png）

###### 图 8-8。单击此按钮以调整图形的比例，以适应所有可用数据

您还可以单击图 8-9 中显示的按钮，使每个图形变大。

！[TensorBoard 中图标的屏幕截图]（Images/timl_0809.png）

###### 图 8-9。单击此按钮以放大图形

除了图形外，TensorBoard 还可以显示输入传入模型。单击 IMAGES 选项卡，显示类似于图 8-10 的视图。这是在训练期间输入到模型中的频谱图的示例。

！[TensorBoard 中 IMAGES 选项卡的屏幕截图]（Images/timl_0810.png）

###### 图 8-10。TensorBoard 的 IMAGES 选项卡

### 等待训练完成

训练模型将需要一到两个小时，所以我们现在的工作是耐心等待。幸运的是，我们有 TensorBoard 漂亮的图形来娱乐我们。

随着训练的进行，您会注意到指标在一定范围内跳动。这是正常的，但它使图形看起来模糊且难以阅读。为了更容易看到训练的进展，我们可以使用 TensorFlow 的平滑功能。

图 8-11 显示了应用默认平滑度的图形；请注意它们有多模糊。

！[TensorBoard 中图形的屏幕截图]（Images/timl_0811.png）

###### 图 8-11。默认平滑度的训练图

通过调整图 8-12 中显示的平滑滑块，我们可以增加平滑度，使趋势更加明显。

！[TensorBoard 的*平滑*滑块的屏幕截图]（Images/timl_0812.png）

###### 图 8-12。TensorBoard 的平滑滑块

图 8-13 显示了具有更高平滑度级别的相同图形。原始数据以较浅的颜色可见，在下面。

！[TensorBoard 中图形的屏幕截图]（Images/timl_0813.png）

###### 图 8-13。增加平滑度的训练图

#### 保持 Colab 运行

为了防止废弃的项目占用资源，如果 Colab 没有被积极使用，它将关闭您的运行时。因为我们的训练需要一段时间，所以我们需要防止这种情况发生。我们需要考虑一些事情。

首先，如果我们没有在与 Colab 浏览器标签进行活动交互，Web 用户界面将与后端运行时断开连接，训练脚本正在执行的地方。几分钟后会发生这种情况，并且会导致您的 TensorBoard 图表停止更新最新的训练指标。如果发生这种情况，无需恐慌—您的训练仍在后台运行。

如果您的运行时已断开连接，您将在 Colab 的用户界面中看到一个重新连接按钮，如图 8-14 所示。点击此按钮以重新连接您的运行时。

![Colab 的重新连接按钮截图](img/timl_0814.png)

###### 图 8-14\. Colab 的重新连接按钮

断开连接的运行时并不是什么大问题，但 Colab 的下一个超时需要一些注意。*如果您在 90 分钟内不与 Colab 进行交互，您的运行时实例将被回收*。这是一个问题：您将丢失所有的训练进度，以及实例中存储的任何数据！

为了避免这种情况发生，您只需要每 90 分钟至少与 Colab 进行一次交互。打开标签页，确保运行时已连接，并查看您美丽的图表。只要在 90 分钟过去之前这样做，连接就会保持打开状态。

###### 警告

即使您关闭了 Colab 标签页，运行时也会在后台继续运行长达 90 分钟。只要在浏览器中打开原始 URL，您就可以重新连接到运行时，并继续之前的操作。

然而，当标签页关闭时，TensorBoard 将消失。如果在重新打开标签页时训练仍在进行，您将无法查看 TensorBoard，直到训练完成。

最后，*Colab 运行时的最长寿命为 12 小时*。如果您的训练时间超过 12 小时，那就倒霉了—Colab 将在训练完成之前关闭并重置您的实例。如果您的训练可能持续这么长时间，您应该避免使用 Colab，并使用“其他运行脚本的方法”中描述的替代方案之一。幸运的是，训练我们的唤醒词模型不会花费那么长时间。

当您的图表显示了 18000 步的数据时，训练就完成了！现在我们必须运行几个命令来准备我们的模型进行部署。不用担心—这部分要快得多。

### 冻结图表

正如您在本书中早些时候学到的，训练是一个迭代调整模型权重和偏差的过程，直到它产生有用的预测。训练脚本将这些权重和偏差写入*检查点*文件。每一百步写入一个检查点。这意味着如果训练在中途失败，可以从最近的检查点重新启动而不会丢失进度。

*train.py*脚本被调用时带有一个参数，`--train_dir`，用于指定这些检查点文件将被写入的位置。在我们的 Colab 中，它被设置为*/content/speech_commands_train*。

您可以通过打开 Colab 的左侧面板来查看检查点文件，该面板具有一个文件浏览器。要这样做，请点击图 8-15 中显示的按钮。

![打开 Colab 侧边栏的按钮截图](img/timl_0815.png)

###### 图 8-15\. 打开 Colab 侧边栏的按钮

在此面板中，点击“文件”选项卡以查看运行时的文件系统。如果您打开*speech_commands_train/*目录，您将看到检查点文件，如图 8-16 所示。每个文件名中的数字表示保存检查点的步骤。

![Colab 的文件浏览器显示检查点文件列表的截图](img/timl_0816.png)

###### 图 8-16\. Colab 的文件浏览器显示检查点文件列表

一个 TensorFlow 模型由两个主要部分组成：

+   训练产生的权重和偏差

+   将模型的输入与这些权重和偏差结合起来产生模型的输出的操作图

此时，我们的模型操作在 Python 脚本中定义，并且其训练的权重和偏差在最新的检查点文件中。我们需要将这两者合并为一个具有特定格式的单个模型文件，以便我们可以用来运行推断。创建此模型文件的过程称为*冻结*——我们正在创建一个具有*冻结*权重的图的静态表示。

为了冻结我们的模型，我们运行一个脚本。您将在下一个单元格中找到它，在“冻结图”部分。脚本的调用如下：

```py
!python tensorflow/tensorflow/examples/speech_commands/freeze.py \
  --model_architecture=tiny_conv --window_stride=20 --preprocess=micro \
  --wanted_words=${WANTED_WORDS} --quantize=1 \
  --output_file=/content/tiny_conv.pb \
  --start_checkpoint=/content/speech_commands_train/tiny_conv. \
  ckpt-${TOTAL_STEPS}
```

为了指向正确的操作图以冻结的脚本，我们传递了一些与训练中使用的相同参数。我们还传递了最终检查点文件的路径，该文件的文件名以训练步骤的总数结尾。

运行此单元格以冻结图。冻结的图将输出到名为*tiny_conv.pb*的文件中。

这个文件是完全训练过的 TensorFlow 模型。它可以被 TensorFlow 加载并用于运行推断。这很棒，但它仍然是常规 TensorFlow 使用的格式，而不是 TensorFlow Lite。我们的下一步是将模型转换为 TensorFlow Lite 格式。

### 转换为 TensorFlow Lite

转换是另一个简单的步骤：我们只需要运行一个命令。现在我们有一个冻结的图文件可以使用，我们将使用`toco`，TensorFlow Lite 转换器的命令行界面。

在“转换模型”部分，运行第一个单元格：

```py
!toco
  --graph_def_file=/content/tiny_conv.pb --output_file= \
  /content/tiny_conv.tflite \
  --input_shapes=1,49,40,1 --input_arrays=Reshape_2
  --output_arrays='labels_softmax' \
  --inference_type=QUANTIZED_UINT8 --mean_values=0 --std_dev_values=9.8077
```

在参数中，我们指定要转换的模型，TensorFlow Lite 模型文件的输出位置，以及一些取决于模型架构的其他值。因为模型在训练期间被量化，我们还提供了一些参数（`inference_type`，`mean_values`和`std_dev_values`），指导转换器如何将其低精度值映射到实数。

您可能想知道为什么`input_shape`参数在宽度、高度和通道参数之前有一个前导`1`。这是批处理大小；为了在训练期间提高效率，我们一次发送很多输入，但当我们在实时应用中运行时，我们每次只处理一个样本，这就是为什么批处理大小固定为`1`。

转换后的模型将被写入*tiny_conv.tflite*。恭喜！这是一个完全成型的 TensorFlow Lite 模型！

查看这个模型有多小，在下一个单元格中运行以下代码：

```py
import os
model_size = os.path.getsize("/content/tiny_conv.tflite")
print("Model is %d bytes" % model_size)
```

输出显示模型非常小：`模型大小为 18208 字节`。

我们的下一步是将这个模型转换为可以部署到微控制器的形式。

### 创建一个 C 数组

回到“转换为 C 文件”中，我们使用`xxd`命令将 TensorFlow Lite 模型转换为 C 数组。我们将在下一个单元格中做同样的事情：

```py
# Install xxd if it is not available
!apt-get -qq install xxd
# Save the file as a C source file
!xxd -i /content/tiny_conv.tflite > /content/tiny_conv.cc
# Print the source file
!cat /content/tiny_conv.cc
```

输出的最后部分将是文件的内容，其中包括一个 C 数组和一个保存其长度的整数，如下所示（您看到的确切值可能略有不同）：

```py
unsigned char _content_tiny_conv_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00,
  // ...
  0x00, 0x09, 0x06, 0x00, 0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x04
};
unsigned int _content_tiny_conv_tflite_len = 18208;
```

这段代码也被写入一个文件*tiny_conv.cc*，您可以使用 Colab 的文件浏览器下载。因为您的 Colab 运行时将在 12 小时后到期，现在将此文件下载到您的计算机是一个好主意。

接下来，我们将把这个新训练过的模型与`micro_speech`项目集成起来，以便我们可以将其部署到一些硬件上。

# 在我们的项目中使用模型

要使用我们的新模型，我们需要做三件事：

1.  在[*micro_features/tiny_conv_micro_features_model_data.cc*](https://oreil.ly/EAR0U)中，用我们的新模型替换原始模型数据。

1.  在[*micro_features/micro_model_settings.cc*](https://oreil.ly/bqw67)中用我们的新“on”和“off”标签更新标签名称。

1.  更新特定设备的*command_responder.cc*以执行我们对新标签的操作。

## 替换模型

要替换模型，请在文本编辑器中打开*micro_features/tiny_conv_micro_features_model_data.cc*。

###### 注意

如果你正在使用 Arduino 示例，该文件将显示为 Arduino IDE 中的一个选项卡。它的名称将是*micro_features_tiny_conv_micro_features_model_data.cpp*。如果你正在使用 SparkFun Edge，你可以直接在本地的 TensorFlow 存储库副本中编辑文件。如果你正在使用 STM32F746G，你应该在 Mbed 项目目录中编辑文件。

*tiny_conv_micro_features_model_data.cc*文件包含一个看起来像这样的数组声明：

```py
const unsigned char
    g_tiny_conv_micro_features_model_data[] DATA_ALIGN_ATTRIBUTE = {
        0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
        0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00,
        //...
        0x00, 0x09, 0x06, 0x00, 0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x04};
const int g_tiny_conv_micro_features_model_data_len = 18208;
```

需要替换数组的内容以及常量`g_tiny_conv_micro_features_model_data_len`的值，如果已经更改。

为此，打开你在上一节末尾下载的*tiny_conv.cc*文件。复制并粘贴数组的内容，但不包括定义，到*tiny_conv_micro_features_model_data.cc*中定义的数组中。确保你正在覆盖数组的内容，但不是它的声明。

在*tiny_conv.cc*的底部，你会找到`_content_tiny_conv_tflite_len`，一个变量，其值表示数组的长度。回到*tiny_conv_micro_features_model_data.cc*，用这个变量的值替换`g_tiny_conv_micro_features_model_data_len`的值。然后保存文件；你已经完成了更新。

## 更新标签

接下来，打开*micro_features/micro_model_settings.cc*。这个文件包含一个类标签的数组：

```py
const char* kCategoryLabels[kCategoryCount] = {
    "silence",
    "unknown",
    "yes",
    "no",
};
```

为了调整我们的新模型，我们可以简单地将“yes”和“no”交换为“on”和“off”。我们按顺序将标签与模型的输出张量元素匹配，因此重要的是按照它们提供给训练脚本的顺序列出这些标签。

以下是预期的代码：

```py
const char* kCategoryLabels[kCategoryCount] = {
    "silence",
    "unknown",
    "on",
    "off",
};
```

如果你训练了一个具有两个以上标签的模型，只需将它们全部添加到列表中。

我们现在已经完成了切换模型的工作。唯一剩下的步骤是更新使用标签的任何输出代码。

## 更新 command_responder.cc

该项目包含针对 Arduino、SparkFun Edge 和 STM32F746G 的不同设备特定实现的*command_responder.cc*。我们将在以下部分展示如何更新每个设备。

### Arduino

位于*arduino/command_responder.cc*中的 Arduino 命令响应器在听到“yes”时会点亮 LED 3 秒钟。让我们将其更新为在听到“on”或“off”时点亮 LED。在文件中，找到以下`if`语句：

```py
// If we heard a "yes", switch on an LED and store the time.
if (found_command[0] == 'y') {
  last_yes_time = current_time;
  digitalWrite(LED_BUILTIN, HIGH);
}
```

`if`语句测试命令的第一个字母是否为“y”，表示“yes”。如果我们将这个“y”改为“o”，LED 将点亮“on”或“off”，因为它们都以“o”开头：

```py
if (found_command[0] == 'o') {
  last_yes_time = current_time;
  digitalWrite(LED_BUILTIN, HIGH);
}
```

完成这些代码更改后，部署到你的设备并尝试一下。

### SparkFun Edge

位于*sparkfun_edge/command_responder.cc*中的 SparkFun Edge 命令响应器会根据听到的“yes”或“no”点亮不同的 LED。在文件中，找到以下`if`语句：

```py
if (found_command[0] == 'y') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_YELLOW);
}
if (found_command[0] == 'n') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_RED);
}
if (found_command[0] == 'u') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_GREEN);
}
```

很容易更新这些，使得“on”和“off”分别点亮不同的 LED：

```py
if (found_command[0] == 'o' && found_command[1] == 'n') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_YELLOW);
}
if (found_command[0] == 'o' && found_command[1] == 'f') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_RED);
}
if (found_command[0] == 'u') {
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_GREEN);
}
```

因为这两个命令都以相同的字母开头，我们需要查看它们的第二个字母来消除歧义。现在，当说“on”时，黄色 LED 将点亮，当说“off”时，红色 LED 将点亮。

完成更改后，部署并运行代码，使用与“运行示例”中遵循的相同过程。

### STM32F746G

位于*disco_f746ng/command_responder.cc*中的 STM32F746G 命令响应器会根据听到的命令显示不同的单词。在文件中，找到以下`if`语句：

```py
if (*found_command == 'y') {
  lcd.Clear(0xFF0F9D58);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard yes!", CENTER_MODE);
} else if (*found_command == 'n') {
  lcd.Clear(0xFFDB4437);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard no :(", CENTER_MODE);
} else if (*found_command == 'u') {
  lcd.Clear(0xFFF4B400);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard unknown", CENTER_MODE);
} else {
  lcd.Clear(0xFF4285F4);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard silence", CENTER_MODE);
}
```

很容易更新以便响应“on”和“off”：

```py
if (found_command[0] == 'o' && found_command[1] == 'n') {
  lcd.Clear(0xFF0F9D58);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard on!", CENTER_MODE);
} else if (found_command[0] == 'o' && found_command[1] == 'f') {
  lcd.Clear(0xFFDB4437);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard off", CENTER_MODE);
} else if (*found_command == 'u') {
  lcd.Clear(0xFFF4B400);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard unknown", CENTER_MODE);
} else {
  lcd.Clear(0xFF4285F4);
  lcd.DisplayStringAt(0, LINE(5), (uint8_t *)"Heard silence", CENTER_MODE);
}
```

同样，因为这两个命令都以相同的字母开头，我们需要查看它们的第二个字母来消除歧义。现在我们为每个命令显示适当的文本。

## 运行脚本的其他方法

如果你无法使用 Colab，有两种其他推荐的训练模型的方法：

+   在一个带有 GPU 的云虚拟机（VM）中

+   在你的本地工作站上

进行基于 GPU 的训练所需的驱动程序仅在 Linux 上可用。没有 Linux，训练将需要大约四个小时。因此，建议使用带有 GPU 的云虚拟机或类似配置的 Linux 工作站。

设置您的虚拟机或工作站超出了本书的范围。但是，我们有一些建议。如果您使用虚拟机，可以启动一个[Google Cloud 深度学习虚拟机镜像](https://oreil.ly/PVRtP)，该镜像预先配置了所有您进行 GPU 训练所需的依赖项。如果您使用 Linux 工作站，[TensorFlow GPU Docker 镜像](https://oreil.ly/PFYVr)包含了您所需的一切。

要训练模型，您需要安装 TensorFlow 的夜间版本。要卸载任何现有版本并替换为已确认可用的版本，请使用以下命令：

```py
pip uninstall -y tensorflow tensorflow_estimator
pip install -q tf-estimator-nightly==1.14.0.dev2019072901 \
  tf-nightly-gpu==1.15.0.dev20190729
```

接下来，打开命令行并切换到用于存储代码的目录。使用以下命令克隆 TensorFlow 并打开一个已确认可用的特定提交：

```py
git clone -q https://github.com/tensorflow/tensorflow
git -c advice.detachedHead=false -C tensorflow checkout 17ce384df70
```

现在您可以运行*train.py*脚本来训练模型。这将训练一个能识别“是”和“不”的模型，并将检查点文件输出到*/tmp*：

```py
python tensorflow/tensorflow/examples/speech_commands/train.py \
  --model_architecture=tiny_conv --window_stride=20 --preprocess=micro \
  --wanted_words="on,off" --silence_percentage=25 --unknown_percentage=25 \
  --quantize=1 --verbosity=INFO --how_many_training_steps="15000,3000" \
  --learning_rate="0.001,0.0001" --summaries_dir=/tmp/retrain_logs \
  --data_dir=/tmp/speech_dataset --train_dir=/tmp/speech_commands_train
```

训练后，运行以下脚本来冻结模型：

```py
python tensorflow/tensorflow/examples/speech_commands/freeze.py \
  --model_architecture=tiny_conv --window_stride=20 --preprocess=micro \
  --wanted_words="on,off" --quantize=1 --output_file=/tmp/tiny_conv.pb \
  --start_checkpoint=/tmp/speech_commands_train/tiny_conv.ckpt-18000
```

接下来，将模型转换为 TensorFlow Lite 格式：

```py
toco
  --graph_def_file=/tmp/tiny_conv.pb --output_file=/tmp/tiny_conv.tflite \
  --input_shapes=1,49,40,1 --input_arrays=Reshape_2 \
  --output_arrays='labels_softmax' \
  --inference_type=QUANTIZED_UINT8 --mean_values=0 --std_dev_values=9.8077
```

最后，将文件转换为 C 源文件，以便编译到嵌入式系统中：

```py
xxd -i /tmp/tiny_conv.tflite > /tmp/tiny_conv_micro_features_model_data.cc
```

# 模型的工作原理

现在您知道如何训练自己的模型了，让我们探讨一下它是如何工作的。到目前为止，我们将机器学习模型视为黑匣子——我们将训练数据输入其中，最终它会找出如何预测结果。要使用模型并不一定要理解底层发生了什么，但这对于调试问题可能有帮助，而且本身也很有趣。本节将为您提供一些关于模型如何进行预测的见解。

## 可视化输入

图 8-17 说明了实际输入神经网络的内容。这是一个具有单个通道的 2D 数组，因此我们可以将其可视化为单色图像。我们使用 16 KHz 音频样本数据，那么我们如何从源数据得到这种表示？这个过程是机器学习中所谓的“特征生成”的一个示例，目标是将更难处理的输入格式（在本例中是代表一秒音频的 16,000 个数值）转换为机器学习模型更容易理解的内容。如果您之前研究过深度学习的机器视觉用例，您可能没有遇到这种情况，因为图像通常相对容易让网络接受而无需太多预处理；但在许多其他领域，如音频和自然语言处理，仍然常见在将输入馈入模型之前对其进行转换。

![TensorBoard 中 IMAGES 选项卡的屏幕截图](img/timl_0810.png)

###### 图 8-17。TensorBoard 的 IMAGES 选项卡

为了对我们的模型为什么更容易处理预处理输入有直觉，让我们看一下一些音频录音的原始表示，如图 8-18 到 8-21 所示。

![一个人说“是”的 16,000 个音频样本的可视化。](img/timl_0818.png)

###### 图 8-18。一个人说“是”的音频录音的波形

![一个人说“不”的 16,000 个音频样本的可视化。](img/timl_0819.png)

###### 图 8-19。一个人说“不”的音频录音的波形

![一个人说“是”的 16,000 个音频样本的可视化。](img/timl_0820.png)

###### 图 8-20。一个人说“是”的音频录音的另一个波形

![一个人说“不”的 16,000 个音频样本的可视化。](img/timl_0821.png)

###### 图 8-21。一个人说“不”的音频录音的另一个波形

如果没有标签，你会很难区分哪些波形对应相同的单词。现在看看图 8-22 到 8-25，展示了将相同的一秒录音通过特征生成处理后的结果。

![一个人说“是”时的 16,000 个音频样本的可视化。](img/timl_0822.png)

###### 图 8-22。一个人说“是”时的谱图

![一个人说“否”时的 16,000 个音频样本的可视化。](img/timl_0823.png)

###### 图 8-23。一个人说“否”时的谱图

![一个人说“是”时的 16,000 个音频样本的可视化。](img/timl_0824.png)

###### 图 8-24。一个人说“是”时的另一个谱图

![一个人说“否”时的 16,000 个音频样本的可视化。](img/timl_0825.png)

###### 图 8-25。一个人说“否”时的另一个谱图

这些仍然不容易解释，但希望你能看出“是”谱图的形状有点像倒置的 L，而“否”特征显示出不同的形状。我们可以更容易地辨别谱图之间的差异，希望直觉告诉你，对于模型来说做同样的事情更容易。

另一个方面是生成的谱图比样本数据要小得多。每个谱图由 1,960 个数值组成，而波形有 16,000 个。它们是音频数据的摘要，减少了神经网络必须进行的工作量。事实上，一个专门设计的模型，比如[DeepMind 的 WaveNet](https://oreil.ly/IH9J3)，可以将原始样本数据作为输入，但结果模型往往涉及比我们使用的神经网络加手工设计特征组合更多的计算，因此对于资源受限的环境，如嵌入式系统，我们更喜欢这里使用的方法。

## 特征生成是如何工作的？

如果你有处理音频的经验，你可能熟悉像[梅尔频率倒谱系数（MFCCs）](https://oreil.ly/HTAev)这样的方法。这是一种常见的生成我们正在使用的谱图的方法，但我们的示例实际上使用了一种相关但不同的方法。这是谷歌在生产中使用的相同方法，这意味着它已经得到了很多实际验证，但它还没有在研究文献中发表。在这里，我们大致描述了它的工作原理，但对于详细信息，最好的参考是[代码本身](https://oreil.ly/NeOnW)。

该过程开始通过为给定时间片段生成傅立叶变换（也称为快速傅立叶变换或 FFT）-在我们的情况下是 30 毫秒的音频数据。这个 FFT 是在使用[汉宁窗口](https://oreil.ly/jhn8c)过滤的数据上生成的，汉宁窗口是一个钟形函数，减少了 30 毫秒窗口两端样本的影响。傅立叶变换为每个频率产生具有实部和虚部的复数，但我们只关心总能量，因此我们对两个分量的平方求和，然后应用平方根以获得每个频率桶的幅度。

给定*N*个样本，傅立叶变换提供*N*/2 个频率的信息。以每秒 16,000 个样本的速率的 30 毫秒需要 480 个样本，因为我们的 FFT 算法需要二的幂输入，所以我们用零填充到 512 个样本，给我们 256 个频率桶。这比我们需要的要大，因此为了缩小它，我们将相邻频率平均到 40 个降采样桶中。然而，这种降采样不是线性的；相反，它使用基于人类感知的梅尔频率刻度，以便更多地为低频率分配权重，从而为它们提供更多的桶，而高频率则合并到更广泛的桶中。图 8-26 展示了该过程的图表。

![特征生成过程的图表。](img/timl_0826.png)

###### 图 8-26。特征生成过程的图表

这个特征生成器的一个不寻常之处是它包含了一个降噪步骤。这通过保持每个频率桶中的值的运行平均值，然后从当前值中减去这个平均值来实现。其思想是背景噪音随时间保持相对恒定，并显示在特定频率上。通过减去运行平均值，我们有很大机会去除一些噪音的影响，保留我们感兴趣的更快变化的语音。棘手的部分是特征生成器确实保留状态以跟踪每个桶的运行平均值，因此如果您尝试为给定输入重现相同的频谱图输出——就像我们尝试的那样[进行测试](https://oreil.ly/HtPve)——您将需要将该状态重置为正确的值。

噪音降低的另一个部分最初让我们感到惊讶的是它对奇数和偶数频率桶使用不同系数。这导致了您可以在最终生成的特征图像中看到的独特的梳齿图案（图 8-22 至 8-25）。最初我们以为这是一个错误，但在与原始实施者交谈后，我们了解到这实际上是有意为之，以帮助性能。在[Yuxuan Wang 等人的“用于强健和远场关键词检测的可训练前端”](https://oreil.ly/QZ4Yb)的第 4.3 节中对这种方法进行了详细讨论，该论文还包括了进入此特征生成流程的其他设计决策的背景。我们还通过我们的模型进行了实证测试，去除奇数和偶数桶处理差异确实会显着降低评估的准确性。

然后我们使用每通道幅度归一化（PCAN）自动增益，根据运行平均噪音来增强信号。最后，我们对所有桶值应用对数尺度，以便相对较大的频率不会淹没频谱中较安静的部分——这种归一化有助于后续模型处理这些特征。

这个过程总共重复了 49 次，每次之间以 30 毫秒的窗口向前移动 20 毫秒，以覆盖完整的一秒音频输入数据。这产生了一个 40 个元素宽（每个频率桶一个）和 49 行高（每个时间片一个）的值的 2D 数组。

如果这一切听起来很复杂，不用担心。因为实现它的代码都是开源的，您可以在自己的音频项目中重用它。

## 理解模型架构

我们正在使用的神经网络模型被定义为一组操作的小图。您可以在[`create_tiny_conv_model()`函数](https://oreil.ly/fMARv)中找到定义它的代码，并且图 8-27 展示了结果的可视化。

该模型由一个卷积层、一个全连接层和最后的 softmax 层组成。在图中，卷积层标记为“DepthwiseConv2D”，但这只是 TensorFlow Lite 转换器的一个怪癖（事实证明，具有单通道输入图像的卷积层也可以表示为深度卷积）。您还会看到一个标记为“Reshape_1”的层，但这只是一个输入占位符，而不是一个真正的操作。

![将语音识别模型可视化为图形。](img/timl_0827.png)

###### 图 8-27。语音识别模型的图形可视化，由[Netron 工具](https://oreil.ly/UiuXU)提供

卷积层用于在输入图像中发现 2D 模式。每个滤波器是一个值的矩形数组，它作为一个滑动窗口在输入上移动，输出图像表示输入和滤波器在每个点匹配程度。您可以将卷积操作视为在图像上移动一系列矩形滤波器，每个滤波器在每个像素处的结果对应于滤波器与图像中该补丁的相似程度。在我们的情况下，每个滤波器宽 8 像素，高 10 像素，总共有 8 个。图 8-28 到 8-35 显示它们的外观。

![卷积滤波器的可视化。](img/timl_0828.png)

###### 图 8-28。第一个滤波器图像

![卷积滤波器的可视化。](img/timl_0829.png)

###### 图 8-29。第二个滤波器图像

![卷积滤波器的可视化。](img/timl_0830.png)

###### 图 8-30。第三个滤波器图像

![卷积滤波器的可视化。](img/timl_0831.png)

###### 图 8-31。第四个滤波器图像

![卷积滤波器的可视化。](img/timl_0832.png)

###### 图 8-32。第五个滤波器图像

![卷积滤波器的可视化。](img/timl_0833.png)

###### 图 8-33。第六个滤波器图像

![卷积滤波器的可视化。](img/timl_0834.png)

###### 图 8-34。第七个滤波器图像

![卷积滤波器的可视化。](img/timl_0835.png)

###### 图 8-35。第八个滤波器图像

您可以将这些滤波器中的每一个视为输入图像的一个小补丁。该操作试图将此小补丁与看起来相似的输入图像部分进行匹配。当图像与补丁相似时，高值将被写入输出图像的相应部分。直观地说，每个滤波器都是模型已经学会在训练输入中寻找的模式，以帮助它区分不同类别。

因为我们有八个滤波器，所以将有八个不同的输出图像，每个对应于相应滤波器的匹配值，当它在输入上滑动时。这些滤波器输出实际上被合并为一个具有八个通道的单个输出图像。我们已将步幅设置为两个方向，这意味着每次我们将每个滤波器向前滑动两个像素，而不仅仅是一个像素。因为我们跳过每个其他位置，这意味着我们的输出图像是输入大小的一半。

您可以看到在可视化中，输入图像高 49 像素，宽 40 像素，具有单个通道，这是我们在前一节中讨论的特征频谱图所期望的。因为我们在水平和垂直方向上滑动卷积滤波器时跳过每个其他像素，所以卷积的输出是一半大小，即高 25 像素，宽 20 像素。然而有八个滤波器，所以图像变为八个通道深。

下一个操作是全连接层。这是一种不同的模式匹配过程。与在输入上滑动一个小窗口不同，这里为输入张量中的每个值都有一个权重。结果是指示输入与权重匹配程度的指标，在比较每个值之后。您可以将其视为全局模式匹配，其中您有一个理想的结果，您期望作为输入获得，输出是理想值（保存在权重中）与实际输入之间的接近程度。我们模型中的每个类都有自己的权重，因此“静音”，“未知”，“是”和“否”都有一个理想模式，并生成四个输出值。输入中有 4,000 个值`(25 * 20 * 8)`，因此每个类由 4,000 个权重表示。

最后一层是一个 softmax 层。这有效地增加了最高输出和其最近竞争对手之间的差异，这不会改变它们的相对顺序（从全连接层产生最大值的类仍将保持最高），但有助于产生一个更有用的分数。这个分数通常非正式地被称为“概率”，但严格来说，如果没有更多关于输入数据实际混合的校准，你不能可靠地像那样使用它。例如，如果检测器中有更多的单词，那么像“反对建立教会主义”这样的不常见单词可能不太可能出现，而像“好的”这样的单词可能更有可能出现，但根据训练数据的分布，这可能不会反映在原始分数中。

除了这些主要层外，还有偏差被添加到全连接和卷积层的结果中，以帮助调整它们的输出，并在每个之后使用修正线性单元（ReLU）激活函数。ReLU 只是确保没有输出小于零，将任何负结果设置为零的最小值。这种类型的激活函数是使深度学习变得更加有效的突破之一：它帮助训练过程比网络本来会更快地收敛。

## 理解模型输出

模型的最终结果是 softmax 层的输出。这是四个数字，分别对应“沉默”，“未知”，“是”和“否”。这些值是每个类别的分数，具有最高分数的类别是模型的预测，分数代表模型对其预测的信心。例如，如果模型输出是`[10, 4, 231, 80]`，它预测第三个类别“是”是最可能的结果，得分为 231。 （我们以它们的量化形式给出这些值，介于 0 和 255 之间，但因为这些只是相对分数，通常不需要将它们转换回它们的实值等价物。）

有一件棘手的事情是，这个结果是基于分析音频的最后一秒。如果我们每秒只运行一次，可能会得到一个话语，一半在上一秒，一半在当前秒。当模型只听到部分单词时，任何模型都不可能很好地识别单词，因此在这种情况下，单词识别会失败。为了克服这个问题，我们需要比每秒运行模型更频繁，以尽可能高的概率在我们的一秒窗口内捕捉到整个单词。实际上，我们发现我们必须每秒运行 10 到 15 次才能取得良好的结果。

如果我们得到这些结果如此迅速，我们如何决定何时得分足够高？我们实现了一个后处理类，它会随着时间平均分数，并仅在短时间内同一个单词的得分高时触发识别。您可以在[`RecognizeCommands 类`](https://oreil.ly/FuYfL)中看到这个实现。这个类接收模型的原始结果，然后使用累积和平均算法来确定是否有任何类别已经超过了阈值。然后将这些后处理结果传递给[`CommandResponder`](https://oreil.ly/b8ArK)以根据平台的输出能力采取行动。

模型参数都是从训练数据中学习的，但命令识别器使用的算法是手动创建的，所以所有的[阈值](https://oreil.ly/tfNfr)——比如触发识别所需的得分值，或者需要的正结果时间窗口——都是手动选择的。这意味着不能保证它们是最佳的，所以如果在您自己的应用中看到不佳的结果，您可能希望尝试自己调整它们。

更复杂的语音识别模型通常使用能够接收流数据的模型（如递归神经网络），而不是我们在本章中展示的单层卷积网络。将流式处理嵌入到模型设计中意味着您无需进行后处理即可获得准确的结果，尽管这确实使训练变得更加复杂。

# 使用您自己的数据进行训练

您要构建的产品很可能不仅需要回答“是”和“否”，因此您需要训练一个对您关心的音频敏感的模型。我们之前使用的训练脚本旨在让您使用自己的数据创建自定义模型。这个过程中最困难的部分通常是收集足够大的数据集，并确保它适用于您的问题。我们在第十六章中讨论了数据收集和清理的一般方法，但本节涵盖了一些您可以训练自己的音频模型的方法。

## 语音命令数据集

*train.py*脚本默认下载了 Speech Commands 数据集。这是一个开源集合，包含超过 10 万个一秒钟的 WAV 文件，涵盖了许多不同说话者的各种短单词。它由 Google 分发，但话语是从世界各地的志愿者那里收集的。Aakanksha Chowdhery 等人的[“Visual Wake Words Dataset”](https://oreil.ly/EC6nd)提供了更多细节。

除了“是”和“否”之外，数据集还包括另外八个命令词（“打开”，“关闭”，“上”，“下”，“左”，“右”，“停止”和“前进”），以及从“零”到“九”的十个数字。每个单词都有几千个示例。还有其他单词，比如“Marvin”，每个单词的示例要少得多。命令词旨在有足够的话语，以便您可以训练一个合理的模型来识别它们。其他单词旨在用于填充“未知”类别，因此模型可以发现当发出未经训练的单词时，而不是将其误认为是一个命令。

由于训练脚本使用了这个数据集，您可以轻松地训练一个模型，结合一些有很多示例的命令词。如果您使用训练集中存在的单词的逗号分隔列表更新`--wanted_words`参数，并从头开始运行训练，您应该会发现您可以创建一个有用的模型。需要注意的主要事项是，您要限制自己只使用这 10 个命令词和/或数字，否则您将没有足够的示例进行准确训练，并且如果您有超过两个想要的单词，则需要将`--silence_percentage`和`--unknown_percentage`值调低。这两个参数控制训练过程中混合了多少无声和未知样本。*无声*示例实际上并不是完全的沉默；相反，它们是从数据集的*background*文件夹中的 WAV 文件中随机选择的一秒钟的录制背景噪音片段。*未知*样本是从训练集中的任何单词中挑选出来的话语，但不在`wanted_words`列表中。这就是为什么数据集中有一些杂项单词，每个单词的话语相对较少；这让我们有机会认识到很多不同的单词实际上并不是我们正在寻找的单词。这在语音和音频识别中是一个特别的问题，因为我们的产品通常需要在可能从未在训练中遇到的环境中运行。仅在常见英语中就可能出现成千上万个不同的单词，为了有用，模型必须能够忽略那些它没有经过训练的单词。这就是为什么*未知*类别在实践中如此重要。

以下是使用现有数据集训练不同单词的示例：

```py
python tensorflow/examples/speech_commands/train.py \
  --model_architecture=tiny_conv --window_stride=20 --preprocess=micro \
  --wanted_words="up,down,left,right" --silence_percentage=15 \
  --unknown_percentage=15 --quantize=1
```

## 在您自己的数据集上训练

训练脚本的默认设置是使用 Speech Commands，但如果您有自己的数据集，可以使用`--data_dir`参数来使用它。您指向的目录应该像 Speech Commands 一样组织，每个包含一组 WAV 文件的类别都有一个子文件夹。您还需要一个特殊的*background*子文件夹，其中包含您的应用程序预计会遇到的背景噪音类型的较长的 WAV 录音。如果默认的一秒持续时间对您的用例不起作用，您还需要选择一个识别持续时间，并通过`--sample_duration_ms`参数指定。然后，您可以使用`--wanted_words`参数设置要识别的类别。尽管名称如此，这些类别可以是任何类型的音频事件，从玻璃破碎到笑声；只要您有足够的每个类别的 WAV 文件，训练过程应该与语音一样有效。

如果您在根目录*/tmp/my_wavs*中有名为*glass*和*laughter*的 WAV 文件夹，这是如何训练您自己的模型的：

```py
python tensorflow/examples/speech_commands/train.py \
  --model_architecture=tiny_conv --window_stride=20 --preprocess=micro \
  --data_url="" --data_dir=/tmp/my_wavs/ --wanted_words="laughter,glass" \
  --silence_percentage=25 --unknown_percentage=25 --quantize=1
```

通常最困难的部分是找到足够的数据。例如，事实证明，真实的玻璃破碎声与我们在电影中听到的声音效果非常不同。这意味着你需要找到现有的录音，或者安排自己录制一些。由于训练过程可能需要每个类别的成千上万个示例，并且它们需要涵盖在真实应用中可能发生的所有变化，这个数据收集过程可能令人沮丧、昂贵且耗时。

对于图像模型，一个常见的解决方案是使用*迁移学习*，即使用已经在大型公共数据集上训练过的模型，并使用其他数据对不同类别进行微调。这种方法在次要数据集中不需要像从头开始训练那样多的示例，而且通常会产生高准确度的结果。不幸的是，语音模型的迁移学习仍在研究中，但请继续关注。

## 如何录制您自己的音频

如果您需要捕捉您关心的单词的音频，如果您有一个提示说话者并将结果拆分为标记文件的工具，那将会更容易。Speech Commands 数据集是使用[Open Speech Recording app](https://oreil.ly/UWsG3)录制的，这是一个托管应用程序，允许用户通过大多数常见的网络浏览器录制话语。作为用户，您将看到一个网页，首先要求您同意被录制，带有默认的谷歌协议，这是[可以轻松更改的](https://oreil.ly/z5vka)。同意后，您将被发送到一个具有录音控件的新页面。当您按下录制按钮时，单词将作为提示出现，您说的每个单词的音频将被记录。当所有请求的单词都被记录时，您将被要求将结果提交到服务器。

README 中有在 Google Cloud 上运行的说明，但这是一个用 Python 编写的 Flask 应用程序，因此您应该能够将其移植到其他环境中。如果您使用 Google Cloud，您需要更新[*app.yaml*](https://oreil.ly/dV2kv)文件，指向您自己的存储桶，并提供您自己的随机会话密钥（这仅用于哈希，因此可以是任何值）。要自定义记录的单词，您需要编辑[客户端 JavaScript](https://oreil.ly/XcJIe)中的一些数组：一个用于频繁重复的主要单词，一个用于次要填充词。

记录的文件以 OGG 压缩音频的形式存储在 Google Cloud 存储桶中，但训练需要 WAV 文件，因此您需要将它们转换。而且很可能您的一些录音包含错误，比如人们忘记说单词或说得太轻，因此在可能的情况下自动过滤出这些错误是有帮助的。如果您已经在`BUCKET_NAME`变量中设置了您的存储桶名称，您可以通过使用以下 bash 命令将文件复制到本地机器开始：

```py
mkdir oggs
gsutil -m cp gs://${BUCKET_NAME}/* oggs/
```

压缩的 OGG 格式的一个好处是安静或无声的音频会生成非常小的文件，因此一个很好的第一步是删除那些特别小的文件，比如：

```py
find ${BASEDIR}/oggs -iname "*.ogg" -size -5k -delete
```

我们发现将 OGG 转换为 WAV 的最简单方法是使用[FFmpeg 项目](https://ffmpeg.org/)，它提供了一个命令行工具。以下是一组命令，可以将一个目录中的所有 OGG 文件转换为我们需要的格式：

```py
mkdir -p ${BASEDIR}/wavs
find ${BASEDIR}/oggs -iname "*.ogg" -print0 | \
  xargs -0 basename -s .ogg | \
  xargs -I {} ffmpeg -i ${BASEDIR}/oggs/{}.ogg -ar 16000 ${BASEDIR}/wavs/{}.wav
```

开放语音录制应用程序为每个单词记录超过一秒的音频。这确保了用户的话语被捕捉到，即使他们的时间比我们预期的早或晚一点。训练需要一秒钟的录音，并且最好是单词位于每个录音的中间。我们创建了一个小型开源实用程序，用于查看每个录音随时间的音量，以便正确居中并修剪音频，使其仅为一秒钟。在终端中输入以下命令来使用它：

```py
git clone https://github.com/petewarden/extract_loudest_section \
  /tmp/extract_loudest_section_github
pushd /tmp/extract_loudest_section_github
make
popd
mkdir -p ${BASEDIR}/trimmed_wavs
/tmp/extract_loudest_section/gen/bin/extract_loudest_section \
  ${BASEDIR}'/wavs/*.wav' ${BASEDIR}/trimmed_wavs/
```

这将为您提供一个格式正确且所需长度的文件夹，但训练过程需要将 WAV 文件按标签组织到子文件夹中。标签编码在每个文件的名称中，因此我们有一个[示例 Python 脚本](https://oreil.ly/BpQBJ)，它使用这些文件名将它们分类到适当的文件夹中。

## 数据增强

数据增强是另一种有效扩大训练数据并提高准确性的方法。在实践中，这意味着对记录的话语应用音频变换，然后再用于训练。这些变换可以包括改变音量、混入背景噪音，或者轻微修剪片段的开头或结尾。训练脚本默认应用所有这些变换，但您可以使用命令行参数调整它们的使用频率和强度。

###### 警告

这种增强确实有助于使小数据集发挥更大作用，但它不能创造奇迹。如果你应用变换太强烈，可能会使训练输入变形得无法被人识别，这可能导致模型错误地开始触发与预期类别毫不相似的声音。

以下是如何使用其中一些命令行参数来控制增强：

```py
python tensorflow/examples/speech_commands/train.py \
  --model_architecture=tiny_conv --window_stride=20 --preprocess=micro \
  --wanted_words="yes,no" --silence_percentage=25 --unknown_percentage=25 \
  --quantize=1 --background_volume=0.2 --background_frequency=0.7 \
  --time_shift_ms=200
```

## 模型架构

我们之前训练的“是”/“否”模型旨在小而快速。它只有 18 KB，并且执行一次需要 400,000 次算术运算。为了符合这些约束条件，它牺牲了准确性。如果您正在设计自己的应用程序，您可能希望做出不同的权衡，特别是如果您试图识别超过两个类别。您可以通过修改*models.py*文件指定自己的模型架构，然后使用`--model_architecture`参数。您需要编写自己的模型创建函数，例如`create_tiny_conv_model0`，但要指定您想要的模型中的层。然后，您可以更新`create_model0`中的`if`语句，为您的架构命名，并在通过命令行传递架构参数时调用您的新创建函数。您可以查看一些现有的创建函数以获取灵感，包括如何处理辍学。如果您已添加了自己的模型代码，以下是如何调用它的方法：

```py
python tensorflow/examples/speech_commands/train.py \
 --model_architecture=my_model_name --window_stride=20 --preprocess=micro \
  --wanted_words="yes,no" --silence_percentage=25 \--unknown_percentage=25 \
  --quantize=1
```

# 总结

识别具有小内存占用的口语是一个棘手的现实世界问题，解决它需要我们与比简单示例更多的组件一起工作。大多数生产机器学习应用程序需要考虑问题，如特征生成、模型架构选择、数据增强、找到最适合的训练数据，以及如何将模型的结果转化为可操作信息。

根据产品的实际需求，需要考虑很多权衡，希望您现在了解一些选项，以便在从训练转向部署时使用。

在下一章中，我们将探讨如何使用不同类型的数据进行推断，尽管这种数据*看起来*比音频更复杂，但实际上却很容易处理。
