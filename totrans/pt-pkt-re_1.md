# 第一章：PyTorch 简介

PyTorch 是最受欢迎的深度学习 Python 库之一，广泛被人工智能研究社区使用。许多开发人员和研究人员使用 PyTorch 来加速深度学习研究实验和原型设计。

在本章中，我将简要介绍 PyTorch 是什么以及使其受欢迎的一些特点。我还将向您展示如何在本地机器和云端安装和设置 PyTorch 开发环境。通过本章的学习，您将能够验证 PyTorch 已正确安装并运行一个简单的 PyTorch 程序。

# 什么是 PyTorch？

PyTorch 库主要由 Facebook 的人工智能研究实验室（FAIR）开发，是一款免费的开源软件，拥有超过 1700 名贡献者。它允许您轻松运行基于数组的计算，在 Python 中构建动态神经网络，并进行自动微分，具有强大的图形处理单元（GPU）加速——这些都是深度学习研究所需的重要功能。尽管有些人用它来加速张量计算，但大多数人用它来进行深度学习开发。

PyTorch 的简单和灵活接口使快速实验成为可能。您可以加载数据，应用转换，并用几行代码构建模型。然后，您可以灵活地编写定制的训练、验证和测试循环，并轻松部署训练好的模型。

它拥有强大的生态系统和庞大的用户社区，包括斯坦福大学等大学和优步、英伟达和 Salesforce 等公司。2019 年，PyTorch 在机器学习和深度学习会议论文中占据主导地位：69%的计算机视觉与模式识别（CVPR）会议论文使用 PyTorch，超过 75%的计算语言学协会（ACL）和北美 ACL 分会（NAACL）使用它，超过 50%的学习表示国际会议（ICLR）和国际机器学习会议（ICML）也使用它。GitHub 上有超过 60000 个与 PyTorch 相关的存储库。

许多开发人员和研究人员使用 PyTorch 来加速深度学习研究实验和原型设计。其简单的 Python API、GPU 支持和灵活性使其成为学术和商业研究机构中的热门选择。自 2018 年开源以来，PyTorch 已经发布了稳定版本，并可以轻松安装在 Windows、Mac 和 Linux 操作系统上。该框架继续迅速扩展，现在可以方便地部署到云端和移动平台的生产环境中。

# 为什么使用 PyTorch？

如果您正在学习机器学习、进行深度学习研究或构建人工智能系统，您可能需要使用一个深度学习框架。深度学习框架使得执行常见任务如数据加载、预处理、模型设计、训练和部署变得容易。由于其简单性、灵活性和 Python 接口，PyTorch 已经在学术和研究社区中变得非常流行。以下是学习和使用 PyTorch 的一些原因：

PyTorch 很受欢迎

许多公司和研究机构将 PyTorch 作为他们的主要深度学习框架。事实上，一些公司已经在 PyTorch 的基础上构建了他们的自定义机器学习工具。因此，PyTorch 技能需求量大。

PyTorch 得到所有主要云平台的支持，如亚马逊网络服务（AWS）、谷歌云平台（GCP）、微软 Azure 和阿里云

您可以快速启动一个预装有 PyTorch 的虚拟机，进行无摩擦的开发。您可以使用预构建的 Docker 镜像，在云 GPU 平台上进行大规模训练，并在生产规模上运行模型。

PyTorch 得到 Google Colaboratory 和 Kaggle Kernels 的支持

您可以在浏览器中运行 PyTorch 代码，无需安装或配置。您可以通过在内核中直接运行 PyTorch 来参加 Kaggle 竞赛。

PyTorch 是成熟和稳定的

PyTorch 定期维护，现在已经超过 1.8 版本。

PyTorch 支持 CPU、GPU、TPU 和并行处理

您可以使用 GPU 和 TPU 加速训练和推断。张量处理单元（TPUs）是由 Google 开发的人工智能加速的应用特定集成电路（ASIC）芯片，旨在为 NN 硬件加速提供替代 GPU 的选择。通过并行处理，您可以在 CPU 上应用预处理，同时在 GPU 或 TPU 上训练模型。

PyTorch 支持分布式训练

您可以在多台机器上的多个 GPU 上训练神经网络。

PyTorch 支持部署到生产环境

借助新的 TorchScript 和 TorchServe 功能，您可以轻松将模型部署到包括云服务器在内的生产环境中。

PyTorch 开始支持移动部署

尽管目前仍处于实验阶段，但您现在可以将模型部署到 iOS 和 Android 设备上。

PyTorch 拥有庞大的生态系统和一套开源库

诸如 Torchvision、fastai 和 PyTorch Lightning 等库扩展了功能并支持特定领域，如自然语言处理（NLP）和计算机视觉。

PyTorch 还具有 C++前端

尽管本书将重点放在 Python 接口上，但 PyTorch 也支持前端 C++接口。如果您需要构建高性能、低延迟或裸机应用程序，可以使用相同的设计和架构在 C++中编写，就像使用 Python API 一样。

PyTorch 原生支持开放神经网络交换（ONNX）格式

您可以轻松将模型导出为 ONNX 格式，并在 ONNX 兼容的平台、运行时或可视化器中使用它们。

PyTorch 拥有庞大的开发者社区和用户论坛

PyTorch 论坛上有超过 38,000 名用户，通过访问[PyTorch 讨论论坛](https://pytorch.tips/discuss)很容易获得支持或发布问题。

# 入门

如果您熟悉 PyTorch，可能已经安装并设置了开发环境。如果没有，我将在本节中向您展示一些选项。开始的最快方式是使用 Google Colaboratory（或*Colab*）。Google Colab 是一个免费的基于云的开发环境，类似于 Jupyter Notebook，并已安装了 PyTorch。Colab 提供免费的有限 GPU 支持，并与 Google Drive 接口良好，可用于保存和共享笔记本。

如果您没有互联网访问，或者想在自己的硬件上运行 PyTorch 代码，那么我将向您展示如何在本地机器上安装 PyTorch。您可以在 Windows、Linux 和 macOS 操作系统上安装 PyTorch。我建议您拥有 NVIDIA GPU 进行加速，但不是必需的。

最后，您可能希望使用 AWS、Azure 或 GCP 等云平台开发 PyTorch 代码。如果您想使用云平台，我将向您展示在每个平台上快速入门的选项。

## 在 Google Colaboratory 中运行

使用 Google Colab，您可以在浏览器中编写和执行 Python 和 PyTorch 代码。您可以直接将文件保存到 Google Drive 帐户，并轻松与他人共享您的工作。要开始，请访问[Google Colab 网站](https://pytorch.tips/colab)，如图 1-1 所示。

![“Google Colaboratory 欢迎页面”](img/ptpr_0101.png)

###### 图 1-1\. Google Colaboratory 欢迎页面

如果您已经登录到您的 Google 帐户，将会弹出一个窗口。单击右下角的“新笔记本”。如果弹出窗口未出现，请单击“文件”，然后从菜单中选择“新笔记本”。您将被提示登录或创建 Google 帐户，如图 1-2 所示。

![“Google 登录”](img/ptpr_0102.png)

###### 图 1-2\. Google 登录

验证您的配置，导入 PyTorch 库，打印已安装的版本，并检查是否正在使用 GPU，如图 1-3 所示。

![“在 Google Colaboratory 中验证 PyTorch 安装”](img/ptpr_0103.png)

###### 图 1-3\. 在 Google Colaboratory 中验证 PyTorch 安装

默认情况下，我们的 Colab 笔记本不使用 GPU。您需要从运行时菜单中选择更改运行时类型，然后从“硬件加速器”下拉菜单中选择 GPU 并单击保存，如图 1-4 所示。

![“在 Google Colaboratory 中使用 GPU”](img/ptpr_0104.png)

###### 图 1-4\. 在 Google Colaboratory 中使用 GPU

现在再次运行单元格，选择单元格并按 Shift-Enter。您应该看到`is_available()`的输出为`True`，如图 1-5 所示。

![“在 Google Colab 中验证 GPU 是否激活”](img/ptpr_0105.png)

###### 图 1-5\. 在 Google Colaboratory 中验证 GPU 是否激活

###### 注意

Google 提供了一个付费版本称为 Colab Pro，提供更快的 GPU、更长的运行时间和更多内存。对于本书中的示例，免费版本的 Colab 应该足够了。

现在您已经验证了 PyTorch 已安装，并且您也知道版本。您还验证了您有一个可用的 GPU，并且正确安装和运行了适当的驱动程序。接下来，我将向您展示如何在本地机器上验证您的 PyTorch。

## 在本地计算机上运行

在某些情况下，您可能希望在本地机器或自己的服务器上安装 PyTorch。例如，您可能希望使用本地存储，或者使用自己的 GPU 或更快的 GPU 硬件，或者您可能没有互联网访问。运行 PyTorch 不需要 GPU，但需要 GPU 加速才能运行。我建议使用 NVIDIA GPU，因为 PyTorch 与用于 GPU 支持的 Compute Unified Device Architecture（CUDA）驱动程序紧密相关。

###### 警告

首先检查您的 GPU 和 CUDA 版本！PyTorch 仅支持特定的 GPU 和 CUDA 版本，许多 Mac 电脑使用非 NVIDIA GPU。如果您使用的是 Mac，请通过单击菜单栏上的苹果图标，选择“关于本机”，然后单击“显示”选项卡来验证您是否有 NVIDIA GPU。如果您在 Mac 上看到 NVIDIA GPU 并希望使用它，您将需要从头开始构建 PyTorch。如果您没有看到 NVIDIA GPU，则应使用 PyTorch 的仅 CPU 版本或选择另一台具有不同操作系统的计算机。

PyTorch 网站提供了一个[方便的浏览器工具用于安装](https://pytorch.tips/install-local)，如图 1-6 所示。选择最新的稳定版本，您的操作系统，您喜欢的 Python 包管理器（推荐使用 Conda），Python 语言和您的 CUDA 版本。执行命令行并按照您的配置的说明进行操作。请注意先决条件、安装说明和验证方法。

![“”](img/ptpr_0106.png)

###### 图 1-6\. PyTorch 在线安装配置工具

您应该能够在您喜欢的 IDE（Jupyter Notebook、Microsoft Visual Studio Code、PyCharm、Spyder 等）或终端中运行验证代码片段。图 1-7 显示了如何在 Mac 终端上验证 PyTorch 的正确版本是否已安装。相同的命令也可以用于在 Windows 或 Linux 终端中验证。

![“”](img/ptpr_0107.png)

###### 图 1-7\. 使用 Mac 终端验证 PyTorch

## 在云平台上运行

如果您熟悉 AWS、GCP 或 Azure 等云平台，您可以在云中运行 PyTorch。云平台为训练和部署深度学习模型提供强大的硬件和基础设施。请记住，使用云服务，特别是 GPU 实例，会产生额外的费用。要开始，请按照感兴趣的平台的[在线 PyTorch 云设置指南](https://pytorch.tips/start-cloud)中的说明进行操作。

设置云环境超出了本书的范围，但我将总结可用的选项。每个平台都提供虚拟机实例以及托管服务来支持 PyTorch 开发。

### 在 AWS 上运行

AWS 提供多种在云中运行 PyTorch 的选项。如果您更喜欢全面托管服务，可以使用 AWS SageMaker，或者如果您更喜欢管理自己的基础架构，可以使用 AWS 深度学习 Amazon 机器映像（AMI）或容器：

Amazon SageMaker

这是一个全面托管的服务，用于训练和部署模型。您可以从仪表板运行 Jupyter 笔记本，并使用 SageMaker Python SDK 在云中训练和部署模型。您可以在专用 GPU 实例上运行您的笔记本。

AWS 深度学习 AMI

这些是预配置的虚拟机环境。您可以选择 Conda AMI，其中预先安装了许多库（包括 PyTorch），或者如果您更喜欢一个干净的环境来设置私有存储库或自定义构建，可以使用基本 AMI。

AWS 深度学习容器

这些是预先安装了 PyTorch 的 Docker 镜像。它们使您可以跳过从头开始构建和优化环境的过程，主要用于部署。

有关如何入门的更详细信息，请查看[“在 AWS 上开始使用 PyTorch”说明](https://pytorch.tips/start-aws)。

### 在 Microsoft Azure 上运行

Azure 还提供多种在云中运行 PyTorch 的选项。您可以使用名为 Azure Machine Learning 的全面托管服务开发 PyTorch 模型，或者如果您更喜欢管理自己的基础架构，可以运行数据科学虚拟机（DSVMs）：

Azure Machine Learning

这是一个用于构建和部署模型的企业级机器学习服务。它包括拖放设计器和 MLOps 功能，可与现有的 DevOps 流程集成。

DSVMs

这些是预配置的虚拟机环境。它们预先安装了 PyTorch 和其他深度学习框架以及开发工具，如 Jupyter Notebook 和 VS Code。

有关如何入门的更详细信息，请查看[Azure Machine Learning 文档](https://pytorch.tips/azure)。

### 在 Google 云平台上运行

GCP 还提供多种在云中运行 PyTorch 的选项。您可以使用名为 AI 平台笔记本的托管服务开发 PyTorch 模型，或者如果您更喜欢管理自己的基础架构，可以运行深度学习 VM 镜像：

AI 平台笔记本

这是一个托管服务，其集成的 JupyterLab 环境允许您创建预配置的 GPU 实例。

深度学习 VM 镜像

这些是预配置的虚拟机环境。它们预先安装了 PyTorch 和其他深度学习框架以及开发工具。

有关如何入门的更详细信息，请查看 Google Cloud 的[“AI 和机器学习产品”说明](https://pytorch.tips/google-cloud)。

## 验证您的 PyTorch 环境

无论您使用 Colab、本地计算机还是您喜爱的云平台，您都应该验证 PyTorch 是否已正确安装，并检查是否有 GPU 可用。您已经在 Colab 中看到了如何执行此操作。要验证 PyTorch 是否已正确安装，请使用以下代码片段。该代码导入 PyTorch 库，打印版本，并检查是否有 GPU 可用：

```py
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

###### 警告

您使用`import torch`导入库，而不是`import pytorch`。PyTorch 最初基于`torch`库，这是一个基于 C 和 Lua 编程语言的开源机器学习框架。保持库命名为`torch`允许 Torch 代码与更高效的 PyTorch 实现重用。

# 一个有趣的例子

现在您已经验证了您的环境是否正确配置，让我们编写一个有趣的例子，展示 PyTorch 的一些特性，并演示机器学习中的最佳实践。在这个例子中，我们将构建一个经典的图像分类器，尝试根据 1,000 个可能的类别或选择来识别图像的内容。

您可以从[本书的 GitHub 存储库](https://github.com/joe-papa/pytorch-book)访问此示例并跟随。尝试在 Google Colab、本地计算机或 AWS、Azure 或 GCP 等云平台上运行代码。不用担心理解机器学习的所有概念。我们将在本书中更详细地介绍它们。

###### 注意

在实践中，您将在代码开头导入所有必要的库。然而，在这个例子中，我们将在使用时导入库，这样您就可以看到每个任务需要哪些库。

首先，让我们选择一个我们想要分类的图像。在这个例子中，我们将选择一杯美味的新鲜热咖啡。使用以下代码将咖啡图像下载到您的本地环境：

```py
import urllib.request

url = url = 'https://pytorch.tips/coffee'
fpath = 'coffee.jpg'
urllib.request.urlretrieve(url, fpath)
```

请注意，代码使用`urllib`库的`urlretrieve()`函数从网络获取图像。我们通过指定`fpath`将文件重命名为*coffee.jpg*。

接下来，我们使用 Pillow 库（PIL）读取我们的本地图像：

```py
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('coffee.jpg')
plt.imshow(img)
```

图 1-8 展示了我们的图像是什么样子的。我们可以使用`matplotlib`的`imshow()`函数在我们的系统上显示图像，就像前面的代码所示的那样。

![分类器的输入图像 - 一杯新鲜热咖啡](img/ptpr_0108.png)

###### 图 1-8。分类器的输入图像

请注意我们还没有使用 PyTorch。这里就是事情变得令人兴奋的地方。接下来，我们将把我们的图像传递给一个预训练的图像分类神经网络（NN）—但在这之前，我们需要*预处理*我们的图像。在机器学习中，预处理数据是非常常见的，因为 NN 期望输入满足某些要求。

在我们的示例中，图像数据是一个 RGB 1600 × 1200 像素的 JPEG 格式图像。我们需要应用一系列预处理步骤，称为*转换*，将图像转换为 NN 的正确格式。我们使用 Torchvision 在下面的代码中实现这一点：

```py
import torch
from torchvision import transforms

transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])])

img_tensor = transform(img)
print(type(img_tensor), img_tensor.shape)
# out:
# <class 'torch.tensor'> torch.Size([3, 224, 224])
```

我们使用`Compose()`转换来定义一系列用于预处理我们的图像的转换。首先，我们需要调整大小并裁剪图像以适应 NN。图像目前是 PIL 格式，因为我们之前是这样读取的。但是我们的 NN 需要一个张量输入，所以我们将 PIL 图像转换为张量。

张量是 PyTorch 中的基本数据对象，我们将在整个下一章中探索它们。您可以将张量视为 NumPy 数组或带有许多额外功能的数值数组。现在，我们只需将我们的图像转换为一个数字的张量数组，使其准备好。

我们应用了另一个叫做`Normalize()`的转换，来重新缩放像素值的范围在 0 和 1 之间。均值和标准差（std）的值是基于用于训练模型的数据预先计算的。对图像进行归一化可以提高分类器的准确性。

最后，我们调用`transform(img)`来将所有的转换应用到图像上。正如你所看到的，`img_tensor`是一个 3 × 224 × 224 的`torch.Tensor`，代表着一个 3 通道、224 × 224 像素的图像。

高效的机器学习过程会批处理数据，我们的模型会期望一批数据。然而，我们只有一张图像，所以我们需要创建一个大小为 1 的批次，如下面的代码所示：

```py
batch = img_tensor.unsqueeze(0)
print(batch.shape)
# out: torch.Size([1, 3, 224, 224])
```

我们使用 PyTorch 的`unsqueeze()`函数向我们的张量添加一个维度，并创建一个大小为 1 的批次。现在我们有一个大小为 1 × 3 × 224 × 224 的张量，代表一个批次大小为 1 和 3 通道（RGB）的 224 × 224 像素。PyTorch 提供了许多有用的函数，比如`unsqueeze()`来操作张量，我们将在下一章中探索其中许多函数。

现在我们的图像已经准备好用于我们的分类器 NN 了！我们将使用一个名为 AlexNet 的著名图像分类器。AlexNet 在 2012 年的 ImageNet 大规模视觉识别挑战赛中获胜。使用 Torchvision 很容易加载这个模型，如下面的代码所示：

```py
from torchvision import models

model = models.alexnet(pretrained=True)
```

我们将在这里使用一个预训练的模型，所以不需要训练它。AlexNet 模型已经使用数百万张图像进行了预训练，并且在分类图像方面表现得相当不错。让我们传入我们的图像，看看它的表现：

```py
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# out(results will vary): cpu

model.eval()
model.to(device)
y = model(batch.to(device))
print(y.shape)
# out: torch.Size([1, 1000])
```

GPU 加速是 PyTorch 的一个关键优势。在第一行中，我们使用 PyTorch 的`cuda.is_available()`函数来查看我们的机器是否有 GPU。这是 PyTorch 代码中非常常见的一行，我们将在第二章和第六章进一步探讨 GPU。我们只对一个图像进行分类，所以这里不需要 GPU，但如果我们有一个巨大的批次，使用 GPU 可能会加快速度。

`model.eval()`函数配置我们的 AlexNet 模型进行推断或预测（与训练相对）。模型的某些组件仅在训练期间使用，我们不希望在这里使用它们。使用`model.to(device)`和`batch.to(device)`将我们的模型和输入数据发送到 GPU（如果可用），执行`model(batch.to(device))`运行我们的分类器。

输出`y`包含一个批次的 1,000 个输出。由于我们的批次只包含一个图像，第一维是`1`，而类的数量是`1000`，每个类有一个值。值越高，图像包含该类的可能性就越大。以下代码找到获胜的类：

```py
y_max, index = torch.max(y,1)
print(index, y_max)
# out: tensor([967]) tensor([22.3059],
#    grad_fn=<MaxBackward0>)
```

使用 PyTorch 的`max()`函数，我们看到索引为 967 的类具有最高值 22.3059，因此是获胜者。但是，我们不知道类 967 代表什么。让我们加载包含类名的文件并找出：

```py
url = 'https://pytorch.tips/imagenet-labels'

fpath = 'imagenet_class_labels.txt'
urllib.request.urlretrieve(url, fpath)

with open('imagenet_class_labels.txt') as f:
  classes = [line.strip() for line in f.readlines()]

print(classes[967])
# out: 967: 'espresso',
```

就像我们之前做的那样，我们使用`urlretrieve()`下载包含每个类描述的文本文件。然后，我们使用`readlines()`读取文件，并创建一个包含类名的列表。当我们`print(classes[967])`时，它显示类 967 是*espresso*！

使用 PyTorch 的`softmax()`函数，我们可以将输出值转换为概率：

```py
prob = torch.nn.functional.softmax(y, dim=1)[0] * 100
print(classes[index[0]], prob[index[0]].item())
#967: 'espresso', 87.85208892822266
```

要打印索引处的概率，我们使用 PyTorch 的`tensor.item()`方法。`item()`方法经常被使用，并返回张量中包含的数值。结果显示，模型有 87.85%的把握这是一张浓缩咖啡的图像。

我们可以使用 PyTorch 的`sort()`函数对输出概率进行排序，并查看前五个：

```py
_, indices = torch.sort(y, descending=True)

for idx in indices[0][:5]:
  print(classes[idx], prob[idx].item())
# out:
# 967: 'espresso', 87.85208892822266
# 968: 'cup', 7.28359317779541
# 504: 'coffee mug', 4.33521032333374
# 925: 'consomme', 0.36686763167381287
# 960: 'chocolate sauce, chocolate syrup',
#    0.09037172049283981
```

我们看到模型预测图像是*espresso*的概率为 87.85%。它还以 7.28%的概率预测*cup*，以 4.3%的概率预测*coffee mug*，但它似乎非常确信图像是一杯浓缩咖啡。

您可能现在感觉需要一杯浓缩咖啡。在那个示例中，我们涵盖了很多内容！实际上，实现所有这些的核心代码要短得多。假设您已经下载了文件，您只需要运行以下代码来使用 AlexNet 对图像进行分类：

```py
import torch
from torchvision import transforms, models

transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])])

img_tensor = transform(img)
batch = img_tensor.unsqueeze(0)
model = models.alexnet(pretrained=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)
y = model(batch.to(device))

prob = torch.nn.functional.softmax(y, dim=1)[0] * 100
_, indices = torch.sort(y, descending=True)
for idx in indices[0][:5]:
  print(classes[idx], prob[idx].item())
```

这就是如何使用 PyTorch 构建图像分类器。尝试通过模型运行自己的图像，并查看它们的分类情况。还可以尝试在另一个平台上完成示例。例如，如果您使用 Colab 运行代码，请尝试在本地或云中运行它。

恭喜，您已经验证了您的环境已正确配置，并且可以执行 PyTorch 代码！我们将在本书的其余部分更深入地探讨每个主题。在下一章中，我们将探讨 PyTorch 的基础知识，并提供张量及其操作的快速参考。
