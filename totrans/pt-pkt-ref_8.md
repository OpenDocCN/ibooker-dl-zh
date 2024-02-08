# 第八章：PyTorch 生态系统和其他资源

在之前的章节中，您已经学会了使用 PyTorch 设计和部署深度学习模型所需的一切。您已经学会了如何在不同平台上构建、训练、测试和加速您的模型，以及如何将这些模型部署到云端和边缘设备。正如您所见，PyTorch 在开发和部署环境中都具有强大的功能，并且高度可扩展，允许您创建符合您需求的定制化。

为了总结这个参考指南，我们将探索 PyTorch 生态系统、其他支持库和额外资源。PyTorch 生态系统是 PyTorch 最强大的优势之一。它提供了丰富的项目、工具、模型、库和平台，用于探索人工智能并加速您的人工智能开发。

PyTorch 生态系统包括由研究人员、第三方供应商和 PyTorch 社区创建的项目和库。这些项目得到了 PyTorch 团队的认可，以确保它们的质量和实用性。

此外，PyTorch 项目还包括支持特定领域的其他库，包括用于计算机视觉的 Torchvision 和用于 NLP 的 Torchtext。PyTorch 还支持其他包，如 TensorBoard 用于可视化，还有大量的学习资源供进一步研究，如 Papers with Code 和 PyTorch Academy。

在本章中，我们将首先概述 PyTorch 生态系统及其支持的项目和工具的高级视图。然后，我们将深入了解一些最强大和流行的资源，提供关于它们的使用和 API 的参考资料。最后，我将向您展示如何通过各种教程、书籍、课程和其他培训资源进一步学习。

让我们开始看看生态系统提供了什么。

# PyTorch 生态系统

截至 2021 年初，[PyTorch 生态系统](https://pytorch.tips/ecosystem)拥有超过 50 个库和项目，这个列表还在不断增长。其中一些是特定领域的项目，例如专门用于计算机视觉或 NLP 解决方案的项目。其他项目，如 PyTorch Lightning 和 fastai，提供了编写简洁代码的框架，而像 PySyft 和 Crypten 这样的项目支持安全性和隐私性。还有支持强化学习、游戏模型、模型可解释性和加速的项目。在本节中，我们将探索包含在 PyTorch 生态系统中的项目。

表 8-1 提供了支持*计算机视觉*应用的生态系统项目列表。

表 8-1. 计算机视觉项目

| 项目 | 描述 |
| --- | --- |
| Torchvision | PyTorch 的计算机视觉库，提供常见的转换、模型和实用程序，以支持计算机视觉应用（[*https://pytorch.tips/torchvision*](https://pytorch.tips/torchvision)） |
| Detectron2 | Facebook 的目标检测和分割平台（[*https://pytorch.tips/detectron2*](https://pytorch.tips/detectron2)） |
| Albumentations | 图像增强库（[*https://pytorch.tips/albumentations*](https://pytorch.tips/albumentations)） |
| PyTorch3D | 用于 3D 计算机视觉的可重用组件集合（[*https://pytorch.tips/pytorch3d*](https://pytorch.tips/pytorch3d)） |
| Kornia | 用于计算机视觉的可微分模块库（[*https://pytorch.tips/kornia*](https://pytorch.tips/kornia)） |
| MONAI | 用于医疗影像深度学习的框架（[*https://pytorch.tips/monai*](https://pytorch.tips/monai)） |
| TorchIO | 用于 3D 医学图像的工具包（[*https://pytorch.tips/torchio*](https://pytorch.tips/torchio)） |

Torchvision 是计算机视觉应用中最强大的库之一，包含在 PyTorch 项目中。它也由 PyTorch 开发团队维护。我们将在本章后面更详细地介绍 Torchvision API。

PyTorch3D 和 TorchIO 为 3D 成像提供了额外支持，而 TorchIO 和 MONAI 专注于医学成像应用。Detectron2 是一个强大的物体检测平台。如果您正在进行计算机视觉研究和开发，这些扩展可能有助于加速您的结果。

与计算机视觉一样，过去十年来在 NLP 研究中取得了重大进展，NLP 应用也得到了 PyTorch 的良好支持。

表 8-2 提供了支持*NLP 和基于音频*应用的生态系统项目列表。

表 8-2. NLP 和音频项目

| 项目 | 描述 |
| --- | --- |
| Torchtext | PyTorch 的自然语言处理和文本处理库（[*https://pytorch.tips/torchtext*](https://pytorch.tips/torchtext)） |
| Flair | NLP 的简单框架（[*https://pytorch.tips/flair*](https://pytorch.tips/flair)） |
| AllenNLP | 用于设计和评估 NLP 模型的库（[*https://pytorch.tips/allennlp*](https://pytorch.tips/allennlp)） |
| ParlAI | 用于共享、训练和测试对话模型的框架（[*https://pytorch.tips/parlai*](https://pytorch.tips/parlai)） |
| NeMo | 用于会话 AI 的工具包（[*https://pytorch.tips/nemo*](https://pytorch.tips/nemo)） |
| PyTorch NLP | NLP 的基本工具（[*https://pytorch.tips/pytorchnlp*](https://pytorch.tips/pytorchnlp)） |
| Translate | Facebook 的机器翻译平台（[*https://pytorch.tips/translate*](https://pytorch.tips/translate)） |
| TorchAudio | PyTorch 的音频预处理库（[*https://pytorch.tips/torchaudio*](https://pytorch.tips/torchaudio)） |

与 Torchvision 一样，Torchtext 作为 PyTorch 项目的一部分包含在内，并由 PyTorch 开发团队维护。Torchtext 为处理文本数据和开发基于 NLP 的模型提供了强大的功能。

Flair、AllenNLP 和 PyTorch NLP 为基于文本的处理和 NLP 模型开发提供了额外功能。ParlAI 和 NeMo 提供了开发对话和会话 AI 系统的工具，而 Translate 专注于机器翻译。

TorchAudio 提供处理音频文件（如语音和音乐）的功能。

强化学习和游戏也是研究的快速增长领域，有工具支持它们使用 PyTorch。

表 8-3 提供了支持*游戏和强化学习*应用的生态系统项目列表。

表 8-3. 游戏和强化学习项目

| 项目 | 描述 |
| --- | --- |
| ELF | 用于在游戏环境中训练和测试算法的项目（[*https://pytorch.tips/elf*](https://pytorch.tips/elf)） |
| PFRL | 深度强化学习算法库（[*https://pytorch.tips/pfrl*](https://pytorch.tips/pfrl)） |

ELF（广泛、轻量级、灵活的游戏研究平台）是 Facebook 开发的开源项目，重新实现了像 AlphaGoZero 和 AlphaZero 这样的游戏算法。PFRL（首选强化学习）是由 Preferred Networks 开发的基于 PyTorch 的开源深度强化学习库，是 Chainer 和 ChainerRL 的创造者。它可以用来创建强化学习的基线算法。PFRL 目前为 11 个基于原始研究论文的关键深度强化学习算法提供了可重现性脚本。

正如您在本书中所看到的，PyTorch 是一个高度可定制的框架。这种特性有时会导致需要为常见任务经常编写相同的样板代码。为了帮助开发人员更快地编写代码并消除样板代码的需要，几个 PyTorch 项目提供了高级编程 API 或与其他高级框架（如 scikit-learn）的兼容性。

表 8-4 提供了支持*高级编程*的生态系统项目列表。

表 8-4. 高级编程项目

| 项目 | 描述 |
| --- | --- |
| fastai | 简化使用现代实践进行训练的库（[*https://pytorch.tips/fastai*](https://pytorch.tips/fastai)） |
| PyTorch Lightning | 可定制的类 Keras ML 库，消除样板代码（[*https://pytorch.tips/lightning*](https://pytorch.tips/lightning)） |
| Ignite | 用于编写紧凑、功能齐全的训练循环的库（[*https://pytorch.tips/ignite*](https://pytorch.tips/ignite)） |
| Catalyst | 用于紧凑强化学习流水线的框架（[*https://pytorch.tips/catalyst*](https://pytorch.tips/catalyst)） |
| skorch | 提供与 scikit-learn 兼容的 PyTorch（[*https://pytorch.tips/skorch*](https://pytorch.tips/skorch)） |
| Hydra | 用于配置复杂应用程序的框架（[*https://pytorch.tips/hydra*](https://pytorch.tips/hydra)） |
| higher | 促进复杂元学习算法实现的库（[*https://pytorch.tips/higher*](https://pytorch.tips/higher)） |
| Poutyne | 用于样板代码的类 Keras 框架（[*https://pytorch.tips/poutyne*](https://pytorch.tips/poutyne)） |

Fastai 是建立在 PyTorch 上的研究和学习框架。它有全面的文档，并自早期提供了 PyTorch 的高级 API。您可以通过查阅其文档和免费的[在线课程](https://pytorch.tips/fastai)或阅读 Jeremy Howard 和 Sylvain Gugger（O'Reilly）合著的书籍[*使用 fastai 和 PyTorch 进行编码的深度学习*](https://pytorch.tips/fastai-book)来快速掌握该框架。

PyTorch Lightning 也已成为 PyTorch 非常受欢迎的高级编程 API 之一。它为训练、验证和测试循环提供了所有必要的样板代码，同时允许您轻松添加自定义方法。

Ignite 和 Catalyst 也是流行的高级框架，而 skorch 和 Poutyne 分别提供了类似于 scikit-learn 和 Keras 的接口。Hydra 和 higher 用于简化复杂应用程序的配置。

除了高级框架外，生态系统中还有支持硬件加速和优化推理的软件包。

表 8-5 提供了支持*推理加速*应用程序的生态系统项目列表。

表 8-5。推理项目

| 项目 | 描述 |
| --- | --- |
| Glow | 用于硬件加速的 ML 编译器（[*https://pytorch.tips/glow*](https://pytorch.tips/glow)） |
| Hummingbird | 编译经过训练的模型以实现更快推理的库（[*https://pytorch.tips/hummingbird*](https://pytorch.tips/hummingbird)） |

Glow 是用于硬件加速器的机器学习编译器和执行引擎，可以用作高级深度学习框架的后端。该编译器允许进行最先进的优化和神经网络图的代码生成。Hummingbird 是由微软开发的开源项目，是一个库，用于将经过训练的传统 ML 模型编译为张量计算，并无缝地利用 PyTorch 加速传统 ML 模型。

除了加速推理外，PyTorch 生态系统还包含用于加速训练和使用分布式训练优化模型的项目。

表 8-6 提供了支持*分布式训练和模型优化*的生态系统项目列表。

表 8-6。分布式训练和模型优化项目

| 项目 | 描述 |
| --- | --- |
| Ray | 用于构建和运行分布式应用程序的快速、简单框架（[*https://pytorch.tips/ray*](https://pytorch.tips/ray)） |
| Horovod | 用于 TensorFlow、Keras、PyTorch 和 Apache MXNet 的分布式深度学习训练框架（[*https://pytorch.tips/horovod*](https://pytorch.tips/horovod)） |
| DeepSpeed | 优化库（[*https://pytorch.tips/deepspeed*](https://pytorch.tips/deepspeed)） |
| Optuna | 自动化超参数搜索和优化（[*https://pytorch.tips/optuna*](https://pytorch.tips/optuna)） |
| Polyaxon | 用于构建、训练和监控大规模深度学习应用程序的平台（[*https://pytorch.tips/polyaxon*](https://pytorch.tips/polyaxon)) |
| Determined | 使用共享 GPU 和协作训练模型的平台（[*https://pytorch.tips/determined*](https://pytorch.tips/determined)) |
| Allegro Trains | 包含深度学习实验管理器、版本控制和机器学习操作的库（[*https://pytorch.tips/allegro*](https://pytorch.tips/allegro)) |

Ray 是一个用于构建分布式应用程序的 Python API，并打包了其他库以加速机器学习工作负载。我们在第六章中使用了其中一个软件包 Ray Tune 来在分布式系统上调整超参数。Ray 是一个非常强大的软件包，还可以支持可扩展的强化学习、分布式训练和可扩展的服务。Horovod 是另一个分布式框架。它专注于分布式训练，并可与 Ray 一起使用。

DeepSpeed、Optuna 和 Allegro Trains 还支持超参数调优和模型优化。Polyaxon 可用于规模化训练和监控模型，而 Determined 专注于共享 GPU 以加速训练。

随着 PyTorch 的流行，已经开发了许多专门的软件包来支持特定领域和特定工具。这些工具中的许多旨在改进模型或数据的预处理。

表 8-7 提供了支持*建模和数据处理*的生态系统项目列表。

表 8-7. 建模和数据处理项目

| 项目 | 描述 |
| --- | --- |
| TensorBoard | TensorBoard 的数据和模型可视化工具已集成到 PyTorch 中（[*https://pytorch.tips/pytorch-tensorboard*](https://pytorch.tips/pytorch-tensorboard)) |
| PyTorch Geometric | 用于 PyTorch 的几何深度学习扩展库（[*https://pytorch.tips/geometric*](https://pytorch.tips/geometric)) |
| Pyro | 灵活且可扩展的深度概率建模（[*https://pytorch.tips/pyro*](https://pytorch.tips/pyro)) |
| Deep Graph Library (DGL) | 用于实现图神经网络的库（[*https://pytorch.tips/dgl*](https://pytorch.tips/dgl)) |
| MMF | Facebook 的多模型深度学习（视觉和语言）模块化框架（[*https://pytorch.tips/mmf*](https://pytorch.tips/mmf)) |
| GPyTorch | 用于创建可扩展高斯过程模型的库（[*https://pytorch.tips/gpytorch*](https://pytorch.tips/gpytorch)) |
| BoTorch | 用于贝叶斯优化的库（[*https://pytorch.tips/botorch*](https://pytorch.tips/botorch)) |
| Torch Points 3D | 用于非结构化 3D 空间数据的框架（[*https://pytorch.tips/torchpoints3d*](https://pytorch.tips/torchpoints3d)) |
| TensorLy | 用于张量方法和深度张量神经网络的高级 API（[*https://pytorch.tips/tensorly*](https://pytorch.tips/tensorly))([*https://pytorch.tips/advertorch*](https://pytorch.tips/advertorch)) |
| BaaL | 从贝叶斯理论中实现主动学习（[*https://pytorch.tips/baal*](https://pytorch.tips/baal)) |
| PennyLane | 量子机器学习库（[*https://pytorch.tips/pennylane*](https://pytorch.tips/pennylane)) |

TensorBoard 是为 TensorFlow 开发的非常流行的可视化工具，也可以用于 PyTorch。我们将在本章后面介绍这个工具及其 PyTorch API。

PyTorch Geometric、Pyro、GPyTorch、BoTorch 和 BaaL 都支持不同类型的建模，如几何建模、概率建模、高斯建模和贝叶斯优化。

Facebook 的 MMF 是一个功能丰富的多模态建模软件包，而 Torch Points 3D 可用于对通用的 3D 空间数据进行建模。

PyTorch 作为一个工具的成熟和稳定性体现在用于支持安全和隐私的软件包的出现。随着法规要求系统在这些领域合规，安全和隐私问题变得更加重要。

表 8-8 提供了支持*安全性和隐私性*的生态系统项目列表。

表 8-8\. 安全和隐私项目

| 项目 | 描述 |
| --- | --- |
| AdverTorch | 用于对抗性示例和防御攻击的模块 |
| PySyft | 用于模型加密和隐私的库（[*https://pytorch.tips/pysyft*](https://pytorch.tips/pysyft)) |
| Opacus | 用于训练具有差分隐私的模型的库（[*https://pytorch.tips/opacus*](https://pytorch.tips/opacus)) |
| CrypTen | 隐私保护 ML 的框架（[*https://pytorch.tips/crypten*](https://pytorch.tips/crypten)) |

PySyft、Opacus 和 CrypTen 是支持安全性和隐私性的 PyTorch 包。它们添加了保护和加密模型以及用于创建模型的数据的功能。

通常，深度学习似乎是一个黑匣子，开发人员不知道模型为什么做出决策。然而，如今，这种缺乏透明度已不再可接受：人们越来越意识到公司及其高管必须对其算法的公平性和运作负责。模型可解释性对于研究人员、开发人员和公司高管来说很重要，以了解模型为何产生其结果。

表 8-9 显示了支持*模型* *可解释性*的生态系统项目。

表 8-9\. 模型可解释性项目

| 项目 | 描述 |
| --- | --- |
| Captum | 用于模型可解释性的库（[*https://pytorch.tips/captum*](https://pytorch.tips/captum)) |
| 视觉归因 | 用于模型可解释性的最新视觉归因方法的 PyTorch 实现（[*https://pytorch.tips/visual-attribution*](https://pytorch.tips/visual-attribution)) |

目前，Captum 是支持模型可解释性的首要 PyTorch 项目。视觉归因包对解释计算机视觉模型和识别图像显著性很有用。随着领域的扩展，更多的项目肯定会进入这个领域。

正如您所看到的，PyTorch 生态系统包括广泛的开源项目，可以在许多不同的方面帮助您。也许您正在开展一个可以使其他研究人员受益的项目。如果您希望将您的项目纳入官方 PyTorch 生态系统，请访问[PyTorch 生态系统申请页面](https://pytorch.tips/join-ecosystem)。

在考虑应用程序时，PyTorch 团队寻找符合以下要求的项目：

+   您的项目使用 PyTorch 来改善用户体验、添加新功能或加快训练/推理速度。

+   您的项目稳定、维护良好，并包含足够的基础设施、文档和技术支持。

生态系统不断增长。要获取最新的项目列表，请访问[PyTorch 生态系统网站](https://pytorch.tips/ecosystem)。要向我们更新书中的新项目，请发送电子邮件至作者邮箱 jpapa@joepapa.ai。

接下来，我们将更深入地了解一些 PyTorch 项目的支持工具和库。显然，我们无法在本书中涵盖所有可用的库和工具，但在接下来的章节中，我们将探索一些最受欢迎和有用的库，以帮助您更深入地了解它们的 API 和用法。

# Torchvision 用于图像和视频

我们在本书中使用了 Torchvision，它是计算机视觉研究中最强大和有用的 PyTorch 库之一。从技术上讲，Torchvision 包是 PyTorch 项目的一部分。它包括一系列流行的数据集、模型架构和常见的图像转换。

## 数据集和 I/O

Torchvision 提供了大量的数据集。它们包含在`torchvision.datasets`库中，可以通过创建数据集对象来访问，如下面的代码所示：

```py
import torchvision

train_data = torchvision.datasets.CIFAR10(
          root=".",
          train=True,
          transform=None,
          download=True)
```

您只需调用构造函数并传入适当的选项。此代码使用训练数据从 CIFAR-10 数据集创建数据集对象，不使用任何转换。它会在当前目录中查找数据集文件，如果文件不存在，它将下载它们。

表 8-10 提供了 Torchvision 提供的数据集的全面列表。

表 8-10。Torchvision 数据集

| 数据集 | 描述 |
| --- | --- |
| CelebA | 大规模人脸属性数据集，包含超过 200,000 张名人图像，每张图像有 40 个属性注释。 |
| CIFAR-10 | CIFAR-10 数据集包含 60,000 个 32×32 彩色图像，分为 10 个类别，分为 50,000 个训练图像和 10,000 个测试图像。还提供了包含 100 个类别的 CIFAR-100 数据集。 |
| Cityscapes | 包含来自 50 个不同城市街景记录的视频序列的大规模数据集，带有注释。 |
| COCO | 大规模目标检测、分割和字幕数据集。 |
| DatasetFolder | 用于从文件夹结构中创建任何数据集。 |
| EMNIST | MNIST 的手写字母扩展。 |
| FakeData | 一个返回随机生成图像作为 PIL 图像的虚假数据集。 |
| Fashion-MNIST | Zalando 服装图像数据集，符合 MNIST 格式（60,000 个训练示例，10,000 个测试示例，28×28 灰度图像，10 个类别）。 |
| Flickr | Flickr 8,000 张图像数据集。 |
| HMDB51 | 大型人体运动视频序列数据库。 |
| ImageFolder | 用于从文件夹结构中创建图像数据集。 |
| ImageNet | 包含 14,197,122 张图像和 21,841 个单词短语的图像分类数据集。 |
| Kinetics-400 | 大规模动作识别视频数据集，包含 650,000 个持续 10 秒的视频剪辑，涵盖高达 700 个人类动作类别，如演奏乐器、握手和拥抱。 |
| KMNIST | Kuzushiji-MNIST，MNIST 数据集的替代品（70,000 个 28×28 灰度图像），其中每个字符代表平假名的 10 行之一。 |
| LSUN | 每个 10 个场景类别和 20 个对象类别的一百万标记图像。 |
| MNIST | 手写的单个数字，28×28 灰度图像，有 60,000 个训练和 10,000 个测试样本。 |
| Omniglot | 由 50 种不同字母表的 1,623 个不同手写字符生成的数据集。 |
| PhotoTour | 包含 1,024×1,024 位图图像的照片旅游数据集，每个图像包含一个 16×16 的图像块数组。 |
| Places365 | 包含 400 多个独特场景类别的 10,000,000 张图像数据集，每个类别有 5,000 至 30,000 张训练图像。 |
| QMNIST | Facebook 的项目，从 NIST 特殊数据库 19 中找到的原始数据生成 MNIST 数据集。 |
| SBD | 包含 11,355 张图像的语义分割注释的语义边界数据集。 |
| SBU | Stony Brook 大学（SBU）标题照片数据集，包含超过 1,000,000 张带标题的图像。 |
| STL10 | 用于无监督学习的类似于 CIFAR-10 的数据集。96×96 彩色图像的 10 个类别，包括 5,000 个训练图像，8,000 个测试图像和 100,000 个未标记图像。 |
| SVHN | 街景房屋号码数据集，类似于 MNIST，但是在自然场景彩色图像中有 10 倍的数据。 |
| UCF101 | 包含来自 101 个动作类别的 13,320 个视频的动作识别数据集。 |
| USPS | 包含 16×16 手写文本图像的数据集，有 10 个类别，7,291 个训练图像和 2,007 个测试图像。 |
| VOC | 用于目标类别识别的 PASCAL 视觉对象类别图像数据集。2012 年版本有 20 个类别，11,530 个训练/验证图像，27,450 个感兴趣区域（ROI）标注对象和 6,929 个分割。 |

Torchvision 不断添加更多数据集。要获取最新列表，请访问[Torchvision 文档](https://pytorch.tips/torchvision-datasets)。

## 模型

Torchvision 还提供了一个广泛的模型列表，包括模块架构和预训练权重（如果有的话）。通过调用相应的构造函数，可以轻松创建模型对象，如下所示：

```py
import torchvision

model = torchvision.models.vgg16(pretrained=False)
```

这段代码创建了一个带有随机权重的 VGG16 模型，因为没有使用预训练权重。通过使用类似的构造函数并设置适当的参数，可以实例化许多不同的计算机视觉模型。Torchvision 使用 PyTorch 的`torch.utils.model_zoo`提供预训练模型。可以通过传递`pretrained=True`来构建这些模型。

表 8-11 提供了 Torchvision 中包含的模型的全面列表，按类别分类。这些模型在研究界广为人知，表中包含了与每个模型相关的研究论文的参考文献。

表 8-11. Torchvision 模型

| 模型 | 论文 |
| --- | --- |
| **分类** |  |
| AlexNet | “用于并行化卷积神经网络的一个奇怪技巧,” 作者：Alex Krizhevsky |
| VGG | “用于大规模图像识别的非常深度卷积网络,” 作者：Karen Simonyan 和 Andrew Zisserman |
| ResNet | “用于图像识别的深度残差学习,” 作者：Kaiming He 等 |
| SqueezeNet | “SqueezeNet: AlexNet 级别的准确性，参数减少 50 倍，模型大小<0.5MB,” 作者：Forrest N. Iandola 等 |
| DenseNet | “密集连接的卷积网络,” 作者：Gao Huang 等 |
| Inception v3 | “重新思考计算机视觉中的 Inception 架构,” 作者：Christian Szegedy 等 |
| GoogLeNet | “使用卷积深入研究,” 作者：Christian Szegedy 等 |
| ShuffleNet v2 | “ShuffleNet V2: 高效 CNN 架构设计的实用指南,” 作者：马宁宁等 |
| MobileNet v2 | “MobileNetV2: 反向残差和线性瓶颈,” 作者：Mark Sandler 等 |
| ResNeXt | “用于深度神经网络的聚合残差变换,” 作者：Saining Xie 等 |
| Wide ResNet | “宽残差网络,” 作者：Sergey Zagoruyko 和 Nikos Komodakis |
| MNASNet | “MnasNet: 面向移动设备的神经架构搜索,” 作者：Mingxing Tan 等 |
| **语义分割** |  |
| FCN ResNet50 | “用于语义分割的全卷积网络,” 作者：Jonathan Long 等 |
| FCN ResNet101 | 参见上文 |
| DeepLabV3 ResNet50 | “重新思考空洞卷积用于语义图像分割,” 作者：Liang-Chieh Chen 等 |
| DeepLabV3 ResNet101 | 参见上文 |
| **目标检测** |  |
| Faster R-CNN ResNet-50 | “FPNFaster R-CNN: 实时目标检测与区域建议网络,” 作者：Shaoqing Ren 等 |
| Mask R-CNN ResNet-50 FPN | “Mask R-CNN,” 作者：Kaiming He 等 |
| **视频分类** |  |
| ResNet 3D 18 | “仔细研究时空卷积用于动作识别,” 作者：Du Tran 等 |
| ResNet MC 18 | 参见上文 |
| ResNet (2+1)D | 参见上文 |

Torchvision 还在不断添加新的计算机视觉模型。要获取最新列表，请访问[Torchvision 文档](https://pytorch.tips/torchvision-models)。

## 变换、操作和实用程序

Torchvision 还提供了一套全面的变换、操作和实用程序集合，以帮助图像预处理和数据准备。应用变换的常见方法是形成一组变换的组合，并将这个`transforms`对象传递给数据集构造函数，如下面的代码所示：

```py
from torchvision import transforms, datasets

train_transforms = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(
                      (0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010)),
                      ])
train_data = datasets.CIFAR10(
                  root=".",
                  train=True,
                  transform=train_transforms)
```

在这里，我们创建了一个复合变换，使用`ToTensor()`将数据转换为张量，然后使用预定的均值和标准差对图像数据进行归一化。将`transform`参数设置为`train_transforms`对象会配置数据集在访问数据时应用一系列变换。

表 8-12 提供了`torchvision.transforms`中可用转换的完整列表。在本表和表 8-13 中以*`斜体`*显示的转换目前不受 TorchScript 支持。

表 8-12\. Torchvision 转换

| 转换 | 描述 |
| --- | --- |
| **操作转换** |  |
| `Compose()` | 基于其他转换序列创建一个转换 |
| `CenterCrop(size)` | 以给定大小在中心裁剪图像 |
| `ColorJitter(brightness=0,` `contrast=0,` `saturation=0, hue=0)` | 随机改变图像的亮度、对比度、饱和度和色调 |
| `FiveCrop(size)` | 将图像裁剪成四个角和中心裁剪 |
| `Grayscale(num_output_channels=1)` | 将彩色图像转换为灰度图像 |
| `Pad(`*`padding`*`,` `fill=0,` `padding_mode=constant)` | 使用给定值填充图像的边缘 |
| `RandomAffine(`*`degrees`*`,` `translate=None,` `scale=None, shear=None, resample=0, fillcolor=0)` | 随机应用仿射变换 |
| `RandomApply(transforms, p=0.5)` | 以给定概率随机应用一系列转换 |
| `RandomCrop(`*`size`*`,` `padding=None, pad_if_needed=False, fill=0,` `padding_mode=constant)` | 在随机位置裁剪图像 |
| `RandomGrayscale(p=0.1)` | 以给定概率随机将图像转换为灰度图像 |
| `RandomHorizontalFlip(p=0.5)` | 以给定概率随机水平翻转图像 |
| `RandomPerspective(distor⁠⁠tion_​scale=0.5, p=0.5,` `interpolation=2,` `fill=0)` | 应用随机透视变换 |
| `RandomResizedCrop(`*`size`*`,` `scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333),` `interpolation=2)` | 使用随机大小和长宽比调整图像 |
| `RandomRotation(`*`degrees`*`,` `resample=False,` `expand=False,` `center=None,` `fill=None)` | 随机旋转图像 |
| `RandomVerticalFlip(p=0.5)` | 以给定概率随机垂直翻转图像 |
| `Resize(`*`size`*`,` `interpolation=2)` | 将图像调整为随机大小 |
| `TenCrop(`*`size`*`,` `vertical_flip=False)` | 将图像裁剪成四个角和中心裁剪，并额外提供每个的翻转版本 |
| `GaussianBlur(`*`kernel_size`*`,` `sigma=(0.1, 2.0))` | 使用随机核应用高斯模糊 |
| **转换转换** |  |
| *`ToPILImage(mode=None)`* | 将张量或`numpy.ndarray`转换为 PIL 图像 |
| *`ToTensor()`* | 将 PIL 图像或`ndarray`转换为张量 |
| **通用转换** |  |
| *`Lambda(lambda)`* | 将用户定义的`lambda`作为转换应用 |

大多数转换可以在张量或 PIL 格式的图像上进行操作，其形状为`[..., C, H, W]`，其中`...`表示任意数量的前导维度。然而，一些转换只能在 PIL 图像或张量图像数据上操作。

在表 8-13 中列出的转换仅适用于 PIL 图像。这些转换目前不受 TorchScript 支持。

表 8-13\. Torchvision 仅支持 PIL 的转换

| 转换 | 描述 |
| --- | --- |
| *`RandomChoice(transforms)`* | 从列表中随机选择一个转换应用 |
| *`RandomOrder(transforms)`* | 以随机顺序应用一系列转换 |

在表 8-14 中列出的转换仅适用于张量图像。

表 8-14\. Torchvision 仅支持张量的转换

| 转换 | 描述 |
| --- | --- |
| `LinearTransformation(​trans⁠⁠formation_matrix,` `mean_vector)` | 根据离线计算的方形变换矩阵和`mean_vector`对张量图像应用线性变换。 |
| `Normalize(mean, std, inplace=False)` | 使用给定的均值和标准差对张量图像进行归一化。 |
| `RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)` | 随机选择一个矩形区域并擦除其像素。 |
| `ConvertImageDtype(dtype: torch.dtype)` | 将张量图像转换为新的数据类型，并自动缩放其值以匹配该类型 |

###### 注意

在为 C++使用脚本转换时，请使用`torch.nn.Sequential()`而不是`torchvision.transforms.Compose()`。以下代码显示了一个示例：

```py
>>> transforms = torch.nn.Sequential(
        transforms.CenterCrop(10),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224,
            0.225)),
        )

>>> scripted_transforms = torch.jit.script(transforms)

```

在前面的表中列出的许多转换包含用于指定参数的随机数生成器。例如，`RandomResizedCrop()`将图像裁剪为随机大小和长宽比。

Torchvision 还提供了功能性转换作为`torchvision.transforms.functional`包的一部分。您可以使用这些转换来执行具有您选择的特定参数集的转换。例如，您可以调用`torchvision.transforms.functional.adjust_brightness()`来调整一个或多个图像的亮度。

表 8-15 提供了支持的功能性转换列表。

表 8-15。Torchvision 功能性转换

| 功能性转换和实用程序 |
| --- |
| `adjust_brightness(`*`img: torch.Tensor,`* *`brightness_factor: float`*`)` |
| `adjust_contrast(`*`img: torch.Tensor,`* *`contrast_factor: float`*`)` |
| `adjust_gamma(`*`img: torch.Tensor, gamma: float,`* *`gain: float = 1`*`)` |
| `adjust_hue(`*`img: torch.Tensor, hue_factor: float`*`)` → *`torch.Tensor`* |
| `adjust_saturation(`*`img: torch.Tensor,`* *`saturation_factor: float`*`)` |
| `affine(`*`img: torch.Tensor, angle: float,`* *`translate: List[int],`* *`scale: float, shear: List[float],`* *`resample: int = 0, fillcolor: Optional[int] = None`*`)` |
| `center_crop(`*`img: torch.Tensor, output_size: List[int]`*`)` |
| `convert_image_dtype(`*`image: torch.Tensor,`* *`dtype: torch.dtype = torch.float32`*`)` |
| `crop(`*`img: torch.Tensor, top: int, left: int,`* *`height: int, width: int`*`)` |
| `erase(`*`img: torch.Tensor, i: int, j: int, h: int,`* *`w: int, v: torch.Tensor, inplace: bool = False`*`)` |
| `five_crop(`*`img: torch.Tensor, size: List[int]`*`)` |
| `gaussian_blur(`*`img: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None`*`)` |
| `hflip(`*`img: torch.Tensor`*`)` |
| `normalize(`*`tensor: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False`*`)` |
| `pad(`*`img: torch.Tensor, padding: List[int],`* *`fill: int = 0, padding_mode: str = constant`*`)` |
| `perspective(`*`img: torch.Tensor, startpoints: List[List[int]], endpoints: List[List[int]],`* *`interpolation: int = 2, fill: Optional[int] = None`*`)` |
| `pil_to_tensor(`*`pic`*`)` |
| `resize(`*`img: torch.Tensor, size: List[int],`* *`interpolation: int = 2`*`)` |
| `resized_crop(`*`img: torch.Tensor, top: int, left: int, height: int, width: int, size: List[int],`* *`interpolation: int = 2`*`)` |
| `rgb_to_grayscale(`*`img: torch.Tensor,`* *`num_output_channels: int = 1`*`)` |
| `rotate(`*`img: torch.Tensor, angle: float,`* *`resample: int = 0, expand: bool = False,`* *`center: Optional[List[int]] = None,`* *`fill: Optional[int] = None`*`)` |
| `ten_crop(`*`img: torch.Tensor, size: List[int],`* *`vertical_flip: bool = False`*`)` |
| `to_grayscale(`*`img, num_output_channels=1`*`)` |
| `to_pil_image(`*`pic, mode=None`*`)` |
| `to_tensor(`*`pic`*`)` |
| `vflip(`*`img: torch.Tensor`*`)` |
| `utils.save_image(`*`tensor: Union[torch.Tensor, List[torch.Tensor]], fp: Union[str, pathlib.Path,`* *`BinaryIO],`* *`nrow: int = 8, padding: int = 2, normalize: bool = False, range: Optional[Tuple[int, int]] = None, scale_each: bool = False, pad_value: int = 0, format: Optional[str] = None`*`)` |
| `utils.make_grid(`*`tensor: Union[torch.Tensor, List[torch.Tensor]], nrow: int = 8, padding: int = 2, normalize: bool = False, range: Optional[Tuple[int, int]] = None, scale_each: bool = False,`* *`pad_value: int = 0`*`)` |

如上表所示，Torchvision 提供了一组强大的功能操作，可用于处理图像数据。每个操作都有自己的一组参数，用于强大的控制。

此外，Torchvision 提供了用于简化 I/O 和操作的函数。表 8-16 列出了其中一些函数。

表 8-16\. Torchvision 的 I/O 和操作函数

| 函数 |
| --- |
| **视频** |
| `io.read_video(`*`filename: str, start_pts: int = 0, end_pts: Optional[float] = None, pts_unit: str = pts`*`)` |
| `io.read_video_timestamps9`*`filename: str, pts_unit: str = pts`*`)` |
| `io.write_video9`*`filename: str, video_array:`* *`torch.Tensor,`* *`_fps: float, video_codec: str = *libx264*, options: Optional[Dict[str, Any]] = None`*`)` |
| **细粒度视频** |
| `io.VideoReader(`*`path, stream=video`*`)` |
| **图像** |
| `io.decode_image(`*`input: torch.Tensor`*`)` |
| `io.encode_jpeg(`*`input: torch.Tensor, quality: int = 75`*`)` |
| `io.read_image(`*`path: str`*`)` |
| `io.write_jpeg(`*`input: torch.Tensor, filename: str,`* *`quality: int = 75`*`)` |
| `io.encode_png(`*`input: torch.Tensor, compression_level: int = 6`*`)` |
| `io.write_png(`*`input: torch.Tensor, filename: str,`* *`compression_level: int = 6`*`)` |

上述函数是为了让您能够快速读取和写入多种格式的视频和图像文件。它们让您能够加快图像和视频处理的速度，而无需从头开始编写这些函数。

如您所见，Torchvision 是一个功能丰富、得到良好支持且成熟的 PyTorch 包。本节提供了 Torchvision API 的快速参考。在下一节中，我们将探索另一个用于 NLP 和文本应用的流行 PyTorch 包 Torchtext。

# Torchtext 用于 NLP

Torchtext 包含一系列用于数据处理的实用工具和流行的 NLP 数据集。Torchtext API 与 Torchvision API 略有不同，但整体方法是相同的。

## 创建数据集对象

首先创建一个数据集，并描述一个预处理流水线，就像我们在 Torchvision 转换中所做的那样。Torchtext 提供了一组知名数据集。例如，我们可以加载 IMDb 数据集，如下面的代码所示：

```py
from torchtext.datasets import IMDB

train_iter, test_iter = \
  IMDB(split=('train', 'test'))

next(train_iter)
# out:
# ('neg',
# 'I rented I AM CURIOUS-YELLOW ...)
```

我们自动创建一个迭代器，并可以使用`next()`访问数据。

###### 警告

Torchtext 在 PyTorch 1.8 中显著改变了其 API。如果本节中的代码返回错误，您可能需要升级 PyTorch 的版本。

## 预处理数据

Torchtext 还提供了预处理文本和创建数据管道的功能。预处理任务可能包括定义分词器、词汇表和数值嵌入。

在新的 Torchtext API 中，您可以使用`data.get_tokenizer()`函数访问不同的分词器，如下面的代码所示：

```py
from torchtext.data.utils \
  import get_tokenizer

tokenizer = get_tokenizer('basic_english')
```

在新 API 中创建词汇表也是灵活的。您可以直接使用`Vocab`类构建词汇表，如下面的代码所示：

```py
from collections import Counter
from torchtext.vocab import Vocab

train_iter = IMDB(split='train')
counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter,
              min_freq=10,
              specials=('<unk>',
                        '<BOS>',
                        '<EOS>',
                        '<PAD>'))
```

如您所见，我们可以设置`min_freq`来指定词汇表中的截止频率。我们还可以将特殊符号分配给特殊符号，比如`<BOS>`和`<EOS>`，如`Vocab`类的构造函数所示。

另一个有用的功能是为文本和标签定义转换，如下面的代码所示：

```py
text_transform = lambda x: [vocab['<BOS>']] \
  + [vocab[token] \
     for token in tokenizer(x)] + [vocab['<EOS>']]

label_transform = lambda x: 1 \
  if x == 'pos' else 0

print(text_transform("programming is awesome"))
# out: [1, 8320, 12, 1156, 2]
```

我们将文本字符串传递给我们的转换，然后使用词汇表和分词器对数据进行预处理。

## 创建用于批处理的数据加载器

现在我们已经加载并预处理了数据，最后一步是创建一个数据加载器，以从数据集中对数据进行采样和批处理。我们可以使用以下代码创建一个数据加载器：

```py
from torch.utils.data import DataLoader

train_iter = IMDB(split='train')
train_dataloader = DataLoader(
    list(train_iter),
    batch_size=8,
    shuffle=True)

# for text, label in train_dataloader
```

您可能会注意到，这段代码与我们在 Torchvision 中创建数据加载器的代码相似。我们不是传入数据集对象，而是将`train_iter`转换为`list()`传入。`DataLoader()`构造函数还接受`batch_sampler`和`collate_fcn`参数（在上述代码中未显示；请参阅[文档](https://pytorch.tips/data)），因此您可以自定义数据集的采样和整理方式。创建数据加载器后，使用它来训练您的模型，如上述代码注释所示。

Torchtext 具有许多有用的功能。让我们探索 API 中提供的内容。

## 数据（torchtext.data）

`torchtext.data` API 提供了在 PyTorch 中创建基于文本的数据集对象的函数。表 8-17 列出了`torchtext.data`中可用的函数。

表 8-17\. Torchtext 数据

| 函数 | 描述 |
| --- | --- |
| **`torchtext.data.utils`** |  |
| `get_tokenizer(`*`tokenizer,`* *`language=en`*`)` | 为字符串句子生成一个分词器函数 |
| `ngrams_itera⁠⁠tor(`*`token_​list, ngrams`*`)` | 返回一个迭代器，产生给定标记及其 ngrams |
| **`torchtext.data.functional`** |  |
| `generate_sp_model(​`*`file⁠⁠name, vocab_size=20000, model_type=unigram, model_prefix=m_user`*`)` | 训练一个`SentencePiece`分词器 |
| `load_sp_model(`*`spm`*`)` | 从文件加载一个`SentencePiece`模型 |
| `sentencepiece_​numeri⁠⁠cal⁠⁠izer(`*`sp_model`*`)` | 创建一个生成器，接受文本句子并根据`SentencePiece`模型输出相应的标识符 |
| `sentencepiece_​token⁠⁠izer(`*`sp_model`*`)` | 创建一个生成器，接受文本句子并根据`SentencePiece`模型输出相应的标记 |
| `custom_replace(`*`replace_​pat⁠⁠tern`*`)` | 作为一个转换器，将文本字符串转换 |
| `simple_space_split(​`*`itera⁠⁠tor`*`)` | 作为一个转换器，通过空格分割文本字符串 |
| `numerical⁠⁠ize_tokens_​from_iterator(`*`vocab,`* *`iterator,`* *`removed_tokens=None`*`)` | 从一个标记迭代器中产生一个标识符列表，使用`vocab` |
| **`torchtext.data.metrics`** |  |
| `bleu_score(`*`candidate_​cor⁠⁠pus, references_corpus, max_n=4, weights=[0.25, 0.25, 0.25, 0.25]`*`)` | 计算候选翻译语料库和参考翻译语料库之间的 BLEU 分数 |

正如您所看到的，`torchtext.data`子模块支持根据字段创建数据集对象的函数，以及加载、预处理和迭代批次。接下来让我们看看 Torchtext 库中提供的 NLP 数据集有哪些。

## 数据集（torchtext.datasets）

Torchtext 支持从流行论文和研究中加载数据集。您可以找到用于语言建模、情感分析、文本分类、问题分类、蕴涵、机器翻译、序列标记、问题回答和无监督学习的数据集。

表 8-18 提供了 Torchtext 中包含的数据集的全面列表。

表 8-18\. Torchtext 数据集

| 函数 | 描述 |
| --- | --- |
| **文本分类** |  |
| `TextClassificationDataset(`*`vocab, data, labels`*`)` | 通用文本分类数据集 |
| `IMDB(`*`root=*.data*, split=`*`(`*`*train*, *test*`*`))` | 包含来自 IMDb 的 50,000 条评论，标记为正面或负面的二元情感分析数据集 |
| `AG_NEWS(`*`root=*.data*, split=`*`(`*`*train*, *test*`*`))` | 包含四个主题标记的新闻文章数据集 |
| `SogouNews(`*`root=*.data*, split=(*train*, *test*`*`))` | 包含五个主题标记的新闻文章数据集 |
| `DBpedia(`*`root=*.data*, split=(*train*, *test*`*`))` | 包含 14 个类别标记的新闻文章数据集 |
| `YelpReviewPolarity(`*`root=*.data*, split=(*train*, *test*`*`))` | 包含 50 万条 Yelp 评论的二元分类数据集 |
| `YelpReviewFull(`*`root=*.data*, split=(*train*, *test*`*`))` | 包含 50 万条 Yelp 评论的数据集，具有细粒度（五类）分类 |
| `YahooAnswers(`*`root=*.data*, split=`*`(`*`*train*, *test*`*`))` | 包含 10 个不同类别的 Yahoo 答案的数据集 |
| `AmazonReviewPolarity(`*`root=*.data*, split=`*`(`*`*train*, *test*`*`))` | 包含亚马逊评论的数据集，具有二元分类 |
| `AmazonReviewFull(`*`root=*.data*, split=`*`(`*`*train*, *test*`*`))` | 包含亚马逊评论的数据集，具有细粒度（五类）分类 |
| **语言建模** |  |
| `LanguageModelingDataset(`*`path, text_field, newline_eos=True,`* *`encoding=*utf-8*, **kwargs`*`)` | 通用语言建模数据集类 |
| `WikiText2(`*`root=*.data*, split=`*`(`*`*train*, *valid*, *test*`*`))` | WikiText 长期依赖语言建模数据集，从维基百科上经过验证的“优秀”和“精选”文章中提取的超过 1 亿个标记的集合 |
| `WikiText103(`*`root=*.data*, split=`*`(`*`*train*, *valid*, *test*`*`))` | 更大的 WikiText 数据集 |
| `PennTreebank(`*`root=*.data*, split=`*`(`*`*train*, *valid*, *test*`*`))` | 最初为词性（POS）标记创建的相对较小的数据集 |
| **机器翻译** |  |
| `TranslationDataset(`*`path, exts, fields, **kwargs`*`)` | 通用翻译数据集类 |
| `IWSLT2016(`*`root=*.data*, split=`*`(`*`*train*, *valid*, *test*`*`)`*`,`* *`language_pair=`*`(`*`*de*, *en*`)`*`,`* *`valid_set=*tst2013*, test_set=*tst2014*`*`)` | 国际口语翻译会议（IWSLT）2016 TED 演讲翻译任务 |
| `IWSLT2017(`*`root=*.data*, split=`*`(`*`*train*, *valid*, *test*`*`)`*`,`* *`language_pair=`*`(`*`*de*, *en*`*`))` | 国际口语翻译会议（IWSLT）2017 TED 演讲翻译任务 |
| **序列标记** |  |
| `SequenceTaggingDataset(`*`path, fields, encoding=*utf-8*,`* *`separator=*t*, **kwargs`*`)` | 通用序列标记数据集类 |
| `UDPOS(`*`root=*.data*, split=`*`(`*`*train*, *valid*, *test*`*`))` | 通用依存关系版本 2 词性标记数据 |
| `CoNLL2000Chunking(`*`root=*.data*, split=`*`(`*`*train*, *test*`*`))` | 下载和加载 Conference on Computational Natural Language Learning (CoNLL) 2000 分块数据集的命令 |
| **问答** |  |
| `SQuAD1(`*`root=*.data*, split=`*`(`*`*train*, *dev*`*`))` | 创建斯坦福问答数据集（SQuAD）1.0 数据集，这是一个由众包工作者在一组维基百科文章上提出的问题的阅读理解数据集 |
| `SQuAD2(`*`root=.data, split=`*`(`*`train, dev`*`))` | 创建斯坦福问答数据集（SQuAD）2.0 数据集，该数据集通过添加超过 5 万个无法回答的问题扩展了 1.0 数据集 |

Torchtext 开发人员始终在添加新的数据集。要获取最新列表，请访问[Torchtext 数据集文档](https://pytorch.tips/torchtext-datasets)。

加载数据后，无论是来自现有数据集还是您创建的数据集，您都需要在训练模型和运行推理之前将文本数据转换为数值数据。为此，我们使用提供映射以执行这些转换的词汇表和词嵌入。接下来，我们将检查用于支持词汇表的 Torchtext 函数。

## 词汇表（torchtext.vocab）

Torchtext 提供了通用类和特定类来支持流行的词汇表。表 8-19 提供了`torchtext.vocab`中的类列表，以支持词汇表的创建和使用。

表 8-19. Torchtext 词汇表

| 功能 | 描述 |
| --- | --- |
| **词汇表类** |  |
| `Vocab(counter, max_size=None, min_freq=1,` `specials=(<unk>,` `<pad>),` *`vectors=None,`* *`unk_init=None,`* *`vectors_cache=None,`* *`specials_first=True`*`)` | 定义将用于数值化字段的词汇表对象 |
| `SubwordVocab(`*`counter, max_size=None,`* *`specials=<pad>,`* *`vectors=None,`* *`unk_init=<method zero_of torch._C._TensorBase objects>`*`)` | 从`collections.Counter`创建一个`revtok`子词汇表 |
| `Vectors(`*`name, cache=None, url=None, unk_init=None,`* *`max_vectors=None`*`)` | 用于词向量嵌入的通用类 |
| **预训练词嵌入** |  |
| `GloVe(`*`name=*840B*, dim=300, **kwargs`*`)` | 全局向量（GloVe）模型，用于分布式词表示，由斯坦福大学开发 |
| `FastText(`*`language=en, **kwargs`*`)` | 294 种语言的预训练词嵌入，由 Facebook 的 AI 研究实验室创建 |
| `CharNGram(`*`**kwargs`*`)` | CharNGram 嵌入，一种学习基于字符的组合模型以嵌入文本序列的简单方法 |
| **杂项** |  |
| `build_vocab_from_​itera⁠⁠tor(​`*`iter⁠⁠ator, num_lines=None`*`)` | 通过循环遍历迭代器构建词汇表 |

正如您所看到的，Torchtext 提供了一套强大的功能，支持基于文本的建模和 NLP 研究。欲了解更多信息，请访问[Torchtext 文档](https://pytorch.tips/torchtext)。

无论您是为 NLP、计算机视觉或其他领域开发深度学习模型，能够在开发过程中可视化模型、数据和性能指标是很有帮助的。在下一节中，我们将探索另一个强大的用于可视化的包，称为 TensorBoard。

# 用于可视化的 TensorBoard

TensorBoard 是一个可视化工具包，包含在 PyTorch 的主要竞争深度学习框架 TensorFlow 中。PyTorch 没有开发自己的可视化工具包，而是与 TensorBoard 集成，并原生地利用其可视化能力。

使用 TensorBoard，您可以可视化学习曲线、标量数据、模型架构、权重分布和 3D 数据嵌入，以及跟踪超参数实验结果。本节将向您展示如何在 PyTorch 中使用 TensorBoard，并提供 TensorBoard API 的参考。

TensorBoard 应用程序在本地或远程服务器上运行，显示和用户界面在浏览器中运行。我们还可以在 Jupyter Notebook 或 Google Colab 中运行 TensorBoard。

我将在本书中使用 Colab 来演示 TensorBoard 的功能，但在本地或远程云中运行它的过程非常类似。Colab 预装了 TensorBoard，您可以直接在单元格中使用魔术命令运行它，如下所示的代码：

```py
%load_ext tensorboard
%tensorboard --logdir ./runs/
```

首先我们加载`tensorboard`扩展，然后运行`tensorboard`并指定保存事件文件的日志目录。事件文件保存了来自 PyTorch 的数据，将在 TensorBoard 应用程序中显示。

由于我们尚未创建任何事件文件，您将看到一个空的显示，如图 8-1 所示。

![“TensorBoard 应用程序”](img/ptpr_0801.png)

###### 图 8-1\. TensorBoard 应用程序

通过单击右上角菜单中 INACTIVE 旁边的箭头，您将看到可能的显示选项卡。一个常用的显示选项卡是 SCALARS 选项卡。此选项卡可以显示随时间变化的任何标量值。我们经常使用 SCALARS 显示来查看损失和准确率训练曲线。让我们看看如何在您的 PyTorch 代码中保存标量值以供 TensorBoard 使用。

###### 注意

PyTorch 与 TensorBoard 的集成最初是由一个名为 TensorBoardX 的开源项目实现的。自那时起，TensorBoard 支持已集成到 PyTorch 项目中，作为`torch.utils.tensorboard`包，并由 PyTorch 开发团队积极维护。

首先让我们导入 PyTorch 的 TensorBoard 接口，并设置 PyTorch 以便与 TensorBoard 一起使用，如下所示的代码：

```py
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter() ![1](Images/1.png)
```

①

默认情况下，写入器将输出到*./runs/*目录。

我们只需从 PyTorch 的`tensorboard`包中导入`SummaryWriter`类，并实例化一个`SummaryWriter`对象。要将数据写入 TensorBoard，我们只需要调用`SummaryWriter`对象的方法。在模型训练时保存我们的损失数值，我们使用`add_scalar()`方法，如下面的代码所示：

```py
N_EPOCHS = 10
for epoch in range(N_EPOCHS):

    epoch_loss = 0.0
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print("Epoch: {} Loss: {}".format(epoch,
        epoch_loss/len(trainloader)))
    writer.add_scalar('Loss/train',
        epoch_loss/len(trainloader), epoch) ![1](Images/1.png)
```

①

将`loss.item()`作为事件记录到`tensorboard`。

这只是一个示例训练循环。您可以假设`model`已经被定义，`trainloader`已经被创建。代码不仅在每个 epoch 打印损失，还将其记录到`tensorboard`事件中。我们可以刷新之前单元格中的 TensorBoard 应用程序，或者使用`%tensorboard`命令创建另一个单元格。

## 使用 SCALARS 查看学习曲线

TensorBoard 提供了绘制一个或多个标量值随时间变化的能力。在深度学习开发中，这对于显示模型训练时的指标非常有用。通过查看损失或准确率等指标，您可以轻松地看到您的模型训练是否稳定并持续改进。

图 8-2 展示了使用 TensorBoard 显示学习曲线的示例。

您可以通过滑动平滑因子与显示进行交互，也可以通过将鼠标悬停在绘图上查看每个 epoch 的曲线。TensorBoard 允许您应用平滑处理以消除不稳定性并显示整体进展。

![“使用 TensorBoard 可视化学习曲线”](img/ptpr_0802.png)

###### 图 8-2\. TensorBoard 学习曲线

## 使用 GRAPHS 查看模型架构

TensorBoard 的另一个有用功能是使用图形可视化您的深度学习模型。要将图形保存到事件文件中，我们将使用`add_graph()`方法，如下面的代码所示：

```py
model = vgg16(preTrained=True)
writer.add_graph(model)
```

在这段代码中，我们实例化了一个 VGG16 模型，并将模型写入事件文件。我们可以通过刷新现有的 TensorBoard 单元格或创建一个新的单元格来显示模型图。图 8-3 展示了 TensorBoard 中的图形可视化工具。

![“使用 TensorBoard 可视化模型图”](img/ptpr_0803.png)

###### 图 8-3\. TensorBoard 模型图

该图是交互式的。您可以单击每个模块并展开以查看底层模块。这个工具对于理解现有模型并验证您的模型图是否符合其预期设计非常有用。

## 使用图像、文本和投影仪的数据

您还可以使用 TensorBoard 查看不同类型的数据，例如图像、文本和 3D 嵌入。在这些情况下，您将分别使用`add_image()`、`add_text()`和`add_projection()`方法将数据写入事件文件。

图 8-4 显示了来自 Fashion-MNIST 数据集的一批图像数据。

通过检查图像数据的批次，您可以验证数据是否符合预期，或者识别数据或结果中的错误。TensorBoard 还提供了监听音频数据、显示文本数据以及查看多维数据或数据嵌入的 3D 投影的能力。

![“使用 TensorBoard 可视化图像数据”](img/ptpr_0804.png)

###### 图 8-4\. TensorBoard 图像显示

## 使用 DISTRIBUTIONS 和 HISTOGRAMS 查看权重分布

TensorBoard 的另一个有用功能是显示分布和直方图。这使您可以查看大量数据以验证预期行为或识别问题。

模型开发中的一个常见任务是确保避免*梯度消失问题*。当模型权重变为零或接近零时，梯度消失就会发生。当这种情况发生时，神经元基本上会死亡，无法再更新。

如果我们可视化我们的权重分布，很容易看到大部分权重值已经达到零。

图 8-5 展示了 TensorBoard 中的 DISTRIBUTIONS 选项卡。在这里，我们可以检查我们的权重值的分布。

如您在图 8-5 中所见，TensorBoard 可以以 3D 显示分布，因此很容易看到分布随时间或每个时期如何变化。

![“使用 TensorBoard 的权重分布”](img/ptpr_0805.png)

###### 图 8-5\. TensorBoard 权重分布

## 具有 HPARAMS 的超参数

在运行深度学习实验时，很容易迷失不同超参数集的跟踪，以尝试假设。TensorBoard 提供了一种在每次实验期间跟踪超参数值并将值及其结果制表的方法。

图 8-6 显示了我们如何跟踪实验及其相应的超参数和结果的示例。

在 HPARAMS 选项卡中，您可以以表格视图、平行坐标视图或散点图矩阵视图查看结果。每个实验由其会话组名称、超参数（如丢失百分比和优化器算法）以及结果指标（如准确性）标识。HPARAMS 表格可帮助您跟踪实验和结果。

当您完成向 TensorBoard 事件文件写入数据时，应使用`close()`方法，如下所示：

```py
writer.close()
```

这将调用析构函数并释放用于摘要写入器的任何内存。

![“在 TensorBoard 中跟踪超参数”](img/ptpr_0806.png)

###### 图 8-6\. TensorBoard 超参数跟踪

## TensorBoard API

PyTorch TensorBoard API 非常简单。它作为`torch.utils.tensorboard`的一部分包含在`torch.utils`包中。表 8-20 显示了用于将 PyTorch 与 TensorBoard 接口的函数的全面列表。

表 8-20\. PyTorch TensorBoard API

| 方法 | 描述 |
| --- | --- |
| `SummaryWriter(`*`log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix=''`*`)` | 创建`SummaryWriter`对象 |
| `flush()` | 将事件文件刷新到磁盘；确保所有待处理事件都已写入磁盘 |
| `close()` | 释放`SummaryWriter`对象并关闭事件文件 |
| `add_scalar(`*`tag, scalar_value, global_step=None, walltime=None`*`)` | 将标量写入事件文件 |
| `add_scalars(`*`main_tag, tag_scalar_dict, global_step=None, walltime=None`*`)` | 将多个标量写入事件文件以在同一图中显示多个标量 |
| `add_custom_scalars(`*`layout`*`)` | 通过收集标量中的图表标签创建特殊图表 |
| `add_histogram(`*`tag, values, global_step=None, bins=tensorflow, walltime=None, max_bins=None`*`)` | 为直方图显示写入数据 |
| `add_image(`*`tag, img_tensor, global_step=None, walltime=None, dataformats=CHW`*`)` | 写入图像数据 |
| `add_images(`*`tag, img_tensor, global_step=None, walltime=None, dataformats=NCHW`*`)` | 将多个图像写入同一显示 |
| `add_figure(`*`tag, figure, global_step=None, close=True, walltime=None`*`)` | 将`matplotlib`类型的图绘制为图像 |
| `add_video(`*`tag, vid_tensor, global_step=None, fps=4, walltime=None``)`* | 写入视频 |
| `add_audio(`*`tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None`*`)` | 将音频文件写入事件摘要 |
| `add_text(`*`tag, text_string, global_step=None, walltime=None`*`)` | 将文本数据写入摘要 |
| `add_graph(`*`model, input_to_model=None, verbose=False`*`)` | 将模型图或计算图写入摘要 |
| `add_embedding(`*`mat, metadata=None, label_img=None, global_step=None, tag=default, metadata_header=None`*`)` | 将嵌入投影仪数据写入摘要 |
| `add_pr_curve(`*`tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None`*`)` | 在不同阈值下写入精度/召回率曲线 |
| `add_mesh(`*`tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None`*`)` | 将网格或 3D 点云添加到 TensorBoard |
| `add_hparams(`*`hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None`*`)`：向 TensorBoard 中添加一组超参数以进行比较。 |

如表 8-20 所示，API 很简单。您可以使用`SummaryWriter()`、`flush()`和`close()`方法来管理写入对象，并使用其他函数向 TensorBoard 事件文件添加数据。

有关 TensorBoard PyTorch API 的更多详细信息，请访问 TensorBoard API 文档。有关如何使用 TensorBoard 应用程序本身的更多详细信息，请访问 TensorBoard 文档。

TensorBoard 通过提供可视化工具解决了在 PyTorch 中开发深度学习模型时的一个主要挑战。另一个主要挑战是跟上最新研究和最先进的解决方案。研究人员经常需要重现结果并利用代码来对比自己的设计。在接下来的部分中，我们将探讨 Papers with Code，这是一个您可以使用的资源来解决这个问题。

# Papers with Code

Papers with Code（PwC）是一个网站，它整理了机器学习研究论文及其相应的代码，这些代码通常是用 PyTorch 编写的。PwC 允许您轻松重现实验并扩展当前研究，该网站还允许您找到给定机器学习主题的表现最佳的研究论文。例如，想要找到最佳的图像分类模型及其代码吗？只需点击图像分类瓷砖，您将看到研究领域的摘要以及 GitHub 上相应论文和代码的基准和链接。图 8-7 展示了图像分类的示例列表。

“Papers With Code Image Classification Listing”图片

###### 图 8-7. Papers with Code

PwC 并不是一个专门的 PyTorch 项目；然而，PwC 提供的大多数代码都使用 PyTorch。它可能有助于您了解当前最先进的研究并解决您在深度学习和人工智能方面的问题。在 PwC 网站上探索更多。

# 额外的 PyTorch 资源

阅读完这本书后，您应该对 PyTorch 及其功能有很好的理解。然而，总是有新的方面可以探索和实践。在本节中，我将提供一些额外资源的列表，您可以查看以了解更多信息，并提升您在 PyTorch 中的技能。

## 教程

PyTorch 网站提供了大量的文档和教程。如果您正在寻找更多的代码示例，这个资源是一个很好的起点。图 8-8 展示了 PyTorch 教程网站，您可以选择标签来帮助您找到感兴趣的教程。

“PyTorch 教程网站”图片

###### 图 8-8. PyTorch 教程

该网站包括一个 60 分钟的闪电战、PyTorch 食谱、教程和 PyTorch 备忘单。大多数代码和教程都可以在 GitHub 上找到，并且可以在 VS Code、Jupyter Notebook 和 Colab 中运行。

60 分钟闪电战是一个很好的起点，可以帮助您恢复技能或复习 PyTorch 的基础知识。PyTorch 食谱是关于如何使用特定 PyTorch 功能的简短、可操作的示例。PyTorch 教程比食谱稍长，由多个步骤组成，以实现或演示一个结果。

目前，您可以找到与以下主题相关的教程：

+   音频

+   最佳实践

+   C++

+   CUDA

+   扩展 PyTorch

+   FX

+   前端 API

+   入门

+   图像/视频

+   可解释性

+   内存格式

+   移动

+   模型优化

+   并行和分布式训练

+   生产

+   性能分析

+   量化

+   强化学习

+   TensorBoard

+   文本

+   TorchScript

PyTorch 团队不断添加新资源，这个列表肯定会发生变化。有关更多信息和最新教程，请访问 PyTorch 教程网站。

## 书籍

教程是学习的好方法，但也许您更喜欢阅读有关 PyTorch 的更多信息，并从多位作者的不同视角获得不同观点。表 8-21 提供了与 PyTorch 相关的其他书籍列表。

表 8-21\. PyTorch 书籍

| 书籍 | 出版商，年份 | 摘要 |
| --- | --- | --- |
| *云原生机器学习* by Carl Osipov | Manning, 2021 | 学习如何在 AWS 上部署 PyTorch 模型 |
| *使用 fastai 和 PyTorch 进行编码人员的深度学习* by Jeremy Howard 和 Sylvain Gugger | O’Reilly, 2020 | 学习如何在没有博士学位的情况下构建人工智能应用程序 |
| *使用 PyTorch 进行深度学习* by Eli Stevens 等 | Manning, 2019 | 学习如何使用 Python 工具构建、训练和调整神经网络 |
| *使用 PyTorch 进行深度学习* by Vishnu Subramanian | Packt, 2018 | 学习如何使用 PyTorch 构建神经网络模型 |
| *使用 PyTorch 1.x 进行实用生成对抗网络* by John Hany 和 Greg Walters | Packt, 2019 | 学习如何使用 Python 实现下一代神经网络，构建强大的 GAN 模型 |
| *Hands-On Natural Language Processing with PyTorch 1.x* by Thomas Dop | Packt, 2020 | 学习如何利用深度学习和自然语言处理技术构建智能的人工智能驱动的语言应用程序 |
| *使用 PyTorch 1.0 进行实用神经网络* by Vihar Kurama | Packt, 2019 | 学习如何在 PyTorch 中实现深度学习架构 |
| *使用 PyTorch 进行自然语言处理* by Delip Rao 和 Brian McMahan | O’Reilly, 2019 | 学习如何利用深度学习构建智能语言应用程序 |
| *使用 PyTorch 进行实用深度学习* by Nihkil Ketkar | Apress, 2020 | 学习如何使用 Python 优化 GAN |
| *为深度学习编程 PyTorch* by Ian Pointer | O’Reilly, 2019 | 学习如何创建和部署深度学习应用程序 |
| *PyTorch 人工智能基础* by Jibin Mathew | Packt, 2020 | 学习如何设计、构建和部署自己的 PyTorch 1.x AI 模型 |
| *PyTorch 食谱* by Pradeepta Mishra | Apress, 2019 | 学习如何在 PyTorch 中解决问题 |

## 在线课程和现场培训

如果您更喜欢在线视频课程和现场培训研讨会，您可以选择扩展您的 PyTorch 知识和技能的选项。您可以继续从 PyTorch Academy、Udemy、Coursera、Udacity、Skillshare、DataCamp、Pluralsight、edX、O’Reilly Learning 和 LinkedIn Learning 等在线讲师那里学习。一些课程是免费的，而其他课程需要付费或订阅。

表 8-22 列出了撰写时可用的 PyTorch 在线课程的选择。

表 8-22\. PyTorch 课程

| 课程 | 讲师 | 平台 |
| --- | --- | --- |
| 开始使用 PyTorch 开发 | Joe Papa | [PyTorch Academy](https://pytorchacademy.com) |
| PyTorch 基础 | Joe Papa | [PyTorch Academy](https://pytorchacademy.com) |
| 高级 PyTorch | Joe Papa | [PyTorch Academy](https://pytorchacademy.com) |
| 使用 PyTorch 进行深度学习入门 | Ismail Elezi | [DataCamp](https://www.datacamp.com/courses/deep-learning-with-pytorch) |
| PyTorch 基础 | Janani Ravi | [Pluralsight](https://www.pluralsight.com/courses/foundations-pytorch) |
| 使用 PyTorch 的深度神经网络 | IBM | [Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch) |
| 用于机器学习的 PyTorch 基础 | IBM | [edX](https://www.edx.org/course/pytorch-basics-for-machine-learning) |
| 使用 PyTorch 进行深度学习入门 | Facebook AI | [Udacity](https://www.udacity.com/course/deep-learning-pytorch%E2%80%94ud188) |
| PyTorch：深度学习和人工智能 | 懒惰的程序员 | [Udemy](https://www.udemy.com/course/pytorch-deep-learning/) |
| 用于深度学习和计算机视觉的 PyTorch | Rayan Slim 等 | [Udemy](https://www.udemy.com/course/pytorch-for-deep-learning-and-computer-vision) |
| PyTorch 入门 | Dan We | [Skillshare](https://www.skillshare.com/classes/Pytorch-for-beginners-how-machine-learning-with-pytorch-really-works/1042152565) |
| PyTorch 基础培训：深度学习 | Jonathan Fernandes | [LinkedIn Learning](https://www.linkedin.com/learning/pytorch-essential-training-deep-learning) |
| 使用 PyTorch 介绍深度学习 | Goku Mohandas 和 Alfredo Canziani | [O’Reilly Learning](https://learning.oreilly.com/videos/introduction-to-deep/9781491989944/) |

本章提供了扩展您学习、研究和开发 PyTorch 的资源。您可以将这些材料作为 PyTorch 项目和 PyTorch 生态系统中众多软件包的快速参考。当您希望扩展您的技能和知识时，可以返回本章，获取其他培训材料的想法。

恭喜您完成了这本书！您已经走了很长一段路，掌握了张量，理解了模型开发过程，并探索了使用 PyTorch 的参考设计。此外，您还学会了如何定制 PyTorch，创建自己的特性，加速训练，优化模型，并将您的神经网络部署到云端和边缘设备。最后，我们探索了 PyTorch 生态系统，调查了关键软件包如 Torchvision、Torchtext 和 TensorBoard，并了解了通过教程、书籍和在线课程扩展知识的其他方法。

无论您将来要处理什么项目，我希望您能一次又一次地返回这本书。我也希望您继续扩展您的技能，并掌握 PyTorch 的能力，开发创新的新深度学习工具和系统。不要让您的新知识和技能消失。去构建一些有趣的东西，在世界上产生影响！

让我知道您创造了什么！我希望在[PyTorch Academy](https://pytorchacademy.com)的课程中见到您，并随时通过电子邮件（jpapa@joepapa.ai）、Twitter（@JoePapaAI）或 LinkedIn（@MrJoePapa）联系我。
