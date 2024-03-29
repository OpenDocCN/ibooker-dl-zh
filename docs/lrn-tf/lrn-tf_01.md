# 第一章：介绍

本章提供了 TensorFlow 及其主要用途的高层概述：实现和部署深度学习系统。我们首先对深度学习进行了非常简要的介绍。然后展示 TensorFlow，展示了它在构建机器智能方面的一些令人兴奋的用途，然后列出了它的主要特性和属性。

# 深入探讨

从大型公司到新兴初创公司，工程师和数据科学家正在收集大量数据，并使用机器学习算法来回答复杂问题并构建智能系统。在这个领域的任何地方，与深度学习相关的算法类最近取得了巨大成功，通常将传统方法远远甩在后面。深度学习今天被用于理解图像、自然语言和语音的内容，应用范围从移动应用到自动驾驶汽车。这一领域的发展速度惊人，深度学习正在扩展到其他领域和数据类型，比如用于药物发现的复杂化学和基因结构，以及公共卫生保健中的高维医疗记录。

深度学习方法，也被称为深度神经网络，最初受到人类大脑庞大的相互连接的神经元网络的粗略启发。在深度学习中，我们将数百万个数据实例输入到神经元网络中，教导它们从原始输入中识别模式。深度神经网络接受原始输入（比如图像中的像素值）并将它们转换为有用的表示，提取更高级的特征（比如图像中的形状和边缘），通过组合越来越小的信息片段来捕捉复杂的概念，解决挑战性任务，比如图像分类。这些网络通过自动学习来构建抽象表示，通过适应和自我纠正来拟合数据中观察到的模式。自动构建数据表示的能力是深度神经网络相对于传统机器学习的关键优势，传统机器学习通常需要领域专业知识和手动特征工程才能进行任何“学习”。

![](img/letf_0101.png)

###### 图 1-1 图像分类与深度神经网络的示例。网络接受原始输入（图像中的像素值）并学习将其转换为有用的表示，以获得准确的图像分类。

这本书是关于谷歌的深度学习框架 TensorFlow。多年来，深度学习算法已经在谷歌的许多产品和领域中被使用，比如搜索、翻译、广告、计算机视觉和语音识别。事实上，TensorFlow 是谷歌用于实现和部署深度神经网络的第二代系统，继承了 2011 年开始的 DistBelief 项目。

TensorFlow 于 2015 年 11 月以 Apache 2.0 许可证的开源框架形式发布给公众，已经在业界掀起了风暴，其应用远远超出了谷歌内部项目。其可扩展性和灵活性，加上谷歌工程师们继续维护和发展的强大力量，使 TensorFlow 成为进行深度学习的领先系统。

## 使用 TensorFlow 进行人工智能系统

在深入讨论 TensorFlow 及其主要特性之前，我们将简要介绍 TensorFlow 在一些尖端的现实世界应用中的使用示例，包括谷歌及其他地方。

### 预训练模型：全面的计算机视觉

深度学习真正闪耀的一个主要领域是计算机视觉。计算机视觉中的一个基本任务是图像分类——构建接收图像并返回最佳描述类别集的算法和系统。研究人员、数据科学家和工程师设计了先进的深度神经网络，可以在理解视觉内容方面获得高度准确的结果。这些深度网络通常在大量图像数据上进行训练，需要大量时间、资源和精力。然而，趋势逐渐增长，研究人员正在公开发布预训练模型——已经训练好的深度神经网络，用户可以下载并应用到他们的数据中（图 1-2）。

![](img/letf_0102.png)

###### 图 1-2。使用预训练 TensorFlow 模型的高级计算机视觉。

TensorFlow 带有有用的实用程序，允许用户获取和应用尖端的预训练模型。我们将在本书中看到几个实际示例，并深入了解细节。

### 为图像生成丰富的自然语言描述

深度学习研究中一个令人兴奋的领域是为视觉内容生成自然语言描述（图 1-3）。这个领域的一个关键任务是图像字幕——教导模型为图像输出简洁准确的字幕。在这里，也提供了结合自然语言理解和计算机视觉的先进预训练 TensorFlow 模型。

![](img/letf_0103.png)

###### 图 1-3。通过图像字幕从图像到文本（示例说明）。

### 文本摘要

自然语言理解（NLU）是构建人工智能系统的关键能力。每天产生大量文本：网络内容、社交媒体、新闻、电子邮件、内部企业通信等等。最受追捧的能力之一是总结文本，将长篇文档转化为简洁连贯的句子，提取原始文本中的关键信息（图 1-4）。正如我们将在本书中看到的，TensorFlow 具有强大的功能，可以用于训练深度 NLU 网络，也可以用于自动文本摘要。

![](img/letf_0104.png)

###### 图 1-4。智能文本摘要的示例插图。

# TensorFlow：名字的含义是什么？

深度神经网络，正如我们所示的术语和插图所暗示的，都是关于神经元网络的，每个神经元学习执行自己的操作，作为更大图像的一部分。像图像这样的数据作为输入进入这个网络，并在训练时适应自身，或在部署系统中预测输出。

张量是在深度学习中表示数据的标准方式。简单来说，张量只是多维数组，是对具有更高维度的数据的扩展。就像黑白（灰度）图像被表示为像素值的“表格”一样，RGB 图像被表示为张量（三维数组），每个像素具有三个值对应于红、绿和蓝色分量。

在 TensorFlow 中，计算被看作是一个*数据流图*（图 1-5）。广义上说，在这个图中，节点表示操作（如加法或乘法），边表示数据（张量）在系统中流动。在接下来的章节中，我们将深入探讨这些概念，并通过许多示例学会理解它们。

![](img/letf_0105.png)

###### 图 1-5。数据流计算图。数据以张量的形式流经由计算操作组成的图，构成我们的深度神经网络。 

# 高层概述

TensorFlow 在最一般的术语中是一个基于数据流图的数值计算软件框架。然而，它主要设计为表达和实现机器学习算法的接口，其中深度神经网络是其中的主要算法之一。

TensorFlow 设计时考虑了可移植性，使这些计算图能够在各种环境和硬件平台上执行。例如，使用基本相同的代码，同一个 TensorFlow 神经网络可以在云端训练，分布在许多机器的集群上，或者在单个笔记本电脑上进行训练。它可以部署用于在专用服务器上提供预测，或者在 Android 或 iOS 等移动设备平台上，或者树莓派单板计算机上。当然，TensorFlow 也与 Linux、macOS 和 Windows 操作系统兼容。

TensorFlow 的核心是 C++，它有两种主要的高级前端语言和接口，用于表达和执行计算图。最发达的前端是 Python，大多数研究人员和数据科学家使用。C++前端提供了相当低级的 API，适用于在嵌入式系统和其他场景中进行高效执行。

除了可移植性，TensorFlow 的另一个关键方面是其灵活性，允许研究人员和数据科学家相对轻松地表达模型。有时，将现代深度学习研究和实践视为玩“乐高”积木是有启发性的，用其他块替换网络块并观察结果，有时设计新的块。正如我们将在本书中看到的，TensorFlow 提供了有用的工具来使用这些模块化块，结合灵活的 API，使用户能够编写新的模块。在深度学习中，网络是通过基于梯度下降优化的反馈过程进行训练的。TensorFlow 灵活支持许多优化算法，所有这些算法都具有自动微分功能——用户无需提前指定任何梯度，因为 TensorFlow 会根据用户提供的计算图和损失函数自动推导梯度。为了监视、调试和可视化训练过程，并简化实验，TensorFlow 附带了 TensorBoard（图 1-6），这是一个在浏览器中运行的简单可视化工具，我们将在本书中始终使用。

![](img/letf_0106.png)

###### 图 1-6。TensorFlow 的可视化工具 TensorBoard，用于监视、调试和分析训练过程和实验。

TensorFlow 的灵活性对数据科学家和研究人员的关键支持是高级抽象库。在计算机视觉或 NLU 的最先进深度神经网络中，编写 TensorFlow 代码可能会耗费精力，变得复杂、冗长和繁琐。Keras 和 TF-Slim 等抽象库提供了对底层库中的“乐高积木”的简化高级访问，有助于简化数据流图的构建、训练和推理。对数据科学家和工程师的另一个关键支持是 TF-Slim 和 TensorFlow 附带的预训练模型。这些模型是在大量数据和强大计算资源上进行训练的，这些资源通常难以获得，或者至少需要大量努力才能获取和设置。例如，使用 Keras 或 TF-Slim，只需几行代码就可以使用这些先进模型对传入数据进行推理，还可以微调模型以适应新数据。

TensorFlow 的灵活性和可移植性有助于使研究到生产的流程顺畅，减少了数据科学家将模型推送到产品部署和工程师将算法思想转化为稳健代码所需的时间和精力。

# TensorFlow 抽象

TensorFlow 还配备了抽象库，如 Keras 和 TF-Slim，提供了对 TensorFlow 的简化高级访问。这些抽象，我们将在本书后面看到，有助于简化数据流图的构建，并使我们能够用更少的代码进行训练和推断。

但除了灵活性和可移植性之外，TensorFlow 还具有一系列属性和工具，使其对构建现实世界人工智能系统的工程师具有吸引力。它自然支持分布式训练 - 实际上，它被谷歌和其他大型行业参与者用于在许多机器的集群上训练大规模网络的海量数据。在本地实现中，使用多个硬件设备进行训练只需要对用于单个设备的代码进行少量更改。当从本地转移到分布式时，代码也基本保持不变，这使得在云中使用 TensorFlow，如在亚马逊网络服务（AWS）或谷歌云上，特别具有吸引力。此外，正如我们将在本书后面看到的那样，TensorFlow 还具有许多旨在提高可伸缩性的功能。这些功能包括支持使用线程和队列进行异步计算，高效的 I/O 和数据格式等等。

深度学习不断快速发展，TensorFlow 也在不断更新和增加新的令人兴奋的功能，带来更好的可用性、性能和价值。

# 总结

通过本章描述的一系列工具和功能，很明显为什么在短短一年多的时间里 TensorFlow 吸引了如此多的关注。本书旨在首先迅速让您了解基础并准备好工作，然后我们将深入探讨 TensorFlow 的世界，带来令人兴奋和实用的示例。
