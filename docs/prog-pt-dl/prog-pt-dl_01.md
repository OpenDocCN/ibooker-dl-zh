# 第一章：开始使用 PyTorch

在本章中，我们设置了使用 PyTorch 所需的一切。一旦我们完成了这一步，随后的每一章都将在这个初始基础上构建，因此很重要我们做对。这导致我们的第一个基本问题：您应该构建一个自定义深度学习计算机，还是只使用众多可用的基于云的资源之一？

# 构建自定义深度学习机器

在深度学习中，有一种冲动，想要为自己的计算需求建造一个庞然大物。您可以花费数天时间查看不同类型的显卡，了解可能的 CPU 选择提供的内存通道，购买最佳类型的内存，以及购买多大的 SSD 驱动器以尽可能快地访问磁盘。我并不是在宣称自己对此免疫；几年前，我花了一个月的时间列出零件清单，在我的餐桌上组装了一台新电脑。

我的建议，特别是对于新手来说，是：不要这样做。您可以轻松地在一台您可能不会经常使用的机器上花费数千美元。相反，我建议您通过使用云资源（无论是亚马逊网络服务、谷歌云还是微软 Azure）来阅读本书，然后再考虑是否需要为自己建造一台机器，如果您觉得需要一台全天候运行的单机。您*不需要*在硬件上进行巨额投资来运行本书中的任何代码。

您可能永远不需要为自己建造一台定制机器。有一个甜蜜的点，如果您知道您的计算总是会受限于一台机器（最多几个 GPU），那么建造一个定制机器可能会更便宜。然而，如果您的计算开始需要跨越多台机器和 GPU，云再次变得有吸引力。考虑到组装一台定制机器的成本，我建议您在深入之前三思而后行。

如果我还没有劝阻您自己组装机器，接下来的部分将提供您需要做的建议。

## GPU

每个深度学习盒子的核心，GPU，将为大多数 PyTorch 的计算提供动力，并且很可能是您机器中最昂贵的组件。近年来，由于它们在挖掘比特币等加密货币中的使用，GPU 的价格已经上涨，供应量也在减少。幸运的是，这个泡沫似乎正在消退，GPU 的供应量又变得更加充裕。

在撰写本文时，我建议选择 NVIDIA GeForce RTX 2080 Ti。如果想要更便宜的选项，可以选择 1080 Ti（尽管如果您因预算原因考虑选择 1080 Ti，我再次建议您考虑云选项）。尽管 AMD 制造的 GPU 卡确实存在，但它们在 PyTorch 中的支持目前还不够好，无法推荐除 NVIDIA 卡以外的其他选择。但请留意他们的 ROCm 技术，这将最终使它们成为 GPU 领域的可信替代品。

## CPU/主板

您可能会选择 Z370 系列主板。许多人会告诉您，CPU 对于深度学习并不重要，只要有强大的 GPU，您可以使用速度较低的 CPU。但根据我的经验，CPU 往往会成为瓶颈，尤其在处理增强数据时。

## RAM

更多的 RAM 是好的，因为这意味着您可以将更多数据保存在内存中，而不必访问速度慢得多的磁盘存储（尤其是在训练阶段非常重要）。您应该至少考虑为您的机器配备 64GB DDR4 内存。

## 存储

自定义机箱的存储应该分为两类：首先，一个 M2 接口固态硬盘（SSD）——尽可能大——用于存储您正在积极工作的项目时保持尽可能快的访问速度的*热*数据。对于第二类存储，添加一个 4TB 串行 ATA（SATA）驱动器，用于您当前不在积极工作的数据，并根据需要转移到*热*和*冷*存储。

我建议您查看[PCPartPicker](https://pcpartpicker.com)来浏览其他人的深度学习机器（您还可以看到所有奇怪和疯狂的机箱设计想法！）。您将了解到机器零件清单和相关价格，这些价格可能会大幅波动，尤其是 GPU 卡的价格。

现在您已经查看了本地的物理机器选项，是时候转向云端了。

# 云端深度学习

好了，那么为什么云端选项更好呢？尤其是如果您已经查看了亚马逊网络服务（AWS）的定价方案，并计算出构建深度学习机器将在六个月内收回成本？想一想：如果您刚开始，您不会在这六个月内全天候使用那台机器。您就是不会。这意味着您可以关闭云端机器，并支付存储数据的几分钱。

如果您刚开始，您不需要立即使用 NVIDIA 的庞大 Tesla V100 卡连接到您的云实例。您可以从更便宜的（有时甚至是免费的）基于 K80 的实例开始，然后在准备好时升级到更强大的卡。这比在自定义盒子上购买基本 GPU 卡并升级到 2080Ti 便宜一点。此外，如果您想在单个实例中添加八张 V100 卡，您只需点击几下即可。试试用您自己的硬件做到这一点。

另一个问题是维护。如果您养成良好的习惯，定期重新创建云实例（最好每次回来进行实验时都重新开始），您几乎总是会有一个最新的机器。如果您有自己的机器，更新就取决于您。这就是我承认我有自己的定制深度学习机器的地方，我忽视了它上面的 Ubuntu 安装很长时间，结果是它不再接收支持的更新，最终花了一整天的时间来让系统恢复到可以再次接收更新的状态。令人尴尬。

无论如何，您已经决定转向云端。万岁！接下来：选择哪个提供商？

## Google Colaboratory

但等等——在我们查看提供商之前，如果您根本不想做任何工作怎么办？不想建立一台机器或者不想费心设置云端实例？哪里有真正懒惰的选择？谷歌为您提供了正确的东西。[*Colaboratory*（或*Colab*）](https://colab.research.google.com)是一个大多数免费、无需安装的定制 Jupyter Notebook 环境。您需要一个谷歌账号来设置您自己的笔记本。图 1-1 显示了在 Colab 中创建的笔记本的屏幕截图。

Colab 之所以成为深度学习的绝佳方式，是因为它包含了预安装的 TensorFlow 和 PyTorch 版本，因此您无需进行任何设置，只需键入`import torch`，每个用户都可以免费获得长达 12 小时的连续运行时间的 NVIDIA T4 GPU。免费的。从这个角度来看，实证研究表明，您在训练时大约可以获得 1080 Ti 速度的一半，但内存额外增加了 5GB，因此您可以存储更大的模型。它还提供了连接到更近期的 GPU 和谷歌定制的 TPU 硬件的能力，但您几乎可以使用 Colab 免费完成本书中的每个示例。因此，我建议一开始就使用 Colab，并随后根据需要决定是否扩展到专用云实例和/或您自己的个人深度学习服务器。

![谷歌 Colab](img/ppdl_0101.png)

###### 图 1-1\. 谷歌 Colab（实验室）

Colab 是零工作量的方法，但你可能想要对安装方式或在云端实例上获取安全外壳（SSH）访问有更多控制，因此让我们看看主要云服务提供商提供了什么。

## 云服务提供商

三大云服务提供商（亚马逊网络服务、谷歌云平台和微软的 Azure）都提供基于 GPU 的实例（也称为*虚拟机*或*VMs*）和官方镜像以部署在这些实例上。它们提供了一切你需要的，无需自己安装驱动程序和 Python 库即可运行。让我们看看每个提供商提供了什么。

### 亚马逊网络服务

AWS，云市场的 800 磅大猩猩，乐意满足你的 GPU 需求，并提供 P2 和 P3 实例类型来帮助你。（G3 实例类型更多用于实际的基于图形的应用程序，如视频编码，所以我们这里不涉及。）P2 实例使用较旧的 NVIDIA K80 卡（最多可以连接 16 个到一个实例），而 P3 实例使用快速的 NVIDIA V100 卡（如果你敢的话，你可以在一个实例上连接八个）。

如果你要使用 AWS，我建议选择`p2.xlarge`类。在撰写本书时，这将花费你每小时仅 90 美分，并为你提供足够的计算能力来完成示例。当你开始参加一些有挑战性的 Kaggle 比赛时，你可能会想升级到 P3 类。

在 AWS 上创建并运行深度学习框非常容易：

1.  登录到 AWS 控制台。

1.  选择 EC2 并点击启动实例。

1.  搜索深度学习 AMI（Ubuntu）选项并选择它。

1.  选择`p2.xlarge`作为你的实例类型。

1.  启动实例，可以创建新的密钥对或重用现有的密钥对。

1.  通过使用 SSH 连接并将本地机器上的端口 8888 重定向到实例来连接到实例：

    ```py

    ssh-Llocalhost:8888:localhost:8888\ -i*`your``.pem``filename`*ubuntu@*`your``instance``DNS`*
    ```

1.  通过输入`**jupyter notebook**`来启动 Jupyter Notebook。复制生成的 URL 并粘贴到浏览器中以访问 Jupyter。

记得在不使用时关闭你的实例！你可以通过在 Web 界面中右键单击实例并选择关闭选项来实现这一点。这将关闭实例，并在实例不运行时不会向你收费。然而，即使实例关闭，你仍会被收取为其分配的存储空间费用，所以请注意。要完全删除实例和存储，请选择终止选项。

### Azure

与 AWS 一样，Azure 提供了一些更便宜的基于 K80 的实例和更昂贵的 Tesla V100 实例。Azure 还提供基于较旧的 P100 硬件的实例，作为其他两种之间的中间点。同样，我建议本书使用单个 K80（NC6）的实例类型，这也每小时花费 90 美分，并根据需要转移到其他 NC、NCv2（P100）或 NCv3（V100）类型。

以下是如何在 Azure 中设置 VM：

1.  登录到 Azure 门户，并在 Azure Marketplace 中找到 Data Science Virtual Machine 镜像。

1.  点击立即获取按钮。

1.  填写 VM 的详细信息（为其命名，选择 SSD 磁盘而不是 HDD，一个 SSH 用户名/密码，将实例计费到的订阅，以及将位置设置为最接近你的提供 NC 实例类型的地点）。

1.  点击创建选项。实例应该在大约五分钟内被配置。

1.  你可以使用指定给该实例的公共域名系统（DNS）名称的用户名/密码来使用 SSH。

1.  当实例被配置时，Jupyter Notebook 应该运行；导航至*http://`实例的 DNS 名称`:8000*并使用你用于 SSH 登录的用户名/密码组合登录。

### 谷歌云平台

除了像亚马逊和 Azure 一样提供 K80、P100 和 V100 支持的实例外，Google Cloud Platform（GCP）还为那些具有巨大数据和计算需求的人提供了上述的 TPUs。您不需要本书中的 TPUs，它们价格昂贵，但它们*将*与 PyTorch 1.0 一起使用，因此不要认为您必须使用 TensorFlow 才能利用它们，如果您有一个需要使用它们的项目。

开始使用 Google Cloud 也非常简单：

1.  在 GCP Marketplace 上搜索 Deep Learning VM。

1.  在 Compute Engine 上点击启动。

1.  为实例命名并将其分配给您最近的区域。

1.  将机器类型设置为 8 个 vCPU。

1.  将 GPU 设置为 1 K80。

1.  确保在框架部分中选择 PyTorch 1.0。

1.  选择“第一次启动时自动安装 NVIDIA GPU？”复选框。

1.  将启动磁盘设置为 SSD 持久磁盘。

1.  单击部署选项。虚拟机将需要大约 5 分钟才能完全部署。

1.  要连接到实例上的 Jupyter，请确保您已登录到`gcloud`中的正确项目，并发出以下命令：

    ```py
    gcloud compute ssh _INSTANCE_NAME_ -- -L 8080:localhost:8080
    ```

Google Cloud 的费用大约为每小时 70 美分，是三家主要云服务提供商中最便宜的。

## 应该使用哪个云服务提供商？

如果没有任何事情吸引您，我建议使用 Google Cloud Platform（GCP）；这是最便宜的选择，如果需要，您可以扩展到使用 TPUs，比 AWS 或 Azure 提供的灵活性更大。但如果您已经在另外两个平台中的一个上拥有资源，那么在这些环境中运行将完全没问题。

一旦您的云实例运行起来，您将能够登录到其 Jupyter Notebook 的副本，所以下面让我们来看看。

# 使用 Jupyter Notebook

如果您以前没有接触过它，这里是关于 Jupyter Notebook 的简介：这个基于浏览器的环境允许您将实时代码与文本、图像和可视化混合在一起，已经成为全球数据科学家的事实标准工具之一。在 Jupyter 中创建的笔记本可以轻松共享；实际上，您会在[本书中的所有笔记本](https://oreil.ly/iBh4V)中找到。您可以在图 1-2 中看到 Jupyter Notebook 的截图。

在本书中，我们不会使用 Jupyter 的任何高级功能；您只需要知道如何创建一个新的笔记本，以及 Shift-Enter 如何运行单元格的内容。但如果您以前从未使用过它，我建议在进入第二章之前浏览[Jupyter 文档](https://oreil.ly/-Yhff)。

![Jupyter Notebook](img/ppdl_0102.png)

###### 图 1-2. Jupyter Notebook

在我们开始使用 PyTorch 之前，我们将讨论最后一件事：如何手动安装所有内容。

# 从头开始安装 PyTorch

也许您想对软件有更多控制，而不是使用之前提供的云镜像之一。或者您需要特定版本的 PyTorch 来运行您的代码。或者，尽管我发出了所有警告，您真的想要在地下室中安装那台设备。让我们看看如何在 Linux 服务器上通用安装 PyTorch。

###### 警告

您可以使用 Python 2.*x*与 PyTorch 一起使用，但我强烈建议不要这样做。尽管 Python 2.*x*到 3.*x*的升级已经进行了十多年，但越来越多的软件包开始放弃对 Python 2.*x*的支持。因此，除非有充分理由，确保您的系统正在运行 Python 3。

## 下载 CUDA

尽管 PyTorch 可以完全在 CPU 模式下运行，但在大多数情况下，需要 GPU 支持的 PyTorch 才能实现实际用途，因此我们需要 GPU 支持。这相当简单；假设您有一张 NVIDIA 卡，这是由他们的 Compute Unified Device Architecture（CUDA）API 提供的。[下载适合您 Linux 版本的适当软件包格式](https://oreil.ly/Gx_q2)并安装软件包。

对于 Red Hat Enterprise Linux（RHEL）7：

```py
sudo rpm -i cuda-repo-rhel7-10-0local-10.0.130-410.48-1.0-1.x86_64.rpm
sudo yum clean all
sudo yum install cuda
```

对于 Ubuntu 18.04：

```py
sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

## Anaconda

Python 有各种打包系统，所有这些系统都有好坏之分。与 PyTorch 的开发人员一样，我建议您安装 Anaconda，这是一个专门为数据科学家提供最佳软件包分发的打包系统。与 CUDA 一样，它相当容易安装。

前往[Anaconda](https://oreil.ly/9hAxg)并选择适合您机器的安装文件。因为这是一个通过 shell 脚本在您的系统上执行的大型存档，我建议您在下载的文件上运行`md5sum`并将其与[签名列表](https://oreil.ly/anuhu)进行比对，然后再使用`bash Anaconda3-VERSION-Linux-x86_64.sh`执行以确保您机器上的签名与网页上的签名匹配。这可以确保下载的文件没有被篡改，并且可以安全地在您的系统上运行。脚本将提供有关将要安装的位置的几个提示；除非有充分的理由，否则请接受默认设置。

###### 注意

您可能会想：“我能在我的 MacBook 上做这个吗？”遗憾的是，如今大多数 Mac 都配备有 Intel 或 AMD GPU，实际上不支持在 GPU 加速模式下运行 PyTorch。我建议您使用 Colab 或云服务提供商，而不是尝试在本地使用 Mac。

## 最后，PyTorch！（和 Jupyter Notebook）

现在您已经安装了 Anaconda，使用 PyTorch 很简单：

```py
conda install pytorch torchvision -c pytorch
```

这将安装 PyTorch 和我们在接下来的几章中使用的`torchvision`库，用于创建与图像一起工作的深度学习架构。Anaconda 还为我们安装了 Jupyter Notebook，因此我们可以通过启动它来开始：

```py
jupyter notebook
```

在浏览器中前往*http://`YOUR-IP-ADDRESS`:8888*，创建一个新的笔记本，并输入以下内容：

```py
import torch
print(torch.cuda.is_available())
print(torch.rand(2,2))
```

这应该产生类似于这样的输出：

```py
True
 0.6040  0.6647
 0.9286  0.4210
[torch.FloatTensor of size 2x2]
```

如果`cuda.is_available()`返回`False`，则需要调试您的 CUDA 安装，以便 PyTorch 可以看到您的显卡。在您的实例上，张量的值将不同。

但这个张量是什么？张量几乎是 PyTorch 中的一切，因此您需要知道它们是什么以及它们可以为您做什么。

# 张量

*张量*既是数字的容器，也是定义在产生新张量之间的张量之间的转换规则的集合。对于我们来说，将*张量*视为多维数组可能是最容易的。每个张量都有一个与其维度空间对应的*秩*。一个简单的标量（例如，1）可以表示为秩为 0 的张量，一个向量是秩为 1 的，一个*n*×*n*矩阵是秩为 2 的，依此类推。在前面的示例中，我们使用`torch.rand()`创建了一个具有随机值的秩为 2 的张量。我们也可以从列表中创建它们：

```py
x = torch.tensor([[0,0,1],[1,1,1],[0,0,0]])
x
>tensor([[0, 0, 1],
    [1, 1, 1],
    [0, 0, 0]])
```

我们可以通过使用标准的 Python 索引在张量中更改元素：

```py
x[0][0] = 5
>tensor([[5, 0, 1],
    [1, 1, 1],
    [0, 0, 0]])
```

您可以使用特殊的创建函数生成特定类型的张量。特别是，`ones()`和`zeroes()`将分别生成填充有 1 和 0 的张量：

```py
torch.zeros(2,2)
> tensor([[0., 0.],
    [0., 0.]])
```

您可以使用张量执行标准的数学运算（例如，将两个张量相加）：

```py
tensor.ones(1,2) + tensor.ones(1,2)
> tensor([[2., 2.]])
```

如果您有一个秩为 0 的张量，可以使用`item()`提取值：

```py
torch.rand(1).item()
> 0.34106671810150146
```

张量可以存在于 CPU 或 GPU 上，并且可以通过使用`to()`函数在设备之间进行复制：

```py
cpu_tensor = tensor.rand(2)
cpu_tensor.device
> device(type='cpu')

gpu_tensor = cpu_tensor.to("cuda")
gpu_tensor.device
> device(type='cuda', index=0)
```

## 张量操作

如果您查看[PyTorch 文档](https://oreil.ly/1Ev0-)，您会发现有很多函数可以应用于张量——从查找最大元素到应用傅立叶变换等。在本书中，您不需要了解所有这些函数来将图像、文本和音频转换为张量并对其进行操作，但您需要了解一些。我强烈建议您在完成本书后浏览文档。现在我们将逐一介绍将在接下来的章节中使用的所有函数。

首先，我们经常需要找到张量中的最大项以及包含最大值的*索引*（因为这通常对应于神经网络在最终预测中决定的类）。这可以通过`max()`和`argmax()`函数来实现。我们还可以使用`item()`从 1D 张量中提取标准的 Python 值。

```py
torch.rand(2,2).max()
> tensor(0.4726)
torch.rand(2,2).max().item()
> 0.8649941086769104
```

有时，我们可能想要改变张量的类型；例如，从`LongTensor`到`FloatTensor`。我们可以使用`to()`来实现：

```py
long_tensor = torch.tensor([[0,0,1],[1,1,1],[0,0,0]])
long_tensor.type()
> 'torch.LongTensor'
float_tensor = torch.tensor([[0,0,1],[1,1,1],[0,0,0]]).to(dtype=torch.float32)
float_tensor.type()
> 'torch.FloatTensor'
```

大多数在张量上操作并返回张量的函数会创建一个新的张量来存储结果。然而，如果你想节省内存，可以查看是否定义了一个*原地*函数，它的名称应该与原始函数相同，但在末尾加上下划线(`_`)。

```py
random_tensor = torch.rand(2,2)
random_tensor.log2()
>tensor([[-1.9001, -1.5013],
        [-1.8836, -0.5320]])
random_tensor.log2_()
> tensor([[-1.9001, -1.5013],
        [-1.8836, -0.5320]])
```

另一个常见的操作是*重塑*张量。这通常是因为你的神经网络层可能需要一个与你当前要输入的形状略有不同的输入形状。例如，手写数字的 Modified National Institute of Standards and Technology (MNIST)数据集是一组 28×28 的图像，但它的打包方式是长度为 784 的数组。为了使用我们正在构建的网络，我们需要将它们转换回 1×28×28 的张量（前导的 1 是通道数——通常是红、绿和蓝，但由于 MNIST 数字只是灰度的，我们只有一个通道）。我们可以使用`view()`或`reshape()`来实现：

```py
flat_tensor = torch.rand(784)
viewed_tensor = flat_tensor.view(1,28,28)
viewed_tensor.shape
> torch.Size([1, 28, 28])
reshaped_tensor = flat_tensor.reshape(1,28,28)
reshaped_tensor.shape
> torch.Size([1, 28, 28])
```

请注意，重塑后的张量形状必须与原始张量的总元素数相同。如果你尝试`flat_tensor.reshape(3,28,28)`，你会看到这样的错误：

```py
RuntimeError Traceback (most recent call last)
<ipython-input-26-774c70ba5c08> in <module>()
----> 1 flat_tensor.reshape(3,28,28)

RuntimeError: shape '[3, 28, 28]' is invalid for input of size 784
```

现在你可能想知道`view()`和`reshape()`之间的区别是什么。答案是`view()`作为原始张量的视图操作，所以如果底层数据发生变化，视图也会发生变化（反之亦然）。然而，如果所需的视图不是*连续*的，`view()`可能会抛出错误；也就是说，如果它不与从头开始创建的具有所需形状的新张量共享相同的内存块。如果发生这种情况，你必须在使用`view()`之前调用`tensor.contiguous()`。然而，`reshape()`在幕后完成所有这些工作，所以一般来说，我建议使用`reshape()`而不是`view()`。

最后，你可能需要重新排列张量的维度。你可能会在处理图像时遇到这种情况，图像通常以`[height, width, channel]`的张量形式存储，但 PyTorch 更喜欢以`[channel, height, width]`的形式处理。你可以使用`permute()`来以一种相当简单的方式处理这些：

```py
hwc_tensor = torch.rand(640, 480, 3)
chw_tensor = hwc_tensor.permute(2,0,1)
chw_tensor.shape
> torch.Size([3, 640, 480])
```

在这里，我们刚刚对一个`[640,480,3]`的张量应用了`permute`，参数是张量维度的索引，所以我们希望最终维度（由于从零开始索引，是 2）在张量的前面，后面是剩下的两个维度按照原始顺序。

## 张量广播

从 NumPy 借鉴的*广播*允许你在张量和较小张量之间执行操作。如果从它们的尾部维度开始向后看，你可以在两个张量之间进行广播：

+   两个维度相等。

+   一个维度是 1。

在我们使用广播时，它有效是因为 1 有一个维度是 1，而且没有其他维度，1 可以扩展到另一个张量。如果我们尝试将一个`[2,2]`张量加到一个`[3,3]`张量上，我们会得到这样的错误消息：

```py
The size of tensor a (2) must match the size of
tensor b (3) at non-singleton dimension 1
```

但是我们可以毫无问题地将一个`[1,3]`张量加到一个`[3,3]`张量上。广播是一个方便的小功能，可以增加代码的简洁性，并且通常比手动扩展张量更快。

关于张量的一切你需要开始的内容就到这里了！我们将在书中后面遇到其他一些操作，但这已经足够让你深入第二章了。

# 结论

无论是在云端还是在本地机器上，您现在应该已经安装了 PyTorch。我已经介绍了该库的基本构建模块，*张量*，您已经简要了解了 Jupyter Notebook。这就是您开始的全部所需！在下一章中，您将利用到目前为止所见的一切来开始构建神经网络和对图像进行分类，所以在继续之前，请确保您对张量和 Jupyter 感到舒适。

# 进一步阅读

+   [Project Jupyter 文档](https://jupyter.org/documentation)

+   [PyTorch 文档](https://pytorch.org/docs/stable)

+   [AWS 深度学习 AMI](https://oreil.ly/G9Ldx)

+   [Azure 数据科学虚拟机](https://oreil.ly/YjzVB)

+   [Google 深度学习虚拟机镜像](https://oreil.ly/NFpeG)
