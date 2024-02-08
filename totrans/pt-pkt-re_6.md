# 第六章：PyTorch 加速和优化

在前几章中，您学习了如何使用 PyTorch 的内置功能，并通过创建自己的自定义组件来扩展这些功能，从而使您能够快速设计新模型和算法来训练它们。

然而，当处理非常大的数据集或更复杂的模型时，将模型训练在单个 CPU 或 GPU 上可能需要非常长的时间——可能需要几天甚至几周才能获得初步结果。训练时间更长可能会变得令人沮丧，特别是当您想要使用不同的超参数配置进行许多实验时。

在本章中，我们将探讨使用 PyTorch 加速和优化模型开发的最新技术。首先，我们将看看如何使用张量处理单元（TPU）而不是 GPU 设备，并考虑在使用 TPU 时可以提高性能的情况。接下来，我将向您展示如何使用 PyTorch 的内置功能进行并行处理和分布式训练。这将为跨多个 GPU 和多台机器训练模型提供一个快速参考，以便在更多硬件资源可用时快速扩展您的训练。在探索加速训练的方法之后，我们将看看如何使用高级技术（如超参数调整、量化和剪枝）来优化您的模型。

本章还将提供参考代码，以便轻松入门，并提供我们使用的关键软件包和库的参考资料。一旦您创建了自己的模型和训练循环，您可以返回到本章获取有关如何加速和优化训练过程的提示。

让我们开始探讨如何在 TPU 上运行您的模型。

# 在 TPU 上的 PyTorch

随着深度学习和人工智能的不断部署，公司正在开发定制硬件芯片或 ASIC，旨在优化硬件中的模型性能。谷歌开发了自己的用于神经网络加速的 ASIC，称为 TPU。由于 TPU 是为神经网络设计的，它没有 GPU 的一些缺点，GPU 是为图形处理而设计的。谷歌的 TPU 现在可以作为谷歌云 TPU 的一部分供您使用。您还可以在 Google Colab 上运行 TPU。

在前几章中，我向您展示了如何使用 GPU 测试和训练您的深度模型。如果您的用例符合以下条件，您应该继续使用 CPU 和 GPU 进行训练：

+   您有小型或中型模型以及小批量大小。

+   您的模型训练时间不长。

+   数据的进出是您的主要瓶颈。

+   您的计算经常是分支的或主要是逐元素完成的，或者您使用稀疏内存访问。

+   您需要使用高精度。双精度不适合在 TPU 上使用。

另一方面，有几个原因可能会导致您希望使用 TPU 而不是 GPU 进行训练。TPU 在执行密集向量和矩阵计算方面非常快速。它们针对特定工作负载进行了优化。如果您的用例符合以下情况，您应该强烈考虑使用 TPU：

+   您的模型主要由矩阵计算组成。

+   您的模型训练时间很长。

+   您希望在 TPU 上运行整个训练循环的多次迭代。

在 TPU 上运行与在 CPU 或 GPU 上运行非常相似。让我们回顾一下如何在 GPU 上训练模型的以下代码：

```py
device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")①model.to(device)②forepochinrange(n_epochs):fordataintrainloader:input,labels=datainput=input.to(device)③labels=labels.to(device)③optimizer.zero_grad()output=model(input)loss=criterion(input,labels)loss.backward()optimizer.step()
```

①

如果有 GPU 可用，请将设备配置为 GPU。

②

将模型发送到设备。

③

将输入和标签发送到 GPU。

换句话说，我们将模型、输入和标签移至 GPU，其余工作由系统完成。在 TPU 上训练网络几乎与在 GPU 上训练相同，只是您需要使用*PyTorch/XLA*（加速线性代数）包，因为 TPU 目前不受 PyTorch 原生支持。

让我们在 Google Colab 上使用 Cloud TPU 训练我们的模型。打开一个新的 Colab 笔记本，并从运行时菜单中选择更改运行时类型。然后从“硬件加速器”下拉菜单中选择 TPU，如图 6-1 所示。Google Colab 提供免费的 Cloud TPU 系统，包括远程 CPU 主机和每个具有两个核心的四个 TPU 芯片。

![ptpr 0601](img/ptpr_0601.png)

###### 图 6-1。在 Google Colab 中使用 TPU

由于 Colab 默认未安装 PyTorch/XLA，我们需要首先安装它，使用以下命令。这将安装最新的“夜间”版本，但如果需要，您可以选择其他版本：

```py
&#33;curl 'https://raw.githubusercontent.com/pytorch' \'/xla/master/contrib/scripts/env-setup.py'\
-opytorch-xla-env-setup.py&#33;python pytorch-xla-env-setup.py --version &#34;nightly&#34; // ①
```

<1>这些是打算在笔记本中运行的命令。在命令行上运行时，请省略“!”。

安装 PyTorch/XLA 后，我们可以导入该软件包并将数据移动到 TPU：

```py
import torch_xla.core.xla_model as xm

device = xm.xla_device()
```

请注意，我们这里不使用`torch.cuda.is_available()`，因为它仅适用于 GPU。不幸的是，TPU 没有`is_available()`方法。如果您的环境未配置为 TPU，您将收到错误消息。

设备设置完成后，其余代码完全相同：

```py
model.to(device)forepochinrange(n_epochs):fordataintrainloader:input,labels=datainput=input.to(device)labels=labels.to(device)optimizer.zero_grad()output=model(input)loss=criterion(input,labels)loss.backward()optimizer.step()print(output.device)①# out: xla:1
```

①

如果 Colab 配置为 TPU，您应该看到 `xla:1`。

PyTorch/XLA 是一个用于 XLA 操作的通用库，可能支持除 TPU 之外的其他专用 ASIC。有关 PyTorch/XLA 的更多信息，请访问[PyTorch/XLA GitHub 存储库](https://pytorch.tips/xla)。

在 TPU 上运行仍然存在许多限制，GPU 支持更加普遍。因此，大多数 PyTorch 开发人员将首先使用单个 GPU 对其代码进行基准测试，然后再探索使用单个 TPU 或多个 GPU 加速其代码。

我们已经在本书的前面部分介绍了如何使用单个 GPU。在下一节中，我将向您展示如何在具有多个 GPU 的机器上训练您的模型。

# 在多个 GPU 上的 PyTorch（单台机器）

在加速训练和开发时，充分利用您可用的硬件资源非常重要。如果您有一台本地计算机或网络服务器可以访问多个 GPU，本节将向您展示如何充分利用系统上的 GPU。此外，您可能希望通过在单个实例上使用云 GPU 来扩展 GPU 资源。这通常是在考虑分布式训练方法之前的第一级扩展。

在多个 GPU 上运行代码通常称为*并行处理*。并行处理有两种方法：*数据*并行处理和*模型*并行处理。在数据并行处理期间，数据批次在多个 GPU 之间分割，而每个 GPU 运行模型的副本。在模型并行处理期间，模型在多个 GPU 之间分割，数据批次被管道传送到每个部分。

数据并行处理在实践中更常用。模型并行处理通常保留用于模型不适合单个 GPU 的情况。我将在本节中向您展示如何执行这两种类型的处理。

## 数据并行处理

图 6-2 说明了数据并行处理的工作原理。在此过程中，每个数据批次被分成*N*部分（*N*是主机上可用的 GPU 数量）。*N*通常是 2 的幂。每个 GPU 持有模型的副本，并且为批次的每个部分计算梯度和损失。在每次迭代结束时，梯度和损失被合并。这种方法适用于较大的批次大小和模型适合单个 GPU 的用例。

PyTorch 可以使用*单进程，多线程方法*或使用*多进程*方法来实现数据并行处理。单进程，多线程方法只需要一行额外的代码，但在许多情况下性能不佳。

![ptpr 0602](img/ptpr_0602.png)

###### 图 6-2。数据并行处理

不幸的是，由于 Python 的全局解释器锁（GIL）在线程之间的争用、模型的每次迭代复制以及输入散布和输出收集引入的额外开销，多线程性能较差。您可能想尝试这种方法，因为它非常简单，但在大多数情况下，您可能会使用多进程方法。

### 使用 nn.DataParallel 的多线程方法

PyTorch 的`nn`模块原生支持多线程的数据并行处理。您只需要在将模型发送到 GPU 之前将其包装在`nn.DataParallel`中，如下面的代码所示。在这里，我们假设您已经实例化了您的模型：

```py
if torch.cuda.device_count() > 1:
  print("This machine has",
        torch.cuda.device_count(),
        "GPUs available.")
  model = nn.DataParallel(model)

model.to("cuda")
```

首先，我们检查确保我们有多个 GPU，然后我们使用`nn.DataParallel()`在将模型发送到 GPU 之前设置数据并行处理。

这种多线程方法是在多个 GPU 上运行的最简单方式；然而，多进程方法通常在单台机器上表现更好。此外，多进程方法也可以用于跨多台机器运行，我们将在本章后面看到。

### 使用 DDP 的多进程方法（首选）

最好使用多进程方法在多个 GPU 上训练您的模型。PyTorch 通过其`nn.parallel.DistributedDataProcessing`模块支持这一点。分布式数据处理（DDP）可以在单台机器上的多个进程或跨多台机器的多个进程中使用。我们将从单台机器开始。

有四个步骤需要修改您的代码：

1.  使用*torch.distributed*初始化一个进程组。

1.  使用*torch.nn.to()*创建一个本地模型。

1.  使用*torch.nn.parallel*将模型包装在 DDP 中。

1.  使用*torch.multiprocessing*生成进程。

以下代码演示了如何将您的模型转换为 DDP 训练。我们将其分解为步骤。首先，导入必要的库：

```py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel \
  import DistributedDataParallel as DDP
```

请注意，我们正在使用三个新库—*torch.distributed*、*torch.multiprocessing*和*torch.nn.parallel*。以下代码向您展示如何创建一个分布式训练循环：

```py
defdist_training_loop(rank,world_size,dataloader,model,loss_fn,optimizer):dist.init_process_group("gloo",rank=rank,world_size=world_size)①model=model.to(rank)②ddp_model=DDP(model,device_ids=[rank])③optimizer=optimizer(ddp_model.parameters(),lr=0.001)forepochsinrange(n_epochs):forinput,labelsindataloader:input=input.to(rank)labels=labels.to(rank)④optimizer.zero_grad()outputs=ddp_model(input)⑤loss=loss_fn(outputs,labels)loss.backward()optimizer.step()dist.destroy_process_group()
```

①

使用`world_size`进程设置一个进程组。

②

将模型移动到 ID 为`rank`的 GPU。

③

将模型包装在 DDP 中。

④

将输入和标签移动到 ID 为`rank`的 GPU。

⑤

调用 DDP 模型进行前向传递。

DDP 将模型状态从`rank0`进程广播到所有其他进程，因此我们不必担心不同进程具有具有不同初始化权重的模型。

DDP 处理了低级别的进程间通信，使您可以将模型视为本地模型。在反向传播过程中，当`loss.backward()`返回时，DDP 会自动同步梯度并将同步的梯度张量放在`params.grad`中。

现在我们已经定义了进程，我们需要使用`spawn()`函数创建这些进程，如下面的代码所示：

```py
if __name__=="__main__":
  world_size = 2
  mp.spawn(dist_training_loop,
      args=(world_size,),
      nprocs=world_size,
      join=True)
```

在这里，我们将代码作为`main`运行，生成两个进程，每个进程都有自己的 GPU。这就是如何在单台机器上的多个 GPU 上运行数据并行处理。

###### 警告

GPU 设备不能在进程之间共享。

如果您的模型不适合单个 GPU 或者使用较小的批量大小，您可以考虑使用模型并行处理而不是数据并行处理。接下来我们将看看这个。

## 模型并行处理

图 6-3 展示了模型并行处理的工作原理。在这个过程中，模型被分割到同一台机器上的 *N* 个 GPU 中。如果我们按顺序处理数据批次，下一个 GPU 将始终等待前一个 GPU 完成，这违背了并行处理的目的。因此，我们需要对数据处理进行流水线处理，以便每个 GPU 在任何给定时刻都在运行。当我们对数据进行流水线处理时，只有前 *N* 个批次按顺序运行，然后每个后续运行会激活所有 GPU。

![ptpr 0603](img/ptpr_0603.png)

###### 图 6-3\. 模型并行处理

实现模型并行处理并不像数据并行处理那样简单，它需要您重新编写模型。您需要定义模型如何跨多个 GPU 分割以及数据在前向传递中如何进行流水线处理。通常通过为模型编写一个子类，具有特定数量的 GPU 的多 GPU 实现来完成这一点。

以下代码演示了 AlexNet 的双 GPU 实现：

```py
classTwoGPUAlexNet(AlexNet):def__init__(self):super(ModelParallelAlexNet,self).__init__(num_classes=num_classes,*args,**kwargs)self.features.to('cuda:0')self.avgpool.to('cuda:0')self.classifier.to('cuda:1')self.split_size=split_sizedefforward(self,x):splits=iter(x.split(self.split_size,dim=0))s_next=next(splits)s_prev=self.seq1(s_next).to('cuda:1')ret=[]fors_nextinsplits:s_prev=self.seq2(s_prev)①ret.append(self.fc(s_prev.view(s_prev.size(0),-1)))s_prev=self.seq1(s_next).to('cuda:1')②s_prev=self.seq2(s_prev)ret.append(self.fc(s_prev.view(s_prev.size(0),-1)))returntorch.cat(ret)
```

①

`s_prev` 在 `cuda:1` 上运行。

②

`s_next` 在 `cuda:0` 上运行，可以与 `s_prev` 并行运行。

因为我们从 `AlexNet` 类派生一个子类，我们继承了它的模型结构，所以不需要创建我们自己的层。相反，我们需要描述模型的哪些部分放在 GPU0 上，哪些部分放在 GPU1 上。然后我们需要在 `forward()` 方法中通过每个 GPU 管道传递数据来实现 GPU 流水线。当训练模型时，您需要将标签放在最后一个 GPU 上，如下面的代码所示：

```py
model=TwoGPUAlexNet()loss_fn=nn.MSELoss()optimizer=optim.SGD(model.parameters(),lr=0.001)forepochsinrange(n_epochs):forinput,labelsindataloader;input=input.to("cuda:0")labels=labels.to("cuda:1")①optimizer.zero_grad()outputs=model(input)loss_fn(outputs,labels).backward()optimizer.step()
```

①

将输入发送到 GPU0，将标签发送到 GPU1。

如您所见，训练循环需要更改一行代码，以确保标签位于最后一个 GPU 上，因为在计算损失之前输出将位于那里。

数据并行处理和模型并行处理是利用多个 GPU 进行加速训练的两种有效范式。如果我们能够将这两种方法结合起来并取得更好的结果，那将是多么美妙呢？让我们看看如何实现结合的方法。

## 结合数据并行处理和模型并行处理

您可以将数据并行处理与模型并行处理结合起来，以进一步提高性能。在这种情况下，您将使用 DDP 包装您的模型，将数据批次分发给多个进程。每个进程将使用多个 GPU，并且您的模型将被分割到每个 GPU 中。

我们只需要做两个更改。

1.  将我们的多 GPU 模型类更改为接受设备作为输入。

1.  在前向传递期间省略设置输出设备。DDP 将确定输入和输出数据的放置位置。

以下代码显示了如何修改多 GPU 模型：

```py
class Simple2GPUModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(Simple2GPUModel,
              self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(
                      10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(
                      10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```

在 `__init__()` 构造函数中，我们传入 GPU 设备对象 `dev0` 和 `dev1`，并描述模型的哪些部分位于哪些 GPU 中。这使我们能够在不同进程上实例化新模型，每个模型都有两个 GPU。`forward()` 方法在模型的适当位置将数据从一个 GPU 移动到下一个 GPU。

以下代码显示了训练循环的更改：

```py
defmodel_parallel_training(rank,world_size):print(f"Running DDP with a model parallel")setup(rank,world_size)# set up mp_model and devices for this processdev0=rank*2dev1=rank*2+1mp_model=Simple2GPUModel(dev0,dev1)ddp_mp_model=DDP(mp_model)①loss_fn=nn.MSELoss()optimizer=optim.SGD(ddp_mp_model.parameters(),lr=0.001)forepochsinrange(n_epochs):forinput,labelsindataloader:input=input.to(dev0),labels=labels,to(dev1)②optimizer.zero_grad()outputs=ddp_mp_model(input)③loss=loss_fn(outputs,labels)loss.backward()optimizer.step()cleanup()
```

①

将模型包装在 `DDP` 中。

②

将输入和标签移动到适当的设备 ID。

③

输出在 `dev1` 上。

总之，当在多个 GPU 上使用 PyTorch 时，您有几个选项。您可以使用本节中的参考代码来实现数据并行、模型并行或组合并行处理，以加速模型训练和推断。到目前为止，我们只讨论了单台机器或云实例上的多个 GPU。

在许多情况下，在单台机器上的多个 GPU 上进行并行处理可以将训练时间减少一半或更多，您只需要升级 GPU 卡或利用更大的云 GPU 实例。但是，如果您正在训练非常复杂的模型或使用极其大型的数据集，您可能希望使用多台机器或云实例来加速训练。

好消息是，在多台机器上使用 DDP 与在单台机器上使用 DDP 并没有太大的区别。下一节将展示如何实现这一点。

# 分布式训练（多台机器）

如果在单台机器上训练您的 NN 模型不能满足您的需求，并且您可以访问一组服务器集群，您可以使用 PyTorch 的分布式处理能力将训练扩展到多台机器。PyTorch 的分布式子包 `torch.distributed` 提供了丰富的功能集，以适应各种训练架构和硬件平台。

`torch.distributed` 子包由三个组件组成：DDP、基于 RPC 的分布式训练（RPC）和集体通信（c10d）。我们在上一节中使用了 DDP 在单台机器上运行多个进程，它最适合数据并行处理范式。RPC 是为支持更一般的训练架构而创建的，并且可以用于除数据并行处理范式之外的分布式架构。

c10d 组件是一个用于在进程之间传输张量的通信库。c10d 被 DDP 和 RPC 组件用作后端，PyTorch 提供了 c10d API，因此您可以在自定义分布式应用中使用它。

在本书中，我们将重点介绍使用 DDP 进行分布式训练。但是，如果您有更高级的用例，您可能希望使用 RPC 或 c10d。您可以通过阅读 [PyTorch 文档](https://pytorch.tips/rpc) 了解更多信息。

对于使用 DDP 进行分布式训练，我们将遵循与在单台机器上使用多个进程相同的 DDP 过程。但是，在这种情况下，我们将在单独的机器或实例上运行每个进程。

要在多台机器上运行，我们使用一个指定配置的启动脚本来运行 DDP。启动脚本包含在 `torch.distributed` 中，并且可以按照以下代码执行。假设您有两个节点，节点 0 和节点 1。节点 0 是主节点，IP 地址为 192.168.1.1，空闲端口为 1234。在节点 0 上，您将运行以下脚本：

```py
>>>python-mtorch.distributed.launch--nproc_per_node=NUM_GPUS--nnodes=2--node_rank=0①--master_addr="192.168.1.1"--master_port=1234TRAINING_SCRIPT.py(--arg1--arg2--arg3)
```

①

`node_rank` 被设置为节点 0。

在节点 1 上，您将运行下一个脚本。请注意，此节点的等级是 `1`：

```py
>>>python-mtorch.distributed.launch--nproc_per_node=NUM_GPUS--nnodes=2--node_rank=1①--master_addr="192.168.1.1"--master_port=1234TRAINING_SCRIPT.py(--arg1--arg2--arg3)
```

①

`node_rank` 被设置为节点 1。

如果您想探索此脚本中的可选参数，请运行以下命令：

```py
>>> python -m torch.distributed.launch --help
```

请记住，如果您不使用 DDP 范式，您应该考虑为您的用例使用 RPC 或 c10d API。并行处理和分布式训练可以显著加快模型性能并减少开发时间。在下一节中，我们将考虑通过实施优化模型本身的技术来改善 NN 性能的其他方法。

# 模型优化

模型优化是一个关注 NN 模型的基础实现以及它们如何训练的高级主题。随着这一领域的研究不断发展，PyTorch 已经为模型优化添加了各种功能。在本节中，我们将探讨三个优化领域——超参数调整、量化和剪枝，并为您提供参考代码，供您在自己的设计中使用。

## 超参数调整

深度学习模型开发通常涉及选择许多用于设计模型和训练模型的变量。这些变量称为*超参数*，可以包括架构变体，如层数、层深度和核大小，以及可选阶段，如池化或批量归一化。超参数还可能包括损失函数或优化参数的变体，例如 LR 或权重衰减率。

在这一部分，我将向您展示如何使用一个名为 Ray Tune 的包来管理您的超参数优化。研究人员通常会手动测试一小组超参数。然而，Ray Tune 允许您配置您的超参数，并确定哪些设置对性能最佳。

Ray Tune 支持最先进的超参数搜索算法和分布式训练。它不断更新新功能。让我们看看如何使用 Ray Tune 进行超参数调整。

还记得我们在第三章中为图像分类训练的 LeNet5 模型吗？让我们尝试不同的模型配置和训练参数，看看我们是否可以使用超参数调整来改进我们的模型。

为了使用 Ray Tune，我们需要对我们的模型进行以下更改：

1.  定义我们的超参数及其搜索空间。

1.  编写一个函数来封装我们的训练循环。

1.  运行 Ray Tune 超参数调整。

让我们重新定义我们的模型，以便我们可以配置全连接层中节点的数量，如下面的代码所示：

```py
importtorch.nnasnnimporttorch.nn.functionalasFclassNet(nn.Module):def__init__(self,nodes_1=120,nodes_2=84):super(Net,self).__init__()self.conv1=nn.Conv2d(3,6,5)self.pool=nn.MaxPool2d(2,2)self.conv2=nn.Conv2d(6,16,5)self.fc1=nn.Linear(16*5*5,nodes_1)①self.fc2=nn.Linear(nodes_1,nodes_2)②self.fc3=nn.Linear(nodes_2,10)defforward(self,x):x=self.pool(F.relu(self.conv1(x)))x=self.pool(F.relu(self.conv2(x)))x=x.view(-1,16*5*5)x=F.relu(self.fc1(x))x=F.relu(self.fc2(x))x=self.fc3(x)returnx
```

①

配置`fc1`中的节点。

②

配置`fc2`中的节点。

到目前为止，我们有两个超参数，`nodes_1`和`nodes_2`。让我们还定义另外两个超参数，`lr`和`batch_size`，这样我们就可以在训练中改变学习率和批量大小。

在下面的代码中，我们导入`ray`包并定义超参数配置：

```py
from ray import tune
import numpy as np

config = {
  "nodes_1": tune.sample_from(
      lambda _: 2 ** np.random.randint(2, 9)),
  "nodes_2": tune.sample_from(
      lambda _: 2 ** np.random.randint(2, 9)),
  "lr": tune.loguniform(1e-4, 1e-1),
  "batch_size": tune.choice([2, 4, 8, 16])
  }
```

在每次运行期间，这些参数的值是从指定的搜索空间中选择的。您可以使用方法`tune.sample_from()`和一个`lambda`函数来定义搜索空间，或者您可以使用内置的采样函数。在这种情况下，`layer_1`和`layer_2`分别使用`sample_from()`从`2`到`9`中随机选择一个值。

`lr`和`batch_size`使用内置函数，其中`lr`被随机选择为从 1e-4 到 1e-1 的双精度数，`batch_size`被随机选择为`2`、`4`、`8`或`16`中的一个。

接下来，我们需要将我们的训练循环封装到一个函数中，该函数以配置字典作为输入。这个训练循环函数将被 Ray Tune 调用。

在编写我们的训练循环之前，让我们定义一个函数来加载 CIFAR-10 数据，这样我们可以在训练期间重复使用来自同一目录的数据。下面的代码类似于我们在第三章中使用的数据加载代码：

```py
import torch
import torchvision
from torchvision import transforms

def load_data(data_dir="./data"):
  train_transforms = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(
          (0.4914, 0.4822, 0.4465),
          (0.2023, 0.1994, 0.2010))])

  test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010))])

  trainset = torchvision.datasets.CIFAR10(
      root=data_dir, train=True,
      download=True, transform=train_transforms)

  testset = torchvision.datasets.CIFAR10(
      root=data_dir, train=False,
      download=True, transform=test_transforms)

  return trainset, testset
```

现在我们可以将训练循环封装成一个函数，*train_model()*，如下面的代码所示。这是一个大段的代码；但是，这应该对您来说很熟悉：

```py
fromtorchimportoptimfromtorchimportnnfromtorch.utils.dataimportrandom_splitdeftrain_model(config):device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")model=Net(config['nodes_1'],config['nodes_2']).to(device=device)①criterion=nn.CrossEntropyLoss()optimizer=optim.SGD(model.parameters(),lr=config['lr'],momentum=0.9)②trainset,testset=load_data()test_abs=int(len(trainset)*0.8)train_subset,val_subset=random_split(trainset,[test_abs,len(trainset)-test_abs])trainloader=torch.utils.data.DataLoader(train_subset,batch_size=int(config["batch_size"]),shuffle=True)③valloader=torch.utils.data.DataLoader(val_subset,batch_size=int(config["batch_size"]),shuffle=True)③forepochinrange(10):train_loss=0.0epoch_steps=0fordataintrainloader:inputs,labels=datainputs=inputs.to(device)labels=labels.to(device)optimizer.zero_grad()outputs=model(inputs)loss=criterion(outputs,labels)loss.backward()optimizer.step()train_loss+=loss.item()val_loss=0.0total=0correct=0fordatainvalloader:withtorch.no_grad():inputs,labels=datainputs=inputs.to(device)labels=labels.to(device)outputs=model(inputs)_,predicted=torch.max(outputs.data,1)total+=labels.size(0)correct+=\
(predicted==labels).sum().item()loss=criterion(outputs,labels)val_loss+=loss.cpu().numpy()print(f'epoch: {epoch} ',f'train_loss: ',f'{train_loss/len(trainloader)}',f'val_loss: ',f'{val_loss/len(valloader)}',f'val_acc: {correct/total}')tune.report(loss=(val_loss/len(valloader)),accuracy=correct/total)
```

①

使模型层可配置。

②

使学习率可配置。

③

使批量大小可配置。

接下来我们想要运行 Ray Tune，但首先我们需要确定我们想要使用的调度程序和报告程序。调度程序确定 Ray Tune 如何搜索和选择超参数，而报告程序指定我们希望如何查看结果。让我们在下面的代码中设置它们：

```py
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2)

reporter = CLIReporter(
    metric_columns=["loss",
                    "accuracy",
                    "training_iteration"])
```

对于调度器，我们将使用异步连续减半算法（ASHA）进行超参数搜索，并指示它最小化损失。对于报告器，我们将配置一个 CLI 报告器，以便在每次运行时在 CLI 上报告损失、准确性、训练迭代和选择的超参数。

最后，我们可以使用以下代码中显示的`run()`方法运行 Ray Tune：

```py
from functools import partial

result = tune.run(
    partial(train_model),
    resources_per_trial={"cpu": 2, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter)
```

我们提供资源并指定配置。我们传入我们的配置字典，指定样本或运行的数量，并传入我们的`scheduler`和`reporter`函数。

Ray Tune 将报告结果。`get_best_trial()`方法返回一个包含有关最佳试验信息的对象。我们可以打印出产生最佳结果的超参数设置，如下面的代码所示：

```py
best_trial = result.get_best_trial(
    "loss", "min", "last")
print("Best trial config: {}".format(
    best_trial.config))
print("Best trial final validation loss:",
      "{}".format(
          best_trial.last_result["loss"]))
print("Best trial final validation accuracy:",
      "{}".format(
          best_trial.last_result["accuracy"]))
```

您可能会发现 Ray Tune API 的其他功能有用。表 6-1 列出了`tune.schedulers`中可用的调度器。

表 6-1. Ray Tune 调度器

| 调度方法 | 描述 |
| --- | --- |
| ASHA | 运行异步连续减半算法的调度器 |
| HyperBand | 运行 HyperBand 早停算法的调度器 |
| 中位数停止规则 | 基于中位数停止规则的调度器，如[“Google Vizier: A Service for Black-Box Optimization”](https://research.google.com/pubs/pub46180.html)中所述。 |
| 基于人口的训练 | 基于人口训练算法的调度器 |
| 基于人口的训练重放 | 重放人口训练运行的调度器 |
| BOHB | 使用贝叶斯优化和 HyperBand 的调度器 |
| FIFOScheduler | 简单的调度器，按提交顺序运行试验 |
| TrialScheduler | 基于试验的调度器 |
| Shim 实例化 | 基于提供的字符串的调度器 |

更多信息可以在[Ray Tune 文档](https://pytorch.tips/ray)中找到。正如您所看到的，Ray Tune 具有丰富的功能集，但也有其他支持 PyTorch 的超参数包。这些包括[Allegro Trains](https://pytorch.tips/allegro)和[Optuna](https://pytorch.tips/optuna)。

通过找到最佳设置，超参数调整可以显著提高 NN 模型的性能。接下来，我们将探讨另一种优化模型的技术：量化。

## 量化

NNs 实现为计算图，它们的计算通常使用 32 位（或在某些情况下，64 位）浮点数。然而，我们可以使我们的计算使用低精度数字，并通过应用量化仍然实现可比较的结果。

*量化*是指使用低精度数据进行计算和访问内存的技术。这些技术可以减小模型大小，减少内存带宽，并由于内存带宽节省和使用 int8 算术进行更快的推断而执行更快的计算。

一种快速的量化方法是将所有计算精度减半。让我们再次考虑我们的 LeNet5 模型示例，如下面的代码所示：

```py
import torch
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(
            F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(
            F.relu(self.conv2(x)), 2)
        x = x.view(-1,
                   int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet5()
```

默认情况下，所有计算和内存都实现为 float32。我们可以使用以下代码检查模型参数的数据类型：

```py
for n, p in model.named_parameters():
  print(n, ": ", p.dtype)

# out:
# conv1.weight :  torch.float32
# conv1.bias :  torch.float32
# conv2.weight :  torch.float32
# conv2.bias :  torch.float32
# fc1.weight :  torch.float32
# fc1.bias :  torch.float32
# fc2.weight :  torch.float32
# fc2.bias :  torch.float32
# fc3.weight :  torch.float32
# fc3.bias :  torch.float32
```

如预期，我们的数据类型是 float32。然而，我们可以使用`half()`方法在一行代码中将模型减少到半精度：

```py
model = model.half()

for n, p in model.named_parameters():
  print(n, ": ", p.dtype)

# out:
# conv1.weight :  torch.float16
# conv1.bias :  torch.float16
# conv2.weight :  torch.float16
# conv2.bias :  torch.float16
# fc1.weight :  torch.float16
# fc1.bias :  torch.float16
# fc2.weight :  torch.float16
# fc2.bias :  torch.float16
# fc3.weight :  torch.float16
# fc3.bias :  torch.float16
```

现在我们的计算和内存值是 float16。使用`half()`通常是量化模型的一种快速简单方法。值得一试，看看性能是否适合您的用例。

然而，在许多情况下，我们不希望以相同方式量化每个计算，并且我们可能需要将量化超出 float16 值。对于这些其他情况，PyTorch 提供了三种额外的量化模式：动态量化、训练后静态量化和量化感知训练（QAT）。

当权重的计算或内存带宽限制吞吐量时，使用动态量化。这通常适用于 LSTM、RNN、双向编码器表示来自变压器（BERT）或变压器网络。当激活的内存带宽限制吞吐量时，通常适用于 CNN 的静态量化。当静态量化无法满足精度要求时，使用 QAT。

让我们为每种类型提供一些参考代码。所有类型将权重转换为 int8。它们在处理激活和内存访问方面有所不同。

*动态量化*是最简单的类型。它会将激活即时转换为 int8。计算使用高效的 int8 值，但激活以浮点格式读取和写入内存。

以下代码向您展示了如何使用动态量化量化模型：

```py
import torch.quantization

quantized_model = \
  torch.quantization.quantize_dynamic(
      model,
      {torch.nn.Linear},
      dtype=torch.qint8)
```

我们所需做的就是传入我们的模型并指定量化层和量化级别。

###### 警告

量化取决于用于运行量化模型的后端。目前，量化运算符仅在以下后端中支持 CPU 推断：x86（*fbgemm*）和 ARM（`qnnpack`）。然而，量化感知训练在完全浮点数上进行，并且可以在 GPU 或 CPU 上运行。

*后训练静态量化*可通过观察训练期间不同激活的分布，并决定在推断时如何量化这些激活来进一步降低延迟。这种类型的量化允许我们在操作之间传递量化值，而无需在内存中来回转换浮点数和整数：

```py
static_quant_model = LeNet5()
static_quant_model.qconfig = \
  torch.quantization.get_default_qconfig('fbgemm')

torch.quantization.prepare(
    static_quant_model, inplace=True)
torch.quantization.convert(
    static_quant_model, inplace=True)
```

后训练静态量化需要配置和训练以准备使用。我们配置后端以使用 x86（`fbgemm`），并调用`torch.quantization.prepare`来插入观察器以校准模型并收集统计信息。然后我们将模型转换为量化版本。

*量化感知训练*通常会产生最佳精度。在这种情况下，所有权重和激活在训练的前向和后向传递期间都被“伪量化”。浮点值四舍五入为 int8 等效值，但计算仍然以浮点数进行。也就是说，在训练期间进行量化时，权重调整是“知道”的。以下代码显示了如何使用 QAT 量化模型：

```py
qat_model = LeNet5()
qat_mode.qconfig = \
  torch.quantization.get_default_qat_qconfig('fbgemm')

torch.quantization.prepare_qat(
    qat_model, inplace=True)
torch.quantization.convert(
    qat_model, inplace=True)
```

再次，我们需要配置后端并准备模型，然后调用`convert()`来量化模型。

PyTorch 的量化功能正在不断发展，目前处于测试阶段。请参考[PyTorch 文档](https://pytorch.tips/quantization)获取有关如何使用量化包的最新信息。

## 修剪

现代深度学习模型可能具有数百万个参数，并且可能难以部署。但是，模型是过度参数化的，参数通常可以减少而几乎不影响准确性或模型性能。*修剪*是一种通过最小影响性能来减少模型参数数量的技术。这使您可以部署具有更少内存、更低功耗和减少硬件资源的模型。

### 修剪模型示例

修剪可以应用于`nn.module`。由于`nn.module`可能包含单个层、多个层或整个模型，因此可以将修剪应用于单个层、多个层或整个模型本身。让我们考虑我们的 LeNet5 模型示例：

```py
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(
            F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(
            F.relu(self.conv2(x)), 2)
        x = x.view(-1,
                   int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

我们的 LeNet5 模型有五个子模块——`conv1`、`conv2`、`fc1`、`fc2`和`fc3`。模型参数包括其权重和偏差，可以使用`named_parameters()`方法显示。让我们看看`conv1`层的参数：

```py
device = torch.device("cuda" if
  torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)

print(list(model.conv1.named_parameters()))
# out:
# [('weight', Parameter containing:
# tensor([[[[0.0560, 0.0066, ..., 0.0183, 0.0783]]]],
#        device='cuda:0',
#        requires_grad=True)),
#  ('bias', Parameter containing:
# tensor([0.0754, -0.0356, ..., -0.0111, 0.0984],
#        device='cuda:0',
#        requires_grad=True))]
```

### 本地和全局修剪

*本地修剪*是指我们仅修剪模型的特定部分。通过这种技术，我们可以将本地修剪应用于单个层或模块。只需调用修剪方法，传入层，并设置其选项如下代码所示：

```py
import torch.nn.utils.prune as prune

prune.random_unstructured(model.conv1,
                          name="weight",
                          amount=0.25)
```

此示例将随机非结构化修剪应用于我们模型中`conv1`层中名为`weight`的参数。这只修剪了权重参数。我们也可以使用以下代码修剪偏置参数：

```py
prune.random_unstructured(model.conv1,
                          name="bias",
                          amount=0.25)
```

修剪可以进行迭代应用，因此您可以使用不同维度上的其他修剪方法进一步修剪相同的参数。

您可以以不同方式修剪模块和参数。例如，您可能希望按模块或层类型修剪，并将修剪应用于卷积层和线性层的方式不同。以下代码演示了一种方法： 

```py
model=LeNet5().to(device)forname,moduleinmodel.named_modules():ifisinstance(module,torch.nn.Conv2d):prune.random_unstructured(module,name='weight',amount=0.3)①elifisinstance(module,torch.nn.Linear):prune.random_unstructured(module,name='weight',amount=0.5)// ②
```

①

通过 30%修剪所有 2D 卷积层。

②

通过 50%修剪所有线性层。

另一个使用修剪 API 的方法是应用*全局修剪*，即我们将修剪方法应用于整个模型。例如，我们可以全局修剪我们模型参数的 25%，这可能会导致每个层的不同修剪率。以下代码演示了一种应用全局修剪的方法：

```py
model = LeNet5().to(device)

parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.25)
```

在这里我们修剪整个模型中所有参数的 25%。

### 修剪 API

PyTorch 在其`torch.nn.utils.prune`模块中提供了对修剪的内置支持。表 6-2 列出了修剪 API 中可用的函数。

表 6-2\. 修剪函数

| 函数 | 描述 |
| --- | --- |
| `is_pruned(*module*)` | 检查模块是否已修剪 |
| `remove(*module*, *name*)` | 从模块中删除修剪重参数化和从前向钩子中删除修剪方法 |
| `custom_from_mask(*module*, *name*, *mask*)` | 通过应用`mask`中的预先计算的掩码，修剪与`module`中名为`name`的参数对应的张量 |
| `global_unstructured(*params*, *pruning_method*)` | 通过应用指定的`pruning_method`，全局修剪与`params`中所有参数对应的张量 |
| `ln_structured(*module*, *name*, *amount*, *n*, *dim*)` | 通过移除指定`dim`上具有最低 L`n`-范数的（当前未修剪的）通道，修剪与`module`中名为`name`的参数对应的张量中的指定`amount` |
| `random_structured(*module*, *name*, *amount*, *dim*)` | 通过随机选择指定`dim`上的通道，移除与`module`中名为`name`的参数对应的张量中的指定`amount`（当前未修剪的）通道 |
| `l1_unstructured(*module*, *name*, *amount*)` | 通过移除具有最低 L1-范数的指定`amount`（当前未修剪的）单元，修剪与`module`中名为`name`的参数对应的张量 |
| `random_unstructured(*module*, *name*, *amount*)` | 通过随机选择指定`amount`的（当前未修剪的）单元，修剪与`module`中名为`name`的参数对应的张量 |

### 自定义修剪方法

如果找不到适合您需求的修剪方法，您可以创建自己的修剪方法。为此，请从`torch.nn.utils.prune`中提供的`BasePruningMethod`类创建一个子类。在大多数情况下，您可以将`call()`、`apply_mask()`、`apply()`、`prune()`和`remove()`方法保持原样。

但是，您需要编写自己的`__init__()`构造函数和`compute_mask()`方法来描述您的修剪方法如何计算掩码。此外，您需要指定修剪类型（`structured`、`unstructured`或`global`）。以下代码显示了一个示例：

```py
class MyPruningMethod(prune.BasePruningMethod):
  PRUNING_TYPE = 'unstructured'

  def compute_mask(self, t, default_mask):
    mask = default_mask.clone()
    mask.view(-1)[::2] = 0
    return mask

def my_unstructured(module, name):
  MyPruningMethod.apply(module, name)
  return module
```

首先我们定义类。此示例根据`compute_mask()`中的代码定义了每隔一个参数进行修剪。`PRUNING_TYPE`用于配置修剪类型为`unstructured`。然后我们包含并应用一个实例化该方法的函数。您可以按以下方式将此修剪应用于您的模型：

```py
model = LeNet5().to(device)
my_unstructured(model.fc1, name='bias')
```

您现在已经创建了自己的自定义修剪方法，并可以在本地或全局应用它。

本章向您展示了如何使用 PyTorch 加速培训并优化模型。下一步是将您的模型和创新部署到世界上。在下一章中，您将学习如何将您的模型部署到云端、移动设备和边缘设备，并且我将提供一些参考代码来构建快速应用程序，展示您的设计。
