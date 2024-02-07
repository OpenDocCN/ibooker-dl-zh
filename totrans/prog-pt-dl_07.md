# 第七章. 调试 PyTorch 模型

到目前为止，我们在本书中创建了许多模型，但在本章中，我们简要地看一下如何解释它们并弄清楚底层发生了什么。我们看一下如何使用 PyTorch 钩子和类激活映射来确定模型决策的焦点，以及如何将 PyTorch 连接到 Google 的 TensorBoard 进行调试。我将展示如何使用火焰图来识别转换和训练管道中的瓶颈，并提供一个加速缓慢转换的示例。最后，我们将看看如何在处理更大的模型时通过*检查点*来交换计算和内存。不过，首先，简要谈谈您的数据。

# 凌晨 3 点。你的数据在做什么？

在我们深入研究像 TensorBoard 或梯度检查点这样的闪亮东西之前，问问自己：您了解您的数据吗？如果您正在对输入进行分类，您是否在所有可用标签上拥有平衡的样本？在训练、验证和测试集中？

而且，您确定您的标签是*正确的吗？*像 MNIST 和 CIFAR-10（加拿大高级研究所）这样的重要基于图像的数据集已知包含一些不正确的标签。您应该检查您的数据，特别是如果类别彼此相似，比如狗品种或植物品种。简单地对数据进行合理性检查可能会节省大量时间，如果您发现，比如说，一个标签类别只有微小的图像，而其他所有类别都有大分辨率的示例。

一旦您确保数据处于良好状态，那么是的，让我们转到 TensorBoard 开始检查模型中的一些可能问题。

# TensorBoard

*TensorBoard*是一个用于可视化神经网络各个方面的 Web 应用程序。它允许轻松实时查看诸如准确性、损失激活值等统计数据，以及您想要发送的任何内容。尽管它是为 TensorFlow 编写的，但它具有如此通用和相当简单的 API，以至于在 PyTorch 中使用它与在 TensorFlow 中使用它并没有太大不同。让我们安装它，看看我们如何使用它来获取有关我们模型的一些见解。

###### 注意

在阅读 PyTorch 时，您可能会遇到一个名为[Visdom](https://oreil.ly/rZqv2)的应用程序，这是 Facebook 对 TensorBoard 的替代方案。在 PyTorch v1.1 之前，支持可视化的方式是使用 Visdom 与 PyTorch，同时第三方库如`tensorboardX`可用于与 TensorBoard 集成。虽然 Visdom 仍在维护，但在 v1.1 及以上版本中包含了官方的 TensorBoard 集成，这表明 PyTorch 的开发人员已经认识到 TensorBoard 是事实上的神经网络可视化工具。

## 安装 TensorBoard

安装 TensorBoard 可以使用`pip`或`conda`：

```py
pip install tensorboard
conda install tensorboard
```

###### 注意

PyTorch 需要 v1.14 或更高版本的 TensorBoard。

然后可以在命令行上启动 TensorBoard：

```py
tensorboard --logdir=runs
```

然后，您可以转到*http://`[your-machine]`:6006*，您将看到图 7-1 中显示的欢迎屏幕。现在我们可以向应用程序发送数据。

![Tensorboard](img/ppdl_0701.png)

###### 图 7-1. TensorBoard

## 将数据发送到 TensorBoard

使用 PyTorch 的 TensorBoard 模块位于`torch.utils.tensorboard`中：

```py
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('example', 3)
```

我们使用`SummaryWriter`类与 TensorBoard 通信，使用标准的日志输出位置*./runs*，可以通过使用带有标签的`add_scalar`发送标量。由于`SummaryWriter`是异步工作的，可能需要一会儿，但您应该看到 TensorBoard 更新，如图 7-2 所示。

![Tensorboard 中的示例数据点](img/ppdl_0702.png)

###### 图 7-2. TensorBoard 中的示例数据点

这并不是很令人兴奋，对吧？让我们写一个循环，从初始起点发送更新：

```py
import random
value = 10
writer.add_scalar('test_loop', value, 0)
for i in range(1,10000):
  value += random.random() - 0.5
  writer.add_scalar('test_loop', value, i)
```

通过传递我们在循环中的位置，如图 7-3 所示，TensorBoard 会给我们一个绘制我们从 10 开始进行的随机漫步的图。如果我们再次运行代码，我们会看到它在显示中生成了一个不同的*run*，我们可以在网页的左侧选择是否要查看所有运行或只查看特定的一些。

![在 tensorboard 中绘制随机漫步](img/ppdl_0703.png)

###### 图 7-3\. 在 TensorBoard 中绘制随机漫步

我们可以用这个函数来替换训练循环中的`print`语句。我们也可以发送模型本身以在 TensorBoard 中得到表示！

```py
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms,models

writer = SummaryWriter()
model = models.resnet18(False)
writer.add_graph(model,torch.rand([1,3,224,224]))

def train(model, optimizer, loss_fn, train_data_loader, test_data_loader, epochs=20):
    model = model.train()
    iteration = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input, target = batch
            output = model(input)
            loss = loss_fn(output, target)
            writer.add_scalar('loss', loss, epoch)
            loss.backward()
            optimizer.step()

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            input, target = batch
            output = model(input)
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
            print("Epoch {}, accuracy = {:.2f}".format(epoch,
                   num_correct / num_examples)
            writer.add_scalar('accuracy', num_correct / num_examples, epoch)
        iterations += 1
```

当使用`add_graph()`时，我们需要发送一个张量来跟踪模型，以及模型本身。一旦发生这种情况，你应该在 TensorBoard 中看到`GRAPHS`出现，并且如图 7-4 所示，点击大的 ResNet 块会显示模型结构的更多细节。

![可视化 ResNet](img/ppdl_0704.png)

###### 图 7-4\. 可视化 ResNet

现在我们可以将准确性和损失信息以及模型结构发送到 TensorBoard。通过聚合多次运行的准确性和损失信息，我们可以看到特定运行与其他运行有何不同，这在尝试弄清楚为什么训练运行产生糟糕结果时是一个有用的线索。我们很快会回到 TensorBoard，但首先让我们看看 PyTorch 为调试提供的其他功能。

## PyTorch 钩子

PyTorch 有*钩子*，它们是可以附加到张量或模块的前向或后向传递的函数。当 PyTorch 在传递过程中遇到带有钩子的模块时，它会调用已注册的钩子。在张量上注册的钩子在计算其梯度时会被调用。

钩子是操纵模块和张量的潜在强大方式，因为如果你愿意，你可以完全替换钩子中的输出。你可以改变梯度，屏蔽激活，替换模块中的所有偏置等等。然而，在本章中，我们只会将它们用作在数据流过程中获取有关网络信息的一种方式。

给定一个 ResNet-18 模型，我们可以使用`register_forward_hook`在模型的特定部分附加一个前向钩子：

```py
def print_hook(self, module, input, output):
  print(f"Shape of input is {input.shape}")

model = models.resnet18()
hook_ref  = model.fc.register_forward_hook(print_hook)
model(torch.rand([1,3,224,224]))
hook_ref.remove()
model(torch.rand([1,3,224,224]))
```

如果你运行这段代码，你应该会看到打印出的文本，显示模型的线性分类器层的输入形状。请注意，第二次通过模型传递随机张量时，你不应该看到`print`语句。当我们向模块或张量添加钩子时，PyTorch 会返回对该钩子的引用。我们应该始终保存该引用（这里我们在`hook_ref`中这样做），然后在完成时调用`remove()`。如果你不保存引用，那么它将一直存在并占用宝贵的内存（并在传递过程中浪费计算资源）。反向钩子的工作方式相同，只是你要调用`register_backward_hook()`。

当然，如果我们可以`print()`某些内容，我们肯定可以将其发送到 TensorBoard！让我们看看如何使用钩子和 TensorBoard 来获取训练过程中关于我们层的重要统计信息。

## 绘制均值和标准差

首先，我们设置一个函数，将输出层的均值和标准差发送到 TensorBoard：

```py
def send_stats(i, module, input, output):
  writer.add_scalar(f"{i}-mean",output.data.std())
  writer.add_scalar(f"{i}-stddev",output.data.std())
```

我们不能单独使用这个来设置一个前向钩子，但是使用 Python 函数`partial()`，我们可以创建一系列前向钩子，它们将自动附加到具有设置`i`值的层，以确保正确的值被路由到 TensorBoard 中的正确图表中：

```py
from functools import partial

for i,m in enumerate(model.children()):
  m.register_forward_hook(partial(send_stats, i))
```

请注意，我们正在使用`model.children()`，它只会附加到模型的每个顶层块，因此如果我们有一个`nn.Sequential()`层（在基于 ResNet 的模型中会有），我们只会将钩子附加到该块，而不是每个`nn.Sequential`列表中的单个模块。

如果我们使用通常的训练函数训练我们的模型，我们应该看到激活开始流入 TensorBoard，如图 7-5 所示。您将不得不在 UI 中切换到挂钟时间，因为我们不再使用钩子将*步骤*信息发送回 TensorBoard（因为我们只在调用 PyTorch 钩子时获取模块信息）。

![Tensorboard 中模块的均值和标准差](img/ppdl_0705.png)

###### 图 7-5。TensorBoard 中模块的均值和标准差

现在，我在第二章中提到，理想情况下，神经网络中的层应该具有均值为 0，标准差为 1，以确保我们的计算不会无限制地增长或减少到零。查看 TensorBoard 中的层。它们看起来是否保持在这个值范围内？图表有时会突然上升然后崩溃吗？如果是这样，这可能是网络训练困难的信号。在图 7-5 中，我们的均值接近零，但标准差也非常接近零。如果您的网络的许多层中都发生这种情况，这可能表明您的激活函数（例如`ReLU`）并不完全适合您的问题领域。尝试使用其他函数进行实验，看看它们是否可以提高模型的性能；PyTorch 的`LeakyReLU`是一个很好的替代品，提供与标准`ReLU`类似的激活，但可以传递更多信息，这可能有助于训练。

关于 TensorBoard 的介绍就到这里，但是“进一步阅读”将指引您查阅更多资源。与此同时，让我们看看如何让模型解释它是如何做出决定的。

## 类激活映射

*类激活映射*（CAM）是一种在网络对传入张量进行分类后可视化激活的技术。在基于图像的分类器中，通常显示为热图覆盖在原始图像上，如图 7-6 所示。

使用 Casper 生成类激活映射

###### 图 7-6。Casper 的类激活映射

从热图中，我们可以直观地了解网络是如何从可用的 ImageNet 类中决定*波斯猫*的。网络的激活在猫的脸部和身体周围最高，在图像的其他地方较低。

要生成热图，我们捕获网络的最终卷积层的激活，就在它进入“线性”层之前，因为这样我们可以看到组合的 CNN 层认为在从图像到类的最终映射中重要的是什么。幸运的是，有了 PyTorch 的钩子功能，这是相当简单的。我们将钩子封装在一个类`SaveActivations`中：

```py
class SaveActivations():
    activations=None
    def __init__(self, m):
      self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
      self.features = output.data
    def remove(self):
      self.hook.remove()
```

然后，我们将 Casper 的图像通过网络（对 ImageNet 进行归一化），应用`softmax`将输出张量转换为概率，并使用`torch.topk()`作为提取最大概率及其索引的方法：

```py
import torch
from torchvision import models, transforms
from torch.nn import functional as F

casper = Image.open("casper.jpg")
# Imagenet mean/std

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

display_transform = transforms.Compose([
   transforms.Resize((224,224))])

casper_tensor = preprocess(casper)

model = models.resnet18(pretrained=True)
model.eval()
casper_activations = SaveActivations(model.layer_4)
prediction = model(casper_tensor.unsqueeze(0))
pred_probabilities = F.softmax(prediction).data.squeeze()
casper_activations.remove()
torch.topk(pred_probabilities,1)
```

###### 注意

我还没有解释`torch.nn.functional`，但最好的理解方法是它包含在`torch.nn`中提供的*函数*的实现。例如，如果您创建`torch.nn.softmax()`的实例，您将获得一个具有执行`softmax`的`forward()`方法的对象。如果您查看`torch.nn.softmax()`的实际源代码，您会看到该方法只是调用`F.softmax()`。由于我们不需要将`softmax`作为网络的一部分，我们只是调用底层函数。

如果我们现在访问`casper_activations.activations`，我们将看到它已经被一个张量填充，其中包含我们需要的最终卷积层的激活。然后我们这样做：

```py
fts = sf[0].features[idx]
        prob = np.exp(to_np(log_prob))
        preds = np.argmax(prob[idx])
        fts_np = to_np(fts)
        f2=np.dot(np.rollaxis(fts_np,0,3), prob[idx])
        f2-=f2.min()
        f2/=f2.max()
        f2
plt.imshow(dx)
plt.imshow(scipy.misc.imresize(f2, dx.shape), alpha=0.5, cmap='jet');
```

这计算了来自 Casper 的激活的点积（我们索引为 0 是因为输入张量的第一维中有批处理，记住）。如第一章中提到的，PyTorch 以 C × H × W 格式存储图像数据，因此我们接下来需要将维度重新排列为 H × W × C 以显示图像。然后，我们从张量中去除最小值，并通过最大值进行缩放，以确保我们只关注结果热图中最高的激活（即，与*波斯猫*相关的内容）。最后，我们使用一些`matplot`魔法来显示 Casper，然后在顶部显示张量，调整大小并给出标准的`jet`颜色映射。请注意，通过用不同的类替换`idx`，您可以看到热图指示图像中存在哪些激活（如果有的话）在分类时。因此，如果模型预测*汽车*，您可以看到图像的哪些部分被用来做出这个决定。Casper 的第二高概率是*安哥拉兔*，我们可以从该索引的 CAM 中看到它专注于他非常蓬松的毛皮！

我们已经了解了模型在做出决策时的情况。接下来，我们将调查模型在训练循环或推断期间大部分时间都在做什么。

# 火焰图

与 TensorBoard 相比，*火焰图*并不是专门为神经网络创建的。不，甚至不是为了 TensorFlow。事实上，火焰图的起源可以追溯到 2011 年，当时一位名叫 Brendan Gregg 的工程师在一家名为 Joyent 的公司工作，他想出了这种技术来帮助调试他在 MySQL 中遇到的问题。这个想法是将大量的堆栈跟踪转换成单个图像，这本身就可以呈现出 CPU 在一段时间内的运行情况。

###### 注意

Brendan Gregg 现在在 Netflix 工作，并有大量与性能相关的工作可供阅读和消化。

以 MySQL 插入表中的一行为例，我们每秒对*堆栈*进行数百次或数千次的采样。每次采样时，我们会得到一个*堆栈跟踪*，显示出该时刻堆栈中的所有函数。因此，如果我们在一个被另一个函数调用的函数中，我们将得到一个包含调用者和被调用者函数的跟踪。一个采样跟踪看起来像这样：

```py
65.00%     0.00%  mysqld   [kernel.kallsyms]   [k] entry_SYSCALL_64_fastpath
             |
             ---entry_SYSCALL_64_fastpath
                |
                |--18.75%-- sys_io_getevents
                |          read_events
                |          schedule
                |          __schedule
                |          finish_task_switch
                |
                |--10.00%-- sys_fsync
                |          do_fsync
                |          vfs_fsync_range
                |          ext4_sync_file
                |          |
                |          |--8.75%-- jbd2_complete_transaction
                |          |          jbd2_log_wait_commit
                |          |          |
                |          |          |--6.25%-- _cond_resched
                |          |          |          preempt_schedule_common
                |          |          |          __schedule
```

这里有很多信息；这只是一个 400KB 堆栈跟踪集的一个小样本。即使有这种整理（可能不是所有堆栈跟踪中都有），要看清楚这里发生了什么也是很困难的。

另一方面，火焰图版本简单明了，如您在图 7-7 中所见。y 轴是堆栈高度，x 轴是，虽然*不是时间*，但表示了在采样时该函数在堆栈中出现的频率。因此，如果我们在堆栈顶部有一个函数占据了 80%的图形，我们就会知道程序在该函数中花费了大量的运行时间，也许我们应该查看该函数，看看是什么让它运行如此缓慢。

![MySQL 火焰图](img/ppdl_0707.png)

###### 图 7-7\. MySQL 火焰图

您可能会问，“这与深度学习有什么关系？”好吧，没错；在深度学习研究中，一个常见的说法是，当训练变慢时，您只需再购买 10 个 GPU 或向谷歌支付更多 TPU Pod 的费用。但也许您的训练流水线并不是完全受 GPU 限制。也许您有一个非常慢的转换，当您获得所有那些闪亮的新显卡时，它们并没有像您想象的那样有所帮助。火焰图提供了一种简单、一目了然的方法来识别 CPU 限制的瓶颈，这在实际的深度学习解决方案中经常发生。例如，还记得我们在第四章中谈到的所有基于图像的转换吗？大多数都使用 Python Imaging Library，并且完全受 CPU 限制。对于大型数据集，您将在训练循环中一遍又一遍地执行这些转换！因此，虽然它们在深度学习的背景下并不经常被提及，但火焰图是您工具箱中很好的工具。如果没有其他办法，您可以将它们用作向老板证明您确实受到 GPU 限制，并且您需要在下周四之前获得所有那些 TPU 积分！我们将看看如何从您的训练周期中获取火焰图，并通过将慢转换从 CPU 移动到 GPU 来修复它。

## 安装 py-spy

有许多方法可以生成可以转换为火焰图的堆栈跟踪。前一节中生成的是使用 Linux 工具`perf`生成的，这是一个复杂而强大的工具。我们将采取一个相对简单的选项，并使用`py-spy`，一个基于 Rust 的堆栈分析器，直接生成火焰图。通过`pip`安装它：

```py
pip install py-spy
```

您可以通过使用`--pid`参数找到正在运行进程的进程标识符（PID），并附加`py-spy`：

```py
py-spy --flame profile.svg --pid 12345
```

或者您可以传入一个 Python 脚本，这是我们在本章中运行它的方式。首先，让我们在一个简单的 Python 脚本上运行它：

```py
import torch
import torchvision

def get_model():
    return torchvision.models.resnet18(pretrained=True)

def get_pred(model):
    return model(torch.rand([1,3,224,224]))

model = get_model()

for i in range(1,10000):
    get_pred(model)
```

将此保存为*flametest.py*，然后让我们在其上运行`py-spy`，每秒采样 99 次，运行 30 秒：

```py
py-spy -r 99 -d 30 --flame profile.svg -- python t.py
```

在浏览器中打开*profile.svg*文件，让我们看看生成的图形。

## 阅读火焰图

图 7-8 展示了图形大致应该是什么样子（由于采样的原因，它在您的机器上可能不会完全像这样）。您可能首先注意到的是图形是向下的，而不是向上的。`py-spy`以*icicle*格式编写火焰图，因此堆栈看起来像钟乳石，而不是经典火焰图的火焰。我更喜欢正常格式，但`py-spy`不提供更改选项，而且这并没有太大的区别。

![ResNet 加载和推理的火焰图](img/ppdl_0708.png)

###### 图 7-8\. ResNet 加载和推理的火焰图

一眼看去，您应该看到大部分执行时间都花在各种`forward()`调用中，这是有道理的，因为我们正在使用模型进行大量预测。左侧的那些小块呢？如果您单击它们，您会发现 SVG 文件会放大，如图 7-9 所示。

![放大的火焰图](img/ppdl_0709.png)

###### 图 7-9\. 放大的火焰图

在这里，我们可以看到脚本设置了 ResNet-18 模块，并调用`load_state_dict()`来从磁盘加载保存的权重（因为我们使用`pretrained=True`调用它）。您可以单击“重置缩放”以返回完整的火焰图。此外，右侧的搜索栏将用紫色突出显示匹配的条形，如果您试图查找一个函数。尝试使用*resnet*，它将显示堆栈中名称中带有*resnet*的每个函数调用。这对于查找不经常出现在堆栈中的函数或查看该模式在整个图中出现的频率很有用。

玩一下 SVG，看看在这个示例中 BatchNorm 和池化等东西占用了多少 CPU 时间。接下来，我们将看一种使用火焰图来查找问题、修复问题并使用另一个火焰图验证的方法。

## 修复慢转换

在现实情况下，你的数据管道的一部分可能会导致减速。如果你有一个慢转换，这将是一个特别的问题，因为它将在训练批次期间被调用多次，导致在创建模型时出现巨大的瓶颈。这里是一个示例转换管道和一个数据加载器：

```py
import torch
import torchvision
from torch import optim
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.utils.data
from PIL import Image
import numpy as np

device = "cuda:0"
model = models.resnet18(pretrained=True)
model.to(device)

class BadRandom(object):
    def __call__(self, img):
        img_np = np.array(img)
        random = np.random.random_sample(img_np.shape)
        out_np = img_np + random
        out = Image.fromarray(out_np.astype('uint8'), 'RGB')
        return out

    def __repr__(self):
        str = f"{self.__class__.__name__  }"
        return str

train_data_path = "catfish/train"
image_transforms =
torchvision.transforms.Compose(
  [transforms.Resize((224,224)),BadRandom(), transforms.ToTensor()])
```

我们不会运行完整的训练循环；相反，我们模拟了从训练数据加载器中提取图像的 10 个时期：

```py
train_data = torchvision.datasets.ImageFolder(root=train_data_path,
transform=image_transforms)
batch_size=32
train_data_loader = torch.utils.data.DataLoader(train_data,
batch_size=batch_size)

optimizer = optim.Adam(model.parameters(), lr=2e-2)
criterion = nn.CrossEntropyLoss()

def train(model, optimizer, loss_fn,  train_loader, val_loader,
epochs=20, device='cuda:0'):
    model.to(device)
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            ww, target = batch
            ww = ww.to(device)
            target= target.to(device)
            output = model(ww)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            input, target = batch
            input = input.to(device)
            target= target.to(device)
            output = model(input)
            correct = torch.eq(torch.max(output, dim=1)[1], target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        print("Epoch {}, accuracy = {:.2f}"
        .format(epoch, num_correct / num_examples))

train(model,optimizer,criterion,
train_data_loader,train_data_loader,epochs=10)
```

让我们像以前一样在`py-spy`下运行该代码：

```py
py-spy -r 99 -d 120 --flame slowloader.svg -- python slowloader.py
```

如果你打开生成的*slowloader.svg*，你应该会看到类似于图 7-10 的东西。尽管火焰图大部分时间都被用于加载图像并将其转换为张量，但我们在应用随机噪声上花费了采样运行时间的 16.87%。看看代码，我们的`BadRandom`实现是在 PIL 阶段应用噪声，而不是在张量阶段，所以我们受制于图像处理库和 NumPy，而不是 PyTorch 本身。因此，我们的第一个想法可能是重写转换，使其在张量而不是 PIL 图像上操作。这可能会更快，但并非总是如此——在进行性能更改时的重要事情始终是要测量一切。

![带有 BadRandom 的火焰图](img/ppdl_0710.png)

###### 图 7-10。带有 BadRandom 的火焰图

但有一件奇怪的事情，一直贯穿整本书，尽管我直到现在才注意到它：你是否注意到我们从数据加载器中提取批次，然后将这些批次放入 GPU？因为转换发生在加载器从数据集类获取批次时，这些转换总是会在 CPU 上发生。在某些情况下，这可能会导致一些疯狂的横向思维。我们在每个图像上应用随机噪声。如果我们能一次在每个图像上应用随机噪声呢？

这里可能一开始看起来有点费解的部分是：我们向图像添加随机噪声。我们可以将其写为*x + y*，其中*x*是我们的图像，*y*是我们的噪声。我们知道图像和噪声都是 3D 的（宽度、高度、通道），所以这里我们所做的就是矩阵乘法。在一个批次中，我们将这样做*z*次。我们只是在从加载器中取出每个图像时对每个图像进行迭代。但请考虑，在加载过程结束时，图像被转换为张量，一个批次的*[z, c, h, w]*。那么，你难道不能只是添加一个形状为*[z, c, h, w]*的随机张量，以这种方式应用随机噪声吗？而不是按顺序应用噪声，它一次性完成。现在我们有了一个矩阵运算，以及一个非常昂贵的 GPU，它碰巧非常擅长矩阵运算。在 Jupyter Notebook 中尝试这样做，看看 CPU 和 GPU 张量矩阵操作之间的差异：

```py
cpu_t1 = torch.rand(64,3,224,224)
cpu_t2 = torch.rand(64,3,224,224)
%timeit cpu_t1 + cpu_t2
>> 5.39 ms ± 4.29 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

gpu_t1 = torch.rand(64,3,224,224).to("cuda")
gpu_t2 = torch.rand(64,3,224,224).to("cuda")
%timeit gpu_t1 + gpu_t2
>> 297 µs ± 338 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

这样做的速度快了近 20 倍。我们可以将这个转换从我们的数据加载器中取出，等到整个批次都准备好后再执行矩阵运算：

```py
def add_noise_gpu(tensor, device):
  random_noise = torch_rand_like(tensor).to(device)
  return tensor.add_(random_noise)
```

在我们的训练循环中，在`input.to(device)`之后添加这行：

```py
input = add_noise_gpu(input, device)
```

然后从转换管道中移除`BadRandom`转换，并使用`py-spy`再次进行测试。新的火焰图显示在图 7-11 中。它如此之快，以至于它甚至不再在我们的采样频率下显示。我们刚刚将代码加速了近 17％！现在，并非所有标准转换都可以以 GPU 友好的方式编写，但如果可能的话，如果转换正在减慢你的速度，那么这绝对是一个值得考虑的选项。

![带有 GPU 加速随机噪声的火焰图](img/ppdl_0711.png)

###### 图 7-11。带有 GPU 加速随机噪声的火焰图

现在我们已经考虑了计算，是时候看看房间里的另一个大象了：内存，特别是 GPU 上的内存。

# 调试 GPU 问题

在本节中，我们将更深入地研究 GPU 本身。在训练更大的深度学习模型时，您很快会发现，您花了很多钱购买的闪亮 GPU（或者更明智地，连接到基于云的实例）经常陷入困境，痛苦地抱怨内存不足。但是那个 GPU 有几千兆字节的存储空间！您怎么可能用完？

模型往往会占用大量内存。例如，ResNet-152 大约有 6000 万个激活，所有这些都占据了 GPU 上宝贵的空间。让我们看看如何查看 GPU 内部，以确定在内存不足时可能发生了什么。

## 检查您的 GPU

假设您正在使用 NVIDIA GPU（如果使用其他设备，请查看备用 GPU 供应商的驱动程序网站以获取他们自己的实用程序），CUDA 安装包括一个非常有用的命令行工具，称为`nvidia-smi`。当不带参数运行时，此工具可以为您提供有关 GPU 上使用的内存的快照，甚至更好的是，是谁在使用它！图 7-12 显示了在终端中运行`nvidia-smi`的输出。在笔记本中，您可以通过使用`!nvidia-smi`调用该实用程序。

![从 nvidia-smi 输出](img/ppdl_0712.png)

###### 图 7-12。从 nvidia-smi 输出

这个示例来自我家里运行的一台 1080 Ti 机器。我正在运行一堆笔记本，每个笔记本都占用了一部分内存，但有一个占用了 4GB！您可以使用`os.getpid()`获取笔记本的当前 PID。结果表明，占用最多内存的进程实际上是我用来测试上一节中 GPU 变换的实验性笔记本！您可以想象，随着模型、批数据以及前向和后向传递的数据，内存很快会变得紧张。

###### 注意

我还有一些进程在运行，也许令人惊讶的是，正在进行图形处理——即 X 服务器和 GNOME。除非您构建了本地机器，否则几乎肯定看不到这些。

此外，PyTorch 将为每个进程分配大约 0.5GB 的内存给自身和 CUDA。这意味着最好一次只处理一个项目，而不要像我这样到处运行 Jupyter Notebook（您可以使用内核菜单关闭与笔记本连接的 Python 进程）。

仅运行`nvidia-smi`将为您提供 GPU 使用情况的当前快照，但您可以使用`-l`标志获得持续输出。以下是一个示例命令，每 5 秒将转储时间戳、已使用内存、空闲内存、总内存和 GPU 利用率：

```py
nvidia-smi --query-gpu=timestamp,
memory.used, memory.free,memory.total,utilization.gpu --format=csv -l 5
```

如果您真的认为 GPU 使用的内存比应该使用的要多，可以尝试让 Python 的垃圾收集器参与其中。如果您有一个不再需要的`tensor_to_be_deleted`，并且希望它从 GPU 中消失，那么来自 fast.ai 库深处的一个提示是使用`del`将其推开：

```py
import gc
del tensor_to_be_deleted
gc.collect()
```

如果您在 Jupyter Notebook 中进行大量工作，创建和重新创建模型，可能会发现删除一些引用并通过使用`gc.collect()`调用垃圾收集器将收回一些内存。如果您仍然遇到内存问题，请继续阅读，因为可能会有解决您困扰的答案！

## 梯度检查点

尽管在上一节中介绍了所有删除和垃圾收集技巧，您可能仍然会发现自己内存不足。对于大多数应用程序来说，下一步要做的事情是减少在训练循环中通过模型的数据批量大小。这样做会起作用，但您将增加每个时代的训练时间，并且很可能模型不会像使用足够内存处理更大批量大小的等效模型那样好，因为您将在每次传递中看到更多数据集。但是，我们可以通过使用*梯度检查点*在 PyTorch 中为大型模型交换计算和内存。

处理更大模型时的一个问题是，前向和后向传递会产生大量中间状态，所有这些状态都会占用 GPU 内存。梯度检查点的目标是通过*分段*模型来减少可能同时存在于 GPU 上的状态量。这种方法意味着您可以在非分段模型的情况下具有四到十倍的批量大小，但这会使训练更加计算密集。在前向传递期间，PyTorch 会将输入和参数保存到一个段中，但实际上不执行前向传递。在后向传递期间，PyTorch 会检索这些内容，并为该段计算前向传递。中间值会传递到下一个段，但这些值必须仅在段与段之间执行。

将模型分割成这些段的工作由`torch.utils.checkpoint.checkpoint_sequential()`处理。它适用于`nn.Sequential`层或生成的层列表，但需要注意它们需要按照模型中出现的顺序排列。以下是它在 AlexNet 的`features`模块上的工作方式：

```py
from torch.utils.checkpoint import checkpoint_sequential
import torch.nn as nn

class CheckpointedAlexNet(nn.Module):

    def __init__(self, num_classes=1000, chunks=2):
        super(CheckpointedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = checkpoint_sequential(self.features, chunks, x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
```

正如您所看到的，当需要时，检查点是模型的一个简单补充。我们在新版本的模型中添加了一个`chunks`参数，默认情况下将其分成两个部分。然后，我们只需要调用`checkpoint_sequential`与`features`模块，段数和我们的输入。就是这样！

在检查点中的一个小问题是，它与`BatchNorm`或`Dropout`层的交互方式会导致不良行为。为了解决这个问题，您可以在这些层之前和之后只检查点模型的部分。在我们的`CheckpointedAlexNet`中，我们可以将`classifier`模块分成两部分：一个包含未检查点的`Dropout`层，以及一个包含我们的`Linear`层的最终`nn.Sequential`模块，我们可以以与`features`相同的方式检查点。

如果您发现为了使模型运行而减少批量大小，请在要求更大的 GPU 之前考虑检查点！

# 结论

希望现在您已经具备了在训练模型不如预期时寻找答案的能力。从清理数据到运行火焰图或 TensorBoard 可视化，您有很多工具可供使用；您还看到了如何通过 GPU 转换以及使用检查点来交换内存和计算。

拥有经过适当训练和调试的模型，我们正走向最严酷的领域：*生产*。

# 进一步阅读

+   [TensorBoard 文档](https://oreil.ly/MELKl)

+   [TensorBoard GitHub](https://oreil.ly/21bIM)

+   Fast.ai 第 10 课：[深入了解模型](https://oreil.ly/K4dz-)

+   [对 ResNet 模型中 BatchNorm 的调查](https://oreil.ly/EXdK3)

+   深入探讨如何使用 Brendan Gregg 生成[火焰图](https://oreil.ly/4Ectg)

+   [nvidia-smi 文档](https://oreil.ly/W1g0n)

+   [PyTorch 梯度检查点文档](https://oreil.ly/v0apy)
