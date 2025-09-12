# 附录 A. 安装 Python、Jupyter Notebook 和 PyTorch

在您的计算机上安装 Python 和管理库和包的多种方式存在。本书使用 Anaconda，这是一个开源的 Python 发行版、包管理器和环境管理工具。Anaconda 因其用户友好性和能够轻松安装大量库和包而脱颖而出，否则这些库和包的安装可能会很痛苦，甚至根本不可能安装。

具体来说，Anaconda 允许用户通过“conda install”和“pip install”两种方式安装包，从而扩大了可用资源的范围。本附录将指导您为本书中的所有项目创建一个专门的 Python 虚拟环境。这种分割确保了本书中使用的库和包与在其他无关项目中使用的任何库保持隔离，从而消除了任何潜在干扰。

我们将使用 Jupyter Notebook 作为我们的集成开发环境（IDE）。我将指导您在您刚刚创建的 Python 虚拟环境中安装 Jupyter Notebook。最后，我将根据您的计算机是否配备了支持 CUDA 的 GPU，引导您安装 PyTorch、Torchvision 和 Torchaudio。

## A.1 安装 Python 和设置虚拟环境

在本节中，我将根据您的操作系统引导您在计算机上安装 Anaconda 的过程。之后，您将为本书中的所有项目创建一个 Python 虚拟环境。最后，您将安装 Jupyter Notebook 作为您的 IDE 来运行本书中的 Python 程序。

### A.1.1 安装 Anaconda

要通过 Anaconda 发行版安装 Python，请按照以下步骤操作。

首先，访问 [`www.anaconda.com/download/success`](https://www.anaconda.com/download/success) 并滚动到网页底部。找到并下载针对您特定操作系统（无论是 Windows、macOS 还是 Linux）的最新 Python 3 版本。

如果您使用的是 Windows，请从该链接下载最新的 Python 3 图形安装程序。点击安装程序并按照提供的说明进行安装。要确认 Anaconda 在您的计算机上已成功安装，您可以在计算机上搜索“Anaconda Navigator”应用程序。如果您可以启动该应用程序，则表示 Anaconda 已成功安装。

对于 macOS 用户，建议使用最新的 Python 3 图形安装程序，尽管也提供了命令行安装选项。运行安装程序并遵循提供的说明。通过在您的计算机上搜索“Anaconda Navigator”应用程序来验证 Anaconda 的成功安装。如果您可以启动该应用程序，则表示 Anaconda 已成功安装。

Linux 的安装过程比其他操作系统更复杂，因为没有图形安装程序。首先，确定最新的 Linux 版本。选择适当的 x86 或 Power8 和 Power9 包。点击下载最新的安装器 bash 脚本。默认情况下，安装器 bash 脚本通常保存在您的计算机的下载文件夹中。在终端中执行 bash 脚本来安装 Anaconda。安装完成后，通过运行以下命令来激活它：

```py
source ~/.bashrc
```

要访问 Anaconda Navigator，请在终端中输入以下命令：

```py
anaconda-navigator
```

如果您能成功在 Linux 系统上启动 Anaconda Navigator，则您的 Anaconda 安装已完成。

练习 A.1

根据您的操作系统在您的计算机上安装 Anaconda。安装完成后，打开计算机上的 Anaconda Navigator 应用程序以确认安装。

### A.1.2 设置 Python 虚拟环境

非常推荐您为本书创建一个单独的虚拟环境。让我们将其命名为 *dgai*。在 Anaconda 命令提示符（Windows）或终端（Mac 和 Linux）中执行以下命令：

```py
conda create -n dgai
```

按下键盘上的 Enter 键后，按照屏幕上的说明操作，当提示 y/n 时按 y。要激活虚拟环境，请在相同的 Anaconda 命令提示符（Windows）或终端（Mac 和 Linux）中运行以下命令：

```py
conda activate dgai
```

虚拟环境将您为本书使用的 Python 包和库与其他用途的包和库隔离开来。这防止了任何不希望发生的干扰。

练习 A.2

在您的计算机上创建一个 Python 虚拟环境 *dgai*。安装完成后，激活该虚拟环境。

### A.1.3 安装 Jupyter Notebook

现在，让我们在您计算机上新建的虚拟环境中安装 Jupyter Notebook。

首先，在 Anaconda 命令提示符（Windows）或终端（Mac 或 Linux）中运行以下代码行以激活虚拟环境：

```py
conda activate dgai
```

在虚拟环境中安装 Jupyter Notebook，运行以下命令

```py
conda install notebook
```

按照屏幕上的说明完成安装。

要启动 Jupyter Notebook，请在终端中执行以下命令：

```py
jupyter notebook
```

Jupyter Notebook 应用程序将在您的默认浏览器中打开。

练习 A.3

在 Python 虚拟环境 *dgai* 中安装 Jupyter Notebook。安装完成后，在您的计算机上打开 Jupyter Notebook 应用程序以确认安装。

## A.2 安装 PyTorch

在本节中，我将根据您的计算机上是否有启用 CUDA 的 GPU 来指导您安装 PyTorch。官方 PyTorch 网站 [`pytorch.org/get-started/locally/`](https://pytorch.org/get-started/locally/) 提供了有关带有或没有 CUDA 的 PyTorch 安装的更新。我鼓励您查看网站以获取任何更新。

CUDA 仅在 Windows 或 Linux 上可用，不在 Mac 上。要找出你的电脑是否配备了 CUDA 支持的 GPU，打开 Windows PowerShell（在 Windows 中）或终端（在 Linux 中）并输入以下命令：

```py
nvidia-smi
```

如果你的电脑有 CUDA 支持的 GPU，你应该会看到一个类似于图 A.1 的输出。此外，注意图右上角所示的 CUDA 版本，因为你在安装 PyTorch 时需要这个信息。图 A.1 显示我的电脑上的 CUDA 版本是 11.8。你的电脑上的版本可能不同。

![](img/APPA_F01_Liu.png)

图 A.1 检查你的电脑是否具有 CUDA 支持的 GPU

如果你运行`nvidia-smi`命令后看到错误信息，说明你的电脑没有 CUDA 支持的 GPU。

在第一小节中，我将讨论如果你电脑上没有 CUDA 支持的 GPU，如何安装 PyTorch。你可以使用 CPU 来训练这本书中所有的生成式 AI 模型。这只需要更长的时间。然而，我会提供预训练的模型，这样你就可以见证生成式 AI 的实际应用。

另一方面，如果你使用 Windows 或 Linux 操作系统，并且电脑上确实有 CUDA 支持的 GPU，我将在下一小节中指导你安装带有 CUDA 的 PyTorch。

### A.2.1 不使用 CUDA 安装 PyTorch

要使用 CPU 训练安装 PyTorch，首先在 Anaconda 提示符（在 Windows 中）或终端（在 Mac 或 Linux 中）中运行以下代码行来激活虚拟环境*dgai*：

```py
conda activate dgai
```

你应该能在提示符前看到*(dgai)*，这表明你现在处于*dgai*虚拟环境。要安装 PyTorch，输入以下命令行：

```py
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

按照屏幕上的说明完成安装。在这里，我们一次性安装了三个库：PyTorch、Torchaudio 和 Torchvision。Torchaudio 是一个用于处理音频和信号的库，我们需要它来生成这本书中的音乐。我们还将广泛使用 Torchvision 库来处理图像。

如果你的 Mac 电脑配备了 Apple 硅或 AMD GPU，并且运行 macOS 12.3 或更高版本，你可以使用新的 Metal Performance Shaders 后端来加速 GPU 训练。更多信息请参阅[`developer.apple.com/metal/pytorch/`](https://developer.apple.com/metal/pytorch/)和[`pytorch.org/get-started/locally/`](https://pytorch.org/get-started/locally/)。

要检查三个库是否在你的电脑上成功安装，运行以下代码行：

```py
import torch, torchvision, torchaudio

print(torch.__version__)
print(torchvision.__version__)
print(torchaudio.__version__)
```

我电脑上的输出显示

```py
2.0.1
0.15.2
2.0.2
```

如果你没有看到错误信息，说明你已经在电脑上成功安装了 PyTorch。

### A.2.2 使用 CUDA 安装 PyTorch

要使用 CUDA 安装 PyTorch，首先找出你 GPU 的 CUDA 版本，如图 A.1 右上角所示。我的 CUDA 版本是 11.8，因此我将用它作为安装示例。

如果你访问 PyTorch 网站[`pytorch.org/get-started/locally/`](https://pytorch.org/get-started/locally/)，你会看到一个如图 A.2 所示的交互式界面。

一旦进入，选择你的操作系统，选择 Conda 作为包，Python 作为语言，以及根据你在上一步中找到的信息，选择 CUDA 11.8 或 CUDA 12.1 作为你的计算机平台。如果你的计算机上的 CUDA 版本既不是 11.8 也不是 12.1，请选择最接近你版本的选项，它将可以工作。例如，如果一台计算机的 CUDA 版本为 12.4，而有人使用了 CUDA 12.1，安装将成功。

需要运行的命令将在底部面板中显示。例如，我使用的是 Windows 操作系统，我的 GPU 上安装了 CUDA 11.8。因此，我的命令显示在图 A.2 的底部面板中。

![](img/APPA_F02_Liu.png)

图 A.2 如何安装 PyTorch 的交互式界面

一旦你知道如何运行命令来安装带有 CUDA 的 PyTorch，通过在 Anaconda 提示符（Windows）或终端（Linux）中运行以下代码行来激活虚拟环境：

```py
conda activate dgai
```

然后执行你在上一步中找到的命令行。对我来说，命令行是

```py
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

按照屏幕上的说明完成安装。在这里，我们一次性安装了三个库：PyTorch、Torchaudio 和 Torchvision。Torchaudio 是一个用于处理音频和信号的库，我们需要它来生成这本书中的音乐。我们还在书中广泛使用了 Torchvision 库来处理图像。

为了确保你已正确安装 PyTorch，在 Jupyter Notebook 的新单元格中运行以下代码行：

```py
import torch, torchvision, torchaudio

print(torch.__version__)
print(torchvision.__version__)
print(torchaudio.__version__)
device=”cuda” if torch.cuda.is_available() else "cpu"
print(device)
```

我的计算机上的输出如下所示：

```py
2.0.1
0.15.2
2.0.2
cuda
```

输出的最后一行说`cuda`，表示我已经安装了带有 CUDA 的 PyTorch。如果你在计算机上安装了不带 CUDA 的 PyTorch，输出将是`cpu`。

练习 A.4

根据你的操作系统和计算机是否具有 GPU 训练加速，在你的计算机上安装 PyTorch、Torchvision 和 Torchaudio。安装完成后，打印出你刚刚安装的三个库的版本。
