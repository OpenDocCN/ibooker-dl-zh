## 附录 A：安装 tfjs-node-gpu 及其依赖项

要在 Node.js 中使用 GPU 加速版的 TensorFlow.js（tfjs-node-gpu），你需要在你的计算机上安装 CUDA 和 CuDNN。首先，计算机应配备有支持 CUDA 的 NVIDIA GPU。要检查你的计算机中的 GPU 是否满足该要求，请访问 [`developer.nvidia.com/cuda-gpus`](https://developer.nvidia.com/cuda-gpus)。

接下来，我们列出了 Linux 和 Windows 上的驱动程序和库安装的详细步骤，因为这两个操作系统是目前支持 tfjs-node-gpu 的两个操作系统。

## A.1\. 在 Linux 上安装 tfjs-node-gpu

1.  我们假设你已经在系统上安装了 Node.js 和 npm，并且 node 和 npm 的路径已包含在你的系统路径中。如果没有，请查看 [`nodejs.org/en/download/`](https://nodejs.org/en/download/) 获取可下载的安装程序。

1.  从 [`developer.nvidia.com/cuda-downloads`](https://developer.nvidia.com/cuda-downloads) 下载 CUDA Toolkit。务必选择适合你打算使用的 tfjs-node-gpu 版本的适当版本。在撰写本文时，tfjs-node-gpu 的最新版本为 1.2.10，与 CUDA Toolkit 版本 10.0 兼容。此外，请确保选择正确的操作系统（Linux）、架构（例如，用于主流 Intel CPU 的 x86_64）、Linux 发行版和发行版的版本。你将有下载几种类型安装程序的选项。在这里，我们假设你下载了“runfile（local）”文件（而不是，例如，本地 .deb 包）以供后续步骤使用。

1.  在你的下载文件夹中，使刚下载的 runfile 可执行。例如，

    ```js
    chmod +x cuda_10.0.130_410.48_linux.run
    ```

1.  使用`sudo`来运行 runfile。注意，CUDA Toolkit 安装过程可能需要安装或升级 NVIDIA 驱动程序，如果您机器上已安装的 NVIDIA 驱动程序版本过旧或尚未安装此类驱动程序。如果是这种情况，你需要停止 X 服务器，转到仅 shell 模式。在 Ubuntu 和 Debian 发行版中，你可以使用快捷键 Ctrl-Alt-F1 进入仅 shell 模式。按照屏幕上的提示安装 CUDA Toolkit，然后重新启动机器。如果你在仅 shell 模式下，你可以重新启动回到正常的 GUI 模式。

1.  如果步骤 3 完成正确，`nvidia-smi`命令现在应该可在你的路径中使用了。你可以使用它来检查 GPU 的状态。它提供了有关安装在你的机器上的 NVIDIA GPU 的名称、温度传感器读数、风扇速度、处理器和内存使用情况，以及当前 NVIDIA 驱动程序版本的信息。当你使用 tfjs-node-gpu 训练深度神经网络时，它是一个方便的实时监视 GPU 的工具。`nvidia-smi` 的典型输出信息如下（请注意，此机器上有两个 NVIDIA GPU）：

    ```js
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 384.111                Driver Version: 384.111                   |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Quadro P1000        Off  | 00000000:65:00.0  On |                  N/A |
    | 41%   53C    P0   ERR! /  N/A |    620MiB /  4035MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Quadro M4000        Off  | 00000000:B3:00.0 Off |                  N/A |
    | 46%   30C    P8    11W / 120W |      2MiB /  8121MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      3876      G   /usr/lib/xorg/Xorg                           283MiB |
    +-----------------------------------------------------------------------------+
    ```

1.  将 64 位 CUDA 库文件的路径添加到你的`LD_LIBRARY_PATH`环境变量中。假设你正在使用 bash shell，你可以将以下行添加到你的 .bashrc 文件中：

    ```js
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${PATH}"
    ```

    tfjs-node-gpu 在启动时使用`LD_LIBRARY_PATH`环境变量来找到所需的动态库文件。

1.  从[`developer.nvidia.com/cudnn`](https://developer.nvidia.com/cudnn)下载 CuDNN。为什么除了 CUDA 还需要 CuDNN 呢？这是因为 CUDA 是一个通用的计算库，除了深度学习之外还可以应用在其他领域（例如流体力学）。CuDNN 是 NVIDIA 基于 CUDA 构建的加速深度神经网络操作的库。NVIDIA 可能要求你创建一个登录账号并回答一些调查问题才能下载 CuDNN。一定要下载与之前步骤安装的 CUDA Toolkit 版本相匹配的 CuDNN 版本。例如，CuDNN 7.6 与 CUDA Toolkit 10.0 一起使用。

1.  与 CUDA Toolkit 不同，下载的 CuDNN 没有可执行安装程序。相反，它是一个压缩的 tarball，其中包含了一些动态库文件和 C/C++头文件。这些文件应该被提取并复制到适当的目标文件夹中。你可以使用如下的一系列命令来实现这一点：

    ```js
    tar xzvf cudnn-10.0-linux-x64-v7.6.4.38.tgz
    cp cuda/lib64/* /usr/local/cuda/lib64
    cp cuda/include/* /usr/local/cuda/include
    ```

1.  现在，所有必需的驱动程序和库都已安装完成，你可以通过在 node 中导入 tfjs-node-gpu 来快速验证 CUDA 和 CuDNN：

    ```js
    npm i @tensorflow/tfjs @tensorflow/tfjs-node-gpu
    node
    ```

    然后，在 Node.js 命令行界面上，

    ```js
    > const tf = require('@tensorflow/tfjs');
    > require('@tensorflow/tfjs-node-gpu');
    ```

    如果一切顺利，你应该会看到一系列日志行，确认发现了一个（或多个，取决于你的系统配置）可以被 tfjs-node-gpu 使用的 GPU：

    ```js
    2018-09-04 13:08:17.602543: I
    tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0
    with properties:
     name: Quadro M4000 major: 5 minor: 2 memoryClockRate(GHz): 0.7725
     pciBusID: 0000:b3:00.0
     totalMemory: 7.93GiB freeMemory: 7.86GiB
     2018-09-04 13:08:17.602571: I
    tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible
    gpu devices: 0
     2018-09-04 13:08:18.157029: I
    tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device
    interconnect StreamExecutor with strength 1 edge matrix:
     2018-09-04 13:08:18.157054: I
    tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0
     2018-09-04 13:08:18.157061: I
    tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N
     2018-09-04 13:08:18.157213: I
    tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created
    TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with
    7584 MB memory) -> physical GPU (device: 0, name: Quadro M4000, pci bus
    id: 0000:b3:00.0, compute capability: 5.2)
    ```

1.  现在，你已经准备好使用 tfjs-node-gpu 的所有功能了。只需确保在你的 package.json 中包含以下依赖项（或其后续版本）：

    ```js
      ...
      "dependencies": {
        "@tensorflow/tfjs": "⁰.12.6",
        "@tensorflow/tfjs-node": "⁰.1.14",
        ...
      }
      ...
    ```

    在你的主要的 .js 文件中，确保导入基本的依赖项，包括`@tensorflow/tfjs`和`@tensorflow/tfjs-node-gpu`。前者给你提供了 TensorFlow.js 的通用 API，而后者将 TensorFlow.js 操作与基于 CUDA 和 CuDNN 实现的高性能计算内核相连：

    ```js
    const tf = require('@tensorflow/tfjs');
    require('@tensorflow/tfjs-node-gpu');
    ```

## A.2\. 在 Windows 上安装 tfjs-node-gpu

1.  确保您的 Windows 符合 CUDA Toolkit 的系统要求。某些 Windows 版本和 32 位机器架构不受 CUDA Toolkit 的支持。有关更多详情，请参阅 [`docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements)。

1.  我们假设您已在系统上安装了 Node.js 和 npm，并且 Node.js 和 npm 的路径在您系统的环境变量 `Path` 中可用。如果没有，请访问 [`nodejs.org/en/download/`](https://nodejs.org/en/download/) 下载安装程序。

1.  安装 Microsoft Visual Studio，因为它是安装 CUDA Toolkit 所必需的。请参阅步骤 1 中相同的链接，以了解应安装哪个版本的 Visual Studio。

1.  下载并安装 Windows 版 CUDA Toolkit。在撰写本文时，运行 tfjs-node-gpu 需要 CUDA 10.0（最新版本：1.2.10）。请务必为您的 Windows 发行版选择正确的安装程序。支持 Windows 7 和 Windows 10 的安装程序。此步骤需要管理员权限。

1.  下载 CuDNN。确保 CuDNN 的版本与 CUDA 的版本匹配。例如，CuDNN 7.6 与 CUDA Toolkit 10.0 匹配。在下载 CuDNN 之前，NVIDIA 可能要求您在其网站上创建帐户并回答一些调查问题。

1.  与 CUDA Toolkit 安装程序不同，您刚下载的 CuDNN 是一个压缩文件。解压它，您将看到其中有三个文件夹：cuda/bin、cuda/include 和 cuda/lib/x64。找到 CUDA Toolkit 安装的目录（默认情况下，它类似于 C:/Program Files/NVIDIA CUDA Toolkit 10.0/cuda）。将解压后的文件复制到那里相应名称的子文件夹中。例如，解压的 zip 存档中的 cuda/bin 中的文件应复制到 C:/Program Files/NVIDIA CUDA Toolkit 10.0/cuda/bin。此步骤可能还需要管理员权限。

1.  安装 CUDA Toolkit 和 CuDNN 后，请重新启动 Windows 系统。我们发现这对于所有新安装的库都能正确加载以供 tfjs-node-gpu 使用是必要的。

1.  安装 npm 包 `window-build-tools`。这是下一步安装 npm 包 `@tensorflow/tfjs-node-gpu` 所必需的：

    ```js
    npm install --add-python-to-path='true' --global windows-build-tools
    ```

1.  使用 npm 安装包 `@tensorflow/tfjs` 和 `@tensorflow/tfjs-node-gpu`：

    ```js
    npm -i @tensorflow/tfjs @tensorflow/tfjs-node-gpu
    ```

1.  要验证安装是否成功，请打开节点命令行并运行

    ```js
    > const tf = require('@tensorflow/tfjs');
    > require('@tensorflow/tfjs-node-gpu');
    ```

    确保这两个命令都能顺利完成。在第二个命令之后，您应该在控制台中看到一些由 TensorFlow GPU 共享库打印的日志行。这些行将列出 tfjs-node-gpu 已识别并将在后续深度学习程序中使用的 CUDA 启用的 GPU 的详细信息。
