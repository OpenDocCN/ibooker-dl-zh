# 附录 B 安装和配置 PyTorch Geometric

## B.1 安装 PyTorch Geometric

PyTorch Geometric (PyG) 是一个基于 PyTorch 构建的库，用于处理图神经网络（GNNs）。最新版本的 pytorch geometric 可以通过以下命令安装：pip install torch_geometric。只需要 PyTorch 作为依赖项。要安装带有其扩展的 PyG，您需要确保已安装并兼容的正确版本的 Compute Unified Device Architecture (CUDA)、PyTorch 和 PyG。

### B.1.1 在 Windows/Linux 上

如果您使用的是 Windows 或 Linux 系统，请按照以下步骤操作：

+   *安装 PyTorch*。首先，为您的系统安装适当的 PyTorch 版本。您可以在官方 PyTorch 网站上找到说明（[`pytorch.org/get-started/locally/`](https://pytorch.org/get-started/locally/))。如果您有 NVIDIA GPU，请确保选择正确的 CUDA 版本。

+   *查找 PyTorch CUDA 版本*。在安装 PyTorch 后，通过在 Python 中运行以下命令来检查其版本和构建时使用的 CUDA 版本：

```py
import torch
print(torch.__version__)
print(torch.version.cuda)
```

这也可以通过以下命令行运行：

```py
!python -c "import torch; print(torch.__version__)"
!python -c "import torch; print(torch.version.cuda)"
```

此代码的输出将在下一步中使用。

+   *安装 PyG 依赖项*。从 PyG 仓库安装 PyG 依赖项（`torch-scatter`、`torch-sparse`、`torch-cluster`、`torch-spline-conv`），指定正确的 CUDA 版本：

```py
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f
https://data.pyg.org/whl/torch-${PYTORCH}+${CUDA}.xhtml
```

在此代码中，将`${PYTORCH}`替换为您的 PyTorch 版本（例如，1.13.1），将`${CUDA}`替换为上一步中的 CUDA 版本（例如，cu117）。

+   *安装 PyG*。最后，安装 PyG 库本身：

```py
pip install torch-geometric
```

### B.1.2 在 MacOS 上

由于 Mac 没有配备 Nvidia GPU，您可以通过遵循上一节中的相同步骤安装 PyG 的`cpu`版本，但在安装依赖项时使用`cpu`而不是`CUDA`版本。

### B.1.3 兼容性问题

在安装扩展时，确保 CUDA、PyTorch 和 PyG 的版本匹配，以避免兼容性问题。使用不匹配的版本可能导致安装或运行时出错。始终参考官方文档以获取最新的安装说明和版本兼容性信息。在编写本书时，我们遇到了一些令人沮丧的错误，这些错误只有在安装正确的 CUDA、PyTorch 和 PyG 组合后才能解决。

从处理与 PyG 兼容的工具（如 Open Graph Benchmark (OGB)和 DistributedDataParallel (DDP)）的经历中，我们获得的一个特别见解是，它们可能只与 PyTorch 的特定版本兼容。在第七章中，分布式计算示例仅适用于 PyTorch v2.0.1 和 CUDA v11.8。
