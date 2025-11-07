# 附录 A. 准备工作

## A.1 Python

在本书中，所有代码都是用 Python 编写的。您可以从 Python 语言网站（[`www.python.org/downloads/`](https://www.python.org/downloads/））下载并安装适用于您操作系统的最新版本。本书使用的 Python 版本是 Python 3.7，但任何更高版本都应该同样适用。本书还使用了各种开源 Python 包来构建机器学习模型，并对它们进行解释和可视化。现在，让我们下载本书中使用的所有代码并安装所有相关 Python 包。

## A.2 Git 代码仓库

本书中的所有代码都可以从本书的网站（[https://www.manning.com/books/interpretable-ai](https://www.manning.com/books/interpretable-ai)）和 GitHub 上的 Git 仓库（[`github.com/thampiman/interpretable-ai-book`](https://github.com/thampiman/interpretable-ai-book)）下载。GitHub 上的仓库组织成文件夹，每个文件夹对应一个章节。如果您是 Git 和 GitHub 版本控制的新手，可以查看 GitHub 提供的材料（[`mng.bz/KBXg`](https://shortener.manning.com/KBXg)），以了解更多相关信息。您可以从命令行下载或克隆仓库，如下所示：

```
$> git clone https://github.com/thampiman/interpretable-ai-book.git
```

## A.3 Conda 环境

Conda 是一个开源系统，用于 Python 和其他语言的包、依赖和环境管理。您可以通过遵循 Conda 网站（[`mng.bz/9Keq`](https://shortener.manning.com/9Keq)）上的说明来在您的操作系统上安装 Conda。安装后，Conda 允许您轻松地查找和安装 Python 包，并将您的环境从一个机器导出并在另一个机器上重新创建。本书使用的 Python 包以 Conda 环境的形式导出，以便您可以在目标机器上轻松地重新创建它们。环境文件以 YAML 文件格式导出，可以在仓库中的`packages`文件夹中找到。然后，您可以从机器上下载的仓库目录中运行以下命令来创建 Conda 环境：

```
$> conda env create -f packages/environment.yml
```

此命令将安装本书所需的全部 Python 包，并创建一个名为`interpretable-ai`的 Conda 环境。如果您已经创建了环境并且想要更新它，可以运行以下命令：

```
$> conda env update -f packages/environment.yml
```

一旦创建或更新了环境，您应该通过运行以下命令来激活 Conda 环境：

```
$> conda activate interpretable-ai
```

## A.4 Jupyter 笔记本

本书中的代码结构为 Jupyter 笔记本。Jupyter 是一个开源的 Web 应用程序，用于轻松创建和运行实时 Python 代码、方程式、可视化和标记文本。Jupyter 笔记本在数据科学和机器学习社区中得到广泛应用。在下载源代码并安装所有相关 Python 包后，您现在可以准备好在 Jupyter 上运行本书中的代码。从您机器上下载的存储库目录中，您可以运行以下命令来启动 Jupyter Web 应用程序：

```
$> jupyter notebook
```

您可以通过浏览器在 http://<HOSTNAME>:8888 访问 Jupyter Web 应用程序。将<HOSTNAME>替换为您运行机器的主机名或 IP 地址。

## A.5 Docker

Conda 包/环境管理系统确实存在一些限制。它有时在多个操作系统、同一操作系统的不同版本或不同硬件上可能无法按预期工作。如果您在创建上一节中详细说明的 Conda 环境时遇到问题，您可以使用 Docker 作为替代。Docker 是一个用于打包软件依赖项的系统，确保每个人的环境都是相同的。您可以通过遵循 Docker 网站上的说明（[`www.docker.com/get-started`](https://www.docker.com/get-started)）来在您的操作系统上安装 Docker。安装完成后，您可以从命令行运行以下命令，从您机器上下载的存储库目录中构建 Docker 镜像：

```
$> docker build . -t interpretable-ai
```

注意，`interpretable-ai`标签用于 Docker 镜像。如果此命令运行成功，Docker 应打印构建的镜像标识符。您还可以通过运行以下命令查看构建的镜像的详细信息：

```
$> docker images
```

运行下一个命令来使用构建的镜像运行 Docker 容器并启动 Jupyter Web 应用程序：

```
$> docker run -p 8888:8888 interpretable-ai:latest
```

此命令应启动 Jupyter 笔记本应用程序，您应该可以通过从浏览器访问 http://<HOSTNAME>:8888 来运行本书中的所有代码。将<HOSTNAME>替换为您运行机器的主机名或 IP 地址。
