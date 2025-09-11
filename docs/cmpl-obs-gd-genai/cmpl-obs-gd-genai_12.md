# 附录 B：安装 Python

如果需要手动执行此操作，请参考在 Windows、macOS 和 Linux 机器上安装 Python 和 Pip 的指南。

## B.1 在 Windows 上安装 Python

要下载 Python 包，请转到官方 Python 网站。确保您下载的是最新版本（目前为止，是 Python 3.x）。根据您的系统架构（32 位或 64 位）选择适当的版本。大多数现代计算机都是 64 位，但您可以通过右键单击“This PC”（或“My Computer”）并选择“Properties”来确认。

下载安装程序后，运行它。勾选“Add Python x.x to PATH”的复选框。这将使得从命令提示符中更容易运行 Python 和 pip。然后点击“Install Now”。

您可以通过按下`Win + R`，键入`cmd`并按 Enter 来打开命令提示符来验证您的安装。要检查 Python 是否成功安装，请键入`python --version`并按 Enter。您应该看到显示的版本号。

pip 通常包含在 Python 的最新版本中。要检查 pip 是否已安装，请在命令提示符中键入`pip --version`并按 Enter。如果您看到版本信息，则已安装 pip；否则，您需要手动安装它。

要完成此操作，请从官方 Python 包管理机构网站下载“get-pip.py”脚本，并将脚本保存到计算机上的某个位置。打开命令提示符，并使用`cd`命令导航到您保存“get-pip.py”的目录。例如：

```py
cd C:\Users\YourUsername\Downloads
```

然后运行此命令：

```py
python get-pip.py
```

要验证您的 pip 安装，请在命令提示符中运行`pip --version`。

使用安装了 Python 和 pip 的环境，可以使用命令`pip install package-name`来安装包。

## B.2 在 macOS 上安装 Python

macOS 通常预装了 Python 的版本。要检查 Python 是否已安装，请打开终端（您可以在应用程序 > 实用工具文件夹中找到它），然后键入：

```py
python3 --version
```

如果您看到版本号，则已安装 Python。如果没有，请按照下面的步骤安装。

Homebrew 是 macOS 的一个流行的软件包管理器，它使安装软件更加容易。如果您尚未安装 Homebrew，可以使用以下命令在终端中安装它：

```py
/bin/bash -c "$(curl -fsSL \
  https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
Install Python 3:
```

如果您使用 Homebrew，可以通过在终端中运行以下命令来安装 Python 3：

```py
brew install python
```

如果您不使用 Homebrew，可以从 Python 网站下载官方 Python 安装程序。

以下是您需要执行的步骤：

+   下载最新版本的 Python 3.x

+   运行您下载的安装程序包，并按照安装说明操作

+   验证安装

安装完成后，您应该能够通过在终端中输入 python3 来访问 Python，并使用以下命令验证已安装的*版本*

```py
python3 --version.
```

Python 3 通常预装了 pip。要验证 pip 是否已安装，请运行：

```py
pip3 --version
```

如果您看到版本信息，则已准备就绪。如果没有，请手动安装 pip：

+   下载“get-pip.py”脚本。

+   打开终端并使用 cd 命令导航到保存了“get-pip.py”的目录。例如：cd ~/Downloads

+   运行以下命令：`sudo python3 get-pip.py`

安装了 pip 后，您可以在终端中运行`pip3 --version`来检查其版本。

安装了 Python 和 pip 后，您可以开始使用 Python 并从 PyPI 安装软件包。要安装软件包，请使用命令`pip3 install package-name`。

请记住，您可能需要在终端中使用`python3`和`pip3`（而不是`python`和`pip`）来确保您正在使用 Python 3 及其相关的 pip。

## B.3 在 Linux 上安装 pip Python 包管理器

请注意，某些 Linux 发行版预先安装了 Python，因此最好在安装新版本之前先检查一下。

要做到这一点，请打开终端并键入：

```py
python3 --version
```

如果您看到一个版本号，则已安装了 Python。如果没有，请按照以下步骤安装它。

在安装软件之前，最好先更新您的包管理器。对于使用 apt 的系统（Debian/Ubuntu），请使用：

```py
sudo apt update
```

对于使用 dnf 的系统（Fedora），请使用：

```py
sudo dnf update
```

要安装 Python 3，请使用包管理器。这意味着：`sudo apt install python3`或`sudo dnf install python3` - 取决于您的系统。包名称可能会根据您的发行版略有不同。

安装完成后，您应该能够通过在终端中输入 python3 来访问 Python 3，使用：

```py
python3 --version.
```

Python 3 通常预先安装了 pip。要验证是否安装了 pip，请运行：

```py
pip3 --version
```

如果您看到版本信息，那就万事俱备了。如果没有，则可以使用`sudo apt install python3-pip`或`sudo dnf install python3-pip`手动安装 pip。同样，这些命令中的`3`部分在某些系统上可能是默认设置，因此您可能需要省略`3`部分。

安装了 pip 后，您可以通过在终端中运行`pip3 --version`来检查其版本。安装了 Python 和 pip 后，您可以开始使用 Python 并从 PyPI 安装软件包。要安装软件包，请使用`command pip3 install package-name`。
