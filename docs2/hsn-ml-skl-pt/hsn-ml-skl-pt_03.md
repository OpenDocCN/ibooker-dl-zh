# 第二章\. 全过程机器学习项目

在本章中，你将从头到尾完成一个示例项目，假装你是一家房地产公司新聘请的数据科学家。这个例子是虚构的；目的是说明机器学习项目的关键步骤，而不是学习有关房地产业务的知识。以下是我们将要经历的几个主要步骤：

1.  看看大局。

1.  获取数据。

1.  探索和可视化数据以获得洞察。

1.  为机器学习算法准备数据。

1.  选择一个模型并对其进行训练。

1.  微调你的模型。

1.  展示你的解决方案。

1.  启动、监控和维护你的系统。

# 处理真实数据

当你学习机器学习时，最好使用真实世界的数据进行实验，而不是人工数据集。幸运的是，有数千个公开数据集可供选择，涵盖各种领域。以下是一些你可以使用的流行公开数据仓库，以获取数据：

+   [谷歌数据集搜索](https://datasetsearch.research.google.com)

+   [Hugging Face 数据集](https://huggingface.co/docs/datasets)

+   [OpenML.org](https://openml.org)

+   [Kaggle.com](https://kaggle.com/datasets)

+   [加州大学欧文分校机器学习数据仓库](https://archive.ics.uci.edu)

+   [斯坦福大学大型网络数据集收藏](https://snap.stanford.edu/data)

+   [亚马逊的 AWS 数据集](https://registry.opendata.aws)

+   [美国政府的开放数据](https://data.gov)

+   [DataPortals.org](https://dataportals.org)

+   [维基百科的机器学习数据集列表](https://homl.info/9)

在本章中，我们将使用来自 StatLib 仓库的加利福尼亚房价数据集（见图 2-1）。这个数据集基于 1990 年加利福尼亚人口普查的数据。它并不完全是最新的（当时旧金山地区的一所好房子仍然负担得起），但它具有许多学习特性，因此我们将假装它是最近的数据。为了教学目的，我添加了一个分类属性并删除了一些特征。

![显示加利福尼亚房价数据，用彩色点表示中位数房屋价值，点的大小表示人口密度的加利福尼亚地图](img/hmls_0201.png)

###### 图 2-1\. 加利福尼亚房价

# 看看大局

欢迎来到机器学习住房公司！你的第一个任务是使用加利福尼亚人口普查数据来构建该州房价的模型。这些数据包括加利福尼亚每个街区组的指标，如人口、中位数收入和中位数房价。街区组是美国人口普查局发布样本数据的最小地理单位（街区组通常有 600 到 3,000 的人口）。我将简称为“地区”。

你的模型应该从这些数据中学习，并能够预测任何地区的平均房价，给定所有其他指标。

###### 小贴士

由于你是一个有组织的数据科学家，你应该做的第一件事是拿出你的机器学习项目清单。你可以从[*https://homl.info/checklist*](https://homl.info/checklist)上的清单开始；它应该对大多数机器学习项目来说都相当适用，但请确保根据你的需求进行调整。在本章中，我们将讨论许多清单项，但也会跳过一些，要么是因为它们是自我解释的，要么是因为它们将在后面的章节中讨论。

## 明确问题

你首先要问你的老板的是业务目标究竟是什么。构建一个模型可能不是最终目标。公司期望如何使用和从该模型中获益？了解目标很重要，因为它将决定你如何明确问题，你将选择哪些算法，你将使用哪些性能指标来评估你的模型，以及你将投入多少精力来调整它。

你的老板回答说，你的模型输出（对一个区域中位数房价的预测）将对于确定是否值得在该地区投资至关重要。更具体地说，你的模型输出将被输入到另一个机器学习系统中（见图 2-2），以及一些其他信号。⁠^(2) 因此，使我们的房价模型尽可能准确是非常重要的。

接下来你需要问你的老板当前解决方案（如果有的话）是什么样子。当前情况通常会为你提供性能的参考，以及如何解决问题的见解。你的老板回答说，区域房价目前是由专家手动估算的：一个团队收集一个区域最新的信息，当他们无法获得中位数房价时，他们会使用复杂的规则进行估算。

![展示房地产机器学习流程的图表，突出从区域数据到区域定价、投资分析和投资的流程](img/hmls_0202.png)

###### 图 2-2\. 房地产投资的机器学习流程

这既昂贵又耗时，他们的估算并不理想；在那些他们设法找到实际中位数房价的情况下，他们常常发现他们的估算误差超过 30%。这就是为什么公司认为，训练一个模型来预测一个区域的平均房价，给定该区域的其他数据，将会很有用。人口普查数据看起来是一个很好的数据集，可以用于此目的，因为它包括了数千个区域的中位数房价以及其他数据。

在获得所有这些信息后，您现在可以开始设计您的系统了。首先，确定模型需要的训练监督类型：是监督学习、无监督学习、半监督学习、自监督学习还是强化学习任务？它是分类任务、回归任务还是其他任务？您应该使用批处理学习技术还是在线学习技术？在继续阅读之前，请暂停并尝试自己回答这些问题。

您找到答案了吗？让我们看看。这显然是一个典型的监督学习任务，因为模型可以用*标记*的例子进行训练（每个实例都带有预期的输出，即该地区的平均房价）。它是一个典型的回归任务，因为模型将被要求预测一个值。更具体地说，这是一个*多元回归*问题，因为系统将使用多个特征进行预测（地区的总人口、平均收入等）。它也是一个*单变量回归*问题，因为我们只尝试预测每个地区的单个值。如果我们试图为每个地区预测多个值，那么它将是一个*多元回归*问题。最后，系统中没有连续的数据流进入，没有特别需要快速调整数据的需求，而且数据量足够小，可以放入内存，因此普通的批处理学习应该就足够了。

###### 小贴士

如果数据量巨大，您可以选择将批处理学习工作分散到多个服务器上（使用 MapReduce 技术）或使用在线学习技术。

## 选择性能指标

您的下一步是选择一个性能指标。对于回归问题，一个典型的性能指标是*均方根误差*（RMSE）。它给出了系统在预测中通常犯多少错误的概览，对大误差给予更高的权重。方程式 2-1 展示了计算 RMSE 的数学公式。

##### 方程式 2-1\. 均方根误差 (RMSE)

$RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (h(\boldsymbol{x}_i, \boldsymbol{y}, h) - y_i)²}$

虽然 RMSE 通常是回归任务的首选性能指标，但在某些情况下，您可能更喜欢使用另一个函数，尤其是在数据中有许多异常值时，因为 RMSE 对它们相当敏感。在这种情况下，您可以考虑使用*平均绝对误差*（MAE，也称为*平均绝对偏差*），如方程式 2-2 所示：

##### 方程式 2-2\. 平均绝对误差 (MAE)

$MAE left-parenthesis bold upper X comma bold y comma h right-parenthesis equals StartFraction 1 Over m EndFraction sigma-summation Underscript i equals 1 Overscript m Endscripts StartAbsoluteValue h left-parenthesis bold x Superscript left-parenthesis i right-parenthesis Baseline right-parenthesis minus y Superscript left-parenthesis i right-parenthesis Baseline EndAbsoluteValue$

RMSE 和 MAE 都是衡量两个向量之间距离的方法：预测向量和目标值向量。可能的距离度量，或称**范数**，有很多种：

+   计算平方和的根（RMSE）对应于**欧几里得范数**：这是我们所有人都熟悉的距离概念。它也称为ℓ[2] **范数**，表示为∥ · ∥[2]（或简称∥ · ∥）。

+   计算绝对值之和（MAE）对应于ℓ[1] **范数**，表示为∥ · ∥[1]。有时它被称为**曼哈顿范数**，因为它测量了在只能沿正交城市街区行走的条件下，两个点之间的距离。

+   更一般地，包含*n*个元素的向量**v**的ℓ[*k*] **范数**定义为 ∥**v**∥[*k*] = (|*v*[1]|^(*k*) + |*v*[2]|^(*k*) + ... + |*v*[*n*]|^(*k*))^(1/*k*). ℓ[0]给出向量中非零元素的数量，而ℓ[∞]给出向量中的最大绝对值。

范数指数越高，它就越关注大值而忽略小值。这就是为什么 RMSE 比 MAE 对异常值更敏感。但是，当异常值非常罕见（如钟形曲线中）时，RMSE 表现非常好，通常更受欢迎。

## 检查假设

最后，列出并验证到目前为止（由你或其他人）所做出的假设是一个好的实践；这可以帮助你及早发现严重问题。例如，你的系统输出的区域价格将被输入到下游机器学习系统中，你假设这些价格将被用作此类。但是，如果下游系统将价格转换为类别（例如，“便宜”、“中等”或“昂贵”），然后使用这些类别而不是价格本身，那会怎样？在这种情况下，价格完全正确并不重要；你的系统只需要正确分类。如果是这样，那么问题应该被界定为分类任务，而不是回归任务。你不想在为回归系统工作了数月之后才发现这一点。

幸运的是，在与负责下游系统的团队交谈后，你确信他们确实需要实际的价格，而不仅仅是类别。太好了！一切准备就绪，绿灯亮了，你现在可以开始编码了！

# 获取数据

是时候动手实践了。不要犹豫，拿起您的笔记本电脑，浏览代码示例。正如我在前言中提到的，本书中的所有代码示例都是开源的，并以 Jupyter 笔记本的形式在线提供[在线](https://github.com/ageron/handson-mlp)，这些是包含文本、图像和可执行代码片段（在我们的案例中是 Python）的交互式文档。在这本书中，我将假设您正在 Google Colab 上运行这些笔记本，这是一个免费服务，允许您直接在线运行任何 Jupyter 笔记本，而无需在您的机器上安装任何东西。如果您想使用另一个在线平台（例如 Kaggle）或如果您想在您的机器上本地安装所有内容，请参阅本书 GitHub 页面上的说明。

## 使用 Google Colab 运行代码示例

首先，打开一个网页浏览器并访问[*https://homl.info/colab-p*](https://homl.info/colab-p)：这将带您进入 Google Colab，并显示本书的 Jupyter 笔记本列表（见图 2-3）。您将找到每个章节的一个笔记本，以及一些额外的 NumPy、Matplotlib、Pandas、线性代数和微分计算的笔记本和教程。例如，如果您点击*02_end_to_end_machine_learning_project.ipynb*，来自第二章的笔记本将在 Google Colab 中打开（见图 2-4）。

Jupyter 笔记本由一系列单元格组成。每个单元格包含可执行代码或文本。尝试双击第一个文本单元格（包含句子“欢迎来到机器学习住房公司！”）。这将打开单元格以进行编辑。请注意，Jupyter 笔记本使用 Markdown 语法进行格式化（例如，`**粗体**`，`*斜体*`，`# 标题`，`url`等）。尝试修改此文本，然后按 Shift-Enter 查看结果。

![Google Colab 界面显示 GitHub 上“ageron/handson-mlp”存储库中的 Jupyter 笔记本列表，其中“02_end_to_end_machine_learning_project.ipynb”突出显示。](img/hmls_0203.png)

###### 图 2-3\. Google Colab 中的笔记本列表

![Google Colab 笔记本的截图，显示标题为“第二章 – 端到端机器学习项目”的部分，包含编辑和运行文本和代码单元格的说明。](img/hmls_0204.png)

###### 图 2-4\. 您在 Google Colab 中的笔记本

接下来，通过选择菜单中的“插入”→“代码单元格”来创建一个新的代码单元格。或者，您也可以点击工具栏上的+代码按钮，或者将鼠标悬停在单元格底部直到看到+代码和+文本出现，然后点击+代码。在新的代码单元格中，输入一些 Python 代码，例如`print("Hello World")`，然后按 Shift-Enter 运行此代码（或点击单元格左侧的▷按钮）。

如果你尚未登录您的 Google 账户，现在会被要求登录（如果您还没有 Google 账户，您需要创建一个）。一旦登录，当你尝试运行代码时，会看到一个安全警告，告诉你这个笔记本不是由 Google 编写的。恶意的人可能会创建一个试图诱骗你输入 Google 凭证的笔记本，以便他们可以访问你的个人数据，所以在运行笔记本之前，总是确保你信任其作者（或者运行之前双重检查每个代码单元将执行的操作）。假设你信任我（或者你计划检查每个代码单元），你现在可以点击“仍然运行”。

Colab 将为您分配一个新的*运行时间*：这是一个位于 Google 服务器上的免费虚拟机，其中包含许多工具和 Python 库，包括大多数章节所需的所有内容（在某些章节中，您需要运行命令来安装额外的库）。这需要几秒钟。接下来，Colab 将自动连接到这个运行时间并使用它来执行您的新代码单元。重要的是，代码是在运行时间上运行的，*而不是*在您的机器上。代码的输出将显示在单元格下方。恭喜你，你在 Colab 上运行了一些 Python 代码！

###### 小贴士

要插入一个新的代码单元，你也可以输入 Ctrl-M（或在 macOS 上输入 Cmd-M）然后按 A（在活动单元上方插入）或 B（在活动单元下方插入）。还有许多其他的快捷键可用：你可以通过输入 Ctrl-M（或在 macOS 上输入 Cmd-M）然后按 H 来查看和编辑它们。如果你选择在 Kaggle 或在自己的机器上使用 JupyterLab 或带有 Jupyter 扩展的 IDE（如 Visual Studio Code）运行笔记本，你会看到一些细微的差异——运行时间被称为*kernels*，用户界面和快捷键略有不同等——但从一种 Jupyter 环境切换到另一种并不太难。

## 保存您的代码更改和您的数据

你可以对 Colab 笔记本进行更改，并且只要你的浏览器标签页保持打开，这些更改就会持续存在。但一旦关闭，更改就会丢失。为了避免这种情况，请确保通过选择“文件”→“在驱动器中保存副本”将笔记本的副本保存到您的 Google Drive。或者，您可以通过选择“文件”→“下载”→“下载 .ipynb”将笔记本下载到您的计算机。然后您可以在稍后访问[*https://colab.research.google.com*](https://colab.research.google.com)并再次打开笔记本（无论是从 Google Drive 还是从您的计算机上传）。

###### 警告

Google Colab 仅适用于交互式使用：你可以在笔记本中随意操作并按需调整代码，但你不能让笔记本长时间无人看管，否则运行时间将被关闭，所有数据都将丢失。

如果笔记本生成了你关心的数据，确保在运行时间关闭之前下载这些数据。为此，点击文件图标（见图 2-5 中的步骤 1），找到你想要下载的文件，点击它旁边的垂直点（步骤 2），然后点击下载（步骤 3）。或者，你可以在运行时挂载你的 Google Drive，使笔记本能够直接将文件读写到 Google Drive，就像它是一个本地目录一样。为此，点击文件图标（步骤 1），然后点击 Google Drive 图标（图 2-5 中圈出的图标）并遵循屏幕上的说明。

![Google Colab 界面截图，显示下载文件或挂载 Google Drive 的步骤，图标和菜单选项已突出显示](img/hmls_0205.png)

###### 图 2-5\. 从 Google Colab 运行时下载文件（步骤 1 到 3），或挂载你的 Google Drive（圈出图标）

默认情况下，你的 Google Drive 将挂载在*/content/drive/MyDrive*。如果你想备份一个数据文件，只需运行`!cp [.keep-together]#/content/my_great_model /content/drive/MyDrive`将其复制到这个目录。# 任何以感叹号（`!`）开头的命令都被视为 shell 命令，而不是 Python 代码：`cp`是 Linux shell 命令，用于将文件从一个路径复制到另一个路径。请注意，Colab 运行时在 Linux 上运行（具体来说是 Ubuntu）。

## 交互式功能的强大与危险

Jupyter 笔记本是交互式的，这是一个非常好的特性：你可以逐个运行每个单元格，在任何地方停止，插入一个单元格，玩弄代码，返回并再次运行相同的单元格，等等，我强烈建议你这样做。如果你只是逐个运行单元格而不进行任何操作，你将不会学得很快。然而，这种灵活性是有代价的：很容易以错误的顺序运行单元格，或者忘记运行一个单元格。如果发生这种情况，后续的代码单元格很可能会失败。例如，每个笔记本中的第一个代码单元格包含设置代码（例如导入），所以请确保你首先运行它，否则将无法工作。

###### 小贴士

如果你遇到任何奇怪的错误，尝试重新启动运行时（通过从菜单中选择“运行时”→“重新启动运行时”），然后从笔记本的开始处再次运行所有单元格。这通常可以解决问题。如果不奏效，很可能是你做的某个更改破坏了笔记本：只需恢复到原始笔记本并再次尝试。如果仍然失败，请在 GitHub 上提交一个问题。

## 书中代码与笔记本代码的差异

你有时可能会注意到这本书中的代码和笔记本中的代码之间有一些小小的差异。这种情况可能由几个原因造成：

+   当你阅读这些内容时，一个库可能已经略有变化，或者也许尽管我尽了最大努力，但在书中我可能犯了一个错误。遗憾的是，我无法神奇地修复你这本书中的代码（除非你正在阅读电子版并且可以下载最新版本），但我可以修复笔记本。所以，如果你从这本书中复制代码后遇到错误，请查找笔记本中的修复代码：我将努力保持它们无错误并且与最新库版本保持更新。

+   笔记本包含一些额外的代码来美化图形（添加标签、设置字体大小等），并将它们以高分辨率保存到这本书中。如果你想忽略这些额外的代码，你可以安全地忽略它们。

我优化了代码的可读性和简洁性：我使其尽可能线性和平坦，定义了非常少的函数或类。目标是确保你运行的代码通常就在你面前，而不是嵌套在几层抽象之中，你需要搜索才能找到。这也使得你可以更容易地玩转代码。为了简单起见，错误处理有限，我将一些不太常见的导入放在了它们所需的位置（而不是像 PEP 8 Python 风格指南推荐的那样放在文件顶部）。话虽如此，你的生产代码不会有很大不同：只是稍微模块化一些，并增加了额外的测试和错误处理。

好的！一旦你熟悉了 Colab，你就可以下载数据了。

## 下载数据

在典型的环境中，你的数据将存储在关系数据库或其他常见的数据存储中，并分布在多个表/文档/文件中。要访问它，你首先需要获取你的凭证和访问授权，并熟悉数据模式。然而，在这个项目中，事情要简单得多：你只需下载一个单个的压缩文件*housing.tgz*，它包含一个名为*housing.csv*的逗号分隔值（CSV）文件，其中包含所有数据。

与手动下载和解压缩数据相比，通常最好是编写一个为你完成这些工作的函数。特别是如果数据经常变化，这很有用：你可以编写一个小脚本，使用该函数获取最新数据（或者你可以设置一个计划任务，定期自动执行该操作）。如果需要将数据集安装到多台机器上，自动化获取数据的过程也很有用。

这里是获取和加载数据的函数：

```py
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing_full = load_housing_data()
```

###### 注意

如果你遇到 SSL `CERTIFICATE_VERIFY_FAILED`错误，那么你很可能是需要安装`certifi`包，如[*https://homl.info/sslerror*](https://homl.info/sslerror)中所述。

当调用`load_housing_data()`时，它会寻找`datasets/housing.tgz`文件。如果找不到，它会在当前目录（在 Colab 中默认为`/content`）内创建`datasets`目录，从`ageron/data` GitHub 仓库下载`housing.tgz`文件，并将其内容提取到`datasets`目录中；这将在`datasets`目录下创建一个名为`housing`的目录，其中包含`housing.csv`文件。最后，该函数将这个 CSV 文件加载到一个包含所有数据的 Pandas DataFrame 对象中，并返回它。

###### 注意

如果你使用的是 Python 3.12 或 3.13，你应该在`extractall()`方法的参数中添加`filter='data'`：这限制了提取算法可以执行的操作，并提高了安全性（更多详情请参阅文档）。

## 快速查看数据结构

你可以通过 DataFrame 的`head()`方法查看数据的前五行（见图 2-6）。

![展示住房数据集前五行截图，包括经度、纬度、住房中位数年龄、中位数收入、海洋邻近度和中位数房价的截图](img/hmls_0206.png)

###### 图 2-6\. 数据集中的前五行

每一行代表一个区域。共有 10 个属性（它们并非都在截图显示中）：`longitude`（经度）、`latitude`（纬度）、`housing_median_age`（住房中位数年龄）、`total_rooms`（总房间数）、`total_bedrooms`（总卧室数）、`population`（人口）、`households`（家庭数）、`median_income`（中位数收入）、`median_house_value`（中位数房价）和`ocean_proximity`（海洋邻近度）。

`info()` 方法非常有用，可以快速获取数据的描述，特别是总行数、每个属性的类型以及非空值的数量：

```py
>>> housing_full.info() `<class 'pandas.core.frame.DataFrame'>`
`RangeIndex: 20640 entries, 0 to 20639`
`Data columns (total 10 columns):`
 `#   Column              Non-Null Count  Dtype`
`---  ------              --------------  -----`
 `0   longitude           20640 non-null  float64`
 `1   latitude            20640 non-null  float64`
 `2   housing_median_age  20640 non-null  float64`
 `3   total_rooms         20640 non-null  float64`
 `4   total_bedrooms      20433 non-null  float64`
 `5   population          20640 non-null  float64`
 `6   households          20640 non-null  float64`
 `7   median_income       20640 non-null  float64`
 `8   median_house_value  20640 non-null  float64`
 `9   ocean_proximity     20640 non-null  object`
`dtypes: float64(9), object(1)`
`memory usage: 1.6+ MB`
```

```py`` ###### Note    In this book, when a code example contains a mix of code and outputs, as is the case here, it is formatted like in the Python interpreter for better readability: the code lines are prefixed with `>>>` (or `...` for indented blocks), and the outputs have no prefix.    There are 20,640 instances in the dataset, which means that it is fairly small by machine learning standards, but it’s perfect to get started. You notice that the `total_bedrooms` attribute has only 20,433 non-null values, meaning that 207 districts are missing this feature. You will need to take care of this later.    All attributes are numerical, except for `ocean_proximity`. Its type is `object`, so it could hold any kind of Python object. But since you loaded this data from a CSV file, you know that it must be a text attribute. When you looked at the top five rows, you probably noticed that the values in the `ocean_proximity` column were repetitive, which means that it is probably a categorical attribute. You can find out what categories exist and how many districts belong to each category by using the `value_counts()` method:    ``` >>> housing_full["ocean_proximity"].value_counts() `ocean_proximity` `<1H OCEAN     9136` `INLAND        6551` `NEAR OCEAN    2658` `NEAR BAY      2290` `ISLAND           5` `Name: count, dtype: int64` ```py   ````Let’s look at the other fields. The `describe()` method shows a summary of the numerical attributes (Figure 2-7).  ![A table generated by the `describe()` method, displaying statistical summaries for numerical attributes such as count, mean, standard deviation, minimum, maximum, and percentiles.](img/hmls_0207.png)  ###### Figure 2-7\. Summary of each numerical attribute    The `count`, `mean`, `min`, and `max` rows are self-explanatory. Note that the null values are ignored (so, for example, the `count` of `total_bedrooms` is 20,433, not 20,640). The `std` row shows the *standard deviation*, which measures how dispersed the values are.⁠^(5) The `25%`, `50%`, and `75%` rows show the corresponding *percentiles*: a percentile indicates the value below which a given percentage of observations in a group of observations fall. For example, 25% of the districts have a `housing_median_age` lower than 18, while 50% are lower than 29, and 75% are lower than 37\. These are often called the 25th percentile (or first *quartile*), the median, and the 75th percentile (or third quartile).    Another quick way to get a feel of the type of data you are dealing with is to plot a histogram for each numerical attribute. A histogram shows the number of instances (on the vertical axis) that have a given value range (on the horizontal axis). You can either plot this one attribute at a time, or you can call the `hist()` method on the whole dataset (as shown in the following code example), and it will plot a histogram for each numerical attribute (see Figure 2-8).  ![Histograms displaying the distribution of various numerical attributes such as longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income, and median house value.](img/hmls_0208.png)  ###### Figure 2-8\. A histogram for each numerical attribute    The number of value ranges can be adjusted using the `bins` argument (try playing with it to see how it affects the histograms):    ```py import matplotlib.pyplot as plt  housing_full.hist(bins=50, figsize=(12, 8)) plt.show() ```    Looking at these histograms, you notice a few things:    *   First, the median income attribute does not look like it is expressed in US dollars (USD). After checking with the team that collected the data, you are told that the data has been scaled and capped at 15 (actually, 15.0001) for higher median incomes, and at 0.5 (actually, 0.4999) for lower median incomes. The numbers represent roughly tens of thousands of dollars (e.g., 3 actually means about $30,000). Working with preprocessed attributes is common in machine learning, and it is not necessarily a problem, but you should try to understand how the data was computed.           *   The housing median age and the median house value were also capped. The latter may be a serious problem since it is your target attribute (your labels). Your machine learning algorithms may learn that prices never go beyond that limit. You need to check with your client team (the team that will use your system’s output) to see if this is a problem or not. If they tell you that they need precise predictions even beyond $500,000, then you have two options:               *   Collect proper labels for the districts whose labels were capped.                       *   Remove those districts from the training set (and also from the test set, since your system should not be evaluated poorly if it predicts values beyond $500,000).                   *   These attributes have very different scales. We will discuss this later in this chapter when we explore feature scaling.           *   Finally, many histograms are *skewed right*: they extend much farther to the right of the median than to the left. This may make it a bit harder for some machine learning algorithms to detect patterns. Later, you’ll try transforming these attributes to have more symmetrical and bell-shaped distributions.              You should now have a better understanding of the kind of data you’re dealing with.```py` `````  ```py`````` ```py````` ## Create a Test Set    Before you look at the data any further, you need to create a test set, put it aside, and never look at it. It may seem strange to voluntarily set aside part of the data at this stage. After all, you have only taken a quick glance at the data, and surely you should learn a whole lot more about it before you decide what algorithms to use, right? This is true, but your brain is an amazing pattern detection system, which also means that it is highly prone to overfitting: if you look at the test set, you may stumble upon some seemingly interesting pattern in the test data that leads you to select a particular kind of machine learning model. When you estimate the generalization error using the test set, your estimate will be too optimistic, and you will launch a system that will not perform as well as expected. This is called *data snooping* bias.    Creating a test set is theoretically simple; pick some instances randomly, typically 20% of the dataset (or less if your dataset is very large), and set them aside:    ```py import numpy as np  def shuffle_and_split_data(data, test_ratio, rng):     shuffled_indices = rng.permutation(len(data))     test_set_size = int(len(data) * test_ratio)     test_indices = shuffled_indices[:test_set_size]     train_indices = shuffled_indices[test_set_size:]     return data.iloc[train_indices], data.iloc[test_indices] ```    You can then use this function like this:    ```py >>> rng = np.random.default_rng()  # default random number generator `>>>` `train_set``,` `test_set` `=` `shuffle_and_split_data``(``housing_full``,` `0.2``,` `rng``)` ```` `>>>` `len``(``train_set``)` ```py `16512` `>>>` `len``(``test_set``)` `` `4128` `` ``` ```py` ```   ```py```` ```py``` ````` Well, this works, but it is not
