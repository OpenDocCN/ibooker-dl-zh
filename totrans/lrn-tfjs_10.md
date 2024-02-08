# 第九章。分类模型和数据分析

> “先见之明，后事之师。”
> 
> —Amelia Barr

你不仅仅是把数据丢进模型中是有原因的。神经网络以极快的速度运行并执行复杂的计算，就像人类可以瞬间做出反应一样。然而，对于人类和机器学习模型来说，反应很少包含合理的上下文。处理脏乱和令人困惑的数据会导致次优的模型，甚至什么都不会得到。在这一章中，你将探索识别、加载、清理和优化数据的过程，以提高 TensorFlow.js 模型的训练准确性。

我们将：

+   确定如何创建分类模型

+   学习如何处理 CSV 数据

+   了解 Danfo.js 和 DataFrames

+   确定如何将混乱的数据输入训练中（整理你的数据）

+   练习绘制和分析数据

+   了解机器学习笔记本

+   揭示特征工程的核心概念

当你完成这一章时，你将有信心收集大量数据，分析数据，并通过使用上下文来创建有助于模型训练的特征来测试你的直觉。

在这一章中，你将构建一个*Titanic*生死分类器。30 岁的 Kate Connolly 小姐，持有三等舱票，会生还吗？让我们训练一个模型来获取这些信息，并给出生还的可能性。

# 分类模型

到目前为止，你训练了一个输出数字的模型。你消化的大多数模型与你创建的模型有些不同。在第八章中，你实现了线性回归，但在这一章中，你将实现一个分类模型（有时称为*逻辑回归*）。

毒性、MobileNet，甚至井字棋模型输出一种选择，从一组选项中选择。它们使用一组总和为一的数字，而不是一个没有范围的单个数字。这是分类模型的常见结构。一个旨在识别三种不同选项的模型将给出与每个选项对应的数字。

试图预测分类的模型需要一种将输出值映射到其相关类别的方法。到目前为止，在分类模型中，最常见的方法是输出它们的概率。要创建一个执行此操作的模型，你只需要在最终层实现特殊的激活函数：

###### 提示

记住，激活函数帮助你的神经网络以非线性方式运行。每个激活函数使得一个层以所需的非线性方式运行，最终层的激活直接转化为输出。重要的是确保你学会了哪种激活函数会给你所需的模型输出。

在这本书中使用的模型中，你一遍又一遍看到的激活函数被称为*softmax*激活。这是一组值，它们的总和为一。例如，如果你的模型需要一个 True/False 输出，你会期望模型输出两个值，一个用于识别`true`的概率，另一个用于`false`。例如，对于这个模型，softmax 可能输出`[0.66, 0.34]`，经过一些四舍五入。

这可以扩展到 N 个值的 N 个分类*只要类别是互斥的*。在设计模型时，你会在最终层强制使用 softmax，并且输出的数量将是你希望支持的类别数量。为了实现 True 或 False 的结果，你的模型架构将在最终层上使用 softmax 激活，有两个输出。

```py
// Final layer softmax True/False
model.add(
  tf.layers.dense({
    units: 2,
    activation: "softmax"
  })
);
```

如果您试图从输入中检测几件事情会发生什么？例如，胸部 X 光可能同时对肺炎和肺气肿呈阳性。在这种情况下，Softmax 不起作用，因为输出必须总和为一，对一个的信心必须与另一个对抗。在这种情况下，有一种激活可以强制每个节点的值在零和一之间，因此您可以实现每个节点的概率。这种激活称为 *sigmoid* 激活。这可以扩展到 N 个值，用于 N 个不相互排斥的分类。这意味着您可以通过具有 `sigmoid` 的单个输出来实现真/假模型（二元分类），其中接近零为假，接近一为真：

```py
// Final layer sigmoid True/False
model.add(
  tf.layers.dense({
    units: 1,
    activation: "sigmoid",
  })
);
```

是的，这些激活名称很奇怪，但它们并不复杂。您可以通过研究这些激活函数的工作原理背后的数学，在 YouTube 的兔子洞中轻松度过一天。但最重要的是，了解它们在分类中的用法。在 表 9-1 中，您将看到一些示例。

表 9-1\. 二元分类示例

| 激活 | 输出 | 结果分析 |
| --- | --- | --- |
| sigmoid | `[0.999999]` | 99% 确定是真的 |
| softmax | `[0.99, 0.01]` | 99% 确定是真的 |
| sigmoid | `[0.100000]` | 10% 确定是真的（因此 90% 是假的） |
| softmax | `[0.10, 0.90]` | 90% 确定是假的 |

当您处理真/假时，您使用 `softmax` 与 `sigmoid` 的区别消失。在选择最终层的激活时，您选择哪种激活没有真正的区别，因为没有一种可以排除另一种。在本章中，我们将在最后一层使用 sigmoid 以简化。

如果您试图对多个事物进行分类，您需要在 `sigmoid` 或 `softmax` 之间做出明智的选择。本书将重申和澄清这些激活函数的使用情况。

# 泰坦尼克号

1912 年 4 月 15 日，“不沉的” RMS *泰坦尼克号*（请参见 图 9-1）沉没了。这场悲剧在历史书籍中广为流传，充满了傲慢的故事，甚至有一部由莱昂纳多·迪卡普里奥和凯特·温丝莱特主演的电影。这场悲剧充满了一丝令人敬畏的死亡好奇。如果您在拉斯维加斯卢克索的 *泰坦尼克号* 展览中，您的门票会分配给您一位乘客的名字，并告诉您您的票价、舱位等等关于您生活的几件事。当您浏览船只和住宿时，您可以通过您门票上的人的眼睛体验它。在展览结束时，您会发现您门票上印刷的人是否幸存下来。

![泰坦尼克号概况](img/ltjs_0901.png)

###### 图 9-1\. RMS 泰坦尼克号

谁生存谁没有是 100%随机的吗？熟悉历史或看过电影的人都知道这不是一个抛硬币的事情。也许您可以训练一个模型来发现数据中的模式。幸运的是，客人日志和幸存者名单可供我们使用。

## 泰坦尼克数据集

与大多数事物一样，数据现在已转录为数字格式。 *泰坦尼克* 名单以逗号分隔的值（CSV）形式可用。这种表格数据可以被任何电子表格软件读取。有很多副本的 *泰坦尼克* 数据集可用，并且它们通常具有相同的信息。我们将使用的 CSV 文件可以在本章的相关代码中的 [额外文件夹](https://oreil.ly/ry4Pf) 中找到。

这个 *泰坦尼克* 数据集包含在 表 9-2 中显示的列数据。

表 9-2\. 泰坦尼克数据

| 列 | 定义 | 图例 |
| --- | --- | --- |
| 生存 | 生存 | 0 = 否，1 = 是 |
| pclass | 票类 | 1 = 1 等，2 = 2 等，3 = 3 等 |
| 性别 | 性别 |  |
| 年龄 | 年龄 |  |
| 兄弟姐妹或配偶数量 | 兄弟姐妹或配偶在船上的数量 |  |
| 父母或子女数量 | 父母或子女在船上的数量 |  |
| 票号 | 票号 |  |
| 票价 | 乘客票价 |  |
| 船舱 | 船舱号码 |  |
| embarked | 登船港口 | C = 瑟堡, Q = 昆士敦, S = 南安普敦 |

那么如何将这些 CSV 数据转换为张量形式呢？一种方法是读取 CSV 文件，并将每个输入转换为张量表示进行训练。当您试图尝试哪些列和格式对训练模型最有用时，这听起来是一个相当重要的任务。

在 Python 社区中，一种流行的加载、修改和训练数据的方法是使用一个名为[Pandas](https://pandas.pydata.org)的库。这个开源库在数据分析中很常见。虽然这对 Python 开发人员非常有用，但 JavaScript 中存在类似工具的需求很大。

# Danfo.js

[Danfo.js](https://danfo.jsdata.org)是 Pandas 的 JavaScript 开源替代品。Danfo.js 的 API 被故意保持与 Pandas 接近，以便利用信息体验共享。甚至 Danfo.js 中的函数名称都是`snake_case`而不是标准的 JavaScript`camelCase`格式。这意味着您可以在 Danfo.js 中最小地进行翻译，利用 Pandas 的多年教程。

我们将使用 Danfo.js 来读取*Titanic* CSV 并将其修改为 TensorFlow.js 张量。要开始，您需要将 Danfo.js 添加到项目中。

要安装 Danfo.js 的 Node 版本，您将运行以下命令：

```py
$ npm i danfojs-node
```

如果您使用简单的 Node.js，则可以`require` Danfo.js，或者如果您已经配置了代码以使用 ES6+，则可以`import`：

```py
const dfd = require("danfojs-node");
```

###### 注意

Danfo.js 也可以在浏览器中运行。本章依赖于比平常更多的打印信息，因此利用完整的终端窗口并依赖 Node.js 的简单性来访问本地文件是有意义的。

Danfo.js 在幕后由 TensorFlow.js 提供支持，但它提供了常见的数据读取和处理实用程序。

## 为泰坦尼克号做准备

机器学习最常见的批评之一是它看起来像一个金鹅。您可能认为接下来的步骤是将模型连接到 CSV 文件，点击“训练”，然后休息一天，去公园散步。尽管每天都在努力改进机器学习的自动化，但数据很少以“准备就绪”的格式存在。

本章中的*Titanic*数据包含诱人的 Train 和 Test CSV 文件。然而，使用 Danfo.js，我们很快就会看到提供的数据远未准备好加载到张量中。本章的目标是让您识别这种形式的数据并做好适当的准备。

### 读取 CSV

CSV 文件被加载到一个称为 DataFrame 的结构中。DataFrame 类似于带有可能不同类型列和适合这些类型的行的电子表格，就像一系列对象。

DataFrame 有能力将其内容打印到控制台，以及许多其他辅助函数以编程方式查看和编辑内容。

让我们回顾一下以下代码，它将 CSV 文件读入 DataFrame，然后在控制台上打印几行：

```py
constdf=awaitdfd.read_csv("file://../../extra/titanic data/train.csv");①df.head().print();②
```

①

`read_csv`方法可以从 URL 或本地文件 URI 中读取。

②

DataFrame 可以限制为前五行，然后打印。

正在加载的 CSV 是训练数据，`print()`命令将 DataFrame 的内容记录到控制台。结果显示在控制台中，如图 9-2 所示。

![Head printout](img/ltjs_0902.png)

###### 图 9-2。打印 CSV DataFrame 头

在检查数据内容时，您可能会注意到一些奇怪的条目，特别是在`Cabin`列中，显示为`NaN`。这些代表数据集中的缺失数据。这是您不能直接将 CSV 连接到模型的原因之一：重要的是要确定如何处理缺失信息。我们将很快评估这个问题。

Danfo.js 和 Pandas 有许多有用的命令，可以帮助您熟悉加载的数据。一个流行的方法是调用`.describe()`，它试图分析每列的内容作为报告：

```py
// Print the describe data
df.describe().print();
```

如果打印 DataFrame 的`describe`数据，您将看到您加载的 CSV 有 891 个条目，以及它们的最大值、最小值、中位数等的打印输出，以便您验证信息。打印的表格看起来像图 9-3。

![描述打印输出](img/ltjs_0903.png)

###### 图 9-3。描述 DataFrame

一些列已从图 9-3 中删除，因为它们包含非数字数据。这是您将在 Danfo.js 中轻松解决的问题。

### 调查 CSV

这个 CSV 反映了数据的真实世界，其中经常会有缺失信息。在训练之前，您需要处理这个问题。

您可以使用`isna()`找到所有缺失字段，它将为每个缺失字段返回`true`或`false`。然后，您可以对这些值进行求和或计数以获得结果。以下是将报告数据集的空单元格或属性的代码：

```py
// Count of empty spots
empty_spots = df.isna().sum();
empty_spots.print();
// Find the average
empty_rate = empty_spots.div(df.isna().count());
empty_rate.print();
```

通过结果，您可以看到以下内容：

+   空的`Age`数值：177（20%）

+   空的`Cabin`数值：687（77%）

+   空的`Embarked`数值：2（0.002%）

从对缺失数据量的简短查看中，您可以看到您无法避免清理这些数据。解决缺失值问题将至关重要，删除像`PassengerId`这样的无用列，并最终对您想保留的非数字列进行编码。

为了不必重复操作，您可以将 CSV 文件合并、清理，然后创建两个准备好用于训练和测试的新 CSV 文件。

目前，这些是步骤：

1.  合并 CSV 文件。

1.  清理 DataFrame。

1.  从 DataFrame 重新创建 CSV 文件。

### 合并 CSV

要合并 CSV 文件，您将创建两个 DataFrame，然后沿着轴连接它们，就像对张量一样。您可能会感觉到张量训练引导您管理和清理数据的路径，并且这并非偶然。尽管术语可能略有不同，但您从前几章积累的概念和直觉将对您有所帮助。

```py
// Load the training CSV constdf=awaitdfd.read_csv("file://../../extra/titanic data/train.csv");console.log("Train Size",df.shape[0])①// Load the test CSV constdft=awaitdfd.read_csv("file://../../extra/titanic data/test.csv");console.log("Test Size",dft.shape[0])②constmega=dfd.concat({df_list:[df,dft],axis: 0})mega.describe().print()③
```

①

打印“训练集大小为 891”

②

打印“测试集大小为 418”

③

显示一个包含 1,309 的表

使用熟悉的语法，您已经加载了两个 CSV 文件，并将它们合并成一个名为`mega`的 DataFrame，现在您可以对其进行清理。

### 清理 CSV

在这里，您将处理空白并确定哪些数据实际上是有用的。您需要执行三个操作来正确准备用于训练的 CSV 数据：

1.  修剪特征。

1.  处理空白。

1.  迁移到数字。

修剪特征意味着删除对结果影响很小或没有影响的特征。为此，您可以尝试实验、绘制数据图表，或者简单地运用您的个人直觉。要修剪特征，您可以使用 DataFrame 的`.drop`函数。`.drop`函数可以从 DataFrame 中删除整个列或指定的行。

对于这个数据集，我们将删除对结果影响较小的列，例如乘客的姓名、ID、票和舱位。您可能会认为其中许多特征可能非常重要，您是对的。但是，我们将让您在本书之外的范围内研究这些特征。

```py
// Remove feature columns that seem less useful
const clean = mega.drop({
  columns: ["Name", "PassengerId", "Ticket", "Cabin"],
});
```

要处理空白，您可以填充或删除行。填充空行是一种称为*插补*的技术。虽然这是一个很好的技能可以深入研究，但它可能会变得复杂。在本章中，我们将采取简单的方法，仅删除任何具有缺失值的行。要删除任何具有空数据的行，我们可以使用`dropna()`函数。

###### 警告

这是在删除列之后*之后*完成的至关重要。否则，`Cabin`列中 77%的缺失数据将破坏数据集。

您可以使用以下代码删除所有空行：

```py
// Remove all rows that have empty spots
const onlyFull = clean.dropna();
console.log(`After mega-clean the row-count is now ${onlyFull.shape[0]}`);
```

此代码的结果将数据集从 1,309 行减少到 1,043 行。将其视为一种懒惰的实验。

最后，您剩下两列是字符串而不是数字（`Embarked`和`Sex`）。这些需要转换为数字。

`Embarked`的值，供参考，分别是：C = 瑟堡，Q = 昆士敦，S = 南安普敦。有几种方法可以对其进行编码。一种方法是用数字等价物对其进行编码。Danfo.js 有一个`LabelEncoder`，它可以读取整个列，然后将值转换为数字编码的等价物。`LabelEncoder`将标签编码为介于`0`和`n-1`之间的值。要对`Embarked`列进行编码，您可以使用以下代码：

```py
// Handle embarked characters - convert to numbers constencode=newdfd.LabelEncoder();①encode.fit(onlyFull["Embarked"]);②onlyFull["Embarked"]=encode.transform(onlyFull["Embarked"].values);③onlyFull.head().print();④
```

①

创建一个新的`LabelEncoder`实例。

②

适合对`Embarked`列的内容进行编码的实例。

③

将列转换为值，然后立即用生成的列覆盖当前列。

④

打印前五行以验证替换是否发生。

您可能会对像第 3 步那样覆盖 DataFrame 列的能力感到惊讶。这是处理 DataFrame 而不是张量的许多好处之一，尽管 TensorFlow.js 张量在幕后支持 Danfo.js。

现在您可以使用相同的技巧对`male` / `female`字符串进行编码。（请注意，出于模型目的和乘客名单中可用数据的考虑，我们将性别简化为二进制。）完成后，您的整个数据集现在是数字的。如果在 DataFrame 上调用`describe`，它将呈现所有列，而不仅仅是几列。

### 保存新的 CSV 文件

现在您已经创建了一个可用于训练的数据集，您需要返回两个 CSV 文件，这两个文件进行了友好的测试和训练拆分。

您可以使用 Danfo.js 的`.sample`重新拆分 DataFrame。`.sample`方法会从 DataFrame 中随机选择 N 行。从那里，您可以将剩余的未选择值创建为测试集。要删除已抽样的值，您可以按索引而不是整个列删除行。

DataFrame 对象具有`to_csv`转换器，可选择性地接受要写入的文件参数。`to_csv`命令会写入参数文件并返回一个 promise，该 promise 解析为 CSV 内容。重新拆分 DataFrame 并写入两个文件的整个代码可能如下所示：

```py
// 800 random to training
const newTrain = onlyFull.sample(800)
console.log(`newTrain row count: ${newTrain.shape[0]}`)
// The rest to testing (drop via row index)
const newTest = onlyFull.drop({index: newTrain.index, axis: 0})
console.log(`newTest row count: ${newTest.shape[0]}`)

// Write the CSV files
await newTrain.to_csv('../../extra/cleaned/newTrain.csv')
await newTest.to_csv('../../extra/cleaned/newTest.csv')
console.log('Files written!')
```

现在您有两个文件，一个包含 800 行，另一个包含 243 行用于测试。

## 泰坦尼克号数据的训练

在对数据进行训练之前，您需要处理最后一步，即经典的机器学习标记输入和预期输出（X 和 Y，分别）。这意味着您需要将答案（`Survived`列）与其他输入分开。为此，您可以使用`iloc`声明要创建新 DataFrame 的列的索引。

由于第一列是`Survived`列，您将使 X 跳过该列并抓取其余所有列。您将从 DataFrame 的索引一到末尾进行识别。这写作`1:`。您可以写`1:9`，这将抓取相同的集合，但`1:`表示“从索引零之后的所有内容”。`iloc`索引格式表示您为 DataFrame 子集选择的范围。

Y 值，或*答案*，是通过抓取`Survived`列来选择的。由于这是单列，无需使用`iloc`。*不要忘记对测试数据集执行相同操作*。

机器学习模型期望张量，而由于 Danfo.js 建立在 TensorFlow.js 上，将 DataFrame 转换为张量非常简单。最终，您可以通过访问`.tensor`属性将 DataFrame 转换为张量。

```py
// Get cleaned data
const df = await dfd.read_csv("file://../../extra/cleaned/newTrain.csv");
console.log("Train Size", df.shape[0]);
const dft = await dfd.read_csv("file://../../extra/cleaned/newTest.csv");
console.log("Test Size", dft.shape[0]);

// Split train into X/Y
const trainX = df.iloc({ columns: [`1:`] }).tensor;
const trainY = df["Survived"].tensor;

// Split test into X/Y
const testX = dft.iloc({ columns: [`1:`] }).tensor;
const testY = dft["Survived"].tensor;
```

这些值已准备好被馈送到一个用于训练的模型中。

我在这个问题上使用的模型经过很少的研究后是一个具有三个隐藏层和一个具有 Sigmoid 激活的输出张量的序列层模型。

模型的组成如下：

```py
model.add(tf.layers.dense({inputShape,units: 120,activation:"relu",①kernelInitializer:"heNormal",②}));model.add(tf.layers.dense({units: 64,activation:"relu"}));model.add(tf.layers.dense({units: 32,activation:"relu"}));model.add(tf.layers.dense({units: 1,activation:"sigmoid",③}));model.compile({optimizer:"adam",loss:"binaryCrossentropy",④metrics:["accuracy"],⑤});
```

①

每一层都使用 ReLU 激活，直到最后一层。

②

这一行告诉模型根据算法初始化权重，而不是简单地将模型的初始权重设置为完全随机。这有时可以帮助模型更接近答案。在这种情况下并不是关键，但这是 TensorFlow.js 的一个有用功能。

③

最后一层使用 Sigmoid 激活来打印一个介于零和一之间的数字（生存或未生存）。

④

在训练二元分类器时，最好使用一个与二元分类一起工作的花哨命名的函数来评估损失。

⑤

这显示了日志中的准确性，而不仅仅是损失。

当您将模型`fit`到数据时，您可以识别测试数据，并获得模型以前从未见过的数据的结果。这有助于防止过拟合：

```py
awaitmodel.fit(trainX,trainY,{batchSize: 32,epochs: 100,validationData:[testX,testY]①})
```

①

提供模型应该在每个 epoch 上验证的数据。

###### 注意

在前面的`fit`方法中显示的训练配置没有利用回调。如果您在`tfjs-node`上训练，您将自动看到训练结果打印到控制台。如果您使用`tfjs`，您需要添加一个`onEpochEnd`回调来打印训练和验证准确性。这两者的示例都在相关的[本章源代码](https://oreil.ly/39p7V)中提供。

在训练了 100 个 epoch 后，这个模型在训练数据上的准确率为 83%，在测试集的验证上也是 83%。从技术上讲，每次训练的结果会有所不同，但它们应该几乎相同：`acc=0.827 loss=0.404 val_acc=0.831 val_loss=0.406`。

该模型已经识别出一些模式，并击败了纯粹的机会（50%准确率）。很多人在这里停下来庆祝创造一个几乎没有努力就能工作 83%的模型。然而，这也是一个很好的机会来认识 Danfo.js 和特征工程的好处。

# 特征工程

如果你在互联网上浏览一下，80%是*Titanic*数据集的一个常见准确率分数。我们已经超过了这个分数，而且没有真正的努力。然而，仍然有改进模型的空间，这直接来源于改进数据。

抛出空白数据是一个好选择吗？存在可以更好强调的相关性吗？模式是否被正确组织为模型？您能预先处理和组织数据得越好，模型就越能找到和强调模式。许多机器学习的突破都来自于在将模式传递给神经网络之前简化模式的技术。

这是“只是倾倒数据”停滞不前的地方，特征工程开始发展。Danfo.js 让您通过分析模式和强调关键特征来提升您的特征。您可以在交互式的 Node.js 读取求值打印循环（REPL）中进行这项工作，或者甚至可以利用为评估和反馈循环构建的网页。

让我们尝试通过确定并向数据添加特征来提高上述模型的准确率至 83%以上，使用一个名为 Dnotebook 的 Danfo.js Notebook。

## Dnotebook

Danfo 笔记本，或[Dnotebook](https://dnotebook.jsdata.org)，是一个交互式网页，用于使用 Danfo.js 实验、原型设计和定制数据。Python 的等价物称为 Jupyter 笔记本。您可以通过这个笔记本实现的数据科学将极大地帮助您的模型。

我们将使用 Dnotebook 来创建和共享实时代码，以及利用内置的图表功能来查找*泰坦尼克号*数据集中的关键特征和相关性。

通过创建全局命令来安装 Dnotebook：

```py
$ npm install -g dnotebook
```

当您运行`$ dnotebook`时，将自动运行本地服务器并打开一个页面到本地笔记本站点，它看起来有点像图 9-4。

![Dnotebook 新鲜截图](img/ltjs_0904.png)

###### 图 9-4。正在运行的新鲜 Dnotebook

每个 Dnotebook 单元格可以是代码或文本。文本采用 Markdown 格式。代码可以打印输出，并且未使用`const`或`let`初始化的变量可以在单元格之间保留。请参见图 9-5 中的示例。

![Dnotebook 演示截图](img/ltjs_0905.png)

###### 图 9-5。使用 Dnotebook 单元格

图 9-5 中的笔记本可以从本章的[*extra/dnotebooks*](https://oreil.ly/pPvQu)文件夹中的*explaining_vars.json*文件中下载并加载。这使得它适合用于实验、保存和共享。

## 泰坦尼克号视觉

如果您可以在数据中找到相关性，您可以将其作为训练数据中的附加特征强调，并在理想情况下提高模型的准确性。使用 Dnotebook，您可以可视化数据并在途中添加评论。这是分析数据集的绝佳资源。我们将加载两个 CSV 文件并将它们组合，然后直接在笔记本中打印结果。

您可以创建自己的笔记本，或者可以从相关源代码加载显示的笔记本的 JSON。只要您能够跟上图 9-6 中显示的内容，任何方法都可以。

![指导性代码截图](img/ltjs_0906.png)

###### 图 9-6。加载 CSV 并在 Dnotebook 中组合它们

`load_csv`命令类似于`read_csv`命令，但在加载 CSV 内容时在网页上显示友好的加载动画。您可能还注意到了`table`命令的使用。`table`命令类似于 DataFrame 的`print()`，只是它为笔记本生成了输出的 HTML 表格，就像您在图 9-6 中看到的那样。

现在您已经有了数据，让我们寻找可以强调的重要区别，以供我们的模型使用。在电影《泰坦尼克号》中，当装载救生艇时他们大声喊着“妇女和儿童优先”。那真的发生了吗？一个想法是检查男性与女性的幸存率。您可以通过使用`groupby`来做到这一点。然后您可以打印每个组的平均值。

```py
grp = mega_df.groupby(['Sex'])
table(grp.col(['Survived']).mean())
```

而且*哇啊！*您可以看到 83%的女性幸存下来，而只有 14%的男性幸存下来，正如图 9-7 中所示。

![幸存率截图](img/ltjs_0907.png)

###### 图 9-7。女性更有可能幸存

您可能会想知道也许只是因为*泰坦尼克号*上有更多女性，这就解释了倾斜的结果，所以您可以快速使用`count()`来检查，而不是像刚才那样使用`mean()`：

```py
survival_count = grp.col(['Survived']).count()
table(survival_count)
```

通过打印的结果，您可以看到尽管幸存比例偏向女性，但幸存的男性要多得多。这意味着性别是幸存机会的一个很好的指标，因此应该强调这一特征。

使用 Dnotebook 的真正优势在于它利用了 Danfo.js 图表。例如，如果我们想看到幸存者的直方图，而不是分组用户，您可以查询所有幸存者，然后绘制结果。

要查询幸存者，您可以使用 DataFrame 的 query 方法：

```py
survivors = mega_df.query({column: "Survived", is: "==", to: 1 })
```

然后，要在 Dnotebooks 中打印图表，您可以使用内置的`viz`命令，该命令需要一个 ID 和回调函数，用于填充笔记本中生成的 DIV。

直方图可以使用以下方式创建：

```py
viz(`agehist`, x => survivors["Age"].plot(x).hist())
```

然后笔记本将显示生成的图表，如图 9-8 所示。

![存活直方图的屏幕截图](img/ltjs_0908.png)

###### 图 9-8\. 幸存者年龄直方图

在这里，您可以看到儿童的显着存活率高于老年人。再次，确定每个年龄组的数量和百分比可能值得，但似乎特定的年龄组或区间比其他年龄组表现更好。这给了我们可能改进模型的第二种方法。

让我们利用我们现在拥有的信息，再次尝试打破 83%准确率的记录。

## 创建特征（又称预处理）

在成长过程中，我被告知激活的神经元越多，记忆就会越强烈，因此请记住气味、颜色和事实。让我们看看神经网络是否也是如此。我们将乘客性别移动到两个输入，并创建一个经常称为*分桶*或*分箱*的年龄分组。

我们要做的第一件事是将性别从一列移动到两列。这通常称为*独热编码*。目前，`Sex`具有数字编码。乘客性别的独热编码版本将`0`转换为`[1, 0]`，将`1`转换为`[0, 1]`，成功地将值移动到两列/单元。转换后，您删除`Sex`列并插入两列，看起来像图 9-9。

![Danfo One-Hot Coded](img/ltjs_0909.png)

###### 图 9-9\. 描述性别独热编码

要进行独热编码，Danfo.js 和 Pandas 都有一个`get_dummies`方法，可以将一列转换为多个列，其中只有一个列的值为 1。在 TensorFlow.js 中，进行独热编码的方法称为`oneHot`，但在 Danfo.js 中，`get_dummies`是向二进制变量致敬的方法，统计学中通常称为*虚拟变量*。编码结果后，您可以使用`drop`和`addColumn`进行切换：

```py
// Handle person sex - convert to one-hot constsexOneHot=dfd.get_dummies(mega['Sex'])①sexOneHot.head().print()// Swap one column for two mega.drop({columns:['Sex'],axis: 1,inplace: true})②mega.addColumn({column:'male',value: sexOneHot['0']})③mega.addColumn({column:'female',value: sexOneHot['1']})
```

①

使用`get_dummies`对列进行编码

②

在`Sex`列上使用`inplace`删除

③

添加新列，将标题切换为男性/女性

接下来，您可以使用`apply`方法为年龄创建桶。`apply`方法允许您在整个列上运行条件代码。根据我们的需求，我们将定义一个在我们的图表中看到的重要年龄组的函数，如下所示：

```py
// Group children, young, and over 40yrs
function ageToBucket(x) {
  if (x < 10) {
    return 0
  } else if (x < 40) {
    return 1
  } else {
    return 2
  }
}
```

然后，您可以使用您定义的`ageToBucket`函数创建并添加一个完全新的列来存储这些桶：

```py
// Create Age buckets
ageBuckets = mega['Age'].apply(ageToBucket)
mega.addColumn({ column: 'Age_bucket', value: ageBuckets })
```

这添加了一个值范围从零到二的整列。

最后，我们可以将我们的数据归一化为介于零和一之间的数字。缩放值会使值之间的差异归一化，以便模型可以识别模式和缩放原始数字中扭曲的差异。

###### 注意

将归一化视为一种特征。如果您正在处理来自各个国家的 10 种不同货币，可能会感到困惑。归一化会缩放输入，使它们具有相对影响的大小。

```py
const scaler = new dfd.MinMaxScaler()
scaledData = scaler.fit(featuredData)
scaledData.head().print()
```

从这里，您可以为训练编写两个 CSV 文件并开始！另一个选项是您可以编写一个单独的 CSV 文件，而不是使用特定的 X 和 Y 值设置`validationData`，您可以设置一个名为`validationSplit`的属性，该属性将为验证数据拆分出一定比例的数据。这样可以节省我们一些时间和麻烦，所以让我们使用`validationSplit`来训练模型，而不是显式传递`validationData`。

生成的`fit`如下所示：

```py
await model.fit(trainX, trainY, {
  batchSize: 32,
  epochs: 100,
  // Keep random 20% for validation on the fly.
  // The 20% is selected at the beginning of the training session.
  validationSplit: 0.2,
})
```

模型使用新数据进行 100 个时代的训练，如果您使用`tfjs-node`，即使没有定义回调函数，也可以看到打印的结果。

## 特征工程训练结果

上次，模型准确率约为 83%。现在，使用相同的模型结构但添加了一些特征，我们达到了 87%的训练准确率和 87%的验证准确率。具体来说，我的结果是`acc=0.867 loss=0.304 val_acc=0.871 val_loss=0.370`。

准确性提高了，损失值低于以前。真正了不起的是，准确性和验证准确性都是对齐的，因此模型不太可能过拟合。这通常是神经网络在泰坦尼克号数据集中的较好得分之一。对于这样一个奇怪的问题，创建一个相当准确的模型已经达到了解释如何从数据中提取有用信息的目的。

## 审查结果

解决泰坦尼克号问题以达到 87%的准确率需要一些技巧。您可能仍然在想结果是否可以改进，答案肯定是“是”，因为其他人已经在排行榜上发布了更令人印象深刻的分数。在没有排行榜的情况下，评估是否有增长空间的常见方法是与一个受过教育的人在面对相同问题时的得分进行比较。

如果您是一个高分狂热者，章节挑战将有助于改进我们已经创建的令人印象深刻的模型。一定要练习工程特征，而不是过度训练，从而使模型过度拟合以基本上记住答案。

查找重要值、归一化特征和强调显著相关性是机器学习训练中的一项有用技能，现在您可以使用 Danfo.js 来实现这一点。

# 章节回顾

那么在本章开始时我们识别的那个个体发生了什么？凯特·康诺利小姐，一个 30 岁的持有三等舱票的女人，*确实*幸存了泰坦尼克号事故，模型也认同。

我们是否错过了一些提高机器学习模型准确性的史诗机会？也许我们应该用`-1`填充空值而不是删除它们？也许我们应该研究一下泰坦尼克号的船舱结构？或者我们应该查看`parch`、`sibsp`和`pclass`，为独自旅行的三等舱乘客创建一个新列？“我永远不会放手！”

并非所有数据都可以像泰坦尼克号数据集那样被清理和特征化，但这对于机器学习来说是一次有用的数据科学冒险。有很多 CSV 文件可用，自信地加载、理解和处理它们对于构建新颖模型至关重要。像 Danfo.js 这样的工具使您能够处理这些海量数据，现在您可以将其添加到您的机器学习工具箱中。

###### 注意

如果您已经是其他 JavaScript 笔记本的粉丝，比如[ObservableHQ.com](https://observablehq.com)，Danfo.js 也可以导入并与这些笔记本轻松集成。

处理数据是一件复杂的事情。有些问题更加明确，根本不需要对特征进行任何调整。如果您感兴趣，可以看看像[帕尔默企鹅](https://oreil.ly/CiNv5)这样的更简单的数据集。这些企鹅根据它们的嘴的形状和大小明显地区分为不同的物种。另一个简单的胜利是第七章中提到的鸢尾花数据集。

## 章节挑战：船只发生了什么

您知道在泰坦尼克号沉没中没有一个牧师幸存下来吗？像先生、夫人、小姐、牧师等这样的头衔桶/箱可能对模型的学习有用。这些*敬称*——是的，就是它们被称为的——可以从被丢弃的`Name`列中收集和分析。

在这个章节挑战中，使用 Danfo.js 识别在泰坦尼克号上使用的敬称及其相关的生存率。这是一个让您熟悉 Dnotebooks 的绝佳机会。

您可以在附录 B 中找到这个挑战的答案。

## 审查问题

让我们回顾一下你在本章编写的代码中学到的教训。花点时间回答以下问题：

1.  对于一个石头-剪刀-布分类器，你会使用什么样的激活函数？

1.  在一个 sigmoid“狗还是不是狗”模型的最终层中会放置多少个节点？

1.  加载一个具有内置 Danfo.js 的交互式本地托管笔记本的命令是什么？

1.  如何将具有相同列的两个 CSV 文件的数据合并？

1.  你会使用什么命令将单个列进行独热编码成多个列？

1.  你可以使用什么来将 DataFrame 的所有值在 0 和 1 之间进行缩放？

这些练习的解决方案可以在附录 A 中找到。
