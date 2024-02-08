# 第三章。使用DeepChem进行机器学习

本章简要介绍了使用DeepChem进行机器学习的内容，DeepChem是建立在TensorFlow平台之上的库，旨在促进在生命科学领域中使用深度学习。DeepChem提供了大量适用于生命科学应用的模型、算法和数据集。在本书的其余部分，我们将使用DeepChem来进行案例研究。

# 为什么不直接使用Keras、TensorFlow或PyTorch？

这是一个常见问题。简短的答案是，这些软件包的开发人员专注于支持对他们的核心用户有用的某些类型的用例。例如，对于图像处理、文本处理和语音分析有广泛的支持。但是在这些库中通常没有类似的支持来处理分子、基因数据集或显微镜数据集。DeepChem的目标是为这些应用程序提供一流的支持。这意味着添加自定义的深度学习原语、支持所需的文件类型，以及为这些用例提供广泛的教程和文档。

DeepChem还设计为与TensorFlow生态系统很好地集成，因此您应该能够将DeepChem代码与其他TensorFlow应用程序代码混合使用。

在本章的其余部分，我们将假设您已经在您的计算机上安装了DeepChem，并且准备运行示例。如果您尚未安装DeepChem，不要担心。只需访问[DeepChem网站](https://deepchem.io/)，并按照您系统的安装说明进行操作。

# DeepChem对Windows的支持

目前，DeepChem不支持在Windows上安装。如果可能的话，我们建议您使用Mac或Linux工作站来完成本书中的示例。我们从用户那里得知，DeepChem可以在更现代的Windows发行版中的Windows子系统Linux（WSL）上运行。

如果您无法获得Mac或Linux机器的访问权限，或者无法使用WSL，我们很乐意帮助您获得DeepChem在Windows上的支持。请联系作者，告诉我们您遇到的具体问题，我们将尽力解决。我们希望在未来版本的书中取消这一限制，并为未来的读者提供对Windows的支持。

# DeepChem数据集

DeepChem使用`Dataset`对象的基本抽象来封装用于机器学习的数据。`Dataset`包含有关一组样本的信息：输入向量`x`、目标输出向量`y`，以及可能包括每个样本表示的描述等其他信息。有不同方式存储数据的`Dataset`的子类。特别是，`NumpyDataset`对象作为NumPy数组的便捷包装器，并将被广泛使用。在本节中，我们将演示如何使用`NumpyDataset`进行一个简单的代码案例研究。所有这些代码都可以在交互式Python解释器中输入；在适当的情况下，输出将显示出来。

我们从一些简单的导入开始：

```py
import deepchem as dc
import numpy as np

```

让我们现在构建一些简单的NumPy数组：

```py
x = np.random.random((4, 5))
y = np.random.random((4, 1))

```

这个数据集将有四个样本。数组`x`对于每个样本有五个元素（“特征”），而`y`对于每个样本有一个元素。让我们快速查看我们抽样的实际数组（请注意，当您在本地运行此代码时，您应该期望看到不同的数字，因为您的随机种子将是不同的）：

```py
In : x
Out:
array([[0.960767 , 0.31300931, 0.23342295, 0.59850938, 0.30457302],
   [0.48891533, 0.69610528, 0.02846666, 0.20008034, 0.94781389],
   [0.17353084, 0.95867152, 0.73392433, 0.47493093, 0.4970179 ],
   [0.15392434, 0.95759308, 0.72501478, 0.38191593, 0.16335888]])

In : y
Out:
array([[0.00631553],
   [0.69677301],
   [0.16545319],
   [0.04906014]])

```

让我们现在将这些数组封装在一个`NumpyDataset`对象中：

```py
dataset = dc.data.NumpyDataset(x, y)

```

我们可以解开`dataset`对象，以获取我们存储在其中的原始数组：

```py
In : print(dataset.X)
[[0.960767 0.31300931 0.23342295 0.59850938 0.30457302]
[0.48891533 0.69610528 0.02846666 0.20008034 0.94781389]
[0.17353084 0.95867152 0.73392433 0.47493093 0.4970179 ]
[0.15392434 0.95759308 0.72501478 0.38191593 0.16335888]]

In : print(dataset.y)
[[0.00631553]
[0.69677301]
[0.16545319]
[0.04906014]]

```

请注意，这些数组与原始数组`x`和`y`相同：

```py
In : np.array_equal(x, dataset.X)
Out : True

In : np.array_equal(y, dataset.y)
Out : True

```

# 其他类型的数据集

DeepChem支持其他类型的`Dataset`对象，如前所述。当处理无法完全存储在计算机内存中的大型数据集时，这些类型主要变得有用。DeepChem还集成了使用TensorFlow的`tf.data`数据集加载工具的功能。我们将在需要时涉及这些更高级的库功能。

# 训练一个模型来预测分子的毒性

在这一部分，我们将演示如何使用DeepChem来训练一个模型来预测分子的毒性。在后面的章节中，我们将更深入地解释分子毒性预测的工作原理，但在这一部分，我们将把它作为一个黑盒示例，展示DeepChem模型如何用于解决机器学习挑战。让我们从一对必要的导入开始：

```py
import numpy as np
import deepchem as dc

```

接下来是加载用于训练机器学习模型的相关毒性数据集。DeepChem维护一个名为`dc.molnet`（MoleculeNet的缩写）的模块，其中包含一些用于机器学习实验的预处理数据集。特别是，我们将使用`dc.molnet.load_tox21()`函数，它将为我们加载和处理Tox21毒性数据集。当您第一次运行这些命令时，DeepChem将在您的计算机上本地处理数据集。您应该期望看到如下的处理说明：

```py
In : tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21()
Out: Loading raw samples now.
shard_size: 8192
About to start loading CSV from /tmp/tox21.CSV.gz
Loading shard 1 of size 8192.
Featurizing sample 0
Featurizing sample 1000
Featurizing sample 2000
Featurizing sample 3000
Featurizing sample 4000
Featurizing sample 5000
Featurizing sample 6000
Featurizing sample 7000
TIMING: featurizing shard 0 took 15.671 s
TIMING: dataset construction took 16.277 s
Loading dataset from disk.
TIMING: dataset construction took 1.344 s
Loading dataset from disk.
TIMING: dataset construction took 1.165 s
Loading dataset from disk.
TIMING: dataset construction took 0.779 s
Loading dataset from disk.
TIMING: dataset construction took 0.726 s
Loading dataset from disk.

```

*特征化*的过程是将包含有关分子信息的数据集转换为矩阵和向量，以便在机器学习分析中使用。我们将在后续章节中更深入地探讨这个过程。不过，让我们从这里开始，快速查看我们处理过的数据。

`dc.molnet.load_tox21()`函数返回多个输出：`tox21_tasks`、`tox21_datasets`和`transformers`。让我们简要地看一下每一个：

```py
In : tox21_tasks
Out:
['NR-AR',
'NR-AR-LBD',
'NR-AhR',
'NR-Aromatase',
'NR-ER',
'NR-ER-LBD',
'NR-PPAR-gamma',
'SR-ARE',
'SR-ATAD5',
'SR-HSE',
'SR-MMP',
'SR-p53']

In : len(tox21_tasks)
Out: 12

```

这里的12个任务中的每一个对应于一个特定的生物实验。在这种情况下，这些任务中的每一个都是针对*酶活性测定*的，该测定衡量了Tox21数据集中的分子是否与所讨论的*生物靶标*结合。诸如`NR-AR`等术语对应于这些靶标。在这种情况下，这些靶标中的每一个都是一种被认为与潜在治疗分子的毒性反应相关联的特定酶。

# 我需要了解多少生物学知识？

对于进入生命科学领域的计算机科学家和工程师来说，生物学术语的范围可能令人眼花缭乱。然而，并不需要深入了解生物学就能开始在生命科学领域产生影响。如果您的主要背景是计算机科学，尝试以计算机科学的类比方式理解生物系统可能会有所帮助。想象细胞或动物是您无法控制的复杂遗留代码库。作为工程师，您有一些关于这些系统（测定）的实验性测量数据，可以用来对底层机制有一些了解。机器学习是理解生物系统的一种非常强大的工具，因为学习算法能够以大多数自动的方式提取有用的相关性。这使得即使是生物学初学者有时也能发现深刻的生物洞察。

在本书的其余部分，我们会简要讨论基础生物学。这些说明可以作为进入广阔生物学文献的入口点。公共参考资料，如维基百科，通常包含大量有用的信息，可以帮助启动您的生物学教育。

接下来，让我们考虑`tox21_datasets`。使用复数形式的提示表明，这个字段实际上是一个包含多个`dc.data.Dataset`对象的元组：

```py
In : tox21_datasets
Out:
(<deepchem.data.datasets.DiskDataset at 0x7f9804d6c390>,
<deepchem.data.datasets.DiskDataset at 0x7f9804d6c780>,
<deepchem.data.datasets.DiskDataset at 0x7f9804c5a518>)

```

在这种情况下，这些数据集对应于您在上一章中了解的训练、验证和测试集。您可能注意到这些是`DiskDataset`对象；`dc.molnet`模块会将这些数据集缓存在您的磁盘上，这样您就不需要反复对Tox21数据集进行特征化。让我们正确地拆分这些数据集：

```py
train_dataset, valid_dataset, test_dataset = tox21_datasets

```

处理新数据集时，首先查看它们的形状非常有用。要这样做，请检查`shape`属性：

```py
In : train_dataset.X.shape
Out: (6264, 1024)

In : valid_dataset.X.shape
Out: (783, 1024)

In : test_dataset.X.shape
Out: (784, 1024)

```

`train_dataset`包含总共6,264个样本，每个样本都有一个长度为1,024的相关特征向量。同样，`valid_dataset`和`test_dataset`分别包含783和784个样本。现在让我们快速查看这些数据集的`y`向量：

```py
In : np.shape(train_dataset.y)
Out: (6264, 12)

In : np.shape(valid_dataset.y)
Out: (783, 12)

In : np.shape(test_dataset.y)
Out: (784, 12)

```

每个样本有12个数据点，也称为*标签*。这些对应于我们之前讨论的12个任务。在这个特定的数据集中，样本对应于分子，任务对应于生化测定，每个标签是特定分子上特定测定的结果。这些是我们想要训练模型来预测的内容。

然而，有一个复杂之处：Tox21的实际实验数据集并没有测试每个生物实验中的每种分子。这意味着一些标签是没有意义的占位符。对于一些分子的一些属性，我们根本没有数据，因此在训练和测试模型时需要忽略这些数组的元素。

我们如何找出哪些标签实际上被测量了？我们可以检查数据集的`w`字段，记录其*权重*。每当我们为模型计算损失函数时，我们在对任务和样本求和之前乘以`w`。这可以用于几个目的，其中一个是标记缺失数据。如果一个标签的权重为0，则该标签不会影响损失，并且在训练过程中会被忽略。让我们深入挖掘一下，找出我们的数据集中实际测量了多少个标签：

```py
In : train_dataset.w.shape
Out: (6264, 12)

In : np.count_nonzero(train_dataset.w)
Out: 62166

In : np.count_nonzero(train_dataset.w == 0)
Out: 13002

```

在标签数组中的6,264×12 = 75,168个元素中，只有62,166个实际测量过。其他13,002个对应于缺失的测量值，应该被忽略。您可能会问，为什么我们仍然保留这样的条目。答案主要是为了方便；不规则形状的数组比带有一组权重的常规矩阵更难在代码中进行推理和处理。

# 处理数据集具有挑战性

在这里需要注意的是，为了在生命科学中使用，清理和处理数据集可能非常具有挑战性。许多原始数据集将包含系统性的错误类别。如果所讨论的数据集是由外部组织（合同研究机构或CRO）进行的实验构建的，那么该数据集很可能是系统性错误的。因此，许多生命科学组织都会保留内部的科学家，他们的工作是验证和清理这些数据集。

一般来说，如果您的机器学习算法在生命科学任务中无法正常工作，很可能根本原因不是算法本身，而是您使用的数据源中存在的系统性错误。

现在让我们检查`transformers`，这是`load_tox21()`返回的最终输出。*转换器*是一种以某种方式修改数据集的对象。DeepChem提供许多可以以有用方式操作数据的转换器。在MoleculeNet中找到的数据加载例程总是返回已应用于数据的转换器列表，因为您可能以后需要它们来“取消转换”数据。让我们看看这种情况下有什么：

```py
In : transformers
Out: [<deepchem.trans.transformers.BalancingTransformer at 0x7f99dd73c6d8>]

```

这里的数据已经通过 `BalancingTransformer` 进行了转换。这个类用于纠正不平衡的数据。在 Tox21 的情况下，大多数分子不与大多数目标结合。事实上，超过 90% 的标签是 0。这意味着一个模型可以轻松地通过始终预测 0 来获得超过 90% 的准确率，无论输入是什么。不幸的是，那个模型将是完全无用的！在分类任务中，不平衡的数据，即某些类别的训练样本比其他类别多得多，是一个常见问题。

幸运的是，有一个简单的解决方案：调整数据集的权重矩阵以进行补偿。`BalancingTransformer` 调整单个数据点的权重，使得分配给每个类别的总权重相同。这样，损失函数对任何一个类别都没有系统偏好。损失只能通过学会正确区分类别来减少。

现在我们已经探索了 Tox21 数据集，让我们开始探索如何在这些数据集上训练模型。DeepChem 的 `dc.models` 子模块包含各种不同的生命科学特定模型。所有这些不同的模型都继承自父类 `dc.models.Model`。这个父类旨在提供一个遵循常见 Python 约定的通用 API。如果您使用过其他 Python 机器学习包，您应该会发现许多 `dc.models.Model` 方法看起来非常熟悉。

在本章中，我们不会深入探讨这些模型是如何构建的细节。相反，我们将提供一个实例，展示如何实例化一个标准的 DeepChem 模型，`dc.models.MultitaskClassifier`。这个模型构建了一个全连接网络（MLP），将输入特征映射到多个输出预测。这对于 *多任务* 问题非常有用，其中每个样本有多个标签。它非常适合我们的 Tox21 数据集，因为我们有 12 个不同的检测任务需要同时预测。让我们看看如何在 DeepChem 中构建一个 `MultitaskClassifier`：

```py
model = dc.models.MultitaskClassifier(n_tasks=12,
n_features=1024,
layer_sizes=[1000])

```

这里有各种不同的选项。让我们简要回顾一下。`n_tasks` 是任务的数量，`n_features` 是每个样本的输入特征数量。正如我们之前看到的，Tox21 数据集有 12 个任务和每个样本 1,024 个特征。`layer_sizes` 是一个设置网络中完全连接隐藏层数量和每个隐藏层宽度的列表。在这种情况下，我们指定有一个宽度为 1,000 的单隐藏层。

现在我们已经构建了模型，我们如何在 Tox21 数据集上训练它呢？每个 `Model` 对象都有一个 `fit()` 方法，用于将模型拟合到包含在 `Dataset` 对象中的数据中。然后，对我们的 `MultitaskClassifier` 对象进行拟合是一个简单的调用：

```py
model.fit(train_dataset, nb_epoch=10)

```

请注意，我们在这里添加了一个标志。`nb_epoch=10` 表示将进行 10 个梯度下降训练周期。一个 *epoch* 指的是对数据集中所有样本进行一次完整遍历。为了训练模型，您将训练集分成批次，并对每个批次进行一步梯度下降。在理想情况下，您会在数据用尽之前达到一个良好优化的模型。实际上，通常没有足够的训练数据，所以在模型完全训练之前就用尽了数据。然后您需要开始重复使用数据，对数据集进行额外的遍历。这样可以使用更少的数据训练模型，但使用的 epoch 越多，最终得到过拟合模型的可能性就越大。

现在让我们评估训练模型的性能。为了评估模型的工作效果，有必要指定一个度量标准。DeepChem类`dc.metrics.Metric`提供了一种为模型指定度量标准的通用方法。对于Tox21数据集，ROC AUC分数是一个有用的度量标准，所以让我们使用它进行分析。然而，请注意这里的一个细微之处：有多个Tox21任务。我们应该在哪一个上计算ROC AUC？一个好的策略是计算所有任务的平均ROC AUC分数。幸运的是，这很容易做到：

```py
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

```

由于我们指定了`np.mean`，所有任务的ROC AUC分数的平均值将被报告。DeepChem模型支持评估函数`model.evaluate()`，该函数评估模型在给定数据集和度量标准上的性能。

# ROC AUC

我们想将分子分类为有毒或无毒，但模型输出连续数字，而不是离散预测。在实践中，您选择一个阈值值，并预测当输出大于阈值时分子是有毒的。较低的阈值将产生许多假阳性（预测安全分子实际上是有毒的）。较高的阈值将产生较少的假阳性，但会产生更多的假阴性（错误地预测有毒分子是安全的）。

接收器操作特性（ROC）曲线是一种方便的可视化权衡方式。您可以尝试许多不同的阈值值，然后绘制真正阳性率与假阳性率随着阈值变化而变化的曲线。一个示例显示在[图3-1](#the_roc_curve_for_one_of_the_twelve)中。

ROC AUC是ROC曲线下的总面积。曲线下的面积（AUC）提供了模型区分不同类别的能力的指示。如果存在任何阈值值，每个样本都被正确分类，ROC AUC分数为1。在另一个极端，如果模型输出与真实类别无关的完全随机值，ROC AUC分数为0.5。这使得它成为一个用于总结分类器工作效果的有用数字。这只是一个启发式方法，但是它是一个流行的方法。

```py
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

```

现在我们已经计算了分数，让我们来看看！

```py
In : `print``(``train_scores``)`
...: `print``(``test_scores``)`
Out
{'mean-roc_auc_score': 0.9659541853946179}
{'mean-roc_auc_score': 0.7915464001982299}

```

请注意，我们在训练集上的得分（0.96）比测试集上的得分（0.79）要好得多。这表明模型已经过拟合。我们真正关心的是测试集得分。这些数字在这个数据集上并不是最好的可能值 - 在撰写本文时，Tox21数据集的最先进ROC AUC分数略低于0.9 - 但对于一个开箱即用的系统来说，它们并不算差。其中一个12个任务的完整ROC曲线显示在[图3-1](#the_roc_curve_for_one_of_the_twelve)中。

![](Images/dlls_0301.png)

###### 图3-1。12个任务中的一个的ROC曲线。虚线对角线显示了一个只是随机猜测的模型的曲线。实际曲线始终远高于对角线，表明我们比随机猜测要好得多。

# 案例研究：训练一个MNIST模型

在前一节中，我们介绍了使用DeepChem训练机器学习模型的基础知识。然而，我们使用了一个预先制作的模型类`dc.models.MultitaskClassifier`。有时候，您可能希望创建一个新的深度学习架构，而不是使用一个预配置的架构。在本节中，我们将讨论如何在MNIST数字识别数据集上训练卷积神经网络。与之前的示例不同，这次我们将自己指定完整的深度学习架构。为此，我们将介绍`dc.models.TensorGraph`类，它提供了在DeepChem中构建深度架构的框架。

# 何时使用预制模型有意义？

在本节中，我们将在MNIST上使用自定义架构。在之前的示例中，我们使用了“罐头”（即预定义）架构。每种选择何时合理？如果对于某个问题有一个经过充分调试的罐头架构，那么使用它可能是合理的。但如果你正在处理一个没有组合好这样的架构的新数据集，通常需要创建一个自定义架构。熟悉使用罐头和自定义架构是很重要的，因此我们在本章中包含了每种类型的示例。

## MNIST手写数字识别数据集

MNIST手写数字识别数据集（参见[图3-2](#samples_drawn_from_the_mnist_handwritten_digit_recognition_dataset)）需要构建一个机器学习模型，可以正确分类手写数字。挑战在于对0到9的数字进行分类，给定28×28像素的黑白图像。数据集包含60,000个训练示例和10,000个测试示例。

![](Images/dlls_0302.png)

###### 图3-2。从MNIST手写数字识别数据集中抽取的样本。 (来源：[GitHub](https://github.com/mnielsen/rmnist/blob/master/data/rmnist_10.png))

就机器学习问题而言，MNIST数据集并不特别具有挑战性。数十年的研究已经产生了最先进的算法，在这个数据集上实现了接近100%的测试集准确率。因此，MNIST数据集不再适用于研究工作，但对于教学目的来说是一个很好的工具。

# DeepChem只适用于生命科学吗？

正如我们在本章前面提到的，完全可以使用其他深度学习包来进行生命科学应用。同样，也可以使用DeepChem构建通用的机器学习系统。虽然在DeepChem中构建电影推荐系统可能比使用更专门的工具更困难，但这是完全可行的。而且有充分的理由：已经有多项研究探讨了将推荐系统算法用于分子结合预测的应用。在一个领域使用的机器学习架构通常会延伸到其他领域，因此保持创新工作所需的灵活性是很重要的。

## MNIST的卷积架构

DeepChem使用`TensorGraph`类来构建非标准的深度学习架构。在本节中，我们将逐步介绍构建卷积架构所需的代码，如[图3-3](#this_diagram_illustrates_the_artchitecture_that_we_will_construct_in_this)所示。它从两个卷积层开始，用于识别图像中的局部特征。然后是两个全连接层，用于从这些局部特征预测数字。

![](Images/dlls_0303.png)

###### 图3-3。本节中将构建的用于处理MNIST数据集的架构示意图。

首先，执行以下命令下载原始的MNIST数据文件并将其存储在本地：

```py
mkdir MNIST_data
cd MNIST_data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
cd ..

```

现在让我们加载这些数据集：

```py
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

```

我们将处理这些原始数据，使其适合DeepChem进行分析。让我们从必要的导入开始：

```py
import deepchem as dc
import tensorflow as tf
import deepchem.models.tensorgraph.layers as layers

```

子模块`deepchem.models.tensorgraph.layers`包含一系列“层”。这些层作为深度架构的构建模块，可以组合起来构建新的深度学习架构。我们将很快展示层对象是如何使用的。接下来，我们构建`NumpyDataset`对象，用来包装MNIST的训练和测试数据集：

```py
train_dataset = dc.data.NumpyDataset(mnist.train.images, mnist.train.labels)
test_dataset = dc.data.NumpyDataset(mnist.test.images, mnist.test.labels)

```

请注意，尽管最初没有定义测试数据集，但TensorFlow的`input_data()`函数会负责分离出一个适当的测试数据集供我们使用。有了训练和测试数据集，我们现在可以转向定义MNIST卷积网络的架构。

这是基于的关键概念是可以组合层对象来构建新模型。正如我们在上一章中讨论的，每个层从前面的层接收输入并计算一个输出，该输出可以传递给后续层。在最开始，有接收特征和标签的输入层。在另一端是返回执行计算结果的输出层。在这个例子中，我们将组合一系列层以构建一个图像处理卷积网络。我们首先定义一个新的`TensorGraph`对象：

```py
model = dc.models.TensorGraph(model_dir='mnist')

```

`model_dir`选项指定应保存模型参数的目录。您可以省略这一点，就像我们在之前的例子中所做的那样，但是那样模型就不会被保存。一旦Python解释器退出，您辛苦训练模型的所有工作都将被抛弃！指定一个目录允许您稍后重新加载模型并进行新的预测。

注意，由于`TensorGraph`继承自`Model`，因此该对象是`dc.models.Model`的一个实例，并支持我们之前看到的相同的`fit()`和`evaluate()`函数：

```py
In : isinstance(model, dc.models.Model)
Out: True

```

我们还没有向`model`中添加任何内容，因此我们的模型可能不太有趣。让我们通过使用`Feature`和`Label`类为特征和标签添加一些输入：

```py
feature = layers.Feature(shape=(None, 784))
label = layers.Label(shape=(None, 10))

```

MNIST包含大小为28×28的图像。当展平时，这些形成长度为784的特征向量。标签具有第二维度为10，因为有10个可能的数字值，并且该向量是独热编码的。请注意，`None`被用作输入维度。在构建在TensorFlow上的系统中，值`None`通常表示给定层能够接受该维度上任何大小的输入。换句话说，我们的对象`feature`能够接受形状为`(20, 784)`和`(97, 784)`的输入。在这种情况下，第一个维度对应于批量大小，因此我们的模型将能够接受任意数量样本的批次。

# 独热编码

MNIST数据集是分类的。也就是说，对象属于有限列表中的一个潜在类别。在这种情况下，这些类别是数字0到9。我们如何将这些类别馈送到机器学习系统中？一个明显的答案是简单地输入一个从0到9取值的单个数字。然而，出于各种技术原因，这种编码通常似乎效果不佳。人们通常使用的替代方法是*独热编码*。MNIST的每个标签是一个长度为10的向量，其中一个元素设置为1，其他所有元素设置为0。如果非零值在第0个索引处，则标签对应于数字0。如果非零值在第9个索引处，则标签对应于数字9。

为了将卷积层应用于我们的输入，我们需要将我们的平面特征向量转换为形状为`(28, 28)`的矩阵。为此，我们将使用一个`Reshape`层：

```py
make_image = layers.Reshape(shape=(None, 28, 28), in_layers=feature)

```

这里再次值`None`表示可以处理任意批量大小。请注意我们有一个关键字参数`in_layers=feature`。这表示`Reshape`层以我们先前的`Feature`层`feature`作为输入。现在我们已成功地重塑了输入，我们可以将其传递给卷积层：

```py
conv2d_1 = layers.Conv2D(num_outputs=32, activation_fn=tf.nn.relu,
                                         in_layers=make_image)
conv2d_2 = layers.Conv2D(num_outputs=64, activation_fn=tf.nn.relu,
                                         in_layers=conv2d_1)

```

在这里，`Conv2D`类对其输入的每个样本应用2D卷积，然后通过修正线性单元（ReLU）激活函数传递。请注意如何使用`in_layers`将先前的层传递给后续层作为输入。我们希望最后应用`Dense`（全连接）层到卷积层的输出。但是，`Conv2D`层的输出是2D的，因此我们首先需要应用一个`Flatten`层将我们的输入展平为一维（更准确地说，`Conv2D`层为每个样本产生一个2D输出，因此其输出具有三个维度；`Flatten`层将其折叠为每个样本的单个维度，或者总共两个维度）：

```py
flatten = layers.Flatten(in_layers=conv2d_2)
dense1 = layers.Dense(out_channels=1024, activation_fn=tf.nn.relu, 
					 in_layers=flatten)
dense2 = layers.Dense(out_channels=10, activation_fn=None, in_layers=dense1)

```

`Dense`层中的`out_channels`参数指定了层的宽度。第一层每个样本输出1,024个值，但第二层输出10个值，对应于我们的10个可能的数字值。现在我们希望将此输出连接到损失函数，以便我们可以训练输出以准确预测类别。我们将使用`SoftMaxCrossEntropy`损失来执行这种形式的训练：

```py
smce = layers.SoftMaxCrossEntropy(in_layers=[label, dense2])
loss = layers.ReduceMean(in_layers=smce)
model.set_loss(loss)

```

请注意，`SoftMaxCrossEntropy`层接受最后一个`Dense`层的标签和输出作为输入。它计算每个样本的损失函数的值，因此我们需要对所有样本进行平均以获得最终损失。这是通过调用`model.set_loss()`将`ReduceMean`层设置为我们模型的损失函数来完成的。

# SoftMax和SoftMaxCrossEntropy

通常希望模型输出概率分布。对于MNIST，我们希望输出给定样本代表每个数字的概率。每个输出必须为正，并且它们必须总和为1。实现这一点的一种简单方法是让模型计算任意数字，然后通过令人困惑地命名为*softmax*函数传递它们：

指数在分子中确保所有值为正，并且分母中的总和确保它们加起来为1。如果<math><mi>x</mi></math>的一个元素远远大于其他元素，则相应的输出元素非常接近1，而所有其他输出则非常接近0。

`SoftMaxCrossEntropy`首先使用softmax函数将输出转换为概率，然后计算这些概率与标签的交叉熵。请记住，标签是独热编码的：正确类别为1，其他所有类别为0。您可以将其视为概率分布！当正确类别的预测概率尽可能接近1时，损失最小。这两个操作（softmax后跟交叉熵）经常一起出现，将它们作为单个步骤进行计算比分开执行更稳定。

为了数值稳定性，像`SoftMaxCrossEntropy`这样的层会计算对数概率。我们需要使用`SoftMax`层来转换输出以获得每个类别的输出概率。我们将使用`model.add_output()`将此输出添加到`model`中：

```py
output = layers.SoftMax(in_layers=dense2)
model.add_output(output)

```

现在我们可以使用与上一节中调用的相同的`fit()`函数来训练模型：

```py
model.fit(train_dataset, nb_epoch=10)

```

请注意，这个方法调用可能需要一些时间在标准笔记本电脑上执行！如果函数执行速度不够快，请尝试使用`nb_epoch=1`。结果会更糟，但您将能够更快地完成本章的其余部分。

这次我们将定义我们的度量为准确率，即正确预测的标签比例：

```py
metric = dc.metrics.Metric(dc.metrics.accuracy_score)

```

然后我们可以使用与之前相同的计算来计算准确率：

```py
train_scores = model.evaluate(train_dataset, [metric])
test_scores = model.evaluate(test_dataset, [metric])

```

这会产生出色的性能：训练集的准确率为0.999，测试集的准确率为0.991。我们的模型正确识别了超过99%的测试集样本。

# 尝试获取GPU访问权限

正如您在本章中看到的，深度学习代码可能运行得相当慢！在一台好的笔记本电脑上训练卷积神经网络可能需要超过一个小时才能完成。这是因为这段代码依赖于对图像数据的大量线性代数运算。大多数CPU并不适合执行这些类型的计算。

如果可能的话，尽量获取现代图形处理单元的访问权限。这些卡最初是为游戏开发的，但现在用于许多类型的数值计算。大多数现代深度学习工作负载在GPU上运行速度要快得多。您将在本书中看到的示例也将更容易使用GPU完成。

如果无法获得GPU的访问权限，也不用担心。您仍然可以完成本书中的练习，只是可能会花费更长的时间（您可能需要在等待代码运行完成时喝杯咖啡或读本书）。

# 结论

在这一章中，您已经学会了如何使用DeepChem库来实现一些简单的机器学习系统。在本书的其余部分中，我们将继续使用DeepChem作为我们的首选库，所以如果您还没有对该库的基本知识有很好的掌握，不要担心。将会有更多的例子出现。

在接下来的章节中，我们将开始介绍在生命科学数据集上进行有效机器学习所需的基本概念。在下一章中，我们将向您介绍分子上的机器学习。
