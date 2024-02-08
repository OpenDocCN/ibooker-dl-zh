# 第8章。嵌入和表示学习

# 学习低维表示

在上一章中，我们用一个简单的论点来激励卷积架构。我们的输入向量越大，我们的模型就越大。具有大量参数的大型模型具有表现力，但它们也越来越需要数据。这意味着如果没有足够大量的训练数据，我们很可能会过拟合。卷积架构通过减少模型中的参数数量而不一定减少表现力来帮助我们应对维度灾难。

无论如何，卷积网络仍然需要大量标记的训练数据。对于许多问题来说，标记数据是稀缺且昂贵的。本章的目标是在标记数据稀缺但野生的无标记数据丰富的情况下开发有效的学习模型。我们将通过无监督学习*嵌入*或低维表示来解决这个问题。因为这些无监督模型可以帮助我们摆脱自动特征选择的繁重工作，我们可以使用生成的嵌入来使用需要更少数据的较小模型解决学习问题。这个过程总结在[图8-1](#using_embeddings_to_automate_feature_selection)中。

在开发学习良好嵌入的算法过程中，我们还将探索学习低维表示的其他应用，如可视化和语义哈希。我们将从考虑所有重要信息已经包含在原始输入向量中的情况开始。在这种情况下，学习嵌入等同于开发一种有效的压缩算法。

![](Images/fdl2_0801.png)

###### 图8-1\. 在面对稀缺标记数据时使用嵌入来自动化特征选择的示例

在下一节中，我们将介绍*主成分分析*（PCA），这是一种经典的降维方法。在接下来的章节中，我们将探索更强大的神经方法来学习压缩嵌入。

# 主成分分析

PCA的基本概念是找到一组轴，这些轴传达了关于我们数据集的最多信息。更具体地说，如果我们有*d*维数据，我们希望找到一个新的 <math alttext="m小于d"><mrow><mi>m</mi> <mo><</mo> <mi>d</mi></mrow></math> 维度的集合，尽可能保留原始数据集中的有价值信息。为简单起见，让我们选择 <math alttext="d等于2，m等于1"><mrow><mi>d</mi> <mo>=</mo> <mn>2</mn> <mo>,</mo> <mi>m</mi> <mo>=</mo> <mn>1</mn></mrow></math>。假设方差对应于信息，我们可以通过迭代过程执行这种转换。首先，我们找到一个沿着数据集具有最大方差的单位向量。因为这个方向包含最多信息，我们选择这个方向作为我们的第一个轴。然后从与这个第一个选择正交的向量集中，我们选择一个沿着数据集具有最大方差的新单位向量。这是我们的第二个轴。

我们继续这个过程，直到找到一共*d*个代表新轴的新向量。我们将数据投影到这组新轴上。然后我们决定一个好的值*m*，并且丢弃除了前*m*个轴（存储最多信息的主成分）之外的所有轴。结果显示在[图8-2](#illustration_of_pca)中。

![](Images/fdl2_0802.png)

###### 图8-2\. 主成分分析的示例，用于降维以捕获包含最多信息的维度（通过方差代表）

对于数学倾向的人来说，我们可以将这个操作视为投影到由数据集的相关矩阵的前*m*个特征向量张成的向量空间上，当数据集已经进行了z-score标准化（每个输入维度的零均值和单位方差）时，这等同于数据集的协方差矩阵。让我们将数据集表示为一个维度为 <math alttext="n times d"><mrow><mi>n</mi> <mo>×</mo> <mi>d</mi></mrow></math> 的矩阵**X**（即，*n*个输入，*d*个维度）。我们希望创建一个维度为 <math alttext="n times m"><mrow><mi>n</mi> <mo>×</mo> <mi>m</mi></mrow></math> 的嵌入矩阵**T**。我们可以使用关系**T** = **X**计算矩阵，其中**W**的每一列对应于矩阵 <math alttext="StartFraction 1 Over n EndFraction"><mfrac><mn>1</mn> <mi>n</mi></mfrac></math> **X**^Τ**X**的特征向量。具有线性代数背景或核心数据科学经验的人可能会看到PCA和奇异值分解（SVD）之间的惊人相似之处，我们将在[“理论：PCA和SVD”](#theory_sidebar)中更深入地讨论。

虽然PCA在几十年来一直被用于降维，但它在捕捉重要的分段线性或非线性关系方面表现得非常糟糕。例如，看看[图8-3](#situation_in_which_pca)中所示的例子。

这个例子展示了从两个同心圆中随机选择的数据点。我们希望PCA将转换这个数据集，以便我们可以选择一个新的轴，使我们能够轻松地分开这些点。不幸的是，这里没有一个线性方向包含比另一个更多的信息（在所有方向上方差相等）。相反，作为人类，我们注意到信息以非线性方式进行编码，即点距离原点的远近。有了这些信息，我们注意到极坐标变换（将点表示为它们距离原点的距离，作为新的水平轴，以及它们相对于原始x轴的角度，作为新的垂直轴）正好起到了作用。

[图8-3](#situation_in_which_pca)突出了像PCA这样的方法在捕捉复杂数据集中重要关系方面的缺点。因为我们在现实中可能遇到的大多数数据集（图像、文本等）都具有非线性关系，所以我们必须开发一种能够进行非线性降维的理论。深度学习从业者通过使用神经模型来弥补这一差距，我们将在下一节中介绍。

！[](Images/fdl2_0803.png)

###### 图8-3。PCA在数据降维方面无法进行最佳转换的情况

# 激励自动编码器架构

当我们谈论前馈网络时，我们讨论了每一层如何逐渐学习更相关的输入表示。事实上，在[第7章](ch07.xhtml#convolutional_neural_networks)中，我们取出了最终卷积层的输出，并将其用作输入图像的低维表示。暂且不谈我们希望以无监督的方式生成这些低维表示，总体上这些方法存在根本问题。具体来说，虽然所选层确实包含来自输入的信息，但网络已经被训练为关注解决手头任务关键的输入方面。因此，与输入的一些可能对其他分类任务重要但可能比当前任务不太重要的元素相关的信息丢失是相当显著的。

然而，这里的基本直觉仍然适用。我们定义了一个称为*自动编码器*的新网络架构。我们首先将输入压缩成一个低维向量。网络的这一部分被称为*编码器*，因为它负责生成低维嵌入或*编码*。网络的第二部分，而不是将嵌入映射到任意标签，而是尝试反转网络前半部分的计算并重构原始输入。这部分被称为*解码器*。整体架构如[图8-4](#autoencoder_architecture_attempts_to_construct)所示。

![](Images/fdl2_0804.png)

###### 图8-4。自动编码器架构试图将高维输入构建成低维嵌入，然后使用该低维嵌入来重构输入

为了展示自动编码器的惊人有效性，我们将构建并可视化自动编码器架构，如[图8-4](#autoencoder_architecture_attempts_to_construct)所示。具体来说，我们将突出其与PCA相比更好地分离MNIST数字的能力。

# 在PyTorch中实现自动编码器

2006年Hinton和Salakhutdinov撰写的开创性论文“使用神经网络降低数据的维度”描述了自动编码器。他们的假设是，神经模型提供的非线性复杂性将使他们能够捕捉线性方法（如PCA）所忽略的结构。为了证明这一点，他们在MNIST上进行了一个实验，使用自动编码器和PCA将数据集减少为2D数据点。在本节中，我们将重新创建他们的实验设置，以验证这一假设，并进一步探索前馈自动编码器的架构和属性。

[图8-5](#experimental_setup_for_dimensionality_reduction)中显示的设置是基于相同原则构建的，但现在将2D嵌入视为输入，并且网络试图重构原始图像。因为我们实质上是应用一个逆操作，所以我们设计解码器网络，使得自动编码器的形状像一个沙漏。解码器网络的输出是一个784维向量，可以重构为一个28×28的图像：

```py
class Decoder(nn.Module):
  def __init__(self, n_in, n_hidden_1, n_hidden_2, n_hidden_3, n_out):
    super(Decoder, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Linear(n_in, n_hidden_1, bias=True),
        nn.BatchNorm1d(n_hidden_1),
        nn.Sigmoid())
    self.layer2 = nn.Sequential(
        nn.Linear(n_hidden_1, n_hidden_2, bias=True),
        nn.BatchNorm1d(n_hidden_2),
        nn.Sigmoid())
    self.layer3 = nn.Sequential(
        nn.Linear(n_hidden_2, n_hidden_3, bias=True),
        nn.BatchNorm1d(n_hidden_3),
        nn.Sigmoid())
    n_size = math.floor(math.sqrt(n_out))
    self.layer4 = nn.Sequential(
        nn.Linear(n_hidden_3, n_out, bias=True),
        nn.BatchNorm1d(n_out),
        nn.Sigmoid(),
        nn.Unflatten(1, torch.Size([1, n_size,n_size])))

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    return self.layer4(x)

```

![](Images/fdl2_0805.png)

###### 图8-5。Hinton和Salakhutdinov在2006年使用的MNIST数据集降维实验设置

为了加快训练速度，我们将重用我们在[第7章](ch07.xhtml#convolutional_neural_networks)中使用的批量归一化策略。此外，因为我们想要可视化结果，我们将避免在我们的神经元中引入尖锐的转变。在这个例子中，我们将使用S形神经元而不是我们通常的ReLU神经元：

```py
decoder = Decoder(2,250,500,1000,784)

```

最后，我们需要构建一个描述我们模型功能如何的度量（或目标函数）。具体来说，我们想要衡量重构与原始图像之间的接近程度。我们可以通过简单地计算原始784维输入和重构的784维输出之间的距离来衡量这一点。更具体地说，给定一个输入向量I和一个重构O，我们希望最小化I和O之间的差值的值，也称为两个向量之间的L2范数。我们将这个函数平均到整个小批次上以生成我们的最终目标函数。最后，我们将使用Adam优化器训练网络，使用`torch.utils.tensorboard.SummaryWriter`在每个小批次记录所产生的错误的标量摘要。在PyTorch中，我们可以简洁地表示损失和训练操作如下：

```py
loss_fn = nn.MSELoss()
optimizer = optim.Adam(decoder.parameters(),
                       lr = 0.001,
                       betas=(0.9,0.999),
                       eps=1e-08)

trainset = datasets.MNIST('.',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
trainloader = DataLoader(trainset,
                         batch_size=32,
                         shuffle=True)
# Training Loop
NUM_EPOCHS = 5
for epoch in range(NUM_EPOCHS):
  for input, labels in trainloader:
    optimizer.zero_grad()
    code = encoder(input)
    output = decoder(code)
    #print(input.shape, output.shape)
    loss = loss_fn(output, input)
    optimizer.step()
  print(f"Epoch: {epoch} Loss: {loss}")

```

最后，我们需要一种方法来评估我们模型的泛化能力。像往常一样，我们将使用一个验证数据集，并计算相同的L2范数测量来评估模型。此外，我们将收集图像摘要，以便我们可以比较输入图像和重构图像：

```py
i = 0
with torch.no_grad():
  for images, labels in trainloader:
    if i == 3:
      break
    grid = utils.make_grid(images)
    plt.figure()
    plt.imshow(grid.permute(1,2,0))

    code = encoder(images)
    output = decoder(code)

    grid = utils.make_grid(output)
    plt.figure()
    plt.imshow(grid.permute(1,2,0))
    i += 1

```

我们可以使用TensorBoard可视化模型图、训练和验证成本以及图像摘要。只需运行以下命令：

```py
$ tensorboard --logdir ~/path/to/mnist_autoencoder_hidden=2_logs

```

然后将浏览器导航到*http://localhost:6006/*。 “Graph”选项卡的结果显示在[图8-6](#tensorflow_allows_us_to_neatly_view)中。

由于我们对模型图的组件进行了命名空间处理，我们的模型组织得很好。我们可以轻松地点击组件并深入研究，追踪数据如何通过编码器的各个层和解码器的各个层流动，优化器如何读取我们训练模块的输出，以及梯度如何影响模型的所有组件。

我们还可视化训练（每个小批次后）和验证成本（每个时代后），密切监控曲线以防止过拟合。训练期间成本的TensorBoard可视化显示在[图8-7](#cost_incurred_on_the_training_set)中。正如我们所期望的那样，对于一个成功的模型，训练和验证曲线都会下降，直到渐近平稳。大约在200个时代之后，我们获得了一个验证成本为4.78。虽然曲线看起来很有希望，但乍一看很难理解我们是否已经达到了“好”的成本平台，还是我们的模型仍然在重构原始输入方面做得很差。

![](Images/fdl2_0806.png)

###### 图8-6. TensorBoard允许我们清晰地查看计算图的高级组件和数据流（顶部），并且还可以点击查看各个子组件的数据流（底部）

![](Images/fdl2_0807.png)

###### 图8-7. 训练集上产生的成本（每个小批次后记录）和验证集上产生的成本（每个时代后记录）

为了了解这意味着什么，让我们探索MNIST数据集。我们从数据集中选择一个任意的数字1的图像，并将其称为*X*。在[图8-8](#compared_to_all_other_digits)中，我们将这个图像与数据集中的所有其他图像进行比较。具体来说，对于每个数字类别，我们计算与*X*与该类别的每个实例的L2成本的平均值。作为视觉辅助，我们还包括每个数字类别的所有实例的平均值。

![](Images/fdl2_0808.png)

###### 图8-8。左侧的数字1图像与MNIST数据集中的所有其他数字进行比较；每个数字类别在视觉上用其所有成员的平均值表示，并用与左侧的1与该类别所有成员的L2成本的平均值标记

在MNIST中，*X*平均与其他1之间的距离为5.75个单位。以L2距离来看，与*X*最接近的非1数字是7（8.94个单位），最远的数字是0（11.05个单位）。根据这些测量结果，很明显，我们的自动编码器产生了高质量的重构，平均成本为4.78。

因为我们正在收集图像摘要，我们可以通过直接检查输入图像和重构来直接确认这一假设。来自测试集的三个随机选择样本的重构显示在[图8-9](#side_by_side_comparison_of_original_inputs)中。

![](Images/fdl2_0809.png)

###### 图8-9。原始输入（来自验证集）和经过5、100和200个epoch训练后的重构的并排比较

经过五个epoch，我们可以开始看出自动编码器捕捉到原始图像的一些关键笔画，但在大部分情况下，重构仍然是与相关数字的模糊混合。到了100个epoch，0和4被强烈的笔画重构，但看起来自动编码器仍然难以区分5、3和可能8。然而，到了200个epoch，很明显即使是这种更困难的模糊也被澄清，所有数字都被清晰地重构。

最后，我们将通过探索传统PCA和自动编码器生成的2D代码来完成本节。我们希望展示自动编码器产生更好的可视化效果。特别是，我们希望展示自动编码器在视觉上更好地区分不同数字类别的实例。我们将从快速介绍生成2D PCA代码的代码开始：

```py
from sklearn import decomposition
import input_data

mnist = input_data.read_data_sets("data/", one_hot=False)
pca = decomposition.PCA(n_components=2)
pca.fit(mnist.train.images)
pca_codes = pca.transform(mnist.test.images)

```

首先我们拉取MNIST数据集。我们设置了标志`one_hot=False`，因为我们希望标签以整数形式提供，而不是作为one-hot向量（简单提醒一下，表示MNIST标签的one-hot向量将是一个大小为10的向量，其中第i个分量设置为1以表示数字i，其余分量设置为零）。我们使用常用的机器学习库*scikit-learn*执行PCA，设置`n_components=2`参数，以便scikit-learn知道生成2D代码。我们还可以从2D代码重构原始图像并可视化重构结果：

```py
from matplotlib import pyplot as plt

pca_recon = pca.inverse_transform(pca_codes[:1])
plt.imshow(pca_recon[0].reshape((28,28)), cmap=plt.cm.gray)
plt.show()
```

代码片段展示了如何可视化测试数据集中的第一幅图像，但我们可以轻松修改代码以可视化数据集的任意子集。将PCA重构与自动编码器重构在[图8-10](#comparing_reconstructions_by_pca_and_autoencoder)中进行比较，很明显自动编码器远远优于具有2D代码的PCA。事实上，PCA的性能有点像自动编码器训练五个epoch后。它难以区分5和3、8，0和8，4和9。使用30维代码重复相同实验，对PCA重构提供了显著改进，但仍然明显差于30维自动编码器。

![](Images/fdl2_0810.png)

###### 图8-10。将PCA和自动编码器的重建结果并排比较

现在，为了完成实验，我们必须加载一个保存的PyTorch模型，检索2D代码，并绘制PCA和自动编码器代码。我们小心地重建PyTorch图，确保与训练期间设置的一样。我们将训练期间保存的模型检查点路径作为命令行参数传递给脚本。最后，我们使用自定义绘图函数生成图例，并适当着色不同数字类别的数据点。

在[图8-11](#two_dimensional_embeddings_produced_by_pca)中的可视化结果中，很难看出2D PCA代码中的可分离聚类；自动编码器显然在对不同数字类别的代码进行聚类方面做得非常出色。这意味着一个简单的机器学习模型将能够更有效地对由自动编码器嵌入组成的数据点进行分类，与PCA嵌入相比。

![](Images/fdl2_0811.png)

###### 图8-11。PCA生成的2D嵌入（顶部）和自动编码器生成的2D嵌入（底部）

在本节中，我们成功地建立并训练了一个前馈自动编码器，并证明了生成的嵌入优于PCA，这是一种经典的降维方法。在下一节中，我们将探讨一种称为去噪的概念，它作为一种正则化形式，使我们的嵌入更加健壮。

# 去噪以强制生成稳健的表示形式

*去噪* 提高了自动编码器生成对噪声具有抗性的嵌入的能力。人类的感知能力对噪声非常有抵抗力。例如，看看[图8-12](#despite_the_corruption)。尽管我在每个图像中破坏了一半的像素，但您仍然可以轻松辨认出数字。事实上，即使是容易混淆的数字（如2和7）仍然可以区分开来。

![](Images/fdl2_0812.png)

###### 图8-12。人类感知能力使我们能够识别甚至被遮挡的数字

观察这种现象的一种方式是从概率的角度来看。即使我们暴露于图像的随机像素采样中，如果我们有足够的信息，我们的大脑仍然能够以最大概率得出像素代表的真实情况。我们的大脑能够，确实地，填补空白以得出结论。即使我们的视网膜只接收到一个数字的损坏版本，我们的大脑仍然能够重现我们通常用来表示该数字图像的一组激活（即代码或嵌入）。这是我们希望在我们的嵌入算法中强制执行的一种属性，这是由Vincent等人在2008年首次探索的，当时他们引入了*去噪自动编码器*。

去噪背后的基本原理非常简单。我们通过将输入图像中的一定百分比的像素设为零来破坏输入图像。给定原始输入*X*，让我们称破坏版本为 <math alttext="upper C left-parenthesis upper X right-parenthesis"><mrow><mi>C</mi> <mo>(</mo> <mi>X</mi> <mo>)</mo></mrow></math> 。去噪自动编码器与普通自动编码器相同，除了一个细节：编码器网络的输入是破坏的 <math alttext="upper C left-parenthesis upper X right-parenthesis"><mrow><mi>C</mi> <mo>(</mo> <mi>X</mi> <mo>)</mo></mrow></math> 而不是*X*。换句话说，自动编码器被迫学习对每个输入都具有抗干扰机制的代码，并且能够通过缺失的信息插值来重新创建原始的未损坏图像。

我们也可以更几何地思考这个过程。假设我们有一个带有各种标签的2D数据集。让我们取特定类别（即具有某个固定标签）中的所有数据点，并将这些数据点的子集称为*S*。虽然在可视化时，任意抽样的点可能呈现任何形式，但我们假设对于现实生活中的类别，存在一些统一所有*S*中点的潜在结构。这种潜在的、统一的几何结构被称为*流形*。当我们降低数据的维度时，我们希望捕捉到的形状就是流形；正如Bengio等人在2013年所描述的，我们的自动编码器在通过瓶颈（代码层）推送数据后学习重建数据时，隐式地学习了这个流形。自动编码器必须弄清楚一个点属于哪个流形，当尝试生成一个具有潜在不同标签的实例的重建时。

举个例子，让我们考虑[图8-13](#denoising_objective)中的情景，其中*S*中的点是一个简单的低维流形（图中的实心圆）。在A部分，我们看到*S*中的数据点（黑色x）和最能描述它们的流形。我们还观察到我们的损坏操作的近似。具体来说，箭头和非同心圆展示了损坏可能移动或修改数据点的所有方式。考虑到我们将这种损坏操作应用于每个数据点（即整个流形），这种损坏操作人为地扩展了数据集，不仅包括流形，还包括流形周围空间中的所有点，直到最大误差边界。这个边界由A部分的虚线圆表示，数据集的扩展由B部分的x表示。最后，自动编码器被迫学习将这个空间中的所有数据点折叠回流形。换句话说，通过学习数据点中哪些方面是可概括的、宽泛的，哪些方面是“噪音”，去噪自动编码器学会了近似*S*的潜在流形。

![](Images/fdl2_0813.png)

###### 图8-13。去噪目标使我们的模型通过最小化表示之间的误差（C部分的箭头）来学习流形（黑色圆），通过学习将损坏的数据（B和C部分的浅色x）映射到未损坏的数据（黑色x）

考虑到去噪的哲学动机，我们现在可以对我们的自动编码器脚本进行小的修改，构建一个去噪自动编码器：

```py
def corrupt_input(x):
    corrupting_matrix = 2.0*torch.rand_like(x)

    return x * corrupting_matrix

# x = mnist data image of shape 28*28=784
x = torch.rand((28,28))
corrupt = 1.0 # set to 1.0 to corrupt input
c_x = (corrupt_input(x) * corrupt) + (x * (1 - corrupt))

```

如果`corrupt`变量等于1，此代码片段会损坏输入，如果`corrupt`变量等于0，则不会损坏输入。在进行这种修改后，我们可以重新运行我们的自动编码器，得到[图8-14](#apply_a_corruption_operation)中显示的重建结果。很明显，去噪自动编码器已经忠实地复制了我们令人难以置信的人类能力，填补缺失的像素。

![](Images/fdl2_0814.png)

###### 图8-14。我们对数据集应用损坏操作，并训练一个去噪自动编码器来重建原始的、未损坏的图像

# 自动编码器中的稀疏性

深度学习中最困难的一个方面是一个被称为*可解释性*的问题。可解释性是衡量机器学习模型的一个属性，用于衡量检查和解释其过程和/或输出的难易程度。由于构成模型的非线性和大量参数，深度模型通常难以解释。虽然深度模型通常更准确，但缺乏可解释性通常会阻碍它们在高价值但高风险的应用中的采用。例如，如果一个机器学习模型预测患者是否患有癌症，医生可能会希望得到解释以确认模型的结论。

我们可以通过探索自动编码器的输出特征来解决可解释性的一个方面。一般来说，自动编码器的表示是密集的，这对于我们在对输入进行连贯修改时表示如何变化具有影响。考虑在[图8-15](#activations_of_a_dense_representation)中的情况。

![](Images/fdl2_0815.png)

###### 图8-15\. 密集表示的激活以难以解释的方式结合和叠加多个特征的信息

自动编码器产生*密集*表示，即原始图像的表示被高度压缩。由于表示中只有有限的维度可用，表示的激活以极其难以分解的方式结合了多个特征的信息。结果是，当我们添加组件或删除组件时，输出表示以意想不到的方式变化。几乎不可能解释表示是如何生成的以及为什么生成的。

对我们来说，理想的结果是我们能够构建一个表示，其中高级特征与代码中的各个组件之间存在一对一的对应，或接近一对一的对应。当我们能够实现这一点时，我们就非常接近[图8-16](#right_combo_of_space_and_sparsity)中描述的系统，该系统显示了随着添加和删除组件表示的变化。表示是图像中各个笔画的总和。通过正确的空间和稀疏组合，表示更具可解释性。

![](Images/fdl2_0816.png)

###### 图8-16\. 表示中的激活随着笔画的添加和删除而变化

虽然这是理想的结果，但我们必须考虑可以利用哪些机制来实现表示中的可解释性。问题显然在于代码层的瓶颈容量；但不幸的是，仅增加代码层的容量是不够的。在中等情况下，虽然我们可以增加代码层的大小，但没有机制可以阻止自动编码器捕捉到的每个单独特征影响具有较小幅度的大部分组件。在更极端的情况下，捕捉到的特征更复杂，因此更丰富，代码层的容量可能甚至大于输入的维度。在这种情况下，代码层的容量非常大，以至于模型可能实际上执行“复制”操作，其中代码层学习不到任何有用的表示。

我们真正想要的是强制自动编码器尽可能少地利用表示向量的组件，同时有效地重建输入。这类似于在简单神经网络中使用正则化来防止过拟合的原理，正如我们在[第4章](ch04.xhtml#training_feed_forward)中讨论的那样，只是我们希望尽可能多的组件为零（或非常接近零）。与[第4章](ch04.xhtml#training_feed_forward)一样，我们将通过在目标函数中添加稀疏惩罚来实现这一点，这会增加具有大量非零组件的任何表示的成本：

<math alttext="upper E Subscript Sparse Baseline equals upper E plus beta dot SparsityPenalty"><mrow><msub><mi>E</mi> <mtext>Sparse</mtext></msub> <mo>=</mo> <mi>E</mi> <mo>+</mo> <mi>β</mi> <mo>·</mo> <mtext>SparsityPenalty</mtext></mrow></math>

<math alttext="beta"><mi>β</mi></math>的值决定了我们在追求稀疏性的同时牺牲生成更好重建的程度。对于数学倾向的人来说，您可以将每个表示的每个组件的值视为具有未知均值的随机变量的结果。然后，我们将使用一个衡量观察这个随机变量（每个组件的值）的分布和已知均值为0的随机变量的分布之间差异的度量。用于此目的的常用度量是Kullback-Leibler（通常称为KL）散度。关于自动编码器中稀疏性的进一步讨论超出了本文的范围，但已被Ranzato等人（2007年^([4](ch08.xhtml#idm45934167858016))和2008年^([5](ch08.xhtml#idm45934167856400))）涵盖。最近，Makhzani和Frey（2014年）^([6](ch08.xhtml#idm45934167853584))研究了在编码层之前引入一个中间函数的理论性质和经验有效性，该函数将表示中的最大激活值之外的所有值都归零。这些*k-稀疏自动编码器*被证明与其他稀疏机制一样有效，尽管实现和理解起来非常简单（以及在计算上更有效）。

这结束了我们对自动编码器的讨论。我们已经探讨了如何使用自动编码器通过总结其内容来找到数据点的强表示。当独立数据点丰富并包含有关其结构的所有相关信息时，这种降维的机制效果很好。在下一节中，我们将探讨当主要信息源是数据点的上下文而不是数据点本身时，我们可以使用的策略。

# 当上下文比输入向量更具信息性时

到目前为止，我们主要关注了降维的概念。在降维中，我们通常有包含大量噪音的丰富输入，这些噪音覆盖了我们关心的核心结构信息。在这些情况下，我们希望提取这些基本信息，同时忽略与数据的基本理解无关的变化和噪音。

在其他情况下，我们有输入表示几乎没有关于我们试图捕捉的内容的信息。在这些情况下，我们的目标不是提取信息，而是从上下文中收集信息以构建有用的表示。在这一点上，所有这些可能听起来太抽象而不实用，所以让我们用一个真实的例子来具体化这些想法。

为语言构建模型是一项棘手的工作。构建语言模型时我们必须克服的第一个问题是找到表示单词的好方法。乍一看，如何构建一个好的表示并不完全清楚。让我们从天真的方法开始，考虑[图8-17](#example_of_generating_one_hot_vector_reps)。

![](Images/fdl2_0817.png)

###### 图8-17。使用简单文档生成单热向量表示单词

如果一个文档有一个词汇表<math alttext="upper V"><mi>V</mi></math>，其中有<math alttext="StartAbsoluteValue upper V EndAbsoluteValue"><mrow><mo>|</mo> <mi>V</mi> <mo>|</mo></mrow></math>个单词，我们可以用单热向量表示这些单词。我们有<math alttext="StartAbsoluteValue upper V EndAbsoluteValue"><mrow><mo>|</mo> <mi>V</mi> <mo>|</mo></mrow></math>维表示向量，并将每个唯一的单词与此向量中的一个索引关联起来。为了表示唯一的单词<math alttext="w Subscript i"><msub><mi>w</mi> <mi>i</mi></msub></math>，我们将向量的第<math alttext="i Superscript t h"><msup><mi>i</mi> <mrow><mi>t</mi><mi>h</mi></mrow></msup></math>个分量设置为1，并将所有其他分量归零。

然而，这种表示方案似乎相当任意。这种向量化并不会使相似的词成为相似的向量。这是有问题的，因为我们希望我们的模型知道“jump”和“leap”这两个词有相似的含义。同样，我们希望我们的模型知道哪些词是动词、名词或介词。将单词进行朴素的独热编码到向量中并不捕捉这些特征。为了解决这一挑战，我们需要找到一种方法来发现这些关系，并将这些信息编码到一个向量中。

事实证明，发现单词之间关系的一种方法是分析它们周围的上下文。例如，同义词“jump”和“leap”可以在各自的上下文中互换使用。此外，这两个词通常出现在主语执行动作的直接对象上。当我们阅读时，我们一直在使用这个原则。例如，如果我们读到句子“The warmonger argued with the crowd”，即使我们不知道字典定义，我们也可以立即对“warmonger”这个词做出推断。在这个上下文中，“warmonger”在我们知道是一个动词之前出现，这使得“warmonger”很可能是一个名词，是这个句子的主语。此外，“warmonger”在“arguing”，这可能意味着“warmonger”通常是一个好斗或好争论的个体。总的来说，如[图8-18](#id_words_with_similar_meanings)所示，通过分析上下文（即围绕目标词的固定窗口词），我们可以快速推断单词的含义。

![](Images/fdl2_0818.png)

###### 图8-18. 分析上下文以确定单词含义

事实证明，我们可以使用构建自动编码器时使用的相同原则来构建一个生成强大、分布式表示的网络。[图8-19](#general_architectures_for_designing_encoders)展示了两种策略。一种可能的方法（如A所示）通过编码器网络将目标传递，以创建一个嵌入。然后，我们有一个解码器网络接受这个嵌入；但是与自动编码器中尝试重建原始输入不同，解码器尝试从上下文中构建一个词。第二种可能的方法（如B所示）完全相反：编码器将上下文中的一个词作为输入，生成目标。

![](Images/fdl2_0819.png)

###### 图8-19. 设计编码器和解码器的一般架构，通过将单词映射到它们各自的上下文（A）或反之（B）生成嵌入

在接下来的部分，我们将描述如何使用这种策略（以及一些性能上的轻微修改）来实际产生单词嵌入。

# Word2Vec框架

Word2Vec是由Mikolov等人开创的用于生成单词嵌入的框架。原始论文详细介绍了生成嵌入的两种策略，类似于我们在前一节讨论的编码上下文的两种策略。

Mikolov等人介绍的Word2Vec的第一种模型是Continuous Bag of Words（CBOW）模型。这个模型与[图8-19](#general_architectures_for_designing_encoders)中的策略B非常相似。CBOW模型使用编码器从完整上下文（作为一个输入）创建嵌入并预测目标词。事实证明，这种策略对于较小的数据集效果最好，这一属性在原始论文中进一步讨论。

Word2Vec的第二种模型是Skip-Gram模型，由Mikolov等人介绍。Skip-Gram模型与CBOW相反，将目标词作为输入，然后尝试预测上下文中的一个词。让我们通过一个玩具示例来探索Skip-Gram模型的数据集是什么样子的。

考虑句子“the boy went to the bank.”如果我们将这个句子分解成一系列的（上下文，目标）对，我们将得到[([the, went], boy), ([boy, to], went), ([went, the], to), ([to, bank], the)]。进一步地，我们必须将每个（上下文，目标）对拆分成（输入，输出）对，其中输入是目标，输出是上下文中的一个词。从第一个对([the, went], boy)开始，我们将生成两对(boy, the)和(boy, went)。我们继续将这个操作应用到每个（上下文，目标）对，以构建我们的数据集。最后，我们用词汇表中的索引替换每个单词的唯一索引 <math alttext="i element-of StartSet 0 comma 1 comma ellipsis comma StartAbsoluteValue upper V EndAbsoluteValue minus 1 EndSet"><mrow><mi>i</mi> <mo>∈</mo> <mo>{</mo> <mn>0</mn> <mo>,</mo> <mn>1</mn> <mo>,</mo> <mo>...</mo> <mo>,</mo> <mo>|</mo> <mi>V</mi> <mo>|</mo> <mo>-</mo> <mn>1</mn> <mo>}</mo></mrow></math>。

编码器的结构非常简单。它本质上是一个查找表，有<math alttext="StartAbsoluteValue upper V EndAbsoluteValue"><mrow><mo>|</mo> <mi>V</mi> <mo>|</mo></mrow></math>行，其中第<math alttext="i Superscript t h"><msup><mi>i</mi> <mrow><mi>t</mi><mi>h</mi></mrow></msup></math>行是对应于第<math alttext="i Superscript t h"><msup><mi>i</mi> <mrow><mi>t</mi><mi>h</mi></mrow></msup></math>个词汇的嵌入。编码器所需做的就是获取输入单词的索引，并输出查找表中的适当行。这是一个高效的操作，因为在GPU上，这个操作可以表示为查找表的转置和表示输入单词的独热向量的乘积。我们可以在PyTorch中简单实现这一点，使用以下PyTorch函数：

```py
emb = nn.Embedding(10, 100)
x = torch.tensor([0])
out = emb(x)
```

其中`out`是嵌入矩阵，`x`是我们想要查找的索引张量。有关可选参数的信息，请参阅[PyTorch API文档](https://oreil.ly/NaQWV)。

解码器稍微复杂，因为我们对性能进行了一些修改。构建解码器的朴素方法是尝试重建输出的独热编码向量，我们可以使用一个普通的前馈层和softmax来实现。唯一的问题是效率低下，因为我们必须在整个词汇空间上产生一个概率分布。

为了减少参数数量，Mikolov等人使用了一种实现解码器的策略，称为噪声对比估计（NCE）。该策略在[图8-20](#illustration_of_noise_contrastive_esimation)中有所说明。二元逻辑回归比较目标的嵌入与上下文单词的嵌入以及随机抽样的非上下文单词。我们构建了一个损失函数，描述了嵌入如何有效地使得识别目标上下文中的单词与目标外的单词成为可能。

![](Images/fdl2_0820.png)

###### 图8-20。NCE策略

NCE策略使用查找表来找到输出的嵌入，以及来自词汇表的非上下文输入的嵌入。然后，我们使用二元逻辑回归模型，一次一个，获取输入嵌入和输出或随机选择的嵌入，然后输出一个值介于0到1之间，对应于比较嵌入表示词汇单词是否存在于输入上下文中的概率。然后，我们取与非上下文比较对应的概率之和，并减去与上下文比较对应的概率。这个值是我们要最小化的目标函数（在模型表现完美的最佳情况下，该值将为-1）。

在PyTorch中实现NCE的示例可以在[GitHub](https://oreil.ly/lH2ip)上找到。

虽然Word2Vec毫无疑问不是一个深度机器学习模型，但我们在这里讨论它有很多原因。首先，它在主题上代表了一种策略（使用上下文找到嵌入），这种策略可以推广到许多深度学习模型。当我们在[第9章](ch09.xhtml#ch07)学习序列分析模型时，我们将看到这种策略用于生成skip-thought向量以嵌入句子。此外，当我们从[第9章](ch09.xhtml#ch07)开始构建更多语言模型时，我们将发现使用Word2Vec嵌入代替独热向量来表示单词将产生更好的结果。

现在我们了解了如何设计Skip-Gram模型及其重要性，我们可以开始在PyTorch中实现它。

# 实现Skip-Gram架构

为了构建我们的Skip-Gram模型的数据集，我们将使用`input_word_data.py`中的PyTorch Word2Vec数据读取器的修改版本。我们将首先设置一些重要的训练参数，并定期检查我们的模型。特别值得注意的是，我们使用32个示例的小批量大小，并训练5个时期（完整通过数据集）。我们将使用大小为128的嵌入。我们将使用每个目标词左右各五个单词的上下文窗口，并从该窗口中随机选择四个上下文单词。最后，我们将使用64个随机选择的非上下文单词进行NCE。

实现嵌入层并不特别复杂。我们只需用一个值矩阵初始化查找表：

```py
vocab_size = 500
emb_vector_len = 128

embedding = nn.Embedding(num_embeddings = vocab_size,
                         embedding_dim = emb_vector_len)

```

PyTorch目前没有内置的NCE损失函数。但是，互联网上有一些实现。一个例子是*info-nce-pytorch*库：

```py
pip install info-nce-pytorch
```

我们利用`InfoNCE`来计算每个训练样本的NCE成本，然后将所有结果编译到一个小批量中进行单一测量：

```py
loss = InfoNCE()
batch_size, embedding_size = 32, 128
query = embedding(outputs)
positive_key = embedding(targets)
output = loss(query, positive_key)

```

现在我们已经将我们的目标函数表达为NCE成本的平均值，我们像往常一样设置训练。在这里，我们跟随Mikolov等人的脚步，使用学习率为0.1的随机梯度下降：

```py
optimizer = optim.SGD(embedding.parameters(),
                      lr = 0.1)
def train(inputs, targets, embedding):
  optimizer.zero_grad()
  input_emb = embedding(inputs)
  target_emb = embedding(targets)
  loss = loss_fn(input_emb, target_emb)
  loss.backward()
  optimizer.step()
  return loss

```

我们还定期使用验证函数检查模型，该函数将查找表中的嵌入归一化，并使用余弦相似度计算一组验证单词与词汇表中所有其他单词之间的距离：

```py
cosine_similarity = nn.CosineSimilarity()

def evaluate(inputs, targets, embedding):
  with torch.no_grad():
    input_emb = embedding(inputs)
    target_emb = embedding(targets)
    norm = torch.sum(input_emb, dim=1)
    normalized = input_emb/norm
    score = cosine_similarity(normalized, target_emb)
    return normalized, score

```

将所有这些组件放在一起，我们终于准备好运行Skip-Gram模型。我们略过这部分代码，因为它与我们过去构建模型的方式非常相似。唯一的区别是在检查步骤中的额外代码。我们从我们的词汇表中最常见的500个单词中随机选择20个验证单词。对于这些单词中的每一个，我们使用我们构建的余弦相似度函数找到最近的邻居：

```py
n_epochs=1
for epoch in range(n_epochs):
  # Train
  running_loss = 0.0
  for inputs, targets in trainloader:
    loss = train(inputs, targets)
    running_loss += loss.item()

  writer.add_scalar('Train Loss',
                    running_loss/len(trainloader), epoch)
  #Validate
  running_score = 0.0
  for inputs, targets in valloader:
    _, score = evaluate(inputs, targets)
    running_score += score

  writer.add_scalar('Val Score',
                    running_score/len(valloader), epoch)

```

代码开始运行，我们可以开始看到模型随时间的演变。在开始时，模型在嵌入方面表现不佳（从检查步骤中可以看出）。然而，当训练完成时，模型显然已经找到了有效捕捉单词含义的表示：

```py
ancient: egyptian, cultures, mythology, civilization, etruscan, 
greek, classical, preserved

however: but, argued, necessarily, suggest, certainly, nor, 
believe, believed

type: typical, kind, subset, form, combination, single, 
description, meant

white: yellow, black, red, blue, colors, grey, bright, dark

system: operating, systems, unix, component, variant, versions, 
version, essentially

energy: kinetic, amount, heat, gravitational, nucleus, 
radiation, particles, transfer

world: ii, tournament, match, greatest, war, ever, championship, 
cold

y: z, x, n, p, f, variable, mathrm, sum,

line: lines, ball, straight, circle, facing, edge, goal, yards,

among: amongst, prominent, most, while, famous, particularly, 
argue, many

image: png, jpg, width, images, gallery, aloe, gif, angel

kingdom: states, turkey, britain, nations, islands, namely, 
ireland, rest

long: short, narrow, thousand, just, extended, span, length, 
shorter

through: into, passing, behind, capture, across, when, apart, 
goal

i: you, t, know, really, me, want, myself, we

source: essential, implementation, important, software, content, 
genetic, alcohol, application

because: thus, while, possibility, consequently, furthermore, 
but, certainly, moral

eight: six, seven, five, nine, one, four, three, b

french: spanish, jacques, pierre, dutch, italian, du, english, 
belgian

written: translated, inspired, poetry, alphabet, hebrew, 
letters, words, read

```

虽然不完美，但这里捕捉到了一些引人注目的有意义的聚类。数字、国家和文化被紧密聚集在一起。代词“I”与其他代词聚集在一起。单词“world”有趣地接近“championship”和“war”。而单词“written”被发现与“translated”、“poetry”、“alphabet”、“letters”和“words”相似。

最后，我们通过在[图8-21](#viz_of_skip_gram_embeddings)中可视化我们的词嵌入来结束本节。为了在2D空间中显示我们的128维嵌入，我们将使用一种称为t-SNE的可视化方法。如果您还记得，我们在[第7章](ch07.xhtml#convolutional_neural_networks)中也使用t-SNE来可视化ImageNet中图像之间的关系。使用t-SNE非常简单，因为它在常用的机器学习库scikit-learn中有一个内置函数。

我们可以使用以下代码构建可视化：

```py
tsne = TSNE(perplexity=30, n_components=2, init='pca', 
            n_iter=5000)
plot_embeddings = np.asfarray(final_embeddings[:plot_num,:], 
                              dtype='float')
low_dim_embs = tsne.fit_transform(plot_embeddings)
labels = [reverse_dictionary[i] for i in xrange(plot_only)]
data.plot_with_labels(low_dim_embs, labels)

```

在[图8-21](#viz_of_skip_gram_embeddings)中，我们注意到相似的概念比不同的概念更接近，表明我们的嵌入对于单词的功能和定义编码了有意义的信息。

![](Images/fdl2_0821.png)

###### 图8-21。使用t-SNE的Skip-Gram嵌入

有关单词嵌入的属性和有趣模式（动词时态、国家和首都、类比完成等）的更详细探讨，我们建议您参考原始的Mikolov等人的论文。

# 摘要

在本章中，我们探讨了表示学习中的各种方法。我们了解了如何使用自动编码器进行有效的降维。我们还学习了去噪和稀疏性，这些增强了自动编码器的有用属性。在讨论完自动编码器后，我们将注意力转向当输入的上下文比输入本身更具信息性时的表示学习。我们学习了如何使用Skip-Gram模型为英语单词生成嵌入，这将在我们探索用于理解语言的深度学习模型时非常有用。在下一章中，我们将在此基础上分析语言和其他序列使用深度学习。

^([1](ch08.xhtml#idm45934168575888-marker)) Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. “使用神经网络降低数据的维度。”*科学*313.5786（2006）：504-507。

^([2](ch08.xhtml#idm45934167915920-marker)) Vincent, Pascal, et al. “使用去噪自动编码器提取和组合稳健特征。”*第25届国际机器学习会议论文集*。ACM，2008年。

^([3](ch08.xhtml#idm45934167906880-marker)) Bengio, Yoshua, et al. “广义去噪自动编码器作为生成模型。”*神经信息处理系统进展*。2013年。

^([4](ch08.xhtml#idm45934167858016-marker)) Ranzato, Marc’Aurelio, et al. “使用基于能量的模型高效学习稀疏表示。”*第19届神经信息处理系统国际会议论文集*。MIT出版社，2006年。

^([5](ch08.xhtml#idm45934167856400-marker)) Ranzato, Marc’Aurelio, and Martin Szummer. “半监督学习中的紧凑文档表示与深度网络。”*第25届国际机器学习会议论文集*。ACM，2008年。

^([6](ch08.xhtml#idm45934167853584-marker)) Makhzani, Alireza, and Brendan Frey. “k-稀疏自动编码器。”*arXiv预印本arXiv*：1312.5663（2013）。

^([7](ch08.xhtml#idm45934167802832-marker)) Mikolov, Tomas, et al. “单词和短语的分布式表示及其组合性。”*神经信息处理系统进展*。2013年。

^([8](ch08.xhtml#idm45934167799008-marker)) Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. “在向量空间中高效估计单词表示。”*ICLR研讨会*，2013年。
