# 第十一章\. 深度神经网络的训练

在第十章中，您使用 PyTorch 构建、训练和微调了几个人工神经网络。但它们只是具有几个隐藏层的浅层网络。如果您需要解决一个复杂的问题，比如在高分辨率图像中检测数百种类型的对象，您可能需要训练一个深度更大的 ANN，可能包含数十甚至数百层，每层包含数百个神经元，通过数十万个连接相互连接。训练深度神经网络并非易事。以下是一些您可能会遇到的问题：

+   在训练过程中，您可能会遇到在反向传播通过深度神经网络（DNN）时梯度不断变小或变大的问题。这两个问题都使得训练低层变得非常困难。

+   您可能没有足够的训练数据来支持这样一个大的网络，或者标注成本可能过高。

+   训练可能非常缓慢。

+   具有数百万个参数的模型可能会严重过拟合训练集，尤其是如果没有足够的训练实例或者它们太嘈杂的情况下。

在本章中，我们将逐一探讨这些问题，并介绍各种解决方法。我们将首先探讨梯度消失和爆炸问题及其最流行的解决方案，包括智能权重初始化、更好的激活函数、批归一化、层归一化和梯度裁剪。接下来，我们将探讨迁移学习和无监督预训练，这可以帮助您在少量标记数据的情况下处理复杂任务。然后，我们将讨论各种优化器，它们可以极大地加速大型模型的训练。我们还将讨论如何在训练过程中调整学习率以加快训练速度并产生更好的模型。最后，我们将介绍适用于大型神经网络的几种流行正则化技术：ℓ[1]和ℓ[2]正则化、dropout、蒙特卡洛 dropout 和最大范数正则化。

使用这些工具，您将能够训练各种深度网络。欢迎来到*深度*学习！

# 梯度消失/爆炸问题

如第九章所述，反向传播算法的第二阶段是通过从输出层到输入层，沿途传播误差梯度来工作的。一旦算法计算了网络中每个参数相对于成本函数梯度的值，它就会使用这些梯度通过梯度下降步骤来更新每个参数。

不幸的是，随着算法向底层推进，梯度往往会越来越小。因此，梯度下降更新几乎不会改变底层连接权重，训练永远不会收敛到一个好的解。这被称为*梯度消失*问题。在某些情况下，情况可能相反：梯度会越来越大，直到层得到极端巨大的权重更新，算法发散。这被称为*梯度爆炸*问题，在循环神经网络（见第十三章）中最为常见。更普遍地说，深度神经网络受到不稳定梯度的困扰；不同的层可能以非常不同的速度学习。

这种不幸的行为很久以前就被观察到了，这也是深度神经网络在 2000 年代初大部分被放弃的原因之一。当训练深度神经网络时，不清楚是什么导致了梯度如此不稳定，但在 2010 年的一篇论文（https://homl.info/47）中，Xavier Glorot 和 Yoshua Bengio 提供了一些线索。⁠^(1) 作者发现了一些嫌疑人，包括当时最受欢迎的 sigmoid（逻辑）激活函数和最流行的权重初始化技术（即均值为 0，标准差为 1 的正态分布）。简而言之，他们表明，使用这种激活函数和这种初始化方案，每一层的输出方差远大于其输入的方差。在网络中向前推进时，方差在每一层之后都会增加，直到激活函数在顶层饱和。这种饱和实际上因为 sigmoid 函数的均值为 0.5 而不是 0（双曲正切函数的均值为 0，在深度网络中表现略好于 sigmoid 函数）而变得更糟。

观察 sigmoid 激活函数（见图 11-1），你可以看到当输入变得很大（负数或正数）时，函数在 0 或 1 处饱和，导数非常接近 0（即曲线在两端都是平的）。因此，当反向传播开始时，它几乎没有任何梯度可以传播回网络，而存在的少量梯度在反向传播通过顶层的过程中不断被稀释，所以实际上留给底层的东西很少。

![说明 sigmoid 激活函数的图，展示了它在大负数输入时饱和于 0，在大正数输入时饱和于 1，中间有一个线性区域。](img/hmls_1101.png)

###### 图 11-1\. Sigmoid 激活函数饱和

## Glorot 初始化和 He 初始化

在他们的论文中，Glorot 和 Bengio 提出了一种显著缓解不稳定梯度问题的方法。他们指出，我们需要信号在两个方向上正确流动：在预测时的前向方向，以及在反向传播梯度时的反向方向。我们不希望信号消失，也不希望它爆炸并饱和。为了使信号正确流动，作者们认为我们需要每一层的输出方差等于其输入方差，⁠^(2) 并且在反向方向通过一层之前和之后，梯度需要具有相同的方差（如果你对数学细节感兴趣，请查看论文）。实际上，除非层有相等数量的输入和输出（这些数字被称为层的 *fan-in* 和 *fan-out*），否则不可能保证两者都成立。但 Glorot 和 Bengio 提出了一种在实践中证明非常有效的良好折衷方案：每一层的连接权重必须随机初始化，如 Equation 11-1 中所述，其中 *fan*[avg] = (*fan*[in] + *fan*[out]) / 2。这种初始化策略被称为 *Xavier initialization* 或 *Glorot initialization*，以论文的第一作者命名。

##### 方程式 11-1\. Glorot initialization（使用 sigmoid 激活函数时）

<mtable columnalign="left"><mtr><mtd><mtext>均值为 0 且方差为 </mtext><msup><mi>σ</mi><mn>2</mn></msup><mo>=</mo><mfrac><mn>1</mn><msub><mi mathvariant="italic">fan</mi><mtext>avg</mtext></msub></mfrac></mtd></mtr><mtr><mtd><mtext>或者在一个均匀分布之间 </mtext><mo>-</mo><mi>r</mi><mtext> 和 </mtext><mo>+</mo><mi>r</mi><mtext>，其中 </mtext><mi>r</mi><mo>=</mo><msqrt><mfrac><mn>3</mn><msub><mi mathvariant="italic">fan</mi><mtext>avg</mtext></msub></mfrac></msqrt></mtd></mtr></mtable>

如果你将 Equation 11-1 中的 *fan*[avg] 替换为 *fan*[in]，你将得到一种 Yann LeCun 在 20 世纪 90 年代提出的初始化策略。他称之为 *LeCun initialization*。Genevieve Orr 和 Klaus-Robert Müller 甚至在他们的 1998 年书籍 *Neural Networks: Tricks of the Trade* (Springer) 中推荐了它。当 *fan*[in] = *fan*[out] 时，LeCun initialization 等同于 Glorot initialization。研究人员花了十多年时间才意识到这个技巧的重要性。使用 Glorot initialization 可以显著加快训练速度，这是导致深度学习成功的关键技巧之一。

一些论文为不同的激活函数提供了类似的策略，最著名的是 Kaiming He 等人于 2015 年发表的一篇论文[2015 paper by Kaiming He et al](https://homl.info/48)。⁠^(3) 这些策略仅在方差缩放的比例以及是否使用*fan*[avg]或*fan*[in]上有所不同，如表 11-1 所示（对于均匀分布，只需使用<mi>r</mi><mo>=</mo><msqrt><mn>3</mn><msup><mi>σ</mi><mn>2</mn></msup></msqrt>）。为 ReLU 激活函数及其变体提出的初始化策略被称为*He 初始化*或*Kaiming 初始化*，以论文的第一作者命名。对于 SELU，使用 Yann LeCun 的初始化方法，最好使用正态分布。我们将在稍后介绍所有这些激活函数。

表 11-1\. 每种激活函数的初始化参数

| 初始化 | 激活函数 | *σ*² (正态分布) |
| --- | --- | --- |
| Xavier Glorot | None, tanh, sigmoid, softmax | 1 / *fan*[avg] |
| Kaiming He | ReLU, Leaky ReLU, ELU, GELU, Swish, Mish, SwiGLU, ReLU² | 2 / *fan*[in] |
| Yann LeCun | SELU | 1 / *fan*[in] |

由于历史原因，PyTorch 的`nn.Linear`模块使用 Kaiming 均匀初始化来初始化其权重，但权重会乘以一个因子$StartRoot 6 EndRoot$（并且偏置项也会随机初始化）。遗憾的是，这并不是任何常见激活函数的最佳缩放比例。一种解决方案是在创建`nn.Linear`层后立即将权重乘以$StartRoot 6 EndRoot$（即 6 的 0.5 次方），以获得适当的 Kaiming 初始化。为此，我们可以更新参数的`data`属性。我们还将清零偏置，因为没有随机初始化它们的任何好处：

```py
import torch
import torch.nn as nn

layer = nn.Linear(40, 10)
layer.weight.data *= 6 ** 0.5  # Kaiming init (or 3 ** 0.5 for LeCun init)
torch.zero_(layer.bias.data)
```

这方法可行，但使用`torch.nn.init`模块中可用的初始化函数会更清晰且错误更少：

```py
nn.init.kaiming_uniform_(layer.weight)
nn.init.zeros_(layer.bias)
```

如果你想将相同的初始化方法应用到模型中每个`nn.Linear`层的权重上，你可以在创建每个`nn.Linear`层之后在模型的构造函数中这样做。或者，你可以编写`nn.Linear`类的子类，并调整其构造函数以按需初始化权重。但可能最简单的方法是编写一个函数，该函数接受一个模块，检查它是否是`nn.Linear`类的实例，如果是，则对其权重应用所需的初始化函数。然后，你可以通过将其传递给模型的`apply()`方法来将此函数应用到模型及其所有子模块。例如：

```py
def use_he_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        nn.init.zeros_(module.bias)

model = nn.Sequential(nn.Linear(50, 40), nn.ReLU(), nn.Linear(40, 1), nn.ReLU())
model.apply(use_he_init)
```

`torch.nn.init`模块还包含一个`orthogonal_()`函数，该函数使用随机正交矩阵初始化权重，正如 Andrew Saxe 等人在 2014 年发表的[论文](https://homl.info/ortho-init)中提出的。正交矩阵具有许多有用的数学性质，包括它们保持范数的事实：给定一个正交矩阵**W**和一个输入向量**x**，**Wx**的范数等于**x**的范数，因此输入的幅度在输出中得到了保留。当输入被标准化时，这会在层中产生稳定的方差，从而防止激活和梯度在深度网络中消失或爆炸（至少在训练的初期）。这种初始化技术比前面讨论的初始化技术要少见得多，但它可以在循环神经网络（第十三章）或生成对抗网络（第十八章）中工作得很好。

就这样！适当地缩放权重将为深度神经网络提供一个更好的训练起点。

###### 小贴士

在分类器中，在初始化过程中降低输出层的权重通常是一个好主意（例如，乘以 10）。事实上，这将导致训练开始时的 logits 更小，这意味着它们将更接近，因此估计的概率也将更接近。换句话说，这鼓励模型在训练开始时对其预测不太自信：这将避免极端损失和巨大的梯度，这通常会导致模型在训练开始时权重随机跳动，浪费时间和可能阻止模型学习任何东西。

## 更好的激活函数

Glorot 和 Bengio 在 2010 年的论文中的一个洞见是，不稳定梯度的部分问题源于激活函数的选择不当。在此之前，大多数人假设如果大自然选择在生物神经元中使用接近 sigmoid 激活函数的东西，那么它们肯定是一个很好的选择。但事实表明，其他激活函数在深度神经网络中表现得更好——特别是 ReLU 激活函数，主要是因为它对于正值不会饱和，而且它计算速度非常快。

很不幸，ReLU 激活函数并不完美。它存在一个被称为*渐死 ReLU*的问题：在训练过程中，一些神经元实际上“死亡”，意味着它们停止输出除了 0 以外的任何内容。在某些情况下，你可能发现你网络中一半的神经元都“死亡”了，尤其是如果你使用了较大的学习率。当一个神经元的权重被调整到使得 ReLU 函数的输入（即神经元输入的加权和加上其偏置项）对于训练集中的所有实例都是负值时，神经元就会“死亡”。当这种情况发生时，它只会持续输出 0，因为当 ReLU 函数的输入为负值时，其梯度为 0，所以梯度下降不再影响它。⁠^(6)

为了解决这个问题，你可能想使用 ReLU 函数的一个变体，比如*Leaky ReLU*。

### Leaky ReLU

Leaky ReLU 激活函数定义为 LeakyReLU*α* = max(*αz*, *z*)（见图 11-2）。超参数*α*定义了函数“泄漏”的程度：它是函数在*z* < 0 时的斜率。对于*z* < 0 有斜率确保 Leaky ReLU 永远不会真正“死亡”；它们可以进入长时间的昏迷，但最终有苏醒的机会。Bing Xu 等人于 2015 年发表的一篇[论文](https://homl.info/49)⁠^(7)比较了几种 ReLU 激活函数的变体，其结论之一是 Leaky ReLU 变体总是优于严格的 ReLU 激活函数。实际上，将*α*设置为 0.2（巨大的泄漏）似乎比*α* = 0.01（小的泄漏）有更好的性能。该论文还评估了*随机 Leaky ReLU*（RReLU），其中*α*在训练期间随机选择一个范围内的值，并在测试期间固定为平均值。RReLU 也表现相当不错，似乎起到了正则化的作用，减少了过拟合的风险。最后，该论文评估了*参数 Leaky ReLU*（PReLU），其中*α*在训练期间被允许学习：它不再是超参数，而成为一个可以像其他任何参数一样通过反向传播进行修改的参数。PReLU 据报道在大型图像数据集上显著优于 ReLU，但在较小的数据集上存在过拟合训练集的风险。

![Leaky ReLU 激活函数的示意图，说明了其负值的斜率，展示了“泄漏”现象。](img/hmls_1102.png)

###### 图 11-2\. Leaky ReLU：与 ReLU 类似，但负值具有较小的斜率

如您所预期，PyTorch 包含了这些激活函数各自的模块：`nn.LeakyReLU`、`nn.RReLU` 和 `nn.PReLU`。就像其他 ReLU 变体一样，您应该与 Kaiming 初始化一起使用这些模块，但由于负斜率，方差应略小一些：它应该通过一个因子 1 + *α*² 缩放。PyTorch 支持：您可以将 *α* 超参数传递给 `kaiming_uniform_()` 和 `kaiming_normal_()` 函数，以及 `nonlinearity="leaky_relu"` 以获得适当的调整后的 Kaiming 初始化：

```py
alpha = 0.2
model = nn.Sequential(nn.Linear(50, 40), nn.LeakyReLU(negative_slope=alpha))
nn.init.kaiming_uniform_(model[0].weight, alpha, nonlinearity="leaky_relu")
```

ReLU、leaky ReLU 和 PReLU 都存在它们不是平滑函数的事实：它们的斜率在 *z* = 0 处突然改变。正如我们在 第四章 中讨论 lasso 时所看到的，这种导数中的不连续性可以使梯度下降在最优解周围弹跳并减慢收敛速度。因此，现在我们将查看 ReLU 激活函数的一些平滑变体，从 ELU 和 SELU 开始。

### ELU 和 SELU

2015 年，Djork-Arné Clevert 等人发表的一篇 [论文](https://homl.info/50)⁠^(8) 提出了一种新的激活函数，称为 *指数线性单元*（ELU），在作者们的实验中优于所有 ReLU 变体：训练时间减少，神经网络在测试集上的表现更好。方程 11-2 展示了此激活函数的定义。

##### 方程 11-2\. ELU 激活函数

<mrow><msub><mo form="prefix">ELU</mo> <mi>α</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>=</mo> <mfenced separators="" open="{" close=""><mtable><mtr><mtd columnalign="left"><mrow><mi>α</mi> <mo>(</mo> <mtext>exp</mtext> <mo>(</mo> <mi>z</mi> <mo>)</mo> <mo>-</mo> <mn>1</mn> <mo>)</mo></mrow></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mi>z</mi> <mo><</mo> <mn>0</mn></mrow></mtd></mtr> <mtr><mtd columnalign="left"><mi>z</mi></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mi>z</mi> <mo>≥</mo> <mn>0</mn></mrow></mtd></mtr></mtable></mfenced></mrow>

ELU 激活函数在许多方面与 ReLU 函数相似（参见 图 11-3），但有几个主要区别：

+   当 *z* < 0 时，函数取负值，这使得单元的平均输出更接近 0，有助于缓解梯度消失问题。超参数 *α* 定义了当 *z* 是一个很大的负数时，ELU 函数趋近的值的相反数。它通常设置为 1，但您可以像调整任何其他超参数一样调整它。

+   当 *z* < 0 时，它具有非零梯度，这避免了死亡神经元问题。

+   当 *α* 等于 1 时，函数在所有地方都是平滑的，包括在 *z* = 0 附近，这有助于加快梯度下降，因为它不会在 *z* = 0 的左右弹跳得那么厉害。

使用 ELU 与 PyTorch 一样简单，只需使用 `nn.ELU` 模块，以及 Kaiming 初始化。ELU 激活函数的主要缺点是它比 ReLU 函数及其变体计算速度慢（由于使用了指数函数）。它在训练期间更快的收敛速度可能可以弥补这种慢速计算，但仍然，在测试时间一个 ELU 网络将比一个 ReLU 网络慢一点。

![比较 ELU 和 SELU 激活函数的图表，显示 SELU 的缩放值高于 ELU。](img/hmls_1103.png)

###### 图 11-3\. ELU 和 SELU 激活函数

不久之后，Günter Klambauer 等人在 2017 年发表的一篇[论文](https://homl.info/selu)⁠^(9)中介绍了 *缩放 ELU* (SELU) 激活函数：正如其名称所暗示的，它是 ELU 激活函数的一个缩放变体（大约是 ELU 的 1.05 倍，使用 *α* ≈ 1.67）。作者表明，如果你构建一个由堆叠的密集层（即 MLP）组成的神经网络，并且如果所有隐藏层都使用 SELU 激活函数，那么网络将 *自我归一化*：每个层的输出在训练过程中将倾向于保持均值为 0 和标准差为 1，这解决了梯度消失/爆炸问题。因此，SELU 激活函数可能比其他激活函数更适合 MLP，尤其是深度 MLP。要在 PyTorch 中使用它，只需使用 `nn.SELU`。然而，自我归一化发生有几个条件（参见论文中的数学证明）：

+   输入特征必须标准化：均值为 0，标准差为 1。

+   每个隐藏层的权重都必须使用 LeCun 正态初始化。

+   自我归一化属性仅在简单的 MLP 中得到保证。如果你尝试在其他架构中使用 SELU，如循环网络（参见第十三章）或具有 *跳跃连接*（即跳过层的连接，例如在 Wide & Deep 神经网络中），它可能不会优于 ELU。

+   你不能使用正则化技术，如 ℓ[1] 或 ℓ[2] 正则化、批归一化、层归一化、最大归一化或常规 dropout（这些将在本章后面讨论）。

这些是重要的约束条件，因此尽管 SELU 有其承诺，但它并没有获得太多的关注。此外，其他激活函数似乎在大多数任务上都能相当一致地优于它。让我们看看其中一些最受欢迎的。

### GELU, Swish, SwiGLU, Mish, 和 RELU²

*高斯误差线性单元* (*GELU*) 是由 Dan Hendrycks 和 Kevin Gimpel 在一篇[2016 年的论文](https://homl.info/gelu)中引入的.^(10) 再次，你可以将其视为 ReLU 激活函数的一个平滑变体。其定义在 方程 11-3 中给出，其中 Φ 是标准高斯累积分布函数 (CDF)：Φ(*z*) 对应于从均值为 0 和方差 1 的正态分布中随机抽取的值小于 *z* 的概率。

##### 方程 11-3\. GELU 激活函数

<mrow><mi>GELU</mi><mo>(</mo><mi>z</mi><mo>)</mo></mrow><mo>=</mo><mi>z</mi><mi mathvariant="normal">Φ</mi><mo>(</mo><mi>z</mi><mo>)</mo>

如您在图 11-4 中看到的，GELU 与 ReLU 相似：当其输入*z*非常负时，它接近 0；当*z*非常正时，它接近*z*。然而，与我们之前讨论的所有激活函数都是凸性和单调的相反，GELU 激活函数既不是：从左到右，它开始是直线，然后波动下降，在约-0.17（z ≈ -0.75）处达到低点，最后弹起并最终直线向上。这种相当复杂的形状以及它在每个点都有曲率的事实可能解释了为什么它工作得如此之好，尤其是在复杂任务中：梯度下降可能更容易拟合复杂模式。在实践中，它通常优于之前讨论的任何其他激活函数。然而，它计算上稍微复杂一些，它提供的性能提升并不总是足以证明额外成本是合理的。尽管如此，可以证明它大约等于*z*σ(1.702*z*)，其中σ是 sigmoid 函数：使用这个近似也工作得很好，并且它具有计算速度更快的优势。

![比较 GELU、Swish、参数化 Swish、Mish 和 ReLU²激活函数的图表，展示了它们在输入值上的行为差异。](img/hmls_1104.png)

###### 图 11-4\. GELU、Swish、参数化 Swish、Mish 和 ReLU²激活函数

GELU 论文还介绍了*sigmoid 线性单元*（SiLU）激活函数，它等于*z*σ(*z*)，但在作者们的测试中输给了 GELU。有趣的是，Prajit Ramachandran 等人于 2017 年发表的一篇[论文](https://homl.info/swish)通过自动搜索好的激活函数重新发现了 SiLU 函数。作者们将其命名为*Swish*，这个名字流行起来。在他们的论文中，Swish 优于所有其他函数，包括 GELU。Ramachandran 等人后来通过添加一个额外的标量超参数*β*来缩放 sigmoid 函数的输入，推广了 Swish。广义 Swish 函数是 Swish*β* = *z*σ(*βz*)，因此 GELU 大约等于使用*β* = 1.702 的广义 Swish 函数。您可以像调整任何其他超参数一样调整*β*。或者，也可以使*β*可训练，并让梯度下降优化它（有点像 PReLU）：通常整个模型只有一个可训练的*β*参数，或者每层只有一个，以保持模型高效并避免过拟合。

一种流行的 Swish 变体是 [*SwiGLU*](https://homl.info/swiglu):⁠^(13) 输入通过 Swish 激活函数，同时并行通过一个线性层，然后逐项相乘输出。这就是 SwiGLU(**z**) = Swish*β* ⊗ Linear(**z**)。这通常通过将前一个线性层的输出维度加倍，然后沿着特征维度将输出分成两部分以获得 **z**[1] 和 **z**[2]，最后应用：SwiGLU*β* = Swish*β* ⊗ **z**[2]。这是 Facebook 研究人员在 2016 年引入的 [*门控线性单元* (GLU)](https://homl.info/glu)⁠^(14) 的一个变体。逐项乘法给模型提供了更多的表达能力，允许它学习何时关闭（即乘以 0）或放大特定特征：这被称为 *门控机制*。SwiGLU 在现代变压器中非常常见（参见 第十五章）。

另一种类似于 GELU 的激活函数是 *Mish*，它由 Diganta Misra 在 [2019 年的一篇论文](https://homl.info/mish)中提出.^(15) 它被定义为 mish(*z*) = *z*tanh(softplus(*z*))，其中 softplus(*z*) = log(1 + exp(*z*))。就像 GELU 和 Swish 一样，它是一个平滑、非凸、非单调的 ReLU 变体，而且作者再次进行了许多实验，发现 Mish 通常优于其他激活函数——甚至比 Swish 和 GELU 略胜一筹。![图 11-4](img/#gelu_swish_mish_plot) 展示了 GELU、Swish（默认 *β* = 1 和 *β* = 0.6）以及最后的 Mish。如图所示，当 *z* 为负值时，Mish 几乎完美地与 Swish 重叠，而当 *z* 为正值时，几乎完美地与 GELU 重叠。

最后，在 2021 年，Google 研究人员运行了一个自动化的架构搜索来改进大型变压器，搜索发现了一个非常简单但有效的激活函数：[ReLU²](https://homl.info/relu2)。⁠^(16) 如其名所示，它只是 ReLU 的平方：ReLU²(*z*) = (max(0, *z*))²。它具有 ReLU 的所有特性（简单性、计算效率、稀疏输出、正侧无饱和）但它也有在 *z* = 0 处的平滑梯度，并且通常优于其他激活函数，特别是对于稀疏模型。然而，训练可能不太稳定，部分原因是因为它对异常值和死亡 ReLUs 的敏感性增加。

###### 小贴士

那么，你应该为你的深度神经网络隐藏层使用哪种激活函数呢？ReLU 对于大多数任务来说仍然是一个好的默认选择：它通常和更复杂的激活函数一样好，而且计算速度非常快，许多库和硬件加速器都提供了 ReLU 特定的优化。然而，对于复杂任务，Swish 可能是一个更好的默认选择，你甚至可以尝试带有可学习*β*参数的参数化 Swish，用于最复杂的任务。Mish 和 SwiGLU 可能会给你带来略微更好的结果，但它们需要更多的计算。如果你非常关心运行时延迟，那么你可能更喜欢漏斗 ReLU，或者对于复杂任务，选择参数化漏斗 ReLU，甚至 ReLU²，特别是对于稀疏模型。

PyTorch 原生支持 GELU、Mish 和 Swish（分别使用`nn.GELU`、`nn.Mish`和`nn.SiLU`）。要实现 SwiGLU，将前一个线性层的输出维度加倍，然后使用`z1, z2 = z.chunk(2, dim=-1)`将其输出分成两部分，并计算`F.silu(beta * z1) * z2`（其中`F`是`torch.nn.functional`）。对于 ReLU²，只需计算`F.relu(z).square()`。PyTorch 还包括几个激活函数的简化和近似版本，这些版本计算速度更快，在训练过程中通常更稳定。这些简化版本的名字以“Hard”开头，例如`nn.Hardsigmoid`、`nn.Hardtanh`和`nn.Hardswish`，它们通常用于移动设备。

关于激活函数就到这里！现在，让我们看看解决梯度不稳定问题的另一种完全不同的方法：批标准化。

## 批标准化

虽然使用 Kaiming 初始化与 ReLU（或其任何变体）结合可以在训练初期显著减少梯度消失/爆炸问题的风险，但这并不能保证它们不会在训练过程中再次出现。

在 2015 年的一篇论文[2015 paper](https://homl.info/51)，⁠^(17)中，Sergey Ioffe 和 Christian Szegedy 提出了一种称为*批标准化*（BN）的技术，用于解决这些问题。这项技术包括在每个隐藏层的激活函数之前或之后添加一个操作。这个操作简单地将每个输入归零并标准化，然后使用每个层的两个新参数向量来缩放和移动结果：一个用于缩放，另一个用于移动。换句话说，这个操作让模型学习每个层输入的最佳缩放和均值。在许多情况下，如果你将 BN 层作为你的神经网络的第一层，你就不需要标准化你的训练集（不需要`StandardScaler`）；BN 层会为你完成这个任务（好吧，大约是这样，因为它一次只查看一个批次，它还可以重新缩放和移动每个输入特征）。

为了将输入归零并标准化，算法需要估计每个输入的均值和标准差。它是通过评估当前小批次的输入均值和标准差来做到这一点的（因此得名“批标准化”）。整个操作步骤在公式 11-4 中逐步总结。

##### 公式 11-4\. 批标准化算法

<mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mn>1</mn> <mo lspace="0%" rspace="0%">.</mo></mrow></mtd> <mtd columnalign="left"><mrow><msub><mi mathvariant="bold">μ</mi> <mi>B</mi></msub> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>1</mn> <msub><mi>m</mi> <mi>B</mi></msub></mfrac></mstyle> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <msub><mi>m</mi> <mi>B</mi></msub></munderover> <msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>2</mn> <mo lspace="0%" rspace="0%">.</mo></mrow></mtd> <mtd columnalign="left"><mrow><msup><mrow><msub><mi mathvariant="bold">σ</mi> <mi>B</mi></msub></mrow> <mn>2</mn></msup> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>1</mn> <msub><mi>m</mi> <mi>B</mi></msub></mfrac></mstyle> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <msub><mi>m</mi> <mi>B</mi></msub></munderover> <msup><mrow><mo>(</mo><msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>-</mo><msub><mi mathvariant="bold">μ</mi> <mi>B</mi></msub> <mo>)</mo></mrow> <mn>2</mn></msup></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>3</mn> <mo lspace="0%" rspace="0%">.</mo></mrow></mtd> <mtd columnalign="left"><mrow><msup><mover accent="true"><mi mathvariant="bold">x</mi> <mo>^</mo></mover> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mrow><msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>-</mo><msub><mi mathvariant="bold">μ</mi> <mi>B</mi></msub></mrow> <msqrt><mrow><msup><mrow><msub><mi mathvariant="bold">σ</mi> <mi>B</mi></msub></mrow> <mn>2</mn></msup> <mo>+</mo><mi>ε</mi></mrow></msqrt></mfrac></mstyle></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>4</mn> <mo lspace="0%" rspace="0%">.</mo></mrow></mtd> <mtd columnalign="left"><mrow><msup><mi mathvariant="bold">z</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>=</mo> <mi mathvariant="bold">γ</mi> <mo>⊗</mo> <msup><mover accent="true"><mi mathvariant="bold">x</mi> <mo>^</mo></mover> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>+</mo> <mi mathvariant="bold">β</mi></mrow></mtd></mtr></mtable>

在此算法中：

+   **μ**[*B*]是整个小批次*B*中输入均值的向量（它包含每个输入的一个均值）。

+   *m*[*B*]是小批次中实例的数量。

+   **x**^((*i*)) 是批归一化层实例*i*的输入向量。

+   **σ**[*B*]是输入标准差的向量，在整个小批量上评估（它包含每个输入的一个标准差）。

+   <mover accent="true"><mi mathvariant="bold">x</mi> <mo>^</mo></mover> ^((*i*)) 是实例*i*的零均值和归一化输入向量。

+   *ε*是一个很小的数，它避免了除以零并确保梯度不会变得太大（通常是 10^(–5)）。这被称为*平滑项*。

+   **γ**是层的输出缩放参数向量（它包含每个输入的一个缩放参数）。

+   ⊗ 表示逐元素乘法（每个输入乘以其相应的输出缩放参数）。

+   **β**是层的输出偏移（偏移）参数向量（它包含每个输入的一个偏移参数）。每个输入都通过其相应的偏移参数进行偏移。

+   **z**^((*i*)) 是 BN 操作的输出。它是输入的缩放和偏移版本。

因此，在训练期间，BN 标准化其输入，然后缩放和偏移它们。很好！测试时间呢？嗯，并不简单。实际上，我们可能需要为单个实例而不是实例批次进行预测：在这种情况下，我们将无法计算每个输入的标准差。此外，即使我们有实例批次，它可能太小，或者实例可能不是独立同分布的，因此计算批次实例的统计量将不可靠。一种解决方案是在训练结束时等待，然后将整个训练集通过神经网络运行，并计算 BN 层每个输入的均值和标准差。然后可以使用这些“最终”输入均值和标准差来代替预测时使用的批输入均值和标准差。

然而，大多数批归一化的实现都是通过使用层的批输入均值和方差的移动平均来估计这些最终统计量的。这就是当你使用 PyTorch 的批归一化层，例如`nn.BatchNorm1d`（我们将在下一节中讨论）时，PyTorch 自动执行的操作。总的来说，每个批归一化层中学习到四个参数向量：**γ**（输出缩放向量）和**β**（输出偏移向量）通过常规的反向传播学习，而**μ**（最终输入均值向量）和**σ**²（最终输入方差向量）使用指数移动平均来估计。请注意，**μ**和**σ**²是在训练期间估计的，但它们仅在训练完成后使用，一旦你使用`model.eval()`将模型切换到评估模式：**μ**和**σ**²将替换方程 11-4 中的**μ**[*B*]和**σ**[*B*]²。

Ioffe 和 Szegedy 证明了批量归一化显著提高了他们实验中所有深度神经网络的表现，这在 ImageNet 分类任务上带来了巨大的改进（ImageNet 是一个包含许多类别的图像的大型数据库，常用于评估计算机视觉系统）。梯度消失问题得到了显著减少，以至于他们可以使用饱和激活函数如 tanh 和 sigmoid。网络对权重初始化的敏感性也大大降低。作者能够使用更大的学习率，显著加快学习过程。具体来说，他们指出：

> 将批量归一化应用于最先进的图像分类模型，批量归一化在 14 倍更少的训练步骤下达到了相同的准确率，并且显著优于原始模型。[……] 使用批量归一化的网络集成，我们在 ImageNet 分类任务上超越了已发表的最好结果：达到 4.9%的 top-5 验证错误率（和 4.8%的测试错误率），超过了人类评分员的准确率。

最后，就像源源不断的礼物一样，批量归一化（batch norm）充当正则化器的作用，减少了其他正则化技术（如后续章节中描述的 dropout）的需求。

然而，批量归一化确实给模型增加了一些复杂性（尽管它可以消除前面讨论中提到的对输入数据进行归一化的需求）。此外，还存在运行时惩罚：由于在每个层中需要额外的计算，神经网络预测速度变慢。幸运的是，通常在训练后可以将 BN 层与前一层的权重和偏差融合，从而避免运行时惩罚。这是通过更新前一层的权重和偏差，使其直接产生适当规模和偏移的输出来实现的。例如，如果前一层的计算是**XW** + **b**，那么 BN 层将计算**γ** ⊗ (**XW** + **b** – **μ**) / **σ** + **β**（忽略分母中的平滑项*ε*）。如果我们定义**W**′ = **γ**⊗**W** / **σ**和**b**′ = **γ** ⊗ (**b** – **μ**) / **σ** + **β**，则方程简化为**XW**′ + **b**′。因此，如果我们用更新的权重和偏差（**W**′和**b**′）替换前一层的权重和偏差（**W**和**b**），我们就可以去掉 BN 层。这是`optimize_for_inference()`（见第十章）执行的一种优化。 

###### 注意

你可能会发现训练相当慢，因为使用批量归一化时，每个 epoch 所需的时间要多得多。然而，这通常会被 BN 带来的更快收敛所抵消，因此达到相同性能所需的 epoch 会更少。总的来说，*wall time*（墙上的时钟测量时间）通常会更短。

### 使用 PyTorch 实现批量归一化

与 PyTorch 中的大多数事情一样，实现批量归一化（batch norm）既简单又直观。只需在每个隐藏层的激活函数之前或之后添加一个`nn.BatchNorm1d`层，并指定每个 BN 层的输入数量。你还可以将 BN 层作为模型的第一层，这样可以省去手动标准化输入的需求。例如，让我们创建一个 Fashion MNIST 图像分类器（类似于我们在第十章中构建的），将 BN 作为模型的第一层（在展平输入图像之后），然后在每个隐藏层之后再次使用：

```py
model = nn.Sequential(
    nn.Flatten(),
    nn.BatchNorm1d(1 * 28 * 28),
    nn.Linear(1 * 28 * 28, 300),
    nn.ReLU(),
    nn.BatchNorm1d(300),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.BatchNorm1d(100),
    nn.Linear(100, 10)
)
```

你现在可以像在第十章中学到的那样正常训练模型，这就完成了！在这个只有两个隐藏层的微小示例中，批量归一化可能不会产生很大的影响，但对于更深层的网络，它可能带来巨大的差异。

###### 警告

由于批量归一化在训练和评估期间表现不同，因此在训练期间切换到训练模式（使用`model.train()`）以及在评估期间切换到评估模式（使用`model.eval()》）至关重要。忘记这样做是犯得最常见错误之一。

如果你查看第一个 BN 层的参数，你会发现两个：`weight`和`bias`，它们对应于方程 11-4 中的**γ**和**β**：

```py
>>> dict(model[1].named_parameters()).keys() `dict_keys(['weight', 'bias'])`
```

```py`` And if you look at the buffers of this same BN layer, you will find three: `run⁠ning_​mean`, `running_var`, and `num_batches_tracked`. The first two correspond to the running means **μ** and **σ**² discussed earlier, and `num_batches_tracked` simply counts the number of batches seen during training:    ``` >>> dict(model[1].named_buffers()).keys() `dict_keys(['running_mean', 'running_var', 'num_batches_tracked'])` ```py   ````The authors of the BN paper argued in favor of adding the BN layers before the activation functions, rather than after (as we just did). There is some debate about this, and it seems to depend on the task, so you can experiment with this to see which option works best on your dataset. If you move the BN layers before the activation functions, you can also remove the bias term from the previous `nn.Linear` layers by setting their `bias` hyperparameter to `False`. Indeed, a batch-norm layer already includes one bias term per input. You can also drop the first BN layer to avoid sandwiching the first hidden layer between two BN layers, but this means you should normalize the training set before training. The updated code looks like this:    ```py model = nn.Sequential(     nn.Flatten(),     nn.Linear(1 * 28 * 28, 300, bias=False),     nn.BatchNorm1d(300),     nn.ReLU(),     nn.Linear(300, 100, bias=False),     nn.BatchNorm1d(100),     nn.ReLU(),     nn.Linear(100, 10) ) ```    The `nn.BatchNorm1d` class has a few hyperparameters you can tweak. The defaults will usually be fine, but you may occasionally need to tweak the `momentum`. This hyperparameter is used by the `BatchNorm1d` layer when it updates the exponential moving averages; given a new value **v** (i.e., a new vector of input means or variances computed over the current batch), the layer updates the running average <mover accent="true"><mi mathvariant="bold">v</mi><mo>^</mo></mover> using the following equation:  <mrow><mover accent="true"><mi mathvariant="bold">v</mi><mo>^</mo></mover> <mo>←</mo> <mi mathvariant="bold">v</mi> <mo>×</mo> <mtext>momentum</mtext> <mo>+</mo> <mover accent="true"><mi mathvariant="bold">v</mi><mo>^</mo></mover> <mo>×</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <mtext>momentum</mtext> <mo>)</mo></mrow></mrow>  A good momentum value is typically close to 0; for example, 0.01 or 0.001\. You want more 0s for smaller mini-batches, and fewer for larger mini-batches. The default is 0.1, which is good for large batch sizes, but not great for small batch sizes such as 32 or 64.    ###### Warning    When people talk about `momentum` in the context of a running mean, they usually refer to the weight of the current running mean in the update equation. Sadly, for historical reasons, PyTorch uses the opposite meaning in the BN layers. However, other parts of PyTorch use the conventional meaning (e.g., in optimizers), so don’t get confused.```py` `````  ```py ``### Batch norm 1D, 2D, and 3D    In the previous examples, we flattened the input images before sending them through the first `nn.BatchNorm1d` layer. This is because an `nn.BatchNorm1d` layer works on batches of shape `[batch_size, num_features]` (just like the `nn.Linear` layer does), so you would get an error if you moved it before the `nn.Flatten` layer.    However, you could use an `nn.BatchNorm2d` layer before the `nn.Flatten` layer: indeed, it expects its inputs to be image batches of shape `[batch_size, channels, height, width]`, and it computes the batch mean and variance across both the batch dimension (dimension 0) and the spatial dimensions (dimensions 2 and 3). This means that all pixels in the same batch and channel get normalized using the same mean and variance: the `nn.BatchNorm2d` layer only has one weight per channel and one bias per channel (e.g., three weights and three bias terms for color images with three channels for red, green, and blue). This generally works better when dealing with image datasets.    There’s also an `nn.BatchNorm3d` layer which expects batches of shape `[batch_size, channels, depth, height, width]`: this is useful for datasets of 3D images, such as CT scans.    The `nn.BatchNorm1d` layer can also work on batches of sequences. The convention in PyTorch is to represent batches of sequences as 3D tensors of shape `[batch_size, sequence_length, num_features]`. For example, suppose you work on particle physics and you have a dataset of particle trajectories, where each trajectory is composed of a sequence of 100 points in 3D space, then a batch of 32 trajectories will have a shape of `[32, 100, 3]`. However, the `nn.BatchNorm1d` layer expects the shape to be `[batch_size, num_features, sequence_length]`, and it computes the batch mean and variance across the first and last dimensions to get one mean and variance per feature. So you must permute the last two dimensions of the data using `X.permute(0, 2, 1)` before letting it go through the `nn.BatchNorm1d` layer. We will discuss sequences further in Chapter 13.    Batch normalization has become one of the most-used layers in deep neural networks, especially deep convolutional neural networks discussed in Chapter 12, to the point that it is often omitted in the architecture diagrams: it is assumed that BN is added after every layer. That said, it is not perfect. In particular, the computed statistics for an instance are biased by the other samples in a batch, which may reduce performance (especially for small batch sizes). Moreover, BN struggles with some architectures, such as recurrent nets, as we will see in Chapter 13. For these reasons, batch-norm is more and more often replaced by layer-norm.`` ```  ```py`` ````## Layer Normalization    Layer normalization (LN) is very similar to batch norm, but instead of normalizing across the batch dimension, LN normalizes across the feature dimensions. This simple idea was introduced by Jimmy Lei Ba et al. in a [2016 paper](https://homl.info/layernorm),⁠^(18) and initially applied mostly to recurrent nets. However, in recent years it has been successfully applied to many other architectures, such as convolutional nets, transformers, diffusion nets, and more.    One advantage is that LN can compute the required statistics on the fly, at each time step, independently for each instance. This also means that it behaves the same way during training and testing (as opposed to BN), and it does not need to use exponential moving averages to estimate the feature statistics across all instances in the training set, like BN does. Lastly, LN learns a scale and an offset parameter for each input feature, just like BN does.    PyTorch includes an `nn.LayerNorm` module. To create an instance, you must simply indicate the size of the dimensions that you want to normalize over. These must be the last dimension(s) of the inputs. For example, if the inputs are batches of 100 × 200 RGB images of shape `[3, 100, 200]`, and you want to normalize each image over each of the three color channels separately, you would use the following `nn.LayerNorm` module:    ```py inputs = torch.randn(32, 3, 100, 200)  # a batch of random RGB images layer_norm = nn.LayerNorm([100, 200]) result = layer_norm(inputs)  # normalizes over the last two dimensions ```    The following code produces the same result:    ```py means = inputs.mean(dim=[2, 3], keepdim=True)  # shape: [32, 3, 1, 1] vars_ = inputs.var(dim=[2, 3], keepdim=True, unbiased=False)  # shape: same stds = torch.sqrt(vars_ + layer_norm.eps)  # eps is a smoothing term (1e-5) result = layer_norm.weight * (inputs - means) / stds + layer_norm.bias # result shape: [32, 3, 100, 200] ```    However, most computer vision architectures that use LN normalize over all channels at once. For this, you must include the size of the channel dimension when creating the `nn.LayerNorm` module:    ```py layer_norm = nn.LayerNorm([3, 100, 200]) result = layer_norm(inputs)  # normalizes over the last three dimensions ```    And that’s all there is to it! Now let’s look
