# 第11章。训练深度神经网络

在[第10章](ch10.html#ann_chapter)中，您构建、训练和微调了您的第一个人工神经网络。但它们是浅层网络，只有几个隐藏层。如果您需要解决一个复杂的问题，比如在高分辨率图像中检测数百种对象，您可能需要训练一个更深的人工神经网络，也许有10层或更多层，每一层包含数百个神经元，通过数十万个连接相连。训练深度神经网络并不是一件轻松的事情。以下是您可能遇到的一些问题：

+   在训练过程中，当反向传播通过DNN向后流动时，您可能会面临梯度变得越来越小或越来越大的问题。这两个问题都会使得较低层非常难以训练。

+   您可能没有足够的训练数据来训练这样一个庞大的网络，或者标记成本太高。

+   训练可能会非常缓慢。

+   一个拥有数百万参数的模型会严重增加过拟合训练集的风险，特别是如果训练实例不足或者太嘈杂。

在本章中，我们将逐个讨论这些问题，并提出解决方法。我们将首先探讨梯度消失和梯度爆炸问题以及它们最流行的解决方案。接下来，我们将看看迁移学习和无监督预训练，这可以帮助您解决复杂任务，即使您只有很少的标记数据。然后，我们将讨论各种优化器，可以极大地加快训练大型模型。最后，我们将介绍一些用于大型神经网络的流行正则化技术。

有了这些工具，您将能够训练非常深的网络。欢迎来到深度学习！

# 梯度消失/爆炸问题

正如在[第10章](ch10.html#ann_chapter)中讨论的那样，反向传播算法的第二阶段是从输出层到输入层，沿途传播错误梯度。一旦算法计算出网络中每个参数相对于成本函数的梯度，它就会使用这些梯度来更新每个参数，进行梯度下降步骤。

不幸的是，随着算法向下进行到更低的层，梯度通常会变得越来越小。结果是，梯度下降更新几乎不会改变较低层的连接权重，训练永远不会收敛到一个好的解决方案。这被称为*梯度消失*问题。在某些情况下，相反的情况可能发生：梯度会变得越来越大，直到层的权重更新变得非常大，算法发散。这是*梯度爆炸*问题，最常出现在递归神经网络中（参见[第15章](ch15.html#rnn_chapter)）。更一般地说，深度神经网络受到不稳定梯度的困扰；不同层可能以非常不同的速度学习。

或者在-r和+r之间的均匀分布，r = sqrt(3 / fan_avg)

在他们的论文中，Glorot和Bengio提出了一种显著减轻不稳定梯度问题的方法。他们指出，我们需要信号在两个方向上正确地流动：在前向方向进行预测时，以及在反向方向进行反向传播梯度时。我们不希望信号消失，也不希望它爆炸和饱和。为了使信号正确地流动，作者认为每一层的输出方差应该等于其输入方差，并且在反向方向通过一层之后，梯度在前后具有相等的方差（如果您对数学细节感兴趣，请查看论文）。实际上，除非层具有相等数量的输入和输出（这些数字称为层的*fan-in*和*fan-out*），否则不可能保证两者都相等，但Glorot和Bengio提出了一个在实践中被证明非常有效的良好折衷方案：每层的连接权重必须随机初始化，如[方程11-1](#xavier_initialization_equation)所述，其中*fan*[avg] = (*fan*[in] + *fan*[out]) / 2。这种初始化策略称为*Xavier初始化*或*Glorot初始化*，以论文的第一作者命名。

观察Sigmoid激活函数（参见[图11-1](#sigmoid_saturation_plot)），您会发现当输入变大（负或正）时，函数在0或1处饱和，导数非常接近0（即曲线在两个极端处平坦）。因此，当反向传播开始时，几乎没有梯度可以通过网络向后传播，存在的微小梯度会随着反向传播通过顶层逐渐稀释，因此对于较低层几乎没有剩余的梯度。

###### 图11-1。Sigmoid激活函数饱和

## Glorot和He初始化

这种不幸的行为早在很久以前就被经验性地观察到，这也是深度神经网络在2000年代初大多被放弃的原因之一。当训练DNN时，梯度不稳定的原因并不清楚，但在2010年的一篇论文中，Xavier Glorot和Yoshua Bengio揭示了一些端倪。作者发现了一些嫌疑人，包括当时最流行的Sigmoid（逻辑）激活函数和权重初始化技术的组合（即均值为0，标准差为1的正态分布）。简而言之，他们表明，使用这种激活函数和初始化方案，每一层的输出方差远大于其输入方差。在网络中前进，每一层的方差在每一层之后都会增加，直到激活函数在顶层饱和。实际上，这种饱和现象被sigmoid函数的均值为0.5而不是0所加剧（双曲正切函数的均值为0，在深度网络中的表现略好于sigmoid函数）。

##### 方程11-1。Glorot初始化（使用Sigmoid激活函数时）

正态分布，均值为0，方差为σ^2 = 1 / fan_avg

如果您在[方程式11-1](#xavier_initialization_equation)中用*fan*[in]替换*fan*[avg]，您将得到Yann LeCun在1990年代提出的初始化策略。他称之为*LeCun初始化*。Genevieve Orr和Klaus-Robert Müller甚至在他们1998年的书*Neural Networks: Tricks of the Trade*（Springer）中推荐了这种方法。当*fan*[in] = *fan*[out]时，LeCun初始化等同于Glorot初始化。研究人员花了十多年的时间才意识到这个技巧有多重要。使用Glorot初始化可以显著加快训练速度，这是深度学习成功的实践之一。

一些论文提供了不同激活函数的类似策略。这些策略仅在方差的规模和它们是否使用*fan*[avg]或*fan*[in]上有所不同，如[表11-1](#initialization_table)所示（对于均匀分布，只需使用<math><mi>r</mi><mo>=</mo><msqrt><mn>3</mn><msup><mi>σ</mi><mn>2</mn></msup></msqrt></math>）。为ReLU激活函数及其变体提出的初始化策略称为*He初始化*或*Kaiming初始化*，以[论文的第一作者](https://homl.info/48)命名。对于SELU，最好使用Yann LeCun的初始化方法，最好使用正态分布。我们将很快介绍所有这些激活函数。

表11-1。每种激活函数的初始化参数

| 初始化 | 激活函数 | *σ*²（正态） |
| --- | --- | --- |
| Glorot | 无，tanh，sigmoid，softmax | 1 / *fan*[avg] |
| He | ReLU，Leaky ReLU，ELU，GELU，Swish，Mish | 2 / *fan*[in] |
| LeCun | SELU | 1 / *fan*[in] |

默认情况下，Keras使用均匀分布的Glorot初始化。当您创建一个层时，您可以通过设置`kernel_initializer="he_uniform"`或`kernel_initializer="he_normal"`来切换到He初始化。

```py
import tensorflow as tf

dense = tf.keras.layers.Dense(50, activation="relu",
                              kernel_initializer="he_normal")
```

或者，您可以使用`VarianceScaling`初始化器获得[表11-1](#initialization_table)中列出的任何初始化方法，甚至更多。例如，如果您想要使用均匀分布并基于*fan*[avg]（而不是*fan*[in]）进行He初始化，您可以使用以下代码：

```py
he_avg_init = tf.keras.initializers.VarianceScaling(scale=2., mode="fan_avg",
                                                    distribution="uniform")
dense = tf.keras.layers.Dense(50, activation="sigmoid",
                              kernel_initializer=he_avg_init)
```

## 更好的激活函数

2010年Glorot和Bengio的一篇论文中的一个见解是，不稳定梯度的问题在一定程度上是由于激活函数的选择不当。直到那时，大多数人都认为，如果自然界选择在生物神经元中使用大致为S形的激活函数，那么它们一定是一个很好的选择。但事实证明，其他激活函数在深度神经网络中表现得更好，特别是ReLU激活函数，主要是因为它对于正值不会饱和，而且计算速度非常快。

不幸的是，ReLU激活函数并不完美。它存在一个称为*dying ReLUs*的问题：在训练过程中，一些神经元实际上“死亡”，意味着它们停止输出除0以外的任何值。在某些情况下，您可能会发现您网络的一半神经元已经死亡，尤其是如果您使用了较大的学习率。当神经元的权重被微调得使得ReLU函数的输入（即神经元输入的加权和加上偏置项）在训练集中的所有实例中都为负时，神经元就会死亡。当这种情况发生时，它只会继续输出零，并且梯度下降不再影响它，因为当其输入为负时，ReLU函数的梯度为零。

为了解决这个问题，您可能希望使用ReLU函数的变体，比如*leaky ReLU*。

### Leaky ReLU

leaky ReLU激活函数定义为LeakyReLU[*α*](*z*) = max(*αz*, *z*)（参见[图11-2](#leaky_relu_plot)）。超参数*α*定义了函数“泄漏”的程度：它是*z* < 0时函数的斜率。对于*z* < 0，具有斜率的leaky ReLU永远不会死亡；它们可能会陷入长时间的昏迷，但最终有机会苏醒。Bing Xu等人在2015年的一篇[论文](https://homl.info/49)比较了几种ReLU激活函数的变体，其中一个结论是，泄漏变体总是优于严格的ReLU激活函数。事实上，设置*α*=0.2（一个巨大的泄漏）似乎比*α*=0.01（一个小泄漏）表现更好。该论文还评估了*随机泄漏ReLU*（RReLU），其中*α*在训练期间在给定范围内随机选择，并在测试期间固定为平均值。RReLU表现也相当不错，并似乎作为正则化器，减少了过拟合训练集的风险。最后，该论文评估了*参数泄漏ReLU*（PReLU），其中*α*在训练期间被授权学习：它不再是一个超参数，而是一个可以像其他参数一样通过反向传播修改的参数。据报道，PReLU在大型图像数据集上明显优于ReLU，但在较小的数据集上存在过拟合训练集的风险。

![mls3 1102](assets/mls3_1102.png)

###### 图11-2\. Leaky ReLU：类似于ReLU，但对负值有一个小的斜率

Keras在`tf.keras.layers`包中包含了`LeakyReLU`和`PReLU`类。就像其他ReLU变体一样，您应该使用He初始化。例如：

```py
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)  # defaults to alpha=0.3
dense = tf.keras.layers.Dense(50, activation=leaky_relu,
                              kernel_initializer="he_normal")
```

如果您愿意，您也可以在模型中将`LeakyReLU`作为一个单独的层来使用；对于训练和预测没有任何影响：

```py
model = tf.keras.models.Sequential([
    [...]  # more layers
    tf.keras.layers.Dense(50, kernel_initializer="he_normal"),  # no activation
    tf.keras.layers.LeakyReLU(alpha=0.2),  # activation as a separate layer
    [...]  # more layers
])
```

对于PReLU，将`LeakyReLU`替换为`PReLU`。目前在Keras中没有官方实现RReLU，但您可以相当容易地实现自己的（要了解如何做到这一点，请参见[第12章](ch12.html#tensorflow_chapter)末尾的练习）。

ReLU、leaky ReLU和PReLU都存在一个问题，即它们不是平滑函数：它们的导数在*z*=0处突然变化。正如我们在[第4章](ch04.html#linear_models_chapter)中讨论lasso时看到的那样，这种不连续性会导致梯度下降在最优点周围反弹，并减慢收敛速度。因此，现在我们将看一些ReLU激活函数的平滑变体，从ELU和SELU开始。

### ELU和SELU

2015年，Djork-Arné Clevert等人提出了一篇[论文](https://homl.info/50)，提出了一种新的激活函数，称为*指数线性单元*（ELU），在作者的实验中表现优于所有ReLU变体：训练时间缩短，神经网络在测试集上表现更好。[方程式11-2](#elu_activation_equation)展示了这个激活函数的定义。

##### 方程式11-2\. ELU激活函数

<math display="block"><mrow><msub><mo form="prefix">ELU</mo> <mi>α</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>=</mo> <mfenced separators="" open="{" close=""><mtable><mtr><mtd columnalign="left"><mrow><mi>α</mi> <mo>(</mo> <mo form="prefix">exp</mo> <mo>(</mo> <mi>z</mi> <mo>)</mo> <mo>-</mo> <mn>1</mn> <mo>)</mo></mrow></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mi>z</mi> <mo><</mo> <mn>0</mn></mrow></mtd></mtr> <mtr><mtd columnalign="left"><mi>z</mi></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mi>z</mi> <mo>≥</mo> <mn>0</mn></mrow></mtd></mtr></mtable></mfenced></mrow></math>

ELU激活函数看起来很像ReLU函数（参见[图11-3](#elu_and_selu_activation_plot)），但有一些主要区别：

+   当*z* < 0时，它会取负值，这使得单元的平均输出更接近于0，并有助于缓解梯度消失问题。超参数*α*定义了当*z*是一个较大的负数时ELU函数接近的值的相反数。通常设置为1，但您可以像调整其他超参数一样进行调整。

+   在*z* < 0时具有非零梯度，避免了死神经元问题。

+   如果*α*等于1，则该函数在任何地方都是平滑的，包括在*z* = 0附近，这有助于加快梯度下降的速度，因为它在*z* = 0的左右两侧不会反弹太多。

在Keras中使用ELU就像设置`activation="elu"`一样简单，与其他ReLU变体一样，应该使用He初始化。ELU激活函数的主要缺点是它的计算速度比ReLU函数及其变体慢（由于使用了指数函数）。在训练期间更快的收敛速度可能会弥补这种缓慢的计算，但是在测试时，ELU网络将比ReLU网络慢一点。

![mls3 1103](assets/mls3_1103.png)

###### 图11-3\. ELU和SELU激活函数

不久之后，Günter Klambauer等人在[2017年的一篇论文](https://homl.info/selu)中介绍了*缩放ELU*（SELU）激活函数：正如其名称所示，它是ELU激活函数的缩放变体（大约是ELU的1.05倍，使用*α* ≈ 1.67）。作者们表明，如果构建一个仅由一堆稠密层（即MLP）组成的神经网络，并且所有隐藏层使用SELU激活函数，那么网络将*自标准化*：每一层的输出在训练过程中倾向于保持均值为0，标准差为1，从而解决了梯度消失/爆炸的问题。因此，SELU激活函数可能在MLP中胜过其他激活函数，尤其是深层网络。要在Keras中使用它，只需设置`activation="selu"`。然而，自标准化发生的条件有一些（请参阅论文进行数学证明）：

+   输入特征必须标准化：均值为0，标准差为1。

+   每个隐藏层的权重必须使用LeCun正态初始化。在Keras中，这意味着设置`kernel_initializer="lecun_normal"`。

+   只有在普通MLP中才能保证自标准化属性。如果尝试在其他架构中使用SELU，如循环网络（参见[第15章](ch15.html#rnn_chapter)）或具有*跳跃连接*（即跳过层的连接，例如在Wide & Deep网络中），它可能不会胜过ELU。

+   您不能使用正则化技术，如ℓ[1]或ℓ[2]正则化、最大范数、批量归一化或常规的dropout（这些将在本章后面讨论）。

这些是重要的限制条件，因此尽管SELU有所承诺，但并没有获得很大的关注。此外，另外三种激活函数似乎在大多数任务上表现出色：GELU、Swish和Mish。

### GELU、Swish和Mish

*GELU*是由Dan Hendrycks和Kevin Gimpel在[2016年的一篇论文](https://homl.info/gelu)中引入的。再次，您可以将其视为ReLU激活函数的平滑变体。其定义在[方程11-3](#gelu_activation_equation)中给出，其中Φ是标准高斯累积分布函数（CDF）：Φ(*z*)对应于从均值为0、方差为1的正态分布中随机抽取的值低于*z*的概率。

##### 方程11-3\. GELU激活函数

<math display="block"><mrow><mi>GELU</mi><mo>(</mo><mi>z</mi><mo>)</mo></mrow><mo>=</mo><mi>z</mi><mi mathvariant="normal">Φ</mi><mo>(</mo><mi>z</mi><mo>)</mo></math>

如您在[图11-4](#gelu_swish_mish_plot)中所见，GELU类似于ReLU：当其输入*z*非常负时，它接近0，当*z*非常正时，它接近*z*。然而，到目前为止我们讨论的所有激活函数都是凸函数且单调递增的，而GELU激活函数则不是：从左到右，它开始直线上升，然后下降，达到大约-0.17的低点（接近z≈-0.75），最后反弹上升并最终向右上方直线前进。这种相当复杂的形状以及它在每个点上都有曲率的事实可能解释了为什么它效果如此好，尤其是对于复杂任务：梯度下降可能更容易拟合复杂模式。在实践中，它通常优于迄今讨论的任何其他激活函数。然而，它的计算成本稍高，提供的性能提升并不总是足以证明额外成本的必要性。尽管如此，可以证明它大致等于*z*σ(1.702 *z*)，其中σ是sigmoid函数：使用这个近似也非常有效，并且计算速度更快。

![mls3 1104](assets/mls3_1104.png)

###### 图11-4. GELU、Swish、参数化Swish和Mish激活函数

GELU论文还介绍了*sigmoid linear unit*（SiLU）激活函数，它等于*z*σ(*z*)，但在作者的测试中被GELU表现得更好。有趣的是，Prajit Ramachandran等人在[2017年的一篇论文](https://homl.info/swish)中重新发现了SiLU函数，通过自动搜索好的激活函数。作者将其命名为*Swish*，这个名字很受欢迎。在他们的论文中，Swish表现优于其他所有函数，包括GELU。Ramachandran等人后来通过添加额外的超参数*β*来推广Swish，用于缩放sigmoid函数的输入。推广后的Swish函数为Swish[*β*](*z*) = *z*σ(*βz*)，因此GELU大致等于使用*β* = 1.702的推广Swish函数。您可以像调整其他超参数一样调整*β*。另外，也可以将*β*设置为可训练的，让梯度下降来优化它：这样可以使您的模型更加强大，但也会有过拟合数据的风险。

另一个相当相似的激活函数是*Mish*，它是由Diganta Misra在[2019年的一篇论文](https://homl.info/mish)中引入的。它被定义为mish(*z*) = *z*tanh(softplus(*z*))，其中softplus(*z*) = log(1 + exp(*z*))。就像GELU和Swish一样，它是ReLU的平滑、非凸、非单调变体，作者再次进行了许多实验，并发现Mish通常优于其他激活函数，甚至比Swish和GELU稍微好一点。[图11-4](#gelu_swish_mish_plot)展示了GELU、Swish（默认*β* = 1和*β* = 0.6）、最后是Mish。如您所见，当*z*为负时，Mish几乎完全重叠于Swish，当*z*为正时，几乎完全重叠于GELU。

###### 提示

那么，对于深度神经网络的隐藏层，你应该使用哪种激活函数？对于简单任务，ReLU仍然是一个很好的默认选择：它通常和更复杂的激活函数一样好，而且计算速度非常快，许多库和硬件加速器提供了ReLU特定的优化。然而，对于更复杂的任务，Swish可能是更好的默认选择，甚至可以尝试带有可学习*β*参数的参数化Swish来处理最复杂的任务。Mish可能会给出稍微更好的结果，但需要更多的计算。如果你非常关心运行时延迟，那么你可能更喜欢leaky ReLU，或者对于更复杂的任务，可以使用参数化leaky ReLU。对于深度MLP，可以尝试使用SELU，但一定要遵守之前列出的约束条件。如果你有多余的时间和计算能力，也可以使用交叉验证来评估其他激活函数。

Keras支持GELU和Swish，只需使用`activation="gelu"`或`activation="swish"`。然而，它目前不支持Mish或广义Swish激活函数（但请参阅[第12章](ch12.html#tensorflow_chapter)了解如何实现自己的激活函数和层）。

激活函数就介绍到这里！现在，让我们看一种完全不同的解决不稳定梯度问题的方法：批量归一化。

## 批量归一化

尽管使用He初始化与ReLU（或其任何变体）可以显著减少训练开始时梯度消失/爆炸问题的危险，但并不能保证它们在训练过程中不会再次出现。

在一篇[2015年的论文](https://homl.info/51)中，Sergey Ioffe和Christian Szegedy提出了一种称为*批量归一化*（BN）的技术，解决了这些问题。该技术包括在模型中在每个隐藏层的激活函数之前或之后添加一个操作。这个操作简单地将每个输入零中心化和归一化，然后使用每层两个新的参数向量进行缩放和移位：一个用于缩放，另一个用于移位。换句话说，该操作让模型学习每个层输入的最佳缩放和均值。在许多情况下，如果将BN层作为神经网络的第一层，您就不需要标准化训练集。也就是说，不需要`StandardScaler`或`Normalization`；BN层会为您完成（大致上，因为它一次只看一个批次，并且还可以重新缩放和移位每个输入特征）。

为了将输入零中心化和归一化，算法需要估计每个输入的均值和标准差。它通过评估当前小批量输入的均值和标准差来实现这一点（因此称为“批量归一化”）。整个操作在[方程式11-4](#batch_normalization_algorithm)中逐步总结。

##### 方程式11-4\. 批量归一化算法

<math display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mn>1</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><msub><mi mathvariant="bold">μ</mi> <mi>B</mi></msub> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>1</mn> <msub><mi>m</mi> <mi>B</mi></msub></mfrac></mstyle> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <msub><mi>m</mi> <mi>B</mi></msub></munderover> <msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>2</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><msup><mrow><msub><mi mathvariant="bold">σ</mi> <mi>B</mi></msub></mrow> <mn>2</mn></msup> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>1</mn> <msub><mi>m</mi> <mi>B</mi></msub></mfrac></mstyle> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <msub><mi>m</mi> <mi>B</mi></msub></munderover> <msup><mrow><mo>(</mo><msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>-</mo><msub><mi mathvariant="bold">μ</mi> <mi>B</mi></msub> <mo>)</mo></mrow> <mn>2</mn></msup></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>3</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><msup><mover accent="true"><mi mathvariant="bold">x</mi> <mo>^</mo></mover> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mrow><msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>-</mo><msub><mi mathvariant="bold">μ</mi> <mi>B</mi></msub></mrow> <msqrt><mrow><msup><mrow><msub><mi mathvariant="bold">σ</mi> <mi>B</mi></msub></mrow> <mn>2</mn></msup> <mo>+</mo><mi>ε</mi></mrow></msqrt></mfrac></mstyle></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>4</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><msup><mi mathvariant="bold">z</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>=</mo> <mi mathvariant="bold">γ</mi> <mo>⊗</mo> <msup><mover accent="true"><mi mathvariant="bold">x</mi> <mo>^</mo></mover> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>+</mo> <mi mathvariant="bold">β</mi></mrow></mtd></mtr></mtable></math>

在这个算法中：

+   **μ**[*B*] 是在整个小批量*B*上评估的输入均值向量（它包含每个输入的一个均值）。

+   *m*[*B*] 是小批量中实例的数量。

+   **σ**[*B*] 是输入标准差的向量，也是在整个小批量上评估的（它包含每个输入的一个标准差）。

+   <math><mover accent="true"><mi mathvariant="bold">x</mi> <mo>^</mo></mover></math> ^((*i*)) 是实例*i*的零中心化和归一化输入向量。

+   *ε* 是一个微小的数字，避免了除以零，并确保梯度不会增长太大（通常为10^（–5））。这被称为*平滑项*。

+   **γ** 是该层的输出比例参数向量（它包含每个输入的一个比例参数）。

+   ⊗ 表示逐元素乘法（每个输入都会乘以其对应的输出比例参数）。

+   **β** 是该层的输出偏移参数向量（它包含每个输入的一个偏移参数）。每个输入都会被其对应的偏移参数偏移。

+   **z**^((*i*)) 是BN操作的输出。它是输入的重新缩放和偏移版本。

因此，在训练期间，BN会标准化其输入，然后重新缩放和偏移它们。很好！那么，在测试时呢？嗯，事情并不那么简单。实际上，我们可能需要为单个实例而不是一批实例进行预测：在这种情况下，我们将无法计算每个输入的均值和标准差。此外，即使我们有一批实例，它可能太小，或者实例可能不是独立且同分布的，因此在批次实例上计算统计数据将是不可靠的。一个解决方案可能是等到训练结束，然后通过神经网络运行整个训练集，并计算BN层每个输入的均值和标准差。这些“最终”输入均值和标准差可以在进行预测时代替批次输入均值和标准差。然而，大多数批次归一化的实现在训练期间通过使用该层输入均值和标准差的指数移动平均值来估计这些最终统计数据。这就是当您使用`BatchNormalization`层时Keras自动执行的操作。总之，在每个批次归一化的层中学习了四个参数向量：**γ**（输出缩放向量）和**β**（输出偏移向量）通过常规反向传播学习，而**μ**（最终输入均值向量）和**σ**（最终输入标准差向量）则使用指数移动平均值进行估计。请注意，**μ**和**σ**是在训练期间估计的，但仅在训练后使用（以替换[公式11-4](#batch_normalization_algorithm)中的批次输入均值和标准差）。

Ioffe和Szegedy证明了批次归一化显著改善了他们进行实验的所有深度神经网络，从而在ImageNet分类任务中取得了巨大的改进（ImageNet是一个大型图像数据库，被分类为许多类别，通常用于评估计算机视觉系统）。梯度消失问题得到了很大程度的减轻，以至于他们可以使用饱和激活函数，如tanh甚至sigmoid激活函数。网络对权重初始化也不那么敏感。作者能够使用更大的学习率，显著加快学习过程。具体来说，他们指出：

> 应用于最先进的图像分类模型，批次归一化在14倍更少的训练步骤下实现了相同的准确性，并且以显著的优势击败了原始模型。[...] 使用一组批次归一化的网络，我们在ImageNet分类上取得了最佳发布结果：达到4.9%的前5验证错误率（和4.8%的测试错误率），超过了人类评分者的准确性。

最后，就像一份源源不断的礼物，批次归一化就像一个正则化器，减少了对其他正则化技术（如本章后面描述的dropout）的需求。

然而，批量归一化确实给模型增加了一些复杂性（尽管它可以消除对输入数据进行归一化的需要，如前面讨论的）。此外，还存在运行时惩罚：由于每一层需要额外的计算，神经网络的预测速度变慢。幸运的是，通常可以在训练后将BN层与前一层融合在一起，从而避免运行时惩罚。这是通过更新前一层的权重和偏置，使其直接产生适当规模和偏移的输出来实现的。例如，如果前一层计算**XW** + **b**，那么BN层将计算**γ** ⊗ (**XW** + **b** - **μ**) / **σ** + **β**（忽略分母中的平滑项*ε*）。如果我们定义**W**′ = **γ**⊗**W** / **σ**和**b**′ = **γ** ⊗ (**b** - **μ**) / **σ** + **β**，则方程简化为**XW**′ + **b**′。因此，如果我们用更新后的权重和偏置（**W**′和**b**′）替换前一层的权重和偏置（**W**和**b**），我们可以摆脱BN层（TFLite的转换器会自动执行此操作；请参阅[第19章](ch19.html#deployment_chapter)）。

###### 注意

您可能会发现训练速度相当慢，因为使用批量归一化时，每个时期需要更多的时间。通常，这通常会被BN的收敛速度更快所抵消，因此需要更少的时期才能达到相同的性能。总的来说，*墙上的时间*通常会更短（这是您墙上时钟上测量的时间）。

### 使用Keras实现批量归一化

与Keras的大多数事物一样，实现批量归一化是简单直观的。只需在每个隐藏层的激活函数之前或之后添加一个`BatchNormalization`层。您还可以将BN层添加为模型中的第一层，但通常在此位置使用普通的`Normalization`层效果一样好（它的唯一缺点是您必须首先调用其`adapt()`方法）。例如，这个模型在每个隐藏层后应用BN，并将其作为模型中的第一层（在展平输入图像之后）：

```py
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(300, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

就这样！在这个只有两个隐藏层的微小示例中，批量归一化不太可能产生很大的影响，但对于更深的网络，它可能产生巨大的差异。

让我们显示模型摘要：

```py
>>> model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
batch_normalization (BatchNo (None, 784)               3136
_________________________________________________________________
dense (Dense)                (None, 300)               235500
_________________________________________________________________
batch_normalization_1 (Batch (None, 300)               1200
_________________________________________________________________
dense_1 (Dense)              (None, 100)               30100
_________________________________________________________________
batch_normalization_2 (Batch (None, 100)               400
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1010
=================================================================
Total params: 271,346
Trainable params: 268,978
Non-trainable params: 2,368
_________________________________________________________________
```

正如您所看到的，每个BN层都会为每个输入添加四个参数：**γ**、**β**、**μ**和**σ**（例如，第一个BN层会添加3,136个参数，即4×784）。最后两个参数，**μ**和**σ**，是移动平均值；它们不受反向传播的影响，因此Keras将它们称为“不可训练”⁠^([13](ch11.html#idm45720198989872))（如果您计算BN参数的总数，3,136 + 1,200 + 400，然后除以2，您将得到2,368，这是该模型中不可训练参数的总数）。

让我们看看第一个BN层的参数。其中两个是可训练的（通过反向传播），另外两个不是：

```py
>>> [(var.name, var.trainable) for var in model.layers[1].variables]
[('batch_normalization/gamma:0', True),
 ('batch_normalization/beta:0', True),
 ('batch_normalization/moving_mean:0', False),
 ('batch_normalization/moving_variance:0', False)]
```

BN论文的作者主张在激活函数之前而不是之后添加BN层（就像我们刚刚做的那样）。关于这一点存在一些争论，因为哪种方式更可取似乎取决于任务-您也可以尝试这个来看看哪个选项在您的数据集上效果最好。要在激活函数之前添加BN层，您必须从隐藏层中删除激活函数，并在BN层之后作为单独的层添加它们。此外，由于批量归一化层包含每个输入的一个偏移参数，您可以在创建时通过传递`use_bias=False`来删除前一层的偏置项。最后，通常可以删除第一个BN层，以避免将第一个隐藏层夹在两个BN层之间。更新后的代码如下：

```py
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

`BatchNormalization`类有很多可以调整的超参数。默认值通常是可以的，但偶尔您可能需要调整`momentum`。当`BatchNormalization`层更新指数移动平均值时，该超参数将被使用；给定一个新值**v**（即，在当前批次上计算的新的输入均值或标准差向量），该层使用以下方程更新运行平均值<math><mover><mi mathvariant="bold">v</mi><mo>^</mo></mover></math>：

<math display="block"><mrow><mover accent="true"><mi mathvariant="bold">v</mi> <mo>^</mo></mover> <mo>←</mo> <mover accent="true"><mi mathvariant="bold">v</mi> <mo>^</mo></mover> <mo>×</mo> <mtext>momentum</mtext> <mo>+</mo> <mi mathvariant="bold">v</mi> <mo>×</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <mtext>momentum</mtext> <mo>)</mo></mrow></mrow></math>

一个良好的动量值通常接近于1；例如，0.9，0.99或0.999。对于更大的数据集和更小的小批量，您希望有更多的9。

另一个重要的超参数是`axis`：它确定应该对哪个轴进行归一化。默认为-1，这意味着默认情况下将归一化最后一个轴（使用在*其他*轴上计算的均值和标准差）。当输入批次为2D（即，批次形状为[*批次大小，特征*]）时，这意味着每个输入特征将基于在批次中所有实例上计算的均值和标准差进行归一化。例如，前面代码示例中的第一个BN层将独立地归一化（和重新缩放和移位）784个输入特征中的每一个。如果我们将第一个BN层移到`Flatten`层之前，那么输入批次将是3D，形状为[*批次大小，高度，宽度*]；因此，BN层将计算28个均值和28个标准差（每个像素列一个，跨批次中的所有实例和列中的所有行计算），并且将使用相同的均值和标准差归一化给定列中的所有像素。还将有28个比例参数和28个移位参数。如果您仍希望独立处理784个像素中的每一个，则应将`axis=[1, 2]`。

批量归一化已经成为深度神经网络中最常用的层之一，特别是在深度卷积神经网络中讨论的（[第14章](ch14.html#cnn_chapter)），以至于在架构图中通常被省略：假定在每一层之后都添加了BN。现在让我们看看最后一种稳定梯度的技术：梯度裁剪。

## 梯度裁剪

另一种缓解梯度爆炸问题的技术是在反向传播过程中裁剪梯度，使其永远不超过某个阈值。这被称为[*梯度裁剪*](https://homl.info/52)。⁠^([14](ch11.html#idm45720198748720)) 这种技术通常用于循环神经网络中，其中使用批量归一化是棘手的（正如您将在[第15章](ch15.html#rnn_chapter)中看到的）。

在Keras中，实现梯度裁剪只需要在创建优化器时设置`clipvalue`或`clipnorm`参数，就像这样：

```py
optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
model.compile([...], optimizer=optimizer)
```

这个优化器将梯度向量的每个分量剪切到-1.0和1.0之间的值。这意味着损失的所有偏导数（对每个可训练参数）将在-1.0和1.0之间被剪切。阈值是您可以调整的超参数。请注意，这可能会改变梯度向量的方向。例如，如果原始梯度向量是[0.9, 100.0]，它主要指向第二轴的方向；但是一旦您按值剪切它，您会得到[0.9, 1.0]，它大致指向两个轴之间的对角线。在实践中，这种方法效果很好。如果您希望确保梯度剪切不改变梯度向量的方向，您应该通过设置`clipnorm`而不是`clipvalue`来按范数剪切。如果其ℓ[2]范数大于您选择的阈值，则会剪切整个梯度。例如，如果设置`clipnorm=1.0`，那么向量[0.9, 100.0]将被剪切为[0.00899964, 0.9999595]，保持其方向但几乎消除第一个分量。如果您观察到梯度在训练过程中爆炸（您可以使用TensorBoard跟踪梯度的大小），您可能希望尝试按值剪切或按范数剪切，使用不同的阈值，看看哪个选项在验证集上表现最好。

# 重用预训练层

通常不建议从头开始训练一个非常大的DNN，而不是先尝试找到一个现有的神经网络，完成与您尝试解决的任务类似的任务（我将在[第14章](ch14.html#cnn_chapter)中讨论如何找到它们）。如果找到这样的神经网络，那么通常可以重用大部分层，除了顶部的层。这种技术称为*迁移学习*。它不仅会显著加快训练速度，而且需要的训练数据明显较少。

假设您可以访问一个经过训练的DNN，用于将图片分类为100个不同的类别，包括动物、植物、车辆和日常物品，现在您想要训练一个DNN来分类特定类型的车辆。这些任务非常相似，甚至部分重叠，因此您应该尝试重用第一个网络的部分（参见[图11-5](#reuse_pretrained_diagram)）。

###### 注意

如果您新任务的输入图片与原始任务中使用的图片大小不同，通常需要添加一个预处理步骤，将它们调整为原始模型期望的大小。更一般地说，当输入具有相似的低级特征时，迁移学习效果最好。

![mls3 1105](assets/mls3_1105.png)

###### 图11-5。重用预训练层

通常应该替换原始模型的输出层，因为它很可能对新任务没有用处，而且可能不会有正确数量的输出。

同样，原始模型的上层隐藏层不太可能像下层那样有用，因为对于新任务最有用的高级特征可能与对原始任务最有用的特征有很大不同。您需要找到要重用的正确层数。

###### 提示

任务越相似，您将希望重用的层次就越多（从较低层次开始）。对于非常相似的任务，尝试保留所有隐藏层，只替换输出层。

首先尝试冻结所有重用的层（即使它们的权重不可训练，以便梯度下降不会修改它们并保持固定），然后训练您的模型并查看其表现。然后尝试解冻顶部一两个隐藏层，让反向传播调整它们，看看性能是否提高。您拥有的训练数据越多，您可以解冻的层次就越多。解冻重用层时降低学习率也很有用：这将避免破坏它们微调的权重。

如果您仍然无法获得良好的性能，并且训练数据很少，尝试删除顶部隐藏层并再次冻结所有剩余的隐藏层。您可以迭代直到找到要重用的正确层数。如果您有大量训练数据，您可以尝试替换顶部隐藏层而不是删除它们，甚至添加更多隐藏层。

## 使用Keras进行迁移学习

让我们看一个例子。假设时尚MNIST数据集仅包含八个类别，例如除凉鞋和衬衫之外的所有类别。有人在该数据集上构建并训练了一个Keras模型，并获得了相当不错的性能（>90%的准确率）。我们将这个模型称为A。现在您想要解决一个不同的任务：您有T恤和套头衫的图像，并且想要训练一个二元分类器：对于T恤（和上衣）为正，对于凉鞋为负。您的数据集非常小；您只有200张带标签的图像。当您为这个任务训练一个新模型（我们称之为模型B），其架构与模型A相同时，您获得了91.85%的测试准确率。在喝早晨咖啡时，您意识到您的任务与任务A非常相似，因此也许迁移学习可以帮助？让我们找出来！

首先，您需要加载模型A并基于该模型的层创建一个新模型。您决定重用除输出层以外的所有层：

```py
[...]  # Assuming model A was already trained and saved to "my_model_A"
model_A = tf.keras.models.load_model("my_model_A")
model_B_on_A = tf.keras.Sequential(model_A.layers[:-1])
model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))
```

请注意，`model_A`和`model_B_on_A`现在共享一些层。当您训练`model_B_on_A`时，它也会影响`model_A`。如果您想避免这种情况，您需要在重用其层之前*克隆*`model_A`。为此，您可以使用`clone_model()`克隆模型A的架构，然后复制其权重：

```py
model_A_clone = tf.keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
```

###### 警告

`tf.keras.models.clone_model()`仅克隆架构，而不是权重。如果您不使用`set_weights()`手动复制它们，那么当首次使用克隆模型时，它们将被随机初始化。

现在您可以为任务B训练`model_B_on_A`，但由于新的输出层是随机初始化的，它将产生大误差（至少在最初的几个时期），因此会产生大误差梯度，可能会破坏重用的权重。为了避免这种情况，一种方法是在最初的几个时期内冻结重用的层，让新层有时间学习合理的权重。为此，将每个层的`trainable`属性设置为`False`并编译模型：

```py
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                     metrics=["accuracy"])
```

###### 注意

在冻结或解冻层之后，您必须始终编译您的模型。

现在您可以为模型训练几个时期，然后解冻重用的层（这需要重新编译模型）并继续训练以微调任务B的重用层。在解冻重用的层之后，通常最好降低学习率，再次避免损坏重用的权重。

```py
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                     metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_valid_B, y_valid_B))
```

那么，最终的结论是什么？好吧，这个模型的测试准确率为93.85%，比91.85%高出两个百分点！这意味着迁移学习将错误率减少了近25%：

```py
>>> model_B_on_A.evaluate(X_test_B, y_test_B)
[0.2546142041683197, 0.9384999871253967]
```

您相信了吗？您不应该相信：我作弊了！我尝试了许多配置，直到找到一个表现出强烈改进的配置。如果您尝试更改类别或随机种子，您会发现改进通常会下降，甚至消失或反转。我所做的被称为“折磨数据直到它招认”。当一篇论文看起来过于积极时，您应该持怀疑态度：也许这种花哨的新技术实际上并没有太大帮助（事实上，它甚至可能降低性能），但作者尝试了许多变体并仅报告了最佳结果（这可能仅仅是纯粹的运气），而没有提及他们在过程中遇到了多少失败。大多数情况下，这并不是恶意的，但这是科学中许多结果永远无法重现的原因之一。

为什么我作弊了？事实证明，迁移学习在小型密集网络上效果不佳，可能是因为小型网络学习的模式较少，而密集网络学习的是非常具体的模式，这些模式不太可能在其他任务中有用。迁移学习最适用于深度卷积神经网络，这些网络倾向于学习更通用的特征检测器（特别是在较低层）。我们将在[第14章](ch14.html#cnn_chapter)中重新讨论迁移学习，使用我们刚讨论的技术（这次不会作弊，我保证！）。

## 无监督预训练

假设您想要解决一个复杂的任务，但您没有太多标记的训练数据，而不幸的是，您找不到一个类似任务训练的模型。不要失去希望！首先，您应该尝试收集更多标记的训练数据，但如果您无法做到，您仍然可以执行*无监督预训练*（见[图11-6](#unsupervised_pretraining_diagram)）。事实上，收集未标记的训练示例通常很便宜，但标记它们却很昂贵。如果您可以收集大量未标记的训练数据，您可以尝试使用它们来训练一个无监督模型，例如自动编码器或生成对抗网络（GAN；见[第17章](ch17.html#autoencoders_chapter)）。然后，您可以重复使用自动编码器的较低层或GAN的鉴别器的较低层，添加顶部的输出层，然后使用监督学习（即使用标记的训练示例）微调最终网络。

正是这种技术在2006年由Geoffrey Hinton及其团队使用，导致了神经网络的复兴和深度学习的成功。直到2010年，无监督预训练（通常使用受限玻尔兹曼机（RBMs；请参阅[*https://homl.info/extra-anns*](https://homl.info/extra-anns)中的笔记本））是深度网络的标准，只有在消失梯度问题得到缓解后，纯粹使用监督学习训练DNN才变得更加普遍。无监督预训练（今天通常使用自动编码器或GAN，而不是RBMs）仍然是一个很好的选择，当您有一个复杂的任务需要解决，没有类似的可重用模型，但有大量未标记的训练数据时。

请注意，在深度学习的早期阶段，训练深度模型是困难的，因此人们会使用一种称为*贪婪逐层预训练*的技术（在[图11-6](#unsupervised_pretraining_diagram)中描述）。他们首先使用单层训练一个无监督模型，通常是一个RBM，然后冻结该层并在其顶部添加另一层，然后再次训练模型（实际上只是训练新层），然后冻结新层并在其顶部添加另一层，再次训练模型，依此类推。如今，事情简单得多：人们通常一次性训练完整的无监督模型，并使用自动编码器或GAN，而不是RBMs。

![mls3 1106](assets/mls3_1106.png)

###### 图11-6。在无监督训练中，模型使用无监督学习技术在所有数据上进行训练，包括未标记的数据，然后使用监督学习技术仅在标记的数据上对最终任务进行微调；无监督部分可以像这里所示一次训练一层，也可以直接训练整个模型

## 辅助任务上的预训练

如果您没有太多标记的训练数据，最后一个选择是在一个辅助任务上训练第一个神经网络，您可以轻松获取或生成标记的训练数据，然后重复使用该网络的较低层来执行实际任务。第一个神经网络的较低层将学习特征检测器，很可能可以被第二个神经网络重复使用。

例如，如果您想构建一个识别人脸的系统，您可能只有每个个体的少量图片，显然不足以训练一个良好的分类器。收集每个人数百张照片是不现实的。但是，您可以在网络上收集大量随机人的照片，并训练第一个神经网络来检测两张不同图片是否展示了同一个人。这样的网络将学习良好的人脸特征检测器，因此重用其较低层将允许您训练一个使用很少训练数据的良好人脸分类器。

对于自然语言处理（NLP）应用，您可以下载数百万个文本文档的语料库，并从中自动生成标记数据。例如，您可以随机屏蔽一些单词并训练模型来预测缺失的单词是什么（例如，它应该预测句子“What ___ you saying?”中缺失的单词可能是“are”或“were”）。如果您可以训练模型在这个任务上达到良好的性能，那么它将已经对语言有相当多的了解，您肯定可以在实际任务中重复使用它，并在标记数据上进行微调（我们将在[第15章](ch15.html#rnn_chapter)中讨论更多的预训练任务）。

###### 注意

*自监督学习*是指从数据本身自动生成标签，例如文本屏蔽示例，然后使用监督学习技术在生成的“标记”数据集上训练模型。

# 更快的优化器

训练一个非常庞大的深度神经网络可能会非常缓慢。到目前为止，我们已经看到了四种加速训练（并达到更好解决方案）的方法：应用良好的连接权重初始化策略，使用良好的激活函数，使用批量归一化，并重用预训练网络的部分（可能是为辅助任务构建的或使用无监督学习）。另一个巨大的加速来自使用比常规梯度下降优化器更快的优化器。在本节中，我们将介绍最流行的优化算法：动量、Nesterov加速梯度、AdaGrad、RMSProp，最后是Adam及其变体。

## 动量

想象一颗保龄球在光滑表面上缓坡滚动：它会从慢慢开始，但很快会积累动量，直到最终达到终端速度（如果有一些摩擦或空气阻力）。这就是*动量优化*的核心思想，[由鲍里斯·波利亚克在1964年提出](https://homl.info/54)。与此相反，常规梯度下降在坡度平缓时会采取小步骤，在坡度陡峭时会采取大步骤，但它永远不会加速。因此，与动量优化相比，常规梯度下降通常要慢得多才能达到最小值。

请记住，梯度下降通过直接减去成本函数*J*(**θ**)相对于权重的梯度（∇[**θ**]*J*(**θ**))乘以学习率*η*来更新权重**θ**。方程式为**θ** ← **θ** - *η*∇[**θ**]*J*(**θ**)。它不关心先前的梯度是什么。如果局部梯度很小，它会走得很慢。

动量优化非常关注先前梯度是什么：在每次迭代中，它从*动量向量* **m**（乘以学习率*η*）中减去局部梯度，然后通过添加这个动量向量来更新权重（参见[方程11-5](#momentum_equation)）。换句话说，梯度被用作加速度，而不是速度。为了模拟某种摩擦机制并防止动量增长过大，该算法引入了一个新的超参数*β*，称为*动量*，必须设置在0（高摩擦）和1（无摩擦）之间。典型的动量值为0.9。

##### 方程11-5. 动量算法

<math display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mn>1</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">m</mi> <mo>←</mo> <mi>β</mi> <mi mathvariant="bold">m</mi> <mo>-</mo> <mi>η</mi> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>2</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">θ</mi> <mo>←</mo> <mi mathvariant="bold">θ</mi> <mo>+</mo> <mi mathvariant="bold">m</mi></mrow></mtd></mtr></mtable></math>

您可以验证，如果梯度保持不变，则终端速度（即权重更新的最大大小）等于该梯度乘以学习率*η*乘以1 / (1 - *β*)（忽略符号）。例如，如果*β* = 0.9，则终端速度等于梯度乘以学习率的10倍，因此动量优化的速度比梯度下降快10倍！这使得动量优化比梯度下降更快地摆脱高原。我们在[第4章](ch04.html#linear_models_chapter)中看到，当输入具有非常不同的比例时，成本函数看起来像一个拉长的碗（参见[图4-7](ch04.html#elongated_bowl_diagram)）。梯度下降很快下降陡峭的斜坡，但然后需要很长时间才能下降到山谷。相比之下，动量优化将会越来越快地滚动到山谷，直到达到底部（最优解）。在不使用批量归一化的深度神经网络中，上层通常会出现具有非常不同比例的输入，因此使用动量优化会有很大帮助。它还可以帮助跳过局部最优解。

###### 注意

由于动量的原因，优化器可能会稍微超调，然后返回，再次超调，并在稳定在最小值之前多次振荡。这是有摩擦力的好处之一：它消除了这些振荡，从而加快了收敛速度。

在Keras中实现动量优化是一件轻而易举的事情：只需使用`SGD`优化器并设置其`momentum`超参数，然后躺下来赚钱！

```py
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
```

动量优化的一个缺点是它增加了另一个需要调整的超参数。然而，在实践中，动量值0.9通常效果很好，几乎总是比常规梯度下降更快。

## Nesterov加速梯度

动量优化的一个小变体，由[Yurii Nesterov于1983年提出](https://homl.info/55)，^([16](ch11.html#idm45720198190768))几乎总是比常规动量优化更快。*Nesterov加速梯度*（NAG）方法，也称为*Nesterov动量优化*，测量成本函数的梯度不是在本地位置**θ**处，而是稍微向前在动量方向，即**θ** + *β***m**（参见[方程11-6](#nesterov_momentum_equation)）。

##### 第11-6方程。Nesterov加速梯度算法

<math display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mn>1</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">m</mi> <mo>←</mo> <mi>β</mi> <mi mathvariant="bold">m</mi> <mo>-</mo> <mi>η</mi> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>+</mo> <mi>β</mi> <mi mathvariant="bold">m</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>2</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">θ</mi> <mo>←</mo> <mi mathvariant="bold">θ</mi> <mo>+</mo> <mi mathvariant="bold">m</mi></mrow></mtd></mtr></mtable></math>

这个小调整有效是因为通常动量向量将指向正确的方向（即朝向最优解），因此使用稍微更准确的梯度测量更有利于使用稍微更远处的梯度，而不是原始位置处的梯度，如您在[图11-7](#nesterov_momentum_diagram)中所见（其中∇[1]表示在起始点**θ**处测量的成本函数的梯度，而∇[2]表示在位于**θ** + *β***m**的点处测量的梯度）。

![mls3 1107](assets/mls3_1107.png)

###### 图11-7。常规与Nesterov动量优化：前者应用动量步骤之前计算的梯度，而后者应用动量步骤之后计算的梯度

如您所见，Nesterov更新最终更接近最优解。随着时间的推移，这些小的改进累积起来，NAG最终比常规动量优化快得多。此外，请注意，当动量将权重推过山谷时，∇[1]继续推动更远，而∇[2]则向山谷底部推回。这有助于减少振荡，因此NAG收敛更快。

要使用NAG，只需在创建`SGD`优化器时设置`nesterov=True`：

```py
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,
                                    nesterov=True)
```

## AdaGrad

考虑再次延长碗问题：梯度下降首先快速沿着最陡的斜坡下降，这并不直指全局最优解，然后它非常缓慢地下降到山谷底部。如果算法能够更早地纠正方向，使其更多地指向全局最优解，那将是很好的。[*AdaGrad*算法](https://homl.info/56)通过沿着最陡的维度缩小梯度向量来实现这种校正（参见[方程11-7](#adagrad_algorithm)）。

##### 方程11-7。AdaGrad算法

<math display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mn>1</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">s</mi> <mo>←</mo> <mi mathvariant="bold">s</mi> <mo>+</mo> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow> <mo>⊗</mo> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>2</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">θ</mi> <mo>←</mo> <mi mathvariant="bold">θ</mi> <mo>-</mo> <mi>η</mi> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow> <mo>⊘</mo> <msqrt><mrow><mi mathvariant="bold">s</mi> <mo>+</mo> <mi>ε</mi></mrow></msqrt></mrow></mtd></mtr></mtable></math>

第一步将梯度的平方累积到向量**s**中（请记住，⊗符号表示逐元素乘法）。这种向量化形式等同于计算*s*[i] ← *s*[i] + (∂*J*(**θ**)/∂*θ*[i])²，对于向量**s**的每个元素*s*[i]来说，换句话说，每个*s*[i]累积了成本函数对参数*θ*[i]的偏导数的平方。如果成本函数沿第*i*维陡峭，那么在每次迭代中*s*[i]将变得越来越大。

第二步几乎与梯度下降完全相同，但有一个重大区别：梯度向量被一个因子<math><msqrt><mrow><mi mathvariant="bold">s</mi><mo>+</mo><mi>ε</mi></mrow></msqrt></math>缩小（⊘符号表示逐元素除法，*ε*是一个平滑项，用于避免除以零，通常设置为10^(–10)）。这个向量化形式等价于同时计算所有参数*θ*[*i*]的<math><msub><mi>θ</mi><mi>i</mi></msub><mo>←</mo><msub><mi>θ</mi><mi>i</mi></msub><mo>-</mo><mi>η</mi><mo>∂</mo><mi>J</mi><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo><mo>/</mo><mo>∂</mo><msub><mi>θ</mi><mi>i</mi></msub><mo>/</mo><msqrt><msub><mi>s</mi><mi>i</mi></msub><mo>+</mo><mi>ε</mi></msqrt></math>。

简而言之，这个算法会衰减学习率，但对于陡峭的维度比对于坡度较缓的维度衰减得更快。这被称为*自适应学习率*。它有助于更直接地指向全局最优（参见[图11-8](#adagrad_diagram)）。另一个好处是它需要更少的调整学习率超参数*η*。

![mls3 1108](assets/mls3_1108.png)

###### 图11-8\. AdaGrad与梯度下降的比较：前者可以更早地纠正方向指向最优点

在简单的二次问题上，AdaGrad通常表现良好，但在训练神经网络时经常会过早停止：学习率被缩小得太多，以至于算法最终在达到全局最优之前完全停止。因此，即使Keras有一个`Adagrad`优化器，你也不应该用它来训练深度神经网络（尽管对于简单任务如线性回归可能是有效的）。不过，理解AdaGrad有助于理解其他自适应学习率优化器。

## RMSProp

正如我们所见，AdaGrad有减速得太快并且永远无法收敛到全局最优的风险。*RMSProp*算法⁠^([18](ch11.html#idm45720198007744))通过仅累积最近迭代的梯度来修复这个问题，而不是自训练开始以来的所有梯度。它通过在第一步中使用指数衰减来实现这一点（参见[方程11-8](#rmsprop_algorithm)）。

##### 方程11-8\. RMSProp算法

<math display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mn>1</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">s</mi> <mo>←</mo> <mi>ρ</mi> <mi mathvariant="bold">s</mi> <mo>+</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <mi>ρ</mi> <mo>)</mo></mrow> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow> <mo>⊗</mo> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>2</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">θ</mi> <mo>←</mo> <mi mathvariant="bold">θ</mi> <mo>-</mo> <mi>η</mi> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow> <mo>⊘</mo> <msqrt><mrow><mi mathvariant="bold">s</mi> <mo>+</mo> <mi>ε</mi></mrow></msqrt></mrow></mtd></mtr></mtable></math>

衰减率*ρ*通常设置为0.9。⁠^([19](ch11.html#idm45720197966960)) 是的，这又是一个新的超参数，但这个默认值通常效果很好，所以你可能根本不需要调整它。

正如你所期望的，Keras有一个`RMSprop`优化器：

```py
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
```

除了在非常简单的问题上，这个优化器几乎总是比AdaGrad表现得更好。事实上，直到Adam优化算法出现之前，它一直是许多研究人员首选的优化算法。

## 亚当

[*Adam*](https://homl.info/59)，代表*自适应矩估计*，结合了动量优化和RMSProp的思想：就像动量优化一样，它跟踪过去梯度的指数衰减平均值；就像RMSProp一样，它跟踪过去梯度的平方的指数衰减平均值（见[Equation 11-9](#adam_algorithm)）。这些是梯度的均值和（未居中）方差的估计。均值通常称为*第一时刻*，而方差通常称为*第二时刻*，因此算法的名称。

##### 方程11-9\. Adam算法

<math display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mn>1</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">m</mi> <mo>←</mo> <msub><mi>β</mi> <mn>1</mn></msub> <mi mathvariant="bold">m</mi> <mo>-</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>β</mi> <mn>1</mn></msub> <mo>)</mo></mrow> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>2</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">s</mi> <mo>←</mo> <msub><mi>β</mi> <mn>2</mn></msub> <mi mathvariant="bold">s</mi> <mo>+</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>β</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow> <mo>⊗</mo> <msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>3</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mover accent="true"><mi mathvariant="bold">m</mi><mo>^</mo></mover> <mo>←</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mi mathvariant="bold">m</mi> <mrow><mn>1</mn> <mo>-</mo> <msup><mrow><msub><mi>β</mi> <mn>1</mn></msub></mrow> <mi>t</mi></msup></mrow></mfrac></mstyle></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>4</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mover accent="true"><mi mathvariant="bold">s</mi><mo>^</mo></mover> <mo>←</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mi mathvariant="bold">s</mi> <mrow><mn>1</mn><mo>-</mo><msup><mrow><msub><mi>β</mi> <mn>2</mn></msub></mrow> <mi>t</mi></msup></mrow></mfrac></mstyle></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mn>5</mn> <mo>.</mo></mrow></mtd> <mtd columnalign="left"><mrow><mi mathvariant="bold">θ</mi> <mo>←</mo> <mi mathvariant="bold">θ</mi> <mo>+</mo> <mi>η</mi> <mover accent="true"><mi mathvariant="bold">m</mi><mo>^</mo></mover> <mo>⊘</mo> <msqrt><mrow><mover accent="true"><mi mathvariant="bold">s</mi><mo>^</mo></mover> <mo>+</mo> <mi>ε</mi></mrow></msqrt></mrow></mtd></mtr></mtable></math>

在这个方程中，*t*代表迭代次数（从1开始）。

如果只看步骤1、2和5，你会注意到Adam与动量优化和RMSProp的相似之处：*β*[1]对应于动量优化中的*β*，*β*[2]对应于RMSProp中的*ρ*。唯一的区别是步骤1计算的是指数衰减平均值而不是指数衰减和，但实际上这些是等价的，除了一个常数因子（衰减平均值只是衰减和的1 - *β*[1]倍）。步骤3和4有点技术细节：由于**m**和**s**初始化为0，在训练开始时它们会偏向于0，因此这两个步骤将有助于在训练开始时提升**m**和**s**。

动量衰减超参数*β*[1]通常初始化为0.9，而缩放衰减超参数*β*[2]通常初始化为0.999。与之前一样，平滑项*ε*通常初始化为一个非常小的数字，如10^(–7)。这些是`Adam`类的默认值。以下是如何在Keras中创建Adam优化器的方法：

```py
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                                     beta_2=0.999)
```

由于Adam是一种自适应学习率算法，类似于AdaGrad和RMSProp，它需要较少调整学习率超参数*η*。您通常可以使用默认值*η*=0.001，使得Adam比梯度下降更容易使用。

###### 提示

如果您开始感到对所有这些不同技术感到不知所措，并想知道如何为您的任务选择合适的技术，不用担心：本章末尾提供了一些实用指南。

最后，值得一提的是Adam的三个变体：AdaMax、Nadam和AdamW。

## AdaMax

Adam论文还介绍了AdaMax。请注意，在[方程式11-9](#adam_algorithm)的第2步中，Adam在**s**中累积梯度的平方（对于最近的梯度有更大的权重）。在第5步中，如果我们忽略*ε*和步骤3和4（这些都是技术细节），Adam通过**s**的平方根缩小参数更新。简而言之，Adam通过时间衰减梯度的ℓ[2]范数缩小参数更新（回想一下，ℓ[2]范数是平方和的平方根）。

AdaMax用ℓ[∞]范数（一种说法是最大值）替换了ℓ[2]范数。具体来说，它用<math><mi mathvariant="bold">s</mi><mo>←</mo><mpadded lspace="-1px"><mi>max</mi><mo>(</mo><msub><mi>β</mi><mn>2</mn></msub><mi mathvariant="bold">s</mi><mo>,</mo> <mo>abs(</mo><msub><mo mathvariant="bold">∇</mo><mi mathvariant="bold">θ</mi></msub><mi>J</mi><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo><mo>)</mo><mo>)</mo></mpadded></math>替换了方程式11-9的第2步，删除了第4步，在第5步中，它通过**s**的因子缩小梯度更新，**s**是时间衰减梯度的绝对值的最大值。

实际上，这使得AdaMax比Adam更稳定，但这确实取决于数据集，总体上Adam表现更好。因此，如果您在某些任务上遇到Adam的问题，这只是另一个您可以尝试的优化器。

## Nadam

Nadam优化是Adam优化加上Nesterov技巧，因此它通常会比Adam收敛速度稍快。在[介绍这种技术的研究报告](https://homl.info/nadam)中，研究员Timothy Dozat比较了许多不同的优化器在各种任务上的表现，发现Nadam通常优于Adam，但有时会被RMSProp超越。

## AdamW

[AdamW](https://homl.info/adamw)是Adam的一个变体，它集成了一种称为*权重衰减*的正则化技术。权重衰减通过将模型的权重在每次训练迭代中乘以一个衰减因子，如0.99，来减小权重的大小。这可能让您想起ℓ[2]正则化（在[第4章](ch04.html#linear_models_chapter)介绍），它也旨在保持权重较小，事实上，可以在数学上证明，当使用SGD时，ℓ[2]正则化等效于权重衰减。然而，当使用Adam或其变体时，ℓ[2]正则化和权重衰减*不*等效：实际上，将Adam与ℓ[2]正则化结合使用会导致模型通常不如SGD产生的模型泛化能力好。AdamW通过正确地将Adam与权重衰减结合来解决这个问题。

###### 警告

自适应优化方法（包括RMSProp、Adam、AdaMax、Nadam和AdamW优化）通常很好，快速收敛到一个好的解决方案。然而，阿希亚·C·威尔逊等人在一篇[2017年的论文](https://homl.info/60)中表明，它们可能导致在某些数据集上泛化能力较差的解决方案。因此，当您对模型的性能感到失望时，请尝试使用NAG：您的数据集可能只是对自适应梯度过敏。还要关注最新的研究，因为它发展迅速。

要在Keras中使用Nadam、AdaMax或AdamW，请将`tf.keras.optimizers.Adam`替换为`tf.keras.optimizers.Nadam`、`tf.keras.optimizers.Adamax`或`tf.keras.optimizers.experimental.AdamW`。对于AdamW，您可能需要调整`weight_decay`超参数。

到目前为止讨论的所有优化技术只依赖于*一阶偏导数*（*雅可比*）。优化文献中还包含基于*二阶偏导数*（*海森*，即雅可比的偏导数）的惊人算法。不幸的是，这些算法很难应用于深度神经网络，因为每个输出有*n*²个海森（其中*n*是参数的数量），而不是每个输出只有*n*个雅可比。由于DNN通常具有成千上万个参数甚至更多，第二阶优化算法通常甚至无法适应内存，即使能够适应，计算海森也太慢。

[表11-2](#optimizer_summary_table)比较了到目前为止我们讨论过的所有优化器（*是不好的，**是平均的，***是好的）。

表11-2。优化器比较

| 类 | 收敛速度 | 收敛质量 |
| --- | --- | --- |
| `SGD` | * | *** |
| `SGD(momentum=...)` | ** | *** |
| `SGD(momentum=..., nesterov=True)` | ** | *** |
| `Adagrad` | *** | *（过早停止） |
| `RMSprop` | *** | ** or *** |
| `Adam` | *** | ** or *** |
| `AdaMax` | *** | ** or *** |
| `Nadam` | *** | ** or *** |
| `AdamW` | *** | ** or *** |

# 学习率调度

找到一个好的学习率非常重要。如果设置得太高，训练可能会发散（如[“梯度下降”](ch04.html#gradientDescent4)中讨论的）。如果设置得太低，训练最终会收敛到最优解，但需要很长时间。如果设置得稍微偏高，它会在一开始就非常快地取得进展，但最终会围绕最优解打转，从未真正稳定下来。如果你的计算预算有限，你可能需要在训练收敛之前中断训练，得到一个次优解（参见[图11-9](#learning_schedule_diagram)）。

![mls3 1109](assets/mls3_1109.png)

###### 图11-9。不同学习率η的学习曲线

如[第10章](ch10.html#ann_chapter)中讨论的，您可以通过训练模型几百次，将学习率从一个非常小的值指数增加到一个非常大的值，然后查看学习曲线并选择一个略低于学习曲线开始迅速上升的学习率来找到一个好的学习率。然后，您可以重新初始化您的模型，并使用该学习率进行训练。

但是你可以比恒定学习率做得更好：如果你从一个较大的学习率开始，然后在训练停止快速取得进展时降低它，你可以比使用最佳恒定学习率更快地达到一个好的解。有许多不同的策略可以在训练过程中降低学习率。从一个低学习率开始，增加它，然后再次降低它也可能是有益的。这些策略被称为*学习计划*（我在[第4章](ch04.html#linear_models_chapter)中简要介绍了这个概念）。这些是最常用的学习计划：

*幂调度*

将学习率设置为迭代次数*t*的函数：*η*(*t*) = *η*[0] / (1 + *t*/*s*)^(*c*)。初始学习率*η*[0]，幂*c*（通常设置为1）和步长*s*是超参数。学习率在每一步下降。经过*s*步，学习率降至*η*[0]的一半。再经过*s*步，它降至*η*[0]的1/3，然后降至*η*[0]的1/4，然后*η*[0]的1/5，依此类推。正如您所看到的，这个调度首先快速下降，然后变得越来越慢。当然，幂调度需要调整*η*[0]和*s*（可能还有*c*）。

*指数调度*

将学习率设置为*η*(*t*) = *η*[0] 0.1^(*t/s*)。学习率将每*s*步逐渐降低10倍。虽然幂调度使学习率降低得越来越慢，指数调度则每*s*步将其降低10倍。

*分段常数调度*

在一些时期内使用恒定的学习率（例如，*η*[0] = 0.1，持续5个时期），然后在另一些时期内使用较小的学习率（例如，*η*[1] = 0.001，持续50个时期），依此类推。尽管这种解决方案可能效果很好，但需要调整以找出正确的学习率序列以及每个学习率使用的时间长度。

*性能调度*

每*N*步测量验证错误（就像提前停止一样），当错误停止下降时，将学习率降低*λ*倍。

*1cycle调度*

1cycle是由Leslie Smith在[2018年的一篇论文](https://homl.info/1cycle)中提出的。与其他方法相反，它从增加初始学习率*η*[0]开始，线性增长到训练中途的*η*[1]。然后在训练的第二半部分线性降低学习率至*η*[0]，最后几个时期通过几个数量级的降低率（仍然是线性）来完成。最大学习率*η*[1]是使用我们用来找到最佳学习率的相同方法选择的，初始学习率*η*[0]通常低10倍。当使用动量时，我们首先使用高动量（例如0.95），然后在训练的前半部分将其降低到较低的动量（例如0.85，线性），然后在训练的后半部分将其提高到最大值（例如0.95），最后几个时期使用该最大值。Smith进行了许多实验，表明这种方法通常能够显著加快训练速度并达到更好的性能。例如，在流行的CIFAR10图像数据集上，这种方法仅在100个时期内达到了91.9%的验证准确率，而通过标准方法（使用相同的神经网络架构）在800个时期内仅达到了90.3%的准确率。这一壮举被称为*超级收敛*。

Andrew Senior等人在[2013年的一篇论文](https://homl.info/63)中比较了使用动量优化训练深度神经网络进行语音识别时一些最流行的学习调度的性能。作者得出结论，在这种情况下，性能调度和指数调度表现良好。他们更青睐指数调度，因为它易于调整，并且收敛到最佳解稍快。他们还提到，它比性能调度更容易实现，但在Keras中，这两个选项都很容易。也就是说，1cycle方法似乎表现得更好。

在Keras中实现幂调度是最简单的选择——只需在创建优化器时设置`衰减`超参数：

```py
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-4)
```

`衰减`是*s*的倒数（将学习率除以一个单位所需的步数），Keras假设*c*等于1。

指数调度和分段调度也很简单。您首先需要定义一个函数，该函数接受当前epoch并返回学习率。例如，让我们实现指数调度：

```py
def exponential_decay_fn(epoch):
    return 0.01 * 0.1 ** (epoch / 20)
```

如果您不想硬编码 *η*[0] 和 *s*，您可以创建一个返回配置函数的函数：

```py
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
```

接下来，创建一个 `LearningRateScheduler` 回调，将调度函数传递给它，并将此回调传递给 `fit()` 方法：

```py
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train, y_train, [...], callbacks=[lr_scheduler])
```

`LearningRateScheduler` 将在每个epoch开始时更新优化器的 `learning_rate` 属性。通常每个epoch更新一次学习率就足够了，但是如果您希望更频繁地更新它，例如在每一步，您可以随时编写自己的回调（请参阅本章笔记本中“指数调度”部分的示例）。在每一步更新学习率可能有助于处理每个epoch中的许多步骤。或者，您可以使用 `tf.keras.​optimiz⁠ers.schedules` 方法，稍后会进行描述。

###### 提示

训练后，`history.history["lr"]` 可以让您访问训练过程中使用的学习率列表。

调度函数可以选择将当前学习率作为第二个参数。例如，以下调度函数将前一个学习率乘以0.1^(1/20)，这将导致相同的指数衰减（除了衰减现在从第0个epoch开始而不是第1个）：

```py
def exponential_decay_fn(epoch, lr):
    return lr * 0.1 ** (1 / 20)
```

这个实现依赖于优化器的初始学习率（与之前的实现相反），所以请确保适当设置它。

当您保存一个模型时，优化器及其学习率也会被保存。这意味着使用这个新的调度函数，您可以加载一个训练好的模型，并继续在离开的地方继续训练，没有问题。然而，如果您的调度函数使用 `epoch` 参数，情况就不那么简单了：epoch 不会被保存，并且每次调用 `fit()` 方法时都会被重置为 0。如果您要继续训练一个模型，这可能会导致一个非常大的学习率，这可能会损坏模型的权重。一个解决方案是手动设置 `fit()` 方法的 `initial_epoch` 参数，使 `epoch` 从正确的值开始。

对于分段常数调度，您可以使用以下类似的调度函数（与之前一样，如果您愿意，您可以定义一个更通用的函数；请参阅笔记本中“分段常数调度”部分的示例），然后创建一个带有此函数的 `LearningRateScheduler` 回调，并将其传递给 `fit()` 方法，就像对指数调度一样：

```py
def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
```

对于性能调度，请使用 `ReduceLROnPlateau` 回调。例如，如果您将以下回调传递给 `fit()` 方法，每当最佳验证损失连续五个epoch没有改善时，它将把学习率乘以0.5（还有其他选项可用；请查看文档以获取更多详细信息）：

```py
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
history = model.fit(X_train, y_train, [...], callbacks=[lr_scheduler])
```

最后，Keras提供了另一种实现学习率调度的方法：您可以使用 `tf.keras.​opti⁠mizers.schedules` 中可用的类之一定义一个调度学习率，然后将其传递给任何优化器。这种方法在每一步而不是每个epoch更新学习率。例如，以下是如何实现与我们之前定义的 `exponential_decay_fn()` 函数相同的指数调度：

```py
import math

batch_size = 32
n_epochs = 25
n_steps = n_epochs * math.ceil(len(X_train) / batch_size)
scheduled_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=n_steps, decay_rate=0.1)
optimizer = tf.keras.optimizers.SGD(learning_rate=scheduled_learning_rate)
```

这很简单明了，而且当您保存模型时，学习率及其调度（包括其状态）也会被保存。

至于1cycle，Keras 不支持它，但是可以通过创建一个自定义回调，在每次迭代时修改学习率来实现它，代码不到30行。要从回调的 `on_batch_begin()` 方法中更新优化器的学习率，您需要调用 `tf.keras.​back⁠end.set_value(self.model.optimizer.learning_rate`, `new_learning_rate)`。请参阅笔记本中的“1Cycle Scheduling”部分以获取示例。

总之，指数衰减、性能调度和1cycle可以显著加快收敛速度，所以试一试吧！

# 通过正则化避免过拟合

> 有了四个参数，我可以拟合一只大象，有了五个我可以让它摇动它的鼻子。
> 
> 约翰·冯·诺伊曼，引用自恩里科·费米在《自然》427中

拥有成千上万个参数，你可以拟合整个动物园。深度神经网络通常有数万个参数，有时甚至有数百万个。这给予它们极大的自由度，意味着它们可以拟合各种复杂的数据集。但这种极大的灵活性也使得网络容易过拟合训练集。通常需要正则化来防止这种情况发生。

我们已经在[第10章](ch10.html#ann_chapter)中实现了最好的正则化技术之一：提前停止。此外，即使批量归一化是为了解决不稳定梯度问题而设计的，它也像一个相当不错的正则化器。在本节中，我们将研究神经网络的其他流行正则化技术：ℓ[1] 和 ℓ[2] 正则化、dropout 和最大范数正则化。

## ℓ[1] 和 ℓ[2] 正则化

就像你在[第4章](ch04.html#linear_models_chapter)中为简单线性模型所做的那样，你可以使用 ℓ[2] 正则化来约束神经网络的连接权重，和/或者使用 ℓ[1] 正则化如果你想要一个稀疏模型（其中许多权重等于0）。以下是如何将 ℓ[2] 正则化应用于Keras层的连接权重，使用正则化因子为0.01：

```py
layer = tf.keras.layers.Dense(100, activation="relu",
                              kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))
```

`l2()` 函数返回一个正则化器，在训练过程中的每一步都会调用它来计算正则化损失。然后将其添加到最终损失中。正如你所期望的那样，如果你想要 ℓ[1] 正则化，你可以简单地使用`tf.keras.regularizers.l1()`；如果你想要同时使用 ℓ[1] 和 ℓ[2] 正则化，可以使用`tf.keras.regularizers.l1_l2()`（指定两个正则化因子）。

由于通常希望在网络的所有层中应用相同的正则化器，以及在所有隐藏层中使用相同的激活函数和相同的初始化策略，你可能会发现自己重复相同的参数。这会使代码变得丑陋且容易出错。为了避免这种情况，你可以尝试重构代码以使用循环。另一个选择是使用Python的`functools.partial()`函数，它允许你为任何可调用对象创建一个薄包装器，并设置一些默认参数值：

```py
from functools import partial

RegularizedDense = partial(tf.keras.layers.Dense,
                           activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    RegularizedDense(100),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax")
])
```

###### 警告

正如我们之前看到的，当使用 SGD、动量优化和 Nesterov 动量优化时，ℓ[2] 正则化是可以的，但在使用 Adam 及其变种时不行。如果你想要在使用 Adam 时进行权重衰减，那么不要使用 ℓ[2] 正则化：使用 AdamW 替代。

## Dropout

*Dropout* 是深度神经网络中最流行的正则化技术之一。它是由 Geoffrey Hinton 等人在2012年的一篇论文中提出的，并在2014年由 Nitish Srivastava 等人进一步详细阐述，已被证明非常成功：许多最先进的神经网络使用了dropout，因为它使它们的准确率提高了1%–2%。这听起来可能不多，但当一个模型已经有95%的准确率时，获得2%的准确率提升意味着将错误率减少了近40%（从5%的错误率降至大约3%）。

这是一个相当简单的算法：在每个训练步骤中，每个神经元（包括输入神经元，但始终不包括输出神经元）都有一个概率*p*在训练期间暂时“被丢弃”，这意味着在这个训练步骤中它将被完全忽略，但在下一个步骤中可能会活跃（参见[图11-10](#dropout_diagram)）。超参数*p*称为*dropout率*，通常设置在10%到50%之间：在循环神经网络中更接近20%-30%（参见[第15章](ch15.html#rnn_chapter)），在卷积神经网络中更接近40%-50%（参见[第14章](ch14.html#cnn_chapter)）。训练后，神经元不再被丢弃。这就是全部（除了我们将立即讨论的一个技术细节）。

最初令人惊讶的是，这种破坏性技术居然有效。如果一家公司告诉员工每天早上抛硬币决定是否去上班，公司会表现得更好吗？谁知道呢；也许会！公司将被迫调整其组织；它不能依赖任何一个人来操作咖啡机或执行其他关键任务，因此这种专业知识必须分散到几个人身上。员工必须学会与许多同事合作，而不仅仅是少数几个人。公司将变得更具弹性。如果有人离职，这不会有太大影响。目前尚不清楚这种想法是否适用于公司，但对于神经网络来说，它确实有效。使用dropout训练的神经元无法与其相邻的神经元共同适应；它们必须尽可能独立地发挥作用。它们也不能过度依赖少数输入神经元；它们必须关注每个输入神经元。它们最终对输入的轻微变化不太敏感。最终，您将获得一个更健壮的网络，具有更好的泛化能力。

![mls3 1110](assets/mls3_1110.png)

###### 图11-10。使用dropout正则化，每次训练迭代中，一个或多个层中的所有神经元的随机子集（除了输出层）会“被丢弃”；这些神经元在这次迭代中输出为0（由虚线箭头表示）

理解dropout的另一种方法是意识到在每个训练步骤中生成了一个独特的神经网络。由于每个神经元可以存在或不存在，因此存在2^(*N*)个可能的网络（其中*N*是可丢弃神经元的总数）。这是一个如此巨大的数字，以至于同一个神经网络被重复抽样几乎是不可能的。一旦您运行了10,000个训练步骤，您实际上已经训练了10,000个不同的神经网络，每个神经网络只有一个训练实例。这些神经网络显然不是独立的，因为它们共享许多权重，但它们仍然是不同的。最终的神经网络可以看作是所有这些较小神经网络的平均集合。

###### 提示

在实践中，通常只能将dropout应用于顶部一到三层的神经元（不包括输出层）。

有一个小但重要的技术细节。假设*p*=75%：平均每次训练步骤中只有25%的神经元是活跃的。这意味着在训练后，神经元将连接到四倍于训练期间的输入神经元。为了补偿这一事实，我们需要在训练期间将每个神经元的输入连接权重乘以四。如果不这样做，神经网络在训练期间和训练后将看到不同的数据，表现不佳。更一般地，在训练期间，我们需要将连接权重除以“保留概率”（1-*p*）。

使用Keras实现dropout，可以使用`tf.keras.layers.Dropout`层。在训练期间，它会随机丢弃一些输入（将它们设置为0），并将剩余的输入除以保留概率。训练结束后，它什么也不做；它只是将输入传递给下一层。以下代码在每个密集层之前应用了dropout正则化，使用了0.2的dropout率：

```py
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(100, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])
[...]  # compile and train the model
```

###### 警告

由于dropout只在训练期间激活，比较训练损失和验证损失可能会产生误导。特别是，模型可能会过度拟合训练集，但训练和验证损失却相似。因此，请确保在没有dropout的情况下评估训练损失（例如，在训练后）。

如果观察到模型过拟合，可以增加dropout率。相反，如果模型对训练集拟合不足，可以尝试减少dropout率。对于大型层，增加dropout率，对于小型层，减少dropout率也有帮助。此外，许多最先进的架构仅在最后一个隐藏层之后使用dropout，因此如果全局dropout太强，您可能想尝试这样做。

Dropout确实会显著减慢收敛速度，但在适当调整后通常会得到更好的模型。因此，额外的时间和精力通常是值得的，特别是对于大型模型。

###### 提示

如果要对基于SELU激活函数的自正则化网络进行正则化（如前面讨论的），应该使用*alpha dropout*：这是一种保留其输入均值和标准差的dropout变体。它是在与SELU一起引入的同一篇论文中提出的，因为常规dropout会破坏自正则化。

## 蒙特卡洛（MC）Dropout

2016年，Yarin Gal和Zoubin Ghahramani的一篇[论文](https://homl.info/mcdropout)建立了使用dropout的更多好理由：

+   首先，该论文建立了dropout网络（即包含`Dropout`层的神经网络）与近似贝叶斯推断之间的深刻联系，为dropout提供了坚实的数学理论基础。

+   其次，作者引入了一种强大的技术称为*MC dropout*，它可以提升任何经过训练的dropout模型的性能，而无需重新训练它甚至修改它。它还提供了模型不确定性的更好度量，并且可以在几行代码中实现。

如果这一切听起来像某种“奇怪的技巧”点击诱饵，那么看看以下代码。这是MC dropout的完整实现，增强了我们之前训练的dropout模型而无需重新训练它：

```py
import numpy as np

y_probas = np.stack([model(X_test, training=True)
                     for sample in range(100)])
y_proba = y_probas.mean(axis=0)
```

请注意，`model(X)`类似于`model.predict(X)`，只是它返回一个张量而不是NumPy数组，并支持`training`参数。在这个代码示例中，设置`training=True`确保`Dropout`层保持活动状态，因此所有预测都会有些不同。我们只对测试集进行100次预测，并计算它们的平均值。更具体地说，每次调用模型都会返回一个矩阵，每个实例一行，每个类别一列。因为测试集中有10,000个实例和10个类别，所以这是一个形状为[10000, 10]的矩阵。我们堆叠了100个这样的矩阵，所以`y_probas`是一个形状为[100, 10000, 10]的3D数组。一旦我们在第一个维度上取平均值（`axis=0`），我们得到`y_proba`，一个形状为[10000, 10]的数组，就像我们在单次预测中得到的一样。就是这样！在打开dropout的情况下对多次预测取平均值会给我们一个通常比关闭dropout的单次预测结果更可靠的蒙特卡洛估计。例如，让我们看看模型对Fashion MNIST测试集中第一个实例的预测，关闭dropout：

```py
>>> model.predict(X_test[:1]).round(3)
array([[0\.   , 0\.   , 0\.   , 0\.   , 0\.   , 0.024, 0\.   , 0.132, 0\.   ,
 0.844]], dtype=float32)
```

模型相当自信（84.4%）这张图片属于第9类（踝靴）。与MC dropout预测进行比较：

```py
>>> y_proba[0].round(3)
array([0\.   , 0\.   , 0\.   , 0\.   , 0\.   , 0.067, 0\.   , 0.209, 0.001,
 0.723], dtype=float32)
```

模型似乎仍然更喜欢类别9，但其置信度降至72.3%，类别5（凉鞋）和7（运动鞋）的估计概率增加，这是有道理的，因为它们也是鞋类。

MC dropout倾向于提高模型概率估计的可靠性。这意味着它不太可能自信但错误，这可能是危险的：想象一下一个自动驾驶汽车自信地忽略一个停车标志。了解哪些其他类别最有可能也很有用。此外，您可以查看[概率估计的标准差](https://xkcd.com/2110)：

```py
>>> y_std = y_probas.std(axis=0)
>>> y_std[0].round(3)
array([0\.   , 0\.   , 0\.   , 0.001, 0\.   , 0.096, 0\.   , 0.162, 0.001,
 0.183], dtype=float32)
```

显然，类别9的概率估计存在相当大的方差：标准差为0.183，应与估计的概率0.723进行比较：如果您正在构建一个风险敏感的系统（例如医疗或金融系统），您可能会对这种不确定的预测极为谨慎。您绝对不会将其视为84.4%的自信预测。模型的准确性也从87.0%略微提高到87.2%：

```py
>>> y_pred = y_proba.argmax(axis=1)
>>> accuracy = (y_pred == y_test).sum() / len(y_test)
>>> accuracy
0.8717
```

###### 注意

您使用的蒙特卡洛样本数量（在此示例中为100）是一个可以调整的超参数。它越高，预测和不确定性估计就越准确。但是，如果您将其加倍，推断时间也将加倍。此外，在一定数量的样本之上，您将注意到改进很小。您的任务是根据您的应用程序找到延迟和准确性之间的正确权衡。

如果您的模型包含在训练期间以特殊方式行为的其他层（例如`BatchNormalization`层），那么您不应该像我们刚刚做的那样强制训练模式。相反，您应该用以下`MCDropout`类替换`Dropout`层：⁠^([30](ch11.html#idm45720196529216))

```py
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)
```

在这里，我们只是子类化`Dropout`层，并覆盖`call()`方法以强制其`training`参数为`True`（请参阅[第12章](ch12.html#tensorflow_chapter)）。类似地，您可以通过子类化`AlphaDropout`来定义一个`MCAlphaDropout`类。如果您从头开始创建一个模型，只需使用`MCDropout`而不是`Dropout`。但是，如果您已经使用`Dropout`训练了一个模型，您需要创建一个与现有模型相同但使用`Dropout`而不是`MCDropout`的新模型，然后将现有模型的权重复制到新模型中。

简而言之，MC dropout是一种很棒的技术，可以提升dropout模型并提供更好的不确定性估计。当然，由于在训练期间只是常规的dropout，因此它也起到了正则化的作用。

## 最大范数正则化

神经网络的另一种流行的正则化技术称为*最大范数正则化*：对于每个神经元，它约束传入连接的权重**w**，使得∥ **w** ∥[2] ≤ *r*，其中*r*是最大范数超参数，∥ · ∥[2]是ℓ[2]范数。

最大范数正则化不会向整体损失函数添加正则化损失项。相反，通常是在每个训练步骤之后计算∥ **w** ∥[2]，并在需要时重新缩放**w**（**w** ← **w** *r* / ∥ **w** ∥[2]）。

减小*r*会增加正则化的程度，并有助于减少过拟合。最大范数正则化还可以帮助缓解不稳定的梯度问题（如果您没有使用批量归一化）。

在Keras中实现最大范数正则化，将每个隐藏层的`kernel_constraint`参数设置为具有适当最大值的`max_norm()`约束，如下所示：

```py
dense = tf.keras.layers.Dense(
    100, activation="relu", kernel_initializer="he_normal",
    kernel_constraint=tf.keras.constraints.max_norm(1.))
```

在每次训练迭代之后，模型的`fit()`方法将调用`max_norm()`返回的对象，将该层的权重传递给它，并得到重新缩放的权重，然后替换该层的权重。正如您将在[第12章](ch12.html#tensorflow_chapter)中看到的，如果需要，您可以定义自己的自定义约束函数，并将其用作`kernel_constraint`。您还可以通过设置`bias_constraint`参数来约束偏置项。

`max_norm()`函数有一个默认为`0`的`axis`参数。一个`Dense`层通常具有形状为[*输入数量*，*神经元数量*]的权重，因此使用`axis=0`意味着最大范数约束将独立应用于每个神经元的权重向量。如果您想在卷积层中使用最大范数（参见[第14章](ch14.html#cnn_chapter)），请确保适当设置`max_norm()`约束的`axis`参数（通常为`axis=[0, 1, 2]`）。

# 总结和实用指南

在本章中，我们涵盖了各种技术，您可能想知道应该使用哪些技术。这取决于任务，目前还没有明确的共识，但我发现[表11-3](#default_deep_neural_network_config)中的配置在大多数情况下都能很好地工作，而不需要太多的超参数调整。尽管如此，请不要将这些默认值视为硬性规则！

表11-3. 默认DNN配置

| 超参数 | 默认值 |
| --- | --- |
| 内核初始化器 | He初始化 |
| 激活函数 | 如果是浅层则为ReLU；如果是深层则为Swish |
| 归一化 | 如果是浅层则为无；如果是深层则为批量归一化 |
| 正则化 | 提前停止；如果需要则使用权重衰减 |
| 优化器 | Nesterov加速梯度或AdamW |
| 学习率调度 | 性能调度或1cycle |

如果网络是简单的密集层堆叠，则它可以自我归一化，您应该使用[表11-4](#self_norm_deep_neural_network_config)中的配置。

表11-4. 自我归一化网络的DNN配置

| 超参数 | 默认值 |
| --- | --- |
| 内核初始化器 | LeCun初始化 |
| 激活函数 | SELU |
| 归一化 | 无（自我归一化） |
| 正则化 | 如果需要则使用Alpha dropout |
| 优化器 | Nesterov加速梯度 |
| 学习率调度 | 性能调度或1cycle |

不要忘记对输入特征进行归一化！您还应尝试重用预训练神经网络的部分，如果您可以找到一个解决类似问题的模型，或者如果您有大量未标记数据，则使用无监督预训练，或者如果您有大量类似任务的标记数据，则使用辅助任务的预训练。

虽然前面的指南应该涵盖了大多数情况，但这里有一些例外情况：

+   如果您需要一个稀疏模型，您可以使用ℓ[1]正则化（并在训练后选择性地将微小权重归零）。如果您需要一个更稀疏的模型，您可以使用TensorFlow模型优化工具包。这将破坏自我归一化，因此在这种情况下应使用默认配置。

+   如果您需要一个低延迟模型（执行闪电般快速预测的模型），您可能需要使用更少的层，使用快速激活函数（如ReLU或leaky ReLU），并在训练后将批量归一化层折叠到前面的层中。拥有一个稀疏模型也会有所帮助。最后，您可能希望将浮点精度从32位减少到16位甚至8位（参见[“将模型部署到移动设备或嵌入式设备”](ch19.html#deployingModelMobileEmbedded)）。再次，查看TF-MOT。

+   如果您正在构建一个风险敏感的应用程序，或者推断延迟在您的应用程序中并不是非常重要，您可以使用MC dropout来提高性能，并获得更可靠的概率估计，以及不确定性估计。

有了这些指导，您现在已经准备好训练非常深的网络了！我希望您现在相信，只使用方便的Keras API就可以走很长一段路。然而，可能会有一天，当您需要更多控制时，例如编写自定义损失函数或调整训练算法时。对于这种情况，您将需要使用TensorFlow的较低级别API，您将在下一章中看到。

# 练习

1.  Glorot初始化和He初始化旨在解决什么问题？

1.  只要使用He初始化随机选择的值将所有权重初始化为相同值，这样做可以吗？

1.  将偏置项初始化为0可以吗？

1.  在本章讨论的每种激活函数中，您希望在哪些情况下使用？

1.  当使用`SGD`优化器时，如果将`momentum`超参数设置得太接近1（例如0.99999），可能会发生什么？

1.  列出三种可以生成稀疏模型的方法。

1.  Dropout会减慢训练速度吗？它会减慢推断速度（即对新实例进行预测）吗？MC dropout呢？

1.  练习在CIFAR10图像数据集上训练深度神经网络：

    1.  构建一个具有20个每层100个神经元的隐藏层的DNN（这太多了，但这是这个练习的重点）。使用He初始化和Swish激活函数。

    1.  使用Nadam优化和提前停止，在CIFAR10数据集上训练网络。您可以使用`tf.keras.datasets.cifar10.load_​data()`加载数据集。该数据集由60,000个32×32像素的彩色图像组成（50,000个用于训练，10,000个用于测试），具有10个类别，因此您需要一个具有10个神经元的softmax输出层。记得每次更改模型架构或超参数时都要搜索正确的学习率。

    1.  现在尝试添加批量归一化并比较学习曲线：它是否比以前收敛得更快？它是否产生更好的模型？它如何影响训练速度？

    1.  尝试用SELU替换批量归一化，并进行必要的调整以确保网络自我归一化（即标准化输入特征，使用LeCun正态初始化，确保DNN仅包含一系列密集层等）。

    1.  尝试使用alpha dropout对模型进行正则化。然后，在不重新训练模型的情况下，看看是否可以通过MC dropout获得更好的准确性。

    1.  使用1cycle调度重新训练您的模型，看看它是否提高了训练速度和模型准确性。

这些练习的解决方案可在本章笔记本的末尾找到，网址为[*https://homl.info/colab3*](https://homl.info/colab3)。

^([1](ch11.html#idm45720199815184-marker)) Xavier Glorot和Yoshua Bengio，“理解训练深度前馈神经网络的困难”，*第13届人工智能和统计国际会议论文集*（2010）：249-256。

^([2](ch11.html#idm45720199802080-marker)) 这里有一个类比：如果将麦克风放大器的旋钮调得太接近零，人们就听不到您的声音，但如果将其调得太接近最大值，您的声音将被饱和，人们将听不懂您在说什么。现在想象一下这样一系列放大器：它们都需要适当设置，以便您的声音在链的末端响亮清晰地传出。您的声音必须以与进入时相同的幅度从每个放大器中传出。

^([3](ch11.html#idm45720199777104-marker)) 例如，Kaiming He等人，“深入研究整流器：在ImageNet分类上超越人类水平表现”，*2015年IEEE国际计算机视觉大会论文集*（2015）：1026-1034。

^([4](ch11.html#idm45720199613584-marker)) 如果神经元下面的层中的输入随时间演变并最终返回到ReLU激活函数再次获得正输入的范围内，死神经元可能会复活。例如，如果梯度下降调整了死神经元下面的神经元，这种情况可能会发生。

^([5](ch11.html#idm45720199596240-marker)) Bing Xu等人，“卷积网络中修正激活的实证评估”，arXiv预印本arXiv:1505.00853（2015）。

^([6](ch11.html#idm45720199427712-marker)) Djork-Arné Clevert等人，“指数线性单元（ELUs）快速准确的深度网络学习”，*国际学习表示会议论文集*，arXiv预印本（2015年）。

^([7](ch11.html#idm45720199386128-marker)) Günter Klambauer等人，“自正则化神经网络”，*第31届国际神经信息处理系统会议论文集*（2017）：972–981。

^([8](ch11.html#idm45720199369856-marker)) Dan Hendrycks和Kevin Gimpel，“高斯误差线性单元（GELUs）”，arXiv预印本arXiv:1606.08415（2016）。

^([9](ch11.html#idm45720199356368-marker)) 如果曲线上任意两点之间的线段永远不会低于曲线，则函数是凸的。单调函数只增加或只减少。

^([10](ch11.html#idm45720199347968-marker)) Prajit Ramachandran等人，“寻找激活函数”，arXiv预印本arXiv:1710.05941（2017）。

^([11](ch11.html#idm45720199337776-marker)) Diganta Misra，“Mish：一种自正则化的非单调激活函数”，arXiv预印本arXiv:1908.08681（2019）。

^([12](ch11.html#idm45720199311872-marker)) Sergey Ioffe和Christian Szegedy，“批量归一化：通过减少内部协变量转移加速深度网络训练”，*第32届国际机器学习会议论文集*（2015）：448–456。

^([13](ch11.html#idm45720198989872-marker)) 然而，它们是根据训练数据在训练期间估计的，因此可以说它们是可训练的。在Keras中，“不可训练”实际上意味着“不受反向传播影响”。

^([14](ch11.html#idm45720198748720-marker)) Razvan Pascanu等人，“关于训练递归神经网络的困难”，*第30届国际机器学习会议论文集*（2013）：1310–1318。

^([15](ch11.html#idm45720198261376-marker)) Boris T. Polyak，“加速迭代方法收敛的一些方法”，*苏联计算数学和数学物理杂志* 4，第5期（1964）：1–17。

^([16](ch11.html#idm45720198190768-marker)) Yurii Nesterov，“一种具有收敛速率*O*(1/*k*²)的无约束凸最小化问题方法”，*苏联科学院学报* 269（1983）：543–547。

^([17](ch11.html#idm45720198095888-marker)) John Duchi等人，“用于在线学习和随机优化的自适应次梯度方法”，*机器学习研究杂志* 12（2011）：2121–2159。

^([18](ch11.html#idm45720198007744-marker)) 该算法由Geoffrey Hinton和Tijmen Tieleman于2012年创建，并由Geoffrey Hinton在他关于神经网络的Coursera课程中介绍（幻灯片：[*https://homl.info/57*](https://homl.info/57)；视频：[*https://homl.info/58*](https://homl.info/58)）。有趣的是，由于作者没有撰写描述该算法的论文，研究人员经常在其论文中引用“第6e讲座的第29张幻灯片”。

^([19](ch11.html#idm45720197966960-marker)) *ρ*是希腊字母rho。

^([20](ch11.html#idm45720197921776-marker)) Diederik P. Kingma和Jimmy Ba，“Adam：一种随机优化方法”，arXiv预印本arXiv:1412.6980（2014）。

^([21](ch11.html#idm45720197757088-marker)) Timothy Dozat，“将Nesterov动量合并到Adam中”（2016）。

^([22](ch11.html#idm45720197753840-marker)) Ilya Loshchilov和Frank Hutter，“解耦权重衰减正则化”，arXiv预印本arXiv:1711.05101（2017）。

^([23](ch11.html#idm45720197744832-marker)) Ashia C. Wilson等人，“机器学习中自适应梯度方法的边际价值”，*神经信息处理系统进展* 30（2017）：4148–4158。

Leslie N. Smith，“神经网络超参数的纪律性方法：第1部分—学习率、批量大小、动量和权重衰减”，arXiv预印本arXiv:1803.09820（2018）。

Andrew Senior等人，“深度神经网络在语音识别中的学习率的实证研究”，*IEEE国际会议论文集*（2013）：6724–6728。

Geoffrey E. Hinton等人，“通过防止特征探测器的共适应来改进神经网络”，arXiv预印本arXiv:1207.0580（2012）。

Nitish Srivastava等人，“Dropout：防止神经网络过拟合的简单方法”，*机器学习研究杂志* 15（2014）：1929–1958。

Yarin Gal和Zoubin Ghahramani，“Dropout作为贝叶斯近似：在深度学习中表示模型不确定性”，*第33届国际机器学习会议论文集*（2016）：1050–1059。

具体来说，他们表明训练一个dropout网络在数学上等同于在一种特定类型的概率模型中进行近似贝叶斯推断，这种模型被称为*深高斯过程*。

这个`MCDropout`类将与所有Keras API一起工作，包括顺序API。如果您只关心功能API或子类API，您不必创建一个`MCDropout`类；您可以创建一个常规的`Dropout`层，并使用`training=True`调用它。
