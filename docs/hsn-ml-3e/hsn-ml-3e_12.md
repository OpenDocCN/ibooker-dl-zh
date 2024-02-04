# 第10章。使用Keras介绍人工神经网络

鸟类启发我们飞行，牛蒡植物启发了钩带，自然启发了无数更多的发明。因此，看看大脑的结构以获取如何构建智能机器的灵感似乎是合乎逻辑的。这就是激发*人工神经网络*（ANNs）的逻辑，这是受到我们大脑中生物神经元网络启发的机器学习模型。然而，尽管飞机受到鸟类的启发，但它们不必拍打翅膀才能飞行。同样，人工神经网络逐渐与其生物表亲有所不同。一些研究人员甚至主张我们应该完全放弃生物类比（例如，使用“单元”而不是“神经元”），以免将我们的创造力限制在生物学上可行的系统中。⁠^（[1](ch10.html#idm45720205300576)）

ANNs是深度学习的核心。它们多才多艺，强大且可扩展，使其成为处理大规模和高度复杂的机器学习任务的理想选择，例如对数十亿张图像进行分类（例如Google Images），为语音识别服务提供动力（例如苹果的Siri），每天向数亿用户推荐最佳观看视频（例如YouTube），或学会击败围棋世界冠军（DeepMind的AlphaGo）。

本章的第一部分介绍了人工神经网络，从快速浏览最初的ANN架构开始，一直到如今广泛使用的多层感知器（其他架构将在接下来的章节中探讨）。在第二部分中，我们将看看如何使用TensorFlow的Keras API实现神经网络。这是一个设计精美且简单的高级API，用于构建、训练、评估和运行神经网络。但不要被它的简单性所迷惑：它足够表达和灵活，可以让您构建各种各样的神经网络架构。实际上，对于大多数用例来说，它可能已经足够了。如果您需要额外的灵活性，您始终可以使用其较低级别的API编写自定义Keras组件，甚至直接使用TensorFlow，正如您将在[第12章](ch12.html#tensorflow_chapter)中看到的。

但首先，让我们回到过去，看看人工神经网络是如何产生的！

# 从生物到人工神经元

令人惊讶的是，人工神经网络已经存在了相当长的时间：它们最早是由神经生理学家沃伦·麦卡洛克和数学家沃尔特·皮茨于1943年首次提出的。在他们的[里程碑论文](https://homl.info/43)⁠^（[2](ch10.html#idm45720205291952)）“神经活动中内在的思想逻辑演算”，麦卡洛克和皮茨提出了一个简化的计算模型，说明了生物神经元如何在动物大脑中共同工作以使用*命题逻辑*执行复杂计算。这是第一个人工神经网络架构。从那时起，许多其他架构已经被发明，正如您将看到的。

人工神经网络的早期成功导致了人们普遍相信我们很快将与真正智能的机器交谈。当在1960年代清楚地意识到这一承诺将无法实现（至少在相当长一段时间内）时，资金转向其他地方，人工神经网络进入了一个漫长的冬天。在20世纪80年代初，发明了新的架构并开发了更好的训练技术，引发了对*连接主义*的兴趣复苏，即神经网络的研究。但进展缓慢，到了20世纪90年代，其他强大的机器学习技术已经被发明出来，例如支持向量机（参见[第5章](ch05.html#svm_chapter)）。这些技术似乎提供了比人工神经网络更好的结果和更强的理论基础，因此神经网络的研究再次被搁置。

我们现在正在目睹对人工神经网络的又一波兴趣。这波潮流会像以前的那些一样消失吗？好吧，以下是一些有理由相信这一次不同的好理由，以及对人工神经网络的重新兴趣将对我们的生活产生更深远影响的原因：

+   现在有大量的数据可用于训练神经网络，人工神经网络在非常大型和复杂的问题上经常胜过其他机器学习技术。

+   自1990年以来计算能力的巨大增长现在使得在合理的时间内训练大型神经网络成为可能。这在一定程度上归功于摩尔定律（集成电路中的元件数量在过去50年里大约每2年翻一番），但也要感谢游戏行业，它刺激了数以百万计的强大GPU卡的生产。此外，云平台使这种能力对每个人都可获得。

+   训练算法已经得到改进。公平地说，它们与1990年代使用的算法只有略微不同，但这些相对较小的调整产生了巨大的积极影响。

+   一些人工神经网络的理论限制在实践中被证明是良性的。例如，许多人认为人工神经网络训练算法注定会陷入局部最优解，但事实证明，在实践中这并不是一个大问题，特别是对于更大的神经网络：局部最优解通常表现几乎和全局最优解一样好。

+   人工神经网络似乎已经进入了资金和进展的良性循环。基于人工神经网络的惊人产品经常成为头条新闻，这吸引了越来越多的关注和资金，导致了越来越多的进展和更多惊人的产品。

## 生物神经元

在我们讨论人工神经元之前，让我们快速看一下生物神经元（在[图10-1](#biological_neuron_wikipedia)中表示）。它是一种在动物大脑中大多数发现的不寻常的细胞。它由一个包含细胞核和大多数细胞复杂组分的*细胞体*组成，许多分支延伸称为*树突*，以及一个非常长的延伸称为*轴突*。轴突的长度可能仅比细胞体长几倍，或者长达成千上万倍。在其末端附近，轴突分裂成许多称为*末梢*的分支，而在这些分支的顶端是微小的结构称为*突触终端*（或简称*突触*），它们连接到其他神经元的树突或细胞体。生物神经元产生称为*动作电位*（APs，或简称*信号*）的短电脉冲，这些电脉冲沿着轴突传播，并使突触释放称为*神经递质*的化学信号。当一个神经元在几毫秒内接收到足够量的这些神经递质时，它会发出自己的电脉冲（实际上，这取决于神经递质，因为其中一些会抑制神经元的发放）。

![mls3 1001](assets/mls3_1001.png)

###### 图10-1\. 一个生物神经元⁠^([4](ch10.html#idm45720205264656))

因此，单个生物神经元似乎表现出简单的方式，但它们组织在一个庞大的网络中，有数十亿个神经元，每个神经元通常连接到成千上万个其他神经元。高度复杂的计算可以通过一个相当简单的神经元网络执行，就像一个复杂的蚁丘可以从简单的蚂蚁的共同努力中出现一样。生物神经网络（BNNs）的架构是活跃研究的主题，但大脑的某些部分已经被绘制出来。这些努力表明，神经元通常组织成连续的层，特别是在大脑的外层皮层（大脑的外层），如[图10-2](#biological_neural_network_wikipedia)所示。

![mls3 1002](assets/mls3_1002.png)

###### 图10-2\. 生物神经网络中的多个层（人类皮层）⁠^([6](ch10.html#idm45720205254384))

## 使用神经元进行逻辑计算

McCulloch和Pitts提出了生物神经元的一个非常简单的模型，后来被称为*人工神经元*：它具有一个或多个二进制（开/关）输入和一个二进制输出。当其输入中的活动超过一定数量时，人工神经元会激活其输出。在他们的论文中，McCulloch和Pitts表明，即使使用这样简化的模型，也可以构建一个可以计算任何您想要的逻辑命题的人工神经元网络。为了了解这样一个网络是如何工作的，让我们构建一些执行各种逻辑计算的人工神经网络（请参见[图10-3](#nn_propositional_logic_diagram)），假设当至少两个输入连接处于活动状态时，神经元被激活。

![mls3 1003](assets/mls3_1003.png)

###### 图10-3。执行简单逻辑计算的人工神经网络

让我们看看这些网络的作用：

+   左侧的第一个网络是恒等函数：如果神经元A被激活，则神经元C也会被激活（因为它从神经元A接收到两个输入信号）；但如果神经元A处于关闭状态，则神经元C也会关闭。

+   第二个网络执行逻辑AND操作：只有当神经元A和B都被激活时，神经元C才会被激活（单个输入信号不足以激活神经元C）。

+   第三个网络执行逻辑OR操作：只有当神经元A或神经元B被激活（或两者都被激活）时，神经元C才会被激活。

+   最后，如果我们假设一个输入连接可以抑制神经元的活动（这是生物神经元的情况），那么第四个网络将计算一个稍微更复杂的逻辑命题：只有当神经元A处于活动状态且神经元B处于关闭状态时，神经元C才会被激活。如果神经元A一直处于活动状态，那么您将得到一个逻辑NOT：当神经元B处于关闭状态时，神经元C处于活动状态，反之亦然。

您可以想象这些网络如何组合以计算复杂的逻辑表达式（请参见本章末尾的练习示例）。

## 感知器

感知器是最简单的人工神经网络架构之一，由Frank Rosenblatt于1957年发明。它基于一个略有不同的人工神经元（见[图10-4](#artificial_neuron_diagram)）称为*阈值逻辑单元*（TLU），有时也称为*线性阈值单元*（LTU）。输入和输出是数字（而不是二进制的开/关值），每个输入连接都与一个权重相关联。TLU首先计算其输入的线性函数：*z* = *w*[1] *x*[1] + *w*[2] *x*[2] + ⋯ + *w*[*n*] *x*[*n*] + *b* = **w**^⊺ **x** + *b*。然后它将结果应用于*阶跃函数*：*h*[**w**](**x**) = step(*z*)。因此，这几乎就像逻辑回归，只是它使用了一个阶跃函数而不是逻辑函数（[第4章](ch04.html#linear_models_chapter)）。就像在逻辑回归中一样，模型参数是输入权重**w**和偏置项*b*。

![mls3 1004](assets/mls3_1004.png)

###### 图10-4。TLU：计算其输入**w**^⊺ **x**的加权和，加上偏置项*b*，然后应用一个阶跃函数

感知器中最常用的阶跃函数是*海维赛德阶跃函数*（见[方程式10-1](#step_functions_equation)）。有时也会使用符号函数。

##### 方程式10-1。感知器中常用的阶跃函数（假设阈值=0）

<math display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mo form="prefix">heaviside</mo> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>=</mo> <mfenced separators="" open="{" close=""><mtable><mtr><mtd columnalign="left"><mn>0</mn></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mi>z</mi> <mo><</mo> <mn>0</mn></mrow></mtd></mtr> <mtr><mtd columnalign="left"><mn>1</mn></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mi>z</mi> <mo>≥</mo> <mn>0</mn></mrow></mtd></mtr></mtable></mfenced></mrow></mtd> <mtd columnalign="left"><mrow><mo form="prefix">sgn</mo> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>=</mo> <mfenced separators="" open="{" close=""><mtable><mtr><mtd columnalign="left"><mrow><mo>-</mo> <mn>1</mn></mrow></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mi>z</mi> <mo><</mo> <mn>0</mn></mrow></mtd></mtr> <mtr><mtd columnalign="left"><mn>0</mn></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mi>z</mi> <mo>=</mo> <mn>0</mn></mrow></mtd></mtr> <mtr><mtd columnalign="left"><mrow><mo>+</mo> <mn>1</mn></mrow></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mi>z</mi> <mo>></mo> <mn>0</mn></mrow></mtd></mtr></mtable></mfenced></mrow></mtd></mtr></mtable></math>

一个单个的TLU可以用于简单的线性二元分类。它计算其输入的线性函数，如果结果超过阈值，则输出正类。否则，输出负类。这可能让你想起了逻辑回归（[第4章](ch04.html#linear_models_chapter)）或线性SVM分类（[第5章](ch05.html#svm_chapter)）。例如，你可以使用一个单个的TLU基于花瓣长度和宽度对鸢尾花进行分类。训练这样一个TLU需要找到正确的*w*[1]、*w*[2]和*b*的值（训练算法将很快讨论）。

一个感知器由一个或多个TLU组成，组织在一个单层中，其中每个TLU连接到每个输入。这样的一层被称为*全连接层*或*密集层*。输入构成*输入层*。由于TLU层产生最终输出，因此被称为*输出层*。例如，一个具有两个输入和三个输出的感知器在[图10-5](#perceptron_diagram)中表示。

![mls3 1005](assets/mls3_1005.png)

###### 图10-5。具有两个输入和三个输出神经元的感知器的架构

这个感知器可以同时将实例分类为三个不同的二进制类别，这使它成为一个多标签分类器。它也可以用于多类分类。

由于线性代数的魔力，[方程10-2](#neural_network_layer_equation)可以用来高效地计算一层人工神经元对多个实例的输出。

##### 方程10-2。计算全连接层的输出

<math display="block"><mrow><msub><mi>h</mi> <mrow><mi mathvariant="bold">W</mi><mo>,</mo><mi mathvariant="bold">b</mi></mrow></msub> <mrow><mo>(</mo> <mi mathvariant="bold">X</mi> <mo>)</mo></mrow> <mo>=</mo> <mi mathvariant="normal">ϕ</mi> <mrow><mo>(</mo> <mi mathvariant="bold">X</mi> <mi mathvariant="bold">W</mi> <mo>+</mo> <mi mathvariant="bold">b</mi> <mo>)</mo></mrow></mrow></math>

在这个方程中：

+   如常，**X**代表输入特征的矩阵。每个实例一行，每个特征一列。

+   权重矩阵**W**包含所有的连接权重。它每行对应一个输入，每列对应一个神经元。

+   偏置向量**b**包含所有的偏置项：每个神经元一个。

+   函数ϕ被称为*激活函数*：当人工神经元是TLU时，它是一个阶跃函数（我们将很快讨论其他激活函数）。

###### 注意

在数学中，矩阵和向量的和是未定义的。然而，在数据科学中，我们允许“广播”：将一个向量添加到矩阵中意味着将它添加到矩阵中的每一行。因此，**XW** + **b**首先将**X**乘以**W**，得到一个每个实例一行、每个输出一列的矩阵，然后将向量**b**添加到该矩阵的每一行，这将使每个偏置项添加到相应的输出中，对每个实例都是如此。此外，ϕ然后逐项应用于结果矩阵中的每个项目。

那么，感知器是如何训练的呢？Rosenblatt提出的感知器训练算法在很大程度上受到*Hebb规则*的启发。在他1949年的书《行为的组织》（Wiley）中，Donald Hebb建议，当一个生物神经元经常触发另一个神经元时，这两个神经元之间的连接会变得更加强大。 Siegrid Löwel后来用引人注目的短语总结了Hebb的想法，“一起激活的细胞，一起连接”；也就是说，当两个神经元同时激活时，它们之间的连接权重倾向于增加。这个规则后来被称为Hebb规则（或*Hebbian学习*）。感知器使用这个规则的变体进行训练，该规则考虑了网络在进行预测时所产生的错误；感知器学习规则加强了有助于减少错误的连接。更具体地说，感知器一次馈送一个训练实例，并为每个实例进行预测。对于每个产生错误预测的输出神经元，它加强了从输入到正确预测的贡献的连接权重。该规则显示在[方程10-3](#perceptron_update_rule)中。

##### 方程10-3。感知器学习规则（权重更新）

<math display="block"><mrow><msup><mrow><msub><mi>w</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub></mrow> <mrow><mo>(</mo><mtext>next</mtext><mtext>step</mtext><mo>)</mo></mrow></msup> <mo>=</mo> <msub><mi>w</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>+</mo> <mi>η</mi> <mrow><mo>(</mo> <msub><mi>y</mi> <mi>j</mi></msub> <mo>-</mo> <msub><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mi>j</mi></msub> <mo>)</mo></mrow> <msub><mi>x</mi> <mi>i</mi></msub></mrow></math>

在这个方程中：

+   *w*[*i*,] [*j*]是第*i*个输入和第*j*个神经元之间的连接权重。

+   *x*[*i*]是当前训练实例的第*i*个输入值。

+   <math><msub><mover><mi>y</mi><mo>^</mo></mover><mi>j</mi></msub></math>是当前训练实例的第*j*个输出神经元的输出。

+   *y*[*j*]是当前训练实例的第*j*个输出神经元的目标输出。

+   *η*是学习率（参见[第4章](ch04.html#linear_models_chapter)）。

每个输出神经元的决策边界是线性的，因此感知器无法学习复杂的模式（就像逻辑回归分类器一样）。然而，如果训练实例是线性可分的，Rosenblatt证明了这个算法会收敛到一个解决方案。这被称为*感知器收敛定理*。

Scikit-Learn提供了一个`Perceptron`类，可以像你期望的那样使用，例如在鸢尾花数据集上（在[第4章](ch04.html#linear_models_chapter)介绍）。

```py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)  # Iris setosa

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

X_new = [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)  # predicts True and False for these 2 flowers
```

您可能已经注意到感知器学习算法与随机梯度下降（在[第4章](ch04.html#linear_models_chapter)介绍）非常相似。事实上，Scikit-Learn的`Perceptron`类等同于使用具有以下超参数的`SGDClassifier`：`loss="perceptron"`、`learning_rate="constant"`、`eta0=1`（学习率）和`penalty=None`（无正则化）。

在他们1969年的专著*感知器*中，Marvin Minsky和Seymour Papert强调了感知器的一些严重弱点，特别是它们无法解决一些微不足道的问题（例如*异或*（XOR）分类问题；请参见[图10-6](#xor_diagram)的左侧）。这也适用于任何其他线性分类模型（如逻辑回归分类器），但研究人员对感知器寄予了更高的期望，有些人对此感到如此失望，以至于完全放弃了神经网络，转而研究更高级的问题，如逻辑、问题解决和搜索。实际应用的缺乏也没有帮助。

事实证明，通过堆叠多个感知器可以消除一些感知器的限制。结果得到的人工神经网络称为*多层感知器*（MLP）。MLP可以解决XOR问题，您可以通过计算[图10-6](#xor_diagram)右侧所代表的MLP的输出来验证：对于输入（0, 0）或（1, 1），网络输出为0，对于输入（0, 1）或（1, 0），它输出为1。尝试验证这个网络确实解决了XOR问题！

![mls3 1006](assets/mls3_1006.png)

###### 图10-6\. XOR分类问题及解决该问题的MLP

###### 注意

与逻辑回归分类器相反，感知器不会输出类概率。这是偏爱逻辑回归而不是感知器的一个原因。此外，感知器默认不使用任何正则化，训练会在训练集上没有更多预测错误时停止，因此该模型通常不会像逻辑回归或线性SVM分类器那样泛化得很好。然而，感知器可能训练速度稍快。

## 多层感知器和反向传播

一个MLP由一个输入层、一个或多个称为*隐藏层*的TLU层以及一个称为*输出层*的TLU层组成（请参见[图10-7](#mlp_diagram)）。靠近输入层的层通常称为*较低层*，靠近输出的层通常称为*较高层*。

![mls3 1007](assets/mls3_1007.png)

###### 图10-7\. 一个具有两个输入、一个包含四个神经元的隐藏层和三个输出神经元的多层感知器的架构

###### 注意

信号只能单向流动（从输入到输出），因此这种架构是*前馈神经网络*（FNN）的一个例子。

当一个人工神经网络包含深度堆叠的隐藏层时，它被称为*深度神经网络*（DNN）。深度学习领域研究DNNs，更一般地，它对包含深度堆叠计算的模型感兴趣。尽管如此，许多人在涉及神经网络时都谈论深度学习（即使是浅层的）。

多年来，研究人员努力寻找一种训练MLP的方法，但没有成功。在1960年代初，一些研究人员讨论了使用梯度下降来训练神经网络的可能性，但正如我们在[第4章](ch04.html#linear_models_chapter)中看到的，这需要计算模型参数的梯度与模型误差之间的关系；当时如何有效地处理这样一个包含如此多参数的复杂模型，尤其是使用当时的计算机时，这并不清楚。

然后，在1970年，一位名叫Seppo Linnainmaa的研究人员在他的硕士论文中介绍了一种自动高效计算所有梯度的技术。这个算法现在被称为*反向模式自动微分*（或简称*反向模式自动微分*）。通过网络的两次遍历（一次前向，一次后向），它能够计算神经网络中每个模型参数的误差梯度。换句话说，它可以找出如何调整每个连接权重和每个偏差以减少神经网络的误差。然后可以使用这些梯度执行梯度下降步骤。如果重复这个自动计算梯度和梯度下降步骤的过程，神经网络的误差将逐渐下降，直到最终达到最小值。这种反向模式自动微分和梯度下降的组合现在被称为*反向传播*（或简称*反向传播*）。

###### 注意

有各种自动微分技术，各有利弊。*反向模式自动微分*在要求对具有许多变量（例如连接权重和偏差）和少量输出（例如一个损失）进行微分时非常适用。如果想了解更多关于自动微分的信息，请查看[附录 B](app02.html#autodiff_appendix)。

反向传播实际上可以应用于各种计算图，不仅仅是神经网络：事实上，Linnainmaa的硕士论文并不是关于神经网络的，而是更为普遍。在反向传播开始用于训练神经网络之前，还需要几年时间，但它仍然不是主流。然后，在1985年，David Rumelhart、Geoffrey Hinton和Ronald Williams发表了一篇[开创性的论文](https://homl.info/44)⁠^([10](ch10.html#idm45720204912752))，分析了反向传播如何使神经网络学习到有用的内部表示。他们的结果非常令人印象深刻，以至于反向传播很快在该领域中流行起来。如今，它是迄今为止最受欢迎的神经网络训练技术。

让我们再详细介绍一下反向传播的工作原理：

+   它一次处理一个小批量（例如，每个包含32个实例），并多次遍历整个训练集。每次遍历称为*纪元*。

+   每个小批量通过输入层进入网络。然后，算法计算小批量中每个实例的第一个隐藏层中所有神经元的输出。结果传递到下一层，计算其输出并传递到下一层，依此类推，直到得到最后一层的输出，即输出层。这是*前向传递*：它与进行预测完全相同，只是所有中间结果都被保留，因为它们需要用于反向传递。

+   接下来，算法测量网络的输出误差（即，使用比较期望输出和网络实际输出的损失函数，并返回一些误差度量）。

+   然后计算每个输出偏差和每个连接到输出层的连接对误差的贡献。这是通过应用*链式法则*（可能是微积分中最基本的规则）进行分析的，使得这一步骤快速而精确。

+   然后，算法测量每个下一层中每个连接贡献的误差量，再次使用链式法则，向后工作直到达到输入层。正如前面解释的那样，这个反向传递有效地测量了网络中所有连接权重和偏差的误差梯度，通过网络向后传播误差梯度（因此算法的名称）。

+   最后，算法执行梯度下降步骤，调整网络中所有连接权重，使用刚刚计算的误差梯度。

###### 警告

重要的是要随机初始化所有隐藏层的连接权重，否则训练将失败。例如，如果你将所有权重和偏置初始化为零，那么给定层中的所有神经元将完全相同，因此反向传播将以完全相同的方式影响它们，因此它们将保持相同。换句话说，尽管每层有数百个神经元，但你的模型将表现得好像每层只有一个神经元：它不会太聪明。相反，如果你随机初始化权重，你会*打破对称*，并允许反向传播训练一个多样化的神经元团队。

简而言之，反向传播对一个小批量进行预测（前向传播），测量误差，然后逆向遍历每一层以测量每个参数的误差贡献（反向传播），最后调整连接权重和偏置以减少误差（梯度下降步骤）。

为了使反向传播正常工作，Rumelhart和他的同事对MLP的架构进行了关键更改：他们用逻辑函数替换了阶跃函数，*σ*(*z*) = 1 / (1 + exp(–*z*))，也称为S形函数。这是必不可少的，因为阶跃函数只包含平坦段，因此没有梯度可用（梯度下降无法在平坦表面上移动），而S形函数在任何地方都有明确定义的非零导数，允许梯度下降在每一步都取得一些进展。事实上，反向传播算法与许多其他激活函数一起工作得很好，不仅仅是S形函数。这里有另外两个流行的选择：

双曲正切函数：tanh(*z*) = 2*σ*(2*z*) – 1

就像S形函数一样，这个激活函数是*S*形的，连续的，可微的，但其输出值范围是-1到1（而不是S形函数的0到1）。这个范围倾向于使每一层的输出在训练开始时更多或更少地集中在0附近，这通常有助于加快收敛速度。

修正线性单元函数：ReLU(*z*) = max(0, *z*)

ReLU函数在*z* = 0处不可微（斜率突然变化，可能导致梯度下降跳动），其导数在*z* < 0时为0。然而，在实践中，它工作得很好，并且计算速度快，因此已经成为默认选择。重要的是，它没有最大输出值有助于减少梯度下降过程中的一些问题（我们将在[第11章](ch11.html#deep_chapter)中回到这个问题）。

这些流行的激活函数及其导数在[图10-8](#activation_functions_plot)中表示。但等等！为什么我们需要激活函数呢？如果你串联几个线性变换，你得到的只是一个线性变换。例如，如果f(*x*) = 2*x* + 3，g(*x*) = 5*x* – 1，那么串联这两个线性函数会给你另一个线性函数：f(g(*x*)) = 2(5*x* – 1) + 3 = 10*x* + 1。因此，如果在层之间没有一些非线性，那么即使是深层堆叠也等效于单层，你无法用它解决非常复杂的问题。相反，具有非线性激活的足够大的DNN在理论上可以逼近任何连续函数。

![mls3 1008](assets/mls3_1008.png)

###### 图10-8。激活函数（左）及其导数（右）

好了！你知道神经网络是从哪里来的，它们的架构是什么，以及如何计算它们的输出。你也学到了反向传播算法。但神经网络到底能做什么呢？

## 回归MLP

首先，MLP可以用于回归任务。如果要预测单个值（例如，给定房屋的许多特征，预测房屋的价格），则只需一个输出神经元：其输出是预测值。对于多变量回归（即一次预测多个值），您需要每个输出维度一个输出神经元。例如，要在图像中定位对象的中心，您需要预测2D坐标，因此需要两个输出神经元。如果还想在对象周围放置一个边界框，则需要另外两个数字：对象的宽度和高度。因此，您最终会得到四个输出神经元。

Scikit-Learn包括一个`MLPRegressor`类，让我们使用它来构建一个MLP，其中包含三个隐藏层，每个隐藏层由50个神经元组成，并在加利福尼亚房屋数据集上进行训练。为简单起见，我们将使用Scikit-Learn的`fetch_california_housing()`函数来加载数据。这个数据集比我们在[第2章](ch02.html#project_chapter)中使用的数据集简单，因为它只包含数值特征（没有`ocean_proximity`特征），并且没有缺失值。以下代码首先获取并拆分数据集，然后创建一个管道来标准化输入特征，然后将它们发送到`MLPRegressor`。这对于神经网络非常重要，因为它们是使用梯度下降进行训练的，正如我们在[第4章](ch04.html#linear_models_chapter)中看到的，当特征具有非常不同的尺度时，梯度下降不会收敛得很好。最后，代码训练模型并评估其验证错误。该模型在隐藏层中使用ReLU激活函数，并使用一种称为*Adam*的梯度下降变体（参见[第11章](ch11.html#deep_chapter)）来最小化均方误差，还有一点ℓ[2]正则化（您可以通过`alpha`超参数来控制）：

```py
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)  # about 0.505
```

我们得到了约0.505的验证RMSE，这与使用随机森林分类器得到的结果相当。对于第一次尝试来说，这还不错！

请注意，此MLP不使用任何激活函数用于输出层，因此可以自由输出任何值。这通常没问题，但是如果要确保输出始终为正值，则应在输出层中使用ReLU激活函数，或者使用*softplus*激活函数，它是ReLU的平滑变体：softplus(*z*) = log(1 + exp(*z*))。当*z*为负时，softplus接近0，当*z*为正时，softplus接近*z*。最后，如果要确保预测始终落在给定值范围内，则应使用sigmoid函数或双曲正切，并将目标缩放到适当的范围：sigmoid为0到1，tanh为-1到1。遗憾的是，`MLPRegressor`类不支持输出层中的激活函数。

###### 警告

在几行代码中使用Scikit-Learn构建和训练标准MLP非常方便，但神经网络的功能有限。这就是为什么我们将在本章的第二部分切换到Keras的原因。

`MLPRegressor`类使用均方误差，这通常是回归任务中想要的，但是如果训练集中有很多异常值，您可能更喜欢使用平均绝对误差。或者，您可能希望使用*Huber损失*，它是两者的组合。当误差小于阈值*δ*（通常为1）时，它是二次的，但是当误差大于*δ*时，它是线性的。线性部分使其对异常值不太敏感，而二次部分使其比平均绝对误差更快收敛并更精确。但是，`MLPRegressor`只支持MSE。

[表10-1](#regression_mlp_architecture)总结了回归MLP的典型架构。

表10-1\. 典型的回归MLP架构

| 超参数 | 典型值 |
| --- | --- |
| #隐藏层 | 取决于问题，但通常为1到5 |
| #每个隐藏层的神经元数 | 取决于问题，但通常为10到100 |
| #输出神经元 | 每个预测维度1个 |
| 隐藏激活 | ReLU |
| 输出激活 | 无，或ReLU/softplus（如果是正输出）或sigmoid/tanh（如果是有界输出） |
| 损失函数 | MSE，或者如果有异常值则为Huber |

## 分类MLP

MLP也可以用于分类任务。对于二元分类问题，您只需要一个使用sigmoid激活函数的输出神经元：输出将是0到1之间的数字，您可以将其解释为正类的估计概率。负类的估计概率等于1减去该数字。

MLP也可以轻松处理多标签二元分类任务（参见[第3章](ch03.html#classification_chapter)）。例如，您可以有一个电子邮件分类系统，预测每封传入的电子邮件是垃圾邮件还是正常邮件，并同时预测它是紧急还是非紧急邮件。在这种情况下，您需要两个输出神经元，都使用sigmoid激活函数：第一个将输出电子邮件是垃圾邮件的概率，第二个将输出它是紧急邮件的概率。更一般地，您将为每个正类分配一个输出神经元。请注意，输出概率不一定相加为1。这使模型可以输出任何标签组合：您可以有非紧急的正常邮件、紧急的正常邮件、非紧急的垃圾邮件，甚至可能是紧急的垃圾邮件（尽管那可能是一个错误）。

如果每个实例只能属于一个类别，且有三个或更多可能的类别（例如，数字图像分类中的类别0到9），那么您需要每个类别一个输出神经元，并且应该为整个输出层使用softmax激活函数（参见[图10-9](#fnn_for_classification_diagram)）。Softmax函数（在[第4章](ch04.html#linear_models_chapter)介绍）将确保所有估计的概率在0和1之间，并且它们相加为1，因为类别是互斥的。正如您在[第3章](ch03.html#classification_chapter)中看到的，这被称为多类分类。

关于损失函数，由于我们正在预测概率分布，交叉熵损失（或*x-熵*或简称对数损失，参见[第4章](ch04.html#linear_models_chapter)）通常是一个不错的选择。

![mls3 1009](assets/mls3_1009.png)

###### 图10-9。用于分类的现代MLP（包括ReLU和softmax）

Scikit-Learn在`sklearn.neural_network`包中有一个`MLPClassifier`类。它几乎与`MLPRegressor`类相同，只是它最小化交叉熵而不是均方误差。现在尝试一下，例如在鸢尾花数据集上。这几乎是一个线性任务，因此一个具有5到10个神经元的单层应该足够（确保对特征进行缩放）。

[表10-2](#classification_mlp_architecture)总结了分类MLP的典型架构。

表10-2。典型的分类MLP架构

| 超参数 | 二元分类 | 多标签二元分类 | 多类分类 |
| --- | --- | --- | --- |
| #隐藏层 | 通常为1到5层，取决于任务 |
| #输出神经元 | 1 | 每个二元标签1个 | 每个类别1个 |
| 输出层激活 | Sigmoid | Sigmoid | Softmax |
| 损失函数 | X-熵 | X-熵 | X-熵 |

###### 提示

在继续之前，我建议您完成本章末尾的练习1。您将尝试各种神经网络架构，并使用*TensorFlow playground*可视化它们的输出。这将非常有助于更好地理解MLP，包括所有超参数（层数和神经元数量、激活函数等）的影响。

现在您已经掌握了开始使用Keras实现MLP所需的所有概念！

# 使用Keras实现MLP

Keras是TensorFlow的高级深度学习API：它允许您构建、训练、评估和执行各种神经网络。最初，Keras库是由François Chollet作为研究项目的一部分开发的⁠^([12](ch10.html#idm45720204597504))，并于2015年3月作为一个独立的开源项目发布。由于其易用性、灵活性和美观的设计，它很快就受到了欢迎。

###### 注意

Keras曾支持多个后端，包括TensorFlow、PlaidML、Theano和Microsoft Cognitive Toolkit（CNTK）（最后两个遗憾地已弃用），但自版本2.4以来，Keras仅支持TensorFlow。同样，TensorFlow曾包括多个高级API，但在TensorFlow 2发布时，Keras被正式选择为其首选的高级API。安装TensorFlow将自动安装Keras，并且没有安装TensorFlow，Keras将无法工作。简而言之，Keras和TensorFlow相爱并结为夫妻。其他流行的深度学习库包括[Facebook的PyTorch](https://pytorch.org)和[Google的JAX](https://github.com/google/jax)。^([13](ch10.html#idm45720204591712))

现在让我们使用Keras！我们将首先构建一个用于图像分类的MLP。

###### 注意

Colab运行时已预装了最新版本的TensorFlow和Keras。但是，如果您想在自己的机器上安装它们，请参阅[*https://homl.info/install*](https://homl.info/install)上的安装说明。

## 使用顺序API构建图像分类器

首先，我们需要加载一个数据集。我们将使用时尚MNIST，它是MNIST的一个替代品（在[第3章](ch03.html#classification_chapter)介绍）。它与MNIST具有完全相同的格式（70,000个28×28像素的灰度图像，共10个类），但图像代表时尚物品而不是手写数字，因此每个类更加多样化，问题变得比MNIST更具挑战性。例如，一个简单的线性模型在MNIST上达到约92%的准确率，但在时尚MNIST上只有约83%。

### 使用Keras加载数据集

Keras提供了一些实用函数来获取和加载常见数据集，包括MNIST、时尚MNIST等。让我们加载时尚MNIST。它已经被洗牌并分成一个训练集（60,000张图片）和一个测试集（10,000张图片），但我们将从训练集中保留最后的5,000张图片用于验证：

```py
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
```

###### 提示

TensorFlow通常被导入为`tf`，Keras API可通过`tf.keras`使用。

使用Keras加载MNIST或时尚MNIST时，与Scikit-Learn相比的一个重要区别是，每个图像都表示为一个28×28的数组，而不是大小为784的一维数组。此外，像素强度表示为整数（从0到255），而不是浮点数（从0.0到255.0）。让我们看看训练集的形状和数据类型：

```py
>>> X_train.shape
(55000, 28, 28)
>>> X_train.dtype
dtype('uint8')
```

为简单起见，我们将通过将它们除以255.0来将像素强度缩放到0-1范围（这也将它们转换为浮点数）：

```py
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.
```

对于MNIST，当标签等于5时，这意味着图像代表手写数字5。简单。然而，对于时尚MNIST，我们需要类名列表以了解我们正在处理的内容：

```py
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

例如，训练集中的第一张图像代表一个踝靴：

```py
>>> class_names[y_train[0]]
'Ankle boot'
```

[图10-10](#fashion_mnist_plot)显示了时尚MNIST数据集的一些样本。

![mls3 1010](assets/mls3_1010.png)

###### 图10-10。时尚MNIST的样本

### 使用顺序API创建模型

现在让我们构建神经网络！这是一个具有两个隐藏层的分类MLP：

```py
tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

让我们逐行查看这段代码：

+   首先，设置 TensorFlow 的随机种子以使结果可重现：每次运行笔记本时，隐藏层和输出层的随机权重将保持相同。您还可以选择使用 `tf.keras.utils.set_random_seed()` 函数，它方便地为 TensorFlow、Python (`random.seed()`) 和 NumPy (`np.random.seed()`) 设置随机种子。

+   下一行创建一个 `Sequential` 模型。这是 Keras 模型中最简单的一种，用于仅由一系列按顺序连接的层组成的神经网络。这被称为顺序 API。

+   接下来，我们构建第一层（一个 `Input` 层）并将其添加到模型中。我们指定输入的 `shape`，它不包括批量大小，只包括实例的形状。Keras 需要知道输入的形状，以便确定第一个隐藏层的连接权重矩阵的形状。

+   然后我们添加一个 `Flatten` 层。它的作用是将每个输入图像转换为 1D 数组：例如，如果它接收到一个形状为 [32, 28, 28] 的批量，它将将其重塑为 [32, 784]。换句话说，如果它接收到输入数据 `X`，它会计算 `X.reshape(-1, 784)`。这个层没有任何参数；它只是用来进行一些简单的预处理。

+   接下来我们添加一个具有 300 个神经元的 `Dense` 隐藏层。它将使用 ReLU 激活函数。每个 `Dense` 层都管理着自己的权重矩阵，其中包含神经元与它们的输入之间的所有连接权重。它还管理着一个偏置项向量（每个神经元一个）。当它接收到一些输入数据时，它会计算 [方程 10-2](#neural_network_layer_equation)。

+   然后我们添加一个具有 100 个神经元的第二个 `Dense` 隐藏层，同样使用 ReLU 激活函数。

+   最后，我们添加一个具有 10 个神经元（每个类一个）的 `Dense` 输出层，使用 softmax 激活函数，因为类是互斥的。

###### 提示

指定 `activation="relu"` 等同于指定 `activation=tf.keras.activations.relu`。其他激活函数可以在 `tf.keras.activations` 包中找到。我们将在本书中使用许多这些激活函数；请参阅 [*https://keras.io/api/layers/activations*](https://keras.io/api/layers/activations) 获取完整列表。我们还将在 [第 12 章](ch12.html#tensorflow_chapter) 中定义我们自己的自定义激活函数。

与刚刚逐个添加层不同，通常更方便的做法是在创建 `Sequential` 模型时传递一个层列表。您还可以删除 `Input` 层，而是在第一层中指定 `input_shape`：

```py
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

模型的 `summary()` 方法显示了所有模型的层，包括每个层的名称（除非在创建层时设置了名称，否则会自动生成），其输出形状（`None` 表示批量大小可以是任意值），以及其参数数量。摘要以总参数数量结束，包括可训练和不可训练参数。在这里我们只有可训练参数（您将在本章后面看到一些不可训练参数）：

```py
>>> model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 784)               0

 dense (Dense)               (None, 300)               235500

 dense_1 (Dense)             (None, 100)               30100

 dense_2 (Dense)             (None, 10)                1010

=================================================================
Total params: 266,610
Trainable params: 266,610
Non-trainable params: 0
_________________________________________________________________
```

注意，`Dense` 层通常具有*大量*参数。例如，第一个隐藏层有 784 × 300 个连接权重，再加上 300 个偏置项，总共有 235,500 个参数！这使得模型具有相当大的灵活性来拟合训练数据，但也意味着模型有过拟合的风险，特别是当训练数据不多时。我们稍后会回到这个问题。

模型中的每个层必须具有唯一的名称（例如，`"dense_2"`）。您可以使用构造函数的`name`参数显式设置层名称，但通常最好让Keras自动命名层，就像我们刚刚做的那样。Keras获取层的类名并将其转换为蛇形命名法（例如，`MyCoolLayer`类的层默认命名为`"my_cool_layer"`）。Keras还确保名称在全局范围内是唯一的，即使跨模型也是如此，如果需要，会附加索引，例如`"dense_2"`。但是为什么要确保名称在模型之间是唯一的呢？这样可以轻松合并模型而不会出现名称冲突。

###### 提示

Keras管理的所有全局状态都存储在*Keras会话*中，您可以使用`tf.keras.backend.clear_session()`清除它。特别是，这将重置名称计数器。

您可以使用`layers`属性轻松获取模型的层列表，或使用`get_layer()`方法按名称访问层：

```py
>>> model.layers
[<keras.layers.core.flatten.Flatten at 0x7fa1dea02250>,
 <keras.layers.core.dense.Dense at 0x7fa1c8f42520>,
 <keras.layers.core.dense.Dense at 0x7fa188be7ac0>,
 <keras.layers.core.dense.Dense at 0x7fa188be7fa0>]
>>> hidden1 = model.layers[1]
>>> hidden1.name
'dense'
>>> model.get_layer('dense') is hidden1
True
```

可以使用其`get_weights()`和`set_weights()`方法访问层的所有参数。对于`Dense`层，这包括连接权重和偏差项：

```py
>>> weights, biases = hidden1.get_weights()
>>> weights
array([[ 0.02448617, -0.00877795, -0.02189048, ...,  0.03859074, -0.06889391],
 [ 0.00476504, -0.03105379, -0.0586676 , ..., -0.02763776, -0.04165364],
 ...,
 [ 0.07061854, -0.06960931,  0.07038955, ..., 0.00034875,  0.02878492],
 [-0.06022581,  0.01577859, -0.02585464, ..., 0.00272203, -0.06793761]],
 dtype=float32)
>>> weights.shape
(784, 300)
>>> biases
array([0., 0., 0., 0., 0., 0., 0., 0., 0., ...,  0., 0., 0.], dtype=float32)
>>> biases.shape
(300,)
```

请注意，`Dense`层随机初始化连接权重（这是为了打破对称性，如前所述），偏差初始化为零，这是可以的。如果要使用不同的初始化方法，可以在创建层时设置`kernel_initializer`（*kernel*是连接权重矩阵的另一个名称）或`bias_initializer`。我们将在[第11章](ch11.html#deep_chapter)进一步讨论初始化器，完整列表在[*https://keras.io/api/layers/initializers*](https://keras.io/api/layers/initializers)。

###### 注意

权重矩阵的形状取决于输入的数量，这就是为什么在创建模型时我们指定了`input_shape`。如果您没有指定输入形状，没关系：Keras会等到知道输入形状后才真正构建模型参数。这将在您提供一些数据（例如，在训练期间）或调用其`build()`方法时发生。在模型参数构建之前，您将无法执行某些操作，例如显示模型摘要或保存模型。因此，如果在创建模型时知道输入形状，最好指定它。

### 编译模型

创建模型后，必须调用其`compile()`方法来指定要使用的损失函数和优化器。可选地，您可以指定在训练和评估过程中计算的额外指标列表：

```py
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
```

###### 注意

使用`loss="sparse_categorical_crossentropy"`等同于使用`loss=tf.keras.losses.sparse_categorical_​cross⁠entropy`。同样，使用`optimizer="sgd"`等同于使用`optimizer=tf.keras.optimizers.SGD()`，使用`metrics=["accuracy"]`等同于使用`metrics=​[tf.keras.metrics.sparse_categorical_accuracy`]（使用此损失时）。在本书中，我们将使用许多其他损失、优化器和指标；有关完整列表，请参见[*https://keras.io/api/losses*](https://keras.io/api/losses)、[*https://keras.io/api/optimizers*](https://keras.io/api/optimizers)和[*https://keras.io/api/metrics*](https://keras.io/api/metrics)。

这段代码需要解释。我们使用`"sparse_categorical_crossentropy"`损失，因为我们有稀疏标签（即，对于每个实例，只有一个目标类索引，本例中为0到9），并且类是互斥的。如果相反，对于每个实例有一个目标概率类（例如，独热向量，例如，`[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]`表示类3），那么我们需要使用`"categorical_crossentropy"`损失。如果我们进行二元分类或多标签二元分类，则在输出层中使用`"sigmoid"`激活函数，而不是`"softmax"`激活函数，并且我们将使用`"binary_crossentropy"`损失。

###### 提示

如果你想将稀疏标签（即类别索引）转换为独热向量标签，请使用`tf.keras.utils.to_categorical()`函数。要反过来，使用带有`axis=1`的`np.argmax()`函数。

关于优化器，`"sgd"`表示我们将使用随机梯度下降来训练模型。换句话说，Keras将执行前面描述的反向传播算法（即反向模式自动微分加梯度下降）。我们将在第11章中讨论更高效的优化器。它们改进了梯度下降，而不是自动微分。

###### 注意

当使用`SGD`优化器时，调整学习率是很重要的。因此，通常你会想要使用`optimizer=tf.keras.optimizers.SGD(learning_rate=__???__)`来设置学习率，而不是`optimizer="sgd"`，后者默认学习率为0.01。

最后，由于这是一个分类器，所以在训练和评估过程中测量其准确性是有用的，这就是为什么我们设置`metrics=["accuracy"]`。

### 训练和评估模型

现在模型已经准备好进行训练了。为此，我们只需要调用它的`fit()`方法：

```py
>>> history = model.fit(X_train, y_train, epochs=30,
...                     validation_data=(X_valid, y_valid))
...
Epoch 1/30
1719/1719 [==============================] - 2s 989us/step
 - loss: 0.7220 - sparse_categorical_accuracy: 0.7649
 - val_loss: 0.4959 - val_sparse_categorical_accuracy: 0.8332
Epoch 2/30
1719/1719 [==============================] - 2s 964us/step
 - loss: 0.4825 - sparse_categorical_accuracy: 0.8332
 - val_loss: 0.4567 - val_sparse_categorical_accuracy: 0.8384
[...]
Epoch 30/30
1719/1719 [==============================] - 2s 963us/step
 - loss: 0.2235 - sparse_categorical_accuracy: 0.9200
 - val_loss: 0.3056 - val_sparse_categorical_accuracy: 0.8894
```

我们传递输入特征（`X_train`）和目标类别（`y_train`），以及训练的时期数量（否则默认为1，这绝对不足以收敛到一个好的解决方案）。我们还传递一个验证集（这是可选的）。Keras将在每个时期结束时在这个集合上测量损失和额外的指标，这对于查看模型的实际表现非常有用。如果在训练集上的表现比在验证集上好得多，那么你的模型可能过度拟合训练集，或者存在错误，比如训练集和验证集之间的数据不匹配。

###### 提示

形状错误是非常常见的，特别是在刚开始时，所以你应该熟悉错误消息：尝试用错误形状的输入和/或标签拟合模型，看看你得到的错误。同样，尝试用`loss="categorical_crossentropy"`而不是`loss="sparse_categorical_crossentropy"`来编译模型。或者你可以移除`Flatten`层。

就是这样！神经网络已经训练好了。在训练过程中的每个时期，Keras会在进度条的左侧显示迄今为止处理的小批量数量。批量大小默认为32，由于训练集有55,000张图像，模型每个时期会经过1,719个批次：1,718个大小为32，1个大小为24。在进度条之后，你可以看到每个样本的平均训练时间，以及训练集和验证集上的损失和准确性（或者你要求的任何其他额外指标）。请注意，训练损失下降了，这是一个好迹象，验证准确性在30个时期后达到了88.94%。这略低于训练准确性，所以有一点过拟合，但不是很严重。

###### 提示

不要使用`validation_data`参数传递验证集，你可以将`validation_split`设置为你希望Keras用于验证的训练集比例。例如，`validation_split=0.1`告诉Keras使用数据的最后10%（在洗牌之前）作为验证集。

如果训练集非常倾斜，某些类别过度表示，而其他类别则表示不足，那么在调用 `fit()` 方法时设置 `class_weight` 参数会很有用，以给予少数类别更大的权重，而给予多数类别更小的权重。这些权重将在计算损失时由 Keras 使用。如果需要每个实例的权重，可以设置 `sample_weight` 参数。如果同时提供了 `class_weight` 和 `sample_weight`，那么 Keras 会将它们相乘。每个实例的权重可能很有用，例如，如果一些实例由专家标记，而其他实例使用众包平台标记：你可能希望给前者更多的权重。您还可以为验证集提供样本权重（但不是类别权重），方法是将它们作为 `validation_data` 元组的第三个项目添加。

`fit()` 方法返回一个 `History` 对象，其中包含训练参数 (`history.params`)、经历的每个 epoch 的列表 (`history.epoch`)，最重要的是一个字典 (`history.history`)，其中包含每个 epoch 结束时在训练集和验证集（如果有的话）上测量的损失和额外指标。如果使用这个字典创建一个 Pandas DataFrame，并调用它的 `plot()` 方法，就可以得到 [Figure 10-11](#keras_learning_curves_plot) 中显示的学习曲线：

```py
import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.show()
```

![mls3 1011](assets/mls3_1011.png)

###### 图 10-11\. 学习曲线：每个 epoch 结束时测量的平均训练损失和准确率，以及每个 epoch 结束时测量的平均验证损失和准确率

您可以看到，在训练过程中，训练准确率和验证准确率都在稳步增加，而训练损失和验证损失都在减少。这是好的。验证曲线在开始时相对接近，但随着时间的推移，它们之间的差距变得更大，这表明存在一些过拟合。在这种特殊情况下，模型在训练开始阶段在验证集上的表现似乎比在训练集上好，但实际情况并非如此。验证错误是在 *每个* epoch 结束时计算的，而训练错误是在 *每个* epoch *期间* 使用运行平均值计算的，因此训练曲线应该向左移动半个 epoch。如果这样做，您会看到在训练开始阶段，训练和验证曲线几乎完美重合。

训练集的性能最终会超过验证集的性能，这通常是在训练足够长时间后的情况。你可以看出模型还没有完全收敛，因为验证损失仍在下降，所以你可能应该继续训练。只需再次调用 `fit()` 方法，因为 Keras 会从离开的地方继续训练：你应该能够达到约 89.8% 的验证准确率，而训练准确率将继续上升到 100%（这并不总是情况）。

如果你对模型的性能不满意，你应该回去调整超参数。首先要检查的是学习率。如果这没有帮助，尝试另一个优化器（并在更改任何超参数后重新调整学习率）。如果性能仍然不理想，那么尝试调整模型超参数，如层数、每层神经元的数量以及每个隐藏层要使用的激活函数类型。你也可以尝试调整其他超参数，比如批量大小（可以在`fit()`方法中使用`batch_size`参数设置，默认为32）。我们将在本章末回到超参数调整。一旦你对模型的验证准确率感到满意，你应该在部署模型到生产环境之前在测试集上评估它以估计泛化误差。你可以使用`evaluate()`方法轻松实现这一点（它还支持其他几个参数，如`batch_size`和`sample_weight`；请查看文档以获取更多详细信息）：

```py
>>> model.evaluate(X_test, y_test)
313/313 [==============================] - 0s 626us/step
 - loss: 0.3243 - sparse_categorical_accuracy: 0.8864
[0.32431697845458984, 0.8863999843597412]
```

正如你在[第2章](ch02.html#project_chapter)中看到的，通常在测试集上的性能会略低于验证集，因为超参数是在验证集上调整的，而不是在测试集上（然而，在这个例子中，我们没有进行任何超参数调整，所以较低的准确率只是运气不佳）。记住要抵制在测试集上调整超参数的诱惑，否则你对泛化误差的估计将会过于乐观。

### 使用模型进行预测

现在让我们使用模型的`predict()`方法对新实例进行预测。由于我们没有实际的新实例，我们将只使用测试集的前三个实例：

```py
>>> X_new = X_test[:3]
>>> y_proba = model.predict(X_new)
>>> y_proba.round(2)
array([[0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0.01, 0\.  , 0.02, 0\.  , 0.97],
 [0\.  , 0\.  , 0.99, 0\.  , 0.01, 0\.  , 0\.  , 0\.  , 0\.  , 0\.  ],
 [0\.  , 1\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0\.  ]],
 dtype=float32)
```

对于每个实例，模型会为每个类别（从类别0到类别9）估计一个概率。这类似于Scikit-Learn分类器中`predict_proba()`方法的输出。例如，对于第一幅图像，它估计类别9（踝靴）的概率为96%，类别7（运动鞋）的概率为2%，类别5（凉鞋）的概率为1%，其他类别的概率可以忽略不计。换句话说，它非常确信第一幅图像是鞋类，很可能是踝靴，但也可能是运动鞋或凉鞋。如果你只关心估计概率最高的类别（即使概率很低），那么你可以使用`argmax()`方法来获取每个实例的最高概率类别索引：

```py
>>> import numpy as np
>>> y_pred = y_proba.argmax(axis=-1)
>>> y_pred
array([9, 2, 1])
>>> np.array(class_names)[y_pred]
array(['Ankle boot', 'Pullover', 'Trouser'], dtype='<U11')
```

在这里，分类器实际上正确分类了所有三幅图像（这些图像显示在[图10-12](#fashion_mnist_images_plot)中）：

```py
>>> y_new = y_test[:3]
>>> y_new
array([9, 2, 1], dtype=uint8)
```

![mls3 1012](assets/mls3_1012.png)

###### 图10-12。正确分类的时尚MNIST图像

现在你知道如何使用Sequential API构建、训练和评估分类MLP了。但是回归呢？

## 使用Sequential API构建回归MLP

让我们回到加利福尼亚房屋问题，并使用与之前相同的MLP，由3个每层50个神经元组成的隐藏层，但这次使用Keras构建它。

使用顺序API构建、训练、评估和使用回归MLP与分类问题的操作非常相似。以下代码示例中的主要区别在于输出层只有一个神经元（因为我们只想预测一个值），并且没有使用激活函数，损失函数是均方误差，度量标准是RMSE，我们使用了像Scikit-Learn的`MLPRegressor`一样的Adam优化器。此外，在这个例子中，我们不需要`Flatten`层，而是使用`Normalization`层作为第一层：它执行的操作与Scikit-Learn的`StandardScaler`相同，但必须使用其`adapt()`方法拟合训练数据*之前*调用模型的`fit()`方法。 （Keras还有其他预处理层，将在[第13章](ch13.html#data_chapter)中介绍）。让我们来看一下：

```py
tf.random.set_seed(42)
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
```

###### 注意

当您调用`adapt()`方法时，`Normalization`层会学习训练数据中的特征均值和标准差。然而，当您显示模型的摘要时，这些统计数据被列为不可训练的。这是因为这些参数不受梯度下降的影响。

正如您所看到的，顺序API非常清晰和简单。然而，虽然`Sequential`模型非常常见，但有时构建具有更复杂拓扑结构或多个输入或输出的神经网络是很有用的。为此，Keras提供了功能API。

## 使用功能API构建复杂模型

非顺序神经网络的一个例子是*Wide & Deep*神经网络。这种神经网络架构是由Heng-Tze Cheng等人在2016年的一篇论文中介绍的。它直接连接所有或部分输入到输出层，如[图10-13](#wide_deep_diagram)所示。这种架构使得神经网络能够学习深层模式（使用深层路径）和简单规则（通过短路径）。相比之下，常规的MLP强制所有数据通过完整的层堆栈流动；因此，数据中的简单模式可能会被这一系列转换所扭曲。

![mls3 1013](assets/mls3_1013.png)

###### 图10-13。Wide & Deep神经网络

让我们构建这样一个神经网络来解决加利福尼亚房屋问题：

```py
normalization_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)

model = tf.keras.Model(inputs=[input_], outputs=[output])
```

在高层次上，前五行创建了构建模型所需的所有层，接下来的六行使用这些层就像函数一样从输入到输出，最后一行通过指向输入和输出创建了一个Keras `Model`对象。让我们更详细地看一下这段代码：

+   首先，我们创建五个层：一个`Normalization`层用于标准化输入，两个具有30个神经元的`Dense`层，使用ReLU激活函数，一个`Concatenate`层，以及一个没有任何激活函数的单个神经元的输出层的`Dense`层。

+   接下来，我们创建一个`Input`对象（变量名`input_`用于避免遮蔽Python内置的`input()`函数）。这是模型将接收的输入类型的规范，包括其`shape`和可选的`dtype`，默认为32位浮点数。一个模型实际上可能有多个输入，您很快就会看到。

+   然后，我们像使用函数一样使用`Normalization`层，将其传递给`Input`对象。这就是为什么这被称为功能API。请注意，我们只是告诉Keras应该如何连接这些层；实际上还没有处理任何数据，因为`Input`对象只是一个数据规范。换句话说，它是一个符号输入。这个调用的输出也是符号的：`normalized`不存储任何实际数据，它只是用来构建模型。

+   同样，我们将`normalized`传递给`hidden_layer1`，输出`hidden1`，然后将`hidden1`传递给`hidden_layer2`，输出`hidden2`。

+   到目前为止，我们已经按顺序连接了层，然后使用`concat_layer`将输入和第二个隐藏层的输出连接起来。再次强调，实际数据尚未连接：这都是符号化的，用于构建模型。

+   然后我们将`concat`传递给`output_layer`，这给我们最终的`output`。

+   最后，我们创建一个Keras`Model`，指定要使用的输入和输出。

构建了这个Keras模型之后，一切都和之前一样，所以这里不需要重复：编译模型，调整`Normalization`层，拟合模型，评估模型，并用它进行预测。

但是，如果您想通过宽路径发送一部分特征，并通过深路径发送另一部分特征（可能有重叠），如[图10-14](#multiple_inputs_diagram)所示呢？在这种情况下，一个解决方案是使用多个输入。例如，假设我们想通过宽路径发送五个特征（特征0到4），并通过深路径发送六个特征（特征2到7）。我们可以这样做：

```py
input_wide = tf.keras.layers.Input(shape=[5])  # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6])  # features 2 to 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])
```

![mls3 1014](assets/mls3_1014.png)

###### 图10-14。处理多个输入

在这个例子中，与之前的例子相比，有几点需要注意：

+   每个`Dense`层都是在同一行上创建并调用的。这是一种常见的做法，因为它使代码更简洁而不失清晰度。但是，我们不能对`Normalization`层这样做，因为我们需要对该层进行引用，以便在拟合模型之前调用其`adapt()`方法。

+   我们使用了`tf.keras.layers.concatenate()`，它创建了一个`Concatenate`层，并使用给定的输入调用它。

+   在创建模型时，我们指定了`inputs=[input_wide, input_deep]`，因为有两个输入。

现在我们可以像往常一样编译模型，但是在调用`fit()`方法时，不是传递单个输入矩阵`X_train`，而是必须传递一对矩阵（`X_train_wide, X_train_deep`），每个输入一个。对于`X_valid`，以及在调用`evaluate()`或`predict()`时的`X_test`和`X_new`也是如此：

```py
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit((X_train_wide, X_train_deep), y_train, epochs=20,
                    validation_data=((X_valid_wide, X_valid_deep), y_valid))
mse_test = model.evaluate((X_test_wide, X_test_deep), y_test)
y_pred = model.predict((X_new_wide, X_new_deep))
```

###### 提示

您可以传递一个字典`{"input_wide": X_train_wide, "input_deep": X_train_deep}`，而不是传递一个元组（`X_train_wide, X_train_deep`），如果在创建输入时设置了`name="input_wide"`和`name="input_deep"`。当有多个输入时，这是非常推荐的，可以澄清代码并避免顺序错误。

还有许多用例需要多个输出：

+   任务可能需要这样做。例如，您可能希望在图片中定位和分类主要对象。这既是一个回归任务，也是一个分类任务。

+   同样，您可能有基于相同数据的多个独立任务。当然，您可以为每个任务训练一个神经网络，但在许多情况下，通过训练一个单一神经网络，每个任务一个输出，您将在所有任务上获得更好的结果。这是因为神经网络可以学习数据中对所有任务都有用的特征。例如，您可以对面部图片执行*多任务分类*，使用一个输出来对人的面部表情（微笑，惊讶等）进行分类，另一个输出用于识别他们是否戴眼镜。

+   另一个用例是作为正则化技术（即，一种训练约束，其目标是减少过拟合，从而提高模型的泛化能力）。例如，您可能希望在神经网络架构中添加一个辅助输出（参见[图10-15](#multiple_outputs_diagram)），以确保网络的基础部分自己学到一些有用的东西，而不依赖于网络的其余部分。

![mls3 1015](assets/mls3_1015.png)

###### 图10-15。处理多个输出，在这个例子中添加一个辅助输出进行正则化

添加额外的输出非常容易：我们只需将其连接到适当的层并将其添加到模型的输出列表中。例如，以下代码构建了[图10-15](#multiple_outputs_diagram)中表示的网络：

```py
[...]  # Same as above, up to the main output layer
output = tf.keras.layers.Dense(1)(concat)
aux_output = tf.keras.layers.Dense(1)(hidden2)
model = tf.keras.Model(inputs=[input_wide, input_deep],
                       outputs=[output, aux_output])
```

每个输出都需要自己的损失函数。因此，当我们编译模型时，应该传递一个损失列表。如果我们传递一个单一损失，Keras将假定所有输出都必须使用相同的损失。默认情况下，Keras将计算所有损失并简单地将它们相加以获得用于训练的最终损失。由于我们更关心主要输出而不是辅助输出（因为它仅用于正则化），我们希望给主要输出的损失分配更大的权重。幸运的是，在编译模型时可以设置所有损失权重：

```py
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=("mse", "mse"), loss_weights=(0.9, 0.1), optimizer=optimizer,
              metrics=["RootMeanSquaredError"])
```

###### 提示

您可以传递一个字典`loss={"output": "mse", "aux_output": "mse"}`，而不是传递一个元组`loss=("mse", "mse")`，假设您使用`name="output"`和`name="aux_output"`创建了输出层。就像对于输入一样，这样可以澄清代码并避免在有多个输出时出现错误。您还可以为`loss_weights`传递一个字典。

现在当我们训练模型时，我们需要为每个输出提供标签。在这个例子中，主要输出和辅助输出应该尝试预测相同的事物，因此它们应该使用相同的标签。因此，我们需要传递`(y_train, y_train)`，或者如果输出被命名为`"output"`和`"aux_output"`，则传递一个字典`{"output": y_train, "aux_output": y_train}`，而不是传递`y_train`。对于`y_valid`和`y_test`也是一样的：

```py
norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=20,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid))
)
```

当我们评估模型时，Keras会返回损失的加权和，以及所有单独的损失和指标：

```py
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
```

###### 提示

如果设置`return_dict=True`，那么`evaluate()`将返回一个字典而不是一个大元组。

类似地，`predict()`方法将为每个输出返回预测：

```py
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
```

`predict()`方法返回一个元组，并且没有`return_dict`参数以获得一个字典。但是，您可以使用`model.output_names`创建一个：

```py
y_pred_tuple = model.predict((X_new_wide, X_new_deep))
y_pred = dict(zip(model.output_names, y_pred_tuple))
```

正如您所看到的，您可以使用功能API构建各种架构。接下来，我们将看一下您可以构建Keras模型的最后一种方法。

## 使用子类API构建动态模型

顺序API和功能API都是声明式的：您首先声明要使用哪些层以及它们应该如何连接，然后才能开始向模型提供一些数据进行训练或推断。这有许多优点：模型可以很容易地被保存、克隆和共享；其结构可以被显示和分析；框架可以推断形状并检查类型，因此可以在任何数据通过模型之前尽早捕获错误。调试也相当简单，因为整个模型是一组静态图层。但是反过来也是如此：它是静态的。一些模型涉及循环、变化的形状、条件分支和其他动态行为。对于这种情况，或者如果您更喜欢更具有命令式编程风格，子类API适合您。

使用这种方法，您可以对`Model`类进行子类化，在构造函数中创建所需的层，并在`call()`方法中使用它们执行您想要的计算。例如，创建以下`WideAndDeepModel`类的实例会给我们一个与我们刚刚使用功能API构建的模型等效的模型：

```py
class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)  # needed to support naming the model
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output

model = WideAndDeepModel(30, activation="relu", name="my_cool_model")
```

这个例子看起来与前一个例子相似，只是我们在构造函数中将层的创建与它们在`call()`方法中的使用分开。而且我们不需要创建`Input`对象：我们可以在`call()`方法中使用`input`参数。

现在我们有了一个模型实例，我们可以对其进行编译，调整其归一化层（例如，使用`model.norm_layer_wide.adapt(...)`和`model.norm_​layer_deep.adapt(...)`），拟合它，评估它，并使用它进行预测，就像我们使用功能API一样。

这个API的一个重要区别是，您可以在`call()`方法中包含几乎任何您想要的东西：`for`循环，`if`语句，低级别的TensorFlow操作——您的想象力是唯一的限制（参见[第12章](ch12.html#tensorflow_chapter)）！这使得它成为一个很好的API，特别适用于研究人员尝试新想法。然而，这种额外的灵活性是有代价的：您的模型架构被隐藏在`call()`方法中，因此Keras无法轻松地检查它；模型无法使用`tf.keras.models.clone_model()`进行克隆；当您调用`summary()`方法时，您只会得到一个层列表，而没有关于它们如何连接在一起的任何信息。此外，Keras无法提前检查类型和形状，容易出错。因此，除非您真的需要额外的灵活性，否则您可能应该坚持使用顺序API或功能API。

###### 提示

Keras模型可以像常规层一样使用，因此您可以轻松地将它们组合在一起构建复杂的架构。

现在您知道如何使用Keras构建和训练神经网络，您会想要保存它们！

## 保存和恢复模型

保存训练好的Keras模型就是这么简单：

```py
model.save("my_keras_model", save_format="tf")
```

当您设置`save_format="tf"`时，Keras会使用TensorFlow的*SavedModel*格式保存模型：这是一个目录（带有给定名称），包含多个文件和子目录。特别是，*saved_model.pb*文件包含模型的架构和逻辑，以序列化的计算图形式，因此您不需要部署模型的源代码才能在生产中使用它；SavedModel就足够了（您将在[第12章](ch12.html#tensorflow_chapter)中看到这是如何工作的）。*keras_metadata.pb*文件包含Keras所需的额外信息。*variables*子目录包含所有参数值（包括连接权重、偏差、归一化统计数据和优化器参数），如果模型非常大，可能会分成多个文件。最后，*assets*目录可能包含额外的文件，例如数据样本、特征名称、类名等。默认情况下，*assets*目录为空。由于优化器也被保存了，包括其超参数和可能存在的任何状态，加载模型后，您可以继续训练。

###### 注意

如果设置`save_format="h5"`或使用以*.h5*、*.hdf5*或*.keras*结尾的文件名，则Keras将使用基于HDF5格式的Keras特定格式将模型保存到单个文件中。然而，大多数TensorFlow部署工具需要使用SavedModel格式。

通常会有一个脚本用于训练模型并保存它，以及一个或多个脚本（或Web服务）用于加载模型并用于评估或进行预测。加载模型和保存模型一样简单：

```py
model = tf.keras.models.load_model("my_keras_model")
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
```

您还可以使用`save_weights()`和`load_weights()`来仅保存和加载参数值。这包括连接权重、偏差、预处理统计数据、优化器状态等。参数值保存在一个或多个文件中，例如*my_weights.data-00004-of-00052*，再加上一个索引文件，如*my_weights.index*。

仅保存权重比保存整个模型更快，占用更少的磁盘空间，因此在训练过程中保存快速检查点非常完美。如果您正在训练一个大模型，需要数小时或数天，那么您必须定期保存检查点以防计算机崩溃。但是如何告诉`fit()`方法保存检查点呢？使用回调。

## 使用回调

`fit()`方法接受一个`callbacks`参数，让您可以指定一个对象列表，Keras会在训练之前和之后、每个时代之前和之后，甚至在处理每个批次之前和之后调用它们。例如，`ModelCheckpoint`回调会在训练期间定期保存模型的检查点，默认情况下在每个时代结束时：

```py
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("my_checkpoints",
                                                   save_weights_only=True)
history = model.fit([...], callbacks=[checkpoint_cb])
```

此外，在训练过程中使用验证集时，您可以在创建 `ModelCheckpoint` 时设置 `save_best_only=True`。在这种情况下，它只会在模型在验证集上的表现迄今为止最好时保存您的模型。这样，您就不需要担心训练时间过长和过拟合训练集：只需在训练后恢复最后保存的模型，这将是验证集上的最佳模型。这是实现提前停止的一种方式（在[第4章](ch04.html#linear_models_chapter)中介绍），但它实际上不会停止训练。

另一种方法是使用 `EarlyStopping` 回调。当在一定数量的周期（由 `patience` 参数定义）内在验证集上测量不到进展时，它将中断训练，如果您设置 `restore_best_weights=True`，它将在训练结束时回滚到最佳模型。您可以结合这两个回调来保存模型的检查点，以防计算机崩溃，并在没有进展时提前中断训练，以避免浪费时间和资源并减少过拟合：

```py
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)
history = model.fit([...], callbacks=[checkpoint_cb, early_stopping_cb])
```

由于训练将在没有进展时自动停止（只需确保学习率不要太小，否则可能会一直缓慢进展直到结束），所以可以将周期数设置为一个较大的值。`EarlyStopping` 回调将在 RAM 中存储最佳模型的权重，并在训练结束时为您恢复它们。

###### 提示

在 [`tf.keras.callbacks` 包](https://keras.io/api/callbacks)中还有许多其他回调可用。

如果您需要额外的控制，您可以轻松编写自己的自定义回调。例如，以下自定义回调将在训练过程中显示验证损失和训练损失之间的比率（例如，用于检测过拟合）：

```py
class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"] / logs["loss"]
        print(f"Epoch={epoch}, val/train={ratio:.2f}")
```

正如您可能期望的那样，您可以实现 `on_train_begin()`、`on_train_end()`、`on_epoch_begin()`、`on_epoch_end()`、`on_batch_begin()` 和 `on_batch_end()`。回调也可以在评估和预测期间使用，如果您需要的话（例如，用于调试）。对于评估，您应该实现 `on_test_begin()`、`on_test_end()`、`on_test_batch_begin()` 或 `on_test_batch_end()`，这些方法由 `evaluate()` 调用。对于预测，您应该实现 `on_predict_begin()`、`on_predict_end()`、`on_predict_batch_begin()` 或 `on_predict_batch_end()`，这些方法由 `predict()` 调用。

现在让我们再看看在使用 Keras 时您绝对应该拥有的另一个工具：TensorBoard。

## 使用 TensorBoard 进行可视化

TensorBoard 是一个很棒的交互式可视化工具，您可以使用它来查看训练过程中的学习曲线，比较多次运行之间的曲线和指标，可视化计算图，分析训练统计数据，查看模型生成的图像，将复杂的多维数据投影到 3D 并自动为您进行聚类，*分析*您的网络（即，测量其速度以识别瓶颈），等等！

TensorBoard 在安装 TensorFlow 时会自动安装。但是，您需要一个 TensorBoard 插件来可视化分析数据。如果您按照[*https://homl.info/install*](https://homl.info/install)上的安装说明在本地运行所有内容，那么您已经安装了插件，但如果您在使用 Colab，则必须运行以下命令：

```py
%pip install -q -U tensorboard-plugin-profile
```

要使用TensorBoard，必须修改程序，以便将要可视化的数据输出到称为*事件文件*的特殊二进制日志文件中。每个二进制数据记录称为*摘要*。TensorBoard服务器将监视日志目录，并自动捕捉更改并更新可视化：这使您能够可视化实时数据（有短暂延迟），例如训练期间的学习曲线。通常，您希望将TensorBoard服务器指向一个根日志目录，并配置程序，使其在每次运行时写入不同的子目录。这样，同一个TensorBoard服务器实例将允许您可视化和比较程序的多次运行中的数据，而不会混淆一切。

让我们将根日志目录命名为*my_logs*，并定义一个小函数，根据当前日期和时间生成日志子目录的路径，以便在每次运行时都不同：

```py
from pathlib import Path
from time import strftime

def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

run_logdir = get_run_logdir()  # e.g., my_logs/run_2022_08_01_17_25_59
```

好消息是，Keras提供了一个方便的`TensorBoard()`回调，它会为您创建日志目录（以及必要时的父目录），并在训练过程中创建事件文件并写入摘要。它将测量模型的训练和验证损失和指标（在本例中是MSE和RMSE），还会对神经网络进行分析。使用起来很简单：

```py
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir,
                                                profile_batch=(100, 200))
history = model.fit([...], callbacks=[tensorboard_cb])
```

就是这样！在这个例子中，它将在第一个时期的100和200批之间对网络进行分析。为什么是100和200？嗯，神经网络通常需要几批数据来“热身”，所以你不希望太早进行分析，而且分析会使用资源，最好不要为每一批数据都进行分析。

接下来，尝试将学习率从0.001更改为0.002，然后再次运行代码，使用一个新的日志子目录。你将得到一个类似于这样的目录结构：

```py
my_logs
├── run_2022_08_01_17_25_59
│   ├── train
│   │   ├── events.out.tfevents.1659331561.my_host_name.42042.0.v2
│   │   ├── events.out.tfevents.1659331562.my_host_name.profile-empty
│   │   └── plugins
│   │       └── profile
│   │           └── 2022_08_01_17_26_02
│   │               ├── my_host_name.input_pipeline.pb
│   │               └── [...]
│   └── validation
│       └── events.out.tfevents.1659331562.my_host_name.42042.1.v2
└── run_2022_08_01_17_31_12
    └── [...]

```

每次运行都有一个目录，每个目录包含一个用于训练日志和一个用于验证日志的子目录。两者都包含事件文件，而训练日志还包括分析跟踪。

现在你已经准备好事件文件，是时候启动TensorBoard服务器了。可以直接在Jupyter或Colab中使用TensorBoard的Jupyter扩展来完成，该扩展会随TensorBoard库一起安装。这个扩展在Colab中是预安装的。以下代码加载了TensorBoard的Jupyter扩展，第二行启动了一个TensorBoard服务器，连接到这个服务器并直接在Jupyter中显示用户界面。服务器会监听大于或等于6006的第一个可用TCP端口（或者您可以使用`--port`选项设置您想要的端口）。

```py
%load_ext tensorboard
%tensorboard --logdir=./my_logs
```

###### 提示

如果你在自己的机器上运行所有内容，可以通过在终端中执行`tensorboard --logdir=./my_logs`来启动TensorBoard。您必须首先激活安装了TensorBoard的Conda环境，并转到*handson-ml3*目录。一旦服务器启动，访问[*http://localhost:6006*](http://localhost:6006)。

现在你应该看到TensorBoard的用户界面。点击SCALARS选项卡查看学习曲线（参见[图10-16](#tensorboard_diagram)）。在左下角，选择要可视化的日志（例如第一次和第二次运行的训练日志），然后点击`epoch_loss`标量。注意，训练损失在两次运行期间都很好地下降了，但在第二次运行中，由于更高的学习率，下降速度稍快。

![mls3 1016](assets/mls3_1016.png)

###### 图10-16。使用TensorBoard可视化学习曲线

您还可以在GRAPHS选项卡中可视化整个计算图，在PROJECTOR选项卡中将学习的权重投影到3D中，在PROFILE选项卡中查看性能跟踪。`TensorBoard()`回调还有选项可以记录额外的数据（请参阅文档以获取更多详细信息）。您可以点击右上角的刷新按钮（⟳）使TensorBoard刷新数据，也可以点击设置按钮（⚙）激活自动刷新并指定刷新间隔。

此外，TensorFlow在`tf.summary`包中提供了一个较低级别的API。以下代码使用`create_file_writer()`函数创建一个`SummaryWriter`，并将此写入器用作Python上下文来记录标量、直方图、图像、音频和文本，所有这些都可以使用TensorBoard进行可视化：

```py
test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(str(test_logdir))
with writer.as_default():
    for step in range(1, 1000 + 1):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)

        data = (np.random.randn(100) + 2) * step / 100  # gets larger
        tf.summary.histogram("my_hist", data, buckets=50, step=step)

        images = np.random.rand(2, 32, 32, 3) * step / 1000  # gets brighter
        tf.summary.image("my_images", images, step=step)

        texts = ["The step is " + str(step), "Its square is " + str(step ** 2)]
        tf.summary.text("my_text", texts, step=step)

        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)
```

如果您运行此代码并在TensorBoard中点击刷新按钮，您将看到几个选项卡出现：IMAGES、AUDIO、DISTRIBUTIONS、HISTOGRAMS和TEXT。尝试点击IMAGES选项卡，并使用每个图像上方的滑块查看不同时间步的图像。同样，转到AUDIO选项卡并尝试在不同时间步听音频。正如您所看到的，TensorBoard甚至在TensorFlow或深度学习之外也是一个有用的工具。

###### 提示

您可以通过将结果发布到[*https://tensorboard.dev*](https://tensorboard.dev)来在线共享您的结果。为此，只需运行`!tensorboard dev upload` `--logdir` `./my_logs`。第一次运行时，它会要求您接受条款和条件并进行身份验证。然后您的日志将被上传，您将获得一个永久链接，以在TensorBoard界面中查看您的结果。

让我们总结一下你在本章学到的内容：你现在知道神经网络的起源，MLP是什么以及如何将其用于分类和回归，如何使用Keras的顺序API构建MLP，以及如何使用功能API或子类API构建更复杂的模型架构（包括Wide & Deep模型，以及具有多个输入和输出的模型）。您还学会了如何保存和恢复模型，以及如何使用回调函数进行检查点、提前停止等。最后，您学会了如何使用TensorBoard进行可视化。您已经可以开始使用神经网络来解决许多问题了！但是，您可能想知道如何选择隐藏层的数量、网络中的神经元数量以及所有其他超参数。让我们现在来看看这个问题。

# 微调神经网络超参数

神经网络的灵活性也是它们的主要缺点之一：有许多超参数需要调整。不仅可以使用任何想象得到的网络架构，甚至在基本的MLP中，您可以更改层的数量、每层中要使用的神经元数量和激活函数的类型、权重初始化逻辑、要使用的优化器类型、学习率、批量大小等。您如何知道哪种超参数组合对您的任务最好？

一种选择是将您的Keras模型转换为Scikit-Learn估计器，然后使用`GridSearchCV`或`RandomizedSearchCV`来微调超参数，就像您在[第2章](ch02.html#project_chapter)中所做的那样。为此，您可以使用SciKeras库中的`KerasRegressor`和`KerasClassifier`包装类（有关更多详细信息，请参阅[*https://github.com/adriangb/scikeras*](https://github.com/adriangb/scikeras)）。但是，还有一种更好的方法：您可以使用*Keras Tuner*库，这是一个用于Keras模型的超参数调整库。它提供了几种调整策略，可以高度定制，并且与TensorBoard有很好的集成。让我们看看如何使用它。

如果您按照[*https://homl.info/install*](https://homl.info/install)中的安装说明在本地运行所有内容，那么您已经安装了 Keras Tuner，但如果您使用 Colab，则需要运行 `%pip install -q -U keras-tuner`。接下来，导入 `keras_tuner`，通常为 `kt`，然后编写一个函数来构建、编译并返回一个 Keras 模型。该函数必须接受一个 `kt.HyperParameters` 对象作为参数，它可以用来定义超参数（整数、浮点数、字符串等）以及它们可能的取值范围，这些超参数可以用来构建和编译模型。例如，以下函数构建并编译了一个用于分类时尚 MNIST 图像的 MLP，使用超参数如隐藏层的数量（`n_hidden`）、每层神经元的数量（`n_neurons`）、学习率（`learning_rate`）和要使用的优化器类型（`optimizer`）：

```py
import keras_tuner as kt

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model
```

函数的第一部分定义了超参数。例如，`hp.Int("n_hidden", min_value=0, max_value=8, default=2)` 检查了名为 `"n_hidden"` 的超参数是否已经存在于 `hp` 的 `HyperParameters` 对象中，如果存在，则返回其值。如果不存在，则注册一个新的整数超参数，名为 `"n_hidden"`，其可能的取值范围从 0 到 8（包括边界），并返回默认值，在本例中默认值为 2（当未设置 `default` 时，返回 `min_value`）。 `"n_neurons"` 超参数以类似的方式注册。 `"learning_rate"` 超参数注册为一个浮点数，范围从 10^(-4) 到 10^(-2)，由于 `sampling="log"`，所有尺度的学习率将被等概率采样。最后，`optimizer` 超参数注册了两个可能的值："sgd" 或 "adam"（默认值是第一个，本例中为 "sgd"）。根据 `optimizer` 的值，我们创建一个具有给定学习率的 `SGD` 优化器或 `Adam` 优化器。

函数的第二部分只是使用超参数值构建模型。它创建一个 `Sequential` 模型，从一个 `Flatten` 层开始，然后是请求的隐藏层数量（由 `n_hidden` 超参数确定）使用 ReLU 激活函数，以及一个具有 10 个神经元（每类一个）的输出层，使用 softmax 激活函数。最后，函数编译模型并返回它。

现在，如果您想进行基本的随机搜索，可以创建一个 `kt.RandomSearch` 调谐器，将 `build_model` 函数传递给构造函数，并调用调谐器的 `search()` 方法：

```py
random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)
random_search_tuner.search(X_train, y_train, epochs=10,
                           validation_data=(X_valid, y_valid))
```

`RandomSearch` 调谐器首先使用一个空的 `Hyperparameters` 对象调用 `build_model()` 一次，以收集所有超参数规范。然后，在这个例子中，它运行 5 个试验；对于每个试验，它使用在其各自范围内随机抽样的超参数构建一个模型，然后对该模型进行 10 个周期的训练，并将其保存到 *my_fashion_mnist/my_rnd_search* 目录的子目录中。由于 `overwrite=True`，在训练开始之前 *my_rnd_search* 目录将被删除。如果您再次运行此代码，但使用 `overwrite=False` 和 `max_trials=10`，调谐器将继续从上次停止的地方进行调谐，运行 5 个额外的试验：这意味着您不必一次性运行所有试验。最后，由于 `objective` 设置为 `"val_accuracy"`，调谐器更喜欢具有更高验证准确性的模型，因此一旦调谐器完成搜索，您可以像这样获取最佳模型：

```py
top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]
```

您还可以调用 `get_best_hyperparameters()` 来获取最佳模型的 `kt.HyperParameters`：

```py
>>> top3_params = random_search_tuner.get_best_hyperparameters(num_trials=3)
>>> top3_params[0].values  # best hyperparameter values
{'n_hidden': 5,
 'n_neurons': 70,
 'learning_rate': 0.00041268008323824807,
 'optimizer': 'adam'}
```

每个调谐器都由一个所谓的*oracle*指导：在每次试验之前，调谐器会询问oracle告诉它下一个试验应该是什么。`RandomSearch`调谐器使用`RandomSearchOracle`，它非常基本：就像我们之前看到的那样，它只是随机选择下一个试验。由于oracle跟踪所有试验，您可以要求它给出最佳试验，并显示该试验的摘要：

```py
>>> best_trial = random_search_tuner.oracle.get_best_trials(num_trials=1)[0]
>>> best_trial.summary()
Trial summary
Hyperparameters:
n_hidden: 5
n_neurons: 70
learning_rate: 0.00041268008323824807
optimizer: adam
Score: 0.8736000061035156
```

这显示了最佳超参数（与之前一样），以及验证准确率。您也可以直接访问所有指标：

```py
>>> best_trial.metrics.get_last_value("val_accuracy")
0.8736000061035156
```

如果您对最佳模型的性能感到满意，您可以在完整的训练集（`X_train_full`和`y_train_full`）上继续训练几个时期，然后在测试集上评估它，并将其部署到生产环境（参见[第19章](ch19.html#deployment_chapter)）：

```py
best_model.fit(X_train_full, y_train_full, epochs=10)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
```

在某些情况下，您可能希望微调数据预处理超参数或`model.fit()`参数，比如批量大小。为此，您必须使用略有不同的技术：而不是编写一个`build_model()`函数，您必须子类化`kt.HyperModel`类并定义两个方法，`build()`和`fit()`。`build()`方法执行与`build_model()`函数完全相同的操作。`fit()`方法接受一个`HyperParameters`对象和一个已编译的模型作为参数，以及所有`model.fit()`参数，并拟合模型并返回`History`对象。关键是，`fit()`方法可以使用超参数来决定如何预处理数据，调整批量大小等。例如，以下类构建了与之前相同的模型，具有相同的超参数，但它还使用一个布尔型`"normalize"`超参数来控制是否在拟合模型之前标准化训练数据：

```py
class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_model(hp)

    def fit(self, hp, model, X, y, **kwargs):
        if hp.Boolean("normalize"):
            norm_layer = tf.keras.layers.Normalization()
            X = norm_layer(X)
        return model.fit(X, y, **kwargs)
```

然后，您可以将此类的实例传递给您选择的调谐器，而不是传递`build_model`函数。例如，让我们基于`MyClassificationHyperModel`实例构建一个`kt.Hyperband`调谐器：

```py
hyperband_tuner = kt.Hyperband(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42,
    max_epochs=10, factor=3, hyperband_iterations=2,
    overwrite=True, directory="my_fashion_mnist", project_name="hyperband")
```

这个调谐器类似于我们在[第2章](ch02.html#project_chapter)中讨论的`HalvingRandomSearchCV`类：它首先为少数时期训练许多不同的模型，然后消除最差的模型，仅保留前`1 / factor`个模型（在这种情况下是前三分之一），重复此选择过程，直到只剩下一个模型。`max_epochs`参数控制最佳模型将被训练的最大时期数。在这种情况下，整个过程重复两次（`hyperband_iterations=2`）。每个超带迭代中所有模型的总训练时期数约为`max_epochs * (log(max_epochs) / log(factor)) ** 2`，因此在这个例子中大约为44个时期。其他参数与`kt.RandomSearch`相同。

现在让我们运行Hyperband调谐器。我们将使用`TensorBoard`回调，这次指向根日志目录（调谐器将负责为每个试验使用不同的子目录），以及一个`EarlyStopping`回调：

```py
root_logdir = Path(hyperband_tuner.project_dir) / "tensorboard"
tensorboard_cb = tf.keras.callbacks.TensorBoard(root_logdir)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2)
hyperband_tuner.search(X_train, y_train, epochs=10,
                       validation_data=(X_valid, y_valid),
                       callbacks=[early_stopping_cb, tensorboard_cb])
```

现在，如果您打开TensorBoard，将`--logdir`指向*my_fashion_mnist/hyperband/tensorboard*目录，您将看到所有试验结果的展示。确保访问HPARAMS选项卡：它包含了所有尝试过的超参数组合的摘要，以及相应的指标。请注意，在HPARAMS选项卡内部有三个选项卡：表格视图、平行坐标视图和散点图矩阵视图。在左侧面板的下部，取消选中除了`validation.epoch_accuracy`之外的所有指标：这将使图表更清晰。在平行坐标视图中，尝试选择`validation.epoch_accuracy`列中的高值范围：这将仅显示达到良好性能的超参数组合。单击其中一个超参数组合，相应的学习曲线将出现在页面底部。花些时间浏览每个选项卡；这将帮助您了解每个超参数对性能的影响，以及超参数之间的相互作用。

Hyperband比纯随机搜索更聪明，因为它分配资源的方式更为高效，但在其核心部分仍然是随机探索超参数空间；它快速，但粗糙。然而，Keras Tuner还包括一个`kt.BayesianOptimization`调谐器：这种算法通过拟合一个称为*高斯过程*的概率模型逐渐学习哪些超参数空间区域最有前途。这使得它逐渐聚焦于最佳超参数。缺点是该算法有自己的超参数：`alpha`代表您在试验中期望性能指标中的噪声水平（默认为10^(–4)），`beta`指定您希望算法探索而不仅仅利用已知的超参数空间中的良好区域（默认为2.6）。除此之外，这个调谐器可以像之前的调谐器一样使用：

```py
bayesian_opt_tuner = kt.BayesianOptimization(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42,
    max_trials=10, alpha=1e-4, beta=2.6,
    overwrite=True, directory="my_fashion_mnist", project_name="bayesian_opt")
bayesian_opt_tuner.search([...])
```

超参数调整仍然是一个活跃的研究领域，许多其他方法正在被探索。例如，查看DeepMind出色的[2017年论文](https://homl.info/pbt)，其中作者使用进化算法共同优化了一组模型和它们的超参数。谷歌也采用了进化方法，不仅用于搜索超参数，还用于探索各种模型架构：它为谷歌Vertex AI上的AutoML服务提供动力（参见[第19章](ch19.html#deployment_chapter)）。术语*AutoML*指的是任何系统，它负责ML工作流的大部分。进化算法甚至已成功用于训练单个神经网络，取代了无处不在的梯度下降！例如，查看Uber在[2017年发布的文章](https://homl.info/neuroevol)，作者介绍了他们的*Deep Neuroevolution*技术。

尽管有这些令人兴奋的进展和所有这些工具和服务，但仍然有必要了解每个超参数的合理值，以便您可以构建一个快速原型并限制搜索空间。以下部分提供了选择MLP中隐藏层和神经元数量以及选择一些主要超参数的良好值的指导方针。

## 隐藏层的数量

对于许多问题，您可以从一个隐藏层开始并获得合理的结果。具有一个隐藏层的MLP在理论上可以建模甚至最复杂的函数，只要它有足够的神经元。但对于复杂问题，深度网络比浅层网络具有更高的*参数效率*：它们可以使用指数级较少的神经元来建模复杂函数，从而使它们在相同数量的训练数据下达到更好的性能。

要理解为什么，假设您被要求使用绘图软件画一片森林，但是禁止复制和粘贴任何东西。这将需要大量的时间：您必须逐个绘制每棵树，一枝一枝，一叶一叶。如果您可以绘制一片叶子，复制并粘贴它以绘制一根树枝，然后复制并粘贴该树枝以创建一棵树，最后复制并粘贴这棵树以制作一片森林，您将很快完成。现实世界的数据通常以这种分层方式结构化，深度神经网络会自动利用这一事实：较低的隐藏层模拟低级结构（例如各种形状和方向的线段），中间隐藏层将这些低级结构组合起来模拟中级结构（例如正方形、圆形），最高隐藏层和输出层将这些中级结构组合起来模拟高级结构（例如人脸）。

这种分层结构不仅有助于深度神经网络更快地收敛到一个好的解决方案，而且还提高了它们对新数据集的泛化能力。例如，如果您已经训练了一个模型来识别图片中的人脸，现在想要训练一个新的神经网络来识别发型，您可以通过重用第一个网络的较低层来启动训练。而不是随机初始化新神经网络的前几层的权重和偏置，您可以将它们初始化为第一个网络较低层的权重和偏置的值。这样网络就不必从头学习出现在大多数图片中的所有低级结构；它只需要学习更高级的结构（例如发型）。这就是所谓的*迁移学习*。

总之，对于许多问题，您可以从只有一个或两个隐藏层开始，神经网络就能正常工作。例如，您可以仅使用一个具有几百个神经元的隐藏层在MNIST数据集上轻松达到 97% 以上的准确率，使用两个具有相同总神经元数量的隐藏层在大致相同的训练时间内达到 98% 以上的准确率。对于更复杂的问题，您可以增加隐藏层的数量，直到开始过拟合训练集。非常复杂的任务，例如大型图像分类或语音识别，通常需要具有数十层（甚至数百层，但不是全连接的，如您将在[第14章](ch14.html#cnn_chapter)中看到的）的网络，并且需要大量的训练数据。您很少需要从头开始训练这样的网络：更常见的做法是重用执行类似任务的预训练最先进网络的部分。这样训练速度会更快，需要的数据量也会更少（我们将在[第11章](ch11.html#deep_chapter)中讨论这一点）。

## 隐藏层中的神经元数量

输入层和输出层的神经元数量取决于您的任务所需的输入和输出类型。例如，MNIST任务需要 28 × 28 = 784 个输入和 10 个输出神经元。

至于隐藏层，过去常见的做法是将它们大小设计成金字塔形，每一层的神经元数量越来越少——其理由是许多低级特征可以融合成远远较少的高级特征。一个典型的用于MNIST的神经网络可能有 3 个隐藏层，第一个有 300 个神经元，第二个有 200 个，第三个有 100 个。然而，这种做法已经被大多数人放弃，因为似乎在大多数情况下，在所有隐藏层中使用相同数量的神经元表现得同样好，甚至更好；此外，只需调整一个超参数，而不是每一层一个。尽管如此，根据数据集的不同，有时将第一个隐藏层设计得比其他隐藏层更大可能会有所帮助。

就像层数一样，您可以尝试逐渐增加神经元的数量，直到网络开始过拟合。或者，您可以尝试构建一个比实际需要的层数和神经元稍多一点的模型，然后使用提前停止和其他正则化技术来防止过度拟合。Google的科学家Vincent Vanhoucke将此称为“伸展裤”方法：不要浪费时间寻找完全符合您尺寸的裤子，只需使用大号伸展裤，它们会缩小到合适的尺寸。通过这种方法，您可以避免可能破坏模型的瓶颈层。实际上，如果一层的神经元太少，它将没有足够的表征能力来保留来自输入的所有有用信息（例如，具有两个神经元的层只能输出2D数据，因此如果它以3D数据作为输入，一些信息将丢失）。无论网络的其余部分有多大和强大，该信息都将永远无法恢复。

###### 提示

一般来说，增加层数而不是每层的神经元数量会更有效。

## 学习率、批量大小和其他超参数

隐藏层和神经元的数量并不是您可以在MLP中调整的唯一超参数。以下是一些最重要的超参数，以及如何设置它们的提示：

学习率

学习率可以说是最重要的超参数。一般来说，最佳学习率约为最大学习率的一半（即训练算法发散的学习率上限，如我们在第4章中看到的）。找到一个好的学习率的方法是训练模型几百次迭代，从非常低的学习率（例如，10^（-5））开始，逐渐增加到非常大的值（例如，10）。这是通过在每次迭代时将学习率乘以一个常数因子来完成的（例如，通过（10 / 10^（-5））^（1 / 500）在500次迭代中从10^（-5）增加到10）。如果将损失作为学习率的函数绘制出来（使用对数刻度的学习率），您应该会看到它一开始下降。但过一段时间，学习率将变得太大，因此损失会迅速上升：最佳学习率将略低于损失开始上升的点（通常比转折点低约10倍）。然后，您可以重新初始化您的模型，并使用这个好的学习率进行正常训练。我们将在第11章中探讨更多学习率优化技术。

优化器

选择比普通的小批量梯度下降更好的优化器（并调整其超参数）也非常重要。我们将在第11章中研究几种高级优化器。

批量大小

批量大小可能会对模型的性能和训练时间产生重大影响。使用大批量大小的主要好处是硬件加速器如GPU可以高效处理它们（参见[第19章](ch19.html#deployment_chapter)），因此训练算法将每秒看到更多实例。因此，许多研究人员和从业者建议使用能够适应GPU RAM的最大批量大小。然而，有一个问题：在实践中，大批量大小通常会导致训练不稳定，特别是在训练开始时，由此产生的模型可能不会像使用小批量大小训练的模型那样泛化得好。2018年4月，Yann LeCun甚至在推特上发表了“朋友们不要让朋友们使用大于32的小批量”的言论，引用了Dominic Masters和Carlo Luschi在[2018年的一篇论文](https://homl.info/smallbatch)的结论，该论文认为使用小批量（从2到32）更可取，因为小批量在更短的训练时间内产生更好的模型。然而，其他研究结果却指向相反的方向。例如，2017年，Elad Hoffer等人的论文和Priya Goyal等人的论文显示，可以使用非常大的批量大小（高达8,192），并结合各种技术，如学习率预热（即从小学习率开始训练，然后逐渐增加，如[第11章](ch11.html#deep_chapter)中讨论的那样），以获得非常短的训练时间，而不会出现泛化差距。因此，一种策略是尝试使用大批量大小，结合学习率预热，如果训练不稳定或最终性能令人失望，则尝试改用小批量大小。

激活函数

我们在本章前面讨论了如何选择激活函数：一般来说，ReLU激活函数将是所有隐藏层的一个很好的默认选择，但对于输出层，它真的取决于您的任务。

迭代次数

在大多数情况下，实际上不需要调整训练迭代次数：只需使用早停止即可。

###### 提示

最佳学习率取决于其他超参数，尤其是批量大小，因此如果您修改任何超参数，请确保同时更新学习率。

有关调整神经网络超参数的最佳实践，请查看Leslie Smith的优秀[2018年论文](https://homl.info/1cycle)。

这结束了我们关于人工神经网络及其在Keras中的实现的介绍。在接下来的几章中，我们将讨论训练非常深的网络的技术。我们还将探讨如何使用TensorFlow的低级API自定义模型，以及如何使用tf.data API高效加载和预处理数据。我们将深入研究其他流行的神经网络架构：用于图像处理的卷积神经网络，用于序列数据和文本的循环神经网络和transformers，用于表示学习的自动编码器，以及用于建模和生成数据的生成对抗网络。

# 练习

1.  [TensorFlow playground](https://playground.tensorflow.org)是由TensorFlow团队构建的一个方便的神经网络模拟器。在这个练习中，您将只需点击几下就可以训练几个二元分类器，并调整模型的架构和超参数，以便对神经网络的工作原理和超参数的作用有一些直观的认识。花一些时间来探索以下内容：

    1.  神经网络学习的模式。尝试通过点击运行按钮（左上角）训练默认的神经网络。注意到它如何快速找到分类任务的良好解决方案。第一个隐藏层中的神经元已经学会了简单的模式，而第二个隐藏层中的神经元已经学会了将第一个隐藏层的简单模式组合成更复杂的模式。一般来说，层数越多，模式就越复杂。

    1.  激活函数。尝试用ReLU激活函数替换tanh激活函数，并重新训练网络。注意到它找到解决方案的速度更快，但这次边界是线性的。这是由于ReLU函数的形状。

    1.  局部最小值的风险。修改网络架构，只有一个有三个神经元的隐藏层。多次训练它（要重置网络权重，点击播放按钮旁边的重置按钮）。注意到训练时间变化很大，有时甚至会卡在局部最小值上。

    1.  当神经网络太小时会发生什么。移除一个神经元，只保留两个。注意到神经网络现在无法找到一个好的解决方案，即使你尝试多次。模型参数太少，系统地欠拟合训练集。

    1.  当神经网络足够大时会发生什么。将神经元数量设置为八，并多次训练网络。注意到现在训练速度一致快速，从不卡住。这突显了神经网络理论中的一个重要发现：大型神经网络很少会卡在局部最小值上，即使卡住了，这些局部最优解通常几乎和全局最优解一样好。然而，它们仍然可能在长时间的高原上卡住。

    1.  深度网络中梯度消失的风险。选择螺旋数据集（“DATA”下方的右下数据集），并将网络架构更改为每个有八个神经元的四个隐藏层。注意到训练时间更长，经常在高原上卡住很长时间。还要注意到最高层（右侧）的神经元比最低层（左侧）的神经元进化得更快。这个问题被称为*梯度消失*问题，可以通过更好的权重初始化和其他技术、更好的优化器（如AdaGrad或Adam）或批量归一化（在[第11章](ch11.html#deep_chapter)中讨论）来缓解。

    1.  更进一步。花一个小时左右的时间玩弄其他参数，了解它们的作用，建立对神经网络的直观理解。

1.  使用原始人工神经元（如[图10-3](#nn_propositional_logic_diagram)中的人工神经元）绘制一个ANN，计算*A* ⊕ *B*（其中 ⊕ 表示异或操作）。提示：*A* ⊕ *B* = (*A* ∧ ¬ *B*) ∨ (¬ *A* ∧ *B*)。

1.  通常更倾向于使用逻辑回归分类器而不是经典感知器（即使用感知器训练算法训练的阈值逻辑单元的单层）。如何调整感知器使其等效于逻辑回归分类器？

1.  为什么Sigmoid激活函数是训练第一个MLP的关键因素？

1.  列出三种流行的激活函数。你能画出它们吗？

1.  假设你有一个MLP，由一个具有10个传递神经元的输入层、一个具有50个人工神经元的隐藏层和一个具有3个人工神经元的输出层组成。所有人工神经元都使用ReLU激活函数。

    1.  输入矩阵**X**的形状是什么？

    1.  隐藏层权重矩阵**W**[*h*]和偏置向量**b**[*h*]的形状是什么？

    1.  输出层权重矩阵**W**[*o*]和偏置向量**b**[*o*]的形状是什么？

    1.  网络输出矩阵**Y**的形状是什么？

    1.  写出计算网络输出矩阵**Y**的方程，作为**X**、**W**[*h*]、**b**[*h*]、**W**[*o*]和**b**[*o*]的函数。

1.  如果你想将电子邮件分类为垃圾邮件或正常邮件，输出层需要多少个神经元？输出层应该使用什么激活函数？如果你想处理MNIST数据集，输出层需要多少个神经元，应该使用哪种激活函数？对于让你的网络预测房价，如[第2章](ch02.html#project_chapter)中所述，需要多少个神经元，应该使用什么激活函数？

1.  什么是反向传播，它是如何工作的？反向传播和反向模式自动微分之间有什么区别？

1.  在基本的MLP中，你可以调整哪些超参数？如果MLP过拟合训练数据，你可以如何调整这些超参数来尝试解决问题？

1.  在MNIST数据集上训练一个深度MLP（可以使用`tf.keras.datasets.mnist.load_data()`加载）。看看你是否可以通过手动调整超参数获得超过98%的准确率。尝试使用本章介绍的方法搜索最佳学习率（即通过指数增长学习率，绘制损失曲线，并找到损失飙升的点）。接下来，尝试使用Keras Tuner调整超参数，包括保存检查点、使用早停止，并使用TensorBoard绘制学习曲线。

这些练习的解决方案可以在本章笔记本的末尾找到，网址为[*https://homl.info/colab3*](https://homl.info/colab3)。

^([1](ch10.html#idm45720205300576-marker)) 你可以通过对生物启发开放，而不害怕创建生物不现实的模型，来获得两全其美，只要它们运行良好。

^([2](ch10.html#idm45720205291952-marker)) Warren S. McCulloch和Walter Pitts，“神经活动中固有思想的逻辑演算”，《数学生物学公报》5卷4期（1943年）：115-113。

^([3](ch10.html#idm45720205271104-marker)) 它们实际上并没有连接，只是非常接近，可以非常快速地交换化学信号。

^([4](ch10.html#idm45720205264656-marker)) Bruce Blaus绘制的图像（[知识共享3.0](https://creativecommons.org/licenses/by/3.0)）。来源：[*https://en.wikipedia.org/wiki/Neuron*](https://en.wikipedia.org/wiki/Neuron)。

^([5](ch10.html#idm45720205261008-marker)) 在机器学习的背景下，“神经网络”一词通常指的是人工神经网络，而不是生物神经网络。

^([6](ch10.html#idm45720205254384-marker)) S. Ramon y Cajal绘制的皮层层析图（公有领域）。来源：[*https://en.wikipedia.org/wiki/Cerebral_cortex*](https://en.wikipedia.org/wiki/Cerebral_cortex)。

^([7](ch10.html#idm45720205089040-marker)) 请注意，这个解决方案并不唯一：当数据点线性可分时，有无穷多个可以将它们分开的超平面。

^([8](ch10.html#idm45720204942848-marker)) 例如，当输入为（0，1）时，左下神经元计算0 × 1 + 1 × 1 - 3 / 2 = -1 / 2，为负数，因此输出为0。右下神经元计算0 × 1 + 1 × 1 - 1 / 2 = 1 / 2，为正数，因此输出为1。输出神经元接收前两个神经元的输出作为输入，因此计算0 × (-1) + 1 × 1 - 1 / 2 = 1 / 2。这是正数，因此输出为1。

^([9](ch10.html#idm45720204921696-marker)) 在20世纪90年代，具有两个以上隐藏层的人工神经网络被认为是深度的。如今，常见的是看到具有数十层甚至数百层的人工神经网络，因此“深度”的定义非常模糊。

^([10](ch10.html#idm45720204912752-marker)) 大卫·鲁梅尔哈特等人，“通过误差传播学习内部表示”（国防技术信息中心技术报告，1985年9月）。

^([11](ch10.html#idm45720204883120-marker)) 生物神经元似乎实现了一个大致呈S形的激活函数，因此研究人员长时间坚持使用Sigmoid函数。但事实证明，在人工神经网络中，ReLU通常效果更好。这是生物类比可能误导的一个案例。

^([12](ch10.html#idm45720204597504-marker)) ONEIROS 项目（开放式神经电子智能机器人操作系统）。Chollet 在 2015 年加入了谷歌，继续领导 Keras 项目。

^([13](ch10.html#idm45720204591712-marker)) PyTorch 的 API 与 Keras 的相似，因此一旦你了解了 Keras，如果你想要的话，切换到 PyTorch 并不困难。PyTorch 在 2018 年的普及程度呈指数增长，这在很大程度上要归功于其简单性和出色的文档，而这些正是 TensorFlow 1.x 当时的主要弱点。然而，TensorFlow 2 和 PyTorch 一样简单，部分原因是它已经将 Keras 作为其官方高级 API，并且开发人员大大简化和清理了其余的 API。文档也已经完全重新组织，现在更容易找到所需的内容。同样，PyTorch 的主要弱点（例如，有限的可移植性和没有计算图分析）在 PyTorch 1.0 中已经得到了很大程度的解决。健康的竞争对每个人都有益。

^([14](ch10.html#idm45720204105760-marker)) 您还可以使用 `tf.keras.utils.plot_model()` 生成模型的图像。

^([15](ch10.html#idm45720203290048-marker)) Heng-Tze Cheng 等人，[“广泛和深度学习用于推荐系统”](https://homl.info/widedeep)，*第一届深度学习推荐系统研讨会论文集*（2016）：7–10。

^([16](ch10.html#idm45720203287488-marker)) 短路径也可以用于向神经网络提供手动设计的特征。

^([17](ch10.html#idm45720201814912-marker)) Keras 模型有一个 `output` 属性，所以我们不能将其用作主输出层的名称，这就是为什么我们将其重命名为 `main_output`。

^([18](ch10.html#idm45720201799248-marker)) 目前这是默认设置，但 Keras 团队正在研究一种可能成为未来默认设置的新格式，因此我更喜欢明确设置格式以保证未来兼容。

^([19](ch10.html#idm45720200186752-marker)) Hyperband 实际上比连续减半法更复杂；参见 Lisha Li 等人的[论文](https://homl.info/hyperband)，“Hyperband: 一种新颖的基于贝叶斯的超参数优化方法”，*机器学习研究杂志* 18（2018年4月）：1–52。

^([20](ch10.html#idm45720199979472-marker)) Max Jaderberg 等人，“神经网络的基于人口的训练”，arXiv 预印本 arXiv:1711.09846（2017）。

^([21](ch10.html#idm45720199898400-marker)) Dominic Masters 和 Carlo Luschi，“重新审视深度神经网络的小批量训练”，arXiv 预印本 arXiv:1804.07612（2018）。

^([22](ch10.html#idm45720199896896-marker)) Elad Hoffer 等人，“训练时间更长，泛化效果更好：弥合神经网络大批量训练的泛化差距”，*第31届国际神经信息处理系统会议论文集*（2017）：1729–1739。

^([23](ch10.html#idm45720199894992-marker)) Priya Goyal 等人，“准确、大型小批量 SGD：在1小时内训练 ImageNet”，arXiv 预印本 arXiv:1706.02677（2017）。

^([24](ch10.html#idm45720199885792-marker)) Leslie N. Smith，“神经网络超参数的纪律性方法：第1部分—学习率、批量大小、动量和权重衰减”，arXiv 预印本 arXiv:1803.09820（2018）。

^([25](ch10.html#idm45720199882240-marker)) 在[*https://homl.info/extra-anns*](https://homl.info/extra-anns)的在线笔记本中还介绍了一些额外的人工神经网络架构。
