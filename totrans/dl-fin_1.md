# 第2章。深度学习的基本概率方法

技术的兴起和可访问性使每个人都能够部署机器学习和深度学习算法进行数据分析和优化。但不幸的是，这意味着许多用户不了解不同学习模型的基础和基础知识。这使得机器学习对他们来说简直就是一个黑匣子，这是一场灾难的前兆。

理解和掌握概率、统计和数学的基本概念对于理解和掌握数据以及创建试图解释和预测数据的模型至关重要。本章介绍了理解不同学习算法所需的数值概念的基础知识，或者至少展示了你可以从哪里开始建立知识以掌握这些数学主题。

为简单起见，本书中使用的术语*机器学习*指的是所有类型的学习模型（如机器学习、深度学习和强化学习）。

# 概率入门

*概率*是描述随机变量和随机事件的一切。世界充满了随机性，找到穿越混乱的最佳方法是尝试使用概率方法来解释它。诚然，短语*解释混乱*可能是一个矛盾修饰语，因为混乱实际上无法解释，但我们人类无法放弃对不确定事件的控制，随着进步，我们已经开发出工具来理解这个可怕的世界。

你可能会想知道在尝试为金融交易开发机器学习算法时理解概率基础的用途是什么。这是一个合理的问题，你必须知道，一个学科的基础不一定与它相似。

例如，要成为一名飞行员，你必须首先学习空气动力学，其中充满了与毕业后获得的最终技能不相似的技术概念。这与本章中所做的类似；通过学习概率基础知识、统计概念和数学主题，你将开始走上成为机器学习开发人员的正确道路。

了解你正在学习的东西的用途应该会给你动力。以下是一些对机器学习重要的关键概率主题：

概率分布函数

*概率分布*描述了随机变量的各种结果的可能性。对于许多机器学习技术来说，理解典型概率分布的特征和属性是至关重要的。概率分布函数还描述了不同类型的时间序列数据，从而有助于选择正确的算法。

贝叶斯定理用于更新概率

贝叶斯定理是概率论的基石，提供了一种在新数据的光下更新事件概率的方法。它被纳入到各种机器学习技术中，包括贝叶斯网络和分类器。

假设检验

*假设检验*用于确定基于数据样本的人口断言更可能是真实还是错误的。许多机器学习模型在其过程中使用假设检验。

决策树

*决策树*是一种借鉴了条件概率等概率概念的机器学习算法，本章涵盖了这个概念。更详细的内容，决策树在第7章中有详细介绍。

信息论

*信息论*是关于信息如何被量化、存储和传输的复杂研究。它被纳入到许多机器学习技术中，包括决策树。

# 概率概念简介

概率信息中最基本的部分是*随机变量*，它是一个不确定的数字或结果。随机变量用于模拟被认为是不确定的事件，例如货币对未来回报。

随机变量要么是离散的，要么是连续的。*离散随机变量*具有有限的值集，而*连续随机变量*具有在某个区间内的值。考虑以下两个例子以澄清事情：

+   离散随机变量的一个例子是掷骰子的结果。它们受以下集合的限制{1, 2, 3, 4, 5, 6}。

+   连续随机变量的一个例子是EURUSD的每日价格回报（1欧元兑换美元的汇率）。

随机变量由*概率分布*描述，这是给出这些随机变量的每个可能值的概率的函数。通常，直方图用于显示概率。直方图绘制将在本章后面讨论。

在任何时刻，某个事件发生的概率在0和1之间。这意味着概率被分配给随机变量，范围在0到1之间，其中概率为0表示发生的机会为零，概率为1表示发生的确定性。

您也可以将其以百分比表示，范围从0%到100%。这两个数字之间的值是有效的，这意味着某个事件发生的概率可以是0.5133（51.33%）。考虑掷一个有六个面的骰子。在不以任何方式操纵骰子的情况下，得到3的概率是多少？

由于骰子有六个面，每个结果有六个相等的概率，这意味着对于任何结果，概率如下找到：

<math alttext="upper P left-parenthesis x right-parenthesis equals one-sixth equals 0.167"><mrow><mi>P</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mn>6</mn></mfrac></mstyle> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>167</mn></mrow></math>

用*P(x)*表示事件*x*的概率。这给出了问题的答案：

<math alttext="upper P left-parenthesis 3 right-parenthesis equals one-sixth equals 0.167"><mrow><mi>P</mi> <mrow><mo>(</mo> <mn>3</mn> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mn>6</mn></mfrac></mstyle> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>167</mn></mrow></math>

掷骰子时，只能有一个结果。它不能同时给出3和4，因为一面必须占优势。这就是*互斥性*的概念。互斥事件（例如在掷骰子时得到3或得到4）最终总和为1。看看以下例子：

<math alttext="upper P left-parenthesis 1 right-parenthesis equals one-sixth equals 0.167"><mrow><mi>P</mi> <mrow><mo>(</mo> <mn>1</mn> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mn>6</mn></mfrac></mstyle> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>167</mn></mrow></math>

<math alttext="upper P left-parenthesis 2 right-parenthesis equals one-sixth equals 0.167"><mrow><mi>P</mi> <mrow><mo>(</mo> <mn>2</mn> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mn>6</mn></mfrac></mstyle> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>167</mn></mrow></math>

<math alttext="upper P left-parenthesis 3 right-parenthesis equals one-sixth equals 0.167"><mrow><mi>P</mi> <mrow><mo>(</mo> <mn>3</mn> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mn>6</mn></mfrac></mstyle> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>167</mn></mrow></math>

<math alttext="upper P left-parenthesis 4 right-parenthesis equals one-sixth equals 0.167"><mrow><mi>P</mi> <mrow><mo>(</mo> <mn>4</mn> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mn>6</mn></mfrac></mstyle> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>167</mn></mrow></math>

<math alttext="upper P left-parenthesis 5 right-parenthesis equals one-sixth equals 0.167"><mrow><mi>P</mi> <mrow><mo>(</mo> <mn>5</mn> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mn>6</mn></mfrac></mstyle> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>167</mn></mrow></math>

<math alttext="upper P left-parenthesis 6 right-parenthesis equals one-sixth equals 0.167"><mrow><mi>P</mi> <mrow><mo>(</mo> <mn>6</mn> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mn>6</mn></mfrac></mstyle> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>167</mn></mrow></math>

将所有这些互斥事件相加得到1，这意味着六面骰子中可能概率的总和如下：

<math alttext="upper P left-parenthesis 1 right-parenthesis plus upper P left-parenthesis 2 right-parenthesis plus upper P left-parenthesis 3 right-parenthesis plus upper P left-parenthesis 4 right-parenthesis plus upper P left-parenthesis 5 right-parenthesis plus upper P left-parenthesis 6 right-parenthesis equals 1"><mrow><mi>P</mi> <mo>(</mo> <mn>1</mn> <mo>)</mo> <mo>+</mo> <mi>P</mi> <mo>(</mo> <mn>2</mn> <mo>)</mo> <mo>+</mo> <mi>P</mi> <mo>(</mo> <mn>3</mn> <mo>)</mo> <mo>+</mo> <mi>P</mi> <mo>(</mo> <mn>4</mn> <mo>)</mo> <mo>+</mo> <mi>P</mi> <mo>(</mo> <mn>5</mn> <mo>)</mo> <mo>+</mo> <mi>P</mi> <mo>(</mo> <mn>6</mn> <mo>)</mo> <mo>=</mo> <mn>1</mn></mrow></math>

###### 注

声明随机变量有0.8的发生概率等同于声明相同变量有0.2的不发生概率。

概率测量可以是条件的或无条件的。*条件概率*是指事件发生影响另一个事件发生的概率。例如，鉴于积极的就业数据，主权利率上涨的概率是条件概率的一个例子。给定事件B发生的情况下事件A发生的概率由以下数学符号表示：*P(A|B)*

相比之下，*无条件概率*不依赖于其他事件。以条件概率为例，您可以制定一个无条件概率计算，该计算测量利率上涨的概率，而不考虑其他经济事件。

概率具有特定的加法和乘法规则，具有自己的解释。在看例子之前，让我们先看一下公式。两个事件实现的*联合概率*是它们都发生的概率。它是使用以下公式计算的：

<math alttext="upper P left-parenthesis upper A upper B right-parenthesis equals upper P left-parenthesis upper A vertical-bar upper B right-parenthesis times upper P left-parenthesis upper B right-parenthesis"><mrow><mi>P</mi> <mo>(</mo> <mi>A</mi> <mi>B</mi> <mo>)</mo> <mo>=</mo> <mi>P</mi> <mo>(</mo> <mi>A</mi> <mo>|</mo> <mi>B</mi> <mo>)</mo> <mo>×</mo> <mi>P</mi> <mo>(</mo> <mi>B</mi> <mo>)</mo></mrow></math>

该公式表示的是A和B同时发生的概率是A在B发生的情况下发生的概率乘以B发生的概率。因此，等式的右侧将一个条件概率乘以一个无条件概率。

*加法规则*用于确定至少发生两个结果中的一个的概率。这有两种方式：第一种处理互斥事件，第二种处理非互斥事件：

如果事件不是互斥的，则为避免重复计数，公式如下：

<math alttext="upper P left-parenthesis upper A o r upper B right-parenthesis equals upper P left-parenthesis upper A right-parenthesis plus upper P left-parenthesis upper B right-parenthesis minus upper P left-parenthesis upper A upper B right-parenthesis"><mrow><mi>P</mi> <mo>(</mo> <mi>A</mi> <mi>o</mi> <mi>r</mi> <mi>B</mi> <mo>)</mo> <mo>=</mo> <mi>P</mi> <mo>(</mo> <mi>A</mi> <mo>)</mo> <mo>+</mo> <mi>P</mi> <mo>(</mo> <mi>B</mi> <mo>)</mo> <mo>-</mo> <mi>P</mi> <mo>(</mo> <mi>A</mi> <mi>B</mi> <mo>)</mo></mrow></math>

如果事件是互斥的，则公式简化为以下形式：

<math alttext="upper P left-parenthesis upper A upper B right-parenthesis equals 0"><mrow><mi>P</mi> <mo>(</mo> <mi>A</mi> <mi>B</mi> <mo>)</mo> <mo>=</mo> <mn>0</mn></mrow></math>

<math alttext="upper P left-parenthesis upper A o r upper B right-parenthesis equals upper P left-parenthesis upper A right-parenthesis plus upper P left-parenthesis upper B right-parenthesis minus 0"><mrow><mi>P</mi> <mo>(</mo> <mi>A</mi> <mi>o</mi> <mi>r</mi> <mi>B</mi> <mo>)</mo> <mo>=</mo> <mi>P</mi> <mo>(</mo> <mi>A</mi> <mo>)</mo> <mo>+</mo> <mi>P</mi> <mo>(</mo> <mi>B</mi> <mo>)</mo> <mo>-</mo> <mn>0</mn></mrow></math>

<math alttext="upper P left-parenthesis upper A o r upper B right-parenthesis equals upper P left-parenthesis upper A right-parenthesis plus upper P left-parenthesis upper B right-parenthesis"><mrow><mi>P</mi> <mo>(</mo> <mi>A</mi> <mi>o</mi> <mi>r</mi> <mi>B</mi> <mo>)</mo> <mo>=</mo> <mi>P</mi> <mo>(</mo> <mi>A</mi> <mo>)</mo> <mo>+</mo> <mi>P</mi> <mo>(</mo> <mi>B</mi> <mo>)</mo></mrow></math>

注意，在互斥事件中，只能实现A或B，因此两者都发生的概率为零。要理解为什么需要减去A和B的联合概率，请看图2-1。

![](Images/dlf_0333.png)

###### 图2-1。概率的加法规则

注意，当A或B发生的概率互斥时，必须不包括它们的联合概率。现在让我们来看看独立事件的概念。

*独立事件*不相互关联（例如，掷骰子两次）。联合概率计算如下：

<math alttext="upper P left-parenthesis upper A upper B right-parenthesis equals upper P left-parenthesis upper A right-parenthesis times upper P left-parenthesis upper B right-parenthesis"><mrow><mi>P</mi> <mo>(</mo> <mi>A</mi> <mi>B</mi> <mo>)</mo> <mo>=</mo> <mi>P</mi> <mo>(</mo> <mi>A</mi> <mo>)</mo> <mo>×</mo> <mi>P</mi> <mo>(</mo> <mi>B</mi> <mo>)</mo></mrow></math>

因此，独立事件指的是一个事件的发生对其他事件的发生绝对没有影响。现在，让我们看一个例子来验证这些概念。考虑一个简单的抛硬币。得到正面的概率不取决于您在上一个抛硬币中得到的结果。因此，得到正面的概率始终为0.50（50%）。更进一步，连续五次抛硬币后只得到正面的概率是多少？

由于每个事件的概率与前一个或下一个事件是独立的，因此公式如下：

<math alttext="upper P left-parenthesis x right-parenthesis equals 0.50 times 0.50 times 0.50 times 0.50 times 0.50 equals 0.03125 equals 3.125 percent-sign"><mrow><mi>P</mi> <mo>(</mo> <mi>x</mi> <mo>)</mo> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>50</mn> <mo>×</mo> <mn>0</mn> <mo>.</mo> <mn>50</mn> <mo>×</mo> <mn>0</mn> <mo>.</mo> <mn>50</mn> <mo>×</mo> <mn>0</mn> <mo>.</mo> <mn>50</mn> <mo>×</mo> <mn>0</mn> <mo>.</mo> <mn>50</mn> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>03125</mn> <mo>=</mo> <mn>3</mn> <mo>.</mo> <mn>125</mn> <mo>%</mo></mrow></math>

随机变量的*期望值*是不同结果的加权平均值。因此，期望值实际上是指均值的另一种方式。在数学上，期望值如下：

<math alttext="upper E left-parenthesis upper X right-parenthesis equals sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis upper P left-parenthesis x Subscript i Baseline right-parenthesis x Subscript i Baseline right-parenthesis"><mrow><mi>E</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>)</mo></mrow> <mo>=</mo> <msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></msubsup> <mrow><mo>(</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></math>

看一下表2-1，尝试计算一年中某个月的下一个就业人数的期望值。

表2-1。就业表

| 非农就业人数 | 概率 |
| --- | --- |
| 300,000 | 0.1 |
| 400,000 | 0.3 |
| 500,000 | 0.5 |
| 600,000 | 0.1 |

*非农就业人数*是指美国劳工部发布的每月报告，提供有关全国有薪雇员总数的信息，不包括农业部门的从业人员以及政府和非营利组织的从业人员。

从表2-1中，经济学家假设有50%的概率总有薪员工人数增加50万，有30%的概率总有薪员工人数增加40万。因此期望值为：

<math alttext="upper E left-parenthesis upper X right-parenthesis equals left-parenthesis 300 comma 000 times 0.1 right-parenthesis plus left-parenthesis 400 comma 000 times 0.3 right-parenthesis plus left-parenthesis 500 comma 000 times 0.5 right-parenthesis plus left-parenthesis 600 comma 000 times 0.1 right-parenthesis equals 460 comma 000"><mrow><mi>E</mi> <mo>(</mo> <mi>X</mi> <mo>)</mo> <mo>=</mo> <mo>(</mo> <mn>300</mn> <mo>,</mo> <mn>000</mn> <mo>×</mo> <mn>0</mn> <mo>.</mo> <mn>1</mn> <mo>)</mo> <mo>+</mo> <mo>(</mo> <mn>400</mn> <mo>,</mo> <mn>000</mn> <mo>×</mo> <mn>0</mn> <mo>.</mo> <mn>3</mn> <mo>)</mo> <mo>+</mo> <mo>(</mo> <mn>500</mn> <mo>,</mo> <mn>000</mn> <mo>×</mo> <mn>0</mn> <mo>.</mo> <mn>5</mn> <mo>)</mo> <mo>+</mo> <mo>(</mo> <mn>600</mn> <mo>,</mo> <mn>000</mn> <mo>×</mo> <mn>0</mn> <mo>.</mo> <mn>1</mn> <mo>)</mo> <mo>=</mo> <mn>460</mn> <mo>,</mo> <mn>000</mn></mrow></math>

因此，代表经济学家共识的数字是460,000，因为它是大多数预测的最接近加权值。它是代表数据集的值。

在介绍概率的部分结束之前，存在一个称为*贝叶斯定理*的数学公式，根据先前相关事件的知识估计事件的可能性。贝叶斯定理的公式如下：

<math alttext="upper P left-parenthesis upper A vertical-bar upper B right-parenthesis equals StartFraction upper P left-parenthesis upper B vertical-bar upper A right-parenthesis period upper P left-parenthesis upper A right-parenthesis Over upper P left-parenthesis upper B right-parenthesis EndFraction"><mrow><mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>|</mo> <mi>B</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>P</mi><mo>(</mo><mi>B</mi><mo>|</mo><mi>A</mi><mo>)</mo><mo>.</mo><mi>P</mi><mo>(</mo><mi>A</mi><mo>)</mo></mrow> <mrow><mi>P</mi><mo>(</mo><mi>B</mi><mo>)</mo></mrow></mfrac></mrow></math>

其中：

+   *P(A|B)*是在事件B发生的情况下事件A发生的概率。

+   *P(B|A)*是在事件A发生的情况下事件B发生的概率。

+   *P(A)*是事件A发生的概率。

+   *P(B)*是事件B发生的概率。

换句话说，贝叶斯定理允许您根据新信息更新对事件概率的信念。

###### 注意

本节的主要要点如下：

+   概率描述随机变量和随机事件。它是介于0和1之间的值。

+   事件的概率可以组合在一起形成更复杂的情景。

+   预期结果是指定宇宙中每个概率的加权平均值。

# 抽样和假设检验

当人口很大时，会采取代表性样本，使其成为数据的主要描述者。以美国为例。其民主制度意味着人民有权决定自己的命运，但不可能去问每个人关于每个话题的详细意见。这就是为什么要举行选举并选举代表，以便他们代表人民行事。

*抽样*指的是在更大的人口中选择数据样本并对人口的统计特性做出结论的行为。有几种不同的抽样方法。最常见的是以下几种：

简单随机抽样

在简单随机抽样中，人口中的每个元素被选入样本的机会是相等的。这可以是在一个标记的人口中生成的随机数，其中每个个体被选中的概率相同。

分层抽样

使用分层抽样，人口根据某些特征分成组，然后从每个组中按比例随机抽取一个简单随机样本。

集群抽样

使用集群抽样，人口被分成集群，然后从中选择一个随机样本。然后，所选集群中的所有元素都包括在样本中。

系统抽样

使用系统抽样，通过从人口中每隔*n*个个体选择一个元素来选择一个元素，其中*n*是一个固定数字。这意味着它不是随机的，而是事先指定的。

一个经验法则是，你获得的数据越多，度量就越能反映人口。在机器学习领域，抽样非常重要，因为很多时候，你正在抽取数据样本来代表真实的人口。例如，在对策略进行回测时，你需要将整个数据集分成一个*训练样本*和一个*测试样本*，其中第一个是算法了解其结构的数据样本，第二个是算法测试其预测能力的数据样本。

类似地，另一个使用抽样的例子是*交叉验证*，这是一种将数据集分成两个或更多子组的技术。模型使用一个子集进行训练，然后使用其他子集进行测试。对于数据的各种子集，这个过程会重复多次，然后确定模型的平均性能。

这些术语将在接下来的章节中更深入地讨论。现在你应该明白，在机器学习中（甚至在使用优化技术的深度学习中），抽样的概念非常重要。

抽样并不完美，可能会出现错误，就像任何其他估计方法一样。*抽样误差*指的是样本的统计量与人口的统计量之间的差异（如果已知）。*统计量*是描述分析数据集的度量（一个例子是均值，在第三章中关于统计学的更详细内容中会看到）。那么，你应该有多少最小样本量才能对人口进行推断呢？经验法则是至少有30个观察值，越多越好。这将引出*中心极限定理*，该定理指出从人口中抽取的随机样本将在样本变大时逼近正态分布（一种对称的钟形概率分布）。

中心极限定理使得推断和结论的应用变得简单，因为假设检验与正态分布相配。在进行假设检验之前，让我们看看置信区间，即预期总体参数的值范围。置信区间通常通过从点估计值中加减一个因子来构建。例如，给定一个样本均值 x̄，可以构建一个置信区间如下：

<math alttext="x overbar plus-or-minus left-parenthesis r e l i a b i l i t y f a c t o r times s t a n d a r d e r r o r right-parenthesis"><mrow><mover accent="true"><mi>x</mi> <mo>¯</mo></mover> <mo>±</mo> <mrow><mo>(</mo> <mi>r</mi> <mi>e</mi> <mi>l</mi> <mi>i</mi> <mi>a</mi> <mi>b</mi> <mi>i</mi> <mi>l</mi> <mi>i</mi> <mi>t</mi> <mi>y</mi> <mi>f</mi> <mi>a</mi> <mi>c</mi> <mi>t</mi> <mi>o</mi> <mi>r</mi> <mo>×</mo> <mi>s</mi> <mi>t</mi> <mi>a</mi> <mi>n</mi> <mi>d</mi> <mi>a</mi> <mi>r</mi> <mi>d</mi> <mi>e</mi> <mi>r</mi> <mi>r</mi> <mi>o</mi> <mi>r</mi> <mo>)</mo></mrow></mrow></math>

让我们逐步理解计算过程。样本均值是对总体的估计，并且是因为不可能计算总体均值而计算的，因此，通过进行随机抽样，假设样本均值应该等于总体均值。然而，在现实生活中，事情可能会有所不同，这就是为什么应该使用概率方法构建置信区间的原因。

###### 注意

显著性水平是置信区间的阈值。例如，95%的置信区间意味着有95%的置信度，估计值应该落在某个范围内。剩下的5%概率不在这个范围内，称为显著性水平（通常用α符号表示）。

可靠性因子是一个统计量，取决于估计值的分布以及它落入置信区间的概率。为了简单起见，让我们假设总体的方差是正态的，总体是正态分布的。对于5%的显著性水平（因此，置信区间为95%），在这种情况下可靠性因子为1.96（获得这个数字的方式与讨论的内容不太相关）。

标准误差是样本的标准差。标准差在第3章中有更深入的讨论；现在只需知道它代表了围绕均值的不同值的波动。标准误差使用以下公式计算：

<math alttext="s equals StartFraction sigma Over StartRoot n EndRoot EndFraction"><mrow><mi>s</mi> <mo>=</mo> <mfrac><mi>σ</mi> <msqrt><mi>n</mi></msqrt></mfrac></mrow></math>

<math alttext="sigma i s t h e p o p u l a t i o n s t a n d a r d d e v i a t i o n"><mrow><mi>σ</mi> <mi>i</mi> <mi>s</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>p</mi> <mi>o</mi> <mi>p</mi> <mi>u</mi> <mi>l</mi> <mi>a</mi> <mi>t</mi> <mi>i</mi> <mi>o</mi> <mi>n</mi> <mi>s</mi> <mi>t</mi> <mi>a</mi> <mi>n</mi> <mi>d</mi> <mi>a</mi> <mi>r</mi> <mi>d</mi> <mi>d</mi> <mi>e</mi> <mi>v</mi> <mi>i</mi> <mi>a</mi> <mi>t</mi> <mi>i</mi> <mi>o</mi> <mi>n</mi></mrow></math>

<math alttext="StartRoot n EndRoot i s t h e s q u a r e r o o t o f t h e p o p u l a t i o n n u m b e r"><mrow><msqrt><mi>n</mi></msqrt> <mi>i</mi> <mi>s</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>s</mi> <mi>q</mi> <mi>u</mi> <mi>a</mi> <mi>r</mi> <mi>e</mi> <mi>r</mi> <mi>o</mi> <mi>o</mi> <mi>t</mi> <mi>o</mi> <mi>f</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>p</mi> <mi>o</mi> <mi>p</mi> <mi>u</mi> <mi>l</mi> <mi>a</mi> <mi>t</mi> <mi>i</mi> <mi>o</mi> <mi>n</mi> <mi>n</mi> <mi>u</mi> <mi>m</mi> <mi>b</mi> <mi>e</mi> <mi>r</mi></mrow></math>

值得知道的是，对于1%的显著性水平，可靠性因子为2.575，对于10%的显著性水平，可靠性因子为1.645。让我们通过一个实际例子来理解所有这些数学。

考虑一个包含100种金融工具（债券、货币对、股票、结构化产品等）的总体。这些工具的年平均回报率为1.4%。假设总体标准差为4.34%，在1%显著性水平（99%置信区间）下，均值的置信区间是多少？

答案就是将数值代入以下公式：

<math alttext="1.4 percent-sign plus-or-minus 2.575 times StartFraction 4.34 percent-sign Over StartRoot 100 EndRoot EndFraction equals 1.4 percent-sign plus-or-minus 1.11 percent-sign"><mrow><mn>1</mn> <mo>.</mo> <mn>4</mn> <mo>%</mo> <mo>±</mo> <mn>2</mn> <mo>.</mo> <mn>575</mn> <mo>×</mo> <mfrac><mrow><mn>4</mn><mo>.</mo><mn>34</mn><mo>%</mo></mrow> <msqrt><mn>100</mn></msqrt></mfrac> <mo>=</mo> <mn>1</mn> <mo>.</mo> <mn>4</mn> <mo>%</mo> <mo>±</mo> <mn>1</mn> <mo>.</mo> <mn>11</mn> <mo>%</mo></mrow></math>

这意味着置信区间在（0.29%，2.51%）之间。

让我们看另一个例子。假设贵金属和工业金属（如黄金和铜）的年回报率是正态分布的，均值为3.5%，已知总体标准差为5.1%。在10%显著性水平下，5种不同商品的年回报率的置信区间是多少？答案如下：

<math alttext="3.5 percent-sign plus-or-minus 1.645 times StartFraction 3.5 percent-sign Over StartRoot 5 EndRoot EndFraction equals 3.5 percent-sign plus-or-minus 2.23 percent-sign"><mrow><mn>3</mn> <mo>.</mo> <mn>5</mn> <mo>%</mo> <mo>±</mo> <mn>1</mn> <mo>.</mo> <mn>645</mn> <mo>×</mo> <mfrac><mrow><mn>3</mn><mo>.</mo><mn>5</mn><mo>%</mo></mrow> <msqrt><mn>5</mn></msqrt></mfrac> <mo>=</mo> <mn>3</mn> <mo>.</mo> <mn>5</mn> <mo>%</mo> <mo>±</mo> <mn>2</mn> <mo>.</mo> <mn>23</mn> <mo>%</mo></mrow></math>

这意味着置信区间在（1.27%，5.8%）之间。

###### 注意

如果样本量较小和/或总体标准差未知，则 t-分布可能比正态分布更好。

t-分布是一种概率分布，用于模拟样本均值的分布，当样本量较小和/或总体标准差未知时。它在形状上类似于正态分布，但尾部更重，这代表了与较小样本量相关的不确定性。

在结束有关抽样和估计的讨论之前，以下列表显示了根据总体特征给出的适当分布：

+   具有已知方差的小型正态分布应使用正态分布的可靠性因子。

+   具有已知方差的大型正态分布应使用正态分布的可靠性因子。

+   具有未知方差的小正态分布应使用t分布的可靠性因子。

+   具有未知方差的大正态分布应使用t分布的可靠性因子。

+   具有已知方差的大非正态分布应使用正态分布的可靠性因子。

+   具有已知方差的大非正态分布应使用t分布的可靠性因子。

请记住，*大*意味着观察数量大于30。前一个列表中未涵盖的组合是复杂的，超出了本讨论的范围。

下一步是假设检验，这是一种从数据样本中得出结论的关键概率技术。这部分非常重要，因为它几乎用于所有类型的统计分析和模型中。

在统计学中，*假设检验*是一种从少量数据样本中得出关于总体的结论的技术。它涉及制定两个竞争性假设，*零假设*和*备择假设*，关于总体参数，然后使用样本数据确定哪个更有可能是准确的。

例如，金融分析师正在从风险角度评估两个投资组合。他们制定了两个假设：

+   零假设表明两个投资组合的波动性没有显著差异。

+   备择假设表明两个投资组合的波动性存在显著差异。

然后使用统计分析来测试假设，以确定波动性差异是否具有统计学意义或纯粹是由于偶然因素。

根据零假设和备择假设的定义，使用样本数据计算一个检验统计量。为了评估结果的显著性，然后将检验统计量与从标准分布中抽取的临界值进行比较。如果检验统计量在关键区域内，则拒绝零假设并接受备择假设。如果检验统计量不在关键区域内，则不拒绝零假设，并得出支持备择假设的证据不足的结论。

这些都是花言巧语，说的其实是假设检验基本上是创建两种相反的情景，进行概率检查，然后决定哪种情景更有可能是真实的。假设检验可以采取两种形式：

+   *单侧检验*：一个例子是测试某些金融工具的回报是否大于零。

+   *双侧检验*：一个例子是测试某些金融工具的回报是否与零不同（意味着可以大于或小于零）。

###### 注意

假设检验通常是双侧的。

零假设是您希望拒绝的假设，因此希望被拒绝并接受备择方案。双侧检验采用以下一般形式：

<math alttext="upper H 0 colon x equals x 0"><mrow><msub><mi>H</mi> <mn>0</mn></msub> <mo>:</mo> <mi>x</mi> <mo>=</mo> <msub><mi>x</mi> <mn>0</mn></msub></mrow></math>

<math alttext="upper H Subscript a Baseline colon x not-equals x 0"><mrow><msub><mi>H</mi> <mi>a</mi></msub> <mo>:</mo> <mi>x</mi> <mo>≠</mo> <msub><mi>x</mi> <mn>0</mn></msub></mrow></math>

由于备择方案允许数值在零上下两侧（这是零假设中规定的水平），应该有两个临界值。因此，双侧检验的规则是，如果检验统计量大于上临界值或小于下临界值，则拒绝零假设。例如，对于正态分布的数据，检验统计量与临界值（在5%显著性水平下）进行比较，分别为+1.96和-1.96。如果检验统计量落在+1.96和-1.96之间的范围之外，则拒绝零假设。

假设检验的过程包括计算检验统计量。通过将总体参数的点估计与零假设的假设值进行比较来计算它。然后，两者都通过样本的标准误差进行缩放。数学表示如下：

<math alttext="t e s t s t a t i s t i c equals StartFraction s a m p l e s t a t i s t i c minus h y p o t h e s i z e d v a l u e Over s t a n d a r d e r r o r EndFraction"><mrow><mi>t</mi> <mi>e</mi> <mi>s</mi> <mi>t</mi> <mi>s</mi> <mi>t</mi> <mi>a</mi> <mi>t</mi> <mi>i</mi> <mi>s</mi> <mi>t</mi> <mi>i</mi> <mi>c</mi> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mrow><mi>s</mi><mi>a</mi><mi>m</mi><mi>p</mi><mi>l</mi><mi>e</mi><mi>s</mi><mi>t</mi><mi>a</mi><mi>t</mi><mi>i</mi><mi>s</mi><mi>t</mi><mi>i</mi><mi>c</mi><mo>-</mo><mi>h</mi><mi>y</mi><mi>p</mi><mi>o</mi><mi>t</mi><mi>h</mi><mi>e</mi><mi>s</mi><mi>i</mi><mi>z</mi><mi>e</mi><mi>d</mi><mi>v</mi><mi>a</mi><mi>l</mi><mi>u</mi><mi>e</mi></mrow> <mrow><mi>s</mi><mi>t</mi><mi>a</mi><mi>n</mi><mi>d</mi><mi>a</mi><mi>r</mi><mi>d</mi><mi>e</mi><mi>r</mi><mi>r</mi><mi>o</mi><mi>r</mi></mrow></mfrac></mstyle></mrow></math>

假设检验中的一个重要考虑因素是样本可能不具代表性，这导致对总体的描述出现错误。这导致了两种类型的错误：

+   *I型错误*：当拒绝虚无假设，即使它是真的时发生。

+   *II型错误*：当未能拒绝虚无假设，即使它是错误的时发生。

直观地说，显著性水平是发生I型错误的概率。请记住，如果α = 5%，那么错误地拒绝真实的虚无假设的概率为5%。举个例子会更清楚。

考虑一位分析师对一个长短投资组合20年期间的年度回报进行研究。平均年回报率为1%，标准偏差为2%。分析师认为年均回报率不等于零，并希望为此构建一个95%的置信区间，然后进行假设检验：

1.  列出变量。样本大小为20，标准偏差为2%，平均值为1%。

1.  根据公式计算标准误差，在这种情况下为0.44%。

1.  定义95%置信区间的临界值，即+1.96和-1.96。

1.  因此置信区间为（0.13%，1.86%）。

1.  指定虚无假设，根据分析师的意见，这是一个双尾检验。虚无假设是年回报等于零。如果测试统计量小于-1.96或大于+1.96，则应拒绝它。

1.  使用公式找到测试统计量为2.27。因此，虚无假设被拒绝。

还有一个重要的指标要讨论：p值。*p值*是在零假设成立的情况下，看到一个比统计测试中看到的更极端的测试统计量的概率。将p值与显著性水平（通常为0.05）进行比较，可以帮助您理解它。如果p值小于或等于显著性水平，则结果被认为具有统计学意义，零假设被拒绝，支持备择假设。

如果p值小于5%的显著性水平，这意味着如果零假设成立，看到一个与当前测试统计量一样极端的测试统计量的概率为5%。另一种定义p值的方式是将其视为可以拒绝零假设的最小显著性水平。

###### 注意

本节的主要要点如下：

+   抽样是指在人口中收集数据，以便对上述人口的统计特性做出结论。

+   在机器学习中广泛使用抽样。一个例子是交叉验证。

+   假设检验是一种从少量数据中得出对总体的结论的技术。

# 信息论入门

*信息论*是一个复杂的抽象数学领域，与概率密切相关。它研究信息如何被量化、存储和传输。当涉及到事件发生时，有三种情况：

+   *不确定性*：如果事件尚未发生。

+   *惊喜*：如果事件刚刚发生。

+   *信息*：如果事件已经发生。

信息论中的一个关键概念是*熵*：消息或信息源中的不确定性或随机性水平。它描述了事件或消息的意外程度。相反，*信息增益*衡量了在接收新信息时熵（惊喜）的减少。

基本上，信息理论描述了事件的惊喜。当事件发生的概率很低时，它具有更多的惊喜，因此提供更多信息。同样，当事件发生的概率很高时，它具有更少的惊喜，因此提供更少的信息。您应该记住的是，从不太可能事件中学到的信息量大于从可能事件中学到的信息量。

在深入了解信息理论之前，了解什么是*对数*以及*指数*是很重要的。一般的指数函数将某个常数或变量提升到某个幂次方：

<math alttext="f left-parenthesis x right-parenthesis equals a Superscript x"><mrow><mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <msup><mi>a</mi> <mi>x</mi></msup></mrow></math>

换句话说，*一个数字的指数*是您将其乘以自身的次数：

<math alttext="4 cubed equals 4 times 4 times 4 equals 64"><mrow><msup><mn>4</mn> <mn>3</mn></msup> <mo>=</mo> <mn>4</mn> <mo>×</mo> <mn>4</mn> <mo>×</mo> <mn>4</mn> <mo>=</mo> <mn>64</mn></mrow></math>

相比之下，对数是指数的相反，其目的是找到指数（从前面的例子中知道4和64，找到3）：

<math alttext="log Subscript 4 Baseline left-parenthesis 64 right-parenthesis equals 3"><mrow><msub><mo form="prefix">log</mo> <mn>4</mn></msub> <mrow><mo>(</mo> <mn>64</mn> <mo>)</mo></mrow> <mo>=</mo> <mn>3</mn></mrow></math>

因此，对数是一个数字乘以另一个数字得到另一个数字的次数的答案。由于它们是互为反函数，您可以将它们一起使用来简化甚至解决x。看下面的例子：

<math alttext="log Subscript 4 Baseline left-parenthesis x right-parenthesis equals 3"><mrow><msub><mo form="prefix">log</mo> <mn>4</mn></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <mn>3</mn></mrow></math>

这里的目标是在给定对数函数的情况下找到x。第一步只是在一侧使用指数函数，因为希望它取消右侧的对数（记住，反函数互相抵消）。这给我们带来了以下结果：

<math alttext="4 Superscript l o g 4 left-parenthesis x right-parenthesis Baseline equals 4 cubed"><mrow><msup><mn>4</mn> <mrow><mi>l</mi><mi>o</mi><msub><mi>g</mi> <mn>4</mn></msub> <mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow></msup> <mo>=</mo> <msup><mn>4</mn> <mn>3</mn></msup></mrow></math>

<math alttext="x equals 4 cubed"><mrow><mi>x</mi> <mo>=</mo> <msup><mn>4</mn> <mn>3</mn></msup></mrow></math>

<math alttext="x equals 64"><mrow><mi>x</mi> <mo>=</mo> <mn>64</mn></mrow></math>

对数可以有不同的底数。然而，最常用的对数的底数是10。在计算机科学中，基数为2的对数表示比特（二进制位）。因此，信息被表示为比特。信息增益的公式如下：

<math alttext="upper H left-parenthesis x Subscript i Baseline right-parenthesis equals minus l o g 2 left-parenthesis upper P left-parenthesis x Subscript i Baseline right-parenthesis right-parenthesis"><mrow><mi>H</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>=</mo> <mo>-</mo> <mi>l</mi> <mi>o</mi> <msub><mi>g</mi> <mn>2</mn></msub> <mrow><mo>(</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>

假设有两个变量*x*和*y*，其中*x*的概率为1（100%，因此确定），而*y*的概率为0.5（50%，因此大部分是随机的），在这两种情况下信息是多少？答案如下：

<math alttext="upper H left-parenthesis x right-parenthesis equals minus l o g 2 left-parenthesis upper P left-parenthesis 1 right-parenthesis right-parenthesis equals 0"><mrow><mi>H</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <mo>-</mo> <mi>l</mi> <mi>o</mi> <msub><mi>g</mi> <mn>2</mn></msub> <mrow><mo>(</mo> <mi>P</mi> <mrow><mo>(</mo> <mn>1</mn> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mn>0</mn></mrow></math>

<math alttext="upper H left-parenthesis y right-parenthesis equals minus l o g 2 left-parenthesis upper P left-parenthesis 0.5 right-parenthesis right-parenthesis equals 1"><mrow><mi>H</mi> <mrow><mo>(</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>=</mo> <mo>-</mo> <mi>l</mi> <mi>o</mi> <msub><mi>g</mi> <mn>2</mn></msub> <mrow><mo>(</mo> <mi>P</mi> <mrow><mo>(</mo> <mn>0</mn> <mo>.</mo> <mn>5</mn> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mn>1</mn></mrow></math>

因此，确定事件提供零信息，而有一半机会实现的事件具有信息1。那么概率为0.05（5%）的非常不可能事件*z*的信息是多少？

<math alttext="upper H left-parenthesis z right-parenthesis equals minus l o g 2 left-parenthesis upper P left-parenthesis 0.05 right-parenthesis right-parenthesis equals 4.32"><mrow><mi>H</mi> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>=</mo> <mo>-</mo> <mi>l</mi> <mi>o</mi> <msub><mi>g</mi> <mn>2</mn></msub> <mrow><mo>(</mo> <mi>P</mi> <mrow><mo>(</mo> <mn>0</mn> <mo>.</mo> <mn>05</mn> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mn>4</mn> <mo>.</mo> <mn>32</mn></mrow></math>

因此，概率和信息之间存在负相关关系是信息理论的原则之一。熵和信息是相关概念，但它们具有不同的含义和应用。

*熵*是用来评估系统有多混乱或随机的度量标准。熵描述了信号的不确定性或不可预测性。系统或通信中的无序或不可预测程度随着熵的增加而增加。

*信息*是接收信号后熵或不确定性减少的结果。信号减少接收者的不确定性或熵的能力随着其信息内容的增加而增加。

###### 注意

每当所有事件等可能时，熵达到最大值。

熵使用以下公式计算：

<math alttext="upper S left-parenthesis x Subscript n Baseline right-parenthesis equals sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis minus l o g 2 left-parenthesis upper P left-parenthesis x Subscript i Baseline right-parenthesis right-parenthesis period left-parenthesis upper P left-parenthesis x Subscript i Baseline right-parenthesis right-parenthesis right-parenthesis"><mrow><mi>S</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>n</mi></msub> <mo>)</mo></mrow> <mo>=</mo> <msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></msubsup> <mrow><mo>(</mo> <mo>-</mo> <mi>l</mi> <mi>o</mi> <msub><mi>g</mi> <mn>2</mn></msub> <mrow><mo>(</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>.</mo> <mrow><mo>(</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>

因此，它是对数乘以各自概率的和的平均值。

现在，让我们讨论本节的最后一个概念，*信息增益*。通过改变数据集引起的熵的减少是通过信息增益来计算的。

信息增益是第7章决策树中的关键概念之一，因此在了解决策树是什么之后，您可能希望参考本节。

主要通过比较数据集在转换前后的熵来计算信息增益。请记住，当随机事件的所有结果具有相同的概率时，熵达到最大值。这也可以表示为一个分布，其中对称分布（如正态分布）具有高熵，而偏斜分布具有低熵。

###### 注意

最小化熵与最大化信息增益有关。

在结束这个信息论的介绍部分之前（当讨论决策树时，你将更深入地了解它），让我们看看*互信息*的概念。这个度量是在两个变量之间计算的，因此有*互*的名称，它衡量了在给定另一个变量的情况下变量不确定性的减少。互信息的公式如下：

<math alttext="upper M upper I left-parenthesis x comma y right-parenthesis equals upper S left-parenthesis x right-parenthesis minus upper S left-parenthesis x vertical-bar y right-parenthesis"><mrow><mi>M</mi> <mi>I</mi> <mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo> <mo>=</mo> <mi>S</mi> <mo>(</mo> <mi>x</mi> <mo>)</mo> <mo>-</mo> <mi>S</mi> <mo>(</mo> <mi>x</mi> <mo>|</mo> <mi>y</mi> <mo>)</mo></mrow></math>

<math alttext="upper M upper I left-parenthesis x comma y right-parenthesis i s t h e m u t u a l i n f o r m a t i o n o f x a n d y"><mrow><mi>M</mi> <mi>I</mi> <mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo> <mi>i</mi> <mi>s</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>m</mi> <mi>u</mi> <mi>t</mi> <mi>u</mi> <mi>a</mi> <mi>l</mi> <mi>i</mi> <mi>n</mi> <mi>f</mi> <mi>o</mi> <mi>r</mi> <mi>m</mi> <mi>a</mi> <mi>t</mi> <mi>i</mi> <mi>o</mi> <mi>n</mi> <mi>o</mi> <mi>f</mi> <mi>x</mi> <mi>a</mi> <mi>n</mi> <mi>d</mi> <mi>y</mi></mrow></math>

<math alttext="upper S left-parenthesis x right-parenthesis i s t h e e n t r o p y o f x"><mrow><mi>S</mi> <mo>(</mo> <mi>x</mi> <mo>)</mo> <mi>i</mi> <mi>s</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>r</mi> <mi>o</mi> <mi>p</mi> <mi>y</mi> <mi>o</mi> <mi>f</mi> <mi>x</mi></mrow></math>

<math alttext="upper S left-parenthesis x vertical-bar y right-parenthesis i s t h e c o n d i t i o n a l e n t r o p y o f x g i v e n y"><mrow><mi>S</mi> <mo>(</mo> <mi>x</mi> <mo>|</mo> <mi>y</mi> <mo>)</mo> <mi>i</mi> <mi>s</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>c</mi> <mi>o</mi> <mi>n</mi> <mi>d</mi> <mi>i</mi> <mi>t</mi> <mi>i</mi> <mi>o</mi> <mi>n</mi> <mi>a</mi> <mi>l</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>r</mi> <mi>o</mi> <mi>p</mi> <mi>y</mi> <mi>o</mi> <mi>f</mi> <mi>x</mi> <mi>g</mi> <mi>i</mi> <mi>v</mi> <mi>e</mi> <mi>n</mi> <mi>y</mi></mrow></math>

互信息因此衡量了变量之间的依赖关系。互信息越大，变量之间的关系就越大（零值表示独立变量）。记住这个概念，因为你将在第三章中的处理相关性的部分中看到它。这是因为互信息也可以衡量变量之间的非线性相关性。

###### 注意

让我们总结一下你需要在信息论中保留的内容，以便对即将到来的内容有基本的了解：

+   信息论使用概率的概念来计算在机器学习模型和其他计算中使用的信息和熵（如相关性）。

+   信息是接收信号导致的熵或不确定性的减少。熵是用来评估系统有多混乱或随机的度量标准。

+   互信息是两个随机变量之间依赖关系的度量。它也可以用来计算两者之间的相关性。

+   信息论中的工具被用于一些机器学习模型，如决策树。

## 总结

概率在继续更高级主题之前提供了一个基本框架。本章概述了在处理机器学习和深度学习模型时可能遇到的概念。了解如何计算概率以及如何执行假设检验是很重要的（尽管实际上算法会为您执行这些操作）。

下一章非常重要，介绍了你需要的统计知识，不仅适用于机器学习，还适用于金融交易和复杂数据分析。
