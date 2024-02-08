# 第二章。概率基础

概率是一门量化我们对事件的不确定性的数学领域。例如，当掷骰子或抛硬币时，除非骰子或硬币本身存在任何不规则性，否则我们对即将发生的结果感到不确定。然而，我们可以通过概率来量化我们对每种可能结果的信念。例如，我们说每次抛硬币时硬币出现正面的概率是 <math alttext="one-half"><mfrac><mn>1</mn> <mn>2</mn></mfrac></math> 。每次掷骰子时，我们说骰子朝上的概率是 <math alttext="one-sixth"><mfrac><mn>1</mn> <mn>6</mn></mfrac></math> 。这些是我们在日常生活中轻松谈论的概率，但我们如何定义和有效利用它们呢？在本章中，我们将讨论概率的基础知识以及它们与深度学习中的关键概念的联系。

# 事件和概率

当进行像掷骰子或抛硬币这样的试验时，我们直观地对试验的可能结果赋予一些信念。在本节中，我们旨在形式化其中一些概念。特别是，我们将从在这个*离散*空间中工作开始，其中离散表示有限或可数无限的可能性。掷骰子和抛硬币都在离散空间中——掷一个公平的骰子有六种可能结果，抛一个公平的硬币有两种可能。我们将实验的整个可能性集合称为*样本空间*。例如，从一到六的数字将构成掷一个公平骰子的样本空间。我们可以将*事件*定义为样本空间的子集。至少掷出三的事件对应于之前定义的样本空间中三、四、五和六中的任何数字朝上的骰子。一组在样本空间中所有结果上总和为一的概率被称为该样本空间上的*概率分布*，这些分布将是我们讨论的主要焦点。

一般来说，我们不会过多担心这些概率的确切来源，因为这需要进行更严格和彻底的检查，超出了本文的范围。然而，我们将对不同的解释提供一些直觉。在高层次上，*频率主义*观点认为结果的概率来自于长期实验中的频率。在公平骰子的情况下，这种观点声称我们可以说在给定的投掷中骰子的任何一面出现的概率是 <math alttext="one-sixth"><mfrac><mn>1</mn> <mn>6</mn></mfrac></math> ，因为进行大量投掷并计算每一面出现的次数将给我们一个大致为这个分数的估计。随着实验中投掷次数的增加，我们看到这个估计越来越接近极限 <math alttext="one-sixth"><mfrac><mn>1</mn> <mn>6</mn></mfrac></math> ，结果的概率。

另一方面，*贝叶斯*概率观是基于量化我们对假设的先验信念以及我们如何根据新数据更新信念。对于一个公平的骰子，贝叶斯观点会声称没有先验信息，无论是来自骰子的结构还是投掷过程，都会暗示任何一面的骰子更有可能出现。因此，我们会说每个结果的概率是<math alttext="one-sixth"><mfrac><mn>1</mn> <mn>6</mn></mfrac></math>，这是我们的先验信念。在这种情况下，与每个结果相关的概率集合，即全部为<math alttext="one-sixth"><mfrac><mn>1</mn> <mn>6</mn></mfrac></math>，被称为我们的*先验*。随着我们看到新数据，贝叶斯观点给了我们一种方法来相应地更新我们的先验，我们称这种新信念为我们的*后验*。这种贝叶斯观点有时直接应用于神经网络训练，我们首先假设网络中的每个权重都有一些与之相关的先验。当我们训练网络时，我们相应地更新与每个权重相关的先验，以更好地适应我们看到的数据。在训练结束时，我们留下了与每个权重相关的后验分布。

在本章中，我们将假设与任何结果相关的概率是通过合理的方法确定的，并关注如何操纵这些概率以用于我们的分析。我们从概率的四个原则开始，特别是在离散空间中：

1.  样本空间中所有可能结果的概率之和必须等于一。换句话说，样本空间中所有结果的概率分布必须总和为一。这在直觉上应该是有意义的，因为样本空间中的所有结果集合必须代表所有可能性的整体集合。概率分布不总和为一将意味着存在未考虑的可能性，这是矛盾的。数学上，我们说对于任何有效的概率分布，<math alttext="sigma-summation Underscript o Endscripts upper P left-parenthesis o right-parenthesis equals 1"><mrow><msub><mo>∑</mo> <mi>o</mi></msub> <mi>P</mi> <mrow><mo>(</mo> <mi>o</mi> <mo>)</mo></mrow> <mo>=</mo> <mn>1</mn></mrow></math>，其中<math alttext="o"><mi>o</mi></math>代表一个结果。

1.  设<math alttext="upper E 1"><msub><mi>E</mi> <mn>1</mn></msub></math>为一个事件，并回想我们将事件定义为可能结果的子集。我们称<math alttext="upper E 1 Superscript c"><msubsup><mi>E</mi> <mn>1</mn> <mi>c</mi></msubsup></math>为*事件<math alttext="upper E 1"><msub><mi>E</mi> <mn>1</mn></msub></math>的补集*，或者说是样本空间中不在<math alttext="upper E 1"><msub><mi>E</mi> <mn>1</mn></msub></math>中的所有可能结果。概率的第二原则是<math alttext="upper P left-parenthesis upper E 1 right-parenthesis equals 1 minus upper P left-parenthesis upper E 1 Superscript c Baseline right-parenthesis"><mrow><mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>)</mo></mrow> <mo>=</mo> <mn>1</mn> <mo>-</mo> <mi>P</mi> <mrow><mo>(</mo> <msubsup><mi>E</mi> <mn>1</mn> <mi>c</mi></msubsup> <mo>)</mo></mrow></mrow></math>。这只是对第一原则的应用-如果这不是真的，显然会与第一原则相矛盾。在图 2-1 中，我们看到了一个例子，其中*S*代表了所有结果的整体空间，事件及其补集一起形成了*S*的全部内容。

    ![](img/fdl2_0201.png)

    ###### 图 2-1。事件 A 及其补集相互作用形成了所有可能性的整体集合 S。补集简单地定义了最初不在 A 中的所有可能性。

1.  设 <math alttext="upper E 1"><msub><mi>E</mi> <mn>1</mn></msub></math> 和 <math alttext="upper E 2"><msub><mi>E</mi> <mn>2</mn></msub></math> 是两个事件，其中 <math alttext="upper E 1"><msub><mi>E</mi> <mn>1</mn></msub></math> 是 <math alttext="upper E 2"><msub><mi>E</mi> <mn>2</mn></msub></math> 的子集（不一定是严格的）。第三个原则是 <math alttext="upper P left-parenthesis upper E 1 right-parenthesis less-than-or-equal-to upper P left-parenthesis upper E 2 right-parenthesis"><mrow><mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>)</mo></mrow> <mo>≤</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow></mrow></math> 。再次强调，这并不会太令人惊讶——第二个事件至少有第一个事件的那么多结果，而且第二个事件是第一个事件的超集，包含了第一个事件的所有结果。如果这个原则不成立，那就意味着存在具有负概率的结果，这在我们的定义中是不可能的。

1.  概率的第四个也是最后一个原则是包含和排除原则，它规定了 <math alttext="upper P left-parenthesis upper A union upper B right-parenthesis equals upper P left-parenthesis upper A right-parenthesis plus upper P left-parenthesis upper B right-parenthesis minus upper P left-parenthesis upper A intersection upper B right-parenthesis"><mrow><mi>P</mi> <mo>(</mo> <mi>A</mi> <mo>∪</mo> <mi>B</mi> <mo>)</mo> <mo>=</mo> <mi>P</mi> <mo>(</mo> <mi>A</mi> <mo>)</mo> <mo>+</mo> <mi>P</mi> <mo>(</mo> <mi>B</mi> <mo>)</mo> <mo>-</mo> <mi>P</mi> <mo>(</mo> <mi>A</mi> <mo>∩</mo> <mi>B</mi> <mo>)</mo></mrow></math> 。对于不熟悉这个术语的人来说，<math alttext="union"><mo>∪</mo></math> 表示两个事件的*并集*，这是一个集合操作，将两个事件返回一个包含来自两个原始集合的所有元素的事件。而 <math alttext="intersection"><mo>∩</mo></math> ，或*交集*，是一个集合操作，返回一个包含属于两个原始集合的所有元素的事件。所述等式背后的思想是，通过简单地对*A*和*B*的概率求和，我们会重复计算属于两个集合的元素。因此，为了准确地获得并集的概率，我们必须减去交集的概率。在图 2-2 中，我们展示了两个事件及其交集在物理上的样子，而并集则是两个事件的组合区域中的所有结果。

    ![](img/fdl2_0202.png)

    ###### 图 2-2。中间的薄片是两个集合之间的重叠部分，包含了同时在两个集合中的所有结果。并集是两个圆圈组合区域中的所有事件；如果我们简单地将它们的概率相加，我们将重复计算中间薄片中的所有结果。

这些概率原则渗透到与该领域有关的一切事物中。例如，在深度学习中，我们的大多数问题可以归为两类：*回归*和*分类*。在分类问题中，我们训练一个神经模型，可以预测输入属于一组离散类别中的哪一个的可能性。例如，著名的 MNIST 数字数据集为我们提供了 0 到 9 范围内的数字图片和相关的数字标签。我们的目标是构建一个*分类器*，可以接收这张图片并返回最有可能的标签作为猜测。这自然地被制定为一个概率问题——分类器产生一个关于样本空间 0 到 9 的概率分布，对于任何给定的输入，它的最佳猜测是被分配了最高概率的数字。这与我们的原则有什么关系？由于分类器产生一个概率分布，它必须遵循这些原则。例如，与每个数字相关的概率必须相加为一——这是一个快速的粗略检查，以确保模型没有错误。在下一节中，我们将涵盖最初给定相关信息影响我们信念的概率以及如何使用该信息。

# 条件概率

了解信息通常会改变我们的信念，从而改变我们的概率。回到我们经典的骰子示例，我们可能认为掷骰子是公平的，而实际上骰子的核心有一个隐藏的重量，使得它更有可能掷出大于三的数字。当我们掷骰子时，当然会开始注意到这种模式，我们对骰子公平性的信念开始转变。这正是条件概率的核心。我们不再简单地考虑*P(偏向)*或*P(公平)*，而是要考虑像*P(偏向|信息)*这样的概率。这个量，我们称之为*条件概率*，可以理解为“在我们看到的信息的情况下，骰子偏向的概率”。

我们如何直观地思考这些概率？首先，我们必须想象我们现在处于一个不同的宇宙中，而不是我们开始时的那个宇宙。新的宇宙是一个包含了我们自实验开始以来看到的信息的宇宙，例如我们过去的骰子点数。回到我们的 MNIST 示例，训练好的神经网络产生的概率分布实际上是一个条件概率分布。例如，输入图像为零的概率可以看作是*P(0|input)*。简单来说，我们想找到的是在我们馈送给神经网络的特定输入图像中组成的所有像素的情况下零的概率。我们的新宇宙是输入像素已经具有这种特定值配置的宇宙。这与简单地看*P(0)*，即返回零的概率是不同的，我们可以从先验信念的角度来思考。如果没有任何关于输入像素配置的知识，我们没有理由相信返回零的可能性比其他数字更有可能或更不可能。

有时，看到某些信息并不会改变我们的概率——我们称之为*独立性*。例如，汤姆·布雷迪可能在我们的实验第三次掷骰子后投出了一个触摸得分，但将这些信息纳入我们的新宇宙中应该（希望如此！）不会对骰子有偏倚的可能性产生影响。我们将这种独立性属性表述为*P(有偏|汤姆·布雷迪投出触摸得分) = P(有偏)*。请注意，任何满足这一属性的两个事件<math alttext="上标 E1"><msub><mi>E</mi> <mn>1</mn></msub></math>和<math alttext="上标 E2"><msub><mi>E</mi> <mn>2</mn></msub></math>都是独立的。也许稍微有些违反直觉的是，如果到目前为止我们所有的掷骰子结果在数值上并没有改变我们对骰子公平性的先验信念（也许到目前为止的掷骰子结果在一到六之间均匀出现，而我们最初的先验信念是骰子是公平的），我们仍然会说这些事件是独立的。最后，请注意独立性是对称的：如果<math alttext="P(E1|E2) = P(E1)"><mrow><mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>|</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>=</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>)</mo></mrow></mrow></math>，那么也有<math alttext="P(E2|E1) = P(E2)"><mrow><mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>|</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>)</mo></mrow> <mo>=</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow></mrow></math>。

在前一节中，我们介绍了交集和并集符号。事实证明，我们可以将交集操作分解为概率的乘积。我们有以下等式：<math alttext="upper P left-parenthesis upper E 1 intersection upper E 2 right-parenthesis equals upper P left-parenthesis upper E 1 vertical-bar upper E 2 right-parenthesis asterisk upper P left-parenthesis upper E 2 right-parenthesis"><mrow><mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>∩</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>=</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>|</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow></mrow></math>。让我们解释一下这里的直觉。在左边，我们有两个事件<math alttext="upper E 1"><msub><mi>E</mi> <mn>1</mn></msub></math>和<math alttext="upper E 2"><msub><mi>E</mi> <mn>2</mn></msub></math>同时发生的概率。在右边，我们有相同的想法，但表达略有不同。在这两个事件都发生的宇宙中，到达这个宇宙的一种方式是首先发生<math alttext="upper E 2"><msub><mi>E</mi> <mn>2</mn></msub></math>，然后是<math alttext="upper E 1"><msub><mi>E</mi> <mn>1</mn></msub></math>。将这种直觉转化为数学术语，我们必须首先找到<math alttext="upper E 2"><msub><mi>E</mi> <mn>2</mn></msub></math>发生的概率，然后是在<math alttext="upper E 2"><msub><mi>E</mi> <mn>2</mn></msub></math>已经发生的宇宙中<math alttext="upper E 1"><msub><mi>E</mi> <mn>1</mn></msub></math>发生的概率。我们如何结合这两个概率？直觉上，将它们相乘是有意义的——我们必须让两个事件都发生，第一个是无条件的，第二个是在第一个已经发生的宇宙中。请注意，这些事件的顺序并不重要，因为这两条路径都将我们带到同一个宇宙。因此，更完整地说，<math alttext="upper P left-parenthesis upper E 1 intersection upper E 2 right-parenthesis equals upper P left-parenthesis upper E 1 vertical-bar upper E 2 right-parenthesis asterisk upper P left-parenthesis upper E 2 right-parenthesis equals upper P left-parenthesis upper E 2 vertical-bar upper E 1 right-parenthesis asterisk upper P left-parenthesis upper E 1 right-parenthesis"><mrow><mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>∩</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>=</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>|</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>=</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>|</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>)</mo></mrow></mrow></math>。

然而，其中一些路径比其他路径更有物理意义。例如，如果我们将*<math alttext="upper E 1"><msub><mi>E</mi> <mn>1</mn></msub></math>*看作某人感染疾病的事件，将<math alttext="upper E 2"><msub><mi>E</mi> <mn>2</mn></msub></math>看作患者出现疾病症状的事件，那么患者先感染疾病然后出现症状的路径比反过来的路径更有物理意义。

在两个事件独立的情况下，我们有<math alttext="upper P left-parenthesis upper E 1 intersection upper E 2 right-parenthesis equals upper P left-parenthesis upper E 1 vertical-bar upper E 2 right-parenthesis asterisk upper P left-parenthesis upper E 2 right-parenthesis equals upper P left-parenthesis upper E 1 right-parenthesis asterisk upper P left-parenthesis upper E 2 right-parenthesis period"><mrow><mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>∩</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>=</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>|</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>=</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>1</mn></msub> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>E</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mo>.</mo></mrow></math> 希望这些能有一些直观的理解。在独立的情况下，事件<math alttext="upper E 2"><msub><mi>E</mi> <mn>2</mn></msub></math> 的发生不会影响事件<math alttext="upper E 1"><msub><mi>E</mi> <mn>1</mn></msub></math> 发生的概率；即，将这些信息纳入新的宇宙中不会影响下一个事件的概率。在接下来的部分中，我们将讨论随机变量，它们是事件的相关总结，也有自己的概率分布。

# 随机变量

再次，让我们考虑抛硬币的实验。如果我们抛硬币有限次数，自然会产生一些问题。在我们的实验中遇到了多少次正面？多少次反面？第一个正面前有多少次反面？在这样一个实验中，每个结果都有对应的答案。比如，如果我们抛硬币 5 次，得到了序列 TTHHT，我们看到了两次正面，三次反面，以及第一个正面前有两次反面。

我们可以将*随机变量*看作是一个从样本空间到另一个空间的映射或函数，比如图 2-3 中的整数。这样一个函数将以 TTHHT 作为输入，并根据我们提出的问题输出三个答案中的一个。随机变量取得的值将是与实验结果相关联的输出。虽然随机变量是确定性的，因为它们将给定的输入映射到单个输出，但它们在输出空间中也有与之相关的分布。这是由于实验中固有的随机性——根据输入结果的概率，其相应的输出可能比其他输出更有可能。

![](img/fdl2_0203.png)

###### 图 2-3. 随机变量 X、Y 和 Z 都作用于相同的样本空间，但具有不同的输出。记住你正在测量什么是很重要的！

###### 注意

请注意，多个输入可能映射到相同的输出。例如，*X(HHH)* = 3，除了*X(HHTH)*在图 2-3 中也是如此。

一个简单的开始方法是将这个映射看作是一个恒等函数——无论我们抛硬币还是掷骰子，它在输出空间中的映射与输入完全相同。将正面编码为 1，反面编码为 0，我们可以定义一个代表硬币翻转的随机变量，即硬币是正面的情况，即*C(1) = 1*，其中*C*是我们的随机变量。在掷骰子的情况下，映射的输出与我们掷出的数字相同，即*D(5) = 5*，其中*D*是我们的随机变量。

为什么我们要关心随机变量及其分布？事实证明，它们在深度学习和机器学习中起着至关重要的作用。例如，在第四章中，我们将介绍*辍学*的概念，这是一种在神经网络中减少过拟合的技术。辍学层的想法是，在训练期间，它独立且随机地以一定概率屏蔽前一层中的每个神经元。这可以防止网络过度依赖特定连接或子网络。我们可以将前一层中的每个神经元看作代表硬币翻转类型实验。唯一的区别是，我们设置了这个实验的概率，而不是一个公平硬币具有默认概率*mfrac 1 2*的概率。每个神经元都有与之关联的随机变量*X*，如果辍学层决定屏蔽它，则输入为 1，否则为 0。*X*是一个从输入空间到输出空间的恒等函数，即*X(1) = 1*和*X(0) = 0*。

随机变量，一般来说，不必是恒等映射。您可以想到的大多数函数都是将输入空间映射到定义了随机变量的输出空间的有效方法。例如，如果输入空间是每个可能长度为*n*的硬币翻转序列，函数可以是计算序列中头的数量并对其进行平方。一些随机变量甚至可以表示为其他随机变量的函数，或者是函数的函数，我们稍后会讨论。如果我们再次考虑每个可能长度为*n*的硬币翻转序列的输入空间，那么计算输入序列中头的数量的随机变量与计算每个单独硬币翻转是否为头并将所有这些值求和的随机变量是相同的。在数学术语中，我们说*X 等于 sigma-summation Underscript i equals 1 Overscript n Endscripts upper C Subscript i*，其中*X*是表示头的总数的随机变量，*C i*是与第*i*次硬币翻转相关的二进制随机变量。回到辍学的例子，我们可以将代表被屏蔽神经元总数的随机变量看作是代表每个神经元的二进制随机变量之和。

将来，当我们想要提及随机变量取特定值*c*的事件时（域是我们一直在提到的输出空间，例如硬币翻转序列中的头数），我们将简洁地写为*X = c*。我们将随机变量取特定值的概率表示为*P(X = c)*，例如。随机变量在输出空间中取任何给定值的概率只是映射到它的输入的概率之和。这应该有一些直观的意义，因为这基本上是概率的第四原则，其中任何两个事件之间的交集是空集，因为我们从的所有事件都是独立的、不同的输入。请注意，*P(X)*本身也是一个遵循第一节描述的概率的所有基本原则的概率分布。在下一节中，我们将考虑关于随机变量的统计数据。

# 期望值

正如我们讨论的，随机变量是从输入空间到输出空间的映射，其中输入根据某种概率分布生成。随机变量可以被视为输入的相关摘要，并且根据我们提出的问题可以采用多种形式。有时，了解关于随机变量的统计数据是有用的。例如，如果我们抛硬币八次，我们平均期望看到多少次正面？当然，我们并不总是看到平均头数——我们看到的头数会有多大变化？第一个数量是我们称之为随机变量的*期望*，第二个是随机变量的*方差*。

对于随机变量*X*，我们将其期望表示为<math alttext="double-struck upper E left-bracket upper X right-bracket"><mrow><mi>𝔼</mi> <mo>[</mo> <mi>X</mi> <mo>]</mo></mrow></math>。我们可以将其视为*X*取值的平均值，按照每个结果的概率加权。数学上，这被写为<math alttext="double-struck upper E left-bracket upper X right-bracket equals sigma-summation Underscript o Endscripts o asterisk upper P left-parenthesis upper X equals o right-parenthesis"><mrow><mi>𝔼</mi> <mrow><mo>[</mo> <mi>X</mi> <mo>]</mo></mrow> <mo>=</mo> <msub><mo>∑</mo> <mi>o</mi></msub> <mi>o</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>=</mo> <mi>o</mi> <mo>)</mo></mrow></mrow></math>。请注意，如果所有结果*o*是等可能的，我们得到所有结果的简单平均值。使用结果的概率作为加权是有意义的，因为某些结果比其他结果更有可能发生，我们观察到的平均值将偏向这些结果。对于一次公平抛硬币，预期的正面数量将是<math alttext="sigma-summation Underscript o element-of StartSet 0 comma 1 EndSet Endscripts o asterisk upper P left-parenthesis o right-parenthesis equals 0 asterisk 0.5 plus 1 asterisk 0.5 equals 0.5"><mrow><msub><mo>∑</mo> <mrow><mi>o</mi><mo>∈</mo><mo>{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>}</mo></mrow></msub> <mi>o</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>o</mi> <mo>)</mo></mrow> <mo>=</mo> <mn>0</mn> <mo>*</mo> <mn>0</mn> <mo>.</mo> <mn>5</mn> <mo>+</mo> <mn>1</mn> <mo>*</mo> <mn>0</mn> <mo>.</mo> <mn>5</mn> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>5</mn></mrow></math>。换句话说，我们预计在任何给定的公平抛硬币中看到半个正面。当然，这在物理上没有意义，因为我们永远不可能抛出半个正面，但这给了你一个关于我们在长期实验中预期看到的比例的想法。

回到我们的例子，长度为*n*的硬币序列，让我们尝试找到这样一个序列中预期的正面数量。我们有*n* + 1 种可能的正面数量，根据我们的公式，我们需要找到获得每个可能数量的概率作为我们的权重。数学上，我们需要计算<math alttext="sigma-summation Underscript x element-of StartSet 0 comma ellipsis comma n EndSet Endscripts x asterisk upper P left-parenthesis upper X equals x right-parenthesis"><mrow><msub><mo>∑</mo> <mrow><mi>x</mi><mo>∈</mo><mo>{</mo><mn>0</mn><mo>,</mo><mo>...</mo><mo>,</mo><mi>n</mi><mo>}</mo></mrow></msub> <mi>x</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>=</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>，其中*X*是代表正面总数的随机变量。然而，随着*n*变得越来越大，执行这个计算开始变得越来越复杂。

相反，让我们将<math alttext="上标 X 下标 i"><msub><mi>X</mi> <mi>i</mi></msub></math>表示为第*i*次抛硬币的二元随机变量，并使用我们在上一节中所做的观察，即能够将总头数分解为所有单个抛硬币的头/尾之和。由于我们知道<math alttext="上标 X 等于上标 X1 加上上标 X2 加上省略号加上上标 X 下标 n"><mrow><mi>X</mi> <mo>=</mo> <msub><mi>X</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>X</mi> <mn>2</mn></msub> <mo>+</mo> <mo>...</mo> <mo>+</mo> <msub><mi>X</mi> <mi>n</mi></msub></mrow></math>，我们也可以说<math alttext="双黑体上标 E 左括号上标 X 右括号等于双黑体上标 E 左括号上标 X1 加上上标 X2 加上省略号加上上标 X 下标 n 基线右括号"><mrow><mi>𝔼</mi> <mrow><mo>[</mo> <mi>X</mi> <mo>]</mo></mrow> <mo>=</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msub><mi>X</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>X</mi> <mn>2</mn></msub> <mo>+</mo> <mo>...</mo> <mo>+</mo> <msub><mi>X</mi> <mi>n</mi></msub> <mo>]</mo></mrow></mrow></math>。这种替换如何使我们的问题更容易？我们现在引入*期望的线性性*的概念，它表明我们可以将右侧分解为和<math alttext="双黑体上标 E 左括号上标 X1 右括号加双黑体上标 E 左括号上标 X2 右括号加省略号加双黑体上标 E 左括号上标 X 下标 n 基线右括号"><mrow><mi>𝔼</mi> <mrow><mo>[</mo> <msub><mi>X</mi> <mn>1</mn></msub> <mo>]</mo></mrow> <mo>+</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msub><mi>X</mi> <mn>2</mn></msub> <mo>]</mo></mrow> <mo>+</mo> <mo>...</mo> <mo>+</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msub><mi>X</mi> <mi>n</mi></msub> <mo>]</mo></mrow></mrow></math>。我们知道每次抛硬币的期望头数为 0.5，因此在*n*次抛硬币序列中的期望头数只是 0.5**n*。这比沿着以前的路线走要简单得多，因为这种方法的难度不随着抛硬币次数的增加而增加。

让我们更详细地讨论我们所做的简化。从数学上讲，如果我们有任何两个独立的随机变量*A*和*B*：

<math alttext="double-struck upper E left-bracket upper A plus upper B right-bracket equals sigma-summation Underscript a comma b Endscripts left-parenthesis a plus b right-parenthesis asterisk upper P left-parenthesis upper A equals a comma upper B equals b right-parenthesis"><mrow><mi>𝔼</mi> <mrow><mo>[</mo> <mi>A</mi> <mo>+</mo> <mi>B</mi> <mo>]</mo></mrow> <mo>=</mo> <msub><mo>∑</mo> <mrow><mi>a</mi><mo>,</mo><mi>b</mi></mrow></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>+</mo> <mi>b</mi> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>=</mo> <mi>a</mi> <mo>,</mo> <mi>B</mi> <mo>=</mo> <mi>b</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="equals sigma-summation Underscript a comma b Endscripts left-parenthesis a plus b right-parenthesis asterisk upper P left-parenthesis upper A equals a right-parenthesis asterisk upper P left-parenthesis upper B equals b right-parenthesis"><mrow><mo>=</mo> <msub><mo>∑</mo> <mrow><mi>a</mi><mo>,</mo><mi>b</mi></mrow></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>+</mo> <mi>b</mi> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>=</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>=</mo> <mi>b</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="equals sigma-summation Underscript a comma b Endscripts a asterisk upper P left-parenthesis upper A equals a right-parenthesis asterisk upper P left-parenthesis upper B equals b right-parenthesis plus b asterisk upper P left-parenthesis upper A equals a right-parenthesis asterisk upper P left-parenthesis upper B equals b right-parenthesis"><mrow><mo>=</mo> <msub><mo>∑</mo> <mrow><mi>a</mi><mo>,</mo><mi>b</mi></mrow></msub> <mi>a</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>=</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>=</mo> <mi>b</mi> <mo>)</mo></mrow> <mo>+</mo> <mi>b</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>=</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>=</mo> <mi>b</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="equals sigma-summation Underscript a comma b Endscripts a asterisk upper P left-parenthesis upper A equals a right-parenthesis asterisk upper P left-parenthesis upper B equals b right-parenthesis plus sigma-summation Underscript a comma b Endscripts b asterisk upper P left-parenthesis upper A equals a right-parenthesis asterisk upper P left-parenthesis upper B equals b right-parenthesis"><mrow><mo>=</mo> <msub><mo>∑</mo> <mrow><mi>a</mi><mo>,</mo><mi>b</mi></mrow></msub> <mi>a</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>=</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>=</mo> <mi>b</mi> <mo>)</mo></mrow> <mo>+</mo> <msub><mo>∑</mo> <mrow><mi>a</mi><mo>,</mo><mi>b</mi></mrow></msub> <mi>b</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>=</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>=</mo> <mi>b</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="equals sigma-summation Underscript a Endscripts a asterisk upper P left-parenthesis upper A equals a right-parenthesis sigma-summation Underscript b Endscripts upper P left-parenthesis upper B equals b right-parenthesis plus sigma-summation Underscript b Endscripts b asterisk upper P left-parenthesis upper B equals b right-parenthesis sigma-summation Underscript a Endscripts upper P left-parenthesis upper A equals a right-parenthesis"><mrow><mo>=</mo> <msub><mo>∑</mo> <mi>a</mi></msub> <mi>a</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>=</mo> <mi>a</mi> <mo>)</mo></mrow> <msub><mo>∑</mo> <mi>b</mi></msub> <mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>=</mo> <mi>b</mi> <mo>)</mo></mrow> <mo>+</mo> <msub><mo>∑</mo> <mi>b</mi></msub> <mi>b</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>=</mo> <mi>b</mi> <mo>)</mo></mrow> <msub><mo>∑</mo> <mi>a</mi></msub> <mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>=</mo> <mi>a</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="equals sigma-summation Underscript a Endscripts a asterisk upper P left-parenthesis upper A equals a right-parenthesis plus sigma-summation Underscript b Endscripts b asterisk upper P left-parenthesis upper B equals b right-parenthesis"><mrow><mo>=</mo> <msub><mo>∑</mo> <mi>a</mi></msub> <mi>a</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>=</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>+</mo> <msub><mo>∑</mo> <mi>b</mi></msub> <mi>b</mi> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>=</mo> <mi>b</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="equals double-struck upper E left-bracket upper A right-bracket plus double-struck upper E left-bracket upper B right-bracket"><mrow><mo>=</mo> <mi>𝔼</mi> <mo>[</mo> <mi>A</mi> <mo>]</mo> <mo>+</mo> <mi>𝔼</mi> <mo>[</mo> <mi>B</mi> <mo>]</mo></mrow></math>

###### 注意

请注意，在我们将事件*A=a*和事件*B=b*的概率分解为两个单独概率的乘积时，我们在本章前面讨论的独立性假设在这里得到了应用。推导的其余部分不需要额外的假设，因此我们建议您自行进行代数运算。尽管我们不会展示依赖情况，但期望的线性性也适用于依赖的随机变量。

回到辍学的例子，掩码神经元的总数的期望可以分解为对每个神经元的期望之和。掩码神经元的期望数量，类似于抛硬币序列中头的期望数量，是*p*n*，其中*p*是被掩码的概率（以及代表神经元的每个单个二元随机变量的期望），*n*是神经元的数量。

正如前面提到的，我们并不总是在每次实验重复中看到事件发生的期望次数。在某些情况下，例如之前单次公平抛硬币中头的期望次数，我们永远也看不到！接下来，我们将量化实验重复中与期望值的平均偏差，或方差。

# 方差

我们定义方差，或 Var(*X*)，为<math alttext="double-struck upper E left-bracket left-parenthesis upper X minus mu right-parenthesis squared right-bracket"><mrow><mi>𝔼</mi> <mo>[</mo> <msup><mrow><mo>(</mo><mi>X</mi><mo>-</mo><mi>μ</mi><mo>)</mo></mrow> <mn>2</mn></msup> <mo>]</mo></mrow></math>，其中我们让<math alttext="mu equals double-struck upper E left-bracket upper X right-bracket"><mrow><mi>μ</mi> <mo>=</mo> <mi>𝔼</mi> <mo>[</mo> <mi>X</mi> <mo>]</mo></mrow></math>。简单来说，这个度量表示值*X*取值与其期望之间的平均平方差。请注意，<math alttext="left-parenthesis upper X minus mu right-parenthesis squared"><msup><mrow><mo>(</mo><mi>X</mi><mo>-</mo><mi>μ</mi><mo>)</mo></mrow> <mn>2</mn></msup></math>本身也是一个随机变量，因为它是一个函数的函数（*X*），而函数仍然是一个函数。虽然我们不会详细讨论为什么我们特别使用这个公式，但我们鼓励您思考为什么我们不使用<math alttext="double-struck upper E left-bracket upper X minus mu right-bracket"><mrow><mi>𝔼</mi> <mo>[</mo> <mi>X</mi> <mo>-</mo> <mi>μ</mi> <mo>]</mo></mrow></math>这样的公式。为了获得方差的稍微简化形式，我们可以进行以下简化：

<math alttext="double-struck upper E left-bracket left-parenthesis upper X minus mu right-parenthesis squared right-bracket equals double-struck upper E left-bracket upper X squared minus 2 mu upper X plus mu squared right-bracket"><mrow><mi>𝔼</mi> <mrow><mo>[</mo> <msup><mrow><mo>(</mo><mi>X</mi><mo>-</mo><mi>μ</mi><mo>)</mo></mrow> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>=</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>X</mi> <mn>2</mn></msup> <mo>-</mo> <mn>2</mn> <mi>μ</mi> <mi>X</mi> <mo>+</mo> <msup><mi>μ</mi> <mn>2</mn></msup> <mo>]</mo></mrow></mrow></math>

<math alttext="equals double-struck upper E left-bracket upper X squared right-bracket minus double-struck upper E left-bracket 2 mu upper X right-bracket plus double-struck upper E left-bracket mu squared right-bracket"><mrow><mo>=</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>X</mi> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>-</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <mn>2</mn> <mi>μ</mi> <mi>X</mi> <mo>]</mo></mrow> <mo>+</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>μ</mi> <mn>2</mn></msup> <mo>]</mo></mrow></mrow></math>

<math alttext="equals double-struck upper E left-bracket upper X squared right-bracket minus 2 mu double-struck upper E left-bracket upper X right-bracket plus mu squared"><mrow><mo>=</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>X</mi> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>-</mo> <mn>2</mn> <mi>μ</mi> <mi>𝔼</mi> <mrow><mo>[</mo> <mi>X</mi> <mo>]</mo></mrow> <mo>+</mo> <msup><mi>μ</mi> <mn>2</mn></msup></mrow></math>

<math alttext="equals double-struck upper E left-bracket upper X squared right-bracket minus 2 double-struck upper E left-bracket upper X right-bracket squared plus double-struck upper E left-bracket upper X right-bracket squared"><mrow><mo>=</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>X</mi> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>-</mo> <mn>2</mn> <mi>𝔼</mi> <msup><mrow><mo>[</mo><mi>X</mi><mo>]</mo></mrow> <mn>2</mn></msup> <mo>+</mo> <mi>𝔼</mi> <msup><mrow><mo>[</mo><mi>X</mi><mo>]</mo></mrow> <mn>2</mn></msup></mrow></math>

<math alttext="equals double-struck upper E left-bracket upper X squared right-bracket minus double-struck upper E left-bracket upper X right-bracket squared"><mrow><mo>=</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>X</mi> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>-</mo> <mi>𝔼</mi> <msup><mrow><mo>[</mo><mi>X</mi><mo>]</mo></mrow> <mn>2</mn></msup></mrow></math>

让我们花点时间逐步进行这些步骤。在第一步中，我们通过经典的二项式展开完全表达随机变量作为其所有组成项。在第二步中，我们执行期望的线性性，将组成项分解为它们自己的单独期望。在第三步中，我们注意到<math alttext="mu"><mi>μ</mi></math>，或者<math alttext="double-struck upper E left-bracket upper X right-bracket"><mrow><mi>𝔼</mi> <mo>[</mo> <mi>X</mi> <mo>]</mo></mrow></math>，及其平方都是常数，因此可以从周围的期望中提取出来。它们是常数，因为它们不是值*X*的函数，而是使用整个域（*X*可以取的值集合）进行评估。常数可以看作是只能取一个值的随机变量，即常数本身。因此，它们的期望值，或者随机变量取值的平均值，就是常数本身，因为我们总是看到这个常数。最后的步骤是代数操作，将我们带到简化的结果。让我们使用这个公式来找到表示单个神经元在辍学下的二进制随机变量的方差，*p*是神经元被屏蔽的概率：

<math alttext="double-struck upper E left-bracket upper X squared right-bracket minus double-struck upper E left-bracket upper X right-bracket squared equals sigma-summation Underscript x element-of 0 comma 1 Endscripts x squared asterisk upper P left-parenthesis upper X equals x right-parenthesis minus left-parenthesis sigma-summation Underscript x element-of 0 comma 1 Endscripts x asterisk upper P left-parenthesis upper X equals x right-parenthesis right-parenthesis squared"><mrow><mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>X</mi> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>-</mo> <mi>𝔼</mi> <msup><mrow><mo>[</mo><mi>X</mi><mo>]</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msub><mo>∑</mo> <mrow><mi>x</mi><mo>∈</mo><mrow><mn>0</mn><mo>,</mo><mn>1</mn></mrow></mrow></msub> <msup><mi>x</mi> <mn>2</mn></msup> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>=</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>-</mo> <msup><mrow><mo>(</mo><msub><mo>∑</mo> <mrow><mi>x</mi><mo>∈</mo><mrow><mn>0</mn><mo>,</mo><mn>1</mn></mrow></mrow></msub> <mi>x</mi><mo>*</mo><mi>P</mi><mrow><mo>(</mo><mi>X</mi><mo>=</mo><mi>x</mi><mo>)</mo></mrow><mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>

<math alttext="equals sigma-summation Underscript x element-of 0 comma 1 Endscripts x squared asterisk upper P left-parenthesis upper X equals x right-parenthesis minus p squared"><mrow><mo>=</mo> <msub><mo>∑</mo> <mrow><mi>x</mi><mo>∈</mo><mrow><mn>0</mn><mo>,</mo><mn>1</mn></mrow></mrow></msub> <msup><mi>x</mi> <mn>2</mn></msup> <mo>*</mo> <mi>P</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>=</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>-</mo> <msup><mi>p</mi> <mn>2</mn></msup></mrow></math>

<math alttext="equals p minus p squared"><mrow><mo>=</mo> <mi>p</mi> <mo>-</mo> <msup><mi>p</mi> <mn>2</mn></msup></mrow></math>

<math alttext="equals p left-parenthesis 1 minus p right-parenthesis"><mrow><mo>=</mo> <mi>p</mi> <mo>(</mo> <mn>1</mn> <mo>-</mo> <mi>p</mi> <mo>)</mo></mrow></math>

这些简化应该是合理的。我们从“期望”中知道，表示神经元的二进制随机变量的期望值只是*p*，其余是代数简化。我们强烈鼓励您自己进行这些推导。当我们开始思考代表整个层中被屏蔽神经元数量的随机变量时，我们自然会问是否存在与期望相似的方差线性性质。不幸的是，该性质通常不成立：

<math alttext="upper V a r left-parenthesis upper A plus upper B right-parenthesis equals double-struck upper E left-bracket left-parenthesis upper A plus upper B right-parenthesis squared right-bracket minus double-struck upper E left-bracket upper A plus upper B right-bracket squared"><mrow><mi>V</mi> <mi>a</mi> <mi>r</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>+</mo> <mi>B</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mrow><mo>(</mo><mi>A</mi><mo>+</mo><mi>B</mi><mo>)</mo></mrow> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>-</mo> <mi>𝔼</mi> <msup><mrow><mo>[</mo><mi>A</mi><mo>+</mo><mi>B</mi><mo>]</mo></mrow> <mn>2</mn></msup></mrow></math>

<math alttext="equals double-struck upper E left-bracket upper A squared plus 2 asterisk upper A asterisk upper B plus upper B squared right-bracket minus left-parenthesis double-struck upper E left-bracket upper A right-bracket plus double-struck upper E left-bracket upper B right-bracket right-parenthesis squared"><mrow><mo>=</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>A</mi> <mn>2</mn></msup> <mo>+</mo> <mn>2</mn> <mo>*</mo> <mi>A</mi> <mo>*</mo> <mi>B</mi> <mo>+</mo> <msup><mi>B</mi> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>-</mo> <msup><mrow><mo>(</mo><mi>𝔼</mi><mrow><mo>[</mo><mi>A</mi><mo>]</mo></mrow><mo>+</mo><mi>𝔼</mi><mrow><mo>[</mo><mi>B</mi><mo>]</mo></mrow><mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>

<math alttext="equals double-struck upper E left-bracket upper A squared right-bracket plus 2 double-struck upper E left-bracket upper A asterisk upper B right-bracket plus double-struck upper E left-bracket upper B squared right-bracket minus double-struck upper E left-bracket upper A right-bracket squared minus 2 double-struck upper E left-bracket upper A right-bracket double-struck upper E left-bracket upper B right-bracket minus double-struck upper E left-bracket upper B right-bracket squared"><mrow><mo>=</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>A</mi> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>+</mo> <mn>2</mn> <mi>𝔼</mi> <mrow><mo>[</mo> <mi>A</mi> <mo>*</mo> <mi>B</mi> <mo>]</mo></mrow> <mo>+</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>B</mi> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>-</mo> <mi>𝔼</mi> <msup><mrow><mo>[</mo><mi>A</mi><mo>]</mo></mrow> <mn>2</mn></msup> <mo>-</mo> <mn>2</mn> <mi>𝔼</mi> <mrow><mo>[</mo> <mi>A</mi> <mo>]</mo></mrow> <mi>𝔼</mi> <mrow><mo>[</mo> <mi>B</mi> <mo>]</mo></mrow> <mo>-</mo> <mi>𝔼</mi> <msup><mrow><mo>[</mo><mi>B</mi><mo>]</mo></mrow> <mn>2</mn></msup></mrow></math>

<math alttext="equals double-struck upper E left-bracket upper A squared right-bracket minus double-struck upper E left-bracket upper A right-bracket squared plus double-struck upper E left-bracket upper B squared right-bracket minus double-struck upper E left-bracket upper B right-bracket squared plus 2 double-struck upper E left-bracket upper A asterisk upper B right-bracket minus 2 double-struck upper E left-bracket upper A right-bracket double-struck upper E left-bracket upper B right-bracket"><mrow><mo>=</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>A</mi> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>-</mo> <mi>𝔼</mi> <msup><mrow><mo>[</mo><mi>A</mi><mo>]</mo></mrow> <mn>2</mn></msup> <mo>+</mo> <mi>𝔼</mi> <mrow><mo>[</mo> <msup><mi>B</mi> <mn>2</mn></msup> <mo>]</mo></mrow> <mo>-</mo> <mi>𝔼</mi> <msup><mrow><mo>[</mo><mi>B</mi><mo>]</mo></mrow> <mn>2</mn></msup> <mo>+</mo> <mn>2</mn> <mi>𝔼</mi> <mrow><mo>[</mo> <mi>A</mi> <mo>*</mo> <mi>B</mi> <mo>]</mo></mrow> <mo>-</mo> <mn>2</mn> <mi>𝔼</mi> <mrow><mo>[</mo> <mi>A</mi> <mo>]</mo></mrow> <mi>𝔼</mi> <mrow><mo>[</mo> <mi>B</mi> <mo>]</mo></mrow></mrow></math>

<math alttext="equals upper V a r left-parenthesis upper A right-parenthesis plus upper V a r left-parenthesis upper B right-parenthesis plus 2 left-parenthesis double-struck upper E left-bracket upper A asterisk upper B right-bracket minus double-struck upper E left-bracket upper A right-bracket double-struck upper E left-bracket upper B right-bracket right-parenthesis"><mrow><mo>=</mo> <mi>V</mi> <mi>a</mi> <mi>r</mi> <mo>(</mo> <mi>A</mi> <mo>)</mo> <mo>+</mo> <mi>V</mi> <mi>a</mi> <mi>r</mi> <mo>(</mo> <mi>B</mi> <mo>)</mo> <mo>+</mo> <mn>2</mn> <mo>(</mo> <mi>𝔼</mi> <mo>[</mo> <mi>A</mi> <mo>*</mo> <mi>B</mi> <mo>]</mo> <mo>-</mo> <mi>𝔼</mi> <mo>[</mo> <mi>A</mi> <mo>]</mo> <mi>𝔼</mi> <mo>[</mo> <mi>B</mi> <mo>]</mo> <mo>)</mo></mrow></math>

<math alttext="equals upper V a r left-parenthesis upper A right-parenthesis plus upper V a r left-parenthesis upper B right-parenthesis plus 2 upper C o v left-parenthesis upper A comma upper B right-parenthesis"><mrow><mo>=</mo> <mi>V</mi> <mi>a</mi> <mi>r</mi> <mo>(</mo> <mi>A</mi> <mo>)</mo> <mo>+</mo> <mi>V</mi> <mi>a</mi> <mi>r</mi> <mo>(</mo> <mi>B</mi> <mo>)</mo> <mo>+</mo> <mn>2</mn> <mi>C</mi> <mi>o</mi> <mi>v</mi> <mo>(</mo> <mi>A</mi> <mo>,</mo> <mi>B</mi> <mo>)</mo></mrow></math>

正如我们从最后一行可以看到的那样，表达式中的最后一项，我们称之为两个随机变量之间的*协方差*，破坏了我们对线性的希望。然而，协方差是概率中的另一个关键概念——协方差的直觉是它衡量了两个随机变量之间的依赖关系。当一个随机变量更完全地确定另一个随机变量的值（想象*A*是一系列抛硬币的正面数量，*B*是同一系列抛硬币的反面数量），协方差的大小会增加。因此，可以推断，如果*A*和*B*是独立的随机变量，它们之间的协方差应该为零，在这种特殊情况下线性应该成立。我们强烈鼓励您通过数学来证明这一点。

回到辍学的例子，被屏蔽神经元的总数的方差可以分解为每个神经元的方差之和，因为每个神经元都是独立屏蔽的。被屏蔽神经元的数量的方差是*p(1-p)*n*，其中*p(1-p)*是任何给定神经元的方差，*n*是神经元的数量。辍学中的期望和方差使我们能够更深入地理解在深度神经网络中应用这样一个层时我们期望看到什么。

# 贝叶斯定理

回到我们关于条件概率的讨论，我们注意到两个事件之间的交集的概率可以写成条件分布和单个事件的分布的乘积。现在让我们将这个翻译成随机变量的语言，现在我们已经介绍了这个新术语。我们将*A*表示为一个随机变量，*B*表示第二个随机变量。让*a*是*A*可以取的值，*b*是*B*可以取的值。对于随机变量的交集操作的类比是*联合概率分布 P(A=a,B=b)*，表示*A=a*和*B=b*的事件。我们可以将*A=a*和*B=b*看作是单独的事件，当我们写*P(A=a,B=b)*时，我们考虑的是两个事件都发生的概率，即它们的交集。请注意，我们通常将联合概率分布写为*P(A,B)*，因为这包含了随机变量*A*和*B*的所有可能的联合设置。

我们之前提到，交集操作可以写成条件分布和单个事件的分布的乘积。将这个重写为随机变量的格式，我们有*P(A=a,B=b) = P(A=a|B=b)P(B=b)*。更一般地，考虑两个随机变量的所有可能的联合设置，我们有*P(A,B) = P(A|B)P(B)*。我们还讨论了总是存在第二种写这个联合分布的方法：*P(A=a,B=b) = P(B=b|A=a)P(A=a)*，更一般地，*P(A,B) = P(B|A)P(A)*。我们注意到有时其中一种路径比另一种更有意义。例如，在症状由*A*表示，疾病由*B*表示的情况下，*B*取一个值*b*，然后*A*在那个宇宙中取一个值*a*的路径比反向更有意义，因为从生物学上讲，人们先感染疾病，然后才表现出该疾病的症状。

然而，这并不意味着反向不是有用的。几乎普遍情况下，人们出现轻微症状时会去医院，医疗专业人员必须尝试从这些症状中推断出最可能的疾病以有效治疗潜在疾病。*贝叶斯定理*给了我们一种计算观察到的症状后疾病概率的方法。由于相同的联合概率分布可以用前一段提到的两种方式写出，我们有以下等式：

<math alttext="upper P left-parenthesis upper B vertical-bar upper A right-parenthesis equals StartFraction upper P left-parenthesis upper A vertical-bar upper B right-parenthesis upper P left-parenthesis upper B right-parenthesis Over upper P left-parenthesis upper A right-parenthesis EndFraction"><mrow><mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>|</mo> <mi>A</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>P</mi><mo>(</mo><mi>A</mi><mo>|</mo><mi>B</mi><mo>)</mo><mi>P</mi><mo>(</mo><mi>B</mi><mo>)</mo></mrow> <mrow><mi>P</mi><mo>(</mo><mi>A</mi><mo>)</mo></mrow></mfrac></mrow></math>

如果*B*代表疾病，而*A*代表症状，这为我们提供了一种计算观察到的症状后任何疾病可能性的方法。让我们分析右侧，看看等式是否也具有直观意义。疾病给定症状的可能性乘以疾病的可能性只是联合分布，这在这里作为分子是有意义的。分母是看到这些症状的可能性，这也可以表示为分子在所有可能疾病上求和。这是更一般过程的一个实例，称为*边缘化*，或通过对子集的所有可能配置求和来从联合分布中移除一部分随机变量：

<math alttext="upper P left-parenthesis upper A right-parenthesis equals sigma-summation Underscript b Endscripts upper P left-parenthesis upper A comma upper B equals b right-parenthesis"><mrow><mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mo>∑</mo> <mi>b</mi></msub> <mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>,</mo> <mi>B</mi> <mo>=</mo> <mi>b</mi> <mo>)</mo></mrow></mrow></math>

更简洁地说，我们有：

<math alttext="upper P left-parenthesis upper B equals b Subscript q u e r y Baseline vertical-bar upper A right-parenthesis equals StartFraction upper P left-parenthesis upper B equals b Subscript q u e r y Baseline comma upper A right-parenthesis Over sigma-summation Underscript b Endscripts upper P left-parenthesis upper B equals b comma upper A right-parenthesis EndFraction"><mrow><mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>=</mo> <msub><mi>b</mi> <mrow><mi>q</mi><mi>u</mi><mi>e</mi><mi>r</mi><mi>y</mi></mrow></msub> <mo>|</mo> <mi>A</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>P</mi><mo>(</mo><mi>B</mi><mo>=</mo><msub><mi>b</mi> <mrow><mi>q</mi><mi>u</mi><mi>e</mi><mi>r</mi><mi>y</mi></mrow></msub> <mo>,</mo><mi>A</mi><mo>)</mo></mrow> <mrow><msub><mo>∑</mo> <mi>b</mi></msub> <mi>P</mi><mrow><mo>(</mo><mi>B</mi><mo>=</mo><mi>b</mi><mo>,</mo><mi>A</mi><mo>)</mo></mrow></mrow></mfrac></mrow></math>

贝叶斯定理是概率在现实世界中非常有价值的应用，特别是在疾病预测的情况下。此外，如果我们用代表特定疾病检测结果的随机变量替换症状的随机变量，并用代表特定疾病存在的随机变量替换所有疾病的随机变量，我们可以使用贝叶斯定理推断在进行特定疾病检测后实际患有特定疾病的可能性。这是大多数医院面临的常见问题，尤其是在考虑到 COVID-19 的爆发时，这对流行病学尤为重要。

# 熵，交叉熵和 KL 散度

概率分布，根据定义，为我们提供了比较各种可能事件发生的可能性的方法。然而，即使我们知道最可能发生的事件（或事件），在进行实验时我们将看到各种各样的事件。在本节中，我们首先考虑定义一个单一度量，该度量包含概率分布中所有不确定性的问题，我们将其定义为分布的*熵*。

让我们设定以下情景。我是一名正在进行实验的研究人员。实验可能是简单的抛硬币或掷骰子。你正在记录实验的结果。我们在不同的房间里，但通过电话线连接。我进行实验并收到结果，然后通过电话将结果传达给你。你在笔记本中记录那个结果，你选择某个二进制字符串表示作为你写下的内容。作为一名抄写员，在这种情况下是必要的-我可能进行数百次试验，我的记忆是有限的，所以我无法记住所有试验的结果。

例如，如果我掷骰子，我们都不知道骰子的公平性，你可以将结果标记为“0”，“1”，“10”，“11”，“100”和“101”。每当我向你传达实验结果时，你将该结果的对应字符串表示添加到包含到目前为止所有结果的字符串的末尾。如果我掷出一个一，然后两个二，最后一个一，根据迄今为止定义的编码方案，你会写下“0110”。

在所有实验运行结束后，我与您开会，尝试将这个字符串“0110”解密为一系列结果，以供我的研究使用。然而，作为研究人员，我对这个字符串感到困惑——它代表一个一，接着两个二，最后一个一吗？还是代表一个一，接着一个二，再接着一个三？甚至是一个一，接着一个四，再接着一个一？看起来至少有几种可能的翻译方式可以使用编码方案将这个字符串转换为结果。

为了防止这种情况再次发生，我们决定对您用于表示结果的二进制字符串施加一些限制。我们使用所谓的*前缀编码*，它不允许不同结果的二进制字符串表示成为彼此的前缀。不难理解为什么这会导致字符串到结果的唯一翻译。假设我们有一个二进制字符串，其中的某个前缀我们已成功解码为一系列结果。要解码剩余的字符串，或者后缀，我们必须首先找到系列中的下一个结果。当我们找到这个后缀的前缀被翻译为一个结果时，我们已经知道，根据定义，没有更小的前缀可以翻译为有效的结果。现在我们有一个更大的前缀二进制字符串已成功翻译为一系列结果。然后我们递归使用这种逻辑，直到达到字符串的末尾。

现在我们有了一些关于结果的字符串表示的指导方针，我们使用“0”表示一，使用“10”表示二，使用“110”表示三，使用“1110”表示四，使用“11110”表示五，使用“111110”表示六重新进行原始实验。然而，正如前面提到的，我可能进行数百次试验，作为抄写员，您可能希望限制您需要写入的数量。在没有关于骰子的信息的情况下，我们无法做得比这更好。假设每个结果出现的概率为 1/6，您每次试验需要写下的预期字母数量为 3.5。例如，如果我们将一设置为“000”，将二设置为“001”，将三设置为“010”，将四设置为“011”，将五设置为“100”，将六设置为“101”，我们可以降至 3。

但如果我们知道有关骰子的信息呢？例如，如果它是一个加权骰子，几乎总是出现六点？在这种情况下，您可能希望为六分配一个更短的二进制字符串，例如“0”（而不是将“0”分配给一），这样您就可以限制您需要写入的预期数量。直观地讲，随着任何单次试验的结果变得越来越确定，通过将最可能的结果分配给最短的二进制字符串，您需要写入的预期字符数量就会降低。

这引发了一个问题：在结果上给定一个概率分布，什么是最佳的编码方案，其中最佳被定义为每次试验需要写入的最少预期字符数量？尽管整个情况可能有点刻意，但它为我们提供了一个稍微不同的视角，通过它我们可以理解概率分布中的不确定性。正如我们所指出的，随着实验结果变得越来越确定，最佳的编码方案将允许抄写员每次试验的预期字符数量变得越来越少。例如，在极端情况下，如果我们事先知道六点总是会出现，抄写员就不需要写任何东西。

事实证明，尽管我们在这里不会展示，但你可以做的最好的事情是为每个可能结果*x_i*分配一个长度为*log_2(1/p(x_i))*的二进制字符串，其中*p(x_i)*是其概率。然后，任何给定试验的预期字符串长度将是：

<math alttext="double-struck upper E Subscript p left-parenthesis x right-parenthesis Baseline left-bracket log Subscript 2 Baseline StartFraction 1 Over p left-parenthesis x right-parenthesis EndFraction right-bracket equals sigma-summation Underscript x Subscript i Baseline Endscripts p left-parenthesis x Subscript i Baseline right-parenthesis log Subscript 2 Baseline StartFraction 1 Over p left-parenthesis x Subscript i Baseline right-parenthesis EndFraction"><mrow><msub><mi>𝔼</mi> <mrow><mi>p</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></msub> <mrow><mo>[</mo> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mfrac><mn>1</mn> <mrow><mi>p</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mfrac> <mo>]</mo></mrow> <mo>=</mo> <msub><mo>∑</mo> <msub><mi>x</mi> <mi>i</mi></msub></msub> <mi>p</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mfrac><mn>1</mn> <mrow><mi>p</mi><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mfrac></mrow></math>

<math alttext="equals minus sigma-summation Underscript x Subscript i Baseline Endscripts p left-parenthesis x Subscript i Baseline right-parenthesis log Subscript 2 Baseline p left-parenthesis x Subscript i Baseline right-parenthesis"><mrow><mo>=</mo> <mo>-</mo> <msub><mo>∑</mo> <msub><mi>x</mi> <mi>i</mi></msub></msub> <mi>p</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mi>p</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></math>

这个表达式被定义为概率分布的*熵*。在我们完全确定最终结果的情况下（例如，骰子总是掷出六点），我们可以评估熵的表达式，看到我们得到的结果是 0。

在我们完全确定最终结果的情况下（例如，骰子总是掷出六点），我们可以评估熵的表达式，看到我们得到的结果是 0。此外，具有最高熵的概率分布是将等概率分布在所有可能结果上的分布。这是因为对于任何给定的试验，我们对某个特定结果出现与其他结果出现一样的确定性。因此，我们不能使用将较短的字符串分配给任何单个结果的策略。

现在我们已经定义了熵，我们可以讨论交叉熵，它为我们提供了一种衡量两个分布之间差异的方法。

##### 方程 2-1. 交叉熵

<math alttext="upper C upper E left-parenthesis p StartAbsoluteValue EndAbsoluteValue q right-parenthesis equals double-struck upper E Subscript p left-parenthesis x right-parenthesis Baseline left-bracket log Subscript 2 Baseline StartFraction 1 Over q left-parenthesis x right-parenthesis EndFraction right-bracket equals sigma-summation Underscript x Endscripts p left-parenthesis x right-parenthesis log Subscript 2 Baseline StartFraction 1 Over q left-parenthesis x right-parenthesis EndFraction equals minus sigma-summation Underscript x Endscripts p left-parenthesis x right-parenthesis log Subscript 2 Baseline q left-parenthesis x right-parenthesis"><mrow><mi>C</mi> <mi>E</mi> <mrow><mo>(</mo> <mi>p</mi> <mo>|</mo> <mo>|</mo> <mi>q</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>p</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></msub> <mrow><mo>[</mo> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mfrac><mn>1</mn> <mrow><mi>q</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mfrac> <mo>]</mo></mrow> <mo>=</mo> <msub><mo>∑</mo> <mi>x</mi></msub> <mi>p</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mfrac><mn>1</mn> <mrow><mi>q</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mfrac> <mo>=</mo> <mo>-</mo> <msub><mo>∑</mo> <mi>x</mi></msub> <mi>p</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mrow><mi>q</mi> <mo>(</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>

注意交叉熵中有一个项，可以解释为对每个结果分配的最佳二进制字符串长度，假设结果按概率分布*q(x)*出现。然而，请注意这是相对于*p(x)*的期望，那么我们如何解释整个表达式呢？嗯，我们可以理解交叉熵是指在为分布*q(x)*优化编码方案的情况下，对于任何试验的预期字符串长度，而实际上，所有结果都是根据分布*p(x)*出现的。这在实验中肯定会发生，因为我们对实验的先验信息有限，所以我们假设某个分布*q(x)*来优化我们的编码方案，但随着我们进行试验，我们学到了更多信息，使我们更接近真实分布*p(x)*。

KL 散度将这种逻辑推得更远。如果我们取交叉熵，告诉我们在为不正确的分布*q(x)*优化我们的编码时每次试验预期的比特数，然后从中减去熵，告诉我们在为正确的分布*p(x)*优化时每次试验预期的比特数，我们得到了使用*q(x)*比*p(x)*时表示试验所需的额外比特数的预期值。以下是 KL 散度的表达式：

<math alttext="upper K upper L left-parenthesis p StartAbsoluteValue EndAbsoluteValue q right-parenthesis equals double-struck upper E Subscript p left-parenthesis x right-parenthesis Baseline left-bracket log Subscript 2 Baseline StartFraction 1 Over q left-parenthesis x right-parenthesis EndFraction minus log Subscript 2 Baseline StartFraction 1 Over p left-parenthesis x right-parenthesis EndFraction right-bracket equals double-struck upper E Subscript p left-parenthesis x right-parenthesis Baseline left-bracket log Subscript 2 Baseline StartFraction p left-parenthesis x right-parenthesis Over q left-parenthesis x right-parenthesis EndFraction right-bracket"><mrow><mi>K</mi> <mi>L</mi> <mrow><mo>(</mo> <mi>p</mi> <mo>|</mo> <mo>|</mo> <mi>q</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>p</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></msub> <mrow><mo>[</mo> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mfrac><mn>1</mn> <mrow><mi>q</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mfrac> <mo>-</mo> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mfrac><mn>1</mn> <mrow><mi>p</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mfrac> <mo>]</mo></mrow> <mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>p</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></msub> <mrow><mo>[</mo> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mfrac><mrow><mi>p</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow> <mrow><mi>q</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mfrac> <mo>]</mo></mrow></mrow></math>

在唯一的全局最小值*q(x)* = *p(x)*处，KL 散度恰好为零。为什么这是唯一的最小值有点超出了本文的范围，所以我们把它留给你作为一个练习。

在实践中，当试图将真实分布*p(x)*与学习分布*q(x)*匹配时，KL 散度通常被最小化作为目标函数。大多数模型实际上会最小化交叉熵来代替 KL 散度，这实际上是相同的优化问题，因为 KL 是交叉熵和*p(x)*的熵之间的差异，其中*p(x)*的熵是一个常数，不依赖于参数化*q(x)*的权重。因此，使用任一目标时，对参数化*q(x)*的权重的梯度是相同的。

一个常见的例子是交叉熵/KL 散度被优化的标准神经网络分类器的训练。神经网络的目标是学习一个关于目标类别的分布，使得对于任何给定的例子*x[i]*，<mrow><msub><mi>p</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>y</mi> <mo>|</mo> <mi>x</mi> <mo>=</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow>与真实分布*p(y|x* *= x[i])*匹配，真实分布将所有的概率质量放在真实标签*y[i]*上，其他类别的概率为零。最小化学习分布和所有例子上真实分布之间的交叉熵之和实际上与最小化数据的负对数似然度是完全相同的。这两种都是神经网络训练的有效解释，并导致相同的目标函数。我们鼓励您尝试独立编写这两个表达式，以了解这一点。

# 连续概率分布

到目前为止，我们已经通过离散结果和事件的视角看了概率分布。然而，事实证明，概率分布不仅适用于像 CIFAR-10 目标类别或 MNIST 数字这样的离散结果集。我们可以定义在无限大小样本空间上的概率分布，比如所有的实数。在本节中，我们将扩展前几节涵盖的原则到连续领域。

在连续领域中，概率分布通常被称为*概率密度函数*，或者 PDF。 PDF 是在样本空间上的非负函数，比如所有的实数，积分为 1。回想一下微积分中函数的积分是函数下方区域的面积，由*x*轴界定。 PDF 遵循第一节介绍的基本原则，但是不是将结果的概率相加来得到事件的概率，而是使用积分。例如，假设*X*是一个定义在所有实数上的连续随机变量。如果我们想要知道事件<mrow><mi>P</mi> <mo>(</mo> <mi>X</mi> <mo>≤</mo> <mn>2</mn> <mo>)</mo></mrow>的概率，我们只需要从负无穷积分*X*的 PDF 到 2。

但是对于任何个体结果的概率，比如*P*(*X* = 2)呢？由于我们在连续空间中使用积分来找到概率，任何个体结果的概率实际上是零，因为区域的宽度是无穷小的。我们使用术语*似然度*来区分事件的概率和当我们输入*X*的设置时 PDF 评估的值。似然度仍然很有价值，因为它告诉我们在连续空间中进行实验时最有可能看到的个体结果。在考虑连续概率分布时，我们将只提到事件具有概率，而不是个体结果。

一个著名的连续概率分布例子是实线上某个区间上的*均匀分布*。在均匀分布下，每个结果的可能性相同，意味着没有任何结果比另一个更有可能出现。因此，均匀分布看起来像一个矩形，其中矩形的底部是构成其分布域的区间，高度或每个结果的可能性是使矩形的面积等于 1 的值。图 2-4 显示了区间[0,0.5]上的均匀分布。

![](img/fdl2_0204.png)

###### 图 2-4\. 均匀分布在整个区域上具有均匀高度，显示分布域中的每个值具有相等的可能性。

选择这个例子特别是为了显示连续领域中可能性和概率之间的具体差异。矩形的高度为 2 并非错误——在连续分布中，可能性的幅度没有限制，不像概率必须小于或等于 1。

另一个著名的连续概率分布例子是*高斯分布*，这是数据在现实世界中呈现的更常见方式之一。高斯分布由两个参数定义：其均值<math alttext="mu"><mi>μ</mi></math>和标准差<math alttext="sigma"><mi>σ</mi></math>。高斯分布的概率密度函数为：

<math alttext="f left-parenthesis x semicolon mu comma sigma right-parenthesis equals StartFraction 1 Over sigma StartRoot 2 pi EndRoot EndFraction e Superscript minus one-half left-parenthesis StartFraction x minus mu Over sigma EndFraction right-parenthesis squared"><mrow><mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>;</mo> <mi>μ</mi> <mo>,</mo> <mi>σ</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mn>1</mn> <mrow><mi>σ</mi><msqrt><mrow><mn>2</mn><mi>π</mi></mrow></msqrt></mrow></mfrac> <msup><mi>e</mi> <mrow><mo>-</mo><mfrac><mn>1</mn> <mn>2</mn></mfrac><msup><mrow><mo>(</mo><mfrac><mrow><mi>x</mi><mo>-</mo><mi>μ</mi></mrow> <mi>σ</mi></mfrac><mo>)</mo></mrow> <mn>2</mn></msup></mrow></msup></mrow></math>

为什么这个函数在实域上积分为 1 超出了本章的范围，但高斯分布的一个重要特征是其均值也是其唯一的众数。换句话说，具有最高可能性的结果也是唯一的平均结果。并非所有分布都具有这种特性。例如，图 2-4 没有这种特性。标准高斯图的图形，其均值为零，方差为单位，显示在图 2-5 中（在两个方向的极限中，概率密度函数渐近地趋近于零）。

![](img/fdl2_0205.png)

###### 图 2-5\. 高斯分布呈钟形，中心具有最高可能性，随着问题值越来越远离中心，呈指数级下降。

为什么高斯分布在现实世界的数据中如此普遍？其中一个原因是一个被称为*中心极限定理*（CLT）的定理。该定理指出，独立随机变量的和在和中的变量数趋于无穷时收敛于高斯分布，即使每个变量并非高斯分布。一个例子是应用了辍学层后掩码神经元的数量。当前一层的神经元数量趋于无穷时，掩码神经元的数量（作为独立伯努利随机变量的和，如“随机变量”中讨论的那样），在正确标准化时，近似地分布为标准高斯分布。我们不会在这里深入讨论 CLT，但最近已将其扩展到在某些特殊条件下的弱相关变量。

许多现实世界的数据集可以被看作是许多随机变量的近似和。例如，在给定人口中疾病患病率的分布，类似于应用辍学后掩码神经元的数量，是许多伯努利随机变量的和（其中每个人是一个伯努利随机变量，如果他们患有疾病则值为 1，如果他们没有则值为 0）—尽管可能相关。

连续随机变量仍然是函数，就像我们定义的离散随机变量一样。唯一的区别是这个函数的范围是一个连续空间。要计算连续随机变量的期望和方差，我们只需要用积分替换我们的求和，如下所示：

<math alttext="double-struck upper E left-bracket upper X right-bracket equals integral Underscript x Endscripts x asterisk f left-parenthesis upper X equals x right-parenthesis d x"><mrow><mi>𝔼</mi> <mrow><mo>[</mo> <mi>X</mi> <mo>]</mo></mrow> <mo>=</mo> <msub><mo>∫</mo> <mi>x</mi></msub> <mi>x</mi> <mo>*</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>=</mo> <mi>x</mi> <mo>)</mo></mrow> <mi>d</mi> <mi>x</mi></mrow></math>

<math alttext="upper V a r left-parenthesis upper X right-parenthesis equals integral Underscript x Endscripts left-parenthesis x minus double-struck upper E left-bracket upper X right-bracket right-parenthesis squared asterisk f left-parenthesis upper X equals x right-parenthesis d x"><mrow><mi>V</mi> <mi>a</mi> <mi>r</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mo>∫</mo> <mi>x</mi></msub> <msup><mrow><mo>(</mo><mi>x</mi><mo>-</mo><mi>𝔼</mi><mrow><mo>[</mo><mi>X</mi><mo>]</mo></mrow><mo>)</mo></mrow> <mn>2</mn></msup> <mo>*</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>=</mo> <mi>x</mi> <mo>)</mo></mrow> <mi>d</mi> <mi>x</mi></mrow></math>

举个例子，让我们评估之前定义的均匀随机变量的期望。但首先，请确认期望值为 0.25 是直观合理的，因为区间的端点是 0 和 0.5，中间的所有值具有相等的可能性。现在，让我们计算积分，看看计算结果是否符合我们的直觉：

<math alttext="integral Subscript 0 Superscript 0.5 Baseline x asterisk f left-parenthesis x right-parenthesis d x equals integral Subscript 0 Superscript 0.5 Baseline 2 x d x"><mrow><msubsup><mo>∫</mo> <mn>0</mn> <mrow><mn>0</mn><mo>.</mo><mn>5</mn></mrow></msubsup> <mi>x</mi> <mo>*</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mi>d</mi> <mi>x</mi> <mo>=</mo> <msubsup><mo>∫</mo> <mn>0</mn> <mrow><mn>0</mn><mo>.</mo><mn>5</mn></mrow></msubsup> <mn>2</mn> <mi>x</mi> <mi>d</mi> <mi>x</mi></mrow></math>

<math alttext="equals x squared vertical-bar Subscript 0 Baseline Superscript 0.5 Baseline"><mrow><mo>=</mo> <msup><mi>x</mi> <mn>2</mn></msup> <msubsup><mrow><mo>|</mo></mrow> <mn>0</mn> <mrow><mn>0</mn><mo>.</mo><mn>5</mn></mrow></msubsup></mrow></math>

<math alttext="equals 0.25"><mrow><mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>25</mn></mrow></math>

其中|符号的上标和下标表示我们将评估前面函数的值，然后取差以得到积分的值。我们看到期望值与我们的直觉相同，这是一个很好的检查。

贝叶斯定理也适用于连续变量。唯一的主要区别是在边缘化一部分变量时，您需要对被边缘化子集的整个域进行积分，而不是对被边缘化子集的所有可能配置进行离散求和。再次，这是将概率原则扩展到连续空间的一个例子，通过用积分取代求和。以下是连续概率分布的贝叶斯定理，遵循“贝叶斯定理”中的符号表示：

<math alttext="upper P left-parenthesis upper B equals b Subscript q u e r y Baseline vertical-bar upper A right-parenthesis equals StartFraction upper P left-parenthesis upper A vertical-bar upper B equals b Subscript q u e r y Baseline right-parenthesis upper P left-parenthesis upper B equals b Subscript q u e r y Baseline right-parenthesis Over upper P left-parenthesis upper A right-parenthesis EndFraction equals StartFraction upper P left-parenthesis upper A vertical-bar upper B equals b Subscript q u e r y Baseline right-parenthesis upper P left-parenthesis upper B equals b Subscript q u e r y Baseline right-parenthesis Over integral Underscript b Endscripts upper P left-parenthesis upper A comma upper B equals b right-parenthesis d b EndFraction"><mrow><mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>=</mo> <msub><mi>b</mi> <mrow><mi>q</mi><mi>u</mi><mi>e</mi><mi>r</mi><mi>y</mi></mrow></msub> <mo>|</mo> <mi>A</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>P</mi><mrow><mo>(</mo><mi>A</mi><mo>|</mo><mi>B</mi><mo>=</mo><msub><mi>b</mi> <mrow><mi>q</mi><mi>u</mi><mi>e</mi><mi>r</mi><mi>y</mi></mrow></msub> <mo>)</mo></mrow><mi>P</mi><mrow><mo>(</mo><mi>B</mi><mo>=</mo><msub><mi>b</mi> <mrow><mi>q</mi><mi>u</mi><mi>e</mi><mi>r</mi><mi>y</mi></mrow></msub> <mo>)</mo></mrow></mrow> <mrow><mi>P</mi><mo>(</mo><mi>A</mi><mo>)</mo></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>P</mi><mrow><mo>(</mo><mi>A</mi><mo>|</mo><mi>B</mi><mo>=</mo><msub><mi>b</mi> <mrow><mi>q</mi><mi>u</mi><mi>e</mi><mi>r</mi><mi>y</mi></mrow></msub> <mo>)</mo></mrow><mi>P</mi><mrow><mo>(</mo><mi>B</mi><mo>=</mo><msub><mi>b</mi> <mrow><mi>q</mi><mi>u</mi><mi>e</mi><mi>r</mi><mi>y</mi></mrow></msub> <mo>)</mo></mrow></mrow> <mrow><msub><mo>∫</mo> <mi>b</mi></msub> <mi>P</mi><mrow><mo>(</mo><mi>A</mi><mo>,</mo><mi>B</mi><mo>=</mo><mi>b</mi><mo>)</mo></mrow><mi>d</mi><mi>b</mi></mrow></mfrac></mrow></math>

最后，我们将讨论熵、交叉熵和 KL 散度。这三者在连续空间中也很好地延伸。我们用积分取代求和，并注意到在前一节介绍的属性仍然成立。例如，在给定域上，熵最高的分布是均匀分布，而两个分布之间的 KL 散度仅在两个分布完全相同时为零。以下是它们的连续形式定义，遵循方程 2-1：

<math alttext="upper H left-parenthesis f left-parenthesis x right-parenthesis right-parenthesis equals minus integral Underscript x Endscripts f left-parenthesis x right-parenthesis log Subscript 2 Baseline f left-parenthesis x right-parenthesis d x"><mrow><mi>H</mi> <mrow><mo>(</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mo>-</mo> <msub><mo>∫</mo> <mi>x</mi></msub> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mi>d</mi> <mi>x</mi></mrow></math>

<math alttext="upper K upper L left-parenthesis f left-parenthesis x right-parenthesis StartAbsoluteValue EndAbsoluteValue g left-parenthesis x right-parenthesis right-parenthesis equals integral Underscript x Endscripts f left-parenthesis x right-parenthesis log Subscript 2 Baseline StartFraction f left-parenthesis x right-parenthesis Over g left-parenthesis x right-parenthesis EndFraction d x"><mrow><mi>K</mi> <mi>L</mi> <mrow><mo>(</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>|</mo> <mo>|</mo> <mi>g</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <msub><mo>∫</mo> <mi>x</mi></msub> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mfrac><mrow><mi>f</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow> <mrow><mi>g</mi><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mfrac> <mi>d</mi> <mi>x</mi></mrow></math>

<math alttext="upper C upper E left-parenthesis f left-parenthesis x right-parenthesis StartAbsoluteValue EndAbsoluteValue g left-parenthesis x right-parenthesis right-parenthesis equals minus integral Underscript x Endscripts f left-parenthesis x right-parenthesis log Subscript 2 Baseline g left-parenthesis x right-parenthesis d x"><mrow><mi>C</mi> <mi>E</mi> <mrow><mo>(</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>|</mo> <mo>|</mo> <mi>g</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mo>-</mo> <msub><mo>∫</mo> <mi>x</mi></msub> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <msub><mo form="prefix">log</mo> <mn>2</mn></msub> <mi>g</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mi>d</mi> <mi>x</mi></mrow></math>

我们将这些概念扩展到连续空间将在第十章中派上用场，我们将许多分布建模为高斯分布。此外，我们将 KL 散度/交叉熵术语作为我们学习分布复杂性的正则化程序。由于只有在查询分布与目标分布匹配时 KL 散度为零，将目标分布设置为高斯分布会迫使学习分布逼近高斯分布。

# 总结

在本章中，我们涵盖了概率的基础知识，首先建立了概率分布基础背后的直觉，然后转向概率的相关应用，如条件概率、随机变量、期望和方差。我们看到了概率在深度学习中的应用，比如神经网络在分类任务中参数化概率分布的方式，以及我们如何量化 dropout 这种神经网络中的正则化技术的数学属性。最后，我们讨论了概率分布中的不确定性测量，如熵，并将这些概念推广到连续领域。

概率是一个影响我们日常生活选择的领域，理解数字背后的含义至关重要。此外，我们希望这个介绍能让你更好地理解未来概念，并将本书的其余部分放在透视中。在下一章中，我们将讨论神经网络的结构以及它们设计背后的动机。
