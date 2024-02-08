# 第三章：使用 TensorFlow 进行线性和逻辑回归

本章将向您展示如何在 TensorFlow 中构建简单但非平凡的学习系统示例。本章的第一部分回顾了构建学习系统的数学基础，特别涵盖了函数、连续性和可微性。我们介绍了损失函数的概念，然后讨论了机器学习如何归结为找到复杂损失函数的最小点的能力。然后我们介绍了梯度下降的概念，并解释了如何使用它来最小化损失函数。我们最后简要讨论了自动微分的算法思想。第二部分重点介绍了这些数学思想支撑的 TensorFlow 概念。这些概念包括占位符、作用域、优化器和 TensorBoard，可以实现学习系统的实际构建和分析。最后一部分提供了如何在 TensorFlow 中训练线性和逻辑回归模型的案例研究。

本章很长，介绍了许多新的概念。如果您在第一次阅读时没有完全理解这些概念的微妙之处，那没关系。我们建议继续前进，以后有需要时再回来参考这里的概念。我们将在本书的其余部分中反复使用这些基础知识，以便让这些思想逐渐沉淀。

# 数学复习

本节回顾了概念上理解机器学习所需的数学工具。我们试图尽量减少所需的希腊符号数量，而是专注于建立概念理解而不是技术操作。

## 函数和可微性

本节将为您提供函数和可微性概念的简要概述。函数*f*是将输入映射到输出的规则。所有计算机编程语言中都有函数，数学上对函数的定义实际上并没有太大不同。然而，在物理学和工程学中常用的数学函数具有其他重要属性，如连续性和可微性。连续函数，粗略地说，是可以在不从纸上抬起铅笔的情况下绘制的函数，如图 3-1 所示。（这当然不是技术定义，但它捕捉了连续性条件的精神。）

![continuous_1.gif](img/tfdl_0301.png)

###### 图 3-1。一些连续函数。

可微性是函数上的一种平滑条件。它表示函数中不允许有尖锐的角或转折（图 3-2）。

![Math_images_4.jpg](img/tfdl_0302.png)

###### 图 3-2。一个可微函数。

可微函数的关键优势在于我们可以利用函数在特定点的斜率作为指导，找到函数高于或低于当前位置的地方。这使我们能够找到函数的*最小值*。可微函数*f*的*导数*，表示为<math><msup><mi>f</mi> <mo>'</mo></msup></math>，是另一个函数，提供原始函数在所有点的斜率。概念上，函数在给定点的导数指示了函数高于或低于当前值的方向。优化算法可以遵循这个指示牌，向* f *的最小值靠近。在最小值处，函数的导数为零。

最初，导数驱动的优化的力量并不明显。几代微积分学生都在纸上进行枯燥的最小化函数练习中受苦。这些练习并不有用，因为找到具有少量输入参数的函数的最小值是一个最好通过图形方式完成的微不足道的练习。导数驱动的优化的力量只有在有数百、数千、数百万或数十亿个变量时才会显现出来。在这些规模上，通过解析理解函数几乎是不可能的，所有的可视化都是充满风险的练习，很可能会忽略函数的关键属性。在这些规模上，函数的*梯度*，一个多变量函数的<math><msup><mi>f</mi> <mo>'</mo></msup></math>的推广，很可能是理解函数及其行为的最强大的数学工具。我们将在本章后面更深入地探讨梯度。（概念上是这样；我们不会在这项工作中涵盖梯度的技术细节。）

在非常高的层面上，机器学习只是函数最小化的行为：学习算法只不过是适当定义的函数的最小值查找器。这个定义具有数学上的简单性优势。但是，这些特殊的可微函数是什么，它们如何在它们的最小值中编码有用的解决方案，我们如何找到它们呢？

## 损失函数

为了解决给定的机器学习问题，数据科学家必须找到一种构建函数的方法，其最小值编码了手头的现实世界问题的解决方案。幸运的是，对于我们这位不幸的数据科学家来说，机器学习文献已经建立了一个丰富的*损失函数*历史，执行这种编码。实际机器学习归结为理解不同类型的可用损失函数，并知道应该将哪种损失函数应用于哪些问题。换句话说，损失函数是将数据科学项目转化为数学的机制。所有的机器学习，以及大部分人工智能，都归结为创建正确的损失函数来解决手头的问题。我们将为您介绍一些常见的损失函数家族。

我们首先注意到，损失函数<math alttext="script upper L"><mi>ℒ</mi></math>必须满足一些数学属性才能有意义。首先，<math alttext="script upper L"><mi>ℒ</mi></math>必须使用数据点*x*和标签*y*。我们通过将损失函数写成<math><mrow><mi>ℒ</mi> <mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow></math>来表示这一点。使用我们在上一章中的术语，*x*和*y*都是张量，<math alttext="script upper L"><mi>ℒ</mi></math>是从张量对到标量的函数。损失函数的函数形式应该是什么？人们常用的一个假设是使损失函数*可加性*。假设<math alttext="left-parenthesis x Subscript i Baseline comma y Subscript i Baseline right-parenthesis"><mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>,</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>)</mo></mrow></math>是示例*i*的可用数据，并且总共有*N*个示例。那么损失函数可以分解为

<math display="block"><mrow><mi>ℒ</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>N</mi></munderover> <msub><mi>ℒ</mi> <mi>i</mi></msub> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>,</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></math>

（在实践中，<math><msub><mi>ℒ</mi> <mi>i</mi></msub></math> 对于每个数据点都是相同的。）这种加法分解带来了许多有用的优势。首先是导数通过加法因子化，因此计算总损失的梯度简化如下：

<math display="block"><mrow><mi>∇</mi> <mi>ℒ</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>N</mi></munderover> <mi>∇</mi> <msub><mi>ℒ</mi> <mi>i</mi></msub> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>,</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></math>

这种数学技巧意味着只要较小的函数<math><msub><mi>ℒ</mi> <mi>i</mi></msub></math>是可微的，总损失函数也将是可微的。由此可见，设计损失函数的问题归结为设计较小函数<math alttext="script upper L Subscript i Baseline left-parenthesis x Subscript i Baseline comma y Subscript i Baseline right-parenthesis"><mrow><msub><mi>ℒ</mi> <mi>i</mi></msub> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>,</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></math>。在我们深入设计<math alttext="script upper L Subscript i"><msub><mi>ℒ</mi> <mi>i</mi></msub></math>之前，我们将方便地进行一个小的旁观，解释分类和回归问题之间的区别。

### 分类和回归

机器学习算法可以广泛地分为监督或无监督问题。监督问题是指数据点*x*和标签*y*都是可用的问题，而无监督问题只有数据点*x*没有标签*y*。一般来说，无监督机器学习更加困难且定义不明确（“理解”数据点*x*是什么意思？）。我们暂时不会深入讨论无监督损失函数，因为在实践中，大多数无监督损失都是巧妙地重新利用监督损失。

监督机器学习可以分为分类和回归两个子问题。分类问题是指您试图设计一个机器学习系统，为给定的数据点分配一个离散标签，比如 0/1（或更一般地<math><mrow><mn>0</mn> <mo>,</mo> <mo>⋯</mo> <mo>,</mo> <mi>n</mi></mrow></math>）。回归是指设计一个机器学习系统，为给定的数据点附加一个实值标签（在<math alttext="double-struck upper R"><mi>ℝ</mi></math>）。

从高层来看，这些问题可能看起来相当不同。离散对象和连续对象通常在数学和常识上被不同对待。然而，机器学习中使用的一种技巧是使用连续、可微的损失函数来编码分类和回归问题。正如我们之前提到的，机器学习的很大一部分就是将复杂的现实系统转化为适当简单的可微函数的艺术。

在接下来的章节中，我们将向您介绍一对数学函数，这对函数将非常有用，可以将分类和回归任务转换为适当的损失函数。

### L²损失

*L*²损失（读作*ell-two*损失）通常用于回归问题。*L*²损失（或者在其他地方通常称为*L*²范数）提供了一个向量大小的度量：

<math display="block"><mrow><msub><mrow><mo>∥</mo><mi>a</mi><mo>∥</mo></mrow> <mn>2</mn></msub> <mo>=</mo> <msqrt><mrow><msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>N</mi></msubsup> <msubsup><mi>a</mi> <mi>i</mi> <mn>2</mn></msubsup></mrow></msqrt></mrow></math>

在这里，*a*被假定为长度为*N*的向量。*L*²范数通常用来定义两个向量之间的距离：

<math display="block"><mrow><msub><mrow><mo>∥</mo><mi>a</mi><mo>-</mo><mi>b</mi><mo>∥</mo></mrow> <mn>2</mn></msub> <mo>=</mo> <msqrt><mrow><msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>N</mi></msubsup> <msup><mrow><mo>(</mo><msub><mi>a</mi> <mi>i</mi></msub> <mo>-</mo><msub><mi>b</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mn>2</mn></msup></mrow></msqrt></mrow></math>

*L*²作为距离测量的概念在解决监督机器学习中的回归问题时非常有用。假设*x*是一组数据，*y*是相关标签。让*f*是一些可微函数，编码我们的机器学习模型。然后为了鼓励*f*预测*y*，我们创建*L*²损失函数。

<math display="block"><mrow><mi>ℒ</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mrow><mo>∥</mo><mi>f</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow><mo>-</mo><mi>y</mi><mo>∥</mo></mrow> <mn>2</mn></msub></mrow></math>

作为一个快速说明，在实践中通常不直接使用*L*²损失，而是使用它的平方。

<math display="block"><mrow><msubsup><mrow><mo>∥</mo><mi>a</mi><mo>-</mo><mi>b</mi><mo>∥</mo></mrow> <mn>2</mn> <mn>2</mn></msubsup> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>N</mi></munderover> <msup><mrow><mo>(</mo><msub><mi>a</mi> <mi>i</mi></msub> <mo>-</mo><msub><mi>b</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>

为了避免在梯度中处理形式为<math alttext="1 slash StartRoot left-parenthesis EndRoot x right-parenthesis"><mrow><mn>1</mn> <mo>/</mo> <msqrt><mo>(</mo></msqrt> <mrow><mi>x</mi> <mo>)</mo></mrow></mrow></math>的术语。我们将在本章和本书的其余部分中反复使用平方*L*²损失。

### 概率分布

在介绍分类问题的损失函数之前，介绍概率分布将会很有用。首先，什么是概率分布，为什么我们应该关心它对机器学习有什么作用？概率是一个深奥的主题，因此我们只会深入到您获得所需的最低理解为止。在高层次上，概率分布提供了一个数学技巧，允许您将一组离散选择放松为一个连续的选择。例如，假设您需要设计一个机器学习系统，预测硬币是正面朝上还是反面朝上。看起来正面朝上/朝下似乎无法编码为连续函数，更不用说可微函数了。那么您如何使用微积分或 TensorFlow 的机制来解决涉及离散选择的问题呢？

进入概率分布。与硬选择不同，让分类器预测正面朝上或反面朝上的机会。例如，分类器可能学习预测正面的概率为 0.75，反面的概率为 0.25。请注意，概率是连续变化的！因此，通过使用离散事件的概率而不是事件本身，您可以巧妙地避开微积分无法真正处理离散事件的问题。

概率分布*p*简单地是所涉及的可能离散事件的概率列表。在这种情况下，*p* = (0.75, 0.25)。另外，您可以将<math alttext="p colon StartSet 0 comma 1 EndSet right-arrow double-struck upper R"><mrow><mi>p</mi> <mo>:</mo> <mo>{</mo> <mn>0</mn> <mo>,</mo> <mn>1</mn> <mo>}</mo> <mo>→</mo> <mi>ℝ</mi></mrow></math>视为从两个元素集合到实数的函数。这种观点在符号上有时会很有用。

我们简要指出，概率分布的技术定义更加复杂。将概率分布分配给实值事件是可行的。我们将在本章后面讨论这样的分布。

### 交叉熵损失

交叉熵是衡量两个概率分布之间距离的数学方法：

<math display="block"><mrow><mi>H</mi> <mrow><mo>(</mo> <mi>p</mi> <mo>,</mo> <mi>q</mi> <mo>)</mo></mrow> <mo>=</mo> <mo>-</mo> <munder><mo>∑</mo> <mi>x</mi></munder> <mi>p</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo form="prefix">log</mo> <mi>q</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>

这里*p*和*q*是两个概率分布。符号*p*(*x*)表示*p*赋予事件*x*的概率。这个定义值得仔细讨论。与*L*²范数一样，*H*提供了距离的概念。请注意，在*p* = *q*的情况下，

<math display="block"><mrow><mi>H</mi> <mrow><mo>(</mo> <mi>p</mi> <mo>,</mo> <mi>p</mi> <mo>)</mo></mrow> <mo>=</mo> <mo>-</mo> <munder><mo>∑</mo> <mi>x</mi></munder> <mi>p</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo form="prefix">log</mo> <mi>p</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>

这个数量是*p*的熵，通常简单地写作*H*(*p*)。这是分布无序程度的度量；当所有事件等可能时，熵最大。*H*(*p*)总是小于或等于*H*(*p*, *q*)。事实上，分布*q*距离*p*越远，交叉熵就越大。我们不会深入探讨这些陈述的确切含义，但将交叉熵视为距离机制的直觉值得记住。

另外，请注意，与*L*²范数不同，*H*是不对称的！也就是说，<math alttext="upper H left-parenthesis p comma q right-parenthesis not-equals upper H left-parenthesis q comma p right-parenthesis"><mrow><mi>H</mi> <mo>(</mo> <mi>p</mi> <mo>,</mo> <mi>q</mi> <mo>)</mo> <mo>≠</mo> <mi>H</mi> <mo>(</mo> <mi>q</mi> <mo>,</mo> <mi>p</mi> <mo>)</mo></mrow></math>。因此，使用交叉熵进行推理可能有点棘手，最好谨慎处理。

回到具体问题，现在假设<math alttext="p equals left-parenthesis y comma 1 minus y right-parenthesis"><mrow><mi>p</mi> <mo>=</mo> <mo>(</mo> <mi>y</mi> <mo>,</mo> <mn>1</mn> <mo>-</mo> <mi>y</mi> <mo>)</mo></mrow></math>是具有两个结果的离散系统的真实数据分布，<math alttext="q equals left-parenthesis y Subscript pred Baseline comma 1 minus y Subscript pred Baseline right-parenthesis"><mrow><mi>q</mi> <mo>=</mo> <mo>(</mo> <msub><mi>y</mi> <mtext>pred</mtext></msub> <mo>,</mo> <mn>1</mn> <mo>-</mo> <msub><mi>y</mi> <mtext>pred</mtext></msub> <mo>)</mo></mrow></math>是机器学习系统预测的。那么交叉熵损失是

<math display="block"><mrow><mi>H</mi> <mrow><mo>(</mo> <mi>p</mi> <mo>,</mo> <mi>q</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>y</mi> <mo form="prefix">log</mo> <msub><mi>y</mi> <mtext>pred</mtext></msub> <mo>+</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <mi>y</mi> <mo>)</mo></mrow> <mo form="prefix">log</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>y</mi> <mtext>pred</mtext></msub> <mo>)</mo></mrow></mrow></math>

这种损失形式在机器学习系统中被广泛使用来训练分类器。经验上，最小化*H*(*p*, *q*)似乎能够构建出很好地复制提供的训练标签的分类器。

## 梯度下降

到目前为止，在这一章中，您已经了解了将函数最小化作为机器学习的代理的概念。简而言之，最小化适当的函数通常足以学会解决所需的任务。为了使用这个框架，您需要使用适当的损失函数，比如*L*²或*H*(*p*, *q*) 交叉熵，以将分类和回归问题转化为适当的损失函数。

# 可学习权重

到目前为止，在本章中，我们已经解释了机器学习是通过最小化适当定义的损失函数<math alttext="script upper L left-parenthesis x comma y right-parenthesis"><mrow><mi>ℒ</mi> <mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow></math>来实现的。也就是说，我们试图找到最小化它的损失函数<math alttext="script upper L"><mi>ℒ</mi></math>的参数。然而，细心的读者会记得(*x*,*y*)是固定的量，不能改变。那么在学习过程中我们改变的是什么参数呢？

输入可学习权重*W*。假设*f*(*x*)是我们希望用机器学习模型拟合的可微函数。我们将规定*f*由选择*W*的方式进行*参数化*。也就是说，我们的函数实际上有两个参数*f*(*W*, *x*)。固定*W*的值会导致一个仅依赖于数据点*x*的函数。这些可学习权重实际上是通过最小化损失函数选择的量。我们将在本章后面看到如何使用`tf.Variable`来编码可学习权重。

但是，现在假设我们已经用适当的损失函数编码了我们的学习问题？在实践中，我们如何找到这个损失函数的最小值？我们将使用的关键技巧是梯度下降最小化。假设*f*是一个依赖于一些权重*W*的函数。那么<math alttext="normal nabla upper W"><mrow><mi>∇</mi> <mi>W</mi></mrow></math>表示的是在*W*中会最大程度增加*f*的方向变化。由此可知，朝着相反方向迈出一步会让我们更接近*f*的最小值。

# 梯度的符号

我们已经将可学习权重*W*的梯度写成了<math alttext="normal nabla upper W"><mrow><mi>∇</mi> <mi>W</mi></mrow></math>。有时，使用以下替代符号表示梯度会更方便：

<math display="block"><mrow><mi>∇</mi> <mi>W</mi> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>ℒ</mi></mrow> <mrow><mi>∂</mi><mi>W</mi></mrow></mfrac></mrow></math>

将这个方程理解为梯度<math alttext="normal nabla upper W"><mrow><mi>∇</mi> <mi>W</mi></mrow></math>编码了最大程度改变损失<math alttext="script upper L"><mi>ℒ</mi></math>的方向。

梯度下降的思想是通过反复遵循负梯度来找到函数的最小值。从算法上讲，这个更新规则可以表示为

<math display="block"><mrow><mi>W</mi> <mo>=</mo> <mi>W</mi> <mo>-</mo> <mi>α</mi> <mi>∇</mi> <mi>W</mi></mrow></math>

其中<math><mi>α</mi></math>是*步长*，决定了新梯度<math alttext="normal nabla upper W"><mrow><mi>∇</mi> <mi>W</mi></mrow></math>被赋予多少权重。这个想法是每次都朝着<math alttext="normal nabla upper W"><mrow><mi>∇</mi> <mi>W</mi></mrow></math>的方向迈出许多小步。注意<math alttext="normal nabla upper W"><mrow><mi>∇</mi> <mi>W</mi></mrow></math>本身是*W*的一个函数，所以实际步骤在每次迭代中都会改变。每一步都对权重矩阵*W*进行一点更新。执行更新的迭代过程通常称为*学习*权重矩阵*W*。

# 使用小批量高效计算梯度

一个问题是计算<math alttext="normal nabla upper W"><mrow><mi>∇</mi> <mi>W</mi></mrow></math>可能非常慢。隐含地，<math alttext="normal nabla upper W"><mrow><mi>∇</mi> <mi>W</mi></mrow></math>取决于损失函数<math alttext="script upper L"><mi>ℒ</mi></math>。由于<math alttext="script upper L"><mi>ℒ</mi></math>取决于整个数据集，对于大型数据集来说，计算<math alttext="normal nabla upper W"><mrow><mi>∇</mi> <mi>W</mi></mrow></math>可能会变得非常缓慢。在实践中，人们通常在称为*minibatch*的数据集的一部分上估计<math><mrow><mi>∇</mi> <mi>W</mi></mrow></math>。每个 minibatch 通常包含 50-100 个样本。Minibatch 的大小是深度学习算法中的一个*超参数*。每个步骤<math><mi>α</mi></math>的步长是另一个超参数。深度学习算法通常具有超参数的集群，这些超参数本身不是通过随机梯度下降学习的。

可学习参数和超参数之间的这种张力是深度结构的弱点和优势之一。超参数的存在为利用专家的强烈直觉提供了很大的空间，而可学习参数则允许数据自己说话。然而，这种灵活性本身很快变成了一个弱点，对于超参数行为的理解有点像黑魔法，阻碍了初学者广泛部署深度学习。我们将在本书的后面花费大量精力讨论超参数优化。

我们通过介绍*时代*的概念来结束本节。一个时代是梯度下降算法在数据*x*上的完整遍历。更具体地说，一个时代包括需要查看给定 minibatch 大小的所有数据所需的梯度下降步骤。例如，假设一个数据集有 1,000 个数据点，训练使用大小为 50 的 minibatch。那么一个时代将包括 20 个梯度下降更新。每个训练时代增加了模型获得的有用知识量。从数学上讲，这将对应于训练集上损失函数值的减少。

早期的时代将导致损失函数的急剧下降。这个过程通常被称为在该数据集上*学习先验*。虽然看起来模型正在快速学习，但实际上它只是在调整自己以适应与手头问题相关的参数空间的部分。后续时代将对应于损失函数的较小下降，但通常在这些后续时代中才会发生有意义的学习。几个时代通常对于一个非平凡的模型来说时间太短，模型通常从 10-1,000 个时代或直到收敛进行训练。虽然这看起来很大，但重要的是要注意，所需的时代数量通常不随手头数据集的大小而增加。因此，梯度下降与数据大小成线性关系，而不是二次关系！这是随机梯度下降方法相对于其他学习算法的最大优势之一。更复杂的学习算法可能只需要对数据集进行一次遍历，但可能使用的总计算量与数据点数量成二次关系。在大数据集的时代，二次运行时间是一个致命的弱点。

跟踪损失函数随着周期数的减少可以是理解学习过程的极其有用的视觉简写。这些图通常被称为损失曲线（见图 3-4）。随着时间的推移，一个经验丰富的从业者可以通过快速查看损失曲线来诊断学习中的常见失败。我们将在本书的过程中对各种深度学习模型的损失曲线给予重要关注。特别是在本章后面，我们将介绍 TensorBoard，这是 TensorFlow 提供的用于跟踪诸如损失函数之类的量的强大可视化套件。

这些规则可以通过链式法则结合起来：

###### 机器学习是定义适合数据集的损失函数，然后将其最小化的艺术。为了最小化损失函数，我们需要计算它们的梯度，并使用梯度下降算法迭代地减少损失。然而，我们仍然需要讨论梯度是如何实际计算的。直到最近，答案是“手动”。机器学习专家会拿出笔和纸，手动计算矩阵导数，以计算学习系统中所有梯度的解析公式。然后这些公式将被手动编码以实现学习算法。这个过程以前是臭名昭著的，不止一位机器学习专家在发表的论文和生产系统中意外梯度错误的故事被发现了多年。

## 图 3-4。一个模型的损失曲线示例。请注意，这个损失曲线来自使用真实梯度（即非小批量估计）训练的模型，因此比您在本书后面遇到的其他损失曲线更平滑。

这种情况已经发生了显著变化，随着自动微分引擎的广泛可用。像 TensorFlow 这样的系统能够自动计算几乎所有损失函数的梯度。这种自动微分是 TensorFlow 和类似系统的最大优势之一，因为机器学习从业者不再需要成为矩阵微积分的专家。然而，了解 TensorFlow 如何自动计算复杂函数的导数仍然很重要。对于那些在微积分入门课程中受苦的读者，你可能记得计算函数的导数是令人惊讶地机械化的。有一系列简单的规则可以应用于计算大多数函数的导数。例如：

数学显示="block"的<math><mrow><mfrac><mi>d</mi> <mrow><mi>d</mi><mi>x</mi></mrow></mfrac> <msup><mi>e</mi> <mi>x</mi></msup> <mo>=</mo> <msup><mi>e</mi> <mi>x</mi></msup></mrow></math>

数学显示="block"的<math><mrow><mfrac><mi>d</mi> <mrow><mi>d</mi><mi>x</mi></mrow></mfrac> <msup><mi>x</mi> <mi>n</mi></msup> <mo>=</mo> <mi>n</mi> <msup><mi>x</mi> <mrow><mi>n</mi><mo>-</mo><mn>1</mn></mrow></msup></mrow></math>

数学显示="block"的<math><mfrac><mi>d</mi> <mrow><mi>d</mi><mi>x</mi></mrow></mfrac> <mi>f</mi> <mrow><mo>(</mo> <mi>g</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <msup><mi>f</mi> <mo>'</mo></msup> <mrow><mo>(</mo> <mi>g</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <msup><mi>g</mi> <mo>'</mo></msup> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>

自动微分系统

其中 <math><msup><mi>f</mi> <mo>'</mo></msup></math> 用于表示 *f* 的导数，<math alttext="g prime"><msup><mi>g</mi> <mo>'</mo></msup></math> 用于表示 *g* 的导数。有了这些规则，很容易想象如何为一维微积分编写自动微分引擎。事实上，在基于 Lisp 的课程中，创建这样一个微分引擎通常是一年级的编程练习。（事实证明，正确解析函数比求导数更加困难。Lisp 使用其语法轻松解析公式，而在其他语言中，等到上编译器课程再做这个练习通常更容易）。

如何将这些规则扩展到更高维度的微积分？搞定数学更加棘手，因为需要考虑更多的数字。例如，给定 *X* = *AB*，其中 *X*、*A*、*B* 都是矩阵，公式变成了

<math display="block"><mrow><mi>∇</mi> <mi>A</mi> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><mi>A</mi></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <msup><mi>B</mi> <mi>T</mi></msup> <mo>=</mo> <mrow><mo>(</mo> <mi>∇</mi> <mi>X</mi> <mo>)</mo></mrow> <msup><mi>B</mi> <mi>T</mi></msup></mrow></math>

这样的公式可以组合起来提供一个矢量和张量微积分的符号微分系统。

# 使用 TensorFlow 进行学习

在本章的其余部分，我们将介绍您学习使用 TensorFlow 创建基本机器学习模型所需的概念。我们将从介绍玩具数据集的概念开始，然后解释如何使用常见的 Python 库创建有意义的玩具数据集。接下来，我们将讨论新的 TensorFlow 想法，如占位符、喂养字典、名称范围、优化器和梯度。下一节将向您展示如何使用这些概念训练简单的回归和分类模型。

## 创建玩具数据集

在本节中，我们将讨论如何创建简单但有意义的合成数据集，或称为玩具数据集，用于训练简单的监督分类和回归模型。

### 对 NumPy 的（极其）简要介绍

我们将大量使用 NumPy 来定义有用的玩具数据集。NumPy 是一个允许操作张量（在 NumPy 中称为 `ndarray`）的 Python 包。示例 3-1 展示了一些基础知识。

##### 示例 3-1。一些基本 NumPy 用法示例

```py
>>> import numpy as np
>>> np.zeros((2,2))
array([[ 0.,  0.],
       [ 0.,  0.]])
>>> np.eye(3)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
```

您可能会注意到 NumPy `ndarray` 操作看起来与 TensorFlow 张量操作非常相似。这种相似性是 TensorFlow 架构师特意设计的。许多关键的 TensorFlow 实用函数具有与 NumPy 中类似函数的参数和形式。出于这个目的，我们不会试图深入介绍 NumPy，并相信读者通过实验来掌握 NumPy 的用法。有许多在线资源提供了 NumPy 的教程介绍。

### 为什么玩具数据集很重要？

在机器学习中，学会正确使用玩具数据集通常至关重要。学习是具有挑战性的，初学者经常犯的一个最常见的错误是尝试在太早的时候在复杂数据上学习非平凡的模型。这些尝试往往以惨败告终，想要成为机器学习者的人会灰心丧气，认为机器学习不适合他们。

当然，真正的罪魁祸首不是学生，而是真实世界数据集具有许多特殊性。经验丰富的数据科学家已经了解到，真实世界数据集通常需要许多清理和预处理转换才能适合学习。深度学习加剧了这个问题，因为大多数深度学习模型对数据中的不完美非常敏感。诸如广泛范围的回归标签或潜在的强噪声模式等问题可能会使梯度下降方法出现问题，即使其他机器学习算法（如随机森林）也不会有问题。

幸运的是，几乎总是可以解决这些问题，但这可能需要数据科学家具有相当的复杂技能。这些敏感性问题可能是机器学习作为一种技术商品化的最大障碍。我们将深入探讨数据清理策略，但目前，我们建议一个更简单的替代方案：使用玩具数据集！

玩具数据集对于理解学习算法至关重要。给定非常简单的合成数据集，可以轻松判断算法是否学习了正确的规则。在更复杂的数据集上，这种判断可能非常具有挑战性。因此，在本章的其余部分，我们将只使用玩具数据集，同时涵盖基于 TensorFlow 的梯度下降学习的基础知识。在接下来的章节中，我们将深入研究具有真实数据的案例研究。

### 使用高斯分布添加噪声

早些时候，我们讨论了离散概率分布作为将离散选择转换为连续值的工具。我们也提到了连续概率分布的概念，但没有深入探讨。

连续概率分布（更准确地称为概率密度函数）是用于建模可能具有一系列结果的随机事件的有用数学工具。对于我们的目的，将概率密度函数视为用于模拟数据收集中的某些测量误差的有用工具就足够了。高斯分布被广泛用于噪声建模。

如图 3-5 所示，注意高斯分布可以具有不同的*均值*<math><mi>μ</mi></math>和*标准差*<math alttext="sigma"><mi>σ</mi></math>。高斯分布的均值是它取的平均值，而标准差是围绕这个平均值的扩散的度量。一般来说，将高斯随机变量添加到某个数量上提供了一种结构化的方式，通过使其稍微变化来模糊这个数量。这是一个非常有用的技巧，用于生成非平凡的合成数据集。

![gaussian.png](img/tfdl_0305.png)

###### 图 3-5。不同均值和标准差的各种高斯概率分布的插图。

我们迅速指出高斯分布也被称为正态分布。均值为<math><mi>μ</mi></math>，标准差为<math alttext="sigma"><mi>σ</mi></math>的高斯分布写为<math alttext="upper N left-parenthesis mu comma sigma right-parenthesis"><mrow><mi>N</mi> <mo>(</mo> <mi>μ</mi> <mo>,</mo> <mi>σ</mi> <mo>)</mo></mrow></math>。这种简写符号很方便，我们将在接下来的章节中多次使用它。

### 玩具回归数据集

最简单的线性回归形式是学习一维线的参数。假设我们的数据点*x*是一维的。然后假设实值标签*y*由线性规则生成

<math display="block"><mrow><mi>y</mi> <mo>=</mo> <mi>w</mi> <mi>x</mi> <mo>+</mo> <mi>b</mi></mrow></math>

在这里，*w*，*b*是必须通过梯度下降从数据中估计出来的可学习参数。为了测试我们是否可以使用 TensorFlow 学习这些参数，我们将生成一个由直线上的点组成的人工数据集。为了使学习挑战稍微困难一些，我们将在数据集中添加少量高斯噪声。

让我们写下我们的直线方程，受到少量高斯噪声的干扰：

<math display="block"><mrow><mi>y</mi> <mo>=</mo> <mi>w</mi> <mi>x</mi> <mo>+</mo> <mi>b</mi> <mo>+</mo> <mi>N</mi> <mo>(</mo> <mn>0</mn> <mo>,</mo> <mi>ϵ</mi> <mo>)</mo></mrow></math>

这里<math alttext="epsilon"><mi>ϵ</mi></math>是噪声项的标准差。然后我们可以使用 NumPy 从这个分布中生成一个人工数据集，如示例 3-2 所示。

##### 示例 3-2. 使用 NumPy 对人工数据集进行抽样

```py
# Generate synthetic data
N = 100
w_true = 5
b_true = 2
noise_scale = .1
x_np = np.random.rand(N, 1)
noise = np.random.normal(scale=noise_scale, size=(N, 1))
# Convert shape of y_np to (N,)
y_np = np.reshape(w_true * x_np + b_true + noise, (-1))
```

我们使用 Matplotlib 在图 3-6 中绘制这个数据集（您可以在与本书相关的[GitHub 存储库](https://github.com/matroid/dlwithtf)中找到确切的绘图代码）以验证合成数据看起来是否合理。如预期的那样，数据分布是一条直线，带有少量测量误差。

![lr_data.png](img/tfdl_0306.png)

###### 图 3-6. 玩具回归数据分布的绘图。

### 玩具分类数据集

创建合成分类数据集有点棘手。从逻辑上讲，我们希望有两个不同的、容易分离的点类。假设数据集只包含两种类型的点，（-1，-1）和（1，1）。然后学习算法将不得不学习一个将这两个数据值分开的规则。

+   *y*[0] = (-1, -1)

+   *y*[1] = (1, 1)

与以前一样，让我们通过向两种类型的点添加一些高斯噪声来增加一些挑战：

+   *y*[0] = (-1, -1) + *N*(0, ϵ)

+   *y*[1] = (1, 1) + *N*(0, ϵ)

然而，这里有一点小技巧。我们的点是二维的，而我们之前引入的高斯噪声是一维的。幸运的是，存在高斯的多变量扩展。我们不会在这里讨论多变量高斯的复杂性，但您不需要理解这些复杂性来跟随我们的讨论。

在示例 3-3 中生成合成数据集的 NumPy 代码比线性回归问题稍微棘手，因为我们必须使用堆叠函数`np.vstack`将两种不同类型的数据点组合在一起，并将它们与不同的标签关联起来。（我们使用相关函数`np.concatenate`将一维标签组合在一起。）

##### 示例 3-3. 使用 NumPy 对玩具分类数据集进行抽样

```py
# Generate synthetic data
N = 100
# Zeros form a Gaussian centered at (-1, -1)
# epsilon is .1
x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(N/2,))
y_zeros = np.zeros((N/2,))
# Ones form a Gaussian centered at (1, 1)
# epsilon is .1
x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)), cov=.1*np.eye(2), size=(N/2,))
y_ones = np.ones((N/2,))

x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])
```

图 3-7 使用 Matplotlib 绘制了这段代码生成的数据，以验证分布是否符合预期。我们看到数据分布在两个清晰分开的类中。

![logistic_data.png](img/tfdl_0307.png)

###### 图 3-7. 玩具分类数据分布的绘图。

## 新的 TensorFlow 概念

在 TensorFlow 中创建简单的机器学习系统将需要您学习一些新的 TensorFlow 概念。

### 占位符

占位符是将信息输入到 TensorFlow 计算图中的一种方式。将占位符视为信息进入 TensorFlow 的输入节点。用于创建占位符的关键函数是`tf.placeholder`（示例 3-4）。

##### 示例 3-4. 创建一个 TensorFlow 占位符

```py
>>> tf.placeholder(tf.float32, shape=(2,2))
<tf.Tensor 'Placeholder:0' shape=(2, 2) dtype=float32>
```

我们将使用占位符将数据点*x*和标签*y*馈送到我们的回归和分类算法中。

### 馈送字典和获取

回想一下，我们可以通过`sess.run(var)`在 TensorFlow 中评估张量。那么我们如何为占位符提供值呢？答案是构建*feed 字典*。Feed 字典是 Python 字典，将 TensorFlow 张量映射到包含这些占位符具体值的`np.ndarray`对象。Feed 字典最好被视为 TensorFlow 计算图的输入。那么输出是什么？TensorFlow 称这些输出为*fetches*。您已经见过 fetches 了。我们在上一章中广泛使用了它们，但没有这样称呼；fetch 是一个张量（或张量），其值是在计算图中的计算（使用 feed 字典中的占位符值）完成后检索的（示例 3-5）。

##### 示例 3-5。使用 fetches

```py
>>> a = tf.placeholder(tf.float32, shape=(1,))
>>> b = tf.placeholder(tf.float32, shape=(1,))
>>> c = a + b
>>> with tf.Session() as sess:
        c_eval = sess.run(c, {a: [1.], b: [2.]})
        print(c_eval)
[ 3.]
```

### 命名空间

在复杂的 TensorFlow 程序中，将在整个程序中定义许多张量、变量和占位符。`tf.name_scope(name)`为管理这些变量集合提供了一个简单的作用域机制（示例 3-6）。在`tf.name_scope(name)`调用的作用域内创建的所有计算图元素将在其名称前加上`name`。

这种组织工具在与 TensorBoard 结合使用时最有用，因为它有助于可视化系统自动将图元素分组到相同的命名空间中。您将在下一节中进一步了解 TensorBoard。

##### 示例 3-6。使用命名空间来组织占位符

```py
>>> N = 5
>>> with tf.name_scope("placeholders"):
      x = tf.placeholder(tf.float32, (N, 1))
      y = tf.placeholder(tf.float32, (N,))
>>> x
<tf.Tensor 'placeholders/Placeholder:0' shape=(5, 1) dtype=float32>
```

### 优化器

在前两节介绍的基本概念已经暗示了在 TensorFlow 中如何进行机器学习。您已经学会了如何为数据点和标签添加占位符，以及如何使用张量操作定义损失函数。缺失的部分是您仍然不知道如何使用 TensorFlow 执行梯度下降。

实际上，可以直接在 Python 中使用 TensorFlow 原语定义优化算法，TensorFlow 在`tf.train`模块中提供了一系列优化算法。这些算法可以作为节点添加到 TensorFlow 计算图中。

# 我应该使用哪个优化器？

在`tf.train`中有许多可能的优化器可用。简短预览中包括`tf.train.GradientDescentOptimizer`、`tf.train.MomentumOptimizer`、`tf.train.AdagradOptimizer`、`tf.train.AdamOptimizer`等。这些不同优化器之间有什么区别呢？

几乎所有这些优化器都是基于梯度下降的思想。回想一下我们之前介绍的简单梯度下降规则：

<math display="block"><mrow><mi>W</mi> <mo>=</mo> <mi>W</mi> <mo>-</mo> <mi>α</mi> <mi>∇</mi> <mi>W</mi></mrow></math>

从数学上讲，这个更新规则是原始的。研究人员发现了许多数学技巧，可以在不使用太多额外计算的情况下实现更快的优化。一般来说，`tf.train.AdamOptimizer`是一个相对稳健的好默认值。（许多优化方法对超参数的选择非常敏感。对于初学者来说，最好避开更复杂的方法，直到他们对不同优化算法的行为有很好的理解。）

示例 3-7 是一小段代码，它向计算图中添加了一个优化器，用于最小化预定义的损失`l`。

##### 示例 3-7。向 TensorFlow 计算图添加 Adam 优化器

```py
learning_rate = .001
with tf.name_scope("optim"):
  train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
```

### 使用 TensorFlow 计算梯度

我们之前提到，在 TensorFlow 中直接实现梯度下降算法是可能的。虽然大多数用例不需要重新实现`tf.train`的内容，但直接查看梯度值以进行调试可能很有用。`tf.gradients`提供了一个有用的工具来实现这一点（示例 3-8）。

##### 示例 3-8。直接计算梯度

```py
>>> W = tf.Variable((3,))
>>> l = tf.reduce_sum(W)
>>> gradW = tf.gradients(l, W)
>>> gradW
[<tf.Tensor 'gradients/Sum_grad/Tile:0' shape=(1,) dtype=int32>]
```

这段代码符号地拉下了损失`l`相对于可学习参数（`tf.Variable`）`W`的梯度。`tf.gradients`返回所需梯度的列表。请注意，梯度本身也是张量！TensorFlow 执行符号微分，这意味着梯度本身是计算图的一部分。TensorFlow 符号梯度的一个很好的副作用是，可以在 TensorFlow 中堆叠导数。这对于更高级的算法有时可能是有用的。

### TensorBoard 的摘要和文件写入器

对张量程序结构有一个视觉理解是非常有用的。TensorFlow 团队提供了 TensorBoard 包来实现这个目的。TensorBoard 启动一个 Web 服务器（默认情况下在 localhost 上），显示 TensorFlow 程序的各种有用的可视化。然而，为了能够使用 TensorBoard 检查 TensorFlow 程序，程序员必须手动编写日志记录语句。`tf.train.FileWriter()`指定了 TensorBoard 程序的日志目录，`tf.summary`将各种 TensorFlow 变量的摘要写入指定的日志目录。在本章中，我们只会使用`tf.summary.scalar`，它总结了一个标量量，以跟踪损失函数的值。`tf.summary.merge_all()`是一个有用的日志辅助工具，它将多个摘要合并为一个摘要以方便使用。

示例 3-9 中的代码片段为损失添加了一个摘要，并指定了一个日志目录。

##### 示例 3-9。为损失添加一个摘要

```py
with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/lr-train', tf.get_default_graph())
```

### 使用 TensorFlow 训练模型

假设现在我们已经为数据点和标签指定了占位符，并且已经用张量操作定义了一个损失。我们已经在计算图中添加了一个优化器节点`train_op`，我们可以使用它来执行梯度下降步骤（虽然我们实际上可能使用不同的优化器，但为了方便起见，我们将更新称为梯度下降）。我们如何迭代地执行梯度下降来在这个数据集上学习？

简单的答案是我们使用 Python 的`for`循环。在每次迭代中，我们使用`sess.run()`来获取图中的`train_op`以及合并的摘要操作`merged`和损失`l`。我们使用一个 feed 字典将所有数据点和标签输入`sess.run()`。

示例 3-10 中的代码片段演示了这种简单的学习方法。请注意，出于教学简单性的考虑，我们不使用小批量。在接下来的章节中，代码将在训练更大的数据集时使用小批量。

##### 示例 3-10。训练模型的简单示例

```py
n_steps = 1000
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # Train model
  for i in range(n_steps):
    feed_dict = {x: x_np, y: y_np}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    print("step %d, loss: %f" % (i, loss))
    train_writer.add_summary(summary, i)
```

# 在 TensorFlow 中训练线性和逻辑模型

本节将在前一节介绍的所有 TensorFlow 概念上进行总结，以在我们在本章中之前介绍的玩具数据集上训练线性和逻辑回归模型。

## 在 TensorFlow 中的线性回归

在本节中，我们将提供代码来定义一个在 TensorFlow 中学习其权重的线性回归模型。这个任务很简单，你可以很容易地在没有 TensorFlow 的情况下完成。然而，在 TensorFlow 中做这个练习是很好的，因为它将整合我们在本章中介绍的新概念。

### 在 TensorFlow 中定义和训练线性回归

线性回归模型很简单：

<math display="block"><mrow><mi>y</mi> <mo>=</mo> <mi>w</mi> <mi>x</mi> <mo>+</mo> <mi>b</mi></mrow></math>

这里*w*和*b*是我们希望学习的权重。我们将这些权重转换为`tf.Variable`对象。然后我们使用张量操作构建*L*²损失：

<math display="block"><mrow><mi>ℒ</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>=</mo> <msup><mrow><mo>(</mo><mi>y</mi><mo>-</mo><mi>w</mi><mi>x</mi><mo>-</mo><mi>b</mi><mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>

示例 3-11 中的代码在 TensorFlow 中实现了这些数学操作。它还使用`tf.name_scope`来分组各种操作，并添加了`tf.train.AdamOptimizer`用于学习和`tf.summary`操作用于 TensorBoard 的使用。

##### 示例 3-11. 定义线性回归模型

```py
# Generate tensorflow graph
with tf.name_scope("placeholders"):
  x = tf.placeholder(tf.float32, (N, 1))
  y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("weights"):
  # Note that x is a scalar, so W is a single learnable weight.
  W = tf.Variable(tf.random_normal((1, 1)))
  b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("prediction"):
  y_pred = tf.matmul(x, W) + b
with tf.name_scope("loss"):
  l = tf.reduce_sum((y - y_pred)**2)
# Add training op
with tf.name_scope("optim"):
  # Set learning rate to .001 as recommended above.
  train_op = tf.train.AdamOptimizer(.001).minimize(l)
with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/lr-train', tf.get_default_graph())
```

示例 3-12 然后训练这个模型，如之前讨论的（不使用小批量）。

##### 示例 3-12. 训练线性回归模型

```py
n_steps = 1000
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # Train model
  for i in range(n_steps):
    feed_dict = {x: x_np, y: y_np}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    print("step %d, loss: %f" % (i, loss))
    train_writer.add_summary(summary, i)
```

此示例的所有代码都在与本书相关的[GitHub 存储库](https://github.com/matroid/dlwithtf)中提供。我们鼓励所有读者运行线性回归示例的完整脚本，以获得对学习算法如何运行的第一手感觉。这个示例足够小，读者不需要访问任何专用计算硬件来运行。

# 线性回归的梯度

我们建模的线性系统的方程是*y* = *wx* + *b*，其中*w*，*b*是可学习的权重。正如我们之前提到的，这个系统的损失是<math alttext="script upper L equals left-parenthesis y minus w x minus b right-parenthesis squared"><mrow><mi>ℒ</mi> <mo>=</mo> <msup><mrow><mo>(</mo><mi>y</mi><mo>-</mo><mi>w</mi><mi>x</mi><mo>-</mo><mi>b</mi><mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>。一些矩阵微积分可以用来直接计算*w*的可学习参数的梯度：

<math display="block"><mrow><mi>∇</mi> <mi>w</mi> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>ℒ</mi></mrow> <mrow><mi>∂</mi><mi>w</mi></mrow></mfrac> <mo>=</mo> <mo>-</mo> <mn>2</mn> <mrow><mo>(</mo> <mi>y</mi> <mo>-</mo> <mi>w</mi> <mi>x</mi> <mo>-</mo> <mi>b</mi> <mo>)</mo></mrow> <msup><mi>x</mi> <mi>T</mi></msup></mrow></math>

对于*b*

<math display="block"><mrow><mi>∇</mi> <mi>b</mi> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>ℒ</mi></mrow> <mrow><mi>∂</mi><mi>b</mi></mrow></mfrac> <mo>=</mo> <mo>-</mo> <mn>2</mn> <mrow><mo>(</mo> <mi>y</mi> <mo>-</mo> <mi>w</mi> <mi>x</mi> <mo>-</mo> <mi>b</mi> <mo>)</mo></mrow></mrow></math>

我们将这些方程放在这里，仅供好奇的读者参考。我们不会试图系统地教授如何计算我们在本书中遇到的损失函数的导数。然而，我们将指出，对于复杂系统，通过手工计算损失函数的导数有助于建立对深度网络学习方式的直觉。这种直觉可以作为设计者的强大指导，因此我们鼓励高级读者自行探索这个主题。

### 使用 TensorBoard 可视化线性回归模型

前一节中定义的模型使用`tf.summary.FileWriter`将日志写入日志目录*/tmp/lr-train*。我们可以使用示例 3-13 中的命令在此日志目录上调用 TensorBoard（TensorBoard 默认与 TensorFlow 一起安装）。

##### 示例 3-13. 调用 TensorBoard

```py
tensorboard --logdir=/tmp/lr-train
```

此命令将在连接到 localhost 的端口上启动 TensorBoard。使用浏览器打开此端口。TensorBoard 屏幕将类似于图 3-8。 （具体外观可能会因您使用的 TensorBoard 版本而有所不同。）

![tensorboard_lr_raw.png](img/tfdl_0308.png)

###### 图 3-8. TensorBoard 面板截图。

转到 Graphs 选项卡，您将看到我们定义的 TensorFlow 架构的可视化，如图 3-9 所示。

![lr_graph.png](img/tfdl_0309.png)

###### 图 3-9. 在 TensorBoard 中可视化线性回归架构。

请注意，此可视化已将属于各种`tf.name_scopes`的所有计算图元素分组。不同的组根据计算图中的依赖关系连接。您可以展开所有分组的元素以查看其内容。图 3-10 展示了扩展的架构。

正如您所看到的，有许多隐藏的节点突然变得可见！TensorFlow 的函数，如`tf.train.AdamOptimizer`，通常会在它们自己的`tf.name_scope`下隐藏许多内部变量。在 TensorBoard 中展开提供了一种简单的方法，可以查看系统实际创建了什么。虽然可视化看起来相当复杂，但大多数细节都是在幕后，您暂时不需要担心。

![lr_expanded.png](img/tfdl_0310.png)

###### 图 3-10。架构的扩展可视化。

返回主页选项卡并打开摘要部分。现在您应该看到一个类似于图 3-11 的损失曲线。请注意平滑下降的形状。损失在开始时迅速下降，然后逐渐减少并稳定下来。

![lr_loss_tensorboard.png](img/tfdl_0311.png)

###### 图 3-11。在 TensorBoard 中查看损失曲线。

# 视觉和非视觉调试风格

像 TensorBoard 这样的工具是否必要才能充分利用像 TensorFlow 这样的系统？这取决于情况。使用 GUI 或交互式调试器是否是成为专业程序员的必要条件？

不同的程序员有不同的风格。有些人会发现 TensorBoard 的可视化能力成为张量编程工作流程中至关重要的一部分。其他人可能会发现 TensorBoard 并不是特别有用，会更多地使用打印语句进行调试。张量编程和调试的这两种风格都是有效的，就像有些优秀的程序员信誓旦旦地使用调试器，而有些则憎恶它们一样。

总的来说，TensorBoard 对于调试和建立对手头数据集的基本直觉非常有用。我们建议您遵循最适合您的风格。

### 用于评估回归模型的指标

到目前为止，我们还没有讨论如何评估训练模型是否真正学到了东西。评估模型是否训练的第一个工具是查看损失曲线，以确保其具有合理的形状。您在上一节中学习了如何做到这一点。接下来要尝试什么？

现在我们希望您查看与模型相关的*指标*。指标是用于比较预测标签和真实标签的工具。对于回归问题，有两个常见的指标：*R*²和 RMSE（均方根误差）。*R*²是两个变量之间相关性的度量，取值介于+1 和 0 之间。+1 表示完美相关，而 0 表示没有相关性。在数学上，两个数据集*X*和*Y*的*R*²定义如下：

<math display="block"><mrow><msup><mi>R</mi> <mn>2</mn></msup> <mo>=</mo> <mfrac><mrow><mtext>cov</mtext><msup><mrow><mo>(</mo><mi>X</mi><mo>,</mo><mi>Y</mi><mo>)</mo></mrow> <mn>2</mn></msup></mrow> <mrow><msubsup><mi>σ</mi> <mi>X</mi> <mn>2</mn></msubsup> <msubsup><mi>σ</mi> <mi>Y</mi> <mn>2</mn></msubsup></mrow></mfrac></mrow></math>

其中 cov(*X*, *Y*)是*X*和*Y*的协方差，衡量两个数据集共同变化的程度，而<math alttext="sigma Subscript upper X"><msub><mi>σ</mi> <mi>X</mi></msub></math>和<math alttext="sigma Subscript upper Y"><msub><mi>σ</mi> <mi>Y</mi></msub></math>是标准差，衡量每个集合的变化程度。直观地说，*R*²衡量了每个集合中独立变化的多少可以通过它们的共同变化来解释。

# 多种类型的 R²！

请注意，实践中有两种常见的*R*²定义。一个常见的初学者（和专家）错误是混淆这两个定义。在本书中，我们将始终使用平方的皮尔逊相关系数（图 3-12）。另一种定义称为确定系数。这种另一种*R*²通常更加令人困惑，因为它不像平方的皮尔逊相关系数那样有 0 的下限。

在图 3-12 中，预测值和真实值高度相关，*R*²接近 1。看起来学习在这个系统上做得很好，并成功学习到了真实规则。*不要那么快*。您会注意到图中两个轴的刻度不同！原来*R*²不会因为刻度的差异而受到惩罚。为了理解这个系统发生了什么，我们需要考虑图 3-13 中的另一个度量。

![lr_pred.png](img/tfdl_0312.png)

###### 图 3-12。绘制皮尔逊相关系数。

![lr_learned.png](img/tfdl_0313.png)

###### 图 3-13。绘制均方根误差（RMSE）。

RMSE 是预测值和真实值之间平均差异的度量。在图 3-13 中，我们将预测值和真实标签作为两个单独的函数绘制，使用数据点*x*作为我们的 x 轴。请注意，学习到的线并不是真实函数！RMSE 相对较高，诊断了错误，而*R*²没有发现这个错误。

这个系统发生了什么？为什么尽管经过训练收敛，TensorFlow 仍然没有学习到正确的函数？这个例子很好地说明了梯度下降算法的一个弱点。不能保证找到真正的解决方案！梯度下降算法可能会陷入*局部最小值*。也就是说，它可能找到看起来不错的解决方案，但实际上并不是损失函数<math alttext="script upper L"><mi>ℒ</mi></math>的最低最小值。

那么为什么要使用梯度下降呢？对于简单的系统，确实往往最好避免梯度下降，而使用其他性能更好的算法。然而，在复杂的系统中，比如我们将在后面的章节中展示的系统，还没有比梯度下降表现更好的替代算法。我们鼓励您记住这一点，因为我们将继续深入学习。

## TensorFlow 中的逻辑回归

在本节中，我们将使用 TensorFlow 定义一个简单的分类器。首先考虑分类器的方程是什么。通常使用的数学技巧是利用 S 形函数。S 形函数，在图 3-14 中绘制，通常用<math alttext="sigma"><mi>σ</mi></math>表示，是从实数<math alttext="double-struck upper R"><mi>ℝ</mi></math>到(0, 1)的函数。这个特性很方便，因为我们可以将 S 形函数的输出解释为事件发生的概率。（将离散事件转换为连续值的技巧是机器学习中的一个常见主题。）

![logistic.gif](img/tfdl_0314.png)

###### 图 3-14。绘制 S 形函数。

用于预测离散 0/1 变量概率的方程如下。这些方程定义了一个简单的逻辑回归模型：

<math display="block"><mrow><msub><mi>y</mi> <mn>0</mn></msub> <mo>=</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>w</mi> <mi>x</mi> <mo>+</mo> <mi>b</mi> <mo>)</mo></mrow></mrow></math><math display="block"><mrow><msub><mi>y</mi> <mn>1</mn></msub> <mo>=</mo> <mn>1</mn> <mo>-</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>w</mi> <mi>x</mi> <mo>+</mo> <mi>b</mi> <mo>)</mo></mrow></mrow></math>

TensorFlow 提供了用于计算 S 形值的交叉熵损失的实用函数。其中最简单的函数是`tf.nn.sigmoid_cross_​entropy_with_logits`。（对数几率是 S 形函数的反函数。实际上，这意味着直接将参数传递给 TensorFlow，而不是 S 形值<math><mrow><mi>σ</mi> <mo>(</mo> <mi>w</mi> <mi>x</mi> <mo>+</mo> <mi>b</mi> <mo>)</mo></mrow></math>本身）。我们建议使用 TensorFlow 的实现，而不是手动定义交叉熵，因为在计算交叉熵损失时会出现棘手的数值问题。

示例 3-14 在 TensorFlow 中定义了一个简单的逻辑回归模型。

##### 示例 3-14。定义一个简单的逻辑回归模型

```py
# Generate tensorflow graph
with tf.name_scope("placeholders"):
  # Note that our datapoints x are 2-dimensional.
  x = tf.placeholder(tf.float32, (N, 2))
  y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("weights"):
  W = tf.Variable(tf.random_normal((2, 1)))
  b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("prediction"):
  y_logit = tf.squeeze(tf.matmul(x, W) + b)
  # the sigmoid gives the class probability of 1
  y_one_prob = tf.sigmoid(y_logit)
  # Rounding P(y=1) will give the correct prediction.
  y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
  # Compute the cross-entropy term for each datapoint
  entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
  # Sum all contributions
  l = tf.reduce_sum(entropy)
with tf.name_scope("optim"):
  train_op = tf.train.AdamOptimizer(.01).minimize(l)

  train_writer = tf.summary.FileWriter('/tmp/logistic-train', tf.get_default_graph())
```

示例 3-15 中的此模型的训练代码与线性回归模型的代码相同。

##### 示例 3-15。训练逻辑回归模型

```py
n_steps = 1000
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # Train model
  for i in range(n_steps):
    feed_dict = {x: x_np, y: y_np}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    print("loss: %f" % loss)
    train_writer.add_summary(summary, i)
```

### 使用 TensorBoard 可视化逻辑回归模型

与之前一样，您可以使用 TensorBoard 来可视化模型。首先，像图 3-15 中所示，可视化损失函数。请注意，与以前一样，损失函数遵循一种整齐的模式。损失函数有一个陡峭的下降，然后是逐渐平滑。

![logistic_loss_tensorboard.png](img/tfdl_0315.png)

###### 图 3-15。可视化逻辑回归损失函数。

您还可以在 TensorBoard 中查看 TensorFlow 图。由于作用域结构类似于线性回归所使用的结构，简化的图显示方式并没有太大不同，如图 3-16 所示。

![logistic_graph.png](img/tfdl_0316.png)

###### 图 3-16。可视化逻辑回归的计算图。

然而，如果您扩展这个分组图中的节点，就像图 3-17 中所示，您会发现底层的计算图是不同的。特别是，损失函数与线性回归所使用的损失函数有很大不同（这是应该的）。

![logistic_expanded.png](img/tfdl_0317.png)

###### 图 3-17。逻辑回归的扩展计算图。

### 评估分类模型的指标

现在您已经为逻辑回归训练了一个分类模型，需要了解适用于评估分类模型的指标。虽然逻辑回归的方程比线性回归的方程复杂，但基本的评估指标更简单。分类准确率只是检查学习模型正确分类的数据点的比例。实际上，稍加努力，就可以推导出逻辑回归模型学习的*分隔线*。这条线显示了模型学习到的分隔正负示例的边界。（我们将推导这条线从逻辑回归方程中的练习留给感兴趣的读者。解决方案在本节的代码中。）

我们在图 3-18 中显示了学习到的类别和分隔线。请注意，这条线清晰地分隔了正负示例，并且具有完美的准确率（1.0）。这个结果提出了一个有趣的观点。回归通常比分类更难解决。在图 3-18 中，有许多可能的线可以很好地分隔数据点，但只有一条线可以完美地匹配线性回归的数据。

![logistic_pred.png](img/tfdl_0318.png)

###### 图 3-18。查看逻辑回归的学习类别和分隔线。

# 回顾

在本章中，我们向您展示了如何在 TensorFlow 中构建和训练一些简单的学习系统。我们首先回顾了一些基础数学概念，包括损失函数和梯度下降。然后，我们向您介绍了一些新的 TensorFlow 概念，如占位符、作用域和 TensorBoard。我们以在玩具数据集上训练线性和逻辑回归系统的案例研究结束了本章。本章涵盖了很多内容，如果您还没有完全掌握，也没关系。本章介绍的基础知识将贯穿本书的其余部分。

在第四章中，我们将向您介绍您的第一个深度学习模型和全连接网络，并向您展示如何在 TensorFlow 中定义和训练全连接网络。在接下来的章节中，我们将探索更复杂的深度网络，但所有这些架构都将使用本章介绍的相同基本学习原则。
