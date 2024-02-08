# 第四章。训练前馈神经网络

# 快餐问题

我们开始理解如何使用深度学习解决一些有趣的问题，但仍然有一个重要问题：我们如何确切地找出参数向量（神经网络中所有连接的权重）应该是什么？这是通过一个通常称为*训练*的过程来实现的（见图 4-1）。在训练过程中，我们向神经网络展示大量的训练示例，并迭代修改权重，以最小化我们在训练示例上的错误。经过足够多的示例后，我们期望我们的神经网络将非常有效地解决其被训练做的任务。

![ ](img/fdl2_0401.png)

###### 图 4-1。这是我们想要为快餐问题训练的神经元

让我们继续使用第三章中的一个例子，涉及线性神经元：每天，我们购买由汉堡包、薯条和苏打组成的餐厅餐。我们为每种物品购买一些份量。我们想要能够预测一顿饭会花费我们多少钱，但物品没有价格标签。收银员告诉我们的唯一事情是餐费的总价。我们想要训练一个单一的线性神经元来解决这个问题。我们该如何做？

一个想法是在选择我们的训练案例时要聪明。对于一顿饭，我们可以只购买一份汉堡包，对于另一顿饭，我们可以只购买一份薯条，然后对于我们最后一顿饭，我们可以只购买一份苏打。一般来说，聪明地选择训练示例是一个好主意。许多研究表明，通过设计一个聪明的训练集，您可以使您的神经网络更加有效。仅使用这种方法的问题在于，在实际情况下，它很少能让您接近解决方案。例如，在图像识别中，这种策略没有明显的类似物。这不是一个实际的解决方案。

相反，我们试图激励一个在一般情况下表现良好的解决方案。假设我们有一大批训练示例。然后我们可以使用图中的简单公式计算神经网络在第 i 个训练示例上的输出。我们希望训练神经元，以便我们选择可能最优的权重——最小化我们在训练示例上的错误。在这种情况下，假设我们想要在我们遇到的所有训练示例上最小化平方误差。更正式地说，如果我们知道<math alttext="t Superscript left-parenthesis i right-parenthesis"><msup><mi>t</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></math>是第 i 个训练示例的真实答案，而<math alttext="y Superscript left-parenthesis i right-parenthesis"><msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></math>是神经网络计算出的值，我们希望最小化误差函数*E*的值：

<math alttext="upper E equals one-half sigma-summation Underscript i Endscripts left-parenthesis t Superscript left-parenthesis i right-parenthesis Baseline minus y Superscript left-parenthesis i right-parenthesis Baseline right-parenthesis squared"><mrow><mi>E</mi> <mo>=</mo> <mfrac><mn>1</mn> <mn>2</mn></mfrac> <msub><mo>∑</mo> <mi>i</mi></msub> <msup><mrow><mo>(</mo><msup><mi>t</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>-</mo><msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>

当我们的模型在每个训练示例上做出完全正确的预测时，平方误差为零。此外，*E*越接近 0，我们的模型就越好。因此，我们的目标是选择我们的参数向量<math alttext="theta"><mi>θ</mi></math>（模型中所有权重的值），使*E*尽可能接近 0。

现在在这一点上，您可能会想知道为什么我们需要关注误差函数，当我们可以将这个问题视为一个方程组时。毕竟，我们有一堆未知数（权重），我们有一组方程（每个训练示例一个）。假设我们有一致的训练示例集，那将自动给我们一个误差为 0。

这是一个聪明的观察，但不幸的是这个洞察并不很好地推广。请记住，虽然我们在这里使用的是线性神经元，但在实践中很少使用线性神经元，因为它们在学习方面受到限制。一旦我们开始使用像我们在第三章末尾讨论的 S 型、双曲正切或 ReLU 神经元这样的非线性神经元，我们就不能再建立一个线性方程组。显然，我们需要更好的策略来解决训练过程。

# 梯度下降

让我们通过简化问题来可视化如何最小化所有训练样本上的平方误差。假设我们的线性神经元只有两个输入（因此只有两个权重，<math alttext="w 1"><msub><mi>w</mi> <mn>1</mn></msub></math> 和 <math alttext="w 2"><msub><mi>w</mi> <mn>2</mn></msub></math>）。然后我们可以想象一个三维空间，其中水平维度对应于权重 <math alttext="w 1"><msub><mi>w</mi> <mn>1</mn></msub></math> 和 <math alttext="w 2"><msub><mi>w</mi> <mn>2</mn></msub></math>，垂直维度对应于误差函数 *E* 的值。在这个空间中，水平平面上的点对应于权重的不同设置，这些点的高度对应于产生的误差。如果我们考虑我们在所有可能的权重上所犯的错误，我们会得到这个三维空间中的一个表面，特别是一个二次碗，如图 4-2 所示。

![ ](img/fdl2_0402.png)

###### 图 4-2。线性神经元的二次误差表面

我们还可以方便地将这个表面可视化为一组椭圆形轮廓，其中最小误差位于椭圆的中心。在这个设置中，我们在一个二维平面上工作，其中维度对应于两个权重。轮廓对应于评估为相同值 *E* 的 <math alttext="w 1"><msub><mi>w</mi> <mn>1</mn></msub></math> 和 <math alttext="w 2"><msub><mi>w</mi> <mn>2</mn></msub></math> 的设置。轮廓越接近，斜率越陡。事实上，最陡下降的方向总是垂直于轮廓。这个方向被表示为一个称为*梯度*的向量。

现在我们可以制定一个高层策略，来找到最小化误差函数的权重值。假设我们随机初始化网络的权重，这样我们就会发现自己在水平平面上的某个位置。通过评估当前位置的梯度，我们可以找到最陡下降的方向，并朝这个方向迈出一步。然后我们会发现自己在一个比之前更接近最小值的新位置。我们可以通过在这个新位置上取梯度并朝这个新方向迈出一步来重新评估最陡下降的方向。很容易看出，如图 4-3 所示，遵循这个策略最终会将我们带到最小误差点。这个算法被称为*梯度下降*，我们将用它来解决训练单个神经元和更一般的训练整个网络的问题。^(1)

![ ](img/fdl2_0403.png)

###### 图 4-3。将误差表面可视化为一组轮廓

# Δ规则和学习率

在我们推导出训练快餐神经元的确切算法之前，我们先简要提一下*超参数*。除了神经网络中定义的权重参数外，学习算法还需要一些额外的参数来执行训练过程。其中一个被称为*学习率*的超参数。

在实践中，在沿着等高线垂直移动的每一步中，我们需要确定在重新计算新方向之前要走多远。这个距离需要取决于表面的陡峭程度。为什么？我们离最小值越近，我们就希望向前迈出的步子越短。我们知道我们离最小值很近，因为表面更加平坦，所以我们可以使用陡峭度作为我们离最小值有多近的指标。然而，如果我们的误差表面相当平缓，训练可能需要大量时间。因此，我们经常将梯度乘以一个因子<math alttext="epsilon"><mi>ϵ</mi></math>，即学习率。选择学习率是一个困难的问题（图 4-4）。正如我们刚刚讨论的，如果我们选择的学习率太小，我们在训练过程中可能需要花费太长时间。但是，如果我们选择的学习率太大，我们很可能会开始偏离最小值。在第五章中，我们将学习各种利用自适应学习率的优化技术，以自动化选择学习率的过程。

![ ](img/fdl2_0404.png)

###### 图 4-4。当我们的学习率过大时，收敛变得困难

现在，我们终于准备好推导训练线性神经元的*delta 规则*。为了计算如何改变每个权重，我们评估梯度，这实质上是关于每个权重的误差函数的偏导数。换句话说，我们希望：

在这一节和下一节中，我们将处理训练神经元和利用非线性的神经网络。我们使用 S 形神经元作为模型，并将其他非线性神经元的推导留给您作为练习。为简单起见，我们假设神经元不使用偏置项，尽管我们的分析很容易扩展到这种情况。我们只需要假设偏置是一个权重，其输入值始终为一。

让我们回顾一下逻辑神经元如何从其输入计算输出值的机制：

# 具有 S 形神经元的梯度下降

应用这种在每次迭代中改变权重的方法，我们最终能够利用梯度下降。

z = Σw_k x_k

y = 1 / (1 + e^(-z))

神经元计算其输入的加权和，logit*z*。然后将其 logit 输入到输入函数中计算*y*，其最终输出。幸运的是，这些函数有很好的导数，这使得学习变得容易！对于学习，我们希望计算误差函数对权重的梯度。为此，我们首先计算 logit 对输入和权重的导数：

<math alttext="StartFraction normal partial-differential z Over normal partial-differential w Subscript k Baseline EndFraction equals x Subscript k"><mrow><mfrac><mrow><mi>∂</mi><mi>z</mi></mrow> <mrow><mi>∂</mi><msub><mi>w</mi> <mi>k</mi></msub></mrow></mfrac> <mo>=</mo> <msub><mi>x</mi> <mi>k</mi></msub></mrow></math>

<math alttext="StartFraction normal partial-differential z Over normal partial-differential x Subscript k Baseline EndFraction equals w Subscript k"><mrow><mfrac><mrow><mi>∂</mi><mi>z</mi></mrow> <mrow><mi>∂</mi><msub><mi>x</mi> <mi>k</mi></msub></mrow></mfrac> <mo>=</mo> <msub><mi>w</mi> <mi>k</mi></msub></mrow></math>

而且，令人惊讶的是，如果你用输出来表达输出对 logit 的导数，那么输出对 logit 的导数就非常简单：

<math alttext="StartFraction d y Over d z EndFraction equals StartFraction e Superscript negative z Baseline Over left-parenthesis 1 plus e Superscript negative z Baseline right-parenthesis squared EndFraction"><mrow><mfrac><mrow><mi>d</mi><mi>y</mi></mrow> <mrow><mi>d</mi><mi>z</mi></mrow></mfrac> <mo>=</mo> <mfrac><msup><mi>e</mi> <mrow><mo>-</mo><mi>z</mi></mrow></msup> <msup><mfenced separators="" open="(" close=")"><mn>1</mn><mo>+</mo><msup><mi>e</mi> <mrow><mo>-</mo><mi>z</mi></mrow></msup></mfenced> <mn>2</mn></msup></mfrac></mrow></math>

<math alttext="equals StartFraction 1 Over 1 plus e Superscript negative z Baseline EndFraction StartFraction e Superscript negative z Baseline Over 1 plus e Superscript negative z Baseline EndFraction"><mrow><mo>=</mo> <mfrac><mn>1</mn> <mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi> <mrow><mo>-</mo><mi>z</mi></mrow></msup></mrow></mfrac> <mfrac><msup><mi>e</mi> <mrow><mo>-</mo><mi>z</mi></mrow></msup> <mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi> <mrow><mo>-</mo><mi>z</mi></mrow></msup></mrow></mfrac></mrow></math>

<math alttext="equals StartFraction 1 Over 1 plus e Superscript negative z Baseline EndFraction left-parenthesis 1 minus StartFraction 1 Over 1 plus e Superscript negative z Baseline EndFraction right-parenthesis"><mrow><mo>=</mo> <mfrac><mn>1</mn> <mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi> <mrow><mo>-</mo><mi>z</mi></mrow></msup></mrow></mfrac> <mfenced separators="" open="(" close=")"><mn>1</mn> <mo>-</mo> <mfrac><mn>1</mn> <mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi> <mrow><mo>-</mo><mi>z</mi></mrow></msup></mrow></mfrac></mfenced></mrow></math>

<math alttext="equals y left-parenthesis 1 minus y right-parenthesis"><mrow><mo>=</mo> <mi>y</mi> <mo>(</mo> <mn>1</mn> <mo>-</mo> <mi>y</mi> <mo>)</mo></mrow></math>

然后我们使用链式法则得到输出对每个权重的导数：

<math alttext="StartFraction normal partial-differential y Over normal partial-differential w Subscript k Baseline EndFraction equals StartFraction d y Over d z EndFraction StartFraction normal partial-differential z Over normal partial-differential w Subscript k Baseline EndFraction equals x Subscript k Baseline y left-parenthesis 1 minus y right-parenthesis"><mrow><mfrac><mrow><mi>∂</mi><mi>y</mi></mrow> <mrow><mi>∂</mi><msub><mi>w</mi> <mi>k</mi></msub></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>d</mi><mi>y</mi></mrow> <mrow><mi>d</mi><mi>z</mi></mrow></mfrac> <mfrac><mrow><mi>∂</mi><mi>z</mi></mrow> <mrow><mi>∂</mi><msub><mi>w</mi> <mi>k</mi></msub></mrow></mfrac> <mo>=</mo> <msub><mi>x</mi> <mi>k</mi></msub> <mi>y</mi> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <mi>y</mi> <mo>)</mo></mrow></mrow></math>

将所有这些结合在一起，我们现在可以计算误差函数对每个权重的导数：

<math alttext="StartFraction normal partial-differential upper E Over normal partial-differential w Subscript k Baseline EndFraction equals sigma-summation Underscript i Endscripts StartFraction normal partial-differential upper E Over normal partial-differential y Superscript left-parenthesis i right-parenthesis Baseline EndFraction StartFraction normal partial-differential y Superscript left-parenthesis i right-parenthesis Baseline Over normal partial-differential w Subscript k Baseline EndFraction equals minus sigma-summation Underscript i Endscripts x Subscript k Superscript left-parenthesis i right-parenthesis Baseline y Superscript left-parenthesis i right-parenthesis Baseline left-parenthesis 1 minus y Superscript left-parenthesis i right-parenthesis Baseline right-parenthesis left-parenthesis t Superscript left-parenthesis i right-parenthesis Baseline minus y Superscript left-parenthesis i right-parenthesis Baseline right-parenthesis"><mrow><mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>w</mi> <mi>k</mi></msub></mrow></mfrac> <mo>=</mo> <msub><mo>∑</mo> <mi>i</mi></msub> <mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mrow></mfrac> <mfrac><mrow><mi>∂</mi><msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msub><mi>w</mi> <mi>k</mi></msub></mrow></mfrac> <mo>=</mo> <mo>-</mo> <msub><mo>∑</mo> <mi>i</mi></msub> <msubsup><mi>x</mi> <mi>k</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup> <msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mfenced separators="" open="(" close=")"><mn>1</mn> <mo>-</mo> <msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mfenced> <mfenced separators="" open="(" close=")"><msup><mi>t</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>-</mo> <msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mfenced></mrow></math>

因此，修改权重的最终规则变为：

<math alttext="normal upper Delta w Subscript k Baseline equals sigma-summation Underscript i Endscripts epsilon x Subscript k Superscript left-parenthesis i right-parenthesis Baseline y Superscript left-parenthesis i right-parenthesis Baseline left-parenthesis 1 minus y Superscript left-parenthesis i right-parenthesis Baseline right-parenthesis left-parenthesis t Superscript left-parenthesis i right-parenthesis Baseline minus y Superscript left-parenthesis i right-parenthesis Baseline right-parenthesis"><mrow><mi>Δ</mi> <msub><mi>w</mi> <mi>k</mi></msub> <mo>=</mo> <msub><mo>∑</mo> <mi>i</mi></msub> <mi>ϵ</mi> <msubsup><mi>x</mi> <mi>k</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup> <msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mfenced separators="" open="(" close=")"><mn>1</mn> <mo>-</mo> <msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mfenced> <mfenced separators="" open="(" close=")"><msup><mi>t</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>-</mo> <msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mfenced></mrow></math>

正如您可能注意到的，新的修改规则就像 delta 规则一样，只是包含了额外的乘法项，以考虑 S 型神经元的逻辑组件。

# 反向传播算法

现在我们终于准备好解决训练多层神经网络（而不仅仅是单个神经元）的问题了。为了完成这个任务，我们将使用一种称为*反向传播*的方法，由 David E. Rumelhart、Geoffrey E. Hinton 和 Ronald J. Williams 在 1986 年首创。那么反向传播的背后是什么想法呢？我们不知道隐藏单元应该做什么，但我们可以计算当我们改变隐藏层的活动时误差变化的速度。从那里，我们可以找出当我们改变单个连接的权重时误差变化的速度。基本上，我们将尝试找到最陡降的路径。唯一的问题是，我们将在一个极高维度的空间中工作。我们首先计算与单个训练示例相关的误差导数。

每个隐藏单元可以影响许多输出单元。因此，我们将不得不以一种信息丰富的方式将许多单独的对误差的影响组合在一起。我们的策略将是动态规划的一种。一旦我们有了一个隐藏单元层的误差导数，我们将使用它们来计算下一层活动的误差导数。一旦我们找到了隐藏单元活动的误差导数，就很容易得到导致隐藏单元的权重的误差导数。我们将重新定义一些符号以便讨论，并参考图 4-5。

![ ](img/fdl2_0405.png)

###### 图 4-5。反向传播算法推导的参考图

我们使用的下标将指代神经元的层。符号*y*将像往常一样指代神经元的活动。类似地，符号*z*将指代神经元的 logit。我们首先看一下动态规划问题的基本情况。具体来说，我们计算输出层的误差函数导数：

<math alttext="upper E equals one-half sigma-summation Underscript j element-of o u t p u t Endscripts left-parenthesis t Subscript j Baseline minus y Subscript j Baseline right-parenthesis squared long right double arrow StartFraction normal partial-differential upper E Over normal partial-differential y Subscript j Baseline EndFraction equals minus left-parenthesis t Subscript j Baseline minus y Subscript j Baseline right-parenthesis"><mrow><mi>E</mi> <mo>=</mo> <mfrac><mn>1</mn> <mn>2</mn></mfrac> <msub><mo>∑</mo> <mrow><mi>j</mi><mo>∈</mo><mi>o</mi><mi>u</mi><mi>t</mi><mi>p</mi><mi>u</mi><mi>t</mi></mrow></msub> <msup><mfenced separators="" open="(" close=")"><msub><mi>t</mi> <mi>j</mi></msub> <mo>-</mo><msub><mi>y</mi> <mi>j</mi></msub></mfenced> <mn>2</mn></msup> <mo>⇒</mo> <mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>y</mi> <mi>j</mi></msub></mrow></mfrac> <mo>=</mo> <mo>-</mo> <mrow><mo>(</mo> <msub><mi>t</mi> <mi>j</mi></msub> <mo>-</mo> <msub><mi>y</mi> <mi>j</mi></msub> <mo>)</mo></mrow></mrow></math>

现在我们来解决归纳步骤。假设我们已经有了层<math alttext="j"><mi>j</mi></math>的误差导数。我们接下来的目标是计算其下一层，即层<math alttext="i"><mi>i</mi></math>的误差导数。为了做到这一点，我们必须积累关于层<math alttext="i"><mi>i</mi></math>中神经元的输出如何影响层<math alttext="j"><mi>j</mi></math>中每个神经元的 logit 的信息。这可以通过以下方式完成，利用这样一个事实：logit 对来自下一层的输入数据的偏导数仅仅是连接<math alttext="w Subscript i j"><msub><mi>w</mi> <mrow><mi>i</mi><mi>j</mi></mrow></msub></math>的权重：

<math alttext="StartFraction normal partial-differential upper E Over normal partial-differential y Subscript i Baseline EndFraction equals sigma-summation Underscript j Endscripts StartFraction normal partial-differential upper E Over normal partial-differential z Subscript j Baseline EndFraction StartFraction d z Subscript j Baseline Over d y Subscript i Baseline EndFraction equals sigma-summation Underscript j Endscripts w Subscript i j Baseline StartFraction normal partial-differential upper E Over normal partial-differential z Subscript j Baseline EndFraction"><mrow><mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>y</mi> <mi>i</mi></msub></mrow></mfrac> <mo>=</mo> <msub><mo>∑</mo> <mi>j</mi></msub> <mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>z</mi> <mi>j</mi></msub></mrow></mfrac> <mfrac><mrow><mi>d</mi><msub><mi>z</mi> <mi>j</mi></msub></mrow> <mrow><mi>d</mi><msub><mi>y</mi> <mi>i</mi></msub></mrow></mfrac> <mo>=</mo> <msub><mo>∑</mo> <mi>j</mi></msub> <msub><mi>w</mi> <mrow><mi>i</mi><mi>j</mi></mrow></msub> <mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>z</mi> <mi>j</mi></msub></mrow></mfrac></mrow></math>

此外，我们观察到以下情况：

<math alttext="StartFraction normal partial-differential upper E Over normal partial-differential z Subscript j Baseline EndFraction equals StartFraction normal partial-differential upper E Over normal partial-differential y Subscript j Baseline EndFraction StartFraction d y Subscript j Baseline Over d z Subscript j Baseline EndFraction equals y Subscript j Baseline left-parenthesis 1 minus y Subscript j Baseline right-parenthesis StartFraction normal partial-differential upper E Over normal partial-differential y Subscript j Baseline EndFraction"><mrow><mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>z</mi> <mi>j</mi></msub></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>y</mi> <mi>j</mi></msub></mrow></mfrac> <mfrac><mrow><mi>d</mi><msub><mi>y</mi> <mi>j</mi></msub></mrow> <mrow><mi>d</mi><msub><mi>z</mi> <mi>j</mi></msub></mrow></mfrac> <mo>=</mo> <msub><mi>y</mi> <mi>j</mi></msub> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>y</mi> <mi>j</mi></msub> <mo>)</mo></mrow> <mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>y</mi> <mi>j</mi></msub></mrow></mfrac></mrow></math>

将这两者结合起来，我们最终可以用层<math alttext="i"><mi>i</mi></math>的误差导数来表示层<math alttext="j"><mi>j</mi></math>的误差导数：

<math alttext="StartFraction normal partial-differential upper E Over normal partial-differential y Subscript i Baseline EndFraction equals sigma-summation Underscript j Endscripts w Subscript i j Baseline y Subscript j Baseline left-parenthesis 1 minus y Subscript j Baseline right-parenthesis StartFraction normal partial-differential upper E Over normal partial-differential y Subscript j Baseline EndFraction"><mrow><mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>y</mi> <mi>i</mi></msub></mrow></mfrac> <mo>=</mo> <msub><mo>∑</mo> <mi>j</mi></msub> <msub><mi>w</mi> <mrow><mi>i</mi><mi>j</mi></mrow></msub> <msub><mi>y</mi> <mi>j</mi></msub> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>y</mi> <mi>j</mi></msub> <mo>)</mo></mrow> <mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>y</mi> <mi>j</mi></msub></mrow></mfrac></mrow></math>

一旦我们完成了整个动态规划例程，适当地填满了表格中所有关于隐藏单元活动的偏导数（关于误差函数的），我们就可以确定误差如何随权重变化。这给了我们一种在每个训练示例之后修改权重的方法：

<math alttext="StartFraction normal partial-differential upper E Over normal partial-differential w Subscript i j Baseline EndFraction equals StartFraction normal partial-differential z Subscript j Baseline Over normal partial-differential w Subscript i j Baseline EndFraction StartFraction normal partial-differential upper E Over normal partial-differential z Subscript j Baseline EndFraction equals y Subscript i Baseline y Subscript j Baseline left-parenthesis 1 minus y Subscript j Baseline right-parenthesis StartFraction normal partial-differential upper E Over normal partial-differential y Subscript j Baseline EndFraction"><mrow><mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>w</mi> <mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>∂</mi><msub><mi>z</mi> <mi>j</mi></msub></mrow> <mrow><mi>∂</mi><msub><mi>w</mi> <mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow></mfrac> <mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>z</mi> <mi>j</mi></msub></mrow></mfrac> <mo>=</mo> <msub><mi>y</mi> <mi>i</mi></msub> <msub><mi>y</mi> <mi>j</mi></msub> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>y</mi> <mi>j</mi></msub> <mo>)</mo></mrow> <mfrac><mrow><mi>∂</mi><mi>E</mi></mrow> <mrow><mi>∂</mi><msub><mi>y</mi> <mi>j</mi></msub></mrow></mfrac></mrow></math>

最后，为了完成算法，就像以前一样，我们只需对数据集中的所有训练示例的偏导数求和。这给出了以下修改公式：

<math alttext="normal upper Delta w Subscript i j Baseline equals minus sigma-summation Underscript k element-of d a t a s e t Endscripts epsilon y Subscript i Superscript left-parenthesis k right-parenthesis Baseline y Subscript j Superscript left-parenthesis k right-parenthesis Baseline left-parenthesis 1 minus y Subscript j Superscript left-parenthesis k right-parenthesis Baseline right-parenthesis StartFraction normal partial-differential upper E Superscript left-parenthesis k right-parenthesis Baseline Over normal partial-differential y Subscript j Superscript left-parenthesis k right-parenthesis Baseline EndFraction"><mrow><mi>Δ</mi> <msub><mi>w</mi> <mrow><mi>i</mi><mi>j</mi></mrow></msub> <mo>=</mo> <mo>-</mo> <msub><mo>∑</mo> <mrow><mi>k</mi><mo>∈</mo><mi>d</mi><mi>a</mi><mi>t</mi><mi>a</mi><mi>s</mi><mi>e</mi><mi>t</mi></mrow></msub> <mi>ϵ</mi> <msubsup><mi>y</mi> <mi>i</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msubsup> <msubsup><mi>y</mi> <mi>j</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msubsup> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msubsup><mi>y</mi> <mi>j</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msubsup> <mo>)</mo></mrow> <mfrac><mrow><mi>∂</mi><msup><mi>E</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msubsup><mi>y</mi> <mi>j</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msubsup></mrow></mfrac></mrow></math>

这完成了我们对反向传播算法的描述。

# 随机梯度下降和小批量梯度下降

在我们描述的算法中“反向传播算法”，我们使用了一种称为*批量* *梯度下降*的梯度下降版本。批量梯度下降的思想是使用整个数据集来计算误差表面，然后沿着梯度的方向采取最陡下降的路径。对于简单的二次误差表面，这样做效果很好。但在大多数情况下，我们的误差表面可能更加复杂。让我们考虑图 4-6 中的情况。

![](img/fdl2_0406.png)

###### 图 4-6。批量梯度下降对鞍点敏感，可能导致过早收敛

我们只有一个权重，并且我们使用随机初始化和批量梯度下降来找到其最佳设置。然而，误差表面有一个平坦区域（在高维空间中也称为鞍点），如果我们运气不好，可能会在执行梯度下降时陷入困境。

另一种潜在的方法是*随机梯度下降*（SGD），在每次迭代中，我们的误差表面只针对单个示例进行估计。这种方法由图 4-7 说明，我们的误差表面是动态的，而不是单一静态的误差表面。因此，在这种随机表面上下降显著提高了我们在平坦区域中导航的能力。

![](img/fdl2_0407.png)

###### 图 4-7。随机误差表面相对于批量误差表面波动，避免鞍点

然而，SGD 的主要缺点是，一次查看一个示例产生的误差可能不足以对误差表面进行良好的近似。这反过来可能导致梯度下降花费大量时间。解决这个问题的一种方法是使用*小批量梯度下降*。在小批量梯度下降中，每次迭代我们计算与总数据集的某个子集相关的误差表面（而不仅仅是单个示例）。这个子集称为*小批量*，除了学习率之外，小批量大小是另一个超参数。小批量在批量梯度下降的效率和随机梯度下降的局部最小值避免之间取得平衡。在反向传播的背景下，我们的权重更新步骤变为：

<math alttext="normal upper Delta w Subscript i j Baseline equals minus sigma-summation Underscript k element-of m i n i b a t c h Endscripts epsilon y Subscript i Superscript left-parenthesis k right-parenthesis Baseline y Subscript j Superscript left-parenthesis k right-parenthesis Baseline left-parenthesis 1 minus y Subscript j Superscript left-parenthesis k right-parenthesis Baseline right-parenthesis StartFraction normal partial-differential upper E Superscript left-parenthesis k right-parenthesis Baseline Over normal partial-differential y Subscript j Superscript left-parenthesis k right-parenthesis Baseline EndFraction"><mrow><mi>Δ</mi> <msub><mi>w</mi> <mrow><mi>i</mi><mi>j</mi></mrow></msub> <mo>=</mo> <mo>-</mo> <msub><mo>∑</mo> <mrow><mi>k</mi><mo>∈</mo><mi>m</mi><mi>i</mi><mi>n</mi><mi>i</mi><mi>b</mi><mi>a</mi><mi>t</mi><mi>c</mi><mi>h</mi></mrow></msub> <mi>ϵ</mi> <msubsup><mi>y</mi> <mi>i</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msubsup> <msubsup><mi>y</mi> <mi>j</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msubsup> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msubsup><mi>y</mi> <mi>j</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msubsup> <mo>)</mo></mrow> <mfrac><mrow><mi>∂</mi><msup><mi>E</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msubsup><mi>y</mi> <mi>j</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msubsup></mrow></mfrac></mrow></math>

这与我们在上一节中推导的内容完全相同，但是不是对数据集中的所有示例求和，而是对当前小批量中的示例求和。有关为什么随机梯度下降和小批量梯度下降会导致对整个数据集的梯度估计无偏的更多理论讨论，请参阅“神经网络学习理论”。

# 测试集、验证集和过拟合

人工神经网络的一个主要问题是模型非常复杂。例如，让我们考虑一个神经网络，从 MNIST 数据库中提取图像（28×28 像素），输入到两个具有 30 个神经元的隐藏层，最终到达一个具有 10 个神经元的 softmax 层。网络中的总参数数量接近 25,000。这可能会带来问题，为了理解原因，让我们考虑一个新的玩具示例，由图 4-8 说明。

![ ](img/fdl2_0408.png)

###### 图 4-8. 可能描述我们数据集的两个潜在模型：线性模型与 12 次多项式

我们在一个平面上给出了一堆数据点，我们的目标是找到一个最好描述这个数据集的曲线（即，将允许我们根据其*x*坐标预测一个新点的*y*坐标）。使用这些数据，我们训练了两种不同的模型：一个线性模型和一个 12 次多项式。我们应该信任哪条曲线？几乎没有一个训练样本正确的线？还是每个数据集中的每个点都命中的复杂曲线？此时，我们可能会相信线性拟合，因为它看起来不那么牵强。但为了确保，让我们向我们的数据集添加更多数据。结果显示在图 4-9 中。

现在判决很明确：线性模型不仅在主观上更好，而且在数量上也更好（使用平方误差度量）。这带来了一个有趣的关于训练和评估机器学习模型的观点。通过构建一个非常复杂的模型，很容易完美地拟合我们的训练数据集，因为我们给予我们的模型足够的自由度来扭曲自己以适应训练集中的观察结果。但是当我们在新数据上评估这样一个复杂模型时，表现很差。换句话说，模型不能很好地*泛化*。这是一种称为*过拟合*的现象，也是机器学习工程师必须应对的最大挑战之一。在深度学习中，这个问题变得更加严重，因为我们的神经网络有大量包含许多神经元的层。这些模型中的连接数是天文数字，达到了百万级。因此，过拟合是司空见惯的。

![ ](img/fdl2_0409.png)

###### 图 4-9. 在新数据上评估我们的模型表明，线性拟合比 12 次多项式更好

让我们看看这在神经网络的背景下是什么样子。假设我们有一个具有两个输入、大小为 2 的 softmax 输出和一个包含 3、6 或 20 个神经元的隐藏层的神经网络。我们使用小批量梯度下降（批量大小为 10）来训练这些网络，结果使用[ConvNetJS](http://stanford.io/2pOdNhy)可视化，显示在图 4-10 中。

![ ](img/fdl2_0410.png)

###### 图 4-10. 一个包含 3、6 和 20 个神经元（按顺序）的隐藏层的神经网络的可视化

从这些图像中已经很明显，随着网络中连接数的增加，我们过拟合数据的倾向也增加。当我们使神经网络变得更深时，我们也可以看到过拟合的现象。这些结果显示在图 4-11 中。

![ ](img/fdl2_0411.png)

###### 图 4-11. 一个、两个和四个包含三个神经元的隐藏层的神经网络（按顺序）

这导致了三个主要观察结果。首先，机器学习工程师始终在过拟合和模型复杂性之间进行直接权衡。如果模型不够复杂，可能不足以捕获解决问题所需的所有有用信息。然而，如果我们的模型非常复杂（特别是如果我们手头的数据量有限），我们就有过拟合的风险。深度学习采取解决复杂问题的方法，使用复杂模型并采取额外的对策来防止过拟合。我们将在本章和后续章节中看到许多这些措施。

其次，使用用于训练模型的数据来评估模型是具有误导性的。使用图 4-8 中的例子，这将错误地表明 12 次多项式模型优于线性拟合。因此，我们几乎从不在整个数据集上训练模型。相反，我们将数据分成*训练集*和*测试集*（图 4-12）。这使我们能够通过直接测量模型在尚未见过的新数据上的泛化效果来公平评估我们的模型。

###### 警告

在现实世界中，大型数据集很难获得，因此在训练过程中不使用所有可用数据可能会显得浪费。因此，可能会诱人地重复使用训练数据进行测试，或者在编制测试数据时采取捷径。请注意：如果测试集构建不当，我们将无法对模型得出任何有意义的结论。

![ ](img/fdl2_0412.png)

###### 图 4-12。非重叠的训练和测试集

第三，很可能在训练数据的过程中，有一个时间点，我们开始过度拟合训练集，而不是学习有用的特征。为了避免这种情况，我们希望能够在开始过拟合时停止训练过程，以防止泛化能力差。为了做到这一点，我们将训练过程分成*epochs*。一个 epoch 是对整个训练集的单次迭代。如果我们有一个大小为<math alttext="d"><mi>d</mi></math>的训练集，并且使用批量梯度下降的批量大小为<math alttext="b"><mi>b</mi></math>，那么一个 epoch 相当于<math alttext="StartFraction d Over b EndFraction"><mfrac><mi>d</mi> <mi>b</mi></mfrac></math>个模型更新。在每个 epoch 结束时，我们希望衡量我们的模型泛化的效果。为了做到这一点，我们使用一个额外的*验证集*，如图 4-13 所示。

![ ](img/fdl2_0413.png)

###### 图 4-13。用于在训练过程中防止过拟合的验证集

在一个 epoch 结束时，验证集将告诉我们模型在尚未见过的数据上的表现。如果训练集上的准确性继续增加，而验证集上的准确性保持不变（或下降），这表明是时候停止训练了，因为我们正在过拟合。

验证集还有助于作为*超参数优化*过程中准确性的代理度量。到目前为止，我们已经涵盖了几个超参数（学习率、小批量大小等），但我们尚未开发出如何找到这些超参数的最佳值的框架。找到超参数的最佳设置的一个潜在方法是应用*网格搜索*，在网格搜索中，我们从有限的选项集中为每个超参数选择一个值（例如，<math alttext="epsilon element-of StartSet 0.001 comma 0.01 comma 0.1 EndSet comma batch size element-of StartSet 16 comma 64 comma 128 EndSet comma ellipsis"><mrow><mi>ϵ</mi> <mo>∈</mo> <mo>{</mo> <mn>0</mn> <mo>.</mo> <mn>001</mn> <mo>,</mo> <mn>0</mn> <mo>.</mo> <mn>01</mn> <mo>,</mo> <mn>0</mn> <mo>.</mo> <mn>1</mn> <mo>}</mo> <mo>,</mo> <mtext>batch</mtext> <mtext>size</mtext> <mo>∈</mo> <mo>{</mo> <mn>16</mn> <mo>,</mo> <mn>64</mn> <mo>,</mo> <mn>128</mn> <mo>}</mo> <mo>,</mo> <mo>...</mo></mrow></math>），并使用每种超参数选择的所有可能排列来训练模型。我们选择在验证集上性能最佳的超参数组合，并报告使用最佳组合训练的模型在测试集上的准确性。^(3)

考虑到这一点，在我们开始描述直接对抗过拟合的各种方法之前，让我们先概述构建和训练深度学习模型时使用的工作流程。工作流程在图 4-14 中有详细描述。这有点复杂，但了解流程对确保我们正确训练神经网络至关重要。

![ ](img/fdl2_0414.png)

###### 图 4-14。深度学习模型训练和评估的详细工作流程

首先，我们严格定义我们的问题。这涉及确定我们的输入、潜在输出以及两者的向量化表示。例如，假设我们的目标是训练一个深度学习模型来识别癌症。我们的输入将是一个 RBG 图像，可以表示为像素值的向量。我们的输出将是三种互斥可能性的概率分布：（1）正常、（2）良性肿瘤（尚未转移的癌症）或（3）恶性肿瘤（已经转移到其他器官的癌症）。

在定义问题之后，我们需要构建一个神经网络架构来解决问题。我们的输入层必须具有适当的大小，以接受来自图像的原始数据，输出层必须是大小为 3 的 softmax。我们还必须定义网络的内部架构（隐藏层的数量，连接性等）。当我们讨论卷积神经网络时，我们将进一步讨论图像识别模型的架构。此时，我们还希望收集大量用于训练或建模的数据。这些数据可能是由医学专家标记的统一大小的病理图像。我们将这些数据洗牌并分成单独的训练、验证和测试集。

最后，我们准备开始梯度下降。我们每次在训练集上训练模型一个时代。在每个时代结束时，我们确保我们在训练集和验证集上的错误在减少。当其中一个停止改善时，我们终止并确保我们对模型在测试数据上的表现感到满意。如果我们感到不满意，我们需要重新考虑我们的架构或重新考虑我们收集的数据是否具有进行我们感兴趣的预测所需的信息。如果我们的训练集错误停止改善，我们可能需要更好地捕捉数据中的重要特征。如果我们的验证集错误停止改善，我们可能需要采取措施防止过拟合。

如果我们对模型在训练数据上的表现满意，那么我们可以在测试数据上衡量其性能，这是模型在此之前从未见过的数据。如果结果不尽人意，我们需要在数据集中添加更多数据，因为测试集似乎包含了在训练集中未充分代表的示例类型。否则，我们就完成了！

# 在深度神经网络中预防过拟合

在训练过程中已经提出了几种技术来防止过拟合。在本节中，我们将详细讨论这些技术。

一种对抗过拟合的方法称为*正则化*。正则化通过添加额外的惩罚大权重的项来修改我们最小化的目标函数。我们改变目标函数，使其变为<math alttext="upper E r r o r plus lamda f left-parenthesis theta right-parenthesis"><mrow><mi>E</mi> <mi>r</mi> <mi>r</mi> <mi>o</mi> <mi>r</mi> <mo>+</mo> <mi>λ</mi> <mi>f</mi> <mo>(</mo> <mi>θ</mi> <mo>)</mo></mrow></math>，其中<math alttext="f left-parenthesis theta right-parenthesis"><mrow><mi>f</mi> <mo>(</mo> <mi>θ</mi> <mo>)</mo></mrow></math>随着<math alttext="theta"><mi>θ</mi></math>的分量增大而增大，<math alttext="lamda"><mi>λ</mi></math>是正则化强度（另一个超参数）。我们选择的<math alttext="lamda"><mi>λ</mi></math>的值决定了我们想要保护免受过拟合的程度。<math alttext="lamda equals 0"><mrow><mi>λ</mi> <mo>=</mo> <mn>0</mn></mrow></math>意味着我们不采取任何措施来防止过拟合。如果<math alttext="lamda"><mi>λ</mi></math>太大，那么我们的模型将优先保持<math alttext="theta"><mi>θ</mi></math>尽可能小，而不是试图找到在训练集上表现良好的参数值。因此，选择<math alttext="lamda"><mi>λ</mi></math>是一项非常重要的任务，可能需要一些试错。

机器学习中最常见的正则化类型是*L2*正则化。^(4) 它可以通过将神经网络中所有权重的平方幅度添加到误差函数中来实现。换句话说，对于神经网络中的每个权重<math alttext="w"><mi>w</mi></math>，我们将<math alttext="one-half lamda w squared"><mrow><mfrac><mn>1</mn> <mn>2</mn></mfrac> <mi>λ</mi> <msup><mi>w</mi> <mn>2</mn></msup></mrow></math>添加到误差函数中。L2 正则化具有对尖峰权重向量进行严厉惩罚并更喜欢扩散权重向量的直观解释。这具有鼓励网络稍微使用其所有输入而不是仅大量使用部分输入的吸引性属性。特别值得注意的是，在梯度下降更新期间，使用 L2 正则化最终意味着每个权重都会线性衰减至零。由于这种现象，L2 正则化通常也被称为*权重衰减*。

我们可以使用 ConvNetJS 来可视化 L2 正则化的效果。与图 2-10 和 2-11 类似，我们使用一个具有 2 个输入、大小为 2 的 softmax 输出和一个包含 20 个神经元的隐藏层的神经网络。我们使用小批量梯度下降（批量大小为 10）和正则化强度为 0.01、0.1 和 1 来训练网络。结果可以在图 4-15 中看到。

![ ](img/fdl2_0415.png)

###### 图 4-15。使用正则化强度为 0.01、0.1 和 1（按顺序）训练的神经网络的可视化。

另一种常见的正则化类型是 *L1 正则化*。在这里，我们为神经网络中的每个权重 <math alttext="w"><mi>w</mi></math> 添加项 <math alttext="lamda StartAbsoluteValue w EndAbsoluteValue"><mrow><mi>λ</mi> <mfenced open="|" close="|"><mi>w</mi></mfenced></mrow></math>。L1 正则化具有引人注目的特性，即在优化过程中导致权重向量变得稀疏（即接近于零）。使用 L1 正则化的神经元最终只使用它们最重要的输入的一个小子集，并且对输入中的噪声变得相当抗拒。相比之下，来自 L2 正则化的权重向量通常是分散的、小数。当您想要准确了解哪些特征对决策有贡献时，L1 正则化是有用的。如果不需要这种特征分析水平，我们更倾向于使用 L2 正则化，因为它在经验上表现更好。

*最大范数约束* 有类似的目标，试图限制 <math alttext="theta"><mi>θ</mi></math> 变得过大，但它们更直接地实现这一目标。最大范数约束对每个神经元的传入权重向量的幅度施加绝对上限，并使用投影梯度下降来强制执行约束。因此，每当梯度下降步骤使传入权重向量移动，使得 <math alttext="StartAbsoluteValue EndAbsoluteValue w StartAbsoluteValue EndAbsoluteValue Subscript 2 Baseline greater-than c"><mrow><msub><mfenced open="|" close="|"><mfenced open="|" close="|"><mi>w</mi></mfenced></mfenced> <mn>2</mn></msub> <mo>></mo> <mi>c</mi></mrow></math> 时，我们将向量投影回以半径 <math alttext="c"><mi>c</mi></math>（以原点为中心）的球体上。典型的 <math alttext="c"><mi>c</mi></math> 值为 3 和 4。其中一个好处是参数向量不会失控增长（即使学习率过高），因为权重的更新始终受到限制。

*Dropout* 是一种不同的方法，用于防止过拟合，在深度神经网络中已经成为最受欢迎的防止过拟合的方法之一。在训练过程中，通过以概率 <math alttext="p"><mi>p</mi></math>（一个超参数）保持神经元活跃，或者将其设置为零来实现 dropout。直观地说，这迫使网络即使在缺少某些信息的情况下也能准确。它防止网络过于依赖任何一个神经元（或任何少量神经元的组合）。更数学化地表达，dropout 通过提供一种有效地近似组合指数级别不同的神经网络架构的方法来防止过拟合。dropout 的过程在 图 4-16 中表达。

辍学是相当直观的，但有一些重要的细微差别需要考虑。首先，我们希望在测试时间神经元的输出等同于训练时间的预期输出。我们可以通过在测试时间缩放输出来天真地修复这个问题。例如，如果 <math alttext="p equals 0.5"><mrow><mi>p</mi> <mo>=</mo> <mn>0</mn> <mo>.</mo> <mn>5</mn></mrow></math> ，神经元必须在测试时间将其输出减半，以便具有与训练期间相同（预期的）输出。这很容易理解，因为神经元的输出以概率 <math alttext="1 minus p"><mrow><mn>1</mn> <mo>-</mo> <mi>p</mi></mrow></math> 被设置为 0。这意味着如果一个神经元在辍学之前的输出是 *x*，那么在辍学之后，预期输出将是 <math alttext="upper E left-bracket output right-bracket equals p x plus left-parenthesis 1 minus p right-parenthesis dot 0 equals p x"><mrow><mi>E</mi> <mo>[</mo> <mtext>output</mtext> <mo>]</mo> <mo>=</mo> <mi>p</mi> <mi>x</mi> <mo>+</mo> <mo>(</mo> <mn>1</mn> <mo>-</mo> <mi>p</mi> <mo>)</mo> <mo>·</mo> <mn>0</mn> <mo>=</mo> <mi>p</mi> <mi>x</mi></mrow></math> 。然而，这种天真的辍学实现是不可取的，因为它要求在测试时间缩放神经元的输出。测试时间的性能对模型评估非常关键，因此最好始终使用*反向辍学*，其中缩放发生在训练时间而不是在测试时间。在反向辍学中，任何激活未被消除的神经元在传播到下一层之前，其输出被除以 <math alttext="p"><mi>p</mi></math> 。通过这种修复，<math alttext="upper E left-bracket output right-bracket equals p dot StartFraction x Over p EndFraction plus left-parenthesis 1 minus p right-parenthesis dot 0 equals x"><mrow><mi>E</mi> <mrow><mo>[</mo> <mtext>output</mtext> <mo>]</mo></mrow> <mo>=</mo> <mi>p</mi> <mo>·</mo> <mfrac><mi>x</mi> <mi>p</mi></mfrac> <mo>+</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <mi>p</mi> <mo>)</mo></mrow> <mo>·</mo> <mn>0</mn> <mo>=</mo> <mi>x</mi></mrow></math> ，我们可以避免在测试时间任意缩放神经元的输出。

![ ](img/fdl2_0416.png)

###### 图 4-16。在每个训练的小批量期间，辍学将网络中的每个神经元设置为不活跃状态

# 摘要

在本章中，我们学习了训练前馈神经网络所涉及的所有基础知识。我们讨论了梯度下降，反向传播算法，以及我们可以使用的各种方法来防止过拟合。在下一章中，当我们使用 PyTorch 库高效实现我们的第一个神经网络时，我们将把这些教训付诸实践。然后在第六章中，我们将回到优化目标函数以训练神经网络的问题，并设计算法来显著提高性能。这些改进将使我们能够处理更多的数据，这意味着我们将能够构建更全面的模型。

^(1) Rosenbloom, P. “The Method of Steepest Descent.” *Proceedings of Symposia in Applied Mathematics*. Vol. 6\. 1956.

^(2) Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. “Learning Representations by Back-Propagating Errors.” *Cognitive Modeling* 5.3 (1988): 1.

^(3) Nelder, John A., and Roger Mead. “A Simplex Method for Function Minimization.” *The Computer Journal* 7.4 (1965): 308-313.

^(4) Tikhonov, Andrei Nikolaevich, and Vladlen Borisovich Glasko. “Use of the Regularization Method in Non-Linear Problems.” *USSR Computational Mathematics and Mathematical Physics* 5.3 (1965): 93-107.

^(5) Srebro, Nathan, Jason DM Rennie, and Tommi S. Jaakkola. “Maximum-Margin Matrix Factorization.” *NIPS*, Vol. 17, 2004.

^(6) Srivastava, Nitish 等人。“Dropout: 一种简单的方法防止神经网络过拟合。” *机器学习研究杂志* 15.1 (2014): 1929-1958.
