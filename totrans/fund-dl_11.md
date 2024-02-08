# 第十一章：可解释性方法

# 概述

可解释性领域广泛，可以独特地应用于各种任务。简而言之，可解释性定义了模型向第三方“解释”其决策的能力。许多现代架构并没有通过构建具有这种能力。例如，神经网络就是这类现代架构的典型例子。术语“不透明”经常用来描述神经网络，无论是在媒体还是文献中。这是因为，没有事后技术来解释神经网络的最终分类或回归结果，训练模型内部发生的数据转换对最终用户来说是不清楚且难以解释的。我们只知道我们输入了一个示例，然后输出了一个结果。虽然我们可以检查神经网络学习到的权重，但所有这些权重的组合是一个极其复杂的函数。这使得很难确定输入的哪个部分最终对最终结果产生了最大的贡献。

已经设计了各种事后方法来解释神经网络的输出——*显著性映射*就是一个典型例子。显著性映射测量了经过训练模型的输出相对于输入的梯度。根据梯度的定义，具有最大幅度梯度的输入位置在其值稍微改变时会对输出值或分类情况产生最大影响。因此，显著性映射将解释具有最大幅度梯度的位置（及其相应的值）作为对最终结果贡献最大的输入部分。

然而，这并不是可解释性的全部和终结。显著性映射的一个问题是，它可能有点嘈杂，特别是当我们考虑像图像分类这样的任务时在单个像素级别的梯度。此外，如果输入的性质是分类的而不是连续的，例如句子的独热编码，那么相对于输入的梯度本身是不可解释的，因为输入空间是不连续的。

此外，正如前面提到的，手头的任务通常是决定什么是有效的可解释性方法的关键。我们将在接下来的部分中更详细地阐述这一点。

通常情况下，可解释性是以性能为代价的。将可解释性构建到模型中通常会通过做出简化模型假设（偏差-方差权衡中的偏差）而增加一些偏差，例如在普通线性回归中，我们假设特征和目标变量之间存在线性关系。然而，这些简化假设正是使得普通线性回归中输入特征和目标变量之间的关系比复杂的神经架构更清晰的原因。

这一切都引出了一个问题：为什么我们首先关心可解释性？在一个越来越被技术、复杂算法和机器学习主导的世界中，解释决策的能力是至关重要的。特别是在医学等领域，患者的生命受到威胁，或者在金融领域，人们的财务生计岌岌可危时，解释模型的决策是向广泛采用迈出的关键一步。在接下来的部分中，我们将介绍一些经典模型，这些模型在设计中具有强烈的可解释性概念。

# 决策树和基于树的算法

大多数经典数据科学和机器学习方法都具有某种形式的可解释性。基于树的算法是其中的一个明显例子。决策树旨在根据一系列条件语句对输入进行分类，树中的每个节点都与一个条件语句相关联。要了解经过训练的基于树的模型是如何做出决策的，我们只需对于任何给定的输入，在树中的每个节点处按照正确的分支进行跟随（图 11-1）。

![](img/fdl2_1101.png)

###### 图 11-1. 训练用于分类鸟类物种的决策树。给定一组鸟类特征，在每个节点处按照右侧的“Yes”或“No”分支到达最终分类。

更复杂的基于树的算法，如由大型决策树组成的随机森林算法，也是可解释的。例如，在分类的情况下，随机森林算法通过将给定的输入通过每个决策树运行，然后将决策树中的大多数输出类作为最终输出（或在回归的情况下取平均值）来运行。通过算法的构造，我们知道随机森林是如何得出关于输入的最终结论的。

除了在个体示例级别上的可解释性外，决策树及其更复杂的集成在全局级别具有内置的特征重要性度量。例如，在训练决策树时，必须确定要分割的特征以及要分割的该特征的阈值。在分类制度下，进行此操作的一种方法是通过在建议的特征上以建议的阈值进行分割来计算信息增益。为了构建我们的思维，让我们将可能的训练标签视为离散概率分布的定义域，其中每个标签的概率是该标签在训练数据集中出现的频率（图 11-2）。

![](img/fdl2_1102.png)

###### 图 11-2. 标签概率。

回想一下第二章，总结概率分布中不确定性的度量是分布的熵。当给定一个建议的特征和相关的阈值来进行分割时，我们可以根据我们将为每个输入示例遵循的分支将训练数据人口分成至少两个独立的组。每个子组现在都有自己对可能标签的分布，我们通过计算训练数据集的熵与每个子组的熵的加权和之间的差异来计算信息增益，其中权重与每个子组中的元素数量成比例。在每个分支点处具有最高信息增益的特征和相关的阈值是最佳分割。

为什么这有效？虽然我们不会在这里进行严格的证明，但考虑一个问题，我们有一个带有二进制标签的分子数据集，例如，指示每个化合物是否有毒，我们想建立一个分类器来预测化合物的毒性。还假设与每个化合物相关的特征之一是一个二进制特征，即分子是否含有酚官能团。酚官能团既具有相当的毒性，又是化合物中毒性的常见原因，因此在这个特征上进行分割会导致两个明显分开的子组。

包含酚官能团的正子组可能由于酚的毒性水平而几乎没有假阳性。不包含酚官能团的负子组可能由于酚是毒性的常见原因而几乎没有假阴性。因此，每个子组的相关熵非常低，因为每个子组中化合物的真实标签分布在一个标签上非常集中。从整个数据集的相关熵中减去它们的加权熵之和展示了显著的信息增益（图 11-3）。

![](img/fdl2_1103.png)

###### 图 11-3. 原始数据集可以分为 30%有毒和 70%无毒，其中真实标签为 1 表示有毒，否则为 0。根据是否含有酚将 n 个示例分成两个子组，大大集中了每个子组中的真实概率在一个标签上。

这与我们对酚基的先验知识非常吻合——由于其在有毒化合物中的广泛存在和毒性水平，我们本来就期望它是毒性分类的一个重要特征。我们在决策树中选择特征和它们的分裂方式实际上与我们在更一般的算法框架中处理“贪婪算法”的方式相同。贪婪算法在每个决策点选择最优的局部行动，并根据问题的属性，这些局部最优行动的组合导致全局最优解。决策树类似地在每个决策点选择在局部导致某些指标上最大增益的特征和分裂。例如，我们刚刚在毒性分类中使用了信息增益，尽管我们只展示了一个分裂的结果，假设在酚特征上分裂导致最高的信息增益，我们在树的每个级别的每个交叉点执行相同的贪婪过程。然而，事实证明，为数据集找到全局最优决策树的问题是 NP 完全的，这意味着在这个文本中，这是计算上非常困难的。我们能够以可行的方式接近这个问题的最佳方法是贪婪方法，尽管不能证明它会导致全局最优解。

对于树中我们分裂的每个特征，都存在与该特征相关的信息增益。每个特征的重要性顺序只是根据它们的信息增益排序的特征列表。如果我们有一个随机森林而不是单个决策树，我们会在整个森林中对每个特征的信息增益进行平均，并使用平均值进行排序。请注意，在计算信息增益时不需要额外的工作，因为我们首先使用信息增益来训练单个决策树。因此，在基于树的算法中，我们既有每个示例级别的可解释性，又有全局特征重要性的理解。

# 线性回归

线性回归的简要背景：给定一组特征和一个目标变量，我们的目标是找到最佳的特征线性组合，以逼近目标变量。这个模型的隐含假设是输入特征与目标变量呈线性关系。我们将“最佳”定义为导致线性组合与基本事实之间具有最低均方根误差的系数集合：

<math alttext="y equals beta dot x plus epsilon comma epsilon tilde upper N left-parenthesis 0 comma sigma squared right-parenthesis"><mrow><mi>y</mi> <mo>=</mo> <mi>β</mi> <mo>·</mo> <mi>x</mi> <mo>+</mo> <mi>ϵ</mi> <mo>,</mo> <mi>ϵ</mi> <mo>∼</mo> <mi>N</mi> <mo>(</mo> <mn>0</mn> <mo>,</mo> <msup><mi>σ</mi> <mn>2</mn></msup> <mo>)</mo></mrow></math>

其中 <math alttext="beta"><mi>β</mi></math> 代表系数向量。我们内置的全局特征重要性概念直接源自此。与具有最大幅度系数对应的特征在回归中是全局最重要的特征。

那么，特征重要性的一个例子级别的概念呢？回想一下，为了对给定示例进行预测，我们取示例和学习系数之间的点积。逻辑上，对最终结果产生最大贡献的特征与特征系数乘积相关的特征是最重要的预测特征。毫不费力地，我们在线性回归中几乎内置了一个例子级别和全局级别的可解释性概念。

然而，当考虑特征重要性时，线性回归存在一些未解决的问题。例如，在多元回归中特征之间存在显著相关性时，模型往往很难分离这些相关特征对输出的影响。在“SHAP”中，我们将描述 Shapley 值，这些值旨在衡量在这种情况下给定特征对输出的边际、无偏影响。

# 评估特征重要性的方法

对于没有内置特征重要性的模型，研究人员多年来已经开发了各种方法来评估它。在本节中，我们将讨论一些在行业中使用的方法，以及它们的优点和缺点。

## 排列特征重要性

排列特征重要性背后的想法非常简单：假设我们有一个经过训练的神经模型*f*和一组特征*U*，*f*已经在这些特征上进行了训练。我们想要了解单个特征*s*对*f*的预测有什么影响。一种方法是随机重新排列数据集中*s*的取值，并测量预测准确性的降低。如果特征*s*本来就没有增加太多的预测准确性，那么在使用排列样本时，我们应该看到*f*的预测准确性降低得很少。相反，如果特征*s*本来就对输出有预测能力，那么在对数据集中的*s*的值进行排列时，我们应该看到预测准确性大幅下降。实质上，如果特征*s*最初与真实标签强相关，那么随机化*s*的值将破坏这种强相关性，并使其在预测真实标签方面失效。

不幸的是，与所有可解释性方法一样，这种方法并不完美。想象一下，我们的目标是某个地区的冰淇淋销售额，*U*中的两个特征是彼此之间一英里半径内的两个温度传感器的读数。我们预期这些特征中的每一个独立地对冰淇淋销售额具有相当大的预测能力，因为我们的目标具有季节性。然而，如果我们在这个数据集上执行先前介绍的排列方法，我们会出乎意料地得到这两个特征的低特征重要性。为什么会这样？尽管这些特征中的每一个都对目标有很强的预测能力，但由于两个温度传感器的紧密接近，它们也具有很强的相关性。此外，每次只对其中一个特征进行排列以计算其重要性意味着另一个特征保持不变，保留了这两个特征中包含的大部分预测信息。因此，我们会看到*f*对这两个特征的预测性能几乎没有变化，导致我们认为天气对冰淇淋销售没有预测能力。

这里的故事教训是，我们必须始终牢记数据集中特征之间的相关性。在将这些特征通过任何形式的预测建模算法（简单或复杂）之前，了解特征之间的关系是良好的数据科学和机器学习实践。一种方法是将每个经过 z-score 处理的特征相互绘制，以获得特征相关性的视觉概念。

## 部分依赖图

部分依赖图，简称 PDP，衡量了模型中包含的特征子集对输出的边际影响。正如之前讨论的，以无偏的方式测量这种边际影响对于复杂的神经模型来说是困难的。在回归的情况下，我们可以将训练好的神经网络（或任何其他复杂的、不可解释的模型）表示为一个函数*f*，它以特征集*U*作为输入，并在实数中输出一个值。想象一下，作为这个模型的用户，你正在寻找一种可以测量任意特征子集*S*对*f*输出的边际影响的可解释性方法。也就是说，如果我们给定一个任意的特征集*S*的设置，我们希望计算在这个设置条件下函数*f*的期望输出。对*f*的期望是在*U \ S*上取的，即*U*中其余特征（在已知*S*的设置条件下）。直观地说，我们已经将特征子集*U \ S*边缘化，并得到了一个新函数*f’*的输出，它只以特征集*S*作为输入。如果我们对足够多的*S*设置进行这个过程，我们就可以学习到*f’*随着特征集*S*变化而变化的模式。

例如，假设输出是某个地区道路上的车辆数量，我们的特征集*S*包括一个单一特征：该地区的降水量。构成*U \ S*的特征可能是诸如时间、地理位置、人口密度等变量。通过对一系列降水量运行上述过程，我们可以估计在每个水平上预期看到的道路上的车辆数量，并观察随着降水量的升高或降低而出现的趋势。绘制这种趋势就是 PDP 的作用。

一些重要的注意事项：首先，我们并不打算实际学习*f’*，而是使用我们训练好的模型*f*来估计它。学习*f’*本身需要为每个要解释的潜在子集*S*重新训练，这在特征数量方面是指数级的，因此难以处理。其次，目前尚不清楚我们将如何计算在*U \ S*上取的*f*的期望。正如我们将很快看到的，PDP 方法解决了这第二点。在深入讨论之前，这里是我们刚刚描述的过程的一个简单而具体的数学公式：

<math alttext="f prime left-parenthesis upper S right-parenthesis equals double-struck upper E Subscript upper U minus upper S vertical-bar upper S Baseline left-bracket f left-parenthesis upper U minus upper S comma upper S right-parenthesis right-bracket"><mrow><mi>f</mi> <mi>â</mi> <mi></mi> <mi></mi> <mrow><mo>(</mo> <mi>S</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>U</mi><mo>∖</mo><mi>S</mi><mo>|</mo><mi>S</mi></mrow></msub> <mrow><mo>[</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>U</mi> <mo>∖</mo> <mi>S</mi> <mo>,</mo> <mi>S</mi> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

如前所述，条件期望有点难以估计。到目前为止，在文本中，我们通过经验平均值以无偏的方式近似期望。然而，为了估计条件期望，我们受到了一个进一步的限制，即我们必须仅对包含所讨论的*S*的确切设置的样本取平均值。不幸的是，我们从*U*的基础分布中获得的唯一样本都包含在我们提供的数据集中。在*U*的特征通常是连续的情况下，我们在数据集中看到所讨论的*S*的确切设置的可能性甚至一次都很低。为了解决这个问题，PDP 做出了一个独立性假设，允许我们直接使用整个数据集来估计这个期望。

<math alttext="f prime left-parenthesis upper S right-parenthesis equals double-struck upper E Subscript upper U minus upper S vertical-bar upper S Baseline left-bracket f left-parenthesis upper U minus upper S comma upper S right-parenthesis right-bracket equals double-struck upper E Subscript upper U minus upper S Baseline left-bracket f left-parenthesis upper U minus upper S comma upper S right-parenthesis right-bracket almost-equals StartFraction 1 Over n EndFraction sigma-summation Underscript i equals 1 Overscript n Endscripts f left-parenthesis left-parenthesis upper U minus upper S right-parenthesis Superscript i Baseline comma upper S right-parenthesis"><mrow><mi>f</mi> <mi>â</mi> <mi></mi> <mi></mi> <mrow><mo>(</mo> <mi>S</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>U</mi><mo>∖</mo><mi>S</mi><mo>|</mo><mi>S</mi></mrow></msub> <mrow><mo>[</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>U</mi> <mo>∖</mo> <mi>S</mi> <mo>,</mo> <mi>S</mi> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>U</mi><mo>∖</mo><mi>S</mi></mrow></msub> <mrow><mo>[</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>U</mi> <mo>∖</mo> <mi>S</mi> <mo>,</mo> <mi>S</mi> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>≈</mo> <mfrac><mn>1</mn> <mi>n</mi></mfrac> <msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></msubsup> <mi>f</mi> <mrow><mo>(</mo> <msup><mrow><mo>(</mo><mi>U</mi><mo>∖</mo><mi>S</mi><mo>)</mo></mrow> <mi>i</mi></msup> <mo>,</mo> <mi>S</mi> <mo>)</mo></mrow></mrow></math>

在数据集中有*n*个样本。PDP 假设*S*中的特征与*U \ S*中的特征是独立的。这个假设使我们能够无差别地使用所有的训练样本来计算期望，因为在这个假设下，*U \ S*的抽样与*S*的设置是独立的。我们现在有了一个具体的方法来估计任意特征子集*S*对*f*输出的边际影响。

如果*S*中的特征与*U \ S*中的特征之间存在显著相关性，那么我们生成的 PDP 可能不反映*S*对输出的真实边际效应，因为我们的抽样假设存在偏差。基本上，我们会对许多极不可能发生的样本进行平均，这意味着我们（1）不能期望*f*在这些样本上生成有意义的输出，以及（2）对不准确反映*U*上的基础分布关系的样本输出进行平均。

第二个问题可能很明显，但为了说明第一个问题，想象一下你已经在 MNIST 数据集上完全训练了一个神经网络。现在，我在网上找到一张狗的图片并将这张图片通过网络运行。碰巧，网络以高置信度返回一个 9——我们应该相信这些结果吗？由于输入图像完全超出了模型期望看到的图像分布范围，我们不能相信模型能够推广到这种程度。尽管我们与 PDP 的情况稍微不同，但是类似——我们基本上创建了这些不太可能的、超出分布范围的“弗兰肯样本”，并期望*f*在这些样本上产生有意义的输出。PDP 做出的独立假设是该方法的固有限制，因为我们唯一拥有的样本是数据集中的样本。此外，PDP 经常用于分析小特征子集（<math alttext="2"><mrow><mo>≤</mo> <mn>2</mn></mrow></math>），因为人类只能直观地解释最多三个维度。不过，PDP 可以是一种有效的方法，用于可视化输入特征子集与复杂模型输出之间的趋势。

# 抽取性理性化

抽取性理性化，或者选择保留大部分或全部相关信息以预测属性所需的输入的简洁部分，是一种内置的可解释性形式在示例级别。在本节中，我们将回顾论文“理性化神经预测”的方法，试图在自然语言空间中实现这一目标。在这篇论文中，任务是属性预测：给定一个文本评论，预测文本的一些属性。该论文专门使用了一个啤酒评论数据集，其中每个评论包括一些文本以及外观评分、气味评分和口感评分。

高性能但不可解释的方法是使用循环架构训练经典属性预测器，然后使用香草回归神经网络，该神经网络以循环架构生成的最终嵌入作为输入，如图 11-4 所示。

![](img/fdl2_1104.png)

###### 图 11-4。描绘了经典属性预测器，其中 x 是原始句子的编码，h(x)是在到达 x 末尾后由循环架构产生的隐藏状态，y 是标准前馈神经架构的结果。

本文的目标是额外生成理由，或者选择性的，输入文本中与被预测属性最相关的简洁部分，同时限制对性能的影响。这就是为什么这种理性化方法被称为“抽取性”——它通过提取输入的相关部分来工作。你可能会想为什么要强调简洁性。如果模型生成的理由没有简洁性的限制或惩罚，那么模型就没有理由只返回整个输入，这是一个微不足道的解决方案。当然，如果理由是整个输入，那么预测输出所需的所有信息都在理由中。

如何修改所提出的属性预测器的结构，以便也能产生原因作为内置机制？本文提出了一种双网络方法，第一个网络被称为生成器，第二个网络被称为编码器。生成器是一个负责选择原因的 RNN，而编码器是一个负责仅根据原因而不是整个输入来预测属性的 RNN。这背后的逻辑是，给定正确的目标函数，生成器将必须学会选择输入文本的有意义部分，以便准确预测真实评分。生成器参数化了一个分布，覆盖了可以应用于输入的所有可能的二进制掩码，其中 1 表示该词应包含在原因中，0 表示否。图 11-5 展示了所提出的两步架构，其中编码器只是之前图中显示的单步属性预测器，z 代表从生成器中采样的二进制掩码。

![](img/fdl2_1105.png)

###### 图 11-5. 生成器参数化了给定输入 x 的掩码 z 的分布，我们从中采样以获得输入到编码器的输入。编码器遵循之前所示的经典属性预测器的相同结构。

更正式地，我们将输入文本*x*表示为一个向量，其中*x[i]*表示位置*i*处的标记。生成器参数化了分布*p(z|x)*，其中*z*是由各个伯努利随机变量*z[i]*组成的向量，如果*x[i]*应包含在原因中，则每个*z[i]*取值为 1，否则为 0。注意*z*的长度与*x*相同，这取决于*x*。我们如何表示这个分布呢？第一步是做一个合理的条件独立假设，即所有*z[i]*在给定*x*的条件下互相独立：这是一个非常合理的假设，因为关于*x[i]*是否应该包含在原因中的所有信息应该包含在*x*本身中（标记*x[i]*及其周围的上下文）。将这转换为神经网络术语，我们可以通过将完全连接的层应用于生成器的每个最终隐藏状态*h[i]*，然后跟随一个 sigmoid 激活，独立地得到*z[i]*取值为 1 的概率，我们很快将看到。

在进入目标函数的具体细节之前，我们将更详细地描述生成器和编码器的架构。生成器和编码器都是循环架构，其中循环单元可以是 LSTM 或 GRU。正如前一段所述，生成器为每个标记*x[i]*生成一个隐藏单元*h[i]*。标记的最终嵌入由两个中间嵌入组成：第一个中间嵌入是通过标记的前向传递的结果，而第二个中间嵌入是通过标记的后向传递的结果。更正式地说，我们有：

<math alttext="StartLayout 1st Row  ModifyingAbove h With right-arrow Subscript i Baseline equals ModifyingAbove f With right-arrow left-parenthesis ModifyingAbove h With right-arrow Subscript i minus 1 Baseline comma x Subscript i Baseline right-parenthesis 2nd Row  ModifyingAbove h With left-arrow Subscript i Baseline equals ModifyingAbove f With left-arrow left-parenthesis ModifyingAbove h With left-arrow Subscript i plus 1 Baseline comma x Subscript i Baseline right-parenthesis 3rd Row  h Subscript i Baseline equals concat left-parenthesis ModifyingAbove h With right-arrow Subscript i Baseline comma ModifyingAbove h With left-arrow Subscript i Baseline right-parenthesis EndLayout"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><msub><mover accent="true"><mi>h</mi> <mo>→</mo></mover> <mi>i</mi></msub> <mo>=</mo> <mover accent="true"><mi>f</mi> <mo>→</mo></mover> <mrow><mo>(</mo> <msub><mover accent="true"><mi>h</mi> <mo>→</mo></mover> <mrow><mi>i</mi><mo>-</mo><mn>1</mn></mrow></msub> <mo>,</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><msub><mover accent="true"><mi>h</mi> <mo>←</mo></mover> <mi>i</mi></msub> <mo>=</mo> <mover accent="true"><mi>f</mi> <mo>←</mo></mover> <mrow><mo>(</mo> <msub><mover accent="true"><mi>h</mi> <mo>←</mo></mover> <mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>,</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><msub><mi>h</mi> <mi>i</mi></msub> <mo>=</mo> <mtext>concat</mtext> <mrow><mo>(</mo> <msub><mover accent="true"><mi>h</mi> <mo>→</mo></mover> <mi>i</mi></msub> <mo>,</mo> <msub><mover accent="true"><mi>h</mi> <mo>←</mo></mover> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></mtd></mtr></mtable></math>

其中<math alttext="ModifyingAbove f With right-arrow"><mover accent="true"><mi>f</mi> <mo>→</mo></mover></math>和<math alttext="ModifyingAbove f With left-arrow"><mover accent="true"><mi>f</mi> <mo>←</mo></mover></math>对应于两个独立的循环单元，前者在前向传递上训练，后者在后向传递上训练。从这个公式中，我们可以看到最终嵌入是双向的，包含来自标记整个上下文的信息，而不仅仅是单向的信息。然后，本文对每个嵌入应用单个全连接层和 Sigmoid 函数，以生成每个标记的独立伯努利随机变量：

<math alttext="p left-parenthesis z Subscript i Baseline vertical-bar x right-parenthesis equals sigma left-parenthesis w Subscript z Baseline dot h Subscript i Baseline plus b Subscript z Baseline right-parenthesis"><mrow><mi>p</mi> <mrow><mo>(</mo> <msub><mi>z</mi> <mi>i</mi></msub> <mo>|</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>σ</mi> <mrow><mo>(</mo> <msub><mi>w</mi> <mi>z</mi></msub> <mo>·</mo> <msub><mi>h</mi> <mi>i</mi></msub> <mo>+</mo> <msub><mi>b</mi> <mi>z</mi></msub> <mo>)</mo></mrow></mrow></math>

编码器也是一个循环架构，但由于其预测与文本相关联的评分的目的，它被设计为一个回归架构。因此，编码器可以像我们在前面部分提到的普通属性预测器一样设计。

那么，对于同时训练这两个网络的正确目标函数是什么？除了我们可能希望生成器产生的理由的任何约束条件外，我们还必须确保预测器准确。如果预测器不准确，生成器就没有理由产生有意义的理由。将所有这些放在一起形成数学公式，我们有以下目标函数：

<math alttext="StartLayout 1st Row  theta Superscript asterisk Baseline comma phi Superscript asterisk Baseline equals argmin Subscript theta comma phi Baseline upper L left-parenthesis theta comma phi right-parenthesis 2nd Row  upper L left-parenthesis theta comma phi right-parenthesis equals sigma-summation Underscript left-parenthesis x comma y right-parenthesis element-of upper D Endscripts double-struck upper E Subscript z tilde g e n Sub Subscript theta Subscript left-parenthesis x right-parenthesis Baseline left-bracket cost left-parenthesis x comma y comma z right-parenthesis right-bracket 3rd Row  cost left-parenthesis x comma y comma z right-parenthesis equals lamda 1 asterisk StartAbsoluteValue z EndAbsoluteValue plus lamda 2 asterisk sigma-summation Underscript t Endscripts StartAbsoluteValue z Subscript t Baseline minus z Subscript t minus 1 Baseline EndAbsoluteValue plus StartAbsoluteValue EndAbsoluteValue e n c Subscript phi Baseline left-parenthesis x comma z right-parenthesis minus y StartAbsoluteValue EndAbsoluteValue Subscript 2 Superscript 2 EndLayout" display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><msup><mi>θ</mi> <mo>*</mo></msup> <mo>,</mo> <msup><mi>φ</mi> <mo>*</mo></msup> <mo>=</mo> <msub><mtext>argmin</mtext> <mrow><mi>θ</mi><mo>,</mo><mi>φ</mi></mrow></msub> <mi>L</mi> <mrow><mo>(</mo> <mi>θ</mi> <mo>,</mo> <mi>φ</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mi>L</mi> <mrow><mo>(</mo> <mi>θ</mi> <mo>,</mo> <mi>φ</mi> <mo>)</mo></mrow> <mo>=</mo> <munder><mo>∑</mo> <mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo><mo>∈</mo><mi>D</mi></mrow></munder> <msub><mi>𝔼</mi> <mrow><mi>z</mi><mo>∼</mo><mi>g</mi><mi>e</mi><msub><mi>n</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow></msub> <mrow><mo>[</mo> <mtext>cost</mtext> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mtext>cost</mtext> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mi>λ</mi> <mn>1</mn></msub> <mrow><mo>*</mo> <mo>|</mo> <mi>z</mi> <mo>|</mo></mrow> <mo>+</mo> <msub><mi>λ</mi> <mn>2</mn></msub> <mo>*</mo> <munder><mo>∑</mo> <mi>t</mi></munder> <mrow><mo>|</mo></mrow> <msub><mi>z</mi> <mi>t</mi></msub> <mo>-</mo> <msub><mi>z</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub> <mrow><mo>|</mo> <mo>+</mo> <mo>|</mo> <mo>|</mo> <mi>e</mi> <mi>n</mi></mrow> <msub><mi>c</mi> <mi>φ</mi></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>-</mo> <mi>y</mi> <msubsup><mrow><mo>|</mo><mo>|</mo></mrow> <mn>2</mn> <mn>2</mn></msubsup></mrow></mtd></mtr></mtable></math>

其中λ1 和λ2 是我们可以在验证过程中调整的超参数。本文中使用的成本函数还包含一个连续性惩罚，当理由在文本中间隔而不是一个连续的块时，惩罚更高。我们希望最小化每个训练示例的预期成本之和，其中理由根据生成器分布进行抽样。由于*z*的配置数量随着*x*的长度呈指数增长，精确计算预期成本是计算上的障碍，因此我们希望能够通过一些经验抽样估计来近似预期成本的梯度。

这对于成本函数相对于编码器参数的梯度是可行的，但是当我们尝试为生成器做同样的事情时，我们遇到了一个类似的问题，就像我们第一次尝试优化 VAE 编码器时一样：

<math alttext="normal nabla Subscript theta Baseline double-struck upper E Subscript z tilde g e n Sub Subscript theta Subscript left-parenthesis x right-parenthesis Baseline left-bracket cost left-parenthesis x comma y comma z right-parenthesis right-bracket equals sigma-summation Underscript z Endscripts cost left-parenthesis x comma y comma z right-parenthesis asterisk normal nabla Subscript theta Baseline p Subscript theta Baseline left-parenthesis z vertical-bar x right-parenthesis"><mrow><msub><mi>∇</mi> <mi>θ</mi></msub> <msub><mi>𝔼</mi> <mrow><mi>z</mi><mo>∼</mo><mi>g</mi><mi>e</mi><msub><mi>n</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow></msub> <mrow><mo>[</mo> <mtext>cost</mtext> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>=</mo> <msub><mo>∑</mo> <mi>z</mi></msub> <mtext>cost</mtext> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>*</mo> <msub><mi>∇</mi> <mi>θ</mi></msub> <msub><mi>p</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>|</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>

请注意，成本函数间接地是通过从生成器中抽样而不是通过θ的函数，因此可以视为常数。我们无法将其重新表达为期望，因为梯度是相对于我们从中抽样的分布。本文使用 log 技巧来解决这个问题，我们在 VAE 部分也介绍过这个技巧：

<math alttext="StartLayout 1st Row  sigma-summation Underscript z Endscripts cost left-parenthesis x comma y comma z right-parenthesis asterisk normal nabla Subscript theta Baseline p Subscript theta Baseline left-parenthesis z vertical-bar x right-parenthesis 2nd Row  equals sigma-summation Underscript z Endscripts cost left-parenthesis x comma y comma z right-parenthesis asterisk p Subscript theta Baseline left-parenthesis z vertical-bar x right-parenthesis asterisk normal nabla Subscript theta Baseline log p Subscript theta Baseline left-parenthesis z vertical-bar x right-parenthesis 3rd Row  equals double-struck upper E Subscript z tilde g e n Sub Subscript theta Subscript left-parenthesis x right-parenthesis Baseline left-bracket cost left-parenthesis x comma y comma z right-parenthesis asterisk normal nabla Subscript theta Baseline log p Subscript theta Baseline left-parenthesis z vertical-bar x right-parenthesis right-bracket EndLayout"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><munder><mo>∑</mo> <mi>z</mi></munder> <mtext>cost</mtext> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>*</mo> <msub><mi>∇</mi> <mi>θ</mi></msub> <msub><mi>p</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>|</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mo>=</mo> <munder><mo>∑</mo> <mi>z</mi></munder> <mtext>cost</mtext> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>*</mo> <msub><mi>p</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>|</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>*</mo> <msub><mi>∇</mi> <mi>θ</mi></msub> <mo form="prefix">log</mo> <msub><mi>p</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>|</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>z</mi><mo>∼</mo><mi>g</mi><mi>e</mi><msub><mi>n</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow></msub> <mrow><mo>[</mo> <mtext>cost</mtext> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>*</mo> <msub><mi>∇</mi> <mi>θ</mi></msub> <mo form="prefix">log</mo> <msub><mi>p</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>|</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></mtd></mtr></mtable></math>

成本函数相对于编码器参数的梯度就是：

<math alttext="StartLayout 1st Row  normal nabla Subscript phi Baseline double-struck upper E Subscript z tilde g e n Sub Subscript theta Subscript left-parenthesis x right-parenthesis Baseline left-bracket cost left-parenthesis x comma y comma z right-parenthesis right-bracket 2nd Row  equals sigma-summation Underscript z Endscripts p Subscript theta Baseline left-parenthesis z vertical-bar x right-parenthesis asterisk normal nabla Subscript phi Baseline cost left-parenthesis x comma y comma z right-parenthesis 3rd Row  equals double-struck upper E Subscript z tilde g e n Sub Subscript theta Subscript left-parenthesis x right-parenthesis Baseline left-bracket normal nabla Subscript phi Baseline cost left-parenthesis x comma y comma z right-parenthesis right-bracket EndLayout"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><msub><mi>∇</mi> <mi>φ</mi></msub> <msub><mi>𝔼</mi> <mrow><mi>z</mi><mo>∼</mo><mi>g</mi><mi>e</mi><msub><mi>n</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow></msub> <mrow><mo>[</mo> <mtext>cost</mtext> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mo>=</mo> <munder><mo>∑</mo> <mi>z</mi></munder> <msub><mi>p</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>|</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>*</mo> <msub><mi>∇</mi> <mi>φ</mi></msub> <mtext>cost</mtext> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>z</mi><mo>∼</mo><mi>g</mi><mi>e</mi><msub><mi>n</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow></msub> <mrow><mo>[</mo> <msub><mi>∇</mi> <mi>φ</mi></msub> <mtext>cost</mtext> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>,</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></mtd></mtr></mtable></math>

这类似于在执行 SGD 或小批量梯度下降时的期望梯度的标准经验估计。

我们如何同时训练这两个网络？也许从一个单一的训练示例开始会更容易。我们首先从数据集中随机选择一个训练示例，其中一个训练示例包括一个文本评论和一个相关的评分，并将文本评论输入生成器。生成器现在代表了给定输入文本评论的所有可能二进制掩码的概率分布，可以通过独立抽样每个*z[i]*来从中抽样，这是由于我们之前的条件独立声明。每个抽样的二进制掩码代表一个可能的理由，然后我们将其馈送给编码器进行预测。在获得每个理由的编码器结果后，我们就有了计算每个理由的成本函数所需的所有信息。这足以更新编码器的权重，但要更新生成器的权重，我们还需要跟踪理由的对数似然，或者对于每个抽样的*z^k*，即*log p Subscript theta Baseline left-parenthesis z Superscript k Baseline vertical-bar x right-parenthesis*。

现在我们有了一个训练的机制，那么如何将其转化为验证和测试我们的模型呢？在验证和测试阶段，我们不再从生成器中对二进制掩码进行抽样，而是根据生成器的概率分布选择最可能的二进制掩码。为了选择最可能的二进制掩码，我们只需要为输入测试评论*x*中的每个*x[i]*选择最可能的*z[i]*，这是由于我们之前的条件独立假设。这是一个非常合理的测试方法，因为这是我们在现实世界中使用该模型时确定预期理由的方式。

您可能已经注意到了与注意力概念的一些相似之处。毕竟，生成的二进制掩码可以被视为一组权重向量，我们用这些权重乘以构成输入文本评论的特征向量，其中这些权重要么是 0 要么是 1，而不是标准注意力中实现的一些连续加权方案。事实上，本文的作者提到他们的方法可以被视为一种“硬”注意力形式，其中我们根据概率分布完全屏蔽或输入输入令牌，而不是计算输入中特征向量的加权平均值。您可能会想知道为什么在这种情况下硬注意力比前一节中提出的“软”注意力方案更有意义。在这种情况下，硬注意力方案更有意义，因为句子中的单词上的分数权重很难解释为重要性的度量，而选择文本中的严格子集作为评分解释则更容易解释。

# 石灰

石灰，或称为局部可解释的模型无关解释，是一种可解释性技术，应用于经过训练的模型而不是模型本身的内置特性。石灰是一种逐例解释方法，意味着它生成了对潜在复杂行为的简单、局部解释。它还是模型无关的，这意味着在应用石灰时，基础模型的结构本身并不重要。

在描述 LIME 的方法论之前，作者花了一些时间来勾勒出他们认为是任何解释器必要组成部分的几个特征。第一个是可解释性，意味着解释器应该提供一个“输入变量和响应之间的定性关系”，这对用户来说应该是容易理解的。即使原始模型中使用的特征是不可解释的，解释器也必须使用人类可以解释的特征。例如，在自然语言处理的应用中，即使基础模型利用复杂的词嵌入来表示任何给定的单词，解释器也必须使用人类可以理解的特征，比如原始单词本身。

第二个特征是局部忠实度，这意味着解释器必须在所选示例的某个邻域内与基础模型表现出类似的行为。我们可能会问，为什么是局部而不是全局的忠实度？然而，正如论文所指出的，全局忠实度相当难以实现，需要领域内的重大进展——如果能够实现全局忠实度，那么解释性领域的大部分问题将得到解决。因此，我们选择局部忠实度。

第三个是解释器对模型不可知，这意味着，正如我们之前解释的那样，基础模型的结构本身不应该重要。基础模型可以是线性回归模型，也可以是复杂的卷积神经网络结构，解释器仍然应该能够满足其他三个特征。解释器对模型不可知允许基础模型结构的灵活性，这是可取的，因为这不需要解释器结构的改变。

第四个也是最后一个特征是全局视角，即选择代表模型行为的一部分示例的解释。这有助于建立用户对模型的信任。

现在我们将花一些时间来发展 LIME 的方法论。正如所述，原始模型的特征可能对人类不可解释（对于大多数复杂模型来说通常如此），因此解释器使用的特征将不同于基础模型使用的特征。解释器使用的特征可以是 NLP 任务中的单词，也可以是化学性质预测任务中的功能组——即，最终用户可以轻松理解的单位或可解释的组件。因此，任何转换为解释器特征空间的示例都将成为一个二进制向量，其中每个索引与一个不同的可解释组件（如功能组）相关联。在任何索引*i*处的 1 表示原始示例中存在相关的可解释组件，而 0 表示原始示例中缺少该组件。根据参考文献中使用的符号，我们将表示为要解释的示例的原始特征表示为 <math alttext="x element-of double-struck upper R Superscript d"><mrow><mi>x</mi> <mo>∈</mo> <msup><mi>ℝ</mi> <mi>d</mi></msup></mrow></math>，表示为由解释器处理的表示为 <math alttext="x prime element-of StartSet 0 comma 1 EndSet Superscript d prime"><mrow><mi>x</mi> <mi>â</mi> <mi></mi> <mi></mi> <mo>∈</mo> <msup><mrow><mo>{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>}</mo></mrow> <mrow><mi>d</mi><mi>â</mi><mi></mi><mi></mi></mrow></msup></mrow></math>，其中*d’*是可解释组件的数量。

此外，论文定义*G*为一类潜在可解释模型，如线性回归或随机森林，解释器是一个实例<math alttext="g element-of upper G"><mrow><mi>g</mi> <mo>∈</mo> <mi>G</mi></mrow></math>。*g*作用于实例*x'*并返回基础模型范围内的值。我们将基础模型表示为*f*，它作用于实例*x*，是从<math alttext="double-struck upper R Superscript d Baseline right-arrow double-struck upper R"><mrow><msup><mi>ℝ</mi> <mi>d</mi></msup> <mo>→</mo> <mi>ℝ</mi></mrow></math>到[0,1]范围的函数，其中*f*返回一个概率分布。此外，论文定义了一个接近度度量或核<math alttext="pi Subscript x Baseline left-parenthesis z right-parenthesis"><mrow><msub><mi>π</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow></mrow></math>，围绕实例*x*定义。这个函数可以以多种方式定义——大多数 LIME 的实现使用一个指数核，在*x*处达到最大值，并随着离*x*越来越远而指数级下降。

在高层次上，LIME 试图找到最小化类似于损失函数的解释<math alttext="g Superscript asterisk"><msup><mi>g</mi> <mo>*</mo></msup></math>：

<math alttext="g Superscript asterisk Baseline equals argmin Subscript g element-of upper G Baseline upper L left-parenthesis f comma g comma x right-parenthesis plus omega left-parenthesis g right-parenthesis"><mrow><msup><mi>g</mi> <mo>*</mo></msup> <mo>=</mo> <msub><mtext>argmin</mtext> <mrow><mi>g</mi><mo>∈</mo><mi>G</mi></mrow></msub> <mi>L</mi> <mrow><mo>(</mo> <mi>f</mi> <mo>,</mo> <mi>g</mi> <mo>,</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>+</mo> <mi>ω</mi> <mrow><mo>(</mo> <mi>g</mi> <mo>)</mo></mrow></mrow></math>

*L(f,g,x)*是*g*在问题实例*x*周围建模*f*不忠实的度量，<math alttext="omega left-parenthesis g right-parenthesis"><mrow><mi>ω</mi> <mo>(</mo> <mi>g</mi> <mo>)</mo></mrow></math>是*g*的复杂度的度量。因此，最小化它们的和会得到一个具有局部忠实度和可解释性特征的最佳解释器<math alttext="g Superscript asterisk"><msup><mi>g</mi> <mo>*</mo></msup></math>。

我们如何衡量潜在解释器的不忠实度？论文的方法是从*x'*的附近对*z'*进行采样，将*z'*转换回原始特征空间中的示例*z*，并计算*f(z)*和*g(z')*之间的差异。差异代表该样本的损失——如果*g(z')*远离*f(z)*，那么它就不忠实于该点的模型预测。然后，我们可以使用核<math alttext="pi Subscript x Baseline left-parenthesis z right-parenthesis"><mrow><msub><mi>π</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow></mrow></math>对这种损失进行加权，随着样本*z*越来越远离原始示例*x*，这种损失逐渐减少。将这些放在一起，损失函数看起来像：

<math alttext="upper L left-parenthesis f comma g comma x right-parenthesis equals sigma-summation Underscript z comma z Superscript prime Baseline Endscripts pi Subscript x Baseline left-parenthesis z right-parenthesis asterisk left-parenthesis f left-parenthesis z right-parenthesis minus g left-parenthesis z prime right-parenthesis right-parenthesis squared"><mrow><mi>L</mi> <mrow><mo>(</mo> <mi>f</mi> <mo>,</mo> <mi>g</mi> <mo>,</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mo>∑</mo> <mrow><mi>z</mi><mo>,</mo><mi>z</mi><mi>â</mi><mi></mi><mi></mi></mrow></msub> <msub><mi>π</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>*</mo> <msup><mrow><mo>(</mo><mi>f</mi><mrow><mo>(</mo><mi>z</mi><mo>)</mo></mrow><mo>-</mo><mi>g</mi><mrow><mo>(</mo><mi>z</mi><mi>â</mi><mi></mi><mi></mi><mo>)</mo></mrow><mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>

我们如何获得用于这个损失函数的样本*z'*？论文从*x'*的附近对*z'*进行采样，通过随机选择*x'*的非零分量的子集，并将样本的所有其他索引设置为零来实现这一目的（图 11-6）。

![](img/fdl2_1106.png)

###### 图 11-6。x 可以被视为一些高维输入，比如图像，而 x'的每个索引与一些可解释特征相关联，其中 1 表示该特征在 x 中存在。采样过程选择 x'中的一些非零索引的子集，以保持 w'和 z'中的每个非零索引，然后将它们映射回原始输入空间。

然后，LIME 将这些样本*z'*映射回原始特征空间中的样本*z*，以便我们可以通过*f(z) - g(z')*来衡量解释器的忠实度。

LIME 还考虑了解释器的复杂性，通过<math alttext="omega left-parenthesis g right-parenthesis"><mrow><mi>ω</mi> <mo>(</mo> <mi>g</mi> <mo>)</mo></mrow></math>，这强调了可行解释器的可解释性方面。在*G*代表线性模型类的特定情况下，论文使用了一个版本的<math alttext="omega"><mi>ω</mi></math>，它对*g*中的非零权重数量设置了一个硬限制：

<math alttext="omega left-parenthesis g right-parenthesis equals normal infinity asterisk 1 left-bracket StartAbsoluteValue EndAbsoluteValue w Subscript g Baseline StartAbsoluteValue EndAbsoluteValue Subscript 0 Baseline greater-than upper K right-bracket"><mrow><mi>ω</mi> <mrow><mo>(</mo> <mi>g</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>∞</mi> <mo>*</mo> <mn>1</mn> <mo>[</mo> <mo>|</mo> <mo>|</mo> <msub><mi>w</mi> <mi>g</mi></msub> <mo>|</mo> <msub><mo>|</mo> <mn>0</mn></msub> <mo>></mo> <mi>K</mi> <mo>]</mo></mrow></math>

其中<math alttext="omega left-parenthesis g right-parenthesis"><mrow><mi>ω</mi> <mo>(</mo> <mi>g</mi> <mo>)</mo></mrow></math>代表*g*的权重向量，L0 范数计算<math alttext="omega left-parenthesis g right-parenthesis"><mrow><mi>ω</mi> <mo>(</mo> <mi>g</mi> <mo>)</mo></mrow></math>中的非零元素数量，*1[*]*是指示函数，如果函数内的条件满足则结果为 1，否则为 0。结果是<math alttext="omega left-parenthesis g right-parenthesis"><mrow><mi>ω</mi> <mo>(</mo> <mi>g</mi> <mo>)</mo></mrow></math>在<math alttext="omega left-parenthesis g right-parenthesis"><mrow><mi>ω</mi> <mo>(</mo> <mi>g</mi> <mo>)</mo></mrow></math>中有超过*K*个非零元素时取得无穷大的值，否则为 0。这确保了所选的<math alttext="omega left-parenthesis g right-parenthesis"><mrow><mi>ω</mi> <mo>(</mo> <mi>g</mi> <mo>)</mo></mrow></math>最多有*K*个非零元素，因为通过将权重归零直到最多有*K*个非零权重，总是可以比任何具有超过*K*个非零元素的提议<math alttext="omega left-parenthesis g right-parenthesis"><mrow><mi>ω</mi> <mo>(</mo> <mi>g</mi> <mo>)</mo></mrow></math>更好。这种正则化方法可能与您过去遇到的正则化方法不同，比如对权重向量的 L1 或 L2 范数。事实上，为了优化论文中定义的目标函数，作者们利用了一个他们称之为 K-LASSO 的算法，该算法首先通过 LASSO 选择*K*个特征，然后执行标准的最小二乘优化。

经过 LIME 的处理，我们得到了一个最优的解释器*g*，在这种情况下，它是一个具有最多*K*个非零权重的线性模型。现在我们必须检查*g*是否满足作者在论文开头设定的目标。首先，*g*必须是可解释的。由于我们选择了一个相对简单的解释器模型*G*，在这个例子中是线性模型，我们只需要解释模型在所选示例*x*周围的行为的（最多）*K*个非零权重的值。与非零权重相关联的可解释组件被认为在该区域的预测中最为重要。在局部忠实度方面，我们的优化过程有助于通过最小化解释器的预测与模型预测之间的最小二乘损失来确保局部忠实度。然而，存在一些限制；例如，论文指出，如果底层模型即使在我们解释的示例的短范围内也是高度非线性的，我们的线性解释器将无法充分展现模型的局部行为。关于模型不可知性，需要注意的是，LIME 的方法并不关心底层模型的结构。LIME 运行所需的仅仅是来自底层模型的预测*f(z)*。最后，为了获得全局视角，我们可以选择代表模型行为的示例，并向用户展示它们的解释。

# SHAP

SHAP，或 Shapley Additive Explanations，是一种针对复杂模型的每个预测的可解释性方法。介绍 SHAP 方法的论文首先提供了一个框架，作者认为这个框架统一了领域中各种可解释性方法。这个框架被称为*加性特征归因*，其中所有这个框架的实例都使用作用于二进制变量的线性解释模型：

<math alttext="g left-parenthesis x prime right-parenthesis equals phi 0 plus sigma-summation Underscript i equals 1 Overscript upper M Endscripts phi Subscript i Baseline x prime Subscript i"><mrow><mi>g</mi> <mrow><mo>(</mo> <msup><mi>x</mi> <mo>'</mo></msup> <mo>)</mo></mrow> <mo>=</mo> <msub><mi>φ</mi> <mn>0</mn></msub> <mo>+</mo> <msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>M</mi></msubsup> <msub><mi>φ</mi> <mi>i</mi></msub> <msubsup><mi>x</mi> <mi>i</mi> <mo>'</mo></msubsup></mrow></math>

其中*M*是二进制变量的数量。例如，当使用一类线性解释器模型时，LIME 完全遵循这个框架，因为要解释的每个示例首先被转换为一个关于可解释组件的二进制向量。事实证明，在加性特征归因框架中，存在一个具有三个理想属性的唯一解决方案：局部准确性、缺失性和一致性。在讨论这个唯一解决方案之前，我们将更详细地描述这三个属性。

第一个是*局部准确性*，它指出解释器模型必须在被解释的示例上与基础模型完全匹配。这是一个理想的属性，因为至少应该完美解释被解释的示例。重要的是要注意，并非所有的可解释性框架都必须遵循这个属性。例如，LIME 生成的解释器，正如其原始论文中所述并在前一节中描述的那样，不一定在 SHAP 作者定义的局部准确性方面是准确的。这将在本节末尾进一步讨论。在 SHAP 中，局部准确性在数学上定义为：

<math alttext="f left-parenthesis x right-parenthesis equals g left-parenthesis x prime right-parenthesis equals phi 0 plus sigma-summation Underscript i equals 1 Overscript upper M Endscripts phi Subscript i Baseline x Subscript i Superscript prime"><mrow><mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>g</mi> <mrow><mo>(</mo> <mi>x</mi> <mi>â</mi> <mi></mi> <mi></mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mi>φ</mi> <mn>0</mn></msub> <mo>+</mo> <msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>M</mi></msubsup> <msub><mi>φ</mi> <mi>i</mi></msub> <msubsup><mi>x</mi> <mi>i</mi> <msup><mo>'</mo></msup></msubsup></mrow></math>

注意*x'*是一个简化的特征向量，其中*x'*中的每个特征都是一个二进制变量，表示原始输入空间中复杂特征的存在或缺失。第二个理想属性是*缺失性*，它指出如果*x'*包含值为零的特征，则解释器模型中与这些特征相关的权重也应为零。这也是一个理想的属性，因为在线性解释器*g*下，值为零的特征对输出没有影响，因此在解释器中不需要为该特征分配非零权重。

最后，第三个理想属性是*一致性*。该属性指出，如果基础模型发生变化，使得解释器空间中的一个特征要么增加要么保持其贡献恒定，无论与原始模型相比解释器空间中其他特征的值如何，那么与该特征相关的解释器权重应该对于更改后的基础模型而言比原始模型更大。这是一个复杂的概念，因此我们更精确地用数学符号表示：

<math alttext="If f prime left-parenthesis h Subscript x Baseline left-parenthesis z Superscript prime Baseline right-parenthesis right-parenthesis minus f prime left-parenthesis h Subscript x Baseline left-parenthesis z prime minus StartSet i EndSet right-parenthesis right-parenthesis greater-than-or-equal-to f left-parenthesis h Subscript x Baseline left-parenthesis z Superscript prime Baseline right-parenthesis right-parenthesis minus f left-parenthesis h Subscript x Baseline left-parenthesis z prime minus StartSet i EndSet right-parenthesis right-parenthesis comma for-all z Superscript prime Baseline comma then phi Subscript i Baseline left-parenthesis f prime comma x right-parenthesis greater-than-or-equal-to phi Subscript i Baseline left-parenthesis f comma x right-parenthesis"><mrow><mtext>If</mtext> <msup><mi>f</mi> <mo>'</mo></msup> <mrow><mo>(</mo> <msub><mi>h</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <msup><mi>z</mi> <mo>'</mo></msup> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>-</mo> <msup><mi>f</mi> <mo>'</mo></msup> <mrow><mo>(</mo> <msub><mi>h</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mi>â</mi> <mi></mi> <mi></mi> <mo>∖</mo> <mrow><mo>{</mo> <mi>i</mi> <mo>}</mo></mrow> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>≥</mo> <mi>f</mi> <mrow><mo>(</mo> <msub><mi>h</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <msup><mi>z</mi> <mo>'</mo></msup> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>-</mo> <mi>f</mi> <mrow><mo>(</mo> <msub><mi>h</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mi>â</mi> <mi></mi> <mi></mi> <mo>∖</mo> <mrow><mo>{</mo> <mi>i</mi> <mo>}</mo></mrow> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>,</mo> <mo>∀</mo> <msup><mi>z</mi> <mo>'</mo></msup> <mo>,</mo> <mtext>then</mtext> <msub><mi>φ</mi> <mi>i</mi></msub> <mrow><mo>(</mo> <msup><mi>f</mi> <mo>'</mo></msup> <mo>,</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>≥</mo> <msub><mi>φ</mi> <mi>i</mi></msub> <mrow><mo>(</mo> <mi>f</mi> <mo>,</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>

*h*是将可解释空间的输入映射回原始输入空间的函数。为什么一致性是一个理想的属性？对于新模型，输入空间中对应更复杂特征的存在与其缺失之间的差值大于或等于旧模型的差值，无论其他特征设置如何。因此，我们应该至少在新模型的解释器中给予它与旧模型相同的权重，因为它在新模型中的重要性显然更大。

如前所述，对于每个基础模型 f，都存在一个在加性归因框架内满足所有三个属性的唯一 g。虽然我们不会在这里展示这一点，但这个结果是从合作博弈理论的早期结果中得出的，学到的权重被称为 Shapley 值。Shapley 值最初是为了量化多元线性回归模型中每个示例特征的重要性，其中个体特征具有显著的相关性。这是一个重要的问题，特别是在由于模糊性导致哪些特征最具预测性的情况下。可能情况是特征 A 与目标 y 相关，但考虑到特征 B 时，结果是特征 A 提供的附加值微不足道（即，预测没有显著变化，测试统计量保持相对恒定等）。另一方面，特征 B 可能在个体情况下和包括特征 A 的情况下都提供显著的预测能力。

确定包括两个特征的普通多元回归中特征 A 和 B 的相对重要性是困难的，因为它们之间的相关性不可忽略。Shapley 值揭示了这些关系，并计算了给定特征的真实边际影响，我们很快就会看到。以下是 Shapley 值的公式，其中 i 代表感兴趣的特征：

<math alttext="phi Subscript i Baseline equals sigma-summation Underscript upper S element-of upper F minus StartSet i EndSet Endscripts StartFraction StartAbsoluteValue upper S EndAbsoluteValue factorial asterisk left-parenthesis StartAbsoluteValue upper F EndAbsoluteValue minus StartAbsoluteValue upper S EndAbsoluteValue minus 1 right-parenthesis factorial Over StartAbsoluteValue upper F EndAbsoluteValue factorial EndFraction asterisk left-bracket f Subscript upper S union StartSet i EndSet Baseline left-parenthesis x Subscript upper S union StartSet i EndSet Baseline right-parenthesis minus f Subscript upper S Baseline left-parenthesis x Subscript upper S Baseline right-parenthesis right-bracket"><mrow><msub><mi>φ</mi> <mi>i</mi></msub> <mo>=</mo> <msub><mo>∑</mo> <mrow><mi>S</mi><mo>∈</mo><mi>F</mi><mo>∖</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></msub> <mfrac><mrow><mo>|</mo><mi>S</mi><mo>|</mo><mo>!</mo><mo>*</mo><mo>(</mo><mo>|</mo><mi>F</mi><mo>|</mo><mo>-</mo><mo>|</mo><mi>S</mi><mo>|</mo><mo>-</mo><mn>1</mn><mo>)</mo><mo>!</mo></mrow> <mrow><mo>|</mo><mi>F</mi><mo>|</mo><mo>!</mo></mrow></mfrac> <mo>*</mo> <mrow><mo>[</mo> <msub><mi>f</mi> <mrow><mi>S</mi><mo>∪</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></msub> <mrow><mo>(</mo> <msub><mi>x</mi> <mrow><mi>S</mi><mo>∪</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></msub> <mo>)</mo></mrow> <mo>-</mo> <msub><mi>f</mi> <mi>S</mi></msub> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>S</mi></msub> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

现在我们将分解这个公式。直觉上，对于一个个体特征的 Shapley 值是通过首先计算模型在一个包含特征 i 的特征子集 S 上训练的预测与在相同特征子集 S 上但不包含特征 i 的模型训练的预测之间的差异来计算的。最终的 Shapley 值是所有可能特征子集 S 的这些差异的加权和。

为了找到这些差异，我们可以首先训练一个只使用一些特征子集 S 的多元线性回归模型 f_S，然后训练第二个多元线性回归模型 f_{S∪{i}}，它使用特征子集 S∪{i}。让我们解释的示例表示为 x，x[A]表示与某个特征子集 A 对应的 x 的部分。差异 f_{S∪{i}}(x_{S∪{i}}) - f_S(x_S)表示当我们包含特征 i 时预测变化多少。另外，注意这个公式是对所有可能的特征子集求和，这意味着如果计算出的特征 i 的 Shapley 值很高，那么包含特征 i 和不包含特征 i 的差异在大多数可能的特征子集中可能是显著的。这个结果表明，无论 S 中的特征如何，特征 i 通常都会增加显著的预测能力，这由高 Shapley 值捕捉到。在前面提供的示例中，我们会发现特征 B 的 Shapley 值比特征 A 高。

此外，加权方案背后的直觉是，它实现了对特征重要性的更加无偏的结果，因为给定大小的子集在所有子集的集合中出现的频率要么更高，要么更低。给定大小的子集的数量是使用选择函数计算的，这是来自计数和概率的概念。当这被倒置并用作加权方案时，一个子集的结果，其大小在所有可能子集的集合中出现得更频繁，比如，一个仅包含除了问题中的特征之外的单个特征的特征子集，将被赋予比较少的权重。正如前面所述，我们不会完全证明为什么这是无偏的，但我们希望这能让直觉更加清晰。

您可能会注意到，对任何合理数量的特征计算精确的 Shapley 回归值是不可行的。这将涉及在所有可能的特征子集上训练回归模型，其中特征子集的数量（因此需要训练的模型）随着特征数量的增加呈指数增长。我们转而通过抽样来进行近似。给定一个要解释的示例*x*和在一些特征子集*S*上训练的回归模型*f*，我们可以通过对特征*i*的分布关于*x*的特征设置<mi>S</mi>∖{<mi>i</mi>}的期望来计算*f*：

<math alttext="f Subscript upper S minus StartSet i EndSet Baseline equals double-struck upper E Subscript p left-parenthesis x Sub Subscript i Subscript vertical-bar x Sub Subscript upper S minus StartSet i EndSet Subscript right-parenthesis Baseline left-bracket f Subscript upper S Baseline left-parenthesis x Subscript upper S minus StartSet i EndSet Baseline comma x Subscript i Baseline right-parenthesis right-bracket"><mrow><msub><mi>f</mi> <mrow><mi>S</mi><mo>∖</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></msub> <mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>p</mi><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>|</mo><msub><mi>x</mi> <mrow><mi>S</mi><mo>∖</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></msub> <mo>)</mo></mrow></msub> <mrow><mo>[</mo> <msub><mi>f</mi> <mi>S</mi></msub> <mrow><mo>(</mo> <msub><mi>x</mi> <mrow><mi>S</mi><mo>∖</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></msub> <mo>,</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

其中粗体表示<mi>x</mi><mi>S</mi>∖{<mi>i</mi>}被视为从*x*中获取的已知实体，即被解释的示例，而*x[i]*被视为未知的，即是一个随机变量而不是取自*x*提供的值。正如本书中一贯的主题，我们可以通过抽样和平均来以无偏的方式近似前述期望。

<math alttext="StartLayout 1st Row  double-struck upper E Subscript p left-parenthesis x Sub Subscript i Subscript vertical-bar x Sub Subscript upper S minus StartSet i EndSet Subscript right-parenthesis Baseline left-bracket f Subscript upper S Baseline left-parenthesis x Subscript upper S minus StartSet i EndSet Baseline comma x Subscript i Baseline right-parenthesis right-bracket 2nd Row  equals double-struck upper E Subscript p left-parenthesis x Sub Subscript i Subscript right-parenthesis Baseline left-bracket f Subscript upper S Baseline left-parenthesis x Subscript upper S minus StartSet i EndSet Baseline comma x Subscript i Baseline right-parenthesis right-bracket 3rd Row  almost-equals StartFraction 1 Over n EndFraction sigma-summation Underscript j equals 1 Overscript n Endscripts f Subscript upper S Baseline left-parenthesis x Subscript upper S minus StartSet i EndSet Baseline comma x Subscript i Superscript left-parenthesis j right-parenthesis Baseline right-parenthesis EndLayout"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><msub><mi>𝔼</mi> <mrow><mi>p</mi><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>|</mo><msub><mi>x</mi> <mrow><mi>S</mi><mo>∖</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></msub> <mo>)</mo></mrow></msub> <mrow><mo>[</mo> <msub><mi>f</mi> <mi>S</mi></msub> <mrow><mo>(</mo> <msub><mi>x</mi> <mrow><mi>S</mi><mo>∖</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></msub> <mo>,</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>p</mi><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow></msub> <mrow><mo>[</mo> <msub><mi>f</mi> <mi>S</mi></msub> <mrow><mo>(</mo> <msub><mi>x</mi> <mrow><mi>S</mi><mo>∖</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></msub> <mo>,</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mo>≈</mo> <mfrac><mn>1</mn> <mi>n</mi></mfrac> <munderover><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <msub><mi>f</mi> <mi>S</mi></msub> <mrow><mo>(</mo> <msub><mi>x</mi> <mrow><mi>S</mi><mo>∖</mo><mo>{</mo><mi>i</mi><mo>}</mo></mrow></msub> <mo>,</mo> <msubsup><mi>x</mi> <mi>i</mi> <mrow><mo>(</mo><mi>j</mi><mo>)</mo></mrow></msubsup> <mo>)</mo></mrow></mrow></mtd></mtr></mtable></math>

您可能已经注意到我们刚刚描述的过程与“部分依赖图”中描述的估计过程之间的相似之处。事实上，这些都在做同样的事情——请注意，我们再次假设特征子集<mrow><mi>S</mi> <mo>∖</mo> <mo>{</mo> <mi>i</mi> <mo>}</mo></mrow>和特征*i*之间是独立的，这使我们可以不加区分地使用数据集中特征*i*的所有样本。

作者们在一般情况下提出了 SHAP 值，其中 SHAP 值由以下公式给出：

<math alttext="phi Subscript i Baseline left-parenthesis f comma x right-parenthesis equals sigma-summation Underscript z prime subset-of-or-equal-to x Superscript prime Baseline Endscripts StartFraction StartAbsoluteValue z Superscript prime Baseline EndAbsoluteValue factorial asterisk left-parenthesis upper M minus StartAbsoluteValue z Superscript prime Baseline EndAbsoluteValue minus 1 right-parenthesis factorial Over upper M factorial EndFraction left-bracket f left-parenthesis h Subscript x Baseline left-parenthesis z prime right-parenthesis right-parenthesis minus f left-parenthesis h Subscript x Baseline left-parenthesis z prime minus StartSet i EndSet right-parenthesis right-parenthesis right-bracket"><mrow><msub><mi>φ</mi> <mi>i</mi></msub> <mrow><mo>(</mo> <mi>f</mi> <mo>,</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mo>∑</mo> <mrow><msup><mi>z</mi> <mo>'</mo></msup> <mo>⊆</mo><msup><mi>x</mi> <mo>'</mo></msup></mrow></msub> <mfrac><mrow><mrow><mo>|</mo></mrow><msup><mi>z</mi> <mo>'</mo></msup> <mrow><mo>|</mo><mo>!</mo><mo>*</mo><mo>(</mo><mi>M</mi><mo>-</mo><mo>|</mo></mrow><msup><mi>z</mi> <mo>'</mo></msup> <mrow><mo>|</mo><mo>-</mo><mn>1</mn><mo>)</mo></mrow><mo>!</mo></mrow> <mrow><mi>M</mi><mo>!</mo></mrow></mfrac> <mrow><mo>[</mo> <mi>f</mi> <mrow><mo>(</mo> <msub><mi>h</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <msup><mi>z</mi> <mo>'</mo></msup> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>-</mo> <mi>f</mi> <mrow><mo>(</mo> <msub><mi>h</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <msup><mi>z</mi> <mo>'</mo></msup> <mo>∖</mo> <mrow><mo>{</mo> <mi>i</mi> <mo>}</mo></mrow> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

其中*z'*是*x'*的非零分量的子集。此外，<math alttext="z prime minus StartSet i EndSet"><mrow><msup><mi>z</mi> <mo>'</mo></msup> <mrow><mo>∖</mo> <mrow><mo>{</mo> <mi>i</mi> <mo>}</mo></mrow></mrow></mrow></math>表示将可解释空间中的特征*i*设置为零。请注意，如果特征*i*在输入*x'*中已经为零，则该公式也会输出零，因为<math alttext="f left-parenthesis h Subscript x Baseline left-parenthesis z Superscript prime Baseline right-parenthesis right-parenthesis equals f left-parenthesis h Subscript x Baseline left-parenthesis z prime minus StartSet i EndSet right-parenthesis right-parenthesis comma for-all z Superscript prime Baseline subset-of-or-equal-to x prime"><mrow><mi>f</mi> <mrow><mo>(</mo> <msub><mi>h</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <msup><mi>z</mi> <mo>'</mo></msup> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mi>f</mi> <mrow><mo>(</mo> <msub><mi>h</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mi>â</mi> <mi></mi> <mi></mi> <mo>∖</mo> <mrow><mo>{</mo> <mi>i</mi> <mo>}</mo></mrow> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>,</mo> <mo>∀</mo> <mi>z</mi> <mi>â</mi> <mi></mi> <mi></mi> <mo>⊆</mo> <msup><mi>x</mi> <mo>'</mo></msup></mrow></math>。这个快速检查表明该公式确实满足缺失性的属性。由 SHAP 值组成的向量完全定义了加法归因框架中每个特征的最佳解释模型*g*，其中最佳表示*g*满足前面定义的三个属性：局部准确性、一致性和缺失性。我们可以立即看到提出的 SHAP 值与多元回归中的 Shapley 值之间的相似之处。此外，我们可以使用相同的抽样过程来估计 SHAP 值。

正如讨论的那样，LIME 处于加法归因框架中。在原始的 LIME 论文中，通过专门的优化过程选择了最佳的解释模型*g*，该过程首先选择了*k*个特征具有非零贡献，然后执行标准的最小二乘优化以获得*g*的最终权重。由于这些启发式方法，包括核选择<math alttext="pi Subscript x Baseline left-parenthesis z right-parenthesis"><mrow><msub><mi>π</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow></mrow></math>，无法保证使用原始 LIME 论文中提出的程序选择的解释符合 SHAP 的局部准确性、缺失性和一致性标准。

然而，LIME 论文中提出的优化过程确实可以获得一个满足 LIME 中提出的解释模型标准的解释器；回顾前一节中关于可解释性、局部忠实度、模型无关性和全局视角的概念。我们特别指出这一点是为了表明不同的知识渊博的个体并不一定对于解释器何为可解释性有完全相同的理解，而且可解释性作为一个概念随着时间的推移而发展。

事实证明，在 LIME 中存在一种确切的形式来衡量接近度<math alttext="pi Subscript x Baseline left-parenthesis z right-parenthesis"><mrow><msub><mi>π</mi> <mi>x</mi></msub> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow></mrow></math>，<math alttext="omega"><mi>ω</mi></math>，和损失函数*L*，当最小化时，会得到一个满足解释性的最佳解释器*g*，满足解释性的三个 SHAP 标准：

<math alttext="StartLayout 1st Row  omega left-parenthesis g right-parenthesis equals 0 2nd Row  pi Subscript x prime Baseline left-parenthesis z prime right-parenthesis equals StartFraction upper M minus 1 Over left-parenthesis upper M choose StartAbsoluteValue z Superscript prime Baseline EndAbsoluteValue right-parenthesis asterisk StartAbsoluteValue z prime EndAbsoluteValue asterisk left-parenthesis upper M minus StartAbsoluteValue z prime EndAbsoluteValue right-parenthesis EndFraction 3rd Row  upper L left-parenthesis f comma g comma pi right-parenthesis equals sigma-summation Underscript z prime element-of upper Z Endscripts left-parenthesis f left-parenthesis h Subscript x Baseline left-parenthesis z prime right-parenthesis right-parenthesis minus g left-parenthesis z prime right-parenthesis right-parenthesis squared asterisk pi Subscript x prime Baseline left-parenthesis z prime right-parenthesis EndLayout"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mi>ω</mi> <mo>(</mo> <mi>g</mi> <mo>)</mo> <mo>=</mo> <mn>0</mn></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><msub><mi>π</mi> <msup><mi>x</mi> <mo>'</mo></msup></msub> <mrow><mo>(</mo> <msup><mi>z</mi> <mo>'</mo></msup> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>M</mi><mo>-</mo><mn>1</mn></mrow> <mrow><mrow><mo>(</mo><mi>M</mi><mtext>choose</mtext><mo>|</mo></mrow><msup><mi>z</mi> <mo>'</mo></msup> <mrow><mo>|</mo><mo>)</mo><mo>*</mo><mo>|</mo></mrow><msup><mi>z</mi> <mo>'</mo></msup> <mrow><mo>|</mo><mo>*</mo><mo>(</mo><mi>M</mi><mo>-</mo><mo>|</mo></mrow><msup><mi>z</mi> <mo>'</mo></msup> <mrow><mo>|</mo><mo>)</mo></mrow></mrow></mfrac></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mi>L</mi> <mrow><mo>(</mo> <mi>f</mi> <mo>,</mo> <mi>g</mi> <mo>,</mo> <mi>π</mi> <mo>)</mo></mrow> <mo>=</mo> <munder><mo>∑</mo> <mrow><mi>z</mi><mi>â</mi><mi></mi><mi></mi><mo>∈</mo><mi>Z</mi></mrow></munder> <msup><mrow><mo>(</mo><mi>f</mi><mrow><mo>(</mo><msub><mi>h</mi> <mi>x</mi></msub> <mrow><mo>(</mo><msup><mi>z</mi> <mo>'</mo></msup> <mo>)</mo></mrow><mo>)</mo></mrow><mo>-</mo><mi>g</mi><mrow><mo>(</mo><msup><mi>z</mi> <mo>'</mo></msup> <mo>)</mo></mrow><mo>)</mo></mrow> <mn>2</mn></msup> <mo>*</mo> <msub><mi>π</mi> <msup><mi>x</mi> <mo>'</mo></msup></msub> <mrow><mo>(</mo> <msup><mi>z</mi> <mo>'</mo></msup> <mo>)</mo></mrow></mrow></mtd></mtr></mtable></math>

我们可以使用加权最小二乘优化来优化这个损失函数，以获得唯一的最优*g*。请注意，这里的核函数在解释上与原始 LIME 论文中提出的核函数选择是不同的。SHAP 核函数是对称的，而不是像 LIME 中的核函数随着样本远离被解释的示例而减小。这可以通过检查核函数在*|z'| = k*和*|z'| = M – k*时的输出来验证。实际上，仅仅通过观察公式，我们就可以看到核函数的值甚至不依赖于*x'*。

总之，SHAP 值通过首先定义加法归因框架，将几种现有的可解释性方法统一起来，其次通过证明在这个框架内存在一个满足三个理想属性的唯一最优解释器的存在。

# 摘要

尽管可解释性通常以各种形式出现，但它们都旨在能够解释模型行为。我们了解到，并非每个模型都是通过构建可解释的，即使是那些可能表面上看起来是可解释的。例如，尽管普通的线性回归似乎在设计上是相当可解释的，但特征之间的相关性可能会混淆这个最初清晰的图像。此外，我们了解到模型本身内置的可解释性方法，如抽取式合理化，以及事后可解释性方法，如 LIME 和 SHAP。正确的可解释性形式通常取决于领域——例如，在图像分类中使用基于梯度的方法可能是有意义的，但在语言问题中可能不那么合适。在先前章节讨论的软注意力方案对于情感分析可能不如我们在抽取式合理化部分提出的硬选择方法那么理想。最后，我们了解到可解释性在研究中甚至在整个领域中并不具有完全相同的含义——请注意我们对 LIME 和 SHAP 生成的最优解释器之间差异的讨论。我们希望这一章对广阔的可解释性研究领域起到了富有成效的探索作用。

¹ 雷等人。“理性化神经预测。”*arXiv Preprint arXiv*:1606.04155\. 2016 年。

² 里贝罗等人。“我为什么应该相信你？解释任何分类器的预测。”*arXiv Preprint arXiv*:1602.04938\. 2016 年。

³ 兰德伯格等人。“解释模型预测的统一方法。”*arXiv Preprint arXiv*:1705.07874\. 2017 年。
