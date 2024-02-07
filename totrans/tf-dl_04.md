# 第四章。全连接的深度网络

本章将向您介绍全连接的深度网络。全连接网络是深度学习的主力军，用于成千上万的应用。全连接网络的主要优势在于它们是“结构不可知的”。也就是说，不需要对输入做出特殊的假设（例如，输入由图像或视频组成）。我们将利用这种通用性，使用全连接的深度网络来解决本章后面的化学建模问题。

我们简要探讨支撑全连接网络的数学理论。特别是，我们探讨全连接架构是“通用逼近器”，能够学习任何函数的概念。这个概念解释了全连接架构的通用性，但也伴随着我们深入讨论的许多注意事项。

虽然结构不可知使全连接网络非常广泛适用，但这种网络的性能往往比针对问题空间结构调整的专用网络要弱。我们将在本章后面讨论全连接架构的一些限制。

# 什么是全连接的深度网络？

全连接神经网络由一系列全连接层组成。全连接层是从<math alttext="double-struck upper R Superscript m"><msup><mi>ℝ</mi> <mi>m</mi></msup></math>到<math alttext="double-struck upper R Superscript n"><msup><mi>ℝ</mi> <mi>n</mi></msup></math>的函数。每个输出维度都依赖于每个输入维度。在图 4-1 中，全连接层的图示如下。

![FCLayer.png](img/tfdl_0401.png)

###### 图 4-1. 深度网络中的全连接层。

让我们更深入地了解全连接网络的数学形式。让<math><mrow><mi>x</mi> <mo>∈</mo> <msup><mi>ℝ</mi> <mi>m</mi></msup></mrow></math>表示全连接层的输入。让<math alttext="y Subscript i Baseline element-of double-struck upper R"><mrow><msub><mi>y</mi> <mi>i</mi></msub> <mo>∈</mo> <mi>ℝ</mi></mrow></math>是全连接层的第 i 个输出。那么<math><mrow><msub><mi>y</mi> <mi>i</mi></msub> <mo>∈</mo> <mi>ℝ</mi></mrow></math>的计算如下：

<math display="block"><mrow><msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo> <mi>σ</mi> <mrow><mo>(</mo> <msub><mi>w</mi> <mn>1</mn></msub> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <mo>⋯</mo> <mo>+</mo> <msub><mi>w</mi> <mi>m</mi></msub> <msub><mi>x</mi> <mi>m</mi></msub> <mo>)</mo></mrow></mrow></math>

在这里，<math alttext="sigma"><mi>σ</mi></math> 是一个非线性函数（暂时将<math><mi>σ</mi></math>视为前一章介绍的 Sigmoid 函数），<math alttext="w Subscript i"><msub><mi>w</mi> <mi>i</mi></msub></math> 是网络中可学习的参数。完整的输出*y*如下：

<math display="block"><mrow><mi>y</mi> <mo>=</mo> <mfenced open="(" close=")"><mtable><mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <msub><mi>w</mi> <mrow><mn>1</mn><mo>,</mo><mn>1</mn></mrow></msub> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <mo>⋯</mo> <mo>+</mo> <msub><mi>w</mi> <mrow><mn>1</mn><mo>,</mo><mi>m</mi></mrow></msub> <msub><mi>x</mi> <mi>m</mi></msub> <mo>)</mo></mrow></mtd></mtr> <mtr><mtd><mo>⋮</mo></mtd></mtr> <mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <msub><mi>w</mi> <mrow><mi>n</mi><mo>,</mo><mn>1</mn></mrow></msub> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <mo>⋯</mo> <mo>+</mo> <msub><mi>w</mi> <mrow><mi>n</mi><mo>,</mo><mi>m</mi></mrow></msub> <msub><mi>x</mi> <mi>m</mi></msub> <mo>)</mo></mrow></mtd></mtr></mtable></mfenced></mrow></math>

请注意，可以直接堆叠全连接网络。具有多个全连接网络的网络通常被称为“深度”网络，如图 4-2 所示。

![multilayer_fcnet.png](img/tfdl_0402.png)

###### 图 4-2。一个多层深度全连接网络。

作为一个快速实现的注意事项，注意单个神经元的方程看起来非常类似于两个向量的点积（回想一下张量基础的讨论）。对于一层神经元，通常为了效率目的，将*y*计算为矩阵乘积是很方便的：

<math display="block"><mrow><mi>y</mi> <mo>=</mo> <mi>σ</mi> <mo>(</mo> <mi>w</mi> <mi>x</mi> <mo>)</mo></mrow></math>

其中 sigma 是一个矩阵在<math alttext="双击上 R 上标 n 次 m"><msup><mi>ℝ</mi> <mrow><mi>n</mi><mo>×</mo><mi>m</mi></mrow></msup></math>，非线性<math alttext="sigma"><mi>σ</mi></math>是逐分量应用的。

# 全连接网络中的“神经元”

全连接网络中的节点通常被称为“神经元”。因此，在文献中，全连接网络通常被称为“神经网络”。这种命名方式在很大程度上是历史的偶然。

在 1940 年代，沃伦·S·麦卡洛克和沃尔特·皮茨发表了一篇关于大脑的第一个数学模型，认为神经元能够计算布尔量上的任意函数。这项工作的后继者稍微完善了这个逻辑模型，通过使数学“神经元”成为在零和一之间变化的连续函数。如果这些函数的输入足够大，神经元就会“发射”（取值为一），否则就是静止的。通过可调权重的添加，这个描述与之前的方程匹配。

这才是真正的神经元行为吗？当然不是！一个真实的神经元（图 4-3）是一个极其复杂的引擎，拥有超过 100 万亿个原子，以及数以万计的不同信号蛋白质，能够对不同信号做出反应。微处理器比一个一行方程更好地类比于神经元。

![neuron.png](img/tfdl_0403.png)

###### 图 4-3。神经元的更生物学准确的表示。

在许多方面，生物神经元和人工神经元之间的这种脱节是非常不幸的。未经培训的专家读到令人激动的新闻稿，声称已经创建了拥有数十亿“神经元”的人工神经网络（而大脑只有 1000 亿个生物神经元），并且合理地认为科学家们已经接近创造人类水平的智能。不用说，深度学习的最新技术距离这样的成就还有几十年（甚至几个世纪）的距离。

当您进一步了解深度学习时，您可能会遇到关于人工智能的夸大宣传。不要害怕指出这些声明。目前的深度学习是一套在快速硬件上解决微积分问题的技术。它不是*终结者*的前身（图 4-4）。

![terminator.png](img/tfdl_0404.png)

###### 图 4-4。不幸的是（或者也许是幸运的），这本书不会教你如何构建一个终结者！

# AI 寒冬

人工智能经历了多轮繁荣和衰退的发展。这种循环性的发展是该领域的特点。每一次学习的新进展都会引发一波乐观情绪，其中预言家声称人类水平（或超人类）的智能即将出现。几年后，没有这样的智能体现出来，失望的资助者退出。由此产生的时期被称为 AI 寒冬。

迄今为止已经有多次 AI 寒冬。作为一种思考练习，我们鼓励您考虑下一次 AI 寒冬将在何时发生。当前的深度学习进展解决了比以往任何一波进步更多的实际问题。AI 是否可能最终脱颖而出，摆脱繁荣和衰退的周期，或者您认为我们很快就会迎来 AI 的“大萧条”？

## 使用反向传播学习全连接网络

完全连接的神经网络的第一个版本是感知器（图 4-5），由 Frank Rosenblatt 在 1950 年代创建。这些感知器与我们在前面的方程中介绍的“神经元”是相同的。

![perceptron.png](img/tfdl_0405.png)

###### 图 4-5。感知器的示意图。

感知器是通过自定义的“感知器”规则进行训练的。虽然它们在解决简单问题时有一定用处，但感知器在根本上受到限制。1960 年代末 Marvin Minsky 和 Seymour Papert 的书《感知器》证明了简单感知器无法学习 XOR 函数。图 4-6 说明了这个说法的证明。

![xor2.gif](img/tfdl_0406.png)

###### 图 4-6。感知器的线性规则无法学习感知器。

这个问题通过多层感知器（另一个称为深度全连接网络的名称）的发明得以解决。这一发明是一个巨大的成就，因为早期的简单学习算法无法有效地学习深度网络。 “信用分配”问题困扰着它们；算法如何决定哪个神经元学习什么？

解决这个问题的完整方法需要反向传播。反向传播是学习神经网络权重的通用规则。不幸的是，关于反向传播的复杂解释在文献中泛滥。这种情况很不幸，因为反向传播只是自动微分的另一个说法。

假设<math alttext="f left-parenthesis theta comma x right-parenthesis"><mrow><mi>f</mi> <mo>(</mo> <mi>θ</mi> <mo>,</mo> <mi>x</mi> <mo>)</mo></mrow></math>是代表深度全连接网络的函数。这里<math alttext="x"><mi>x</mi></math>是完全连接网络的输入，<math alttext="theta"><mi>θ</mi></math>是可学习的权重。然后，反向传播算法简单地计算<math alttext="StartFraction normal partial-differential f Over normal partial-differential theta EndFraction"><mfrac><mrow><mi>∂</mi><mi>f</mi></mrow> <mrow><mi>∂</mi><mi>θ</mi></mrow></mfrac></math>。在实践中，实现反向传播以处理所有可能出现的*f*函数的复杂性。幸运的是，TensorFlow 已经为我们处理了这一点！

## 通用收敛定理

前面的讨论涉及到了深度全连接网络是强大逼近的想法。McCulloch 和 Pitts 表明逻辑网络可以编码（几乎）任何布尔函数。Rosenblatt 的感知器是 McCulloch 和 Pitt 的逻辑函数的连续模拟，但被 Minsky 和 Papert 证明在根本上受到限制。多层感知器试图解决简单感知器的限制，并且在经验上似乎能够学习复杂函数。然而，从理论上讲，尚不清楚这种经验能力是否存在未被发现的限制。 1989 年，George Cybenko 证明了多层感知器能够表示任意函数。这一演示为全连接网络作为学习架构的普遍性主张提供了相当大的支持，部分解释了它们持续受欢迎的原因。

然而，如果在上世纪 80 年代后期人们已经理解了反向传播和全连接网络理论，为什么“深度”学习没有更早变得更受欢迎呢？这种失败的很大一部分是由于计算能力的限制；学习全连接网络需要大量的计算能力。此外，由于对好的超参数缺乏理解，深度网络非常难以训练。因此，计算要求较低的替代学习算法，如 SVM，变得更受欢迎。深度学习近年来的流行部分原因是更好的计算硬件的增加可用性，使计算速度更快，另一部分原因是对能够实现稳定学习的良好训练方案的增加理解。

# 通用逼近是否令人惊讶？

通用逼近性质在数学中比人们可能期望的更常见。例如，Stone-Weierstrass 定理证明了在闭区间上的任何连续函数都可以是一个合适的多项式函数。进一步放宽我们的标准，泰勒级数和傅里叶级数本身提供了一些通用逼近能力（在它们的收敛域内）。通用收敛在数学中相当常见的事实部分地为经验观察提供了部分理由，即许多略有不同的全连接网络变体似乎具有通用逼近性质。

# 通用逼近并不意味着通用学习！

通用逼近定理中存在一个关键的微妙之处。全连接网络可以表示任何函数并不意味着反向传播可以学习任何函数！反向传播的一个主要限制是没有保证全连接网络“收敛”；也就是说，找到学习问题的最佳可用解决方案。这个关键的理论差距让几代计算机科学家对神经网络感到不安。即使在今天，许多学者仍然更愿意使用具有更强理论保证的替代算法。

经验研究已经产生了许多实用技巧，使反向传播能够为问题找到好的解决方案。在本章的其余部分中，我们将深入探讨许多这些技巧。对于实践数据科学家来说，通用逼近定理并不是什么需要太认真对待的东西。这是令人放心的，但深度学习的艺术在于掌握使学习有效的实用技巧。

## 为什么要使用深度网络？

通用逼近定理中的一个微妙之处是，事实上它对只有一个全连接层的全连接网络也成立。那么，具有多个全连接层的“深度”学习有什么用呢？事实证明，这个问题在学术和实践领域仍然颇具争议。

在实践中，似乎更深层的网络有时可以在大型数据集上学习更丰富的模型。（然而，这只是一个经验法则；每个实践者都有许多例子，深度全连接网络表现不佳。）这一观察结果导致研究人员假设更深层的网络可以更有效地表示复杂函数。也就是说，相比具有相同数量的神经元的较浅网络，更深的网络可能能够学习更复杂的函数。例如，在第一章中简要提到的 ResNet 架构，具有 130 层，似乎胜过其较浅的竞争对手，如 AlexNet。一般来说，对于固定的神经元预算，堆叠更深层次会产生更好的结果。

文献中提出了一些关于深度网络优势的错误“证明”，但它们都有漏洞。深度与宽度的问题似乎涉及到复杂性理论中的深刻概念（研究解决给定计算问题所需的最小资源量）。目前看来，理论上证明（或否定）深度网络的优越性远远超出了我们数学家的能力范围。

# 训练全连接神经网络

正如我们之前提到的，全连接网络的理论与实践有所不同。在本节中，我们将向您介绍一些关于全连接网络的经验观察，这些观察有助于从业者。我们强烈建议您使用我们的代码（在本章后面介绍）来验证我们的说法。

## 可学习表示

一种思考全连接网络的方式是，每个全连接层都会对问题所在的特征空间进行转换。在工程和物理学中，将问题的表示转换为更易处理的形式的想法是非常古老的。因此，深度学习方法有时被称为“表示学习”。（有趣的事实是，深度学习的一个主要会议被称为“国际学习表示会议”。）

几代分析师已经使用傅立叶变换、勒让德变换、拉普拉斯变换等方法，将复杂的方程和函数简化为更适合手工分析的形式。一种思考深度学习网络的方式是，它们实现了一个适合手头问题的数据驱动转换。

执行特定于问题的转换能力可能非常强大。标准的转换技术无法解决图像或语音分析的问题，而深度网络能够相对轻松地解决这些问题，这是由于学习表示的固有灵活性。这种灵活性是有代价的：深度架构学习到的转换通常比傅立叶变换等数学变换要不那么通用。尽管如此，将深度变换纳入分析工具包中可以成为一个强大的问题解决工具。

有一个合理的观点认为，深度学习只是第一个有效的表示学习方法。将来，可能会有替代的表示学习方法取代深度学习方法。

## 激活函数

我们之前介绍了非线性函数<math alttext="sigma"><mi>σ</mi></math>作为 S 形函数。虽然 S 形函数是全连接网络中的经典非线性，但近年来研究人员发现其他激活函数，特别是修正线性激活（通常缩写为 ReLU 或 relu）<math alttext="sigma left-parenthesis x right-parenthesis equals max left-parenthesis x comma 0 right-parenthesis"><mrow><mi>σ</mi> <mo>(</mo> <mi>x</mi> <mo>)</mo> <mo>=</mo> <mi>max</mi> <mo>(</mo> <mi>x</mi> <mo>,</mo> <mn>0</mn> <mo>)</mo></mrow></math>比 S 形单元效果更好。这种经验观察可能是由于深度网络中的*梯度消失*问题。对于 S 形函数，几乎所有输入值的斜率都为零。因此，对于更深的网络，梯度会趋近于零。对于 ReLU 函数，输入空间的大部分部分斜率都不为零，允许非零梯度传播。图 4-7 展示了 S 形和 ReLU 激活函数并排的情况。

![activation-functions.png](img/tfdl_0407.png)

###### 图 4-7。S 形和 ReLU 激活函数。

## 全连接网络记忆

全连接网络的一个显著特点是，给定足够的时间，它们倾向于完全记住训练数据。因此，将全连接网络训练到“收敛”实际上并不是一个有意义的度量。只要用户愿意等待，网络将继续训练和学习。

对于足够大的网络，训练损失趋向于零是非常常见的。这一经验观察是全连接网络的通用逼近能力最实用的证明之一。然而，请注意，训练损失趋向于零并不意味着网络已经学会了一个更强大的模型。相反，模型很可能已经开始记忆训练集的怪癖，这些怪癖并不适用于任何其他数据点。

值得深入探讨这里我们所说的奇特之处。高维统计学的一个有趣特性是，给定足够大的数据集，将有大量的虚假相关性和模式可供选择。在实践中，全连接网络完全有能力找到并利用这些虚假相关性。控制网络并防止它们以这种方式行为不端对于建模成功至关重要。

## 正则化

正则化是一个通用的统计术语，用于限制记忆化，同时促进可泛化的学习。有许多不同类型的正则化可用，我们将在接下来的几节中介绍。

# 不是您的统计学家的正则化

正则化在统计文献中有着悠久的历史，有许多关于这个主题的论文。不幸的是，只有一部分经典分析适用于深度网络。在统计学中广泛使用的线性模型可能与深度网络表现出截然不同，而在那种情况下建立的许多直觉对于深度网络来说可能是错误的。

与深度网络一起工作的第一条规则，特别是对于具有先前统计建模经验的读者，是相信经验结果胜过过去的直觉。不要假设对于建模深度架构等技术的过去知识有太多意义。相反，建立一个实验来系统地测试您提出的想法。我们将在下一章更深入地讨论这种系统化实验过程。

### Dropout

Dropout 是一种正则化形式，它随机地删除一些输入到全连接层的节点的比例（图 4-8）。在这里，删除一个节点意味着其对应激活函数的贡献被设置为 0。由于没有激活贡献，被删除节点的梯度也降为零。

![dropout.png](img/tfdl_0408.png)

###### 图 4-8。在训练时，Dropout 随机删除网络中的神经元。从经验上看，这种技术通常为网络训练提供强大的正则化。

要删除的节点是在梯度下降的每一步中随机选择的。底层的设计原则是网络将被迫避免“共适应”。简而言之，我们将解释什么是共适应以及它如何在非正则化的深度架构中出现。假设深度网络中的一个神经元学习了一个有用的表示。那么网络中更深层的其他神经元将迅速学会依赖于该特定神经元获取信息。这个过程将使网络变得脆弱，因为网络将过度依赖于该神经元学到的特征，而这些特征可能代表数据集的一个怪癖，而不是学习一个普遍规则。

Dropout 可以防止这种协同适应，因为不再可能依赖于单个强大的神经元的存在（因为该神经元在训练期间可能会随机消失）。因此，其他神经元将被迫“弥补空缺”并学习到有用的表示。理论上的论点是，这个过程应该会产生更强大的学习模型。

在实践中，dropout 有一对经验效果。首先，它防止网络记忆训练数据；使用 dropout 后，即使对于非常大的深度网络，训练损失也不会迅速趋向于 0。其次，dropout 倾向于略微提升模型对新数据的预测能力。这种效果通常适用于各种数据集，这也是 dropout 被认为是一种强大的发明而不仅仅是一个简单的统计技巧的部分原因。

你应该注意，在进行预测时应关闭 dropout。忘记关闭 dropout 可能导致预测比原本更加嘈杂和无用。我们将在本章后面正确讨论如何处理训练和预测中的 dropout。

# 大型网络如何避免过拟合？

对于传统训练有素的统计学家来说，最令人震惊的一点是，深度网络可能经常具有比训练数据中存在的内部自由度更多的内部自由度。在传统统计学中，这些额外的自由度的存在会使模型变得无用，因为不再存在一个保证模型学到的是“真实”的经典意义上的保证。

那么，一个拥有数百万参数的深度网络如何能够在只有数千个示例的数据集上学习到有意义的结果呢？Dropout 可以在这里起到很大的作用，防止蛮力记忆。但是，即使没有使用 dropout，深度网络也会倾向于学习到有用的事实，这种倾向可能是由于反向传播或全连接网络结构的某种特殊性质，我们尚不理解。

### 早停止

正如前面提到的，全连接网络往往会记住放在它们面前的任何东西。因此，在实践中，跟踪网络在一个保留的“验证”集上的表现，并在该验证集上的表现开始下降时停止网络，通常是很有用的。这种简单的技术被称为早停止。

在实践中，早停止可能会很棘手。正如你将看到的，深度网络的损失曲线在正常训练过程中可能会有很大的变化。制定一个能够区分健康变化和明显下降趋势的规则可能需要很大的努力。在实践中，许多从业者只是训练具有不同（固定）时代数量的模型，并选择在验证集上表现最好的模型。图 4-9 展示了训练和测试集准确率随着训练进行而通常变化的情况。

![earlystopping.png](img/tfdl_0409.png)

###### 图 4-9. 训练和测试集的模型准确率随着训练进行而变化。

我们将在接下来的章节中更深入地探讨与验证集一起工作的正确方法。

### 权重正则化

从统计学文献中借鉴的一种经典正则化技术惩罚那些权重增长较大的学习权重。根据前一章的符号表示，让<math alttext="script upper L left-parenthesis x comma y right-parenthesis"><mrow><mi>ℒ</mi> <mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow></math>表示特定模型的损失函数，让<math><mi>θ</mi></math>表示该模型的可学习参数。那么正则化的损失函数定义如下

<math display="block"><mrow><msup><mi>ℒ</mi> <mo>'</mo></msup> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>ℒ</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>+</mo> <mi>α</mi> <mrow><mo>∥</mo> <mi>θ</mi> <mo>∥</mo></mrow></mrow></math>

其中<math alttext="parallel-to theta parallel-to"><mrow><mo>∥</mo> <mi>θ</mi> <mo>∥</mo></mrow></math>是权重惩罚，<math alttext="alpha"><mi>α</mi></math>是一个可调参数。惩罚的两种常见选择是*L*¹和*L*²惩罚

<math display="block"><mrow><msub><mrow><mo>∥</mo><mi>θ</mi><mo>∥</mo></mrow> <mn>2</mn></msub> <mo>=</mo> <msqrt><mrow><msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>N</mi></msubsup> <msubsup><mi>θ</mi> <mi>i</mi> <mn>2</mn></msubsup></mrow></msqrt></mrow></math><math display="block"><mrow><msub><mrow><mo>∥</mo><mi>θ</mi><mo>∥</mo></mrow> <mn>1</mn></msub> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>N</mi></munderover> <mrow><mo>|</mo> <msub><mi>θ</mi> <mi>i</mi></msub> <mo>|</mo></mrow></mrow></math>

其中<math alttext="parallel-to theta parallel-to"><msub><mrow><mo>∥</mo><mi>θ</mi><mo>∥</mo></mrow> <mn>2</mn></msub></math>和<math alttext="parallel-to theta parallel-to"><msub><mrow><mo>∥</mo><mi>θ</mi><mo>∥</mo></mrow> <mn>1</mn></msub></math>分别表示*L*¹和*L*²的惩罚。从个人经验来看，这些惩罚对于深度模型来说往往不如 dropout 和早停止有用。一些从业者仍然使用权重正则化，因此值得了解如何在调整深度网络时应用这些惩罚。

## 训练全连接网络

训练全连接网络需要一些技巧，超出了您在本书中迄今为止看到的内容。首先，与之前的章节不同，我们将在更大的数据集上训练模型。对于这些数据集，我们将向您展示如何使用 minibatches 来加速梯度下降。其次，我们将回到调整学习率的话题。

### Minibatching

对于大型数据集（甚至可能无法完全装入内存），在每一步计算梯度时无法在整个数据集上进行。相反，从业者通常选择一小部分数据（通常是 50-500 个数据点）并在这些数据点上计算梯度。这小部分数据传统上被称为一个 minibatch。

在实践中，minibatching 似乎有助于收敛，因为可以在相同的计算量下进行更多的梯度下降步骤。minibatch 的正确大小是一个经验性问题，通常通过超参数调整来设置。

### 学习率

学习率决定了每个梯度下降步骤的重要性。设置正确的学习率可能会有些棘手。许多初学者设置学习率不正确，然后惊讶地发现他们的模型无法学习或开始返回 NaN。随着 ADAM 等方法的发展，这种情况已经得到了显著改善，但如果模型没有学到任何东西，调整学习率仍然是值得的。

# 在 TensorFlow 中的实现

在这一部分中，我们将向您展示如何在 TensorFlow 中实现一个全连接网络。在这一部分中，我们不需要引入太多新的 TensorFlow 原语，因为我们已经涵盖了大部分所需的基础知识。

## 安装 DeepChem

在这一部分中，您将使用 DeepChem 机器学习工具链进行实验（完整披露：其中一位作者是 DeepChem 的创始人）。有关 DeepChem 的[详细安装说明](https://deepchem.io)可以在线找到，但简要地说，通过`conda`工具进行的 Anaconda 安装可能是最方便的。

## Tox21 数据集

在我们的建模案例研究中，我们将使用一个化学数据集。毒理学家对使用机器学习来预测给定化合物是否有毒非常感兴趣。这个任务非常复杂，因为当今的科学只对人体内发生的代谢过程有有限的了解。然而，生物学家和化学家已经研究出一套有限的实验，可以提供毒性的指示。如果一个化合物在这些实验中是“命中”的，那么人类摄入后可能会有毒。然而，这些实验通常成本很高，因此数据科学家旨在构建能够预测这些实验结果的机器学习模型，用于新分子。

最重要的毒理学数据集之一称为 Tox21。它由 NIH 和 EPA 发布，作为数据科学倡议的一部分，并被用作模型构建挑战中的数据集。这个挑战的获胜者使用了多任务全连接网络（全连接网络的一种变体，其中每个网络为每个数据点预测多个数量）。我们将分析来自 Tox21 集合中的一个数据集。该数据集包含一组经过测试与雄激素受体相互作用的 10,000 种分子。数据科学挑战是预测新分子是否会与雄激素受体相互作用。

处理这个数据集可能有些棘手，因此我们将利用 DeepChem 部分作为 MoleculeNet 数据集收集。DeepChem 将 Tox21 中的每个分子处理为长度为 1024 的比特向量。然后加载数据集只需几个简单的调用到 DeepChem 中（示例 4-1）。

##### 示例 4-1\. 加载 Tox21 数据集

```py
import deepchem as dc

_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w
```

这里的 `X` 变量保存处理过的特征向量，`y` 保存标签，`w` 保存示例权重。标签是与雄激素受体相互作用或不相互作用的化合物的二进制 1/0。Tox21 拥有*不平衡*数据集，其中正例远远少于负例。`w` 保存建议的每个示例权重，给予正例更多的重视（增加罕见示例的重要性是处理不平衡数据集的常见技术）。为简单起见，我们在训练过程中不使用这些权重。所有这些变量都是 NumPy 数组。

Tox21 拥有比我们这里将要分析的更多数据集，因此我们需要删除与这些额外数据集相关联的标签（示例 4-2）。

##### 示例 4-2\. 从 Tox21 中删除额外的数据集

```py
# Remove extra tasks
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]
```

## 接受占位符的小批量

在之前的章节中，我们创建了接受固定大小参数的占位符。在处理小批量数据时，能够输入不同大小的批次通常很方便。假设一个数据集有 947 个元素。那么以小批量大小为 50，最后一个批次将有 47 个元素。这将导致 第三章 中的代码崩溃。幸运的是，TensorFlow 对这种情况有一个简单的解决方法：使用 `None` 作为占位符的维度参数允许占位符在该维度上接受任意大小的张量（示例 4-3）。

##### 示例 4-3\. 定义接受不同大小小批量的占位符

```py
d = 1024
with tf.name_scope("placeholders"):
  x = tf.placeholder(tf.float32, (None, d))
  y = tf.placeholder(tf.float32, (None,))
```

注意 `d` 是 1024，即我们特征向量的维度。

## 实现隐藏层

实现隐藏层的代码与我们在上一章中看到的用于实现逻辑回归的代码非常相似，如 示例 4-4 所示。

##### 示例 4-4\. 定义一个隐藏层

```py
with tf.name_scope("hidden-layer"):
  W = tf.Variable(tf.random_normal((d, n_hidden)))
  b = tf.Variable(tf.random_normal((n_hidden,)))
  x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
```

我们使用 `tf.name_scope` 将引入的变量分组在一起。请注意，我们使用全连接层的矩阵形式。我们使用形式 *xW* 而不是 *Wx*，以便更方便地处理一次输入的小批量。（作为练习，尝试计算涉及的维度，看看为什么会这样。）最后，我们使用内置的 `tf.nn.relu` 激活函数应用 ReLU 非线性。

完全连接层的其余代码与上一章中用于逻辑回归的代码非常相似。为了完整起见，我们展示了用于指定网络的完整代码在例 4-5 中使用。作为一个快速提醒，所有模型的完整代码都可以在与本书相关的 GitHub 存储库中找到。我们强烈建议您尝试运行代码。

##### 例 4-5。定义完全连接的架构

```py
with tf.name_scope("placeholders"):
  x = tf.placeholder(tf.float32, (None, d))
  y = tf.placeholder(tf.float32, (None,))
with tf.name_scope("hidden-layer"):
  W = tf.Variable(tf.random_normal((d, n_hidden)))
  b = tf.Variable(tf.random_normal((n_hidden,)))
  x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
with tf.name_scope("output"):
  W = tf.Variable(tf.random_normal((n_hidden, 1)))
  b = tf.Variable(tf.random_normal((1,)))
  y_logit = tf.matmul(x_hidden, W) + b
  # the sigmoid gives the class probability of 1
  y_one_prob = tf.sigmoid(y_logit)
  # Rounding P(y=1) will give the correct prediction.
  y_pred = tf.round(y_one_prob)
with tf.name_scope("loss"):
  # Compute the cross-entropy term for each datapoint
  y_expand = tf.expand_dims(y, 1)
  entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
  # Sum all contributions
  l = tf.reduce_sum(entropy)

with tf.name_scope("optim"):
  train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)

with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()
```

## 向隐藏层添加 dropout

TensorFlow 负责为我们实现 dropout，内置原语`tf.nn.dropout(x, keep_prob)`，其中`keep_prob`是保留任何给定节点的概率。回想一下我们之前的讨论，我们希望在训练时打开 dropout，在进行预测时关闭 dropout。为了正确处理这一点，我们将引入一个新的占位符`keep_prob`，如例 4-6 所示。

##### 例 4-6。为丢失概率添加一个占位符

```py
keep_prob = tf.placeholder(tf.float32)
```

在训练期间，我们传入所需的值，通常为 0.5，但在测试时，我们将`keep_prob`设置为 1.0，因为我们希望使用所有学习节点进行预测。通过这种设置，在前一节中指定的完全连接网络中添加 dropout 只是一行额外的代码（例 4-7）。

##### 例 4-7。定义一个带有 dropout 的隐藏层

```py
with tf.name_scope("hidden-layer"):
  W = tf.Variable(tf.random_normal((d, n_hidden)))
  b = tf.Variable(tf.random_normal((n_hidden,)))
  x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
  # Apply dropout
  x_hidden = tf.nn.dropout(x_hidden, keep_prob)
```

## 实现小批量处理

为了实现小批量处理，我们需要在每次调用`sess.run`时提取一个小批量的数据。幸运的是，我们的特征和标签已经是 NumPy 数组，我们可以利用 NumPy 对数组的方便语法来切片数组的部分（例 4-8）。

##### 例 4-8。在小批量上进行训练

```py
step = 0
for epoch in range(n_epochs):
  pos = 0
  while pos < N:
    batch_X = train_X[pos:pos+batch_size]
    batch_y = train_y[pos:pos+batch_size]
    feed_dict = {x: batch_X, y: batch_y, keep_prob: dropout_prob}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
    train_writer.add_summary(summary, step)

    step += 1
    pos += batch_size
```

## 评估模型准确性

为了评估模型的准确性，标准做法要求在未用于训练的数据上测量模型的准确性（即验证集）。然而，数据不平衡使这一点变得棘手。我们在上一章中使用的分类准确度指标简单地衡量了被正确标记的数据点的比例。然而，我们数据集中 95%的数据被标记为 0，只有 5%被标记为 1。因此，全 0 模型（将所有内容标记为负面的模型）将实现 95%的准确性！这不是我们想要的。

更好的选择是增加正例的权重，使其更重要。为此，我们使用 MoleculeNet 推荐的每个示例权重来计算加权分类准确性，其中正样本的权重是负样本的 19 倍。在这种加权准确性下，全 0 模型的准确率将达到 50%，这似乎更为合理。

对于计算加权准确性，我们使用`sklearn.metrics`中的函数`accuracy_score(true, pred, sample_weight=given_sample_weight`。这个函数有一个关键字参数`sample_weight`，让我们可以为每个数据点指定所需的权重。我们使用这个函数在训练集和验证集上计算加权指标（例 4-9）。

##### 例 4-9。计算加权准确性

```py
train_weighted_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
print("Train Weighted Classification Accuracy: %f" % train_weighted_score)
valid_weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
print("Valid Weighted Classification Accuracy: %f" % valid_weighted_score)
```

虽然我们可以自己重新实现这个函数，但有时使用 Python 数据科学基础设施中的标准函数会更容易（并且更少出错）。了解这种基础设施和可用函数是作为一名实践数据科学家的一部分。现在，我们可以训练模型（在默认设置下进行 10 个时期）并评估其准确性：

```py
Train Weighted Classification Accuracy: 0.742045
Valid Weighted Classification Accuracy: 0.648828
```

在第五章中，我们将向您展示系统地提高这种准确性的方法，并更仔细地调整我们的完全连接模型。

## 使用 TensorBoard 跟踪模型收敛

现在我们已经指定了我们的模型，让我们使用 TensorBoard 来检查模型。让我们首先在 TensorBoard 中检查图结构（图 4-10）。

该图与逻辑回归的图类似，只是增加了一个新的隐藏层。让我们扩展隐藏层，看看里面有什么（图 4-11）。

![fcgraph.png](img/tfdl_0410.png)

###### 图 4-10。可视化全连接网络的计算图。

![hidden_expand.png](img/tfdl_0411.png)

###### 图 4-11。可视化全连接网络的扩展计算图。

您可以看到这里如何表示新的可训练变量和 dropout 操作。一切看起来都在正确的位置。让我们通过查看随时间变化的损失曲线来结束（图 4-12）。

![fcnet_loss_curve.png](img/tfdl_0412.png)

###### 图 4-12。可视化全连接网络的损失曲线。

正如我们在前一节中看到的那样，损失曲线呈下降趋势。但是，让我们放大一下，看看这个损失在近距离下是什么样子的（图 4-13）。

![fcnet_zoomed_loss.png](img/tfdl_0413.png)

###### 图 4-13。放大损失曲线的一部分。

请注意，损失看起来更加崎岖！这是使用小批量训练的代价之一。我们不再拥有在前几节中看到的漂亮、平滑的损失曲线。

# 回顾

在本章中，我们向您介绍了全连接深度网络。我们深入研究了这些网络的数学理论，并探讨了“通用逼近”的概念，这在一定程度上解释了全连接网络的学习能力。我们以一个案例研究结束，您在该案例中训练了一个深度全连接架构的 Tox21 数据集。

在本章中，我们还没有向您展示如何调整全连接网络以实现良好的预测性能。在第五章中，我们将讨论“超参数优化”，即调整网络参数的过程，并让您调整本章介绍的 Tox21 网络的参数。
