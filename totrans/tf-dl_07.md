# 第7章。循环神经网络

到目前为止，在这本书中，我们向您介绍了使用深度学习处理各种类型输入的方法。我们从简单的线性和逻辑回归开始，这些回归是在固定维度的特征向量上进行的，然后讨论了全连接的深度网络。这些模型接受任意大小的特征向量，这些向量的大小是固定的，预先确定的。这些模型不对编码到这些向量中的数据类型做任何假设。另一方面，卷积网络对其数据的结构做出了强烈的假设。卷积网络的输入必须满足一个允许定义局部感受野的局部性假设。

到目前为止，我们如何使用我们描述的网络来处理句子等数据？句子确实具有一些局部性质（附近的单词通常相关），因此确实可以使用一维卷积网络来处理句子数据。尽管如此，大多数从业者倾向于使用不同类型的架构，即循环神经网络，以处理数据序列。

循环神经网络（RNNs）被设计成允许深度网络处理数据序列。RNNs假设传入的数据采用向量或张量序列的形式。如果我们将句子中的每个单词转换为向量（稍后会详细介绍如何做到这一点），那么句子可以被馈送到RNN中。同样，视频（被视为图像序列）也可以通过RNN进行处理。在每个序列位置，RNN对该序列位置的输入应用任意非线性转换。这种非线性转换对所有序列步骤都是共享的。

前面段落中的描述有点抽象，但事实证明它非常强大。在本章中，您将了解有关RNN结构的更多细节，以及如何在TensorFlow中实现RNN。我们还将讨论RNN如何在实践中用于执行诸如抽样新句子或为诸如聊天机器人之类的应用生成文本的任务。

本章的案例研究在Penn Treebank语料库上训练了一个循环神经网络语言模型，这是从《华尔街日报》文章中提取的句子集合。本教程改编自TensorFlow官方文档关于循环网络的教程。（如果您对我们所做的更改感兴趣，我们鼓励您访问TensorFlow网站上的原始教程。）与往常一样，我们建议您跟着本书相关的[GitHub存储库](https://github.com/matroid/dlwithtf)中的代码进行学习。

# 循环架构概述

循环架构对建模非常复杂的时变数据集非常有用。时变数据集传统上称为*时间序列*。[图7-1](#ch7-timeseries)显示了一些时间序列数据集。

![time_series.png](assets/tfdl_0701.png)

###### 图7-1。我们可能感兴趣建模的一些时间序列数据集。

在时间序列建模中，我们设计学习系统，该系统能够学习演化规则，即根据过去学习系统的未来如何演变。从数学上讲，假设在每个时间步骤，我们接收到一个数据点<math alttext="x Subscript t"><msub><mi>x</mi> <mi>t</mi></msub></math>，其中*t*是当前时间。然后，时间序列方法试图学习某个函数*f*，使得

<math display="block"><mrow><msub><mi>x</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>=</mo> <mi>f</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <mo>⋯</mo> <mo>,</mo> <msub><mi>x</mi> <mi>t</mi></msub> <mo>)</mo></mrow></mrow></math>

这个想法是*f*很好地编码了系统的基本动态，从数据中学习它将使学习系统能够预测手头系统的未来。在实践中，学习一个依赖于所有过去输入的函数太过繁琐，因此学习系统通常假设所有关于上一个数据点 <math alttext="x 1 comma ellipsis comma x Subscript t minus 1 Baseline"><mrow><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <mo>⋯</mo> <mo>,</mo> <msub><mi>x</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></mrow></math> 的信息可以被编码为某个固定向量 <math alttext="h Subscript t"><msub><mi>h</mi> <mi>t</mi></msub></math> 。然后，更新方程简化为以下格式

<math display="block"><mrow><msub><mi>x</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>,</mo> <msub><mi>h</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>=</mo> <mi>f</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>h</mi> <mi>t</mi></msub> <mo>)</mo></mrow></mrow></math>

请注意，我们假设这里的相同函数*f*适用于所有时间步*t*。也就是说，我们假设时间序列是*稳态*的（参见[图7-2](#ch7-timeseriesmodel)）。这种假设对许多系统来说是不成立的，特别是包括股票市场，今天的规则不一定适用于明天。

![RNN-unrolled.png](assets/tfdl_0702.png)

###### 图7-2。具有稳态演化规则的时间序列的数学模型。请记住，稳态系统是指其基本动态不随时间变化的系统。

这个方程与循环神经网络有什么关系呢？基本答案源自我们在[第4章](ch04.html#fully_connected_networks)中介绍的通用逼近定理。函数*f*可以是任意复杂的，因此使用全连接的深度网络来学习*f*似乎是一个合理的想法。这种直觉基本上定义了RNN。一个简单的循环网络可以被视为一个被重复应用到数据的每个时间步长的全连接网络。

事实上，循环神经网络只有在复杂的高维时间序列中才真正变得有趣。对于更简单的系统，通常有经典的信号处理时间序列方法可以很好地建模时间动态。然而，对于复杂系统，如语音（请参见[图7-3](#ch7-spect)中的语音谱图），RNN将展现出自己的能力，并提供其他方法无法提供的功能。

![sepctrogram.GIF](assets/tfdl_0703.png)

###### 图7-3。代表语音样本中发现的频率的语音谱图。

# 循环细胞

# 梯度不稳定

随着时间的推移，循环网络往往会降低信号。可以将其视为在每个时间步长上通过乘法因子衰减信号。因此，经过50个时间步长后，信号会相当衰减。

由于这种不稳定性，训练长时间序列上的循环神经网络一直是具有挑战性的。已经出现了许多方法来对抗这种不稳定性，我们将在本节的其余部分讨论这些方法。

有许多关于简单循环神经网络概念的阐述，在实际应用中已被证明更成功。在本节中，我们将简要回顾其中一些变体。

## 长短期记忆（LSTM）

标准循环细胞的挑战之一是来自遥远过去的信号会迅速衰减。因此，RNN可能无法学习复杂依赖关系的模型。这种失败在语言建模等应用中尤为显著，其中单词可能对先前短语有复杂的依赖关系。

解决这个问题的一个潜在方案是允许过去的状态无修改地传递。长短期记忆（LSTM）架构提出了一种机制，允许过去的状态以最小的修改传递到现在。经验表明，使用LSTM“单元”（如[图7-4](#ch7-lstm)所示）在学习性能方面似乎比使用完全连接层的简单递归神经网络表现更好。

![colah_lstm.png](assets/tfdl_0704.png)

###### 图7-4. 长短期记忆（LSTM）单元。LSTM在保留输入的长距离依赖性方面比标准递归神经网络表现更好。因此，LSTM通常被用于复杂的序列数据，如自然语言。

# 这么多方程！

LSTM方程涉及许多复杂的术语。如果您有兴趣准确理解LSTM背后的数学直觉，我们鼓励您用铅笔和纸玩弄这些方程，并尝试对LSTM单元进行求导。

然而，对于其他主要关注使用递归架构解决实际问题的读者，我们认为深入研究LSTM工作的细节并非绝对必要。相反，保持高层次的直觉，即允许过去状态传递，并深入研究本章的示例代码。

# 优化递归网络

与完全连接网络或卷积网络不同，LSTM涉及一些复杂的数学运算和控制流操作。因此，即使使用现代GPU硬件，训练大型递归网络在规模上仍然具有挑战性。

已经付出了大量努力来优化RNN实现，以便在GPU硬件上快速运行。特别是，Nvidia已将RNN集成到其CuDNN库中，该库提供了专门针对GPU上训练深度网络的优化代码。对于TensorFlow用户来说，与CuDNN等库的集成是在TensorFlow内部完成的，因此您不需要过多担心代码优化（除非您正在处理非常大规模的数据集）。我们将在[第9章](ch09.html#training_large_deep_networks)中更深入地讨论深度神经网络的硬件需求。

## 门控循环单元（GRU）

LSTM单元的复杂性，无论是概念上还是计算上，都激发了许多研究人员试图简化LSTM方程，同时保留原始方程的性能增益和建模能力。

有许多可替代LSTM的竞争者，但其中一种领先者是门控循环单元（GRU），如[图7-5](#ch7-gru)所示。GRU去除了LSTM的一个子组件，但经验表明其性能与LSTM相似。GRU可能是序列建模项目中LSTM单元的合适替代品。

![lstm_gru.png](assets/tfdl_0705.png)

###### 图7-5. 门控循环单元（GRU）单元。GRU以较低的计算成本保留了许多LSTM的优点。

# 递归模型的应用

虽然递归神经网络是对建模时间序列数据集有用的工具，但递归网络还有一系列其他应用。这些应用包括自然语言建模、机器翻译、化学逆合成以及使用神经图灵机进行任意计算。在本节中，我们简要介绍了一些这些令人兴奋的应用。

## 从递归网络中采样

到目前为止，我们已经教会了您循环网络如何学习对数据序列的时间演变进行建模。如果您理解了一组序列的演变规则，那么您应该能够从训练序列的分布中采样新的序列。事实证明，训练模型可以生成良好的序列。迄今为止最有用的应用是语言建模。能够生成逼真的句子是一个非常有用的工具，支撑着自动完成和聊天机器人等系统。

# 为什么我们不使用GAN来处理序列？

在[第6章](ch06.html#convolutional_neural_networks)中，我们讨论了生成新图像的问题。我们讨论了诸如变分自动编码器之类的模型，这些模型只生成模糊的图像，并引入了能够生成清晰图像的生成对抗网络技术。然而，问题仍然存在：如果我们需要GAN来获得良好的图像样本，为什么我们不用它们来获得良好的句子？

事实证明，今天的生成对抗模型在采样序列方面表现平平。目前尚不清楚原因。对GAN的理论理解仍然非常薄弱（即使按照深度学习理论的标准），但是关于博弈论均衡发现的某些东西似乎在序列方面表现不如图像。

## Seq2seq模型

序列到序列（seq2seq）模型是强大的工具，使模型能够将一个序列转换为另一个序列。序列到序列模型的核心思想是使用一个编码循环网络，将输入序列嵌入到向量空间中，同时使用一个解码网络，使得可以对输出序列进行采样，如前面的句子所述。[图7-6](#ch7-seq2seq)说明了一个seq2seq模型。

![seq2seq_colah.png](assets/tfdl_0706.png)

###### 图7-6。序列到序列模型是强大的工具，可以学习序列转换。它们已经应用于机器翻译（例如，将一系列英语单词转换为中文）和化学逆合成（将一系列化学产品转换为一系列反应物）。

事情变得有趣起来，因为编码器和解码器层本身可以很深。 （RNN层可以以自然的方式堆叠。）Google神经机器翻译（GNMT）系统有许多堆叠的编码和解码层。由于这种强大的表征能力，它能够执行远远超出其最近的非深度竞争对手能力的最先进的翻译。[图7-7](#ch7-gnmt)说明了GNMT架构。

![google_nmt.png](assets/tfdl_0707.png)

###### 图7-7。Google神经机器翻译（GNMT）架构是一个深度的seq2seq模型，学习执行机器翻译。

到目前为止，我们主要讨论了自然语言处理的应用，seq2seq架构在其他领域有着无数的应用。其中一位作者已经使用seq2seq架构来执行化学逆合成，即将分子分解为更简单的组分。[图7-8](#ch7-seqret)说明。

![seq2seq_retrosynthesis.png](assets/tfdl_0708.png)

###### 图7-8。化学逆合成的seq2seq模型将一系列化学产品转化为一系列化学反应物。

# 神经图灵机

机器学习的梦想是向抽象堆栈的更高层次发展：从学习短模式匹配引擎到学习执行任意计算。神经图灵机是这种演变中的一个强大步骤。

图灵机是计算理论中的一个重要贡献。它是第一个能够执行任何计算的机器的数学模型。图灵机维护一个提供已执行计算的内存的“带子”。机器的第二部分是一个在单个带子单元上执行转换的“头”。图灵机的见解是，“头”并不需要非常复杂就能执行任意复杂的计算。

神经图灵机（NTM）是将图灵机本身转变为神经网络的一种非常巧妙的尝试。在这种转变中的技巧是将离散动作转变为软连续函数（这是深度学习中反复出现的技巧，所以请注意！）

图灵机头部与RNN单元非常相似！因此，NTM可以被端到端地训练，以学习执行任意计算，至少在原则上是这样的（[图7-9](#ch7-neuraltm)）。实际上，NTM能够执行的计算集合存在严重的限制。梯度流不稳定性（一如既往）限制了可以学习的内容。需要更多的研究和实验来设计NTM的后继者，使其能够学习更有用的函数。

![turing_machine.png](assets/tfdl_0709.png)

###### 图7-9。神经图灵机（NTM）是图灵机的可学习版本。它维护一个带子，可以在其中存储中间计算的输出。虽然NTM有许多实际限制，但可能它们的智能后代将能够学习强大的算法。

# 图灵完备性

图灵完备性的概念在计算机科学中是一个重要的概念。如果一种编程语言能够执行图灵机能够执行的任何计算，那么它被称为图灵完备。图灵机本身是为了提供一个数学模型，说明一个函数“可计算”的含义而发明的。该机器提供了读取、写入和存储各种指令的能力，这些抽象原语是所有计算机的基础。

随着时间的推移，大量的工作表明图灵机紧密地模拟了物理世界中可执行的计算集合。在第一次近似中，如果可以证明图灵机无法执行某个计算，那么任何计算设备也无法执行。另一方面，如果可以证明计算系统可以执行图灵机的基本操作，那么它就是“图灵完备”的，可以原则上执行任何计算。一些令人惊讶的系统是图灵完备的。如果感兴趣，我们鼓励您阅读更多关于这个主题的内容。

# 循环网络是图灵完备的

也许并不令人惊讶的是，NTM能够执行图灵机能够执行的任何计算，因此是图灵完备的。然而，一个较少人知道的事实是，普通的循环神经网络本身也是图灵完备的！换句话说，原则上，循环神经网络能够学习执行任意计算。

基本思想是转换操作符可以学习执行基本的读取、写入和存储操作。随时间展开的循环网络允许执行复杂的计算。在某种意义上，这个事实不应该太令人惊讶。通用逼近定理已经证明全连接网络能够学习任意函数。随时间将任意函数链接在一起导致任意计算。（尽管正式证明这一点所需的技术细节是艰巨的。）

# 在实践中使用循环神经网络

在本节中，您将了解如何在Penn Treebank数据集上使用递归神经网络进行语言建模，这是一个由*华尔街日报*文章构建的自然语言数据集。 我们将介绍执行此建模所需的TensorFlow基元，并将指导您完成准备数据进行训练所需的数据处理和预处理步骤。 我们鼓励您跟着尝试在与本书相关的[GitHub存储库](https://github.com/matroid/dlwithtf)中运行代码。

# 处理Penn Treebank语料库

Penn Treebank包含一个由*华尔街日报*文章组成的百万字语料库。 此语料库可用于字符级或单词级建模（预测给定前面的句子中的下一个字符或单词的任务）。 使用训练模型的困惑度来衡量模型的有效性（稍后将详细介绍此指标）。

Penn Treebank语料库由句子组成。 我们如何将句子转换为可以馈送到机器学习系统（例如递归语言模型）的形式？ 请记住，机器学习模型接受张量（递归模型接受张量序列）作为输入。 因此，我们需要将单词转换为机器学习的张量。

将单词转换为向量的最简单方法是使用“一热”编码。 在此编码中，假设我们的语言数据集使用具有<math><mrow><mo>|</mo> <mi>V</mi> <mo>|</mo></mrow></math>个单词的词汇表。 然后，每个单词转换为形状为<math><mrow><mo>(</mo> <mo>|</mo> <mi>V</mi> <mo>|</mo> <mo>)</mo></mrow></math>的向量。 此向量的所有条目都为零，除了一个条目，在索引处，该索引对应于当前单词。 有关此嵌入的示例，请参见[图7-10](#ch7-one-hot)。

![one-hot.jpg](assets/tfdl_0710.png)

###### 图7-10。 一热编码将单词转换为只有一个非零条目的向量（通常设置为一）。 向量中的不同索引唯一表示语言语料库中的单词。

也可以使用更复杂的嵌入。 基本思想类似于一热编码。 每个单词与唯一向量相关联。 但是，关键区别在于可以直接从数据中学习此编码向量，以获得对于当前数据集有意义的单词的“单词嵌入”。 我们将在本章后面向您展示如何学习单词嵌入。

为了处理Penn Treebank数据，我们需要找到语料库中使用的单词的词汇表，然后将每个单词转换为其关联的单词向量。 然后，我们将展示如何将处理后的数据馈送到TensorFlow模型中。

# Penn Treebank的限制

Penn Treebank是语言建模的一个非常有用的数据集，但对于最先进的语言模型来说已经不再构成挑战； 研究人员已经在这个集合的特殊性上过拟合了模型。 最先进的研究将使用更大的数据集，例如十亿字语料库语言基准。 但是，对于我们的探索目的，Penn Treebank已经足够。

## 预处理代码

[示例7-1](#ch7-readwords)中的代码片段读取了与Penn Treebank语料库相关的原始文件。 语料库存储在每行一个句子的形式中。 通过一些Python字符串处理，将`"\n"`换行标记替换为固定标记`"<eos>"`，然后将文件拆分为标记列表。

##### 示例7-1。 此函数读取原始Penn Treebank文件

```py
def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()
```

定义了`_read_words`后，我们可以使用[示例7-2](#ch7-vocab)中定义的`_build_vocab`函数构建与给定文件相关联的词汇表。我们简单地读取文件中的单词，并使用Python的`collections`库计算文件中唯一单词的数量。为方便起见，我们构建一个字典对象，将单词映射到它们的唯一整数标识符（在词汇表中的位置）。将所有这些联系在一起，`_file_to_word_ids`将文件转换为单词标识符列表（[示例7-3](#ch7-file-to-words)）。

##### 示例7-2。此函数构建一个由指定文件中所有单词组成的词汇表

```py
def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id
```

##### 示例7-3。此函数将文件中的单词转换为id号

```py
def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]
```

有了这些实用工具，我们可以使用函数`ptb_raw_data`（[示例7-4](#ch7-ptb-raw)）处理Penn Treebank语料库。请注意，训练、验证和测试数据集是预先指定的，因此我们只需要将每个文件读入一个唯一索引列表中。

##### 示例7-4。此函数从指定位置加载Penn Treebank数据

```py
def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

 Reads PTB text files, converts strings to integer ids,
 and performs mini-batching of the inputs.

 The PTB dataset comes from Tomas Mikolov's webpage:
 http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

 Args:
 data_path: string path to the directory where simple-examples.tgz
 has been extracted.

 Returns:
 tuple (train_data, valid_data, test_data, vocabulary)
 where each of the data objects can be passed to PTBIterator.
 """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary
```

# tf.GFile和tf.Flags

TensorFlow是一个庞大的项目，包含许多部分。虽然大部分库专注于机器学习，但也有相当大比例的库专门用于加载和处理数据。其中一些功能提供了在其他地方找不到的有用功能。然而，加载功能的其他部分则不太有用。

`tf.GFile`和`tf.FLags`提供的功能与标准Python文件处理和`argparse`几乎相同。这些工具的来源是历史性的。对于Google来说，内部代码标准要求使用自定义文件处理程序和标志处理程序。然而，对于我们其他人来说，尽可能使用标准Python工具是更好的风格。这样做对于可读性和稳定性更有利。

## 将数据加载到TensorFlow中

在本节中，我们将介绍加载我们处理过的索引到TensorFlow所需的代码。为此，我们将向您介绍一些新的TensorFlow机制。到目前为止，我们已经使用feed字典将数据传递到TensorFlow中。虽然feed字典对于小型玩具数据集来说是可以接受的，但对于较大的数据集来说，它们通常不是一个好选择，因为引入了大量Python开销，涉及打包和解包字典。为了更高性能的代码，最好使用TensorFlow队列。

`tf.Queue`提供了一种异步加载数据的方式。这允许将GPU计算线程与CPU绑定的数据预处理线程解耦。这种解耦对于希望保持GPU最大活跃性的大型数据集特别有用。

可以将`tf.Queue`对象提供给TensorFlow占位符以训练模型并实现更高的性能。我们将在本章后面演示如何做到这一点。

在[示例7-5](#ch7-ptb-producer)中介绍的`ptb_producer`函数将原始索引列表转换为可以将数据传递到TensorFlow计算图中的`tf.Queues`。让我们首先介绍一些我们使用的计算原语。`tf.train.range_input_producer`是一个方便的操作，从输入张量生成一个`tf.Queue`。方法`tf.Queue.dequeue()`从队列中提取一个张量进行训练。`tf.strided_slice`提取与当前小批量数据对应的部分张量。

##### 示例7-5。此函数从指定位置加载Penn Treebank数据

```py
def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

 This chunks up raw_data into batches of examples and returns
 Tensors that are drawn from these batches.

 Args:
 raw_data: one of the raw data outputs from ptb_raw_data.
 batch_size: int, the batch size.
 num_steps: int, the number of unrolls.
 name: the name of this operation (optional).

 Returns:
 A pair of Tensors, each shaped [batch_size, num_steps]. The
 second element of the tuple is the same data time-shifted to the
 right by one.

 Raises:
 tf.errors.InvalidArgumentError: if batch_size or num_steps are
 too high.
 """
  with tf.name_scope(name, "PTBProducer",
                     [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data",
                                    dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size,
                                      shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y
```

# tf.data

TensorFlow（从版本1.4开始）支持一个新模块`tf.data`，其中包含一个新类`tf.data.Dataset`，提供了一个明确的API来表示数据流。很可能`tf.data`最终会取代队列成为首选的输入模式，特别是因为它具有经过深思熟虑的功能API。

在撰写本文时，`tf.data`模块刚刚发布，与API的其他部分相比仍然相对不成熟，因此我们决定在示例中继续使用队列。但是，我们鼓励您自己了解`tf.data`。

## 基本循环架构

我们将使用LSTM单元来对Penn Treebank进行建模，因为LSTMs通常在语言建模挑战中表现出优越性能。函数`tf.contrib.rnn.BasicLSTMCell`已经为我们实现了基本的LSTM单元，因此无需自行实现（[示例7-6](#ch7-lstm-wrap)）。

##### 示例7-6。这个函数从tf.contrib中包装了一个LSTM单元

```py
def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(
      size, forget_bias=0.0, state_is_tuple=True,
      reuse=tf.get_variable_scope().reuse)
```

# 使用TensorFlow Contrib代码是否可以接受？

请注意，我们使用的LSTM实现来自`tf.contrib`。在工业强度项目中使用`tf.contrib`中的代码是否可以接受？对此仍有争议。根据我们的个人经验，`tf.contrib`中的代码往往比核心TensorFlow库中的代码稍微不稳定，但通常仍然相当可靠。`tf.contrib`中通常有许多有用的库和实用程序，而核心TensorFlow库中并没有。我们建议根据需要使用`tf.contrib`中的部分代码，但请注意您使用的部分并在核心TensorFlow库中有等价物时进行替换。

[示例7-7](#ch7-embed)中的代码片段指示TensorFlow为我们的词汇表中的每个单词学习一个词嵌入。对我们来说关键的函数是`tf.nn.embedding_lookup`，它允许我们执行正确的张量查找操作。请注意，我们需要手动将嵌入矩阵定义为TensorFlow变量。

##### 示例7-7。为词汇表中的每个单词学习一个词嵌入

```py
with tf.device("/cpu:0"):
  embedding = tf.get_variable(
      "embedding", [vocab_size, size], dtype=tf.float32)
  inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
```

有了我们手头的词向量，我们只需要将LSTM单元（使用函数`lstm_cell`）应用于序列中的每个词向量。为此，我们只需使用Python的`for`循环来构建所需的一系列对`cell()`的调用。这里只有一个技巧：我们需要确保在每个时间步重复使用相同的变量，因为LSTM单元应在每个时间步执行相同的操作。幸运的是，变量作用域的`reuse_variables()`方法使我们能够轻松做到这一点。参见[示例7-8](#ch7-embed2)。

##### 示例7-8。将LSTM单元应用于输入序列中的每个词向量

```py
outputs = []
state = self._initial_state
with tf.variable_scope("RNN"):
  for time_step in range(num_steps):
    if time_step > 0: tf.get_variable_scope().reuse_variables()
    (cell_output, state) = cell(inputs[:, time_step, :], state)
    outputs.append(cell_output)
```

现在剩下的就是定义与图相关的损失，以便对其进行训练。幸运的是，TensorFlow在`tf.contrib`中提供了用于训练语言模型的损失。我们只需要调用`tf.contrib.seq2seq.sequence_loss`（[示例7-9](#ch7-seqloss)）。在底层，这个损失实际上是一种困惑度。

##### 示例7-9。添加序列损失

```py
# use the contrib sequence loss and average over the batches
loss = tf.contrib.seq2seq.sequence_loss(
   logits,
   input_.targets,
   tf.ones([batch_size, num_steps], dtype=tf.float32),
   average_across_timesteps=False,
   average_across_batch=True
)
# update the cost variables
self._cost = cost = tf.reduce_sum(loss)
```

# 困惑

困惑度经常用于语言建模挑战。它是二元交叉熵的一种变体，用于衡量学习分布与数据真实分布之间的接近程度。经验上，困惑度在许多语言建模挑战中都被证明是有用的，我们在这里也利用了它（因为`sequence_loss`只是实现了针对序列的困惑度）。

然后我们可以使用标准的梯度下降方法训练这个图。我们略去了底层代码的一些混乱细节，但建议您如果感兴趣可以查看GitHub。评估训练模型的质量也变得简单，因为困惑度既用作训练损失又用作评估指标。因此，我们可以简单地显示`self._cost`来评估模型的训练情况。我们鼓励您自己训练模型！

## 读者的挑战

尝试通过尝试不同的模型架构降低Penn Treebank上的困惑度。请注意，这些实验可能会在没有GPU的情况下耗费时间。

# 回顾

本章向您介绍了循环神经网络（RNNs），这是一种用于学习序列数据的强大架构。RNNs能够学习控制数据序列的基本演变规则。虽然RNNs可以用于建模简单的时间序列，但在建模复杂的序列数据（如语音和自然语言）时最为强大。

我们向您介绍了许多RNN变体，如LSTMs和GRUs，它们在具有复杂长程交互的数据上表现更好，并且还简要讨论了神经图灵机的令人兴奋的前景。我们以一个深入的案例研究结束了本章，该案例将LSTMs应用于对Penn Treebank进行建模。

在[第8章](ch08.html#reinforcement_learning)中，我们将向您介绍强化学习，这是一种学习玩游戏的强大技术。
