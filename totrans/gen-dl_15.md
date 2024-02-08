# 第十一章：音乐生成

音乐作曲是一个复杂而创造性的过程，涉及将不同的音乐元素（如旋律、和声、节奏和音色）结合在一起。虽然传统上认为这是一种独特的人类活动，但最近的进展使得生成既能让耳朵愉悦又具有长期结构的音乐成为可能。

音乐生成最流行的技术之一是 Transformer，因为音乐可以被视为一个序列预测问题。这些模型已经被调整为通过将音符视为一系列标记（类似于句子中的单词）来生成音乐。Transformer 模型学会根据先前的音符预测序列中的下一个音符，从而生成一段音乐。

MuseGAN 采用了一种完全不同的方法来生成音乐。与 Transformer 逐音符生成音乐不同，MuseGAN 通过将音乐视为一个*图像*，由音高轴和时间轴组成，一次生成整个音乐曲目。此外，MuseGAN 将不同的音乐组成部分（如和弦、风格、旋律和节奏）分开，以便可以独立控制它们。

在本章中，我们将学习如何处理音乐数据，并应用 Transformer 和 MuseGAN 来生成与给定训练集风格相似的音乐。

# 介绍

为了让机器创作出我们耳朵愉悦的音乐，它必须掌握我们在第九章中看到的与文本相关的许多技术挑战。特别是，我们的模型必须能够学习并重新创建音乐的顺序结构，并能够从一系列可能性中选择后续音符。

然而，音乐生成面临着文本生成所没有的额外挑战，即音高和节奏。音乐通常是复调的，即有几条音符流同时在不同乐器上演奏，这些音符组合在一起形成既不和谐（冲突）又和谐（和谐）的和声。文本生成只需要我们处理一条文本流，与音乐中存在的并行和弦流相比。

此外，文本生成可以逐字处理。与文本数据不同，音乐是一种多部分、交织在一起的声音织锦，不一定同时传递——听音乐的乐趣很大程度上来自于整个合奏中不同节奏之间的相互作用。例如，吉他手可能弹奏一连串更快的音符，而钢琴家则弹奏一个较长的持续和弦。因此，逐音符生成音乐是复杂的，因为我们通常不希望所有乐器同时改变音符。

我们将从简化问题开始，专注于为单一（单声部）音乐线生成音乐。许多来自第九章关于文本生成的技术也可以用于音乐生成，因为这两个任务有许多共同的主题。我们将首先训练一个 Transformer 来生成类似于 J.S.巴赫大提琴组曲风格的音乐，并看看注意机制如何使模型能够专注于先前的音符，以确定最自然的后续音符。然后，我们将处理复调音乐生成的任务，并探讨如何使用基于 GAN 的架构来为多声部创作音乐。

# 用于音乐生成的 Transformer

我们将在这里构建的模型是一个解码器 Transformer，灵感来自于 OpenAI 的*MuseNet*，它也利用了一个解码器 Transformer（类似于 GPT-3），训练以预测给定一系列先前音符的下一个音符。

在音乐生成任务中，随着音乐的进行，序列的长度<math alttext="upper N"><mi>N</mi></math>变得很大，这意味着每个头部的<math alttext="upper N times upper N"><mrow><mi>N</mi> <mo>×</mo> <mi>N</mi></mrow></math>注意力矩阵变得昂贵且难以存储和计算。我们理想情况下不希望将输入序列剪切为少量标记，因为我们希望模型围绕长期结构构建乐曲，并重复几分钟前的主题和乐句，就像人类作曲家一样。

为了解决这个问题，MuseNet 利用了一种称为[*Sparse Transformer*](https://oreil.ly/euQiL)的 Transformer 形式。注意矩阵中的每个输出位置仅计算一部分输入位置的权重，从而减少了训练模型所需的计算复杂性和内存。MuseNet 因此可以在 4,096 个标记上进行全注意力操作，并可以学习跨多种风格的长期结构和旋律结构。 （例如，查看 OpenAI 在 SoundCloud 上的[肖邦](https://oreil.ly/cmwsO)和[莫扎特](https://oreil.ly/-T-Je)的录音。）

要看到音乐短语的延续通常受几个小节前的音符影响，看看巴赫大提琴组曲第 1 号前奏的开头小节吧（图 11-1）。

![](img/gdl2_1101.png)

###### 图 11-1。巴赫的大提琴组曲第 1 号（前奏）

# 小节

*小节*（或*节拍*）是包含固定数量的拍子的音乐小单位，并由穿过五线谱的垂直线标记出来。如果你能够数 1、2、1、2，那么每个小节有两拍，你可能在听进行曲。如果你能够数 1、2、3、1、2、3，那么每个小节有三拍，你可能在听华尔兹。

你认为接下来会是什么音符？即使你没有音乐训练，你可能仍然能猜到。如果你说是 G（与乐曲的第一个音符相同），那么你是正确的。你是怎么知道的？你可能能够看到每个小节和半小节都以相同的音符开头，并利用这些信息来做出决定。我们希望我们的模型能够执行相同的技巧——特别是，我们希望它能够关注前半小节中的特定音符，当前一个低 G 被记录时。基于注意力的模型，如 Transformer，将能够在不必在许多小节之间保持隐藏状态的情况下，合并这种长期回顾。

任何尝试音乐生成任务的人首先必须对音乐理论有基本的了解。在下一节中，我们将介绍阅读音乐所需的基本知识以及如何将其数值化，以便将音乐转换为训练 Transformer 所需的输入数据。

# 运行此示例的代码

此示例的代码可以在位于书存储库中的 Jupyter 笔记本*notebooks/11_music/01_transformer/transformer.ipynb*中找到。

## 巴赫大提琴组曲数据集

我们将使用的原始数据集是 J.S.巴赫的大提琴组曲的一组 MIDI 文件。您可以通过在书的存储库中运行数据集下载脚本来下载数据集，如示例 11-1 所示。这将把 MIDI 文件保存到本地的*/data*文件夹中。

##### 示例 11-1。下载 J.S.巴赫大提琴组曲数据集

```py
bash scripts/download_music_data.sh
```

`要查看并听取模型生成的音乐，您需要一些能够生成乐谱的软件。[MuseScore](https://musescore.org)是一个很好的工具，可以免费下载。`  `## 解析 MIDI 文件

我们将使用 Python 库`music21`将 MIDI 文件加载到 Python 中进行处理。示例 11-2 展示了如何加载一个 MIDI 文件并可视化它（图 11-2），既作为乐谱又作为结构化数据。

![](img/gdl2_1102.png)

###### 图 11-2。音乐符号

##### 示例 11-2。导入 MIDI 文件

```py
import music21

file = "/app/data/bach-cello/cs1-2all.mid"
example_score = music21.converter.parse(file).chordify()
```

# 八度

每个音符名称后面的数字表示音符所在的*八度*——因为音符名称（A 到 G）重复，这是为了唯一标识音符的音高。例如，`G2`是低于`G3`的一个八度。

现在是时候将乐谱转换成更像文本的东西了！我们首先循环遍历每个乐谱，并将乐曲中每个元素的音符和持续时间提取到两个单独的文本字符串中，元素之间用空格分隔。我们将乐曲的调号和拍号编码为特殊符号，持续时间为零。

# 单声部与复调音乐

在这个第一个例子中，我们将把音乐视为*单声部*（一条单独的线），只取任何和弦的最高音。有时我们可能希望保持各声部分开，以生成*复调*性质的音乐。这带来了我们将在本章后面解决的额外挑战。

这个过程的输出显示在图 11-3 中——将其与图 11-2 进行比较，以便看到原始音乐数据如何转换为这两个字符串。

![](img/gdl2_1103.png)

###### 图 11-3。*音符*文本字符串和*持续时间*文本字符串的示例，对应于图 11-2

这看起来更像我们之前处理过的文本数据。*单词*是音符-持续时间组合，我们应该尝试构建一个模型，根据先前音符和持续时间的序列来预测下一个音符和持续时间。音乐和文本生成之间的一个关键区别是，我们需要构建一个可以同时处理音符和持续时间预测的模型——即，我们需要处理两个信息流，而不是我们在第九章中看到的单一文本流。

## 标记化

为了创建将训练模型的数据集，我们首先需要像之前为文本语料库中的每个单词所做的那样，对每个音符和持续时间进行标记化。我们可以通过使用`TextVectorization`层，分别应用于音符和持续时间，来实现这一点，如示例 11-3 所示。

##### 示例 11-3。标记化音符和持续时间

```py
def create_dataset(elements):
    ds = (
        tf.data.Dataset.from_tensor_slices(elements)
        .batch(BATCH_SIZE, drop_remainder = True)
        .shuffle(1000)
    )
    vectorize_layer = layers.TextVectorization(
        standardize = None, output_mode="int"
    )
    vectorize_layer.adapt(ds)
    vocab = vectorize_layer.get_vocabulary()
    return ds, vectorize_layer, vocab

notes_seq_ds, notes_vectorize_layer, notes_vocab = create_dataset(notes)
durations_seq_ds, durations_vectorize_layer, durations_vocab = create_dataset(
    durations
)
seq_ds = tf.data.Dataset.zip((notes_seq_ds, durations_seq_ds))
```

完整的解析和标记化过程显示在图 11-4 中。

![](img/gdl2_1104.png)

###### 图 11-4。解析 MIDI 文件并对音符和持续时间进行标记化

## 创建训练集

预处理的最后一步是创建我们将馈送给 Transformer 的训练集。

我们通过将音符和持续时间字符串分成 50 个元素的块来实现这一点，使用滑动窗口技术。输出只是输入窗口向后移动一个音符，这样 Transformer 就被训练来预测未来一个时间步的元素的音符和持续时间，给定窗口中的先前元素。这个示例（仅用四个元素的滑动窗口进行演示）显示在图 11-5 中。

![](img/gdl2_1105.png)

###### 图 11-5。音乐 Transformer 模型的输入和输出——在这个例子中，使用宽度为 4 的滑动窗口创建输入块，然后将其移动一个元素以创建目标输出

我们将在 Transformer 中使用的架构与我们在第九章中用于文本生成的架构相同，但有一些关键的区别。

## 正弦位置编码

首先，我们将介绍一种不同类型的令牌位置编码。在第九章中，我们使用了一个简单的`Embedding`层来编码每个令牌的位置，有效地将每个整数位置映射到模型学习的不同向量。因此，我们需要定义一个最大长度（<math alttext="upper N"><mi>N</mi></math>），该序列可以是，并在这个序列长度上进行训练。这种方法的缺点是无法推断出比这个最大长度更长的序列。您将不得不将输入剪切到最后的<math alttext="upper N"><mi>N</mi></math>个令牌，如果您试图生成长篇内容，则这并不理想。

为了避免这个问题，我们可以转而使用一种称为*sine position embedding*的不同类型的嵌入。这类似于我们在第八章中用来编码扩散模型噪声方差的嵌入。具体来说，以下函数用于将输入序列中单词的位置（<math alttext="p o s"><mrow><mi>p</mi> <mi>o</mi> <mi>s</mi></mrow></math>）转换为长度为<math alttext="d"><mi>d</mi></math>的唯一向量：

<math alttext="StartLayout 1st Row 1st Column upper P upper E Subscript p o s comma 2 i 2nd Column equals 3rd Column sine left-parenthesis StartFraction p o s Over 10 comma 000 Superscript 2 i slash d Baseline EndFraction right-parenthesis 2nd Row 1st Column upper P upper E Subscript p o s comma 2 i plus 1 2nd Column equals 3rd Column cosine left-parenthesis StartFraction p o s Over 10 comma 000 Superscript left-parenthesis 2 i plus 1 right-parenthesis slash d Baseline EndFraction right-parenthesis EndLayout" display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mi>P</mi> <msub><mi>E</mi> <mrow><mi>p</mi><mi>o</mi><mi>s</mi><mo>,</mo><mn>2</mn><mi>i</mi></mrow></msub></mrow></mtd> <mtd><mo>=</mo></mtd> <mtd columnalign="left"><mrow><mo form="prefix">sin</mo> <mo>(</mo> <mfrac><mrow><mi>p</mi><mi>o</mi><mi>s</mi></mrow> <mrow><mn>10</mn><mo>,</mo><msup><mn>000</mn> <mrow><mn>2</mn><mi>i</mi><mo>/</mo><mi>d</mi></mrow></msup></mrow></mfrac> <mo>)</mo></mrow></mtd></mtr> <mtr><mtd columnalign="right"><mrow><mi>P</mi> <msub><mi>E</mi> <mrow><mi>p</mi><mi>o</mi><mi>s</mi><mo>,</mo><mn>2</mn><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow></mtd> <mtd><mo>=</mo></mtd> <mtd columnalign="left"><mrow><mo form="prefix">cos</mo> <mo>(</mo> <mfrac><mrow><mi>p</mi><mi>o</mi><mi>s</mi></mrow> <mrow><mn>10</mn><mo>,</mo><msup><mn>000</mn> <mrow><mo>(</mo><mn>2</mn><mi>i</mi><mo>+</mo><mn>1</mn><mo>)</mo><mo>/</mo><mi>d</mi></mrow></msup></mrow></mfrac> <mo>)</mo></mrow></mtd></mtr></mtable></math>

对于较小的<math alttext="i"><mi>i</mi></math>，这个函数的波长很短，因此函数值沿着位置轴快速变化。较大的<math alttext="i"><mi>i</mi></math>值会产生更长的波长。因此，每个位置都有自己独特的编码，这是不同波长的特定组合。

###### 提示

请注意，此嵌入是为所有可能的位置值定义的。它是一个确定性函数（即，模型不会学习它），它使用三角函数来为每个可能的位置定义一个唯一的编码。

*Keras NLP*模块具有一个内置层，为我们实现了这种嵌入 - 因此，我们可以定义我们的`TokenAndPositionEmbedding`层，如示例 11-4 所示。

##### 示例 11-4。对音符和持续时间进行标记化

```py
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras_nlp.layers.SinePositionEncoding()

    def call(self, x):
        embedding = self.token_emb(x)
        positions = self.pos_emb(embedding)
        return embedding + positions
```

图 11-6 显示了如何将这两种嵌入（令牌和位置）相加以产生序列的整体嵌入。

![](img/gdl2_1106.png)

###### 图 11-6。`TokenAndPositionEmbedding`层将令牌嵌入添加到正弦位置嵌入中，以产生序列的整体嵌入

## 多个输入和输出

现在我们有两个输入流（音符和持续时间）和两个输出流（预测音符和持续时间）。因此，我们需要调整我们的 Transformer 架构以适应这一点。

处理双输入流的方法有很多种。我们可以创建代表每个音符-持续时间对的令牌，然后将序列视为单个令牌流。然而，这样做的缺点是无法表示在训练集中未见过的音符-持续时间对（例如，我们可能独立地看到了`G#2`音符和`1/3`持续时间，但从未一起出现过，因此没有`G#2:1/3`的令牌）。

相反，我们选择分别嵌入音符和持续时间令牌，然后使用连接层创建输入的单一表示，该表示可以被下游 Transformer 块使用。类似地，Transformer 块的输出传递给两个独立的密集层，代表了预测的音符和持续时间概率。整体架构如图 11-7 所示。层输出形状显示了批量大小`b`和序列长度`l`。

![](img/gdl2_1107.png)

###### 图 11-7。音乐生成 Transformer 的架构

另一种方法是将音符和持续时间标记交错到一个单一的输入流中，并让模型学习输出应该是一个音符和持续时间标记交替的单一流。这增加了确保当模型尚未学会如何正确交错标记时，输出仍然可以解析的复杂性。

###### 提示

设计您的模型没有*对*或*错*的方式——其中一部分乐趣就是尝试不同的设置，看看哪种对您最有效！

## 音乐生成 Transformer 的分析

我们将从头开始生成一些音乐，通过向网络提供一个`START`音符标记和`0.0`持续时间标记（即，我们告诉模型假设它是从乐曲的开头开始）。然后我们可以使用与我们在第九章中用于生成文本序列的相同迭代技术来生成一个音乐段落，如下所示：

1.  给定当前序列（音符和持续时间），模型预测两个分布，一个是下一个音符的分布，另一个是下一个持续时间的分布。

1.  我们从这两个分布中进行采样，使用一个`temperature`参数来控制在采样过程中我们希望有多少变化。

1.  选择的音符和持续时间被附加到相应的输入序列中。

1.  这个过程会重复进行，对于我们希望生成的元素数量，会有新的输入序列。

图 11-8 展示了在训练过程的各个时期由模型从头开始生成的音乐示例。我们对音符和持续时间使用了 0.5 的温度。 

![](img/gdl2_1108.png)

###### 图 11-8。当仅使用一个`START`音符标记和`0.0`持续时间标记作为种子时，模型生成的乐段示例

在本节中，我们大部分的分析将集中在音符预测上，而不是持续时间，因为对于巴赫的大提琴组曲来说，和声的复杂性更难捕捉，因此更值得研究。然而，您也可以将相同的分析应用于模型的节奏预测，这对于您可能用来训练该模型的其他音乐风格可能特别相关（比如鼓声）。

关于在图 11-8 中生成的乐段有几点需要注意。首先，看到随着训练的进行，音乐变得越来越复杂。一开始，模型通过坚持使用相同的音符和节奏来保险。到了第 10 个时期，模型已经开始生成小段音符，到了第 20 个时期，它产生了有趣的节奏，并且牢固地确立在一个固定的调（E ♭大调）中。

其次，我们可以通过绘制每个时间步的预测分布的热图来分析随时间变化的音符分布。图 11-9 展示了在图 11-8 中第 20 个时期的示例的热图。

![](img/gdl2_1109.png)

###### 图 11-9。随着时间推移可能的下一个音符的分布（在第 20 个时期）：方块越暗，模型对下一个音符在这个音高的确定性就越高

这里需要注意的一个有趣的点是，模型显然已经学会了哪些音符属于特定的*调*，因为在不属于该调的音符处存在分布中的间隙。例如，在音符 54（对应于 G ♭/F ♯）的行上有一个灰色间隙。在 E ♭大调的音乐作品中，这个音符极不可能出现。模型在生成过程的早期就确立了调，并且随着乐曲的进行，模型选择更有可能出现在该调中的音符，通过关注代表它的标记。

值得一提的是，模型还学会了巴赫特有的风格，即在大提琴上降到低音结束一个乐句，然后又反弹回来开始下一个乐句。看看大约在第 20 个音符附近，乐句以低音 E♭结束——在巴赫大提琴组曲中，通常会回到乐器更高、更响亮的音域开始下一个乐句，这正是模型的预测。在低音 E♭（音高编号 39）和下一个音符之间有一个很大的灰色间隙，预测下一个音符将在音高编号 50 左右，而不是继续在乐器的低音区域漂浮。

最后，我们应该检查我们的注意力机制是否按预期工作。图 11-10 中的水平轴显示了生成的音符序列；垂直轴显示了网络在预测水平轴上的每个音符时所关注的位置。每个方块的颜色显示了在生成序列的每个点上所有头部中的最大注意力权重。方块越暗，表示在序列中这个位置上应用的注意力越多。为简单起见，我们在这个图表中只显示了音符，但网络也会关注每个音符的持续时间。

我们可以看到，在初始调号、拍号和休止符中，网络选择几乎全部注意力放在`START`标记上。这是有道理的，因为这些特征总是出现在音乐片段的开头——一旦音符开始流动，`START`标记基本上就不再受到关注。

当我们超过最初的几个音符时，我们可以看到网络主要关注大约最后两到四个音符，并很少对四个音符之前的音符给予重要权重。再次，这是有道理的；前四个音符中可能包含足够的信息，以了解乐句可能如何继续。此外，一些音符更强烈地回到 D 小调的调号上——例如`E3`（乐曲的第 7 个音符）和`B-2`（B♭-乐曲的第 14 个音符）。这很有趣，因为这些正是依赖 D 小调调号来消除任何模糊的确切音符。网络必须*回顾*调号才能知道调号中有一个 B♭（而不是 B 自然音），但调号中没有一个 E♭（必须使用 E 自然音）。

![](img/gdl2_1110.png)

###### 图 11-10。矩阵中每个方块的颜色表示在水平轴上预测音符时，对垂直轴上每个位置给予的注意力量

还有一些例子表明，网络选择忽略附近的某些音符或休止符，因为它们对理解乐句并没有提供额外信息。例如，倒数第二个音符（`A2`）对三个音符前的`B-2`并不特别关注，但对四个音符前的`A2`稍微更关注。对于模型来说，看位于节拍上的`A2`比看位于节拍外的`B-2`更有趣，后者只是一个过渡音。

请记住，我们并没有告诉模型哪些音符相关，哪些音符属于哪个调号——它通过研究巴赫的音乐自己弄清楚了这一点。

## 多声部音乐的标记化

我们在本节中探讨的 Transformer 对单线（单声部）音乐效果很好，但它能够适应多线（复调）音乐吗？

挑战在于如何将不同的音乐线表示为单个令牌序列。在前一节中，我们决定将音符和音符持续时间分成网络的两个不同输入和输出，但我们也看到我们可以将这些令牌交错成一个单一流。我们可以使用相同的想法来处理复调音乐。这里将介绍两种不同的方法：*网格标记化*和*基于事件的标记化*，正如 2018 年的论文“音乐 Transformer：生成具有长期结构的音乐”中所讨论的那样。¹

### 网格标记化

考虑 J.S.巴赫赞美诗中的两小节音乐。有四个不同的声部（女高音[S]，中音[A]，男高音[T]，低音[B]），分别写在不同的五线谱上。

![](img/gdl2_1111.png)

###### 图 11-11。J.S.巴赫赞美诗的前两小节

我们可以想象在网格上绘制这段音乐，其中 y 轴表示音符的音高，x 轴表示自作品开始以来经过的 16 分音符（四分音符）的数量。如果网格方块被填充，那么在那个时间点有音符在播放。所有四个声部都绘制在同一个网格上。这个网格被称为*钢琴卷*，因为它类似于一卷纸上打孔的物理卷，这在数字系统发明之前被用作记录机制。

我们可以通过首先沿着四个声部，然后沿着时间步骤顺序移动，将网格序列化为令牌流。这将产生一个令牌序列<math alttext="upper S 1 comma upper A 1 comma upper T 1 comma upper B 1 comma upper S 2 comma upper A 2 comma upper T 2 comma upper B 2 comma ellipsis"><mrow><msub><mi>S</mi> <mn>1</mn></msub> <mo>,</mo> <msub><mi>A</mi> <mn>1</mn></msub> <mo>,</mo> <msub><mi>T</mi> <mn>1</mn></msub> <mo>,</mo> <msub><mi>B</mi> <mn>1</mn></msub> <mo>,</mo> <msub><mi>S</mi> <mn>2</mn></msub> <mo>,</mo> <msub><mi>A</mi> <mn>2</mn></msub> <mo>,</mo> <msub><mi>T</mi> <mn>2</mn></msub> <mo>,</mo> <msub><mi>B</mi> <mn>2</mn></msub> <mo>,</mo> <mo>...</mo></mrow></math>，其中下标表示时间步骤，如图 11-12 所示。

![](img/gdl2_1112.png)

###### 图 11-12。为巴赫赞美诗的前两小节创建网格标记化

然后，我们将训练我们的 Transformer 模型以预测给定先前令牌的下一个令牌。我们可以通过将序列在时间上以四个音符一组（每个声部一个）展开来将生成的序列解码回网格结构。尽管同一个音符经常被分割成多个令牌，并且在其他声部的令牌之间有令牌，但这种技术效果出奇地好。

然而，也存在一些缺点。首先，请注意，模型无法区分一个长音符和相同音高的两个较短相邻音符。这是因为标记化并没有明确编码音符的持续时间，只是在每个时间步是否存在音符。

其次，这种方法要求音乐具有可以分成合理大小块的规则节拍。例如，使用当前系统，我们无法编码三连音（在一个拍子中演奏的三个音符）。我们可以将音乐分成每个四分音符（四分音符）12 个步骤，而不是 4 个步骤，这将使表示相同音乐段所需的令牌数量增加三倍，增加训练过程的开销，并影响模型的回溯能力。

最后，我们不清楚如何将其他组件添加到标记化中，比如动态（每个声部的音乐是大声还是安静）或速度变化。我们被钢琴卷的二维网格结构所限制，这提供了一种方便的表示音高和节奏的方式，但不一定是一种容易融入使音乐听起来有趣的其他组件的方式。

### 基于事件的标记化

更灵活的方法是使用基于事件的令牌化。这可以被看作是一个词汇表，字面上描述了音乐是如何作为一系列事件创建的，使用丰富的令牌集。

例如，在图 11-13 中，我们使用三种类型的令牌：

+   `NOTE_ON<*音高*>`（开始播放给定音高的音符）

+   `NOTE_OFF<*音高*>`（停止播放给定音高的音符）

+   `TIME_SHIFT<*步骤*>`（按给定步骤向前移动时间）

这个词汇表可以用来创建一个描述音乐构造的序列，作为一组指令。

![](img/gdl2_1113.png)

###### 图 11-13。巴赫赞美诗第一小节的事件令牌化

我们可以轻松地将其他类型的令牌纳入这个词汇表中，以表示后续音符的动态和速度变化。通过使用`TIME_SHIFT<0.33>`令牌，这种方法还提供了一种在四分音符背景下生成三连音的方法。总的来说，这是一个更具表现力的令牌化框架，尽管对于 Transformer 来说，学习训练集音乐中固有模式可能更复杂，因为它在定义上比网格方法更少结构化。

###### 提示

我鼓励您尝试实施这些复调技术，并使用您在本书中迄今为止积累的所有知识在新的令牌化数据集上训练 Transformer。我还建议查看我们的 Tristan Behrens 博士关于音乐生成研究的指南，可在[GitHub](https://oreil.ly/YfaiJ)上找到，该指南提供了关于使用深度学习进行音乐生成的不同论文的全面概述。

在下一节中，我们将采用完全不同的方法来进行音乐生成，使用 GAN。# MuseGAN

您可能认为图 11-12 中显示的钢琴卷看起来有点像现代艺术品。这引发了一个问题——我们实际上是否可以将这个钢琴卷视为*图片*，并利用图像生成方法而不是序列生成技术？

正如我们将看到的，对于这个问题的答案是肯定的，我们可以直接将音乐生成视为图像生成问题。这意味着我们可以应用对图像生成问题非常有效的基于卷积的技术，特别是 GAN。

MuseGAN 是在 2017 年的论文“MuseGAN:用于符号音乐生成和伴奏的多轨序列生成对抗网络”中引入的。作者展示了通过一种新颖的 GAN 框架训练模型生成复调、多轨、多小节音乐是可能的。此外，他们展示了通过将喂给生成器的噪声向量的责任分解，他们能够对音乐的高级时间和基于轨道的特征进行精细控制。

让我们首先介绍 J.S.巴赫赞美诗数据集。

# 运行此示例的代码

此示例的代码可以在书籍存储库中的*notebooks/11_music/02_musegan/musegan.ipynb*中找到。

## 巴赫赞美诗数据集

要开始这个项目，您首先需要下载我们将用于训练 MuseGAN 的 MIDI 文件。我们将使用包含四声部的 229 首 J.S.巴赫赞美诗数据集。

您可以通过在书籍存储库中运行巴赫赞美诗数据集下载器脚本来下载数据集，如示例 11-5 所示。这将把 MIDI 文件保存到本地的*/data*文件夹中。

##### 示例 11-5。下载巴赫赞美诗数据集

```py
bash scripts/download_bach_chorale_data.sh
```

数据集由每个时间步长的四个数字数组组成：每个声部的 MIDI 音符音高。在这个数据集中，一个时间步长等于一个 16 分音符（半音符）。因此，例如，在 4 个四分音符拍的单个小节中，有 16 个时间步长。数据集会自动分成*训练*、*验证*和*测试*集。我们将使用*训练*数据集来训练 MuseGAN。

首先，我们需要将数据整理成正确的形状以供 GAN 使用。在这个示例中，我们将生成两小节音乐，因此我们将提取每个赞美诗的前两小节。每小节包括 16 个时间步长，四个声部中有潜在的 84 个音高。

###### 提示

从现在开始，声部将被称为*轨道*，以保持术语与原始论文一致。

因此，转换后的数据将具有以下形状：

```py
[BATCH_SIZE, N_BARS, N_STEPS_PER_BAR, N_PITCHES, N_TRACKS]
```

其中：

```py
BATCH_SIZE = 64
N_BARS = 2
N_STEPS_PER_BAR = 16
N_PITCHES = 84
N_TRACKS = 4
```

为了将数据整理成这种形状，我们将音高数字进行独热编码，转换为长度为 84 的向量，并将每个音符序列分成两小节，每小节包括 16 个时间步长。我们在这里做出的假设是数据集中的每个赞美诗每小节有四拍，这是合理的，即使不是这种情况，也不会对模型的训练产生不利影响。

图 11-14 展示了两小节原始数据如何转换为我们将用来训练 GAN 的转换后的钢琴卷帘数据集。

![](img/gdl2_1114.png)

###### 图 11-14。将两小节原始数据处理成我们可以用来训练 GAN 的钢琴卷帘数据`  `## MuseGAN 生成器

像所有 GAN 一样，MuseGAN 由一个生成器和一个评论家组成。生成器试图用其音乐创作愚弄评论家，评论家试图通过确保能够区分生成器伪造的巴赫赞美诗和真实的赞美诗来阻止这种情况发生。

MuseGAN 的不同之处在于生成器不仅接受单个噪声向量作为输入，而是有四个单独的输入，分别对应音乐的四个不同特征：和弦、风格、旋律和节奏。通过独立操纵这些输入中的每一个，我们可以改变生成音乐的高级属性。

生成器的高级视图显示在图 11-15 中。

![](img/gdl2_1115.png)

###### MuseGAN 生成器的高级图表

图表显示和弦和旋律输入首先通过一个*时间网络*，输出一个维度等于要生成的小节数的张量。风格和节奏输入不会以这种方式在时间上拉伸，因为它们在整个乐曲中保持不变。

然后，为了为特定轨道的特定小节生成特定小节，来自和弦、风格、旋律和节奏部分的相关输出被连接起来形成一个更长的向量。然后将其传递给小节生成器，最终输出指定轨道的指定小节。

通过连接所有轨道的生成小节，我们创建了一个可以与评论家的真实分数进行比较的分数。

让我们首先看看如何构建一个时间网络。

### 时间网络

时间网络的工作是将长度为`Z_DIM = 32`的单个输入噪声向量转换为每个小节的不同噪声向量（长度也为 32）的神经网络，由卷积转置层组成。构建这个网络的 Keras 代码在示例 11-6 中显示。

##### 示例 11-6。构建时间网络

```py
def conv_t(x, f, k, s, a, p, bn):
    x = layers.Conv2DTranspose(
                filters = f
                , kernel_size = k
                , padding = p
                , strides = s
                , kernel_initializer = initializer
                )(x)
    if bn:
        x = layers.BatchNormalization(momentum = 0.9)(x)

    x = layers.Activation(a)(x)
    return x

def TemporalNetwork():
    input_layer = layers.Input(shape=(Z_DIM,), name='temporal_input') ![1](img/1.png)
    x = layers.Reshape([1,1,Z_DIM])(input_layer) ![2](img/2.png)
    x = conv_t(
        x, f=1024, k=(2,1), s=(1,1), a = 'relu', p = 'valid', bn = True
    ) ![3](img/3.png)
    x = conv_t(
        x, f=Z_DIM, k=(N_BARS - 1,1), s=(1,1), a = 'relu', p = 'valid', bn = True
    )
    output_layer = layers.Reshape([N_BARS, Z_DIM])(x) ![4](img/4.png)
    return models.Model(input_layer, output_layer)
```

![1](img/#co_music_generation_CO1-1)

时间网络的输入是长度为 32 的向量（`Z_DIM`）。

![2](img/#co_music_generation_CO1-2)

我们将这个向量重塑为一个具有 32 个通道的 1×1 张量，以便我们可以对其应用二维卷积转置操作。

![3](img/#co_music_generation_CO1-3)

我们应用`Conv2DTranspose`层来沿一个轴扩展张量的大小，使其与`N_BARS`的长度相同。

![4](img/#co_music_generation_CO1-4)

我们使用 `Reshape` 层去除不必要的额外维度。

我们使用卷积操作而不是要求两个独立的向量进入网络的原因是，我们希望网络学习如何以一种一致的方式让一个小节跟随另一个小节。使用神经网络沿着时间轴扩展输入向量意味着模型有机会学习音乐如何跨越小节流动，而不是将每个小节视为完全独立于上一个的。

### 和弦、风格、旋律和 groove

现在让我们更仔细地看一下喂给生成器的四种不同输入：

和弦

和弦输入是一个长度为 `Z_DIM` 的单一噪声向量。这个向量的作用是控制音乐随时间的总体进展，跨越轨道共享，因此我们使用 `TemporalNetwork` 将这个单一向量转换为每个小节的不同潜在向量。请注意，虽然我们称这个输入为和弦，但它实际上可以控制音乐中每个小节变化的任何内容，比如一般的节奏风格，而不是特定于任何特定轨道。

风格

风格输入也是长度为 `Z_DIM` 的向量。这个向量在不经过转换的情况下传递，因此在所有小节和轨道上都是相同的。它可以被视为控制乐曲整体风格的向量（即，它会一致地影响所有小节和轨道）。

旋律

旋律输入是一个形状为 `[N_TRACKS, Z_DIM]` 的数组—也就是说，我们为每个轨道提供长度为 `Z_DIM` 的随机噪声向量。

这些向量中的每一个都通过轨道特定的 `TemporalNetwork`，其中轨道之间的权重不共享。输出是每个轨道的每个小节的长度为 `Z_DIM` 的向量。因此，模型可以使用这些输入向量来独立地微调每个小节和轨道的内容。

Groove

groove 输入也是一个形状为 `[N_TRACKS, Z_DIM]` 的数组，即每个轨道的长度为 `Z_DIM` 的随机噪声向量。与旋律输入不同，这些向量不通过时间网络，而是直接传递，就像风格向量一样。因此，每个 groove 向量将影响轨道的整体属性，跨越所有小节。

我们可以总结每个 MuseGAN 生成器组件的责任，如 表 11-1 所示。

表 11-1\. MuseGAN 生成器的组件

|  | 输出在小节之间不同吗？ | 输出在部分之间不同吗？ |
| --- | --- | --- |
| 风格 | Ｘ | Ｘ |
| Groove | Ｘ | ✓ |
| 和弦 | ✓ | Ｘ |
| 旋律 | ✓ | ✓ |

MuseGAN 生成器的最后一部分是 *小节生成器*—让我们看看如何使用它来将和弦、风格、旋律和 groove 组件的输出粘合在一起。

### 小节生成器

小节生成器接收四个潜在向量——来自和弦、风格、旋律和 groove 组件。这些被连接起来产生长度为 `4 * Z_DIM` 的输入向量。输出是单个轨道的单个小节的钢琴卷表示—即，形状为 `[1, n_steps_per_bar, n_pitches, 1]` 的张量。

小节生成器只是一个使用卷积转置层来扩展输入向量的时间和音高维度的神经网络。我们为每个轨道创建一个小节生成器，轨道之间的权重不共享。构建 `BarGenerator` 的 Keras 代码在 示例 11-7 中给出。

##### 示例 11-7\. 构建 `BarGenerator`

```py
def BarGenerator():

    input_layer = layers.Input(shape=(Z_DIM * 4,), name='bar_generator_input') ![1](img/1.png)

    x = layers.Dense(1024)(input_layer) ![2](img/2.png)
    x = layers.BatchNormalization(momentum = 0.9)(x)
    x = layers.Activation('relu')(x)
    x = layers.Reshape([2,1,512])(x)

    x = conv_t(x, f=512, k=(2,1), s=(2,1), a= 'relu',  p = 'same', bn = True) ![3](img/3.png)
    x = conv_t(x, f=256, k=(2,1), s=(2,1), a= 'relu', p = 'same', bn = True)
    x = conv_t(x, f=256, k=(2,1), s=(2,1), a= 'relu', p = 'same', bn = True)
    x = conv_t(x, f=256, k=(1,7), s=(1,7), a= 'relu', p = 'same', bn = True) ![4](img/4.png)
    x = conv_t(x, f=1, k=(1,12), s=(1,12), a= 'tanh', p = 'same', bn = False) ![5](img/5.png)

    output_layer = layers.Reshape([1, N_STEPS_PER_BAR , N_PITCHES ,1])(x) ![6](img/6.png)

    return models.Model(input_layer, output_layer)
```

![1](img/#co_music_generation_CO2-1)

bar 生成器的输入是长度为 `4 * Z_DIM` 的向量。

![2](img/#co_music_generation_CO2-2)

通过一个 `Dense` 层后，我们重新塑造张量以准备进行卷积转置操作。

![3](img/#co_music_generation_CO2-3)

首先我们沿着时间步长轴扩展张量…​

![4](img/#co_music_generation_CO2-4)

…​然后沿着音高轴。

![5](img/#co_music_generation_CO2-5)

最终层应用了 tanh 激活，因为我们将使用 WGAN-GP（需要 tanh 输出激活）来训练网络。

![6](img/#co_music_generation_CO2-6)

张量被重塑以添加两个大小为 1 的额外维度，以准备与其他小节和轨道连接。

### 将所有内容整合在一起

最终，MuseGAN 生成器接受四个输入噪声张量（和弦、风格、旋律和节奏），并将它们转换为一个多轨多小节乐谱。构建 MuseGAN 生成器的 Keras 代码在示例 11-8 中提供。

##### 示例 11-8。构建 MuseGAN 生成器

```py
def Generator():
    chords_input = layers.Input(shape=(Z_DIM,), name='chords_input') ![1](img/1.png)
    style_input = layers.Input(shape=(Z_DIM,), name='style_input')
    melody_input = layers.Input(shape=(N_TRACKS, Z_DIM), name='melody_input')
    groove_input = layers.Input(shape=(N_TRACKS, Z_DIM), name='groove_input')

    chords_tempNetwork = TemporalNetwork() ![2](img/2.png)
    chords_over_time = chords_tempNetwork(chords_input)

    melody_over_time = [None] * N_TRACKS
    melody_tempNetwork = [None] * N_TRACKS
    for track in range(N_TRACKS):
        melody_tempNetwork[track] = TemporalNetwork() ![3](img/3.png)
        melody_track = layers.Lambda(lambda x, track = track: x[:,track,:])(
            melody_input
        )
        melody_over_time[track] = melody_tempNetworktrack

    barGen = [None] * N_TRACKS
    for track in range(N_TRACKS):
        barGen[track] = BarGenerator() ![4](img/4.png)

    bars_output = [None] * N_BARS
    c = [None] * N_BARS
    for bar in range(N_BARS): ![5](img/5.png)
        track_output = [None] * N_TRACKS

        c[bar] = layers.Lambda(lambda x, bar = bar: x[:,bar,:])(chords_over_time)
        s = style_input

        for track in range(N_TRACKS):

            m = layers.Lambda(lambda x, bar = bar: x[:,bar,:])(
                melody_over_time[track]
            )
            g = layers.Lambda(lambda x, track = track: x[:,track,:])(
                groove_input
            )

            z_input = layers.Concatenate(
                axis = 1, name = 'total_input_bar_{}_track_{}'.format(bar, track)
            )([c[bar],s,m,g])

            track_output[track] = barGentrack

        bars_output[bar] = layers.Concatenate(axis = -1)(track_output)

    generator_output = layers.Concatenate(axis = 1, name = 'concat_bars')(
        bars_output
    ) ![6](img/6.png)

    return models.Model(
        [chords_input, style_input, melody_input, groove_input], generator_output
    ) ![7](img/7.png)

generator = Generator()
```

![1](img/#co_music_generation_CO3-1)

定义生成器的输入。

![2](img/#co_music_generation_CO3-2)

通过时间网络传递和弦输入。

![3](img/#co_music_generation_CO3-3)

通过时间网络传递旋律输入。

![4](img/#co_music_generation_CO3-4)

为每个轨道创建一个独立的小节生成器网络。

![5](img/#co_music_generation_CO3-5)

循环遍历轨道和小节，为每种组合创建一个生成的小节。

![6](img/#co_music_generation_CO3-6)

将所有内容连接在一起形成单个输出张量。

![7](img/#co_music_generation_CO3-7)

MuseGAN 模型接受四个不同的噪声张量作为输入，并输出一个生成的多轨多小节乐谱。

## MuseGAN 评论家

与生成器相比，评论家的架构要简单得多（这在 GAN 中经常是这样）。

评论家试图区分生成器创建的完整多轨多小节乐谱和巴赫赞美诗的真实节选。它是一个卷积神经网络，主要由将乐谱折叠成单个输出预测的`Conv3D`层组成。

# Conv3D 层

到目前为止，在本书中，我们只使用了适用于三维输入图像（宽度、高度、通道）的`Conv2D`层。在这里，我们必须使用`Conv3D`层，它们类似于`Conv2D`层，但接受四维输入张量（`n_bars`、`n_steps_per_bar`、`n_pitches`、`n_tracks`）。

我们在评论家中不使用批量归一化层，因为我们将使用 WGAN-GP 框架来训练 GAN，这是不允许的。

构建评论家的 Keras 代码在示例 11-9 中给出。

##### 示例 11-9。构建 MuseGAN 评论家

```py
def conv(x, f, k, s, p):
    x = layers.Conv3D(filters = f
                , kernel_size = k
                , padding = p
                , strides = s
                , kernel_initializer = initializer
                )(x)
    x = layers.LeakyReLU()(x)
    return x

def Critic():
    critic_input = layers.Input(
        shape=(N_BARS, N_STEPS_PER_BAR, N_PITCHES, N_TRACKS),
        name='critic_input'
    ) ![1](img/1.png)

    x = critic_input
    x = conv(x, f=128, k = (2,1,1), s = (1,1,1), p = 'valid') ![2](img/2.png)
    x = conv(x, f=128, k = (N_BARS - 1,1,1), s = (1,1,1), p = 'valid')

    x = conv(x, f=128, k = (1,1,12), s = (1,1,12), p = 'same') ![3](img/3.png)
    x = conv(x, f=128, k = (1,1,7), s = (1,1,7), p = 'same')

    x = conv(x, f=128, k = (1,2,1), s = (1,2,1), p = 'same') ![4](img/4.png)
    x = conv(x, f=128, k = (1,2,1), s = (1,2,1), p = 'same')
    x = conv(x, f=256, k = (1,4,1), s = (1,2,1), p = 'same')
    x = conv(x, f=512, k = (1,3,1), s = (1,2,1), p = 'same')

    x = layers.Flatten()(x)

    x = layers.Dense(1024, kernel_initializer = initializer)(x)
    x = layers.LeakyReLU()(x)

    critic_output = layers.Dense(
        1, activation=None, kernel_initializer = initializer
    )(x) ![5](img/5.png)

    return models.Model(critic_input, critic_output)

critic = Critic()
```

![1](img/#co_music_generation_CO4-1)

评论家的输入是一个多轨多小节乐谱数组，每个形状为`[N_BARS, N_STEPS_PER_BAR, N_PITCHES, N_TRACKS]`。

![2](img/#co_music_generation_CO4-2)

首先，我们沿着小节轴折叠张量。由于我们使用的是 4D 张量，所以在评论家中应用`Conv3D`层。

![3](img/#co_music_generation_CO4-3)

接下来，我们沿着音高轴折叠张量。

![4](img/#co_music_generation_CO4-4)

最后，我们沿着时间步轴折叠张量。

![5](img/#co_music_generation_CO4-5)

输出是一个具有单个单元且没有激活函数的`Dense`层，这是 WGAN-GP 框架所需的。

## MuseGAN 的分析

我们可以通过生成一个乐谱，然后调整一些输入噪声参数来查看对输出的影响来进行一些实验。

生成器的输出是一个值范围在[-1, 1]的数组（由于最终层的 tanh 激活函数）。为了将其转换为每个轨道的单个音符，我们选择每个时间步的所有 84 个音高中具有最大值的音符。在原始的 MuseGAN 论文中，作者使用了阈值 0，因为每个轨道可以包含多个音符；然而，在这种情况下，我们可以简单地取最大值来确保每个时间步每个轨道恰好有一个音符，这与巴赫赞美诗的情况相同。

图 11-16 显示了模型从随机正态分布的噪声向量生成的乐谱（左上角）。我们可以通过欧几里德距离找到数据集中最接近的乐谱，并检查我们生成的乐谱是否是数据集中已经存在的音乐片段的副本——最接近的乐谱显示在其正下方，我们可以看到它与我们生成的乐谱并不相似。

现在让我们玩弄输入噪声来微调我们生成的乐谱。首先，我们可以尝试改变和弦噪声向量——图 11-16 中左下角的乐谱显示了结果。我们可以看到每个轨道都已经改变，正如预期的那样，而且两个小节展现出不同的特性。在第二小节中，贝斯线更加动态，顶部音乐线的音高比第一小节更高。这是因为影响两个小节的潜在向量是不同的，因为输入和弦向量通过了一个时间网络。

###### 图 11-16。MuseGAN 预测乐谱的示例，显示训练数据中最接近的真实乐谱以及通过改变输入噪声而影响生成乐谱的情况

# 总结

当我们改变风格向量（右上角）时，两个小节以类似的方式改变。整个乐段的风格已经从原始生成的乐谱中改变，以一种一致的方式（即，相同的潜在向量被用来调整所有轨道和小节）。

我们还可以通过旋律和节奏输入单独改变轨道。在图 11-16 中间右侧的乐谱中，我们可以看到仅改变顶部音乐线的旋律噪声输入的效果。所有其他部分保持不变，但顶部音符发生了显著变化。此外，我们可以看到顶部音乐线两个小节之间的节奏变化：第二小节比第一小节更动态，包含比第一小节更快的音符。

最后，图中右下角的乐谱显示了当我们仅改变贝斯的节奏输入参数时预测的乐谱。同样，所有其他部分保持不变，但贝斯部分是不同的。此外，贝斯的整体模式在小节之间保持相似，这是我们所期望的。

这展示了如何每个输入参数可以直接影响生成的音乐序列的高级特征，就像我们之前能够调整 VAE 和 GAN 的潜在向量以改变生成图像的外观一样。模型的一个缺点是必须预先指定要生成的小节数。为了解决这个问题，作者展示了模型的一个扩展，允许将先前的小节作为输入馈入，使模型能够通过不断将最近预测的小节作为额外输入来生成长形乐谱。

在本章中，我们探讨了两种不同类型的音乐生成模型：Transformer 和 MuseGAN。

Transformer 的设计类似于我们在第九章中看到的用于文本生成的网络。音乐和文本生成有很多共同点，通常可以同时用于两者的类似技术。我们通过将两个输入和输出流（音符和持续时间）纳入 Transformer 架构来扩展了 Transformer 架构。我们看到模型能够通过准确生成巴赫音乐来学习关于调式和音阶等概念。

我们还探讨了如何调整标记化过程以处理多声部（多轨）音乐生成。网格标记化将乐谱的钢琴卷表示序列化，使我们能够在描述每个音轨中存在哪个音符的令牌的单个流上训练 Transformer，在离散的、等间隔的时间步长间隔内。基于事件的标记化产生了一个*配方*，描述了如何以顺序方式创建多行音乐，通过一系列指令的单个流。这两种方法都有优缺点——Transformer 基于的音乐生成方法的成功或失败往往严重依赖于标记化方法的选择。

我们还看到生成音乐并不总是需要顺序方法——MuseGAN 使用卷积来生成具有多轨的多声部乐谱，将乐谱视为图像，其中轨道是图像的各个通道。MuseGAN 的新颖之处在于四个输入噪声向量（和弦、风格、旋律和节奏）的组织方式，使得可以对音乐的高级特征保持完全控制。虽然底层的和声仍然不像巴赫的那样完美或多样化，但这是对一个极其难以掌握的问题的良好尝试，并突显了 GAN 处理各种问题的能力。

¹ 黄成志安娜等人，“音乐 Transformer：生成具有长期结构的音乐”，2018 年 9 月 12 日，[*https://arxiv.org/abs/1809.04281*](https://arxiv.org/abs/1809.04281)。

² 董浩文等人，“MuseGAN：用于符号音乐生成和伴奏的多轨序列生成对抗网络”，2017 年 9 月 19 日，[*https://arxiv.org/abs/1709.06298*](https://arxiv.org/abs/1709.06298)。
