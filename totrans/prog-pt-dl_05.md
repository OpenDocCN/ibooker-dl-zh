# 第五章：文本分类

我们暂时离开图像，转而关注另一个领域，深度学习在传统技术上已经被证明是一个重大进步的地方：自然语言处理（NLP）。一个很好的例子是谷歌翻译。最初，处理翻译的代码有 500,000 行代码。基于 TensorFlow 的新系统大约有 500 行代码，性能比旧方法更好。

最近的突破还发生在将迁移学习（你在第四章中学到的）引入到 NLP 问题中。新的架构，如 Transformer 架构，已经导致了像 OpenAI 的 GPT-2 这样的网络的创建，其更大的变体产生的文本几乎具有人类般的质量（事实上，OpenAI 没有发布这个模型的权重，因为担心它被恶意使用）。

本章提供了循环神经网络和嵌入的快速介绍。然后我们探讨了`torchtext`库以及如何使用基于 LSTM 的模型进行文本处理。

# 循环神经网络

如果我们回顾一下迄今为止我们如何使用基于 CNN 的架构，我们可以看到它们总是在一个完整的时间快照上工作。但考虑这两个句子片段：

```py
The cat sat on the mat.

She got up and impatiently climbed on the chair, meowing for food.
```

假设你将这两个句子一个接一个地馈送到 CNN 中，并问，“猫在哪里？”你会遇到问题，因为网络没有“记忆”的概念。当处理具有时间域的数据时（例如文本、语音、视频和时间序列数据）这是非常重要的。通过*循环神经网络*（RNNs）通过*隐藏状态*给神经网络提供了记忆来解决这个问题。

RNN 是什么样子的？我最喜欢的解释是，“想象一个神经网络和一个`for`循环相交。” 图 5-1 显示了一个经典 RNN 结构的图表。

![经典 RNN 图](img/ppdl_0501.png)

###### 图 5-1。一个 RNN

我们在时间步*t*添加输入，得到*ht*的*隐藏*输出状态，并且输出也被馈送回 RNN 用于下一个时间步。我们可以展开这个网络，深入了解正在发生的事情，如图 5-2 所示。

！展开的 RNN 图

###### 图 5-2。一个展开的 RNN

我们这里有一组全连接层（具有共享参数）、一系列输入和输出。输入数据被馈送到网络中，然后预测输出作为序列中的下一个项目。在展开的视图中，我们可以看到 RNN 可以被看作是一系列全连接层的管道，连续的输入被馈送到序列中的下一层（在层之间插入了通常的非线性，如`ReLU`）。当我们完成了预测的序列后，我们必须通过 RNN 将错误反向传播回去。因为这涉及到通过网络的步骤向后走，这个过程被称为通过时间的反向传播。错误是在整个序列上计算的，然后网络像图 5-2 中展开一样，为每个时间步计算梯度并组合以更新网络的共享参数。你可以想象它是在单独的网络上进行反向传播，然后将所有梯度相加。

这就是 RNN 的理论。但这种简单的结构存在问题，我们需要讨论如何通过更新的架构来克服这些问题。

# 长短期记忆网络

实际上，RNN 在*梯度消失*问题上特别容易受到影响，我们在第二章中讨论过，或者更糟糕的情况是*梯度爆炸*，其中你的错误趋向于无穷大。这两种情况都不好，因此 RNN 无法解决许多它们被认为适合的问题。这一切在 1997 年 Sepp Hochreiter 和 Jürgen Schmidhuber 引入长短期记忆（LSTM）变体的时候发生了改变。

图 5-3 展示了一个 LSTM 层。我知道，这里有很多东西，但并不太复杂。真诚地说。

![LSTM 图示](img/ppdl_0503.png)

###### 图 5-3\. 一个 LSTM

好的，我承认，这确实令人生畏。关键是要考虑三个门（输入、输出和遗忘）。在标准的 RNN 中，我们会永远“记住”一切。但这并不是我们大脑的工作方式（可悲！），LSTM 的遗忘门允许我们建模这样一个想法，即随着我们在输入链中继续，链的开始变得不那么重要。LSTM 忘记多少是在训练过程中学习的，因此，如果对网络来说最好是非常健忘的，遗忘门参数就会这样做。

*单元*最终成为网络层的“记忆”，而输入、输出和遗忘门将决定数据如何在层中流动。数据可能只是通过，它可能“写入”到单元中，这些数据可能（或可能不会！）通过输出门被传递到下一层，并受到输出门的影响。

这些部分的组合足以解决梯度消失问题，并且还具有图灵完备性，因此理论上，你可以使用其中之一进行计算机上的任何计算。

当然，事情并没有止步于此。自 LSTM 以来，RNN 领域发生了几项发展，我们将在接下来的部分中涵盖一些主要内容。

## 门控循环单元

自 1997 年以来，已经创建了许多基本 LSTM 网络的变体，其中大多数你可能不需要了解，除非你感兴趣。然而，2014 年出现的一个变体，门控循环单元（GRU），值得了解，因为在某些领域它变得非常流行。图 5-4 展示了一个 GRU 架构的组成。

![GRU 图示](img/ppdl_0504.png)

###### 图 5-4\. 一个 GRU

主要的要点是 GRU 已经将遗忘门与输出门合并。这意味着它比 LSTM 具有更少的参数，因此倾向于更快地训练并在运行时使用更少的资源。出于这些原因，以及它们基本上是 LSTM 的替代品，它们变得非常流行。然而，严格来说，它们比 LSTM 弱一些，因为合并了遗忘和输出门，所以一般我建议在网络中尝试使用 GRU 或 LSTM，并看看哪个表现更好。或者只是接受 LSTM 在训练时可能会慢一些，但最终可能是最佳选择。你不必追随最新的潮流——真诚地说！

## 双向 LSTM

LSTM 的另一个常见变体是*双向* LSTM 或简称*biLSTM*。正如你目前所看到的，传统的 LSTM（以及 RNN 总体）在训练过程中可以查看过去并做出决策。不幸的是，有时候你也需要看到未来。这在翻译和手写识别等应用中尤为重要，因为当前状态之后发生的事情可能与之前状态一样重要，以确定输出。

双向 LSTM 以最简单的方式解决了这个问题：它本质上是两个堆叠的 LSTM，其中输入在一个 LSTM 中向前发送，而在第二个 LSTM 中向后发送。图 5-5 展示了双向 LSTM 如何双向跨越其输入以产生输出。

![双向 LSTM 图示](img/ppdl_0505.png)

###### 图 5-5\. 一个双向 LSTM

PyTorch 通过在创建`LSTM()`单元时传入`bidirectional=True`参数来轻松创建双向 LSTM，您将在本章后面看到。

这完成了我们对基于 RNN 的架构的介绍。在第九章中，当我们研究基于 Transformer 的 BERT 和 GPT-2 模型时，我们将回到架构问题。

# 嵌入

我们几乎可以开始编写一些代码了！但在此之前，您可能会想到一个小细节：我们如何在网络中表示单词？毕竟，我们正在将数字张量输入到网络中，并获得张量输出。对于图像，将它们转换为表示红/绿/蓝分量值的张量似乎是一件很明显的事情，因为它们已经自然地被认为是数组，因为它们带有高度和宽度。但是单词？句子？这将如何运作？

最简单的方法仍然是许多自然语言处理方法中常见的方法之一，称为*one-hot 编码*。这很简单！让我们从本章开头的第一句话开始看：

```py
The cat sat on the mat.
```

如果我们考虑这是我们世界的整个词汇表，我们有一个张量`[the, cat, sat, on, mat]`。one-hot 编码简单地意味着我们创建一个与词汇表大小相同的向量，并为其中的每个单词分配一个向量，其中一个参数设置为 1，其余设置为 0：

```py
the — [1 0 0 0 0]
cat — [0 1 0 0 0]
sat — [0 0 1 0 0]
on  — [0 0 0 1 0]
mat — [0 0 0 0 1]
```

我们现在已经将单词转换为向量，可以将它们输入到我们的网络中。此外，我们可以向我们的词汇表中添加额外的符号，比如`UNK`（未知，用于不在词汇表中的单词）和`START/STOP`来表示句子的开头和结尾。

当我们在示例词汇表中添加另一个单词时，one-hot 编码会显示出一些限制：*kitty*。根据我们的编码方案，*kitty*将由`[0 0 0 0 0 1]`表示（其他向量都用零填充）。首先，您可以看到，如果我们要建模一个现实的单词集，我们的向量将非常长，几乎没有信息。其次，也许更重要的是，我们知道*猫*和*小猫*之间存在*非常强烈*的关系（还有*该死*，但幸运的是在我们的词汇表中被跳过了！），这是无法用 one-hot 编码表示的；这两个单词是完全不同的东西。

最近变得更受欢迎的一种方法是用*嵌入矩阵*替换 one-hot 编码（当然，one-hot 编码本身就是一个嵌入矩阵，只是不包含有关单词之间关系的任何信息）。这个想法是将向量空间的维度压缩到更易管理的尺寸，并利用空间本身的优势。

例如，如果我们在 2D 空间中有一个嵌入，也许*cat*可以用张量`[0.56, 0.45]`表示，*kitten*可以用`[0.56, 0.445]`表示，而*mat*可以用`[0.2, -0.1]`表示。我们在向量空间中将相似的单词聚集在一起，并可以使用欧几里得或余弦距离函数进行距离检查，以确定单词之间的接近程度。那么我们如何确定单词在向量空间中的位置呢？嵌入层与您迄今在构建神经网络中看到的任何其他层没有区别；我们随机初始化向量空间，希望训练过程更新参数，使相似的单词或概念相互靠近。

嵌入向量的一个著名例子是*word2vec*，它是由 Google 在 2013 年发布的。这是使用浅层神经网络训练的一组词嵌入，它揭示了向量空间转换似乎捕捉到了有关支撑单词的概念的一些内容。在其常被引用的发现中，如果您提取*King*、*Man*和*Woman*的向量，然后从*King*中减去*Man*的向量并加上*Woman*的向量，您将得到一个代表*Queen*的向量表示。自从*word2vec*以来，其他预训练的嵌入也已经可用，如*ELMo*、*GloVe*和*fasttext*。

至于在 PyTorch 中使用嵌入，非常简单：

```py
embed = nn.Embedding(vocab_size, dimension_size)
```

这将包含一个`vocab_size` x `dimension_size`的张量，随机初始化。我更喜欢认为它只是一个巨大的数组或查找表。您词汇表中的每个单词索引到一个大小为`dimension_size`的向量条目，所以如果我们回到我们的猫及其在垫子上的史诗般的冒险，我们会得到这样的东西：

```py
cat_mat_embed = nn.Embedding(5, 2)
cat_tensor = Tensor([1])
cat_mat_embed.forward(cat_tensor)

> tensor([[ 1.7793, -0.3127]], grad_fn=<EmbeddingBackward>)
```

我们创建了我们的嵌入，一个包含*cat*在我们词汇表中位置的张量，并通过层的`forward()`方法传递它。这给了我们我们的随机嵌入。结果还指出，我们有一个梯度函数，我们可以在将其与损失函数结合后用于更新参数。

我们现在已经学习了所有的理论，可以开始构建一些东西了！

# torchtext

就像`torchvision`一样，PyTorch 提供了一个官方库`torchtext`，用于处理文本处理管道。然而，`torchtext`并没有像`torchvision`那样经过充分测试，也没有像`torchvision`那样受到很多关注，这意味着它不太容易使用或文档不够完善。但它仍然是一个强大的库，可以处理构建基于文本的数据集的许多琐碎工作，所以我们将在本章的其余部分中使用它。

安装`torchtext`相当简单。您可以使用标准的`pip`：

```py
pip install torchtext
```

或者特定的`conda`渠道：

```py
conda install -c derickl torchtext
```

您还需要安装*spaCy*（一个 NLP 库）和 pandas，如果您的系统上没有它们的话（再次使用`pip`或`conda`）。我们使用*spaCy*来处理`torchtext`管道中的文本，使用 pandas 来探索和清理我们的数据。

## 获取我们的数据：推文！

在这一部分，我们构建一个情感分析模型，所以让我们获取一个数据集。`torchtext`通过`torchtext.datasets`模块提供了一堆内置数据集，但我们将从头开始工作，以便了解构建自定义数据集并将其馈送到我们创建的模型的感觉。我们使用[Sentiment140 数据集](http://help.sentiment140.com/for-students)。这是基于 Twitter 上的推文，每个推文被排名为 0 表示负面，2 表示中性，4 表示积极。

下载 zip 存档并解压。我们使用文件*training.1600000.processed.noemoticon.csv*。让我们使用 pandas 查看文件：

```py
import pandas as pd
tweetsDF = pd.read_csv("training.1600000.processed.noemoticon.csv",
                        header=None)
```

此时您可能会遇到这样的错误：

```py
UnicodeDecodeError: 'utf-8' codec can't decode bytes in
position 80-81: invalid continuation byte
```

恭喜你，现在你是一个真正的数据科学家，你可以处理数据清洗了！从错误消息中可以看出，pandas 使用的默认基于 C 的 CSV 解析器不喜欢文件中的一些 Unicode，所以我们需要切换到基于 Python 的解析器：

```py
tweetsDF = pd.read_csv("training.1600000.processed.noemoticon.csv",
engine="python", header=None)
```

让我们通过显示前五行来查看数据的结构：

```py
>>> tweetDF.head(5)
0  0  1467810672  ...  NO_QUERY   scotthamilton  is upset that ...
1  0  1467810917  ...  NO_QUERY        mattycus  @Kenichan I dived many times ...
2  0  1467811184  ...  NO_QUERY         ElleCTF    my whole body feels itchy
3  0  1467811193  ...  NO_QUERY          Karoli  @nationwideclass no, it's ...
4  0  1467811372  ...  NO_QUERY        joy_wolf  @Kwesidei not the whole crew
```

令人恼火的是，这个 CSV 中没有标题字段（再次欢迎来到数据科学家的世界！），但通过查看网站并运用我们的直觉，我们可以看到我们感兴趣的是最后一列（推文文本）和第一列（我们的标签）。然而，标签不是很好，所以让我们做一些特征工程来解决这个问题。让我们看看我们的训练集中有哪些计数：

```py
>>> tweetsDF[0].value_counts()
4    800000
0    800000
Name: 0, dtype: int64
```

有趣的是，在训练数据集中没有中性值。这意味着我们可以将问题制定为 0 和 1 之间的二元选择，并从中得出我们的预测，但目前我们坚持原始计划，即未来可能会有中性推文。为了将类别编码为从 0 开始的数字，我们首先从标签列创建一个`category`类型的列：

```py
tweetsDF["sentiment_cat"] = tweetsDF[0].astype('category')
```

然后我们将这些类别编码为另一列中的数字信息：

```py
tweetsDF["sentiment"] = tweetsDF["sentiment_cat"].cat.codes
```

然后我们将修改后的 CSV 保存回磁盘：

```py
tweetsDF.to_csv("train-processed.csv", header=None, index=None)
```

我建议您保存另一个 CSV 文件，其中包含 160 万条推文的小样本，供您进行测试：

```py
tweetsDF.sample(10000).to_csv("train-processed-sample.csv", header=None,
    index=None)
```

现在我们需要告诉`torchtext`我们认为对于创建数据集而言重要的内容。

## 定义字段

`torchtext`采用了一种直接的方法来生成数据集：您告诉它您想要什么，它将为您处理原始 CSV（或 JSON）数据。您首先通过定义*字段*来实现这一点。`Field`类有相当多的参数可以分配给它，尽管您可能不会同时使用所有这些参数，但表 5-1 提供了一个方便的指南，说明您可以使用`Field`做什么。

表 5-1\. 字段参数类型

| 参数 | 描述 | 默认值 |
| --- | --- | --- |
| `sequential` | 字段是否表示序列数据（即文本）。如果设置为`False`，则不会应用标记化。 | `True` |
| `use_vocab` | 是否包含`Vocab`对象。如果设置为`False`，字段应包含数字数据。 | `True` |
| `init_token` | 将添加到此字段开头以指示数据开始的令牌。 | `None` |
| `eos_token` | 附加到每个序列末尾的句子结束令牌。 | `None` |
| `fix_length` | 如果设置为整数，所有条目将填充到此长度。如果为`None`，序列长度将是灵活的。 | `None` |
| `dtype` | 张量批次的类型。 | `torch.long` |
| `lower` | 将序列转换为小写。 | `False` |
| `tokenize` | 将执行序列标记化的函数。如果设置为`spacy`，将使用 spaCy 分词器。 | `string.split` |
| `pad_token` | 将用作填充的令牌。 | `<pad>` |
| `unk_token` | 用于表示`Vocab` `dict`中不存在的单词的令牌。 | `<unk>` |
| `pad_first` | 在序列开始处填充。 | `False` |
| `truncate_first` | 在序列开头截断（如果需要）。 | `False` |

正如我们所指出的，我们只对标签和推文文本感兴趣。我们通过使用`Field`数据类型来定义这些内容：

```py
from torchtext import data

LABEL = data.LabelField()
TWEET = data.Field(tokenize='spacy', lower=true)
```

我们将`LABEL`定义为`LabelField`，它是`Field`的子类，将`sequential`设置为`False`（因为它是我们的数字类别）。`TWEET`是一个标准的`Field`对象，我们决定使用 spaCy 分词器并将所有文本转换为小写，但在其他方面，我们使用前面表中列出的默认值。如果在运行此示例时，构建词汇表的步骤花费了很长时间，请尝试删除`tokenize`参数并重新运行。这将使用默认值，即简单地按空格分割，这将大大加快标记化步骤，尽管创建的词汇表不如 spaCy 创建的那么好。

定义了这些字段后，我们现在需要生成一个列表，将它们映射到 CSV 中的行列表：

```py
 fields = [('score',None), ('id',None),('date',None),('query',None),
      ('name',None),
      ('tweet', TWEET),('category',None),('label',LABEL)]
```

有了我们声明的字段，我们现在使用`TabularDataset`将该定义应用于 CSV：

```py
twitterDataset = torchtext.data.TabularDataset(
        path="training-processed.csv",
        format="CSV",
        fields=fields,
        skip_header=False)
```

这可能需要一些时间，特别是使用 spaCy 解析器。最后，我们可以使用`split()`方法将其拆分为训练、测试和验证集：

```py
(train, test, valid) = twitterDataset.split(split_ratio=[0.8,0.1,0.1])

(len(train),len(test),len(valid))
> (1280000, 160000, 160000)
```

以下是从数据集中提取的示例：

```py
>vars(train.examples[7])

{'label': '6681',
 'tweet': ['woah',
  ',',
  'hell',
  'in',
  'chapel',
  'thrill',
  'is',
  'closed',
  '.',
  'no',
  'more',
  'sweaty',
  'basement',
  'dance',
  'parties',
  '?',
  '?']}
```

在一个令人惊讶的巧合中，随机选择的推文提到了我经常访问的教堂山俱乐部的关闭。看看您在数据中的浏览中是否发现了任何奇怪的事情！

## 构建词汇表

传统上，在这一点上，我们将构建数据集中每个单词的独热编码——这是一个相当乏味的过程。幸运的是，`torchtext`会为我们做这个工作，并且还允许传入一个`max_size`参数来限制词汇表中最常见的单词。通常这样做是为了防止构建一个巨大的、占用内存的模型。毕竟，我们不希望我们的 GPU 被压倒。让我们将词汇表限制在训练集中最多 20,000 个单词：

```py
vocab_size = 20000
TWEET.build_vocab(train, max_size = vocab_size)
```

然后我们可以查询`vocab`类实例对象，以发现关于我们数据集的一些信息。首先，我们问传统的“我们的词汇量有多大？”：

```py
len(TWEET.vocab)
> 20002
```

等等，*等等，什么？* 是的，我们指定了 20,000，但默认情况下，`torchtext`会添加两个特殊的标记，`<unk>`表示未知单词（例如，那些被我们指定的 20,000 `max_size`截断的单词），以及`<pad>`，一个填充标记，将用于将所有文本填充到大致相同的大小，以帮助在 GPU 上进行有效的批处理（请记住，GPU 的速度来自于对常规批次的操作）。当您声明一个字段时，您还可以指定`eos_token`或`init_token`符号，但它们不是默认包含的。

现在让我们来看看词汇表中最常见的单词：

```py
>TWEET.vocab.freqs.most_common(10)
[('!', 44802),
 ('.', 40088),
 ('I', 33133),
 (' ', 29484),
 ('to', 28024),
 ('the', 24389),
 (',', 23951),
('a', 18366),
 ('i', 17189),
('and', 14252)]
```

基本上符合您的预期，因为我们的 spaCy 分词器没有去除停用词。（因为它只有 140 个字符，如果我们去除了停用词，我们的模型将丢失太多信息。）

我们几乎已经完成了我们的数据集。我们只需要创建一个数据加载器来输入到我们的训练循环中。`torchtext`提供了`BucketIterator`方法，它将生成一个称为`Batch`的东西，几乎与我们在图像上使用的数据加载器相同，但又有所不同。（很快您将看到，我们必须更新我们的训练循环来处理`Batch`接口的一些奇怪之处。）

```py
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
(train, valid, test),
batch_size = 32,
device = device)
```

将所有内容放在一起，这是构建我们数据集的完整代码：

```py
from torchtext import data

device = "cuda"
LABEL = data.LabelField()
TWEET = data.Field(tokenize='spacy', lower=true)

fields = [('score',None), ('id',None),('date',None),('query',None),
      ('name',None),
      ('tweet', TWEET),('category',None),('label',LABEL)]

twitterDataset = torchtext.data.TabularDataset(
        path="training-processed.csv",
        format="CSV",
        fields=fields,
        skip_header=False)

(train, test, valid) = twitterDataset.split(split_ratio=[0.8,0.1,0.1])

vocab_size = 20002
TWEET.build_vocab(train, max_size = vocab_size)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
(train, valid, test),
batch_size = 32,
device = device)
```

有了我们的数据处理完成，我们可以继续定义我们的模型。

## 创建我们的模型

我们在 PyTorch 中使用了我们在本章前半部分讨论过的`Embedding`和`LSTM`模块来构建一个简单的推文分类模型：

```py
import torch.nn as nn

class OurFirstLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size):
        super(OurFirstLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim,
                hidden_size=hidden_size, num_layers=1)
        self.predictor = nn.Linear(hidden_size, 2)

    def forward(self, seq):
        output, (hidden,_) = self.encoder(self.embedding(seq))
        preds = self.predictor(hidden.squeeze(0))
        return preds

model = OurFirstLSTM(100,300, 20002)
model.to(device)
```

在这个模型中，我们所做的就是创建三个层。首先，我们的推文中的单词被推送到一个`Embedding`层中，我们已经将其建立为一个 300 维的向量嵌入。然后将其输入到一个具有 100 个隐藏特征的`LSTM`中（同样，我们正在从 300 维的输入中进行压缩，就像我们在处理图像时所做的那样）。最后，LSTM 的输出（处理传入推文后的最终隐藏状态）被推送到一个标准的全连接层中，有三个输出对应于我们的三个可能的类别（负面、积极或中性）。接下来我们转向训练循环！

## 更新训练循环

由于一些`torchtext`的怪癖，我们需要编写一个稍微修改过的训练循环。首先，我们创建一个优化器（通常我们使用 Adam）和一个损失函数。因为对于每个推文，我们有三个潜在的类别，所以我们使用`CrossEntropyLoss()`作为我们的损失函数。然而，事实证明数据集中只有两个类别；如果我们假设只有两个类别，我们实际上可以改变模型的输出，使其产生一个介于 0 和 1 之间的单个数字，然后使用二元交叉熵（BCE）损失（我们可以将将输出压缩在 0 和 1 之间的 sigmoid 层和 BCE 层组合成一个单一的 PyTorch 损失函数，`BCEWithLogitsLoss()`）。我提到这一点是因为如果您正在编写一个必须始终处于一种状态或另一种状态的分类器，那么它比我们即将使用的标准交叉熵损失更合适。

```py
optimizer = optim.Adam(model.parameters(), lr=2e-2)
criterion = nn.CrossEntropyLoss()

def train(epochs, model, optimizer, criterion, train_iterator, valid_iterator):
    for epoch in range(1, epochs + 1):

        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(train_iterator):
            opt.zero_grad()
            predict = model(batch.tweet)
            loss = criterion(predict,batch.label)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * batch.tweet.size(0)
        training_loss /= len(train_iterator)

        model.eval()
        for batch_idx,batch in enumerate(valid_iterator):
            predict = model(batch.tweet)
            loss = criterion(predict,batch.label)
            valid_loss += loss.data.item() * x.size(0)

        valid_loss /= len(valid_iterator)
        print('Epoch: {}, Training Loss: {:.2f},
        Validation Loss: {:.2f}'.format(epoch, training_loss, valid_loss))
```

在这个新的训练循环中需要注意的主要事项是，我们必须引用`batch.tweet`和`batch.label`来获取我们感兴趣的特定字段；它们并不像在`torchvision`中那样从枚举器中很好地脱落出来。

一旦我们使用这个函数训练了我们的模型，我们就可以用它来对一些推文进行简单的情感分析。

## 分类推文

`torchtext`的另一个麻烦是它有点难以预测事物。你可以模拟内部发生的处理流程，并在该流程的输出上进行所需的预测，如这个小函数所示：

```py
def classify_tweet(tweet):
    categories = {0: "Negative", 1:"Positive"}
    processed = TWEET.process([TWEET.preprocess(tweet)])
    return categories[model(processed).argmax().item()]
```

我们必须调用`preprocess()`，它执行基于 spaCy 的标记化。之后，我们可以调用`process()`将标记基于我们已构建的词汇表转换为张量。我们唯一需要注意的是`torchtext`期望一批字符串，因此在将其传递给处理函数之前，我们必须将其转换为列表的列表。然后我们将其馈送到模型中。这将产生一个如下所示的张量：

```py
tensor([[ 0.7828, -0.0024]]
```

具有最高值的张量元素对应于模型选择的类别，因此我们使用`argmax()`来获取该索引，然后使用`item()`将零维张量转换为 Python 整数，然后将其索引到我们的`categories`字典中。

训练完我们的模型后，让我们看看如何执行你在第 2–4 章中学到的其他技巧和技术。

# 数据增强

你可能会想知道如何增强文本数据。毕竟，你不能像处理图像那样水平翻转文本！但你可以使用一些文本技术，为模型提供更多训练信息。首先，你可以用同义词替换句子中的单词，如下所示：

```py
The cat sat on the mat
```

可以变成

```py
The cat sat on the rug
```

除了猫坚持认为地毯比垫子更柔软之外，句子的含义并没有改变。但是*mat*和*rug*将映射到词汇表中的不同索引，因此模型将学习到这两个句子映射到相同标签，并希望这两个单词之间存在联系，因为句子中的其他内容都是相同的。

2019 年初，论文“EDA：用于提高文本分类任务性能的简单数据增强技术”提出了另外三种增强策略：随机插入、随机交换和随机删除。让我们看看每种方法。³

## 随机插入

*随机插入*技术查看一个句子，然后随机插入现有非停用词的同义词*n*次。假设你有一种获取单词同义词和消除停用词（常见单词如*and*、*it*、*the*等）的方法，通过`get_synonyms()`和`get_stopwords()`在这个函数中显示，但没有实现，这个实现如下：

```py
def random_insertion(sentence,n):
    words = remove_stopwords(sentence)
    for _ in range(n):
        new_synonym = get_synonyms(random.choice(words))
        sentence.insert(randrange(len(sentence)+1), new_synonym)
    return sentence
```

在实践中，替换`cat`的示例可能如下所示：

```py
The cat sat on the mat
The cat mat sat on feline the mat
```

## 随机删除

顾名思义，*随机删除*从句子中删除单词。给定概率参数`p`，它将遍历句子，并根据该随机概率决定是否删除单词：

```py
def random_deletion(words, p=0.5):
    if len(words) == 1:
        return words
    remaining = list(filter(lambda x: random.uniform(0,1) > p,words))
    if len(remaining) == 0:
        return [random.choice(words)]
    else
        return remaining
```

该实现处理边缘情况——如果只有一个单词，该技术将返回它；如果我们最终删除了句子中的所有单词，该技术将从原始集合中随机抽取一个单词。

## 随机交换

*随机交换*增强接受一个句子，然后在其中* n *次交换单词，每次迭代都在先前交换的句子上进行。这里是一个实现：

```py
def random_swap(sentence, n=5):
    length = range(len(sentence))
    for _ in range(n):
        idx1, idx2 = random.sample(length, 2)
        sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1]
    return sentence
```

我们根据句子的长度随机抽取两个随机数，然后一直交换直到达到*n*。

EDA 论文中的技术在使用少量标记示例（大约 500 个）时平均提高了约 3%的准确性。如果你的数据集中有 5000 个以上的示例，该论文建议这种改进可能会降至 0.8%或更低，因为模型从更多可用数据量中获得更好的泛化能力，而不是从 EDA 提供的改进中获得。

## 回译

另一种流行的增强数据集的方法是*回译*。这涉及将一个句子从我们的目标语言翻译成一个或多个其他语言，然后将它们全部翻译回原始语言。我们可以使用 Python 库`googletrans`来实现这个目的。在写作时，你可以使用`pip`安装它，因为它似乎不在`conda`中：

```py
pip install googletrans
```

然后，我们可以将我们的句子从英语翻译成法语，然后再翻译回英语：

```py
import googletrans
import googletrans.Translator

translator = Translator()

sentences = ['The cat sat on the mat']

translation_fr = translator.translate(sentences, dest='fr')
fr_text = [t.text for t in translations_fr]
translation_en = translator.translate(fr_text, dest='en')
en_text = [t.text for t in translation_en]
print(en_text)

>> ['The cat sat on the carpet']
```

这样我们就得到了一个从英语到法语再到英语的增强句子，但让我们再进一步，随机选择一种语言：

```py
import random

available_langs = list(googletrans.LANGUAGES.keys())
tr_lang = random.choice(available_langs)
print(f"Translating to {googletrans.LANGUAGES[tr_lang]}")

translations = translator.translate(sentences, dest=tr_lang)
t_text = [t.text for t in translations]
print(t_text)

translations_en_random = translator.translate(t_text, src=tr_lang, dest='en')
en_text = [t.text for t in translations_en_random]
print(en_text)
```

在这种情况下，我们使用`random.choice`来选择一个随机语言，将句子翻译成该语言，然后再翻译回来。我们还将语言传递给`src`参数，以帮助谷歌翻译的语言检测。试一试，看看它有多像*电话*这个老游戏。

你需要了解一些限制。首先，一次只能翻译最多 15,000 个字符，尽管如果你只是翻译句子的话，这应该不会是太大的问题。其次，如果你要在一个大型数据集上使用这个方法，你应该在云实例上进行数据增强，而不是在家里的电脑上，因为如果谷歌封禁了你的 IP，你将无法正常使用谷歌翻译！确保你一次发送几批数据而不是整个数据集。这也应该允许你在谷歌翻译后端出现错误时重新启动翻译批次。

## 增强和 torchtext

到目前为止，你可能已经注意到我所说的关于增强的一切都没有涉及`torchtext`。遗憾的是，这是有原因的。与`torchvision`或`torchaudio`不同，`torchtext`并没有提供转换管道，这有点让人恼火。它确实提供了一种执行预处理和后处理的方式，但这只在标记（单词）级别上操作，这对于同义词替换可能足够了，但对于像回译这样的操作并没有提供足够的控制。如果你尝试在增强中利用这些管道，你应该在预处理管道中而不是后处理管道中进行，因为在后处理管道中你只会看到由整数组成的张量，你需要通过词汇规则将其映射到单词。

出于这些原因，我建议不要浪费时间试图把`torchtext`搞得一团糟来进行数据增强。相反，使用诸如回译之类的技术在 PyTorch 之外进行增强，生成新数据并将其输入模型，就像它是*真实*数据一样。

增强已经讨论完毕，但在结束本章之前，我们应该解决一个悬而未决的问题。

## 迁移学习？

也许你会想知道为什么我们还没有谈论迁移学习。毕竟，这是一个关键技术，可以帮助我们创建准确的基于图像的模型，那么为什么我们不能在这里做呢？事实证明，在 LSTM 网络上实现迁移学习有点困难。但并非不可能。我们将在第九章中回到这个主题，你将看到如何在基于 LSTM 和 Transformer 的网络上实现迁移学习。

# 结论

在这一章中，我们涵盖了一个文本处理流程，涵盖了编码和嵌入，一个简单的基于 LSTM 的神经网络用于执行分类，以及一些针对基于文本数据的数据增强策略。到目前为止，您有很多可以尝试的内容。在标记化阶段，我选择将每条推文都转换为小写。这是自然语言处理中的一种流行方法，但它确实丢弃了推文中的潜在信息。想想看：“为什么这不起作用？”对我们来说甚至更暗示了负面情绪，而不是“为什么这不起作用？”但是在它进入模型之前，我们已经丢弃了这两条推文之间的差异。因此，一定要尝试在标记化文本中保留大小写敏感性。尝试从输入文本中删除停用词，看看是否有助于提高准确性。传统的自然语言处理方法非常强调删除停用词，但我经常发现当保留输入中的停用词时，深度学习技术可以表现得更好（这是我们在本章中所做的）。这是因为它们为模型提供了更多的上下文信息，而将句子简化为*仅包含*重要单词的情况可能会丢失文本中的细微差别。

您可能还想改变嵌入向量的大小。更大的向量意味着嵌入可以捕捉更多关于其建模的单词的信息，但会使用更多内存。尝试从 100 到 1,000 维的嵌入，并查看它如何影响训练时间和准确性。

最后，您也可以尝试使用 LSTM。我们使用了一种简单的方法，但您可以增加`num_layers`以创建堆叠的 LSTM，增加或减少层中隐藏特征的数量，或设置`bidirectional=true`以创建双向 LSTM。将整个 LSTM 替换为 GRU 层也是一个有趣的尝试；它训练速度更快吗？准确性更高吗？尝试实验并看看您会发现什么！

与此同时，我们将从文本转向音频领域，使用`torchaudio`。

# 进一步阅读

+   [“长短期记忆”](https://oreil.ly/WKcxO) 作者：S. Hochreiter 和 J. Schmidhuber（1997 年）

+   [“使用 RNN 编码器-解码器学习短语表示进行统计机器翻译”](https://arxiv.org/abs/1406.1078) 作者：Kyunghyun Cho 等（2014 年）

+   [“双向 LSTM-CRF 模型用于序列标记”](https://arxiv.org/abs/1508.01991) 作者：Zhiheng Huang 等（2015 年）

+   [“注意力机制是您所需要的一切”](https://arxiv.org/abs/1706.03762) 作者：Ashish Vaswani 等（2017 年）

¹ 请注意，使用 CNN 也可以做到这些事情；在过去几年中，已经进行了大量深入研究，以将基于 CNN 的网络应用于时间域。我们不会在这里涵盖它们，但[“时间卷积网络：动作分割的统一方法”](https://arxiv.org/abs/1608.08242) 作者：Colin Lea 等（2016 年）提供了更多信息。还有 seq2seq！

² 参见[“在向量空间中高效估计单词表示”](https://arxiv.org/abs/1301.3781) 作者：Tomas Mikolov 等（2013 年）

³ 参见[“EDA：用于提升文本分类任务性能的简单数据增强技术”](https://arxiv.org/abs/1901.11196) 作者：Jason W. Wei 和 Kai Zou（2019 年）
