# 第9章。序列分析模型

Surya Bhupatiraju

# 分析可变长度输入

到目前为止，我们只处理了具有固定大小的数据：来自MNIST、CIFAR-10和ImageNet的图像。这些模型非常强大，但在许多情况下，固定长度模型是不够的。我们日常生活中绝大多数的互动都需要对序列有深入的理解——无论是阅读早报、准备一碗麦片、听收音机、观看演示还是决定在股市上执行交易。为了适应可变长度的输入，我们必须更加聪明地设计深度学习模型的方法。

[图9-1](#feedforward_networks_thrive)说明了我们的前馈神经网络在分析序列时会出现问题。如果序列与输入层大小相同，模型可以按预期执行。甚至可以通过在输入末尾填充零直到达到适当长度来处理较小的输入。然而，一旦输入超过输入层的大小，朴素地使用前馈网络就不再起作用。

前馈网络在固定输入大小的问题上表现出色。零填充可以解决处理较小输入的问题，但是当朴素地使用时，这些模型在输入超过固定输入大小时会出现问题。

![](Images/fdl2_0901.png)

###### 图9-1。损坏的前馈网络

然而，并非一切希望都已失去。在接下来的几节中，我们将探讨几种策略，可以利用“黑客”前馈网络来处理序列。在本章后面，我们将分析这些黑客的局限性，并讨论新的架构来解决这些问题。我们将通过讨论迄今为止探索的一些最先进的架构来结束本章，以解决复制人类级别逻辑推理和认知的一些最困难挑战。

# 使用神经N-Grams解决seq2seq

在本节中，我们将开始探索一个前馈神经网络架构，可以处理一段文本并生成一系列词性（POS）标签。换句话说，我们希望适当地标记输入文本中的每个单词，如名词、动词、介词等。这在[图9-2](#example_of_an_accurate_pos)中有一个示例。虽然构建一个可以阅读故事后回答问题的人工智能的复杂性不同，但这是朝着开发一个能够理解单词在句子中如何使用含义的算法的坚实第一步。这个问题也很有趣，因为它是一类问题的一个实例，被称为*seq2seq*，目标是将输入序列转换为相应的输出序列。其他著名的seq2seq问题包括在语言之间翻译文本（我们将在本章后面处理）、文本摘要和将语音转录为文本。

![](Images/fdl2_0902.png)

###### 图9-2。英语句子准确的POS解析示例

正如我们讨论过的，如何一次性处理一段文本以预测完整的POS标签序列并不明显。相反，我们利用了一种类似于我们在上一章中开发单词的分布式向量表示的技巧。关键观察是：*不需要考虑长期依赖性来预测任何给定单词的POS*。 

这一观察的含义是，我们可以通过使用固定长度的子序列，而不是使用整个序列同时预测所有POS标签，逐个预测每个POS标签。特别是，我们利用从感兴趣的单词开始并向过去扩展n个单词的子序列。这种*神经n-gram策略*在[图9-3](#perform_seq2seq)中有所描述。

![](Images/fdl2_0903.png)

###### 图9-3。在我们可以忽略长期依赖性时使用前馈网络执行seq2seq

具体来说，当我们预测输入中第i个单词的词性标签时，我们使用第i-n+1到第i个单词作为输入。我们将这个子序列称为*上下文窗口*。为了处理整个文本，我们将首先将网络定位在文本的开头。然后，我们将继续将网络的上下文窗口每次移动一个单词，预测最右边单词的词性标签，直到达到输入的末尾。

利用上一章的词嵌入策略，我们将使用单词的压缩表示，而不是独热向量。这将使我们能够减少模型中的参数数量，并加快学习速度。

# 实现词性标注器

现在我们对词性网络架构有了深入的理解，我们可以深入实现。在高层次上，网络由一个利用三元上下文窗口的输入层组成。我们将使用300维的词嵌入，从而得到一个大小为900的上下文窗口。前馈网络将有两个隐藏层，分别为512个神经元和256个神经元。然后，输出层将是一个softmax，计算POS标签输出在44个可能标签空间上的概率分布。像往常一样，我们将使用Adam优化器和默认的超参数设置，总共训练1000个时代，并利用批量归一化进行正则化。

实际的网络与我们过去实现的网络非常相似。相反，构建词性标注器的难点在于准备数据集。我们将利用从[Google News](https://oreil.ly/Rsu9A)生成的预训练词嵌入。它包括了300万个单词和短语的向量，并在大约1000亿个单词上进行了训练。我们可以使用`gensim` Python包来读取数据集。Google Colab已经预先安装了`gensim`。如果您使用另一台机器，可以使用`pip`来安装该包。您还需要下载Google News数据文件：

```py
$ pip install gensim
$ wget https://s3.amazonaws.com/dl4j-distribution/
  GoogleNews-vectors-negative300.bin.gz -O googlenews.bin.gz

```

接下来，我们可以使用以下命令将这些向量加载到内存中：

```py
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('./googlenews.bin.gz',
                                          binary=True)

```

然而，这种操作的问题在于它非常慢（根据您的机器规格，可能需要长达一小时）。为了避免每次运行程序时都将整个数据集加载到内存中，特别是在调试代码或尝试不同超参数时，我们使用轻量级数据库[LevelDB](http://leveldb.org)将相关子集的向量缓存到磁盘上。为了构建适当的Python绑定（允许我们从Python与LevelDB实例交互），我们只需使用以下命令：

```py
$ pip install leveldb
```

正如我们提到的，`gensim`模型包含三百万个单词，比我们的数据集要大。为了提高效率，我们将有选择地缓存数据集中的单词向量，并丢弃其他所有内容。为了找出我们想要缓存的单词，让我们从[CoNLL-2000任务](https://oreil.ly/8qJeZ)下载POS数据集。

```py
$ wget http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz 
  -O - | gunzip |
  cut -f1,2 -d" " > pos.train.txt

$ wget http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz 
  -O - | gunzip |
  cut -f1,2 -d " " > pos.test.txt

```

数据集由格式化为一系列行的连续文本组成，其中第一个元素是一个单词，第二个元素是相应的词性。以下是训练数据集的前几行：

```py
Confidence NN
in IN
the DT
pound NN
is VBZ
widely RB
expected VBN
to TO
take VB
another DT
sharp JJ
dive NN
if IN
trade NN
figures NNS
for IN
September NNP
, ,
due JJ
for IN
release NN
tomorrow NN
...
```

为了将数据集的格式与`gensim`模型匹配，我们需要进行一些预处理。例如，该模型用'#'字符替换数字，将适当的单词组合成实体（例如，将“New_York”视为一个单词而不是两个单独的单词），并在原始数据使用破折号时使用下划线。我们对数据集进行预处理以符合这个模型模式，使用以下代码（类似的代码用于处理训练数据）：

```py
def create_pos_dataset(filein, fileout):
  dataset = []
  with open(filein) as f:
    dataset_raw = f.readlines()
    dataset_raw = [e.split() for e in dataset_raw
                        if len(e.split()) > 0]

  counter = 0
  while counter < len(dataset_raw):
    pair = dataset_raw[counter]
    if counter < len(dataset_raw) - 1:
      next_pair = dataset_raw[counter + 1]
      if (pair[0] + "_" + next_pair[0] in model) and \
      (pair[1] == next_pair[1]):
        dataset.append([pair[0] + "_" + next_pair[0], pair[1]])
        counter += 2
        continue

    word = re.sub("\d", "#", pair[0])
    word = re.sub("-", "_", word)

    if word in model:
      dataset.append([word, pair[1]])
      counter += 1
      continue

    if "_" in word:
      subwords = word.split("_")
      for subword in subwords:
        if not (subword.isspace() or len(subword) == 0):
          dataset.append([subword, pair[1]])
      counter += 1
      continue

    dataset.append([word, pair[1]])
    counter += 1

  with open(fileout, 'w') as processed_file:
    for item in dataset:
      processed_file.write("%s\n" % (item[0] + " " + item[1]))

  return dataset

train_pos_dataset = create_pos_dataset('./pos.train.txt',
                                       './pos.train.processed.txt')
test_pos_dataset = create_pos_dataset('./pos.test.txt',
                                      './pos.test.processed.txt')

```

现在我们已经适当处理了用于加载的数据集，我们可以加载LevelDB中的单词。如果`gensim`模型中存在单词或短语，我们可以将其缓存到LevelDB实例中。如果没有，我们会随机选择一个向量来表示该标记，并将其缓存，以便在再次遇到时记得使用相同的向量：

```py
import leveldb
db = leveldb.LevelDB("./word2vecdb")

counter = 0
dataset_vocab = {}
tags_to_index = {}
index_to_tags = {}
index = 0
for pair in train_pos_dataset + test_pos_dataset:
  if pair[0] not in dataset_vocab:
    dataset_vocab[pair[0]] = index
    index += 1
  if pair[1] not in tags_to_index:
    tags_to_index[pair[1]] = counter
    index_to_tags[counter] = pair[1]
    counter += 1

nonmodel_cache = {}

counter = 1
total = len(dataset_vocab.keys())
for word in dataset_vocab:

  if word in model:
    db.Put(bytes(word,'utf-8'), model[word])
  elif word in nonmodel_cache:
    db.Put(bytes(word,'utf-8'), nonmodel_cache[word])
  else:
    #print(word)
    nonmodel_cache[word] = np.random.uniform(-0.25,
                                             0.25,
                                             300).astype(np.float32)
    db.Put(bytes(word,'utf-8'), nonmodel_cache[word])
  counter += 1

```

第一次运行脚本后，如果数据已经存在，我们可以直接从数据库加载数据：

```py
db = leveldb.LevelDB("./word2vecdb")

x = db.Get(bytes('Confidence','utf-8'))
print(np.frombuffer(x,dtype='float32').shape)
# out: (300,)

```

接下来，我们为训练和测试数据集构建数据集对象，我们可以使用这些对象生成用于训练和测试的小批量数据。构建数据集对象需要访问LevelDB `db`，`dataset`，将POS标记映射到输出向量中的索引的字典`tags_to_index`，以及一个布尔值`get_all`，用于确定获取小批量数据时是否应默认检索完整集合：

```py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NgramPOSDataset(Dataset):
  def __init__(self, db, dataset, tags_to_index, n_grams):
    super(NgramPOSDataset, self).__init__()
    self.db = db
    self.dataset = dataset
    self.tags_to_index = tags_to_index
    self.n_grams = n_grams

  def __getitem__(self, index):
    ngram_vector = np.array([])

    for ngram_index in range(index, index + self.n_grams):
      word, _ = self.dataset[ngram_index]
      vector_bytes = self.db.Get(bytes(word, 'utf-8'))
      vector = np.frombuffer(vector_bytes, dtype='float32')
      ngram_vector = np.append(ngram_vector, vector)

      _, tag = self.dataset[index + int(np.floor(self.n_grams/2))]
      label = self.tags_to_index[tag]
    return torch.tensor(ngram_vector, dtype=torch.float32), label

  def __len__(self):
    return (len(self.dataset) - self.n_grams + 1)

trainset = NgramPOSDataset(db, train_pos_dataset, tags_to_index, 3)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

```

最后，我们设计我们的前馈网络与之前章节中的方法类似。我们省略了代码的讨论，并参考文件*Ch09_01_POS_Tagger.ipynb*在[书的存储库](https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book)中。

每个时代，我们通过解析句子“这个女人，在拿起她的伞后，去了银行存钱。”来手动检查模型。在训练100个时代后，算法达到了超过96%的准确率，并几乎完美地解析了验证句子（它犯了一个可以理解的错误，混淆了第一次出现“her”一词的所有格代词和人称代词标记）。我们将通过在TensorBoard中包含我们模型性能的可视化来结束这一部分[图9-4](#tensorboard_viz_of_feedforward_pos)。

POS标记模型是一个很好的练习，但它主要是重复我们在之前章节学到的概念。在本章的其余部分，我们将开始思考更复杂的与序列相关的学习任务。为了解决这些更困难的问题，我们需要涉及全新的概念，开发新的架构，并开始探索现代深度学习研究的前沿。我们将从下一个依存句法分析问题开始。

![](Images/fdl2_0904.png)

###### 图9-4。我们前馈POS标记模型的TensorBoard可视化

# 依存句法分析和SyntaxNet

我们用来解决POS标记任务的框架相当简单。有时候，我们需要更有创意地解决seq2seq问题，特别是在问题复杂性增加时。在本节中，我们将探讨采用创造性数据结构来解决困难的seq2seq问题的策略。作为一个说明性例子，我们将探讨依存句法分析问题。

构建依存句法分析树的想法是映射句子中单词之间的关系。例如，看一下[图9-5](#example_of_a_dependency_parse)中的依存关系。单词“I”和“taxi”是单词“took”的子节点，具体来说是动词的主语和直接宾语。

![](Images/fdl2_0905.png)

###### 图9-5。一个依存句法分析的示例，生成句子中单词之间关系的树

将树表示为序列的一种方法是将其线性化。让我们考虑[图9-6](#linearize_two_example_trees)中的示例。基本上，如果您有一个以根`R`为根，子节点`A`（通过边`r_a`连接），`B`（通过边`r_b`连接）和`C`（通过边`r_c`连接）的图，我们可以将表示线性化为`(R, r_a, A, r_b, B, r_c, C)`。我们甚至可以表示更复杂的图。例如，假设节点`B`实际上有两个名为`D`（通过边`b_d`连接）和`E`（通过边`b_e`连接）的子节点。我们可以将这个新图表示为`(R, r_a, A, r_b, [B, b_d, D, b_e, E], r_c, C)`。

![](Images/fdl2_0906.png)

###### 图9-6。我们线性化了两个示例树：为了视觉清晰起见，图中省略了边标签

使用这种范式，我们可以将我们的示例依赖解析线性化，如[图9-7](#linearization_of_dependency_parse_tree)所示。

![](Images/fdl2_0907.png)

###### 图9-7。依赖解析树示例的线性化

这个seq2seq问题的一个解释是读取输入句子并生成一个代表输入依赖解析线性化的输出令牌序列。然而，我们可能不太清楚如何将我们从前一节中的策略移植过来，那里单词和它们的POS标签之间有明确的一对一映射。此外，我们可以通过查看附近的上下文轻松地做出关于POS标签的决定。对于依赖解析，句子中单词的顺序与线性化中的令牌的顺序之间没有明确的关系。看起来依赖解析任务要求我们识别可能跨越大量单词的边缘。因此，乍一看，这种设置似乎直接违反了我们不需要考虑任何长期依赖的假设。

为了使问题更容易解决，我们将依赖解析任务重新考虑为找到一系列有效的“动作”，以生成正确的依赖解析。这种技术，称为*弧-标准*系统，最初由Nivre在2004年描述，后来在2014年由Chen和Manning在神经环境中利用。[^1](ch09.xhtml#idm45934164188240)在弧-标准系统中，我们首先将句子的前两个词放入堆栈，并保留缓冲区中的剩余词，如[图9-8](#we_have_three_options)所示。

![](Images/fdl2_0908.png)

###### 图9-8。弧-标准系统中的三个选项：将一个词从缓冲区移动到堆栈，从右元素到左元素绘制弧（左弧），或从左元素到右元素绘制弧（右弧）

在任何步骤中，我们可以采取三种可能的行动类别之一：

Shift

将一个词从缓冲区移动到堆栈的前面。

左弧

将堆栈前面的两个元素合并为一个单元，其中最右边元素的根是父节点，最左边元素的根是子节点。

右弧

将堆栈前面的两个元素合并为一个单元，其中左元素的根是父节点，右元素的根是子节点。

我们注意到，虽然执行shift只有一种方法，但弧动作可以有许多不同的风格，每种风格由分配给生成的弧的依赖标签区分。也就是说，在本节中，我们将简化我们的讨论和说明，将每个决策视为三种行动中的一种选择（而不是数十种行动）。

当缓冲区为空且堆栈中有一个元素时（表示完整的依赖解析），我们终止此过程。为了完整地说明这个过程，我们展示了一系列生成我们示例输入句子的依赖解析的动作，如[图9-9](#sequence_of_actions)所示。

![](Images/fdl2_0909.png)

###### 图9-9。一系列动作的序列，导致正确的依赖解析；我们省略标签

将这个决策框架重新表述为一个学习问题并不太困难。在每一步，我们采取当前的配置，并通过提取描述配置的大量特征（堆栈/缓冲区中特定位置的单词，这些位置单词的特定子节点，词性标签等）对配置进行向量化。在训练时，我们可以将这个向量输入到一个前馈网络中，并将其对下一步要采取的行动的预测与人类语言学家做出的黄金标准决策进行比较。在实际应用中，我们可以采取网络推荐的行动，将其应用于配置，并将这个新配置作为下一步的起点（特征提取，行动预测和行动应用）。这个过程在[图9-10](#neural_framework_for_arc_standard_dependency)中显示。

![](Images/fdl2_0910.png)

###### 图9-10。用于arc-standard依赖解析的神经框架

综上所述，这些想法构成了谷歌的SyntaxNet的核心，这是用于依赖解析的最先进的开源实现。深入讨论实现的细节超出了本文的范围，但我们建议您参考[开源存储库](https://oreil.ly/UT1ga)，其中包含Parsey McParseface的实现，这是截至本文发表时最准确的公开报告的英语解析器。

# 束搜索和全局归一化

在前一节中，我们描述了在实践中部署SyntaxNet的一个天真策略。这个策略是纯粹*贪婪*的；也就是说，我们选择具有最高概率的预测，而不考虑可能会因为早期错误而陷入困境。在POS的例子中，做出错误的预测基本上是无关紧要的。这是因为每个预测都可以被视为一个纯粹独立的子问题（给定预测的结果不会影响下一步的输入）。

在SyntaxNet中，这个假设不再成立，因为我们在第*n*步的预测会影响我们在第*n+1*步使用的输入。这意味着我们所犯的任何错误都会影响所有后续决策。此外，一旦错误变得明显，就没有好的方法“回头”并纠正错误。

*花园路径句*是这一点很重要的一个极端案例。考虑以下句子：“The complex houses married and single soldiers and their families.”第一次浏览时令人困惑。大多数人将“complex”解释为形容词，“houses”解释为名词，“married”解释为过去时动词。尽管这在语义上几乎没有意义，并且在阅读句子的其余部分时开始出现问题。相反，我们意识到“complex”是一个名词（如军事基地），而“houses”是一个动词。换句话说，这个句子暗示着军事基地包含士兵（可能是单身或已婚）和他们的家人。一个*贪婪*版本的SyntaxNet会无法纠正早期解析错误，即将“complex”视为描述“houses”的形容词，因此无法正确解析整个句子。

为了弥补这个缺点，我们使用一种称为*束搜索*的策略，如[图9-11](#illustration_of_using_beam_search)所示。我们通常在像SyntaxNet这样的情况下利用束搜索，其中我们网络在特定步骤的输出会影响未来步骤中使用的输入。束搜索的基本思想是，我们不是在每一步贪婪地选择最有可能的预测，而是维护一个*束*，其中包含前*k*个动作序列及其相关概率的最可能的假设（最多固定的*束大小b*）。束搜索可以分为两个主要阶段：扩展和修剪。

![](Images/fdl2_0911.png)

###### 图9-11。在部署经过训练的SyntaxNet模型时使用束搜索（束大小为2）

在*扩展*步骤中，我们将每个假设视为SyntaxNet的可能输入。假设SyntaxNet对总共 <math alttext="StartAbsoluteValue upper A EndAbsoluteValue"><mrow><mo>|</mo> <mi>A</mi> <mo>|</mo></mrow></math> 个操作空间产生一个概率分布。然后，我们计算前 <math alttext="k plus 1"><mrow><mi>k</mi> <mo>+</mo> <mn>1</mn></mrow></math> 个操作序列的每个可能假设的概率。然后，在*修剪*步骤中，我们仅保留具有最大概率的*b*个假设，而不是总共 <math alttext="b StartAbsoluteValue upper A EndAbsoluteValue"><mrow><mi>b</mi> <mo>|</mo> <mi>A</mi> <mo>|</mo></mrow></math> 个选项。正如[图9-11](#illustration_of_using_beam_search)所示，束搜索使SyntaxNet能够通过在句子中早期考虑可能性较小的假设来纠正不正确的预测。事实上，深入研究所示例子，贪婪方法会建议正确的移动序列应该是一个移动，然后是一个左弧。实际上，最佳（最高概率）选项应该是使用左弧后跟右弧。束搜索与束大小为2呈现了这个结果。

完整的开源版本将这一步推进到更深层次，并尝试将束搜索的概念引入到网络训练的过程中。正如Andor等人在2016年所描述的那样，这个*全局归一化*的过程在实践中提供了强大的理论保证和明显的性能增益，相对于*局部归一化*。在局部归一化网络中，我们的网络被要求在给定配置的情况下选择最佳操作。网络输出一个通过softmax层归一化的分数。这旨在对所有可能的操作建模一个概率分布，考虑到迄今为止执行的操作。我们的损失函数试图将概率分布强制为理想输出（即，对于正确操作的概率为1，对于所有其他操作为0）。交叉熵损失出色地确保了这一点。

在全局归一化网络中，我们对分数的解释略有不同。我们不是通过softmax将分数转换为每个操作的概率分布，而是将所有假设动作序列的分数相加。确保我们选择正确的假设序列的一种方法是计算所有可能假设的总和，然后应用softmax层生成概率分布。理论上，我们可以使用与局部归一化网络中相同的交叉熵损失函数。然而，这种策略的问题在于可能的假设序列数量太大，难以处理。即使考虑到平均句子长度为10，每个左右弧有1个移动和7个标签的保守总操作数为15，这对应于1,000,000,000,000,000个可能的假设。

为了使这个问题可解，如[图9-12](#make_global_normalization_in_syntaxnet_tractable)所示，我们应用一个固定大小的束搜索，直到我们要么（1）到达句子的结尾，要么（2）正确的动作序列不再包含在束中。然后，我们构建一个损失函数，试图通过最大化相对于其他假设的分数来尽可能将“黄金标准”动作序列（用蓝色突出显示）推到束的顶部。虽然我们不会在这里深入讨论如何构建这个损失函数的细节，但我们建议您参考2016年Andor等人的原始论文。^([3](ch09.xhtml#idm45934164119456)) 该论文还描述了一个更复杂的词性标注器，它使用全局归一化和束搜索来显著提高准确性（与我们在本章前面构建的词性标注器相比）。

![](Images/fdl2_0912.png)

###### 图9-12. 训练和束搜索的耦合可以使SyntaxNet中的全局归一化可解

# 有关有状态深度学习模型的案例

虽然我们已经探索了几种将前馈网络适应序列分析的技巧，但我们尚未真正找到一个优雅的解决方案来进行序列分析。在词性标注器示例中，我们明确假设可以忽略长期依赖关系。通过引入束搜索和全局归一化的概念，我们能够克服这种假设的一些局限性，但即使如此，问题空间仍受限于输入序列中的元素与输出序列中的元素之间存在一对一的映射的情况。例如，即使在依赖解析模型中，我们也必须重新制定问题，以发现在构建解析树和弧标准动作时，输入配置序列之间存在一对一的映射。

然而，有时任务比找到输入和输出序列之间的一对一映射要复杂得多。例如，我们可能希望开发一个模型，可以一次消耗整个输入序列，然后得出整个输入的情感是积极的还是消极的。我们将在本章后面构建一个简单的模型来执行这个任务。我们可能希望一个算法可以消耗一个复杂的输入（比如一幅图像），然后逐字生成一个描述输入的句子。我们甚至可能希望将句子从一种语言翻译成另一种语言（例如，从英语到法语）。在所有这些情况下，输入标记和输出标记之间没有明显的映射。相反，这个过程更像是[图9-13](#ideal_model_for_sequence_analysis)中的情况。

![](Images/fdl2_0913.png)

###### 图9-13. 用于序列分析的理想模型可以在长时间内存储信息，从而产生一个连贯的“思考”向量，可以用来生成答案

这个想法很简单。我们希望我们的模型在阅读输入序列的过程中保持某种记忆。当它阅读输入时，模型应该能够修改这个记忆库，考虑到它观察到的信息。当它到达输入序列的末尾时，内部记忆包含一个代表原始输入的关键信息，即意义的“思考”。然后，如[图9-13](#ideal_model_for_sequence_analysis)所示，我们应该能够使用这个思考向量来为原始序列产生一个标签，或者产生一个适当的输出序列（翻译、描述、抽象摘要等）。

这里的概念是我们在之前的章节中没有探讨过的。前馈网络本质上是“无状态”的。在训练后，前馈网络是一个静态结构。它无法在输入之间保持记忆，或者根据过去看到的输入改变其处理输入的方式。为了执行这种策略，我们需要重新考虑如何构建神经网络，以创建“有状态”的深度学习模型。为此，我们将不得不回到如何在单个神经元级别思考网络。在下一节中，我们将探讨*循环连接*（与迄今为止我们研究的前馈连接相对）如何使模型保持状态，同时描述一类称为*循环神经网络*（RNNs）的模型。

# 循环神经网络

RNNs最早在1980年代引入，但最近由于几项智力和硬件突破，使得它们易于训练而重新流行起来。RNNs与前馈网络不同，因为它们利用一种特殊类型的神经层，称为循环层，使得网络能够在使用网络之间保持状态。

[图9-14](#recurrent_layer_contains)展示了循环层的神经结构。所有神经元都有(1)来自前一层所有神经元的传入连接和(2)指向后续层所有神经元的传出连接。然而，我们注意到循环层神经元并不只有这些连接。与前馈层不同，循环层还具有循环连接，用于在同一层神经元之间传播信息。一个完全连接的循环层使得每个神经元都与其层中的每个其他神经元(包括自身)之间有信息流。因此，一个具有  <math alttext="r"><mi>r</mi></math>  个神经元的循环层共有  <math alttext="r squared"><msup><mi>r</mi> <mn>2</mn></msup></math>  个循环连接。

![](Images/fdl2_0914.png)

###### 图9-14。循环层包含循环连接，即位于同一层的神经元之间的连接

为了更好地理解RNNs的工作原理，让我们探讨一下在适当训练后一个RNN是如何运作的。每当我们想要处理一个新序列时，我们会创建一个模型的新实例。我们可以通过将网络实例的生命周期划分为离散的时间步来推理包含循环层的网络。在每个时间步，我们将输入的下一个元素提供给模型。前馈连接表示从一个神经元到另一个神经元的信息流，传输的数据是当前时间步的计算神经元激活。然而，循环连接表示数据是*上一个*时间步存储的神经元激活的信息流。因此，循环网络中神经元的激活表示网络实例的累积状态。循环层中神经元的初始激活是我们模型的参数，我们像确定每个连接的权重的最佳值一样确定它们的最佳值在训练过程中。

事实证明，给定一个固定的RNN实例寿命（比如*t*个时间步），我们实际上可以将实例表示为一个前馈网络（尽管结构不规则）。这种巧妙的转换，如[图9-15](#an_rnn_through_time)所示，通常被称为“通过时间展开”RNN。让我们考虑图中的示例RNN。我们想要将两个输入序列（每个维度为1）映射到一个单一输出（也是维度为1）。我们通过将单个循环层的神经元复制*t*次来执行转换，每个时间步都复制一次。我们类似地复制输入和输出层的神经元。我们重新绘制每个时间副本内的前馈连接，就像在原始网络中一样。然后我们将循环连接绘制为从每个时间副本到下一个时间副本的前馈连接（因为循环连接携带前一个时间步的神经元激活）。

![](Images/fdl2_0915.png)

###### 图9-15. 我们可以通过时间运行RNN来将其表示为一个可以使用反向传播训练的前馈网络

我们现在也可以通过计算展开版本的梯度来训练RNN。这意味着我们为前馈网络使用的所有反向传播技术也适用于训练RNN。然而，我们遇到了一个问题。在使用每批训练示例之后，我们需要根据计算的误差导数修改权重。在我们的展开网络中，我们有一组连接，它们都对应于原始RNN中的同一连接。然而，为这些展开连接计算的误差导数不能保证是相等的（实际上，可能不会相等）。我们可以通过对属于同一组的所有连接的误差导数进行平均或求和来规避这个问题。这使我们能够利用一个考虑到所有作用于连接权重的动态的误差导数，以便我们试图迫使网络构建准确的输出。

# 消失梯度的挑战

我们使用有状态的网络模型的动机在于捕捉输入序列中的长期依赖关系。一个具有大内存库（即具有相当大的循环层）的RNN能够总结这些依赖关系似乎是合理的。事实上，从理论上讲，Kilian和Siegelmann在1996年证明了RNN是一个通用的功能表示。换句话说，通过足够的神经元和正确的参数设置，RNN可以用来表示输入和输出序列之间的任何功能映射。

这个理论很有前途，但并不一定能转化为实践。知道RNN可以表示任意任意函数是很好的，但更有用的是知道是否可以通过应用梯度下降算法从头开始教会RNN一个现实的功能映射。如果发现这是不切实际的，我们将陷入困境，因此我们有必要严谨地探讨这个问题。让我们从考虑最简单的可能的RNN开始，如[图9-16](#single_neuron_fully_connected)所示，具有一个输入神经元，一个输出神经元和一个具有一个神经元的全连接循环层。

![](Images/fdl2_0916.png)

###### 图9-16. 一个单神经元，完全连接的循环层（压缩和展开），用于研究基于梯度的学习算法

让我们从简单的开始。给定非线性 <math alttext="f"><mi>f</mi></math>，我们可以将循环层中隐藏神经元在时间步 *t* 的激活 <math alttext="h Superscript left-parenthesis t right-parenthesis"><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup></math> 表示为以下形式，其中 <math alttext="i Superscript left-parenthesis t right-parenthesis"><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup></math> 是来自时间步*t*的输入神经元的传入逻辑：

<math alttext="h Superscript left-parenthesis t right-parenthesis Baseline equals f left-parenthesis w Subscript i n Superscript left-parenthesis t right-parenthesis Baseline i Superscript left-parenthesis t right-parenthesis Baseline plus w Subscript r e c Superscript left-parenthesis t minus 1 right-parenthesis Baseline h Superscript left-parenthesis t minus 1 right-parenthesis Baseline right-parenthesis"><mrow><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup> <mo>=</mo> <mi>f</mi> <mfenced separators="" open="(" close=")"><msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msubsup> <msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup> <mo>+</mo> <msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup> <msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msup></mfenced></mrow></math>

让我们尝试计算隐藏神经元的激活如何响应过去*k*个时间步的输入逻辑的变化。在分析反向传播梯度表达式的这个组件时，我们可以开始量化从过去输入中保留多少“记忆”。我们首先取偏导数并应用链式法则：

<math alttext="StartFraction normal partial-differential h Superscript left-parenthesis t right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction equals f prime left-parenthesis w Subscript i n Superscript left-parenthesis t right-parenthesis Baseline i Superscript left-parenthesis t right-parenthesis Baseline plus w Subscript r e c Superscript left-parenthesis t minus 1 right-parenthesis Baseline h Superscript left-parenthesis t minus 1 right-parenthesis Baseline right-parenthesis StartFraction normal partial-differential Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction left-parenthesis w Subscript i n Superscript left-parenthesis t right-parenthesis Baseline i Superscript left-parenthesis t right-parenthesis Baseline plus w Subscript r e c Superscript left-parenthesis t minus 1 right-parenthesis Baseline h Superscript left-parenthesis t minus 1 right-parenthesis Baseline right-parenthesis"><mrow><mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac> <mo>=</mo> <msup><mi>f</mi> <mo>'</mo></msup> <mfenced separators="" open="(" close=")"><msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msubsup> <msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup> <mo>+</mo> <msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup> <msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msup></mfenced> <mfrac><mi>∂</mi> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac> <mfenced separators="" open="(" close=")"><msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msubsup> <msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup> <mo>+</mo> <msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup> <msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msup></mfenced></mrow></math>

因为输入和循环权重的值与时间步 <math alttext="t minus k"><mrow><mi>t</mi> <mo>-</mo> <mi>k</mi></mrow></math> 的输入逻辑无关，我们可以进一步简化这个表达式：

<math alttext="StartFraction normal partial-differential h Superscript left-parenthesis t right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction equals f prime left-parenthesis w Subscript i n Superscript left-parenthesis t right-parenthesis Baseline i Superscript left-parenthesis t right-parenthesis Baseline plus w Subscript r e c Superscript left-parenthesis t minus 1 right-parenthesis Baseline h Superscript left-parenthesis t minus 1 right-parenthesis Baseline right-parenthesis w Subscript r e c Superscript left-parenthesis t minus 1 right-parenthesis Baseline StartFraction normal partial-differential h Superscript left-parenthesis t minus 1 right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction"><mrow><mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac> <mo>=</mo> <msup><mi>f</mi> <mo>'</mo></msup> <mfenced separators="" open="(" close=")"><msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msubsup> <msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup> <mo>+</mo> <msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup> <msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msup></mfenced> <msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup> <mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac></mrow></math>

因为我们关心这个导数的大小，我们可以对两边取绝对值。我们也知道对于所有常见的非线性（tanh、logistic和ReLU非线性）， <math alttext="StartAbsoluteValue f prime EndAbsoluteValue"><mfenced separators="" open="|" close="|"><msup><mi>f</mi> <mo>'</mo></msup></mfenced></math> 的最大值最多为1。这导致以下递归不等式：

<math alttext="StartAbsoluteValue StartFraction normal partial-differential h Superscript left-parenthesis t right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction EndAbsoluteValue less-than-or-equal-to StartAbsoluteValue w Subscript r e c Superscript left-parenthesis t minus 1 right-parenthesis Baseline EndAbsoluteValue dot StartAbsoluteValue StartFraction normal partial-differential h Superscript left-parenthesis t minus 1 right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction EndAbsoluteValue"><mrow><mfenced separators="" open="|" close="|"><mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac></mfenced> <mo>≤</mo> <mfenced separators="" open="|" close="|"><msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup></mfenced> <mo>·</mo> <mfenced separators="" open="|" close="|"><mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac></mfenced></mrow></math>

我们可以继续递归地扩展这个不等式，直到达到基本情况，在步骤 <math alttext="t minus k"><mrow><mi>t</mi> <mo>-</mo> <mi>k</mi></mrow></math> 处：

<math alttext="StartAbsoluteValue StartFraction normal partial-differential h Superscript left-parenthesis t right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction EndAbsoluteValue less-than-or-equal-to StartAbsoluteValue w Subscript r e c Superscript left-parenthesis t minus 1 right-parenthesis Baseline EndAbsoluteValue dot ellipsis dot StartAbsoluteValue w Subscript r e c Superscript left-parenthesis t minus k right-parenthesis Baseline EndAbsoluteValue dot StartAbsoluteValue StartFraction normal partial-differential h Superscript left-parenthesis t minus k right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction EndAbsoluteValue"><mrow><mfenced separators="" open="|" close="|"><mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac></mfenced> <mo>≤</mo> <mfenced separators="" open="|" close="|"><msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup></mfenced> <mo>·</mo> <mo>...</mo> <mo>·</mo> <mfenced separators="" open="|" close="|"><msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msubsup></mfenced> <mo>·</mo> <mfenced separators="" open="|" close="|"><mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac></mfenced></mrow></math>

我们可以类似地评估这个偏导数：

<math alttext="h Superscript left-parenthesis t minus k right-parenthesis Baseline equals f left-parenthesis w Subscript i n Superscript left-parenthesis t minus k right-parenthesis Baseline i Superscript left-parenthesis t minus k right-parenthesis Baseline plus w Subscript r e c Superscript left-parenthesis t minus k minus 1 right-parenthesis Baseline h Superscript left-parenthesis t minus k minus 1 right-parenthesis Baseline right-parenthesis"><mrow><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup> <mo>=</mo> <mi>f</mi> <mfenced separators="" open="(" close=")"><msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msubsup> <msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup> <mo>+</mo> <msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup> <msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msup></mfenced></mrow></math>

<math alttext="StartFraction normal partial-differential h Superscript left-parenthesis t minus k right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction equals f prime left-parenthesis w Subscript i n Superscript left-parenthesis t minus k right-parenthesis Baseline i Superscript left-parenthesis t minus k right-parenthesis Baseline plus w Subscript r e c Superscript left-parenthesis t minus k minus 1 right-parenthesis Baseline h Superscript left-parenthesis t minus k minus 1 right-parenthesis Baseline right-parenthesis StartFraction normal partial-differential Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction left-parenthesis w Subscript i n Superscript left-parenthesis t minus k right-parenthesis Baseline i Superscript left-parenthesis t minus k right-parenthesis Baseline plus w Subscript r e c Superscript left-parenthesis t minus k minus 1 right-parenthesis Baseline h Superscript left-parenthesis t minus k minus 1 right-parenthesis Baseline right-parenthesis"><mrow><mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac> <mo>=</mo> <msup><mi>f</mi> <mo>'</mo></msup> <mfenced separators="" open="(" close=")"><msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msubsup> <msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup> <mo>+</mo> <msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup> <msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msup></mfenced> <mfrac><mi>∂</mi> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac> <mfenced separators="" open="(" close=")"><msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msubsup> <msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup> <mo>+</mo> <msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup> <msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msup></mfenced></mrow></math>

在这个表达式中，时间为 <math alttext="t minus k minus 1"><mrow><mi>t</mi> <mo>-</mo> <mi>k</mi> <mo>-</mo> <mn>1</mn></mrow></math> 的隐藏激活与时间为 <math alttext="t minus k"><mrow><mi>t</mi> <mo>-</mo> <mi>k</mi></mrow></math> 的输入值无关。因此我们可以将这个表达式重写为：

<math alttext="StartFraction normal partial-differential h Superscript left-parenthesis t minus k right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction equals f prime left-parenthesis w Subscript i n Superscript left-parenthesis t minus k right-parenthesis Baseline i Superscript left-parenthesis t minus k right-parenthesis Baseline plus w Subscript r e c Superscript left-parenthesis t minus k minus 1 right-parenthesis Baseline h Superscript left-parenthesis t minus k minus 1 right-parenthesis Baseline right-parenthesis w Subscript i n Superscript left-parenthesis t minus k right-parenthesis"><mrow><mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac> <mo>=</mo> <msup><mi>f</mi> <mo>'</mo></msup> <mfenced separators="" open="(" close=")"><msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msubsup> <msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup> <mo>+</mo> <msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup> <msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msup></mfenced> <msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msubsup></mrow></math>

最后，对两边取绝对值，并再次应用关于 <math alttext="StartAbsoluteValue f prime EndAbsoluteValue"><mfenced separators="" open="|" close="|"><msup><mi>f</mi> <mo>'</mo></msup></mfenced></math> 最大值的观察，我们可以写成：

<math alttext="StartAbsoluteValue StartFraction normal partial-differential h Superscript left-parenthesis t minus k right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction EndAbsoluteValue less-than-or-equal-to StartAbsoluteValue w Subscript i n Superscript left-parenthesis t minus k right-parenthesis Baseline EndAbsoluteValue"><mrow><mfenced separators="" open="|" close="|"><mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac></mfenced> <mo>≤</mo> <mfenced separators="" open="|" close="|"><msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msubsup></mfenced></mrow></math>

这导致最终的不等式（我们可以简化，因为我们约束不同时间步的连接具有相等的值）：

<math alttext="StartAbsoluteValue StartFraction normal partial-differential h Superscript left-parenthesis t right-parenthesis Baseline Over normal partial-differential i Superscript left-parenthesis t minus k right-parenthesis Baseline EndFraction EndAbsoluteValue less-than-or-equal-to StartAbsoluteValue w Subscript r e c Superscript left-parenthesis t minus 1 right-parenthesis Baseline EndAbsoluteValue dot ellipsis dot StartAbsoluteValue w Subscript r e c Superscript left-parenthesis t minus k right-parenthesis Baseline EndAbsoluteValue dot StartAbsoluteValue w Subscript i n Superscript left-parenthesis t minus k right-parenthesis Baseline EndAbsoluteValue equals StartAbsoluteValue w Subscript r e c Baseline EndAbsoluteValue Superscript k Baseline dot w Subscript i n"><mrow><mfenced separators="" open="|" close="|"><mfrac><mrow><mi>∂</mi><msup><mi>h</mi> <mrow><mo>(</mo><mi>t</mi><mo>)</mo></mrow></msup></mrow> <mrow><mi>∂</mi><msup><mi>i</mi> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msup></mrow></mfrac></mfenced> <mo>≤</mo> <mfenced separators="" open="|" close="|"><msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mn>1</mn><mo>)</mo></mrow></msubsup></mfenced> <mo>·</mo> <mo>...</mo> <mo>·</mo> <mfenced separators="" open="|" close="|"><msubsup><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msubsup></mfenced> <mo>·</mo> <mfenced separators="" open="|" close="|"><msubsup><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow> <mrow><mo>(</mo><mi>t</mi><mo>-</mo><mi>k</mi><mo>)</mo></mrow></msubsup></mfenced> <mo>=</mo> <msup><mfenced separators="" open="|" close="|"><msub><mi>w</mi> <mrow><mi>r</mi><mi>e</mi><mi>c</mi></mrow></msub></mfenced> <mi>k</mi></msup> <mo>·</mo> <msub><mi>w</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></mrow></math>

这个关系在很大程度上限制了时间为 <math alttext="t minus k"><mrow><mi>t</mi> <mo>-</mo> <mi>k</mi></mrow></math> 的输入变化对时间为*t*的隐藏状态的影响。因为我们模型的权重在训练开始时被初始化为小值，所以这个导数的值随着*k*的增加而接近零。换句话说，当计算与过去几个时间步的输入相关时，梯度迅速减小，严重限制了我们模型学习长期依赖关系的能力。这个问题通常被称为*梯度消失*问题，严重影响了普通RNN的学习能力。为了解决这个限制，我们将在下一节中探讨对循环层的一个极具影响力的变化，即长短期记忆。

# 长短期记忆单元

为了解决梯度消失的问题，Sepp Hochreiter和Jürgen Schmidhuber引入了*长短期记忆*（LSTM）架构。该架构背后的基本原则是，网络将被设计用于可靠地传输重要信息到未来的许多时间步。设计考虑导致了[图9-17](#architecture_of_an_lstm_unit)中所示的架构。

![](Images/fdl2_0917.png)

###### 图9-17\. LSTM单元的架构，以张量（由箭头表示）和操作（由内部块表示）级别进行说明

为了讨论的目的，我们将从单个神经元级别退后一步，开始讨论网络作为张量集合和张量上的操作。正如图所示，LSTM单元由几个关键组件组成。 LSTM架构的一个核心组件是*内存单元*，在图中心的粗体循环表示的张量。内存单元保存了它随时间学到的关键信息，并且网络被设计为在许多时间步长上有效地保持内存单元中的有用信息。在每个时间步长，LSTM单元通过三个不同的阶段用新信息修改内存单元。首先，单元必须确定要保留多少先前的记忆。这由*保持门*确定，详细显示在[图9-18](#architecture_of_the_keep_gate)中。

![](Images/fdl2_0918.png)

###### 图9-18\. LSTM单元保持门的架构

保持门的基本思想很简单。来自上一个时间步的内存状态张量充满信息，但其中一些信息可能已经过时（因此可能需要被擦除）。我们通过尝试计算一个比特张量（一个由零和一组成的张量）来弄清楚内存状态张量中哪些元素仍然相关，哪些元素是无关的。如果比特张量中的特定位置包含1，则表示内存单元中的该位置仍然相关且应该保留。如果该特定位置相反包含0，则表示内存单元中的该位置不再相关且应该被擦除。我们通过将本时间步的输入和上一个时间步的LSTM单元输出连接起来，并对结果张量应用sigmoid层来近似这个比特张量。正如您可能记得的那样，sigmoid神经元大部分时间输出接近0或接近1的值（唯一的例外是当输入接近0时）。因此，sigmoid层的输出是比特张量的一个接近近似，我们可以使用这个来完成保持门。

一旦我们弄清楚了要在旧状态中保留什么信息和要擦除什么信息，我们就准备考虑我们想要写入内存状态的信息。这部分LSTM单元称为*写入门*，在[图9-19](#architecture_of_the_write_gate)中描述。这可以分解为两个主要部分。第一个组件是弄清楚我们想要写入状态的信息。这通过tanh层计算以创建一个中间张量。第二个组件是弄清楚我们实际上想要包含到新状态中的计算张量的哪些组件，以及我们在写入之前想要丢弃哪些组件。我们通过使用与我们在保持门中使用的相同策略（一个sigmoid层）来近似一个由0和1组成的比特向量。我们将比特向量与我们的中间张量相乘，然后将结果相加以创建LSTM的新状态向量。

![](Images/fdl2_0919.png)

###### 图9-19\. LSTM单元写入门的架构

在每个时间步，我们希望LSTM单元提供一个输出。虽然我们可以直接将状态向量视为输出，但LSTM单元经过设计，通过发出一个输出张量来提供更多灵活性，该输出张量是对状态向量表示的“解释”或外部“通信”。输出门的架构显示在[图9-20](#architecture_of_the_output_gate)中。我们使用几乎相同的结构作为写入门：（1）tanh层从状态向量创建一个中间张量，（2）sigmoid层使用当前输入和先前输出产生一个位张量掩码，（3）中间张量与位张量相乘以产生最终输出。

![](Images/fdl2_0920.png)

###### 图9-20\. LSTM单元的输出门架构

那么为什么这比使用原始RNN单元更好呢？关键观察是当我们将LSTM单元在时间上展开时，信息如何在网络中传播。展开的架构显示在[图9-21](#unrolling_an_lstm_unit)中。在顶部，我们可以观察到状态向量的传播，其相互作用主要是线性的。结果是，将过去几个时间步的输入与当前输出相关联的梯度不会像在普通RNN架构中那样急剧减弱。这意味着LSTM可以比我们最初的RNN公式更有效地学习长期关系。

![](Images/fdl2_0921.png)

###### 图9-21\. 通过时间展开LSTM单元

最后，我们想要了解使用LSTM单元生成任意架构有多容易。LSTM有多“可组合”？我们是否需要牺牲灵活性来使用LSTM单元而不是普通的RNN？就像我们可以堆叠RNN层以创建更具表现力和容量的模型一样，我们可以堆叠LSTM单元，其中第二个单元的输入是第一个单元的输出，第三个单元的输入是第二个单元的输出，依此类推。[图9-22](#composimg_lstm_units)展示了由两个LSTM单元组成的多细胞架构的工作原理。这意味着我们可以在任何使用普通RNN层的地方轻松替换为LSTM单元。

![](Images/fdl2_0922.png)

###### 图9-22\. 就像我们可以堆叠循环层一样，组合LSTM单元

现在我们已经克服了梯度消失的问题，并了解了LSTM单元的内部工作原理，我们准备深入研究我们的第一个RNN模型的实现。

# PyTorch用于RNN模型的原语

PyTorch提供了几个原语，我们可以直接使用它们来构建RNN模型。首先，我们有代表RNN层或LSTM单元的`torch.nn.RNNCell`对象：

```py
import torch.nn as nn

cell_1 = nn.RNNCell(input_size = 10,
                    hidden_size = 20,
                    nonlinearity='tanh')

cell_2 = nn.LSTMCell(input_size = 10,
                     hidden_size = 20)

cell_3 = nn.GRUCell(input_size = 10,
                    hidden_size = 20)

```

`RNNCell`抽象表示普通的循环神经元层，而`LSTMCell`表示LSTM单元的实现。PyTorch还包括一种称为*门控循环单元*（GRU）的LSTM单元变体，由Yoshua Bengio的团队于2014年提出。所有这些单元的关键初始化变量是隐藏状态向量的大小或`hidden_size`。

除了原语，PyTorch还提供了用于堆叠层的多层RNN和LSTM类。如果我们想要堆叠循环单元或层，我们可以使用以下内容：

```py
multi_layer_rnn = nn.RNN(input_size = 10,
                         hidden_size = 20,
                         num_layers = 2,
                         nonlinearity = 'tanh')

multi_layer_lstm = nn.LSTM(input_size = 10,
                           hidden_size = 20,
                           num_layers = 2)

```

我们还可以使用`dropout`参数来对LSTM的输入和输出应用指定保留概率的dropout。如果`dropout`参数不为零，则模型会在每个LSTM层的输出上引入一个dropout层，除了最后一层，dropout概率等于`dropout`：

```py
multi_layer_rnn = nn.RNN(input_size = 10,
                         hidden_size = 20,
                         num_layers = 2,
                         nonlinearity = 'tanh',
                         batch_first = False,
                         dropout = 0.5)

multi_layer_lstm = nn.LSTM(input_size = 10,
                           hidden_size = 20,
                           num_layers = 2,
                           batch_first = False,
                           dropout = 0.5)

```

如图所示，多层RNN和LSTM类还提供了一个`batch_first`参数。如果`batch_first`等于`True`，那么输入和输出张量将以`(batch, seq, feature)`的形式提供，而不是`(seq, batch, feature)`。请注意，这不适用于隐藏状态或单元状态。`batch_first`的默认值为`False`。有关详细信息，请参阅PyTorch文档。

最后，通过调用PyTorch LSTM构造函数来实例化一个RNN：

```py
input = torch.randn(5, 3, 10) # (time_steps, batch, input_size)
h_0 = torch.randn(2, 3, 20) # (n_layers, batch_size, hidden_size)
c_0 = torch.randn(2, 3, 20) # (n_layers, batch_size, hidden_size)

rnn = nn.LSTM(10, 20, 2) # (input_size, hidden_size, num_layers)
output_n, (hn, cn) = rnn(input, (h_0, c_0))

```

调用`rnn`的结果是表示RNN输出的张量`output_n`，以及每一层的最终状态向量。第一个张量`hn`包含每一层的隐藏状态向量，保存了时间`n`时刻的输出门的输出。第二个张量`cn`包含每一层的记忆单元的状态向量，即写门的输出。`hn`和`cn`的大小均为`(n_layers, batch_size, hidden_size)`。

现在我们已经了解了在PyTorch中构建RNN的工具，接下来我们将在下一节中构建我们的第一个LSTM，重点是情感分析任务。

# 实现情感分析模型

在本节中，我们尝试分析来自大型电影评论数据集的电影评论的情感。该数据集包含来自IMDb的50,000条评论，每条评论都被标记为积极或消极情感。我们使用一个简单的LSTM模型利用辍学来学习如何对电影评论的情感进行分类。LSTM模型将逐个单词消耗电影评论。一旦它消耗完整个评论，我们将使用其输出作为二元分类的基础，将情感映射为“积极”或“消极”。

让我们从PyTorch库Torchtext开始加载数据集，Torchtext已经预装在Google Colab中。如果您在另一台机器上运行，可以通过运行以下命令来安装Torchtext：

```py
$ pip install torchtext

```

安装完包后，我们可以下载数据集并定义一个分词器。Torchtext通过`torchtext.datasets`和`torchtext.data.utils`子模块提供了许多自然语言处理（NLP）数据集和分词器。我们将使用内置的IMDb数据集和PyTorch提供的标准`'basic_english'`分词器。

```py
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer

# Load dataset
train_iter = IMDB(split=('train'))

# Define tokenizer and build vocabulary
tokenizer = get_tokenizer('basic_english')
```

到目前为止，我们一直在使用PyTorch的映射式数据集。Torchtext将NLP数据集返回为可迭代式数据集，这对于流式数据更为合适。接下来，我们需要基于训练数据集创建一个词汇表，并修剪词汇表，只包括最常见的30,000个单词。然后，我们需要将每个输入序列填充到500个单词的长度，并处理标签。

```py
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# build vocab from iterator and add a list of any special tokens
text_vocab = build_vocab_from_iterator(yield_tokens(train_iter),
                                       specials=['<unk>', '<pad>'])
text_vocab.set_default_index(text_vocab['<unk>'])

```

如上所示，Torchtext提供了一个函数`build_vocab_from_iterator`来创建词汇表。但是，该函数期望以标记列表作为输入，其中`next(train_iter)`会返回一个元组`(label_string, review_string)`。为了满足这一要求，我们定义了一个函数，在数据集被迭代时产生标记。最后，我们添加了未知和填充的特殊标记，并设置了默认值。

接下来，我们需要实际修剪词汇表并填充评论序列，以及将标签字符串`'neg'`或`'pos'`转换为数字。我们通过为标签和评论字符串定义一个流水线函数来实现这一点：

```py
def text_pipeline(x, max_size=512):
   text = tokenizer(x)

   # reduce vocab size
   pruned_text = []
   for token in text:
     if text_vocab.get_stoi()[token] >= 30000:
       token = '<unk>'
     pruned_text.append(token)

   # pad sequence or truncate
   if len(pruned_text) <= max_size:
     pruned_text += ['<pad>'] * (max_size - len(pruned_text))
   else:
     pruned_text = pruned_text[0:max_size]
   return text_vocab(pruned_text)

label_pipeline = lambda x: (0 if (x == 'neg') else 1)
```

`text_pipeline`函数将输入转换为500维向量。每个向量对应一个电影评论，其中向量的第i^th个分量对应于评论中第i^th个单词在我们的全局词典中的索引，该词典包含30,000个单词。为了完成数据准备，我们创建了一个特殊的Python类，用于从底层数据集中提供所需大小的小批量数据。

我们可以使用PyTorch中内置的`DataLoader`类来对数据集进行批处理采样。在这样做之前，我们需要定义一个函数`collate_batch`，告诉`DataLoader`如何预处理每个批次：

```py
def collate_batch(batch):
  label_list, text_list = [], []
  for label, review in batch:
    label_list.append(label_pipeline(label))
    text_list.append(text_pipeline(review))
  return (torch.tensor(label_list, dtype=torch.long),
          torch.tensor(text_list, dtype=torch.int32))

```

`collate_batch`函数只是通过每个相应的管道运行标签和评论字符串，并将批处理作为张量元组`(labels_batch, reviews_batch)`返回。一旦定义了`collate_fn`，我们只需使用IMDb构造函数加载数据集，并使用`DataLoader`构造函数配置数据加载器：

```py
from torch.utils.data import DataLoader

train_iter, val_iter = IMDB(split=('train','test'))
trainloader = DataLoader(train_iter,
                         batch_size = 4,
                         shuffle=False,
                         collate_fn=collate_batch)
valloader = DataLoader(val_iter,
                       batch_size = 4,
                       shuffle=False,
                       collate_fn=collate_batch)

```

我们使用`torchtext.datasets.IMDB` Python类来为我们在训练情感分析模型时使用的训练和验证集提供服务。

现在数据已经准备就绪，我们将逐步构建情感分析模型。首先，我们将希望将输入评论中的每个单词映射到一个单词向量。为此，我们将利用一个嵌入层，正如您可能从[第8章](ch08.xhtml#embedding_and_representing_learning)中回忆的那样，这是一个简单的查找表，存储与每个单词对应的嵌入向量。

与以前的例子不同，我们将学习词嵌入作为一个单独的问题（即通过构建Skip-Gram模型），我们将通过将嵌入矩阵视为完整问题中的参数矩阵来同时学习词嵌入和情感分析问题。我们通过使用PyTorch原语来管理嵌入来实现这一点（请记住，`input`代表一次完整的小批量，而不仅仅是一个电影评论向量）：

```py
import torch.nn as nn

embedding = nn.Embedding(
                      num_embeddings=30000,
                      embedding_dim=512,
                      padding_idx=text_vocab.get_stoi()['<pad>'])

```

然后，我们将嵌入层的结果传递给使用我们在前一节中看到的原语构建带有丢失的LSTM。LSTM的实现可以如下实现：

```py
class TextClassifier(nn.Module):
  def __init__(self):
    super(TextClassifier,self).__init__()
    self.layer_1 = nn.Embedding(
                      num_embeddings=30000,
                      embedding_dim=512,
                      padding_idx=1)                      
    self.layer_2 = nn.LSTMCell(input_size=512, hidden_size=512)
    self.layer_3 = nn.Dropout(p=0.5)
    self.layer_4 = nn.Sequential(
                      nn.Linear(512, 2),
                      nn.Sigmoid(),
                      nn.BatchNorm1d(2))

  def forward(self, x):
    x = self.layer_1(x)
    x = x.permute(1,0,2)
    h = torch.rand(x.shape[1], 512)
    c = torch.rand(x.shape[1], 512)
    for t in range(x.shape[0]):
      h, c = self.layer_2(x[t], (h,c))
      h = self.layer_3(h)
    return self.layer_4(h)

```

我们最后使用一个批归一化的隐藏层，与我们在以前的例子中一次又一次使用的隐藏层相同。将所有这些组件串联在一起，我们可以通过调用`TextClassifier`来构建模型：

```py
model = TextClassifier()

```

我们省略了设置摘要统计信息、保存中间快照和创建会话所涉及的其他样板，因为它与本书中构建的其他模型相同（请参阅[GitHub存储库](https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book)）。然后，我们可以使用TensorBoard运行和可视化我们模型的性能（参见[图9-23](#fig0723)）。

![](Images/fdl2_0923.png)

###### 图9-23。我们电影评论情感模型的训练成本、验证成本和准确率

在训练开始时，模型稍微稳定性不佳，而在训练结束时，模型明显开始过拟合，训练成本和验证成本明显分歧。然而，在其最佳性能时，模型表现相当有效，并且在测试集上大约达到86%的准确率。恭喜！您已经构建了您的第一个RNN。

# 使用循环神经网络解决seq2seq任务

现在我们已经建立了对RNN的深入理解，我们准备重新审视seq2seq问题。我们在本章开始时以一个seq2seq任务的示例开始：将句子中的单词序列映射到POS标签序列。解决这个问题是可行的，因为我们不需要考虑长期依赖性来生成适当的标签。但是有几个seq2seq问题，例如语言之间的翻译或为视频创建摘要，其中长期依赖性对模型的成功至关重要。这就是RNN的用武之地。

seq2seq的RNN方法看起来很像我们在上一章中讨论的自动编码器。seq2seq模型由两个独立的网络组成。第一个网络称为*编码器*网络。编码器网络是一个循环网络（通常使用LSTM单元），它消耗整个输入序列。编码器网络的目标是生成对输入的简洁理解，并将其总结为由编码器网络的最终状态表示的一个单一思想。然后我们使用一个*解码器*网络，其起始状态由编码器网络的最终状态初始化，逐个标记地生成目标输出序列。在每一步中，解码器网络将其上一个时间步的输出作为当前时间步的输入。整个过程在[图9-24](#encoder_decoder_recurrent_network_schema)中可视化。

！[](Images/fdl2_0924.png)

###### 图9-24。我们如何使用编码器/解码器循环网络架构来解决seq2seq问题

在这个设置中，我们试图将一句英语句子翻译成法语。我们对输入句子进行标记，并使用一个嵌入（类似于我们在前一节构建的情感分析模型中的方法），逐个单词作为编码器网络的输入。在句子结束时，我们使用一个特殊的“序列结束”（EOS）标记来指示输入序列的结束。然后我们取编码器网络的隐藏状态，并将其用作解码器网络的初始化。解码器网络的第一个输入是EOS标记，输出被解释为预测的法语翻译的第一个单词。从那时起，我们使用解码器网络的输出作为下一个时间步的输入。直到解码器网络发出EOS标记作为其输出，此时我们知道网络已经完成了对原始英语句子的翻译。我们将在本章后面剖析这个网络的实际开源实现（通过一些增强和技巧来提高准确性）。

seq2seq RNN架构也可以重新用于学习序列的良好嵌入。例如，2015年Kiros等人发明了*skip-thought向量*的概念，借鉴了自动编码器框架和第8章中讨论的Skip-Gram模型的结构特征。skip-thought向量通过将一段话分成一组由连续句子组成的三元组来生成。作者使用了一个编码器网络和两个解码器网络，如[图9-25](#skip_thought_seq2seq_architecture)所示。

！[](Images/fdl2_0925.png)

###### 图9-25。skip-thought seq2seq架构用于生成整个句子的嵌入表示

编码器网络消耗了我们想要生成简洁表示的句子（存储在编码器网络的最终隐藏状态中）。然后是解码步骤。第一个解码器网络将以该表示作为其自己隐藏状态的初始化，并尝试重建出出现在输入句子之前的句子。第二个解码器网络将尝试出现在输入句子之后的句子。整个系统在这些三元组上端到端地进行训练，一旦完成，就可以用来生成看似连贯的文本段落，同时提高关键句子级分类任务的性能。

以下是从原始论文中摘录的故事生成示例：

```py
she grabbed my hand .
"come on . "
she fluttered her back in the air .
"i think we're at your place . I ca n't come get you . "
he locked himself back up
" no . she will . "
kyrian shook his head
```

现在我们已经了解了如何利用RNN来解决seq2seq问题，我们几乎准备好尝试构建自己的模型了。然而，在那之前，我们还有一个重要的挑战要解决，在下一节中我们将直面讨论seq2seq RNN中的注意力概念。

# 用注意力增强循环网络

让我们更深入地思考翻译问题。如果你曾尝试学习一门外语，你会知道在完成翻译时有几个有用的步骤。首先，阅读完整的句子以理解你想要传达的概念是有帮助的。然后，逐字逐句地写出翻译，每个词都逻辑地跟在你之前写的词后面。但翻译的一个重要方面是，当你构成新句子时，你经常会参考原始文本，关注与当前翻译相关的特定部分。在每一步中，你都在关注原始“输入”的最相关部分，以便能够做出关于下一个要写在页面上的词的最佳决定。

回想一下我们对seq2seq的方法。通过消耗完整的输入并将其总结为“思考”隐藏在其隐藏状态中，编码器网络有效地实现了翻译过程的第一部分。通过使用先前的输出作为当前输入，解码器网络实现了翻译过程的第二部分。然而，我们的seq2seq方法尚未捕捉到*注意力*这一现象，这是我们需要工程化的最后一个构建块。

当前，在给定时间步骤*t*，解码器网络的唯一输入是在时间步骤<t-1>的输出。给解码器网络一些关于原始句子的视觉信息的一种方法是让解码器访问编码器网络的所有输出（我们之前完全忽略了）。这些输出对我们很有趣，因为它们代表编码器网络在看到每个新标记后内部状态的演变。这种策略的一个提议实现如[图9-26](#attempt_at_engineering_attentional_abilities)所示。这个尝试失败了，因为它未能动态选择要关注的输入的最相关部分。

![](Images/fdl2_0926.png)

###### 图9-26。在seq2seq架构中尝试工程化注意力能力

然而，这种方法存在一个关键缺陷。问题在于，在每个时间步骤，解码器以完全相同的方式考虑编码器网络的所有输出。然而，在翻译过程中，人类显然不是这样做的。在翻译不同部分时，我们会关注原始文本的不同方面。关键的认识在于，仅仅让解码器访问所有输出是不够的。相反，我们必须设计一种机制，使解码器网络能够动态关注编码器输出的特定子集。

我们可以通过改变连接操作的输入来解决这个问题，使用Bahdanau等人2015年的提议作为灵感。而不是直接使用编码器网络的原始输出，我们对编码器的输出进行加权操作。我们利用解码器网络在时间<t-1>的状态作为加权操作的基础。

权重操作在[图9-27](#modification_to_our_original_proposal)中进行了说明。首先，我们为编码器的每个输出创建一个标量（一个单一的数字，而不是张量）相关性分数。该分数是通过计算每个编码器输出与时间<t减1>的解码器状态之间的点积来生成的。然后，我们使用softmax操作对这些分数进行归一化。最后，我们使用这些归一化的分数来分别缩放编码器的输出，然后将它们插入连接操作中。关键在于，为每个编码器输出计算的相对分数表示该特定编码器输出对于解码器在时间步*t*上的决策有多重要。事实上，正如我们将在后面看到的，我们可以通过检查softmax的输出来可视化哪些输入部分对于每个时间步的翻译最为相关。

![](Images/fdl2_0927.png)

###### 图9-27. 根据上一个时间步的解码器网络的隐藏状态实现的动态注意力机制的我们原始提议的修改

掌握了将注意力引入seq2seq架构的策略后，我们终于准备好使用RNN模型将英语句子翻译成法语。但在我们开始之前，值得注意的是，注意力在超越语言翻译的问题中非常适用。在语音转文本问题中，注意力可以帮助算法动态关注音频的相应部分，同时将音频转录为文本。同样，注意力可以用于改进图像字幕算法，帮助字幕算法在撰写字幕时专注于输入图像的特定部分。每当输入的特定部分与正确生成相应输出的部分高度相关时，注意力都可以显著提高性能。

# 剖析神经翻译网络

最先进的神经翻译网络使用了许多不同的技术和进展，这些技术和进展建立在基本的seq2seq编码器-解码器架构之上。注意力机制，如前一节所详细介绍的，是一个重要且关键的架构改进。在本节中，我们将剖析一个完全实现的神经机器翻译系统，包括数据处理步骤，构建模型，训练模型，最终将其用作翻译系统，将英语短语转换为法语短语。

在训练和最终使用神经机器翻译系统的流程与大多数机器学习流程类似：收集数据，准备数据，构建模型，训练模型，评估模型的进展，最终使用训练好的模型来预测或推断一些有用的东西。我们在这里回顾每个步骤。

我们首先从[国际口语翻译研讨会（IWSLT2016）存储库](https://wit3.fbk.eu/2016-01)中收集数据，该存储库包含用于训练翻译系统的大型语料库。对于我们的用例，我们将使用英语到法语的数据。请注意，如果我们想要能够翻译不同语言，我们将不得不使用新数据从头开始训练模型。然后，我们将数据预处理为在训练和推断时我们的模型可以轻松使用的格式。这将涉及对英语和法语短语中的句子进行一定程度的清理和标记化。接下来是一系列用于准备数据的技术，稍后我们将介绍这些技术的实现。

第一步是通过*标记化*将句子和短语解析为更适合模型的格式。这是将特定的英语或法语句子离散化为其组成标记的过程。例如，一个简单的单词级标记器将消耗句子“I read.”来生成数组["I", "read", "."]，或者它将消耗法语句子“Je lis.”来生成数组["Je", "lis", "."]。

字符级标记化可能会将句子分解为单个字符或成对字符，如["I", " ", "r", "e", "a", "d", "."]和["I ", "re", "ad", "."]。一种标记化方式可能比另一种更好，每种都有其优缺点。例如，单词级标记化将确保模型生成来自某个字典的单词，但字典的大小可能太大，以至于在解码过程中难以有效地选择。这实际上是一个已知的问题，我们将在接下来的讨论中解决。

另一方面，使用字符级标记化的解码器可能不会产生可理解的输出，但解码器必须选择的总字典要小得多，因为它只是所有可打印ASCII字符的集合。在本教程中，我们使用单词级标记化，但我们鼓励您尝试不同的标记化以观察其影响。值得注意的是，我们还必须在所有输出序列的末尾添加一个特殊的EOS字符，因为我们需要提供一种明确的方式让解码器指示它已经解码结束。我们不能使用常规标点，因为我们不能假设我们正在翻译完整的句子。请注意，我们在源序列中不需要EOS字符，因为我们正在预先格式化地喂入这些数据，不需要一个EOS字符来表示我们源序列的结束。

下一个优化涉及进一步修改我们如何表示每个源序列和目标序列的方式，并引入了一个称为*桶装*的概念。这是一种主要用于序列到序列任务，特别是机器翻译的方法，可以帮助模型有效地处理不同长度的句子或短语。我们首先描述了喂入训练数据的朴素方法，并说明了这种方法的缺点。通常，在喂入编码器和解码器标记时，源序列和目标序列的长度在不同示例对之间并不总是相等。例如，源序列可能长度为*X*，目标序列可能长度为*Y*。看起来我们需要不同的seq2seq网络来适应每个(*X, Y*)对，但这似乎立即显得浪费和低效。相反，我们可以稍微改进，如果我们将每个序列*填充*到一定长度，如[图9-28](#naive_strategy_for_padding_sequences)所示，假设我们使用单词级的标记化，并且我们已经在目标序列中添加了EOS标记。

![](Images/fdl2_0928.png)

###### 图9-28。填充序列的朴素策略

这一步省去了为每对源和目标长度构建不同的seq2seq模型的麻烦。然而，这引入了另一个问题：如果有一个非常长的序列，那意味着我们必须将每个其他序列*填充到该长度*。这将使一个短序列填充到末尾所需的计算资源与一个带有少量填充标记的长序列一样多，这是浪费的，可能会给我们的模型带来重大性能损失。我们可以考虑将语料库中的每个句子分解为短语，使得每个短语的长度不超过某个最大限制，但如何分解相应的翻译并不清楚。这就是桶装如何帮助我们的地方。

分桶是一个想法，我们可以将编码器和解码器对放入相似大小的桶中，并且只填充到每个桶中序列的最大长度。例如，我们可以表示一组桶，[(5, 10), (10, 15), (20, 25), (30, 40)]，列表中的每个元组分别是源序列和目标序列的最大长度。借用前面的例子，我们可以将序列对(["I", "read", "."], ["Je", "lis", ".", "EOS"])放入第一个桶中，因为源序列小于5个标记，目标序列小于10个标记。然后我们将(["See", "you", "in", "a", "little", "while"], ["A", "tout", "a", "l’heure", "EOS"])放入第二个桶，依此类推。这种技术允许我们在两个极端之间取得折衷，只填充必要的部分，如[图9-29](#padding_sequences_with_buckets)所示。

![](Images/fdl2_0929.png)

###### 图9-29。使用桶填充序列

使用分桶在训练和测试时显示出显著的加速，并且允许开发人员和框架编写非常优化的代码，以利用任何来自桶的序列将具有相同的大小，并以一种允许进一步GPU效率的方式打包数据。

在正确填充序列后，我们需要为目标序列添加一个额外的标记：*一个GO标记*。这个GO标记将向解码器发出信号，表示解码需要开始，在这一点上，解码器将接管并开始解码。

在数据准备方面我们做的最后一个改进是反转源序列。研究人员发现这样做可以提高性能，这已经成为训练神经机器翻译模型时尝试的标准技巧。这有点工程上的技巧，但考虑到我们的固定大小神经状态只能容纳有限信息，处理句子开头时编码的信息可能会在处理句子后部时被覆盖。在许多语言对中，句子开头比句子结尾更难翻译，因此通过反转句子的这种技巧可以通过让句子开头最后发言来提高翻译准确性。有了这些想法，最终的序列看起来如[图9-30](#padding_scheme_reversing_the_inputs)所示。

![](Images/fdl2_0930.png)

###### 图9-30。最终的填充方案，使用桶，反转输入，并添加GO标记

使用这些描述的技术，我们现在可以详细说明实现。首先，我们加载数据集，然后定义我们的分词器和词汇表。我们不在这里定义词嵌入，因为我们将训练我们的模型来计算它们。PyTorch的Torchtext库支持`torch.text.datasets`中的IWSLT2016：

```py
from torchtext.datasets import IWSLT2016

train_iter = IWSLT2016(split=('train'),
                       language_pair=('en','fr'))

```

数据集构造函数返回一个可迭代样式数据集，可以使用`next(train_iter)`检索英语和法语句子对。我们将使用这种可迭代样式数据集在代码中稍后创建分桶数据集以进行批处理。

现在，让我们为每种语言定义分词器和词汇表。PyTorch提供了一个`get_tokenizer`函数，可用于常见的分词器。在这里，我们将为每种语言使用`spacy`分词器。您可能需要先下载`spacy`语言文件：

```py
pip install -U spacy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm

```

一旦我们有了语言文件，我们可以按以下方式创建分词器：

```py
from torchtext.data.utils import get_tokenizer

tokenizer_en = get_tokenizer('spacy',language='en_core_web_sm')
tokenizer_fr = get_tokenizer('spacy',language='fr_core_news_sm')

```

接下来，我们将使用PyTorch的`build_vocab_from_iterator`函数为英语和法语创建词汇表。该函数从单一语言的可迭代样式数据集中获取标记并创建词汇表。由于我们的数据集包含英语和法语句子，我们创建一个`yield_tokens`函数，仅返回英语或法语标记，并将其传递给`build_vocab_from_iterator`：

```py
def yield_tokens(data_iter, language):
    if language == 'en':
      for data_sample in data_iter:
          yield tokenizer_en(data_sample[0])
    else:
      for data_sample in data_iter:
        yield tokenizer_fr(data_sample[1])

UNK_IDX, PAD_IDX, GO_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<go>', '<eos>']

# Create Vocabs
train_iter = IWSLT2016(root='.data', split=('train'),
                             language_pair=('en', 'fr'))

vocab_en = build_vocab_from_iterator(
                  yield_tokens(train_iter, 'en'),
                  min_freq=1,
                  specials=special_symbols,
                  special_first=True)

train_iter = IWSLT2016(root='.data', split=('train'),
                             language_pair=('en', 'fr'))
vocab_fr = build_vocab_from_iterator(
                  yield_tokens(train_iter, 'fr'),
                  min_freq=1,
                  specials=special_symbols,
                  special_first=True)

```

请注意，在构建法语词汇表之前，我们需要重新加载`train_iter`以重新启动可迭代样式数据集。我们还添加了特殊标记及其索引。

现在我们有了数据集、分词器和词汇表，我们需要创建函数来预处理标记并生成分桶数据的批次。首先让我们定义一个`process_tokens`函数来应用我们之前讨论的改进：

```py
def process_tokens(source, target, bucket_sizes):
  # find bucket_index
  for i in range(len(bucket_sizes)+2):
    # truncate if we exhauset list of buckets
    if i >= len(bucket_sizes):
      bucket = bucket_sizes[i-1]
      bucket_id = i-1
      if len(source) > bucket[0]:
        source = source[:bucket[0]]
      if len(target) > (bucket[1]-2):
        target = target[:bucket[1]-2]
      break

    bucket = bucket_sizes[i]
    if (len(source) < bucket[0]) and ((len(target)+1) < bucket[1]):
      bucket_id = i
      break

  source += ((bucket_sizes[bucket_id][0] - len(source)) * ['<pad>'])
  source = list(reversed(source))

  target.insert(0,'<go>')
  target.append('<eos>')
  target += (bucket_sizes[bucket_id][1] - len(target)) * ['<pad>']

  return vocab_en(source), vocab_fr(target), bucket_id

```

在这个函数中，我们传入变量大小的源标记和目标标记列表，以及一个桶大小列表。首先，我们决定适合源标记和目标标记列表的最小桶大小。然后，我们通过填充和反转序列来处理源标记，如前面所述。对于目标标记，我们在开头添加一个`<go>`标记，并在末尾添加一个`<eos>`标记，然后填充到桶大小。在确定最小桶大小时，我们考虑了两个添加的标记`<go>`和`<eos>`。

现在我们有一个函数，它接受源标记和目标标记的列表，并适当地准备它们。接下来，我们需要为我们的模型和训练循环收集一批数据。为此，我们将使用内置的PyTorch`Dataset`和`DataLoader`类。

我们将为每个桶大小分开`Dataset`和`DataLoader`。这种方法将使我们能够利用`DataLoader`的内置功能进行随机分批和并行处理。

首先，我们通过对PyTorch的`Dataset`类进行子类化来创建一个`BucketedDataset`类。由于这将是一个映射样式的数据集，我们需要为数据访问定义`__getitem__`和`__len__`方法：

```py
from torch.utils.data import Dataset

class BucketedDataset(Dataset):
  def __init__(self, bucketed_dataset, bucket_size):
    super(BucketedDataset, self).__init__()
    self.length = len(bucketed_dataset)
    self.input_len = bucket_size[0]
    self.target_len = bucket_size[1]
    self.bucketed_dataset = bucketed_dataset

  def __getitem__(self, index):
    return (torch.tensor(self.bucketed_dataset[index][0],
                         dtype=torch.float32),
            torch.tensor(self.bucketed_dataset[index][1],
                         dtype=torch.float32))

  def __len__(self):
    return self.length

bucketed_datasets = []
for i, dataset in enumerate(datasets):
  bucketed_datasets.append(BucketedDataset(dataset,
                                           bucket_sizes[i]))

```

我们在`bucketed_datasets`中创建了一个`BucketedDataset`对象的列表，每个桶大小一个。`BucketedDataset`构造函数还将我们的词汇整数转换为PyTorch张量，以便我们稍后可以将它们传递给我们的模型。

接下来，我们使用PyTorch的`DataLoader`类为`bucketed_datasets`中的每个数据集创建数据加载器。由于我们创建了`Dataset`对象，我们可以在不编写任何额外代码的情况下获得`DataLoader`类的批处理能力：

```py
from torch.utils.data import DataLoader

dataloaders = []
for dataset in bucketed_datasets:
  dataloaders.append(DataLoader(dataset,
                                batch_size=32,
                                shuffle=True))

```

数据加载器列表保存了每个桶大小的数据加载器，因此当我们运行训练或测试循环时，我们将选择一个桶大小（对于训练是随机的），并使用相应的数据加载器来获取一批编码器和解码器输入：

```py
for epoch in range(n_epochs):
  # exhaust all dataloaders randomly
  # keep track of when we used up all values
  dataloader_sizes = []
  for dataloader in dataloaders:
    dataloader_sizes.append(len(dataloader))

  while np.array(dataloader_sizes).sum() != 0:
    bucket_id = torch.randint(0,len(bucket_sizes),(1,1)).item()
    if dataloader_sizes[bucket_id] == 0:
      continue
    source, target = next(iter(dataloaders[bucket_id]))
    dataloader_sizes[bucket_id] -= 1
    loss = train(encoder_inputs,
                 decoder_inputs,
                 target_weights,
                 bucket_id)

```

我们测量预测时间产生的损失，同时跟踪其他运行指标：

```py
loss += step_loss / steps_per_checkpoint current_step += 1

```

最后，根据全局变量的指示，我们会定期执行一些任务。首先，我们打印出先前批次的统计信息，如损失、学习率和困惑度。如果发现损失没有减少，模型可能已经陷入局部最优解。为了帮助模型摆脱这种情况，我们会降低学习率，以便它不会朝任何特定方向大幅跳跃。在这一点上，我们还会将模型及其权重和激活保存到磁盘上。

这就结束了关于训练和使用模型的高层细节。我们已经大大抽象了模型本身的细节。更多内容，请参阅[书籍的存储库](https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book)。

通过这样，我们成功完成了一个相当复杂的神经机器翻译系统实现细节的完整介绍。生产系统有一些额外的技巧，这些技巧不太具有一般性，并且这些系统是在巨大的计算服务器上训练的，以确保达到最先进的性能。

参考资料，这个确切的模型在八个NVIDIA Telsa M40 GPU上训练了四天。我们在图[9-31](#plot_of_perplexity_on_training_data)和[9-32](#plot_of_learning_rate_over_time)中展示了困惑度的图表，并展示了随时间变化的学习率。在[图9-31](#plot_of_perplexity_on_training_data)中，我们看到在50,000个epochs之后，困惑度从大约6下降到4，这对于神经机器翻译系统来说是一个合理的分数。在[图9-32](#plot_of_learning_rate_over_time)中，我们观察到学习率几乎平稳地下降到0。这意味着在我们停止训练时，模型正在接近稳定状态。

![](Images/fdl2_0931.png)

###### 图9-31. 随时间变化的训练数据困惑度图

![](Images/fdl2_0932.png)

###### 图9-32. 随时间变化的学习率图

为了更明确地展示注意力模型，我们可以可视化解码器LSTM在将句子从英语翻译成法语时计算的注意力。特别是，我们知道当编码器LSTM更新其单元状态以将句子压缩为连续的向量表示时，它还会在每个时间步计算隐藏状态。我们知道解码器LSTM对这些隐藏状态进行凸组求和，我们可以将这个求和看作是注意力机制；当某个隐藏状态上有更多的权重时，我们可以将其解释为模型更多地关注在该时间步输入的标记上。这正是我们在[图9-33](#explicitly_viz_the_weights)中可视化的内容。

要翻译的英语句子在顶行，结果的法语翻译在第一列。方块颜色越浅，解码器在解码该行元素时对该特定列付出的注意力越多。也就是说，注意力图中的*(i, j)^(th)*元素显示了在将法语句子的*i^(th)*标记翻译为英语句子的*j^(th)*标记时，对英语句子的*j^(th)*标记付出的注意力量。

我们立即看到注意力机制似乎运作得相当不错。尽管模型的预测中存在轻微噪音，但通常会在正确的区域放置大量注意力。添加网络层可能有助于产生更清晰的注意力。一个令人印象深刻的方面是，“the European Economic”这个短语在法语中被翻译为“zone économique européenne”，因此注意力权重反映了这种翻转。当从英语翻译到不从左到右顺畅解析的不同语言时，这种注意力模式可能会更有趣。

![](Images/fdl2_0933.png)

###### 图9-33. 当解码器关注编码器中的隐藏状态时，可视化凸组的权重

随着最基本的架构被理解和实现，我们现在将继续研究RNN的令人兴奋的新发展，并开始探索更复杂的学习。

# 自注意力和变压器

早些时候，我们讨论了一种注意力形式，这种形式首次在2015年由Bahdanau等人提出。具体来说，我们使用一个简单的前馈神经网络来计算每个编码器隐藏状态与当前时间步的解码器状态的对齐分数。在本节中，我们将讨论一种称为*缩放点积注意力*的不同形式的注意力，它在*自注意力*中的使用，以及*变压器*，这是一种最近的语言建模突破。基于变压器的模型主要取代了LSTM，并已被证明在许多序列到序列问题中具有更高的质量。

点积注意力实际上就像听起来的那样简单——这种方法计算编码器隐藏状态之间的点积作为对齐分数。这些权重用于计算上下文向量，这是编码器隐藏状态的凸组合（通过softmax）。为什么使用点积来衡量对齐？正如我们在[第1章](ch01.xhtml#fundamentals_of_linear_algebra_for_deep_learning)中学到的，两个向量的点积可以表示为两个向量的范数和它们之间夹角的余弦的乘积。当两个向量之间的夹角趋近于零时，余弦趋近于一。此外，从三角学中我们知道，当输入角度在0度到180度之间时，余弦的范围是1到-1，这是我们需要考虑的角度域的唯一部分。点积具有一个很好的性质，即当两个向量之间的夹角变小时，点积变大。这使我们能够将点积作为相似性的自然度量。

2017年，Vaswani等人^([7](ch09.xhtml#idm45934165152352))通过引入一个缩放因子——隐藏状态维度的平方根，对现有的点积注意力框架进行了修改。Vaswani等人承认，随着隐藏状态表示在维度上变得越来越大，我们预计会看到更多高幅度的点积。为了理解包含这个缩放因子的原因，假设每个 <math alttext="h Subscript i"><msub><mi>h</mi> <mi>i</mi></msub></math> 的索引都是从均值为零、单位方差的随机变量中独立且相同地抽取的。让我们计算它们的点积的期望和方差：

<math alttext="double-struck upper E left-bracket s Subscript t Superscript upper T Baseline h Subscript i Baseline right-bracket equals sigma-summation Underscript j equals 1 Overscript k Endscripts double-struck upper E left-bracket s Subscript t comma j Baseline asterisk h Subscript i comma j Baseline right-bracket"><mrow><mi>𝔼</mi> <mrow><mo>[</mo> <msubsup><mi>s</mi> <mi>t</mi> <mi>T</mi></msubsup> <msub><mi>h</mi> <mi>i</mi></msub> <mo>]</mo></mrow> <mo>=</mo> <msubsup><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>k</mi></msubsup> <mi>𝔼</mi> <mrow><mo>[</mo> <msub><mi>s</mi> <mrow><mi>t</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>*</mo> <msub><mi>h</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>]</mo></mrow></mrow></math>

= <math alttext="sigma-summation Underscript j equals 1 Overscript k Endscripts double-struck upper E left-bracket s Subscript t comma j Baseline right-bracket double-struck upper E left-bracket h Subscript i comma j Baseline right-bracket"><mrow><msubsup><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>k</mi></msubsup> <mi>𝔼</mi> <mrow><mo>[</mo> <msub><mi>s</mi> <mrow><mi>t</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>]</mo></mrow> <mi>𝔼</mi> <mrow><mo>[</mo> <msub><mi>h</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>]</mo></mrow></mrow></math>

<math alttext="equals 0"><mrow><mo>=</mo> <mn>0</mn></mrow></math>

<math alttext="upper V a r left-parenthesis s Subscript t Superscript upper T Baseline h Subscript i Baseline right-parenthesis equals sigma-summation Underscript j equals 1 Overscript k Endscripts upper V a r left-parenthesis s Subscript t comma j Baseline asterisk h Subscript i comma j Baseline right-parenthesis"><mrow><mi>V</mi> <mi>a</mi> <mi>r</mi> <mrow><mo>(</mo> <msubsup><mi>s</mi> <mi>t</mi> <mi>T</mi></msubsup> <msub><mi>h</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>=</mo> <msubsup><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>k</mi></msubsup> <mi>V</mi> <mi>a</mi> <mi>r</mi> <mrow><mo>(</mo> <msub><mi>s</mi> <mrow><mi>t</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>*</mo> <msub><mi>h</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>)</mo></mrow></mrow></math>

<math alttext="equals sigma-summation Underscript j equals 1 Overscript k Endscripts double-struck upper E left-bracket left-parenthesis s Subscript t comma j Superscript 2 Baseline asterisk h Subscript i comma j Superscript 2 Baseline right-parenthesis right-bracket minus double-struck upper E left-bracket s Subscript t comma j Baseline asterisk h Subscript i comma j Baseline right-bracket squared"><mrow><mo>=</mo> <msubsup><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>k</mi></msubsup> <mi>𝔼</mi> <mrow><mo>[</mo> <mrow><mo>(</mo> <msubsup><mi>s</mi> <mrow><mi>t</mi><mo>,</mo><mi>j</mi></mrow> <mn>2</mn></msubsup> <mo>*</mo> <msubsup><mi>h</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow> <mn>2</mn></msubsup> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>-</mo> <mi>𝔼</mi> <msup><mrow><mo>[</mo><msub><mi>s</mi> <mrow><mi>t</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>*</mo><msub><mi>h</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>]</mo></mrow> <mn>2</mn></msup></mrow></math>

<math alttext="equals sigma-summation Underscript j equals 1 Overscript k Endscripts double-struck upper E left-bracket s Subscript t comma j Superscript 2 Baseline right-bracket double-struck upper E left-bracket h Subscript i comma j Superscript 2 Baseline right-bracket"><mrow><mo>=</mo> <msubsup><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>k</mi></msubsup> <mi>𝔼</mi> <mrow><mo>[</mo> <msubsup><mi>s</mi> <mrow><mi>t</mi><mo>,</mo><mi>j</mi></mrow> <mn>2</mn></msubsup> <mo>]</mo></mrow> <mi>𝔼</mi> <mrow><mo>[</mo> <msubsup><mi>h</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow> <mn>2</mn></msubsup> <mo>]</mo></mrow></mrow></math>

<math alttext="equals sigma-summation Underscript j equals 1 Overscript k Endscripts 1"><mrow><mo>=</mo> <msubsup><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>k</mi></msubsup> <mn>1</mn></mrow></math>

<math alttext="equals k"><mrow><mo>=</mo> <mi>k</mi></mrow></math>

让我们回顾一下导致我们对期望和方差得出这些结论的步骤。期望中的第一个等式是由于期望的线性性，因为点积可以表示为每个索引的乘积之和。第二个等式来自于每个期望中的两个随机变量是独立的这一事实，因此我们可以将乘积的期望分离为期望的乘积。最后一步直接来自于这些单独期望都为零的事实。

方差中的第一个等式是由于当各个项都是独立时方差的线性性。第二个等式只是方差的定义。第三个等式使用了我们计算点积期望的结果（我们可以将期望的平方分离为期望的平方的乘积，其中每个单独的期望都为零）。此外，平方的乘积的期望可以分解为平方的期望的乘积，因为每个随机变量的平方与所有其他随机变量的平方是独立的。倒数第二个等式来自于每个随机变量的平方的期望就是随机变量的方差（因为每个随机变量的期望为零）。最后一个等式直接得出。

我们看到点积的期望值为零，而其方差为k，即隐藏表示的维度。因此，随着维度的增加，方差增加——这意味着更高概率看到高幅度的点积。

不幸的是，随着更多高幅度点积的存在，由于softmax函数，梯度变得更小。虽然我们不会在这里推导它，但这在直觉上是有意义的——回想一下在神经网络中用于分类问题的softmax的使用。随着神经网络对正确预测变得越来越自信（即对真实索引的高logit值），梯度变得越来越小。Vaswani等人引入的缩放因子减小了点积的幅度，导致更大的梯度和更好的学习。

现在我们已经介绍了缩放的点积注意力，我们将注意力转向自注意力。在前面的章节中，我们通过机器翻译的背景看到了注意力，其中我们有一个包含英语和法语句子的训练集，目标是能够将看不见的英语句子翻译成法语。在这类特定问题中，通过目标法语句子存在直接监督。然而，注意力也可以完全自包含地使用。直觉是，给定一个英语句子，我们可能能够通过学习句子或段落中标记之间的关系来执行更深入的情感分析、更有效的机器阅读和更好的理解。

转换器，我们本节的最后一个主题，利用了缩放的点积注意力和自注意力。转换器架构（Vaswani等人，2017）具有编码器和解码器架构，其中编码器和解码器内部都存在自注意力，以及编码器和解码器之间的标准注意力。编码器和解码器中的自注意力层允许每个位置在其各自架构中关注当前位置之前的所有位置。标准注意力允许解码器关注每个编码器隐藏状态，如前文所述。

# 摘要

在本章中，我们深入探讨了序列分析的世界。我们分析了如何修改前馈网络以处理序列，发展了对RNN的深刻理解，并探讨了注意力机制如何实现从语言翻译到音频转录等令人难以置信的应用。序列分析是一个领域，不仅涉及自然语言问题，还涉及金融领域的主题，比如对金融资产回报的时间序列分析。任何涉及纵向分析或跨时间分析的领域都可以使用本章描述的序列分析应用。我们建议您通过在不同领域实施序列分析来加深对其的理解，并通过将自然语言技术的结果与各领域的最新技术进行比较。在某些情况下，本文介绍的技术可能不是最合适的建模选择，我们建议您深入思考为什么这里所做的建模假设可能不适用于广泛应用。序列分析是一个强大的工具，在几乎所有技术应用中都有一席之地，不仅仅是自然语言。

^([1](ch09.xhtml#idm45934164188240-marker)) Nivre, Joakim. “Incrementality in Deterministic Dependency Parsing.” *Proceedings of the Workshop on Incremental Parsing: Bringing Engineering and Cognition Together*. Association for Computational Linguistics, 2004; Chen, Danqi, and Christopher D. Manning. “A Fast and Accurate Dependency Parser Using Neural Networks.” *EMNLP*. 2014.

^([2](ch09.xhtml#idm45934164127728-marker)) Andor, Daniel, et al. “Globally Normalized Transition-Based Neural Networks.” *arXiv preprint* *arXiv*:1603.06042 (2016).

^([3](ch09.xhtml#idm45934164119456-marker)) 同上。

^([4](ch09.xhtml#idm45934164070368-marker)) Kilian, Joe, 和 Hava T. Siegelmann。“Sigmoid神经网络的动态普适性。” *信息与计算* 128.1 (1996): 48-56。

^([5](ch09.xhtml#idm45934164517408-marker)) Kiros, Ryan, 等。“Skip-Thought Vectors。” *神经信息处理系统的进展*。2015年。

^([6](ch09.xhtml#idm45934164490960-marker)) Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. “通过联合学习对齐和翻译的神经机器翻译。” *arXiv预印本arXiv*:1409.0473 (2014)。

^([7](ch09.xhtml#idm45934165152352-marker)) Vaswani等。“注意力就是一切。” *arXiv预印本arXiv*:1706.03762 2017年。
