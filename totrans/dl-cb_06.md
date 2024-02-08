# 第六章：问题匹配

我们现在已经看到了一些示例，说明我们如何构建和使用词嵌入来比较术语。自然而然地，我们会问如何将这个想法扩展到更大的文本块。我们能否为整个句子或段落创建语义嵌入？在本章中，我们将尝试做到这一点：我们将使用来自 Stack Exchange 的数据为整个问题构建嵌入；然后我们可以使用这些嵌入来查找相似的文档或问题。

我们将从互联网档案馆下载和解析我们的训练数据。然后我们将简要探讨 Pandas 如何帮助分析数据。当涉及到对数据进行特征化和构建模型时，我们让 Keras 来处理繁重的工作。然后我们将研究如何从 Pandas `DataFrame`中提供此模型以及如何运行它以得出结论。

本章的代码可以在以下笔记本中找到：

```py
06.1 Question matching
```

# 6.1 从 Stack Exchange 获取数据

## 问题

您需要访问大量问题来启动您的训练。

## 解决方案

使用互联网档案馆检索问题的转储。

Stack Exchange 数据转储可以在[互联网档案馆](https://archive.org/details/stackexchange)上免费获取，该网站提供许多有趣的数据集（并努力提供整个网络的存档）。数据以一个 ZIP 文件的形式布置在 Stack Exchange 的每个领域上（例如旅行、科幻等）。让我们下载旅行部分的文件：

```py
xml_7z = utils.get_file(
    fname='travel.stackexchange.com.7z',
    origin=('https://ia800107.us.archive.org/27/'
            'items/stackexchange/travel.stackexchange.com.7z'),
)
```

虽然输入技术上是一个 XML 文件，但结构足够简单，我们可以通过仅读取单独的行并拆分字段来处理。当然，这有点脆弱。我们将限制自己处理数据集中的 100 万条记录；这可以避免内存使用过多，并且应该足够让我们处理。我们将处理后的数据保存为 JSON 文件，这样下次就不必再次处理：

```py
def extract_stackexchange(filename, limit=1000000):
    json_file = filename + 'limit=%s.json' % limit

    rows = []
    for i, line in enumerate(os.popen('7z x -so "%s" Posts.xml'
                             % filename)):
        line = str(line)
        if not line.startswith('  <row'):
            continue

        if i % 1000 == 0:
            print('\r%05d/%05d' % (i, limit), end='', flush=True)

        parts = line[6:-5].split('"')
        record = {}
        for i in range(0, len(parts), 2):
            k = parts[i].replace('=', '').strip()
            v = parts[i+1].strip()
            record[k] = v
        rows.append(record)

        if len(rows) > limit:
            break

    with open(json_file, 'w') as fout:
        json.dump(rows, fout)

    return rows

rows = download_stackexchange()
```

## 讨论

Stack Exchange 数据集是一个提供问题/答案对的很好的来源，并附带一个很好的可重用许可证。只要您进行归属，您可以以几乎任何方式使用它。将压缩的 XML 转换为更易消耗的 JSON 格式是一个很好的预处理步骤。

# 6.2 使用 Pandas 探索数据

## 问题

如何快速探索大型数据集，以确保其中包含您期望的内容？

## 解决方案

使用 Python 的 Pandas。

Pandas 是 Python 中用于数据处理的强大框架。在某些方面，它类似于电子表格；数据存储在行和列中，我们可以快速过滤、转换和聚合记录。让我们首先将我们的 Python 字典行转换为`DataFrame`。Pandas 会尝试“猜测”一些列的类型。我们将强制将我们关心的列转换为正确的格式：

```py
df = pd.DataFrame.from_records(rows)
df = df.set_index('Id', drop=False)
df['Title'] = df['Title'].fillna('').astype('str')
df['Tags'] = df['Tags'].fillna('').astype('str')
df['Body'] = df['Body'].fillna('').astype('str')
df['Id'] = df['Id'].astype('int')
df['PostTypeId'] = df['PostTypeId'].astype('int')
```

通过`df.head`，我们现在可以看到我们数据库中发生了什么。

我们也可以使用 Pandas 快速查看我们数据中的热门问题：

```py
list(df[df['ViewCount'] > 2500000]['Title'])
```

```py
['How to horizontally center a &lt;div&gt; in another &lt;div&gt;?',
 'What is the best comment in source code you have ever encountered?',
 'How do I generate random integers within a specific range in Java?',
 'How to redirect to another webpage in JavaScript/jQuery?',
 'How can I get query string values in JavaScript?',
 'How to check whether a checkbox is checked in jQuery?',
 'How do I undo the last commit(s) in Git?',
 'Iterate through a HashMap',
 'Get selected value in dropdown list using JavaScript?',
 'How do I declare and initialize an array in Java?']
```

正如你所期望的那样，最受欢迎的问题是关于经常使用的语言的一般问题。

## 讨论

Pandas 是一个很好的工具，适用于许多类型的数据分析，无论您只是想简单查看数据还是想进行深入分析。尽管很容易尝试利用 Pandas 来完成许多任务，但不幸的是，Pandas 的接口并不规范，对于复杂操作，性能可能会明显不如使用真正的数据库。在 Pandas 中进行查找比使用 Python 字典要昂贵得多，所以要小心！

# 6.3 使用 Keras 对文本进行特征化

## 问题

如何快速从文本创建特征向量？

## 解决方案

使用 Keras 的`Tokenizer`类。

在我们可以将文本输入模型之前，我们需要将其转换为特征向量。一个常见的方法是为文本中的前*N*个单词分配一个整数，然后用其整数替换每个单词。Keras 使这变得非常简单：

```py
from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 50000

tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(df['Body'] + ' ' + df['Title'])
```

现在让我们对整个数据集的标题和正文进行标记化：

```py
df['title_tokens'] = tokenizer.texts_to_sequences(df['Title'])
df['body_tokens'] = tokenizer.texts_to_sequences(df['Body'])
```

## 讨论

通过使用标记器将文本转换为一系列数字是使文本可被神经网络消化的经典方法之一。在前一章中，我们是基于每个字符进行文本转换的。基于字符的模型以单个字符作为输入（无需标记器）。权衡在于训练模型所需的时间：因为您强制模型学习如何对单词进行标记化和词干化，所以您需要更多的训练数据和更多的时间。

基于每个单词处理文本的一个缺点是，文本中出现的不同单词的数量没有实际上限，特别是如果我们必须处理拼写错误和错误。在这个示例中，我们只关注出现在前 50,000 位的单词，这是解决这个问题的一种方法。

# 6.4 构建问题/答案模型

## 问题

如何计算问题的嵌入？

## 解决方案

训练一个模型来预测 Stack Exchange 数据集中的问题和答案是否匹配。

每当我们构建一个模型时，我们应该问自己的第一个问题是：“我们的目标是什么？”也就是说，模型将尝试对什么进行分类？

理想情况下，我们会有一个“与此类似的问题列表”，我们可以用来训练我们的模型。不幸的是，获取这样的数据集将非常昂贵！相反，我们将依赖于一个替代目标：让我们看看是否可以训练我们的模型，即给定一个问题，区分匹配答案和来自随机问题的答案。这将迫使模型学习标题和正文的良好表示。

我们首先通过定义我们的输入来启动我们的模型。在这种情况下，我们有两个输入，标题（问题）和正文（答案）：

```py
title = layers.Input(shape=(None,), dtype='int32', name='title')
body = layers.Input(shape=(None,), dtype='int32', name='body')
```

两者长度不同，因此我们必须对它们进行填充。每个字段的数据将是一个整数列表，每个整数对应标题或正文中的一个单词。

现在我们想要定义一组共享的层，两个输入都将通过这些层。我们首先要为输入构建一个嵌入，然后屏蔽无效值，并将所有单词的值相加：

```py
 embedding = layers.Embedding(
        mask_zero=True,
        input_dim=vocab_size,
        output_dim=embedding_size
    )

mask = layers.Masking(mask_value=0)
def _combine_sum(v):
    return K.sum(v, axis=2)

sum_layer = layers.Lambda(_combine_sum)
```

在这里，我们指定了 `vocab_size`（我们的词汇表中有多少单词）和 `embedding_size`（每个单词的嵌入应该有多宽；例如，GoogleNews 的向量是 300 维）。

现在让我们将这些层应用于我们的单词输入：

```py
title_sum = sum_layer(mask(embedding(title)))
body_sum = sum_layer(mask(embedding(body)))
```

现在我们有了标题和正文的单个向量，我们可以通过余弦距离将它们相互比较，就像我们在 Recipe 4.2 中所做的那样。在 Keras 中，这通过 `dot` 层来表示：

```py
sim = layers.dot([title_sum, word_sum], normalize=True, axes=1)
```

最后，我们可以定义我们的模型。它接受标题和正文作为输入，并输出两者之间的相似度：

```py
sim_model = models.Model(inputs=[title,body], outputs=[sim])
sim_model.compile(loss='mse', optimizer='rmsprop')
```

## 讨论

我们在这里构建的模型学习匹配问题和答案，但实际上我们给它的唯一自由是改变单词的嵌入，使标题和正文的嵌入之和相匹配。这应该为我们提供问题的嵌入，使得相似的问题具有相似的嵌入，因为相似的问题将有相似的答案。

我们的训练模型编译时使用了两个参数，告诉 Keras 如何改进模型：

损失函数

这告诉系统一个给定答案有多“错误”。例如，如果我们告诉网络 `title_a` 和 `body_a` 应该输出 1.0，但网络预测为 0.8，那么这是多么糟糕的错误？当我们有多个输出时，这将变得更加复杂，但我们稍后会涵盖这一点。对于这个模型，我们将使用 *均方误差*。对于前面的例子，这意味着我们将通过 (1.0–0.8) ** 2，或 0.04 来惩罚模型。这种损失将通过模型传播回去，并在模型看到一个示例时改进每次的嵌入。

优化器

有许多方法可以利用损失来改进我们的模型。这些被称为*优化策略*或*优化器*。幸运的是，Keras 内置了许多可靠的优化器，所以我们不必太担心这个问题：我们只需要选择一个合适的。在这种情况下，我们使用`rmsprop`优化器，这个优化器在各种问题上表现非常好。

# 6.5 使用 Pandas 训练模型

## 问题

如何在 Pandas 中包含的数据上训练模型？

## 解决方案

构建一个数据生成器，利用 Pandas 的过滤和采样特性。

与前一个配方一样，我们将训练我们的模型来区分问题标题和正确答案（正文）与另一个随机问题的答案。我们可以将其写成一个迭代我们数据集的生成器。它将为正确的问题标题和正文输出 1，为随机标题和正文输出 0：

```py
def data_generator(batch_size, negative_samples=1):
    questions = df[df['PostTypeId'] == 1]
    all_q_ids = list(questions.index)

    batch_x_a = []
    batch_x_b = []
    batch_y = []

    def _add(x_a, x_b, y):
        batch_x_a.append(x_a[:MAX_DOC_LEN])
        batch_x_b.append(x_b[:MAX_DOC_LEN])
        batch_y.append(y)

    while True:
        questions = questions.sample(frac=1.0)

        for i, q in questions.iterrows():
            _add(q['title_tokens'], q['body_tokens'], 1)

            negative_q = random.sample(all_q_ids, negative_samples)
            for nq_id in negative_q:
                _add(q['title_tokens'],
                     df.at[nq_id, 'body_tokens'], 0)

            if len(batch_y) >= batch_size:
                yield ({
                    'title': pad_sequences(batch_x_a, maxlen=None),
                    'body': pad_sequences(batch_x_b, maxlen=None),
                }, np.asarray(batch_y))

                batch_x_a = []
                batch_x_b = []
                batch_y = []
```

这里唯一的复杂性是数据的分批处理。这并不是绝对必要的，但对性能非常重要。所有深度学习模型都被优化为一次处理数据块。要使用的最佳批量大小取决于您正在处理的问题。使用更大的批量意味着您的模型在每次更新时看到更多数据，因此可以更准确地更新其权重，但另一方面它不能经常更新。更大的批量大小也需要更多内存。最好从小批量开始，然后将批量大小加倍，直到结果不再改善。

现在让我们训练模型：

```py
sim_model.fit_generator(
    data_generator(batch_size=128),
    epochs=10,
    steps_per_epoch=1000
)
```

我们将进行 10,000 步的训练，分为 10 个包含 1,000 步的时期。每一步将处理 128 个文档，因此我们的网络最终将看到 1.28M 个训练示例。如果您有 GPU，您会惊讶地发现这个过程运行得多么快！

# 6.6 检查相似性

## 问题

您想使用 Keras 通过另一个网络的权重来预测值。

## 解决方案

构建一个使用原始网络不同输入和输出层的第二个模型，但共享其他一些层。

我们的`sim_model`已经训练过了，并且作为其中的一部分学会了如何从标题到`title_sum`，这正是我们想要的。只做这件事的模型是：

```py
embedding_model = models.Model(inputs=[title], outputs=[title_sum])
```

现在我们可以使用“嵌入”模型为数据集中的每个问题计算一个表示。让我们将这个封装成一个易于重复使用的类：

```py
questions = df[df['PostTypeId'] == 1]['Title'].reset_index(drop=True)
question_tokens = pad_sequences(tokenizer.texts_to_sequences(questions))

class EmbeddingWrapper(object):
    def __init__(self, model):
        self._questions = questions
        self._idx_to_question = {i:s for (i, s) in enumerate(questions)}
        self._weights = model.predict({'title': question_tokens},
                                      verbose=1, batch_size=1024)
        self._model = model
        self._norm = np.sqrt(np.sum(self._weights * self._weights
                                    + 1e-5, axis=1))

    def nearest(self, question, n=10):
        tokens = tokenizer.texts_to_sequences([sentence])
        q_embedding = self._model.predict(np.asarray(tokens))[0]
        q_norm= np.sqrt(np.dot(q_embedding, q_embedding))
        dist = np.dot(self._weights, q_embedding) / (q_norm * self._norm)

        top_idx = np.argsort(dist)[-n:]
        return pd.DataFrame.from_records([
            {'question': self._r[i], ‘similarity': float(dist[i])}
            for i in top_idx
        ])
```

现在我们可以使用它了：

```py
lookup = EmbeddingWrapper(model=sum_embedding_trained)
lookup.nearest('Python Postgres object relational model')
```

这产生了以下结果：

| 相似性 | 问题 |
| --- | --- |
| 0.892392 | 使用 django 和 SQLAlchemy 但后端... |
| 0.893417 | 自动生成/更新表的 Python ORM... |
| 0.893883 | 在 SqlA 中进行动态表创建和 ORM 映射... |
| 0.896096 | 使用 count、group_by 和 order_by 的 SQLAlchemy... |
| 0.897706 | 使用 ORM 扫描大表？ |
| 0.902693 | 使用 SQLAlchemy 高效更新数据库... |
| 0.911446 | 有哪些好的 Python ORM 解决方案？ |
| 0.922449 | Python ORM |
| 0.924316 | 从 r...构建类的 Python 库... |
| 0.930865 | 允许创建表和构建的 Python ORM... |

在非常短的训练时间内，我们的网络设法弄清楚了“SQL”、“查询”和“插入”都与 Postgres 有关！

## 讨论

在这个配方中，我们看到了如何使用网络的一部分来预测我们想要的值，即使整个网络是为了预测其他东西而训练的。Keras 的功能 API 提供了层之间、它们如何连接以及哪种输入和输出层组成模型的良好分离。

正如我们将在本书后面看到的，这给了我们很大的灵活性。我们可以取一个预训练的网络，并使用其中一个中间层作为输出层，或者我们可以取其中一个中间层并添加一些新的层（参见第九章）。我们甚至可以反向运行网络（参见第十二章）。
