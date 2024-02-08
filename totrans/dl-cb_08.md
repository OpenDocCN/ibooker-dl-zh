# 第8章。序列到序列映射

在本章中，我们将研究使用序列到序列网络来学习文本片段之间的转换。这是一种相对较新的技术，具有诱人的可能性。谷歌声称已经通过这种技术大大改进了其Google翻译产品；此外，它已经开源了一个版本，可以纯粹基于平行文本学习语言翻译。

我们不会一开始就走得那么远。相反，我们将从一个简单的模型开始，学习英语中复数形式的规则。之后，我们将从古腾堡计划的19世纪小说中提取对话，并在其中训练一个聊天机器人。对于这个最后的项目，我们将不得不放弃在笔记本中运行Keras的安全性，并使用谷歌的开源seq2seq工具包。

以下笔记本包含本章相关的代码：

```py
08.1 Sequence to sequence mapping
08.2 Import Gutenberg
08.3 Subword tokenizing
```

# 8.1 训练一个简单的序列到序列模型

## 问题

你如何训练一个模型来逆向工程一个转换？

## 解决方案

使用序列到序列的映射器。

在[第5章](ch05.html#text_generation)中，我们看到了如何使用循环网络来“学习”序列的规则。模型学习如何最好地表示一个序列，以便能够预测下一个元素是什么。序列到序列映射建立在此基础上，但现在模型学习根据第一个序列预测不同的序列。

我们可以使用这个来学习各种转换。让我们考虑将英语中的单数名词转换为复数名词。乍一看，这似乎只是在单词后添加一个*s*，但当你仔细观察时，你会发现规则实际上相当复杂。

这个模型与我们在[第5章](ch05.html#text_generation)中使用的模型非常相似，但现在不仅输入是一个序列，输出也是一个序列。这是通过使用`RepeatVector`层实现的，它允许我们从输入映射到输出向量：

```py
def create_seq2seq(num_nodes, num_layers):
    question = Input(shape=(max_question_len, len(chars),
                     name='question'))
    repeat = RepeatVector(max_expected_len)(question)
    prev = input
    for _ in range(num_layers)::
        lstm = LSTM(num_nodes, return_sequences=True,
                    name='lstm_layer_%d' % (i + 1))(prev)
        prev = lstm
    dense = TimeDistributed(Dense(num_chars, name='dense',
                            activation='softmax'))(prev)
    model = Model(inputs=[input], outputs=[dense])
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
```

数据的预处理与以前大致相同。我们从文件*data/plurals.txt*中读取数据并将其向量化。一个要考虑的技巧是是否要颠倒输入中的字符串。如果输入被颠倒，那么生成输出就像展开处理，这可能更容易。

模型需要相当长的时间才能达到接近99%的精度。然而，大部分时间都花在学习如何重现单词的单数和复数形式共享的前缀上。事实上，当我们检查模型在达到超过99%精度时的性能时，我们会发现大部分错误仍然在这个领域。

## 讨论

序列到序列模型是强大的工具，只要有足够的资源，就可以学习几乎任何转换。学习从英语的单数到复数的规则只是一个简单的例子。这些模型是领先科技公司提供的最先进的机器翻译解决方案的基本元素。

像这个配方中的简单模型可以学习如何在罗马数字中添加数字，或者学习在书面英语和音标英语之间进行翻译，这在构建文本到语音系统时是一个有用的第一步。

在接下来的几个配方中，我们将看到如何使用这种技术基于从19世纪小说中提取的对话来训练一个聊天机器人。

# 8.2 从文本中提取对话

## 问题

你如何获取大量的对话语料库？

## 解决方案

解析一些从古腾堡计划中提供的文本，并提取所有的对话。

让我们从古腾堡计划中下载一套书籍。我们可以下载所有的书籍，但在这里我们将专注于那些作者出生在1835年之后的作品。这样可以使对话保持现代。*data/books.json*文档包含相关的参考资料：

```py
with open('data/gutenberg_index.json') as fin:
    authors = json.load(fin)
recent = [x for x in authors
          if 'birthdate' in x and x['birthdate'] > 1830]
[(x['name'], x['birthdate'], x['english_books']) for x in recent[:5]]
```

```py
[('Twain, Mark', 1835, 210),
 ('Ebers, Georg', 1837, 164),
 ('Parker, Gilbert', 1862, 135),
 ('Fenn, George Manville', 1831, 128),
 ('Jacobs, W. W. (William Wymark)', 1863, 112)]
```

这些书大多以ASCII格式一致排版。段落之间用双换行符分隔，对话几乎总是使用双引号。少量书籍也使用单引号，但我们将忽略这些，因为单引号也出现在文本的其他地方。我们假设对话会持续下去，只要引号外的文本长度不超过100个字符（例如，“嗨，”他说，“你好吗？”）：

```py
def extract_conversations(text, quote='"'):
    paragraphs = PARAGRAPH_SPLIT_RE.split(text.strip())
    conversations = [['']]
    for paragraph in paragraphs:
        chunks = paragraph.replace('\n', ' ').split(quote)
        for i in range((len(chunks) + 1) // 2):
            if (len(chunks[i * 2]) > 100
                or len(chunks) == 1) and conversations[-1] != ['']:
                if conversations[-1][-1] == '':
                    del conversations[-1][-1]
                conversations.append([''])
            if i * 2 + 1 < len(chunks):
                chunk = chunks[i * 2 + 1]
                if chunk:
                    if conversations[-1][-1]:
                        if chunk[0] >= 'A' and chunk[0] <= 'Z':
                            if conversations[-1][-1].endswith(','):
                                conversations[-1][-1] = \
                                     conversations[-1][-1][:-1]
                            conversations[-1][-1] += '.'
                        conversations[-1][-1] += ' '
                    conversations[-1][-1] += chunk
        if conversations[-1][-1]:
            conversations[-1].append('')

    return [x for x in conversations if len(x) > 1]
```

处理前1000位作者的数据可以得到一组良好的对话数据：

```py
for author in recent[:1000]:
    for book in author['books']:
        txt = strip_headers(load_etext(int(book[0]))).strip()
        conversations += extract_conversations(txt)
```

这需要一些时间，所以最好将结果保存到文件中：

```py
with open('gutenberg.txt', 'w') as fout:
    for conv in conversations:
        fout.write('\n'.join(conv) + '\n\n')
```

## 讨论

正如我们在[第5章](ch05.html#text_generation)中看到的，Project Gutenberg是一个很好的来源，可以自由使用文本，只要我们不介意它们有点过时，因为它们必须已经过期。

该项目是在关于布局和插图的担忧之前开始的，因此所有文档都是以纯ASCII格式生成的。虽然这不是实际书籍的最佳格式，但它使解析相对容易。段落之间用双换行符分隔，没有智能引号或任何标记。

# 8.3 处理开放词汇

## 问题

如何仅使用固定数量的标记完全标记化文本？

## 解决方案

使用子词单元进行标记化。

在前一章中，我们只是跳过了我们前50,000个单词的词汇表中找不到的单词。通过子词单元标记化，我们将不经常出现的单词分解为更小的子单元，直到所有单词和子单元适合我们的固定大小的词汇表。

例如，如果我们有*working*和*worked*这两个词，我们可以将它们分解为*work-*、*-ed*和*-ing*。这三个标记很可能会与我们词汇表中的其他标记重叠，因此这可能会减少我们整体词汇表的大小。所使用的算法很简单。我们将所有标记分解为它们的各个字母。此时，每个字母都是一个子词标记，而且我们可能少于最大数量的标记。然后找出在我们的标记中最常出现的一对子词标记。在英语中，这通常是（*t*，*h*）。然后将这些子词标记连接起来。这通常会增加一个子词标记的数量，除非我们的一对中的一个项目现在已经用完。我们继续这样做，直到我们有所需数量的子词和单词标记。

尽管代码并不复杂，但使用这个算法的[开源版本](https://github.com/rsennrich/subword-nmt)是有意义的。标记化是一个三步过程。

第一步是对我们的语料库进行标记化。默认的标记器只是分割文本，这意味着它保留所有标点符号，通常附加在前一个单词上。我们想要更高级的东西。我们希望除了问号之外剥离所有标点符号。我们还将所有内容转换为小写，并用空格替换下划线：

```py
RE_TOKEN = re.compile('(\w+|\?)', re.UNICODE)
token_counter = Counter()
with open('gutenberg.txt') as fin:
    for line in fin:
        line = line.lower().replace('_', ' ')
        token_counter.update(RE_TOKEN.findall(line))
with open('gutenberg.tok', 'w') as fout:
    for token, count in token_counter.items():
        fout.write('%s\t%d\n' % (token, count))
```

现在我们可以学习子词标记：

```py
./learn_bpe.py -s 25000 < gutenberg.tok > gutenberg.bpe
```

然后我们可以将它们应用于任何文本：

```py
./apply_bpe.py -c gutenberg.bpe < some_text.txt > some_text.bpe.txt
```

生成的*some_text.bpe.txt*看起来像我们的原始语料库，只是罕见的标记被分解并以*@@*结尾表示继续。

## 讨论

将文本标记化为单词是减少文档大小的有效方法。正如我们在[第7章](ch07.html#suggest_emojis)中看到的，它还允许我们通过加载预训练的单词嵌入来启动我们的学习。然而，存在一个缺点：较大的文本包含太多不同的单词，我们无法希望覆盖所有单词。一个解决方案是跳过不在我们词汇表中的单词，或用固定的`UNKNOWN`标记替换它们。这对于情感分析来说效果还不错，但对于我们想要生成输出文本的任务来说，这是相当不令人满意的。在这种情况下，子词单元标记化是一个很好的解决方案。

另一个选择是最近开始受到关注的选项，即训练一个字符级模型来为词汇表中不存在的单词生成嵌入。

# 8.4 训练一个seq2seq聊天机器人

## 问题

您想要训练一个深度学习模型来复制对话语料库的特征。

## 解决方案

使用Google的seq2seq框架。

[配方8.1](#training-a-simple-s-t-s-model)中的模型能够学习序列之间的关系，甚至是相当复杂的关系。然而，序列到序列模型很难调整性能。在2017年初，Google发布了seq2seq，这是一个专门为这种应用程序开发的库，可以直接在TensorFlow上运行。它让我们专注于模型的超参数，而不是代码的细节。

seq2seq框架希望其输入被分成训练、评估和开发集。每个集合应包含一个源文件和一个目标文件，其中匹配的行定义了模型的输入和输出。在我们的情况下，源应包含对话的提示，目标应包含答案。然后，模型将尝试学习如何从提示转换为答案，有效地学习如何进行对话。

第一步是将我们的对话分成（源，目标）对。对于对话中的每一对连续的句子，我们提取第一句和最后一句作为源和目标：

```py
RE_TOKEN = re.compile('(\w+|\?)', re.UNICODE)
def tokenize(st):
    st = st.lower().replace('_', ' ')
    return ' '.join(RE_TOKEN.findall(st))

pairs = []
prev = None
with open('data/gutenberg.txt') as fin:
    for line in fin:
        line = line.strip()
        if line:
            sentences = nltk.sent_tokenize(line)
            if prev:
                pairs.append((prev, tokenize(sentences[0])))
            prev = tokenize(sentences[-1])
        else:
            prev = None
```

现在让我们洗牌我们的对，并将它们分成我们的三个组，`dev`和`test`集合各代表我们数据的5%：

```py
random.shuffle(pairs)
ss = len(pairs) // 20

data = {'dev': pairs[:ss],
        'test': pairs[ss:ss * 2],
        'train': pairs[ss * 2:]}
```

接下来，我们需要解压这些对，并将它们放入正确的目录结构中：

```py
for tag, pairs2 in data.items():
    path = 'seq2seq/%s' % tag
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(path + '/sources.txt', 'wt') as sources:
        with open(path + '/targets.txt', 'wt') as targets:
            for source, target in pairs2:
                sources.write(source + '\n')
                targets.write(target + '\n')
```

是时候训练网络了。克隆seq2seq存储库并安装依赖项。您可能希望在一个单独的`virtualenv`中执行此操作：

```py
git clone https://github.com/google/seq2seq.git
cd seq2seq
pip install -e .
```

现在让我们设置一个指向我们已经准备好的数据的环境变量：

```py
Export SEQ2SEQROOT=/path/to/data/seq2seq
```

seq2seq库包含一些配置文件，我们可以在*example_configs*目录中混合匹配。在这种情况下，我们想要训练一个大型模型，其中包括：

```py
python -m bin.train \                                                                                                               --config_paths="                                                                                                                                                ./example_configs/nmt_large.yml,
      ./example_configs/train_seq2seq.yml" \
  --model_params "
      vocab_source: $SEQ2SEQROOT/gutenberg.tok
      vocab_target: $SEQ2SEQROOT/gutenberg.tok" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $SEQ2SEQROOT/train/sources.txt
      target_files:
        - $SEQ2SEQROOT/train/targets.txt" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $SEQ2SEQROOT/dev/sources.txt
       target_files:
        - $SEQ2SEQROOT/dev/targets.txt" \
  --batch_size 1024  --eval_every_n_steps 5000 \
  --train_steps 5000000 \
  --output_dir $SEQ2SEQROOT/model_large
```

不幸的是，即使在具有强大GPU的系统上，也需要很多天才能获得一些像样的结果。然而，笔记本中的*zoo*文件夹包含一个预训练模型，如果您等不及的话。

该库没有提供一种交互式运行模型的方法。在[第16章](ch16.html#productionizing)中，我们将探讨如何做到这一点，但现在我们可以通过将我们的测试问题添加到一个文件（例如，*/tmp/test_questions.txt*）并运行以下命令来快速获得一些结果：

```py
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $SEQ2SEQROOT/model_large \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - '/tmp/test_questions.txt'"
```

一个简单的对话可以运行：

```py
> hi
hi
> what is your name ?
sam barker
> how do you feel ?
Fine
> good night
good night
```

对于更复杂的句子，有时会有点碰运气。

## 讨论

seq2seq模型的主要用途似乎是自动翻译，尽管它也对图像字幕和文本摘要有效。文档中包含了一个[教程](https://google.github.io/seq2seq/nmt/)，介绍如何训练一个在几周或几个月内学会不错的英语-德语翻译的模型，具体取决于您的硬件。Google声称，将序列到序列模型作为其机器翻译工作的核心部分，显著提高了质量。

一个有趣的思考序列到序列映射的方式是将其视为嵌入过程。对于翻译，源句和目标句都被投影到一个多维空间中，模型学习一个投影，使得意思相同的句子最终在该空间中的同一点附近。这导致了“零翻译”的有趣可能性；如果一个模型学会了在芬兰语和英语之间进行翻译，然后稍后在英语和希腊语之间进行翻译，并且它使用相同的语义空间，它也可以直接在芬兰语和希腊语之间进行翻译。这就打开了“思维向量”的可能性，这是相对复杂的想法的嵌入，具有与我们在[第3章](ch03.html#word_embeddings)中看到的“词向量”类似的属性。
