# 第七章：建议表情符号

在本章中，我们将构建一个模型，根据一小段文本建议表情符号。我们将首先基于一组带有各种情感标签的推文开发一个简单的情感分类器，如快乐、爱、惊讶等。我们首先尝试一个贝叶斯分类器，以了解基线性能，并查看这个分类器可以学到什么。然后我们将切换到卷积网络，并查看各种调整这个分类器的方法。

接下来我们将看看如何使用 Twitter API 收集推文，然后我们将应用配方 7.3 中的卷积模型，然后转向一个单词级模型。然后我们将构建并应用一个递归单词级网络，并比较这三种不同的模型。

最后，我们将把这三个模型组合成一个胜过任何一个的集成模型。

最终模型表现得非常不错，只需要整合到一个移动应用程序中！

本章的代码可以在这些笔记本中找到：

```py
07.1 Text Classification
07.2 Emoji Suggestions
07.3 Tweet Embeddings
```

# 7.1 构建一个简单的情感分类器

## 问题

如何确定文本中表达的情感？

## 解决方案

找到一个由标记了情感的句子组成的数据集，并对其运行一个简单的分类器。

在尝试复杂的东西之前，首先尝试在一个 readily 可用的数据集上尝试我们能想到的最简单的事情是一个好主意。在这种情况下，我们将尝试基于一个已发布数据集构建一个简单的情感分类器。在接下来的配方中，我们将尝试做一些更复杂的事情。

快速的谷歌搜索让我们找到了一个来自 CrowdFlower 的不错的数据集，其中包含推文和情感标签。由于情感标签在某种程度上类似于表情符号，这是一个很好的开始。让我们下载文件并看一眼：

```py
import pandas as pd
from keras.utils.data_utils import get_file
import nb_utils

emotion_csv = get_file('text_emotion.csv',
                       'https://www.crowdflower.com/wp-content/'
                       'uploads/2016/07/text_emotion.csv')
emotion_df = pd.read_csv(emotion_csv)

emotion_df.head()
```

这导致：

| 推文 ID | 情感 | 作者 | 内容 |
| --- | --- | --- | --- |
| 0 | 1956967341 | 空 | xoshayzers @tiffanylue 我知道我在听坏习惯... |
| 1 | 1956967666 | 伤心 | wannamama 躺在床上头疼，等待... |
| 2 | 1956967696 | 伤心 | coolfunky 葬礼仪式...阴郁的星期五... |
| 3 | 1956967789 | 热情 | czareaquino 想很快和朋友们一起出去！ |
| 4 | 1956968416 | 中性 | xkilljoyx @dannycastillo 我们想和某人交易... |

我们还可以检查各种情绪发生的频率：

```py
emotion_df['sentiment'].value_counts()
```

```py
neutral       8638
worry         8459
happiness     5209
sadness       5165
love          3842
surprise      2187
```

一些最简单的模型通常会产生令人惊讶的好结果，来自朴素贝叶斯家族。我们将从使用`sklearn`提供的方法对数据进行编码。`TfidfVectorizer`根据其逆文档频率为单词分配权重；经常出现的单词获得较低的权重，因为它们往往不太具有信息性。`LabelEncoder`为它看到的不同标签分配唯一的整数：

```py
tfidf_vec = TfidfVectorizer(max_features=VOCAB_SIZE)
label_encoder = LabelEncoder()
linear_x = tfidf_vec.fit_transform(emotion_df['content'])
linear_y = label_encoder.fit_transform(emotion_df['sentiment'])
```

有了这些数据，我们现在可以构建贝叶斯模型并评估它：

```py
bayes = MultinomialNB()
bayes.fit(linear_x, linear_y)
pred = bayes.predict(linear_x)
precision_score(pred, linear_y, average='micro')
```

```py
0.28022727272727271
```

我们有 28%的正确率。如果我们总是预测最可能的类别，我们会得到略高于 20%，所以我们有了一个良好的开端。还有一些其他简单的分类器可以尝试，可能会做得更好，但速度较慢：

```py
classifiers = {'sgd': SGDClassifier(loss='hinge'),
               'svm': SVC(),
               'random_forrest': RandomForestClassifier()}

for lbl, clf in classifiers.items():
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(lbl, precision_score(predictions, y_test, average='micro'))
```

```py
random_forrest 0.283939393939
svm 0.218636363636
sgd 0.325454545455
```

## 讨论

尝试“可能起作用的最简单的事情”有助于我们快速入门，并让我们了解数据是否具有足够的信号来完成我们想要做的工作。

贝叶斯分类器在早期的电子邮件垃圾邮件对抗中表现非常有效。然而，它们假设每个因素的贡献是彼此独立的——因此在这种情况下，推文中的每个单词对预测标签都有一定影响，独立于其他单词——这显然并非总是如此。一个简单的例子是，在句子中插入单词“not”可以否定表达的情感。尽管如此，该模型很容易构建，并且可以很快为我们带来结果，而且结果是可以理解的。一般来说，如果贝叶斯模型在您的数据上没有产生好的结果，使用更复杂的东西可能不会有太大帮助。

###### 注意

贝叶斯模型通常比我们天真地期望的要好。关于这一点已经有一些有趣的研究。在机器学习之前，它们帮助破译了恩尼格玛密码，也帮助驱动了第一个电子邮件垃圾邮件检测器。

# 7.2 检查一个简单分类器

## 问题

如何查看一个简单分类器学到了什么？

## 解决方案

查看使分类器输出结果的贡献因素。

使用贝叶斯方法的一个优点是我们可以理解模型。正如我们在前面的配方中讨论的那样，贝叶斯模型假设每个单词的贡献与其他单词无关，因此为了了解我们的模型学到了什么，我们可以询问模型对个别单词的看法。

现在记住，模型期望一系列文档，每个文档都编码为一个向量，其长度等于词汇表的大小，每个元素编码为该文档中对应单词相对频率与所有文档的比率。因此，每个只包含一个单词的文档集合将是一个对角线上有 1 的方阵；第 n 个文档将对词汇表中的所有单词都有零，除了单词 n。现在我们可以为每个单词预测标签的可能性：

```py
d = eye(len(tfidf_vec.vocabulary_))
word_pred = bayes.predict_proba(d)
```

然后我们可以查看所有预测，并找到每个类别的单词分数。我们将这些存储在一个`Counter`对象中，以便我们可以轻松访问贡献最大的单词：

```py
by_cls = defaultdict(Counter)
for word_idx, pred in enumerate(word_pred):
    for class_idx, score in enumerate(pred):
        cls = label_encoder.classes_[class_idx]
        by_cls[cls][inverse_vocab[word_idx]] = score
```

让我们打印结果：

```py
for k in by_cls:
    words = [x[0] for x in by_cls[k].most_common(5)]
    print(k, ':', ' '.join(words))
```

```py
happiness : excited woohoo excellent yay wars
hate : hate hates suck fucking zomberellamcfox
boredom : squeaking ouuut cleanin sooooooo candyland3
enthusiasm : lena_distractia foolproofdiva attending krisswouldhowse tatt
fun : xbox bamboozle sanctuary oldies toodaayy
love : love mothers mommies moms loved
surprise : surprise wow surprised wtf surprisingly
empty : makinitrite conversating less_than_3 shakeyourjunk kimbermuffin
anger : confuzzled fridaaaayyyyy aaaaaaaaaaa transtelecom filthy
worry : worried poor throat hurts sick
relief : finally relax mastered relief inspiration
sadness : sad sadly cry cried miss
neutral : www painting souljaboytellem link frenchieb
```

## 讨论

在深入研究更复杂的内容之前检查一个简单模型学到了什么是一个有用的练习。尽管深度学习模型非常强大，但事实是很难真正了解它们在做什么。我们可以大致了解它们的工作原理，但要真正理解训练结果中的数百万个权重几乎是不可能的。

我们的贝叶斯模型的结果符合我们的预期。单词“sad”是“sadness”类的指示，“wow”是惊讶的指示。令人感动的是，单词“mothers”是爱的强烈指示。

我们看到了一堆奇怪的单词，比如“kimbermuffin”和“makinitrite”。检查后发现这些是 Twitter 用户名。 “foolproofdiva”只是一个非常热情的人。根据目标，我们可能会考虑将这些过滤掉。

# 7.3 使用卷积网络进行情感分析

## 问题

您想尝试使用深度网络来确定文本中表达的情感。

## 解决方案

使用卷积网络。

CNNs 更常用于图像识别（参见第九章），但它们在某些文本分类任务中也表现良好。其思想是在文本上滑动一个窗口，从而将一系列项目转换为（更短的）特征序列。在这种情况下，项目将是字符。每一步都使用相同的权重，因此我们不必多次学习相同的内容——单词“cat”在推文中的任何位置都表示“cat”：

```py
char_input = Input(shape=(max_sequence_len, num_chars), name='input')

conv_1x = Conv1D(128, 6, activation='relu', padding='valid')(char_input)
max_pool_1x = MaxPooling1D(6)(conv_1x)
conv_2x = Conv1D(256, 6, activation='relu', padding='valid')(max_pool_1x)
max_pool_2x = MaxPooling1D(6)(conv_2x)

flatten = Flatten()(max_pool_2x)
dense = Dense(128, activation='relu')(flatten)
preds = Dense(num_labels, activation='softmax')(dense)

model = Model(char_input, preds)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
```

为了使模型运行，我们首先必须对数据进行向量化。我们将使用在前面的配方中看到的相同的一热编码，将每个字符编码为一个填满所有零的向量，除了第 n 个条目，其中 n 对应于我们要编码的字符：

```py
chars = list(sorted(set(chain(*emotion_df['content']))))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
max_sequence_len = max(len(x) for x in emotion_df['content'])

char_vectors = []
for txt in emotion_df['content']:
    vec = np.zeros((max_sequence_len, len(char_to_idx)))
    vec[np.arange(len(txt)), [char_to_idx[ch] for ch in txt]] = 1
    char_vectors.append(vec)
char_vectors = np.asarray(char_vectors)
char_vectors = pad_sequences(char_vectors)
labels = label_encoder.transform(emotion_df['sentiment'])
```

让我们将数据分成训练集和测试集：

```py
def split(lst):
    training_count = int(0.9 * len(char_vectors))
    return lst[:training_count], lst[training_count:]

training_char_vectors, test_char_vectors = split(char_vectors)
training_labels, test_labels = split(labels)
```

现在我们可以训练模型并评估它：

```py
char_cnn_model.fit(training_char_vectors, training_labels,
                   epochs=20, batch_size=1024)
char_cnn_model.evaluate(test_char_vectors, test_labels)
```

经过 20 个时代，训练准确率达到 0.39，但测试准确率只有 0.31。这种差异可以通过过拟合来解释；模型不仅学习了数据的一般方面，这些方面也适用于测试集，而且开始记忆部分训练数据。这类似于学生学习哪些答案与哪些问题匹配，而不理解为什么。

## 讨论

卷积网络在我们希望网络学习独立于发生位置的情况下效果很好。对于图像识别，我们不希望网络为每个像素单独学习；我们希望它学会独立于图像中发生位置的特征。

同样，对于文本，我们希望模型学会，如果推文中出现“爱”这个词，那么“爱”将是一个好的标签。我们不希望模型为每个位置单独学习这一点。CNN 通过在文本上运行一个滑动窗口来实现这一点。在这种情况下，我们使用大小为 6 的窗口，因此我们每次取 6 个字符；对于包含 125 个字符的推文，我们会应用这个过程 120 次。

关键的是，这 120 个神经元中的每一个都使用相同的权重，因此它们都学习相同的东西。在卷积之后，我们应用一个`max_pooling`层。这一层将取六个神经元的组并输出它们激活的最大值。我们可以将其视为将任何神经元中最强的理论传递给下一层。它还将大小减小了六分之一。

在我们的模型中，我们有两个卷积/最大池化层，它将输入从 167×100 的大小更改为 3×256。我们可以将这些看作是增加抽象级别的步骤。在输入级别，我们只知道在 167 个位置中的每一个位置上出现了 100 个不同字符中的哪一个。在最后一个卷积之后，我们有 3 个 256 个向量，它们分别编码了推文开头、中间和结尾发生的情况。

# 7.4 收集 Twitter 数据

## 问题

如何自动收集大量用于训练目的的 Twitter 数据？

## 解决方案

使用 Twitter API。

首先要做的是前往[*https://apps.twitter.com*](https://apps.twitter.com)注册一个新应用。点击“创建新应用”按钮并填写表格。我们不会代表用户做任何事情，所以可以将回调 URL 字段留空。

完成后，您应该有两个密钥和两个允许访问 API 的密钥。让我们将它们存储在相应的变量中：

```py
CONSUMER_KEY = '<your value>'
CONSUMER_SECRET = '<your value>'
ACCESS_TOKEN = '<your value>'
ACCESS_SECRET = '<your value>'
```

现在我们可以构建一个认证对象：

```py
auth=twitter.OAuth(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    token=ACCESS_TOKEN,
    token_secret=ACCESS_SECRET,
)
```

Twitter API 有两部分。REST API 使得可以调用各种函数来搜索推文、获取用户的状态，甚至发布到 Twitter。在这个示例中，我们将使用流 API。

如果你付费给 Twitter，你将获得一个包含所有推文的流。如果你不付费，你会得到所有推文的一个样本。这对我们的目的已经足够了：

```py
status_stream = twitter.TwitterStream(auth=auth).statuses
```

`stream`对象有一个迭代器`sample`，它将产生推文。让我们使用`itertools.islice`来查看其中一些：

```py
[x['text'] for x in itertools.islice(stream.sample(), 0, 5) if x.get('text')]
```

在这种情况下，我们只想要英文推文，并且至少包含一个表情符号：

```py
def english_has_emoji(tweet):
    if tweet.get('lang') != 'en':
        return False
    return any(ch for ch in tweet.get('text', '') if ch in emoji.UNICODE_EMOJI)
```

现在我们可以获取包含至少一个表情符号的一百条推文：

```py
tweets = list(itertools.islice(
    filter(english_has_emoji, status_stream.sample()), 0, 100))
```

我们每秒获得两到三条推文，这还不错，但要花一段时间才能拥有一个规模可观的训练集。我们只关心那些只有一种类型的表情符号的推文，我们只想保留该表情符号和文本：

```py
stripped = []
for tweet in tweets:
    text = tweet['text']
    emojis = {ch for ch in text if ch in emoji.UNICODE_EMOJI}
    if len(emojis) == 1:
        emoiji = emojis.pop()
        text = ''.join(ch for ch in text if ch != emoiji)
        stripped.append((text, emoiji))
```

## 讨论

Twitter 可以是一个非常有用的训练数据来源。每条推文都有大量与之相关的元数据，从发布推文的账户到图片和哈希标签。在本章中，我们只使用语言元信息，但这是一个值得探索的丰富领域。

# 7.5 一个简单的表情符号预测器

## 问题

如何预测最匹配一段文本的表情符号？

## 解决方案

重新利用来自 Recipe 7.3 的情感分类器。

如果在上一步中收集了大量推文，可以使用这些。如果没有，可以在*data/emojis.txt*中找到一个好的样本。让我们将这些读入 Pandas 的`DataFrame`。我们将过滤掉出现次数少于 1000 次的任何表情符号：

```py
all_tweets = pd.read_csv('data/emojis.txt',
        sep='\t', header=None, names=['text', 'emoji'])
tweets = all_tweets.groupby('emoji').filter(lambda c:len(c) > 1000)
tweets['emoji'].value_counts()
```

这个数据集太大了，无法以向量化形式保存在内存中，所以我们将使用生成器进行训练。Pandas 方便地提供了一个`sample`方法，允许我们使用以下`data_generator`：

```py
def data_generator(tweets, batch_size):
    while True:
        batch = tweets.sample(batch_size)
        X = np.zeros((batch_size, max_sequence_len, len(chars)))
        y = np.zeros((batch_size,))
        for row_idx, (_, row) in enumerate(batch.iterrows()):
            y[row_idx] = emoji_to_idx[row['emoji']]
            for ch_idx, ch in enumerate(row['text']):
                X[row_idx, ch_idx, char_to_idx[ch]] = 1
        yield X, y
```

我们现在可以在不进行修改的情况下从 Recipe 7.3 训练模型：

```py
train_tweets, test_tweets = train_test_split(tweets, test_size=0.1)
BATCH_SIZE = 512
char_cnn_model.fit_generator(
    data_generator(train_tweets, batch_size=BATCH_SIZE),
    epochs=20,
    steps_per_epoch=len(train_tweets) / BATCH_SIZE,
    verbose=2
)
```

模型训练到大约 40%的精度。即使考虑到顶部表情符号比底部表情符号更频繁出现，这听起来还是相当不错的。如果我们在评估集上运行模型，精度得分会从 40%下降到略高于 35%：

```py
char_cnn_model.evaluate_generator(
    data_generator(test_tweets, batch_size=BATCH_SIZE),
    steps=len(test_tweets) / BATCH_SIZE
)
```

```py
[3.0898117224375405, 0.35545459692028986]
```

## 讨论

在不对模型本身进行任何更改的情况下，我们能够为推文建议表情符号，而不是运行情感分类。这并不太令人惊讶；在某种程度上，表情符号是作者应用的情感标签。对于这两个任务性能大致相同可能不太出乎意料，因为我们有更多的标签，而且我们预计标签会更加嘈杂。

# 7.6 Dropout 和多窗口

## 问题

你如何提高网络的性能？

## 解决方案

增加可训练变量的数量，同时引入了一种使更大的网络难以过拟合的技术——dropout。

增加神经网络的表达能力的简单方法是使其更大，可以通过使单个层更大或向网络添加更多层来实现。具有更多变量的网络具有更高的学习能力，并且可以更好地泛化。然而，这并非是免费的；在某个时候，网络开始*过拟合*。(Recipe 1.3 更详细地描述了这个问题。)

让我们从扩展当前网络开始。在上一个配方中，我们为卷积使用了步长 6。六个字符似乎是一个合理的数量来捕捉局部信息，但也稍微随意。为什么不是四或五呢？实际上我们可以做这三种然后将结果合并：

```py
layers = []
for window in (4, 5, 6):
    conv_1x = Conv1D(128, window, activation='relu',
                     padding='valid')(char_input)
    max_pool_1x = MaxPooling1D(4)(conv_1x)
    conv_2x = Conv1D(256, window, activation='relu',
                     padding='valid')(max_pool_1x)
    max_pool_2x = MaxPooling1D(4)(conv_2x)
    layers.append(max_pool_2x)

merged = Concatenate(axis=1)(layers)
```

使用这个网络的额外层，训练过程中精度提高到 47%。但不幸的是，测试集上的精度仅达到 37%。这仍然比之前稍微好一点，但过拟合差距已经增加了很多。

有许多防止过拟合的技术，它们的共同点是限制模型可以学习的内容。其中最流行的之一是添加`Dropout`层。在训练期间，`Dropout`会随机将所有神经元的权重设置为零的一部分。这迫使网络更加稳健地学习，因为它不能依赖于特定的神经元存在。在预测期间，所有神经元都在工作，这会平均结果并减少异常值。这减缓了过拟合的速度。

在 Keras 中，我们像添加任何其他层一样添加`Dropout`。我们的模型随后变为：

```py
    for window in (4, 5, 6):
        conv_1x = Conv1D(128, window,
                         activation='relu', padding='valid')(char_input)
        max_pool_1x = MaxPooling1D(4)(conv_1x)
        dropout_1x = Dropout(drop_out)(max_pool_1x)
        conv_2x = Conv1D(256, window,
                         activation='relu', padding='valid')(dropout_1x)
        max_pool_2x = MaxPooling1D(4)(conv_2x)
        dropout_2x = Dropout(drop_out)(max_pool_2x)
        layers.append(dropout_2x)

    merged = Concatenate(axis=1)(layers)

    dropout = Dropout(drop_out)(merged)
```

选择 dropout 值有点艺术性。较高的值意味着更稳健的模型，但训练速度也更慢。使用 0.2 进行训练将训练精度提高到 0.43，测试精度提高到 0.39，这表明我们仍然可以更高。 

## 讨论

这个配方提供了一些我们可以使用的技术来改善网络性能的想法。通过添加更多层，尝试不同的窗口，并在不同位置引入`Dropout`层，我们有很多旋钮可以调整来优化我们的网络。找到最佳值的过程称为*超参数调整*。

有一些框架可以通过尝试各种组合来自动找到最佳参数。由于它们需要多次训练模型，您需要耐心等待或者可以同时训练多个实例来并行训练您的模型。

# 7.7 构建一个单词级模型

## 问题

推文是单词，而不仅仅是随机字符。您如何利用这一事实？

## 解决方案

训练一个以单词嵌入序列作为输入而不是字符序列的模型。

首先要做的是对我们的推文进行标记化。我们将构建一个保留前 50000 个单词的标记器，将其应用于我们的训练和测试集，然后填充两者，使它们具有统一的长度：

```py
VOCAB_SIZE = 50000
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(tweets['text'])
training_tokens = tokenizer.texts_to_sequences(train_tweets['text'])
test_tokens = tokenizer.texts_to_sequences(test_tweets['text'])
max_num_tokens = max(len(x) for x in chain(training_tokens, test_tokens))
training_tokens = pad_sequences(training_tokens, maxlen=max_num_tokens)
test_tokens = pad_sequences(test_tokens, maxlen=max_num_tokens)
```

我们可以通过使用预训练的嵌入来快速启动我们的模型（请参见第三章）。我们将使用一个实用函数`load_wv2`加载权重，它将加载 Word2vec 嵌入并将其与我们语料库中的单词匹配。这将构建一个矩阵，每个令牌包含来自 Word2vec 模型的权重的一行：

```py
def load_w2v(tokenizer=None):
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_vectors, binary=True)

    total_count = sum(tokenizer.word_counts.values())
    idf_dict = {k: np.log(total_count/v)
                for (k,v) in tokenizer.word_counts.items()}

    w2v = np.zeros((tokenizer.num_words, w2v_model.syn0.shape[1]))
    idf = np.zeros((tokenizer.num_words, 1))

    for k, v in tokenizer.word_index.items():
        if < tokenizer.num_words and k in w2v_model:
            w2v[v] = w2v_model[k]
            idf[v] = idf_dict[k]

    return w2v, idf
```

现在我们可以创建一个与我们的字符模型非常相似的模型，主要是改变如何处理输入。我们的输入接受一系列令牌，嵌入层在我们刚刚创建的矩阵中查找每个令牌：

```py
    message = Input(shape=(max_num_tokens,), dtype='int32', name='title')
    embedding = Embedding(mask_zero=False, input_dim=vocab_size,
                          output_dim=embedding_weights.shape[1],
                          weights=[embedding_weights],
                          trainable=False,
                          name='cnn_embedding')(message)
```

这个模型可以工作，但效果不如字符模型好。我们可以调整各种超参数，但差距相当大（字符级模型的精度为 38%，而单词级模型的精度为 30%）。有一件事可以改变这种情况——将嵌入层的`trainable`属性设置为`True`。这有助于将单词级模型的精度提高到 36%，但这也意味着我们使用了错误的嵌入。我们将在下一个配方中看看如何解决这个问题。

## 讨论

一个单词级模型比一个字符级模型对输入数据有更广泛的视角，因为它查看的是单词的簇，而不是字符的簇。我们使用单词嵌入来快速开始，而不是使用字符的独热编码。在这里，我们通过表示每个单词的向量来表示该单词的语义值作为模型的输入。（有关单词嵌入的更多信息，请参见第三章。）

这个配方中介绍的模型并没有超越我们的字符级模型，也没有比我们在配方 7.1 中看到的贝叶斯模型做得更好。这表明我们预训练的单词嵌入的权重与我们的问题不匹配。如果我们将嵌入层设置为可训练，事情会好得多；如果允许它更改这些嵌入，模型会有所改进。我们将在下一个配方中更详细地讨论这个问题。

权重不匹配并不令人惊讶。Word2vec 模型是在 Google 新闻上训练的，它的语言使用方式与社交媒体上的平均使用方式有很大不同。例如，流行的标签在 Google 新闻语料库中不会出现，而它们似乎对于分类推文非常重要。

# 7.8 构建您自己的嵌入

## 问题

如何获取与您的语料库匹配的单词嵌入？

## 解决方案

训练您自己的单词嵌入。

`gensim`包不仅让我们可以使用预训练的嵌入模型，还可以训练新的嵌入。它需要的唯一东西是一个生成器，产生令牌序列。它将使用这个生成器来建立词汇表，然后通过多次遍历生成器来训练模型。以下对象将遍历一系列推文，清理它们，并对其进行标记化：

```py
class TokensYielder(object):
    def __init__(self, tweet_count, stream):
        self.tweet_count = tweet_count
        self.stream = stream

    def __iter__(self):
        print('!')
        count = self.tweet_count
        for tweet in self.stream:
            if tweet.get('lang') != 'en':
                continue
            text = tweet['text']
            text = html.unescape(text)
            text = RE_WHITESPACE.sub(' ', text)
            text = RE_URL.sub(' ', text)
            text = strip_accents(text)
            text = ''.join(ch for ch in text if ord(ch) < 128)
            if text.startswith('RT '):
                text = text[3:]
            text = text.strip()
            if text:
                yield text_to_word_sequence(text)
                count -= 1
                if count <= 0:
                    break
```

现在我们可以训练模型了。明智的做法是收集一周左右的推文，将它们保存在一组文件中（每行一个 JSON 文档是一种流行的格式），然后将一个遍历文件的生成器传递到`TokensYielder`中。

在我们开始这项工作并等待一周让我们的推文涓涓细流进来之前，我们可以通过获取 100,000 条经过筛选的推文来测试这是否有效：

```py
tweets = list(TokensYielder(100000,
              twitter.TwitterStream(auth=auth).statuses.sample()))
```

然后构建模型：

```py
model = gensim.models.Word2Vec(tweets, min_count=2)
```

查看单词“爱”的最近邻居，我们发现我们确实有自己的领域特定的嵌入——只有在 Twitter 上，“453”与“爱”相关，因为在线上它是“酷故事，兄弟”的缩写：

```py
model.wv.most_similar(positive=['love'], topn=5)
```

```py
[('hate', 0.7243724465370178),
 ('loved', 0.7227891087532043),
 ('453', 0.707709789276123),
 ('melanin', 0.7069753408432007),
 ('appreciate', 0.696381688117981)]
```

“黑色素”稍微出乎意料。

## 讨论

使用现有的词嵌入是一个快速入门的好方法，但只适用于我们处理的文本与嵌入训练的文本相似的情况。在这种情况不成立且我们可以访问大量与我们正在训练的文本相似的文本的情况下，我们可以轻松地训练自己的词嵌入。

正如我们在前一篇文章中看到的，一个训练新嵌入的替代方法是使用现有的嵌入，但将层的`trainable`属性设置为`True`。这将使网络调整嵌入层中单词的权重，并在缺失的地方找到新的单词。

# 7.9 使用递归神经网络进行分类

## 问题

当然有一种方法可以利用推文是一系列单词的事实。你可以怎么做呢？

## 解决方案

使用单词级递归网络进行分类。

卷积网络适用于在输入流中发现局部模式。对于情感分析，这通常效果很好；某些短语会独立于它们出现的位置影响句子的情感。然而，建议表情符号的任务中有一个时间元素，我们没有利用 CNN。与推文相关联的表情符号通常是推文的结论。在这种情况下，RNN 可能更合适。

我们看到了如何教 RNN 生成文本在第五章。我们可以使用类似的方法来建议表情符号。就像单词级 CNN 一样，我们将输入转换为它们的嵌入的单词。一个单层 LSTM 表现得相当不错：

```py
def create_lstm_model(vocab_size, embedding_size=None, embedding_weights=None):
    message = layers.Input(shape=(None,), dtype='int32', name='title')
    embedding = Embedding(mask_zero=False, input_dim=vocab_size,
                          output_dim=embedding_weights.shape[1],
                          weights=[embedding_weights],
                          trainable=True,
                          name='lstm_embedding')(message)

    lstm_1 = layers.LSTM(units=128, return_sequences=False)(embedding)
    category = layers.Dense(units=len(emojis), activation='softmax')(lstm_1)

    model = Model(
        inputs=[message],
        outputs=[category],
    )
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    return model
```

经过 10 个时期，我们在训练集上达到了 50%的精度，在测试集上达到了 40%，远远超过了 CNN 模型。

## 讨论

我们在这里使用的 LSTM 模型明显优于我们的单词级 CNN。我们可以将这种卓越的性能归因于推文是序列，推文末尾发生的事情与开头发生的事情有不同的影响。

由于我们的字符级 CNN 往往比我们的单词级 CNN 做得更好，而我们的单词级 LSTM 比字符级 CNN 做得更好，我们可能会想知道字符级 LSTM 是否会更好。结果表明并不是。

原因是，如果我们一次向 LSTM 输入一个字符，到达末尾时，它大部分时间都会忘记推文开头发生的事情。如果我们一次向 LSTM 输入一个单词，它就能克服这个问题。还要注意，我们的字符级 CNN 实际上并不是一次处理一个字符。我们一次使用四、五或六个字符的序列，并且将多个卷积层堆叠在一起，这样平均推文在最高级别只剩下三个特征向量。

我们可以尝试将两者结合起来，通过创建一个 CNN 将推文压缩成更高抽象级别的片段，然后将这些向量输入到 LSTM 中得出最终结论。当然，这与我们的单词级 LSTM 的工作方式非常接近。我们不是使用 CNN 对文本片段进行分类，而是使用预训练的词嵌入在每个单词级别上执行相同的操作。

# 7.10 可视化（不）一致性

## 问题

您希望可视化您构建的不同模型在实践中的比较。

## 解决方案

使用 Pandas 显示它们的一致性和不一致性。

精度给我们一个关于我们的模型表现如何的概念。虽然建议表情符号是一个相当嘈杂的任务，但是将我们的各种模型的表现并排进行比较是非常有用的。Pandas 是一个很好的工具。

让我们首先将字符模型的测试数据作为向量而不是生成器导入：

```py
test_char_vectors, _ = next(data_generator(test_tweets, None))
```

现在让我们对前 100 个项目进行预测：

```py
predictions = {
    label: [emojis[np.argmax(x)] for x in pred]
    for label, pred in (
        ('lstm', lstm_model.predict(test_tokens[:100])),
        ('char_cnn', char_cnn_model.predict(test_char_vectors[:100])),
        ('cnn', cnn_model.predict(test_tokens[:100])),
    )
}
```

现在我们可以构建并显示一个 Pandas `DataFrame`，其中包含每个模型的前 25 个预测，以及推文文本和原始表情符号：

```py
pd.options.display.max_colwidth = 128
test_df = test_tweets[:100].reset_index()
eval_df = pd.DataFrame({
    'content': test_df['text'],
    'true': test_df['emoji'],
    **predictions
})
eval_df[['content', 'true', 'char_cnn', 'cnn', 'lstm']].head(25)
```

这样就得到了：

| # | 内容 | 真实 | char_cnn | cnn | lstm |
| --- | --- | --- | --- | --- | --- |
| 0 | @Gurmeetramrahim @RedFMIndia @rjraunac #8DaysToLionHeart 太棒了 | ![](img/clapping-hands.png) | ![](img/thumbs-up.png) | ![](img/clapping-hands.png) | ![](img/face-throwing-a-kiss.png) |
| 1 | @suchsmallgods 我迫不及待想向他展示这些推文 | ![](img/smiling-face-with-horns.png) | ![](img/face-with-tears-of-joy.png) | ![](img/red-heart.png) | ![](img/loudly-crying-face.png) |
| 2 | @Captain_RedWolf 我有大约 20 套 lol 比你领先太多了 | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) |
| 3 | @OtherkinOK 刚刚在@EPfestival，太棒了！下一站是@whelanslive，2016 年 11 月 11 日星期五。 | ![](img/ok-hand-sign.png) | ![](img/flexed-biceps.png) | ![](img/red-heart.png) | ![](img/smiling-face-with-smiling-eyes.png) |
| 4 | @jochendria: KathNiel 与 GForce Jorge。#PushAwardsKathNiels | ![](img/blue-heart.png) | ![](img/blue-heart.png) | ![](img/blue-heart.png) | ![](img/blue-heart.png) |
| 5 | 好的 | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/red-heart.png) | ![](img/face-with-tears-of-joy.png) |
| 6 | “Distraught 意味着心烦意乱” “那意味着困惑对吧？” -@ReevesDakota | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/red-heart.png) | ![](img/loudly-crying-face.png) |
| 7 | @JennLiri 宝贝，怎么了，打电话回来，我想听这首铃声 | ![](img/face-with-rolling-eyes.png) | ![](img/face-with-rolling-eyes.png) | ![](img/copyright-symbol.png) | ![](img/weary-face.png) |
| 8 | 珍想要做朋友吗？我们可以成为朋友。爱你，女孩。#BachelorInParadise | ![](img/red-heart.png) | ![](img/crying-face.png) | ![](img/red-heart.png) | ![](img/red-heart.png) |
| 9 | @amwalker38: 去关注这些热门账号 @the1stMe420 @DanaDeelish @So_deelish @aka_teemoney38 @CamPromoXXX @SexyLThings @l... | ![](img/downpointing-backhand.png) | ![](img/crown.png) | ![](img/fire.png) | ![](img/sparkles.png) |
| 10 | @gspisak: 我总是取笑那些提前 30 分钟以上来接孩子的父母，今天轮到我了，至少我得到了... | ![](img/see-no-evil-monkey.png) | ![](img/upside-down-face.png) | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) |
| 11 | @ShawnMendes: 多伦多广告牌。太酷了！@spotify #ShawnXSpotify 去找到你所在城市的广告牌 | ![](img/smiling-face-with-smiling-eyes.png) | ![](img/smiling-face-with-smiling-eyes.png) | ![](img/smiling-face-with-smiling-eyes.png) | ![](img/smiling-face-with-smiling-eyes.png) |
| 12 | @kayleeburt77 我可以要你的号码吗？我好像把我的弄丢了。 | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/thinking-face.png) |
| 13 | @KentMurphy: 蒂姆·提博在职业棒球比赛中第一球就击出了一支全垒打 | ![](img/flushed-face.png) | ![](img/flushed-face.png) | ![](img/flushed-face.png) | ![](img/flushed-face.png) |
| 14 | @HailKingSoup... | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/loudly-crying-face.png) | ![](img/face-with-tears-of-joy.png) |
| 15 | @RoxeteraRibbons 同样，我必须找出证明 | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/smiling-face-with-smiling-eyes.png) |
| 16 | @theseoulstory: 九月回归：2PM，SHINee，INFINITE，BTS，Red Velvet，Gain，Song Jieun，Kanto... | ![](img/fire.png) | ![](img/fire.png) | ![](img/fire.png) | ![](img/fire.png) |
| 17 | @VixenMusicLabel - 和平与爱 | ![](img/victory-hand.png) | ![](img/red-heart.png) | ![](img/face-throwing-a-kiss.png) | ![](img/red-heart.png) |
| 18 | @iDrinkGallons 抱歉 | ![](img/slightly-frowning-face.png) | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) |
| 19 | @StarYouFollow: 19- Frisson | ![](img/face-with-rolling-eyes.png) | ![](img/ok-hand-sign.png) | ![](img/smiling-face-with-heart-shaped-eyes.png) | ![](img/sparkles.png) |
| 20 | @RapsDaiIy: 别错过 Ugly God | ![](img/fire.png) | ![](img/fire.png) | ![](img/fire.png) | ![](img/fire.png) |
| 21 | 怎么我的所有班次都这么快被接走了？！什么鬼 | ![](img/loudly-crying-face.png) | ![](img/weary-face.png) | ![](img/face-with-tears-of-joy.png) | ![](img/weary-face.png) |
| 22 | @ShadowhuntersTV: #Shadowhunters 粉丝，你们会给这个父女#FlashbackFriday 亲密时刻打几分？ | ![](img/red-heart.png) | ![](img/red-heart.png) | ![](img/red-heart.png) | ![](img/red-heart.png) |
| 23 | @mbaylisxo: 感谢上帝，我有一套制服，不用每天担心穿什么 | ![](img/smiling-face-with-open-mouth-and-cold-sweat.png) | ![](img/face-with-rolling-eyes.png) | ![](img/red-heart.png) | ![](img/smiling-face-with-smiling-eyes.png) |
| 24 | 心情波动如... | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-tears-of-joy.png) | ![](img/face-with-rolling-eyes.png) |

浏览这些结果，我们可以看到通常当模型出错时，它们会落在与原始推文中非常相似的表情符号上。有时，预测似乎比实际使用的更有意义，有时候没有一个模型表现得很好。

## 讨论

查看实际数据可以帮助我们看到我们的模型出错的地方。在这种情况下，提高性能的一个简单方法是将所有相似的表情符号视为相同的。不同的心形和不同的笑脸表达的基本上是相同的。

一个替代方案是为表情符号学习嵌入。这将给我们一个关于表情符号相关性的概念。然后，我们可以有一个损失函数，考虑到这种相似性，而不是一个硬性的正确/错误度量。

# 7.11 结合模型

## 问题

您希望利用模型的联合预测能力获得更好的答案。

## 解决方案

将模型组合成一个集成模型。

群体智慧的概念——即群体意见的平均值通常比任何具体意见更准确——也适用于机器学习模型。我们可以通过使用三个输入将所有三个模型合并为一个模型，并使用 Keras 的`Average`层来组合我们模型的输出：

```py
def prediction_layer(model):
    layers = [layer for layer in model.layers
              if layer.name.endswith('_predictions')]
    return layers[0].output

def create_ensemble(*models):
    inputs = [model.input for model in models]
    predictions = [prediction_layer(model) for model in models]
    merged = Average()(predictions)
    model = Model(
        inputs=inputs,
        outputs=[merged],
    )
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
```

我们需要一个不同的数据生成器来训练这个模型；而不是指定一个输入，我们现在有三个输入。由于它们有不同的名称，我们可以让我们的数据生成器产生一个字典来提供这三个输入。我们还需要做一些整理工作，使字符级数据与单词级数据对齐：

```py
def combined_data_generator(tweets, tokens, batch_size):
    tweets = tweets.reset_index()
    while True:
        batch_idx = random.sample(range(len(tweets)), batch_size)
        tweet_batch = tweets.iloc[batch_idx]
        token_batch = tokens[batch_idx]
        char_vec = np.zeros((batch_size, max_sequence_len, len(chars)))
        token_vec = np.zeros((batch_size, max_num_tokens))
        y = np.zeros((batch_size,))
        it = enumerate(zip(token_batch, tweet_batch.iterrows()))
        for row_idx, (token_row, (_, tweet_row)) in it:
            y[row_idx] = emoji_to_idx[tweet_row['emoji']]
            for ch_idx, ch in enumerate(tweet_row['text']):
                char_vec[row_idx, ch_idx, char_to_idx[ch]] = 1
            token_vec[row_idx, :] = token_row
        yield {'char_cnn_input': char_vec,
               'cnn_input': token_vec,
               'lstm_input': token_vec}, y
```

然后我们可以使用以下方式训练模型：

```py
BATCH_SIZE = 512
ensemble.fit_generator(
    combined_data_generator(train_tweets, training_tokens, BATCH_SIZE),
    epochs=20,
    steps_per_epoch=len(train_tweets) / BATCH_SIZE,
    verbose=2,
    callbacks=[early]
)
```

## 讨论

组合模型或集成模型是将各种方法结合到一个模型中解决问题的好方法。在像 Kaggle 这样的流行机器学习竞赛中，获胜者几乎总是基于这种技术，这并非巧合。

而不是将模型几乎完全分开，然后在最后使用`Average`层将它们连接起来，我们也可以在更早的阶段将它们连接起来，例如通过连接每个模型的第一个密集层。实际上，这在更复杂的 CNN 中是我们所做的一部分，我们使用了各种窗口大小的小子网，然后将它们连接起来得出最终结论。
