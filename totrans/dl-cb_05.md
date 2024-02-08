# 第5章 生成类似示例文本风格的文本

在本章中，我们将看看如何使用递归神经网络（RNN）生成类似文本体的文本。这将产生有趣的演示。人们已经使用这种类型的网络生成从婴儿姓名到颜色描述等各种内容。这些演示是熟悉递归网络的好方法。RNN也有它们的实际用途——在本书的后面，我们将使用它们来训练聊天机器人，并基于收集的播放列表构建音乐推荐系统，RNN已经被用于生产中跟踪视频中的对象。

递归神经网络是一种在处理时间或序列时有帮助的神经网络类型。我们将首先查看Project Gutenberg作为免费书籍的来源，并使用一些简单的代码下载威廉·莎士比亚的作品集。接下来，我们将使用RNN生成似乎是莎士比亚风格的文本（如果您不太注意的话），通过训练网络下载的文本。然后，我们将在Python代码上重复这一技巧，并看看如何改变输出。最后，由于Python代码具有可预测的结构，我们可以查看哪些神经元在哪些代码位上激活，并可视化我们的RNN的工作原理。

本章的代码可以在以下Python笔记本中找到：

```py
05.1 Generating Text in the Style of an Example Text
```

# 5.1 获取公共领域书籍的文本

## 问题

您想要下载一些公共领域书籍的完整文本，以用于训练您的模型。

## 解决方案

使用Project Gutenberg的Python API。

Project Gutenberg包含超过5万本书的完整文本。有一个方便的Python API可用于浏览和下载这些书籍。如果我们知道ID，我们可以下载任何一本书：

```py
shakespeare = load_etext(100)
shakespeare = strip_headers(shakespeare)
```

我们可以通过浏览网站并从书籍的URL中提取书籍的ID，或者通过作者或标题查询[*http://www.gutenberg.org/*](http://www.gutenberg.org/) 来获取书籍的ID。但在查询之前，我们需要填充元信息缓存。这将创建一个包含所有可用书籍的本地数据库。这需要一点时间，但只需要做一次：

```py
cache = get_metadata_cache()
cache.populate()
```

我们现在可以发现莎士比亚的所有作品：

```py
for text_id in get_etexts('author', 'Shakespeare, William'):
    print(text_id, list(get_metadata('title', text_id))[0])
```

## 讨论

Project Gutenberg是一个志愿者项目，旨在数字化图书。它专注于提供美国版权已过期的英语中最重要的书籍，尽管它也有其他语言的书籍。该项目始于1971年，早于迈克尔·哈特发明万维网之前。

在1923年之前在美国出版的任何作品都属于公共领域，因此古腾堡收藏中的大多数书籍都比这个年代更久远。这意味着语言可能有些过时，但对于自然语言处理来说，该收藏仍然是无与伦比的训练数据来源。通过Python API进行访问不仅使访问变得容易，而且还尊重该网站为自动下载文本设置的限制。

# 5.2 生成类似莎士比亚的文本

## 问题

如何以特定风格生成文本？

## 解决方案

使用字符级RNN。

让我们首先获取莎士比亚的作品集。我们将放弃诗歌，这样我们就只剩下戏剧的一组更一致的作品。诗歌恰好被收集在第一个条目中：

```py
shakespeare = strip_headers(load_etext(100))
plays = shakespeare.split('\nTHE END\n', 1)[-1]
```

我们将逐个字符地输入文本，并对每个字符进行独热编码——也就是说，每个字符将被编码为一个包含所有0和一个1的向量。为此，我们需要知道我们将遇到哪些字符：

```py
chars = list(sorted(set(plays)))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
```

让我们创建一个模型，该模型将接收一个字符序列并预测一个字符序列。我们将把序列输入到多个LSTM层中进行处理。`TimeDistributed`层让我们的模型再次输出一个序列：

```py
def char_rnn_model(num_chars, num_layers, num_nodes=512, dropout=0.1):
    input = Input(shape=(None, num_chars), name='input')
    prev = input
    for i in range(num_layers):
        prev = LSTM(num_nodes, return_sequences=True)(prev)
    dense = TimeDistributed(Dense(num_chars, name='dense',
                                  activation='softmax'))(prev)
    model = Model(inputs=[input], outputs=[dense])
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model
```

我们将向网络中随机提供剧本片段，因此生成器似乎是合适的。生成器将产生成对序列块，其中成对的序列仅相差一个字符：

```py
def data_generator(all_text, num_chars, batch_size):
    X = np.zeros((batch_size, CHUNK_SIZE, num_chars))
    y = np.zeros((batch_size, CHUNK_SIZE, num_chars))
    while True:
        for row in range(batch_size):
            idx = random.randrange(len(all_text) - CHUNK_SIZE - 1)
            chunk = np.zeros((CHUNK_SIZE + 1, num_chars))
            for i in range(CHUNK_SIZE + 1):
                chunk[i, char_to_idx[all_text[idx + i]]] = 1
            X[row, :, :] = chunk[:CHUNK_SIZE]
            y[row, :, :] = chunk[1:]
        yield X, y
```

现在我们将训练模型。我们将设置`steps_per_epoch`，以便每个字符都有足够的机会被网络看到：

```py
model.fit_generator(
    data_generator(plays, len(chars), batch_size=256),
    epochs=10,
    steps_per_epoch=2 * len(plays) / (256 * CHUNK_SIZE),
    verbose=2
)
```

训练后，我们可以生成一些输出。我们从剧本中随机选择一个片段，让模型猜测下一个字符是什么。然后我们将下一个字符添加到片段中，并重复，直到达到所需的字符数：

```py
def generate_output(model, start_index=None, diversity=1.0, amount=400):
    if start_index is None:
        start_index = random.randint(0, len(plays) - CHUNK_SIZE - 1)
    fragment = plays[start_index: start_index + CHUNK_SIZE]
    generated = fragment
    for i in range(amount):
        x = np.zeros((1, CHUNK_SIZE, len(chars)))
        for t, char in enumerate(fragment):
            x[0, t, char_to_idx[char]] = 1.
        preds = model.predict(x, verbose=0)[0]
        preds = np.asarray(preds[len(generated) - 1])
        next_index = np.argmax(preds)
        next_char = chars[next_index]

        generated += next_char
        fragment = fragment[1:] + next_char
    return generated

for line in generate_output(model).split('\n'):
    print(line)
```

经过10个时期，我们应该看到一些让我们想起莎士比亚的文本，但我们需要大约30个时期才能开始看起来像是可以愚弄一个不太注意的读者：

```py
FOURTH CITIZEN. They were all the summer hearts.
  The King is a virtuous mistress.
CLEOPATRA. I do not know what I have seen him damn'd in no man
  That we have spoken with the season of the world,
  And therefore I will not speak with you.
  I have a son of Greece, and my son
  That we have seen the sea, the seasons of the world
  I will not stay the like offence.
```

```py
OLIVIA. If it be aught and servants, and something
  have not been a great deal of state)) of the world, I will not stay
  the forest was the fair and not by the way.
SECOND LORD. I will not serve your hour.
FIRST SOLDIER. Here is a princely son, and the world
  in a place where the world is all along.
SECOND LORD. I will not see thee this:
  He hath a heart of men to men may strike and starve.
  I have a son of Greece, whom they say,
  The whiteneth made him like a deadly hand
  And make the seasons of the world,
  And then the seasons and a fine hands are parted
  To the present winter's parts of this deed.
  The manner of the world shall not be a man.
  The King hath sent for thee.
  The world is slain.
```

克利奥帕特拉和第二勋爵都有一个希腊的儿子，但目前的冬天和世界被杀死是合适的*权力的游戏*。

## 讨论

在这个配方中，我们看到了如何使用RNN生成特定风格的文本。结果相当令人信服，尤其是考虑到模型是基于字符级别进行预测的。由于LSTM架构，网络能够学习跨越相当大序列的关系——不仅仅是单词，还有句子，甚至是莎士比亚戏剧布局的基本结构。

尽管这里展示的示例并不是非常实用，但RNN可能是实用的。每当我们想要让网络学习一系列项目时，RNN可能是一个不错的选择。

其他人使用这种技术构建的玩具应用程序生成了婴儿姓名、油漆颜色名称，甚至食谱。

更实用的RNN可以用于预测用户在智能手机键盘应用程序中要输入的下一个字符，或者在训练了一组开局后预测下一步棋。这种类型的网络还被用来预测诸如天气模式或甚至股市价格等序列。

循环网络相当易变。看似对网络架构的微小更改可能会导致它们不再收敛，因为所谓的*梯度爆炸问题*。有时在训练过程中，经过多个时期取得进展后，网络似乎会崩溃并开始忘记它所学到的东西。一如既往，最好从一些简单的有效方法开始，逐步增加复杂性，同时跟踪所做的更改。

有关RNN的稍微深入的讨论，请参阅[第1章](ch01.html#tools_techniques)。

# 使用RNN编写代码

## 问题

如何使用神经网络生成Python代码？

## 解决方案

训练一个循环神经网络，使用Python分发的Python代码运行您的脚本。

实际上，我们可以为这个任务使用与上一个配方中几乎相同的模型。就深度学习而言，关键是获取数据。Python附带许多模块的源代码。由于它们存储在*random.py*模块所在的目录中，我们可以使用以下方法收集它们：

```py
def find_python(rootdir):
    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for fn in filenames:
            if fn.endswith('.py'):
                matches.append(os.path.join(root, fn))

    return matches
srcs = find_python(random.__file__.rsplit('/', 1)[0])
```

然后我们可以读取所有这些源文件，并将它们连接成一个文档，并开始生成新的片段，就像我们在上一个配方中对莎士比亚文本所做的那样。这种方法效果相当不错，但是在生成片段时，很明显很大一部分Python源代码实际上是英语。英语既出现在注释中，也出现在字符串的内容中。我们希望我们的模型学习Python，而不是英语！

去除注释很容易：

```py
COMMENT_RE = re.compile('#.*')
src = COMMENT_RE.sub('', src)
```

删除字符串的内容稍微复杂一些。一些字符串包含有用的模式，而不是英语。作为一个粗略的规则，我们将用`"MSG"`替换任何具有超过六个字母和至少一个空格的文本片段：

```py
def replacer(value):
    if ' ' in value and sum(1 for ch in value if ch.isalpha()) > 6:
        return 'MSG'
    return value
```

使用正则表达式可以简洁地找到字符串文字的出现。但是正则表达式速度相当慢，而且我们正在对大量代码进行运行。在这种情况下，最好只扫描字符串：

```py
def replace_literals(st):
    res = []
    start_text = start_quote = i = 0
    quote = ''
    while i < len(st):
        if quote:
            if st[i: i + len(quote)] == quote:
                quote = ''
                start_text = i
                res.append(replacer(st[start_quote: i]))
        elif st[i] in '"\'':
            quote = st[i]
            if i < len(st) - 2 and st[i + 1] == st[i + 2] == quote:
                quote = 3 * quote
            start_quote = i + len(quote)
            res.append(st[start_text: start_quote])
        if st[i] == '\n' and len(quote) == 1:
            start_text = i
            res.append(quote)
            quote = ''
        if st[i] == '\\':
            i += 1
        i += 1
    return ''.join(res) + st[start_text:]
```

即使以这种方式清理，我们最终会得到几兆字节的纯Python代码。现在我们可以像以前一样训练模型，但是在Python代码而不是戏剧上。经过大约30个时代，我们应该有可行的东西，并且可以生成代码。

## 讨论

生成Python代码与编写莎士比亚风格的戏剧没有什么不同——至少对于神经网络来说是这样。我们已经看到，清理输入数据是神经网络数据处理的一个重要方面。在这种情况下，我们确保从源代码中删除了大部分英语的痕迹。这样，网络就可以专注于学习Python，而不会被学习英语分散注意力。

我们可以进一步规范输入。例如，我们可以首先将所有源代码通过“漂亮的打印机”传递，以便它们都具有相同的布局，这样我们的网络就可以专注于学习这一点，而不是当前代码中的多样性。更进一步的一步是使用内置的标记器对Python代码进行标记化，然后让网络学习这个解析版本，并使用`untokenize`生成代码。

# 5.4 控制输出的温度

## 问题

您想控制生成代码的变化性。

## 解决方案

将预测作为概率分布，而不是选择最高值。

在莎士比亚的例子中，我们选择了预测中得分最高的字符。这种方法会导致模型最喜欢的输出。缺点是我们对每个开始都得到相同的输出。由于我们从实际的莎士比亚文本中选择了一个随机的开始序列，这并不重要。但是如果我们想要生成Python函数，最好总是以相同的方式开始——比如以`/ndef`开始，并查看各种解决方案。

我们的网络的预测是通过softmax激活函数得出的结果，因此可以看作是一个概率分布。因此，我们可以让`numpy.random.multinomial`给出答案，而不是选择最大值。`multinomial`运行*n*次实验，并考虑结果的可能性。通过将*n* = 1运行，我们得到了我们想要的结果。

在这一点上，我们可以引入在如何绘制结果时引入温度的概念。这个想法是，温度越高，结果就越随机，而较低的温度则更接近我们之前看到的纯确定性结果。我们通过相应地缩放预测的对数，然后再次应用softmax函数来获得概率。将所有这些放在一起，我们得到：

```py
def generate_code(model, start_with='\ndef ',
                  end_with='\n\n', diversity=1.0):
    generated = start_with
    yield generated
    for i in range(2000):
        x = np.zeros((1, len(generated), len(chars)))
        for t, char in enumerate(generated):
            x[0, t, char_to_idx[char]] = 1.
        preds = model.predict(x, verbose=0)[0]

        preds = np.asarray(preds[len(generated) - 1]).astype('float64')
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        next_index = np.argmax(probas)
        next_char = chars[next_index]
        yield next_char

        generated += next_char
        if generated.endswith(end_with):
            break
```

我们终于准备好玩一些了。在`diversity=1.0`时，生成了以下代码。请注意模型生成了我们的`"MSG"`占位符，并且除了混淆了`val`和`value`之外，几乎让我们运行了代码：

```py
def _calculate_ratio(val):
    """MSG"""
    if value and value[0] != '0':
        raise errors.HeaderParseError(
            "MSG".format(Storable))
    return value
```

## 讨论

将softmax激活函数的输出作为概率分布使我们能够获得与模型“意图”相对应的各种结果。一个额外的好处是，它使我们能够引入温度的概念，因此我们可以控制输出的“随机性”程度。在[第13章](ch13.html#autoencoders)中，我们将看到*变分* *自动编码器*如何使用类似的技术来控制生成的随机性。

如果我们不注意细节，生成的Python代码肯定可以通过真实代码。进一步改进结果的一种方法是在生成的代码上调用`compile`函数，并且仅保留编译的代码。这样我们可以确保它至少在语法上是正确的。稍微变化这种方法的一种方法是在语法错误时不重新开始，而只是删除发生错误的行以及其后的所有内容，然后重试。

# 5.5 可视化循环网络激活

## 问题

如何获得对循环网络正在执行的操作的洞察？

## 解决方案

提取神经元处理文本时的激活。由于我们将要可视化神经元，将它们的数量减少是有意义的。这会稍微降低模型的性能，但使事情变得更简单：

```py
flat_model = char_rnn_model(len(py_chars), num_layers=1, num_nodes=512)
```

这个模型稍微简单一些，得到的结果略微不够准确，但足够用于可视化。Keras有一个方便的方法叫做`function`，它允许我们指定一个输入和一个输出层，然后运行网络中需要的部分来进行转换。以下方法为网络提供了一段文本（一系列字符），并获取特定层的激活值：

```py
def activations(model, code):
    x = np.zeros((1, len(code), len(py_char_to_idx)))
    for t, char in enumerate(code):
        x[0, t, py_char_to_idx[char]] = 1.
    output = model.get_layer('lstm_3').output
    f = K.function([model.input], [output])
    return f([x])[0][0]
```

现在的问题是要看哪些神经元。即使我们简化的模型有512个神经元。LSTM中的激活值在-1到1之间，因此找到有趣的神经元的一个简单方法就是选择与每个字符对应的最高值。`np.argmax(act, axis=1)`将帮助我们实现这一点。我们可以使用以下方式可视化这些神经元：

```py
img = np.full((len(neurons) + 1, len(code), 3), 128)
scores = (act[:, neurons].T + 1) / 2
img[1:, :, 0] = 255 * (1 - scores)
img[1:, :, 1] = 255 * scores
```

这将产生一个小的位图。当我们放大位图并在顶部绘制代码时，我们得到：

![RNN中的神经元激活](assets/dlcb_05in01.png)

这看起来很有趣。顶部的神经元似乎跟踪新语句的开始位置。带有绿色条的神经元跟踪空格，但仅限于用于缩进。倒数第二个神经元似乎在有`=`符号时激活，但在有`==`时不激活，这表明网络学会了赋值和相等之间的区别。

## 讨论

深度学习模型可能非常有效，但它们的结果很难解释。我们更多或少了解训练和推理的机制，但往往很难解释一个具体的结果，除了指向实际的计算。可视化激活是使网络学到的内容更清晰的一种方式。

查看每个字符的最高激活的神经元很快地得到了一组可能感兴趣的神经元。或者，我们可以明确地尝试寻找在特定情况下激活的神经元，例如在括号内。

一旦我们有一个看起来有趣的特定神经元，我们可以使用相同的着色技术来突出显示更大的代码块。
