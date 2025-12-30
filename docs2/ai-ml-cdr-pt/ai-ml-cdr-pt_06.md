# 第五章 自然语言处理简介

自然语言处理（NLP）是人工智能（AI）中的一种技术，它处理语言的理解。它涉及编程技术来创建可以理解语言、分类内容，甚至生成和创作新语言作品的模型。它也是大型语言模型（LLMs）如 GPT、Gemini 和 Claude 的底层基础。我们将在后面的章节中探讨 LLMs，但首先，在接下来的几章中，我们将探讨更基础的 NLP，以便为你即将学习的内容做好准备。

还有许多服务使用自然语言处理（NLP）来创建应用程序，如聊天机器人，但这本书的范围不包括这一点——相反，我们将探讨 NLP 的基础以及如何建模语言，以便你可以训练神经网络来理解和分类文本。在后面的章节中，你还将学习如何使用机器学习（ML）模型的预测元素来写一些诗歌。这不仅仅是为了好玩——这也是学习如何使用支撑生成式人工智能（AI）的基于转换器的模型的一个先导。

我们将从这个章节开始，探讨如何将语言分解成数字，以及如何使用这些数字在神经网络中使用。  

# 将语言编码成数字

最终，计算机处理的是数字，因此为了处理语言，你需要将其转换为数字，这个过程称为*编码*。

你可以用许多方式将语言编码成数字。最常见的是通过字母编码，就像在程序中存储字符串时自然发生的那样。然而，在内存中，你并不存储字母*a*，而是它的编码——可能是 ASCII 或 Unicode 值或其他。例如，考虑单词*listen*。你可以用 ASCII 将其编码为数字 76, 73, 83, 84, 69 和 78。这是好的，因为现在你可以用数字来表示这个单词。但考虑单词*silent*，它是*listen*的字母表重组。相同的数字代表这个单词，尽管顺序不同，这可能会使构建一个理解文本的模型变得更加困难。

一个更好的选择可能是使用数字来编码整个单词，而不是单词中的字母。在这种情况下，*silent*可以是数字*x*，而*listen*可以是数字*y*，它们不会相互重叠。

使用这种技术，考虑一个句子如“我爱我的狗。”你可以用数字[1, 2, 3, 4]来编码它。如果你想编码“我爱我的猫”，你可以用[1, 2, 3, 5]来编码。到现在，你可能已经到了可以判断句子具有相似意义的地步，因为它们在数值上相似——换句话说，[1, 2, 3, 4]看起来非常像[1, 2, 3, 5]。

代表单词的数字也被称为*标记*，因此这个过程被称为*分词*。你将在下一节中探索如何在代码中实现这一点。

## 开始使用分词

PyTorch 生态系统包含许多用于标记化的库，它将单词转换为标记。你可能在代码示例中看到的一个常见标记器是`torchtext`，但自 2023 年以来它已经被弃用。所以，在使用它时要小心，特别是因为 PyTorch 版本在进步，但它没有。因此，一些替代方案是使用自定义标记器、来自其他地方的预训练标记器，或者（令人惊讶的是）来自 Keras 生态系统的标记器。

### 使用自定义标记器

为了给你一个简单的例子，这里有一些我用来创建自定义标记器的代码，将一个小语料库（两个句子）的单词转换为标记：

```py
import torch

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

# Tokenization function
def tokenize(text):
    return text.lower().split()

# Build the vocabulary
def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        tokens = tokenize(sentence)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1 
    return vocab

# Create the vocabulary index
vocab = build_vocab(sentences)

print("Vocabulary Index:", vocab)
```

###### 注意

单词*corpus*通常用来表示一组你将用于训练的文本项。它字面上是你将用于训练模型和创建标记器的文本*主体*。

这个输出的结果如下：

```py
Vocabulary Index: {'today': 1, 'is': 2, 'a': 3, 'sunny': 4, 'day': 5, 'rainy': 6}
```

如你所见，标记器完成了一个非常简单的任务，创建了一个包含我的词汇表的单词列表，每次它遇到一个独特的单词，它就会将其添加到列表中。所以第一个句子，“今天是个晴天”，产生了五个标记，对应五个单词：“今天”、“是”、“一个”、“晴天”和“天”。第二个句子有*大多数*这些单词是共同的，除了“雨天”，所以它成为了第六个标记。

另一方面，你可以想象，对于一个非常大的语料库，这个过程会非常慢。

### 使用 Hugging Face 的预训练标记器

考虑到这一点，我将使用 Hugging Face 的`transformers`库和其中的预构建标记器。在这种情况下，因为`transformers`库支持许多语言模型，而这些语言模型需要标记器来处理它们的文本语料库，所以这个在数百万个单词上训练的标记器是免费供你使用的。它的覆盖范围比你自己创建的更大，而且它是免费且易于使用的！

如果你还没有这个库，你可以使用以下命令安装它：

```py
!pip install transformers
```

现在，让我们用一个简单的例子来看看它的实际应用：

```py
from transformers import BertTokenizerFast

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, 
                           return_tensors='pt')

# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids) 
           for ids in encoded_inputs["input_ids"]]

# To get the word index similar to Keras' tokenizer
word_index = tokenizer.get_vocab()

print("Tokens:", tokens)
print("Token IDs:", encoded_inputs['input_ids'])
print("Word Index:", dict(list(word_index.items())[:10]))  
# show only the first 10 for brevity

```

它的输出看起来像这样：

```py
Tokens: [['[CLS]', 'today', 'is', 'a', 'sunny', 'day', '[SEP]'], 
         ['[CLS]', 'today', 'is', 'a', 'rainy', 'day', '[SEP]']]

Token IDs: tensor([
        [  101,  2651,  2003,  1037, 11559,  2154,   102],
        [  101,  2651,  2003,  1037, 16373,  2154,   102]])

Word Index: {'protestant': 8330, 'initial': 3988, '##pt': 13876, 
             'charters': 23010, '243': 22884, 'ref': 25416, '##dies': 18389, 
             '##uchi': 15217, 'sainte': 16947, 'annette': 22521}
```

现在，让我们来分解这个过程。我们首先从`transformers`库中导入`BertTokenizerFast`。这个类可以用多个预训练的标记器进行初始化，我们选择了`'bert-base-uncased'`这个版本。你可能想知道这究竟是什么！好吧，这里的想法是我想使用一个预训练的标记器，通常这些标记器是与它们训练的模型配对的。BERT（代表来自转换器的双向编码器表示）是由谷歌在一个大型语料库上训练的模型，拥有 30,000 个词汇。你可以在 Hugging Face 模型仓库中找到这样的模型，当你深入一个模型时，你通常会看到获取其标记器的转换器代码。例如，看看[这个页面](https://oreil.ly/Ok7L9)——虽然我没有使用这个模型，但我仍然可以获取它的标记器而不是创建一个自定义的标记器。

在这种情况下，我们创建一个 `tokenizer` 对象并指定它可以分词的单词数量。这将是从单词语料库中生成的最大分词数。我们这里有一个非常小的语料库，只包含六个独特的单词，所以我们将在指定的最大值一百以下。

一旦我有了分词器，我就可以直接将文本传递给它：

```py
# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, 
                           return_tensors='pt')

```

我们将在本章稍后探索填充和截断，但就目前而言，你应该注意 `return_tensors='pt'` 参数。这对我们 PyTorch 开发者来说是一个很好的便利，因为返回值将是 `torch.Tensor` 对象，这使我们很容易处理。

BERT 模型在原始分词上使用了一些叠加，例如 `attention_masking`，这意味着它使用每个单词的 `IDs` 而不是原始分词。这超出了本章的范围，但对你当前的影响是，如果你不需要所有这些。如果你只需要分词，你必须以以下方式提取分词，注意你的句子在 BERT 分词器中被编码为 `input_Ids`：

```py
# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids) 
           for ids in encoded_inputs["input_ids"]]
```

一旦你完成了这个，你就可以轻松地打印出以下 `Tokens` 集合：

```py
Tokens: [['[CLS]', 'today', 'is', 'a', 'sunny', 'day', '[SEP]'], 
         ['[CLS]', 'today', 'is', 'a', 'rainy', 'day', '[SEP]']]
```

现在，你可能想知道 `[CLS]` 和 `[SEP]` 是什么——以及 BERT 模型是如何被训练成期望句子以 `[CLS]`（用于*分类器*）开始，以 `[SEP]`（用于*分隔符*）结束或分隔的。这两个表达式分别被分词为值 101 和 102，所以当你打印出你句子的分词值时，你会看到这个：

```py
Token IDs: tensor([
        [  101,  2651,  2003,  1037, 11559,  2154,   102],
        [  101,  2651,  2003,  1037, 16373,  2154,   102]])
```

从这个中，你可以推导出 *今天* 在 BERT 中是分词 2651，*是* 是分词 2003，等等。

所以，这完全取决于你如何处理这个问题。对于使用小型数据集进行学习，自定义分词器可能就足够了。但一旦你开始处理更大的数据集，你可能希望选择一个预训练的分词器。在这种情况下，你可能必须处理一些开销——所以在本章的其余部分，我将使用自定义代码来分词和预处理文本，而不需要像 BERT 分词器那样的开销。

无论哪种方式，一旦你的句子中的单词被分词，下一步就是将你的句子转换成数字列表，其中数字是单词作为键的值。这个过程被称为*序列化*。

## 将句子转换为序列

现在你已经看到了如何将单词转换为数字，下一步是将句子编码成数字序列，你可以这样做：

```py
def text_to_sequence(text, vocab):
    return [vocab.get(token, 0) for token in tokenize(text)]  
# 0 for unknown words

```

然后，你会得到表示三个句子的序列。记住，单词索引是这样的：

```py
Vocabulary Index: {'today': 1, 'is': 2, 'a': 3, 'sunny': 4, 'day': 5, 'rainy': 6}
```

输出将看起来像这样：

```py
[1, 2, 3, 4, 5]
[1, 2, 3, 6, 5]
```

然后，你可以用数字替换单词，你会发现句子是有意义的。

现在，考虑一下如果你在一组数据上训练神经网络会发生什么。典型的模式是，你有一组用于训练的数据，但你知道它不会覆盖 100%的需求，但你希望尽可能多地覆盖。在 NLP 的情况下，你可能在训练数据中有成千上万的单词，它们在许多不同的上下文中使用，但你不能在每一个可能的上下文中都有每一个可能的单词。所以当你向你的神经网络展示一些新的、之前未见过的文本，其中包含之前未见过的单词时，可能会发生什么？你已经猜到了——网络会感到困惑，因为它根本没有任何这些单词的上下文，因此，它给出的任何预测都会受到负面影响。

### 使用词汇表外的标记

你可以使用一个工具来处理这些情况，那就是一个*词汇表外*（OOV）*标记*，它可以帮助你的神经网络理解包含之前未见过的文本的数据的上下文。例如，给定之前的简小语料库，假设你想要处理这样的句子：

```py
test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]
```

记住，你并不是将这个输入添加到现有的文本语料库中（你可以将其视为你的训练数据），而是考虑一个预训练的网络可能会如何看待这段文本。比如说，你用你已经使用过的单词和现有的分词器来分词，就像这样：

```py
for test_sentence in test_data:
  test_seq = text_to_sequence(test_sentence, vocab)
  print(test_seq)
```

那么，你的结果将看起来像这样：

```py
[1, 2, 3, 0, 5]
[0, 0, 0, 6, 0]
```

因此，新的句子，将标记换回单词，将是“今天是<UNK>的一天”和“<UNK> <UNK> <UNK>雨天<UNK>。”

在这里，我使用标签<UNK>（代表*未知*）来表示标记 0。如果你查看我之前展示的`text_to_sequence`代码，它使用`0`来表示其字典中不存在的单词。当然，你可以使用任何你喜欢的值。

### 理解填充

在训练神经网络时，你通常需要所有数据都具有相同的形状。回想一下，在之前章节中，当你用图像训练时，你会重新格式化图像以具有相同的宽度和高度。对于文本，你面临相同的问题——一旦你将单词分词并将句子转换为序列，它们的长度都可以不同。但为了使它们具有相同的大小和形状，你可以使用*填充*。

我们到目前为止所使用的所有句子都是由五个单词组成的，所以你可以看到我们的序列是五个标记。但是，如果你有一些较长的句子会怎样呢？比如说，一些句子有 5 个单词，一些有 8 个单词，一些有 10 个单词。为了使神经网络处理它们，它们需要具有相同的长度！你可以通过延长较短的句子将所有内容转换为 10 个单词，通过截断较长的句子的一部分将所有内容转换为 5 个单词，或者遵循其他策略！

为了探索填充，让我们将另一个更长的句子添加到语料库中：

```py
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]
```

当你进行序列化时，你会看到你的数字列表有不同的长度。还请注意，如果你没有重新分词来构建新的词汇表，后两个句子将充满零：

```py
[1, 2, 3, 4, 5]
[1, 2, 3, 6, 5]
[2, 0, 4, 0]
[0, 0, 0, 0, 0, 0, 0, 1]
```

所以，别忘了调用：

```py
vocab = build_vocab(sentences)
```

然后，你将为你的分词器中的新单词创建新的标记，输出将看起来像这样：

```py
[1, 2, 3, 4, 5]
[1, 2, 3, 6, 5]
[2, 7, 4, 8]
[9, 10, 11, 12, 13, 14, 15, 1]
```

记住，当你在前几章中训练神经网络时，你的神经网络输入层需要图像具有一致的大小和形状。在 NLP 的大部分情况下也是如此。（对于所谓的*ragged tensors*有一个例外，但这超出了本章的范围。）因此，我们需要一种方法来使我们的句子具有相同的长度。

这里有一个简单的填充函数：

```py
def pad_sequences(sequences, maxlen):
    return [seq + [0] * (maxlen - len(seq)) if len(seq) < maxlen 
            else seq[:maxlen] for seq in sequences]
```

这个函数会将序列中的每个数组重塑为与最大长度相同的长度。所以，假设我们使用以下代码对句子进行序列化后，再进行填充：

```py
for sentence in sentences:
  seq = text_to_sequence(sentence, vocab)
  padded_seq = pad_sequences([seq], maxlen=10)  # Example maxlen
  print(padded_seq)
```

然后，输出将看起来像这样：

```py
[[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]]
[[1, 2, 3, 6, 5, 0, 0, 0, 0, 0]]
[[2, 7, 4, 8, 0, 0, 0, 0, 0, 0]]
[[9, 10, 11, 12, 13, 14, 15, 1, 0, 0]]
```

现在，由于`maxlen`参数，每个序列的长度都是 10。这是一个相当简单的实现，如果你更认真地使用它，你可能会想要在此基础上构建。例如，你可能想要考虑如果你有一个比最大长度更长的序列会发生什么。目前，它会在最大长度之后截断一切，但你可能希望它表现出不同的行为！

还请注意，如果你使用的是像我们之前展示的 BERT 这样的现成分词器，那么这些功能中的许多可能已经可用，所以请确保进行实验。

# 移除停用词和清洗文本

在下一节中，你将查看一些真实世界的数据集，你会发现数据集中往往有一些你*不希望*包含的文本。你可能还希望过滤掉所谓的*停用词*——如“the”、“and”和“but”——它们太常见且不增加任何意义。你也可能遇到很多 HTML 标签，有一个干净的方法来移除它们会很好。你可能还想过滤掉粗俗的词语、标点符号或名字。稍后，我们将探索一个包含用户 ID 的推文数据集，我们希望过滤掉这些 ID。

虽然每个任务都基于你的文本语料库而有所不同，但你可以通过以下三种主要方式来程序化地清理你的文本。

## 移除 HTML 标签

你可以做的第一件事是移除 HTML 标签，幸运的是，有一个名为 BeautifulSoup 的库可以使这个过程变得简单。例如，如果你的句子包含如`<br>`这样的 HTML 标签，你可以通过以下代码来移除它们：

```py
from bs4 import BeautifulSoup
soup = BeautifulSoup(sentence)
sentence = soup.get_text()
```

## 移除停用词

第二件事是移除停用词，一个常见的方法是拥有一个停用词列表，并通过移除停用词的实例来预处理你的句子。以下是一个简化的示例：

```py
stopwords = ["a", "about", "above", ... "yours", "yourself", "yourselves"]
```

你可以在本章的一些[在线示例](https://github.com/lmoroney/PyTorch-Book-FIles)中找到一个完整的停用词列表。

然后，当你遍历你的句子时，你可以使用如下代码来从你的句子中去除停用词：

```py
words = sentence.split()
filtered_sentence = ""
for word in words:
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + " "
sentences.append(filtered_sentence)
```

## 去除标点符号

你可以做的第三件事是去除标点符号，你想要这样做是因为标点符号可能会欺骗停用词过滤器。我们刚刚展示的那个就是寻找被空格包围的单词，所以它不会发现紧随句号或逗号之后的停用词。

使用 Python 字符串库提供的翻译函数可以轻松解决这个问题。但请注意这种方法，因为在某些情况下它可能会影响 NLP 分析，尤其是在检测情感时。

该库还包含一个名为`string.punctuation`的常量，其中包含常用标点符号的列表，因此要从单词中去除它们，你可以这样做：

```py
import string
table = str.maketrans('', '', string.punctuation)
words = sentence.split()
filtered_sentence = ""
for word in words:
    word = word.translate(table)
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + " "
sentences.append(filtered_sentence)
```

在这里，在过滤停用词之前，常量会从句子中的每个单词去除标点符号。所以，如果分割一个句子给你的是单词*it*，这个单词将被转换为*it*，然后作为停用词被去除。然而，请注意，在这样做的时候，你可能需要更新你的停用词列表。这些列表通常包含缩写词和像*you’ll*这样的缩写，翻译器会将*you’ll*转换为*youll*。所以，如果你想过滤掉这些单词，你需要更新你的停用词列表以包括它们。

按照这三个步骤将为你提供一套更干净的文本集。但当然，每个数据集都会有其独特的特性，你需要与之打交道。

# 使用真实数据源

现在你已经看到了获取句子、用词索引编码它们以及排序结果的基础知识，你可以通过使用一些知名的公共数据集并利用 Python 提供的工具将它们转换成易于排序的格式来提升到下一个层次。我们将从一个已经为你做了大量工作的数据集开始：IMDb 数据集。之后，我们将通过处理基于 JSON 的数据集和包含情感数据的几个逗号分隔值（CSV）数据集来获得更多实践经验。

## 获取文本数据集

我们在第四章中探索了一些数据集，所以如果你在这个部分遇到任何概念上的困难，你可以在那里快速复习。然而，在撰写本文时，访问基于文本的数据集有些不寻常。鉴于 torchtext 库已被弃用，其内置数据集的未来走向尚不明确，所以在本节中我们将亲自动手处理原始数据。

我们将首先探索 IMDb 评论数据集，这是一个包含来自互联网电影数据库（IMDb）的 50,000 条标记电影评论的数据集，每条评论都被判定为正面或负面情感。

此代码将下载原始数据集并将其解压缩到文件夹中，其中已经为我们预先制作了训练和测试分割。然后，它们将存储在子目录中，每个子目录中都有名为 `pos` 和 `neg` 的进一步子目录，这些子目录确定了包含的文本文件的标签：

```py
import os
import urllib.request
import tarfile

def download_and_extract(url, destination):
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)
    file_path = os.path.join(destination, "aclImdb_v1.tar.gz")

    if not os.path.exists(file_path):
        print("Downloading the dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")

    if "aclImdb" not in os.listdir(destination):
        print("Extracting the dataset...")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=destination)
        print("Extraction complete.")

# URL for the dataset
dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
download_and_extract(dataset_url, "./data")
```

文件结构将类似于图 5-1 中的结构。

![](img/aiml_0501.png)

###### 图 5-1\. 探索 IMDb 数据集结构

在这个图中，您可以看到 *test/pos* 目录及其中的前几个文件。请注意，这些是文本文件，因此为了创建分词器和词汇表，我们必须读取文件，而不是像早期示例中那样读取内存中的字符串。

让我们来看看为这个自定义分词器编写的代码：

```py
from collections import Counter
import os

# Simple tokenizer
def tokenize(text):
    return text.lower().split()

# Build vocabulary
def build_vocab(path):
    counter = Counter()
    for folder in ["pos", "neg"]:
        folder_path = os.path.join(path, folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', 
                                   encoding='utf-8') as file:
                counter.update(tokenize(file.read()))
    return {word: i+1 for i, word in enumerate(counter)} # Starting index from 1

vocab = build_vocab("./data/aclImdb/train/")
```

这是一段非常直接的代码，它只是遍历每个文件，并将发现的每个新单词添加到词汇表中，为每个单词分配一个新的标记值。通常，您只想对训练数据进行此操作，理解测试数据中会有不在训练数据中的单词，并且它们将使用 OOV（未知标记）进行分词。

输出应该如下所示（已截断）：

```py
{'a': 1, 'year': 2, 'or': 3, 'so': 4, 'ago,': 5, 'i': 6, 'was': 7…
```

这是一个简单的分词器，因为它看到的第一个单词获得第一个标记，第二个获得第二个，依此类推。出于性能原因，通常更好的做法是，语料库中更频繁的单词获得更早的标记，而较少频繁的单词获得较晚的标记。我们将在稍后探讨这一点。

然后，您可以像之前那样进行序列化和填充：

```py
def text_to_sequence(text, vocab):
    return [vocab.get(token, 0) for token in tokenize(text)]  # 0 for unknown

def pad_sequences(sequences, maxlen):
    return [seq + [0] * (maxlen - len(seq)) 
           if len(seq) < maxlen else seq[:maxlen] for seq in sequences]

# Example use
text = "This is an example."
seq = text_to_sequence(text, vocab)
padded_seq = pad_sequences([seq], maxlen=256)  # Example maxlen
print(seq)
```

例如，我们的句子 `This is an example` 将输出为 `[30, 56, 144, 16040]`，因为这些是分配给这些单词的标记。填充的序列将有一个包含 256 个值的张量，其中这些标记作为前 4 个，接下来的 252 个是零！

现在，让我们更新分词器，按频率顺序处理单词。这次更新改变了分词器，以便我们将所有文件加载到内存中，并计算每个单词的实例以获取频率表：

```py
# Build vocabulary
def build_vocab(path):
    counter = Counter()
    for folder in ["pos", "neg"]:
        folder_path = os.path.join(path, folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', 
                                   encoding='utf-8') as file:
                counter.update(tokenize(file.read()))

    # Sort words by frequency in descending order
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # Create vocabulary with indices starting from 1
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words)}
    vocab['<pad>'] = 0  # Add padding token with index 0
    return vocab
```

然后，我们可以输出词汇表，如下所示的频率表。由于词汇表太大，无法显示整个索引，但这里列出了前 20 个单词。请注意，分词器按数据集中单词的频率顺序列出它们，因此像 *the*、*and* 和 *a* 这样的常用词被索引：

```py
{'the': 1, 'a': 2, 'and': 3, 'of': 4, 'to': 5, 'is': 6, 'in': 7, 'i': 8, 
'this': 9, 'that': 10, 'it': 11, '/><br': 12, 'was': 13, 'as': 14,
'for': 15, 'with': 16, 'but': 17, 'on': 18, 'movie': 19, 'his': 20,
```

这些是停用词，如前文所述。这些词的存在可能会影响您的训练精度，因为它们是最常见的词，并且它们没有区分性（即，它们可能存在于正面和负面评论中），因此它们会给我们的训练添加噪声。

还请注意，*br* 包含在这个列表中，因为它在这个语料库中常用作 `<br>` HTML 标签。

您还可以更新代码以使用 `BeautifulSoup` 来删除 HTML 标签，并且您可以从提供的列表中删除停用词。为此，您可以更新分词器，使用 `BeautifulSoup` 来删除 HTML 标签：

```py
# Simple tokenizer
from bs4 import BeautifulSoup

# Note that the list of stopwords is defined in the source code. 
# It’s an array of words. You can define your own or just get the one from 
# the book’s github.

def tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()  # Extract text from HTML
    return [word.lower() for word in cleaned_text.split() if word.lower() 
            not in stopwords]
```

现在，当您打印出您的单词索引时，您将看到这个：

```py
{'movie': 1, 'not': 2, 'film': 3, 'one': 4, 'like': 5, 'just': 6, "it's": 7, 
 'even': 8, 'good': 9, 'no': 10, 'really': 11, 'can': 12, 'see': 13, '-': 14, 
 'get': 15, 'will': 16, 'much': 17, 'story': 18, 'also': 19, 'first': 20
```

你可以看到，这比之前干净多了。然而，总有改进的空间，当我查看完整索引时，我注意到一些不常见的单词在末尾是无意义的。通常，审稿人会合并单词，例如使用破折号（如 *annoying-conclusion*）或斜杠（如 *him/her*），而去除标点符号会错误地将这些合并的单词变成一个单词。或者，如前述代码所示，破折号（*-*）字符足够常见，以至于被标记化。你可以通过将其添加为停用词来去除它。

现在你有了语料库的标记化器，你可以对句子进行编码。例如，我们在本章前面看到的简单句子将输出如下：

```py
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

[[1094, 6112, 246, 0, 0, 0, 0, 0]]
[[1094, 6730, 246, 0, 0, 0, 0, 0]]
[[6112, 25065, 0, 0, 0, 0, 0, 0]]

```

如果你解码这些，你会看到停用词被删除，你得到的是编码为 `today sunny day`、`today rainy day` 和 `sunny today` 的句子。

如果你想在代码中完成这个操作，你可以创建一个新的 `dict`，其中键和值的顺序是相反的（即，对于单词索引中的键/值对，你可以将值作为键，将键作为值），然后从那里进行查找。以下是代码：

```py
reverse_word_index = dict(
    [(value, key) for (key, value) in vocab.items()])

decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in seq])

print(decoded_review)

```

这将给出以下结果：

```py
today sunny day
```

存储标记文本数据的一种常见方式是逗号分隔值（CSV）格式。我们将在下一节中讨论这一点。

## 从 CSV 文件中获取文本

NLP 数据也通常以 CSV 文件格式提供。在接下来的几章中，你将使用我改编自开源 [Sentiment Analysis in Text 数据集](https://oreil.ly/7ZKEU) 的 CSV 数据。该数据集的创建者从 Twitter（现在称为 X）获取了数据。你将使用两个不同的数据集，一个将情感简化为“正面”或“负面”以进行二元分类，另一个使用完整的情感标签范围。这两个数据集使用相同的结构，所以我只展示二元版本。

虽然 *CSV* 这个名字似乎暗示了一个标准文件格式，其中值以逗号分隔，但实际上有各种各样的格式可以被认为是 CSV，而且对任何特定标准的遵守非常少。为了解决这个问题，Python 的 csv 库使得处理 CSV 文件变得简单直接。在这种情况下，数据是按每行两个值存储的。第一个值是一个数字（0 或 1），表示情感是负面还是正面，第二个值是一个包含文本的字符串。

以下代码片段将读取 CSV 并执行类似于我们在上一节中看到的预处理。对于完整的代码，请查看本书的仓库。该代码在复合词的标点符号周围添加空格，使用 `BeautifulSoup` 去除 HTML 内容，然后删除所有标点符号：

```py
import csv
sentences=[]
labels=[]
with open('/tmp/binary-emotion.csv', encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        labels.append(int(row[0]))
        sentence = row[1].lower()
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        soup = BeautifulSoup(sentence)
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence = ""
        for word in words:
            word = word.translate(table)
            if word not in stopwords:
                filtered_sentence = filtered_sentence + word + " "
        sentences.append(filtered_sentence)
```

这将给你一个包含 35,327 个句子的列表。

注意，此代码针对特定数据。它的目的是帮助你了解你可能需要执行的任务类型，以便使某些东西工作，并不旨在成为你必须为每个任务执行的所有事情的详尽列表——因此你的里程可能会有所不同。

### 创建训练集和测试集

现在文本语料库已经被读入一个句子列表中，你需要将其分割成训练集和测试集以训练模型。例如，如果你想用 28,000 个句子进行训练，其余的保留用于测试，你可以使用如下代码：

```py
training_size = 28000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
```

现在你有了训练集，你可以编辑分词器和词汇构建器，从这个语料库中创建单词索引。由于语料库是一个字符串的内存数组（`training_sentences`），这个过程要简单得多：

```py
from collections import Counter

# Assuming the tokenize function is defined elsewhere
def tokenize(text):
    # Tokenization logic, removing HTML and stopwords as discussed earlier
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    tokens = cleaned_text.lower().split()
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens

def build_vocab(sentences):
    counter = Counter()
    for text in sentences:
        counter.update(tokenize(text))

    # Sort words by frequency in descending order
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # Create vocabulary with indices starting from 1
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words)}
    vocab['<pad>'] = 0  # Add padding token with index 0
    return vocab

vocab = build_vocab(training_sentences)
print(vocab)
```

你可以使用相同的辅助函数将文本转换为序列，然后进行填充，如下所示：

```py
print(testing_sentences[1])
seq = text_to_sequence(testing_sentences[1], vocab)
print(seq)
```

结果将如下所示：

```py
made many new friends twitter around usa another bike across usa trip amazing 
 see people
[146, 259, 30, 110, 53, 198, 2161, 111, 752, 970, 2161, 407, 217, 26, 73]
```

另一种常见的结构化数据格式，尤其是在响应网络调用时，是 JavaScript 对象表示法（JSON）。我们将探讨如何读取 JSON 数据。

## 从 JSON 文件中获取文本

JSON 是一个开放标准文件格式，常用于数据交换，尤其是在网络应用程序中。它是可读的，并设计为使用名称/值对，因此特别适合用于标记文本。在 Kaggle 数据集中搜索 JSON 会得到超过 2,500 个结果。例如，流行的数据集，如斯坦福问答数据集（SQuAD），就是存储在 JSON 中的。

JSON 具有非常简单的语法，其中对象包含在大括号中，作为名称/值对，每个对之间用逗号分隔。例如，表示我名字的 JSON 对象如下所示：

```py
{"firstName" : "Laurence",
 "lastName" : "Moroney"}
```

JSON 还支持数组，这与 Python 列表非常相似，并且用方括号语法表示。以下是一个示例：

```py
[
 {"firstName" : "Laurence",
 "lastName" : "Moroney"},
 {"firstName" : "Sharon",
 "lastName" : "Agathon"}
]
```

对象也可以包含数组，因此这是完全有效的 JSON：

```py
[
 {"firstName" : "Laurence",
 "lastName" : "Moroney",
 "emails": ["lmoroney@gmail.com", "lmoroney@galactica.net"]
 },
 {"firstName" : "Sharon",
 "lastName" : "Agathon",
 "emails": ["sharon@galactica.net", "boomer@cylon.org"]
 }
]
```

一个存储在 JSON 中且非常有趣的数据集是“用于讽刺检测的新闻标题数据集”，由[Rishabh Misra](https://oreil.ly/wZ3oD)创建，可在[Kaggle](https://oreil.ly/_AScB)上找到。这个数据集收集了两个来源的新闻标题：*The Onion*用于搞笑或讽刺的标题，而*HuffPost*用于普通标题。

讽刺数据集的文件结构非常简单：

```py
{"is_sarcastic": 1 or 0, 
 "headline": String containing headline, 
 "article_link": String Containing link}
```

该数据集包含大约 26,000 个项目，每个项目一行。为了在 Python 中使其更易于阅读，我创建了一个版本，将这些项目包含在一个数组中，这样数据集就可以作为一个单独的列表读取，该列表用于本章的源代码。

### 读取 JSON 文件

Python 的 json 库使得读取 JSON 文件变得简单。鉴于 JSON 使用名称/值对，你可以根据名称进行索引。因此，例如，对于讽刺数据集，你可以创建一个指向 JSON 文件的文件句柄，使用`json`库打开它，然后让一个可迭代对象逐行读取，通过字段名称获取数据项。

这是代码：

```py
import json
with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)
    for item in datastore:
        sentence = item['headline'].lower()
        label= item['is_sarcastic']
        link = item['article_link']
```

这使得你可以轻松地创建句子和标签的列表，就像你在本章中做的那样，然后对句子进行标记化。你还可以在阅读句子时即时进行预处理，去除停用词、HTML 标签、标点符号等。

这是创建句子、标签和 URL 列表的完整代码，同时确保句子被清理掉不需要的单词和字符：

```py
with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = [] 
labels = []
urls = []
for item in datastore:
    sentence = item['headline'].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    sentences.append(filtered_sentence)
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
```

如前所述，你可以将这些数据分成训练集和测试集。如果你想使用数据集中的 23,000 个中的 26,000 个项进行训练，你可以这样做：

```py
training_size = 23000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
```

现在你已经将它们作为`in_memory`字符串数组，对它们进行标记化和序列化将完全与对讽刺数据集进行标记化和序列化的方式相同。

希望类似外观的代码能帮助你看到在为神经网络进行分类或生成文本准备时可以遵循的模式。在下一章中，你将学习如何使用嵌入构建文本分类器，而在第七章中，你将通过探索循环神经网络将这一步更进一步。然后，在第第八章中，你将学习如何进一步增强序列数据以创建一个能够生成新文本的神经网络！

###### 小贴士

正则表达式（也称为 Regex）是排序、过滤和清理文本的出色工具。它们的语法通常很难理解，学习起来也很困难，但我发现 Gemini、Claude 和 ChatGPT 等生成式 AI 工具在这里非常有用。

# 摘要

在前面的章节中，你使用了图像来构建分类器。根据定义，图像具有明确的维度——你知道它们的宽度和高度以及格式。另一方面，文本处理起来可能要困难得多。它通常是未结构化的，可能包含不希望的内容，如格式说明，不一定包含你想要的内容，并且通常需要过滤以去除无意义或不相关的内容。

在本章中，你看到了如何使用词标记化将文本转换为数字，然后探讨了如何以各种格式读取和过滤文本。掌握这些技能后，你现在可以迈出下一步，学习如何从单词中推断出*意义*——这是理解自然语言的第一个步骤。
