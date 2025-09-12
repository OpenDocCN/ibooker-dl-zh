# 第十章：训练 Transformer 以将英语翻译成法语

本章涵盖

+   将英语和法语短语分词为子词

+   理解词嵌入和位置编码

+   从头开始训练 Transformer 以将英语翻译成法语

+   使用训练好的 Transformer 将英语短语翻译成法语

在上一章中，我们从头开始构建了一个 Transformer，可以根据“Attention Is All You Need.”这篇论文在任意两种语言之间进行翻译。具体来说，我们实现了自注意力机制，使用查询、键和值向量来计算缩放点积注意力（SDPA）。

为了更深入地理解自注意力机制和 Transformer，我们将在本章中使用英语到法语翻译作为案例研究。通过探索将英语句子转换为法语的模型训练过程，你将深入了解 Transformer 的架构和注意力机制的运作。

想象一下，你已经收集了超过 47,000 个英语到法语翻译对。你的目标是使用这个数据集训练第九章中的编码器-解码器 Transformer。本章将带你了解项目的所有阶段。首先，你将使用子词分词将英语和法语短语分解成标记。然后，你将构建你的英语和法语词汇表，其中包含每种语言中所有唯一的标记。词汇表允许你将英语和法语短语表示为索引序列。之后，你将使用词嵌入将这些索引（本质上是一维向量）转换为紧凑的向量表示。我们将在词嵌入中添加位置编码以形成输入嵌入。位置编码允许 Transformer 知道序列中标记的顺序。

最后，你将训练第九章中的编码器-解码器 Transformer，使用英语到法语翻译的集合作为训练数据集，将英语翻译成法语。训练完成后，你将学会使用训练好的 Transformer 将常见的英语短语翻译成法语。具体来说，你将使用编码器来捕捉英语短语的含义。然后，你将使用训练好的 Transformer 中的解码器以自回归的方式生成法语翻译，从开始标记`"BOS"`开始。在每一步中，解码器根据之前生成的标记和解码器的输出生成最可能的下一个标记，直到预测的标记是`"EOS"`，这标志着句子的结束。训练好的模型可以像使用谷歌翻译进行任务一样，准确地将常见的英语短语翻译成法语。

## 10.1 子词分词

正如我们在第八章中讨论的，有三种标记化方法：字符级标记化、词级标记化和子词标记化。在本章中，我们将使用子词标记化，它在其他两种方法之间取得平衡。它在词汇表中保留了常用词的完整性，并将不常见或更复杂的词分解成子组件。

在本节中，你将学习如何将英文和法语短语标记为子词。然后，你将创建将标记映射到索引的字典。训练数据随后被转换为索引序列，并放入批量中进行训练。

### 10.1.1 标记英文和法语短语

访问[`mng.bz/WVAw`](https://mng.bz/WVAw)下载包含我从各种来源收集的英文到法语翻译的 zip 文件。解压文件并将 en2fr.csv 放在计算机上的/files/文件夹中。

我们将加载数据并打印出一句英文短语及其法语翻译，如下所示：

```py
import pandas as pd

df=pd.read_csv("files/en2fr.csv")                                ①
num_examples=len(df)                                             ②
print(f"there are {num_examples} examples in the training data")
print(df.iloc[30856]["en"])                                      ③
print(df.iloc[30856]["fr"])                                      ④
```

① 加载 CSV 文件

② 计算数据中有多少对短语

③ 打印出一个英文短语的示例

④ 打印出相应的法语翻译

前面代码片段的输出如下

```py
there are 47173 examples in the training data
How are you?
Comment êtes-vous?
```

训练数据中有 47,173 对英文到法语的翻译。我们已打印出英文短语“你好吗？”及其对应的法语翻译“Comment êtes-vous？”作为示例。

在这个 Jupyter Notebook 的新单元格中运行以下代码行，以在您的计算机上安装`transformers`库：

```py
!pip install transformers
```

接下来，我们将对数据集中的英文和法语短语进行标记。我们将使用 Hugging Face 的预训练 XLM 模型作为标记器，因为它擅长处理多种语言，包括英文和法语短语。

列表 10.1 预训练标记器

```py
from transformers import XLMTokenizer                           ①

tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")

tokenized_en=tokenizer.tokenize("I don't speak French.")        ②
print(tokenized_en)
tokenized_fr=tokenizer.tokenize("Je ne parle pas français.")    ③
print(tokenized_fr)
print(tokenizer.tokenize("How are you?"))
print(tokenizer.tokenize("Comment êtes-vous?"))
```

① 导入预训练标记器

② 使用标记器对英文句子进行标记

③ 标记一个法语句子

列表 10.1 的输出如下

```py
['i</w>', 'don</w>', "'t</w>", 'speak</w>', 'fr', 'ench</w>', '.</w>']
['je</w>', 'ne</w>', 'parle</w>', 'pas</w>', 'franc', 'ais</w>', '.</w>']
['how</w>', 'are</w>', 'you</w>', '?</w>']
['comment</w>', 'et', 'es-vous</w>', '?</w>']
```

在前面的代码块中，我们使用 XLM 模型预训练的标记器将英文句子“我不说法语。”分解成一组标记。在第八章中，你开发了一个自定义的词级标记器。然而，本章介绍了使用更高效的预训练子词标记器，其有效性超过了词级标记器。因此，句子“我不说法语。”被标记为`['i', 'don', "'t", 'speak', 'fr', 'ench', '.']`。同样，法语句子“Je ne parle pas français.”被分割成六个标记：`['je', 'ne', 'parle', 'pas', 'franc', 'ais', '.']`。我们还将英文短语“你好吗？”及其法语翻译进行了标记。结果显示在上面的输出最后两行。

注意：你可能已经注意到，XLM 模型使用 '`</w>`' 作为标记分隔符，除非两个标记是同一个单词的一部分。子词标记化通常导致每个标记要么是一个完整的单词或标点符号，但有时一个单词会被分成音节。例如，单词“French”被分成“fr”和“ench。”值得注意的是，模型在“fr”和“ench”之间不会插入 `</w>`，因为这些音节共同构成了单词“French”。

深度学习模型如 Transformers 不能直接处理原始文本；因此，在将文本输入模型之前，我们需要将文本转换为数值表示。为此，我们创建一个字典，将所有英语标记映射到整数。

列表 10.2 将英语标记映射到索引

```py
from collections import Counter

en=df["en"].tolist()                                           ①

en_tokens=[["BOS"]+tokenizer.tokenize(x)+["EOS"] for x in en]  ②
PAD=0
UNK=1
word_count=Counter()
for sentence in en_tokens:
    for word in sentence:
        word_count[word]+=1
frequency=word_count.most_common(50000)                        ③
total_en_words=len(frequency)+2
en_word_dict={w[0]:idx+2 for idx,w in enumerate(frequency)}    ④
en_word_dict["PAD"]=PAD
en_word_dict["UNK"]=UNK
en_idx_dict={v:k for k,v in en_word_dict.items()}              ⑤
```

① 从训练数据集中获取所有英语句子

② 对所有英语句子进行标记化

③ 计算标记的频率

④ 创建一个字典将标记映射到索引

⑤ 创建一个字典将索引映射到标记

我们分别在每句话的开始和结束处插入标记 `"BOS"`（句子开始）和 `"EOS"`（句子结束）。字典 `en_word_dict` 为每个标记分配一个唯一的整数值。此外，用于填充的 `"PAD"` 标记被分配整数 0，而代表未知标记的 `"UNK"` 标记被分配整数 1。反向字典 `en_idx_dict` 将整数（索引）映射回相应的标记。这种反向映射对于将整数序列转换回标记序列至关重要，使我们能够重建原始的英语短语。

使用字典 `en_word_dict`，我们可以将英语句子“我不说法语。”转换为其数值表示。这个过程涉及在字典中查找每个标记以找到其对应的整数值。例如：

```py
enidx=[en_word_dict.get(i,UNK) for i in tokenized_en]   
print(enidx)
```

上述代码行产生以下输出：

```py
[15, 100, 38, 377, 476, 574, 5]
```

这意味着英语句子“我不说法语。”现在由一系列整数[15, 100, 38, 377, 476, 574, 5]表示。

我们还可以使用字典 `en_idx_dict` 将数值表示转换回标记。这个过程涉及将数值序列中的每个整数映射回字典中定义的相应标记。以下是操作方法：

```py
entokens=[en_idx_dict.get(i,"UNK") for i in enidx]          ①
print(entokens)
en_phrase="".join(entokens)                                 ②
en_phrase=en_phrase.replace("</w>"," ")                     ③
for x in '''?:;.,'("-!&)%''':
    en_phrase=en_phrase.replace(f" {x}",f"{x}")             ④
print(en_phrase)
```

① 将索引转换为标记

② 将标记连接成一个字符串

③ 将分隔符替换为空格

④ 删除标点符号前的空格

上述代码片段的输出是

```py
['i</w>', 'don</w>', "'t</w>", 'speak</w>', 'fr', 'ench</w>', '.</w>']
i don't speak french. 
```

字典`en_idx_dict`用于将数字转换回它们原始的标记。在此之后，这些标记被转换成完整的英语短语。这是通过首先将标记连接成一个字符串，然后将分隔符`''</w>''`替换为空格来完成的。我们还移除了标点符号前的空格。请注意，恢复的英语短语全部为小写字母，因为预训练的标记器自动将大写字母转换为小写以减少唯一标记的数量。正如你将在下一章中看到的，一些模型，如 GPT2 和 ChatGPT，并不这样做；因此，它们的词汇量更大。

练习 10.1

在列表 10.1 中，我们将句子“你好？”分解成了标记`['how</w>', 'are</w>', 'you</w>', '?</w>']`。按照本小节中的步骤进行操作，(i) 使用字典`en_word_dict`将标记转换为索引；(ii) 使用字典`en_idx_dict`将索引转换回标记；(iii) 通过将标记连接成一个字符串，将分隔符`'</w>'`改为空格，并移除标点符号前的空格来恢复英语句子。

我们可以将相同的步骤应用于法语短语，将标记映射到索引，反之亦然。

列表 10.3 将法语标记映射到索引

```py
fr=df["fr"].tolist()       
fr_tokens=[["BOS"]+tokenizer.tokenize(x)+["EOS"] for x in fr]  ①
word_count=Counter()
for sentence in fr_tokens:
    for word in sentence:
        word_count[word]+=1
frequency=word_count.most_common(50000)                        ②
total_fr_words=len(frequency)+2
fr_word_dict={w[0]:idx+2 for idx,w in enumerate(frequency)}    ③
fr_word_dict["PAD"]=PAD
fr_word_dict["UNK"]=UNK
fr_idx_dict={v:k for k,v in fr_word_dict.items()}              ④
```

① 将所有法语句子进行标记化

② 统计法语标记的频率

③ 创建一个将法语标记映射到索引的字典

④ 创建一个将索引映射到法语标记的字典

字典`fr_word_dict`为每个法语标记分配一个整数，而`fr_idx_dict`将这些整数映射回它们相应的法语标记。接下来，我将演示如何将法语短语“Je ne parle pas français.”转换成其数值表示：

```py
fridx=[fr_word_dict.get(i,UNK) for i in tokenized_fr]   
print(fridx)
```

前一个代码片段的输出结果是

```py
[28, 40, 231, 32, 726, 370, 4]
```

法语短语“Je ne parle pas français.”的标记被转换成一系列整数，如下所示。

我们可以使用字典`fr_idx_dict`将数值表示转换回法语标记。这涉及到将序列中的每个数字转换回字典中相应的法语标记。一旦检索到标记，它们就可以被连接起来以重建原始的法语短语。以下是完成方式：

```py
frtokens=[fr_idx_dict.get(i,"UNK") for i in fridx] 
print(frtokens)
fr_phrase="".join(frtokens)
fr_phrase=fr_phrase.replace("</w>"," ")
for x in '''?:;.,'("-!&)%''':
    fr_phrase=fr_phrase.replace(f" {x}",f"{x}")  
print(fr_phrase)
```

前一个代码块输出的结果是

```py
['je</w>', 'ne</w>', 'parle</w>', 'pas</w>', 'franc', 'ais</w>', '.</w>']
je ne parle pas francais. 
```

重要的是要认识到恢复的法语短语并不完全匹配其原始形式。这种差异是由于标记化过程，该过程将所有大写字母转换为小写，并消除了法语中的重音符号。

练习 10.2

在列表 10.1 中，我们将句子“Comment êtes-vous?”分解为标记`['comment</w>', 'et', 'es-vous</w>', '?</w>']`。按照本小节中的步骤进行以下操作：(i) 使用字典`fr_word_dict`将标记转换为索引；(ii) 使用字典`fr_idx_dict`将索引转换回标记；(iii) 通过将标记连接成一个字符串来恢复法语文句，将分隔符`'</w>'`更改为空格，并删除标点符号前的空格。

将四个字典保存在您的计算机上的文件夹/files/中，以便您可以加载它们并在稍后开始翻译，而无需担心首先将标记映射到索引，反之亦然：

```py
import pickle

with open("files/dict.p","wb") as fb:
    pickle.dump((en_word_dict,en_idx_dict,
                 fr_word_dict,fr_idx_dict),fb)
```

现在四个字典已保存为单个 pickle 文件`dict.p`。或者，您也可以从本书的 GitHub 仓库下载该文件。

### 10.1.2 序列填充和批次创建

我们将在训练期间将训练数据划分为批次以提高计算效率和加速收敛，正如我们在前面的章节中所做的那样。

为其他数据格式（如图像）创建批次是直接的：只需将特定数量的输入分组形成一个批次，因为它们都具有相同的大小。然而，在自然语言处理中，由于句子长度的不同，批次可能会更复杂。为了在批次内标准化长度，我们填充较短的序列。这种一致性至关重要，因为输入到 Transformer 中的数值表示需要具有相同的长度。例如，一个批次中的英文短语长度可能不同（这也可能发生在批次中的法语文句）。为了解决这个问题，我们在批次中较短的短语的数值表示的末尾添加零，确保所有输入到 Transformer 模型中的长度都相等。

注意，在机器翻译中，在每句话的开始和结尾加入`BOS`和`EOS`标记，以及在一个批次中对较短的序列进行填充，这是一个显著的特征。这种区别源于输入由整个句子或短语组成。相比之下，正如你将在下一章中看到的，训练一个文本生成模型不需要这些过程；模型的输入包含一个预定的标记数量。

我们首先将所有英文短语转换为它们的数值表示，然后对法语文句应用相同的过程：

```py
out_en_ids=[[en_word_dict.get(w,UNK) for w in s] for s in en_tokens]
out_fr_ids=[[fr_word_dict.get(w,UNK) for w in s] for s in fr_tokens]
sorted_ids=sorted(range(len(out_en_ids)),
                  key=lambda x:len(out_en_ids[x]))
out_en_ids=[out_en_ids[x] for x in sorted_ids]
out_fr_ids=[out_fr_ids[x] for x in sorted_ids]
```

接下来，我们将数值表示放入批次进行训练：

```py
import numpy as np

batch_size=128
idx_list=np.arange(0,len(en_tokens),batch_size)
np.random.shuffle(idx_list)

batch_indexs=[]
for idx in idx_list:
    batch_indexs.append(np.arange(idx,min(len(en_tokens),
                                          idx+batch_size)))
```

注意，我们在将观察结果放入批次之前，已经根据英文短语的长度对训练数据集中的观察结果进行了排序。这种方法确保了每个批次中的观察结果具有可比的长度，从而减少了填充的需要。因此，这种方法不仅减少了训练数据的总体大小，还加速了训练过程。

为了将批次中的序列填充到相同的长度，我们定义了以下函数：

```py
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)                                                   ①
    padded_seq = np.array([np.concatenate([x, [padding] * (ML - len(x))])
        if len(x) < ML else x for x in X])                        ②
    return padded_seq
```

① 找出批次中最长序列的长度。

② 如果批次比最长的序列短，则在序列末尾添加 0。

函数 `seq_padding()` 首先在批次中识别最长的序列。然后，它将零添加到较短的序列的末尾，以确保批次中的每个序列都与这个最大长度匹配。

为了节省空间，我们在本地模块 ch09util.py 中创建了一个 `Batch()` 类，您在上一个章节中已下载（见图 10.1）。

列表 10.4 在本地模块中创建一个 `B`atc`h()` 类

```py
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class Batch:
    def __init__(self, src, trg=None, pad=0):
        src = torch.from_numpy(src).to(DEVICE).long()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)              ①
        if trg is not None:
            trg = torch.from_numpy(trg).to(DEVICE).long()
            self.trg = trg[:, :-1]                              ②
            self.trg_y = trg[:, 1:]                             ③
            self.trg_mask = make_std_mask(self.trg, pad)        ④
            self.ntokens = (self.trg_y != pad).data.sum()
```

① 创建一个源掩码以隐藏句子末尾的填充

② 为解码器创建输入

③ 将输入向右移动一个标记并将其用作输出

④ 创建一个目标掩码

![](img/CH10_F01_Liu.png)

图 10.1 `Batch()` 类的作用是什么？`Batch()` 类接受两个输入：`src` 和 `trg`，分别代表源语言和目标语言的索引序列。它向训练数据添加了几个属性：`src_mask`，用于隐藏填充的源掩码；`modified trg`，解码器的输入；`trg_y`，解码器的输出；`trg_mask`，用于隐藏填充和未来标记的目标掩码。

`Batch()` 类处理一批英语和法语短语，将它们转换为适合训练的格式。为了使这个解释更具体，以英语短语“How are you?”及其法语对应短语“Comment êtes-vous?”为例。`Batch()` 类接收两个输入：`src`，代表“How are you?”中标记的索引序列，以及 `trg`，代表“Comment êtes-vous?”中标记的索引序列。这个类生成一个张量 `src_mask`，用于隐藏句子末尾的填充。例如，句子“How are you?”被分解成六个标记：`['BOS', 'how', 'are', 'you', '?', 'EOS']`。如果这个序列是长度为八个标记的批次的一部分，则在末尾添加两个零。`src_mask` 张量指示模型在这种情况下忽略最后的两个标记。

`Batch()` 类还准备了 Transformer 解码器的输入和输出。以法语短语“Comment êtes-vous?”为例，它被转换成六个标记：`['BOS', 'comment', 'et', 'es-vous', '?', 'EOS']`。这些前五个标记的索引作为解码器的输入，命名为 `trg`。接下来，我们将这个输入向右移动一个标记以形成解码器的输出，`trg_y`。因此，输入包含 `['BOS', 'comment', 'et', 'es-vous', '?']` 的索引，而输出则包含 `['comment', 'et', 'es-vous', '?', 'EOS']` 的索引。这种方法与我们第八章讨论的内容相似，旨在迫使模型根据前面的标记预测下一个标记。

`Batch()` 类还生成了一个用于解码器输入的掩码，`trg_mask`。这个掩码的目的是隐藏输入中的后续标记，确保模型仅依赖于先前标记进行预测。这个掩码是由 `make_std_mask()` 函数生成的，该函数定义在本地模块 ch09util 中：

```py
import numpy as np
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    output = torch.from_numpy(subsequent_mask) == 0
    return output
def make_std_mask(tgt, pad):
    tgt_mask=(tgt != pad).unsqueeze(-2)
    output=tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return output 
```

`subsequent_mask()` 函数为序列生成一个特定的掩码，指示模型仅关注实际序列，并忽略末尾的填充零，这些填充零仅用于标准化序列长度。另一方面，`make_std_mask()` 函数构建了一个针对目标序列的标准掩码。这个标准掩码具有双重作用，即隐藏填充零和目标序列中的后续标记。

接下来，我们导入 `Batch()` 类从本地模块，并使用它来创建训练数据批次：

```py
from utils.ch09util import Batch

class BatchLoader():
    def __init__(self):
        self.idx=0
    def __iter__(self):
        return self
    def __next__(self):
        self.idx += 1
        if self.idx<=len(batch_indexs):
            b=batch_indexs[self.idx-1]
            batch_en=[out_en_ids[x] for x in b]
            batch_fr=[out_fr_ids[x] for x in b]
            batch_en=seq_padding(batch_en)
            batch_fr=seq_padding(batch_fr)
            return Batch(batch_en,batch_fr)
        raise StopIteration
```

`BatchLoader()` 类创建用于训练的数据批次。列表中的每个批次包含 128 对，其中每对包含一个英语短语及其对应的法语翻译的数值表示。

## 10.2 词嵌入和位置编码

在上一节进行分词之后，英语和法语短语被表示为一系列的索引。在本节中，您将使用词嵌入将这些索引（本质上是一维热向量）转换为紧凑的向量表示。这样做可以捕捉短语中标记的语义信息和相互关系。词嵌入还可以提高训练效率：与庞大的热向量相比，词嵌入使用连续的、低维向量来减少模型的复杂性和维度。

注意力机制同时处理短语中的所有标记，而不是按顺序处理。这提高了其效率，但本身并不允许它识别标记的序列顺序。因此，我们将通过使用不同频率的正弦和余弦函数，将位置编码添加到输入嵌入中。

### 10.2.1 词嵌入

英语和法语短语的数值表示涉及大量的索引。为了确定每种语言所需的唯一索引的确切数量，我们可以计算 `en_word_dict` 和 `fr_word_dict` 字典中唯一元素的数量。这样做会生成每种语言词汇表中唯一标记的总数（我们将在后面将它们作为 Transformer 的输入使用）：

```py
src_vocab = len(en_word_dict)
tgt_vocab = len(fr_word_dict)
print(f"there are {src_vocab} distinct English tokens")
print(f"there are {tgt_vocab} distinct French tokens")
```

输出是

```py
there are 11055 distinct English tokens
there are 11239 distinct French tokens
```

在我们的数据集中，有 11,055 个唯一的英语标记和 11,239 个唯一的法语标记。对这些使用一维热编码会导致训练时参数数量过高。为了解决这个问题，我们将采用词嵌入，它将数值表示压缩成连续的向量，每个向量的长度为 `d_model = 256`。

这是通过使用定义在本地模块 ch09util 中的 `Embeddings()` 类来实现的：

```py
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        out = self.lut(x) * math.sqrt(self.d_model)
        return out
```

之前定义的 `Embeddings()` 类使用了 PyTorch 的 `Embedding()` 类。它还将输出乘以 `d_model` 的平方根，即 256。这种乘法是为了抵消在计算注意力分数过程中发生的除以 `d_model` 的平方根。`Embeddings()` 类降低了英语和法语短语的数值表示的维度。我们在第八章详细讨论了 PyTorch 的 `Embedding()` 类是如何工作的。

### 10.2.2 位置编码

为了准确表示输入和输出中元素序列的顺序，我们在本地模块中引入了 `PositionalEncoding()` 类。

列表 10.5 计算位置编码的类

```py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):       ①
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0., max_len, 
                                device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0., d_model, 2, device=DEVICE)
            * -(math.log(10000.0) / d_model))
        pe_pos = torch.mul(position, div_term)
        pe[:, 0::2] = torch.sin(pe_pos)                        ②
        pe[:, 1::2] = torch.cos(pe_pos)                        ③
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  

    def forward(self, x):
        x=x+self.pe[:,:x.size(1)].requires_grad_(False)        ④
        out=self.dropout(x)
        return out
```

① 初始化类，允许最大 5,000 个位置

② 将正弦函数应用于向量的偶数索引

③ 将余弦函数应用于向量的奇数索引

④ 将位置编码添加到词嵌入中

`PositionalEncoding()` 类使用正弦函数对偶数索引进行编码，使用余弦函数对奇数索引进行编码来生成序列位置的向量。需要注意的是，在 `PositionalEncoding()` 类中，包含了 `requires_grad_(False)` 参数，因为这些值不需要进行训练。它们在所有输入中保持不变，并且在训练过程中不会改变。

例如，来自英语短语 `['BOS', 'how', 'are', 'you', '?', 'EOS']` 的六个标记的索引首先通过一个词嵌入层进行处理。这一步将这些索引转换为一个维度为 (1, 6, 256) 的张量：1 表示批处理中只有一个序列；6 表示序列中有 6 个标记；256 表示每个标记由一个 256 价值的向量表示。在完成词嵌入过程后，`PositionalEncoding()` 类被用来计算对应于标记 `['BOS', 'how', 'are', 'you', '?', 'EOS']` 的索引的位置编码。这样做是为了向模型提供有关每个标记在序列中位置的信息。更好的是，我们可以通过以下代码块告诉你前六个标记的位置编码的确切值：

```py
from utils.ch09util import PositionalEncoding
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pe = PositionalEncoding(256, 0.1)                              ①
x = torch.zeros(1, 8, 256).to(DEVICE)                          ②
y = pe.forward(x)                                              ③
print(f"the shape of positional encoding is {y.shape}")
print(y)                                                       ④
```

① 实例化 PositionalEncoding() 类并将模型维度设置为 256

② 创建一个词嵌入并将其填充为零

③ 通过向词嵌入添加位置编码来计算输入嵌入

④ 打印出输入嵌入，由于词嵌入被设置为零，因此与位置编码相同

我们首先创建一个 `PositionalEncoding()` 类的实例 `pe`，将模型维度设置为 256，并将 dropout 率设置为 0.1。由于这个类的输出是词嵌入和位置编码的和，我们创建一个填充为零的词嵌入并将其输入到 `pe` 中：这样输出就是位置编码。

运行前面的代码块后，你会看到以下输出：

```py
the shape of positional encoding is torch.Size([1, 8, 256])
tensor([[[ 0.0000e+00,  1.1111e+00,  0.0000e+00,  ...,  0.0000e+00,
           0.0000e+00,  1.1111e+00],
         [ 9.3497e-01,  6.0034e-01,  8.9107e-01,  ...,  1.1111e+00,
           1.1940e-04,  1.1111e+00],
         [ 0.0000e+00, -4.6239e-01,  1.0646e+00,  ...,  1.1111e+00,
           2.3880e-04,  1.1111e+00],
         ...,
         [-1.0655e+00,  3.1518e-01, -1.1091e+00,  ...,  1.1111e+00,
           5.9700e-04,  1.1111e+00],
         [-3.1046e-01,  1.0669e+00, -0.0000e+00,  ...,  0.0000e+00,
           7.1640e-04,  1.1111e+00],
         [ 7.2999e-01,  8.3767e-01,  2.5419e-01,  ...,  1.1111e+00,
           8.3581e-04,  1.1111e+00]]], device='cuda:0')
```

前面的张量表示了英语短语“你好吗？”的位置编码。重要的是要注意，这个位置编码也有(1, 6, 256)的维度，这与“你好吗？”的词嵌入大小相匹配。下一步是将词嵌入和位置编码组合成一个单独的张量。

位置编码的一个基本特征是，无论输入序列是什么，它们的值都是相同的。这意味着无论具体的输入序列是什么，第一个标记的位置编码始终是相同的 256 值向量，如上输出所示，即`[0.0000e+00, 1.1111e+00, ..., 1.1111e+00]`。同样，第二个标记的位置编码始终是`[9.3497e-01, 6.0034e-01, ..., 1.1111e+00]`，依此类推。它们的值在训练过程中也不会改变。

## 10.3 训练英语到法语翻译的 Transformer

我们构建的英语到法语翻译模型可以被视为一个多类别分类器。核心目标是预测在翻译英语句子时法语词汇中的下一个标记。这与我们在第二章讨论的图像分类项目有些相似，尽管这个模型要复杂得多。这种复杂性需要仔细选择损失函数、优化器和训练循环参数。

在本节中，我们将详细说明选择合适的损失函数和优化器的过程。我们将使用英语到法语翻译的批次作为我们的训练数据集来训练 Transformer。在模型训练完成后，你将学习如何将常见的英语短语翻译成法语。

### 10.3.1 损失函数和优化器

首先，我们从本地模块 ch09util.py 中导入`create_model()`函数，构建一个 Transformer，以便我们可以训练它将英语翻译成法语：

```py
from utils.ch09util import create_model

model = create_model(src_vocab, tgt_vocab, N=6,
    d_model=256, d_ff=1024, h=8, dropout=0.1)
```

论文“Attention Is All You Need”在构建模型时使用了各种超参数的组合。在这里，我们选择了一个维度为 256，8 个头的模型，因为我们发现这个组合在我们的设置中在将英语翻译成法语方面做得很好。感兴趣的读者可以使用验证集来调整超参数，以选择他们自己项目中的最佳模型。

我们将遵循原始论文“Attention Is All You Need”并在训练过程中使用标签平滑。标签平滑通常在训练深度神经网络时使用，以提高模型的泛化能力。它用于解决过自信问题（预测概率大于真实概率）和分类中的过拟合问题。具体来说，它通过调整目标标签来修改模型的学习方式，旨在降低模型对训练数据的信心，这可能导致在未见数据上的更好性能。

在一个典型的分类任务中，目标标签以单热编码格式表示。这种表示意味着对每个训练样本标签正确性的绝对确定性。使用绝对确定性进行训练可能导致两个主要问题。第一个是过拟合：模型对其预测过于自信，过于紧密地拟合训练数据，这可能会损害其在新、未见数据上的性能。第二个问题是校准不良：以这种方式训练的模型通常输出过自信的概率。例如，它们可能会为正确类别输出 99%的概率，而实际上，这种信心应该更低。

标签平滑调整目标标签以降低其信心。对于一个三分类问题，你可能会得到类似`[0.9, 0.05, 0.05]`的目标标签。这种方法通过惩罚过自信的输出，鼓励模型不要对其预测过于自信。平滑后的标签是原始标签和一些其他标签分布（通常是均匀分布）的混合。

我们在本地模块 ch09util 中定义了以下`LabelSmoothing()`类。

列表 10.6 用于执行标签平滑的类

```py
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')  
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()                              ①
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, 
               target.data.unsqueeze(1), self.confidence)       ②
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        output = self.criterion(x, true_dist.clone().detach())  ③
        return output
```

① 从模型中提取预测值

② 从训练数据中提取实际标签并向其添加噪声

③ 在计算损失时使用平滑后的标签作为目标

`LabelSmoothing()`类首先从模型中提取预测值。然后，它通过添加噪声来平滑训练数据集中的实际标签。参数`smoothing`控制我们向实际标签注入多少噪声。例如，如果你设置`smoothing=0.1`，标签`[1, 0, 0]`将被平滑为`[0.9, 0.05, 0.05]`；如果你设置`smoothing=0.05`，它将被平滑为`[0.95, 0.025, 0.025]`。然后，该类通过比较预测值与平滑后的标签来计算损失。

与前几章一样，我们使用的优化器是 Adam 优化器。然而，我们不是在整个训练过程中使用恒定的学习率，而是在本地模块中定义了`NoamOpt()`类来在训练过程中改变学习率：

```py
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup                                  ①
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    def step(self):                                           ②
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    def rate(self, step=None):
        if step is None:
            step = self._step
        output = self.factor * (self.model_size ** (-0.5) *
        min(step ** (-0.5), step * self.warmup ** (-1.5)))    ③
        return output
```

① 定义预热步骤

② 一个`step()`方法，用于将优化器应用于调整模型参数

③ 根据步骤计算学习率

如前所述的`NoamOpt()`类实现了预热学习率策略。首先，它在训练的初始预热步骤中线性增加学习率。在此预热期之后，该类随后降低学习率，按训练步骤数的倒数平方进行调整。

接下来，我们创建用于训练的优化器：

```py
from utils.ch09util import NoamOpt

optimizer = NoamOpt(256, 1, 2000, torch.optim.Adam(
    model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
```

为了定义训练的损失函数，我们首先在本地模块中创建了以下`SimpleLossCompute()`类。

列表 10.7 用于计算损失的类

```py
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    def __call__(self, x, y, norm):
        x = self.generator(x)                                    ①
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm    ②
        loss.backward()                                          ③
        if self.opt is not None:
            self.opt.step()                                      ④
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()
```

① 使用模型进行预测

② 比较预测值与标签以计算损失，利用标签平滑

③ 计算相对于模型参数的梯度

④ 调整模型参数（反向传播）

`SimpleLossCompute()` 类设计有三个关键元素：`generator` 作为预测模型；`criterion`，这是一个计算损失的功能；以及 `opt`，优化器。这个类通过利用生成器进行预测来处理一个批次的训练数据，表示为 (x, y)。随后，它通过比较这些预测与实际标签 y（由之前定义的 `LabelSmoothing()` 类处理；实际标签 y 将在过程中进行平滑）来评估损失。该类计算相对于模型参数的梯度，并利用优化器相应地更新这些参数。

我们现在可以定义损失函数：

```py
from utils.ch09util import (LabelSmoothing,
       SimpleLossCompute)

criterion = LabelSmoothing(tgt_vocab, 
                           padding_idx=0, smoothing=0.1)
loss_func = SimpleLossCompute(
            model.generator, criterion, optimizer)
```

接下来，我们将使用本章前面准备的数据来训练 Transformer。

### 10.3.2 训练循环

我们可以将训练数据分成训练集和验证集，并训练模型，直到模型在验证集上的性能不再提高，这与我们在第二章中所做的一样。然而，为了节省空间，我们将训练模型 100 个周期。我们将计算每个批次的损失和标记数。在每个周期之后，我们计算该周期的平均损失，作为总损失与总标记数的比率。

列表 10.8 训练一个 Transformer 将英语翻译成法语

```py
for epoch in range(100):
    model.train()
    tloss=0
    tokens=0
    for batch in BatchLoader():
        out = model(batch.src, batch.trg, 
                    batch.src_mask, batch.trg_mask)            ①
        loss = loss_func(out, batch.trg_y, batch.ntokens)      ②
        tloss += loss
        tokens += batch.ntokens                                ③
    print(f"Epoch {epoch}, average loss: {tloss/tokens}")
torch.save(model.state_dict(),"files/en2fr.pth")               ④
```

① 使用 Transformer 进行预测

② 计算损失并调整模型参数

③ 计算批次的标记数

④ 训练后保存训练模型的权重

如果你使用的是支持 CUDA 的 GPU，这个过程可能需要几个小时。如果你使用 CPU 训练，可能需要整整一天。一旦训练完成，模型权重将保存在你的电脑上作为 en2fr.pth。或者，你也可以从我的网站上下载训练好的权重（[`gattonweb.uky.edu/faculty/lium/gai/ch9.zip`](https://gattonweb.uky.edu/faculty/lium/gai/ch9.zip)）。

## 10.4 使用训练好的模型将英语翻译成法语

现在你已经训练了 Transformer，你可以用它将任何英文句子翻译成法语。我们定义了一个名为 `translate()` 的函数，如下所示。

列表 10.9 定义一个 `translate()` 函数以将英语翻译成法语

```py
def translate(eng):
    tokenized_en=tokenizer.tokenize(eng)
    tokenized_en=["BOS"]+tokenized_en+["EOS"]
    enidx=[en_word_dict.get(i,UNK) for i in tokenized_en]  
    src=torch.tensor(enidx).long().to(DEVICE).unsqueeze(0)    
    src_mask=(src!=0).unsqueeze(-2)
    memory=model.encode(src,src_mask)                           ①
    start_symbol=fr_word_dict["BOS"]
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    translation=[]
    for i in range(100):
        out = model.decode(memory,src_mask,ys,
        subsequent_mask(ys.size(1)).type_as(src.data))          ②
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]    
        ys = torch.cat([ys, torch.ones(1, 1).type_as(
            src.data).fill_(next_word)], dim=1)
        sym = fr_idx_dict[ys[0, -1].item()]
        if sym != 'EOS':
            translation.append(sym)
        else:
            break                                               ③
    trans="".join(translation)
    trans=trans.replace("</w>"," ") 
    for x in '''?:;.,'("-!&)%''':
        trans=trans.replace(f" {x}",f"{x}")                     ④
    print(trans) 
    return trans
```

① 使用编码器将英文短语转换为向量表示

② 使用解码器预测下一个标记

③ 当下一个标记是“EOS”时停止翻译

④ 将预测的标记连接起来形成一个法语句子

要将英语短语翻译成法语，我们首先使用分词器将英语句子转换为标记。然后，我们在短语的开始和结束处添加`"BOS"`和`"EOS"`。我们使用本章前面创建的`en_word_dict`字典将标记转换为索引。我们将索引序列输入到训练好的模型的编码器中。编码器产生一个抽象向量表示，并将其传递给解码器。

根据编码器产生的英语句子的抽象向量表示，训练好的模型中的解码器以自回归的方式开始翻译，从开始标记`"BOS"`开始。在每个时间步，解码器根据先前生成的标记生成最可能的下一个标记，直到预测的标记是`"EOS"`，这标志着句子的结束。注意，这与第八章中讨论的文本生成方法略有不同，在那里下一个标记是随机选择的，根据其预测概率。在这里，选择下一个标记的方法是确定性的，这意味着我们主要关注准确性，因此我们选择概率最高的标记。然而，如果您希望翻译具有创造性，您可以像第八章中那样切换到随机预测，并使用`top-K`采样和温度。

最后，我们将标记分隔符更改为空格，并移除标点符号前的空格。输出结果是格式整洁的法语翻译。

让我们尝试使用`translate()`函数翻译英语短语“今天是个美好的一天！”：

```py
from utils.ch09util import subsequent_mask

with open("files/dict.p","rb") as fb:
    en_word_dict,en_idx_dict,\
    fr_word_dict,fr_idx_dict=pickle.load(fb)
trained_weights=torch.load("files/en2fr.pth",
                           map_location=DEVICE)
model.load_state_dict(trained_weights)
model.eval()
eng = "Today is a beautiful day!"
translated_fr = translate(eng)
```

输出结果是

```py
aujourd'hui est une belle journee!
```

您可以通过使用，比如说，谷歌翻译来验证法语翻译确实意味着“今天是个美好的一天！”

让我们尝试一个更长的句子，看看训练好的模型是否能够成功翻译：

```py
eng = "A little boy in jeans climbs a small tree while another child looks on."
translated_fr = translate(eng)
```

输出结果是

```py
un petit garcon en jeans grimpe un petit arbre tandis qu'un autre enfant regarde. 
```

当我用谷歌翻译将前面的输出翻译回英语时，它说，“一个穿牛仔裤的小男孩爬上一棵小树，另一个孩子在一旁观看”——并不完全与原始的英语句子相同，但意思是一样的。

接下来，我们将测试训练好的模型是否为两个英语句子“我不会说法语。”和“I do not speak French.”生成相同的翻译。首先，让我们尝试句子“I don’t speak French.”：

```py
eng = "I don't speak French."
translated_fr = translate(eng)
```

输出结果是

```py
je ne parle pas francais. 
```

现在，让我们尝试句子“I do not speak French.”：

```py
eng = "I do not speak French."
translated_fr = translate(eng)
```

这次输出结果是

```py
je ne parle pas francais. 
```

结果表明，这两个句子的法语翻译完全相同。这表明 Transformer 的编码器组件成功地把握了这两个短语的语义本质。然后，它将它们表示为相似的抽象连续向量形式，随后传递给解码器。解码器随后根据这些向量生成翻译，并产生相同的结果。

练习 10.3

使用 `translate()` 函数将以下两个英文句子翻译成法语。将结果与谷歌翻译的结果进行比较，看看它们是否相同：(i) 我喜欢冬天滑雪！(ii) 你好吗？

在本章中，您训练了一个编码器-解码器 Transformer，通过使用超过 47,000 对英法翻译来将英语翻译成法语。训练的模型表现良好，能够正确翻译常见的英语短语！

在接下来的章节中，您将探索仅解码器 Transformer。您将学习从头开始构建它们，并使用它们生成比第八章中使用长短期记忆生成的文本更连贯的文本。

## 摘要

+   与处理数据序列的循环神经网络不同，Transformers 并行处理输入数据，例如句子。这种并行性提高了它们的效率，但并不固有地允许它们识别输入的序列顺序。为了解决这个问题，Transformers 将位置编码添加到输入嵌入中。这些位置编码是分配给输入序列中每个位置的独特向量，并在维度上与输入嵌入对齐。

+   标签平滑在训练深度神经网络时常用，以提高模型的泛化能力。它用于解决过度自信问题（预测概率大于真实概率）和分类中的过拟合问题。具体来说，它通过调整目标标签来修改模型的学习方式，旨在降低模型对训练数据的信心，这可能导致在未见过的数据上表现更好。

+   基于编码器输出的捕获英语短语意义的输出，训练的 Transformer 中的解码器以自回归的方式开始翻译，从开始标记 `"BOS"` 开始。在每个时间步，解码器根据先前生成的标记生成最可能的下一个标记，直到预测标记是 `"EOS"`，这表示句子的结束。

* * *

^(1) Vaswani 等人，2017，“Attention Is All You Need.” [`arxiv.org/abs/1706.03762`](https://arxiv.org/abs/1706.03762).
