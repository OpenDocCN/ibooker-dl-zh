# 第十一章：从零开始构建生成式预训练变换器

本章涵盖

+   从零开始构建生成式预训练变换器

+   因果自注意力

+   从预训练模型中提取和加载权重

+   使用 GPT-2 生成连贯的文本，ChatGPT 和 GPT-4 的前辈

生成式预训练变换器 2（GPT-2）是由 OpenAI 开发的高级大型语言模型（LLM），于 2019 年 2 月宣布发布。它在自然语言处理（NLP）领域取得了重大里程碑，并为开发更复杂的模型铺平了道路，包括其继任者 ChatGPT 和 GPT-4。

GPT-2 是其前辈 GPT-1 的改进，旨在根据给定的提示生成连贯且上下文相关的文本，展示了在多种风格和主题上模仿人类文本生成的非凡能力。在其宣布时，OpenAI 最初决定不向公众发布 GPT-2 最强大的版本（也是本章中你将从头开始构建的，拥有 150 亿参数）。主要担忧是潜在的误用，例如生成误导性新闻文章、在线冒充个人或自动化生产侮辱性或虚假内容。这一决定在 AI 和科技社区中引发了关于 AI 开发伦理和创新与安全之间平衡的激烈辩论。

OpenAI 后来采用了分阶段发布策略，逐步使模型的小版本可用，同时监控效果并探索安全部署策略。最终，在 2019 年 11 月，OpenAI 发布了完整模型，以及几个数据集和一个检测模型生成文本的工具，为负责任的 AI 使用讨论做出了贡献。正因为这次发布，你将学习如何从 GPT-2 中提取预训练权重并将它们加载到你创建的 GPT-2 模型中。

GPT-2 基于我们在第九章和第十章讨论的变换器架构。然而，与之前创建的英法翻译器不同，GPT-2 是一个仅解码器的变换器，这意味着模型中没有编码器堆栈。在将英语短语翻译成法语时，编码器捕捉英语短语的含义并将其传递给解码器以生成翻译。然而，在文本生成任务中，模型不需要编码器来理解不同的语言。相反，它仅使用解码器架构根据句子中的前一个标记生成文本。与其他变换器模型一样，GPT-2 使用自注意力机制并行处理输入数据，显著提高了训练 LLM 的效率和效果。

GPT-2 在大量文本数据语料库上进行了预训练，本质上是在给定句子中前一个词的情况下预测句子中的下一个词。这种训练使模型能够学习广泛的语言模式、语法和知识。

在本章中，你将从零开始学习构建 GPT-2XL，这是 GPT-2 的最大版本。之后，你将学习如何从 Hugging Face（一个托管和协作机器学习模型、数据集和应用的 AI 社区）中提取预训练的权重并将它们加载到自己的 GPT-2 模型中。你将通过向模型提供提示来使用你的 GPT-2 生成文本。GPT-2 计算可能下一个标记的概率并从中采样。它可以根据接收到的输入提示生成连贯且与上下文相关的段落文本。此外，正如你在第八章中所做的那样，你可以通过使用 `temperature` 和 `top-K` 采样来控制生成文本的创造性。

虽然 GPT-2 在自然语言处理领域取得了显著的进步，但调整你的期望并认识到其固有的局限性是至关重要的。直接将 GPT-2 与 ChatGPT 或 GPT-4 进行比较是不恰当的，因为 GPT-2XL 只有 15 亿个参数，而 ChatGPT 有 1750 亿个参数，GPT-4 的估计参数量为 1.76 万亿。GPT-2 的主要局限性之一是它对其生成的内容的真正理解不足。该模型根据其训练数据中单词的概率分布预测序列中的下一个单词，这可以生成语法正确且看似合逻辑的文本。然而，该模型缺乏对词语背后含义的真正理解，可能导致潜在的不准确、无意义的陈述或肤浅的内容。

另一个关键因素是 GPT-2 的有限上下文意识。虽然它可以在短文本跨度内保持连贯性，但在较长的段落中会遇到困难，可能导致连贯性丧失、矛盾或不相关的内容。我们应谨慎不要高估模型生成需要持续关注上下文和细节的长篇内容的能力。因此，虽然 GPT-2 在自然语言处理领域迈出了重要的一步，但以健康程度的怀疑态度对待其生成的文本并设定现实期望是非常重要的。

## 11.1 GPT-2 架构和因果自注意力

GPT-2 作为仅基于解码器的 Transformer（它根据句子中的先前标记生成文本，无需编码器理解不同语言），与第九章和第十章中讨论的英法翻译器的解码器组件相呼应。与它的双语版本不同，GPT-2 缺少编码器，因此在输出生成过程中不包含编码器派生的输入。该模型完全依赖于序列中的先前标记来生成其输出。

在本节中，我们将讨论 GPT-2 的架构。我们还将深入了解因果自注意力机制，这是 GPT-2 模型的核心。

### 11.1.1 GPT-2 的架构

GPT-2 有四种不同的尺寸：小（S）、中（M）、大（L）和超大（XL），每种尺寸的能力各不相同。我们的主要关注点将是功能最强大的版本，即 GPT-2XL。最小的 GPT-2 模型大约有 124 百万个参数，而超大版本则有大约 15 亿个参数。它是 GPT-2 模型中最强大的，拥有最多的参数。GPT-2XL 能够理解复杂语境，生成连贯且细腻的文本。

GPT-2 由许多相同的解码器块组成。超大版本有 48 个解码器块，而其他三个版本分别有 12、24 和 36 个解码器块。每个解码器块由两个不同的子层组成。第一个子层是一个因果自注意力层，我将在不久的将来详细解释。第二个子层是一个基本的、位置相关的、全连接的前馈网络，正如我们在英语到法语翻译器中的编码器和解码器块中所看到的。每个子层都包含层归一化和残差连接，以稳定训练过程。

图 11.1 是 GPT-2 架构的示意图。

![图片](img/CH11_F01_Liu.png)

图 11.1 GPT-2 模型的架构。GPT-2 是一个仅包含解码器的 Transformer，由 N 个相同的解码器层组成。每个解码器块包含两个子层。第一个子层是一个因果自注意力层。第二个是一个前馈网络。每个子层都使用层归一化和残差连接。输入首先通过词嵌入和位置编码，然后将总和传递给解码器。解码器的输出经过层归一化和线性层。

GPT-2 首先将一系列标记的索引通过词嵌入和位置编码传递，以获得输入嵌入（我将在不久的将来解释这一过程）。输入嵌入依次通过 N 个解码器块。之后，输出通过层归一化和线性层。GPT-2 的输出数量是词汇表中的唯一标记数量（所有 GPT-2 版本共有 50,257 个标记）。该模型旨在根据序列中的所有前一个标记预测下一个标记。

为了训练 GPT-2，OpenAI 使用了一个名为 WebText 的数据集，该数据集是从互联网上自动收集的。该数据集包含各种文本，包括 Reddit 链接等高度点赞的网站，旨在涵盖广泛的人类语言和主题。估计该数据集包含大约 40GB 的文本。

训练数据被分解成固定长度的序列（所有 GPT-2 版本的长度为 1,024 个标记）并用作输入。这些序列向右移动一个标记，并在训练过程中用作模型的输出。由于模型使用因果自注意力，其中序列中的未来标记在训练过程中被屏蔽（即隐藏），这实际上是在训练模型根据序列中所有之前的标记来预测下一个标记。

### 11.1.2 GPT-2 中的词嵌入和位置编码

GPT-2 使用一种称为字节对编码器（Byte Pair Encoder，BPE）的子词分词方法将文本分解成单个标记（在大多数情况下是整个单词或标点符号，但对于不常见的单词则是音节）。这些标记随后被映射到 0 到 50,256 之间的索引，因为词汇量大小为 50,257。GPT-2 将训练数据中的文本转换为通过词嵌入捕获其意义的向量表示，这与你在前两章中所做的方式类似。

为了给你一个具体的例子，短语“this is a prompt”首先通过 BPE 分词转换为四个标记，`['this', ' is', ' a', ' prompt']`。然后每个标记由一个大小为 50,257 的独热变量表示。GPT-2 模型将它们通过词嵌入层压缩成具有更小浮点值大小的压缩向量，例如 GPT-2XL 中的长度为 1,600（其他三个版本的 GPT-2 的长度分别为 768、1,024 和 1,280）。通过词嵌入，短语“this is a prompt”被表示为一个 4 × 1,600 大小的矩阵，而不是原始的 4 × 50,257。词嵌入显著减少了模型参数的数量，并使训练更加高效。图 11.2 的左侧展示了词嵌入的工作原理。

![图片](img/CH11_F02_Liu.png)

图 11.2 GPT-2 首先将序列中的每个标记表示为一个 50,276 位的独热向量。序列的标记表示通过词嵌入层压缩，形成一个维度为 1,600 的嵌入。GPT-2 还使用一个 1,024 位的独热向量来表示序列中的每个位置。序列的位置表示通过位置编码层压缩，形成一个同样维度为 1,600 的嵌入。词嵌入和位置编码被相加以形成输入嵌入。

GPT-2，与其他 Transformer 类似，并行处理输入数据，这本质上导致它无法识别输入数据的序列顺序。为了解决这个问题，我们需要向输入嵌入中添加位置编码。GPT-2 采用了一种独特的方法来进行位置编码，与 2017 年发表的具有里程碑意义的论文“Attention Is All You Need”中概述的方法不同。相反，GPT-2 的位置编码技术与词嵌入相似。鉴于该模型能够处理输入序列中的最多 1,024 个标记，序列中的每个位置最初由一个同样大小的 one-hot 向量表示。例如，在序列“this is a prompt”中，第一个标记由一个 one-hot 向量表示，其中所有元素都是零，除了第一个，它被设置为 1。第二个标记遵循同样的模式，由一个向量表示，其中除了第二个元素之外的所有元素都是零。因此，“this is a prompt”这个短语的序列表示表现为一个 4 × 1,024 的矩阵，如图 11.2 右上角所示。

为了生成位置编码，序列的位置表示通过一个维度为 1,024 × 1,600 的线性神经网络进行处理。该网络中的权重在初始化时是随机的，并在训练过程中进行优化。因此，序列中每个标记的位置编码是一个 1,600 维的向量，与词嵌入向量的维度相匹配。一个序列的输入嵌入是其词嵌入和位置编码的总和，如图 11.2 底部所示。在短语“this is a prompt”的上下文中，词嵌入和位置编码都结构化为 4 × 1,600 的矩阵。因此，“this is a prompt”的输入嵌入，即这两个矩阵的总和，保持了 4 × 1,600 的维度。

### 11.1.3 GPT-2 中的因果自注意力

因果自注意力是 GPT-2 模型（以及在 GPT 系列模型中）中的一个关键机制，它使模型能够通过条件化先前生成的标记序列来生成文本。这与我们在第九章和第十章讨论的英语到法语翻译器中每个解码器层第一子层的掩码自注意力相似，尽管实现上略有不同。

注意：在此上下文中，“因果”这一概念指的是模型确保对给定标记的预测只能受到序列中先于它的标记的影响，尊重文本生成的因果（时间向前）方向。这对于生成连贯且上下文相关的文本输出至关重要。

自注意力是一种机制，允许输入序列中的每个标记关注同一序列中的所有其他标记。在 GPT-2 等 Transformer 模型的情况下，自注意力使模型能够在处理特定标记时权衡其他标记的重要性，从而捕捉句子中单词的上下文和关系。

为了确保因果性，GPT-2 的自注意力机制被修改，使得任何给定的标记只能关注它自己和序列中之前出现的标记。这是通过在注意力计算中屏蔽未来标记（即在序列中当前标记之后出现的标记）来实现的，确保模型在预测序列中的下一个标记时不能“看到”或受到未来标记的影响。例如，在短语“this is a prompt”中，当模型使用单词“this”来预测单词“is”时，掩码隐藏了第一次时间步中的最后三个单词。为了实现这一点，我们在计算注意力分数时将对应未来标记的位置设置为负无穷大。在 softmax 激活后，未来标记被分配零权重，从而有效地从注意力计算中移除。

让我们用一个具体的例子来说明因果自注意力在代码中是如何工作的。短语“this is a prompt”的输入嵌入在词嵌入和位置编码之后是一个 4 × 1,600 的矩阵。然后我们通过 GPT-2 的 N 个解码器层传递这个输入嵌入。在每个解码器层中，它首先通过以下因果自注意力子层。输入嵌入通过三个神经网络传递以创建查询 Q、键 K 和值 V，如下所示。

列表 11.1 创建`query`、`key`和`value`向量

```py
import torch
import torch.nn as nn

torch.manual_seed(42)
x=torch.randn((1,4,1600))                        ①
c_attn=nn.Linear(1600,1600*3)                    ②
B,T,C=x.size()
q,k,v=c_attn(x).split(1600,dim=2)                ③
print(f"the shape of Q vector is {q.size()}")
print(f"the shape of K vector is {k.size()}")
print(f"the shape of V vector is {v.size()}")    ④
```

① 创建三个神经网络

② 创建输入嵌入 x

③ 将输入嵌入传递给三个神经网络以创建 Q、K 和 V

④ 打印出 Q、K 和 V 的大小

我们首先创建一个大小为 4 × 1,600 的矩阵，与“this is a prompt”的输入嵌入大小相同。然后我们通过三个大小为 1,600 × 1,600 的神经网络传递输入嵌入，以获得查询 Q、键 K 和值 V。如果您运行前面的代码块，您将看到以下输出：

```py
the shape of Q vector is torch.Size([1, 4, 1600])
the shape of K vector is torch.Size([1, 4, 1600])
the shape of V vector is torch.Size([1, 4, 1600])
```

Q、K 和 V 的形状都是 4 × 1,600。接下来，我们不是使用一个头，而是将它们分成 25 个并行头。每个头关注输入的不同部分或方面，使模型能够捕捉更广泛的信息，并对输入数据形成更详细和上下文化的理解。因此，我们有了 25 组 Q、K 和 V：

```py
hs=C//25
k = k.view(B, T, 25, hs).transpose(1, 2) 
q = q.view(B, T, 25, hs).transpose(1, 2) 
v = v.view(B, T, 25, hs).transpose(1, 2)         ①
print(f"the shape of Q vector is {q.size()}")
print(f"the shape of K vector is {k.size()}")
print(f"the shape of V vector is {v.size()}")    ②
```

① 将 Q、K 和 V 分为 25 个头

② 打印出多头 Q、K 和 V 的大小

如果您运行前面的代码块，您将看到以下输出：

```py
the shape of Q vector is torch.Size([1, 25, 4, 64])
the shape of K vector is torch.Size([1, 25, 4, 64])
the shape of V vector is torch.Size([1, 25, 4, 64])
```

现在，Q、K 和 V 的形状是 25 × 4 × 64：这意味着我们有 25 个头；每个头有一组查询、键和值，大小都是 4 × 64。

接下来，我们计算每个头中的缩放注意力分数：

```py
import math
scaled_att = (q @ k.transpose(-2, -1)) *\
            (1.0 / math.sqrt(k.size(-1)))
print(scaled_att[0,0])
```

缩放后的注意力分数是每个头部中 Q 和 K 的点积，并按 K 的维度的平方根进行缩放，即 1,600/25 = 64。缩放后的注意力分数在每个头部形成一个 4 × 4 矩阵，我们在第一个头部打印出这些值：

```py
tensor([[ 0.2334,  0.1385, -0.1305,  0.2664],
        [ 0.2916,  0.1044,  0.0095,  0.0993],
        [ 0.8250,  0.2454,  0.0214,  0.8667],
        [-0.1557,  0.2034,  0.2172, -0.2740]], grad_fn=<SelectBackward0>)
```

第一个头部的缩放注意力分数也显示在图 11.3 底部左边的表格中。

练习 11.1

张量 `scaled_att` 包含 25 个头部中的缩放注意力分数。我们之前已经打印出了第一个头部的这些值。你是如何打印出第二个头部的缩放注意力分数的？

接下来，我们对缩放后的注意力分数应用一个掩码，以隐藏序列中的未来标记：

```py
mask=torch.tril(torch.ones(4,4))              ①
print(mask)
masked_scaled_att=scaled_att.masked_fill(\
    mask == 0, float('-inf'))                 ②
print(masked_scaled_att[0,0])
```

① 创建一个掩码

② 通过将未来标记的值更改为 –∞ 来对缩放后的注意力分数应用掩码

![](img/CH11_F03_Liu.png)

图 11.3 如何在因果自注意力中计算掩码注意力权重。掩码应用于缩放后的注意力分数，使得对应未来标记的值（矩阵中主对角线以上的值）变为 –∞。然后我们对掩码后的缩放注意力分数应用 softmax 函数，从而获得掩码注意力权重。掩码确保给定标记的预测只能受到序列中先于它的标记的影响，而不是未来标记的影响。这对于生成连贯且上下文相关的文本输出至关重要。

如果你运行前面的代码，你会看到以下输出：

```py
tensor([[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]])
tensor([[ 0.2334,    -inf,    -inf,    -inf],
        [ 0.2916,  0.1044,    -inf,    -inf],
        [ 0.8250,  0.2454,  0.0214,    -inf],
        [-0.1557,  0.2034,  0.2172, -0.2740]], grad_fn=<SelectBackward0>)
```

该掩码是一个 4 × 4 矩阵，如图 11.3 顶部所示。掩码的下半部分（主对角线以下的值）为 1，而掩码的上半部分（主对角线以上的值）为 0。当这个掩码应用于缩放后的注意力分数时，矩阵上半部分的值变为 –∞（图 11.3 中间底部）。这样，当我们对缩放后的注意力分数应用 softmax 函数时，注意力权重矩阵的上半部分被填充为 0（图 11.3 右下角）：

```py
import torch.nn.functional as F
att = F.softmax(masked_scaled_att, dim=-1)
print(att[0,0])
```

我们使用以下值打印出第一个头部的注意力权重：

```py
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.5467, 0.4533, 0.0000, 0.0000],
        [0.4980, 0.2790, 0.2230, 0.0000],
        [0.2095, 0.3001, 0.3042, 0.1862]], grad_fn=<SelectBackward0>)
```

第一行表示在第一个时间步，标记“this”只关注自身，而不关注任何未来的标记。同样，如果你看第二行，标记“this is”相互关注，但不关注未来的标记“a prompt”。

注意：在这个数值示例中，权重未经过训练，所以不要将这些值直接理解为注意力权重。我们使用它们作为示例来说明因果自注意力是如何工作的。

练习 11.2

我们已经打印出了第一个头部的注意力权重。你是如何打印出最后一个（即第 25 个）头部的注意力权重的？

最后，我们计算每个头部的注意力向量，它是注意力权重和值向量的点积。然后，将 25 个头部的注意力向量合并为一个单一的注意力向量：

```py
y=att@v
y = y.transpose(1, 2).contiguous().view(B, T, C)
print(y.shape)
```

输出结果为

```py
torch.Size([1, 4, 1600])
```

因果自注意力机制后的最终输出是一个 4×1,600 的矩阵，与因果自注意力子层的输入大小相同。解码器层被设计成输入和输出具有相同的维度，这使得我们可以堆叠许多解码器层来增加模型的表示能力，并在训练期间实现层次特征提取。

## 11.2 从头构建 GPT-2XL

现在你已经了解了 GPT-2 的架构以及其核心成分因果自注意力机制的工作原理，让我们从头开始创建 GPT-2 的最大版本。

在本节中，你将首先学习使用 GPT-2 中的子词分词方法，即字节对编码器（BPE）分词器，将文本分解成单个标记。你还将学习 GPT-2 中前馈网络使用的 GELU 激活函数。之后，你将编写因果自注意力机制，并将其与前馈网络结合形成一个解码器块。最后，你将堆叠 48 个解码器块来创建 GPT-2XL 模型。本章的代码改编自 Andrej Kaparthy 的优秀 GitHub 仓库（[`github.com/karpathy/minGPT`](https://github.com/karpathy/minGPT)）。如果你想要深入了解 GPT-2 的工作原理，我鼓励你阅读该仓库。

### 11.2.1 BPE 分词

GPT-2 使用一种称为字节对编码器（BPE）的子词分词方法，这是一种数据压缩技术，已被改编用于 NLP 任务中的文本分词。它因在训练 LLM（如 GPT 系列和 BERT）中的应用而特别知名。BPE 的主要目标是以一种平衡词汇量和分词文本长度的方法将一段文本编码成一系列标记。

BPE 通过迭代地将数据集中最频繁出现的连续字符对合并成一个新的标记来工作，前提是满足某些条件。这个过程会重复进行，直到达到所需的词汇量或没有更多的合并是有益的。BPE 允许对文本进行高效表示，在字符级和词级分词之间取得平衡。它有助于在不显著增加序列长度的同时减少词汇量，这对于 NLP 模型的性能至关重要。

我们在第八章讨论了三种类型分词方法的优缺点（字符级、词级和子词分词）。此外，你还在第八章从头实现了词级分词器（并在第十二章再次这样做）。因此，在本章中，我们将直接借用 OpenAI 的分词方法。BPE 的详细工作原理超出了本书的范围。你需要知道的是，它首先将文本转换为子词标记，然后是相应的索引。

从安德烈·卡帕西（Andrej Karpathy）的 GitHub 仓库下载文件`bpe.py`，[`mng.bz/861B`](https://mng.bz/861B)，并将其放置在您的计算机上的/utils/文件夹中。在本章中，我们将使用该文件作为本地模块。正如安德烈·卡帕西在他的 GitHub 仓库中解释的那样，该模块基于 OpenAI 的实现[`mng.bz/EOlj`](https://mng.bz/EOlj)，但进行了轻微修改，使其更容易理解。

要了解模块`bpe.py`如何将文本转换为标记然后转换为索引，让我们尝试一个示例：

```py
from utils.bpe import get_encoder

example="This is the original text."                       ①
bpe_encoder=get_encoder()                                  ②
response=bpe_encoder.encode_and_show_work(example)
print(response["tokens"])                                  ③
```

① 示例句子的文本

② 从 bpe.py 模块实例化 get_encoder()类

③ 分词示例文本并打印出标记

输出结果为

```py
['This', ' is', ' the', ' original', ' text', '.']
```

BPE 分词器将示例文本“这是原始文本。”分割成六个标记，如前述输出所示。请注意，BPE 分词器不会将大写字母转换为小写字母。这导致更具有意义的分词，但也导致独特的标记数量大大增加。实际上，所有版本的 GPT-2 模型词汇量大小为 50,276，比前几章的词汇量大几倍。

我们还可以使用`bpe.py`模块将标记映射到索引：

```py
print(response['bpe_idx'])
```

输出结果为

```py
[1212, 318, 262, 2656, 2420, 13]
```

上述列表包含对应于示例文本“这是原始文本。”中的六个标记的六个索引。

我们也可以根据索引恢复文本：

```py
from utils.bpe import BPETokenizer 

tokenizer = BPETokenizer()                                  ①
out=tokenizer.decode(torch.LongTensor(response['bpe_idx'])) ②
print(out) 
```

① 从 bpe.py 模块实例化 BPETokenizer()类

② 使用分词器根据索引恢复文本

前述代码块输出的结果为

```py
This is the original text.
```

如您所见，BPE 分词器已将示例文本恢复到其原始形式。

练习 11.3

使用 BPE 分词器将短语“this is a prompt”分割成标记。之后，将标记映射到索引。最后，根据索引恢复短语。

### 11.2.2 高斯误差线性单元激活函数

高斯误差线性单元（GELU）激活函数用于 GPT-2 中每个解码器块的馈送前子层。GELU 提供了一种线性和非线性激活特性的混合，这在深度学习任务中已被发现可以增强模型性能，尤其是在 NLP 领域。

GELU 提供了一个非线性、平滑的曲线，与像 ReLU 这样的其他函数相比，在训练期间允许进行更细微的调整。这种平滑性有助于更有效地优化神经网络，因为它为反向传播提供了更连续的梯度。为了比较 GELU 与我们的首选激活函数 ReLU，我们首先定义一个 GELU()类：

```py
class GELU(nn.Module):
    def forward(self, x):
        return 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*\
                       (x + 0.044715 * torch.pow(x, 3.0))))
```

ReLU 函数在它有尖角的地方不可微分。相比之下，GELU 激活函数在所有地方都是可微分的，并提供了一个更好的学习过程。接下来，我们绘制 GELU 激活函数的图像，并与 ReLU 进行比较。

列表 11.2 比较两个激活函数：GELU 和 ReLU

```py
import matplotlib.pyplot as plt
import numpy as np

genu=GELU()
def relu(x):                                           ①
    y=torch.zeros(len(x))
    for i in range(len(x)):
        if x[i]>0:
            y[i]=x[i]
    return y                 
xs = torch.linspace(-6,6,300)
ys=relu(xs)
gs=genu(xs)
fig, ax = plt.subplots(figsize=(6,4),dpi=300)
plt.xlim(-3,3)
plt.ylim(-0.5,3.5)
plt.plot(xs, ys, color='blue', label="ReLU")           ②
plt.plot(xs, gs, "--", color='red', label="GELU")      ③
plt.legend(fontsize=15)
plt.xlabel("values of x")
plt.ylabel("values of $ReLU(x)$ and $GELU(x)$")
plt.title("The ReLU and GELU Activation Functions")
plt.show()
```

① 定义一个表示 ReLU 的函数

② 用实线绘制 ReLU 激活函数

③ 用虚线绘制 GELU 激活函数

如果你运行前面的代码块，你会看到一个如图 11.4 所示的图形。

![图片](img/CH11_F04_Liu.png)

图 11.4 比较 GELU 激活函数与 ReLU。实线是 ReLU 激活函数，而虚线是 GELU 激活函数。ReLU 在某个地方有拐角，因此不是处处可导。相比之下，GELU 在所有地方都是可导的。GELU 的这种平滑性有助于更有效地优化神经网络，因为它在训练过程中为反向传播提供了更连续的梯度。

此外，GELU 公式的制定使其能够更有效地模拟输入数据分布。它结合了线性和高斯分布建模的特性，这对于在 NLP 任务中遇到的复杂、多变的数据特别有益。这种能力有助于捕捉语言数据中的微妙模式，提高模型对文本的理解和生成。

### 11.2.3 因果自注意力

如我们之前所解释的，因果自注意力是 GPT-2 模型的核心元素。接下来，我们将从头开始在 PyTorch 中实现这一机制。

我们首先指定本章将要构建的 GPT-2XL 模型中的超参数。为此，我们定义了一个`Config()`类，其值如下所示。

列表 11.3 在 GPT-2XL 中指定超参数

```py
class Config():                                       ①
    def __init__(self):
        self.n_layer = 48
        self.n_head = 25
        self.n_embd = 1600
        self.vocab_size = 50257
        self.block_size = 1024 
        self.embd_pdrop = 0.1 
        self.resid_pdrop = 0.1 
        self.attn_pdrop = 0.1                         ②

config=Config()                                       ③
```

① 定义一个 Config()类

② 将模型超参数作为类的属性放置

③ 实例化 Config()类

我们定义了一个`Config()`类，并在其中创建了一些属性，用作 GPT-2XL 模型中的超参数。`n_layer`属性表示我们构建的 GPT-2XL 模型将包含 48 个解码器层（我们使用“解码器块”和“解码器层”这两个术语可以互换）。`n_head`属性表示在计算因果自注意力时，我们将 Q、K 和 V 分割成 25 个并行头。`n_embd`属性表示嵌入维度是 1,600：每个标记将由一个 1,600 值的向量表示。`vocab_size`属性表示词汇表中有 50,257 个独特的标记。`block_size`属性表示输入到 GPT-2XL 模型中的序列最多包含 1,024 个标记。dropout 率都设置为 0.1。

在上一节中，我详细解释了因果自注意力是如何工作的。接下来，我们将定义一个`CausalSelfAttention()`类来实现它。

列表 11.4 实现因果自注意力

```py
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(\
                   config.block_size, config.block_size))
             .view(1, 1, config.block_size, config.block_size)) ①
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    def forward(self, x):
        B, T, C = x.size() 
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)     ②
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2) 
        q = q.view(B, T, self.n_head, hs).transpose(1, 2) 
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)       ③
        att = (q @ k.transpose(-2, -1)) *\
            (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, \
                              float(‚-inf'))
        att = F.softmax(att, dim=-1)                            ④
        att = self.attn_dropout(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)        ⑤
        y = self.resid_dropout(self.c_proj(y))
        return y
```

① 创建一个掩码并将其注册为缓冲区，因为它不需要更新

② 将输入嵌入通过三个神经网络传递以获得 Q、K 和 V

③ 将 Q、K 和 V 分割成多个头

④ 计算每个头的掩码注意力权重

⑤ 将所有头的注意力向量连接成一个单一的注意力向量

在 PyTorch 中，`register_buffer`是一种将张量注册为缓冲区的方法。缓冲区中的变量不被视为模型的可学习参数；因此，它们在反向传播期间不会被更新。在前面的代码块中，我们创建了一个掩码并将其注册为缓冲区。这会影响我们稍后提取和加载模型权重的方式：在从 GPT-2XL 检索权重时，我们将省略掩码。 

正如我们在第一节中解释的，输入嵌入通过三个神经网络来获取查询 Q、键 K 和值 V。然后我们将它们分成 25 个头，并在每个头中计算掩码自注意力。之后，我们将 25 个注意力向量重新组合成一个单一的注意力向量，这是前一个`CausalSelfAttention()`类的输出。

### 11.2.4 构建 GPT-2XL 模型

接下来，我们在因果自注意力子层中添加一个前馈网络，以形成一个解码器块，如下所示。

列表 11.5 构建解码器块

```py
class Block(nn.Module):
    def __init__(self, config):                                 ①
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj = nn.Linear(4 * config.n_embd, config.n_embd),
            act    = GELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf=lambda x:m.dropout(m.c_proj(m.act(m.c_fc(x)))) 
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))                         ②
        x = x + self.mlpf(self.ln_2(x))                         ③
        return x
```

① 初始化 Block()类

② 块中的第一个子层是因果自注意力子层，包含层归一化和残差连接。

③ 块中的第二个子层是一个前馈网络，包含 GELU 激活、层归一化和残差连接。

每个解码器块由两个子层组成。第一个子层是因果自注意力机制，包含层归一化和残差连接。解码器块内的第二个子层是前馈网络，它结合了 GELU 激活函数，以及层归一化和残差连接。

我们堆叠 48 个解码器层来形成 GPT-2XL 模型的主体，如下所示。

列表 11.6 构建 GPT-2XL 模型

```py
class GPT2XL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) 
                               for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),))
        self.lm_head = nn.Linear(config.n_embd,
                                 config.vocab_size, bias=False)
    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0,t,dtype=torch.long).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)    
        pos_emb = self.transformer.wpe(pos)    
        x = self.transformer.drop(tok_emb + pos_emb)            ①
        for block in self.transformer.h:
            x = block(x)                                        ②
        x = self.transformer.ln_f(x)                            ③
        logits = self.lm_head(x)                                ④
        loss = None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),
                           targets.view(-1), ignore_index=-1)
        return logits, loss
```

① 计算输入嵌入为词嵌入和位置编码之和

② 将输入嵌入通过 48 个解码器块

③ 再次应用层归一化

④ 将线性头附加到输出上，使得输出的数量等于唯一标记的数量

我们在本章的第一节中解释了如何在`GPT2XL()`类中构建模型。模型的输入由对应于词汇表中标记的索引序列组成。我们首先将输入通过词嵌入和位置编码；然后我们将这两个嵌入相加形成输入嵌入。输入嵌入经过 48 个解码器块。之后，我们对输出应用层归一化，然后附加一个线性头，使得输出的数量为 50,257，即词汇表的大小。输出是词汇表中 50,257 个标记的对数几率。稍后，我们将对对数几率应用 softmax 激活函数，以获得生成文本时词汇表中唯一标记的概率分布。

注意：由于模型太大，我们没有将其移动到 GPU 上。这导致本章后面文本生成的速度较低。然而，如果你有访问带有大内存（例如，超过 32GB）的 CUDA 启用 GPU 的权限，你可以将模型移动到 GPU 上以实现更快的文本生成。

接下来，我们将通过实例化我们之前定义的 `GPT2XL()` 类来创建 GPT-2XL 模型：

```py
model=GPT2XL(config)
num=sum(p.numel() for p in model.transformer.parameters())
print("number of parameters: %.2fM" % (num/1e6,))
```

我们还计算模型主体中的参数数量。输出结果是

```py
number of parameters: 1557.61M
```

前面的输出显示 GPT-2XL 有超过 15 亿个参数。请注意，这个数字不包括模型末尾线性头部的参数。根据下游任务的不同，我们可以将不同的头部附加到模型上。由于我们的重点是文本生成，我们附加了一个线性头部以确保输出的数量等于词汇表中的唯一标记数量。

注意：在 GPT-2、ChatGPT 或 BERT 等大型语言模型中，输出头部指的是模型中负责根据处理后的输入产生实际输出的最后一层。这个输出会根据模型执行的任务而变化。在文本生成中，输出头部通常是一个线性层，它将最终的隐藏状态转换为词汇表中每个标记的 logits。这些 logits 然后通过 softmax 函数生成词汇表上的概率分布，用于预测序列中的下一个标记。对于分类任务，输出头部通常由一个线性层和一个 softmax 函数组成。线性层将模型的最终隐藏状态转换为每个类别的 logits，softmax 函数将这些 logits 转换为每个类别的概率。输出头部的具体架构可能因模型和任务而异，但其主要功能是将处理后的输入映射到所需的输出格式（例如，类别概率、标记概率等）。

最后，你可以打印出 GPT-2XL 模型的结构：

```py
print(model)
```

输出结果是

```py
GPT2XL(
  (transformer): ModuleDict(
    (wte): Embedding(50257, 1600)
    (wpe): Embedding(1024, 1600)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-47): 48 x Block(
        (ln_1): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=1600, out_features=4800, bias=True)
          (c_proj): Linear(in_features=1600, out_features=1600, bias=True)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
        (mlp): ModuleDict(
          (c_fc): Linear(in_features=1600, out_features=6400, bias=True)
          (c_proj): Linear(in_features=6400, out_features=1600, bias=True)
          (act): GELU()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=1600, out_features=50257, bias=False)
)
```

它显示了 GPT-2XL 模型中的详细块和层。

就这样，你从头开始创建了 GPT-2XL 模型！

## 11.3 加载预训练权重并生成文本

尽管你刚刚创建了 GPT-2XL 模型，但它尚未经过训练。因此，你不能用它生成任何有意义的文本。

由于模型参数数量庞大，没有超级计算设施就无法训练模型，更不用说训练模型所需的数据量了。幸运的是，包括最大的 GPT-2 模型 GPT-2XL 在内的预训练权重于 2019 年 11 月 5 日由 OpenAI 向公众发布（参见 OpenAI 网站上的声明，[`openai.com/research/gpt-2-1-5b-release`](https://openai.com/research/gpt-2-1-5b-release)，以及一家美国科技新闻网站 The Verge 的报道，[`mng.bz/NBm7`](https://mng.bz/NBm7)）。因此，我们将加载预训练权重以在本节中生成文本。

### 11.3.1 加载 GPT-2XL 的预训练参数

我们将使用 Hugging Face 团队开发的 `transformers` 库来提取 GPT-2XL 中的预训练权重。

首先，在 Jupyter Notebook 的新单元中运行以下代码行以在您的计算机上安装 `transformers` 库：

```py
!pip install transformers
```

接下来，我们从 `transformers` 库中导入 GPT2 模型并提取 GPT-2XL 中的预训练权重：

```py
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained('gpt2-xl')         ①
sd_hf = model_hf.state_dict()                                 ②
print(model_hf)                                               ③
```

① 加载预训练的 GPT-2XL 模型

② 提取模型权重

③ 打印出原始 OpenAI GTP-2XL 模型的模型结构

上一段代码块输出的结果是

```py
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 1600)
    (wpe): Embedding(1024, 1600)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-47): 48 x GPT2Block(
        (ln_1): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()                                   ①
          (c_proj): Conv1D()                                   ①
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()                                     ①
          (c_proj): Conv1D()                                   ①
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((1600,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=1600, out_features=50257, bias=False)
)
```

① OpenAI 使用了 Conv1d 层而不是我们使用的线性层

如果你将这个模型结构与上一节中的模型结构进行比较，你会注意到它们是相同的，只是线性层被 Conv1d 层所取代。正如我们在第九章和第十章中解释的，在前馈网络中，我们将输入中的值视为独立的元素，而不是一个序列。因此，我们通常称它为一维卷积网络。OpenAI 检查点在模型中使用线性层的地方使用了 Conv1d 模块。因此，当我们从 Hugging Face 提取模型权重并将其放置在我们的模型中时，我们需要转置某些权重矩阵。

要理解它是如何工作的，让我们看看 OpenAI GPT-2XL 模型第一个解码块中前馈网络第一层的权重。我们可以按以下方式打印出其形状：

```py
print(model_hf.transformer.h[0].mlp.c_fc.weight.shape)
```

输出结果是

```py
torch.Size([1600, 6400])
```

Conv1d 层中的权重矩阵是一个大小为 (1,600, 6,400) 的张量。

现在，如果我们看看我们刚刚构建的模型中相同的权重矩阵，其形状是

```py
print(model.transformer.h[0].mlp.c_fc.weight.shape)
```

这次的输出结果是

```py
torch.Size([6400, 1600])
```

我们模型中线性层的权重矩阵是一个大小为 (6,400, 1,600) 的张量，它是 OpenAI GPT-2XL 权重矩阵的转置矩阵。因此，在我们将权重矩阵放置在我们的模型之前，我们需要将 OpenAI GPT-2XL 模型中所有 Conv1d 层的权重矩阵进行转置。

接下来，我们将原始 OpenAI GPT-2XL 模型中的参数命名为 `keys`：

```py
keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] 
```

注意，我们在上一行代码中排除了以`attn.masked_bias`结尾的参数。OpenAI GPT-2 使用它们来实现未来标记的掩码。由于我们在`CausalSelfAttention()`类中创建了我们的掩码并将其注册为 PyTorch 中的缓冲区，因此我们不需要从 OpenAI 加载以`attn.masked_bias`结尾的参数。

我们将从头创建的 GPT-2XL 模型中的参数命名为`sd`：

```py
sd=model.state_dict()
```

接下来，我们将从 OpenAI GPT-2XL 中提取预训练权重并将它们放置到我们自己的模型中：

```py
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
              'mlp.c_fc.weight', 'mlp.c_proj.weight']          ①
for k in keys:
    if any(k.endswith(w) for w in transposed):
        with torch.no_grad():
            sd[k].copy_(sd_hf[k].t())                          ②
    else:
        with torch.no_grad():
            sd[k].copy_(sd_hf[k])                              ③
```

① 发现 OpenAI 使用 Conv1d 模块而不是线性模块的层

② 对于这些层，我们在将权重放置到我们的模型中之前，转置权重矩阵。

③ 否则，简单地从 OpenAI 复制权重并将它们放置到我们的模型中

我们从 Hugging Face 提取 OpenAI 的预训练权重并将它们放置到我们自己的模型中。在这个过程中，我们确保每当 OpenAI 检查点使用 Conv1d 模块而不是普通线性模块时，我们都会转置权重矩阵。

现在我们的模型已经配备了来自 OpenAI 的预训练权重。我们可以使用该模型生成连贯的文本。

### 11.3.2 定义一个 generate()函数以生成文本

借助来自 OpenAI GPT-2XL 模型的预训练权重，我们将使用我们从头创建的 GPT2 模型来生成文本。

在生成文本时，我们将一个与提示中的标记对应的索引序列输入到模型中。模型预测下一个标记对应的索引，并将预测附加到序列的末尾以形成新的序列。然后它使用新的序列再次进行预测。它一直这样做，直到模型生成固定数量的新标记或对话结束（由特殊标记`<|endoftext|>`表示）。

GPT 中的特殊标记<|endoftext|>

GPT 模型使用来自各种来源的文本进行训练。在这个阶段，使用一个独特的标记`<|endoftext|>`来区分不同来源的文本。在文本生成阶段，在遇到这个特殊标记时停止对话至关重要。如果不这样做，可能会触发无关新话题的启动，导致随后生成的文本与当前讨论无关。

为了达到这个目的，我们定义了一个`sample()`函数，向当前序列中添加一定数量的新索引。它接受一个索引序列作为输入，以供 GPT-2XL 模型使用。它一次预测一个索引，并将新索引添加到运行序列的末尾。它停止直到达到指定的步数`max_new_tokens`或预测的下一个标记是`<|endoftext|>`，这表示对话结束。如果我们不停下来，模型可能会随机开始一个无关的话题。`sample()`函数的定义如下所示。

列表 11.7 逐个预测下一个索引

```py
model.eval()
def sample(idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):                            ①
        if idx.size(1) <= config.block_size:
            idx_cond = idx  
        else:
            idx_cond = idx[:, -config.block_size:]
        logits, _ = model(idx_cond)                            ②
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')        ③
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next.item()==tokenizer.encoder.encoder['<|endoftext|>']:
            break                                              ④
        idx = torch.cat((idx, idx_next), dim=1)                ⑤
    return idx
```

① 生成固定数量的新索引

② 使用 GPT-2XL 预测下一个索引

③ 如果使用 top-K 采样，将低于 top K 选择的 logits 设置为-∞

④ 如果下一个标记是<|endoftext|>，则停止预测

⑤ 将新的预测附加到序列的末尾

`sample()`函数使用 GPT-2XL 向运行序列中添加新的索引。它包含两个参数，`temperature`和`top_k`，以调节生成输出的新颖性，其工作方式与第八章中描述的相同。该函数返回一个新的索引序列。

接下来，我们定义一个`generate()`函数，根据提示（prompt）生成文本。它首先将提示中的文本转换为一系列索引。然后，它将这个序列输入到我们刚刚定义的`sample()`函数中，以生成一个新的索引序列。最后，`generate()`函数将新的索引序列转换回文本。

列表 11.8：使用 GPT-2XL 生成文本的函数

```py
def generate(prompt, max_new_tokens, temperature=1.0,
             top_k=None):
    if prompt == '':
        x=torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]],
                         dtype=torch.long)                     ①
    else:
        x = tokenizer(prompt)                                  ②
    y = sample(x, max_new_tokens, temperature, top_k)          ③
    out = tokenizer.decode(y.squeeze())                        ④
    print(out)
```

① 如果提示为空，则使用<|endoftext|>作为提示

② 将提示转换为一系列索引

③ 使用 sample()函数生成新的索引

④ 将新的索引序列转换回文本

`generate()`函数与我们在第八章中介绍的那个版本相似，但有一个显著的区别：它使用 GPT-2XL 进行预测，远离之前使用的 LSTM 模型。该函数接受一个提示作为其初始输入，将这个提示转换成一系列索引，然后输入到模型中预测后续的索引。在生成预定数量的新索引后，该函数将整个索引序列转换回文本形式。

### 11.3.3 使用 GPT-2XL 进行文本生成

现在我们已经定义了`generate()`函数，我们可以使用它来生成文本。

尤其是使用`generate()`函数可以进行无条件文本生成，这意味着提示（prompt）为空。模型将随机生成文本。这在创意写作中可能很有用：生成的文本可以用作灵感或个人创意作品的起点。让我们试试看：

```py
prompt=""
torch.manual_seed(42)
generate(prompt, max_new_tokens=100, temperature=1.0,
             top_k=None)
```

输出是

```py
<|endoftext|>Feedback from Ham Radio Recalls

I discovered a tune sticking in my head -- I'd heard it mentioned on several occasions, but hadn't investigated further.

The tune sounded familiar to a tune I'd previously heard on the 550 micro. 
During that same time period I've heard other people's receipients drone on
the idea of the DSH-94013, notably Kim Weaver's instructions in her 
Interview on Radio Ham; and both Scott Mcystem and Steve Simmons' concepts.
```

如您所见，前面的输出在逻辑上是一致的，语法正确，但可能不是事实准确的。我快速进行了谷歌搜索，文本似乎并未从任何在线来源复制。

练习 11.4

通过将提示设置为空字符串，温度设置为 0.9，最大新标记数量设置为 100，`top_k`设置为 40，并在 PyTorch 中将随机种子数设置为 42，无条件生成文本。看看输出结果是什么。

为了评估 GPT-2XL 能否根据前面的标记生成连贯的文本，我们将使用提示“I went to the kitchen and”并在提示后生成 10 个额外的标记。我们将重复这个过程五次，以确定生成的文本是否与典型的厨房活动相符：

```py
prompt="I went to the kitchen and"
for i in range(5):
    torch.manual_seed(i)
    generate(prompt, max_new_tokens=10, temperature=1.0,
                 top_k=None)
```

输出是

```py
I went to the kitchen and said, you're not going to believe this.
I went to the kitchen and noticed a female producer open a drawer in which was
I went to the kitchen and asked who was going to be right there and A
I went to the kitchen and took a small vial of bourbon and a little
I went to the kitchen and found the bottle of wine, and poured it into
```

这些结果表明，生成的文本包括与某人交谈、注意到某事以及饮用饮料等活动，这些都是典型的厨房活动。这证明了 GPT-2XL 可以生成与给定上下文相关的文本。

接下来，我们使用“Lexington 是肯塔基州第二大城市”作为提示，并要求`generate()`函数添加多达 100 个新标记：

```py
prompt="Lexington is the second largest city in the state of Kentucky"
torch.manual_seed(42)
generate(prompt, max_new_tokens=100, temperature=1.0,
             top_k=None)
```

输出结果为

```py
Lexington is the second largest city in the state of Kentucky. It caters to
those who want to make everything in tune with being with friends and 
enjoying a jaunt through the down to Earth lifestyle. To do so, they are 
blessed with several venues large and small to fill their every need while 
residing micro- cozy with nature within the landmarks of the city.

In a moment we look at ten up and coming suchache music acts from the 
Lexington area to draw upon your attention.

Lyrikhop

This Lexington-based group
```

再次，这段文本是连贯的。尽管生成的文本可能不是事实准确的。GPT-2XL 模型在本质上是被训练来根据句子中的前一个标记预测下一个标记的。前一个输出显示，该模型已经达到了这个目标：生成的文本在语法上是正确的，看起来似乎是逻辑的。它显示了在序列的早期部分记住文本并生成与上下文相关的后续单词的能力。例如，当第一句话讨论列克星敦市时，大约 90 个标记后，模型提到了列克星敦地区的音乐表演。

此外，正如引言中提到的，GPT-2 有其局限性。鉴于其大小小于 ChatGPT 的 1%和 GPT-4 的 0.1%，它不应被要求与 ChatGPT 或 GPT-4 保持相同的标准。GPT-3 有 1750 亿参数，生成的文本比 GPT-2 更连贯，但预训练的权重并未向公众发布。

接下来，我们将探讨`temperature`和`top-K`采样如何影响 GPT-2XL 生成的文本。我们将`temperature`设置为 0.9，`top_k`设置为 50，并保持其他参数不变。让我们看看生成的文本是什么样的：

```py
torch.manual_seed(42)
generate(prompt, max_new_tokens=100, temperature=0.9,
             top_k=50)  
```

输出结果为

```py
Lexington is the second largest city in the state of Kentucky. It is also 
the state capital. The population of Lexington was 1,731,947 in the 2011 
Census. The city is well-known for its many parks, including Arboretum, 
Zoo, Aquarium and the Kentucky Science Center, as well as its restaurants, 
such as the famous Kentucky Derby Festival.

In the United States, there are at least 28 counties in this state with a
population of more than 100,000, according to the 2010 census.
```

生成的文本看起来比以前更连贯。然而，内容在事实上并不准确。它编造了许多关于肯塔基州列克星敦市的事实，例如“2011 年人口普查中，列克星敦的人口为 1,731,947。”

练习 11.5

通过将`temperature`设置为 1.2 和`top_k`设置为 None，并使用“Lexington 是肯塔基州第二大城市”作为起始提示来生成文本。在 PyTorch 中将随机种子数设置为 42，并将最大新标记数设置为 100。

在本章中，你从头开始学习了如何构建 GPT-2，它是 ChatGPT 和 GPT-4 的前身。之后，你从 OpenAI 发布的 GPT-2XL 模型中提取了预训练的权重，并将它们加载到你的模型中。你见证了模型生成的连贯文本。

由于 GPT-2XL 模型（15 亿参数）的体积庞大，没有超级计算设施就无法训练该模型。在下一章中，你将创建一个与 GPT-2 结构相似但只有约 512 万个参数的小型 GPT 模型。你将使用欧内斯特·海明威的小说文本来训练模型。训练后的模型将生成与海明威风格相匹配的连贯文本！

## 摘要

+   GPT-2 是 OpenAI 开发的高级 LLM，于 2019 年 2 月宣布。它在自然语言处理领域取得了重大突破，并为开发更复杂的模型铺平了道路，包括其继任者 ChatGPT 和 GPT-4。

+   GPT-2 是一个仅包含解码器的 Transformer 模型，这意味着模型中没有编码器堆栈。与其他 Transformer 模型一样，GPT-2 使用自注意力机制并行处理输入数据，显著提高了训练大型语言模型（LLMs）的效率和效果。

+   GPT-2 在位置编码方面采用了与 2017 年开创性论文“Attention Is All You Need”中使用的不同方法。相反，GPT-2 的位置编码技术与词嵌入技术相平行。

+   GPT-2 的前馈子层中使用了 GELU 激活函数。GELU 提供了线性和非线性激活特性的混合，这些特性被发现可以增强深度学习任务中的模型性能，尤其是在自然语言处理（NLPs）和训练 LLMs 方面。

+   我们可以从头开始构建一个 GPT-2 模型，并加载 OpenAI 发布的预训练权重。你创建的 GPT-2 模型可以生成与原始 OpenAI GPT-2 模型一样连贯的文本。
