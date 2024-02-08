# 第九章 变压器

我们在第五章中看到，我们可以使用循环神经网络（RNNs）（如 LSTM 和 GRU）在文本数据上构建生成模型。这些自回归模型一次处理一个令牌的顺序数据，不断更新一个捕获输入当前潜在表示的隐藏向量。可以设计 RNN 以通过在隐藏向量上应用密集层和 softmax 激活来预测序列中的下一个单词。直到 2017 年，这被认为是生成文本的最复杂方式，当一篇论文永久改变了文本生成的格局。

# 介绍

谷歌 Brain 的论文，自信地命名为“注意力就是一切”^(1)，因推广*注意力*的概念而闻名，这个概念现在驱动着大多数最先进的文本生成模型。

作者展示了如何创建称为*变压器*的强大神经网络，用于顺序建模，而不需要复杂的循环或卷积架构，而只依赖于注意机制。这种方法克服了 RNN 方法的一个关键缺点，即难以并行化，因为它必须一次处理一个令牌的序列。变压器是高度可并行化的，使它们能够在大规模数据集上进行训练。

在本章中，我们将深入探讨现代文本生成模型如何利用 Transformer 架构在文本生成挑战中达到最先进的性能。特别是，我们将探索一种称为*生成式预训练变压器*（GPT）的自回归模型，它驱动着 OpenAI 的 GPT-4 模型，被广泛认为是当前文本生成领域的最先进技术。

# GPT

OpenAI 于 2018 年 6 月推出了 GPT，在论文“通过生成式预训练改进语言理解”中^(2)，几乎与原始 Transformer 论文出现一年后完全一致。

在本文中，作者展示了如何训练 Transformer 架构以预测序列中的下一个单词，然后随后对特定下游任务进行微调。

GPT 的预训练过程涉及在名为 BookCorpus 的大型文本语料库上训练模型（来自不同流派的 7,000 本未发表书籍的 4.5 GB 文本）。在预训练期间，模型被训练以预测给定前面单词的序列中的下一个单词。这个过程被称为*语言建模*，用于教导模型理解自然语言的结构和模式。

在预训练之后，GPT 模型可以通过提供较小的、特定于任务的数据集来进行微调以适应特定任务。微调涉及调整模型的参数以更好地适应手头的任务。例如，模型可以针对分类、相似性评分或问题回答等任务进行微调。

自 GPT 架构推出以来，OpenAI 通过发布后续模型如 GPT-2、GPT-3、GPT-3.5 和 GPT-4 对其进行了改进和扩展。这些模型在更大的数据集上进行训练，并具有更大的容量，因此可以生成更复杂和连贯的文本。研究人员和行业从业者广泛采用了 GPT 模型，并为自然语言处理任务的重大进展做出了贡献。

在本章中，我们将构建我们自己的变体 GPT 模型，该模型在较少数据上进行训练，但仍利用相同的组件和基本原则。

# 运行此示例的代码

此示例的代码可以在位于书籍存储库中的 Jupyter 笔记本中找到，位置为*notebooks/09_transformer/01_gpt/gpt.ipynb*。

该代码改编自由 Apoorv Nandan 创建的优秀[GPT 教程](https://oreil.ly/J86pg)，该教程可在 Keras 网站上找到。

## 葡萄酒评论数据集

我们将使用通过 Kaggle 提供的[Wine Reviews 数据集](https://oreil.ly/DC9EG)。这是一个包含超过 130,000 条葡萄酒评论的数据集，附带元数据，如描述和价格。

您可以通过在书库中运行 Kaggle 数据集下载脚本来下载数据集，如示例 9-1 所示。这将把葡萄酒评论和相关元数据保存在本地的*/data*文件夹中。

##### 示例 9-1\. 下载葡萄酒评论数据集

```py
bash scripts/download_kaggle_data.sh zynicide wine-reviews
```

`数据准备步骤与第五章中用于准备输入到 LSTM 的数据的步骤是相同的，因此我们不会在这里详细重复它们。如图 9-1 所示，步骤如下：

1.  加载数据并创建每种葡萄酒的文本字符串描述列表。

1.  用空格填充标点符号，以便每个标点符号被视为一个单独的单词。

1.  通过`TextVectorization`层将字符串传递，对数据进行标记化，并将每个字符串填充/裁剪到固定长度。

1.  创建一个训练集，其中输入是标记化的文本字符串，输出是预测的相同字符串向后移动一个标记。

![](img/gdl2_0901.png)

###### 图 9-1\. Transformer 的数据处理`  `## 注意力

了解 GPT 如何工作的第一步是了解*注意力机制*的工作原理。这个机制是使 Transformer 架构与循环方法在语言建模方面独特和不同的地方。当我们对注意力有了扎实的理解后，我们将看到它如何在 GPT 等 Transformer 架构中使用。

当您写作时，句子中下一个词的选择受到您已经写过的其他单词的影响。例如，假设您开始一个句子如下：

```py
The pink elephant tried to get into the car but it was too
```

显然，下一个词应该是与*big*同义的。我们怎么知道这一点？

句子中的某些其他单词对帮助我们做出决定很重要。例如，它是大象而不是树懒，意味着我们更喜欢*big*而不是*slow*。如果它是游泳池而不是汽车，我们可能会选择*scared*作为*big*的一个可能替代。最后，*getting into*汽车的行为意味着大小是问题所在——如果大象试图*压扁*汽车，我们可能会选择*fast*作为最后一个词，现在*it*指的是汽车。

句子中的其他单词一点都不重要。例如，大象是粉红色这个事实对我们选择最终词汇没有影响。同样，句子中的次要单词（*the*、*but*、*it*等）给句子以语法形式，但在这里并不重要，以确定所需形容词。

换句话说，我们正在*关注*句子中的某些单词，而基本上忽略其他单词。如果我们的模型也能做同样的事情，那不是很好吗？

Transformer 中的注意力机制（也称为*注意力头*）旨在做到这一点。它能够决定从输入的哪个位置提取信息，以有效地提取有用信息而不被无关细节混淆。这使得它非常适应各种情况，因为它可以在推断时决定在哪里寻找信息。

相比之下，循环层试图建立一个捕捉每个时间步输入的整体表示的通用隐藏状态。这种方法的一个弱点是，已经合并到隐藏向量中的许多单词对当前任务（例如，预测下一个单词）并不直接相关，正如我们刚刚看到的。注意力头不会遇到这个问题，因为它们可以选择如何从附近的单词中组合信息，具体取决于上下文。

## 查询、键和值

那么，注意力头如何决定在哪里查找信息呢？在深入细节之前，让我们以高层次的方式探讨它是如何工作的，使用我们的*粉色大象*示例。

想象一下，我们想预测跟在单词*too*后面的是什么。为了帮助完成这个任务，其他前面的单词发表意见，但他们的贡献受到他们对自己预测跟在*too*后面的单词的信心程度的加权。例如，单词*elephant*可能自信地贡献说，它更有可能是与大小或响度相关的单词，而单词*was*没有太多可以提供来缩小可能性。

换句话说，我们可以将注意力头视为一种信息检索系统，其中一个“查询”（“后面跟着什么词？”）被转换为一个*键/值*存储（句子中的其他单词），输出结果是值的加权和，权重由查询和每个键之间的*共鸣*决定。

我们现在将详细介绍这个过程（图 9-2），再次参考我们的*粉色大象*句子。

![](img/gdl2_0902.png)

###### 图 9-2。注意力头的机制

*查询*（<math alttext="upper Q"><mi>Q</mi></math>）可以被视为当前任务的表示（例如，“后面跟着什么词？”）。在这个例子中，它是从单词*too*的嵌入中导出的，通过将其通过权重矩阵<math alttext="upper W Subscript upper Q"><msub><mi>W</mi> <mi>Q</mi></msub></math>传递来将向量的维度从<math alttext="d Subscript e"><msub><mi>d</mi> <mi>e</mi></msub></math>更改为<math alttext="d Subscript k"><msub><mi>d</mi> <mi>k</mi></msub></math>。

*键*向量（<math alttext="upper K"><mi>K</mi></math>）是句子中每个单词的表示——您可以将这些视为每个单词可以帮助的预测任务的描述。它们以类似的方式导出查询，通过将每个嵌入通过权重矩阵<math alttext="upper W Subscript upper K"><msub><mi>W</mi> <mi>K</mi></msub></math>传递来将每个向量的维度从<math alttext="d Subscript e"><msub><mi>d</mi> <mi>e</mi></msub></math>更改为<math alttext="d Subscript k"><msub><mi>d</mi> <mi>k</mi></msub></math>。请注意，键和查询具有相同的长度（<math alttext="d Subscript k"><msub><mi>d</mi> <mi>k</mi></msub></math>）。

在注意力头内部，每个键与查询之间的向量对之间使用点积进行比较（<math alttext="upper Q upper K Superscript upper T"><mrow><mi>Q</mi> <msup><mi>K</mi> <mi>T</mi></msup></mrow></math>）。这就是为什么键和查询必须具有相同的长度。对于特定的键/查询对，这个数字越高，键与查询的共鸣就越强，因此它可以更多地对注意力头的输出做出贡献。结果向量被缩放为<math alttext="StartRoot d Subscript k Baseline EndRoot"><msqrt><msub><mi>d</mi> <mi>k</mi></msub></msqrt></math>，以保持向量和的方差稳定（大约等于 1），并且应用 softmax 以确保贡献总和为 1。这是一个*注意力权重*向量。

*值*向量（<math alttext="upper V"><mi>V</mi></math>）也是句子中单词的表示——您可以将这些视为每个单词的未加权贡献。它们通过将每个嵌入通过权重矩阵<math alttext="upper W Subscript upper V"><msub><mi>W</mi> <mi>V</mi></msub></math>传递来导出，以将每个向量的维度从<math alttext="d Subscript e"><msub><mi>d</mi> <mi>e</mi></msub></math>更改为<math alttext="d Subscript v"><msub><mi>d</mi> <mi>v</mi></msub></math>。请注意，值向量不一定要与键和查询具有相同的长度（但通常为了简单起见）。

值向量乘以注意力权重，给出给定<math alttext="upper Q"><mi>Q</mi></math>，<math alttext="upper K"><mi>K</mi></math>和<math alttext="upper V"><mi>V</mi></math>的*注意力*，如方程 9-1 所示。

##### 方程 9-1。注意力方程

<math alttext="StartLayout 1st Row  upper A t t e n t i o n left-parenthesis upper Q comma upper K comma upper V right-parenthesis equals s o f t m a x left-parenthesis StartFraction upper Q upper K Superscript upper T Baseline Over StartRoot d Subscript k Baseline EndRoot EndFraction right-parenthesis upper V EndLayout" display="block"><mtable displaystyle="true"><mtr><mtd columnalign="right"><mrow><mi>A</mi> <mi>t</mi> <mi>t</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>i</mi> <mi>o</mi> <mi>n</mi> <mrow><mo>(</mo> <mi>Q</mi> <mo>,</mo> <mi>K</mi> <mo>,</mo> <mi>V</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>s</mi> <mi>o</mi> <mi>f</mi> <mi>t</mi> <mi>m</mi> <mi>a</mi> <mi>x</mi> <mrow><mo>(</mo> <mfrac><mrow><mi>Q</mi><msup><mi>K</mi> <mi>T</mi></msup></mrow> <msqrt><msub><mi>d</mi> <mi>k</mi></msub></msqrt></mfrac> <mo>)</mo></mrow> <mi>V</mi></mrow></mtd></mtr></mtable></math>

从注意力头中获取最终输出向量，将注意力求和得到长度为<math alttext="d Subscript v"><msub><mi>d</mi> <mi>v</mi></msub></math>的向量。这个*上下文向量*捕捉了句子中单词对于预测接下来的单词是什么的任务的混合意见。

## 多头注意力

没有理由只停留在一个注意力头上！在 Keras 中，我们可以构建一个`MultiHeadAttention`层，将多个注意力头的输出连接起来，使每个头学习不同的注意力机制，从而使整个层能够学习更复杂的关系。

连接的输出通过一个最终的权重矩阵<math alttext="upper W Subscript upper O"><msub><mi>W</mi> <mi>O</mi></msub></math>传递，将向量投影到所需的输出维度，这在我们的情况下与查询的输入维度相同（<math alttext="d Subscript e"><msub><mi>d</mi> <mi>e</mi></msub></math>），以便层可以顺序堆叠在一起。

图 9-3 展示了一个`MultiHeadAttention`层的输出是如何构建的。在 Keras 中，我们可以简单地写下示例 9-2 中显示的代码来创建这样一个层。

##### 示例 9-2。在 Keras 中创建一个`MultiHeadAttention`层

```py
layers.MultiHeadAttention(
    num_heads = 4, ![1](img/1.png)
    key_dim = 128, ![2](img/2.png)
    value_dim = 64, ![3](img/3.png)
    output_shape = 256 ![4](img/4.png)
    )
```

![1](img/#co_transformers_CO1-1)

这个多头注意力层有四个头。

![2](img/#co_transformers_CO1-2)

键（和查询）是长度为 128 的向量。

![3](img/#co_transformers_CO1-3)

值（因此也是每个头的输出）是长度为 64 的向量。

![4](img/#co_transformers_CO1-4)

输出向量的长度为 256。

![](img/gdl2_0903.png)

###### 图 9-3。一个具有四个头的多头注意力层

## 因果掩码

到目前为止，我们假设我们的注意力头的查询输入是一个单一的向量。然而，在训练期间为了效率，我们理想情况下希望注意力层能够一次操作输入中的每个单词，为每个单词预测接下来的单词。换句话说，我们希望我们的 GPT 模型能够并行处理一组查询向量（即一个矩阵）。

您可能会认为我们可以将向量批量处理成一个矩阵，让线性代数处理剩下的部分。这是正确的，但我们需要一个额外的步骤——我们需要对查询/键的点积应用一个掩码，以避免未来单词的信息泄漏。这被称为*因果掩码*，在图 9-4 中显示。

![](img/gdl2_0904.png)

###### 图 9-4。对一批输入查询计算注意力分数的矩阵，使用因果注意力掩码隐藏对查询不可用的键（因为它们在句子中后面）

如果没有这个掩码，我们的 GPT 模型将能够完美地猜测句子中的下一个单词，因为它将使用单词本身的键作为特征！创建因果掩码的代码显示在示例 9-3 中，结果的`numpy`数组（转置以匹配图表）显示在图 9-5 中。

##### 示例 9-3。因果掩码函数

```py
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)

np.transpose(causal_attention_mask(1, 10, 10, dtype = tf.int32)[0])
```

![](img/gdl2_0905.png)

###### 图 9-5。作为`numpy`数组的因果掩码——1 表示未掩码，0 表示掩码

###### 提示

因果掩码仅在*解码器 Transformer*（如 GPT）中需要，其中任务是根据先前的标记顺序生成标记。在训练期间屏蔽未来标记因此至关重要。

其他类型的 Transformer（例如*编码器 Transformer*）不需要因果掩码，因为它们不是训练来预测下一个标记。例如，Google 的 BERT 预测给定句子中的掩码单词，因此它可以使用单词之前和之后的上下文。^(3)

我们将在本章末尾更详细地探讨不同类型的 Transformer。

这结束了我们对存在于所有 Transformer 中的多头注意力机制的解释。令人惊讶的是，这样一个有影响力的层的可学习参数仅由每个注意力头的三个密集连接权重矩阵（<math alttext="upper W Subscript upper Q"><msub><mi>W</mi> <mi>Q</mi></msub></math>，<math alttext="upper W Subscript upper K"><msub><mi>W</mi> <mi>K</mi></msub></math>，<math alttext="upper W Subscript upper V"><msub><mi>W</mi> <mi>V</mi></msub></math>）和一个进一步的权重矩阵来重塑输出（<math alttext="upper W Subscript upper O"><msub><mi>W</mi> <mi>O</mi></msub></math>）。在多头注意力层中完全没有卷积或循环机制！

接下来，我们将退一步，看看多头注意力层如何形成更大组件的一部分，这个组件被称为*Transformer 块*。

## Transformer 块

*Transformer 块*是 Transformer 中的一个单一组件，它应用一些跳跃连接、前馈（密集）层和在多头注意力层周围的归一化。Transformer 块的示意图显示在图 9-6 中。

![](img/gdl2_0906.png)

###### 图 9-6。一个 Transformer 块

首先，注意到查询是如何在多头注意力层周围传递并添加到输出中的——这是一个跳跃连接，在现代深度学习架构中很常见。这意味着我们可以构建非常深的神经网络，不会受到梯度消失问题的困扰，因为跳跃连接提供了一个无梯度的*高速公路*，允许网络将信息向前传递而不中断。

其次，在 Transformer 块中使用*层归一化*来提供训练过程的稳定性。我们已经在本书中看到了批归一化层的作用，其中每个通道的输出被归一化为均值为 0，标准差为 1。归一化统计量是跨批次和空间维度计算的。

相比之下，在 Transformer 块中，层归一化通过计算跨通道的归一化统计量来归一化批次中每个序列的每个位置。就归一化统计量的计算方式而言，它与批归一化完全相反。显示批归一化和层归一化之间差异的示意图显示在图 9-7 中。

![](img/gdl2_0907.png)

###### 图 9-7。层归一化与批归一化——归一化统计量是跨蓝色单元计算的（来源：[Sheng 等人，2020](https://arxiv.org/pdf/2003.07845.pdf))^(4)

# 层归一化与批归一化

层归一化在原始 GPT 论文中使用，并且通常用于基于文本的任务，以避免在批次中的序列之间创建归一化依赖关系。然而，最近的工作，如 Shen 等人的挑战了这一假设，显示通过一些调整，一种形式的批归一化仍然可以在 Transformer 中使用，胜过更传统的层归一化。

最后，在 Transformer 块中包含了一组前馈（即密集连接）层，以允许组件在网络深入时提取更高级别的特征。

在 Keras 中展示了一个 Transformer 块的实现，详见示例 9-4。

##### 示例 9-4。Keras 中的`TransformerBlock`层

```py
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate=0.1): ![1](img/1.png)
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attn = layers.MultiHeadAttention(
            num_heads, key_dim, output_shape = embed_dim
        )
        self.dropout_1 = layers.Dropout(self.dropout_rate)
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = layers.Dense(self.ff_dim, activation="relu")
        self.ffn_2 = layers.Dense(self.embed_dim)
        self.dropout_2 = layers.Dropout(self.dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(
            batch_size, seq_len, seq_len, tf.bool
        ) ![2](img/2.png)
        attention_output, attention_scores = self.attn(
            inputs,
            inputs,
            attention_mask=causal_mask,
            return_attention_scores=True
        ) ![3](img/3.png)
        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(inputs + attention_output) ![4](img/4.png)
        ffn_1 = self.ffn_1(out1) ![5](img/5.png)
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        return (self.ln_2(out1 + ffn_output), attention_scores) ![6](img/6.png)
```

![1](img/#co_transformers_CO2-1)

构成`TransformerBlock`层的子层在初始化函数中定义。

![2](img/#co_transformers_CO2-2)

因果掩码被创建用来隐藏查询中的未来键。

![3](img/#co_transformers_CO2-3)

创建了多头注意力层，并指定了注意力掩码。

![4](img/#co_transformers_CO2-4)

第一个*加和归一化*层。

![5](img/#co_transformers_CO2-5)

前馈层。

![6](img/#co_transformers_CO2-6)

第二个*加和归一化*层。

## 位置编码

在我们能够将所有内容整合在一起训练我们的 GPT 模型之前，还有一个最后的步骤要解决。您可能已经注意到，在多头注意力层中，没有任何关心键的顺序的内容。每个键和查询之间的点积是并行计算的，而不是像递归神经网络那样顺序计算。这是一种优势（因为并行化效率提高），但也是一个问题，因为我们显然需要注意力层能够预测以下两个句子的不同输出：

+   狗看着男孩然后…（叫？）

+   男孩看着狗然后…（微笑？）

为了解决这个问题，我们在创建初始 Transformer 块的输入时使用一种称为*位置编码*的技术。我们不仅使用*标记嵌入*对每个标记进行编码，还使用*位置嵌入*对标记的位置进行编码。

*标记嵌入*是使用标准的`Embedding`层创建的，将每个标记转换为一个学习到的向量。我们可以以相同的方式创建*位置嵌入*，使用标准的`Embedding`层将每个整数位置转换为一个学习到的向量。

###### 提示

虽然 GPT 使用`Embedding`层来嵌入位置，但原始 Transformer 论文使用三角函数——我们将在第十一章中介绍这种替代方法，当我们探索音乐生成时。

为构建联合标记-位置编码，将标记嵌入加到位置嵌入中，如图 9-8 所示。这样，序列中每个单词的含义和位置都被捕捉在一个向量中。

![](img/gdl2_0908.png)

###### 图 9-8\. 将标记嵌入添加到位置嵌入以给出标记位置编码

定义我们的`TokenAndPositionEmbedding`层的代码显示在示例 9-5 中。

##### 示例 9-5\. `TokenAndPositionEmbedding`层

```py
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size =vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        ) ![1](img/1.png)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim) ![2](img/2.png)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions ![3](img/3.png)
```

![1](img/#co_transformers_CO3-1)

标记使用`Embedding`层进行嵌入。

![2](img/#co_transformers_CO3-2)

标记的位置也使用`Embedding`层进行嵌入。

![3](img/#co_transformers_CO3-3)

该层的输出是标记和位置嵌入的总和。

## 训练 GPT

现在我们准备构建和训练我们的 GPT 模型！为了将所有内容整合在一起，我们需要将输入文本通过标记和位置嵌入层，然后通过我们的 Transformer 块。网络的最终输出是一个简单的具有 softmax 激活函数的`Dense`层，覆盖词汇表中的单词数量。

###### 提示

为简单起见，我们将只使用一个 Transformer 块，而不是论文中的 12 个。

整体架构显示在图 9-9 中，相应的代码在示例 9-6 中提供。

![](img/gdl2_0909.png)

###### 图 9-9\. 简化的 GPT 模型架构

##### 示例 9-6\. 在 Keras 中的 GPT 模型

```py
MAX_LEN = 80
VOCAB_SIZE = 10000
EMBEDDING_DIM = 256
N_HEADS = 2
KEY_DIM = 256
FEED_FORWARD_DIM = 256

inputs = layers.Input(shape=(None,), dtype=tf.int32) ![1](img/1.png)
x = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM)(inputs) ![2](img/2.png)
x, attention_scores = TransformerBlock(
    N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM
)(x) ![3](img/3.png)
outputs = layers.Dense(VOCAB_SIZE, activation = 'softmax')(x) ![4](img/4.png)
gpt = models.Model(inputs=inputs, outputs=[outputs, attention]) ![5](img/5.png)
gpt.compile("adam", loss=[losses.SparseCategoricalCrossentropy(), None]) ![6](img/6.png)
gpt.fit(train_ds, epochs=5)
```

![1](img/#co_transformers_CO4-1)

输入被填充（用零填充）。

![2](img/#co_transformers_CO4-2)

文本使用`TokenAndPositionEmbedding`层进行编码。

![3](img/#co_transformers_CO4-3)

编码通过`TransformerBlock`传递。

![4](img/#co_transformers_CO4-4)

转换后的输出通过具有 softmax 激活的`Dense`层传递，以预测后续单词的分布。

![5](img/#co_transformers_CO4-5)

`Model`以单词标记序列作为输入，并输出预测的后续单词分布。还返回了 Transformer 块的输出，以便我们可以检查模型如何引导其注意力。

![6](img/#co_transformers_CO4-6)

模型使用预测的单词分布上的`SparseCategoricalCrossentropy`损失进行编译。

## GPT 的分析

现在我们已经编译并训练了我们的 GPT 模型，我们可以开始使用它生成长文本字符串。我们还可以询问从`TransformerBlock`输出的注意权重，以了解 Transformer 在生成过程中不同点处寻找信息的位置。

### 生成文本

我们可以通过以下过程生成新文本：

1.  将现有单词序列馈送到网络中，并要求它预测接下来的单词。

1.  将此单词附加到现有序列并重复。

网络将为每个单词输出一组概率，我们可以从中进行抽样，因此我们可以使文本生成具有随机性，而不是确定性。

我们将使用在第五章中引入的相同`TextGenerator`类进行 LSTM 文本生成，包括指定采样过程的确定性程度的`temperature`参数。让我们看看这在两个不同的温度值（图 9-10）下是如何运作的。

![](img/gdl2_0910.png)

###### 图 9-10。在`temperature = 1.0`和`temperature = 0.5`时生成的输出。

关于这两段文字有几点需要注意。首先，两者在风格上与原始训练集中的葡萄酒评论相似。它们都以葡萄酒的产地和类型开头，而葡萄酒类型在整个段落中保持一致（例如，它不会在中途更换颜色）。正如我们在第五章中看到的，使用温度为 1.0 生成的文本更加冒险，因此比温度为 0.5 的示例不够准确。因此，使用温度为 1.0 生成多个样本将导致更多的变化，因为模型正在从具有更大方差的概率分布中进行抽样。

### 查看注意力分数

我们还可以要求模型告诉我们在决定句子中的下一个单词时，每个单词放置了多少注意力。`TransformerBlock`输出每个头的注意权重，这是对句子中前面单词的 softmax 分布。

为了证明这一点，图 9-11 显示了三个不同输入提示的前五个具有最高概率的标记，以及两个注意力头的平均注意力，针对每个前面的单词。根据其注意力分数对前面的单词进行着色，两个注意力头的平均值。深蓝色表示对该单词放置更多的注意力。

![](img/gdl2_0911.png)

###### 图 9-11。各种序列后单词概率分布

在第一个示例中，模型密切关注国家（*德国*），以决定与地区相关的单词。这是有道理的！为了选择一个地区，它需要从与国家相关的单词中获取大量信息，以确保它们匹配。它不需要太关注前两个标记（*葡萄酒评论*），因为它们不包含有关地区的任何有用信息。

在第二个例子中，它需要参考葡萄（*雷司令*），因此它关注第一次提到它的时间。它可以通过直接关注这个词来提取这个信息，无论这个词在句子中有多远（在 80 个单词的上限内）。请注意，这与递归神经网络非常不同，后者依赖于隐藏状态来维护整个序列的所有有趣信息，以便在需要时可以利用——这是一种效率低下得多的方法。

最终的序列展示了我们的 GPT 模型如何基于信息的组合选择适当的形容词的例子。这里的注意力再次集中在葡萄（*雷司令*）上，但也集中在它含有*残留糖*的事实上。由于雷司令通常是一种甜酒，而且已经提到了糖，因此将其描述为*略带甜味*而不是*略带泥土味*是有道理的。

以这种方式询问网络非常有启发性，可以准确了解它从哪里提取信息，以便对每个后续单词做出准确的决策。我强烈建议尝试玩弄输入提示，看看是否可以让模型关注句子中非常遥远的单词，以说服自己关注模型的注意力模型比传统的递归模型更具有力量！`  `# 其他 Transformer

我们的 GPT 模型是一个*解码器 Transformer*——它一次生成一个标记的文本字符串，并使用因果屏蔽只关注输入字符串中的先前单词。还有*编码器 Transformer*，它不使用因果屏蔽——相反，它关注整个输入字符串以提取输入的有意义的上下文表示。对于其他任务，比如语言翻译，还有*编码器-解码器 Transformer*，可以将一个文本字符串翻译成另一个；这种模型包含编码器 Transformer 块和解码器 Transformer 块。

表 9-1 总结了三种 Transformer 的类型，以及每种架构的最佳示例和典型用例。

表 9-1。三种 Transformer 架构

| 类型 | 示例 | 用例 |
| --- | --- | --- |
| 编码器 | BERT（谷歌） | 句子分类、命名实体识别、抽取式问答 |
| 编码器-解码器 | T5（谷歌） | 摘要、翻译、问答 |
| 解码器 | GPT-3（OpenAI） | 文本生成 |

一个众所周知的编码器 Transformer 的例子是谷歌开发的*双向编码器表示来自 Transformer*（BERT）模型，它可以根据缺失单词的上下文预测句子中的缺失单词（Devlin 等，2018）。

# 编码器 Transformer

编码器 Transformer 通常用于需要全面理解输入的任务，比如句子分类、命名实体识别和抽取式问答。它们不用于文本生成任务，因此我们不会在本书中详细探讨它们——有关更多信息，请参阅 Lewis Tunstall 等人的[*使用 Transformer 进行自然语言处理*](https://www.oreilly.com/library/view/natural-language-processing/9781098136789)（O'Reilly）。

在接下来的章节中，我们将探讨编码器-解码器 Transformer 的工作原理，并讨论 OpenAI 发布的原始 GPT 模型架构的扩展，包括专门为对话应用设计的 ChatGPT。

## T5

一个使用编码器-解码器结构的现代 Transformer 的例子是谷歌的 T5 模型。这个模型将一系列任务重新构建为文本到文本的框架，包括翻译、语言可接受性、句子相似性和文档摘要，如图 9-12 所示。

![](img/gdl2_0912.png)

###### 图 9-12。T5 如何将一系列任务重新构建为文本到文本框架的示例，包括翻译、语言可接受性、句子相似性和文档摘要（来源：[Raffel et al., 2019](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)）

T5 模型架构与原始 Transformer 论文中使用的编码器-解码器架构非常相似，如图 9-13 所示。关键区别在于 T5 是在一个庞大的 750GB 文本语料库（Colossal Clean Crawled Corpus，或 C4）上进行训练的，而原始 Transformer 论文仅关注语言翻译，因此它是在 1.4GB 的英德句对上进行训练的。

![](img/gdl2_0913.png)

###### 图 9-13。编码器-解码器 Transformer 模型：每个灰色框是一个 Transformer 块（来源：[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)）

这个图表中的大部分内容对我们来说已经很熟悉了——我们可以看到 Transformer 块被重复，并且使用位置嵌入来捕捉输入序列的顺序。这个模型与我们在本章前面构建的 GPT 模型之间的两个关键区别如下：

+   在左侧，一组*编码器*Transformer 块对待翻译的序列进行编码。请注意，注意力层上没有因果屏蔽。这是因为我们不生成更多文本来扩展要翻译的序列；我们只想学习一个可以提供给解码器的整个序列的良好表示。因此，编码器中的注意力层可以完全不加屏蔽，以捕捉单词之间的所有交叉依赖关系，无论顺序如何。

+   在右侧，一组*解码器*Transformer 块生成翻译文本。初始注意力层是*自指*的（即，键、值和查询来自相同的输入），并且使用因果屏蔽确保来自未来标记的信息不会泄漏到当前要预测的单词。然而，我们可以看到随后的注意力层从编码器中提取键和值，只留下查询从解码器本身传递。这被称为*交叉引用*注意力，意味着解码器可以关注输入序列的编码器表示。这就是解码器知道翻译需要传达什么含义的方式！

图 9-14 展示了一个交叉引用注意力的示例。解码器层的两个注意力头能够共同提供单词*the*的正确德语翻译，当它在*the street*的上下文中使用时。在德语中，根据名词的性别有三个定冠词（*der, die, das*），但 Transformer 知道选择*die*，因为一个注意力头能够关注单词*street*（德语中的一个女性词），而另一个关注要翻译的单词（*the*）。

![](img/gdl2_0914.png)

###### 图 9-14。一个示例，展示一个注意力头关注单词“the”，另一个关注单词“street”，以便正确将单词“the”翻译为德语单词“die”，作为“Straße”的女性定冠词

###### 提示

这个例子来自[Tensor2Tensor GitHub 存储库](https://oreil.ly/84lIA)，其中包含一个 Colab 笔记本，让您可以玩转一个经过训练的编码器-解码器 Transformer 模型，并查看编码器和解码器的注意力机制如何影响将给定句子翻译成德语。

## GPT-3 和 GPT-4

自 2018 年 GPT 的原始出版以来，OpenAI 已发布了多个更新版本，改进了原始模型，如表 9-2 所示。

表 9-2。OpenAI 的 GPT 系列模型的演变

| 模型 | 日期 | 层 | 注意力头 | 词嵌入大小 | 上下文窗口 | 参数数量 | 训练数据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 2018 年 6 月 | 12 | 12 | 768 | 512 | 120,000,000 | BookCorpus：来自未发表书籍的 4.5 GB 文本 |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 年 2 月 | 48 | 48 | 1,600 | 1,024 | 1,500,000,000 | WebText：来自 Reddit 外链的 40 GB 文本 |
| [GPT-3](https://arxiv.org/abs/2005.14165) | 2020 年 5 月 | 96 | 96 | 12,888 | 2,048 | 175,000,000,000 | CommonCrawl，WebText，英文维基百科，书籍语料库等：570 GB |
| [GPT-4](https://arxiv.org/abs/2303.08774) | 2023 年 3 月 | - | - | - | - | - | - |

GPT-3 的模型架构与原始 GPT 模型非常相似，只是规模更大，训练数据更多。在撰写本文时，GPT-4 处于有限的测试阶段——OpenAI 尚未公开发布模型的结构和规模的详细信息，尽管我们知道它能够接受图像作为输入，因此首次跨越成为多模态模型。GPT-3 和 GPT-4 的模型权重不是开源的，尽管这些模型可以通过[商业工具和 API](https://platform.openai.com)获得。

GPT-3 也可以[根据您自己的训练数据进行微调](https://oreil.ly/B-Koo)——这使您可以提供多个示例，说明它应该如何对特定风格的提示做出反应，通过物理更新网络的权重。在许多情况下，这可能是不必要的，因为 GPT-3 可以通过在提示本身提供几个示例来告诉它如何对特定风格的提示做出反应（这被称为*few-shot learning*）。微调的好处在于，您不需要在每个单独的输入提示中提供这些示例，从长远来看可以节省成本。

给定系统提示句子的 GPT-3 输出示例显示在图 9-15 中。

![](img/gdl2_0915.png)

###### 图 9-15。GPT-3 如何扩展给定系统提示的示例

诸如 GPT 之类的语言模型在规模上受益巨大——无论是模型权重的数量还是数据集的大小。大型语言模型能力的上限尚未达到，研究人员继续推动着使用越来越大的模型和数据集所能实现的边界。

## ChatGPT

在 GPT-4 的测试版发布几个月前，OpenAI 宣布了[*ChatGPT*](https://chat.openai.com)——这是一个允许用户通过对话界面与其一系列大型语言模型进行交互的工具。2022 年 11 月的原始版本由*GPT-3.5*提供支持，这个版本比 GPT-3 更强大，经过微调以进行对话回应。

示例对话显示在图 9-16 中。请注意，代理能够在输入之间保持状态，理解第二个问题中提到的*attention*指的是 Transformer 上下文中的注意力，而不是一个人的专注能力。

![](img/gdl2_0916.png)

###### 图 9-16。ChatGPT 回答有关 Transformer 的问题的示例

在撰写本文时，尚无描述 ChatGPT 工作详细信息的官方论文，但根据官方[博客文章](https://openai.com/blog/chatgpt)，我们知道它使用一种称为*reinforcement learning from human feedback*（RLHF）的技术来微调 GPT-3.5 模型。这种技术也在 ChatGPT 小组早期的论文^(6)中使用，该论文介绍了*InstructGPT*模型，这是一个经过微调的 GPT-3 模型，专门设计用于更准确地遵循书面说明。

ChatGPT 的训练过程如下：

1.  *监督微调*：收集人类编写的对话输入（提示）和期望输出的演示数据集。这用于使用监督学习微调基础语言模型（GPT-3.5）。

1.  *奖励建模*：向人类标记者展示提示的示例和几个抽样的模型输出，并要求他们将输出从最好到最差进行排名。训练一个奖励模型，预测给定对话历史的每个输出的得分。

1.  *强化学习*：将对话视为一个强化学习环境，其中*策略*是基础语言模型，初始化为从步骤 1 中微调的模型。给定当前的*状态*（对话历史），策略输出一个*动作*（一系列标记），由在步骤 2 中训练的奖励模型评分。然后可以训练一个强化学习算法——近端策略优化（PPO），通过调整语言模型的权重来最大化奖励。

# 强化学习

有关强化学习的介绍，请参阅第十二章，在那里我们探讨了生成模型如何在强化学习环境中使用。

RLHF 过程如图 9-17 所示。

![](img/gdl2_0917.png)

###### 图 9-17。ChatGPT 中使用的强化学习来自人类反馈微调过程的示意图（来源：[OpenAI](https://openai.com/blog/chatgpt)）

虽然 ChatGPT 仍然存在许多限制（例如有时“产生”事实不正确的信息），但它是一个强大的示例，展示了 Transformers 如何用于构建生成模型，可以产生复杂、长期和新颖的输出，往往难以区分是否为人类生成的文本。像 ChatGPT 这样的模型迄今取得的进展证明了人工智能的潜力及其对世界的变革性影响。

此外，显而易见的是，基于人工智能的沟通和互动将继续在未来快速发展。像*Visual ChatGPT*^(7)这样的项目现在正在将 ChatGPT 的语言能力与 Stable Diffusion 等视觉基础模型相结合，使用户不仅可以通过文本与 ChatGPT 互动，还可以通过图像。在像 Visual ChatGPT 和 GPT-4 这样的项目中融合语言和视觉能力，有望开启人机交互的新时代。

# 总结

在本章中，我们探讨了 Transformer 模型架构，并构建了一个 GPT 的版本——用于最先进文本生成的模型。

GPT 利用一种称为注意力的机制，消除了循环层（例如 LSTM）的需求。它类似于信息检索系统，利用查询、键和值来决定它想要从每个输入标记中提取多少信息。

注意力头可以组合在一起形成所谓的多头注意力层。然后将它们包装在一个 Transformer 块中，其中包括围绕注意力层的层归一化和跳过连接。Transformer 块可以堆叠以创建非常深的神经网络。

因果屏蔽用于确保 GPT 不能从下游标记泄漏信息到当前预测中。此外，还使用一种称为位置编码的技术，以确保输入序列的顺序不会丢失，而是与传统的词嵌入一起嵌入到输入中。

在分析 GPT 的输出时，我们看到不仅可以生成新的文本段落，还可以审查网络的注意力层，以了解它在句子中查找信息以改善预测的位置。GPT 可以在不丢失信号的情况下访问远处的信息，因为注意力分数是并行计算的，不依赖于通过网络顺序传递的隐藏状态，这与循环神经网络的情况不同。

我们看到了 Transformer 有三个系列（编码器、解码器和编码器-解码器）以及每个系列可以完成的不同任务。最后，我们探讨了其他大型语言模型的结构和训练过程，如谷歌的 T5 和 OpenAI 的 ChatGPT。

^(1) Ashish Vaswani 等人，“注意力就是一切”，2017 年 6 月 12 日，[*https://arxiv.org/abs/1706.03762*](https://arxiv.org/abs/1706.03762)。

^(2) Alec Radford 等人，“通过生成式预训练改进语言理解”，2018 年 6 月 11 日，[*https://openai.com/research/language-unsupervised*](https://openai.com/research/language-unsupervised)。

^(3) Jacob Devlin 等人，“BERT: 深度双向 Transformer 的语言理解预训练”，2018 年 10 月 11 日，[*https://arxiv.org/abs/1810.04805*](https://arxiv.org/abs/1810.04805)。

^(4) Sheng Shen 等人，“PowerNorm: 重新思考 Transformer 中的批归一化”，2020 年 6 月 28 日，[*https://arxiv.org/abs/2003.07845*](https://arxiv.org/abs/2003.07845)。

^(5) Colin Raffel 等人，“探索统一文本到文本 Transformer 的迁移学习极限”，2019 年 10 月 23 日，[*https://arxiv.org/abs/1910.10683*](https://arxiv.org/abs/1910.10683)。

^(6) Long Ouyang 等人，“使用人类反馈训练语言模型遵循指令”，2022 年 3 月 4 日，[*https://arxiv.org/abs/2203.02155*](https://arxiv.org/abs/2203.02155)。

^(7) Chenfei Wu 等人，“Visual ChatGPT: 使用视觉基础模型进行对话、绘画和编辑”，2023 年 3 月 8 日，[*https://arxiv.org/abs/2303.04671*](https://arxiv.org/abs/2303.04671)。
