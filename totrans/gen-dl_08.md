# 第五章。自回归模型

到目前为止，我们已经探讨了两种涉及潜变量的生成模型家族——变分自动编码器（VAEs）和生成对抗网络（GANs）。在这两种情况下，引入了一个新变量，其分布易于抽样，模型学习如何将此变量*解码*回原始领域。

现在我们将把注意力转向*自回归模型*——一类通过将生成建模问题简化为一个顺序过程的模型家族。自回归模型将预测条件放在序列中的先前值上，而不是在潜在随机变量上。因此，它们试图明确地对数据生成分布建模，而不是对其进行近似（如 VAEs 的情况）。

在本章中，我们将探讨两种不同的自回归模型：长短期记忆网络和 PixelCNN。我们将把 LSTM 应用于文本数据，将 PixelCNN 应用于图像数据。我们将在第九章中详细介绍另一个非常成功的自回归模型 Transformer。

# 介绍

为了理解 LSTM 的工作原理，我们将首先访问一个奇怪的监狱，那里的囚犯们组成了一个文学社团...​

Sopp 先生及其众包寓言的故事是对一种臭名昭著的用于文本等序列数据的自回归技术的类比：长短期记忆网络。

# 长短期记忆网络（LSTM）

LSTM 是一种特殊类型的循环神经网络（RNN）。RNN 包含一个循环层（或*单元），能够通过使其在特定时间步的输出成为下一个时间步的输入的一部分来处理序列数据。

当 RNN 首次引入时，循环层非常简单，仅包含一个 tanh 运算符，确保在时间步之间传递的信息在-1 和 1 之间缩放。然而，这种方法被证明存在梯度消失问题，并且在处理长序列数据时不具备良好的可扩展性。

LSTM 单元最初是在 1997 年由 Sepp Hochreiter 和 Jürgen Schmidhuber 的一篇论文中首次引入的。^(1)在这篇论文中，作者描述了 LSTM 不会像普通 RNN 那样遭受梯度消失问题，并且可以在数百个时间步长的序列上进行训练。自那时以来，LSTM 架构已经被改进和改良，变体如门控循环单元（本章后面讨论）现在被广泛应用并作为 Keras 中的层可用。

LSTM 已经应用于涉及序列数据的各种问题，包括时间序列预测、情感分析和音频分类。在本章中，我们将使用 LSTM 来解决文本生成的挑战。

# 运行此示例的代码

此示例的代码可以在位于书籍存储库中的 Jupyter 笔记本中找到，路径为*notebooks/05_autoregressive/01_lstm/lstm.ipynb*。

## 食谱数据集

我们将使用通过 Kaggle 提供的[Epicurious 食谱数据集](https://oreil.ly/laNUt)。这是一个包含超过 20,000 个食谱的数据集，附带有营养信息和配料清单等元数据。

您可以通过在书籍存储库中运行 Kaggle 数据集下载脚本来下载数据集，如示例 5-1 所示。这将把食谱和相关元数据保存到本地的*/data*文件夹中。

##### 示例 5-1。下载 Epicurious 食谱数据集

```py
bash scripts/download_kaggle_data.sh hugodarwood epirecipes
```

`示例 5-2 展示了如何加载和过滤数据，以便只保留具有标题和描述的食谱。示例中给出了一个食谱文本字符串，详见示例 5-3。

##### 示例 5-2。加载数据

```py
with open('/app/data/epirecipes/full_format_recipes.json') as json_data:
    recipe_data = json.load(json_data)

filtered_data = [
    'Recipe for ' + x['title']+ ' | ' + ' '.join(x['directions'])
    for x in recipe_data
    if 'title' in x
    and x['title'] is not None
    and 'directions' in x
    and x['directions'] is not None
]
```

##### 示例 5-3。来自食谱数据集的文本字符串

```py
Recipe for Ham Persillade with Mustard Potato Salad and Mashed Peas  | Chop enough
parsley leaves to measure 1 tablespoon; reserve. Chop remaining leaves and stems
and simmer with broth and garlic in a small saucepan, covered, 5 minutes.
Meanwhile, sprinkle gelatin over water in a medium bowl and let soften 1 minute.
Strain broth through a fine-mesh sieve into bowl with gelatin and stir to dissolve.
Season with salt and pepper. Set bowl in an ice bath and cool to room temperature,
stirring. Toss ham with reserved parsley and divide among jars. Pour gelatin on top
and chill until set, at least 1 hour. Whisk together mayonnaise, mustard, vinegar,
1/4 teaspoon salt, and 1/4 teaspoon pepper in a large bowl. Stir in celery,
cornichons, and potatoes. Pulse peas with marjoram, oil, 1/2 teaspoon pepper, and
1/4 teaspoon salt in a food processor to a coarse mash. Layer peas, then potato
salad, over ham.
```

在看如何在 Keras 中构建 LSTM 网络之前，我们必须先快速了解文本数据的结构以及它与本书中迄今为止看到的图像数据有何不同。## 处理文本数据

文本和图像数据之间存在几个关键差异，这意味着许多适用于图像数据的方法并不适用于文本数据。特别是：

+   文本数据由离散块（字符或单词）组成，而图像中的像素是连续色谱中的点。我们可以轻松地将绿色像素变成蓝色，但我们不清楚应该如何使单词“猫”更像单词“狗”，例如。这意味着我们可以轻松地将反向传播应用于图像数据，因为我们可以计算损失函数相对于单个像素的梯度，以确定像素颜色应该如何改变以最小化损失的方向。对于离散文本数据，我们不能明显地以同样的方式应用反向传播，因此我们需要找到解决这个问题的方法。

+   文本数据具有时间维度但没有空间维度，而图像数据具有两个空间维度但没有时间维度。文本数据中单词的顺序非常重要，单词倒过来就没有意义，而图像通常可以翻转而不影响内容。此外，单词之间通常存在长期的顺序依赖关系，模型需要捕捉这些依赖关系：例如，回答问题或延续代词的上下文。对于图像数据，所有像素可以同时处理。

+   文本数据对个体单位（单词或字符）的微小变化非常敏感。图像数据通常对个体像素单位的变化不太敏感——即使一些像素被改变，房子的图片仍然可以被识别为房子——但是对于文本数据，即使改变几个单词也可能极大地改变段落的含义，或使其毫无意义。这使得训练模型生成连贯文本非常困难，因为每个单词对段落的整体含义至关重要。

+   文本数据具有基于规则的语法结构，而图像数据不遵循有关如何分配像素值的固定规则。例如，在任何情况下写“猫坐在上面”都没有语法意义。还有一些语义规则极其难以建模；即使从语法上讲，“我在海滩上”这个陈述没有问题，但意义上是不通顺的。

# 基于文本的生成式深度学习的进展

直到最近，大多数最复杂的生成式深度学习模型都集中在图像数据上，因为前面列表中提到的许多挑战甚至超出了最先进技术的范围。然而，在过去的五年中，在基于文本的生成式深度学习领域取得了惊人的进展，这要归功于 Transformer 模型架构的引入，我们将在第九章中探讨。

考虑到这些要点，让我们现在来看看我们需要采取哪些步骤，以便将文本数据整理成适合训练 LSTM 网络的形式。

## 标记化

第一步是清理和标记化文本。标记化是将文本分割成单独的单位，如单词或字符的过程。

如何对文本进行标记化取决于您尝试使用文本生成模型实现什么目标。使用单词和字符标记都有利弊，您的选择将影响您在建模之前需要如何清理文本以及模型输出。

如果使用单词标记：

+   所有文本都可以转换为小写，以确保句子开头的大写单词与句子中间出现的相同单词以相同方式进行标记化。然而，在某些情况下，这可能不是理想的；例如，一些专有名词，如姓名或地点，可能受益于保持大写，以便它们被独立标记化。

+   文本*词汇*（训练集中不同单词的集合）可能非常庞大，有些单词可能非常稀疏，甚至可能只出现一次。将稀疏单词替换为*未知单词*的标记可能是明智的选择，而不是将它们作为单独的标记包含在内，以减少神经网络需要学习的权重数量。

+   单词可以进行*词干处理*，意味着它们被简化为最简单的形式，以便动词的不同时态保持标记化在一起。例如，*browse*、*browsing*、*browses*和*browsed*都将被词干处理为*brows*。

+   您需要将标点标记化，或者完全删除它。

+   使用单词标记化意味着模型永远无法预测训练词汇表之外的单词。

如果您使用字符标记：

+   模型可能生成字符序列，形成训练词汇表之外的新单词——在某些情况下，这可能是可取的，但在其他情况下则不是。

+   大写字母可以转换为它们的小写对应词，也可以保留为单独的标记。

+   使用字符标记时，词汇量通常较小。这对模型训练速度有益，因为最终输出层中需要学习的权重较少。

在这个示例中，我们将使用小写单词标记化，不进行词干处理。我们还将标记化标点符号，因为我们希望模型能够预测何时结束句子或使用逗号，例如。

示例 5-4 中的代码清理并标记文本。

##### 示例 5-4。标记化

```py
def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}])", r' \1 ', s)
    s = re.sub(' +', ' ', s)
    return s

text_data = [pad_punctuation(x) for x in filtered_data] ![1](img/1.png)

text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(32).shuffle(1000) ![2](img/2.png)

vectorize_layer = layers.TextVectorization( ![3](img/3.png)
    standardize = 'lower',
    max_tokens = 10000,
    output_mode = "int",
    output_sequence_length = 200 + 1,
)

vectorize_layer.adapt(text_ds) ![4](img/4.png)
vocab = vectorize_layer.get_vocabulary() ![5](img/5.png)
```

![1](img/#co_autoregressive_models_CO1-1)

填充标点符号，将它们视为单独的单词。

![2](img/#co_autoregressive_models_CO1-2)

转换为 TensorFlow 数据集。

![3](img/#co_autoregressive_models_CO1-3)

创建一个 Keras `TextVectorization`层，将文本转换为小写，为最常见的 10,000 个单词分配相应的整数标记，并将序列修剪或填充到 201 个标记长。

![4](img/#co_autoregressive_models_CO1-4)

将`TextVectorization`层应用于训练数据。

![5](img/#co_autoregressive_models_CO1-5)

`vocab`变量存储一个单词标记列表。

在标记化后，一个配方的示例显示在示例 5-5 中。我们用于训练模型的序列长度是训练过程的一个参数。在这个示例中，我们选择使用长度为 200 的序列长度，因此我们将配方填充或裁剪到比这个长度多一个，以便我们创建目标变量（在下一节中详细介绍）。为了实现这个期望的长度，向量的末尾用零填充。

# 停止标记

`0`标记被称为*停止标记*，表示文本字符串已经结束。

##### 示例 5-5。示例 5-3 中的配方进行了标记化

```py
[  26   16  557    1    8  298  335  189    4 1054  494   27  332  228
  235  262    5  594   11  133   22  311    2  332   45  262    4  671
    4   70    8  171    4   81    6    9   65   80    3  121    3   59
   12    2  299    3   88  650   20   39    6    9   29   21    4   67
  529   11  164    2  320  171  102    9  374   13  643  306   25   21
    8  650    4   42    5  931    2   63    8   24    4   33    2  114
   21    6  178  181 1245    4   60    5  140  112    3   48    2  117
  557    8  285  235    4  200  292  980    2  107  650   28   72    4
  108   10  114    3   57  204   11  172    2   73  110  482    3  298
    3  190    3   11   23   32  142   24    3    4   11   23   32  142
   33    6    9   30   21    2   42    6  353    3 3224    3    4  150
    2  437  494    8 1281    3   37    3   11   23   15  142   33    3
    4   11   23   32  142   24    6    9  291  188    5    9  412  572
    2  230  494    3   46  335  189    3   20  557    2    0    0    0
    0    0    0    0    0]
```

在示例 5-6 中，我们可以看到一部分标记列表映射到它们各自的索引。该层将`0`标记保留为填充（即停止标记），将`1`标记保留为超出前 10000 个单词的未知单词（例如，persillade）。其他单词按频率顺序分配标记。要包含在词汇表中的单词数量也是训练过程的一个参数。包含的单词越多，您在文本中看到的*未知*标记就越少；但是，您的模型需要更大以容纳更大的词汇量。

##### 示例 5-6。`TextVectorization`层的词汇表

```py
0:
1: [UNK]
2: .
3: ,
4: and
5: to
6: in
7: the
8: with
9: a
```

## 创建训练集

我们的 LSTM 将被训练以预测序列中的下一个单词，给定此点之前的一系列单词。例如，我们可以向模型提供*烤鸡配煮熟的*的标记，期望模型输出一个合适的下一个单词（例如*土豆*，而不是*香蕉*）。

因此，我们可以简单地将整个序列向后移动一个标记，以创建我们的目标变量。

数据集生成步骤可以通过示例 5-7 中的代码实现。

##### 示例 5-7。创建训练数据集

```py
def prepare_inputs(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y

train_ds = text_ds.map(prepare_inputs) ![1](img/1.png)
```

![1](img/#co_autoregressive_models_CO2-1)

创建包含食谱标记（输入）和相同向量向后移动一个标记（目标）的训练集。

## LSTM 架构

整个 LSTM 模型的架构如表 5-1 所示。模型的输入是一系列整数标记，输出是 10,000 个词汇表中每个单词在序列中出现的概率。为了详细了解这是如何工作的，我们需要介绍两种新的层类型，即`Embedding`和`LSTM`。

表 5-1。LSTM 模型的摘要

| 层（类型） | 输出形状 | 参数 # |
| --- | --- | --- |
| InputLayer | (None, None) | 0 |
| Embedding | (None, None, 100) | 1,000,000 |
| LSTM | (None, None, 128) | 117,248 |
| Dense | (None, None, 10000) | 1,290,000 |
| 总参数 | 2,407,248 |
| 可训练参数 | 2,407,248 |
| 不可训练参数 | 0 |

# LSTM 的输入层

请注意，`Input`层不需要我们提前指定序列长度。批处理大小和序列长度都是灵活的（因此形状为`(None, None)`）。这是因为所有下游层对通过的序列长度都是不可知的。

## 嵌入层

*嵌入层*本质上是一个查找表，将每个整数标记转换为长度为`embedding_size`的向量，如图 5-2 所示。模型通过*权重*学习查找向量。因此，该层学习的权重数量等于词汇表的大小乘以嵌入向量的维度（即 10,000 × 100 = 1,000,000）。

![](img/gdl2_0502.png)

###### 图 5-2。嵌入层是每个整数标记的查找表

我们将每个整数标记嵌入到连续向量中，因为这使得模型能够学习每个单词的表示，这些表示可以通过反向传播进行更新。我们也可以只对每个输入标记进行独热编码，但使用嵌入层更可取，因为它使得嵌入本身是可训练的，从而使模型在决定如何嵌入每个标记以提高性能时更加灵活。

因此，`Input`层将形状为`[batch_size, seq_length]`的整数序列张量传递给`Embedding`层，后者输出形状为`[batch_size, seq_length, embedding_size]`的张量。然后将其传递给`LSTM`层(图 5-3)。

![](img/gdl2_0503.png)

###### 图 5-3。单个序列在嵌入层中流动

## LSTM 层

要理解 LSTM 层，我们首先必须看一下通用循环层的工作原理。

循环层具有特殊属性，能够处理顺序输入数据<math alttext="x 1 comma ellipsis comma x Subscript n Baseline"><mrow><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <mo>⋯</mo> <mo>,</mo> <msub><mi>x</mi> <mi>n</mi></msub></mrow></math>。随着序列中的每个元素<math alttext="x Subscript t"><msub><mi>x</mi> <mi>t</mi></msub></math>逐个时间步通过，它会更新其*隐藏状态*<math alttext="h Subscript t"><msub><mi>h</mi> <mi>t</mi></msub></math>。

隐藏状态是一个向量，其长度等于细胞中的*单元*数——它可以被视为细胞对序列的当前理解。在时间步<math alttext="t"><mi>t</mi></math>，细胞使用先前的隐藏状态值<math alttext="h Subscript t minus 1"><msub><mi>h</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>，以及当前时间步的数据<math alttext="x Subscript t"><msub><mi>x</mi> <mi>t</mi></msub></math>，产生一个更新的隐藏状态向量<math alttext="h Subscript t"><msub><mi>h</mi> <mi>t</mi></msub></math>。这个循环过程持续到序列结束。一旦序列结束，该层输出细胞的最终隐藏状态<math alttext="h Subscript n"><msub><mi>h</mi> <mi>n</mi></msub></math>，然后传递给网络的下一层。这个过程在图 5-4 中显示。

![](img/gdl2_0504.png)

###### 图 5-4。循环层的简单图示

为了更详细地解释这一点，让我们展开这个过程，这样我们就可以看到单个序列是如何通过该层传递的（图 5-5）。

# 细胞权重

重要的是要记住，这个图中的所有细胞共享相同的权重（因为它们实际上是相同的细胞）。这个图与图 5-4 没有区别；只是以不同的方式绘制了循环层的机制。

![](img/gdl2_0505.png)

###### 图 5-5。单个序列如何流经循环层

在这里，我们通过在每个时间步绘制细胞的副本来表示循环过程，并展示隐藏状态如何在流经细胞时不断更新。我们可以清楚地看到先前的隐藏状态如何与当前的顺序数据点（即当前嵌入的单词向量）混合以产生下一个隐藏状态。该层的输出是细胞的最终隐藏状态，在输入序列中的每个单词都被处理后。

###### 警告

细胞的输出被称为*隐藏*状态是一个不幸的命名惯例——它并不真正隐藏，你不应该这样认为。事实上，最后一个隐藏状态是该层的整体输出，我们将利用这一点，稍后在本章中我们可以访问每个时间步的隐藏状态。

## LSTM 细胞

现在我们已经看到了一个通用循环层是如何工作的，让我们来看看单个 LSTM 细胞的内部。

LSTM 细胞的工作是输出一个新的隐藏状态，<math alttext="h Subscript t"><msub><mi>h</mi> <mi>t</mi></msub></math>，给定其先前的隐藏状态，<math alttext="h Subscript t minus 1"><msub><mi>h</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>，和当前的单词嵌入，<math alttext="x Subscript t"><msub><mi>x</mi> <mi>t</mi></msub></math>。回顾一下，<math alttext="h Subscript t"><msub><mi>h</mi> <mi>t</mi></msub></math>的长度等于 LSTM 中的单元数。这是在定义层时设置的一个参数，与序列的长度无关。

###### 警告

确保不要混淆术语*细胞*和*单元*。在 LSTM 层中有一个细胞，由它包含的单元数定义，就像我们早期故事中的囚犯细胞包含许多囚犯一样。我们经常将循环层绘制为展开的细胞链，因为这有助于可视化如何在每个时间步更新隐藏状态。

LSTM 单元格维护一个单元格状态，<math alttext="上标 C 下标 t"><msub><mi>C</mi> <mi>t</mi></msub></math>，可以被视为单元格对序列当前状态的内部信念。这与隐藏状态，<math alttext="h 下标 t"><msub><mi>h</mi> <mi>t</mi></msub></math>，是不同的，隐藏状态最终在最后一个时间步输出。单元格状态与隐藏状态相同长度（单元格中的单元数）。

让我们更仔细地看一下单个单元格以及隐藏状态是如何更新的（图 5-6）。

隐藏状态在六个步骤中更新：

1.  上一个时间步的隐藏状态，<math alttext="h 下标 t 减 1"><msub><mi>h</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>，和当前的单词嵌入，<math alttext="x 下标 t"><msub><mi>x</mi> <mi>t</mi></msub></math>，被连接起来并通过*遗忘*门传递。这个门只是一个带有权重矩阵 <math alttext="上标 W 下标 f"><msub><mi>W</mi> <mi>f</mi></msub></math>，偏置 <math alttext="b 下标 f"><msub><mi>b</mi> <mi>f</mi></msub></math> 和 sigmoid 激活函数的稠密层。得到的向量，<math alttext="f 下标 t"><msub><mi>f</mi> <mi>t</msub></math>，长度等于单元格中的单元数，并包含介于 0 和 1 之间的值，确定了应该保留多少先前的单元格状态，<math alttext="上标 C 下标 t 减 1"><msub><mi>C</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>。

![](img/gdl2_0506.png)

###### 图 5-6\. LSTM 单元格

1.  连接的向量也通过一个*输入*门传递，类似于遗忘门，它是一个带有权重矩阵 <math alttext="上标 W 下标 i"><msub><mi>W</mi> <mi>i</mi></msub></math>，偏置 <math alttext="b 下标 i"><msub><mi>b</mi> <mi>i</msub></math> 和 sigmoid 激活函数的稠密层。这个门的输出，<math alttext="i 下标 t"><msub><mi>i</mi> <mi>t</msub></math>，长度等于单元格中的单元数，并包含介于 0 和 1 之间的值，确定了新信息将被添加到先前单元格状态，<math alttext="上标 C 下标 t 减 1"><msub><mi>C</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>，的程度。

1.  连接的向量也通过一个带有权重矩阵 <math alttext="上标 W 上标 C"><msub><mi>W</mi> <mi>C</mi></msub></math>，偏置 <math alttext="b 上标 C"><msub><mi>b</mi> <mi>C</mi></msub></math> 和 tanh 激活函数的稠密层，生成一个向量 <math alttext="上标 C overTilde 下标 t"><msub><mover accent="true"><mi>C</mi> <mo>˜</mo></mover> <mi>t</msub></math>，其中包含单元格希望考虑保留的新信息。它的长度也等于单元格中的单元数，并包含介于-1 和 1 之间的值。

1.  <math alttext="f 下标 t"><msub><mi>f</mi> <mi>t</mi></msub></math> 和 <math alttext="上标 C 下标 t 减 1"><msub><mi>C</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math> 逐元素相乘并加到 <math alttext="i 下标 t"><msub><mi>i</mi> <mi>t</mi></msub></math> 和 <math alttext="上标 C overTilde 下标 t"><msub><mover accent="true"><mi>C</mi> <mo>˜</mo></mover> <mi>t</mi></msub></math> 的逐元素乘积中。这代表了遗忘先前单元格状态的部分，并添加新的相关信息以生成更新后的单元格状态，<math alttext="上标 C 下标 t"><msub><mi>C</mi> <mi>t</mi></msub></math>。

1.  连接后的向量通过一个*输出*门传递：一个带有权重矩阵<math alttext="upper W Subscript o"><msub><mi>W</mi> <mi>o</mi></msub></math>、偏置<math alttext="b Subscript o"><msub><mi>b</mi> <mi>o</mi></msub></math>和 sigmoid 激活函数的稠密层。得到的向量<math alttext="o Subscript t"><msub><mi>o</mi> <mi>t</mi></msub></math>的长度等于单元格中的单元数，并存储介于 0 和 1 之间的值，确定要从单元格中输出的更新后的单元格状态<math alttext="upper C Subscript t"><msub><mi>C</mi> <mi>t</mi></msub></math>的多少。

1.  <math alttext="o Subscript t"><msub><mi>o</mi> <mi>t</mi></msub></math>与更新后的单元格状态<math alttext="upper C Subscript t"><msub><mi>C</mi> <mi>t</mi></msub></math>进行逐元素相乘，然后应用 tanh 激活函数产生新的隐藏状态<math alttext="h Subscript t"><msub><mi>h</mi> <mi>t</mi></msub></math>。

# Keras LSTM 层

所有这些复杂性都包含在 Keras 的`LSTM`层类型中，因此您不必担心自己实现它！

## 训练 LSTM

构建、编译和训练 LSTM 的代码在 Example 5-8 中给出。

##### Example 5-8\. 构建、编译和训练 LSTM

```py
inputs = layers.Input(shape=(None,), dtype="int32") ![1](img/1.png)
x = layers.Embedding(10000, 100)(inputs) ![2](img/2.png)
x = layers.LSTM(128, return_sequences=True)(x) ![3](img/3.png)
outputs = layers.Dense(10000, activation = 'softmax')(x) ![4](img/4.png)
lstm = models.Model(inputs, outputs) ![5](img/5.png)

loss_fn = losses.SparseCategoricalCrossentropy()
lstm.compile("adam", loss_fn) ![6](img/6.png)
lstm.fit(train_ds, epochs=25) ![7](img/7.png)
```

![1](img/#co_autoregressive_models_CO3-1)

`Input`层不需要我们提前指定序列长度（可以是灵活的），所以我们使用`None`作为占位符。

![2](img/#co_autoregressive_models_CO3-2)

`Embedding`层需要两个参数，词汇量的大小（10,000 个标记）和嵌入向量的维度（100）。

![3](img/#co_autoregressive_models_CO3-3)

LSTM 层要求我们指定隐藏向量的维度（128）。我们还选择返回完整的隐藏状态序列，而不仅仅是最终时间步的隐藏状态。

![4](img/#co_autoregressive_models_CO3-4)

`Dense`层将每个时间步的隐藏状态转换为下一个标记的概率向量。

![5](img/#co_autoregressive_models_CO3-5)

整体的`Model`在给定一系列标记的输入序列时预测下一个标记。它为序列中的每个标记执行此操作。

![6](img/#co_autoregressive_models_CO3-6)

该模型使用`SparseCategoricalCrossentropy`损失进行编译——这与分类交叉熵相同，但在标签为整数而不是独热编码向量时使用。

![7](img/#co_autoregressive_models_CO3-7)

模型适合训练数据集。

在 Figure 5-7 中，您可以看到 LSTM 训练过程的前几个时期——请注意随着损失指标下降，示例输出变得更加易懂。Figure 5-8 显示了整个训练过程中交叉熵损失指标的下降。

![](img/gdl2_0507.png)

###### Figure 5-7\. LSTM 训练过程的前几个时期

![](img/gdl2_0508.png)

###### Figure 5-8\. LSTM 训练过程中的交叉熵损失指标按时期

## LSTM 的分析

现在我们已经编译和训练了 LSTM，我们可以开始使用它通过以下过程生成长文本字符串：

1.  用现有的单词序列喂给网络，并要求它预测下一个单词。

1.  将这个单词附加到现有序列并重复。

网络将为每个单词输出一组概率，我们可以从中进行采样。因此，我们可以使文本生成具有随机性，而不是确定性。此外，我们可以引入一个*温度*参数到采样过程中，以指示我们希望过程有多确定性。

# 温度参数

接近 0 的温度使采样更加确定性（即，具有最高概率的单词很可能被选择），而温度为 1 意味着每个单词都以模型输出的概率被选择。

这是通过在示例 5-9 中的代码实现的，该代码创建了一个回调函数，可以在每个训练周期结束时用于生成文本。

##### 示例 5-9。`TextGenerator`回调函数

```py
class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        } ![1](img/1.png)

    def sample_from(self, probs, temperature): ![2](img/2.png)
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ] ![3](img/3.png)
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0: ![4](img/4.png)
            x = np.array([start_tokens])
            y = self.model.predict(x) ![5](img/5.png)
            sample_token, probs = self.sample_from(y[0][-1], temperature) ![6](img/6.png)
            info.append({'prompt': start_prompt , 'word_probs': probs})
            start_tokens.append(sample_token) ![7](img/7.png)
            start_prompt = start_prompt + ' ' + self.index_to_word[sample_token]
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("recipe for", max_tokens = 100, temperature = 1.0)
```

![1](img/#co_autoregressive_models_CO4-1)

创建一个反向词汇映射（从单词到标记）。

![2](img/#co_autoregressive_models_CO4-2)

此函数使用`temperature`缩放因子更新概率。

![3](img/#co_autoregressive_models_CO4-3)

起始提示是您想要给模型以开始生成过程的一串单词（例如，*recipe for*）。首先将这些单词转换为标记列表。

![4](img/#co_autoregressive_models_CO4-4)

序列生成直到达到`max_tokens`长度或产生停止令牌（0）为止。

![5](img/#co_autoregressive_models_CO4-5)

模型输出每个单词成为序列中下一个单词的概率。

![6](img/#co_autoregressive_models_CO4-6)

概率通过采样器传递以输出下一个单词，由`temperature`参数化。

![7](img/#co_autoregressive_models_CO4-7)

我们将新单词附加到提示文本中，准备进行生成过程的下一次迭代。

让我们看看这在实际中是如何运作的，使用两个不同的温度值（图 5-9）。

![](img/gdl2_0509.png)

###### 图 5-9。在`temperature = 1.0`和`temperature = 0.2`时生成的输出

关于这两段文字有几点需要注意。首先，两者在风格上与原始训练集中的食谱相似。它们都以食谱标题开头，并包含通常语法正确的结构。不同之处在于，温度为 1.0 的生成文本更加冒险，因此比温度为 0.2 的示例不够准确。因此，使用温度为 1.0 生成多个样本将导致更多的变化，因为模型正在从具有更大方差的概率分布中进行抽样。

为了证明这一点，图 5-10 显示了一系列提示的前五个具有最高概率的标记，对于两个温度值。

![](img/gdl2_0510.png)

###### 图 5-10。在不同序列后的单词概率分布，对于温度值为 1.0 和 0.2

该模型能够在一系列上下文中生成下一个最可能的单词的适当分布。例如，即使模型从未被告知过名词、动词或数字等词类，它通常能够将单词分为这些类别并以语法正确的方式使用它们。

此外，该模型能够选择一个适当的动词来开始食谱说明，这取决于前面的标题。对于烤蔬菜，它选择`preheat`、`prepare`、`heat`、`put`或`combine`作为最可能的可能性，而对于冰淇淋，它选择`in`、`combine`、`stir`、`whisk`和`mix`。这表明该模型对于根据其成分而异的食谱之间的差异具有一定的上下文理解。

还要注意`temperature = 0.2`示例的概率更加倾向于第一个选择标记。这就是为什么当温度较低时，生成的变化通常较少的原因。

虽然我们的基本 LSTM 模型在生成逼真文本方面表现出色，但很明显它仍然难以理解所生成单词的一些语义含义。它引入了一些不太可能搭配在一起的成分（例如，酸味日本土豆、山核桃碎屑和果冻）！在某些情况下，这可能是可取的——比如，如果我们希望我们的 LSTM 生成有趣和独特的单词模式——但在其他情况下，我们需要我们的模型对单词如何组合在一起以及在文本中引入的想法有更深入的理解和更长的记忆。

在下一节中，我们将探讨如何改进我们的基本 LSTM 网络。在第九章中，我们将看一看一种新型的自回归模型，Transformer，将语言建模提升到一个新的水平。

前一节中的模型是一个简单的示例，展示了如何训练 LSTM 学习如何以给定风格生成文本。在本节中，我们将探讨这个想法的几个扩展。

## 堆叠循环网络

我们刚刚看到的网络包含一个单独的 LSTM 层，但我们也可以训练具有堆叠 LSTM 层的网络，以便从文本中学习更深层次的特征。

为了实现这一点，我们只需在第一层之后引入另一层 LSTM。第二层 LSTM 可以使用第一层的隐藏状态作为其输入数据。这在图 5-11 中显示，整体模型架构在表 5-2 中显示。

![](img/gdl2_0511.png)

###### 图 5-11。多层 RNN 的示意图：g[t]表示第一层的隐藏状态，h[t]表示第二层的隐藏状态

表 5-2。堆叠 LSTM 的模型摘要

| 层（类型） | 输出形状 | 参数 # |
| --- | --- | --- |
| 输入层 | (None, None) | 0 |
| 嵌入 | (None, None, 100) | 1,000,000 |
| LSTM | (None, None, 128) | 117,248 |
| LSTM | (None, None, 128) | 131,584 |
| 稠密 | (None, None, 10000) | 1,290,000 |
| 总参数 | 2,538,832 |
| 可训练参数 | 2,538,832 |
| 不可训练参数 | 0 |

构建堆叠 LSTM 的代码在示例 5-10 中给出。

##### 示例 5-10。构建堆叠 LSTM

```py
text_in = layers.Input(shape = (None,))
embedding = layers.Embedding(total_words, embedding_size)(text_in)
x = layers.LSTM(n_units, return_sequences = True)(x)
x = layers.LSTM(n_units, return_sequences = True)(x)
probabilites = layers.Dense(total_words, activation = 'softmax')(x)
model = models.Model(text_in, probabilites)
```

## 门控循环单元

另一种常用的 RNN 层是*门控循环单元*（GRU）。^(2) 与 LSTM 单元的主要区别如下：

1.  *遗忘*和*输入*门被*重置*和*更新*门替换。

1.  没有*细胞状态*或*输出*门，只有从细胞输出的*隐藏状态*。

隐藏状态通过四个步骤更新，如图 5-12 所示。

![](img/gdl2_0512.png)

###### 图 5-12。单个 GRU 单元

过程如下：

1.  上一个时间步的隐藏状态，<math alttext="h Subscript t minus 1"><msub><mi>h</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>，和当前单词嵌入，<math alttext="x Subscript t"><msub><mi>x</mi> <mi>t</mi></msub></math>，被串联并用于创建*重置*门。这个门是一个密集层，带有权重矩阵<math alttext="upper W Subscript r"><msub><mi>W</mi> <mi>r</mi></msub></math>和一个 sigmoid 激活函数。得到的向量，<math alttext="r Subscript t"><msub><mi>r</mi> <mi>t</mi></msub></math>，长度等于细胞中的单元数，并存储介于 0 和 1 之间的值，确定应该将多少上一个隐藏状态，<math alttext="h Subscript t minus 1"><msub><mi>h</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>，传递到新信念的计算中。

1.  重置门应用于隐藏状态，<math alttext="h Subscript t minus 1"><msub><mi>h</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>，并与当前单词嵌入<math alttext="x Subscript t"><msub><mi>x</mi> <mi>t</mi></msub></math>连接。然后将该向量馈送到具有权重矩阵<math alttext="upper W"><mi>W</mi></math>和 tanh 激活函数的密集层，以生成一个向量<math alttext="h overTilde Subscript t"><msub><mover accent="true"><mi>h</mi> <mo>˜</mo></mover> <mi>t</mi></msub></math>，其中存储了细胞的新信念。它的长度等于细胞中的单元数，并存储在-1 和 1 之间的值。

1.  前一个时间步的隐藏状态<math alttext="h Subscript t minus 1"><msub><mi>h</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>和当前单词嵌入<math alttext="x Subscript t"><msub><mi>x</mi> <mi>t</mi></msub></math>的连接也用于创建*更新*门。该门是一个具有权重矩阵<math alttext="upper W Subscript z"><msub><mi>W</mi> <mi>z</mi></msub></math>和 sigmoid 激活的密集层。生成的向量<math alttext="z Subscript t"><msub><mi>z</mi> <mi>t</msub></math>的长度等于细胞中的单元数，并存储在 0 和 1 之间的值，用于确定新信念<math alttext="h overTilde Subscript t"><msub><mover accent="true"><mi>h</mi> <mo>˜</mo></mover> <mi>t</mi></msub></math>的多少要混合到当前隐藏状态<math alttext="h Subscript t minus 1"><msub><mi>h</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>中。

1.  细胞的新信念<math alttext="h overTilde Subscript t"><msub><mover accent="true"><mi>h</mi> <mo>˜</mo></mover> <mi>t</mi></msub></math>和当前隐藏状态<math alttext="h Subscript t minus 1"><msub><mi>h</mi> <mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub></math>按照更新门<math alttext="z Subscript t"><msub><mi>z</mi> <mi>t</msub></math>确定的比例混合，以产生更新后的隐藏状态<math alttext="h Subscript t"><msub><mi>h</mi> <mi>t</msub></math>，从细胞中输出。

## 双向细胞

对于预测问题，在推断时模型可以获得整个文本，没有理由只在正向方向处理序列 - 它同样可以被反向处理。`Bidirectional`层通过存储两组隐藏状态来利用这一点：一组是由序列在通常的正向方向处理时产生的，另一组是在序列被反向处理时产生的。这样，该层可以从给定时间步之前和之后的信息中学习。

在 Keras 中，这被实现为对循环层的包装，如示例 5-11 所示。

##### 示例 5-11。构建双向 GRU 层

```py
layer = layers.Bidirectional(layers.GRU(100))
```

# 隐藏状态

结果层中的隐藏状态是长度等于包装细胞中单元数两倍的向量（正向和反向隐藏状态的连接）。因此，在此示例中，该层的隐藏状态是长度为 200 的向量。

到目前为止，我们只将自回归模型（LSTMs）应用于文本数据。在下一节中，我们将看到如何使用自回归模型来生成图像。

# PixelCNN

2016 年，van den Oord 等人^(3)提出了一种通过预测下一个像素的可能性来逐像素生成图像的模型。该模型称为*PixelCNN*，可以训练以自回归方式生成图像。

我们需要介绍两个新概念来理解 PixelCNN - *掩码卷积层*和*残差块*。

# 运行此示例的代码

此示例的代码可以在位于书籍存储库中的 Jupyter 笔记本中找到，路径为*notebooks/05_autoregressive/02_pixelcnn/pixelcnn.ipynb*。

该代码改编自由 ADMoreau 创建的出色的[PixelCNN 教程](https://keras.io/examples/generative/pixelcnn)，可在 Keras 网站上找到。 

## 掩码卷积层

正如我们在第二章中看到的，卷积层可以通过应用一系列滤波器从图像中提取特征。在特定像素处的层的输出是滤波器权重乘以围绕像素中心的小正方形上一层值的加权和。这种方法可以检测边缘和纹理，而在更深的层中，可以检测形状和更高级的特征。

虽然卷积层在特征检测方面非常有用，但不能直接以自回归的方式使用，因为像素上没有顺序。它们依赖于所有像素都被平等对待的事实——没有像素被视为图像的*开始*或*结束*。这与我们在本章中已经看到的文本数据形成对比，其中令牌有明确的顺序，因此可以轻松应用循环模型，如 LSTM。

为了能够以自回归的方式将卷积层应用于图像生成，我们必须首先对像素进行排序，并确保滤波器只能看到在问题像素之前的像素。然后，我们可以通过将卷积滤波器应用于当前图像来一次生成一个像素，以预测下一个像素的值。

我们首先需要为像素选择一个顺序——一个明智的建议是按照从左上到右下的顺序对像素进行排序，首先沿着行移动，然后沿着列向下移动。

然后，我们对卷积滤波器进行掩码处理，以便每个像素处的层的输出仅受到在问题像素之前的像素值的影响。这是通过将一个由 1 和 0 组成的掩码与滤波器权重矩阵相乘来实现的，以便在目标像素之后的任何像素的值都被置为零。

在 PixelCNN 中实际上有两种不同类型的掩码，如图 5-13 所示：

+   类型 A，中心像素的值被掩码

+   类型 B，中心像素的值*未*被掩码

![](img/gdl2_0513.png)

###### 图 5-13。左：卷积滤波器掩码；右：应用于一组像素以预测中心像素值分布的掩码（来源：[van den Oord 等人，2016](https://arxiv.org/pdf/1606.05328))

初始的掩码卷积层（即直接应用于输入图像的层）不能使用中心像素，因为这正是我们希望网络猜测的像素！然而，后续层可以使用中心像素，因为这将仅根据原始输入图像中前面像素的信息计算出来。

我们可以在示例 5-12 中看到如何使用 Keras 构建`MaskedConvLayer`。

##### 示例 5-12。Keras 中的`MaskedConvLayer`

```py
class MaskedConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(MaskedConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs) ![1](img/1.png)

    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape) ![2](img/2.png)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0 ![3](img/3.png)
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0 ![4](img/4.png)
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0 ![5](img/5.png)

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask) ![6](img/6.png)
        return self.conv(inputs)
```

![1](img/#co_autoregressive_models_CO5-1)

`MaskedConvLayer`基于普通的`Conv2D`层。

![2](img/#co_autoregressive_models_CO5-2)

掩码初始化为全零。

![3](img/#co_autoregressive_models_CO5-3)

前面行中的像素将被一个 1 解除掩码。

![4](img/#co_autoregressive_models_CO5-4)

前面列中在同一行中的像素将被一个 1 解除掩码。

![5](img/#co_autoregressive_models_CO5-5)

如果掩码类型为 B，则中心像素将被一个 1 解除掩码。

![6](img/#co_autoregressive_models_CO5-6)

掩码与滤波器权重相乘。

请注意，这个简化的例子假设是灰度图像（即，只有一个通道）。如果是彩色图像，我们将有三个颜色通道，我们也可以对它们进行排序，例如，红色通道在蓝色通道之前，蓝色通道在绿色通道之前。

## 残差块

现在我们已经看到如何对卷积层进行掩码，我们可以开始构建我们的 PixelCNN。我们将使用的核心构建块是残差块。

*残差块*是一组层，其中输出在传递到网络的其余部分之前添加到输入中。换句话说，输入有一条*快速通道*到输出，而无需经过中间层——这被称为*跳跃连接*。包含跳跃连接的理由是，如果最佳转换只是保持输入不变，这可以通过简单地将中间层的权重置零来实现。如果没有跳跃连接，网络将不得不通过中间层找到一个恒等映射，这要困难得多。

我们在 PixelCNN 中的残差块的图示在图 5-14 中显示。

![](img/gdl2_0514.png)

###### 图 5-14。一个 PixelCNN 残差块（箭头旁边是滤波器的数量，层旁边是滤波器大小）

我们可以使用示例 5-13 中显示的代码构建一个`ResidualBlock`。

##### 示例 5-13。一个`ResidualBlock`

```py
class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(
            filters=filters // 2, kernel_size=1, activation="relu"
        ) ![1](img/1.png)
        self.pixel_conv = MaskedConv2D(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        ) ![2](img/2.png)
        self.conv2 = layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        ) ![3](img/3.png)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return layers.add([inputs, x]) ![4](img/4.png)
```

![1](img/#co_autoregressive_models_CO6-1)

初始的`Conv2D`层将通道数量减半。

![2](img/#co_autoregressive_models_CO6-2)

Type B `MaskedConv2D`层，核大小为 3，仅使用来自五个像素的信息——上面一行中的三个像素，左边一个像素和焦点像素本身。

![3](img/#co_autoregressive_models_CO6-3)

最终的`Conv2D`层将通道数量加倍，以再次匹配输入形状。

![4](img/#co_autoregressive_models_CO6-4)

卷积层的输出与输入相加——这是跳跃连接。

## 训练 PixelCNN

在示例 5-14 中，我们组合了整个 PixelCNN 网络，大致遵循原始论文中的结构。在原始论文中，输出层是一个有 256 个滤波器的`Conv2D`层，使用 softmax 激活。换句话说，网络试图通过预测正确的像素值来重新创建其输入，有点像自动编码器。不同之处在于，PixelCNN 受到限制，以便不允许来自早期像素的信息流通过影响每个像素的预测，这是由于网络设计方式，使用`MaskedConv2D`层。

这种方法的一个挑战是网络无法理解，比如说，像素值 200 非常接近像素值 201。它必须独立学习每个像素输出值，这意味着即使对于最简单的数据集，训练也可能非常缓慢。因此，在我们的实现中，我们简化输入，使每个像素只能取四个值之一。这样，我们可以使用一个有 4 个滤波器的`Conv2D`输出层，而不是 256 个。

##### 示例 5-14。PixelCNN 架构

```py
inputs = layers.Input(shape=(16, 16, 1)) ![1](img/1.png)
x = MaskedConv2D(mask_type="A"
                   , filters=128
                   , kernel_size=7
                   , activation="relu"
                   , padding="same")(inputs)![2](img/2.png)

for _ in range(5):
    x = ResidualBlock(filters=128)(x) ![3](img/3.png)

for _ in range(2):
    x = MaskedConv2D(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )(x) ![4](img/4.png)

out = layers.Conv2D(
    filters=4, kernel_size=1, strides=1, activation="softmax", padding="valid"
)(x) ![5](img/5.png)

pixel_cnn = models.Model(inputs, out) ![6](img/6.png)

adam = optimizers.Adam(learning_rate=0.0005)
pixel_cnn.compile(optimizer=adam, loss="sparse_categorical_crossentropy")

pixel_cnn.fit(
    input_data
    , output_data
    , batch_size=128
    , epochs=150
) ![7](img/7.png)
```

![1](img/#co_autoregressive_models_CO7-1)

模型的`Input`是一个尺寸为 16×16×1 的灰度图像，输入值在 0 到 1 之间缩放。

![2](img/#co_autoregressive_models_CO7-2)

第一个 Type A `MaskedConv2D`层，核大小为 7，使用来自 24 个像素的信息——在焦点像素上面的三行中的 21 个像素和左边的 3 个像素（焦点像素本身不使用）。

![3](img/#co_autoregressive_models_CO7-3)

五个`ResidualBlock`层组被顺序堆叠。

![4](img/#co_autoregressive_models_CO7-4)

两个 Type B `MaskedConv2D`层，核大小为 1，作为每个像素通道数量的`Dense`层。

![5](img/#co_autoregressive_models_CO7-5)

最终的`Conv2D`层将通道数减少到四——本示例中的像素级别数。

![6](img/#co_autoregressive_models_CO7-6)

`Model`被构建为接受一幅图像并输出相同尺寸的图像。

![7](img/#co_autoregressive_models_CO7-7)

拟合模型——`input_data`在范围[0,1]（浮点数）内缩放；`output_data`在范围[0,3]（整数）内缩放。

## PixelCNN 的分析

我们可以在我们在第三章中遇到的 Fashion-MNIST 数据集上训练我们的 PixelCNN。要生成新图像，我们需要要求模型根据所有先前像素预测下一个像素，逐个像素进行预测。与诸如变分自动编码器的模型相比，这是一个非常缓慢的过程！对于一幅 32×32 的灰度图像，我们需要使用模型进行 1,024 次顺序预测，而不是我们需要为 VAE 进行的单次预测。这是自回归模型如 PixelCNN 的主要缺点之一——由于采样过程的顺序性质，它们从中采样速度较慢。

因此，我们使用图像尺寸为 16×16，而不是 32×32，以加快生成新图像的速度。生成回调类如示例 5-15 所示。

##### 示例 5-15。使用 PixelCNN 生成新图像

```py
class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs)

    def generate(self, temperature):
        generated_images = np.zeros(
            shape=(self.num_img,) + (pixel_cnn.input_shape)[1:]
        ) ![1](img/1.png)
        batch, rows, cols, channels = generated_images.shape

        for row in range(rows):
            for col in range(cols):
                for channel in range(channels):
                    probs = self.model.predict(generated_images)[
                        :, row, col, :
                    ] ![2](img/2.png)
                    generated_images[:, row, col, channel] = [
                        self.sample_from(x, temperature) for x in probs
                    ] ![3](img/3.png)
                    generated_images[:, row, col, channel] /= 4 ![4](img/4.png)
        return generated_images

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.generate(temperature = 1.0)
        display(
            generated_images,
            save_to = "./output/generated_img_%03d.png" % (epoch)
        s)

img_generator_callback = ImageGenerator(num_img=10)
```

![1](img/#co_autoregressive_models_CO8-1)

从一批空白图像（全零）开始。

![2](img/#co_autoregressive_models_CO8-2)

循环遍历当前图像的行、列和通道，预测下一个像素值的分布。

![3](img/#co_autoregressive_models_CO8-3)

从预测分布中抽取一个像素级别（对于我们的示例，范围在[0,3]内）。

![4](img/#co_autoregressive_models_CO8-4)

将像素级别转换为范围[0,1]并覆盖当前图像中的像素值，准备好进行下一次循环迭代。

在图 5-15 中，我们可以看到原始训练集中的几幅图像，以及由 PixelCNN 生成的图像。

![](img/gdl2_0515.png)

###### 图 5-15。训练集中的示例图像和由 PixelCNN 模型生成的图像

该模型在重新创建原始图像的整体形状和风格方面做得很好！令人惊讶的是，我们可以将图像视为一系列令牌（像素值），并应用自回归模型如 PixelCNN 来生成逼真的样本。

如前所述，自回归模型的一个缺点是它们从中采样速度较慢，这就是为什么本书中提供了它们应用的一个简单示例。然而，正如我们将在第十章中看到的，更复杂形式的自回归模型可以应用于图像以产生最先进的输出。在这种情况下，缓慢的生成速度是为了获得卓越质量输出而必须付出的代价。

自原始论文发表以来，PixelCNN 的架构和训练过程已经进行了几项改进。以下部分介绍了其中一项变化——使用混合分布，并演示了如何使用内置的 TensorFlow 函数训练带有此改进的 PixelCNN 模型。

## 混合分布

对于我们之前的示例，我们将 PixelCNN 的输出减少到只有 4 个像素级别，以确保网络不必学习 256 个独立像素值的分布，这将减慢训练过程。然而，这远非理想——对于彩色图像，我们不希望我们的画布仅限于少数可能的颜色。

为了解决这个问题，我们可以使网络的输出成为*混合分布*，而不是对 256 个离散像素值进行 softmax，遵循 Salimans 等人提出的想法。4 混合分布简单地是两个或更多其他概率分布的混合。例如，我们可以有五个具有不同参数的逻辑分布的混合分布。混合分布还需要一个离散分类分布，表示选择混合中包含的每个分布的概率。示例显示在图 5-16 中。 

![](img/gdl2_0516.png)

###### 图 5-16。三个具有不同参数的正态分布的混合分布——三个正态分布上的分类分布为`[0.5, 0.3, 0.2]`

要从混合分布中抽样，我们首先从分类分布中抽样以选择特定的子分布，然后以通常的方式从中抽样。这样，我们可以用相对较少的参数创建复杂的分布。例如，图 5-16 中的混合分布仅需要八个参数——两个用于分类分布，以及三个正态分布的均值和方差。这与定义整个像素范围上的分类分布所需的 255 个参数相比要少。

方便地，TensorFlow Probability 库提供了一个函数，允许我们用一行代码创建具有混合分布输出的 PixelCNN。示例 5-16 说明了如何使用此函数构建 PixelCNN。

# 运行此示例的代码

此示例的代码可以在书籍存储库中的 Jupyter 笔记本*notebooks/05_autoregressive/03_pixelcnn_md/pixelcnn_md.ipynb*中找到。

##### 示例 5-16。使用 TensorFlow 函数构建 PixelCNN

```py
import tensorflow_probability as tfp

dist = tfp.distributions.PixelCNN(
    image_shape=(32, 32, 1),
    num_resnet=1,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=5,
    dropout_p=.3,
) ![1](img/1.png)

image_input = layers.Input(shape=(32, 32, 1)) ![2](img/2.png)

log_prob = dist.log_prob(image_input)

model = models.Model(inputs=image_input, outputs=log_prob) ![3](img/3.png)
model.add_loss(-tf.reduce_mean(log_prob)) ![4](img/4.png)
```

![1](img/#co_autoregressive_models_CO9-1)

将 PixelCNN 定义为一个分布——即，输出层是由五个逻辑分布组成的混合分布。

![2](img/#co_autoregressive_models_CO9-2)

输入是大小为 32×32×1 的灰度图像。

![3](img/#co_autoregressive_models_CO9-3)

`Model`以灰度图像作为输入，并输出在 PixelCNN 计算的混合分布下图像的对数似然。

![4](img/#co_autoregressive_models_CO9-4)

损失函数是输入图像批次上的平均负对数似然。

该模型的训练方式与以前相同，但这次接受整数像素值作为输入，范围为[0, 255]。可以使用`sample`函数从分布中生成输出，如示例 5-17 所示。

##### 示例 5-17。从 PixelCNN 混合分布中抽样

```py
dist.sample(10).numpy()
```

示例生成的图像显示在图 5-17 中。与以前的示例不同的是，现在正在利用完整的像素值范围。

![](img/gdl2_0517.png)

###### 图 5-17。使用混合分布输出的 PixelCNN 的输出

# 总结

在本章中，我们看到了自回归模型，如循环神经网络如何应用于生成模仿特定写作风格的文本序列，以及 PixelCNN 如何以顺序方式生成图像，每次一个像素。

我们探索了两种不同类型的循环层——长短期记忆（LSTM）和门控循环单元（GRU）——并看到这些单元如何可以堆叠或双向化以形成更复杂的网络架构。我们构建了一个 LSTM 来使用 Keras 生成逼真的食谱，并看到如何操纵采样过程的温度以增加或减少输出的随机性。

我们还看到了如何以自回归方式生成图像，使用了 PixelCNN。我们使用 Keras 从头开始构建了一个 PixelCNN，编写了掩膜卷积层和残差块，以允许信息在网络中流动，从而只能使用前面的像素来生成当前的像素。最后，我们讨论了 TensorFlow Probability 库提供了一个独立的 `PixelCNN` 函数，实现了混合分布作为输出层，使我们能够进一步改进学习过程。

在下一章中，我们将探讨另一种生成建模家族，明确地对数据生成分布进行建模—正规化流模型。

^(1) Sepp Hochreiter 和 Jürgen Schmidhuber, “长短期记忆,” *神经计算* 9 (1997): 1735–1780, [*https://www.bioinf.jku.at/publications/older/2604.pdf*](https://www.bioinf.jku.at/publications/older/2604.pdf).

^(2) Kyunghyun Cho 等人, “使用 RNN 编码器-解码器学习短语表示进行统计机器翻译,” 2014 年 6 月 3 日, [*https://arxiv.org/abs/1406.1078*](https://arxiv.org/abs/1406.1078).

^(3) Aaron van den Oord 等人, “像素递归神经网络,” 2016 年 8 月 19 日, [*https://arxiv.org/abs/1601.06759*](https://arxiv.org/abs/1601.06759).

^(4) Tim Salimans 等人, “PixelCNN++: 使用离散化逻辑混合似然和其他修改改进 PixelCNN,” 2017 年 1 月 19 日, [*http://arxiv.org/abs/1701.05517*](http://arxiv.org/abs/1701.05517).
