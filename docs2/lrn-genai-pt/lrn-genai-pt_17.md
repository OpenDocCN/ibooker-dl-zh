# 14 构建和训练音乐 Transformer

本章涵盖了

+   使用控制信息和速度值来表示音乐

+   将音乐标记化为一系列索引

+   构建和训练音乐 Transformer

+   使用训练好的 Transformer 生成音乐事件

+   将音乐事件转换回可播放的 MIDI 文件

为你最喜欢的音乐家不再与我们同在而感到难过？不再难过：生成式 AI 可以将他们带回舞台！

以 Layered Reality 为例，这是一家位于伦敦的公司，正在开发名为《猫王进化》的项目。1 目标？使用 AI 复活传奇的猫王艾维斯·普雷斯利。通过将艾维斯的大量官方档案材料，包括视频剪辑、照片和音乐，输入到一个复杂的计算机模型中，这个 AI 猫王学会了以惊人的相似度模仿他的唱歌、说话、跳舞和走路。结果？一场数字表演，捕捉了已故国王本人的精髓。

《猫王进化》项目是生成式 AI 在各个行业产生变革性影响的杰出例子。在前一章中，你探讨了使用 MuseGAN 创建可以以多轨音乐作品为假的音乐的方法。MuseGAN 将一首音乐视为一个多维对象，类似于图像，并生成与训练数据集中的音乐相似的音乐作品。然后，由评论家评估真实和 AI 生成的音乐，这有助于改进 AI 生成的音乐，直到它与真实音乐无法区分。

在本章中，你将采用一种不同的方法来处理 AI 音乐创作，将其视为一系列音乐事件。我们将应用第十一章和第十二章中讨论的文本生成技术，来预测序列中的下一个元素。具体来说，你将开发一个类似 GPT 风格的模型，根据序列中所有先前事件来预测下一个音乐事件。由于 GPT 风格的 Transformer 具有可扩展性和自注意力机制，这些机制有助于它们捕捉长距离依赖关系并理解上下文，因此它们非常适合这项任务。你将创建的音乐 Transformer 具有 2016 万个参数，足够捕捉音乐作品中不同音符的长期关系，但同时也足够小，可以在合理的时间内进行训练。

我们将使用来自谷歌 Magenta 团队的 Maestro 钢琴音乐作为我们的训练数据。你将学习如何首先将音乐乐器数字接口（MIDI）文件转换为音乐音符序列，类似于自然语言处理（NLP）中的原始文本数据。然后你将把音乐音符分解成称为音乐事件的小片段，类似于 NLP 中的标记。由于神经网络只能接受数值输入，因此你需要将每个独特的事件标记映射到一个索引。有了这个，训练数据中的音乐作品就被转换成了索引序列，准备好输入到神经网络中。

为了训练音乐 Transformer 以预测序列中下一个标记，基于当前标记和序列中所有之前的标记，我们将创建长度为 2,048 的索引序列作为输入（特征 x）。然后我们将序列向右移动一个索引，并使用它们作为输出（目标 y）。我们将 (x, y) 对输入到音乐 Transformer 中以训练模型。一旦训练完成，我们将使用一个短索引序列作为提示并将其输入到音乐 Transformer 中以预测下一个标记，然后将该标记附加到提示中形成一个新的序列。这个新序列被反馈到模型中进行进一步的预测，这个过程会重复进行，直到序列达到期望的长度。

你将看到训练好的音乐 Transformer 可以生成模仿训练数据集中风格的逼真音乐。此外，与第十三章中生成的音乐不同，你将学习如何控制音乐作品的艺术性。你将通过调整预测的 logits 与温度参数的比例来实现这一点，就像你在前几章中控制生成文本的艺术性时做的那样。

## 14.1 音乐 Transformer 简介

音乐 Transformer 的概念于 2018 年提出.^(2) 这种创新方法扩展了最初为 NLP 任务设计的 Transformer 架构，应用于音乐生成领域。正如前几章所讨论的，Transformers 使用自注意力机制来有效地把握上下文，并捕捉序列中元素之间的长距离依赖关系。

类似地，音乐 Transformer 被设计成通过学习大量现有音乐数据集来生成音乐序列。该模型被训练成根据先前的音乐事件来预测序列中的下一个音乐事件，通过理解训练数据中不同音乐元素之间的模式、结构和关系。

训练音乐 Transformer 的关键步骤在于找出如何将音乐表示为一系列独特的音乐事件，类似于 NLP 中的标记。在上一章中，你学习了如何将一首音乐表示为 4D 对象。在本章中，你将探索一种替代的音乐表示方法，即通过控制信息和速度值实现的基于性能的音乐表示.^(3) 基于此，你将把一首音乐转换为四种类型的音乐事件：音符开启、音符关闭、时间移动和速度。

音符开启信号表示一个音符的开始演奏，指定音符的音高。音符关闭表示音符的结束，告诉乐器停止演奏该音符。时间移动表示两个音乐事件之间经过的时间量。速度衡量演奏音符的力量或速度，较高的值对应更强的、更响亮的声音。每种类型的音乐事件都有许多不同的值。每个独特的事件将被映射到不同的索引，有效地将一首音乐转换为一个索引序列。然后，你将应用第十一章和第十二章中讨论的 GPT 模型，创建一个仅具有解码器的音乐 Transformer，以预测序列中的下一个音乐事件。

在本节中，你将首先通过控制信息和速度值了解基于性能的音乐表示。然后，你将探索如何将音乐作品表示为一系列音乐事件。最后，你将学习构建和训练 Transformer 以生成音乐的步骤。

### 14.1.1 基于性能的音乐表示

基于性能的音乐表示通常使用 MIDI 格式实现，该格式通过控制信息和速度值捕捉音乐表演的细微差别。在 MIDI 中，音符通过音符开启和音符关闭消息表示，这些消息包含每个音符的音高和速度信息。

正如我们在第十三章中讨论的，音高值范围从 0 到 127，每个值对应于八度中的一个半音。例如，音高值 60 对应于 C4 音符，而音高值 74 对应于 D5 音符。速度值，同样范围从 0 到 127，表示音符的动态，较高的值表示更响亮或更有力的演奏。通过结合这些控制信息和速度值，MIDI 序列可以捕捉现场表演的表达细节，允许通过兼容 MIDI 的乐器和软件进行表达性的回放。

为了给你一个具体的例子，说明如何通过控制信息和速度值表示一首音乐，请考虑以下列表中显示的五个音符。

列表 14.1 基于性能的音乐表示中的示例音符

```py
<[SNote] time: 1.0325520833333333 type: note_on, value: 74, velocity: 86>
<[SNote] time: 1.0442708333333333 type: note_on, value: 38, velocity: 77>
<[SNote] time: 1.2265625 type: note_off, value: 74, velocity: None>
<[SNote] time: 1.2395833333333333 type: note_on, value: 73, velocity: 69>
<[SNote] time: 1.2408854166666665 type: note_on, value: 37, velocity: 64>
```

这些是训练数据集中你将在本章使用的音乐作品中的前五个音符。第一个音符的大致时间戳为 1.03 秒，音高值为 74（D5）的音符以 86 的速度开始演奏。观察第二个音符，你可以推断出大约 0.01 秒后（因为时间戳现在是 1.04 秒），一个音高值为 38 的音符以 77 的速度开始演奏，以此类推。

这些音乐符号类似于自然语言处理中的原始文本；我们不能直接将它们输入到音乐 Transformer 中训练模型。我们首先需要将音符“令牌化”，然后将令牌转换为索引，再输入到模型中。

为了令牌化音乐音符，我们将使用 0.01 秒的增量来表示音乐，以减少音乐作品中的时间步数。此外，我们将控制消息与速度值分开，并将它们视为音乐作品的独立元素。具体来说，我们将使用音符开启、音符关闭、时间偏移和速度事件来表示音乐。一旦这样做，前五个音符可以表示为以下事件（为了简洁，省略了一些事件）。

列表 14.2 音乐作品的令牌化表示

```py
<Event type: time_shift, value: 99>, 
 <Event type: time_shift, value: 2>, 
 <Event type: velocity, value: 21>, 
 <Event type: note_on, value: 74>, 
 <Event type: time_shift, value: 0>, 
 <Event type: velocity, value: 19>, 
 <Event type: note_on, value: 38>, 
 <Event type: time_shift, value: 17>, 
 <Event type: note_off, value: 74>, 
 <Event type: time_shift, value: 0>, 
 <Event type: velocity, value: 17>, 
 <Event type: note_on, value: 73>, 
 <Event type: velocity, value: 16>, 
 <Event type: note_on, value: 37>, 
 <Event type: time_shift, value: 0>
…
```

我们将以 0.01 秒的增量计算时间偏移，并将从 0.01 秒到 1 秒的时间偏移以 100 个不同的值进行令牌化。因此，时间偏移事件被令牌化为 100 个独特的事件令牌：值为 0 表示 0.01 秒的时间间隔，值为 1 表示 0.02 秒的时间间隔，以此类推，直到 99，表示 1 秒的时间间隔。如果一个时间偏移超过 1 秒，你可以使用多个时间偏移令牌来表示它。例如，列表 14.2 中的前两个令牌都是时间偏移令牌，值分别为 99 和 2，分别表示 1 秒和 0.03 秒的时间间隔。这与列表 14.1 中第一个音乐音符的时间戳相匹配：1.0326 秒。

列表 14.2 也显示了速度是音乐事件的一种独立类型。我们将速度值放入 32 个等间隔的箱子中，将原始的速度值（范围从 0 到 127）转换为 32 个值中的一个，范围从 0 到 31。这就是为什么列表 14.1 中第一个音符的原始速度值 86 现在在列表 14.2 中表示为速度事件，值为 21（因为 86 落在第 22 个箱子中，Python 使用零基索引）。

表 14.1 显示了四种不同令牌化事件的意义、它们的值范围以及每个事件令牌的意义。

表 14.1 不同事件令牌的意义

| 事件令牌类型 | 事件令牌值范围 | 事件令牌的意义 |
| --- | --- | --- |
| `note_on` | 0–127 | 在某个音高值处开始演奏。例如，值为 74 的`note_on`表示开始演奏 D5 音符。 |
| `note_off` | 0–127 | 释放某个音符。例如，值为 60 的`note_off`表示停止演奏 C4 音符。 |
| `time_shift` | 0–99 | `time_shift`值是 0.01 秒的增量。例如，0 表示 0.01 秒，2 表示 0.03 秒，99 表示 1 秒。 |
| `velocity` | 0–31 | 原始速度值被放入 32 个箱子中。使用箱子值。例如，原始速度值为 86 现在有一个标记化值为 21。 |

与 NLP 中采用的方法类似，我们将每个唯一的标记转换为索引，以便我们可以将数据输入到神经网络中。根据表 14.1，有 128 个唯一的音符开启事件标记，128 个音符关闭事件标记，32 个速度事件标记和 100 个时间偏移事件标记。这导致总共有 128 + 128 + 32 + 100 = 388 个唯一标记。因此，我们根据表 14.2 中提供的映射将这些 388 个唯一标记转换为从 0 到 387 的索引。

表 14.2 事件标记到索引和索引到事件标记的映射

| 标记类型 | 索引范围 | 事件标记到索引 | 索引到事件标记 |
| --- | --- | --- | --- |
| `note_on` | 0–127 | `note_on`标记的值。例如，具有 74 个值的`note_on`标记被分配一个索引值为 74。 | 如果索引范围是 0 到 127，将标记类型设置为`note_on`并将值设置为索引值。例如，索引值 63 映射到具有 63 个值的`note_on`标记。 |
| `note_off` | 128–255 | 128 加上`note_off`标记的值。例如，具有 60 个值的`note_off`标记被分配一个索引值为 188（因为 128+60=188）。 | 如果索引范围是 128 到 255，将标记类型设置为`note_off`并将值设置为索引减去 128。例如，索引 180 映射到具有 52 个值的`note_off`标记。 |
| `time_shift` | 256–355 | 256 加上`time_shift`标记的值。例如，具有 16 个值的`time_shift`标记被分配一个索引值为 272（因为 256+16=272）。 | 如果索引范围是 256 到 355，将标记类型设置为`time_shift`并将值设置为索引减去 256。例如，索引 288 映射到具有 32 个值的`time_shift`标记。 |
| `velocity` | 356–387 | 356 加上速度标记的值。例如，具有 21 个值的速度标记被分配一个索引值为 377（因为 356+21=377）。 | 如果索引范围是 356 到 387，将标记类型设置为`velocity`并将值设置为索引减去 356。例如，索引 380 映射到具有 24 个值的`velocity`标记。 |

表 14.2 概述了事件标记到索引的转换。音符开启标记被分配从 0 到 127 的索引值，其中索引值对应于标记中的音高数。音符关闭标记被分配从 128 到 255 的索引值，索引值是 128 加上音高数。时间偏移标记被分配从 256 到 355 的索引值，索引值是 256 加上时间偏移值。最后，速度标记被分配从 356 到 387 的索引值，索引值是 356 加上速度箱子数。

使用这种标记到索引的映射，我们将每首音乐转换成一系列索引。我们将对此训练数据集中的所有音乐作品应用此转换，并使用生成的序列来训练我们的音乐 Transformer（其细节将在后面解释）。一旦训练完成，我们将使用 Transformer 以序列的形式生成音乐。最后一步是将此序列转换回 MIDI 格式，以便我们可以在计算机上播放和欣赏音乐。

表 14.2 的最后一列提供了将索引转换回事件标记的指导。我们首先根据索引所在的范围确定标记类型。表 14.2 的第二列中的四个范围对应于表的第一列中的四种标记类型。为了获得每种标记类型的值，我们将索引值分别减去 0、128、256 和 356，分别对应四种类型的标记。然后，这些标记化的事件被转换为 MIDI 格式的音符，准备好在计算机上播放。

### 14.1.2 音乐 Transformer 架构

在第九章中，我们构建了一个编码器-解码器 Transformer，在第十一章和第十二章中，我们专注于仅解码器 Transformer。与编码器捕获源语言含义并将其传递给解码器以生成翻译的语言翻译任务不同，音乐生成不需要编码器理解不同的语言。相反，模型根据音乐序列中的先前事件标记生成后续事件标记。因此，我们将为我们的音乐生成任务构建一个仅解码器的 Transformer。

我们的音乐 Transformer，与其他 Transformer 模型一样，利用自注意力机制来捕捉音乐作品中不同音乐事件之间的长距离依赖关系，从而生成连贯且逼真的音乐。尽管我们的音乐 Transformer 在大小上与我们在第十一章和第十二章中构建的 GPT 模型不同，但它具有相同的核心架构。它遵循与 GPT-2 模型相同的结构设计，但尺寸显著更小，这使得在没有超级计算设施的情况下进行训练成为可能。

具体来说，我们的音乐 Transformer 由 6 个解码器层组成，嵌入维度为 512，这意味着每个标记在词嵌入后都由一个 512 维的向量表示。与原始 2017 年论文“Attention Is All You Need”中使用正弦和余弦函数进行位置编码不同，我们使用嵌入层来学习序列中不同位置的位置编码。因此，序列中的每个位置也由一个 512 维的向量表示。为了计算因果自注意力，我们使用 8 个并行注意力头来捕捉序列中标记的不同含义，每个注意力头的维度为 64（512/8）。

与 GPT-2 模型中 50,257 的词汇量相比，我们的模型具有更小的词汇量，为 390（388 个不同的事件标记，加上一个表示序列结束的标记和一个填充较短的序列的标记；我将在后面解释为什么需要填充）。这使得我们可以在音乐 Transformer 中将最大序列长度设置为 2,048，这比 GPT-2 模型中的最大序列长度 1,024 长得多。这种选择是必要的，以便捕捉序列中音乐音符的长期关系。具有这些超参数值，我们的音乐 Transformer 具有 1,016 万参数。

图 14.1 展示了本章我们将创建的音乐 Transformer 的架构。它与你在第十一章和第十二章中构建的 GPT 模型的架构相似。图 14.1 还显示了训练过程中数据通过模型时的大小。

我们构建的音乐 Transformer 的输入包括输入嵌入，如图 14.1 底部所示。输入嵌入是输入序列的词嵌入和位置编码的总和。然后，这个输入嵌入依次通过六个解码器块。

![图片](img/CH14_F01_Liu.png)

图 14.1 展示了音乐 Transformer 的架构。MIDI 格式的音乐文件首先被转换为音乐事件的序列。这些事件随后被标记化并转换为索引。我们将这些索引组织成 2,048 个元素的序列，每个批次包含 2 个这样的序列。输入序列首先进行词嵌入和位置编码；输入嵌入是这两个组件的总和。然后，这个输入嵌入通过六个解码器层进行处理，每个层都利用自注意力机制来捕捉序列中不同音乐事件之间的关系。经过解码器层后，输出经过层归一化以确保训练过程中的稳定性。然后，它通过一个线性层，输出大小为 390，这对应于词汇表中的独特标记数量。这个最终输出代表了序列中下一个音乐事件的预测对数几率。

如第十一章和第十二章所述，每个解码器层由两个子层组成：一个因果自注意力层和一个前馈网络。此外，我们对每个子层应用层归一化和残差连接，以增强模型稳定性和学习能力。

经过解码器层后，输出经过层归一化，然后输入到一个线性层。我们模型中的输出数量对应于词汇表中的独特音乐事件标记的数量，即 390。模型的输出是下一个音乐事件标记的对数几率。

之后，我们将应用 softmax 函数到这些 logits 上，以获得所有可能事件标记的概率分布。该模型被设计用来根据当前标记和音乐序列中所有之前的标记来预测下一个事件标记，使其能够生成连贯且音乐上合理的序列。

### 14.1.3 训练音乐 Transformer

既然我们已经了解了如何构建用于音乐生成的音乐 Transformer，那么让我们概述一下音乐 Transformer 的训练过程。

模型生成的音乐风格受用于训练的音乐作品的影响。我们将使用 Google 的 Magenta 团队的钢琴表演来训练我们的模型。图 14.2 说明了训练音乐生成 Transformer 所涉及的步骤。

![图 14.2 音乐 Transformer 生成音乐的训练过程](img/CH14_F02_Liu.png)

图 14.2 音乐 Transformer 生成音乐的训练过程

与我们在 NLP 任务中采取的方法类似，我们音乐 Transformer 训练过程中的第一步是将原始训练数据转换为数值形式，以便将其输入到模型中。具体来说，我们首先将训练集中的 MIDI 文件转换为音乐音符序列。然后，我们将这些音符进一步标记化，通过将它们转换为 388 个独特的事件/标记中的 1 个。标记化后，我们为每个标记分配一个唯一的索引（即一个整数），将训练集中的音乐作品转换为整数序列（见图 14.2 中的步骤 1）。

接下来，我们将整数序列转换为训练数据，通过将此序列划分为等长的序列（见图 14.2 中的步骤 2）。我们允许每个序列中最多有 2,048 个索引。选择 2,048 允许我们捕捉音乐序列中音乐事件之间的长距离依赖关系，以创建逼真的音乐。这些序列形成我们模型的特征（x 变量）。正如我们在前几章中训练 GPT 模型生成文本时所做的，我们将输入序列窗口向右滑动一个索引，并将其用作训练数据中的输出（y 变量；见图 14.2 中的步骤 3）。这样做迫使我们的模型根据音乐序列中的当前标记和所有之前的标记来预测序列中的下一个音乐标记。

输入和输出对作为音乐 Transformer 的训练数据（x, y）。在训练过程中，你将遍历训练数据。在前向传递中，你将输入序列 x 通过音乐 Transformer（步骤 4）。音乐 Transformer 然后根据模型中的当前参数进行预测（步骤 5）。你通过比较预测的下一个标记与步骤 3 获得的输出来计算交叉熵损失。换句话说，你将模型的预测与真实值（步骤 6）进行比较。最后，你将调整音乐 Transformer 中的参数，以便在下一个迭代中，模型的预测更接近实际输出，最小化交叉熵损失（步骤 7）。该模型本质上是在执行一个多类别分类问题：它从词汇表中的所有独特音乐标记中预测下一个标记。

你将通过多次迭代重复步骤 3 到 7。在每次迭代后，模型参数都会调整以改善下一个标记的预测。这个过程将重复进行 50 个 epoch。

要使用训练好的模型生成新的音乐作品，我们从测试集中获取一个音乐作品，对其进行标记化，并将其转换为一系列长索引。我们将使用前 250 个索引作为提示（200 或 300 个索引将产生类似的结果）。然后，我们要求训练好的音乐 Transformer 生成新的索引，直到序列达到一定长度（例如，1,000 个索引）。然后，我们将索引序列转换回 MIDI 文件，以便在您的计算机上播放。

## 14.2 音乐作品的标记化

在掌握了音乐 Transformer 的结构和其训练方法之后，我们将从第一步开始：对训练数据集中的音乐作品进行标记化和索引。

我们将首先使用基于性能的表示（如第一部分所述）来表示音乐作品为音符，类似于自然语言处理中的原始文本。之后，我们将这些音符划分为一系列事件，类似于自然语言处理中的标记。每个独特的事件将被分配一个不同的索引。利用这个映射，我们将训练数据集中的所有音乐作品转换为索引序列。

接下来，我们将这些索引序列标准化为固定长度，具体为 2,048 个索引的序列，并将它们用作特征输入（x）。通过将窗口向右移动一个索引，我们将生成相应的输出序列（y）。然后，我们将输入和输出（x, y）对分组成批次，为章节后面的音乐 Transformer 训练做准备。

由于我们需要 `pretty_midi` 和 `music21` 库来处理 MIDI 文件，请在 Jupyter Notebook 应用程序的新单元格中执行以下代码行：

```py
!pip install pretty_midi music21
```

### 14.2.1 下载训练数据

我们将从由谷歌 Magenta 团队提供的 MAESTRO 数据集中获取钢琴演奏，该数据集可在[`storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip`](https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip)获得，并下载 ZIP 文件。下载后，解压缩它，并将生成的文件夹/maestro-v2.0.0/移动到您计算机上的/files/目录中。

确保/maestro-v2.0.0/文件夹包含 4 个文件（其中一个应命名为“maestro-v2.0.0.json”）和 10 个子文件夹。每个子文件夹应包含超过 100 个 MIDI 文件。为了熟悉训练数据中音乐片段的声音，尝试使用您喜欢的音乐播放器打开一些 MIDI 文件。

接下来，我们将 MIDI 文件分成训练、验证和测试子集。首先，在/files/maestro-v2.0.0/目录下创建三个子文件夹：

```py
import os

os.makedirs("files/maestro-v2.0.0/train", exist_ok=True)
os.makedirs("files/maestro-v2.0.0/val", exist_ok=True)
os.makedirs("files/maestro-v2.0.0/test", exist_ok=True)
```

为了方便处理 MIDI 文件，访问凯文·杨的 GitHub 仓库[`github.com/jason9693/midi-neural-processor`](https://github.com/jason9693/midi-neural-processor)，下载 processor.py 文件，并将其放置在您计算机上的/utils/文件夹中。或者，您也可以从本书的 GitHub 仓库[`github.com/markhliu/DGAI`](https://github.com/markhliu/DGAI)中获取该文件。我们将使用此文件作为本地模块，将 MIDI 文件转换为一系列索引，反之亦然。这种方法使我们能够专注于开发、训练和利用音乐 Transformer，而无需陷入音乐格式转换的细节。同时，我将提供一个简单的示例，说明这个过程是如何工作的，这样您就可以使用该模块自己将 MIDI 文件和一系列索引之间进行转换。

此外，您还需要从本书的 GitHub 仓库下载 ch14util.py 文件，并将其放置在您计算机上的/utils/目录中。我们将使用 ch14util.py 文件作为另一个本地模块来定义音乐 Transformer 模型。

/maestro-v2.0.0/文件夹中的 maestro-v2.0.0.json 文件包含所有 MIDI 文件及其指定的子集（训练、验证或测试）。基于这些信息，我们将 MIDI 文件分类到三个相应的子文件夹中。

列表 14.3 将训练数据分割为训练、验证和测试子集

```py
import json
import pickle
from utils.processor import encode_midi

file="files/maestro-v2.0.0/maestro-v2.0.0.json"

with open(file,"r") as fb:
    maestro_json=json.load(fb)                            ①

for x in maestro_json:                                    ②
    mid=rf'files/maestro-v2.0.0/{x["midi_filename"]}'
    split_type = x["split"]                               ③
    f_name = mid.split("/")[-1] + ".pickle"
    if(split_type == "train"):
        o_file = rf'files/maestro-v2.0.0/train/{f_name}'
    elif(split_type == "validation"):
        o_file = rf'files/maestro-v2.0.0/val/{f_name}'
    elif(split_type == "test"):
        o_file = rf'files/maestro-v2.0.0/test/{f_name}'
    prepped = encode_midi(mid)
    with open(o_file,"wb") as f:
        pickle.dump(prepped, f)
```

① 加载 JSON 文件

② 遍历训练数据中的所有文件

根据 JSON 文件中的说明，将文件放置在训练、验证或测试子文件夹中

您下载的 JavaScript 对象表示法（JSON）文件将训练数据集中的每个文件分类到三个子集之一：训练、验证和测试。在执行前面的代码列表后，如果您在计算机上的/train/、/val/和/test/文件夹中查看，您应该在每个文件夹中找到许多文件。为了验证这三个文件夹中每个文件夹的文件数量，您可以执行以下检查：

```py
train_size=len(os.listdir('files/maestro-v2.0.0/train'))
print(f"there are {train_size} files in the train set")
val_size=len(os.listdir('files/maestro-v2.0.0/val'))
print(f"there are {val_size} files in the validation set")
test_size=len(os.listdir('files/maestro-v2.0.0/test'))
print(f"there are {test_size} files in the test set")
```

前一个代码块输出的结果是

```py
there are 967 files in the train set
there are 137 files in the validation set
there are 178 files in the test set
```

结果显示，训练、验证和测试子集中分别有 967、137 和 178 首音乐作品。

### 14.2.2 MIDI 文件标记化

接下来，我们将每个 MIDI 文件表示为一串音乐音符。

列表 14.4 将 MIDI 文件转换为音乐音符序列

```py
import pickle
from utils.processor import encode_midi
import pretty_midi
from utils.processor import (_control_preprocess,
    _note_preprocess,_divide_note,
    _make_time_sift_events,_snote2events)

file='MIDI-Unprocessed_Chamber1_MID--AUDIO_07_R3_2018_wav--2'
name=rf'files/maestro-v2.0.0/2018/{file}.midi'               ①

events=[]
notes=[]
song=pretty_midi.PrettyMIDI(name)
for inst in song.instruments:
    inst_notes=inst.notes
    ctrls=_control_preprocess([ctrl for ctrl in 
       inst.control_changes if ctrl.number == 64])
    notes += _note_preprocess(ctrls, inst_notes)             ②
dnotes = _divide_note(notes)                                 ③
dnotes.sort(key=lambda x: x.time)    
for i in range(5):
    print(dnotes[i])   
```

① 从训练数据集中选择一个 MIDI 文件

② 从音乐中提取音乐事件

③ 将所有音乐事件放入列表 dnotes 中

我们已经从训练数据集中选择了一个 MIDI 文件，并使用 processor.py 本地模块将其转换为音乐音符序列。前面代码列表的输出如下：

```py
<[SNote] time: 1.0325520833333333 type: note_on, value: 74, velocity: 86>
<[SNote] time: 1.0442708333333333 type: note_on, value: 38, velocity: 77>
<[SNote] time: 1.2265625 type: note_off, value: 74, velocity: None>
<[SNote] time: 1.2395833333333333 type: note_on, value: 73, velocity: 69>
<[SNote] time: 1.2408854166666665 type: note_on, value: 37, velocity: 64>
```

这里显示的输出显示了 MIDI 文件中的前五个音符。你可能已经注意到输出中的时间表示是连续的。某些音符同时包含`note_on`和`velocity`属性，由于时间表示的连续性，这会导致大量独特的音乐事件，从而复杂化了标记过程。此外，不同的`note_on`和`velocity`值的组合很大（每个都可以假设 128 个不同的值，范围从 0 到 127），导致词汇表大小过大。这反过来又会使得训练变得不切实际。

为了减轻这个问题并减小词汇表大小，我们进一步将这些音乐音符转换为标记化事件：

```py
cur_time = 0
cur_vel = 0
for snote in dnotes:
    events += _make_time_sift_events(prev_time=cur_time,    ①
                                     post_time=snote.time)
    events += _snote2events(snote=snote, prev_vel=cur_vel)  ②
    cur_time = snote.time
    cur_vel = snote.velocity    
indexes=[e.to_int() for e in events]   
for i in range(15):                                         ③
    print(events[i]) 
```

① 将时间离散化以减少独特事件的数目

② 将音乐音符转换为事件

③ 打印出前 15 个事件

输出如下：

```py
<Event type: time_shift, value: 99>
<Event type: time_shift, value: 2>
<Event type: velocity, value: 21>
<Event type: note_on, value: 74>
<Event type: time_shift, value: 0>
<Event type: velocity, value: 19>
<Event type: note_on, value: 38>
<Event type: time_shift, value: 17>
<Event type: note_off, value: 74>
<Event type: time_shift, value: 0>
<Event type: velocity, value: 17>
<Event type: note_on, value: 73>
<Event type: velocity, value: 16>
<Event type: note_on, value: 37>
<Event type: time_shift, value: 0>
```

音乐作品现在由四种类型的事件表示：音符开启、音符关闭、时间移动和速度。每种事件类型包含不同的值，总共产生 388 个独特的事件，如前面表格 14.2 中详细说明。将 MIDI 文件转换为这种独特事件序列的具体细节对于构建和训练音乐 Transformer 不是必要的。因此，我们不会深入探讨这个话题；感兴趣的读者可以参考前面提到的 Huang 等人（2018）的研究。你需要知道的是如何使用 processor.py 模块将 MIDI 文件转换为索引序列，反之亦然。在下面的子节中，你将学习如何完成这个任务。

### 14.2.3 准备训练数据

我们已经学会了如何将音乐作品转换为标记，然后转换为索引。下一步涉及准备训练数据，以便我们可以在本章后面利用它来训练音乐 Transformer。为了实现这一点，我们定义了以下列表中所示的`create_xys()`函数。

列表 14.5 创建训练数据

```py
import torch,os,pickle

max_seq=2048
def create_xys(folder):  
    files=[os.path.join(folder,f) for f in os.listdir(folder)]
    xys=[]
    for f in files:
        with open(f,"rb") as fb:
            music=pickle.load(fb)
        music=torch.LongTensor(music)      
        x=torch.full((max_seq,),389, dtype=torch.long)
        y=torch.full((max_seq,),389, dtype=torch.long)      ①
        length=len(music)
        if length<=max_seq:
            print(length)
            x[:length]=music                                ②
            y[:length-1]=music[1:]                          ③
            y[length-1]=388                                 ④
        else:
            x=music[:max_seq]
            y=music[1:max_seq+1]   
        xys.append((x,y))
    return xys
```

① 创建长度为 2,048 个索引的(x, y)序列，并将索引 399 设置为填充索引

② 使用最多 2,048 个索引的序列作为输入

③ 将窗口向右滑动一个索引，并使用它作为输出

④ 设置结束索引为 388

正如我们在整本书中反复看到的那样，在序列预测任务中，我们使用一个序列 x 作为输入。然后我们将序列向右移动一个位置以创建输出序列。这种方法迫使模型根据序列中的当前元素和所有前面的元素来预测下一个元素。为了为我们的音乐 Transformer 准备训练数据，我们将构建(x, y)对，其中 x 是输入，y 是输出。x 和 y 都包含 2,048 个索引——足够长以捕捉序列中音乐音符的长期关系，但又不至于太长而阻碍训练过程。

我们将遍历下载的训练数据集中的所有音乐作品。如果一个音乐作品的长度超过 2,048 个索引，我们将使用前 2,048 个索引作为输入 x。对于输出 y，我们将使用从第二个位置到第 2,049 个位置的索引。在音乐作品长度小于或等于 2,048 个索引的罕见情况下，我们将使用索引 389 填充序列，以确保 x 和 y 的长度都是 2,048 个索引。此外，我们使用索引 388 来表示序列 y 的结束。

如第一部分所述，总共有 388 个独特的事件标记，索引从 0 到 387。由于我们使用 388 来表示 y 序列的结束，并使用 389 来填充序列，因此总共有 390 个独特的索引，范围从 0 到 389。

我们现在可以将`create_xys()`函数应用于训练子集：

```py
trainfolder='files/maestro-v2.0.0/train'
train=create_xys(trainfolder)
```

输出如下

```py
15
5
1643
1771
586
```

这表明在训练子集中的 967 首音乐作品中，只有 5 首的长度小于 2,048 个索引。它们的长度在之前的输出中显示。

我们还把`create_xys()`函数应用于验证和测试子集：

```py
valfolder='files/maestro-v2.0.0/val'
testfolder='files/maestro-v2.0.0/test'
print("processing the validation set")
val=create_xys(valfolder)
print("processing the test set")
test=create_xys(testfolder)
```

输出如下

```py
processing the validation set
processing the test set
1837
```

这表明验证子集中的所有音乐作品长度都超过 2,048 个索引。测试子集中只有一首音乐作品的长度小于 2,048 个索引。

让我们打印出验证子集中的一份文件，看看它是什么样子：

```py
val1, _ = val[0]
print(val1.shape)
print(val1)
```

输出如下：

```py
torch.Size([2048])
tensor([324, 366,  67,  ...,  60, 264, 369])
```

验证集第一对中的 x 序列长度为 2,048 个索引，具有诸如 324、367 等值。让我们使用`processor.py`模块来解码序列到一个 MIDI 文件，这样你就可以听到它的声音：

```py
from utils.processor import decode_midi

file_path="files/val1.midi"
decode_midi(val1.cpu().numpy(), file_path=file_path)
```

`decode_midi()`函数将索引序列转换为 MIDI 文件，可以在你的电脑上播放。在运行前面的代码块后，用电脑上的音乐播放器打开 val1.midi 文件，听听它的声音。

练习 14.1

使用`processor.py`本地模块中的`decode_midi()`函数将训练子集中的第一首音乐作品转换为 MIDI 文件。将其保存为 train1.midi 到你的电脑上。用电脑上的音乐播放器打开它，感受一下我们用于训练数据类型的音乐。

最后，我们创建一个数据加载器，以便数据以批次的格式进行训练：

```py
from torch.utils.data import DataLoader

batch_size=2
trainloader=DataLoader(train,batch_size=batch_size,
                       shuffle=True)
```

为了防止您的 GPU 内存耗尽，我们将使用 2 个批处理大小，因为我们创建了非常长的序列，每个序列包含 2,048 个索引。如果需要，可以将批处理大小减少到 1 或切换到 CPU 训练。

有了这些，我们的训练数据已经准备好了。在接下来的两个部分中，我们将从头开始构建一个音乐 Transformer，然后使用我们刚刚准备好的训练数据进行训练。

## 14.3 构建用于生成音乐的 GPT

现在我们已经准备好了训练数据，我们将从头开始构建一个用于音乐生成的 GPT 模型。这个模型的架构将与我们在第十一章中开发的 GPT-2XL 模型和第十二章中的文本生成器相似。然而，由于我们选择的特定超参数，我们的音乐 Transformer 的大小将有所不同。

为了节省空间，我们将模型构建放在本地模块 ch14util.py 中。在这里，我们的重点是音乐 Transformer 选择使用的超参数。具体来说，我们将决定`n_layer`的值，即模型中解码器的层数；`n_head`，用于计算因果自注意力的并行头数；`n_embd`，嵌入维度；以及`block_size`，输入序列中的标记数。

### 14.3.1 音乐 Transformer 中的超参数

打开您之前从本书的 GitHub 仓库下载的文件 ch14util.py。在里面，您会发现几个函数和类，它们与第十二章中定义的完全相同。

正如本书中我们看到的所有 GPT 模型一样，解码器块中的前馈网络使用高斯误差线性单元（GELU）激活函数。因此，我们在 ch14util.py 中定义了一个 GELU 类，这与我们在第十二章中做的一样。

我们使用一个`Config()`类来存储音乐 Transformer 中使用的所有超参数：

```py
from torch import nn
class Config():
    def __init__(self):
        self.n_layer = 6
        self.n_head = 8
        self.n_embd = 512
        self.vocab_size = 390
        self.block_size = 2048 
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1
config=Config()
device="cuda" if torch.cuda.is_available() else "cpu"
```

`Config()`类中的属性作为我们音乐 Transformer 的超参数。我们将`n_layer`属性的值设置为 6，表示我们的音乐 Transformer 由 6 个解码器层组成。这比我们在第十二章中构建的 GPT 模型中的解码器层数要多。每个解码器层处理输入序列并引入一个抽象或表示的层次。随着信息穿越更多层，模型能够捕捉到数据中更复杂的模式和关系。这种深度对于我们的音乐 Transformer 理解并生成复杂的音乐作品至关重要。

`n_head`属性设置为 8，表示在计算因果自注意力时，我们将查询 Q、键 K 和值 V 向量分为八个并行头。`n_embd`属性设置为 512，表示嵌入维度为 512：每个事件标记将由一个包含 512 个值的向量表示。`vocab_size`属性由词汇表中的唯一标记数量确定，为 390。如前所述，有 388 个唯一的事件标记，我们添加了 1 个标记来表示序列的结束，并添加了另一个标记来填充较短的序列，以便所有序列的长度都为 2,048。`block_size`属性设置为 2,048，表示输入序列最多包含 2,048 个标记。我们将 dropout 比率设置为 0.1，与第十一章和第十二章相同。

与所有 Transformer 一样，我们的音乐 Transformer 使用自注意力机制来捕捉序列中不同元素之间的关系。因此，我们在本地模块 ch14util 中定义了一个`CausalSelfAttention()`类，它与第十二章中定义的`CausalSelfAttention()`类相同。

### 14.3.2 构建音乐 Transformer

我们将前馈网络与因果自注意力子层结合形成一个解码块（即解码层）。我们对每个子层应用层归一化和残差连接以提高稳定性和性能。为此，我们在本地模块中定义了一个`Block()`类来创建解码块，它与我们在第十二章中定义的`Block()`类相同。

然后，我们在音乐 Transformer 的上方堆叠六个解码块，形成其主体。为了实现这一点，我们在本地模块中定义了一个`Model()`类。正如我们在本书中看到的所有 GPT 模型一样，我们使用通过 PyTorch 中的`Embedding()`类学习到的位置编码，而不是原始 2017 年论文“Attention Is All You Need”中的固定位置编码。有关两种位置编码方法之间的差异，请参阅第十一章。

模型的输入由对应于词汇表中音乐事件标记的索引序列组成。我们将输入通过词嵌入和位置编码传递，并将两者相加以形成输入嵌入。然后，输入嵌入通过六个解码层。之后，我们对输出应用层归一化，并将其连接到一个线性头，以便输出的数量为 390，即词汇表的大小。输出是词汇表中 390 个标记的 logits。稍后，我们将对 logits 应用 softmax 激活函数，以获得生成音乐时词汇表中唯一音乐标记的概率分布。

接下来，我们将通过实例化我们在本地模块中定义的`Model()`类来创建我们的音乐 Transformer：

```py
from utils.ch14util import Model

model=Model(config)
model.to(device)
num=sum(p.numel() for p in model.transformer.parameters())
print("number of parameters: %.2fM" % (num/1e6,))
print(model)
```

输出是

```py
number of parameters: 20.16M
Model(
  (transformer): ModuleDict(
    (wte): Embedding(390, 512)
    (wpe): Embedding(2048, 512)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-5): 6 x Block(
        (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=512, out_features=1536, bias=True)
          (c_proj): Linear(in_features=512, out_features=512, bias=True)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): ModuleDict(
          (c_fc): Linear(in_features=512, out_features=2048, bias=True)
          (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          (act): GELU()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=512, out_features=390, bias=False)
)
```

我们的音乐 Transformer 由 2016 万个参数组成，这个数字比拥有超过 15 亿个参数的 GPT-2XL 小得多。尽管如此，我们的音乐 Transformer 的规模超过了我们在第十二章中构建的仅包含 512 万个参数的文本生成器。尽管存在这些差异，所有三个模型都是基于仅解码器 Transformer 架构。差异仅在于超参数，如嵌入维度、解码器层数、词汇量大小等。

## 14.4 训练和使用音乐 Transformer

在本节中，您将使用本章前期准备好的训练数据批次来训练您刚刚构建的音乐 Transformer。为了加快过程，我们将对模型进行 100 个周期的训练，然后停止训练过程。对于感兴趣的人来说，您可以使用验证集来确定何时停止训练，根据模型在验证集上的性能，就像我们在第二章中所做的那样。

一旦模型训练完成，我们将以一系列索引的形式提供给它一个提示。然后，我们将请求训练好的音乐 Transformer 生成下一个索引。这个新的索引被附加到提示中，更新的提示被送回模型进行另一个预测。这个过程会迭代重复，直到序列达到一定的长度。

与第十三章中生成的音乐不同，我们可以通过应用不同的温度来控制音乐作品的创造性。

### 14.4.1 训练音乐 Transformer

和往常一样，我们将使用 Adam 优化器进行训练。鉴于我们的音乐 Transformer 实质上执行的是一个多类别分类任务，我们将使用交叉熵损失作为我们的损失函数：

```py
lr=0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func=torch.nn.CrossEntropyLoss(ignore_index=389)
```

在之前的损失函数中，`ignore_index=389` 参数指示程序在目标序列（即序列 y）中遇到索引 389 时忽略它，因为这个索引仅用于填充目的，并不代表音乐作品中的任何特定事件标记。

我们将接着对模型进行 100 个周期的训练。

列表 14.6 训练音乐 Transformer 生成音乐

```py
model.train()  
for i in range(1,101):
    tloss = 0.
    for idx, (x,y) in enumerate(trainloader):              ①
        x,y=x.to(device),y.to(device)
        output = model(x)
        loss=loss_func(output.view(-1,output.size(-1)),
                           y.view(-1))                     ②
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1)     ③
        optimizer.step()                                   ④
    print(f'epoch {i} loss {tloss/(idx+1)}') 
torch.save(model.state_dict(),f'files/musicTrans.pth')     ⑤
```

① 遍历所有训练数据批次

② 将模型预测与实际输出进行比较

③ 将梯度范数裁剪到 1

④ 调整模型参数以最小化损失

⑤ 训练后保存模型

在训练过程中，我们将所有输入序列 x 在一个批次中通过模型来获得预测。然后，我们将这些预测与批次中的相应输出序列 y 进行比较，并计算交叉熵损失。之后，我们调整模型参数以最小化这个损失。需要注意的是，我们已经将梯度范数裁剪到 1，以防止潜在的梯度爆炸问题。

如果您有 CUDA 支持的 GPU，上述训练过程大约需要 3 小时。训练完成后，训练好的模型权重 musicTrans.pth 将保存在您的计算机上。或者，您可以从我的网站[`mng.bz/V2pW`](https://mng.bz/V2pW)下载训练好的权重。

### 14.4.2 使用训练好的 Transformer 进行音乐生成

现在我们已经训练了一个音乐 Transformer，我们可以进行音乐生成了。

与文本生成过程类似，音乐生成始于将一系列索引（代表事件标记）作为提示输入到模型中。我们将从测试集中选择一首音乐作品，并使用前 250 个音乐事件作为提示：

```py
from utils.processor import decode_midi

prompt, _  = test[42]
prompt = prompt.to(device)
len_prompt=250
file_path = "files/prompt.midi"
decode_midi(prompt[:len_prompt].cpu().numpy(),
            file_path=file_path)
```

我们随机选择了一个索引（在我们的例子中是 42）并使用它从测试子集中检索一首歌曲。我们只保留前 250 个音乐事件，稍后我们将这些事件输入到训练好的模型中以预测下一个音乐事件。为了比较，我们将提示保存为 MIDI 文件 prompt.midi 到本地文件夹中。

练习 14.2

使用`decode_midi()`函数将测试集中第二首音乐的第一个 250 个音乐事件转换为 MIDI 文件。将其保存在您的计算机上的 prompt2.midi。

为了简化音乐生成过程，我们将定义一个`sample()`函数。该函数接受一个索引序列作为输入，代表一小段音乐。然后它迭代地预测并追加新的索引到序列中，直到达到指定的长度`seq_length`。实现方式如下所示。

列表 14.7 音乐生成中的`sample()`函数

```py
softmax=torch.nn.Softmax(dim=-1)
def sample(prompt,seq_length=1000,temperature=1):
    gen_seq=torch.full((1,seq_length),389,dtype=torch.long).to(device)
    idx=len(prompt)
    gen_seq[..., :idx]=prompt.type(torch.long).to(device)    
    while(idx < seq_length):                                       ①
        y=softmax(model(gen_seq[..., :idx])/temperature)[...,:388] ②
        probs=y[:, idx-1, :]
        distrib=torch.distributions.categorical.Categorical(probs=probs)
        next_token=distrib.sample()                                ③
        gen_seq[:, idx]=next_token
        idx+=1
    return gen_seq[:, :idx]                                        ④
```

① 生成新的索引直到序列达到一定长度

② 将预测值除以温度，然后在 logits 上应用 softmax 函数

③ 从预测的概率分布中采样以生成新的索引

④ 输出整个序列

`sample()`函数的一个参数是温度，它调节生成音乐的创造性。如有需要，请参考第八章了解其工作原理。由于我们可以仅通过温度参数调整生成音乐的原创性和多样性，因此在此实例中省略了`top-K`采样以简化过程。正如我们在本书中之前三次讨论过`top-K`采样（在第 8、11 和 12 章），感兴趣的读者可以尝试将`top-K`采样结合到`sample()`函数中。

接下来，我们将加载训练好的权重到模型中：

```py
model.load_state_dict(torch.load("files/musicTrans.pth",
    map_location=device))
model.eval()
```

然后，我们调用`sample()`函数来生成一段音乐：

```py
from utils.processor import encode_midi

file_path = "files/prompt.midi"
prompt = torch.tensor(encode_midi(file_path))
generated_music=sample(prompt, seq_length=1000)
```

首先，我们利用处理器模块中的`encode_midi()`函数将 MIDI 文件 prompt.midi 转换为索引序列。然后我们使用这个序列作为`sample()`函数中的提示来生成由 1,000 个索引组成的音乐作品。

最后，我们将生成的索引序列转换为 MIDI 格式：

```py
music_data = generated_music[0].cpu().numpy()
file_path = 'files/musicTrans.midi'
decode_midi(music_data, file_path=file_path)
```

我们在 processor.py 模块中使用了`decode_midi()`函数，将生成的索引序列转换成您电脑上的 MIDI 文件，即 musicTrans.midi。在您的电脑上打开这两个文件，prompt.midi 和 musicTrans.midi，并聆听它们。prompt.midi 中的音乐大约持续 10 秒。musicTrans.midi 中的音乐大约持续 40 秒，最后的 30 秒是由音乐变换器生成的新音乐。生成的音乐应该听起来像我网站上的音乐作品：[`mng.bz/x6dg`](https://mng.bz/x6dg)。

上述代码块可能产生类似于以下输出的结果：

```py
info removed pitch: 52
info removed pitch: 83
info removed pitch: 55
info removed pitch: 68
```

在生成的音乐中，可能会有一些音符需要被移除。例如，如果生成的音乐作品尝试关闭音符 52，但音符 52 最初从未被开启，那么我们就不能关闭它。因此，我们需要移除这样的音符。

练习 14.3

使用训练好的音乐变换器模型生成包含 1,200 个音符的音乐作品，保持温度参数为 1。使用您在 14.2 练习中生成的 prompt2.midi 文件中的索引序列作为提示。将生成的音乐保存到您电脑上的名为 musicTrans2.midi 的文件中。

您可以通过将温度参数设置为大于 1 的值来提高音乐的创意，如下所示：

```py
file_path = "files/prompt.midi"
prompt = torch.tensor(encode_midi(file_path))
generated_music=sample(prompt, seq_length=1000,temperature=1.5)
music_data = generated_music[0].cpu().numpy()
file_path = 'files/musicHiTemp.midi'
decode_midi(music_data, file_path=file_path)
```

我们将温度设置为 1.5。生成的音乐保存为 musicHiTemp.midi 文件在您的电脑上。打开该文件并聆听，看看您是否能辨别出与 musicTrans.midi 文件中的音乐相比有任何差异。

练习 14.4

使用训练好的音乐变换器模型生成包含 1,000 个索引的音乐作品，将温度参数设置为 0.7。使用 prompt.midi 文件中的索引序列作为提示。将生成的音乐保存到您电脑上的名为 musicLowTemp.midi 的文件中。打开此文件聆听生成的音乐，看看新作品与 musicTrans.midi 文件中的音乐之间是否有可辨别的差异。

在本章中，您已经学习了如何从头开始构建和训练音乐变换器，基于您在前面章节中使用的仅解码器变换器架构。在下一章中，您将探索基于扩散的模型，这些模型是像 OpenAI 的 DALL-E 2 和 Google 的 Imagen 这样的文本到图像变换器的核心。

## 摘要

+   音乐的性能表示使我们能够将音乐作品表示为一串音符，这些音符包括控制信息和速度值。这些音符可以进一步简化为四种音乐事件：音符开启、音符关闭、时间移动和速度。每种事件类型可以假设各种值。因此，我们可以将音乐作品转换为一串标记，然后转换为索引。

+   音乐 Transformer 架构是对最初为 NLP 任务设计的 Transformer 架构进行适配，用于音乐生成。该模型旨在通过学习大量现有音乐数据集来生成音乐音符序列。它通过识别训练数据中各种音乐元素之间的模式、结构和关系，被训练来根据前面的音符预测序列中的下一个音符。

+   正如文本生成一样，我们可以使用温度来调节生成音乐的创造力。

* * *

^(1) Chloe Veltman，2024 年 3 月 15 日。“仅仅因为你的最爱歌手已经去世，并不意味着你不能看到他们‘现场’。” [`mng.bz/r1de`](https://mng.bz/r1de)。

^(2) Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Noam Shazeer, Ian Simon, Curtis Hawthorne, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, 和 Douglas Eck，2018 年，“Music Transformer。” [`arxiv.org/abs/1809.04281`](https://arxiv.org/abs/1809.04281)。

^(3) 例如，参见 Hawthorne 等人，2018 年，“使用 MAESTRO 数据集实现分解钢琴音乐建模和生成。” [`arxiv.org/abs/1810.12247`](https://arxiv.org/abs/1810.12247)。
