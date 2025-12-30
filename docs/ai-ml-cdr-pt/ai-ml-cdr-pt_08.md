# 第七章\. 自然语言处理中的循环神经网络

在第五章中，你看到了如何标记和序列化文本，将句子转换为可以被神经网络输入的数字张量。然后你在第六章中扩展了这一点，通过查看嵌入，这些嵌入构成了一种让具有相似意义的单词聚集在一起的方法，以便计算情感。正如你所看到的，通过构建讽刺分类器，这确实很有效。但是，这有一个限制：即句子不仅仅是单词的集合——而且，单词出现的*顺序*往往决定了它们的整体意义。此外，形容词可以增加或改变它们旁边名词的意义。例如，单词*蓝色*从情感角度来看可能没有意义，*天空*也是如此，但是当你把它们放在一起得到*蓝天*时，它表明了一种通常积极的明确情感。最后，一些名词可能对其他名词有资格限制，例如在*雨云*、*写字台*和*咖啡杯*中。

为了考虑这样的序列，你需要采取一个额外的方法：你需要在模型架构中考虑*递归*。在本章中，你将了解不同的实现方式。我们将探讨如何学习序列信息，以及你如何利用这些信息来创建一种能够更好地理解文本的模型：*循环神经网络*（RNN）。

# 递归的基础

为了理解递归可能的工作方式，让我们首先考虑本书迄今为止使用的模型的局限性。最终，创建一个模型看起来有点像图 7-1。你提供数据和标签，并定义模型架构，然后模型学习适合数据的规则，这些规则随后作为应用程序编程接口（API）提供给你，为你提供对未来数据的预测标签。

![](img/aiml_0701.png)

###### 图 7-1\. 模型创建的高级视图

但是，如你所见，数据是整体处理的。没有涉及粒度，也没有努力理解数据发生的顺序。这意味着单词*蓝色*和*天空*在句子“今天我很沮丧，因为天空是灰的”和“今天我很高兴，有一片美丽的蓝天”中没有任何不同的意义。对我们来说，这些词的使用差异是明显的，但对具有这里所示架构的模型来说，实际上并没有差异。

那么，我们如何解决这个问题呢？让我们首先探索递归的本质，然后你将能够看到基本的 RNN 是如何工作的。

考虑著名的斐波那契数列。如果你不熟悉它，我已经在图 7-2 中列出了一些。

![](img/aiml_0702.png)

###### 图 7-2\. 斐波那契序列的前几个数字

这个序列背后的想法是每个数字都是它前面两个数字的和。所以如果我们从 1 和 2 开始，下一个数字是 1 + 2，即 3。接下来的数字是 2 + 3，即 5，然后是 3 + 5，即 8，以此类推。

我们可以将这个放在计算图中，得到图 7-3。

![aiml_0703.png](img/aiml_0703.png)

###### 图 7-3\. 斐波那契序列的计算图表示

在这里，您可以看到我们将 1 和 2 输入到函数中，得到输出 3。然后我们将第二个参数（2）带到下一步，并连同前一步的输出（3）一起输入到函数中。这个输出的结果是 5，然后它与前一步的第二个参数（3）一起输入到函数中，产生输出 8。这个过程无限期地继续下去，每个操作都依赖于之前的操作。左上角的 1 在过程中“存活”下来——它是被输入到第二个操作中的 3 的一个元素，它是被输入到第三个操作中的 5 的一个元素，以此类推。因此，1 的一些本质在整个序列中得到了保留，尽管它对整体值的影响减弱了。

这与循环神经元的架构类似。您可以在图 7-4 中看到循环神经元的典型表示。

![aiml_0704.png](img/aiml_0704.png)

###### 图 7-4\. 循环神经元

一个值*x*在时间步长被输入到函数*F*中，所以它通常被标记为*x*[*t*]。这在该时间步长产生一个输出*y*，通常被标记为*y*[*t*]。它还产生一个传递到下一步的值，这由从*F*到自身的箭头表示。

如果您查看循环神经元如何在时间步长中相互工作，这会使得这个过程更加清晰，您可以在图 7-5 中看到这一点。

在这里，*x*[0]被操作以得到*y*[0]和一个传递的值。下一步得到这个值和*x*[1]，并产生*y*[1]和一个传递的值。接下来的一步得到这个值和*x*[2]，并产生*y*[2]和一个传递的值，依此类推。这与我们看到的斐波那契序列类似，而且我在尝试记住 RNN 的工作方式时，总是发现这是一个有用的记忆法。

![aiml_0705.png](img/aiml_0705.png)

###### 图 7-5\. 时间步长中的循环神经元

# 扩展语言中的递归

在上一节中，您看到了一个 RNN 如何在多个时间步长上操作以帮助在序列中维持上下文。确实，我们将在本书的后面部分使用 RNN 进行序列建模——但在处理语言时，使用像图 7-4 和图 7-5 中展示的简单 RNN 时，可能会错过一些细微之处。就像前面提到的斐波那契数列示例一样，携带的上下文量会随着时间的推移而减少。第 1 步神经元输出的影响在第 2 步时很大，在第 3 步时较小，在第 4 步时更小，依此类推。因此，如果我们有一个句子像“今天有一个美丽的蓝色<某物>”，那么单词*蓝色*将对下一个单词可能是什么有强烈的影响：我们可以猜测它很可能是*天空*。但是，句子中较早的部分的上下文怎么办？例如，考虑句子“我在爱尔兰生活过，所以在高中时，我不得不学习如何说和写<某物>。”

那个<某物>是*盖尔语*，但真正给我们这个上下文的单词是*爱尔兰*，它在句子中要早得多。因此，为了能够识别<某物>应该是什么，我们需要一种方法来在更长的距离上保持上下文。RNN 的短期记忆需要更长，为此，人们发明了一种称为*长短期记忆*（LSTM）的架构增强。

虽然我不会深入探讨 LSTM 工作原理的底层架构，但图 7-6 中展示的高级图解已经清楚地说明了主要观点。要了解更多关于 LSTM 内部操作的信息，请查看 Christopher Olah 关于此主题的出色[博客文章](https://oreil.ly/6KcFA)。

LSTM 架构通过添加一个“细胞状态”来增强基本的 RNN，这使得上下文不仅可以从一步到下一步维持，还可以在整个步骤序列中维持。记住这些是像神经元一样学习的神经元，你可以看到这种增强确保了随着时间的推移，重要的上下文将被学习。

![图片](img/aiml_0706.png)

###### 图 7-6. LSTM 架构的高级视图

LSTM 的一个重要部分是它可以*双向*——时间步长可以向前和向后迭代，以便可以从两个方向学习上下文。通常，一个单词的上下文可以来自句子中的*之后*，而不仅仅是之前。

请参阅图 7-7 以了解其高级视图。

![图片](img/aiml_0707.png)

###### 图 7-7. LSTM 双向架构的高级视图

这就是从 0 到`number_of_steps`方向的评估方式，也是从`number_of_steps`到 0 的评估方式。在每一步，*y*结果是对“正向”传递和“反向”传递的聚合。您可以在图 7-8 中看到这一点。

![图片](img/aiml_0708.png)

###### 图 7-8. 双向 LSTM

当涉及到网络的训练时，很容易将 LSTM 的双向性质与 *forward* 和 *backward* 这些术语混淆，但它们是非常不同的。当我提到正向和反向传递时，我指的是设置神经元参数以及它们从学习过程中进行更新的过程。不要将 LSTM 关注的值（作为序列中的下一个或前一个标记）与此混淆。

此外，将每个时间步的每个神经元视为 F0, F1, F2 等。时间步的方向已显示，因此正向计算 F1 的值为 F1(->)，反向计算为 (<-)F1\. 这些值的总和给出了该时间步的 *y* 值。此外，细胞状态是双向的，这在管理句子中的上下文中非常有用。再次考虑句子“我在爱尔兰生活过，所以在高中时，我不得不学习如何说和写 <something>”，你可以看到 <something> 是如何通过上下文词 *Ireland* 被认定为 *Gaelic* 的。但如果情况相反：“我在 <this country> 生活过，所以在高中时，我不得不学习如何说和写 Gaelic”？你可以看到通过句子中的 *backward* 追溯，我们可以了解 <this country> 应该是什么。因此，使用双向 LSTM 对于理解文本中的情感非常强大。（而且正如你将在第八章 Chapter 8 中看到的那样，它们在生成文本方面也非常强大！）

当然，LSTM（尤其是双向 LSTM）有很多内容，因此预期训练会较慢。这就是为什么值得投资 GPU，或者至少在 Google Colab 中使用托管 GPU 的原因。

# 使用 RNNs 创建文本分类器

在第六章 Chapter 6 中，你通过使用嵌入来创建 Sarcasm 数据集的分类器进行了实验。在那个例子中，你将单词转换为向量，然后聚合它们，并将它们输入到密集层进行分类。但是，当你使用 RNN 层（如 LSTM）时，你不需要进行聚合，可以直接将嵌入层的输出输入到循环层。当涉及到循环层的维度时，你经常会看到的一个经验法则是它的大小与嵌入维度相同。这不是必需的，但它可以是一个好的起点。此外，请注意，虽然我在第六章 Chapter 6 中提到嵌入维度通常是词汇表大小的四次方根，但在使用 RNN 时，你经常会看到这个规则可能被忽略，因为这会使循环层的大小变得太小。

在这个例子中，我将隐藏层中神经元的数量作为一个起点，你可以从这里进行实验。

例如，您可以将您在 第六章 中开发的讽刺分类器的简单模型架构更新为以下内容，以使用双向 LSTM：

```py
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, 
                 hidden_dim=24, lstm_layers=1):
        super(TextClassificationModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length)

        # Get embeddings
        embedded = self.embedding(x)  
        # Shape: (batch_size, sequence_length, embedding_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)  
        # Shape: (batch_size, sequence_length, hidden_dim)

        # Transpose for global pooling 
        # (expecting: batch, channels, sequence_length)
        lstm_out = lstm_out.transpose(1, 2)  
        # Shape: (batch_size, hidden_dim, sequence_length)

        # Apply global pooling
        pooled = self.global_pool(lstm_out)  
        # Shape: (batch_size, hidden_dim, 1)
        pooled = pooled.squeeze(–1)  # Shape: (batch_size, hidden_dim)

        # Pass through fully connected layers
        x = self.relu(self.fc1(pooled))
        x = self.sigmoid(self.fc2(x))

        return x
```

您可以将损失函数和分类器设置为这个。注意，学习率 LR 是 0.001，或 1e–3：

```py
# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, 
                       betas=(0.9, 0.999), amsgrad=False)
```

当您打印出模型架构摘要时，您会看到如下内容：

```py
==========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================
TextClassificationModel                  [32, 1]                   --
├─Embedding: 1-1                         [32, 85, 7]               14,000
├─LSTM: 1-2                              [32, 85, 48]              6,336
├─AdaptiveAvgPool1d: 1-3                 [32, 48, 1]               --
├─Linear: 1-4                            [32, 24]                  1,176
├─ReLU: 1-5                              [32, 24]                  --
├─Linear: 1-6                            [32, 1]                   25
├─Sigmoid: 1-7                           [32, 1]                   --
==========================================================================
Total params: 21,537
Trainable params: 21,537
Non-trainable params: 0
Total mult-adds (M): 17.72
==========================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 1.20
Params size (MB): 0.09
Estimated Total Size (MB): 1.31
==========================================================================
```

注意，词汇量大小为 2,000，嵌入维度为 7。这给嵌入层带来了 14,000 个参数，双向层将有 48 个神经元（24 个输出，24 个回传）和 85 个字符的序列长度

图 7-9 展示了在超过三个时期使用此方法训练的结果。

这给我们一个只有 21,537 个参数的网络。如您所见，该网络在训练数据上的准确率迅速攀升至 85%，但验证数据在约 75% 处停滞不前。这与我们之前得到的结果相似，但检查 图 7-10 中的损失图表显示，尽管测试集在 15 个时期后损失发散，但验证损失开始增加，表明我们出现了过拟合。

![](img/aiml_0709.png)

###### 图 7-9\. 在 30 个时期内 LSTM 的准确率

![](img/aiml_0710.png)

###### 图 7-10\. 在 30 个时期内使用 LSTM 的损失

然而，这只是一个使用具有 24 个神经元的隐藏层的单个 LSTM 层。在下一节中，您将看到如何使用堆叠的 LSTM 并探讨其对分类此数据集准确率的影响。

## 堆叠 LSTM

在上一节中，您看到了如何使用 LSTM 层在嵌入层之后来帮助分类讽刺数据集的内容。但是，LSTM 可以堆叠在一起，这种方法在许多最先进的 NLP 模型中都有应用。

使用 PyTorch 堆叠 LSTM 非常简单。您只需将其作为额外的层添加，就像添加任何其他层一样，但您需要小心指定维度。例如，如果第一个 LSTM 有 *x* 个隐藏层，那么下一个 LSTM 将有 *x* 个输入。如果 LSTM 是双向的，那么下一个需要加倍大小。以下是一个示例：

```py
# First LSTM layer
self.lstm1 = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    num_layers=lstm_layers,
    batch_first=True,
    bidirectional=True
)

# Second LSTM layer
# Note: Input size is hidden_dim*2 because first LSTM is bidirectional.
self.lstm2 = nn.LSTM(
    input_size=hidden_dim * 2,
    hidden_size=hidden_dim,
    num_layers=lstm_layers,
    batch_first=True,
    bidirectional=True
)

```

注意，第一层的 `input_size` 是嵌入维度，因为它位于嵌入层之前。第二个 LSTM 的输入大小为 (`hidden_dim * 2`)，因为第一个 LSTM 的输出大小就是该尺寸，考虑到它是双向的。

模型架构将看起来像这样：

```py
==========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================
TextClassificationModel                  [32, 1]                   --
├─Embedding: 1-1                         [32, 85, 7]               14,000
├─LSTM: 1-2                              [32, 85, 48]              6,336
├─LSTM: 1-3                              [32, 85, 48]              14,208
├─AdaptiveAvgPool1d: 1-4                 [32, 48, 1]               --
├─Linear: 1-5                            [32, 24]                  1,176
├─ReLU: 1-6                              [32, 24]                  --
├─Linear: 1-7                            [32, 1]                   25
├─Sigmoid: 1-8                           [32, 1]                   --
==========================================================================
Total params: 35,745
Trainable params: 35,745
Non-trainable params: 0
Total mult-adds (M): 56.37
==========================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 2.25
Params size (MB): 0.14
Estimated Total Size (MB): 2.41
==========================================================================

```

添加额外的层将给我们带来大约 14,000 个额外的参数需要学习，这增加了大约 75%。因此，它可能会减慢网络的运行速度，但如果收益合理，成本相对较低。

训练了三百个周期后，结果看起来像 图 7-11。虽然验证集上的准确率保持平稳，但检查损失（如图 7-12 所示）却讲述了一个不同的故事。如图 7-12 所示，虽然训练和验证的准确率看起来不错，但验证损失迅速上升，这是一个明显的过拟合信号。

![](img/aiml_0711.png)

###### 图 7-11\. 堆叠 LSTM 架构的准确率

这种过拟合（表现为训练准确率随着损失平滑下降而接近 100%，而验证准确率相对稳定且损失急剧增加）是模型对训练集过度专业化的结果。正如 第六章 中的例子所示，这表明如果你只看准确率指标而不检查损失，很容易陷入虚假的安全感。

![](img/aiml_0712.png)

###### 图 7-12\. 堆叠 LSTM 架构的损失

### 优化堆叠 LSTM

在 第六章 中，你看到了降低过拟合的一个非常有效的方法是降低学习率。在这里探索它是否对 RNN 也有积极的影响是值得的。

例如，以下代码将学习率 (LR) 降低 50%，从 0.0001 降低到 0.00005：

```py
# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), `lr``=``0.00005`, 
                       betas=(0.9, 0.999), amsgrad=False)

```

图 7-13 展示了这对其训练的影响。如图所示，验证准确率有细微差异，表明我们过拟合的程度有所减少。

![](img/aiml_0713.png)

###### 图 7-13\. 降低学习率对堆叠 LSTM 准确率的影响

虽然 图 7-14 的初步观察同样表明降低学习率对损失有良好的影响，曲线没有如此急剧上升，但值得更仔细地观察。我们看到训练集上的损失实际上略高于前一个例子（约 0.35 与约 0.27），而验证集上的损失较低（约 0.5 与 0.6）。

调整学习率超参数显然值得研究。

事实上，进一步实验表明，降低学习率 (LR) 可以显著提高训练和验证曲线的收敛性，这表明虽然网络在训练后准确性有所下降，但我们能看出它泛化得更好。图 7-15 和 7-16 展示了使用较低学习率 (.0003 而不是 .0005) 的影响。

![](img/aiml_0714.png)

###### 图 7-14\. 降低学习率对堆叠 LSTM 损失的影响

![](img/aiml_0715.png)

###### 图 7-15\. 堆叠 LSTM 的进一步降低学习率后的准确率

![](img/aiml_0716.png)

###### 图 7-16\. 进一步降低学习率 (LR) 和堆叠 LSTM 的损失

事实上，将 LR 进一步降低到 0.00001，可能给出了更好的结果，如图 7-17 和 7-18 所示。与之前的图表一样，虽然整体准确率不是很好，损失也更高，但这表明我们正在接近这个网络架构的“真实”结果，而不是被训练数据上的过拟合所误导，从而产生虚假的安全感。

除了改变 LR 参数外，你还应该考虑在 LSTM 层中使用 dropout。它的工作方式与密集层完全相同，如第三章第三章：超越基础，检测图像中的特征中所述，其中随机神经元被丢弃以防止邻近偏差影响学习。话虽如此，你应小心不要将其设置得太低，因为当你开始调整不同的架构时，你可能会冻结网络的学习能力。

**![](img/aiml_0717.png)

###### 图 7-17\. 较低 LR 的准确率

**![](img/aiml_0718.png)

###### 图 7-18\. 较低 LR 的损失

除了改变 LR 参数外，你还应该考虑在 LSTM 层中使用 dropout。它的工作方式与密集层完全相同，如第三章第三章：超越基础，检测图像中的特征中所述，其中随机神经元被丢弃以防止邻近偏差影响学习。

你可以通过使用`nn.Dropout`来实现 dropout。以下是一个示例：

```py
self.embedding_dropout = nn.Dropout(dropout_rate)
self.lstm_dropout = nn.Dropout(dropout_rate)
self.final_dropout = nn.Dropout(dropout_rate)
```

然后，在你的前向传递中，你可以应用适当的 dropout，如下所示：

```py
def forward(self, x):
    # Get embeddings
    embedded = self.embedding(x)  

    # Apply first dropout after embedding layer
    embedded = self.embedding_dropout(embedded)

    lstm1_out, _ = self.lstm1(embedded)

    # Apply dropout between LSTM layers
    lstm1_out = self.lstm_dropout(lstm1_out)

    lstm2_out, _ = self.lstm2(lstm1_out)

    # Apply final dropout
    lstm2_out = self.final_dropout(lstm2_out)

    lstm_out = lstm2_out.transpose(1, 2)

    pooled = self.global_pool(lstm_out)  
    pooled = pooled.squeeze(–1) 

    x = self.relu(self.fc1(pooled))
    x = self.sigmoid(self.fc2(x))

    return x
```

当我使用在 dropout 之前测试过的最低 LR 运行时，网络没有学习。因此，我将 LR 调整回 0.0003，并使用此 dropout 运行了 300 个 epoch（注意，dropout 率是 0.2，因此大约有 20%的神经元被随机丢弃）。准确率结果可以在图 7-19 中看到。训练和验证曲线仍然很接近，但它们达到了大于 75%的准确率，而没有 dropout 时，很难超过 70%。

**![](img/aiml_0719.png)

###### 图 7-19\. 使用 dropout 的堆叠 LSTMs 的准确率

如你所见，使用 dropout 可以对网络的准确率产生积极影响，这是好事！人们总是担心丢失神经元会使你的模型表现更差，但正如我们在这里所看到的，情况并非如此。但使用 dropout 时一定要小心，因为它如果不适当使用可能会导致欠拟合或过拟合。

如图 7-20 所示，这也有助于降低损失。虽然曲线明显发散，但它们比之前更接近，验证集在约 0.45 的损失处趋于平坦，这也证明了改进！正如这个例子所示，dropout 是另一种可以用来提高基于 LSTM 的 RNN 性能的实用技术。

值得探索这些避免数据过拟合的技术，也值得探索我们在第六章中提到的数据预处理技术。但还有一件事我们还没有尝试：一种迁移学习的形式，其中你可以使用预训练的词嵌入而不是尝试学习自己的。我们将在下一节中探讨这一点。

![图片](img/aiml_0720.png)

###### 图 7-20.启用 dropout 的 LSTMs 的损失曲线**  **# 使用预训练嵌入的 RNN

在所有之前的例子中，你收集了用于训练集的完整单词集，并使用它们训练嵌入。你最初将它们聚合起来，然后输入到密集网络中，在本章中，你探讨了如何使用 RNN 来改进结果。在这个过程中，你受到数据集中单词的限制以及如何使用该数据集的标签来学习它们的嵌入。

现在，回想一下第四章，我们讨论了迁移学习。如果你不是自己学习嵌入，而是可以使用预训练的嵌入，那么会怎样？研究人员已经完成了将单词转换为向量的艰苦工作，并且这些向量已经被证明是有效的。正如我们在第六章中看到的，一个例子是斯坦福大学的 Jeffrey Pennington、Richard Socher 和 Christopher Manning 开发的[GloVe (Global Vectors for Word Representation)模型](https://oreil.ly/4ENdQ)。

在这种情况下，研究人员已经分享了他们在各种数据集上的预训练词向量：

+   一个包含 60 亿个标记、40 万个单词的词汇表，在 50、100、200 和 300 个维度上，单词来自维基百科和 Gigaword。

+   来自公共爬取数据的 420 亿个标记、1900 万个单词的词汇表，在 300 个维度上。

+   来自公共爬取数据的 840 亿个标记、2200 万个单词的词汇表，在 300 个维度上。

+   来自 2 亿条推文的 Twitter 爬取数据中，一个包含 27 亿个标记、1200 万个单词的词汇表，在 25、50、100 和 200 个维度上。

由于这些向量已经预训练，你可以在 PyTorch 代码中简单地重用它们，而不是从头开始学习。首先，你将不得不下载 GloVe 数据。我选择使用 50 维的 60 亿单词版本，使用以下代码下载并解压它：

```py
import urllib.request
import zipfile

# Download GloVe embeddings
url = "https://nlp.stanford.edu/data/glove.6B.zip"
urllib.request.urlretrieve(url, "glove.6B.zip")

# Unzip
with zipfile.ZipFile("glove.6B.zip", 'r') as zip_ref:
    zip_ref.extractall()

# You can use glove.6B.50d.txt (50 dimensions)
# or glove.6B.100d.txt (100 dimensions)
```

文件中的每个条目都是一个单词，后面跟着为其学习到的维度系数。使用这种方法最简单的方式是创建一个字典，其中键是单词，值是嵌入。你可以这样设置这个字典：

```py
import numpy as np
glove_embeddings = dict()
f = open('glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs
f.close()
```

到目前为止，你将能够通过使用它作为键来查找任何单词的系数集。例如，要查看单词*frog*的嵌入，你可以使用这个：

```py
glove_embeddings['frog']
```

拥有这些预训练嵌入后，你现在可以将它们加载到你的神经网络架构中的嵌入层，并使用它们作为预训练嵌入而不是从头开始学习。请参阅以下模型架构定义。如果`pretrained_embeddings`的值不为空，则嵌入层的权重将从该值加载。如果`freeze_embeddings`为`True`，则它们将被冻结；否则，它们将用作学习的起点（即，你将根据你的语料库微调嵌入）：

```py
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=16, 
                 dropout_rate=0.25, pretrained_embeddings=None, 
                 freeze_embeddings=True, lstm_layers=2):
        super(TextClassificationModel, self).__init__()

        # Initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
```

此模型显示了总共 406.817 个参数，其中只有 6,817 个是可训练的，所以训练将会很快！

```py
==========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================
TextClassificationModel                  [32, 1]                   --
├─Embedding: 1-1                         [32, 60, 50]              (400,000)
├─Dropout: 1-2                           [32, 60, 50]              --
├─LSTM: 1-3                              [32, 60, 16]              6,528
├─Dropout: 1-4                           [32, 60, 16]              --
├─AdaptiveAvgPool1d: 1-5                 [32, 16, 1]               --
├─Linear: 1-6                            [32, 16]                  272
├─ReLU: 1-7                              [32, 16]                  --
├─Dropout: 1-8                           [32, 16]                  --
├─Linear: 1-9                            [32, 1]                   17
├─Sigmoid: 1-10                          [32, 1]                   --
==========================================================================
Total params: 406,817
Trainable params: 6,817
Non-trainable params: 400,000
Total mult-adds (M): 25.34
==========================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 1.02
Params size (MB): 1.63
Estimated Total Size (MB): 2.66
==========================================================================

```

你现在可以像以前一样进行训练，并可以看到这个架构，即使用预训练嵌入和堆叠的 LSTMs，如何真正地减少过拟合！图 7-21 显示了使用 LSTMs 和预训练 GloVe 嵌入在讽刺语料库上的训练与验证准确率，而图 7-22 显示了训练与验证的损失，曲线的接近程度表明我们没有过拟合。

![](img/aiml_0721.png)

###### 图 7-21。使用 LSTMs 和 GloVe 在讽刺语料库上的训练与验证准确率

![](img/aiml_0722.png)

###### 图 7-22。使用 LSTMs 和 GloVe 在讽刺语料库上的训练与验证损失

为了进一步分析，你需要考虑你的词汇量大小。在前一章中，你为了避免过拟合所做的优化之一是为了防止嵌入层被低频词的学习所负担：你通过使用常用词的较小词汇量来避免过拟合。在这种情况下，由于 GloVe 已经为你学习了单词嵌入，你可以扩展词汇量——但扩展多少呢？

首先要探索的是你的语料库中有多少单词实际上在 GloVe 集中。它有 120 万个单词，但无法保证它有*所有*你的单词。

当构建`word_index`时，你可以使用一个非常大的数字调用`build_vocab_glove`，并且它将忽略超过总量的任何单词。例如，假设你调用这个：

```py
word_index = build_vocab_glove(training_sentences, max_vocab_size=100,000)
```

使用讽刺语料库，你将获得一个词汇量大小为 22,457 的结果。如果你愿意，可以探索 GloVe 嵌入以查看其中有多少单词存在于 GloVE 中。首先，为嵌入创建一个字典并将 GloVE 文件读取到其中：

```py
embeddings_dict = {}
embedding_dim = 50
glove_file = f'glove.6B.{embedding_dim}d.txt'

# Read GloVe embeddings
print(f"Loading GloVe embeddings from {glove_file}...")
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = vector
```

然后，你可以将这个与你从整个语料库中创建的 `word_index` 进行比较：

```py
found_words = 0
for word, idx in word_index.items():
    if word in embeddings_dict:
        found_words += 1
print(found_words)
```

在讽刺的情况下，有 21,291 个单词被发现在 GloVE 中，这几乎是全部，所以你在第六章中使用的原则来选择你应该训练多少（即，选择那些频率足够高以产生信号的单词）仍然适用！

使用这种方法，我选择使用 8,000 个词汇量（而不是之前为了避免过拟合而使用的 2,000 个），以获得你刚才看到的成果。然后我用 *The Onion* 的标题（讽刺数据集中讽刺标题的来源）和其他句子进行了测试，如下所示：

```py
test_sentences = ["It Was, For, Uh, Medical Reasons, Says Doctor To 
                   Boris Johnson, Explaining Why They Had To Give Him Haircut",
                  "It's a beautiful sunny day",
                  "I lived in Ireland, so in high school they made me 
                   learn to speak and write in Gaelic",
                  "Census Foot Soldiers Swarm Neighborhoods, Kick Down 
                   Doors To Tally Household Sizes"]

```

这些标题的结果如下——记住，接近 50%（0.5）的值被认为是中性的，接近 0 的被认为是非讽刺的，而接近 1 的被认为是讽刺的：

```py
tensor([[0.9316],
        [0.1603],
        [0.6959],
        [0.9594]], device='cuda:0')

Text: It Was, For, Uh, Medical Reasons, Says Doctor To Boris Johnson, 
      Explaining Why They Had To Give Him Haircut
Probability: 0.9316
Classification: Sarcastic
--------------------------------------------------------------------------

Text: It's a beautiful sunny day
Probability: 0.1603
Classification: Not Sarcastic
--------------------------------------------------------------------------

Text: I lived in Ireland, so in high school they made me learn to speak 
      and write in Gaelic
Probability: 0.6959
Classification: Sarcastic
--------------------------------------------------------------------------

Text: Census Foot Soldiers Swarm Neighborhoods, Kick Down Doors To Tally 
      Household Sizes
Probability: 0.9594
Classification: Sarcastic
--------------------------------------------------------------------------
```

前四句，其中一句来自 *The Onion*，显示有 93% 以上的可能性是讽刺。关于天气的陈述被强烈地认为是非讽刺的（16%），而关于在爱尔兰上高中的句子被认为可能是讽刺的，但信心不高（69%）。

# 摘要

本章向你介绍了循环神经网络，它们在设计上使用面向序列的逻辑，可以帮助你根据句子中包含的单词以及它们的顺序来理解句子的情感。你看到了一个基本的 RNN 是如何工作的，以及 LSTM 如何在此基础上构建以使长期保持上下文。这些模型是流行且著名的“transformers”模型的先驱，这些模型被用来支撑生成式 AI。

你还使用了 LSTM 来改进你一直在工作的情感分析模型，然后你研究了 RNN 的过拟合问题以及改进它们的技术，包括使用预训练嵌入的迁移学习。

在第八章中，你将使用你迄今为止所学的内容来探索如何预测单词，从那里，你将能够创建一个为你创建文本和写诗的模型！**
