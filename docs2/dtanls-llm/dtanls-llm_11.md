# 9 优化成本和质量

### 本章涵盖内容

+   模型选择和调整

+   提示工程

+   模型微调

使用大型语言模型分析数据是快速烧钱的好方法。如果你已经使用 GPT-4（或类似的大型模型）了一段时间，你可能已经注意到费用是如何快速累积的，迫使你定期充值账户。但我们是否总是需要使用最大（也是最昂贵）的模型？我们能否让小型模型几乎以同样的效果运行？我们如何花最少的钱获得最大的效益？

本章主要讲述在使用大型数据集上的语言模型时如何节省金钱。幸运的是，我们有很多选择来实现这一点。首先，在大型语言模型方面，我们有相当多的选择。选择一个尽可能小（或者说，更便宜）的模型，同时仍然能够很好地完成我们的分析任务，这可以在很大程度上帮助我们平衡预算。其次，模型通常具有各种调整参数，使我们能够从整体文本生成策略调整到特定标记的（去）优先级调整。我们希望在那里优化我们的设置，将小型模型转变为特定任务的 GPT-4 替代品。第三，我们可以使用提示工程来调整我们向模型提问的方式，有时会得到令人惊讶的不同结果！

最后，如果上述方法都不奏效，我们可以选择创建自己的模型，这些模型高度定制化，仅针对我们关心的任务。当然，如果我们不想在预训练上花费数百万，我们就不会从头开始训练新模型。相反，我们通常会选择使用仅几百个样本对现有模型进行微调。这通常足以比使用基础模型获得显著更好的性能。

当然，最佳方案取决于我们试图解决的问题以及数据属性。幸运的是，如果我们想分析大量数据，我们可以在数据样本上尝试不同的调整选项，花费一点钱。很可能，一旦我们分析了整个数据集，这种前期投资就会得到回报！在本章中，我们将在一个示例场景中应用所有这些调整选项。

## 9.1 示例场景

你回到了 Banana，试图对用户评论进行分类。用户可以在 Banana 网站上留下关于 Banana 产品体验的自由文本评论。你想要知道这些评论是积极的（即，用户对产品感到满意）还是消极的（即，阅读它们会吓跑潜在客户！）。当然，你可以使用语言模型来完成这项任务（你可以在第四章中看到）。例如，你可以使用 GPT-4（在撰写本文时，这是 OpenAI 用于文本处理的最大模型）。向 GPT-4 提供一个评论，以及如何对其进行分类的说明（包括可能的标签描述，如“积极”和“消极”），输出应该对大多数评论都是正确的。

然而，使用 GPT-4 分析数据每千个标记的成本约为 6 美分。这（6 美分）可能听起来不多，但 Banana 每天都会收到数千条产品评论！让我们假设平均评论包含大约 100 个标记（大约 400 个字符）。此外，让我们假设 Banana 每天收到大约 10,000 条评论。这意味着你每天收集 100 × 10,000 个标记：大约每天 1 百万个标记，每年 3.65 亿个标记。分析一年的评论需要多少钱？大约是 365,000,000 × (0.06/1000) = 21,900 美元。

这可能会在你的预算上留下一些缺口！难道你不能以更低的价格获得它吗？例如，在撰写本文时，GPT-3.5 Turbo 的价格仅为每千个标记大约 0.0005 美元（标记的价格因是否读取或生成而有所不同，但为了简化计算，我们暂时忽略这一点）。这意味着分析一年的评论只需 365,000,000 × (0.0005/1000) = 182.5 美元。这要好得多！但为了获得令人满意的输出质量，你可能需要做一些额外的工作，以确保你以最佳方式使用该模型。

小贴士：除了 GPT-3.5 Turbo，你还可以在以下示例中使用其他模型，如 GPT-4o mini（模型 ID 为`gpt-4o-mini`）。

这就是我们将在本例中做的事情。从我们分类器的最简单实现开始，我们将逐步改进我们的实现，并尝试本章介绍中讨论的所有各种调优选项！

## 9.2 未调优分类器

让我们从我们分类器的基版开始。再次强调，目标是取一个评论并决定它应该被分类为正面（`pos`）还是负面（`neg`）。我们将使用以下提示模板来分类评论：

```py
[Review]
Is the sentiment positive or negative?
Answer ("pos"/"neg"):    
```

在这个提示模板中，`[Review]`是一个占位符，将被实际评论文本替换。例如，替换后，我们的提示可能看起来像这样（前两行对应于评论的缩略版，显然是 Banana TV 上的一部新电影流媒体，不符合评论者的口味）：

```py
I am willing to tolerate almost anything in a Sci-Fi movie, 
but this was almost intolerable. ...
Is the sentiment positive or negative?
Answer ("pos"/"neg"):    
```

理想情况下，如果我们向 GPT 模型发送这个提示，我们期望得到 `pos` 或 `neg` 作为回复（在这个特定情况下，我们期望 `neg`）。列表 9.1 展示了完整的 Python 代码；我们不会花太多时间讨论它，因为它与我们第四章中看到的分类器类似。`create_prompt` 函数（**1**）为特定评论实例化提示模板（存储在输入参数 `text` 中）。结果是我们可以使用 `call_llm` 函数（**2**）发送给我们的语言模型的提示。在这里我们调用 GPT-3.5 Turbo（**3**）（节省成本）。我们还设置 `temperature` 为 `0`，这意味着我们在生成输出时最小化随机性。这意味着当你重复运行代码时，你应该看到相同的结果。你也许还会注意到，列表 9.1 中的 `call_llm` 比我们之前看到的版本要长一些。那是因为我们不仅检索了我们的语言模型生成的答案，还检索了使用的标记数量（**4**）。计算标记数量将允许我们计算数据样本上的调用成本。

##### 列表 9.1 将评论分类为正面或负面：基础版本

```py
import argparse
import openai
import pandas as pd
import time

client = openai.OpenAI()

def create_prompt(text):                         #1
    """ Create prompt for sentiment classification.

    Args:
        text: text to classify.

    Returns:
        Prompt for text classification.
    """
    task = 'Is the sentiment positive or negative?'
    answer_format = 'Answer ("pos"/"neg")'
    return f'{text}\n{task}\n{answer_format}:'

def call_llm(prompt):                              #2
    """ Query large language model and return answer.

    Args:
        prompt: input prompt for language model.

    Returns:
        Answer by language model and total number of tokens.
    """
    for nr_retries in range(1, 4):
        try:
             #3
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role':'user', 'content':prompt}
                    ],
                temperature=0
                )
             #4
            answer = response.choices[0].message.content
            nr_tokens = response.usage.total_tokens
            return answer, nr_tokens

        except Exception as e:
            print(f'Exception: {e}')
            time.sleep(nr_retries * 2)

    raise Exception('Cannot query OpenAI model!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()                   #5
    parser.add_argument('file_path', type=str, help='Path to input file')
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)

    nr_correct = 0
    nr_tokens = 0

    for _, row in df.iterrows():  #6

        text = row['text']       #7
        prompt = create_prompt(text)
        label, current_tokens = call_llm(prompt)

        ground_truth = row['sentiment']      #8
        if label == ground_truth:
            nr_correct += 1
        nr_tokens += current_tokens

        print(f'Label: {label}; Ground truth: {ground_truth}')

    print(f'Number of correct labels:\t{nr_correct}')
    print(f'Number of tokens used   :\t{nr_tokens}')
```

#1 生成提示

#2 调用语言模型

#3 生成答案

#4 提取答案和标记使用情况

#5 解析参数

#6 遍历评论

#7 分类评论

#8 更新计数器

我们将假设要分类的评论存储在一个 .csv 文件中。我们期望用户指定该 .csv 文件的路径作为命令行参数（**5**）。在读取 .csv 文件后，我们按照它们在输入文件中出现的顺序遍历评论（**6**）。对于每条评论，我们提取相关的文本（**7**）（我们假设它存储在 `text` 列），创建一个用于分类的提示，并调用语言模型。结果是语言模型生成的答案文本（希望它是两个类别标签之一，`pos` 或 `neg`），以及使用的标记数量。

我们的目标是尝试不同的查询语言模型的方法，并比较输出质量和成本。为了判断输出质量，我们假设输入 .csv 文件不仅包含评论文本，还包含一个真实标签。这意味着我们假设每条评论都已经与正确的类别标签相关联，存储在 `sentiment` 列（因为我们的两个类别标签描述了评论的情感）。在收到语言模型的输出后，我们将输出与真实标签（**8**）进行比较，并更新正确分类的评论数量（变量 `nr_correct`）。同时，我们汇总使用的总标记数量（因为处理费用与它成正比）并将它们存储在名为 `nr_tokens` 的计数器中。遍历所有评论后，列表 9.1 打印出最终的分类正确数量和使用的标记数量。

## 9.3 模型调整

让我们试试！你可以在书籍网站上找到“未调优分类器”下的 9.1 列表。我们重用了第四章中的电影评论；在第四章部分搜索“Reviews.csv”链接。该文件包含 10 篇评论，以及相应的 ground truth。假设 9.1 列表和评论存储在磁盘上的同一文件夹中。打开你的终端，切换到该文件夹，并运行以下命令：

```py
python basic_classifier.py reviews.csv
```

你应该看到以下输出：

```py
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: pos; Ground truth: pos
Label: pos; Ground truth: neg  #1
Label: pos; Ground truth: neg
Label: negative; Ground truth: neg  #2
Label: negative; Ground truth: pos
Label: neg; Ground truth: neg
Number of correct labels:    6
Number of tokens used   :    2228
```

#1 错误的标签

#2 不存在的标签

前十行描述了每个评论的结果。我们有了语言模型生成的标签和从输入文件中获取的 ground-truth 标签。最后，我们有正确分类的评论数量和使用的 token 数量。

在 10 篇评论中，我们正确分类了 6 篇。嗯，至少比 50%好，但这仍然不是一个很好的结果。出了什么问题？查看输出给我们一些想法。有些情况（**1**）中，语言模型简单地选择了错误的类别标签。这并不意外。然而，也有情况（**2**）中，语言模型选择了一个甚至不存在的类别标签！当然，这并不太离谱（`negative`而不是`neg`），这似乎很容易修复。

我们专注于语言模型只生成我们两个可能的类别标签之一这个可能容易解决的问题。我们如何做到这一点？输入`logit_bias`参数。`logit_bias`参数允许用户更改某些 token 被选中的可能性（我们在第三章中简要讨论了这一点和其他 GPT 参数）。在这种情况下，我们希望显著增加与我们的两个类别标签（`neg`和`pos`）相关的 token 的概率。`logit_bias`参数指定为一个 Python 字典，将 token ID 映射到偏差。正偏差意味着我们希望增加语言模型生成相应 token 的概率。负偏差意味着我们降低生成相关 token 的概率。

在这个例子中，我们希望增加 GPT-3.5 选择代表类别标签的两个 token 之一的机会。因此，我们希望为这两个 token ID 选择一个高偏差。偏差分数范围从-100 到+100。我们将选择最大值，并将代表类别标签的 token 分配偏差+100。首先，我们需要找到它们的 token ID。语言模型将文本表示为 token ID 的序列。要更改 token 偏差，我们需要参考我们关心的 token 的 ID。

*tokenizer*是将文本转换为 token ID 的组件。你可以在[`platform.openai.com/tokenizer`](https://platform.openai.com/tokenizer)找到所有 GPT 模型的 tokenizer。我们使用 GPT-3.5，所以选择标记为 GPT 3.5 & GPT-4 的那个。图 9.1 显示了 tokenizer 的 Web 界面。

我们可以在文本框中输入文本并点击标记 ID 按钮来查看输入文本的标记 ID。使用标记化器，我们了解到`pos`标记的 ID 为 981，而`neg`标记的 ID 为 29875。现在我们准备好在模型调用中添加偏差，如下所示：

```py
import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role':'user', 'content':prompt}
        ],
    logit_bias = {981:100, 29875:100},  #1
    temperature=0
    )
```

#1 定义了偏差

![figure](img/CH09_F01_Trummer.png)

##### 图 9.1 GPT 标记化器在[`platform.openai.com/tokenizer`](https://platform.openai.com/tokenizer)：输入文本以学习相关的标记 ID。

与之前的调用（在列表 9.1 中）相比，我们通过将我们感兴趣的两种标记的 ID（`pos`标记 ID 为 981 和`neg`标记 ID 为 29875）映射到最高的偏差值 100 来添加 logit 偏差（**1**）。这应该可以解决生成与类标签不对应的标记的问题，对吧？

警告 下述代码会导致问题，并导致长时间运行和显著的货币费用。在没有整合本节末尾提出的修复方案之前，不要尝试它！

让我们试一试以确保。您可以将 logit 偏差添加到列表 9.1 中的代码。或者，在本章的后面，我们将介绍一个可调整的分类器版本，它将允许您尝试不同的调整参数组合（包括 logit 偏差）。如果您使用添加了偏差的分类器执行，您可能会看到以下类似的输出（实际上，由于执行代码需要很长时间并且产生不可忽视的成本，您可能只想相信我）： 

```py
 #1
Label: negnegnegnegnegnegnegnegnegneg ...; Ground truth: neg
Label: negposnegnegnegnegnegnegnegneg ...; Ground truth: neg
Label: negposnegnegnegnegnegnegnegneg ...; Ground truth: neg
Label: negposnegposnegnegnegnegnegneg ...; Ground truth: neg
Label: posnegposnegposnegposnegposneg ...; Ground truth: pos
Label: posnegpospospospospospospospos ...; Ground truth: neg
Label: posnegpospospospospospospospos ...; Ground truth: neg
Label: negposnegposnegposnegposnegpos ...; Ground truth: neg
Label: negposnegposnegposnegposnegpos ...; Ground truth: pos
Label: negposnegposnegnegnegnegnegneg ...; Ground truth: neg
Number of correct labels:    0
Number of tokens used   :    2318  #2
```

#1 每个输入都没有标签：

#2 增加标记使用

哦，不——没有一次正确的分类！发生了什么？将生成的“标签”与真实值进行比较揭示了问题（**1**）：我们只生成了两种可能的标记（这很好！）但太多了（这并不好！）。这增加了标记消耗（**2**）（注意，输出长度被限制以生成示例输出；否则，标记消耗会更高），但更重要的是，这意味着我们的输出与任何类标签都不对应。

##### 为什么模型会生成这么多标记？

我们本质上将模型限制为仅使用两个标记来生成文本。这些是我们希望在输出中看到的两个标记。然而，我们忘记启用模型生成任何表示输出结束的标记！这就是为什么模型无法停止生成的原因！

解决这个问题有多种方法。当然，我们可以添加后处理步骤来从语言模型生成的输出中仅提取第一个标记。这样（大多数情况下）就能解决我们关于类别标签的问题。看看输出，你会发现使用第一个标记在 10 个案例中有 7 个是正确的。然而，这种方法（还有）存在另一个问题：我们正在为最终不使用的标记付费！这显然不是我们想要的。所以让我们通过限制输出长度来进一步调整我们的模型。我们只需要一个标记（这之所以可行，是因为我们的两个可能的类别标签可以用一个标记来表示）。这就是`max_tokens`参数的作用。让我们在调用我们的语言模型时使用它：

```py
response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role':'user', 'content':prompt}
        ],
    logit_bias = {981:100, 29875:100},  #1
    temperature=0, max_tokens=1
    )
```

#1 定义偏差

当你尝试它（这应该既快又便宜）时，你应该看到以下输出：

```py
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: pos; Ground truth: pos
Label: pos; Ground truth: neg
Label: pos; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: pos
Label: neg; Ground truth: neg
Number of correct labels:    7  #1
Number of tokens used   :    2228  #2
```

#1 改进了未调整的分类器

#2 减少标记使用

好多了！我们将正确处理的案例数量从未调整版本中的六个提高到了七个（**1**）。这可能听起来不多。然而，从整个数据集的角度来看，这实际上意味着我们将精度从 60%提高到了 70%：也就是说，现在将有数千条评论将被正确分类！当然，这里有一个警告。在现实中，你可能需要使用一个更大的样本。由于随机变化，你在样本上观察到的准确性可能不代表整个数据集的准确性。为了简化问题（并且在你尝试时成本相对较低），我们在这里限制自己只使用 10 个样本。作为额外的奖励，我们的标记消耗再次减少了（**2**）（实际上，与没有任何输出大小限制的版本相比，标记消耗的差距可能要大得多）。请注意，这里讨论的两个参数只是可用调整选项的一小部分。你将在第三章中找到更多关于相关参数的详细信息。每次你为新的任务调整模型时，务必考虑所有可能相关的参数。然后在数据样本上尝试一些合理的设置，看看哪个选项表现最好。

## 9.4 模型选择

假设我们已经通过调整当前模型达到了性能提升的极限。我们还能做什么呢？当然，我们可以选择一个不同的模型。在上一章中，我们看到了一些 GPT 的替代品。如果你可以选择一个专门针对你感兴趣的任务（例如，文本分类）进行训练的模型，那么这通常值得一看。其他可能影响你模型选择的因素包括你打算应用模型的数据是否敏感，以及是否可以将这些数据发送给特定的语言模型提供商是否可接受。

如果你想了解不同模型的相对性能，请查看 [`crfm.stanford.edu/helm/lite/latest/`](https://crfm.stanford.edu/helm/lite/latest/)。这个网站包含了 HELM（斯坦福大学对语言模型的整体评估）的结果。该基准比较了在各种场景下的语言模型，并包含了特定任务的结果，以及在不同场景下的平均性能。你可能想查看一下，以了解哪些模型可能对你感兴趣。然而，由于各种因素都可能影响语言模型的表现，仍然值得在你感兴趣的特定任务上评估不同的模型。

为了简化问题，我们只考虑将 GPT-4 作为 GPT-3.5 Turbo（我们之前使用的模型）的替代品。在语言模型调用中替换模型的名称：

```py
response = client.chat.completions.create(
    model='gpt-4',
    messages=[
        {'role':'user', 'content':prompt}
        ],
    logit_bias = {981:100, 29875:100},  #1
    temperature=0, max_tokens=1
    )
```

#1 定义了偏差

运行生成的代码应该产生以下输出：

```py
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: pos; Ground truth: pos
Label: pos; Ground truth: neg
Label: pos; Ground truth: neg
Label: neg; Ground truth: neg
Label: pos; Ground truth: pos  #1
Label: neg; Ground truth: neg
Number of correct labels:    8  #2
Number of tokens used   :    2228  #3
```

#1 正确的分类结果

#2 目前最佳结果

#3 相同数量的标记

与先前的版本相比，GPT-4 成功地准确解决了另一个测试案例（**1**）！这使我们的准确率达到了 80% （**2**），而我们的标记消耗保持不变（**3**）。顺便说一下，如果我们更改模型，这并不保证会发生。因为不同的模型可能使用不同的标记器，表示相同的文本可能需要不同数量的标记。在这个特定案例中，因为 GPT-4 和 GPT-3.5 使用相同的标记器，所以标记的数量没有变化。

这是否意味着我们支付了相同数量的金钱？并不完全是这样。因为 GPT-4 每个标记的收费要高得多，我们现在的支出大约是之前的 120 倍（GPT-4 和 GPT-3.5 Turbo 每个标记价格的相对差异）。这就是为什么我们试图让 GPT-3.5 尽可能地表现良好，而不依赖于 GPT-4。

在模型选择和模型调整过程中，有时亲自查看测试数据是有意义的。这让你对各种模型的优点和局限性有更好的印象，并使你能够判断你的模型表现不佳的测试案例是否具有代表性。例如，以下评论被 GPT-4 正确解决，但 GPT-3.5 则没有：

```py
If you want to see a film starring Stan Laurel from the Laurel & Hardy 
comedies, this is not the film for you. Stan would not begin to find the 
character and rhythms of those films for another two years. If, however, 
you want a good travesty of the Rudolph Valentino BLOOD AND SAND, which 
had been made the previous year, this is the movie for you. All the 
stops are pulled out, both in physical comedy and on the title cards 
and if the movie is not held together by character, the plot of 
Valentino's movie is used -- well sort of.
```

这篇评论包含了积极（接近结尾）以及消极（开头）的方面。尽管最终结论是积极的，我们可能得出结论，为了正确分析像这篇评论这样的边缘案例而花费更多的钱是不值得的。

## 9.5 提示工程

除了交换模型的选择，我们还能做些什么来提高我们模型的性能？我们还没有探讨的一个领域是我们用于分类的提示定义。改变提示模板可以对结果质量产生重大影响。提示调整通常至关重要的这一事实甚至导致了专门术语“提示工程”的引入，描述了搜索最佳提示模板的过程。更重要的是，提示工程带来的挑战导致了多个平台的创建，为各种不同的任务提供提示模板。如果你对提示变体没有想法，可以看看[`promptbase.com/`](https://promptbase.com/)，[`prompthero.com/`](https://prompthero.com/)和类似平台。这些平台的商业模式是使用户能够购买和销售针对特定任务优化特定模型性能的提示模板。

确定哪个提示效果最佳通常需要一些实验。接下来，我们将关注基础知识，并探讨通过改变提示来提高输出质量的一种经典技术。我们在这里讨论的是少样本学习，这意味着我们通过给它一些示例来帮助模型。这是我们日常生活中知道的事情：仅基于纯描述很难理解一个新任务或方法。看到一些示例来掌握它要好得多。例如，在前面的章节中，我们只需讨论几个相关的模型调整参数的语义。但不是更好吗？在具体示例场景中看到它们是如何调整的？

当然是。语言模型“感觉”到同样的方式，添加一些有用的示例通常可以提高它们的性能。那么我们如何向它们展示示例呢？很简单：我们将这些示例指定为提示的一部分。例如，在我们的分类场景中，我们希望语言模型能够对评论进行分类。一个例子就是一个评论和参考类别标签。

我们将使用以下提示模板将单个样本集成到提示中：

```py
[Sample Review]
Is the sentiment positive or negative?
Answer ("pos"/"neg"):[Sample Solution]
[Review to Classify]
Is the sentiment positive or negative?
Answer ("pos"/"neg"):
```

如果我们用样例评论、样例评论解决方案和我们要分类的评论替换占位符，我们得到的提示如下：

```py
Now, I won't deny that when I purchased #1
this off eBay, I had high expectations. ...
Is the sentiment positive or negative?  #2
Answer ("pos"/"neg"):neg          #3
I am willing to tolerate almost anything            #4
in a Sci-Fi movie, but this was almost intolerable. ...
Is the sentiment positive or negative?              #5
Answer ("pos"/"neg"): 
```

#1 样例评论

#2 指令

#3 样例解决方案

#4 待分类评论

#5 指令

你可以看到一个样例评论（**1**），指令（**2**），以及样例评论的参考类别（**3**）。之后，你找到我们想要分类的评论（**4**）和分类指令（再次）(**5**)，但还没有解决方案（当然不是——这正是我们希望语言模型生成的）。在这个提示中，我们向模型提供了一个正确解决任务的示例。这样做可能有助于模型更好地理解我们要求它做什么。

当然，在提示中提供样本有许多选项。我们选择了最简单直接的方法：我们为两个评论使用相同的提示结构两次。因为我们使用的是完全相同的结构，所以我们的提示稍微有些冗余：我们重复了任务指令（**2** 和 **5**），包括两个可能的类别标签的指定。虽然我们在这里不会这样做，但尝试以不同的方式将示例整合到提示中，去除冗余并缩短提示长度（从而减少处理的令牌数量，最终减少处理费用）可能很有趣。

到目前为止，我们只考虑了添加单个示例。但有时候，看到一个示例可能还不够。这就是为什么为语言模型添加多个示例也可能是有意义的。让我们假设我们有一些样本：带有相关类别标签的评论，存储在一个名为 `samples` 的数据框中。我们可以使用以下代码生成整合这些样本的提示：

```py
def create_single_text_prompt(text, label):   #1
    """ Create prompt for classifying a single text.

    Args:
        text: text to classify.
        label: correct class label (empty if unavailable).

    Returns:
        Prompt for text classification.
    """
    task = 'Is the sentiment positive or negative?'
    answer_format = 'Answer ("pos"/"neg")'
    return f'{text}\n{task}\n{answer_format}:{label}'

def create_prompt(text, samples):            #2
    """ Generates prompt for sentiment classification.

    Args:
        text: classify this text.
        samples: integrate these samples into prompt.

    Returns:
        Input for LLM.
    """
    parts = []
    for _, row in samples.iterrows():  #3
        sample_text = row['text']
        sample_label = row['sentiment']
        prompt = create_single_text_prompt(sample_text, sample_label)
        parts += [prompt]

    prompt = create_single_text_prompt(text, ")  #4
    parts += [prompt]
    return '\n'.join(parts)
```

#1 为一条评论创建提示

#2 为所有评论生成提示

#3 集成样本

#4 添加评论以进行分类

`create_single_text_prompt` 函数（**1**）实例化以下模板，用于单个评论：

```py
[Review]
Is the sentiment positive or negative?
Answer ("pos"/"neg"):[Label]
```

我们使用相同的函数来指定样本评论，以及指定我们希望语言模型为我们解决的问题的分类任务。如果我们指定样本评论，`[Label]` 占位符将被替换为对应评论的参考类别标签。如果我们指定语言模型应解决的问题，我们还没有正确的类别标签。在这种情况下，我们将 `[Label]` 占位符替换为空字符串。这将由语言模型来完成，以实际类别标签完成提示。

`create_prompt` 函数（**2**）生成完整的提示，考虑所有样本评论以及我们想要分类的评论。首先（**3**），它遍历样本评论。我们假设我们的 `samples` 数据框在 `text` 列中存储评论文本，在 `sentiment` 列中存储相关的类别标签。我们使用之前讨论过的 `create_single_text_prompt` 函数（**4**）为样本评论添加一个提示部分。最后，我们添加指令来分类我们感兴趣的评论。

让我们切换回使用 GPT-3.5 Turbo。然而，这次我们将使用我们新的提示生成函数。目前，我们将限制自己在提示中使用单个示例评论。在书的配套网站上，您可以在“评论训练”部分找到带有正确类别标签的训练评论，链接到文件 train_reviews.csv。这个文件中的评论与 reviews.csv 文件中的评论（我们用来测试我们的方法）不重叠。将 train_reviews.csv 中的第一条评论作为样本添加到提示中，现在您应该看到以下输出：

```py
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: pos; Ground truth: pos
Label: pos; Ground truth: neg
Label: pos; Ground truth: neg
Label: neg; Ground truth: neg
Label: pos; Ground truth: pos
Label: neg; Ground truth: neg
Number of correct labels:    8  #1
Number of tokens used   :    4078  #2
```

#1 等同于 GPT-4 的结果

#2 标记使用量大约翻倍

欢呼！我们已经将精度提高到 80%（**1**）。这与我们在原始提示（没有样本评论）上使用 GPT-4 所获得的精度相同。同时，我们的标记使用量也有所增加（**2**）。更准确地说，因为我们为每个提示添加了第二个评论（即，我们有一个样本评论和一个要分类的评论），与上一个版本相比，我们的标记消耗量大约翻倍。然而，与在较短的提示上使用 GPT-4 相比，我们当前的方法仍然便宜约 60 倍（因为使用 GPT-4 大约比使用 GPT-3.5 Turbo 贵 120 倍）。

## 9.6 可调分类器

现在我们已经看到了很多调整选项，你可能想尝试新的变体。例如，如果我们添加了样本，是否还需要添加偏差（本质上限制输出为两个可能的类别标签）？当我们使用更大的模型并在提示中添加多个样本时，能否获得更高的精度？将你的代码更改为尝试新的组合很快就会变得繁琐。但不用担心，我们已经为你准备好了！在本书的网站上，你可以在“可调分类器”部分找到列表 9.2。这个实现允许你通过设置正确的命令行参数来尝试所有调整变体。我们将快速讨论代码，该代码整合了之前讨论的所有代码变体。

生成提示（**1**）的工作方式如上一节所述。`create_prompt` 函数接受要分类的评论文本和样本评论作为输入。样本评论被添加到提示中，可能支持语言模型对感兴趣评论的分类。请注意，我们仍然可以在不指定任何样本的情况下看到语言模型的表现（通过不指定任何样本）。没有样本的分类对应于一个特殊情况。

##### 列表 9.2 情感分类器的可调版本

```py
import argparse
import openai
import pandas as pd
import time

client = openai.OpenAI()

def create_single_text_prompt(text, label):
    """ Create prompt for classifying a single text.

    Args:
        text: text to classify.
        label: correct class label (empty if unavailable).

    Returns:
        Prompt for text classification.
    """
    task = 'Is the sentiment positive or negative?'
    answer_format = 'Answer ("pos"/"neg")'
    return f'{text}\n{task}\n{answer_format}:{label}'

def create_prompt(text, samples):             #1
    """ Generates prompt for sentiment classification.

    Args:
        text: classify this text.
        samples: integrate these samples into prompt.

    Returns:
        Input for LLM.
    """
    parts = []
    for _, row in samples.iterrows():
        sample_text = row['text']
        sample_label = row['sentiment']
        prompt = create_single_text_prompt(sample_text, sample_label)
        parts += [prompt]

    prompt = create_single_text_prompt(text, ")
    parts += [prompt]
    return '\n'.join(parts)
 #2
def call_llm(prompt, model, max_tokens, out_tokens):
    """ Query large language model and return answer.

    Args:
        prompt: input prompt for language model.
        model: name of OpenAI model to choose.
        max_tokens: maximum output length in tokens.
        out_tokens: prioritize these token IDs in output.

    Returns:
        Answer by language model and total number of tokens.
    """
    optional_parameters = {}
    if max_tokens:
        optional_parameters['max_tokens'] = max_tokens
    if out_tokens:
        logit_bias = {int(tid):100 for tid in out_tokens.split(',')}
        optional_parameters['logit_bias'] = logit_bias

    for nr_retries in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role':'user', 'content':prompt}
                    ],
                **optional_parameters, temperature=0
                )

            answer = response.choices[0].message.content
            nr_tokens = response.usage.total_tokens
            return answer, nr_tokens

        except Exception as e:
            print(f'Exception: {e}')
            time.sleep(nr_retries * 2)

    raise Exception('Cannot query OpenAI model!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()       #3
    parser.add_argument('file_path', type=str, help='Path to input file')
    parser.add_argument('model', type=str, help='Name of OpenAI model')
    parser.add_argument('max_tokens', type=int, help='Maximum output size')
    parser.add_argument('out_tokens', type=str, help='Tokens to prioritize')
    parser.add_argument('nr_samples', type=int, help='Number of samples')
    parser.add_argument('sample_path', type=str, help='Path to samples')
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)

    samples = pd.DataFrame()  #4
    if args.nr_samples:
        samples = pd.read_csv(args.sample_path)
        samples = samples[:args.nr_samples]

    nr_correct = 0
    nr_tokens = 0

    for _, row in df.iterrows():

        text = row['text']   #5
        prompt = create_prompt(text, samples)
        label, current_tokens = call_llm(
            prompt, args.model, 
            args.max_tokens, 
            args.out_tokens)

        ground_truth = row['sentiment']  #6
        if label == ground_truth:
            nr_correct += 1
        nr_tokens += current_tokens

        print(f'Label: {label}; Ground truth: {ground_truth}')

         #7
    print(f'Number of correct labels:\t{nr_correct}')
    print(f'Number of tokens used   :\t{nr_tokens}')
```

#1 使用样本生成提示

#2 使用参数调用语言模型

#3 解析命令行参数

#4 从磁盘读取样本

#5 分类评论

#6 更新计数器

#7 打印计数器

我们的 `call_llm` 函数（**2**）整合了之前提到的所有调整参数。首先是调用模型的名称（`model` 参数）。其次，我们可以指定最大输出标记数（`max_tokens`）。最后，我们可以指定偏差：在生成输出时应优先考虑的标记。`out_tokens` 参数允许用户指定一个以逗号分隔的标记 ID 列表，我们将为这些标记分配高优先级（本质上限制输出为这些标记之一）。尽管模型名称是必需的，但将 `max_tokens` 参数设置为 `0` 和将 `out_tokens` 参数设置为空字符串允许我们避免更改 OpenAI 的默认设置。

可调分类器使用相当多的命令行参数（**3**）。让我们按照你需要指定的顺序来讨论它们：

+   `file_path`—包含用于评估我们的语言模型的评论的.csv 文件的路径

+   `model`—我们想要使用的语言模型的名称（例如，`gpt-3.5-turbo`)

+   `max_tokens`—每个输入评论生成输出标记的最大数量

+   `out_tokens`—在生成输出时优先考虑的标记的逗号分隔列表

+   `nr_samples`—要整合到每个提示中的评论样本数量

+   `sample_path`—包含带有正确类别标签的评论的.csv 文件的路径，用作样本（如果将`nr_samples`参数设置为`0`，则可以为空）

警告 限制输出标记的数量几乎总是个好主意。特别是，你应该在将输出偏向特定标记而不包括任何“停止”标记（表示输出结束）时这样做。

在解析输入参数后，分类器从磁盘读取样本（**4**）并对评论进行分类（**5**），同时更新最终打印的计数器（**6**）。

让我们看看如何模拟我们迄今为止讨论的所有不同版本的分类器。使用以下调用应该会给出我们未调整的分类器版本，假设 reviews.csv 文件位于代码本身所在的目录中：

```py
python tunable_classifier.py reviews.csv
    gpt-3.5-turbo 0 "" 0 ""
```

注意，我们没有指定任何优先考虑的标记（我们指定空字符串），不限制输出长度（将其设置为`0`表示没有限制），并将提示中的样本数量设置为`0`（这意味着我们也可以将包含样本的文件路径设置为空字符串）。

另一方面，以下命令将给出一个限制输出长度同时优先考虑对应于我们类别标签的标记的版本：

```py
python tunable_classifier.py reviews.csv
    gpt-3.5-turbo 1 "981,29875" 0 ""
```

最后，我们可以通过以下命令获取我们讨论的最后版本，每次提示使用一个样本，同时像以前一样调整模型（假设 train_reviews.csv 文件位于与代码相同的存储库中）：

```py
python tunable_classifier.py reviews.csv
    gpt-3.5-turbo 1 "981,29875" 1 "train_reviews.csv"
```

尝试我们尚未讨论的新组合吧！

## 9.7 微调

到目前为止，我们已经尽我们所能从现有模型中榨取最佳性能。这些模型已经针对可能与我们感兴趣的任务相似但不完全相同的目标进行了训练。如果能够得到一个专门针对我们任务的定制模型，那岂不是很好？使用微调就可以实现这一点。让我们看看如何在实践中使用 OpenAI 的模型实现微调。

微调意味着我们取一个现有的模型，例如 OpenAI 的 GPT-3.5 Turbo 模型，并使其专门针对我们感兴趣的任务。当然，从原则上讲，我们可以从头开始训练我们的模型。但那通常成本过高，而且此外，我们通常找不到足够的特定任务训练数据来在训练期间维持一个大模型。这就是为什么依赖微调要好得多。

微调通常是我们为了最大化特定任务的性能而尝试的最后一件事。原因是微调需要在时间和金钱上做出一定的前期投资。在微调过程中，我们支付 OpenAI 为其基础模型之一创建一个定制的版本，专门用于我们的任务。价格基于训练数据的大小和训练数据被读取的次数（即，*epoch*的数量）。例如，在撰写本文时，微调 GPT-3.5 Turbo 的费用大约是每 1,000 个训练数据 token 和 epoch 0.8 美分。此外，微调后，我们还需要支付使用微调模型的费用。与基础版本相比，微调模型的每 token 价格更高。这在理论上是有道理的，因为至少在理论上，微调模型应该在我们的特定任务上表现更好。

微调的一个可能优势是提高模型输出的准确性。另一个可能的优势是，我们可能能够缩短我们的提示。当使用通用模型时，提示需要包含要执行的任务的描述（以及所有相关数据）。另一方面，我们的微调模型应该专门用于执行单个任务，并且在该任务上表现良好。如果模型只需要执行一个任务，原则上应该可以省略任务描述，因为它已经是隐含的。除了任务描述之外，我们还可以省略对通用模型有帮助但不是专门模型所必需的其他信息。例如，对于通用模型来说，可能需要将样本集成到提示中，以获得合理的输出质量，而对于微调版本来说，这可能是多余的。

在我们的特定场景中，我们希望将评论映射到类标签（基于评论作者的潜在情感）。之前，我们将分类任务作为提示的一部分进行了指定（甚至提供了一些有用的示例）。现在，也许在微调模型时，我们可以省略这些指令。更准确地说，我们可能不再需要使用以下提示（包含样本评论（**1**）、指令（**2**）和样本解决方案（**3**），以及要分类的评论（**4**）和相应的指令（**5**））：

```py
Now, I won't deny that when I purchased #1
this off eBay, I had high expectations. ...
Is the sentiment positive or negative?  #2
Answer ("pos"/"neg"):neg          #3
I am willing to tolerate almost anything            #4
in a Sci-Fi movie, but this was almost intolerable. ...
Is the sentiment positive or negative?              #5
Answer ("pos"/"neg"): 
```

#1 样本评论

#2 指令

#3 样本解决方案

#4 分类评论

#5 指令

相反，我们可以假设模型隐含地知道它应该对评论进行分类以及哪些类标签是可用的。基于这个假设，我们可以将提示简化为以下内容：

```py
I am willing to tolerate almost anything
in a Sci-Fi movie, but this was almost intolerable. ...
```

这个提示仅仅陈述了我们想要分类的评论。我们假设所有其他特定任务的信息（例如指令和样本）模型已经隐含地知道。正如你肯定注意到的，这个提示比之前的版本要短得多。这意味着当我们使用微调模型而不是基础版本时，我们可能会节省一些钱。另一方面，请记住，使用微调模型比使用基础版本每 token 的成本更高。我们将相应的计算推迟到以后。但首先，让我们看看我们是否可以通过微调使这样的简洁提示在实践上发挥作用。

## 9.8 生成训练数据

首先，我们必须生成我们的微调训练数据。我们将使用包含在文件 train_reviews.csv 中的评论及其关联的标签，该文件可在配套网站上的“评论训练”部分找到。OpenAI 期望微调的训练数据具有一个非常特定的格式。在我们能够微调之前，我们需要将我们的.csv 数据转换为所需的格式。

微调 OpenAI 的聊天模型的数据通常采用与模型成功交互的形式（即，模型产生我们理想中想要产生的输出的例子）。在 OpenAI 的聊天模型的情况下，这种交互通过消息历史来描述。每条消息都由一个 Python 字典对象描述。例如，以下描述了一个成功完成，给定之前的示例评论作为输入：

```py
{'messages':[
    {'role':'user', 'content':'I am willing to tolerate almost anything 
    ↪ ...'},
    {'role':'assistant', 'content':'neg'}
]}
```

这是一个负面评论（即，评论作者不想推荐这部电影），因此，我们理想中希望模型生成包含单个 token `neg` 的消息。这就是这里描述的交互。

为了使微调变得值得，你通常至少需要使用 50 个样本，最多几千个样本。使用更多样本进行微调可以提高性能，但成本也更高。另一方面，这是一个一次性费用，因为你可以为可能的大型数据集重复使用相同的微调模型（并且微调模型的费用不取决于用于微调的训练数据量）。示例文件（reviews_train.csv）包含 100 个样本，因此它位于微调可能变得有用的数据大小范围内。

OpenAI 期望微调数据以 JSON-line 格式（这类文件通常有.jsonl 后缀）。符合此格式的文件实际上每行包含一个 Python 字典。在这种情况下，每行描述了与模型的一次成功交互（使用与上一个示例相同的格式）。为了更容易地从 Python 处理 JSON-line 文件，我们将使用`jsonlines`库。作为第一步，前往终端并使用以下命令安装库：

```py
pip install jsonlines==4.0
```

现在我们可以使用库将我们的 .csv 数据转换为 OpenAI 所需的格式。列表 9.3 使用 `get_samples` 函数（**1**）准备所需格式的样本。输入是一个包含训练样本的 pandas DataFrame（`df` 参数），这些样本以通常的格式表示（我们假设 `text` 列包含评论，而 `sentiment` 列包含相关的类标签）。我们将每个样本转换为与模型成功消息交换。首先，我们创建用户发送的消息（**2**），它仅包括评论文本。其次，我们创建模型要生成的期望答案消息（与“助手”角色相关）（**3**）。完整的训练样本集是一系列消息交换，每个都以前述格式准备。

##### 列表 9.3 为微调生成训练数据

```py
import argparse
import jsonlines
import pandas

def get_samples(df):                    #1
    """ Generate samples from a data frame.

    Args:
        df: data frame containing samples.

    Returns:
        List of samples in OpenAI format for fine-tuning.
    """
    samples = []
    for _, row in df.iterrows():
         #2
        text = row['text']
        user_message = {'role':'user', 'content':text}
         #3
        label = row['sentiment']
        assistant_message = {'role':'assistant', 'content':label}

        sample = {'messages':[user_message, assistant_message]}
        samples += [sample]

    return samples

if __name__ == '__main__':
    #4
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to input')
    parser.add_argument('out_path', type=str, help='Path to output')
    args = parser.parse_args()

    df = pandas.read_csv(args.in_path)
    samples = get_samples(df)    
     #5
    with jsonlines.open(args.out_path, 'w') as file:
        for sample in samples:
            file.write(sample)
```

#1 生成训练数据

#2 创建用户消息

#3 创建助手消息

#4 解析命令行参数

#5 以新格式存储训练数据

列表 9.3 预期输入一个包含训练样本 .csv 文件的路径，以及输出文件的路径（**4**）。输出文件遵循 JSON-lines 格式，因此我们理想地分配一个以 .jsonl 结尾的输出路径。在将输入 .csv 文件转换为微调格式后，我们使用 `jsonlines` 库将转换后的样本写入 JSON-lines 文件（**5**）。

如往常一样，您不需要为此列表输入代码。您可以在网站上的“准备微调”部分找到它。使用以下命令在终端中运行它（我们假设 train_reviews.csv 文件位于与代码相同的存储库中）：

```py
python prep_fine_tuning.py   train_reviews.csv train_reviews.jsonl
```

您可能需要手动检查由运行此命令（希望）生成的 train_reviews.jsonl 文件。您应该看到每行一个训练样本，以 Python 字典的形式表示。

## 9.9 开始微调作业

现在我们已经将训练数据格式化正确，我们可以在 OpenAI 的平台上创建一个微调作业。当然，因为模型仅存储在 OpenAI 的平台上，我们无法自行进行微调。相反，我们将我们的训练数据发送给 OpenAI，并请求使用这些数据创建一个定制模型。要创建一个定制模型，我们首先必须选择一个基础模型。在这种情况下，我们将从 GPT-3.5 Turbo 模型开始（这使得与迄今为止获得的结果进行比较更容易）。

我们可以使用以下代码片段创建微调作业（假设 `in_path` 是包含训练数据的文件的路径）：

```py
import openai
client = openai.OpenAI()

reply = client.files.create(
    file=open(in_path, 'rb'), purpose='fine-tune')
```

`reply` 对象将包含有关我们微调作业的元数据的 Python 对象（假设作业创建成功）。最重要的是，我们在 `reply.id` 字段中获取了我们刚刚创建的作业的 ID。微调作业通常需要一段时间（我们描述的微调作业通常需要大约 15 分钟）。这意味着我们必须等待我们的微调模型被创建。作业 ID 允许我们验证微调作业的状态，并在模型可用时检索新创建的模型的 ID。我们可以使用以下 Python 代码检索关于微调作业的状态信息：

```py
reply = client.fine_tuning.jobs.retrieve(job_id)
```

`reply.status` 字段报告微调作业的状态，最终将达到 `succeeded` 值。在那之后，我们可以在 `reply.fine_tuned_model` 中检索微调模型的 ID。

列表 9.4 开始微调过程，等待相应的作业完成，并最终打印出生成的模型的 ID。给定包含训练数据的文件路径，代码首先上传包含训练数据的文件（**1**）。它检索 OpenAI 分配的文件 ID，并使用它来创建微调作业（**2**）。然后，我们迭代直到微调作业成功完成（**3**）。在每次迭代中，我们打印出一个计时器（测量自微调作业开始以来的秒数）并检查作业的状态更新（**4**）。最后，我们检索模型 ID 并打印它（**5**）。

##### 列表 9.4 使用训练数据微调 GPT 模型

```py
import argparse
import openai
import time

client = openai.OpenAI()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to input file')
    args = parser.parse_args()

    reply = client.files.create(              #1
        file=open(args.in_path, 'rb'), purpose='fine-tune')
    file_id = reply.id

    reply = client.fine_tuning.jobs.create(       #2
        training_file=file_id, model='gpt-3.5-turbo')
    job_id = reply.id
    print(f'Job ID: {job_id}')

    status = None
    start_s = time.time()

    while not (status == 'succeeded'):  #3

        time.sleep(5)
        total_s = time.time() - start_s
        print(f'Fine-tuning since {total_s} seconds.')
         #4
        reply = client.fine_tuning.jobs.retrieve(job_id) 
        status = reply.status
        print(f'Status: {status}')
    #5
    print(f'Fine-tuning is finished!')
    model_id = reply.fine_tuned_model
    print(f'Model ID: {model_id}')
```

#1 将训练数据上传到 OpenAI

#2 创建微调作业

#3 循环直到作业完成

#4 获取作业状态

#5 获取微调模型的 ID

你可以在“开始微调”网页上找到代码。使用以下命令运行它（其中 train_reviews.jsonl 是之前生成的文件）：

```py
python fine_tune.py train_reviews.jsonl
```

如果你运行脚本直到完成，你将看到如下输出（这当然只是输出的一部分；点代表缺失的行）：

```py
Job ID: ...
Fine-tuning since 5.00495171546936 seconds.
Status: validating_files
...
Fine-tuning since 46.79299879074097 seconds.
Status: running
...
Fine-tuning since 834.6565797328949 seconds.
Status: succeeded
Fine-tuning is finished!
Model ID: ft:gpt-3.5-turbo-0613...
```

在打印出作业 ID 后，我们会定期收到关于作业状态的更新，通常是从 `validating_files` 到 `running`，然后（希望）到 `succeeded`。问题是作业可能需要一段时间才能完成（对于前面的例子，大约需要 14 分钟）。如果你不想连续运行脚本（例如，为了关闭电脑），你可以在微调作业开始后中断脚本（你将知道因为脚本在那个点打印出了作业 ID）。微调作业将在 OpenAI 的服务器上按计划进行。根据你的设置，你甚至可能会收到一封电子邮件通知你作业已完成。否则，你可以定期运行此脚本。

##### 列表 9.5 检查微调作业的状态

```py
import argparse
import openai

client = openai.OpenAI()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('job_id', type=str, help='ID of fine-tuning job')
    args = parser.parse_args()
     #1
    job_info = client.fine_tuning.jobs.retrieve(args.job_id)
    print(job_info)
```

#1 获取并打印作业元数据

给定工作 ID（从列出 9.4 的输出中检索），脚本检索并打印工作元数据（**1**），包括工作状态和成功完成工作后的结果模型 ID。

## 9.10 使用微调模型

恭喜！你已经创建了一个专门化的模型，针对你关心的任务（审查分类）进行了微调。你该如何使用它？幸运的是，使用 OpenAI 库这样做很简单。我们不再指定标准模型之一（例如，`gpt-3.5-turbo`）的名称，而是现在指定我们微调模型的 ID，如下所示（将占位符`[Fine-tuned model ID]`替换为实际的模型 ID）：

```py
import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model='[Fine-tuned model ID]',
    messages=[
        {'role':'user', 'content':prompt}
        ]
    )
```

如前所述，我们假设`prompt`变量包含提示文本。然而，对于我们的微调模型，提示有所不同。以前，我们描述了分类任务，包括审查文本。现在我们已训练我们的自定义模型，将审查文本单独映射到适当的类别。这意味着我们的提示生成函数简化为以下内容（实际上，你可能会争论创建专用函数不再需要）：

```py
def create_prompt(text):
    """ Create prompt for sentiment classification.

    Args:
        text: text to classify.

    Returns:
        Prompt for text classification.
    """
    return text
```

我们不是生成多部分提示，而是将审查文本返回以进行分类。你可能想了解使用简化提示与原始模型（`gpt-3.5-turbo`）时会发生什么。你将看到如下输出：

```py
Label: I understand your concern about smoking in movies, 
especially those intended for children and adolescents. 
Smoking in films can have an influence on young viewers 
and potentially normalize the behavior. However, it is 
important to note that not all instances of smoking in 
movies are the result of intentional product placement 
or sponsorship by tobacco companies.
...
```

显然，模型对我们的意图感到困惑——也就是说，我们期望它如何处理输入的审查。它没有生成正确的类别标签，而是写下了详尽的评论，评论了审查中提出的主要观点。这是意料之中的。想象一下，如果有人给你一份没有进一步指示的审查，你会知道那个人想让你分类审查，更不用说可能的类别的正确标签了？这几乎是不可能的，语言模型也是如此。

然而，如果我们切换到我们的微调模型并使用相同的提示作为输入，我们将得到以下输出而不是：

```py
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: neg
Label: pos; Ground truth: pos
Label: pos; Ground truth: neg
Label: pos; Ground truth: neg
Label: neg; Ground truth: neg
Label: neg; Ground truth: pos
Label: neg; Ground truth: neg
Number of correct labels:    7  #1
Number of tokens used   :    2085  #2
```

#1 提高准确性

#2 降低令牌消耗

注意，即使没有设置任何微调参数（或在提示中提供任何样本），我们现在得到的准确率是 70%（**1**），而不是原始版本的 60%！而且，与初始版本相比，使用的令牌数量减少了大约 200（**2**）。这是因为我们在每个提示中省略了指令（和类别标签）。

好的！我们已经看到，我们可以微调一个模型以准确分类评论，同时减少提示大小。但问题仍然存在：这是否值得？让我们做一些计算来找出答案。我们暂时不考虑微调模型的成本，因为我们只需要做一次（在我们的示例场景中，我们假设我们想要分析一年的评论）。在不进行微调的情况下，当利用调整参数（设置偏差和输出标记数的限制）时，我们可以使用通用模型达到相同的准确率（70%）。在这种情况下，我们为我们的 10 个样本评论使用了 2,228 个标记。微调后，我们只为我们的样本评论使用了 2,085 个标记。然而，对于通用模型，我们每 1,000 个输入标记支付 0.05 美分。另一方面，对于微调模型，我们每 1,000 个标记支付 0.3 美分。这意味着微调后我们的每标记成本提高了六倍！在这个特定场景中，处理标记数的适度减少并不能抵消每标记更高费用的增加。

通常，微调在提高质量和可能降低成本方面非常有帮助。然而，请注意，它伴随着各种开销。在生产中使用微调模型之前，请进行实验评估，进行计算，并确保它是值得的！

## 摘要

+   调整参数设置可以影响模型性能和成本。

+   考虑限制输出长度并引入标记对数偏差。

+   不要总是使用可用的最大模型，因为这样做会增加成本。

+   通过在样本上评估来确定最适合您任务的模型。

+   提示的设计可以对性能产生重大影响。

+   在提示中包含正确解决的任务样本，以进行少样本学习。

+   微调允许您将基础模型专门化到您关心的任务上。它可能允许您通过专门化来减少提示大小。

+   微调产生的开销与训练数据量成正比。当您使用生成的模型时，它还会增加每标记的成本。
