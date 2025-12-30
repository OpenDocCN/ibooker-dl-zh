# 前言

2022 年 11 月 30 日，总部位于旧金山的公司 OpenAI[公开发布了 ChatGPT](https://oreil.ly/uAnsr)—这是一个病毒式 AI 聊天机器人，可以像人类一样生成内容、回答问题和解决问题。自发布以来两个月内，ChatGPT 吸引了超过[1 亿每月活跃用户](https://oreil.ly/ATsLe)，这是新消费技术应用的最高采用率（到目前为止）。ChatGPT 是一个由 OpenAI 的 GPT-3.5 系列大型语言模型（LLM）的指令和对话调整版本驱动的聊天机器人体验。我们很快就会接触到这些概念的定义。

###### 注意

使用或不用 LangChain 构建 LLM 应用都需要使用 LLM。在这本书中，我们将使用 OpenAI API 作为我们在代码示例中使用的 LLM 提供商（价格列表可在其[平台](https://oreil.ly/-YYoR)上查看）。与 LangChain 合作的一个好处是，你可以使用 OpenAI 或替代的商业或开源 LLM 提供商来跟随所有这些示例。

三个月后，OpenAI[发布了 ChatGPT API](https://oreil.ly/DwU7R)，为开发者提供了访问聊天和语音到文本功能的能力。这启动了无数新的应用和技术发展，这些应用和技术发展在*生成式 AI*这个宽松的术语下展开。

在我们定义生成式 AI 和 LLM 之前，让我们先谈谈*机器学习*（ML）的概念。一些计算机*算法*（想象一下实现某些预定义任务的可重复食谱，例如排序一副扑克牌）是由软件工程师直接编写的。其他计算机算法则是从大量的训练示例中*学习*的——软件工程师的工作从编写算法本身转变为编写创建算法的训练逻辑。在 ML 领域，大量的注意力都集中在开发用于预测各种事物的算法上，从明天的天气到亚马逊司机的最有效配送路线。

随着 LLM 和其他生成式模型（例如用于生成图像的扩散模型，我们在这本书中不涉及）的出现，那些相同的 ML 技术现在被应用于生成新内容的问题，例如同时具有独特性和受训练数据示例启发的文本段落或绘画。特别是 LLM 是专门用于生成文本的生成式模型。

LLM 与之前的 ML 算法有两个其他的不同之处：

+   它们在大量数据上进行训练；从头开始训练这些模型会非常昂贵。

+   它们更加灵活。

同样的文本生成模型可以用于摘要、翻译、分类等等，而之前的 ML 模型通常是为了特定任务进行训练和使用的。

这两个差异共同导致软件工程师的工作再次发生转变，他们投入越来越多的时间来研究如何让 LLM 为他们的用例工作。这正是 LangChain 的核心所在。

到 2023 年底，竞争性的 LLMs 出现，包括 Anthropic 的 Claude 和 Google 的 Bard（后来更名为 Gemini），为这些新能力提供了更广泛的访问。随后，数千家成功的初创公司和大型企业将生成式 AI API 整合到构建各种用例的应用程序中，从客户支持聊天机器人到编写和调试代码。

2022 年 10 月 22 日，Harrison Chase 在 GitHub 上[发布了 LangChain 开源库的第一个提交](https://oreil.ly/mCdYZ)。LangChain 始于这样一个认识，即最有趣的 LLM 应用需要使用 LLM 与“其他计算或知识来源”一起使用。[“其他计算或知识来源”](https://oreil.ly/uXiPi)。例如，你可以尝试让 LLM 生成这个问题的答案：

```py
How many balls are left after splitting 1,234 balls evenly among 123 people?
```

你可能会对其数学能力感到失望。然而，如果你将其与计算器函数配对，你可以指示 LLM 将问题重新措辞为一个计算器可以处理的数据输入：

```py
1,234 % 123
```

然后，你可以将其传递给计算器函数，并得到你原始问题的准确答案。LangChain 是第一个（并且，在撰写本文时，也是最大的）提供这些构建块和工具库的库，可以可靠地将它们组合成更大的应用程序。在讨论如何使用这些新工具构建引人入胜的应用程序之前，让我们更熟悉 LLMs 和 LangChain。

# LLMs 简明指南

用通俗易懂的话来说，LLMs 是经过训练的算法，它们接收文本输入并预测和生成类似人类的文本输出。本质上，它们就像许多智能手机上常见的自动完成功能，但被推向了极致。

让我们分解一下术语*大型语言模型*：

+   “大型”指的是这些模型在学习和训练过程中使用的训练数据和参数的大小。例如，OpenAI 的 GPT-3 模型包含 1750 亿个*参数*，这些参数是通过在 4500 太字节文本数据上训练获得的.^(1) 神经网络模型中的*参数*由控制每个*神经元*输出的数字及其与相邻神经元的相对权重组成。（连接到哪些其他神经元的神经元对于每个神经网络架构都是不同的，这超出了本书的范围。）

+   *语言模型* 指的是一种计算机算法，经过训练可以接收书面文本（英语或其他语言）并产生书面文本输出（同一种语言或不同的一种）。这些是 *神经网络*，一种类似于人类大脑风格化概念的机器学习模型，最终输出是由许多简单数学函数（称为 *神经元*）的个别输出及其相互连接的组合产生的。如果许多神经元以特定方式组织，并且经过适当的训练过程和训练数据，这将产生一个能够解释单个单词和句子含义的模型，这使得它们可以用于生成合理、可读的书面文本。

由于训练数据中英语的普遍性，大多数模型在英语上的表现优于其他说英语较少的语言。通过“更好”我们指的是更容易让它们在英语中产生期望的输出。有一些 LLM 是为多语言输出设计的，例如 [BLOOM](https://oreil.ly/Nq7w0)，它使用更大比例的其他语言的训练数据。有趣的是，即使在以英语为主的训练语料库上训练的 LLM 中，不同语言之间的性能差异也没有预期的那么大。研究人员发现，LLM 能够将其部分语义理解转移到其他语言上。（^(2））

总的来说，*大型语言模型* 是在大规模文本上训练的大规模通用语言模型的实例。换句话说，这些模型已经从大量文本数据集的规律中学习——书籍、文章、论坛和其他公开可用的来源，以执行各种文本相关任务。这些任务包括文本生成、摘要、翻译、分类等。

假设我们指示一个 LLM 完成以下句子：

```py
The capital of England is _______.
```

LLM 将接受该输入文本并预测正确的输出答案为 `London`。这看起来像是魔法，但并非如此。在底层，LLM 估计给定先前单词序列的单词序列的概率。

###### 小贴士

从技术角度讲，该模型基于标记进行预测，而不是基于单词。一个 *标记* 代表文本的原子单位。标记可以代表单个字符、单词、子词，甚至更大的语言单位，具体取决于所使用的特定标记化方法。例如，使用 GPT-3.5 的标记化器（称为 `cl100k`），短语 *good morning dearest friend* 将由 [五个标记](https://oreil.ly/dU83b) 组成（使用 `_` 来表示空格字符）：

`Good`

使用标记 ID `19045`

`_morning`

使用标记 ID `6693`

`_de`

使用标记 ID `409`

`arest`

使用标记 ID `15795`

`_friend`

使用标记 ID `4333`

通常，标记化器（tokenizer）的训练目标是使最常见的单词编码为单个标记，例如，单词*morning*被编码为标记`6693`。不常见的单词或其他语言的单词（通常标记化器是在英语文本上训练的）需要多个标记来编码。例如，单词*dearest*被编码为标记`409, 15795`。对于常见的英语文本，一个标记平均跨越四个字符的文本，或大约四分之三的单词。

LLM 预测能力的驱动引擎被称为*变换器神经网络架构*。3 变换器架构使模型能够处理数据序列，如句子或代码行，并预测序列中最可能出现的下一个单词。变换器的设计是通过考虑每个单词与其他单词的关系来理解句子中每个单词的上下文。这使得模型能够构建对句子、段落等（换句话说，单词序列）的综合理解，即其各部分之间的联合意义。

因此，当模型看到单词序列*the capital of England is*时，它会根据其在训练期间看到的类似示例进行预测。在模型的训练语料库中，单词*England*（或代表它的标记）通常会出现在与*France*、*United States*、*China*等单词相似的句子中。单词*capital*会在包含*England*、*France*和*US*等单词以及*London*、*Paris*、*Washington*等单词的许多句子中出现。这种在模型训练过程中的重复导致了正确预测序列中下一个单词应该是*London*的能力。

你提供给模型的指令和输入文本被称为*提示*。提示可以显著影响 LLM 输出的质量。关于*提示设计*或*提示工程*有几种最佳实践，包括提供清晰简洁的指令和上下文示例，我们将在本书后面讨论。在进一步探讨提示之前，让我们看看可供你使用的不同类型的 LLM。

基础类型，所有其他类型都由此派生，通常被称为*预训练的 LLM*：它通过自我监督的方式在非常大量的文本上进行了训练（这些文本来自互联网和书籍、报纸、代码、视频脚本等）。这意味着——与监督机器学习不同，在训练之前，研究人员需要组装一对对的*输入*到*预期输出*的数据集——对于 LLM 来说，这些对是从训练数据中推断出来的。实际上，使用如此庞大的数据集的唯一可行方法是从训练数据中自动组装这些对。实现这一目标的技术包括以下两种：

预测下一个单词

从训练数据中的每一句话中移除最后一个词，这样就产生了一对*输入*和*预期输出*，例如“英国的首都是 ___”和“伦敦”。

预测缺失的单词

同样，如果你取每一句话并从中去掉一个词，你现在就有其他输入和预期输出的对，例如“英国的 ___ 是伦敦”和“首都”。

这些模型本身使用起来相当困难，它们需要你用一个合适的词缀来预热回答。例如，如果你想了解英国的首都，你可能通过提示模型“英国的首都是”来得到回答，而不是更自然的“英国的首都是什么？”

## 指令调整型大型语言模型

[研究人员](https://oreil.ly/lP6hr)通过进一步训练（在上一节中描述的漫长且昂贵的训练之上进行的额外训练），也称为在以下方面对预训练的大型语言模型进行微调：

任务特定数据集

这些是由研究人员手动组装的问答数据集对，提供了用户可能向模型提出的一些常见问题的期望回答示例。例如，数据集可能包含以下一对：*Q: 英国的首都是什么？ A: 英国的首都是伦敦。* 与预训练数据集不同，这些是手动组装的，因此它们必然要小得多：

来自人类反馈的强化学习（RLHF）

通过使用[RLHF 方法](https://oreil.ly/lrlAK)，那些手动组装的数据集通过模型产生的输出收到的用户反馈得到了增强。例如，用户 A 更喜欢“英国的首都是伦敦”而不是“伦敦是英国的首都”作为之前问题的答案。

指令调整对于扩大能够使用大型语言模型构建应用程序的人数至关重要，因为现在它们可以用“指令”来提示，通常是问题的形式，例如，“英国的首都是什么？”，而不是“英国的首都是”。

## 对话调整型大型语言模型

专为对话或聊天目的定制的模型是对指令调整型大型语言模型的进一步增强。不同的大型语言模型提供商使用不同的技术，因此这并不一定适用于所有*聊天模型*，但通常这是通过以下方式实现的：

对话数据集

手动组装的微调数据集被扩展，包括更多多轮对话交互的示例，即提示-回复对的序列。

聊天格式

模型的输入和输出格式在自由文本之上增加了一层结构，将文本划分为与角色相关的部分（以及可选的其他元数据，如名称）。通常，可用的角色包括*系统*（用于指令和任务的框架）、*用户*（实际的任务或问题）和*助手*（用于模型的输出）。这种方法源于早期的[提示工程技术](https://oreil.ly/dINx0)，使得调整模型的输出更加容易，同时让模型更难将用户输入与指令混淆。将用户输入与先前指令混淆也称为*越狱*，例如，可能导致精心设计的提示被暴露给最终用户，这些提示可能包括商业机密。

## 微调的 LLM

微调的 LLM 是通过在特定任务的专有数据集上进一步训练基础 LLM 创建的。技术上讲，指令微调和对话微调的 LLM 都是微调的 LLM，但“微调的 LLM”这个术语通常用来描述由开发者针对特定任务进行调整的 LLM。例如，一个模型可以被微调以从上市公司的年度报告中准确提取情绪、风险因素和关键财务数据。通常，微调模型在所选任务上的性能有所提高，但以泛化能力的损失为代价。也就是说，它们在回答与相关任务无关的查询时变得不那么有能力。

在本书的其余部分，当我们使用术语*LLM*时，我们指的是指令微调的 LLM，而对于*聊天模型*，我们指的是对话指令的 LLM，如本节之前定义的。这些应该是您使用 LLM 时的得力助手——在开始新的 LLM 应用程序时首先使用的工具。

现在，在我们深入探讨 LangChain 之前，让我们快速讨论一些常见的 LLM 提示技术。

# 提示简明指南

正如我们之前提到的，与 LLM 合作的软件工程师的主要任务不是训练一个 LLM，甚至通常也不是微调一个 LLM，而是将现有的 LLM 用于您应用程序所需的任务。有商业 LLM 提供商，如 OpenAI、Anthropic 和 Google，以及开源 LLM（[Llama](https://oreil.ly/ld3Fu)、[Gemma](https://oreil.ly/RGKfi)等），免费提供给他人构建。为您的任务调整现有 LLM 称为*提示工程*。

在过去两年中，已经开发了许多提示技术，从广义上讲，这本书是关于如何使用 LangChain 进行提示工程——如何使用 LangChain 让 LLM 做你想做的事情。但在我们深入探讨 LangChain 之前，先回顾一下这些技术是有帮助的（如果您的最喜欢的[提示技术](https://oreil.ly/8uGK_)没有列在这里，我们提前表示歉意；要涵盖的内容太多）。

为了跟随本节内容，我们建议您将这些提示复制到 OpenAI 游乐场中亲自尝试：

1.  在[*http://platform.openai.com*](http://platform.openai.com)创建一个 OpenAI API 的账户，这将允许你通过 Python 或 JavaScript 代码程序化地使用 OpenAI LLMs，也就是说，使用 API 从你的代码中调用。它还将为你提供访问 OpenAI Playgrounds 的权限，在那里你可以通过网页浏览器进行实验。

1.  如有必要，为你的新 OpenAI 账户添加支付详情。OpenAI 是 LLMs 的商业提供商，每次你通过 OpenAI 的 API 或通过 Playground 使用他们的模型时都会收费。你可以在他们的[网站](https://oreil.ly/MiKRD)上找到最新的定价。在过去的两年里，随着新功能和优化的引入，使用 OpenAI 模型的价格已经大幅下降。

1.  前往[OpenAI Playground](https://oreil.ly/rxiAG)，你就可以尝试以下提示符了。在这本书中，我们将使用 OpenAI API。

1.  一旦你导航到 Playground，你将在屏幕右侧看到一个预设的面板，包括你选择的项目。如果你继续向下查看面板，你将在“模型配置”标题下看到“温度”。将温度切换从中间移动到左侧，直到数字显示为 0.00。本质上，温度控制 LLM 输出的随机性。温度越低，模型输出越确定。

现在转到提示符！

## 零样本提示

第一种最直接提示技术是简单地指示 LLM 执行所需的任务：

```py
How old was the 30th president of the United States when his wife's mother died?
```

这通常是你应该尝试的第一种方法，它通常适用于简单的问题，特别是当答案很可能包含在训练数据中时。如果我们用前面的提示符提示 OpenAI 的`gpt-3.5-turbo`，则会返回以下内容：

```py
The 30th president of the United States, Calvin Coolidge, was 48 years old when
his wife's mother passed away in 1926.
```

###### 注意

你可能会得到与我们不同的结果。LLMs 生成响应的方式中存在一定的随机性，OpenAI 在你尝试时可能已经更新了模型。

虽然模型正确地识别了第 30 任总统，但答案并不完全正确。通常，你需要迭代提示和响应以获得可靠的系统。接下来的几节将扩展如何做到这一点。

## 思维链

一个非常有用的迭代是进一步指示模型[花时间*思考*](https://oreil.ly/8xWcs)。这项技术已被发现可以增加各种任务的表现。有趣的是，一篇更近期的论文^(4)表明，这项技术还可以降低在思考会降低人类表现的任务中 LLM 的表现。称为思维链（CoT）提示，这通常是通过在提示前添加指令来完成的，指示 LLM 如何得出答案：

```py
Think step by step.

How old was the 30th president of the United States when his wife's mother died?
```

现在让我们看看这个新提示符模型会产生什么结果：

```py
1\. First, find out who the 30th president of the United States was. This can be
done by researching a list of US presidents or by searching for "30th president 
of the United States" online. You will find that the 30th president was Calvin 
Coolidge.

2\. Next, find out the year when Calvin Coolidge was the president. This can also 
be found by researching online or in a history book. You will find that he was 
the president from 1923-1929.

3\. Now, you need to determine the year when Calvin Coolidge's wife's mother died. 
This can be done by researching the life and death of his wife, Grace Goodhue 
Coolidge. You will find that she lived from 1879-1957.

4\. Calculate the age of Calvin Coolidge's wife's mother when she died. She was 
78 years old.

5\. Determine the age of Calvin Coolidge when his wife's mother died. Subtract 
the year of his birth, 1872, from the year his mother-in-law died, 1957\. This 
means that Calvin Coolidge was 85 years old when his wife's mother passed away.
```

注意到提示中的唯一变化是添加了`Think step by step`，以及这一变化在模型输出中的巨大差异。现在模型输出了一系列步骤，你可以在尝试回答这个问题时遵循这些步骤。不幸的是，尽管步骤非常合理，但最终输出仍然不正确，因为它在事实上有一些错误，即它列出了柯立芝总统妻子的母亲错误的出生和去世年份。让我们看看如何改进这一点。

## 检索增强生成

*检索增强生成*（RAG）包括找到相关的文本片段，也称为*上下文*，例如你可以在百科全书找到的事实，并将这些上下文包含在提示中。RAG 技术可以（并且在实际应用中应该）与 CoT（思维链）结合使用，但为了简单起见，我们在这里一次使用这些技术中的一个。以下是一个包含 RAG 的提示示例：

```py
Context:

- Calvin Coolidge (born John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; July 4, 1872 – 
January 5, 1933) was an American attorney and politician who served as the 
30th president of the United States from 1923 to 1929.

- Grace Anna Coolidge (née Goodhue; January 3, 1879 – July 8, 1957) was the
wife of the 30th president of the United States, Calvin Coolidge.

- Grace Anna Goodhue was born on January 3, 1879, in Burlington, Vermont, the 
only child of Andrew Issachar Goodhue and Lemira Barrett Goodhue.

- Lemira A. Goodhue (Barrett) ; Birthdate: April 26, 1849 ; Birthplace: 
Burlington, Chittenden County, VT, United States ; Death: October 24, 1929.

How old was the 30th president of the United States when his wife's mother died?
```

模型的输出如下：

```py
The 30th president of the United States, Calvin Coolidge, was 54 years old when 
his wife's mother, Lemira A. Goodhue, died on October 24, 1929.
```

现在我们离正确答案更近了，但正如我们之前提到的，LLM 在处理标准数学问题方面并不出色。在这种情况下，54 岁的最终结果比实际年龄少了 3 岁。让我们看看我们如何改进这一点。

## 工具调用

*工具调用*技术包括在提示前添加一个外部函数列表，这些函数是 LLM（大型语言模型）可以使用的，以及每个函数的用途描述和如何在输出中指示它*想要*使用这些函数的说明。最后，你——应用程序的开发者——应该解析输出并调用适当的函数。以下是一种实现方式：

```py
Tools:

- calculator: This tool accepts math expressions and returns their result.

- search: This tool accepts search engine queries and returns the first search 
result.

If you want to use tools to arrive at the answer, output the list of tools and
inputs in CSV format, with this header row `tool,input`.

How old was the 30th president of the United States when his wife's mother died?
```

这可能是你得到的输出：

```py
tool,input

calculator,2023-1892

search,"What age was Calvin Coolidge when his mother-in-law died?"
```

虽然 LLM 正确地遵循了输出格式说明，但选择的工具和输入并不适用于这个问题。这涉及到在提示 LLM 时需要记住的最重要的事情之一：*每种提示技术在与（某些）其他技术结合使用时最有用*。例如，我们可以通过将工具调用、思维链和 RAG 结合到一个提示中，使用所有三种技术来改进这一点。让我们看看这会是什么样子：

```py
Context:

- Calvin Coolidge (born John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; July 4, 1872 –
January 5, 1933) was an American attorney and politician who served as the 30th
president of the United States from 1923 to 1929.

- Grace Anna Coolidge (née Goodhue; January 3, 1879 – July 8, 1957) was the wife 
of the 30th president of the United States, Calvin Coolidge.

- Grace Anna Goodhue was born on January 3, 1879, in Burlington, Vermont, the 
only child of Andrew Issachar Goodhue and Lemira Barrett Goodhue.

- Lemira A. Goodhue (Barrett) ; Birthdate: April 26, 1849 ; Birthplace: 
Burlington, Chittenden County, VT, United States ; Death: October 24, 1929.

Tools:

- calculator: This tool accepts math expressions and returns their result.

If you want to use tools to arrive at the answer, output the list of tools and 
inputs in CSV format, with this header row `tool,input`.

Think step by step.

How old was the 30th president of the United States when his wife's mother died?
```

使用这个提示，也许经过几次尝试，我们可能会得到以下输出：

```py
tool,input

calculator,1929 - 1872
```

如果我们解析那个 CSV 输出，并让计算器函数执行操作`1929 - 1827`，我们最终得到正确答案：57 岁。

如前例所示，通过结合 RAG（检索增强生成）、思维链和工具调用，你可以检索到最相关的数据来支撑你的模型输出，然后逐步引导它，确保它有效地使用这个上下文。

## 少样本提示

最后，我们来到另一个非常有用的提示技术：*少样本提示*。这包括向 LLM 提供其他问题的示例和正确答案，这使得 LLM 能够*学习*如何执行新任务，而无需进行额外的训练或微调。与微调相比，少样本提示更加灵活——你可以在查询时即时进行——但功能较弱，你可能会通过微调获得更好的性能。尽管如此，你应该通常总是先尝试少样本提示，然后再进行微调：

静态少样本提示

少样本提示的最基本版本是组装一个预定的少量示例列表，这些示例包含在提示中。

动态少样本提示

如果你组装了一个包含许多示例的数据集，你可以选择每个新查询的最相关示例。

下一个部分将介绍如何使用 LangChain 构建使用 LLM 和这些提示技术的应用程序。

# LangChain 及其重要性

LangChain 是早期提供 LLM 和提示构建块以及将它们可靠地组合成更大应用程序的工具的开源库之一。截至写作时，LangChain 已经积累了超过[2800 万月下载量](https://oreil.ly/8OKbf)，[99,000 GitHub 星标](https://oreil.ly/bF5pc)，以及生成 AI 领域最大的开发者社区（[72,000+](https://oreil.ly/PNWL3)）。它使没有机器学习背景的软件工程师能够利用 LLM 的力量来构建各种应用程序，从 AI 聊天机器人到能够负责任地进行推理和采取行动的 AI 代理。

LangChain 建立在上一节强调的思想之上：提示技术在使用时结合最为有用。为了使这更加容易，LangChain 为每种主要的提示技术提供了简单的*抽象*。通过抽象，我们指的是将那些技术的思想封装到易于使用的包装器中的 Python 和 JavaScript 函数和类。这些抽象被设计得可以很好地协同工作，并组合成更大的 LLM 应用程序。

首先，LangChain 提供了与主要 LLM 提供商的集成，包括商业的（[OpenAI](https://oreil.ly/TTLXA)，[Anthropic](https://oreil.ly/O4UXw)，[Google](https://oreil.ly/12g3Z)等）和开源的（[Llama](https://oreil.ly/5WAVi)，[Gemma](https://oreil.ly/-40Ne)等）。这些集成共享一个通用接口，使得尝试新宣布的 LLM 变得非常容易，并让你避免被锁定在单一提供商。我们将在第一章中使用这些。

LangChain 还提供了 *提示模板* 抽象，这使得你可以多次重用提示，将提示中的静态文本与每次发送到 LLM 以生成完成时不同的占位符分开。我们将在第一章 1 中更多地讨论这些内容。LangChain 提示也可以存储在 LangChain Hub 中，以便与队友分享。

LangChain 包含许多与第三方服务的集成（例如 Google Sheets、Wolfram Alpha、Zapier，仅举几例），这些集成以 *工具* 的形式暴露出来，这是工具调用技术中使用的函数的标准接口。

对于 RAG，LangChain 提供了与主要 *嵌入模型*（设计用于输出句子、段落等含义的数值表示，即 *嵌入* 的语言模型）、*向量存储*（专门用于存储嵌入的数据库）和 *向量索引*（具有向量存储能力的常规数据库）的集成。你将在第二章 2 和第三章 3 中了解更多关于这些内容。

对于 CoT，LangChain（通过 LangGraph 库）提供了 *代理* 抽象，这些抽象结合了思维链推理和工具调用，最初由 [ReAct 论文](https://oreil.ly/27BIC) 提出。这使得构建执行以下操作的 LLM 应用成为可能：

1.  推理出需要采取的步骤。

1.  将这些步骤转换为外部工具调用。

1.  接收这些工具调用的输出。

1.  重复直到任务完成。

我们在第五章 5 到第八章 8 中涵盖了这些内容。

对于聊天机器人用例，跟踪之前的交互并在生成对未来的交互的响应时使用这些交互变得很有用。这被称为 *记忆*，第四章 4 讨论了在 LangChain 中使用它。

最后，LangChain 提供了将这些构建块组合成统一应用的工具。第一章 1 到第六章 6 讨论了更多关于这一点的内容。

除了这个库之外，LangChain 还提供了 [LangSmith](https://oreil.ly/geRgx)——一个用于帮助调试、测试、部署和监控 AI 工作流的平台，以及 LangGraph 平台——一个用于部署和扩展 LangGraph 代理的平台。我们在第九章 9 和第十章 10 中涵盖了这些内容。

# 从这本书中可以期待什么

通过这本书，我们希望传达将 LLM 添加到你的软件工程工具包中的兴奋和可能性。

我们之所以涉足编程，是因为我们喜欢构建事物，完成项目，看到最终产品，并意识到世界上有新事物出现，而我们就是创造者。使用 LLM 进行编程对我们来说非常激动人心，因为它扩大了我们能够构建的事物范围，使以前难以做到的事情变得简单（例如，从长文本中提取相关数字），并使以前不可能的事情成为可能——试想一年前尝试构建一个自动助手，你最终会陷入我们所有人都熟悉和喜爱的“电话树地狱”，这是从拨打客户支持电话号码时遇到的。

现在有了 LLM 和 LangChain，你实际上可以构建令人愉快的助手（或无数其他应用）来与你聊天，并能以非常合理的程度理解你的意图。这种差异是巨大的！如果你觉得这很令人兴奋（就像我们一样），那么你就来对地方了。

在本前言中，我们为您回顾了 LLM 的工作原理以及为什么这会给你“构建事物”的超能力。拥有这些理解语言并能以对话式英语（或任何其他语言）输出答案的非常大的 ML 模型，通过提示工程使其可编程，提供了一种多功能的语言生成工具。到本书结束时，我们希望你能看到这有多么强大。

我们将从大部分使用普通英语指令定制的 AI 聊天机器人开始。这一点本身就足以让人眼前一亮：你现在可以不写代码就“编程”你应用程序的部分行为。

然后是下一个功能：给你的聊天机器人提供访问你自己的文档的能力，这使它从通用助手转变为对任何你能够找到文本库的人类知识领域的知识丰富的助手。这将允许你让聊天机器人回答问题或总结你写的文档，例如。

之后，我们将使聊天机器人记住你之前的对话。这将从两个方面改进它：与记得你之前聊过什么的聊天机器人进行对话会感觉更加自然，而且随着时间的推移，聊天机器人可以根据每个用户的偏好进行个性化。

接下来，我们将使用思维链和工具调用技术，使聊天机器人能够规划和执行这些计划，并迭代地进行。这将使其能够处理更复杂的请求，例如撰写关于你选择的主题的研究报告。

当你使用你的聊天机器人执行更复杂的任务时，你会感到需要给它提供工具以便与你协作。这包括在你采取行动之前打断或授权行动的能力，以及为聊天机器人提供在采取行动之前请求更多信息或澄清的能力。

最后，我们将向您展示如何将您的聊天机器人部署到生产环境中，并讨论在采取这一步骤之前和之后需要考虑的因素，包括延迟、可靠性和安全性。然后我们将向您展示如何监控生产环境中的聊天机器人，并在其使用过程中持续改进它。

在此过程中，我们将向您传授每种技术的细节，以便您完成本书后，您实际上已经为您的软件工程工具箱添加了新的工具（或两个）。

# 本书使用的约定

本书使用的排版约定如下：

*斜体*

表示新术语、URL、电子邮件地址、文件名和文件扩展名。

`常宽字体`

用于程序列表，以及段落中引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

###### 提示

此元素表示提示或建议。

###### 注意

此元素表示一般性说明。

# 使用代码示例

补充材料（代码示例、练习等）可在[*https://oreil.ly/supp-LearningLangChain*](https://oreil.ly/supp-LearningLangChain)下载。

如果您在使用代码示例时遇到技术问题或问题，请通过*support@oreilly.com*发送电子邮件。

本书旨在帮助您完成工作。一般来说，如果本书提供了示例代码，您可以在您的程序和文档中使用它。除非您正在复制代码的很大一部分，否则您无需联系我们获取许可。例如，编写一个使用本书中多个代码片段的程序不需要许可。通过引用本书并引用示例代码来回答问题不需要许可。将本书的大量示例代码纳入您产品的文档中则需要许可。

我们感谢，但通常不需要归属。归属通常包括标题、作者、出版商和 ISBN。例如：“*Learning LangChain* by Mayo Oshin and Nuno Campos (O’Reilly). Copyright 2025 Olumayowa “Mayo” Olufemi Oshin, 978-1-098-16728-8。”

如果您认为您对代码示例的使用超出了合理使用或上述许可的范围，请随时通过*permissions@oreilly.com*联系我们。

# O’Reilly 在线学习

###### 注意

40 多年来，[*O’Reilly Media*](https://oreilly.com)一直提供技术和商业培训、知识和洞察力，以帮助公司成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专业知识。O’Reilly 的在线学习平台为您提供按需访问现场培训课程、深入的学习路径、交互式编码环境以及来自 O’Reilly 和 200 多家其他出版商的大量文本和视频。更多信息，请访问[*https://oreilly.com*](https://oreilly.com)。

# 如何联系我们

请将有关本书的评论和问题寄给出版社：

+   O’Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-889-8969 (美国或加拿大)

+   707-827-7019 (国际或本地)

+   707-829-0104 (传真)

+   *support@oreilly.com*

+   [*https://oreilly.com/about/contact.html*](https://oreilly.com/about/contact.html)

我们为本书有一个网页，其中列出了勘误表、示例和任何其他附加信息。您可以通过[*https://oreil.ly/learning-langchain*](https://oreil.ly/learning-langchain)访问此页面。

欲了解我们书籍和课程的新闻和信息，请访问[*https://oreilly.com*](https://oreilly.com)。

在 LinkedIn 上找到我们：[*https://linkedin.com/company/oreilly-media*](https://linkedin.com/company/oreilly-media)。

在 YouTube 上关注我们：[*https://youtube.com/oreillymedia*](https://youtube.com/oreillymedia)。

# 致谢

我们想对审稿人——Rajat Kant Goel、Douglas Bailley、Tom Taulli、Gourav Bais 和 Jacob Lee——表示感谢，他们为改进本书提供了宝贵的技术反馈。

^(1) Tom B. Brown 等人，[“Language Models Are Few-Shot Learners”](https://oreil.ly/1qoM6)，arXiv，2020 年 7 月 22 日。

^(2) 张翔等人，[“当你的问题不是英文时不要相信 ChatGPT：关于多语言能力和 LLMs 类型的研究”](https://oreil.ly/u5Cy1)，2023 年自然语言处理实证方法会议论文集，2023 年 12 月 6 日至 10 日。

^(3) 更多信息，请参阅 Ashish Vaswani 等人，[“Attention Is All You Need"](https://oreil.ly/Frtul)，arXiv，2017 年 6 月 12 日。

^(4) 刘瑞等人 [“Mind Your Step (by Step): Chain-of-Thought Can Reduce Performance on Tasks Where Thinking Makes Humans Worse”](https://oreil.ly/UHFp9)，arXiv，2024 年 11 月 8 日。
