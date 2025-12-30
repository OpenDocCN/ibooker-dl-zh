# 第一章. 提示的五个原则

*提示工程* 是一个发现可靠产生有用或期望结果提示的过程。

*提示* 是你在与像 ChatGPT 或 Midjourney 这样的 AI 模型交互时提供的输入，通常是文本。提示作为一组指令，模型使用这些指令来预测期望的响应：来自 *大型语言模型*（LLMs）如 [ChatGPT](https://chat.openai.com) 的文本，或来自 *扩散模型* 如 [Midjourney](https://www.midjourney.com) 的图像。

这里是一个产品名称生成器提示输入的简单示例（灵感来源于 [OpenAI 的示例](https://oreil.ly/Fc8cq)），以及 ChatGPT 的结果输出。

输入：

```py
Can I have a list of product names for a pair of shoes that
can fit any foot size?
```

输出：

```py
Certainly! Here's a list of product names for a pair of shoes
that can fit any foot size:

UniFit SoleStrides
FlexiSize All-Fit Shoes
OmniFit FootFlex
...
AnyStep AdaptiWalkers
FlexiRange UniversalSteps
PerfectGlide MultiFit Shoes
```

这对于原始提示来说已经是一个了不起的响应，它感觉像是魔法，因为我们几乎不费吹灰之力就达到了这里。随着最先进模型的改进，你第一次尝试就能得到 *足够好* 的结果的几率会增加。对于任何与 AI 的临时互动，你不会再次执行相同的任务，原始方法就足够了。

然而，如果你计划将这个提示投入生产，你会从投入更多工作使其正确中受益。错误会花费你金钱，这是基于 OpenAI 根据提示和响应的长度收取的费用，以及修复错误所花费的时间。如果你正在构建一个拥有数千用户的名称生成器产品，有一些明显的问题你想要尝试解决：

模糊的方向

你没有向 AI 说明你想要的名称风格或它应该具有的属性。你想要一个单词还是组合？单词可以随意创造，还是它们必须是真正的英语？你想要 AI 模仿一个以出色的产品名称而闻名的你钦佩的人吗？

未格式化的输出

你会逐行收到一个分隔的名字列表，长度不固定。当你多次运行这个提示时，有时它会返回一个带编号的列表，并且经常在开头有文本，这使得程序解析变得困难。

缺少的示例

你没有给出任何关于 *好* 名称的示例。它是使用其训练数据的平均值自动完成的，即整个互联网（及其固有的偏见），但这正是你想要的吗？理想情况下，你会给它提供成功的名称示例，行业中的常见名称，或者甚至只是你喜欢的其他名称。

有限的评估

你没有一致或可扩展的方式来定义哪些名称是好是坏，所以你必须手动审查每个响应。如果你可以建立评分系统或其他形式的测量，你可以优化提示以获得更好的结果，并确定它失败了多少次。

没有任务划分

你在这里对单个提示的要求很多：产品命名中有许多因素需要考虑，这个重要的任务被天真地一次性外包给了 AI，没有任何任务专业化或对它如何处理这个任务的了解。

解决这些问题是我们在这本书中使用的核心原则的基础。有许许多多不同的方式可以要求一个 AI 模型完成相同的任务，即使是微小的变化也可能产生很大的影响。LLMs 通过持续预测下一个标记（大约是四分之三的单词），从你的提示中开始，来工作。每个新的标记都是基于其出现的概率来选择的，其中包含一定的随机性（由*温度*参数控制）。如图 1-1 所示，单词*鞋子*在名字*AnyFit*之后出现的概率较低（0.88%），而更可预测的回应会是*运动型*（72.35%）。

![pega 0101](img/pega_0101.png)

###### 图 1-1\. 响应如何分解成标记

LLMs 是在互联网的几乎所有文本上训练的，然后进一步微调以提供有用的响应。平均提示将返回平均响应，这导致一些人当他们的结果没有达到预期时感到失望。你放在提示中的内容会改变每个生成单词的概率，因此这对你得到的结果有很大影响。这些模型已经看到了人类产生的最好和最坏的东西，如果你知道正确的方式去问，它们几乎可以模仿任何东西。OpenAI 根据提示和响应中使用的[标记数量](https://openai.com/pricing)来收费，因此提示工程师需要通过优化提示以成本、质量和可靠性来确保这些标记的价值。

这里是应用了几个提示工程技术的相同示例。我们以史蒂夫·乔布斯的方式请求名字，说明我们想要一个以逗号分隔的列表，并提供了一些任务完成得好的示例。

输入：

```py
Brainstorm a list of product names for a shoe that fits any
foot size, in the style of Steve Jobs.

Return the results as a comma-separated list, in this format:
Product description: A shoe that fits any foot size
Product names: [list of 3 product names]

## Examples
Product description: A refrigerator that dispenses beer
Product names: iBarFridge, iFridgeBeer, iDrinkBeerFridge

Product description: A watch that can tell accurate time in
space
Product names: iNaut, iSpace, iTime

Product description: A home milkshake maker
Product names: iShake, iSmoothie, iShake Mini
```

输出：

```py
Product description: A shoe that fits any foot size
Product names: iFitFoot, iPerfectFit, iShoeSize
```

虽然没有任何提示是完美的，但这个提示被优化了，可以可靠地以正确的格式提供坚实的商品名称。你的产品名称生成器的用户可以选择除史蒂夫·乔布斯之外的其他人，如果需要，他们可以更改响应格式，并且这个提示的输出可以成为另一个输入。最后，你可以根据用户反馈定期更新你在提示中使用的示例，使你的系统随着时间的推移变得更智能。

# 提示五原则概述

优化此提示的过程遵循提示的“五个原则”，我们将使用本章节的其余部分来分析这些原则，并在整本书中回顾。它们与我们在讨论天真文本提示时提出的五个问题完全对应。您将在本书的其余部分找到对这些原则的引用，以帮助您了解它们在实际中的应用。提示的五个原则如下：

指明方向

详细描述期望的风格，或引用相关角色

指定格式

定义要遵循的规则和所需响应的结构

提供示例

插入一组多样化的测试案例，其中任务已完成

评估质量

识别错误并评估响应，测试驱动性能的因素。

分工

将任务拆分为多个步骤，以实现复杂目标

这些原则不是短暂的提示或技巧，而是普遍接受的惯例，适用于与任何级别的智能（生物或人工）一起工作。这些原则是模型无关的，并且无论您使用哪种生成文本或图像模型，都应该有助于改进您的提示。我们于 2022 年 7 月首次在博客文章“提示工程：从文字到艺术和文案”中发布这些原则[“Prompt Engineering: From Words to Art and Copy”](https://oreil.ly/RYYiV)，并且它们经受了时间的考验，包括与 OpenAI 自己的[提示工程指南](https://oreil.ly/dF8q-)非常接近，该指南一年后发布。任何与生成 AI 模型紧密合作的人可能会汇聚到解决常见问题的类似策略，本书中您将看到数百个示例，说明它们如何有助于改进您的提示。

我们为文本和图像生成提供了可下载的单页文档，您可以在应用这些原则时将其作为清单使用。这些文档是为我们流行的 Udemy 课程[AI 训练营的完整提示工程](https://oreil.ly/V40zg)（70,000+ 学生）制作的，该课程基于相同的原则，但材料与本书不同。

+   [文本生成单页文档](https://oreil.ly/VCcgy)

+   [图像生成单页文档](https://oreil.ly/q7wQF)

为了展示这些原则同样适用于提示图像模型，让我们使用以下示例，并解释如何将提示的五个原则应用于这个特定场景。将整个输入提示复制并粘贴到 Discord 的 Midjourney Bot 中，包括图像链接，在输入`**/imagine**`以触发提示框出现之后（需要免费[Discord](https://discord.com)账户和付费[Midjourney](https://www.midjourney.com)账户）。

输入：

```py
https://s.mj.run/TKAsyhNiKmc stock photo of business meeting
of 4 people watching on white MacBook on top of glass-top
table, Panasonic, DC-GH5
```

图 1-2 显示了输出。

![pega 0102](img/pega_0102.png)

###### 图 1-2. 商务会议的股票照片

这个提示利用了 Midjourney 能够以一个基础图像为例的能力，通过将图像上传到 Discord，然后将 URL 复制粘贴到提示中（[*https://s.mj.run/TKAsyhNiKmc*](https://s.mj.run/TKAsyhNiKmc)），这里使用了 Unsplash 的免费图片（图 1-3）。如果你在提示中遇到错误，请尝试自己上传图像并查看 Midjourney 的文档（[Midjourney 的文档](https://oreil.ly/UTxpX)）以了解任何格式更改。

![pega 0103](img/pega_0103.png)

###### 图 1-3. 由 Mimi Thian 在[Unsplash](https://oreil.ly/J4Hkr)拍摄的照片

让我们比较这个精心设计的提示与如果你以最简单的方式天真地要求一张股票照片时从 Midjourney 得到的输出。图 1-4 显示了一个没有提示工程得到的例子，这张图像比通常预期的股票照片风格更暗、更具有风格化。

输入：

```py
people in a business meeting
```

图 1-4 显示了输出。

尽管在 Midjourney v5 及以后的版本中这个问题不太突出，但社区反馈机制（当用户选择一个图像进行放大到更高分辨率时，这个选择可能会用于训练模型）据报道已经使模型偏向于*幻想*美学，这不太适合股票照片的使用场景。Midjourney 的早期采用者来自数字艺术界，自然倾向于幻想和科幻风格，即使这种美学不适合，这些风格也可以在模型的结果中反映出来。

![pega 0104](img/pega_0104.png)

###### 图 1-4. 商务会议中的人们

在这本书中使用的示例将与 ChatGPT Plus（GPT-4）作为文本模型以及 Midjourney v6 或 Stable Diffusion XL 作为图像模型兼容，尽管我们会指定是否重要。这些基础模型是当前最先进的技术，擅长各种任务。原则旨在尽可能保证未来兼容性，所以如果你在 GPT-5、Midjourney v7 或 Stable Diffusion XXL 发布时阅读这本书，或者如果你使用的是其他供应商如 Google，这里学到的所有内容仍然应该是有用的。

# 1. 给出方向

之前讨论的简单文本提示的一个问题是，它没有向 AI 说明你想要的*类型*的产品名称。在某种程度上，给产品命名是一项主观的活动，如果不给 AI 一个你喜欢的名称的想法，它猜对的概率很低。

顺便说一句，即使没有良好的*概要*，人类也很难完成这项任务，这也是为什么创意和品牌代理机构需要从客户那里获得任何任务的详细说明。

###### 小贴士

虽然这不是完美的映射，但可以想象人类可能需要什么样的上下文来完成这项任务，并尝试将其包括在提示中。

在我们给出的示例提示中，我们通过使用 *角色扮演* 来给出方向，在那个例子中是模仿史蒂夫·乔布斯（Steve Jobs）的风格，他因标志性产品命名而闻名。如果你改变提示的这个方面，使其指向训练数据中其他著名人物（以及匹配正确的风格），你会得到截然不同的结果。

输入：

```py
Brainstorm a list of product names for a shoe that fits any
foot size, in the style of Elon Musk.

Return the results as a comma-separated list, in this format:
Product description: A shoe that fits any foot size
Product names: [list of 3 product names]

## Examples
Product description: A refrigerator that dispenses beer
Product names: Teslacool, FridgeX, Neuradrink

Product description: A watch that can tell accurate time in
space
Product names: WatchX, iSpace, The Watch Company

Product description: A home milkshake maker
Product names: ShakeX, The Milkshake Company, HyperShake
```

输出：

```py
ShoeX, MarsFit, The Shoe Company
```

也有一些规则或最佳实践是你应该遵循的，这些可以包含在提示中作为上下文，以引导 AI 向一个有效的名字发展。这种技术有时被称为 *预热* 或 *内部检索*，它简单但有效 ([刘等，2021](https://oreil.ly/1lqzK))。从询问最佳实践建议开始对话，然后要求它遵循自己的建议，这会有很大帮助。实际上，你是在用它来生成自己的方向。

输入：

```py
Please give me 5 tips for naming products based on expert
industry advice.
```

输出：

```py
Certainly! Naming a product is crucial as it often gives the
first impression to potential customers. Here are five expert
tips to guide you in naming products:

1\. **Simplicity & Memorability**: Opt for names that are
short, easy to spell, and memorable. Long, complicated names
can be difficult for customers to recall. Remember, the
easier the name is to remember, the more likely it is that
consumers will think of your product when making a purchase
decision.

...

5\. **Positive Connotations**: Names that evoke positive
emotions or imagery can create a favorable perception of
your product. Avoid names that have negative or
controversial associations in any of the primary languages
or cultures where you intend to do business. It's often
worthwhile to conduct linguistic checks and market testing
to ensure there aren't unintended meanings.

Lastly, once you've shortlisted a few names, it's a good
idea to run them by diverse focus groups or conduct surveys
to gauge reactions. Also, ensure that the name you choose is
available for trademarking and doesn't infringe on existing
trademarks.
```

然后在同一个聊天窗口中，当模型有过去给出的建议的上下文时，你提出你想要完成的任务的初始提示。

输入：

```py
Using this advice, can I have a list of product names for a
pair of shoes that can fit any foot size?
```

输出：

```py
Certainly! Based on the criteria given:

1\. **Simplicity & Memorability**
2\. **Relevance & Descriptiveness**
3\. **Differentiation & Uniqueness**
4\. **Adaptable & Scalable**
5\. **Positive Connotations**

Here's a list of potential product names for shoes that can
fit any foot size:

1\. **FitAll Walkers**: A simple and memorable name that
instantly suggests a universal fit.

...

10\. **TrueFit Universals**: Emphasizing genuine comfort and
fit for all.

Before finalizing any name, remember to check for trademarks
to ensure there's no conflict with existing products or
brands.
```

在我们的经验中，另一个富有成效的策略是采取针对你想要完成的任务的最好建议，并将其插入到提示中。例如，你可以采用 [Brandwatch 的 5 条产品命名黄金法则](https://oreil.ly/3bWjz) 或其他你找到的受信任的外部资源，并将其作为上下文插入到提示中。这将显著增加提示的长度，这会花费更多的钱（当作为开发者使用 API 时），但如果响应的质量有所提高，这可能值得这种权衡。

输入：

```py
Can I have a list of product names for a pair of shoes that
can fit any foot size? Please follow the five rules below:

1\. It should be readable and writable.
If your product name is hard to pronounce, people won’t talk about it,
and if they can’t write it down (and spell it correctly!)
when they hear it, how do you expect them to Google it?

Keep it simple and don’t go with any wacky spellings just for the sake of it.

...

5\. It should evoke an emotion, feeling, or idea.
Your product name should tie back into what your product is, what the
feeling you want people to have when experiencing your product is, and/or
what idea are you trying to get across.

It should be emotive and inspiring.
```

输出：

```py
1\. FlexFit Footwear
2\. OmniSize Sneakers
3\. AdaptStride Shoes
...
18\. OmniComfort Kicks
19\. FlexSize Footwear
20\. Boundless Soles
```

提供方向的方法还有很多。在图像生成示例中，通过指定商务会议是在一个玻璃桌周围举行的来给出方向。如果你只改变这个细节，你可以得到一个完全不同的图像，如图 图 1-5 所详细说明的那样。

输入：

```py
https://s.mj.run/TKAsyhNiKmc stock photo of business meeting
of four people gathered around a campfire outdoors in the
woods, Panasonic, DC-GH5
```

图 1-5 展示了输出结果。

![pega 0105](img/pega_0105.png)

###### 图 1-5\. 丛林中的商务会议股票照片

角色扮演对于图像生成也很重要，你可以通过提供要模仿的艺术家或艺术风格的名字来给 Midjourney 指明方向。在人工智能艺术界中，梵高是一个非常重要的艺术家，他以其大胆、戏剧性的笔触和生动的色彩运用而闻名。看看当你将他的名字包含在提示中时会发生什么，如图 图 1-6 所示。

输入：

```py
people in a business meeting, by Van Gogh
```

图 1-6 展示了输出结果。

![pega 0106](img/pega_0106.png)

###### 图 1-6\. 梵高风格的商务会议场景

要使最后一个提示生效，你需要删除很多其他指导。例如，去掉基础图像和单词“股票照片”，以及相机“松下，DC-GH5”，有助于引入梵高的风格。你可能会遇到的问题是，通常过多的指导会导致模型快速达到它无法解决的冲突组合。如果你的提示过于具体，训练数据中可能没有足够的样本来生成符合所有你标准的图像。在这种情况下，你应该选择哪个元素更重要（在这种情况下，是梵高），并据此做出决定。

指导是使用最广泛和最普遍的原则之一。它可以采取简单地使用正确的描述性词语来阐明你的意图，或者模仿相关商业名人的形象。虽然过多的指导可能会限制模型的创造力，但指导不足是更常见的问题。

# 2. 指定格式

AI 模型是通用的翻译器。这不仅意味着从法语翻译成英语，或从乌尔都语翻译成克林贡语，还包括在数据结构之间，如从 JSON 到 YAML，或从自然语言到 Python 代码之间的翻译。这些模型能够以几乎任何格式返回响应，因此提示工程的一个重要部分是找到指定你想要的响应格式的方法。

不时你会发现，相同的提示会返回不同的格式，例如，数字列表而不是逗号分隔列表。大多数时候这并不是什么大问题，因为大多数提示都是一次性的，并且是在 ChatGPT 或 Midjourney 中输入的。然而，当你将 AI 工具集成到生产软件中时，偶尔的格式变化可能会导致各种错误。

就像与人类合作一样，通过提前指定你期望的响应格式，你可以避免浪费精力。对于文本生成模型，输出 JSON 而不是简单的有序列表通常很有帮助，因为这是 API 响应的通用格式，这使得解析和查找错误更加简单，同时也可以用来渲染应用程序的前端 HTML。YAML 也是另一个流行的选择，因为它强制执行可解析的结构，同时仍然简单且易于阅读。

在你给出的原始提示中，你通过提供的示例和提示末尾的冒号来指示它应该直接完成列表。要将格式更改为 JSON，你需要更新两者，并留下 JSON 未完成，这样 GPT-4 就会知道完成它。

输入：

```py
Return a comma-separated list of product names in JSON for
"A pair of shoes that can fit any foot size.".
Return only JSON.

Examples:
[{
		"Product description": "A home milkshake maker.",
		"Product names": ["HomeShaker", "Fit Shaker",
		"QuickShake", "Shake Maker"]
	},
	{
		"Product description": "A watch that can tell
		accurate time in space.",
		"Product names": ["AstroTime", "SpaceGuard",
		"Orbit-Accurate", "EliptoTime"]}
]
```

输出：

```py
[
	{
		"Product description": "A pair of shoes that can \
		fit any foot size.",
		"Product names": ["FlexFit Footwear", "OneSize Step",
		"Adapt-a-Shoe", "Universal Walker"]
	}
]
```

我们得到的输出是包含产品名称的完整 JSON。这可以随后被解析并用于程序化，在应用程序或本地脚本中使用。从这个点开始，使用 Python 标准 *json* 库之类的 JSON 解析器检查格式错误也很容易，因为损坏的 JSON 会导致解析错误，这可以作为重试提示或继续之前进行调查的触发器。如果你仍然没有得到正确的格式，在提示的开始或结束处指定，或者在使用聊天模型时在系统消息中指定可能会有所帮助：“你是一个只以 JSON 响应的有用助手”，或者如果模型参数中可用，指定[JSON 输出](https://oreil.ly/E7wua)（这在[Llama 模型](https://oreil.ly/yU27T)中被称为 *grammars*）。

###### 小贴士

如果你不熟悉 JSON，W3Schools [有一个很好的介绍](https://oreil.ly/Xakgc)。

对于图像生成模型来说，格式非常重要，因为修改图像的机会几乎是无限的。它们包括像“股票照片”、“插图”和“油画”这样的明显格式，到更不寻常的格式，如“行车记录仪视频”、“冰雕”或“在 Minecraft 中”（见图 1-7）。

输入：

```py
business meeting of four people watching on MacBook on top of
table, in Minecraft
```

图 1-7 展示了输出结果。

![pega 0107](img/pega_0107.png)

###### 图 1-7\. Minecraft 中的商务会议

在设置格式时，通常需要移除可能与指定格式冲突的提示的其他方面。例如，如果你提供了一张股票照片的底图，结果将是股票照片和所需格式的某种组合。在一定程度上，图像生成模型可以推广到他们在训练集中之前未见过的新的场景和组合，但根据我们的经验，无关元素的层级越多，你得到不合适图像的可能性就越大。

第一原则和第二原则之间往往存在一些重叠，即给出方向和指定格式。后者是关于定义你想要的输出类型，例如 JSON 格式，或股票照片的格式。前者是关于你想要的响应风格，独立于格式，例如以史蒂夫·乔布斯风格的产品名称，或梵高风格的商务会议图像。当风格和格式发生冲突时，通常最好通过删除对最终结果不那么重要的元素来解决。

# 3\. 提供示例

原始提示没有给出你认为*好的*名字的任何例子。因此，响应接近互联网的平均水平，你可以做得更好。研究人员将没有例子的提示称为*零样本*，当人工智能甚至能够零样本完成任务时，这总是一个令人愉快的惊喜：这是强大模型的标志。如果你不提供任何例子，你是在索取很多而回报很少。即使提供一个例子（*单样本*）也能大大帮助，研究人员通常测试模型在多个例子（*少样本*）下的表现。其中一项著名的研究是 GPT-3 论文[“Language Models are Few-Shot Learners”](https://oreil.ly/KW5PS)，其结果在图 1-8 中展示，显示添加一个例子与提示结合可以提高某些任务的准确性，从 10%提高到近 50%！

![pega 0108](img/pega_0108.png)

###### 图 1-8\. 上下文中的示例数量

当向同事简要介绍一项新任务或培训初级员工时，自然地会包括一些该任务之前做得好的例子。与人工智能合作也是如此，提示的强大之处往往取决于所使用的例子。提供例子有时比试图解释你为什么喜欢这些例子要容易，因此当你在尝试完成的任务的主题领域不是领域专家时，这种技术最为有效。你可以在提示中放入的文本量是有限的（截至写作时，Midjourney 上大约有 6,000 个字符，ChatGPT 免费版大约有 32,000 个字符），因此提示工程的大部分工作涉及选择和插入多样且富有教育意义的例子。

在可靠性和创造力之间有一个权衡：超过三个到五个例子，你的结果将变得更加可靠，但会牺牲创造力。你提供的例子越多，它们之间的多样性越少，响应就越有可能与你的例子相匹配。如果你将前一个提示中的所有例子都改为动物名字，这将强烈影响响应，可靠地只返回包含动物名字的名称。

输入：

```py
Brainstorm a list of product names for a shoe that fits any
foot size.

Return the results as a comma-separated list, in this format:
Product description: A shoe that fits any foot size
Product names: [list of 3 product names]

## Examples:
Product description: A home milkshake maker.
Product names: Fast Panda, Healthy Bear, Compact Koala

Product description: A watch that can tell accurate time in
space.
Product names: AstroLamb, Space Bear, Eagle Orbit

Product description: A refrigerator that dispenses beer
Product names: BearFridge, Cool Cat, PenguinBox
```

输出：

```py
Product description: A shoe that fits any foot size
Product names: FlexiFox, ChameleonStep, PandaPaws
```

当然，这存在错过返回一个更适合在 AI 有限空间内发挥作用的更好名字的风险。在处理边缘情况或罕见场景时，示例的多样性和变化不足也是一个问题。包含一到三个示例很容易，并且几乎总是有积极的效果，但超过这个数量，就变得必须实验包含的示例数量以及它们之间的相似性。有证据表明（[Hsieh 等人，2023](https://oreil.ly/6Ixcw)），提供方向比提供示例更有效，而且通常收集好的示例并不简单，因此通常谨慎地首先尝试提供方向的原则。

在图像生成领域，提供示例通常以在提示中提供基础图像的形式出现，在开源[Stable Diffusion](https://oreil.ly/huVRu)社区中称为*img2img*。根据使用的图像生成模型，这些图像可以作为模型生成的基础，这极大地影响了结果。你可以保持提示的所有内容不变，但用提供的不同基础图像替换，以产生截然不同的效果，如图 1-9 所示。

输入：

```py
stock photo of business meeting of 4 people watching on
white MacBook on top of glass-top table, Panasonic, DC-GH5
```

图 1-9 展示了输出结果。

![pega 0109](img/pega_0109.png)

###### 图 1-9. 四人商务会议的股票照片

在这种情况下，通过替换图 1-10 中显示的图片，该图片也来自 Unsplash，你可以看到模型被拉向了不同的方向，并且现在包含了白板和便利贴。

###### 警告

这些示例展示了图像生成模型的能力，但在上传基础图像用于提示时，我们需要谨慎行事。请检查你计划上传并用于提示的基础图像的许可，并避免使用明显受版权保护的照片。这样做可能会让你陷入法律纠纷，并且违反了所有主要图像生成模型提供商的服务条款。

![pega 0110](img/pega_0110.png)

###### 图 1-10. 由 Jason Goodman 在[Unsplash](https://oreil.ly/ZbzZy)拍摄的照片

# 4. 评估质量

到目前为止，还没有反馈回路来判断你响应的质量，除了运行提示并查看结果的基本试错法，这被称为[*盲提示*](https://oreil.ly/42rSz)。当你的提示仅用于临时执行单一任务且很少再次访问时，这是可以接受的。然而，当你多次重用相同的提示或构建依赖于提示的生产应用程序时，你需要对结果进行更严格的测量。

评估性能有多种方式，这主要取决于您希望完成哪些任务。当一个新的 AI 模型发布时，重点往往在于模型在*评估*（评估）上的表现如何，这是一个标准化的问题集，具有预定义的答案或评分标准，用于测试模型间的性能。不同的模型在不同类型的任务上表现不同，不能保证之前有效的提示在新模型上也能很好地翻译。OpenAI 已经将其用于基准测试 LLM 性能的 evals 框架开源，并鼓励其他人贡献额外的评估模板。

除了标准的学术评估外，还有一些更具新闻价值的测试，例如[GPT-4 通过律师资格考试](https://oreil.ly/txhSZ)。对于更主观的任务，评估可能很困难，对于小型团队来说可能耗时或成本高昂。在某些情况下，研究人员已经转向使用更先进的模型，如 GPT-4，来评估来自不那么复杂的模型的响应，正如在[发布基于 Meta 的 Llama 开源模型的微调模型 Vicuna-13B](https://oreil.ly/NW3WX)时所做的（参见图 1-11）。

![pega 0111](img/pega_0111.png)

###### 图 1-11. Vicuna GPT-4 评估

在撰写科学论文或评估新的基础模型发布时，需要更严格的评估技术，但通常您只需比基本的试错多走一步。您可能会发现，在 Jupyter Notebook 中实现的简单点赞/踩不点赞系统可以为提示优化增加一些严谨性，而不会增加太多开销。一个常见的测试是看提供示例是否值得额外的提示长度成本，或者是否可以在提示中不提供示例也能过得去。第一步是为每个提示的多次运行获取响应并将它们存储在电子表格中，我们将在设置好环境后进行此操作。

您可以使用`pip install openai`命令安装 OpenAI Python 包。如果您遇到与此包的兼容性问题，请创建一个虚拟环境并安装我们的[*requirements.txt*](https://oreil.ly/2KDV6)（请参阅前言中的说明）。

要使用 API，您需要[创建一个 OpenAI 账户](https://oreil.ly/oGv4j)，然后[在此处获取您的 API 密钥](https://oreil.ly/oHID1)。

###### 警告

由于安全原因，不建议在脚本中硬编码 API 密钥。相反，请使用环境变量或配置文件来管理您的密钥。

一旦您有了 API 密钥，执行以下命令将其分配为环境变量至关重要，用您的实际 API 密钥值替换`api_key`：

```py
export OPENAI_API_KEY="api_key"
```

```py`` Or on Windows:    ``` 设置 OPENAI_API_KEY=api_key ```py   ````或者，如果您不想预先设置 API 密钥，那么您可以在初始化模型时手动设置密钥，或者从 *.env* 文件中加载它，使用 *[python-dotenv](https://oreil.ly/IaQjS)*。首先，使用 `pip install python-dotenv` 安装库，然后在脚本或笔记本的顶部使用以下代码加载环境变量：    ```py from dotenv import load_dotenv  load_dotenv()  # take environment variables from .env. ```    第一步是获取每个提示的多次运行响应并将它们存储在电子表格中。    输入：    ```py # Define two variants of the prompt to test zero-shot # vs few-shot prompt_A = """Product description: A pair of shoes that can fit any foot size. Seed words: adaptable, fit, omni-fit. Product names:"""  prompt_B = """Product description: A home milkshake maker. Seed words: fast, healthy, compact. Product names: HomeShaker, Fit Shaker, QuickShake, Shake Maker  Product description: A watch that can tell accurate time in space. Seed words: astronaut, space-hardened, eliptical orbit Product names: AstroTime, SpaceGuard, Orbit-Accurate, EliptoTime.  Product description: A pair of shoes that can fit any foot size. Seed words: adaptable, fit, omni-fit. Product names:"""  test_prompts = [prompt_A, prompt_B]  import pandas as pd from openai import OpenAI import os  # Set your OpenAI key as an environment variable # https://platform.openai.com/api-keys client = OpenAI(   api_key=os.environ['OPENAI_API_KEY'],  # Default )  def get_response(prompt):     response = client.chat.completions.create(         model="gpt-3.5-turbo",         messages=[             {                 "role": "system",                 "content": "You are a helpful assistant."             },             {                 "role": "user",                 "content": prompt             }         ]     )     return response.choices[0].message.content  # Iterate through the prompts and get responses responses = [] num_tests = 5  for idx, prompt in enumerate(test_prompts):     # prompt number as a letter     var_name = chr(ord('A') + idx)      for i in range(num_tests):         # Get a response from the model         response = get_response(prompt)          data = {             "variant": var_name,             "prompt": prompt,             "response": response             }         responses.append(data)  # Convert responses into a dataframe df = pd.DataFrame(responses)  # Save the dataframe as a CSV file df.to_csv("responses.csv", index=False)  print(df) ```    输出：    ```py   variant                                             prompt   \ 0       A  Product description: A pair of shoes that can ... 1       A  Product description: A pair of shoes that can ... 2       A  Product description: A pair of shoes that can ... 3       A  Product description: A pair of shoes that can ... 4       A  Product description: A pair of shoes that can ... 5       B  Product description: A home milkshake maker.\n... 6       B  Product description: A home milkshake maker.\n... 7       B  Product description: A home milkshake maker.\n... 8       B  Product description: A home milkshake maker.\n... 9       B  Product description: A home milkshake maker.\n...                                              response 0  1\. Adapt-a-Fit Shoes \n2\. Omni-Fit Footwear \n... 1  1\. OmniFit Shoes\n2\. Adapt-a-Sneaks \n3\. OneFi... 2  1\. Adapt-a-fit\n2\. Flexi-fit shoes\n3\. Omni-fe... 3  1\. Adapt-A-Sole\n2\. FitFlex\n3\. Omni-FitX\n4\. ... 4  1\. Omni-Fit Shoes\n2\. Adapt-a-Fit Shoes\n3\. An... 5  Adapt-a-Fit, Perfect Fit Shoes, OmniShoe, OneS... 6       FitAll, OmniFit Shoes, SizeLess, AdaptaShoes 7       AdaptaFit, OmniShoe, PerfectFit, AllSizeFit. 8  FitMaster, AdaptoShoe, OmniFit, AnySize Footwe... 9        Adapt-a-Shoe, PerfectFit, OmniSize, FitForm ```    在这里，我们使用 OpenAI API 生成对一组提示的模型响应，并将结果存储在数据框中，该数据框保存为 CSV 文件。这是它的工作原理：    1.  定义了两种提示变体，每种变体都包含产品描述、种子词和潜在的产品名称，但 `prompt_B` 提供了两个示例。           2.  调用导入语句以导入 Pandas 库、OpenAI 库和 os 库。           3.  `get_response` 函数接受一个提示作为输入，并从 `gpt-3.5-turbo` 模型返回一个响应。提示作为用户消息传递给模型，同时传递一个系统消息以设置模型的行为。           4.  将两种提示变体存储在 `test_prompts` 列表中。           5.  创建一个空列表 `responses` 来存储生成的响应，并将变量 `num_tests` 设置为 5。           6.  使用嵌套循环生成响应。外循环遍历每个提示，内循环为每个提示生成 `num_tests`（在这种情况下为五个）个响应。               1.  使用 `enumerate` 函数获取 `test_prompts` 中每个提示的索引和值。然后，将此索引转换为相应的 uppercase 字母（例如，0 变为 *A*，1 变为 *B*），用作变体名称。                       2.  对于每次迭代，使用当前提示调用 `get_response` 函数以从模型生成一个响应。                       3.  创建一个包含变体名称、提示和模型响应的字典，并将此字典追加到 `responses` 列表中。                   7.  生成所有响应后，将 `responses` 列表（现在是一个字典列表）转换为 Pandas DataFrame。           8.  然后使用 Pandas 内置的 `to_csv` 函数将此数据框保存到 CSV 文件中，使文件为 *responses.csv*，使用 `index=False` 以便不写入行索引。           9.  最后，将数据框打印到控制台。              在电子表格中拥有这些响应已经很有用，因为您可以在打印的响应中立即看到 `prompt_A`（零样本）在前五行中给出一个编号列表，而 `prompt_B`（少样本）在最后五行中倾向于输出逗号分隔的行内列表的所需格式。下一步是对每个响应进行评分，这最好在盲选和随机的情况下进行，以避免偏向某个提示。    输入：    ```py import ipywidgets as widgets from IPython.display import display import pandas as pd  # load the responses.csv file df = pd.read_csv("responses.csv")  # Shuffle the dataframe df = df.sample(frac=1).reset_index(drop=True)  # df is your dataframe and 'response' is the column with the # text you want to test response_index = 0 # add a new column to store feedback df['feedback'] = pd.Series(dtype='str')  def on_button_clicked(b):     global response_index     #  convert thumbs up / down to 1 / 0     user_feedback = 1 if b.description == "\U0001F44D" else 0      # update the feedback column     df.at[response_index, 'feedback'] = user_feedback      response_index += 1     if response_index < len(df):         update_response()     else:         # save the feedback to a CSV file         df.to_csv("results.csv", index=False)          print("A/B testing completed. Here's the results:")         # Calculate score and num rows for each variant         summary_df = df.groupby('variant').agg(             count=('feedback', 'count'),             score=('feedback', 'mean')).reset_index()         print(summary_df)  def update_response():     new_response = df.iloc[response_index]['response']     if pd.notna(new_response):         new_response = "<p>" + new_response + "</p>"     else:         new_response = "<p>No response</p>"     response.value = new_response     count_label.value = f"Response: {response_index + 1}"     count_label.value += f"/{len(df)}"  response = widgets.HTML() count_label = widgets.Label()  update_response()  thumbs_up_button = widgets.Button(description='\U0001F44D') thumbs_up_button.on_click(on_button_clicked)  thumbs_down_button = widgets.Button(     description='\U0001F44E') thumbs_down_button.on_click(on_button_clicked)  button_box = widgets.HBox([thumbs_down_button, thumbs_up_button])  display(response, button_box, count_label) ```    输出显示在 图 1-12：  ![pega 0112](img/pega_0112.png)  ###### 图 1-12\. 点赞/踩评分系统    如果您在 Jupyter Notebook 中运行此代码，将显示一个小部件，显示每个 AI 响应，并带有点赞或踩按钮（见 图 1-12）。这提供了一个简单的界面，可以快速标记响应，且开销最小。如果您想在 Jupyter Notebook 之外做这件事，可以将点赞和踩按钮的 emoji 改为 *Y* 和 *N*，并使用内置的 `input()` 函数实现一个纯文本替换，即 iPyWidgets。    完成标记响应后，您将获得输出，它显示了每个提示的性能。    输出：    ```py A/B testing completed. Here's the results:   variant  count  score 0       A      5    0.2 1       B      5    0.6 ```    数据框被随机打乱，每个响应都是盲选的（看不到提示），因此您可以得到每个提示性能的准确情况。以下是逐步解释：    1.  导入了三个模块：`ipywidgets`、`IPython.display` 和 `pandas`。`ipywidgets` 包含用于 Jupyter Notebook 和 IPython 内核的交互式 HTML 小部件。`IPython.display` 提供用于显示各种类型输出的类，如图像、声音、显示 HTML 等。Pandas 是一个强大的数据处理库。           2.  使用 Pandas 库读取 CSV 文件 *responses.csv*，其中包含您想要测试的响应。这创建了一个名为 `df` 的 Pandas DataFrame。           3.  使用 `sample()` 函数和 `frac=1` 对 `df` 进行打乱，这意味着它使用所有行。使用 `reset_index(drop=True)` 将索引重置为标准索引 0、1、2、…、n。           4.  脚本将 `response_index` 定义为 0。这用于跟踪用户当前查看的数据框中的哪个响应。           5.  在数据框 `df` 中添加了一个名为 `feedback` 的新列，数据类型为 `str` 或字符串。           6.  接下来，脚本定义了一个函数 `on_button_clicked(b)`，该函数将在界面中的任一按钮被点击时执行。               1.  函数首先检查点击的按钮的 `description` 是否是点赞按钮 (`\U0001F44D`; ![点赞 1f44d](img/thumbs-up_1f44d.png))，并将 `user_feedback` 设置为 1，或者如果它是踩按钮 (`\U0001F44E` ![踩 1f44e](img/thumbs-down_1f44e.png))，则将其设置为 0。                       2.  然后，它将 `feedback` 列表中的 `user_feedback` 更新到当前 `response_index`。                       3.  之后，它将 `response_index` 增加，以移动到下一个响应。                       4.  如果 `response_index` 仍然小于响应的总数（即数据框的长度），则调用函数 `update_response()`。                       5.  如果没有更多的响应，则将数据框保存到新的 CSV 文件 *results.csv*，然后打印一条消息，并打印结果的摘要，按变体显示收到的反馈数量和每个变体的平均分数（平均值）。                   7.  函数 `update_response()` 从数据框获取下一个响应，将其包装在段落 HTML 标签中（如果它不为空），更新 `response` 小部件以显示新的响应，并更新 `count_label` 小部件以反映当前响应编号和响应总数。           8.  实例化了两个小部件，`response`（一个 HTML 小部件）和 `count_label`（一个标签小部件）。然后调用 `update_response()` 函数以使用第一个响应和适当的标签初始化这些小部件。           9.  创建了两个更多的小部件，`thumbs_up_button` 和 `thumbs_down_button`（都是按钮小部件），分别使用点赞和踩 emoji 作为它们的描述。两个按钮都配置为在点击时调用 `on_button_clicked()` 函数。           10.  使用 `HBox` 函数将两个按钮组合成一个水平框（`button_box`）。           11.  最后，使用来自 `IPython.display` 模块的 `display()` 函数将 `response`、`button_box` 和 `count_label` 小部件显示给用户。              这种简单的评分系统可以用来判断提示质量并处理边缘情况。通常在 10 次提示测试中，您会发现偏差，否则您可能直到开始在生产中使用它之前都不会发现。缺点是手动评分大量响应可能会很繁琐，而且您的评分可能不代表您目标受众的偏好。然而，即使是少量测试也可以揭示两种提示策略之间的巨大差异，并在达到生产之前揭示非明显问题。    在提示的迭代和测试中，可以显著缩短提示的长度，从而降低系统的成本和延迟。如果您可以找到另一个性能相同（或更好）但使用更短提示的提示，您就可以大幅度扩大您的运营规模。通常，您会在这一过程中发现复杂提示的许多元素都是完全多余的，甚至是有害的。    *点赞* 或其他手动标记的质量指标不必是唯一的评判标准。人类评估通常被认为是反馈的最准确形式。然而，手动评分大量样本可能会很繁琐且成本高昂。在许多情况下，例如数学或分类用例，可能可以建立 *ground truth*（测试用例的参考答案）以编程方式评分结果，从而可以大幅度扩大您的测试和监控工作。以下列表并不全面，因为有许多动机可以编程方式评估您的提示：    成本      使用大量令牌或仅与更昂贵的模型工作的提示可能不适合生产使用。      延迟      令牌越多，或所需的模型越大，完成任务所需的时间就越长，这可能会损害用户体验。      调用      许多 AI 系统需要多次循环调用以完成任务，这可能会严重减慢进程。      性能      实现某种形式的外部反馈系统，例如物理引擎或其他用于预测现实世界结果的模型。      分类      确定提示正确标记给定文本的频率，使用另一个 AI 模型或基于规则的标记。      推理      确定 AI 未能应用逻辑推理或数学错误的实例与参考案例。      幻觉      看看您遇到幻觉的频率，如新术语的发明，这些术语不包括在提示的上下文中。      安全性      使用安全过滤器或检测系统标记系统可能返回不安全或不希望的结果的场景。      拒绝      查找系统错误地拒绝满足合理用户请求的频率，通过标记已知的拒绝语言。      对抗      使提示能够抵御已知的 [提示注入](https://oreil.ly/KGAqe) 攻击，这些攻击可以使模型运行不希望提示而不是您编程的提示。      相似性      使用共享的单词和短语 ([BLEU 或 ROGUE](https://oreil.ly/iEGZ9)) 或向量距离（在第五章中解释）来衡量生成文本和参考文本之间的相似性。      一旦您开始对哪些示例是好的进行评分，您就可以更容易地更新用于提示的示例，从而随着时间的推移使您的系统变得更智能。这些反馈的数据也可以用于微调示例，一旦您可以 [提供几千个示例](https://oreil.ly/DZ-br)，微调就开始超越提示工程，如图 图 1-13 所示。  ![pega 0113](img/pega_0113.png)  ###### 图 1-13\. 提示值是多少数据点？    从点赞或踩到 3 点、5 点或 10 点评分系统，您可以获得对您提示质量更细粒度的反馈。还可能通过并排比较响应来确定聚合相对性能，而不是逐个查看响应。从这些中，您可以构建一个公平的跨模型比较，使用 *[Elo 评分](https://oreil.ly/TlldE)*，这在象棋中很受欢迎，由 *lmsys.org* 在 [Chatbot Arena](https://oreil.ly/P2IcU) 中使用。    对于图像生成，评估通常采用 *排列* 提示的形式，其中您输入多个方向或格式，并为每个组合生成一个图像。然后可以扫描或稍后以网格的形式排列图像，以显示提示的不同元素对最终图像的影响。    输入：    ```py {stock photo, oil painting, illustration} of business meeting of {four, eight} people watching on white MacBook on top of glass-top table ```    在 Midjourney 中，这将编译成六个不同的提示，每个提示对应于三种格式（股票照片、油画、插图）和两种人数（四、八）的组合。    输入：    ```py 1\. stock photo of business meeting of four people watching on white MacBook on top of glass-top table  2\. stock photo of business meeting of eight people watching on white MacBook on top of glass-top table  3\. oil painting of business meeting of four people watching on white MacBook on top of glass-top table  4\. oil painting of business meeting of eight people watching on white MacBook on top of glass-top table  5\. illustration of business meeting of four people watching on white MacBook on top of glass-top table  6\. illustration of business meeting of eight people watching on white MacBook on top of glass-top table ```    每个提示通常生成其自己的四个图像，这使得输出稍微难以看到。我们从每个提示中选出一个进行放大，然后将它们组合在一起在一个网格中，如图 图 1-14 所示。您会注意到模型并不总是得到正确的人数（生成式 AI 模型在数学上令人惊讶地糟糕），但它通过在右侧的照片中添加比左侧更多的人数正确地推断了一般意图。    图 1-14 显示了输出。  ![pega 0114](img/pega_0114.png)  ###### 图 1-14\. 提示排列网格    对于具有类似 Stable Diffusion 的 API 的模型，您可以更容易地操作照片并以网格格式显示它们，以便于扫描。您还可以操作图像的随机种子，以固定一种风格
