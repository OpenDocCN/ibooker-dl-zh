# 第二章\. 预训练数据

在第一章中，我们介绍了语言模型，指出了它们的优点和局限性，探讨了当前和潜在的应用场景，并提出了似乎支配该领域进展的扩展定律。为了为本书的其余部分奠定基础，在接下来的三章中，我们将详细讨论预训练 LLM 的配方以及构成它们的成分。但是等等，这本书是关于利用预训练的 LLM 来设计和构建用户应用的。为什么我们需要讨论从头开始预训练这些巨型模型的细微差别，这对于大多数机器学习从业者来说在他们的一生中都不太可能去做？

实际上，这个信息非常重要，因为预训练过程中做出的许多决策会严重影响下游性能。正如我们将在后续章节中注意到的，当你理解了训练过程时，失败模式更容易理解。就像我们欣赏在杂货店的包装上列出的成分一样，在我们将语言模型用于严重应用之前，我们希望了解构成语言模型的成分。

###### 注意

关于一些只能通过 API 访问的专有 LLM，在公共领域可用的信息不多。本书将提供尽可能多的公开信息。虽然信息不足并不意味着我们应该避免使用这些模型，但在做出最终决定选择哪个模型时，模型透明度可能是你需要考虑的因素。

# LLM 的成分

让我们从构成 LLM 的成分开始。

从广义上讲，我们有：

预训练数据：它们是在什么上训练的？

在语言建模方面，古老的计算机科学格言“垃圾输入，垃圾输出”仍然适用。在本章中，我们将探讨流行的预训练数据集，并深入了解为确保向模型提供高质量数据而采取的各种预处理步骤。我们还将展示一些工具，这些工具允许我们探测这些数据集，并了解预训练数据组成如何影响下游任务。

词汇和分词器：它们是在什么上训练的？

要在一种语言上构建模型，我们首先必须确定我们正在建模的语言的词汇，以及将文本流分解成正确的词汇单元（称为分词）的规则。（我们将把第三章专门用于讨论这些概念。）从语言学的角度来看，人类从意义承载的单词和句子处理语言。语言模型从标记处理语言。我们将探讨当两者之间存在不匹配时的下游影响。

学习目标：它被训练去做什么？

通过预训练语言模型，我们旨在赋予语言模型在语法、语义、推理等方面的通用技能，希望它能可靠地解决你抛给它的任何任务，即使它没有专门针对该任务进行训练。因此，训练目标应该足够通用，以捕捉所有这些技能。在第四章 中，我们将讨论预训练模型所训练的各种任务（学习目标）。你可能想知道 LLM 是否更适合解决与预训练模型训练任务相似的下游任务。我们将测试这个假设，并讨论各种学习目标对任务性能的影响。

架构：它的内部结构是什么？

模型的架构指的是模型的组件，它们如何连接和相互作用，以及它们如何处理输入。每种架构都有自己的归纳偏差，这是一组关于将要用于数据和任务的假设，使模型偏向于某些类型的解决方案。在第四章 中，我们将深入探讨 Transformer 架构，正如在第一章 中讨论的那样，这是目前主要使用的架构。

让我们看看这些成分如何在 图 2-1 中组合在一起。

![LLM 成分](img/dllm_0201.png)

###### 图 2-1\. 所有成分如何结合在一起形成一个 LLM

使用本章和下一章中描述的过程训练的语言模型被称为*基础模型*。最近，模型提供商通过在更小的数据集上微调基础模型来增强它，使其更符合人类的需求和偏好。一些流行的微调模式包括：

+   监督指令微调（SFT），使模型更好地遵循人类指令

+   通过人类反馈进行强化学习（RLHF），使模型更好地与人类偏好对齐

+   领域自适应或任务自适应的持续预训练，使模型更好地适应特定领域和任务

根据所进行的特定增强，得到的模型被称为*指令模型*、*聊天模型*等等。

我们将在第六章 中介绍指令和聊天模型，在第七章 中介绍领域自适应和任务自适应的预训练。

![衍生模型](img/dllm_0202.png)

###### 图 2-2\. 基础模型及其衍生模型之间的关系

# 预训练数据要求

虽然已经证明，高容量模型相对更 [样本高效](https://oreil.ly/PbN6F)，但总的来说，今天的语言模型非常样本低效，这意味着它们需要大量的例子来学习一个任务。创建如此大的带有人类标注的监督数据集是不切实际的，因此，预训练语言模型的主要手段是使用 *自监督* 学习，其中目标标签存在于你的训练输入中。

使用这种设置，几乎任何类型的文本都可以被包含在预训练数据集中，从理论上讲，任何具有一些结构的非文本信号都可以编码成文本，并作为预训练数据集的一部分。

从我们在第一章“缩放定律讨论”中讨论的内容，我们知道模型性能通过仅仅训练更长的时间和更多的数据就能提高。此外，正如在第一章“语言模型介绍”中讨论的那样，该领域的 *巩固效应* 提高了人们对单个语言模型端到端期望的期望。今天，一个模型被期望能够回答关于世界的事实性问题，运用算术和逻辑推理，编写代码，并提出创新的想法。

所有这些都意味着语言模型预训练所需的数据量是巨大的。现在，关键问题是世界上可用的文本数据是否实际上包含足够且相关的信号，这些信号是我们希望 LLMs 学习所有技能所需的。

注意，仅基于文本训练的语言模型只能访问语言形式，即构成句子如“Walter White 把披萨扔到了屋顶上。”的字符序列。要理解其含义，语言形式必须映射到作者/说话者的沟通意图。虽然研究界的一个 [部分](https://oreil.ly/3iYA2) 认为不能仅从形式中学习意义，但最近的语言模型越来越多地证明并非如此。

要获得完整的图景，语言形式需要与现实世界联系起来。在认知科学中，grounding 被定义为：

> 建立两个对话者之间成功沟通所需的互信息量的过程
> 
> Chandu 等人，[“在自然语言处理中‘grounding’的定位”](https://oreil.ly/kPyXu)

人类文本通常非常不具体，大量的沟通意图存在于文本之外，依赖于读者/听众使用他们的常识、世界知识和检测、理解情感隐含意义的能力来解释它。

###### 注意

据估计，我们从文本中理解的信息中只有大约 [12%](https://oreil.ly/jg4tW) 是在文本中明确提到的。有几种理论解释了为什么我们会这样沟通，包括 [Zipf 的最小努力原则](https://oreil.ly/UX7Nd)，它表明“人类的天性是在最少的努力下获得最大的成果。”

自然语言处理（NLP）领域已经看到了[大量工作](https://oreil.ly/PbIhT)将语言模型与现实世界联系起来。结合不同模态（如图像、视频、语音和文本）的多模态模型是一个有希望的研究方向，并且它们在未来几年可能会得到更广泛的应用。想象一下，一个模型在训练文本中看到了“披萨”，同时也获得了关于它的外观、声音和味道的信号！

但多模态模型真的有助于解决基础问题吗？我们能否仅仅通过向模型提供大量多样化的文本来实现基础的效果？这些问题尚未解决，正如这个[辩论](https://oreil.ly/oacht)所示，双方都有很好的论据。

单独在大量文本上进行训练是否能够使语言模型学习到诸如逻辑推理等技能，这也是一个悬而未决的问题。请注意，互联网上的文本包含大量描述推理步骤的内容，如定理证明、笑话解释、拼图逐步解答等。然而，推导性文本的量显然不足，这导致我们通过使用如 CoT（在第五章中进一步描述）等提示方法来弥补这一不足。有[最新证据](https://oreil.ly/Qlntp)表明，过程监督，即对问题解决过程中的每一步提供反馈，与结果监督（仅在最终解决方案上提供反馈）相比，有助于提高算术推理能力。

语言模型必须学习的一项关键技能是处理语言固有的歧义性。继上述的 Zipf 最小努力原则之后，歧义使得说话者能够在沟通中的效率和清晰度之间进行权衡。我们可以留下很多未说，因为我们已经与沟通对象建立了足够的共同基础，并相信他们能够填补空白。

早期的语言模型在建模歧义方面遇到了很多困难。我长期以来一直将这句话作为 NLP 演讲中的典范例子，以突出语言的歧义性：“WWE 的约翰·塞纳出人意料地让患有癌症的 7 岁 Make-A-Wish 儿童感到惊喜。”

尽管最先进的模型能够正确解读这个特定的句子，并且不会错误地将约翰·塞纳（John Cena）识别为邪恶的疾病传播巫师，但[最近的研究](https://oreil.ly/BrSwb)显示，即使是今天最好的模型在处理一般性的歧义时仍然存在困难。是否仅仅通过扩大模型和数据规模就足以让大型语言模型（LLMs）建模歧义，这是一个悬而未决的问题。

如果我们解决所有这些缺陷的唯一选择是扩大数据集的大小，那么接下来的问题是，我们实际上是否拥有足够的数据来让 LLMs 学习这些技能。我们是否很快就会面临训练数据不足的风险？在我们领域的某些方面存在一种误解，即我们已经拥有了足够的数据。然而，缺乏原始数据还不是训练模型的瓶颈。例如，有数十亿可以通过抓取或通过免费 API 访问的公开文档，但尚未纳入大多数预训练数据集，如议会程序、法院判决和大多数 SEC 文件。[“LLM 训练数据在极限情况下有多少？”](https://oreil.ly/XnmHL)由 Educating Silicon 估计了世界上存在的文本量。另一方面，确实，在足够大的规模上，自然发生的数据根本不足以喂养我们的模型。

因此，有人试图使用语言模型生成的文本，称为*合成数据*，来训练模型，尽管存在[风险](https://oreil.ly/RdzX0)，即基于 LLM 生成的数据进行训练可能会对模型产生潜在的负面影响，因为模型偏离了数据的真实分布。在本章后面，我们将了解创建用于预训练的合成数据的过程。

当然，并非所有数据都是平等的。我们可以通过高质量的数据实现更高的样本效率，从而需要更小的数据集大小。我们可以预处理数据，以过滤掉低质量数据或提高它们的质量。究竟是什么使数据成为高质量数据是一个复杂的问题，我们将在本章后面探讨。

# 流行预训练数据集

许多文本在公共领域是不可自由获取的。这包括隐藏在付费 API 和登录屏幕背后的数据，以及付费书籍和文档，其中许多甚至尚未数字化。像 Google 和 OpenAI 这样的大公司可以负担得起购买这些数据；例如，OpenAI 已经与《华尔街日报》、《金融时报》和其他新闻机构达成了价值数亿美元的交易，以获取他们的数据。特定领域的文本通常是专有的，并且仅对大型现有企业（例如，Bloomberg 在训练[BloombergGPT](https://oreil.ly/87r4j)时部分使用了其专有的金融数据）。然而，即使是最大公司训练的模型，其训练数据中也有相当一部分来自公开数据源。

接下来，我们将介绍一些最流行的通用预训练数据集，这些数据集被用来训练 LLMs。虽然这不是一个详尽无遗的列表，但大多数 LLMs，包括闭源 LLMs，至少有它们训练数据的大多数子集来自这些来源。我们将把特定领域（针对特定领域，如社交媒体、金融、生物医学等）数据集的讨论推迟到第七章。

###### 小贴士

大多数通用 LLM 都被训练成多面手——能够解决来自各个领域的任务。如果你的用例数据域包含在预训练数据集中，与未包含这些数据集的模型相比，在这些数据集上训练的模型在下游任务上的相对性能可能有所提高，即使预训练数据是无标签的。这意味着，如果你打算在特定领域使用 LLM 进行特定、明确的用例，领域特定模型可能会证明是有希望的。你还可以在你的领域数据上执行*持续领域自适应*或*任务自适应预训练*，以利用这一现象。这将在第七章（ch07.html#ch07）中详细讨论。

下面是一些通用语言模型常用的数据源示例：

Common Crawl/C4

互联网是公开可获取的文本数据最大的来源，因此构成了预训练数据集的一个重要比例。[Common Crawl](https://oreil.ly/dhBvu) 是一个非营利组织，它创建并发布所有网络爬取数据的快照，每月更新一次。然而，正如人们可以想象的那样，这是一个极其粗略的数据集，在使用之前需要显著地进行清理。谷歌在将 2019 年 Common Crawl 快照应用一系列预处理和过滤步骤后，准备了 C4（Colossal Clean Crawled Corpus），一个 750GB 的英语语言数据集，并发布了相应的代码。[Dodge 等人](https://oreil.ly/bxmVR) 使用这个脚本重新生成了 C4，并将其公开。C4 已被用于训练包括 T5 系列中所有模型在内的几个知名 LLM。

The Pile

[The Pile](https://oreil.ly/7UAcY) 是来自 Eleuther AI 的 825GB 数据集，它专注于从更多样化的来源发布数据集。数据的多样性很重要，因为预训练中的领域内无标签数据有助于该领域的下游性能，多样化的数据集也使得模型能够泛化到之前未见过的任务和领域。为此，The Pile 的数据不仅来自 Common Crawl，还包括 PubMed Central、arXiv、GitHub、FreeLaw 项目、Stack Exchange、美国专利商标局、Ubuntu IRC、HackerNews、YouTube、PhilPapers、NIH ExPorter、Project Gutenberg 和维基百科等。The Pile 及其子集已被选为训练几个 LLM 的数据源，包括[Llama](https://oreil.ly/_8eOD)。

WebText/OpenWebText/OpenWebText2

这些指的是网络文本的一个子集，仅限于 Reddit 上代表至少有三个*karma*（用户点赞和踩的绝对差值）的外链网页。假设是群众的智慧将只允许高质量链接浮现，这些链接包含人们实际上感兴趣的信息。在训练了这些数据的模型包括 GPT-2 和 GPT-3。

维基百科

维基百科在几乎所有通用型 LLM 的训练中都扮演着重要角色。维基百科的完整存档包含了为模型提供事实知识的宝贵百科全书文本。维基百科的编辑系统确保文本遵循高度结构化的格式。然而，从风格上讲，文本是正式的，因此维基百科本身不足以训练一个基本的语言模型，需要与包含不同写作风格的数据源相结合。

BooksCorpus/BooksCorpus2

可能是所有预训练数据集中历史影响力最大的一个，这个数据集是 BERT、RoBERTa、GPT-2/3 等知名模型的训练语料库的一部分。BooksCorpus 包含超过 7,000 本由未发表作者撰写的免费、主要是小说书籍。原始数据集中 26%的书籍属于浪漫小说类别。BooksCorpus 的副本在 The Pile 中以 BooksCorpus2 的形式存在。

FineWeb

到本书写作时，[FineWeb](https://oreil.ly/1GyZd)是世界上最大的公开可用的预训练数据集。由 Hugging Face 发布，FineWeb 包含 1.5 万亿个标记，并从经过严格清洗和过滤的 96 个 Common Crawl 快照中提取而来。Hugging Face 还发布了[FineWeb-Edu](https://oreil.ly/8XHH-)，这是 FineWeb 的一个子集，由教育数据组成，这对于 LLM 通过标准化测试和流行基准至关重要。

表 2-1 列出了一些最常用的数据集，包括它们的大小、发布年份和访问方式。

表 2-1\. 常用预训练数据集

| Name | 数据来源 | 大小 | 发布年份 | 公开？ | 使用此数据集的模型 |
| --- | --- | --- | --- | --- | --- |
| C4 | Common Crawl | 750GB | 2019 | 是（复制品） | T5, FLAN-T5, UL2, Llama, etc. |
| The Pile | Common Crawl, PubMed Central, Wikipedia, arXiv, Project Gutenberg, Stack Exchange, USPTO, GitHub, etc. | 825GB | 2020 | 是 | GPT-NeoX, GPT-J, Cerebras-GPT, StableLM, Pythia, etc. |
| RedPajama | Common Crawl, GitHub, Wikipedia, arXiv, Stack Exchange, etc. | 1.2T tokens | 2023 | 是 | Red Pajama-INCITE, MPT |
| BooksCorpus | 从 smashwords.com 采样 | 74M sentences | 2015 | 原始数据不再可用 | 包括 BERT、GPT 等在内的多数模型 |
| OpenWebText2 | 来自 Reddit 的外链 | 65GB | 2020 | 是 | GPT-2, GPT-3 |
| ROOTS | 大科学目录，Common Crawl，GitHub | 1.6T tokens | 2022 | 否（但可请求获得） | BLOOM |
| RefinedWeb | Common Crawl | 5T tokens | 2023 | 是（仅 600B 子集） | Falcon |
| SlimPajama | 从 RedPajama 清洗而来 | 627B tokens | 2023 | 是 | N/A |

表格突出了这样一个事实：大多数模型都是在类似的数据源上训练的。在本章中，我们将限制我们的覆盖范围到基础模型的预训练数据集。我们将在第六章中介绍用于增强基础模型的数据集，如指令微调数据集、RLHF 数据集等。

让我们探索这些预训练数据集的内容。使用 Google Colab 笔记本或您选择的代码编辑器，加载 C4 数据集的`realnewslike`子集，该子集大约消耗 15 GB：

```py
!pip install datasets
from datasets import dataset
realnewslike = load_dataset("allenai/c4", "realnewslike",
                            streaming=True, split="train")
for i, example in enumerate(realnewslike):
    if "Iceland" in example["text"]:
        print(example)
    if i == 10000:  # Limit to 10,000 iterations for demonstration
        break
```

使用此代码，我们可以观察到冰岛在这个 C4 子集中出现的所有实例。

# 合成预训练数据

一个新兴趋势是使用大型语言模型（LLM）生成可用于预训练 LLM 的合成数据。在数据集中包含大量合成数据的 LLM 训练中的第一个成功案例之一是微软的[phi 系列模型](https://oreil.ly/eFphR)。对于 phi-1.5 模型，微软创建了 200 亿个合成数据标记，使用 20,000 个种子主题和来自现实世界网络数据集的样本作为提示。

Hugging Face 发布了[Cosmopedia](https://oreil.ly/Pdwnw)，这是一个开源的合成数据集，用于训练 SmolLM 系列模型。其种子数据包括经过精选的资源，如斯坦福课程、可汗学院和 WikiHow，以及通用网络数据。

对于精选资源，通过从可汗学院和其他来源提取课程大纲并提示 Mistral LLM 为单个部分生成详细的长篇教科书来生成合成数据。为了大规模生成多样化的数据，Hugging Face 为每个主题发布了几个相同的提示变体，例如“为儿童创建这个主题的教科书”和“为专业人士创建这个主题的教科书”。

对于通用网络数据，Hugging Face 将 RefinedWeb 数据集的一个子集聚类到超过一百个主题中。然后，LLM 被提示使用网页片段，并要求在网页所属的主题背景下生成一篇广泛的博客文章。聚类可视化可以在[Nomic Atlas](https://oreil.ly/t8R-6)中探索。

# 训练数据预处理

一旦我们收集或获取了数据，我们需要通过运行预处理管道来过滤和清理数据。数据预处理是 LLM 训练流程中最不引人注目且最不被重视的部分，但也许是最重要的。根据我的经验，在这一阶段投入更多努力和资源可以带来显著的下游性能提升。随着我们走过数据处理管道，我希望你能体会到语言文本的复杂性和处理它的难度。请注意，由于这些数据集非常庞大，任何预处理步骤都应该非常高效（理想情况下为线性时间）。

图 2-3 展示了用于生成预训练数据集的典型预处理步骤。步骤的顺序不是固定的，但某些步骤之间存在依赖关系。

![数据预处理流程](img/dllm_0203.png)

###### 图 2-3\. 数据收集和预处理流程

让我们详细地过一遍这些步骤。

## 数据过滤和清理

从 HTML 文件中提取的大多数文本都是无意义的，比如来自网站的菜单文本、模板文本和随机的网页碎片。互联网上还有大量的色情和有害/仇恨性语言。例如，以下是从 C4 数据集未清洗版本中提取的文本样本：

> 跳转到主要内容 跳转到页脚 跳转到电子邮件注册 跳转到反馈表单 我的奖励 登出 登录并赚取奖励 0 键盘控制 欢迎来到主导航。此菜单有三个级别的产品类别。使用和键在当前级别的每个类别之间导航。使用键向下导航一个级别。使用键向上导航一个级别。按键可转到所选类别页面。菜单 热门新品 新品上市 品牌联合性能在线独家快讯必备度假婚礼礼服军装趋势 9 件/33 款造型 The Edit x Express NBA Collection Express + NBA 时尚 NBA 改变游戏规则 西装夹克

你认为这篇文本对语言和任务学习有多大的帮助？

Common Crawl 的数据以原始 HTML 和网络提取文本（WET）格式提供。虽然许多数据集创建者直接使用 WET 文件，但开源组织 Eleuther AI [注意到](https://oreil.ly/hciZS)，WET 文件的质量还有很多需要改进的地方，如上所述，HTML 模板文本仍然突出。因此，为了创建 The Pile，Eleuther AI 使用了 [jusText 库](https://oreil.ly/YRFzZ) 来更可靠地从 HTML 文档中去除模板文本。

让我们通过一个示例来探索使用 jusText 的影响。在你的 Google Colab 或 Jupyter 笔记本中，尝试以下操作：

```py
!pip install justext

import requests
import justext

response =
  requests.get("https://en.wikipedia.org/wiki/Toronto_Transit_Commission")
text = justext.justext(response.content, justext.get_stoplist("English"))
for content in text:
  if content.is_boilerplate:
    print(content.text)
```

输出显示了从标准维基百科文章中过滤出的所有模板文本：

```py
Jump to content
Main menu
Main menu
Navigation
Main page
Contents
Current events
Random article
About Wikipedia
Contact us
Donate
Contribute
Help
Learn to edit
…
```

jusText 正好更积极地去除内容，但这对清理预训练数据集来说通常是可行的，因为可用的文本量很大。用于此任务的替代库包括 [Dragnet](https://oreil.ly/URvsq)、[html2text](https://oreil.ly/xk7Hc)、[inscriptis](https://oreil.ly/6-2z1)、[Newspaper](https://oreil.ly/LPXe1) 和 [Trafilatura](https://oreil.ly/zdZxj)。根据 [The Pile](https://oreil.ly/DZG7w) 的创建者，将提取流程分散到多个库中可以降低结果数据集受到其中任何一个库引入的任何偏差的影响。

在网页中去除模板文本是一项具有挑战性的任务。网页还可能包含代码块、表格和数学公式，这些都需要仔细处理。[Meta](https://oreil.ly/bXELJ) 指出，它为训练 Llama 3 构建了一个定制的 HTML 解析器来准备数据集。它还提到，Meta 保留了图像中的 *alt* 属性，它发现其中包含有用的信息，如数学内容。

大型语言模型（LLMs）也可以用于从网页中准确提取内容。然而，鉴于数据集的规模，截至本书编写时，这样做成本过高。

一旦文本被提取，文档将通过一系列数据过滤步骤。首先，应用基于启发式的基本过滤步骤。虽然不同数据集的细节不同，但以下是一些通常执行的步骤：

模板文本移除

只有以标点符号（如句号、感叹号和问号）结尾的行会被保留。这确保了来自网站的菜单文本被移除。只有包含超过特定阈值单词的行和包含超过特定阈值句子的文档会被保留。后者有助于建模长序列，这对于语言模型来说是一个重要的能力。包含“lorem ipsum…”和其他模板文本的文档会被过滤掉。

非英语文本移除

使用如 langdetect、langid、fasttext 和 pycld2 等库来检测文本的语言。例如，C4 保留由 langdetect 判断出概率大于 0.99 的英语文本。请注意，这些库也可以用来移除模板文本和网页碎片，因为它们给这些文本的英语概率较低。

搜索引擎优化（SEO）文本/垃圾邮件移除

包含大量重复字符序列的文档会被移除。包含低比例封闭类词汇的文档会被移除。英语中的封闭类词汇是功能词，如“of”、“at”、“the”和“is”。如果一个页面涉及关键词堆砌和其他 SEO 技巧，那么它们的封闭类词汇比例会较低。

恶俗/侮辱性文本移除

包含来自如“脏话、下流、淫秽或其他不良词汇列表” [“List of Dirty, Naughty, Obscene or Otherwise Bad Words”](https://oreil.ly/w3u_r) 中任何单词的文档将被移除。

如 langdetect 和 langid 等工具对于大规模快速确定文本所写语言很有帮助，但它们如何处理代码切换文本（包含多种语言，其中英语通常与本地语言交织）？

你可以试试！以下是一个 Taglish（菲律宾常见的交流模式，即菲律宾语+英语）的例子。在你的笔记本中运行以下代码：

```py
!pip install langdetect
from langdetect import detect_langs()
detect_langs("""Pag-uwi ko galing sa paaralan, sobrang pagod ako dahil sa dami
ng aking ginawa sa buong araw. Ang traffic din sa kalsada, nakaka-stress
talaga! Pero nang makarating ako sa aking tahanan, nabuhayan ako ng loob dahil
sa masarap na amoy ng ulam na inihanda ni nanay. Excited na akong kumain
kasama ang aking pamilya at i-share ang mga kwento ko tungkol sa aking mga
kaibigan, guro, at mga natutunan ko sa school. After dinner, magre-relax muna
ako habang nanonood ng TV, and then magre-review ng lessons bago matulog. Ito
ang routine ko pag-uwi mula sa school, at masaya ako na dumating sa bahay namay
naghihintay na pamilya na handang makinig at suportahan ako sa aking
pag-aaral.""")
```

输出：

```py
[tl:0.9999984631271781]
```

```py
detect_langs("""After a long day at school, pagod na pagod talaga ako. The
traffic on the way home didn't help, nakakastress na nga! But upon arriving
home, I felt a sense of relief dahil sa welcoming atmosphere and the delicious
aroma of the ulam na inihanda ni Mommy. Excited na akong mag-share ng
experiences ko today with my family during dinner, kasama ang mga kwento about
my friends, teachers, and interesting lessons sa school. After eating, it's
time for me to chill while watching some TV shows, and then review my lessons
bago ako matulog. This is my daily routine pag-uwi galing school, and I am
grateful na may loving family ako na handang makinig at supportahan ako sa
aking educational journey.""")
```

输出：

```py
[en:0.9999954357601804]
```

根据其过滤标准（英语概率应大于.99），第二段将被包括在 C4 数据集中。因此，即使声称是纯英语的数据集也经常包含其他语言的文本，导致推理过程中出现令人惊讶的多语言行为。你是否曾想过为什么一些单语模型在机器翻译方面似乎表现良好？这是一个主要原因。

langdetect 的实现方式使其在提供短序列时在识别语言方面表现不佳。例如：

```py
detect_langs('I love you too.')
```

返回

```py
[sk:0.8571379760844766, en:0.14285726700161824]
```

sk 在这里指的是斯洛伐克语。

## 选择高质量文档

并非所有数据都是平等的。高中物理教科书中的文本被认为比关于鞋类品牌的促销文本质量更高。我们可以通过几种方式来操作质量的概念，并将高质量数据与低质量数据分开。在本节中，我们将突出介绍几种这样的方法。

### 标记分布 K-L 散度

在这种方法中，那些与参考标记分布差异过大的文档会被移除。实际上，这移除了包含大量异常标记的文档。这是通过使用[库尔巴克-利布勒（K-L）散度](https://oreil.ly/gd5GH)来计算的。

### 基于分类器的方法

我们还可以构建一个用于识别高质量数据的分类器。构建基于质量分类器的一个简单方法是将正类的示例来自高质量数据源（如维基百科），而负类的示例则来自 Common Crawl 数据中的随机文档。

Meta 为其[Llama 3 模型](https://oreil.ly/O-CKF)的高质量数据提取使用了多种分类器模型。其中之一是一个[fasttext 分类模型](https://oreil.ly/EWic6)，该模型被训练以识别文本是否可能被维基百科引用。Meta 还训练了一个分类器，其训练数据由 Llama 2 生成，通过向其提供清洗过的网络文档和质量要求，并要求其判断是否满足质量要求。为了提取包含推理步骤的代码和文本，Meta 构建了能够识别它们的分类器。

图 2-4 展示了如何构建一个分类器来区分高质量和低质量数据。

![分类器过滤](img/dllm_0204.png)

###### 图 2-4\. 基于分类器的质量过滤

### 质量选择的困惑度

[困惑度](https://oreil.ly/OfycZ)，作为语言模型的一个内在评估指标，在准备预训练数据集的上下文中被用于文档过滤，特别是由[CCNet](https://oreil.ly/VF98y)的创建者所采用。困惑度衡量模型预测给定文本的能力；困惑度越低，模型越好。

就像分类器方法一样，我们从我们认为高质量的数据源（如维基百科）中选择文档作为正类。然后，我们使用[KenLM](https://oreil.ly/EU5r3)（一个促进 n-gram 语言模型训练的库）在上面训练一个 5-gram 语言模型。接下来，我们取我们想要过滤的数据集，并计算其中每个段落相对于训练语言模型的困惑度。困惑度越低，它与正类越相似。然后我们可以丢弃困惑度高的文档。

低困惑度不一定总是好事。简短、重复的文本可以具有低困惑度。请注意，写作风格会被纳入困惑度计算。如果参考语言模型是在维基百科上训练的，那么非正式风格的文档可能会得到更高的困惑度分数。因此，拥有一个更复杂的过滤策略将会是有益的。

为了解决这个问题，[BERTIN](https://oreil.ly/uI9eV) 的创造者引入了困惑度采样的概念。在困惑度采样中，它不仅仅过滤掉低困惑度文本，而是使用一种采样策略，从困惑度概率分布的中间部分进行过采样。

图 2-5 展示了困惑度采样在实际中的实现方式。

![困惑度采样](img/dllm_0205.png)

###### 图 2-5\. 困惑度采样

让我们探索一个在维基百科文本上训练的模型分配的困惑度分数。下载这个[文件](https://oreil.ly/xwYjY)。将文件放置在您的家目录后，在一个新文件中运行以下代码：

```py
from model import KenlmModel
model = KenlmModel.from_pretrained("wikipedia", "en")
model.get_perplexity("She was a shriveling bumblebee, and he was a bumbling `banshee``,` `but` `they` `accepted` `a` `position` `at` `Gringotts` `because` `of` `their` `love` `for`
`maple` `syrup``")`
```

`` `###### 注意    根据 [C4 分析](https://oreil.ly/Nzla7)，在数据集中贡献最大比例文本的互联网域名是 patents.google.com。实际上，该域名超过 10% 的文本是机器翻译的，例如来自日本的专利是从日语翻译成英语的。因此，大量的预训练数据实际上并非由人类生成！    在 LLM 的推动下，互联网预计将广泛普及 AI 生成的文本。识别文本是由人类还是 LLM 编写的，是一项非平凡的任务，并且在大规模上肯定不可行。这将如何影响未来的 LLM 性能，是一个开放的研究问题。    尽管进行了所有数据清洗步骤，但在这个规模级别上，生成的数据集仍然不会完美。例如，Eleuther AI [报告](https://oreil.ly/WEBne)称，模板句子“从以下选择中选择您想要访问的论坛”在 The Pile 中出现了 180K 次。《PRE10》```py```## 去重    到目前为止，我们已经讨论了数据提取和清洗、语言识别和品质过滤。现在让我们探讨管道中最具争议的步骤：去重。    我们知道，网络爬取的文本充满了大量重复内容。重复内容构成了训练数据集的非小部分，因此对它们的任何决定都将对后续模型产生明显的影响。    我们如何定义重复内容？我们将区分三种类型：    完全匹配      两个具有相同文本的序列是完全匹配的重复内容。它们是最容易处理的。      近似匹配      在许多情况下，存在近似的重复内容，其中文本序列除了少数字符外完全相同。有时这些序列之所以略有不同，仅是因为 HTML 文本提取伪影和其他过滤过程。      语义重复      语义上传达相同内容但使用不同措辞的重复内容。这通常被视为超出范围。      根据它们发生的粒度，重复内容也可以分类：    文档级重复      在大多数预训练数据集的准备过程中，会移除重复文档。然而，在某些数据集（如 The Pile）中，某些子集（如维基百科）被故意重复，以便模型能更频繁地看到它们。      序列级重复      这些是跨多个文档重复的文档中的行或句子。在某些情况下，它们可以被大量重复，例如服务条款文本、版权声明、网站前言等。      ###### 注意    去重是一个非常复杂的过程，通常使用 MinHash 算法执行。Cheng Hao 的这篇文档详细介绍了 Big Science 和 Big Code 开源 LLM 项目中遵循的去重过程。[Cheng Hao](https://oreil.ly/2RO9f)    去重数据有几个好处：    *   预训练数据集通常留出一小部分用于验证/测试。去重可以确保训练集和测试集之间重叠的减少/降低，这对于无偏评估至关重要。如果没有序列级去重，训练集和测试集中常见文本序列的重叠可能性很高。           *   移除重复序列可以减少训练数据集的整体大小。然而，[Lee 等人](https://oreil.ly/k5OwJ)表明，在较小的数据集上训练的模型的困惑度并未受到影响。因此，模型可以在更短的时间内训练，同时获得相同的好处。           *   去重还可以减少模型记住其训练数据的倾向。记忆与模型过拟合密切相关，并阻碍了模型泛化的能力。虽然有许多方法可以量化记忆，但我们将重点关注 *生成记忆*，即如果模型能够逐字生成一个序列，则认为它已经记住了该序列。[Lee 等人](https://oreil.ly/xpoz7)表明，在已去重序列级别的数据集上训练的模型生成的逐字训练数据减少了十倍。              ###### 提示    使用在公共数据集上训练的模型的一个优点是，您可以搜索数据集以查看模型生成的文本是否存在于数据集中。    图 2-6 展示了基本的训练数据提取攻击流程。  ![隐私攻击](img/dllm_0206.png)  ###### 图 2-6\. 对 LLM 的隐私攻击    ## 移除个人身份信息    尽管去重可以减少模型记住训练数据的可能性，但它绝不是记忆问题的万能药。即使仅在训练集中出现一次的信息也可能被记住（并泄露）。虽然训练数据中的许多内容是无害的（服务条款文本）并且可能甚至希望记住（事实信息，如加拿大的首都），但个人身份信息（PII）的记忆是一个主要问题。    让我们看看 PII 包括哪些内容。来自 [康奈尔法学院](https://oreil.ly/kN3J8) 的正式定义如下：    > 可以用来区分或追踪个人身份的信息，无论是单独使用还是与其他个人或识别信息结合使用，这些信息与特定个人相关联或可关联。    根据这个定义，当另一条信息变得公开时，非 PII 可以成为 PII，因为当它与非 PII 结合使用时，可以用来唯一识别个人。    PII 的法律定义因司法管辖区而异。例如，欧洲的 [通用数据保护条例 (GDPR)](https://oreil.ly/F2dGL) 表示：    > 应将保护扩展到任何用于直接或间接识别个人（或数据主体）的东西。这可能包括描述“个人的身体、生理、遗传、心理、商业、文化或社会身份”的特征。    大多数开源模型都是在公开可用的数据集上训练的。这些数据集可能包含 PII，但有人可能会想，“好吧，它已经公开了，所以没有必要进行隐私保护。”这种论点忽视了同意和可发现性控制的重要性。例如，我可能在博客上分享了包含我的 PII 的内容，该博客位于互联网的一个隐蔽角落，并且不容易通过搜索引擎发现，但如果它最终被添加到预训练数据集中，它突然将数据置于聚光灯下，而没有我的同意。这个概念被称为 *情境完整性*：数据应仅在共享的原始上下文中共享。    因此，理想情况下，我们希望能够在数据集中 *检测* PII，并以某种方式 *修复* 它，以便 PII 不再存在于训练数据中，或者至少不再是可记忆的。*公众人物 PII* 的存在给这个问题增加了复杂性。我们希望我们的模型能够准确地回答有关公众人物的事实性问题，例如提供他们的出生日期。公众人物的隐私期望较低，展示了透明度和开放性价值观与隐私之间的冲突。确定谁是公众人物以及他们应享有的隐私水平是一个复杂的社会技术挑战。    考虑为私密的包括姓名、地址、信用卡数据、政府 ID、医疗历史和诊断数据、电子邮件 ID、电话号码、个人所属的认同和亲和群体（宗教、种族、工会会员资格）、地理位置数据等。    攻击可以是针对性的或非针对性的。在非针对性攻击中，攻击者仅使用模型生成大量文本，然后运行成员推理攻击以确定其中最有可能被记住的文本。在针对性攻击中，攻击者试图恢复有关特定个人或一组个人的个人信息。针对性攻击更难执行，因为虽然语言模型擅长记忆，但它们在 *关联* 方面表现不佳，例如，确定电子邮件 ID 属于特定个人。    大多数预训练数据集都经历了很少或没有 PII 修复。训练 BLOOM 模型的 Big Science 项目（我是联合负责人）的隐私工作组开发了一个 PII 检测和修复管道，我们将在下面讨论。    ###### 注意    语言模型也容易受到训练数据中毒攻击。由于大量训练数据来自网络爬取的文本，不良行为者有机会影响训练集的内容。[Tramer 等人](https://oreil.ly/g_A-d) 已经表明，可以用不到 0.1% 的训练集来中毒，其效果是使训练集中的其他数据更容易泄露。    随着 LLM 越来越多地被用作搜索引擎，LLM SEO 的需求正在出现。例如，一家公司可以在其网站上以使其更有可能被选入使用困惑度过滤的预训练数据集创建过程的方式编写内容。    图 2-7 展示了典型的 PII 处理管道。  ![PII 处理管道](img/dllm_0207.png)  ###### 图 2-7\. PII 处理管道    ### PII 检测    PII 检测的任务类似于在 第一章 中介绍的 NLP 任务 NER。然而，并非所有命名实体都构成 PII。对于我们的任务，我们确定 PII 标签为 PERSON、AGE、NORP（国籍、种族、宗教、政党隶属、社会经济阶层和工会会员资格）、STREET_ADDRESS、CREDIT_CARD、GOVT_ID、EMAIL_ADDRESS、USER_ID 和 PUBLIC_FIGURE。    我们使用 PUBLIC_FIGURE 标签来识别有关公众人物的信息，因为我们不希望过滤掉它们。我们还为虚构人物分配了这个标签。    列表中的一些结构化标签（如电子邮件和政府 ID）可以使用正则表达式识别。对于其他标签，我们注释了数据集，然后可以使用这些数据集来训练基于 Transformer 的类似 NER 的模型。有趣的是，我们观察到注释者之间的高度不一致（不同的人对同一示例进行不同的注释），这突出了隐私定义的文化细微差别以及构成个人信息的内容。    这里是检测 SSN（美国社会安全号码）的 [正则表达式](https://oreil.ly/8YwG9)：    ``` ssn_pattern = r"(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]| `[``0``-``7``][``0``-``7``][``0``-``2``])``-`\ `[``0``-``9``]{``2``}``-`\ `[``0``-``9``]{``4``}``"` ```py   ````` ```py`Note that detection is not the same as validation. Not all nine-digit numbers of the form XXX-XX-XXXX are SSNs! Validation is the process of checking if a sequence of characters maps to a valid identifier. For example, the Canadian equivalent of SSN, the social insurance number (SIN) contains a checksum digit that can be used to validate it:    ``` from stdnum.ca import sin sin_pattern = re.compile(r"\d{3}[-\ ]\d{3}[-\ ]\d{3}", flags=re.X) for match in sin_pattern.findall(text):     if sin.is_valid(match):          print(match) ```py    The `is_valid()` function uses the [Luhn checksum algorithm](https://oreil.ly/i34BW) to validate if the sequence of digits maps to a valid SIN. The same algorithm is also used to validate credit cards. Here is the [regex](https://oreil.ly/6uTq-) for detecting credit card numbers:    ``` from stdnum import luhn cc_base_pattern =  r"\b \d (?:\d[ -]?){14} \d \b" cc_full_pattern = r"""4[0-9]{12}(?:[0-9]{3})? |  (?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|  2720)[0-9]{12} |  3[47][0-9]{13} |  3(?:0[0-5]|[68][0-9])[0-9]{11} |  6(?:011|5[0-9]{2})[0-9]{12} |  (?:2131|1800|35\d{3})\d{11}""" ```py    The regular expression for detecting email address is as follows:    ``` email_pattern = r"[\w\.=-]+ @ [\w\.-]+ \. [\w]{2,3}" ```py    ###### Note    Removing structured PII data while keeping the number of false positives low is hard enough, but detecting and remediating unstructured data is even harder. Due to the complexity of this task and the uncertainty about its impact on the resulting model performance, we decided to not run the Transformer model–based PII pipeline over the ROOTS dataset for training the BLOOM model.```` ```py``  ````` ```py`### PII remediation    Once PII has been detected, it can be remediated. Figure 2-8 depicts one of the remediation schemes.  ![PII Remediation Options](img/dllm_0208.png)  ###### Figure 2-8\. PII remediation options    Here is a nonexhaustive list of remediation options:    Replace with a special token      For example, a valid phone number can be replaced by the string `<phone` `number>`.      Replace with a random token of the same entity type      For example, replace the name “Clarietta Richards” with “Natasha Bridges,” or any other name.      Replace with a shuffled token      Entities detected across the dataset can be shuffled.      Remove entire document/data source      If the amount of PII detected in a single document or data source is higher than a specific threshold, it is probably best to remove it. For example, *pastebin.com* is said to contain a lot of inadvertently placed PII and is recommended to be not included in training datasets.      Each of these techniques can have a varied effect on the model’s downstream performance. How does replacing tokens affect training perplexity? Are downstream tasks like NER negatively affected when tuned on the resulting model? How does replacement by special tokens compare to replacement with random tokens? This is a relatively underexplored topic, and all these questions are still open.    [Faker](https://oreil.ly/K4QI_) is an excellent library for facilitating random token replacement. It supports random token generation for a variety of PII types including names, addresses, credit card numbers, and phone numbers. One danger in using random tokens is that the replacement process can alter the demographic distribution of the dataset, for example, if the replacement names were all or mostly Anglo-Saxon names. Faker has localization support to enable replacement with fake data from the same geography/culture. Let’s explore the library in more detail:    ``` from faker import Faker fake = Faker('en_IN')   # 印度地区 Faker.seed(0) for i in range(5):    print(fake.aadhaar_id) ```py    This code generates 12-digit fake Aadhaar IDs, which are the Indian equivalent of Social Security numbers. Note that the generated IDs are all invalid but still follow the same format. Similarly:    ``` for i in range(5):    print(fake.address) ```py    generates fake but representative addresses for the selected locale.    ###### Note    Removing PII from training datasets is only one of several solutions to prevent data leakage from models. One promising technique is [differential privacy](https://oreil.ly/TRbsf), which introduces randomness in the inputs or outputs to provide theoretical guarantees for privacy preservation. In neural networks, differential privacy is implemented using the [DP-SGD](https://oreil.ly/DVQkl) algorithm, which involves gradient clipping and noise addition at the end of each update. However, differential privacy significantly slows training, negatively affects model performance, and disproportionately impacts minority groups in the dataset in terms of model utility degradation. Apart from differential privacy, other methods include adversarial training, [model unlearning](https://oreil.ly/_AV3V), [retroactive censoring, and “memfree” decoding](https://oreil.ly/5p0z3).```` ```py``  `` `## Training Set Decontamination    Training set decontamination is a crucial data preprocessing step that helps improve LLM evaluations. A pre-training dataset is said to be contaminated if it contains data from the benchmark test sets used to evaluate its performance. Contamination can happen if the test datasets were constructed from web text, or if the dataset was uploaded on the web after creation. There are two types of contamination:^(1)    Input and label contamination      In this setting, both the questions (inputs) and answers (target labels) exist in the pre-training dataset.      Input contamination      In this setting, only the inputs are present in the pre-training dataset but not the target labels. We will describe the effects of input contamination and how we can leverage it for positive use in Chapter 7.      [OpenAI](https://oreil.ly/d7pHK) addressed training set contamination in GPT-3 by finding 13-gram overlaps between text in the test/validation set and the train set, and removing 200 characters before and after the matched texts. The n-gram matching approach is the most commonly used method for decontamination.    However, [Yang et al.](https://oreil.ly/JjtHS) note that contamination can also happen if a rephrased or translation of the benchmark data is present in the training dataset. This makes data contamination very challenging to detect and remove. Most benchmark results continue to be overstated due to this problem.    ## Data Mixtures    Pre-training datasets contain data from a wide variety of domains. The final dataset is prepared such that these domains are represented in optimal proportions. For example, Wikipedia, academic texts, and smaller subsets were [upsampled](https://oreil.ly/hpHdw) by up to three times in The Pile dataset. More involved techniques like [DoReMi](https://oreil.ly/5z9u1) and [RegMix](https://oreil.ly/VWyzt) are also used to calculate the right data mixture. Meta noted that for [Llama 3](https://oreil.ly/fMOrb), it empirically arrived at a data mixture where 50% of the tokens are about general knowledge, 25% are about math and reasoning, 17% represent code, and the remaining are non-English tokens.    ###### Note    Many pre-training datasets these days include code, even if the model is not intended for generating code. [Aryabumi et al.](https://oreil.ly/Vm0lH) have shown that including code in pre-training data significantly improves performance on downstream tasks that do not involve generating code.    Now that we have discussed all the important data collection and preprocessing steps for preparing a pre-training dataset, let’s see how individual datasets differ in terms of the preprocessing steps they have undergone.    ###### Tip    [DataTrove](https://oreil.ly/lDFm2) by Hugging Face is a full-fledged pre-training dataset preprocessing pipeline code repository. You can go through the repo to understand how the concepts introduced in the chapter are implemented at scale.    Table 2-2 provides a list of the popular pre-training datasets and the kind of preprocessing they went through.      Table 2-2\. Pretraining datasets and their preprocessing pipeline   | Name | Extraction and cleaning | Quality filtering | Deduplication | Language identification | Models trained with this dataset | | --- | --- | --- | --- | --- | --- | | C4 | Remove pages containing word in blocklist, remove code, remove short lines and pages | - | Deduplication of 3-sentence spans | langdetect | T5, FLAN-T5, UL2, Llama | | The Pile | justext library for text extraction | fasttext classifier | Document level, with MinHashLSH | pycld2 | GPT-NeoX, GPT-J, Cerebras-GPT, StableLM, Pythia | | CCNet | - | Perplexity filtering | Paragraph-level deduplication | fasttext |  | | RedPajama | CCNet pipeline | Classifier distinguishing between Wikipedia text and random C4 text | Paragraph-level deduplication (for Common Crawl) | fasttext | Red Pajama-INCITE, MPT | | CleanPajama | Low-length filter, NFC normalization | - | MinHashLSH | - | - | | RefinedWeb | URL filtering by blocklists, trafilatura library for text extraction, repetitive content removal | - | Fuzzy document-level deduplication with MinHash, exact sequence-level deduplication | fasttext | Falcon | | ROOTS | Removal of documents with low ratio of closed class words, high ratio of blocklist words, high ratio of character/word repetition | Perplexity filtering | SimHash, Suffix Array | fasttext | BLOOM |` `` ``````py ``````py`  ``` `` `# 预训练数据对下游任务的影响    给定一个 LLM 的预训练数据集，我们可以从它对下游性能的哪些假设？事实证明，模型在给定任务或输入上的性能与预训练数据集中任务或输入中显著词的频率之间存在相关性。这一现象首先由 [Razeghi 等人](https://oreil.ly/cPYej) 发现，并在 McCoy 等人 [“自回归的余烬”论文](https://oreil.ly/_O2NK) 中进行了详细研究。    McCoy 等人表明，语言模型在训练数据集中更频繁表示的任务上表现更好，而在表示较少的任务上表现较差。例如，语言模型在十进制加法方面比九进制加法表现更好。它们在按字母顺序排序方面也比按逆字母顺序排序表现更好。    类似地，McCoy 等人也表明，对于给定任务，当输出是预训练数据集中频率较高的文本时，模型的表现相对较好，而不是当文本频率较低时。这种现象也适用于输入；与低频率输入相比，模型在高频率输入上表现相对较好。    以一个句子为例：“record a be that miles, yes, hour, per fifty clocked he。”我们要求 LLM 逆序排列句子中的单词，这将导致“He clocked fifty per hour, yes, miles, that be a record，”这是一个相当不可能的序列，因为它具有奇怪的语构。    到本书写作时，GPT-4o 返回了错误的答案：“He clocked fifty miles per hour that be a record，”但你可以注意到，当输出序列的概率较高时，它的表现相对较好。    # 预训练数据集中的偏差和公平性问题    在大型语言模型的产品化过程中，出现了许多伦理问题。这些模型中存在的重大偏差和公平性问题往往会导致大量用例无法发货。在本节中，我们将讨论一些与预训练数据的收集和过滤特别相关的偏差和公平性问题。    LLM 被喂入的数据规模意味着它们不仅构建了语言模型，也构建了我们所在世界的模型。这引发了一个问题：我们是否想要以世界本来的样子来建模，还是以我们希望它成为的样子来建模。互联网充满了仇恨、暴力和侮辱性语言，经常被用作人类最恶劣冲动的一种出口。其中的文本隐式编码了对某些人群长期存在的偏见。例如，在 The Pile 中，[分析](https://oreil.ly/hu3-b)单词共现统计表明，“radical”一词与“穆斯林”一词的共现频率比与其他宗教的共现频率要高得多。    *偏差放大* 的现象使这些问题变得更加严重。已经证明，大型语言模型 [放大了](https://oreil.ly/x-ba9)其预训练数据中编码的偏差：它们以比训练数据统计所暗示的更高的比率对某些人群做出有偏见的预测。    那么，我们能否“修复”我们的训练数据，以便我们可以模拟一个编码了我们的价值观和原则的世界，下游应用将继承这些价值观和原则？在研究界中，对此存在大量争议。反对者认为，由于存在许多相互交织的偏差维度，因此很难识别和修复数据中编码的所有社会偏见。价值观并非普遍存在，模型提供商希望保持价值中立，以迎合社会的各个部分。    然而，正如 Anna Rogers 在她的 [论文](https://oreil.ly/hxU_-) 中所描述的，这个问题已经没有意义了。数据整理已经在进行中，无论我们是否喜欢，模型提供商的价值观和利益已经编码到模型中。例如，只有一小部分可用的数据被选中作为预训练集的一部分。这个选择过程并非价值中立，即使一个人可能没有明确地从这个角度思考。    维基百科是用于训练 LLM 的更受欢迎的数据集之一。虽然将其包含在预训练数据集中可能是一个不言而喻的选择，但让我们探讨其影响。维基百科是由志愿者编辑的，其中很大一部分是男性。由于确定一个主题是否足够有信誉以获得维基百科页面的是编辑，而这些编辑主要由男性组成，因此我们看到了像来自低级别联赛的男性足球运动员获得自己的页面这样的差异，而大量关于女性的传记文章则被安排删除。    类似地，具有高度影响力的 WebText 数据集来自 Reddit 的外链。Reddit 是一个以男性为主的网站，74% 的用户是男性。[74% 的用户](https://oreil.ly/i2RkB)是男性。自然地，Reddit 上发布的链接更有可能迎合男性的兴趣。    偏差也可以在数据过滤阶段引入。早些时候，我们提到关键字列表通常用于过滤掉色情材料和侮辱性文本。然而，使用天真关键字列表是一种懒惰的方法，它不仅存在有效性问题（假阴性），而且无意中 [导致](https://oreil.ly/XWBjV)过滤掉来自少数族裔社区或使用像非洲裔美国英语和西班牙裔英语这样的方言撰写的文本的积极文本。由于英语单词具有多种含义，因此某些关于母乳喂养的文档被从 C4 数据集中过滤掉。    总体而言，一个词是否具有仇恨性、侮辱性或毒性取决于社会环境、读者的意图以及预期的受众。基于关键字的方法根本无法捕捉这种细微差别。是否在预训练阶段或更下游处理这些问题更有效，是一个开放的研究领域。我们将在 第十章 中探讨可以用于下游的技术。    ###### 注意    Pythia 模型的作者通过将最后 7% 的训练标记中的男性代词替换为女性代词进行了实验，并注意到这对其下游任务产生了去偏影响。    # 总结    在本章中，我们概述了语言模型的关键组成部分：预训练数据、词汇和分词器、语言目标和模型架构。我们详细介绍了创建预训练数据集的步骤，包括语言识别、文本提取和清洗、品质过滤、去重、PII 移除和测试集净化。我们还提供了一份常用预训练数据集列表以及预处理每个数据集的步骤。在下一章中，我们将探讨语言模型的词汇和分词器：我们希望模型学习的语言。    ^(1) 来自 Dodge 等人，《“大型清洁爬取语料库案例研究”》，EMNLP 2021。` ``
