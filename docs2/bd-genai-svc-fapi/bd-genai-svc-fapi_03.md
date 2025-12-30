# 第一章。简介

这项工作使用 AI 进行了翻译。我们很高兴收到你的反馈和评论：translation-feedback@oreilly.com

到本章结束时，你应该能够识别 Genai 在你应用程序路线图中的角色以及与之相关的挑战。

# 什么是生成式人工智能？

*生成式 AI*是自动学习的一个子集，它专注于使用在数据集上训练的模型创建新内容。*训练模型*，这是一个代表训练数据中的模型和分布的数学模型，可以产生与训练数据集相似的新数据。

为了说明这些概念，想象你在一个包含蝴蝶图像的数据集上训练一个模型。该模型学习蝴蝶图像像素之间的复杂关系。一旦训练完成，你可以采样该模型以创建新的蝴蝶图像，这些图像在原始数据集中不存在。这些图像将包含与原始蝴蝶图像的相似之处，同时保持差异。

###### 注意

使用训练数据中学习到的模型来创建基于模型的新内容被称为推断。

图 1-1 展示了整个过程。

![bgai 0101](img/bgai_0101.png)

###### 图 1-1。一个用于创建新蝴蝶照片的生成训练模型

由于我们不希望生成与训练数据集相同的结果，我们在采样过程中添加随机噪声以创建结果的变化。这种影响样本生成的随机成分使得生成模型成为*概率性的*，并将生成模型与固定计算函数（例如，平均多张图像的像素以创建新的图像）区分开来。

当你处理 GenAI 解决方案时，你可能会遇到六种生成模型家族，包括：^(1)

变分自编码器（VAE）

VAE 学习在低维数学空间（称为*潜在空间* - 见图 1-2）中编码数据，然后在生成新数据时将其解码回原始空间。

生成对抗网络（GAN）

GAN 是一对相互对抗的神经网络（一个判别器和生成器），在训练过程中学习数据中的模型。一旦训练完成，你可以使用生成器来创建新的数据。

自回归模型

这些模型学习根据先前值预测序列的下一个值。

流模型规范化

这些模型将简单的概率分布（数据中的模型）转换为更复杂的分布以生成新数据。

基于能量的模型（EBM）

EBM（能量基模型）基于统计力学，定义了一个能量函数，该函数将较低的能量分配给观测数据，将较高的能量分配给其他配置。

扩散模型

扩散器学习向训练数据中添加噪声以创建纯净的噪声分布。然后，它们学习逐步从采样的点（从纯净的噪声分布）中去除噪声以生成新的数据。

Transformer

Transformer 可以以极高的效率并行建模大型序列数据，如文本语料库。这些模型使用*自注意力*机制来捕捉序列中元素之间的上下文和关系。给定一个新的序列，它们可以使用学到的模型来生成新的数据序列。Transformer 通常被用作*语言模型*来处理和生成文本数据，因为它们更好地处理文本中的长距离关系。OpenAI 的 ChatGPT 是一个定义为*预训练生成式 Transformer*（GPT）的语言模型。

一个值得注意的事实是，这些生成性模型通常只能处理某些类型的数据或*模式*，如文本、图像、音频、视频、点云或甚至 3D 网格。其中一些甚至是*多模态*的，例如 OpenAI 的 GPT-4，它能够原生地处理更多模式，如文本、音频和图像。

为了解释这些生成式人工智能（GenAI）的概念，我将使用一个图像生成的案例来举例。其他案例包括用作聊天机器人或文档解析器的语言模型、用于生成音乐或语音合成的音频模型以及用于创建 AI 头像和深度伪造的视频生成器。你可能已经看到了超过一打的其他案例，还有数百个案例有待被发现。

一个事实是确定的：生成式人工智能服务将推动未来应用的发展。让我们看看原因。

# 为什么生成式人工智能服务将推动未来应用的发展

我们使用计算机来自动化日常问题的解决方案。

在过去，自动化一个流程需要手动编码业务规则，这可能既长又无聊，尤其是在依赖于手动编写的规则来检测垃圾邮件等复杂问题时。如今，可以训练一个模型来理解业务流程的细微差别。一旦训练完成，这个模型可以超越作为应用程序代码实现的手动编写的规则，从而取代这些规则。

这种基于模型的自动化转变催生了基于人工智能的应用程序，它们解决了一系列问题，如价格优化、产品推荐或天气预报。在这个浪潮中，出现了生成型模型，它们与其他类型的人工智能的区别在于它们产生多媒体内容（文本、代码、图像、音频、视频等）的能力，而传统人工智能则更多地集中在预测和分类上。

作为软件工程师，我认为这些模型具有一些将影响未来应用程序开发路线图的能力：它们可以：

+   促进创造性过程

+   提供与上下文相关的解决方案

+   个性化用户体验

+   最小化客户查询解决的延迟

+   作为复杂系统的接口

+   自动化手动行政活动

+   规模化和民主化内容生成

我们将更详细地分析每个功能。

## 促进创造性过程

掌握技能和获取知识在认知上具有挑战性：你可能会花很多时间学习和实践，才能形成原创的想法，以产生原创和创造性的内容，如论文或设计。

在创造性过程中，你可能会遇到写作障碍：难以想象和可视化场景，难以在想法之间导航，难以构建叙事，难以构建论点和理解概念之间的关系。创造性过程需要深刻理解你创作背后的目的，以及清晰意识到灵感和你打算借鉴的想法来源。通常，当你坐下来创作新事物，如原创论文时，你可能发现从空屏幕或空白纸张开始很难。你需要对某个主题进行深入研究，以形成你的观点和想要撰写的叙述。

创造性过程不仅适用于写作，也适用于设计。例如，当你设计用户界面时，你可能需要几个小时的研究设计，浏览设计网站以找到关于色彩调色板、布局和构图的想法，然后再开始设计。在一张白纸上创造真正原创的东西可能感觉就像徒手攀岩。你需要灵感，并需要遵循创造性过程。

产生原创内容需要创造力。因此，对于人类来说，从零开始产生原创想法是一项非凡的任务。新的想法和创作通常基于灵感、其他作品的联系和改编。创造力涉及复杂且非线性的思维和情感智力，这使得它难以复制或用规则和算法自动化。然而，今天可以使用生成式人工智能来模仿创造力。

生成式 AI 工具可以帮助你简化过程，将来自人类广泛知识库的各种想法和概念联系起来。使用这些工具，你可以遇到需要理解广泛互联知识库和不同概念之间相互作用的新想法。此外，这些工具可以帮助你想象难以可视化的场景或概念。为了说明这一点，试着想象示例 1-1 中描述的场景。

##### 示例 1-1\. 对人类难以可视化的场景的描述

```py
An endless biomechanical forest where trees have metallic roots and glowing neon
leaves, each tree trunk embedded with rotating gears and digital screens
displaying alien glyphs; the ground is a mix of crystalline soil and pulsating
organic veins, while a surreal sky shifts between a digital glitch matrix and a
shimmering aurora made of liquid light.
```

如果不习惯于想象这类概念，这可能很难想象。然而，借助生成模型，任何人都可以将复杂的概念可视化并向他人传达。

将 Esempio 1-1 的场景描述提供给图像生成工具[DALL-E 3 (OpenAI)](https://oreil.ly/Z80Qm)，可以得到图 1-3 中显示的输出。

![bgai 0103](img/bgai_0103.png)

###### 图 1-3\. 由 [DALL-E 3](https://oreil.ly/Z80Qm) 生成的图像

看到这些生成式 AI 工具如何帮助你可视化并传达复杂概念是令人着迷的。这些工具让你能够扩展你的想象力并激发你的创造力。当你感到受阻或难以传达或想象新想法时，你可以求助于这些工具寻求帮助。

在未来，我看到应用程序将包括类似功能，以帮助用户在他们的创作过程中。如果你的应用程序为用户提供多种建议作为依据，那么它们可以帮助用户进入状态并激发动力。

## 提供相关解决方案的建议

经常会遇到一些特定领域的问题，这些问题没有现成的解决方案。这些问题的解决方案并不明显，需要大量的研究、尝试和错误、与其他专家的咨询和阅读。开发者非常了解这种情况，因为找到与编程问题相关的解决方案可能很复杂，并不简单。

这是因为开发者必须在特定背景下解决问题：没有对“情况”的详细描述，就无法定义问题，而“情境”是在 *背景下* 产生的。

事实上，*上下文*将潜在的解决方案限制在了一个问题上。

使用搜索引擎时，人们会寻找包含或可能包含相关解决方案的关键词信息源。当开发者寻找解决方案时，他们会将错误日志粘贴到 Google 上，并被引导到像 [Stack Overflow](https://stackoverflow.com) 这样的编程问答网站上。因此，开发者必须希望找到在相同背景下遇到相同问题并已提供解决方案的人。这种方法在寻找编程问题的解决方案方面效率不高。你并不总是能在这些网站上找到作为开发者所需的解决方案。

开发者正在转向生成式 AI 来解决编程问题。通过提供一个描述问题上下文的提示，AI 可以生成潜在的解决方案。更好的是，与代码编辑器的集成非常有用，可以为语言模型提供上下文，这在 Google 或 Stack Overflow 上进行搜索时是不可能的。因此，这些 IA 模型可以生成基于在线论坛和不同问答网站知识库的上下文相关和基于知识的解决方案。有了这些提出的解决方案，你就可以决定哪一个适合你。

正因如此，使用 GenAI 编码工具通常比在论坛和在线网站上搜索解决方案更快。即使是编程问答网站 [Stack Overflow](https://oreil.ly/nOX_K) 也已经将流量下降率（~14%）归因于那些在 GPT-4 发布后尝试了语言模型和代码生成器的开发者。这个数字可能更高，因为该网站的一些用户在公司的博客帖子下评论，表示他们认为网站上的用户活动急剧减少。实际上，截至我们撰写本文时，与 2018 年相比，网站上的 [提问和 upvote 活动下降了 60%](https://oreil.ly/P6Kur)。

在任何情况下，Stack Overflow 预计未来随着引入 GenAI 编码工具，这些工具将使编码民主化、扩大开发者社区并创造新的编程挑战，其流量将增加和减少。问答网站的力量不仅在于找到答案，还在于理解周围的讨论以及参考来源的重要性.^(2) 因此，这些网站将因其专家社区和由人类编辑的内容而继续成为开发者的宝贵资源，这些内容保证了答案或解决方案的正确性和质量.^(3)

## 用户个性化体验

现代软件的客户和用户在使用现代应用程序时，期待一定程度的定制化和互动性。

将生成模型，如语言模型，集成到现有应用程序中，可以创新用户与系统交互的方式。您不再需要通过传统界面进行点击多个屏幕的传统交互，而是可以通过与聊天机器人进行自然语言对话来获取您所需的信息或执行您需要的操作。例如，当您在旅行规划网站上浏览时，您可以描述您的理想假期，并让聊天机器人根据平台对航空公司、住宿提供商和度假套餐数据库的访问，为您准备一个行程。或者，如果您已经预订了假期，您可以要求根据从您的账户中获取的行程细节提供旅游建议。然后，聊天机器人可以向您描述结果并征求您的反馈。

这些语言模型可以充当个人助理，通过提出相关的问题，直到将您的偏好和独特需求映射到产品目录中，以生成个性化推荐。这些虚拟助手能够理解您的意图，并根据您的具体情况提出相关建议。如果您不喜欢这些建议，您可以提供反馈以完善您喜欢的建议。

在教育领域，这些 GenAI 模型可以被用来描述或可视化难以理解的概念，并适应每个学生的偏好和学习能力。

在游戏和虚拟现实（VR）中，GenAI 可以被用来构建基于用户与应用程序交互的动态环境和世界。例如，在角色扮演游戏（RPG）中，可以根据用户实时做出的对话决策和选择，使用大型语言模型即时生成角色叙述和故事。这个过程为玩家和这些应用程序的用户创造了一个独特的体验。

## 将客户查询的延迟降到最低

除了个性化助手之外，企业通常需要支持来管理大量客户服务查询。由于查询量很大，客户往往需要等待漫长的排队或几天的工作日才能从公司那里得到回复。此外，随着企业运营复杂性和客户数量的增加，及时解决客户查询可能变得更加昂贵，并需要广泛的员工培训。

GenAI 能够简化客户服务流程，既方便客户也方便企业。现在，客户可以与一个能够访问数据库和相关资源的语言模型进行聊天或打电话，几分钟内就能解决查询，而不是几天。

与传统聊天机器人通常基于一系列手工制作的规则和预定义脚本不同，由 GenAI 驱动的聊天机器人可以做得更好：

+   理解对话的上下文

+   考虑用户偏好

+   形成动态和个性化的响应

+   接受并适应用户反馈

+   处理意外的查询，特别是关于历史对话或更广泛的内容。

这些因素使得 GenAI 聊天机器人能够与客户进行更自然和多样化的互动。这些机器人将成为那些希望在将问题交给人工客服之前快速获得答案的客户的第一接触点。作为客户，如果你希望通过避免长时间排队和快速解决问题，你可能会更愿意先与这些 GenAI 聊天机器人之一交谈。

这些例子只是所有可能集成到现有应用程序中的功能表面的冰山一角。这种生成模型的灵活性和敏捷性为未来的新应用打开了众多可能性。

## 作为复杂系统的接口

在当今社会，许多人仍然在与数据库或开发者工具等复杂系统交互时遇到问题。非开发者可能需要访问信息或执行活动，而无需具备在这些复杂系统中执行命令的必要技能。LLMs 和 GenAI 模型可以充当这些系统和用户之间的接口。

用户可以使用自然语言提供提示，而 GenAI 模型可以编写并执行对复杂系统的查询。例如，一位投资经理可以要求一个 GenAI 聊天机器人汇总公司数据库中的投资组合表现，而无需由专业人士发送报告请求。另一个例子是 Photoshop 的新生成填充工具，它根据上下文生成图像层并执行修改，为不熟悉 Photoshop 各类工具的用户提供服务。

不同的人工智能初创公司已经开发了 GenAI 应用程序，其中用户通过自然语言与语言模型交互以执行操作。这些初创公司正在使用语言模型取代复杂的流程和工作流，以及用户界面中的多屏幕点击。

尽管 GenAI 模型可以作为数据库或 API 等复杂系统的界面，但开发者仍需实施安全屏障和保护措施，你将在第九章（Capitolo 9）关于人工智能安全的学习中了解到。这些集成需要谨慎管理，以避免通过生成模型对这些系统产生有害查询和攻击向量。

## 自动化手动行政活动

在许多历史悠久的大型公司中，通常有几个团队在执行不太为内部团队和客户所见的手动行政活动。

典型的行政活动通常涉及手动处理具有复杂布局的文档，如发票、采购订单和付款收据。直到不久前，这些活动仍然主要是手动的，因为每份文档的布局和信息排列可能都是视觉上独特的，需要人工验证或批准。此外，为自动化这些流程开发的任何软件都可能脆弱，并且在极限情况下可能需要高精度和正确性。

现在，语言模型和其他生成模型可以进一步自动化这些手动流程的一些部分，并提高其准确性。如果现有的自动化因为极限情况或流程变更而无法工作，语言模型可以介入来验证结果是否符合某些标准，填补空白或报告需要人工审查的元素。

## 扩展和民主化内容生成

人们喜欢新鲜内容，总是寻找新的想法去探索。现在，作家们可以在撰写博客文章时使用 GenAI 工具进行研究和构思。通过与模型对话，他们可以进行头脑风暴，生成草稿。

对于内容生成来说，生产力的提升是巨大的。你不再需要执行低级认知任务，如总结研究或单独重述句子。撰写一篇高质量的博客文章所需的时间从几天缩短到几小时。在开始使用 GenAI 填补空白之前，你可以先专注于内容的轮廓、流程和结构。当你难以排列正确的词语以获得清晰和简洁时，GenAI 工具可以大放异彩。然而，使文章有趣的是风格和写作流程，而不仅仅是内容。

许多公司已经开始使用这些工具来探索想法和编写文档、提案、社交媒体和博客。

Nel complesso, questi sono diversi motivi per cui credo che in futuro un numero maggiore di sviluppatori integrerà le funzioni di GenAI nelle loro applicazioni. La tecnologia è ancora agli albori e ci sono ancora molte sfide da superare prima che GenAI possa essere adottato su larga scala.

# Come costruire un servizio di intelligenza artificiale generativa

I modelli generativi hanno bisogno di accedere a numerose informazioni contestuali per fornire risposte più accurate e pertinenti. In alcuni casi, possono anche aver bisogno di accedere a strumenti per eseguire azioni per conto dell'utente, ad esempio per effettuare un ordine eseguendo una funzione personalizzata. Di conseguenza, potrebbe essere necessario costruire delle API attorno ai modelli generativi (come wrapper) per occuparsi delle integrazioni con fonti di dati esterni (ad esempio, database, API, ecc.) e del controllo dell'accesso degli utenti al modello.

Per costruire questi wrapper API, puoi collocare i modelli generativi dietro un server web HTTP e implementare le integrazioni, i controlli e i router necessari, come mostrato nella Figura 1-4.

![bgai 0104](img/bgai_0104.png)

###### Figura 1-4\. Server web FastAPI con integrazioni di fonti di dati che servono un modello generativo

Il server web controlla l'accesso alle fonti di dati e al modello. Sotto il cofano, il server può interrogare il database e i servizi esterni per arricchire i prompt degli utenti con informazioni rilevanti e generare output più pertinenti. Una volta generati gli output, il livello di controllo può effettuare il sanity-check mentre i router restituiscono le risposte finali all'utente.

###### Suggerimento

Puoi anche fare un passo avanti configurando un modello di linguaggio per costruire un'istruzione per un altro sistema e passarla a un altro componente per eseguire quei comandi, ad esempio per interagire con un database o effettuare una chiamata API.

In sintesi, il server web agisce come un intermediario cruciale che gestisce l'accesso ai dati, arricchisce i prompt degli utenti e controlla la qualità dei risultati generati prima di inoltrarli agli utenti. Oltre a servire i modelli generativi agli utenti, questo approccio a più livelli migliora la rilevanza e l'affidabilità delle risposte dei modelli generativi.

# Perché costruire servizi di intelligenza artificiale generativa con FastAPI?

I servizi di IA generativa richiedono framework web performanti come motori backend che alimentano i servizi e le applicazioni event-driven. FastAPI, uno dei framework web più popolari in Python, [è in grado di competere](https://oreil.ly/LmEg7) in termini di prestazioni con altri framework web popolari come *gin (Golang)* o *express (Node.js)*, pur mantenendo la ricchezza dell'ecosistema di deep learning di Python. I framework non Python non hanno questa integrazione diretta necessaria per lavorare con un modello di IA generativa all'interno di un servizio.

在 Python 生态系统内，存在多个用于创建 API 服务的 Web 框架。最受欢迎的选项包括：

Django

这是一个内置电池的全栈框架。它是一个成熟的框架，拥有广泛的社区和大量支持。

Flask

一个轻量级且可扩展的微框架。

FastAPI

这是一个现代的 Web 框架，旨在保证速度和性能。它是一个全栈框架，提供内置的电池。

FastAPI，尽管它最近才进入 Python Web 框架的领域，已经获得了牵引力和人气。在我们撰写本文时，FastAPI 是 Python Web 框架中下载增长最快的框架，也是 GitHub 上最受欢迎的第二个 Web 框架。它正朝着比 Django 更受欢迎的方向发展，这得益于其 GitHub 星标数量的不断增长（在我们撰写本文时约为 80,000 个）。

在所提到的框架中，Flask 因其声誉、社区支持和可扩展性而位居下载量之首。然而，作为一个微框架，它提供的预定义功能有限，例如，它提供了开箱即用的模式验证支持。

Django 也因其通过 Django Rest Framework 创建 API 和遵循 MVC（模型-视图-控制器）模式的单体应用而受到欢迎，但它对异步 API 的支持不够成熟，可能会限制性能，并可能给创建可读 API 增加复杂性和额外成本。

与其他 Web 框架相比，FastAPI 提供了多种功能，如数据验证、类型安全、自动文档和内置的 Web 服务器。因此，熟悉 Python 的开发者可能会从过时的框架如 Django 转向 FastAPI。我认为开发者的卓越经验、开发自由、出色的性能以及通过生命周期事件最近对 AI 模型的支撑可能有助于这一现象。

本书讨论了使用 FastAPI 框架开发能够自主执行操作并与外部服务交互的生成式人工智能服务的实现细节。

为了学习最重要的概念，我将引导你通过一个基础项目，你可以在阅读本书的同时进行工作。

# 什么阻碍了生成式 AI 服务的采用

组织在采用 IA 生成服务时必须面对各种挑战。存在与 IA 生成结果的不精确性、相关性、质量和一致性相关的问题。此外，还有关于数据隐私、网络安全和如果用于生产，模型潜在滥用和不当使用的担忧。因此，企业还不愿完全赋予这些模型自主权。在将它们直接连接到敏感系统（如内部数据库或支付系统）时存在犹豫。

将 IA 服务与现有系统（如内部数据库、Web 界面和外部 API）集成可能是一个挑战。这种集成可能因为兼容性问题、技术能力需求、可能存在的现有流程中断、恶意攻击者对这些系统的攻击尝试以及类似的安全和隐私数据担忧而变得困难。

想要为面向客户的用例使用该服务的公司希望模型的响应保持一致性和相关性，并确保结果不会冒犯或不适当。

使用这些生成性模型生产原创和高质量内容也存在一些限制。如前所述，这些 GenAI 工具有效地将各种想法和概念结合在特定领域内，但它们无法产生全新的或新颖的想法；相反，它们通过重新组合和重新表述现有信息来使其看起来新颖。此外，在生成过程中，它们遵循常见模式，这些模式可能是通用的、重复的且缺乏启发性。最后，它们可能产生看似合理的输出，但实际上是完全错误和虚构的，不是基于事实或现实的。

###### 注意

在 GenAI 模型产生虚构事实和错误信息的情况下，这些情况被定义为*幻觉*。

这些模型产生幻觉的趋势阻止了它们在需要高度精确结果的应用场景中的采用，例如医疗诊断、法律咨询和自动化考试。

一些挑战，如数据隐私和安全问题，可以通过最佳软件工程实践来解决，你将在本书中找到更多相关信息。解决其他挑战需要优化模型输入或调整模型（通过在特定用例中引入新的示例来调整其参数）以提高相关性、质量、一致性和结果的一致性。

# 项目概要

在这本书中，我将指导你使用 FastAPI 作为基础 Web 框架构建一个生成式人工智能服务。

该服务包括：

+   与不同模型集成，包括用于文本和聊天生成的语言模型、用于语音合成的音频模型以及用于图像生成的稳定扩散模型。

+   以文本、音频或图像的形式实时生成针对用户查询的响应。

+   使用 RAG 技术通过向量数据库“对话”已上传的文档

+   执行网络抓取并与内部数据库、外部系统和 API 通信，以在响应查询时收集足够的信息

+   将对话时间线记录在关系型数据库中

+   通过基于令牌的凭据和 GitHub 身份登录进行用户认证

+   通过权限保护限制基于用户权限的响应。

+   使用护栏提供足够的保护，以防止不当使用和滥用

由于本书专注于创建 API 服务，你将学习如何使用 Python Streamlit 包和简单的 HTML 来开发用户界面。在实际应用中，你可能会将你的生成式 AI 服务与使用 React 库或 Next.js 框架构建的定制用户界面接口，以获得模块化、可扩展性和可伸缩性。

# 摘要

在本章中，你学习了生成式人工智能的概念以及它如何能够使用训练数据中的模型以各种模式（如文本、音频、视频等）创建数据。你还看到了该技术的多个实际示例和用例，以及为什么大多数未来的应用程序都将由生成式 AI 的能力驱动。

你还学习了生成式 AI 如何简化创造性过程、消除中间人、个性化用户体验以及使复杂系统和内容生成民主化。此外，你了解了阻碍生成式 AI 广泛采用的多个挑战及其解决方案。最后，你学习了如何使用 FastAPI 框架和本书中的代码示例构建生成式 AI API 服务。

在下一章中，你将学习了解 FastAPI，这将使你能够实现自己的生成式 AI 服务。

^(1) 你可以在 David Foster 的《生成式深度学习》（O'Reilly，2024）中找到有关这些模型的更多信息。

^(2) 近期生成的 AI 工具现在可以提供与解决方案一起的参考资料（例如，[phind.com](https://phind.com)）。
