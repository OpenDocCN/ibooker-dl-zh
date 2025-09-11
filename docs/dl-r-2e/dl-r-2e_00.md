# 前言

如果你拿起这本书，你可能已经意识到了近年来深度学习对人工智能领域所代表的非凡进步。我们从几乎无法使用的计算机视觉和自然语言处理到在你每天使用的产品中大规模部署的高性能系统。这一突然进步的后果影响几乎每个行业。我们已经将深度学习应用于跨越医学成像、农业、自动驾驶、教育、灾害预防和制造等不同领域的一系列重要问题。

然而，我认为深度学习仍处于早期阶段。到目前为止，它只实现了其潜力的一小部分。随着时间的推移，它将走向能够帮助的每一个问题——这是一个需要几十年时间的转变。

要开始将深度学习技术部署到每一个可以解决的问题上，我们需要让尽可能多的人包括非专家——那些不是研究人员或研究生的人——能够接触到它。为了使深度学习发挥其全部潜力，我们需要彻底民主化它。今天，我相信我们正处于一个历史性转变的关键时刻，深度学习正在从学术实验室和大型科技公司的研发部门走向成为每个开发者工具箱中无处不在的一部分——类似于上世纪 90 年代末的 web 开发轨迹。现在几乎任何人都可以建立一个网站或网络应用，而在 1998 年，这可能需要一个小团队的专业工程师。在不久的将来，任何有想法和基本编码技能的人都将能够构建能够从数据中学习的智能应用程序。

当我在 2015 年 3 月发布了 Keras 深度学习框架的第一个版本时，AI 的民主化并不是我的初衷。我已经在机器学习领域做了几年的研究，并建立了 Keras 来帮助我进行实验。但自 2015 年以来，成千上万的新人进入了深度学习领域；其中许多人选择了 Keras 作为他们的首选工具。当我看到许多聪明的人以意想不到的强大方式使用 Keras 时，我开始非常关心 AI 的可访问性和民主化。我意识到，我们传播这些技术的越广泛，它们就变得越有用和有价值。可访问性很快成为 Keras 开发的一个明确目标，在短短几年内，Keras 开发者社区在这方面取得了巨大的成就。我们已经让数十万人掌握了深度学习，这些人又在解决一些直到最近被认为无法解决的问题。

你手中的这本书是让尽可能多的人了解深度学习的又一步。Keras 一直需要一个配套课程，以同时涵盖深度学习的基础知识、深度学习最佳实践和 Keras 使用模式。在 2016 年和 2017 年，我尽力制作了这样一个课程，它成为了本书的第一版，于 2017 年 12 月发布。它迅速成为了一本机器学习畅销书，销量超过了 5 万册，并被翻译成了 12 种语言。

然而，深度学习领域发展迅速。自第一版发布以来，许多重要的进展已经发生——TensorFlow 2 的发布、Transformer 架构的日益流行等等。因此，2019 年底，我开始更新我的书。我最初相当幼稚地认为，它将包含大约 50%的新内容，并且最终长度大致与第一版相同。实际上，在两年的工作后，它的长度超过了三分之一，约有 75%是全新内容。它不仅是一次更新，而且是一本全新的书。

我写这本书的重点是尽可能使深度学习背后的概念以及它们的实现变得易于理解。这样做并不需要我简化任何内容——我坚信在深度学习中没有难懂的理念。我希望你会发现这本书有价值，并且它能让你开始构建智能应用程序并解决对你而言重要的问题。

# 致谢

首先，我要感谢 Keras 社区让这本书得以实现。在过去的六年里，Keras 已经发展成为拥有数百名开源贡献者和超过一百万用户的平台。你们的贡献和反馈让 Keras 成为今天的样子。

更加个人化地，我要感谢我的妻子在 Keras 的开发和这本书的写作过程中给予我的无尽支持。

我还要感谢 Google 支持 Keras 项目。看到 Keras 被采用为 TensorFlow 的高级 API 真是太棒了。Keras 和 TensorFlow 之间的顺畅集成极大地有利于 TensorFlow 用户和 Keras 用户，并使深度学习变得更加易于使用。

我要感谢 Manning 出版社的人员让这本书得以实现：出版商 Marjan Bace 以及编辑和制作团队的所有人，包括 Michael Stephens、Jennifer Stout、Aleksandar Dragosavljević、Andy Marinkovich、Pamela Hunt、Susan Honeywell、Keri Hales、Paul Wells，以及许多在幕后工作的其他人。

非常感谢所有审阅者：Arnaldo Ayala Meyer、Davide Cremonesi、Dhinakaran Venkat、Edward Lee、Fernando García Sedano、Joel Kotarski、Marcio Nicolau、Michael Petrey、Peter Henstock、Shahnawaz Ali、Sourav Biswas、Thiago Britto Borges、Tony Dubitsky、Vlad Navitski，以及所有其他给我们提供反馈的人。你们的建议帮助这本书变得更好。

在技术方面，特别感谢担任本书技术校对的 Ninoslav Cerkez。

# 致谢

首先，我要感谢 Keras 社区让这本书得以实现。在过去的六年里，Keras 已经发展成为拥有数百名开源贡献者和超过一百万用户的平台。你们的贡献和反馈让 Keras 成为今天的样子。

更加个人化地，我要感谢我的妻子在 Keras 的开发和这本书的写作过程中给予我的无尽支持。

我还要感谢 Google 支持 Keras 项目。看到 Keras 被采用为 TensorFlow 的高级 API 真是太棒了。Keras 和 TensorFlow 之间的顺畅集成极大地有利于 TensorFlow 用户和 Keras 用户，并使深度学习变得更加易于使用。

我要感谢 Manning 出版社的人员让这本书得以实现：出版商 Marjan Bace 以及编辑和制作团队的所有人，包括 Michael Stephens、Jennifer Stout、Aleksandar Dragosavljević、Andy Marinkovich、Pamela Hunt、Susan Honeywell、Keri Hales、Paul Wells，以及许多在幕后工作的其他人。

非常感谢所有审阅者：Arnaldo Ayala Meyer、Davide Cremonesi、Dhinakaran Venkat、Edward Lee、Fernando García Sedano、Joel Kotarski、Marcio Nicolau、Michael Petrey、Peter Henstock、Shahnawaz Ali、Sourav Biswas、Thiago Britto Borges、Tony Dubitsky、Vlad Navitski，以及所有其他给我们提供反馈的人。你们的建议帮助这本书变得更好。

在技术方面，特别感谢担任本书技术校对的 Ninoslav Cerkez。

# 关于本书

本书是为任何希望从零开始探索深度学习或扩展对深度学习理解的人编写的。无论你是一名实践的机器学习工程师、数据科学家还是大学生，你都会在这些页面中找到价值。

你将以一种易于理解的方式探索深度学习——从简单开始，然后逐步掌握最先进的技术。你会发现本书在直觉、理论和动手实践之间取得了平衡。它避免使用数学符号，而更倾向于通过详细的代码片段和直观的心理模型来解释机器学习和深度学习的核心思想。你将从丰富的代码示例中学习，这些示例包括广泛的评论、实用的建议以及对开始使用深度学习解决具体问题所需知道的一切的简单高级解释。

代码示例使用了深度学习框架 Keras，其数值引擎为 TensorFlow 2。它们展示了截至 2022 年的现代 Keras 和 TensorFlow 2 最佳实践。

读完本书后，你将对深度学习是什么、什么时候适用以及它的局限性有一个扎实的理解。你将熟悉处理和解决机器学习问题的标准工作流程，并且你将知道如何解决常遇到的问题。你将能够使用 Keras 解决从计算机视觉到自然语言处理的实际问题：图像分类、图像分割、时间序列预测、文本分类、机器翻译、文本生成等等。

## 谁应该阅读本书？

这本书是为具有 R 编程经验的人们编写的，他们想要开始学习机器学习和深度学习。但这本书也对许多不同类型的读者有价值：

+   如果你是一个熟悉机器学习的数据科学家，这本书将为你提供一个扎实的、实用的深度学习入门，这是机器学习中增长最快、最重要的子领域。

+   如果你是一名希望开始使用 Keras 框架的深度学习研究人员或实践者，你会发现本书是理想的 Keras 速成课程。

+   如果你是一名在正式环境中学习深度学习的研究生，你会发现本书是对你教育的一个实用补充，帮助你建立对深度神经网络行为的直觉，并让你熟悉关键的最佳实践。

即使是不经常编码的技术人员，他们也会发现本书对基本和高级深度学习概念的介绍很有用。

要理解代码示例，你需要具备合理的 R 熟练程度。你不需要有机器学习或深度学习的先前经验：本书从零开始覆盖了所有必要的基础知识。你也不需要有高级数学背景——高中水平的数学应该足以跟上。

## 关于代码

本书包含许多源代码示例，其格式既有编号的代码段，也有与普通文本并排的代码。在这两种情形中，源代码都采用固定宽度字体进行格式化，以与普通文本区分开。运行代码后的输出也采用固定宽度字体格式化，但左侧还带有一个垂直灰条。在整本书中，您会发现代码和代码输出以此交替呈现：

print("R 真是棒极了！")

[1] "R 真是棒极了！"

在许多情况下，原始源码已经被重新格式化；我们添加了换行符和重新排列缩进，以适应书中可用的页面空间。在极少数情况下，甚至这些也不够用，代码段中包含了行延续标志（![Image](img/common01.jpg)）。此外，在文本中描述代码时，源代码中的注释通常会被移除。代码标注伴随许多代码片段，突出重要概念。

您可以从这本书的 liveBook（在线）版本中获取可执行的代码片段，网址为 [`livebook.manning.com/book/deep-learning-with-r-second-edition/`](https://livebook.manning.com/book/deep-learning-with-r-second-edition/)，并且在 GitHub 上获得 R 脚本，网址为 [`github.com/t-kalinowski/deep-learning-with-R-2nd-edition-code`](https://github.com/t-kalinowski/deep-learning-with-R-2nd-edition-code)。

## liveBook 讨论论坛

购买*Deep Learning with R, Second Edition*，包含免费访问 liveBook，Manning 的在线阅读平台。通过 liveBook 的独家讨论功能，您可以对全书或特定部分或段落附加评论。您可以为自己做笔记，提出和回答技术问题，还可以从作者和其他用户那里得到帮助。要访问论坛，请前往 [`livebook.manning.com/book/deep-learning-with-r-second-edition/`](https://livebook.manning.com/book/deep-learning-with-r-second-edition/)。您也可以在 [`livebook.manning.com/discussion`](https://livebook.manning.com/discussion) 了解更多有关 Manning 论坛和行为规则的信息。

Manning 对我们读者的承诺是提供一个场所，读者个人之间以及读者与作者之间能够进行有意义的对话。这并不意味着作者有义务参与特定数量的讨论，作者对论坛的贡献仍然是自愿的（且未付酬）。我们建议您尝试向作者提出一些具有挑战性的问题，以免他们失去兴趣！只要这本书还在售卖中，论坛和以前的讨论记录将可以从出版商网站上访问到。

## 关于封面插图

*Deep Learning with R, Second Edition* 封面上的插图“1700 年中国女士的风格”取自托马斯·杰弗里斯（Thomas Jefferys）的一本书，出版时间介于 1757 年至 1772 年之间。

在那些日子里，通过人们的服装很容易辨认出他们的居住地和职业或地位。曼宁通过基于几个世纪前丰富多样的地区文化的书封面，以及像这样的图片收藏，庆祝了计算机业务的独创性和主动性。

# 关于作者

**FRANÇOIS CHOLLET** 是 Keras 的创建者，这是最广泛使用的深度学习框架之一。他目前是 Google 的软件工程师，负责领导 Keras 团队。此外，他还进行关于抽象、推理以及如何在人工智能中实现更大的一般性的研究。

**TOMASZ KALINOWSKI** 是 RStudio 的软件工程师，担任 TensorFlow 和 Keras R 包的维护者。在之前的角色中，他曾作为科学家和工程师工作，将机器学习应用于各种各样的数据集和领域。

**J.J. ALLAIRE** 是 RStudio 的创始人，也是 RStudio IDE 的创建者。J.J. 是 TensorFlow 和 Keras 的 R 接口的作者。
