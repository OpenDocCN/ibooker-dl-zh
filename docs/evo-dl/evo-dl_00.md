# 前言

## 前言

当我在 25 多年前开始我的机器学习和人工智能职业生涯时，两种主导技术被认为是下一个大事件。这两种技术都显示出解决复杂问题的潜力，并且它们在计算上等效。这两种技术是进化算法和神经网络（深度学习）。

在接下来的几十年里，我见证了进化算法的急剧下降和深度学习的爆炸性增长。虽然这场战斗是通过计算效率赢得的，但深度学习也展示了众多新颖的应用。另一方面，对于大部分情况，进化算法和遗传算法的知识和使用已经减少到脚注的地位。

我写这本书的目的是展示进化算法和遗传算法为深度学习系统提供的好处。随着深度学习时代成熟到自动化机器学习（AutoML）时代，这些好处尤其相关，在这个时代，能够自动化大规模和广泛范围的模型开发正在成为主流。

我也相信，通过观察进化，我们可以帮助我们的通用人工智能和智能的研究。毕竟，进化是自然界用来形成我们智能的工具，那么为什么不能用它来提升人工智能呢？我的猜测是我们太急躁和自大，认为人类可以独自解决这个问题。

通过写这本书，我希望展示进化方法在深度学习之上的力量，作为一种打破常规的思考方式。我的希望是，它以有趣和创新的途径展示了进化方法的基础，同时也涉足进化的深度学习网络（即 NEAT）和本能学习的高级领域。本能学习是我对我们应该更加关注生物生命如何进化，并在我们寻找更智能的人工网络中反映这些相同特性的看法。

## 致谢

我想要感谢开源社区，特别是以下项目：

+   分布式进化算法在 Python 中（DEAP）—[`github.com/DEAP/deap`](https://github.com/DEAP/deap)

+   基于 Python 的基因表达编程框架（GEPPY）—[`github.com/ShuhuaGao/geppy`](https://github.com/ShuhuaGao/geppy)

+   Python 中增强拓扑结构的神经进化（NEAT Python）—[`github.com/CodeReclaimers/neat-python`](https://github.com/CodeReclaimers/neat-python)

+   OpenAI Gym—[`github.com/openai/gym`](https://github.com/openai/gym)

+   Keras/TensorFlow—[`github.com/tensorflow/tensorflow`](https://github.com/tensorflow/tensorflow)

+   PyTorch—[`github.com/pytorch/pytorch`](https://github.com/pytorch/pytorch)

没有其他人不懈地投入工作和时间来开发和维护这些存储库，像这样的书是不可能的。这些也是任何对提高进化算法（EA）或深度学习（DL）技能感兴趣的人的极好资源。

特别感谢我的家人，他们一直支持我的写作、教学和演讲活动。他们总是愿意阅读一段或一节内容，并给我提供意见，无论是好是坏。

非常感谢 Manning 出版社的编辑和生产团队，他们帮助创建了这本书。

感谢所有审稿人：Al Krinker、Alexey Vyskubov、Bhagvan Kommadi、David Paccoud、Dinesh Ghanta、Domingo Salazar、Howard Bandy、Edmund Ronald、Erik Sapper、Guillaume Alleon、Jasmine Alkin、Jesús Antonino Juárez Guerrero、John Williams、Jose San Leandro、Juan J. Durillo、kali kaneko、Maria Ana、Maxim Volgin、Nick Decroos、Ninoslav Čerkez、Oliver Korten、Or Golan、Raj Kumar、Ricardo Di Pasquale、Riccardo Marotti、Sergio Govoni、Sadhana G、Simone Sguazza、Shivakumar Swaminathan、Szymon Harabasz 和 Thomas Heiman。你们的建议帮助使这本书更加完善。

最后，我还要感谢查尔斯·达尔文，他的灵感来源于他写出的开创性作品《物种起源》。作为一个非常虔诚的宗教徒，查尔斯在二十年的时间里内部挣扎，在信仰和观察之间斗争，最终决定出版他的书。最终，他展现了勇气和信任科学，超越了信仰和当时的主流思想。这是我写一本结合进化和深度学习的书时所受到的启发。

## 关于本书

本书向读者介绍了进化算法和遗传算法，从解决有趣的机器学习问题到将概念与深度学习相结合。本书首先介绍 Python 中的模拟和进化算法的概念。随着内容的深入，重点转向展示价值，包括深度学习的应用。

### 适合阅读本书的人群

你应该有扎实的 Python 基础，并理解核心机器学习和数据科学概念。对于理解后续章节中的概念，深度学习背景将是必不可少的。

### 本书组织结构：路线图

本书分为三个部分：入门、优化深度学习和高级应用。在第一部分“入门”中，我们首先介绍模拟、进化以及遗传和其他算法的基础知识。接着，我们转向展示进化算法在深度学习中的应用以及遗传搜索的多种应用。最后，我们通过探讨生成建模、强化学习和通用智能的高级应用来结束本书。以下是各章节的摘要：

+   第一部分：入门

    +   *第一章：介绍进化深度学习*——本章介绍了将进化算法与深度学习相结合的概念。

    +   *第二章：介绍进化计算*——本章提供了计算模拟的基本介绍以及如何利用进化。

    +   *第三章：使用 DEAP 介绍遗传算法*—本章介绍了遗传算法的概念以及如何使用 DEAP 框架。

    +   *第四章：使用 DEAP 进行更多进化计算*—本章探讨了从旅行商问题到生成蒙娜丽莎图像的遗传和进化算法的有趣应用。

+   第二部分：优化深度学习

    +   *第五章：自动化超参数优化*—本章展示了使用遗传或进化算法在深度学习系统中优化超参数的几种方法。

    +   *第六章：神经进化优化*—在本章中，我们探讨了使用神经进化对深度学习系统的网络架构进行优化。

    +   *第七章：进化的卷积神经网络*—本章探讨了使用进化优化卷积神经网络架构的高级应用。

+   第三部分：高级应用

    +   *第八章：演化的自编码器*—本章介绍了或回顾了使用自编码器的生成模型的基础知识。然后，它展示了进化如何发展演化的自编码器。

    +   *第九章：生成深度学习和进化*—本章在上一章的基础上，介绍了生成对抗网络及其如何通过进化进行优化。

    +   *第十章：NEAT：拓扑增强的神经进化*—本章介绍了 NEAT，并涵盖了它如何应用于各种基线应用。

    +   *第十一章：使用 NEAT 进行进化学习*—本章讨论了强化学习和深度强化学习的基础知识，然后展示了如何使用 NEAT 在 OpenAI Gym 上解决一些难题。

    +   *第十二章：进化机器学习和未来*—最后一章探讨了机器学习进化的未来，以及它如何为通用人工智能提供洞见。

虽然这本书旨在从头到尾阅读，但并非所有读者都有时间、背景或兴趣阅读所有内容。以下是一个快速指南，可以帮助你选择想要关注的章节或部分：

+   *第一部分：入门*—如果你是模拟和进化或遗传计算的初学者，请务必阅读本节。本节也可以作为一个有用的复习，并展示了几个有趣的应用。

+   *第二部分：优化深度学习*—如果你确实需要优化用于神经进化或超参数调整的深度学习系统，请阅读本节或其中的特定章节。

+   *第三部分：高级应用*—本部分中的章节分为三个子部分：进化的生成建模（第八章和第九章）、NEAT（第十章和第十一章）和本能学习（第十二章）。这些子部分可以独立处理。

### 关于代码

本书的所有代码都是使用 Google Colab 笔记本编写的，可以在作者的 GitHub 仓库中找到：[`github.com/cxbxmxcx/EvolutionaryDeepLearning`](https://github.com/cxbxmxcx/EvolutionaryDeepLearning)。要运行代码，您只需在浏览器中导航到 GitHub 仓库，并找到相关的代码示例。所有代码示例都带有章节编号的前缀和示例编号，例如，EDL_2_2_Simulating_Life.ipynb。从那里，只需点击 Google Colab 徽标即可在 Colab 中启动笔记本。所有依赖项要么已在 Colab 中预先安装，要么作为笔记本的一部分安装。

本书包含许多源代码示例，既有编号列表，也有与普通文本混排。在这两种情况下，源代码都使用 `fixed-width font like this` 这样的固定宽度字体格式化，以将其与普通文本区分开来。有时代码也会用 **`in` `bold`** 加粗来突出显示与章节中先前步骤不同的代码，例如，当新功能添加到现有代码行时。

在许多情况下，原始源代码已被重新格式化；我们添加了换行并重新调整了缩进，以适应书中的可用页面空间。在极少数情况下，即使这样也不够，列表中还包括了行续续标记（➥）。此外，当代码在文本中描述时，源代码中的注释通常也会从列表中删除。代码注释伴随着许多列表，突出显示重要概念。

您可以从本书的 liveBook（在线）版本中获取可执行的代码片段，网址为 [`livebook.manning.com/book/evolutionary-deep-learning`](https://livebook.manning.com/book/evolutionary-deep-learning)。书中示例的完整代码可以从 Manning 网站 [`www.manning.com/books/evolutionary-deep-learning`](https://www.manning.com/books/evolutionary-deep-learning) 和 GitHub [`github.com/cxbxmxcx/EvolutionaryDeepLearning`](https://github.com/cxbxmxcx/EvolutionaryDeepLearning) 下载。

### liveBook 讨论论坛

购买《进化深度学习》包括免费访问 liveBook，Manning 的在线阅读平台。使用 liveBook 的独家讨论功能，您可以在全局或特定章节或段落中添加评论。为自己做笔记、提问和回答技术问题以及从作者和其他用户那里获得帮助都非常简单。要访问论坛，请访问 [`livebook.manning.com/book/evolutionary-deep-learning/discussion`](https://livebook.manning.com/book/evolutionary-deep-learning/discussion)。您还可以在 [`livebook.manning.com/discussion`](https://livebook.manning.com/discussion) 了解更多关于 Manning 论坛和行为准则的信息。

Manning 对我们读者的承诺是提供一个场所，在这里个人读者之间以及读者与作者之间可以进行有意义的对话。这不是对作者参与特定数量活动的承诺，作者对论坛的贡献仍然是自愿的（且未付费）。我们建议您尝试向作者提出一些挑战性的问题，以免他的兴趣偏离！只要这本书还在印刷，论坛和先前讨论的存档将可通过出版社的网站访问。

## 关于作者

![图片](img/micheal.png)

Micheal Lanham 是一位经验丰富的软件和技术创新者，拥有 25 年的经验。在这段时间里，他作为研发开发者，在游戏、图形、网络、桌面、工程、人工智能、GIS 以及为各种行业开发机器学习应用等领域开发了广泛的软件应用。在千年之交，Micheal 开始与游戏开发中的神经网络和进化算法合作。他利用这些技能和经验，作为 GIS 和大数据/企业架构师，增强和游戏化了一系列工程和商业应用。自 2016 年底以来，Micheal 一直是一位热心的作者和演讲者，将他的知识回馈给社区。目前，他已经完成了关于增强现实、音效设计、机器学习和人工智能的众多书籍。他在人工智能和软件开发领域享有众多声誉，但目前专注于生成建模、强化学习和机器学习操作。Micheal 与家人居住在加拿大卡尔加里，目前正致力于撰写、教学和演讲关于人工智能、机器学习操作和工程软件开发。

## 关于封面插图

《进化深度学习》封面上的图像是“Kourilien 人”，或“库页岛人”，取自 Jacques Grasset de Saint-Sauveur 的收藏，1788 年出版。每一幅插图都是手工精心绘制和着色的。

在那些日子里，仅凭人们的服饰就能轻易地识别出他们居住的地方以及他们的职业或社会地位。Manning 通过基于几个世纪前丰富多样的地域文化的书封面，庆祝计算机行业的创新精神和主动性，这些文化通过像这一系列图片这样的收藏品被重新带回生活。
