# 前言

随着神经网络在2000年代的复苏，深度学习已经成为一个非常活跃的研究领域，为现代机器学习铺平了道路。本书使用解释和示例来帮助您理解这个复杂领域中的主要概念。像谷歌、微软和Facebook这样的大公司已经注意到这一点，并且正在积极发展内部的深度学习团队。对于我们其他人来说，深度学习仍然是一个相当复杂和难以理解的主题。研究论文充斥着行话，而零散的在线教程对于帮助建立对深度学习从业者如何以及为什么处理问题的强大直觉几乎没有帮助。我们的目标是弥合这一差距。

在本书的第二版中，我们提供了更严谨的数学背景部分，旨在更好地为您准备本书其余部分的材料。此外，我们已经更新了有关序列分析、计算机视觉和强化学习的章节，深入探讨了这些领域的最新进展。最后，我们在生成建模和可解释性领域添加了新的章节，为您提供对深度学习领域更广泛的视角。我们希望这些更新能激励您自己练习深度学习，并将所学应用于解决现实世界中的有意义问题。

# 先决条件和目标

本书面向具有基本微积分和Python编程理解的受众。在最新版本中，我们提供了广泛的数学背景章节，特别是线性代数和概率，为您准备前方的材料。

希望通过本书的学习，您将具备使用深度学习解决问题的直觉，了解现代深度学习方法的历史背景，并熟悉使用PyTorch开源库实现深度学习算法。

# 本书的组织结构是怎样的？

本书的前几章致力于通过深入研究线性代数和概率来培养数学素养，这些内容深深嵌入在深度学习领域中。接下来的几章讨论了前馈神经网络的结构，如何在代码中实现它们，以及如何在真实数据集上训练和评估它们。本书的其余部分致力于深度学习的具体应用，并理解为这些应用开发的专门学习技术和神经网络架构背后的直觉。尽管我们在这些后面的部分涵盖了先进的研究，但我们希望提供从第一原理和易于理解的角度出发的这些技术的分解。

# 本书中使用的约定

本书中使用以下排版约定：

*斜体*

表示新术语、URL、电子邮件地址、文件名和文件扩展名。

`等宽字体`

用于程序清单，以及在段落中引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

###### 注意

此元素表示一般说明。

###### 警告

此元素表示警告或注意。

# 使用代码示例

可下载补充材料（代码示例、练习等）请访问[*https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book*](https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book)。

如果您有技术问题或在使用代码示例时遇到问题，请发送电子邮件至[*bookquestions@oreilly.com*](mailto:bookquestions@oreilly.com)。

这本书旨在帮助您完成工作。一般来说，如果本书提供了示例代码，您可以在您的程序和文档中使用它。除非您复制了代码的大部分内容，否则无需征得我们的许可。例如，编写一个使用本书多个代码块的程序不需要许可。销售或分发 O’Reilly 书籍中的示例需要许可。引用本书回答问题并引用示例代码不需要许可。将本书中大量示例代码整合到产品文档中需要许可。

我们感激，但不要求署名。署名通常包括标题、作者、出版商和 ISBN。例如：“*深度学习基础* 作者 Nithin Buduma、Nikhil Buduma 和 Joe Papa（O’Reilly）。版权所有 2022 Nithin Buduma 和 Mobile Insights Technology Group, LLC，978-1-492-08218-7。”

如果您觉得您使用的代码示例超出了合理使用范围或上述许可，请随时联系我们：[*permissions@oreilly.com*](mailto:permissions@oreilly.com)。

# O’Reilly Online Learning

###### 注

40 多年来，[*O’Reilly Media*](https://oreilly.com) 为企业提供技术和商业培训、知识和见解，帮助它们取得成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专长。O’Reilly 的在线学习平台为您提供按需访问实时培训课程、深入学习路径、交互式编码环境以及来自 O’Reilly 和其他 200 多家出版商的大量文本和视频。欲了解更多信息，请访问 [*https://oreilly.com*](https://oreilly.com)。

# 如何联系我们

请将有关本书的评论和问题发送至出版商：

+   O’Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-998-9938（美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们有一个本书的网页，上面列出了勘误、示例和任何额外信息。您可以访问这个页面：[*https://oreil.ly/fundamentals-of-deep-learning-2e*](https://oreil.ly/fundamentals-of-deep-learning-2e)。

发送邮件至 [*bookquestions@oreilly.com*](mailto:bookquestions@oreilly.com) 来评论或提出关于本书的技术问题。

有关我们的书籍和课程的新闻和信息，请访问 [*https://oreilly.com*](https://oreilly.com)。

在 LinkedIn 上找到我们：[*https://www.linkedin.com/company/oreilly-media*](https://www.linkedin.com/company/oreilly-media)。

在 Twitter 上关注我们：[*https://twitter.com/oreillymedia*](https://twitter.com/oreillymedia)。

在 YouTube 上关注我们：[*https://www.youtube.com/oreillymedia*](https://www.youtube.com/oreillymedia)。

# 致谢

我们要感谢几位在完成本文过程中发挥关键作用的人。首先要感谢 Mostafa Samir 和 Surya Bhupatiraju，他们对第[7](ch09.xhtml#ch07)章和第[8](ch12.xhtml#ch08)章的内容做出了重大贡献。我们还感谢 Mohamed (Hassan) Kane 和 Anish Athalye 的贡献，他们在本书的 GitHub 代码库中工作了早期版本的代码示例。

## Nithin 和 Nikhil

没有我们编辑 Shannon Cutt 的无尽支持和专业知识，这本书就不可能完成。我们还要感谢我们的评论员 Isaac Hodes、David Andrzejewski、Aaron Schumacher、Vishwesh Ravi Shrimali、Manjeet Dahiya、Ankur Patel 和 Suneeta Mall，他们对原始草稿提供了深思熟虑、深入和技术性评论。最后，我们要感谢我们的朋友和家人，包括 Jeff Dean、Venkat Buduma、William 和 Jack，在我们完成本书手稿的过程中提供的所有见解。

## Joe

使用PyTorch更新这本书的代码是一次愉快而令人兴奋的经历。像这样的努力是一个人无法完成的。首先，我想感谢PyTorch社区及其2100多名贡献者，他们不断发展和改进PyTorch及其深度学习能力。正是因为有你们，我们才能展示本书中描述的概念。

我永远感激Rebecca Novack让我参与这个项目，并对我作为作者的信任。非常感谢Melissa Potter和O'Reilly制作人员让这个更新版本得以实现。

我要感谢Matt Kirk的鼓励和支持。在这一切中，他一直是我的支柱。感谢我们无数次充满想法和资源的聊天。

特别感谢我的孩子们，Savannah、Caroline、George和Forrest，在爸爸工作时耐心理解。最重要的是，感谢我的妻子Emily，她一直支持我的梦想。在我努力写代码的时候，她照顾我们的新生儿，整夜不眠，同时确保“大”孩子们的需求也得到满足。没有她，我对这个项目的贡献是不可能的。
