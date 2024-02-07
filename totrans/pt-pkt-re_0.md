# 前言

我们生活在激动人心的时代！我们中的一些人有幸经历了技术的巨大进步——个人计算机的发明，互联网的兴起，手机的普及以及社交媒体的出现。现在，人工智能领域正在发生重大突破！

看到并参与这种变革是令人兴奋的。我认为我们才刚刚开始，想到未来十年世界可能会发生怎样的变化，这是令人惊奇的。我们能够生活在这个时代并参与人工智能的扩展，这是多么伟大的事情？

毫无疑问，PyTorch已经实现了一些深度学习和人工智能领域的最好进展。它是免费下载和使用的，任何拥有计算机或互联网连接的人都可以运行人工智能实验。除了像这样更全面的参考资料外，还有许多免费和廉价的培训课程、博客文章和教程可以帮助您。任何人都可以开始使用PyTorch进行机器学习和人工智能。

# 谁应该阅读这本书

这本书是为对机器学习和人工智能感兴趣的初学者和高级用户编写的。最好具有一些编写Python代码的经验以及对数据科学和机器学习的基本理解。

如果您刚开始学习机器学习，这本书将帮助您学习PyTorch的基础知识并提供一些简单的示例。如果您已经使用其他框架，如TensorFlow、Caffe2或MXNet，这本书将帮助您熟悉PyTorch的API和编程思维方式，以便扩展您的技能。

如果您已经使用PyTorch一段时间，这本书将帮助您扩展对加速和优化等高级主题的知识，并在您日常开发中使用PyTorch时提供快速参考资源。

# 我为什么写这本书

学习和掌握PyTorch可能会非常令人兴奋。有很多东西可以探索！当我开始学习PyTorch时，我希望有一个资源可以教会我一切。我想要一些东西，可以让我对PyTorch提供的内容有一个很好的高层次了解，但也可以在我需要深入了解时提供示例和足够的细节。

有一些关于PyTorch的很好的书籍和课程，但它们通常侧重于张量和深度学习模型的训练。PyTorch的在线文档也非常好，并提供了许多细节和示例；然而，我发现使用它通常很麻烦。我不断地不得不点击以学习或谷歌我需要知道的内容。我需要一本书放在桌子上，我可以标记并在编码时作为参考。

我的目标是这将是您的终极PyTorch参考资料。除了通读以获取对PyTorch资源的高层次理解之外，我希望您为您的开发工作标记关键部分并将其放在桌子上。这样，如果您忘记了某些内容，您可以立即得到答案。如果您更喜欢电子书或在线书籍，您可以在线收藏本书。无论您如何使用，我希望这本书能帮助您用PyTorch创建一些令人惊人的新技术！

# 浏览本书

如果您刚开始学习PyTorch，您应该从第1章开始，并按顺序阅读每一章。这些章节从初学者到高级主题逐渐展开。如果您已经有一些PyTorch经验，您可能想跳到您最感兴趣的主题。不要忘记查看关于PyTorch生态系统的第8章。您一定会发现一些新东西！

这本书大致组织如下：

+   第1章简要介绍了PyTorch，帮助您设置开发环境，并提供一个有趣的示例供您尝试。

+   第2章介绍了张量，PyTorch的基本构建块。这是PyTorch中一切的基础。

+   第[3](ch03.xhtml#deep_learning_development_with_pytorch)章全面介绍了您如何使用PyTorch进行深度学习，而第[4](ch04.xhtml#neural_network_development_reference_designs)章提供了示例参考设计，让您可以看到PyTorch的实际应用。

+   第[5](ch05.xhtml#Chapter_5)章和第[6](ch06.xhtml#pytorth_acceleration_and_optimization)章涵盖了更高级的主题。第[5](ch05.xhtml#Chapter_5)章向您展示了如何自定义PyTorch组件以适应您自己的工作，而第[6](ch06.xhtml#pytorth_acceleration_and_optimization)章则向您展示了如何加速训练并优化您的模型。

+   第[7](ch07.xhtml#deploying_pytorch_to_production)章向您展示如何通过本地机器、云服务器和移动或边缘设备将PyTorch部署到生产环境。

+   第[8](ch08.xhtml#pytorch_ecosystem_and_additional_resources)章引导您下一步，介绍PyTorch生态系统，描述流行的软件包，并列出额外的培训资源。

# 本书中使用的约定

本书使用以下印刷约定：

*斜体*

表示新术语、URL、电子邮件地址、文件名和文件扩展名。

`等宽字体`

用于程序清单，以及在段落中引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

**`等宽粗体`**

显示用户应该按照字面意义输入的命令或其他文本。此外，在表格中，粗体用于强调函数。

*`等宽斜体`*

显示应该用用户提供的值或上下文确定的值替换的文本。此外，在表格中列出的转换目前不受TorchScript支持。

# 使用代码示例

补充材料（代码示例、练习等）可在[*https://github.com/joe-papa/pytorch-book*](https://github.com/joe-papa/pytorch-book)下载。

如果您有技术问题或在使用代码示例时遇到问题，请发送电子邮件至[*bookquestions@oreilly.com*](mailto:bookquestions@oreilly.com)。

本书旨在帮助您完成工作。一般来说，如果本书提供示例代码，您可以在您的程序和文档中使用它。除非您复制了代码的大部分内容，否则无需征得我们的许可。例如，编写一个使用本书中几个代码块的程序不需要许可。销售或分发O'Reilly图书中的示例需要许可。通过引用本书回答问题并引用示例代码不需要许可。将本书中大量示例代码合并到产品文档中需要许可。

我们感谢，但通常不要求署名。署名通常包括标题、作者、出版商和ISBN。例如：“*PyTorch口袋参考* 作者Joe Papa（O'Reilly）。版权所有2021年Mobile Insights Technology Group，LLC，978-1-492-09000-7。”

如果您认为您对代码示例的使用超出了合理使用范围或上述许可，请随时通过[*permissions@oreilly.com*](mailto:permissions@oreilly.com)与我们联系。

# O'Reilly在线学习

###### 注意

40多年来，[*O'Reilly Media*](http://oreilly.com)提供技术和商业培训、知识和见解，帮助公司取得成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专长。O'Reilly的在线学习平台为您提供按需访问实时培训课程、深入学习路径、交互式编码环境以及来自O'Reilly和其他200多家出版商的大量文本和视频。有关更多信息，请访问[*http://oreilly.com*](http://oreilly.com)。

# 如何联系我们

请将有关本书的评论和问题发送给出版商：

+   O'Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-998-9938（在美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们为这本书建立了一个网页，列出勘误、示例和任何额外信息。您可以访问[*https://oreil.ly/PyTorch-pocket*](https://oreil.ly/PyTorch-pocket)。

发送电子邮件至[*bookquestions@oreilly.com*](mailto:bookquestions@oreilly.com)评论或询问有关本书的技术问题。

关于我们的图书和课程的新闻和信息，请访问[*http://oreilly.com*](http://oreilly.com)。

在Facebook上找到我们：[*http://facebook.com/oreilly*](http://facebook.com/oreilly)

在Twitter上关注我们：[*http://twitter.com/oreillymedia*](http://twitter.com/oreillymedia)

在YouTube上观看我们：[*http://youtube.com/oreillymedia*](http://youtube.com/oreillymedia)

# 致谢

作为读者，我经常在阅读其他作者的致谢时感到惊讶。写一本书并不是一件小事，写一本好书需要许多人的支持。阅读致谢是一个不断提醒我们不能独自完成的过程。

我感谢我的朋友Matt Kirk的支持和鼓励，多年前在O'Reilly会议上认识他。他对个人发展的共同热情激励着我创作图书和课程，帮助他人充分发挥个人和职业潜力。在疫情期间，我们每周的Zoom聊天和自助项目确实帮助我保持理智。没有Matt，这本书就不可能完成。

我要感谢Rebecca Novack建议这个项目并给我机会，以及O'Reilly的工作人员让这个项目得以实现。

写一本书需要努力，但写一本好书需要关心读者的专注审阅者。我要感谢Mike Drob、Axel Sirota和Jeff Bleiel花时间审查这本书并提供无数建议。Mike的建议增加了许多实用资源，否则我可能会忽视。他确保我们使用的是最先进的工具和您在在线文档中找不到的最佳实践。

Axel对细节的关注令人难以置信。我感谢他对本书代码和技术细节的审查和努力。Jeff是一位出色的编辑。我感谢他对本书的顺序和流程提出的建议。他显著帮助我成为一个更好的作者。

PyTorch真正是一个社区项目。我感谢Facebook和超过1700名贡献者开发了这个机器学习框架。我特别要感谢那些创建文档和教程来帮助像我这样的人快速学习PyTorch的人。

对我帮助最大的一些人包括Suraj Subramanian、Seth Juarez、Cassie Breviu、Dmitry Soshnikov、Ari Bornstein、Soumith Chintala、Justin Johnson、Jeremy Howard、Rachel Thomas、Francisco Ingham、Sasank Chilamkurthy、Nathan Inkawhich、Sean Robertson、Ben Trevett、Avinash Sajjanshetty、James Reed、Michael Suo、Michela Paganini、Shen Li、Séb Arnold、Rohan Varma、Pritam Damania、Jeff Tang，以及关于PyTorch主题的无数博主和YouTuber。

我感谢Manbir Gulati介绍我认识PyTorch，感谢Rob Miller给我机会领导PyTorch的AI项目。我也感谢与我的朋友Isaac Privitera分享这本书的深度学习思想。

当然，没有我妈妈Grace的辛勤工作和奉献精神，我在生活中无法取得任何成就，她带领我们从不起眼的开始，给了我和我哥哥一个生活的机会。我每天都在想念她。

特别感谢我的哥哥文尼，在完成家庭项目时给予了很大帮助，让我有更多时间写作。我感激我的继父卢，在我写书时给予的鼓励。我还要感谢我的孩子们，萨凡娜、卡罗琳和乔治，在爸爸工作时耐心理解。

最后，我想感谢我的妻子艾米莉。她一直无限支持我的想法和梦想。当我着手写这本书的任务时，当然又一次依靠了她。在疫情期间照顾我们的三个孩子并承担新的责任是一项艰巨的任务。

然而，她一直是我完成写作所需的支持。事实上，在写这本书的过程中，我们发现我们正在期待第四个孩子的到来！我的妻子总是带着微笑和笑话（通常是拿我开玩笑），我很爱她。
