# 前言

深度学习在过去几年中已经成为构建从数据中学习的智能系统的首要技术。深度神经网络最初受到人类大脑学习方式的粗略启发，通过大量数据训练以解决具有前所未有准确度的复杂任务。随着开源框架使这项技术广泛可用，它已成为任何涉及大数据和机器学习的人必须掌握的技能。

TensorFlow目前是领先的深度学习开源软件，被越来越多的从业者用于计算机视觉、自然语言处理（NLP）、语音识别和一般预测分析。

本书是一本针对数据科学家、工程师、学生和研究人员设计的TensorFlow端到端指南。本书采用适合广泛技术受众的实践方法，使初学者可以轻松入门，同时深入探讨高级主题，并展示如何构建生产就绪系统。

在本书中，您将学习如何：

1.  快速轻松地开始使用TensorFlow。

1.  使用TensorFlow从头开始构建模型。

1.  训练和理解计算机视觉和NLP中流行的深度学习模型。

1.  使用广泛的抽象库使开发更加简单和快速。

1.  通过排队和多线程、在集群上训练和在生产中提供输出来扩展TensorFlow。

1.  还有更多！

本书由具有丰富工业和学术研究经验的数据科学家撰写。作者采用实用直观的例子、插图和见解，结合实践方法，适合寻求构建生产就绪系统的从业者，以及希望学习理解和构建灵活强大模型的读者。

# 先决条件

本书假设读者具有一些基本的Python编程知识，包括对科学库NumPy的基本了解。

本书中涉及并直观解释了机器学习概念。对于希望深入了解的读者，建议具有一定水平的机器学习、线性代数、微积分、概率和统计知识。

# 本书使用的约定

本书中使用以下排版约定：

*斜体*

指示新术语、URL、电子邮件地址、文件名和文件扩展名。

`常量宽度`

用于程序清单，以及在段落中引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

**`常量宽度粗体`**

显示用户应该按原样输入的命令或其他文本。

*`常量宽度斜体`*

显示应替换为用户提供的值或由上下文确定的值的文本。

# 使用代码示例

可下载补充材料（代码示例、练习等）[*https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow*](https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow)。

本书旨在帮助您完成工作。一般来说，如果本书提供了示例代码，您可以在您的程序和文档中使用它。除非您要复制代码的大部分内容，否则无需征得我们的许可。例如，编写一个使用本书中几个代码块的程序不需要许可。出售或分发包含O'Reilly图书示例的CD-ROM需要许可。回答问题并引用本书中的示例代码不需要许可。将本书中大量示例代码合并到产品文档中需要许可。

我们感激，但不要求署名。署名通常包括标题、作者、出版商和ISBN。例如：“*Learning TensorFlow* by Tom Hope, Yehezkel S. Resheff, and Itay Lieder (O’Reilly). Copyright 2017 Tom Hope, Itay Lieder, and Yehezkel S. Resheff, 978-1-491-97851-1.”

如果您觉得您对代码示例的使用超出了合理使用范围或上述许可，请随时通过[*permissions@oreilly.com*](mailto:permissions@oreilly.com)与我们联系。

# O'Reilly Safari

###### 注意

[*Safari*](http://oreilly.com/safari)（原Safari Books Online）是一个面向企业、政府、教育工作者和个人的基于会员制的培训和参考平台。

会员可以访问来自250多家出版商的数千本图书、培训视频、学习路径、交互式教程和策划播放列表，包括O'Reilly Media、哈佛商业评论、Prentice Hall专业、Addison-Wesley专业、微软出版社、Sams、Que、Peachpit出版社、Adobe、Focal Press、思科出版社、约翰威利与儿子、Syngress、摩根考夫曼、IBM红皮书、Packt、Adobe出版社、FT出版社、Apress、Manning、New Riders、麦格劳希尔、琼斯与巴特利特以及课程技术等。

有关更多信息，请访问[*http://oreilly.com/safari*](http://www.oreilly.com/safari)。

# 如何联系我们

请将有关本书的评论和问题发送至出版商：

+   O'Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-998-9938（美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们为这本书创建了一个网页，列出勘误、示例和任何额外信息。您可以在[*http://bit.ly/learning-tensorflow*](http://bit.ly/learning-tensorflow)上访问这个页面。

要评论或提出关于这本书的技术问题，请发送电子邮件至[*bookquestions@oreilly.com*](mailto:bookquestions@oreilly.com)。

有关我们的图书、课程、会议和新闻的更多信息，请访问我们的网站[*http://www.oreilly.com*](http://www.oreilly.com)。

在Facebook上找到我们：[*http://facebook.com/oreilly*](http://facebook.com/oreilly)

在Twitter上关注我们：[*http://twitter.com/oreillymedia*](http://twitter.com/oreillymedia)

在YouTube上关注我们：[*http://www.youtube.com/oreillymedia*](http://www.youtube.com/oreillymedia)

# 致谢

作者要感谢为这本书提供反馈意见的审阅者：Chris Fregly、Marvin Bertin、Oren Sar Shalom和Yoni Lavi。我们还要感谢Nicole Tache和O'Reilly团队，使写作这本书成为一种乐趣。

当然，感谢所有在Google工作的人，没有他们就不会有TensorFlow。
