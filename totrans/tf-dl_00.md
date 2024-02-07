# 前言

本书将通过 TensorFlow 向您介绍机器学习的基础知识。TensorFlow 是谷歌的新软件库，用于深度学习，使工程师能够设计和部署复杂的深度学习架构变得简单。您将学习如何使用 TensorFlow 构建能够检测图像中的对象、理解人类文本以及预测潜在药物属性的系统。此外，您将直观地了解 TensorFlow 作为执行张量计算的系统的潜力，并学习如何将 TensorFlow 用于传统机器学习范围之外的任务。

重要的是，《TensorFlow for Deep Learning》是为从业者编写的首批深度学习书籍之一。它通过实际示例教授基本概念，并从基础开始建立对机器学习基础的理解。本书的目标读者是实践开发人员，他们擅长设计软件系统，但不一定擅长创建学习系统。有时我们会使用一些基本的线性代数和微积分，但我们将复习所有必要的基础知识。我们还预计我们的书将对熟悉脚本编写但不一定擅长设计学习算法的科学家和其他专业人士有所帮助。

# 本书中使用的约定

本书中使用以下排版约定：

*斜体*

指示新术语、URL、电子邮件地址、文件名和文件扩展名。

`等宽`

用于程序清单，以及在段落中引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

**`等宽粗体`**

显示用户应直接输入的命令或其他文本。

*`等宽斜体`*

显示应由用户提供值或由上下文确定值替换的文本。

###### 提示

此元素表示提示或建议。

###### 注意

此元素表示一般说明。

###### 警告

此元素表示警告或注意事项。

# 使用代码示例

可下载补充材料（代码示例、练习等）位于[*https://github.com/matroid/dlwithtf*](https://github.com/matroid/dlwithtf)。

本书旨在帮助您完成工作。一般来说，如果本书提供示例代码，您可以在您的程序和文档中使用它。除非您复制了代码的大部分内容，否则无需联系我们请求许可。例如，编写一个使用本书中几个代码块的程序不需要许可。出售或分发包含 O’Reilly 图书示例的 CD-ROM 需要许可。通过引用本书回答问题并引用示例代码不需要许可。将本书中大量示例代码整合到产品文档中需要许可。

我们感谢，但不要求署名。署名通常包括标题、作者、出版商和 ISBN。例如：“*TensorFlow for Deep Learning* by Bharath Ramsundar and Reza Bosagh Zadeh (O’Reilly). Copyright 2018 Reza Zadeh, Bharath Ramsundar, 978-1-491-98045-3.”

如果您觉得您使用的代码示例超出了合理使用范围或上述给出的许可，请随时通过*permissions@oreilly.com*与我们联系。

# O’Reilly Safari

###### 注意

[*Safari*](http://oreilly.com/safari)（前身为 Safari Books Online）是面向企业、政府、教育者和个人的基于会员制的培训和参考平台。

会员可以访问来自 250 多家出版商的数千本书籍、培训视频、学习路径、交互式教程和策划播放列表，包括 O'Reilly Media、哈佛商业评论、Prentice Hall 专业、Addison-Wesley 专业、微软出版社、Sams、Que、Peachpit Press、Adobe、Focal Press、思科出版社、约翰·威利与儿子、Syngress、摩根·考夫曼、IBM 红皮书、Packt、Adobe 出版社、FT 出版社、Apress、Manning、New Riders、麦格劳希尔、琼斯与巴特利特以及 Course Technology 等。

更多信息，请访问[*http://oreilly.com/safari*](http://oreilly.com/safari)。

# 如何联系我们

请将有关此书的评论和问题发送至出版商：

+   O'Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-998-9938（美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们为这本书创建了一个网页，列出勘误、示例和任何其他信息。您可以访问此页面：[*http://bit.ly/tensorflowForDeepLearning*](http://bit.ly/tensorflowForDeepLearning)。

要就此书发表评论或提出技术问题，请发送电子邮件至*bookquestions@oreilly.com*。

有关我们的图书、课程、会议和新闻的更多信息，请访问我们的网站：[*http://www.oreilly.com*](http://www.oreilly.com)。

在 Facebook 上找到我们：[*http://facebook.com/oreilly*](http://facebook.com/oreilly)

在 Twitter 上关注我们：[*http://twitter.com/oreillymedia*](http://twitter.com/oreillymedia)

在 YouTube 上观看我们：[*http://www.youtube.com/oreillymedia*](http://www.youtube.com/oreillymedia)

# 致谢

Bharath 感谢他的博士导师在晚上和周末让他工作在这本书上，并特别感谢他的家人在整个过程中给予的大力支持。

Reza 感谢开源社区，许多软件和计算机科学都是基于这些社区。开源软件是人类知识的最大集中之一，没有整个社区的支持，这本书是不可能的。
