# 前言

近年来，生命科学和数据科学已经融合。机器人技术和自动化的进步使化学家和生物学家能够产生大量数据。如今，科学家们一天内能够生成的数据量比 20 年前的前辈整个职业生涯中能够生成的数据量还要多。这种快速生成数据的能力也带来了许多新的科学挑战。我们不再处于将数据加载到电子表格中并制作几张图表的时代。为了从这些数据集中提炼科学知识，我们必须能够识别和提取非显而易见的关系。

在过去几年中出现的一种强大的工具是*深度学习*，这是一类算法，已经彻底改变了解决问题的方法，如图像分析、语言翻译和语音识别。深度学习算法擅长识别和利用大型数据集中的模式。出于这些原因，深度学习在生命科学领域有广泛的应用。本书概述了深度学习在遗传学、药物发现和医学诊断等领域的应用。我们描述的许多示例都附带有代码示例，为读者提供了对方法的实际介绍，并为未来的研究和探索提供了起点。

# 本书中使用的约定

本书中使用以下排版约定：

*斜体*

指示新术语、URL、电子邮件地址、文件名和文件扩展名。

`固定宽度`

用于程序清单，以及在段落中引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

**`粗体固定宽度`**

显示用户应该按照字面意思输入的命令或其他文本。

*`斜体固定宽度`*

显示应该由用户提供值或由上下文确定值替换的文本。

###### 提示

此元素表示提示或建议。

###### 注意

此元素表示一般说明。

###### 警告

此元素表示警告或注意事项。

# 使用代码示例

附加材料（代码示例、练习等）可在[*https://github.com/deepchem/DeepLearningLifeSciences*](https://github.com/deepchem/DeepLearningLifeSciences)下载。

这本书旨在帮助您完成工作。一般来说，如果本书提供了示例代码，您可以在您的程序和文档中使用它。除非您复制了代码的大部分内容，否则无需征得我们的许可。例如，编写一个使用本书中几个代码块的程序不需要许可。出售或分发包含 O’Reilly 图书示例的 CD-ROM 需要许可。引用本书并引用示例代码回答问题不需要许可。将本书中大量示例代码合并到产品文档中需要许可。

我们感谢，但不要求署名。署名通常包括标题、作者、出版商和 ISBN。例如：“*生命科学的深度学习*，作者 Bharath Ramsundar、Peter Eastman、Patrick Walters 和 Vijay Pande（O’Reilly）。版权 2019 年 Bharath Ramsundar、Karl Leswing、Peter Eastman 和 Vijay Pande，978-1-492-03983-9。”

如果您认为您使用的代码示例超出了合理使用范围或上述许可，请随时通过*permissions@oreilly.com*与我们联系。

# O’Reilly 在线学习

###### 注意

近 40 年来，[*O’Reilly*](http://oreilly.com)提供技术和商业培训、知识和见解，帮助公司取得成功。

我们独特的专家和创新者网络通过图书、文章、会议和我们的在线学习平台分享他们的知识和专长。O’Reilly 的在线学习平台为您提供按需访问实时培训课程、深入学习路径、交互式编码环境以及来自 O’Reilly 和其他 200 多家出版商的大量文本和视频。欲了解更多信息，请访问 [`oreilly.com`](http://www.oreilly.com)。

# 如何联系我们

请将有关此书的评论和问题发送给出版商：

+   O’Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-998-9938（美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们为这本书创建了一个网页，列出勘误、示例和任何额外信息。您可以在 [`bit.ly/deep-lrng-for-life-science`](http://bit.ly/deep-lrng-for-life-science) 访问此页面。

要就此书发表评论或提出技术问题，请发送电子邮件至 bookquestions@oreilly.com。

关于我们的图书、课程、会议和新闻的更多信息，请访问我们的网站：[`www.oreilly.com`](http://www.oreilly.com)。

在 Facebook 上找到我们：[`facebook.com/oreilly`](http://facebook.com/oreilly)

在 Twitter 上关注我们：[`twitter.com/oreillymedia`](http://twitter.com/oreillymedia)

在 YouTube 上观看我们：[`www.youtube.com/oreillymedia`](http://www.youtube.com/oreillymedia)

# 致谢

我们要感谢 Nicole Tache，我们在 O’Reilly 的编辑，以及技术审阅员和测试审阅员对本书的宝贵贡献。此外，我们还要感谢 Karl Leswing 和 Zhenqin（Michael）Wu 对代码的贡献，以及 Johnny Israeli 对基因组学章节的宝贵建议。

Bharath 感谢他的家人在许多漫长的周末和夜晚为这本书工作时给予的支持和鼓励。

Peter 感谢他的妻子一直以来的支持，以及许多同事们对机器学习的教导。

Pat 感谢他的妻子 Andrea，以及他的女儿 Alee 和 Maddy，感谢她们的爱和支持。他还要感谢 Vertex Pharmaceuticals 和 Relay Therapeutics 的过去和现在的同事们，从他们那里学到了很多。

最后，我们要感谢 DeepChem 开源社区在整个项目期间给予的鼓励和支持。
