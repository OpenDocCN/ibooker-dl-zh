# 前言

长久以来，电子产品一直吸引着我的想象力。我们学会了从地球中挖掘岩石，以神秘的方式对其进行精炼，并生产出令人眼花缭乱的微小组件，我们根据神秘的法则将它们组合在一起，赋予它们一些生命的本质。

在我八岁的时候，电池、开关和灯丝灯已经足够迷人了，更不用说我家里家用电脑内部的处理器了。随着岁月的流逝，我对电子和软件原理有了一些了解，使这些发明能够运行。但总让我印象深刻的是，一系列简单的元素如何组合在一起创造出一个微妙而复杂的东西，而深度学习真的将这一点推向了新的高度。

这本书的一个例子是一个深度学习网络，从某种意义上说，它懂得如何看。它由成千上万个虚拟的“神经元”组成，每个神经元都遵循一些简单的规则并输出一个数字。单独来看，每个神经元并不能做太多事情，但是结合起来，并且通过训练，给予一点人类知识，它们可以理解我们复杂的世界。

这个想法中有一些魔力：简单的算法在由沙子、金属和塑料制成的微型计算机上运行，可以体现出人类理解的一部分。这就是 TinyML 的本质，这是 Pete 创造的一个术语，将在第一章中介绍。在本书的页面中，您将找到构建这些东西所需的工具。

感谢您成为我们的读者。这是一个复杂的主题，但我们努力保持简单并解释您需要的所有概念。我们希望您喜欢我们所写的内容，我们很期待看到您创造的东西！

Daniel Situnayake

# 本书中使用的约定

本书中使用了以下排版约定：

*斜体*

指示新术语、URL、电子邮件地址、文件名和文件扩展名。

`常量宽度`

用于程序清单，以及在段落中引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

**`常量宽度粗体`**

显示用户应该按原样输入的命令或其他文本。

*`常量宽度斜体`*

显示应该用用户提供的值或由上下文确定的值替换的文本。

###### 提示

这个元素表示提示或建议。

###### 注意

这个元素表示一般注释。

###### 警告

这个元素表示警告或注意。

# 使用代码示例

补充材料（代码示例、练习等）可在[*https://tinymlbook.com/supplemental*](https://tinymlbook.com/supplemental)下载。

如果您有技术问题或在使用代码示例时遇到问题，请发送电子邮件至*bookquestions@oreilly.com*。

这本书旨在帮助您完成工作。一般来说，如果本书提供了示例代码，您可以在您的程序和文档中使用它。除非您复制了大量代码的部分，否则无需联系我们请求许可。例如，编写一个使用本书中几个代码块的程序不需要许可。销售或分发 O’Reilly 书籍中的示例需要许可。通过引用本书回答问题并引用示例代码不需要许可。将本书中大量示例代码合并到产品文档中需要许可。

我们感谢，但通常不要求署名。署名通常包括标题、作者、出版商和 ISBN。例如：“*TinyML* by Pete Warden and Daniel Situnayake (O’Reilly). Copyright Pete Warden and Daniel Situnayake, 978-1-492-05204-3.”

如果您觉得您对代码示例的使用超出了合理使用范围或上述给出的许可，请随时联系我们，邮箱为*permissions@oreilly.com*。

# O’Reilly 在线学习

###### 注意

[*O'Reilly Media*](http://oreilly.com)已经提供技术和商业培训、知识和见解帮助公司成功超过 40 年。

我们独特的专家和创新者网络通过书籍、文章、会议和我们的在线学习平台分享他们的知识和专长。O'Reilly 的在线学习平台为您提供按需访问实时培训课程、深入学习路径、交互式编码环境以及来自 O'Reilly 和其他 200 多家出版商的大量文本和视频。欲了解更多信息，请访问[*http://oreilly.com*](http://oreilly.com)。

# 如何联系我们

请将有关此书的评论和问题发送给出版商：

+   O'Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-998-9938（在美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们为这本书创建了一个网页，列出勘误、示例和任何其他信息。您可以访问[*https://oreil.ly/tiny*](https://oreil.ly/tiny)。

发送电子邮件至*tinyml-book@googlegroups.com*评论或提出有关此书的技术问题。

有关我们的书籍、课程、会议和新闻的更多信息，请访问我们的网站[*http://www.oreilly.com*](http://www.oreilly.com)。

在 Facebook 上找到我们：[*http://facebook.com/oreilly*](http://facebook.com/oreilly)

在 Twitter 上关注我们：[*http://twitter.com/oreillymedia*](http://twitter.com/oreillymedia)

在 YouTube 上观看我们：[*http://www.youtube.com/oreillymedia*](http://www.youtube.com/oreillymedia)

# 致谢

我们要特别感谢 Nicole Tache 出色的编辑工作，Jennifer Wang 的启发性魔杖示例，以及 Neil Tan 在 uTensor 库中进行的开创性嵌入式 ML 工作。没有 Rajat Monga 和 Sarah Sirajuddin 的专业支持，我们无法完成这本书的写作。我们还要感谢我们的合作伙伴 Joanne Ladolcetta 和 Lauren Ward 的耐心。

这本书是来自硬件、软件和研究领域数百人的努力成果，特别是来自 TensorFlow 团队。虽然我们只能提及一部分人，对于我们遗漏的每个人表示歉意，我们要感谢：Mehmet Ali Anil，Alasdair Allan，Raziel Alvarez，Paige Bailey，Massimo Banzi，Raj Batra，Mary Bennion，Jeff Bier，Lukas Biewald，Ian Bratt，Laurence Campbell，Andrew Cavanaugh，Lawrence Chan，Vikas Chandra，Marcus Chang，Tony Chiang，Aakanksha Chowdhery，Rod Crawford，Robert David，Tim Davis，Hongyang Deng，Wolff Dobson，Jared Duke，Jens Elofsson，Johan Euphrosine，Martino Facchin，Limor Fried，Nupur Garg，Nicholas Gillian，Evgeni Gousev，Alessandro Grande，Song Han，Justin Hong，Sara Hooker，Andrew Howard，Magnus Hyttsten，Advait Jain，Nat Jeffries，Michael Jones，Mat Kelcey，Kurt Keutzer，Fredrik Knutsson，Nick Kreeger，Nic Lane，Shuangfeng Li，Mike Liang，Yu-Cheng Ling，Renjie Liu，Mike Loukides，Owen Lyke，Cristian Maglie，Bill Mark，Matthew Mattina，Sandeep Mistry，Amit Mittra，Laurence Moroney，Boris Murmann，Ian Nappier，Meghna Natraj，Ben Nuttall，Dominic Pajak，Dave Patterson，Dario Pennisi，Jahnell Pereira，Raaj Prasad，Frederic Rechtenstein，Vikas Reddi，Rocky Rhodes，David Rim，Kazunori Sato，Nathan Seidle，Andrew Selle，Arpit Shah，Marcus Shawcroft，Zach Shelby，Suharsh Sivakumar，Ravishankar Sivalingam，Rex St. John，Dominic Symes，Olivier Temam，Phillip Torrone，Stephan Uphoff，Eben Upton，Lu Wang，Tiezhen Wang，Paul Whatmough，Tom White，Edd Wilder-James 和 Wei Xiao。
