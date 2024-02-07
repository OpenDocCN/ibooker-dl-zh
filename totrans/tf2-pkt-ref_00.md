# 前言

TensorFlow 生态系统已经发展成许多不同的框架，以服务各种角色和功能。这种灵活性是其被广泛采用的原因之一，但也增加了数据科学家、机器学习（ML）工程师和其他技术利益相关者的学习曲线。有很多方法可以管理 TensorFlow 模型用于常见任务，比如数据和特征工程、数据摄取、模型选择、训练模式、交叉验证防止过拟合以及部署策略，选择可能会让人感到不知所措。

这本便携参考书将帮助您在 TensorFlow 中做出选择，包括如何使用 Python 中的 TensorFlow 2.0 设计模式设置常见的数据科学和 ML 工作流程。示例描述和演示了 TensorFlow 编码模式和您在 ML 项目工作中可能经常遇到的其他任务。您可以将其用作操作指南和参考书。

本书适用于当前和潜在的 ML 工程师、数据科学家和企业 ML 解决方案架构师，他们希望在 TensorFlow 建模中的可重用模式和最佳实践方面提升知识和经验。也许您已经阅读过一本介绍性的 TensorFlow 书籍，并且对数据科学领域保持了解。本书假定您具有使用 Python（可能还有 NumPy、pandas 和 JSON 库）进行数据工程、特征工程例程和构建 TensorFlow 模型的实际经验。熟悉常见数据结构，如列表、字典和 NumPy 数组，也将非常有帮助。

与许多其他 TensorFlow 书籍不同，本书围绕您可能需要执行的任务进行结构化，比如：

+   何时以及为什么应该将训练数据作为 NumPy 数组或流式数据集传递？（第 2 和 5 章）

+   如何利用预训练模型进行迁移学习？（第 3 和 4 章）

+   您应该使用通用的 fit 函数进行训练，还是编写自定义训练循环？（第六章）

+   您应该如何管理和利用模型检查点？（第七章）

+   如何使用 TensorBoard 审查训练过程？（第七章）

+   如果你的数据无法全部放入运行时的内存中，你如何使用多个加速器（如 GPU）进行分布式训练？（第八章）

+   在推断期间如何将数据传递给模型以及如何处理输出？（第九章）

+   您的模型是否公平？（第十章）

如果您正在处理这类问题，本书将对您有所帮助。

# 本书使用的约定

本书使用以下排版约定：

*斜体*

指示新术语、URL、电子邮件地址、文件名和文件扩展名。

`等宽字体`

用于程序清单，以及在段落中引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

**`等宽粗体`**

显示用户应该按照字面意思输入的命令或其他文本。

*`等宽斜体`*

显示应该用用户提供的值或上下文确定的值替换的文本。

###### 提示

这个元素表示提示或建议。

# 使用代码示例

补充材料（代码示例、练习等）可在[*https://github.com/shinchan75034/tensorflow-pocket-ref*](https://github.com/shinchan75034/tensorflow-pocket-ref)下载。

如果您有技术问题或在使用代码示例时遇到问题，请发送电子邮件至*bookquestions@oreilly.com*。

这本书旨在帮助您完成工作。一般来说，如果本书提供示例代码，您可以在程序和文档中使用它。除非您复制了代码的大部分内容，否则无需联系我们以获得许可。例如，编写一个程序使用本书中的几个代码块不需要许可。销售或分发 O'Reilly 图书中的示例需要许可。通过引用本书回答问题并引用示例代码不需要许可。将本书中大量示例代码整合到产品文档中需要许可。

我们感谢，但通常不要求署名。署名通常包括标题、作者、出版商和 ISBN。例如：“*TensorFlow 2 Pocket Reference* by KC Jung (O’Reilly). Copyright 2021 Favola Vera, LLC, 978-1-492-08918-6.”

如果您认为您对代码示例的使用超出了合理使用范围或上述许可，请随时通过*permissions@oreilly.com*与我们联系。

# O’Reilly 在线学习

###### 注意

40 多年来，[*O’Reilly Media*](http://oreilly.com)提供技术和商业培训、知识和见解，帮助公司取得成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专长。O'Reilly 的在线学习平台为您提供按需访问实时培训课程、深入学习路径、交互式编码环境以及来自 O'Reilly 和其他 200 多家出版商的大量文本和视频。欲了解更多信息，请访问[*http://oreilly.com*](http://oreilly.com)。

# 如何联系我们

请将有关本书的评论和问题发送至出版商：

+   O’Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-998-9938（美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们为本书设有网页，列出勘误、示例和任何其他信息。您可以在[*https://oreil.ly/tensorflow2pr*](https://oreil.ly/tensorflow2pr)访问此页面。

发送电子邮件至*bookquestions@oreilly.com*以评论或提出关于本书的技术问题。

有关我们的图书和课程的新闻和信息，请访问[*http://oreilly.com*](http://oreilly.com)。

在 Facebook 上找到我们：[*http://facebook.com/oreilly*](http://facebook.com/oreilly)

在 Twitter 上关注我们：[*http://twitter.com/oreillymedia*](http://twitter.com/oreillymedia)

在 YouTube 上关注我们：[*http://youtube.com/oreillymedia*](http://youtube.com/oreillymedia)

# 致谢

我非常感谢 O'Reilly 编辑们的周到和专业工作。此外，我还要感谢技术审阅者：Tony Holdroyd、Pablo Marin、Giorgio Saez 和 Axel Sirota，感谢他们宝贵的反馈和建议。最后，特别感谢 Rebecca Novack 和 Sarah Grey 给了我一个机会，并与我合作撰写本书。
