# 前言

生成式 AI 创新的快速步伐承诺将改变我们的生活和工作的方式，但跟上它的步伐变得越来越困难。在 arXiv 上发表的[AI 论文数量正在呈指数增长](https://oreil.ly/EN5ay)，[Stable Diffusion](https://oreil.ly/QX-yy)是历史上增长最快的开源项目之一，AI 艺术工具[Midjourney 的 Discord 服务器](https://oreil.ly/ZVZ5o)拥有数千万成员，甚至超过了最大的游戏社区。最吸引公众想象力的是 OpenAI 发布的 ChatGPT，[两个月内达到 1 亿用户](https://oreil.ly/FbYWk)，使其成为历史上增长最快的消费应用。学会与 AI 合作迅速成为最受欢迎的技能之一。

每个在专业上使用 AI 的人很快就会意识到，输出的质量在很大程度上取决于你提供的输入。*提示工程*这一学科已经形成了一套最佳实践，用于提高 AI 模型的可靠性、效率和准确性。“在十年内，世界上半数的工作将涉及提示工程”，[罗宾·李声称](https://oreil.ly/IdIfO)，他是中国科技巨头百度的联合创始人和首席执行官。然而，我们预计提示将成为许多工作所需的一项技能，类似于精通 Microsoft Excel，而不是一个流行的职位名称。这股新的颠覆浪潮正在改变我们关于计算机的所有认知。我们习惯于编写每次都能返回相同结果的算法——但对于 AI 来说并非如此，其响应是非确定性的。成本和延迟再次成为现实因素，几十年来摩尔定律让我们对几乎无成本的实时计算感到自满。最大的障碍是这些模型自信地编造事物，被称为*幻觉*，这让我们重新思考评估我们工作准确性的方式。

自 2020 年 GPT-3 测试版以来，我们一直在使用生成式 AI，随着我们观察到模型的发展，许多早期的提示技巧和黑客手段变得不再必要。随着时间的推移，一套一致的原则逐渐形成，这些原则在新的模型中仍然有用，并且适用于文本和图像生成。我们根据这些永恒的原则撰写了这本书，帮助您学习可迁移的技能，这些技能在未来五年内无论 AI 如何发展都将有用。与 AI 合作的关键不是“通过在末尾添加一个魔法词来破解提示，从而改变一切”，正如[OpenAI 联合创始人山姆·奥特曼断言](https://oreil.ly/oo262)的那样，但始终重要的是“想法的质量和对所求的理解。”虽然我们不知道五年后是否会称之为“提示工程”，但有效地与生成式 AI 合作将变得更加重要。

# 本书所需的软件要求

本书中的所有代码都是用 Python 编写的，并设计为在 [Jupyter Notebook](https://jupyter.org) 或 [Google Colab notebook](https://colab.research.google.com) 中运行。书中教授的概念可以转移到 JavaScript 或任何其他编程语言，尽管本书的主要重点是提示技巧而不是传统的编程技能。代码可以在 [GitHub](https://oreil.ly/BrightPool) 上找到，我们将在整个书中链接到相关的笔记本。强烈建议您使用 [GitHub 仓库](https://oreil.ly/BrightPool) 并在阅读本书时运行提供的示例。

对于非笔记本示例，您可以在终端中使用 `python content/chapter_x/script.py` 的格式运行脚本，其中 `x` 是章节号，`script.py` 是脚本的名称。在某些情况下，需要将 API 密钥设置为环境变量，我们将在适当的地方说明。使用的包会频繁更新，因此在运行代码示例之前，请在虚拟环境中安装我们的 [*requirements.txt*](https://oreil.ly/BPreq)。

*requirements.txt* 文件是为 Python 3.9 生成的。如果您想使用不同的 Python 版本，您可以从 GitHub 仓库中找到的此 [*requirements.in*](https://oreil.ly/YRwP7) 文件生成一个新的 *requirements.txt*，通过运行以下命令：

```py
`pip install pip-tools`
`pip-compile requirements.in`
```

对于 Mac 用户：

1.  打开终端：您可以在应用程序文件夹中的实用工具下找到终端应用程序，或者使用 Spotlight 搜索它。

1.  导航到您的项目文件夹：使用 `cd` 命令将目录切换到您的项目文件夹。例如：`cd path/to/your/project`。

1.  创建虚拟环境：使用以下命令创建名为 `venv` 的虚拟环境（您可以将其命名为任何名称）：`python3 -m venv venv`。

1.  激活虚拟环境：在安装包之前，您需要激活虚拟环境。使用命令 `source venv/bin/activate` 来完成此操作。

1.  安装包：现在您的虚拟环境已激活，您可以使用 `pip` 安装包。要从 *requirements.txt* 文件安装包，请使用 `pip install -r requirements.txt`。

1.  退出虚拟环境：完成工作后，您可以通过输入 **`deactivate`** 来退出虚拟环境。

对于 Windows 用户：

1.  打开命令提示符：您可以在开始菜单中搜索 `cmd`。

1.  导航到您的项目文件夹：使用 `cd` 命令将目录切换到您的项目文件夹。例如：`cd path\to\your\project`。

1.  创建虚拟环境：使用以下命令创建名为 `venv` 的虚拟环境：`python -m venv venv`。

1.  激活虚拟环境：在 Windows 上激活虚拟环境，请使用 `.\venv\Scripts\activate`。

1.  安装包：在虚拟环境激活状态下，安装所需的包：`pip install -r requirements.txt`。

1.  退出虚拟环境：要退出虚拟环境，只需输入：`deactivate`。

这里有一些关于设置的额外提示：

+   总是确保你的 Python 是最新的，以避免兼容性问题。

+   记得每次在项目上工作时都要激活你的虚拟环境。

+   `requirements.txt`文件应该位于你创建虚拟环境的同一目录中，或者当你使用`pip install -r`时，你应该指定其路径。

假设你有 OpenAI 开发者账户，因为你的`OPENAI_API_KEY`必须在导入 OpenAI 库的任何示例中设置为环境变量，我们使用的是版本 1.0。设置开发环境的快速入门指南可以在 OpenAI 网站上找到的[OpenAI 文档](https://oreil.ly/YqbrY)中。

你还必须确保你的 OpenAI 账户中启用了计费，并且已附加有效的支付方式以运行书中的一些代码。书中未特别说明时使用 GPT-4，尽管我们简要介绍了 Anthropic 的竞争产品[Claude 3 模型](https://oreil.ly/jY8Ai)，以及 Meta 的开源[Llama 3](https://oreil.ly/BbXZ3)和[Google Gemini](https://oreil.ly/KYgij)。

对于图像生成，我们使用[Midjourney](https://www.midjourney.com)，你需要一个 Discord 账户来注册，尽管这些原则同样适用于 DALL-E 3（通过 ChatGPT Plus 订阅或通过 API 获得）或 Stable Diffusion（作为[API](https://oreil.ly/cmTtW)提供，或者如果你的电脑有 GPU，它可以在本地运行[Stable Diffusion](https://oreil.ly/Ha0T5)）。本书中的图像生成示例使用 Midjourney v6，Stable Diffusion v1.5（因为许多扩展仍然只与这个版本兼容），或[Stable Diffusion XL](https://oreil.ly/S0P4s)，并且当这很重要时，我们会指定差异。

我们尽可能使用开源库提供示例，尽管在适当的情况下我们也包括商业供应商——例如，第五章关于向量数据库的章节展示了开源库 FAISS 和付费供应商 Pinecone。书中展示的示例应该可以轻松修改以适应不同的模型和供应商，并且所教授的技能是可迁移的。第四章关于高级文本生成专注于 LLM 框架 LangChain，而第九章关于高级图像生成基于 AUTOMATIC1111 的开源 Stable Diffusion Web UI。

# 本书使用的约定

本书使用的以下排版约定：

`斜体`

表示新术语、URL、电子邮件地址、文件名和文件扩展名。

`常宽字体`

用于程序列表，以及段落中引用程序元素，如变量或函数名、数据库、数据类型、环境变量、语句和关键字。

**`常宽粗体`**

显示用户应逐字输入的命令或其他文本。

`常宽斜体`

显示应替换为用户提供的值或由上下文确定的值的文本。

###### 小贴士

此元素表示一个提示或建议。

###### 注意

此元素表示一般性说明。

###### 警告

此元素表示警告或注意。

在整本书中，我们强化了我们所说的五个提示原则，确定哪个原则最适合当前示例。您可能想参考第一章，其中详细描述了这些原则。

# 原则名称

这将解释原则是如何应用于当前示例或文本部分的。

# 使用代码示例

补充材料（代码示例、练习等）可在[*https://oreil.ly/prompt-engineering-for-generative-ai*](https://oreil.ly/prompt-engineering-for-generative-ai)下载。

如果您对代码示例有技术问题或使用上的问题，请发送电子邮件至*bookquestions@oreilly.com*。

本书旨在帮助您完成工作。一般来说，如果本书提供了示例代码，您可以在您的程序和文档中使用它。除非您正在复制代码的很大一部分，否则您不需要联系我们获得许可。例如，编写一个使用本书中几个代码片段的程序不需要许可。通过引用本书并引用示例代码来回答问题不需要许可。将本书的大量示例代码纳入您产品的文档中需要许可。

我们感谢，但通常不需要署名。署名通常包括标题、作者、出版社和 ISBN。例如：“*《生成式 AI 的提示工程》由詹姆斯·菲尼克斯和迈克·泰勒（O’Reilly）著。版权所有 2024 萨克斯弗拉吉，LLC 和 Just Understanding Data LTD，978-1-098-15343-4。””

如果您认为您对代码示例的使用超出了合理使用或上述许可，请随时联系我们*permissions@oreilly.com*。

# O’Reilly 在线学习

###### 注意

40 多年来，[*O’Reilly Media*](https://oreilly.com)一直为科技公司提供技术和商业培训、知识和洞察力，以帮助公司成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专业知识。O’Reilly 的在线学习平台为您提供按需访问实时培训课程、深入的学习路径、交互式编码环境以及来自 O’Reilly 和 200 多家其他出版商的大量文本和视频。更多信息，请访问[*https://oreilly.com*](https://oreilly.com)。

# 如何联系我们

请将有关本书的评论和问题寄给出版社：

+   O’Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-889-8969（美国或加拿大）

+   707-827-7019（国际或本地）

+   707-829-0104（传真）

+   *support@oreilly.com*

+   [*https://www.oreilly.com/about/contact.html*](https://www.oreilly.com/about/contact.html)

我们为这本书有一个网页，上面列出了勘误表、示例和任何其他附加信息。您可以通过[*https://oreil.ly/prompt-engineering-generativeAI*](https://oreil.ly/prompt-engineering-generativeAI)访问此页面。

想了解我们书籍和课程的相关新闻和信息，请访问[*https://oreilly.com*](https://oreilly.com)。

在 LinkedIn 上找到我们：[*https://linkedin.com/company/oreilly-media*](https://linkedin.com/company/oreilly-media)。

在 YouTube 上关注我们：[*https://youtube.com/oreillymedia*](https://youtube.com/oreillymedia)。

# 致谢

我们想感谢以下人员在本书技术审阅中的贡献以及他们在纠正快速变化目标时的耐心：

+   Mayo Oshin，早期 LangChain 贡献者，[SeinnAI Analytics](https://www.siennaianalytics.com)创始人

+   Ellis Crosby，[Scarlett Panda](https://www.scarlettpanda.com)创始人及 AI 代理机构[Incremen.to](https://incremen.to)创始人

+   Dave Pawson，O’Reilly 出版社[*XSL-FO*](https://oreil.ly/XSL-FO)作者

+   Mark Phoenix，高级软件工程师

+   Aditya Goel，GenAI 顾问

+   Sanyam Kumar，Genmab 数据科学部副总监

+   Lakshmanan Sethu，Google Gen AI Solutions 的 TAM

+   Janit Anjaria，Aurora Innovation Inc.员工，TLM

我们也感谢我们的家人对他们耐心和理解，并想向他们保证，我们仍然更喜欢与他们交谈而不是与 ChatGPT。
