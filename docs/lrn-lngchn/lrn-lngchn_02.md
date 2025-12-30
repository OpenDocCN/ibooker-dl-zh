# 第二章：RAG 第一部分：索引你的数据

在上一章中，你学习了使用 LangChain 创建 LLM 应用程序的重要构建块。你还构建了一个简单的 AI 聊天机器人，它由发送给模型的提示和模型生成的输出组成。但这个简单的聊天机器人存在重大局限性。

如果你的用例需要模型未训练过的知识，怎么办？例如，假设你想使用 AI 来询问一家公司的问题，但这些信息包含在私有的 PDF 或其他类型的文档中。虽然我们已经看到模型提供商正在丰富他们的训练数据集，以包括越来越多的世界公共信息（无论其存储格式如何），但在 LLM 的知识库中仍然存在两个主要限制：

私有数据

根据定义，未公开的信息不包括在 LLM 的训练数据中。

当前的新闻事件

训练一个 LLM 是一个昂贵且耗时的过程，可能需要多年时间，数据收集是第一步之一。这导致了所谓的知识截止点，即 LLM 对现实世界事件没有知识的一个日期；这通常会是训练集最终确定的日期。这可能是几个月到几年前的日期，具体取决于所讨论的模型。

在任何情况下，模型很可能会产生幻觉（找到误导性或错误的信息）并给出不准确的信息。调整提示也不会解决这个问题，因为它依赖于模型当前的知识。

# 目标：为 LLM 选择相关上下文

如果你需要的唯一私有/当前数据只是 LLM 用例中的一到两页文本，那么本章将会短得多：你只需要将整个文本包含在你发送给模型的每个提示中，就可以使该信息对 LLM 可用。

使数据对 LLM 可用性的挑战首先是一个数量问题。你拥有的信息比可以放入你发送给 LLM 的每个提示中的信息要多。每次调用模型时，你如何在你的大量文本集合中选择一个小的子集？或者换句话说，你如何（在模型的帮助下）选择与每个问题最相关的文本？

在本章和下一章中，你将学习如何通过两个步骤克服这一挑战：

1.  *索引*你的文档，即以你的应用程序可以轻松找到每个问题的最相关文档的方式预处理它们

1.  *检索*来自索引的外部数据并将其用作 LLM 的*上下文*，以便根据你的数据生成准确输出的过程

本章重点介绍索引，这是第一步，涉及将你的文档预处理成 LLM 可以理解和搜索的格式。这种技术称为*检索增强生成*（RAG）。但在我们开始之前，让我们讨论一下*为什么*你的文档需要预处理。

假设你希望使用 LLM 来分析特斯拉 2022 年年度报告中的财务表现和风险，该报告存储为 PDF 格式的文本。你的目标是能够提出像“特斯拉在 2022 年面临了哪些关键风险？”这样的问题，并基于文档风险因素部分的内容得到一个类似人类的回答。[特斯拉 2022 年年度报告](https://oreil.ly/Bp51E)。

分解来看，有四个关键步骤（如图 2-1 所示）需要采取以实现这一目标：

1.  从文档中提取文本。

1.  将文本分割成可管理的块。

1.  将文本转换为计算机可以理解的数字。

1.  将这些文本的数字表示存储在某个地方，以便快速检索文档的相关部分以回答给定的问题。

![一个图表的图表  描述自动生成](img/lelc_0201.png)

###### 图 2-1\. 预处理文档以供 LLM 使用的四个关键步骤

图 2-1 说明了这种预处理和转换文档的流程，这个过程被称为摄取。*摄取*简单来说就是将你的文档转换为计算机可以理解和分析的数字，并将它们存储在一种特殊类型的数据库中以便高效检索。这些数字正式上被称为*嵌入*，而这种特殊类型的数据库被称为*向量存储器*。让我们更详细地了解一下嵌入是什么以及为什么它们很重要，从比 LLM 驱动的嵌入更简单的东西开始。

# 嵌入：将文本转换为数字

*嵌入*指的是将文本表示为（长）数字序列。这是一种有损表示——也就是说，你不能从这些数字序列中恢复原始文本，因此你通常存储原始文本和这种数字表示。

那么，为什么要费这个功夫呢？因为当你与数字打交道时，你会获得灵活性和力量：你可以在单词上做数学运算！让我们看看为什么这会令人兴奋。

## LLM 出现之前的嵌入

在 LLM 出现之前，计算机科学家就已经在使用嵌入技术了——例如，为了在网站上实现全文搜索功能或将电子邮件分类为垃圾邮件。让我们看看一个例子：

1.  取这三个句子：

    +   多么晴朗的一天啊。

    +   今天的天空如此明亮。

    +   好几周没见过晴朗的天气了。

1.  列出它们中的所有唯一单词：*what*，*a*，*sunny*，*day*，*such*，*bright*，等等。

1.  对于每个句子，逐个单词地检查，如果单词不存在，则分配数字 0，如果句子中只使用一次，则分配数字 1，如果出现两次，则分配数字 2，依此类推。

表 2-1 显示了结果。

表 2-1\. 三个句子的词嵌入

| Word | What a sunny day. | Such bright skies today. | I haven’t seen a sunny day in weeks. |
| --- | --- | --- | --- |
| *what* | 1 | 0 | 0 |
| *a* | 1 | 0 | 1 |
| *sunny* | 1 | 0 | 1 |
| *day* | 1 | 0 | 1 |
| *such* | 0 | 1 | 0 |
| *bright* | 0 | 1 | 0 |
| *skies* | 0 | 1 | 0 |
| *today* | 0 | 1 | 0 |
| *I* | 0 | 0 | 1 |
| *haven’t* | 0 | 0 | 1 |
| *seen* | 0 | 0 | 1 |
| *in* | 0 | 0 | 1 |
| *weeks* | 0 | 0 | 1 |

在此模型中，*我已经好几个星期没见过晴天了*的嵌入是数字序列*0 1 1 1 0 0 0 0 1 1 1 1 1*。这被称为*词袋模型*，这些嵌入也被称为*稀疏嵌入*（或稀疏向量——*向量*是数字序列的另一种说法），因为很多数字将是 0。大多数英语句子只使用所有现有英语单词的一小部分。

您可以使用此模型成功进行：

关键词搜索

您可以找到包含给定单词或单词的文档。

文档分类

您可以为之前标记为垃圾邮件或非垃圾邮件的示例集合计算嵌入，取平均值，并得到每个类别（垃圾邮件或非垃圾邮件）的平均词频。然后，每个新文档与这些平均值进行比较，并根据结果进行分类。

这里的限制是模型没有对意义的认识，只有实际使用的词语。例如，*晴天*和*晴朗的天空*的嵌入看起来非常不同。事实上，它们没有共同的词语，尽管我们知道它们有相似的意义。或者，在电子邮件分类问题中，一个潜在的垃圾邮件发送者可以通过用同义词替换常见的“垃圾邮件单词”来欺骗过滤器。

在下一节中，我们将看到语义嵌入如何通过使用数字来表示文本的意义，而不是文本中找到的确切词语，来解决这一限制。

## 基于 LLM 的嵌入

我们将跳过所有介于其中的 ML（机器学习）发展，直接跳到基于 LLM 的嵌入。只需知道，从上一节中概述的简单方法到本节中描述的复杂方法，有一个逐步的演变过程。

您可以将嵌入模型视为 LLM（大型语言模型）训练过程的衍生。如果您还记得前言，LLM 的训练过程（从大量书面文本中学习）使 LLM 能够根据上下文理解词语和句子的意义，从而完成提示（或输入）的最合适的延续（输出）。这种能力源于从训练文本中词语如何共同使用中学习到的对上下文中词语和句子意义的理解。这种对提示（或语义）意义的*理解*可以提取为输入文本的数值表示（或嵌入），并且可以直接用于一些非常有趣的应用场景。

实际上，大多数嵌入模型都是为了这个目的而训练的，它们遵循与 LLM 相似的一些架构和训练过程，因为这样更有效率，并且会产生高质量的嵌入。1

然后，*嵌入模型*是一种算法，它接受一段文本并输出其意义的数值表示——技术上讲，是一长串浮点数（小数），通常在 100 到 2000 个数字之间，或称为*维度*。这些也被称为*密集*嵌入，与上一节的*稀疏*嵌入相对，因为在这里通常所有维度都不同于 0。

###### 小贴士

不同的模型产生不同的数字和不同大小的列表。所有这些都是针对每个模型的特定属性；也就是说，即使列表的大小匹配，也不能比较不同模型的嵌入。应始终避免将不同模型的嵌入组合在一起。

## 语义嵌入解释

考虑这三个单词：*狮子*、*宠物*和*狗*。直观上，哪一对单词在第一眼看上去具有相似的特征？显然的答案是*宠物*和*狗*。但是计算机没有能力挖掘这种直觉或对英语语言的细微理解。为了使计算机能够区分狮子、宠物或狗，你需要能够将它们翻译成计算机的语言，即数字*。

图 2-2 阐述了将每个单词转换成保留其意义的假设性数字表示。

![一个圆的图的图  描述自动生成，置信度中等](img/lelc_0202.png)

###### 图 2-2. 单词的语义表示

图 2-2 显示了每个单词及其相应的语义嵌入。请注意，这些数字本身没有特定的意义，而是两个意义相近的单词（或句子）的数字序列应该比无关单词的序列更*接近*。正如你所看到的，每个数字都是一个*浮点值*，每个都代表一个语义*维度*。让我们看看我们所说的*接近*是什么意思：

如果我们在三维空间中绘制这些向量，它可能看起来像图 2-3。

![一个狗和一个狗的图  描述自动生成](img/lelc_0203.png)

###### 图 2-3. 多维空间中单词向量的图

图 2-3 显示了*宠物*和*狗*向量在距离上比*狮子*图更接近。我们还可以观察到，每个图之间的角度根据它们的相似程度而变化。例如，单词*宠物*和*狮子*之间的角度比*宠物*和*狗*之间的角度更宽，这表明后者单词对之间有更多的相似性。两个向量之间的角度越窄或距离越短，它们的相似性就越高。

在多维空间中计算两个向量之间相似度的有效方法称为余弦相似度。*余弦相似度*计算向量的点积，并将其除以它们模长的乘积，输出一个介于 –1 和 1 之间的数字，其中 0 表示向量没有相关性，–1 表示它们完全不同，1 表示它们完全相同。因此，在我们的三个词中，*pet* 和 *dog* 之间的余弦相似度可能是 0.75，但 *pet* 和 *lion* 之间的相似度可能只有 0.1。

将句子转换为捕获语义意义的嵌入，然后执行计算以在不同句子之间找到语义相似性，这使得我们能够让 LLM 找到与我们的特斯拉 PDF 文档等大量文本相关的最相关文档。现在你已经了解了整体情况，让我们回顾一下预处理文档的第一步（索引）。

# 将你的文档转换为文本

如本章开头所述，预处理你的文档的第一步是将它转换为文本。为了实现这一点，你需要构建逻辑来解析和提取文档，同时尽量减少质量损失。幸运的是，LangChain 提供了处理解析逻辑的 *文档加载器*，使你能够将来自各种来源的数据“加载”到包含文本和相关元数据的 `Document` 类中。

例如，考虑一个简单的 *.txt* 文件。你可以简单地导入 LangChain 的 `TextLoader` 类来提取文本，如下所示：

*Python*

```py
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./test.txt")
loader.load()
```

*JavaScript*

```py
import { TextLoader } from "langchain/document_loaders/fs/text";

const loader = new TextLoader("./test.txt");

const docs = await loader.load();
```

*输出：*

```py
[Document(page_content='text content \n', metadata={'line_number': 0, 'source': 
    './test.txt'})]
```

之前的代码块假设你在当前目录下有一个名为 `test.txt` 的文件。所有 LangChain 文档加载器的使用遵循类似的模式：

1.  从长长的 [集成](https://oreil.ly/iLJ33) 列表中挑选适合你文档类型的加载器。

1.  创建一个加载器的实例，包括任何配置它的参数，包括你的文档位置（通常是文件系统路径或网址）。

1.  通过调用 `load()` 加载文档，它返回一个列表，包含准备好传递到下一阶段的文档（更多内容即将揭晓）。

除了 *.txt* 文件外，LangChain 还提供了其他流行文件类型的文档加载器，包括 *.csv*、*.json* 和 Markdown，以及与 Slack 和 Notion 等流行平台的集成。

例如，你可以使用 `WebBaseLoader` 从网络 URL 加载 HTML 并将其解析为文本。

安装 beautifulsoup4 包：

```py
pip install beautifulsoup4
```

*Python*

```py
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.langchain.com/")
loader.load()
```

*JavaScript*

```py
// install cheerio: npm install cheerio
import { 
  CheerioWebBaseLoader 
} from "@langchain/community/document_loaders/web/cheerio";

const loader = new CheerioWebBaseLoader("https://www.langchain.com/");

const docs = await loader.load();
```

在我们特斯拉 PDF 用例的情况下，我们可以利用 LangChain 的 `PDFLoader` 从 PDF 文档中提取文本：

*Python*

```py
# install the pdf parsing library
# pip install pypdf

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./test.pdf")
pages = loader.load()
```

*JavaScript*

```py
// install the pdf parsing library: npm install pdf-parse

import { PDFLoader } from "langchain/document_loaders/fs/pdf";

const loader = new PDFLoader("./test.pdf");

const docs = await loader.load();
```

文本已从 PDF 文档中提取出来，并存储在`Document`类中。但有一个问题。加载的文档长度超过 100,000 个字符，因此它无法适应大多数 LLMs 或嵌入模型的上下文窗口。为了克服这一限制，我们需要将`Document`分割成可管理的文本块，我们可以在以后将其转换为嵌入并执行语义搜索，这把我们带到了第二步（检索）。

###### 小贴士

LLMs 和嵌入模型在设计时对它们可以处理的输入和输出标记的大小有一个硬性限制。这个限制通常被称为*上下文窗口*，通常适用于输入和输出的组合；也就是说，如果上下文窗口是 100（我们将在下一节讨论单位），而你的输入长度为 90，则输出长度最多为 10。上下文窗口通常以标记的数量来衡量，例如 8,192 个标记。正如在序言中提到的，标记是文本作为数字的表示，每个标记通常覆盖三个到四个英文字符。

# 将文本分割成块

初看起来，将大量文本分割成块似乎很简单，但保持*语义上*相关的（通过意义相关）文本块在一起是一个复杂的过程。为了更容易地将大型文档分割成小但仍有意义的文本片段，LangChain 提供了`RecursiveCharacterTextSplitter`，它执行以下操作：

1.  按重要性顺序列出分隔符。默认情况下，这些是：

    1.  段落分隔符：`\n\n`

    1.  行分隔符：`\n`

    1.  单词分隔符：空格字符

1.  为了尊重给定的块大小，例如，1,000 个字符，首先分割段落。

1.  对于任何超过所需块大小的段落，通过下一分隔符（行）进行分割。继续分割，直到所有块的大小都小于所需长度，或者没有更多的分隔符可以尝试。

1.  将每个块作为`Document`发出，传递原始文档的元数据，以及关于原始文档中位置的其他信息。

让我们看看一个例子：

*Python*

```py
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./test.txt") # or any other loader
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splitted_docs = splitter.split_documents(docs)
```

*JavaScript*

```py
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const loader = new TextLoader("./test.txt"); // or any other loader 
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const splittedDocs = await splitter.splitDocuments(docs)
```

在前面的代码中，由文档加载器创建的文档被分割成每个 1,000 个字符的块，块之间有 200 个字符的重叠，以保持一定的上下文。结果也是一个文档列表，其中每个文档的长度最多为 1,000 个字符，沿着文本的自然分割进行分割——段落、换行符，最后是单词。这利用了文本的结构，以保持每个块是一致的、可读的文本片段。

`RecursiveCharacterTextSplitter` 也可以用来将代码语言和 Markdown 分割成语义块。这是通过使用每个语言的特定关键字作为分隔符来完成的，这确保了例如每个函数的主体都保持在同一个块中，而不是被分割到几个块中。通常，由于编程语言比书面文本有更多的结构，因此不需要在块之间使用重叠。LangChain 包含了多种流行语言的分隔符，例如 Python、JS、Markdown、HTML 等。以下是一个示例：

*Python*

```py
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

PYTHON_CODE = """
def hello_world():
 print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
```

*JavaScript*

```py
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const PYTHON_CODE = `
def hello_world():
 print("Hello, World!")

# Call the function
hello_world()
`;

const pythonSplitter = RecursiveCharacterTextSplitter.fromLanguage("python", {
  chunkSize: 50,
  chunkOverlap: 0,
});
const pythonDocs = await pythonSplitter.createDocuments([PYTHON_CODE]);
```

*输出结果：*

```py
[Document(page_content='def hello_world():\n    print("Hello, World!")'),
    Document(page_content='# Call the function\nhello_world()')]
```

注意我们仍然像之前一样使用 `RecursiveCharacterTextSplitter`，但现在我们使用 `from_language` 方法为其创建一个特定语言的实例。这个方法接受语言的名称以及通常的块大小等参数。同时注意我们现在使用的是 `create_documents` 方法，它接受一个字符串列表，而不是我们之前拥有的文档列表。当您想要分割的文本不来自文档加载器，而只有原始文本字符串时，此方法非常有用。

您还可以使用 `create_documents` 方法的可选第二个参数来传递与每个文本字符串关联的元数据列表。此元数据列表的长度应与字符串列表相同，并将用于填充每个返回的 `Document` 的元数据字段。

让我们看看一个使用元数据参数的 Markdown 文本示例：

*Python*

```py
markdown_text = """
# LangChain

⚡ Building applications with LLMs through composability ⚡

## Quick Install

```bash

pip install langchain

```py

As an open source project in a rapidly developing field, we are extremely open 
 to contributions.
"""

md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
md_docs = md_splitter.create_documents([markdown_text], 
    [{"source": "https://www.langchain.com"}])
```

*JavaScript*

```py
const markdownText = `
# LangChain

⚡ Building applications with LLMs through composability ⚡

## Quick Install

\`\`\`bash
pip install langchain
\`\`\`

As an open source project in a rapidly developing field, we are extremely 
 open to contributions.
`;

const mdSplitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
  chunkSize: 60,
  chunkOverlap: 0,
});
const mdDocs = await mdSplitter.createDocuments([markdownText], 
  [{"source": "https://www.langchain.com"}]);
```

*输出结果：*

```py
[Document(page_content='# LangChain', 
    metadata={"source": "https://www.langchain.com"}),
 Document(page_content='⚡ Building applications with LLMs through composability 
    ⚡', metadata={"source": "https://www.langchain.com"}),
 Document(page_content='## Quick Install\n\n```bash'

    metadata={"source": "https://www.langchain.com"}),

Document(page_content='pip install langchain',

    metadata={"source": "https://www.langchain.com"}),

Document(page_content='```py', metadata={"source": "https://www.langchain.com"}),
 Document(page_content='As an open source project in a rapidly developing field, 
    we', metadata={"source": "https://www.langchain.com"}),
 Document(page_content='are extremely open to contributions.', 
    metadata={"source": "https://www.langchain.com"})]
```

注意两点：

+   文本在 Markdown 文档的自然停止点处进行分割；例如，标题进入一个块，其下的文本行在另一个块中，依此类推。

+   我们在第二个参数中传递的元数据附加到每个生成的文档上，这使得您可以跟踪例如文档的来源以及您可以去哪里查看原始内容。

# 生成文本嵌入

LangChain 还有一个 `Embeddings` 类，用于与文本嵌入模型（包括 OpenAI、Cohere 和 Hugging Face）接口，并生成文本的向量表示。这个类提供了两种方法：一种用于嵌入文档，另一种用于嵌入查询。前者接受文本字符串列表作为输入，而后者接受单个文本字符串。

这里是一个使用 [OpenAI 的嵌入模型](https://oreil.ly/9tnzQ)嵌入文档的示例：

*Python*

```py
from langchain_openai import OpenAIEmbeddings

model = OpenAIEmbeddings()

embeddings = model.embed_documents([
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
])
```

*JavaScript*

```py
import { OpenAIEmbeddings } from "@langchain/openai";

const model = new OpenAIEmbeddings();

const embeddings = await embeddings.embedDocuments([
  "Hi there!",
  "Oh, hello!",
  "What' s your name?",
  "My friends call me World",
  "Hello World!"
]);
```

*输出结果：*

```py
[
  [
    -0.004845875, 0.004899438, -0.016358767, -0.024475135, -0.017341806,
      0.012571548, -0.019156644, 0.009036391, -0.010227379, -0.026945334,
      0.022861943, 0.010321903, -0.023479493, -0.0066544134, 0.007977734,
    0.0026371893, 0.025206111, -0.012048521, 0.012943339, 0.013094575,
    -0.010580265, -0.003509951, 0.004070787, 0.008639394, -0.020631202,
    ... 1511 more items
  ]
  [
      -0.009446913, -0.013253193, 0.013174579, 0.0057552797, -0.038993083,
      0.0077763423, -0.0260478, -0.0114384955, -0.0022683728, -0.016509168,
      0.041797023, 0.01787183, 0.00552271, -0.0049789557, 0.018146982,
      -0.01542166, 0.033752076, 0.006112323, 0.023872782, -0.016535373,
      -0.006623321, 0.016116094, -0.0061090477, -0.0044155475, -0.016627092,
    ... 1511 more items
  ]
  ... 3 more items
]
```

注意您可以同时嵌入多个文档；您应该优先选择这种方式，而不是逐个嵌入，因为它将更有效率（因为这些模型是如何构建的）。您会得到一个包含多个数字列表的列表——每个内部列表都是一个向量或嵌入，如前文所述。

现在让我们看看一个端到端的示例，使用我们迄今为止看到的三个功能：

+   文档加载器，将任何文档转换为纯文本

+   文本分割器，将每个大型文档分割成许多较小的文档

+   嵌入模型，创建每个分割的语义的数值表示

这里是代码：

*Python*

```py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

## Load the document 

loader = TextLoader("./test.txt")
doc = loader.load()

"""
[
 Document(page_content='Document loaders\n\nUse document loaders to load data 
 from a source as `Document`\'s. A `Document` is a piece of text\nand 
 associated metadata. For example, there are document loaders for 
 loading a simple `.txt` file, for loading the text\ncontents of any web 
 page, or even for loading a transcript of a YouTube video.\n\nEvery 
 document loader exposes two methods:\n1\. "Load": load documents from 
 the configured source\n2\. "Load and split": load documents from the 
 configured source and split them using the passed in text 
 splitter\n\nThey optionally implement:\n\n3\. "Lazy load": load 
 documents into memory lazily\n', metadata={'source': 'test.txt'})
]
"""

## Split the document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
)
chunks = text_splitter.split_documents(doc)

## Generate embeddings

embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)
"""
[[0.0053587136790156364,
 -0.0004999046213924885,
 0.038883671164512634,
 -0.003001077566295862,
 -0.00900818221271038, ...], ...]
"""
```

*JavaScript*

```py
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";

// Load the document 

const loader = new TextLoader("./test.txt");
const docs = await loader.load();

// Split the document

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const chunks = await splitter.splitDocuments(docs)

// Generate embeddings

const model = new OpenAIEmbeddings();
await embeddings.embedDocuments(chunks.map(c => c.pageContent));
```

一旦您从您的文档中生成了嵌入，下一步就是将它们存储在一个称为向量存储的特殊数据库中。

# 在向量存储中存储嵌入

在本章前面，我们讨论了余弦相似度计算，用于衡量向量空间中向量的相似度。向量存储是一个数据库，旨在存储向量并高效快速地执行复杂计算，如余弦相似度。

与专门存储结构化数据（如 JSON 文档或符合关系数据库模式的 schema 数据）的传统数据库不同，向量存储处理非结构化数据，包括文本和图像。像传统数据库一样，向量存储能够执行创建、读取、更新、删除（CRUD）和搜索操作。

向量存储解锁了各种用例，包括可扩展的应用程序，这些应用程序利用 AI 来回答有关大型文档的问题，如图 2-4 所示。

![存储的示意图  描述由系统自动生成](img/lelc_0204.png)

###### 图 2-4\. 从向量存储中加载、嵌入、存储和检索相关文档

图 2-4 说明了文档嵌入如何插入到向量存储中，以及当发送查询时，如何从向量存储中检索相似的嵌入。

目前，有大量的向量存储提供商可供选择，每个都专注于不同的功能。您的选择应取决于您应用程序的关键需求，包括多租户、元数据过滤能力、性能、成本和可扩展性。

尽管向量存储是专为管理向量数据而构建的利基数据库，但使用它们也有一些缺点：

+   大多数向量存储相对较新，可能无法经受时间的考验。

+   管理和优化向量存储可能具有相对陡峭的学习曲线。

+   管理单独的数据库会增加您的应用程序的复杂性，并可能耗尽宝贵的资源。

幸运的是，向量存储功能最近通过`pgvector`扩展扩展到了 PostgreSQL（一个流行的开源关系数据库）。这使得你可以使用你已熟悉的相同数据库，并为你的事务表（例如你的用户表）以及你的向量搜索表提供动力。

## 使用 PGVector 进行设置

要使用 Postgres 和 PGVector，你需要遵循一些设置步骤：

1.  确保你的计算机上已安装 Docker，按照[操作系统的说明](https://oreil.ly/Gn28O)进行操作。

1.  在你的终端中运行以下命令；它将在你的计算机上启动一个运行在端口 6024 的 Postgres 实例：

    ```py
    docker run \
        --name pgvector-container \
        -e POSTGRES_USER=langchain \
        -e POSTGRES_PASSWORD=langchain \
        -e POSTGRES_DB=langchain \
        -p 6024:5432 \
        -d pgvector/pgvector:pg16
    ```

    打开你的 Docker 仪表板容器，你应该在`pgvector-container`旁边看到绿色的运行状态。

1.  将连接字符串保存到你的代码中使用；我们稍后会用到它：

    ```py
    postgresql+psycopg://langchain:langchain@localhost:6024/langchain
    ```

## 与向量存储一起工作

在上一节关于嵌入的设置中继续，现在让我们看看在 PGVector 中加载、分割、嵌入和存储文档的示例：

*Python*

```py
# first, pip install langchain-postgres
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import uuid

# Load the document, split it into chunks
raw_documents = TextLoader('./test.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
    chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# embed each chunk and insert it into the vector store
embeddings_model = OpenAIEmbeddings()
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector.from_documents(documents, embeddings_model, connection=connection)
```

*JavaScript*

```py
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { v4 as uuidv4 } from 'uuid';

// Load the document, split it into chunks
const loader = new TextLoader("./test.txt");
const raw_docs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const docs = await splitter.splitDocuments(docs)

// embed each chunk and insert it into the vector store
const embeddings_model = new OpenAIEmbeddings();
const db = await PGVectorStore.fromDocuments(docs, embeddings_model, {
  postgresConnectionOptions: {
    connectionString: 'postgresql://langchain:langchain@localhost:6024/langchain'
  }
})
```

注意我们如何重复使用前几节中的代码，首先使用加载器加载文档，然后将它们分割成更小的块。然后，我们实例化我们想要使用的嵌入模型——在这个例子中是 OpenAI 的。请注意，你在这里可以使用 LangChain 支持的任何其他嵌入模型。

接下来，我们有一行新的代码，它根据给定的文档、嵌入模型和连接字符串创建一个向量存储。这将执行以下几件事：

+   建立与你在计算机上运行的 Postgres 实例的连接（见“使用 PGVector 进行设置”。）

+   运行任何必要的设置，例如创建用于存储文档和向量的表，如果这是你第一次运行它。

+   使用你选择的模型为每个传入的文档创建嵌入。

+   在 Postgres 中存储嵌入、文档的元数据和文档的文本内容，以便进行搜索。

让我们看看搜索文档的样子：

*Python*

```py
db.similarity_search("query", k=4)
```

*JavaScript*

```py
await pgvectorStore.similaritySearch("query", 4);
```

此方法将通过以下过程找到最相关的文档（你之前已索引的文档）：

+   搜索查询——在这个例子中，是单词`query`——将被发送到嵌入模型以检索其嵌入。

+   然后，它将在 Postgres 上运行一个查询，以找到与你的查询最相似的 N（在这个例子中是 4）先前存储的嵌入。

+   最后，它将检索与每个嵌入相关的文本内容和元数据。

+   模型现在可以返回一个按与查询相似度排序的`Document`列表——最相似的排在第一位，其次是第二相似的，依此类推。

你还可以向现有数据库添加更多文档。让我们看看一个例子：

*Python*

```py
ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents(
    [
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,
)
```

*JavaScript*

```py
const ids = [uuidv4(), uuidv4()];

await db.addDocuments(
  [
    {
      pageContent: "there are cats in the pond",
      metadata: {location: "pond", topic: "animals"}
    }, 
    {
      pageContent: "ducks are also found in the pond",
      metadata: {location: "pond", topic: "animals"}
    },
  ], 
  {ids}
);
```

我们在这里使用的`add_documents`方法将遵循与`fromDocuments`类似的过程：

+   使用你选择的模型为每个传入的文档创建嵌入。

+   将嵌入、文档的元数据和文档的文本内容存储在 Postgres 中，以便进行搜索。

在这个例子中，我们使用可选的`ids`参数为每个文档分配标识符，这允许我们稍后更新或删除它们。

这里是一个删除操作的例子：

*Python*

```py
db.delete(ids=[1])
```

*JavaScript*

```py
await db.delete({ ids: [ids[1]] })
```

这通过使用其通用唯一标识符（UUID）删除了第二个文档。现在让我们看看如何以更系统的方式来做这件事。

# 跟踪文档更改

与向量存储一起工作的一个主要挑战是处理经常变化的数据，因为变化意味着重新索引。重新索引可能导致昂贵的嵌入重新计算和现有内容的重复。

幸运的是，LangChain 提供了一个索引 API，使您能够轻松地将文档与向量存储保持同步。该 API 使用一个类（`RecordManager`）来跟踪写入向量存储的文档。在索引内容时，为每个文档计算哈希值，并在`RecordManager`中存储以下信息：

+   文档哈希（页面内容和元数据的哈希）

+   写入时间

+   源 ID（每个文档应在其元数据中包含信息，以确定此文档的最终来源）。

此外，索引 API 提供了清理模式，以帮助您决定如何删除向量存储中现有的文档。例如，如果您在插入之前更改了文档的处理方式，或者源文档已更改，您可能希望删除任何来自与正在索引的新文档相同源的新文档。如果某些源文档已被删除，您将希望删除向量存储中的所有现有文档，并用重新索引的文档替换它们。

模式如下：

+   `None`模式不会进行任何自动清理，允许用户手动清理旧内容。

+   `Incremental`和`full`模式如果源文档或派生文档的内容已更改，则会删除内容的前一个版本。

+   `Full`模式将删除当前索引文档中未包含的任何文档。

这里是一个使用索引 API 的例子，其中 Postgres 数据库已设置为记录管理器：

*Python*

```py
from langchain.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
namespace = "my_docs_namespace"

vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

record_manager = SQLRecordManager(
    namespace,
    db_url="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
)

# Create the schema if it doesn't exist
record_manager.create_schema()

# Create documents
docs = [
    Document(page_content='there are cats in the pond', metadata={
        "id": 1, "source": "cats.txt"}),
    Document(page_content='ducks are also found in the pond', metadata={
        "id": 2, "source": "ducks.txt"}),
]

# Index the documents
index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",  # prevent duplicate documents
    source_id_key="source",  # use the source field as the source_id
)

print("Index attempt 1:", index_1)

# second time you attempt to index, it will not add the documents again
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)

print("Index attempt 2:", index_2)

# If we mutate a document, the new version will be written and all old 
# versions sharing the same source will be deleted.

docs[0].page_content = "I just modified this document!"

index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)

print("Index attempt 3:", index_3)
```

*JavaScript*

```py
/** 
1\. Ensure docker is installed and running (https://docs.docker.com/get-docker/)
2\. Run the following command to start the postgres container:

docker run \
 --name pgvector-container \
 -e POSTGRES_USER=langchain \
 -e POSTGRES_PASSWORD=langchain \
 -e POSTGRES_DB=langchain \
 -p 6024:5432 \
 -d pgvector/pgvector:pg16
3\. Use the connection string below for the postgres container
*/

import { PostgresRecordManager } from '@langchain/community/indexes/postgres';
import { index } from 'langchain/indexes';
import { OpenAIEmbeddings } from '@langchain/openai';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { v4 as uuidv4 } from 'uuid';

const tableName = 'test_langchain';
const connectionString =
  'postgresql://langchain:langchain@localhost:6024/langchain';
// Load the document, split it into chunks

const config = {
  postgresConnectionOptions: {
    connectionString,
  },
  tableName: tableName,
  columns: {
    idColumnName: 'id',
    vectorColumnName: 'vector',
    contentColumnName: 'content',
    metadataColumnName: 'metadata',
  },
};

const vectorStore = await PGVectorStore.initialize(
  new OpenAIEmbeddings(),
  config
);

// Create a new record manager
const recordManagerConfig = {
  postgresConnectionOptions: {
    connectionString,
  },
  tableName: 'upsertion_records',
};
const recordManager = new PostgresRecordManager(
  'test_namespace',
  recordManagerConfig
);

// Create the schema if it doesn't exist
await recordManager.createSchema();

const docs = [
  {
    pageContent: 'there are cats in the pond',
    metadata: { id: uuidv4(), source: 'cats.txt' },
  },
  {
    pageContent: 'ducks are also found in the pond',
    metadata: { id: uuidv4(), source: 'ducks.txt' },
  },
];

// the first attempt will index both documents
const index_attempt_1 = await index({
  docsSource: docs,
  recordManager,
  vectorStore,
  options: {
    // prevent duplicate documents by id from being indexed
    cleanup: 'incremental',
    // the key in the metadata that will be used to identify the document 
    sourceIdKey: 'source', 
  },
});

console.log(index_attempt_1);

// the second attempt will skip indexing because the identical documents 
// already exist
const index_attempt_2 = await index({
  docsSource: docs,
  recordManager,
  vectorStore,
  options: {
    cleanup: 'incremental',
    sourceIdKey: 'source',
  },
});

console.log(index_attempt_2);

// If we mutate a document, the new version will be written and all old 
// versions sharing the same source will be deleted.
docs[0].pageContent = 'I modified the first document content';
const index_attempt_3 = await index({
  docsSource: docs,
  recordManager,
  vectorStore,
  options: {
    cleanup: 'incremental',
    sourceIdKey: 'source',
  },
});

console.log(index_attempt_3);
```

首先，您创建一个记录管理器，它跟踪哪些文档之前已被索引。然后您使用`index`函数将您的向量存储与新的文档列表同步。在这个例子中，我们使用增量模式，因此任何具有与之前相同的 ID 的文档将被新版本替换。

# 索引优化

基本的 RAG 索引阶段涉及对给定文档的简单文本分割和嵌入。然而，这种基本方法会导致检索结果不一致，以及相对较高的幻觉发生率，尤其是在数据源包含图像和表格时。

有各种策略可以增强索引阶段的准确性和性能。在下一节中，我们将介绍其中的三种：MultiVectorRetriever、RAPTOR 和 ColBERT。

## MultiVectorRetriever

一个包含文本和表格混合的文档不能简单地通过文本分割成块并嵌入为上下文：整个表格可以轻易丢失。为了解决这个问题，我们可以将用于答案合成的文档与用于检索器的引用解耦。图 2-5 说明了这一点。

![2024 年 3 月 16 日下午 5 点 54 分 55 秒的屏幕截图.png](img/lelc_0205.png)

###### 图 2-5\. 索引单个文档的多个表示

例如，在一个包含表格的文档中，我们首先可以生成和嵌入表格元素的摘要，确保每个摘要都包含一个指向完整原始表格的`id`引用。接下来，我们将原始引用的表格存储在单独的 docstore 中。最后，当用户的查询检索到表格摘要时，我们将整个引用的原始表格作为上下文传递给最终发送给 LLM 进行答案合成的提示。这种方法使我们能够向模型提供回答问题所需的全局信息背景。

这里有一个例子。首先，让我们使用 LLM 生成文档的摘要：

*Python*

```py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import uuid

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "summaries"
embeddings_model = OpenAIEmbeddings()
# Load the document
loader = TextLoader("./test.txt", encoding="utf-8")
docs = loader.load()

print("length of loaded docs: ", len(docs[0].page_content))
# Split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# The rest of your code remains the same, starting from:
prompt_text = "Summarize the following document:\n\n{doc}"

prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
summarize_chain = {
    "doc": lambda x: x.page_content} | prompt | llm | StrOutputParser()

# batch the chain across the chunks
summaries = summarize_chain.batch(chunks, {"max_concurrency": 5})
```

接下来，让我们定义向量存储和 docstore 以存储原始摘要及其嵌入：

*Python*

```py
# The vectorstore to use to index the child chunks
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# indexing the summaries in our vector store, whilst retaining the original 
# documents in our document store:
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Changed from summaries to chunks since we need same length as docs
doc_ids = [str(uuid.uuid4()) for _ in chunks]

# Each summary is linked to the original document by the doc_id
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# Add the document summaries to the vector store for similarity search
retriever.vectorstore.add_documents(summary_docs)

# Store the original documents in the document store, linked to their summaries 
# via doc_ids
# This allows us to first search summaries efficiently, then fetch the full 
# docs when needed
retriever.docstore.mset(list(zip(doc_ids, chunks)))

# vector store retrieves the summaries
sub_docs = retriever.vectorstore.similarity_search(
    "chapter on philosophy", k=2)
```

最后，让我们根据查询检索相关的完整上下文文档：

*Python*

```py
# Whereas the retriever will return the larger source document chunks:
retrieved_docs = retriever.invoke("chapter on philosophy")

```

下面是 JavaScript 中的完整实现：

*JavaScript*

```py
import * as uuid from 'uuid';
import { MultiVectorRetriever } from 'langchain/retrievers/multi_vector';
import { OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { InMemoryStore } from '@langchain/core/stores';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { Document } from '@langchain/core/documents';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';

const connectionString =
  'postgresql://langchain:langchain@localhost:6024/langchain';
const collectionName = 'summaries';

const textLoader = new TextLoader('./test.txt');
const parentDocuments = await textLoader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 10000,
  chunkOverlap: 20,
});
const docs = await splitter.splitDocuments(parentDocuments);

const prompt = PromptTemplate.fromTemplate(
  `Summarize the following document:\n\n{doc}`
);

const llm = new ChatOpenAI({ modelName: 'gpt-3.5-turbo' });

const chain = RunnableSequence.from([
  { doc: (doc) => doc.pageContent },
  prompt,
  llm,
  new StringOutputParser(),
]);

// batch summarization chain across the chunks
const summaries = await chain.batch(docs, {
  maxConcurrency: 5,
});

const idKey = 'doc_id';
const docIds = docs.map((_) => uuid.v4());
// create summary docs with metadata linking to the original docs
const summaryDocs = summaries.map((summary, i) => {
  const summaryDoc = new Document({
    pageContent: summary,
    metadata: {
      [idKey]: docIds[i],
    },
  });
  return summaryDoc;
});

// The byteStore to use to store the original chunks
const byteStore = new InMemoryStore();

// vector store for the summaries
const vectorStore = await PGVectorStore.fromDocuments(
  docs,
  new OpenAIEmbeddings(),
  {
    postgresConnectionOptions: {
      connectionString,
    },
  }
);

const retriever = new MultiVectorRetriever({
  vectorstore: vectorStore,
  byteStore,
  idKey,
});

const keyValuePairs = docs.map((originalDoc, i) => [docIds[i], originalDoc]);

// Use the retriever to add the original chunks to the document store
await retriever.docstore.mset(keyValuePairs);

// Vectorstore alone retrieves the small chunks
const vectorstoreResult = await retriever.vectorstore.similaritySearch(
  'chapter on philosophy',
  2
);
console.log(`summary: ${vectorstoreResult[0].pageContent}`);
console.log(
  `summary retrieved length: ${vectorstoreResult[0].pageContent.length}`
);

// Retriever returns larger chunk result
const retrieverResult = await retriever.invoke('chapter on philosophy');
console.log(
  `multi-vector retrieved chunk length: ${retrieverResult[0].pageContent.length}`
);
```

## RAPTOR：递归抽象处理用于树状检索

RAG 系统需要处理低级问题，这些问题引用了单个文档中找到的特定事实，或者处理高级问题，这些问题提炼了跨越多个文档的思想。使用典型的基于文档块的最邻近邻居（k-NN）检索处理这两种类型的问题可能是一个挑战。

*递归抽象处理用于树状检索*（RAPTOR）是一种有效的策略，它涉及创建捕获高级概念的文档摘要，嵌入和聚类这些文档，然后[总结每个聚类](https://oreil.ly/VdIpJ).^(2)。这是递归进行的，产生了一个具有越来越高级概念的摘要树。然后，摘要和初始文档一起索引，覆盖从低级到高级的用户问题。图 2-6 说明了这一点。

![2024 年 3 月 16 日下午 6 点 16 分 21 秒的屏幕截图.png](img/lelc_0206.png)

###### 图 2-6\. 递归总结文档

## ColBERT：优化嵌入

在索引阶段使用嵌入模型的一个挑战是，它们将文本压缩成固定长度的（向量）表示，这些表示捕获了文档的语义内容。尽管这种压缩对检索很有用，但嵌入无关或冗余的内容可能导致最终 LLM 输出的幻觉。

解决这个问题的一个方案是执行以下操作：

1.  为文档和查询中的每个标记生成上下文嵌入。

1.  计算并评分每个查询标记与所有文档标记之间的相似度。

1.  将每个查询嵌入与任何文档嵌入的最大相似度分数相加，以获得每个文档的评分。

这导致了一种细粒度和有效的嵌入方法，以实现更好的检索。幸运的是，名为 ColBERT 的嵌入模型体现了解决这个问题的方案.^(3)

下面是如何利用 ColBERT 进行数据最优嵌入的方法：

*Python*

```py
# RAGatouille is a library that makes it simple to use ColBERT
#! pip install -U ragatouille

from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

import requests

def get_wikipedia_page(title: str):
    """
 Retrieve the full text content of a Wikipedia page.

 :param title: str - Title of the Wikipedia page.
 :return: str - Full text content of the page as raw string.
 """
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

full_document = get_wikipedia_page("Hayao_Miyazaki")

## Create an index
RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)

#query
results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
results

#utilize langchain retriever
retriever = RAG.as_langchain_retriever(k=3)
retriever.invoke("What animation studio did Miyazaki found?")
```

通过使用 ColBERT，你可以提高 LLM 用作上下文的检索文档的相关性。

# 摘要

在本章中，你学习了如何使用各种 LangChain 模块准备和预处理你的文档以用于你的 LLM 应用。文档加载器使你能够从数据源中提取文本，文本分割器帮助你将文档分割成语义上相似的片段，嵌入模型将你的文本转换为它们意义的向量表示。

此外，向量存储允许你对这些嵌入执行 CRUD 操作，同时进行复杂的计算以计算语义上相似的文本片段。最后，索引优化策略使你的 AI 应用能够提高嵌入的质量并准确检索包含半结构化数据（包括表格）的文档。

在第三章中，你将学习如何根据查询从你的向量存储中高效检索最相似的文档片段，将其作为模型可以看到的上下文，然后生成准确的输出。

^(1) Arvind Neelakantan 等人，[“通过对比预训练进行文本和代码嵌入”](https://oreil.ly/YOVmh)，arXiv，2022 年 1 月 21 日。

^(2) Parth Sarthi 等人，[“RAPTOR：用于树状检索的递归抽象处理”](https://oreil.ly/hS4NB)，arXiv，2024 年 1 月 31 日。该论文在 ICLR 2024 上发表。

^(3) Keshav Santhanam 等人，[“ColBERTv2：通过轻量级后期交互实现有效和高效的检索”](https://oreil.ly/9spW2)，arXiv，2021 年 12 月 2 日。
