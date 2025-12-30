# 第一章：使用 LangChain 的 LLM 基础

前言让你领略了 LLM 提示的强大功能，我们亲眼见证了不同的提示技术如何影响从 LLM 中获得的结果，尤其是在谨慎结合时。实际上，构建好的 LLM 应用的挑战在于如何有效地构建发送给模型的提示，并处理模型的预测以返回准确的结果（参见图 1-1）。

![图片](img/lelc_0101.png)

###### 图 1-1. 将 LLM 作为应用程序有用部分的挑战

如果你能够解决这个问题，那么你已经在构建 LLM 应用的道路上走得很远了，无论是简单还是复杂的 LLM 应用。在本章中，你将了解 LangChain 的构建块如何映射到 LLM 概念，以及当它们有效结合时，如何帮助你构建 LLM 应用。但首先，侧边栏“为什么使用 LangChain？”是一个关于为什么我们认为使用 LangChain 构建 LLM 应用有用的简要介绍。

# 使用 LangChain 进行设置

为了跟随本章的其余部分以及后续章节，我们建议首先在你的计算机上设置 LangChain。

请参阅前言中有关设置 OpenAI 账户的说明，并在尚未完成的情况下完成这些操作。如果你更喜欢使用不同的 LLM 提供商，请参阅“为什么使用 LangChain？”以获取替代方案。

然后，前往 OpenAI 网站上的[API 密钥页面](https://oreil.ly/BKrtV)（在登录 OpenAI 账户后），创建一个 API 密钥，并保存它——你很快就会需要它。

###### 注意

在本书中，我们将展示 Python 和 JavaScript（JS）的代码示例。LangChain 在这两种语言中都提供相同的功能，所以只需选择你最熟悉的一种，并在整本书中遵循相应的代码片段（每种语言的代码示例是等效的）。

首先，为使用 Python 的读者提供一些设置说明：

1.  确保你已经安装了 Python。请参阅[针对你操作系统的说明](https://oreil.ly/20K9l)。

1.  如果你想在笔记本环境中运行示例，请安装 Jupyter。你可以在终端中运行`pip install notebook`来完成此操作。

1.  通过在终端运行以下命令安装 LangChain 库：

    ```py
    pip install langchain langchain-openai langchain-community 
    pip install langchain-text-splitters langchain-postgres
    ```

1.  将本节开头生成的 OpenAI API 密钥在终端环境中可用。你可以通过运行以下命令来完成此操作：

    ```py
    export OPENAI_API_KEY=your-key
    ```

1.  不要忘记将`your-key`替换为你之前生成的 API 密钥。

1.  通过运行以下命令打开 Jupyter 笔记本：

    ```py
    jupyter notebook
    ```

现在，你准备好跟随 Python 代码示例了。

这里是为使用 JavaScript 的读者提供的说明：

1.  将本节开头生成的 OpenAI API 密钥在终端环境中可用。你可以通过运行以下命令来完成此操作：

    ```py
    export OPENAI_API_KEY=your-key
    ```

1.  不要忘记将`your-key`替换为您之前生成的 API 密钥。

1.  如果您想将示例作为 Node.js 脚本运行，请按照[说明](https://oreil.ly/5gjiO)安装 Node。

1.  通过在您的终端中运行以下命令安装 LangChain 库：

    ```py
    npm install langchain @langchain/openai @langchain/community
    npm install @langchain/core pg
    ```

1.  将每个示例保存为*.js*文件，并使用`node ./file.js`运行它。

# 在 LangChain 中使用 LLM

回顾一下，LLM 是大多数生成式 AI 应用的驱动引擎。LangChain 提供了两个简单的接口来与任何 LLM API 提供商交互：

+   聊天模型

+   LLMs

LLM 接口简单地接受一个字符串提示作为输入，将输入发送给模型提供商，然后返回模型预测作为输出。

让我们导入 LangChain 的 OpenAI LLM 包装器，使用简单的提示来`invoke`模型预测：

*Python*

```py
from langchain_openai.llms import OpenAI

model = OpenAI(model="gpt-3.5-turbo")

model.invoke("The sky is")
```

*JavaScript*

```py
import { OpenAI } from "@langchain/openai";

const model = new OpenAI({ model: "gpt-3.5-turbo" });

await model.invoke("The sky is");
```

*输出：*

```py
Blue!
```

###### 小贴士

注意传递给`OpenAI`的参数`model`。这是使用 LLM 或聊天模型时配置的最常见参数，即要使用的底层模型，因为大多数提供商提供具有不同能力和成本折衷的多个模型（通常较大的模型功能更强，但成本更高，速度更慢）。请参阅[OpenAI 提供的模型概述](https://oreil.ly/dM886)。

其他有用的配置参数包括以下内容，大多数提供商都提供。

`temperature`

这控制了用于生成输出的采样算法。较低的值产生更可预测的输出（例如，0.1），而较高的值产生更具创造性或意外的结果（例如，0.9）。不同的任务需要不同的参数值。例如，生成结构化输出通常受益于较低的温度，而创意写作任务则更适合较高的值：

`max_tokens`

这限制了输出的尺寸（和成本）。较低的值可能会导致 LLM 在达到自然结束之前停止生成输出，因此可能看起来被截断了。

除了这些，每个提供商都公开了一组不同的参数。我们建议查看您选择的文档。例如，请参阅[OpenAI 的平台](https://oreil.ly/5O1RW)。

或者，聊天模型界面允许用户和模型之间进行双向对话。之所以是独立的界面，是因为流行的 LLM 提供商，如 OpenAI，将发送给模型和从模型发送的消息区分成*用户*、*助手*和*系统*角色（在这里*角色*表示消息包含的内容类型）：

系统角色

用于模型应使用的指令来回答用户问题

用户角色

用于用户的查询和用户产生的任何其他内容

助手角色

用于模型生成的内容

聊天模型的界面使得在您的 AI 聊天机器人应用程序中配置和管理转换变得更加容易。以下是一个使用 LangChain 的 ChatOpenAI 模型的示例：

*Python*

```py
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI()
prompt = [HumanMessage("What is the capital of France?")]

model.invoke(prompt)
```

*JavaScript*

```py
import { ChatOpenAI } from '@langchain/openai'
import { HumanMessage } from '@langchain/core/messages'

const model = new ChatOpenAI()
const prompt = [new HumanMessage('What is the capital of France?')]

await model.invoke(prompt)
```

*输出：*

```py
AIMessage(content='The capital of France is Paris.')
```

与单个提示字符串不同，聊天模型利用与之前提到的每个角色关联的不同类型的聊天消息接口。以下是一些例子：

`HumanMessage`

从人类用户角色的角度发送的消息

`AIMessage`

从人类与交互的 AI 的角度发送的消息，具有助手角色

`SystemMessage`

设置 AI 应遵循的指令的消息，具有系统角色

`ChatMessage`

允许任意设置角色的消息

让我们在我们的例子中包含一个`SystemMessage`指令：

*Python*

```py
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI()
system_msg = SystemMessage(
    '''You are a helpful assistant that responds to questions with three 
 exclamation marks.'''
)
human_msg = HumanMessage('What is the capital of France?')

model.invoke([system_msg, human_msg])
```

*JavaScript*

```py
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const model = new ChatOpenAI();
const prompt = [
  new SystemMessage(
    `You are a helpful assistant that responds to questions with three 
 exclamation marks.`,
  ),
  new HumanMessage("What is the capital of France?"),
];

await model.invoke(prompt);
```

*输出：*

```py
AIMessage('Paris!!!')
```

如你所见，模型遵循了`SystemMessage`中提供的指令，即使它没有出现在用户的问题中。这使你可以根据用户的输入预先配置你的 AI 应用程序以相对可预测的方式响应。

# 使 LLM 提示可重用

前一节展示了`prompt`指令如何显著影响模型的输出。提示有助于模型理解上下文并生成针对查询的相关答案。

这里是一个详细提示的例子：

```py
Answer the question based on the context below. If the question cannot be
answered using the information provided, answer with "I don't know".

Context: The most recent advancements in NLP are being driven by Large Language 
Models (LLMs). These models outperform their smaller counterparts and have
become invaluable for developers who are creating applications with NLP 
capabilities. Developers can tap into these models through Hugging Face's
`transformers` library, or by utilizing OpenAI and Cohere's offerings through
the `openai` and `cohere` libraries, respectively.

Question: Which model providers offer LLMs?

Answer:
```

尽管提示看起来像是一个简单的字符串，但挑战在于确定文本应该包含什么，以及它应该如何根据用户的输入而变化。在这个例子中，上下文和问题值是硬编码的，但如果我们想动态地传递这些值怎么办？

幸运的是，LangChain 提供了提示模板接口，这使得构建具有动态输入的提示变得容易：

*Python*

```py
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("""Answer the question based on the
 context below. If the question cannot be answered using the information 
 provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)

template.invoke({
    "context": """The most recent advancements in NLP are being driven by Large 
 Language Models (LLMs). These models outperform their smaller 
 counterparts and have become invaluable for developers who are creating 
 applications with NLP capabilities. Developers can tap into these 
 models through Hugging Face's `transformers` library, or by utilizing 
 OpenAI and Cohere's offerings through the `openai` and `cohere` 
 libraries, respectively.""",
    "question": "Which model providers offer LLMs?"
})
```

*JavaS**cript*

```py
import { PromptTemplate } from '@langchain/core/prompts'

const template = PromptTemplate.fromTemplate(`Answer the question based on the 
 context below. If the question cannot be answered using the information 
 provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: `)

await template.invoke({
  context: `The most recent advancements in NLP are being driven by Large 
 Language Models (LLMs). These models outperform their smaller 
 counterparts and have become invaluable for developers who are creating 
 applications with NLP capabilities. Developers can tap into these models 
 through Hugging Face's \`transformers\` library, or by utilizing OpenAI 
 and Cohere's offerings through the \`openai\` and \`cohere\` libraries, 
 respectively.`,
  question: "Which model providers offer LLMs?"
})
```

*输出：*

```py
StringPromptValue(text='Answer the question based on the context below. If the 
    question cannot be answered using the information provided, answer with "I
    don\'t know".\n\nContext: The most recent advancements in NLP are being 
    driven by Large Language Models (LLMs). These models outperform their 
    smaller counterparts and have become invaluable for developers who are 
    creating applications with NLP capabilities. Developers can tap into these 
    models through Hugging Face\'s `transformers` library, or by utilizing 
    OpenAI and Cohere\'s offerings through the `openai` and `cohere` libraries, 
    respectively.\n\nQuestion: Which model providers offer LLMs?\n\nAnswer: ')
```

这个例子将之前块中的静态提示变为动态。`template`包含最终提示的结构以及动态输入插入的定义。

因此，该模板可以用作构建多个静态、特定提示的配方。当你使用一些特定值格式化提示时——在这个例子中，`context`和`question`——你将得到一个准备好的静态提示，可以传递给一个 LLM。

正如你所见，`question`参数通过`invoke`函数动态传递。默认情况下，LangChain 提示遵循 Python 的`f-string`语法来定义动态参数——任何被大括号包围的单词，如`{question}`，都是运行时传入值的占位符。在前一个例子中，`{question}`被替换为`“Which model providers offer LLMs?”`

让我们看看如何使用 LangChain 将这个例子输入到 LLM OpenAI 模型中：

*Python*

```py
from langchain_openai.llms import OpenAI
from langchain_core.prompts import PromptTemplate

# both `template` and `model` can be reused many times

template = PromptTemplate.from_template("""Answer the question based on the 
 context below. If the question cannot be answered using the information 
 provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)

model = OpenAI()

# `prompt` and `completion` are the results of using template and model once

prompt = template.invoke({
    "context": """The most recent advancements in NLP are being driven by Large
 Language Models (LLMs). These models outperform their smaller 
 counterparts and have become invaluable for developers who are creating 
 applications with NLP capabilities. Developers can tap into these 
 models through Hugging Face's `transformers` library, or by utilizing 
 OpenAI and Cohere's offerings through the `openai` and `cohere` 
 libraries, respectively.""",
    "question": "Which model providers offer LLMs?"
})

completion = model.invoke(prompt)
```

*JavaScript*

```py
import { PromptTemplate } from '@langchain/core/prompts'
import { OpenAI } from '@langchain/openai'

const model = new OpenAI()
const template = PromptTemplate.fromTemplate(`Answer the question based on the 
 context below. If the question cannot be answered using the information 
 provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: `)

const prompt = await template.invoke({
  context: `The most recent advancements in NLP are being driven by Large 
 Language Models (LLMs). These models outperform their smaller 
 counterparts and have become invaluable for developers who are creating 
 applications with NLP capabilities. Developers can tap into these models 
 through Hugging Face's \`transformers\` library, or by utilizing OpenAI 
 and Cohere's offerings through the \`openai\` and \`cohere\` libraries, 
 respectively.`,
  question: "Which model providers offer LLMs?"
})

await model.invoke(prompt)
```

*输出：*

```py
Hugging Face's `transformers` library, OpenAI using the `openai` library, and 
Cohere using the `cohere` library offer LLMs.
```

如果你想要构建一个 AI 聊天应用程序，可以使用`ChatPromptTemplate`来提供基于聊天消息角色的动态输入：

*Python*

```py
from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ('system', '''Answer the question based on the context below. If the 
 question cannot be answered using the information provided, answer with 
 "I don\'t know".'''),
    ('human', 'Context: {context}'),
    ('human', 'Question: {question}'),
])

template.invoke({
    "context": """The most recent advancements in NLP are being driven by Large 
 Language Models (LLMs). These models outperform their smaller 
 counterparts and have become invaluable for developers who are creating 
 applications with NLP capabilities. Developers can tap into these 
 models through Hugging Face's `transformers` library, or by utilizing 
 OpenAI and Cohere's offerings through the `openai` and `cohere` 
 libraries, respectively.""",
    "question": "Which model providers offer LLMs?"
})
```

*JavaScript*

```py
import { ChatPromptTemplate } from '@langchain/core/prompts'

const template = ChatPromptTemplate.fromMessages([
  ['system', `Answer the question based on the context below. If the question 
 cannot be answered using the information provided, answer with "I 
 don\'t know".`],
  ['human', 'Context: {context}'],
  ['human', 'Question: {question}'],
])

await template.invoke({
  context: `The most recent advancements in NLP are being driven by Large 
 Language Models (LLMs). These models outperform their smaller 
 counterparts and have become invaluable for developers who are creating 
 applications with NLP capabilities. Developers can tap into these models 
 through Hugging Face's \`transformers\` library, or by utilizing OpenAI 
 and Cohere's offerings through the \`openai\` and \`cohere\` libraries, 
 respectively.`,
  question: "Which model providers offer LLMs?"
})
```

*输出：*

```py
ChatPromptValue(messages=[SystemMessage(content='Answer the question based on 
    the context below. If the question cannot be answered using the information 
    provided, answer with "I don\'t know".'), HumanMessage(content="Context: 
    The most recent advancements in NLP are being driven by Large Language 
    Models (LLMs). These models outperform their smaller counterparts and have 
    become invaluable for developers who are creating applications with NLP 
    capabilities. Developers can tap into these models through Hugging Face\'s 
    `transformers` library, or by utilizing OpenAI and Cohere\'s offerings 
    through the `openai` and `cohere` libraries, respectively."), HumanMessage
    (content='Question: Which model providers offer LLMs?')])
```

注意提示中包含的指令在 `SystemMessage` 中，以及包含动态 `context` 和 `question` 变量的两个 `HumanMessage` 实例。您仍然可以以相同的方式格式化模板，并获取一个静态提示，可以将其传递给大型语言模型以获得预测输出：

*Python*

```py
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# both `template` and `model` can be reused many times

template = ChatPromptTemplate.from_messages([
    ('system', '''Answer the question based on the context below. If the 
 question cannot be answered using the information provided, answer
 with "I don\'t know".'''),
    ('human', 'Context: {context}'),
    ('human', 'Question: {question}'),
])

model = ChatOpenAI()

# `prompt` and `completion` are the results of using template and model once

prompt = template.invoke({
    "context": """The most recent advancements in NLP are being driven by 
 Large Language Models (LLMs). These models outperform their smaller 
 counterparts and have become invaluable for developers who are creating 
 applications with NLP capabilities. Developers can tap into these 
 models through Hugging Face's `transformers` library, or by utilizing 
 OpenAI and Cohere's offerings through the `openai` and `cohere` 
 libraries, respectively.""",
    "question": "Which model providers offer LLMs?"
})

model.invoke(prompt)
```

*JavaScript*

```py
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { ChatOpenAI } from '@langchain/openai'

const model = new ChatOpenAI()
const template = ChatPromptTemplate.fromMessages([
  ['system', `Answer the question based on the context below. If the question 
 cannot be answered using the information provided, answer with "I 
 don\'t know".`],
  ['human', 'Context: {context}'],
  ['human', 'Question: {question}'],
])

const prompt = await template.invoke({
  context: `The most recent advancements in NLP are being driven by Large 
 Language Models (LLMs). These models outperform their smaller 
 counterparts and have become invaluable for developers who are creating 
 applications with NLP capabilities. Developers can tap into these models 
 through Hugging Face's \`transformers\` library, or by utilizing OpenAI 
 and Cohere's offerings through the \`openai\` and \`cohere\` libraries, 
 respectively.`,
  question: "Which model providers offer LLMs?"
})

await model.invoke(prompt)
```

*输出：*

```py
AIMessage(content="Hugging Face's `transformers` library, OpenAI using the 
    `openai` library, and Cohere using the `cohere` library offer LLMs.")
```

# 从 LLM 获取特定格式

纯文本输出很有用，但可能存在需要 LLM 生成结构化输出的用例——即以机器可读的格式输出，例如 JSON、XML、CSV，甚至以 Python 或 JavaScript 等编程语言输出。当您打算将输出传递给其他代码，使 LLM 在您的更大应用中发挥作用时，这非常有用。

## JSON 输出

使用 LLM 生成最常见格式的是 JSON。JSON 输出可以（例如）通过网络发送到您的前端代码或保存到数据库中。

当生成 JSON 时，首先要定义 LLM 在生成输出时需要遵守的模式。然后，您应将该模式包含在提示中，以及您想要用作源文本的文本。让我们看一个例子：

*Python*

```py
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel

class AnswerWithJustification(BaseModel):
    '''An answer to the user's question along with justification for the 
 answer.'''
    answer: str
    '''The answer to the user's question'''
    justification: str
    '''Justification for the answer'''

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)

structured_llm.invoke("""What weighs more, a pound of bricks or a pound 
 of feathers""")
```

*JavaScript*

```py
import { ChatOpenAI } from '@langchain/openai'
import { z } from "zod";

const answerSchema = z
  .object({
    answer: z.string().describe("The answer to the user's question"),
    justification: z.string().describe(`Justification for the 
 answer`),
  })
  .describe(`An answer to the user's question along with justification for 
 the answer.`);

const model = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0,
}).withStructuredOutput(answerSchema)
await model.invoke("What weighs more, a pound of bricks or a pound of feathers")
```

*输出：*

```py
{
  answer: "They weigh the same",
  justification: "Both a pound of bricks and a pound of feathers weigh one pound. 
    The weight is the same, but the volu"... 42 more characters
}
```

因此，首先定义一个模式。在 Python 中，这可以通过 Pydantic（一个用于验证数据与模式匹配的库）最简单地完成。在 JS 中，这可以通过 Zod（一个等效的库）最简单地完成。`with_structured_output` 方法将使用该模式进行两件事：

+   该模式将被转换为 `JSONSchema` 对象（一种用于描述 JSON 数据形状 [类型、名称、描述] 的 JSON 格式），并将其发送到 LLM。对于每个 LLM，LangChain 会选择最佳方法来完成此操作，通常是函数调用或提示。

+   该模式还将用于在返回之前验证 LLM 返回的输出；这确保了生成的输出完全符合您传入的模式。

## 其他机器可读格式与输出解析器

您还可以使用 LLM 或聊天模型以其他格式生成输出，例如 CSV 或 XML。这就是输出解析器派上用场的地方。*输出解析器* 是帮助您结构化大型语言模型响应的类。它们有两个功能：

提供格式指令

可以使用输出解析器在提示中注入一些额外的指令，这将有助于引导 LLM 以其能解析的格式输出文本。

验证和解析输出

主要功能是将 LLM 或聊天模型的文本输出渲染为更结构化的格式，例如列表、XML 或其他格式。这可能包括删除无关信息、纠正不完整的输出以及验证解析的值。

下面是一个输出解析器的工作示例：

*Python*

```py
from langchain_core.output_parsers import CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()
items = parser.invoke("apple, banana, cherry")
```

*JavaScript*

```py
import { CommaSeparatedListOutputParser } from '@langchain/core/output_parsers'

const parser = new CommaSeparatedListOutputParser()

await parser.invoke("apple, banana, cherry")
```

*输出：*

```py
['apple', 'banana', 'cherry']
```

LangChain 为各种用例提供了各种输出解析器，包括 CSV、XML 等。我们将在下一节中看到如何将输出解析器与模型和提示结合使用。

# 组装 LLM 应用程序的多个部分

你迄今为止学到的关键组件是 LangChain 框架的基本构建块。这引出了关键问题：你如何有效地将它们组合起来以构建你的 LLM 应用程序？

## 使用可运行接口

如你所注意到的，迄今为止使用的所有代码示例都使用了类似的接口和 `invoke()` 方法来从模型（或提示模板，或输出解析器）生成输出。所有组件都具有以下功能：

+   这些方法有一个共同的接口：

    +   `invoke`：将单个输入转换为输出

    +   `batch`：高效地将多个输入转换为多个输出

    +   `stream`：以流的形式从单个输入中输出，当输出产生时

+   有内置的重试、回退、模式以及运行时配置的实用工具。

+   在 Python 中，这三个方法都有 `asyncio` 的等效方法。

因此，所有组件的行为方式相同，为其中一个组件学习到的接口适用于所有：

*Python*

```py
from langchain_openai.llms import ChatOpenAI

model = ChatOpenAI()

completion = model.invoke('Hi there!') 
# Hi!

completions = model.batch(['Hi there!', 'Bye!'])
# ['Hi!', 'See you!']

for token in model.stream('Bye!'):
    print(token)
    # Good
    # bye
    # !
```

*JavaScript*

```py
import { ChatOpenAI } from '@langchain/openai'

const model = new ChatOpenAI()

const completion = await model.invoke('Hi there!') 
// Hi!

const completions = await model.batch(['Hi there!', 'Bye!'])
// ['Hi!', 'See you!']

for await (const token of await model.stream('Bye!')) {
  console.log(token)
  // Good
  // bye
  // !
}
```

在这个例子中，你可以看到三种主要方法是如何工作的：

+   `invoke()` 接受单个输入并返回单个输出。

+   `batch()` 接受一个输出列表并返回一个输出列表。

+   `stream()` 接受单个输入并返回输出部分的迭代器，当输出可用时。

在某些情况下，如果底层组件不支持迭代输出，将会有一个包含所有输出的单个部分。

你可以通过两种方式组合这些组件：

命令式

直接调用你的组件，例如，使用 `model.invoke(...)`。

声明式

使用 LangChain 表达式语言（LCEL），如即将介绍的章节所述

表 1-1 总结了它们之间的区别，我们将在接下来的操作中看到每个示例。

表 1-1\. 命令式和声明式组合的主要区别。

|   | 命令式 | 声明式 |
| --- | --- | --- |
| 语法 | Python 或 JavaScript 的全部 | LCEL |
| 并行执行 | Python：使用线程或协程 JavaScript：使用 `Promise.all` | 自动 |
| 流式传输 | 使用 `yield` 关键字 | 自动 |
| 异步执行 | 使用异步函数 | 自动 |

## 命令式组合

*命令式组合* 只是一个花哨的名称，指的是编写你习惯的代码，将这些组件组合成函数和类。以下是一个结合提示、模型和输出解析器的示例：

*Python*

```py
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# the building blocks

template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('human', '{question}'),
])

model = ChatOpenAI()

# combine them in a function
# @chain decorator adds the same Runnable interface for any function you write

@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)

# use it

chatbot.invoke({"question": "Which model providers offer LLMs?"})
```

*JavaScript*

```py
import {ChatOpenAI} from '@langchain/openai'
import {ChatPromptTemplate} from '@langchain/core/prompts'
import {RunnableLambda} from '@langchain/core/runnables'

// the building blocks

const template = ChatPromptTemplate.fromMessages([
  ['system', 'You are a helpful assistant.'],
  ['human', '{question}'],
])

const model = new ChatOpenAI()

// combine them in a function
// RunnableLambda adds the same Runnable interface for any function you write

const chatbot = RunnableLambda.from(async values => {
  const prompt = await template.invoke(values)
  return await model.invoke(prompt)
})

// use it

await chatbot.invoke({
  "question": "Which model providers offer LLMs?"
})
```

*输出：*

```py
AIMessage(content="Hugging Face's `transformers` library, OpenAI using the 
    `openai` library, and Cohere using the `cohere` library offer LLMs.")
```

前面的例子是一个完整的聊天机器人示例，使用提示和聊天模型。正如你所看到的，它使用了熟悉的 Python 语法并支持你可能在那个函数中添加的任何自定义逻辑。

另一方面，如果你想启用流式传输或异步支持，你必须修改你的函数以支持它。例如，可以通过以下方式添加流式传输支持：

*Python*

```py
@chain
def chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token

for part in chatbot.stream({
    "question": "Which model providers offer LLMs?"
}):
    print(part)
```

*JavaScript*

```py
const chatbot = RunnableLambda.from(async function* (values) {
  const prompt = await template.invoke(values)
  for await (const token of await model.stream(prompt)) {
    yield token
  }
})

for await (const token of await chatbot.stream({
  "question": "Which model providers offer LLMs?"
})) {
  console.log(token)
}
```

*输出：*

```py
AIMessageChunk(content="Hugging")
AIMessageChunk(content=" Face's")
AIMessageChunk(content=" `transformers`")
...
```

因此，无论是使用 JS 还是 Python，你都可以通过返回你想要流式的值并使用 `stream` 来调用它来为你的自定义函数启用流式传输。

对于异步执行，你需要这样重写你的函数：

*Python*

```py
@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)

await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})
# > AIMessage(content="""Hugging Face's `transformers` library, OpenAI using
    the `openai` library, and Cohere using the `cohere` library offer LLMs.""")
```

这个只适用于 Python，因为在 JavaScript 中异步执行是唯一的选择。

## 声明式组合

LCEL 是一种用于组合 LangChain 组件的*声明式语言*。LangChain 将 LCEL 组合编译成*优化后的执行计划*，具有自动并行化、流式处理、跟踪和异步支持。

让我们用 LCEL 来查看相同的例子：

*Python*

```py
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# the building blocks

template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('human', '{question}'),
])

model = ChatOpenAI()

# combine them with the | operator

chatbot = template | model

# use it

chatbot.invoke({"question": "Which model providers offer LLMs?"})
```

*JavaScript*

```py
import { ChatOpenAI } from '@langchain/openai'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { RunnableLambda } from '@langchain/core/runnables'

// the building blocks

const template = ChatPromptTemplate.fromMessages([
  ['system', 'You are a helpful assistant.'],
  ['human', '{question}'],
])

const model = new ChatOpenAI()

// combine them in a function

const chatbot = template.pipe(model)

// use it

await chatbot.invoke({
  "question": "Which model providers offer LLMs?"
})
```

*输出：*

```py
AIMessage(content="Hugging Face's `transformers` library, OpenAI using the 
    `openai` library, and Cohere using the `cohere` library offer LLMs.")
```

关键的是，两个例子中的最后一行是相同的——也就是说，你使用函数和 LCEL 序列的方式相同，使用`invoke/stream/batch`。在这个版本中，你不需要做任何事情就可以使用流式处理：

*Python*

```py
chatbot = template | model

for part in chatbot.stream({
    "question": "Which model providers offer LLMs?"
}):
    print(part)
    # > AIMessageChunk(content="Hugging")
    # > AIMessageChunk(content=" Face's")
    # > AIMessageChunk(content=" `transformers`")
    # ...
```

*JavaScript*

```py
const chatbot = template.pipe(model)

for await (const token of await chatbot.stream({
  "question": "Which model providers offer LLMs?"
})) {
  console.log(token)
}
```

并且，对于 Python 来说，使用异步方法也是一样的：

*Python*

```py
chatbot = template | model

await chatbot.ainvoke({
    "question": "Which model providers offer LLMs?"
})
```

# 摘要

在本章中，你学习了构建 LLM 应用程序所需的构建块和关键组件，使用 LangChain 构建 LLM 应用程序。LLM 应用程序本质上是一个由大型语言模型（用于做出预测）、提示指令（用于引导模型达到期望的输出）以及可选的输出解析器（用于转换模型输出的格式）组成的链。

所有 LangChain 组件都共享相同的接口，使用`invoke`、`stream`和`batch`方法来处理各种输入和输出。它们可以通过直接调用或使用 LCEL 声明式地组合和执行。

如果你想编写大量的自定义逻辑，命令式方法是有用的，而声明式方法则适用于简单地组装现有组件，且定制化有限。

在第二章中，你将学习如何提供外部数据作为*上下文*给你的 AI 聊天机器人，这样你就可以构建一个 LLM 应用程序，让你能够“聊天”你的数据。
