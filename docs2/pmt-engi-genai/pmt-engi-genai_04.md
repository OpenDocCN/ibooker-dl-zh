# 第四章\. 使用 LangChain 进行文本生成的高级技术

使用简单的提示工程技术通常适用于大多数任务，但有时你需要使用更强大的工具包来解决复杂的生成式 AI 问题。这些问题和任务包括：

上下文长度

将整本书总结成可消化的摘要。

结合 LLM 的输入/输出序列

为一本书创造故事，包括角色、情节和世界观构建。

执行复杂的推理任务

LLM 作为代理。例如，你可以创建一个 LLM 代理来帮助你实现个人健身目标。

要熟练应对这样的复杂生成式 AI 挑战，熟悉开源框架 LangChain 非常有帮助。这个工具极大地简化并增强了你的 LLM 的工作流程。

# LangChain 简介

LangChain 是一个多功能的框架，它使创建利用 LLM 的应用程序成为可能，并且作为[Python](https://oreil.ly/YPid-)和[TypeScript](https://oreil.ly/5Vl0W)包提供。其核心原则是，最有影响力和独特的应用程序不仅会通过 API 与语言模型接口，而且还会：

提高数据意识

该框架旨在在语言模型和外部数据源之间建立无缝连接。

增强代理能力

它致力于使语言模型能够与环境互动并产生影响。

如图 4-1 所示，LangChain 框架提供了一系列模块化抽象，这对于与 LLM 一起工作至关重要，以及这些抽象的广泛实现。

![pega 0401](img/pega_0401.png)

###### 图 4-1\. LangChain LLM 框架的主要模块

每个模块都设计得易于使用，可以独立或一起高效地利用。目前 LangChain 中有六个常见的模块：

模型 I/O

处理与模型相关的输入/输出操作

检索

专注于检索对 LLM 相关的文本

链

也称为*LangChain 可运行链*，链允许构建 LLM 操作或函数调用的序列

代理

允许链根据高级指令或指示做出使用哪些工具的决定

内存

在链的不同运行之间保持应用程序的状态

回调

在特定事件上运行附加代码，例如每次生成新标记时

## 环境设置

你可以使用以下任一命令在你的终端上安装 LangChain：

+   `pip install langchain langchain-openai`

+   `conda install -c conda-forge langchain langchain-openai`

如果你希望安装整本书的包要求，可以使用 GitHub 仓库中的[*requirements.txt*](https://oreil.ly/WKOma)文件。

建议在虚拟环境中安装包：

创建虚拟环境

`python -m venv venv`

激活虚拟环境

`source venv/bin/activate`

安装依赖项

`pip install -r requirements.txt`

LangChain 需要与一个或多个模型提供商进行集成。例如，要使用 OpenAI 的模型 API，您需要使用`pip install openai`安装他们的 Python 包。

如第一章中所述，最佳实践是在您的终端中设置一个名为`OPENAI_API_KEY`的环境变量，或使用`python-dotenv`从*.env*文件中加载它（[python-dotenv](https://oreil.ly/wvuO7)）。然而，对于原型设计，您可以选择跳过此步骤，通过在 LangChain 中加载聊天模型时直接传递您的 API 密钥：

```py
from langchain_openai.chat_models import ChatOpenAI
chat = ChatOpenAI(api_key="api_key")
```

###### 警告

由于安全原因，不建议在脚本中硬编码 API 密钥。相反，利用环境变量或配置文件来管理您的密钥。

在 LLM 不断演变的领域中，您可能会遇到不同模型 API 之间差异的挑战。接口缺乏标准化可能会在提示工程中引入额外的复杂性层，并阻碍不同模型无缝集成到您的项目中。

这就是 LangChain 发挥作用的地方。作为一个全面的框架，LangChain 允许您轻松消费不同模型的多种接口。

LangChain 的功能确保您在切换模型时不需要重新发明提示或代码。其平台无关的方法促进了广泛模型的快速实验，例如[Anthropic](https://www.anthropic.com)、[Vertex AI](https://cloud.google.com/vertex-ai)、[OpenAI](https://openai.com)和[BedrockChat](https://oreil.ly/bedrock)。这不仅加快了模型评估过程，而且通过简化复杂的模型集成，节省了宝贵的时间和资源。

在接下来的章节中，您将使用 OpenAI 包及其在 LangChain 中的 API。

# 聊天模型

如 GPT-4 之类的聊天模型已成为与 OpenAI API 交互的主要方式。它们不是提供简单的“输入文本，输出文本”响应，而是提出一种交互方法，其中*聊天消息*是输入和输出元素。

使用聊天模型生成 LLM 响应涉及将一个或多个消息输入到聊天模型中。在 LangChain 的语境中，目前接受的消息类型有`AIMessage`、`HumanMessage`和`SystemMessage`。聊天模型的输出始终是`AIMessage`。

SystemMessage

代表应作为 AI 系统指令的信息。这些用于以某种方式指导 AI 的行为或行动。

HumanMessage

代表来自与 AI 系统交互的人类信息。这可能是一个问题、一个命令或任何其他来自人类用户的信息，AI 需要处理并响应。

AIMessage

代表来自 AI 系统本身的信息。这通常是 AI 对`HumanMessage`或`SystemMessage`指令的响应。

###### 注意

确保利用`SystemMessage`来传达明确的指示。OpenAI 已经改进了 GPT-4 和即将推出的 LLM 模型，特别关注此类消息中给出的指南。

让我们在 LangChain 中创建一个笑话生成器。

输入：

```py
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0.5)
messages = [SystemMessage(content='''Act as a senior software engineer
at a startup company.'''),
HumanMessage(content='''Please can you provide a funny joke
about software engineers?''')]
response = chat.invoke(input=messages)
print(response.content)
```

输出：

```py
Sure, here's a lighthearted joke for you: `Why` `did` `the` `software` `engineer` `go` `broke``?`
`Because` `he` `lost` `his` `domain` `in` `a` `bet` `and` `couldn``'t afford to renew it.`
```
