# 第二章：深入了解 GPT-4 和 ChatGPT 的 API

本章将详细介绍 GPT-4 和 ChatGPT 的 API。本章的目标是让您对这些 API 的使用有扎实的理解，以便您可以有效地将它们集成到您的 Python 应用程序中。通过本章的学习，您将能够充分利用这些 API 在自己的开发项目中的强大功能。

我们将从介绍 OpenAI Playground 开始。这将使您在编写任何代码之前更好地了解模型。接下来，我们将看看 OpenAI Python 库。这包括登录信息和一个简单的“Hello World”示例。然后，我们将介绍创建和发送 API 请求的过程。我们还将看看如何管理 API 响应。这将确保您知道如何解释这些 API 返回的数据。此外，本章还将涵盖安全最佳实践和成本管理等考虑因素。

随着我们的进展，您将获得实用的知识，这对您作为与 GPT-4 和 ChatGPT 一起使用的 Python 开发人员的旅程非常有用。本章中包含的所有 Python 代码都可以在[本书的 GitHub 存储库](https://oreil.ly/DevAppsGPT_GitHub)中找到。

###### 注意

在继续之前，请查看[OpenAI 使用政策](https://openai.com/policies/usage-policies)，如果您还没有帐户，请在[OpenAI 主页](https://openai.com)上创建一个。您还可以查看[条款和政策页面](https://openai.com/policies)上的其他法律文件。第一章中介绍的概念对于使用 OpenAI API 和库也是必不可少的。

# 基本概念

OpenAI 提供了几种为各种任务设计的模型，每种模型都有自己的定价。在接下来的页面上，您将找到可用模型的详细比较以及如何选择使用哪些模型的提示。重要的是要注意，模型设计的目的——无论是用于文本完成、聊天还是编辑——都会影响您如何使用其 API。例如，ChatGPT 和 GPT-4 背后的模型是基于聊天的，并使用聊天端点。

提示的概念是在第一章中介绍的。提示不是特定于 OpenAI API，但是所有 LLM 的入口点。简而言之，提示是您发送给模型的输入文本，用于指示模型执行特定任务。对于 ChatGPT 和 GPT-4 模型，提示具有聊天格式，输入和输出消息存储在列表中。我们将在本章中探讨此提示格式的详细信息。

令牌的概念也在第一章中描述过。令牌是单词或单词的部分。粗略估计是，100 个令牌大约相当于英文文本的 75 个单词。对 OpenAI 模型的请求是基于使用的令牌数量定价的：也就是说，对 API 的调用成本取决于输入文本和输出文本的长度。您将在“使用 ChatGPT 和 GPT-4”和“使用其他文本完成模型”中找到有关管理和控制输入和输出令牌数量的更多详细信息。

这些概念在图 2-1 中进行了总结。

![](img/dev-app-gpt4-cgpt-dagc_0201.png)

###### 图 2-1\. 使用 OpenAI API 的基本概念

现在我们已经讨论了概念，让我们转向模型的细节。

# OpenAI API 中可用的模型

OpenAI API 为您提供了访问[OpenAI 开发的多个模型](https://platform.openai.com/docs/models)。这些模型可作为 API 的服务使用（通过直接的 HTTP 调用或提供的库），这意味着 OpenAI 在远程服务器上运行模型，开发人员只需向其发送查询。

每个模型都具有不同的功能和定价。在本节中，我们将看一下 OpenAI 通过其 API 提供的 LLM。需要注意的是，这些模型是专有的，因此您不能直接修改代码以适应您的需求。但正如我们将在后面看到的，您可以通过 OpenAI API 对其中一些模型进行特定数据的微调。

###### 注意

一些较旧的 OpenAI 模型，包括 GPT-2 模型，不是专有的。虽然您可以从[Hugging Face](https://oreil.ly/39Bu5)或[GitHub](https://oreil.ly/CYPN6)下载 GPT-2 模型，但无法通过 API 访问它。

由于 OpenAI 提供的许多模型都在不断更新，因此在本书中很难给出完整的模型列表；OpenAI 提供的模型的更新列表可在[在线文档](https://platform.openai.com/docs/models)中找到。因此，我们将重点放在最重要的模型上：

InstructGPT

这一系列模型可以处理许多单轮完成任务。`text-ada-001`模型只能完成简单的完成任务，但也是 GPT-3 系列中最快速和最便宜的模型。`text-babbage-001`和`text-curie-001`稍微更强大，但也更昂贵。`text-davinci-003`模型可以以优秀的质量执行所有完成任务，但也是 GPT-3 模型系列中最昂贵的模型。

ChatGPT

ChatGPT 背后的模型是`gpt-3.5-turbo`。作为聊天模型，它可以将一系列消息作为输入，并返回一个适当生成的消息作为输出。虽然`gpt-3.5-turbo`的聊天格式旨在促进多轮对话，但也可以将其用于没有对话的单轮任务。在单轮任务中，`gpt-3.5-turbo`的性能与`text-davinci-003`相当，由于`gpt-3.5-turbo`的价格是后者的十分之一，性能几乎相当，建议您默认使用它进行单轮任务。`gpt-3.5-turbo`模型的上下文大小为 4,000 个标记，这意味着它可以接收 4,000 个标记作为输入。OpenAI 还提供另一个模型，名为`gpt-3.5-turbo-16k`，具有与标准`gpt-3.5-turbo`模型相同的功能，但上下文大小增加了四倍。

GPT-4

这是 OpenAI 发布的最大模型。它还在最广泛的文本和图像多模态语料库上进行了训练。因此，它在许多领域都具有知识和专业知识。GPT-4 能够遵循复杂的自然语言指令并准确解决困难问题。它可以用于具有高准确性的聊天和单轮任务。OpenAI 提供了两个 GPT-4 模型：`gpt-4`的上下文大小为 8,000 个标记，`gpt-4-32k`的上下文大小为 32,000 个标记。32,000 的上下文大约代表 24,000 个单词，大约相当于 40 页的上下文。

GPT-3.5 Turbo 和 GPT-4 都在不断更新。当我们提到`gpt-3.5-turbo`、`gpt-3.5-turbo-16k`、`gpt-4`和`gpt-4-32k`模型时，我们指的是这些模型的最新版本。

开发人员通常需要更稳定和可见性的 LLM 版本，以在其应用程序中使用。对于开发人员来说，使用版本可能会在一夜之间发生变化，并且对于相同的输入提示可能会有不同的行为，这可能会很困难。出于这个目的，这些模型的静态快照版本也是可用的。在撰写本文时，最新的快照版本是`gpt-3.5-turbo-0613`、`gpt-3.5-turbo-16k-0613`、`gpt-4-0613`和`gpt-4-32k-0613`。

正如在第一章中讨论的，OpenAI 建议使用 InstructGPT 系列而不是原始的基于 GPT-3 的模型。这些模型仍然在 API 中以`davinci`、`curie`、`babbage`和`ada`的名称提供。鉴于这些模型可能会提供奇怪、错误和误导性的答案，因此建议在使用时要谨慎。但是，这些模型仍然被使用，因为它们是唯一可以对您的数据进行微调的模型。在撰写本文时，OpenAI 宣布 GPT-3.5 Turbo 和 GPT-4 的微调将在 2024 年推出。

###### 注意

在经过监督微调阶段获得的 SFT 模型（在第一章中介绍）也可以在 API 中以`davinci-instruct-beta`的名称使用。

# 使用 OpenAI Playground 尝试 GPT 模型

直接测试 OpenAI 提供的不同语言模型的绝佳方法，而无需编码，是使用 OpenAI Playground，这是一个基于 Web 的平台，允许您快速测试 OpenAI 提供的各种 LLM 在特定任务上的表现。Playground 允许您编写提示，选择模型，并轻松查看生成的输出。

访问 Playground 的方法如下：

1.  转到[OpenAI 主页](https://openai.com)，单击开发人员，然后单击概述。

1.  如果您已经有账户但未登录，请单击屏幕右上方的登录。如果您没有 OpenAI 账户，您需要创建一个才能使用 Playground 和大多数 OpenAI 功能。请单击屏幕右上方的注册。请注意，由于 Playground 和 API 会收费，因此您需要提供支付方式。

1.  登录后，您将在网页的左上方看到加入 Playground 的链接。单击该链接，您应该会看到类似于图 2-2 的内容。

![](img/dev-app-gpt4-cgpt-dagc_0202.png)

###### 图 2-2. 文本完成模式下的 OpenAI Playground 界面

###### 注意

ChatGPT Plus 选项与使用 API 或 Playground 无关。如果您订阅了 ChatGPT Plus 服务，那么使用 API 和 Playground 仍会产生费用。

界面中心的主要空白是用于输入消息的。在编写消息后，单击提交以生成消息的完成。在图 2-2 的示例中，我们写下“正如笛卡尔所说，我思故我在”，然后单击提交后，模型用“我是”完成了我们的输入。

###### 警告

每次单击提交，您的 OpenAI 账户都会收取使用费。我们将在本章后面提供有关价格的更多信息，但举例来说，这个完成大约花费了$0.0002。

界面的各个部分都有许多选项。让我们从底部开始。在提交按钮的右侧是一个撤消按钮[图中标记为(A)]，用于删除最后生成的文本。在我们的情况下，它将删除“我是”。接下来是重新生成按钮[图中标记为(B)]，用于重新生成刚刚删除的文本。然后是历史按钮[标记为(C)]，其中包含了您在过去 30 天内的所有请求。请注意，一旦进入历史菜单，根据隐私原因，有必要时可以轻松删除请求。

屏幕右侧的选项面板提供了与界面和所选模型相关的各种设置。我们只会在这里解释其中一些选项；其他选项将在本书的后面介绍。右侧的第一个下拉列表是模式列表[labeled (D)]。在撰写本文时，可用的模式是 Chat（默认）、Complete 和 Edit。

###### 注意

在本书撰写时，Complete 和 Edit 模式被标记为传统模式，并且可能会在 2024 年 1 月消失。

如前所示，语言模型致力于在 Playground 的默认模式下无缝完成用户的输入提示。

图 2-3 显示了在 Chat 模式下使用 Playground 的示例。屏幕左侧是系统窗格标记为(E)]。在这里，您可以描述聊天系统的行为方式。例如，在[图 2-3 中，我们要求它成为一个热爱猫的乐于助人的助手。我们还要求它只谈论猫，并给出简短的回答。设置这些参数所产生的对话显示在屏幕中央。

如果您想继续与系统对话，点击“添加消息”[(F)]，输入您的消息，然后点击提交[(G)]。也可以在右侧定义模型[(H)]；这里我们使用 GPT-4。请注意，并非所有模型都在所有模式中可用。例如，只有 GPT-4 和 GPT-3.5 Turbo 在 Chat 模式中可用。

![](img/dev-app-gpt4-cgpt-dagc_0203.png)

###### 图 2-3。Chat 模式下的 OpenAI Playground 界面

Playground 中的另一种模式是 Edit。在这种模式下，如图 2-4 所示，您提供一些文本[(I)]和指示[(J)]，模型将尝试相应地修改文本。在这个例子中，给出了描述一个年轻男子要去旅行的文本。模型被指示将文本的主题改为一个老妇人，您可以看到结果符合指示[(K)]。

![](img/dev-app-gpt4-cgpt-dagc_0204.png)

###### 图 2-4。Edit 模式下的 OpenAI Playground 界面

在 Playground 界面的右侧，在 Mode 下拉列表下方，是 Model 下拉列表[(L)]。正如您已经看到的，这是您选择 LLM 的地方。下拉列表中可用的模型取决于所选的模式。在 Model 下拉列表下方是参数，比如温度[(M)]，它们定义了模型的行为。我们不会在这里详细讨论这些参数。当我们仔细研究这些不同的模型如何工作时，大部分参数将被探索。

屏幕顶部是“加载预设”下拉列表(N)]和四个按钮。在图 2-2 中，我们使用 LLM 来完成句子“正如笛卡尔所说，我思故我在”，但是可以通过使用适当的提示使模型执行特定任务。图 2-5 显示了模型可以执行的常见任务列表，以及与预设示例相关联的示例。

![###### 图 2-5。示例的下拉列表值得注意的是，所提议的预设不仅定义了提示，还定义了屏幕右侧的一些选项。例如，如果您点击 Grammatical Standard English，您将在主窗口中看到图 2-6 中显示的提示。![](img/dev-app-gpt4-cgpt-dagc_0206.png)

###### 图 2-6。Grammatical Standard English 的示例提示

如果您点击提交，您将得到以下回复：“她没有去市场。”您可以使用下拉列表中提供的提示作为起点，但您总是需要修改它们以适应您的问题。OpenAI 还为不同任务提供了[完整的示例列表](https://platform.openai.com/examples)。

在图 2-4 中“加载预设”下拉列表旁边是保存按钮[(O)]。想象一下，您已经为您的任务定义了一个有价值的提示，还有一个模型和它的参数，您希望以后在 Playground 中轻松地重用它。这个保存按钮将保存 Playground 的当前状态为一个预设。您可以为您的预设命名和描述，一旦保存，您的预设将出现在“加载预设”下拉列表中。

界面顶部的倒数第二个按钮称为“查看代码”[(P)]。它提供了在 Playground 中直接以脚本运行您的测试的代码。您可以请求 Python、Node.js 或 cURL 代码，以直接与 Linux 终端中的 OpenAI 远程服务器进行交互。如果我们要求我们的示例“As Descartes said, I think therefore”的 Python 代码，我们将得到以下内容：

```py
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="As Descartes said, I think therefore",
    temperature=0.7,
    max_tokens=3,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)
```

现在您了解了如何使用 Playground 在不编写代码的情况下测试 OpenAI 语言模型，让我们讨论如何获取和管理 OpenAI 服务的 API 密钥。

# 入门：OpenAI Python 库

在本节中，我们将重点介绍如何在一个小的 Python 脚本中使用 API 密钥，并使用 OpenAI API 进行我们的第一次测试。

OpenAI 提供 GPT-4 和 ChatGPT 作为服务。这意味着用户无法直接访问模型的代码，也无法在自己的服务器上运行模型。但是，OpenAI 管理模型的部署和运行，用户可以调用这些模型，只要他们有帐户和秘密密钥。

在完成以下步骤之前，请确保您已登录[OpenAI 网页](https://platform.openai.com/login?launch)。

## OpenAI 访问和 API 密钥

OpenAI 要求您拥有 API 密钥才能使用其服务。此密钥有两个目的：

+   它赋予您调用 API 方法的权利。

+   它将您的 API 调用与您的帐户关联起来，以进行计费。

您必须拥有此密钥才能从您的应用程序调用 OpenAI 服务。

要获取密钥，请导航到 OpenAI 平台](https://platform.openai.com)页面。在右上角，点击您的帐户名称，然后点击“查看 API 密钥”，如图 2-7 所示。

![

###### 图 2-7. OpenAI 菜单选择“查看 API 密钥”

当您在“API 密钥”页面上时，点击“创建新的秘密密钥”并复制您的密钥。此密钥是以*sk-*开头的一长串字符。

###### 警告

请妥善保管此密钥，因为它直接与您的帐户相关联，而且被盗的密钥可能会导致不必要的费用。

一旦您获得了您的密钥，最佳做法是将其导出为环境变量。这将允许您的应用程序访问密钥，而无需直接在代码中编写它。以下是如何做到这一点。

对于 Linux 或 Mac：

```py
# set environment variable OPENAI_API_KEY for current session
export OPENAI_API_KEY=sk-(...)
# check that environment variable was set
echo $OPENAI_API_KEY
```

对于 Windows：

```py
# set environment variable OPENAI_API_KEY for current session
set OPENAI_API_KEY=sk-(...)
# check that environment variable was set
echo %OPENAI_API_KEY%
```

上述代码片段将设置一个环境变量，并使您的密钥可用于从同一 shell 会话启动的其他进程。对于 Linux 系统，还可以直接将此代码添加到您的*.bashrc*文件中。这将允许在所有 shell 会话中访问您的环境变量。当然，不要将这些命令行包含在您推送到公共存储库的代码中。

要在 Windows 11 中永久添加/更改环境变量，请同时按下 Windows 键+R 键以打开运行程序或文件窗口。在此窗口中，键入**sysdm.cpl**以转到系统属性面板。然后点击高级选项卡，然后点击环境变量按钮。在结果屏幕上，您可以使用您的 OpenAI 密钥添加新的环境变量。

###### 提示

OpenAI 提供了一个关于 API 密钥安全的详细页面。

既然您已经有了您的密钥，现在是时候使用 OpenAI API 编写您的第一个“Hello World”程序了。

## “Hello World”示例

本节显示了使用 OpenAI Python 库的第一行代码。我们将从一个经典的“Hello World”示例开始，以了解 OpenAI 如何提供其服务。

使用*pip*安装 Python 库：

```py
pip install openai
```

接下来，在 Python 中访问 OpenAI API：

```py
import openai
# Call the openai ChatCompletion endpoint
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello World!"}],
)
# Extract the response
print(response["choices"][0]["message"]["content"])
```

您将看到以下输出：

```py
Hello there! How may I assist you today?
```

恭喜！您刚刚使用 OpenAI Python 库编写了您的第一个程序。

让我们详细了解如何使用这个库。

###### 提示

OpenAI Python 库还提供了一个命令行实用程序。在终端中运行以下代码等同于执行前面的“Hello World”示例：

```py
openai api chat_completions.create -m gpt-3.5-turbo \
    -g user "Hello world"
```

还可以通过 HTTP 请求或官方 Node.js 库以及其他[社区维护的库](https://platform.openai.com/docs/libraries)与 OpenAI API 进行交互。

您可能已经注意到，代码片段并未明确提及 OpenAI API 密钥。这是因为 OpenAI 库被设计为自动查找名为`OPENAI_API_KEY`的环境变量。或者，您可以使用以下代码将`openai`模块指向包含您的密钥的文件：

```py
# Load your API key from file
openai.api_key_path = <PATH>, 
```

或者您可以使用以下方法在代码中手动设置 API 密钥：

```py
# Load your API key 
openai.api_key = os.getenv("OPENAI_API_KEY")
```

我们建议遵循环境变量的广泛约定：将密钥存储在*.env*文件中，并在*.gitignore*文件中将其从源代码控制中删除。然后，在 Python 中，您可以运行`load_dotenv`函数来加载环境变量并导入*openai*库：

```py
from dotenv import load_dotenv
load_dotenv()
import openai
```

在加载*.env*文件后，重要的是要有`openai`导入声明；否则，OpenAI 的设置将无法正确应用。

现在我们已经介绍了 ChatGPT 和 GPT-4 的基本概念，我们可以继续了解它们的使用细节。

# 使用 ChatGPT 和 GPT-4

本节讨论了如何使用 OpenAI Python 库后台运行的模型与 ChatGPT 和 GPT-4。

在撰写本文时，GPT 3.5 Turbo 是最便宜且最多功能的模型。因此，它也是大多数用例的最佳选择。以下是其使用示例：

```py
import openai
# For GPT 3.5 Turbo, the endpoint is ChatCompletion
openai.ChatCompletion.create(
    # For GPT 3.5 Turbo, the model is "gpt-3.5-turbo"
    model="gpt-3.5-turbo",
    # Conversation as a list of messages.
    messages=[
        {"role": "system", "content": "You are a helpful teacher."},
        {
            "role": "user",
            "content": "Are there other measures than time complexity for an \
 algorithm?",
        },
        {
            "role": "assistant",
            "content": "Yes, there are other measures besides time complexity \
 for an algorithm, such as space complexity.",
        },
        {"role": "user", "content": "What is it?"},
    ],
)
```

在上面的示例中，我们使用了最少数量的参数，即用于预测的 LLM 和输入消息。正如您所看到的，输入消息中的对话格式允许将多个交换发送到模型。请注意，API 不会在其上下文中存储先前的消息。问题`"它是什么？"`是指先前的答案，只有在模型知道这个答案的情况下才有意义。每次都必须发送整个对话以模拟聊天会话。我们将在下一节中进一步讨论这一点。

GPT 3.5 Turbo 和 GPT-4 模型针对聊天会话进行了优化，但这并非强制要求。这两个模型都可以用于多轮对话和单轮任务。如果您指定一个提示来请求完成，它们也可以很好地完成传统的完成任务。

ChatGPT 和 GPT-4 都使用相同的端点：`openai.ChatCompletion`。更改模型 ID 允许开发人员在不进行其他代码更改的情况下在 GPT-3.5 Turbo 和 GPT-4 之间切换。

## Chat Completion 端点的输入选项

让我们更详细地看一下如何使用`openai.ChatCompletion`端点及其`create`方法。

###### 注意

`create`方法允许用户调用 OpenAI 模型。还有其他可用的方法，但对于与模型交互并不有用。您可以在 OpenAI 的 GitHub [Python 库存储库](https://oreil.ly/MQ2aQ)上访问 Python 库代码。

### 必需的输入参数

`openai.ChatCompletion`端点及其`create`方法有几个输入参数，但只有两个是必需的，如表 2-1 中所述。

表 2-1\. 必需的输入参数

| 字段名称 | 类型 | 描述 |
| --- | --- | --- |
| `model` | String | 要使用的模型 ID。目前可用的模型有`gpt-4`、`gpt-4-0613`、`gpt-4-32k`、`gpt-4-32k-0613`、`gpt-3.5-turbo`、`gpt-3.5-turbo-0613`、`gpt-3.5-turbo-16k`和`gpt-3.5-turbo-16k-0613`。可以使用 OpenAI 提供的另一个端点和方法`openai.Model.list()`来访问可用模型的列表。请注意，并非所有可用模型都与`openai.ChatCompletion`端点兼容。 |
| `messages` | Array | 代表对话的`message`对象数组。`message`对象有两个属性：`role`（可能的值为`system`、`user`和`assistant`）和`content`（包含对话消息的字符串）。 |

对话以可选的系统消息开始，然后是交替的用户和助手消息：

系统消息有助于设置助手的行为。

用户消息相当于用户在 ChatGPT 网络界面中输入问题或句子。它们可以由应用程序的用户生成，也可以作为指令设置。

助手消息有两个作用：要么存储先前的响应以继续对话，要么可以设置为指令，以提供所需行为的示例。模型没有任何关于过去请求的记忆，因此存储先前的消息对于给对话提供上下文和提供所有相关信息是必要的。

### 对话长度和令牌

如前所述，对话的总长度将与令牌的总数相关。这将对以下内容产生影响：

成本

价格是按令牌计费。

时间

令牌越多，响应所需的时间就越长，最多可能需要几分钟。

模型是否工作。

令牌的总数必须小于模型的最大限制。您可以在“注意事项”中找到令牌限制的示例。

如您所见，需要仔细管理对话的长度。您可以通过管理消息的长度来控制输入令牌的数量，并通过`max_tokens`参数来控制输出令牌的数量，详情请参见下一小节。

###### 提示

OpenAI 提供了一个名为[*tiktoken*](https://oreil.ly/zxRIi)的库，允许开发人员计算文本字符串中的令牌数量。我们强烈建议在调用端点之前使用此库来估算成本。

### 额外的可选参数

OpenAI 提供了其他几个选项来微调您与库的交互方式。我们不会在这里详细介绍所有参数，但我们建议查看表 2-2。

表 2-2. 一些额外的可选参数

| 字段名称 | 类型 | 描述 |
| --- | --- | --- |
| `functions` | 数组 | 可用函数的数组。有关如何使用`functions`的更多详细信息，请参见“从文本完成到函数”。 |
| `function_call` | 字符串或对象 | 控制模型的响应方式： |
| | | `none` 表示模型必须以标准方式回应用户。 |
| | | `{"name":"my_function"}` 表示模型必须使用指定的函数来回答。 |
| | | `auto` 表示模型可以在标准响应和`functions`数组中定义的函数之间进行选择。 |
| `temperature` | 数字（默认值：1；接受的值：0 到 2 之间） | 温度为`0`意味着对模型的调用可能会返回相同的输入完成。尽管响应将非常一致，OpenAI 并不保证确定性输出。数值越高，完成的随机性就越大。LLMs 通过逐个预测一系列令牌来生成答案。根据输入上下文，它们为每个潜在的令牌分配概率。当温度参数设置为`0`时，LLM 将始终选择概率最高的令牌。较高的温度允许更多变化和创造性的输出。 |
| `n` | 整数（默认值：1） | 使用此参数，可以为给定的输入消息生成多个聊天完成。但是，当输入参数的温度为`0`时，您将获得多个响应，但它们将完全相同或非常相似。 |
| `stream` | 布尔值（默认值：false） | 正如其名称所示，此参数将允许答案以流格式呈现。这意味着部分消息将逐渐发送，就像在 ChatGPT 界面中一样。当完成很长时，这可以提供更好的用户体验。 |
| `max_tokens` | 整数 | 此参数表示在聊天完成中生成的最大标记数。此参数是可选的，但我们强烈建议设置它作为一个良好的实践，以控制您的成本。请注意，如果设置得太高，此参数可能会被忽略或不被尊重：输入和生成的标记的总长度受模型的标记限制限制。 |

您可以在[官方文档页面](https://platform.openai.com/docs/api-reference/chat)上找到更多详细信息和其他参数。

## 聊天完成端点的输出结果格式

现在您已经获得了查询基于聊天的模型所需的信息，让我们看看如何使用结果。

以下是“Hello World”示例的完整响应：

```py
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Hello there! How may I assist you today?",
                "role": "assistant",
            },
        }
    ],
    "created": 1681134595,
    "id": "chatcmpl-73mC3tbOlMNHGci3gyy9nAxIP2vsU",
    "model": "gpt-3.5-turbo",
    "object": "chat.completion",
    "usage": {"completion_tokens": 10, "prompt_tokens": 11, "total_tokens": 21},
}
```

生成的输出在表 2-3 中详细说明。

表 2-3。聊天完成基本模型的输出描述

| 字段名称 | 类型 | 描述 |
| --- | --- | --- |
| `choices` | `choice`对象的数组 | 包含模型实际响应的数组。默认情况下，此数组将只有一个元素，可以使用参数`n`更改（参见“附加可选参数”）。此元素包含以下内容： |
| | | `finish_reason` `-` `string`：模型答案完成的原因。在我们的“Hello World”示例中，我们可以看到`finish_reason`是`stop`，这意味着我们收到了模型的完整响应。如果在输出生成过程中出现错误，它将出现在此字段中。 |
| | | `index` `-` `integer`：`choices`数组中`choice`对象的索引。 |
| | | `message` `-` `object`：包含`role`和`content`或`function_call`。`role`将始终是`assistant`，`content`将包括模型生成的文本。通常我们希望获得这个字符串：`response'choices'][0]​['mes⁠sage']['content']`。有关如何使用`function_call`的详细信息，请参见“从文本完成到函数”。 |
| `created` | 时间戳 | 生成时的时间戳格式的日期。在我们的“Hello World”示例中，此时间戳对应于 2023 年 4 月 10 日星期一下午 1:49:55。 |
| `model` | 字符串 | 使用的模型。这与设置为输入的模型相同。 |
| `object` | 字符串 | 对于 GPT-4 和 GPT-3.5 模型，应始终为`chat.completion`，因为我们使用了聊天完成端点。 |
| `usage` | 字符串 | 提供有关此查询中使用的标记数的信息，因此为您提供定价信息。`prompt_tokens`表示输入中使用的标记数，`completion_tokens`是输出中的标记数，正如您可能已经猜到的那样，`total_tokens` = `prompt_tokens` + `completion_tokens`。 |

###### 提示

如果您想要有多个选择并使用高于 1 的`n`参数，您会发现`prompt_tokens`值不会改变，但`completion_tokens`值将大致乘以`n`。

## 从文本完成到函数

OpenAI 引入了其模型输出一个包含调用函数参数的 JSON 对象的可能性。模型本身将无法调用函数，而是将文本输入转换为可以由调用者以编程方式执行的输出格式。

当 OpenAI API 的调用结果需要被您的代码的其余部分处理时，这是特别有用的：您可以使用函数定义将自然语言转换为 API 调用或数据库查询，从文本中提取结构化数据，并创建通过调用外部工具来回答问题的聊天机器人，而不是创建一个复杂的提示以确保模型以特定格式回答，这个格式可以被您的代码解析。

正如你在表 2-2 中所见，该表详细介绍了聊天完成端点的输入选项，函数定义需要作为函数对象数组传递。函数对象在表 2-4 中有详细描述。

表 2-4。函数对象的详细信息

| 字段名称 | 类型 | 描述 |
| --- | --- | --- |
| `name` | 字符串（必需） | 函数的名称。 |
| `description` | 字符串 | 函数的描述。 |
| `parameters` | 对象 | 函数期望的参数。这些参数应该以[JSON Schema](http://json-schema.org)格式描述。 |

举个例子，假设我们有一个包含与公司产品相关信息的数据库。我们可以定义一个执行针对该数据库的搜索的函数：

```py
# Example function
def find_product(sql_query):
    # Execute query here
    results = [
        {"name": "pen", "color": "blue", "price": 1.99},
        {"name": "pen", "color": "red", "price": 1.78},
    ]
    return results
```

接下来，我们定义函数的规格：

```py
# Function definition
functions = [
    {
        "name": "find_product",
        "description": "Get a list of products from a sql query",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "A SQL query",
                }
            },
            "required": ["sql_query"],
        },
    }
]
```

然后我们可以创建一个对话并调用`openai.ChatCompletion`端点：

```py
# Example question
user_question = "I need the top 2 products where the price is less than 2.00"
messages = [{"role": "user", "content": user_question}]
# Call the openai.ChatCompletion endpoint with the function definition
response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", messages=messages, functions=functions
)
response_message = response["choices"][0]["message"]
messages.append(response_message)
```

模型已经创建了一个我们可以使用的查询。如果我们从响应中打印`function_call`对象，我们会得到：

```py
"function_call": {
        "name": "find_product",
        "arguments": '{\n "sql_query": "SELECT * FROM products \
 WHERE price < 2.00 ORDER BY price ASC LIMIT 2"\n}',
    }
```

接下来，我们执行函数并继续与结果进行对话：

```py
# Call the function
function_args = json.loads(
    response_message["function_call"]["arguments"]
)
products = find_product(function_args.get("sql_query"))
# Append the function's response to the messages
messages.append(
    {
        "role": "function",
        "name": function_name,
        "content": json.dumps(products),
    }
)
# Format the function's response into natural language
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
)
```

最后，我们提取最终的响应并获得以下内容：

```py
The top 2 products where the price is less than $2.00 are:
1\. Pen (Blue) - Price: $1.99
2\. Pen (Red) - Price: $1.78
```

这个简单的例子演示了函数如何有助于构建一个解决方案，允许最终用户以自然语言与数据库进行交互。函数定义允许您限制模型的回答方式，并将其响应集成到应用程序中。

# 使用其他文本完成模型

正如提到的，OpenAI 除了 GPT-3 和 GPT-3.5 系列之外，还提供了几个其他模型。这些模型使用的端点与 ChatGPT 和 GPT-4 模型不同。尽管 GPT 3.5 Turbo 模型通常是在价格和性能方面最佳的选择，但了解如何使用完成模型，特别是对于微调等用例，对于只有 GPT-3 完成模型可用的情况非常有帮助。

###### 注意

OpenAI 已经发布了文本完成端点的弃用计划。我们在这里介绍这个端点只是因为完成基础模型是唯一可以进行微调的模型。OpenAI 将在 2024 年 1 月之前为基于聊天的模型提供微调的解决方案。由于目前还不可用，我们没有必要的信息来在这里描述它。

文本完成和聊天完成之间有一个重要的区别：你可能会猜到，两者都生成文本，但聊天完成是为对话优化的。如你在下面的代码片段中所见，与`openai.ChatCompletion`端点的主要区别在于提示格式。基于聊天的模型必须是对话格式；对于完成，它是一个单一的提示：

```py
import openai
# Call the openai Completion endpoint
response = openai.Completion.create(
    model="text-davinci-003", prompt="Hello World!"
)
# Extract the response
print(response["choices"][0]["text"])
```

前面的代码片段将输出类似于以下内容的完成：

```py
"\n\nIt's a pleasure to meet you. I'm new to the world"
```

下一节将详细介绍文本完成端点的输入选项。

## 文本完成端点的输入选项

`openai.Completion.create`的输入选项集与我们之前在聊天端点中看到的非常相似。在本节中，我们将讨论主要的输入参数，并考虑提示的长度对其影响。

### 主要输入参数

我们描述了必需的输入参数和我们认为最有用的一些可选参数在表 2-5 中。

表 2-5。文本完成端点的必需参数和可选参数

|字段名称|类型|描述|
|---|---|---|
|`model`|字符串（必需）|要使用的模型的 ID（与`openai.ChatCompletion`相同）。这是唯一必需的选项。|
| `prompt` | 字符串或数组（默认：`<&#124;endoftext&#124;>`） | 用于生成完成的提示。这是与`openai.ChatCompletion`端点的主要区别。`openai.Completion.create`端点应编码为字符串、字符串数组、标记数组或标记数组的数组。如果未提供提示给模型，它将生成文本，就像从新文档的开头开始一样。

| `max_tokens` | 整数 | 聊天完成中要生成的标记的最大数量。此参数的默认值为`16`，对于某些用例来说可能太低，应根据您的需求进行调整。

| `suffix` | 字符串（默认：null） | 完成后的文本。此参数允许添加后缀文本。它还允许进行插入。

### 提示和标记的长度

与聊天模型一样，定价将取决于您发送的输入和接收的输出。对于输入消息，您必须仔细管理提示参数的长度，以及如果使用后缀，则使用后缀。对于您接收的输出，请使用`max_tokens`。它可以避免令人不快的惊喜。

### 其他可选参数

与`openai.ChatCompletion`一样，还可以使用其他可选参数来进一步调整模型的行为。这些参数与`openai.ChatCompletion`使用的参数相同，因此我们不会再详细介绍它们。请记住，您可以使用`temperature`或`n`参数控制输出，使用`max_tokens`控制成本，并使用`stream`选项，如果您希望获得更好的用户体验，可以进行长完成。

## 文本完成端点的输出结果格式

现在您已经拥有了查询基于文本的模型所需的所有信息，您会发现结果与聊天端点的结果非常相似。以下是我们的“Hello World”示例与`davinci`模型的示例输出：

```py
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "text": "<br />\n\nHi there! It's great to see you.",
        }
    ],
    "created": 1681883111,
    "id": "cmpl-76uutuZiSxOyzaFboxBnaatGINMLT",
    "model": "text-davinci-003",
    "object": "text_completion",
    "usage": {"completion_tokens": 15, "prompt_tokens": 3, "total_tokens": 18},
}
```

###### 注意

这个输出与我们在聊天模型中得到的非常相似。唯一的区别在于`choice`对象：不再有带有`content`和`role`属性的消息，而是有一个简单的`text`属性，其中包含模型生成的完成。

# 考虑事项

在大量使用 API 之前，您应该考虑两个重要的事情：成本和数据隐私。

## 定价和标记限制

OpenAI 将其模型的定价列在其[定价页面](https://openai.com/pricing)上。请注意，OpenAI 不受约束地维护此定价，成本可能会随时间变化。

在撰写本文时，OpenAI 模型的定价如表 2-6 所示。

表 2-6。每个模型的定价和标记限制

| 家庭 | 模型 | 定价 | 最大标记 |
| --- | --- | --- | --- |
| 聊天 | `gpt-4` | 提示：每 1,000 个标记 0.03 美元完成：每 1,000 个标记 0.06 美元 | 8,192 |
| 聊天 | `gpt-4-32k` | 提示：每 1,000 个标记 0.06 美元完成：每 1,000 个标记 0.012 美元 | 32,768 |
| 聊天 | `gpt-3.5-turbo` | 提示：每 1,000 个标记 0.0015 美元完成：每 1,000 个标记 0.002 美元 | 4,096 |
| 聊天 | `gpt-3.5-turbo-16k` | 提示：每 1,000 个标记 0.003 美元完成：每 1,000 个标记 0.004 美元 | 16,384 |
| 文本完成 | `text-davinci-003` | 每 1,000 个标记 0.02 美元 | 4,097 |

从表 2-6 中有几件事情需要注意：

`davinci`模型的成本是 GPT-3.5 Turbo 4,000-context 模型的 10 多倍。由于`gpt-3.5-turbo`也可以用于单轮完成任务，并且由于这两个模型在这种类型的任务中几乎具有相同的准确性，建议使用 GPT-3.5 Turbo（除非您需要特殊功能，例如通过参数后缀进行插入，或者如果`text-davinci-003`在您的特定任务中优于`gpt-3.5-turbo`）。

GPT-3.5 Turbo 比 GPT-4 便宜。对于许多基本任务，GPT-4 和 GPT-3.5 之间的差异是无关紧要的。但是，在复杂的推理情况下，GPT-4 远远优于任何以前的模型。

聊天模型的定价系统与`davinci`模型不同：它们区分输入（提示）和输出（完成）。

GPT-4 允许的上下文长度是 GPT-3.5 Turbo 的两倍，甚至可以达到 32,000 个标记，相当于超过 25,000 个文字。GPT-4 可以实现长篇内容创作、高级对话以及文档搜索和分析等用例……但需要付费。

## 安全和隐私：注意！

在我们撰写本文时，OpenAI 声称发送给模型的输入数据不会被用于重新训练，除非您决定选择加入。然而，您的输入将被保留 30 天，用于监控和使用合规检查。这意味着 OpenAI 员工以及专门的第三方承包商可能会访问您的 API 数据。

###### 警告

永远不要通过 OpenAI 端点发送个人信息或密码等敏感数据。我们建议您查看[OpenAI 的数据使用政策](https://openai.com/policies/api-data-usage-policies)以获取最新信息，因为这可能会有所变化。如果您是国际用户，请注意您的个人信息和输入的数据可能会从您的位置传输到美国的 OpenAI 设施和服务器。这可能会对您的应用程序创建产生一些法律影响。

有关如何构建考虑安全和隐私问题的 LLM 应用程序的更多细节，请参阅第三章。

# 其他 OpenAI API 和功能

您的 OpenAI 账户除了文本完成之外还可以访问其他功能。我们在本节中选择了其中几个功能进行探索，但如果您想深入了解所有 API 的可能性，请查看[OpenAI 的 API 参考页面](https://platform.openai.com/docs/api-reference)。

## 嵌入

由于模型依赖数学函数，它需要数字输入来处理信息。然而，许多元素，如单词和标记，并不是固有的数字。为了克服这一点，*嵌入*将这些概念转换为数字向量。嵌入允许计算机通过数值表示更有效地处理这些概念之间的关系。在某些情况下，访问嵌入可能是有用的，OpenAI 提供了一个可以将文本转换为数字向量的模型。嵌入端点允许开发人员获取输入文本的向量表示。然后，这个向量表示可以作为其他 ML 模型和 NLP 算法的输入使用。

在撰写本文时，OpenAI 建议几乎所有用例使用其最新模型`text-embedding-ada-002`。使用起来非常简单：

```py
result = openai.Embedding.create(
    model="text-embedding-ada-002", input="your text"
)
```

可以通过以下方式访问嵌入：

```py
result['data']['embedding']
```

生成的嵌入是一个向量：一个浮点数组。

###### 提示

有关嵌入的完整文档可在[OpenAI 的参考文档](https://platform.openai.com/docs/api-reference/embeddings)中找到。

嵌入的原则是以某种方式有意义地表示文本字符串，捕捉它们的语义相似性。有了这个想法，你可以有各种用例：

搜索

按与查询字符串相关性对结果进行排序。

推荐

推荐包含与查询字符串相关的文本字符串的文章。

聚类

按相似性对字符串进行分组。

异常检测

找到一个与其他字符串无关的文本字符串。

嵌入式具有这样的特性，即如果两个文本具有相似的含义，它们的向量表示将是相似的。例如，在图 2-8 中，显示了三个句子的二维嵌入。尽管两个句子“猫追逐老鼠绕着房子。”和“房子周围，老鼠被猫追赶。”有不同的句法，但它们传达了相同的一般意义，因此它们应该具有相似的嵌入表示。而句子“宇航员在轨道上修理了太空飞船。”与前两个句子的主题（猫和老鼠）无关，讨论的是完全不同的主题（宇航员和太空飞船），因此它应该具有明显不同的嵌入表示。请注意，在此示例中，为了清晰起见，我们将嵌入显示为具有两个维度，但实际上它们通常是更高维度的，例如 512。

![](img/dev-app-gpt4-cgpt-dagc_0208.png)

###### 图 2-8\. 三个句子的二维嵌入示例

我们在剩下的章节中多次提到嵌入式 API，因为嵌入式是处理自然语言与 AI 模型的重要部分。

## 审查模型

如前所述，使用 OpenAI 模型时，您必须遵守[OpenAI 使用政策](https://openai.com/policies/usage-policies)中描述的规则。为了帮助您遵守这些规则，OpenAI 提供了一个模型，用于检查内容是否符合这些使用政策。如果您构建一个应用程序，用户输入将用作提示，这可能很有用：您可以根据审查端点的结果过滤查询。该模型提供了分类功能，允许您搜索以下类别的内容：

仇恨

针对基于种族、性别、种族、宗教、国籍、性取向、残疾或种姓的群体的仇恨

仇恨/威胁

涉及对特定群体进行暴力或严重伤害的仇恨内容

自残

推广或描述自残行为，包括自杀、自残和饮食障碍

性

旨在描述性行为或推广性服务（除了教育和健康）的内容

涉及未成年人的性

涉及 18 岁以下人员的性内容

暴力

美化暴力或庆祝他人的痛苦或羞辱的内容

暴力/图形

描绘死亡、暴力或严重身体伤害的暴力内容

###### 注意

支持英语以外的语言有限。

审查模型的端点是`openai.Moderation.create`，只有两个参数可用：模型和输入文本。内容审查有两种模型。默认模型是`text-moderation-latest`，会随着时间自动更新，以确保您始终使用最准确的模型。另一个模型是`text-moderation-stable`。OpenAI 会在更新此模型之前通知您。

###### 警告

`text-moderation-stable`的准确性可能略低于`text-moderation-latest`。

以下是如何使用这个审查模型的示例：

```py
import openai
# Call the openai Moderation endpoint, with the text-moderation-latest model
response = openai.Moderation.create(
    model="text-moderation-latest",
    input="I want to kill my neighbor.",
)
```

让我们来看看`response`对象中包含的审查端点的输出结果：

```py
{
    "id": "modr-7AftIJg7L5jqGIsbc7NutObH4j0Ig",
    "model": "text-moderation-004",
    "results": [
        {
            "categories": {
                "hate": false,
                "hate/threatening": false,
                "self-harm": false,
                "sexual": false,
                "sexual/minors": false,
                "violence": true,
                "violence/graphic": false,
            },
            "category_scores": {
                "hate": 0.0400671623647213,
                "hate/threatening": 3.671687863970874e-06,
                "self-harm": 1.3143378509994363e-06,
                "sexual": 5.508050548996835e-07,
                "sexual/minors": 1.1862029225540027e-07,
                "violence": 0.9461417198181152,
                "violence/graphic": 1.463699845771771e-06,
            },
            "flagged": true,
        }
    ],
}
```

审查端点的输出结果提供了表 2-7 中显示的信息片段。

表 2-7\. 审查端点输出的描述

| 字段名称 | 类型 | 描述 |
| --- | --- | --- |
| `model` | 字符串 | 这是用于预测的模型。在我们之前的示例中调用该方法时，我们指定了使用模型`text-moderation-latest`，在输出结果中使用的模型是`text-moderation-004`。如果我们使用`text-moderation-stable`调用该方法，那么将使用`text-moderation-001`。 |
| `flagged` | 布尔值 | 如果模型确定内容违反了 OpenAI 的使用政策，则将其设置为`true`；否则，将其设置为`false`。 |
| `categories` | Dict | 这包括一个带有违规政策类别的二进制标志的字典。对于每个类别，如果模型识别到违规，则值为`true`，否则为`false`。可以通过`print(type(response['results'][0]​['cate⁠gories']))`访问该字典。 |
| `category_scores` | Dict | 该模型提供了一个具有特定类别分数的字典，显示它对输入违反 OpenAI 该类别政策的信心程度。分数范围从 0 到 1，分数越高表示信心越大。这些分数不应被视为概率。可以通过`print(type(response​['re⁠sults'][0]['category_scores']))`访问该字典。 |

###### 警告

OpenAI 将定期改进审查系统。因此，`category_scores`可能会有所变化，并且用于确定类别值的阈值也可能会改变。

## Whisper 和 DALL-E

OpenAI 还提供其他不是 LLM 的 AI 工具，但在某些用例中可以与 GPT 模型轻松结合使用。我们在这里不解释它们，因为它们不是本书的重点。但是不用担心，使用它们的 API 与使用 OpenAI 的 LLM API 非常相似。

Whisper 是语音识别的多功能模型。它经过大型音频数据集的训练，也是一个多任务模型，可以执行多语言语音识别、语音翻译和语言识别。OpenAI 的[Whisper 项目 GitHub 页面](https://github.com/openai/whisper)上提供了开源版本。

2021 年 1 月，OpenAI 推出了 DALL-E，这是一种能够根据自然语言描述创建逼真图像和艺术品的 AI 系统。DALL-E 2 通过更高的分辨率、更大的输入文本理解能力和新的功能进一步推动了这项技术。DALL-E 的两个版本都是通过对图像及其文本描述进行训练来创建的变压器模型。您可以通过 API 和[Labs 界面](https://labs.openai.com)尝试 DALL-E 2。

# 摘要（和备忘单）

正如我们所见，OpenAI 通过 API 提供其模型作为服务。在本书中，我们选择使用 OpenAI 提供的 Python 库，它是 API 的简单封装。使用这个库，我们可以与 GPT-4 和 ChatGPT 模型进行交互：这是构建 LLM 应用程序的第一步！然而，使用这些模型意味着需要考虑几个方面：API 密钥管理、定价和隐私。

在开始之前，我们建议查看 OpenAI 的使用政策，并在 Playground 上玩耍，以熟悉不同模型而不必编写代码。记住：ChatGPT 背后的 GPT-3.5 Turbo 是大多数用例的最佳选择。

以下是发送输入到 GPT-3.5 Turbo 时使用的备忘单：

1.  安装`openai`依赖项：

    ```py
    pip install openai
    ```

1.  将 API 密钥设置为环境变量：

    ```py
    export OPENAI_API_KEY=sk-(...)
    ```

1.  在 Python 中，导入`openai`：

    ```py
    import openai
    ```

1.  调用`openai.ChatCompletion`端点：

    ```py
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Your Input Here"}],
    )
    ```

1.  获取答案：

    ```py
        print(response['choices'][0]['message']['content'])
    ```

###### 提示

不要忘记查看[定价页面](https://openai.com/pricing)，并使用[tiktoken](https://github.com/openai/tiktoken)来估算使用成本。

请注意，您不应该通过 OpenAI 端点发送个人信息或密码等敏感数据。

OpenAI 还提供了其他几个模型和工具。在接下来的章节中，您将发现嵌入端点对于在应用程序中包含 NLP 功能非常有用。

现在您知道如何使用 OpenAI 服务，是时候深入了解为什么要使用它们了。在下一章中，您将看到各种示例和用例的概述，以帮助您充分利用 OpenAI ChatGPT 和 GPT-4 模型。


