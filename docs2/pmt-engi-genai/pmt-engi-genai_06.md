# 第六章：具有记忆和工具的自主代理

本章深入探讨了思维链推理的重要性以及大型语言模型（LLM）作为代理通过复杂问题的推理能力。通过将复杂问题分解成更小、更易于管理的组件，LLM 可以提供更全面和有效的解决方案。你还将了解构成自主代理的组件，如输入、目标或奖励函数以及可用的操作。

# 思维链

AI 通过复杂问题的推理能力对于创建有效、可靠和用户友好的应用至关重要。

*思维链推理*（CoT）是一种引导 LLM 通过一系列步骤或逻辑连接来得出结论或解决问题的方法。这种方法对于需要更深入理解上下文或考虑多个因素的作业特别有用。

[CoT](https://oreil.ly/fAeLo)要求 LLM 通过复杂问题进行 *思考*，将它们分解成更小、更易于管理的组件。这允许 LLM 单独关注每个部分，确保对当前问题的更深入理解。

在实践中，思维链推理可能涉及：

+   要求 LLM 对其决策提供解释

+   在做出最终决定之前规划多个步骤

在接下来的章节中，你将探索无效和有效的思维链推理的例子。我们还将讨论构建有效思维链推理的各种技术以及它们如何集成到 AI 应用中。

让我们想象一个用户希望 AI 生成一个全面的营销计划来推广一款新的软件产品。

输入：

```py
Create a marketing plan for a new software product.
```

输出：

```py
1\. Launch social media campaign
2\. Send out email newsletters
3\. Offer limited-time discounts
4\. Collaborate with influencers
5\. Organize webinars
```

在这个例子中，GPT-4 没有使用思维链推理，也没有针对营销计划的特定方面进行说明。LLM 生成了一般性的营销策略列表，这些策略可以适用于任何产品，而不是专注于新软件产品的独特特性。

输入：

```py
Create a detailed marketing plan for a new project management software
product targeting small and medium-sized businesses.

The budget is $20,000, and we want to focus on digital marketing channels.

Consider the success of similar products in the market and provide a
step-by-step plan that includes the most effective promotional tactics.
```

输出：

```py
Market Research
a. Analyze competitors and their marketing strategies
b. Identify the unique selling points of the new project management software
c. Define the target audience and their pain points

Branding and Messaging
a. Develop a consistent brand identity
b. Craft compelling messaging that addresses the target audience's pain
points
c. Create marketing assets such as website, blog, and social media content

...

Email Marketing
a. Develop an email marketing strategy to nurture leads and convert them
into customers
b. Send personalized onboarding emails to new users with helpful tips and
resources
```

现在，GPT-4 专门针对新软件产品的独特特性，展示了有效的思维链推理。

# 给予指导

请注意短语 *step-by-step*，这是 CoT 中的一个关键要素。通过将这个短语纳入你的提示中，你是在要求 LLM 通过生成高度有效的软件产品所需的步骤进行推理。

此外，通过提供 20,000 美元的预算和软件类型，GPT-4 能够提供更加相关和具体化的响应。

# 代理

生成式 AI 模型催生了 *基于代理的架构*。从概念上讲，代理在指定环境中行动、感知并做出决策，以实现预定义的目标。

代理可以执行各种操作，例如执行 Python 函数；之后，代理将观察发生了什么，并决定是否完成或采取下一步行动。

代理将持续循环一系列动作和观察，直到没有更多的动作，如下面的伪代码所示：

```py
next_action = agent.get_action(...)
while next_action != AgentFinish:
    observation = run(next_action)
    next_action = agent.get_action(..., next_action, observation)
return next_action
```

代理的行为由三个主要组成部分控制：

输入

这些是代理从其环境中接收到的感官刺激或数据点。输入可以多种多样，从视觉（如图像）和听觉（如音频文件）到热信号等。

目标或奖励函数

这代表了代理动作的指导原则。在基于目标的框架中，代理被要求达到一个特定的最终状态。在基于奖励的设置中，代理被驱动去最大化随时间累积的奖励，通常是在动态环境中。

可用动作

*动作空间*是代理在任何给定时刻可以采取的允许动作的范围。[可以执行的动作](https://oreil.ly/5AVfM)。这个空间的范围和性质取决于手头的任务。

为了进一步解释这些概念，可以考虑自动驾驶汽车：

输入

汽车的传感器，如摄像头、激光雷达和超声波传感器，提供有关环境的连续数据流。这可能包括关于附近车辆、行人、道路状况和交通信号的信息。

目标或奖励函数

对于自动驾驶汽车来说，主要目标是安全且高效地从 A 点到 B 点的导航。如果我们使用基于奖励的系统，汽车可能会因为保持与其他物体的安全距离、遵守速度限制和遵守交通规则而获得正奖励。相反，它可能会因为危险行为，如紧急制动或偏离车道，而收到负奖励。特斯拉特别使用无需干预行驶的英里数作为它们的奖励函数。

可用动作

汽车的动作空间包括加速、减速、转向、换道等。每个动作都是基于当前输入数据和由目标或奖励函数定义的目标来选择的。

你会发现，在自动驾驶汽车等系统中，代理依赖于诸如输入、目标/奖励函数和可用动作等基础原则。然而，当深入研究 GPT 等 LLM 领域时，存在一套专门针对其独特性质的动态。

这是它们如何满足你的需求的：

输入

对于大型语言模型（LLMs），入口主要是通过文本。但这并不限制你可以使用的信息量。无论你处理的是温度读数、音乐符号，还是复杂的数据结构，你的挑战在于将这些内容塑造成适合 LLM 的文本表示。想想视频：虽然原始素材可能看起来不兼容，但视频文本转录允许 LLM 为你提取见解。

利用目标驱动的指令

LLMs 主要使用您在文本提示中定义的目标。通过创建具有明确目标的提示，您不仅能够访问 LLM 的丰富知识，而且实际上在规划其推理路径。将其视为绘制蓝图：您的具体提示指导模型，引导它将总体目标分解成一系列系统的步骤。

通过功能性工具制定行动

LLMs 不仅限于简单的文本生成；您还可以实现更多。通过集成*现成的工具*或*自定义开发的工具*，您可以装备 LLMs 执行各种任务，从 API 调用到数据库交互，甚至编排外部系统。工具可以用任何编程语言编写，通过添加更多工具，您实际上在*扩展 LLM 能够实现的行为空间*。

也有一些不同的组件可以直接应用于 LLMs：

记忆

在代理步骤之间存储状态是理想的；这在聊天机器人中尤其有用，记住之前的聊天历史可以提供更好的用户体验。

代理规划/执行策略

实现高级目标有多种方式，其中规划和执行的结合是至关重要的。

检索

LLMs 可以使用不同类型的检索方法。在向量数据库中的语义相似性是最常见的，但还有其他方法，例如将来自 SQL 数据库的定制信息包含到提示中。

让我们更深入地探讨共享和不同的组件，并探索实现细节。

## 理解和行动（ReAct）

有许多代理框架旨在最终提高 LLM 向目标响应的能力。原始框架是*ReAct*，它是 CoT 的改进版本，允许 LLM 在通过工具采取行动后创建观察。然后，这些观察被转化为关于在下一步中应该使用什么*正确工具*的*思维*（图 6-1）。LLM 继续推理，直到出现`'Final Answer'`字符串值或达到最大迭代次数。

![ReAct 框架](img/pega_0601.png)

###### 图 6-1. ReAct 框架

[ReAct](https://oreil.ly/ssdnL)框架结合了任务分解、思维循环和多种工具来解决疑问。让我们探索 ReAct 中的思维循环：

1.  观察环境。

1.  用思维解释环境。

1.  决定一个行动。

1.  在环境中采取行动。

1.  重复步骤 1-4，直到找到解决方案或迭代次数过多（解决方案是“我已经找到答案”）。

您可以通过使用前面的思维循环，同时向 LLM 提供多个输入，如以下内容，轻松创建一个 ReAct 风格的提示：

+   `{问题}`：您希望得到回答的查询。

+   `{tools}`：这些指的是可以用于完成整体任务中某个步骤的函数。通常的做法是包括一个工具列表，其中每个工具都是一个 Python 函数、一个名称以及该函数及其目的的描述。

以下是一个实现 ReAct 模式并使用大括号`{}`字符包裹的提示变量`{question}`的提示：

```py
You will attempt to solve the problem of finding the answer to a question.
Use chain-of-thought reasoning to solve through the problem, using the
following pattern:

1\. Observe the original question:
original_question: original_problem_text
2\. Create an observation with the following pattern:
observation: observation_text
3\. Create a thought based on the observation with the following pattern:
thought: thought_text
4\. Use tools to act on the thought with the following pattern:
action: tool_name
action_input: tool_input

Do not guess or assume the tool results. Instead, provide a structured
output that includes the action and action_input.

You have access to the following tools: {tools}.

original_problem: {question}

Based on the provided tool result:

Either provide the next observation, action, action_input, or the final
answer if available.

If you are providing the final answer, you must return the following pattern:
"I've found the answer: final_answer"
```

下面是提示的分解：

1.  提示的引入清楚地确立了 LLM 的目的：`你将尝试解决找到问题答案的问题。`

1.  然后概述问题解决方法：`使用链式思维推理通过以下模式解决问题：`

1.  链式思维推理的步骤随后被列出：

    +   LLM 首先观察原始问题，然后对其形成观察：`original_question: original_problem_text`，`observation: observation_text`。

    +   基于这个观察，AI 应该制定一个表示推理过程中一步的思想：`thought: thought_text`。

    +   在确立了一个思想之后，它然后决定使用可用的工具之一采取行动：`action: tool_name`，`action_input: tool_input`。

1.  然后提醒 LLM 不要对工具可能返回的内容做出假设，它应该明确说明其预期的行动和相应的输入。

1.  `你有以下工具可用：{tools}` 通知 LLM 它可用于解决问题的工具。

1.  LLM 必须解决的实际上是以下问题：`original_problem: {question}`。

1.  最后，根据其行动的结果，提供了 LLM 应该如何响应的说明。它可以选择继续进行新的观察、行动和输入，或者如果找到了解决方案，提供最终答案。

提示概述了一个系统化的问题解决过程，其中 LLM 观察一个问题，思考它，决定采取行动，并重复此过程，直到找到解决方案。

## 理由和行动实施

既然你已经了解了 ReAct，那么创建一个简单的 Python 实现来复制 LangChain 自动执行的操作就很重要了，这样你可以建立起对 LLM 响应之间真正发生的事情的直觉。

为了保持简单，这个例子将不会实现循环，并假设输出可以从单个工具调用中获得。

为了创建基本的 ReAct 实现，你需要实现以下内容：

1.  在每一个思维过程中，你需要提取 LLM 想要使用的工具。因此，你会提取最后的`action`和`action_input`。`action`代表工具名称，而`action_input`由函数参数的值组成。

1.  检查 LLM 是否认为它已经找到了最终答案，如果是这样，那么思维循环就结束了。

你可以使用正则表达式从 LLM 响应中提取`action`和`action_input`值：

```py
import re

# Sample text:
text = """
Action: search_on_google
Action_Input: Tom Hanks's current wife

action: search_on_wikipedia
action_input: How old is Rita Wilson in 2023

action : search_on_google
action input: some other query
"""

# Compile regex patterns:
action_pattern = re.compile(r"(?i)action\s*:\s*([^\n]+)", re.MULTILINE)
action_input_pattern = re.compile(r"(?i)action\s*_*input\s*:\s*([^\n]+)",
re.MULTILINE)

# Find all occurrences of action and action_input:
actions = action_pattern.findall(text)
action_inputs = action_input_pattern.findall(text)

# Extract the last occurrence of action and action_input:
last_action = actions[-1] if actions else None
last_action_input = action_inputs[-1] if action_inputs else None

print("Last Action:", last_action)
print("Last Action Input:", last_action_input)
# Last Action: search_on_google
# Last Action Input: some other query
```

让我们分解这个正则表达式以提取 `action`：

+   `action_pattern = re.compile(r"(?i)action\s*:\s*([^\n]+)", re.MULTILINE)`

+   `(?i)`: 这被称为 *内联标志*，使正则表达式模式不区分大小写。这意味着模式将匹配“action”，“Action”，“ACTION”或任何其他大小写字母的组合。

+   `action`: 模式的一部分匹配单词 *action* 字面意义。由于不区分大小写的标志，它将匹配任何大小写形式的单词。

+   `\s*`: 模式的一部分匹配零个或多个空白字符（空格、制表符等）。`\*` 表示“零个或多个”，`\s` 是正则表达式中空白字符的缩写。

+   `:` 模式的一部分匹配冒号字符字面意义。

+   `\s*`: 这与之前的 `\s\*` 部分相同，匹配冒号后面的零个或多个空白字符。

+   `+([^\n]++)`: 这个模式是一个捕获组，由括号表示。它匹配一个或多个不是换行符的字符。方括号 `[]` 内部的 `^` 取反了字符类，`\n` 代表换行符。`+` 表示“一个或多个”。这个组匹配的文本将在使用 `findall()` 函数时被提取。

+   `re.MULTILINE`: 这个标志是传递给 `re.compile()` 函数的。它告诉正则表达式引擎输入文本可能有多个行，因此模式应该逐行应用。

+   在正则表达式中，方括号 `[]` 用于定义字符类，这是一组你想要匹配的字符。例如，`[abc]` 将匹配任何单个字符，该字符是 `'a'`、`'b'` 或 `'c'` 之一。

+   当你在字符类开头添加一个上标 `^` 时，它将字符类取反，意味着它将匹配任何不在字符类中的字符。换句话说，它反转了你想要匹配的字符集。

+   因此，当我们使用 `[^abc]` 时，它将匹配任何不是 `'a'`、`'b'` 或 `'c'` 的单个字符。在正则表达式模式 `+([^\n]++)` 中，字符类是 `[^n]`，这意味着它将匹配任何不是换行符的字符。取反字符类后面的 `+` 表示模式应该匹配一个或多个不是换行的字符。

+   通过在捕获组中使用取反的字符类 `[^n]`，我们确保正则表达式引擎捕获的文本直到行尾，不包括换行符本身。这在我们要提取单词 *action* 或 *action input* 后面直到行尾的文本时很有用。

总体来说，这个正则表达式模式匹配单词 *action*（不区分大小写）后面跟着可选的空白字符，一个冒号，再跟着可选的空白字符，然后捕获任何直到行尾的文本。

这两个正则表达式模式之间的唯一区别是它们在开头寻找的文本：

1.  `action_pattern` 寻找单词 `"action"`。

1.  `action_input_pattern` 寻找单词 `"action_input"`。

您现在可以将正则表达式抽象成一个 Python 函数，该函数将始终找到最后一个 `action` 和 `action_input`：

```py
def extract_last_action_and_input(text):
    # Compile regex patterns
    action_pattern = re.compile(r"(?i)action\s*:\s*([^\n]+)", re.MULTILINE)
    action_input_pattern = re.compile(
        r"(?i)action\s*_*input\s*:\s*([^\n]+)", re.MULTILINE
    )

    # Find all occurrences of action and action_input
    actions = action_pattern.findall(text)
    action_inputs = action_input_pattern.findall(text)

    # Extract the last occurrence of action and action_input
    last_action = actions[-1] if actions else None
    last_action_input = action_inputs[-1] if action_inputs else None

    return {"action": last_action, "action_input": last_action_input}

extract_last_action_and_input(text)
# {'action': 'search_on_google', 'action_input': 'some other query'}
```

要确定和提取 LLM 是否发现了最终答案，您也可以使用正则表达式：

```py
def extract_final_answer(text):
    final_answer_pattern = re.compile(
        r"(?i)I've found the answer:\s*([^\n]+)", re.MULTILINE
    )
    final_answers = final_answer_pattern.findall(text)
    if final_answers:
        return final_answers[0]
    else:
        return None

final_answer_text = "I've found the answer: final_answer"
print(extract_final_answer(final_answer_text))
# final_answer
```

###### 警告

LLM 不总是以预期的方式响应，因此您的应用程序需要能够处理正则表达式解析错误。几种方法包括使用 LLM 修复上一个 LLM 的响应或使用上一个状态发出另一个新的 LLM 请求。

您现在可以组合所有组件；以下是一个逐步说明：

```py
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
```

初始化 `ChatOpenAI` 实例：

```py
chat = ChatOpenAI(model_kwargs={"stop": ["tool_result:"],})
```

添加 `stop` 序列会强制 LLM 在遇到短语 `"tool_result:"` 后停止生成新标记。这有助于通过停止工具使用时的幻觉。

定义可用的工具：

```py
tools = {}

def search_on_google(query: str):
    return f"Jason Derulo doesn't have a wife or partner."

tools["search_on_google"] = {
    "function": search_on_google,
    "description": "Searches on google for a query",
}
```

设置基本提示模板：

```py
base_prompt = """
You will attempt to solve the problem of finding the answer to a question.
Use chain-of-thought reasoning to solve through the problem, using the
following pattern:

1\. Observe the original question:
original_question: original_problem_text
2\. Create an observation with the following pattern:
observation: observation_text
3\. Create a thought based on the observation with the following pattern:
thought: thought_text
4\. Use tools to act on the thought with the following pattern:
action: tool_name
action_input: tool_input

Do not guess or assume the tool results. Instead, provide a structured
output that includes the action and action_input.

You have access to the following tools: {tools}.

original_problem: {question}
"""
```

生成模型输出：

```py
output = chat.invoke(SystemMessagePromptTemplate \
.from_template(template=base_prompt) \
.format_messages(tools=tools, question="Is Jason Derulo with a partner?"))
print(output)
```

提取最后一个 `action`、`action_input` 并调用相关函数：

```py
tool_name = extract_last_action_and_input(output.content)["action"]
tool_input = extract_last_action_and_input(output.content)["action_input"]
tool_result = tools[tool_name]"function"
```

打印工具详细信息：

```py
print(f"""The agent has opted to use the following tool:
tool_name: {tool_name}
tool_input: {tool_input}
tool_result: {tool_result}"""
)
```

设置包含工具结果的当前提示：

```py
current_prompt = """
You are answering this query: Is Jason Derulo with a partner?

Based on the provided tool result:
tool_result: {tool_result}

Either provide the next observation, action, action_input, or the final
answer if available. If you are providing the final answer, you must return
the following pattern: "I've found the answer: final_answer"
"""
```

为当前提示生成模型输出：

```py
output = chat.invoke(SystemMessagePromptTemplate. \
from_template(template=current_prompt) \
.format_messages(tool_result=tool_result))
```

打印当前提示的模型输出：

```py
print("----------\n\nThe model output is:", output.content)
final_answer = extract_final_answer(output.content)
if final_answer:
    print(f"answer: {final_answer}")
else:
    print("No final answer found.")
```

输出：

```py
'''content='1\. Observe the original question:\nIs Jason Derulo with a
partner?\n\n2\. Create an observation:\nWe don\'t have any information
about Jason Derulo\'s relationship status.\n\n3\. Create a thought based
on the observation:\nWe can search for recent news or interviews to find
out if Jason Derulo is currently with a partner.\n\n4\. Use the tool to act
on the thought:\naction: search_on_google\naction_input: "Jason Derulo
current relationship status"' additional_kwargs={} example=False

----------
The agent has opted to use the following tool:
tool_name: search_on_google
tool_input: "Jason Derulo current relationship status"
tool_result: Jason Derulo doesn't have a wife or partner.
----------

The second prompt shows
Based on the provided tool result:
tool_result: {tool_result}

Either provide the next observation, action, action_input, or the final
answer if available. If you are providing the final answer, you must
return the following pattern: "I've found the answer: final_answer"
----------

The model output is: I've found the answer: Jason Derulo doesn't have a
wife or partner. answer: Jason Derulo doesn't have a wife or partner.'''
```

前面的步骤提供了一个非常简单的 ReAct 实现。在这种情况下，LLM 决定使用 `search_on_google` 工具，其 `action_input` 为 `"Jason Derulo current relationship status"`。

###### 注意

LangChain 代理将自动以简洁的方式执行所有前面的步骤，包括提供多次工具使用（通过循环）以及处理工具失败的情况，当代理无法解析 `action` 或 `action_input` 时。

在探索 LangChain 代理及其提供的功能之前，学习 *工具* 以及如何创建和使用它们至关重要。

## 使用工具：

由于大型语言模型如 GPT-4 只能生成文本，提供可以执行其他操作的工具（如与数据库交互或读取/写入文件）可以有效地提高 LLM 的能力。*工具* 简单来说是一个预定义的函数，允许代理执行特定的操作。

代理提示的一个常见部分可能包括以下内容：

```py
You are looking to accomplish: {goal}
You have access to the following {tools}
```

大多数工具都是用编程语言编写的函数。在探索 LangChain 的过程中，您会发现它提供了三种不同的工具创建/使用方法：

+   创建您自己的自定义工具。

+   使用现有工具。

+   利用 `AgentToolkits`，这些工具捆绑在一起以完成特定任务。

让我们从创建一个使用 LangChain 检查给定字符串长度的自定义工具开始：

```py
# Import necessary classes and functions:
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# Defining the LLM to use:
model = ChatOpenAI()

# Function to count the number of characters in a string:
def count_characters_in_string(string):
    return len(string)

# Create a list of tools:
# Currently, only one tool is defined that counts characters in a text string.
tools = [
    Tool.from_function(
        func=count_characters_in_string,
        name="Count Characters in a text string",
        description="Count the number of characters in a text string",
    )
]

# Download a react prompt!
prompt = hub.pull("hwchase17/react")

# Construct the ReAct agent:
agent = create_react_agent(model, tools, prompt)

# Initialize an agent with the defined tools and
# Create an agent executor by passing in the agent and tools:
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Invoke the agent with a query to count the characters in the given word:
agent_executor.invoke({"input": '''How many characters are in the word
"supercalifragilisticexpialidocious"?'''})

# 'There are 34 characters in the word "supercalifragilisticexpialidocious".'
```

在导入必要的模块后，您初始化一个 `ChatOpenAI` 聊天模型。然后创建一个名为 `count_characters_in_string` 的函数，该函数计算任何给定字符串的长度。此函数封装在一个 `Tool` 对象中，提供描述性名称和说明其作用。

随后，你使用 `create_react_agent` 初始化你的代理，结合定义的 `Tool`、`ChatOpenAI` 模型和从 LangChain 中心拉取的 react 提示。这设置了一个全面的交互式代理。

使用 `AgentExecutor`，代理配备了工具，并启用了详细输出，允许进行详细记录。

最后，`agent_executor.invoke(...)` 执行了一个关于“supercalifragilisticexpialidocious”中字符数的查询。代理利用定义的工具来计算并返回该单词中精确的字符数。

在 示例 6-1 中，你可以看到代理决定使用名为 `Action` 的 `Characters` `in a text string`，并带有 `Action Input`：`'supercalifragilisticexpialidocious'`。这种模式与您之前制作的简单 ReAct 实现极为相似。

##### 示例 6-1\. 单个工具，代理输出

```py
Entering new AgentExecutor change...
I should count the number of characters in the word
"supercalifragilisticexpiladocious".
Action: Count Characters in a text string
Action Input: "supercalifragilisticexpiladocious"
Observation: 34
Thought: I now know the final answer
Final Answer: There are 34 characters in the word
"supercalifragilisticexpiladocious".
```

# 给出指示

为你的 Python 函数和工具描述编写表达性的名称将提高 LLM 有效选择正确工具的能力。

# 使用 LLM 作为 API（OpenAI 函数）

如 第四章 中所述，OpenAI [发布了更多针对功能调用的微调 LLM](https://oreil.ly/hYTus)。这很重要，因为它为工具使用提供了标准 ReAct 模式的替代方案。它与 ReAct 类似，因为你仍然在利用 LLM 作为 *推理引擎*。

如 图 6-2 所示，函数调用允许 LLM 容易地将用户的输入转换为天气 API 调用。

![使用 OpenAI 函数的功能调用流程](img/pega_0602.png)

###### 图 6-2\. 使用 OpenAI 函数的功能调用流程

LangChain 允许用户轻松地在不同的代理类型之间切换，包括 ReAct、OpenAI 函数等。

请参阅 表 6-1 以全面比较不同的代理类型。

表 6-1\. 代理类型比较

| 代理类型 | 描述 |
| --- | --- |
| OpenAI 函数 | 与 gpt-3.5-turbo-0613 和 gpt-4-0613 等微调模型一起工作，用于函数调用。它智能地输出 JSON 对象以进行函数调用。最适合采用此格式的开源模型和提供商。注意：已被 OpenAI 工具取代。 |
| OpenAI 工具 | 新模型的增强版本，能够调用一个或多个函数。它智能地输出 JSON 对象以进行这些函数调用，优化响应效率并减少某些架构中的响应时间。 |
| XML 代理 | 适用于像 Anthropic 的 Claude 这样的语言模型，这些模型在 XML 推理/写作方面表现出色。最佳搭配常规 LLM（非聊天模型）和接受单个字符串输入的非结构化工具。 |
| JSON 聊天代理 | 定制用于擅长 JSON 格式的语言模型。此代理使用 JSON 格式其输出，支持需要 JSON 输出的聊天模型。 |
| 结构化聊天 | 能够使用多输入工具，此代理旨在处理需要结构化输入和响应的复杂任务。 |
| ReAct | 实现 ReAct 逻辑，使用 Tavily 的搜索与文档存储或搜索工具进行交互。 |
| 自问自答与搜索 | 利用中间答案工具进行事实性问题解决，遵循自问自答与搜索方法。适用于需要快速准确事实答案的场景。 |

让我们使用预包装的工具，例如`计算器`，通过从 LangChain 文档中调用 OpenAI 功能来回答数学问题：

```py
# Import necessary modules and functions from the langchain package:
from langchain.chains import (
    LLMMathChain,
)
from langchain import hub
from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor
from langchain_openai.chat_models import ChatOpenAI

# Initialize the ChatOpenAI with temperature set to 0:
model = ChatOpenAI(temperature=0)

# Create a LLMMathChain instance using the ChatOpenAI model:
llm_math_chain = LLMMathChain.from_llm(llm=model, verbose=True)

# Download the prompt from the hub:
prompt = hub.pull("hwchase17/openai-functions-agent")

tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run, # run the LLMMathChain
        description="useful for when you need to answer questions about math",
        return_direct=True,
    ),
]

# Create an agent using the ChatOpenAI model and the tools:
agent = create_openai_functions_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What is 5 + 5?"})
print(result)
# {'input': 'What is 5 + 5?', 'output': 'Answer: 10'}
```

在初始化必要的库之后，你将使用`ChatOpenAI`，将`temperature`参数设置为 0 以获得确定性输出。通过使用`hub.pull("...")`，你可以轻松下载保存在 LangChainHub 上的提示。

此模型随后与名为`计算器`的工具相结合，利用`LLMMathChain`的能力来计算数学查询。OpenAI 函数代理随后决定使用`计算器`工具来计算`5 + 5`并返回`答案：10`。

接下来，你可以给一个代理配备多个工具，增强其多功能性。为了测试这一点，让我们给我们的代理添加一个额外的`工具`对象，使其能够执行模拟的 Google 搜索：

```py
def google_search(query: str) -> str:
    return "James Phoenix is 31 years old."

# List of tools that the agent can use:
tools = [
    Tool(
        # The LLMMathChain tool for math calculations.
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
    ),
    Tool(
        # Tool for counting characters in a string.
        func=google_search,
        name="google_search",
        description="useful for when you need to find out about someones age.",
    ),
]

# Create an agent using the ChatOpenAI model and the tools:
agent = create_openai_functions_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Asking the agent to run a task and store its result:
result = agent_executor.invoke(
    {
        "input": """Task: Google search for James Phoenix's age.
 Then square it."""}
)
print(result)
# {'input': "...", 'output': 'James Phoenix is 31 years old.
# Squaring his age, we get 961.'}
```

当执行时，代理将首先调用`google_search`函数，然后继续执行`llm_math_chain.run`函数。通过混合自定义和预包装的工具，你可以显著提高代理的灵活性。

###### 注意

根据你提供的工具数量，LLM 将限制或增加解决不同用户查询的能力。此外，如果你添加了太多工具，LLM 在解决问题的每一步可能会对使用哪些工具感到困惑。

这里有一些推荐你想要探索的工具：

[Google 搜索](https://oreil.ly/TjrnF)

使 LLM 能够执行网络搜索，这提供了及时和相关的上下文。

[文件系统工具](https://oreil.ly/5tAB0)

对于管理文件至关重要，无论是涉及读取、写入还是重新组织文件。有了这些工具，你的 LLM 可以更有效地与文件系统交互。

[Requests](https://oreil.ly/vZjm1)

一个实用的工具，使 LLM 能够执行创建、读取、更新和删除（CRUD）功能的 HTTP 请求。

[Twilio](https://oreil.ly/ECS4r)

通过允许 LLM 通过 Twilio 发送短信或 WhatsApp 消息，增强 LLM 的功能。

# 分工合作并评估质量

使用工具时，确保适当分配任务。例如，将通信服务委托给 Twilio，同时分配与 HTTP 相关的请求。此外，持续评估每个工具执行任务的表现和质量至关重要。

不同的工具可能被调用得更频繁或更少，这将影响你的 LLM 代理的性能。监控工具使用情况将为你提供关于代理整体性能的见解。

# 比较 OpenAI 函数和 ReAct

OpenAI 函数和 ReAct 框架为使用生成式 AI 模型执行任务提供了独特的功能。了解它们之间的差异可以帮助你确定哪个更适合你的特定用例。

OpenAI 函数以简单直接的方式运行。在这个设置中，LLM 在运行时决定是否执行函数。当集成到对话代理中时，这很有益，因为它提供了包括以下在内的几个特性：

运行时决策

LLM 在实时自主决定是否执行某个（些）函数。

单个工具执行

OpenAI 函数非常适合需要单个工具执行的任务。

实现简便

OpenAI 函数可以轻松地与对话代理合并。

并行函数调用

对于需要多个解析的单个任务执行，OpenAI 函数提供并行函数调用，在同一个 API 请求中调用多个函数。

## OpenAI 函数的用例

如果你的任务涉及明确的行动，如简单的搜索或数据提取，OpenAI 函数是一个理想的选择。

## ReAct

如果你需要涉及多个顺序工具使用和更深入反思先前行动的执行，ReAct 就派上用场。与函数调用相比，ReAct 被设计成通过许多*思维循环*来实现更高层次的目标，这使得它适合具有多个意图的查询。

尽管 ReAct 作为代理与`conversational-react`兼容，但它尚未提供与函数调用相同级别的稳定性，并且通常更倾向于使用工具而不是简单地以文本形式响应。尽管如此，如果你的任务需要连续执行，ReAct 生成多个思维循环并在每次决定使用单个工具的能力展示了几个显著特性，包括：

迭代思维过程

ReAct 允许代理为复杂任务生成多个思维循环。

多意图处理

ReAct 有效地处理具有多个意图的查询，因此它适合复杂任务。

多个工具执行

适用于需要顺序执行多个工具的任务。

## ReAct 的用例

如果你正在处理一个需要反思先前行动或连续使用多个函数的项目，例如保存一次采访然后通过电子邮件发送，ReAct 是最佳选择。

为了帮助决策，请参阅表 6-2 中的全面比较。

表 6-2\. OpenAI 函数与 ReAct 的特性比较

| 特性 | OpenAI 函数 | ReAct |
| --- | --- | --- |
| 运行时决策 | ✓ | ✓ |
| 单个工具执行 | ✓ | ✓ |
| 实现简便 | ✓ | x |
| 并行函数调用 | ✓ | x |
| 迭代思维过程 | x | ✓ |
| 多意图处理 | ✓ | ✓ |
| 顺序工具执行 | x | ✓ |
| 可定制的提示 | ✓ | ✓ |

# 给予指导

当与不同的 AI 框架交互时，理解每个框架都有其优势和权衡是至关重要的。每个框架将为您的 LLM 提供独特的形式指引。

# 代理工具包

*[代理工具包](https://oreil.ly/_v6dm)*是 LangChain 的集成，提供了多个工具，并将它们串联起来，使您能够快速自动化任务。

首先，通过在您的终端中输入`**pip install langchain_experimental pandas tabulate langchain-community pymongo --upgrade**`来安装更多包。流行的代理工具包包括：

+   CSV Agent

+   Gmail Toolkit

+   OpenAI Agent

+   Python Agent

+   JSON Agent

+   Pandas DataFrame Agent

CSV 代理使用 Pandas DataFrame 代理和`python_repl_ast`工具来调查*.csv*文件。您可以要求量化数据、识别列名或创建相关矩阵。

在[共享仓库](https://oreil.ly/x6FHn)的*content/chapter_6*中创建一个新的 Jupyter Notebook 或 Python 文件，然后您需要导入`create_csv_agent`、`ChatOpenAI`和`AgentType`。`create_csv_agent`函数需要一个 LLM、数据集`文件路径`和`agent_type`：

```py
# Importing the relevant packages:
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai.chat_models import ChatOpenAI

# Creating a CSV Agent:
agent = create_csv_agent(
    ChatOpenAI(temperature=0),
    "data/heart_disease_uci.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent.invoke("How many rows of data are in the file?")
# '920'

agent.invoke("What are the columns within the dataset?")
# "'id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs',
# 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'"

agent.invoke("Create a correlation matrix for the data and save it to a file.")
# "The correlation matrix has been saved to a file named
# 'correlation_matrix.csv'."
```

您甚至可以通过 SQLDatabase 代理与 SQL 数据库进行交互：

```py
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_openai.chat_models import ChatOpenAI

db = SQLDatabase.from_uri("sqlite:///./data/demo.db")
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))

# Creating an agent executor:
agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# Identifying all of the tables:
agent_executor.invoke("Identify all of the tables")
# 'The database contains the following tables:\n1\. Orders\n2\. Products\n3\. Users'
```

```py
user_sql = agent_executor.invoke(
    '''Add 5 new users to the database. Their names are:
 John, Mary, Peter, Paul, and Jane.'''
)
'''Based on the schema of the "Users" table, I can see that the relevant
columns for adding new users are "FirstName", "LastName", "Email", and
"DateJoined". I will now run the SQL query to add the new
users.\n\n```sql\nINSERT INTO Users (FirstName, LastName, Email,

DateJoined)\nVALUES (\'John\', \'Doe\', \'john.doe@email.com\',

\'2023-05-01\'), \n(\'Mary\', \'Johnson\', \'mary.johnson@email.com\',

\'2023-05-02\'),\n (\'Peter\', \'Smith\', \'peter.smith@email.com\',

\'2023-05-03\'),\n (\'Paul\', \'Brown\', \'paul.brown@email.com\',

\'2023-05-04\'),\n (\'Jane\', \'Davis\', \'jane.davis@email.com\',

\'2023-05-05\');\n```py\n\nPlease note that I have added the new users with
the specified names and email addresses. The "DateJoined" column is set to the
respective dates mentioned.'''
```

首先，`agent_executor`检查 SQL 数据库以了解数据库模式，然后代理编写并执行一个 SQL 语句，成功将五个用户添加到 SQL 表中。

# 自定义标准代理

考虑如何自定义 LangChain 代理是值得的。关键函数参数可以包括以下内容：

+   `prefix`和`suffix`是直接插入到代理中的提示模板。

+   `max_iterations`和`max_execution_time`为您提供了限制 API 和计算成本的方法，以防代理陷入无限循环：

```py
# This the function signature for demonstration purposes and is not executable.
def create_sql_agent(
    llm: BaseLanguageModel,
    toolkit: SQLDatabaseToolkit,
    agent_type: Any | None = None,
    callback_manager: BaseCallbackManager | None = None,
    prefix: str = SQL_PREFIX,
    suffix: str | None = None,
    format_instructions: str | None = None,
    input_variables: List[str] | None = None,
    top_k: int = 10,
    max_iterations: int | None = 15,
    max_execution_time: float | None = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Dict[str, Any] | None = None,
    extra_tools: Sequence[BaseTool] = (),
    **kwargs: Any
) -> AgentExecutor
```

让我们更新之前创建的`agent_executor`，以便代理可以执行更多的 SQL 语句。`SQL_PREFIX`直接插入到`create_sql_agent`函数中作为`prefix`。此外，您还将插入之前代理中推荐的`user_sql`，该代理不会直接运行`INSERT`、`UPDATE`或`EDIT`命令；然而，新代理将愉快地执行对 SQLite 数据库的 CRUD（创建、读取、更新、删除）操作：

```py
SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to
run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain
always limit your query to at most {top_k} results. You can order the
results by a relevant column to return the most interesting examples in
the database. Never query for all the columns from a specific table, only
ask for the relevant columns given the question. You have access to tools
for interacting with the database. Only use the below tools. Only use the
information returned by the below tools to construct your final answer. You
MUST double-check your query before executing it. If you get an error while
executing a query, rewrite the query and try again. If the question does
not seem related to the database, just return "I don't know" as the answer.
"""

agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prefix=SQL_PREFIX,
)

agent_executor.invoke(user_sql)
# '...sql\nINSERT INTO Users (FirstName, LastName, Email,
# DateJoined)\nVALUES (...)...'

# Testing that Peter was inserted into the database:
agent_executor.invoke("Do we have a Peter in the database?")
'''Yes, we have a Peter in the database. Their details are as follows:\n-
First Name: Peter...'''
```

# LCEL 中的自定义代理

使用 LCEL 创建自定义代理非常简单；让我们使用一个工具创建一个聊天模型：

```py
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1\. Create the model:
llm = ChatOpenAI(temperature=0)

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

# 2\. Create the tools:
tools = [get_word_length]
```

接下来，您将使用系统消息、用户消息和`MessagesPlaceholder`来设置提示，这允许代理存储其中间步骤：

```py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 3\. Create the Prompt:
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are very powerful assistant, but don't know current events
 and aren't good at calculating word length.""",
        ),
        ("user", "{input}"),
        # This is where the agent will write/read its messages from
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
```

在创建代理之前，你需要直接将工具绑定到 LLM 以进行函数调用：

```py
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

# 4\. Formats the python function tools into JSON schema and binds
# them to the model:
llm_with_tools = llm.bind_tools(tools=[convert_to_openai_tool(t)
for t in tools])

from langchain.agents.output_parsers.openai_tools \
import OpenAIToolsAgentOutputParser

# 5\. Setting up the agent chain:
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
```

下面是代码的逐步解析：

1. 导入工具转换函数

你首先导入`convert_to_openai_tool`。这允许你将 Python 函数工具转换为 JSON 模式，使其与 OpenAI 的 LLM 兼容。

2. 将工具绑定到你的语言模型（LLM）

接下来，你将工具绑定到你的 LLM。通过遍历你的`tools`列表中的每个工具并使用`convert_to_openai_tool`进行转换，你实际上创建了`llm_with_tools`。这使你的 LLM 具备了定义工具的功能。

3. 导入代理格式化和解析函数

在这里，你导入`format_to_openai_tool_messages`和`OpenAIToolsAgentOutputParser`。这些格式化代理的草稿本并解析与工具绑定的 LLM 的输出。

4. 设置你的代理链

在这个最终且至关重要的步骤中，你设置了代理链。

+   你可以直接处理用户的输入来引领整个过程。

+   然后你策略性地将中间步骤格式化为 OpenAI 函数消息。

+   然后将调用`llm_with_tools`。

+   `OpenAIToolsAgentOutputParser`用于解析输出。

最后，让我们创建并使用`AgentExecutor`：

```py
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "How many letters in the word Software?"})
#{'input': 'How many letters in the word Software?',
# 'output': 'There are 8 letters in the word "Software".'}
```

LCEL 代理使用`.invoke(...)`函数并正确地识别出单词*software*中有八个字母。

# 理解和使用记忆

与 LLM 交互时，理解记忆的角色和重要性至关重要。这不仅关乎这些模型如何回忆信息，还关乎长期记忆（LTM）和短期记忆（STM）之间的战略互动。

## 长期记忆

将长期记忆视为 LLM 的图书馆。它是一个庞大的、经过精心挑选的数据集合，存储从文本到概念框架的一切。这个知识库帮助模型理解和生成响应。

应用包括：

矢量数据库

这些数据库可以存储非结构化文本数据，为模型在生成内容时提供参考点。通过索引和分类这些数据，LLM 可以通过*相似度距离度量*快速检索相关信息。

自我反思

高级应用包括一个自我反思、记录和存储思想的 LLM。想象一个 LLM，它细致地观察书评平台上的用户模式，并将这些模式作为深入见解进行分类。随着时间的推移，它能够精确地指出偏好，例如喜欢的文学类型和写作风格。这些见解通过检索进行存储和访问。当用户寻求书籍推荐时，由检索到的上下文驱动的 LLM，提供符合他们口味的定制建议。

自定义检索器

创建特定的检索函数可以显著提高 LLM 的效率。与人类记忆系统进行类比，这些函数可以根据数据的相关性、自上次记忆以来经过的时间以及其在实现特定目标中的效用来优先排序数据。

## 短期记忆

在 LLMs 中的短期记忆类似于一个临时工作空间。在这里，最近的交互、活跃的任务或进行的对话被保持在最前沿，以确保连续性和上下文。

应用包括：

对话历史

对于聊天机器人来说，跟踪对话历史至关重要。它允许机器人维护多个交流的上下文，防止冗余查询并确保对话自然流畅。

避免重复

当用户提出相似或相同的查询时，STM 变得非常有价值。通过参考其短期回忆，模型可以提供一致的答案或根据应用程序的需求多样化其响应。

在简要介绍了 LTM 和 STM 的基础概念之后，让我们转向实际应用，特别是在问答（QA）系统的领域。

## QA 对话代理中的短期记忆

想象一下 Eva，一个电子商务平台的虚拟客户支持代理。用户可能会有几个相互关联的查询：

+   用户：“电子产品退货政策有多长时间？”

+   Eva：“电子产品的退货政策是 30 天。”

+   用户：“关于服装商品呢？”

+   Eva，利用 STM：“对于服装商品，是 45 天。您想了解其他任何类别吗？”

注意，通过利用短期记忆（STM），Eva 无缝地继续对话，预测可能的后续问题。这种流畅性仅因有效部署短期记忆，使代理能够将对话视为连贯的交互，而不是孤立的问答。

对于开发者和提示工程师来说，理解和利用这一点可以显著提升用户体验，培养出有意义、高效且类似人类的互动。

# LangChain 中的记忆

LangChain 提供了向 LLMs 添加记忆的简单技术。如图 6-3 所示，链中的每个记忆系统都负责两个基本操作：读取和存储。

理解这一点至关重要，即每个链都有内在的步骤，需要特定的输入。虽然用户提供了一些这些数据，但链也可以从其记忆中获取其他信息。

![LangChain 中的记忆](img/pega_0603.png)

###### 图 6-3\. LangChain 中的记忆

在链的每个操作中，都有两个与其记忆的关键交互：

+   *在收集初始用户数据之后但在执行之前*，链从其记忆中检索信息，添加到用户的输入中。

+   *在链完成之后但在返回答案之前*，链会将当前运行的输入和输出写入记忆，以便在未来的运行中参考。

在创建记忆系统时，您需要做出两个关键的选择：

+   存储状态的方法

+   查询记忆状态的方法

## 保存状态

在表面之下，生成式 AI 模型的基础记忆是以一系列聊天消息的形式组织的。这些消息可以存储在临时的内存列表中，或者锚定在更持久的数据库中。对于那些倾向于长期存储的人来说，有各种各样的[数据库集成可用](https://oreil.ly/ECD_n)，简化了流程，并让你免于手动集成的麻烦。

通过五到六行代码，你可以轻松地集成一个基于`session_id`参数的独特`MongoDBChatMessageHistory`：

```py
# Provide the connection string to connect to the MongoDB database.
connection_string = "mongodb://mongo_user:password123@mongo:27017"

chat_message_history = MongoDBChatMessageHistory(
    session_id="test_session",
    connection_string=connection_string,
    database_name="my_db",
    collection_name="chat_histories",
)

chat_message_history.add_user_message("I love programming!!")
chat_message_history.add_ai_message("What do you like about it?")

chat_message_history.messages
# [HumanMessage(content='I love programming!!',
# AIMessage(content='What do you like about it?')
```

## 查询状态

一个基本的记忆框架可能只是每次交互时传递最新的消息。一个稍微复杂一点的设置可能会提炼出最后一组消息的清晰摘要。一个更高级的设置会从对话中识别特定实体，并仅传递当前会话中突出显示的这些实体的数据。

不同的应用对记忆查询有不同的需求。LangChain 的记忆工具包将帮助你创建简单的记忆基础设施，并在必要时赋予你构建定制系统的能力。

## ConversationBufferMemory

LangChain 中有各种类型的记忆，其中最受欢迎的是 ConversationBufferMemory。这允许你存储多个聊天消息，没有对聊天历史大小的限制。

首先导入`ConversationBufferMemory`，然后你可以使用`save_context`函数添加上下文。`load_memory_variables`函数返回一个包含`Human`和`AI`消息的 Python 字典：

```py
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.load_memory_variables({})
# {'history': 'Human: hi\nAI: whats up'}
```

你还可以通过将`return_messages=True`添加到`ConversationBufferMemory`来返回 LangChain 的架构消息，即`SystemMessage`、`AIMessage`或`HumanMessage`：

```py
memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.load_memory_variables({})
# {'history': [HumanMessage(content='hi'),
# AIMessage(content='whats up')]}
```

让我们在 LCEL 中直接将记忆添加到一个链中：

```py
# Using within a chain:
from langchain.memory import ConversationBufferMemory
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

memory = ConversationBufferMemory(return_messages=True)

model = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Act as a chatbot that helps users with their queries."),
        # The history of the conversation
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
chain = (
    {
        "input": lambda x: x["input"],
        "history": RunnableLambda(memory.load_memory_variables) | \
        itemgetter("history"),
    }
    | prompt
    | model
    | StrOutputParser()
)
```

注意`MessagesPlaceholder`有一个`variable_name`为`"history"`。这与`ConversationBufferMemory`中的`memory`键对齐，允许之前的聊天历史直接格式化到`ChatPromptTemplate`中。

在设置好 LCEL 链之后，让我们调用它并将消息保存到`memory`变量中：

```py
inputs = {"input": "Hi my name is James!"}
result = chain.invoke(inputs)
memory.save_context(inputs, {"outputs": result})
print(memory.load_memory_variables({}))

# {'history': [HumanMessage(content='Hi my name is James!'),
# AIMessage(content='Hello James! How can I assist you today?')]}
```

记忆包含两条消息，一条`HumanMessage`和一条`AIMessage`；两者都通过使用`save_context`函数保存到记忆中。让我们测试 LCEL 链是否能够使用之前的内容来回答新的问题：

```py
inputs = {"input": "What is my name?"}
second_result = chain.invoke(inputs)
print(second_result)
# Your name is James.
```

LCEL 链现在能够使用之前的消息来回答新的查询了！

此外，你还可以通过在`ChatPromptTemplate`中添加`MessagesPlaceHolder`并将记忆添加到`AgentExecutor`中来轻松地为代理添加记忆：

```py
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a very powerful assistant, but don't know current events
and aren't good at calculating word length.""",
        ),
        # This is where the agent will write/read its messages from
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ]
)

# ... The rest of the code remains the same as before ...

# Create an agent executor by passing in the agent, tools, and memory:
memory = ConversationBufferMemory(return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
memory=memory)
```

你可以在本地的[Jupyter Notebook](https://oreil.ly/LXQNy)中查看完整的实现。

通过利用这种记忆，你的代理可以提供更加情境感知和流畅的对话体验，从而无需额外的工具来回忆过去的交互。

ConversationBufferMemory 没有缓冲区限制，但不同的内存类型，如 ConversationSummaryBufferMemory，允许你指定最大标记限制，超过该限制后，对话将被总结：

```py
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.load_memory_variables({})
# {'history': 'Human: hi\nAI: whats up'}
```

###### 注意

默认情况下，内存存储在 Python 进程的本地。这种方法本质上是短暂的，并受会话或进程生命周期的限制。对于需要时间连续性和能够从历史数据中学习应用程序，转向数据库支持的内存变得至关重要。

对于基于数据库的内存（[database-backed memory](https://oreil.ly/nTBox)），有几种集成方式，这些方式将内存使用从短期、会话特定的上下文转变为更健壮的长期存储解决方案。

# LangChain 中其他流行的内存类型

虽然 ConversationBufferMemory 是一种众所周知的内存类型，但它有一些限制，例如上下文长度限制、可能缺乏相关性以及缺乏摘要。为了解决这些问题，LangChain 提供了几种其他内存类型。

## ConversationBufferWindowMemory

此类型维护最近交互的滑动窗口，确保缓冲区不会过大。功能包括以下内容：

+   仅保留最后`K`次交互

+   可以返回历史记录作为字符串或消息列表

```py
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})
# Returns: {'history': 'Human: not much you\nAI: not much'}
memory.load_memory_variables({})
```

## ConversationSummaryMemory

这一种方法会随着时间的推移对对话进行压缩和总结，非常适合较长的对话，因为逐字记录消息历史可能会消耗大量的标记。功能包括以下内容：

+   在线总结对话

+   可以返回历史记录作为摘要字符串或系统消息列表

+   允许直接预测新的摘要

+   可以使用现有的消息或摘要进行初始化

```py
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain_openai import OpenAI

memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.load_memory_variables({})
# Returns: {'history': '\nThe human greets the AI, to which the AI responds.'}
```

## ConversationSummaryBufferMemory

这是一个混合内存，它不仅维护最近交互的缓冲区，还将较老的交互编译成摘要。

功能包括以下内容：

+   使用标记长度来确定何时刷新交互

+   可以返回带有最近交互的历史记录摘要或消息列表

+   允许直接预测新的摘要

```py
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai.chat_models import ChatOpenAI

memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=10)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.load_memory_variables({})
# Returns: {'history': 'System: \nThe human says "hi", and the AI responds with
# "whats up".\nHuman: not much you\nAI: not much'}
```

## ConversationTokenBufferMemory

这一种方法使用标记长度来保持最近的交互，并确定何时刷新交互。

功能包括以下内容：

+   使用标记长度进行刷新

+   可以返回历史记录作为字符串或消息列表

```py
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai.chat_models import ChatOpenAI

memory = ConversationTokenBufferMemory(llm=ChatOpenAI(),
max_token_limit=50)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.load_memory_variables({})
# Returns: {'history': 'Human: not much you\nAI: not much'}
```

你已经了解了在 LangChain 中内存的重要性。现在，你也明白了如何使用 LangChain 的内存工具包构建和自定义内存系统，包括存储状态和查询内存的方法；你看到了将 MongoDBChatMessageHistory 集成和利用多才多艺的 ConversationBufferMemory 的示例。

让我们总结一下 LangChain 中可用的不同内存类型及其可能特别有用的场景：

ConversationBufferWindowMemory

此内存类型维护最近的交互，因此在对话的上下文至关重要且不使缓冲区过大时非常有用。

交互总结内存

对于长对话来说，这种内存类型提供了对话的摘要版本，节省了宝贵的令牌空间。

交互总结缓冲内存

适用于你不仅想记录最近交互，还想将旧交互汇总成总结的情况，从而提供一种混合方法。

词汇缓冲内存

当定义特定令牌长度至关重要且需要维护最近交互的缓冲区时，这种内存类型很有用。它根据令牌长度确定何时刷新交互。

了解可用的不同内存选项可以帮助你根据具体情况选择最适合你的需求的一个。

# 指导方向

即使你在确定要使用哪种内存类型时，也要记住适当地指导 AI 模型。例如，使用 ConversationBufferWindowMemory，你需要指定你想要保留的最近交互的数量（`K`）。为了获得最佳结果，请明确你的要求。

# 带有内存的 OpenAI 函数代理

通过 GitHub 上的一个综合示例深入了解代理，[GitHub](https://oreil.ly/jyLab)。在这个示例中，你将发现 OpenAI 如何整合几个关键组件：

+   使用聊天消息进行内存管理

+   使用能够处理多个函数参数的工具，如 API 请求和文件保存

+   集成自定义 `SystemMessage` 以指导和定义代理的行为

为了说明，考虑一下 Python 函数的文档字符串是如何用来提供工具描述的：

```py
from langchain.tools import StructuredTool

def save_interview(raw_interview_text: str):
    """Tool to save the interview. You must pass the entire interview and
 conversation in here. The interview will then be saved to a local file.
 Remember to include all of the previous chat messages. Include all of
 the messages with the user and the AI, here is a good response:
 AI: some text
 Human: some text
 ...
 ---
 """
    # Save to local file:
    with open("interview.txt", "w") as f:
        f.write(raw_interview_text)
    return f'''Interview saved! Content: {raw_interview_text}. File:
 interview.txt. You must tell the user that the interview is saved.'''

save_interview = StructuredTool.from_function(save_interview)
```

`StructuredTool.from_function()` 将创建一个 LangChain 工具，该工具能够接受多个函数参数。

# 指导并指定格式

Python 函数中的文档字符串展示了指导 LLM 使用 `raw_interview_text` 参数内容的指定格式。

此外，`return` 语句强调指导 LLM 通知用户访谈已被存储。这确保代理在工具执行后返回更具对话性的响应。

为了进一步展示提示工程技巧，让我们检查笔记本中的另一个 Python 代码片段：

```py
from pydantic.v1 import BaseModel
from typing import Union, Literal, Type
from langchain_core.tools import BaseTool

class ArgumentType(BaseModel):
    url: str
    file_type: Union[Literal["pdf"], Literal["txt"]]

class SummarizeFileFromURL(BaseTool):
    name = "SummarizeFileFromURL"
    description = "Summarize a file from a URL."
    args_schema: Type[ArgumentType] = ArgumentType
```

在这个示例中，`args_schema` 在 `SummarizeFileFromURL` 类中使用。该属性利用 `ArgumentType` 类，确保在执行之前验证工具的参数。具体来说，它强制提供有效的 URL 字符串，并且 `file_type` 参数应该是 `"pdf"` 或 `"txt"`。

通过添加验证检查，你可以确保代理正确处理功能参数，这反过来又提高了工具执行的整体可靠性和效率。

# 高级代理框架

你现在已经了解了 ReAct 和 OpenAI 函数，但还有其他几个代理框架。其他两个流行的框架包括 *计划和执行代理* 和 *思维树*。

## 计划和执行代理

而不是让大型语言模型（LLM）进行任务规划和工具执行，你可以将这一过程分为两个独立的模块。每个模块都可以由一个可以访问目标、当前任务和已完成任务的独立 LLM 分别处理。

计划与执行框架的两个流行版本包括[BabyAGI](https://oreil.ly/xeijG)和[AutoGPT](https://oreil.ly/M4z8K)。

图 6-4 展示了 BabyAGI 的智能体设置，该设置旨在将 OpenAI LLM 与 Chroma/Weaviate 等向量数据库合并，以创建一个强大、自适应的任务管理系统。

在一个连续的循环中，智能体首先获取一个任务，并将其传递给`execution_agent`，该智能体利用 OpenAI 根据上下文数据执行任务。之后，结果被丰富并存档在 Chroma/Weaviate 中。

`task_creation_agent`随后介入，利用 OpenAI 从先前任务的目标和结果中识别新任务。这些任务以字典列表的形式呈现，为结果任务提供结构。

然后`prioritization_agent`与 OpenAI 交互，重新排列任务列表，确保与主要目标保持一致。这些智能体的协同作用确保系统始终在进化，以信息化的方式持续生成和优先处理任务。整合[Chroma](https://oreil.ly/9R3pU)或[Weaviate](https://oreil.ly/2wu-y)起着至关重要的作用，因为它提供了一个上下文数据的宝库，确保任务始终与其预定义的目标保持一致。

![BabyAGI](img/pega_0604.png)

###### 图 6-4\. BabyAGI 的智能体架构

[计划与执行智能体类型](https://oreil.ly/8vYF5)确实存在于 LangChain 中，尽管它仍然是实验性的。

## 思维树

随着语言模型在解决各种任务中的应用不断扩大，它们的推理方法仍然局限于基于标记的、线性的处理。这种方法在许多情况下都有效，但在需要高级战略预见或初始决策至关重要的任务面前就有限了。[思维树（ToT）框架](https://oreil.ly/1rYDI)是一种利用语言模型的新方法，它超越了传统的思维链提示技术(图 6-5)。

思维树（ToT）的核心前提是允许在连贯的文本块之间进行探索，这些文本块被称为*思维*。这些思维代表了问题解决的各个阶段，有助于语言模型进行更谨慎的决策过程。模型不再局限于一条推理路径，可以探索各种推理轨迹，并在每一步自我评估其决策。该框架旨在允许进行前瞻性规划、回顾过去的决策，并做出总体选择。

![思维树](img/pega_0605.png)

###### 图 6-5\. 思维树（ToT）

其成功的证据来自于对需要复杂规划或搜索能力的任务的实验结果。在一个像*24 点游戏*这样的游戏中，传统的 GPT-4 在提示使用思维链时，成功率为 4%。相比之下，ToT 方法将这一数字飙升至令人印象深刻的 74%。这种范式转变不仅限于游戏。ToT 方法在创意写作和迷你填字游戏等领域的应用也显示出其多功能性。

与理论相辅相成的是[LangChain 实现](https://oreil.ly/fub1z)，它展示了 ToT 如何实现。一个数独谜题作为说明性示例，主要目的是用数字替换通配符字符(*)，同时遵守数独规则。

ToT 不仅仅是一种新的方法；它是在我们设想语言模型推理方面的范式转变。通过为模型提供思考、回溯和策略规划的能力，ToT 正在重新定义 AI 问题解决的边界。

如果你将 ToT 视为指挥 LLMs 的策略，LangChain 回调可以被视为诊断和确保这些策略顺利运行的工具。让我们深入了解如何有效地利用这一功能。

# 回调

LangChain 的[*回调*](https://oreil.ly/8EhXl)使您能够无缝监控和定位应用程序中的问题。到目前为止，您在`AgentExecutor`链中遇到了`verbose=True`参数：

```py
AgentExecutor(.., verbose=True)
```

此参数记录了用于调试的有用输出，但如果你热衷于跟踪特定事件呢？请进入回调，这是你的首选解决方案。

`BaseCallbackHandler`类在您的生成式 AI 模型执行期间充当监控和响应各种事件的基础。该类中的每个方法都对应于特定的阶段，如模型的开始、结束，甚至在模型运行时发生的错误。例如，当 LLM 开始其操作时，会触发`on_llm_start`。同样，`on_chain_error`和`on_tool_end`等方法会对链中的错误或使用工具后的错误做出反应：

```py
class BaseCallbackHandler:
    """Base callback handler that can be used to handle callbacks from
 langchain."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str],
        **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

    def on_chat_model_start(
        self, serialized: Dict[str, Any],
        messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any],
        **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
```

每个回调都可以作用域为类或单个请求。

## 全局（构造函数）回调

在构造函数中定义回调，例如`AgentExecutor(callbacks=[handler], tags=['a-tag'])`，它们会在对该实例的每次调用时被激活。这些回调仅限于该特定实例。为了说明，当一个处理程序在创建`LLMChain`时被传递，它不会与任何子链交互：

```py
from langchain.agents import AgentExecutor
from langchain.callbacks import StdOutCallbackHandler

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    callbacks=[StdOutCallbackHandler()],
    tags=['a-tag'])

agent_executor.invoke({"input": "How many letters in the word Software?"})
```

你包含的标签，如`'a-tag'`，在追踪和整理您的生成式 AI 设置输出方面非常有用。特别是在具有许多链的大型项目中，利用标签可以显著简化您的流程。

## 请求特定回调

另一方面，回调可以在`invoke()`方法中定义。例如，对`LLMChain`的请求可能会随后触发另一个`LLMChain`请求，并应用相同的处理程序：

```py
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

handler = StdOutCallbackHandler()
llm = OpenAI()
prompt = PromptTemplate.from_template("What is 1 + {number} = ")
chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"number": 2}, {"callbacks": [handler]})
```

## 详细说明论点

常用实用工具 `verbose` 参数对大多数 API 对象都是可用的。当您使用 `AgentExecutor(verbose=True)` 时，它与将 `ConsoleCallbackHandler` 集成到对象的回调参数及其后代中相同。它通过直接将每个事件记录到您的控制台而充当有用的调试工具。

## 何时使用哪个？

构造函数回调

适用于跨整个链的日志记录或监控等全局任务。如果您目标是跟踪代理的所有交互，请在初始化期间附加处理程序。

请求回调

定制用于特定用例，如流式传输，其中单个请求的输出被转发到专用端点，例如 WebSocket。因此，对于需要将单个请求的输出流式传输到 WebSocket 的场景，处理程序应链接到 `invoke()` 方法。

冗长参数

对于调试和本地 LLM 开发很有用，但它可以生成大量的日志。

## 使用 LangChain 进行令牌计数

LangChain 在您与生成式 AI 模型交互期间提供了有效的令牌计数方法。

您需要设置必要的模块；导入 `asyncio` 模块和 LangChain 包中的相关函数：

```py
import asyncio
from langchain.callbacks import get_openai_callback
from langchain_core.messages import SystemMessage
from langchain_openai.chat_models import ChatOpenAI
model = ChatOpenAI()
```

现在，使用 `get_openai_callback` 上下文管理器发出请求并计算使用的令牌：

```py
with get_openai_callback() as cb:
    model.invoke([SystemMessage(content="My name is James")])
total_tokens = cb.total_tokens
print(total_tokens)
# 25
assert total_tokens > 0
```

执行此代码后，`total_tokens` 将存储您请求使用的令牌数。

在上下文管理器中发出多个请求时，您可以验证总令牌计数是否准确：

```py
with get_openai_callback() as cb:
    model.invoke([SystemMessage(content="My name is James")])
    model.invoke([SystemMessage(content="My name is James")])
assert cb.total_tokens > 0
print(cb.total_tokens)
# 50
```

如您所观察到的，两次发出相同的请求通常会导致 `cb.total_tokens` 是 `total_tokens` 的两倍。

LangChain 支持并发运行，让您可以同时执行多个请求：

```py
# Async callbacks:
with get_openai_callback() as cb:
    await asyncio.gather(
        model.agenerate(
            [
                [SystemMessage(content="Is the meaning of life 42?")],
                [SystemMessage(content="Is the meaning of life 42?")],
            ],
        )
    )
print(cb.__dict__)
# {'successful_requests': 2, 'total_cost': 0.000455,
# 'total_tokens': 235, 'prompt_tokens': 30,
# 'completion_tokens': 205}
```

`cb` 提供了您与 AI 模型的交互的详细分解，提供了对提示工程至关重要的关键指标：

+   `cb.successful_requests` 跟踪已成功执行的请求数量。它是有效处理 API 请求且未遇到错误的直接指标。

+   使用 `cb.total_cost`，您可以得到与您的请求相关的成本的透明视图。这在大量使用 AI 时，对于预算和费用管理是一个关键指标。

+   `cb.total_tokens` 表示在提示和完成过程中使用的累积令牌数。这提供了一个对令牌消耗的整体视图。

+   `cb.prompt_tokens` 提供了您提供的提示中使用的令牌数量的见解。这可以帮助您优化提示，使其简洁而有效。

+   `cb.completion_tokens` 强调了 AI 响应占用的令牌数量。这在分析 AI 响应的冗长或深度时可能很有用。

# 摘要

在本章中，你学习了思维链推理的概念及其在自主代理中的重要性。你发现了 LLM 如何将复杂问题分解成更小的组件以提供有效解决方案。

此外，你探索了生成式 AI 模型中的基于代理的架构，并获得了关于记忆集成和高级代理框架的宝贵见解。你研究了几个代理框架，例如 ReAct 和 OpenAI 函数调用，并了解到这些框架通过利用外部工具来增强 LLM 模型的响应。

在第七章中，你将了解到使用生成式 AI 进行图像生成。你将学习生成式 AI 图像模型的历史，包括每个供应商的优缺点。
