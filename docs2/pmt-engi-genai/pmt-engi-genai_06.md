# 第六章：具有记忆和工具的自主代理

本章深入探讨了思维链推理的重要性以及大型语言模型（LLMs）作为代理通过复杂问题进行推理的能力。通过将复杂问题分解成更小、更易于管理的组件，LLMs 可以提供更全面和有效的解决方案。你还将了解构成自主代理的组件，例如输入、目标或奖励函数以及可用的动作。

# 思维链

AI 通过复杂问题进行推理的能力对于创建有效、可靠和用户友好的应用程序至关重要。

*思维链推理*（CoT）是一种引导 LLM 通过一系列步骤或逻辑连接来得出结论或解决问题的方法。这种方法对于需要更深入理解上下文或考虑多个因素的任务特别有用。

[思维链](https://oreil.ly/fAeLo)要求 LLM 通过复杂问题进行思考，将它们分解成更小、更易于管理的组件。这允许 LLM 单独关注每个部分，确保对当前问题的更深入理解。

在实践中，思维链推理可能涉及：

+   要求 LLM 为其决策提供解释

+   在做出最终决定之前规划多个步骤

在接下来的章节中，你将探索无效和有效思维链推理的例子。我们还将讨论构建有效思维链推理的各种技术以及它们如何集成到 AI 应用中。

让我们想象一下，一个用户希望 AI 生成一个全面的市场推广计划来推广一款新的软件产品。

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

在这个例子中，GPT-4 没有使用思维链推理，也没有针对市场推广计划的特定方面。LLM 生成了一份通用的营销策略列表，这些策略可以适用于任何产品，而不是专注于新软件产品的独特特性。

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

注意到“逐步”这个短语，这是思维链中的一个关键要素。通过将这个短语纳入你的提示，你是在要求 LLM 通过生成高度有效的软件产品所需的步骤进行推理。

此外，通过提供 20,000 美元的预算和软件类型，GPT-4 能够提供更加相关和具体化的响应。

# 代理

生成式 AI 模型催生了基于代理的架构。从概念上讲，代理在指定环境中行动、感知并做出决策以实现预定义的目标。

代理可以采取各种行动，例如执行 Python 函数；之后，代理将观察发生了什么，并决定是否完成或采取下一步行动。

代理将不断循环一系列动作和观察，直到没有更多的动作，如下面的伪代码所示：

```py
next_action = agent.get_action(...)
while next_action != AgentFinish:
    observation = run(next_action)
    next_action = agent.get_action(..., next_action, observation)
return next_action
```

代理的行为由三个主要组成部分控制：

输入

这些是代理从其环境中接收的感官刺激或数据点。输入可以多种多样，从视觉（如图像）和听觉（如音频文件）到热信号等。

目标或奖励函数

这代表了一个代理行为的指导原则。在基于目标的框架中，代理被要求达到一个特定的最终状态。在基于奖励的设置中，代理被驱动去最大化随时间累积的奖励，通常是在动态环境中。

可用的动作

*动作空间*是代理在任何给定时刻可以采取的允许动作的范围。[可以采取的动作](https://oreil.ly/5AVfM)的广度和性质取决于手头的任务。

为了进一步解释这些概念，考虑一下自动驾驶汽车：

输入

汽车的传感器，如摄像头、激光雷达和超声波传感器，提供有关环境的连续数据流。这可能包括关于附近车辆、行人、道路状况和交通信号的信息。

目标或奖励函数

自动驾驶汽车的主要目标是安全且高效地从 A 点到 B 点的导航。如果我们使用基于奖励的系统，汽车可能会因为保持与其他物体的安全距离、遵守速度限制和遵守交通规则而获得积极奖励。相反，它可能会因为危险行为（如紧急制动或偏离车道）而收到负面奖励。特斯拉特别使用无需干预行驶的英里数作为他们的奖励函数。

可用的动作

汽车的动作空间包括加速、减速、转弯、换车道等。每个动作都是根据当前输入数据和由目标或奖励函数定义的目标来选择的。

你会发现，在自动驾驶汽车等系统中，代理依赖于基础原则，如输入、目标/奖励函数和可用的动作。然而，当深入研究 GPT 等 LLM 领域时，存在一套专门的动力机制，专门针对它们的独特性质。

这就是它们如何与您的需求相匹配：

输入

对于 LLM 来说，主要入口是通过文本。但这并不限制你可以使用的信息量。无论你是在处理温度读数、音乐符号，还是复杂的数据结构，你的挑战在于将这些内容塑造成适合 LLM 的文本表示。想想视频：虽然原始素材可能看起来不兼容，但视频文本转录允许 LLM 为你提取见解。

利用目标驱动的指令

LLMs 主要使用你文本提示中定义的目标。通过创建具有目标的有效提示，你不仅访问了 LLM 的广泛知识，而且实际上绘制了它的推理路径。把它想象成绘制蓝图：你的特定提示指导模型，引导它将你的总体目标分解成一系列系统的步骤。

通过功能性工具制定行动

LLMs 不仅限于简单的文本生成；还有更多你可以实现的事情。通过集成*现成的工具*或*自定义开发的工具*，你可以装备 LLMs 执行各种任务，从 API 调用到数据库交互，甚至编排外部系统。工具可以用任何编程语言编写，通过添加更多工具，你实际上是在*扩展 LLM 可以实现的行为空间*。

也有一些组件可以直接应用于 LLMs：

记忆

在代理步骤之间存储状态是理想的；这在聊天机器人中尤其有用，记住之前的聊天历史可以提供更好的用户体验。

代理规划/执行策略

实现高级目标有多种方式，其中规划和执行的结合是至关重要的。

检索

LLMs 可以使用不同类型的检索方法。在向量数据库中的语义相似性是最常见的，但还有其他方法，例如将来自 SQL 数据库的自定义信息包含到提示中。

让我们更深入地探讨共享和不同的组件，并探索实现细节。

## 理解和行动（ReAct）

有许多代理框架最终旨在提高 LLMs 向目标响应的能力。原始框架是*ReAct*，它是 CoT 的改进版本，允许 LLM 在通过工具采取行动后创建观察。然后，这些观察被转化为关于下一步应该使用什么*正确工具*的*思考*（图 6-1）。LLM 会继续推理，直到出现一个`'Final Answer'`字符串值或者达到最大迭代次数。

![ReAct 框架](img/pega_0601.png)

###### 图 6-1\. ReAct 框架

[ReAct](https://oreil.ly/ssdnL)框架结合了任务分解、思考循环和多个工具来解决疑问。让我们探索 ReAct 中的思考循环：

1.  观察环境。

1.  用思考来解释环境。

1.  决定一个行动。

1.  在环境中采取行动。

1.  重复步骤 1-4，直到找到解决方案或迭代次数过多（解决方案是“我已经找到了答案”）。

您可以通过使用前面的思考循环，同时向 LLM 提供多个输入，如：

+   `{question}`: 你想要回答的查询。

+   `{tools}`：这些指的是可以用于完成整体任务中某个步骤的函数。通常的做法是包括一个工具列表，其中每个工具都是一个 Python 函数、一个名称以及该函数及其目的的描述。

以下是一个实现 ReAct 模式且提示变量用 `{}` 字符括起来的提示：

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

1.  提示的引入清楚地确立了 LLM 的目的：`您将尝试解决找到问题答案的问题。`

1.  然后概述了解决问题的方法：`使用链式思维推理通过以下模式解决问题：`

1.  链式思维推理的步骤随后被列出：

    +   LLM 首先观察原始问题，然后对其形成观察：`original_question: original_problem_text`，`observation: observation_text`。

    +   基于此观察，AI 应该制定一个表示推理过程中一步的思想：`thought: thought_text`。

    +   在建立了一个思想之后，它接着决定使用可用的工具之一采取行动：`action: tool_name`，`action_input: tool_input`。

1.  然后提醒 LLM 不要对工具可能返回的内容做出假设，并且它应该明确说明其预期的行动和相应的输入。

1.  `您可以使用以下工具：{tools}` 通知 LLM 它可用于解决问题的工具。

1.  然后介绍了 LLM 必须解决的真正问题：`original_ 问题: {question}`。

1.  最后，根据其行动的结果，提供了 LLM 应如何响应的说明。它可以选择继续进行新的观察、行动和输入，或者如果找到了解决方案，提供最终答案。

提示概述了一个系统化的问题解决过程，其中 LLM 观察一个问题，思考它，决定采取的行动，并重复此过程，直到找到解决方案。

## 理由与行动实施

现在，您已经了解了 ReAct，创建一个简单的 Python 实现来复制 LangChain 自动执行的操作非常重要，这样您就可以构建关于 LLM 响应之间真正发生的事情的直觉。

为了简化，此示例将不会实现循环，并假设输出可以从单个工具调用中获得。

要创建一个基本的 ReAct 实现，您将实现以下内容：

1.  在每个思考过程中，您需要提取 LLM 想要使用的工具。因此，您将提取最后一个 `action` 和 `action_input`。`action` 代表工具名称，而 `action_input` 包含函数参数的值。

1.  检查 LLM 是否认为它已经找到了最终答案，如果是这样，则思想循环结束。

您可以使用正则表达式从 LLM 响应中提取 `action` 和 `action_input` 值：

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

+   `(?i)`: 这被称为 *内联标志*，使得正则表达式模式不区分大小写。这意味着模式将匹配“action”、“Action”、“ACTION”或任何其他大小写组合。

+   `action`: 模式的一部分匹配单词 *action* 字面意义。由于不区分大小写的标志，它将匹配该单词的任何大小写形式。

+   `\s*`: 模式的一部分匹配零个或多个空白字符（空格、制表符等）。`\*` 表示 *零个或多个*，`\s` 是正则表达式的空白字符简写。

+   `:` 模式的一部分匹配字面意义上的冒号。

+   `\s*`: 这与之前的 `\s\*` 部分相同，匹配冒号后面的零个或多个空白字符。

+   `+([^\n]++)`: 这个模式是一个捕获组，由括号表示。它匹配一个或多个不是换行符的字符。方括号 `[]` 内的 `^` 否定字符类，`\n` 表示换行符。`+` 表示 *一个或多个*。该组匹配的文本将在使用 `findall()` 函数时被提取。

+   `re.MULTILINE`: 这是传递给 `re.compile()` 函数的标志。它告诉正则表达式引擎输入文本可能有多个行，因此模式应该逐行应用。

+   在正则表达式中，方括号 `[]` 用于定义字符类，这是一组你想要匹配的字符。例如，`[abc]` 将匹配任何单个字符，该字符是 `'a'`、`'b'` 或 `'c'`。

+   当你在字符类开始处添加一个撇号 `^` 时，它否定字符类，这意味着它将匹配不在字符类中的任何字符。换句话说，它反转了你想要匹配的字符集。

+   因此，当我们使用 `[^abc]` 时，它将匹配任何不是 `'a'`、`'b'` 或 `'c'` 的单个字符。在正则表达式模式 `+([^\n]++)` 中，字符类是 `[^n]`，这意味着它将匹配任何不是换行符 (`\n`) 的字符。字符类否定后的 `+` 表示模式应该匹配一个或多个不是换行符的字符。

+   通过在捕获组中使用否定字符类 `[^n]`，我们确保正则表达式引擎捕获直到行尾的文本，而不包括换行符本身。这在我们要提取单词 *action* 或 *action input* 后直到行尾的文本时很有用。

总体而言，这个正则表达式模式匹配单词 *action*（不区分大小写）后面跟可选的空白，一个冒号，再次跟可选的空白，然后捕获任何直到行尾的文本。

这两个正则表达式模式之间的唯一区别是它们在开始处寻找的文本：

1.  `action_pattern` 寻找单词 `"action"`。

1.  `action_input_pattern` 查找单词 `"action_input"`.

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

为了确定和提取语言模型是否发现了最终答案，您也可以使用正则表达式：

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

语言模型并不总是以预期的方式响应，因此您的应用程序需要能够处理正则表达式解析错误。几种方法包括使用语言模型来修复前一个语言模型的响应或使用前一个状态发出另一个新的语言模型请求。

您现在可以组合所有组件；以下是一个逐步解释：

```py
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
```

初始化 `ChatOpenAI` 实例：

```py
chat = ChatOpenAI(model_kwargs={"stop": ["tool_result:"],})
```

添加一个 `stop` 序列会迫使语言模型在遇到短语 `"tool_result:"` 后停止生成新的标记。这有助于通过停止工具使用时的幻觉。

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

original_problem: {question} `"""`
```

```py```` 生成模型输出：    ```py output = chat.invoke(SystemMessagePromptTemplate \ .from_template(template=base_prompt) \ .format_messages(tools=tools, question="Is Jason Derulo with a partner?")) print(output) ```    提取最后一个 `action`、`action_input` 并调用相关函数：    ```py tool_name = extract_last_action_and_input(output.content)["action"] tool_input = extract_last_action_and_input(output.content)["action_input"] tool_result = tools[tool_name]"function" ```    打印工具详细信息：    ```py print(f"""The agent has opted to use the following tool: tool_name: {tool_name} `tool_input:` `{``tool_input``}` ``` `tool_result:` `{``tool_result``}``"""` `)` ```py ```   ```py``` ````` Set the current prompt with the tool result:    ```py current_prompt = """ You are answering this query: Is Jason Derulo with a partner?  Based on the provided tool result: tool_result: {tool_result} `Either provide the next observation, action, action_input, or the final` `answer if available. If you are providing the final answer, you must return` `the following pattern: "I've found the answer: final_answer"` `"""` ```   ```py`Generate the model output for the current prompt:    ``` output = chat.invoke(SystemMessagePromptTemplate. \ from_template(template=current_prompt) \ .format_messages(tool_result=tool_result)) ```py    Print the model output for the current prompt:    ``` print("----------\n\nThe model output is:", output.content) final_answer = extract_final_answer(output.content) if final_answer:     print(f"answer: {final_answer}") else:     print("No final answer found.") ```py    Output:    ``` '''content='1\. Observe the original question:\nIs Jason Derulo with a partner?\n\n2\. Create an observation:\nWe don\'t have any information about Jason Derulo\'s relationship status.\n\n3\. Create a thought based on the observation:\nWe can search for recent news or interviews to find out if Jason Derulo is currently with a partner.\n\n4\. Use the tool to act on the thought:\naction: search_on_google\naction_input: "Jason Derulo current relationship status"' additional_kwargs={} example=False  ---------- The agent has opted to use the following tool: tool_name: search_on_google tool_input: "Jason Derulo current relationship status" tool_result: Jason Derulo doesn't have a wife or partner. ----------  The second prompt shows Based on the provided tool result: tool_result: {tool_result}  Either provide the next observation, action, action_input, or the final answer if available. If you are providing the final answer, you must return the following pattern: "I've found the answer: final_answer" ----------  The model output is: I've found the answer: Jason Derulo doesn't have a wife or partner. answer: Jason Derulo doesn't have a wife or partner.''' ```py    The preceding steps provide a very simple ReAct implementation. In this case, the LLM decided to use the `search_on_google` tool with `"Jason Derulo current relationship status"` as the `action_input`.    ###### Note    LangChain agents will automatically do all of the preceding steps in a concise manner, as well as provide multiple tool usage (through looping) and handling for tool failures when an agent can’t parse the `action` or `action_input`.    Before exploring LangChain agents and what they have to offer, it’s vital that you learn *tools* and how to create and use them.```` ```py`` ``````py ``````py`  ````` ```py`## Using Tools    As large language models such as GPT-4 can only generate text, providing tools that can perform other actions such as interacting with a database or reading/writing files provides an effective method to increase an LLM’s capabilities. A *tool* is simply a predefined function that permits the agent to take a specific action.    A common part of an agent’s prompt will likely include the following:    ``` 你想要完成的是：{goal} 你可以访问以下 {tools} ```py    Most tools are written as functions within a programming language. As you explore LangChain, you’ll find that it offers three different approaches to tool creation/usage:    *   Create your own custom tools.           *   Use preexisting tools.           *   Leverage `AgentToolkits`, which are multiple tools bundled together to accomplish a specific task.              Let’s start by creating a custom tool that checks the length of a given string using LangChain:    ``` # 导入必要的类和函数：from langchain.agents import AgentExecutor, create_react_agent from langchain import hub from langchain_openai import ChatOpenAI from langchain.tools import Tool  # 定义要使用的 LLM：model = ChatOpenAI()  # 计算字符串中字符数量的函数：def count_characters_in_string(string):     return len(string)  # 创建工具列表：# 目前只定义了一个工具，用于计算文本字符串中的字符数。tools = [     Tool.from_function(         func=count_characters_in_string,         name="Count Characters in a text string",         description="计算文本字符串中的字符数",     ) ]  # 下载一个 React 提示！prompt = hub.pull("hwchase17/react")  # 构建 ReAct 代理：agent = create_react_agent(model, tools, prompt)  # 使用定义的工具和代理初始化代理执行器：agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # 使用查询调用代理以计算给定单词中的字符数：agent_executor.invoke({"input": '''How many characters are in the word "supercalifragilisticexpialidocious"?'''})  # 'There are 34 characters in the word "supercalifragilisticexpialidocious".' ```py    Following the import of necessary modules, you initialize a `ChatOpenAI` chat model. Then create a function called `count_characters_in_string` that computes the length of any given string. This function is encapsulated within a `Tool` object, providing a descriptive name and explanation for its role.    Subsequently, you utilize `create_react_agent` to initialize your agent, combining the defined `Tool`, the `ChatOpenAI` model, and a react prompt pulled from the LangChain hub. This sets up a comprehensive interactive agent.    With `AgentExecutor`, the agent is equipped with the tools and verbose output is enabled, allowing for detailed logging.    Finally, `agent_executor.invoke(...)` is executed with a query about the character count in “supercalifragilisticexpialidocious.” The agent utilizes the defined tool to calculate and return the precise character count in the word.    In Example 6-1, you can see that the agent decided to use the `Action` called `Characters` `in a text string` with an `Action Input`: `'supercalifragilisticexpialidocious'`. This pattern is extremely familiar to the simplistic ReAct implementation that you previously made.    ##### Example 6-1\. A single tool, agent output    ``` 进入新的 AgentExecutor 变更... 我应该计算单词 "supercalifragilisticexpiladocious" 中的字符数。动作：Count Characters in a text string 动作输入： "supercalifragilisticexpiladocious" 观察：34 思考：我现在知道最终答案了。最终答案：单词 "supercalifragilisticexpiladocious" 中有 34 个字符。 ```py    # Give Direction    Writing expressive names for your Python functions and tool descriptions will increase an LLM’s ability to effectively choose the right tools.```` ```py``  ````` ```py`# Using LLMs as an API (OpenAI Functions)    As mentioned in Chapter 4, OpenAI [released more fine-tuned LLMs](https://oreil.ly/hYTus) tailored toward function calling. This is important because it offers an alternative against the standard ReAct pattern for tool use. It’s similar to ReAct in that you’re still utilizing an LLM as a *reasoning engine.*    As shown in Figure 6-2, function calling allows an LLM to easily transform a user’s input into a weather API call.  ![Function calling flow using OpenAI functions](img/pega_0602.png)  ###### Figure 6-2\. Function calling flow using OpenAI functions    LangChain allows users to effortlessly switch between different agent types including ReAct, OpenAI functions, and many more.    Refer to Table 6-1 for a comprehensive comparison of the different agent types.      Table 6-1\. Comparison of agent types   | Agent type | Description | | --- | --- | | OpenAI Functions | Works with fine-tuned models like gpt-3.5-turbo-0613 and gpt-4-0613 for function calling. It intelligently outputs JSON objects for the function calls. Best for open source models and providers adopting this format. Note: deprecated in favor of OpenAI Tools. | | OpenAI Tools | Enhanced version for newer models, capable of invoking one or more functions. It intelligently outputs JSON objects for these function calls, optimizing the response efficiency and reducing response times in some architectures. | | XML Agent | Ideal for language models like Anthropic’s Claude, which excel in XML reasoning/writing. Best used with regular LLMs (not chat models) and unstructured tools accepting single string inputs. | | JSON Chat Agent | Tailored for language models skilled in JSON formatting. This agent uses JSON to format its outputs, supporting chat models for scenarios requiring JSON outputs. | | Structured Chat | Capable of using multi-input tools, this agent is designed for complex tasks requiring structured inputs and responses. | | ReAct | Implements ReAct logic, using tools like Tavily’s Search for interactions with a document store or search tools. | | Self-Ask with Search | Utilizes the Intermediate Answer tool for factual question resolution, following the self-ask with search methodology. Best for scenarios requiring quick and accurate factual answers. |    Let’s use prepackaged tools such as a `Calculator` to answer math questions using OpenAI function calling from the LangChain documentation:    ``` # 从 langchain 包中导入必要的模块和函数：from langchain.chains import (     LLMMathChain, ) from langchain import hub from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor from langchain_openai.chat_models import ChatOpenAI  # 使用温度设置为 0 初始化 ChatOpenAI：model = ChatOpenAI(temperature=0)  # 使用 ChatOpenAI 模型和工具创建 LLMMathChain 实例：llm_math_chain = LLMMathChain.from_llm(llm=model, verbose=True)  # 从 hub 下载提示：prompt = hub.pull("hwchase17/openai-functions-agent")  tools = [     Tool(         name="Calculator",         func=llm_math_chain.run, # 运行 LLMMathChain         description="当需要回答有关数学的问题时很有用",         return_direct=True,     ), ]  # 使用 ChatOpenAI 模型和工具创建代理：agent = create_openai_functions_agent(llm=model, tools=tools, prompt=prompt) agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  result = agent_executor.invoke({"input": "What is 5 + 5?"}) print(result) # {'input': 'What is 5 + 5?', 'output': 'Answer: 10'} ```py    After initiating the necessary libraries, you’ll use `ChatOpenAI`, setting the `temperature` parameter to 0 for deterministic outputs. By using `hub.pull("...")`, you can easily download prompts that have been saved on LangChainHub.    This model is then coupled with a tool named `Calculator` that leverages the capabilities of `LLMMathChain` to compute math queries. The OpenAI functions agent then decides to use the `Calculator` tool to compute `5 + 5` and returns `Answer: 10`.    Following on, you can equip an agent with multiple tools, enhancing its versatility. To test this, let’s add an extra `Tool` object to our agent that allows it to perform a fake Google search:    ``` def google_search(query: str) -> str:     return "James Phoenix is 31 years old."  # 代理可以使用的工具列表：tools = [     Tool(         # 用于数学计算的 LLMMathChain 工具。         func=llm_math_chain.run,         name="Calculator",         description="当需要回答有关数学的问题时很有用",     ),     Tool(         # 用于计算字符串中字符数的工具。         func=google_search,         name="google_search",         description="当需要查找某人的年龄时很有用",     ), ]   # 使用 ChatOpenAI 模型和工具创建代理：agent = create_openai_functions_agent(llm=model, tools=tools, prompt=prompt) agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # 请求代理运行任务并存储其结果：result = agent_executor.invoke(     {         "input": """Task: Google search for James Phoenix's age.  Then square it."""} ) print(result) # {'input': "...", 'output': 'James Phoenix is 31 years old. # Squaring his age, we get 961.'} ```py    When executed, the agent will first invoke the `google_search` function and then proceed to the `llm_math_chain.run` function. By mixing both custom and prepackaged tools, you significantly increase the flexibility of your agents.    ###### Note    Depending upon how many tools you provide, an LLM will either restrict or increase its ability to solve different user queries. Also, if you add too many tools, the LLM may become confused about what tools to use at every step while solving the problem.    Here are several recommended tools that you might want to explore:    [Google search](https://oreil.ly/TjrnF)      Enables an LLM to perform web searches, which provides timely and relevant context.      [File system tools](https://oreil.ly/5tAB0)      Essential for managing files, whether it involves reading, writing, or reorganizing them. Your LLM can interact with the file system more efficiently with them.      [Requests](https://oreil.ly/vZjm1)      A pragmatic tool that makes an LLM capable of executing HTTP requests for create, read, update, and delete (CRUD) functionality.      [Twilio](https://oreil.ly/ECS4r)      Enhance the functionality of your LLM by allowing it to send SMS messages or WhatsApp messages through Twilio.      # Divide Labor and Evaluate Quality    When using tools, make sure you divide the tasks appropriately. For example, entrust Twilio with communication services, while assigning requests for HTTP-related tasks. Additionally, it is crucial to consistently evaluate the performance and quality of the tasks performed by each tool.    Different tools may be called more or less frequently, which will influence your LLM agent’s performance. Monitoring tool usage will offer insights into your agent’s overall performance.    # Comparing OpenAI Functions and ReAct    Both OpenAI functions and the ReAct framework bring unique capabilities to the table for executing tasks with generative AI models. Understanding the differences between them can help you determine which is better suited for your specific use case.    OpenAI functions operate in a straightforward manner. In this setup, the LLM decides at runtime whether to execute a function. This is beneficial when integrated into a conversational agent, as it provides several features including:    Runtime decision making      The LLM autonomously makes the decision on whether a function(s) should be executed or not in real time.      Single tool execution      OpenAI functions are ideal for tasks requiring a single tool execution.      Ease of implementation      OpenAI functions can be easily merged with conversational agents.      Parallel function calling      For single task executions requiring multiple parses, OpenAI functions offer parallel function calling to invoke several functions within the same API request.      ## Use Cases for OpenAI Functions    If your task entails a definitive action such as a simple search or data extraction, OpenAI functions are an ideal choice.    ## ReAct    If you require executions involving multiple sequential tool usage and deeper introspection of previous actions, ReAct comes into play. Compared to function calling, ReAct is designed to go through many *thought loops* to accomplish a higher-level goal, making it suitable for queries with multiple intents.    Despite ReAct’s compatibility with `conversational-react` as an agent, it doesn’t yet offer the same level of stability as function calling and often favors toward using tools over simply responding with text. Nevertheless, if your task requires successive executions, ReAct’s ability to generate many thought loops and decide on a single tool at a time demonstrates several distinct features including:    Iterative thought process      ReAct allows agents to generate numerous thought loops for complex tasks.      Multi-intent handling      ReAct handles queries with multiple intents effectively, thus making it suitable for complex tasks.      Multiple tool execution      Ideal for tasks requiring multiple tool executions sequentially.      ## Use Cases for ReAct    If you’re working on a project that requires introspection of previous actions or uses multiple functions in succession such as saving an interview and then sending it in an email, ReAct is the best choice.    To aid decision making, see a comprehensive comparison in Table 6-2.      Table 6-2\. A feature comparison between OpenAI functions and ReAct   | Feature | OpenAI functions | ReAct | | --- | --- | --- | | Runtime decision making | ✓ | ✓ | | Single tool execution | ✓ | ✓ | | Ease of implementation | ✓ | x | | Parallel function calling | ✓ | x | | Iterative thought process | x | ✓ | | Multi-intent handling | ✓ | ✓ | | Sequential tool execution | x | ✓ | | Customizable prompt | ✓ | ✓ |    # Give Direction    When interacting with different AI frameworks, it’s crucial to understand that each framework has its strengths and trade-offs. Each framework will provide a unique form of direction to your LLM.    # Agent Toolkits    *[Agent toolkits](https://oreil.ly/_v6dm)* are a LangChain integration that provides multiple tools and chains together, allowing you to quickly automate tasks.    First, install some more packages by typing `**pip install langchain_experimental pandas tabulate langchain-community pymongo --upgrade**` on your terminal. Popular agent toolkits include:    *   CSV Agent           *   Gmail Toolkit           *   OpenAI Agent           *   Python Agent           *   JSON Agent           *   Pandas DataFrame Agent              The CSV Agent uses a Pandas DataFrame Agent and `python_repl_ast` tool to investigate a *.csv* file. You can ask it to quantify the data, identify column names, or create a correlation matrix.    Create a new Jupyter Notebook or Python file in *content/chapter_6* of the [shared repository](https://oreil.ly/x6FHn), then you will need to import `create_csv_agent`, `ChatOpenAI`, and `AgentType`. The `create_csv_agent` function requires an LLM, dataset `file path`, and `agent_type`:    ``` # 导入相关包：from langchain.agents.agent_types import AgentType from langchain_experimental.agents.agent_toolkits import create_csv_agent from langchain_openai.chat_models import ChatOpenAI  # 创建 CSV 代理：agent = create_csv_agent(     ChatOpenAI(temperature=0),     "data/heart_disease_uci.csv",     verbose=True,     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, )  agent.invoke("How many rows of data are in the file?") # '920'  agent.invoke("What are the columns within the dataset?") # "'id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', # 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'"  agent.invoke("Create a correlation matrix for the data and save it to a file.") # "The correlation matrix has been saved to a file named # 'correlation_matrix.csv'." ```py    It’s even possible for you to interact with a SQL database via a SQLDatabase agent:    ``` from langchain.agents import create_sql_agent from langchain_community.agent_toolkits import SQLDatabaseToolkit from langchain.sql_database import SQLDatabase from langchain.agents.agent_types import AgentType from langchain_openai.chat_models import ChatOpenAI  db = SQLDatabase.from_uri("sqlite:///./data/demo.db") toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))  # 创建代理执行器：agent_executor = create_sql_agent(     llm=ChatOpenAI(temperature=0),     toolkit=toolkit,     verbose=True,     agent_type=AgentType.OPENAI_FUNCTIONS, )  # 识别所有表：agent_executor.invoke("Identify all of the tables") # 'The database contains the following tables:\n1\. Orders\n2\. Products\n3\. Users' ```py    ``` user_sql = agent_executor.invoke(     '''Add 5 new users to the database. Their names are:  John, Mary, Peter, Paul, and Jane.''' ) '''Based on the schema of the "Users" table, I can see that the relevant columns for adding new users are "FirstName", "LastName", "Email", and "DateJoined". I will now run the SQL query to add the new users.\n\n```pysql\nINSERT INTO Users (FirstName, LastName, Email, DateJoined)\nVALUES (\'John\', \'Doe\', \'john.doe@email.com\', \'2023-05-01\'), \n(\'Mary\', \'Johnson\', \'mary.johnson@email.com\', \'2023-05-02\'),\n (\'Peter\', \'Smith\', \'peter.smith@email.com\', \'2023-05-03\'),\n (\'Paul\', \'Brown\', \'paul.brown@email.com\', \'2023-05-04\'),\n (\'Jane\', \'Davis\', \'jane.davis@email.com\', \'2023-05-05\');\n```\n\nPlease note that I have added the new users with the specified names and email addresses. The "DateJoined" column is set to the respective dates mentioned.''' ```py    First, the `agent_executor` inspects the SQL database to understand the database schema, and then the agent writes and executes a SQL statement that successfully adds five users into the SQL table.    # Customizing Standard Agents    It’s worth considering how to customize LangChain agents. Key function arguments can include the following:    *   `prefix` and `suffix` are the prompt templates that are inserted directly into the agent.           *   `max_iterations` and `max_execution_time` provide you with a way to limit API and compute costs in case an agent becomes stuck in an endless loop:              ``` # 这是用于演示的功能签名，不可执行。def create_sql_agent(     llm: BaseLanguageModel,     toolkit: SQLDatabaseToolkit,     agent_type: Any | None = None,     callback_manager: BaseCallbackManager | None = None,     prefix: str = SQL_PREFIX,     suffix: str | None = None,     format_instructions: str | None = None,     input_variables: List[str] | None = None,     top_k: int = 10,     max_iterations: int | None = 15,     max_execution_time: float | None = None,     early_stopping_method: str = "force",     verbose: bool = False,     agent_executor_kwargs: Dict[str, Any] | None = None,     extra_tools: Sequence[BaseTool] = (),     **kwargs: Any ) -> AgentExecutor ```py    Let’s update the previously created `agent_executor` so that the agent can perform more SQL statements. The `SQL_PREFIX` is directly inserted into the `create_sql_agent` function as the `prefix`. Additionally, you’ll insert the recommended `user_sql` from the previous agent that wouldn’t directly run `INSERT`, `UPDATE`, or `EDIT` commands; however, the new agent will happily execute CRUD (create, read, update, delete) operations against the SQLite database:    ``` SQL_PREFIX = """You are an agent designed to interact with a SQL database. Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies a specific number of examples they wish to obtain always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database. Never query for all the columns from a specific table, only ask for the relevant columns given the question. You have access to tools for interacting with the database. Only use the below tools. Only use the information returned by the below tools to construct your final answer. You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again. If the question does not seem related to the database, just return "I don't know" as the answer. """  agent_executor = create_sql_agent(     llm=ChatOpenAI(temperature=0),     toolkit=toolkit,     verbose=True,     agent_type=AgentType.OPENAI_FUNCTIONS,     prefix=SQL_PREFIX, )  agent_executor.invoke(user_sql) # '...sql\nINSERT INTO Users (FirstName, LastName, Email, # DateJoined)\nVALUES (...)...'  # 测试 Peter 是否已插入数据库：agent_executor.invoke("Do we have a Peter in the database?") '''Yes, we have a Peter in the database. Their details are as follows:\n- First Name: Peter...''' ```py    # Custom Agents in LCEL    It’s very easy to create a custom agent using LCEL; let’s create a chat model with one tool:    ``` from langchain_openai import ChatOpenAI from langchain_core.tools import tool  # 1\. 创建模型：llm = ChatOpenAI(temperature=0)  @tool def get_word_length(word: str) -> int:     """返回单词的长度"""     return len(word)  # 2\. 创建工具：tools = [get_word_length] ```py    Next, you’ll set up the prompt with a system message, user message, and a `MessagesPlaceholder`, which allows the agent to store its intermediate steps:    ``` from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 3\. 创建提示：prompt = ChatPromptTemplate.from_messages(     [         (             "system",             """You are a very powerful assistant, but don't know current events  and aren't good at calculating word length.""",         ),         ("user", "{input}"),         # 这是从代理那里写入/读取消息的地方         MessagesPlaceholder(variable_name="agent_scratchpad"),     ] ) ```py    Before creating an agent, you’ll need to bind the tools directly to the LLM for function calling:    ``` from langchain_core.utils.function_calling import convert_to_openai_tool from langchain.agents.format_scratchpad.openai_tools import (     format_to_openai_tool_messages, )  # 4\. 将 Python 函数工具格式化为 JSON 模式，并将其绑定到模型：llm_with_tools = llm.bind_tools(tools=[convert_to_openai_tool(t) for t in tools])  from langchain.agents.output_parsers.openai_tools \ import OpenAIToolsAgentOutputParser   # 5\. 设置代理链：agent = (     {         "input": lambda x: x["input"],         "agent_scratchpad": lambda x: format_to_openai_tool_messages(             x["intermediate_steps"]         ),     }     | prompt     | llm_with_tools     | OpenAIToolsAgentOutputParser() ) ```py    Here’s a step-by-step walk-through of the code:    1\. Importing tool conversion function      You begin by importing `convert_to_openai_tool`. This allows you to convert Python function tools into a JSON schema, making them compatible with OpenAI’s LLMs.      2\. Binding tools to your language model (LLM)      Next, you bind the tools to your LLM. By iterating over each tool in your `tools` list and converting them with `convert_to_openai_tool`, you effectively create `llm_with_tools`. This equips your LLM with the functionalities of the defined tools.      3\. Importing agent formatting and parsing functions      Here, you import `format_to_openai_tool_messages` and `OpenAIToolsAgentOutputParser`. These format the agent’s scratchpad and parse the output from your LLM bound with tools.      4\. Setting up your agent chain      In this final and crucial step, you set up the agent chain.    *   You take the lead by processing the user’s input directly.           *   You then strategically format intermediate steps into OpenAI function messages.           *   The `llm_with_tools` will then be called.           *   `OpenAIToolsAgentOutputParser` is used to parse the output.                Finally, let’s create and use the `AgentExecutor`:    ``` from langchain.agents import AgentExecutor  agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) agent_executor.invoke({"input": "How many letters in the word Software?"}) #{'input': 'How many letters in the word Software?', # 'output': 'There are 8 letters in the word "Software".'} ```py    The LCEL agent uses the `.invoke(...)` function and correctly identifies that there are eight letters within the word *software*.    # Understanding and Using Memory    When interacting with LLMs, understanding the role and importance of memory is paramount. It’s not just about how these models recall information but also about the strategic interplay between long-term (LTM) and short-term memory (STM).    ## Long-Term Memory    Think of long-term memory as the library of an LLM. It’s the vast, curated collection of data, storing everything from text to conceptual frameworks. This knowledge pool aids the model in comprehending and generating responses.    Applications include:    Vector databases      These databases can store unstructured text data, providing the model with a reference point when generating content. By indexing and categorizing this data, LLMs can swiftly retrieve relevant information via *similarity distance metrics*.      Self-reflection      Advanced applications include an LLM that introspects, records, and stores thoughts. Imagine an LLM that meticulously observes user patterns on a book review platform and catalogs these as deep insights. Over time, it pinpoints preferences, such as favored genres and writing styles. These insights are stored and accessed using retrieval. When users seek book recommendations, the LLM, *powered by the retrieved context*, provides bespoke suggestions aligned with their tastes.      Custom retrievers      Creating specific retrieval functions can significantly boost an LLM’s efficiency. Drawing parallels with human memory systems, these functions can prioritize data based on its relevance, the elapsed time since the last memory, and its utility in achieving a particular objective.      ## Short-Term Memory    Short-term memory in LLMs is akin to a temporary workspace. Here, recent interactions, active tasks, or ongoing conversations are kept at the forefront to ensure continuity and context.    Applications include:    Conversational histories      For chatbots, tracking conversational history is essential. It allows the bot to maintain context over multiple exchanges, preventing redundant queries and ensuring the conversation flows naturally.      Repetition avoidance      STM proves invaluable when similar or identical queries are posed by users. By referencing its short-term recall, the model can provide consistent answers or diversify its responses, based on the application’s requirement.      Having touched upon the foundational concepts of LTM and STM, let’s transition to practical applications, particularly in the realm of question-answer (QA) systems.    ## Short-Term Memory in QA Conversation Agents    Imagine Eva, a virtual customer support agent for an e-commerce platform. A user might have several interlinked queries:    *   User: “How long is the return policy for electronics?” *   Eva: “The return policy for electronics is 30 days.” *   User: “What about for clothing items?” *   Eva, leveraging STM: “For clothing items, it’s 45 days. Would you like to know about any other categories?”    Notice that by utilizing short term memory (STM), Eva seamlessly continues the conversation, anticipating potential follow-up questions. This fluidity is only possible due to the effective deployment of short-term memory, allowing the agent to perceive conversations not as isolated QAs but as a cohesive interaction.    For developers and prompt engineers, understanding and harnessing this can significantly elevate the user experience, fostering engagements that are meaningful, efficient, and humanlike.    # Memory in LangChain    LangChain provides easy techniques for adding memory to LLMs. As shown in Figure 6-3, every memory system in a chain is tasked with two fundamental operations: reading and storing.    It’s pivotal to understand that each chain has innate steps that demand particular inputs. While a user provides some of this data, the chain can also source other pieces of information from its memory.  ![Memory within LangChain](img/pega_0603.png)  ###### Figure 6-3\. Memory within LangChain    In every operation of the chain, there are two crucial interactions with its memory:    *   *After collecting the initial user data but before executing*, the chain retrieves information from its memory, adding to the user’s input.           *   *After the chain has completed but before returning the answer*, a chain will write the inputs and outputs of the current run to memory so that they can be referred to in future runs.              There are two pivotal choices you’ll need to make when creating a memory system:    *   The method of storing state           *   The approach to querying the memory state              ## Preserving the State    Beneath the surface, the foundational memory of generative AI models is structured as a sequence of chat messages. These messages can be stored in temporary in-memory lists or anchored in a more durable database. For those leaning toward long-term storage, there’s a wide range of [database integrations available](https://oreil.ly/ECD_n), streamlining the process and saving you from the hassle of manual integration.    With five to six lines of code, you can easily integrate a `MongoDBChatMessageHistory` that’s unique based on a `session_id` parameter:    ``` # 提供连接字符串以连接到 MongoDB 数据库。connection_string = "mongodb://mongo_user:password123@mongo:27017"  chat_message_history = MongoDBChatMessageHistory(     session_id="test_session",     connection_string=connection_string,     database_name="my_db",     collection_name="chat_histories", )  chat_message_history.add_user_message("I love programming!!") chat_message_history.add_ai_message("What do you like about it?")  chat_message_history.messages # [HumanMessage(content='I love programming!!', # AIMessage(content='What do you like about it?') ```py    ## Querying the State    A basic memory framework might merely relay the latest messages with every interaction. A slightly more nuanced setup might distill a crisp synopsis of the last set of messages. An even more advanced setup would discern specific entities from dialogue and relay only data about those entities highlighted in the ongoing session.    Different applications require varying demands on memory querying. LangChain’s memory toolkit will help you to create simplistic memory infrastructures while empowering you to architect bespoke systems when necessary.    ## ConversationBufferMemory    There are various types of memory within LangChain, and one of the most popular is ConversationBufferMemory. This allows you to store multiple chat messages with no restriction on chat history size.    Start by importing `ConversationBufferMemory`, and you can then add context with the `save_context` function. The `load_memory_variables` function returns a Python dictionary containing the `Human` and `AI` messages:    ``` from langchain.memory import ConversationBufferMemory memory = ConversationBufferMemory() memory.save_context({"input": "hi"}, {"output": "whats up"}) memory.load_memory_variables({}) # {'history': 'Human: hi\nAI: whats up'} ```py    You can also return the LangChain schema messages, i.e., `SystemMessage`, `AIMessage` or `HumanMessage`, by adding `return_messages=True` to `ConversationBufferMemory`:    ``` memory = ConversationBufferMemory(return_messages=True) memory.save_context({"input": "hi"}, {"output": "whats up"}) memory.load_memory_variables({}) # {'history': [HumanMessage(content='hi'), # AIMessage(content='whats up')]} ```py    Let’s add memory directly to a chain in LCEL:    ``` # 在链中使用：from langchain.memory import ConversationBufferMemory from langchain_openai.chat_models import ChatOpenAI from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder from langchain_core.output_parsers import StrOutputParser from langchain_core.runnables import RunnableLambda from operator import itemgetter  memory = ConversationBufferMemory(return_messages=True)  model = ChatOpenAI(temperature=0) prompt = ChatPromptTemplate.from_messages(     [         ("system", "Act as a chatbot that helps users with their queries."),         # 对话历史         MessagesPlaceholder(variable_name="history"),         ("human", "{input}"),     ] ) chain = (     {         "input": lambda x: x["input"],         "history": RunnableLambda(memory.load_memory_variables) | \         itemgetter("history"),     }     | prompt     | model     | StrOutputParser() ) ```py    Notice the `MessagesPlaceholder` has a `variable_name` of `"history"`. This is aligned with the `memory` key within `ConversationBufferMemory`, allowing the previous chat history to be directly formatted into the `ChatPromptTemplate`.    After setting up the LCEL chain, let’s invoke it and save the messages to the `memory` variable:    ``` inputs = {"input": "Hi my name is James!"} result = chain.invoke(inputs) memory.save_context(inputs, {"outputs": result}) print(memory.load_memory_variables({}))  # {'history': [HumanMessage(content='Hi my name is James!'), # AIMessage
