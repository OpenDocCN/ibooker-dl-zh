# 第六章\. 代理架构

建立在第五章中描述的架构之上，本章将涵盖所有当前 LLM 架构中可能最重要的架构——代理架构。首先，我们介绍使 LLM 代理独特之处，然后展示如何构建它们以及如何扩展它们以适应常见用例。

在人工智能领域，创建（智能）代理的历史悠久，正如斯图尔特·罗素和彼得·诺维格在他们所著的《人工智能》（Pearson, 2020）一书中所说：“某种能够行动的东西”可以最简单地定义代理。实际上，“行动”这个词的含义比表面上看要丰富一些：

+   行动需要一些决定做什么的能力。

+   决定做什么意味着可以访问多个可能的行动方案。毕竟，没有选择的决定根本不是决定。

+   为了做出决定，代理还需要获取有关外部环境（代理本身之外的一切）的信息。

因此，一个具有代理性的 LLM 应用必须是一个使用 LLM 从一种或多种可能的行动方案中选择，给定关于世界当前状态或所需下一个状态的一些上下文的应用。这些属性通常通过混合我们在前言中首次遇到的两种提示技术来实现：

工具调用

在你的提示中包含一个外部函数列表，LLM 可以利用这些函数（即它可以决定采取的行动），并提供如何格式化其输出中选择的说明。你很快就会看到提示中的样子。

思维链

研究人员发现，当 LLM 被指示通过将复杂问题分解为一系列按顺序执行的粒度步骤来进行推理时，它们“做出更好的决定”。这通常是通过添加类似“逐步思考”的指令或包括问题和它们分解为几个步骤/行动的例子来完成的。

这里有一个同时使用工具调用和思维链的示例提示：

```py
Tools:
search: this tool accepts a web search query and returns the top results.
calculator: this tool accepts math expressions and returns their result.

If you want to use tools to arrive at the answer, output the list of tools and
inputs in CSV format, with the header row: tool,input.

Think step by step; if you need to make multiple tool calls to arrive at the
answer, return only the first one.

How old was the 30th president of the United States when he died?

tool,input
```

当在温度 0（以确保 LLM 遵循所需的输出格式，CSV）和换行符作为停止序列（这指示 LLM 在达到此字符时停止产生输出）的情况下运行`gpt-3.5-turbo`时，输出，这使得 LLM 产生一个单一的行动（正如预期的那样，因为提示要求这样做）：

```py
search,30th president of the United States
```

最新的 LLM 和聊天模型已经经过微调，以提高它们在工具调用和思维链应用中的性能，从而消除了在提示中添加特定指令的需要：

```py
add example prompt and output for tool-calling model
```

# 计划-执行循环

使代理架构与第五章中讨论的架构不同的概念是我们尚未涉及的概念：由 LLM 驱动的循环。

每个程序员在他们的代码中都遇到过循环。通过“循环”，我们指的是运行相同的代码多次，直到达到停止条件。代理架构的关键是让 LLM 控制停止条件——也就是说，决定何时停止循环。

在这个循环中我们将运行以下内容的某种变体：

+   规划动作或动作

+   执行所述动作（们）

继续前一小节的例子，我们将运行输入为“美国第 30 任总统”的`search`工具，它产生以下输出：

```py
Calvin Coolidge (born John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; July 4, 1872 – January 
5, 1933) was an American attorney and politician who served as the 30th president 
of the United States from 1923 to 1929\. John Calvin Coolidge Jr.
```

然后我们将重新运行提示，增加一点：

```py
Tools:
search: this tool accepts a web search query and returns the top results.
calculator: this tool accepts math expressions and returns their result.
output: this tool ends the interaction. Use it when you have the final answer.

If you want to use tools to arrive at the answer, output the list of tools and 
inputs in CSV format, with this header row: tool,input

Think step by step; if you need to make multiple tool calls to arrive at 
the answer, return only the first one.

How old was the 30th president of the United States when he died?

tool,input

search,30th president of the United States

search: Calvin Coolidge (born John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; July 4, 1872 – 
January 5, 1933) was an American attorney and politician who served as the 30th 
president of the United States from 1923 to 1929\. John Calvin Coolidge Jr.

tool,input
```

以及输出：

```py
calculator,1933 - 1872
```

注意我们添加了两件事：

+   一个“输出”工具——当 LLM 找到最终答案时应该使用，我们也会用它作为停止循环的信号。

+   上一次迭代中工具调用的结果，简单地以工具名称及其（文本）输出。这是为了允许 LLM 继续进行下一步交互。换句话说，我们在告诉 LLM，“嘿，我们得到了你要求的结果，你接下来想做什么？”

让我们继续进行第三次迭代：

```py
Tools:
search: this tool accepts a web search query and returns the top results.
calculator: this tool accepts math expressions and returns their result.

If you want to use tools to arrive at the answer, output the list of tools and 
inputs in CSV format, with this header row: tool,input.
output: this tool ends the interaction. Use it when you have the final answer.

Think step by step; if you need to make multiple tool calls to arrive at 
the answer, return only the first one.

How old was the 30th president of the United States when he died?

tool,input

search,30th president of the United States

search: Calvin Coolidge (born John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; July 4, 1872 – 
January 5, 1933) was an American attorney and politician who served as the 30th 
president of the United States from 1923 to 1929\. John Calvin Coolidge Jr.
tool,input

calculator,1933-1872

calculator: 61

tool, input
```

以及输出：

```py
output, 61
```

使用“计算器”工具的结果，LLM 现在拥有了足够的信息来提供最终答案，因此它选择了“输出”工具，并选择了“61”作为最终答案。

这就是使代理架构如此有用的原因——LLM 被赋予了决策权。下一步是得出答案并决定采取多少步骤——也就是说，何时停止。

这个架构，称为[ReAct](https://oreil.ly/M7hF-)，最初由姚顺宇等人提出。本章的其余部分将探讨如何通过第五章中电子邮件助手示例来提高代理架构的性能。

但首先，让我们看看使用聊天模型和 LangGraph 实现基本代理架构的样子。

# 构建 LangGraph 代理

对于这个例子，我们需要为所选的搜索工具 DuckDuckGo 安装额外的依赖项。对于 Python 的安装：

*Python*

```py
pip install duckduckgo-search
```

对于 JS，我们还需要为计算器工具安装一个依赖项：

*JavaScript*

```py
npm i duck-duck-scrape expr-eval
```

完成这些后，让我们来看看实现代理架构的实际代码：

*Python*

```py
import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)

search = DuckDuckGoSearchRun()
tools = [search, calculator]
model = ChatOpenAI(temperature=0.1).bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def model_node(state: State) -> State:
    res = model.invoke(state["messages"])
    return {"messages": res}

builder = StateGraph(State)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()
```

*JavaScript*

```py
import {
  DuckDuckGoSearch
} from "@langchain/community/tools/duckduckgo_search";
import {
  Calculator
} from "@langchain/community/tools/calculator";
import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
} from "@langchain/langgraph";
import {
  ToolNode,
  toolsCondition
} from "@langchain/langgraph/prebuilt";

const search = new DuckDuckGoSearch();
const calculator = new Calculator();
const tools = [search, calculator];
const model = new ChatOpenAI({
  temperature: 0.1
}).bindTools(tools);

const annotation = Annotation.Root({
  messages: Annotation({
    reducer: messagesStateReducer,
    default: () => []
  }),
});

async function modelNode(state) {
  const res = await model.invoke(state.messages);
  return { messages: res };
}

const builder = new StateGraph(annotation)
  .addNode("model", modelNode)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "model")
  .addConditionalEdges("model", toolsCondition)
  .addEdge("tools", "model");

const graph = builder.compile();
```

视觉表示如图图 6-1 所示。

![模型图  描述自动生成](img/lelc_0601.png)

###### 图 6-1. 代理架构

这里有几个需要注意的地方：

+   在这个例子中，我们使用了两个工具：一个搜索工具和一个计算器工具，但你可以很容易地添加更多或替换我们使用的工具。在 Python 示例中，你还可以看到一个创建自定义工具的例子。

+   我们使用了 LangGraph 附带的两个便利函数。`ToolNode`作为我们图中的一个节点；它执行在状态中找到的最新 AI 消息中请求的工具调用，并返回每个工具的结果的`ToolMessage`。`ToolNode`还处理由工具引发的异常——使用错误消息构建一个`ToolMessage`，然后将其传递给 LLM，LLM 可能会决定如何处理该错误。

+   `tools_condition`作为条件边函数，查看状态中的最新 AI 消息，如果有任何工具要执行，则路由到`tools`节点。否则，结束图。

+   最后，请注意，这个图在模型和工具节点之间循环。也就是说，模型本身负责决定何时结束计算，这是代理架构的关键属性。每当我们在 LangGraph 中编写循环时，我们可能会想使用条件边，因为这允许你定义当图应该退出循环并停止执行时的*停止条件*。

现在让我们看看它在前面的例子中的表现：

*Python*

```py
input = {
    "messages": [
        HumanMessage("""How old was the 30th president of the United States 
 when he died?""")
    ]
}
for c in graph.stream(input):
    print(c)
```

*JavaScript*

```py
const input = {
  messages: [
    HumanMessage(`How old was the 30th president of the United States when he 
 died?`)
  ]
}
for await (const c of await graph.stream(input)) {
  console.log(c)
}
```

*输出：*

```py
{
    "model": {
        "messages": AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "duckduckgo_search",
                    "args": {
                        "query": "30th president of the United States age at 
                            death"
                    },
                    "id": "call_ZWRbPmjvo0fYkwyo4HCYUsar",
                    "type": "tool_call",
                }
            ],
        )
    }
}
{
    "tools": {
        "messages": [
            ToolMessage(
                content="Calvin Coolidge (born July 4, 1872, Plymouth, Vermont, 
                    U.S.—died January 5, 1933, Northampton, Massachusetts) was 
                    the 30th president of the United States (1923-29). Coolidge 
                    acceded to the presidency after the death in office of 
                    Warren G. Harding, just as the Harding scandals were coming 
                    to light....",
                name="duckduckgo_search",
                tool_call_id="call_ZWRbPmjvo0fYkwyo4HCYUsar",
            )
        ]
    }
}
{
    "model": {
        "messages": AIMessage(
            content="Calvin Coolidge, the 30th president of the United States, 
                died on January 5, 1933, at the age of 60.",
        )
    }
}
```

遍历这个输出：

1.  首先，`model`节点执行并决定调用`duckduckgo_search`工具，这导致条件边将我们路由到后面的`tools`节点。

1.  `ToolNode`执行了搜索工具，并打印出了上面的搜索结果，实际上包含答案“年龄和逝世年份。1933 年 1 月 5 日（60 岁）”。

1.  再次调用了`model`工具，这次使用搜索结果作为最新消息，并生成了最终答案（没有更多的工具调用）；因此，条件边结束了图。

接下来，让我们看看对这个基本代理架构的一些有用扩展，定制计划和工具调用。

# 总是首先调用工具

在标准代理架构中，LLM 总是被要求决定下一个调用哪个工具。这种安排有一个明显的优势：它给了 LLM 最大的灵活性，以适应每个到来的用户查询的行为。但这种灵活性是有代价的：不可预测性。例如，如果你，作为应用程序的开发者，知道应该总是首先调用搜索工具，这实际上对你的应用程序是有益的：

1.  它将减少整体延迟，因为它将跳过第一个 LLM 调用，该调用会生成调用搜索工具的请求。

1.  这将防止 LLM 错误地决定对于某些用户查询不需要调用搜索工具。

另一方面，如果你的应用程序没有明确的规则“你应该总是首先调用这个工具”，引入这种约束实际上会使你的应用程序变得更差。

让我们看看这样做是什么样子：

*Python*

```py
import ast
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)

search = DuckDuckGoSearchRun()
tools = [search, calculator]
model = ChatOpenAI(temperature=0.1).bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def model_node(state: State) -> State:
    res = model.invoke(state["messages"])
    return {"messages": res}

def first_model(state: State) -> State:
    query = state["messages"][-1].content
    search_tool_call = ToolCall(
        name="duckduckgo_search", args={"query": query}, id=uuid4().hex
    )
    return {"messages": AIMessage(content="", tool_calls=[search_tool_call])}

builder = StateGraph(State)
builder.add_node("first_model", first_model)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "first_model")
builder.add_edge("first_model", "tools")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()
```

*JavaScript*

```py
import {
  DuckDuckGoSearch
} from "@langchain/community/tools/duckduckgo_search";
import {
  Calculator
} from "@langchain/community/tools/calculator";
import {
  AIMessage,
} from "@langchain/core/messages";
import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
} from "@langchain/langgraph";
import {
  ToolNode,
  toolsCondition
} from "@langchain/langgraph/prebuilt";

const search = new DuckDuckGoSearch();
const calculator = new Calculator();
const tools = [search, calculator];
const model = new ChatOpenAI({ temperature: 0.1 }).bindTools(tools);

const annotation = Annotation.Root({
  messages: Annotation({ reducer: messagesStateReducer, default: () => [] }),
});

async function firstModelNode(state) {
  const query = state.messages[state.messages.length - 1].content;
  const searchToolCall = {
    name: "duckduckgo_search",
    args: { query },
    id: Math.random().toString(),
  };
  return {
    messages: [new AIMessage({ content: "", tool_calls: [searchToolCall] })],
  };
}

async function modelNode(state) {
  const res = await model.invoke(state.messages);
  return { messages: res };
}

const builder = new StateGraph(annotation)
  .addNode("first_model", firstModelNode)
  .addNode("model", modelNode)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "first_model")
  .addEdge("first_model", "tools")
  .addEdge("tools", "model")
  .addConditionalEdges("model", toolsCondition);

const graph = builder.compile();
```

可视表示如图图 6-2 所示。

![一个模型的图  自动生成的描述](img/lelc_0602.png)

###### 图 6-2\. 修改代理架构以始终首先调用特定工具

注意与上一节相比的不同之处：

+   现在，我们开始所有调用，首先调用`first_model`，它根本不调用 LLM。它只是使用用户的原始消息作为搜索查询为搜索工具创建一个工具调用。之前的架构会让 LLM 生成这个工具调用（或它认为更好的其他响应）。

+   之后，我们继续到`tools`，这与之前的例子相同，然后我们继续到之前的`agent`节点。

现在，让我们看看一些示例输出，与之前的查询相同：

*Python*

```py
input = {
    "messages": [
        HumanMessage("""How old was the 30th president of the United States 
 when he died?""")
    ]
}
for c in graph.stream(input):
print(c)
```

*JavaScript*

```py
const input = {
  messages: [
    HumanMessage(`How old was the 30th president of the United States when he 
 died?`)
  ]
}
for await (const c of await graph.stream(input)) {
  console.log(c)
}
```

*输出：*

```py
{
    "first_model": {
        "messages": AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "duckduckgo_search",
                    "args": {
                        "query": "How old was the 30th president of the United 
                            States when he died?"
                    },
                    "id": "9ed4328dcdea4904b1b54487e343a373",
                    "type": "tool_call",
                }
            ],
        )
    }
}
{
    "tools": {
        "messages": [
            ToolMessage(
                content="Calvin Coolidge (born July 4, 1872, Plymouth, Vermont, 
                    U.S.—died January 5, 1933, Northampton, Massachusetts) was 
                    the 30th president of the United States (1923-29). Coolidge 
                    acceded to the presidency after the death in office of 
                    Warren G. Harding, just as the Harding scandals were coming 
                    to light....",
                name="duckduckgo_search",
                tool_call_id="9ed4328dcdea4904b1b54487e343a373",
            )
        ]
    }
}
{
    "model": {
        "messages": AIMessage(
            content="Calvin Coolidge, the 30th president of the United States, 
                was born on July 4, 1872, and died on January 5, 1933\. To 
                calculate his age at the time of his death, we can subtract his 
                birth year from his death year. \n\nAge at death = Death year - 
                Birth year\nAge at death = 1933 - 1872\nAge at death = 61 
                years\n\nCalvin Coolidge was 61 years old when he died.",
        )
    }
}
```

这次，我们跳过了最初的 LLM 调用。我们首先到达`first_model`节点，该节点直接返回搜索工具的工具调用。从那里，我们继续到之前的流程——即执行搜索工具，最后回到`model`节点生成最终答案。

接下来让我们看看当你有很多工具想要提供给 LLM 时你可以做什么。

# 处理许多工具

LLM 远非完美，当给出多个选择或提示中的过多信息时，它们目前遇到的困难更大。这些限制也扩展到下一步行动的计划。当给出许多工具（比如说，超过 10 个）时，规划性能（即 LLM 选择正确工具的能力）开始下降。解决这个问题的方法是减少 LLM 可以选择的工具数量。但如果你确实有很多工具想要用于不同的用户查询呢？

一个优雅的解决方案是使用 RAG 步骤预先选择当前查询中最相关的工具，然后只向 LLM 提供这些工具子集而不是整个工具库。这也可以帮助减少调用 LLM 的成本（商业 LLM 通常根据提示和输出的长度收费）。另一方面，这个 RAG 步骤会给你的应用程序引入额外的延迟，因此只有在你在添加更多工具后看到性能下降时才应该采取这一步骤。

让我们看看如何做到这一点：

*Python*

```py
import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)

search = DuckDuckGoSearchRun()
tools = [search, calculator]

embeddings = OpenAIEmbeddings()
model = ChatOpenAI(temperature=0.1)

tools_retriever = InMemoryVectorStore.from_documents(
    [Document(tool.description, metadata={"name": tool.name}) for tool in tools],
    embeddings,
).as_retriever()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]

def model_node(state: State) -> State:
    selected_tools = [
        tool for tool in tools if tool.name in state["selected_tools"]
    ]
    res = model.bind_tools(selected_tools).invoke(state["messages"])
    return {"messages": res}

def select_tools(state: State) -> State:
    query = state["messages"][-1].content
    tool_docs = tools_retriever.invoke(query)
    return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}

builder = StateGraph(State)
builder.add_node("select_tools", select_tools)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "select_tools")
builder.add_edge("select_tools", "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()
```

*JavaScript*

```py
import { DuckDuckGoSearch } from "@langchain/community/tools/duckduckgo_search";
import { Calculator } from "@langchain/community/tools/calculator";
import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import {
  StateGraph,
  Annotation,
  messagesStateReducer,
  START,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { HumanMessage } from "@langchain/core/messages";

const search = new DuckDuckGoSearch();
const calculator = new Calculator();
const tools = [search, calculator];

const embeddings = new OpenAIEmbeddings();
const model = new ChatOpenAI({ temperature: 0.1 });

const toolsStore = await MemoryVectorStore.fromDocuments(
  tools.map(
    (tool) =>
      new Document({
        pageContent: tool.description,
        metadata: { name: tool.constructor.name },
      })
  ),
  embeddings
);
const toolsRetriever = toolsStore.asRetriever();

const annotation = Annotation.Root({
  messages: Annotation({ reducer: messagesStateReducer, default: () => [] }),
  selected_tools: Annotation(),
});

async function modelNode(state) {
  const selectedTools = tools.filter((tool) =>
    state.selected_tools.includes(tool.constructor.name)
  );
  const res = await model.bindTools(selectedTools).invoke(state.messages);
  return { messages: res };
}

async function selectTools(state) {
  const query = state.messages[state.messages.length - 1].content;
  const toolDocs = await toolsRetriever.invoke(query as string);
  return {
    selected_tools: toolDocs.map((doc) => doc.metadata.name),
  };
}

const builder = new StateGraph(annotation)
  .addNode("select_tools", selectTools)
  .addNode("model", modelNode)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "select_tools")
  .addEdge("select_tools", "model")
  .addConditionalEdges("model", toolsCondition)
  .addEdge("tools", "model");
```

你可以在图 6-3 中看到视觉表示。

![软件开发流程图  自动生成的描述](img/lelc_0603.png)

###### 图 6-3\. 修改代理架构以处理许多工具

###### 注意

这非常类似于常规代理架构。唯一的区别是我们进入实际的代理循环之前会经过`select_tools`节点。之后，它就像我们之前看到的常规代理架构一样工作。

现在，让我们看看之前查询的示例输出：

*Python*

```py
input = {
  "messages": [
    HumanMessage("""How old was the 30th president of the United States when 
 he died?""")
  ]
}
for c in graph.stream(input):
print(c)
```

*JavaScript*

```py
const input = {
  messages: [
    HumanMessage(`How old was the 30th president of the United States when he 
 died?`)
  ]
}
for await (const c of await graph.stream(input)) {
  console.log(c)
}
```

*输出：*

```py
{
    "select_tools": {
        "selected_tools': ['duckduckgo_search', 'calculator']
    }
}
{
    "model": {
        "messages": AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "duckduckgo_search",
                    "args": {
                        "query": "30th president of the United States"
                    },
                    "id": "9ed4328dcdea4904b1b54487e343a373",
                    "type": "tool_call",
                }
            ],
        )
    }
}
{
    "tools": {
        "messages": [
            ToolMessage(
                content="Calvin Coolidge (born July 4, 1872, Plymouth, Vermont, 
                    U.S.—died January 5, 1933, Northampton, Massachusetts) was 
                    the 30th president of the United States (1923-29). Coolidge 
                    acceded to the presidency after the death in office of 
                    Warren G. Harding, just as the Harding scandals were coming 
                    to light....",
                name="duckduckgo_search",
                tool_call_id="9ed4328dcdea4904b1b54487e343a373",
            )
        ]
    }
}
{
    "model": {
        "messages": AIMessage(
            content="Calvin Coolidge, the 30th president of the United States, 
                was born on July 4, 1872, and died on January 5, 1933\. To 
                calculate his age at the time of his death, we can subtract his 
                birth year from his death year. \n\nAge at death = Death year - 
                Birth year\nAge at death = 1933 - 1872\nAge at death = 61 
                years\n\nCalvin Coolidge was 61 years old when he died.",
        )
    }
}
```

注意我们首先做了什么：查询检索器以获取当前用户查询的最相关工具。然后，我们继续到常规代理架构。

# 摘要

本章介绍了*代理*的概念，并讨论了使 LLM 应用*具有代理性*所需的因素：通过使用外部信息，使 LLM 能够决定在多个选项之间进行选择。

我们探讨了使用 LangGraph 构建的标准代理架构，并查看了两项有用的扩展：如何始终首先调用特定工具以及如何处理多个工具。

第七章探讨了代理架构的额外扩展。
