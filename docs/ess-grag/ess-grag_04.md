# 第五章：代理 RAG

### 本章涵盖

+   代理 RAG 是什么

+   我们为什么需要代理 RAG

+   如何实现代理 RAG

在前面的章节中，我们看到了如何使用不同的向量相似度搜索方法来查找相关数据。使用相似度搜索，我们可以在非结构化数据源中找到相关数据，但具有结构的数据往往比非结构化数据更有价值，因为结构本身包含信息。

向数据添加结构可以是一个逐步的过程。我们可以从一个简单的结构开始，然后随着进行添加更复杂的结构。我们在上一章中看到了这一点，我们从一个简单的图数据开始，然后向其中添加了更复杂的结构。

代理 RAG 系统（见图 5.1）是一个提供多种检索代理的系统，这些代理可以检索回答用户问题所需的数据。代理 RAG 系统的起始界面通常是一个检索路由器，其任务是找到最适合执行当前任务的检索器（或检索器）。

实现代理 RAG 系统的一种常见方式是利用 LLM 使用工具的能力（有时称为 *函数调用*）。并非所有 LLM 都具备这种能力，但 OpenAI 的 GPT-3.5 和 GPT-4 就有，这就是我们在本章中要使用的。这可以通过大多数 LLM 使用 ReAct 方法（见 [`arxiv.org/abs/2210.03629`](https://arxiv.org/abs/2210.03629)）来实现，但随着时间的推移，目前的趋势是这一功能将适用于所有 LLM。

![figure](img/5-1.png)

##### 图 5.1 使用代理 RAG 的应用程序数据流

## 5.1 什么是代理 RAG？

代理系统在复杂性和复杂性方面各不相同，但核心思想是系统能够代表用户执行任务。在本章中，我们将探讨一个基本的代理系统，其中系统只需选择使用哪个检索器，并决定找到的上下文是否回答了问题。在更高级的系统中，系统可能会制定计划执行什么类型的任务来解决当前任务。从本章的基本内容开始是一个理解代理系统核心概念的好方法，对于 RAG 任务，这通常就是你所需要的。

代理 RAG 是一个系统，其中提供了多种检索代理来检索回答用户问题所需的数据。成功的代理 RAG 系统需要几个基础部分：

+   *检索路由器* — 一个接收用户问题并返回最佳检索器（或检索器）的功能

+   *检索代理* — 实际上可以用来检索回答用户问题所需数据的检索器

+   *答案评论员* — 一个接收检索器答案并检查原始问题是否得到正确回答的功能

### 5.1.1 检索代理

检索代理是实际用于检索回答用户问题所需数据的检索器。这些检索器可以是非常广泛的，例如向量相似度搜索，或者非常具体的，例如一个硬编码的数据库查询模板，它接收参数，例如第 5.1.2 节中提到的检索路由器。

在大多数代理 RAG（检索增强生成）系统中，一些通用的检索代理是相关的，如向量相似度搜索和 text2cypher。前者适用于非结构化数据源，后者适用于图数据库中的结构化数据，但在现实世界的生产系统中，要使任何这些达到用户期望的水平并不简单。

正因如此，我们需要专门的检索器，它们非常狭窄但执行得非常好。随着我们识别出通用检索器在生成查询以回答问题时遇到的问题，这些专门的检索器可以随着时间的推移而构建。

### 5.1.2 检索路由器

为了选择适合工作的正确检索器，我们有一个叫做检索路由器的机制。检索路由器是一个函数，它接收用户问题并返回最佳检索器。路由器如何做出这个决定可能有所不同，但通常使用 LLM 来做出这个决定。

假设我们有一个类似“法国的首都是什么？”的问题，并且我们编写了两个可用的检索代理（这两个代理都从数据库中检索答案）：

+   `capital_by_country`——一个接收国家名称并返回该国家首都的检索器

+   `country_by_capital`——一个接收首都名称并返回该首都所在国家的检索器

这两个检索器都可以是硬编码的数据库查询，接收国家或首都作为参数。

检索路由器可以是一个 LLM（大型语言模型），它接收用户问题并返回最佳检索器。在这种情况下，LLM 可以返回带有`"France"`（法国）作为提取参数的`capital_by_country`检索器。因此，实际调用检索器的代码将是`capital_by_country("France")`。

这是一个简单的例子，但在现实世界的场景中，可能会有许多检索器可用。检索路由器可能是一个复杂的函数，它使用 LLM 来选择最适合工作的最佳检索器。

### 5.1.3 回答批评者

回答批评者是一个函数，它接收检索器的答案并检查原始问题是否被正确回答。回答批评者是一个阻塞函数，如果答案不正确或不完整，它可以阻止答案返回给用户。

如果一个不完整或不正确的答案被阻止，答案批评者应生成一个新问题，该问题可用于检索正确答案，并进入另一轮检索正确答案。可能的情况是正确答案在数据源中不可用，因此需要从这个循环中设置一些退出标准；答案批评者应能够处理这种情况，并在这种情况下向用户返回消息，告知答案不可用。

## 5.2 为什么我们需要代理 RAG？

代理 RAG 有用的一个领域是我们有多种数据源，并且希望为工作使用最佳数据源。另一个常见用途是当数据源非常广泛或复杂，我们需要专门的检索器来一致地检索所需数据时。

如本书前面所述，通用的检索器，如向量相似度搜索，可以在非结构化数据源中找到相关数据。当我们有如图数据库这样的结构化数据源时，我们可能会使用在第四章中介绍的通用检索器，如 text2cypher。如果数据非常复杂，像 text2cypher 这样的工具在生成正确查询时可能会遇到问题。在这种情况下，可以使用专门的检索器来检索正确数据。例如，这可能是一个窄范围的 text2cypher 检索器或一个硬编码的数据库查询，该查询接受参数。

随着时间的推移，我们可以识别出像 text2cypher 这样的工具在生成查询以回答问题时遇到的问题，并为这些问题构建专门的检索器，并将 text2cypher 作为没有良好特定检索器匹配情况下的通用检索器使用。

这就是代理 RAG 可以发挥作用的地方。有多种检索器可供选择，我们需要为工作选择最佳的检索器，并在将其返回给用户之前评估答案。在生产环境中，这非常有用，可以保持系统性能高和答案质量一致。

## 5.3 如何实现代理 RAG

在本节中，我们将介绍如何实现代理 RAG 系统的基本部分。您可以直接在附带的 Jupyter 笔记本中跟随实现，笔记本地址如下：[`github.com/tomasonjo/kg-rag/blob/main/notebooks/ch05.ipynb`](https://github.com/tomasonjo/kg-rag/blob/main/notebooks/ch05.ipynb)。

注意：在本章的实现中，我们使用我们所说的“电影数据集”。有关数据集的更多信息以及各种加载方式，请参阅附录。

### 5.3.1 实现检索工具

在我们可以将用户输入路由到由正确检索器（s）处理之前，我们需要让检索器可供路由器选择。检索器可以是广泛的，如向量相似度搜索，也可以是非常具体的，如接受参数的硬编码数据库查询模板。

在这个实际示例中，我们将使用一个简单的检索器列表：两个使用 Cypher 模板通过标题和演员名称获取电影，以及一个使用 text2cypher 处理所有其他问题的检索器。如前所述，有用的检索器集合因系统而异，应根据需要随时间添加以提高应用程序的性能。

##### 列表 5.1 可用的检索器工具

```py
text2cypher_description = {
    "type": "function",
    "function": {
        "name": "text2cypher",
        "description": "Query the database with a user question. When other tools don't fit, fallback to use this one.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user question to find the answer for",
                }
            },
            "required": ["question"],
        },
    },
}

def text2cypher(question: str):
    """Query the database with a user question."""
    t2c = Text2Cypher(neo4j_driver)
    t2c.set_prompt_section("question", question)
    cypher = t2c.generate_cypher()
    records, _, _ = neo4j_driver.execute_query(cypher)
    return [record.data() for record in records]

movie_info_by_title_description = {
    "type": "function",
    "function": {
        "name": "movie_info_by_title",
        "description": "Get information about a movie by providing the title",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The movie title",
                }
            },
            "required": ["title"],
        },
    },
}

def movie_info_by_title(title: str):
    """Return movie information by title."""
    query = """
    MATCH (m:Movie)
    WHERE toLower(m.title) CONTAINS $title
    OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
    OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
    RETURN m AS movie, collect(a.name) AS cast, collect(d.name) AS directors
    """
    records, _, _ = neo4j_driver.execute_query(query, title=title.lower())
    return [record.data() for record in records]

movies_info_by_actor_description = {
    "type": "function",
    "function": {
        "name": "movies_info_by_actor",
        "description": "Get information about a movie by providing an actor",
        "parameters": {
            "type": "object",
            "properties": {
                "actor": {
                    "type": "string",
                    "description": "The actor name",
                }
            },
            "required": ["actor"],
        },
    },
}

def movies_info_by_actor(actor: str):
    """Return movie information by actor."""
    query = """
    MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
    OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
    OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
    WHERE toLower(a.name) CONTAINS $actor
    RETURN m AS movie, collect(a.name) AS cast, collect(d.name) AS directors
    """
    records, _, _ = neo4j_driver.execute_query(query, actor=actor.lower())
    return [record.data() for record in records]
```

注意`neo4j_driver`和`text2cypher`是可以在本书的代码仓库中找到的实现导入。

注意：本书编写时，之前的检索器定义遵循 OpenAI 的工具格式。

我们需要小心描述检索器给 LLM 的方式。我们需要确保 LLM 理解检索器并能决定使用哪个检索器。参数的描述也非常重要，以便 LLM 能够正确调用检索器。

注意，LLM 不能实际调用您的检索器；它只能决定使用哪个检索器以及传递给检索器的参数。实际调用检索器需要由调用 LLM 的系统来完成，我们将在下一节中看到。

#### 关于通用检索工具的说明

我们几乎总是包含在我们智能 RAG 系统中的通用检索工具是，当问题的答案已经在问题或其他上下文部分中给出时，将被调用的工具。这个工具通常是一个简单的函数，它从问题或上下文中提取答案并返回它。

一个例子可能是一个像“Dave Smith 的姓氏是什么？”这样的问题。这就是检索器工具可能的样子。

##### 列表 5.2 已在上下文中提供答案的通用检索工具

```py
answer_given_description = {
    "type": "function",
    "function": {
        "name": "answer_given",
        "description": "If a complete answer to the question is already provided in the conversation, use this tool to extract it.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the question",
                }
            },
            "required": ["answer"],
        },
    },
}

def answer_given(answer: str):
    """Extract the answer from a given text."""
    return answer
```

### 5.3.2 实现检索器路由器

检索器路由器是智能 RAG 系统的核心部分。其任务是接收用户问题并返回用于使用的最佳检索器。

在实现检索路由器时，我们将使用一个大型语言模型（LLM）来帮助我们完成任务。我们将向 LLM 提供一个检索器列表和用户问题，然后 LLM 将返回用于为每个问题找到答案的最佳检索器。为了简化，我们将使用具有官方工具/函数调用支持的 LLM，例如 OpenAI 的 GPT-4o。其他 LLM 也可以实现此功能，但实现方式可能不同。

在我们深入研究路由功能之前，我们需要查看一些必要的部分，以便能够成功构建一个智能 RAG 系统。这些部分包括

+   处理工具调用

+   持续查询更新

+   将问题路由到相关检索器

#### 代表 LLM 处理工具调用

当 LLM 返回要使用的最佳检索器时，系统需要调用检索器。这可以通过一个接收检索器和参数并调用检索器的函数来完成。以下列表展示了该函数可能的样子。

##### 列表 5.3 检索器调用函数

```py
def handle_tool_calls(tools: dict[str, any], llm_tool_calls: list[dict[str, any]]):
    output = []
    if llm_tool_calls:
        for tool_call in llm_tool_calls:
            function_to_call = tools[tool_call.function.name]["function"]
            function_args = json.loads(tool_call.function.arguments)
            res = function_to_call(**function_args)
            output.append(res)
    return output
```

我们传递的`tools`是一个字典，其中键是工具的名称，值是实际要调用的函数。`llm_tool_calls`是一个 LLM 决定要使用的工具及其传递给工具的参数的列表。LLM 可以决定它想要对单个问题进行多次函数调用。`llm_tool_calls`参数的形状如下：

```py
[
    {
        "function": {
            "name": "answer_given",
            "arguments": "{\"answer\": \"Dave Smith\"}"
        }
    }
]
```

#### 持续查询更新

当我们稍后到达检索器路由器函数部分时，我们会看到我们将按顺序逐个将问题发送给 LLM。这是一个故意的选择，以便让 LLM 更容易单独处理每个问题，并使将问题路由到正确的检索器更容易。

将问题按顺序发送的一个额外好处是，我们可以使用前一个问题的答案来重写下一个问题。如果用户提出的问题依赖于前一个问题的答案，这可能很有用。

考虑以下示例：“谁赢得了最多的奥斯卡奖项，这个人还活着吗？”这个问题的重写可以是“谁赢得了最多的奥斯卡奖项？”以及“这个人还活着吗？”其中第二个问题依赖于第一个问题的答案。

因此，一旦我们得到了第一个问题的答案，我们希望用新的信息更新剩余的问题。这可以通过调用一个带有原始问题和检索器答案的查询更新器来完成。查询更新器会使用新的信息更新现有的问题。

##### 列表 5.4 查询更新说明

```py
query_update_prompt = """
    You are an expert at updating questions to make them more atomic, specific, and easier to find the answer to.
    You do this by filling in missing information in the question, with the extra information provided to you in previous answers.

    You respond with the updated question that has all information in it.
    Only edit the question if needed. If the original question already is atomic, specific, and easy to answer, you keep the original.
    Do not ask for more information than the original question. Only rephrase the question to make it more complete.

    JSON template to use:
    {
        "question": "question1"
    }
"""
```

查询更新器使用原始问题和检索器的答案被调用。输出是更新后的问题，我们指示 LLM 以 JSON 格式返回更新后的问题。重要的是 LLM 不要要求比原始问题更多的信息——只需重新措辞问题以使其更完整。

##### 列表 5.5 查询更新函数

```py
def query_update(input: str, answers: list[any]) -> str:
    messages = [
        {"role": "system", "content": query_update_prompt},
        *answers,
        {"role": "user", "content": f"The user question to rewrite: '{input}'"},
    ]
    config = {"response_format": {"type": "json_object"}}
    output = chat(messages, model = "gpt-4o", config=config, )
    try:
        return json.loads(output)["question"]
    except json.JSONDecodeError:
        print("Error decoding JSON")
    return []
```

在此基础上，我们可以随着进程更新问题，并确保问题尽可能完整，并尽可能容易找到问题的答案。

#### 路由问题

检索器路由的最后一部分实际上是路由问题到正确的检索器。这是通过调用 LLM 并传递问题和可用工具来完成的，LLM 将返回每个问题的最佳检索器。

首先，我们需要在我们的工具字典中提供我们的工具，这样我们就可以将它们传递给 LLM，同时当需要调用工具时也可以找到它们。让我们首先定义我们可用的工具。

##### 列表 5.6 可用检索器工具字典

```py
tools = {
    "movie_info_by_title": {
        "description": movie_info_by_title_description,
        "function": movie_info_by_title
    },
    "movies_info_by_actor": {
        "description": movies_info_by_actor_description,
        "function": movies_info_by_actor
    },
    "text2cypher": {
        "description": text2cypher_description,
        "function": text2cypher
    },
    "answer_given": {
        "description": answer_given_description,
        "function": answer_given
    }
}
```

在这里，我们将工具描述和实际功能分组到一个字典中，这样我们可以在需要实际调用工具时轻松找到它们。让我们开始向 LLM 的提示，其中我们描述其任务。

##### 列表 5.7 检索路由器指令

```py
tool_picker_prompt = """
    Your job is to choose the right tool needed to respond to the user question.
    The available tools are provided to you in the request.
    Make sure to pass the right and complete arguments to the chosen tool.
"""
```

这是一个相当简短的提示，但足以指导 LLM 选择正确的检索器来完成工作，因为内置的工具/函数调用支持。接下来，我们将查看调用 LLM 的函数。

##### 列表 5.8 检索路由器函数

```py
def route_question(question: str, tools: dict[str, any], answers: list[dict[str, str]]):
    llm_tool_calls = tool_choice(
        [
            {
                "role": "system",
                "content": tool_picker_prompt,
            },
            *answers,
            {
                "role": "user",
                "content": f"The user question to find a tool to answer: '{question}'",
            },
        ],
        model = "gpt-4o",
        tools=[tool["description"] for tool in tools.values()],
    )
    return handle_tool_calls(tools, llm_tool_calls)
```

此函数接收一个单独的问题、可用工具和前一个问题提供的答案。然后，它使用问题和工具调用 LLM，LLM 将返回用于问题的最佳检索器。函数的最后一行是调用我们之前看到的 `handle_tool_calls` 函数，该函数实际调用检索器。

检索路由器的最后一部分是将所有先前部分整合在一起，从用户输入到答案的全过程。我们想要确保有一个循环遍历所有问题，并在过程中更新问题以包含新的信息。

##### 列表 5.9 代理 RAG 函数

```py
def handle_user_input(input: str, answers: list[dict[str, str]] = []):
    updated_question = query_update(input, answers)
    response  = route_question(updated_question, tools, answers)
    answers.append({"role": "assistant", "content": f"For the question: '{updated_question}', we have the answer: '{json.dumps(response)}'"})
    return answers
```

这里需要注意的是，`handle_user_input` 函数可以可选地接收一个答案列表。我们将在 5.3.3 节中讨论这一点。

在此基础上，我们拥有一个完整的代理 RAG 系统，该系统能够接收用户输入并返回答案。系统构建的方式允许根据需要扩展更多的检索器。

我们需要实现一个额外的部分来使系统完整，那就是答案批评家。

### 5.3.3 实现答案批评家

答案批评家的任务是接收所有来自检索器的答案，并检查原始问题是否得到正确回答。LLM 是非确定性的，在重写问题、更新问题和路由问题时可能会出错，因此我们希望设置这个检查以确保我们确实收到了所需答案。

以下列表显示了针对答案批评家的 LLM 指令。

##### 列表 5.10 答案批评家指令

```py
answer_critique_prompt = """
    You are an expert at identifying if questions have been fully answered or if there is an opportunity to enrich the answer.
    The user will provide a question, and you will scan through the provided information to see if the question is answered.
    If anything is missing from the answer, you will provide a set of new questions that can be asked to gather the missing information.
    All new questions must be complete, atomic, and specific.
    However, if the provided information is enough to answer the original question, you will respond with an empty list.

    JSON template to use for finding missing information:
    {
        "questions": ["question1", "question2"]
    }
"""
```

我们遵循之前的模式，使用 JSON 格式和 LLM 的指令。

接下来，我们将查看调用 LLM 的函数。

##### 列表 5.11 答案批评家函数

```py
def critique_answers(question: str, answers: list[dict[str, str]]) -> list[str]:
    messages = [
        {
            "role": "system",
            "content": answer_critique_prompt,
        },
        *answers,
        {
            "role": "user",
            "content": f"The original user question to answer: {question}",
        },
    ]
    config = {"response_format": {"type": "json_object"}}
    output = chat(messages, model="gpt-4o", config=config)
    try:
        return json.loads(output)["questions"]
    except json.JSONDecodeError:
        print("Error decoding JSON")
    return []
```

此函数接收原始问题和检索器提供的答案，并调用 LLM 检查原始问题是否得到正确回答。如果问题没有正确回答，LLM 将返回一系列可以提出的新问题，以收集缺失的信息。

如果我们收到一系列新问题，我们可以再次通过检索路由器来获取缺失信息。我们还应该设置一些退出标准，以避免陷入无法从检索器中获得原始问题答案的循环。

### 5.3.4 整合所有部分

到目前为止，我们已经实现了检索代理、检索路由器和答案批评家。最后一步是将所有这些部分整合到一个主函数中，该函数接收用户输入并返回答案，如果答案可用的话。

以下列表显示了主要功能可能的样子。让我们从对 LLM 的指令开始。

##### 列表 5.12 代理型 RAG 主要指令

```py
main_prompt = """
    Your job is to help the user with their questions.
    You will receive user questions and information needed to answer the questions
    If the information is missing to answer part of or the whole question, you will say that the information
    is missing. You will only use the information provided to you in the prompt to answer the questions.
    You are not allowed to make anything up or use external information.
"""
```

非常重要的是，大型语言模型（LLM）在回答问题时只能使用其提示中提供的信息。这是为了保证系统的连贯性，以及我们能够信任它提供的答案。

接下来，我们将查看主要功能。

##### 列表 5.13 代理型 RAG 主要功能

```py
def main(input: str):
    answers = handle_user_input(input)
    critique = critique_answers(input, answers)

    if critique:
        answers = handle_user_input(" ".join(critique), answers)

    llm_response = chat(
        [
            {"role": "system", "content": main_prompt},
            *answers,
            {"role": "user", "content": f"The user question to answer: {input}"},
        ],
        model="gpt-4o",
    )

    return llm_response
```

主要功能将用户输入通过代理型 RAG 系统运行，并将答案返回给用户。如果答案不完整或不正确，评估功能将返回一系列新问题，这些问题可以用来收集缺失的信息。

我们只对答案进行一次评估；如果评估后答案仍然不完整或不正确，我们将原样返回答案给用户，并依赖 LLM 让用户知道哪些信息不完整。

## 摘要

+   代理型 RAG 是一个提供多种检索代理的系统，用于检索回答用户问题所需的数据。

+   代理型 RAG 系统的主要界面通常是某种用例或检索路由器，其任务是找到最适合执行当前任务的检索器（或检索器组）。

+   代理型 RAG 系统的基本部分包括检索代理、检索路由器和答案评估器。

+   代理型 RAG 系统的主要部分可以使用具有工具/函数调用支持的 LLM 来实现。

+   检索代理可以是通用的或专门的，应根据需要逐步添加，以改善应用程序的性能。

+   答案评估器是一个函数，它接收检索器提供的答案，并检查原始问题是否得到正确回答。
