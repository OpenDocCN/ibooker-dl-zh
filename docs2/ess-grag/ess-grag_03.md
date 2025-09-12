# 第四章：从自然语言问题生成 Cypher 查询

### 本章涵盖

+   查询语言生成的基础知识

+   查询语言生成在 RAG 管道中的位置

+   查询语言生成的实用技巧

+   使用基础模型实现 text2cypher 检索器

+   用于文本 2cypher 的专业（微调）LLM

在前面的章节中，我们已经覆盖了很多内容。我们学习了如何构建知识图谱，从文本中提取信息，并使用这些信息来回答问题。我们还探讨了如何通过使用硬编码的 Cypher 查询来扩展和改进普通的向量搜索检索，从而为 LLM 获取更多相关的上下文。在本章中，我们将更进一步，学习如何从自然语言问题生成 Cypher 查询。这将使我们能够构建一个更灵活和动态的检索系统，能够适应不同类型的问题和知识图谱。

注意：在本章的实现中，我们使用所谓的“电影数据集”。有关数据集的更多信息以及各种加载方式，请参阅附录。

## 4.1 查询语言生成的基础知识

当我们谈论查询语言生成的基础知识时，我们指的是将自然语言问题转换为可以在数据库上执行的语言的过程。更具体地说，我们感兴趣的是从自然语言问题生成 Cypher 查询。大多数 LLM 都知道 Cypher 是什么，也知道该语言的基本语法。在这个过程中，主要挑战是生成一个既正确又与所提问题相关的查询。这需要理解问题的语义以及被查询的知识图谱的模式。

如果我们不提供知识图谱的模式，LLM 只能假设节点、关系和属性的名称。当提供模式时，它充当用户问题的语义与所使用的图模型之间的映射——节点上使用的标签、存在的关联类型、可用的属性以及节点连接到的关联类型。

从自然语言问题生成 Cypher 查询的工作流程可以分解为以下步骤（图 4.1）：

+   从用户那里检索问题。

+   检索知识图谱的模式。

+   定义其他有用的信息，如术语映射、格式说明和少量示例。

+   为 LLM 生成提示。

+   将提示传递给 LLM 以生成 Cypher 查询。

![figure](img/4-1.png)

##### 图 4.1 从自然语言问题生成 Cypher 查询的工作流程

## 4.2 查询语言生成在 RAG 管道中的位置

在前面的章节中，我们看到了如何通过在图的无结构部分执行向量相似度搜索来从知识图谱中获得相关响应。我们还看到了如何使用扩展了硬编码 Cypher 查询的向量相似度搜索来为 LLM 提供更多相关上下文。这些技术的局限性在于它们在可以回答的问题类型上受到限制。

考虑用户问题：“列出由史蒂文·斯皮尔伯格执导的前三部评分最高的电影及其平均分。”这个问题永远不能通过向量相似度搜索来回答，因为它需要在数据库上执行特定类型的查询，Cypher 查询可能如下所示（假设有合理的模式）。

##### 列表 4.1 Cypher 查询

```py
MATCH (:Reviewer)-[r:REVIEWED]->(m:Movie)<-[:DIRECTED]-(:Director {name: 'Steven Spielberg'})
RETURN m.title, AVG(r.score) AS avg_rating
ORDER BY avg_rating DESC
LIMIT 3
```

这个查询更多的是关于以特定方式聚合数据，而不是关于图中最相似的节点。这表明我们希望使用生成的 Cypher 来执行某些类型的查询——当我们寻找的不是图中最相似的节点，或者我们想要以某种方式聚合数据时。在下一章中，我们将探讨如何创建一个代理系统，我们可以提供多个检索器，并为每个用户问题选择最合适的一个，以便能够向用户提供最佳响应。

Text2cypher 也可以作为“万能”检索器，用于那些在系统中没有其他检索器能提供良好匹配的问题类型。

## 4.3 查询语言生成的实用技巧

当从自然语言问题生成 Cypher 查询时，有一些事情需要考虑，以确保生成的查询是正确且相关的。LLM 在生成 Cypher 查询时容易出错，尤其是当输入问题复杂或含糊不清，或者数据库模式元素没有语义命名时。

### 4.3.1 使用少量示例进行上下文学习

少量示例是提高 LLM 在 text2cypher 中性能的绝佳方式。这意味着我们可以向 LLM 提供一些问题和它们相应的 Cypher 查询的示例，LLM 将学会为新的问题生成类似的查询。相比之下，零示例是在我们不向 LLM 提供任何示例的情况下，它必须在没有任何提示的情况下生成查询。

几个示例是针对查询的知识图谱特定的，因此需要为每个知识图谱手动创建。这在您意识到 LLM 误解了模式或经常犯相同类型的错误（期望一个属性而实际上应该是遍历等）时非常有用。

假设您检测到 LLM 正在尝试读取电影的制作国家，并且它在电影节点上寻找一个属性，但实际上国家是图中的一个节点。然后您可以在提示中添加少量示例，让 LLM 知道如何获取国家名称：

电影《黑客帝国》是在哪个国家制作的？

```py
MATCH (m:Movie {title: 'The Matrix'}) RETURN m.country
```

这可以通过在提示 LLM 的几个示例中添加以下内容来解决：

电影《黑客帝国》是在哪个国家制作的？

示例

问题：电影《头号玩家》是在哪个国家制作的？

Cypher：MATCH (m:Movie { title: '头号玩家' })-[:PRODUCED_IN]→(c:Country) RETURN c.name

```py
MATCH (m:Movie {title: 'The Matrix'})-[:PRODUCED_IN]->(c:Country)
↪ RETURN c.name
```

这不仅会解决这个具体问题，而且由于我们现在有一个清晰的例子让 LLM 看到模式以获取国家名称，所以也会解决类似的问题。

### 4.3.2 在提示中使用数据库模式向 LLM 展示知识图谱的结构

知识图谱的模式对于生成正确的 Cypher 查询至关重要。有几种方式可以向 LLM 描述知识图谱模式，根据 Neo4j 内部的研究，格式并不那么重要。

模式应该是提示的一部分，并清楚地说明图中可用的标签、关系类型和属性：

图数据库模式：

仅在模式中提供的关系类型和属性中使用。不要使用模式中未提供的任何其他关系类型或属性。

节点标签和属性：

```py
LabelA {property_a: STRING}
```

关系类型和属性：

```py
REL_TYPE {rel_prop: STRING}
```

关系：

```py
(:LabelA)-[:REL_TYPE]->(:LabelB)
(:LabelA)-[:REL_TYPE]->(:LabelC)
```

您是否希望公开完整的知识图谱以进行查询，可能取决于模式的大小以及它是否与用例相关。自动从 Neo4j 推断模式可能很昂贵，这取决于数据的大小，因此通常从数据库中采样并从中推断模式是常见的做法。

要从 Neo4j 推断模式，我们目前需要使用 APOC 库中的过程，该库免费且在 Neo4j 的 SaaS 产品 Aura 和其他 Neo4j 发行版中均可用。以下列表显示了如何从 Neo4j 数据库中推断模式。

小贴士：您可以在[`neo4j.com/docs/apoc/`](https://neo4j.com/docs/apoc/)了解更多关于 APOC 的信息。

##### 列表 4.2 从 Neo4j 推断模式

```py
NODE_PROPERTIES_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output
"""

REL_PROPERTIES_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
WITH label AS relType, collect({property:property, type:type}) AS properties
RETURN {type: relType, properties: properties} AS output
"""

REL_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE type = "RELATIONSHIP" AND elementType = "node"
UNWIND other AS other_node
RETURN {start: label, type: property, end: toString(other_node)} AS output
"""
```

使用这些查询，我们现在可以获取图数据库的模式并将其用于提示 LLM。让我们运行这些查询并以结构化的方式存储结果，这样我们就可以稍后生成前面的模式字符串。

##### 列表 4.3 运行模式推断查询

```py
def get_structured_schema(driver: neo4j.Driver) -> dict[str, Any]:
    node_labels_response = driver.execute_query(NODE_PROPERTIES_QUERY)
    node_properties = [
        data["output"]
        for data in [r.data() for r in node_labels_response.records]
    ]

    rel_properties_query_response = driver.execute_query(REL_PROPERTIES_QUERY)
    rel_properties = [
        data["output"]
        for data in [r.data() for r in rel_properties_query_response.records]
    ]

    rel_query_response = driver.execute_query(REL_QUERY)
    relationships = [
        data["output"]
        for data in [r.data() for r in rel_query_response.records]
    ]

    return {
        "node_props": {el["labels"]: el["properties"] for el in node_properties},
        "rel_props": {el["type"]: el["properties"] for el in rel_properties},
        "relationships": relationships,
    }
```

在这个结构化响应到位后，我们可以按需格式化模式字符串，并且我们也很容易在提示中探索和实验不同的格式。

要获得本章前面展示的格式，我们可以使用以下列表中显示的函数。

##### 列表 4.4 格式化模式字符串

```py
def get_schema(structured_schema: dict[str, Any]) -> str:
    def _format_props(props: list[dict[str, Any]]) -> str:
        return ", ".join([f"{prop['property']}: {prop['type']}" for prop in props])

    formatted_node_props = [
        f"{label} {{{_format_props(props)}}}"
        for label, props in structured_schema["node_props"].items()
    ]

    formatted_rel_props = [
        f"{rel_type} {{{_format_props(props)}}}"
        for rel_type, props in structured_schema["rel_props"].items()
    ]

    formatted_rels = [
        f"(:{element['start']})-[:{element['type']}]->(:{element['end']})"
        for element in structured_schema["relationships"]
    ]

    return "\n".join(
        [
            "Node labels and properties:",
            "\n".join(formatted_node_props),
            "Relationship types and properties:",
            "\n".join(formatted_rel_props),
            "The relationships:",
            "\n".join(formatted_rels),
        ]
    )
```

使用这个函数，我们现在可以生成可以用于提示 LLM 的模式字符串。

### 4.3.3 添加术语映射以语义地将用户问题映射到模式

LLM 需要知道如何将问题中使用的术语映射到模式中使用的术语。一个设计良好的图模式使用名词和动词作为标签和关系类型，以及形容词和名词作为属性。即使如此，LLM 有时也可能不清楚在哪里使用什么。

注意：这些映射是知识图谱特定的，应该作为提示的一部分；它们在不同知识图谱之间难以重用。

术语映射可能是随着时间的推移而演变的东西，因为当你发现由于 LLM 没有正确理解模式而导致的生成查询问题时。

术语映射：

人物：当用户询问一个职业人物时，他们指的是具有 Person 标签的节点。电影：当用户询问一部电影或电影时，他们指的是具有 Movie 标签的节点。

### 4.3.4 格式说明

不同的 LLM 以不同的方式输出响应。其中一些在 Cypher 查询周围添加代码标签，而另一些则没有。一些在 Cypher 查询之前添加文本；而另一些则没有，等等。

要使它们都以相同的方式输出，你可以在提示中添加格式说明。有用的说明是尝试让 LLM 只输出 Cypher 查询，而不输出其他任何内容。

格式说明：

不要在响应中包含任何解释或道歉。不要回答任何可能要求你构建 Cypher 语句之外的问题。不要包含任何文本，除了生成的 Cypher 语句。只以 CYPHER 回答，不要包含代码块。

## 4.4 使用基础模型实现 text2cypher 生成器

让我们将所有这些应用到实践中，并使用基础模型实现一个 text2cypher 生成器。这里的任务基本上是形成一个包含模式、术语映射、格式说明和少量示例的提示，以便向 LLM 明确我们的意图。

在本章的剩余部分，我们将使用 Neo4j Python 驱动程序和 OpenAI API 实现一个 text2cypher 生成器。为了跟随，你需要访问一个运行中的、空白的 Neo4j 实例。这可以是一个本地安装或云托管实例；只需确保它是空的。你可以直接在附带的 Jupyter 笔记本中跟随实现，笔记本地址如下：[`github.com/tomasonjo/kg-rag/blob/main/notebooks/ch04.ipynb`](https://github.com/tomasonjo/kg-rag/blob/main/notebooks/ch04.ipynb)。

让我们深入探讨。

##### 列表 4.5 提示模板

```py
prompt_template = """
Instructions:
Generate Cypher statement to query a graph database to get the data to answer the following user question.

Graph database schema:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided in the schema.
{schema}

Terminology mapping:
This section is helpful to map terminology between the user question and the graph database schema.
{terminology}
Examples:
The following examples provide useful patterns for querying the graph database.
{examples}

Format instructions:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to
construct a Cypher statement.
Do not include any text except the generated Cypher statement.
ONLY RESPOND WITH CYPHER—NO CODE BLOCKS.

User question: {question}
"""
```

使用这个提示模板，我们现在可以为 LLM 生成提示。假设我们有一个以下用户问题、模式、术语映射和少量示例。

##### 列表 4.6 完整提示示例

```py
question = "Who directed the most movies?"

schema_string = get_schema(neo4j_driver)

terminology_string = """
Persons: When a user asks about a person by trade like actor, writer, director, producer,  or reviewer, they are referring to a node with the label 'Person'.
Movies: When a user asks about a film or movie, they are referring to a node with the label Movie.
"""

examples = [["Who are the two people acted in most movies together?", "MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person) WHERE p1 <> p2 RETURN p1.name, p2.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1"]]

full_prompt = prompt_template.format(question=question, schema=schema_string, terminology=terminology_string,examples="\n".join([f"Question: {e[0]}\nCypher: {e[1]}" for i, e in enumerate(examples)]))
print(full_prompt)
```

如果我们执行这个示例，提示输出将看起来像这样：

说明：生成 Cypher 语句以查询图数据库以获取回答以下用户问题的数据。

图数据库模式：仅使用模式中提供的关系类型和属性。不要使用模式中未提供的任何其他关系类型或属性。节点属性：

```py
Movie {tagline: STRING, title: STRING, released: INTEGER}
Person {born: INTEGER, name: STRING}
```

关系属性：

```py
ACTED_IN {roles: LIST}
REVIEWED {summary: STRING, rating: INTEGER}
```

关系：

```py
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:Person)-[:PRODUCED]->(:Movie)
(:Person)-[:WROTE]->(:Movie)
(:Person)-[:FOLLOWS]->(:Person)
(:Person)-[:REVIEWED]->(:Movie)
```

术语映射：本节有助于在用户问题和图数据库模式之间映射术语。

人物：当用户询问像演员、作家、导演、制片人或评论家这样的职业人物时，他们指的是带有标签'Person'的节点。电影：当用户询问电影或影片时，他们指的是带有标签 Movie 的节点。

示例：以下示例提供了查询图数据库的有用模式。问题：哪两位演员共同出演了最多的电影？

```py
Cypher: MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
↪ WHERE p1 <> p2 RETURN p1.name, p2.name, COUNT(m) AS movieCount
↪ ORDER BY movieCount DESC LIMIT 1
```

格式说明：在您的回答中不要包含任何解释或道歉。不要回答任何可能要求您构建 Cypher 语句之外的问题。不要包含任何除生成的 Cypher 语句之外的文字。只回答 CYPHER——不要使用代码块。

用户问题：谁执导的电影最多？

使用这个提示，我们现在可以生成用户问题的 Cypher 查询。您可以尝试将提示复制到 LLM 中，看看它生成什么。

##### 列表 4.7 生成的 Cypher 查询

```py
MATCH (p:Person)-[:DIRECTED]->(m:Movie)
RETURN p.name, COUNT(m) AS movieCount
ORDER BY movieCount
DESC LIMIT 1
```

## 4.5 专为 text2cypher 优化的（微调的）LLM

在 Neo4j，我们正在通过微调不断改进 text2cypher 的 LLM 性能。我们 Hugging Face 上的开源训练数据可在[`huggingface.co/datasets/neo4j/text2cypher`](https://huggingface.co/datasets/neo4j/text2cypher)找到。我们还提供基于开源 LLM（如 Gemma2、Llama 3.1）的微调模型，可在[`huggingface.co/neo4j`](https://huggingface.co/neo4j)找到。

这些模型在性能上仍然远远落后于像最新 GPT 和 Gemini 模型这样的微调大型模型，但它们效率更高，可以在大型模型太慢的生产系统中使用。大胆尝试它们，并参考少量示例、模式、术语映射和格式说明来提高模型性能。有关我们的微调过程和学习，更多信息请参阅[`mng.bz/MwDW`](https://mng.bz/MwDW)、[`mng.bz/a9v7`](https://mng.bz/a9v7)和[`mng.bz/yNWB`](https://mng.bz/yNWB)。

## 4.6 我们学到的东西以及 text2cypher 能做什么

在本章的代码和信息的基础上，您应该能够为您的知识图谱实现一个 text2cypher 检索器。您应该能够让它为广泛的问题生成正确的 Cypher 查询，并通过提供少量示例、模式、术语映射和格式说明来提高其性能。

随着你识别出它难以应对的问题类型，你可以向提示中添加更多少样本示例，以帮助它学习如何生成正确的查询。随着时间的推移，你会发现生成的查询质量有所提高，检索器变得更加可靠。

## 摘要

+   查询语言生成与 RAG（Retrieval-Augmented Generation）管道很好地结合，作为其他检索方法的补充，尤其是在我们想要聚合数据或从图中获取特定数据时。

+   查询语言生成的有用实践包括使用少样本示例、模式、术语映射和格式说明。

+   我们可以使用基础模型实现一个文本到 Cypher 的检索器，并将提示结构化到 LLM 中。

+   我们可以使用专门（微调）的 LLM（大型语言模型）进行文本到 Cypher 的转换，并提高其性能。
