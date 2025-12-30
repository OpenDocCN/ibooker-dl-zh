# 第三章\. RAG 第二部分：与数据对话

在上一章中，你学习了如何处理数据，并在向量存储中创建和存储嵌入。在本章中，你将学习如何根据用户的查询高效地检索最相关的嵌入和文档片段。这使你能够构建一个包含相关文档作为背景的提示，从而提高 LLM 最终输出的准确性。

这个过程——涉及嵌入用户的查询，从数据源检索相似文档，然后将它们作为背景传递给发送给 LLM 的提示——正式上被称为 *检索增强生成* (RAG)。

RAG 是构建准确、高效且最新的聊天功能 LLM 应用的一个关键组件。在本章中，你将从基础知识进步到高级策略，以构建针对各种数据源（如向量存储和数据库）和数据结构（结构化和非结构化）的有效 RAG 系统。

但首先，让我们定义 RAG 并讨论其优势。

# 介绍检索增强生成

RAG 是一种技术，通过提供外部来源的上下文来增强 LLM 生成输出的准确性。这个术语最初是由 Meta AI 研究人员在一篇论文中提出的，他们发现启用 RAG 的模型比非 RAG 模型更具事实性和具体性.^(1)

没有 RAG，LLM 完全依赖于其预训练的数据，这些数据可能已经过时。例如，让我们向 ChatGPT 提出一个关于当前事件的问题，看看它的响应：

*输入*

```py
Which country is the latest winner of the men’s FIFA World Cup?
```

*输出*

```py
The most recent FIFA World Cup winner was France, who won the tournament in 2018.
```

LLM 的响应在事实上是不正确的，并且过时。在本书出版时，最新的冠军是阿根廷，他们在 2022 年赢得了世界杯。虽然这个例子中的问题可能微不足道，但如果 LLM 的答案被用于事实核查或重要决策，LLM 的幻觉可能会产生灾难性的后果。

为了解决这个问题，我们需要向 LLM 提供事实性、最新的信息，以便它能据此形成准确的响应。继续上一个例子，让我们转到维基百科的 [FIFA 世界杯](https://oreil.ly/LpLOV) 页面，复制简介段落，然后将它作为 *背景* 添加到我们的 ChatGPT 提示中：

```py
Which country is the latest winner of the men's FIFA World Cup?

See context below.

The FIFA World Cup, often called the World Cup, is an international association
football competition among the senior men's national teams of the members of
the Fédération Internationale de Football Association (FIFA), the sport's 
global governing body. The tournament has been held every four years since the 
inaugural tournament in 1930, with the exception of 1942 and 1946 due to the 
Second World War. The reigning champions are Argentina, who won their third 
title at the 2022 tournament.
```

注意到最后一句包含了 LLM 可以使用以提供准确答案的必要上下文。以下是 LLM 的响应：

```py
The latest winner of the men's FIFA World Cup is Argentina, who won their third
title at the 2022 tournament.
```

由于提供了最新的附加背景信息，LLM 能够对提示生成准确的响应。但是，将相关信息作为背景复制粘贴并不适用于生产 AI 应用，也不具备可扩展性。我们需要一个自动化的系统，根据用户的查询获取相关信息，将其作为背景添加到提示中，然后向 LLM 执行生成请求。

## 检索相关文档

一个 AI 应用的 RAG 系统通常遵循三个核心阶段：

索引

此阶段涉及预处理外部数据源并将表示数据的嵌入存储在向量存储中，以便可以轻松检索。

检索

此阶段涉及根据用户查询从向量存储中检索相关嵌入和数据。

生成

此阶段涉及将检索到的相关文档与原始提示合并为一个最终提示，并将其发送到模型进行预测。

三个基本阶段看起来像图 3-1。

![文档的示意图  自动生成的描述](img/lelc_0301.png)

###### 图 3-1\. RAG 的关键阶段

此过程的索引阶段在第二章中进行了详细说明，其中你学习了如何使用文档加载器、文本分割器、嵌入和向量存储。

让我们再次从头开始运行一个示例，从索引阶段开始：

*Python*

```py
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector

# Load the document, split it into chunks
raw_documents = TextLoader('./test.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
    chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# embed each chunk and insert it into the vector store
model = OpenAIEmbeddings()
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector.from_documents(documents, model, connection=connection)
```

*JavaScript*

```py
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";

// Load the document, split it into chunks
const loader = new TextLoader("./test.txt");
const raw_docs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const docs = await splitter.splitDocuments(docs)

// embed each chunk and insert it into the vector store
const model = new OpenAIEmbeddings();
const db = await PGVectorStore.fromDocuments(docs, model, {
  postgresConnectionOptions: {
    connectionString: 'postgresql://langchain:langchain@localhost:6024/langchain'
  }
})
```

第二章有更多关于索引阶段的详细信息。

索引阶段现已完成。为了执行检索阶段，我们需要在用户查询和我们的存储嵌入之间执行相似度搜索计算（例如余弦相似度），以便检索我们索引文档的相关片段（参见图 3-2）。

![2024-02-12 下午 1:36:56 的屏幕截图.png](img/lelc_0302.png)

###### 图 3-2\. 从向量存储中检索相关文档的同时索引文档的示例流程；分层可导航小世界（HNSW）框表示计算文档与用户查询的相似度

图 3-2 说明了检索过程中的步骤：

1.  将用户查询转换为嵌入。

1.  计算与用户查询最相似的向量存储中的嵌入。

1.  检索相关文档嵌入及其对应的文本片段。

我们可以使用 LangChain 以编程方式表示这些步骤，如下所示：

*Python*

```py
# create retriever
retriever `=` db`.`as_retriever()

# fetch relevant documents
docs `=` retriever`.`invoke("""Who are the key figures in the ancient greek 
 history of philosophy?""")
```

*JavaScript*

```py
// create retriever
const retriever = db.asRetriever()

// fetch relevant documents
const docs = await retriever.invoke(`Who are the key figures in the ancient 
 greek history of philosophy?`)
```

注意，我们正在使用一个你之前没有见过的向量存储方法：`as_retriever`。此函数抽象了将用户查询嵌入以及向量存储执行的相关文档检索的底层相似度搜索计算的逻辑。

还有一个参数 `k`，它决定了从向量存储中检索的相关文档数量。例如：

*Python*

```py
# create retriever with k=2
retriever = db.as_retriever(search_kwargs={"k": 2})

# fetch the 2 most relevant documents
docs = retriever.invoke("""Who are the key figures in the ancient greek history 
 of philosophy?""")
```

*JavaScript*

```py
// create retriever with k=2
const retriever = db.asRetriever({k: 2})

// fetch the 2 most relevant documents
const docs = await retriever.invoke(`Who are the key figures in the ancient 
 greek history of philosophy?`)
```

在此示例中，参数 `k` 被指定为 2。这告诉向量存储根据用户查询返回两个最相关的文档。

使用低`k`值可能看起来有些反直觉，但检索更多文档并不总是更好的。检索到的文档越多，你的应用程序性能越慢，提示（以及相关的生成成本）越大，检索到包含不相关信息文本块的可能性也越大，这会导致 LLM 产生幻觉。

现在我们已经完成了 RAG 系统的检索阶段，让我们继续到最后生成阶段。

## 使用相关文档生成 LLM 预测

一旦我们根据用户的查询检索到相关文档，下一步就是将它们添加到原始提示中作为上下文，然后调用模型生成最终输出(图 3-3)。

![一个图表的图表 描述自动生成](img/lelc_0303.png)

###### 图 3-3\. 一个示例流程，展示了索引文档、从向量存储中检索相关文档，并将检索到的文档作为上下文包含在 LLM 提示中

这里是一个代码示例，继续我们之前的例子：

*Python*

```py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template("""Answer the question based only on 
 the following context:
{context}

Question: {question}
""")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

chain = prompt | llm

# fetch relevant documents 
docs = retriever.get_relevant_documents("""Who are the key figures in the 
 ancient greek history of philosophy?""")

# run
chain.invoke({"context": docs,"question": """Who are the key figures in the 
 ancient greek history of philosophy?"""})
```

*JavaScript*

```py
import {ChatOpenAI} from '@langchain/openai'
import {ChatPromptTemplate} from '@langchain/core/prompts'

const retriever = db.asRetriever()

const prompt = ChatPromptTemplate.fromTemplate(`Answer the question based only 
 on the following context:
{context}

Question: {question}
`)

const llm = new ChatOpenAI({temperature: 0, modelName: 'gpt-3.5-turbo'})

const chain = prompt.pipe(llm)

// fetch relevant documents
const docs = await retriever.invoke(`Who are the key figures in the ancient 
 greek history of philosophy?`)

await chain.invoke({context: docs, question: `Who are the key figures in the 
 ancient greek history of philosophy?`})
```

注意以下更改：

+   我们将动态的`context`和`question`变量实现到我们的提示中，这使得我们可以定义一个模型可以使用的`ChatPromptTemplate`来生成响应。

+   我们定义了一个`ChatOpenAI`接口来作为我们的 LLM。温度设置为 0 以消除模型输出中的创造性。

+   我们创建一个链来组合提示和 LLM。提醒：`|`运算符（或在 JS 中的`pipe`方法）将`prompt`的输出用作`llm`的输入。

+   我们`调用`链，传入`context`变量（我们检索到的相关文档）和用户的问题来生成最终输出。

我们可以将这个检索逻辑封装在一个单独的函数中：

*Python*

```py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template("""Answer the question based only on 
 the following context:
{context}

Question: {question}
""")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

@chain
def qa(input):
    # fetch relevant documents 
    docs = retriever.get_relevant_documents(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

# run
qa.invoke("Who are the key figures in the ancient greek history of philosophy?")
```

*JavaScript*

```py
import {ChatOpenAI} from '@langchain/openai'
import {ChatPromptTemplate} from '@langchain/core/prompts'
import {RunnableLambda} from '@langchain/core/runnables'

const retriever = db.asRetriever()

const prompt = ChatPromptTemplate.fromTemplate(`Answer the question based only 
 on the following context:
{context}

Question: {question}
`)

const llm = new ChatOpenAI({temperature: 0, modelName: 'gpt-3.5-turbo'})

const qa = RunnableLambda.from(async input => {
  // fetch relevant documents
  const docs = await retriever.invoke(input)
  // format prompt
  const formatted = await prompt.invoke({context: docs, question: input})
  // generate answer
  const answer = await llm.invoke(formatted)
  return answer
})

await qa.invoke(`Who are the key figures in the ancient greek history of 
 philosophy?`)
```

注意我们现在有一个新的可运行的`qa`函数，只需一个问题和调用它，它会首先获取相关的文档作为上下文，将它们格式化为提示，并最终生成答案。在 Python 代码中，`@chain`装饰器将函数转换为可运行的链。将多个步骤封装到单个函数中的这种概念将是构建有趣 LLM 应用的关键。

你也可以返回检索到的文档以供进一步检查：

*Python*

```py
@chain
def qa(input):
    # fetch relevant documents 
    docs = retriever.get_relevant_documents(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return {"answer": answer, "docs": docs}
```

*JavaScript*

```py
const qa = RunnableLambda.from(async input => {
  // fetch relevant documents
  const docs = await retriever.invoke(input)
  // format prompt
  const formatted = await prompt.invoke({context: docs, question: input})
  // generate answer
  const answer = await llm.invoke(formatted)
  return {answer, docs}
})
```

恭喜！你现在已经构建了一个基本的 RAG 系统，用于为个人使用提供 AI 应用。

然而，一个由多个用户使用的生产就绪 AI 应用需要一个更高级的 RAG 系统。为了构建一个健壮的 RAG 系统，我们需要有效地回答以下问题：

+   我们如何处理用户输入质量的变化？

+   我们如何路由查询以从各种数据源检索相关数据？

+   我们如何将自然语言转换为目标数据源的查询语言？

+   我们如何优化我们的索引过程，即嵌入、文本分割？

接下来，我们将讨论最新的基于研究策略来回答这些问题并构建一个生产就绪的 RAG 系统。这些策略可以总结为图 3-4。

![软件开发流程图  自动生成的描述](img/lelc_0304.png)

###### 图 3-4\. 优化 RAG 系统准确性的有效策略

###### 注意

本章剩余部分的所有代码块都使用我们在本章开头设置的向量存储。

# 查询转换

基本 RAG 系统的一个主要问题是它过于依赖用户查询的质量来生成准确的输出。在生产环境中，用户很可能会以不完整、含糊或措辞不当的方式构建他们的查询，这会导致模型产生幻觉。

*查询转换*是策略的子集，旨在修改用户的输入以回答第一个 RAG 问题：我们如何处理用户输入质量的变化？图 3-5 展示了查询转换策略的范围，从那些使用户的输入更抽象或更具体以生成准确的 LLM 输出的策略。下一节将从一种中间策略开始。

![问题图  自动生成的描述](img/lelc_0305.png)

###### 图 3-5\. 基于抽象级别转换用户查询的各种方法

## 重写-检索-阅读

微软研究团队提出的重写-检索-阅读策略简单地将用户的查询重写，然后再进行检索。2 为了说明这一点，让我们回到上一节中构建的链，这次使用一个措辞不当的用户查询：

*Python*

```py
@chain
def qa(input):
    # fetch relevant documents 
    docs = retriever.get_relevant_documents(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

qa.invoke("""Today I woke up and brushed my teeth, then I sat down to read the 
 news. But then I forgot the food on the cooker. Who are some key figures in 
 the ancient greek history of philosophy?""")
```

*JavaScript*

```py
const qa = RunnableLambda.from(async input => {
  // fetch relevant documents
  const docs = await retriever.invoke(input)
  // format prompt
  const formatted = await prompt.invoke({context: docs, question: input})
  // generate answer
  const answer = await llm.invoke(formatted)
  return answer
})

await qa.invoke(`Today I woke up and brushed my teeth, then I sat down to read 
 the news. But then I forgot the food on the cooker. Who are some key figures 
 in the ancient greek history of philosophy?`)
```

*输出*（记住：如果你重新运行它，你的输出可能与这个不同）：

```py
Based on the given context, there is no information provided.
```

模型未能回答问题，因为它被用户查询中提供的不相关信息所分心。

现在我们来实现重写-检索-阅读提示：

*Python*

```py
rewrite_prompt = ChatPromptTemplate.from_template("""Provide a better search 
 query for web search engine to answer the given question, end the queries 
 with ’**’. Question: {x} Answer:""")

def parse_rewriter_output(message):
    return message.content.strip('"').strip("**")

rewriter = rewrite_prompt | llm | parse_rewriter_output

@chain
def qa_rrr(input):
    # rewrite the query
    new_query = rewriter.invoke(input)
    # fetch relevant documents 
    docs = retriever.get_relevant_documents(new_query)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

# run
qa_rrr.invoke("""Today I woke up and brushed my teeth, then I sat down to read 
 the news. But then I forgot the food on the cooker. Who are some key 
 figures in the ancient greek history of philosophy?""")
```

*JavaScript*

```py
const rewritePrompt = ChatPromptTemplate.fromTemplate(`Provide a better search 
 query for web search engine to answer the given question, end the queries 
 with ’**’. Question: {question} Answer:`)

const rewriter = rewritePrompt.pipe(llm).pipe(message => {
  return message.content.replaceAll('"', '').replaceAll('**')
})

const qa = RunnableLambda.from(async input => {
  const newQuery = await rewriter.invoke({question: input});
  // fetch relevant documents
  const docs = await retriever.invoke(newQuery)
  // format prompt
  const formatted = await prompt.invoke({context: docs, question: input})
  // generate answer
  const answer = await llm.invoke(formatted)
  return answer
})

await qa.invoke(`Today I woke up and brushed my teeth, then I sat down to read 
 the news. But then I forgot the food on the cooker. Who are some key 
 figures in the ancient greek history of philosophy?`)
```

*输出：*

```py
Based on the given context, some key figures in the ancient greek history of 
philosophy include: Themistocles (an Athenian statesman), Pythagoras, and Plato.
```

注意到我们让 LLM 将用户的初始分心查询重写为一个更清晰的查询，并且是那个更集中的查询被传递给检索器以获取最相关的文档。注意：这项技术可以与任何检索方法一起使用，无论是我们这里使用的向量存储，还是例如网络搜索工具。这种方法的缺点是它将额外的延迟引入到你的链中，因为现在我们需要连续进行两次 LLM 调用。

## 多查询检索

一个用户的一个查询可能不足以捕捉回答查询所需的所有信息范围。多查询检索策略通过指示 LLM 根据用户的初始查询生成多个查询来解决此问题，从数据源并行检索每个查询，然后将检索到的结果作为提示上下文来生成最终的模型输出。图 3-6 说明了这一点。

![文档的图示  描述由中等置信度自动生成](img/lelc_0306.png)

###### 图 3-6\. 多查询检索策略的演示

这种策略特别适用于那些一个问题可能需要多个视角来提供全面答案的场景。

这里是一个多查询检索操作的代码示例：

*Python*

```py
from langchain.prompts import ChatPromptTemplate

perspectives_prompt = ChatPromptTemplate.from_template("""You are an AI language 
 model assistant. Your task is to generate five different versions of the 
 given user question to retrieve relevant documents from a vector database. 
 By generating multiple perspectives on the user question, your goal is to 
 help the user overcome some of the limitations of the distance-based 
 similarity search. Provide these alternative questions separated by 
 newlines. Original question: {question}""")

def parse_queries_output(message):
    return message.content.split('\n')

query_gen = perspectives_prompt | llm | parse_queries_output
```

*JavaScript*

```py
const perspectivesPrompt = ChatPromptTemplate.fromTemplate(`You are an AI 
 language model assistant. Your task is to generate five different versions 
 of the given user question to retrieve relevant documents from a vector 
 database. By generating multiple perspectives on the user question, your 
 goal is to help the user overcome some of the limitations of the 
 distance-based similarity search. Provide these alternative questions 
 separated by newlines. Original question: {question}`)

const queryGen = perspectivesPrompt.pipe(llm).pipe(message => {
  return message.content.split('\n')
})
```

注意提示模板是设计用来根据用户的初始查询生成问题变体的。

接下来，我们取生成的查询列表，并行检索每个查询的最相关文档，然后将它们组合以获得所有检索到的相关文档的唯一并集：

*Python*

```py
def get_unique_union(document_lists):
    # Flatten list of lists, and dedupe them
    deduped_docs = {
        doc.page_content: doc
        for sublist in document_lists for doc in sublist
    }
    # return a flat list of unique docs
    return list(deduped_docs.values())

retrieval_chain = query_gen | retriever.batch | get_unique_union
```

*JavaScript*

```py
const retrievalChain = queryGen
  .pipe(retriever.batch.bind(retriever))
  .pipe(documentLists => {
    const dedupedDocs = {}
    documentLists.flat().forEach(doc => {
      dedupedDocs[doc.pageContent] = doc
    })
    return Object.values(dedupedDocs)
  })
```

由于我们从同一个检索器中检索多个（相关）查询，因此很可能其中至少有一些是重复的。在使用它们作为回答问题的上下文之前，我们需要去重，以确保每个实例都是唯一的。在这里，我们通过使用它们的内容（一个字符串）作为字典（或 JS 中的对象）中的键来进行去重，因为字典只能为每个键包含一个条目。在迭代完所有文档后，我们只需简单地获取所有字典值，现在这些值已经没有重复了。

注意我们也使用了`.batch`，它并行运行所有生成的查询，并返回一个结果列表——在这种情况下，是一个文档列表的列表，然后我们将其展平并去重，如前所述。

最后一步是构建一个提示，包括用户的问题和组合检索到的相关文档，以及一个模型接口来生成预测：

*Python*

```py
prompt = ChatPromptTemplate.from_template("""Answer the following question based 
 on this context:

{context}

Question: {question}
""")

@chain
def multi_query_qa(input):
    # fetch relevant documents 
    docs = retrieval_chain.invoke(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

# run
multi_query_qa.invoke("""Who are some key figures in the ancient greek history 
 of philosophy?""")
```

*JavaScript*

```py
const prompt = ChatPromptTemplate.fromTemplate(`Answer the following 
 question based on this context:

{context}

Question: {question}
`)

const multiQueryQa = RunnableLambda.from(async input => {
  // fetch relevant documents
  const docs = await retrievalChain.invoke(input)
  // format prompt
  const formatted = await prompt.invoke({context: docs, question: input})
  // generate answer
  const answer = await llm.invoke(formatted)
  return answer
})

await multiQueryQa.invoke(`Who are some key figures in the ancient greek 
 history of philosophy?`)
```

注意这并不与我们的先前的 QA 链有很大不同，因为所有新的多查询检索逻辑都包含在`retrieval_chain`中。这是有效利用这些技术的关键——将每个技术实现为一个独立的链（在这种情况下，`retrieval_chain`），这使得它们易于采用，甚至可以组合使用。

## RAG-Fusion

RAG-Fusion 策略与多查询检索策略相似，但我们将对所有检索到的文档应用一个最终的重新排序步骤.^(3) 这个重新排序步骤使用了*互逆排名融合*（RRF）算法，该算法涉及结合不同搜索结果的排名以产生单一的、统一的排名。通过结合来自不同查询的排名，我们将最相关的文档拉到最终列表的顶部。RRF 非常适合结合可能具有不同规模或分数分布的查询的结果。

让我们通过代码演示 RAG-Fusion。首先，我们创建一个与多查询检索策略相似的提示，以根据用户查询生成查询列表：

*Python*

```py
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt_rag_fusion = ChatPromptTemplate.from_template("""You are a helpful 
 assistant that generates multiple search queries based on a single input 
 query. \n
 Generate multiple search queries related to: {question} \n
 Output (4 queries):""")

def parse_queries_output(message):
    return message.content.split('\n')

llm = ChatOpenAI(temperature=0)

query_gen = prompt_rag_fusion | llm | parse_queries_output
```

*JavaScript*

```py
import {ChatPromptTemplate} from '@langchain/core/prompts';
import {ChatOpenAI} from '@langchain/openai';
import {RunnableLambda} from '@langchain/core/runnables';

const perspectivesPrompt = ChatPromptTemplate.fromTemplate(`You are a helpful 
 assistant that generates multiple search queries based on a single input 
 query. \n
 Generate multiple search queries related to: {question} \n
 Output (4 queries):`)

const queryGen = perspectivesPrompt.pipe(llm).pipe(message => {
  return message.content.split('\n')
})
```

一旦我们生成了我们的查询，我们就为每个查询检索相关文档，并将它们传递给一个函数以*重新排序*（即根据相关性重新排序）最终的文档列表。

`reciprocal_rank_fusion`函数接受每个查询的搜索结果列表，即文档列表的列表，其中每个内部文档列表按其与该查询的相关性排序。然后，RRF 算法根据文档在不同列表中的排名（或位置）计算每个文档的新分数，并对其进行排序以创建最终的重新排序列表。

在计算融合分数后，该函数按这些分数的降序对文档进行排序，以获得最终的重新排序列表，然后将其返回：

*Python*

```py
def reciprocal_rank_fusion(results: list[list], k=60):
    """reciprocal rank fusion on multiple lists of ranked documents 
 and an optional parameter k used in the RRF formula
 """

    # Initialize a dictionary to hold fused scores for each document
    # Documents will be keyed by their contents to ensure uniqueness
    fused_scores = {}
    documents = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list,
        # with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Use the document contents as the key for uniqueness
            doc_str = doc.page_content
            # If the document hasn't been seen yet,
            # - initialize score to 0
            # - save it for later
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            # Update the score of the document using the RRF formula:
            # 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order 
    # to get the final reranked results
    reranked_doc_strs = sorted(
        fused_scores, key=lambda d: fused_scores[d], reverse=True
    )
    # retrieve the corresponding doc for each doc_str
    return [
        documents[doc_str]
        for doc_str in reranked_doc_strs
    ]

retrieval_chain = generate_queries | retriever.batch | reciprocal_rank_fusion
```

*JavaScript*

```py
function reciprocalRankFusion(results, k = 60) {
  // Initialize a dictionary to hold fused scores for each document
  // Documents will be keyed by their contents to ensure uniqueness
  const fusedScores = {}
  const documents = {}

  results.forEach(docs => {
    docs.forEach((doc, rank) => {
      // Use the document contents as the key for uniqueness
      const key = doc.pageContent
      // If the document hasn't been seen yet,
      // - initialize score to 0
      // - save it for later
      if (!(key in fusedScores)) {
        fusedScores[key] = 0
        documents[key] = 0
      }
      // Update the score of the document using the RRF formula:
      // 1 / (rank + k)
      fusedScores[key] += 1 / (rank + k)
    })
  })

  // Sort the documents based on their fused scores in descending order 
  // to get the final reranked results
  const sorted = Object.entries(fusedScores).sort((a, b) => b[1] - a[1])
  // retrieve the corresponding doc for each key
  return sorted.map(([key]) => documents[key])
}

const retrievalChain = queryGen
  .pipe(retriever.batch.bind(retriever))
  .pipe(reciprocalRankFusion)
```

注意，该函数还接受一个`k`参数，该参数决定了每个查询结果集中文档对最终文档列表的影响程度。值越高表示排名较低的文档对最终列表的影响越大。

最后，我们将我们新的检索链（现在使用 RRF）与之前看到的完整链结合：

*Python*

```py
prompt = ChatPromptTemplate.from_template("""Answer the following question based 
 on this context:

{context}

Question: {question}
""")

llm = ChatOpenAI(temperature=0)

@chain
def multi_query_qa(input):
    # fetch relevant documents 
    docs = retrieval_chain.invoke(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

multi_query_qa.invoke("""Who are some key figures in the ancient greek history 
 of philosophy?""")
```

*JavaScript*

```py
const rewritePrompt = ChatPromptTemplate.fromTemplate(`Answer the following 
 question based on this context:

{context}

Question: {question}
`)

const llm = new ChatOpenAI({temperature: 0})

const multiQueryQa = RunnableLambda.from(async input => {
  // fetch relevant documents
  const docs = await retrievalChain.invoke(input)
  // format prompt
  const formatted = await prompt.invoke({context: docs, question: input})
  // generate answer
  const answer = await llm.invoke(formatted)
  return answer
})

await multiQueryQa.invoke(`Who are some key figures in the ancient greek 
 history of philosophy?`)
```

RAG-Fusion 的优势在于其捕捉用户意图表达、导航复杂查询和扩展检索文档范围的能力，从而实现偶然发现。

## 假设文档嵌入

*假设文档嵌入*（HyDE）是一种策略，涉及根据用户查询创建一个假设文档，嵌入该文档，并根据向量相似性检索相关文档.^(4) HyDE 背后的直觉是，由 LLM 生成的假设文档将比原始查询更接近最相关的文档，如图 3-7 所示。

![2024-02-12 下午 1:12:45 的屏幕截图](img/lelc_0307.png)

###### 图 3-7\. HyDE 在向量空间中比普通查询嵌入更接近文档嵌入的示意图

首先，定义一个提示以生成一个假设文档：

*Python*

```py
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt_hyde = ChatPromptTemplate.from_template("""Please write a passage to 
 answer the question.\n Question: {question} \n Passage:""")

generate_doc = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser() 
)
```

*JavaScript*

```py
import {ChatOpenAI} from '@langchain/openai'
import {ChatPromptTemplate} from '@langchain/core/prompts'
import {RunnableLambda} from '@langchain/core/runnables';

const prompt = ChatPromptTemplate.fromTemplate(`Please write a passage to 
 answer the question
Question: {question}
Passage:`)

const llm = new ChatOpenAI({temperature: 0})

const generateDoc = prompt.pipe(llm).pipe(msg => msg.content)
```

接下来，我们将假设文档作为输入传递给`retriever`，它将生成其嵌入并搜索向量存储中的相似文档：

*Python*

```py
retrieval_chain `=` generate_doc `|` retriever 
```

*JavaScript*

```py
const retrievalChain = generateDoc.pipe(retriever)
```

最后，我们获取检索到的文档，将它们作为上下文传递给最终提示，并指示模型生成输出：

*Python*

```py
prompt = ChatPromptTemplate.from_template("""Answer the following question based 
 on this context:

{context}

Question: {question}
""")

llm = ChatOpenAI(temperature=0)

@chain
def qa(input):
  # fetch relevant documents from the hyde retrieval chain defined earlier
  docs = retrieval_chain.invoke(input)
  # format prompt
  formatted = prompt.invoke({"context": docs, "question": input})
  # generate answer
  answer = llm.invoke(formatted)
  return answer

qa.invoke("""Who are some key figures in the ancient greek history of 
 philosophy?""")
```

*JavaScript*

```py
const prompt = ChatPromptTemplate.fromTemplate(`Answer the following 
 question based on this context:

{context}

Question: {question}
`)

const llm = new ChatOpenAI({temperature: 0})

const qa = RunnableLambda.from(async input => {
  // fetch relevant documents from the hyde retrieval chain defined earlier
  const docs = await retrievalChain.invoke(input)
  // format prompt
  const formatted = await prompt.invoke({context: docs, question: input})
  // generate answer
  const answer = await llm.invoke(formatted)
  return answer
})

await qa.invoke(`Who are some key figures in the ancient greek history of 
 philosophy?`)
```

为了回顾本节中我们所学的内容，查询转换包括以下步骤：获取用户的原始查询并执行以下操作：

+   将查询重写为一个或多个查询

+   将这些查询的结果合并成一个包含最相关结果的单一集合

查询重写可以采取多种形式，但通常是以类似的方式进行的：获取用户的原始查询——您编写的提示——并要求 LLM 编写新的查询或查询。一些典型的更改示例包括：

+   从查询中删除无关或不相关的文本。

+   将查询与过去的对话历史联系起来。例如，为了理解像*关于洛杉矶怎么样*这样的查询，我们需要将其与关于旧金山天气的假设性问题结合起来，以得到一个有用的查询，如*洛杉矶的天气*。

+   通过也检索相关查询的文档来扩大检索相关文档的范围。

+   将复杂的问题分解成多个更简单的问题，然后在最终提示中包含所有这些问题的结果以生成答案。

应用的正确重写策略将取决于您的用例。

现在我们已经介绍了主要的查询转换策略，让我们讨论构建一个健壮的 RAG 系统时需要回答的第二个主要问题：我们如何路由查询以从多个数据源检索相关数据？

# 查询路由

虽然使用单个向量存储很有用，但所需的数据可能存在于各种数据源中，包括关系数据库或其他向量存储。

例如，您可能有两个向量存储：一个用于 LangChain Python 文档，另一个用于 LangChain JS 文档。给定一个用户的问题，我们希望将查询*路由*到适当推断的数据源以检索相关文档。*查询路由*是将用户的查询转发到相关数据源的一种策略。

## 逻辑路由

在*逻辑路由*中，我们向 LLM 提供我们可用的各种数据源的知识，然后让 LLM 根据用户的查询推理出应用哪个数据源，如图图 3-8 所示。

![大脑的示意图  描述由自动生成](img/lelc_0308.png)

###### 图 3-8\. 查询路由到相关数据源

为了实现这一点，我们使用像 GPT-3.5 Turbo 这样的函数调用模型来帮助将每个查询分类到可用的路由之一。*函数调用*涉及定义一个模型可以使用的模式，该模式可以根据查询生成函数的参数。这使得我们能够生成可以用于运行其他函数的结构化输出。以下 Python 代码定义了基于三种不同语言的三个文档的路由器模式：

*Python*

```py
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,
        description="""Given a user question, choose which datasource would be 
 most relevant for answering their question""",
    )

# LLM with function call 
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt 
system = """You are an expert at routing a user question to the appropriate data 
 source.

Based on the programming language the question is referring to, route it to the 
 relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define router 
router `=` prompt `|` structured_llm
```

*JavaScript*

```py
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const routeQuery = z.object({
  datasource: z.enum(["python_docs", "js_docs"]).describe(`Given a user 
 question, choose which datasource would be most relevant for answering 
 their question`),
}).describe("Route a user query to the most relevant datasource.")

const llm = new ChatOpenAI({model: "gpt-3.5-turbo", temperature: 0})
const structuredLlm = llm.withStructuredOutput(routeQuery, {name: "RouteQuery"})

const prompt = ChatPromptTemplate.fromMessages([
  ['system', `You are an expert at routing a user question to the appropriate 
 data source.

Based on the programming language the question is referring to, route it to 
 the relevant data source.`],
  ['human', '{question}']
])

const router = prompt.pipe(structuredLlm)
```

现在我们调用 LLM 根据预定义的模式提取数据源：

*Python*

```py
question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

result = router.invoke({"question": question})

result.datasource
# "python_docs"
```

*JavaScript*

```py
const question = `Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
`

await router.invoke({ question })
```

*输出：*

```py
{
    datasource: "python_docs"
}
```

注意 LLM 如何生成符合我们之前定义的模式的 JSON 输出。这将在许多其他任务中非常有用。

一旦我们提取了相关的数据源，我们就可以将值传递给另一个函数以执行所需的附加逻辑：

*Python*

```py
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here 
        return "chain for python_docs"
    else:
        ### Logic here 
        return "chain for js_docs"

full_chain = router | RunnableLambda(choose_route)
```

*JavaScript*

```py
function chooseRoute(result) {
  if (result.datasource.toLowerCase().includes('python_docs')) {
    return 'chain for python_docs';
  } else {
    return 'chain for js_docs';
  }
} 

const fullChain = router.pipe(chooseRoute) 
```

注意我们并没有进行精确的字符串比较，而是首先将生成的输出转换为小写，然后进行子字符串匹配。这使得我们的链对 LLM 偏离脚本并产生不完全符合我们请求的模式输出具有更强的鲁棒性。

###### 小贴士

对 LLM 输出随机性的鲁棒性是构建 LLM 应用程序时需要牢记的重要主题。

逻辑路由最适合当你有一个定义明确的数据源列表，可以从其中检索相关数据，并由 LLM 利用以生成准确输出时。这些可以是从向量存储到数据库甚至 API 的范围。

## 语义路由

与逻辑路由不同，*语义路由*涉及将代表各种数据源的提示嵌入到用户的查询中，然后执行向量相似度搜索以检索最相似的提示。图 3-9 展示了这一点。

![一个笔记本和大脑的示意图  自动生成的描述](img/lelc_0309.png)

###### 图 3-9\. 语义路由以提高检索文档的准确性

以下是一个语义路由的示例：

*Python*

```py
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Two prompts
physics_template = """You are a very smart physics professor. You are great at 
 answering questions about physics in a concise and easy-to-understand manner. 
 When you don't know the answer to a question, you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering 
 math questions. You are so good because you are able to break down hard 
 problems into their component parts, answer the component parts, and then 
 put them together to answer the broader question.

Here is a question:
{query}"""

# Embed prompts
embeddings = OpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# Route question to prompt
@chain
def prompt_router(query):
    # Embed question
    query_embedding = embeddings.embed_query(query)
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    # Pick the prompt most similar to the input question
    most_similar = prompt_templates[similarity.argmax()]
    return PromptTemplate.from_template(most_similar)

semantic_router = (
    prompt_router
    | ChatOpenAI()
    | StrOutputParser()
)

print(semantic_router.invoke("What's a black hole"))
```

*JavaScript*

```py
import {cosineSimilarity} from '@langchain/core/utils/math'
import {ChatOpenAI, OpenAIEmbeddings} from '@langchain/openai'
import {PromptTemplate} from '@langchain/core/prompts'
import {RunnableLambda} from '@langchain/core/runnables';

const physicsTemplate = `You are a very smart physics professor. You are great 
 at answering questions about physics in a concise and easy-to-understand 
 manner. When you don't know the answer to a question, you admit that you 
 don't know.

Here is a question:
{query}`

const mathTemplate = `You are a very good mathematician. You are great at 
 answering math questions. You are so good because you are able to break down 
 hard problems into their component parts, answer the component parts, and 
 then put them together to answer the broader question.

Here is a question:
{query}`

const embeddings = new OpenAIEmbeddings()

const promptTemplates = [physicsTemplate, mathTemplate]
const promptEmbeddings = await embeddings.embedDocuments(promptTemplates)

const promptRouter = RunnableLambda.from(query => {
  // Embed question
  const queryEmbedding = await embeddings.embedQuery(query)
  // Compute similarity
  const similarities = cosineSimilarity([queryEmbedding], promptEmbeddings)[0]
  // Pick the prompt most similar to the input question
  const mostSimilar = similarities[0] > similarities[1] 
    ? promptTemplates[0] 
    : promptTemplates[1]
  return PromptTemplate.fromTemplate(mostSimilar)
})

const semanticRouter = promptRouter.pipe(new ChatOpenAI())

await semanticRouter.invoke("What's a black hole")
```

现在您已经看到了如何将用户的查询路由到相关的数据源，让我们讨论构建健壮 RAG 系统时的第三个主要问题：“我们如何将自然语言转换为目标数据源的查询语言？”

# 查询构建

如前所述，RAG 是一种有效的策略，可以根据查询从基于向量的存储中嵌入和检索相关的非结构化数据。但大多数可用于生产应用程序的数据是结构化的，通常存储在关系数据库中。此外，嵌入在向量存储中的非结构化数据还包含具有重要信息的结构化元数据。

*查询构建*是将自然语言查询转换为与您交互的数据库或数据源的查询语言的过程。参见图 3-10。

![数据类型的示意图  自动生成的描述](img/lelc_0310.png)

###### 图 3-10\. 各种数据源的查询语言示意图

例如，考虑以下查询：*1980 年关于外星人的电影是什么？*这个问题包含一个可以通过嵌入（*外星人*）检索的非结构化主题，但它还包含潜在的有序组件（*年份 == 1980*）。

以下章节将更深入地探讨查询构建的各种形式。

## 文本到元数据过滤器

大多数向量存储都提供了基于元数据限制向量搜索的能力。在嵌入过程中，我们可以将元数据键值对附加到索引中的向量上，然后在查询索引时指定过滤器表达式。

LangChain 提供了一个 `SelfQueryRetriever`，它抽象了这种逻辑，并使得将自然语言查询转换为针对各种数据源的格式化查询变得更加容易。自查询利用一个 LLM 根据用户的查询和预定义的元数据模式提取和执行相关的元数据过滤器：

*Python*

```py
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

fields = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
description = "Brief summary of a movie"

llm = ChatOpenAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm, db, description, fields,
)

print(retriever.invoke(
    "What's a highly rated (above 8.5) science fiction film?"))
```

*JavaScript*

```py
import { ChatOpenAI } from "@langchain/openai";
import { SelfQueryRetriever } from "langchain/retrievers/self_query";
import { FunctionalTranslator } from "@langchain/core/structured_query";

/**
 * First, we define the attributes we want to be able to query on.
 * in this case, we want to be able to query on the genre, year, director, 
 * rating, and length of the movie.
 * We also provide a description of each attribute and the type of the attribute.
 * This is used to generate the query prompts.
 */
const fields = [
  {
    name: "genre",
    description: "The genre of the movie",
    type: "string or array of strings",
  },
  {
    name: "year",
    description: "The year the movie was released",
    type: "number",
  },
  {
    name: "director",
    description: "The director of the movie",
    type: "string",
  },
  {
    name: "rating",
    description: "The rating of the movie (1-10)",
    type: "number",
  },
  {
    name: "length",
    description: "The length of the movie in minutes",
    type: "number",
  },
];
const description = "Brief summary of a movie";

const llm = new ChatOpenAI();
const attributeInfos = fields.map((field) => new AttributeInfo(field.name,  
  field.description, field.type));

const selfQueryRetriever = SelfQueryRetriever.fromLLM({
  llm,
  db,
  description,
  attributeInfo: attributeInfos,
  /**
 * We need to use a translator that translates the queries into a
 * filter format that the vector store can understand. LangChain provides one 
 * here.
 */
  structuredQueryTranslator: new FunctionalTranslator(),
});

await selfQueryRetriever.invoke(
  "What's a highly rated (above 8.5) science fiction film?"
);
```

这导致检索器将用户查询分成：

+   首先应用于每个文档元数据的过滤器

+   用于在文档上进行语义搜索的查询

要做到这一点，我们必须描述我们的文档元数据包含哪些字段；该描述将包含在提示中。检索器接下来将执行以下操作：

1.  将查询生成提示发送到 LLM。

1.  解析来自 LLM 输出的元数据过滤器和重写的搜索查询。

1.  将 LLM 生成的元数据过滤器转换为适合我们向量存储的格式。

1.  对向量存储执行相似度搜索，仅匹配元数据通过生成的过滤器的文档。

## 文本到 SQL

SQL 和关系数据库是结构化数据的重要来源，但它们不直接与自然语言交互。尽管我们可以简单地使用 LLM 将用户的查询翻译成 SQL 查询，但错误的空间很小。

这里有一些有效的文本到 SQL 转换策略：

数据库描述

为了使 SQL 查询具体化，LLM 必须提供数据库的准确描述。一种常见的文本到 SQL 提示采用本文和其他论文中报道的想法：为每个表提供 `CREATE TABLE` 描述，包括列名和类型.^(5) 我们还可以提供一些（例如，三个）表中的示例行。

少样本示例

通过提供问题-查询匹配的少样本示例来填充提示，可以提高查询生成的准确性。这可以通过在提示中简单地附加标准静态示例来实现，以指导代理如何根据问题构建查询。

见 图 3-11 以了解过程的视觉表示。

![机器的特写，描述自动生成](img/lelc_0311.png)

###### 图 3-11\. 将用户查询转换为 SQL 查询

这里有一个完整的代码示例：

*Python*

```py
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

# replace this with the connection details of your db
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(model="gpt-4", temperature=0)

# convert question to sql query
write_query = create_sql_query_chain(llm, db)

# Execute SQL query
execute_query = QuerySQLDatabaseTool(db=db)

# combined
chain = write_query | execute_query

# invoke the chain
chain.invoke('How many employees are there?');
```

*JavaScript*

```py
import { ChatOpenAI } from "@langchain/openai";
import { createSqlQueryChain } from "langchain/chains/sql_db";
import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";
import { QuerySqlTool } from "langchain/tools/sql";

const datasource = new DataSource({
  type: "sqlite",
  database: "./Chinook.db", // here should be the details of your database
});
const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
});
const llm = new ChatOpenAI({ model: "gpt-4", temperature: 0 });

// convert question to sql query
const writeQuery = await createSqlQueryChain({ llm, db, dialect: "sqlite" });

// execute query
const executeQuery = new QuerySqlTool(db);

// combined
const chain = writeQuery.pipe(executeQuery);

// invoke the chain
await chain.invoke('How many employees are there?');
```

我们首先将用户的查询转换为适合我们数据库方言的 SQL 查询。然后我们在数据库上执行该查询。请注意，在生产应用程序中，由 LLM 从用户输入生成的任意 SQL 查询在数据库上执行是危险的。要在生产中使用这些想法，你需要考虑一系列安全措施以降低意外查询在数据库中运行的风险。以下是一些示例：

+   使用具有只读权限的用户在你的数据库上运行查询。

+   运行查询的数据库用户应只能访问你希望用于查询的表。

+   为此应用程序运行的查询添加超时；这将确保即使生成昂贵的查询，它也会在占用太多数据库资源之前被取消。

这并不是安全考虑的完整列表。LLM 应用的安全是一个目前处于开发中的领域，随着新漏洞的发现，更多的安全措施被添加到建议中。

# 摘要

本章讨论了各种最先进策略，以高效地根据用户的查询检索最相关的文档，并使用你的提示将它们综合起来，以帮助 LLM 生成准确、最新的输出。

如前所述，一个健壮、生产就绪的 RAG 系统需要一系列有效的策略，这些策略可以执行查询转换、查询构建、路由和索引优化。

查询转换使你的 AI 应用能够将模糊或格式错误的用户查询转换为代表性查询，该查询对检索是最优的。查询构建使你的 AI 应用能够将用户的查询转换为存储结构化数据的数据库或数据源的查询语言语法。路由使你的 AI 应用能够动态地将用户的查询路由到相关数据源以检索相关信息。

在第四章中，我们将在此基础上构建知识，为你的 AI 聊天机器人添加记忆功能，使其能够记住并从每次交互中学习。这将使用户能够像 ChatGPT 一样进行多轮对话。

^(1) Patrick Lewis 等人，[“检索增强生成用于知识密集型 NLP 任务”](https://oreil.ly/Qzd2K)，arXiv，2021 年 4 月 12 日。

^(2) Xinbei Ma 等人，[“检索增强大型语言模型的查询重写”](https://oreil.ly/zyw5E)，arXiv，2023 年 10 月 23 日。由微软亚洲研究院资助的研究。

^(3) Zackary Rackauckas，[“RAG-Fusion：检索增强生成的新视角”](https://oreil.ly/k7TTY)，arXiv，2024 年 2 月 21 日。来自《国际自然语言计算杂志》，第 13 卷，第 1 期（2024 年 2 月）。

^(4) 高露宇等，[“无相关性标签的精确零样本密集检索”](https://oreil.ly/7aTnS)，arXiv，2022 年 12 月 20 日。

^(5) 拉吉库马尔·尼塔沙恩等，[“评估大型语言模型的文本到 SQL 能力”](https://oreil.ly/WOrzt)，arXiv，2022 年 3 月 15 日。
