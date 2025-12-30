# Chapter 12\. Retrieval-Augmented Generation

In [Chapter 10](ch10.html#ch10), we demonstrated how to vastly expand the capabilities of LLMs by interfacing them with external data and software. In [Chapter 11](ch11.html#chapter_llm_interfaces), we introduced the concept of embedding-based retrieval, a foundational technique for retrieving relevant data from data stores in response to queries. Armed with this knowledge, let’s explore the application paradigm of augmenting LLMs with external data, called retrieval-augmented generation (RAG), in a holistic fashion.

In this chapter, we will take a comprehensive view of the RAG pipeline, diving deep into each of the steps that make up a typical workflow of a RAG application. We will explore the various decisions involved in operationalizing RAG, including what kind of data we can retrieve, how to retrieve it, and when to retrieve it. We will highlight how RAG can help not only during model inference but also during model training and fine-tuning. We will also compare RAG with other paradigms and discuss scenarios where RAG shines in comparison to alternatives or vice versa.

# The Need for RAG

As introduced in [Chapter 10](ch10.html#ch10), RAG is an umbrella term used to describe a variety of techniques for using external data sources to augment the capabilities of an LLM. Here are some reasons we might want to use RAG:

*   We need the LLMs to access our private/proprietary data, or data that was not part of the LLM’s pre-training datasets. Using RAG is a much more lightweight option than pre-training an LLM on our private data.

*   To reduce the risk of hallucinations, we would like the LLM to refer to data provided through a retrieval mechanism rather than rely on its own internal knowledge. RAG facilitates this. RAG also enables more accurate data citations, connecting LLM outputs to their ground-truth sources.

*   We would like the LLM to answer questions about recent events and concepts that have emerged after the LLM was pre-trained. While there are memory editing techniques for updating LLM parameters with new knowledge like [MEMIT](https://oreil.ly/kxI3j), they are not yet reliable or scalable. As discussed in [Chapter 7](ch07.html#ch07), continually training an LLM to keep its knowledge up-to-date is expensive and risky.

*   We would like the LLM to answer queries involving long-tail entities, which occur only rarely in the pre-training datasets.

# Typical RAG Scenarios

Now that we have seen *why* we need RAG, let’s explore *where* we can utilize it. The four most popular scenarios are:

Retrieving external knowledge

This is the predominant use case that has seen a lot of success with productionization. As discussed earlier in the chapter, we can use RAG to plug LLM knowledge gaps or to reduce hallucination risk.

Retrieving context history

LLMs have a limited context window, but often we need access to more context in order to answer a query than what fits in the context window. We would also like to have longer conversations with the LLM than what fits in the context window. In these cases, we could retrieve parts of the conversation history or session context when needed.

Retrieving in-context training examples

Few-shot learning is an effective approach to help LLMs get acquainted with the input-output mapping of a task. You can make few-shot learning more effective by dynamically selecting few-shot examples based on the current input. The few-shot examples can be retrieved from a training example data store at inference time.

Retrieving tool-related information

As described in [Chapter 10](ch10.html#ch10), LLMs can invoke software tools as part of their workflow. The list of tools available and their description is stored in a tool store. The LLM can then use retrieval for tool selection, selecting the tool best suited to the task. Tool-related information can also include API documentation, for instance.

# Deciding When to Retrieve

For each step in an agentic workflow, the LLM can advance its task using one of the following steps:

*   Use its internal capabilities

*   Choose from among several data stores

*   Choose from among several software tools

There can be tasks that the LLM can fully solve using its parametric memory, but one or more data stores may also contain the requisite data needed to solve them. In these cases, should we just default to using RAG, given all its benefits that we presented earlier?

We have seen earlier in the chapter that LLMs struggle with long-tail information, and that RAG can be an effective means to answer questions about long-tail entities. However, [Mallen et al.](https://oreil.ly/MF7Y1) show that for queries about more popular entities, the LLM might sometimes be better at answering queries than RAG. This is because of the inevitable limitations of the retrieval model, which might retrieve irrelevant or incorrect information that could mislead the LLM.

For a given query, you can dynamically determine whether to use retrieval or to rely on the LLM’s parametric memory. The rules determining the right approach to take include:

*   Whether the query is about a more frequently occurring entity. For example, the LLM is more likely to memorize the birthday of Taylor Swift than of a substitute drummer of a local band whose Wikipedia page is a stub.

*   Whether the query has timeliness constraints, i.e., if the data needed to address the query may not have existed before the LLM’s knowledge cutoff date.

*   Whether the model has been continually pre-trained or memory tuned as described in [Chapter 7](ch07.html#ch07), and the given query relates to concepts over which the training was performed.

If you are using LLMs for general-purpose question answering, Mallen et al. show that you can use sources like Wikipedia as a pseudo-popularity metric for entities. If the entities present in your inputs have an entity count in Wikipedia greater than a threshold, then the LLM can choose to answer the question on its own without using RAG. Note that the threshold can change across LLMs. This strategy works only if you have a good understanding about the datasets the LLM has been pre-trained on.

Dynamically deciding when to retrieve data can also help optimize the model’s latency and responsiveness, as the RAG pipeline will introduce additional overhead.

###### Tip

Dynamic retrieval is mostly useful when you are using very large LLMs. For smaller models (7B or below), it is almost always beneficial to prefer using RAG rather than relying on the LLM’s internal memory.

# The RAG Pipeline

A typical RAG application follows the *retrieve-read* framework, as discussed in [Chapter 11](ch11.html#chapter_llm_interfaces). In response to a query, a retrieval model identifies documents that are relevant to answering the query. These documents are then passed along to the LLM as context, which the LLM can rely on in addition to its internal capabilities to generate a response. In practice, we typically need to add a lot of bells and whistles to get RAG working in a production context. This involves adding several more optional stages to the retrieve-read framework. In practice, your pipeline stages might consist of a *rewrite-retrieve-read-refine-insert-generate* workflow, with some of these steps potentially comprising multiple stages. Later in the chapter, we will go through each of the steps in more detail.

[Figure 12-1](#RAG-pipeline) shows the various stages of the RAG pipeline and the components involved.

![RAG-pipeline](assets/dllm_1201.png)

###### Figure 12-1\. RAG pipeline

###### Tip

As in the rest of the book, we refer to user or LLM requests to retrieve data as queries, and units of text retrieved from the data store as documents.

Let’s illustrate with an example. Consider a RAG application that answers questions about Canadian politics and parliamentary activity. The application has access to a knowledge base containing transcripts of parliamentary proceedings. We will assume that the data is represented using the representation techniques described in [Chapter 11](ch11.html#chapter_llm_interfaces).

When a user issues a query, we might want to rephrase it before sending it to the retriever. Traditionally in the field of information retrieval (IR), this is referred to as query expansion. Query expansion is especially useful because of the vocabulary mismatch between the query space and the document space. The user might use different terminology in the query than that used in the documents. Rephrasing a query can help bridge the vocabulary gap. In general, we would like to rephrase the query in such a way that it improves the chances of the retriever fetching the most relevant documents. This stage is called the *rewrite* stage.

Next, in the *retrieve* stage, a retrieval model is used to retrieve the documents relevant to the query. In [Chapter 11](ch11.html#chapter_llm_interfaces), we discussed embedding-based retrieval, a popular retrieval paradigm in the LLM era. The retrieval stage can be an extensive multi-stage pipeline.

The retrieval can happen over a very large document space. In this case, it is computationally infeasible to use more advanced retrieval models. Therefore, retrieval is usually carried out in a two-step process, with the first step using faster methods (these days, typically embedding-based) to retrieve a list of potentially relevant documents (optimizing recall), and a second step that reranks the retrieved list based on relevance (optimizing precision) so that the top-k ranked documents are then taken as the context to be passed along to the LLM. This stage is called the *rerank* stage.

After identifying the top-k documents relevant to the query, they need to be passed along to the LLM. However, the documents may not fit into the context window and thus need to be shortened. They also could potentially be rephrased in a way that makes it more likely for the LLM to use the context to generate the answer. This is done during the *refine* stage.

Next, we provide the output of the refine step to the LLM. The default approach is to concatenate all the documents in the prompt. However, you could also pass them one at a time, and then ensemble the results. How the documents are ordered in the prompt can also make a difference. Several such techniques determine the way the context is fed to the LLM. This is called the *insert* stage.

Finally, in the *generate* stage, the LLM reads the prompt containing the query and the context and generates the response. The generation can happen all at once or the retrieval process can be interleaved with the generation, i.e., the model can generate a few tokens, then call the retrieval model again to retrieve additional content, generate a few more tokens, and then call the retrieval model again, and so on.

The output of each stage can be run through a *verify* stage to assess the quality of the outputs and even take corrective measures. The verify stage can employ either heuristics or AI-based methods.

In this example, the query was generated by a human user. But if we consider RAG in the context of agentic workflows, the query might be generated by an LLM. In an agentic workflow, the agent can determine at any given point that it needs to retrieve data to progress with its task, which sets the aforementioned pipeline into motion.

Apart from the retrieve and generate steps, the rest of the pipeline is optional, and including other steps depends on your performance and latency tradeoffs.

###### Note

Our example pertains to RAG when used at inference time. RAG can also be applied when pre-training or fine-tuning the model, which we will describe later in the chapter.

Let’s examine each step in the pipeline in detail.

## Rewrite

After a query is issued, it might need to be rewritten to make it more amenable to retrieval. The rewriting process depends on the retrieval models used. As mentioned before, there is usually a mismatch between the query space and the document space, as the vocabulary, phrasing, and semantics used by the query might vary drastically from how the relevant concepts are conveyed in the document.

As an example, consider the query: “Which politicians have complained about the budget not being balanced?”

and the data store contains the text “Senator Paxton: ‘I just can’t stand the sight of our enormous deficit.’”

If you are using traditional retrieval approaches that rely more on keywords, this text may not be selected as relevant during retrieval. Using embedding-based methods bridges the gap as embeddings of similar sentences are closer to each other in embedding space, but it does not entirely solve the problem.

###### Tip

If the query is coming from the user, the user might add instructions along with the query, like, “Which politicians have complained about the budget not being balanced? Provide the results in the form of a table.” In this case you will have to separate the query from the instructions before feeding the query into the retrieval pipeline. This can be done by an LLM using prompting techniques like CoT, ReAct, etc., which we discussed in Chapters [5](ch05.html#chapter_utilizing_llms) and [10](ch10.html#ch10), respectively.

For systems using traditional retrieval techniques, query rewriting is typically performed using query expansion techniques, in which the query is augmented with similar keywords. Basic query expansion techniques include adding synonyms of keywords and other topic information in your query.

A well-tested method for query expansion is pseudo-relevance feedback (PRF). In PRF, the original query is used to retrieve documents, and salient terms from these documents are extracted and added to the original query.

Let’s see how PRF would help with our query, ‘‘Which politicians have complained about the budget not being balanced?” We use a retrieval technique like BM25 (explained later in the chapter) to return a candidate set of k documents. We then use a technique like term frequency or, more effectively, [Tf-IDf](https://oreil.ly/5be9z) to extract the salient terms occurring in these returned documents. For this example the salient phrases turn out to be “fiscal policy,” “deficit,” “financial mismanagement,” and “budgetary reforms.” Adding these phrases to the original query will lead to the text:

> “Senator Paxton: ‘I just can’t stand the sight of our enormous deficit!’” being retrieved successfully.

In recent years, LLM-driven query expansion methods are gaining more prominence. Two such examples are [Query2doc](https://oreil.ly/BDJAN) and [Hypothetical Document Embeddings (HyDE)](https://oreil.ly/tZyUV).

Let’s explore how LLM-based query expansion works. We pass a given query to an LLM and ask it to generate a hypothetical document that contains the answer to the query.

> *Prompt:* For the provided query, generate a hypothetical document in 2 sentences or less that answers the query. Assume the documents are transcripts of Canadian parliamentary proceedings, containing verbatim statements of representatives in the House and Senate.
> 
> *Query:* Which politicians have complained about the budget not being balanced?

The prompt contains:

*   Instructions to generate a hypothetical document

*   Desired length limit of the generated document

*   Description of the document corpus containing relevant documents

*   The original query

The hypothetical document generated by the LLM is as follows:

> *LLM:* “Honorable Senator Mark Wellington stated, *This government’s failure to balance the budget is unacceptable and jeopardizes our economic future.* Similarly, MP Emily Fraser remarked, *We cannot continue on this path of reckless spending without addressing the deficit—it’s time for real fiscal responsibility.*”

Using this hypothetical document as the query will return the sentence ‘‘Senator Paxton: ‘I just can’t stand the sight of our enormous deficit!”’ with a higher similarity score.

While this hypothetical document is not factually accurate, and laughably so (there are no senators or MPs in Canada named Mark Wellington and Emily Fraser), it does contain verbiage and semantics very similar to what an actual politician would say. If we use this document as the query (optionally combining with the original query), then the chances of it being semantically similar to actual instances of politicians talking about the topic is higher than if matched with the query alone.

The length of the hypothetical document could be similar to the typical length of the retrieval unit. You can use a smaller LLM to generate the hypothetical document, as we do not care for factuality guarantees in this setting. However, smaller models are also not as adept as generating quality hypothetical documents, so you will have to manage the tradeoff. Both LangChain and LlamaIndex provide implementations of hypothetical document-based query rewriting.

If the model has been pre-trained or fine-tuned on the data corpus containing the relevant data, then adding descriptions of the corpus in the prompt as shown in the example will make it more likely that the generated document follows the structure, format, and linguistics of that data corpus.

###### Warning

One pitfall of query rewriting techniques is the risk of topic drift. In the case of hypothetical documents, the document may drift into irrelevant topics after the first few tokens. Upweighting the logits bias for tokens in the query can partially address this problem. PRF techniques are also susceptible to topic drift.

You can also combine PRF style techniques with hypothetical documents. Instead of generating hypothetical documents to replace or augment the query, you can use them to extract keywords that you can add to the original query. [Li et al.](https://oreil.ly/cOnMs) propose a technique called *query2document2keyword*. In this technique, the LLM generates a hypothetical document using the query, similar to HyDE. The LLM is then prompted to extract salient keywords from this document.

We can then further improve the quality of the extracted keywords by taking them through a filtering step. The authors propose using the *self-consistency* method, which we discussed in [Chapter 5](ch05.html#chapter_utilizing_llms). To recap, in the self-consistency method, we repeat the keyword generation multiple times, and then select the top keywords based on the number of generations they are present in.

Another way to combine traditional retrieval with LLM-driven query rewriting is to first return the top-k documents from the initial retrieval step, then use LLMs to generate salient keywords from the returned documents and add them to the query.

So far we have discussed techniques that bridge the query document mismatch problem by modifying the query and bringing it closer to the document space. An alternative approach to solve the mismatch problem is to represent the documents in a way that brings them closer to the query space. Examples of this approach include [doc2query](https://oreil.ly/CGUtP) and [contextual retrieval](https://oreil.ly/ZJuIu). While document rewriting techniques initially have a large cost if the data stores are very large, they can reduce latency during inference time as no or little query rewriting needs to be performed. On the other hand, query rewriting techniques are simple to implement and integrate into a RAG workflow.

Yet another form of query rewriting is called query decomposition. For complex queries in an agentic workflow, we can have the LLM divide the task into multiple queries that can be executed sequentially or in parallel, depending on how the query was decomposed. We discussed query decomposition techniques in [Chapter 10](ch10.html#ch10).

###### Note

If your external data is in a structured form like databases, then the query needs to be rewritten into a SQL query or equivalent, as discussed in [Chapter 10](ch10.html#ch10).

Now that we have discussed the query rewriting step of the pipeline, let’s move on to the retrieve step.

## Retrieve

The retrieve step is the most crucial stage of the RAG pipeline. It is easy to see why: all RAG applications are bottlenecked by the quality of retrieval. Even if you are working with the world’s best language model, you won’t be able to get the correct results if the retrieval step didn’t retrieve the correct documents needed to answer the query. Therefore, this step of the pipeline should focus on increasing recall.

Embedding-based retrieval, which we discussed in detail in [Chapter 11](ch11.html#chapter_llm_interfaces), is highly popular. However, traditional information-retrieval techniques should not be dismissed. The right technique to use depends on the expected nature of queries (can a significant proportion of them be answered by just keyword or regex match?), the expected degree of query-document vocabulary mismatch, latency and compute limitations, and performance requirements.

###### Note

The information retrieval (IR) research field has been studying these problems for a long time. Now that retrieval is more relevant than ever in NLP, I am noticing a lot of efforts to reinvent the wheel rather than reusing IR insights. For insights in retrieval research, check out papers from leading IR research conferences like SIGIR, ECIR, TREC, etc.

Embedding-based retrieval methods are not always suitable when you would like all documents containing a specific word or phrase to be retrieved. Therefore it is customary to combine keyword-based methods with embedding methods, called hybrid search. The results from the two methods are combined and fed to the next step of the retrieval pipeline. Most vector databases support hybrid search in some shape or form.

[Figure 12-2](#hybrid-search) shows the retrieval stage in action, using hybrid search.

![Hybrid-Search](assets/dllm_1202.png)

###### Figure 12-2\. Hybrid search

I also highly recommend metadata filters for improving retrieval. The more metadata you gather during the data representation and storage phase, the better the retrieval results. For example, if you have performed topic modeling of your data store in advance, you can restrict your search results to a subset of topics, with the filters being applied either using a hardcoded set of rules or determined by an LLM.

Next, let’s discuss promising recent advances in retrieval.

### Generative retrieval

What if the LLM could identify the right documents(s) that need to be retrieved in response to a query, thus removing the need for retrieval techniques? This is called generative retrieval.

Generative retrieval is implemented by assigning identifiers to documents called docIDs, and teaching the LLM the association between documents and docIDs. A document can be associated with one or more docIDs. Typical docIDs can be:

Single tokens

Each document can be represented by a new token in the vocabulary. This means that, during inference, the model needs to output only a single token for each document it wants to retrieve. [Pradeep et al.](https://oreil.ly/7JYOM) use a T5 model where the encoder vocabulary is the standard T5 vocabulary but the decoder vocabulary contains the docIDs. This approach is feasible only with a small document corpus.

Prefix/subset tokens

[Tay et al.](https://oreil.ly/1p1C8) use the first 64 tokens of a document as the docID, while [Wang et al.](https://oreil.ly/lg3g3) use 64 randomly selected contiguous tokens from the document.

Cluster tokens

You can also perform hierarchical clustering of your document corpus based on its semantics (using embeddings, for example), and the docID can be a concatenation of the cluster IDs at each level of the hierarchy.

Salient keyword tokens

The docIDs can also contain salient keywords representing the topics and themes contained in the document. For example, a document about the Transformer architecture can be represented by the docID “transformer_self-attention_architecture.”

One way to teach the LLM the association between documents and docIDs is by fine-tuning the model. This is referred to as training-based indexing. However, fine-tuning needs a lot of resources and is not suitable in scenarios in which new documents are frequently added to the corpus.

[Askari et al.](https://oreil.ly/K5TAB) show that we can use few-shot learning to build a generative retrieval system without needing to train the model. First, for each document in the corpus, pseudo queries are generated using a language model. The pseudo queries are the queries whose answers are present in the document. These pseudo queries are then fed to a language model in a few-shot setting and asked to generate docIDs. [Figure 12-3](#generative-retrieval) shows training-free generative retrieval in action.

![Generative-Retrieval](assets/dllm_1203.png)

###### Figure 12-3\. Generative retrieval

During inference, the model is provided with a query similar to the setup in [Figure 12-3](#generative-retrieval) and asked to generate the correct docID(s) that are relevant to the query. Constrained beam search is used to ensure that the docID generated by the model corresponds to a valid docID in the corpus.

###### Tip

You can also use generative retrieval to retrieve documents based on their metadata. For example, the model could ask to retrieve Apple’s 2024 annual report. The model can be made to generate the right identifier by either fine-tuning the model or using few-shot learning, as shown in this section.

Ultimately, generative retrieval is suitable only if your document corpus is relatively small, there is limited redundancy within the corpus, or the documents belong to a set of well-defined categories (annual reports of all public companies in the US, for instance).

Next, let’s discuss tightly-coupled retrievers, another new topic in the retrieval space.

### Tightly-coupled retrievers

As seen in [Chapter 11](ch11.html#chapter_llm_interfaces), in embedding-based retrieval, the embedding model is typically independent of the language model to which the retrieval results are fed. We will refer to them as *loosely-coupled* retrievers.

In contrast, a *tightly-coupled* retriever is trained such that it learns from LLM feedback; the model learns to retrieve text that best positions the LLM to generate the correct output for a given query. Tightly-coupled retrievers can be trained together with the generator LLM as part of a single architecture, or they can be trained separately using feedback from the trained LLM.

An example of the latter is [Zhang et al.’s LLM-Embedder](https://oreil.ly/Q__8M), a unified embedding model that can support a variety of retrieval needs in a single model, ranging from knowledge retrieval to retrieving optimal few-shot examples. The model is trained from two types of signals: a contrastive learning setup typically used to train embedding models (presented in [Chapter 11](ch11.html#chapter_llm_interfaces)) and LLM feedback. A retrieval candidate receives a larger reward from LLM feedback if it improves the performance of the LLM in answering the query.

Tightly-coupled retrievers are another tool in your toolkit for improving retrieval. They are by no means a necessary step in the RAG pipeline. As always, experimentation will show how much of a lift (if any) they provide for your application.

Finally, let’s discuss GraphRAG, an up-and-coming retrieval paradigm that leverages knowledge graphs for better retrieval.

### GraphRAG

A key limitation of the retrieval approaches we have discussed so far is their inability to facilitate answering questions that require drawing connections between different parts of the document corpus, as well as questions that involve summarizing high-level themes across the dataset. For example, all the retrieval techniques we discussed so far would do poorly on a query like, “What are the key topics discussed in this dataset?”

One way to address these limitations is by employing knowledge graphs. Microsoft released [GraphRAG](https://oreil.ly/V4n_S), a graph-based RAG system. GraphRAG works by creating a knowledge graph from the underlying data corpus by extraction entities and relationships. The graph is then used to perform hierarchical semantic clustering, with summaries generated for each cluster. These summaries enable answering of thematic questions like, “What are the key topics discussed in this dataset?”

GraphRAG requires a lot of initial compute to prepare the knowledge graph. This can be prohibitive for larger datasets. While it is easy to extract entities, extracting relevant relationships is harder.

Now that we have explored the retrieval stage of the RAG pipeline, let’s move on to the rerank stage.

## Rerank

The retrieval process can be broken into a two-stage or multi-stage process, where the initial stage retrieves a list of documents relevant to the query, followed by one or more *reranking* stages that take the documents and sort them by relevance. The reranker is generally a more complex model than the retriever and thus is run only on the retrieved results (or else we would have just used the reranker as the retriever).

The reranker is usually a language model fine-tuned on the specific use case. You can use BERT-like models for building a relevance classifier, where given a query and a document, the model outputs the probability of the document being relevant to answering the query. These models are called *cross-encoders*, as in these models the query and document are encoded together, as opposed to embedding-based retrieval models we have discussed, called bi-encoders, where the query and document are encoded as separate vectors.

The input for a BERT model acting as a cross-encoder is of the format:

```py
[CLS] query_text [SEP] document_text [SEP]
```

The Sentence Transformers library provides access to cross-encoders, which can be used as rerankers in the RAG pipeline:

```py
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", num_labels=1)

query = 'When was the Apple iPhone 15 launched?'
documents = ['Apple iPhone 15 launched with great fanfare in New York',
'He was foolish enough to believe that gifting an iPhone would
  save the relationship',
'On September 22, 2023, I lined up at the Central Park store for the launch of
  the iPhone 15']

ranks = model.rank(query, documents)
for rank in ranks:
   print(rank['score'], documents[rank['corpus_id']])
```

Because we have set `num_labels = 1`, the model will treat it as a regression task, using the sigmoid activation function to output a score between 0 and 1.

These days, more advanced models like [Contextualized Late Interaction over BERT (ColBERT)](https://oreil.ly/N3fOv) are used for reranking. As opposed to the cross-encoder setup we just discussed, ColBERT-style models allow for pre-computation of document representations, leading to faster inference.

In ColBERT, the query and documents are encoded separately using BERT, generating token-level embedding vectors for each token in the query and documents. For each token in the query, the corresponding embedding is compared to the embeddings of each of the token embeddings of the document, generating similarity scores. The maximum similarity scores for each query token are summed, resulting in the final relevance score. This type of architecture is called *late interaction*, since the query and document are not encoded together but interact together only later in the process. Late interaction saves time compared to traditional cross-encoders, as document embeddings can be created and stored in advance.

[Figure 12-4](#cross-encoders) depicts a ColBERT model in action, illustrating the late interaction between query and documents.

![cross-encodersl](assets/dllm_1204.png)

###### Figure 12-4\. ColBERT

Next, let’s explore a few advanced reranking techniques.

### Query likelihood model (QLM)

A QLM estimates the probability of generating the query given a candidate document as input. You can treat an LLM as a QLM, utilizing its zero-shot capabilities to rank candidate documents based on the query token probabilities. Alternatively, you can fine-tune an LLM on query generation tasks to improve its suitability as a QLM.

A typical prompt for a QLM would look like: “Generate a question that is most relevant to the given document <document content>”.

After getting the top-k documents relevant to a query from the retrieval stage, each document is fed to the LLM with this prompt. The likelihood of the query tokens is then calculated using the model logits. The documents are then sorted by likelihood, providing a relevance ranking.

###### Warning

[Zhuang et al.](https://oreil.ly/QnWWh) show that an instruction-tuned model that doesn’t contain query generation tasks in its instruction-tuning training set loses its capability to be an effective zero-shot QLM. This is yet another case of instruction-tuned models exhibiting degraded performance compared to base models, on tasks they have not been trained on.

Note that to calculate the probability of the query tokens, we need access to the model logits. Most proprietary model providers including OpenAI do not yet provide full access to the model logits as of this book’s writing. Thus, the LLM-as-a-QLM approach can be implemented only using open source models.

In the interest of reducing latency, you would ideally like the QLM to be as small a model as possible. However, smaller models are less effective QLMs. Effectively fine-tuning a smaller LLM for query generation might be the sweet spot.

### LLM distillation for ranking

Earlier in the chapter, we saw how encoder-only models like BERT could serve as rerankers. More recently, decoder LLMs are also being trained to directly rank candidate documents in three ways:

Pointwise ranking

Each candidate document is fed separately to the LLM. The LLM provides a Boolean judgment on its relevance. Alternatively, it can also provide a numerical score, although this is much less reliable.

Pairwise ranking

For each candidate document pair, the LLM indicates which document is more relevant. To get a complete ranking, N² such comparisons need to be made.

Listwise ranking

All the candidate documents are tagged with identifiers and fed to the LLM, and the LLM is asked to generate a ranked list of identifiers according to decreasing order of relevance of corresponding documents.

In general, pointwise ranking is the easiest to use but may not be the [most effective](https://oreil.ly/DvmtC). Listwise ranking might need a large context window, while pairwise ranking needs lots of comparisons. Pairwise ranking is the most effective of these techniques, since it involves direct comparison. [Figure 12-5](#llm-rerankers) shows how pointwise, pairwise, and listwise rankings work.

Examples of ranking LLMs include [RankGPT](https://oreil.ly/6XoOG), [RankVicuna](https://oreil.ly/00Dan), and [RankZephyr](https://oreil.ly/AAbUE).

These models are trained by distilling from larger LLMs, a technique we first learned in [Chapter 9](ch09.html#ch09). For example, the process for training RankVicuna is:

*   Queries in the training set are fed through a first-level retriever like BM25 to generate a list of candidate documents.

*   This list is passed to a larger LLM, which generates a rank-ordered list of candidates.

*   The query and the rank-ordered list are used to fine-tune the smaller LLM.

The creators of [RankVicuna](https://oreil.ly/cFLSc) show that as the effectiveness of the first-level retrieval increases, the possible performance gains from RankVicuna decreases due to diminished returns. They also reported that augmenting the dataset by shuffling the input order of the candidate documents improved model performance.

![llm-rerankers](assets/dllm_1205.png)

###### Figure 12-5\. Decoder LLM rerankers

###### Tip

You can combine the results of the retrieve and the rerank stages to get the final relevance ranking of candidate documents. This is needed to enforce keyword weighting, for example. You can also weight your relevance ranking by metadata like published date (more recent documents are weighted more).

Now that we have discussed the rerank stage, let’s move on to the refine step of the RAG pipeline.

## Refine

Once the candidate texts relevant to the given query are retrieved and selected, they can be fed to the LLM. However, the LLM context window is limited, so we might want to reduce the length of the retrieved texts. We might also want to rephrase it so that it is more amenable to being processed by the LLM. Another possible operation could be to filter out some of the retrieved texts based on certain rules. All of these are conducted during the *refine* stage. In this section, we will discuss two such techniques, summarization and chain-of-note. Let’s start with discussing how we can summarize the retrieved texts.

###### Tip

The refine stage can be a standalone stage, or it can be paired with the final generate stage, where the final response is provided immediately after refining the retrieved documents, as part of the same prompt or prompt chain.

### Summarization

Summarization is useful if the retrieval chunks are relatively large. It can be either extractive or abstractive. Extractive summaries extract key sentences from the original text without modifying it. Abstractive summaries are generated from scratch, drawing on content from the original text. The summarizer can also act as a quality filter; it can output an empty summary if the document is irrelevant to the query. Summaries should be relevant, concise, and faithful to the original text.

###### Note

These summaries are not meant for human consumption but instead meant to be consumed by the LLM. Therefore, they do not always share the same objectives as traditional summarizers. The primary objective here is to generate a summary that helps the LLM output the correct answer.

Should you choose extractive or abstractive summarization? Extractive summaries are almost always faithful as they preserve the meaning of the original text. Abstractive summaries come with the risk of hallucinations. On the other hand, abstractive summaries can potentially be more relevant because of their ability to combine information from different locations within a document and across documents.

While you can leverage the LLM’s zero-shot capabilities for both extractive and abstractive summarization, it is more effective (albeit expensive) to fine-tune them so that the summaries generated are specifically optimized to enable the LLM to generate the correct answer. We will call these tightly-coupled summarizers.

[Xu et al.](https://oreil.ly/XCpyr) introduce techniques for training both extractive and abstractive summarizers. Let’s go through them in detail.

For extractive summarization, we would like to extract a subset of sentences from the retrieved document as its summary. This is done by generating embeddings for the input query and for each sentence in the retrieved document. The top-k sentences that are most similar to the input query in the embedding space are selected as the summary. The embedding distance is a measure of how effective the document sentence is in enabling the LLM to generate the correct output.

The extractive summarizer is trained with contrastive learning, which we discussed in [Chapter 11](ch11.html#chapter_llm_interfaces). Each training example in contrastive learning is a triplet: the anchor sentence, positive example similar to the anchor sentence, and negative examples dissimilar to the anchor sentence. To generate the training examples, for each sentence in the retrieved document, we prefix it to the input query and calculate the likelihood of gold truth output tokens being generated. The sentence with the highest likelihood is taken as the positive example. For negative examples, we choose up to five sentences whose likelihood is below a threshold. This dataset is then used to train the model.

For abstractive summarization, we can distill a larger LLM, i.e., use the outputs from it to fine-tune a smaller LLM.

To generate the training dataset, we can construct some prompt templates and use them with a larger LLM to generate zero-shot summaries of our retrieved documents. Note that we are generating a single summary of all the retrieved documents. Similar to the extractive summarization technique, for each generated summary, we prefix it to the input text and calculate the likelihood of the correct output tokens. We choose the summary with the highest likelihood to be part of our training set.

During inference, if prefixing any given summary has a lower likelihood of generating the correct output than not prefixing any summary at all, then we deem the text represented by the summary to be irrelevant, and an empty summary is generated. This allows us to filter out irrelevant documents.

[Figure 12-6](#abstractive-summarization) depicts the workflow of a tightly-coupled abstractive summarizer during training.

![abstractive-summarization](assets/dllm_1206.png)

###### Figure 12-6\. Abstractive summarization

###### Tip

If you are planning to change your target LLM, you might want to retrain the summary models. While the summarizers can transfer across models, there is still a slight performance degradation.

Tightly-coupled summarizers, while expensive to train initially, can be an effective means of removing irrelevant information from the retrieved text while rephrasing it in a form that reduces ambiguity for the LLM.

### Chain-of-note

Another way to rephrase the retrieved text is to generate *notes*.

It would be detrimental if the retrieved text contains irrelevant content that might mislead the model. In essence, the LLM has to contend with three types of scenarios:

*   The retrieved documents contain the answer to the user query, and the LLM can use it to generate the correct output.

*   The retrieved documents do not contain the answer to the user query, but they provide valuable context that the LLM can then combine with its internal knowledge to come up with the answer.

*   The retrieved documents are irrelevant to the user query and should be ignored.

LLMs are not very good at distinguishing between relevant and irrelevant context. One way to address this is by generating notes for each retrieved document that contains a summary of the retrieved document along with indicating whether it contains the answer to the user query or only contains relevant context but not the answer outright, or is irrelevant. This technique is called chain-of-note (CoN), introduced by [Yu et al.](https://oreil.ly/hPkKm)

Here is an example of how these notes look:

> *Prompt:* Who succeeded Brian Mulroney as the prime minister of Canada in 1993?
> 
> Doc 1: Kim Campbell took over the reins of leadership in Canada in 1993, marking her entry into history as the country’s first female Prime Minister in 1993\. Her tenure, however, was short-lived, as she led the Progressive Conservative Party into a challenging federal election later that year.
> 
> Doc 2: Following Brian Mulroney’s tenure, Canada witnessed the historic appointment of its first female Prime Minister, marking a groundbreaking shift in the nation’s political landscape.

While the first text says Kim Campbell became the prime minister in 1993, it is not apparent whether she immediately succeeded Brian Mulroney or if there was an intervening prime minister. However, the second text makes it clear that it is indeed Kim Campbell, as the text mentions that Brian Mulroney was succeeded by a female prime minister.

The corresponding note generated by the model is the following:

> *LLM:* The first statement mentions that Kim Campbell took over the leadership in Canada in 1993, and the second statement refers to the period following Brian Mulroney’s tenure, leading to the appointment of Canada’s first female prime minister. This sequence implies that Kim Campbell’s assumption of the prime minister role directly followed Brian Mulroney’s tenure, making her his successor.

Using the note as context, the LLM can then generate the correct answer.

The true value of chain-of-note becomes apparent when the context contains irrelevant or insufficient information. The generated note can help the LLM output “I don’t know.”

Consider this query:

> *Prompt:* Did the Green Party of Canada support the proposed 2023 Public Transit bill?

and the retrieved documents are:

> Doc 1: The Green Party of Canada has historically emphasized the importance of sustainable transportation as a key component of their environmental platform, advocating for policies that promote green infrastructure.
> 
> Doc 2: The proposed 2023 Public Transit bill garnered widespread attention and popularity, reflecting a growing public consensus on the importance of sustainable transportation solutions.

The retrieved documents do not contain the answer to the query. The corresponding note generated is:

> *LLM:* While the first passage indicates the party’s historical emphasis on sustainable transportation, and the second passage mentions the bill’s popularity, neither confirms the party’s support or opposition to the 2023 bill.

Using this note, the LLM can generate “unknown” when the retrieved documents do not contain the answer to the query.

An example of a CoN prompt can be:

> *Prompt:* You are provided a query along with {K} passages that potentially contain information that can be used to answer the query. Write notes summarizing the key points from these passages. Discuss the relevance of each of these passages to the given question and state whether the answer to the query can be deduced from the content in these passages.

Again, we can train tightly-coupled CoN models to make it more effective. This can be done by fine-tuning an LLM to elicit CoN behavior.

To generate the fine-tuning dataset, you can prompt an LLM to generate candidate notes for example queries. Human evaluation can then filter out incorrect or poor-quality notes. The final dataset consists of the CoN prompt, the input query, and the retrieved documents as the input, and the corresponding note and the query answer as the output. An LLM can then be fine-tuned on this dataset.

The authors (Yu et al.) introduce a weighted loss scheme during training. The note can be much longer than the answer, and thus equally weighting the loss across all tokens will lead to the note getting significantly more importance during training. This harms model convergence. The weighted loss scheme involves calculating loss across answer tokens 50% of the time.

Using a CoN step is very useful, especially if the retrieval results are known to contain a lot of noise or there is a higher possibility of no relevant documents available to service the query. CoN behavior is harder for smaller models, thus a sufficiently larger model should be used.

Now that we have discussed the refine step of the RAG pipeline, let’s move to the insert step.

## Insert

Once we have determined the content to be fed to the LLM that is going to generate the final response to a query, whether the original retrieved documents or their summaries or notes, we need to decide how we are going to arrange it inside the prompt.

The standard approach is to stuff all the content, or at least as much as can fit, into the context window. An alternative is to feed each retrieved document/summary/note prefixed to the input separately to the LLM, and then combine the outputs.

[Liu et al.](https://oreil.ly/LFR8r) show that language models are more adept at recalling information present at the beginning and the end of the context window as compared to the middle. We can exploit this knowledge to reorder the retrieved documents in the prompt.

Let’s say we retrieved 10 documents for the given query. The documents are ordered according to their relevance: Doc1, Doc2,…Doc10\. These documents can now be arranged in the prompt in the following order:

> Doc1, Doc3, Doc5, Doc7, Doc9, Doc10, Doc8, Doc6, Doc4, Doc2

Thus the least relevant documents exist in the middle of the context window, where they are more likely to be ignored by the model due to current long context recall limitations.

Alternative approaches include arranging the documents in order of relevance, for example:

> Doc1, Doc2, Doc3, Doc4, Doc5, Doc6, Doc7, Doc8, Doc9, Doc10

Or in reverse order of relevance, like:

> Doc10, Doc9, Doc8, Doc7, Doc6, Doc5, Doc4, Doc3, Doc2, Doc1

These ordering schemes are useful only if the input context is very long (upwards of 5,000 tokens).

Finally, let’s discuss the generate step in the RAG pipeline.

## Generate

The LLM generates the final response to the given query during this step. The standard approach is to generate the output all at once. However, you could also interleave the generation and the retrieval process, by generating some output and retrieving more context, and generating some more output, and retrieving more context, and so on.

This approach can be useful in maintaining coherence in long-form text generation. The generated text determines what needs to be retrieved next. This process is called active retrieval.

How do we decide when to stop generating and start a new retrieval step? We could:

*   Retrieve after every N tokens are generated.

*   Retrieve after each textual unit is generated. (A textual unit can be a sentence, paragraph, section, etc.)

*   Retrieve when currently available context is deemed insufficient for generation.

There are several ways to implement the latter. One of them is Forward-Looking Active REtrieval-augmented generation (FLARE). The authors of [FLARE](https://oreil.ly/eZRdy) introduce two methods for active retrieval: FLARE-Instruct and FLARE-Direct.

In FLARE-Instruct, the LLM is prompted to generate queries in a specific syntax whenever it needs additional information to continue coherent generation.

In FLARE-Direct, the LLM generates a candidate-next sentence. If any of the tokens in the generated sentence have probability lower than a threshold, then the retrieval process is activated. If not, then the candidate sentence is accepted as a valid continuation and the generation process continues. If retrieval is to take place, the generated sentence can be used as the query, by masking the low-probability tokens (since they might confuse the retriever if they are irrelevant/incorrect). You can also rephrase the sentence as a question about the low-probability token(s).

Let’s look at an example using FLARE-Instruct:

> *Prompt:* Write an article about Peruth Chemutai, the Ugandan Olympics medal winner.

> *FLARE-Instruct:* Peruth Chemutai [Search(birthdate of Peruth Chemutai)] is a Ugandan long-distance runner who specializes in the 3000 meters steeplechase. She gained international recognition after [Search(what medals did Peruth Chemutai win)] winning the gold medal in the women’s 3000 meters steeplechase at the 2020 Summer Olympics, becoming the first Ugandan woman to win an Olympic gold medal.
> 
> Early Life
> 
> Chemutai was born in the [Search(birthplace of Peruth Chemutai)], a region known for [Search(what is the birthplace of Peruth Chemutai known for?)].’

This is a contrived example, as the true benefits of FLARE can be better appreciated on lengthier outputs. As seen in the output, the model generates search queries that can be used to retrieve factually correct information from data sources.

For the same query, using FLARE-Direct, the model generates the candidate article:

> *FLARE-Direct:* Peruth Chemutai ( born July 10, 1999) is a Ugandan long-distance runner who specializes in the 3000 meters steeplechase. She gained international recognition after winning the gold medal in the women’s 3000 meters steeplechase at the 2020 Summer Olympics, becoming the first Ugandan woman to win an Olympic gold medal.
> 
> Early Life
> 
> Chemutai was born in the Bukwo District, Uganda, a region known for its challenging terrain and passionate long-distance runners.

The underlined tokens are low-probability tokens, which can be refilled by retrieving relevant text. We can either mask the low-probability tokens and use them as the retrieval query or generate standalone queries like, “When was Peruth Chemutai born?” based on the masked tokens.

A crucial aspect of generation includes adding appropriate citations to ground-truth sources. The LLM can be fine-tuned to make it provide citations along with the answer in response to user queries. One such model is [Cohere’s Command-R](https://oreil.ly/v0KUs) model.

As we can see, the RAG pipeline for knowledge retrieval can be rather lengthy. However, for a lot of RAG applications, latency is a key consideration. This increases the importance of smaller language models or faster, non-LLM-based approaches.

Let’s put it all together by revisiting the RAG pipeline diagram first introduced at the beginning of the chapter. [Figure 12-7](#RAG-pipeline2) depicts the workflow of a comprehensive RAG pipeline.

![RAG-pipeline](assets/dllm_1207.png)

###### Figure 12-7\. Comprehensive RAG pipeline

So far, we have focused on using RAG for knowledge retrieval. Let’s now discuss a few other use cases.

# RAG for Memory Management

An underrated application of RAG is expanding the context window of an LLM. To recap, an LLM prompt typically contains the following types of (optional) content:

The pre-prompt or *system prompt*

These are the overarching instructions provided to the LLM included at the beginning of every query. Depending on your customization needs, the system prompt could occupy a significant part of the context window.

The input prompt

This includes the current input and the instruction, optional few-shot training examples, and additional context, possibly fetched using retrieval.

Conversational history

This includes the history of conversations/interaction between the user and the LLM. Including this in the context window enables the user to have a long, coherent conversation with the LLM.

Scratchpad

This includes intermediate output generated by the LLM (discussed in [Chapter 8](ch08.html#ch8)), which can be referred to by the LLM when generating future output. Scratchpad content is an artifact of certain prompting techniques like CoT.

In many cases, the LLM’s limited context window is simply insufficient to incorporate all this data. Moreover, we might like to make the conversational history available to the model through perpetuity, which means it keeps growing across time. Making all the conversational history available to the LLM is a key aspect in enabling personalization.

It’s RAG to the rescue! RAG can be employed in facilitating LLM memory management by swapping in and out relevant content in the prompt as suitable. This is reminiscent of how memory management occurs in operating systems. Let’s explore this abstraction further.

In an OS, memory is organized in a hierarchy, with fast (and expensive) memory being directly accessible to a processor, and higher levels of the hierarchy containing larger and slower (but relatively inexpensive) memory. When the processor needs to access some data, it tries to access it from the lowest level in the memory hierarchy. If the data is not present there, it searches the next level in the hierarchy. If present, it swaps the required data into the lower level and swaps out data that is not currently needed. This way, the OS can support a fast main memory that is directly accessible by the processor and a much larger virtual memory that can be swapped in whenever needed.

This is a very simplified explanation of OS memory management. For a more detailed explanation, check out Tony’s [“Operating System — Hierarchy of Memory”](https://oreil.ly/vcciM).

[Figure 12-8](#os-hierarchy) shows the memory hierarchy of a typical OS.

![os-hierarchy](assets/dllm_1208.png)

###### Figure 12-8\. Typical OS memory hierarchy

Similarly in LLMs, the context window is analogous to the main memory as it is directly accessible to the LLM. However, we can expand the context window indefinitely by implementing a memory system analogous to the OS virtual memory. This helps in personalizing LLMs, providing them with the full access to a user’s conversational history and their implicit and explicit preferences.

Examples of libraries supporting memory management for LLMs include [Letta (formerly MemGPT)](https://oreil.ly/1p8Vu) and [Mem0](https://oreil.ly/dgJaZ).

###### Note

An alternative or complement to swapping memory in and out is to recursively summarize the conversational history. However, summarization is a lossy process and may not be able to preserve the semantics of the text. Valuable nuances like the tone of the writer can be lost during summarization.

# RAG for Selecting In-Context Training Examples

As mentioned at the beginning of the chapter, another application of RAG is to dynamically select training examples for few-shot learning by retrieving the optimal examples from a data store containing a list of training examples. For a given input, the retrieved few-shot examples are supposed to maximize the LLM’s chance of generating the correct answer to a user query.

A simple method is to generate embeddings of the input and retrieve examples whose embeddings are most similar to the input embedding. While this technique is a promising start, we can do much better.

[Wang et al.](https://oreil.ly/r8735) introduce a method called LLM Retriever (LLM-R) that trains a model using LLM feedback to retrieve few-shot training examples whose inclusion will increase the probability of the LLM generating the correct answer. [Figure 12-9](#llm-r) illustrates the LLM-R technique.

![llm-r](assets/dllm_1209.png)

###### Figure 12-9\. LLM-R workflow

For each input query in the training set, we retrieve the top-k few-shot examples by using a retrieval model like BM25\. We then rerank the examples by using LLM feedback. Each example is prefixed to the input and the probability of the ground-truth output tokens is calculated. The examples are then ranked by decreasing order of their log-probabilities. The ranked examples are then used to train a reward model, which is distilled to train the final retrieval model.

# RAG for Model Training

So far, all the RAG applications we have explored are applied during LLM inference. Can we use RAG during model pre-training and fine-tuning as well? Yes, we can! This is an underrated area of study, and I expect to see more LLMs leveraging this in the coming years. Let’s look at an example in detail.

Retrieval-Augmented Language Model (REALM) is one of the pioneering works in the RAG space. REALM integrates the retrieval and generation tasks into a single model. [Figure 12-10](#realm) shows the REALM framework for pre-training and fine-tuning.

![realm](assets/dllm_1210.png)

###### Figure 12-10\. REALM architecture

The REALM architecture is composed of two components: a knowledge retriever and a knowledge-augmented encoder, which is a BERT-like encoder-only model. Both components are differentiable and thus trained together.

The knowledge retriever is used to generate embeddings for all documents in the external knowledge base. Retrieval is performed by finding documents with maximum embedding similarity to the input. During the masked-language modeling pre-training phase, the retriever loss function encourages it to fetch text that helps predict the masked tokens. The masked tokens are then predicted by attending to both the input text and the retrieved text. The retrieved text is supposed to contain relevant context that makes predicting the masked tokens much easier.

REALM also employs these strategies to optimize training:

*   Named entities or dates are masked so that the model can learn to predict them using retrieved context.

*   Not all masked tokens need external knowledge for their prediction. To accommodate this, an empty document is always added to the retrieved documents.

*   The retrieved documents ideally contain the context required to predict the masked token, and not the token itself. Therefore, trivial retrievals that contain the masked token in the retrieved text are not included.

# Limitations of RAG

While RAG is a powerful paradigm that expands the usefulness of LLMs and reduces hallucinations, it doesn’t resolve all the limitations of LLMs. Some pitfalls of using RAG include:

*   Relying on retrieval of text snippets can cause the LLM to depend on surface-level information to answer queries, rather than a deeper understanding of the problem space.

*   Retrieval becomes the limiting factor of the pipeline. If the retrieval process fails to extract suitable candidate text, the LLM’s powerful capabilities will all be for nothing.

*   Sometimes the retrieval process can extract documents that are contradictory to the knowledge contained in the LLM’s parametric memory. Without access to the ground truth, it is difficult for the LLM to resolve these contradictions.

# RAG Versus Long Context

As discussed in [Chapter 5](ch05.html#chapter_utilizing_llms), one of the limitations of LLMs is the limited effective context window available to them. However, this is one of the areas where rapid advances have been made recently. Context windows of at most a few thousand tokens were standard until early 2023, after which companies like [Anthropic](https://oreil.ly/ucbD-) announced support for context windows spanning over 100,000 tokens. In early 2024, Google announced [Gemini 1.5 Pro](https://oreil.ly/rp7pi), with support for one million tokens of context.

To assess the impact on LLM performance as the context size increases, several needle-in-a-haystack tests have been devised. One such implementation by [Greg Kamradt](https://oreil.ly/M8Jc9) facilitates adding a random fact or statement (the needle) to the middle of the context (the haystack) and then asking the LLM questions for which the needle is the answer.

However, it is wise to take these tests with a grain of salt as they often evaluate only the information recall capabilities of an LLM. Moreover, very few problems in the real world are needle-in-the-haystack problems; LLMs are probably not the right tool to solve them anyway. Cheaper and faster retrieval models could adequately perform most needle retrieval tasks.

In many needle-in-a-haystack tests, random sentences or paragraphs are added to the context window as needles, with the rest of the content in the context window being orthogonal to the needle. But this does not mirror the situation in the real world, where most co-occurring text is related in some way. Related text can often act as distractors, preventing the LLM from drawing the right conclusions. In fact, it is one of the reasons for developing rigorous rerank and refine steps in the RAG pipeline!

Long-context models can be useful for analyzing very long documents and also can reduce the complexity of the rerank and refine steps. I recommend empirically calculating the trade-offs where feasible.

Finally, cost is also an important consideration for the long context versus retrieval debate. No doubt, the cost for long-context models will drop significantly in the future, but retrieval will still be relatively cheaper. Forgoing retrieval completely in favor of using long-context models is akin to buying a laptop and storing all your files in RAM instead of disk.

# RAG Versus Fine-Tuning

The debate around using RAG versus fine-tuning boils down to the more fundamental question: what aspects of the task can I perform using the LLM versus relying on external sources?

In cases where external knowledge is required to solve a task, both retrieval and fine-tuning can be used. Retrieval can be used to integrate the knowledge on demand, with the drawback being that the LLM is only exposed to surface-level information and is not provided the chance to learn from connections between the data. On the other end, continued pre-training or fine-tuning can also be used to integrate external knowledge, albeit with an expensive training step.

[Ovadia et al.](https://oreil.ly/Agodo) compared RAG and fine-tuning on tasks requiring external knowledge. They showed that RAG consistently outperformed fine-tuning for knowledge-intensive tasks. As shown earlier in this chapter, LLMs need a lot of samples to memorize a concept or fact. Thus, fine-tuning effectiveness can be improved by repetition or augmentation of the fine-tuning dataset.

Even for knowledge-intensive tasks, RAG versus fine-tuning need not be an either-or decision. If you are working on a specialized domain or need your outputs in a certain style or format, you can fine-tune your LLM on domain- and task-specific data, and use RAG with this fine-tuned model for your downstream applications. In a large proportion of use cases, RAG should be sufficient, and fine-tuning shouldn’t be the first choice of solution.

RAG and fine-tuning can be complementary. Earlier in this chapter, we saw how each step of the RAG pipeline can be optimized using fine-tuning. Similarly, we also saw how RAG can be used to optimize the fine-tuning process. Thus, both retrieval and fine-tuning are powerful parts of your LLM toolkit, and I hope that these chapters have sufficiently prepared you to implement and deploy them in the wild.

# Summary

In this chapter, we conducted a deep dive into the RAG pipeline, exploring in detail the *rewrite-retrieve-rerank-refine-insert-generate* pipeline. We highlighted the effectiveness of RAG in various scenarios, including integration of external knowledge, retrieval of past conversational history, dynamic selection of few-shot learning examples, and tool selection. We also explored the limitations of RAG and scenarios where RAG may not be effective.

In the final chapter, we will explore how we can utilize all the concepts we learned so far to architect and package LLM-driven products that bring value to end users. Effective product design has become all the more important in the age of LLMs, given that a successful LLM product leverages the LLM the best it can for the capabilities it excels at, while at the same time limiting end-user exposure to LLM limitations by means of clever product design. We will also look at several LLM design patterns that put together all the concepts we learned in reusable, debuggable abstractions.