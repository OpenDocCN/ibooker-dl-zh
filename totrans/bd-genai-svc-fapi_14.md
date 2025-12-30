# Chapter 10\. Optimizing AI Services

In this chapter, you’ll learn to further optimize your services via prompt engineering, model quantization, and caching mechanisms.

# Optimization Techniques

The objectives of optimizing an AI service are to either improve output quality or performance (latency, throughput, costs, etc.).

Performance-related optimizations include the following:

*   Using batch processing APIs

*   Caching (keyword, semantic, context, or prompt)

*   Model quantization

Quality-related optimizations include the following:

*   Using structured outputs

*   Prompt engineering

*   Model fine-tuning

Let’s review each in more detail.

## Batch Processing

Often you want an LLM to process batches of entries at the same time. The most obvious solution is to submit multiple API calls per entry. However, the obvious approach can be costly and slow and may lead to your model provider rate limiting you.

In such cases, you can leverage two separate techniques for batch processing your data through an LLM:

*   Updating your structured output schemas to return multiple examples at the same time

*   Identifying and using model provider APIs that are designed for batch processing

The first solution requires you to update your Pydantic models or template prompts to request a list of outputs per request. In this case, you can batch process your data within a handful of requests instead of one per entry.

An implementation of the first solution is shown in [Example 10-1](#batch_processing_structured_outputs).

##### Example 10-1\. Updating structured output schema for parsing multiple items

```py
from pydantic import BaseModel

class BatchDocumentClassification(BaseModel):
    class Category(BaseModel):
        document_id: str
        category: list[str]

    categories: list[Category] ![1](assets/1.png)
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO1-1)

Update the Pydantic model to include a list of `Category` models.

You can now pass the new schema alongside a list of document titles to the OpenAI client to process multiple entries in a single API call. However, an alternative and possibly the best solution will be to use a batch API, if available.

Luckily, model providers such as OpenAI already supply relevant APIs for such use cases. Under the hood, these providers may run task queues to process any single batch job in the background while providing you with status updates until the batch is complete to retrieve the results.

Compared to using standard endpoints directly, you’ll be able to send asynchronous groups of requests with lower costs (up to 50% with OpenAI^([1](ch10.html#id1069))), enjoy higher rate limits, and guarantee completion times. The batch job service is ideal for processing jobs that don’t require immediate responses such as using OpenAI LLMs to parse, classify, or translate large volumes of documents in the background.

To submit a batch job, you’ll need a `jsonl` file where each line contains the details of an individual request to the API, as shown in [Example 10-2](#jsonl). Also as seen in this example, to create the JSONL file, you can iterate over your entries and dynamically generate the file.

##### Example 10-2\. Creating a JSONL file from entries

```py
import json
from uuid import UUID

def create_batch_file(
    entries: list[str],
    system_prompt: str,
    model: str = "gpt-4o-mini",
    filepath: str = "batch.jsonl",
    max_tokens: int = 1024,
) -> None:
    with open(filepath, "w") as file:
        for _, entry in enumerate(entries, start=1):
            request = {
                "custom_id": f"request-{UUID()}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": entry},
                    ],
                    "max_tokens": max_tokens,
                },
            }
            file.write(json.dumps(request) + "\n")
```

Once created, you can submit the file to the batch API for processing, as shown in [Example 10-3](#batch_processing_api).

##### Example 10-3\. Processing batch jobs with the OpenAI Batch API

```py
from loguru import logger
from openai import AsyncOpenAI
from openai.types import Batch

client = AsyncOpenAI()

async def submit_batch_job(filepath: str) -> Batch:
    if ".jsonl" not in filepath:
        raise FileNotFoundError(f"JSONL file not provided at {filepath}")

    file_response = await client.files.create(
        file=open(filepath, "rb"), purpose="batch"
    )

    batch_job_response = await client.batches.create(
        input_file_id=file_response.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "document classification job"},
    )
    return batch_job_response

async def retrieve_batch_results(batch_id: str):
    batch = await client.batches.retrieve(batch_id)
    if (
        status := batch.status == "completed"
        and batch.output_file_id is not None
    ):
        file_content = await client.files.content(batch.output_file_id)
        return file_content
    logger.warning(f"Batch {batch_id} is in {status} status")
```

You can now leverage offline batch endpoints to process multiple entries in one go with guaranteed turnaround times and significant cost savings.

Alongside leveraging structured outputs and batch APIs to optimize your services, you can also leverage caching techniques to significantly speed up response times and resource costs of your servers.

## Caching

In GenAI services, you’ll often rely on data/model response that require significant computations or long processing durations. If you have multiple users requesting the same data, repeating the same operations can be wasteful. Instead, you can use caching techniques for storing and retrieving frequently accessed data to help you optimize your services by speeding up response times, reducing server load, and saving bandwidth and operational costs.

For example, in a public FAQ chatbot where users ask mostly the same questions, you may want to reuse the cached responses for longer periods. On the other hand, for more personalized and dynamic chatbots, you can frequently refresh (i.e., invalidate) the cached response.

###### Tip

You should always consider the frequency of cache refreshes based on the nature of the data and the acceptable level of staleness.

The most relevant caching strategies for GenAI services include:

*   Keyword caching

*   Semantic caching

*   Context or prompt caching

Let’s review each in more detail.

### Keyword caching

If all you need is a simple caching mechanism for storing functions or endpoint responses, you can use *keyword caching*, which involves caching responses based on exact matches of input queries as key-value pairs.

In FastAPI, libraries such as `fastapi-cache` can help you implement keyword caching in a few lines of code, on any functions or endpoints. FastAPI caches also give you the option to attach storage backends such as Redis for centralizing the cache store across your instances.

###### Tip

Alternatively, you can implement your own custom caching mechanism with a cache store using lower-level packages such as `cachetools`.

To get started, all you have to do is to initialize and configure the caching system as part of the application lifespan, as shown in [Example 10-4](#caching_lifespan). You can install FastAPI cache using the following command:

```py
$ pip install "fastapi-cache2[redis]"
```

##### Example 10-4\. Configuring FastAPI cache lifespan

```py
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache") ![1](assets/1.png)
    yield

app = FastAPI(lifespan=lifespan)
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO2-1)

Initialize `FastAPICache` with a `RedisBackend` that doesn’t decode responses so that cached data is stored as bytes (binary). This is because decoding responses would break caching by altering the original response format.

Once the caching system is configured, you can decorate your functions or endpoint handlers to cache their outputs, as shown in [Example 10-5](#caching_functions_endpoint).

##### Example 10-5\. Function and endpoint results caching

```py
from fastapi import APIRouter
from fastapi_cache.decorator import cache

router = APIRouter(prefix="/generate", tags=["Resource"])

@cache()
async def classify_document(title: str) -> str:
    ...

@router.post("/text")
@cache(expire=60) ![1](assets/1.png)
async def serve_text_to_text_controller():
    ...
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO3-1)

The `cache()` decorator must always come last. Invalidate the cache in 60 seconds by setting `expires=60` to recompute the outputs.

The `cache()` decorator shown in [Example 10-5](#caching_functions_endpoint) injects dependencies for the `Request` and `Response` objects so that it can add cache control headers to the outgoing response. These cache control headers instruct clients how to cache the responses on their side by specifying a set of directives (i.e., instructions).

These are a few common cache control directives when sending responses:

`max-age`

Defines the maximum amount of time (in seconds) that a response is considered fresh

`no-cache`

Forces revalidation so that the clients check for constant updates with the server

`no-store`

Prevents caching entirely

`private`

Stores responses in a private cache (e.g., local caches in browsers)

A response could have cache control headers like `Cache-Control: max-age=180, private` to set these directives.^([2](ch10.html#id1072))

Since keyword caching works on exact matches, it’s more suitable for functions and APIs that expect frequently repeated matching inputs. However, in GenAI services that accept variable user queries, you may want to consider other caching mechanisms that rely on the meaning of inputs when returning a cached response. This is where semantic caching can prove useful.

### Semantic caching

*Semantic caching* is a caching mechanism that returns a stored value based on similar inputs.

Under the hood, the system uses encoders and embedding vectors to capture semantics and meanings of inputs. It then performs similarity searches across stored key-value pairs to return a cached response.

In comparison to keyword caching, similar inputs can return the same cached response. Inputs to the system don’t have to be identical to be recognized as similar. Even if such inputs have different sentence structures or formulations or contain inaccuracies, they’ll still be captured as similar for carrying the same meanings. And, the same response is being requested. As an example, the following queries are considered similar for carrying the same intent:

*   How do you build generative services with FastAPI?

*   What is the process of developing FastAPI services for GenAI?

This caching system contributes to significant cost savings^([3](ch10.html#id1076)) by reducing API calls to [30–40%](https://oreil.ly/gjGz6) (i.e., 60–70% cache hit rate) depending on the use case and size of the user base. For instance, Q&A RAG applications that receive frequently asked questions across a large user base could reduce API calls by 69% using a semantic cache.

Within a typical RAG system, there are two places where having a cache can reduce resource-intensive and time-consuming operations:

*   *Before the LLM* to return a cached response instead of generating a new one

*   *Before the vector store* to enrich prompts with cached documents instead of searching and retrieving fresh ones

When integrating a semantic cache component into your RAG system, you should consider whether returning a cached response could negatively impact your application’s user experience. For instance, if caching the LLM responses, both of the following queries would return the same response due to their high semantic similarity, causing the semantic caching system to treat them as nearly identical:

*   Summarize this text in 100 words

*   Summarize this text in 50 words

This makes it feel like your services aren’t responding to queries. As you may still want varied LLM outputs in your application, we’re going to implement a document retrieval semantic cache for your RAG system. [Figure 10-1](#semantic_cache) shows the full system architecture.

![bgai 1001](assets/bgai_1001.png)

###### Figure 10-1\. Semantic caching in RAG system architecture

Let’s start by implementing the semantic caching system from scratch first, and then we’ll review how to offload the functionality to an external library such as `gptcache`.

#### Building a semantic caching service from scratch

You can implement a semantic caching system by implementing the following components:

*   A cache store client

*   A document vector store client

*   An embedding model

[Example 10-6](#semantic_cache_cache_store) shows how to implement the cache store client.

##### Example 10-6\. Cache store client

```py
import uuid
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, PointStruct, ScoredPoint

class CacheClient:
    def __init__(self):
        self.db = AsyncQdrantClient(":memory:") ![1](assets/1.png)
        self.cache_collection_name = "cache"

    async def initialize_database(self) -> None:
        await self.db.create_collection(
            collection_name=self.cache_collection_name,
            vectors_config=models.VectorParams(
                size=384, distance=Distance.EUCLID
            ),
        )

    async def insert(
        self, query_vector: list[float], documents: list[str]
    ) -> None:
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=query_vector,
            payload={"documents": documents},
        )
        await self.db.upload_points(
            collection_name=self.cache_collection_name, points=[point]
        )

    async def search(self, query_vector: list[float]) -> list[ScoredPoint]:
        return await self.db.search(
            collection_name=self.cache_collection_name,
            query_vector=query_vector,
            limit=1,
        )
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO4-1)

Initialize a Qdrant client running on memory acting as a cache store.

Once the cache store client is initialized, you can configure the document vector store by following [Example 10-7](#semantic_cache_doc_store).

##### Example 10-7\. Document store client

```py
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, ScoredPoint

documents = [...] ![1](assets/1.png)

class DocumentStoreClient:
    def __init__(self, host="localhost", port=6333):
        self.db_client = AsyncQdrantClient(host=host, port=port)
        self.collection_name = "docs"

    async def initialize_database(self) -> None:
        await self.db_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=384, distance=Distance.EUCLID
            ),
        )
        await self.db_client.add(
            documents=documents, collection_name=self.collection_name
        )

    async def search(self, query_vector: list[float]) -> list[ScoredPoint]:
        results = await self.db_client.search(
            query_vector=query_vector,
            limit=3,
            collection_name=self.collection_name,
        )
        return results
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO5-1)

Load a collection of documents into the Qdrant vector store.

With both the cache and document vector store clients ready, you can now implement the semantic cache service, as shown in [Example 10-8](#semantic_cache_service), with methods to compute embeddings and performing cache searches.

##### Example 10-8\. Semantic caching system

```py
import time
from loguru import logger
from transformers import AutoModel

...

class SemanticCacheService:
    def __init__(self, threshold: float = 0.35):
        self.embedder = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
        )
        self.euclidean_threshold = threshold
        self.cache_client = CacheClient()
        self.doc_db_client = DocumentStoreClient()

    def get_embedding(self, question) -> list[float]:
        return list(self.embedder.embed(question))[0]

    async def initialize_databases(self):
        await self.cache_client.initialize_databases()
        await self.doc_db_client.initialize_databases()

    async def ask(self, query: str) -> str:
        start_time = time.time()
        vector = self.get_embedding(query)
        if search_results := await self.cache_client.search(vector):
            for s in search_results:
                if s.score <= self.euclidean_threshold: ![1](assets/1.png)
                    logger.debug(f"Found cache with score {s.score:.3f}")
                    elapsed_time = time.time() - start_time
                    logger.debug(f"Time taken: {elapsed_time:.3f} seconds")
                    return s.payload["content"]

        if db_results := await self.doc_db_client.search(vector): ![2](assets/2.png)
            documents = [r.payload["content"] for r in db_results]
            await self.cache_client.insert(vector, documents)
            logger.debug("Query context inserted to Cache.")
            elapsed_time = time.time() - start_time
            logger.debug(f"Time taken: {elapsed_time:.3f} seconds")

        logger.debug("No answer found in Cache or Database.")
        elapsed_time = time.time() - start_time
        logger.debug(f"Time taken: {elapsed_time:.3f} seconds")
        return "No answer available." ![3](assets/3.png)
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO6-1)

Set a similarity threshold. Any score above this threshold will be a cache hit.

[![2](assets/2.png)](#co_optimizing_ai_services_CO6-2)

Query the document store if there is no cache hit. Cache the retrieved documents against the vector embedding of the query as the cache key.

[![3](assets/3.png)](#co_optimizing_ai_services_CO6-3)

If there is no related document or cache available for the given query, return a canned answer.

Now that you have a semantic caching service, you can use it to retrieve cached documents from memory by following [Example 10-9](#semantic_cache_qdrant_usage).

##### Example 10-9\. Implementing a semantic cache in a RAG system with Qdrant

```py
async def main():
    cache_service = SemanticCacheService()
    query_1 = "How to build GenAI services?"
    query_2 = "What is the process for developing GenAI services?"

    cache_service.ask(query_1)
    cache_service.ask(query_2)

asyncio.run(main())

# Query 1:
# Query added to Cache.
# Time taken: 0.822 seconds

# Query 2:
# Found cache with score 0.329
# Time taken: 0.016 seconds
```

You should now have a better understanding of how to implement your own custom semantic caching systems using a vector database client.

#### Semantic caching with GPT cache

If you don’t need to develop your own semantic caching service from scratch, you can also use the modular `gptcache` library that gives you the option to swap various storage, caching, and embedding components.

To configure a semantic cache with `gptcache`, you first need to install the package:

```py
$ pip install gptcache
```

Then load the system on application start, as shown in [Example 10-10](#configrue_gptcache).

##### Example 10-10\. Configuring the GPT cache

```py
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from gptcache import Config, cache
from gptcache.embedding import Onnx
from gptcache.processor.post import random_one
from gptcache.processor.pre import last_content
from gptcache.similarity_evaluation import OnnxModelEvaluation

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    cache.init(
        post_func=random_one, ![1](assets/1.png)
        pre_embedding_func=last_content, ![2](assets/2.png)
        embedding_func=Onnx().to_embeddings, ![3](assets/3.png)
        similarity_evaluation=OnnxModelEvaluation(), ![4](assets/4.png)
        config=Config(similarity_threshold=0.75), ![5](assets/5.png)
    )
    cache.set_openai_key() ![6](assets/6.png)
    yield

app = FastAPI(lifespan=lifespan)
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO7-1)

Select a post-processing callback function to select a random item from the returned cached items.

[![2](assets/2.png)](#co_optimizing_ai_services_CO7-2)

Select a pre-embedding callback function to use the last query for setting a new cache.

[![3](assets/3.png)](#co_optimizing_ai_services_CO7-3)

Use the ONNX embedding model for computing embedding vectors.

[![4](assets/4.png)](#co_optimizing_ai_services_CO7-4)

Use `OnnxModelEvaluation` to compute similarity scores between cached items and a given query.

[![5](assets/5.png)](#co_optimizing_ai_services_CO7-5)

Set the caching configuration options such as a similarity threshold.

[![6](assets/6.png)](#co_optimizing_ai_services_CO7-6)

Provide an OpenAI client API key for GPT Cache to automatically perform semantic caching on LLM API responses.

Once `gptcache` is initialized, it will integrate seamlessly with the OpenAI LLM client across your application. You can now make multiple LLM queries, as shown in [Example 10-11](#semantic_caching_gptcache), knowing that `gptcache` will be caching your LLM responses.

##### Example 10-11\. Semantic caching with the GPT cache

```py
import time
from openai import OpenAI
client = OpenAI()

question = "what's FastAPI"
for _ in range(2):
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    print(f"Question: {question}")
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f"Answer: {response.choices[0].message.content}\n")
```

Using external libraries like `gptcache`, as shown in [Example 10-11](#semantic_caching_gptcache), makes implementing semantic caching straightforward.

Once the caching system is up and running, you can adjust *similarity thresholds* to tune the system’s cache hit rates.

#### Similarity threshold

When building a semantic caching service, you may need to adjust the similarity threshold based on provided queries to achieve high cache hit rates that are accurate. You can refer to the [interactive visualization of semantic cache clusters](https://semanticcachehit.com) shown in [Figure 10-2](#semantic_cache_visualization) to better understand the concept of similarity threshold.

Increasing the threshold value in [Figure 10-2](#semantic_cache_visualization) will result in a less connected graph, while minimizing can produce false positives. Therefore, you may want to run a few experiments to fine-tune the similarity threshold for your own application.

![bgai 1002](assets/bgai_1002.png)

###### Figure 10-2\. Visualization of semantic caching (Source: [semanticcachehit.com](https://semanticcachehit.com))

#### Eviction policies

Another concept relevant to caching is *eviction policies* that control the caching behavior when the caching mechanism reaches its maximum capacity. Selecting the appropriate eviction policy should be appropriate for your own use case.

###### Tip

Since the size of cache memory stores is often limited, you can add an `evict()` method to the `SemanticCachingService` you implemented in [Example 10-8](#semantic_cache_service).

[Table 10-1](#eviction_policies) shows a few eviction policies you can choose from.

Table 10-1\. Eviction policies

| Policy | Description | Use case |
| --- | --- | --- |
| First in, first out (FIFO) | Removes oldest items | When all items have the same priority |
| Least recently used (LRU) | Tracks cache usage across time and removes the least recently accessed item | When recently accessed items are more likely to be accessed again |
| Least frequently used (LFU) | Tracks cache usage across time and removes the least frequently accessed item | When less frequently used items should be removed first |
| Most recently used (MRU) | Tracks cache usage across time and removes the most recently accessed item | Rarely used, used when most recently used items are less likely to be accessed again |
| Random replacement (RR) | Removes a random item from the cache | Simple and fast, used when it doesn’t impact performance |

Choosing the right eviction policy will depend on your use case and application requirements. Generally, you can start with the LRU policy before switching to alternatives.

You should now feel more confident in implementing semantic caching mechanisms that apply to document retrieval or model responses. Next, let’s learn about context or prompt caching, which optimizes queries to models based on their inputs.

### Context/prompt caching

*Context caching*, also known as *prompt caching*, is a caching mechanism suitable for scenarios where you’re referencing large amounts of context repeatedly within small requests. It’s designed to reuse precomputed attention states from frequently reused prompts, eliminating the need for redundant recomputation of the entire input context each time a new request is made.

You should consider using a context cache when your services involve the following:

*   Chatbots with extensive system instructions and long multiturn conversations

*   Repetitive analysis of lengthy video files

*   Recurring queries against large document sets

*   Frequent code repository analysis or bug fixing

*   Document summarizations, talking to books, papers, documentation, podcast transcripts and other long form content

*   Providing a large number of examples in prompt (i.e., in-context learning)

This type of caching can help you to substantially reduce token usage costs by caching large context tokens. According to Anthropic, prompt caching can reduce costs by up to 90% and latency by up to 85% for long prompts.

The authors of the [prompt caching paper](https://oreil.ly/augpd) that presents this technique also claim that:

> We find that Prompt Cache significantly reduces latency in time-to-first-token, especially for longer prompts such as document-based question answering and recommendations. The improvements range from 8× for GPU-based inference to 60× for CPU-based inference, all while maintaining output accuracy and without the need for model parameter modifications.

[Figure 10-3](#context_caching_architecture) visualizes the context caching system architecture.

![bgai 1003](assets/bgai_1003.png)

###### Figure 10-3\. System architecture for context caching

At the time of writing, OpenAI automatically implements prompt caching for all API requests without requiring any code changes or additional costs. [Example 10-12](#context_cachin_anthropic) shows an example of how to use prompt caching when using the Anthropic API.

##### Example 10-12\. Context/prompt caching with Anthropic API

```py
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-7-sonnet-20250219", ![1](assets/1.png)
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are an AI assistant",
        },
        {
            "type": "text",
            "text": "<the entire content of a large document>",
            "cache_control": {"type": "ephemeral"}, ![2](assets/2.png)
        },
    ],
    messages=[{"role": "user", "content": "Summarize the documents in ..."}],
)
print(response)
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO8-1)

Prompt caching is available only with a handful of models including Claude 3.5 Sonnet, Claude 3 Haiku, and Claude 3 Opus.

[![2](assets/2.png)](#co_optimizing_ai_services_CO8-2)

Use the `cache_control` parameter to reuse the large document content across multiple API calls without processing it each time.

Under the hood, the Anthropic client will add `anthropic-beta: prompt-caching-2024-07-31` to the request headers.

At the time of writing, `ephemeral` is the only supported cache type, which corresponds to a 5-minute cache lifetime.

###### Note

As soon as you adopt a context cache, you’re introducing statefulness in requests by preserving tokens across them. This means the data you submit in one request will affect later requests, as the model provider server can use the cached context to maintain continuity between interactions.

With the Gemini API’s context caching feature, you can provide content to the model once, cache the input tokens, and reference these cached tokens for future requests.

Using these cached tokens can save you significant expenses if you avoid repeatedly passing in the same corpus of tokens in high volumes. The caching cost will depend on the size of the input tokens and the desired time to live (TTL) storage duration.

###### Tip

When you cache a set of tokens, you can specify a TTL duration, which is how long the cache should exist before the tokens are automatically deleted. By default, TTL is normally set to 1 hour.

You can see how to use a cached system instruction in [Example 10-13](#context_caching_google). You will also need the Gemini API Python SDK:

```py
$ pip install google-generativeai
```

##### Example 10-13\. Context caching with the Google Gemini API

```py
import datetime
import google.generativeai as genai
from google.generativeai import caching

genai.configure(api_key="your_gemini_api_key")

corpus = genai.upload_file(path="corpus.txt")
cache = caching.CachedContent.create(
    model='models/gemini-1.5-flash-001',
    display_name='fastapi', ![1](assets/1.png)
    system_instruction=(
        "You are an expert AI engineer, and your job is to answer "
        "the user's query based on the files you have access to."
    ),
    contents=[corpus], ![2](assets/2.png)
    ttl=datetime.timedelta(minutes=5),
)

model = genai.GenerativeModel.from_cached_content(cached_content=cache)
response = model.generate_content(
    [
        (
            "Introduce different characters in the movie by describing "
            "their personality, looks, and names. Also list the timestamps "
            "they were introduced for the first time."
        )
    ]
)
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO9-1)

Provide a display name as a cache key or identifier.

[![2](assets/2.png)](#co_optimizing_ai_services_CO9-2)

Pass the corpus to the context caching system. The minimum size of a context cache is 32,768 tokens.

If you run [Example 10-13](#context_caching_google) and print the `response.usage_metadata`, you should receive the following output:

```py
>> print(response.usage_metadata)

prompt_token_count: 696219
cached_content_token_count: 696190
candidates_token_count: 214
total_token_count: 696433
```

Notice how much of the `prompt_token_count` is now being cached when you compare it with the `cached_content_token_count`. The `candidates_token_count` refers to count of output or response tokens coming from the model, which isn’t affected by the caching system.

###### Warning

Gemini models don’t make any distinction between cached tokens and regular input tokens. Cached content will be prefixed to the prompt. This is why the prompt token count isn’t reduced when using caching.

With context caching, you won’t see a drastic reduction in response times but instead will significantly reduce operational costs as you avoid resending extensive system prompts and context tokens. Therefore, this caching strategy is most suitable when you have a large context to work with—for instance, when batch processing files with extensive instructions and examples.

###### Note

Using the same context cache and prompt doesn’t guarantee consistent model responses because the responses from LLMs are nondeterministic. A context cache doesn’t cache any output.

Context caching remains an active area of research. If you want to avoid any vendor lock-in, there is already some progress in this field with open source tools such as [*MemServe*](https://oreil.ly/PXm6B), which implements context caching with an elastic memory pool.

Beyond caching, you can also review your options for reducing model size to speed up response times using techniques such as *model quantization*.

## Model Quantization

If you’re going to be serving models such as LLMs yourself, you should consider *quantizing* (i.e., compressing/shrinking) your models if possible. Often, open source model repositories will also supply quantized versions that you can download and use straightaway without having to go through the quantization process yourself.

*Model quantization* is the adjustment process on the model weights and activations where high-precision model parameters are statistically projected into lower-precision values through a fine-tuning operation using scaling factors on the original parameter distribution. You can then perform all critical inference operations with lower precision, after which you can convert the outputs to higher precision to maintain the quality while improving performance.

Reducing the precision also decreases the memory storage requirements, theoretically lowering energy consumption and speeding up operations like matrix multiplication through integer arithmetic. This also enables models to run on embedded devices, which may only support integer data types.

[Figure 10-4](#quantization_process) shows the full quantization process.

![bgai 1004](assets/bgai_1004.png)

###### Figure 10-4\. Quantization process

You can save more than a handful of gigabytes in GPU memory consumption as low-precision data types such as 8-bit integer would require significantly less RAM per parameter than a data type like 32-bit float.

### Precision versus quality trade-off

[Figure 10-5](#quantization) compares a nonquantized model and a quantized model.

![bgai 1005](assets/bgai_1005.png)

###### Figure 10-5\. Quantization

As each high-precision 32-bit float parameter consumes 4 bytes of GPU memory, a 1B-parameter model would require 4 GB of memory just for inference. If you plan on retraining or fine-tuning the same model, you’ll require at least 24 GB of GPU VRAM. This is because each parameter would also require storing information like gradients, training optimizer states, activations, and temporary memory space, consuming an additional 24 bytes together. This estimates up to 6 times the memory requirement compared to just loading the model weights. The same 1B model would then require a 24 GB GPU, which the best and most expensive consumer graphics cards such as NVIDIA RTX 4090 may still struggle to meet.

Instead of using the standard 32-float, you can select any of the following formats:

*   *16-bit floating-point (FP16)* cuts memory usage in half without much of a hit to model output quality.

*   *8-bit integer (INT8)* offers huge savings in memory but with a significant loss in quality.

*   *16-bit brain floating-point (BFLOAT16)* with a similar range to FP32 balances the memory and quality trade-off.

*   *4-bit integer (INT4)* provides a balance between memory efficiency and computational precision, making it suitable for low-power devices.

*   *1-bit integer (INT1)* uses the lowest precision data type with maximum model size reduction. Research for creating high-quality [1-bit LLMs](https://oreil.ly/QH9nH) is currently under way.

For comparison, [Table 10-2](#quantization_comparison) shows the reduction in model size when you quantize the Llama family models.

Table 10-2\. Impact of quantization on the size of Llama models^([a](ch10.html#id1097))

| Model | Original | FP16 | 8 Bit | 6 Bit | 4 Bit | 2Bit |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 2 70B | 140 GB | 128.5 GB | 73.23 GB | 52.70 GB | 36.20 GB | 28.59 GB |
| Llama 3 8B | 16.07 GB | 14.97 GB | 7.96 GB | 4.34 GB | 4.34 GB | 2.96 GB |
| ^([a](ch10.html#id1097-marker)) Sources: [Llama.cpp GitHub repository](https://oreil.ly/9iYtL) and [Tom Jobbins’s Hugging Face Llama 2 70B model card](https://oreil.ly/BMDtR) |

###### Tip

In addition to the GPU VRAM needed to fit the model, you will also need an extra 5 to 8 GB of GPU VRAM for overhead during model loading.

As per the current state of research, maintaining accuracy with integer-only INT4 and INT1 data types is a challenge, and the performance improvement with INT32 or FP16 is not significant. Therefore, the most popular lower-precision data type is INT8 for inference.

According to [research](https://oreil.ly/C7Lz3), using integer-only arithmetic for inference will be more efficient than floating-point numbers. However, quantizing floating numbers to integers can be tricky. For instance, only 256 values can be represented in INT8, while float32 can represent a wide range of values.

### Floating-point numbers

To understand why projecting 32-bit floats to other formats would save so much in GPU memory, let’s look at how it breaks down.

A 32-bit floating-point number consists of the following types of bits:

*   *Sign* bit describing whether a number is positive or negative

*   *Exponent* bits controlling the scale of the number

*   *Mantissa* bits holding the actual digits determining its precision (also known as *fraction* bits)

You can see a visualization of bits in the aforementioned floating-point numbers in [Figure 10-6](#quantization_bits).

![bgai 1006](assets/bgai_1006.png)

###### Figure 10-6\. Bits in 32-bit float, 16-bit float, and bfloat16 numbers

When you project the FP32 number into other formats, in effect, you’re squeezing it into smaller ranges, losing most of its mantissa bits and adjusting its exponent bits but without losing much of the precision. You can see such a phenomenon in action by referring to [Figure 10-7](#quantization_floating_numbers).

![bgai 1007](assets/bgai_1007.png)

###### Figure 10-7\. Quantization of floating-point numbers to integers

In fact, [research on the quantization strategies for pretrained LLM models](https://oreil.ly/Swfz7) has shown that LLMs with 4-bit quantization can maintain performance similar to their nonquantized counterparts. However, while quantization saves memory, it can also reduce the inference speed of LLMs.

### How to quantize pretrained LLMs

Quantization is the process of compressing large models by weight adjustment. One such technique called [*GPTQ*](https://oreil.ly/rHYKZ) can quantize LLMs with 175 billion parameters in approximately 4 GPU hours, reducing the bit width to 3 or 4 bits per weight, with a negligible accuracy drop relative to the uncompressed model.

The Hugging Face `transformers` and `optimum` library authors have collaborated closely with the `auto-gptq` library developers to provide a simple API for applying GPTQ quantization on open source LLMs. Optimum is a library that provides APIs to perform quantization using different tools.

With the GPTQ quantization, you can quantize your favorite language model to 8, 4, 3, or even 2 bits without a big drop in performance, while maintaining faster inference speeds that are supported by most GPT hardware. You can follow [Example 10-14](#gptq_quantization) to quantize a pretrained model on your own GPU.

The dependencies you need to install to run [Example 10-14](#gptq_quantization) will include the following:

```py
$ pip install auto-gptq optimum transformers accelerate
```

##### Example 10-14\. GPTQ model quantization with Hugging Face and AutoGPTQ libraries

```py
import torch
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m" ![1](assets/1.png)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16
)

quantizer = GPTQQuantizer(
    bits=4,
    dataset="c4", ![2](assets/2.png)
    block_name_to_quantize="model.decoder.layers", ![3](assets/3.png)
    model_seqlen=2048, ![4](assets/4.png)
)
quantized_model = quantizer.quantize_model(model, tokenizer)
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO10-1)

Load the `float16` version of the `facebook/opt-125m` pretrained model prior to quantization.

[![2](assets/2.png)](#co_optimizing_ai_services_CO10-2)

Use the `c4` dataset to calibrate the quantization.

[![3](assets/3.png)](#co_optimizing_ai_services_CO10-3)

Quantize only the model’s decoder layer blocks.

[![4](assets/4.png)](#co_optimizing_ai_services_CO10-4)

Use model sequence length of `2048` to process the dataset.

###### Tip

For reference, a 175B model will require 4 GPU hours on NVIDIA A100 to quantize. However, it’s worth searching the Hugging Face model repository for prequantized models, as you might find that someone has already done the work.

Now that you understand performance optimization techniques, let’s explore how to enhance the quality of your GenAI services using methods like structured outputs.

## Structured Outputs

Foundational models such as LLMs may be used as a component of a data pipeline or connected to downstream applications. For instance, you can use these models to extract and parse information from documents or to generate code that can be executed on other systems.

You can ask the LLM to provide a textual response containing JSON information. You will then have to extract and parse this JSON string using tools like regex and Pydantic. However, there is no guarantee that the model will always adhere to your instructions. Since your downstream systems may rely on JSON outputs, they may throw exceptions and incorrectly handle invalid inputs.

Several utility packages like Instructor have been released to improve the robustness of LLM responses by taking a schema and making several API calls under the hood with various prompt templates to reach a desired output. While these solutions improve robustness, they also add significant costs to your solution due to subsequent API calls to the model providers.

Most recently, model providers have added a feature for requesting structured outputs by supplying schemas when making API calls to the model, as you can see in [Example 10-15](#structured_outputs). This helps to reduce the prompting templating work you have to do yourself and aims to improve the model’s *alignment* to your intent when returning a response.

###### Warning

At the time of writing, only the most recent OpenAI SDK supports Pydantic models for enabling structured outputs.

##### Example 10-15\. Structured outputs

```py
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

client = AsyncOpenAI()

class DocumentClassification(BaseModel): ![1](assets/1.png)
    category: str = Field(..., description="The category of the classification")

async def get_document_classification(
    title: str,
) -> DocumentClassification | str | None:
    response = await client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "classify the provided document into the following: ...",
            },
            {"role": "user", "content": title},
        ],
        response_format=DocumentClassification, ![2](assets/2.png)
    )

    message = response.choices[0].message
    return message.parsed if message.parsed is not None else message.refusal
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO11-1)

Specify a Pydantic model for structured outputs.

[![2](assets/2.png)](#co_optimizing_ai_services_CO11-2)

Provide the defined schema to the model client when making the API call.

If your model provider doesn’t support structured outputs natively, you can still leverage the model’s chat completion capabilities to increase robustness of structured outputs, as shown in [Example 10-16](#structured_outputs_completions).

##### Example 10-16\. Structured outputs based on chat completions prefill

```py
import json
from loguru import logger
from openai import AsyncOpenAI
client = AsyncOpenAI()

system_template = """
Classify the provided document into the following: ...

Provide responses in the following manner json: {"category": "string"}
"""

async def get_document_classification(title: str) -> dict:
    response = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024, ![1](assets/1.png)
        messages=[
            {"role": "system", "content": system_template},
            {"role": "user", "content": title},
            {
                "role": "assistant",
                "content": "The document classification JSON is {", ![2](assets/2.png)
            },
        ],
    )
    message = response.choices[0].message.content or ""
    try:
        return json.loads("{" + message[: message.rfind("}") + 1]) ![3](assets/3.png)
    except json.JSONDecodeError: ![4](assets/4.png)
        logger.warning(f"Failed to parse the response: {message}")
    return {"error": "Refusal response"}
```

[![1](assets/1.png)](#co_optimizing_ai_services_CO12-1)

Limit the output tokens to improve robustness and speed of the structured responses and to reduce costs.

[![2](assets/2.png)](#co_optimizing_ai_services_CO12-2)

Skip the preamble and directly return a JSON by prefilling the assistant response and including a `{` character.

[![3](assets/3.png)](#co_optimizing_ai_services_CO12-3)

Add back the prefilled `{` and then find the closing `}` and extract the JSON substring.

[![4](assets/4.png)](#co_optimizing_ai_services_CO12-4)

Handle cases where there is no JSON in the response—e.g., if there is a refusal.

Following the aforementioned techniques should help you improve the robustness of your data pipelines if they leverage LLMs as a component.

## Prompt Engineering

Prompt engineering is the practice of crafting and refining queries to generative models to produce the most useful and optimized outputs. Without refining prompts, you’d either have to fine-tune models or train a model from scratch to optimize the output quality.

Many argue that the field lacks the scientific rigor to consider it an engineering discipline. However, you can approach the problem from an engineering perspective when refining prompts to get the best quality outputs from your models.

Similar to how you communicate with others to get things done, with the optimized prompts, you can most effectively communicate your intent with the model to improve the chances of getting the responses you want. Therefore, prompting becomes not just an engineering problem but a communication one as well. A model can be compared to a knowledgeable colleague with lots of experience but limited domain knowledge, ready to help you but needs you to provide well-documented instructions, possibly with a few examples to follow and pattern match.

If your prompts are vague and generic, you’ll also get an average response.

Another way of thinking about this optimization problem is to compare the task of model prompting to programming. Instead of writing the code yourself, you’re effectively “coding” a model to be a well-integrated component of a larger application or data pipeline. You can adopt test-driven development (TDD) approaches and refine your prompts until your tests pass. Or, experiment with different models to see which one *aligns* its outputs to your intent the best.

###### Note

Maximizing model *alignment* remains a high-priority objective of many model providers so that their model outputs best satisfy the user’s intent.

### Prompt templates

If your system instructions aren’t methodical, clear, and don’t follow best prompting practices, you may be leaving potential quality and performance optimizations on the table.

As a minimum, you should have clear system prompts that provide specific tasks to the model. Best practice is to follow a systematic template. For instance, draft the model instructions following the *role, context, and task* (RCT) template:

Role

Describes how the model should behave given a scenario and a task. Research has shown that specifying roles for LLMs tends to significantly affect their outputs. As an example, a model may be more forgiving in grading an essay if you give it the role of a primary school teacher. Without a specific role, the model may assume you want the grading to follow university-level academic standards.

###### Note

You can expand on the model’s role even further and describe a *persona* in detail for the model to adopt. Using a persona, the model will exactly know how to behave and make predictions as it has more context on what the role should entail.

Context

Sets the scenario, paints the picture, and provides any relevant and useful information that the model can use as a reference to making predictions. Without an explicit context, the model can only use an implicit context that’ll contain average information of its training data. In a RAG application, the context could be the concatenation of system prompt with the retrieved document chunks from a knowledge store.

Task

Provides clear instructions on what you want the model to perform given a context and a role. When describing the task, make sure you think of the model as a bright and knowledgeable apprentice, ready to jump into action but needs highly clear and unambiguous instructions to follow, potentially with a handful of examples.

Following the aforementioned system template, you should enhance the quality of your model outputs with minimal effort.

### Advanced prompting techniques

Beyond the prompting fundamentals, you can use more advanced techniques that may better fit your use case. Based on a [recent systematic survey of prompting techniques](https://oreil.ly/xynPC), you can group LLM prompts into the following:

*   In-context learning

*   Thought generation

*   Decomposition

*   Ensembling

*   Self-criticism

*   Agentic

Let’s review each in more detail.

#### In-context learning

What sets foundational models such as LLMs apart from traditional machine learning models is their ability to respond to dynamic inputs without the constant need for fine-tuning or retraining.

When you give system instructions to an LLM, you can additionally supply several examples, (i.e., shots) to guide the output generation.

[*Zero-shot prompting*](https://oreil.ly/3F4wb) refers to a prompting approach that doesn’t specify reference examples, yet the model can still successfully complete the given task. If the model struggles without reference examples, you may have to use [*few-shot prompting*](https://oreil.ly/pOSj8) where you provide a handful of examples. There are also use cases where you want to use *dynamic few-shot* prompting where you dynamically insert examples from data fetched from a database or vector store.

Prompting approaches where you specify examples are also termed *in-context learning*. You’re effectively fine-tuning the model’s outputs to your examples and the given task without actually modifying its model weights/parameters, whereas other ML models would require adjustments to their weights.

This is what makes LLMs and foundational models so powerful, since they don’t always require weight adjustment to fit your data and tasks you give them. You can learn about several in-context learning techniques by referring to [Table 10-3](#prompting_techniques_incontext_learning).

Table 10-3\. In-context learning prompting techniques

| Prompting technique | Examples | Use cases |
| --- | --- | --- |
| Zero-shot | Summarize the following…​ | Summarization, Q&A without specific training examples |
| Few-shot | Classify documents based on examples below:[Examples] | Text classification, sentiment analysis, data extraction with few examples |
| Dynamic few-shot | Classify the following documents based on examples below:<Inject examples from a vector store based on a query> | Personalized responses, complex problem-solving |

In-context learning prompts are straightforward, effective, and a great starting point for completing a variety of tasks. For more complex tasks, you can use more advanced prompting approaches like thought generation, decomposition, ensembling, self-criticism, or agentic approaches.

#### Thought generation

Thought generation techniques like [chain of thought (CoT)](https://oreil.ly/BWUYQ) have shown to significantly improve the ability of LLMs to perform complex reasoning.

In COT prompting you ask the model to explain its thought process and reasoning as it provides a response. Variants of CoT include zero-shot or [few-shot CoT](https://oreil.ly/1gjSH) depending on whether you supply examples. A more advanced thought generation technique is [thread of thought (ThoT)](https://oreil.ly/1KyO4) that systematically segments and analyzes chaotic and very complex information or tasks.

[Table 10-4](#prompting_techniques_thought_generation) lists thought generation techniques.

Table 10-4\. Thought generation prompting techniques

| Prompting technique | Examples | Use cases |
| --- | --- | --- |
| Zero-shot chain of thought (CoT) | Let’s think step by step…​ | Mathematical problem-solving, logical reasoning, and multi-step decision-making. |
| Few-shot CoT | Let’s think step by step…​ Here are a few examples:[EXAMPLES] | Scenarios where a few examples can guide the model to perform better, such as nuanced text classification, complex question answering, and creative writing prompts. |
| Thread of thought (ThoT) | Walk me through the problem in manageable parts step by step, summarizing and analyzing as you go…​ | Maintaining context over multiple interactions, such as dialogue systems, interactive storytelling, and long-form content generation. |

#### Decomposition

Decomposition prompting techniques focus on breaking down complex tasks into smaller subtasks so that the model can work through them step-by-step and logically. You can experiment with these approaches alongside thought generation to identify which ones produce the best results for your use case.

These are the most common decomposition prompting techniques:

[Least-to-most](https://oreil.ly/HmsSN)

Ask the model to break a complex problem into smaller problems via logical reduction without solving them. You can then reprompt the model to solve each task one by one.

[Plan-and-solve](https://oreil.ly/aWTzf)

Given a task, ask for a plan to be devised, and then request the model to solve it.

[Tree of thoughts (ToT)](https://oreil.ly/IZdj1)

Create a tree-search problem where a task is broken into multiple branches of steps like a tree. Then, reprompt the model to evaluate and solve each branch of steps.

[Table 10-5](#prompting_techniques_decomposition) shows these decomposition techniques.

Table 10-5\. Decomposition prompting techniques

| Prompting technique | Examples | Use cases |
| --- | --- | --- |
| Least-to-most | Break down the task of…​into smaller tasks. | Complex problem-solving, project management, task decomposition |
| Plan-and-solve | Devise a plan to…​ | Algorithm development, software design, strategic planning |
| Tree of thoughts (ToT) | Create a decision tree for choosing a…​ | Decision-making, problem-solving with multiple solutions, strategic planning with alternatives |

#### Ensembling

*Ensembling* is the process of using multiple prompts to solve the same problem and then aggregating the responses into a final output. You can generate these responses using the same or different models.

The main idea behind ensembling is to reduce the variance of LLM outputs by improving accuracy in exchange for higher usage costs.

Well-known ensembling prompting techniques include the following:

[Self-consistency](https://oreil.ly/_85WS)

Generates multiple reasoning paths and selects the most consistent output as the final result using a majority vote.

[Mixture of reasoning experts (MoRE)](https://oreil.ly/xllKs)

Combines outputs from multiple LLMs with specialized prompts to improve response quality. Each LLM acts as an expert on an area focused on different reasoning tasks such as factual reasoning, logical reduction, common-sense checks, etc.

[Demonstration ensembling (DENSE)](https://oreil.ly/lPEPz)

Creates multiple few-shot prompts from data, then generates a final output by aggregating over the responses.

[Prompt paraphrasing](https://oreil.ly/yP_ka)

Formulates the original prompt into multiple variants via wording.

[Table 10-6](#prompting_techniques_ensemling) shows examples and use cases of these ensembling techniques.

Table 10-6\. Ensembling prompting techniques

| Prompting technique | Examples | Use cases |
| --- | --- | --- |
| Self-consistency | Prompt #1 (run multiple times): Let’s think step by step and complete the following task…​Prompt #2: From the following responses, choose the best/common one by scoring them using…​ | Reducing errors or bias in arithmetic, common-sense tasks, and symbolic reasoning tasks |
| Mixture of reasoning experts (MoRE) | Prompt #1 (run for each expert): You are a reviewer for …​, score the following based on…​Prompt #2: Choose the best expert answer based on an agreement score…​ | Accounting for specialized knowledge areas or domains |
| Demonstration ensembling (DENSE) | Create multiple few-shot examples for translating this text and aggregate the best responses.Generate several few-shot prompts for summarizing this article and combine the outputs. |  
*   Improving output reliability

*   Aggregating diverse perspectives

 |
| Prompt paraphrasing | Prompt #1a: Reword this proposal…​Prompt #1b: Clarify this proposal…​Prompt #1c: Make adjustment to this proposal…​Prompt #2: Choose the best proposal from the following responses based on…​ |  
*   Exploring different interpretations

*   Data augmentation for ensembling

 |

#### Self-criticism

*Self-criticism* prompting techniques focus on using models as AI judges, assessors, or reviewers, either to perform self-checks or to assess the outputs of other models. The criticism or feedback from the first prompt can then be used to improve the response quality in follow-on prompts.

These are several self-criticism prompting strategies:

[Self-calibration](https://oreil.ly/_4YEr)

Ask the LLM to assess the correctness of a response/answer against a question/answer.

[Self-refine](https://oreil.ly/bTQJI)

Refine responses iteratively through self-checks and providing feedback.

[Reversing chain of thought (RCoT)](https://oreil.ly/6ojtr)

Reconstruct the problem from a generated answer, and then generate fine-grained comparisons between the original problem and the reconstructed one to identify inconsistencies.

[Self-verification](https://oreil.ly/Fz3JH)

Generate potential solutions with the CoT technique, and then score each by masking parts of the question and supplying each answer.

[Chain of verification (COVE)](https://oreil.ly/WrrLP)

Create a list of related queries/questions to help verify the correctness of an answer/response.

[Cumulative reasoning](https://oreil.ly/3Hb-6)

Generate potential steps in responding to a query, and then ask the model to accept/reject each step. Finally, check whether it has arrived at the final answer to terminate the process; otherwise, repeat the process.

You can see examples of each self-criticism prompting technique in [Table 10-7](#prompting_techniques_self_criticism).

Table 10-7\. Self-criticism prompting techniques

| Prompting technique | Examples | Use cases |
| --- | --- | --- |
| Self-calibration | Assess the correctness of the following response: [response] for the following question: [question] | Gauge confidence of the answers to accept or revise the original answer. |
| Self-refine | Prompt #1: What is your feedback on the response…​Prompt #2: Using the feedback [Feedback], refine your response on…​ | Reasoning, coding, and generation tasks. |
| Reversing chain-of-thought (RCoT) | Prompt #1: Reconstruct the problem from this answer…​Prompt #2: Generate fine-grained comparison between these queries…​ | Identifying inconsistencies and revising answers. |
| Self-verification | Prompt #1 (run multiple times): Let’s think step by step - generate solution for the following problem…​Prompt #2: Score each solution based on the [masked problem]…​ | Improve on reasoning tasks. |
| Chain of verification (COVE) | Prompt #1: Answer the following question…​Prompt #2: Formulate related questions to check this response: …​Prompt #3 (run for each new related question): Answer the following question: …​Prompt #4: Based on the following information, pick the best answer…​ | Question answering and text-generation tasks. |
| Cumulative reasoning | Prompt #1: Outline steps to respond to the query: …​Prompt #2: Check the following plan and accept/reject steps relevant in responding to the query: …​Prompt #3: Check you’ve arrived at the final answer given the following information…​ | Step-by-step validation of complex queries, logical inference, and mathematical problems. |

#### Agentic

You can take the prompting techniques discussed so far one step further and add access to external tools with complex evaluation algorithms. This process specializes LLMs as *agents*, allowing them to make plans, take actions, and use external systems.

Prompts or *prompt sequences (chains)* drive agentic systems with an engineering focus on creating agent-like behavior from LLMs. These agentic workflows serve users by performing actions on systems that interface with the GenAI models, which are mostly LLMs. Tools, whether *symbolic* like a calculator, or *neural* such as another AI model, form a core component of agentic systems.

###### Tip

If you create a pipeline of multiple model calls with one output forwarded to the same or different model as input, you’ve constructed a *prompt chain*. In principle, you’re using the CoT prompting technique when you leverage prompt chains.

A few agentic prompting techniques include:

[Modular reasoning, knowledge, and language (MRKL)](https://oreil.ly/aWeQu)

Simplest agentic system consisting of an LLM using multiple tools to get and combine information for generating an answer.

[Self-correcting with tool-interactive critiquing (CRITIC)](https://oreil.ly/M-9YL)

Responds to queries, and then self-checks its answer without using external tools. Finally, uses tools to verify or amend responses.

[Program-aided language model (PAL)](https://oreil.ly/0WtKv)

Generates code from queries and sends directly to code interpreters such as Python to generate an answer.^([4](ch10.html#id1128))

[Tool-integrated reasoning agent (ToRA)](https://oreil.ly/pbfv_)

Takes PAL a few steps further by interleaving code generation and reasoning steps as long as needed to provide a satisfactory response.

[Reasoning and acting (ReAct)](https://oreil.ly/aDubr)

Given a problem, generates thoughts, takes actions, receives observations, and repeats the loop with previous information, (i.e., memory) until the problem is solved.

If you want to enable your LLMs to use tools, you can take advantage of *function calling* features from model providers, as in [Example 10-17](#function_calling).

##### Example 10-17\. Function calling for fetching

```py
from openai import OpenAI
from scraper import fetch
client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "Read the content of url and provide a summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url to fetch",
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    }
]

messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant"
        "Use the supplied tools to assist the user.",
    },
    {
        "role": "user",
        "content": "Summarize this paper: https://arxiv.org/abs/2207.05221",
    },
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)
```

As you saw in [Example 10-17](#function_calling), you can create agentic systems by configuring specialized LLMs that have access to custom tools and functions.

## Fine-Tuning

There are cases where prompt engineering alone won’t give the response quality you’re looking for. Fine-tuning is an optimization technique that requires you to adjust the parameters of your GenAI model to better fit your data. For instance, you may fine-tune a language model to learn content of private knowledge bases or to always respond with a certain tone following your brand guidelines.

It’s often not the first technique you should try since it requires effort to collect and prepare data, in addition to training and evaluating models.

### When should you consider fine-tuning?

You may want to consider fine-tuning pretrained GenAI models if one of the following scenarios is true:

*   You have significant token usage costs—for instance, due to requiring extensive system instructions or providing lots of examples in every prompt.

*   Your use case relies on specialized domain expertise that the model needs to learn.

*   You need to reduce the number of hallucinations in responses with a more fine-tuned conservative model.

*   You require higher-quality responses and have sufficient data for fine-tuning.

*   You require lower latency in responses.

Once a model has been fine-tuned, you won’t need to provide as many examples in the prompt. This saves costs and enables lower-latency requests.

###### Warning

Avoid fine-tuning as much as you can.

There are many tasks where prompt engineering alone can help you optimize the quality of your outputs. Iterating over prompts has a much faster feedback loop than iterating over fine-tuning, which relies on creating datasets and running training jobs.

However, if you do end up needing to fine-tune, you’ll notice that the initial prompt engineering efforts would contribute to producing higher-quality training data.

Here are a few cases where fine-tuning can be useful:

*   Teaching a model to respond in a brand style, tone, format, or some other qualitative metric—for instance, to produce standardized reports that comply with regulatory requirements and internal protocols

*   Improving reliability of producing desired outputs such as always having responses conform to a given structured output

*   Achieving correct results to complex queries such as document classification and tagging from hundreds of classes

*   Performing domain-specific specialized tasks such as item classification or industry-specific data interpretation and aggregation

*   Nuanced handling of edge cases

*   Performing skills or tasks that are hard to articulate in prompts such as datetime extraction from unstructured texts

*   Reducing costs by using `gpt-40-mini` or even `gpt-3.5-turbo` instead of `gpt-4o`

*   Teaching a model to use complex tools and APIs when using function calling

### How to fine-tune a pretrained model

For any fine-tuning job, you will need to follow these steps:

1.  Prepare and upload training data.

2.  Submit a fine-tuning training job.

3.  Evaluate and use fine-tuned model.

Depending on the model you’re using, the data must be prepared based on the model provider’s instruction.

For instance, to fine-tune a typical chat model like `gpt-4o-2024-08-06`, you need to prepare your data as a message format, as shown in [Example 10-18](#fine_tune_prepare). At the time of writing, [OpenAI API pricing](https://oreil.ly/MmCNq) for fine-tuning this model is $25/1M training tokens.

##### Example 10-18\. Example training data for a fine-tuning job

```py
// training_data.jsonl

{
    "messages": [
        {
            "role": "system",
            "content": "<text>"
        },
        {
            "role": "user",
            "content": "<text>"
        },
        {
            "role": "assistant",
            "content": "<text>"
        }
    ]
}
// more entries
```

Once your data is prepared, you need to upload the `jsonl` file, get a file ID, and supply that when submitting a fine-tuning job, as you can see in [Example 10-19](#fine_tuning_training).

##### Example 10-19\. Submitting a fine-tuning training job

```py
from openai import OpenAI
client = OpenAI()

response = client.files.create(
    file=open("mydata.jsonl", "rb"), purpose="fine-tune"
)

client.fine_tuning.jobs.create(
    training_file=response.id, model="gpt-4o-mini-2024-07-18"
)
```

Model providers that allow you to submit fine-tuning jobs will also provide APIs for checking the status of submitted jobs and for getting results.

Once the model is fine-tuned, you can retrieve the fine-tuned model ID and pass it to the LLM client, as shown in [Example 10-20](#fine_tuning_usage). Make sure to evaluate the model first before using it in production.

###### Tip

You can also use the testing techniques discussed in [Chapter 11](ch11.html#ch11) when evaluating fine-tuned models.

##### Example 10-20\. Using a fine-tuned model

```py
from openai import OpenAI
client = OpenAI()

fine_tuning_job_id = "ftjob-abc123"
response = client.fine_tuning.jobs.retrieve(fine_tuning_job_id)
fine_tuned_model = response.fine_tuned_model

if fine_tuned_model is None:
    raise ValueError(
        f"Failed to retrieve the fine-tuned model - "
        f"Job ID: {fine_tuning_job_id}"
    )

completion = client.chat.completions.create(
    model=fine_tuned_model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
print(completion.choices[0].message)
```

While these examples show the fine-tuning process with OpenAI, the process will be similar with other providers even if the implementation details may differ.

###### Warning

If you decide to leverage fine-tuning, be mindful that you won’t be able to take advantage of the latest improvements or optimizations in new LLMs, potentially making the fine-tuning process a waste of your time and money.

With this final optimization step, you should now feel confident in building GenAI services that not only meet your security and quality requirements but also achieve your desired throughput and latency metrics.

# Summary

In this chapter, you learned about several optimization strategies to improve the throughput and quality of your services. A few optimizations you added covered various caching (keyword, semantic, context), prompt engineering, model quantization, and fine-tuning.

In the next chapter, we will shift focus to the last step in building AI services: deploying your GenAI solution. This includes exploring deployment patterns for AI services and containerization with Docker.

^([1](ch10.html#id1069-marker)) See the OpenAI Batch API available in the [OpenAI API documentation](https://oreil.ly/0t59w).

^([2](ch10.html#id1072-marker)) Learn more about cache control headers at the [MDN website](https://oreil.ly/-Y5JP).

^([3](ch10.html#id1076-marker)) You may still require a trained embedder model for significant cost savings, as making frequent API calls to an off-the-shelf embedder model could incur additional costs, diminishing your overall savings.

^([4](ch10.html#id1128-marker)) For better security, you still need to sanitize any LLM-generated code before forwarding it to downstream systems for execution.