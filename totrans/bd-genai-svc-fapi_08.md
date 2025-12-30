# Chapter 5\. Achieving Concurrency in AI Workloads

In this chapter, you will learn more about the role and benefits of asynchronous programming in boosting the performance and scalability of your GenAI services. As part of this, you’ll learn to manage concurrent user interactions and interface with external systems such as databases, implement RAG, and read web pages to enrich the context of model prompts. You’ll acquire techniques for effectively dealing with I/O-bound and CPU-bound operations, especially when dealing with external services or handling long-running inference tasks.

We will also dive into strategies for efficiently handling long-running Generative AI inference tasks, including the use of FastAPI event loop for background tasks execution.

# Optimizing GenAI Services for Multiple Users

AI workloads are computationally expensive operations that can inhibit your GenAI services from serving multiple simultaneous requests. In most production scenarios, multiple users will be using your applications. Therefore, your services will be expected to serve requests *concurrently* such that multiple overlapping tasks can be executed. However, if you’re interfacing with GenAI models and external systems such as databases, the filesystem, or the internet, there will be operations that can block other tasks from executing on your server. Long-running operations that can halt the program execution flow are considered *blocking*.

These blocking operations can be twofold:

Input/output (I/O) bound

Where a process has to wait because of data input/output operation, which can come from a user, file, database, network, etc. Examples include reading or writing a file to a disk, making network requests and API calls, sending or receiving data from databases, or waiting for user input.

Compute bound

Where a process has to wait because of a compute-intensive operation on CPU or GPU. Compute-bound programs push the CPU or GPU cores to their limit by performing intensive computations, often blocking them from performing other tasks.^([1](ch05.html#id795)) Examples include data processing, AI model inference or training, 3D object rendering, running simulations, etc.

You have a few strategies to serve multiple users:

System optimization

For I/O-bound tasks like fetching data from a database, working with files on disk, making network requests, or reading web pages

Model optimization

For memory- and compute-bound tasks such as model loading and inference

Queuing system

For handling long-running inference tasks to avoid delays in responding

In this section, we will look at each strategy in more detail. To help solidify your learning, we will also implement several features together that leverage the aforementioned strategies:

*   Building a *web page scraper* for bulk fetching and parsing of HTTP URLs pasted in the chat, so that you can ask your LLM about the content of web pages

*   Adding a *retrieval augmented generation* (RAG) module to your service with a self-hosted vector database such as `qdrant` so that you can upload and talk to your documents via your LLM service

*   Adding a *batch image generation system* so that you can run image generation workloads as background tasks

Before I can show you how to build the aforementioned features, we should dive deeper into the topic of *concurrency* and *parallelism* as understanding both concepts will help you identify the correct strategies to use for your own use cases.

*Concurrency* refers to the ability of a service in handling multiple requests or tasks at the same time, without completing one after another. During concurrent operations, the timeline of multiple tasks can overlap and may start and end at different times.

In Python, you can implement concurrency with a single CPU core by switching between tasks on a single thread (via asynchronous programming) or across different threads (via multithreading).

With multiple cores, you can also implement a subset of concurrency called *parallelism* where tasks are split among several independent workers (via multiprocessing), with each executing tasks simultaneously on their own isolated resources and separate processes.

###### Note

Although there are plans to remove the GIL from Python soon, at the time of this writing it is not possible for multiple threads to simultaneously work through tasks. Therefore, concurrency on a single core can give an illusion of parallelism even though there is one process doing all the work. The single process can only multitask by switching active threads to minimize waiting times of I/O blocking operations.

You can only achieve true parallelism with multiple workers (in multiprocessing).

Even though concurrency and parallelism have many similarities, they aren’t exactly the same concepts. The big difference between them is that concurrency can help you manage multiple tasks by interleaving their execution, which is useful for I/O-bound tasks. Parallelism, on the other hand, involves executing multiple tasks simultaneously, typically on multicore machines, which is more useful for CPU-bound tasks.

You can implement concurrency using approaches like threading or asynchronous programming (i.e., time-slicing on a single-core machine, where tasks are interleaved to give the appearance of simultaneous execution).

[Figure 5-1](#concurrency_parallelism) shows the relationship between concurrency and parallelism.

![bgai 0501](assets/bgai_0501.png)

###### Figure 5-1\. Concurrency and parallelism

In most scalable systems, you can witness both concurrency and parallelism.

Imagine that you’re visiting a fast-food restaurant and placing an order. In a concurrent system, you’ll see the restaurant owner taking orders while cooking burgers, attending to each task time to time, and effectively multitasking by switching between tasks. In a parallel system, you’ll see multiple staff members taking orders and a few others cooking the burgers at the same time. Here different workers handle each task simultaneously.

Without any multithreading or asynchronous programming in a single-threaded process, the process has to wait for blocking operations to finish before it can start new tasks. Without multiprocessing implementing parallelism on multiple cores, computationally expensive operations can block the application from starting other tasks.

[Figure 5-2](#concurrency_parallelism_timeline) shows the distinctions between nonconcurrent execution, concurrent execution without parallelism (single core), and concurrent execution with parallelism (multiple cores).

The three Python execution models shown in [Figure 5-2](#concurrency_parallelism_timeline) are as follows:

No concurrency (synchronous)

A single process (on one core) executes tasks sequentially.

Concurrent and non-parallel

Multiple threads in a single process (on a core) handle tasks concurrently but not in parallel due to Python’s GIL.

Concurrent and parallel

Multiple processes on multiple cores perform the tasks in parallel, making the most of multicore processors for maximum efficiency.

![bgai 0502](assets/bgai_0502.png)

###### Figure 5-2\. Concurrency with and without parallelism

In multiprocessing, each process has access to its own memory space and resources to complete a task in isolation from other processes. This isolation can make processes more stable—since if a process crashes, it won’t affect others—but makes inter-process communication more complex compared to threads, which share the same memory space, as shown in [Figure 5-3](#multiprocessing_resources).

![bgai 0503](assets/bgai_0503.png)

###### Figure 5-3\. Resource sharing in multithreading and multiprocessing

Distributed workloads often use a managing process that coordinates the execution and collaboration of these processes to avoid issues such as data corruption and duplicating work. A good example of multiprocessing is when you serve requests with a load balancer managing traffic to multiple containers, each running an instance of your application.

Both multithreading and asynchronous programming reduce wait time in I/O tasks because the processor can do other work while waiting for I/O. However, they don’t help with tasks that require heavy computation, like AI inference, because the process is busy with computing some results. Therefore, to serve a large self-hosted GenAI model to multiple users, you should either scale services with multiprocessing or use algorithmic model optimizations (via specialized model inference servers like vLLM).

Your first instinct when working with slow models may be to adopt parallelism by creating multiple instances of your FastAPI service (multiprocessing) in a single machine to serve requests in parallel.

Unfortunately, multiple workers running in separate processes will not have access to a shared memory space. As a result, you can’t share artifacts—​like a GenAI model—​loaded in memory between separate instances of your app in FastAPI. Sadly, a new instance of your model will also need to be loaded, which will significantly eat up your hardware resources. This is because FastAPI is a general-purpose web server that doesn’t natively optimize serving GenAI models.

The solution is not parallelism on its own, but to adopt the external model-serving strategy, as discussed in [Chapter 3](ch03.html#ch03).

The only instance where you can treat AI inference workloads as I/O-bound, instead of compute-bound, is when you’re relying on third-party AI provider APIs (e.g., OpenAI API). In this case, you’re offloading the compute-bound tasks to the model provider through network requests.

On your side, the AI inference workloads become I/O-bound through network requests, allowing for the use of concurrency through time slicing. The third-party provider has to worry about scaling their services to handle model inferences—​that are compute-bound—​across their hardware resources.

You can externalize the serving and inference of larger GenAI models such as an LLM, with specialized servers like vLLM, Ray Serve, or NVIDIA Triton.

Later in this chapter, I will detail how these servers maximize inference efficiency of compute-bound operations during model inference while minimizing the model’s memory footprint during the data generation process.

To help you digest what was discussed so far, have a look at the comparison table of concurrency strategies in [Table 5-1](#concurrency_comparison) to understand when and why to use each.

Table 5-1\. Comparison of concurrency strategies

| Strategy | Features | Challenges | Use cases |
| --- | --- | --- | --- |
| No concurrency (synchronous) |  
*   Simple, readable, easy-to-understand code to debug

*   A single CPU core and thread

 |  
*   Potential long waiting times depending on I/O or CPU blocking operations halting the process execution

*   Can’t serve multiple users simultaneously

 |  
*   Single user applications where users can wait for tasks to finish

*   Infrequently used services or applications

 |
| Async IO (asynchronous) |  
*   A single CPU core and thread

*   Multitasking managed by an event loop within the Python process

*   Thread-safe as the Python process manages tasks

*   Maximizes the CPU utilization rate

*   Faster than multithreading and multiprocessing for I/O tasks

 |  
*   Harder to implement in code and can make debugging harder

*   Requires libraries and dependencies that use Async IO features

*   Easy to make mistakes that block the main process (and event loop)

 | Applications that have blocking I/O tasks |
| Multithreading |  
*   A single CPU core but multiple threads within the same process

*   Threads share data and resources

*   Simpler than Async IO to implement in code

*   Multitasking across threads orchestrated by the OS

 |  
*   Difficult to lock resources for each thread to avoid thread-safety issues that can lead to nonreproducible bugs and data corruption

*   Threads can block each other indefinitely (deadlocks)

*   Concurrent access to resources can cause inconsistent results (race conditions)

*   A thread can be denied resources by monopolizing threads (starvation)

*   Creating and destroying threads is computationally expensive

 | Applications or services that have blocking I/O tasks |
| Multiprocessing |  
*   Multiple processes running on several CPU cores

*   Each process is allocated a CPU core and isolated resources

*   Work can be distributed across CPU cores and managed by an orchestrator process using tools like Celery

 |  
*   Sharing hardware resources and objects like a large AI model or data between processes can be complex and requires inter-process communication (IPC) mechanisms or a dedicated shared memory

*   Difficult to keep multiple isolated processes in sync

*   Creating and destroying processes is computationally expensive

 |  
*   Applications or services that have blocking compute-bound tasks

*   Divide-and-conquer type of tasks where processing can be done in isolated chunks

*   Distributing workloads or processing requests across multiple CPU cores

 |

Now that we’ve explored various concurrency strategies, let’s continue by enhancing your services with asynchronous programming to efficiently manage I/O-bound operations. Later we’ll focus on optimizing compute-bound tasks, specifically model inference via specialized servers.

# Optimizing for I/O Tasks with Asynchronous Programming

In this section, we’ll explore the use of asynchronous programming to prevent blocking the main server process with I/O-bound tasks during AI workloads. You’ll also learn about the `asyncio` framework that enables writing asynchronous applications in Python.

## Synchronous Versus Asynchronous (Async) Execution

What is considered an asynchronous application? To answer the question, let’s compare both synchronous and asynchronous programs.

An application is considered *synchronous* when tasks are performed in a sequential order with each task waiting for the previous one to complete before starting. For applications that run infrequently and take only a few seconds to process, synchronous code rarely causes a problem and can make implementations faster and easier. However, if you need concurrency and want the efficiency of your services to be maximized on every core, your services should multitask without waiting for blocking operations to complete. That’s where implementing *asynchronous* (async) concurrency can help.

Let’s look at a few examples of synchronous and async functions to understand how much of a performance boost an async code can give you. In both examples, I will use sleeping to simulate I/O blocking operation, but you can imagine other I/O tasks being performed in real-world scenarios.

[Example 5-1](#sync_execution) shows an example of a synchronous code that simulates an I/O blocking operation with the blocking `time.sleep()` function.

##### Example 5-1\. Synchronous execution

```py
import time

def task():
    print("Start of sync task")
    time.sleep(5) ![1](assets/1.png)
    print("After 5 seconds of sleep")

start = time.time()
for _ in range(3): ![2](assets/2.png)
    task()
duration = time.time() - start
print(f"\nProcess completed in: {duration} seconds")
"""
Start of sync task
After 5 seconds of sleep
Start of sync task
After 5 seconds of sleep
Start of sync task
After 5 seconds of sleep

Process completed in: 15.014271020889282 seconds
"""
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO1-1)

Use `sleep()` to simulate an I/O blocking operation such as sending a network request.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO1-2)

Call the `task()` three times, sequentially. The loop simulates sending multiple network requests, one after another.

Calling `task()` three times in [Example 5-1](#sync_execution) takes 15 seconds to complete as Python waits for the blocking operation `sleep()` to complete.

To develop async programs in Python, you can use the `asyncio` package as part of the standard library of Python 3.5 and later versions. Using `asyncio`, asynchronous code looks similar to sequential synchronous code but with additions of `async` and `await` keywords to perform nonblocking I/O operations.

[Example 5-2](#async_execution) shows how you can use `async` and `await` keywords with `asyncio` to run [Example 5-1](#sync_execution) asynchronously.

##### Example 5-2\. Asynchronous execution

```py
import time
import asyncio

async def task(): ![1](assets/1.png)
    print("Start of async task")
    await asyncio.sleep(5) ![2](assets/2.png)
    print("Task resumed after 5 seconds")

async def spawn_tasks():
    await asyncio.gather(task(), task(), task()) ![3](assets/3.png)

start = time.time()
asyncio.run(spawn_tasks()) ![4](assets/4.png)
duration = time.time() - start

print(f"\nProcess completed in: {duration} seconds")
"""
Start of async task
Start of async task
Start of async task
Task resumed after 5 seconds
Task resumed after 5 seconds
Task resumed after 5 seconds

Process completed in: 5.0057971477508545 seconds ![5](assets/5.png) """
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO2-1)

Implement a `task` coroutine that cedes control to the event loop on blocking operations.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO2-2)

The nonblocking five-second sleep signals to the event loop to run another task while waiting.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO2-3)

Use `asyncio.create_task` to spawn task instances to chain (or gather) and run them concurrently using `asyncio.gather`.

[![4](assets/4.png)](#co_achieving_concurrency_in_ai_workloads_CO2-4)

Create an event loop to schedule async tasks with via the `asyncio.run` method.

[![5](assets/5.png)](#co_achieving_concurrency_in_ai_workloads_CO2-5)

Execution time is 1/3 of the synchronous example since the Python process wasn’t blocked this time.

After running [Example 5-2](#async_execution), you will notice that the `task()` function was concurrently called three times. On the other hand, the code in [Example 5-1](#sync_execution) calls the `task()` function three times sequentially. The async function ran inside the `asyncio`’s event loop, which was responsible for executing the code without waiting.

In any async code, the `await` keyword flags the I/O blocking operations to Python so that they’re executed in a *nonblocking* manner (i.e., they can run without blocking the main process). By being made aware of blocking operations, Python can go and do something else while waiting for blocking operations to finish.

[Example 5-3](#await_keyword) shows how to use the `async` and `await` keywords to declare and run async functions.

##### Example 5-3\. How to use `async` and `await` keywords

```py
import asyncio

async def main():
    print("Before sleeping")
    await asyncio.sleep(3) ![1](assets/1.png)
    print("After sleeping for 3 seconds")

asyncio.run(main()) ![2](assets/2.png)

"""
Before sleeping
After sleeping for 3 seconds ![3](assets/3.png) """
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO3-1)

Simulate a nonblocking I/O operation by `await`ing the `asyncio.sleep()` so that Python can go and do other things while waiting.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO3-2)

You need to call `main()` inside the `asyncio.run()` to execute it as it’s an async function. Otherwise, it will not be executed and returns a *coroutine* object instead. I will cover coroutines shortly.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO3-3)

If you run the code, the second statement will be printed 3 seconds after the first one. In this instance, as there are no other operations to run beyond sleeping, Python runs in idle until the sleep operation is completed.

In [Example 5-3](#await_keyword), I used sleeping as a way to simulate I/O blocking operations such as making network requests.

###### Caution

You can only use the `await` keyword inside a function declared with `async def`. Using `await` outside of an `async` function will raise a `SyntaxError` in Python. Another common pitfall is using blocking code that’s not asynchronous within an `async` function that will inadvertently prevent Python from doing other tasks while waiting.

So, you now understand that in async programs, to keep the main process from being blocked, Python switches between functions as soon as it hits a blocking operation. You now may be wondering:

*   How does Python leverage `asyncio` to pause and resume functions?

*   What is the mechanism that Python’s `asyncio` uses to move from one function to another without forgetting about those that are suspended?

*   How can functions be paused or resumed without losing their state?

To answer the aforementioned questions, let’s dive deeper into the underlying mechanisms within `asyncio`, as understanding the answers to these questions will help you significantly to debug async code in your services.

At the heart of `asyncio` lies a first-class object called an *event loop*, responsible for efficient handling of I/O events, system events, and application context changes.

[Figure 5-4](#event_loop) shows how the `asyncio` event loop undertakes task orchestration in Python.

![bgai 0504](assets/bgai_0504.png)

###### Figure 5-4\. Async IO event loop

The event loop can be compared to a `while True` loop that watches for events or messages emitted by *coroutine functions* within the Python process and dispatches events to switch between functions while waiting for I/O blocking operations to complete. This orchestration allows other functions to execute asynchronously without interruption.

## Async Programming with Model Provider APIs

All three examples I’ve shown you so far are considered to be the “Hello World” examples of async programming. Now, let’s look at a real-world scenario related to building GenAI services where you need to use a model provider’s API—such as OpenAI, Anthropic, or Mistral—since it may be more expensive to serve LLMs yourself.

Additionally, if you stress test the generation endpoints you created in [Chapter 3](ch03.html#ch03) by sending multiple requests in a short timeframe, you will notice long waiting times before each request is processed. This is because you were preloading and hosting the model in the same Python process and CPU core that the server is running on. When you send the first request, the whole server becomes blocked while the inference workload is complete. Since during inference the CPU is working as hard as it can, the inference/generation process is a CPU-bound blocking operation. However, it doesn’t have to be.

When you use a provider’s API, you no longer have CPU-bound AI workloads to worry about since they become I/O-bound for you, and you offload the CPU-bound workloads to the provider. Therefore, it makes sense to know how to leverage async programming to concurrently interact with the model provider’s API.

The good news is API owners will often release both synchronous and asynchronous *clients* and *software development kits* (SDKs) to reduce the work needed to interact with their endpoints.

###### Caution

If you need to make requests to other external services, fetch some data from databases, or ingest content from files, you will add other I/O blocking tasks to the process. These blocking tasks can force the server to keep waiting if you don’t leverage asynchronous programming.

However, any synchronous code can be made async using a [process or thread pool executor](https://oreil.ly/hIDNI) to avoid running the task within the event loop. Instead, you run the asynchronous task on a separate process or thread to prevent blocking the event loop.

You can also verify any async support by checking library documentation or source code for mentions of `async` or `await` keywords. Otherwise, you can try testing whether the tool can be used within an async function without raising a `TypeError` when you use `await` on it.

If a tool, such as a database library, only has a synchronous implementation, then you can’t implement asynchronicity with that tool. The solution will be to switch the tool to an asynchronous equivalent so that can you can use them with the `async` and `await` keywords.

In [Example 5-4](#openai_clients), you will interact with OpenAI GPT-3.5 API via both synchronous and asynchronous OpenAI clients to understand the performance difference between the two.

###### Note

You will need to install the `openai` library:

```py
$ pip install openai
```

##### Example 5-4\. Comparing synchronous and asynchronous OpenAI clients

```py
import os
from fastapi import FastAPI, Body
from openai import OpenAI, AsyncOpenAI

app = FastAPI()

sync_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.post("/sync")
def sync_generate_text(prompt: str = Body(...)):
    completion = sync_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return completion.choices[0].message.content

@app.post("/async")
async def async_generate_text(prompt: str = Body(...)):
    completion = await async_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return completion.choices[0].message.content
```

The difference between the sync and async clients is that with the async version, FastAPI can start processing user inputs in parallel without waiting for a response from the OpenAI API for the previous user input.

By leveraging asynchronous code, you can get a massive boost in throughput and scale to a larger volume of concurrent requests. However, you must be careful when writing asynchronous (async) code.

Here are some common pitfalls and problems you might face with async code:

*   Understanding and debugging errors can be more complex due to the nonlinear execution flow of concurrent tasks.

*   Some libraries, like `aiohttp`, require nested async context managers for proper implementation. This can get confusing pretty fast.

*   Mixing asynchronous and synchronous code can negate any performance benefits, such as if you forget to mark functions with the `async` and `await` keywords.

*   Not using async-compatible tools and libraries can also cancel out any performance benefits; for example, using the `requests` package instead of `aiohttp` for making async API calls.

*   Forgetting to await coroutines within any async function or awaiting non-coroutines can lead to unexpected behavior. All `async` keywords must be followed by an `await`.

*   Improperly managing resources (e.g., open API/database connections or file buffers) can cause memory leaks that freeze your computer. You can also leak memory if you don’t limit the number of concurrent operations in async code.

*   You might also run into concurrency and race condition issues where the thread-safety principle is violated, causing deadlocks on resources leading to data corruption.

This list is not exhaustive, and as you can see, there are several pitfalls to using asynchronous programming. Therefore, I recommend starting with writing synchronous programs first, to understand the basic flow and logic of your code, before dealing with the complexities of migrating to an async implementation.

## Event Loop and Thread Pool in FastAPI

Under the hood, FastAPI can handle both async and sync blocking operations. It does this by running sync handlers in its *thread pool* so that blocking operations don’t stop the *event loop* from executing tasks.

As I mentioned in [Chapter 2](ch02.html#ch02), FastAPI runs on the ASGI web framework via Starlette. If it didn’t, the server would effectively run synchronously, so you would have to wait for each process to finish before it could serve the next. However, using ASGI, the FastAPI server supports concurrency via both multithreading (via a thread pool) and asynchronous programming (via an event loop) to serve multiple requests in parallel, while keeping the main server process from being blocked.

FastAPI sets up the thread pool by instantiating a collection of threads at application startup to reduce the runtime burden of thread creation.^([4](ch05.html#id822)) It then delegates background tasks and synchronous workloads to the thread pool to prevent the event loop from being blocked by any blocking operations inside the synchronous handlers. The event loop is also referred to as the main FastAPI server thread that is responsible for orchestrating the processing of requests.

As I mentioned, the event loop is the core component of every application built on top of `asyncio`, including FastAPI that implements concurrency. Event loops run asynchronous tasks and callbacks, including performing network I/O operations, and running subprocesses. In FastAPI, the event loop is also responsible for orchestrating the asynchronous processing of requests.

If possible, you should run handlers on the event loop (via asynchronous programming) as it can be even more efficient than running them on the thread pool (via multithreading). This is because each thread in the thread pool has to acquire the GIL before it can execute any code bytes, and that requires some computational effort.

Imagine if multiple concurrent users were using both the synchronous and asynchronous OpenAI GPT-3.5 handlers (endpoints) of your FastAPI service, as shown in [Example 5-4](#openai_clients). FastAPI will run the async handler requests on the event loop since that handler uses a nonblocking async OpenAI client. On the other hand, FastAPI has to delegate the synchronous handler requests to the thread pool to protect the event loop from blocking. Since delegating requests (to threads) and switching between threads in a thread pool is more work, the synchronous requests will finish later than their async counterparts.

###### Note

Remember that all of this work—processing both synchronous and async handler requests—is running on a single CPU core within the same FastAPI Python process.

This is so that the CPU idle time is minimized while waiting for responses from OpenAI API.

The differences in performance are shown in [Figure 5-5](#multithreading_vs_async).

![bgai 0505](assets/bgai_0505.png)

###### Figure 5-5\. How multithreading and Async IO handle I/O blocking operations

[Figure 5-5](#multithreading_vs_async) shows that with I/O-bound workloads, async implementations are faster and should be your preferred method if you need concurrency. However, FastAPI does still do a solid job of serving multiple concurrent requests even if it has to work with a synchronous OpenAI client. It simply sends the synchronous API calls within threads of the thread pool to implement some form of concurrency for you. That’s why the FastAPI official documentation tells you to not worry too much about declaring your handler functions as `async def` or `def`.

However, keep in mind that when you declare handlers with `async def`, FastAPI trusts you with performing only nonblocking operations. When you break that trust and execute blocking operations inside `async` routes, the event loop will be blocked and can no longer continue with executing tasks until the blocking operation is finished.

## Blocking the Main Server

If you’re using the `async` keyword when defining your functions, make sure you’re also using the `await` keyword somewhere inside your function and that none of the package dependencies you use inside the function are synchronous.

Avoid declaring route handler functions as `async` if their implementation is synchronous. Otherwise, requests to the affected route handlers will block the main server from processing other requests while the server is waiting for the blocking operation to complete. It won’t matter if the blocking operation is I/O-bound or compute-bound. Therefore, any calls to databases or AI models can still cause the blockage if you’re not careful.

This is an easy mistake to make. For instance, you may use a synchronous dependency inside handlers you’ve declared as async, as shown in [Example 5-5](#blocking_main_thread).

##### Example 5-5\. Incorrect implementation of asynchronous handlers in FastAPI

```py
import os
from fastapi import FastAPI
from openai import AsyncOpenAI, OpenAI

app = FastAPI()

@app.get("/block")
async def block_server_controller():
    completion = sync_client.chat.completions.create(...) ![1](assets/1.png)
    return completion.choices[0].message.content

@app.get("/slow")
def slow_text_generator():
    completion = sync_client.chat.completions.create(...) ![2](assets/2.png)
    return completion.choices[0].message.content

@app.get("/fast")
async def fast_text_generator():
    completion = await async_client.chat.completions.create(...) ![3](assets/3.png)
    return completion.choices[0].message.content
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO4-1)

I/O blocking operation to get ChatGPT API response. Because the route handler is marked async, FastAPI trusts us to not run blocking operations, but as we are, the request will block the event loop (main server thread). Other requests are now blocked until the current request is processed.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO4-2)

A simple synchronous route handler with blocking operation that doesn’t leverage asynchronous features. Sync requests are handed off to the thread pool to run in the background so that the main server is not blocked.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO4-3)

An asynchronous route that is nonblocking.

The request won’t block the main thread and doesn’t need to be handed off to the thread pool. As a result, the FastAPI event loop can process the request much faster using the async OpenAI client.

You now should feel more comfortable implementing new features in your FastAPI service that require performing I/O-bound tasks.

To help solidify your understanding of the I/O concurrency concepts, in the next few sections you will build several new features using concurrency into your FastAPI service. These features include:

Talk to the web

Build and integrate a web scraper module that allows you to ask questions to your self-hosted LLM about the content of a website by providing an HTTP URL.

Talk to documents

Build and integrate a RAG module to process documents into a vector database. A vector database stores data in a way that supports efficient similarity searches. You can then use semantic search, which understands the meaning of queries, to interact with uploaded documents using your LLM.

Both projects will give you a hands-on experience interacting asynchronously with external systems such as websites, a database, and a filesystem.

## Project: Talk to the Web (Web Scraper)

Companies often host a series of internal web pages for manuals, processes, and other documentation as HTML pages. For longer pages, your users may want to provide URLs when asking questions and expect your LLM to fetch and read the content. This is where having a built-in web scraper can come in handy.

There are many ways to build a web scraper for your self-hosted LLM. Depending on your use case, you can use a combination of the following methods:

*   Fetch web pages as HTML and feed the raw HTML (or inner text content) to your LLM to parse the content into your desired format.

*   Use *web scraping frameworks* such as `BeautifulSoup` and `ScraPy` to parse the content of web pages after fetching.

*   Use *headless web browsers* such as Selenium and Microsoft Playwright to dynamically navigate nodes in pages and parse content. Headless browsers are great for navigation single-page applications (SPAs).

###### Caution

You or your users should avoid LLM-powered web scraping tools for illegal purposes. Make sure you have permission before extracting content from URLs:

*   Review each website’s terms of use, especially if there is a mention of web scraping.

*   Use APIs when possible.

*   Ask website owners for permission directly if unsure.

For this mini-project, we will only fetch and feed raw inner text of HTML pages to our LLM since implementing a production-ready scraper can become a book of its own.

The process for building a simple asynchronous scraper is as follows:

1.  Develop a function to match URL patterns using regex on user prompts to the LLM.

2.  If found, loop over the list of provided URLs and asynchronously fetch the pages. We will use an asynchronous HTTP library called `aiohttp` instead of the `requests` since `requests` can only make synchronous network requests.

3.  Develop a parsing function to extract the textual content from fetched HTML.

4.  Feed the parsed page content to the LLM alongside the original user prompt.

[Example 5-6](#web_scraper) demonstrates how you can implement the aforementioned steps.

###### Note

You will need to install a few additional dependencies to run this example:

```py
$ pip install beautifulsoup lxml aiohttp
```

##### Example 5-6\. Building an asynchronous web scraper

```py
# scraper.py

import asyncio
import re

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

def extract_urls(text: str) -> list[str]:
    url_pattern = r"(?P<url>https?:\/\/[^\s]+)" ![1](assets/1.png)
    urls = re.findall(url_pattern, text) ![2](assets/2.png)
    return urls

def parse_inner_text(html_string: str) -> str:
    soup = BeautifulSoup(html_string, "lxml")
    if content := soup.find("div", id="bodyContent"): ![3](assets/3.png)
        return content.get_text()
    logger.warning("Could not parse the HTML content")
    return ""

async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response: ![4](assets/4.png)
        html_string = await response.text()
        return parse_inner_text(html_string)

async def fetch_all(urls: list[str]) -> str:
    async with aiohttp.ClientSession() as session: ![5](assets/5.png)
        results = await asyncio.gather(
            *[fetch(session, url) for url in urls], return_exceptions=True
        )
    success_results = [result for result in results if isinstance(result, str)]
    if len(results) != len(success_results): ![6](assets/6.png)
        logger.warning("Some URLs could not be fetched")
    return " ".join(success_results)
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO5-1)

A simple regex pattern that captures the URLs into a named group called `url` and matches both `http` and `https` protocols. For simplicity, this pattern matches more loosely defined URLs and doesn’t validate the structure of a domain name or path, nor does it account for query strings or anchors in a URL.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO5-2)

Find all nonoverlapping matches of the regex pattern in the text.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO5-3)

Use the `bs4` Beautiful Soup package to parse the HTML string. In Wikipedia pages, the article content is nested within a `div` container with the `id="bodyContent"`, so the parsing logic assumes only Wikipedia URLs will be passed in. You can change this logic for other URLs or just use `soup.getText()` to grab any text content nested within the HTML. However, bear in mind that there will be lots of noise in the parsed content if you parse the raw HTML like that, which can confuse the LLM.

[![4](assets/4.png)](#co_achieving_concurrency_in_ai_workloads_CO5-4)

Given an `aiohttp` session and a URL, perform an asynchronous `get` request. Create a `response` async context manager and `await` the response within this context manager.

[![5](assets/5.png)](#co_achieving_concurrency_in_ai_workloads_CO5-5)

Given a list of URLs, create a client session async context manager to asynchronously perform multiple fetch calls. Since `fetch()` is a coroutine function (i.e., it uses the `await` keyword), `fetch_all()` will need to run multiple `fetch()` coroutines inside the `asyncio.gather()` to be scheduled for asynchronous execution on the event loop.

[![6](assets/6.png)](#co_achieving_concurrency_in_ai_workloads_CO5-6)

Check that all URLs have been fetched successfully and, if not, raise a warning.

You now have the utility scraper functions you need to implement the web scraping feature in your `/generate/text` endpoint.

Next, upgrade the text-to-text handler to use the scraper functions via a dependency in an asynchronous manner, as shown in [Example 5-7](#web_scraper_fastapi).

##### Example 5-7\. Injecting web scraper functionality as a dependency into the FastAPI LLM handler

```py
# dependencies.py

from fastapi import Body
from loguru import logger

from schemas import TextModelRequest
from scraper import extract_urls, fetch_all

async def get_urls_content(body: TextModelRequest = Body(...)) -> str: ![1](assets/1.png)
    urls = extract_urls(body.prompt)
    if urls:
        try:
            urls_content = await fetch_all(urls)
            return urls_content
        except Exception as e:
            logger.warning(f"Failed to fetch one or several URls - Error: {e}")
    return ""

# main.py

from fastapi import Body, Depends, Request
from dependencies import construct_prompt
from schemas import TextModelResponse

@app.post("/generate/text", response_model_exclude_defaults=True) ![2](assets/2.png)
async def serve_text_to_text_controller(
    request: Request,
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content) ![3](assets/3.png)
) -> TextModelResponse:
    ... # rest of controller logic
    prompt = body.prompt + " " + urls_content
    output = generate_text(models["text"], prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO6-1)

Implement a `get_urls_content` FastAPI dependency that gets a user prompt from the request body and finds all URLs. It then returns the content of all URLs as a long string. The dependency has exception handling built in to handle any I/O errors by returning an empty string and logging a warning on the server.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO6-2)

When using `aiohttp` inside FastAPI, you don’t need to manage the event loop yourself because FastAPI, as an asynchronous framework, handles the event loop. You can define your endpoint as an async function and use `aiohttp` to make asynchronous HTTP requests within the handler or via a dependency like in this example.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO6-3)

Inject the results of the `get_urls_content` dependency call to the handler via the FastAPI’s `Depends` class. Using a FastAPI dependency here kept the controller logic small, clean, and readable.

Now, run the Streamlit client in the browser and try your shiny new feature. [Figure 5-6](#llm_summary) shows my experiment.

![bgai 0506](assets/bgai_0506.png)

###### Figure 5-6\. Asking the self-hosted TinyLlama model to summarize a Wikipedia article

Congratulations! You’ve learned how to build a simple nonblocking web scraper to work with your own LLM. In this mini-project, you leveraged `re` package to match URL patterns in the user prompt and then used the `aiohttp` library to asynchronously fetch multiple pages concurrently. You then used the `BeautifulSoup` package to parse the content of Wikipedia articles by grabbing the text content of the `div` container with the ID of `bodyContent` within the fetched HTML string. For other websites or internal company web pages, you can always alter the parsing logic for appropriate parsing. Finally, you wrapped the whole scraping logic inside a FastAPI dependency with exception handling built-in to make use of dependency injection while upgrading the text model-serving handler.

Bear in mind that your scraper can’t handle complex pages with dynamic layouts that are server-rendered. In such cases, you can add a headless browser to your web scraper for navigating dynamic pages.

Additionally, fetching content of external sites will be challenging since most sites may implement anti-scraping protections such as *IP blocking* or *CAPTCHAs* as common deterrents. Maintaining *data quality* and *consistency* with external websites is also an ongoing challenge as you may need to update your scraping scripts regularly to ensure accurate and reliable extraction.

You should now feel more comfortable building GenAI-powered services that need to interact with the web via making asynchronous network requests.

Next, we will look at other I/O asynchronous interactions such as those with databases and the filesystem by building a *talk to your documents* feature.

This functionality allows users to upload documents through the Streamlit interface to your service. The content of the uploaded documents is then extracted, processed, and saved in a database. Subsequently, during user interactions with the LLM, an asynchronous retrieval system retrieves semantically relevant content from the database, which is then used to augment the context provided to the LLM.

This process is referred to as RAG, which we will build as a module for your LLM next.

## Project: Talk to Documents (RAG)

In this project, we will build a RAG module into your GenAI service to give you a hands-on experience interacting asynchronously with external systems such as a database and a filesystem.

You might be curious about the purpose of a RAG module and its necessity. RAG is simply a technique for augmenting the context of LLM prompts with custom data sources for knowledge-intensive tasks.^([5](ch05.html#id830)) It is an effective technique to ground LLM responses in facts contained within data without the need for complex and expensive LLM fine-tuning.

Organizations are eager to implement RAG with their own LLMs since it allows their employees to engage with their massive internal knowledge bases via the LLM. With RAG, businesses expect that the internal knowledge bases, systems, and procedures will be made accessible and readily available to anyone who needs them to answer questions, just when they need it. This accessibility to the company’s body of information is expected to enhance productivity, cut costs and time looking for information, and boost profits for any business.

However, LLMs are susceptible to generating responses that don’t adhere to the instructions given by the user. In other words, the LLM can *hallucinate* responses with information or data that is not based on facts or reality.

These hallucinations can occur due to the model’s reliance on patterns in the data it was trained on rather than direct access to external, up-to-date, and factual data. LLMs can manifest hallucinations with confidently presented yet incorrect or nonsensical answers, fabricated stories, or claims without a basis in truth.

Therefore, for more complex and knowledge-intensive tasks, you will want your LLM to access external knowledge sources to complete tasks. This enables more factual consistency and improves the reliability of the generated responses. [Figure 5-7](#rag) shows the full process.

![bgai 0507](assets/bgai_0507.png)

###### Figure 5-7\. RAG

In this project, you will build a simple RAG module for your LLM service such that users can upload and talk to their documents.

###### Note

There is a lot to know about RAG. It’s enough to fill several textbooks with new papers being published every day for new techniques and algorithms.

I recommend checking out other publications on LLMs to learn about the RAG process and advanced RAG techniques.

The pipeline for RAG consists of the following stages:

1.  *Extraction* of documents from a filesystem to load the textual content in chunks onto memory.

2.  *Transformation* of the textual content by cleaning, splitting, and preparing them to be passed into an embedding model to produce embedding vectors that represent a chunk’s semantic meaning.

3.  *Storage* of embedding vectors alongside metadata, such as the source and text chunk, in a vector store such as Qdrant.

4.  *Retrieval* of semantically relevant embedding vectors by performing a semantic search on the user’s query to the LLM. The original text chunks—​stored as metadata of the retrieved vectors—​are then used to augment (i.e., enhance the context within) the initial prompt provided to the LLM.

5.  *Generation* of LLM response bypassing both the query and retrieved chunks (i.e., context) to the LLM for getting a response.

You can see the full pipeline in [Figure 5-8](#rag_pipeline).

![bgai 0508](assets/bgai_0508.png)

###### Figure 5-8\. RAG pipeline

You can take the pipeline shown in [Figure 5-8](#rag_pipeline) and build it to your existing service. [Figure 5-9](#rag_module) shows the system architecture of a “talk to your documents” service enabled with RAG.

![bgai 0509](assets/bgai_0509.png)

###### Figure 5-9\. Talk to your documents system architecture

[Figure 5-9](#rag_module) outlines how the documents uploaded by users via the Streamlit interface are stored and then fetched for processing and storage into the database for later retrieval to augment the LLM prompts.

The first step before implementing the RAG system in [Figure 5-9](#rag_module) is to include a file upload functionality in both the Streamlit client and your backend API.

Using FastAPI’s `UploadFile` class, you can accept documents from users in chunks and save them into the filesystem or any other file storage solution such as a blob storage. The important item to note here is that this I/O operation is nonblocking through asynchronous programming, which FastAPI’s `UploadFile` class supports.

###### Tip

Since users may upload large documents, FastAPI’s `UploadFile` class supports *chunking* to store the uploaded documents, one piece at a time.

This will prevent your service’s memory from being clogged up. You will also want to protect your service by disallowing users from uploading documents above a certain size.

[Example 5-8](#upload_file) shows how to implement an asynchronous file upload functionality.

###### Tip

You will need to install `aiofiles` package to asynchronously upload files alongside `python-multipart` to receive uploaded files from HTML forms:

```py
$ pip install aiofiles python-multipart
```

##### Example 5-8\. Implementing an asynchronous file upload endpoint

```py
# upload.py

import os
import aiofiles
from aiofiles.os import makedirs
from fastapi import UploadFile

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50  # 50 megabytes

async def save_file(file: UploadFile) -> str:
    await makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    async with aiofiles.open(filepath, "wb") as f:
        while chunk := await file.read(DEFAULT_CHUNK_SIZE):
            await f.write(chunk)
    return filepath

# main.py

from fastapi import FastAPI, HTTPException, status, File
from typing import Annotated
from upload import save_file

@app.post("/upload")
async def file_upload_controller(
    file: Annotated[UploadFile, File(description="Uploaded PDF documents")]
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            detail=f"Only uploading PDF documents are supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    try:
        await save_file(file)
    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"filename": file.filename, "message": "File uploaded successfully"}

# client.py

import requests
import streamlit as st

st.write("Upload a file to FastAPI")
file = st.file_uploader("Choose a file", type=["pdf"])

if st.button("Submit"):
    if file is not None:
        files = {"file": (file.name, file, file.type)}
        response = requests.post("http://localhost:8000/upload", files=files)
        st.write(response.text)
    else:
        st.write("No file uploaded.")
```

You should now be able to upload files via the Streamlit UI, as you can see in [Figure 5-10](#streamlit_upload).

![bgai 0510](assets/bgai_0510.png)

###### Figure 5-10\. Uploading files via Streamlit to the FastAPI service

With upload functionality implemented, you can now turn your attention to building the RAG module. [Figure 5-11](#rag_module_detailed) shows the detailed pipeline, which opens up the data transformation component in [Figure 5-9](#rag_module).

![bgai 0511](assets/bgai_0511.png)

###### Figure 5-11\. Detailed RAG data processing pipeline

As you can see in [Figure 5-11](#rag_module_detailed), you need to asynchronously fetch the stored files from the hard disk and pass them through a data transformation pipeline prior to storage via an asynchronous database client.

The data transformation pipeline consists of the following parts:

Extractor

Extract content of PDFs and store in text files back onto the hard disk.

Loader

Asynchronously load a text file into memory in chunks.

Cleaner

Remove any redundant whitespace or formatting characters from text chunks.

Embedder

Use a pretrained and self-hosted embedding model to convert text into embedding vectors.

Once users upload their PDF files onto your server’s filesystem via the process shown in [Example 5-8](#upload_file), you can immediately convert them into text files via the `pypdf` library. Since there is no asynchronous library for loading binary PDF files, you will want to convert them into text files first.

[Example 5-9](#rag_extract) shows how to load PDFs, extract and process their content, and then store them as text files.

###### Note

You will need to install several packages to run the upcoming examples:

```py
$ pip install qdrant_client aiofiles pypdf loguru
```

##### Example 5-9\. RAG PDF-to-text extractor

```py
# rag/extractor.py

from pypdf import PdfReader

def pdf_text_extractor(filepath: str) -> None:
    content = ""
    pdf_reader = PdfReader(filepath, strict=True) ![1](assets/1.png)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            content += f"{page_text}\n\n" ![2](assets/2.png)
    with open(filepath.replace("pdf", "txt"), "w", encoding="utf-8") as file: ![3](assets/3.png)
        file.write(content)
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO7-1)

Use the `pypdf` library to open a stream pointer to a PDF file with `strict=True` so that any read errors are logged to the terminal. Note that there is no asynchronous implementation of the `pypdf` library, so the function is declared with a normal `def` keyword. It is important to avoid using this function within an asynchronous function to avoid blocking the event loop that runs the main server thread. You will see how FastAPI background tasks can help solve this problem.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO7-2)

Loop over every page in the PDF document, and extract and append all text content into a long string.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO7-3)

Write the content of the PDF document into a text file for downstream processing. Specify `encoding="utf-8"` to avoid problems on platforms like Windows.

The text extractor will convert the PDF files into simple text files that we can stream into memory in chunks using an asynchronous file loader. Each chunk can then be cleaned and embedded into an embedding vector using an open source embedding model such as `jinaai/jina-embeddings-v2-base-en`, available to download from the [Hugging Face model hub](https://oreil.ly/gI74r).

###### Note

I selected the Jina base embedder since it matches the performance of OpenAI’s proprietary `text-embedding-ada-002` model.

[Example 5-10](#rag_transform) shows the implementation of the RAG data transformation pipeline including the async text loader, cleaner, and embedding functions.

##### Example 5-10\. RAG data transformation functions

```py
# rag/transform.py

import re
from typing import Any, AsyncGenerator

import aiofiles
from transformers import AutoModel

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50  # 50 megabytes

embedder = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True ![1](assets/1.png)
)

async def load(filepath: str) -> AsyncGenerator[str, Any]:
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f: ![2](assets/2.png)
        while chunk := await f.read(DEFAULT_CHUNK_SIZE): ![3](assets/3.png)
            yield chunk ![4](assets/4.png)

def clean(text: str) -> str:
    t = text.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\. ,", "", t)
    t = t.replace("..", ".")
    t = t.replace(". .", ".")
    cleaned_text = t.replace("\n", " ").strip()
    return cleaned_text ![5](assets/5.png)

def embed(text: str) -> list[float]:
    return embedder.encode(text).tolist() ![6](assets/6.png)
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO8-1)

Download and use the open source `jina-embeddings-v2-base-en` model to embed text strings into embedding vectors. Set `trust_remote_code=True` to download model weights and tokenizer configurations. Without this parameter set to `True`, the downloaded model weights will be initialized with random values instead of trained values.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO8-2)

Use the `aiofiles` library to open an asynchronous connections to a file on the filesystem.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO8-3)

Load the content of text documents in chunks for memory-efficient I/O operation.

[![4](assets/4.png)](#co_achieving_concurrency_in_ai_workloads_CO8-4)

Instead of returning a `chunk`, yield it so that the `load()` function becomes an *asynchronous generator*. Asynchronous generators can be iterated with `async for loop`s so that blocking operations within them can be `await`ed to let the event loop start/resume other tasks. Both async `for` loops and normal `for` loops, iterate sequentially over the iterable but async `for` loops allow for iteration over an async iterator.

[![5](assets/5.png)](#co_achieving_concurrency_in_ai_workloads_CO8-5)

Clean the text by removing any extra spaces, commas, dots, and line breaks.

[![6](assets/6.png)](#co_achieving_concurrency_in_ai_workloads_CO8-6)

Use the Jina embedding model to convert a text chunk to an embedding vector.

Once the data is processed into embedding vectors, you can store them into the *vector database*.

Unlike conventional alternatives such as relational databases, a vector database is specifically designed for handling data storage and retrieval operations optimized for *semantic searching*, which yields better results compared to keyword searches that can return suboptimal or incomplete results.

The following code examples require you to run a local instance of the `qdrant` vector database on your local machine for the RAG module. Having a local database setup will give you the hands-on experience of working asynchronously with production-grade vector databases. To run the database in a container, you should have Docker installed on your machine and then pull and run the `qdrant` vector database container.^([7](ch05.html#id843)) If you aren’t familiar with Docker, don’t worry. You will learn more about Docker and containerization in [Chapter 12](ch12.html#ch12).

```py
$ docker pull qdrant/qdrant ![1](assets/1.png)
$ docker run -p 6333:6333 -p 6334:6334 \  ![2](assets/2.png)
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \ ![3](assets/3.png)
    qdrant/qdrant
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO9-1)

Download the `qdrant` vector database image from the `qdrant` repository in the Docker registry.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO9-2)

Run the `qdrant/qdrant` image, and then expose and map container ports `6333` and `6334` to the same ports on the host machine.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO9-3)

Mount the `qdrant` database storage to the host machine filesystem at your project’s root directory.

Since database storage and retrieval are I/O operations, you should use an asynchronous database client. Thankfully, `qdrant` provides an asynchronous database client to work with.

###### Tip

You can use other vector database providers such as Weaviate, Elastic, Milvus, Pinecone, Chroma, or others in replacement of Qdrant. Each has a set of features and limitations to consider for your own use case.

If you’re picking another database provider, make sure there is an asynchronous database client available that you can use.

Instead of writing several functions to store and retrieve data from the database, you can use the repository pattern mentioned in [Chapter 2](ch02.html#ch02). With the repository pattern, you can abstract low-level create, read, update, and delete database operations with defaults that match your use case.

[Example 5-11](#rag_repository) shows the repository pattern implementation for the Qdrant vector database.

##### Example 5-11\. Vector database client setup using the repository pattern

```py
# rag/repository.py

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import ScoredPoint

class VectorRepository: ![1](assets/1.png)
    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        self.db_client = AsyncQdrantClient(host=host, port=port)

    async def create_collection(self, collection_name: str, size: int) -> bool: ![2](assets/2.png)
        vectors_config = models.VectorParams(
            size=size, distance=models.Distance.COSINE ![3](assets/3.png)
        )
        response = await self.db_client.get_collections()

        collection_exists = any(
            collection.name == collection_name
            for collection in response.collections
        )
        if collection_exists: ![4](assets/4.png)
            logger.debug(
                f"Collection {collection_name} already exists - recreating it"
            )
            await self.db_client.delete_collection(collection_name)
            return await self.db_client.create_collection(
                collection_name,
                vectors_config=vectors_config,
            )

        logger.debug(f"Creating collection {collection_name}")
        return await self.db_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=size, distance=models.Distance.COSINE
            ),
        )

    async def delete_collection(self, name: str) -> bool:
        logger.debug(f"Deleting collection {name}")
        return await self.db_client.delete_collection(name)

    async def create(
        self,
        collection_name: str,
        embedding_vector: list[float],
        original_text: str,
        source: str,
    ) -> None:
        response = await self.db_client.count(collection_name=collection_name)
        logger.debug(
            f"Creating a new vector with ID {response.count} "
            f"inside the {collection_name}"
        )
        await self.db_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=response.count,
                    vector=embedding_vector,
                    payload={
                        "source": source,
                        "original_text": original_text,
                    },
                )
            ],
        )

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        retrieval_limit: int,
        score_threshold: float, ![5](assets/5.png)
    ) -> list[ScoredPoint]:
        logger.debug(
            f"Searching for relevant items in the {collection_name} collection"
        )
        response = await self.db_client.query_points(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=retrieval_limit,
            score_threshold=score_threshold,
        )
        return response.points
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO10-1)

Use the repository pattern to interact with the vector database via an asynchronous client. Normally, in the repository pattern you will implement the `create`, `get`, `update`, and `delete` methods. But for now let’s implement the `create_​col⁠lection`, `delete_collection`, `create`, and `search` methods.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO10-2)

Vectors need to be stored in a collection. A collection is a named set of points that you can use during a search. Collections are similar to tables in a relational database.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO10-3)

Let the database know that any vectors in this collection should be compared via the cosine similarity calculation that calculates distances between vectors.

[![4](assets/4.png)](#co_achieving_concurrency_in_ai_workloads_CO10-4)

Check whether a collection exists before creating a new one. Otherwise, re-create the collection.

[![5](assets/5.png)](#co_achieving_concurrency_in_ai_workloads_CO10-5)

Set the `retrieval_limit` and `score_threshold` to limit the number of items in the search results.

The `VectorRepository` class should now make it easier to interact with the database.

When storing vector embeddings, you will also store some *metadata* including the name of the source document, the location of the text within source, and the original extracted text. RAG systems rely on this metadata to augment the LLM prompts and to show source information to the users.

###### Tip

Currently, converting text to embedding vectors is an irreversible process. Therefore, you will need to store the text that created the embedding with the embedding vector as metadata.

You can now extend the `VectorRepository` and create the `VectorService` that allow you to chain together the data processing and storage pipeline, as shown in [Example 5-12](#rag_db_service).

##### Example 5-12\. Vector database service

```py
# rag/service.py

import os

from loguru import logger
from .repository import VectorRepository
from .transform import clean, embed, load

class VectorService(VectorRepository): ![1](assets/1.png)
    def __init__(self):
        super().__init__()

    async def store_file_content_in_db( ![2](assets/2.png)
        self,
        filepath: str,
        chunk_size: int = 512,
        collection_name: str = "knowledgebase",
        collection_size: int = 768,
    ) -> None:
        await self.create_collection(collection_name, collection_size)
        logger.debug(f"Inserting {filepath} content into database")
        async for chunk in load(filepath, chunk_size): ![3](assets/3.png)
            logger.debug(f"Inserting '{chunk[0:20]}...' into database")

            embedding_vector = embed(clean(chunk))
            filename = os.path.basename(filepath)
            await self.create(
                collection_name, embedding_vector, chunk, filename
            )

vector_service = VectorService() ![4](assets/4.png)
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO11-1)

Create the `VectorService` class by inheriting the `VectorRepository` class so that you can use and extend common database operation methods from [Example 5-11](#rag_repository).

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO11-2)

Use the `store_file_content_in_db` service method to asynchronously load, transform, and store raw text documents into the database in chunks.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO11-3)

Use an asynchronous generator `load()` to load text chunks from a file asynchronously.

[![4](assets/4.png)](#co_achieving_concurrency_in_ai_workloads_CO11-4)

Create an instance of the `VectorService` to import and use across the application.

The final step in the RAG data processing and storage pipeline is to run the text extraction and storage logic within the `file_upload_controller` as background tasks. The implementation is shown in [Example 5-13](#rag_data_processor) so that the handler can trigger both operations in the background after responding to the user.

##### Example 5-13\. Update the upload handler to process and store PDF file content in the vector database

```py
# main.py

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    UploadFile,
    status,
    HTTPException,
)
from typing import Annotated
from rag import pdf_text_extractor, vector_service

@app.post("/upload")
async def file_upload_controller(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")],
    bg_text_processor: BackgroundTasks, ![1](assets/1.png)
):
    ... # Raise an HTTPException if data upload is not a PDF file
    try:
        filepath = await save_file(file)
        bg_text_processor.add_task(pdf_text_extractor, filepath) ![2](assets/2.png)
        bg_text_processor.add_task( ![3](assets/3.png)
            vector_service.store_file_content_in_db,
            filepath.replace("pdf", "txt"),
            512,
            "knowledgebase",
            768,
        )

    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"filename": file.filename, "message": "File uploaded successfully"}
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO12-1)

Inject the FastAPI background tasks feature into the handler for processing file uploads in the background. FastAPI background tasks will be executed in order shortly after the handler sends a response to the client.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO12-2)

Run the PDF text-extraction function in the background after retuning a response to the client. Since the `pdf_text_extractor` is a synchronous function, FastAPI will run this function on a separate thread within the thread pool to avoid blocking the event loop.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO12-3)

Run the `vector_service.store_file_content_in_db` asynchronous function in the background on the FastAPI managed event loop as soon as the `pdf_text_extractor` has finished processing. Set the function to load content of the text document in chunks of 512 characters and store them in the `knowledgebase` vector collection, which accepts vectors of size 768.

After building the RAG data storage pipeline, you can now focus on the search-and-retrieval system, which will allow you to augment the user prompts to the LLM, with knowledge from the database. [Example 5-14](#rag_generation) integrates the RAG search-and-retrieval operations with the LLM handler to augment the LLM prompts with additional context.

##### Example 5-14\. RAG integration with the LLM-serving endpoint

```py
# dependencies.py

from rag import vector_service
from rag.transform import embed
from schemas import TextModelRequest, TextModelResponse

async def get_rag_content(body: TextModelRequest = Body(...)) -> str: ![1](assets/1.png)
    rag_content = await vector_service.search( ![2](assets/2.png)
        "knowledgebase", embed(body.prompt), 3, 0.7
    )
    rag_content_str = "\n".join( ![3](assets/3.png)
        [c.payload["original_text"] for c in rag_content]
    )

    return rag_content_str

# main.py

... # other imports
from dependencies import get_rag_content, get_urls_content

@app.post("/generate/text", response_model_exclude_defaults=True)
async def serve_text_to_text_controller(
    request: Request,
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content),
    rag_content: str = Depends(get_rag_content), ![4](assets/4.png)
) -> TextModelResponse:
    ... # Raise HTTPException for invalid models
    prompt = body.prompt + " " + urls_content + rag_content
    output = generate_text(models["text"], prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO13-1)

Create the `get_rag_content` dependency function for injection into the LLM-serving handler. This dependency has access to the request `body` and subsequently the user `prompt`.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO13-2)

Use the `vector_service` to search the database for content relevant to the user `prompt`. Convert the user `prompt` to an embedding using the `embed` function when passing to the `vector_service.search` function. Only retrieve the three most relevant items if their cosine similarity score is above `0.7` (or 70%).

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO13-3)

Merge the text payload of the top three most relevant retrieved items as `rag_​con⁠tent_str` and return it.

[![4](assets/4.png)](#co_achieving_concurrency_in_ai_workloads_CO13-4)

Inject the results of the `get_rag_content` dependency function into the LLM handler to augment the final prompt to the LLM with content from the vector database `knowledgebase`. The LLM handler can now fetch content of web pages and the RAG vector database.

If you now visit your browser and upload a PDF document, you should be able to ask questions about it to your LLM. [Figure 5-12](#rag_results) shows my experiment with the service by uploading a sample of this book in its raw form and asking the LLM to describe who I am.

###### Note

Depending on the model and size of the inputs, you may observe performance degradations or exceptions like token length limit issues.

![bgai 0512](assets/bgai_0512.png)

###### Figure 5-12\. Leveraging RAG to provide answers in response to user queries

Congratulations! You now have a fully working RAG system enabled by open source models and a vector database.

This longer project served as a hands-on tutorial for learning concepts related to asynchronous programming and I/O operations with the filesystem and a vector database by building a RAG module for your LLM system. Note that the RAG system we just built together still has many limitations:

*   Text splitting may split words in half leading to poor retrieval and LLM confusion.

*   The LLM may still produce hallucinations and inconsistent outputs even with the augmented prompts.

*   The search-and-retrieval system may perform poorly in certain instances.

*   The augmented prompts may exceed the LLM context window.

*   The retrieved information from the database may lack the relevant facts due to an outdated or incomplete knowledge base, ambiguous queries, or poor retrieval algorithm.

*   The retrieved context may not be ordered based on relevance to the user query.

You can work on improving the RAG module further by implementing various other techniques, which I will not cover in this book:

*   Optimize text splitting, chunk sizing, cleaning and embedding operations.

*   Perform query transformations using the LLM to aid the retrieval and augmentation system via techniques such as prompt compression, chaining, refining, and aggregating, etc., to reduce hallucinations and improve LLM performance.

*   Summarize or break down large augmented prompts to feed the context into the models using a sliding window approach.

*   Enhance retrieval algorithms to handle ambiguous queries and implement fallback mechanisms for incomplete data.

*   Enhance the retrieval performance with methods such as *maximal marginal relevance* (MMR) to enrich the augmentation process with more diverse documents.

*   Implement other advanced RAG techniques like retrieval reranking and filtering, hierarchical database indices, RAG fusion, retrieval augmented thoughts (RAT), etc., to improve the overall generation performance.

I’ll let you research these techniques in more detail and implement them as additional exercises on your own.

In the next section, well review other techniques for optimizing your GenAI services to avoid blocking the server with compute-bound operations such as model inference.

# Optimizing Model Serving for Memory- and Compute-Bound AI Inference Tasks

So far, we’ve looked at optimizing the operations of our service that are I/O bound. You learned to leverage asynchronous programming to interact with the web, databases, and files by building a web scraper and a RAG module.

Using async tools and techniques, your service remained responsive when interacting with the web, the filesystem, and databases. However, if you’re self-hosting the model, switching to async programming techniques won’t fully eliminate the long waiting times. This is because the bottleneck will be model inference operations.

## Compute-Bound Operations

You can speed up the inference by running models on GPUs to massively parallelize computations. Modern GPUs have staggering compute power measured by the number of *floating-point* operations per second (FLOPS), with modern GPUs reaching teraflops (NVIDIA A100) or petaflops (NVIDIA H100) of compute. However, despite their significant power and parallelization capabilities, modern GPU cores are often underutilized under concurrent workloads with larger models.

When self-hosting models on GPUs, model parameters are loaded from disk to RAM (I/O bound) and then moved from RAM to the GPU high-bandwidth memory by the CPU (memory bound). Once model parameters are loaded on the GPU memory, inference is performed (compute bound).

Counterintuitively, model inference for larger GenAI models such as SDXL and LLMs is not I/O- or compute-bound, but rather memory-bound. This means it takes more time to load 1 MB of data into GPU’s compute cores than it takes for those compute cores to process 1 MB of data. Inevitably, to maximize the concurrency of your service, you will need to *batch* the inference requests and fit the largest batch size you can into the GPU high-bandwidth memory.

Therefore, even when using async techniques and latest GPUs, your server can be blocked waiting for billions of model parameters to be loaded to the GPU high-bandwidth memory during each request. To avoid blocking the server, you can decouple the memory-bound model-serving operations from your FastAPI server by externalizing model serving, as we touched upon in [Chapter 3](ch03.html#ch03).

Let’s see how to delegate model serving to another process.

## Externalizing Model Serving

You have several options available to you when externalizing your model-serving workloads. You can either host models on another FastAPI server or use specialized model inference servers.

Specialized inference servers support only a limited set of GenAI model architectures. However, if your model architecture is supported, you will save a lot of time not having to implement inference optimizations yourself. For instance, if you need to self-host LLMs, LLM-serving frameworks can perform several inference optimizations for you such as batch processing, tensor parallelism, quantization, caching, streaming outputs, GPU memory management, etc.

Since we’ve been mostly working with LLMs in this chapter, I will show you how to integrate vLLM, an open source LLM server that can start a FastAPI server for you matching the OpenAI API specification. vLLM also has seamless integration with popular open source Hugging Face model architectures including GPT, Llama, Gemma, Mistral, Falcon, etc.

###### Note

At the time of writing, other LLM hosting servers you can use include NVIDIA Triton Inference Server, Ray Serve, Hugging Face Inference, and OpenLLM, among others.

There are features, benefits, and drawbacks to using each including the supported model architectures. I recommend researching these servers prior to adopting them in your own use cases.

You can start your own vLLM FastAPI server via a single command, as shown in [Example 5-15](#vllm). To run the code in [Example 5-15](#vllm), you will need to install `vllm` using:

```py
$ pip install vllm
```

###### Warning

At the time of writing, vLLM only supports Linux platforms (including WSL) with NVIDIA-compatible GPUs to run CUDA toolkit dependencies. Unfortunately, you can’t install vLLM on Mac or Windows machines for local testing.

vLLM is designed for production inference workloads on NVIDIA GPUs in Linux environments where the server can delegate requests to multiple GPU cores via *tensor parallelism*. It does also support distributed computing when scaling services beyond a single machine via its Ray Serve dependency.

Please consult vLLM documentation for more details related to distributed inference and serving.

##### Example 5-15\. Starting the vLLM FastAPI OpenAI API server for TinyLlama on a Linux machine with 4x 16 GB NVIDIA T4 GPUs

```py
$ python -m vllm.entrypoints.openai.api_server \ ![1](assets/1.png)
--model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \ --dtype float16 \ ![2](assets/2.png)
--tensor-parallel-size 4 \ ![3](assets/3.png)
--api-key "your_secret_token" ![4](assets/4.png)
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO14-1)

Start an OpenAI-compatible API server with FastAPI to serve the TinyLlama model.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO14-2)

Use the `float16` medium precision data type. `float16` is compatible with GPU hardware, whereas `bfloat16` is generally compatible with CPU hardware.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO14-3)

Leverage vLLM tensor parallelism feature to run the API server on four GPUs.

[![4](assets/4.png)](#co_achieving_concurrency_in_ai_workloads_CO14-4)

Set a secret token for basic authentication to secure the LLM server. This is useful for secure machine-to-machine communication, for instance, to directly communicate with your current FastAPI service.

With the vLLM FastAPI server up and running, you can now replace the model-serving logic in your current service with network calls to the vLLM server. Refer to [Example 5-16](#vllm_fastapi_text_generation) for implementation details.

##### Example 5-16\. Replace model serving with asynchronous API calls to the new vLLM server

```py
# models.py

import os
import aiohttp
from loguru import logger

async def generate_text(prompt: str, temperature: float = 0.7) -> str:
    system_prompt = "You are an AI assistant"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    data = {"temperature": temperature, "messages": messages}
    headers = {"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY')}"}
try:
   async with aiohttp.ClientSession() as session: ![1](assets/1.png)
        response = await session.post(
            "http://localhost:8000/v1/chat", json=data, headers=headers
        )
        predictions = await response.json()
except Exception as e:
    logger.error(f"Failed to obtain predictions from vLLM - Error: {e}")
    return (
        "Failed to obtain predictions from vLLM - "
        "See server logs for more details"
    )
try:
    output = predictions["choices"][0]["message"]["content"] ![2](assets/2.png)
    logger.debug(f"Generated text: {output}")
    return output
except KeyError as e:
    logger.error(f"Failed to parse predictions from vLLM - Error: {e}")
    return (
        "Failed to parse predictions from vLLM - "
        "See server logs for more details"
    )
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO15-1)

Use `aiohttp` to create an asynchronous session for sending `POST` requests to the vLLM FastAPI server. This logic replaces the Hugging Face model pipeline inference logic on the current FastAPI server.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO15-2)

Since the vLLM server is OpenAI compatible, you can access the output content by following the OpenAI API specification.

Next, remove the code related to the FastAPI lifespan so that your current service won’t load the TinyLlama model. You can achieve this by following the code in [Example 5-17](#vllm_fastapi_handler).

##### Example 5-17\. Remove the FastAPI lifespan and update the text generation handler to be asynchronous

```py
# main.py

from fastapi import FastAPI, Request
from schemas import TextModelRequest, TextModelResponse
from models import generate_text

# Remove the asynccontextmanager to remove TinyLlama from FastAPI ![1](assets/1.png)
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     models["text"] = load_text_model()
#     yield
#     models.clear()

# Remove the `lifespan` argument from `FastAPI()`
app = FastAPI()

@app.post("/generate/text")
async def serve_text_to_text_controller(
    request: Request, body: TextModelRequest
) -> TextModelResponse: ![2](assets/2.png)
    ...  # controller logic
    output = await generate_text(body.prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO16-1)

There is no need to use FastAPI `lifespan` anymore since the model is now served by an external vLLM FastAPI server.

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO16-2)

Make `serve_text_to_text_controller` an async route handler as it is now performing I/O operations to the vLLM server. It is no longer running synchronous compute-bound model inference operations as those are delegated to the vLLM server to manage.

Congratulations, you’ve now achieved concurrency with your AI inference workloads. You implemented a form of multiprocessing on a single machine by moving your LLM inference workloads to another server. Both servers are now running on separate cores with your LLM server delegating work to multiple GPU cores, leveraging parallelism. This means your main server is now able to process multiple incoming requests and do other tasks than processing one LLM inference operation at a time.

###### Tip

Bear in mind that any concurrency you’ve achieved so far has been limited to a single machine.

To support more concurrent users, you may need more machines with CPU and GPU cores. At that point, distributed computing frameworks like Ray Serve and Kubernetes can help to scale and orchestrate your services beyond a single worker machine using parallelism.

Before integrating vLLM, you would experience long waiting times between requests because your main server was too busy running inference operations. With vLLM, there is now a massive reduction in latency and increase in throughput of your LLM service.

In addition to model compression mechanisms like quantization, vLLM uses other optimization techniques including continuous request batching, cache partitioning (paged attention), reduced GPU memory footprint via memory sharing, and streaming outputs to achieve smaller latency and high throughput.

Let’s look at both the request batching and paged attention mechanisms in more detail to understand how to further optimize LLM inference.

### Request batching and continuous batching

As we discussed in [Chapter 3](ch03.html#ch03), LLMs produce the next token prediction in an autoregressive manner, as you can see in [Figure 5-13](#autoregressive_prediction5).

![bgai 0513](assets/bgai_0513.png)

###### Figure 5-13\. Autoregressive prediction

This means the LLMs must perform several inference iterations in a loop to produce a response, and each iteration produces a single output token. The input sequence grows as each iteration’s output token is appended to the end, and the new sequence is forwarded to the model in the next iteration step. Once the model generates an end-of-sequence token, the generation loop stops. Essentially, the LLM produces a sequence of completion tokens, stopping only after producing a stop token or reaching a maximum sequence length.

The LLM must calculate several attention maps for each token in the sequence so that it can iteratively make the next token predictions.

Fortunately, GPUs can parallelize the attention map calculations for each iteration. As you learned, these attention maps are capturing the meaning and context of each token within the input sequence and are expensive to calculate. Therefore, to optimize inference, LLMs use *key-value* (KV) *caching* to store calculated maps in the GPU memory.

###### Tip

The attention map formula computes a *value (V)* based on a given *query (Q)* and a *key (K)*.

> Q = KV

This calculation has to be done for each token in the sequence but luckily can be vectorized using large matrix multiplication operations on a GPU.

However, storing parameters on the GPU memory for reuse between iterations can consume huge chunks of GPU memory. For instance, a 13B-parameter model consumes nearly 1 MB of state for each token in a sequence on top of all those 13B model parameters. This means there is a limited number of tokens you can store in memory for reuse.

If you’re using a higher-end GPU, such as the A100 with 40 GB RAM, you can only hold 14 K tokens in memory at once, while the rest of the memory is used up for storing 26 GB of model parameters. In short, the GPU memory consumed scales with the base model size plus the length of the token sequence.

To make matters worse, if you need to serve multiple users concurrently by batching requests, your GPU memory has to be shared between multiple LLM inferences. As a result, you have less memory to store longer sequences, and your LLM is constrained to a shorter context window. On the other hand, if you want to maintain a large context window, then you can’t handle more concurrent users. As an example, a sequence length of 2048 means that your batch size will be limited to 7 concurrent requests (or 7 prompt sequences). Realistically, this is an upper-bound limit and doesn’t leave room for storing intermediate computations, which will reduce the aforementioned numbers even further.

What this all means is that LLMs are failing to fully saturate the GPU’s available resources. The primary reason is that a significant portion of the GPU’s memory bandwidth is consumed in loading the model parameters instead of processing inputs.

The first step to reduce the load on your services is to integrate the most efficient models. Often, smaller and more compressed models could do the job you’re asking of them, with a similar performance to their larger counterparts.

Another suitable solution to the GPU underutilization problem is to implement *request batching* where the model processes multiple inputs in groups, reducing the overhead of loading model parameters for each request. This is more efficient in using the chip’s memory bandwidth, leading to higher compute utilization, higher throughput, and less expensive LLM inference. LLM inference servers like vLLM take advantage of batching plus fast attention, KV caching, and paged attention mechanisms to maximize throughput.

You can see the difference of response latency and throughput with and without batching in [Figure 5-14](#with_without_batching).

![bgai 0514](assets/bgai_0514.png)

###### Figure 5-14\. LLM server response latency and throughput with and without batching

There are two ways to implement batching:

Static batching

The size of the batch remains constant.

Dynamic or continuous batching

The size of batch is determined based on demand.

In *static batching*, we wait for a predetermined number of incoming requests to arrive before we batch and process them through the model. However, since requests can finish at any time in a batch, we’re effectively delaying responses to every request—​and increasing latency—​until the whole batch is processed.

Releasing the GPU resource can also be tricky when processing a batch and adding new requests to the batch that may be at different completion states. As a result, the GPU remains underutilized as the generated sequences within a batch vary and don’t match the length of the longest sequence in that batch.

[Figure 5-15](#static_batching) illustrates static batching in the context of LLM inference.

![bgai 0515](assets/bgai_0515.png)

###### Figure 5-15\. Static batching with fixed batch size

In [Figure 5-15](#static_batching) you will notice the white blocks representing underutilized GPU computation time. Only one input sequence in the batch saturated the GPU across the batch’s processing timeline.

Aside from adding unnecessary waiting times and not saturating the GPU utilization, what makes static batching problematic is that users of an LLM-powered chatbot service won’t be providing fixed-length prompts or expect fixed-length outputs. The variance in generation outputs could cause massive underutilization of GPUs.

A solution is to avoid assuming fixed input or output sequences and instead set dynamic batch sizes during the processing of a batch. In *dynamic* or *continuous batching*, the size of batch can be set based on the incoming request sequence length and the available GPU resource. With this approach, new generation requests can be inserted in a batch by replacing completed requests to yield higher GPU utilization than static batching.

[Figure 5-16](#dynamic_batching) shows how dynamic or continuous batching can fully saturate the GPU resource.

![bgai 0516](assets/bgai_0516.png)

###### Figure 5-16\. Dynamic/continuous batching with variable batch size

While the model parameters are loaded, requests can keep flowing in, and the LLM inference server schedules and insert them into the batch to maximize GPU usage. This approach leads to higher throughput and reduced latency.

If you’re building a LLM inference server, you will probably want to bake in the continuous batching mechanism into your server. However, the good news is that the vLLM server already provides continuous batching out of the box with its FastAPI server, so you don’t have to implement all of that yourself. Additionally, it also ships with another important GPU optimization feature, which sets it apart from other alternative LLM inference frameworks: paged attention.

### Paged attention

Efficient memory usage is a critical challenge for systems that handle high-throughput serving, particularly for LLMs. For faster inference, today’s models rely on *KV caches* to store and reuse attention maps, which grow exponentially as input sequence lengths increase.

*Paged attention* is a novel solution designed to minimize the memory demands of these KV caches, subsequently enhancing the memory efficiency of LLMs and making them more viable for use on devices with limited resources. In transformer-based LLMs, attention key and value tensors are generated for each input token to capture essential context. Instead of recalculating these tensors at every step, they’re saved in the GPU memory as a KV cache, which serves as the model’s memory. However, the KV cache can grow to enormous sizes, such as 40 GB for a model with 13B parameters, posing a significant challenge for efficient storage and access, particularly on hardware with constrained resources.

Paged attention introduces a method that breaks down the KV cache into smaller, more manageable segments called *pages*, each holding a KV vector for a set number of tokens. With this segmentation, paged attention can efficiently load and access KV caches during the attention computations. You can compare this technique to how the virtual memory is managed by operating systems, where the logical arrangement of data is separated from its physical storage. Essentially, a block table maps the logical blocks to physical ones, allowing for dynamic allocation of memory as new tokens are processed. The core idea is to avoid memory fragmentation by leveraging logical blocks (instead of physical ones) and use a mapping table to quickly access data stored in a paged physical memory.

You can break down the paged attention mechanism into several steps:

Partitioning the KV cache

The cache is split into fixed-size pages, with each containing a portion of the key-value pairs.

Building the lookup table

A table is created to map query keys to their corresponding pages, facilitating quick allocation and retrieval.

Selective loading

Only the necessary pages for the current input sequence are loaded during inference, reducing the memory footprint.

Attention computation

The model computes attention using the key-value pairs from the loaded pages. This approach aims to make LLMs more accessible by addressing the memory bottleneck, potentially enabling their deployment on a wider range of devices.

The aforementioned steps enable the vLLM server to maximize memory usage efficiency through the mapping of physical and logical memory blocks so that the KV cache is efficiently stored and retrieved during generation.

In a [blog post published on Anyscale.com](https://oreil.ly/WgRfJ), the authors have researched and compared the performance of various LLM-serving frameworks during inference. The authors concluded that leveraging both paged attention and continuous batching mechanisms are so powerful in optimizing GPU memory usage that the vLLM server was able to reduce latencies by 4 times and throughput by up to 23 times.

In the next section, we will turn our attention to GenAI workloads that can take a long time to process and are compute-bound. This is mostly the case with large non-LLM models such as SDXL where performing batch inferences (such as batch image generation) for multiple users may prove challenging.

# Managing Long-Running AI Inference Tasks

With the ability to host models in a separate process outside the FastAPI event loop, you can turn your attention to blocking operations that take a long time to complete.

In the previous section, you leveraged specialized frameworks such as vLLM to externally host and optimize the inference workloads of your LLMs. However, you may still run into models that can take significant time to generate results. To prevent your users from waiting, you should manage tasks that generate models and take a long time to complete.

Several GenAI models such as Stable Diffusion XL may take several minutes, even on a GPU, to produce results. In most cases, you can ask your users to wait until the generation process is complete. But if users are using a single model simultaneously, the server will have to queue these requests. When your users work with generative models, they need to interact with it several times to guide the model to the results they want. This usage pattern creates a large backlog of requests, and users at the end of the queue will have to wait a long time before they see any results.

If there was a way to handle long-running tasks without making the users wait, that would be perfect. Luckily, FastAPI provides a mechanism for solving these kinds of problems.

FastAPI’s *background tasks* is a mechanism you can leverage to respond to users while your models are busy processing the request. You’ve been briefly introduced to this feature while building the RAG module where a background task was populating a vector database with the content of the uploaded PDF documents.

Using background tasks, your users can continue sending requests or carry on with their day without having to wait. You can either save the results to disk or a database for later retrieval or provide a polling system so that their client can ping for updates as the model processes the requests. Another option is to create a live connection between the client and the server so that their UI is updated with the results as soon as it becomes available. All these solutions are doable with FastAPI’s background tasks.

[Example 5-18](#fastapi_background_tasks) shows how to implement background tasks to handle long-running model inferences.

##### Example 5-18\. Using background tasks to handle long-running model inference (e.g., batch generating images)

```py
# main.py

from fastapi import BackgroundTasks
import aiofiles

...

async def batch_generate_image(prompt: str, count: int) -> None:
    images = generate_images(prompt, count) ![1](assets/1.png)
    for i, image in enumerate(images):
        async with aiofiles.open(f"output_{i}.png", mode='wb') as f:
            await f.write(image) ![2](assets/2.png)

@app.get("/generate/image/background")
def serve_image_model_background_controller(
    background_tasks: BackgroundTasks, prompt: str, count: int ![3](assets/3.png)
):
    background_tasks.add_task(batch_generate_image, prompt, count) ![4](assets/4.png)
    return {"message": "Task is being processed in the background"} ![5](assets/5.png)
```

[![1](assets/1.png)](#co_achieving_concurrency_in_ai_workloads_CO17-1)

Generate multiple images in a batch using an external model-serving API like [Ray Serve](https://oreil.ly/NjlV4).

[![2](assets/2.png)](#co_achieving_concurrency_in_ai_workloads_CO17-2)

Loop over the generated images and asynchronously save each to disk using the `aiofiles` library. In production, you can also save output images to cloud storage solutions that clients can directly fetch from.

[![3](assets/3.png)](#co_achieving_concurrency_in_ai_workloads_CO17-3)

Enable the controller to perform background tasks.

[![4](assets/4.png)](#co_achieving_concurrency_in_ai_workloads_CO17-4)

Pass the `batch_generate_image` function definition to a FastAPI background tasks handler with the required arguments.

[![5](assets/5.png)](#co_achieving_concurrency_in_ai_workloads_CO17-5)

Return a generic success message to the client before processing the background task so that the user is not kept waiting.

In [Example 5-18](#fastapi_background_tasks), you’re allowing FastAPI to run inference operations in the background (via an external model server API) such that the event loop remains unblocked to process other incoming requests. You can even run multiple tasks in the background, such as generating images in batches (in separate processes) and sending notification emails. These tasks are added to a queue and processed sequentially without blocking the user. You can then store the generated images and expose an additional endpoint that clients can use to poll for status updates and to retrieve the inference results.

###### Warning

Background tasks run in the same event loop. They won’t provide true parallelism; they only provide concurrency.

If you run heavy CPU-bound operations like AI inference in background tasks, it’ll block the main event loop until all background tasks are completed. Similarly, be careful with async background tasks. If you don’t await the blocking I/O operations, the task will block the main server from responding to other requests, even if it runs in the background. FastAPI runs nonasync background tasks in an internal thread pool.

While FastAPI’s background tasks are a wonderful tool for handling simple batch jobs, it doesn’t scale and can’t handle exceptions or retries as well as specialized tools. Other ML-serving frameworks like Ray Serve, BentoML, and vLLM may handle model serving better at scale by providing features such as request batching. More sophisticated tools like Celery (a queue manager), Redis (a caching database), and RabbitMQ (a message broker) can also be used in combination to implement a more robust and reliable inference pipeline.

# Summary

This chapter explored the complex aspects of applying concurrency in AI systems.

You were introduced to concurrency and parallelism concepts, including several types of blocking operations that prevent you from simultaneously serving users. You discovered concurrency techniques such as multithreading, multiprocessing, and asynchronous programming alongside their differences, similarities, benefits, and drawbacks in various use cases.

Next, you learned about thread pools and event loops, particularly in a FastAPI server environment, and understood their roles in processing requests concurrently. This involved understanding how and why the server can be blocked if you’re not careful how you declare your route handlers.

Later, you discovered how to implement asynchronous programming to manage I/O blocking operations. Through hands-on examples, you developed a deeper understanding of asynchronous interactions with databases and the web content, constructing both a web scraper and a RAG module.

Furthermore, you saw why larger GenAI models can be memory hungry and create memory-bound blocking operations. As part of this, you were introduced to memory optimization techniques such as continuous batching and paged attention in serving LLMs to minimize memory-related bottlenecks.

Finally, you learned about approaches for handling long-running AI inference processes, ensuring your service remains responsive over prolonged operations.

With your knowledge from this chapter, you’re now prepared to apply concurrency principles to your own services, crafting resilient, scalable, and high-performing AI applications.

The ability to handle multiple users simultaneously is a significant milestone. But there are additional optimizations you can perform to improve the user experience of your GenAI services even further. You can provide real-time updates via streaming technologies to progressively show near real-time results to users during generation. This is particularly useful for LLMs that may have longer generation times in conversation scenarios.

The upcoming chapter will explore AI streaming workloads, detailing the use of real-time communication technologies like server-sent events (SSE) and WebSocket (WS). You will learn the difference between these technologies and how to implement model streaming by building endpoints for real-time text-to-text, text-to-speech, and speech-to-text interactions.

# Additional References

*   Kwon, W., et al. (2023). [“Efficient Memory Management for Large Language Model Serving with PagedAttention”](https://oreil.ly/PtCqL). arXiv preprint arXiv:2309.06180.

*   Lewis, P., et al. (2022). [“Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”](https://oreil.ly/r5yVL). arXiv preprint arXiv:2005.11401.

^([1](ch05.html#id795-marker)) A core is an individual processing unit within a CPU or GPU that executes instructions. Modern CPUs and GPUs have multiple cores to perform tasks simultaneously.

^([2](ch05.html#id802-marker)) Multithreading in most languages is parallel (running on multiple cores) and not concurrent. Python is changing over the next coming versions to do the same (free-threaded Python).

^([3](ch05.html#id817-marker)) You can also find a custom implementation in [OpenAI Cookbook on GitHub](https://oreil.ly/8E7GQ).

^([4](ch05.html#id822-marker)) The cost of setting up the threads is still incurred; it’s just done early to avoid doing it on the fly later.

^([5](ch05.html#id830-marker)) P. Lewis et al. (2022), [“Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”](https://oreil.ly/GCk08), arXiv preprint arXiv:2005.11401.

^([6](ch05.html#id841-marker)) A dot product operation multiplies components of two vectors and then sums the results. It can be used to calculate the cosine of the angle between the vectors to quantify their similarity in direction (i.e., alignment). Vector databases use it to perform semantic search on document embeddings.

^([7](ch05.html#id843-marker)) Refer to the [Docker documentation](https://oreil.ly/V4itQ) for installation instructions.