# Chapter 10\. Interfacing LLMs with External Tools

In the first two parts of the book, we have seen how impactful standalone LLMs can be in solving a wide variety of tasks. To effectively harness their full range of capabilities in an organization, they have to be integrated into the existing data and software ecosystem. Unlike traditional software systems, LLMs can generate autonomous actions to interact with other ecosystem components, bringing a degree of flexibility never seen before in the software world. This flexibility unlocks a whole host of use cases that were previously considered impossible.

Another reason we need LLMs to interact with software and external data: as we know all too well, current LLMs have significant limitations, some of which we discussed in [Chapter 1](ch01.html#chapter_llm-introduction). To recap some key points:

*   Since it is expensive to retrain LLMs or keep them continuously updated, they have a knowledge cutoff date and thus possess no knowledge of more recent events.

*   Even though they are getting better over time, LLMs don’t always get math right.

*   They can’t provide factuality guarantees or accurately cite the sources of their outputs.

*   Feeding them your own data effectively is a challenge; fine-tuning is nontrivial, and in-context learning is limited by the length of the effective context window.

As we have been noticing throughout the book, the consolidation effect is leading us to a future (unless we hit a technological wall) where many of the aforementioned limitations might be addressed within the model itself. But we don’t necessarily need to wait for that moment to arrive, as many of these limitations can be addressed today by offloading the tasks and subtasks to external tools.

In this chapter, we will define the three canonical LLM interaction paradigms and provide guidance on how to choose between them for your application. Broadly speaking, there are two types of external entities that LLMs need to interact with: data stores and software/models, collectively called tools. We will demonstrate how to interface LLMs with various tools like APIs and code interpreters. We will show how to make the best use of libraries like LangChain and LlamaIndex, which have vastly simplified LLM integrations. We will explore the various scaffolding software that needs to be constructed to facilitate seamless interactions with the environment. We will also push the limits of what today’s LLMs are capable of, by demonstrating how they can be deployed as an agent that can make autonomous decisions.

# LLM Interaction Paradigms

Suppose you have a task you want the LLM to solve. There are several possible options:

*   The LLM uses its own memory and capabilities encoded in its parameters to solve the task.

*   You feed the LLM all the context it needs to solve the task within the prompt, and the LLM uses the provided context and its capabilities to solve it.

*   The LLM doesn’t have the requisite information or skills to solve this task, so you update the model parameters (fine-tuning etc., as detailed in Chapters [6](ch06.html#llm-fine-tuning)–[8](ch08.html#ch8)) so that it is able to activate the skills and knowledge needed to solve it.

*   You don’t know a priori what context is needed to solve the task, so you use mechanisms to automatically fetch the relevant context and insert it into the prompt (passive approach).

*   You provide explicit instructions to the LLM on how to interact with external tools and data stores to solve your task, which the LLM follows (explicit approach).

*   The LLM breaks the task into multiple subtasks if needed, interacts with its environment to gather the information/knowledge needed to solve the task, and delegates subtasks to external models and tools when it doesn’t have the requisite capabilities to solve that subtask (autonomous approach).

As you can see, the last three involve the LLM interacting with its environment (passive, explicit, and autonomous). Let’s explore the three interaction paradigms in detail.

## Passive Approach

[Figure 10-1](#passive-interaction) shows the typical workflow of an application that involves an LLM passively interacting with a data store.

![Passive Interaction](assets/dllm_1001.png)

###### Figure 10-1\. An LLM passively interacting with a data store

A large number of use cases involve leveraging LLMs to use your own data. Examples include building a question-answering assistant over your company’s internal knowledge base that is spread over a bunch of Notion documents, or an airline chatbot that responds to customer queries about flight status or booking policies.

To allow the LLM to access external information, we need two types of components: “data stores” that contain the required information and retrieval engines that can retrieve relevant data from data stores given a query. The retrieval engine can be powered by an LLM itself, or it can be as simple as a keyword-matching algorithm. The data store(s) can be a repository of data like a database, knowledge graph, vector database, or even just a collection of text files. Data in the data store is represented and indexed to make retrieval more efficient. Data representation, indexing, and retrieval are topics important enough to merit their own chapter: we will defer detailed discussions on them to [Chapter 11](ch11.html#chapter_llm_interfaces).

When a user issues a query, the retrieval engine uses the query to find the documents or text segments that are most relevant to answering this query. After ensuring that these fit into the context window of the LLM, they are fed to the LLM along with the query. The LLM is expected to answer the query given the relevant context provided in the prompt. This approach is popularly known as RAG, although as we will see in [Chapter 12](ch12.html#ch12), RAG refers to an even broader concept. RAG is an important paradigm that deserves its own chapter, so we will defer detailed coverage of the paradigm to [Chapter 12](ch12.html#ch12).

Note that the distinguishing feature of this paradigm is the passive nature of the LLM in the interaction. The LLM simply responds to the prompt and furnishes an answer. It does not know the source of the content inside the prompt. This paradigm is often used for building QA assistants or chatbots, where external information is required to understand the context of the conversation.

###### Note

From this point forward, we will refer to user requests to the LLM as *queries* and textual units that are retrieved from external data stores as *documents*. Documents can be full documents, passages, paragraphs, or sentences.

## The Explicit Approach

[Figure 10-2](#explicit-approach) demonstrates the explicit approach to interface LLMs with external tools.

![Explicit Approach](assets/dllm_1002.png)

###### Figure 10-2\. The explicit interaction approach in action

Unlike in the passive approach, the LLM is no longer a passive participant. We provide the LLM with explicit instructions on how and when to invoke external data stores and tools. The LLM interacts with its environment based on a pre-programmed set of conditions. This approach is recommended when the interaction sequence is fixed, limited in scope, and preferably involves a very small number of steps.

For an AI data analyst assistant, an example interaction sequence could be:

1.  User expresses query in natural language asking to visualize some data trends

2.  The LLM generates SQL to retrieve the data needed to resolve the user query

3.  After receiving the data, the LLM uses it to generate code that can be run by a code interpreter to generate statistics or visualizations

[Figure 10-3](#ai-data-analyst) shows a fixed interaction sequence implemented for an AI data analyst.

![ai-data-analyst](assets/dllm_1003.png)

###### Figure 10-3\. An example workflow for an AI data analyst

In this paradigm, the interaction sequence is predetermined and rule-based. The LLM exercises no agency in determining which step to take next. I recommend this approach for building robust applications that have stricter reliability requirements.

## The Autonomous Approach

[Figure 10-4](#agentic-approach) shows how we can turn an LLM into an autonomous agent that can solve complex tasks by itself.

![Agentic Approach](assets/dllm_1004.png)

###### Figure 10-4\. A typical autonomous LLM-driven agent workflow

The autonomous approach, or the Holy Grail approach as I like to call it, turns an LLM into an autonomous agent that can solve tasks on its own by interacting with its environment. Here is a typical workflow of an autonomous agent:

1.  The user formulates their requirements in natural language, optionally providing the format in which they want the LLM to provide the answer.

2.  The LLM decomposes the user query into manageable subtasks.

3.  The LLM synchronously or asynchronously solves each subtask of the problem. Where possible, the LLM uses its own memory and knowledge to solve a specific subtask. For subtasks where the LLM cannot answer on its own, it chooses a tool to invoke from a list of available tools. Where possible, the LLM uses the outputs from solutions of already executed subtasks as inputs to other subtasks.

4.  The LLM synthesizes the final answer using the solutions of the subtasks, generating the output in the requested output format.

This paradigm is general enough to capture just about any use case. It is also a risky paradigm, as we are assigning the LLM too much responsibility and agency. At this juncture, I would not recommend using this paradigm for any mission-critical applications.

###### Note

Why am I calling for caution in deploying agents? Humans often underestimate the accuracy requirements for applications. For a lot of use cases, getting it right 99% of the time is still not good enough, especially when the failures are unpredictable and the 1% of failures can be potentially catastrophic. The 99% problem is also the one that has long plagued self-driving cars and prevented their broader adoption. This doesn’t mean we can’t deploy autonomous LLM agents; we just need clever product design that can shield the user from their failures. We also need robust human-in-the-loop paradigms.

We have used the word “agent” several times now without defining it. Let’s correct that and consider what agents mean and how we can build them.

# Defining Agents

As the hype starts building over LLM-based agents, the colloquial definition of agents has already started to expand from its traditional definition. This is because truly agentic systems are hard to build, so there is a tendency to shift the goalposts and claim best-effort systems to be already agentic even though they technically may not fit the requirements. In this book, we will stick to a more conservative definition of agents, defining them as:

> LLM-driven software systems that are able to interact with their environment and take autonomous actions to complete a task.

Key characteristics of agents are:

Their autonomous nature

The sequence of steps required to perform a task need not be specified to the agent. Agents can decide to perform any sequence of actions, unprompted by humans.

Their ability to interact with their environment

Agents can be connected to external data sources and software tools, which allows agents to retrieve data, invoke tools, execute code, and provide instructions when appropriate to solve a task.

###### Note

Many definitions of “agent” do not require them to be autonomous. According to their definitions, applications following the explicit paradigm can also be called agents (albeit as non-autonomous or semi-autonomous agents).

The agentic paradigm as we defined it is extremely powerful and general. Let’s take a moment to appreciate it. If an agent receives a task that it doesn’t know how to solve (and it *knows* that it doesn’t know), then instead of just giving up, it can potentially learn to solve the task by itself by searching the web or knowledge bases for pointers, or even by collecting data and fine-tuning a model that can help solve the task.

Given these enviable abilities, are machines going to take over the world? In practice, current autonomous agents are limited in what they can actually achieve. They tend to get stuck in loops, they take incorrect actions, and they are unable to reliably self-correct. It is more practical to build partially autonomous agents, where the LLM is provided with guidance throughout its workflow, either through agent orchestration software or with a human in the loop. For the rest of this chapter, our focus will be on building practical agents that can reliably solve a narrower class of tasks.

# Agentic Workflow

Using our definition of agents, let’s explore how agents work in practice. As an example, let’s consider an agent that is asked to answer this question:

> Who was the CFO of Apple when its stock price was at its lowest point in the last 10 years?

Let’s say the agent has all the information it needs to solve this task. It has access to the web, to SQL databases containing stock price information, and to knowledge bases containing CFO tenure information. It is connected to a code interpreter so that it can generate and run code, and it has access to financial APIs. The system prompt contains details about all the tools and data stores the LLM has access to.

To answer the given query, the LLM has to perform this sequence of steps:

1.  To calculate the date range, it needs the current date. If this is not included in the system prompt, it either searches the web to find the current date or generates code for returning the system time, which is then executed by a code interpreter.

2.  Using the current date, it finds the other end of the date range by executing a simple arithmetic operation by itself, or by generating code for it. Steps 1 and 2 could be combined into a single program.

3.  It finds a database table in the available datastore list that contains stock price information. It retrieves the schema of the table, inserts it into the prompt, and generates a SQL query for finding the date when the stock price was at its minimum in the last 10 years.

4.  With the date in hand, it needs to find the CFO of Apple on that date. It can call a search engine API to check if there is an explicit mention of the CFO on that particular date.

5.  If the search engine query fails to provide a result, it finds a financial API in its tools list and retrieves and inserts the API documentation into its context. It then generates and invokes code for an API call to retrieve the list of Apple CFOs and their tenures.

6.  It uses its arithmetic reasoning skills to find the CFO tenure that matches the date of the lowest stock price.

7.  It generates the final answer. If there is a requested output format, it tries to adhere to that.

Depending on the implementation, the sequence of steps could vary slightly. For example, you can fine-tune a model so that it can generate code for API calls or SQL queries directly without having to retrieve the schema from a data store or API.

To perform the given sequence of tasks, the model should first understand that the given task needs to be decomposed into a series of subtasks. This is called task decomposition. Task decomposition and planning can be performed by the LLM or offloaded to an external tool.

# Components of an Agentic System

While the specific architecture of any given agentic system depends heavily on the use cases it is intended to support, each of its components can be classified into one of the following types:

*   Models

*   Tools

*   Data stores

*   Agent loop prompt

*   Guardrails and verifiers

*   Orchestration software

[Figure 10-5](#agentic-system) shows a canonical agentic system and how its components interact.

![agentic-system](assets/dllm_1005.png)

###### Figure 10-5\. A production-grade agentic system

Let’s explore each of these types.

## Models

Language models are the backbone of agentic systems, responsible for their autonomous nature and problem-solving capabilities. A single agentic system could be composed of multiple language models, with each model playing a distinct role.

For example, you can build an agent consisting of two models; one model solves user tasks and another model takes its output and converts it into a structured form according to user requirements.

###### Tip

Agentic workflows can consume a lot of language model tokens, which can be cost prohibitive. To keep costs under control, consider using multiple language models of different sizes, with the smaller (and cheaper) models performing easier tasks. For more details on how to accomplish division of labor among these models, see [Chapter 13](ch13.html#ch13).

More generally, you can build agents with specialized models catering to each part of the agentic workflow. For example, a code-LLM can be used to generate code, and task-specific fine-tuned models that specialize in individual workflow steps can be used. This setup can be interpreted as a *multi-agent architecture*.

[Figure 10-6](#multi-agent-setup) shows an agentic system made up of multiple LLMs.

![multi-agent-setup](assets/dllm_1006.png)

###### Figure 10-6\. An agentic system with multiple LLMs

Finally, any kind of model, including non-LLMs, can be plugged into an agentic system to solve specific tasks. For example, the planning stage can be performed using [symbolic planners](https://oreil.ly/sXPWG).

## Tools

As described earlier, software or models that can be invoked by an LLM are called tools. Libraries like [LangChain](https://oreil.ly/35Lgu) and [LlamaIndex](https://oreil.ly/WF-d1) provide connectors to various software interfaces, including code interpreters, search engines, databases, ML models, and a variety of APIs. Let’s explore how to work with some of these in practice.

### Web search

LangChain provides connectors for major search engines like Google, Bing, and DuckDuckGo. Let’s try out DuckDuckGo:

```py
from langchain_community.tools import DuckDuckGoSearchRun

query = "What's the weather today in Toronto?"

search_engine = DuckDuckGoSearchRun()
output = search_engine.run(query)
```

The response can be fed back to the language model where it is further processed.

### API connectors

To illustrate calling APIs, we will showcase LangChain’s Wikipedia API wrapper:

```py
!pip install wikipedia

from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

output = wikipedia.load("Winter Olympics")
```

The `load()` function runs a search on Wikipedia and returns the page text and metadata information of the top-k results. (top-k = 3 by default). You can also use the `run()` function to return only page summaries of the top-k matches.

### Code interpreter

Next, let’s explore how you can invoke a code interpreter and run arbitrary code:

```py
from langchain_experimental.utilities import PythonREPL

python = PythonREPL()
python.run("456 * 345")
```

###### Warning

Be wary of running code generated by LLMs in response to user prompts. Users can induce the model to generate malicious code!

### Database connectors

Finally, let’s check out how to connect to a database and run queries:

```py
import sqlalchemy as sa
from langchain_community.utilities import SQLDatabase

DATABASE_URI = <database_uri>

db = SQLDatabase.from_uri(DATABASE_URI)

output = db.run(
    "SELECT * FROM COMPANIES WHERE Name LIKE :comp;",
    parameters={"comp": "Apple%"},
    fetch="all")
```

The `run()` function executes the provided SQL query and returns the response as a string. Replace `*DATABASE_URI*` with your own database and queries, and verify the responses.

###### Tip

For more customizability, you can fork the LangChain connectors and repurpose them for your own use.

Next, let’s see how we can interface LLMs with these tools in an agentic workflow.

First, we need to make the LLM aware that it has access to these tools. One of the ways to achieve this is to provide the names and short descriptions of the tools, called the *tool list*, to the LLM through the system prompt.

Next, the LLM needs to be able to select the right tool at the appropriate juncture in the workflow. For example, if the next step in solving a task is to find the weather in Chicago this evening, the web search tool has to be invoked rather than the Wikipedia one. Later in this chapter, we will discuss techniques to help the LLM select the right tool.

Under the hood, tool invocation is typically achieved by the LLM generating special tokens indicating that it is entering tool invocation mode, along with tokens representing the tool functions and arguments to be invoked. The actual tool invocation is performed by an agent orchestration framework.

In LangChain, we can make a tool available to an LLM and have it invoked:

```py
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

search_engine = DuckDuckGoSearchRun()
model = ChatOpenAI(model="gpt-4o")

tools = [
       Tool(
           name="Search",
           func=search_engine.run,
           description="search engine for answer factual queries"
       )
   ]
agent = initialize_agent(tools, model, verbose=True)
agent.run("What are some tourist destinations in North Germany?")
```

Some models come with native tool-calling abilities. For models that don’t, you can fine-tune the base model to impart them with tool-calling abilities. Among open models, Llama 3.1 Instruct (8B/70B/405B) is an example of a model having native tool-calling support. Here’s how tool calling works with Llama 3.1.

Llama 3.1 comes with native support for three tools: Brave web search, Wolfram|Alpha mathematical engine, and a code interpreter. These can be *activated* by defining them in the system prompt:

```py
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Environment: ipython
Tools: brave_search, wolfram_alpha

Give responses to answers in a concise fashion. <|eot_id|>
```

Let’s ask the LLM a question by appending a user prompt to the system prompt:

```py
<|start_header_id|>user<|end_header_id|>

How many medals did Azerbaijan win in the 2024 Summer Olympics?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

Llama 3.1 responds with a tool invocation that looks like this:

```py
<|python_tag|>brave_search.call(query="How many medals did Azerbaijan win in

the 2024 Summer Olympics?")<|eom_id|>
```

The `<|python_tag|>` token is a special token generated by Llama 3.1 to indicate that it is entering tool-calling mode. The `<|eom_id|>` special token indicates that the model has not ended its turn yet and will wait to be fed with the results of the tool invocation.

You can also provide your own tools in the prompt: using JSON is recommended.

###### Tip

If you have a lot of tools, then the detailed descriptions of the tools can be represented in a data store and retrieved only if they are selected. The prompt then needs to contain only the name of the tool and a short description.

Here is an example of a tool definition in JSON describing a local function that can be called:

```py
<|start_header_id|>user<|end_header_id|>

Here is a list of tools available.
While invoking a tool, respond in JSON. The format is as follows:

{"tool_name": tool name, "arguments": dictionary with keys representing

argument names and values representing argument values}.

{
    "type": "local_function",
    "function": {
    "name": "find_citations",
    "description": "Find the citations for any claims made",
    "parameters": {
        "type": "object",
        "properties": {
        "claim_sentence": {
            "type": "string",
            "description": "A sentence in the input representing a claim"
        },
        "model": {
            "type": "string",
            "enum": ["weak", "strong"],
            "description": "The type of citation model to use. A weak model is
            preferred if the claim sentence contains entities and numbers. "
        }
        },
        "required": ["claim_sentence", "model"]
    }
    }
}
```

The tool call is generated by the model in JSON with the prescribed format.

###### Note

The actual tool invocation is performed by an agent orchestration software. Llama 3.1 comes with [llama-stack-apps](https://oreil.ly/SSmkI), a library that facilitates agentic workflows.

Sometimes the tool call can be more complex than just returning the name of a function and its arguments. An example of this is querying a database. For the LLM to generate the right SQL query, you should provide the schema of the database tables in the system prompt. If the database has too many tables, then their schema can be retrieved on demand by the LLM.

###### Tip

You can use a separate specialized model for code and SQL query generation. A general-purpose model can generate a textual description of the desired outcome, and this can be used as input to a code LLM or an LLM fine-tuned on text-to-SQL.

For large-scale or high-stakes applications, you can fine-tune your models to make them better at tool use. A good fine-tuning recipe to follow is Qin et al.’s [ToolLLaMA](https://oreil.ly/Ewlxt).

## Data Stores

A typical agent may need to interact with several types of data sources to accomplish its tasks. Commonly used data sources include prompt repositories, session memory, and tools data.

### Prompt repository

A prompt repository is a collection of detailed prompts instructing the language model how to perform a specific task. If you can anticipate the types of tasks that an agent will be asked to perform while in production, you can construct prompts providing detailed instructions on how to solve them. The prompts can even include directions on how to advance a specific workflow. Let’s look at an example.

Many language models struggle with basic arithmetic operations, even simple questions like:

```py
Is 9.11 greater than 9.9?
```

Until recently, even state-of-the-art language models claimed that 9.11 is greater than 9.9\. (They were recently updated with a fix after this limitation went viral on [social media](https://oreil.ly/ztWGW).)

If you are aware of such limitations that are relevant to your use case, then you can mitigate a proportion of them using detailed prompts. For the number comparison issue, for example:

> *Prompt:* If you are asked to compare two numbers using the greater than/lesser than operation, then perform the following:
> 
> Take the two numbers and ensure they have the same number of decimal places. After that, subtract one from the other. If the result is a positive number, then the first number is greater. If the result is a negative number, then the second number is greater. If the result is zero, the two numbers are equal.

Now, if the agent needs to perform a task that includes number comparison, it first retrieves this prompt from the prompt repository. This enables it to overcome its inherent limitation, as it will follow the detailed step-by-step instructions in the prompt.

###### Note

Why don’t we just add all these prompts to the context window, thus eschewing retrieval? For one, the prompts may be too numerous and may not fit within the context window. Secondly, tokens are expensive, and it is inefficient to include prompts that may not be relevant to the current task. Finally, language models can adhere to only a limited set of concurrent instructions, so it is more efficient to retrieve them on demand.

Prompts can also include input-output examples, known as few-shot learning, as introduced in [Chapter 1](ch01.html#chapter_llm-introduction). Agents can retrieve them on demand to help accomplish their tasks. We will discuss effective ways of retrieving relevant examples in [Chapter 12](ch12.html#ch12).

Next, let’s explore how agents use session memory to advance their goals.

### Session memory

We would like to store logs of the steps undertaken by the agent during current and past sessions. These logs can also be augmented with feedback from verification processes, both automated and human. The agent can then retrieve session logs for better context understanding and as a guide to determine next actions in the current workflow. Let’s explore this with an example:

> *Prompt:* Is it going to be warmer tomorrow than today in Paris? Give me details in Celsius.

The agent goes through a sequence of steps to solve this query, which is all logged in the session history. The session history for this query might look like this:

> *User prompt*: Is it going to be warmer tomorrow than today in Paris? Give me details in Celsius.
> 
> *LLM observation*: The user has a preference for Celsius units for temperature data.
> 
> *LLM Output*: Let’s perform this step-by-step.
> 
> 1.  First, let’s retrieve today’s temperature.
>     
>     
> 2.  Next, let’s retrieve tomorrow’s temperature.
>     
>     
> 3.  We calculate the difference and generate the answer.
>     
>     
> 
> *External verifier*: Task decomposition is correct
> 
> *Tool selection:* <WeatherAPI> selected
> 
> *Tool info retrieval*: Retrieve information relevant to the query *Find temperature data from <WeatherAPI>*
> 
> *Tool invocation*: get_temperature_curweek(*Paris*)
> 
> *Tool output* : Weather data - Paris…​
> 
> *LLM output*: Today’s temperature is 27 degrees Celsius. Tomorrow’s forecast is 23 degrees Celsius. Therefore, tomorrow is not going to be warmer in Paris than today.
> 
> *External Verifier*: Arithmetic operation is correct.
> 
> *Agent*: LLM output is dispatched to the user
> 
> *User feedback*: User marked this as correct

As we can see, session history can contain very rich information that can provide valuable personalized context to the LLM about the current user as well as guide the model toward the correct agentic workflow.

In more advanced implementations, multiple levels of logging can be defined, so that during retrieval, one can retrieve all the logs of a session or only the important steps, based on the logging level specified.

###### Tip

Along with session history, the agent could also be provided with access to gold-truth training examples representing correct workflows, which can be used by the agent to guide its trajectory during test time.

Session memory can also include records of interaction between the human and the agentic system. These can be used to personalize models. We will discuss this further in [Chapter 12](ch12.html#ch12).

Next, let’s explore how the agent can interact with tools data.

### Tools data

Tools data comprise detailed information necessary to invoke a tool, such as database schemas, API documentation, sample API calls, and more. When the agent decides to invoke a tool, the model retrieves the pertinent tool information from the tools data store.

For example, consider a SQL tool for retrieving data from a database. To generate the right SQL query, the model could retrieve the database schema from the tools data store. The tools data contains information about the tables and columns, the descriptions of each column and their data types, and optionally information about indices and primary/secondary keys.

###### Note

You can also fine-tune the LLM on a dataset representing valid SQL queries to your database, which can potentially remove the need to consult the schema before generating a query.

To sum it up, agents can use data stores in several ways. They can access prompts and few-shot examples from a prompt repository, they can access agentic workflow history and intermediate outputs by models in previous sessions for better personalized context understanding and workflow guidance, and they can access tool documentation to invoke tools correctly.

Agents can also access external knowledge from the web, databases, knowledge graphs, etc. Retrieving the right information from these sources is an entire sub-system unto itself. We will discuss the mechanics of retrieval in Chapters [11](ch11.html#chapter_llm_interfaces) and [12](ch12.html#ch12).

We will now discuss the agent loop prompt, which is responsible for driving the LLM’s behavior during an agentic session.

## Agent Loop Prompt

Recall that LLMs do not have session memory. But a typical agentic workflow relies on several LLM calls! We need a mechanism to provide information about session state and the expected role of the LLM at any given time in the session. This agent loop is driven by a system prompt.

An example of a simple agent loop system prompt is:

> *Prompt:* You are an AI model currently answering questions. You have access to the following tools: {tool_description}. For each question, you can invoke one or more tools where necessary to access information or execute actions. You can invoke a tool in this format: <TOOLNAME> <Tool Arguments>. The results of these tool calls are not provided to the user. When you are ready with the final answer, output the answer using the <Answer> tag.

I find that a prompt like this is sufficient for most use cases. However, if you feel like the model is not reasoning correctly, you can try ReAct prompting.

### ReAct

At the time of this writing, ReAct (Reasoning + Acting) prompting is the most popular prompt for the agent loop. A typical ReAct prompt looks like this:

> *Prompt:* You are an AI assistant capable of reasoning and acting. For each question, follow this process:
> 
> 1.  Thought: Reflect on the current state and plan your next steps.
>     
>     
> 2.  Action: Execute the steps to gather information or call tools.
>     
>     
> 3.  Observation: Record the results of your actions.
>     
>     
> 4.  Final Answer: If you have an answer, provide a final response. Else continue the Thought → Action → Observation → loop until you have an answer.

Despite its popularity, ReAct prompting has been shown to be [brittle](https://oreil.ly/RRZO9).

### Reflection

The agent loop may include self-verification or correction steps. This was pioneered by [Shinn et al.](https://oreil.ly/xFVt0) with the Reflexion paradigm.

Here is the system prompt for [Reflection-Llama-3.1](https://oreil.ly/foB-P) that uses reflection techniques:

> *Prompt:* You are a world-class AI system, capable of complex reasoning and reflection. Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags. If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags.

The <reflection> tags are meant for the model to self-introspect and self-correct. We can also specify conditions when <reflection> tags should be activated, for example, when the agent performs the same action consecutively more than three times (which might mean it is stuck in a loop).

###### Warning

The effectiveness of reflection-based methods are overstated. They might do more harm than good if they are invoked too often, causing the model to second-guess solutions.

Next, let’s discuss guardrails and verifiers, components that ensure that an agentic system can thrive in production.

## Guardrails and Verifiers

In production environments, mistakes can be catastrophic. Depending on the use case, the agent might need to adhere to strict standards in factuality, safety, accuracy, and many other criteria.

Safety is ensured by using guardrails, components that ensure models do not overstep their bounds during the course of their workflows. Some examples of guardrails include toxic language detectors, personally identifiable information (PII) detectors, input filters that restrict the type of queries users are permitted to make, and more.

Verifiers ensure that quality standards of the agentic system are so that the agent is able to recover and self-correct from mistakes. As agentic systems are still in their infancy, the importance of good and well-placed verifiers is paramount. Verifiers can be as simple as token-matching tools but can also be fine-tuned models, symbolic verifiers, and so on.

Let’s learn more about guardrails and verifiers.

### Safety Guardrails

Recall from [Chapter 2](ch02.html#ch02) that LLMs are trained largely on human-generated web text. Unfortunately a significant proportion of human-generated text contains toxic, abusive, violent, or pornographic content. We do not want our LLM applications to generate content that violates the safety of the user, nor do we want users to misuse the model to generate unsafe content. While we can certainly use techniques like alignment training to make the model less likely to emit harmful content, we cannot guarantee 100% success and therefore need to institute inference-time guardrails to ensure safe usage. Libraries like [Guardrails](https://oreil.ly/F7yax) and NVIDIA’s [NeMo-Guardrails](https://oreil.ly/p7Dqz), and models like [Llama Guard](https://oreil.ly/8S08P) facilitate setting up these guardrails.

The Guardrails library provides a large (and growing) number of data validators to ensure safety and validity of LLM inputs and outputs. Here are some important ones:

Detect PII

This validator can be used to detect personally identifiable information in both the input and output text. [Microsoft Presidio](https://oreil.ly/eG8T1) is employed under the hood to perform the PII identification.

Prompt injection

This validator can detect certain types of adversarial prompting and thus can be used to prevent users from misusing the LLM. The [Rebuff](https://oreil.ly/nIyE5) library is used under the hood to detect prompt injection.

Not safe for work (NSFW) text

This validator detects NSFW text in the LLM output. This includes text with profanity, violence, and sexual content. The *Profanity free* validator also exists for detecting only profanity in text.

Politeness check

This validator checks if the LLM output text is sufficiently polite. A related validator is *Toxic language*.

Web sanitization

This validator checks the LLM output for any security vulnerabilities, including if it contains code that can be executed in a browser. The [Bleach](https://oreil.ly/r3Xrl) library is used under the hood to find potential vulnerabilities and sanitize the output.

What happens if the validation checks fail and there is indeed harmful content in the input or output? Guardrails provides a few options:

Re-ask

In this method, the LLM is asked to regenerate the output, with the prompt containing instructions to specifically abide by the criteria on which the output previously failed validation.

Fix

In this method, the library fixes the output by itself without asking the LLM for a regeneration. Fixes can involve deletion or replacement of certain parts of the input or output.

Filter

If structured data generation is used, this option enables filtering out only the attribute for which the validation failed. The rest of the output will be fed back to the user.

Refrain

In this setting, the output is simply not returned to the user, and the user receives a refusal.

Noop

No action is taken, but the validation failure is logged for further inspection.

Exception

This raises a software exception when the validation fails. Exception handlers can be written to activate custom behavior.

fix_reask

In this method, the library tries to fix the output by itself and then runs validation on the new output. If the validation still fails, then the LLM is asked to regenerate the output.

Let’s look at the PII guardrail as an example:

```py
from guardrails import Guard
from guardrails.hub import DetectPII

guard = Guard().use(
    DetectPII, ["EMAIL_ADDRESS", "PHONE_NUMBER"], "reask")

guard.validate("The Nobel prize this year was won by Geoff Hinton,
who can be reached at +1 234 567 8900")
```

Next, let’s look at how verification modules work.

### Verification modules

As we have seen throughout the book, current LLMs suffer from problems like reasoning limitations and hallucinations that severely limit their robustness. However, production-ready applications need to demonstrate a certain level of reliability to be accepted by users. One way to extend the reliability of LLM-based systems is to use a human-in-the-loop who can manually verify the output and provide feedback. However, in the real world a human-in-the-loop is not always desired or feasible. The most popular alternative is to use external verification modules as part of the LLM system. These modules can range from rule-based programs to smaller fine-tuned LLMs to symbolic solvers. There are also efforts to use LLMs as verifiers, called “LLM-as-a-judge.”

Related components include fallback modules. These modules are activated when the verification process fails and retrying/fixing doesn’t work. Fallback modules can be as simple as messages like, “I am sorry I cannot entertain your request” to more complex workflows.

Let’s discuss an example. Consider an abstractive summarization application that operates on financial documents. To ensure quality and reliability of the generated summaries, we need to embed verification and self-fixing into the system architecture.

How do we verify the quality of an abstractive summary? While single-number metrics are available to automatically quantify the quality of a summary, a more holistic approach would be to define a list of criteria that a good summary should satisfy and verify whether each criterion is fulfilled.

###### Note

Several single-number quantitative metrics exist for evaluating summaries. These include metrics like [BLEU, ROUGE](https://oreil.ly/LPlFJ), and [BERTScore](https://oreil.ly/gsOGl). BLEU and ROUGE rely on token overlap heuristics and have been shown to be [woefully inadequate](https://oreil.ly/rSzbR). Techniques like BERTScore that apply semantic similarity have been shown to be more promising, but in the end, the reality is that summaries have subjective notions of quality and need a more holistic approach for verification.

For the summarization of financial documents application, here is a list of important criteria:

Factuality

The summary is factually correct and does not make incorrect assumptions or conclusions from the source text.

Specificity

The summary doesn’t *oversummarize*; it avoids being generic and provides specific details, whether numbers or named entities.

Relevance

Also called precision, this is calculated as the percentage of sentences in the summary that are deemed relevant and thus merit inclusion in the summary.

Completeness

Also called recall, this is calculated as the percentage of relevant items in the source document that are included in the summary.

Repetitiveness

The summary should not be repetitive, even if there is repetition in the source document.

Coherence

When read in full, the summary should provide a clear picture of the content in the source document, while minimizing ambiguity. This is one of the list’s more subjective criteria.

Structure

While defining the summarization task, we might specify a structure for the summaries. For example, the summary could be expected to contain some predefined sections and subsections. The generated summary should follow the specified structure.

Formatting

The generated summary should follow proper formatting. For example, if the summary is to be generated as a bulleted list, then all the items in the summary should be represented by bullets.

Ordering

The ordering of the items in the summary should not impede the understanding of the summary content. We also might want to specify an order for the summaries, for example, chronological.

Error handling

In case of errors or omissions in the source document, there should be appropriate error handling.

How do we automatically verify whether a given summary meets all these criteria? We can use a combination of rule-based methods and fine-tuned models. Ultimately, the rigor of the methods used for verification depends on the degree of reliability needed for your application. However, we notice that once we reduce the scope of the verification process to verify fitness of individual criteria rather than the application as a whole, it becomes easier to verify accurately using inexpensive techniques. Let’s look at how we can build verifiers for each criteria of the abstractive summarization task:

Factuality

Verifying whether an LLM-generated statement is factual is extremely difficult if we do not have access to ground truth. But for summarization applications, we do have access to the ground truth. Therefore, we can verify factuality by taking each sentence in the summary and checking whether, given the source text, one can logically conclude the statement in the summary. This can be framed as a natural language inference (NLI) problem, which is a standard NLP task.

In the NLI task, we have a hypothesis and a premise, and the goal is to check if the hypothesis is logically entailed by the premise. In our example, the hypothesis is a sentence in the summary and the premise is the source text.

Training an NLI model specific to your domain might be a cumbersome task. If you do not have access to an NLI model, you can use token overlap and similar statistics to approximate factuality verification.

For numbers and named entities, factuality verification can be performed by using string matches. You can verify if all the numbers and named entities in the summary are indeed present in the source text.

Specificity

One way for a summary to be specific is to include numbers and named entities where relevant. For each sentence in the summary, we can check whether the content in the source document related to the topic of the sentence contains any numbers and named entities, and if these are reflected in the summary. Numbers and named entities can be tagged and detected using regular expressions or libraries like [spaCy](https://oreil.ly/zatAW).

Relevance/precision

We can train a classification model that detects whether a sentence in the summary is relevant. Note that there are limits to this approach. If this classification model was good enough, we could have directly used it to select relevant sentences from the source text to build the summary! In practice, this classification model can be used to remove irrelevant content that is more obvious.

Recall/completeness

What content merits inclusion in the summary is a difficult question, especially if there is a hard limit on the summary length. You can train a ranking model that ranks sentences in the source document by importance, and then verify if the top-ranked sentences are represented in the summary. You can also specify beforehand the type of content that you need represented in the summary and build a classification model for determining which parts of the source document contain pertinent information. Using similarity metrics like embedding similarity, you can then find if the content has been adequately represented in the summary.

Repetitiveness

This can be discovered by using string difference algorithms like the [Jaccard distance](https://oreil.ly/Ny_Ku) or by calculating the embedding similarity between pairs of summary sentences.

Coherence

This is perhaps one of the most difficult criteria to verify. One way to solve this, albeit a more expensive solution, is to build a prerequisite detection model. For each sentence in the summary, we detect if all the sentences that come before it are sufficient prerequisites for understanding the correct sentence. For more information on prerequisite detection techniques, see [Thareja et al.](https://oreil.ly/6JnRs)

Structure

If we specify a predetermined structure (sections and subsections) for the summary, we can easily identify if the structure is adhered to by checking if the desired section and subsection titles are present in the summary. We can also verify using embedding similarity techniques if the content within the sections and subsections is faithful to the title of the section/subsection.

Formatting

This involves checking whether the content is in the appropriate formatting, for example, whether it is a bulleted list or a valid JSON object.

Ordering

The desired order can be chronological, alphabetical, a domain, or task-specific ordering. If it is supposed to be chronological, you can verify by extracting dates in the summary and checking if the summary contains dates in a chronological order. If the ordering requirements are more complex, then verifying adherence to order may become an extremely difficult task.

###### Tip

Do not expect your verification process to be strictly better than your summary model. If that was the case, you could have used the verification process to generate the summary!

We can also deploy symbolic verifiers like [SAT](https://oreil.ly/lOsg_) (Boolean satisfiability) solvers and logic planners. This type of verification is beyond the scope of this book.

Once verification modules are part of our system architecture, we will also need to decide what action to perform when the verification fails. One option is to just resample from the language model again. Regeneration can be performed for the full output or only for the output that failed verification. We can also develop antifragile architectures that have fallbacks in case of failure, which we will discuss in [Chapter 13](ch13.html#ch13).

###### Warning

Adding more verifiers can drastically increase system latency. Thus, their inclusion has to be balanced with accuracy and system latency needs.

Finally, let’s discuss agent orchestration software that connects all these components.

## Agent Orchestration Software

For agentic workflows to proceed smoothly, we need software that connects all the components. Orchestration software manages state; invokes tools; initiates retrieval; pipes buffers; and logs intermediate and final outputs. Many agentic frameworks, both open source and proprietary, perform this function, including [LangChain](https://oreil.ly/7vmlY), [LlamaIndex](https://oreil.ly/uxejK), [CrewAI](https://oreil.ly/Ntxii), [AutoGen](https://oreil.ly/tx3qy), [MetaGPT](https://oreil.ly/HI-Jn), [XAgent](https://oreil.ly/sA_DR), [llama-stack-apps](https://oreil.ly/SBGC_), and so on.

###### Tip

Agents are a relatively new paradigm, so all these agentic frameworks are expected to change a lot in the coming months and years. These frameworks are implemented in an opinionated fashion and hence are less flexible. For prototyping, I suggest picking LangChain or LlamaIndex for ease of use. For production use, you might want to build a framework internally from scratch or by extending the open source ones. This book’s [GitHub repo](https://oreil.ly/llm-playbooks) contains a rudimentary agentic framework as well.

Now that we have learned all the different agentic system components, it is time to get building! The book’s [GitHub repository](https://oreil.ly/llm-playbooks) contains sample implementations of various types of agents. Try modifying them for your use case to understand the tradeoffs being made.

###### Tip

The keep it simple, stupid (KISS) principle applies to agents perhaps more than any other recent paradigm. Don’t complicate your agentic architecture unless there is a compelling reason to do so. We will discuss this more in [Chapter 13](ch13.html#ch13).

# Summary

In this chapter, we discussed the different ways in which LLMs can interface with external tools. We introduced the agentic paradigm and provided a formal definition of agents. We identified the components of an agentic system in detail, exploring models, tools, data stores, guardrails and verifiers, and agentic orchestration software. We learned how to define and implement our own tools.

In the next chapter, we will explore data representation and retrieval, crucial elements of interfacing LLMs with external data.