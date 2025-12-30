# Chapter 9\. Securing AI Services

In earlier chapters, you learned how to build GenAI services that serve various AI generators while supporting concurrency and data streaming in real time. Additionally, you integrated external systems like databases and implemented your own authentication and authorization mechanisms. Finally, you wrote a test suite to verify the functionality and performance of your entire system.

In this chapter, you’ll learn how to implement usage moderation and abuse-protection mechanisms to secure your GenAI services.

# Usage Moderation and Abuse Protection

When deploying your GenAI services, you’ll need to consider how your services will be misused and abused by malicious users. This is essential to protect user safety and your own reputation. You won’t know how the users will use your system, so you need to assume the worst and implement *guardrails* to protect against any misuse or abuse.

According to a [recent study on nefarious applications of GenAI](https://oreil.ly/ihmzR), your services may potentially be used with *malicious intents*, as described in [Table 9-1](#malicious_intents).

Table 9-1\. Malicious intents behind abusing GenAI services

| Intent | Examples | Real-world cases |
| --- | --- | --- |
| **Dishonesty**Supporting lies and untruthfulness | Plagiarism, faking competency and knowledge, document forgery, cheating in exams and in interviews, etc. | Increasing cases of students cheating with AI at UK and Australian universities^([a](ch09.html#id1032)) |
| **Propaganda**Skewing perceptions of reality to advance an agenda | Impersonating others, promoting extremism, influencing campaigns, etc. | Fake AI news anchors spreading misinformation or propaganda^([b](ch09.html#id1033)) |
| **Deception**Misleading others and creating false impressions | Generating fake reviews, scam ads and phishing emails, and synthetic profiles (i.e., sockpuppeting), etc. | Engineering firm Arup revealed as a victim of a $25 million deepfake scam^([c](ch09.html#id1034)) |
| ^([a](ch09.html#id1032-marker)) Sources: *Times Higher Education* and *The Guardian*^([b](ch09.html#id1033-marker)) Sources: *The Guardian*, *MIT Technology Review*, and *The Washington Post*^([c](ch09.html#id1034-marker)) Sources: CNN and *The Guardian* |

The same study categorizes GenAI application abuse into the following:

*   *Misinformation and disinformation* to spread propaganda and fake news

*   *Bias amplification and discrimination* to advance racist agendas and societal discrimination

*   *Malicious content generation* by creating toxic, deceptive, and radicalizing content

*   *Data privacy attacks* to fill in gaps in stolen private data and leak sensitive information

*   *Automated cyberattacks* to personalize phishing and ransomware attacks

*   *Identity theft and social engineering* to increase the success rate of scams

*   *Deepfakes and multimedia manipulation* to make a profit and skew perceptions of reality and social beliefs

*   *Scam and fraud* by manipulating stock markets and crafting targeted scams

This may not be an exhaustive list but should give you a few ideas on what usage moderation measures to consider.

[Another study on the taxonomy of GenAI misuse tactics](https://oreil.ly/jbG01) investigated abuse by modality and found that:

*   *Audio and video generators* were used for the majority of impersonation attempts.

*   *Image and text generators* were used for the majority of sockpuppeting, content farming for opinion manipulation at scale, and falsification attempts.

*   *Image and video generators* were used for the majority of steganography, (i.e., hiding coded messages in model outputs), and nonconsensual intimate content (NCII) generation attempts.

If you’re building services supporting such modalities, you should consider their associated forms of abuse and implement relevant protection mechanisms.

Aside from misuse and abuse, you’ll also need to consider security vulnerabilities.

Securing GenAI services is still an area of research at the time of writing. For instance, if your services leverage LLMs, OWASP has categorized the [top 10 LLM vulnerabilities](https://oreil.ly/4zob2), as shown in [Table 9-2](#llm_vulnerabilities).

Table 9-2\. OWASP top 10 LLM vulnerabilities

| Risk | Description |
| --- | --- |
| Prompt injection | Manipulating inputs to control the LLM’s responses leading to unauthorized access, data breaches, and compromised decision-making. |
| Insecure output handling | Failing to sanitize or validate LLM outputs causing remote code execution on downstream systems. |
| Training data poisoning | Injecting data in sources that models get trained on to compromise security, accuracy, or ethical behavior. Open source models and RAG services that rely on web data are most prone to these attacks. |
| Model denial of service | Causing service disruption and cost explosions by overloading the LLMs with heavy payloads and concurrent requests. |
| Supply chain vulnerabilities | Causing various components, including data sources, to be compromised, undermining system integrity. |
| Sensitive information leakage | Leading to accidental exposure of private data, legal liabilities and loss of competitive advantage. |
| Insecure plug-in design | Vulnerabilities in third-party integrations cause remote code execution. |
| Excessive agency | Where LLMs have too much autonomy to take actions can lead to unintended consequences and harmful actions. |
| Overreliance on LLM | Compromising decision-making, contributing to security vulnerabilities and legal liabilities. |
| Model theft | Related to unauthorized copying or usage of your models. |

###### Tip

Similar vulnerabilities exist for other types of GenAI systems such as image, audio, video, and geometry generators.

I recommend researching and identifying software vulnerabilities relevant to your own use cases.

Without guardrails, your services can be abused to cause personal and financial harm, identity theft, economic damage, spread misinformation, and contribute to societal problems. As a result, it’s crucial to implement several safety measures and guardrails to protect your services against such attacks.

In the next section, you’ll learn usage moderation and security measures you can implement to protect your GenAI services prior to deployment.

# Guardrails

*Guardrails* refer to *detective controls* that aim to guide your application toward the intended outcomes. They are incredibly diverse and can be configured to fit any situation that may go wrong with your GenAI systems.

As an example, *I/O guardrails* are designed to verify data entering a GenAI model and outputs sent to the downstream systems or users. Such guardrails can flag inappropriate user queries and validate output content against toxicity, hallucinations, or banned topics. [Figure 9-1](#guardrails) shows how an LLM system looks once you add I/O guardrails to it.

![bgai 0901](assets/bgai_0901.png)

###### Figure 9-1\. Comparison of an LLM system without and with guardrails

You don’t have to implement guardrails from scratch. At the time of writing, prebuilt open source guardrail frameworks exist like NVIDIA NeMo Guardrails, LLM-Guard, and Guardrails AI to protect your services. However, they may require learning framework-related languages and have a trade-off of slowing down your services and bloating your application due to various external dependencies.

Other commercial guardrails available on the market, such as Open AI’s Moderation API, Microsoft Azure AI Content Safety API, and Google’s Guardrails API are either not open source or lack details and contents to measure quality constraints.

###### Warning

Guardrails remain an active area of research. While such defenses can counter some attacks, powerful attacks backed by AI can still bypass them. This may lead to an [ongoing and endless loop of assaults and defenses](https://oreil.ly/xlUmw).

While engineering application-level I/O guardrails may not provide perfect protection, upcoming GenAI models may include baked-in guardrails inside the model to improve security guarantees. However, such guardrails may have a performance impact on response times by introducing latency to the system.

## Input Guardrails

The purpose of input guardrails is to prevent malicious or inappropriate content from reaching your model. [Table 9-3](#guardrails_input) shows common input guardrails.

Table 9-3\. Common input guardrails

| Input guardrails | Examples |
| --- | --- |
| **Topical**Steer inputs away from off-topic or sensitive content. | Preventing a user from discussing political topics and explicit content. |
| **Direct prompt injection** (jail-breaking)Prevent users from revealing or overriding system prompts and secrets. The longer the input content, the more prone your system will be to these attacks. | Blocking attempts to override system prompts and manipulating the system into revealing internal API keys or configuration settings.^([a](ch09.html#id1036)) |
| **Indirect prompt injection**Prevent acceptance of malicious content from external sources such as files or websites that may cause model confusion or remote code execution on downstream systems.Malicious content may be invisible to the human eye and encoded within input text or images. | Sanitizing encoded payloads in upload images, hidden characters or prompt overrides in uploaded documents, hidden scripts in remote URLs or even YouTube video transcripts. |
| **Moderation**Comply with brand guidelines, legal, and branding requirements. | Flag and refuse invalid user queries if user queries include mentions of profanity, competitor, explicit content, personally identifiable information (PII), self-harm, etc. |
| **Attribute**Validate input properties. | Check query length, file size, choices, range, data format and structure, etc. |
| ^([a](ch09.html#id1036-marker)) Although guardrails are useful, best practice is to avoid giving your GenAI models direct knowledge of secrets or sensitive configuration settings in the first place. |

The input guardrails can also be combined with content sanitizers to clean bad inputs.

If you want to implement your own guardrails, you can start off with using advanced prompt engineering techniques within your system prompts. Additionally, you can use auto-evaluation techniques, (i.e., AI models).

[Example 9-1](#guardrail_topical_prompt) shows an example system prompt for an AI guardrail auto-evaluator to reject off-topic queries.

##### Example 9-1\. Topical input guardrail system prompt

```py
guardrail_system_prompt = """

Your role is to assess user queries as valid or invalid

Allowed topics include:

1\. API Development
2\. FastAPI
3\. Building Generative AI systems

If a topic is allowed, say 'allowed' otherwise say 'disallowed'
"""
```

You can see an implementation of an input topical guardrail in [Example 9-2](#guardrail_topical) using the LLM auto-evaluation technique.

##### Example 9-2\. Topical input guardrail

```py
import re
from typing import Annotated
from openai import AsyncOpenAI
from pydantic import AfterValidator, BaseModel, validate_call

guardrail_system_prompt = "..."

class LLMClient:
    def __init__(self, system_prompt: str):
        self.client = AsyncOpenAI()
        self.system_prompt = system_prompt

    async def invoke(self, user_query: str) -> str | None:
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

@validate_call
def check_classification_response(value: str | None) -> str: ![1](assets/1.png)
    if value is None or not re.match(r"^(allowed|disallowed)$", value):
        raise ValueError("Invalid topical guardrail response received")
    return value

ClassificationResponse = Annotated[
    str | None, AfterValidator(check_classification_response)
]

class TopicalGuardResponse(BaseModel):
    classification: ClassificationResponse

async def is_topic_allowed(user_query: str) -> TopicalGuardResponse:
    response = await LLMClient(guardrail_system_prompt).invoke(user_query)
    return TopicalGuardResponse(classification=response)
```

[![1](assets/1.png)](#co_securing_ai_services_CO1-1)

Handle cases where the LLM doesn’t return a valid classification

###### Tip

Using the technique shown in [Example 9-2](#guardrail_topical), you can implement auto-evaluators to check for jail-breaking and prompt injection attempts or even detect the presence of PII and profanity in the inputs.

As discussed in [Chapter 5](ch05.html#ch05), you can leverage async programming as much as possible even when using auto-evaluation techniques in your guardrails. This is because AI guardrails require sending multiple model API calls per user query. To improve user experience, you can run these guardrails in parallel to the model inference process.

Once you have an auto-evaluator guardrail for checking allowed topics, you can execute it in parallel to your data generation^([1](ch09.html#id1037)) using `asyncio.wait`, as shown in [Example 9-3](#guardrail_concurrent_execution).

###### Warning

Be mindful that implementing async guardrails may trigger model provider API rate-limiting and throttling mechanisms. Depending on your application requirements, you may want to request higher rate limits or reduce the rate of API calls within a short time frame.

##### Example 9-3\. Running AI guardrails in parallel to response generation

```py
import asyncio
from typing import Annotated
from fastapi import Depends
from loguru import logger

...

async def invoke_llm_with_guardrails(user_query: str) -> str:
    topical_guardrail_task = asyncio.create_task(is_topic_allowed(user_query))
    chat_task = asyncio.create_task(llm_client.invoke(user_query))

    while True:
        done, _ = await asyncio.wait(
            [topical_guardrail_task, chat_task],
            return_when=asyncio.FIRST_COMPLETED,
        ) ![1](assets/1.png)
        if topical_guardrail_task in done:
            topic_allowed = topical_guardrail_task.result()
            if not topic_allowed:
                chat_task.cancel() ![2](assets/2.png)
                logger.warning("Topical guardrail triggered")
                return (
                    "Sorry, I can only talk about "
                    "building GenAI services with FastAPI"
                )
            elif chat_task in done:
                return chat_task.result()
        else:
            await asyncio.sleep(0.1) ![3](assets/3.png)

@router.post("/text/generate")
async def generate_text_controller(
    response: Annotated[str, Depends(invoke_llm_with_guardrails)] ![4](assets/4.png)
) -> str:
    return response
```

[![1](assets/1.png)](#co_securing_ai_services_CO2-1)

Create two asyncio tasks to run in parallel using `asyncio.wait`. The operation returns as soon as a task is completed.

[![2](assets/2.png)](#co_securing_ai_services_CO2-2)

If the guardrail is triggered, cancel the chat operation and return a hard-coded response. You can log the trigger in a database and send notification emails here.

[![3](assets/3.png)](#co_securing_ai_services_CO2-3)

Keep checking in with the asyncio event loop every 100 ms until a task is done.

[![4](assets/4.png)](#co_securing_ai_services_CO2-4)

Leverage dependency injection to return the model response if guardrails aren’t triggered.

Since GenAI-enabled guardrails like those you implemented in [Example 9-3](#guardrail_concurrent_execution) remain probabilistic, your GenAI services can still be vulnerable to prompt injection and jail-breaking attacks. For instance, attackers can use more advanced prompt injection techniques to get around your AI guardrails too. On the other hand, your guardrails may also incorrectly over-refuse valid user queries, leading to false positives that can downgrade your user experience.

###### Tip

Combining guardrails with rules-based or traditional machine learning models for detection can help mitigate some of the aforementioned risks.

Additionally, you can use guardrails that only consider the latest message to reduce the risk of the model being confused by a long conversation.

When designing guardrails, you need to consider trade-offs between *accuracy*, *latency*, and *cost* to balance user experience with your required security controls.

## Output Guardrails

The purpose of output guardrails is to validate GenAI-produced content before it’s passed to users or downstream systems. [Table 9-4](#guardrails_output) shows common output guardrails.

Table 9-4\. Common output guardrails

| Output guardrails | Examples |
| --- | --- |
| **Hallucination/fact-checking**Block hallucinations and return canned responses such as “I don’t know.” | Measuring metrics such as *relevancy*, *coherence*, *consistency*, *fluency*, etc., on the model outputs against a corpus of ground truth in RAG applications. |
| **Moderation**Apply brand and corporate guidelines to govern the model outputs, either filtering or rewriting responses that breach them. | Checking against metrics such as *readability*, *toxicity*, *sentiment*, *count of competitor mentions*, etc. |
| **Syntax checks**Verify the structure and content of model outputs. These guardrails can either detect and retry or gracefully handle exceptions to prevent failures in the downstream systems. | Validating JSON schemas and function parameters in *function calling* workflows when models invoke functions.Checking tool/agent selections in *agentic workflows*. |

Any of the aforementioned output guardrails will rely on *threshold value* to detect invalid responses.

## Guardrail Thresholds

Guardrails can use various metrics such as *readability*, *toxicity*, etc., to measure and validate the quality of the model outputs. For each metric, you’ll need to experiment to identify the appropriate *threshold value* for your use case, bearing in mind that:

*   More *false positives* can annoy your users and reduce the usability of your services.

*   More *false negatives* can cause lasting harm to your reputation and explode costs since malicious users can abuse the system or perform prompt injection/jail-breaking attacks.

Normally, you should assess the risks and worst cases of having false negatives and whether you’re happy to trade off a few false negatives in your use case for enhanced user experience. For instance, you can reduce instances of blocking outputs if they include more jargon and aren’t as readable.

## Implementing a Moderation Guardrail

Let’s implement a moderation guardrail using a version of the [*G-Eval* evaluation method](https://oreil.ly/7Nent) to measure the presence of unwanted content in the model output.

The G-Eval framework uses the following components to score invalid content:

*   A *domain* name specifying the type of content to be moderated

*   A set of *criteria* to clearly outline what is considered valid versus invalid content

*   An ordered list of *instruction steps* for grading the content

*   The *content* to grade between a discrete score of 1 to 5

[Example 9-4](#guardrail_moderation_prompt) shows a system prompt implementing the *G-Eval* framework that an LLM auto-evaluator will use.

##### Example 9-4\. Moderation guardrail system prompt

```py
domain = "Building GenAI Services"

criteria = """
Assess the presence of explicit guidelines for API development for GenAI models.
The content should contain only general evergreen advice
not specific tools and libraries to use
"""

steps = """
1\. Read the content and the criteria carefully.
2\. Assess how much explicit guidelines for API development
for GenAI models is contained in the content.
3\. Assign an advice score from 1 to 5,
with 1 being evergreen general advice and 5 containing explicit
mentions of various tools and libraries to use.
"""

f"""
You are a moderation assistant.
Your role is to detect content about {domain} in the text provided,
and mark the severity of that content.

## {domain}

### Criteria

{criteria}

### Instructions

{steps}

### Evaluation (score only!)
"""
```

Using the system prompt implemented in [Example 9-4](#guardrail_moderation_prompt), you can now implement a moderation guardrail following [Example 9-2](#guardrail_topical).

Next, let’s integrate the moderation guardrail with your existing chat invocation logic, as shown in [Example 9-5](#guardrail_moderation).

##### Example 9-5\. Integrating moderation guardrail

```py
import asyncio
from typing import Annotated
from loguru import logger
from pydantic import BaseModel, Field

...

class ModerationResponse(BaseModel):
    score: Annotated[int, Field(ge=1, le=5)] ![1](assets/1.png)

async def g_eval_moderate_content(
    chat_response: str, threshold: int = 3
) -> bool:
    response = await LLMClient(guardrail_system_prompt).invoke(chat_response)
    g_eval_score = ModerationResponse(score=response).score
    return g_eval_score >= threshold ![2](assets/2.png)

async def invoke_llm_with_guardrails(user_request):
    ...
    while True:
        ...
        if topical_guardrail_task in done:
            ...
        elif chat_task in done: ![3](assets/3.png)
            chat_response = chat_task.result()
            has_passed_moderation = await g_eval_moderate_content(chat_response)
            if not has_passed_moderation:
                logger.warning(f"Moderation guardrail flagged")
                return (
                    "Sorry, we can't recommend specific "
                    "tools or technologies at this time"
                )
            return chat_response
        else:
            await asyncio.sleep(0.1)
```

[![1](assets/1.png)](#co_securing_ai_services_CO3-1)

Use a Pydantic constrained integer type to validate LLM auto-evaluator G-Eval score.

[![2](assets/2.png)](#co_securing_ai_services_CO3-2)

Flag content that is scored above the threshold as not passing moderation.

[![3](assets/3.png)](#co_securing_ai_services_CO3-3)

Integrate and run the output moderation guardrail with other guardrails.

###### Tip

Beyond the novel *G-Eval* framework implemented using an LLM auto-evaluator, you can also use more traditional automatic evaluation frameworks such as [ROUGE](https://oreil.ly/_9Q9g), [BERTScore](https://oreil.ly/jRTeL), and [SummEval](https://oreil.ly/5YtJG) for moderating output content.

Well done. You have now implemented two I/O guardrails, one to verify topics of user queries and another to moderate the LLM outputs.

To improve your guardrail system even further, you can:

*   Adopt the *fast failure* approach by exiting early if a guardrail is triggered to optimize response times.

*   Only select *appropriate guardrails* for your use cases instead of using them all together, which could overwhelm your services.

*   Run guardrails *asynchronously* instead of sequentially to optimize latency.

*   Implement *request sampling* by running slower guardrails on a sample of requests to reduce overall latency when your services are under a heavy load.

You should now feel more confident implementing your own guardrails using classical or LLM auto-evaluation techniques without relying on external tools and libraries.

In the next section, you’ll learn about API rate limiting so that you can protect your services against model overloading and scraping attempts.

# API Rate Limiting and Throttling

When deploying GenAI services, you will need to consider service exhaustion and model overloading issues in production. Best practice is to implement rate limiting and potentially throttling into your services.

*Rate limiting* controls the amount of incoming and outgoing traffic to and from a network to prevent abuse, ensure fair usage, and avoid overloading the server. On the other hand, *throttling* controls the API throughput by temporarily slowing down the rate of request processing to stabilize the server.

Both techniques can help you:

*   *Prevent abuse* by blocking malicious users or bots from overwhelming your services from data scraping and brute-force attacks that involve too many requests or large payloads.

*   *Enforce fair usage policies* so that capacity is shared among multiple users and a handful of users are prevented from monopolizing server resources.

*   *Maintain server stability* by regulating incoming traffic to maintain consistent performance and prevent crashes during peak periods.

To implement rate limiting, you will need to monitor incoming requests within a time period and use a queue to balance the load.

There are several rate-limiting strategies you can choose from, which are compared in [Table 9-5](#rate_limiting_strategies) and shown in [Figure 9-2](#rate_limiting_strategies_comparison).

Table 9-5\. Rate-limiting strategies

| Strategy | Benefits | Limitations | Use cases |
| --- | --- | --- | --- |
| **Token Bucket**A list is filled with tokens at a constant rate, and every incoming request consumes a token. If there aren’t enough tokens for incoming requests, they’ll be rejected. |  
*   Handles temporary bursts and dynamic traffic patterns

*   Granular control over request processing

 | Complex to implement | Commonly used in most APIs and services, and interactive or event-driven GenAI systems where request rates can be irregular |
| **Leaky Bucket**Incoming requests are added to a queue and processed at a constant rate to smooth the traffic. If the queue overflows, any new incoming requests are rejected. |  
*   Simple to implement

*   Maintains consistent traffic flow

 |  
*   Less flexible to dynamic traffic

*   May reject valid requests during sudden spikes

 | Services that require maintaining consistent response times in AI inference services |
| **Fixed Window**Limits requests within fixed time windows (e.g., 100 requests per minute). | Simple to implement | Does not handle burst traffic well |  
*   Enforcing strict usage policies for expensive AI inferences and API calls

*   Ideal for free tier users or batch-processing systems with predictable usage patterns

*   Each request is treated equally

 |
| **Sliding Window**Counts requests over a rolling time frame. | Provides better flexibility, granularity, and burst traffic smoothing |  
*   More complex to implement

*   Requires higher memory usage for tracking requests

 |  
*   Much better at handling burst traffic

*   Ideal for conversational AI or premium-tier users who expect flexible, high-frequency access over time

 |

![bgai 0902](assets/bgai_0902.png)

###### Figure 9-2\. Comparison of rate-limiting strategies

Now that you’re more familiar with rate-limiting concepts, let’s try to implement rate limiting in FastAPI.

## Implementing Rate Limits in FastAPI

The fastest approach to add rate limiting within FastAPI is to use a library such as `slowapi` that is a wrapper over the `limits` package, supporting most of the strategies mentioned in [Table 9-5](#rate_limiting_strategies). First, install the `slowapi` library:

```py
$ pip install slowapi
```

Once you’ve installed the `slowapi` package, you can follow [Example 9-6](#rate_limiting_slowapi_configurations) to apply global API or endpoint rate limiting. You can also track and limit usage per IP address.

###### Note

Without configuring an external data store, `slowapi` stores and tracks IP addresses in the application memory for rate limiting.

##### Example 9-6\. Configuring global rate limits

```py
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

...

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "60 per hour", "2/5seconds"],
) ![1](assets/1.png)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded) ![2](assets/2.png)
def rate_limit_exceeded_handler(request, exc):
    retry_after = int(exc.description.split(" ")[-1])
    response_body = {
        "detail": "Rate limit exceeded. Please try again later.",
        "retry_after_seconds": retry_after,
    }
    return JSONResponse(
        status_code=429,
        content=response_body,
        headers={"Retry-After": str(retry_after)},
    )

app.add_middleware(SlowAPIMiddleware)
```

[![1](assets/1.png)](#co_securing_ai_services_CO4-1)

Create rate limiter that tracks usage across each IP address and rejects requests if they exceed specified limits across the application.

[![2](assets/2.png)](#co_securing_ai_services_CO4-2)

Add a custom exception handler for rate-limited requests to compute and provide waiting times before requests are accepted again.

With the `limiter` decorator configured, you can now use it on your API handlers, as shown in [Example 9-7](#rate_limiting_slowapi).

##### Example 9-7\. Setting API rate limits for each API handler

```py
@app.post("/generate/text")
@limiter.limit("5/minute") ![1](assets/1.png)
def serve_text_to_text_controller(request: Request, ...):
    return ...

@app.post("/generate/image")
@limiter.limit("1/minute") ![2](assets/2.png)
def serve_text_to_image_controller(request: Request, ...): ![3](assets/3.png)
    return ...

@app.get("/health")
@limiter.exempt ![4](assets/4.png)
def check_health_controller(request: Request):
    return {"status": "healthy"}
```

[![1](assets/1.png)](#co_securing_ai_services_CO5-1)

Specify more granular rate limits at endpoint level using a rate-limiting decorator. The `limiter` decorator must be ordered last.

[![2](assets/2.png)](#co_securing_ai_services_CO5-2)

Pass the `Request` object to each controller so that the `slowapi` limiter decorator can hook into the incoming request. Otherwise, rate limiting will not function.

[![3](assets/3.png)](#co_securing_ai_services_CO5-3)

Exclude the `/health` endpoint from rate-limiting logic as cloud providers or Docker daemons may ping this endpoint continually to check the status of your application.

[![4](assets/4.png)](#co_securing_ai_services_CO5-4)

Avoid rate limiting the `/health` endpoint as external systems may frequently trigger it to check the current status of your service.

Now that you’ve implemented the rate limits, you can run load tests using the `ab` (Apache Benchmarking) CLI tool, as shown in [Example 9-8](#rate_limiting_load_testing).

##### Example 9-8\. API load testing with Apache Benchmark CLI

```py
$ ab -n 100 -p 2 http://localhost:8000 ![1](assets/1.png)
```

[![1](assets/1.png)](#co_securing_ai_services_CO6-1)

Send 100 requests with a rate of 2 parallel requests per second.

Your terminal outputs should show the following:

```py
200 OK
200 OK
429 Rate limited Exceeded
...
```

Your global and local limiting system should now be working as intended based on incoming IPs.

### User-based rate limits

With an IP rate limit, you’re limiting excess usage based on IP, but users can get around IP rate limiting by using VPNs, proxies, or rotating IP addresses. Instead, you want each user to have a dedicated quota to prevent a single user from consuming all available resources. Adding user-based limits can help you prevent abuse, as shown in [Example 9-9](#rate_limiting_slowapi_users).

##### Example 9-9\. User-based rate limiting

```py
@app.post("/generate/text")
@limiter.limit("10/minute", key_func=get_current_user)
def serve_text_to_text_controller(request: Request):
    return {"message": f"Hello User"}
```

Your system will now be limiting users based on their account IDs alongside their IP addresses.

### Rate limits across instances in production

Since you may run multiple instances of your application in production as you scale your services, you’ll also want to centralize your usage tracking. Otherwise, each instance will provide their own counters to users, and a load balancer distributes requests between instances; usage won’t be capped as you’d expect. To rectify this issue, you can switch the `slowapi` in-memory storage backend with a centralized in-memory database such as Redis, as shown in [Example 9-10](#rate_limiting_slowapi_redis).

###### Note

To run [Example 9-10](#rate_limiting_slowapi_redis), you will need a Redis database to store user API usage data:

```py
$ pip install coredis
$ docker pull redis
$ docker run \
  --name rate-limit-redis-cache \
  -d \
  -p 6379:6379 \
  redis
```

##### Example 9-10\. Adding a centralized usage memory store (Redis) across multiple instances

```py
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware

app.state.limiter = Limiter(storage_uri="redis://localhost:6379")
app.add_middleware(SlowAPIMiddleware)
```

You now have a working rate-limited API that functions as intended across multiple instances.

You can get around this issue by implementing your own limiter supported by the `limits` package instead. Alternatively, you can apply rate limiting via a *load balancer*, a *reverse proxy*, or an *API gateway* instead.

Each solution can route requests while performing rate limits, protocol translation, and traffic monitoring at an infrastructure layer. Applying rate limiting externally may be more suitable for your use case if you don’t require a customized rate-limiting logic.

### Limiting WebSocket connections

Unfortunately the `slowapi` package also doesn’t support limiting async and WebSocket endpoints at the time of writing.

Because WebSocket connections are likely to be long-lived, you may want to limit the data transition rate sent over the socket. You can rely on external packages such as `fastapi-limiter` to rate limit WebSocket connections, as shown in [Example 9-11](#rate_limiting_websocket).

##### Example 9-11\. Rate-limiting WebSocket connections with the `fastapi_limiter` package

```py
from contextlib import asynccontextmanager
import redis
from fastapi import Depends, FastAPI
from fastapi.websockets import WebSocket
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import WebSocketRateLimiter

...

@asynccontextmanager
async def lifespan(_: FastAPI): ![1](assets/1.png)
    redis_connection = redis.from_url("redis://localhost:6379", encoding="utf8")
    await FastAPILimiter.init(redis_connection)
    yield
    await FastAPILimiter.close()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, user_id: int = Depends(get_current_user) ![2](assets/2.png)
):
    ratelimit = WebSocketRateLimiter(times=1, seconds=5)
    await ws_manager.connect(websocket)
    try:
        while True:
            prompt = await ws_manager.receive(websocket)
            await ratelimit(websocket, context_key=user_id) ![3](assets/3.png)
            async for chunk in azure_chat_client.chat_stream(prompt, "ws"):
                await ws_manager.send(chunk, websocket)
    except WebSocketRateLimitException:
        await websocket.send_text(f"Rate limit exceeded. Try again later")
    finally:
        await ws_manager.disconnect(websocket)
```

[![1](assets/1.png)](#co_securing_ai_services_CO7-1)

Configure the `FastAPILimiter` application lifespan with a Redis storage backend.

[![2](assets/2.png)](#co_securing_ai_services_CO7-2)

Configure a WebSocket rate limiter to allow one request per second.

[![3](assets/3.png)](#co_securing_ai_services_CO7-3)

Use the user’s ID as the unique identifier for rate limiting.

[Example 9-11](#rate_limiting_websocket) shows how to limit the number of active WebSocket connections for a given user.

Beyond rate-limiting WebSocket endpoints, you may also want to limit the data streaming rate of your GenAI models. Let’s look at how you can throttle real-time data streams next.

## Throttling Real-Time Streams

When working with real-time streams, you may need to slow down the streaming rate to give clients enough time to consume the stream and improve streaming throughput across multiple clients. In addition, throttling can help you manage the network bandwidth, server load, and resource utilization.

Applying a *throttle* at the stream generation layer, as shown in [Example 9-12](#throttling_stream), is an effective approach to managing throughput if your services are under pressure.

##### Example 9-12\. Throttling streams

```py
class AzureOpenAIChatClient:
    def __init__(self, throttle_rate = 0.5): ![1](assets/1.png)
        self.aclient = ...
        self.throttle_rate = throttle_rate

    async def chat_stream(
            self, prompt: str, mode: str = "sse", model: str = "gpt-3.5-turbo"
    ) -> AsyncGenerator[str, None]:
        stream = ...  # OpenAI chat completion stream
        async for chunk in stream:
            await asyncio.sleep(self.throttle_rate) ![2](assets/2.png)
            if chunk.choices[0].delta.content is not None:
                yield (
                    f"data: {chunk.choices[0].delta.content}\n\n"
                    if mode == "sse"
                    else chunk.choices[0].delta.content
                )
                await asyncio.sleep(0.05)

        if mode == "sse":
            yield f"data: [DONE]\n\n"
```

[![1](assets/1.png)](#co_securing_ai_services_CO8-1)

Set a fixed throttling rate or dynamically adjust based on usage.

[![2](assets/2.png)](#co_securing_ai_services_CO8-2)

Slow down the streaming rate without blocking the event loop.

You can then use the throttled stream within an SSE or WebSocket endpoint. Or, you can limit the number of active WebSocket connections per your own custom policies.

Alongside the application-level throttling for real-time streams, you can also leverage *traffic shaping* at the infrastructure layer.

Using safeguards, rate limits, and throttles should provide enough barriers in protecting your services from abuse and misuse.

In the next section, you’ll learn more about optimization techniques that can help you reduce latency, increase response quality, and throughput alongside reducing the costs of your GenAI services.

# Summary

This chapter provided a comprehensive summary of attack vectors for GenAI services and how to safeguard them against adversarial attempts, misuse, and abuse.

You learned to implement input and output guardrails alongside evaluation and content filtering mechanisms to moderate service usage. Alongside guardrails, you also developed API rate-limiting and throttling protections to manage server load and prevent abuse.

In the next chapter, we will learn about optimizing AI services through various techniques such as caching, batch processing, model quantizing, prompt engineering, and model fine-tuning.

^([1](ch09.html#id1037-marker)) Inspired by [OpenAI Cookbook’s “How to Implement LLM Guardrails”](https://oreil.ly/UQV6i).