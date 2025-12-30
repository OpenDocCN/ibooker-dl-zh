# Chapter 6\. Real-Time Communication with Generative Models

This chapter will explore AI streaming workloads such as chatbots, detailing the use of real-time communication technologies like SSE and WebSocket. You will learn the difference between these technologies and how to implement model streaming by building endpoints for real-time text-to-text interactions.

# Web Communication Mechanisms

In the previous chapter, you learned about implementing concurrency in AI workflows by leveraging asynchronous programming, background tasks, and continuous batching. With concurrency, your services become more resilient to matching increased demand when multiple users access your application simultaneously. Concurrency solves the problem of allowing simultaneous users to access your service and helps to decrease the waiting times, yet AI data generation remains a resource-intensive and time-consuming task.

Up until this point, you’ve been building endpoints using the conventional HTTP communication where the client sends a request to the server. The web server processes the incoming requests and responds via HTTP messages.

[Figure 6-1](#client_server_architecture) shows the client-server architecture.

![bgai 0601](assets/bgai_0601.png)

###### Figure 6-1\. The client-server architecture (Source: [scaleyourapp.com](https://scaleyourapp.com))

Since the HTTP protocol is stateless, the server treats each incoming request completely independent and unrelated from other requests. This means that multiple incoming requests from differing clients wouldn’t affect how the server responds to each one. As an example, in a conversational AI service that doesn’t use a database, each request may provide the full conversation history and receive the correct response from the server.

The *HTTP request-response* model is a widely adopted API design pattern used across the web due to its simplicity. However, this approach becomes inadequate as soon as the client or the server needs real-time updates.

In the standard HTTP request-response model, your services typically respond to the user’s request once it has been entirely processed. However, if the data generation process is lengthy and sluggish, your users will wait a long time and subsequently be inundated with lots of information at once. Imagine chatting to a bot that takes several minutes to reply, and once it does, you’re shown overwhelming blocks of text.

Alternatively, if you provide the data to the client as it’s being generated, rather than holding off until the entire generation process is complete, you can mitigate lengthy delays and deliver the information in digestible chunks. This approach not only enhances user experience but also maintains user engagement during the ongoing processing of their request.

There will be cases where implementing real-time features can be overkill and escalate the development burden. For instance, some open source models or APIs lack the real-time generation capability. Furthermore, adding data streaming endpoints can add to the complexity of your system on both sides, the server and the client. It means having to handle exceptions differently and manage concurrent connections to the streaming endpoints to avoid memory leakage. If the client disconnects during a stream, there may be a chance for data loss or state drift between the server and the client. And, you may need to implement complex reconnection and state management logic to handle cases where the connection drops.

Maintaining many concurrent open connections can also put a burden on your servers and lead to an increase in hosting and infrastructure costs.

Equally important, you also need to consider the scalability of handling a large number of concurrent streams, your application’s latency requirements, and browser compatibilities with your chosen streaming protocol.

###### Note

Compared to traditional web applications that have some form of I/O or data processing latency, AI applications also have AI model inference latency, depending on the model you’re using.

Since this latency can be significant, your AI services should be able to handle longer waiting times on both the server and client sides, including managing the user experience.

If your use case does benefit from real-time features, then you have a few architectural design patterns you can implement:

*   Regular/short polling

*   Long polling

*   SSE

*   WS

The choice depends on your requirements for user experience, scalability, latency, development cost, and maintainability.

Let’s explore each option in more detail.

## Regular/Short Polling

A method to benefit from semi-real-time updates is to use *regular/short polling*, as shown in [Figure 6-2](#short_polling). In this polling mechanism, the client periodically sends HTTP requests to the server to check for updates at preconfigured intervals. The shorter the intervals, the closer you get to real-time updates but also the higher the traffic you will have to manage.

![bgai 0602](assets/bgai_0602.png)

###### Figure 6-2\. Regular/short polling

You can use this technique if you’re building a service to generate data such as images in batches. The client simply submits a request to start the batch job and is given a unique job/request identifier. It then periodically checks back with the server to confirm the status and outputs of the requested job. The server then responds with new data or provides an empty response (and perhaps a status update) if outputs are yet to be computed.

As you can imagine with short polling, you’ll end up with an excessive number of incoming requests that the server needs to respond to, even when there’s no new information. If you have multiple concurrent users, this approach can quickly overwhelm the server, which limits your application’s scalability. However, you can still reduce server load by using cached responses (i.e., executing status checks on the backend at a tolerable frequency) and implementing rate limiting, which you will learn more about in Chapters [9](ch09.html#ch09) and [10](ch10.html#ch10).

A potential use case for short polling in AI services is when you have some in-progress batch or inference jobs. You can expose endpoints for your clients to use short polling to keep up-to-date with the status of these jobs. And, fetch the results when they’re completed.

An alternative is to leverage long polling instead.

## Long Polling

If you want to reduce the burden on your server while continuing to leverage a real-time polling mechanism, you can implement *long polling* (see [Figure 6-3](#long_polling)), an improved version of regular/short polling.

![bgai 0603](assets/bgai_0603.png)

###### Figure 6-3\. Long polling

With long polling, both the server and the client are configured to prevent *timeouts* (if possible) that occur when either the client or the server gives up on the prolonged request.

###### Tip

Timeouts are observed more often in a typical HTTP request-response cycle when a request takes an extended time to resolve or when there are network issues.

To implement long polling, the server keeps the incoming requests open (i.e., hanging) until there is data available to send back. For instance, this can be useful when you have an LLM with unpredictable processing times. The client is instructed to wait for an extended period of time and avoid aborting and repeating the requests prematurely.

You can use long polling if you need a simple API design and application architecture for processing prolonged jobs, such as multiple AI inferences. This technique allows you to avoid implementing a batch job manager to keep track of jobs for bulk data generation. Instead, the client requests remain open until they are processed, avoiding the constant short polling request-response cycle that can overload the server.

While long polling sounds similar to the typical HTTP request-response model, it differs on how the client handles requests. In long polling, the client typically receives a single message per request. Once the server sends a response, the connection is closed. The client then immediately opens a new connection to wait for the next message. This process repeats, allowing the client to receive multiple messages over time, but each HTTP request-response cycle handles only one message.

Since long polling maintains an open connection until a message is available, it reduces the frequency of requests compared to short polling and implements a near-real-time communication mechanism. However, the server still has to hold onto unfulfilled requests, which consume server resources. Additionally, if there are multiple open requests by the same client, message ordering can be challenging to manage, potentially leading to out-of-order messages.

If you don’t have a specific requirement for using polling mechanisms, a more modern alternative to polling mechanisms for real-time communication is SSE via the Event Source interface.

## Server-Sent Events

*Server-sent events* (SSE) is an HTTP-based mechanism for establishing a persistent and unidirectional connection from the server to the client. While the connection is open, the server can continuously push updates to the client as data becomes available.

Once the client establishes the persistent SSE connection with the server, it won’t need to re-establish it again, unlike the long polling mechanism where the client repeatedly sends requests to the server to maintain an open connection.

When you’re serving GenAI models, SSE will be a more suitable real-time communication mechanism compared to long polling. SSE is designed specifically for handling real-time events and is more efficient than long polling. Due to repeated opening and closing connections, long polling becomes resource intensive and leads to higher latency and overhead. SSE, on the other hand, supports automatic reconnection and event IDs to resume interrupted streams, which long polling lacks.

In SSE, the client makes a standard HTTP `GET` request with an `Accept:text/event-stream` header, and the server responds with a status code of `200` and a `Content-Type: text/event-stream` header. After this handshake, the server can send events to the client over the same connection.

While SSE should be your first choice for real-time applications, you can still opt for a simpler long polling mechanism where updates are infrequent or if your environment doesn’t support persistent connections.

One last important detail to note is that SSE connections are *unidirectional*, meaning that you send a regular HTTP request to the server, and you get the response via SSE. Therefore, they’re only suitable for applications that don’t need to send data to the server. You may have seen SSE in action within news feeds, notifications, and real-time dashboards like stock data charts.

Unsurprisingly, SSE also shines in chat applications when you need to stream LLM responses in a conversation. In this instance, the client can establish a separate persistent connection until the server fully streams the LLM’s response to the user.

###### Note

ChatGPT leverages SSE under the hood to enable real-time responses to user queries.

[Figure 6-4](#server_sent_events) shows how the SSE communication mechanism operates.

![bgai 0604](assets/bgai_0604.png)

###### Figure 6-4\. SSE

To solidify your understanding, we will be building two mini-projects in this chapter using SSE. One to stream data from a mocked data generator, and another to stream LLM responses.

You will learn more details about the SSE mechanism during the aforementioned projects.

In summary, SSE is excellent for establishing persistent unidirectional connections, but what if you need to both send and receive messages during a persistent connection? This is where WebSocket would come in handy.

## WebSocket

The last real-time communication mechanism to cover is WebSocket.

WebSocket is an excellent real-time communication mechanism for establishing persistent *bidirectional connections* between the client and the server for real-time chat, as well as voice and video applications with an AI model. A bidirectional connection means that both sides can send and receive real-time data in any order, as long as a persistent connection is open between the client and the server. It’s designed to work over standard HTTP ports to ensure compatibility with existing security measures. Web applications that require two-way communication with servers benefit the most from this mechanism as they can avoid the overhead and complexity of HTTP polling.

You can use WebSocket in a variety of applications including social feeds, multiplayer games, financial feeds, location-based updates, multimedia chat, etc.

Unlike all other communication mechanisms discussed so far, the WebSocket protocol doesn’t transfer data over HTTP after the initial handshake. Instead, the WebSocket protocol defined in the RFC 6455 specification implements a two-way messaging mechanism (full-duplex) over a single TCP connection. As a result, WebSocket is faster for data transmission than HTTP because it has less protocol overhead and operates at a lower level in the network protocol stack. This is because HTTP sits on top of TCP, so stripping back to TCP will be faster.

###### Tip

WebSocket keeps a socket open on both the client and the server for the duration of the connection. Note that this also makes servers stateful, which makes scaling trickier.

You may now be wondering how the WebSocket protocol works.

According to RFC 6455, to establish a WebSocket connection, the client sends an HTTP “upgrade” request to the server, asking to open a WebSocket connection. This is referred to as the *opening handshake*, which initiates the WebSocket connection lifecycle in the *CONNECTING* state.

###### Warning

Your AI services should be able to handle multiple concurrent handshakes and also authenticate them before opening a connection. New connections can consume server resources, so they must be handled properly by your server.

The HTTP upgrade request should contain a set of required headers, as shown in [Example 6-1](#websocket_handshake).

##### Example 6-1\. WebSocket opening handshake over HTTP

```py
GET ws://localhost:8000/generate/text/stream HTTP/1.1 ![1](assets/1.png) Origin: http://localhost:3000
Connection: Upgrade ![2](assets/2.png) Host: http://localhost:8000
Upgrade: websocket ![2](assets/2.png) Sec-WebSocket-Key: 8WnhvZTK66EVvhDG++RD0w== ![3](assets/3.png) Sec-WebSocket-Protocol: html-chat, text-chat ![4](assets/4.png) Sec-WebSocket-Version: 13
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO1-1)

Make an HTTP upgrade request to the WebSocket endpoint. WebSocket endpoints start with `ws://` instead of the typical `http://`.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO1-2)

Request to upgrade and open a WebSocket connection.

[![3](assets/3.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO1-4)

Use a random, 16-byte, Base64-encoded string to ensure the server supports the WebSocket protocol.

[![4](assets/4.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO1-5)

Use the `html-chat` or the `text-chat` subprotocol if `html-chat` is not available. Subprotocols regulate what data will be exchanged.

###### Warning

In production, always use secure WebSocket `wss://` endpoints.

The `wss://` protocol, similar to `https://`, is not only encrypted but also more reliable. That’s because `ws://` data is not encrypted and visible for any intermediary. Old proxy servers don’t know about WebSocket. They may see “strange” headers and abort the connection.

On the other hand, `wss://` is the secure version of WebSocket, running over Transport Layer Security (TLS), which encrypts the data at the sender and decrypts it at the receiver. So data packets are passed encrypted through proxies. They can’t see what’s inside and let them through.

Once the WebSocket connection is established, text or binary messages can be transmitted in both directions in the form of *message frames*. The connection lifecycle is now in the *OPEN* state.

You can view the WebSocket communication mechanism in [Figure 6-5](#websockets).

![bgai 0605](assets/bgai_0605.png)

###### Figure 6-5\. WS communication

*Message frames* are a way to package and transmit data between the client and server. They aren’t anything unique to WebSocket as they apply to all connections over the TCP protocol that form the basis of HTTP. However, a WebSocket message frame consists of several components:

Fixed header

Describes basic information about the message

Extended payload length (optional)

Provides the actual length of the payload when the length exceeds 125 bytes

Masking key

Masks the payload data in frames sent from the client to the server, preventing certain types of security vulnerabilities, particularly *cache poisoning*^([1](ch06.html#id893)) and *cross-protocol*^([2](ch06.html#id894)) attacks

Payload

Contains the actual message content

Unlike the verbose headers in HTTP requests, WebSocket frames have minimal headers that include the following:

Text frames

Used for UTF-8 encoded text data

Binary frames

Used for binary data

Fragmentation

Used to fragment messages into multiple frames, which are reassembled by the recipient

The beauty of the WebSocket protocol is also its ability to maintain a persistent connection through *control frames*.

*Control frames* are special frames used to manage the connection:

Ping/pong frames

Used to check the connection’s status

Close frame

Used to terminate the connection gracefully

When it’s time to close the WebSocket connection, a close frame is sent by the client or the server. The close frame can optionally specify a status code and/or a reason for closing the connection. At this point, the WebSocket connection enters the *CLOSING* state.

The *CLOSING* state ends once the other party responds with another close frame. This concludes the full WebSocket connection lifecycle at the *CLOSED* state, as shown in [Figure 6-6](#websocket_connection_lifecycle).

![bgai 0606](assets/bgai_0606.png)

###### Figure 6-6\. WebSocket connection lifecycle

As you can see, using the WebSocket communication mechanism can be a bit of overkill for simple applications that won’t require the overheads. For most GenAI applications, SSE connections may be enough.

However, there are GenAI use cases where WebSocket can shine, such as multimedia chat and voice-to-voice applications, collaborative GenAI apps, and real-time transcription services based on bidirectional communication. To gain some hands-on experience, you will be building a speech-to-text application later in this chapter.

Now that you’ve learned about several unique web communication mechanisms for real-time applications, let’s quickly summarize how they all compare.

## Comparing Communication Mechanisms

[Figure 6-7](#communication_mechanims_figure) outlines the aforementioned five communication mechanisms used in web development.

![bgai 0607](assets/bgai_0607.png)

###### Figure 6-7\. Comparison of web communication mechanisms

As you can see from [Figure 6-7](#communication_mechanims_figure), the messaging patterns differ in each approach.

*HTTP request-response* is the most common model supported by all web clients and servers, suitable for RESTful APIs and services that don’t require real-time updates.

*Short/regular polling* involves clients checking for data at set intervals, which is straightforward but can be resource-intensive when scaling services. It is normally used in applications to perform infrequent updates such as in analytics dashboards.

*Long polling* is more efficient for real-time updates by keeping connections open until data is available on the server. However, it can still drain the server resources, making it ideal for near-real-time features such as notifications.

*SSE* maintains a single persistent connection that is server-to-client only, using the HTTP protocol. It is straightforward to set up, leverages the browser’s `EventSource` API and ships with built-in features like reconnection. These factors make SSE suitable for applications requiring live feeds, chat features, and real-time dashboards.

*WebSocket* provides full-duplex (double-sided) communication with low latency and binary data support, but is complex to implement. It is widely used in applications requiring high interactivity and real-time data exchange, such as multiplayer games, chat applications, collaborative tools, and real-time transcription services.

With the invention of SSE and WebSocket and their rising popularity, short/regular polling and long polling are becoming less common real-time mechanisms in web applications.

[Table 6-1](#communication_mechanisms_table) compares the features, challenges, and applications for each mechanism in detail.

Table 6-1\. Comparison of web communication mechanisms

| Communication mechanism | Features | Challenges | Applications |
| --- | --- | --- | --- |
| HTTP request-response | Simple request-and-response model, stateless protocol, supported by all web clients and servers | High latency for real-time updates, inefficient for frequent server-to-client data transfer | RESTful APIs, web services where real-time updates aren’t critical |
| Short/regular polling | Client regularly requests data at intervals, easy to implement | Wasteful of resources when there’s no new data, latency depends on poll intervals | Applications with infrequent updates, simple near-real-time dashboards, status updates for submitted jobs |
| Long polling | More efficient than short polling for real-time updates, maintains open connection until data is available | Can be resource-intensive on the server, complex to manage multiple connections | Real-time notifications, older chat applications |
| Server-sent events | Single persistent connection for updates, built-in reconnection and event ID support | Unidirectional communication from server to the client only | Live feeds, chat application, real-time analytics dashboards |
| WebSocket | Full-duplex communication, low latency, supports binary data | More complex to implement and manage, requires WebSocket support on the server | Multiplayer games, chat applications, collaborative editing tools, video conferencing and webinar apps, real-time transcription and translation apps |

Having reviewed real-time communication mechanisms in detail, let’s dive deeper into SSE and WebSocket by implementing our own streaming endpoints using these two mechanisms. In the next section, you will learn how to implement streaming endpoints working with both technologies.

# Implementing SSE Endpoints

In [Chapter 3](ch03.html#ch03), you learned about LLMs, which are *autoregressive* models that predict the next token based on previous inputs. After each generation step, the output token is appended to the inputs and passed through the model again until a `<stop>` token is generated to break the loop. Instead of waiting for the loop to finish, you can forward the output tokens as they’re being generated to the user as a data stream.

Model providers will normally expose an option for you to set the output mode as a data stream using `stream=True`. With this option set, the model provider can return a data generator instead of the final output to you, which you can directly pass to your FastAPI server for streaming.

To demonstrate this in action, refer to [Example 6-2](#async_azure_openai_client), which implements an asynchronous data generator using the `openai` library.

###### Tip

To run [Example 6-2](#async_azure_openai_client), you will need to create an instance of Azure OpenAI on the Azure portal and create a model deployment. Make note of API endpoint, key, and model deployment name. For [Example 6-2](#async_azure_openai_client), you can use the `2023-05-15` api version.

##### Example 6-2\. Implementing Azure OpenAI async chat client for streaming responses

```py
# stream.py

import asyncio
import os
from typing import AsyncGenerator
from openai import AsyncAzureOpenAI

class AzureOpenAIChatClient: ![1](assets/1.png)
    def __init__(self):
        self.aclient = AsyncAzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["OPENAI_API_ENDPOINT"],
            azure_deployment=os.environ["OPENAI_API_DEPLOYMENT"],
        )

    async def chat_stream(
        self, prompt: str, model: str = "gpt-3.5-turbo"
    ) -> AsyncGenerator[str, None]: ![2](assets/2.png)
        stream = await self.aclient.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            stream=True, ![3](assets/3.png)
        )

        async for chunk in stream:
            yield f"data: {chunk.choices[0].delta.content or ''}\n\n" ![4](assets/4.png)
            await asyncio.sleep(0.05) ![5](assets/5.png)

        yield f"data: [DONE]\n\n"

azure_chat_client = AzureOpenAIChatClient()
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-1)

Create an asynchronous `AzureOpenAIChatClient` to interact with the Azure OpenAI API. The chat client requires an API endpoint, deployment name, key, and version to function.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-2)

Define a `chat_stream` asynchronous generator method that yields each output token from the API.

[![3](assets/3.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-3)

Set the `stream=True` to receive an output stream from the API instead of the full response at once.

[![4](assets/4.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-4)

Loop over the stream and yield each output token or return an empty string if `delta.content` is empty. The `data:` substring should be prefixed to each token so that browsers can correctly parse the content using the `EventSource` API.

[![5](assets/5.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-5)

Slow down the streaming rate to reduce back pressure on the clients.

In [Example 6-2](#async_azure_openai_client), you create an instance of `AsyncAzureOpenAI`, which allows you to chat with the Azure OpenAI models via an API in your private Azure environment.

By setting the `stream=True`, `AsyncAzureOpenAI` returns a data stream (an async generator function) instead of the full model response. You can loop over the data stream and `yield` tokens with the `data:` prefix to comply with the SSE specification. This will let browsers to automatically parse the stream content using the widely available `EventSource` web API.^([3](ch06.html#id906))

###### Warning

When exposing streaming endpoints, you’ll need to consider how fast the clients can consume the data you’re sending them. A good practice is to reduce the streaming rate as you saw in [Example 6-2](#async_azure_openai_client) to reduce the back pressure on clients. You can adjust the throttling by testing your services with different clients on various devices.

## SSE with GET Request

You can now implement the SSE endpoint by passing the chat stream to the FastAPI’s `StreamingResponse` as a `GET` endpoint, as shown in [Example 6-3](#sse_endpoint).

##### Example 6-3\. Implementing an SSE endpoint using the FastAPI’s `StreamingResponse`

```py
# main.py

from fastapi.responses import StreamingResponse
from stream import azure_chat_client

...

@app.get("/generate/text/stream") ![1](assets/1.png)
async def serve_text_to_text_stream_controller(
    prompt: str,
) -> StreamingResponse:
    return StreamingResponse( ![2](assets/2.png)
        azure_chat_client.chat_stream(prompt), media_type="text/event-stream"
    )
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO3-1)

Implement an SSE endpoint with the `GET` method to use with the `EventSource` API on the browser.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO3-2)

Pass the chat stream generator to the `StreamingResponse` to forward the output stream as it is being generated to the client. Set the `media_type=text/event-stream` as per SSE specifications so that the browsers can handle the response correctly.

With the `GET` endpoint set up on the server, you can create a simple HTML form on the client to consume the SSE stream via the `EventSource` interface, as shown in [Example 6-4](#sse_client).

###### Tip

[Example 6-4](#sse_client) doesn’t use any JavaScript libraries or web frameworks. However, there are libraries to assist you in implementing the `EventSource` connection in any framework of your choice such as React, Vue, or SvelteKit.

##### Example 6-4\. Implementing SSE on the client using the browser `EventSource` API

```py
{# pages/client-sse.html #} <!DOCTYPE html>
<html lang="en">
<head>
    <title>SSE with EventSource API</title>
</head>
<body>
<button id="streambtn">Start Streaming</button>
<label for="messageInput">Enter your prompt:</label>
<input type="text" id="messageInput" placeholder="Enter your prompt"> ![1](assets/1.png)
<div style="padding-top: 10px" id="responseContainer"></div> ![2](assets/2.png)

<script>
    let source;
    const button = document.getElementById('streambtn');
    const container = document.getElementById('container');
    const input = document.getElementById('messageInput');

    function resetForm(){
        input.value = '';
        container.textContent = '';
    }

    function handleOpen() {
        console.log('Connection was opened');
    }
    function handleMessage(e){
        if (e.data === '[DONE]') {
            source.close();
            console.log('Connection was closed');
            return;
        }

        container.textContent += e.data;
    }
    function handleClose(e){
        console.error(e);
        source.close()
    }

    button.addEventListener('click', function() { ![3](assets/3.png)
        const message = input.value;
        const url = 'http://localhost:8000/generate/text/stream?prompt=' +
            encodeURIComponent(message);
        resetForm() ![4](assets/4.png)

        source = new EventSource(url); ![5](assets/5.png)
        source.addEventListener('open', handleOpen, false);
        source.addEventListener('message', handleMessage, false);
        source.addEventListener('error', handleClose, false); ![6](assets/6.png)
    });

</script>
</body>
</html>
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-1)

Create a simple HTML input and button for initiating SSE requests.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-2)

Create an empty container to be used as a sink for the stream content.

[![3](assets/3.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-3)

Listen for button `clicks` and run the SSE callback.

[![4](assets/4.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-4)

Reset the content form and response container of previous content.

[![5](assets/5.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-5)

Create a new `EventSource` object and listen to connection state changes to handle events.

[![6](assets/6.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-6)

Log to console when an SSE connection is opened. Handle each message by rendering message content to the response container until the `[DONE]` message is received, which signals that the connection should now be closed. Additionally, close the connection if any errors occur and log the error to the browser’s console.

With the SSE client implemented in [Example 6-4](#sse_client), you can now use it to test your SSE endpoint. However, you need to serve the HTML first.

Create a `pages` directory and then place the HTML file inside. Then *mount* the directory onto your FastAPI server to serve its content as static files, as shown in [Example 6-5](#mounting_static_files). Via mounting, FastAPI takes care of mapping API paths to each file so that you can access them with a browser from the same origin as your server.

##### Example 6-5\. Mounting HTML files on the server as static assets

```py
# main.py

from fastapi.staticfiles import StaticFiles

app.mount("/pages", StaticFiles(directory="pages"), name="pages") ![1](assets/1.png)
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO5-1)

Mount the `pages` directory onto the `/pages` to serve its content as static assets. Once mounted, you can access each file by visiting `*<origin>*/pages/*<filename>*`.

By implementing [Example 6-5](#mounting_static_files), you serve the HTML from the same origin as your API server. This avoids triggering the browser’s CORS security mechanism, which can block outgoing requests reaching your server.

You can now access the HTML page by visiting `http://localhost:8000/pages/sse-client.html`.

### Cross-origin resource sharing

If you try to open the [Example 6-4](#sse_client) HTML file in your browser directly and click the Start Streaming button, you will notice that nothing happens. You can check the browser’s network tab to view what happened to the outgoing requests.

After some investigations, you should notice that your browser has blocked outgoing requests to your server as its preflight *cross-origin resource sharing* (CORS) checks with your server have failed.

CORS is a security mechanism implemented in browsers to control how resources on a web page can be requested from another domain, and is relevant only when sending requests directly from the browser instead of a server. Browsers use CORS to check whether they’re allowed to send requests to the server from a different origin (i.e., domain) than the server.

For example, if your client is hosted on `https://example.com` and it needs to fetch data from an API hosted on `https://api.example.com`, the browser will block this request unless the API server has CORS enabled.

For now, you can bypass these CORS errors by adding a CORS middleware on your server, as you can see in [Example 6-6](#cors), to allow any incoming requests from browsers.

##### Example 6-6\. Apply CORS settings

```py
# main.py

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], ![1](assets/1.png)
)
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO6-1)

Allow incoming requests from any origins, methods (`GET`, `POST`, etc.) and headers.

Streamlit avoids triggering the CORS mechanism by sending requests on its internal server even though the generated UI runs on the browser.

On the other hand, the FastAPI documentation page makes requests from the same origin as the server (i.e., `http://localhost:8000`), so requests by default don’t trigger the CORS security mechanism.

###### Warning

In [Example 6-6](#cors), you configure the CORS middleware to process any incoming requests, effectively bypassing the CORS security mechanism for easier development. In production, you should allow only a handful of origins, methods, and headers to be processed by your server.

If you followed Example [6-5](#mounting_static_files) or [6-6](#cors), you should now be able to view the incoming stream from your SSE endpoint (see [Figure 6-8](#sse_results)).

![bgai 0608](assets/bgai_0608.png)

###### Figure 6-8\. Incoming stream from the SSE endpoint

Congratulations! You now have a full working solution where model responses are directly streamed to your client as soon as generated data becomes available. By implementing this feature, your users will now have a more pleasant experience interacting with your chatbot as they receive responses to their queries in real time.

Your solution also implemented concurrency using an asynchronous client for interacting with the Azure OpenAI API to stream faster responses to your users. You can try using a synchronous client to compare the differences in generation speeds. With an asynchronous client, the generation speed can be so fast that you will receive a block of text at once even though it is actually being streamed to the browser.

### Streaming LLM outputs from Hugging Face models

Now that you’ve learned how to implement SSE endpoints with model providers such as Azure OpenAI, you may be wondering if you can stream model outputs from open source models you’ve previously downloaded from Hugging Face.

Although Hugging Face’s `transformers` library implements a `TextStreamer` component that you can pass to your model pipeline, the easiest solution is to run a separate inference server such as HF Inference Server to implement model streaming.

[Example 6-7](#hf_llm_inference_server) shows how to set up a simple model inference server using Docker by providing a `model-id`.

##### Example 6-7\. Serving HF LLM models via HF Inference Server

```py
$ docker run --runtime nvidia --gpus all \ ![1](assets/1.png)
    -v ~/.cache/huggingface:/root/.cache/huggingface \ ![2](assets/2.png)
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \ ![3](assets/3.png)
    -p 8080:8000 \ ![4](assets/4.png)
    --ipc=host \ ![5](assets/5.png)
    vllm/vllm-openai:latest \ ![1](assets/1.png) ![6](assets/6.png)
    --model mistralai/Mistral-7B-v0.1 ![7](assets/7.png)
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-1)

Use Docker to download and run the latest `vllm/vllm-openai` container on all available NVIDIA GPUs.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-2)

Share a volume with the Docker container to avoid downloading weights every run.

[![3](assets/3.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-3)

Set the secret environment variable to access gated models like `mistralai/Mistral-7B-v0.1`.^([4](ch06.html#id909))

[![4](assets/4.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-4)

Run the inference server on localhost port `8080` by mapping host port `8080` to exposed Docker container port `8000`.

[![5](assets/5.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-5)

Enable inter-process communication (IPC) between the container and the host to allow the container to access the host’s shared memory.

[![6](assets/6.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-7)

The vLLM inference server uses the OpenAI API Specification for LLM serving.

[![7](assets/7.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-8)

Download and use the gated `mistralai/Mistral-7B-v0.1` from Hugging Face Hub.

With the model server running, you can now use an `AsyncInferenceClient` to generate outputs in a streaming format, as shown in [Example 6-8](#hf_llm_streaming).

##### Example 6-8\. Consuming the LLM output stream from HF Inference Stream

```py
import asyncio
from typing import AsyncGenerator
from huggingface_hub import AsyncInferenceClient

client = AsyncInferenceClient("http://localhost:8080")

async def chat_stream(prompt: str) -> AsyncGenerator[str, None]:
    stream = await client.text_generation(prompt, stream=True)
    async for token in stream:
        yield token
        await asyncio.sleep(0.05)
```

While [Example 6-8](#hf_llm_streaming) shows how to use the Hugging Face inference server, you can still use other model-serving frameworks such as [vLLM](https://oreil.ly/LQAzF) that support streaming model responses.

Before we move on to talking about WebSocket, let’s look at consuming another variant of SSE endpoints using the `POST` method.

## SSE with POST Request

The [`EventSource` specification](https://oreil.ly/61ovi) expects `GET` endpoints on the server to correctly consume the incoming SSE stream. This makes implementing real-time applications with SSE straightforward as the `EventSource` interface can handle issues such as connection drops and automatic reconnection.

However, using HTTP `GET` requests comes with its own limitations. `GET` requests are normally less secure than other request methods and more vulnerable to *XSS* attacks.^([5](ch06.html#id912)) In addition, since `GET` requests can’t have any request body, you can only transfer data as part of the URL’s query parameters to the server. The issue is that there is a URL length limit you need to consider and any query parameters must be encoded correctly into the request URL. Therefore, you can’t just append the whole conversation history to the URL as a parameter. Your server must handle maintaining the history of the conversation and keeping track of conversational context with `GET` SSE endpoints.

A common workaround to the aforementioned limitation is to implement a `POST` SSE endpoint even if the SSE specification doesn’t support it. As a result, the implementation will be more complex.

First let’s implement the `POST` endpoint on the server in [Example 6-9](#sse_server_post).

##### Example 6-9\. Implementing SSE endpoint on the server

```py
# main.py

from typing import Annotated
from fastapi import Body, FastAPI
from fastapi.responses import StreamingResponse
from stream import azure_chat_client

@app.post("/generate/text/stream")
async def serve_text_to_text_stream_controller(
    prompt: Annotated[str, Body()]
) -> StreamingResponse:
    return StreamingResponse(
        azure_chat_client.chat_stream(prompt), media_type="text/event-stream"
    )
```

With the `POST` endpoint for streaming chat outputs implemented, you can now develop the client logic to process the SSE stream.

You will have to manually process the incoming streaming yourself using the browser’s `fetch` web interface, as shown in [Example 6-10](#sse_client_post).

##### Example 6-10\. Implementing SSE on the client using the browser `EventSource` API

```py
{# pages/client-sse-post.html #} <!DOCTYPE html>
<html lang="en">
<head>
<title>SSE With Post Request</title>
</head>
<body>
<button id="streambtn">Start Streaming</button>
<label for="messageInput">Enter your prompt:</label>
<input type="text" id="messageInput" placeholder="Enter message">
<div style="padding-top: 10px" id="container"></div>

<script>
    const button = document.getElementById('streambtn');
    const container = document.getElementById('container');
    const input = document.getElementById('messageInput');

    function resetForm(){
        input.value = '';
        container.textContent = '';
    }

    async function stream(message){
        const response = await fetch('http://localhost:8000/generate/text/stream', {
            method: "POST",
            cache: "no-cache",
            keepalive: true,
            headers: {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            body: JSON.stringify({
                prompt: message, ![1](assets/1.png)
            }),
        });

        const reader = response.body.getReader(); ![2](assets/2.png)
        const decoder = new TextDecoder(); ![3](assets/3.png)

        while (true) { ![4](assets/4.png)
            const {value, done} = await reader.read();
            if (done) break;
            container.textContent += decoder.decode(value);
        }
    }

    button.addEventListener('click', async function() { ![5](assets/5.png)
        resetForm()
        await stream(input.value)

    });

</script>
</body>
</html>
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO8-1)

Send a `POST` request to the backend using the browser’s `fetch` interface. Prepare the body as a JSON string as part of the request. Add headers to specify the request body being sent and the response that is expected from the server.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO8-2)

Access the `reader` of the stream from the response body stream.

[![3](assets/3.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO8-3)

Create an instance of a text decoder for processing each message.

[![4](assets/4.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO8-4)

Run an infinite loop and read the next message in the stream using the `reader`. If the stream has ended, `done=true`, so break the loop; otherwise, decode the message with the text decoder and append to the response container’s `textContent` to render.

[![5](assets/5.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO8-5)

Listen on button `click` events to run a callback that resets the form state and makes the SSE connection with the backend endpoint with a prompt.

As you can see from [Example 6-10](#sse_client_post), consuming the SSE stream without the `EventSource` can become complex.

###### Tip

An alternative to [Example 6-10](#sse_client_post) is to use `GET` SSE endpoints but send the large payload to the server beforehand using a `POST` request. The server stores the data and uses it when the SSE connection is established.

SSE also supports cookies, so you can rely on cookies to exchange large payloads in `GET` SSE endpoints.

If you want to consume the SSE endpoint in production, your solution should also support retry functionality, error handling, or even the ability to abort connections.

[Example 6-11](#sse_retry) demonstrates how to implement a client-side retry functionality with an *exponential backoff delay* in JavaScript.^([6](ch06.html#id914))

##### Example 6-11\. Implementing client-side retry functionality with exponential backoff

```py
// pages/client-sse-post.html within <script> tag 
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function stream(
    message,
    maxRetries = 3,
    initialDelay = 1000,
    backoffFactor = 2,
) {
    let delay = initialDelay;
    for (let attempt = 0; attempt < maxRetries; attempt++) { ![1](assets/1.png)
        try { ![2](assets/2.png)
            ... // Establish SSE connection here
            return ![3](assets/3.png)
        } catch (error) {
            console.warn(`Failed to establish SSE connection: ${error}`);
            console.log(
                `Re-establishing connection - attempt number ${attempt + 1}`,
            );
            if (attempt < maxRetries - 1) {
                await sleep(delay); ![4](assets/4.png)
                delay *= backoffFactor; ![5](assets/5.png)
            } else {
                throw error ![6](assets/6.png)
            }
        }
    }
}
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO9-1)

As long as `maxRetries` isn’t reached, attempt to establish the SSE connection. Count each attempt.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO9-2)

Use a `try` and `catch` to handle connection errors.

[![3](assets/3.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO9-3)

Exit the function if successful.

[![4](assets/4.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO9-4)

Pause in `delay` milliseconds before retrying.

[![5](assets/5.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO9-5)

Implement exponential backoff by multiplying a backoff factor to the delay value in each iteration.

[![6](assets/6.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO9-6)

Rethrow the `error` if `maxRetries` is reached.

You should now feel more comfortable implementing your own SSE endpoints for streaming model responses. SSE is the go-to communication mechanism that applications like ChatGPT use for real-time conversations with the model. Since SSE predominantly supports text-based streams, it is ideal for LLM output streaming scenarios.

In the next section, we’re going to implement the same solution using the WebSocket mechanism so that you can compare differences in the implementation details. In addition, you’re going to learn what makes WebSocket ideal for scenarios that require real-time duplex communication such as in live transcription services.

# Implementing WS Endpoints

In this section, you’re going to implement an endpoint using the WebSocket protocol. With this endpoint, you will stream the LLM outputs to the client using WebSocket to compare with the SSE connection. By the end, you will learn the differences and similarities between SSE and WebSocket in streaming LLM outputs in real time.

## Streaming LLM Outputs with WebSocket

FastAPI supports WebSocket through the use of the `WebSocket` interface from the Starlette web framework. As WebSocket connections need to be managed, let’s start by implementing a connection manager to keep track of active connections and managing their states.

You can implement a WebSocket connection manager by following [Example 6-12](#websockets_manager).

##### Example 6-12\. Implementing a WebSocket connection manager

```py
# stream.py

from fastapi.websockets import WebSocket

class WSConnectionManager: ![1](assets/1.png)
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None: ![2](assets/2.png)
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None: ![3](assets/3.png)
        self.active_connections.remove(websocket)
        await websocket.close()

    @staticmethod
    async def receive(websocket: WebSocket) -> str: ![4](assets/4.png)
        return await websocket.receive_text()

    @staticmethod
    async def send(
        message: str | bytes | list | dict, websocket: WebSocket
    ) -> None: ![5](assets/5.png)
        if isinstance(message, str):
            await websocket.send_text(message)
        elif isinstance(message, bytes):
            await websocket.send_bytes(message)
        else:
            await websocket.send_json(message)

ws_manager = WSConnectionManager() ![6](assets/6.png)
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-1)

Create a `WSConnectionManager` to track and handle active WS connections.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-2)

Open a WebSocket connection using the `accept()` method. Add the new connection to the list of active connections.

[![3](assets/3.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-3)

When disconnecting, close the connection and remove the `websocket` instance from the active connections list.

[![4](assets/4.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-4)

Receive incoming messages as text during an open connection.

[![5](assets/5.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-5)

Send messages to the client using the relevant send method.

[![6](assets/6.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-6)

Create a single instance of the `WSConnectionManager` to reuse across the app.

You can also extend the connection manager in [Example 6-12](#websockets_manager) to *broadcast* messages (e.g., real-time system alerts, notifications, or updates) to all connected clients. This is useful in applications such as group chats or collaborative whiteboard/document editing tools.

As the connection manager maintains a pointer to every client via the `active_​con⁠nec⁠tions` list, you can broadcast messages to each client, as shown in [Example 6-13](#websockets_broadcast).

##### Example 6-13\. Broadcasting messages to connected clients using the WebSocket manager

```py
# stream.py

from fastapi.websockets import WebSocket

class WSConnectionManager:
    ...
    async def broadcast(self, message: str | bytes | list | dict) -> None:
        for connection in self.active_connections:
            await self.send(message, connection)
```

With the WebSocket manager implemented, you can now develop a WebSocket endpoint to stream responses to the clients. However, before implementing the endpoint, follow [Example 6-14](#chat_stream_ws) to update the `chat_stream` method so that it yields the stream content in a suitable format for WebSocket connections.

##### Example 6-14\. Update the chat client streaming method to yield content suitable for WebSocket connections

```py
# stream.py

import asyncio
from typing import AsyncGenerator

class AzureOpenAIChatClient:
    def __init__(self):
        self.aclient = ...

    async def chat_stream(
        self, prompt: str, mode: str = "sse", model: str = "gpt-4o"
    ) -> AsyncGenerator[str, None]:
        stream = ...  # OpenAI chat completion stream

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None: ![1](assets/1.png)
                yield (
                    f"data: {chunk.choices[0].delta.content}\n\n"
                    if mode == "sse"
                    else chunk.choices[0].delta.content ![2](assets/2.png)
                )
                await asyncio.sleep(0.05)
        if mode == "sse": ![2](assets/2.png)
            yield f"data: [DONE]\n\n"
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO11-1)

Only yield non-empty content.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO11-2)

Yield the stream content based on connection type (SSE or WS).

After updating the `stream_chat` method, you can focus on adding a WebSocket endpoint. Use the `@app.websocket` to decorate a controller function that uses the FastAPI’s `WebSocket` class, as shown in [Example 6-15](#websocket_endpoint).

##### Example 6-15\. Implementing a WS endpoint

```py
# main.py

import asyncio
from loguru import logger
from fastapi.websockets import WebSocket, WebSocketDisconnect
from stream import ws_manager, azure_chat_client

@app.websocket("/generate/text/streams") ![1](assets/1.png)
async def websocket_endpoint(websocket: WebSocket) -> None:
    logger.info("Connecting to client....")
    await ws_manager.connect(websocket) ![2](assets/2.png)
    try: ![3](assets/3.png)
        while True: ![4](assets/4.png)
            prompt = await ws_manager.receive(websocket) ![5](assets/5.png)
            async for chunk in azure_chat_client.chat_stream(prompt, "ws"):
                await ws_manager.send(chunk, websocket) ![6](assets/6.png)
                await asyncio.sleep(0.05) ![7](assets/7.png)
    except WebSocketDisconnect: ![8](assets/8.png)
        logger.info("Client disconnected")
    except Exception as e: ![9](assets/9.png)
        logger.error(f"Error with the WebSocket connection: {e}")
        await ws_manager.send("An internal server error has occurred")
    finally:
        await ws_manager.disconnect(websocket) ![10](assets/10.png)
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-1)

Create a WebSocket endpoint accessible at `ws://localhost:8000/generate/text/stream`.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-2)

Open the WebSocket connection between the client and the server.

[![3](assets/3.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-3)

As long as the connection is open, keep sending or receiving messages.

[![4](assets/4.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-4)

Handle errors and log important events within the `websocket_controller` to identify root causes of errors and handle unexpected situations gracefully. Break the infinite loop when the connection is closed by the server or the client.

[![5](assets/5.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-5)

When the first message is received, pass it as a prompt to OpenAI API.

[![6](assets/6.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-6)

Asynchronously iterate over the generated chat stream and send each chunk to the client.

[![7](assets/7.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-7)

Wait for a small amount of time before sending the next message to reduce race condition issues and allow the client sufficient time for stream processing.

[![8](assets/8.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-8)

When the client closes the WebSocket connection, the `WebSocketDisconnect` exception is raised.

[![9](assets/9.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-9)

If there is a server-side error during an open connection, log the error and identify the client.

[![10](assets/10.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-10)

Break the infinite loop and gracefully close the WebSocket connection if the stream has finished, there is an internal error, or the client has closed the connection. Remove the connection from the active WebSocket connections list.

Now that you have a WebSocket endpoint, let’s develop the client HTML to test the endpoint (see [Example 6-16](#ws_client)).

##### Example 6-16\. Implement client-side WebSocket connections with error handling and exponential backoff retry functionality

```py
{# pages/client-ws.html #} <!DOCTYPE html>
<html lang="en">
<head>
    <title>Stream with WebSocket</title>
</head>
<body>
<button id="streambtn">Start Streaming</button>
<button id="closebtn">Close Connection</button>
<label for="messageInput">Enter your prompt:</label>
<input type="text" id="messageInput" placeholder="Enter message">
<div style="padding-top: 10px" id="container"></div>

<script>
    const streamButton = document.getElementById('streambtn');
    const closeButton = document.getElementById('closebtn');
    const container = document.getElementById('container');
    const input = document.getElementById('messageInput');

    let ws;
    let retryCount = 0;
    const maxRetries = 5;
    let isError = false;

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function connectWebSocket() {
        ws = new WebSocket("ws://localhost:8000/generate/text/streams"); ![1](assets/1.png)

        ws.onopen = handleOpen;
        ws.onmessage = handleMessage;
        ws.onclose = handleClose;
        ws.onerror = handleError; ![2](assets/2.png)
    }

    function handleOpen(){
        console.log("WebSocket connection opened");
        retryCount = 0;
        isError = false;
    }

    function handleMessage(event) {
        container.textContent += event.data;
    }

    async function handleClose(){ ![3](assets/3.png)
        console.log("WebSocket connection closed");
        if (isError && retryCount < maxRetries) {
            console.warn("Retrying connection...");
            await sleep(Math.pow(2, retryCount) * 1000);
            retryCount++;
            connectWebSocket();
        }
        else if (isError) {
            console.error("Max retries reached. Could not reconnect.");
        }
    }

    function handleError(error) {
        console.error("WebSocket error:", error);
        isError = true;
        ws.close();
    }

    function resetForm(){
        input.value = '';
        container.textContent = '';
    }

    streamButton.addEventListener('click', function() { ![4](assets/4.png)
        const prompt = document.getElementById("messageInput").value;
        if (prompt && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(prompt); ![5](assets/5.png)
        }
        resetForm(); ![6](assets/6.png)
    });

    closeButton.addEventListener('click', function() { ![7](assets/7.png)
        isError = false;
        if (ws) {
            ws.close();
        }
    });

    connectWebSocket(); ![1](assets/1.png)
</script>
</body>
</html>
```

[![1](assets/1.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-1)

Establish a WebSocket connection with the FastAPI server.

[![2](assets/2.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-2)

Add callback handlers to the WebSocket connection instance to handle opening, closing, message, and error events.

[![3](assets/3.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-3)

Gracefully handle connection errors and re-establish the connection with an exponential backoff retry functionality using an `isError` flag.

[![4](assets/4.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-4)

Add an event listener to the streaming button to send the first message to the server.

[![5](assets/5.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-5)

Once the connection is established, send the initial non-empty prompt as the first message to the server.

[![6](assets/6.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-6)

Reset the form to before establishing the WebSocket connection to start.

[![7](assets/7.png)](#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-7)

Add an event listener to the close connection button to close the connection when the button is clicked.

Now you can visit [*http://localhost:8000/pages/client-ws.html*](http://localhost:8000/pages/client-ws.html) to test your WebSocket streaming endpoint (see [Figure 6-9](#ws_results)).

![bgai 0609](assets/bgai_0609.png)

###### Figure 6-9\. Incoming stream from the WebSocket endpoint

You should now have a fully working LLM streaming application with WebSocket. Well done!

You now may be wondering which solution is better: streaming with SSE or WS connections. The answer depends on your application requirements. SSE is simple to implement and is native to HTTP protocol, so most clients support it. If all you need is one-way streaming to the client, then I suggest implementing SSE connections for streaming LLM outputs.

WebSocket connections provide more control to your streaming mechanism and allow for duplex communication within the same connection—for instance, in real-time chat applications with multiple users and the LLM, speech-to-text, text-to-speech, and speech-to-speech services. However, using WebSocket requires upgrading the connection from HTTP to the WebSocket protocol, which legacy clients and older browsers may not support. In addition, you will need to handle exceptions slightly differently with WebSocket endpoints.

## Handling WebSocket Exceptions

Handling WebSocket exceptions differs from traditional HTTP connections. If you refer to [Example 6-15](#websocket_endpoint), you will notice that you’re no longer returning a response with status codes, or `HTTPExceptions`, to the client but rather maintaining an open connection after connection acceptance.

As long as the connection is open, you’re sending and receiving messages. However, as soon as an exception has occurred, you should handle it either by gracefully closing the connection and/or by sending an error message to the client in replacement of an `HTTPException` response.

Since the WebSocket protocol doesn’t support the usual HTTP status codes (`4xx` or `5xx`), you can’t use status codes to notify the clients of server-side issues. Instead, you should send WebSocket messages to clients to notify them of issues before you close any active connections from the server.

During the connection closure, you can use several WebSocket-related status codes to specify the closure reason. Using these closure reasons, you can implement any custom closure behavior on the server or the clients.

[Table 6-2](#ws_status_codes) shows a few common status codes that can be sent with a `CLOSE` frame.

Table 6-2\. WebSocket protocol common status codes

| Status code | Description |
| --- | --- |
| 1000 | Normal closure |
| 1001 | Client navigated away or server has gone down |
| 1002 | An endpoint (i.e., client or server) received data violating the WS protocol (e.g., unmasked packets, invalid payload length) |
| 1003 | An endpoint received unsupported data (e.g., was expecting text, got binary) |
| 1007 | An endpoint received inconsistently encoded data (e.g., non-UTF-8 data within a text message) |
| 1008 | An endpoint received a message that violates its policy; can be used to hide closure details for security reasons |
| 1011 | Internal server error |

You can learn more about other WebSocket status codes in the WebSocket protocol [RFC 6455—Section 7.4](https://oreil.ly/1L_HH).

## Designing APIs for Streaming

Now that you’re more familiar with both SSE and WebSocket endpoint implementations, I want to cover one last important detail around their architectural design.

A common pitfall of designing streaming APIs is exposing an excessive number of streaming endpoints. For instance, if you’re building a chatbot application, you may expose several streaming endpoints, each preconfigured to handle different incoming messages in a single conversation. By using this particular API design pattern, you’re asking the client to switch between endpoints, providing the necessary information in each step while navigating the streaming connections during a single conversation. This design pattern adds to the complexity of both the backend and frontend applications since the conversation states need to be managed on both sides while avoiding race condition and networking issues between components.

A simpler API design pattern is to provide a single entry point for the client to initiate a stream with your GenAI model(s) and use headers, request body, or query parameters to trigger the relevant logic in the backend. With this design, the backend logic is abstracted away from the client, which simplifies state management on the frontend while all routing and business logic are implemented on the backend. Since the backend has access to databases, other services, and customized prompts, it can easily perform CRUD operations and switch between prompts or models to compute a response. Therefore, one endpoint can act as a single entry point for switching logic, manage application states, and generate custom responses.

# Summary

This chapter covered several different strategies for implementing real-time communication via data streaming in your GenAI services.

You learned about several web communication mechanisms including the traditional HTTP request-response model, short/regular polling, long polling, SSE, and WebSocket. You then compared these mechanisms in detail to understand their features, benefits, disadvantages, and use cases, in particular for AI workflows. Finally, you implemented two LLM streaming endpoints using the asynchronous Azure OpenAI client to learn how to leverage SSE and WebSocket real-time communication mechanisms.

In the next chapter, you will learn more about API development workflows when integrating databases for AI services. This will include how to set up, migrate, and interact with databases. You’ll also learn how to handle data storage-and-retrieval operations within streaming endpoints by using FastAPI’s background tasks.

Topics covered in the next chapter will include setting up databases and designing schemas, working with the SQLAlchemy, database migrations, and handling database operations when streaming model outputs.

^([1](ch06.html#id893-marker)) Attackers can use cache poisoning to inject malicious data to caching systems, which then serve incorrect data to users or systems. To protect against this attack, the client and the server mask payloads to appear as random data before sending them.

^([2](ch06.html#id894-marker)) These attacks involve tricking a server into leaking sensitive information by sending an HTTP response to a WebSocket frame.

^([3](ch06.html#id906-marker)) See MDN resources for more details on the [`EventSource` interface](https://oreil.ly/0yuKA).

^([4](ch06.html#id909-marker)) Follow the [“Accessing Private/Gated Models” guide](https://oreil.ly/a7KeV) to generate a Hugging Face user access token.

^([5](ch06.html#id912-marker)) Attackers use the XSS vulnerability to insert harmful scripts into web pages, which are then executed by other users’ browsers.

^([6](ch06.html#id914-marker)) Exponential backoff reduces the chances of API rate-limiting errors by increasing the delay after each retry.