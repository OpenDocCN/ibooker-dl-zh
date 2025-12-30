# 第六章. 使用生成模型进行实时通信

这项工作使用了 AI 进行翻译。我们很高兴收到您的反馈和评论：translation-feedback@oreilly.com

本章将探讨流式 AI 工作负载，如聊天机器人，展示实时通信技术（如 SSE 和 WebSocket）的使用。你将学习这些技术的区别以及如何通过构建实时文本-文本交互端点来实现模型流。

# 网络通信机制

在上一章中，你学习了如何利用异步编程、后台任务和持续批处理来实现人工智能工作流的竞争。有了竞争，你的服务在面对更多用户同时访问应用程序时，需求增加时变得更加健壮。竞争解决了允许同时用户访问你的服务的问题，并有助于减少等待时间，但人工智能数据生成仍然是一项需要大量资源和时间的活动。

到目前为止，你已经使用传统的 HTTP 通信构建了端点，其中客户端向服务器发送请求。Web 服务器处理传入的请求并通过 HTTP 消息进行响应。

图 6-1 展示了客户端-服务器架构。

![bgai 0601](img/bgai_0601.png)

###### 图 6-1. 客户端-服务器架构（来源：[scaleyourapp.com](https://scaleyourapp.com))

由于 HTTP 协议是无状态的，服务器以完全独立和不相关的方式处理每个到达的请求。这意味着来自不同客户端的多个请求不会影响服务器对每个请求的响应方式。例如，在一个不使用数据库的人工智能对话服务中，每个请求都可以提供整个会话的历史记录，并从服务器接收正确的响应。

*HTTP 请求-响应*模型是因其在整个网络中的简单性而被广泛采用的 API 设计模型。然而，一旦客户端或服务器需要实时更新，这种方法就变得不适用。

在标准的 HTTP 请求-响应模型中，你的服务在完全处理完用户请求后响应。然而，如果数据生成过程漫长且缓慢，你的用户将等待很长时间，随后会一次性接收到大量信息。想象一下与一个需要几分钟才能回复的聊天机器人聊天，一旦回复，你会看到过多的文本块。

或者，如果你在数据生成时向客户提供数据，而不是等待整个生成过程完成，你可以减少长时间延迟，并以可消化的片段提供信息。这种方法不仅改善了用户体验，而且在处理请求的过程中保持了用户的参与度。

有时候，实现实时功能可能会过度，并增加开发负担。例如，一些开源模型或 API 没有实时生成的能力。此外，添加数据流端点可能会增加系统两端的复杂性，包括服务器和客户端。这意味着需要以不同的方式处理异常，并管理对数据流端点的并发连接以避免内存损失。如果客户端在流传输过程中断开连接，可能会发生数据丢失或服务器和客户端之间的状态漂移。此外，可能需要实现复杂的重连和状态管理逻辑来处理连接中断的情况。

维持许多并发打开的连接也可能对你的服务器造成负担，并导致托管和基础设施成本的增加。

同样重要的是考虑管理大量并发流的可扩展性，你的应用对延迟的要求以及浏览器与所选流协议的兼容性。

###### 注意

与具有某种形式的 I/O 或数据处理延迟的传统 Web 应用相比，AI 应用还有 AI 模型推断的延迟，这取决于所使用的模型。

由于这种延迟可能相当显著，你的 AI 服务必须在服务器和客户端两方面都能处理更长的等待时间，包括管理用户体验。

如果你的用例受益于实时功能，那么你可以实施以下一些架构设计模型：

+   定期/短周期调查

+   长期调查

+   SSE

+   WS

选择取决于你的用户体验、可扩展性、延迟、开发成本和维护性的要求。

我们将更详细地分析每个选项。

## 定期/短周期调查

一种利用半实时更新的方法是使用*定期/短周期轮询*，如图 6-2 所示。在这种轮询机制中，客户端定期向服务器发送 HTTP 请求，以检查预配置间隔内的更新。间隔越短，越接近实时更新，但需要处理的数据流量也会更高。

![bgai 0602](img/bgai_0602.png)

###### 图 6-2\. 定期/短周期轮询

如果你正在构建一个用于批量生成数据（如批量图像）的服务，你可以使用这种技术。客户端只需发送一个请求来启动批量工作，并为工作/请求分配一个唯一的标识符。然后定期检查服务器以确认所需工作的状态和结果。如果结果还需要计算，服务器会以新数据或空响应（可能还有状态更新）进行回复。

就像你可以想象到的，使用短轮询，你可能会遇到过多的入站请求，服务器必须响应，即使没有新的信息。如果你有多个同时在线的用户，这种方法可能会迅速使服务器过载，限制你应用程序的可扩展性。然而，你可以通过使用缓存中的响应（即以可容忍的频率在后端执行状态检查）和实现速率限制来减少服务器的负载，你将在第九章和第十章中更好地了解这些内容。

在人工智能服务中，短轮询的一个潜在用例是当你有正在进行的批量或推理工作。你可以公开一些端点，允许你的客户使用短轮询来更新这些工作的状态，并在它们完成时检索结果。

另一个选择是利用长轮询。

## 长轮询

如果你想在继续利用实时轮询机制的同时减少服务器负载，你可以实现*长轮询*（见图 6-3），这是常规/短轮询的一个改进版本。

![bgai 0603](img/bgai_0603.png)

###### 图 6-3\. 长轮询

在长轮询中，服务器和客户端都被配置为避免（如果可能的话）当客户端或服务器放弃长时间请求时发生的超时。

###### 建议

在典型的 HTTP 请求-响应循环中，超时情况更常发生在请求需要较长时间才能解决或存在网络问题时。

为了实现长轮询，服务器会保持入站请求（即挂起）直到有数据可以发送。例如，这可能在拥有具有不可预测处理时间的 LLM 时很有用。客户端被指导等待较长时间，以避免过早地中断和重复请求。

如果你需要一个简单的 API 设计和适用于长时间处理工作的应用程序架构，例如人工智能的多次推理，你可以使用长时间轮询。这项技术允许你避免实现一个批处理工作管理器来跟踪批量生成数据的工作。相反，客户端的请求会保持打开状态，直到它们被处理，从而避免了频繁的请求-响应循环，这可能会给服务器带来过载。

虽然长时间轮询看起来与典型的 HTTP 请求-响应模型相似，但它在客户端处理请求的方式上有所不同。在长时间轮询中，客户端通常为每个请求接收一条单独的消息。一旦服务器发送了响应，连接就会被关闭。然后客户端立即打开一个新的连接以等待下一条消息。这个过程会重复进行，使得客户端能够在一段时间内接收更多的消息，但每个 HTTP 请求-响应周期只处理一条消息。

由于长时间轮询在消息可用之前保持连接打开，因此它比短时间轮询减少了请求频率，并实现了一种近乎实时的通信机制。然而，服务器仍然需要保持未满足的请求，这会消耗服务器的资源。此外，如果同一客户端有多个打开的请求，消息排序可能会变得难以管理，可能导致消息顺序混乱。

如果你没有对轮询机制的具体使用要求，实时通信的更现代替代方案是通过 Event Source 接口的 SSE（Server-Sent Events）。

## 服务器发送的事件

由服务器发送的*事件*（SSE）是一种基于 HTTP 的机制，用于在服务器和客户端之间建立持久和单向的连接。当连接打开时，服务器可以在数据变得可用时持续地向客户端发送更新。

一旦客户端与服务器建立了持久的 SSE 连接，它就不再需要重新建立连接，这与长时间轮询机制不同，在长时间轮询中，客户端需要反复向服务器发送请求以保持连接打开。

如果你正在服务生成人工智能的模型，SSE 比长时间轮询更适合作为实时通信机制。SSE 被专门设计来处理实时事件，并且比长时间轮询更高效。由于长时间轮询需要频繁地打开和关闭连接，它需要更多的资源，并可能导致延迟和开销的增加。相反，SSE 支持自动重连和事件 ID，以便恢复中断的流，这是长时间轮询所不具备的。

在 SSE 中，客户端通过发送一个带有`Accept:text/event-stream`头的标准 HTTP `GET`请求来发起操作，服务器则以状态码`200`和`Content-Type: text/event-stream`头作为响应。在此交换之后，服务器可以通过相同的连接向客户端发送事件。

即使 SSE 应该是实时应用的优先选择，但在更新不频繁或你的环境不支持持久连接的情况下，你也可以选择更简单的轮询机制。

需要注意的一个重要细节是，SSE（Server-Sent Events）的连接是**单向的**，也就是说，你向服务器发送一个正常的 HTTP 请求，并通过 SSE 接收响应。因此，它们只适用于不需要向服务器发送数据的那些应用。你可能已经在新闻推送、实时通知和实时仪表板（如股票数据图表）中看到过 SSE 的应用。

在需要将 LLM 的响应在对话中实时传输的应用中，SSE 特别适合。在这种情况下，客户端可以建立一个独立的持久连接，直到服务器完全将 LLM 的响应传输给用户。

###### 注意

ChatGPT 在内部使用 SSE 来允许对用户查询的实时响应。

图 6-4 展示了 SSE 通信机制的工作原理。

![bgai 0604](img/bgai_0604.png)

###### 图 6-4\. SSE

为了加深你的理解，在本章中我们将通过两个使用 SSE 的小型项目来实现：一个用于从模拟数据生成器中流式传输数据，另一个用于流式传输 LLM（大型语言模型）的响应。

你将在上述项目中了解更多关于 SSE 机制的具体细节。

总结来说，SSE 非常适合建立单向的持久连接，但如果在持久连接期间需要发送和接收消息，那么 WebSocket 可能是有用的。

## WebSocket

最后要讨论的实时通信机制是 WebSocket。

WebSocket 是一个优秀的实时通信机制，用于在客户端和服务器之间建立持久的**双向连接**，适用于实时聊天、带有人工智能模型的声音和视频应用。双向连接意味着双方可以实时地、无序地发送和接收数据，只要客户端和服务器之间保持一个持久的连接。它被设计为在标准 HTTP 端口上工作，以确保与现有安全措施的兼容性。需要与服务器进行双向通信的 Web 应用可以从这种机制中获得最大好处，因为它们可以避免 HTTP 轮询的开销和复杂性。

你可以在各种应用程序中使用 WebSocket，包括社交动态、多人游戏、金融动态、基于位置的更新、多媒体聊天等。

与迄今为止讨论的所有其他通信机制不同，WebSocket 协议在初始握手之后不在 HTTP 上传输数据，而是在单个 TCP 连接上实现双向消息机制（全双工）。因此，WebSocket 在数据传输速度上比 HTTP 更快，因为它具有更少的协议开销，并在网络协议栈的较低级别运行。这是因为 HTTP 位于 TCP 之上，因此返回到 TCP 的速度会更快。

###### 建议

WebSocket 在客户端和服务器上保持一个打开的套接字，直到连接的整个持续时间。请注意，这也使得服务器变得静态，这使得扩展更加复杂。

到目前为止，你可能想知道 WebSocket 协议是如何工作的。

根据 RFC 6455，为了建立 WebSocket 连接，客户端向服务器发送一个“更新”HTTP 请求，请求打开一个 WebSocket 连接。这被称为*打开握手*，它标志着 WebSocket 连接生命周期的*CONNECTING*状态的开始。

###### 注意事项

你的 AI 服务必须能够处理多个同时握手并验证它们，然后再打开连接。新连接可能会消耗服务器资源，因此必须由你的服务器正确管理。

HTTP 更新请求必须包含一系列必要的头信息，如 Esempio 6-1 所示。

##### Esempio 6-1\. 通过 HTTP 打开 WebSocket

```py
GET ws://localhost:8000/generate/text/stream HTTP/1.1 ![1](img/1.png) Origin: http://localhost:3000
Connection: Upgrade ![2](img/2.png) Host: http://localhost:8000
Upgrade: websocket ![2](img/2.png) Sec-WebSocket-Key: 8WnhvZTK66EVvhDG++RD0w== ![3](img/3.png) Sec-WebSocket-Protocol: html-chat, text-chat ![4](img/4.png) Sec-WebSocket-Version: 13
```

![1](img/1.png)(#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO1-1)

向 WebSocket 端点发送 HTTP 更新请求。WebSocket 端点以`ws://`开头，而不是典型的`http://`。

![2](img/2.png)(#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO1-2)

更新请求和 WebSocket 连接的打开。

![3](img/3.png)(#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO1-4)

使用一个 16 字节的随机字符串，并使用 Base64 进行编码，以确保服务器支持 WebSocket 协议。

![4](img/4.png)(#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO1-5)

如果`html-chat`不可用，则使用子协议`html-chat`或`text-chat`。子协议控制交换的数据。

###### 注意事项

在生产环境中，始终使用安全的 WebSocket 端点`wss://`。

`wss://`协议，类似于`https://`，不仅加密，而且更可靠。这是因为`ws://`的数据未加密，任何中间代理都可以看到。旧的代理服务器不了解 WebSocket，可能会看到“奇怪”的头部并中断连接。

另一方面，`wss://`是 WebSocket 的安全版本，它基于传输层安全（TLS）工作，加密发送方的数据并在接收方解密。因此，数据包以加密形式通过代理服务器传输，代理服务器无法看到数据内容并允许其通过。

一旦建立 WebSocket 连接，文本或二进制消息可以以*消息帧*的形式在两个方向上传输。现在，连接的生命周期处于*打开*状态。

你可以在图 6-5 中看到 WebSocket 通信机制。

![bgai 0605](img/bgai_0605.png)

###### 图 6-5\. WS 通信

*消息帧*是客户端和服务器之间打包和传输数据的一种方式。它不是 WebSocket 的专属，因为它适用于所有通过 TCP 协议（HTTP 的基础）建立的连接。然而，WebSocket 的消息帧由多个组件组成：

固定头部

描述消息的基本信息

扩展负载长度（可选）

当负载长度超过 125 字节时，提供实际负载长度。

隐藏键

在客户端发送到服务器的帧中隐藏负载数据，防止某些类型的安全漏洞，特别是*缓存中毒*^(1)和*跨协议*^(2)攻击。

负载数据

包含消息的实际内容

与 HTTP 请求的冗长头部不同，WebSocket 帧具有最小头部，包括以下内容：

文本框架

用于 UTF-8 编码的文本数据

二进制框架

用于二进制数据

分帧

用于将消息分帧为多个帧，这些帧由接收方重新组装。

WebSocket 协议的优点之一是其通过*控制帧*保持持久连接的能力。

*控制帧*是用于管理连接的特殊帧：

ping/pong 框架

用于控制连接状态

关闭框架

用于优雅地终止连接

当需要关闭 WebSocket 连接时，客户端或服务器发送一个关闭帧。关闭帧可以可选地指定状态码和/或关闭连接的原因。此时，WebSocket 连接进入*关闭*状态。

当另一端响应另一个关闭帧时，*CLOSING*状态结束，从而在图 6-6 中展示了 WebSocket 连接的生命周期结束于*CLOSED*状态。

![bgai 0606](img/bgai_0606.png)

###### 图 6-6\. WebSocket 连接的生命周期

如你所见，对于不需要额外开销的简单应用，使用 WebSocket 通信机制可能有些过度。对于大多数 GenAI 应用来说，SSE 连接可能已经足够。

然而，在某些 GenAI 用例中，WebSocket 可以大放异彩，例如多媒体聊天和语音对语音的应用、GenAI 协作应用以及基于双向通信的实时转录服务。为了获得一些实践经验，在本章的后面部分，你将实现一个语音到文本的应用程序。

现在你已经了解了针对实时应用的不同 Web 通信机制，让我们快速总结一下它们的对比。

## 通信机制对比

图 6-7 展示了在 Web 开发中使用的五种通信机制。

![bgai 0607](img/bgai_0607.png)

###### 图 6-7\. Web 通信机制的对比

如图 6-7 所示，不同的方法中消息模型各不相同。

*HTTP 请求-响应*是所有客户端和服务器 Web 应用中最常见的模型，适用于 RESTful API 和不需要实时更新的服务。

*短轮询/常规轮询*要求客户端在预定的时间间隔内检查数据的存在，这很简单，但在服务扩展时可能会在资源消耗上变得昂贵。通常用于需要不频繁更新的应用程序，例如分析仪表板。

*长轮询*对于实时更新来说更有效率，因为它保持连接开启，直到服务器上有可用数据。然而，它仍然可能耗尽服务器的资源，这使得它对于需要近似实时功能的操作（如通知）来说很理想。

*服务器发送事件（SSE）*仅通过 HTTP 协议在服务器和客户端之间保持一个持久的单一连接。它易于配置，利用浏览器的`EventSource` API，并提供了如自动重连等集成功能。这些因素使得 SSE 适用于需要实时流、聊天和实时仪表板功能的应用程序。

*WebSocket*提供全双工（双向）通信，低延迟，支持二进制数据，但实现复杂。它广泛应用于需要高交互性和实时数据交换的应用，如多玩家游戏、聊天应用、协作工具和实时转录服务。

随着 SSE 和 WebSocket 的发明及其日益普及，短轮询/常规轮询和长轮询正在成为 Web 应用中越来越不常见的实时机制。

表 6-1\[通信机制表\]比较了每个机制的特点、挑战和应用。

表 6-1\. Web 通信机制对比

| 通信机制 | 特点 | 挑战 | 应用 |
| --- | --- | --- | --- |
| 请求-响应 HTTP | 简单的请求和响应模型，无状态协议，所有客户端和服务器都支持。 | 实时更新延迟高，频繁数据传输从服务器到客户端效率低 | RESTful API，实时更新不是关键的服务 Web |
| 短轮询/常规轮询 | 客户定期请求数据，易于实现 | 没有新数据时资源浪费，延迟取决于轮询间隔 | 更新频率较低的应用，几乎实时的简单仪表板，提交工作的状态更新 |
| 长轮询 | 对于实时更新比短轮询更高效，保持连接打开直到数据可用 | 可能需要服务器大量资源，管理多个连接复杂 | 实时通知，旧版聊天应用 |
| 服务器发送的事件 | 单个持久连接用于更新，集成重连和事件 ID 支持 | 仅从服务器到客户端的单向通信 | 直播源，聊天应用，实时分析仪表板 |
| WebSocket | 全双工通信，低延迟，支持二进制数据 | 实现和管理更复杂，需要服务器支持 WebSocket | 多玩家游戏，聊天应用，协作编辑工具，视频会议和网络研讨会应用，实时转录和翻译应用 |

在详细研究了实时通信机制之后，我们将深入探讨 SSE 和 WebSocket，通过使用这两种机制实现我们的流端点。在下一节中，你将了解如何使用这两种技术实现流端点。

# 实现 SSE 端点

在第三章中，你学习了关于 LLMs 的知识，它们是*自回归*模型，根据先前输入预测下一个令牌。在每个生成阶段之后，输出令牌被添加到输入中，并再次通过模型传递，直到生成一个`<stop>`令牌来中断循环。而不是等待循环结束，你可以将生成的令牌作为数据流发送给用户。

模型提供商通常提供一个选项，允许你设置输出模式为数据流，使用`stream=True`。使用此选项，模型提供商可以返回一个数据生成器而不是最终输出，你可以直接将其传递到你的 FastAPI 服务器进行流式传输。

为了演示其功能，请参考 Esempio 6-2，它使用`openai`库实现了一个异步数据生成器。

###### 建议

要执行 Esempio 6-2，你需要在 Azure 门户上创建一个 Azure OpenAI 实例并创建一个分发模型。注意记录 API 端点、密钥和分发模型名称。对于 Esempio 6-2，你可以使用`2023-05-15`版本的 api。

##### 示例 6-2\. 实现 Azure OpenAI 异步聊天客户端以进行响应流式传输

```py
# stream.py

import asyncio
import os
from typing import AsyncGenerator
from openai import AsyncAzureOpenAI

class AzureOpenAIChatClient: ![1](img/1.png)
    def __init__(self):
        self.aclient = AsyncAzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["OPENAI_API_ENDPOINT"],
            azure_deployment=os.environ["OPENAI_API_DEPLOYMENT"],
        )

    async def chat_stream(
        self, prompt: str, model: str = "gpt-3.5-turbo"
    ) -> AsyncGenerator[str, None]: ![2](img/2.png)
        stream = await self.aclient.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            stream=True, ![3](img/3.png)
        )

        async for chunk in stream:
            yield f"data: {chunk.choices[0].delta.content or ''}\n\n" ![4](img/4.png)
            await asyncio.sleep(0.05) ![5](img/5.png)

        yield f"data: [DONE]\n\n"

azure_chat_client = AzureOpenAIChatClient()
```

![1](img/1.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-1]

创建一个异步客户端 `AzureOpenAIChatClient` 以与 Azure OpenAI API 交互。聊天客户端需要一个 API 端点、分发名称、密钥和版本才能运行。

![2](img/2.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-2]

定义一个异步生成器方法 `chat_stream`，它从 API 生成每个输出令牌。

![3](img/3.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-3]

设置`stream=True`以从 API 接收输出流而不是一次性接收完整响应。

![4](img/4.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-4]

在流上执行循环，并返回每个输出令牌或如果`delta.content`为空，则返回一个空字符串。每个令牌前必须加上前缀`data:`，以便浏览器可以使用`EventSource` API 正确分析内容。

![5](img/5.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO2-5]

减慢流式传输速度以减少对客户端的冲击。

在 Esempio 6-2 中，你创建了一个`AsyncAzureOpenAI`实例，这允许你通过 Azure 私有的 API 环境与 Azure OpenAI 模型进行聊天。

Impostando il prefisso `stream=True`, `AsyncAzureOpenAI` restituisce un flusso di dati (una funzione generatrice asincrona) invece della risposta completa del modello. Puoi eseguire un loop sul flusso di dati e sui token `yield` con il prefisso `data:` per conformarti alle specifiche SSE. In questo modo i browser potranno analizzare automaticamente il contenuto del flusso utilizzando l'API web `EventSource`, ampiamente disponibile.^(3)

###### Avvertenze

Quando esponi endpoint di streaming, dovrai considerare la velocità con cui i client possono consumare i dati che gli stai inviando. Una buona pratica è quella di ridurre la velocità di streaming come hai visto nell'Esempio 6-2 per ridurre la pressione sui client. Puoi regolare il throttling testando i tuoi servizi con diversi client su vari dispositivi.

## SSE con richiesta GET

Ora puoi implementare l'endpoint SSE passando il flusso di chat al sito `StreamingResponse` di FastAPI come endpoint `GET`, come mostrato nell'Esempio 6-3.

##### Esempio 6-3\. Implementazione di un endpoint SSE utilizzando la FastAPI `StreamingResponse`

```py
# main.py

from fastapi.responses import StreamingResponse
from stream import azure_chat_client

...

@app.get("/generate/text/stream") ![1](img/1.png)
async def serve_text_to_text_stream_controller(
    prompt: str,
) -> StreamingResponse:
    return StreamingResponse( ![2](img/2.png)
        azure_chat_client.chat_stream(prompt), media_type="text/event-stream"
    )
```

![1](img/1.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO3-1]

Implementa un endpoint SSE con il metodo `GET` da utilizzare con l'API `EventSource` sul browser.

![2](img/2.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO3-2]

Passa il generatore di flussi di chat a `StreamingResponse` per inoltrare il flusso di output che viene generato al client. Imposta `media_type=text/event-stream` come da specifiche SSE in modo che i browser possano gestire correttamente la risposta.

Con l'endpoint `GET` configurato sul server, puoi creare un semplice modulo HTML sul client per consumare il flusso SSE tramite l'interfaccia `EventSource`, come mostrato nell'Esempio 6-4.

###### Suggerimento

L'esempio 6-4 non utilizza alcuna libreria JavaScript o framework web, ma esistono librerie che ti aiuteranno a implementare la connessione`EventSource` in qualsiasi framework di tua scelta come React, Vue o SvelteKit.

##### Esempio 6-4\. Implementazione di SSE sul client utilizzando l'API del browser `EventSource`

```py
{# pages/client-sse.html #} <!DOCTYPE html>
<html lang="en">
<head>
    <title>SSE with EventSource API</title>
</head>
<body>
<button id="streambtn">Start Streaming</button>
<label for="messageInput">Enter your prompt:</label>
<input type="text" id="messageInput" placeholder="Enter your prompt"> ![1](img/1.png)
<div style="padding-top: 10px" id="responseContainer"></div> ![2](img/2.png)

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

    button.addEventListener('click', function() { ![3](img/3.png)
        const message = input.value;
        const url = 'http://localhost:8000/generate/text/stream?prompt=' +
            encodeURIComponent(message);
        resetForm() ![4](img/4.png)

        source = new EventSource(url); ![5](img/5.png)
        source.addEventListener('open', handleOpen, false);
        source.addEventListener('message', handleMessage, false);
        source.addEventListener('error', handleClose, false); ![6](img/6.png)
    });

</script>
</body>
</html>
```

![1](img/1.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-1]

Crea un semplice input e un pulsante HTML per avviare le richieste SSE.

![2](img/2.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-2]

Crea un contenitore vuoto da utilizzare come lavandino per il contenuto dello stream.

![3](img/3.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-3]

Ascolta il pulsante `clicks` ed esegui il callback SSE.

![4](img/4.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-4]

恢复之前的内容模块和响应容器。

![5](img/5.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-5)

创建一个新的`EventSource`对象并监听连接状态的变化以处理事件。

![6](img/6.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO4-6)

当 SSE 连接打开时在控制台进行记录。它通过在响应容器中渲染消息内容来处理每个消息，直到接收到表示连接必须关闭的消息 `[DONE]`。此外，如果发生错误，它将关闭连接并在浏览器控制台中记录错误。

使用在 Esempio 6-4 中实现的 SSE 客户端，你现在可以用来测试你的 SSE 端点。然而，你必须首先提供 HTML。

创建一个名为`pages`的目录并将 HTML 文件放入其中。然后*挂载*该目录到你的 FastAPI 服务器上，以便将其内容作为静态文件提供服务，如 Esempio 6-5 所示。通过挂载，FastAPI 负责将每个文件的 API 路径映射，这样你就可以从与你的服务器相同的源通过浏览器访问它们。

##### Esempio 6-5. 将 HTML 文件挂载到服务器作为静态资源

```py
# main.py

from fastapi.staticfiles import StaticFiles

app.mount("/pages", StaticFiles(directory="pages"), name="pages") ![1](img/1.png)
```

![1](img/1.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO5-1)

将`pages`目录挂载到`/pages`以将其内容作为静态资源提供服务。挂载后，你可以通过访问`*<origin>*/pages/*<filename>`来访问每个文件。

通过实现 Esempio 6-5，你将从与你的 API 服务器相同的源提供 HTML，从而避免触发浏览器的 CORS 安全机制，这可能会阻止到达你的服务器的出站请求。

现在，你可以通过访问`http://localhost:8000/pages/sse-client.html`来访问 HTML 页面。

### 不同源之间的资源共享

如果你直接在你的浏览器中打开 Esempio 6-4 的 HTML 文件并点击启动流按钮，你会发现没有任何反应。你可以检查浏览器的网络标签来查看出站请求发生了什么。

经过一些调查后，你应该会注意到你的浏览器因为其与你的服务器进行的*跨源资源共享*（CORS）初步检查失败，而阻止了对服务器的出站请求。

CORS 是浏览器中实现的一种安全机制，用于控制网页资源可以被另一个域名请求的方式，并且仅在直接从浏览器发送请求而不是从服务器发送请求时相关。浏览器使用 CORS 来验证是否被授权从不同的源（即域名）向服务器发送请求。

例如，如果你的客户端托管在 `https://example.com` 上，并且需要从托管在 `https://api.example.com` 上的 API 恢复数据，除非 API 服务器启用了 CORS，否则浏览器将阻止该请求。

目前，你可以通过在你的服务器上添加 CORS 中间件来绕过这些 CORS 错误，就像在 Esempio 6-6 中看到的那样，以允许来自浏览器的任何请求。

##### 示例 6-6\. 应用 CORS 设置

```py
# main.py

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], ![1](img/1.png)
)
```

![1](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO6-1)

允许来自任何来源、方法（`GET`、`POST`等）和头信息的传入请求。

Streamlit 通过在其内部服务器上发送请求来避免激活 CORS 机制，即使生成的用户界面在浏览器上执行。

另一方面，FastAPI 的文档页面从同一服务器源执行请求（即 `http://localhost:8000`），因此默认情况下不会激活 CORS 安全机制。

###### 注意事项

在 Esempio 6-6 中，配置 CORS 中间件以处理任何传入的请求，实际上绕过了 CORS 安全机制以简化开发。在生产环境中，你应该允许你的服务器仅处理少量来源、方法和头信息。

如果你已经遵循了示例 6-5 或示例 6-6，现在你应该能够查看来自你的 SSE 端点的传入流（参见图 6-8）。

![bgai 0608](img/bgai_0608.png)

###### 图 6-8\. SSE 端点的传入流

恭喜！现在你有一个功能完善的解决方案，其中模型的响应会在数据生成后直接发送到你的客户。通过实现这个功能，你的用户在与你的聊天机器人交互时将获得更愉快的体验，因为他们会实时收到对他们的查询的响应。

你的解决方案还通过使用异步客户端与 Azure OpenAI 的 API 交互来实现了并发性，以便更快地向你的用户提供响应。你可以尝试使用同步客户端来比较生成速度的差异。使用异步客户端时，生成速度可能非常高，以至于你可能会一次性接收到一个文本块，即使实际上它被传输到浏览器。

### Hugging Face 模型 LLM 的结果流

现在你已经学会了如何使用 Azure OpenAI 等模型提供商实现 SSE 端点，你可能会想知道是否可以传输你之前从 Hugging Face 下载的开源模型输出。

尽管 Hugging Face 的`transformers`库实现了一个可以传递给模型管道的`TextStreamer`组件，但最简单的解决方案是运行一个独立的推理服务器，如 HF Inference Server，以实现模型的流式处理。

esempio 6-7 展示了如何使用 Docker 配置简单的模型推理服务器，并提供一个 `model-id` 网站。

##### 示例 6-7\. 通过 HF 推理服务器提供 HF LLM 模型服务

```py
$ docker run --runtime nvidia --gpus all \ ![1](img/1.png)
    -v ~/.cache/huggingface:/root/.cache/huggingface \ ![2](img/2.png)
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \ ![3](img/3.png)
    -p 8080:8000 \ ![4](img/4.png)
    --ipc=host \ ![5](img/5.png)
    vllm/vllm-openai:latest \ ![1](img/1.png) ![6](img/6.png)
    --model mistralai/Mistral-7B-v0.1 ![7](img/7.png)
```

![1](img/1.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-1)

使用 Docker 在所有可用的 NVIDIA GPU 上下载并运行最新版本的 `vllm/vllm-openai` 容器。

![2](img/2.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-2)

与容器共享一个卷以避免每次执行时都下载大量内容。

![3](img/3.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-3)

设置环境变量 secret 以访问受限访问的模型，例如 `mistralai/Mistral-7B-v0.1`。^(4)

![4](img/4.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-4)

在本地主机 `8080` 端口上运行推理服务器，将主机端口的 `8080` 映射到 Docker 容器暴露的端口 `8000`。

![5](img/5.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-5)

启用容器和主机之间的进程间通信 (IPC)，以便容器可以访问主机的共享内存。

![6](img/6.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-7)

vLLM 推理服务器使用 OpenAI 服务 LLM 的特定 API。

![7](img/7.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO7-8)

下载并使用 Hugging Face Hub 上的 `mistralai/Mistral-7B-v0.1` 网站。

当模型服务器运行时，你现在可以使用 `AsyncInferenceClient` 网站以流式格式生成输出，如 示例 6-8 所示。

##### 示例 6-8\. 从 HF 推理流中消费 LLM 输出流

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

虽然示例 l'Esempio 6-8 展示了如何使用 Hugging Face 的推理服务器，但你仍然可以使用其他支持模型流式响应的模型服务框架，例如 [vLLM](https://oreil.ly/LQAzF)。

在讨论 WebSocket 之前，让我们看看如何使用 `POST` 方法消费另一个 SSE 端点变体。

## SSE 与 POST 请求

[specifica`EventSource`](https://oreil.ly/61ovi) 规范规定服务器上的 `GET` 端点应正确消费传入的 SSE 流。这使得使用 SSE 实现实时应用程序变得简单，因为 `EventSource` 接口能够处理诸如连接丢失和自动重连等问题。

然而，使用 HTTP `GET` 请求有一些限制：与其它请求方法相比，`GET` 请求通常更不安全，更容易受到 *XSS* 攻击。^(5) 此外，由于 `GET` 请求不能有请求体，你只能将数据作为 URL 查询参数的一部分发送到服务器。问题是 URL 长度有限，你必须考虑这一点，并且所有查询参数都必须正确编码在请求 URL 中。因此，你不能简单地将整个会话历史添加到 URL 作为参数。你的服务器必须处理会话历史，并使用 `GET` 的 SSE 端点跟踪会话上下文。

解决这种限制的一个常见方法是在即使 SSE 规范不支持的情况下也实现一个 SSE `POST` 端点。因此，实现将更加复杂。

首先，我们在 示例 6-9 中实现了服务器上的 `POST` 端点。

##### 示例 6-9\. 在服务器上实现 SSE 端点

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

通过实现聊天输出流的 `POST` 端点，你现在可以开发客户端逻辑来处理 SSE 流。

你需要手动使用浏览器中的 `fetch` 网页接口处理传入的流，如 示例 6-10 所示。

##### 示例 6-10\. 使用浏览器 API `EventSource` 在客户端实现 SSE

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
                prompt: message, ![1](img/1.png)
            }),
        });

        const reader = response.body.getReader(); ![2](img/2.png)
        const decoder = new TextDecoder(); ![3](img/3.png)

        while (true) { ![4](img/4.png)
            const {value, done} = await reader.read();
            if (done) break;
            container.textContent += decoder.decode(value);
        }
    }

    button.addEventListener('click', async function() { ![5](img/5.png)
        resetForm()
        await stream(input.value)

    });

</script>
</body>
</html>
```

![1](img/1.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO8-1)

使用浏览器的 `fetch` 接口向后端发送 `POST` 请求。将请求体作为 JSON 字符串包含在请求中。添加头部以指定发送的请求体和期望从服务器收到的响应。

![2](img/2.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO8-2)

从响应体的流中访问 `reader`。

![3](img/3.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO8-3)

创建一个用于处理每条消息的文本解码器实例。

![4](img/4.png) (#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO8-4)

执行一个无限循环，并使用 `reader` 读取流中的下一个消息。如果流已终止，则 `done=true`，因此中断循环；否则，使用文本解码器解码消息并将其添加到 `textContent` 响应容器中以便进行渲染。

![5](img/5.png)

监听按钮的 `click` 事件以执行回调，该回调将恢复模块状态并使用提示与后端端点建立 SSE 连接。

如 示例 6-10 所示，在不使用 `EventSource` 参数的情况下消费 SSE 流可能会变得复杂。

###### 建议

另一种选择是使用 `GET` SSE 端点，但预先通过 `POST` 请求将大型有效载荷发送到服务器。服务器将存储数据并在建立 SSE 连接时使用它们。

SSE 也支持 cookie，因此你可以依赖 cookie 在 `GET` SSE 端点交换大型有效载荷。

如果你想在生产中使用 SSE 端点，你的解决方案还必须支持重试功能、错误处理，甚至中断连接的可能性。

示例 6-11 展示了如何在 JavaScript 中实现带有指数退避延迟的重试客户端功能。^(6)

##### 6-11 示例。客户端实现带指数退避的重试功能

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
    for (let attempt = 0; attempt < maxRetries; attempt++) { ![1](img/1.png)
        try { ![2](img/2.png)
            ... // Establish SSE connection here
            return ![3](img/3.png)
        } catch (error) {
            console.warn(`Failed to establish SSE connection: ${error}`);
            console.log(
                `Re-establishing connection - attempt number ${attempt + 1}`,
            );
            if (attempt < maxRetries - 1) {
                await sleep(delay); ![4](img/4.png)
                delay *= backoffFactor; ![5](img/5.png)
            } else {
                throw error ![6](img/6.png)
            }
        }
    }
}
```

![1](img/1.png)

在达到 `maxRetries` 之前，尝试建立 SSE 连接。计算每次尝试。

![2](img/2.png)

使用 `try` 和 `catch` 来处理连接错误。

![3](img/3.png)

成功时退出函数。

![4](img/4.png)

在重试之前暂停 `delay` 毫秒。

![5](img/5.png)

通过在每次迭代中将退避因子乘以延迟值来实现指数退避。

![6](img/6.png)

当达到 `maxRetries` 时抛出 `error`。

现在，你应该更自在地实现你的 SSE 端点以进行模型响应的流式传输。SSE 是 ChatGPT 等应用程序用于与模型进行实时对话的通信机制。由于 SSE 主要支持基于文本的流，它非常适合 LLM 输出流式传输的场景。

在下一节中，我们将使用 WebSocket 机制实现相同的解决方案，以便比较实现细节的差异。此外，你将了解为什么 WebSocket 对于需要实时双工通信的场景（如现场转录服务）是理想的。

# 实现 WS 端点

在本节中，你将使用 WebSocket 协议实现一个端点。通过此端点，你将使用 WebSocket 将 LLM 的输出传输给客户端，以便与 SSE 连接进行比较。最后，你将学习 SSE 和 WebSocket 在实时 LLM 输出流式传输中的异同。

## 使用 WebSocket 进行 LLM 输出流式传输

FastAPI 通过使用 Starlette 框架的`WebSocket`接口支持 WebSocket。由于 WebSocket 连接需要管理，我们首先实现一个连接管理器来跟踪活动连接及其状态。

你可以按照示例 6-12 实现一个 WebSocket 连接管理器。

##### 示例 6-12\. WebSocket 连接管理器的实现

```py
# stream.py

from fastapi.websockets import WebSocket

class WSConnectionManager: ![1](img/1.png)
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None: ![2](img/2.png)
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None: ![3](img/3.png)
        self.active_connections.remove(websocket)
        await websocket.close()

    @staticmethod
    async def receive(websocket: WebSocket) -> str: ![4](img/4.png)
        return await websocket.receive_text()

    @staticmethod
    async def send(
        message: str | bytes | list | dict, websocket: WebSocket
    ) -> None: ![5](img/5.png)
        if isinstance(message, str):
            await websocket.send_text(message)
        elif isinstance(message, bytes):
            await websocket.send_bytes(message)
        else:
            await websocket.send_json(message)

ws_manager = WSConnectionManager() ![6](img/6.png)
```

![1](img/1.png)[(#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-1)]

创建一个`WSConnectionManager`来跟踪和管理活动的 WS 连接。

![2](img/2.png)[(#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-2)]

使用`accept()`方法打开 WebSocket 连接。将新连接添加到活动连接列表中。

![3](img/3.png)[(#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-3)]

当你断开连接时，关闭连接并从活动连接列表中移除`websocket`实例。

![4](img/4.png)[(#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-4)]

在打开的连接期间接收文本形式的传入消息。

![5](img/5.png)[(#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-5)]

使用相关发送方法向客户端发送消息。

![6](img/6.png)[(#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO10-6)]

创建一个可重用于整个应用的`WSConnectionManager`单例。

你还可以扩展 Esempio 6-12 中的连接管理器以*传输*消息（例如，警告、通知或实时系统更新）给所有连接的客户端。这在群聊或协作白板和文档编辑工具等应用中非常有用。

由于连接管理器通过`active_connections`列表维护每个客户端的指针，你可以向每个客户端发送消息，如 Esempio 6-13 中所示。

##### 示例 6-13\. 使用 WebSocket 管理器向连接的客户端传输消息

```py
# stream.py

from fastapi.websockets import WebSocket

class WSConnectionManager:
    ...
    async def broadcast(self, message: str | bytes | list | dict) -> None:
        for connection in self.active_connections:
            await self.send(message, connection)
```

通过实现 WebSocket 管理器，现在你可以开发一个 WebSocket 端点以向客户端传输响应。然而，在实现端点之前，遵循 Esempio 6-14 以更新`chat_stream`方法，使其产生适合 WebSocket 连接的流内容。

##### 示例 6-14\. 更新聊天客户端的流式传输方法以生成适合 WebSocket 连接的内容

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
            if chunk.choices[0].delta.content is not None: ![1](img/1.png)
                yield (
                    f"data: {chunk.choices[0].delta.content}\n\n"
                    if mode == "sse"
                    else chunk.choices[0].delta.content ![2](img/2.png)
                )
                await asyncio.sleep(0.05)
        if mode == "sse": ![2](img/2.png)
            yield f"data: [DONE]\n\n"
```

![1](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO11-1)

仅返回非空内容。

![2](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO11-2)

根据连接类型（SSE 或 WS）返回流的内容。

更新`stream_chat`方法后，你可以专注于添加 WebSocket 端点。使用`@app.websocket`装饰器装饰控制器中的函数，该函数使用 FastAPI 的`WebSocket`类，如 esempio 6-15 中所示。

##### 示例 6-15\. 实现 WS 端点

```py
# main.py

import asyncio
from loguru import logger
from fastapi.websockets import WebSocket, WebSocketDisconnect
from stream import ws_manager, azure_chat_client

@app.websocket("/generate/text/streams") ![1](img/1.png)
async def websocket_endpoint(websocket: WebSocket) -> None:
    logger.info("Connecting to client....")
    await ws_manager.connect(websocket) ![2](img/2.png)
    try: ![3](img/3.png)
        while True: ![4](img/4.png)
            prompt = await ws_manager.receive(websocket) ![5](img/5.png)
            async for chunk in azure_chat_client.chat_stream(prompt, "ws"):
                await ws_manager.send(chunk, websocket) ![6](img/6.png)
                await asyncio.sleep(0.05) ![7](img/7.png)
    except WebSocketDisconnect: ![8](img/8.png)
        logger.info("Client disconnected")
    except Exception as e: ![9](img/9.png)
        logger.error(f"Error with the WebSocket connection: {e}")
        await ws_manager.send("An internal server error has occurred")
    finally:
        await ws_manager.disconnect(websocket) ![10](img/10.png)
```

![1](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-1)

创建一个可访问的 WebSocket 端点，地址为`ws://localhost:8000/generate/text/stream`。

![2](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-2)

打开客户端和服务器之间的 WebSocket 连接。

![3](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-3)

只要连接打开，就继续发送或接收消息。

![4](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-4)

在`websocket_controller`内部处理错误并记录重要事件，以识别错误原因并以优雅的方式处理意外情况。当服务器或客户端关闭连接时，中断无限循环。

![5](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-5)

当收到第一条消息时，将其作为提示传递给 OpenAI API。

![6](img/6.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-6]

异步迭代生成的聊天流，并将每个片段发送到客户端。

![7](img/7.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-7]

在发送下一个消息之前等待一小段时间，以减少竞争条件问题，并允许客户端有足够的时间处理流。

![8](img/8.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-8]

当客户端关闭 WebSocket 连接时，会引发 `WebSocketDisconnect` 异常。

![9](img/9.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-9]

如果在打开连接期间服务器端发生错误，则记录错误并识别客户。

![10](img/10.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO12-10]

如果流已终止、发生内部错误或客户端关闭连接，则中断无限循环并优雅地关闭 WebSocket 连接。从活动的 WebSocket 连接列表中删除连接。

现在你已经有一个 WebSocket 端点，我们来开发客户端的 HTML 以测试该端点（参见 示例 6-16）。

##### 示例 6-16\. 客户端 WebSocket 连接实现，包括错误处理和指数退避重试功能

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
        ws = new WebSocket("ws://localhost:8000/generate/text/streams"); ![1](img/1.png)

        ws.onopen = handleOpen;
        ws.onmessage = handleMessage;
        ws.onclose = handleClose;
        ws.onerror = handleError; ![2](img/2.png)
    }

    function handleOpen(){
        console.log("WebSocket connection opened");
        retryCount = 0;
        isError = false;
    }

    function handleMessage(event) {
        container.textContent += event.data;
    }

    async function handleClose(){ ![3](img/3.png)
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

    streamButton.addEventListener('click', function() { ![4](img/4.png)
        const prompt = document.getElementById("messageInput").value;
        if (prompt && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(prompt); ![5](img/5.png)
        }
        resetForm(); ![6](img/6.png)
    });

    closeButton.addEventListener('click', function() { ![7](img/7.png)
        isError = false;
        if (ws) {
            ws.close();
        }
    });

    connectWebSocket(); ![1](img/1.png)
</script>
</body>
</html>
```

![1](img/1.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-1]

与 FastAPI 服务器建立 WebSocket 连接。

![2](img/2.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-2]

向 WebSocket 连接实例添加回调处理程序以处理打开、关闭、消息和错误事件。

![3](img/3.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-3]

使用 `isError` 标志优雅地处理连接错误并使用指数退避重试功能重新建立连接。

![4](img/4.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-4]

向流按钮添加事件监听器，以将第一条消息发送到服务器。

![5](img/5.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-5]

一旦建立连接，将非空提示信息作为第一条消息发送到服务器。

![6](img/6.png)[#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-6]

在建立要启动的 WebSocket 连接之前重置模块。

![7](img/#co_real_time_communication___span_class__keep_together__with_generative_models__span__CO13-7)

Aggiungi un ascoltatore di eventi al pulsante di chiusura della connessione per chiudere la connessione quando il pulsante viene cliccato.

Ora puoi visitare [*http://localhost:8000/pages/client-ws.html*](http://localhost:8000/pages/client-ws.html) per testare il tuo endpoint di streaming WebSocket (vedi Figura 6-9).

![bgai 0609](img/bgai_0609.png)

###### Figura 6-9\. Flusso in entrata dall'endpoint WebSocket

Ora dovresti avere un'applicazione di streaming LLM completamente funzionante con WebSocket. Ben fatto!

A questo punto ti starai chiedendo quale sia la soluzione migliore: lo streaming con SSE o le connessioni WS. La risposta dipende dai requisiti della tua applicazione. SSE è semplice da implementare ed è nativo del protocollo HTTP, quindi la maggior parte dei client lo supporta. Se hai bisogno solo di uno streaming unidirezionale verso il client, allora ti consiglio di implementare le connessioni SSE per lo streaming degli output di LLM.

Le connessioni WebSocket forniscono un maggiore controllo al meccanismo di streaming e consentono una comunicazione duplex all'interno della stessa connessione, ad esempio nelle applicazioni di chat in tempo reale con più utenti e nei servizi LLM, speech-to-text, text-to-speech e speech-to-speech. Tuttavia, l'utilizzo di WebSocket richiede l'aggiornamento della connessione da HTTP al protocollo WebSocket, che i client legacy e i browser più vecchi potrebbero non supportare. Inoltre, dovrai gestire le eccezioni in modo leggermente diverso con gli endpoint WebSocket.

## Gestire le eccezioni WebSocket

La gestione delle eccezioni WebSocket è diversa da quella delle connessioni HTTP tradizionali. Se fai riferimento all'Esempio 6-15, noterai che non stai più restituendo al cliente una risposta con codici di stato, o `HTTPExceptions`, ma piuttosto stai mantenendo una connessione aperta dopo l'accettazione della connessione.

Finché la connessione è aperta, puoi inviare e ricevere messaggi. Tuttavia, non appena si verifica un'eccezione, devi gestirla chiudendo con grazia la connessione e/o inviando un messaggio di errore al client in sostituzione della risposta `HTTPException`.

Poiché il protocollo WebSocket non supporta i consueti codici di stato HTTP (`4xx` o `5xx`), non puoi usare i codici di stato per notificare ai client i problemi del lato server. Al contrario, dovresti inviare messaggi WebSocket ai client per notificare loro i problemi prima di chiudere qualsiasi connessione attiva dal server.

Durante la chiusura della connessione, puoi utilizzare diversi codici di stato relativi a WebSocket per specificare il motivo della chiusura. Utilizzando questi motivi di chiusura, puoi implementare qualsiasi comportamento di chiusura personalizzato sul server o sui client.

表 6-2 展示了一些可以通过 `CLOSE` 帧发送的常见状态码。

表 6-2\. WebSocket 协议常见状态码

| 状态码 | 描述 |
| --- | --- |
| 1000 | 正常关闭 |
| 1001 | 客户端已断开连接或服务器已关闭 |
| 1002 | 一个端点（例如，客户端或服务器）接收到了违反 WS 协议的数据（例如，未掩码的包，无效的有效载荷长度）。 |
| 1003 | 一个端点接收到了不支持的数据（例如，期望接收文本却收到了二进制文件）。 |
| 1007 | 一个端点接收到了编码不一致的数据（例如，文本消息中的非 UTF-8 数据）。 |
| 1008 | 一个端点接收到了违反其策略的消息；可以用于隐藏关闭细节以保障安全。 |
| 1011 | 服务器内部错误 |

你可以在 WebSocket 协议的[第 7.4 节](https://oreil.ly/1L_HH)中找到有关其他状态码的更多信息[RFC 6455](https://oreil.ly/1L_HH)。

## 设计用于流式的 API

现在你已经对 SSE 和 WebSocket 的实现有了更多的熟悉，我想着重讨论一个关于它们设计架构的最后一个重要细节。

在设计流式 API 时，一个常见的陷阱是暴露过多的流式端点。例如，如果你正在开发一个聊天机器人应用，你可能需要暴露多个流式端点，每个端点预先配置以处理单个会话中的不同消息。使用这种特定的 API 设计模式，你需要让客户从一个端点切换到另一个端点，在每个步骤中提供必要的信息，并在单个会话中导航流式连接。这种设计模式增加了后端和前端应用程序的复杂性，因为会话状态需要由双方管理，以避免组件之间的竞争条件和网络问题。

一个更简单的 API 设计模型是提供一个单一的入口点供客户端启动与您的 GenAI 模型的流，并使用头部、请求体或查询参数来激活后端的相关逻辑。在这种设计中，后端逻辑从客户端抽象出来，简化了前端的状态管理，而所有路由和业务逻辑都实现在后端。由于后端可以访问数据库、其他服务和自定义提示，它可以轻松执行 CRUD 操作，并在提示之间或模型之间切换以计算响应。因此，一个端点可以作为一个唯一的入口点用于切换逻辑，管理应用程序状态并生成定制响应。

# 摘要

本章讨论了多种策略，用于在您的 GenAI 服务中通过数据流实现实时通信。

你已经学习了多种 Web 通信机制，包括传统的 HTTP 请求-响应模型、短轮询/长轮询、SSE 和 WebSocket。然后你详细比较了这些机制，以理解它们的特性、优点、缺点和用例，特别是对于人工智能工作流程。最后，你使用 Azure OpenAI 的异步客户端实现了两个 LLM 流式传输端点，以学习如何利用实时通信机制 SSE 和 WebSocket。 

在下一章中，你将更好地了解在集成人工智能服务数据库时开发 API 的工作流程，包括如何设置、迁移和与数据库交互。你还将学习如何使用 FastAPI 的背景任务来管理端点中的数据存储和检索操作。

下一章将讨论数据库设置和模式设计、使用 SQLAlchemy 进行操作、数据库迁移以及模型输出流中的数据库操作管理。

^(1) 攻击者可以使用缓存中毒来向缓存系统注入有害数据，然后向用户或系统提供错误数据。为了防止这种攻击，客户端和服务器在发送之前将有效载荷伪装成随机数据。

^(2) 这些攻击包括通过向 WebSocket 帧发送 HTTP 响应来欺骗服务器泄露敏感信息。

^(3) 有关`EventSource`接口的更多详细信息，请参阅 MDN 资源。

^(4) 按照指南“访问私有/保证模型”(https://oreil.ly/a7KeV)生成 Hugging Face 用户的访问令牌。

^(5) 攻击者利用 XSS 漏洞在网页中插入有害脚本，这些脚本随后由其他用户的浏览器执行。

^(6) 指数退避通过在每次尝试后增加延迟来降低 API 速率限制错误的概率。
