# 1 理解大型语言模型

本章节涵盖

+   介绍生成式 AI（特别是大型语言模型）

+   探索生成式 AI 的好处

+   确定何时以及何时不使用生成式 AI

无论你是否意识到，无论你是否愿意承认，你都已经悄然晋升了。每个专业软件工程师都是如此。几乎一夜之间，我们已从职员工程师变成了工程经理。现在，你有了世界上最聪明和最有才华的初级开发者——生成式 AI 是你的新编码伙伴。因此，指导、辅导和执行代码审查应该成为你日常工作的部分。本章将为你概述一个名为大型语言模型（LLMs）的生成式 AI 子集，特别是 ChatGPT、GitHub Copilot 和 AWS CodeWhisperer。

注意：这不是一本传统的编程书籍。你无法像使用脚本一样使用它。你将与 LLM 进行对话，就像任何对话一样，语言和方向会根据模型和先前的上下文而变化。你收到的输出很可能与本书中打印的内容不同。这不应该让你气馁。相反，你应该去探索。旅程和目的地一样有回报。你可能会发现自己无法跟上。要有耐心。如果你自律（并且有些冒险精神），你可以让 GPT 与本书的主题和目标合作：学习如何使用生成式 AI 成为更好的程序员。

## 1.1 加速你的开发

欢迎来到软件开发的新时代，你的开发团队增加了一位非常出色的工程师。生成式 AI 不仅仅是一个工具；它是你的下一个团队成员，准备将你的编程提升到新的高度。想象一下设计复杂的系统，以前所未有的速度编码，以及用你从未想过可能的方式测试——所有这些都有一种从最佳学习到的智能。在这本书中，我们将探讨生成式 AI 不仅会帮助你完成日常编码任务，还能让你实现以前无法触及的壮举，确保更快的开发速度、更高的质量，以及前所未有的创新能力。

LLMs 和生成式 AI 可以极大地加速你的软件开发过程。通过自动化繁琐的样板代码创建，你可以专注于创意问题解决和价值增加的任务。考虑以下例子：你希望 LLM 设置一个 Flask 应用。你可能使用以下提示：

|

![logo-NC](img/logo-NC.png)

| 生成一个 Flask 应用的骨架。 |
| --- |

LLM 可能会输出以下 Python 代码。

列表 1.1 由 LLM 生成的用于构建 Flask 应用的潜在代码

```py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, world!"

if __name__ == '__main__':
    app.run(debug=True)
```

LLMs 是擅长对自然语言提示提供自然语言响应的深度学习模型。你可以想象简单地用普通英语描述你需要的内容，并收到可以立即集成的代码：

|

![](img/logo-NC.png)

| 在 Python 中创建用于管理用户会话的类。 |
| --- |

这里是一个可能的响应。

列表 1.2 生成用于创建会话管理类的潜在代码

```py
import uuid
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, user_id):
        """Create a new session for a user with a unique session ID and expiration time."""
        session_id = str(uuid.uuid4())
        # Sessions expire after 1 hour
        expiration = datetime.now() + timedelta(hours=1)  
        self.sessions[session_id] = {'user_id': 
        user_id, 'expiration': expiration}
        return session_id

    def get_user_id(self, session_id):
        """Retrieve a user ID from a session ID 
               if the session is still valid."""
        session_info = self.sessions.get(session_id)
        if session_info and session_info['expiration'] > datetime.now():
            return session_info['user_id']
        return None  # Session is expired or does not exist

    def delete_session(self, session_id):
        """Delete a session by its ID."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def is_session_valid(self, session_id):
        """Check if a session is still valid."""
        return session_id in self. sessions and self.sessions[session_id]['expiration'] > datetime.now()
```

这些工具通过识别模式和提出修复建议来提高你的调试效率。假设你希望你的 AI 工具分析一段代码并标记潜在的内存泄漏。它可能会生成以下代码。

列表 1.3 使用 LLM 在 Python 中查找潜在的内存泄漏

```py
    def process_data():
    large_data = [x for x in range(1000000)]  # Large list of numbers
    result = sum(large_data)
    print("Result of processing:", result)
    # large_data not cleared or reassigned, 
    # potentially leading to memory retention

process_data()
```

根据 ChatGPT 提供的解释，这段 Python 代码演示了由于未清除使用后的大列表而导致的潜在内存泄漏，如果不正确管理引用，其内容可能会在内存中保留比必要更长的时间。

在改进的代码中，AI 工具建议在使用后明确清除列表或重新分配`None`，以帮助更快地释放内存，尤其是在垃圾回收不是非常积极的环境中。

列表 1.4 “改进”的 LLM 代码以解决潜在的内存泄漏

```py
def process_data():
    large_data = [x for x in range(1000000)]  # Large list of numbers
    result = sum(large_data)
    print("Result of processing:", result)
    large_data = None  # Clear the reference to release memory

process_data()
```

此外，当涉及到重构时，AI 可以提出优化建议，使你的代码更干净、更高效，如以下两个列表所示。

列表 1.5 在建议重构之前的冗长代码

```py
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process_data(self):
        if self. data is not None:
            if len(self.data) > 0:
                processed_data = []
                for d in self.data:
                    if d is not None:
                        if d % 2 == 0:
                            processed_data.append(d)
                return processed_data
            else:
                return []
        else:
            return []

processor = DataProcessor([1, 2, 3, 4, None, 6])
result = processor.process_data()
print("Processed Data:", result)
```

重构后，代码更加易于阅读、维护和符合惯例。

列表 1.6 LLM 重构后的代码，更加简洁

```py
class DataProcessor:
    def __init__(self, data):
        self. data = data or []

    def process_data(self):
        return [d for d in self.data if d is not None and d % 2 == 0]

processor = DataProcessor([1, 2, 3, 4, None, 6])
result = processor.process_data()
print("Processed Data:", result)
```

LLM 的功能不仅限于代码生成；它们足够复杂，可以协助设计软件架构。这种能力允许开发者以更具创造性和战略性的方式与这些模型互动。例如，开发者可以描述系统的整体目标或功能需求，而不是简单地请求特定的代码片段。然后，LLM 可以提出各种架构设计、建议设计模式或概述整个系统的结构。这种方法不仅节省了大量时间，而且利用了 AI 的广泛训练来创新和优化解决方案，可能引入效率或想法，这些是开发者最初可能没有考虑到的。这种灵活性使 LLM 成为软件开发创意和迭代过程中的宝贵伙伴。我们将在第三章中探讨这一点。

此外，通过提高你的交付成果的质量和安全，从代码到文档，这些工具确保你的输出达到最高标准。例如，在集成新库时，AI 可以自动生成安全、高效的实现示例，帮助你避免常见的安全陷阱。

最后，学习新的编程语言或框架变得显著更容易。AI 可以提供实时、上下文感知的指导和文档，帮助您不仅理解，而且实际应用新概念。例如，您是否正在过渡到新的框架如 Dash？您的 AI 助手可以立即生成针对您当前项目上下文的示例代码片段和详细说明。

列表 1.7：LLM 生成的示例代码，展示如何使用库

```py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Sample data creation
dates = pd.date_range(start='1/1/2020', periods=100)
prices = pd.Series(range(100)) + pd.Series(range(100))/2  
# Just a simple series to mimic stock prices
data = pd.DataFrame({'Date': dates, 'Price': prices})

# Initialize the Dash app (typically in your main module)
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Stock Prices Dashboard"),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=data['Date'].min(),
        end_date=data['Date'].max(),
        display_format='MMM D, YYYY',
        start_date_placeholder_text='Start Period',
        end_date_placeholder_text='End Period'
    ),
    dcc.Graph(id='price-graph'),
])

# Callback to update the graph based on the date range picker input
@app.callback(
    Output('price-graph', 'figure'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_graph(start_date, end_date):
    filtered_data = data[(data['Date'] >= 
            start_date) & (data['Date'] <= end_date)]
    figure = px.line(filtered_data, x='Date', 
            y='Price', title='Stock Prices Over Time')
    return figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
```

我们可以在图 1.1 中看到这段代码的输出，这是正在运行的 Dash 代码。

![图片](img/CH01_F01_Crocker2.png)

图 1.1 ChatGPT 根据提示“`使用 dash 创建一个示例仪表板`”创建的股票价格仪表板

LLMs 的真实力量在于它们在开发环境中的集成。例如，由微软开发的 GitHub Copilot 工具，利用 LLMs 的能力，在 Visual Studio Code 等集成开发环境（IDE）中提供实时编码辅助。我们将在第四章中展示这一功能。

本书不仅会解释这些概念，还会通过众多示例进行演示，展示您如何使用 LLMs 显著提高生产力和代码质量。从设置您的环境到解决复杂的编码挑战，您将学习如何充分利用这些智能工具在日常开发中的使用。

## 1.2 开发者对 LLMs 的介绍

尽管这本书主要是一本实践指南，因此理论部分相对较少，但以下部分将为您提供最相关的材料，帮助您充分利用您的新队友。

是的，但我还想了解更多

如果您对 LLMs、神经网络以及所有生成式 AI 背后的理论感兴趣，您应该查看以下两本书：Sebastian Raschka（Manning，2024）即将出版的《从零开始构建大型语言模型》（Build a Large Language Model (From Scratch)）和 David Clinton（Manning，2024）幽默命名的《生成式 AI 完全过时指南》（The Complete Obsolete Guide to Generative AI）。

让我们从一个非常简单的定义开始，了解 LLM 是什么以及它能为您做什么；这样，您就可以正确地向您的老板和同事介绍它。*大型语言模型*是一种人工智能模型，它根据训练数据处理、理解和生成类似人类的文本。这些模型是深度学习的一个子集，在处理自然语言处理（NLP）的各个方面特别先进。

正如其名所示，这些模型不仅在训练数据的数据量上“大”，而且在复杂性和参数数量上也非常大。现代 LLMs，如 OpenAI 的 GPT-4，拥有高达数百亿个参数。

LLMs（大型语言模型）是在大量文本数据上训练的。这种训练包括阅读和分析各种互联网文本、书籍、文章和其他形式的书面沟通，以学习人类语言的结构、细微差别和复杂性。

大多数 LLM 使用 Transformer 架构，这是一种依赖自注意力机制的深度学习模型，它可以根据不同单词在句子中的位置来权衡其重要性。这使得 LLM 能够生成更多上下文相关的文本。典型的 Transformer 模型由一个编码器和一个解码器组成，每个都由多个层组成。

理解 LLMs 的架构有助于更有效地使用它们的特性，并在实际应用中解决它们的局限性。随着这些模型不断进化，它们承诺将为开发者提供更高级的工具，以增强他们的应用程序。

## 1.3 何时使用和何时避免生成式 AI

生成式 AI（以及由此扩展的 LLM）并非万能的解决方案。了解何时使用这些技术，以及识别它们可能不太有效或甚至有问题的情况，对于最大化其好处同时减轻潜在缺点至关重要。我们将从何时适合使用 LLM 开始：

+   提高生产力

    +   *示例*—使用 AI 自动化样板代码、生成文档或在您的 IDE 中提供编码建议。

    +   *第三章和第四章讨论*—这些章节探讨了 GitHub Copilot 等工具如何提高编码效率。

+   学习和探索

    +   *示例*—利用 AI 通过生成示例代码和解释来学习新的编程语言或框架。

    +   *第五章涵盖*—在这里，我们检查 AI 如何加速学习过程，并介绍您了解新技术。

+   处理重复性任务

    +   *示例*—使用 AI 处理重复的软件测试或数据录入任务，从而腾出时间解决更复杂的问题。

    +   *第七章探讨*—讨论测试和维护任务中的自动化。

然而，有些情况下你应该避免使用 LLMs 和生成式 AI 工具，如 ChatGPT 和 GitHub Copilot，主要与数据安全和隐私保护相关。在包含敏感或专有数据的环境中使用 AI 可能会造成意外的数据泄露。这有几个原因，其中之一是部分或全部代码作为上下文发送到模型中，这意味着至少部分专有代码可能会绕过你的防火墙。还有一个问题是它是否可能被包含在下一轮训练的训练数据中。但请放心：我们将在第九章中探讨一些解决这一担忧的方法。

你可能限制使用场景的另一个例子是当需要精确和专业性时。鉴于大型语言模型的一个特点是它们能够在其输出中添加随机性（有时被称为*幻觉*），输出可能包含与真实和正确答案细微的差别。因此，在将其包含在代码库之前，你应该始终验证输出。

尽管生成式 AI 提供了众多优势，但应用它时必须谨慎，考虑到其使用的上下文和项目的具体需求。通过理解何时使用这些强大的工具以及何时需要谨慎行事，开发者可以最大化其效果，并确保技术的道德和高效使用。

## 摘要

+   生成式 AI 既是进化性的也是革命性的。就其是开发者每天使用的工具的另一个迭代而言，它是进化性的。就其将改变我们工作方式而言，它是革命性的。

+   开发的未来将涉及管理生成式 AI。即使是传说中的 10 倍开发者，也不会有与 AI 合作伙伴的开发者相同的生产力；AI 赋能的开发者将以更快的速度、更低的成本生产出高质量的代码，比没有 AI 的开发者要低。我们将花费更多的时间来训练我们的 AI 合作伙伴去做我们想要的事情以及我们想要的方式，而不是在没有 AI 的情况下编写代码。

+   信任但验证 LLM 的输出。
