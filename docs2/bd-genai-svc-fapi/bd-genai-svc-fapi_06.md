# 第四章\. 实现类型安全的 AI 服务

当与由多个贡献者持续更改的复杂代码库一起工作，并且与外部服务（如 API 或数据库）交互时，您将希望遵循最佳实践，例如在构建应用程序时确保类型安全。

本章重点介绍在构建后端服务和 API 时类型安全的重要性。您将学习如何使用 Python 的内置数据类和 Pydantic 数据模型来实现类型安全，并了解它们的相似之处和不同之处。此外，您将探索如何使用自定义验证器与 Pydantic 数据模型一起使用，以防止不良用户输入或数据错误，并学习如何使用 Pydantic 设置来加载和验证环境变量。最后，您将发现处理外部系统模式更改的策略以及管理不断发展的代码库中的复杂性的策略，以防止错误。

到本章结束时，您将拥有一个完全类型化的 GenAI 服务，在处理更改、不良用户输入和不一致模型响应时，该服务更不容易出现错误。

要跟上进度，您可以通过切换到[`ch04-start`分支](https://github.com/Ali-Parandeh/building-generative-ai-services/tree/ch04-start)来找到本章的起始代码。

# 类型安全简介

编程中的*类型*指定可以分配给变量的值以及可以在这些变量上执行的操作。

在 Python 中，常见类型包括以下内容：

整数

表示整数

浮点数

表示有分数部分的数字

字符串

表示字符序列

布尔值

表示`True`或`False`值

###### 小贴士

您可以使用`typing`包导入特殊类型，正如您在第三章中的其他代码示例中看到的那样。

*类型安全*是一种编程实践，确保变量只分配与其定义类型兼容的值。在 Python 中，您可以使用类型来检查代码库中变量的使用情况，特别是如果代码库在复杂性和大小方面增长。然后，类型检查工具（例如，`mypy`）可以使用这些类型来捕获错误的变量分配或操作。

您可以通过声明完全类型化的变量和函数来强制类型约束，如示例 4-1 所示。

##### 示例 4-1\. 在 Python 中使用类型

```py
from datetime import datetime

def timestamp_to_isostring(date: int) -> str:
    return datetime.fromtimestamp(date).isoformat()

print(timestamp_to_isostring(1736680773))
# 2025-01-12T11:19:52.876758

print(timestamp_to_isostring("27 Jan 2025 14:48:00"))
# error: Argument 1 to "timestamp_to_isostring" has incompatible type "str";
# expected "int" [arg-type]
```

代码编辑器和 IDE（例如，VS Code 或 JetBrains PyCharm）也可以使用类型检查扩展，如图 4-1 所示，在编写代码时发出类型违规警告。

![bgai 0401](img/bgai_0401.png)

###### 图 4-1\. 在 VS Code `mypy`扩展中捕获类型错误

在复杂的代码库中，很容易失去对变量、它们的状态和不断变化的模式的跟踪。例如，您可能会忘记`timestamp_to_isostring`函数接受数字作为输入，并错误地将时间戳作为字符串传递，如图 4-1 所示。

当包维护者或外部 API 提供商更新他们的代码时，类型也非常有用。类型检查器可以立即发出警告，帮助你在开发过程中处理这些更改。这样，你将立即被引导到潜在错误源，而无需运行你的代码并测试每个端点。因此，类型安全实践可以通过早期检测节省你的时间，并防止你处理更难以捉摸的运行时错误。

最后，你可以更进一步，在你的部署管道中设置自动类型检查，以防止将破坏性的更改推送到生产环境。

类型安全一开始可能感觉像是一种负担。你必须明确地为每个你编写的函数指定类型，这可能会很麻烦，并会在开发的初期阶段减慢你的速度。

有些人为了快速原型设计和编写更少的样板代码而跳过代码的打字。这种方法更加灵活，更容易使用，而且 Python 足够强大，可以推断简单的类型。此外，一些代码模式（例如具有多类型参数的函数）可能非常动态，因此在实验阶段更容易避免实施严格的类型安全。然而，随着你的服务不可避免地变得复杂并持续变化，这将会为你节省数小时的开发时间。

好消息是，在使用数据库时，一些类型可以使用 Prisma 等工具自动生成，当与外部 API 一起工作时，可以使用客户端生成器。对于外部 API，你通常可以找到包含具有类型提示（即完全类型化的客户端）的官方 SDK，这些类型提示指定了使用 API 时输入和输出的预期类型。如果没有，你可以检查 API 以创建自己的完全类型化的客户端。我将在本书的后面更详细地介绍 Prisma 和 API 客户端生成器。

当你不使用类型时，你可能会暴露于各种可能发生的错误和问题，这些问题可能是因为其他开发人员意外更新了你的服务交互的数据库表或 API 模式。在其他情况下，你可能会更新数据库表——例如删除一个列，并忘记更新与该表交互的代码。

没有类型，你可能永远不会注意到由于更新而导致的破坏性更改。这可能很难调试，因为未处理的下游错误可能无法确定损坏的组件或你自己的开发团队周围未处理的边缘案例的一般问题。结果，可能只需一分钟就能解决的问题可能需要半天甚至更长的时间。

你可以通过广泛的测试始终预防一些生产中的灾难。然而，如果你从一开始就使用类型，那么避免集成和可靠性问题会容易得多。

# 培养良好的编程习惯

如果你以前没有为代码打字，现在开始养成为所有变量、函数参数和返回类型打字的习惯永远不会太晚。

使用类型可以使您的代码更易于阅读，帮助您在早期捕获错误，并在重新访问复杂的代码库时节省大量时间，以便快速理解数据流。

# 实现类型安全

自 Python 3.5 以来，您可以显式声明变量的类型、函数参数和返回值。允许您声明这些类型的语法是 *类型注解*。

## 类型注解

类型注解不会影响应用程序的运行时行为。它们有助于捕获类型错误，尤其是在复杂的大型应用程序中，多人协作时尤为重要。静态类型检查工具，如 `mypy`、`pyright` 或 `pyre`，以及代码编辑器，可以验证从函数存储和返回的数据类型是否与预期类型匹配。

在 Python 应用程序中，类型注解用于：

+   *代码编辑器自动完成支持*

+   *使用 `mypy` 等工具进行静态类型检查*

FastAPI 还利用类型提示来完成：

+   *定义处理程序要求* 包括路径和查询参数、主体、头和依赖项等。

+   需要时*转换数据*

+   *验证来自传入请求、数据库和外部服务的数据*

+   *自动更新生成文档页面的 OpenAPI 规范*

您可以使用 `pip` 安装 `loguru`:

```py
$ pip install loguru
```

示例 4-2 展示了几个类型注解的示例。

##### 示例 4-2\. 使用类型注解来减少代码变更时出现的未来错误

```py
# utils.py

from typing import Literal, TypeAlias
from loguru import logger
import tiktoken

SupportedModels: TypeAlias = Literal["gpt-3.5", "gpt-4"]
PriceTable: TypeAlias = dict[SupportedModels, float] ![1](img/1.png) ![2](img/2.png)
price_table: PriceTable = {"gpt-3.5": 0.0030, "gpt-4": 0.0200} ![3](img/3.png)

def count_tokens(text: str | None) -> int: ![4](img/4.png)
    if text is None:
        logger.warning("Response is None. Assuming 0 tokens used")
        return 0 ![5](img/5.png)
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text)) ![6](img/6.png)

def calculate_usage_costs(
    prompt: str,
    response: str | None,
    model: SupportedModels,
) -> tuple[float, float, float]: ![7](img/7.png)
    if model not in price_table:
        # raise at runtime - in case someone ignores type errors
        raise ValueError(f"Cost calculation is not supported for {model} model.") ![8](img/8.png)
    price = price_table[model] ![9](img/9.png)
    req_costs = price * count_tokens(prompt) / 1000
    res_costs = price * count_tokens(response) / 1000 ![10](img/10.png)
    total_costs = req_costs + res_costs
    return req_costs, res_costs, total_costs ![11](img/11.png)
```

![1](img/#co_implementing_type_safe_ai_services_CO1-1)

使用 Python 的 `typing` 模块中的 `Literal`，它包含在其标准库中.^(1) 声明字面量 `gpt-3.5` 和 `gpt-4` 并将它们分配给 `SupportedModel` *类型别名*。`PriceTable` 也是一个简单的类型别名，它定义了一个字典，其键限于 `SupportedModel` 字面量，其值为 `float` 类型。

![2](img/#co_implementing_type_safe_ai_services_CO1-2)

使用 `TypeAlias` 标记类型别名，以明确它们不是普通变量赋值。类型通常使用驼峰式命名作为最佳实践来区分变量。您现在可以稍后重用 `PriceTable` 类型别名。

![3](img/#co_implementing_type_safe_ai_services_CO1-3)

声明定价表字典，并将 `PriceTable` 类型分配给定价表字典，以显式限制允许的键和值。

![4](img/#co_implementing_type_safe_ai_services_CO1-4)

将 `count_tokens` 函数的类型定义为接受字符串或 `None` 类型，并始终返回整数。在定义 `count_tokens` 时，如果传入 `None` 并引发错误，则代码编辑器和静态检查器会发出警告，如果传入除字符串或 `None` 类型之外的任何其他类型，则会引发错误。

![5](img/#co_implementing_type_safe_ai_services_CO1-5)

即使传入的是 `None` 类型，也返回 `0` 以确保您符合函数类型定义。

![6](img/6.png)[#co_implementing_type_safe_ai_services_CO1-6]

使用 OpenAI 的 `tiktoken` 库对给定文本进行分词，使用与 `gpt-4o` 模型相同的编码。2

![7](img/7.png)[#co_implementing_type_safe_ai_services_CO1-7]

将 `calculate_usage_costs` 函数的类型定义为始终接受文本提示和 `model` 参数的预指定字面量。传递之前声明的 `PriceTable` 类型别名 `price_table`。该函数应返回三个浮点数的元组。

![8](img/8.png)[#co_implementing_type_safe_ai_services_CO1-8]

类型检查器会在传入意外的模型字面量时发出警告，但您应该始终检查函数的输入是否正确，并在运行时如果传入意外的模型参数则引发错误。

![9](img/9.png)[#co_implementing_type_safe_ai_services_CO1-9]

从定价表中获取正确的价格。无需担心异常处理，因为没有机会在这里引发 `KeyError`（如果传入不支持的模型）。如果定价表未更新，函数将提前引发 `ValueError`。捕获 `KeyError`，发出警告称定价表需要更新，然后重新引发 `KeyError`，以便将问题的全部详细信息打印到终端，因为您不能对价格做出假设。

![10](img/10.png)[#co_implementing_type_safe_ai_services_CO1-10]

使用 `count_tokens` 函数计算 LLM 请求和响应的成本。如果由于任何原因 LLM 没有返回响应（返回 `None`），`count_tokens` 可以处理它并假设为零个令牌。

![11](img/11.png)[#co_implementing_type_safe_ai_services_CO1-11]

按照函数类型定义返回三个浮点数的元组。

在一个复杂的代码库中，猜测正在传递的数据类型可能具有挑战性，尤其是如果您在各个地方做了很多更改。使用类型化函数，您可以确信不会将意外的参数传递给尚未支持它的函数。

如 示例 4-2 所示，对代码进行类型检查有助于在更新代码时捕获意外的错误。例如，如果您开始使用新的 LLM 模型，您还不能计算新模型的成本。为了支持其他 LLM 模型的成本计算，您首先应该更新定价表、相关类型定义和任何异常处理逻辑。一旦完成，您可以相当有信心，您的计算逻辑现在已扩展到支持新的模型类型。

## 使用 Annotated

在 示例 4-2 中，您可以使用 `Annotated` 而不是类型别名。`Annotated` 是 `typing` 模块的一个特性——自 Python 3.9 引入——它类似于类型别名，用于重用类型，但它还允许您为您的类型定义 *元数据*。

元数据不会影响类型检查器，但对代码文档、分析和运行时检查很有用。

自从 Python 3.9 介绍以来，您可以使用 `Annotated`，如示例 4-3 所示。

##### 示例 4-3\. 使用 `Annotated` 声明带有元数据的自定义类型

```py
from typing import Annotated, Literal

SupportedModels = Annotated[
    Literal["gpt-3.5-turbo", "gpt-4o"], "Supported text models"
]
PriceTableType = Annotated[
    dict[SupportedModels, float], "Supported model pricing table"
]

prices: PriceTableType = {
    "gpt-4o": 0.000638,
    # error: Dict entry 1 has incompatible type "Literal['gpt4-o']" [dict-item]
    "gpt4-o": 0.000638,
    # error: Dict entry 2 has incompatible type "Literal['gpt-4']" [dict-item]
    "gpt-4": 0.000638,
}
```

[FastAPI 文档](https://oreil.ly/mtGcY)建议使用 `Annotated` 而不是类型别名以提高可重用性、增强代码编辑器中的类型检查以及在运行时捕获问题。

###### 警告

请记住，`Annotated` 功能需要至少两个参数才能工作。第一个应该是传入的类型，其他参数是您想要附加到类型上的注释或元数据，例如描述、验证规则或其他元数据，如示例 4-3 所示。

类型化，虽然本身有益，但并不解决数据处理和结构化的所有方面。幸运的是，Python 的标准库中的 *dataclasses* 有助于扩展类型系统。

让我们看看您如何利用数据类来改进应用程序中的类型化。

## 数据类

数据类是在 Python 3.7 作为标准库的一部分引入的。如果您需要自定义数据结构，可以使用数据类来组织、存储和跨应用程序传输数据。

它们可以帮助避免代码“异味”，例如函数参数膨胀，其中函数难以使用，因为它需要超过几个参数。使用数据类允许您以自定义定义的结构组织您的数据，并将其作为单个项目传递给需要来自不同位置数据的函数。

您可以将示例 4-2 更新为利用数据类，如示例 4-4 所示。

##### 示例 4-4\. 使用数据类强制类型安全

```py
# utils.py

from dataclasses import dataclass
from typing import Literal, TypeAlias
from utils import count_tokens

SupportedModels: TypeAlias = Literal["gpt-3.5", "gpt-4"]
PriceTable: TypeAlias = dict[SupportedModels, float]
prices: PriceTable = {"gpt-3.5": 0.0030, "gpt-4": 0.0200}

@dataclass ![1](img/1.png)
class Message:
    prompt: str
    response: str | None ![2](img/2.png)
    model: SupportedModels

@dataclass
class MessageCostReport:
    req_costs: float
    res_costs: float
    total_costs: float

# Define count_tokens function as normal
...

def calculate_usage_costs(message: Message) -> MessageCostReport: ![3](img/3.png)
    if message.model not in prices :
        # raise at runtime - in case someone ignores type errors
        raise ValueError(
            f"Cost calculation is not supported for {message.model} model."
        )
    price = prices[message.model]
    req_costs = price * count_tokens(message.prompt) / 1000
    res_costs = price * count_tokens(message.response) / 1000
    total_costs = req_costs + res_costs
    return MessageCostReport(
        req_costs=req_costs, res_costs=res_costs, total_costs=total_costs
    )
```

![1](img/1.png)[#co_implementing_type_safe_ai_services_CO2-1]

使用数据类将 `Message` 和 `MessageCost` 类装饰为特殊类，用于存储数据。

![2](img/2.png)[#co_implementing_type_safe_ai_services_CO2-2]

将 `response` 属性的类型指定为 `str` 或 `None`。这与使用 `typing` 模块中的 `Optional[str]` 相似。这种新语法在 Python 3.10 及以后的版本中可用，使用新的联合运算符：`|`。

![3](img/3.png)[#co_implementing_type_safe_ai_services_CO2-3]

将 `calculate_usage_costs` 函数的签名更改为使用预定义的数据类。此更改简化了函数签名。

当您的代码积累代码异味且难以阅读时，应旨在利用数据类。

在示例 4-4 中使用数据类的首要好处是将相关参数分组以简化函数签名。在其他场景中，您可以使用数据类来：

+   消除代码重复

+   缩小代码膨胀（大类或函数）

+   重构数据块（常用在一起的变量）

+   防止意外数据突变

+   促进数据组织

+   促进封装

+   强制数据验证

它们还可以用于实现许多其他代码增强功能。

数据类是改进应用程序中任何位置的数据组织和交换的出色工具。然而，当构建 API 服务时，它们在几个方面并不原生支持：

自动数据解析

在赋值时将 ISO 日期时间格式字符串解析为日期时间对象

字段验证

对字段值赋值执行复杂检查，例如检查字符串是否过长

序列化和反序列化

在使用不常见类型时，在 JSON 和 Python 数据结构之间进行转换

字段过滤

移除未设置或包含 `None` 值的对象字段

提到的任何限制都不会迫使你放弃使用数据类。当你需要创建具有最小样板代码的数据中心类时，应使用数据类而不是普通类，因为它们会自动生成特殊方法、类型注解和默认值支持，从而减少潜在的错误。然而，如 `pydantic` 这样的库支持这些功能，如果你不想实现自己的自定义逻辑（例如，序列化 datetime 对象）。

###### 小贴士

FastAPI 也通过 Pydantic 支持数据类，它实现了具有上述功能的数据类的自定义版本，使你可以迁移大量使用数据类的代码库。

让我们来看看 Pydantic，以及它为什么适合构建 GenAI 服务。

# Pydantic 模型

Pydantic 是最广泛使用的支持自定义验证器和序列化器的数据验证库。Pydantic 的核心逻辑由 Python 中的类型注解控制，并可以以 JSON 格式输出数据，从而允许与其他工具无缝集成。

此外，Pydantic V2 的核心数据验证逻辑已被重写为 Rust，以最大化其速度和性能，使其成为 Python 中最快的几个数据验证库之一。因此，Pydantic 对 FastAPI 以及 Python 生态系统中的 8,000 多个其他包（包括 Hugging Face、Django 和 LangChain）产生了重大影响。它是一个经过实战检验的工具包，由主要科技公司使用，截至撰写本文时，每月有 1.41 亿次下载，使其成为在项目中替代数据类的合适候选。

Pydantic 通过其自己的 `BaseModel` 实现提供了一套广泛的数据验证和处理工具集。Pydantic 模型与数据类有许多相似之处，但在细微之处有所不同。当你创建 Pydantic 模型时，会调用一系列初始化钩子，这些钩子为模型添加数据验证、序列化和 JSON 模式生成功能，这些是纯数据类所缺乏的。

FastAPI 与 Pydantic 紧密集成，并在底层利用其丰富的功能集进行数据处理。类型检查器和代码编辑器也可以像数据类一样读取 Pydantic 模型，以执行检查并提供自动完成功能。

## 如何使用 Pydantic

您可以使用以下命令将 Pydantic 安装到您的项目中：

```py
$ pip install pydantic
```

Pydantic 在其核心实现了一个 `BaseModel`，这是定义模型的主要方法。*模型* 简单地是继承自 `BaseModel` 的类，并使用类型提示定义字段作为注解属性。任何模型都可以用作模式来验证您的数据。

除了分组数据，^(3) Pydantic 模型还允许您指定服务端点的请求和响应需求，并验证来自外部来源的未经验证的传入数据。您还可以使用 Pydantic 模型（以及您将在不久后了解到的验证器）过滤 LLM 输出。

您可以创建自己的 Pydantic 模型，如 示例 4-5 所示。

##### 示例 4-5\. 创建 Pydantic 模型

```py
from typing import Literal
from pydantic import BaseModel

class TextModelRequest(BaseModel): ![1](img/1.png)
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    prompt: str
    temperature: float = 0.0 ![2](img/2.png)
```

![1](img/1.png)[#co_implementing_type_safe_ai_services_CO3-1]

定义继承自 Pydantic `BaseModel` 的 `TextModelRequest` 模型。

![2](img/2.png)[#co_implementing_type_safe_ai_services_CO3-2]

如果未提供显式值，则设置默认值。例如，如果初始化时未提供值，则将 `temperature` 字段设置为 `0.0`。

示例 4-5 也展示了如何将您的数据类转换为 Pydantic 模型以利用其众多功能。

## 复合 Pydantic 模型

使用 Pydantic 模型，您可以声明数据 *模式*，这些模式定义了服务操作中支持的数据结构。此外，您还可以使用继承来构建复合模型，如 示例 4-6 所示。

##### 示例 4-6\. 创建 Pydantic 模型

```py
# schemas.py

from datetime import datetime
from typing import Annotated, Literal
from pydantic import BaseModel

class ModelRequest(BaseModel): ![1](img/1.png)
    prompt: str

class ModelResponse(BaseModel): ![2](img/2.png)
    request_id: str
    ip: str | None
    content: str | None
    created_at: datetime = datetime.now()

class TextModelRequest(ModelRequest):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: float = 0.0

class TextModelResponse(ModelResponse):
    tokens: int

ImageSize = Annotated[tuple[int, int], "Width and height of an image in pixels"]

class ImageModelRequest(ModelRequest): ![3](img/3.png)
    model: Literal["tinysd", "sd1.5"]
    output_size: ImageSize
    num_inference_steps: int = 200

class ImageModelResponse(ModelResponse): ![4](img/4.png)
    size: ImageSize
    url: str
```

![1](img/1.png)[#co_implementing_type_safe_ai_services_CO4-1]

定义继承自 Pydantic `BaseModel` 的 `ModelRequest` 模型。

![2](img/2.png)[#co_implementing_type_safe_ai_services_CO4-2]

定义 `ModelResponse`。如果未提供 `ip` 可选字段的数据，则使用默认值 `None`。`content` 字段可以是字节（用于图像图像）或字符串（用于文本模型）。

![3](img/3.png)[#co_implementing_type_safe_ai_services_CO4-3]

通过继承 `ModelRequest` 来定义 `TextModelRequest` 和 `ImageModelRequest` 模型。默认情况下，可选的温度字段设置为 0.0。`ImageModelRequest` 模型的 `num_inference_steps` 字段是可选的，并设置为 200。这两个模型现在都需要提供提示字符串字段。

![4](img/4.png)[#co_implementing_type_safe_ai_services_CO4-4]

通过继承 `ModelResponse` 模型来定义 `ImageModelResponse` 和 `TextModelResponse` 模型。对于 `TextModelResponse`，提供标记的数量；对于 `ImageModelResponse`，提供图像的像素大小以及下载图像的远程 URL。

如 示例 4-6 所示的模型，您拥有了定义文本和图像生成端点需求所需的模式。

## 字段约束和验证器

除了支持标准类型外，Pydantic 还附带了一些受限类型，如 `EmailStr`、`PositiveInt`、`UUID4`、`AnyHttpUrl` 等，这些类型可以在模型初始化时直接进行数据验证，适用于常见的数据格式。Pydantic 的完整类型列表可在[官方文档](https://oreil.ly/xNbXX)中找到。

###### 注意

一些受限类型，如 `EmailStr`，可能需要安装依赖包才能正常工作，但可以非常有助于验证常见的数据格式，例如电子邮件。

要在 Pydantic 限制类型之上定义更多自定义和复杂的字段约束，您可以使用 Pydantic 的 `Field` 函数与 `Annotated` 类型一起引入验证约束，例如有效的输入范围。

示例 4-7 使用受限类型和 `Field` 函数替换了 示例 4-6 中的标准类型提示，以根据模型约束为您的端点实现更严格的数据要求。

##### 示例 4-7\. 使用受限字段

```py
# schemas.py

from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import BaseModel, Field, HttpUrl, IPvAnyAddress, PositiveInt

class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=10000)] ![1](img/1.png)

class ModelResponse(BaseModel):
    request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)] ![2](img/2.png)
    # no defaults set for ip field
    # raise ValidationError if a valid IP address or None is not provided
    ip: Annotated[str, IPvAnyAddress] | None ![3](img/3.png)
    content: Annotated[str | None, Field(min_length=0, max_length=10000)] ![4](img/4.png)
    created_at: datetime = datetime.now()

class TextModelRequest(ModelRequest):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.0)] ![5](img/5.png)

class TextModelResponse(ModelResponse):
    tokens: Annotated[int, Field(ge=0)]

ImageSize = Annotated ![6
    tuple[PositiveInt, PositiveInt], "Width and height of an image in pixels"
]

class ImageModelRequest(ModelRequest):
    model: Literal["tinysd", "sd1.5"]
    output_size: ImageSize ![6](img/6.png)
    num_inference_steps: Annotated[int, Field(ge=0, le=2000)] = 200 ![7](img/7.png)

class ImageModelResponse(ModelResponse):
    size: ImageSize ![6](img/6.png)
    url: Annotated[str, HttpUrl] | None = None ![8](img/8.png)
```

![1](img/1.png)[#co_implementing_type_safe_ai_services_CO5-1]

将 `str` 标准类型替换为 `Field` 和 `Annotated` 以将字符串长度限制在字符范围内。

![2](img/2.png)[#co_implementing_type_safe_ai_services_CO5-2]

通过传递一个可调用的函数到 `default_factory` 来生成一个新的请求 UUID，该函数将被调用来生成新的 UUID。

![3](img/3.png)[#co_implementing_type_safe_ai_services_CO5-3]

将可选的 `ip` 字段限制为任何有效的 IPv4 或 IPv6 地址范围。如果客户端的 IP 地址无法确定，`None` 也是一个有效的条目。此可选字段没有默认值，因此如果未提供有效的 IP 或 `None`，Pydantic 将引发 `ValidationError`。

![4](img/4.png)[#co_implementing_type_safe_ai_services_CO5-4]

将 `content` 字段限制为 10,000 个字符或字节。

![5](img/5.png)[#co_implementing_type_safe_ai_services_CO5-5]

将温度限制在 `0.0` 和 `1.0` 之间，默认值为 `0.0`。

![6](img/6.png)[#co_implementing_type_safe_ai_services_CO5-6]

重复使用 `Annotated` 限制 `output_size` 字段为正整数，使用 `PositiveInt` 限制类型。`lte` 和 `gte` 关键字分别表示 *小于等于* 和 *大于等于*。

![7](img/7.png)[#co_implementing_type_safe_ai_services_CO5-8]

使用 `Field` 函数将 `num_inference_steps` 字段限制在 `0` 到 `2000` 之间，默认值为 `200`。

![8](img/8.png)[#co_implementing_type_safe_ai_services_CO5-10]

将可选的 `url` 字段限制为任何有效的 HTTP 或 HTTPS URL，其中主机名和顶级域名（TLD）是必需的。

在定义了 示例 4-7 中的模型后，你现在可以对传入或传出的数据进行验证，以匹配您所需的数据要求。在这种情况下，FastAPI 将利用 Pydantic 在请求运行时数据验证检查失败时自动返回错误响应，如 示例 4-8 所示。

##### 示例 4-8\. 数据验证失败时的 FastAPI 错误响应

```py
$ curl -X 'POST' \
  'http://127.0.0.1:8000/validation/failure' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
 "prompt": "string",
 "model": "gpt-4o",
 "temperature": 0
}'

{
  "detail": [
    {
      "type": "literal_error",
      "loc": [
        "body",
        "model"
      ],
      "msg": "Input should be 'tinyllama' or 'gemma2b'",
      "input": "gpt-4o",
      "ctx": {
        "expected": "'tinyllama' or 'gemma2b'"
      }
    }
  ]
}
```

## 自定义字段和模型验证器

Pydantic 进行数据验证检查的另一个优秀功能是 *自定义字段验证器*。示例 4-9 展示了如何在 `ImageModelRequest` 上实现这两种类型的自定义验证器。

##### 示例 4-9\. 为 `ImageModelRequest` 实现自定义字段和模型验证器

```py
# schemas.py

from typing import Annotated, Literal
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    PositiveInt,
    validate_call,
)

ImageSize = Annotated[
    tuple[PositiveInt, PositiveInt], "Width and height of an image in pixels"
]
SupportedModels = Annotated[
    Literal["tinysd", "sd1.5"], "Supported Image Generation Models"
]

@validate_call ![1](img/1.png)
def is_square_image(value: ImageSize) -> ImageSize: ![2](img/2.png)
    if value[0] / value[1] != 1:
        raise ValueError("Only square images are supported")
    if value[0] not in [512, 1024]:
        raise ValueError(f"Invalid output size: {value} - expected 512 or 1024")
    return value

@validate_call ![1](img/1.png)
def is_valid_inference_step(
    num_inference_steps: int, model: SupportedModels
) -> int:
    if model == "tinysd" and num_inference_steps > 2000: ![3](img/3.png)
        raise ValueError(
            "TinySD model cannot have more than 2000 inference steps"
        )
    return num_inference_steps

OutputSize = Annotated[ImageSize, AfterValidator(is_square_image)] ![4](img/4.png)
InferenceSteps = Annotated ![4
    int,
    AfterValidator(
        lambda v, values: is_valid_inference_step(v, values["model"])
    ),
]

class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]

class ImageModelRequest(ModelRequest):
    model: SupportedModels
    output_size: OutputSize ![5](img/5.png)
    num_inference_steps: InferenceSteps = 200 ![6](img/6.png)
```

![1](img/#co_implementing_type_safe_ai_services_CO6-1)

除了静态类型检查外，如果向 `is_square_image` 和 `is_valid_​infer⁠ence_step` 函数传递了不正确的参数，则引发运行时验证错误。

![2](img/#co_implementing_type_safe_ai_services_CO6-2)

`tinysd` 模型只能生成特定尺寸的方形图像。请求非方形图像尺寸（除 `1` 以外的纵横比）应引发 `ValueError`。

![3](img/#co_implementing_type_safe_ai_services_CO6-4)

如果用户请求 `tinysd` 模型的许多推理步骤，则引发 `ValueError`。

![4](img/#co_implementing_type_safe_ai_services_CO6-5)

使用 `OutputSize` 和 `InferenceSteps` 的注解模式创建可重用且更易于阅读的验证器。

![5](img/#co_implementing_type_safe_ai_services_CO6-7)

将 `OutputSize` 字段验证器附加到 `output_size` 字段，以在模型初始化后检查不正确的值。

![6](img/#co_implementing_type_safe_ai_services_CO6-8)

将 `InferenceSteps` 验证器附加到 `ImageModelRequest` 模型，以在模型初始化后对模型字段值进行检查。

如 示例 4-9 所示，使用自定义字段验证器，您现在可以确信您的图像生成端点将受到用户提供的错误配置的保护。

###### 注意

您还可以使用装饰器模式来验证模型字段。可以通过使用 `@field_validator` 或 `@model_validator` 装饰器将特殊方法与模型字段关联起来，以执行条件数据检查。

虽然 `@field_validator` 通过访问单个字段的值来执行检查，但 `@model_validator` 装饰器允许对涉及多个字段的检查。

使用 `after` 验证器，您可以在 Pydantic 完成其解析和验证后执行额外的检查或修改数据。

## 计算字段

与数据类类似，Pydantic 也允许您实现方法来计算来自其他字段派生的字段。

你可以使用 `@computed_field` 装饰器来实现一个计算字段，用于计算令牌和成本的数量，如下所示示例 4-10。

##### 示例 4-10\. 使用计算字段自动计算令牌总数

```py
# schemas.py

from typing import Annotated
from pydantic import computed_field, Field
from utils import count_tokens

...

class TextModelResponse(ModelResponse):
    model: SupportedModels
    price: Annotated[float, Field(ge=0, default=0.01)]
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.0)]

    @property
    @computed_field
    def tokens(self) -> int:
        return count_tokens(self.content)

    @property
    @computed_field
    def cost(self) -> float:
        return self.price * self.tokens
```

计算字段对于封装 Pydantic 模型中的任何字段计算逻辑非常有用，以保持代码整洁。请注意，计算字段仅在将 Pydantic 模型转换为字典时通过 `.model_dump()` 或在 FastAPI API 处理程序返回响应时的序列化时才可访问。

## 模型导出和序列化

由于 Pydantic 模型可以序列化为 JSON，因此你在示例 4-7 中定义的模型也可以被导出到（或从）JSON 字符串或 Python 字典中，同时保持任何复合模式，如下所示示例 4-11。

##### 示例 4-11\. 导出和序列化 `TextModelResponse` 模型

```py
>> response = TextModelResponse(content="FastAPI Generative AI Service", ip=None)
>> response.model_dump(exclude_none=True)
{'content': 'FastAPI Generative AI Service',
 'cost': 0.06,
 'created_at': datetime.datetime(2024, 3, 7, 20, 42, 38, 729410),
 'price': 0.01,
 'request_id': 'a3f18d85dcb442baa887a505ae8d2cd7',
 'tokens': 6}

>> response.model_dump_json(exclude_unset=True)
'{"ip":null,"content":"FastAPI Generative AI Service","tokens":6,"cost":0.06}'
```

## 使用 Pydantic 解析环境变量

除了 `BaseModel`，Pydantic 还实现了一个 `Base` 类，用于从文件中解析设置和秘密。此功能由一个可选的 Pydantic 包 `pydantic-settings` 提供，你可以将其作为依赖项安装：

```py
$ pip install pydantic-settings
```

`BaseSettings` 类提供了可选的 Pydantic 功能，用于从环境变量或秘密文件中加载设置或配置类。使用此功能，设置值可以在代码中设置或由环境变量覆盖。

在生产环境中，这很有用，因为你不想在代码或容器环境中暴露秘密。

当你创建一个继承自 `BaseSettings` 的模型时，模型初始化器将尝试使用提供的默认值设置每个字段的值。如果失败，初始化器将然后从环境变量中读取任何未设置字段的值。

给定一个 dotenv 环境文件 (ENV)：

```py
APP_SECRET=asdlkajdlkajdklaslkldjkasldjkasdjaslk
DATABASE_URL=postgres://sa:password@localhost:5432/cms
CORS_WHITELIST=["https://xyz.azurewebsites.net","http://localhost:3000"]
```

ENV 是一个环境变量文件，可以使用 shell 脚本语法来表示键值对。

示例 4-12 展示了使用 `BaseSettings` 解析环境变量的实际操作。

##### 示例 4-12\. 使用 Pydantic `BaseSettings` 解析环境变量

```py
# settings.py

from typing import Annotated
from pydantic import Field, HttpUrl, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings): ![1](img/1.png)
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8" ![2](img/2.png)
    )

    port: Annotated[int, Field(default=8000)]
    app_secret: Annotated[str, Field(min_length=32)]
    pg_dsn: Annotated[
        PostgresDsn,
        Field(
            alias="DATABASE_URL",
            default="postgres://user:pass@localhost:5432/database",
        ),
    ] ![3](img/3.png)
    cors_whitelist_domains: Annotated[
        set[HttpUrl],
        Field(alias="CORS_WHITELIST", default=["http://localhost:3000"]),
    ] ![4](img/4.png)

settings = AppSettings()
print(settings.model_dump()) ![5](img/5.png)
"""
{'port': 8000
 'app_secret': 'asdlkajdlkajdklaslkldjkasldjkasdjaslk',
 'pg_dsn': MultiHostUrl('postgres://sa:password@localhost:5432/cms'),
 'cors_whitelist_domains': {Url('http://localhost:3000/'),
                            Url('https://xyz.azurewebsites.net/')},
}
"""
```

![1](img/1.png)[#co_implementing_type_safe_ai_services_CO7-1]

声明从 `pydantic_settings` 包继承的 `AppSettings` 类。

![2](img/2.png)[#co_implementing_type_safe_ai_services_CO7-2]

将 `AppSettings` 配置为以 `UTF-8` 编码从项目根目录的 ENV 文件中读取环境变量。默认情况下，snake_case 字段名称将映射到环境变量名称的大写版本。例如，`app_secret` 变为 `APP_SECRET`。

![3](img/3.png)[#co_implementing_type_safe_ai_services_CO7-3]

验证 `DATABASE_URL` 环境变量是否具有有效的 Postgres 连接字符串格式。如果没有提供，则设置默认值。

![4](img/4.png)[#co_implementing_type_safe_ai_services_CO7-4]

检查`CORS_WHITELIST`环境变量是否包含具有主机名和顶级域的唯一 URL 列表。如果没有提供，则将默认值设置为包含单个值`http://localhost:3000`的集合。

![5](img/#co_implementing_type_safe_ai_services_CO7-5)

我们可以通过打印模型的转储来检查`AppSettings`类是否正常工作。

###### 注意

当使用`_env_file`参数时，你可以切换环境文件：

```py
test_settings = AppSettings(_env_file="test.env")
```

## FastAPI 中的数据类或 Pydantic 模型

尽管 dataclasses 只支持常见类型的序列化（例如，`int`，`str`，`list`等）并且在运行时不会执行字段验证，但 FastAPI 仍然可以与 Pydantic 模型和 Python 的数据类一起工作。对于字段验证和附加功能，你应该使用 Pydantic 模型。示例 4-13 展示了如何在 FastAPI 路由处理程序中使用数据类。

##### 示例 4-13\. 在 FastAPI 中使用数据类

```py
# schemas.py

from dataclasses import dataclass
from typing import Literal

@dataclass
class TextModelRequest: ![1](img/1.png)
    model: Literal["tinyLlama", "gemma2b"]
    prompt: str
    temperature: float

@dataclass
class TextModelResponse: ![1](img/1.png)
    response: str
    tokens: int

# main.py

from fastapi import Body, FastAPI, HTTPException, status
from models import generate_text, load_text_model
from schemas import TextModelRequest, TextModelResponse
from utils import count_tokens

# load lifespan
...

app = FastAPI(lifespan=lifespan)

@app.post("/generate/text")
def serve_text_to_text_controller(
    body: TextModelRequest = Body(...),
) -> TextModelResponse: ![2](img/2.png) ![4](img/4.png)
    if body.model not in ["tinyLlama", "gemma2b"]: ![3](img/3.png)
        raise HTTPException(
            detail=f"Model {body.model} is not supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    output = generate_text(models["text"], body.prompt, body.temperature)
    tokens = count_tokens(body.prompt) + count_tokens(output)
    return TextModelResponse(response=output, tokens=tokens) ![4](img/4.png)
```

![1](img/#co_implementing_type_safe_ai_services_CO8-1)

定义用于文本模型请求和响应模式的模型。

![2](img/#co_implementing_type_safe_ai_services_CO8-3)

将处理程序转换为服务带有正文的`POST`请求。然后，声明请求正文为`TextModelRequest`，响应为`TextModelResponse`。静态代码检查器（如`mypy`）将读取类型注解，并在你的控制器未返回预期响应模型时发出警告。

![3](img/#co_implementing_type_safe_ai_services_CO8-5)

明确检查服务是否支持请求`body`中提供的`model`参数。如果不支持，则向客户端返回一个错误的 HTTP 异常响应。

![4](img/#co_implementing_type_safe_ai_services_CO8-4)

FastAPI 将纯数据类转换为 Pydantic 数据类以序列化/反序列化和验证请求和响应数据。

在示例 4-13 中，你通过重构文本模型控制器以适应新的更改和不良用户输入，利用了类型注解。静态类型检查器现在可以在更改发生时帮助你捕获任何数据相关的问题。此外，FastAPI 使用你的类型注解来验证请求和响应，同时自动生成一个 OpenAPI 文档页面，如图 4-2 所示。

你现在可以看到，即使你使用纯数据类，FastAPI 也利用 Pydantic 模型在底层进行数据处理和验证。FastAPI 将你的纯数据类转换为 Pydantic 风格的数据类，以使用其数据验证功能。这种行为是有意为之的，因为如果你有包含几个预存在数据类类型注解的项目，你仍然可以迁移它们，而无需将它们重写为 Pydantic 模型以利用数据验证功能。然而，如果你正在启动一个全新的项目，建议直接使用 Pydantic 模型来替换 Python 的内置数据类。

![bgai 0402](img/bgai_0402.png)

###### 图 4-2\. 使用 vanilla dataclasses 自动生成验证模式

现在，让我们看看如何在您的 FastAPI 应用程序中将 dataclasses 替换为 Pydantic。参见 示例 4-14。

##### 示例 4-14\. 使用 Pydantic 模型请求和响应模式

```py
# main.py

from fastapi import Body, FastAPI, HTTPException, Request, status
from models import generate_text
from schemas import TextModelRequest, TextModelResponse ![1](img/1.png)

# load lifespan
...

app = FastAPI(lifespan=lifespan)

@app.post("/generate/text") ![2](img/2.png)
def serve_text_to_text_controller(
    request: Request, body: TextModelRequest = Body(...)
) -> TextModelResponse:
    if body.model not in ["tinyLlama", "gemma2b"]: ![3](img/3.png)
        raise HTTPException(
            detail=f"Model {body.model} is not supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    output = generate_text(models["text"], body.prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host) ![4](img/4.png)
```

![1](img/1.png) (#co_implementing_type_safe_ai_services_CO9-1)

导入 Pydantic 模型以用于文本模型请求和响应模式。

![2](img/2.png) (#co_implementing_type_safe_ai_services_CO9-2)

将处理器转换为处理带有正文的 `POST` 请求。然后，将请求正文声明为 `TextModelRequest`，将响应声明为 `TextModelResponse`。静态代码检查器如 `mypy` 将读取类型注解，并在您的控制器未返回预期响应模型时引发警告。

![3](img/3.png) (#co_implementing_type_safe_ai_services_CO9-3)

明确检查服务是否支持请求 `body` 中提供的 `model` 参数。如果不支持，向客户端返回一个不良请求 HTTP 异常响应。

![4](img/4.png) (#co_implementing_type_safe_ai_services_CO9-4)

根据函数类型返回 `TextModelResponse` Pydantic 模型。通过 `request.client.host` 使用请求对象访问客户端的 IP 地址。FastAPI 将在底层使用 `.model_dump()` 处理序列化您的模型。由于您还实现了 `tokens` 和 `cost` 属性的计算字段，这些字段将自动包含在您的 API 响应中，无需任何额外工作。

###### 注意

如 示例 4-13 所示，如果您使用 dataclasses 而不是 Pydantic 模型，FastAPI 将将它们转换为 Pydantic dataclasses 以进行序列化/反序列化和验证请求和响应数据。但是，您可能无法利用数据类的高级功能，如字段约束和计算字段。

正如您在 示例 4-14 中所见，Pydantic 可以通过帮助进行类型检查、数据验证、序列化、代码编辑器自动完成和计算属性，提供卓越的开发者体验。

FastAPI 还可以使用您的 Pydantic 模型自动生成 OpenAPI 规范和文档页面，以便您可以无缝地手动测试您的端点。

一旦启动服务器，您应该看到一个带有新 Pydantic 模型和更新约束字段的更新文档页面，如图 4-3 所示。图 4-3。

![bgai 0403](img/bgai_0403.png)

###### 图 4-3\. 使用 Pydantic 模型自动生成 FastAPI 文档

如果您向 `/generate/text` 端点发送请求，您现在应该通过 `TextModelResponse` Pydantic 模型看到预填充的字段，如 示例 4-15 所示。

##### 示例 4-15\. 通过 `TextModelResponse` Pydantic 模型自动填充响应字段

```py
Request

curl -X 'POST' \
    'http://localhost:8000/generate/text' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "prompt": "What is your name?",
    "model": "tinyllama",
    "temperature": 0.01
}'

http://localhost:8000/generate/text

>> Response body
{
    "request_id": "7541204d5c684f429fe43ccf360fЗ3dc",
    "ip": "127.0.0.1"
    "content": "I am not a person. However, I can provide you with information
        about my name. My name is fastapi bot.",
    "created_at": "2024-03-07T16:06:57.492039",
    "price": 0.01,
    "tokens": 25,
    "cost": 0.25
}

>> Response headers

content-length: 259
content-type: application/json
date: Thu, 07 Mar 2024 16:07:01 GMT
server: uvicorn
x-response-time: 22.9243
```

本章中我介绍的 Pydantic 模型功能只是你可用于构建 GenAI 服务的工具中的一小部分。现在你应该更有信心利用 Pydantic 注释你的服务，以提高其可靠性和你的开发体验。

# 摘要

在本章中，你学习了为 GenAI 模型创建完全类型化服务的重要性。你现在理解了如何使用标准类型和约束类型实现类型安全，如何使用 Pydantic 模型进行数据验证，以及如何在你的 GenAI 服务中实现自定义数据验证器。你还发现了验证请求和响应内容以及使用 Pydantic 管理应用程序设置的策略，以防止错误并提高你的开发体验。总的来说，通过跟随实际示例，你学习了如何实现一个健壮、错误更少的 GenAI 服务。

下一章将介绍 AI 工作负载中的异步编程，讨论性能和并行操作。你将了解更多关于 I/O 密集型和 CPU 密集型任务，以及 FastAPI 的后台任务在并发工作流程中的作用和限制。

^(1) 一个 [文本类型](https://oreil.ly/69Pmn) 可以用来向类型检查器指示，被注释的对象的值等同于提供的文本之一。

^(2) OpenAI 的 `tiktoken` 使用 [*字节对编码 (BPE)* 算法](https://oreil.ly/l67GS) 来分词文本。不同的模型使用不同的编码将文本转换为标记。

^(3) 类 C 语言中的结构体和 Python 中的数据类也可以用来分组和传递数据。
