# Chapter 4\. Implementing Type-Safe AI Services

When working with complex codebases that continuously change by multiple contributors and when interacting with external services such as APIs or databases, you will want to follow best practices such as type safety in building your applications.

This chapter focuses on the importance of type safety when building backend services and APIs. You will learn how to implement type safety using Python’s built-in dataclasses and then Pydantic data models, and you will see their similarities and differences. In addition, you will explore how to use Pydantic data models with custom validators to protect against bad user input or incorrect data, and you will learn how to use Pydantic Settings for loading and validating environment variables. Finally, you will discover strategies for dealing with schema changes in external systems and managing complexity in evolving codebases to prevent bugs.

By the end of this chapter, you will have a fully typed GenAI service that is less prone to bugs when dealing with changes, bad user inputs, and inconsistent model responses.

To follow along, you can find the starting code for this chapter by switching to the [`ch04-start` branch](https://github.com/Ali-Parandeh/building-generative-ai-services/tree/ch04-start).

# Introduction to Type Safety

*Types* in programming specify what values can be assigned to variables and operations that can be performed on those variables.

In Python, common types include the following:

Integer

Representing whole numbers

Float

Representing numbers with fractional parts

String

Representing sequences of characters

Boolean

Representing `True` or `False` values

###### Tip

You can use the `typing` package to import special types as you saw in other code examples in [Chapter 3](ch03.html#ch03).

*Type safety* is a programming practice that ensures variables are only assigned values compatible with their defined types. In Python, you can use types to check the usage of variables across a codebase, in particular if the codebase grows in complexity and size. Type checking tools (e.g., `mypy`) can then use these types to catch incorrect variable assignments or operations.

You can enforce type constraints by declaring fully typed variables and functions as shown in [Example 4-1](#type_safe_code).

##### Example 4-1\. Using types in Python

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

Code editors and IDEs (e.g., VS Code or JetBrains PyCharm) can also use type checking extensions, as shown in [Figure 4-1](#type_safety), to raise warnings on type violations as you write code.

![bgai 0401](assets/bgai_0401.png)

###### Figure 4-1\. Catching type errors in VS Code `mypy` extension

In a complex codebase, it is easy to lose track of variables, their states, and constantly changing schemas. For example, you might forget that the `timestamp_to_isostring` function accepts numbers as input and mistakenly pass a timestamp as a string, as shown in [Figure 4-1](#type_safety).

Types are also extremely useful when package maintainers or external API providers update their code. Type checkers can immediately raise warnings to help you address such changes during development. This way, you will be immediately directed to sources of potential errors without having to run your code and test every endpoint. As a result, type safety practices can save you time with early detection and prevent you from dealing with more obscure runtime errors.

Finally, you can go one step further to set up automatic type checks in your deployment pipeline to prevent pushing breaking changes to production environments.

Type safety at first seems like a burden. You have to explicitly type each and every function you write, which can be a hassle and slow you down in the initial phases of development.

Some people skip typing their code for rapid prototyping and to write less boilerplate code. The approach is more flexible and easier to use, and Python is powerful enough to infer simple types. Also, some code patterns (such as functions with multitype arguments) can be so dynamic that it is easier to avoid implementing strict type safety when still experimenting. However, it will come to save you hours of development as inevitably your services become complex and continuously change.

The good news is some of these types can be auto-generated using tools such as Prisma, when working with databases, or client generators, when working with external APIs. For external APIs, you can often find official SDKs containing clients with type hints (i.e., fully typed client) specifying expected types of inputs and outputs for using the API. If not, you can inspect the API to create your own fully typed client. I will cover Prisma and API client generators in more detail later in the book.

When you don’t use types, you open yourself to all sorts of bugs and errors that might occur because other developers unexpectedly updated the database tables or API schemas that your service interacts with. In other cases, you may update a database table—drop a column for instance—and forget to update the code interacting with that table.

Without types, you may never notice breaking changes due to updates. This can be challenging to debug as unhandled downstream errors might not pinpoint the broken component or general issues around unhandled edge cases from your own development team. As a result, what might have taken a minute to resolve can last half a day or even longer.

You can always prevent a few disasters in production with extensive testing. However, it’s much easier to avoid integration and reliability issues if you start using types from the start.

# Developing good programming habits

If you haven’t been typing your code in the past, it is never too late to start getting into the habit of typing all your variables, function parameters, and return types.

Using types will make your code more readable, help you catch bugs early on, and save you a lot of time when you revisit complex codebases to quickly understand how data flows.

# Implementing Type Safety

Since Python 3.5, you can explicitly declare types for your variables, function parameters, and return values. The syntax that allows you to declare these types is *type annotation*.

## Type Annotations

Type annotations don’t affect the runtime behavior of your application. They help catch type errors, particularly in complex larger applications where multiple people are working together. Tools for static type checking, such as `mypy`, `pyright`, or `pyre`, alongside code editors, can validate that the data types stored and returned from functions, match the expected types.

In Python applications, type annotations are used for:

*   *Code editor auto-complete support*

*   *Static type checks* using tools like `mypy`

FastAPI also leverages types hints to:

*   *Define handler requirements* including path and query parameters, bodies, headers, and dependencies, etc.

*   *Convert data* whenever needed

*   *Validate data* from incoming requests, databases, and external services

*   *Auto-update the OpenAPI specification* that powers the generated documentation page

You can install `loguru` using `pip`:

```py
$ pip install loguru
```

[Example 4-2](#type_annotation) shows several examples of type annotation.

##### Example 4-2\. Using type annotation to reduce future bugs as code changes occur

```py
# utils.py

from typing import Literal, TypeAlias
from loguru import logger
import tiktoken

SupportedModels: TypeAlias = Literal["gpt-3.5", "gpt-4"]
PriceTable: TypeAlias = dict[SupportedModels, float] ![1](assets/1.png) ![2](assets/2.png)
price_table: PriceTable = {"gpt-3.5": 0.0030, "gpt-4": 0.0200} ![3](assets/3.png)

def count_tokens(text: str | None) -> int: ![4](assets/4.png)
    if text is None:
        logger.warning("Response is None. Assuming 0 tokens used")
        return 0 ![5](assets/5.png)
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text)) ![6](assets/6.png)

def calculate_usage_costs(
    prompt: str,
    response: str | None,
    model: SupportedModels,
) -> tuple[float, float, float]: ![7](assets/7.png)
    if model not in price_table:
        # raise at runtime - in case someone ignores type errors
        raise ValueError(f"Cost calculation is not supported for {model} model.") ![8](assets/8.png)
    price = price_table[model] ![9](assets/9.png)
    req_costs = price * count_tokens(prompt) / 1000
    res_costs = price * count_tokens(response) / 1000 ![10](assets/10.png)
    total_costs = req_costs + res_costs
    return req_costs, res_costs, total_costs ![11](assets/11.png)
```

[![1](assets/1.png)](#co_implementing_type_safe_ai_services_CO1-1)

Use the `Literal` from Python’s `typing` module included in its standard library.^([1](ch04.html#id749)) Declare literals `gpt-3.5` and `gpt-4` and assign them to `SupportedModel` *type alias*. The `PriceTable` is also a simple type alias that defines a dictionary with keys limited to `SupportedModel` literals and with values of type `float`.

[![2](assets/2.png)](#co_implementing_type_safe_ai_services_CO1-2)

Mark type aliases with `TypeAlias` to be explicit that they’re not a normal variable assignment. Types are also normally declared using CamelCase as a best practice to differentiate them from variables. You can now reuse the `PriceTable` type alias later.

[![3](assets/3.png)](#co_implementing_type_safe_ai_services_CO1-3)

Declare the pricing table dictionary and assign the `PriceTable` type to explicitly limit what keys and values are allowed for in the pricing table dictionary.

[![4](assets/4.png)](#co_implementing_type_safe_ai_services_CO1-4)

Type the `count_tokens` function to accept strings or `None` types and always return an integer. Implement exception handling in case someone tries to pass in anything other than strings or `None` types. When defining `count_tokens`, code editor and static checkers will raise warnings if `count_tokens` doesn’t return an integer even if it receives a `None` and raises errors if any other types other than string or `None`.

[![5](assets/5.png)](#co_implementing_type_safe_ai_services_CO1-5)

Return `0` even if a `None` type is passed to ensure you comply with function typing.

[![6](assets/6.png)](#co_implementing_type_safe_ai_services_CO1-6)

Tokenize the given text using OpenAI’s `tiktoken` library using the same encoding that was used for the `gpt-4o` model.^([2](ch04.html#id750))

[![7](assets/7.png)](#co_implementing_type_safe_ai_services_CO1-7)

Type the `calculate_usage_costs` function to always take a text prompt and the prespecified literals for `model` parameter. Pass the `price_table` with the previously declared `PriceTable` type alias. The function should return a tuple of three floats.

[![8](assets/8.png)](#co_implementing_type_safe_ai_services_CO1-8)

Type checkers will raise warnings when an unexpected model literal is passed in, but you should always check for incorrect inputs to functions and raise errors at runtime if an unexpected model parameter is passed in.

[![9](assets/9.png)](#co_implementing_type_safe_ai_services_CO1-9)

Grab the correct price from the pricing table. No need to worry about exception handling, as there is no chance a `KeyError` can be raised here if an unsupported model is passed in. If the pricing table is not updated, the function will raise a `ValueError` early on. Catch the `KeyError`, issue a warning that pricing table needs updating and then reraise the `KeyError` so that full details of the issue are still printed to the terminal, as you can’t make assumptions about prices.

[![10](assets/10.png)](#co_implementing_type_safe_ai_services_CO1-10)

Use `count_tokens` function to calculate the LLM request and response costs. If for any reason the LLM doesn’t return a response (returns `None`), the `count_tokens` can handle it and assume zero tokens.

[![11](assets/11.png)](#co_implementing_type_safe_ai_services_CO1-11)

Return a tuple of three floats as per function typing.

In a complex codebase, it can be challenging to guess which data types are being passed around, especially if you make lots of changes everywhere. With typed functions, you can be confident that unexpected parameters aren’t passed to functions that don’t yet support it.

As you can see from [Example 4-2](#type_annotation), typing your code assists in catching unexpected bugs as you make updates to your code. For instance, if you start using a new LLM model, you can’t yet calculate costs for the new model. To support cost calculation for other LLM models, you first should update the pricing table, related typing, and any exception handling logic. Once done, you can be pretty confident that your calculation logic is now extended to work with new model types.

## Using Annotated

In [Example 4-2](#type_annotation), you can use `Annotated` instead of type aliases. `Annotated` is a feature of the `typing` module—​introduced in Python 3.9—​and is similar to type aliases for reusing types, but it allows you to also define *metadata* for your types.

The metadata doesn’t affect the type checkers but is useful for code documentation, analysis, and runtime inspections.

Since its introduction in Python 3.9, you can use `Annotated` as shown in [Example 4-3](#annotated_usage).

##### Example 4-3\. Using `Annotated` to declare custom types with metadata

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

The [FastAPI documentation](https://oreil.ly/mtGcY) recommends the use of `Annotated` instead of type aliases for reusability, for enhanced type checks in the code editor, and for catching issues during runtime.

###### Warning

Keep in mind that the `Annotated` feature requires a minimum of two arguments to work. The first should be the type passed in, and the other arguments are the annotation or metadata you want to attach to the type such as a description, validation rule, or other metadata, as shown in [Example 4-3](#annotated_usage).

Typing, while beneficial by itself, doesn’t address all aspects of data handling and structuring. Thankfully, Python’s *dataclasses* from the standard library help to extend the typing system.

Let’s see how you can leverage dataclasses to improve typing across your application.

## Dataclasses

Dataclasses were introduced in Python 3.7 as part of the standard library. If you need custom data structures, you can use dataclasses to organize, store, and transfer data across your application.

They can help with avoiding code “smells” such as function parameter bloat, where a function is hard to use because it requires more than a handful of parameters. Having a dataclass allows you to organize your data in a custom-defined structure and pass it as a single item to functions that require data from different places.

You can update [Example 4-2](#type_annotation) to leverage dataclasses, as shown in [Example 4-4](#dataclasses).

##### Example 4-4\. Using dataclasses to enforce type safety

```py
# utils.py

from dataclasses import dataclass
from typing import Literal, TypeAlias
from utils import count_tokens

SupportedModels: TypeAlias = Literal["gpt-3.5", "gpt-4"]
PriceTable: TypeAlias = dict[SupportedModels, float]
prices: PriceTable = {"gpt-3.5": 0.0030, "gpt-4": 0.0200}

@dataclass ![1](assets/1.png)
class Message:
    prompt: str
    response: str | None ![2](assets/2.png)
    model: SupportedModels

@dataclass
class MessageCostReport:
    req_costs: float
    res_costs: float
    total_costs: float

# Define count_tokens function as normal
...

def calculate_usage_costs(message: Message) -> MessageCostReport: ![3](assets/3.png)
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

[![1](assets/1.png)](#co_implementing_type_safe_ai_services_CO2-1)

Use dataclasses to decorate the `Message` and `MessageCost` classes as special classes for holding data.

[![2](assets/2.png)](#co_implementing_type_safe_ai_services_CO2-2)

Type the `response` attribute to be either a `str` or `None`. This is similar to using `Optional[str]` from the `typing` module. This new syntax is available in Python 3.10 and later, using the new union operator: `|`.

[![3](assets/3.png)](#co_implementing_type_safe_ai_services_CO2-3)

Change the signature of the `calculate_usage_costs` function to use the predefined dataclasses. This change simplifies the function signature.

You should aim to leverage dataclasses when your code accumulates code smells and becomes difficult to read.

The primary benefit of using dataclasses in [Example 4-4](#dataclasses) was to group related parameters to simplify the function signature. In other scenarios, you may use dataclasses to:

*   Eliminate code duplication

*   Shrink down code bloat (large classes or functions)

*   Refactor data clumps (variables that are commonly used together)

*   Prevent inadvertent data mutation

*   Promote data organization

*   Promote encapsulation

*   Enforce data validation

They can also be used to implement many other code enhancements.

Dataclasses are an excellent tool to improve data organization and exchange anywhere in your application. However, they don’t natively support several features when building API services:

Automatic data parsing

Parsing ISO datetime-formatted strings to datetime objects on assignment

Field validation

Performing complex checks on assignment of values to fields, such as checking if a string is too long

Serialization and deserialization

Converting between JSON and Pythonic data structures, especially when using uncommon types

Field filtering

Removing fields of objects that are unset or contain `None` values

None of the mentioned limitations would force you to move away from using dataclasses. You should use dataclasses rather than normal classes when you need to create data-centric classes with minimal boilerplate code, as they automatically generate special methods, type annotations, and support for default values, reducing potential errors. However, libraries such as `pydantic` support these features if you don’t want to implement your own custom logic (e.g., serializing datetime objects).

###### Tip

FastAPI also supports dataclasses through Pydantic, which implements its own version of dataclasses with support for the aforementioned features, enabling you to migrate codebases that heavily use dataclasses.

Let’s take a look at Pydantic next and what makes it great for building GenAI services.

# Pydantic Models

Pydantic is the most widely used data validation library with support for custom validators and serializers. Pydantic’s core logic is controlled by type annotations in Python and can emit data in JSON format, allowing for seamless integration with any other tools.

In addition, the core data validation logic in Pydantic V2 has been rewritten in Rust to maximize its speed and performance, positioning it as one of the fastest data validation libraries in Python. As a result, Pydantic has heavily influenced FastAPI and 8,000 other packages in the Python ecosystem including Hugging Face, Django, and LangChain. It is a battle-tested toolkit used by major tech companies with 141 million downloads a month at the time of writing, making it a suitable candidate for adoption in your projects in replacement for dataclasses.

Pydantic provides an extensive toolset for data validation and processing using its own `BaseModel` implementation. Pydantic models share many similarities with dataclasses but differ in subtle areas. When you create Pydantic models, a set of initialization hooks are called that add data validation, serialization, and JSON schema generation features to the models that vanilla dataclasses lack.

FastAPI tightly integrates with Pydantic and leverages its rich feature set under the hood for data processing. Type checkers and code editors can also read Pydantic models similar to dataclasses to perform checks and provide auto-completions.

## How to Use Pydantic

You can install Pydantic into your project using the following:

```py
$ pip install pydantic
```

Pydantic at its core implements a `BaseModel`, which is the primary method for defining models. *Models* are simply classes that inherit from `BaseModel` and define fields as annotated attributes using type hints. Any models can then be used as schemas to validate your data.

Aside from grouping data,^([3](ch04.html#id762)) Pydantic models let you specify the request and response requirements of your service endpoints and validate incoming untrusted data from external sources. You can also go as far as filter your LLM outputs using Pydantic models (and validators, which you will learn more about shortly).

You can create your own Pydantic models as shown in [Example 4-5](#pydantic_models).

##### Example 4-5\. Creating Pydantic models

```py
from typing import Literal
from pydantic import BaseModel

class TextModelRequest(BaseModel): ![1](assets/1.png)
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    prompt: str
    temperature: float = 0.0 ![2](assets/2.png)
```

[![1](assets/1.png)](#co_implementing_type_safe_ai_services_CO3-1)

Define the `TextModelRequest` model inheriting the Pydantic `BaseModel`.

[![2](assets/2.png)](#co_implementing_type_safe_ai_services_CO3-2)

Set defaults if an explicit value isn’t provided. For instance, set the `temperature` field to `0.0` if a value is not provided on initialization.

[Example 4-5](#pydantic_models) also shows how you can switch your dataclasses into Pydantic models to leverage its many features.

## Compound Pydantic Models

With Pydantic models, you can declare data *schemas*, which define data structures supported in the operations of your service. Additionally, you can also use inheritance for building compound models, as shown in [Example 4-6](#pydantic_compound_models).

##### Example 4-6\. Creating Pydantic models

```py
# schemas.py

from datetime import datetime
from typing import Annotated, Literal
from pydantic import BaseModel

class ModelRequest(BaseModel): ![1](assets/1.png)
    prompt: str

class ModelResponse(BaseModel): ![2](assets/2.png)
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

class ImageModelRequest(ModelRequest): ![3](assets/3.png)
    model: Literal["tinysd", "sd1.5"]
    output_size: ImageSize
    num_inference_steps: int = 200

class ImageModelResponse(ModelResponse): ![4](assets/4.png)
    size: ImageSize
    url: str
```

[![1](assets/1.png)](#co_implementing_type_safe_ai_services_CO4-1)

Define the `ModelRequest` model inheriting the Pydantic `BaseModel`.

[![2](assets/2.png)](#co_implementing_type_safe_ai_services_CO4-2)

Define the `ModelResponse`. If the data for the `ip` optional field is not provided, then use the defaults of `None`. The `content` field can be both bytes (for image images) or string (for text models).

[![3](assets/3.png)](#co_implementing_type_safe_ai_services_CO4-3)

Define the `TextModelRequest` and `ImageModelRequest` models by inheriting `ModelRequest`. The optional temperature field by default is set to 0.0. The `num_inference_steps` field for the `ImageModelRequest` model is optional and set to 200. Both of these models will now require the prompt string field to be provided.

[![4](assets/4.png)](#co_implementing_type_safe_ai_services_CO4-4)

Define the `ImageModelResponse` and `TextModelResponse` models by inheriting the `ModelResponse` model. For `TextModelResponse`, provide the count of tokens, and with `ImageModelResponse`, provide an image size in pixels alongside the remote URL for downloading the image.

With the models shown in [Example 4-6](#pydantic_compound_models), you have schemas needed to define the requirements of your text and image generation endpoints.

## Field Constraints and Validators

Aside from support for standard types, Pydantic also ships with *constrained types* such as `EmailStr`, `PositiveInt`, `UUID4`, `AnyHttpUrl`, and more that can perform data validation out of the box during model initialization for common data formats. The full list of Pydantic types is available in [the official documentation](https://oreil.ly/xNbXX).

###### Note

Some constrained types such as `EmailStr` will require dependency packages to be installed to function but can be extremely useful for validating common data formats such as emails.

To define more custom and complex field constraints on top of Pydantic-constrained types, you can use the `Field` function from Pydantic with the `Annotated` type to introduce validation constraints such as a valid input range.

[Example 4-7](#pydantic_constrained_fields) replaces the standard type hints in [Example 4-6](#pydantic_compound_models) with constrained types and `Field` functions to implement stricter data requirements for your endpoints based on model constraints.

##### Example 4-7\. Using constrained fields

```py
# schemas.py

from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4
from pydantic import BaseModel, Field, HttpUrl, IPvAnyAddress, PositiveInt

class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=10000)] ![1](assets/1.png)

class ModelResponse(BaseModel):
    request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)] ![2](assets/2.png)
    # no defaults set for ip field
    # raise ValidationError if a valid IP address or None is not provided
    ip: Annotated[str, IPvAnyAddress] | None ![3](assets/3.png)
    content: Annotated[str | None, Field(min_length=0, max_length=10000)] ![4](assets/4.png)
    created_at: datetime = datetime.now()

class TextModelRequest(ModelRequest):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.0)] ![5](assets/5.png)

class TextModelResponse(ModelResponse):
    tokens: Annotated[int, Field(ge=0)]

ImageSize = Annotated[ ![6](assets/6.png)
    tuple[PositiveInt, PositiveInt], "Width and height of an image in pixels"
]

class ImageModelRequest(ModelRequest):
    model: Literal["tinysd", "sd1.5"]
    output_size: ImageSize ![6](assets/6.png)
    num_inference_steps: Annotated[int, Field(ge=0, le=2000)] = 200 ![7](assets/7.png)

class ImageModelResponse(ModelResponse):
    size: ImageSize ![6](assets/6.png)
    url: Annotated[str, HttpUrl] | None = None ![8](assets/8.png)
```

[![1](assets/1.png)](#co_implementing_type_safe_ai_services_CO5-1)

Replace the `str` standard type with `Field` and `Annotated` to bound the string length to a range of characters.

[![2](assets/2.png)](#co_implementing_type_safe_ai_services_CO5-2)

Generate a new request UUID by passing a callable to `default_factory` that will be called to generate a new UUID.

[![3](assets/3.png)](#co_implementing_type_safe_ai_services_CO5-3)

Constrain the optional `ip` field to any valid IPv4 or IPv6 address ranges. `None` is also a valid entry if the client’s IP can’t be determined. This optional field doesn’t have a default value, so if a valid IP or `None` is not provided, Pydantic will raise a `ValidationError`.

[![4](assets/4.png)](#co_implementing_type_safe_ai_services_CO5-4)

Constrain the `content` field to 10,000 characters or bytes.

[![5](assets/5.png)](#co_implementing_type_safe_ai_services_CO5-5)

Constrain the temperature between `0.0` and `1.0` with a default value of `0.0`.

[![6](assets/6.png)](#co_implementing_type_safe_ai_services_CO5-6)

Reuse an `Annotated` constrain on the `output_size` field to positive integers using the `PositiveInt` constrained type. The `lte` and `gte` keywords refer to *less than equal* and *greater than equal*, respectively.

[![7](assets/7.png)](#co_implementing_type_safe_ai_services_CO5-8)

Constrain the `num_inference_steps` field with `Field` between `0` and `2000` and a default of `200`.

[![8](assets/8.png)](#co_implementing_type_safe_ai_services_CO5-10)

Constrain the optional `url` field to any valid HTTP or HTTPS URL, where the hostname and top-level domain (TLD) are required.

With the models defined in [Example 4-7](#pydantic_constrained_fields), you can now perform validation on incoming or outgoing data to match the data requirements you have. In such cases, FastAPI will leverage Pydantic to automatically return error responses when data validation checks fail during a request runtime, as shown in [Example 4-8](#fastapi_pydantic_validation_failure).

##### Example 4-8\. FastAPI error response on data validation failure

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

## Custom Field and Model Validators

Another excellent feature of Pydantic for performing data validation checks is *custom field validators*. [Example 4-9](#field_validators) shows how both types of custom validators can be implemented on the `ImageModelRequest`.

##### Example 4-9\. Implementing custom field and model validators for `ImageModelRequest`

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

@validate_call ![1](assets/1.png)
def is_square_image(value: ImageSize) -> ImageSize: ![2](assets/2.png)
    if value[0] / value[1] != 1:
        raise ValueError("Only square images are supported")
    if value[0] not in [512, 1024]:
        raise ValueError(f"Invalid output size: {value} - expected 512 or 1024")
    return value

@validate_call ![1](assets/1.png)
def is_valid_inference_step(
    num_inference_steps: int, model: SupportedModels
) -> int:
    if model == "tinysd" and num_inference_steps > 2000: ![3](assets/3.png)
        raise ValueError(
            "TinySD model cannot have more than 2000 inference steps"
        )
    return num_inference_steps

OutputSize = Annotated[ImageSize, AfterValidator(is_square_image)] ![4](assets/4.png)
InferenceSteps = Annotated[ ![4](assets/4.png)
    int,
    AfterValidator(
        lambda v, values: is_valid_inference_step(v, values["model"])
    ),
]

class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]

class ImageModelRequest(ModelRequest):
    model: SupportedModels
    output_size: OutputSize ![5](assets/5.png)
    num_inference_steps: InferenceSteps = 200 ![6](assets/6.png)
```

[![1](assets/1.png)](#co_implementing_type_safe_ai_services_CO6-1)

In addition to static type checks, raise a runtime validation error if incorrect parameters have been passed to both the `is_square_image` and `is_valid_​infer⁠ence_step` functions.

[![2](assets/2.png)](#co_implementing_type_safe_ai_services_CO6-2)

The `tinysd` model can generate square images in certain sizes only. Asking for a nonsquare image size (an aspect ratio other than `1`) should raise a `ValueError`.

[![3](assets/3.png)](#co_implementing_type_safe_ai_services_CO6-4)

Raise a `ValueError` if the user asks for a large number of inference steps for the `tinysd` model.

[![4](assets/4.png)](#co_implementing_type_safe_ai_services_CO6-5)

Create reusable and more readable validators using the annotated pattern for both `OutputSize` and `InferenceSteps`.

[![5](assets/5.png)](#co_implementing_type_safe_ai_services_CO6-7)

Attach the `OutputSize` field validator to the `output_size` field to check for incorrect values after the model is initialized.

[![6](assets/6.png)](#co_implementing_type_safe_ai_services_CO6-8)

Attach the `InferenceSteps` validator to the `ImageModelRequest` model to perform checks on the model field values *after* the model is initialized.

With custom field validators, as shown in [Example 4-9](#field_validators), you can now be confident that your image generation endpoints will be protected from incorrect configurations provided by users.

###### Note

You can also use the decorator pattern to validate model fields. Special methods can be associated with model fields to execute conditional data checks by employing the `@field_validator` or `@model_validator` decorator.

While `@field_validator` accesses a value of a single field to perform checks, the `@model_validator` decorator allows for checks that involve multiple fields.

With `after` validators, you can perform extra checks or modify the data after Pydantic has completed its parsing and validation.

## Computed Fields

Similar to dataclasses, Pydantic also allows you to implement methods to compute fields derived from other fields.

You can use the `@computed_field` decorator to implement a computed field for calculating count of tokens and cost, as shown in [Example 4-10](#computed_fields).

##### Example 4-10\. Using computed fields to automatically count the total number of tokens

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

Computed fields are useful for encapsulating any field computation logic inside your Pydantic models to keep code organized. Bear in mind that computed fields are only accessible when you convert a Pydantic model to a dictionary using `.model_dump()` or via serialization when a FastAPI API handler returns a response.

## Model Export and Serialization

As Pydantic models can serialize to JSONs, the models you defined in [Example 4-7](#pydantic_constrained_fields) can also be dumped into (or be loaded from) JSON strings or Python dictionaries while maintaining any compound schemas, as shown in [Example 4-11](#model_export).

##### Example 4-11\. Exporting and serializing the `TextModelResponse` model

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

## Parsing Environment Variables with Pydantic

Alongside the `BaseModel`, Pydantic also implements a `Base` class for parsing settings and secrets from files. This feature is provided in an optional Pydantic package called `pydantic-settings`, which you can install as a dependency:

```py
$ pip install pydantic-settings
```

The `BaseSettings` class provides optional Pydantic features for loading a settings or config class from environment variables or secret files. Using this feature, the settings values can be set in code or overridden by environment variables.

This is useful in production where you don’t want to expose secrets inside the code or the container environment.

When you create a model inheriting from `BaseSettings`, the model initializer will attempt to set values of each field using provided defaults. If unsuccessful, the initializer will then read the values of any unset fields from the environment variables.

Given a dotenv environment file (ENV):

```py
APP_SECRET=asdlkajdlkajdklaslkldjkasldjkasdjaslk
DATABASE_URL=postgres://sa:password@localhost:5432/cms
CORS_WHITELIST=["https://xyz.azurewebsites.net","http://localhost:3000"]
```

An ENV is an environment variable file that can use a shell script syntax for key-value pairs.

[Example 4-12](#pydantic_settings) shows parsing environment variables using `BaseSettings` in action.

##### Example 4-12\. Using Pydantic `BaseSettings` to parse environment variables

```py
# settings.py

from typing import Annotated
from pydantic import Field, HttpUrl, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings): ![1](assets/1.png)
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8" ![2](assets/2.png)
    )

    port: Annotated[int, Field(default=8000)]
    app_secret: Annotated[str, Field(min_length=32)]
    pg_dsn: Annotated[
        PostgresDsn,
        Field(
            alias="DATABASE_URL",
            default="postgres://user:pass@localhost:5432/database",
        ),
    ] ![3](assets/3.png)
    cors_whitelist_domains: Annotated[
        set[HttpUrl],
        Field(alias="CORS_WHITELIST", default=["http://localhost:3000"]),
    ] ![4](assets/4.png)

settings = AppSettings()
print(settings.model_dump()) ![5](assets/5.png)
"""
{'port': 8000
 'app_secret': 'asdlkajdlkajdklaslkldjkasldjkasdjaslk',
 'pg_dsn': MultiHostUrl('postgres://sa:password@localhost:5432/cms'),
 'cors_whitelist_domains': {Url('http://localhost:3000/'),
                            Url('https://xyz.azurewebsites.net/')},
}
"""
```

[![1](assets/1.png)](#co_implementing_type_safe_ai_services_CO7-1)

Declare `AppSettings` inheriting from the `BaseSettings` class from the `pydantic_settings` package.

[![2](assets/2.png)](#co_implementing_type_safe_ai_services_CO7-2)

Configure `AppSettings` to read environment variables from the ENV file at the root of a project with the `UTF-8` encoding. By default, the snake_case field names will map to environment variables names that are an uppercase version of those names. For instance, `app_secret` becomes `APP_SECRET`.

[![3](assets/3.png)](#co_implementing_type_safe_ai_services_CO7-3)

Validate that the `DATABASE_URL` environment variable has a valid Postgres connection string format. If not provided, set the default value.

[![4](assets/4.png)](#co_implementing_type_safe_ai_services_CO7-4)

Check that the `CORS_WHITELIST` environment variable has a unique list of valid URLs with hostname and TLDs. If not provided, set the default to a set with a single value of `http://localhost:3000`.

[![5](assets/5.png)](#co_implementing_type_safe_ai_services_CO7-5)

We can check the `AppSettings` class is working by printing a dump of the model.

###### Note

You can switch environment files when using the `_env_file` argument:

```py
test_settings = AppSettings(_env_file="test.env")
```

## Dataclasses or Pydantic Models in FastAPI

Even though dataclasses support serialization of only the common types (e.g., `int`, `str`, `list`, etc.) and won’t perform field validation at runtime, FastAPI can still work with both Pydantic models and Python’s dataclasses. For field validation and additional features, you should use Pydantic models. [Example 4-13](#dataclass_fastaspi) shows how dataclasses can be used in FastAPI route handlers.

##### Example 4-13\. Using dataclasses in FastAPI

```py
# schemas.py

from dataclasses import dataclass
from typing import Literal

@dataclass
class TextModelRequest: ![1](assets/1.png)
    model: Literal["tinyLlama", "gemma2b"]
    prompt: str
    temperature: float

@dataclass
class TextModelResponse: ![1](assets/1.png)
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
) -> TextModelResponse: ![2](assets/2.png) ![4](assets/4.png)
    if body.model not in ["tinyLlama", "gemma2b"]: ![3](assets/3.png)
        raise HTTPException(
            detail=f"Model {body.model} is not supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    output = generate_text(models["text"], body.prompt, body.temperature)
    tokens = count_tokens(body.prompt) + count_tokens(output)
    return TextModelResponse(response=output, tokens=tokens) ![4](assets/4.png)
```

[![1](assets/1.png)](#co_implementing_type_safe_ai_services_CO8-1)

Define models for text model request and response schemas.

[![2](assets/2.png)](#co_implementing_type_safe_ai_services_CO8-3)

Convert the handler to serve `POST` requests with a body. Then, declare the request body as `TextModelRequest` and the response as `TextModelResponse`. Static code checkers like `mypy` will read the type annotations and raise warnings if your controller doesn’t return the expected response model.

[![3](assets/3.png)](#co_implementing_type_safe_ai_services_CO8-5)

Explicitly check whether the service supports the `model` parameter provided in the request `body`. If not, return a bad request HTTP exception response to the client.

[![4](assets/4.png)](#co_implementing_type_safe_ai_services_CO8-4)

FastAPI converts vanilla dataclasses to Pydantic dataclasses to serialize/deserialize and validate the request and response data.

In [Example 4-13](#dataclass_fastaspi), you have leveraged type annotations by refactoring the text model controller to be resilient to new changes and bad user input. Static type checkers can now help you catch any data-related issues as changes occur. In addition, FastAPI used your type annotations to validate request and responses alongside the auto-generation of an OpenAPI documentation page, as shown in [Figure 4-2](#fastapi_dataclasses_docs).

You now see that FastAPI leverages Pydantic models under the hood for data handling and validation, even if you use vanilla dataclasses. FastAPI converts your vanilla dataclasses to Pydantic-flavored dataclasses to use its data validation features. This behavior is intentional because if you have projects with several pre-existing dataclass type annotations, you can still migrate them over without having to rewrite them into Pydantic models for leveraging data validation features. However, if you’re starting a fresh project, it is recommended to use Pydantic models directly in replacement for Python’s built-in dataclasses.

![bgai 0402](assets/bgai_0402.png)

###### Figure 4-2\. Automatic generation of validation schemas using vanilla dataclasses

Now let’s see how you can replace dataclasses with Pydantic in your FastAPI application. See [Example 4-14](#pydantic_fastapi).

##### Example 4-14\. Using Pydantic to model request and response schemas

```py
# main.py

from fastapi import Body, FastAPI, HTTPException, Request, status
from models import generate_text
from schemas import TextModelRequest, TextModelResponse ![1](assets/1.png)

# load lifespan
...

app = FastAPI(lifespan=lifespan)

@app.post("/generate/text") ![2](assets/2.png)
def serve_text_to_text_controller(
    request: Request, body: TextModelRequest = Body(...)
) -> TextModelResponse:
    if body.model not in ["tinyLlama", "gemma2b"]: ![3](assets/3.png)
        raise HTTPException(
            detail=f"Model {body.model} is not supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    output = generate_text(models["text"], body.prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host) ![4](assets/4.png)
```

[![1](assets/1.png)](#co_implementing_type_safe_ai_services_CO9-1)

Import Pydantic models for text model request and response schemas.

[![2](assets/2.png)](#co_implementing_type_safe_ai_services_CO9-2)

Convert the handler to serve `POST` requests with a body. Then, declare the request body as `TextModelRequest` and the response as `TextModelResponse`. Static code checkers like `mypy` will read the type annotations and raise warnings if your controller doesn’t return the expected response model.

[![3](assets/3.png)](#co_implementing_type_safe_ai_services_CO9-3)

Explicitly check whether the service supports the `model` parameter provided in the request `body`. If not, return a bad request HTTP exception response to the client.

[![4](assets/4.png)](#co_implementing_type_safe_ai_services_CO9-4)

Return the `TextModelResponse` Pydantic model as per the function typing. Access the client’s IP address using the request object via `request.client.host`. FastAPI will take care of serializing your model using `.model_dump()` under the hood. As you also implemented the computed fields for `tokens` and `cost` properties, these will automatically will be included in your API response without any additional work.

###### Note

As shown in [Example 4-13](#dataclass_fastaspi), if you use dataclasses instead of Pydantic models, FastAPI will convert them to Pydantic dataclasses to serialize/deserialize and validate the request and response data. However, you may not be able to leverage advanced features such as field constraints and computed fields with dataclasses.

As you can see in [Example 4-14](#pydantic_fastapi), Pydantic can provide exceptional developer experience by helping in type checks, data validation, serialization, code editor auto-completions, and computed attributes.

FastAPI can also use your Pydantic models to auto-generate an OpenAPI specification and documentation page so that you can manually test your endpoints seamlessly.

Once you start the server, you should see an updated documentation page with the new Pydantic models and the updated constrained fields, as shown in [Figure 4-3](#fastapi_pydnatic_models).

![bgai 0403](assets/bgai_0403.png)

###### Figure 4-3\. Automatic generation of FastAPI docs using Pydantic models

If you send a request to the `/generate/text` endpoint, you should now see the prepopulated fields via the `TextModelResponse` Pydantic model, as shown in [Example 4-15](#fastapi_pydnatic_docs).

##### Example 4-15\. Automatic population of the response fields via the `TextModelResponse` Pydantic model

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

The Pydantic model features I covered in this chapter represent just a fraction of the tools at your disposal for constructing GenAI services. You should now feel more confident in leveraging Pydantic to annotate your own services to improve its reliability and your own developer experience.

# Summary

In this chapter, you learned the importance of creating fully typed services for GenAI models. You now understand how to implement type safety with standard and constrained types, how to use Pydantic models for data validation, and how to implement your own custom data validators across your GenAI service. You also discovered strategies for validating request and response content and managing application settings with Pydantic to prevent bugs and to improve your development experience. Overall, by following along with the practical examples, you learned how to implement a robust, less error-prone GenAI service.

The next chapter covers asynchronous programming in AI workloads, discussing performance and parallel operations. You will learn more about I/O-bound and CPU-bound tasks and understand the role and limitations of FastAPI’s background tasks with concurrent workflows.

^([1](ch04.html#id749-marker)) A [`Literal` type](https://oreil.ly/69Pmn) can be used to indicate to type checkers that the annotated object has a value equivalent to one of the provided literals.

^([2](ch04.html#id750-marker)) OpenAI’s `tiktoken` uses the [*Byte-Pair Encoding* (BPE) algorithm](https://oreil.ly/l67GS) to tokenize text. Different models use different encodings to convert text into tokens.

^([3](ch04.html#id762-marker)) Structs in C-like languages and dataclasses in Python can also be used to group and pass data around.