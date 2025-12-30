# 第七章\. 将数据库集成到 AI 服务中

在本章中，你将集成数据库到当前的 API 服务中，以存储和检索用户交互。

本章假设你具备使用数据库和结构化查询语言（SQL）的基本经验，因此它不会涵盖 SQL 编程和数据库工作流程的各个方面。相反，你将学习在将数据库集成到与 LLM 等 GenAI 模型交互的 FastAPI 应用程序时，更高层次的数据库概念、开发工作流程和最佳实践。

作为其中的一部分，你将了解关系型（SQL）数据库与无关系型（noSQL）数据库在应用程序开发中的作用，并能够自信地选择适合你用例的正确数据库。接下来，你将了解更多关于关系型数据库及其相关工具（如对象关系映射器（ORMs）和数据库迁移工具）的功能。最后，作为一个动手练习，你将使用 SQLAlchemy 和 Alembic 将数据库集成到现有的应用程序中，以存储和检索与 LLM 的用户对话。

到本章结束时，你将在选择、配置和解决 GenAI 应用程序中的数据库相关问题方面更有信心。

# 数据库的作用

在构建后端服务时，你通常需要一个数据库来持久化应用程序状态和存储用户数据。在其他情况下，你的应用程序可能不需要数据库，你不应该尝试添加一个，因为任何数据库集成都可能显著增加你服务的复杂性。

在以下几种情况下，你可以选择不使用数据库：

1.  你的应用程序可以在每个新的用户会话启动时从全新状态开始。

1.  重新计算应用程序数据是直接且资源高效的。

1.  应用程序数据足够小，可以存储在内存中。

1.  你的应用程序能够容忍由于服务器错误、重启或其他意外事件等原因导致的数据丢失。

1.  不同的用户会话或应用程序实例不需要共享数据。

1.  你所需的数据可以直接从外部系统、GenAI 模型和其他应用程序 API 中获取，而不是从你自己的数据库中获取。

1.  用户愿意等待为每个新的会话或操作重新计算数据。

1.  你的服务需求允许数据持久化存储在磁盘上的文件、浏览器存储或外部云存储，而不是数据库。使用这些替代方案，你的服务可以容忍数据存储和检索的可靠性不如数据库高效。

1.  你正在构建一个概念验证，需要不惜一切代价避免项目延迟或复杂性。

一个符合上述标准的应用程序示例是一个仅用于演示目的的*GenAI 图像生成器*。

在这个示例中，您不需要存储任何生成的图像，并且您始终可以从一个全新的状态重新启动或使用应用程序。此外，应用程序不需要知道用户是谁。此外，不需要在会话之间共享数据。此外，如果发生服务器错误，由于您可以即时重新生成新的图像，数据丢失的影响最小。

如您所见，至少有几种情况您不需要数据库来构建自己的 GenAI 服务。然而，您可能会想知道何时您确实需要数据库。

要确定何时需要数据库，您需要了解数据库的作用。简而言之，您可以使用它们以高效格式存储、组织和管理工作数据，从而便于检索、操作和分析。此外，数据库还配备了诸如恢复/备份、并发访问管理、索引、缓存和基于角色的访问控制等关键功能，以及许多其他功能，使它们成为任何显示、生成和消耗数据的服务的不可或缺的组成部分。

在下一节中，我们将以显著详细的程度检查数据库的内部运作机制，重点是关系型数据库，本章的实践示例将聚焦于此。通过对数据库内部结构的详细了解，您可以设计出完全优化且适用于生产的 GenAI API。这将使您能够将繁重的工作负载委托给专门为处理数据密集型任务而设计的数据库引擎。

# 数据库系统

既然您已经了解了何时可以利用数据库，那么让我们更深入地了解您可以使用的不同数据库以及它们的工作原理。

您可以通过将数据库分为两大类来构建数据库的心理模型：*关系型*（SQL）和*非关系型*（NoSQL）数据库。

*SQL*与*NoSQL*的分类基于这样一个事实：关系型数据库使用 SQL 的各种方言作为其主要查询语言，而非关系型数据库通常附带自己的专用查询语言。

对于这两种数据库系统类别，您可以采用一种心理模型来理解这些系统的结构。SQL 和 NoSQL 数据库系统通常由以下内容组成：

+   最高级别的*服务器*，它托管整个数据库基础设施并消耗系统资源（CPU、RAM 和存储）。

+   服务器内部的一个或多个*数据库*，它们作为*逻辑容器*，持有相关数据。

+   数据库内部的一个或多个*模式*（取决于数据库软件），它作为*蓝图*，定义了数据的完整结构和各种结构对象，如索引、逻辑约束、触发器等。然而，与关系型数据库不同，NoSQL 数据库服务器可能不会使用严格的模式。

+   在数据库内部（作为模式的一部分）创建的零个或多个*表*（SQL）或*集合*（NoSQL），它们将相关数据分组。

+   每个集合（作为文档）或表格（作为行）中可以有零个或多个*项目*，它们代表特定的记录或实体。

图 7-1 展示了上述分解。

![bgai 0701](img/bgai_0701.png)

###### 图 7-1\. 数据库系统分解

采用如图 7-1 所示的思维模型将帮助您在数据库系统种类不断增多的同时，能够预期到类似的底层机制存在。这种熟悉感有望减少您采用不同数据库系统时的学习曲线。

接下来，让我们简要回顾一下 SQL 和 NoSQL 数据库系统，以便您在构建 API 和服务时能更好地理解它们的使用场景、特性和局限性。

为了帮助您构建关联性和非关联性数据库的思维模型，请参阅图 7-2。

![bgai 0702](img/bgai_0702.png)

###### 图 7-2\. 数据库类型

您还可以使用表 7-1 中的摘要作为本章将要涵盖的数据库类型的参考。

表 7-1\. 数据库比较

| 类型 | 数据模型 | 示例 | 使用场景 |
| --- | --- | --- | --- |
| Key-value stores | 键值对 | Redis, DynamoDB, Memcached | 缓存，会话管理 |
| Graph stores | 节点和边 | Neo4j, Amazon Neptune, ArangoDB | 社交网络，推荐引擎，欺诈检测 |
| Document stores | 文档 | MongoDB, CouchDB, Amazon DocumentDB | 内容管理，电子商务，实时分析 |
| Vector stores | 高维向量 | Pinecone, Weaviate | 推荐系统，图像/文本搜索，机器学习模型存储 |
| Wide-column family stores | 表格，具有行和列 | Apache Cassandra, HBase, ScyllaDB | 时间序列数据，实时分析，日志记录 |

现在您已经对每个常见的关联性和非关联性数据库有了广泛的概述，您可以可视化一个利用这些数据库的实时世界 GenAI 服务。

想象一下，您正在构建一个支持 RAG 的 LLM 服务，它可以与知识库进行交互。这个知识库中的文档相互关联，因此您决定实现一个 RAG 图来捕捉更丰富的上下文。为了实现 RAG 图，您将您的服务与基于图的数据库集成。

现在，为了检索相关的文档块，您还需要将它们嵌入到向量数据库中。作为这一过程的一部分，您还需要一个关系数据库来监控使用情况，并存储用户数据和会话历史。

由于用户可能会提出常见问题，您还决定通过提前生成多个输出来缓存 LLM 的响应。因此，您还向您服务中集成了键值存储。

最后，你希望管理员能够通过版本控制提示的能力来控制系统提示。因此，你将内容管理系统作为提示管理器添加到你的解决方案中。然而，由于提示模板经常发生变化，你也决定集成一个文档数据库。

如你所见，每种数据库类型最终都会解决你复杂 RAG 启用应用程序中的特定问题。一个存储你的后端和用户数据，另一个捕获文档关系，一个存储你的文档嵌入，另一个帮助你存储提示的灵活模式，最后一个帮助你返回缓存输出。

你可以通过 图 7-3 中的可视化了解这些数据库如何协同工作以实现解决方案。

![bgai 0703](img/bgai_0703.png)

###### 图 7-3\. 使用各种数据库类型一起

现在你已经了解了你的 GenAI 服务如何与各种数据库集成，在下一节中，我们将专注于向你的服务添加一个关系型数据库。

# 项目：在关系型数据库中存储与 LLM 的用户对话

在上一节中，我们介绍了添加数据持久性到你的应用程序相关的核心数据库概念。

在本节中，你将集成一个关系型数据库到你的 GenAI 服务中，以便你可以在数据库中存储与 LLM 的用户对话历史。作为这项工作的一个部分，你还将学习管理数据库模式更改和数据迁移的最佳实践、工具和开发工作流程。

对于这个项目，我们将安装一个开源、免费且经过实战检验的 Postgres 关系型数据库，许多企业都在使用它。为了开始，让我们使用 `docker run` 命令下载并运行 Postgres 容器，如 示例 7-1 所示。

##### 示例 7-1\. 下载并运行 Postgres 数据库容器

```py
$ docker run -p 5432:5432  \  ![1](img/1.png) ![2](img/2.png)
	-e POSTGRES_USER=fastapi \
	-e POSTGRES_PASSWORD=mysecretpassword \
	-e POSTGRES_DB=backend_db \
	-e PGDATA=/var/lib/postgresql/data \ ![3](img/3.png)
    -v "$(pwd)"/dbstorage:/var/lib/postgresql/data \ ![4](img/4.png)
    postgres:latest ![1](img/1.png)
```

![1](img/1.png)（[将数据库集成到 AI 服务中](https://wiki.example.org/co_integrating_databases_into_ai_services_CO1-1)）

从 Docker 仓库下载并运行最新的 `postgres` 关系型数据库镜像。

![2](img/2.png)（[将数据库集成到 AI 服务中](https://wiki.example.org/co_integrating_databases_into_ai_services_CO1-2)）

运行 `postgres` 镜像，然后将容器端口 `5432` 映射到主机机器上的相同端口。

![3](img/3.png)（[将数据库集成到 AI 服务中](https://wiki.example.org/co_integrating_databases_into_ai_services_CO1-3)）

使用几个环境变量运行容器，这些变量指定了默认数据库管理员用户名和密码、数据库名称以及容器内的 DBMS 数据位置。

![4](img/4.png)（[将数据库集成到 AI 服务中](https://wiki.example.org/co_integrating_databases_into_ai_services_CO1-4)）

将 Postgres 数据库存储挂载到主机机器文件系统中的当前工作目录下的 `dbstorage` 文件夹。

接下来，让我们安装 `sqlalchemy`、`alembic` 和 `psycopg3` 包：

```py
$ pip install alembic sqlalchemy psycopg3
```

这些经过实战检验的包允许您通过 Python 直接与 Postgres 关系型数据库通信。`psycopg3`是 Python 中流行的 PostgreSQL 数据库适配器，而 SQLAlchemy 是一个 SQL 工具包和 ORM，允许您在 Python 中运行 SQL 查询。

最后，`alembic`包是由 SQLAlchemy 开发者创建的用于与 SQLAlchemy 一起使用的数据库迁移工具。数据迁移工作流程类似于 Git 版本控制系统，但针对您的数据库模式。它允许您管理对模式的更改和更新，从而避免任何数据损坏，跟踪随时间的变化，并在需要时撤销任何更改。

## 定义 ORM 模型

在 Python 中查询数据库的第一步是使用 SQLAlchemy 类定义您的 ORM 模型，如示例 7-2 所示。您可以使用图 8-4 中提到的 ERD 图中的数据模式。

###### 注意

在实现身份验证和授权机制时，您将在下一章中添加`user`表。

##### 示例 7-2\. 定义数据库 ORM 模型

```py
# entities.py

from datetime import UTC, datetime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase): ![1](img/1.png)
    pass

class Conversation(Base): ![2](img/2.png)
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column() ![3](img/3.png)
    model_type: Mapped[str] = mapped_column(index=True) ![4](img/4.png)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC) ![5](img/5.png)
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan" ![6](img/6.png)
    )

class Message(Base): ![7](img/7.png)
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True ![6](img/6.png)
    )
    prompt_content: Mapped[str] = mapped_column()
    response_content: Mapped[str] = mapped_column()
    prompt_tokens: Mapped[int | None] = mapped_column()
    response_tokens: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()
    is_success: Mapped[bool | None] = mapped_column()
    status_code: Mapped[int | None] = mapped_column() ![8](img/8.png) ![9](img/9.png)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )
```

![1](img/#co_integrating_databases_into_ai_services_CO2-1)

声明一个声明性基类，用于创建 SQLAlchemy 模型的 ORM 引擎。

![2](img/#co_integrating_databases_into_ai_services_CO2-2)

创建`Conversation`模型，指定表列、主键和二级索引。

![3](img/#co_integrating_databases_into_ai_services_CO2-3)

使用`mapped_column()`从`Mapped`提供的类型提示中推导列类型。

![4](img/#co_integrating_databases_into_ai_services_CO2-4)

如果您想根据模型类型更快地过滤会话，则对`model_type`进行索引。

![5](img/#co_integrating_databases_into_ai_services_CO2-5)

指定日期时间列的默认值和更新操作。

![6](img/#co_integrating_databases_into_ai_services_CO2-6)

指示如果通过`CASCADE DELETE`操作删除会话，则必须删除所有孤儿消息。

![7](img/#co_integrating_databases_into_ai_services_CO2-7)

创建`Message`模型，指定表列、主键、二级索引、表关系和外键。

![8](img/#co_integrating_databases_into_ai_services_CO2-9)

`messages`表将包含 LLM 提示和响应、使用令牌和成本，以及状态码和成功状态。

![9](img/#co_integrating_databases_into_ai_services_CO2-10)

将`Mapped[int | None]`指定为声明一个可选类型，这样列将允许`NULL`值（即，`nullable=True`）。

一旦您定义了数据模型，您就可以创建一个连接到数据库的连接，以使用指定的配置创建每个表。为了实现这一点，您需要创建一个数据库引擎并实现会话管理。

## 创建数据库引擎和会话管理

示例 7-3 展示了如何使用你的 Postgres 数据库连接字符串创建一个 SQLAlchemy 引擎。一旦创建，你可以使用该引擎和 `Base` 类为你的每个数据模型创建表。

###### 警告

示例 7-3 中的 SQLAlchemy 的 `create_all()` 方法只能在数据库中创建表，但不能修改现有表。这种工作流程仅在原型设计且你愿意在每次运行时用新表重置数据库模式时才有用。

对于生产环境，你应该使用像 `alembic` 这样的数据库迁移工具来更新你的数据库模式，以避免意外的数据丢失。你将很快了解数据库迁移工作流程。

##### 示例 7-3\. 创建 SQLAlchemy 数据库引擎

```py
# database.py

from sqlalchemy.ext.asyncio import create_async_engine
from entities import Base

database_url = ( ![1](img/1.png)
    "postgresql+psycopg://fastapi:mysecretpassword@localhost:5432/backend_db"
)
engine = create_async_engine(database_url, echo=True) ![2](img/2.png)

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all) ![3](img/3.png)

# main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from database import engine, init_db

@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    # other startup operations within the lifespan
    ...
    yield
    await engine.dispose() ![4](img/4.png)

app = FastAPI(lifespan=lifespan)
```

![1](img/#co_integrating_databases_into_ai_services_CO3-1)

对于 Postgres 数据库，连接字符串使用以下模板定义：`*<driver>*://*<username>*:*<password>*@*<origin>*/*<database>*`。

![2](img/#co_integrating_databases_into_ai_services_CO3-2)

使用数据库连接字符串创建一个异步数据库引擎。通过设置 `echo=True` 打开调试日志。

![3](img/#co_integrating_databases_into_ai_services_CO3-3)

使用 示例 7-3 中定义的 SQLAlchemy 模型删除任何现有表，然后创建所有数据库表。

![4](img/#co_integrating_databases_into_ai_services_CO3-4)

在服务器关闭过程中处理数据库引擎。在 FastAPI 的 `lifespan` 上下文管理器内部的 `yield` 关键字之后的任何代码，在请求服务器关闭时执行。

###### 警告

为了清晰起见，环境变量和如数据库连接字符串之类的机密信息在每个代码示例中都是硬编码的。

在生产场景中，永远不要硬编码机密信息和环境变量。利用环境文件、机密管理器和像 Pydantic Settings 这样的工具来处理应用程序的机密信息和变量。

在创建了引擎之后，你现在可以实施一个用于创建数据库会话的工厂函数。会话工厂是一种设计模式，它允许你在你的服务中打开、交互和关闭数据库连接。

由于你可能需要重用会话，你可以使用 FastAPI 的依赖注入系统来缓存和重用每个请求运行时期间的会话，如 示例 7-4 所示。

##### 示例 7-4\. 创建数据库会话 FastAPI 依赖项

```py
# database.py

from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from database import engine

async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession, autocommit=False, autoflush=False ![1](img/1.png)
)
async def get_db_session(): ![2](img/2.png)
    try:
        async with async_session() as session: ![3](img/3.png)
            yield session ![4](img/4.png)
    except:
        await session.rollback() ![5](img/5.png)
        raise
    finally:
        await session.close() ![6](img/6.png)

DBSessionDep = Annotated[AsyncSession, Depends(get_db_session)] ![7](img/7.png)
```

![1](img/#co_integrating_databases_into_ai_services_CO4-1)

创建一个与之前创建的数据库引擎绑定的异步数据库会话工厂，以异步连接到你的 Postgres 实例。使用 `autocommit=false` 禁用事务的自动提交，使用 `autoflush=False` 禁用更改自动刷新到数据库。禁用这两种行为可以给你更多的控制权，有助于防止意外的数据更新，并允许你实现更健壮的事务管理。

![2](img/2.png)[#co_integrating_databases_into_ai_services_CO4-2]

定义一个依赖函数，以便在您的 FastAPI 应用程序中将它重用和注入到路由控制器函数中。由于该函数在 `async with` 中使用了 `yield` 关键字，因此它被视为异步上下文管理器。当它作为依赖项使用时，FastAPI 会将其内部装饰为上下文管理器 `get_db_session`。

![3](img/3.png)[#co_integrating_databases_into_ai_services_CO4-3]

使用数据库会话工厂创建异步会话。上下文管理器有助于管理数据库会话的生命周期，例如在每个会话中打开、交互和关闭数据库连接。

![4](img/4.png)[#co_integrating_databases_into_ai_services_CO4-4]

将数据库会话传递给 `get_db_session` 函数的调用者。

![5](img/5.png)[#co_integrating_databases_into_ai_services_CO4-5]

如果有任何异常，请回滚事务并重新抛出异常。

![6](img/6.png)[#co_integrating_databases_into_ai_services_CO4-6]

在任何情况下，最后都要关闭数据库会话以释放它所持有的任何资源。

![7](img/7.png)[#co_integrating_databases_into_ai_services_CO4-7]

声明一个可跨不同控制器重用的带注释的数据库会话依赖项。

现在您可以通过依赖注入从任何 FastAPI 路由创建数据库会话，让我们为对话资源实现创建、读取、更新和删除（CRUD）端点。

## 实现 CRUD 端点

由于 FastAPI 依赖于 Pydantic 来序列化和验证传入和传出的数据，在实现 CRUD 端点之前，您需要将数据库实体映射到 Pydantic 模型。这避免了将您的 API 架构与数据库模型紧密耦合，从而为您在独立于彼此的情况下开发 API 和数据库提供了自由和灵活性。

您可以遵循示例 7-5 来定义您的 CRUD 模式。

##### 示例 7-5\. 声明对话端点的 Pydantic API 模式

```py
# schemas.py

from datetime import datetime
from pydantic import BaseModel, ConfigDict

class ConversationBase(BaseModel):
    model_config = ConfigDict(from_attributes=True) ![1](img/1.png)

    title: str
    model_type: str

class ConversationCreate(ConversationBase): ![2](img/2.png)
    pass

class ConversationUpdate(ConversationBase): ![2](img/2.png)
    pass

class ConversationOut(ConversationBase): ![2](img/2.png)
    id: int
    created_at: datetime
    updated_at: datetime
```

![1](img/1.png)[#co_integrating_databases_into_ai_services_CO5-1]

设置 Pydantic 模型以读取和验证其他模型（如 SQLAlchemy）的属性，这在处理数据库模型时在 Pydantic 中经常使用。

![2](img/2.png)[#co_integrating_databases_into_ai_services_CO5-2]

根据基础模型创建不同的 Pydantic 模型，以适应不同的用例，例如对话记录的创建和更新，或数据检索。

###### 小贴士

声明 Pydantic 和 SQLAlchemy 模型可能感觉像是代码重复，但它将允许您以您喜欢的任何方式实现数据访问层。

或者，如果你想避免任何代码重复，可以利用`sqlmodel`包，该包将 Pydantic 与 SQLAlchemy 集成，从而减少了大量的代码重复。然而，请注意，由于灵活性和对 SQLAlchemy 高级用例的支持有限，`sqlmodel`可能不适合生产环境。因此，你可能希望为复杂应用程序使用独立的 Pydantic 和 SQLAlchemy 模型。^(1)

现在你已经有了 SQLAlchemy 和 Pydantic 模型，你可以开始开发你的 CRUD API 端点。

在实现 CRUD 端点时，你应该尽可能利用 FastAPI 依赖项来减少数据库往返次数。例如，在检索、更新和删除记录时，你需要使用其 ID 检查数据库中是否存在记录。

你可以实现一个记录检索函数，在获取、更新和删除端点中使用依赖项，如示例 7-6 所示。

###### 警告

请记住，FastAPI 只能在单个请求内缓存`get_conversation`依赖项的输出，而不能跨多个请求。

##### 示例 7-6\. 为`conversations`表实现基于资源的 CRUD 端点

```py
# main.py

from typing import Annotated
from database import DBSessionDep
from entities import Conversation
from fastapi import Depends, FastAPI, HTTPException, status
from schemas import ConversationCreate, ConversationOut, ConversationUpdate
from sqlalchemy import select

...

async def get_conversation(
    conversation_id: int, session: DBSessionDep ![1](img/1.png)
) -> Conversation:
    async with session.begin(): ![2](img/2.png)
        result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalars().first()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conversation

GetConversationDep = Annotated[Conversation, Depends(get_conversation)]

@app.get("/conversations")
async def list_conversations_controller(
    session: DBSessionDep, skip: int = 0, take: int = 100
) -> list[ConversationOut]:
    async with session.begin():
        result = await session.execute(
            select(Conversation).offset(skip).limit(take) ![3](img/3.png)
        )
    return [
        ConversationOut.model_validate(conversation)
        for conversation in result.scalars().all()
    ]

@app.get("/conversations/{id}")
async def get_conversation_controller(
    conversation: GetConversationDep,
) -> ConversationOut:
    return ConversationOut.model_validate(conversation) ![4](img/4.png)

@app.post("/conversations", status_code=status.HTTP_201_CREATED)
async def create_conversation_controller(
    conversation: ConversationCreate, session: DBSessionDep
) -> ConversationOut:
    new_conversation = Conversation(**conversation.model_dump())
    async with session.begin():
        session.add(new_conversation)
        await session.commit() ![5](img/5.png)
        await session.refresh(new_conversation)
    return ConversationOut.model_validate(new_conversation)

@app.put("/conversations/{id}", status_code=status.HTTP_202_ACCEPTED)
async def update_conversation_controller(
    updated_conversation: ConversationUpdate,
    conversation: GetConversationDep,
    session: DBSessionDep,
) -> ConversationOut:
    for key, value in updated_conversation.model_dump().items():
        setattr(conversation, key, value)
    async with session.begin():
        await session.commit() ![5](img/5.png)
        await session.refresh(conversation)
    return ConversationOut.model_validate(conversation)

@app.delete("/conversations/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_controller(
    conversation: GetConversationDep, session: DBSessionDep
) -> None:
    async with session.begin():
        await session.delete(conversation)
        await session.commit() ![5](img/5.png)
```

![1](img/#co_integrating_databases_into_ai_services_CO6-1)

定义一个依赖项以检查会话记录是否存在。如果找不到记录，则引发 404 `HTTPException`；否则，返回检索到的记录。这个依赖项可以通过依赖注入在多个 CRUD 端点之间重用。

![2](img/#co_integrating_databases_into_ai_services_CO6-2)

在每个请求中，在异步上下文中开始异步会话。

![3](img/#co_integrating_databases_into_ai_services_CO6-3)

当列出记录时，只检索记录的子集会更有效率。默认情况下，SQLAlchemy ORM 返回数据库中最新的记录子集，但你可以使用`.offset(skip)`和`.limit(take)`链式方法检索任何记录子集。

![4](img/#co_integrating_databases_into_ai_services_CO6-4)

使用`model_validate()`从 SQLAlchemy 模型创建 Pydantic 模型。如果传递给 SQLAlchemy 的对象无法创建或未通过 Pydantic 的数据验证检查，则引发`ValidationError`。

![5](img/#co_integrating_databases_into_ai_services_CO6-5)

对于会修改记录的操作（即创建、更新和删除），提交事务然后将刷新后的记录发送给客户端，除了成功的删除操作应该返回`None`。

注意通过这种依赖注入方法如何简化了控制器逻辑。

此外，请注意您应该发送给客户端的成功状态码。成功的检索操作应返回 200，而记录创建操作返回 201，更新返回 202，删除返回 204。

恭喜！你现在拥有了一个基于资源的 RESTful API，你可以用它来对你的`conversations`表执行 CRUD 操作。

现在你已经可以实施 CRUD 端点，让我们重构现有的代码示例，以使用你在第二章中学到的*存储库和服务*设计模式。使用这种设计模式，你可以抽象数据库操作，以实现更模块化、可维护和可测试的代码库。

## 存储库和服务设计模式

*存储库*是一种设计模式，它协调应用程序的业务逻辑和数据库访问层——例如，通过 ORM。它包含执行数据库层 CRUD 操作的方法。

在第二章中，你首先看到了图 7-4，它展示了在处理数据库时，存储库在洋葱/分层应用架构中的位置。

![bgai 0704](img/bgai_0704.png)

###### 图 7-4. 在洋葱/分层应用架构中的存储库模式

要实现存储库模式，你可以使用一个*抽象接口*，它强制你以特定的方式定义你的存储库类，正如你在示例 7-7 中看到的那样。

###### 注意

如果你从未使用过*抽象*类，它们是那些不能独立实例化的类。抽象类可以包含没有实现的方法，其子类必须实现这些方法。

具体类是继承自抽象类并实现其所有抽象方法的类。

##### 示例 7-7. 实现存储库抽象接口

```py
# repositories/interfaces.py

from abc import ABC, abstractmethod
from typing import Any

class Repository(ABC): ![1](img/1.png)
    @abstractmethod
    async def list(self) -> list[Any]:
        pass

    @abstractmethod
    async def get(self, uid: int) -> Any:
        pass

    @abstractmethod
    async def create(self, record: Any) -> Any:
        pass

    @abstractmethod
    async def update(self, uid: int, record: Any) -> Any:
        pass

    @abstractmethod
    async def delete(self, uid: int) -> None:
        pass
```

![1](img/#co_integrating_databases_into_ai_services_CO7-1)

定义具有几个与 CRUD 相关的抽象方法签名的抽象`Repository`接口。如果具体子类中没有实现抽象方法，将引发`NotImplementedError`。

现在你有一个`Repository`类，你为每个表声明子类以定义必须遵循基于 CRUD 的方法来执行数据库操作。例如，要执行数据库中对话记录的 CRUD 操作，你可以实现一个具体的`ConversationRepository`类，如示例 7-8 所示。

##### 示例 7-8. 使用抽象存储库接口实现对话存储库

```py
# repositories/conversations.py

from entities import Conversation
from repositories.interfaces import Repository
from schemas import ConversationCreate, ConversationUpdate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

class ConversationRepository(Repository): ![1](img/1.png)
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self, skip: int, take: int) -> list[Conversation]:
        async with self.session.begin():
            result = await self.session.execute(
                select(Conversation).offset(skip).limit(take)
            )
        return [r for r in result.scalars().all()]

    async def get(self, conversation_id: int) -> Conversation | None:
        async with self.session.begin():
            result = await self.session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
        return result.scalars().first()

    async def create(self, conversation: ConversationCreate) -> Conversation:
        new_conversation = Conversation(**conversation.model_dump())
        async with self.session.begin():
            self.session.add(new_conversation)
            await self.session.commit()
            await self.session.refresh(new_conversation)
        return new_conversation

    async def update(
        self, conversation_id: int, updated_conversation: ConversationUpdate
    ) -> Conversation | None:
        conversation = await self.get(conversation_id)
        if not conversation:
            return None
        for key, value in updated_conversation.model_dump().items():
            setattr(conversation, key, value)
        async with self.session.begin():
            await self.session.commit()
            await self.session.refresh(conversation)
        return conversation

    async def delete(self, conversation_id: int) -> None:
        conversation = await self.get(conversation_id)
        if not conversation:
            return
        async with self.session.begin():
            await self.session.delete(conversation)
            await self.session.commit()
```

![1](img/#co_integrating_databases_into_ai_services_CO8-1)

继承抽象`Repository`接口并实现其每个方法，同时遵守方法签名。

你现在已经将对话的数据库逻辑移动到了`ConversationRepository`。这意味着你现在可以将其导入到你的路由控制器函数中并立即开始使用它。

回到你的`main.py`文件，重构你的路由控制器以使用`ConversationRepository`，如示例 7-9 所示。

##### 示例 7-9\. 将会话 CRUD 端点重构为使用存储库模式

```py
# routers/conversations.py

from typing import Annotated
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
...  # Other imports
from repositories import ConversationRepository

...  # Other controllers and dependency implementations

router = APIRouter(prefix="/conversations") ![1](img/1.png)

async def get_conversation(
    conversation_id: int, session: SessionDep
) -> Conversation:
    conversation = await ConversationRepository(session).get(conversation_id) ![2](img/2.png)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conversation

GetConversationDep = Annotated[Conversation, Depends(get_conversation)]

@router.get("")
async def list_conversations_controller(
    session: SessionDep, skip: int = 0, take: int = 100
) -> list[ConversationOut]:
    conversations = await ConversationRepository(session).list(skip, take)
    return [ConversationOut.model_validate(c) for c in conversations]

@router.get("/{id}")
async def get_conversation_controller(
    conversation: GetConversationDep,
) -> ConversationOut:
    return ConversationOut.model_validate(conversation) ![2](img/2.png)

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation_controller(
    conversation: ConversationCreate, session: SessionDep
) -> ConversationOut:
    new_conversation = await ConversationRepository(session).create(
        conversation
    ) ![2](img/2.png)
    return ConversationOut.model_validate(new_conversation)

@router.put("/{id}", status_code=status.HTTP_202_ACCEPTED)
async def update_conversation_controller(
    conversation: GetConversationDep,
    updated_conversation: ConversationUpdate,
    session: SessionDep,
) -> ConversationOut:
    updated_conversation = await ConversationRepository(session).update( ![2](img/2.png)
        conversation.id, updated_conversation
    )
    return ConversationOut.model_validate(updated_conversation)

@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_controller(
    conversation: GetConversationDep, session: SessionDep
) -> None:
    await ConversationRepository(session).delete(conversation.id)

# main.py

from routers.conversations import router as conversations_router

app.include_router(conversations_router) ![1](img/1.png)
```

![1](img/#co_integrating_databases_into_ai_services_CO9-1)

将会话 CRUD 路由放置在单独的 API 路由器上，并将其包含在 FastAPI 应用程序中，以实现模块化 API 设计。

![2](img/#co_integrating_databases_into_ai_services_CO9-2)

将会话 CRUD 路由重构为使用存储库模式，以实现更易读的控制器实现。

你是否注意到，自从数据库逻辑被抽象到 `ConversationRepository` 类中后，你的路由控制器看起来更加整洁了？

你可以将这种方法进一步发展，并实现一个服务模式。*服务*模式是存储库模式的扩展，它将业务逻辑和操作封装在更高层。这些高级操作通常需要更复杂的查询和一系列 CRUD 操作来执行业务逻辑。

例如，你可以实现一个 `ConversationService` 来获取与某个会话或特定用户相关的消息（参见示例 7-10）。由于它扩展了 `ConversationRepository`，你仍然可以访问底层的数据访问 CRUD 方法，如 `list`、`get`、`create`、`update` 和 `delete`。

再次回到你的控制器中，你可以将 `ConversationRepository` 的引用替换为 `ConversationService`。此外，你可以使用相同的服务来添加一个新端点，用于获取单个会话中的消息。

##### 示例 7-10\. 实现会话服务模式

```py
# services/conversations.py

from entities import Message
from repositories.conversations import ConversationRepository
from sqlalchemy import select

class ConversationService(ConversationRepository):
    async def list_messages(self, conversation_id: int) -> list[Message]:
        result = await self.session.execute(
            select(Message).where(Message.conversation_id == conversation_id)
        )
        return [m for m in result.scalars().all()]

# routers/conversations.py

from database import DBSessionDep
from entities import Message
from fastapi import APIRouter
from schemas import MessageOut
from services.conversations import ConversationService

router = APIRouter(prefix="/conversations")

@router.get("/{conversation_id}/messages") ![1](img/1.png)
async def list_conversation_messages_controller(
    conversation: GetConversationDep,
    session: DBSessionDep,
) -> list[Message]:
    messages = await ConversationService(session).list_messages(conversation.id)
    return [MessageOut.model_validate(m) for m in messages]
```

![1](img/#co_integrating_databases_into_ai_services_CO10-1)

添加一个新端点，使用会话 ID 列出会话的消息。

现在，你已经根据存储库和服务模式实现了与你的会话数据交互的完整 RESTful API。

###### 小贴士

现在你对存储库和服务模式更加熟悉了，你可以尝试为 `messages` 表实现 CRUD 端点。

当使用存储库和服务模式时，请注意避免将你的服务与特定的存储库实现紧密耦合，并不要让服务承担过多的责任。保持存储库专注于数据访问和操作，并避免在其中放置业务逻辑。

你还需要正确处理数据库事务和异常，尤其是在执行多个相关操作时。此外，考虑查询的性能影响，例如包含多个 JOIN，并在可能的情况下优化查询。

良好的做法是使用一致的命名约定为你的方法和类命名，并避免硬编码配置设置。

数据库开发工作流程中还有一个方面我们需要接下来解决。那就是管理不断变化的数据库模式，特别是在协作团队中，多个人在开发和生产环境中都在同一个数据库上工作。

# 管理数据库模式变更

你一定注意到了，在 示例 7-3 中，每次你启动 FastAPI 服务器时，你都会删除并重新创建你的数据库表。这在原型设计阶段的发展工作流程中是可以接受的，但在你需要将服务部署到具有活跃用户的实际生产环境中时则完全不适用。你每次更新数据库模式时都不能从头开始重置你的数据库。

你可能还需要一种方法来撤销更改，以防出现故障或你决定回滚某些功能。为了实现这一点，你可以使用 Alembic 这样的数据库迁移工具，它被设计成可以无缝地与 SQLAlchemy ORM 一起工作。

Alembic 允许你以与 Git 等工具帮助你版本控制代码相同的方式版本控制你的数据库模式。当你在具有多个应用程序环境的团队中工作时，它们非常有用，需要跟踪更改或根据需要撤销更新。

要开始，你必须首先通过 `pip` 安装 `alembic`，然后在 FastAPI 项目的根目录下运行 示例 7-11 来初始化它。

##### 示例 7-11\. 初始化 Alembic 环境

```py
$ alembic init
```

Alembic 将在其 `alembic` 文件夹内创建其环境，包括几个文件和一个 `versions` 目录，如图 示例 7-12 所示。

##### 示例 7-12\. 项目根目录内的 Alembic 环境

```py
project/
    alembic.ini
    alembic/
        env.py ![1](img/1.png) README
        script.py.mako
        versions/ ![2](img/2.png) <migration .py files will appear here>
```

![1](img/1.png)[#co_integrating_databases_into_ai_services_CO11-1]

用于指定目标模式和数据库连接的环境文件

![2](img/2.png)[#co_integrating_databases_into_ai_services_CO11-2]

一个用于存放 *迁移* 文件的目录，这些文件指定了如何更新或撤销数据库模式

一旦生成了 Alembic 环境，打开并修改位于 `alembic` 目录下的 *env.py* 文件，如图 示例 7-13 所示，以便它能够访问包含目标模式信息的 SQLAlchemy 元数据对象。

##### 示例 7-13\. 将 Alembic 环境与 SQLAlchemy 模型连接

```py
# alembic/env.py

from entities import Base
from settings import AppSettings

settings = AppSettings()
target_metadata = Base
db_url = str(settings.pg_dsn)

...
```

通过 Alembic 连接到你的 SQLAlchemy 模型后，Alembic 现在可以通过比较你的数据库当前模式与 SQLAlchemy 模型来自动生成迁移文件：

```py
$ alembic revision --autogenerate -m "Initial Migration"
```

此命令将比较定义的 SQLAlchemy 模型与现有数据库模式，并在 `alembic/versions` 目录下自动生成一个 SQL 迁移文件。

如果你打开生成的迁移文件，你应该会看到一个类似 示例 7-14 的文件内容。

##### 示例 7-14\. 初始 Alembic 迁移

```py
# alembic/versions/24c35f32b152.py

from datetime import UTC, datetime
import sqlalchemy as sa
from alembic import op

"""
Revision ID: 2413cf32b712 Revises:
Create Date: 2024-07-11 12:30:17.089406
"""

# revision identifiers, used by Alembic.
revision = "24c35f32b152"
down_revision = None
branch_labels = None

def upgrade():
    op.create_table(
        "conversations",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("model_type", sa.String, index=True, nullable=False),
        sa.Column(
            "created_at", sa.DateTime, default=datetime.now(UTC), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime,
            default=datetime.now(UTC),
            onupdate=datetime.now(UTC),
            nullable=False,
        ),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "conversation_id",
            sa.BigInteger,
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        ),
        sa.Column("prompt_content", sa.Text, nullable=False),
        sa.Column("response_content", sa.Text, nullable=False),
        sa.Column("prompt_tokens", sa.Integer, nullable=True),
        sa.Column("response_tokens", sa.Integer, nullable=True),
        sa.Column("total_tokens", sa.Integer, nullable=True),
        sa.Column("is_success", sa.Boolean, nullable=True),
        sa.Column("status_code", sa.Integer, nullable=True),
        sa.Column(
            "created_at", sa.DateTime, default=datetime.now(UTC), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime,
            default=datetime.now(UTC),
            onupdate=datetime.now(UTC),
            nullable=False,
        ),
    )

def downgrade():
    op.drop_table("messages")
    op.drop_table("conversations")
```

现在你已经更新了第一个迁移文件，你就可以运行它来对数据库进行操作：

```py
$ alembic upgrade head
```

如果您需要回滚操作，可以运行`alembic downgrade`。

Alembic 在底层执行的操作是生成运行或回滚迁移所需的原始 SQL，并在数据库中创建一个`alembic_versions`表。它使用这个表来跟踪已经应用在您的数据库上的迁移，这样再次运行`alembic upgrade head`命令就不会执行任何重复的迁移。

如果在任何情况下，您的数据库模式和迁移历史发生了偏移，您始终可以删除`versions`目录中的文件并截断`alembic_revision`表。然后重新初始化 Alembic，以全新的环境开始对现有数据库进行操作。

###### 警告

在使用迁移文件迁移数据库后，请确保将其提交到 Git 仓库。避免在迁移数据库后重新编辑迁移文件，因为 Alembic 将通过与其版本控制表交叉检查来跳过现有的迁移。

如果迁移文件已经运行，它将不会检测其内容的变化。

要更新数据库模式，请创建一个新的迁移文件。

按照上述工作流程，现在不仅可以版本控制数据库模式，还可以在应用程序需求变化时管理生产环境中的更改。

# 在处理实时流时存储数据

现在，您应该能够实现自己的 CRUD 端点，以检索和修改数据库中的用户对话和消息记录。

一个悬而未决的问题是，如何在数据流式传输端点中处理事务，例如 LLM 将流式输出到客户端。

您无法将数据流式传输到传统的关系型数据库，因为确保流式事务的 ACID 兼容性将是一项挑战。相反，您将希望尽快执行标准数据库操作，一旦 FastAPI 服务器向客户端返回响应。这正是 FastAPI 的后台任务可以解决的问题，正如您在示例 7-15 中看到的那样。

##### 示例 7-15\. 存储 LLM 输出流的内 容

```py
# main.py

from itertools import tee
from database import DBSessionDep
from entities import Message
from fastapi import BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from repositories.conversations import Conversation
from repositories.messages import MessageRepository
from sqlalchemy.ext.asyncio import AsyncSession

async def store_message( ![1](img/1.png)
    prompt_content: str,
    response_content: str,
    conversation_id: int,
    session: AsyncSession,
) -> None:
    message = Message(
        conversation_id=conversation_id,
        prompt_content=prompt_content,
        response_content=response_content,
    )
    await MessageRepository(session).create(message)

@app.get("/text/generate/stream")
async def stream_llm_controller(
    prompt: str,
    background_task: BackgroundTasks,
    session: DBSessionDep,
    conversation: Conversation = Depends(get_conversation), ![2](img/2.png)
) -> StreamingResponse:
    # Invoke LLM and obtain the response stream
    ...
    stream_1, stream_2 = tee(response_stream) ![3](img/3.png)
    background_task.add_task(
        store_message, prompt, "".join(stream_1), conversation.id, session
    ) ![4](img/4.png)
    return StreamingResponse(stream_2)
```

![1](img/1.png)[#co_integrating_databases_into_ai_services_CO12-1]

创建一个函数来存储与对话相关的消息。

![2](img/2.png)[#co_integrating_databases_into_ai_services_CO12-2]

检查对话记录是否存在，并在依赖关系中获取它。

![3](img/3.png)[#co_integrating_databases_into_ai_services_CO12-3]

创建 LLM 流的两个单独副本，一个用于`StreamingResponse`，另一个用于后台任务处理。

![4](img/4.png)[#co_integrating_databases_into_ai_services_CO12-4]

在`StreamingResponse`完成后创建一个后台任务来存储消息。

在示例 7-15 中，您允许 FastAPI 完全将 LLM 响应流式传输到客户端。

无论你使用的是 SSE 还是 WebSocket 端点，都不会有任何区别。一旦请求和响应完全流式传输，调用一个后台任务，传入完整的流式响应内容。在后台任务中，你可以在请求发送后运行一个函数来存储消息，并带有完整的 LLM 响应内容。

使用相同的方法，你甚至可以根据第一条消息的内容为对话生成一个标题。为此，你可以再次调用 LLM，使用对话中第一条消息的内容，请求为对话提供一个合适的标题。一旦生成了对话标题，你可以在数据库中创建对话记录，如示例 7-16 所示。

##### 示例 7-16\. 使用 LLM 根据初始用户提示生成对话标题

```py
from entities import Conversation
from openai import AsyncClient
from repositories.conversations import ConversationRepository
from sqlalchemy.ext.asyncio import AsyncSession

async_client = AsyncClient(...)

async def create_conversation(
    initial_prompt: str, session: AsyncSession
) -> Conversation:
    completion = await async_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Suggest a title for the conversation "
                           "based on the user prompt",
            },
            {
                "role": "user",
                "content": initial_prompt,
            },
        ],
        model="gpt-3.5-turbo",
    )
    title = completion.choices[0].message.content
    conversation = Conversation(
        title=title,
        # add other conversation properties
        ...
    )
    return await ConversationRepository(session).create(conversation)
```

在 FastAPI 中使用 SQLAlchemy 与 Alembic 是一种经过验证和测试的方法来处理关系型数据库，因此你更有可能找到很多关于集成这些技术的资源。

SQLAlchemy ORM 和 Alembic 都允许你与数据库交互并控制其模式的变化。

# 摘要

在本章中，你深入探讨了将数据库集成到你的 FastAPI 应用程序中以存储和检索用户对话的关键方面。

你学会了如何识别何时需要数据库以及如何为你的项目选择合适的数据库类型，无论是关系型还是非关系型。通过理解关系型数据库的底层机制和非关系型数据库的使用场景，你现在可以做出关于数据库选择的明智决策。

你还探索了与关系型数据库一起工作的开发工作流程、工具和最佳实践。这包括学习提高查询性能和效率的技术，以及管理数据库模式变化的策略。此外，你在团队工作中获得了关于管理代码库、数据库模式和数据漂移的见解。

随着你继续前进，下一章将指导你实现用户管理、身份验证和授权机制。这将进一步增强你的应用程序的安全性并提升用户体验，基于你在本章中建立的稳固数据库基础。

^(1) 请参阅此 [reddit 讨论线程](https://oreil.ly/OMaOT)。
