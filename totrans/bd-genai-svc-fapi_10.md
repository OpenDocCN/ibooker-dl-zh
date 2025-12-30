# Chapter 7\. Integrating Databases into AI Services

In this chapter, you’ll integrate a database to your current API service to store and retrieve user interactions.

This chapter assumes basic experience working with databases and Structured Query Language (SQL), so it won’t cover every aspect of SQL programming and database workflows. Instead, you will learn the higher-level database concepts, development workflows, and best practices when integrating databases to your FastAPI applications that interact with GenAI models such as LLMs.

As part of this, you will learn the role of relational (SQL) versus nonrelational (noSQL) databases in application development and will be able to confidently select the right database for your use case. Next, you will understand more about the features of relational databases and associated tooling such as object relational mappers (ORMs) and database migration tools. Finally, as a hands-on exercise, you will integrate a database to your existing application using SQLAlchemy and Alembic, to store and retrieve user conversations with an LLM.

By the end of this chapter, you will feel more confident in selecting, configuring, and resolving database-related issues within your GenAI applications.

# The Role of a Database

When building backend services, you often require a database to persist application state and store user data. In other cases, your application won’t need a database, and you shouldn’t try to add one since any database integration can significantly increase the complexity of your services.

Here are several cases for which you could forgo using a database:

1.  Your application can start from a fresh state on startup for each new user session.

2.  Recalculating the application data is straightforward and resource-efficient.

3.  The application data is small enough to be stored in memory.

4.  Your application is tolerant to data losses due to various reasons such as server errors, restarts, or other unexpected events.

5.  Different user sessions or application instances won’t need to share data.

6.  The data you need can directly be fetched from external systems, GenAI models, and other application APIs, and not your own database.

7.  The user is happy to wait for data to be recomputed for each new session or action.

8.  Your service requirements allow for the data to be persisted in files on disk, the browser storage, or an external cloud storage instead of a database. With these alternatives, your services can tolerate that data storage and retrieval won’t be as reliable and efficient as a database.

9.  You’re building a proof-of-concept and need to avoid project delays or complexity at all costs.

An example application that matches the previous criteria is a *GenAI image generator* only used for demonstration purposes.

In this example, you won’t need to store any generated images, and you can always restart or use the application at any time from a fresh state. Additionally, the application doesn’t need to know who the user is. Plus, there is no need to share data between sessions. Furthermore, if there is a server error, the impact of data loss is minimal since you can regenerate a new image on the fly.

As you can see, there are at least a handful of cases where you won’t need a database to build your own GenAI services. However, you may be wondering when you do really need a database.

To determine when a database is necessary, you will want to understand the role of databases. In short, you can use them to store, organize, and manage data in an efficient format allowing for easy retrieval, manipulation, and analysis. Additionally, databases ship with critical features such as restore/backup, concurrent access management, indexing, caching, and role-based access control, alongside many others, that make them an irreplaceable component of any services that displays, produces and consumes data.

In the next section, we will examine in significant detail the inner workings of databases, with an emphasis on relational databases that practical examples of this chapter will focus on. With a detailed understanding of database internals, you can then design fully optimized and production-ready GenAI APIs. This will then allow you to delegate heavy workloads to the database engine, which is specifically designed for data-heavy tasks.

# Database Systems

Now that you understand when to leverage a database, let’s learn more about different databases you can use and how they work.

You can construct a mental model of databases by placing them into two main categories: *relational* (SQL) and *nonrelational* (NoSQL) databases.

The *SQL* versus *NoSQL* categorization is based on the fact that relational databases use various dialects of SQL as their main query language, whereas nonrelational databases often come packaged with their own specialized query languages.

With both categories of database systems, you can adopt a mental model of how such systems are structured. Both SQL and NoSQL database systems often consist of the following:

*   A *server* at the highest level, which hosts the entire database infrastructure and consumes system resources (CPU, RAM, and storage).

*   One or more *databases* within the server, which act as *logical container(s)* that hold related data.

*   One or more *schemas* within a database (depending on the database software), which serve as a *blueprint* that defines the complete structure of the data and various structural objects such as indexes, logical constraints, triggers, etc. However, NoSQL database servers may not use strict schemas, unlike relational databases.

*   Zero or more *tables* (SQL) or *collections* (NoSQL) created inside the database (as part of a schema), which group related data.

*   Zero or more *items* within each collection (as documents) or table (as rows), which represent specific records or entities.

[Figure 7-1](#database_server_breakdown) visualizes the aforementioned breakdown.

![bgai 0701](assets/bgai_0701.png)

###### Figure 7-1\. Database system breakdown

Adopting a mental model as shown in [Figure 7-1](#database_server_breakdown) will help you navigate the ever-increasing variety of database systems as you can expect similar underlying mechanisms to be present. This familiarity will hopefully reduce your learning curve in adopting different database systems.

Next, let’s briefly review both SQL and NoSQL database systems so that you have a better understanding of their use cases, features, and limitations when building APIs and services.

To help you with creating a mental model of both relational and nonrelational databases, take a look at [Figure 7-2](#db_types).

![bgai 0702](assets/bgai_0702.png)

###### Figure 7-2\. Database types

You can also use the summary in [Table 7-1](#db_comparison) as a reference of the database types that will be covered in this chapter.

Table 7-1\. Comparison of databases

| Type | Data model | Examples | Use cases |
| --- | --- | --- | --- |
| Key-value stores | Key-value pairs | Redis, DynamoDB, Memcached | Caching, session management |
| Graph stores | Nodes and edges | Neo4j, Amazon Neptune, ArangoDB | Social networks, recommendation engines, fraud detection |
| Document stores | Documents | MongoDB, CouchDB, Amazon DocumentDB | Content management, e-commerce, real-time analytics |
| Vector stores | High-dimensional vectors | Pinecone, Weaviate | Recommendation systems, image/text search, ML model storage |
| Wide-column family stores | Tables with rows and columns | Apache Cassandra, HBase, ScyllaDB | Time-series data, real-time analytics, logging |

Now that you have a broad overview of every common relational and nonrelational database, you can visualize a real-world GenAI service that makes use of these databases together.

Imagine you’re building a RAG-enabled LLM service that can talk to a knowledge base. The documents in this knowledge base are related to each other, so you decide to implement a RAG graph to capture a richer context. To implement a RAG graph, you integrate your service with a graph-based database.

Now, to retrieve relevant chunks of documents, you also need to embed them in a vector database. As part of this, you also need a relational database to monitor usage, and store user data and conversation histories.

Since the users may ask common questions, you also decide to cache the LLM responses by generating several outputs in advance. Therefore, you also integrate a key-value store to your service.

Finally, you want to give administrators control over system prompts with the ability to version-control prompts. So, you add a content management system as a prompt manager to your solution. However, since the prompt templates can often change, you also decide to integrate a document database.

As you can see, each database type ends up solving a particular problem in your complex RAG-enabled application. One stores your backend and user data, another captures the document relationships, one stores your document embeddings, another helps store flexible schemas of your prompts, and the last one helps you to return cached outputs.

You can see a visualization of the application architecture in [Figure 7-3](#rag_graph_architecture) to understand how these databases can work together to realize a solution.

![bgai 0703](assets/bgai_0703.png)

###### Figure 7-3\. Using various database types together

Now that you understand how your GenAI services can integrate with a variety of databases, in the next section, we will focus on adding a relational database to your service.

# Project: Storing User Conversations with an LLM in a Relational Database

In the previous section, we covered the core database concepts relevant to adding data persistence to your applications.

In this section, you will integrate a relational database to your GenAI service so that you can store user conversation histories with an LLM in the database. As part of this work, you will also learn the best practices, tooling, and development workflow to manage schema changes and data migrations in your database.

For this project, we will install a Postgres relational database that is open source, free, and battle-tested and is in use by many enterprises. To get started, let’s download and run the Postgres container using `docker run`, as shown in [Example 7-1](#postgres).

##### Example 7-1\. Download and run the Postgres database container

```py
$ docker run -p 5432:5432  \  ![1](assets/1.png) ![2](assets/2.png)
	-e POSTGRES_USER=fastapi \
	-e POSTGRES_PASSWORD=mysecretpassword \
	-e POSTGRES_DB=backend_db \
	-e PGDATA=/var/lib/postgresql/data \ ![3](assets/3.png)
    -v "$(pwd)"/dbstorage:/var/lib/postgresql/data \ ![4](assets/4.png)
    postgres:latest ![1](assets/1.png)
```

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO1-1)

Download and run the latest `postgres` relational database image from in the Docker registry.

[![2](assets/2.png)](#co_integrating_databases_into_ai_services_CO1-2)

Run the `postgres` image and then expose and map container port `5432` to the same ports on the host machine.

[![3](assets/3.png)](#co_integrating_databases_into_ai_services_CO1-3)

Run the container with several environmental variables that specify the default database administrator username and password, database name, and DBMS data location within the container.

[![4](assets/4.png)](#co_integrating_databases_into_ai_services_CO1-4)

Mount the Postgres database storage to the host machine filesystem at a `dbstorage` folder in the present working directory.

Next, let’s install the `sqlalchemy`, `alembic`, and `psycopg3` packages:

```py
$ pip install alembic sqlalchemy psycopg3
```

These battle-tested packages allow you to directly communicate with the Postgres relational database via Python. `psycopg3` is a popular PostgreSQL database adapter for Python, and SQLAlchemy is SQL toolkit and ORM that allows you to run SQL queries against your database in Python.

Lastly, the `alembic` package is a *database migration tool* created by the SQLAlchemy developers for usage with the SQLAlchemy. The data migration workflow is like the Git version control system but for your database schemas. It allows you to manage the changes and updates to your schemas so that you avoid any data corruptions, track changes over time, and revert any changes as required.

## Defining ORM Models

The first step to query your database in Python is to define your ORM models with SQLAlchemy classes, as shown in [Example 7-2](#sqlalchemy_models). You can use the data schemas from the ERD diagram mentioned in [Figure 8-4](ch08.html#erd).

###### Note

You will add the `user` table in the next chapter when implementing authentication and authorization mechanisms.

##### Example 7-2\. Defining database ORM models

```py
# entities.py

from datetime import UTC, datetime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase): ![1](assets/1.png)
    pass

class Conversation(Base): ![2](assets/2.png)
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column() ![3](assets/3.png)
    model_type: Mapped[str] = mapped_column(index=True) ![4](assets/4.png)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC) ![5](assets/5.png)
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan" ![6](assets/6.png)
    )

class Message(Base): ![7](assets/7.png)
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True ![6](assets/6.png)
    )
    prompt_content: Mapped[str] = mapped_column()
    response_content: Mapped[str] = mapped_column()
    prompt_tokens: Mapped[int | None] = mapped_column()
    response_tokens: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()
    is_success: Mapped[bool | None] = mapped_column()
    status_code: Mapped[int | None] = mapped_column() ![8](assets/8.png) ![9](assets/9.png)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )
```

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO2-1)

Declare a declarative base class for creating SQLAlchemy models for its ORM engine.

[![2](assets/2.png)](#co_integrating_databases_into_ai_services_CO2-2)

Create the `Conversation` model specifying the table columns, primary key, and secondary indexes.

[![3](assets/3.png)](#co_integrating_databases_into_ai_services_CO2-3)

Use the `mapped_column()` to derive the column type from the type hint given to `Mapped`.

[![4](assets/4.png)](#co_integrating_databases_into_ai_services_CO2-4)

Index the `model_type` in case you want faster filtering of conversations by model type.

[![5](assets/5.png)](#co_integrating_databases_into_ai_services_CO2-5)

Specify defaults and update operations for datetime columns.

[![6](assets/6.png)](#co_integrating_databases_into_ai_services_CO2-6)

Indicate that all orphan messages must be deleted if a conversation is deleted through a `CASCADE DELETE` operation.

[![7](assets/7.png)](#co_integrating_databases_into_ai_services_CO2-7)

Create the `Message` model specifying the table columns, primary key, secondary indexes, table relationships, and foreign keys.

[![8](assets/8.png)](#co_integrating_databases_into_ai_services_CO2-9)

The `messages` table will contain both the LLM prompts and responses, usage tokens, and costs alongside the status codes and success states.

[![9](assets/9.png)](#co_integrating_databases_into_ai_services_CO2-10)

Specify `Mapped[int | None]` to declare an optional typing so the column will allow `NULL` values (i.e., `nullable=True`).

Once you have your data models defined, you can create a connection to the database to create each table with the specified configurations. To achieve this, you will need to create a *database engine* and implement *session management*.

## Creating a Database Engine and Session Management

[Example 7-3](#sqlalchemy_engine) shows how to create a SQLAlchemy engine using your Postgres database connection string. Once created, you can use the engine and the `Base` class to create tables for each of your data models.

###### Warning

The SQLAlchemy’s `create_all()` method in [Example 7-3](#sqlalchemy_engine) can only create tables in the database but not modify existing tables. This workflow is useful only if you’re prototyping and happy to reset the database schemas with new tables on each run.

For production environments, you should use a database migration tool such as `alembic` to update your database schemas and to avoid unintended data loss. You will learn about the database migration workflow shortly.

##### Example 7-3\. Create the SQLAlchemy database engine

```py
# database.py

from sqlalchemy.ext.asyncio import create_async_engine
from entities import Base

database_url = ( ![1](assets/1.png)
    "postgresql+psycopg://fastapi:mysecretpassword@localhost:5432/backend_db"
)
engine = create_async_engine(database_url, echo=True) ![2](assets/2.png)

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all) ![3](assets/3.png)

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
    await engine.dispose() ![4](assets/4.png)

app = FastAPI(lifespan=lifespan)
```

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO3-1)

For Postgres databases, the connection string is defined using the following template: `*<driver>*://*<username>*:*<password>*@*<origin>*/*<database>*`.

[![2](assets/2.png)](#co_integrating_databases_into_ai_services_CO3-2)

Create an async database engine using the database connection string. Turn on debug logging with `echo=True`.

[![3](assets/3.png)](#co_integrating_databases_into_ai_services_CO3-3)

Drop any existing tables and then create all database tables using the defined SQLAlchemy models in [Example 7-3](#sqlalchemy_engine).

[![4](assets/4.png)](#co_integrating_databases_into_ai_services_CO3-4)

Dispose of the database engine during the server shutdown process. Any code after the `yield` keyword inside the FastAPI’s `lifespan` context manager is executed when server shutdown is requested.

###### Warning

For clarity, environment variables and secrets such as database connection strings are hard-coded in every code example.

In production scenarios, never hard-code secrets and environment variables. Leverage environment files, secret managers, and tools like Pydantic Settings to handle application secrets and variables.

With the engine created, you can now implement a factory function for creating sessions to the database. Session factory is a design pattern that allows you to open, interact with, and close database connections across your services.

Since you may reuse a session, you can use FastAPI’s dependency injection system to cache and reuse sessions across each request runtime, as shown in [Example 7-4](#sqlalchemy_session).

##### Example 7-4\. Creating a database session FastAPI dependency

```py
# database.py

from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from database import engine

async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession, autocommit=False, autoflush=False ![1](assets/1.png)
)
async def get_db_session(): ![2](assets/2.png)
    try:
        async with async_session() as session: ![3](assets/3.png)
            yield session ![4](assets/4.png)
    except:
        await session.rollback() ![5](assets/5.png)
        raise
    finally:
        await session.close() ![6](assets/6.png)

DBSessionDep = Annotated[AsyncSession, Depends(get_db_session)] ![7](assets/7.png)
```

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO4-1)

Create an async database session factory bound to the database engine you created previously to asynchronously connect to your Postgres instance. Disable automatic committing of transactions with `autocommit=false` and automatic flushing of changes to the database with `autoflush=False`. Disabling both behaviors gives you more control, helps prevent unintended data updates, and allows you to implement more robust transaction management.

[![2](assets/2.png)](#co_integrating_databases_into_ai_services_CO4-2)

Define a dependency function to reuse and inject across your FastAPI app into route controller functions. Since the function uses the `yield` keyword within the `async with`, it is considered an async context manager. FastAPI will internally decorate the `get_db_session` as context manager when it is used as a dependency.

[![3](assets/3.png)](#co_integrating_databases_into_ai_services_CO4-3)

Use the database session factory to create an async session. The context manager helps to manage the database session lifecycle such as opening, interacting with, and closing the database connections in each session.

[![4](assets/4.png)](#co_integrating_databases_into_ai_services_CO4-4)

Yield the database session to the caller of the `get_db_session` function.

[![5](assets/5.png)](#co_integrating_databases_into_ai_services_CO4-5)

If there are any exceptions, roll back the transaction and reraise the exception.

[![6](assets/6.png)](#co_integrating_databases_into_ai_services_CO4-6)

In any case, close the database session at the end to release any resources that it holds.

[![7](assets/7.png)](#co_integrating_databases_into_ai_services_CO4-7)

Declare an annotated database session dependency that can be reused across different controllers.

Now that you can create a database session from any FastAPI route via dependency injection, let’s implement the create, read, update, and delete (CRUD) endpoints for the conversations resource.

## Implementing CRUD Endpoints

As FastAPI relies on Pydantic to serialize and validate incoming and outgoing data, before implementing CRUD endpoints, you’ll need to map database entities to Pydantic models. This avoids tightly coupling your API schema with your database models to give you the freedom and flexibility in developing your API and databases independent of each other.

You can follow [Example 7-5](#sqlalchemy_pydantic) to define your CRUD schemas.

##### Example 7-5\. Declaring Pydantic API schemas for conversation endpoints

```py
# schemas.py

from datetime import datetime
from pydantic import BaseModel, ConfigDict

class ConversationBase(BaseModel):
    model_config = ConfigDict(from_attributes=True) ![1](assets/1.png)

    title: str
    model_type: str

class ConversationCreate(ConversationBase): ![2](assets/2.png)
    pass

class ConversationUpdate(ConversationBase): ![2](assets/2.png)
    pass

class ConversationOut(ConversationBase): ![2](assets/2.png)
    id: int
    created_at: datetime
    updated_at: datetime
```

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO5-1)

Set up the Pydantic model to read and validate attributes of other models like SQLAlchemy, which is often used in Pydantic when working with database models.

[![2](assets/2.png)](#co_integrating_databases_into_ai_services_CO5-2)

Create separate Pydantic models based on the base model for different use cases such as conversation record creation and update, or data retrieval.

###### Tip

Having to declare Pydantic and SQLAlchemy models may feel like code duplication but will allow you to implement your data access layer however you like.

Alternatively, if you want to avoid any code duplication, you can leverage the `sqlmodel` package, which integrates Pydantic with SQLAlchemy, removing much of the code duplication. However, bear in mind that `sqlmodel` may not be ideal for production due to limited flexibility and support for advanced use cases with SQLAlchemy. Therefore, you may want to use separate Pydantic and SQLAlchemy models for complex applications.^([1](ch07.html#id951))

Now that you have the SQLAlchemy and Pydantic models, you can start developing your CRUD API endpoints.

When implementing CRUD endpoints, you should try to leverage FastAPI dependencies as much as you can to reduce database round-trips. For instance, when retrieving, updating, and deleting records, you need to check in with the database that a record exists using its ID.

You can implement a record retrieval function to use a dependency across your get, update, and delete endpoints, as shown in [Example 7-6](#sqlalchemy_endpoints).

###### Warning

Bear in mind that FastAPI can only cache the output of the `get_conversation` dependency within a single request and not across multiple requests.

##### Example 7-6\. Implementing resource-based CRUD endpoints for the `conversations` table

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
    conversation_id: int, session: DBSessionDep ![1](assets/1.png)
) -> Conversation:
    async with session.begin(): ![2](assets/2.png)
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
            select(Conversation).offset(skip).limit(take) ![3](assets/3.png)
        )
    return [
        ConversationOut.model_validate(conversation)
        for conversation in result.scalars().all()
    ]

@app.get("/conversations/{id}")
async def get_conversation_controller(
    conversation: GetConversationDep,
) -> ConversationOut:
    return ConversationOut.model_validate(conversation) ![4](assets/4.png)

@app.post("/conversations", status_code=status.HTTP_201_CREATED)
async def create_conversation_controller(
    conversation: ConversationCreate, session: DBSessionDep
) -> ConversationOut:
    new_conversation = Conversation(**conversation.model_dump())
    async with session.begin():
        session.add(new_conversation)
        await session.commit() ![5](assets/5.png)
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
        await session.commit() ![5](assets/5.png)
        await session.refresh(conversation)
    return ConversationOut.model_validate(conversation)

@app.delete("/conversations/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_controller(
    conversation: GetConversationDep, session: DBSessionDep
) -> None:
    async with session.begin():
        await session.delete(conversation)
        await session.commit() ![5](assets/5.png)
```

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO6-1)

Define a dependency to check if the conversation record exists. Raise a 404 `HTTPException` if a record is not found; otherwise, return the retrieved record. This dependency can be reused across several CRUD endpoints through dependency injection.

[![2](assets/2.png)](#co_integrating_databases_into_ai_services_CO6-2)

Begin the async session within an async context manager during each request.

[![3](assets/3.png)](#co_integrating_databases_into_ai_services_CO6-3)

When listing records, it’s more efficient to retrieve only a subset of records. By default, SQLAlchemy ORM returns a subset of most recent records in the database, but you can use the `.offset(skip)` and `.limit(take)` chained methods to retrieve any subset of records.

[![4](assets/4.png)](#co_integrating_databases_into_ai_services_CO6-4)

Create a Pydantic model from a SQLAlchemy model using `model_validate()`. Raises a `ValidationError` if the SQLAlchemy object passed can’t be created or doesn’t pass Pydantic’s data validation checks.

[![5](assets/5.png)](#co_integrating_databases_into_ai_services_CO6-5)

For operations that mutate a record (i.e., create, update, and delete), commit the transaction then send the refreshed record to the client, except for the successful delete operation that should return `None`.

Notice how the controller logic is simplified through this dependency injection approach.

Additionally, pay attention to success status codes you should to send to the client. Successful retrieval operations should return 200, while record creation operations return 201, updates return 202, and deletions return 204.

Congratulations! You now have a resource-based RESTful API that you can use to perform CRUD operations on your `conversations` table.

Now that you can implement CRUD endpoints, let’s refactor the existing code examples to use the *repository and services* design pattern you learned about in [Chapter 2](ch02.html#ch02). With this design pattern, you can abstract the database operations to achieve a more modular, maintainable, and testable codebase.

## Repository and Services Design Pattern

A *repository* is a design pattern that mediates the business logic of your application and the database access layer—for instance, via an ORM. It contains several methods for performing CRUD operations in the database layer.

In [Chapter 2](ch02.html#ch02), you first saw [Figure 7-4](#onion), which showed where the repositories sit within the onion/layered application architecture when working with a database.

![bgai 0704](assets/bgai_0704.png)

###### Figure 7-4\. The repository pattern within the onion/layered application architecture

To implement a repository pattern, you can use an *abstract interface*, which enforces certain constraints on how you define your specific repository classes as you can see in [Example 7-7](#sqlalchemy_repository).

###### Note

If you’ve never used *abstract* classes, they’re classes that can’t be instantiated on their own. Abstract classes can contain methods without implementation that its subclasses must implement.

A concrete class is one that inherits an abstract class and implements each of its abstract methods.

##### Example 7-7\. Implementing a repository abstract interface

```py
# repositories/interfaces.py

from abc import ABC, abstractmethod
from typing import Any

class Repository(ABC): ![1](assets/1.png)
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

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO7-1)

Define the abstract `Repository` interface with several CRUD-related abstract method signatures that subclasses must implement. If an abstract method is not implemented in a concrete subclass, a `NotImplementedError` will be raised.

Now that you have a `Repository` class, you declare subclasses for each of your tables to define how database operations must be performed following the CRUD-based methods. For instance, to perform CRUD operations on the conversation records in the database, you can implement a concrete `ConversationRepository` class, as shown in [Example 7-8](#sqlalchemy_conversation_repository).

##### Example 7-8\. Implementing the conversation repository using the abstract repository interface

```py
# repositories/conversations.py

from entities import Conversation
from repositories.interfaces import Repository
from schemas import ConversationCreate, ConversationUpdate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

class ConversationRepository(Repository): ![1](assets/1.png)
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

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO8-1)

Inherit the abstract `Repository` interface and implement each of its methods while adhering to the method signatures.

You have now moved the database logic for conversations into the `ConversationRepository`. This means you can now import this class into your route controller functions and start using it right away.

Go back to your `main.py` file and refactor your route controllers to use the `ConversationRepository`, as shown in [Example 7-9](#sqlalchemy_conversation_repository_endpoints).

##### Example 7-9\. Refactoring the conversation CRUD endpoints to use the repository pattern

```py
# routers/conversations.py

from typing import Annotated
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
...  # Other imports
from repositories import ConversationRepository

...  # Other controllers and dependency implementations

router = APIRouter(prefix="/conversations") ![1](assets/1.png)

async def get_conversation(
    conversation_id: int, session: SessionDep
) -> Conversation:
    conversation = await ConversationRepository(session).get(conversation_id) ![2](assets/2.png)
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
    return ConversationOut.model_validate(conversation) ![2](assets/2.png)

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation_controller(
    conversation: ConversationCreate, session: SessionDep
) -> ConversationOut:
    new_conversation = await ConversationRepository(session).create(
        conversation
    ) ![2](assets/2.png)
    return ConversationOut.model_validate(new_conversation)

@router.put("/{id}", status_code=status.HTTP_202_ACCEPTED)
async def update_conversation_controller(
    conversation: GetConversationDep,
    updated_conversation: ConversationUpdate,
    session: SessionDep,
) -> ConversationOut:
    updated_conversation = await ConversationRepository(session).update( ![2](assets/2.png)
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

app.include_router(conversations_router) ![1](assets/1.png)
```

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO9-1)

Place conversation CRUD routes on a separate API router and include on the FastAPI application for modular API design.

[![2](assets/2.png)](#co_integrating_databases_into_ai_services_CO9-2)

Refactor conversation CRUD routes to use the repository pattern for more readable controller implementation.

Do you notice how cleaner your route controllers appear now that the database logic has been abstracted within the `ConversationRepository` class?

You can take this approach one step further and implement a service pattern as well. A *service* pattern is an extension of the repository pattern that encapsulates the business logic and operations in a higher layer. These higher-level operations often require more complex queries and a sequence of CRUD operations to be performed to implement the business logic.

As an example, you can implement a `ConversationService` to fetch messages related to a conversation or a specific user (see [Example 7-10](#sqlalchemy_conversation_service)). Since it extends a `ConversationRepository`, you can still access the lower-level data access CRUD methods such as `list`, `get`, `create`, `update`, and `delete`.

Once again you can go back to your controllers and replace references to the `ConversationRepository` with the `ConversationService` instead. Additionally, you can use the same service to add a new endpoint for fetching messages within a single conversation.

##### Example 7-10\. Implementing the conversation services pattern

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

@router.get("/{conversation_id}/messages") ![1](assets/1.png)
async def list_conversation_messages_controller(
    conversation: GetConversationDep,
    session: DBSessionDep,
) -> list[Message]:
    messages = await ConversationService(session).list_messages(conversation.id)
    return [MessageOut.model_validate(m) for m in messages]
```

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO10-1)

Add a new endpoint to list messages of a conversation using the conversation ID.

You now have a fully working RESTful API for interacting with your conversation data following the repository and service patterns.

###### Tip

Now that you’re more familiar with the repository and services pattern, you can try implementing CRUD endpoints for the `messages` table.

When using the repository and service patterns, be mindful that you avoid tightly coupling your services to specific repository implementation and not overload your services with many responsibilities. Keep repositories focused on data access and manipulation and avoid placing business logic in them.

You’ll also need to handle database transactions and exceptions properly, especially when performing multiple related operations. Also, consider performance implications of your queries such as including many JOINs, and optimize queries where you can.

Good practice is to use consistent naming conventions for your methods and classes and to avoid hard-coding configuration settings.

There is one more aspect of the database development workflow that we need to address next. That is managing ever-changing database schemas, in particular in collaborative teams where multiple people are working on the same database both in development and production environments.

# Managing Database Schemas Changes

You must have noticed that in [Example 7-3](#sqlalchemy_engine) you are deleting and re-creating your database tables every time you start your FastAPI server. This is acceptable for development workflows during the prototyping stage, but not at all when you need to deploy your services in production with active users. You can’t reset your database from scratch every time you update your database schema.

You also will probably need a way to revert changes if something breaks or if you decide to roll back certain features. To achieve this, you can use a database migration tool such as Alembic that is designed to work seamlessly with the SQLAlchemy ORM.

Alembic allows you to version control your database schemas the same way that tools like Git can help you version control your code. They’re extremely useful when you’re working in a team with multiple application environments and need to keep track of changes or revert updates as needed.

To get started, you must first install `alembic` via `pip` and then initialize it by running [Example 7-11](#alembic_init) at the root of your FastAPI project.

##### Example 7-11\. Initializing an Alembic environment

```py
$ alembic init
```

Alembic will create its environment within the `alembic` folder with several files and a `versions` directory, as shown in [Example 7-12](#alembic_dir).

##### Example 7-12\. Alembic environment within your project root directory

```py
project/
    alembic.ini
    alembic/
        env.py ![1](assets/1.png) README
        script.py.mako
        versions/ ![2](assets/2.png) <migration .py files will appear here>
```

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO11-1)

An environment file for specifying target schema and database connections

[![2](assets/2.png)](#co_integrating_databases_into_ai_services_CO11-2)

A directory for holding *migrations* files, which specify the instructions on how to update or revert the database schema

Once the Alembic environment is generated, open and modify the *env.py* file located in the `alembic` directory, as shown in [Example 7-13](#alembic_env), so that it gets access to your SQLAlchemy metadata object that contains the target schema information.

##### Example 7-13\. Connect the Alembic environment with your SQLAlchemy models

```py
# alembic/env.py

from entities import Base
from settings import AppSettings

settings = AppSettings()
target_metadata = Base
db_url = str(settings.pg_dsn)

...
```

With Alembic connected to your SQLAlchemy models, Alembic can now auto-generate your migration files by comparing the current schema of your database with your SQLAlchemy models:

```py
$ alembic revision --autogenerate -m "Initial Migration"
```

This command will compare the defined SQLAlchemy models against the existing database schema and automatically generate a SQL migration file under the `alembic/versions` directory.

If you open the generated migration file, you should see a file content similar to [Example 7-14](#alembic_initial_migration).

##### Example 7-14\. The initial Alembic migration

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

Now that you’ve updated your first migration file, you’re ready to run it against the database:

```py
$ alembic upgrade head
```

If your ever need to revert the operation, you can run `alembic downgrade` instead.

What Alembic does under the hood is to generate the raw SQL needed to run or revert a migration and create an `alembic_versions` table in the database. It uses this table to keep track of migrations that have already been applied on your database so that rerunning the `alembic upgrade head` command won’t perform any duplicate migrations.

If in any case, your database schemas and your migration history drift away, you can always remove files from the `versions` directory and truncate the `alembic_revision` table. Then reinitialize Alembic to start with a fresh environment against an existing database.

###### Warning

After migrating a database with a migration file, make sure to commit to a Git repository. Avoid re-editing migration files after migrating a database as Alembic will skip existing migrations by cross-checking them with its versioning table.

If a migration file has already been run, it won’t detect changes in its content.

To update your database schema, create a new migration file instead.

Following the aforementioned workflow will now allow you to not only version control your database schemas but also manage changes to your production environments as your application requirements change.

# Storing Data When Working with Real-Time Streams

You should now be in a position to implement your own CRUD endpoints to retrieve and mutate both user conversation and message records in your database.

One question that remains unanswered is how to handle transactions within data streaming endpoints, such as an LLM streaming outputs to a client.

You can’t stream data into a traditional relational database as ensuring ACID compliance with streaming transactions will prove challenging. Instead, you will want to perform your standard database operation as soon as your FastAPI server returns a response to the client. This challenge is exactly what a FastAPI’s background task can solve, as you can see in [Example 7-15](#db_stream).

##### Example 7-15\. Storing content of an LLM output stream

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

async def store_message( ![1](assets/1.png)
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
    conversation: Conversation = Depends(get_conversation), ![2](assets/2.png)
) -> StreamingResponse:
    # Invoke LLM and obtain the response stream
    ...
    stream_1, stream_2 = tee(response_stream) ![3](assets/3.png)
    background_task.add_task(
        store_message, prompt, "".join(stream_1), conversation.id, session
    ) ![4](assets/4.png)
    return StreamingResponse(stream_2)
```

[![1](assets/1.png)](#co_integrating_databases_into_ai_services_CO12-1)

Create a function to store a message against a conversation.

[![2](assets/2.png)](#co_integrating_databases_into_ai_services_CO12-2)

Check that the conversation record exists and fetch it within a dependency.

[![3](assets/3.png)](#co_integrating_databases_into_ai_services_CO12-3)

Create two separate copies of the LLM stream, one for the `StreamingResponse` and another to process in a background task.

[![4](assets/4.png)](#co_integrating_databases_into_ai_services_CO12-4)

Create a background task to store the message after the `StreamingResponse` is finished.

In [Example 7-15](#db_stream), you allow FastAPI to fully stream the LLM response to the client.

It won’t matter whether you’re using an SSE or WebSocket endpoint. Once a request a response is fully streamed, invoke a background task passing in the full stream response content. Within the background task, you can then run a function to store the message after the request is sent, with the full LLM response content.

Using the same approach, you can even generate a title for a conversation based on the content of the first message. To do this, you can invoke the LLM again with the content of the first message in the conversation, requesting for an appropriate title for the conversation. Once a conversation title is generated, you can create the conversation record in the database, as shown in [Example 7-16](#conversation_title).

##### Example 7-16\. Using the LLM to generate conversation titles based on the initial user prompt

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

Using SQLAlchemy with Alembic is a tried and tested approach to working with relational databases in FastAPI, so you’re more likely to find a lot of resources on integrating these technologies.

Both the SQLAlchemy ORM and Alembic allow you to interact with your database and control the changes to its schemas.

# Summary

In this chapter, you dove into the critical aspects of integrating a database into your FastAPI application to store and retrieve user conversations.

You learned to identify when a database is necessary and how to identify the appropriate type for your project, whether it be relational or nonrelational. By understanding the underlying mechanisms of relational databases and the use cases for nonrelational databases, you’re now equipped to make informed decisions about database selection.

You also explored the development workflow, tooling, and best practices for working with relational databases. This includes learning techniques to improve query performance and efficiency, as well as strategies for managing evolving database schema changes. Additionally, you gained insights into managing codebase, database schema, and data drifts when working in teams.

As you move forward, the next chapter will guide you through implementing user management, authentication, and authorization mechanisms. This will further enhance your application’s security and user experience, building on the solid database foundation you’ve established in this chapter.

^([1](ch07.html#id951-marker)) Refer to this [reddit discussion thread](https://oreil.ly/OMaOT).