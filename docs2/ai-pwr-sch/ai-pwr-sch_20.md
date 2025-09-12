# 附录 B 支持的搜索引擎和向量数据库

我们在整本书中一致地使用开源的 Apache Solr 搜索引擎作为默认的搜索引擎，但代码库中的所有算法都设计为可以与各种搜索引擎和向量数据库一起工作。为此，除了需要特定引擎语法来展示某个点的情况外，我们已实现使用通用 `engine` 接口来搜索功能，这使得您能够轻松地替换您首选的搜索引擎或向量数据库。在本附录中，我们将介绍支持引擎的列表、如何替换默认引擎以及如何在整本书中使用主要抽象（`engine` 和 `collection`）。

## B.1 支持的引擎

支持的引擎列表将随着时间的推移而继续增长，但在出版时，以下引擎最初得到支持：

+   `solr`—Apache Solr

+   `opensearch`—OpenSearch

+   `bonsai`—Bonsai

+   `weaviate`—Weaviate

要查看所有支持引擎的完整、最新列表，请访问 [`aipoweredsearch.com/supported-engines`](https://aipoweredsearch.com/supported-engines)。

## B.2 替换引擎

通常，当运行书籍的代码示例时，您一次只会与一个搜索引擎或向量数据库一起工作。要使用任何特定的引擎，您只需在启动 Docker 时指定该引擎的名称（如上所述）即可。

例如，您可以通过以下方式启动 OpenSearch：

```py
docker compose up opensearch
```

这将启动运行 `opensearch`（或您指定的引擎）所需的任何必要的 Docker 容器，并将此引擎设置为书籍的 Jupyter 笔记本中的活动引擎，以便在所有代码示例中使用。

注意，一些引擎，如托管搜索和基于 API 的服务，不需要任何额外的本地 Docker 容器，因为它们的服务托管在其他地方。此外，一些引擎可能需要额外的配置参数，例如 API 密钥、远程地址/URL、端口等。这些参数可以在项目根目录的 .env 文件中设置。

如果您想在任何时候使用不同的引擎，您可以重新启动 Docker 容器并指定您想要使用的新引擎：

```py
docker compose up bonsai
```

如果您想同时启动多个引擎进行实验，您可以在 `docker compose up` 命令的末尾提供您希望启动的引擎列表：

```py
docker compose up solr opensearch weaviate
```

在您的 `docker compose up` 命令中引用的第一个引擎将被设置为 Jupyter 笔记本中的活动引擎，其他引擎则处于待机状态。

如果您想在您的实时 Jupyter 笔记本中切换到备用引擎之一（例如，切换到 `opensearch`），您可以通过在任何笔记本中运行以下命令在任何时候完成此操作：

```py
import aips
aips.set_engine("opensearch")
```

请记住，如果您为当前未运行的引擎调用 `set_engine`，那么当您尝试使用该引擎时，如果该引擎仍然不可用，这将会导致错误。

您也可以通过运行以下命令在任何时候检查当前设置的引擎：

```py
aips.get_engine().name
```

## B.3 引擎和集合抽象

搜索引擎行业充满了各种术语和概念，我们已经在代码库中尽可能地抽象掉了大部分。大多数搜索引擎最初都是从词汇关键词搜索开始的，后来增加了对向量搜索的支持，而许多向量数据库最初也是从向量搜索开始的，后来又增加了词汇搜索。就我们的目的而言，我们只是将这些系统视为*匹配和排名引擎*，并且我们使用术语*引擎*来指代所有这些系统。

同样，每个引擎都有一个一个或多个逻辑分区或容器来添加数据。在 Solr 和 Weaviate 中，这些容器被称为*collections*；在 OpenSearch、Elasticsearch 和 Redis 中，这些被称为*indexes*；在 Vespa 中，这些被称为*applications*。在 MongoDB 中，原始数据存储在*collection*中，但随后可以复制到*index*中进行搜索。命名在其他引擎中也有进一步的差异。

为了正确的抽象，我们在代码库中始终使用术语*collection*，因此每个实现的引擎都有一个`collection`接口，通过该接口可以查询或添加文档。

在`engine`接口上的常见公共方法包括

+   `engine.create_collection(collection_name)`—创建新的集合

+   `engine.get_collection(collection_name)`—返回现有的集合

在`collection`接口上的常见公共方法包括

+   `collection.search(**request)`—执行搜索并返回结果。单个请求参数应作为 Python 关键字参数传递，例如`collection.search(query="keyword"), limit=10`。

+   `collection.add_documents(docs)`—将一系列文档添加到集合中

+   `collection.write(dataframe)`—将 Spark dataframe 中的每一行写入集合作为文档

+   `collection.commit()`—确保最近添加的文档被持久化并可供搜索

`engine`和`collection`接口还内部实现了用于书中所有数据集的模式定义和管理。

由于`collection.write`方法接受一个 dataframe，我们在从 CSV 或 SQL 等额外数据源加载数据时需要根据需要使用辅助工具：

+   `collection.write(from_csv(csv_file))`—将 CSV 文件中的每一行写入集合作为文档

+   `collection.write(from_sql(sql_query))`—执行 SQL 查询并将返回的每一行写入集合作为文档

从这些额外数据源加载不需要额外的引擎特定实现，因为任何可以映射到 Spark dataframe 的数据源都隐式支持。

## B.4 添加对额外引擎的支持

虽然我们希望最终支持大多数主要搜索引擎和向量数据库，但您可能会发现您喜欢的引擎目前尚未得到支持。如果是这种情况，我们鼓励您为其添加支持，并向代码库提交一个拉取请求。`engine` 和 `collection` 接口被设计得易于实现，您可以使用默认的 `solr` 实现或任何其他已实现的引擎作为参考。

并非所有数据存储都完全支持 *AI-Powered Search* 中实现的所有功能。例如，纯向量数据库可能不支持词汇关键词匹配和排名，而某些搜索引擎可能不支持向量搜索。同样，某些专用功能可能仅在特定引擎中可用。

尽管默认的 `solr` 引擎支持书中实现的全部 *AI-Powered Search* 功能，但其他引擎可能需要采取变通方法、集成额外的库，或将某些功能委托给其他引擎以处理特定算法。例如，大多数引擎没有对语义知识图谱和文本标记的原生支持，因此许多引擎实现会将这些一次性功能委托给其他库。

我们希望 `engine` 和 `collection` 抽象能够让您轻松添加对您喜欢的引擎的支持，并可能将其贡献给本书的代码库，以惠及更广泛的 *AI-Powered Search* 读者和实践者。祝您搜索愉快！
