# 5 存储我们的 ACH 文件

### 本章涵盖

+   在我们的 PostgreSQL 数据库中创建表

+   设计一个能够存储 ACH 文件的数据库

+   使用 Python 和 Pydantic 验证 ACH 记录并将它们存储在我们的数据库中

+   确保通过实现使用 pytest 的单元测试来正确解析和存储我们的记录

在这个冲刺中，我们使用另一个研究尖峰来探索如何定义我们的数据库。数据库在应用程序实例之间存储和持久化我们的数据，同时提供一种查询和确保数据完整性的方式。在这里，我们检查如何在数据库中存储我们的 ACH 文件。经过初步分析后，我们扩展了我们的 API 以在数据库中存储 ACH 文件。沿着这个方向继续，我们还扩展了我们的 ACH 解析器以存储单个字段。最后，我们通过检查存储 ACH 数据如何影响我们的单元和负载测试来结束本章。

在我们的项目中引入数据库是必要的，因为 ACH 文件是一个平面文件。Futuristic FinTech 目前使用的 ACH 系统依赖于平面文件，它们在许多领域都存在挑战，包括性能、查询和数据完整性。例如，如果客户质疑交易何时被加载，必须加载并解析所有 ACH 文件以执行搜索，这非常耗时。此外，考虑到处理的记录数量，将解析后的 ACH 文件保留在内存中变得不可行。

## 5.1 设计我们的数据库

当用户将文件上传到我们的 ACH 仪表板时，我们显然需要保存它们的能力，否则我们的系统将不会很有用。Futuristic FinTech 当前的 ACH 仪表板不使用关系型数据库。相反，一旦上传，文件就会被解析并存储在平面文件中（即非结构化文本文件），这使得实现更复杂的功能变得繁琐。我们正在替换的 ACH 仪表板仅使用文件系统来存储文件。为了提供更高级的处理，我们希望我们的 ACH 仪表板由关系型数据库支持，并且我们通过初步审查和实施各种数据库设计和概念来支持我们的仪表板。通常，我们需要在我们的冲刺中包含这些类型的研究故事，以检查我们在实现所需功能时可能采取的不同方式。

至少有十几种不同的关系型数据库，FinTech 使用了许多种。我们选择的数据库通常已经由我们公司使用的数据库决定。我们见过 FinTech 使用 Oracle、MySQL/MariaDB 和 PostgreSQL——仅举几个例子。在我们的案例中，我们已经设置了一个环境，使得 PostgreSQL 可以在 Docker 容器中运行，我们也看到了如何在启动时构建/初始化表以及如何通过 CloudBeaver 查看我们的数据。现在我们可以开始扩展我们的数据库以容纳存储 ACH 文件。

数据库不仅仅只是存储我们的数据——它们可以帮助确保数据的可靠性和一致性，这个概念被称为引用完整性。在我们的数据库中，引用完整性是一个复杂的说法，意味着我们将确保我们的表适当地相关联，字段定义正确。例如，回想一下 ACH 是一个固定长度格式，这意味着单个字段是固定长度的。我们可能将文件头记录中的文件 ID 修改者存储为`VARCHAR(1)`，因为它只能是一个单个大写字母（或 0 到 9）。同样，我们可能希望将文件控制（即文件尾记录）中的总借方条目金额存储为`MONEY`或`NUMERIC(12,2)`。`NUMERIC(12,2)`定义了一个精度为 12 位有效数字和 2 位小数位的字段，这是小数位数。你将使用`MONEY`还是`NUMERIC`取决于你，但我们更喜欢`NUMERIC(12,2)`表示，因为它与字段定义非常相似。

另一个引用完整性的方面是防止孤立记录。记住，在 ACH 文件中存在记录的层次结构，即文件控制 → 批次头 → 条目记录 → 等。例如，如果我们没有仔细定义我们的数据库，删除一个批次头记录可能会导致孤立记录。一旦我们删除了批次头，属于该批次的全部条目和附加记录（以及批次控制记录）就不再有效，我们应该删除它们。同样，删除文件控制记录应该删除与该文件关联的所有记录。在我们的关系数据库中，我们可以通过创建一个`FOREIGN` `KEY`（它引用其他表条目）并使用`ON DELETE` `CASCADE`来实现这一点。

我们最初的数据库将利用关系数据库的固有优势，通过定义以下内容：

+   *主键**s*—表中每条记录的唯一标识符

+   *外键**s*—两个表之间的链接，其中一个表中的字段（或字段）引用另一个表中的唯一数据（如主键）

+   *字段**s**的约束—例如，`NOT` `NULL`（确保数据存在），`UNIQUE`（确保数据在所有行中是唯一的），以及`DEFAULT`（如果没有提供，则分配默认值）

+   *数据完整性*—通过定义适当的数据类型和字段大小来获得

我们首先将查看仅存储预安排支付和存款（PPD）数据的 ACH 文件，以使事情更简单。PPD 代码通常用于直接存款和定期支付，如工资和养老金，因此它是一个广泛使用的代码，可能会经常影响你（而你却不知道）。为了了解我们的数据库将看起来像什么，我们再次依赖 PlantUML 来渲染一个建议的数据库结构。

##### 列表 5.1  我们数据库的 PlantUML 定义

```py
@startuml #1

object ach_file_uploads { #2
    ach_files_id: UUID   #3
    filename: VARCHAR(255)  #3
    file_hash: VARCHAR(32)  #3
    credit_total: NUMERIC(12,2)  #3
    debit_total: NUMERIC(12,2) 
}

object ach_files {
    ach_files_id: UUID
    record_type: VARCHAR(1)
    record_id: UUID
    parsed: BOOLEAN    
    sequence: NUMERIC
    unparsed_record: VARCHAR(94)
}

object ach_file_header_records {
    record_id: UUID
    fields for record type
}

object ach_batch_header_records {
    record_id: UUID
    file_header_id: UUID
    fields for record type  
}

object ach_entry_detail_ppd_records {
    record_id: UUID
    batch_header_id: UUID
    fields for record type
}

object ach_addenda_ppd_records {
    record_id: UUID
    entry_detail_id: UUID
    fields for record type
}

object ach_batch_control_records {
    record_id: UUID
    batch_header_id: UUID
    fields for record type
}

object ach_file_control_records {
    record_id: UUID
    file_header_id: UUID
    fields for record type
}

ach_file_uploads::ach_files_id <-- ach_files::ach_files_id  #4
 #4
ach_files::record_id <-- ach_file_header_records::record_id #4
ach_files::record_id <-- ach_batch_header_records::record_id #4
ach_files::record_id <-- ach_entry_detail_ppd_records::record_id #4
ach_files::record_id <-- ach_addenda_ppd_records::record_id #4
ach_files::record_id <-- ach_batch_control_records::record_id #4
ach_files::record_id <-- ach_file_control_records::record_id #4
 #4
ach_batch_header_records::file_header_id -> 
ach_file_header_records::record_id #4
ach_entry_detail_ppd_records::batch_header_id -> 
ach_batch_header_records::record_id #4
ach_addenda_ppd_records::entry_detail_id -> 
ach_entry_detail_ppd_records::record_id #4
ach_batch_control_records::batch_header_id --> 
ach_batch_header_records::record_id #4
ach_file_control_records::file_header_id --> 
ach_file_header_records::record_id #4
 #4
@enduml  #4#5
```

#1 开始一个 PlantUML 定义

#2 定义我们的图中的表

#3 定义表中的字段以供我们的图使用

#4 显示表中键之间的关系

#5 结束 PlantUML 定义

前面的定义在图 5.1 中展示，它显示了我们可以如何定义我们的字段以及我们表格之间的关系。这不是表格中字段的详尽列表，但它给了我们一个关于我们的表格将如何关联的想法。箭头表示表中存在的外键约束。例如，我们可以看到 `ach_files` 表中的 `ach_files_id` 字段被定义为通用唯一标识符（UUID），并引用了 `ach_file_uploads` 中的 `ach_files_id`。

![计算机图示 自动生成描述](img/CH05_F01_Kardell.png)

##### 图 5.1 显示了我们表格之间关系的图

图 5.1 也传达了我们希望实现以下目标：

+   维护我们文件中记录的顺序

+   假设记录将无法解析，并为此准备无法解析的记录

+   通过使解析记录引用未解析的记录来维护引用完整性

虽然数据库结构看似满足了这些目标，且图表为我们提供了开始工作的视觉指南，但总有改进的空间。无论这个结构是由主题专家（SME）提供还是由数据库分析师（DBA）的解释，我们在项目进行过程中都可能有机会改进我们的工作。手握我们的图表，我们应该有一个关于数据库外观的想法，并可以开始着手工作。然而，我们需要遵循定义测试、构建表/字段，最后构建 API 的一般模式。当与 SQL 数据库一起工作时，重要的是要理解公司肯定会有不同的方法来管理他们的数据。一些公司可能会将 SQL 部分从开发者那里提取出来，无论是通过使用 SQLAlchemy 这样的对象关系映射器（ORM）还是通过自己开发。ORM 通过抽象数据库并提供如数据库无关性、优化和提升生产力等好处来简化代码。

其他公司可能要求你自己编写 SQL，因为他们喜欢直接 SQL 提供的控制级别。ORM 可能会使复杂查询变得困难或低效。此外，调试查询和性能问题也可能更难追踪。首先，我们在这里展示如何使用我们一直在使用的直接 SQL 命令，然后转向 SQLAlchemy，这样你就可以熟悉这两种方法。无论我们使用哪种方法，或者将它们结合起来，总有几个因素会被添加。通常，现有软件将决定我们的方法，所以请注意不要一开始就偏离现有软件标准太远，因为这可能会造成维护噩梦。

## 5.2 直接使用 SQL

第三章探讨了解析 ACH 文件并创建相应的单元测试。我们还创建了一个简单的 API，该 API 访问数据库并返回结果。因此，我们拥有了将 ACH 文件存储在数据库中的所有必要组件。现在我们只需要将它们组合起来。在我们的第一个方法中，我们更新了解析器以将 ACH 文件存储在数据库中。我们做出以下假设：

+   数据库始终处于运行状态。

+   我们没有想要保留的数据。

+   我们正在从 IDE 中运行我们的代码，而不是在 Docker 内部。

换句话说，我们只是在处理能够解析和存储数据库中记录的过程。我们扩展了之前的`AchFileProcessor`以允许它存储 ACH 文件。下面的列表显示了添加必要的代码以获取数据库连接。由于我们处于研究冲刺阶段，我们硬编码了数据库连接参数，如用户名和密码。稍后，当我们确定这是正确的方法时，我们可以开始抽象一些这些硬编码的值，以便通过使用`环境变量`或秘密管理工具（如 AWS Secrets Manager [`aws.amazon.com/secrets-manager/`](https://aws.amazon.com/secrets-manager/)）或 HashiCorp Vault ([`www.hashicorp.com/products/vault`](https://www.hashicorp.com/products/vault)）来获得更灵活的配置。

##### 列表 5.2 添加数据库字段

```py
class AchFileProcessor:
    records = []
    exceptions = []
    last_trace_number = None
    expected_record_types = ["1"]
 POSTGRES_USER = "someuser" #1
 POSTGRES_PASSWORD = "supersecret" 
 DATABASE_URL = f"dbname={POSTGRES_USER} user={POSTGRES_USER}
➥ password={POSTGRES_PASSWORD} host=postgres port=5432" #2
…    
 def get_db(): #3
 conn = psycopg.connect(self.DATABASE_URL)
 return conn
```

#1 暂时硬编码用户名和密码

#2 使用硬编码的主机和端口，但我们也应该考虑将这些参数参数化。

#3 一个新函数，用于返回数据库连接

这应该为我们提供连接到数据库的能力；然而，我们将需要使用此代码。在我们开始解析 ACH 文件之前，我们只想将未解析的记录存储在数据库中，这可能是一个我们可能想要考虑的提取

–加载-转换（ELT）方法而不是直接通过提取-转换-加载（ETL）方法解析记录。

##### ELT 与 ETL

在处理数据时，有两种处理数据的方法。它们通常与数据仓库和商业智能一起讨论，但 ACH 处理具有独特的挑战。ETL（提取-转换-加载）是一种传统方法，在处理数据时可能会首先出现在我们的脑海中。例如，我们知道我们想要将每个 ACH 记录解析到它们各自的字段中，并将它们存储在数据库中。然而，处理可能不是始终正确格式化的数据是 ACH 处理中的一个挑战。使用 ETL 方法，这些无效数据可能会完全阻止处理。

在 ELT（提取-加载-转换）方法中，数据被加载然后在使用前进行转换。通常，我们在处理非常大的数据和使用如 Snowflake 或 Redshift 这样的数据仓库时看到 ELT，这些数据仓库有足够的处理能力来按请求执行转换。

那么，我们为什么关心这些方法呢？通常，金融机构会对数据进行一些宽容，并且如果错误被认为是可恢复的，它们不会总是拒绝一个文件。这些条件可能因金融机构而异，以及它们的客户。例如，如果控制文件上的记录计数不正确，可能会更新而不是简单地拒绝文件或批次。或者，字段中的无效数据在加载之前可能只是更新为空格。虽然我们可以依赖日志和异常来跟踪更改（我们应该记录任何此类更改），但我们仍然想保留原始文件的记录，以便银行家有审查数据的机会。

### 5.2.1 将记录添加到 ach_files 表

我们将在解析任何记录之前，将未修改的 ACH 文件加载到数据库中。图 5.2 阐述了本节的基本流程。

![](img/CH05_F02_Kardell.png)

##### 图 5.2 5.2 节的流程图

列表 5.3 显示了一个 `ach_files` 表。因此，在解析任何记录之前，我们只需将所有记录添加到该数据库表中。

##### 列表 5.3 简单的 `ach_files` 表

```py
-- Create the uuid extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; #1

-- Create the ach_files table
CREATE TABLE ach_files (
    ach_files_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(), #2
    file_name VARCHAR(255) NOT NULL, 
    unparsed_record VARCHAR(94) NOT NULL, #3
    sequence_number INTEGER NOT NULL #4
);
```

#1 允许 PostgreSQL 创建 UUIDs

#2 主要 UUID 键

#3 以原样存储未解析的 ACH 记录

#4 用于在检索数据时保持排序顺序的序列号

使用此代码，运行 `docker-compose` `down`，`docker-compose` `build` 和 `docker-compose` `up` 应该允许我们的表被创建，现在我们只需更新代码以写入一些记录！

在我们写出任何记录之前，我们必须确保基本功能正常。因此，我们只需在现有的解析例程开头添加以下代码。此代码仅获取数据库连接，获取一个可以用来执行 SQL 语句的光标，然后关闭光标和连接。

##### 列表 5.4  测试连接

```py
    def parse(self, filename) -> [List, List]:

 conn = self.get_db() #1
 cursor = conn.cursor() #2
 cursor.close() #3
 conn.close() #4

        with open(filename, "r") as file:
            lines = file.readlines()
```

#1 调用我们的新函数 get_db() 连接到数据库

#2 创建一个用于执行 SQL 命令的光标

#3 关闭光标

#4 关闭连接

如果我们现在运行代码，我们可能会遇到几个问题。首先，我们硬编码的 `DATABASE_URL` `字符串` 有一个 `postgres` 主机和一个 `5432` 端口。我们正在 IDE 中工作，而不是在 Docker 中，所以 `postgres` 不是应该使用的正确名称。实际上，如果我们运行程序，我们应该看到一个错误 `psycopg.OperationalError:` `connection` `is bad:` `Unknown` `host`。相反，我们想使用 `localhost`，因为我们是从托管 Docker 的系统调用。

此外，我们还需要公开我们容器的端口。我们的 docker-compose.yml 应该看起来像列表 5.5 中的那样。如果不公开端口，我们会看到一个类似于 `psycopg.OperationalError:` `connection` `failed:` `:1),` `port` `5432` `failed:` `could` `not` `receive` `data` `from` `server:` `Socket` `is` `not` `connected` 的错误，这可能会让我们意识到我们的端口有问题。记住，这些类型的问题总是会出现在我们的开发中，我们只需要回溯我们的步骤，看看我们错过了什么。

##### 列表 5.5  更新的 docker-compose.yml 文件

```py
  postgres:
    build: 
      context: ./db
      dockerfile: Dockerfile
 ports:
 - 5432:5432 #1
    env_file:
      - ./.sql_server.conf
```

#1 揭示了标准的 PostgreSQL 端口

一旦我们有一个基本的连接工作，我们就可以开始编写记录。我们可以使用类似于列表 5.4 中的方法来手动处理打开和关闭连接；然而，记住要关闭连接可能会出错，因为我们可能会忘记关闭它们（我们已经在生产环境中看到文件多年未关闭，直到迁移到新版本的软件或另一个供应商时才出现）。我们将使用 Python 的 `with` 语句，因为它在退出特定代码块时会自动处理各种资源的关闭。实际上，我们已经在使用它来读取我们的文件了，所以我们可以简单地在此基础上扩展。

##### 列表 5.6  更新的解析函数

```py
    def parse(self, filename) -> [List, List]:

        with open(filename, "r") as file, self.get_db() as conn: #1
            lines = file.readlines()
            sequence_number = 0  #2

            for line in lines:
                sequence_number += 1 #3
                line = line.replace("\n", "")

                with conn.cursor() as cursor: #4
 cursor.execute(f"INSERT INTO ach_files #4
➥ (file_name, unparsed_record, sequence_number) #4
➥ VALUES (%s, %s, %s)", (filename, line, sequence_number)) #4
 conn.commit() 
```

#1 作为现有 with 语句的一部分创建连接

#2 将我们的 sequence_number 初始化为零

#3 为每条记录递增 sequence_number

#4 使用 with 语句创建游标，插入并提交记录

我们可以重新运行我们的单元测试 `test_parsing_ach_file.py`，这将运行我们的样本文件通过我们的代码，然后检查 CloudBeaver 以验证记录是否已添加。这是一个好的开始：我们可以将记录存储在我们的数据库中，并且使用类似的方法进入我们的单个解析函数并存储数据应该不会太困难。

我们需要做的一件事是更新我们的单元测试，从数据库中获取数据而不是依赖于返回的数据，因为我们的目标是存储所有数据在数据库中，而不是返回任何其他状态。现在，让我们看看如何更新 `pytest` 来从数据库中获取记录数。

##### 列表 5.7  更新的 `pytest`

```py
import os
import psycopg
import pytest

from ach_processor.AchFileProcessor import AchFileProcessor

POSTGRES_USER = "someuser" #1
POSTGRES_PASSWORD = "supersecret"             #1
 #1
DATABASE_URL = f"dbname={POSTGRES_USER} user={POSTGRES_USER}  #1
➥password={POSTGRES_PASSWORD} host=localhost port=5432"  #1
 #1
def get_db():                              #1
 conn = psycopg.connect(DATABASE_URL) #1
 return conn #1

def test_record_count():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "data", "sample.ach")
    expected_result = 41
    parser = AchFileProcessor()
    records, exceptions = parser.parse(file_path)
 with get_db() as conn, conn.cursor() as cursor: #2
 cursor.execute("SELECT COUNT(*) FROM ach_files") 
 record_count = cursor.fetchone()[0]
    assert (
        record_count == expected_result
    ), f"Expected {expected_result}, but got {record_count}"
```

#1 我们从连接到数据库的 AchFileProcessor 中提取代码。这是一个临时解决方案，当需要更多测试连接到数据库时，我们最终需要再次重构代码。

#2 简单查询以获取存储在 ach_files 中的记录数

这个测试在 Docker 首次启动时应能正常工作，但随后的测试将失败，因为记录正在被添加。所以，我们的第二次迭代失败了

```py
Expected :41
Actual   :82
```

我们的第三次迭代失败了

```py
Expected :41
Actual   :123
```

我们需要在每次测试之前清除数据库。理想情况下，我们希望有一个仅在测试长度内存在的数据库，但首先，让我们看看我们如何在每次测试后清除表。我们可以使用一个 `pytest.fixture`，它将为我们的单个测试执行任何设置/拆卸操作。

##### 列表 5.8  `pytest.fixture` 用于设置和拆卸我们的单元测试

```py
@pytest.fixture
def setup_teardown_method():
    print("\nsetup test\n")       #1
    yield                         #2
    print("\nteardown test\n")    #3
    with get_db() as conn, conn.cursor() as cursor: #4
        cursor.execute("TRUNCATE ach_files")        
…
def test_record_count(setup_teardown_method):       #5
```

#1 在 yield 之前执行任何内容。

#2 yield 允许测试执行

#3 在 yield 之后执行任何内容。

#4 获取连接和游标，然后截断表以清除它

#5 在单元测试中包含我们的 fixture

使用这段代码，我们的测试将反复通过，因为每次运行后都会清除表。我们已经解决了数据库在运行之间保留记录的问题。当然，我们需要小心。如果我们的单元测试指向的不是开发服务器上的数据库，我们就有可能清除所需的数据。因此，我们可能想要考虑其他选项，例如内存数据库、模拟或 `pytest-postgresql`。

##### 处理数据库

在单元测试中与数据库交互的一种常见方式是使用内存数据库，例如 SQLite 或 H2。这使我们能够运行一个完全存在于内存中的数据库，因此可以将其隔离到我们的单元测试中。通常，这些好处包括快速执行和数据隔离。缺点是，很多时候，我们的生产数据库和这些内存数据库之间可能存在不存在的功能，这可能导致在尝试创建单元测试时出现问题。例如，SQLite 有五种数据类型来定义数据，而 PostgreSQL 有超过 40 种类型。这并不是说一个本质上比另一个更好——我们只是强调，如果我们的表使用不受支持的数据类型，我们可能会面临哪些挑战。我们可能最终会陷入不必要的战斗，以使我们的单元测试运行。这就是为什么我们应该有额外的工具和技术可以使用。

使用像 pgmock 这样的工具进行模拟也可以消除在测试中需要数据库的需求。在我们的场景中，我们实际上是在测试数据是否已到达数据库，所以模拟并不真正提供可行的解决方案，但可以留待以后考虑。

`Pytest-postgresql` 是一个帮助我们管理 `pytest` 中 `postgresql` 连接的包，它通过允许我们连接到测试数据库或创建/清除/修改测试中的表，提供了两全其美的解决方案。

随着我们的项目进展，我们会发现管理测试数据并保持测试隔离变得越来越困难。在第十章中，我们最终开始引入 Testcontainers 包来隔离我们的测试。当项目基础设施成熟，我们开始将测试作为构建管道的一部分运行时，这种方法也将是有益的。

在大多数应用程序中，存储数据是必要的，但正如我们所看到的，它也增加了我们设计和编码的复杂性。在本节中，我们从小处着手，确保我们能够连接到数据库，并通过存储未解析的记录来最小化所需的代码更改。随着我们继续前进，我们应用程序的复杂性将逐渐增加。在我们处理解析和存储我们的 ACH 文件时，我们还应该记住，我们将从数据库中检索和汇总数据以用于我们的 ACH 仪表板。一种轻松存储我们的 ACH 文件的方法可能适合一种数据库结构，而 ACH 仪表板可能更适合另一种结构。我们的任务是在这两个目标之间找到一个可接受的平衡。

## 5.3 存储文件头记录

ACH 文件头记录应该是我们在 ACH 文件中遇到的第一个 ACH 记录。因此，将这个记录作为第一个探索的记录添加到数据库中是有意义的。我们首先展示如何使用 ChatGPT 来处理这个问题，然后展示如何通过在我们的 IDE 中安装 GitHub Copilot 来完成一个完整的示例。

### 5.3.1 使用生成式 AI

生成式 AI 可以帮助处理很多样板代码，这些代码在一段时间后可能会变得重复。根据你的经验水平，这些样板代码可能对你来说是新的，并且多次进行这个过程可能是有益的。一旦变得乏味，这可能是一个很好的迹象，表明我们可以开始依赖生成式 AI 工具。例如，我们可以用以下通用提示来提示 ChatGPT：

**![image](img/Prompt-Icon.png)** 请创建一个 Postgres 表来存储解析后的 Nacha 文件头记录。

然后，我们从 ChatGPT 那里收到了一个`CREATE TABLE`语句，它很好地结合了`CHAR`、`VARCHAR`和`NOT NULL`。

##### 列表 5.9  ChatGPT 生成的文件头 40

```py
CREATE TABLE nacha_file_header (
    id SERIAL PRIMARY KEY,
    record_type_code CHAR(1) NOT NULL,
    priority_code CHAR(2) NOT NULL,
    immediate_destination CHAR(10) NOT NULL,
    immediate_origin CHAR(10) NOT NULL,
    file_creation_date CHAR(6) NOT NULL,
    file_creation_time CHAR(4),  #1
    file_id_modifier CHAR(1) NOT NULL,
    record_size CHAR(3) NOT NULL,
    blocking_factor CHAR(2) NOT NULL,
    format_code CHAR(1) NOT NULL,
    immediate_destination_name VARCHAR(23), #2
    immediate_origin_name VARCHAR(23),  #3
    reference_code VARCHAR(8) 
);
```

#1 `file_creation_time`是一个可选字段，因此它可能是 NULL。

#2 这些字段在 Nacha 标准中也是可选的。注意 ChatGPT 如何使用`VARCHAR`而不是`CHAR`，因为这些字段可能会用空格填充。

从个人经验来看，我们更喜欢在大多数字段中使用`VARCHAR`以避免不必要的填充。我们没有在使用`CHAR`与`VARCHAR`之间遇到任何有意义的性能影响。存储 ACH 记录可能是使用`CHAR`有意义的领域之一，因为固定长度字段不会有任何不必要的空间。`CHAR`在声明过大时往往会被误用，任何未使用的空间都会被填充。

出于好奇，我们询问了 ChatGPT 在 Postgres 数据库中`CHAR`或`VARCHAR`哪个性能更好。在对比了两者之后，它（未经我们要求）更新了示例，使用`VARCHAR`而不是`CHAR`！我们对此表示赞同，因为我们的偏好是使用`VARCHAR`。

### 5.3.2 完整示例

如果我们有一个明确的目标或者不介意花时间配置提示，ChatGPT 可以非常有帮助。否则，我们可能想通过 Copilot 的帮助完成这个过程。图 5.3 显示了本节我们将使用的流程。在这里，我们更新过程末尾的单元测试，因为这是一个相对较短的开发周期。如果我们测试前不花太多时间编码，那么我们在稍微编码后进行单元测试应该没问题。

![图片](img/CH05_F03_Kardell.png)

##### 图 5.3  本节流程和相关的代码列表

利用我们对存储记录的了解，我们应该能够将解析后的记录存储在数据库中，一旦完成，其他记录格式应该就会就位。回想一下，我们像以下列表所示那样将解析后的记录作为字典返回。

##### 列表 5.10  包含解析文件头部记录的字典

```py
        return {
            "record_type_code": line[0],   #1
            "priority_code": line[1:3],    
            "immediate_destination": line[3:13].strip(), #2
            "immediate_origin": line[13:23].strip(), 
            "file_creation_date": line[23:29],
            "file_creation_time": line[29:33],
            "file_id_modifier": line[33],
            "record_size": line[34:37],
            "blocking_factor": line[37:39],
            "format_code": line[39],
            "immediate_destination_name": line[40:63].strip(),
            "immediate_origin_name": line[63:86].strip(),
            "reference_code": line[86:94].strip(),
        }
```

#1 我们使用硬编码的值作为偏移量，因为偏移量不会改变，并且我们希望在出现问题时能够快速引用相关字段。

#2 当需要时，我们从字段中移除额外的空格。

现在我们想将那条记录存储在数据库中，而不仅仅是简单地返回它。我们可以保持这种解析方法，并从另一个也感兴趣将数据存储在数据库中的例程中调用它。我们可能选择的其他方法之一是创建专门的解析器类或实用方法来处理解析。任何这些方法都可以帮助保持代码的整洁和可重用性。为了简化，我们将采取这个例程，并将其数据简单地存储在数据库中。

首先，我们想在数据库中创建一个用于存储 ACH 文件头部的表格，接下来的列表显示了示例表格。在这个阶段，我们将保持简单，只提供存储数据所需的字段，而不引用任何外键或其他基本约束，如`字段` `长度` 和 `NOT` `NULL`。由于安装了 Copilot，许多这些字段名称被自动填充，这为我们节省了一些时间和精力。

##### 列表 5.11  存储 ACH 文件头部的表格

```py
CREATE TABLE ach_file_headers (
    ach_file_headers_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    record_type_code VARCHAR(1) NOT NULL,
    priority_code VARCHAR(2) NOT NULL,
    immediate_destination VARCHAR(10) NOT NULL,
    immediate_origin VARCHAR(10) NOT NULL,
    file_creation_date VARCHAR(6) NOT NULL,
    file_creation_time VARCHAR(4),
    file_id_modifier VARCHAR(1) NOT NULL,
    record_size VARCHAR(3) NOT NULL,
    blocking_factor VARCHAR(2) NOT NULL,
    format_code VARCHAR(1) NOT NULL,
    immediate_destination_name VARCHAR(23),
    immediate_origin_name VARCHAR(23),
    reference_code VARCHAR(8)
);
```

一旦我们有了表格，我们就可以继续更新我们的代码。我们选择将连接对象（如列表 5.4 中创建的那样）传递到我们的例程中。我们也可以将连接对象作为我们类的一部分存储，但通过作为参数传递将使单元测试我们的例程更容易。不同的情况可能需要不同的方法，所以这绝对不是完成我们任务的唯一方式。

##### 列表 5.12  传递连接对象

```py
match record_type:
    case "1":
        result = self._parse_file_header(conn, line)
    case "5":
```

现在我们有了连接对象，我们可以更新解析例程以存储数据。

##### 列表 5.13  更新 `_parse_file_header` 例程

```py
def _parse_file_header(self, conn: Connection[tuple[Any, ...]],
➥ line: str) -> Dict[str, str]: #1
    self.expected_record_types = ["5"]

    file_header = {
    … #2
    }

    conn.execute(f"INSERT INTO ach_file_headers (ach_file_headers_id, " #3
➥                     f"record_type_code, priority_code, 
➥ immediate_destination, immediate_origin," 
➥                     f"file_creation_date, file_creation_time, 
➥file_id_modifier, record_size," 
                       f"blocking_factor, format_code,
➥immediate_destination_name," 
                       f"immediate_origin_name, reference_code) " 
                       f"VALUES (DEFAULT, %(record_type_code)s, #4
%(priority_code)s, %(immediate_destination)s, "
                       f"%(immediate_origin)s, %(file_creation_date)s,
➥%(file_creation_time)s, "
                       f"%(file_id_modifier)s, %(record_size)s,
➥%(blocking_factor)s, %(format_code)s, "
                       f"%(immediate_destination_name)s,
➥%(immediate_origin_name)s, %(reference_code)s)" 
➥                           , file_header) #5

        return file_header
```

#1 我们添加了一个连接参数。

#2 返回语句变成了名为 file_header 的变量。

#3 我们可以直接在连接上执行以插入记录。

#4 我们使用命名变量作为占位符。

#5 文件头变量传递了我们的值。

我们现在应该能够将文件头记录插入数据库。接下来，我们可以回过头来更新单元测试。

##### 列表 5.14 更新的单元测试示例

```py
import pytest
import psycopg
from psycopg.rows import dict_row
from typing import Dict
from ach_processor.AchFileProcessor import AchFileProcessor

POSTGRES_USER = "someuser"
POSTGRES_PASSWORD = "supersecret"

DATABASE_URL = f"dbname={POSTGRES_USER} user={POSTGRES_USER} 
➥ password={POSTGRES_PASSWORD} host=localhost port=5432"

@pytest.fixture
def setup_teardown_method():
    print("\nsetup test\n")
    yield
    print("\nteardown test\n")
    with get_db() as conn:
        conn.execute("TRUNCATE ach_file_headers")

def get_db(row_factory = None): #1
    conn = psycopg.connect(DATABASE_URL, row_factory=row_factory) #2
    return conn

def test_parse_file_header(setup_teardown_method):
    sample_header = "101 267084131 6910001340402200830A094101DEST NAME
➥              ORIGIN NAME            XXXXXXXX"

    expected_result: Dict[str:str] = {
… #3
    }

    parser = AchFileProcessor()
    with get_db() as conn:
        result = parser._parse_file_header(conn, sample_header) #4

    with get_db(dict_row) as conn: #5
        actual_result = conn.execute("SELECT * FROM 
➥ ach_file_headers").fetchone() #6
        del actual_result["ach_file_headers_id"] #7

    assert result == expected_result, f"Expected {expected_result},
➥ but got {result}"
    assert actual_result == expected_result,
➥ f"Expected {expected_result}, but got {actual_result}" #8
```

#1 创建另一个带有 row_factory 参数的 get_db 函数

#2 在 connect 方法中使用新参数

#3 预期结果与之前相同。

#4 保持当前的解析结果不变

#5 获取另一个连接，指定结果应以字典形式返回

#6 返回结果；actual_result 将是一个字典。

#7 从返回结果中删除 ach_file_headers_id

#8 比较两个结果

在这里，我们重复了一些代码，例如带有新参数的`get_db`方法。随着我们通过代码，我们必须关注这种重复，并在可能的情况下考虑将这些方法提取到实用程序或辅助类中。JetBrains 的 IDE（以及其他）通常会指出重复的代码，并提供将代码提取到函数中的自动化选项。

我们也保留了原始的结果比较，因为该方法仍然在返回记录。随着我们继续通过添加更多功能来改进项目，我们可能会移除它，转而使用其他返回信息（例如，记录是否被解析）。我们现在应该了解解析后的 ACH 记录如何存储在数据库中。对于 ACH 仪表板，为了提供一个具有聚合 ACH 数据的有意义界面，我们需要能够在数据库中存储所有解析的 ACH 信息。在接下来的章节中，我们将探讨代码如何随着我们的工作而演变，不仅扩展功能，还解决一些非功能性要求，如可维护性和可测试性。

## 5.4 存储剩余的 ACH 记录

我们现在已经创建了一个数据库并在两个单独的表中存储了数据，因此我们应该有足够的框架来在数据库中存储剩余记录的数据。对于所有剩余的表，过程都是相同的：

+   创建表并重启 Docker。

+   更新解析例程。

+   更新单元测试并验证你的结果。

如果采用测试驱动开发方法，我们可以交换更新解析例程和单元测试。无论如何，我们都应该工作在短反馈周期中，让任何错误在过程早期而不是在实现了一打表之后被发现。我们还发现了一些清理代码并使其更好的机会。

根据我们数据库的预期结构，这可能是一个添加几个表的好地方。检查数据库结构是否合理——是否有什么需要改变的地方？

下一个部分将介绍从处理必须添加的其他表中学习的一些经验和见解。

### 5.4.1 存储 ACH 文件挑战：经验教训

在添加额外的表格时，我们应该对代码进行一些观察，并抓住机会改进我们的代码。有没有哪些特别突出的？重要的是要注意并解决可能的痛点，例如重复的代码、低效的处理、混乱的逻辑等等。有时候，通过清理代码使其更加直接和易于维护，我们真的能感受到成就感。我们将回顾我们在添加额外数据库表格时遇到的一些问题。

由于我们使用的是 `psycopg3`（它被混淆地定义为 `psycopg`）而不是 `psycopg2`，我们发现我们的生成式 AI 工具往往没有充分利用一些新的增强功能。例如，GitHub Copilot 最初坚持声明 `cursor` 方法，但似乎学会了我们的偏好，即使用连接，过了一段时间后，它就不再提供这些方法了。这是有道理的，因为 Copilot 应该学习和适应我们的风格和编码偏好。在这些工具的最新版本中，我们也看到了检索增强生成（RAG）的加入，这有助于大型语言模型（LLMs）如 ChatGPT 保持与最新信息的同步。当然，我们还将长期观察它们的性能，因为大量的训练数据并没有使用这些新功能。

在创建表格时，GitHub Copilot 在命名列方面做得很好，大多数情况下它们与文档中的字段名称相匹配，只有少数小的例外。这很有帮助，因为我们已经在我们的 Python 代码中创建了字段名称以匹配 NACHA 标准，并且因为我们能够在 SQL 查询中使用命名参数，所以从返回记录到写入数据库的过渡变得非常容易。另一个我们一直很欣赏的功能（尽管它并不总是起作用）是 Copilot 在编写单元测试方面的协助能力。我们已经为需要解析的每个记录创建了单元测试，当我们将其分解为字典以进行比较时，Copilot 提供了帮助，并用我们的测试数据填充了各个字段。尽管在某些情况下它可能错了一个或两个字符，但总体上它确实很有帮助。

在我们构建单元测试的过程中，我们最终创建了一个 `SqlUtils` 类来处理一些重复的任务。我们首先将数据库参数，如用户、密码和 URL 移动到例程中。然后后来，我们扩展了这个例程以处理行工厂参数的传递，这样我们就可以返回一个值的字典。我们还创建了一个例程来截断表，以便重复测试有干净的表可以工作。因此，当我们期望检索单个记录但检索到多个记录时，我们的断言不会失败。

与使用 `SqlUtils` 类去除代码重复类似，我们也通过创建一个变量来保存表名，并在必要时使用 Python f-strings 创建 SQL 命令（但不是用于参数传递），从而在整个单元测试中移除了对表名的硬编码。需要注意的是，尽管我们将其视为内部工具，并且不期望输入恶意代码，但我们仍然尽可能使用参数化查询。然而，我们也考虑了检查传递的表名是否与数据库信息模式中的信息匹配的可能性，以确保它们是有效的。然而，对于内部工具来说，这似乎有点过度了。

我们在编码时多次因为忘记将设置/拆卸 `pytest.fixture` 放入单元测试方法中而受到咬伤。这通常会在我们反复运行测试时导致错误，因为数据库并不总是处于干净的状态。这种情况发生得足够频繁，以至于我们考虑创建一个类层次结构，以便包含清理表的操作，这样我们就可以避免自己给自己找麻烦。然而，在我们目前的流程中，添加这个功能似乎还为时过早，所以我们暂时回避了这个问题。

## 5.5 存储异常

现在我们应该已经掌握了使用 Python 和 PostgreSQL 的工作基础。成功将我们的 sample.ach 文件存储到数据库中应该会增强我们的信心。然而，我们也应该注意到我们的异常还没有被存储在数据库中。我们希望也能跟踪这些异常。文件因各种原因被拒绝是很常见的，ACH 处理器需要能够确定文件是否可以手动修复，或者是否需要从原始方请求一个新的文件。图 5.4 展示了这一部分的流程。

![图片](img/CH05_F04_Kardell.png)

##### 图 5.4  本节流程和相关的代码列表

第一项任务是创建表。我们从一个简单的记录开始，该记录只包含错误描述。随着我们项目的扩展，我们可能会发现我们需要按记录类型分解异常，添加对特定记录的引用（以帮助维护数据库的完整性，以防记录被删除或更新），或者实施其他一些随着项目增强而变得更加明显的改进。然而，这些关注点超出了我们当前需要完成的工作范围，所以我们将这个问题留到第八章再解决。

##### 列表 5.15  简单的异常表

```py
CREATE TABLE ach_exceptions (
    ach_exceptions_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exception_description VARCHAR(255) NOT NULL
);
```

创建了这个表之后，我们可以重新启动 Docker，并在 AchFile­Processor.py 中添加一个方法来插入记录。

##### 列表 5.16  写入表的简单方法

```py
    def _add_exception(self, conn: Connection[tuple[Any, ...]],
➥ exception: str) -> None: #1
        conn.execute(f"INSERT INTO ach_exceptions#2
➥ (ach_exceptions_id, exception_description) "  #2
                     f"VALUES (DEFAULT, %(exception)s)", #2
➥ {"exception": exception} )   #2
        return
```

#1 简单方法传递一个连接和异常字符串

#2 将字符串插入到表中

完成这项任务后，我们可以用一个调用替换我们用来保存异常的数组。我们还需要更新覆盖这些异常的各种测试用例，例如错误的附加指示符、无效的记录长度以及确保跟踪号是递增的。更新这些单元测试提供了创建可维护代码的额外机会。

我们移除了将异常返回给调用例程的操作，因此像`records,` `exceptions` `=` `parser.parse(file_path)`这样的代码现在变为`records` `=` `parser.parse(file_path)`。然而，这个变化使我们立即检索异常，因为我们的单元测试正在验证异常的数量和异常消息文本。我们选择向`SqlUtils`添加另一个方法来处理这个问题。

##### 列表 5.17  获取 ACH 异常的方法

```py
    def get_exceptions() -> list:
        with SqlUtils.get_db() as conn:  #1
            exceptions = conn.execute(f"SELECT exception_description
➥ FROM ach_exceptions").fetchall() #2
        flattened_list = [item[0] for item in exceptions] #3
        return flattened_list  
```

#1 与关键字一起使用以获取数据库连接

#2 在单个执行命令中获取所有异常

#3 返回异常列表而不是元组

在放置了辅助函数之后，我们现在可以用`exceptions =` `SqlUtils.get_exceptions()`返回异常，并且现有的单元测试逻辑应该无需任何修改即可工作。

由于我们将异常存储在表中，现在我们的单元测试中需要截断多个表。我们可以继续使用新的表调用`SqlUtils.truncate()`方法。在这个阶段，每个测试最多只有两个表。然而，我们更希望有一种方法来清除所有表，因为这将确保每个测试的数据库都是空的。列表 5.17 展示了截断所有表的方法。显然，这种方法应该谨慎使用，因为我们现在正在截断数据库中的所有表。我们也可以删除并重新创建数据库；然而，这样做我们可以对每个表进行单独访问，从而获得更多的控制。我们曾与采用这种类型方法的项目合作，以检查表中是否存在意外数据，例如确定数据是否意外写入其他表。当然，我们的需求会因项目而异，可能你不需要或希望使用这种方法截断数据。

##### 列表 5.18  截断所有表的 SQL

```py
DO $$         #1
   DECLARE        #2
      r RECORD;   
BEGIN             #3
   FOR r IN SELECT table_name FROM information_schema.tables
➥ WHERE #D  table_schema = 'public' #4
   LOOP                  #4
      EXECUTE 'ALTER TABLE ' || quote_ident(r.table_name) #4
➥ || ' DISABLE TRIGGER ALL';  #4
   END LOOP;  #4

   EXECUTE (  #5
      SELECT 'TRUNCATE TABLE ' || string_agg( #5
➥quote_ident(table_name), ', ') || ' CASCADE'   #5
        FROM information_schema.tables #5
       WHERE table_schema = 'public'   #5
   );  #5

   FOR r IN SELECT table_name FROM information_schema.tables
➥ WHERE table_schema = 'public' #6
   LOOP #6
      EXECUTE 'ALTER TABLE ' || quote_ident(r.table_name) || #6
➥ ' ENABLE TRIGGER ALL';  #6
   END LOOP;  #6
END $$; #7
```

#1 创建一个匿名代码块

#2 声明一个类型为 RECORD 的变量，作为没有预定义结构的行的占位符

#3 表示新事务的开始

#4 禁用表上的任何触发器

#5 收集表列表并截断它们

#6 重新启用触发器

#7 提交当前事务

到目前为止，我们应该已经到达了与第三章中我们运行代码相似的地方。回想一下，我们能够对一些简单的 ACH 文件进行一些基本的解析，主要的变化是我们现在将我们辛勤工作的结果存储到数据库中。我们可以花点时间自我祝贺，但只能短暂地，因为虽然我们能够解析文件，但我们没有让用户加载文件的方法。接下来的几节将探讨扩展我们的 API 以上传 ACH 文件，并查看它对我们数据库产生的级联效应。

## 5.6 上传 ACH 文件

我们可能认为这是我们迄今为止最重要的改变。当然，能够解析文件是必要的，但对于许多开发者来说，最令人感到有成就感的是能够与用户进行交互。这可能是因为能够与我们的项目一起工作和交互本身就感觉像是一种成就。我们还认为单元测试给我们带来了几乎相同的成就感，这就是为什么我们如此喜欢测试！图 5.5 显示了本节的流程。

![图片](img/CH05_F05_Kardell.png)

##### 图 5.5 本节的流程和相关的代码列表

我们之前使用硬编码的值构建了一个基本的 API。我们将使用那段代码并添加上传 ACH 文件的功能。从那里，我们将扩展 API 以从数据库中检索数据而不是硬编码的值。在原始的 ACH 仪表板中，一个问题是我们处理 ACH 文件时缺乏控制。

在创建新的 API 之前，让我们确保我们可以使用硬编码的值对现有的端点进行单元测试。接下来的列表显示了它的样子。Copilot 能够在我们输入时生成大部分代码，所以我们只需要确保它按照我们的意图执行。

##### 列表 5.19 FastAPI 的单元测试

```py
from fastapi.testclient import TestClient #1

from app.main import app  #2

client = TestClient(app) #3

def test_read_files(): #4
    response = client.get("/api/v1/files") #5
 assert response.status_code == 200 #6
 assert response.json() == [{"file": "File_1"}, #6
➥ {"file": "File_2"}] 
```

#1 导入单元测试所需的 TestClient

#2 导入我们的应用程序

#3 定义客户端

#4 定义了测试方法

#5 调用我们的端点并保存响应

#6 断言语句验证响应

所以，这相当简单。现在，我们希望能够将一个 ACH 文件`POST`到后端，获取一个上传成功的响应，然后开始解析文件。为什么不在解析文件后再向用户返回响应呢？我们的 ACH 文件处理可能需要一些时间，特别是当我们考虑到它最终需要与其他系统交互以执行诸如验证账户/余额、OFAC 验证、欺诈分析以及 ACH 处理器可能想要执行的其他任务时。为了避免让用户等待，我们可以验证文件已上传，并启动一个任务来执行解析。

我们已经有一个用于上传文件的端点，因此我们可以添加一个单元测试然后上传文件。由于文件将通过 `HTTP` `POST` 请求上传，它将使用 `multipart-formdata` 编码。这意味着我们需要 `python-multipart` 包，因为这是解析使用该格式的请求的要求。如果我们没有安装它，我们将收到一个友好的提醒（以错误的形式）。

##### 列表 5.20  `python-multipart` 未安装

```py
E   RuntimeError: Form data requires "python-multipart" to be installed. 
E   You can install "python-multipart" with: 
E   
E   pip install python-multipart
```

创建上传测试应类似于以下列表。

##### 列表 5.21  上传文件的单元测试

```py
def test_upload_file():
    with open("data/sample.ach", "rb") as test_file:  #1
        response = client.post("/api/v1/files", #2
  files={"file": test_file})    
    assert response.status_code == 201  #3
```

#1 打开样本文件

#2 使用客户端发送文件

#3 确保我们收到 20 1 状态

我们还需要更新端点以接收文件。由于我们只对状态码感兴趣，因此不需要返回任何内容。

##### 列表 5.22  更新后的端点以接收文件

```py
from fastapi import APIRouter, Request, status,
➥ File, UploadFile #1
@router.post("", status_code=status.HTTP_201_CREATED)
async def create_file(file: UploadFile = File(...)): #2
    return None  #3
```

#1 需要额外的导入。

#2 定义文件为 UploadFile

#3 不返回任何内容；仅返回必要的状态码

现在我们应该有一个成功的单元测试，然后真正的任务可以开始。手动上传文件或通过某些自动过程应该是我们所有 ACH 表的驱动力。我们应该存储一些文件信息，如文件名、上传时间、文件哈希（用于跟踪重复文件），以及我们可能决定需要的其他信息。然后，该记录的 UUID 应该包含在任何子表中（我们刚刚创建的所有之前的表）。如果我们有更多的经验或以不同的顺序处理问题（例如，首先上传文件），我们可能会避免更多的返工，但我们也可能需要从开始引入更多的数据库概念。这种方法的好处是实际需要进行的返工以合并更改。通常，开发者会因为害怕改变而变得瘫痪。大型复杂系统有时就像一座纸牌屋，一次错误的移动可能导致一切崩溃。拥有良好的单元测试覆盖率和对测试的信心可以大大减轻这些恐惧。害怕改变和改进工作软件最终会导致代码腐烂。

我们将重新利用 `ach_files` 表作为主表，包含上传信息，并将 `ach_files` 重命名为 `ach_records`，使其唯一的工作是存储未解析的 ach 记录。更新的表定义如下所示。

##### 列表 5.23  `ach_files` 和 `ach_records` 的更新后的表列表

```py
CREATE TABLE ach_files (         #1
    ach_files_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(), 
    file_name VARCHAR(255) NOT NULL, 
    file_hash VARCHAR(32) NOT NULL, 
    created_at TIMESTAMP NOT NULL DEFAULT NOW(), 
); #2

CREATE TABLE ach_records (  #3
    ach_records_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ach_files_id UUID NOT NULL REFERENCES ach_files(ach_files_id)
➥ ON DELETE CASCADE ON UPDATE CASCADE, #4
    file_name VARCHAR(255) NOT NULL,  #5
    unparsed_record VARCHAR(94) NOT NULL,
    sequence_number INTEGER NOT NULL
);
```

#1 ach_files 表被复制并重新用于存储上传详情。

#2 ach_files 表被复制并重新用于存储上传详情。

#3 之前的 ach_files 将成为 ach_records。

#4 创建一个名为 ach_files_id 的外键，它引用 ach_files 表中的 ach_files_id

#5 移除 file_name，因为它现在存储在 ach_files 表中

新的表 `ach_records` 使用 `REFERENCES`、`ON DELETE` 和 `ON UPDATE` 关键字创建到 `ach_files` 表的外键。这个特性允许数据库保持其引用完整性。例如，当我们从 `ach_files` 中删除一个 ACH 文件时，我们不希望进入每个表并删除相关数据。相反，我们希望在表之间定义一个关系，如果我们删除了 `ach_file`，则所有相关数据都会被删除。一旦我们完成了更新我们的表，我们就可以看到这个功能的效果。这也会影响我们的测试。一旦我们实现了引用完整性，我们需要确保 `FOREIGN KEY` 约束得到维护。

例如，如果我们想让单元测试将记录写入 `ach_records` 表，我们需要一个有效的 `ach_files_id`（它必须在 `ach_files` 表中存在）。因此，我们可能会考虑扩展我们之前开发的 `SqlUtils` 类，设置一些通用的记录并使这更容易。维护引用完整性在我们设置测试时可能意味着额外的工作，但它值得实现。我们曾在具有引用完整性的系统中工作，并看到程序因不完整的关系而崩溃或循环。通常，缺乏引用完整性在开发人员编写了各种实用程序程序来扫描和修复数据的系统中是明显的（我们不得不在多个场合自己这样做）。

此外，我们还想存储 ACH 文件的 MD5（或您喜欢的任何算法）哈希值，以便识别重复文件。首先，我们可以使用 `Get-FileHash -Path ".\sample.ach" -Algorithm MD5 | Select-Object -ExpandProperty` 从命令行获取哈希值，在我们的例子中打印出 `18F3C08227C3A60D418B3772213A94E3`。我们将保留这些信息，因为我们将在数据库中以 Python 计算的哈希值存储，并期望它相同。

有趣的是，在编写函数时，Copilot 提醒我们使用字符串插值来编写 SQL 查询，`f"INSERT INTO ach_files (file_name, md5_hash) VALUES ('{file.filename}', '{md5_hash}')`，但我们继续使用我们的标准参数化查询，因为我们想保持安全的编码实践。使用字符串插值的查询容易受到 SQL 注入 ([`mng.bz/QDNj`](https://mng.bz/QDNj)) 的攻击。我们的 `/api/v1/files` 端点的代码在下一个列表中显示。

##### 列表 5.24  更新端点以上传 ACH 文件

```py
@router.post("", status_code=status.HTTP_201_CREATED)
async def create_file(file: UploadFile = File(...)):
    contents = await file.read() #1
 md5_hash  = hashlib.md5(contents).hexdigest() #2
 with DbUtils.get_db_connection() as conn: #3
 conn.execute( #4
 f""" #4
 INSERT INTO ach_files (file_name, file_hash) #4
 VALUES (%s, %s) #4
 """, (file.filename, md5_hash) #4
 ) #4
    return None #4
```

#1 读取上传的文件

#2 使用 hashlib 创建文件的哈希值

#3 使用我们新创建的 DbUtils 返回连接

#4 简单 SQL 将记录插入到 ach_files

我们创建了一个简单的 `DbUtils` 类来返回数据库连接。我们想象它将在我们的项目中多个地方使用，因此将其提取到自己的文件中。

##### 列表 5.25  用于创建数据库连接的 `DbUtils`

```py
import psycopg

POSTGRES_USER = "someuser" #1
POSTGRES_PASSWORD = "supersecret"  #1
 #1
DATABASE_URL = f"dbname={POSTGRES_USER} user={POSTGRES_USER} #1
➥ password={POSTGRES_PASSWORD} host=localhost port=5432" #1

def get_db_connection(row_factory = None):  #2
    conn = psycopg.connect(DATABASE_URL,  #2
➥row_factory=row_factory)  #2
    return conn  
```

#1 硬编码的值将在以后被移除

#2 创建并返回一个连接

通过这些更改，我们应该能够运行单元测试，然后转到我们的 CloudBeaver UI，查看新添加的记录并手动比较记录哈希。然而，手动检查记录并不有趣，所以作为最后一部分，我们应该更新我们的单元测试以包含相同的`SqlUtils`，就像我们之前看到的那样，并确保我们的表中至少有一条记录。以下列表显示了更新的单元测试。

##### 列表 5.26  更新后的上传文件单元测试

```py
…
from tests.SqlUtils import SqlUtils #1
import pytest
…
@pytest.fixture #2
def setup_teardown_method():  #2
 yield #2
 SqlUtils.truncate_all() #2
…
def test_upload_file(setup_teardown_method): #3
    with open("data/sample.ach", "rb") as test_file:
        response = client.post("/api/v1/files", files={"file": test_file})
    assert response.status_code == 201
 assert SqlUtils.get_row_count_of_1('ach_files') #4
```

#1 必要的导入

#2 定义我们的固定装置以确保所有表都被截断

#3 将固定装置包含在我们的单元测试中

#4 确保 ach_files 表中有一行

能够成功上传文件可能看起来不是什么大事，但我们必须记住，我们 ACH 仪表板的所有其他功能都基于这一部分。现在我们可以上传 ACH 文件，我们可以使用诸如错误检查和恢复、账户贷记/借记、欺诈分析等功能。完成这项任务后，我们可以扩展数据库结构。

## 5.7 以完整性存储记录

如前所述，我们希望我们的数据库具有引用完整性，这可以简化数据库工具中的导航，更重要的是，具有防止悬挂记录的能力。我们已经看到，没有引用完整性的数据库需要定期运行的维护程序来清理那些悬挂记录，以及由不完整数据引起的无限循环和程序崩溃。使用关系数据库的一个巨大好处是它们对数据完整性的支持。我们强烈建议在设计数据库时牢记这一点。

在我们之前的例子中，我们已经定义了`PRIMARY` `KEY`以及如何更新`ach_records`表。在列表 5.23 中，我们通过使用`REFERENCES`关键字添加了第一个`FOREIGN` `KEY`，这使我们开始了维护引用完整性的道路。我们继续使用外键更新剩余的表和代码，并审查使用它们对我们的表、代码和单元测试的影响。

如果我们现在尝试运行我们的单元测试，我们会看到引用`UndefinedColumn`的错误，因为我们更新了列表 5.23 中的表定义。这不是坏事。我们构建了单元测试，这使我们能够有信心重构我们的代码。我们从深入我们的`test_record_count`单元测试（来自列表 5.7）开始，解决我们的`psycopg.errors.UndefinedColumn:` `column` `"unparsed_record"` of `relation` `"ach_files"` `does` `not` `exist`的错误。有问题的代码如下所示。

##### 列表 5.27  无效的列名

```py
conn.execute("""
                INSERT INTO ach_files
[CA] (file_name, unparsed_record, sequence_number) 
                VALUES (%s, %s, %s)
                """, (os.path.basename(filename), line, sequence_number))
```

根据你的 IDE 和设置，这个语句可能已经被标记。在 PyCharm 中，我们已经定义了一个数据库连接，因此字段 `unparsed_record` 和 `sequence_number` 被标记为 `Unable to resolve column` 错误。如果你的 IDE 没有这个功能，那么堆栈跟踪也会显示行号。根据这些信息，我们注意到 `INSERT` 语句指向了错误的表，因为我们已经将 `ach_files` 重命名为 `ach_records`。更改这一点并重新运行测试会得到一个新的错误 `psycopg.errors.NotNullViolation: null value in column "ach_files_id" of relation "ach_records" violates not-null constraint`。我们定义了 `ach_files_id` 为外键，它不能为空——实际上，它必须指向 `ach_files` 表中的一个有效记录。回想一下，我们的 API 会在文件上传时创建这个记录。因此，现在 `AchFileProcessor` 可能需要同时调用正在解析的文件名和有效的 `ach_files_id`。我们可以更新单元测试，使解析例程使用这个 ID 被调用。`parser.parse(file_path)` 将需要变成 `parser.parse(ach_files_id, file_path)`，但定义 `ach_files_id` 并不足以解决问题。我们需要一种方法来创建一个有效的 `ach_files_id`，因为我们必须保持数据库的完整性。我们可以在测试期间删除约束，这可能是一个选项，如果我们的单元测试主要关注功能的话。然而，在这种情况下，我们希望保持这个约束。因此，我们需要创建一个插入记录的函数，然后退后一步考虑我们的前进路径，因为我们可能想要采取几条不同的路线。

记住我们已经在代码中编写了一个 `INSERT` 语句来将 `ach_files` 记录插入到我们的 API 中。我们可以在单元测试中重复这个查询并获取插入的值。这会有效，但我们会重复代码，而我们希望尽可能避免这种情况。我们可以在 `SqlUtils` 中添加一个例程来为我们处理这个任务，这样其他方法也可以访问它，因为其他单元测试可能也需要这个功能。然而，`SqlUtils` 的目的是仅在我们测试时帮助我们，我们已经看到我们还需要在其他地方使用这个功能。也许我们应该为 `AchFileProcessor` 创建一个工具类来插入我们想要的记录，这将允许我们重构现有的 SQL 查询，使其从解析例程中分离出来。没有明确的答案。根据项目、需求、编程风格和经验，我们可能会看到其他替代方案或对如何处理这个问题有偏好。在这种情况下，我们认为使用 Pydantic 是前进的最佳方式。

## 5.8 使用 Pydantic

在上一节中，我们遇到了一个问题，即如何处理插入记录以使我们的单元测试工作。虽然我们可以采取几种方法，但 Pydantic 将有助于进一步重构和扩展我们的项目。我们之前在第四章中介绍了 Pydantic 来帮助我们文档化 API，因此我们知道它已经有一些好处。另一个好处是，我们将能够以允许开发者不必记住哪些字段存在的方式来定义我们的表和字段。换句话说，我们开始从解析逻辑中抽象 SQL。让我们从我们的`ach_files`表开始应用这一点，这样我们就可以创建一个记录并返回一个有效的记录 ID，这正是我们真正想要做的，以解决我们的第一个单元测试问题。

##### 为什么没有对象关系模型框架？

对象关系模型（ORM）有许多好处，并且在许多不同的行业中都得到了广泛的应用。我们现在推迟将如 SQLAlchemy 之类的 ORM 纳入其中，因为我们想确保读者能够直接接触到 SQL，以防那些技能需要复习。

之后，我们将展示如何将 ORM 包含到项目中，所以请耐心等待。当然，如果你熟悉 ORM，你可以直接跳进去开始使用它们。

我们首先为我们的表创建一个 Pydantic 模型，如下所示。这个简单的模型为将要写入我们的`ach_files`表的建模提供了一个基本布局。

##### 列表 5.28  我们为`ach_files`的 Pydantic 模型

```py
from typing import Optional           #1
from datetime import datetime          #1
from pydantic import BaseModel, UUID4 #1

class AchFileSchema(BaseModel):  #2
    id: Optional[UUID4] = None   #3
    file_name: str                #3
    file_hash: str                #3
    created_at: Optional[datetime] = None #3
```

#1 必要的导入语句

#2 该类扩展了 Pydantic BaseModel。

#3 我们的字段定义；注意 Optional 关键字用于数据库将提供的字段（如 ID）。

接下来，我们可以定义一个类，它将提供一些基本的创建、读取、更新和删除（CRUD）方法来处理与数据库表的工作。以下列表显示了我们将创建的`AchFileSql`类，以封装处理我们的`ach_files`数据库表的逻辑。

##### 列表 5.29  `AchFileSql`类

```py
from typing import Optional  #1
from uuid import UUID    #1
from psycopg.rows import dict_row #1
from ach_processor.database import DbUtils #1
from ach_processor.schemas.AchFileSchema  #1
➥import AchFileSchema #1

class AchFileSql:
    def insert_record(self, ach_file: AchFileSchema)
➥ -> UUID: #2
        with DbUtils.get_db_connection() as conn:
            result = conn.execute(
                """
                INSERT INTO ach_files(ach_files_id,
➥ file_name, file_hash, created_at)
                               VALUES 
➥(DEFAULT, %(file_name)s, %(file_hash)s, DEFAULT)
                               RETURNING ach_files_id #3
                                """, ach_file.model_dump()) #4

        return result.fetchone()[0]

    def get_record(self, ach_file_id: UUID)
➥ -> AchFileSchema: #5
        with DbUtils.get_db_connection(row_factory=class_row(AchFileSchema))
➥ as conn: #6
            result = conn.execute(
                """
                SELECT * FROM ach_files WHERE ach_files_id = %s
                """, [ach_file_id.hex])

            record = result.fetchone()

            if not record: #7
                raise KeyError(f"Record with id #7
➥ {ach_file_id} not found")  #7

            return record
```

#1 类所需的导入

#2 创建一个函数来插入记录并返回 UUID

#3 使用 RETURNING 关键字返回新插入记录的 ID

#4 使用 model_dump 创建一个引用字段的字典

#5 创建一个函数来返回一个指定的记录

#6 通过使用 class_row 的 row_factory，我们可以直接返回记录。

#7 如果没有找到任何内容，将引发错误

最后，我们创建一个单元测试来验证我们新创建的类。我们创建这个单元测试作为一个类，只是为了展示另一种帮助组织测试的方法。我们还引入了`pytest`固定件的`autouse`选项，这样我们就不必在每一个方法中都包含它们。

##### 列表 5.30  测试我们的新类

```py
import pytest  #1
from tests.SqlUtils import SqlUtils #1
from ach_processor.database.AchFileSql import AchFileSql #1
from ach_processor.schemas.AchFileSchema #1
➥ import AchFileSchema #1

class TestAchFileSql:
    @pytest.fixture(autouse=True)   #2
    def setup_teardown_method(self):
        print("\nsetup test\n")
        yield
        SqlUtils.truncate_all()  #3

    def test_insert_record(self):
        ach_file_record = AchFileSchema(
            file_name="sample.ach",
            file_hash="1234567890"
        )
        sql = AchFileSql()
        ach_file_id = sql.insert_record(ach_file_record)  #4
        retrieved_record = sql.get_record(ach_file_id)    #5
        assert SqlUtils.get_row_count_of_1("ach_files")
➥ is True, "Expected 1 record" #6
        assert retrieved_record.file_name == ach_file_record.file_name, #7
➥ f"Expected {ach_file_record.file_name}, but got 
{retrieved_record.file_name}"  #7
```

#1 必要的导入

#2 我们现在使用 autouse。

#3 完成后截断表

#4 插入记录

#5 立即返回记录

#6 断言语句验证我们的结果

这是一个将我们的项目提升到下一个层次的良好开端。你可能想知道为什么我们没有首先以测试驱动开发的方式编写我们的测试。有时，采取“先测试后”的方法更容易，尤其是在演示新概念时。再次强调，这并不是关于编写测试，而是关于在一个短周期内工作。因此，一旦我们有了可以测试的内容，我们就开始测试它。

记住，使用 Pydantic，我们可以获得字段验证和记录我们的 API 的能力。我们稍后会查看这一点，但现在，我们应该继续重构我们的代码以利用 Pydantic。一旦我们使用 Pydantic 重构了代码并且单元测试通过，我们就可以继续处理 API。

## 5.9 经验教训

将 Pydantic 包含到我们的代码中并对其进行分离的过程要求我们不仅重构了 ach_file_processor.py，还重构了我们的单元测试。重构使我们能够在两个领域都改进我们的代码，并获得更干净、更容易测试的代码。不幸的是，当我们完成了根据图 5.1 中的原始规范创建的数据库结构后，我们也遇到了一些问题。你在重构代码之前注意到任何结构上的问题了吗？

将外键插入到该数据库结构中揭示了在 `ach_files` 表中维护数据完整性的问题。虽然从那些子表中删除记录的行为是正确的，但 `ach_files` 表并没有删除我们想要的全部记录。例如，如果我们删除一个批次，我们期望相关的类型 6-8 记录也会被删除，但在当前的结构中这是不可能发生的。

这种情况并不少见。通常，当给定一个项目的规范时——根据项目利益相关者的经验以及用于研究项目领域的允许时间——可能无法完全梳理出所有设计细节。在扩展敏捷的背景下，我们可能会将这个项目视为一种启用器——更具体地说，是一种探索启用器。探索启用器有助于探索潜在的解决方案，包括研究和原型制作等活动。如果我们没有在项目开始之前充分完成这些活动，我们可能会遇到我们在这里遇到的问题。处理这种情况有几种替代方案，正确的答案可能取决于对项目要求的重新评估。

为了回顾项目要求，我们希望找到一个解决方案，

+   在处理数据库时提供了 ETL 和 ELT 选项。

+   在整个文件中提供了数据完整性。由于 ACH 文件是一个层次结构，因此存在多种场景，当文件、批次或条目被删除时，需要清理额外的记录。

让我们提出一些可能需要向业务利益相关者展示的不同选项。

首先，我们可以放弃将单个记录解析到数据库中，而只存储未解析的记录。这无疑会简化数据库设计，因为我们只需处理一个表。当然，这种方法限制了关系数据库的实用性，并且需要额外的应用程序代码来处理我们之前提到过的数据完整性。例如，如果用户想要删除一批记录，我们必须确保应用程序代码删除了所有记录，这将扩大并复杂化我们的应用程序代码。

第二，我们可以放弃在表中存储未解析的记录。如果我们确定我们不需要未解析的记录，这可能是一个潜在的解决方案。当然，这也意味着我们的文件需要符合数据库约束，而且业务已经要求在无效数据（如数值字段中的非数值数据）可能导致解析记录被数据库拒绝的情况下保留未解析的记录。这似乎是一个难以绕过的硬性要求。

第三，我们可以考虑设置数据库触发器来删除记录。数据库触发器是一段代码，可以在数据库中发生某些事件时自动执行。我们可能在解析记录表中创建触发器，当记录被删除时，也会删除相关的未解析记录。但这听起来并不有趣。

我们选择了另一种途径来解决这个问题——将表重新组织成我们为每种类型使用一个未解析记录表的结构。这需要对表及其相关应用程序代码进行大量重构，但对我们来说，如果我们想要保持保留未解析和解析记录的要求，这是最有意义的。更新的数据库结构图如图 5.6 所示，其中还包括了来自图 5.1 的原布局引用。

这种结构为我们提供了最初所寻找的数据完整性，但现在我们必须找到一种方法来查看所有未解析的记录。为了实现这一点，我们创建了一个数据库视图。数据库视图是一个虚拟表，是存储查询的结果。通过使用视图，我们可以避免自己和其他需要使用我们数据库的人进行繁琐的将未解析数据关联起来的任务。接下来的列表显示了创建的数据库视图。

##### 列表 5.31 创建数据库视图

```py
CREATE VIEW combined_ach_records AS
SELECT 
    r1.ach_records_type_1_id AS primary_key, 
    r1.unparsed_record, 
    r1.sequence_number,
    r1.ach_files_id
FROM 
    ach_records_type_1 AS r1

UNION ALL                 #1

SELECT 
    r5.ach_records_type_5_id, 
    r5.unparsed_record, 
    r5.sequence_number,
    r1_r5.ach_files_id
FROM 
    ach_records_type_5 AS r5
JOIN ach_records_type_1 AS r1_r5 
    USING (ach_records_type_1_id) #2
 #3

…
UNION ALL

SELECT 
       r9.ach_records_type_9_id, 
       r9.unparsed_record, 
       r9.sequence_number,
       r1_r9.ach_files_id
  FROM 
       ach_records_type_9 AS r9
  JOIN ach_records_type_1 AS r1_r9 
 USING (ach_records_type_1_id)
```

#1 `UNION ALL`将合并连续的`SELECT`语句的结果。

#2 使用`USING`语句可以提供更简洁的语法，而不是使用`ON table.field = table.field`。在 PostgreSQL 中，当连接字段在表中具有相同的名称时，可以使用此功能。

![图片](img/CH05_F06_Kardell.png)

##### 图 5.6 更新后的表结构图

这段代码允许我们查看整个 ACH 文件，就像它被上传到系统中一样，没有任何我们的处理操作。有一种查看原始文件的方法，这允许我们向用户提供查看整个 ACH 文件内容以及异常记录的选项，以帮助他们调试任何问题。

## 5.10 编码更改

既然我们已经确定了数据库结构，那么编写任何额外的单元测试以及更新现有的测试以正确运行就是一个问题了。让我们看看为了支持这个结构所做的更改。

### 5.10.1 为未解析的 ACH 记录创建 Pydantic 架构

所有未解析的记录都共享 `unparsed_record` 和序列号字段。因此，这是一个创建一个将继承这些字段以避免每次都输入它们的类结构的好机会。我们创建了一个 `AchRecordBaseSchema`。

##### 列表 5.32  我们 ACH 记录的基架构

```py
from abc import ABC  #1
from pydantic import BaseModel #2

class AchRecordBaseSchema(ABC, BaseModel):  #3
    unparsed_record: str    #4
    sequence_number: int  
```

#1 导入 ABC 允许我们创建一个抽象类。

#2 `BaseModel` 是 Pydantic 所必需的。

#3 我们的这个类从 ABC 和 `BaseModel` 继承。

#4 所有 ACH 记录类中都将存在的字段

使用该架构，我们可以定义每个记录类型，如下所示。

##### 列表 5.33  ACH 记录架构

```py
from typing import Optional #1
from pydantic import UUID4 #1
 #1
from ach_processor.schemas.ach_record.ach_record_base_schema import AchRecordBaseSchema #1
 #1
class AchRecordType1Schema(AchRecordBaseSchema): #2
    ach_records_type_1_id: Optional[UUID4] = None #3
    ach_files_id: UUID4 #4
```

#1 必须导入

#2 由于它是 `AchRecordBaseSchema` 的子类，而 `AchRecordBaseSchema` 是 `BaseModel` 的子类，因此这是一个 Pydantic 类。

#3 我们有一个标记为可选的 ID 字段，因为它将由数据库分配。

#4 `ach_files_id` 字段是必须的，因为它是一个外键，指向已上传的文件。

### 5.10.2 为解析的 ACH 记录创建 Pydantic 架构

在这个阶段，对于每个解析记录的 Pydantic 定义并不那么有趣，因为我们只是继承了 Pydantic 的 `BaseModel` 记录并定义了必要的字段。我们将在稍后扩展字段以用于约束、验证和文档。目前，我们只需保持它们简单。

##### 列表 5.34  ACH 批次头记录的 Pydantic 架构

```py
from pydantic import BaseModel, UUID4

class AchBatchHeaderSchema(BaseModel):
    ach_records_type_5_id: UUID4
    record_type_code: str
    service_class_code: str
    company_name: str
    company_discretionary_data: str
    company_identification: str
    standard_entry_class_code: str
    company_entry_description: str
    company_descriptive_date: str
    effective_entry_date: str
    settlement_date: str
    originator_status_code: str
    originating_dfi_identification: str
    batch_number: str
```

### 5.10.3 单元测试更改

作为代码重构的一部分，我们确保我们的测试也得到了清理（列表 5.35）。首先，我们更新了 `setup_teardown_method` 以将 `autouse` 设置为 `true`，并确保首先执行 `SqlUtils.truncate_all` 方法。我们可能之前选择在测试运行后清除表，这是一个很好的实践，可以清理测试中的任何数据。然而，这也带来了一个不幸的副作用，即在测试失败时也会清理数据，这在我们想要在测试后检查数据库时并不是很有帮助。为了使调试和故障排除更容易，我们决定在测试之前清除数据。这也确保了数据库已准备好，因为我们不需要依赖之前的测试来自动清理。添加 `autouse` 参数意味着我们不再需要将 fixture 传递给我们的测试。我们还使用了 `truncate_all` 而不是特定的表，因为我们现在使用了多个表。

##### 列表 5.35 更新的 `pytest` 固定装置

```py
@pytest.fixture(autouse=True)
def setup_teardown_method():
    SqlUtils.truncate_all()
    yield
```

## 5.11 设计和不同方法

向数据库添加外键需要进行相当多的重工作，所以我们可能会考虑如何最小化这项工作，或者为什么我们没有一开始就添加它们。正如你所见，我们深入解析了 ACH 文件，并完成了这项工作，然后转向支持一些需要大量重工作的附加功能。这主要是因为我们想在深入研究整体功能和额外的数据库知识之前，先展示一些 ACH 基础知识。

然而，让我们考虑一下我们是否从功能角度来处理这个问题，以及用户是否需要上传这些文件。如果我们当时从`ach_files`表及其相关端点开始，我们就可以从一开始就包含外键。假设你具备处理 ACH 文件和 API 的知识和经验，这肯定是一个有效的方法。然后我们可以以同样的方式继续进行，唯一的区别是我们可能从一开始就有一个设计得更好的数据库。

这只是说明初始设计同样重要，以及你解决问题的方法。我们之前讨论过启用故事，或者也称为研究尖峰。它们可以降低风险，帮助我们理解需求，并更好地理解需要执行的任务。我们无法强调这些类型故事的重要性，尤其是在处理大型复杂系统时。我们总是可以预期在项目进行过程中需要一些重工作，无论是由于未识别的需求还是无法预见的范围变化。希望研究尖峰可以帮助最小化这些情况，但企业往往难以将时间投入到他们看不到即时收益的事情上。这显然会成为一个问题，因为通过启用故事暴露的问题将不会被识别，在 PI 规划中也不会被发现。我们最终可能采取的方法可能需要重工作，而这些重工作本可以在早期识别出来，因此我们的故事点可以正确分配。

我们已经看到了许多例子，一个项目似乎已经完成，但最终某个主题专家指出在系统演示中遗漏了某些内容，需要重工作。重要的是要记住，我们有时必须处理这些类型的状况，因为这可能会经常诱使我们选择一条更简单的出路。在这种情况下，我们需要评估变化对项目时间表、重工作量、风险、技术债务等因素的影响。例如，企业可能决定我们不需要数据库的引用完整性，并且编写一个可以手动运行的程序来检查悬空记录/缺失关系会更快。

## 摘要

+   本章展示了如何创建一个数据库，该数据库可以存储我们的未更改记录，以及一个解析后的记录版本，这在需要提供加载的 ACH 文件详细信息时是有益的。

+   我们在后期过程中看到了向数据库添加引用完整性的影响，以及需要重新编写我们的代码和单元测试的需求。当一个功能需要重新工作才能正确实现时，它通常可以被放在次要位置。

+   尽管实现这些类型的功能需要额外的工作，但向团队成员和管理层倡导它们同样重要，以确保它们不会被忽视。

+   定义数据库对于应用程序中的数据存储、查询和完整性至关重要。

+   当作为平面文件处理时，ACH 文件在性能、查询和数据完整性方面面临挑战。

+   关系数据库提供优势，例如主键和外键、约束和数据完整性。

+   实现引用完整性可以防止孤立记录并确保数据库一致性。

+   在评估数据库设计和各种实现方法时，研究激增（使能故事）是有益的。

+   ELT 和 ETL 在处理 ACH 文件和处理错误方面提供不同的好处。

+   Pydantic 帮助建模数据库表，抽象 SQL，并增强文档和验证。

+   上传文件和集成 API 是扩展 ACH 系统功能的基础。

+   数据和引用完整性对于关系数据库至关重要，可以防止错误并提高可靠性。

+   持续测试、重构和重新审视初始设计选择有助于维护和改进数据库性能和结构。
