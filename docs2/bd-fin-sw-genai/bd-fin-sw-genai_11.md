# 第九章：搜索和审计

### 本章涵盖

+   对我们的 ACH 仪表板的另一轮迭代

+   为 ACH 交易添加搜索功能

+   用户行为审计

在前面的章节中，我们构建了最小可行产品（MVP），经历了一个迭代过程，我们收到了反馈并利用它来推动下一组增强功能。本章将继续使用相同的概念。随着对异常处理和多种 UI/UX 调整的扩展支持发布，我们应该期待对这些功能和用户所需的其他功能反馈。

用户希望能够根据几个标准搜索交易，因为他们需要处理客户关于交易何时处理的投诉。此外，其他利益相关者要求能够审计与仪表板的交互，以衡量客户参与度并设置未来的用户跟踪。

## 9.1 迭代计划

在敏捷世界中，团队需要在开始任何工作之前为迭代做准备。我们已经有了关于这个迭代需要完成什么工作的想法，但还必须定义故事，以确切知道任何给定故事中期望的内容，并确定可能与之相关的任何验收标准。我们继续遵循我们故事的标准格式：“作为[角色]，我[想要]，[以便]”。例如，为了向我们的产品添加搜索功能，我们可能有一个类似这样的故事：“作为用户，我想要能够搜索交易，以便我能够更好地研究客户关于他们交易的询问”。请随意练习剩余任务中附加用户故事的措辞（即使只是在脑海中），我们将在本章中涵盖这些任务。

我们花点时间使用另一个 Gantt 图来可视化这些变化。同样，图表展示了我们变化的时序，可以向任何对项目感兴趣的利益相关者展示。根据我们的经验，这是在项目即将完成时与利益相关者沟通最有效的方法之一。通常，我们团队之外的利益相关者可能不熟悉更敏捷的进度跟踪方式，他们只想知道我们的代码何时可用。一个简单的 Gantt 图可以给他们一个关于工作进展情况的想法。以下列表显示了 PlantUML 中的 Gantt 图定义。

##### 列表 9.1 PlantUML Gantt 图定义

```py
@startgantt    #1
saturday are closed  #2
sunday are closed    

header ACH Dashboard Enhancements - Round 2  #3

Project starts 2024-08-15
-- Searching Enhancement --    #4
[Search API Design] starts 2024-08-15 and lasts 2 days   #5
[UI Design/Changes] starts at \   #5
            [Search API Design]'s end and lasts 4 days   #5

-- Auditing Enhancement -- #5
[Database Design] starts at 2024-08-15 and lasts 2 days #5
[Review APIs] starts at [Database Design]'s #5
➥ end and lasts 1 days #5
[Add Auditing logic] starts at [Review APIs]'s end #5
➥ and lasts 3 days #5
[UI Support for Auditing] starts at \   #5
            [Add Auditing logic]'s end and lasts 3 days  #5

[Sprint Complete] happens at 2024-08-28   #6

@endgantt  #7
```

#1 开始 Gantt 图定义

#2 定义应跳过的天数

#3 我们图表的标题

#4 定义搜索增强的时序；注意，长行可以使用反斜杠字符拆分为多行。反斜杠后面的任何尾随空格都可能导致解析错误。

#5 定义审计增强的时序

#6 定义里程碑

#7 结束 Gantt 图定义

图 9.1 展示了生成的甘特图。它不仅为我们提供了我们在开发过程中的路线图，而且创建图表有助于我们开始思考任何给定增强的步骤和要求。

![日历截图  自动生成的描述](img/CH09_F01_Kardell.png)

##### 图 9.1 甘特图显示时间框架

我们现在应该对需要采取的步骤有一个概念，可以立即开始工作。

## 9.2 搜索 ACH 交易

哪个文件包含了那笔交易？那笔交易是什么时候加载的？这是我们希望通过这次增强能够回答的问题类型。在这本书的整个过程中，我们一直关注预先安排的支付和存款（PPD）批次，其中包含了工资、公用事业费和抵押贷款支付等。如果客户的公用事业费处理不当，他们的水/电被切断，我们可以肯定我们会听到关于这件事的消息。当然，这是一个极端情况。更有可能的是，客户已经收到特定公司的通知，他们的付款尚未收到，他们必须支付滞纳金。无论如何，当客户打电话来时，他们都不会太高兴。

我们需要考虑添加在加载的文件中搜索名称和金额的能力。这个功能将需要更改 UI，因为我们需要能够输入搜索信息，并需要一个新 API 来支持实际的搜索和返回结果。我们当前的关注点是能够在我们的有限数据集中找到交易。在实践中，ACH 记录应从收据日期起保存六年，并且每年有数十亿笔支付通过 ACH 网络流动，因此我们可能会拥有大量的数据集需要搜索。在项目准备投入生产之前，我们需要评估其他搜索机制，除了我们的实时方法之外。

### 9.2.1 BDD 测试搜索

在组合一个功能以帮助验证新的（尚未编写的）搜索 API 之前，我们首先必须创建一个我们可以用于我们功能的文件。我们可以使用我们现有的文件创建功能（ach_file_creation.feature）并添加一个新的场景来为特定用户创建一个已知金额的文件。虽然我们可以在现有的文件中进行搜索，但将文件与功能分开以保持更清洁，这样我们的测试将更加稳健。以下列表展示了为顾客萨莉·萨弗创建一个文件，其中包含一笔单美元的交易，这应该足以让我们开始。

##### 列表 9.2  为萨莉·萨弗创建文件

```py
Scenario: Create an ACH file with a single transaction
➥ for customer "Sally Saver"
   Given I want to create an ACH file named "sally_saver.ach"
   And I want to have an immediate destination of "990000013"
   And I want to have an immediate origin of "987654321"
   And I want to have 1 batch with ACH credits only and
➥ a standard entry class code of "PPD"
   And I want 1 entries per batch with random amounts between 100 and 100
   And I want to use individual names of "Sally Saver"
   And I want to have company name "My Company" and company id "1234567890"
   When my ACH is created
   Then I should have a file of the same name
   And there should be 1 batch in the file
   And there should be 1 entries in the file
```

在创建完文件后，我们现在可以开始设置我们的 BDD 测试来搜索交易。这应该会让我们对整个过程的工作方式有一些了解。例如，我们知道我们想要能够通过至少个人名称和金额进行搜索。我们项目中的各种利益相关者可能需要讨论的一些问题可能包括

+   我们是否将在 API 端将这些搜索分开（例如，允许`customer`和`amount`的查询参数）？还是我们将允许一个搜索字段，并让后端尝试查询正确的字段？

+   字段是否互斥？如果我们正在搜索客户，我们也能搜索账户吗？

+   如果我们允许使用多个字段进行搜索，这些字段将使用 AND 还是 OR 组合？这应该由用户决定吗？

提出这些问题（以及它们的答案）将有助于指导我们的设计过程。目前，我们将采用一种方法，即向用户展示一个单独的搜索框，该搜索框将支持我们想要执行的任何搜索。我们还将把决定用户想要搜索什么的责任放在服务器端，而不是用户界面端。另一种方法是将 UI 分割字段，并允许 API 接受不同的查询参数。在考虑了我们的搜索方法后，我们可以创建一个场景来测试我们的 API 搜索，使用客户名称作为测试。

##### 列表 9.3  按个人名称搜索

```py
Scenario: Search by individual name
   Given that I have a clean database
   And that I have posted the file "sally_saver.ach"
   When I search for transactions with search criteria "Sally Saver"
   Then I should receive a list of "1"
➥ transaction with the amount of "1.00"
```

需要添加新的语法来支持`When`和`Then`子句，以便支持我们的交易搜索。代码如下所示。

##### 列表 9.4  更新 test_file_api.py

```py
…
scenarios("../features/search_ach_transactions.feature") #1
…
@when(parsers.parse('I search for transactions #2
➥ with search criteria \"{search_criteria}\"')) 
def search_request_for_transactions(api_response, #3
➥ search_criteria):  #3
    response =  #3
 client.get(f"/api/v1/files/transactions/search?  #3
➥ criteria={search_criteria}")  #3
    assert response.status_code == 200, response.text #3
  api_response["response"] = response.json()  #3
…
@then(parsers.parse('I should receive a list of #4
➥ \"{transaction_count}\" transaction with the amount of #5
➥ \"{transaction_amount}\"')) 
def has_expected_transaction_and_amount(api_response, transaction_count, transaction_amount): #5
    assert len(api_response["response"]) == 1, f"Expected #5
{transaction_count} transaction, but got #5
➥ {len(api_response['response'])}"  #5
    response = api_response["response"][0]["amount"]  #5
    assert transaction_amount == response,  #5
➥ f"Expected {transaction_amount} in {response}"  #5
```

#1 确保新场景已加载

#2 使用@when 注解方法并解析行

#3 定义一个函数来调用我们的方法并保存响应

#4 使用@then 注解方法并解析行

#5 定义一个函数来验证 API 响应的结果

现在我们有了测试背后的支持代码，如果我们运行它，我们应该期待看到什么？没错——会返回一个`404 错误`！这是可以预料的，因为我们还没有定义端点。在下一节中，我们将通过添加 API 来解决这个问题。

### 9.2.2 搜索 API

在这一点上，我们希望有一个 API，它从用户那里获取一些输入，然后在数据库中搜索交易。如前所述，我们只将搜索个人名称和金额。我们期望 API 使用一个名为`criteria`的单个查询参数被调用。API 将确定参数是否适用于个人名称或金额。

这并不是我们唯一可能采取的方法。我们还可以将搜索条件拆分为单独的字段，这样会使服务器端更容易一些，但可能从用户的角度来看搜索会更加繁琐。另一种方法是将指定搜索条件与关键词（例如，允许搜索如`individual` `name` `=` `"user"` `and` `amount` `=` `$1.00`）相结合。然后我们需要进行更多的解析以获取我们想要使用的字段。另一个考虑因素是我们要使用`GET`还是`POST`。如前所述，我们的示例使用`GET`，并传递一个查询参数，这是我们将会搜索的内容。这种方法是可行的，但带有查询参数的 URL 可能会出现在网络服务器、防火墙、应用程序和第三方日志中。如果我们启用搜索可能敏感的信息，如账户和税务识别号码，我们可能需要重新考虑这种方法，因为我们不希望这样的敏感信息出现在日志中。没有错误的答案——根据我们的受众和我们所期望的搜索内容，我们可能会被告知采用特定的方法。

我们首先构建一个简单的端点来处理当前返回的`404` `HTTP`状态码。以下列表显示了端点的初始实现。

##### 列表 9.5  搜索的初始端点定义

```py
@router.get(   #1
    path="/transactions/search",  #1
    response_model=list[str],  #1
    summary="Retrieve ACH Transactions",  #1
    description=  #1
      "Retrieve the transactions matching #1
➥ the specified criteria.",  #1
    response_description="ACH Transactions.", #1
    response_model_exclude_none=True, #1
    tags=["ACH Files"], #1
) #1
async def get_ach_transactions(criteria: str = Query(..., #2
  description="Search criteria for transactions")) #2
➥ -> list[str]: 
    return [f"ACH Transactions for {criteria}"] #3
```

#1 定义带有支持文档的 GET 请求

#2 定义指定查询参数的方法

#3 返回包含查询参数的硬编码数组条目

虽然这段代码可能帮助我们绕过`404 错误`，但它并没有为我们创建一个有用的端点做出太多贡献。为此，我们需要构建 SQL 查询和返回数据的响应对象。我们首先创建一个`TransactionSearchResponse`类来存储我们的数据。以下列表显示了`TransactionSearchResponse`和预期的字段。如果我们发现我们缺少某些内容，我们总是可以回来添加。

##### 列表 9.6  `TransactionSearchResponse` 类

```py
from decimal import Decimal   #1
from typing import Annotated #1
from pydantic import BaseModel, Field, UUID4

class TransactionSearchResponse(BaseModel): #2
    file_id: UUID4 = Field( #3
        ..., description=  #3
➥"Unique identifier for the ACH file.",  #3
        title="File ID"  #3
    )  #3
    batch_header_id: UUID4 = Field(  #3
        ...,  #3
        description=  #3
➥"Unique identifier for the ACH Batch Header.",  #3
        title="Batch Header ID",  #3
    )  #3
    entry_id: UUID4 = Field(  #3
        ...,  #3
        description=  #3
➥"Unique identifier for the ACH Entry Detail.",  #3
        title="Entry Detail ID",  #3
    )  #3
    filename: str = Field(  #3
        ..., description="The name of the ACH file.",   #3
        title="Filename", max_length=255 #3
    )  #3
    individual_name: str = Field(  #3
        ...,  #3
        description=  #3
➥"The name of the individual or company for the entry.",  #3
        title="Individual Name",  #3
        max_length=22,  #3
    )  #3
    amount: Annotated[  #3
        Decimal,  #3
        Field(  #3
            ...,  #3
            description="The amount of the entry.",  #3
            title="Amount",  #3
            max_digits=10,  #3
            decimal_places=2,  #3
        ),  #3
    ]  #3
 #3
    class Config:  #3#4
        json_schema_extra = { 
            "example": { 
                "file_id": 
➥"123e4567-e89b-12d3-a456-426614174000", #5
  "record_type_5_id": 
➥ "123e4567-e89b-12d3-a456-426614174001", 
                "record_type_6_id":  
➥"123e4567-e89b-12d3-a456-426614174002", 
                "filename": "test.ach", 
                "individual_name": "John Doe", 
                "amount": "100.00", 
            } 
        } 
```

#1 我们的标准导入语句用于 Pydantic 字段

#2 使用 Pydantic 的 BaseModel 作为基类定义类

#3 定义我们类的各种字段和约束

#4 提供用于文档的响应对象示例

#5 提供用于文档的响应对象示例

现在我们有了存储响应的地方，我们需要填充它。我们通过使用一个新的类`TransactionSearchSql`来完成，这个类将存储任何与交易搜索相关的代码。我们希望在我们的第一次搜索尝试中保持简单，所以我们处理以下三种情况：

+   通过单个金额进行搜索

+   通过金额范围进行搜索

+   通过个人名称进行搜索

我们创建了一个名为 `get_transactions` 的方法，如下所示。在这里，我们有两个正则表达式来确定是否传递了金额或使用了金额范围。否则，我们默认使用名称。对于大多数人来说，代码中最棘手的部分可能是正则表达式的使用。

##### 列表 9.7  `get_transaction` 方法

```py
def get_transactions(self, criteria: str) -> #1
   list[TransactionSearchResponse]: 
   amount_pattern = r"^\d+\.\d{2}$" #2
   multiple_amounts_pattern = #3
➥ r"(^\d+\.\d{2})\s+(\d+\.\d{2})$" 
   if re.match(amount_pattern, criteria): #4
      return self._get_transactions_using_amount(criteria) 
   elif match := re.match(multiple_amounts_pattern,  #5
➥ criteria): 
      begin_amount, end_amount = match.groups() #6
      return self._get_transactions_using_amount_range #7
➥(begin_amount,➥ #8
 end_amount) 
   else: #8
      return self._get_transactions_using_➥ #9
➥individual_name(criteria) 
```

#1 定义了一个返回交易列表的函数

#2 匹配单个金额的模式

#3 匹配两个金额的模式

#4 将标准与单个金额进行比对并调用相应的方法

#5 使用 walrus 运算符存储 multiple_amounts_pattern 的匹配结果

#6 使用 match 的 groups() 方法检索金额

#7 使用检索到的参数调用方法

#8 否则，没有匹配的金额，我们应该使用单个名称。

在继续之前，让我们分解正则表达式以确保你理解它们的用法。以下是一些重要点：

+   每个模式字符串都以前缀 `r` 开头，表示原始字符串，这防止了我们不得不转义任何字符串。否则，我们不得不使用 `\\` 而不是 `\`。

+   字符 `^` 和 `$` 是输入边界断言，分别表示字符串的开始和结束。这些帮助我们消除包含我们感兴趣模式之外字符的匹配字符串。

+   `\d` 匹配任何数字（0–9），而 `+` 是一个量词，表示匹配前一个原子（正则表达式的最基本单元）的一个或多个出现。

+   `\.` 匹配实际的点字符，因为单个点否则会匹配任何字符。

+   `\d` 匹配任何数字（0–9），而添加 `{2}` 是另一个量词，指定要匹配的原子数量。

+   `\s` 匹配任何空白字符。同样，我们看到加号量词，它允许两个金额之间至少有一个字符，并且可以有任何数量的空白字符。

+   将正则表达式的一部分括起来创建一个捕获组。然后，你可以稍后引用这些组，正如我们在使用 `match.groups()` 提取找到的值时所看到的那样。

正则表达式不仅限于金融行业，你将在整个职业生涯中找到它们。虽然它们可能难以掌握，但生成式 AI 工具是帮助分解正则表达式并提供更多见解的绝佳方式。熟能生巧，为了获得更多正则表达式的实践经验，你可能想查看 David Mertz 的 *正则表达式谜题和 AI 编码助手*（2023，Manning）。

在此基础上，让我们看看根据提供的搜索条件实际检索交易的函数。在我们的案例中，查询本身保持不变，只是每个查询的 `Where` 子句是唯一的。以下列表显示了 `_get_transactions_using_amount` 函数，该函数查找精确金额并返回任何结果。

##### 列表 9.8  通过金额检索事务

```py
    def _get_transactions_using_amount(
        self, criteria: str
    ) -> list[TransactionSearchResponse]:
        with get_db_connection(
            row_factory=class_row(TransactionSearchResponse) #1
        ) as conn:
            sql = self.get_sql_selection_query() #2
            sql += "WHERE aepd.amount = %s" #3
            result = conn.execute(sql, [criteria]) #4
            return result.fetchall() #5
```

#1 我们声明该行将为 TransactionSearchResponse 类型。

#2 我们试图选择的常规列

#3 为查询添加 Where 子句

#4 使用我们想要传递给查询的标准

#5 获取所有行并返回响应

我们已经将 SQL 查询的公共部分提取到 `get_sql_selection_query` 方法中，这使得可以在查询中添加 `Where` 子句。当通过金额搜索事务时，我们使用了精确匹配。当使用金额范围时，我们使用 Postgres 语法并使用关键字 `BETWEEN`。

##### 列表 9.9  使用 `BETWEEN` 关键字

```py
sql += "WHERE aepd.amount BETWEEN %s AND %s" #1
```

#1 使用 BETWEEN 在两个金额之间进行搜索；通常比多个条件更简洁、更易读

类似地，当搜索单个名称时，我们使用 `ILIKE` 关键字，如列表 9.10 所示。使用 `ILIKE` 允许我们进行不区分大小写的搜索——否则，我们是在搜索相同的字符串。请注意，ILIKE 是 Postgres 特有的命令。其他数据库如 MySQL 使用 `LIKE` 关键字，但默认情况下不区分大小写。仍然，其他数据库如 Oracle 是区分大小写的，并且需要在比较中使用 `UPPER` 函数来实现不区分大小写的搜索。所以，始终要意识到你正在使用的数据库。

我们可以考虑在标准中添加一个百分号（`%`），这是一个 SQL 通配符，如果其后有任何内容出现，它将匹配单个名称。或者，我们可能可以将字符串中找到的任何空格替换为通配符以扩展我们的搜索结果。如果需要额外的搜索功能，这些只是基本起点。

##### 列表 9.10  不区分大小写的搜索

```py
sql += "WHERE aepd.individual_name ILIKE %s" #1
```

#1 使用 ILIKE 对单个名称进行不区分大小写的搜索。

以下列表显示了检索事务信息的查询的公共部分。除了需要将记录连接起来以深入事务细节之外，我们还重命名了字段以确保它们可以存储在我们的响应对象中。

##### 列表 9.11  常见事务查询选择标准

```py
def get_sql_selection_query(self):
   return """
          SELECT art1.ach_files_id AS file_id, #1
                 art5.ach_records_type_5_id AS batch_header_id,  #2
                 art6.ach_records_type_6_id AS entry_id,   #2
                 af.file_name AS filename,   #2
                 aepd.individual_name AS individual_name,  #2
                 aepd.amount AS amount   #2
            FROM ach_files af #2
      INNER JOIN ach_records_type_1 art1 #2
➥ USING ( ach_files_id )  #2
      INNER JOIN ach_records_type_5 art5 #2
➥ USING ( ach_records_type_1_id )  #2
      INNER JOIN ach_records_type_6 art6 #2
➥ USING ( ach_records_type_5_id )  #2
      INNER JOIN ach_entry_ppd_details aepd #2
➥ USING ( ach_records_type_6_id )  #2
   """
```

#1 我们对象所需的选取标准。字段前缀有助于确定它们所属的文件。当名称唯一时，这并不总是必要的，但对于维护代码是有帮助的。

#2 连接各种记录

当查询运行并返回我们的数据时，我们现在应该看到所有测试都已通过。接下来，我们看看如何将搜索功能添加到我们的用户界面中。

### 9.2.3 UI 搜索页面

要将搜索功能添加到我们的页面中，我们首先在侧边栏导航中添加一个图标。这意味着我们可以更新 NavButtons.tsx 来添加一个带有图标的按钮。

##### 列表 9.12  更新 NavButtons.tsx

```py
…
import {CloudUpload, Logout, Error, Search} from "@mui/icons-material"; #1
…
            <ListItemButton onClick={() => route.push("/search")}>
                <ListItemIcon>
 <Search/> #2
                </ListItemIcon>
                <ListItemText primary="Search"/>
            </ListItemButton>
…
```

#1 导入搜索图标

#2 包含搜索图标

当然，如果我们尝试点击按钮，我们会收到一个`404` `NOT` `FOUND`错误消息，因为我们还没有定义我们的实际页面。与之前的页面一样，我们将创建一个简单的页面，这将允许我们解决`404`错误消息，并为我们开发页面的其余部分提供一个起点。以下列表显示了`src/app/search/page.tsx`。

##### 列表 9.13  搜索页面草稿

```py
export default function SearchPage() { #1

    return (
        <ThemeProvider theme={defaultTheme}>
            <Box sx={{display: 'flex'}}>
                <CssBaseline/>
                <StandardNavigation/>
                <Box
                    component="main"
                    sx={{
                        backgroundColor: (theme) =>
                            theme.palette.mode === 'light'
                                ? theme.palette.grey[100]
                                : theme.palette.grey[900],
                        flexGrow: 1,
                        height: '100vh',
                        overflow: 'auto',
                    }}
                >
                    <Toolbar />
 <Typography>Search Page</Typography> #2
                </Box>
            </Box>
        </ThemeProvider>
    );
}
```

#1 创建一个简单的搜索页面

#2 用来识别页面，以便我们知道我们已经导航到该页面

在我们的页面草稿完成后，我们可以点击搜索按钮，应该会看到我们的导航和搜索页面。现在，我们可以添加必要的组件，以创建一个能够显示结果的搜索页面。最难的部分是决定如何布局页面。从之前的 API 调用（例如第八章中返回的异常），我们已经知道如何从 API 获取数据并在 DataGrid 组件中显示它。我们将遵循相同的步骤：定义一个用于存储数据的对象，一个用于显示数据的组件，然后将它添加到我们的搜索页面中。

我们首先定义一个接口`AchTransactionSearchResponse`来存储 API 响应将传递回的数据。以下列表显示了字段及其数据类型。

##### 列表 9.14  `AchTransactionSearchResponse` 接口

```py
export interface AchTransactionSearchResponse { #1
    file_id: string;  #2
    batch_header_id: string;  #2
    entry_id: string;  #2
    filename: string;  #2
    individual_name: string;  #2
    amount: string;  #2
}
```

#1 导出`AchTransactionSearchResponse`使其在所有地方可用

#2 API 返回的字段

一旦我们有一个地方来存储检索到的数据，我们就可以创建一个组件来显示这些数据。这个`AchTransactionSearchResults`组件将类似于我们创建的其他组件，我们将通过`AchTransactionSearchResultsProps`的结果对象将 API 调用的结果传递给组件。我们利用列定义上的`renderCell`属性添加链接，允许用户直接跳转到特定搜索结果的文件或批次。此外，由于我们的结果没有 ID 列，我们必须使用`getRowId`属性在 DataGrid 组件中定义一个。`entry_id`可以作为我们的 ID 列，因为对于每个结果，`entry_id`是交易的 UUID，它是唯一的。以下列表显示了组件的更重要的部分。

##### 列表 9.15  `AchTransactionSearchResults` 组件

```py
interface AchTransactionSearchResultsProps {
    results: AchTransactionSearchResponse[];
}

export default function AchTransactionSearchResults #1
➥({results}:  #1
Readonly<AchTransactionSearchResultsProps>) {  #1

    const route = useRouter(); #2

    const columns: GridColDef[] = [
        {field: 'filename', headerName: 'Filename', #3
➥ width: 150}, 
        …
        {field: 'viewFile', headerName: '',  #4
                 sortable: false, width: 150,   #4
                 renderCell: (params) => (   #4
            <Link onClick={() =>  #4
                route.push(`/fileDetails/${params.row.file_id}`)}  #4
                sx={{ cursor: 'pointer' }}>Jump to file...</Link>  #4
            )},  #4
        {field: 'viewBatch', headerName: '', #5#6
                 sortable: false, width: 150,   #6
                 renderCell: (params) => (  #6
            <Link onClick={() =>   #6
route.push(`/fileDetails/${params.row.file_id}/   #6
➥batchDetails/${params.row.batch_header_id}`)}   #6
              sx={{ cursor: 'pointer' }}>Jump to batch...</Link>  #6
            )}  #6
    ];

    return (
…
       <DataGrid columns={columns} 
                 rows={results} 
                 getRowId={(row) => row.entry_id} #7
       />
…
    );
}
```

#1 导出搜索结果函数并声明参数为只读

#2 使用路由进行导航目的

#3 开始定义要显示的列

#4 定义一个列，该列将创建一个链接，点击后将导航到文件详情页面

#5 定义一个列，该列将创建一个链接，点击后将导航到批量详情页面

#6

#7 由于没有名为 id 的具体参数，我们指定哪个字段可以使用。

现在我们有了存储数据的地方和展示数据的方式，剩下要做的就是调用 API 并将其传递给我们的组件。为此，我们更新了我们的搜索页面，添加了一个文本输入字段和一个搜索按钮来执行实际的 API 调用。我们使用 `TextField` 的 `onChange` 事件将用户输入的文本存储到 `searchCriteria` 字段中。当用户点击搜索按钮时，将触发 `onClick` 事件，并执行 `handleSearch`。`handleSearch` 调用我们的 API 并存储结果。更新的搜索页面的重要部分如下所示。

##### 列表 9.16  更新的搜索页面

```py
…
export default function SearchPage() {
    const [searchCriteria, setSearchCriteria] = #1
➥ useState<string>(''); 
    const [results, setResults] = #2
useState<AchTransactionSearchResponse[]>([]); 

    const handleChange = (event: { target: { value: #3
React.SetStateAction<string>; }; }) => {  #3
        setSearchCriteria(event.target.value);  #3
    };  #3

    const handleSearch = async () => { #4
        …        axios.get<AchTransactionSearchResponse[]> #5
➥(`${apiUrl}/files/transactions/search?criteria=${searchCriteria}`, { 
…
    return (
…
       <Box sx={{display: 'flex', 
                 flexDirection: 'column', 
                 alignItems: 'center', gap: 2}}>
          <TextField id="search" #6
                     label="Search"  #6
                     variant="standard"   #6
                     onChange={handleChange}   #6
                     sx={{ width: '40%' }} />  #6
          <Button variant="outlined" #7
                  color="primary" onClick={handleSearch}>  #7
             Search #7
          </Button> 
       </Box>
       <AchTransactionSearchResults results={results}/> #8
…
    );
}
```

#1 将搜索标准初始化为空字符串

#2 将结果初始化为空字符串

#3 handleChange 函数将输入的文本保存以供以后使用。

#4 handleSearch 函数调用 API 并使用 setResults 保存结果。

#5 使用 axios 获取响应

#6 定义一个用于用户输入搜索字符串的 TextField 元素

#7 定义一个将搜索输入字符串的按钮元素

#8 一个用于显示结果的组件

现在我们有一个可以执行基本交易搜索并提供直接跳转到找到交易的文件或批次的搜索页面。接下来，我们通过 Playwright 加强测试，以获取我们搜索功能的一些集成级别测试。

### 9.2.4 使用 Playwright 进行集成测试

在我们使搜索我们的交易成为可能之后，添加一个集成测试将是一个很好的选择，我们可以用它来进行集成和回归测试。我们设置了一个测试，需要我们

1.  上传文件以解析

1.  导航到搜索页面

1.  填充搜索页面

1.  等待 ACH 响应

1.  验证结果是否已填充

1.  拍摄屏幕截图

我们创建了一个名为 test_search_page.py 的 Python 脚本来执行所有这些必要的步骤。此外，我们希望有选项可以在浏览器窗口显示的情况下调试我们的测试；为了实现这一点，我们需要创建一些 Pytest 固定装置。

##### 列表 9.17  Playwright 的 Pytest 固定装置

```py
@pytest.fixture(scope="session") #1
def browser(): #2
    with sync_playwright() as p:  #3
        browser = p.chromium.launch(headless=False) 
        yield browser #3
        browser.close() #4

#5
@pytest.fixture(scope="function") 
def page(browser): #6
  context = browser.new_context()  #7
 pages = context.pages
 page = pages[0] if pages else context.new_page() #7
  yield page #8
    context.close() #9
```

#1 定义一个具有会话范围的固定装置，这将导致固定装置在测试会话期间持续存在

#2 定义一个名为 browser 的函数，用于创建浏览器实例。我们使用 headless=False 以逐步进行测试会话并查看浏览器正在做什么。使用 headless=True 将隐藏浏览器窗口，这是运行单元测试的默认方式。

#3 yield 浏览器命令将返回浏览器对象。

#4 当所有测试完成后，关闭浏览器。

#5 定义一个具有函数范围的固定装置，这是默认范围

#6 创建一个用于测试的页面

#7 如果存在，则使用默认选项卡；否则，创建一个新的选项卡

#8 使用 yield 返回页面，以便在测试中可用

#9 测试运行后清除上下文

这就足够设置 Playwright 结构了。现在，我们想要确保我们的数据库为空，并在开始集成测试之前加载我们的 ACH 文件。

##### 列表 9.18  清空数据库并加载我们的文件

```py
@pytest.fixture(autouse=True) #1
def setup_teardown_method():
    SqlUtils.truncate_all() #2

    ach_file = "./data/sally_saver.ach" #3
    file_path = get_absolute_path("./data/sally_saver.ach")  #3
    parser = AchFileProcessor()  #3
    ach_files_id = SqlUtils.create_ach_file_record #3
➥(ach_file, str(randint(1, 99999999)))  #3
    parser.parse(ach_files_id, file_path)  #3
    yield #3

def get_absolute_path(relative_path): #4
    return current_file_dir / relative_path
```

#1 定义一个设置为 autouse 为 True 的 fixture。这会导致它默认包含在每个测试中。

#2 清空我们的数据库

#3 将测试文件加载到数据库中

#4 确保路径正确，基于我们从哪里运行

现在设置完成，我们可以开始使用 Playwright 进行实际测试。

##### 列表 9.19  Playwright 测试

```py
def test_dashboard(page: Page):
    page.goto("http://localhost:3000/search") #1
    page.expect_navigation(wait_until="load") 
    expect(page).to_have_url("http://localhost:3000/search") #2
    search_criteria = page.get_by_role("textbox") #3
    search_criteria.fill("sally saver") 
    search_button = page.locator("#searchbtn") #4
    with page.expect_response("**/files/transactions/search*") #5
➥ as response_info:  #6
        search_button.click() 

    response = response_info.value #6
    assert response.status == 200
    search_result = page.locator('div[title="Sally Saver"]') #7
    expect(search_result).to_be_visible() 
    page.screenshot(path="screenshots/search_results.png") #8
```

#1 导航到搜索屏幕并确保页面已加载

#2 验证我们是否在正确的页面上

#3 找到文本框并输入文本“sally saver”

#4 找到搜索按钮

#5 预期当点击搜索按钮时会有 API 响应

#6 检查响应状态码为 200

#7 我们应该能够看到一个包含文本“Sally Saver”的条目。

#8 为永存性拍摄屏幕截图

现在你已经知道了如何执行一些集成测试，你可能需要调用 API 并验证结果。在下一节中，我们将探讨许多行业中开发的一个关键方面：保持审计跟踪。

## 9.3 审计用户交互

互联网安全中心提出的几个关键安全控制之一是审计日志管理（[`mng.bz/gaGE`](https://mng.bz/gaGE)）。虽然我们不会涵盖所有方面，但我们将至少从确保我们收集有关应用程序使用情况的数据开始。记住，收集数据是第一步。如果我们没有监控、审查和必要时接收警报的策略，日志就毫无用处。

许多时候，我们解决过的问题最初是因为有人识别到日志中发生错误而被引入了错误的方向。进一步的研究表明，错误已经发生数月（远至日志容易获取的时候）并且不是当前问题的源头。有时，在生产的紧急事件中，由于需要快速解决问题，你很容易陷入这种陷阱。通过适当的审计日志管理，你可以更好地理解基线应用程序行为，并监控应用程序滥用，以及其他许多好处。

尽管有许多现成的商业工具（COTS）如 Splunk、Dynatrace、DataDog 和 Sentry 可以帮助进行日志记录和可观察性，但我们主要关注将数据记录到数据库中，并在必要时从那里扩展。

### 9.3.1 数据库设计

如果我们想要开始进行审计，我们需要创建一个新的数据库表来存储我们的数据。这将基于您的应用程序需求以及您试图实现的目标。在我们的案例中，我们将跟踪进入系统的 API 请求。我们引入了一种新的数据类型来处理主机和网络——`ip_address`字段的`INET`数据类型。虽然使用字符串数据类型对于`ip_address`来说当然也可以工作，但 Postgres 提供了处理`INET`的额外比较和函数，这可以使生活更加轻松。例如，我们可以通过使用`WHERE ip_address` `<<` `'192.168.1.0/24'`来搜索给定范围内的所有地址。如果我们想查看来自我们怀疑滥用我们应用程序的地址列表的活动，这种搜索可能会有所帮助。以下列表显示了表的创建。

##### 列表 9.20  `audit_log`表

```py
CREATE TABLE audit_log ( #1
    audit_log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(), #2
    user_id VARCHAR(25) DEFAULT NULL,
    ip_address INET DEFAULT NULL, #3
    user_agent VARCHAR(255) DEFAULT NULL,
 http_request VARCHAR(10) DEFAULT NULL,
  url VARCHAR(255) DEFAULT NULL,
    message TEXT NOT NULL
);
```

#1 创建名为 audit_log 的表

#2 当记录使用 TIMESTAMP 数据类型创建时保存。我们还使用 NOT NULL，以确保字段必须始终被填充。最后，我们默认使用 Postgres 中的 NOW()函数，该函数将在记录插入时填充字段，如果没有提供值。

#3 创建一个名为 ip_address 的字段，其数据类型为 INET，这是一个方便处理网络地址的数据类型

在表就绪后，我们可以创建编写这些审计记录所需的代码。

### 9.3.2 为审计进行的 TDD 测试

我们首先为 API 创建单元测试，然后添加必要的代码以使测试通过。我们也应该考虑 BDD 风格的测试吗？这个答案取决于业务如何选择使用审计日志。如果日志主要作为开发者使用的东西，那么业务可能不需要与我们合作来开发测试。相反，如果业务计划使用审计日志来拉取活动以帮助确定指标，我们可能需要与他们合作来开发测试。我们可以很容易地看到业务提出那种类型的要求，并需要更新我们的测试以读取类似`WHEN` `I` `REQUEST` `A` `LIST` `OF` `FILES` `THEN` `I` `SHOULD` `SEE AN` `AUDIT` `LOG` `RECORD` `INDICATING` `FILES` `WERE` `SHOWN`的内容。现在，我们将坚持我们的标准单元测试。单元测试定义了一个记录然后检索它，以便我们可以比较字段。我们目前有一个`user_id`字段，但由于我们目前没有跟踪用户，我们将留空。

##### 列表 9.21  审计日志的单元测试

```py
class TestAuditLog:
    @pytest.fixture(autouse=True) #1
    def setup_teardown_method(self):  #1
        SqlUtils.truncate_all()  #1
        yield #1

    def test_insert_record(self): 
        log_record = AuditLogRecord( #2
            ip_address="192.168.1.0/24",  #2
            user_agent="Mozilla/5.0",  #2
            http_request="GET",  #2
            http_response=200,  #2
            url="/api/v1/files",  #2
            message="Test audit log insert",  #2
        )  #2
        logger = AuditLog() #3
        audit_record_id = logger.log_record(log_record)  #3
        retrieved_record =   #3
➥logger.get_log_record(audit_record_id)  #3
        excluded_fields = {"audit_log_id", "created_at"} #4
        assert SqlUtils.get_row_count_of_1("audit_log") is True
➥, "Expected 1 record" 
        assert retrieved_record.dict(exclude=excluded_fields) 
➥ == #D
     log_record.dict( 
            exclude=excluded_fields
        ), f"Expected {log_record}, 
➥ but got {retrieved_record}" 
```

#1 每次清除数据库

#2 创建一个虚拟记录

#3 使用 AuditLog 类创建和检索记录

#4 从字典中排除指定的字段，因为这些字段由数据库填充且不影响比较；然后断言记录匹配

单元测试需要一个 `AuditLogRecord` 对象，这是一个标准的 Pydantic 模型，用于存储我们的数据。在这个模型中，我们使用了两种新的数据类型——`IPvAnyNetwork` 和 `AnyUrl`（两者都可作为从 Pydantic 导入的导入项）。`IPvAnyNetwork` 对于表示我们在表中定义的 `INET` 字段效果很好，并且对于 IP 地址提供了预期的验证，无论它们是单个地址（例如，`127.0.0.1`）还是一系列地址（例如，`192.168.1.0/24`）。当然，我们不会期望在这个字段中看到地址范围。`AnyUrl` 字段类似地提供了对 URL 的验证，使我们免于编写正则表达式来验证字段。列表 9.22 显示了我们的 `AuditLogRecord` 类，为了简洁起见，删除了一些不太有趣的字段。

##### 列表 9.22  `AuditLogRecord` 类

```py
class AuditLogRecord(BaseModel):
…
    ip_address: Optional[IPvAnyNetwork] = Field( #1
        None,
        title="IP Address",
        description="IP address from which the request originated.",
    )
…
    url: Optional[AnyUrl] = Field( #2
        None, 
        title="URL", 
        description="URL accessed by the request.", max_length=255
    )
…
    @field_validator("url", mode="before") #3
    @classmethod #3
    def convert_url_to_str(cls, v: AnyUrl) -> str:  #3
        return str(v) 
```

#1 将 ip_address 定义为 IPvAnyNetwork 类型，这允许 Pydantic 验证 IP 地址和网络

#2 将 URL 定义为 AnyUrl 类型，这允许 Pydantic 验证 URL 是否格式正确

#3 定义了一个字段验证器，以便将 URL 转换为字符串以存储在数据库中

如列表 9.22 所示，我们也使用字段验证器将 URL 字段作为字符串返回。这是因为当我们尝试将其插入数据库时，我们会收到错误，因为 URL 数据类型与字符串不兼容。我们可以在负责插入记录的审计日志类中明确编码它，但这种方法可以很好地封装一切，其他开发者不必担心它，这有助于减少认知负荷。

能够写入记录并使这个单元测试通过的最后一步是生成一个实际要插入数据库的类。在以下列表中，我们创建了一个包含 `log_record` 和 `get_log_record` 方法的 `AuditLog` 类。对于其他数据库记录，我们特别使用了 `Sql` 作为名称的一部分。在这种情况下，我们并不一定想将其绑定到数据库。

##### 列表 9.23  `AuditLog` 类

```py
class AuditLog:
    @staticmethod
    def log_record(log_record: AuditLogRecord):
        with get_db_connection(row_factory=dict_row) as conn: #1
            log_record_dict = log_record.dict()  #1
            log_record_dict["url"] = str(log_record_dict["url"]) 
            result = conn.execute( #2
                """  #2
           INSERT INTO audit_log (user_id, ip_address,  #2
                       user_agent, http_request, http_response,  #2
  url, message)  #2
           VALUES (%(user_id)s,  #2
                %(ip_address)s,  #2
                %(user_agent)s,  #2
                %(http_request)s,  #2
                %(http_response)s,  #2
                %(url)s,  #2
                %(message)s #2
                )  #2
           RETURNING audit_log_id #2
            """,  #2
                log_record_dict,  #2
            )  #2
 #2
        return result.fetchone()["audit_log_id"] #3

    @staticmethod
    def get_log_record(audit_log_id: str):
        with get_db_connection(row_factory=
➥class_row(AuditLogRecord)) as conn:
            result = conn.execute(
                """
                SELECT *
                FROM audit_log
                WHERE audit_log_id = %(audit_log_id)s
                """,
                {"audit_log_id": audit_log_id},
            )

        return result.fetchone()
```

#1 定义了一个要使用的字典

#2 定义了要插入的字段；我们将返回从数据库中获得的 audit_log_id。

#3 因为使用了 dict_row 作为 row_factory，我们可以直接引用 audit_log_id。

在此基础上，我们的单元测试通过了，我们可以在任何想要的地方添加审计记录。现在，让我们看看如何将我们所学的内容与 FastAPI 框架集成。

### 9.3.3 审计逻辑

向我们的应用程序添加审计的最明显方式是通过调用 `log_record` 方法并传递一个 `AuditLogRecord`，正如我们在单元测试中所看到的。我们当然可以取我们的现有端点之一并添加调用，最终得到以下类似列表的内容，这允许我们写入消息，但并没有捕获我们为定义的字段所定义的一些其他可能有用的信息。

##### 列表 9.24  添加我们的 API 记录的一种方式

```py
async def read_files() -> list[AchFilesResponse]:
    AuditLog().log_record( #1
        AuditLogRecord(  #1
            message="User retrieved ACH files -test"  #1
        )  #1
    )  #1
    return AchFileSql().get_files_response()
```

#1 在 API 执行时手动调用 log_record 方法，但我们遗漏了许多有趣的字段

我们可以通过包括请求字段来增强前面的列表。然后我们可以从请求对象中提取额外的信息。

##### 列表 9.25  在我们的 API 中包含请求信息

```py
async def read_files(request: Request) -> #1
➥ list[AchFilesResponse]: 
    AuditLog().log_record(
        AuditLogRecord(
            ip_address=request.client.host, #2
            message="User retrieved ACH files -test"
        )
    )
    return AchFileSql().get_files_response()
```

#1 更新方法以包括请求对象

#2 我们现在可以访问一些有趣的字段

虽然这样可行，但它要求开发者添加比我们更愿意的更多样板逻辑。由于我们主要对审计 API 的使用感兴趣，我们可以采取另一种方法。

## 9.4 使用中间件进行日志记录

为了记录 API 请求，我们可以考虑添加一个具有唯一日志记录请求目的的中间件函数。我们可以使用请求和响应，这使我们能够捕获响应代码。虽然我们也可以考虑配置我们的 Nginx 日志以捕获生产环境中的信息，但了解如何在 FastAPI 中实现这一点也是有帮助的，因为它可以广泛应用于其他需求。我们使用`@app.middleware`注解来编写我们的日志记录。

##### 列表 9.26  日志记录的中间件层

```py
@app.middleware("http") #1
async def log_requests(request: Request, call_next): #2
    response = await call_next(request) #3
    log_message = getattr(request.state, 'log_message', #4
                  "Default audit log message") 
    log_record = AuditLogRecord( #5
        ip_address=request.client.host,  #5
        user_agent=request.headers.get #5
➥('user-agent', 'unknown'),  #5
        http_request=request.method,  #5
 http_response=response.status_code, #5
 url=str(request.url), #5
 message=log_message, #5
  )
    logger = AuditLog() #6
    logger.log_record(log_record) 
    return response #7
```

#1 定义了一个中间件组件，该组件为每个 HTTP 请求执行

#2 定义一个日志请求的方法

#3 将请求传递给下一层

#4 获取在请求.state 中定义的 log_message，如果不存在则默认

#5 使用我们期望的请求字段构建一个 AuditLogRecord

#6 编写日志消息

#7 返回响应

插入该中间件层并使其执行日志记录相对容易。然后我们可以记录所有传入的调用，这消除了样板代码的需求。此外，它使我们的代码更简洁，让开发者专注于他们的代码。唯一的缺点是添加消息需要更多的努力。当然，消息将反映`REST` `API`调用正在做什么。因此，从某种意义上说，记录消息是多余的（并且是空间的浪费）。例如，当日志显示他们在`/api/v1/files`上执行`GET`操作时，我们是否需要知道有人查看 ACH 文件？当然，我们可能需要出于业务原因这样做，因为当被审查时，对于不熟悉数据的人来说，解释消息更容易。与其浪费存储空间，我们可能决定有一个实用方法，可以在提取期间执行该解释并插入适当的消息。用例可能各不相同，我们只想说明一些替代方案。在这种情况下，我们确实想在数据库表中存储文本，我们将使用装饰器来完成这项工作。以下列表显示了如何使用我们定义的`log_message`注解在请求状态中包含消息。

##### 列表 9.27  `log_message`注解

```py
def log_message(message: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request: Request = kwargs.get("request", None)
            if request: #1
                request.state.log_message = message
            return await func(*args, **kwargs)

        return wrapper

    return decorator
```

#1 如果有请求，它将保存传递的消息。

最后，我们只需要对 API 进行一些小的更改。通过使用我们新的注解，我们可以自定义写入数据库的消息。以下列表显示了我们对 API 调用的最终更改。

##### 列表 9.28  日志的最终更改

```py
@log_message("Retrieving ACH files") #1
async def read_files(request: Request) -> #2
➥ list[AchFilesResponse]: 
    return AchFileSql().get_files_response()
```

#1 为我们的每个 API 调用添加一个独特的消息

#2 确保请求对象是方法的一部分

添加这部分逻辑就完成了我们的 API 消息记录。现在，任何我们用`@log_message`注解的 API 都会写入日志消息。这只是追踪和审计应用程序的开始。通过探索诸如 OpenTelemetry、OpenTracing 或 Prometheus 等项目，您可以极大地扩展在应用程序中收集数据的能力，仅举几个例子。

## 9.5 查看审计日志

现在我们有了应用程序的日志 API 请求，让我们添加在应用程序中查看日志的能力。这个页面将像第八章中的异常页面一样工作。我们需要检索审计记录并将它们放入 DataGrid 中以便查看。当然，我们可以根据需要扩展功能，但到目前为止，仅仅返回数据就是我们需要完成的任务。

### 9.5.1 创建页面

我们在`src/app/audit`下创建`page.tsx`页面。这个页面将负责向 API 发出请求，然后将返回的数据传递给一个可以显示记录的组件。以下列表显示了基本页面。

##### 列表 9.29  审计页面

```py
export default function AuditPage() {

    const [entries, setEntries] = useState<AuditResponse[]>([]);

  useEffect(() => {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? '';
        axios.get<AuditResponse[]>(`${apiUrl}/audits`, { #1
            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(response => {
                console.log(`Response data ${JSON.stringify(response.data)}`);
                setEntries(response.data);
            })
            .catch(error => {
                console.log(error);
            });
    }, []);

    return (
…
       <AuditRecords records={entries}/> #2
…
    );
}
```

#1 从数据库获取审计记录

#2 显示审计记录

为了完成这一页，我们还想创建`AuditResponse`接口，如列表 9.30 所示。我们保持名称与 API 中的名称一致，因为这样我们工作量更小。然而，在前几章中，我们已经展示了将字段映射到数据内部表示的过程，这一点在这里仍然有效。请记住，我们通过精确匹配 API 所承担的权衡——最大的问题是当 API 发生变化或当我们迁移到另一个供应商（字段名称可能发生变化）时。这些更改必须在整个应用程序中传播，而不是在单个点上。我们建议始终尝试使用一层抽象来防止这类问题。没有人愿意花费时间将`created_at`更改为`created_on`，并需要在可能触及的每个地方重新测试。

##### 列表 9.30  `AuditResponse`接口

```py
export interface AuditResponse {
    audit_log_id: string;
    created_at: Date;
    user_id: string;
    ip_address: string;
    user_agent: string;
    http_request: string;
    http_response: number;
    url: string;
    message: string;
}
```

### 9.5.2 创建页面组件

我们将数据传递给我们的 `AuditLogRecords` 组件，以便处理展示逻辑。在这个阶段，我们希望看到组件如何帮助我们将代码分解成更小、更易于管理的部分。我们试图让页面负责检索数据，而页面上的组件则负责展示这些数据。在应用程序最初开发时，我们往往会有更多重复的代码。这通常是因为开发者试图赶在截止日期前完成任务。随着代码的成熟（以及开发者对应用程序有更好的理解），我们通常可以开始创建更通用的代码。将我们的代码分解成组件有助于我们更快地达到这一点。下一个列表显示了展示审计记录的代码。

##### 列表 9.31  `AuditLogRecords` 组件

```py
interface AuditRecordsProps {
    records: AuditResponse[];
}

export default function AuditRecords({records}: Readonly<AuditRecordsProps>) {

    const [isOpen, setIsOpen] = useState<boolean>(false); #1
    const [auditDetails, setAuditDetails] = #2
                   useState<AuditResponse | null>(null); 
    const columns: GridColDef[] = [
        {field: 'view', headerName: 'View',
➥ sortable: false, width: 10, renderCell: (params) => (
                <IconButton
                    onClick={(e) => { #3
                        e.preventDefault();  #3
                        setAuditDetails(params.row);  #3
                        setIsOpen(true);  #3
                    }}
                    color="primary"
                >
                    <InfoIcon />
                </IconButton>
            )},
        {field: 'created_at', headerName: 'Date',
➥ width: 150, valueGetter: (params) => 
convertDateFormat(params.value)},
➥        {field: 'ip_address', 
➥headerName: 'IP Address', width: 150, valueGetter: (params) => 
➥ stripSubnet(params.value)},
        {field: 'message', headerName: 'Audit Message', width: 300},
    ];

    return (
        <ThemeProvider theme={defaultTheme}>
            <Box sx={{display: 'flex'}}>
                <CssBaseline/>
                    <Container maxWidth="lg" sx={{mt: 4, mb: 4, ml: 4}}>
                        <Paper
                            sx={{
                                p: 2,
                                display: 'flex',
                                flexDirection: 'column',
                                height: 480,
                                width: '100%'
                            }}
                        >
                            <DataGrid columns={columns} #4
➥ rows={records} getRowId={(row) => row.audit_log_id} /> 
                        </Paper>
                    </Container>
            </Box>
            <AuditModal open={isOpen} #5
➥] onClose={setIsOpen.bind({}, false)} auditData={auditDetails} /> 
        </ThemeProvider>
    );
}
```

#1 定义一个字段来跟踪模态

#2 定义模态的审计详情内容

#3 当按钮被点击时，将信息传递给模态并设置它现在是打开的

#4 使用 DataGrid 元素来显示文本

#5 包含 `AuditModal` 组件以显示详细内容

我们还构建了一个额外的实用方法作为组件的一部分——`stripSubnet` 方法。此方法的代码如下所示。注意它只移除尾部的 `/32`，那么为什么我们有移除它的函数，它究竟在哪里？

##### 列表 9.32  `stripSubnet` 实用方法

```py
export function stripSubnet(ipWithSubnet: string): string {
  const [ip, subnet] = ipWithSubnet.split('/'); #1

  if (subnet === '32' ) { #2
    return ip; #3
  } else {
    return ipWithSubnet; #4
  }

}
```

#1 在斜杠处分割 IP 地址

#2 如果子网为 32，则表示单个 IP 地址。

#3 仅当 IP 地址为单个 IP 时返回 IP

#4 否则，返回传入的字符串

`/32` 是无类域间路由（CIDR）表示法的一部分，它是一种定义 IP 地址范围的方式。`/32` 表示给定的地址是一个单独的地址，它只是明确表示这一点的一种方式。为了说明，我们可以用 `192.168.1.0/24` 来表示 `192.168.1.0` 和 `192.168.1.255` 之间的地址。这解释了 `/32` 是什么，但没有解释为什么它在那里。我们看到 `/32` 是因为我们选择在 Pydantic 模型中以 `IPvAnyNetwork` 的形式表示 IP 地址。

Pydantic 支持许多网络类型。目前我们主要关注的是`IPvAnyAddress`和`IPvAnyNetwork`。使用`INET`作为我们的`ip_address`列的决定是问题的催化剂。`INET`数据类型支持使用 CIDR 表示法存储 IP 地址。因此，我们选择在 Pydantic 中使用一个紧密模拟该功能的类型，尽管我们表中应该只有一个 IP 地址。我们使用这个例子来说明设计决策如何产生下游影响。列的`INET`要求可能是因为有查询 IP 地址范围的需求，即使审计记录不需要包含多个 IP 地址，使用`INET`数据类型也是有意义的。也许，这些决策（无论好坏）在 UI 组件到达我们手中之前就已经做出了。业务不希望用户看到 IP 地址中的`/32`部分，因为大多数用户不会理解这种表示法，所以我们需要纠正显示。

下面的列表展示了帮助说明 Pydantic 类型`IPvAnyAddress`和`IPvAnyNetwork`之间差异的一些额外的单元测试。

##### 列表 9.33  网络类型的单元测试

```py
class IpModel(BaseModel):
    ip_any_address: IPvAnyAddress
    ip_any_network: IPvAnyNetwork

class TestPydanticNetworkTypes:

    def test_valid_any_address(self):
        my_address = IpModel(ip_any_address="127.0.0.1",
➥ ip_any_network="127.0.0.1")
        assert str(my_address.ip_any_address) == "127.0.0.1"

    def test_invalid_any_address(self):
        with pytest.raises(ValueError):
            IpModel(ip_any_address="127.0.0.256",
➥ ip_any_network="127.0.0.1")

    def test_valid_any_network(self):
        my_address = IpModel(ip_any_address="127.0.0.1",
➥ ip_any_network="127.0.0.1")
        assert str(my_address.ip_any_network) == "127.0.0.1/32"

    def test_invalid_any_network(self):
        with pytest.raises(ValueError):
            IpModel(ip_any_address="127.0.0.1",
➥ ip_any_network="127.0.0.256")
```

我们还使用了一个自定义组件`AuditModal`。当点击审计记录的详细信息时，我们希望显示一个格式化的窗口，包含审计信息（列表 9.34）。这是对审计记录的更完整视图，因此我们不需要在初始视图中添加杂乱。`AuditModal`是一个相对简单的组件。因为我们已经返回了整个记录，这只是一个数据的展示，与上一章中我们在用户点击查看图标时特别检索 ACH 记录数据的情况不同。记住，区别在于我们不想在没有用户请求的情况下返回可能的 NPI 数据。现在我们有了查看审计信息的能力，监控用户请求 NPI 数据，并解决任何潜在的数据滥用（无论是有意还是无意）。

##### 列表 9.34  `AuditRecords`组件

```py
interface AuditRecordsProps {
    records: AuditResponse[];
}

export default function AuditRecords({records}:
➥ Readonly<AuditRecordsProps>) {

    const [isOpen, setIsOpen] = useState<boolean>(false);
    const [auditDetails, setAuditDetails] =
➥ useState<AuditResponse | null>(null);
    const columns: GridColDef[] = [
        {field: 'view', headerName: 'View', sortable: false,
➥ width: 10, renderCell: (params) => (
                <IconButton #1
                    onClick={(e) => { 
                        e.preventDefault(); 
                        setAuditDetails(params.row); 
                        setIsOpen(true); #2
                    }} 
                    color="primary" 
                > 
                    <InfoIcon />
                </IconButton>
            )},
        {field: 'created_at', headerName: 'Date', width: 150,
➥ valueGetter: (params) => convertDateFormat(params.value)},
➥        {field: 'ip_address', headerName: 'IP Address',
➥ width: 150, valueGetter: (params) => stripSubnet(params.value)},
        {field: 'message', headerName: 'Audit Message', width: 300},
    ];

    return (
        <ThemeProvider theme={defaultTheme}>
            <Box sx={{display: 'flex'}}>
                <CssBaseline/>
                    <Container maxWidth="lg" sx={{mt: 4, mb: 4, ml: 4}}>
                        <Paper
                            sx={{
                                p: 2,
                                display: 'flex',
                                flexDirection: 'column',
                                height: 480,
                                width: '100%'
                            }}
                        >
                            <DataGrid columns={columns}
➥ rows={records} getRowId={(row) => row.audit_log_id} /> #3
                        </Paper>
                    </Container>
            </Box>
            <AuditModal open={isOpen} onClose=
➥{setIsOpen.bind({}, false)} auditData={auditDetails} />  #4
        </ThemeProvider>
    );
}
```

#1 可点击的按钮，显示审计记录的详细信息

#2 可点击的按钮，显示审计记录的详细信息

#3 显示所有审计记录结果的 DataGrid 元素

#4 当点击图标时显示的 AuditModal 组件

在能够拉取和查看审计数据的能力之后，我们缺少拼图中的关键一块——让我们添加 API 端点来检索我们的数据。

### 9.5.3 添加审计日志的 API

虽然我们构建了检索记录的初始 SQL 调用和显示数据的 UI 层，但我们尚未构建 API 本身。我们现在需要确保我们可以调用 API 并返回数据。下面的列表显示，完成这项任务只需要几行代码，其中大部分都是帮助文档化 API 供他人使用。我们创建了名为`routers/audit.py`的单独文件来创建审计 API。因为这个端点与 ACH 文件无关，我们希望将其分离。

##### 列表 9.35  审计路由器

```py
@router.get(
    path="",
    response_model=list[AuditLogRecord],
    summary="Retrieve Audit Information",
    description="Retrieve audit records for the ACH API.",
    response_description="The overview of the requested audit records.",
    tags=["Audit"],
)
@log_message("Audit Information")
async def read_audit_information(request: Request) -> list[AuditLogRecord]:
    return AuditLog().get_all_log_records() #1
```

#1 简单调用以获取和返回所有审计记录

由于我们为了便于维护而分离了代码，API 将无法访问，直到我们使用`include_router`方法添加它。我们最初是为`router/files.py`文件这样做，直到这一点，我们所有的 API 都放入了那个文件，这意味着现在没有必要重新访问`main.py`文件。下一个列表显示了访问我们的 API 所需的附加行。

##### 列表 9.36  添加路由器

```py
app = FastAPI()

app.include_router(files.router)
app.include_router(audit.router) #1

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins that should be allowed
    allow_credentials=False,  # Allows cookies to be included
    allow_methods=["*"],  # List of HTTP methods allowed for CORS
    allow_headers=["*"],  # List of HTTP headers allowed for CORS
)
```

#1 处理审计 API 请求的附加路由器

那就是最后要放置的部分。现在我们应该拥有完整的审计功能。我们可以通过额外的日志记录和更具体的业务场景日志记录来扩展这一点，例如无效输入和其他我们可能希望记录的不仅仅是 API 访问确认的场景。

## 摘要

+   每一次连续的迭代都有助于提升我们应用程序的功能。我们致力于解决最终用户提出的要求/关注点、业务所需的新功能以及技术债务。

+   能够进行搜索是我们应用程序的一个必要功能。交易在一天中的不同时间通过不同的文件加载。如果客户对交易何时加载有疑问，我们需要能够快速提供答案。

+   无论我们是致力于保护我们的应用程序免受入侵者或不满的员工侵害，还是为了更深入地了解应用程序，这一切都始于适当的审计，以保障安全和合规，并伴随一般分析。

+   我们谈到了不仅记录日志的重要性，而且拥有一个成功的日志管理策略将在我们的成功中扮演一部分。我们不应相信应用程序正在运行或未被攻击——相反，我们必须有数据并对其进行分析。

+   您已经了解到，根据用户反馈扩展异常处理和增强 UI/UX 的重要性，以提升整体用户满意度和系统功能。

+   在有效解决客户投诉并确保更好的用户体验中，通过多个标准启用交易搜索的重要性得到了强调。

+   利益相关者对审计功能的需求强调了跟踪仪表板交互以了解用户参与度和规划未来系统改进的价值。

+   我们关注了敏捷冲刺计划的重要性，包括定义清晰的用户故事和验收标准，以确保对齐和成功交付冲刺目标。

+   Gantt 图在可视化项目时间线方面非常有效，有助于利益相关者的沟通和期望管理。

+   在大型数据集中对强大搜索功能的需求强调了高效搜索 API 的必要性，这促使我们探索用户界面更改和实时搜索方法。

+   开发 BDD 测试说明了测试在验证新搜索功能中的关键作用，确保它们满足指定的场景和用户需求。

+   实施搜索 API 涉及理解各种处理标准的方法，突出了灵活且用户友好的搜索功能的重要性。

+   对日志和审计的探索强调了在监控 API 使用和用户交互中实施安全实践的重要性，以确保安全和问责制的系统行为。

+   设计审计日志数据库强化了捕获详细请求信息（如用户代理和 IP 地址）的需求，以实现全面的安全和分析。

+   在 FastAPI 中使用中间件演示了如何简化日志过程，这减少了冗余代码并增强了代码的可维护性。

+   将审计日志视图集成到应用程序中，展示了透明度和监控系统活动以支持持续改进和问责制的价值。
