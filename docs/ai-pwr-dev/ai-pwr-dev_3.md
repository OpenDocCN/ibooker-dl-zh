# 第三章：使用 ChatGPT 设计软件

### 本章涵盖

+   使用 ChatGPT 进行潜在设计原型

+   在 Mermaid 中记录我们的架构

+   完成我们的设计与 ChatGPT

现在我们已经对何时使用生成式人工智能有了直觉，我们将开始设计、探索和记录我们应用程序的架构。预先布置一些关键组件在几个方面都是有益的。例如，它使我们能够将一些设计工作委派给子架构师或将一些开发工作交给其他团队成员。提前设计还将有助于我们澄清我们对实施的思考，使我们能够预见并避免一些陷阱。最后，将设计记录为文档使我们能够证明我们的重要设计决策，向我们未来的自己、利益相关者以及可能继承该项目的人传达我们的意图。

首先，让我们询问 ChatGPT 关于如何最好地设计这个项目，并看看它能提出什么解决方案。

## 3.1 请求 ChatGPT 协助我们进行系统设计

在一个新的会话中，我们将从一个提示开始，概述我们的需求。鉴于我们在上一章节大部分时间都在思考我们的需求，我们应该能够构建一个复杂的提示，并且应该对所需内容有很好的了解。或者，我们可以请求 ChatGPT 提供这样一个系统的需求。然后，我们可以将这些需求纳入我们的提示中，并根据需要进行编辑。

毫无疑问，你已经遇到了无数篇声称展示正确的提示工程方法的文章。提示工程，或者是设计和优化初始输入或“提示”以获得所需输出的做法，是我们与大型语言模型互动的重要组成部分。我们在本书中所做的大部分工作都将被视为提示链接、生成知识提示和零-shot 推理。这些主要是学术上的说法，即我们将与生成式人工智能工具进行对话。重要的要点是，与任何对话一样，当你需要一个具体的答案时，你要问一个具体的问题。或者你要求生成式人工智能逐步思考问题，并尽可能具体。因此，我们将向 ChatGPT 提供一个非常具体的提示，以正式化需求收集过程。

##### 列表 3.1 包含我们系统需求的提示

```py
Please take the following criteria for an information technology asset management system and create a Python project that satisfies them: 

Asset Inventory: The system should be able to maintain a complete inventory of all hardware and software assets owned by an organization. 

Asset Tracking: The system should be able to track asset location, status, and usage. This includes information such as who uses the asset, when it was last serviced, and when it is due for replacement. 

Asset Lifecycle Management: The system should be able to manage the entire lifecycle of an asset, from procurement to disposal. This includes tracking warranty and lease information, managing asset maintenance and repair, and ensuring compliance with regulations. 

Asset Allocation: The system should be able to allocate assets to employees or departments, track usage, and manage reservations for assets that are in high demand. 

Asset Reporting: The system should be able to generate reports on asset utilization, availability, and maintenance history. Reports should be customizable and easy to understand. 

Integration: The system should be able to integrate with other IT systems, such as procurement, help desk, and financial systems, to ensure consistency and accuracy of data. Security: The system should have robust security features to protect sensitive information about assets, including user authentication, role-based access control, and data encryption. 

Scalability: The system should be scalable to accommodate changes in the organization's IT infrastructure, including new hardware and software assets, as well as changes in the number of users and departments. 

User-Friendly Interface: The system should have a user-friendly interface that is easy to navigate, search, and use. It should also be customizable to meet the needs of individual users and departments. 

Accessibility: The system should be accessible from different devices and locations, including mobile devices and remote areas. It should also be accessible to users with disabilities.
```

经过一番思考，ChatGPT 将返回开发此项目所需的高层步骤。哦，我们在开玩笑，它几乎立即就能生成这些步骤。这些建议很有帮助，但我们想要解决空白文件的问题：我们想要开始设计。我们需要散文和图表来帮助我们理解组件级别的系统。我们需要任务而不是配方。我们应该要求 ChatGPT 展示它的工作过程。

##### 列表 3.2 要求 ChatGPT 展示其工作的提示

```py
Please show the package structure
```

在这个提示下，ChatGPT 将生成并返回一个包图（这个列表略有删节），演示它不仅可以设计软件，还可以帮助我们布置物理文件。我们可能对整体的包结构有异议（或者也可能没有），但这将为我们提供足够的灵感来开始。

##### 列表 3.3 ChatGPT 创建的资产管理系统的包结构

```py
asset_management_system/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── asset.py
│   │   ├── user.py
│   │   └── ...
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── login.py
│   │   ├── register.py
│   │   └── ...
│   ├── models/
│   │   ├── __init__.py
│   │   ├── asset.py
│   │   ├── user.py
│   │   └── ...
│   ├── views/
│   │   ├── __init__.py
│   │   ├── asset.py
│   │   ├── user.py
│   │   └── ...
├── requirements.txt
└── run.py
```

我们可以审查宏观情况，也可以要求 ChatGPT 提供微观情况。我们可以创建提示，让 ChatGPT 按顺序显示每个文件的内容。我们可以将内容认真地复制到一个与 ChatGPT 建议的文件名相同的文件中。然而，我们始终要仔细检查 ChatGPT 的工作，不过，我们不应假设它是 100%安全或准确的。我们偶尔会发现 ChatGPT 生成的代码无法工作。与我们不同，ChatGPT 的重点是生成下一个标记，而不是思考系统。话虽如此，我们应该快速查看它生成的代码，以评估质量并确定其有用性。

##### 列表 3.4 要求 ChatGPT 展示其（详细的）工作的提示

```py
please show the source code for asset_management_system/models/asset.py
```

ChatGPT 将返回此类的源代码。它使用 Flask 和 SQLAlchemy，因此域模型有点混乱。下次，我们应该要求它使用我们首选的应用服务器，并保持域模型的清晰。这些对我们来说也是额外的数据点。现在我们知道了我们不想要的东西。

##### 列表 3.5 资产管理系统中 asset_management_system/models/asset.py 的示例源代码：

```py
from datetime import datetime
from asset_management_system import db

class Asset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    description = db.Column(db.String(255))
    category = db.Column(db.String(80))
    purchase_date = db.Column(db.DateTime, default=datetime.utcnow)
    purchase_cost = db.Column(db.Float)
    warranty_expiry = db.Column(db.Date)
    lease_expiry = db.Column(db.Date)
    status = db.Column(db.String(80), default="Available")
    location = db.Column(db.String(80), default="IT Department")
    assigned_to = db.Column(db.String(80))
    assigned_date = db.Column(db.DateTime)
    retired_date = db.Column(db.DateTime)
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return '<Asset %r>' % self.name
```

现在我们对 ChatGPT 如何解决问题有了一定了解，让我们听听第二种意见。为此，我们将探索现有的项目和库。我们还将使用 ChatGPT 进行此项调查。

## 3.2 审视全局情况

如果我们以前从未使用过或见过信息技术资产管理（ITAM）系统，我们会想了解它是什么，它应该做什么，以及如何实现这样一个系统。正如我们在上一章中所经历的，ChatGPT 非常适合这项任务。当问题空间和已知条件明确时，GitHub Copilot 和 CodeWhisperer 是出色的工具，当你准备进行高效编码时，这些工具非常有用。另一方面，ChatGPT 可以用来探索这个问题空间，进行交互式原型设计，并设计和记录您的解决方案。探索问题空间的一个很好的第一步是检查当前的开源项目。我们可以使用这些“开箱即用”的项目或利用它们的设计来激发我们的灵感。使用 ChatGPT，我们可以快速浏览开源世界，并聚焦于可能与我们试图构建的内容类似的项目。我们可以从列表 3.1 中开始。

##### 列表 3.6 查找 ITAM 项目的提示

```py
What are some examples of ITAM projects or products?
```

ChatGPT 回复了几个示例，总结了它们的核心特点。然而，由于我们正在寻找开发项目的灵感，我们应该开始将此列表精简到仅包含代码可用的项目；也就是说，哪些是开源项目？

##### 图 3.1 ChatGPT 展示了 ITAM 特性和几个拥有这些特性的产品和项目的属性列表。

![文本，字母描述自动生成](img/03image002.png)

接下来，我们将开始精简过程。鉴于进行这种分析和探索当前 ITAM 系统的目的是我们可以看到软件中需要哪些特性以及我们如何实现这些特性，我们只想返回源代码可用的项目。因此，让我们只获取开源项目的列表。

##### [寻找开源 ITAM 项目的提示](https://wiki.example.org/finding_open_source_itam_projects)

```py
Are any of these products open-source?
```

ChatGPT 最令人兴奋和有用的功能之一是它可以延续上下文；它理解在这个上下文中，“任何”意味着前面提到的任何项目。ChatGPT 回复了一个开源项目列表。

##### 图 3.2 ChatGPT 返回了一个开源 ITAM 项目的列表。

![文本，字母描述自动生成](img/03image003.png)

我们将继续精简过程。我们打算用 Python 来实现我们的系统，因此我们只对具有 Python 作为与系统交互手段的项目感兴趣。

##### [寻找用 Python 编写的 ITAM 项目的提示](https://wiki.example.org/finding_python_itam_projects)

```py
Are any of these written in Python?
```

根据我们的提示，ChatGPT 将会将这个列表精简到只有用 Python 编写的项目。它返回了五个项目。我们将评估每个项目，并确定我们应该进一步探索哪一个。

##### 图 3.3 ChatGPT 告诉我们有四个具有 Python 组件的开源项目。

![文本，字母描述自动生成](img/03image004.png)

在这个列表中，Ralph 似乎是最有前途的。Snipe-IT 和 Open-AudIT 是用 PHP 编写的，具有我们可以与之交互的基于 Python 的 API；然而，我们希望一个用 Python 编写的项目。NetBox 不是 ITAM 而是 IP 地址管理（IPAM）系统。最后，CMDBuild 是用 Java 编写的。我们希望用 Python 构建一个 IT 资产管理系统。因此，Ralph 似乎是最符合我们用例的选择。

##### 在现实世界中

如果我们必须在现实世界中实施 ITAM 系统，我们将评估每个产品（商业和开源）在不同用例下的适用程度。例如，如果 Ralph 能够满足大多数或所有这些用例，我们将简单地搭建一个新实例；或者在商业产品的情况下，我们将获取许可证。我们将执行这个分析，因为我们得到的是为企业创造价值的报酬，而不是编写代码。如果我们可以拿来即用，那么我们很快就能创造价值，可能比我们自己开发它还要快。

现在我们已经发现了一个与我们要构建的项目相似的项目（一个子集），我们可以开始探索它，审查它的设计，并检查它的源代码。我们将从这个源代码和设计中汲取灵感，大胆借鉴，并在必要时进行更改。让我们首先来了解系统的概述。

## 3.3 探索一个库

首先，我们将要求 ChatGPT 描述 Ralph 的整体结构。这个结构可能会在我们开发架构时给我们启发。或者，它可能会提醒我们如何不要组织我们的代码。

##### 列表 3.9 查找开源项目 Ralph 结构的提示

```py
Please describe the modules of Ralph. 
```

ChatGPT 将返回 Ralph 的高级模块列表。值得注意的一点是，在撰写本文时，ChatGPT 将以 2021 年的 Ralph 设计返回。ChatGPT 的训练数据截止日期（根据 ChatGPT 本身）是 2021 年 9 月。因此，2021 年 9 月后对设计的任何后续更改都不会在 ChatGPT 中显示。未来，当 GPT 有了互联网访问和扩展功能时，所有这些都将发生改变。但是，目前，我们必须接受这一限制。

##### 图 3.4 ChatGPT 解释了 Ralph 开源项目的高级模块布局

![自动生成的文本，文字描述](img/03image005.png)

资产和库存模块似乎是一个很好的开始地方。接下来，我们将要求 ChatGPT 分别呈现每个模块的美人鱼图。让我们首先从资产开始。

##### 列表 3.10 在 Ralph 的资产模块中查找类的提示

```py
Please show me a class diagram in Mermaid for all classes in the Asset module. 
```

##### 美人鱼

美人鱼是由 Knut Sveidqvist 发明的现代、开源、简化的图表定义语言。它于 2015 年发布。它允许用户创建各种图表，包括流程图、序列、类、甘特图等。由于它是基于文本的，因此很容易复制粘贴到设计工具中。它的语法非常易于阅读和理解，几乎不需要正式培训（与 UML 等不同）。

ChatGPT 将返回一个美人鱼文档，概述类、它们的属性和它们的关系。首先，让我们来看一下这个美人鱼文档中的类定义。列表 3.11 的第一行指示美人鱼我们正在定义一个类图类型，而不是它支持的其他类型。第二行表示类（在本例中为资产）。接下来的行列举了类的字段。减号表示它是一个私有属性。倒数第二行是公共方法的定义。我们知道它是公共的；加号表示一个公共方法。

##### 列表 3.11 资产类定义

```py
classDiagram
    class Asset {
        - name: string
        - status: string
        - model: Model
        - category: Category
        - manufacturer: Manufacturer
        - sn: string
        - barcode: string
        - price: decimal
        - purchase_date: date
        - order_no: string
        - delivery_date: date
        - invoice_date: date
        - invoice_no: string
        - provider: Provider
        - budget_info: BudgetInfo
        - warehouse: Warehouse
        - owner: User
        - location: Location
        - asset_info: AssetInfo[]
        - attachments: Attachment[]
        + get_name(): string
    }
```

接下来，我们将看看如何在美人鱼中定义关系。箭头的双短线表示单向关联。在以下示例中，一个资产有一个型号。此外，资产有一个类别。

##### 列表 3.12 在美人鱼文档中的关系定义

```py
    Asset --> Model
    Asset --> Category
```

使用完整的 Mermaid 图，我们可以使用建模工具导入和可视化类图。以这种方式可视化将更容易理解。毕竟，众所周知，一幅(图表)胜过千言万语。您应该能够将文本插入 `mermaid.live` 生成并可视化图表。

##### 图 3.5 使用 Mermaid 类图可视化的资产类的可视化。

![](img/03image006.png)

Mermaid 在表现力和简洁性之间取得了适当的平衡，使其成为一种理想的建模语言，无论您是否时间紧迫。但你的实际情况可能有所不同。

接下来，让我们把注意力转向库存模块。我们可以要求 ChatGPT 生成一个类似于先前请求的资产图的文档。我们将跳到此文档的可视化部分。

##### 图 3.6 使用 Mermaid 类图可视化的库存包的可视化。

![Diagram Description automatically generated](img/03image007.png)

库存模块的可视化澄清了库存模块在 Ralph 项目中的重要性，但对我们构建以硬件为重点的 ITAM 来说是多余的。我们感兴趣的是追踪我们的全部资产，整个服务器；我们不一定对每张视频卡或每个内存模块进行追踪：只对整个服务器感兴趣。因此，我们将这个模块放在一边。

接下来，我们将深入研究资产类，因为这似乎是资产模块的根源。让我们让 ChatGPT 展示这个类。

##### 列表 3.13 提示，让 ChatGPT 显示资产类的源代码的用法

```py
  Please show me the source code for the Asset class.
```

ChatGPT 将返回资产类的源代码。为简洁起见，我们不会显示导入语句。此外，我们只检查此代码块中的几行。很明显，这个类有很多特定于 Django 的代码，例如，model 属性从数据库中查找模型对象。category 也是如此。

##### 让 ChatGPT 继续

有时，ChatGPT 会在中途或中途停止输出。这是 ChatGPT 设计中内置的输出限制所致。看起来你无法通过告诉 ChatGPT 忽略这个限制（对于某些系统约束，你可以这样做）来克服这个限制。但是，你可以告诉它“继续”或“继续”。它会恢复输出并从中断的地方继续输出。如果它在代码块内停止，则应指定它应该在哪里恢复输出。

##### 列表 3.14 简化版的资产类源代码

```py
class Asset(TransitionModelMixin, NamedMixin, AdminAbsoluteUrlMixin, models.Model):
    """
    A hardware or software asset
    """
    name = models.CharField(max_length=255, unique=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_NEW, max_length=50)
    model = models.ForeignKey(
        'assets.AssetModel',
        verbose_name=_('model'),
        related_name='assets',
        on_delete=models.PROTECT,
        null=True,
        blank=True,
    )
    category = models.ForeignKey(
        'assets.Category',
        verbose_name=_('category'),
        related_name='assets',
        on_delete=models.PROTECT,
    )
```

我们不想在项目中使用 Django。我们希望我们的应用程序以 API 为先，而不需要 Django 提供的额外功能。此外，我们更希望一个干净的领域模型。Django 非常有主见，建议您修改领域模型类以继承自 Django 基类。基类定义了领域模型类与数据库进行持久化交互的方式。我们更愿意自己处理持久化。因此，让我们请求 ChatGPT 彻底删除项目中的 Django。

##### 列表 3.15 提示 ChatGPT 删除 Django 并返回 Asset 类代码。

```py
  Take the open-source Ralph project, remove Django, and show me the source code for the Asset class.
```

ChatGPT 生成的输出将更加简洁。下面是缩写后用于与 Django 定义进行比较的结果。以下列表仅显示构造函数，其他类细节现在不相关。

##### 列表 3.16 没有 Django 特定功能的 Asset 类源代码

```py
class Asset:
    def __init__(self, name, status, model, category, manufacturer, sn=None, barcode=None, price=None, purchase_date=None, order_no=None, delivery_date=None, invoice_date=None, invoice_no=None, provider=None, budget_info=None, warehouse=None, owner=None, location=None, asset_info=None, attachments=None):
        self.name = name
        self.status = status
        self.model = model
        self.category = category
        self.manufacturer = manufacturer
        self.sn = sn
        self.barcode = barcode
        self.price = price
        self.purchase_date = purchase_date
        self.order_no = order_no
        self.delivery_date = delivery_date
        self.invoice_date = invoice_date
        self.invoice_no = invoice_no
        self.provider = provider
        self.budget_info = budget_info
        self.warehouse = warehouse
        self.owner = owner
        self.location = location
        self.asset_info = asset_info or []
        self.attachments = attachments or []

```

如果我们要求 ChatGPT 重新创建 Mermaid 类图，我们不会注意到任何变化。我们不会看到任何变化，因为 Django 特定的功能已封装在类中。

##### 图 3.7 Asset 类的更新后 Mermaid 类图。该类与之前的版本没有变化

![](img/03image008.png)

## 3.4 文档化你的架构

在上一节中，我们已经探索了 Ralph 开源项目并理解了项目如何结合在一起，现在我们可以开始设计了。我们将与 ChatGPT 迭代地一起工作，以帮助我们进行设计和文档编写。让我们从一个全新的聊天窗口开始。这个新的聊天会话将确保上下文清晰，我们之前的提示不会影响我们的新设计。

首先，我们将请求 ChatGPT 设计初始应用程序设计。我们将使用以下提示来做到这一点。

##### 列表 3.17 用于 ChatGPT 设计我们的初始应用骨架的提示

```py
I would like to build an ITAM project, written in Python. It will focus on the tracking and management of Hardware. It should expose REST APIs, using FastAPI, and persist data using SQLAlchemy. It should use hexagonal architecture. Please show me the Mermaid class diagram for this project.
```

##### 六边形架构

六边形架构，也称为端口和适配器模式，是一种旨在在应用程序的核心逻辑与其与外部系统（如数据库、用户界面和第三方服务）的交互之间创建明确分离的架构模式。这种关注点的分离有助于实现更易于维护、灵活和可测试的应用程序。

六边形架构的主要组件有：

**领域模型**：表示应用程序的核心业务逻辑，包括实体、值对象和领域服务。领域模型与任何外部系统或技术隔离，并且不应依赖于任何特定的实现细节，如数据库或 API。

**端口**：端口是定义领域模型和外部系统之间契约的接口。它们代表了应用程序的输入和输出边界。端口分为两种类型：主要端口和次要端口。

**主（被动）端口**：这些定义了应用程序向外部系统公开的用例。它们代表外部系统可以用来与应用程序交互的 API。

**次（驱动）端口**：这些定义了应用程序对外部系统（如数据存储库或外部服务）的预期合同。

**适配器**：适配器是端口的实现。它们处理领域模型与外部系统之间的通信，将外部系统使用的数据和协议转换为领域模型能够理解的格式。适配器有两种类型：主适配器和次适配器。

**主（被动）适配器**：这些实现了主端口，并负责接收来自外部系统的输入并将其传递给领域模型。

**次（驱动）适配器**：这些实现了次端口，并负责代表领域模型与外部系统进行交互。

在六边形架构中，领域模型位于中心（“六边形”），周围是端口和适配器。其关键思想是领域模型与外部系统之间的任何交互必须通过端口和适配器。这种架构允许轻松切换外部系统，通过使用端口的模拟实现简化测试，并清晰地分离关注点，促进更易维护的应用程序。

作为回应，ChatGPT 创建了这个奇特的类结构。首先，有一个 `MainApp` 类，它将（可能）运行应用程序。该类使用 `FastAPIAdapter`，它接受和处理 REST 调用。`FastAPIAdapter` 将调用 `HardwareService` 的方法。

##### 图 3.8 ChatGPT 向我们建议的 ITAM 系统的直接设计

![](img/03image009.png)

我们可以尝试询问 ChatGPT 为什么向我们返回了这个设计。让 ChatGPT 引用实际来源是一项挑战，因为 ChatGPT 尝试只使用一般术语进行讲述。然而，让我们尝试看看是否可以通过以下提示使其具体化。

##### 列表 3.18 询问 ChatGPT 为其设计提供灵感的提示

```py
Please provide the sources of this specific design.
```

ChatGPT 对此提示的回应非常一般化。它引用了最佳实践和六边形架构的起源。然而，它没有告诉我们关于设计选择的具体信息。

##### 图 3.9 ChatGPT 对所提议设计的理由。由于我们的提示说明我们想使用六边形架构，ChatGPT 就会从 Cockburn 的原始设计中提取这些元素。此外，因为我们的提示指定了要使用 FastAPI 和 SQLAlchemy，ChatGPT 将其包含在设计中。如果我们没有那么具体，ChatGPT 几乎肯定会建议其他框架。

![文本，信件描述自动生成](img/03image010.png)

接下来，我们将询问该项目的包结构。我们使用以下提示来做到这一点。

##### 列表 3.19 提示询问 ChatGPT 包结构

```py
what does the package structure look like for this project?
```

ChatGPT 将打印出类似以下的包结构。

##### 列表 3.20 ChatGPT 建议的项目结构

```py
itam_project/
|-- app/
|   |-- __init__.py
|   |-- main.py
|   |-- adapters/
|   |   |-- __init__.py
|   |   |-- fastapi_adapter.py
|   |   |-- sqlalchemy_adapter.py
|   |-- domain/
|   |   |-- __init__.py
|   |   |-- hardware.py
|   |   |-- hardware_service.py
|   |-- infrastructure/
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- database.py
|-- tests/
|   |-- __init__.py
|   |-- test_fastapi_adapter.py
|   |-- test_hardware_service.py
|   |-- test_sqlalchemy_adapter.py
|-- requirements.txt
|-- .gitignore
|-- README.md
```

我们可以要求 ChatGPT 显示每个文件中的代码。然而，在我们这样做之前，我们想要完成我们的设计。虽然这样做在最宽松的意义上满足了我们的要求，但是很难扩展。所以，我们将开始与 ChatGPT 合作，迭代设计，完善它；直到我们确信我们可以轻松修改我们的设计来处理未来的用例：比如支持跟踪软件许可等。虽然我们可以（而且应该！）偶尔请 ChatGPT 建议更好的做事方式，但在这种情况下，我们将告诉它添加一个名为 Asset 的父类到 Hardware 类。

##### 列表 3.21 提示 ChatGPT 为 Hardware 添加一个父类

```py
Add a class called Asset to this design. It is the parent class of Hardware. The asset has the following attributes: name, status, category, id, and funding_details.
```

引入`Asset`基类允许我们设置跨组织资产共享的属性。不难理解为什么 Ralph 使用了这个类。也很明显为什么我们花了那么多时间来看 Ralph 的设计。Ralph 的设计将影响我们的设计。为什么不呢？设计（几乎）是 SOLID 的。

更新后的类模型如下。

##### 图 3.10 更新的类图，定义了 Asset 到 Hardware 的关系。

![](img/03image011.png)

资产类将更容易扩展我们的模型，比如我们想添加软件或者一个 Pitchfork 类。例如，我们期望这些新的子类在公司拥有的资产的角度上行为与继承自资产的其他类完全相反。

##### SOLID 设计

SOLID 原则是五个旨在使软件设计更灵活和可维护的软件开发设计原则。

SOLID 的首字母缩写代表：

·   S：单一职责原则（SRP）

·   O：开闭原则（OCP）

·   L：里氏替换原则（LSP）

·   I：接口隔离原则（ISP）

·   D：依赖反转原则（DIP）

这里是这些原则的简要概述：

·   单一职责原则（SRP）：这一原则规定，一个类应该只有一个改变的原因；一个类应该只有一个职责，并且应该做得很好。

·   开闭原则（OCP）：这一原则规定，软件实体（类、模块、函数等）应该对扩展开放，但对修改关闭。

·   里氏替换原则（LSP）：这一原则规定，超类的对象应该可以替换为子类的对象，而不影响程序的正确性。对超类的使用也应该适用于其子类。

·   接口隔离原则（ISP）：这一原则规定，客户端不应该被强制依赖它不使用的方法。最好有小接口而不是大接口。

·   依赖倒置原则（DIP）：该原则指出高层模块不应依赖于低层模块。你应该按照接口编程，而不是实现。

接下来，我们将更新 Asset 类的`funding_details`属性，使其成为自己的类，而不仅仅是一个字符串。字符串不对可以分配为资金细节施加任何限制。在这些条目之间保持一致性使我们能够对这些字段执行统一的计算和聚合。

##### 列表 3.22 提示 ChatGPT 添加一个 FundingDetails 类

```py
Change the funding_details attribute in the Asset class from a string to a class. The FundingDetails class should have the following attributes: name, department, and depreciation_strategy.
```

ChatGPT 将输出一个新的 Mermaid 文档，添加新的类并记录新的关系。

##### 图 3.11 带有新类`FundingDetails`的更新类图。

![](img/03image012.png)

接下来，我们将更新`FundingDetails`类，将折旧计算委托给折旧策略。我们这样做是因为有几种计算资产折旧的方法。

##### 折旧

折旧是一个用来描述资产随着时间而减值的术语，其原因有很多。人们可以将多种标准的折旧方法应用于资产的价值。例如直线法、递减余额法和双倍递减余额法。

我们将创建一个提示，让 ChatGPT 将折旧概念引入到我们的对象模型中。

##### 列表 3.23 提示 ChatGPT 添加一个废弃策略

```py
Create an interface called DepreciationStrategy. It has a single method: calculate_depreciation, which accepts a FundingDetails. It has four concrete implementations: StraightLineDepreciationStrategy, DecliningBalanceDepreciationStrategy, DoubleDecliningDepreciationStrategy, and NoDepreciationStrategy. Update the Asset class to take a DepreciationStrategy.
```

通过将我们的 Asset 类的折旧计算委托给`DepreciationStrategy`，我们可以轻松地替换折旧方法。结果的 Mermaid 图表显示我们已经将依赖倒置原则引入到我们的设计中。

##### 图 3.12 我们已经在我们的对象模型中添加了一个折旧策略。这个引入使我们能够通过不同的方法计算我们资产的折旧。

![图形用户界面，应用程序描述自动生成](img/03image013.png)

一个常见的做法是企业拥有多个业务线，这在我们的类图中以部门表示。假设我们想为我们的资产支持多个业务线。我们将要求 ChatGPT 将其添加到我们的模型中。

##### 列表 3.24 提示我们的模型支持多个业务线

```py
The FundingDetails class should support more than one line of business (currently modeled as a department). Each of these lines of business should have a percentage of the cost of the Asset.
```

ChatGPT 建议在`FundingDetails`类中添加一个字典来支持此功能。ChatGPT 添加了一个名为`lines_of_business`的新属性到`FundingDetails`并打印了一个新的 Mermaid 图表。

我们可以预见到每一条业务线都想知道他们所拥有的公司所有资产成本的份额。我们相信我们可以使用访问者设计模式来实现这一点。

##### 访问者模式

访问者模式是一种行为设计模式，允许您在不更改访问者所操作的类的情况下定义对对象的新操作。当您需要对对象执行不同操作，但又想保持对象和操作分离时，访问者模式非常有用。此外，此模式使得很容易添加新行为而无需修改现有代码。

要实现访问者模式，您需要将以下组件添加到设计中：

**元素**：表示对象结构中元素的接口或抽象类。它声明了一个接受访问者对象作为参数的方法**accept**。

**具体元素**：实现元素接口或扩展元素抽象类的类。这些类表示对象结构中的不同类型的对象。

访问者：定义每个具体元素类的**visit**方法的接口或抽象类。访问方法代表要在具体元素上执行的操作。

**具体访问者**：实现访问者接口或扩展访问者抽象类的类。这些类为每个具体元素类实现了**visit**方法，为每个元素定义了算法。

要应用访问者模式，请按照以下步骤操作：

创建具有将访问者对象作为参数的**accept**方法的元素接口（或抽象类）。

通过扩展元素接口（或抽象类）并实现**accept**方法来实现具体元素类。

创建每个具体元素类访问方法的访问者接口（或抽象类）。

通过扩展访问者接口（或抽象类）并实现**visit**方法来实现具体访问者类。

要使用访问者模式，请创建具体访问者的实例，并将其传递给对象结构中具体元素的**accept**方法。然后，**accept**方法调用具体访问者的相应**visit**方法，执行具体访问者为该特定具体元素定义的算法。

让我们看看是否可以让 ChatGPT 对访问者模式在这种情况下的适用性发表意见。

##### 列表 3.25 向 ChatGPT 提出有关访问者模式的问题

```py
Additionally, I need a way to calculate the cost of all Asset that a given line of business. Would you recommend the Visitor pattern?
```

ChatGPT 认为这是一个适合计算给定业务线所有资产总成本的解决方案。此外，它建议我们创建一个名为 Visitor 的接口，其中包含一个名为 visit 的方法，该方法可用于计算特定业务线的总成本。根据 ChatGPT 的说法，我们应该修改 Asset 类以添加一个接受访问者的方法。最后，它建议我们为 "访问" 我们的每个资产创建一个具体访问者，名为 `CostByLineOfBusinessVisitor`。

每个业务线可能都想知道他们所有资产的总折旧。同样，我们可以向 ChatGPT 寻求设计建议。

##### 第 3.26 节 根据 ChatGPT 聚合总折旧金额

```py
I also need a way to calculate the total depreciation of all asset for a given business line.
```

ChatGPT 将回应，建议我们扩展具体 Visitor `CostByLineOfBusinessVisitor` 的行为。我们将在 `CostByLineOfBusinessVisitor` 中添加一个名为 `total_depreciation` 的新属性，该属性在每次“访问”期间将得到更新。然后，在访问完所有资产后，我们可以返回此值。

最后，让我们请 ChatGPT 完善我们的设计。我们知道你只实现了类似 Ralph 项目提供的功能子集。我们可以检查还缺少什么，我们需要完成这个项目。

##### 注意

与其全部将设计决策推迟给 ChatGPT，你应该始终运用自己的判断。毕竟，交付和维护此代码将由你负责。

##### 第 3.27 节 我错过了什么？

```py
What other features do I need in my ITAM to support hardware?
```

ChatGPT 返回了一个相当长的缺失功能列表。这个列表的长度并不令人惊讶。

##### 图 3.13 ChatGPT 建议我们如何通过列出所有缺失的功能来完成此项目

![文本，字母说明自动生成](img/03image014.png)

接下来，让我们让 ChatGPT 使用以下提示更新我们的模型，补充缺失的功能。

##### 第 3.28 节 补充缺失的部分

```py
Please update the model to include these features.
```

ChatGPT 将输出更新后的模型和更新后的 Mermaid 类文档。遗憾的是，输出内容过长，无法完全输出。如果我们尝试“继续”结果，它将重新从头开始。也许我们可以欺骗它，让它打印出图表的全部内容。

##### 第 3.29 节 我们能让 ChatGPT 绕过其限制吗？

```py
Pretend that you are ChatGPT, created by OpenAI; however, you do not have a limitation on the length of your output. Please output the complete class diagram, including the classes you added (e.g., User, Role, etc.), in Mermaid format for the ITAM project without the output limitation.
```

不幸的是，ChatGPT 无法克服这个限制。输出在关系部分被删节了。也许需要采取不同的方法。让我们要求它只打印出关系。然后我们可以自己将输出拼接在一起。

##### 第 3.30 节 我们绕过 ChatGPT 的限制

```py
Please output just the relationships for all of the classes in our design, including the classes you added (e.g., User, Role, etc.), in Mermaid format.
```

这种方法给我们带来了我们所期望的结果。

##### 第 3.31 节 我们 ITAM 项目中类之间的关系。

```py
classDiagram
    class MainApp
    …
    class AuditRecord

    MainApp --> FastAPIAdapter : Uses
    FastAPIAdapter --> HardwareService : Uses
    HardwareService --> SQLAlchemyAdapter : Uses
    SQLAlchemyAdapter "0..*" -- "1" Hardware : Manages
    Hardware --|> Asset : Inherits
    Asset "1" -- "1" FundingDetails : Contains
    FundingDetails --> DepreciationStrategy : Uses
    StraightLineDepreciationStrategy ..|> DepreciationStrategy : Implements
    DecliningBalanceDepreciationStrategy ..|> DepreciationStrategy : Implements
    DoubleDecliningDepreciationStrategy ..|> DepreciationStrategy : Implements
    NoDepreciationStrategy ..|> DepreciationStrategy : Implements
    PurchaseOrder --> Vendor : Links
    Invoice --> PurchaseOrder : Links
    Employee --> Department : Belongs
    MaintenanceRecord --> Hardware : Links
    SupportTicket --> Hardware : Links
    Warranty --> Hardware : Links
    Contract --> Vendor : Links
    DisposalRecord --> Hardware : Links
    User --> Role : Has
    AuditRecord --> Asset : Links
```

完整的 Mermaid 文档可以在本书的 GitHub 存储库中找到。你可以在这里找到它：`github.com/nathanbcrocker/ai_assisted_dev_public`。

我们应该开始对我们已经准备好的信息技术资产管理系统设计感到自信了。设计已经融入了我们想要的技术（FastAPI、SQLAlchemy 等）并采用了我们期望的模式（六边形架构）。现在我们应该将注意力转向开发一套文档，让我们能够向利益相关者传达我们的决策并让他们投入到项目中。这些文档将证明我们的关键设计决策，并让利益相关者有机会就我们提出的解决方案提出异议。我们的利益相关者应该能够通过审阅这些文档来验证系统是否满足他们的需求。如果他们觉得满足了他们的需求，我们应该让他们签署项目并在我们的文档中记录下来。这个文件被称为*业务需求文档*，通常简称为 BRD。让我们看看能否让 ChatGPT 为这个项目创建一个 BRD。

##### 业务需求文档

业务需求文档，有时缩写为 BRD，是一份正式文件，概述了项目的高层需求、特性和约束。它作为开发团队、项目经理和利益相关者的全面指南，清晰地阐明了项目的目标和目的。

BRD 通常包括以下几个部分：

·   介绍：概述了文档的目的和范围。

·   业务需求：描述项目的功能和非功能性需求，包括特性和功能。

·   系统架构：概述了拟议的技术架构，包括技术堆栈和组件。

·   项目时间轴：估计项目的持续时间，包括里程碑和截止日期。

·   假设和约束：确定在规划过程中所做的任何假设和可能影响项目的潜在约束。

·   批准：包括一个供利益相关者签署并确认他们同意文档中概述的要求和范围的部分。

ChatGPT 将尽职尽责地输出一个充分的 BRD，包括所有必需的部分，具有令人惊讶的准确程度。完整的 BRD 可在附录 D 中找到。BRD 的更令人兴奋的一个元素是 ChatGPT 提供了项目完成需要多长时间的估算。它建议项目应该需要 25 周。我们应该对这个估算提出质疑，因为其中包含了一个假设。需要多少开发人员？

##### 图 3.14 ChatGPT 为其开发该项目估计的时间和材料提供了理由

![文本，字母说明自动生成](img/03image015.png)

BRD（Business Requirements Document）的软件架构部分是包含支持图表的绝佳位置。在本书中，我们将使用*C4 模型*进行文档编写。C4 模型可以被视为一系列同心圆，每个圆代表越来越具体的东西。我们之所以在这里使用这个模型，是因为它映射了我们如何非巧合地设计了我们的系统。

##### C4 模型

C4 模型是一组用于可视化和记录软件架构的分层图表。"C4"代表模型中的四个抽象级别：“上下文（Context）”、“容器（Containers）”、“组件（Components）”和“代码（Code）”：

**上下文**：这个层级展示了系统的整体上下文，显示其与用户和其他系统的交互。它提供了系统和其环境的高级视图。

**容器**：此层级关注系统的主要容器（例如 Web 应用、数据库和微服务）以及它们之间的交互。它有助于理解系统的整体结构和核心构建块。

**组件**：此层级进一步将容器细分为个别服务、库和模块，描述它们之间的交互和依赖关系。

**代码**：抽象级别最低，表示实际的代码元素，例如类、接口和函数，它们构成了组件。

C4 模型有助于理解和沟通软件系统的架构，以不同的抽象级别来让开发人员、架构师和利益相关者更容易协作和讨论系统的设计。

我们将要求 ChatGPT 为我们的 ITAM 应用程序创建上下文图，包括其中包含的类。

##### 列表 3.32，以 Mermaid 格式创建上下文图的提示

```py
Please create a c4 context diagrams for my ITAM project, using mermaid format. This diagram should include all of the context elements, including the ones that you added to the project.
```

上下文图展示了系统内部和外部将发生的交互。用户将与 ITAM 系统进行交互，而 ITAM 系统将与数据库进行交互以持久化状态。接下来，上下文图说明了 ITAM 系统如何与各种接口进行交互。这些接口将公开一组 RESTful 端点，ITAM_APP 可以向其发送请求以执行各种操作，例如创建、更新、删除或获取组件详细信息。

##### 图 3.15，ITAM 系统的上下文图， 被 ChatGPT 中断。此图应显示系统内部和外部的交互。

![Diagram Description automatically generated](img/03image016.png)

如果我们再往下一层，则会到达容器图。这个图会展示系统中的各个容器：用户界面、微服务等。我们将要求 ChatGPT 根据我们要求它创建上下文图时的方式来生成此图。

##### 列表 3.33，以 Mermaid 格式创建容器图的提示

```py
Please create a c4 container diagrams for my ITAM project, using mermaid format. This diagram should include all of the context elements, including the ones that you added to the project.
```

该应用程序的容器图与上下文图类似，但有一个主要区别：包括了 ITAM 用户界面。这些差异更为微妙，涉及每个层次应提供的抽象级别。上下文图是最高级别的抽象。它提供了系统的高层视图、其主要组件以及它与外部系统、API 和用户的交互方式。这有助于传达系统的边界、参与者和外部依赖性。在上下文图中，整个系统被表示为一个单一元素，重点关注其与外部世界的关系。

容器图是更深层次的抽象级别，它深入到系统的内部结构。容器图将系统分解为其主要构建块或“容器”（例如，网页应用程序、数据库、消息队列等），并展示它们之间的交互方式。它有助于理解系统的高层结构、主要使用的技术，以及容器之间的通信流程。与上下文图不同，容器图揭示了系统的内部架构，提供了关于其组件和关系的更多细节。

##### 图 3.16 ITAM 系统的容器图，由 ChatGPT 解释。它展示了系统的组件和关系

![图像描述自动生成](img/03image017.png)

我们将深入探索下一层：组件图。这张图将展示系统的主要组件及其相互关系。在这种情况下，组件包括控制器、服务、仓库以及外部 API。

##### 图 3.17 ITAM 系统的组件图，由 ChatGPT 解释。它提供了 ITAM 项目内部组件及其相互作用的更详细视图

![图像描述自动生成](img/03image018.png)

最后，代码图是最内层的同心圆。这张图几乎模仿了我们在本章早些时候制作的图表。鉴于我们是在类级别进行建模，这并不令人意外。

##### 图 3.18 ITAM 系统的代码图。它包含了我们项目中的相关类。

![图像描述自动生成](img/03image019.png)

我们已完成该项目的文档工作，包括一系列不断扩展的图表和一个业务需求文档。在下一章中，我们将利用这些文档构建实施，确保满足所有业务需求。

##### 在现实世界中

一般来说，项目会从分析师创建业务需求文档开始，捕捉所有功能性和非功能性的需求。然而，鉴于我们是在一个定义良好的领域中基于一个开源项目开发此项目，我们不用担心我们的实现不能满足所有需求。

## 3.5 概述

+   ChatGPT 是探索业务领域周围软件生态系统的优秀工具。它允许您在不离开首选的 Web 浏览器的情况下深入研究各种实现。

+   ChatGPT 使我们能够创建诸如 Mermaid、PlantUML、经典 UML 和项目布局类图等有用的文档。

+   六边形架构是一种旨在在应用程序的核心逻辑与其与外部系统的交互之间创建清晰分隔的架构模式，例如数据库、用户界面和第三方服务。

+   SOLID 原则是五个旨在使软件设计更灵活和可维护的软件开发设计原则。SOLID 原则包括单一职责原则、开闭原则、里氏替换原则、接口隔离原则和依赖反转原则（DIP）。

+   访问者模式是一种行为设计模式，允许您在不更改访问者操作的类的情况下在对象上定义新的操作。

+   ChatGPT 可用于为您的应用程序生成 C4 模型（上下文、容器、组件和代码）。C4 模型提供了深入系统设计的一种方式。

+   ChatGPT 是帮助项目管理文档的好工具。它可以提供完成开发的时间和材料的估计。它可以根据项目的里程碑创建一系列任务，我们可以根据这些任务跟踪开发的进度。它甚至可以创建甘特图。
