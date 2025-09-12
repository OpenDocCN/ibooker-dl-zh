# 附录 A 技术要求

本附录描述了如何安装本书中使用的软件。

## A.1 克隆 GitHub 仓库

要在您的本地机器上克隆本书的 GitHub 仓库，您可以采用以下两种策略之一：

+   使用终端

+   使用 GitHub Desktop

### A.1.1 使用终端

要从命令行克隆本书的 GitHub 仓库，请按照以下步骤操作：

1.  如果您还没有安装 Git 套件（[`git-scm.com/downloads`](https://git-scm.com/downloads)），请安装它。

1.  打开您计算机上的终端。

1.  导航到您想要克隆仓库的目录。

1.  运行命令 `git clone https://github.com/alod83/Data-Storytelling-with-Altair-and-AI/tree/main`。

1.  等待仓库被克隆到您的本地机器。

### A.1.2 使用 GitHub Desktop

要从 GitHub Desktop 克隆本书的 GitHub 仓库，请按照以下步骤操作：

1.  从他们的官方网站下载并安装 GitHub Desktop：[`desktop.github.com/`](https://desktop.github.com/)。

1.  启动 GitHub Desktop。

1.  登录您的 GitHub 账户，如果需要，请创建一个新账户。

1.  点击“文件”菜单，选择“克隆仓库”。

1.  在“克隆仓库”窗口中，选择“URL”选项卡。

1.  在“仓库 URL”字段中输入仓库 URL（[`github.com/alod83/Data-Storytelling-with-Altair-and-AI/tree/main`](https://github.com/alod83/Data-Storytelling-with-Altair-and-AI/tree/main)）。

1.  选择您想要克隆仓库的本地路径。

1.  点击“克隆”按钮。

1.  等待 GitHub Desktop 将仓库克隆到您的本地机器。

## A.2 安装 Python 包

本书中的示例使用 Python 3.8。您可以从官方网站下载它：[`www.python.org/downloads/release/python-3810/`](https://www.python.org/downloads/release/python-3810/)。

本书描述的示例使用以下 Python 包：

+   langchain==0.1.12

+   langchain-community==0.0.28

+   langchain-core==0.1.32

+   langchain-openai==0.0.8

+   langchain-text-splitters==0.0.1

+   altair==5.3.0

+   chromadb==0.4.22

+   jupyterlab==3.5.1

+   ydata-profiling==4.6.0

+   matplotlib==3.5.0

+   numpy==1.24.4

+   pandas==1.3.4

+   unstructured==0.10.19

您可以通过运行命令 `pip install <package_name>` 简单地安装这些包的最新版本。然而，由于技术发展迅速，这些包在您阅读本书时可能已经更新。为确保您仍然可以按照本书中的代码运行，请在您的计算机上创建一个虚拟环境并安装本书中使用的特定包版本。要创建虚拟环境，请打开终端并使用 `pip install virtualenv` 安装 `virtualenv` 包。然后，运行以下列表中描述的命令。

##### 列表 A.1 创建和运行虚拟环境

```py
python -m venv env
source env/bin/activate
```

要停用虚拟环境，只需运行 `deactivate` 命令。在虚拟环境中，通过运行 `pip install <package_name>==<version>` 命令为每个包安装前面的包。或者，使用位于 GitHub 仓库根目录下的 requirements.txt 文件，并运行以下命令：`pip install -r requirements.txt`。

## A.3 安装 GitHub Copilot

要使用 GitHub Copilot，您必须为您的个人 GitHub 账户设置免费试用或订阅。如果您是教师或学生，您可以在以下链接设置免费订阅计划：[`education.github.com/discount_requests/pack_application`](https://education.github.com/discount_requests/pack_application)。

一旦您的账户设置为使用 GitHub Copilot，将其配置为 Visual Studio Code (VSC)的扩展，这是一个免费的开源代码编辑器，专为开发者编写和调试代码而设计。从其官方网站下载 VSC：[`visualstudio.microsoft.com/it/downloads/`](https://visualstudio.microsoft.com/it/downloads/)。

要启动 GitHub Copilot，打开 Visual Studio 并导航到“扩展”选项卡。下载并安装 GitHub Copilot 扩展，然后从仪表板中选择“连接到您的账户”。输入您的 GitHub 凭据。登录后，扩展将检测现有仓库并提供配置新项目的选项。

## A.4 配置 ChatGPT

要使用 ChatGPT，您必须在 Open AI 网站上（[`openai.com/`](https://openai.com/)）设置账户。在撰写本书时，ChatGPT 版本 GPT-4 是免费的。

要访问 ChatGPT，请访问 [`chat.openai.com/`](https://chat.openai.com/)，登录您的账户，并在输入文本框中开始编写提示，就像实时聊天一样。每当您想开始一个新话题时，通过点击左上角的“新建聊天”按钮创建一个新的聊天会话。

ChatGPT 保留单个聊天会话中所有提示的历史记录。这意味着您可以逐步编写指令，基于之前的提示进行构建，并保持对话的连贯性。

网页界面还提供了一种付费账户，它提供了一些额外的功能，例如使用高级模型的能力。在这本书中，我们使用网页界面的免费版本。

## A.5 安装 Open AI API

指向 [`openai.com/`](https://openai.com/)，登录您的账户或创建一个新账户。登录后，点击“API”。要使用 Open AI API，您必须将现有账户升级为付费账户。点击屏幕右上角的个人头像，在“管理账户 | 计费”旁边。接下来，通过添加支付方式来添加信用额度。OpenAI 官方定价页面（[`openai.com/pricing`](https://openai.com/pricing)）列出了定价详情。5 美元的信用额度应该足以进行本书中的实验。

接下来，您可以按照以下步骤配置 API 密钥：

1.  点击仪表板右上角的“个人”按钮。

1.  在下拉菜单中选择“查看 API 密钥”。

1.  点击“创建新的密钥”。

1.  插入密钥名称，创建一个密钥。

1.  点击复制符号以复制密钥值，并将其粘贴在安全的地方。

1.  点击“完成”。

一旦创建了 API 密钥，您就可以在 Python 脚本中使用 Open AI API。

以下列表展示了使用 Open AI API 的示例，要求 ChatGPT 根据输入提示生成一些输出。您也可以在本书附录 A/openai-test.py 中找到以下代码。

##### 列表 A.2 调用 ChatGPT API

```py
import openai
openai.api_key = 'MY_API_KEY'
prompt = 'What is the capital of Italy?'

messages = [{"role": "user", "content": prompt}]
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0, #1
)

print(response.choices[0].message.content.strip())
```

#1 温度 [0,1] 定义了输出中的随机程度。

注意：使用 `chat.completions.create()` 方法调用 ChatGPT API。

## A.6 安装 LangChain

您可以使用 LangChain 与您首选的大型语言模型 (LLM) 一起使用。在这本书中，我们使用 OpenAI，它需要一个 API 密钥。更多详情，请参阅与 OpenAI 安装相关的章节。

以下列表展示了使用 LangChain 在 Python 中与 OpenAI API 交互的示例。您也可以在本书附录 A/langchain-test.py 中找到此示例的代码。

##### 列表 A.3 使用 LangChain

```py
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = 'MY_API_KEY'

llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

print(
    llm.invoke("What is the capital of Italy?")
)
```

Windows 用户在使用 LangChain 时可能会遇到 `"ModuleNotFoundError:` `No` `module` `named` `'pwd'"` 错误。这是因为 `pwd` 模块在 Windows 平台上不可用。解决这个问题的方法之一是在使用 LangChain 之前添加以下代码行。

##### 列表 A.4 在 Windows 上使用 `pwd`

```py
try:
    import pwd
except ImportError:
    import winpwd as pwd
```

更多详情，请参阅 LangChain 官方文档 ([`www.langchain.com/`](https://www.langchain.com/))。

## A.7 安装 Chroma

Chroma 使用 SQLite 作为数据库。以下列表展示了如何连接到 Chroma 数据库。

##### 列表 A.5 导入 Chroma

```py
import chromadb

client = chromadb.Client()
```

更多详情，请参阅 Chroma 官方文档 ([`docs.trychroma.com/getting-started`](https://docs.trychroma.com/getting-started))。

## A.8 配置 DALL-E

要使用 DALL-E，您必须在 Open AI 网站上设置一个账户 ([`openai.com`](https://openai.com))。然后，您可以通过两种方式与 DALL-E 交互：网页界面或 Open AI API。在这本书中，我们更倾向于使用网页界面，因为它有几个优点。Open AI API 需要付费账户，而网页界面需要购买信用来生成图像。此外，如果您在 2023 年 4 月 6 日之前创建了 DALL-E 账户，每个月您都有一个免费信用池。

要使用网页界面，请访问 [`labs.openai.com`](https://labs.openai.com)，登录您的账户，并在输入文本框中编写您的提示。要使用 Open AI API，您必须将现有账户升级为付费账户。接下来，您可以按照第 2.5 节所述配置 API。一旦创建了 API 密钥，您就可以在 Python 脚本中使用 Open AI API。

以下列表展示了使用 Open AI API 的一个示例：根据输入提示让 DALL-E 生成图像。代码也包含在本书附录 A/dalle-test.py 的 GitHub 仓库中。

##### 列表 A.6 调用 DALL-E API

```py
import openai
import requests

openai.api_key = 'MY_API_KEY'
prompt = 'Create a painting of a beautiful sunset over a calm lake.'

n=1
response = openai.images.generate(
  prompt=prompt,
  n=n,  #1
  size='1024x1024'
)

i = 0
for image_data in response.data:
    print(image_data.url)
    img = requests.get(image_data.url).content
    with open(f"image-{i}.png", 'wb') as handler:
        handler.write(img)
    i += 1

print(output)
```

#1 生成图像的数量

注意：使用 `Image.create()` 方法调用 DALL-E API。输出包含生成图像的 URL。

此外，您还可以使用 `Image.create_edit()` 方法修改现有的图像。更多详细信息，请参阅 DALL-E 官方文档（[`platform.openai.com/docs/guides/images/usage`](https://platform.openai.com/docs/guides/images/usage)）。
