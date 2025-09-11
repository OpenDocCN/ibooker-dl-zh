# 第三章：使用 GPT-3 进行编程

几乎所有 GPT-3 的 NLP 功能都是使用 Python 编程语言创建的。但为了实现更广泛的可访问性，API 内置支持所有主要的编程语言，因此用户可以使用他们选择的编程语言构建基于 GPT-3 的应用程序。

在本节中，我们将通过复制一个示例来说明这是如何工作的，该示例涉及三种常见的编程语言：Python、Go 和 Java。

提醒一下：在每个特定语言的部分中，我们假设你对所讨论的编程语言有基本的了解。如果没有，你可以安全地跳过该部分。

# 使用 Python 的 OpenAI API

Python 是数据科学和机器学习任务中最流行的语言。与传统的数据科学编程语言（如 R 和 Stata）相比，Python 的优势在于其可扩展性和与数据库的良好集成。它被广泛使用，并有一个蓬勃发展的开发者社区，使其生态系统保持更新。Python 容易学习，并且带有诸如 NumPy 和 pandas 等有用的数据科学库。

你可以使用一个叫做 [Chronology](https://oreil.ly/9eULI) 的库将 GPT-3 与 Python 配对，该库提供了一个简单直观的接口。Chronology 可以减轻每次编写所有代码的单调工作。它的特点包括以下内容：

+   它异步调用 OpenAI API，允许你同时生成多个提示完成。

+   你可以轻松创建和修改训练提示；例如，修改一个不同示例中使用的训练提示相对比较简单。

+   它允许你通过将一个提示的输出插入到另一个提示中来链接提示。

Chronology 托管在 PyPI 上，并支持 Python 3.6 及以上版本。要安装该库，请运行以下命令：

```py
pip install chronological
```

在通过 PyPI 安装 Python 库之后，让我们看一下如何通过 GPT-3 来总结一个给定文本文档的二年级阅读水平的示例。我们将向你展示如何调用 API，将训练提示作为请求发送，并将总结完成作为输出。我们已经在 [GitHub 存储库](https://oreil.ly/nVaC9) 中为你发布了代码。

在此示例中，我们将使用以下训练提示：

```py
My second-grader asked me what this passage means:
"""
Olive oil is a liquid fat obtained from olives (the fruit of *Olea europaea*; 
family Oleaceae)...
"""
I rephrased it for him, in plain language a second-grader can understand:
"""
```

首先，导入以下依赖项：

```py
# Importing Dependencies
from chronological import read_prompt, cleaned_completion, main
```

现在我们可以创建一个函数，该函数读取训练提示并提供完成输出。我们将此函数异步化，这允许我们进行并行函数调用。我们将使用以下 API 参数配置：

+   最大标记数=100

+   执行引擎=“Davinci”

+   温度=0.5

+   顶部-p=1

+   频率惩罚=0.2

+   停止序列=[“\n\n”]

```py
# Takes in the training prompt and returns the completed response 
async def summarization_example(): 
    # Takes in a text file(summarize_for_a_2nd_grader) as the input prompt
    prompt_summarize = read_prompt('summarize_for_a_2nd_grader')    
    # Calling the completion method along with the specific GPT-3 parameters
    completion_summarize = await cleaned_completion(prompt_summarize, 
    max_tokens=100, engine="davinci", temperature=0.5, top_p=1, 
    frequency_penalty=0.2, stop=["\n\n"])
    # Return the completion response
    return completion_summarize
```

现在我们可以创建一个异步工作流，使用库提供的`main`函数调用该工作流，并在控制台中打印输出：

```py
# Designing the end-to-end async workflow, capable of running multiple prompts 
# in parallel  
async def workflow():
    # Making async call to the summarization function
    text_summ_example = await summarization_example()
    # Printing the result in console
    print('-------------------------')
    print('Basic Example Response: {0}'.format(text_summ_example))
    print('-------------------------')
# invoke Chronology by using the main function to run the async workflow
main(workflow)
```

将其保存为名为*text_summarization.py*的 Python 脚本，并从终端运行以生成输出。您可以从根目录运行以下命令：

```py
python text_summarization.py
```

执行脚本后，您的控制台应打印出以下提示的摘要：

```py
-------------------------
Basic Example Response: Olive oil is a liquid fat that comes from olives. 
Olives grow on a tree called an olive tree. The olive tree is the most common 
tree in the Mediterranean. People use the oil to cook with, to put on their 
salads, and as a fuel for lamps.
-------------------------
```

如果您对 Python 不熟悉并且想要在不编写代码的情况下链式连接不同的提示，您可以使用基于[Chronology 库](https://oreil.ly/f02Cx)构建的[无代码界面](https://oreil.ly/L2TUK)来通过拖放创建提示工作流程。请查看我们的 GitHub[存储库](https://oreil.ly/AfTQM)以了解更多您可以如何使用 Python 编程与 GPT-3 进行交互的示例。

# 使用 Go 调用 OpenAI API

Go 是一种开源编程语言，结合了其他编程语言的最佳特性，将解释的、动态类型的语言的编程便利性与静态类型的、编译的语言的效率和安全性结合在一起。开发人员通常称其为“21 世纪的 C 语言”。

Go 是构建需要高安全性、高速度和高模块化的项目的首选语言。这使得它成为金融科技行业许多项目的有吸引力的选择。Go 的主要特点如下：

+   使用便捷

+   最先进的生产力

+   高效的静态类型化

+   网络高效性能

+   充分利用多核处理能力

如果您完全不熟悉 Go 并想试试，您可以[查看文档](https://oreil.ly/3ZjiB)开始使用。

安装完成并且了解了 Go 编程的基础知识后，您可以按照以下步骤使用[用于 GPT-3 的 Go API 包装器](https://oreil.ly/j6lUY)。要了解更多关于创建 Go 模块的信息，请参阅[本教程](https://oreil.ly/w1334)。

首先，您将创建一个模块来跟踪和导入代码依赖项。使用以下命令创建并初始化`gogpt`模块：

```py
go mod init gogpt
```

创建`gogpt`模块后，让我们将其指向[此 GitHub 存储库](https://oreil.ly/6o2Hj)以下载使用 API 的必要依赖项和包。使用以下命令：

```py
go get github.com/sashabaranov/go-gpt3
go get: added github.com/sashabaranov/go-gpt3 v0.0.0-20210606183212-2be4a268a894
```

我们将使用与上一节相同的文本摘要示例。（您可以在以下[存储库](https://oreil.ly/r5HhV)中找到所有代码。）

让我们首先导入必要的依赖和包：

```py
# Calling the package main
package main
# Importing Dependencies
import (
    "fmt"
    "io/ioutil"
    "context"
    gogpt "github.com/sashabaranov/go-gpt3"
)
```

Go 编程将源文件组织成称为*包*的系统目录，这使得在 Go 应用程序之间重用代码变得更容易。在代码的第一行中，我们调用`main`包，告诉 Go 编译器该包应该编译为可执行程序，而不是共享库。

###### 注意

在 Go 语言中，每当你构建可重用的代码片段时，你会开发一个作为共享库的包。但是当你开发可执行程序时，你将使用 `main` 包来使包成为可执行程序。因为我们将在该包中调用主函数，`main` 将被设置为我们可执行程序的入口点。

现在，你将创建一个 `main` 函数，它将承载读取训练提示和提供完成输出的整个逻辑。使用以下配置作为 API 参数：

+   最大标记数=100

+   执行引擎=“Davinci”

+   温度=0.5

+   Top-p=1

+   频率惩罚=0.2

+   停止序列=[“\n\n”]

```py
func main() {
    c := gogpt.NewClient("OPENAI-API-KEY")
    ctx := context.Background()    
    prompt, err := ioutil.ReadFile("prompts/summarize_for_a_2nd_grader.txt")
    req := gogpt.CompletionRequest{
        MaxTokens: 100,
        Temperature: 0.5,
        TopP: 1.0,
        Stop: []string{"\n\n"},
        FrequencyPenalty: 0.2,
        Prompt: string(prompt),
    }
    resp, err := c.CreateCompletion(ctx, "davinci", req)
    if err != nil {
        return
    }

    fmt.Println("-------------------------")
    fmt.Println(resp.Choices[0].Text)
    fmt.Println("-------------------------")
}
```

该代码执行以下任务：

1.  通过提供 API 令牌设置一个新的 API 客户端，然后将其留在后台运行。

1.  以文本文件形式读取 " " 作为提示，从 *prompts* 文件夹中读取。

1.  通过提供训练提示和指定 API 参数值（如温度、Top P、停止序列等）来创建完成请求。

1.  调用 `create completion` 函数，并向其提供 API 客户端、完成请求和执行引擎。

1.  以完成的形式生成响应，该响应朝向控制台的末尾打印。

然后，你可以将代码文件保存为 *text_summarization.go* 并从终端运行它以生成输出。使用以下命令从你的根文件夹运行该文件：

```py
go run text_summarization.go
```

一旦你执行了该文件，你的控制台将打印出以下输出：

```py
-------------------------
Olive oil is a liquid fat that comes from olives. Olives grow on a tree called an 
olive tree. The olive tree is the most common tree in the Mediterranean. People 
use the oil to cook with, to put on their salads, and as a fuel for lamps.
-------------------------
```

要了解如何使用 Go 编程与 GPT-3 进行交互的更多示例，请访问我们的 GitHub [存储库](https://oreil.ly/r5HhV)。

# 使用 Java 调用 OpenAI API

Java 是用于开发传统软件系统的最古老和最流行的编程语言之一；它还是一个带有运行时环境的平台。它是由 Sun Microsystems（现在是 Oracle 的子公司）于 1995 年开发的，截至今天，超过三十亿台设备都在运行它。它是一种通用的、基于类的、面向对象的编程语言，旨在具有较少的实现依赖性。其语法与 C 和 C++ 类似。三分之二的软件行业仍然将 Java 作为其核心编程语言。

让我们再次使用橄榄油文本摘要示例。与 Python 和 Go 一样，我们将向你展示如何调用 API、将训练提示作为请求发送，并使用 Java 获取摘要完成作为输出。

要在本地计算机上逐步浏览代码，请克隆我们的 GitHub [存储库](https://oreil.ly/gpt3-repo)。在克隆的存储库中转到 *Programming_with_GPT-3* 文件夹，然后打开 *GPT-3_Java* 文件夹。

首先，导入所有相关的依赖项：

```py
package example;

// Importing Dependencies 
import java.util.*;  
import java.io.*;
import com.theokanning.openai.OpenAiService;
import com.theokanning.openai.completion.CompletionRequest;
import com.theokanning.openai.engine.Engine;
```

现在，你将创建一个名为 `OpenAiApiExample` 的类。你的所有代码都将成为它的一部分。在该类下，首先使用 API 令牌创建一个 `OpenAiService` 对象：

```py
class OpenAiApiExample {
    public static void main(String... args) throws FileNotFoundException {

        String token = "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";
        OpenAiService service = new OpenAiService(token);
```

现在 OpenAI API 的连接已经建立成一个*service object*的形式。从*prompts*文件夹中读取训练提示：

```py
// Reading the training prompt from the prompts folder
File file = new File("D:\\GPT-3 Book\\Programming with GPT-3\\GPT-3 
Java\\example\\src\\main\\java\\example\\prompts\\
summarize_for_a_2nd_grader.txt");

Scanner sc = new Scanner(file);

// we just need to use \\Z as delimiter
sc.useDelimiter("\\Z");

// pp is the string consisting of the training prompt
String pp = sc.next();
```

然后，您可以使用以下配置创建一个 API 参数的完成请求：

+   最大令牌数=100

+   执行引擎=“Davinci”

+   温度=0.5

+   Top-p=1

+   频率惩罚=0.2

+   停止序列=[“\n\n”]

```py
// Creating a list of strings to used as stop sequence
List<String> li = new ArrayList<String>();    
li.add("\n\n"''");
// Creating a completion request with the API parameters
CompletionRequest completionRequest = CompletionRequest.builder().prompt(pp)
.maxTokens(100).temperature(0.5).topP(1.0).frequencyPenalty(0.2).stop(li)
.echo(true).build();
// Using the service object to fetch the completion response 
service.createCompletion("davinci",completionRequest).getChoices().forEach
(System.out::println);
```

将代码文件保存为*text_summarization.java*并从终端运行以生成输出。您可以使用以下命令从根文件夹运行文件：

```py
./gradlew example:run
```

您的控制台应该打印与以前示例相同的摘要。有关如何使用 Java 与 GPT-3 进行交互的更多示例，请参阅我们的 GitHub [存储库](https://oreil.ly/amtjY)。

# GPT-3 沙盒由 Streamlit 提供支持

在这一节中，我们将带您了解 GPT-3 沙盒，这是我们创建的一个开源工具，提供样板代码，帮助您只需几行 Python 代码就能将您的想法变为现实。我们将向您展示如何使用它，以及如何根据您的特定应用进行自定义。

我们沙盒的目标是让您能够创建酷炫的 Web 应用程序，无论您的技术背景如何。它是建立在 Streamlit 框架之上的。

为了配合本书，我们还创建了一个[视频系列](https://oreil.ly/jQrlG)，其中提供了逐步创建和部署 GPT-3 应用程序的说明，您可以通过扫描 图 3-1 中的 QR 码来访问。请在阅读本章时一并参考。

![](img/gpt3_0301.png)

###### 图 3-1\. GPT-3 沙盒视频系列的 QR 码

我们在示例中使用 Visual Studio Code 作为 IDE，但您可以随意使用任何 IDE。在开始之前，您需要安装 IDE。请确保您正在运行 Python 版本为 3.7 或更高版本。您可以通过运行以下命令确认已安装的版本：

```py
python --version
```

通过在您的 IDE 中打开新终端并使用以下命令克隆此[存储库](https://oreil.ly/gpt3-repo)的代码：

```py
git clone https://github.com/Shubhamsaboo/kairos_gpt3
```

在您克隆存储库之后，您 IDE 中的代码结构应该如 图 3-2 所示。

![](img/gpt3_0302.png)

###### 图 3-2\. 沙盒文件目录结构

您所需要创建和部署 Web 应用程序的一切都已经在代码中存在。您只需要调整一些文件，以定制沙盒以适应您的特定用例。

创建一个[Python 虚拟环境](https://oreil.ly/iUWv4)，命名为*env*。然后您可以安装所需的依赖项。

转到*email_generation*文件夹。你的路径应该像这样：

```py
(env) kairos_gpt3\GPT-3 Sandbox\email_generation>
```

从那里，运行以下命令：

```py
(env) kairos_gpt3\GPT-3 Sandbox\email_generation> pip install -r requirements.txt
```

现在您可以开始定制沙箱代码了。 您需要查看的第一个文件是*training_data.py*。 打开该文件并用您想要使用的训练提示替换默认提示。 您可以使用 GPT-3 Playground 尝试不同的训练提示（有关自定义沙箱的更多信息，请参阅第二章和我们的[视频](https://oreil.ly/nCIgG)）。

您现在可以根据应用程序用例的要求调整 API 参数（最大标记数、执行引擎、温度、Top P、频率惩罚、停止序列）。 我们建议您在 Playground 中尝试不同的参数值以确定哪些值适用于您的用例。 一旦您获得满意的结果，那么您就可以在*model_training_service.py*文件中更改值。

就是这样！ 您的基于 GPT-3 的 Web 应用程序现在已准备就绪。 您可以使用以下命令在本地运行它：

```py
(env) kairos_gpt3\GPT-3 Sandbox\email_generation> streamlit run gpt_app.py
```

确保它正常工作，然后您可以使用 Streamlit 共享将应用程序部署到互联网，以向更广泛的受众展示它。 我们的[视频](https://oreil.ly/h5uTe)提供了完整的部署演示。

###### 注意

本应用程序遵循一个简单的工作流程，其中训练提示从 UI 接收单个输入并生成响应。 如果您的应用程序需要更复杂的工作流程，其中训练提示接受多个输入，请通过查看脚本*app1.py*、*app2.py*和*gpt_app.py*来自定义 UI 元素。 有关详细信息，请参阅[Streamlit 文档](https://docs.streamlit.io)。

在接下来的几章中，我们将探讨 GPT-3 的不同应用以及成功的企业是如何基于它构建的。

# 使用基于 GPT-3 的应用程序上线

您准备好将基于 GPT-3 的应用程序投入生产了吗？

在您开始之前，让我们讨论一些风险缓解措施。 在第六章中，您将了解一些 GPT-3 可能被用来造成伤害的方法。 为了防范这些恶意行为，OpenAI 已经制定了必须在您的应用程序上线之前遵循的准则和程序。 目前，您可以向五个人提供 API 而无需事先批准，但是如果需要更多，则需要申请预上线生产审查。 我们强烈建议您在申请之前阅读[使用指南](https://oreil.ly/oGBGJ)。

当你提交[Pre-Launch Review Request](https://oreil.ly/SO0VZ)时，OpenAI 团队会详细审查你的用例，并标记任何潜在违反 API [安全最佳实践](https://oreil.ly/gXRhC)的行为。如果你的请求被批准，OpenAI 将为你提供一个最大的消费限制，随着你建立记录，这个限制会随着时间增加。随着用户基础的增长，你可以提交[Quota Increase Request](https://oreil.ly/b25mG)。这样一来，你就可以自由地构建和部署你的应用程序，而 OpenAI 则会监控其对平台的潜在影响。

# 结论

在本章中，我们学习了如何使用 OpenAI API 来编程使用 Python、Go 和 Java 语言。我们还介绍了使用 Streamlit 创建的低代码沙盒环境，它将帮助你快速将你的想法变成应用程序。最后，我们查看了上线 GPT-3 应用程序的关键要求。这一章节为你提供了 API 的编程前景；未来我们将更深入地探讨 GPT-3 赋能的蓬勃生态系统。
