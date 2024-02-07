# 第7章. 将PyTorch部署到生产环境

到目前为止，本书大部分内容都集中在模型设计和训练上。早期章节向您展示了如何使用PyTorch的内置能力设计您的模型，并创建自定义的NN模块、损失函数、优化器和其他算法。在上一章中，我们看到了如何使用分布式训练和模型优化来加速模型训练时间，并最大程度地减少运行模型所需的资源。

到目前为止，您已经拥有创建一些经过良好训练、尖端的NN模型所需的一切，但不要让您的创新孤立存在。现在是时候通过应用程序将您的模型部署到世界上了。

过去，从研究到生产是一项具有挑战性的任务，需要一个软件工程团队将PyTorch模型移至一个框架并将其集成到（通常非Python）生产环境中。如今，PyTorch包括内置工具和外部库，支持快速部署到各种生产环境。

在本章中，我们专注于将您的模型部署到推理而非训练，并探讨如何将经过训练的PyTorch模型部署到各种应用程序中。首先，我将描述PyTorch内置的各种能力和工具，供您用于部署。像TorchServe和TorchScript这样的工具使您能够轻松地将PyTorch模型部署到云端、移动设备或边缘设备。

根据应用程序和环境，您可能有多种部署选项，每种选项都有其自身的权衡。我将向您展示如何在多个云端和边缘环境中部署您的PyTorch模型的示例。您将学习如何部署到用于开发和生产规模的Web服务器，iOS和Android移动设备，以及基于ARM处理器、GPU和现场可编程门阵列（FPGA）硬件的物联网（IoT）设备。

本章还将提供参考代码，包括我们使用的关键API和库的引用，以便轻松入门。当您需要部署模型时，您可以参考本章进行快速查阅，以便在云端或移动环境中展示您的应用程序。

让我们开始回顾PyTorch提供的资源，以帮助您部署您的模型。

# PyTorch部署工具和库

PyTorch包括内置工具和能力，以便将您的模型部署到生产环境和边缘设备。在本节中，我们将探索这些工具，并在本章的其余部分将它们应用于各种环境中。

PyTorch的部署能力包括其自然的Python API，以及TorchServe、TorchScript、ONNX和移动库。由于PyTorch的自然API基于Python，PyTorch模型可以在任何支持Python的环境中直接部署。

[表7-1](#table_deployment_resources) 总结了可用于部署的各种资源，并指示如何适当地使用每种资源。

表7-1. 部署PyTorch资源

| 资源 | 使用 |
| --- | --- |
| Python API | 进行快速原型设计、训练和实验；编写Python运行时程序。 |
| TorchScript | 提高性能和可移植性（例如，在C++中加载和运行模型）；编写非Python运行时或严格的延迟和性能要求。 |
| TorchServe | 一个快速的生产环境工具，具有模型存储、A/B测试、监控和RESTful API。 |
| ONNX | 部署到具有ONNX运行时或FPGA设备的系统。 |
| 移动库 | 部署到iOS和Android设备。 |

以下各节为每个部署资源提供参考和一些示例代码。在每种情况下，我们将使用相同的示例模型，接下来将对其进行描述。

## 常见示例模型

对于本章提供的每个部署资源示例和应用程序，以及参考代码，我们将使用相同的模型。对于我们的示例，我们将使用一个使用ImageNet数据预训练的VGG16模型来部署一个图像分类器。这样，每个部分都可以专注于使用的部署方法，而不是模型本身。对于每种方法，您可以用自己的模型替换VGG16模型，并按照相同的工作流程来实现您自己设计的结果。

以下代码实例化了本章中将要使用的模型：

```py
from torchvision.models import vgg16

model = vgg16(pretrained=True)
```

我们之前使用过VGG16模型。为了让您了解模型的复杂性，让我们使用以下代码打印出可训练参数的数量：

```py
import numpy as np

model_parameters = filter(lambda p:
      p.requires_grad, model.parameters())

params = sum([np.prod(p.size()) for
      p in model_parameters])
print(params)

# out: 138357544
```

VGG16模型有138,357,544个可训练参数。当我们逐步进行每种方法时，请记住在这个复杂性水平上的性能。您可以将其用作比较您模型复杂性的粗略基准。

在实例化VGG16模型后，将其部署到Python应用程序中需要很少的工作。事实上，在之前的章节中测试我们的模型时，我们已经这样做了。在我们进入其他方法之前，让我们再次回顾一下这个过程。

## Python API

Python API并不是一个新资源。这是我们在整本书中一直在使用的资源。我在这里提到它是为了指出您可以在不更改代码的情况下部署您的PyTorch模型。在这种情况下，您只需从任何Python应用程序中以评估模式调用您的模型，如下面的代码所示：

```py
import system
import torch

if __name__ == "__main__":
  model = MyModel()
  model.load_state_dict(torch.load(PATH))
  model.eval()
  outputs = model(inputs)
  print(outputs)
```

该代码加载模型，传入输入，并打印输出。这是一个简单的独立Python应用程序。您将看到如何在本章后面使用RESTful API和Flask将模型部署到Python Web服务器。使用Flask Web服务器，您可以构建一个快速的浏览器应用程序，展示您模型的能力。

Python并不总是在生产环境中使用，因为它的性能较慢，而且缺乏真正的多线程。如果您的生产环境使用另一种语言（例如C++、Java、Rust或Go），您可以将您的模型转换为TorchScript代码。

## TorchScript

TorchScript是一种序列化和优化PyTorch模型代码的方式，使得PyTorch模型可以在非Python运行环境中保存和执行，而不依赖于Python。TorchScript通常用于在C++中运行PyTorch模型，并与支持C++绑定的任何语言一起使用。

TorchScript代表了一个PyTorch模型的格式，可以被TorchScript编译器理解、编译和序列化。TorchScript编译器创建了一个序列化的优化版本的您的模型，可以在C++应用程序中使用。要在C++中加载您的TorchScript模型，您将使用名为*LibTorch*的PyTorch C++ API库。

有两种方法可以将您的PyTorch模型转换为TorchScript。第一种称为*tracing*，这是一个过程，在这个过程中，您传入一个示例输入并使用一行代码进行转换。在大多数情况下使用。第二种称为*scripting*，当您的模型具有更复杂的控制代码时使用。例如，如果您的模型具有依赖于输入本身的条件`if`语句，您将需要使用scripting。让我们看一下每种情况的一些参考代码。

由于我们的VGG16示例模型没有任何控制流，我们可以使用跟踪来将我们的模型转换为TorchScript，如下面的代码所示：

```py
import torch

model = vgg16(pretrained=True)
example_input = torch.rand(1, 3, 224, 224)
torchscript_model = torch.jit.trace(model,
                            example_input)
torchscript_model.save("traced_vgg16_model.pt")
```

该代码创建了一个Python可调用模型`torchscript_model`，可以使用正常的PyTorch方法进行评估，例如`output = torchscript_model(inputs)`。一旦我们保存了模型，就可以在C++应用程序中使用它。

###### 注意

在PyTorch中评估模型的“正常”方法通常被称为`eager mode`，因为这是开发中评估模型的最快方法。

如果我们的模型使用了控制流，我们将需要使用注释方法将其转换为TorchScript。让我们考虑以下模型：

```py
import torch.nn as nn

class ControlFlowModel(nn.Module):
  def __init__(self, N):
    super(ControlFlowModel, self).__init__()
    self.fc = nn.Linear(N,100)

  def forward(self, input):
    if input.sum() > 0:
      output = input
    else:
      output = -input
    return output

model = ControlFlowModel(10)
torchcript_model = torch.jit.script(model)
torchscript_model.save("scripted_vgg16_model.pt")
```

在这个例子中，`ControlFlowModel`的输出和权重取决于输入值。在这种情况下，我们需要使用`torch.jit.script()`，然后我们可以像跟踪一样将模型保存到TorchScript中。

现在我们可以在C++应用程序中使用我们的模型，如下所示的C++代码：

```py

include<torch/script.h>#include <iostream>#include <memory>intmain(intargc,constchar*argv[]){if(argc!=2){std::cerr<<"usage: example-app">>\
"*`<path-to-exported-script-module>`*\n";return-1;}torch::jit::script::Modulemodel;model=torch::jit::load(argv[1]);std::vector<torch::jit::IValue>inputs;inputs.push_back(\
torch::ones({1,3,224,224}));at::Tensoroutput=model.forward(inputs).toTensor();std::cout\
<<output.slice(/*dim=*/1,\
/*start=*/0,/*end=*/5)\
<<'\n';}}
```

我们将TorchScript模块的文件名传递给程序，并使用`torch::jit::load()`加载模型。然后我们创建一个示例输入向量，将其通过我们的TorchScript模型运行，并将输出转换为张量，打印到`stdout`。

TorchScript API提供了额外的函数来支持将模型转换为TorchScript。[表7-2](#tableTorchScriptapi)列出了支持的函数。

表7-2\. TorchScript API函数

| 函数 | 描述 |
| --- | --- |
| `script(`*`obj[, optimize, _frames_up, _rcb]`*`)` | 检查源代码，使用*TorchScript*编译器将其编译为*TorchScript*代码，并返回一个`ScriptModule`或`ScriptFunction` |
| `trace(`*`func, example_inputs[, optimize, ...]`*`)` | 跟踪一个函数并返回一个可执行的或者`ScriptFunction`，将使用即时编译进行优化 |
| `script_if_tracing(`*`fn`*`)` | 在跟踪期间首次调用`fn`时编译 |
| `trace_module(`*`mod, inputs[, optimize, ...]`*`)` | 跟踪一个模块并返回一个可执行的`ScriptModule`，将使用即时编译进行优化 |
| `fork(`*`func, *args, **kwargs`*`)` | 创建一个异步任务，执行`func`并引用此执行结果的值 |
| `wait(`*`future`*`)` | 强制完成一个`torch.jit.Future[T]`异步任务，返回任务的结果 |
| `ScriptModule()` | 将脚本封装成C++ `torch::jit::Module` |
| `ScriptFunction()` | 与`ScriptModule()`相同，但表示一个单独的函数，没有任何属性或参数 |
| `freeze(`*`mod[, preserved_attrs]`*`)` | 克隆一个`ScriptModule`，并尝试将克隆模块的子模块、参数和属性内联为*TorchScript* IR图中的常量 |
| `save(`*`m, f[, _extra_files]`*`)` | 保存模块的离线版本，用于在单独的进程中使用 |
| `load(`*`f[, map_location, _extra_files]`*`)` | 加载之前使用`torch.jit.save()`保存的`ScriptModule`或`ScriptFunction` |
| `ignore(`*`[drop]`*`)` | 指示编译器忽略一个函数或方法，并保留为Python函数 |
| `unused(`*`fn`*`)` | 指示编译器忽略一个函数或方法，并替换为引发异常 |
| `isinstance(`*`obj, target_type`*`)` | 在TorchScript中提供容器类型细化 |

在本节中，我们使用TorchScript来提高模型在C++应用程序或绑定到C++的语言中的性能。然而，规模化部署PyTorch模型需要额外的功能，比如打包模型、配置运行环境、暴露API端点、日志和监控，以及管理多个模型版本。幸运的是，PyTorch提供了一个名为TorchServe的工具，以便于这些任务，并快速部署您的模型进行规模推理。

## TorchServe

TorchServe是一个开源的模型服务框架，可以轻松部署训练好的PyTorch模型。它由AWS工程师开发，并于2020年4月与Facebook联合发布，目前由AWS积极维护。TorchServe支持部署模型到生产环境所需的所有功能，包括多模型服务、模型版本控制用于A/B测试、日志和监控指标、以及与其他系统集成的RESTful API。[图7-1](#fig_torchserve)展示了TorchServe的工作原理。

![“TorchServe Architecture”](Images/ptpr_0701.png)

###### 图7-1\. TorchServe架构

客户端应用程序通过多个 API 与 TorchServe 进行交互。推理 API 提供主要的推理请求和预测。客户端应用程序通过 RESTful API 请求发送输入数据，并接收预测结果。管理 API 允许您注册和管理已部署的模型。您可以注册、注销、设置默认模型、配置 A/B 测试、检查状态，并为模型指定工作人员数量。指标 API 允许您监视每个模型的性能。

TorchServe 运行所有模型实例并捕获服务器日志。它处理前端 API 并管理模型存储到磁盘。TorchServe 还为常见应用程序提供了许多默认处理程序，如目标检测和文本分类。处理程序负责将 API 中的数据转换为模型将处理的格式。这有助于加快部署速度，因为您不必为这些常见应用程序编写自定义代码。

###### 警告

TorchServe 是实验性的，可能会发生变化。

要通过 TorchServe 部署您的模型，您需要遵循几个步骤。首先，您需要安装 TorchServe 的工具。然后，您将使用模型存档工具打包您的模型。一旦您的模型被存档，您将运行 TorchServe Web 服务器。一旦 Web 服务器运行，您可以使用其 API 请求预测，管理您的模型，执行监控，或访问服务器日志。让我们看看如何执行每个步骤。

### 安装 TorchServe 和 torch-model-archiver

AWS 在 Amazon SageMaker 或 Amazon EC2 实例中提供了预安装的 TorchServe 机器。如果您使用不同的云提供商，请在开始之前检查是否存在预安装实例。如果您使用本地服务器或需要安装 TorchServe，请参阅[TorchServe安装说明](https://pytorch.tips/torchserve-install)。

尝试的一个简单方法是使用 `conda` 或 `pip` 进行安装，如下所示：

```py

$ `conda``install``torchserve``torch-model-archiver``-c``pytorch`$ `pip``install``torchserve``torch-model-archiver`
```

如果遇到问题，请参考上述链接中的 TorchServe 安装说明。

### 打包模型存档

TorchServe 有能力将所有模型工件打包到单个模型存档文件中。为此，我们将使用我们在上一步中安装的 `torch-model-archiver` 命令行工具。它将模型检查点以及 `state_dict` 打包到一个 *.mar* 文件中，TorchServe 服务器将使用该文件来提供模型服务。

您可以使用 `torch-model-archiver` 来存档您的 TorchScript 模型以及标准的 “eager 模式” 实现，如下所示。

对于 TorchScript 模型，命令行如下：

```py

$`torch``-``model``-``archiver``-``-``model``-``name``vgg16``-``-``version``1.0``-``-``serialized``-``file``model``.``pt``-``-``handler``image_classifier`
```

我们将模型设置为我们的示例 VGG16 模型，并使用保存的序列化文件 *model.pt*。在这种情况下，我们也可以使用默认的 `image_classifier` 处理程序。

对于标准的 eager 模式模型，我们将使用以下命令：

```py

$`torch``-``model``-``archiver``-``-``model``-``name``vgg16``-``-``version``1.0``-``-``model``-``file``model``.``py``-``-``serialized``-``file``model``.``pt``-``-``handler``image_classifier`
```

这与之前的命令类似，但我们还需要指定模型文件 *model.py*。

`torch-model-archiver` 工具的完整选项集如下所示：

```py

$ `torch-model-archiver``-h`usage:torch-model-archiver[-h]--model-nameMODEL_NAME--versionMODEL_VERSION_NUMBER--model-fileMODEL_FILE_PATH--serialized-fileMODEL_SERIALIZED_PATH--handlerHANDLER[--runtime{python,python2,python3}][--export-pathEXPORT_PATH][-f][--requirements-file]
```

表 7-3\. 模型存档工具选项

| 选项 | 描述 |
| --- | --- |
| `-h`, `--help` | 帮助消息。显示帮助消息后，程序将退出。 |
| `--model-name *MODEL_NAME*` | 导出的模型名称。导出的文件将命名为 *<model-name>.mar*，如果未指定 `--export-path`，则将保存在当前工作目录中，否则将保存在导出路径下。 |
| `--serialized-file` `*SERIALIZED_FILE*` | 指向包含 `state_dict` 的 _.pt_ 或 _.pth_ 文件的路径，对于 eager 模式，或者包含可执行 `ScriptModule` 的路径，对于 TorchScript。 |
| `--model-file *MODEL_FILE*` | 指向包含模型架构的 Python 文件的路径。对于 eager 模式模型，此参数是必需的。模型架构文件必须只包含一个从 `torch.nn.modules` 扩展的类定义。 |
| `--handler *HANDLER*` | *TorchServe*的默认处理程序名称或处理自定义*TorchServe*推理逻辑的Python文件路径。 |
| `--extra-files *EXTRA_FILES*` | 逗号分隔的额外依赖文件的路径。 |
| `--runtime *{python, python2, python3}*` | 运行时指定要在其上运行推理代码的语言。默认运行时为`RuntimeType.PYTHON`，但目前支持以下运行时：`python`，`python2`和`python3`。 |
| `--export-path *EXPORT_PATH*` | 导出的_.mar_文件将保存在的路径。这是一个可选参数。如果未指定`--export-path`，文件将保存在当前工作目录中。 |
| `--archive-format *{tgz, no-archive, default}*` | 存档模型工件的格式。`tgz`以*<model-name>.tar.gz*格式创建模型存档。如果平台托管需要模型工件为*.tar.gz*，请使用此选项。`no-archive`在*<export-path>*/*<model-name>*处创建模型工件的非存档版本。由于此选择，将在该位置创建一个MANIFEST文件，而不会对这些模型文件进行归档。`default`以*<model-name>.mar*格式创建模型存档。这是默认的归档格式。以此格式归档的模型将可以轻松托管在*TorchServe*上。 |
| `-f`,`--force` | 当指定`-f`或`--force`标志时，将覆盖具有与`--model-name`中提供的相同名称的现有*.mar*文件，该文件位于`--export-path`指定的路径中。 |
| `-v`, `--version` | 模型的版本。 |
| `-r`, `--requirements-file` | 指定包含要由*TorchServe*安装的模型特定Python包列表的*requirements.txt*文件的路径，以实现无缝模型服务。 |

我们可以将模型存档*.mar*文件保存在*/models*文件夹中。我们将使用这个作为我们的模型存储。接下来，让我们运行TorchServe Web服务器。

### 运行TorchServe

TorchServe包括一个从命令行运行的内置Web服务器。它将一个或多个PyTorch模型包装在一组REST API中，并提供控件以配置端口、主机和日志记录。以下命令启动Web服务器，其中所有模型都位于*/models*文件夹中的模型存储中：

```py

$ `torchserve``--model-store``/models``--start``--models``all`
```

完整的选项集显示在[表7-4](#table_torchserve_options)中。

表7-4\. TorchServe选项

| 选项 | 描述 |
| --- | --- |
| `--model-store` *+MODEL_STORE+* *+(mandatory)+* | 指定模型存储位置，可以加载模型 |
| `-h`, `--help` | 显示帮助消息并退出 |
| `-v`, `--version` | 返回TorchServe的版本 |
| `--start` | 启动模型服务器 |
| `--stop` | 停止模型服务器 |
| `--ts-config` *+TS_CONFIG+* | 指示TorchServe的配置文件 |
| `--models` *+MODEL_PATH1 MODEL_NAME=MODEL_PATH2… [MODEL_PATH1 MODEL_NAME=MODEL_PATH2… …]+* | 指示要使用*`[model_name=]model_location`*格式加载的模型；位置可以是HTTP URL、模型存档文件或包含模型存档文件的目录在`MODEL_STORE`中 |
| `--log-config` *+LOG_CONFIG+* | 指示TorchServe的log4j配置文件 |
| `--ncs`, `--no-config-snapshots` | 禁用快照功能 |

现在TorchServe Web服务器正在运行，您可以使用推理API发送数据并请求预测。

### 请求预测

您可以使用推理API传递数据并请求预测。推理API在端口8080上侦听，默认情况下仅从本地主机访问。要更改默认设置，请参阅[TorchServe文档](https://pytorch.org/serve/configuration.html)。要从服务器获取预测，我们使用推理API的`Service.Predictions` gRPC API，并通过REST调用到*/predictions/<model_name>*，如下面的命令行中使用`curl`所示： 

```py

$c`url``http://localhost:8080/predictions/vgg16``-T``hot_dog.jpg`
```

代码假设我们有一个图像文件*hot_dog.jpg.* JSON格式的响应看起来像这样：

```py
{
    "class": "n02175045 hot dog",
    "probability": 0.788482002828
}
```

您还可以使用推理API进行健康检查，使用以下请求：

```py

$ `curl``http://localhost:8080/ping`
```

如果服务器正在运行，响应将如下所示：

```py
{
  "health": "healthy!"
}
```

要查看推理API的完整列表，请使用以下命令：

```py

$ `curl``-X``OPTIONS``http://localhost:8080`
```

### 日志记录和监控

您可以使用Metrics API配置指标，并在部署时监视和记录模型的性能。Metrics API监听端口8082，默认情况下仅从本地主机访问，但在配置TorchServe服务器时可以更改默认设置。以下命令说明如何访问指标：

```py

$ `curl``http://127.0.0.1:8082/metrics`# HELP ts_inference_latency_microseconds #    Cumulative inference # TYPE ts_inference_latency_microseconds counter ts_inference_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",...ts_inference_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noop"...# HELP ts_inference_requests_total Total number of inference ... # TYPE ts_inference_requests_total counter ts_inference_requests_total{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",...ts_inference_requests_total{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noop"...# HELP ts_queue_latency_microseconds Cumulative queue duration ... # TYPE ts_queue_latency_microseconds counter ts_queue_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",...ts_queue_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noop"...
```

默认的指标端点返回Prometheus格式的指标。Prometheus是一个免费软件应用程序，用于事件监控和警报，它使用HTTP拉模型记录实时指标到时间序列数据库中。您可以使用`curl`请求查询指标，或者将Prometheus服务器指向端点并使用Grafana进行仪表板。有关更多详细信息，请参阅[Metrics API文档](https://pytorch.tips/serve-metrics)。

指标被记录到文件中。TorchServe还支持其他类型的服务器日志记录，包括访问日志和TorchServe日志。访问日志记录推理请求以及完成请求所需的时间。根据*properties*文件的定义，访问日志被收集在*<log_location>/access_log.log*文件中。TorchServe日志收集来自TorchServe及其后端工作人员的所有日志。

TorchServe支持超出默认设置的指标和日志记录功能。指标和日志可以以许多不同的方式进行配置。此外，您可以创建自定义日志。有关TorchServe的指标和日志自定义以及其他高级功能的更多信息，请参阅[TorchServe文档](https://pytorch.tips/torchserve)。

###### 注意

*NVIDIA Triton推理服务器*变得越来越受欢迎，也用于在生产环境中规模部署AI模型。尽管不是PyTorch项目的一部分，但在部署到NVIDIA GPU时，您可能希望考虑Triton推理服务器作为TorchServe的替代方案。

Triton推理服务器是开源软件，可以从本地存储、GCP或AWS S3加载模型。Triton支持在单个或多个GPU上运行多个模型，低延迟和共享内存，以及模型集成。Triton相对于TorchServe的一些可能优势包括：

+   Triton已经不再是测试版。

+   这是在NVIDIA硬件上进行推理的最快方式（常见）。

+   它可以使用`int4`量化。

+   您可以直接从PyTorch转换而无需ONNX。

作为Docker容器提供，Triton推理服务器还与Kubernetes集成，用于编排、指标和自动扩展。有关更多信息，请访问[NVIDIA Triton推理服务器文档](https://pytorch.tips/triton)。

## ONNX

如果您的平台不支持PyTorch，并且无法在部署中使用TorchScript/C++或TorchServe，那么您的部署平台可能支持Open Neural Network Exchange（ONNX）格式。ONNX格式定义了一组通用操作符和通用文件格式，以便深度学习工程师可以在各种框架、工具、运行时和编译器之间使用模型。

ONNX是由Facebook和Microsoft开发的，旨在允许PyTorch和其他框架（如Caffe2和Microsoft认知工具包（CTK））之间的模型互操作性。目前，ONNX由多家供应商的推理运行时支持，包括Cadence Systems、Habana、Intel AI、NVIDIA、Qualcomm、腾讯、Windows和Xilinx。

一个示例用例是在Xilinx FPGA设备上进行边缘部署。FPGA设备是可以使用特定逻辑编程的定制芯片。它们被边缘设备用于低延迟或高性能应用，如视频。如果您想将您的新创新模型部署到FPGA设备上，您首先需要将其转换为ONNX格式，然后使用Xilinx FPGA开发工具生成带有您模型实现的FPGA图像。

让我们看一个如何将模型导出为ONNX的示例，再次使用我们的VGG16模型。ONNX导出器可以使用追踪或脚本。我们在TorchScript的早期部分学习了关于追踪和脚本的内容。我们可以通过简单地提供模型和一个示例输入来使用追踪。以下代码显示了我们如何使用追踪将我们的VGG16模型导出为ONNX：

```py
model = vgg16(pretrained=True)
example_input = torch.rand(1, 3, 224, 224)
onnx_model = torch.onnx.export(model,
                               example_input,
                               "vgg16.onnx")
```

我们定义一个示例输入并调用`torch.onnx.export()`。生成的文件*vgg16.onnx*是一个包含我们导出的VGG16模型的网络结构和参数的二进制protobuf文件。

如果我们想要验证我们的模型是否正确转换为ONNX，我们可以使用ONNX检查器，如下所示：

```py
import onnx

model = onnx.load("vgg16.onnx")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)
```

此代码使用Python ONNX库加载模型，运行检查器，并打印出模型的可读版本。在运行代码之前，您可能需要安装ONNX库，使用`conda`或`pip`。

要了解更多关于转换为ONNX或在ONNX运行时中运行的信息，请查看PyTorch网站上的[ONNX教程](https://pytorch.tips/onnx-tutorial)。

除了TorchScript、TorchServe和ONNX之外，还正在开发更多工具来支持PyTorch模型部署。让我们考虑一些用于将模型部署到移动平台的工具。

## 移动库

Android和iPhone设备不断发展，并在其定制芯片组中添加对深度学习加速的本机支持。此外，由于需要减少延迟、保护隐私以及在应用程序中与深度学习模型无缝交互的增长需求，部署到移动设备变得更加复杂。这是因为移动运行时可能与开发人员使用的训练环境有显著不同，导致在移动部署过程中出现错误和挑战。

PyTorch Mobile解决了这些挑战，并提供了一个从训练到移动部署的端到端工作流程。PyTorch Mobile可用于iOS、Android和Linux，并提供用于移动应用程序所需的预处理和集成任务的API。基本工作流程如[图7-2](#fig_pytorch_mobile)所示。

![“PyTorch Mobile Workflow”](Images/ptpr_0702.png)

###### 图7-2\. PyTorch移动工作流程

您可以像通常在PyTorch中设计模型一样开始。然后，您可以对模型进行量化，以减少其复杂性，同时最小程度地降低性能。随后，您可以使用追踪或脚本转换为TorchScript，并使用`torch.utils`优化模型以适用于移动设备。接下来，保存您的模型并使用适当的移动库进行部署。Android使用Maven PyTorch库，iOS使用CocoPods与LibTorch pod。

###### 警告

PyTorch Mobile仍在开发中，可能会发生变化。

有关PyTorch Mobile的最新详细信息，请参考[PyTorch Mobile文档](https://pytorch.tips/mobile)。

现在我们已经探讨了一些PyTorch工具，用于部署我们的模型，让我们看一些参考应用程序和代码，用于部署到云端和边缘设备。首先，我将向您展示如何使用Flask构建开发Web服务器。

# 部署到Flask应用程序

在部署到全面生产之前，您可能希望将模型部署到开发Web服务器。这使您能够将深度学习算法与其他系统集成，并快速构建原型以演示您的新模型。使用Python使用Flask构建开发服务器的最简单方法之一。

Flask是一个用Python编写的简单微型Web框架。它被称为“微”框架，因为它不包括数据库抽象层、表单验证、上传处理、各种身份验证技术或其他可能由其他库提供的内容。我们不会在本书中深入讨论Flask，但我会向您展示如何使用Flask在Python中部署您的模型。

我们还将公开一个REST API，以便其他应用程序可以传入数据并接收预测。在以下示例中，我们将部署我们预训练的VGG16模型并对图像进行分类。首先，我们将定义我们的API端点、请求类型和响应类型。我们的API端点将位于*/predict*，接受POST请求（包括图像文件）。响应将以JSON格式返回，并包含来自ImageNet数据集的`class_id`和`class_name`。

让我们创建我们的主要Flask文件，称为*app.py*。首先我们将导入所需的包：

```py
import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
```

我们将使用`io`将字节转换为图像，使用`json`处理JSON格式数据。我们将使用`torchvision`创建我们的VGG16模型，并将图像数据转换为适合我们模型的格式。最后，我们导入`Flask`、`jsonnify`和`request`来处理API请求和响应。

在创建我们的Web服务器之前，让我们定义一个`get_prediction()`函数，该函数读取图像数据，预处理它，将其传递到我们的模型，并返回图像类别：

```py
import json
imagenet_class_index = json.load(
    open("./imagenet_class_index.json"))

model = models.vgg16(pretrained=True)

image_transforms = transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(
          [0.485, 0.456, 0.406],
          [0.229, 0.224, 0.225])])

def get_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    tensor = image_transforms(image)
    outputs = model(tensor)
    _, y = outputs.max(1)
    predicted_idx = str(y.item())
    return imagenet_class_index[predicted_idx]
```

由于我们的模型将返回一个表示类别的数字，我们需要一个查找表将此数字转换为类别名称。我们通过读取JSON转换文件创建一个名为`imagenet_class_index`的字典。然后，我们实例化我们的VGG16模型，并定义我们的图像转换，以预处理一个PIL图像，将其调整大小、中心裁剪、转换为张量并进行归一化。在将图像发送到我们的模型之前，这些步骤是必需的。

我们的`get_prediction()`函数基于接收到的字节创建一个PIL图像对象，并应用所需的图像转换以创建输入张量。接下来，我们执行前向传递（或模型推断），并找到具有最高概率的类别`y`。最后，我们使用输出类值查找类名。

现在我们有了预处理图像、通过我们的模型传递图像并返回预测类别的代码，我们可以创建我们的Flask Web服务器和端点，并部署我们的模型，如下所示的代码：

```py
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
  img_bytes = file.read()
  class_id, class_name = \
    get_prediction(image_bytes=img_bytes)
  return jsonify({'class_id': class_id,
                 'class_name': class_name})
```

我们的Web服务器对象称为`app`。我们已经创建了它，但它还没有运行。我们将我们的端点设置为*/predict*，并配置它以处理POST请求。当Web服务器接收到POST请求时，它将执行`predict()`函数，读取图像，获取预测，并以JSON格式返回图像类别。

就是这样！现在我们只需要添加以下代码，以便在执行*app.py*时运行Web服务器：

```py
if __name__ == '__main__':
    app.run()
```

要测试我们的Web服务器，可以按照以下方式运行它：

```py
>>> FLASK_ENV=development FLASK_APP=app.py flask run
```

我们可以使用一个简单的Python文件和`requests`库发送图像：

```py
import requests

resp = requests.post(
    "http://localhost:5000/predict",
    files={"file": open('cat.jpg','rb')})

print(resp.json())

>>> {"class_id": "n02124075", "class_name": "Egyptian_cat"}
```

在这个例子中，我们在本地机器的端口5000（`localhost:5000`）上运行一个Web服务器。您可能希望在Google Colab中运行开发Web服务器，以利用云GPU。接下来我将向您展示如何做到这一点。

# Colab Flask应用程序

也许您一直在Colab中开发您的PyTorch模型，以利用其快速开发或其GPU。Colab提供了一个虚拟机（VM），它将其`localhost`路由到我们机器的本地主机。要将其暴露到公共URL，我们可以使用一个名为`ngrok`的库。

首先在Colab中安装`ngrok`：

```py
!pip install flask-ngrok
```

要使用`ngrok`运行我们的Flask应用程序，我们只需要添加两行代码，如下注释所示：

```py
fromflask_ngrokimportrun_with_ngrok![1](Images/1.png)@app.route("/")defhome():return"<h1>Running Flask on Google Colab!</h1>"app.run()app=Flask(__name__)run_with_ngrok(app)![2](Images/2.png)@app.route('/predict',methods=['POST'])defpredict():ifrequest.method=='POST':file=request.files['file']img_bytes=file.read()class_id,class_name=\
get_prediction(image_bytes=img_bytes)returnjsonify({'class_id':class_id,'class_name':class_name})app.run()![3](Images/3.png)
```

[![1](Images/1.png)](#co_deploying_pytorch_to_production_CO1-1)

导入`ngrok`库。

[![2](Images/2.png)](#co_deploying_pytorch_to_production_CO1-2)

当应用程序运行时启动`ngrok`。

[![3](Images/3.png)](#co_deploying_pytorch_to_production_CO1-3)

由于我们在Colab中运行，我们不需要检查`main`。

我已经省略了其他导入和`get_prediction()`函数，因为它们没有改变。现在您可以在Colab中运行开发Web服务器，以便更快地进行原型设计。`ngrok`库为在Colab中运行的服务器提供了安全的URL；当运行Flask应用程序时，您将在Colab笔记本输出中找到URL。例如，以下输出显示URL为[*http://c0c97117ba27.ngrok.io*](http://c0c97117ba27.ngrok.io)：

```py

 * Serving Flask app "__main__" (lazy loading)
 * Environment: production
   WARNING: This is a development server.
     Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Running on http://c0c97117ba27.ngrok.io
 * Traffic stats available on http://127.0.0.1:4040
127.0.0.1 - - [08/Dec/2020 20:46:05] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [08/Dec/2020 20:46:05]
  "GET /favicon.ico HTTP/1.1" 404 -
127.0.0.1 - - [08/Dec/2020 20:46:06] "GET / HTTP/1.1" 200 -

```

再次，您可以发送带有图像的POST请求来测试Web服务器。您可以在本地或另一个Colab笔记本中运行以下代码：

```py
import requests

resp = requests.post(
      "http://c0c97117ba27.ngrok.io/predict",
      files={"file": open('cat.jpg','rb')})

print(resp.json())

# out :
# {"class_id": "n02124075",
#  "class_name": "Egyptian_cat"}
```

请注意URL已更改。在Flask应用程序中部署您的模型是快速测试和使用REST API的`get_prediction()`函数的好方法。但是，在这里我们的Flask应用程序用作开发Web服务器，而不是用于生产部署。在大规模部署模型时，您将需要解决模型管理、A/B测试、监控、日志记录和其他任务等问题，以确保您的模型服务器正常工作。要在大规模生产环境中部署，我们将使用TorchServe。

# 使用TorchServe部署到云端

在此示例中，我们将将VGG16图像分类器部署到生产环境。假设我们的公司制作了一个软件工具，将根据图像中出现的对象对零售产品图像集进行分类。该公司正在迅速发展，现在支持数百万家每天使用该工具的小型企业。

作为机器学习工程团队的一部分，您需要将模型部署到生产环境，并提供一个简单的REST API，软件工具将使用该API对其图像进行分类。因为我们希望尽快部署某些东西，所以我们将在AWS EC2实例中使用Docker容器。

# 使用Docker快速入门

TorchServe提供了脚本，可以基于各种平台和选项创建Docker镜像。运行Docker容器可以消除重新安装运行TorchServe所需的所有依赖项的需要。此外，我们可以通过使用Kubernetes旋转多个Docker容器来扩展我们的模型推理。首先，我们必须根据我们在EC2实例上拥有的资源创建Docker镜像。

第一步是克隆TorchServe存储库并导航到*Docker*文件夹，使用以下命令：

```py

$ git clone https://github.com/pytorch/serve.git
cd serve/docker

```

接下来，我们需要将VGG16模型归档添加到Docker镜像中。我们通过向下载存档模型文件并将其保存在*/home/model-server/*目录中的Dockerfile添加以下行来实现这一点：

```py

$ curl -o /home/model-server/vgg16.pth \
    https://download.pytorch.org/models/vgg16.pth

```

现在我们可以运行*build_image.sh*脚本，创建一个安装了公共二进制文件的Docker镜像。由于我们在带有GPU的EC2实例上运行，我们将使用`-g`标志，如下所示：

```py

$ ./build_image.sh -g

```

您可以运行**`./build_image.sh -h`**查看其他选项。

一旦我们创建了Docker镜像，我们可以使用以下命令运行容器：

```py

$ docker run --rm -it --gpus '"device=1"' \
    -p 8080:8080 -p 8081:8081 -p 8082:8082 \
    -p 7070:7070 -p 7071:7071 \
    pytorch/torchserve:latest-gpu

```

这个命令将启动容器，将8080/81/82和7070/71端口暴露给外部世界的本地主机。它使用一个带有最新CUDA版本的GPU。

现在我们的TorchServe Docker容器正在运行。我们公司的软件工具可以通过将图像文件发送到*ourip.com/predict*来发送推理请求，并可以通过JSON接收图像分类。

有关在Docker中运行TorchServe的更多详细信息，请参阅[TorchServe Docker文档](https://pytorch.tips/torchserve-docker)。要了解有关TorchServe的更多信息，请访问[TorchServe存储库](https://pytorch.tips/torchserve-github)。

现在，您可以使用Flask进行开发或使用TorchServe进行生产，将模型部署到本地计算机和云服务器。这对于原型设计和通过REST API与其他应用程序集成非常有用。接下来，您将扩展部署能力，将模型部署到云之外：在下一节中，我们将探讨如何将模型部署到移动设备和其他边缘设备。

# 部署到移动和边缘

边缘设备通常是与用户或环境直接交互并在设备上直接运行机器学习计算的（通常是小型）硬件系统，而不是在云中的集中服务器上运行。一些边缘设备的例子包括手机和平板电脑，智能手表和心率监测器等可穿戴设备，以及工业传感器和家用恒温器等其他物联网设备。有一个越来越大的需求在边缘设备上运行深度学习算法，以保护隐私，减少数据传输，最小化延迟，并支持实时的新交互式用例。

首先我们将探讨如何在iOS和Android移动设备上部署PyTorch模型，然后我们将涵盖其他边缘设备。PyTorch对边缘部署的支持有限但在不断增长。这些部分将提供一些参考代码，帮助您开始使用PyTorch Mobile。

## iOS

根据苹果公司的数据，截至2021年1月，全球有超过16.5亿活跃的iOS设备。随着每个新模型和定制处理单元的推出，对机器学习硬件加速的支持不断增长。学习如何将PyTorch模型部署到iOS为您打开了许多机会，可以基于深度学习创建iOS应用程序。

要将模型部署到iOS设备，您需要学习如何使用Xcode等开发工具创建iOS应用程序。我们不会在本书中涵盖iOS开发，但您可以在[PyTorch iOS示例应用GitHub存储库](https://pytorch.tips/ios-demo)中找到“Hello, World”程序和示例代码，以帮助您构建您的应用程序。

让我们描述将我们的VGG16网络部署到iOS应用程序的工作流程。iOS将使用PyTorch C++ API与我们的模型进行接口，因此我们需要首先将我们的模型转换并保存为TorchScript。然后我们将在Objective-C中包装C++函数，以便iOS Swift代码可以访问API。我们将使用Swift加载和预处理图像，然后将图像数据传入我们的模型以预测其类别。

首先我们将使用追踪将我们的VGG16模型转换为TorchScript并保存为*model.pt*，如下面的代码所示：

```py
importtorchimporttorchvisionfromtorch.utils.mobile_optimizer \ importoptimize_for_mobilemodel=torchvision.models.vgg16(pretrained=True)model.eval()example=torch.rand(1,3,224,224)![1](Images/1.png)traced_script_module=\
torch.jit.trace(model,example)![2](Images/2.png)torchscript_model_optimized=\
optimize_for_mobile(traced_script_module)![3](Images/3.png)torchscript_model_optimized.save("model.pt")![4](Images/4.png)
```

[![1](Images/1.png)](#co_deploying_pytorch_to_production_CO2-1)

使用随机数据定义`example`。

[![2](Images/2.png)](#co_deploying_pytorch_to_production_CO2-2)

将`model`转换为TorchScript。

[![3](Images/3.png)](#co_deploying_pytorch_to_production_CO2-3)

优化代码的新步骤。

[![4](Images/4.png)](#co_deploying_pytorch_to_production_CO2-4)

保存模型。

如前所述，使用追踪需要定义一个示例输入，我们使用随机数据来做到这一点。然后我们使用`torch.jit.trace()`将模型转换为TorchScript。然后我们添加一个新步骤，使用`torch.utils.mobile_optimizer`包为移动平台优化TorchScript代码。最后，我们将模型保存到名为*model.pt*的文件中。

现在我们需要编写我们的Swift iOS应用程序。我们的iOS应用程序将使用PyTorch C++库，我们可以通过CocoaPods安装如下：

```py

$ pod install

```

然后我们需要编写一些Swift代码来加载一个示例图像。您可以在将来通过访问设备上的相机或照片来改进这一点，但现在我们将保持简单：

```py
let image = UIImage(named: "image.jpg")! \
  imageView.image = image

let resizedImage = image.resized(
  to: CGSize(width: 224, height: 224))

guard var pixelBuffer = resizedImage.normalized()
else return
```

在这里，我们将图像调整大小为224×224像素，并运行一个函数来规范化图像数据。

接下来，我们将加载并实例化我们的模型到我们的iOS应用程序中，如下面的代码所示：

```py
private lazy var module: TorchModule = {
    if let filePath = Bundle.main.path(
      forResource: "model", ofType: "pt"),

        let module = TorchModule(
                 fileAtPath: filePath) {
          return module
    } else {
        fatalError("Can't find the model file!")
    }
}()
```

iOS是用Swift编写的，Swift无法与C++进行接口，因此我们需要使用一个Objective-C类`TorchModule`作为`torch::jit::script::Module`的包装器。

现在我们的模型已加载，我们可以通过将预处理的图像数据传入我们的模型并运行预测来预测图像的类别，如下面的代码所示：

```py
guard let outputs = module.predict(image:
  UnsafeMutableRawPointer(&pixelBuffer))
else {
    return
}
```

在内部，`predict()` Objective-C包装器调用C++ `forward()`函数如下：

```py
at::Tensor tensor = torch::from_blob(
  imageBuffer, {1, 3, 224, 224}, at::kFloat);

torch::autograd::AutoGradMode guard(false);
auto outputTensor = _impl.forward(
  {tensor}).toTensor();
float* floatBuffer =
  outputTensor.data_ptr<float>();
```

当您运行示例应用程序时，您应该看到类似于[图7-3](#fig_ios_example)的输出，用于示例图像文件。

这个图像分类示例只是对为iOS设备编码能力的一个小表示。对于更高级的用例，您仍然可以遵循相同的流程：转换并保存为TorchScript，创建一个Objective-C包装器，预处理输入，并调用您的`predict()`函数。接下来，我们将为部署PyTorch到Android移动设备遵循类似的流程。

![“iOS示例”](Images/ptpr_0703.png)

###### 图7-3\. iOS示例

## Android

Android移动设备在全球范围内也被广泛使用，据估计，2021年初移动设备的市场份额超过70%。这意味着也有巨大的机会将PyTorch模型部署到Android设备上。

Android使用PyTorch Android API，您需要安装Android开发工具来构建示例应用程序。使用Android Studio，您将能够安装Android本机开发工具包（NDK）和软件开发工具包（SDK）。我们不会在本书中涵盖Android开发，但您可以在[PyTorch Android示例GitHub存储库](https://pytorch.tips/android-demo)中找到“Hello, World”程序和示例代码，以帮助您构建您的应用程序。

在Android设备上部署PyTorch模型的工作流程与我们用于iOS的过程非常相似。我们仍然需要将我们的模型转换为TorchScript以便与PyTorch Android API一起使用。然而，由于API本身支持加载和运行我们的TorchScript模型，我们无需像在iOS中那样将其包装在C++代码中。相反，我们将使用Java编写一个Android应用程序，该应用程序加载和预处理图像文件，将其传递给我们的模型进行推理，并返回结果。

让我们将VGG16模型部署到Android。首先，我们将模型转换为TorchScript，就像我们为iOS所做的那样，如下面的代码所示：

```py
import torch
import torchvision
from torch.utils.mobile_optimizer \
  import optimize_for_mobile

model = torchvision.models.vgg16(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)

traced_script_module = \
  torch.jit.trace(model, example)
torchscript_model_optimized = \
  optimize_for_mobile(traced_script_module)
torchscript_model_optimized.save("model.pt")
```

我们使用`torch.jit.trace()`进行跟踪将模型转换为TorchScript。然后，我们添加一个新步骤，使用`torch.utils.mobile_optimizer`包为移动平台优化TorchScript代码。最后，我们将模型保存到名为*model.pt*的文件中。

接下来，我们使用Java创建我们的Android应用程序。我们通过将以下代码添加到*build.gradle*将PyTorch Android API添加到我们的应用程序作为Gradle依赖项：

```py
repositories {
  jcenter()
}

dependencies {
  implementation
    'org.pytorch:pytorch_android:1.4.0'
  implementation
    'org.pytorch:pytorch_android_torchvision:1.4.0'
}
```

接下来，我们编写我们的Android应用程序。我们首先加载图像并使用以下代码对其进行预处理：

```py
Bitmap bitmap = \
  BitmapFactory.decodeStream(
    getAssets().open("image.jpg"));

Tensor inputTensor = \
  TensorImageUtils.bitmapToFloat32Tensor(
    bitmap,
    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
    TensorImageUtils.TORCHVISION_NORM_STD_RGB);
```

现在我们有了我们的图像，我们可以预测它的类别，但首先我们必须加载我们的模型，如下所示：

```py
Module module = Module.load(
  assetFilePath(this, "model.pt"));
```

然后我们可以运行推理来预测图像的类别，并使用以下代码处理结果：

```py
Tensor outputTensor = module.forward(
  IValue.from(inputTensor)).toTensor();
float[] scores = \
  outputTensor.getDataAsFloatArray();

float maxScore = -Float.MAX_VALUE;
int maxScoreIdx = -1;
for (int i = 0; i < scores.length; i++) {
  if (scores[i] > maxScore) {
    maxScore = scores[i];
    maxScoreIdx = i;
  }
}
String className = \
  ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];
```

这个工作流程可以用于更高级的用例。您可以使用设备上的相机或照片或其他Android传感器来创建更复杂的应用程序。有关更多PyTorch Android演示应用程序，请访问[PyTorch Android演示应用程序GitHub存储库](https://pytorch.tips/android-demo-repo)。

## 其他边缘设备

运行iOS或Android的移动设备代表一种边缘设备，但还有许多可以执行深度学习算法的设备。边缘设备通常使用定制硬件构建用于特定应用。其他边缘设备的示例包括传感器、视频设备、医疗监视器、软件定义无线电、恒温器、农业机械和制造传感器以检测缺陷。

大多数边缘设备包括计算机处理器、GPU、FPGA或其他定制ASIC计算机芯片，能够运行深度学习模型。那么，如何将PyTorch模型部署到这些边缘设备呢？这取决于设备上使用了哪些处理组件。让我们探讨一些常用芯片的想法：

CPU

如果您的边缘设备使用CPU，如英特尔或AMD处理器，PyTorch可以在Python和C++中使用TorchScript和C++前端API部署。移动和边缘CPU芯片组通常经过优化以最小化功耗，并且边缘设备上的内存可能更有限。在部署之前通过修剪或量化优化您的模型以最小化运行推理所需的功耗和内存可能是值得的。

ARMs

ARM处理器是一类具有简化指令集的计算机处理器。它们通常以比英特尔或AMD CPU更低的功耗和时钟速度运行，并可以包含在片上系统（SoCs）中。除了处理器，SoCs芯片通常还包括其他电子设备，如可编程FPGA逻辑或GPU。目前正在开发在ARM设备上运行PyTorch的Linux。

微控制器

微控制器是通常用于非常简单控制任务的非常有限的处理器。一些流行的微控制器包括Arduino和Beaglebone处理器。由于可用资源有限，对微控制器的支持有限。

GPUs

边缘设备可能包括GPU芯片。NVIDIA GPU是最广泛支持的GPU，但其他公司（如AMD和英特尔）也制造GPU芯片。NVIDIA在其GPU开发套件中支持PyTorch，包括其Jetson Nano、Xavier和NX板。

FPGAs

PyTorch模型可以部署到许多FPGA设备，包括赛灵思（最近被AMD收购）和英特尔FPGA设备系列。这两个平台都不支持直接的PyTorch部署；但是它们支持ONNX格式。典型的方法是将PyTorch模型转换为ONNX，并使用FPGA开发工具从ONNX模型创建FPGA逻辑。

TPUs

谷歌的TPU芯片也正在部署到边缘设备上。PyTorch通过XLA库支持，如[“在TPU上的PyTorch”](ch06.xhtml#onatpu)中所述。将模型部署到使用TPU的边缘设备可以使您使用XLA库进行推理。

ASICs

许多公司正在开发自己的定制芯片或ASIC，以高度优化和高效的方式实现模型设计。部署PyTorch模型的能力将严重依赖于定制ASIC芯片设计和开发工具所支持的功能。在某些情况下，如果ASIC支持，您可以使用PyTorch/XLA库。

当部署PyTorch模型到边缘设备时，请考虑系统上可用的处理组件。根据可用的芯片，研究利用C++前端API、利用TorchScript、将模型转换为ONNX格式，或访问PyTorch XLA库以部署模型的选项。

在本章中，您学习了如何使用标准的Python API、TorchScript/C++、TorchServe、ONNX和PyTorch移动库来部署您的模型进行推理。本章还提供了参考代码，以在本地开发服务器或云中的生产环境中使用Flask和TorchServe，以及在iOS和Android设备上部署PyTorch模型。

PyTorch支持一个庞大、活跃的有用工具生态系统，用于模型开发和部署。我们将在下一章中探讨这个生态系统，该章还提供了一些最受欢迎的PyTorch工具的参考代码。
