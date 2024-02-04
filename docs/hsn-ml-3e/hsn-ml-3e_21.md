# 第19章。在规模上训练和部署TensorFlow模型

一旦您拥有一个能够做出惊人预测的美丽模型，您会怎么处理呢？嗯，您需要将其投入生产！这可能只是在一批数据上运行模型，也许编写一个每晚运行该模型的脚本。然而，通常情况下会更加复杂。您的基础设施的各个部分可能需要在实时数据上使用该模型，这种情况下，您可能会希望将模型封装在一个Web服务中：这样，您的基础设施的任何部分都可以随时使用简单的REST API（或其他协议）查询模型，正如我们在[第2章](ch02.html#project_chapter)中讨论的那样。但随着时间的推移，您需要定期使用新数据对模型进行重新训练，并将更新后的版本推送到生产环境。您必须处理模型版本控制，优雅地从一个模型过渡到另一个模型，可能在出现问题时回滚到上一个模型，并可能并行运行多个不同的模型来执行*A/B实验*。如果您的产品变得成功，您的服务可能会开始每秒收到大量查询（QPS），并且必须扩展以支持负载。如您将在本章中看到的，一个很好的扩展服务的解决方案是使用TF Serving，无论是在您自己的硬件基础设施上还是通过诸如Google Vertex AI之类的云服务。它将有效地为您提供模型服务，处理优雅的模型过渡等。如果您使用云平台，您还将获得许多额外功能，例如强大的监控工具。

此外，如果你有大量的训练数据和计算密集型模型，那么训练时间可能会变得过长。如果你的产品需要快速适应变化，那么长时间的训练可能会成为一个阻碍因素（例如，想象一下一个新闻推荐系统在推广上周的新闻）。更重要的是，长时间的训练会阻止你尝试新想法。在机器学习（以及许多其他领域），很难事先知道哪些想法会奏效，因此你应该尽可能快地尝试尽可能多的想法。加快训练的一种方法是使用硬件加速器，如GPU或TPU。为了更快地训练，你可以在多台配备多个硬件加速器的机器上训练模型。TensorFlow的简单而强大的分布策略API使这一切变得容易，你将会看到。

在这一章中，我们将学习如何部署模型，首先使用TF Serving，然后使用Vertex AI。我们还将简要介绍如何将模型部署到移动应用程序、嵌入式设备和Web应用程序。然后我们将讨论如何使用GPU加速计算，以及如何使用分布策略API在多个设备和服务器上训练模型。最后，我们将探讨如何使用Vertex AI 在规模上训练模型并微调其超参数。这是很多要讨论的话题，让我们开始吧！

# 为 TensorFlow 模型提供服务

一旦您训练了一个TensorFlow模型，您可以在任何Python代码中轻松地使用它：如果它是一个Keras模型，只需调用它的`predict()`方法！但随着基础设施的增长，会出现一个更好的选择，即将您的模型封装在一个小型服务中，其唯一作用是进行预测，并让基础设施的其余部分查询它（例如，通过REST或gRPC API）。这样可以将您的模型与基础设施的其余部分解耦，从而可以轻松地切换模型版本或根据需要扩展服务（独立于您的基础设施的其余部分），执行A/B实验，并确保所有软件组件依赖于相同的模型版本。这也简化了测试和开发等工作。您可以使用任何您想要的技术（例如，使用Flask库）创建自己的微服务，但为什么要重新发明轮子，当您可以直接使用TF Serving呢？

## 使用TensorFlow Serving

TF Serving是一个非常高效、经过实战验证的模型服务器，用C++编写。它可以承受高负载，为您的模型提供多个版本，并监视模型存储库以自动部署最新版本，等等（参见[图19-1](#tf_serving_diagram)）。

![mls3 1901](assets/mls3_1901.png)

###### 图19-1。TF Serving可以为多个模型提供服务，并自动部署每个模型的最新版本。

假设您已经使用Keras训练了一个MNIST模型，并且希望将其部署到TF Serving。您需要做的第一件事是将此模型导出为SavedModel格式，该格式在[第10章](ch10.html#ann_chapter)中介绍。

### 导出SavedModels

您已经知道如何保存模型：只需调用`model.save()`。现在要对模型进行版本控制，您只需要为每个模型版本创建一个子目录。很简单！

```py
from pathlib import Path
import tensorflow as tf

X_train, X_valid, X_test = [...]  # load and split the MNIST dataset
model = [...]  # build & train an MNIST model (also handles image preprocessing)

model_name = "my_mnist_model"
model_version = "0001"
model_path = Path(model_name) / model_version
model.save(model_path, save_format="tf")
```

通常最好将所有预处理层包含在最终导出的模型中，这样一旦部署到生产环境中，模型就可以以其自然形式摄取数据。这样可以避免在使用模型的应用程序中单独处理预处理工作。将预处理步骤捆绑在模型中也使得以后更新它们更加简单，并限制了模型与所需预处理步骤之间不匹配的风险。

###### 警告

由于SavedModel保存了计算图，因此它只能用于基于纯粹的TensorFlow操作的模型，不包括`tf.py_function()`操作，该操作包装任意的Python代码。

TensorFlow带有一个小的`saved_model_cli`命令行界面，用于检查SavedModels。让我们使用它来检查我们导出的模型：

```py
$ saved_model_cli show --dir my_mnist_model/0001
The given SavedModel contains the following tag-sets:
'serve'

```

这个输出是什么意思？嗯，一个SavedModel包含一个或多个*metagraphs*。一个metagraph是一个计算图加上一些函数签名定义，包括它们的输入和输出名称、类型和形状。每个metagraph都由一组标签标识。例如，您可能希望有一个包含完整计算图的metagraph，包括训练操作：您通常会将这个标记为`"train"`。您可能有另一个包含经过修剪的计算图的metagraph，只包含预测操作，包括一些特定于GPU的操作：这个可能被标记为`"serve", "gpu"`。您可能还想要其他metagraphs。这可以使用TensorFlow的低级[SavedModel API](https://homl.info/savedmodel)来完成。然而，当您使用Keras模型的`save()`方法保存模型时，它会保存一个标记为`"serve"`的单个metagraph。让我们检查一下这个`"serve"`标签集：

```py
$ saved_model_cli show --dir 0001/my_mnist_model --tag_set serve
The given SavedModel MetaGraphDef contains SignatureDefs with these keys:
SignatureDef key: "__saved_model_init_op"
SignatureDef key: "serving_default"

```

这个元图包含两个签名定义：一个名为`"__saved_model_init_op"`的初始化函数，您不需要担心，以及一个名为`"serving_default"`的默认服务函数。当保存一个Keras模型时，默认的服务函数是模型的`call()`方法，用于进行预测，这一点您已经知道了。让我们更详细地了解这个服务函数：

```py
$ saved_model_cli show --dir 0001/my_mnist_model --tag_set serve \
                       --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
 inputs['flatten_input'] tensor_info:
 dtype: DT_UINT8
 shape: (-1, 28, 28)
 name: serving_default_flatten_input:0
The given SavedModel SignatureDef contains the following output(s):
 outputs['dense_1'] tensor_info:
 dtype: DT_FLOAT
 shape: (-1, 10)
 name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict

```

请注意，函数的输入被命名为`"flatten_input"`，输出被命名为`"dense_1"`。这些对应于Keras模型的输入和输出层名称。您还可以看到输入和输出数据的类型和形状。看起来不错！

现在您已经有了一个SavedModel，下一步是安装TF Serving。

### 安装和启动TensorFlow Serving

有许多安装TF Serving的方法：使用系统的软件包管理器，使用Docker镜像，从源代码安装等。由于Colab运行在Ubuntu上，我们可以像这样使用Ubuntu的`apt`软件包管理器：

```py
url = "https://storage.googleapis.com/tensorflow-serving-apt"
src = "stable tensorflow-model-server tensorflow-model-server-universal"
!echo 'deb {url} {src}' > /etc/apt/sources.list.d/tensorflow-serving.list
!curl '{url}/tensorflow-serving.release.pub.gpg' | apt-key add -
!apt update -q && apt-get install -y tensorflow-model-server
%pip install -q -U tensorflow-serving-api
```

这段代码首先将TensorFlow的软件包存储库添加到Ubuntu的软件包源列表中。然后它下载TensorFlow的公共GPG密钥，并将其添加到软件包管理器的密钥列表中，以便验证TensorFlow的软件包签名。接下来，它使用`apt`来安装`tensorflow-model-server`软件包。最后，它安装`tensorflow-serving-api`库，这是我们与服务器通信所需的库。

现在我们想要启动服务器。该命令将需要基本模型目录的绝对路径（即`my_mnist_model`的路径，而不是`0001`），所以让我们将其保存到`MODEL_DIR`环境变量中：

```py
import os

os.environ["MODEL_DIR"] = str(model_path.parent.absolute())
```

然后我们可以启动服务器：

```py
%%bash --bg
tensorflow_model_server \
     --port=8500 \
     --rest_api_port=8501 \
     --model_name=my_mnist_model \
     --model_base_path="${MODEL_DIR}" >my_server.log 2>&1
```

在Jupyter或Colab中，`%%bash --bg`魔术命令将单元格作为bash脚本执行，在后台运行。`>my_server.log 2>&1`部分将标准输出和标准错误重定向到*my_server.log*文件。就是这样！TF Serving现在在后台运行，其日志保存在*my_server.log*中。它加载了我们的MNIST模型（版本1），现在正在分别等待gRPC和REST请求，端口分别为8500和8501。

现在服务器已经启动运行，让我们首先使用REST API，然后使用gRPC API进行查询。

### 通过REST API查询TF Serving

让我们从创建查询开始。它必须包含您想要调用的函数签名的名称，当然还有输入数据。由于请求必须使用JSON格式，我们必须将输入图像从NumPy数组转换为Python列表：

```py
import json

X_new = X_test[:3]  # pretend we have 3 new digit images to classify
request_json = json.dumps({
    "signature_name": "serving_default",
    "instances": X_new.tolist(),
})
```

请注意，JSON格式是100%基于文本的。请求字符串如下所示：

```py
>>> request_json
'{"signature_name": "serving_default", "instances": [[[0, 0, 0, 0, ... ]]]}'
```

现在让我们通过HTTP POST请求将这个请求发送到TF Serving。这可以使用`requests`库来完成（它不是Python标准库的一部分，但在Colab上是预安装的）：

```py
import requests

server_url = "http://localhost:8501/v1/models/my_mnist_model:predict"
response = requests.post(server_url, data=request_json)
response.raise_for_status()  # raise an exception in case of error
response = response.json()
```

如果一切顺利，响应应该是一个包含单个`"predictions"`键的字典。相应的值是预测列表。这个列表是一个Python列表，所以让我们将其转换为NumPy数组，并将其中包含的浮点数四舍五入到第二位小数：

```py
>>> import numpy as np
>>> y_proba = np.array(response["predictions"])
>>> y_proba.round(2)
array([[0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 1\.  , 0\.  , 0\.  ],
 [0\.  , 0\.  , 0.99, 0.01, 0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0\.  ],
 [0\.  , 0.97, 0.01, 0\.  , 0\.  , 0\.  , 0\.  , 0.01, 0\.  , 0\.  ]])
```

万岁，我们有了预测！模型几乎100%确信第一张图片是7，99%确信第二张图片是2，97%确信第三张图片是1。这是正确的。

REST API 简单易用，当输入和输出数据不太大时效果很好。此外，几乎任何客户端应用程序都可以在没有额外依赖的情况下进行 REST 查询，而其他协议并不总是那么容易获得。然而，它基于 JSON，这是基于文本且相当冗长的。例如，我们不得不将 NumPy 数组转换为 Python 列表，每个浮点数最终都表示为一个字符串。这非常低效，无论是在序列化/反序列化时间方面——我们必须将所有浮点数转换为字符串然后再转回来——还是在有效载荷大小方面：许多浮点数最终使用超过 15 个字符来表示，这相当于 32 位浮点数超过 120 位！这将导致在传输大型 NumPy 数组时出现高延迟和带宽使用。因此，让我们看看如何改用 gRPC。

###### 提示

在传输大量数据或延迟重要时，最好使用gRPC API，如果客户端支持的话，因为它使用紧凑的二进制格式和基于HTTP/2 framing的高效通信协议。

### 通过gRPC API查询TF Serving

gRPC API期望一个序列化的`PredictRequest`协议缓冲区作为输入，并输出一个序列化的`PredictResponse`协议缓冲区。这些protobufs是`tensorflow-serving-api`库的一部分，我们之前安装过。首先，让我们创建请求：

```py
from tensorflow_serving.apis.predict_pb2 import PredictRequest

request = PredictRequest()
request.model_spec.name = model_name
request.model_spec.signature_name = "serving_default"
input_name = model.input_names[0]  # == "flatten_input"
request.inputs[input_name].CopyFrom(tf.make_tensor_proto(X_new))
```

这段代码创建了一个`PredictRequest`协议缓冲区，并填充了必需的字段，包括模型名称（之前定义的），我们想要调用的函数的签名名称，最后是输入数据，以`Tensor`协议缓冲区的形式。`tf.make_tensor_proto()`函数根据给定的张量或NumPy数组创建一个`Tensor`协议缓冲区，这里是`X_new`。

接下来，我们将向服务器发送请求并获取其响应。为此，我们将需要`grpcio`库，该库已预先安装在Colab中：

```py
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel('localhost:8500')
predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
response = predict_service.Predict(request, timeout=10.0)
```

代码非常简单：在导入之后，我们在TCP端口8500上创建一个到*localhost*的gRPC通信通道，然后我们在该通道上创建一个gRPC服务，并使用它发送一个带有10秒超时的请求。请注意，调用是同步的：它将阻塞，直到收到响应或超时期限到期。在此示例中，通道是不安全的（没有加密，没有身份验证），但gRPC和TF Serving也支持通过SSL/TLS的安全通道。

接下来，让我们将`PredictResponse`协议缓冲区转换为张量：

```py
output_name = model.output_names[0]  # == "dense_1"
outputs_proto = response.outputs[output_name]
y_proba = tf.make_ndarray(outputs_proto)
```

如果您运行此代码并打印`y_proba.round(2)`，您将获得与之前完全相同的估计类概率。这就是全部内容：只需几行代码，您现在就可以远程访问您的TensorFlow模型，使用REST或gRPC。

### 部署新的模型版本

现在让我们创建一个新的模型版本并导出一个SavedModel，这次导出到*my_mnist_model/0002*目录：

```py
model = [...]  # build and train a new MNIST model version

model_version = "0002"
model_path = Path(model_name) / model_version
model.save(model_path, save_format="tf")
```

在固定的时间间隔（延迟可配置），TF Serving会检查模型目录是否有新的模型版本。如果找到一个新版本，它会自动优雅地处理过渡：默认情况下，它会用前一个模型版本回答待处理的请求（如果有的话），同时用新版本处理新请求。一旦每个待处理的请求都得到回答，之前的模型版本就会被卸载。您可以在TF Serving日志（*my_server.log*）中看到这个过程：

```py
[...]
Reading SavedModel from: /models/my_mnist_model/0002
Reading meta graph with tags { serve }
[...]
Successfully loaded servable version {name: my_mnist_model version: 2}
Quiescing servable version {name: my_mnist_model version: 1}
Done quiescing servable version {name: my_mnist_model version: 1}
Unloading servable version {name: my_mnist_model version: 1}

```

###### 提示

如果SavedModel包含*assets/extra*目录中的一些示例实例，您可以配置TF Serving在开始使用它来处理请求之前在这些实例上运行新模型。这称为*模型预热*：它将确保一切都被正确加载，避免第一次请求的长响应时间。

这种方法提供了平稳的过渡，但可能会使用过多的RAM，特别是GPU RAM，通常是最有限的。在这种情况下，您可以配置TF Serving，使其处理所有挂起的请求与先前的模型版本，并在加载和使用新的模型版本之前卸载它。这种配置将避免同时加载两个模型版本，但服务将在短时间内不可用。

正如您所看到的，TF Serving使部署新模型变得简单。此外，如果您发现第二个版本的效果不如预期，那么回滚到第一个版本就像删除*my_mnist_model/0002*目录一样简单。

###### 提示

TF Serving的另一个重要特性是其自动批处理能力，您可以在启动时使用`--enable_batching`选项来激活它。当TF Serving在短时间内接收到多个请求时（延迟可配置），它会在使用模型之前自动将它们批处理在一起。通过利用GPU的性能，这将显著提高性能。一旦模型返回预测结果，TF Serving会将每个预测结果分发给正确的客户端。通过增加批处理延迟（参见`--batching_parameters_file`选项），您可以在一定程度上牺牲一点延迟以获得更大的吞吐量。

如果您希望每秒获得许多查询，您将希望在多台服务器上部署TF Serving并负载平衡查询（请参见[图19-2](#tf_serving_load_balancing_diagram)）。这将需要在这些服务器上部署和管理许多TF Serving容器。处理这一问题的一种方法是使用诸如[Kubernetes](https://kubernetes.io)之类的工具，它是一个简化跨多台服务器容器编排的开源系统。如果您不想购买、维护和升级所有硬件基础设施，您将希望在云平台上使用虚拟机，如Amazon AWS、Microsoft Azure、Google Cloud Platform、IBM Cloud、Alibaba Cloud、Oracle Cloud或其他平台即服务（PaaS）提供商。管理所有虚拟机，处理容器编排（即使借助Kubernetes的帮助），照顾TF Serving配置、调整和监控——所有这些都可能成为一项全职工作。幸运的是，一些服务提供商可以为您处理所有这些事务。在本章中，我们将使用Vertex AI：它是今天唯一支持TPUs的平台；它支持TensorFlow 2、Scikit-Learn和XGBoost；并提供一套不错的人工智能服务。在这个领域还有其他几家提供商也能够提供TensorFlow模型的服务，比如Amazon AWS SageMaker和Microsoft AI Platform，所以请确保也查看它们。

![mls3 1902](assets/mls3_1902.png)

###### 图19-2。使用负载平衡扩展TF Serving

现在让我们看看如何在云上提供我们出色的MNIST模型！

## 在Vertex AI上创建一个预测服务

Vertex AI是Google Cloud Platform（GCP）内的一个平台，提供各种与人工智能相关的工具和服务。您可以上传数据集，让人类对其进行标记，将常用特征存储在特征存储中，并将其用于训练或生产中，使用多个GPU或TPU服务器进行模型训练，并具有自动超参数调整或模型架构搜索（AutoML）功能。您还可以管理已训练的模型，使用它们对大量数据进行批量预测，为数据工作流程安排多个作业，通过REST或gRPC以规模化方式提供模型服务，并在名为*Workbench*的托管Jupyter环境中对数据和模型进行实验。甚至还有一个*Matching Engine*服务，可以非常高效地比较向量（即，近似最近邻）。GCP还包括其他AI服务，例如计算机视觉、翻译、语音转文本等API。

在我们开始之前，有一些设置需要处理：

1.  登录您的Google账户，然后转到[Google Cloud Platform控制台](https://console.cloud.google.com)（参见[图19-3](#gcp_screenshot)）。如果您没有Google账户，您将需要创建一个。

1.  如果这是您第一次使用GCP，您将需要阅读并接受条款和条件。新用户可以获得免费试用，包括价值300美元的GCP信用，您可以在90天内使用（截至2022年5月）。您只需要其中的一小部分来支付本章中将使用的服务。注册免费试用后，您仍然需要创建一个付款配置文件并输入您的信用卡号码：这是用于验证目的——可能是为了避免人们多次使用免费试用，但您不会被收取前300美元的费用，之后只有在您选择升级到付费账户时才会收费。

    ![mls3 1903](assets/mls3_1903.png)

    ###### 图19-3. Google Cloud Platform控制台

1.  如果您以前使用过GCP并且您的免费试用已经过期，那么您在本章中将使用的服务将会花费一些钱。这不应该太多，特别是如果您记得在不再需要这些服务时关闭它们。在运行任何服务之前，请确保您理解并同意定价条件。如果服务最终花费超出您的预期，我在此不承担任何责任！还请确保您的计费账户是活动的。要检查，请打开左上角的☰导航菜单，点击计费，然后确保您已设置付款方式并且计费账户是活动的。

1.  GCP 中的每个资源都属于一个 *项目*。这包括您可能使用的所有虚拟机、存储的文件和运行的训练作业。当您创建一个帐户时，GCP 会自动为您创建一个名为“我的第一个项目”的项目。如果您愿意，可以通过转到项目设置来更改其显示名称：在 ☰ 导航菜单中，选择“IAM 和管理员 → 设置”，更改项目的显示名称，然后单击“保存”。请注意，项目还有一个唯一的 ID 和编号。您可以在创建项目时选择项目 ID，但以后无法更改。项目编号是自动生成的，无法更更改。如果您想创建一个新项目，请单击页面顶部的项目名称，然后单击“新项目”并输入项目名称。您还可以单击“编辑”来设置项目 ID。确保此新项目的计费处于活动状态，以便可以对服务费用进行计费（如果有免费信用）。

    ###### 警告

    请始终设置提醒，以便在您知道只需要几个小时时关闭服务，否则您可能会让其运行数天或数月，从而产生潜在的显著成本。

1.  现在您已经拥有GCP帐户和项目，并且计费已激活，您必须激活所需的API。在☰导航菜单中，选择“API和服务”，确保启用了Cloud Storage API。如果需要，点击+启用API和服务，找到Cloud Storage，并启用它。还要启用Vertex AI API。

您可以继续通过GCP控制台完成所有操作，但我建议改用Python：这样您可以编写脚本来自动化几乎任何您想要在GCP上完成的任务，而且通常比通过菜单和表单点击更方便，特别是对于常见任务。

在您使用任何GCP服务之前，您需要做的第一件事是进行身份验证。在使用Colab时最简单的解决方案是执行以下代码：

```py
from google.colab import auth

auth.authenticate_user()
```

认证过程基于OAuth 2.0：一个弹出窗口会要求您确认您希望Colab笔记本访问您的Google凭据。如果您接受，您必须选择与GCP相同的Google帐户。然后，您将被要求确认您同意授予Colab对Google Drive和GCP中所有数据的完全访问权限。如果您允许访问，只有当前笔记本将具有访问权限，并且仅在Colab运行时到期之前。显然，只有在您信任笔记本中的代码时才应接受此操作。

###### 警告

如果您*不*使用来自https://github.com/ageron/handson-ml3的官方笔记本，则应格外小心：如果笔记本的作者心怀不轨，他们可能包含代码来对您的数据进行任何操作。

现在让我们创建一个Google Cloud Storage存储桶来存储我们的SavedModels（GCS的*存储桶*是您数据的容器）。为此，我们将使用预先安装在Colab中的`google-cloud-storage`库。我们首先创建一个`Client`对象，它将作为与GCS的接口，然后我们使用它来创建存储桶：

```py
from google.cloud import storage

project_id = "my_project"  # change this to your project ID
bucket_name = "my_bucket"  # change this to a unique bucket name
location = "us-central1"

storage_client = storage.Client(project=project_id)
bucket = storage_client.create_bucket(bucket_name, location=location)
```

###### 提示

如果您想重用现有的存储桶，请将最后一行替换为`bucket = storage_client.bucket(bucket_name)`。确保`location`设置为存储桶的地区。

GCS使用单个全球命名空间用于存储桶，因此像“machine-learning”这样的简单名称很可能不可用。确保存储桶名称符合DNS命名约定，因为它可能在DNS记录中使用。此外，存储桶名称是公开的，因此不要在名称中放入任何私人信息。通常使用您的域名、公司名称或项目ID作为前缀以确保唯一性，或者只需在名称中使用一个随机数字。

如果您想要，可以更改区域，但请确保选择支持GPU的区域。此外，您可能需要考虑到不同区域之间价格差异很大，一些区域产生的CO₂比其他区域多得多，一些区域不支持所有服务，并且使用单一区域存储桶可以提高性能。有关更多详细信息，请参阅[Google Cloud的区域列表](https://homl.info/regions)和[Vertex AI的位置文档](https://homl.info/locations)。如果您不确定，最好选择`"us-central1"`。

接下来，让我们将*my_mnist_model*目录上传到新的存储桶。在GCS中，文件被称为*blobs*（或*objects*），在幕后它们都只是放在存储桶中，没有任何目录结构。Blob名称可以是任意的Unicode字符串，甚至可以包含斜杠(`/`)。GCP控制台和其他工具使用这些斜杠来产生目录的幻觉。因此，当我们上传*my_mnist_model*目录时，我们只关心文件，而不是目录。

```py
def upload_directory(bucket, dirpath):
    dirpath = Path(dirpath)
    for filepath in dirpath.glob("**/*"):
        if filepath.is_file():
            blob = bucket.blob(filepath.relative_to(dirpath.parent).as_posix())
            blob.upload_from_filename(filepath)

upload_directory(bucket, "my_mnist_model")
```

这个函数现在运行良好，但如果有很多文件要上传，它会非常慢。通过多线程可以很容易地大大加快速度（请参阅笔记本中的实现）。或者，如果您有Google Cloud CLI，则可以使用以下命令：

```py
!gsutil -m cp -r my_mnist_model gs://{bucket_name}/
```

接下来，让我们告诉Vertex AI关于我们的MNIST模型。要与Vertex AI通信，我们可以使用`google-cloud-aiplatform`库（它仍然使用旧的AI Platform名称而不是Vertex AI）。它在Colab中没有预安装，所以我们需要安装它。之后，我们可以导入该库并进行初始化——只需指定一些项目ID和位置的默认值——然后我们可以创建一个新的Vertex AI模型：我们指定一个显示名称，我们模型的GCS路径（在这种情况下是版本0001），以及我们希望Vertex AI使用的Docker容器的URL来运行此模型。如果您访问该URL并向上导航一个级别，您将找到其他可以使用的容器。这个支持带有GPU的TensorFlow 2.8：

```py
from google.cloud import aiplatform

server_image = "gcr.io/cloud-aiplatform/prediction/tf2-gpu.2-8:latest"

aiplatform.init(project=project_id, location=location)
mnist_model = aiplatform.Model.upload(
    display_name="mnist",
    artifact_uri=f"gs://{bucket_name}/my_mnist_model/0001",
    serving_container_image_uri=server_image,
)
```

现在让我们部署这个模型，这样我们就可以通过gRPC或REST API查询它以进行预测。为此，我们首先需要创建一个*端点*。这是客户端应用程序在想要访问服务时连接的地方。然后我们需要将我们的模型部署到这个端点：

```py
endpoint = aiplatform.Endpoint.create(display_name="mnist-endpoint")

endpoint.deploy(
    mnist_model,
    min_replica_count=1,
    max_replica_count=5,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=1
)
```

这段代码可能需要几分钟才能运行，因为Vertex AI 需要设置一个虚拟机。在这个例子中，我们使用一个相当基本的 `n1-standard-4` 类型的机器（查看[*https://homl.info/machinetypes*](https://homl.info/machinetypes) 获取其他类型）。我们还使用了一个基本的 `NVIDIA_TESLA_K80` 类型的 GPU（查看[*https://homl.info/accelerators*](https://homl.info/accelerators) 获取其他类型）。如果您选择的区域不是 `"us-central1"`，那么您可能需要将机器类型或加速器类型更改为该区域支持的值（例如，并非所有区域都有 Nvidia Tesla K80 GPU）。

###### 注意

Google Cloud Platform 实施各种 GPU 配额，包括全球范围和每个地区：您不能在未经 Google 授权的情况下创建成千上万个 GPU 节点。要检查您的配额，请在 GCP 控制台中打开“IAM 和管理员 → 配额”。如果某些配额太低（例如，如果您需要在特定地区更多的 GPU），您可以要求增加它们；通常需要大约 48 小时。

Vertex AI将最初生成最少数量的计算节点（在这种情况下只有一个），每当每秒查询次数变得过高时，它将生成更多节点（最多为您定义的最大数量，这种情况下为五个），并在它们之间负载均衡查询。如果一段时间内QPS速率下降，Vertex AI将自动停止额外的计算节点。因此，成本直接与负载、您选择的机器和加速器类型以及您在GCS上存储的数据量相关。这种定价模型非常适合偶尔使用者和有重要使用高峰的服务。对于初创公司来说也是理想的：价格保持低延迟到公司真正开始运营。

恭喜，您已经将第一个模型部署到云端！现在让我们查询这个预测服务：

```py
response = endpoint.predict(instances=X_new.tolist())
```

我们首先需要将要分类的图像转换为Python列表，就像我们之前使用REST API向TF Serving发送请求时所做的那样。响应对象包含预测结果，表示为Python浮点数列表的列表。让我们将它们四舍五入到两位小数并将它们转换为NumPy数组：

```py
>>> import numpy as np
>>> np.round(response.predictions, 2)
array([[0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 1\.  , 0\.  , 0\.  ],
 [0\.  , 0\.  , 0.99, 0.01, 0\.  , 0\.  , 0\.  , 0\.  , 0\.  , 0\.  ],
 [0\.  , 0.97, 0.01, 0\.  , 0\.  , 0\.  , 0\.  , 0.01, 0\.  , 0\.  ]])
```

是的！我们得到了与之前完全相同的预测结果。我们现在在云上有一个很好的预测服务，我们可以从任何地方安全地查询，并且可以根据QPS的数量自动扩展或缩小。当您使用完端点后，请不要忘记将其删除，以避免无谓地支付费用：

```py
endpoint.undeploy_all()  # undeploy all models from the endpoint
endpoint.delete()
```

现在让我们看看如何在Vertex AI上运行作业，对可能非常大的数据批次进行预测。

## 在Vertex AI上运行批量预测作业

如果我们需要进行大量预测，那么我们可以请求 Vertex AI 为我们运行预测作业，而不是重复调用我们的预测服务。这不需要端点，只需要一个模型。例如，让我们在测试集的前 100 张图像上运行一个预测作业，使用我们的 MNIST 模型。为此，我们首先需要准备批处理并将其上传到 GCS。一种方法是创建一个文件，每行包含一个实例，每个实例都格式化为 JSON 值——这种格式称为 *JSON Lines*——然后将此文件传递给 Vertex AI。因此，让我们在一个新目录中创建一个 JSON Lines 文件，然后将此目录上传到 GCS：

```py
batch_path = Path("my_mnist_batch")
batch_path.mkdir(exist_ok=True)
with open(batch_path / "my_mnist_batch.jsonl", "w") as jsonl_file:
    for image in X_test[:100].tolist():
        jsonl_file.write(json.dumps(image))
        jsonl_file.write("\n")

upload_directory(bucket, batch_path)
```

现在我们准备启动预测作业，指定作业的名称、要使用的机器和加速器的类型和数量，刚刚创建的 JSON Lines 文件的 GCS 路径，以及 Vertex AI 将保存模型预测的 GCS 目录的路径：

```py
batch_prediction_job = mnist_model.batch_predict(
    job_display_name="my_batch_prediction_job",
    machine_type="n1-standard-4",
    starting_replica_count=1,
    max_replica_count=5,
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=1,
    gcs_source=[f"gs://{bucket_name}/{batch_path.name}/my_mnist_batch.jsonl"],
    gcs_destination_prefix=f"gs://{bucket_name}/my_mnist_predictions/",
    sync=True  # set to False if you don't want to wait for completion
)
```

###### 提示

对于大批量数据，您可以将输入拆分为多个JSON Lines文件，并通过`gcs_source`参数列出它们。

这将需要几分钟的时间，主要是为了在Vertex AI上生成计算节点。一旦这个命令完成，预测将会以类似*prediction.results-00001-of-00002*的文件集合中可用。这些文件默认使用JSON Lines格式，每个值都是包含实例及其对应预测（即10个概率）的字典。实例按照输入的顺序列出。该作业还会输出*prediction-errors*文件，如果出现问题，这些文件对于调试可能会有用。我们可以使用`batch_prediction_job.iter_outputs()`迭代所有这些输出文件，所以让我们遍历所有的预测并将它们存储在`y_probas`数组中：

```py
y_probas = []
for blob in batch_prediction_job.iter_outputs():
    if "prediction.results" in blob.name:
        for line in blob.download_as_text().splitlines():
            y_proba = json.loads(line)["prediction"]
            y_probas.append(y_proba)
```

现在让我们看看这些预测有多好：

```py
>>> y_pred = np.argmax(y_probas, axis=1)
>>> accuracy = np.sum(y_pred == y_test[:100]) / 100
0.98
```

很好，98%的准确率！

JSON Lines格式是默认格式，但是当处理大型实例（如图像）时，它太冗长了。幸运的是，`batch_predict()`方法接受一个`instances_format`参数，让您可以选择另一种格式。它默认为`"jsonl"`，但您可以将其更改为`"csv"`、`"tf-record"`、`"tf-record-gzip"`、`"bigquery"`或`"file-list"`。如果将其设置为`"file-list"`，那么`gcs_source`参数应指向一个文本文件，其中每行包含一个输入文件路径；例如，指向PNG图像文件。Vertex AI将读取这些文件作为二进制文件，使用Base64对其进行编码，并将生成的字节字符串传递给模型。这意味着您必须在模型中添加一个预处理层来解析Base64字符串，使用`tf.io.decode_base64()`。如果文件是图像，则必须使用类似`tf.io.decode_image()`或`tf.io.decode_png()`的函数来解析结果，如[第13章](ch13.html#data_chapter)中所讨论的。

当您完成使用模型后，如果需要，可以通过运行`mnist_model.delete()`来删除它。您还可以删除在您的GCS存储桶中创建的目录，可选地删除存储桶本身（如果为空），以及批量预测作业。

```py
for prefix in ["my_mnist_model/", "my_mnist_batch/", "my_mnist_predictions/"]:
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        blob.delete()

bucket.delete()  # if the bucket is empty
batch_prediction_job.delete()
```

您现在知道如何将模型部署到Vertex AI，创建预测服务，并运行批量预测作业。但是如果您想将模型部署到移动应用程序，或者嵌入式设备，比如加热控制系统、健身追踪器或自动驾驶汽车呢？

# 将模型部署到移动设备或嵌入式设备

机器学习模型不仅限于在拥有多个GPU的大型集中式服务器上运行：它们可以更接近数据源运行（这被称为*边缘计算*），例如在用户的移动设备或嵌入式设备中。去中心化计算并将其移向边缘有许多好处：它使设备即使未连接到互联网时也能智能化，通过不必将数据发送到远程服务器来减少延迟并减轻服务器负载，并且可能提高隐私性，因为用户的数据可以保留在设备上。

然而，将模型部署到边缘也有其缺点。与强大的多GPU服务器相比，设备的计算资源通常很少。一个大模型可能无法适应设备，可能使用过多的RAM和CPU，并且可能下载时间过长。结果，应用可能变得无响应，设备可能会发热并迅速耗尽电池。为了避免这一切，您需要制作一个轻量级且高效的模型，而不会牺牲太多准确性。 [TFLite](https://tensorflow.org/lite)库提供了几个工具，帮助您将模型部署到边缘，主要有三个目标：

+   减小模型大小，缩短下载时间并减少RAM使用量。

+   减少每次预测所需的计算量，以减少延迟、电池使用量和发热。

+   使模型适应特定设备的限制。

为了减小模型大小，TFLite 的模型转换器可以接受 SavedModel 并将其压缩为基于 [FlatBuffers](https://google.github.io/flatbuffers) 的更轻量级格式。这是一个高效的跨平台序列化库（有点像协议缓冲区），最初由谷歌为游戏创建。它设计成可以直接将 FlatBuffers 加载到 RAM 中，无需任何预处理：这样可以减少加载时间和内存占用。一旦模型加载到移动设备或嵌入式设备中，TFLite 解释器将执行它以进行预测。以下是如何将 SavedModel 转换为 FlatBuffer 并保存为 *.tflite* 文件的方法：

```py
converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
tflite_model = converter.convert()
with open("my_converted_savedmodel.tflite", "wb") as f:
    f.write(tflite_model)
```

###### 提示

您还可以使用 `tf.lite.TFLiteConverter.from_keras_model(model)` 将 Keras 模型直接保存为 FlatBuffer 格式。

转换器还优化模型，既缩小模型大小，又减少延迟。它修剪所有不需要进行预测的操作（例如训练操作），并在可能的情况下优化计算；例如，3 × *a* + 4 ×_ a_ + 5 × *a* 将被转换为12 × *a*。此外，它尝试在可能的情况下融合操作。例如，如果可能的话，批量归一化层最终会合并到前一层的加法和乘法操作中。要了解TFLite可以对模型进行多少优化，可以下载其中一个[预训练的TFLite模型](https://homl.info/litemodels)，例如*Inception_V1_quant*（点击*tflite&pb*），解压缩存档，然后打开优秀的[Netron图形可视化工具](https://netron.app)并上传*.pb*文件以查看原始模型。这是一个庞大而复杂的图形，对吧？接下来，打开优化后的*.tflite*模型，惊叹于其美丽！

除了简单地使用较小的神经网络架构之外，您可以减小模型大小的另一种方法是使用较小的位宽：例如，如果您使用半精度浮点数（16位）而不是常规浮点数（32位），模型大小将缩小2倍，代价是（通常很小的）准确度下降。此外，训练速度将更快，您将使用大约一半的GPU内存。

TFLite的转换器可以进一步将模型权重量化为固定点、8位整数！与使用32位浮点数相比，这导致了四倍的大小减小。最简单的方法称为*后训练量化*：它只是在训练后量化权重，使用一种相当基本但高效的对称量化技术。它找到最大绝对权重值*m*，然后将浮点范围–*m*到+*m*映射到固定点（整数）范围–127到+127。例如，如果权重范围从–1.5到+0.8，则字节–127、0和+127将分别对应于浮点–1.5、0.0和+1.5（参见[图19-5](#quantization_diagram)）。请注意，当使用对称量化时，0.0始终映射为0。还请注意，在此示例中不会使用字节值+68到+127，因为它们映射到大于+0.8的浮点数。

![mls3 1905](assets/mls3_1905.png)

###### 图19-5。从32位浮点数到8位整数，使用对称量化

要执行这种训练后的量化，只需在调用`convert()`方法之前将`DEFAULT`添加到转换器优化列表中：

```py
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

这种技术显著减小了模型的大小，使得下载速度更快，占用的存储空间更少。在运行时，量化的权重在使用之前会被转换回浮点数。这些恢复的浮点数与原始浮点数并不完全相同，但也不会相差太远，因此精度损失通常是可以接受的。为了避免一直重新计算浮点值，这样会严重减慢模型的速度，TFLite会对其进行缓存：不幸的是，这意味着这种技术并不会减少RAM的使用量，也不会加快模型的速度。它主要用于减小应用程序的大小。

减少延迟和功耗的最有效方法是对激活进行量化，使得计算可以完全使用整数，而无需任何浮点运算。即使使用相同的位宽（例如，32位整数而不是32位浮点数），整数计算使用的CPU周期更少，消耗的能量更少，产生的热量也更少。如果还减少位宽（例如，降至8位整数），可以获得巨大的加速。此外，一些神经网络加速器设备（如Google的Edge TPU）只能处理整数，因此权重和激活的完全量化是强制性的。这可以在训练后完成；它需要一个校准步骤来找到激活的最大绝对值，因此您需要向TFLite提供代表性的训练数据样本（不需要很大），它将通过模型处理数据并测量量化所需的激活统计信息。这一步通常很快。

量化的主要问题是它会失去一点准确性：这类似于在权重和激活中添加噪声。如果准确性下降太严重，那么您可能需要使用*量化感知训练*。这意味着向模型添加虚假量化操作，以便它在训练过程中学会忽略量化噪声；最终的权重将更加稳健地适应量化。此外，校准步骤可以在训练过程中自动处理，这简化了整个过程。

我已经解释了TFLite的核心概念，但要完全编写移动或嵌入式应用程序需要一本专门的书。幸运的是，一些书籍存在：如果您想了解有关为移动和嵌入式设备构建TensorFlow应用程序的更多信息，请查看O'Reilly的书籍[*TinyML: Machine Learning with TensorFlow on Arduino and Ultra-Low Power Micro-Controllers*](https://homl.info/tinyml)，作者是Pete Warden（TFLite团队的前负责人）和Daniel Situnayake，以及[*AI and Machine Learning for On-Device Development*](https://homl.info/ondevice)，作者是Laurence Moroney。

那么，如果您想在网站中使用您的模型，在用户的浏览器中直接运行呢？

# 在网页中运行模型

在客户端，即用户的浏览器中运行您的机器学习模型，而不是在服务器端运行，可以在许多场景下非常有用，例如：

+   当您的网络应用经常在用户的连接不稳定或缓慢的情况下使用（例如，徒步者的网站），因此在客户端直接运行模型是使您的网站可靠的唯一方法。

+   当您需要模型的响应尽可能快时（例如，用于在线游戏）。消除查询服务器进行预测的需要肯定会减少延迟，并使网站更加响应。

+   当您的网络服务基于一些私人用户数据进行预测，并且您希望通过在客户端进行预测来保护用户的隐私，以便私人数据永远不必离开用户的设备。

对于所有这些场景，您可以使用[TensorFlow.js（TFJS）JavaScript库](https://tensorflow.org/js)。该库可以在用户的浏览器中加载TFLite模型并直接进行预测。例如，以下JavaScript模块导入了TFJS库，下载了一个预训练的MobileNet模型，并使用该模型对图像进行分类并记录预测结果。您可以在[*https://homl.info/tfjscode*](https://homl.info/tfjscode)上尝试这段代码，使用Glitch.com，这是一个允许您免费在浏览器中构建Web应用程序的网站；点击页面右下角的预览按钮查看代码的运行情况：

```py
import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest";
import "https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@1.0.0";

const image = document.getElementById("image");

mobilenet.load().then(model => {
    model.classify(image).then(predictions => {
        for (var i = 0; i < predictions.length; i++) {
            let className = predictions[i].className
            let proba = (predictions[i].probability * 100).toFixed(1)
            console.log(className + " : " + proba + "%");
        }
    });
});
```

甚至可以将这个网站转变成一个*渐进式Web应用程序*（PWA）：这是一个遵守一系列标准的网站，使其可以在任何浏览器中查看，甚至可以在移动设备上作为独立应用程序安装。例如，在移动设备上尝试访问[*https://homl.info/tfjswpa*](https://homl.info/tfjswpa)：大多数现代浏览器会询问您是否想要将TFJS演示添加到主屏幕。如果您接受，您将在应用程序列表中看到一个新图标。点击此图标将在其自己的窗口中加载TFJS演示网站，就像常规移动应用程序一样。PWA甚至可以配置为离线工作，通过使用*服务工作者*：这是一个在浏览器中以自己独立线程运行的JavaScript模块，拦截网络请求，使其可以缓存资源，从而使PWA可以更快地运行，甚至完全离线运行。它还可以传递推送消息，在后台运行任务等。PWA允许您管理Web和移动设备的单个代码库。它们还使得更容易确保所有用户运行您应用程序的相同版本。您可以在Glitch.com上玩这个TFJS演示的PWA代码，网址是[*https://homl.info/wpacode*](https://homl.info/wpacode)。

###### 提示

在[*https://tensorflow.org/js/demos*](https://tensorflow.org/js/demos)上查看更多在您的浏览器中运行的机器学习模型的演示。

TFJS还支持在您的网络浏览器中直接训练模型！而且速度相当快。如果您的计算机有GPU卡，那么TFJS通常可以使用它，即使它不是Nvidia卡。实际上，TFJS将在可用时使用WebGL，由于现代网络浏览器通常支持各种GPU卡，TFJS实际上支持的GPU卡比常规的TensorFlow更多（后者仅支持Nvidia卡）。

在用户的网络浏览器中训练模型可以特别有用，可以确保用户的数据保持私密。模型可以在中央进行训练，然后在浏览器中根据用户的数据进行本地微调。如果您对这个话题感兴趣，请查看[*联邦学习*](https://tensorflow.org/federated)。

再次强调，要全面涵盖这个主题需要一本完整的书。如果您想了解更多关于TensorFlow.js的内容，请查看O'reilly图书《云端、移动和边缘的实用深度学习》（Anirudh Koul等著）或《学习TensorFlow.js》（Gant Laborde著）。

现在您已经看到如何将TensorFlow模型部署到TF Serving，或者通过Vertex AI部署到云端，或者使用TFLite部署到移动和嵌入式设备，或者使用TFJS部署到Web浏览器，让我们讨论如何使用GPU加速计算。

# 使用GPU加速计算

在[第11章](ch11.html#deep_chapter)中，我们看了几种可以显著加快训练速度的技术：更好的权重初始化、复杂的优化器等等。但即使使用了所有这些技术，使用单个CPU的单台机器训练大型神经网络可能需要几个小时、几天，甚至几周，具体取决于任务。由于GPU的出现，这种训练时间可以缩短到几分钟或几小时。这不仅节省了大量时间，还意味着您可以更轻松地尝试各种模型，并经常使用新数据重新训练您的模型。

在之前的章节中，我们在Google Colab上使用了启用GPU的运行时。您只需从运行时菜单中选择“更改运行时类型”，然后选择GPU加速器类型；TensorFlow会自动检测GPU并使用它加速计算，代码与没有GPU时完全相同。然后，在本章中，您看到了如何将模型部署到Vertex AI上的多个启用GPU的计算节点：只需在创建Vertex AI模型时选择正确的启用GPU的Docker镜像，并在调用`endpoint.deploy()`时选择所需的GPU类型。但是，如果您想购买自己的GPU怎么办？如果您想在单台机器上的CPU和多个GPU设备之间分发计算（参见[图19-6](#multiple_devices_diagram)）？这是我们现在将讨论的内容，然后在本章的后面部分我们将讨论如何在多个服务器上分发计算。

![mls3 1906](assets/mls3_1906.png)

###### 图19-6。在多个设备上并行执行TensorFlow图

## 获取自己的GPU

如果你知道你将会长时间大量使用GPU，那么购买自己的GPU可能是经济上合理的。你可能也想在本地训练模型，因为你不想将数据上传到云端。或者你只是想购买一张用于游戏的GPU卡，并且想将其用于深度学习。

如果您决定购买GPU卡，那么请花些时间做出正确的选择。您需要考虑您的任务所需的RAM数量（例如，图像处理或NLP通常至少需要10GB），带宽（即您可以将数据发送到GPU和从GPU中发送数据的速度），核心数量，冷却系统等。Tim Dettmers撰写了一篇[优秀的博客文章](https://homl.info/66)来帮助您选择：我鼓励您仔细阅读。在撰写本文时，TensorFlow仅支持[具有CUDA Compute Capability 3.5+的Nvidia卡](https://homl.info/cudagpus)（当然还有Google的TPU），但它可能会将其支持扩展到其他制造商，因此请务必查看[TensorFlow的文档](https://tensorflow.org/install)以了解今天支持哪些设备。

如果您选择Nvidia GPU卡，您将需要安装适当的Nvidia驱动程序和几个Nvidia库。这些包括*计算统一设备架构*库（CUDA）工具包，它允许开发人员使用支持CUDA的GPU进行各种计算（不仅仅是图形加速），以及*CUDA深度神经网络*库（cuDNN），一个GPU加速的常见DNN计算库，例如激活层、归一化、前向和反向卷积以及池化（参见[第14章](ch14.html#cnn_chapter)）。cuDNN是Nvidia的深度学习SDK的一部分。请注意，您需要创建一个Nvidia开发者帐户才能下载它。TensorFlow使用CUDA和cuDNN来控制GPU卡并加速计算（参见[图19-7](#cuda_cudnn_diagram)）。

![mls3 1907](assets/mls3_1907.png)

###### 图19-7. TensorFlow使用CUDA和cuDNN来控制GPU并加速DNNs

安装了GPU卡和所有必需的驱动程序和库之后，您可以使用`nvidia-smi`命令来检查一切是否正确安装。该命令列出了可用的GPU卡，以及每张卡上运行的所有进程。在这个例子中，这是一张Nvidia Tesla T4 GPU卡，大约有15GB的可用内存，并且当前没有任何进程在运行：

```py
$ nvidia-smi
Sun Apr 10 04:52:10 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   34C    P8     9W /  70W |      3MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```

要检查TensorFlow是否真正看到您的GPU，请运行以下命令并确保结果不为空：

```py
>>> physical_gpus = tf.config.list_physical_devices("GPU")
>>> physical_gpus
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 管理GPU内存

默认情况下，TensorFlow在第一次运行计算时会自动占用几乎所有可用GPU的RAM，以限制GPU RAM的碎片化。这意味着如果您尝试启动第二个TensorFlow程序（或任何需要GPU的程序），它将很快耗尽RAM。这种情况并不像您可能认为的那样经常发生，因为通常您会在一台机器上运行一个单独的TensorFlow程序：通常是一个训练脚本、一个TF Serving节点或一个Jupyter笔记本。如果出于某种原因需要运行多个程序（例如，在同一台机器上并行训练两个不同的模型），那么您需要更均匀地在这些进程之间分配GPU RAM。

如果您的机器上有多个GPU卡，一个简单的解决方案是将每个GPU卡分配给单个进程。为此，您可以设置`CUDA_VISIBLE_DEVICES`环境变量，以便每个进程只能看到适当的GPU卡。还要设置`CUDA_DEVICE_ORDER`环境变量为`PCI_BUS_ID`，以确保每个ID始终指向相同的GPU卡。例如，如果您有四个GPU卡，您可以启动两个程序，将两个GPU分配给每个程序，通过在两个单独的终端窗口中执行以下命令来实现：

```py
$ CUDA_DEVICE_ORDER=PCI_BUS_IDCUDA_VISIBLE_DEVICES=0,1python3program_1.py*`#` `and``in``another``terminal:`*$ CUDA_DEVICE_ORDER=PCI_BUS_IDCUDA_VISIBLE_DEVICES=3,2python3program_2.py
```

程序1将只看到GPU卡0和1，分别命名为`"/gpu:0"`和`"/gpu:1"`，在TensorFlow中，程序2将只看到GPU卡2和3，分别命名为`"/gpu:1"`和`"/gpu:0"`（注意顺序）。一切都将正常工作（参见[图19-8](#splitting_gpus_diagram)）。当然，您也可以在Python中通过设置`os.environ["CUDA_DEVICE_ORDER"]`和`os.environ["CUDA_VISIBLE_DEVICES"]`来定义这些环境变量，只要在使用TensorFlow之前这样做。

![mls3 1908](assets/mls3_1908.png)

###### 图19-8。每个程序获得两个GPU

另一个选项是告诉TensorFlow只获取特定数量的GPU RAM。这必须在导入TensorFlow后立即完成。例如，要使TensorFlow只在每个GPU上获取2 GiB的RAM，您必须为每个物理GPU设备创建一个*逻辑GPU设备*（有时称为*虚拟GPU设备*），并将其内存限制设置为2 GiB（即2,048 MiB）:

```py
for gpu in physical_gpus:
    tf.config.set_logical_device_configuration(
        gpu,
        [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
    )
```

假设您有四个GPU，每个GPU至少有4 GiB的RAM：在这种情况下，可以并行运行两个像这样的程序，每个程序使用所有四个GPU卡（请参见[图19-9](#sharing_gpus_diagram)）。如果在两个程序同时运行时运行`nvidia-smi`命令，则应该看到每个进程在每张卡上占用2 GiB的RAM。

![mls3 1909](assets/mls3_1909.png)

###### 图19-9。每个程序都可以获得四个GPU，但每个GPU只有2 GiB的RAM

另一个选项是告诉TensorFlow只在需要时获取内存。同样，在导入TensorFlow后必须立即执行此操作：

```py
for gpu in physical_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

另一种方法是将`TF_FORCE_GPU_ALLOW_GROWTH`环境变量设置为`true`。使用这个选项，TensorFlow一旦分配了内存就不会释放它（再次，为了避免内存碎片化），除非程序结束。使用这个选项很难保证确定性行为（例如，一个程序可能会崩溃，因为另一个程序的内存使用量激增），因此在生产环境中，您可能会选择之前的选项之一。然而，有一些情况下它非常有用：例如，当您使用一台机器运行多个Jupyter笔记本时，其中几个使用了TensorFlow。在Colab运行时，`TF_FORCE_GPU_ALLOW_GROWTH`环境变量被设置为`true`。

最后，在某些情况下，您可能希望将一个GPU分成两个或更多*逻辑设备*。例如，如果您只有一个物理GPU，比如在Colab运行时，但您想要测试一个多GPU算法，这将非常有用。以下代码将GPU＃0分成两个逻辑设备，每个设备有2 GiB的RAM（同样，在导入TensorFlow后立即执行）：

```py
tf.config.set_logical_device_configuration(
    physical_gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=2048),
     tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
)
```

这两个逻辑设备被称为`"/gpu:0"`和`"/gpu:1"`, 你可以像使用两个普通GPU一样使用它们。你可以像这样列出所有逻辑设备：

```py
>>> logical_gpus = tf.config.list_logical_devices("GPU")
>>> logical_gpus
[LogicalDevice(name='/device:GPU:0', device_type='GPU'),
 LogicalDevice(name='/device:GPU:1', device_type='GPU')]
```

现在让我们看看TensorFlow如何决定应该使用哪些设备来放置变量和执行操作。

## 将操作和变量放在设备上

Keras和tf.data通常会很好地将操作和变量放在它们应该在的位置，但如果您想要更多控制，您也可以手动将操作和变量放在每个设备上：

+   通常，您希望将数据预处理操作放在CPU上，并将神经网络操作放在GPU上。

+   GPU通常具有相对有限的通信带宽，因此重要的是要避免不必要的数据传输进出GPU。

+   向机器添加更多的CPU RAM是简单且相对便宜的，因此通常有很多，而GPU RAM是内置在GPU中的：它是一种昂贵且有限的资源，因此如果一个变量在接下来的几个训练步骤中不需要，它可能应该放在CPU上（例如，数据集通常应该放在CPU上）。

默认情况下，所有变量和操作都将放置在第一个GPU上（命名为`"/gpu:0"`），除非变量和操作没有GPU内核：这些将放置在CPU上（始终命名为`"/cpu:0"`）。张量或变量的`device`属性告诉您它被放置在哪个设备上。

```py
>>> a = tf.Variable([1., 2., 3.])  # float32 variable goes to the GPU
>>> a.device
'/job:localhost/replica:0/task:0/device:GPU:0'
>>> b = tf.Variable([1, 2, 3])  # int32 variable goes to the CPU
>>> b.device
'/job:localhost/replica:0/task:0/device:CPU:0'
```

您现在可以安全地忽略前缀`/job:localhost/replica:0/task:0`；我们将在本章后面讨论作业、副本和任务。正如您所看到的，第一个变量被放置在GPU＃0上，这是默认设备。但是，第二个变量被放置在CPU上：这是因为整数变量没有GPU内核，或者涉及整数张量的操作没有GPU内核，因此TensorFlow回退到CPU。

如果您想在与默认设备不同的设备上执行操作，请使用`tf.device()`上下文：

```py
>>> with tf.device("/cpu:0"):
...     c = tf.Variable([1., 2., 3.])
...
>>> c.device
'/job:localhost/replica:0/task:0/device:CPU:0'
```

###### 注意

CPU始终被视为单个设备（`"/cpu:0"`），即使您的计算机有多个CPU核心。放置在CPU上的任何操作，如果具有多线程内核，则可能在多个核心上并行运行。

如果您明确尝试将操作或变量放置在不存在或没有内核的设备上，那么TensorFlow将悄悄地回退到默认选择的设备。当您希望能够在不具有相同数量的GPU的不同机器上运行相同的代码时，这是很有用的。但是，如果您希望获得异常，可以运行`tf.config.set_soft_device_placement(False)`。

现在，TensorFlow 如何在多个设备上执行操作呢？

## 跨多个设备并行执行

正如我们在[第12章](ch12.html#tensorflow_chapter)中看到的，使用TF函数的一个好处是并行性。让我们更仔细地看一下这一点。当TensorFlow运行一个TF函数时，它首先分析其图形，找到需要评估的操作列表，并计算每个操作的依赖关系数量。然后TensorFlow将每个具有零依赖关系的操作（即每个源操作）添加到该操作设备的评估队列中（参见[图19-10](#parallelization_diagram)）。一旦一个操作被评估，依赖于它的每个操作的依赖计数器都会减少。一旦一个操作的依赖计数器达到零，它就会被推送到其设备的评估队列中。一旦所有输出都被计算出来，它们就会被返回。

![mls3 1910](assets/mls3_1910.png)

###### 图19-10. TensorFlow图的并行执行

CPU的评估队列中的操作被分派到一个称为*inter-op线程池*的线程池中。如果CPU有多个核心，那么这些操作将有效地并行评估。一些操作具有多线程CPU内核：这些内核将其任务分割为多个子操作，这些子操作被放置在另一个评估队列中，并分派到一个称为*intra-op线程池*的第二线程池中（由所有多线程CPU内核共享）。简而言之，多个操作和子操作可能在不同的CPU核心上并行评估。

对于GPU来说，情况要简单一些。GPU的评估队列中的操作是按顺序评估的。然而，大多数操作都有多线程GPU内核，通常由TensorFlow依赖的库实现，比如CUDA和cuDNN。这些实现有自己的线程池，它们通常会利用尽可能多的GPU线程（这就是为什么GPU不需要一个跨操作线程池的原因：每个操作已经占用了大部分GPU线程）。

例如，在[图19-10](#parallelization_diagram)中，操作A、B和C是源操作，因此它们可以立即被评估。操作A和B被放置在CPU上，因此它们被发送到CPU的评估队列，然后被分派到跨操作线程池并立即并行评估。操作A恰好有一个多线程内核；它的计算被分成三部分，在操作线程池中并行执行。操作C进入GPU #0的评估队列，在这个例子中，它的GPU内核恰好使用cuDNN，它管理自己的内部操作线程池，并在许多GPU线程之间并行运行操作。假设C先完成。D和E的依赖计数器被减少到0，因此两个操作都被推送到GPU #0的评估队列，并按顺序执行。请注意，即使D和E都依赖于C，C也只被评估一次。假设B接下来完成。然后F的依赖计数器从4减少到3，由于不为0，它暂时不运行。一旦A、D和E完成，那么F的依赖计数器达到0，它被推送到CPU的评估队列并被评估。最后，TensorFlow返回请求的输出。

TensorFlow执行的另一个神奇之处是当TF函数修改状态资源（例如变量）时：它确保执行顺序与代码中的顺序匹配，即使语句之间没有显式依赖关系。例如，如果您的TF函数包含`v.assign_add(1)`，然后是`v.assign(v * 2)`，TensorFlow将确保这些操作按照这个顺序执行。

###### 提示

您可以通过调用`tf.config.threading.set_inter_op_parallelism_threads()`来控制inter-op线程池中的线程数。要设置intra-op线程数，请使用`tf.config.threading.set_intra_op_parallelism_threads()`。如果您不希望TensorFlow使用所有CPU核心，或者希望它是单线程的，这将非常有用。⁠^([12](ch19.html#idm45720159777008))

有了这些，您就拥有了在任何设备上运行任何操作并利用GPU的能力所需的一切！以下是您可以做的一些事情：

+   您可以并行训练多个模型，每个模型都在自己的GPU上：只需为每个模型编写一个训练脚本，并在并行运行时设置`CUDA_DEVICE_ORDER`和`CUDA_VISIBLE_DEVICES`，以便每个脚本只能看到一个GPU设备。这对于超参数调整非常有用，因为您可以并行训练具有不同超参数的多个模型。如果您有一台具有两个GPU的单台机器，并且在一个GPU上训练一个模型需要一个小时，那么并行训练两个模型，每个模型都在自己专用的GPU上，只需要一个小时。简单！

+   您可以在单个GPU上训练一个模型，并在CPU上并行执行所有预处理操作，使用数据集的`prefetch()`方法提前准备好接下来的几批数据，以便在GPU需要时立即使用（参见第13章）。

+   如果您的模型接受两个图像作为输入，并在使用两个CNN处理它们之前将它们连接起来，那么如果您将每个CNN放在不同的GPU上，它可能会运行得更快。

+   您可以创建一个高效的集成：只需在每个GPU上放置一个不同训练过的模型，这样您就可以更快地获得所有预测结果，以生成集成的最终预测。

但是如果您想通过使用多个GPU加速训练呢？

# 在多个设备上训练模型

训练单个模型跨多个设备有两种主要方法：*模型并行*，其中模型在设备之间分割，和*数据并行*，其中模型在每个设备上复制，并且每个副本在不同的数据子集上进行训练。让我们看看这两种选择。

## 模型并行

到目前为止，我们已经在单个设备上训练了每个神经网络。如果我们想要在多个设备上训练单个神经网络怎么办？这需要将模型分割成单独的块，并在不同的设备上运行每个块。不幸的是，这种模型并行化实际上非常棘手，其有效性确实取决于神经网络的架构。对于全连接网络，从这种方法中通常无法获得太多好处。直觉上，似乎将模型分割的一种简单方法是将每一层放在不同的设备上，但这并不起作用，因为每一层都需要等待前一层的输出才能执行任何操作。也许你可以垂直切割它——例如，将每一层的左半部分放在一个设备上，右半部分放在另一个设备上？这样稍微好一些，因为每一层的两半确实可以并行工作，但问题在于下一层的每一半都需要上一层两半的输出，因此会有大量的跨设备通信（由虚线箭头表示）。这很可能会完全抵消并行计算的好处，因为跨设备通信速度很慢（当设备位于不同的机器上时更是如此）。

![mls3 1911](assets/mls3_1911.png)

###### 图19-11。拆分完全连接的神经网络

一些神经网络架构，如卷积神经网络（参见[第14章](ch14.html#cnn_chapter)），包含仅部分连接到较低层的层，因此更容易以有效的方式在设备之间分发块（参见[图19-12](#split_partially_connected_diagram)）。

![mls3 1912](assets/mls3_1912.png)

###### 图19-12。拆分部分连接的神经网络

深度递归神经网络（参见[第15章](ch15.html#rnn_chapter)）可以更有效地跨多个GPU进行分割。如果将网络水平分割，将每一层放在不同的设备上，并将输入序列输入网络进行处理，那么在第一个时间步中只有一个设备会处于活动状态（处理序列的第一个值），在第二个时间步中两个设备会处于活动状态（第二层将处理第一层的输出值，而第一层将处理第二个值），当信号传播到输出层时，所有设备将同时处于活动状态（[图19-13](#split_rnn_network_diagram)）。尽管设备之间仍然存在大量的跨设备通信，但由于每个单元可能相当复杂，理论上并行运行多个单元的好处可能会超过通信惩罚。然而，在实践中，在单个GPU上运行的常规堆叠`LSTM`层实际上运行得更快。

![mls3 1913](assets/mls3_1913.png)

###### 图19-13。拆分深度递归神经网络

简而言之，模型并行可能会加快某些类型的神经网络的运行或训练速度，但并非所有类型的神经网络都适用，并且需要特别注意和调整，例如确保需要进行通信的设备在同一台机器上运行。接下来我们将看一个更简单且通常更有效的选择：数据并行。

## 数据并行

另一种并行训练神经网络的方法是在每个设备上复制它，并在所有副本上同时运行每个训练步骤，为每个副本使用不同的小批量。然后对每个副本计算的梯度进行平均，并将结果用于更新模型参数。这被称为*数据并行*，有时也称为*单程序，多数据*（SPMD）。这个想法有许多变体，让我们看看最重要的几种。

### 使用镜像策略的数据并行

可以说，最简单的方法是在所有GPU上完全镜像所有模型参数，并始终在每个GPU上应用完全相同的参数更新。这样，所有副本始终保持完全相同。这被称为*镜像策略*，在使用单台机器时特别高效（参见[图19-14](#mirrored_strategy_diagram)）。

![mls3 1914](assets/mls3_1914.png)

###### 图19-14. 使用镜像策略的数据并行

使用这种方法的棘手部分是高效地计算所有GPU的所有梯度的平均值，并将结果分布到所有GPU上。这可以使用*AllReduce*算法来完成，这是一类算法，多个节点合作以高效地执行*reduce操作*（例如计算平均值、总和和最大值），同时确保所有节点获得相同的最终结果。幸运的是，有现成的实现这种算法，您将会看到。

### 集中式参数的数据并行

另一种方法是将模型参数存储在执行计算的GPU设备之外（称为*工作器*）；例如，在CPU上（参见[图19-15](#data_parallelism_diagram)）。在分布式设置中，您可以将所有参数放在一个或多个仅称为*参数服务器*的CPU服务器上，其唯一作用是托管和更新参数。

![mls3 1915](assets/mls3_1915.png)

###### 图19-15. 集中式参数的数据并行

镜像策略强制所有GPU上的权重更新同步进行，而这种集中式方法允许同步或异步更新。让我们来看看这两种选择的优缺点。

#### 同步更新

在*同步更新*中，聚合器会等待所有梯度可用后再计算平均梯度并将其传递给优化器，优化器将更新模型参数。一旦一个副本完成计算其梯度，它必须等待参数更新后才能继续下一个小批量。缺点是一些设备可能比其他设备慢，因此快速设备将不得不在每一步等待慢速设备，使整个过程与最慢设备一样慢。此外，参数将几乎同时复制到每个设备上（在梯度应用后立即），这可能会饱和参数服务器的带宽。

###### 提示

为了减少每个步骤的等待时间，您可以忽略最慢几个副本（通常约10%）的梯度。例如，您可以运行20个副本，但每个步骤只聚合来自最快的18个副本的梯度，并忽略最后2个的梯度。一旦参数更新，前18个副本可以立即开始工作，而无需等待最慢的2个副本。这种设置通常被描述为有18个副本加上2个*备用副本*。

#### 异步更新

使用异步更新时，每当一个副本完成梯度计算后，梯度立即用于更新模型参数。没有聚合（它删除了“均值”步骤在[图19-15](#data_parallelism_diagram)中）和没有同步。副本独立于其他副本工作。由于不需要等待其他副本，这种方法每分钟可以运行更多的训练步骤。此外，尽管参数仍然需要在每一步复制到每个设备，但对于每个副本，这发生在不同的时间，因此带宽饱和的风险降低了。

使用异步更新的数据并行是一个吸引人的选择，因为它简单、没有同步延迟，并且更好地利用了带宽。然而，尽管在实践中它表现得相当不错，但它能够工作几乎令人惊讶！事实上，当一个副本基于某些参数值计算梯度完成时，这些参数将已经被其他副本多次更新（如果有*N*个副本，则平均更新*N* - 1次），并且无法保证计算出的梯度仍然指向正确的方向（参见[图19-16](#stale_gradients_diagram)）。当梯度严重过时时，它们被称为*过时梯度*：它们可以减慢收敛速度，引入噪声和摆动效应（学习曲线可能包含临时振荡），甚至可能使训练算法发散。

![mls3 1916](assets/mls3_1916.png)

###### 图19-16。使用异步更新时的过时梯度

有几种方法可以减少陈旧梯度的影响：

+   降低学习率。

+   丢弃陈旧的梯度或将其缩小。

+   调整小批量大小。

+   在开始的几个时期只使用一个副本（这被称为*热身阶段*）。在训练开始阶段，梯度通常很大，参数还没有稳定在成本函数的谷底，因此陈旧的梯度可能会造成更大的损害，不同的副本可能会将参数推向完全不同的方向。

2016年，Google Brain团队发表的一篇论文对各种方法进行了基准测试，发现使用同步更新和一些备用副本比使用异步更新更有效，不仅收敛更快，而且产生了更好的模型。然而，这仍然是一个活跃的研究领域，所以你不应该立刻排除异步更新。

### 带宽饱和

无论您使用同步还是异步更新，具有集中参数的数据并行仍然需要在每个训练步骤开始时将模型参数从参数服务器传递到每个副本，并在每个训练步骤结束时将梯度传递到另一个方向。同样，当使用镜像策略时，每个GPU生成的梯度将需要与每个其他GPU共享。不幸的是，通常会出现这样一种情况，即添加额外的GPU将不会改善性能，因为将数据移入和移出GPU RAM（以及在分布式设置中跨网络）所花费的时间将超过通过分割计算负载获得的加速效果。在那一点上，添加更多的GPU将只会加剧带宽饱和，并实际上减慢训练速度。

饱和对于大型密集模型来说更严重，因为它们有很多参数和梯度需要传输。对于小型模型来说，饱和程度较轻（但并行化增益有限），对于大型稀疏模型也较轻，因为梯度通常大部分为零，可以有效传输。Google Brain 项目的发起人和负责人 Jeff Dean [报告](https://homl.info/69) 在将计算分布到 50 个 GPU 上时，密集模型的典型加速为 25-40 倍，而在 500 个 GPU 上训练稀疏模型时，加速为 300 倍。正如你所看到的，稀疏模型确实更好地扩展。以下是一些具体例子：

+   神经机器翻译：在 8 个 GPU 上加速 6 倍

+   Inception/ImageNet：在 50 个 GPU 上加速 32 倍

+   RankBrain：在 500 个 GPU 上加速 300 倍

有很多研究正在进行，以缓解带宽饱和问题，目标是使训练能够与可用的GPU数量成线性比例扩展。例如，卡内基梅隆大学、斯坦福大学和微软研究团队在2018年提出了一个名为*PipeDream*的系统，成功将网络通信减少了90%以上，使得可以在多台机器上训练大型模型成为可能。他们使用了一种称为*管道并行*的新技术来实现这一目标，该技术结合了模型并行和数据并行：模型被切分成连续的部分，称为*阶段*，每个阶段在不同的机器上进行训练。这导致了一个异步的管道，所有机器都在很少的空闲时间内并行工作。在训练过程中，每个阶段交替进行一轮前向传播和一轮反向传播：它从输入队列中提取一个小批量数据，处理它，并将输出发送到下一个阶段的输入队列，然后从梯度队列中提取一个小批量的梯度，反向传播这些梯度并更新自己的模型参数，并将反向传播的梯度推送到前一个阶段的梯度队列。然后它一遍又一遍地重复整个过程。每个阶段还可以独立地使用常规的数据并行（例如使用镜像策略），而不受其他阶段的影响。

![mls3 1917](assets/mls3_1917.png)

###### 图19-17。PipeDream的管道并行性

然而，正如在这里展示的那样，PipeDream 不会工作得那么好。要理解原因，考虑在 [Figure 19-17](#pipedream_diagram) 中的第 5 个小批次：当它在前向传递过程中经过第 1 阶段时，来自第 4 个小批次的梯度尚未通过该阶段进行反向传播，但是当第 5 个小批次的梯度流回到第 1 阶段时，第 4 个小批次的梯度将已经被用来更新模型参数，因此第 5 个小批次的梯度将有点过时。正如我们所看到的，这可能会降低训练速度和准确性，甚至使其发散：阶段越多，这个问题就会变得越糟糕。论文的作者提出了缓解这个问题的方法：例如，每个阶段在前向传播过程中保存权重，并在反向传播过程中恢复它们，以确保相同的权重用于前向传递和反向传递。这被称为*权重存储*。由于这一点，PipeDream 展示了令人印象深刻的扩展能力，远远超出了简单的数据并行性。

这个研究领域的最新突破是由谷歌研究人员在一篇[2022年的论文](https://homl.info/pathways)中发表的：他们开发了一个名为*Pathways*的系统，利用自动模型并行、异步团队调度等技术，实现了数千个TPU几乎100%的硬件利用率！*调度*意味着组织每个任务必须运行的时间和位置，*团队调度*意味着同时并行运行相关任务，并且彼此靠近，以减少任务等待其他任务输出的时间。正如我们在[第16章](ch16.html#nlp_chapter)中看到的，这个系统被用来在超过6,000个TPU上训练一个庞大的语言模型，几乎实现了100%的硬件利用率：这是一个令人惊叹的工程壮举。

在撰写本文时，Pathways尚未公开，但很可能在不久的将来，您将能够使用Pathways或类似系统在Vertex AI上训练大型模型。与此同时，为了减少饱和问题，您可能会希望使用一些强大的GPU，而不是大量的弱GPU，如果您需要在多台服务器上训练模型，您应该将GPU分组在少数且连接非常良好的服务器上。您还可以尝试将浮点精度从32位（`tf.float32`）降低到16位（`tf.bfloat16`）。这将减少一半的数据传输量，通常不会对收敛速度或模型性能产生太大影响。最后，如果您正在使用集中式参数，您可以将参数分片（分割）到多个参数服务器上：增加更多的参数服务器将减少每个服务器上的网络负载，并限制带宽饱和的风险。

好的，现在我们已经讨论了所有的理论，让我们实际在多个GPU上训练一个模型！

## 使用分布策略API进行规模训练

幸运的是，TensorFlow带有一个非常好的API，它负责处理将模型分布在多个设备和机器上的所有复杂性：*分布策略API*。要在所有可用的GPU上（暂时只在单台机器上）使用数据并行性和镜像策略训练一个Keras模型，只需创建一个`MirroredStrategy`对象，调用它的`scope()`方法以获取一个分布上下文，并将模型的创建和编译包装在该上下文中。然后正常调用模型的`fit()`方法：

```py
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([...])  # create a Keras model normally
    model.compile([...])  # compile the model normally

batch_size = 100  # preferably divisible by the number of replicas
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_valid, y_valid), batch_size=batch_size)
```

在底层，Keras是分布感知的，因此在这个`MirroredStrategy`上下文中，它知道必须在所有可用的GPU设备上复制所有变量和操作。如果你查看模型的权重，它们是`MirroredVariable`类型的：

```py
>>> type(model.weights[0])
tensorflow.python.distribute.values.MirroredVariable
```

请注意，`fit()` 方法会自动将每个训练批次在所有副本之间进行分割，因此最好确保批次大小可以被副本数量（即可用的 GPU 数量）整除，以便所有副本获得相同大小的批次。就是这样！训练通常会比使用单个设备快得多，而且代码更改确实很小。

训练模型完成后，您可以使用它高效地进行预测：调用`predict()`方法，它会自动将批处理在所有副本之间分割，以并行方式进行预测。再次强调，批处理大小必须能够被副本数量整除。如果调用模型的`save()`方法，它将被保存为常规模型，*而不是*具有多个副本的镜像模型。因此，当您加载它时，它将像常规模型一样运行，在单个设备上：默认情况下在GPU＃0上，如果没有GPU则在CPU上。如果您想加载一个模型并在所有可用设备上运行它，您必须在分发上下文中调用`tf.keras.models.load_model()`：

```py
with strategy.scope():
    model = tf.keras.models.load_model("my_mirrored_model")
```

如果您只想使用所有可用GPU设备的子集，您可以将列表传递给`MirroredStrategy`的构造函数：

```py
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
```

默认情况下，`MirroredStrategy`类使用*NVIDIA Collective Communications Library*（NCCL）进行AllReduce均值操作，但您可以通过将`cross_device_ops`参数设置为`tf.distribute.HierarchicalCopyAllReduce`类的实例或`tf.distribute.ReductionToOneDevice`类的实例来更改它。默认的NCCL选项基于`tf.distribute.NcclAllReduce`类，通常更快，但这取决于GPU的数量和类型，因此您可能想尝试一下其他选择。

如果您想尝试使用集中式参数的数据并行性，请将`MirroredStrategy`替换为`CentralStorageStrategy`：

```py
strategy = tf.distribute.experimental.CentralStorageStrategy()
```

您可以选择设置`compute_devices`参数来指定要用作工作器的设备列表-默认情况下将使用所有可用的GPU-您还可以选择设置`parameter_device`参数来指定要存储参数的设备。默认情况下将使用CPU，或者如果只有一个GPU，则使用GPU。

现在让我们看看如何在一组TensorFlow服务器上训练模型！

## 在TensorFlow集群上训练模型

*TensorFlow集群*是一组在并行运行的TensorFlow进程，通常在不同的机器上，并相互通信以完成一些工作，例如训练或执行神经网络模型。集群中的每个TF进程被称为*任务*或*TF服务器*。它有一个IP地址，一个端口和一个类型（也称为*角色*或*工作*）。类型可以是`"worker"`、`"chief"`、`"ps"`（参数服务器）或`"evaluator"`：

+   每个*worker*执行计算，通常在一台或多台GPU的机器上。

+   首席执行计算任务（它是一个工作者），但也处理额外的工作，比如编写TensorBoard日志或保存检查点。集群中只有一个首席。如果没有明确指定首席，则按照惯例第一个工作者就是首席。

+   参数服务器只跟踪变量值，并且通常在仅有CPU的机器上。这种类型的任务只能与`ParameterServerStrategy`一起使用。

+   评估者显然负责评估。这种类型并不经常使用，当使用时，通常只有一个评估者。

要启动一个TensorFlow集群，必须首先定义其规范。这意味着定义每个任务的IP地址、TCP端口和类型。例如，以下*集群规范*定义了一个有三个任务的集群（两个工作者和一个参数服务器；参见[图19-18](#cluster_diagram)）。集群规范是一个字典，每个作业对应一个键，值是任务地址（*IP*:*port*）的列表：

```py
cluster_spec = {
    "worker": [
        "machine-a.example.com:2222",     # /job:worker/task:0
        "machine-b.example.com:2222"      # /job:worker/task:1
    ],
    "ps": ["machine-a.example.com:2221"]  # /job:ps/task:0
}
```

通常每台机器上会有一个任务，但正如这个示例所示，如果需要，您可以在同一台机器上配置多个任务。在这种情况下，如果它们共享相同的GPU，请确保RAM适当分配，如前面讨论的那样。

###### 警告

默认情况下，集群中的每个任务可以与其他任务通信，因此请确保配置防火墙以授权这些机器之间这些端口上的所有通信（如果每台机器使用相同的端口，则通常更简单）。

![mls3 1918](assets/mls3_1918.png)

###### 图19-18。一个示例TensorFlow集群

当您开始一个任务时，您必须给它指定集群规范，并且还必须告诉它它的类型和索引是什么（例如，worker #0）。一次性指定所有内容的最简单方法（包括集群规范和当前任务的类型和索引）是在启动TensorFlow之前设置`TF_CONFIG`环境变量。它必须是一个JSON编码的字典，包含集群规范（在`"cluster"`键下）和当前任务的类型和索引（在`"task"`键下）。例如，以下`TF_CONFIG`环境变量使用我们刚刚定义的集群，并指定要启动的任务是worker #0：

```py
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": cluster_spec,
    "task": {"type": "worker", "index": 0}
})
```

###### 提示

通常您希望在Python之外定义`TF_CONFIG`环境变量，这样代码就不需要包含当前任务的类型和索引（这样可以在所有工作节点上使用相同的代码）。

现在让我们在集群上训练一个模型！我们将从镜像策略开始。首先，您需要为每个任务适当设置`TF_CONFIG`环境变量。集群规范中不应该有参数服务器（删除集群规范中的`"ps"`键），通常每台机器上只需要一个工作节点。确保为每个任务设置不同的任务索引。最后，在每个工作节点上运行以下脚本：

```py
import tempfile
import tensorflow as tf

strategy = tf.distribute.MultiWorkerMirroredStrategy()  # at the start!
resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
print(f"Starting task {resolver.task_type} #{resolver.task_id}")
[...] # load and split the MNIST dataset

with strategy.scope():
    model = tf.keras.Sequential([...])  # build the Keras model
    model.compile([...])  # compile the model

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10)

if resolver.task_id == 0:  # the chief saves the model to the right location
    model.save("my_mnist_multiworker_model", save_format="tf")
else:
    tmpdir = tempfile.mkdtemp()  # other workers save to a temporary directory
    model.save(tmpdir, save_format="tf")
    tf.io.gfile.rmtree(tmpdir)  # and we can delete this directory at the end!
```

这几乎是您之前使用的相同代码，只是这次您正在使用`MultiWorkerMirroredStrategy`。当您在第一个工作节点上启动此脚本时，它们将在AllReduce步骤处保持阻塞，但是一旦最后一个工作节点启动，训练将开始，并且您将看到它们以完全相同的速度前进，因为它们在每一步都进行同步。

###### 警告

在使用`MultiWorkerMirroredStrategy`时，重要的是确保所有工作人员做同样的事情，包括保存模型检查点或编写TensorBoard日志，即使您只保留主要写入的内容。这是因为这些操作可能需要运行AllReduce操作，因此所有工作人员必须保持同步。

这个分发策略有两种AllReduce实现方式：基于gRPC的环形AllReduce算法用于网络通信，以及NCCL的实现。要使用哪种最佳算法取决于工作人员数量、GPU数量和类型，以及网络情况。默认情况下，TensorFlow会应用一些启发式方法为您选择合适的算法，但您可以强制使用NCCL（或RING）如下：

```py
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CollectiveCommunication.NCCL))
```

如果您希望使用参数服务器实现异步数据并行处理，请将策略更改为`ParameterServerStrategy`，添加一个或多个参数服务器，并为每个任务适当配置`TF_CONFIG`。请注意，虽然工作人员将异步工作，但每个工作人员上的副本将同步工作。

最后，如果您可以访问[Google Cloud上的TPU](https://cloud.google.com/tpu)——例如，如果您在Colab中设置加速器类型为TPU——那么您可以像这样创建一个`TPUStrategy`：

```py
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
```

这需要在导入TensorFlow后立即运行。然后您可以正常使用这个策略。

###### 提示

如果您是研究人员，您可能有资格免费使用TPU；请查看[*https://tensorflow.org/tfrc*](https://tensorflow.org/tfrc)获取更多详细信息。

现在您可以跨多个GPU和多个服务器训练模型：给自己一个鼓励！然而，如果您想训练一个非常大的模型，您将需要许多GPU，跨多个服务器，这将要求要么购买大量硬件，要么管理大量云虚拟机。在许多情况下，使用一个云服务来为您提供所有这些基础设施的配置和管理会更方便、更经济，只有在您需要时才会提供。让我们看看如何使用Vertex AI来实现这一点。

## 在Vertex AI上运行大型训练作业

Vertex AI允许您使用自己的训练代码创建自定义训练作业。实际上，您可以几乎使用与在自己的TF集群上使用的相同的训练代码。您必须更改的主要内容是首席应该保存模型、检查点和TensorBoard日志的位置。首席必须将模型保存到GCS，使用Vertex AI在`AIP_MODEL_DIR`环境变量中提供的路径，而不是将模型保存到本地目录。对于模型检查点和TensorBoard日志，您应该分别使用`AIP_CHECKPOINT_DIR`和`AIP_TENSORBOARD_LOG_DIR`环境变量中包含的路径。当然，您还必须确保训练数据可以从虚拟机访问，例如在GCS上，或者从另一个GCP服务（如BigQuery）或直接从网络上访问。最后，Vertex AI明确设置了`"chief"`任务类型，因此您应该使用`resolved.task_type == "chief"`来识别首席，而不是使用`resolved.task_id == 0`：

```py
import os
[...]  # other imports, create MultiWorkerMirroredStrategy, and resolver

if resolver.task_type == "chief":
    model_dir = os.getenv("AIP_MODEL_DIR")  # paths provided by Vertex AI
    tensorboard_log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR")
    checkpoint_dir = os.getenv("AIP_CHECKPOINT_DIR")
else:
    tmp_dir = Path(tempfile.mkdtemp())  # other workers use temporary dirs
    model_dir = tmp_dir / "model"
    tensorboard_log_dir = tmp_dir / "logs"
    checkpoint_dir = tmp_dir / "ckpt"

callbacks = [tf.keras.callbacks.TensorBoard(tensorboard_log_dir),
             tf.keras.callbacks.ModelCheckpoint(checkpoint_dir)]
[...]  # build and  compile using the strategy scope, just like earlier
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10,
          callbacks=callbacks)
model.save(model_dir, save_format="tf")
```

###### 提示

如果您将训练数据放在GCS上，您可以创建一个`tf.data.TextLineDataset`或`tf.data.TFRecordDataset`来访问它：只需将GCS路径作为文件名（例如，*gs://my_bucket/data/001.csv*）。这些数据集依赖于`tf.io.gfile`包来访问文件：它支持本地文件和GCS文件。

现在您可以在Vertex AI上基于这个脚本创建一个自定义训练作业。您需要指定作业名称、训练脚本的路径、用于训练的Docker镜像、用于预测的镜像（训练后）、您可能需要的任何其他Python库，以及最后Vertex AI应该使用作为存储训练脚本的暂存目录的存储桶。默认情况下，这也是训练脚本将保存训练模型、TensorBoard日志和模型检查点（如果有的话）的地方。让我们创建这个作业：

```py
custom_training_job = aiplatform.CustomTrainingJob(
    display_name="my_custom_training_job",
    script_path="my_vertex_ai_training_task.py",
    container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest",
    model_serving_container_image_uri=server_image,
    requirements=["gcsfs==2022.3.0"],  # not needed, this is just an example
    staging_bucket=f"gs://{bucket_name}/staging"
)
```

现在让我们在两个拥有两个GPU的工作节点上运行它：

```py
mnist_model2 = custom_training_job.run(
    machine_type="n1-standard-4",
    replica_count=2,
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=2,
)
```

这就是全部内容：Vertex AI将为您请求的计算节点进行配置（在您的配额范围内），并在这些节点上运行您的训练脚本。一旦作业完成，`run()`方法将返回一个经过训练的模型，您可以像之前创建的那样使用它：您可以部署到端点，或者用它进行批量预测。如果在训练过程中出现任何问题，您可以在GCP控制台中查看日志：在☰导航菜单中，选择Vertex AI → 训练，点击您的训练作业，然后点击查看日志。或者，您可以点击自定义作业选项卡，复制作业的ID（例如，1234），然后从☰导航菜单中选择日志记录，并查询`resource.labels.job_id=1234`。

###### 提示

要可视化训练进度，只需启动TensorBoard，并将其`--logdir`指向日志的GCS路径。它将使用*应用程序默认凭据*，您可以使用`gcloud auth application-default login`进行设置。如果您喜欢，Vertex AI还提供托管的TensorBoard服务器。

如果您想尝试一些超参数值，一个选项是运行多个作业。您可以通过在调用`run()`方法时设置`args`参数将超参数值作为命令行参数传递给您的脚本，或者您可以使用`environment_variables`参数将它们作为环境变量传递。

然而，如果您想在云上运行一个大型的超参数调整作业，一个更好的选择是使用Vertex AI的超参数调整服务。让我们看看如何做。

## Vertex AI上的超参数调整

Vertex AI的超参数调整服务基于贝叶斯优化算法，能够快速找到最佳的超参数组合。要使用它，首先需要创建一个接受超参数值作为命令行参数的训练脚本。例如，您的脚本可以像这样使用`argparse`标准库：

```py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_hidden", type=int, default=2)
parser.add_argument("--n_neurons", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--optimizer", default="adam")
args = parser.parse_args()
```

超参数调整服务将多次调用您的脚本，每次使用不同的超参数值：每次运行称为*trial*，一组试验称为*study*。然后，您的训练脚本必须使用给定的超参数值来构建和编译模型。如果需要，您可以使用镜像分发策略，以便每个试验在多GPU机器上运行。然后脚本可以加载数据集并训练模型。例如：

```py
import tensorflow as tf

def build_model(args):
    with tf.distribute.MirroredStrategy().scope():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=[28, 28], dtype=tf.uint8))
        for _ in range(args.n_hidden):
            model.add(tf.keras.layers.Dense(args.n_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(10, activation="softmax"))
        opt = tf.keras.optimizers.get(args.optimizer)
        opt.learning_rate = args.learning_rate
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        return model

[...]  # load the dataset
model = build_model(args)
history = model.fit([...])
```

###### 提示

您可以使用我们之前提到的`AIP_*`环境变量来确定在哪里保存检查点、TensorBoard日志和最终模型。

最后，脚本必须将模型的性能报告给Vertex AI的超参数调整服务，以便它决定尝试哪些超参数。为此，您必须使用`hypertune`库，在Vertex AI训练VM上自动安装：

```py
import hypertune

hypertune = hypertune.HyperTune()
hypertune.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="accuracy",  # name of the reported metric
    metric_value=max(history.history["val_accuracy"]),  # metric value
    global_step=model.optimizer.iterations.numpy(),
)
```

现在您的训练脚本已准备就绪，您需要定义要在其上运行的机器类型。为此，您必须定义一个自定义作业，Vertex AI 将使用它作为每个试验的模板：

```py
trial_job = aiplatform.CustomJob.from_local_script(
    display_name="my_search_trial_job",
    script_path="my_vertex_ai_trial.py",  # path to your training script
    container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest",
    staging_bucket=f"gs://{bucket_name}/staging",
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=2,  # in this example, each trial will have 2 GPUs
)
```

最后，您准备好创建并运行超参数调整作业：

```py
from google.cloud.aiplatform import hyperparameter_tuning as hpt

hp_job = aiplatform.HyperparameterTuningJob(
    display_name="my_hp_search_job",
    custom_job=trial_job,
    metric_spec={"accuracy": "maximize"},
    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(min=1e-3, max=10, scale="log"),
        "n_neurons": hpt.IntegerParameterSpec(min=1, max=300, scale="linear"),
        "n_hidden": hpt.IntegerParameterSpec(min=1, max=10, scale="linear"),
        "optimizer": hpt.CategoricalParameterSpec(["sgd", "adam"]),
    },
    max_trial_count=100,
    parallel_trial_count=20,
)
hp_job.run()
```

在这里，我们告诉 Vertex AI 最大化名为 `"accuracy"` 的指标：这个名称必须与训练脚本报告的指标名称匹配。我们还定义了搜索空间，使用对数尺度来设置学习率，使用线性（即均匀）尺度来设置其他超参数。超参数的名称必须与训练脚本的命令行参数匹配。然后我们将最大试验次数设置为 100，同时最大并行运行的试验次数设置为 20。如果你将并行试验的数量增加到（比如）60，总搜索时间将显著减少，最多可减少到 3 倍。但前 60 个试验将同时开始，因此它们将无法从其他试验的反馈中受益。因此，您应该增加最大试验次数来补偿，例如增加到大约 140。

这将需要相当长的时间。一旦作业完成，您可以使用 `hp_job.trials` 获取试验结果。每个试验结果都表示为一个protobuf对象，包含超参数值和结果指标。让我们找到最佳试验：

```py
def get_final_metric(trial, metric_id):
    for metric in trial.final_measurement.metrics:
        if metric.metric_id == metric_id:
            return metric.value

trials = hp_job.trials
trial_accuracies = [get_final_metric(trial, "accuracy") for trial in trials]
best_trial = trials[np.argmax(trial_accuracies)]
```

现在让我们看看这个试验的准确率，以及其超参数值：

```py
>>> max(trial_accuracies)
0.977400004863739
>>> best_trial.id
'98'
>>> best_trial.parameters
[parameter_id: "learning_rate" value { number_value: 0.001 },
 parameter_id: "n_hidden" value { number_value: 8.0 },
 parameter_id: "n_neurons" value { number_value: 216.0 },
 parameter_id: "optimizer" value { string_value: "adam" }
]
```

就是这样！现在您可以获取这个试验的SavedModel，可选择性地再训练一下，并将其部署到生产环境中。

###### 提示

Vertex AI还包括一个AutoML服务，完全负责为您找到合适的模型架构并为您进行训练。您只需要将数据集以特定格式上传到Vertex AI，这取决于数据集的类型（图像、文本、表格、视频等），然后创建一个AutoML训练作业，指向数据集并指定您愿意花费的最大计算小时数。请参阅笔记本中的示例。

现在你拥有了所有需要创建最先进的神经网络架构并使用各种分布策略进行规模化训练的工具和知识，可以在自己的基础设施或云上部署它们，然后在任何地方部署它们。换句话说，你现在拥有超能力：好好利用它们！

# 练习

1.  SavedModel包含什么？如何检查其内容？

1.  什么时候应该使用TF Serving？它的主要特点是什么？有哪些工具可以用来部署它？

1.  如何在多个TF Serving实例上部署模型？

1.  在查询由TF Serving提供的模型时，何时应该使用gRPC API而不是REST API？

1.  TFLite通过哪些不同的方式减小模型的大小，使其能在移动设备或嵌入式设备上运行？

1.  什么是量化感知训练，为什么需要它？

1.  什么是模型并行和数据并行？为什么通常推荐后者？

1.  在多台服务器上训练模型时，您可以使用哪些分发策略？您如何选择使用哪种？

1.  训练一个模型（任何您喜欢的模型）并部署到TF Serving或Google Vertex AI。编写客户端代码，使用REST API或gRPC API查询它。更新模型并部署新版本。您的客户端代码现在将查询新版本。回滚到第一个版本。

1.  在同一台机器上使用`MirroredStrategy`在多个GPU上训练任何模型（如果您无法访问GPU，可以使用带有GPU运行时的Google Colab并创建两个逻辑GPU）。再次使用`CentralStorageStrategy`训练模型并比较训练时间。

1.  在Vertex AI上微调您选择的模型，使用Keras Tuner或Vertex AI的超参数调整服务。

这些练习的解决方案可以在本章笔记本的末尾找到，网址为[*https://homl.info/colab3*](https://homl.info/colab3)。

# 谢谢！

在我们结束这本书的最后一章之前，我想感谢您读到最后一段。我真诚地希望您阅读这本书和我写作时一样开心，并且它对您的项目，无论大小，都有用。

如果您发现错误，请发送反馈。更一般地，我很想知道您的想法，所以请不要犹豫通过O'Reilly、*ageron/handson-ml3* GitHub项目或Twitter上的@aureliengeron与我联系。

继续前进，我给你的最好建议是练习和练习：尝试完成所有的练习（如果你还没有这样做），玩一下笔记本电脑，加入Kaggle或其他机器学习社区，观看机器学习课程，阅读论文，参加会议，与专家会面。事情发展迅速，所以尽量保持最新。一些YouTube频道定期以非常易懂的方式详细介绍深度学习论文。我特别推荐Yannic Kilcher、Letitia Parcalabescu和Xander Steenbrugge的频道。要了解引人入胜的机器学习讨论和更高层次的见解，请务必查看ML Street Talk和Lex Fridman的频道。拥有一个具体的项目要去做也会极大地帮助，无论是为了工作还是为了娱乐（最好两者兼顾），所以如果你一直梦想着建造某样东西，就试一试吧！逐步工作；不要立即朝着月球开火，而是专注于你的项目，一步一步地构建它。这需要耐心和毅力，但当你拥有一个行走的机器人，或一个工作的聊天机器人，或者其他你喜欢的任何东西时，这将是极其有益的！

我最大的希望是这本书能激发你构建一个美妙的ML应用程序，使我们所有人受益。它会是什么样的？

—*Aurélien Géron*

^([1](ch19.html#idm45720162442960-marker)) A/B实验包括在不同的用户子集上测试产品的两个不同版本，以检查哪个版本效果最好并获得其他见解。

^([2](ch19.html#idm45720162441360-marker)) Google AI平台（以前称为Google ML引擎）和Google AutoML在2021年合并为Google Vertex AI。

^([3](ch19.html#idm45720162435760-marker)) REST（或RESTful）API是一种使用标准HTTP动词（如GET、POST、PUT和DELETE）以及使用JSON输入和输出的API。gRPC协议更复杂但更高效；数据使用协议缓冲区进行交换（参见[第13章](ch13.html#data_chapter)）。

如果您对Docker不熟悉，它允许您轻松下载一组打包在*Docker镜像*中的应用程序（包括所有依赖项和通常一些良好的默认配置），然后使用*Docker引擎*在您的系统上运行它们。当您运行一个镜像时，引擎会创建一个保持应用程序与您自己系统良好隔离的*Docker容器*，但如果您愿意，可以给它一些有限的访问权限。它类似于虚拟机，但速度更快、更轻，因为容器直接依赖于主机的内核。这意味着镜像不需要包含或运行自己的内核。

还有GPU镜像可用，以及其他安装选项。有关更多详细信息，请查看官方[安装说明](https://homl.info/tfserving)。

公平地说，这可以通过首先序列化数据，然后将其编码为Base64，然后创建REST请求来减轻。此外，REST请求可以使用gzip进行压缩，从而显著减少有效负载大小。

还要查看TensorFlow的[Graph Transform Tool](https://homl.info/tfgtt)，用于修改和优化计算图。

例如，PWA必须包含不同移动设备大小的图标，必须通过HTTPS提供，必须包含包含应用程序名称和背景颜色等元数据的清单文件。

请查看TensorFlow文档，获取详细和最新的安装说明，因为它们经常更改。

^([10](ch19.html#idm45720159934800-marker)) 正如我们在[第12章](ch12.html#tensorflow_chapter)中所看到的，内核是特定数据类型和设备类型的操作实现。例如，`float32` `tf.matmul()` 操作有一个 GPU 内核，但 `int32` `tf.matmul()` 没有 GPU 内核，只有一个 CPU 内核。

^([11](ch19.html#idm45720159930608-marker)) 您还可以使用 `tf.debugging.set_log_device_placement(True)` 来记录所有设备放置情况。

^([12](ch19.html#idm45720159777008-marker)) 如果您想要保证完美的可重现性，这可能很有用，正如我在[这个视频](https://homl.info/repro)中所解释的，基于 TF 1。

^([13](ch19.html#idm45720159772080-marker)) 在撰写本文时，它只是将数据预取到 CPU RAM，但使用 `tf.data.experimental.pre⁠fetch​_to_device()` 可以使其预取数据并将其推送到您选择的设备，以便 GPU 不必等待数据传输而浪费时间。

如果两个CNN相同，则称为*孪生神经网络*。

如果您对模型并行性感兴趣，请查看[Mesh TensorFlow](https://github.com/tensorflow/mesh)。

这个名字有点令人困惑，因为听起来好像有些副本是特殊的，什么也不做。实际上，所有副本都是等价的：它们都努力成为每个训练步骤中最快的，失败者在每一步都会变化（除非某些设备真的比其他设备慢）。但是，这意味着如果一个或两个服务器崩溃，训练将继续进行得很好。

Jianmin Chen等人，“重新审视分布式同步SGD”，arXiv预印本arXiv:1604.00981（2016）。

^([18](ch19.html#idm45720159675552-marker)) Aaron Harlap等人，“PipeDream: 快速高效的管道并行DNN训练”，arXiv预印本arXiv:1806.03377（2018）。

^([19](ch19.html#idm45720159666736-marker)) Paul Barham等人，“Pathways: 异步分布式数据流ML”，arXiv预印本arXiv:2203.12533（2022）。

^([20](ch19.html#idm45720159438096-marker)) 有关AllReduce算法的更多详细信息，请阅读Yuichiro Ueno的文章，该文章介绍了深度学习背后的技术，以及Sylvain Jeaugey的文章，该文章介绍了如何使用NCCL大规模扩展深度学习训练。
