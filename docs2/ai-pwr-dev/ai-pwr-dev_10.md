# 7 编码基础设施和管理部署

本章涵盖

+   在 Copilot 的帮助下创建 Dockerfile

+   使用大型语言模型起草基础设施代码

+   使用容器注册库管理 Docker 镜像

+   利用 Kubernetes 的力量

+   使用 GitHub Actions 无缝发布您的代码

没有什么比一个应用程序闲置不用更令人沮丧的了。因此，快速将经过良好测试的应用程序推向生产是每个合格开发者的目标声明。因为我们已经在上一章测试了我们的产品，所以它现在可以准备发布了。

本章将重点关注从开发到产品发布的那个关键时刻。在这个关键阶段，理解部署策略和最佳实践变得至关重要，以确保产品发布成功。

在我们的应用程序成功得到保护和测试后，是时候将我们的注意力转向产品的发布了。为此，我们将利用大型语言模型（LLMs）强大的功能来探索针对云基础设施的定制部署选项。

通过利用大型语言模型（LLMs）的力量并接受它们的部署选项和方法，我们可以自信地穿梭于启动产品的复杂领域，向客户交付一个强大且可扩展的解决方案，同时利用云计算的好处。

首先，我们将开发 Docker 的部署文件。我们将探讨如何创建 Docker 镜像并定义部署文件。此外，我们还将讨论容器化我们的应用程序和实现无缝部署的最佳实践。

接下来，我们将使用 Terraform 来定义我们的基础设施代码，并自动化在亚马逊网络服务（AWS）上部署弹性计算云（EC2）实例。我们将演示如何编写 Terraform 脚本来在 EC2 实例上配置和部署我们的应用程序，确保基础设施设置的一致性和可重复性。

然后，我们将利用大型语言模型（LLMs）将我们的应用程序部署到 Kubernetes（AWS Elastic Kubernetes Service [EKS]/Elastic Container Service [ECS]）。我们将让 GitHub Copilot 创建适当的 Kubernetes 部署文件，以简化我们的部署流程并高效管理应用程序的生命周期。鉴于我们的应用程序相对简单，我们不需要像 Helm 这样的 Kubernetes 包管理器。然而，随着服务和依赖的复杂性和增长，你可能希望将其作为选项之一进行探索。幸运的是，Copilot 还可以为你编写 Helm 图表！

最后，我们将简要展示如何使用 GitHub actions 从本地迁移到自动化部署。通过将 LLMs 与这个广泛使用的持续集成和持续部署（CI/CD）工具集成，我们可以自动化构建和部署流程，确保更快、更高效的部署。

注意：本章使用 AWS 作为我们的云服务提供商，但本章中涵盖的原则和实践可以适应并应用于其他云平台，甚至在没有虚拟化（裸金属）的本地基础设施上，使我们能够根据业务需求的变化调整和扩展产品部署策略。你会发现，通过采用大型语言模型（LLMs）和使用基础设施即代码（infrastructure as code），你可以（部分地）减轻云平台非常常见的供应商锁定问题。

注意，如果您选择将此（或任何应用程序）部署到 AWS，您的活动将产生相关费用。AWS 和大多数云服务提供商提供免费试用以学习他们的平台（例如 Google Cloud Platform 和 Azure），但一旦这些信用额度到期，您可能会收到一个相当意外的账单。如果您决定跟随本章的内容，您需要设置一个您能舒适承担的阈值警报。Andreas Wittig 和 Michael Wittig 的*Amazon Web Services in Action, Third Edition*（Manning，2023；[www.manning.com/books/amazon-web-services-in-action-third-edition](https://www.manning.com/books/amazon-web-services-in-action-third-edition)）的第 1.9 节是设置此类计费通知警报的极好资源。

## 7.1 构建 Docker 镜像并在本地“部署”

如您从第六章可能记得的那样，Docker 是一个容器化平台，允许您在传统意义上几乎不需要安装应用程序（除了 Docker 之外）的情况下运行应用程序。与模拟整个操作系统的虚拟机不同，容器共享宿主系统的内核（操作系统的核心部分）并使用宿主系统的操作系统功能，同时将应用程序进程和文件系统与宿主系统隔离。这使得您可以在单个宿主系统上运行多个隔离的应用程序，每个应用程序都有自己的环境和资源限制。图 7.1 展示了 Docker 运行时与宿主之间的关系。

![图 7.1 Docker 容器与宿主系统关系图](img/CH07_F01_Crocker2.png)

图 7.1 Docker 利用宿主操作系统的功能，同时隔离每个容器。这使得与虚拟机相比，Docker 容器更轻量，因为它们不需要完整的操作系统来运行。

从生产准备的角度来看，其中一个更令人兴奋的功能是，Docker 使得运行某些意义上可以自我修复的应用程序变得更加容易。如果它们在运行时失败或崩溃，您可以配置它们在无需干预的情况下重启。在本节中，我们将使用 Copilot 创建一个文件（称为*Dockerfile*），我们将从这个文件构建我们的*Docker 镜像*。

定义*Docker 镜像*就像 Docker 容器的蓝图。它们是可移植的，包括应用程序运行所需的所有依赖项（库、环境变量、代码等）。

正在运行的 Docker 实例被称为 Docker *容器*。鉴于它们的轻量级特性，我们可以在单个主机上运行多个容器而不会出现问题。我们可以这样做，因为容器化技术共享 OS 内核，在隔离的用户空间中运行。

注意：最初，我想使用 AWS CodeWhisperer 作为本章的 LLM。鉴于预期的云平台，这似乎是合理的。然而，在撰写本文时，AWS CodeWhisperer 仅支持编程语言编程。它没有基础设施即代码的功能。

我们将使用以下提示来让 Copilot 为我们草拟 Dockerfile：

```py
# Create a Dockerfile for this Python app. The main class is main.py. Use Python 3.10 and install
# the dependencies using the requirements.txt file in this directory. The app should run on port 8080.
```

您可能只剩下一个空文件（除了这个注释）。基础设施即代码的支持是不断发展的（与一般的 LLM 生态系统类似）。根据 Copilot Chat，GitHub Copilot 能够为您创建 Dockerfile——但是您必须通过以下步骤来激励它：

1.  在 Dockerfile 中输入`FROM python:`并等待 Copilot 建议要使用的 Python 版本。选择您想要使用的版本。

1.  输入`WORKDIR /app`以设置容器的当前工作目录。

1.  输入`COPY . /app`以将项目内容复制到容器中。

1.  输入`RUN pip install --trusted-host pypi.python.org -r requirements.txt`以安装项目的依赖项。

1.  输入`EXPOSE 8080`以暴露容器的 8080 端口。

1.  输入`CMD ["python", "main.py"]`以指定容器启动时运行的命令。

或者，您可能希望将之前写入 Dockerfile 中的相同提示复制粘贴到 Copilot Chat 的提示窗口中。Copilot Chat 将为您提供所需的 Dockerfile 内容。

列表 7.1 构建 Docker 镜像的 Dockerfile

```py
FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 8080
CMD ["python", "main.py"]
```

使用 Dockerfile，我们将构建用于部署和运行我们应用程序的镜像。我们可以输入以下命令来构建我们的应用程序（从 Dockerfile 所在的目录运行，并且不要忘记最后的点）。您需要互联网访问来下载依赖项并创建镜像：

```py
docker build -t itam:latest .
```

构建 Docker 镜像可能需要几秒钟到几分钟，具体取决于您的系统上安装了哪些镜像和包以及您的互联网连接速度。您的耐心将得到回报，因为您将很快拥有一个可以在几乎任何地方安装的应用程序，从最基础的商品硬件到您最喜欢的云提供商提供的最大型硬件。然而，在运行之前，您需要尝试在本地运行它。如果您忘记了命令，Copilot Chat 将乐意并乐于提供帮助：

```py
docker run -p 8000:8000 -d --name itam itam:latest
```

您可以通过在命令行中输入以下命令来确认您的 Docker 容器正在运行：`docker ps | grep itam`。您应该能看到正在运行的实例。

## 7.2 通过 Copilot 搭建基础设施

在创建和测试应用程序时，使用你电脑上的 Docker 镜像是有用的。但是，当到了启动你的应用程序的时候，你需要一台比本地电脑更强大的机器。在本节中，我们将使用 GitHub Copilot 帮助我们通过让 Copilot 编写基础设施代码工具 Terraform 所需的必要部署描述符来设置和控制我们的 AWS 基础设施。Terraform 由 HashiCorp 制作，允许我们使用领域特定语言（DSL）来编写我们希望基础设施看起来像什么。这种 DSL 使我们免于理解每个云服务提供商用于配置硬件的所有复杂性和细微差别。此外，它还允许我们使用基础设施代码存储和版本化我们的基础设施。

首先，我们想要创建一个名为 ec2.tf 的文件，并添加提示告知 Copilot 我们打算将其作为 Terraform 文件，以及我们希望如何搭建我们的基础设施。请注意，Copilot 需要我们在输入给定行的第一个单词之前才能继续。

列表 7.2 示例 Terraform 文件，包括实例大小

```py
# Create a Terraform file that provisions an AWS EC2 instance of type t2.micro, installs the Docker daemon, and returns the instance's hostname.
provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "example" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
}
```

你可能会发现 Copilot 跳过了一个小但至关重要的细节：它没有提供安装和配置 Docker 的代码。鉴于 Docker 是运行我们的应用程序所必需的，我们需要纠正这个疏忽。实际上，你可能需要手动更新文件以包含安装 Docker 的命令，如下所示：

```py
resource "aws_instance" "example" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

   user_data = <<-EOF
                #!/bin/bash
                sudo yum update -y
                sudo yum install -y docker
                sudo service docker start
                sudo usermod -a -G docker ec2-user
                sudo docker run -d -p 80:80 nginx
                EOF
}
```

Copilot 应该生成一个完整的 Terraform 文件，类似于以下列表。你的代码可能并不完全匹配列表，但这没关系，只要它包含关键特性：提供者、实例、添加 Docker 守护进程的脚本、密钥对和安全组。

列表 7.3 创建最小 EC2 实例的 Terraform 文件

```py
# Create a Terraform file that provisions an AWS EC2 instance of type t2.micro, installs the Docker daemon, and returns the hostname of the instance.
provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "example" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

   user_data = <<-EOF
                #!/bin/bash
                sudo yum update -y
                sudo yum install -y docker
                sudo service docker start
                sudo usermod -a -G docker ec2-user
                sudo docker run -d -p 80:80 nginx
                EOF
    connection {
        type        = "ssh"
        user        = "ec2-user"
        private_key = file("~/.ssh/id_rsa")
        host        = self.public_ip
    }

    lifecycle {
        create_before_destroy = true
    }

    depends_on = [aws_security_group.allow_http]
}

resource "aws_security_group" "allow_http" {
  name        = "allow_http"
  description = "Allow HTTP inbound traffic"
  vpc_id      = "vpc-12345678"

  ingress {
    description = "HTTP from VPC"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    }
}

output "public_dns" {
  value = aws_instance.example.public_dns
    }
```

如果你使用的是默认的虚拟专用云（VPC），则`vpc_id`条目不是严格必要的。你会发现 AWS 团队选择的许多默认配置和约定是有意义的；如果你有更严格的安全要求，或者如果你对你的基础设施了如指掌，并假设一切，你可能会考虑从头开始使用 Terraform 设置一个新的 VPC。你需要将第 21 行的密钥对条目更改为你有访问权限的密钥对。

一旦你满意地完成了这个文件，运行`terraform init`命令。这个命令初始化一个新的或现有的 Terraform 工作目录。它会下载并安装你在配置文件中指定的所需提供者插件和模块，并准备好一切以便开始。

接下来，Terraform 将解释它打算做出的更改。你可以使用`terraform plan`命令来完成这个操作。这个命令为你的基础设施更改创建一个执行计划：它显示当你应用你的配置文件时，Terraform 将如何更改你的基础设施。计划将显示哪些资源将被创建、修改或销毁，以及将对你的基础设施做出的任何其他更改。

注意：当你第一次运行`terraform plan`时可能会遇到错误：“错误：配置 Terraform AWS Provider：未找到有效的 Terraform AWS Provider 凭证来源。”当你尝试连接到 AWS 但无法向 AWS 提供适当的凭证时，你会遇到这个错误。为了解决这个问题，你需要创建（或编辑）名为~/.aws/credentials 的文件，并添加你的 ITAM AWS 访问密钥 ID 和 AWS 秘密访问密钥凭证。你可以在*Amazon Web Services in Action，第三版*的 4.2.2 节“配置 CLI”中找到更多关于如何正确完成此操作的详细信息。

最后，为了应用 Terraform 的更改，你将使用`terraform apply`命令。然后，Terraform 将读取当前目录中的配置文件，并将任何更改应用到你的基础设施上。如果你在最后一次运行`terraform apply`之后对配置文件进行了任何更改——例如，如果你需要启动一个新的数据库实例或更改 EC2 的大小——Terraform 将显示更改的预览，并提示你在应用更改之前进行确认。

如果你应用这些更改，几分钟内你将有一个全新的 EC2 实例在你的 VPC 中运行。然而，这仅仅是方程的一半。拥有触手可及的计算能力是极好的，但你还需要一些东西来应用这种力量。在这种情况下，我们可以使用这个 EC2 实例来运行我们的 ISAM 系统。下一节简要演示了将本地构建的镜像传输到另一台机器的过程。

## 7.3 以困难的方式移动 Docker 镜像

首先，我们将从本地机器导出一个 Docker 镜像并将其加载到远程机器上。我们将使用`docker save`和`load`命令来完成这项任务。你可以在本地机器上使用`docker save`命令将镜像保存到一个 tar 归档文件中。以下命令将镜像保存到名为<image-name>.tar 的 tar 归档文件中：

```py
docker save -o <image-name>.tar <image-name>:<tag>
```

接下来，使用文件传输协议，如安全复制协议（SCP）或安全文件传输协议（SFTP），将 tar 存档传输到远程机器。你可以在远程机器上使用`docker load`命令从 tar 存档加载镜像：`docker load -i <image-name>.tar`。这将把镜像加载到远程机器上的本地 Docker 镜像缓存中。一旦镜像被加载，使用`docker run`命令启动镜像并运行 Docker 容器，就像你在构建它之后所做的那样。然后，将此镜像添加到你的 Docker compose 文件中，其中包含 Postgres 数据库和 Kafka 实例。

注意：关于 Terraform 的讨论被大大简化了。当你准备好认真使用 Terraform 时，你应该查阅 Scott Winkler 的《Terraform in Action》（Manning，2021 年；[www.manning.com/books/terraform-in-action](https://www.manning.com/books/terraform-in-action)）。

本节探讨了如何打包镜像并在远程主机上加载它们。这个过程是可脚本化的，但随着容器注册库的出现，现在管理部署比以往任何时候都更容易，无需将它们发送到整个互联网。在下一节中，我们将探讨这样一个工具：亚马逊的弹性容器注册库（ECR）。

## 7.4 以简单方式移动 Docker 镜像

Docker 镜像，我们容器的蓝图，是容器化应用的基本构建块。正确管理它们确保我们保持干净、高效和有序的开发和部署工作流程。Amazon ECR 作为一个完全管理的 Docker 容器注册库，使得开发者能够轻松地存储、管理和部署 Docker 容器镜像。

首先，让我们深入了解如何将 Docker 镜像推送到 ECR。这个过程对于使你的镜像可用于使用和部署至关重要。我们将逐步介绍设置你的本地环境、使用 ECR 进行认证以及推送你的镜像。在我们能够将镜像移动到 ECR 之前，我们必须创建一个用于存放该镜像的仓库。这可以通过 AWS 管理控制台完成，或者，就像我们很快要做的那样，使用 AWS 命令行界面（CLI）。创建用于镜像的新仓库的命令是

```py
aws ecr create-repository --repository-name itam
```

接下来，你需要使用 ECR 仓库 URL 和镜像名称给你的 Docker 镜像打标签。你可能想称之为`latest`或使用语义版本控制。打标签将允许你轻松回滚或前进到系统版本。使用以下命令给你的应用程序镜像打上`latest`标签：

```py
docker tag itam:latest 
123456789012.dkr.ecr.us-west-2.amazonaws.com/itam:latest
```

现在，使用`aws ecr get-login-password`命令对 Docker 进行 ECR 注册库的认证。这将生成一个用于认证 Docker 到注册库的 Docker `login`命令。登录命令如下

```py
aws ecr get-login-password --region us-west-2 | 
docker login --username AWS --password-stdin 
123456789012.dkr.ecr.us-west-2.amazonaws.com
```

最后，使用`docker push`命令将 Docker 镜像推送到 ECR 注册库：

```py
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/itam:latest
```

一旦镜像存放在你的注册表中，你的部署选项大大增加。例如，你可以编写一个 bash 脚本，登录到 EC2 实例并执行 `docker pull` 来下载和在该 EC2 上运行镜像。或者，你可能希望采用更可靠的部署模式。在下一节中，我们将介绍如何在名为 Elastic Kubernetes Service (EKS) 的强大云服务上设置和启动我们的应用程序的过程。EKS 是 AWS 提供的托管 Kubernetes 服务。让我们深入探讨吧！

## 7.5 将我们的应用程序部署到 AWS Elastic Kubernetes Service

Kubernetes 相比于在 EC2 实例上简单地运行 Docker 镜像提供了许多好处。首先，使用 Kubernetes 管理和扩展我们的应用程序变得更加简单。此外，使用 Kubernetes，我们不需要花费很多额外的时间去思考我们的基础设施应该是什么样子。而且，多亏了其对名为 *pods* 的镜像生命周期的自动管理，我们的应用程序将基本上是自我修复的。这意味着如果出现问题，Kubernetes 可以自动修复它，确保我们的应用程序始终运行顺畅。

首先，我们需要一个用 YAML（Yet Another Markup Language 或 YAML Ain’t Markup Language，取决于你问谁）编写的部署描述符，这将描述我们希望 ITAM 系统始终保持的状态。这个文件（通常称为 deployment.yaml）将提供 Kubernetes 将与之比较的当前运行系统的模板，并根据需要做出修正。

列表 7.4 ITAM 系统的 Kubernetes 部署文件

```py
# Create a Kubernetes deployment file for the itam application. The image name is itam:latest
# The deployment will run on port 8000

apiVersion: apps/v1
kind: Deployment
metadata:
  name: itam-deployment
  labels:
    app: itam
spec:
  replicas: 1
  selector:
    matchLabels:
      app: itam
  template:
    metadata:
      labels:
        app: itam
    spec:
      containers:
      - name: itam
        image: itam:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
```

然而，这不会起作用。Kubernetes 将无法找到我们在部署描述文件中引用的镜像。为了纠正这个问题，我们需要告诉 Kubernetes 使用我们新创建的 ECR。幸运的是，这并不像听起来那么具有挑战性。我们只需要更新文件中的镜像条目，使其指向 ECR 镜像，以及授予 EKS 访问 ECR 的权限（好吧，可能有点复杂，但它是可管理的）。

首先，更新部署 YAML 以使用 ECR 镜像：

```py
image: 123456789012.dkr.ecr.us-west-2.amazonaws.com/itam:latest. 
```

然后，你需要为 EKS 定义一个策略，并使用 AWS CLI 或身份和访问管理（IAM）管理控制台应用该策略。尽管应用策略超出了本书的范围，但你可以使用 Copilot 来定义它。生成的策略将类似于以下列表。

列表 7.5 允许 EKS 从 ECR 拉取镜像的 IAM 策略

```py
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowPull",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::<aws_account_id>:role/<role>"
      },
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability"
      ],
      "Resource": "arn:aws:ecr:<region>:<aws_account_id>:
repository/<repository_name>"
    }
  ]
}
```

一旦 EKS 能够从 ECR 拉取镜像，你将看到一个 pod 开始运行。然而，你无法外部访问这个 pod。你需要创建一个服务。在 Kubernetes 中，*服务* 是一个抽象概念，它定义了一组逻辑上的 pod（你在 Kubernetes 对象模型中创建或部署的最小和最简单的单元）以及访问它们的策略。

服务使得应用程序的不同部分以及不同应用程序之间能够进行通信。它们通过将 Pod 暴露给网络和其他 Kubernetes 中的 Pod 来帮助分配网络流量和负载均衡。

列表 7.6 Kubernetes 服务文件，以启用我们的应用程序的外部访问

```py
# Please create a service for the application that uses a load balancer type egress
apiVersion: v1
kind: Service
metadata:
  name: itam-service
spec:
  type: LoadBalancer
  selector:
    app: itam
  ports:
  - name: http
    port: 80
    targetPort: 8000
```

Kubernetes 负责将所有请求从入口路由到服务，然后到正在运行的 Pod，无论它们运行在哪个主机上。这允许无缝故障转移。Kubernetes 预期事情会失败。它依赖于此。因此，许多分布式系统中的最佳实践都内置到了 Kubernetes 中。到达 Kube 是构建一个可靠、高可用系统的重大第一步。在下一节中，我们将探讨如何重复和持续地减轻将我们的应用程序部署到 Kubernetes 的负担。我们将查看如何使用 GitHub Actions 构建一个小型部署管道。

## 7.6 在 GitHub Actions 中设置持续集成/持续部署（CI/CD）管道

如果发布很困难，那么它就不会经常进行。这限制了我们对应用程序增值的能力，从而也限制了我们对利益相关者的增值。然而，自动化部署过程显著减少了发布所需的时间。这使得更频繁的发布成为可能，加速了开发步伐，并能够更快地将功能交付给用户。持续集成/持续部署（CI/CD）管道限制了与部署相关的风险。通过进行更小、更频繁的更新，任何出现的问题都可以被隔离和快速修复，最小化对最终用户潜在的影响。这些管道促进了代码更改的无缝集成，并加速了部署，简化了软件发布过程。

GitHub Actions 允许我们在 GitHub 仓库中直接构建定制的 CI/CD 管道。这使得开发工作流程更加高效，并能够自动化各种步骤，让我们能够专注于编码，而不是集成和部署的物流。

本节简要介绍了使用 GitHub Actions 和 GitHub Copilot 设置 CI/CD 管道。请注意，这不会是一个全面的指南，而是一个介绍潜在好处和一般工作流程的概述。这应该作为入门指南，让你了解这些工具如何被用来优化你的软件开发过程。

首先，在你的项目中的路径 .github/workflows 下创建一个文件。注意前面的点。你可以把这个文件命名为 itam.yaml 或者你想要的任何名字。在这个文件的第一个行，添加以下提示：

```py
# Create a GitHub Actions workflow that builds the ITAM application on every merge to the main branch and deploys it to EKS. 
```

注意：像本章中我们交给 Copilot 的许多与基础设施相关任务一样，Copilot 在创建此文件时需要我们提供大量帮助。我们需要了解这个文件的结构以及如何开始每一行。在这种情况下，向 ChatGPT 或 Copilot Chat 请求为我们构建文件是有意义的。

文件的这部分概述了此操作何时应该执行。on:push 指令表示当向主分支进行 git push 时，此操作应该执行。此文件中只有一个作业，包含多个步骤。这个名为“build”的作业使用内嵌函数`login-ecr`登录到我们的 ECR。

列表 7.7 构建我们的应用程序的 GitHub Actions 文件的开始部分

```py
# Create a GitHub Actions workflow that builds the ITAM application on every merge to the main branch and deploys it to EKS.
name: Build and Deploy to EKS

on:
  push:
    branches:
      - main
jobs:
```

构建作业首先会从我们的 GitHub 仓库检出代码。它使用模块`actions/checkout`版本 2 中编写的代码。同样，接下来它将获取 EKS CLI 并配置凭证以连接到 EKS。请注意，AWS 访问密钥和秘密是自动传递到应用程序中的值。GitHub Actions 使用内置的秘密管理系统来存储敏感数据，如 API 密钥、密码和证书。该系统集成到 GitHub 平台中，允许你在仓库和组织级别添加、删除或更新秘密（以及其他敏感数据）。在存储之前，秘密会被加密，不会显示在日志中或可供下载。它们仅作为环境变量暴露给 GitHub Actions 运行器，这是一种处理敏感数据的安全方式。

同样，您可以在操作中创建环境参数并使用它们。例如，看看变量`ECR_REGISTRY`。这个变量是使用`login-ecr`函数的输出创建的。在这种情况下，您仍然需要在您的 Actions 文件中硬编码 ECR。然而，您应该这样做是为了保持一致性，并且需要在文件中仅在一个地方管理它。大多数这些步骤应该看起来很熟悉，因为我们已经在整个章节中使用了它们。这就是自动化的魔力：它为您完成这些工作。

列表 7.8 我们的 GitHub Actions 文件的构建和部署步骤

```py
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up EKS CLI
      uses: aws-actions/amazon-eks-cli@v0.1.0

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: itam
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

    - name: Deploy to EKS
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: itam
        IMAGE_TAG: ${{ github.sha }}
      run: |
        envsubst < k8s/deployment.yaml | kubectl apply -f -
        envsubst < k8s/service.yaml | kubectl apply -f -
```

文件的最后部分登录到 AWS ECR。Actions 文件中的步骤调用此操作。完成后，它将输出返回到调用函数。

列表 7.9 构建和部署到 EKS 的 GitHub Actions 文件

```py
  login-ecr:
    runs-on: ubuntu-latest
    steps:
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
      with:
        registry: <your-ecr-registry>
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

探索代码即基础设施使我们能够理解它在任何项目中的关键作用以及如何通过代码更好地管理。像 Terraform 这样的工具为管理基础设施提供了简化的解决方案，GitHub 以代码为中心的功能有助于维护整体工作流程。

通过 GitHub Actions 等平台引入 CI/CD 管道，突出了自动化软件交付过程的重要性。自动化此类过程可以提高软件开发生命周期的速度和可靠性，并最大限度地减少人为错误的可能性。

将基础设施作为代码管理的旅程是不断演变的，新的工具和实践不断涌现。这需要持续学习和适应的心态。本章为您展示了其优势和可能性。

## 摘要

+   你了解了从应用开发到产品发布的转变，包括部署策略、云基础设施的最佳实践，以及使用 Docker 和 Terraform 高效管理和容器化应用的方法。

+   本章解释了如何通过 Kubernetes 管理应用部署，包括创建 YAML 部署描述符、形成用于网络流量分配的服务，以及在亚马逊的弹性 Kubernetes 服务（EKS）上部署。

+   你发现了如何将部署方法适应不同的环境，无论是各种云平台还是本地环境，以及 GitHub Copilot 如何帮助准确创建 Dockerfile 和 Terraform 文件。

+   最后，我们探讨了将 Docker 镜像导出到远程机器、推送到亚马逊的弹性容器注册库（ECR）以及使用 GitHub Actions 迁移到自动化部署的过程。
