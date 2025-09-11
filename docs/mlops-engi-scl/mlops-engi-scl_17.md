# 附录 B：使用 Docker 入门

如果您曾经使用过公共云来启用您的应用程序进行自动缩放，即您可以轻松地添加或删除应用程序集群的计算节点功能，那么您已经使用了虚拟服务实例。您甚至可能使用过类似 ssh 的程序登录到您的实例，然后通过这个 ssh 会话远程管理它们。乍一看，Docker 容器实例似乎与虚拟服务器没有什么不同。如果您通过 ssh 登录到 Docker 容器中，与通过 AWS EC2 等公共云服务托管的虚拟服务器的会话相比，您可能甚至感觉不出差别。但是，虽然与 Docker 有关的传统公共云服务虚拟服务器存在相似之处，但 Docker 提供的重要功能是需要知道的。

一个了解 Docker 的可接近方式是将其看作是轻量级（与虚拟服务器相比）的虚拟化。这包括以下维度：

+   存储方面，Docker 镜像快照占用的磁盘空间比传统的虚拟服务器/机器镜像更少。

+   内存方面，由于 Docker 容器实例消耗的内存比客户机实例（虚拟服务器）少。

+   启动速度方面，Docker 容器比其虚拟服务器等同物启动得更快。

+   性能方面，由于运行在 Docker 容器中的程序与运行在虚拟客户机实例中的程序相比，几乎没有任何 CPU 开销。

然而，Docker 容器和虚拟服务器之间的差异在核心硬件/软件级别上更加根本。传统的虚拟化技术（例如 VMWare、Xen）将主机计算机硬件进行虚拟化，或者创建基于软件的代理来实现底层硬件组件的虚拟化，包括中央处理单元、存储、网络设备等，通过在硬盘和内存中实例化带有操作系统副本和设备驱动程序及其他支持软件的客户机环境。相比之下，Docker 容器虚拟化操作系统，以便每个客户机容器实例共享相同的操作系统，但在操作上却像单独拥有对整个操作系统的隔离访问权限一样。

## B.1 使用 Docker 入门

如果您的环境中没有安装 Docker，则可以通过访问[`labs.play-with-docker.com/`](https://labs.play-with-docker.com/)，获取一个带有 Docker 的实验室环境。

Docker 是一个多义词，它描述了各种 Docker 技术组件（例如 Docker 引擎、Docker Swarm 等）、Docker 公司本身，以及在 hub.docker.com 维护的 Docker 容器镜像注册表。安装 Docker 引擎时，您的环境中没有任何 Docker 镜像安装。

假设您已经正确配置了 Docker 引擎和 Docker 主机软件，那么您可以通过在 Shell 环境中运行以下命令的变体来使用 Docker，即经典的 hello-world 程序：

```py
docker run hello-world
```

假设您尚未下载（pull）hello-world Docker 镜像，这应该输出以下内容：

```py
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
0e03bdcc26d7: Pull complete
Digest: sha256:7f0a9f93b4aa3022c3a4c147a449bf11e094
➥ 1a1fd0bf4a8e6c9408b2600777c5
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
```

重要的是要理解 run 是一个组合的 Docker 命令，在幕后执行多个操作。以下将解释 run 执行的命令，以帮助您理解它的作用。

由于您已经完成了使用 Docker 的基本 hello-world 风格示例，让我们尝试使用流行的 nginx web 服务器来进行一个稍微复杂一些的示例。要从 hub.docker.com（也称为 Docker Hub）下载 Docker 镜像到本地 Docker 主机，您可以执行如下的 pull 命令：

```py
docker pull nginx
```

这应该输出以下内容：

```py
docker pull nginx
Using default tag: latest
latest: Pulling from library/nginx
bf5952930446: Pull complete
cb9a6de05e5a: Pull complete
9513ea0afb93: Pull complete
b49ea07d2e93: Pull complete
a5e4a503d449: Pull complete
Digest: sha256:b0ad43f7ee5edbc0effbc14645ae7055e21b
➥ c1973aee5150745632a24a752661
Status: Downloaded newer image for nginx:latest
docker.io/library/nginx:latest
```

注意 由于 nginx 镜像可能在此书创建之时已经发生了变化，因此您在消息中看到的哈希码可能与示例中的不对应，但是本附录中的概念都是适用的，不管示例中的哈希码具体值是什么。

pull 命令生成的消息表明 Docker 默认使用了 nginx 镜像的标签 latest。由于也可以指定完全限定的域名从而让 Docker 拉取镜像，Docker 默认也使用 Docker Hub FQN docker.io/library 作为唯一标识 nginx 镜像的前缀。

注意 pull 命令返回的消息中提到的各种哈希码，例如

```py
bf5952930446: Pull complete.
```

在 pull 命令执行时，您观察到的每个 Pull complete 消息前面的哈希码值（以及您运行 pull 命令时观察到的下载进度消息）都是 Docker 容器镜像所使用的联合文件系统中的一个层的唯一标识符或*指纹*。相比之下，跟随 Digest: sha256: 消息的哈希码是整个 nginx Docker 镜像的唯一指纹。

一旦镜像位于您的 Docker 主机服务器上，您就可以使用它来创建 Docker 容器的实例。该容器是前面描述的轻量级虚拟机，或者是运行在与 Docker 主机服务器操作系统的其余部分近乎隔离的虚拟客户操作系统环境。

要创建容器，您可以执行以下命令

```py
docker create  nginx
```

这应该返回类似以下的唯一容器 ID：

```py
cf33323ab079979200429323c2a6043935399653b4bc7a5c86
➥ 553220451cfdb1
```

您可以在命令中使用完整且冗长的容器 ID，也可以使用 Docker 允许您指定容器 ID 的前几个字符，只要在您的 Docker 主机环境中是唯一的即可。要确认容器是否已在您的环境中创建，您可以使用 docker ls -a | grep <CONTAINER_ID> 命令，其中 docker ls -a 列出您环境中的所有容器，并且管道过滤器 grep 命令筛选出您需要的容器。例如，由于我创建的容器的 ID 以 cf33 开头，我可以执行

```py
docker ps -a | grep cf33
```

在我的情况下，输出如下：

```py
cf33323ab079        nginx                 
➥ "/docker-entrypoint...." 
➥ 5 minutes ago       
➥     Created                                       ecstatic_gagarin
```

注意，Docker 自动为容器创建了一个易于记忆和在命令行中指定的可读的 Docker 容器 ID，名为 ecstatic_gagarin，与哈希码相比更容易记忆和指定。此外，由于容器刚刚从镜像创建而从未启动，因此容器的状态为已创建。要启动容器，您可以执行

```py
docker start -p 8080:80 CONTAINER_ID
```

用您的容器 ID 值或前缀替换 CONTAINER_ID。输出只是回显容器 ID，但您可以通过重新运行确认容器已更改状态

```py
docker ps -a | grep CONTAINER_ID
```

这应该会报告容器的正常运行时间，类似于以下内容：

```py
cf33323ab079        nginx                          
➥ "/docker-entrypoint...."   11 minutes ago      
➥ Up 2 minutes              80/tcp              
➥ ecstatic_gagarin
```

尽管您可能期望您应该能够访问 NGINX Web 服务器，因为您启动了一个 nginx 容器，但这是不正确的。简单地启动容器不包括将在客户容器环境中打开的端口映射（暴露）到主机环境的步骤。要解决此问题，您可以使用

```py
docker stop CONTAINER_ID
```

这应该会回显您的 CONTAINER_ID 值。

接下来，使用端口 80（Web 服务器 HTTP 端口）作为主机 Docker 环境的端口 8080 重新运行容器。可以通过以下方式执行：

```py
docker run -p 8080:80 nginx
```

这将调用一个新的 Docker 容器的新实例，并在终端中返回 NGINX 服务的日志消息。此时，如果您打开 Web 浏览器并导航到您的 Docker 主机服务器 IP 地址的端口 8080，例如通过导航到 127.0.0.1:8080，您应该会看到带有消息的 HTML 页面：

```py
Welcome to nginx!
```

此时，Docker 创建的容器实例的行为与您执行 docker start 时观察到的不同。在这种情况下，如果您在终端会话中按下 Ctrl-C，容器实例将终止，您可以通过重新运行 docker ps 轻松确认。这次，docker ps 不应显示任何正在运行的容器实例，因为您刚刚通过按下 Ctrl-C 关闭了它。

为了防止 Docker 容器实例接管您的终端会话，您可以通过指定 -d 参数在分离模式下重新运行它：

```py
docker run -d -p 8080:80 nginx
```

这应该会返回您刚刚启动的实例的容器 ID。

当然，拥有一个只显示“欢迎使用 nginx！”消息的 Web 服务器并不有趣。要更改用于提供欢迎网页的 HTML 文件的内容需要做什么？

您可以先确认包含欢迎消息的 index.html 文件的位置。exec 命令允许您使用主机 shell 的 docker CLI 在运行的客户容器实例中执行任意 Linux 命令。例如，要输出您的 nginx 实例中 /usr/share/nginx/html/index.html 文件的内容，请运行

```py
docker exec CONTAINER_ID /bin/bash -c 
➥ 'head /usr/share/nginx/html/index.html'
```

如果您为您的 nginx 容器实例使用正确的 CONTAINER_ID 值，则应输出

```py
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
<style>
```

请注意，在 exec 命令中，您指定要使用 /bin/bash 执行 Bash shell，并使用 -c 标志和 head /usr/share/nginx/html/index.html 作为实际命令的命令行参数指定 shell 命令。请记住，head 命令可用于输出文件的前五行。

类似地，您可以通过更改客户容器实例中的内容轻松修改 index.html 文件的内容。如果执行

```py
docker exec CONTAINER_ID /bin/bash 
➥ -c 'echo "Hello from my Docker tutorial" > 
➥ /usr/share/nginx/html/index.html'
```

刷新浏览器中的 localhost:8080 页面，你应该收到“Hello from my Docker tutorial”消息。

重要的是，您意识到 index.html 文件的更改发生在容器实例中，*而不是*在用于启动实例的容器镜像中。如果您对用于启动容器实例的容器镜像所做的更改感到不确定，您可以使用 diff 命令找出详细信息：

```py
docker diff CONTAINER_ID
```

这将根据对 index.html 文件的更改和 NGINX Web 服务器更改（C）或添加（A）所输出以下内容：

```py
C /usr
C /usr/share
C /usr/share/nginx
C /usr/share/nginx/html
C /usr/share/nginx/html/index.html
C /var
C /var/cache
C /var/cache/nginx
A /var/cache/nginx/client_temp
A /var/cache/nginx/fastcgi_temp
A /var/cache/nginx/proxy_temp
A /var/cache/nginx/scgi_temp
A /var/cache/nginx/uwsgi_temp
C /etc
C /etc/nginx
C /etc/nginx/conf.d
C /etc/nginx/conf.d/default.conf
C /run
A /run/nginx.pid
```

在第 B.2 节，您将了解如何创建自己的自定义 Docker 镜像，以便可以持久保存所需的更改并在许多 Docker 容器实例之间重复使用它们。

当您经常在 Docker 主机环境中启动和停止多个容器实例时，将它们作为一批管理是很方便的。您可以使用以下命令列出所有容器实例 ID

```py
docker ps -aq
```

这应该返回类似以下的列表：

```py
c32eaafa76c1
078c98061959
...
a74e24994390
6da8b3d1f0e1
```

省略号表示您可能有任意数量的容器 ID 由命令返回。要停止环境中的所有容器实例，您可以使用 xargs 命令

```py
docker ps -aq | xargs docker stop
```

这会停止所有容器实例。接下来，您可以重复使用 docker rm 结合 xargs 来移除任何剩余的容器实例：

```py
docker ps -aq | xargs docker rm
```

在停止并删除 docker 容器实例后，如果重新运行

```py
docker ps -aq
```

你应该看到一个空的响应，这意味着你的 Docker 主机环境中没有任何容器实例。

## B.2 构建自定义镜像

创建自己的 Docker 镜像并与世界分享是非常简单的。它始于一个 Dockerfile，这是一个声明性规范，用于如何获取现有（基础）容器镜像并使用您自己的更改扩展它（考虑在其上添加层）。

您应该通过创建和导航至一个空目录 tmp 开始构建自己的 Docker 镜像的过程：

```py
mkdir tmp
```

准备一个空目录是个好习惯，因为 Docker 在构建过程中会复制目录的内容（称为上下文目录），所以如果您意外地从包含大量不相关内容的目录启动构建过程，您将不得不等待 Docker 不必要地复制这些不相关的内容，而不是立即返回结果镜像。

由于每个 Docker 镜像都以基础镜像开始，因此 Dockerfile 必须在构建过程中使用 FROM 语句指定要使用的基础镜像的标识符。此示例继续使用 NGINX Web 服务器：

```py
echo "FROM nginx:latest" > Dockerfile
```

在这里，echo 命令不会产生输出，而是在当前目录中创建一个新的 Dockerfile，其中包含一个包含 FROM 语句的单行，指定 nginx:latest 作为基础镜像。现在，您已准备好使用以下构建命令构建您的第一个自定义 NGINX 镜像：

```py
docker build -t just-nginx:latest -f Dockerfile tmp/
```

应该输出

```py
docker build -t just-nginx:latest -f Dockerfile tmp/
Sending build context to Docker daemon  1.583kB
Step 1/1 : FROM nginx:latest
 ---> 4bb46517cac3
Successfully built 4bb46517cac3
Successfully tagged just-nginx:latest
```

此时，您可以确认您在 Docker 主机环境中有一个新的 Docker 镜像

```py
docker image ls | grep nginx
```

这会产生一个输出，可能会让您对奇怪的创建日期时间戳感到惊讶。在我的情况下，对于镜像 ID 4bb46517cac3，时间戳报告了 3 周前的创建日期

```py
just-nginx      latest                        
➥ 4bb46517cac3        3 weeks ago         133MB
```

请记住，Docker 依赖于基于哈希代码的指纹来对图像层和整个容器镜像进行识别。由于您的 Dockerfile 没有对图像进行任何更改，所以哈希代码保持不变，尽管元数据值（just-nginx）发生了变化。

那么有关实际更改基础 Docker 镜像的示例呢？您可以首先创建自己的自定义 index.html 文件，您希望在访问 NGINX Web 服务器时看到其呈现。请注意，使用以下命令将该文件创建在 tmp 子目录中

```py
echo 
➥ '<html><body>Welcome to my custom nginx message!
➥ </body></html>' > tmp/index.html
```

准备好 index.html 文件后，您可以使用命令修改 Dockerfile，在构建过程中将文件复制到镜像中，

```py
echo 'COPY index.html 
➥ /usr/share/nginx/html/index.html' >> Dockerfile
```

因此整个 Dockerfile 应该包括以下内容：

```py
FROM nginx:latest
COPY index.html /usr/share/nginx/html/index.html
```

此时，您已经准备好使用自定义欢迎消息构建另一个镜像。运行

```py
docker build -t custom-nginx:latest -f Dockerfile tmp/
```

应该输出

```py
Sending build context to Docker daemon  2.607kB
Step 1/2 : FROM nginx:latest
 ---> 4bb46517cac3
Step 2/2 : COPY index.html 
➥ /usr/share/nginx/html/index.html
 ---> c0a21724aa7a
Successfully built c0a21724aa7a
Successfully tagged custom-nginx:latest
```

其中哈希代码可能与您的不匹配。

请注意，Docker COPY 命令完成成功，因为您将 tmp 用作构建上下文目录，并且 index.html 存在于 tmp 中。通常，在构建过程中想要复制到 Docker 镜像中的任何文件都必须位于构建上下文目录中。

现在，您已经准备好启动新构建的镜像，

```py
docker run -d -p 8080:80 custom-nginx:latest
```

并确认如果您访问 localhost:8080，NGINX 会响应

```py
Welcome to my custom nginx message!
```

## B.3 共享您的自定义镜像给世界

在您可以将 Docker 镜像上传到 Docker 镜像注册表之前，您必须在 hub.docker.com 上创建您的个人帐户。假设您已经创建了您的帐户并且拥有 Docker Hub 的用户名和密码，您可以使用这些凭据从命令行登录：

```py
docker login
```

成功登录后，您应该观察到类似以下的输出：

```py
docker login
Login with your Docker ID to push and pull images 
➥ from Docker Hub. If you don't have a Docker ID, 
➥ head over to https://hub.docker.com to create one.
Username: YOUR_USER_NAME
Password: YOUR_PASSWORD
Login Succeeded
```

要使 Docker 镜像准备好上传，必须以您的 Docker 用户名作为前缀对其进行标记。要将此前缀分配给您的 custom-nginx 镜像，可以使用 tag 命令

```py
docker tag custom-nginx:latest 
➥ YOUR_USER_NAME/custom-nginx:latest
```

将 YOUR_USER_NAME 替换为您的 Docker Hub 用户名。要上传（推送）您的镜像到 Docker Hub，可以执行

```py
docker push YOUR_USER_NAME/custom-nginx:latest
```

这应该会产生类似以下的输出：

```py
The push refers to repository 
➥ [docker.io/YOUR_USER_NAME/custom-nginx]
088b6bf061ef: Pushed
550333325e31: Pushed
22ea89b1a816: Pushed
a4d893caa5c9: Pushed
0338db614b95: Pushed
d0f104dc0a1f: Pushed
latest: digest: sha256:9d12a3fc5cbb0a20e9be7afca476
➥ a0603a38fcee6ccfedf698300c6023c4b444 size: 1569
```

这表明你可以重新登录到你的 Docker 注册表仪表板，网址为 hub.docker.com，并确认`custom-nginx:latest`镜像已经在你的注册表中可用。
