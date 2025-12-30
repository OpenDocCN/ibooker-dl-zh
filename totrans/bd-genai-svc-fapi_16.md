# Chapter 12\. Deployment of AI Services

In this final chapter, it is time to complete your GenAI solution by deploying it. You’re going to learn several deployment strategies and, as part of deployment, containerize your services with Docker following its best practices.

# Deployment Options

You now have a working GenAI service that you want to make accessible to your users. What are your deployment options? There are a few common deployment strategies you can adapt to make your apps accessible to users:

*   Virtual machines (VMs)

*   Serverless functions

*   Managed application platforms

*   Containerization

Let’s explore each in more detail.

## Deploying to Virtual Machines

If you plan to use your own on-premises servers or prefer to deploy your services on the same hardware hosting your other applications for high isolation and security, you can deploy your GenAI service to a VM.

A VM is a software emulation of a physical computer running an operating system (OS) and applications. It’s no different from a physical computer like a laptop, smartphone, or server.

The VM’s *host* system provides resources such as CPU, memory, and storage, while a software layer called the *hypervisor* manages the VM and allocates resources from the host to the VM. The resources that the hypervisor allocates to the VM is the *virtual hardware* that its OS and applications run on.

The VM could run directly on host’s hardware (bare metal) or on a conventional operating system (i.e., be hosted). As a result, the OS installed within the VM is then referred to as the *guest OS*.

[Figure 12-1](#deployment_vm_architecture) shows the virtualization technology system architecture.

![bgai 1201](assets/bgai_1201.png)

###### Figure 12-1\. Virtualization system architecture

Cloud providers or your own data center can consist of several physical servers, each hosting multiple VMs with their own guest OS and hosted applications. For cost-effective resource sharing, these VMs may share the same mounted physical storage drive even though they’re contained within fully isolated environments, as you can see in [Figure 12-2](#deployment_vm_data_center).

![bgai 1202](assets/bgai_1202.png)

###### Figure 12-2\. Hosted VMs in a data center

The benefit of using a VM is that you have direct access to the guest OS, virtual hardware resources, and GPU drivers. If there are any issues with deployment, you can connect to the VM via the *Secure Shell Transfer* (SHH) protocol to inspect application logs, set up application environment, and debug production issues.

Deploying your services to VMs will be as straightforward as cloning your code repository to the VM and then installing the required dependencies, packages, and drivers to successfully start up your application. However, the recommended way to do this is to use a containerization platform such as Docker running on the VM to enable continuous deployments and other benefits. You should also ensure you size your VM resources appropriately so that your services aren’t starved for CPU/GPU cores, memory, or disk storage.

With on-premises VMs, you can save on-cloud hosting or server rental costs and can fully secure your application environments to a handful of users, isolated from the public internet. These benefits are also achievable with cloud VMs but require additional networking and resource configuration to set up. In addition, you can have access to GPU hardware and configure drivers for your application requirements.

Bear in mind that using the VM deployment pattern may not be easily scalable and requires significant effort to maintain. Additionally, VM servers normally run 24/7 incurring constant running costs, unless you automate their startup and shutdown based on your needs. You’ll be responsible for applying security patches, OS updates, and package upgrades alongside any networking configurations. With direct access to hardware resources, you’ll also have more decisions to make that can slow you down, leading to decision fatigue.

My advice is to deploy to a VM if you don’t plan to scale your services anytime soon or need to maintain low server costs and a secure isolated application environment for a handful of users. In addition, make sure you’ve planned sufficient time for deployment, networking, and configuration of your VMs.

## Deploying to Serverless Functions

Aside from VMs, you can also deploy your services on cloud functions that cloud providers supply as *serverless* systems. In serverless computing, your code is executed in response to events such as database changes, updates to blobs in a storage, HTTP requests, or messages added to a queue. This means you pay only for the requests or compute resources your services use, rather than for an entire server as with a continuously running VM.

Serverless deployments are often useful when:

*   You want to have event-driven systems instead of a running VM, which might be on 24/7

*   You want to deploy your API services using a serverless architecture that’s highly cost-efficient

*   Your services are to perform batch processing jobs

*   You need workflow automation

The term *serverless* doesn’t mean that cloud functions don’t require hardware resources to execute but rather that the management of these resources is handled by the cloud provider. This allows you to focus on writing application code without worrying about server and OS-level details.

Cloud providers instantiate compute resources to meet the demand of their customers. Often, there is a surge in demand, requiring them to create additional resources in advance to handle the demand spike. However, once the demand drops, excess unallocated compute resources remain that must be either shut down or shared among other customers.

Removing and creating resources is an intensive compute operation to perform. At scale, these operations carry significant costs for cloud providers. Therefore, cloud providers prefer to keep these resources running as much as possible and distribute them among existing customers to maximize billing.

To encourage customers to use these excess compute, they’ve built cloud function services that you can leverage to run your backend services on excess (i.e., serverless) compute. Luckily, there are packages such as Magnum that allow you to package FastAPI services on AWS cloud functions. You will soon see that FastAPI services can also be deployed as Azure cloud functions.

What you need to bear in mind is that these functions are allocated only a small amount of resources and have a short timeout. However, you can request longer timeouts and compute resources to be allocated, but it may take longer to receive these allocations, leading to higher latencies for your users.

###### Warning

If your business logic consumes a lot of resources or requires longer than a handful of minutes to execute, cloud functions may not be a suitable deployment option for you.

However, you can split your FastAPI services across multiple functions, with each function handling a single exposed endpoint. This way, you can deploy parts of your service as cloud functions, reducing the portion of the FastAPI service that needs to be deployed using other methods.

The main advantage of using serverless functions for deploying your services is their scalability. You can scale your applications as needed and pay only a fraction of the cost compared to reserving dedicated VM resources. Cloud providers typically charge based on the number of function executions and runtime, often with generous monthly quotas. This means that if your functions run quickly and you have a moderate number of concurrent users, you might be able to host all your services for free.

Furthermore, cloud providers also supply function runtimes that you can install locally for local testing and development so that you can significantly shorten development iterations.

Each cloud provider has their own approach to deploying serverless functions. Often, you require an entry script such as *main.py* that can import dependencies from other modules as needed. Alongside the entry point script, you’ll need to upload a function host JSON configuration file alongside *requirements.txt* for required dependencies to be installed on deployment on a Python runtime.

You can then deploy functions by uploading all the required files as a zipped directory or using CI/CD pipelines that authenticate with the provider and execute the deployment commands within your cloud project.

As an example, let’s try to deploy a bare-bones FastAPI app that returns LLM responses. The structure of the project will be as follows:

```py
project/
│
├── host.json
├── main.py
├── app.py
└── requirements.txt
```

You can then package a FastAPI app as an [Azure serverless function](https://oreil.ly/ZaOuF) by following the upcoming code examples.

You will need to install the `azure-functions` package to run Azure’s serverless function runtime for local development and testing:

```py
$ pip install azure-functions
```

Then, create *host.json* by following [Example 12-1](#deployment_function_azure_host).

##### Example 12-1\. Azure Functions host configurations (host.json)

```py
{
  "version": "2.0",
  "extensions": {
    "http": {
        "routePrefix": ""
    }
  }
}
```

Afterward, implement your GenAI service with the FastAPI service as usual by following [Example 12-2](#deployment_function_azure_app).

##### Example 12-2\. Simple FastAPI application serving LLM responses

```py
# app.py

import azure.functions as func
from fastapi import FastAPI

app = FastAPI()

@app.post("/generate/text", response_model_exclude_defaults=True)
async def serve_text_to_text_controller(prompt):
...
```

Finally, wrap your FastAPI `app` within `func.AsgiFunctionApp` for the Azure serverless function runtime to hook into it, as shown in [Example 12-3](#deployment_function_azure_function).

##### Example 12-3\. Deploying a FastAPI service with Azure Functions

```py
# function.py

import azure.functions as func
from main import app as fastapi_app

app = func.AsgiFunctionApp(
    app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS
)
```

You can then start the function app by running the `func start` command, which should be available as a CLI command once you install the `azure-functions` package:

```py
$ func start

>> Found the following functions:
>> Functions:
>>        http_app_func: [GET,POST,DELETE,HEAD,PATCH,PUT,OPTIONS] \
                          http://localhost:7071//{*route}

>> Job host started
```

You can then try URLs corresponding to the handlers in the app by sending HTTP requests to both simple and the parameterized paths:

```py
http://localhost:7071/generate/text
http://localhost:7071/<other-paths>
```

Once ready, you can then deploy your FastAPI wrapped serverless function to the Azure cloud and then run the following command:

```py
$ func azure functionapp publish <FunctionAppName>
```

The `publish` command will then publish the project files from the project directory to `<FunctionAppName>` as a ZIP deployment package.

After deployment, you can then test different paths on the deployed URL:

```py
http://<FunctionAppName>.azurewebsites.net/generate/text
http://<FunctionAppName>.azurewebsites.net/<other-paths>
```

###### Warning

Your chosen cloud provider may not support serving a FastAPI server within its function runtime. If that’s the case, you may want to seek alternative deployment options. Otherwise, you’ll need to migrate the logic of your endpoints to the supported web framework of the function runtime and create separate functions for each endpoint.

As you see, deploying your FastAPI service as cloud functions is straightforward and allows you to delegate the management and scaling of your services to cloud providers.

Bear in mind that if you decide to serve a GenAI model in your service, cloud functions wouldn’t be suitable deployment targets due to their short timeout periods (10 minutes). Instead, you’d want to use a model provider API in your services so that you have reliable and scalable access to the model without being constrained by execution time limits.

## Deploying to Managed App Platforms

In addition to cloud functions or VMs, you can upload your codebase as ZIP files to app platforms managed by cloud providers. Managed app platforms let you delegate several tasks related to maintenance and management of your services to the cloud provider. In exchange, you pay only for the compute resources managed by the cloud provider that serve your application. The cloud provider systems allocate and optimize resources based on your application’s needs.

Examples of such services include Azure App Services, AWS Elastic Beanstalk, Google App Engine, or Digital Ocean app platform.

Third-party platforms such as Heroku, Hugging Face Spaces, railway.app, render.com, or fly.io also exist for deploying your services directly from code in repositories, which abstract away certain decisions from you so that you can deploy faster and easier. Under the hood, third-party managed app platforms may be using the infrastructure of main cloud providers like Azure, Google, or AWS.

The main benefit of deploying to managed app platforms is the ease and speed of deployment, networking, scaling, and maintaining your services. Such platforms provide you with tools you need to secure, monitor, scale, and manage your services without having to worry about the underlying resource allocations, security, or software updates. They can let you configure load balancers, SSL certificates, domain mappings, monitoring, and staging environments so that you can focus more on application development than deployment workload of the project.

Because these platforms follow the platform-as-a-service (PaaS) payment model, you’ll be billed a higher rate compared to relying on your own infrastructure or using lower-level resources such as bare-bone VMs or serverless compute options. Alternative services may use the infrastructure-as-a-service (IaaS) payment model that often is more cost-effective.

Personally, I find managed app platforms a convenient way to deploy my applications without much hassle. If I’m working on a prototype and need to get my services available to users as fast as possible, managed app platforms is my first go-to option. Although, bear in mind that if you need access to GPU hardware for running inference services, you’ll have to rely on dedicated VMs, on-premises servers, or specialized AI platform services to serve your models. The app platforms can only provide CPU, memory, and disk storage for serving backend services or frontend applications.

###### Tip

A handful of managed cloud provider AI platforms include Azure Machine Learning Studio or Azure AI, Google Cloud Vertex AI Platform, AWS Bedrock and SageMaker, or IBM Watson Studio.

There are also third-party platforms for hosting your models including Hugging Face Inference Endpoints, Weights & Biases Platform, or Replicate.

Deploying from code repositories will often require you to add certain configuration files to the root of your project depending on which app platform you will be deploying to. The process also depends on whether the app platform supports the application runtime, libraries, and framework versions you’re using, so a successful deployment isn’t always guaranteed. It’s also often challenging to migrate to supported runtimes or versions.

Due to these unforeseen issues, many engineers are switching to containerization technologies such as Docker or Podman to package up and deploy their services. These containerized applications can then be deployed directly to any app platform supporting containers with guarantees that the application will run no matter what the underlying resources, runtime, or dependency versions are.

Deploying services with containers is now one of the most reliable strategies for shipping your applications to production for users to access.

## Deploying with Containers

A *container* is a loosely isolated environment designed for building and running applications. Containers can run your services quickly and reliably in any computing environment by packaging your code with all the required dependencies.

Under the hood, containers rely on an OS-virtualization method that enables them to run on physical hardware, in the cloud, on VMs, or across multiple operating systems.

###### Tip

Similar to managed app platforms and serverless functions, you can configure containers to automatically restart and self-heal, if your application exits for any reason.

Unlike VMs whose underlying technologies rely on virtualization, containers rely on containerization.

Containerization packages applications and their dependencies into lightweight, isolated units that share the host OS kernel. On the other hand, virtualization enables running multiple operating systems on a single physical machine using hypervisors. Therefore, unlike virtual machines, containers don’t virtualize hardware resources. Instead, they run on top of a container runtime platform that abstracts the resources, making them lightweight (i.e., as low as a few megabytes to store) and faster than VMs since they don’t require a separate OS per container.

###### Note

In essence, virtualization is about abstracting hardware resources on the host machine while containerization is about abstracting the operating system kernel and running all application components inside an isolated unit called a *container*.

[Figure 12-3](#deployment_container_architecture) compares the virtualization and containerization system architectures.

![bgai 1203](assets/bgai_1203.png)

###### Figure 12-3\. Comparison of containerization and virtualization system architectures

The main benefit from using containers is their *portability*, *boot-up speed*, *compactness*, and *reliability* across various computing environments, as they don’t require a guest OS and a hypervisor software layer.

This makes them perfect for deploying your services with minimal resources, deployment effort and overheads. They boot up faster than a VM, and scaling them is also more straightforward. You can add more containers to *horizontally scale* your services.

To help with containerizing your applications, you can rely on platforms such as Docker that have been battle-tested across the MLOps and DevOps communities.

# Containerization with Docker

Docker is a containerization platform used to build, ship, and run containers. At the time of writing, Docker has around [22% market share](https://oreil.ly/A5x63) in the virtualization platforms market with more than 9 million developers and [11 billion monthly image downloads](https://oreil.ly/8-wx4), making it the most popular containerization platform. Many server environments and cloud providers support Docker within many variants of Linux and Windows server.

Chances are if you need to deploy your GenAI services, the easiest and most straightforward option will be to use Docker to containerize your application. However, to get comfortable with Docker, you need to understand its architecture and the underlying subsystems such as storage and networking.

## Docker Architecture

The Docker system is composed of an engine, a client, and a server:

Docker engine

The engine consists of several components including a client and a server running on the same host OS.

Docker client

Docker comes with both a *CLI tool* named `docker` and a graphical user interface (GUI) application called *Docker Desktop*. Using the client-server implementation, the Docker client can communicate with the local or a remote server instance using a REST API to manage containers by running commands such as running, stopping, and terminating containers. You can also use the client to pull images from an image registry.

Docker server

The server is a *daemon* named `dockerd`. The Docker daemon responds to the client HTTP requests via the REST API and can interact with other daemons. It’s also responsible for tracking the lifecycle of containers.

The Docker platform also allows you to create and configure objects such as *networks*, *storage volumes*, *plug-ins*, and service objects to support your deployments.

Most important, to containerize your applications with Docker, you’ll need to build Docker images.

A *Docker image* is a portable package containing software and acts as a recipe for creating and running your application containers. In essence, a container is an in-memory instance of an image.

###### Tip

A container image is *immutable*, so once you’ve built one, you can’t change it. You can only add to an image and not subtract. You’ll have to re-create a new one if you want to apply changes.

Docker images are the first step toward containerizing your services as you’ll learn in the next section.

## Building Docker Images

Let’s imagine you have a small GenAI service using FastAPI, as shown in [Example 12-4](#docker_app), that you want to containerize.

##### Example 12-4\. A simple GenAI FastAPI service

```py
# main.py

from fastapi import FastAPI
from models import generate_text ![1](assets/1.png)

app = FastAPI()

@app.post("/generate")
def generate_text(prompt: str):
    return generate_text(prompt)
```

[![1](assets/1.png)](#co_deployment_of_ai_services_CO1-1)

Assume that the `generate_text` function is calling a model provider API or an external model server.

To build this application into a container image, you’ll need to write instructions in a text file called a *Dockerfile*. Inside this Dockerfile, you can specify the following components:

*   The *base* image to create a new image from, supplying the OS and environment upon which additional application layers are built

*   Commands to update the guest OS and install additional software

*   Build artifacts to include such as your application code

*   Services to expose like storage and networking configuration

*   The command to run when the container starts

[Example 12-5](#containers_dockerfile) illustrates how to build an application image in a Dockerfile.

##### Example 12-5\. Dockerfile to containerize a FastAPI application

```py
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base ![1](assets/1.png)

WORKDIR /code ![2](assets/2.png)

COPY requirements.txt . ![3](assets/3.png)

RUN pip install --no-cache-dir --upgrade -r requirements.txt ![4](assets/4.png)

COPY . . ![5](assets/5.png)

EXPOSE 8000 ![6](assets/6.png)

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] ![7](assets/7.png)
```

[![1](assets/1.png)](#co_deployment_of_ai_services_CO2-1)

Use the official Python 3.12 slim image as the `base` image.^([1](ch12.html#id1291))

[![2](assets/2.png)](#co_deployment_of_ai_services_CO2-2)

Set the working directory inside the container to `/code`.

[![3](assets/3.png)](#co_deployment_of_ai_services_CO2-3)

Copy the `requirements.txt` file from the host to the current directory in the container.

[![4](assets/4.png)](#co_deployment_of_ai_services_CO2-4)

Install the Python dependencies listed in `requirements.txt` without using the cache.

[![5](assets/5.png)](#co_deployment_of_ai_services_CO2-5)

Copy all files from the host’s current directory to the current directory in the container.

[![6](assets/6.png)](#co_deployment_of_ai_services_CO2-6)

Inform Docker daemon that the application inside the container is listening on `8000` at runtime. The `EXPOSE` command doesn’t automatically map or allow access on ports.^([2](ch12.html#id1292))

[![7](assets/7.png)](#co_deployment_of_ai_services_CO2-7)

Run the `uvicorn` server with the application module and host/port configuration, when container is launched.

We won’t be covering the full [Dockerfile specification](https://oreil.ly/8fJ6l) in this chapter. However, notice how each command changes the image structure that enables you to run your full GenAI services within a container.

You can use the `docker build` command to build the image in [Example 12-5](#containers_dockerfile):

```py
$ docker build -t genai-service .
```

Notice the steps listed in the output. When each step executes, a new layer gets added to the image you’re building.

Once you have a container image, you can then use container registries to store, share, and download images.

## Container Registries

To store and distribute images in a version-controlled environment, you can use *container registries*, which include both the public or private flavors.

*Docker Hub* is a managed software-as-a-service (SaaS) container registry for storing and distributing images you create.

Docker Hub is public by default. However, you can also use self-hosted or cloud provider private registries such as Azure Container Registry (ACR), AWS Elastic Container Registry (ECR), or Google Cloud Artifact Registry.

You can view the full Docker platform system architecture in [Figure 12-4](#docker_architecture).

![bgai 1204](assets/bgai_1204.png)

###### Figure 12-4\. Docker platform system architecture

As you can see in [Figure 12-4](#docker_architecture), the Docker daemon manages containers and images. It creates containers from images and communicates with the Docker client, handling commands to build and run images. The Docker daemon can also pull images from or push them to a registry (e.g., Docker Hub) that contains images like Ubuntu, Redis, or PostgreSQL.

Using the Docker Hub registry, you can access other contributed images alongside distributing and version controlling your own. Registries like Docker Hub play a crucial role in scaling your services as container orchestration platforms like Kubernetes need access to registries to pull and run multiple container instances from images.

You can pull public images from Docker Hub using the `docker pull` command:

```py
$ docker image pull python:3.12-slim

bookworm: Pulling from library/python
Digest: sha256:3f1d6c17773a45c97bd8f158d665c9709d7b29ed7917ac934086ad96f92e4510
Status: Downloaded newer image for python:3.12-slim
docker.io/library/python:3.12-slim
```

When you push and pull images, you’ll need to specify a *tag* using the `<name>:<tag>` syntax. If you don’t provide a tag, Docker engine will use the `latest` tag by default.

Aside from pulling, you can also store your own images in container registries. First, you need to build and tag your image with both a version label and the image repository URL:

```py
$ docker build -t genai-service:latest .

$ docker image tag genai-service:latest docker.io/myrepo/genai-service:latest
```

Once your image is built and tagged, you can then push it to the Docker Hub container registry using the `docker push` command. You may need to log in first to authenticate with the hub:

```py
$ docker login

$ docker image push docker.io/myrepo/genai-service:latest

195be5f8be1d: Pushed
```

###### Warning

Be careful that during a push, you don’t overwrite the tag for an image in many repositories. For instance, an image built and tagged `genai:latest` in a repository can be overwritten by another image tagged `genai:latest`.

Now that your image is stored in the registry, you can pull it down on another machine^([3](ch12.html#id1297)) or at a later time to run the image without the need to rebuild it.

## Container Filesystem and Docker Layers

When building the image, Docker uses a special filesystem called the `Unionfs` (stackable unification filesystem) to merge the contents of several directories (i.e., *branches* or in Docker terminology *layers*), while keeping their physical content separate.

Using `Unionfs`, directories of distinct filesystems can be combined and overlaid to form a single coherent virtual filesystem, as shown in [Figure 12-5](#docker_unionfs).

![bgai 1205](assets/bgai_1205.png)

###### Figure 12-5\. Unified virtual filesystem from multiple filesystems

Using the `Unionfs`, Docker can add or remove branches as you build out your container filesystem from an image.

To illustrate the mechanism of layered architecture in containers, let’s review the image from [Example 12-5](#containers_dockerfile).

When building the image using [Example 12-5](#containers_dockerfile), you’re layering a Python 3.12 base image running on a Linux distribution on top of a root filesystem. Next, you’re adding *requirements.txt* on top of the Python base image and then installing dependencies on top of *requirements.txt* layer. You then add a new layer by coping the content of your project directory into the container, layering it on top of everything else. Finally, when you start the container with the `uvicorn` command, you add a final writable layer as part of the container filesystem. As a result, the ordering of layers becomes important when building Docker images.

[Figure 12-6](#docker_branches) shows the layered filesystem architecture.

![bgai 1206](assets/bgai_1206.png)

###### Figure 12-6\. Layered Unionfs filesystem architecture

In [Example 12-5](#containers_dockerfile), each of the command steps is creating a cached image as the build process finalizes the container image. To run commands, intermediate containers are created and then automatically deleted after. The underlying cached image is kept on the build host and isn’t removed. These temporary images are layered over the previous image and combined into a single image once all steps are completed. This optimization allows future builds to reuse these images to speed up build times.

At the end, the container will comprise one or more image layers and a final ephemeral container layer (i.e., that won’t be persisted) when the container is destroyed.

## Docker Storage

In this section, you will learn about various Docker storage mechanisms. During the development of your services as containers, you can use these tools to manage data persistence, sharing data between containers and maintaining state between container restarts.

When working with containers, your application may need to write data to the disk, which will persist in an *ephemeral* storage. Ephemeral storage is a short-lived, temporary storage deleted once the container is stopped, restarted, or removed. If you restart your container, you’ll notice that previously persisted data is no longer available. Under the hood, Docker writes the runtime data to an ephemeral writable container layer in the container’s virtual filesystem.

###### Warning

You’ll lose all your application generated data and log files you’ve written to disk during a container’s runtime if you rely on the container’s default storage configuration.

To avoid loss of application runtime data and logs, you have several storage options available that enable you to persist data during a container’s lifetime. During development, you can use *volumes* or *bind mounts* to persist data to the host OS filesystem or rely on local databases for persisting data.

[Table 12-1](#docker_storage_options) shows the Docker storage mount options.

Table 12-1\. Docker storage mounts

| Storage | Description | Use cases |
| --- | --- | --- |
| Volumes | I/O optimized and preferred storage solution. Managed by Docker and stored in a specific location on the host but decoupled from the host filesystem structure. | If you need to store and share data across multiple containers.If you don’t need to modify files or directories from the host. |
| Bind mounts | Mount files or directories on host into the container but have limited functionality compared to volumes. | If you want both containers and host processes to access and modify host’s files and directories. For instance, during local development and testing. |
| Temporary (tmpfs) mounts | Stores data in the host’s memory (RAM) and never written to the container or host’s filesystem. | If you need high-performance temporary storage for sensitive or nonstateful data that won’t persist after the container stops. |

[Figure 12-7](#docker_storage_mounts) shows the different types of mounts.

![bgai 1207](assets/bgai_1207.png)

###### Figure 12-7\. Docker storage mounts

We’ll now study each storage option in detail so you can simulate your production environment locally with Docker containers using the appropriate storage. When deploying containers to production within a cloud environment, you can use a database or cloud storage offering for persisting data instead of Docker volumes or bind mounts to centralize storage across multiple containers.

### Docker volumes

Docker allows you to create isolated *volumes* for persisting application data between container runtimes. To create a volume, you can run the following command:

```py
$ docker volume create -n data
```

Once created, you can use volumes to persist data between container runs. Volumes also allow you to persist data when you use database and memory store containers.

###### Warning

Restarting a database container with new environment variables may not be enough to reset them with new settings.

Some database systems may require you to re-create the container volume if you need to update settings like administrator user credentials.

By default, any volumes you create will be stored on the host machine filesystem until you explicitly remove them via the `docker volume remove` command.

### Bind mounts

In addition to volumes, you can also use filesystem mappings via volume *bind mounts* that map directories residing on the host filesystem to the container filesystem, as shown in [Figure 12-8](#docker_bind_mounts).

![bgai 1208](assets/bgai_1208.png)

###### Figure 12-8\. Bind mounts between host filesystem and a container

The mounts happen as you start your container. With the mounted directories, you can then directly access them from within the container. You can read and persist data to the mounted directories as you run and stop your containers.

To run a container with a volume bind mount, you can use the following command:

```py
$ docker run -v src:/app genai-service
```

Here, the `-v` flag allows you to map the host directory to a container directory using the `<host_dir>:<container_dir>` syntax.

###### Warning

The functionality of the `COPY` command you use in a Dockerfile is different from directory mounting.

The former makes a separate copy of a host directory into the container during the image build process while the latter allows you to access and update the mapped host directory from within the container.

This means that if you’re not careful, you can unintentionally modify or delete all your original files on the host machine permanently, from within the container.

Bind mount volumes can still be useful in a local development environment. As you change the source code of your services, you’ll be able to observe the real-time impact of modifications on the running application containers.

### Temporary mounts (tmpfs)

If you have some nonpersistent data such as model caches or sensitive files that you don’t need to store permanently, you should consider using temporary *tmpfs mounts*.

This temporary mount will only persist the data to the host memory (RAM) during the container’s runtime and increases the container’s performance by avoiding writes into the container’s writeable layer.

When containerizing GenAI applications, you can use temporary mounts to store cached results, intermediate model computations, temporary files, and session-specific logs that you won’t need once the container stops.

###### Tip

The container’s writeable layer is tightly coupled with the host machine through a storage driver to implement the union filesystem. Therefore, writing to the container’s writable layer reduces performance due to this additional layer of abstraction.

Instead, you can use data volumes for persistent storage that writes directly to the host filesystem or tmpfs mounts for temporary in-memory storage.

Unlike bind mounts and volumes, you can’t share the tmpfs mount between containers, and the functionality is available only on Linux systems. In addition, if you adjust directory permissions on tmpfs mounts, they can reset when the container restarts.

Here are a few other use cases of tmpfs mounts:

*   Temporarily storing data caches, API responses, logs, test data, configuration files, and AI model artifacts in-memory

*   Avoiding I/O writes to disks while working with library APIs that require file-like objects

*   Simulating high-speed I/O with rapid file access and writes

*   Preventing excessive or unnecessary disk writes if you need temporary directories

To set a tmpfs mount, you can use the following command:

```py
$ docker run --tmpfs /cache genai-service
```

Here, you are setting a tmpfs mount on the `/cache` directory for model caches, which will cease to exist once the container stops.

### Handling filesystem permissions

A big source of frustration and a security consideration for many developers new to Docker is managing directory permissions when using filesystem bind mounts between the host OS and the container.

By default, Docker runs containers as the `root` user leading to containers having full read/write access to mounted directories on the host OS. If the `root` user inside the container creates directories or files, they will be owned by `root` on the host as well. You can then face permission issues if you have a nonroot user account on the host when you try to access or modify these directories or files.

###### Warning

Running containers as the default `root` user is also a great security risk if a malicious actor gets access to the container since they’ll have access to the host system as `root`. Additionally, if you run a compromised image, you might risk executing malicious code on your host system with `root` privileges.

To mitigate permission issues when running containers with bind mounts, you can use the `--user` flag to run the container as a nonroot user:

```py
$ docker run --user genai-service
```

Alternatively, you can create and switch to a nonroot user within the final layers of the image build inside the Dockerfile, as shown in [Example 12-6](#docker_permissions).

##### Example 12-6\. Creating and switching to nonroot user when building container images (Ubuntu/Debian containers only)

```py
ARG USERNAME=fastapi ![1](assets/1.png)
ARG USER_UID=1001
ARG USER_GID=1002

RUN groupadd --gid $USER_GID $USERNAME \ ![2](assets/2.png)
    && adduser \
    --disabled-password \
    --shell "/sbin/nologin" \ ![3](assets/3.png)
    --gecos "" \
    --home "/nonexistent" \
    --no-create-home \ ![4](assets/4.png)
    --uid "${UID}" \
    --gid $USER_GID
    $USERNAME ![5](assets/5.png)

USER $USERNAME ![6](assets/6.png)

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

[![1](assets/1.png)](#co_deployment_of_ai_services_CO3-1)

Use build arguments to specify variables during the image build.

[![2](assets/2.png)](#co_deployment_of_ai_services_CO3-2)

Create a user group with the given `USER_GID`.

[![3](assets/3.png)](#co_deployment_of_ai_services_CO3-3)

Disable user login completely including password-based login.

[![4](assets/4.png)](#co_deployment_of_ai_services_CO3-4)

Avoid creating a home directory for the user.

[![5](assets/5.png)](#co_deployment_of_ai_services_CO3-5)

Create a nonroot user account with the given `$USER_UID` and assign it to the newly created `USER_GID` group. Set the name of the user account to `fastapi`.

[![6](assets/6.png)](#co_deployment_of_ai_services_CO3-6)

Switch to the nonroot `fastapi` user.

###### Tip

Often, you’ll need to install packages or add configurations that require privileged disk access or permissions. You should only switch to a nonroot user at the end of an image build once you’ve completed such installations and configurations. Avoid switching back and forth between root and nonroot users to prevent any unnecessary complexity and excess image layers.

If you hit issues with creating new groups or users in [Example 12-6](#docker_permissions), try changing the `USER_UID` and `USER_GID` as those IDs may already be in use by another nonroot user in the image.

Let’s assume that during the image creation, the `root` user in the container has created the `myscripts` folder. You can inspect filesystem permissions using the `ls -l` command, which returns the following output:

```py
total 12
drw-r--r-- 2 root root 4096 Oct  1 10:00 myscripts
```

You can read permissions `drwxr-xr-x` for the `myscripts` directory using the following breakdown:

*   `d`: Specifies that `myscripts` is a directory; otherwise would show a `-`.

*   `rwx`: Owner `root` user can (r)read, (w)rite, and e(x)ecute files in this directory.

*   `r--`: Group `root` members can perform (r)ead-only operations but can’t write or execute any files.

*   `r--`: Everyone else can read the file but cannot write to or execute it.^([4](ch12.html#id1317))

If you want to set ownership or permissions on the `myscripts` directory, you can use the `chmod` or `chown` commands in Linux systems.

Use the `chown` command to change the directory owner on host so that you can edit the files in your code editor:

```py
# Set file or directory ownership
$ sudo chown -R username:groupname mydir
```

Alternatively, if you only need to execute the scripts in the `myscripts` directory, use the `chmod` command to change the file or directory permissions:

```py
# Set execute permissions using flags
$ sudo chmod -R +x myscripts

# Set execute permissions in a numeric form
$ sudo chmod -R 755 myscripts
```

###### Tip

The `-R` flag will recursively set the ownership or permissions on a nested directory.

This command will allow `root` group members and other users to execute files in the `myscripts` directory. Others can execute the files only if they use the `bash` command. However, only the owner can modify them.

If you inspect the filesystem permissions again using `ls -l`, you’ll see the following output:

```py
total 12
drwxr-xr-x 2 root root 4096 Oct  1 10:00 myscripts
```

*   `rwx`: Owner `root` user can still (r)read, (w)rite, and e(x)ecute files in this directory.

*   `r-x`: Group `root` members can perform (r)ead and e(x)ecute operations but can’t modify any files.

*   `r-x`: Anyone else can’t modify files in `myscripts` directory but can read and execute them.

You can use [Example 12-7](#docker_permissions_execute) to set permissions when creating directories inside an image.

##### Example 12-7\. Creating scripts folder and allowing files to be executed (Ubuntu/Debian containers only)

```py
RUN mkdir -p scripts

COPY scripts scripts

RUN chmod -R +x scripts
```

The instructions in [Example 12-7](#docker_permissions_execute) will allow you to configure permissions to execute files in the `scripts` directory from within the container.

###### Warning

When using container volumes, be careful with mount bindings as they replace the permissions inside the container with those from the host filesystem.

The most frustrating issues when working with containers will be related to filesystem permissions. Therefore, knowing how to set and correct file permissions will save you hours of development when working with containers that produce or modify artifacts on the host machine.

## Docker Networking

Docker networking is one of the hardest concepts to grasp in multicontainer projects. This section covers how Docker networking works and how to set up local containers to communicate, simulating production environments during development.

Often, when you’re deploying to production environments in the cloud, you configure networking using the cloud provider’s solutions. However, if you need to connect containers in a development environment for local testing or deploying on on-premises resources, then you’ll benefit from understanding how Docker networking works.

If you’re developing GenAI services that interact with external systems like databases, chances are you’ll be using multiple containers; one for your application and one for running each of your databases or external systems.

Docker ships with a networking subsystem that allows containers to connect with each other on the same or different hosts. You can even connect containers via internet-facing hosts.

When you create containers using the `docker run` command, they’ll have networking enabled by default on a *bridge network* so that they can make outgoing connections. However, they won’t expose or publish their ports to the outside world.

###### Warning

With the default settings, Docker interacts with the OS kernels to configure *firewall rules* (e.g., `iptables` and `ip6tables` rules on Linux) to implement network isolation, port publishing, and filtering.

Since Docker can override these firewall rules, if you have a port on host like `8000` closed, Docker can force it open and expose it outside the host machine when you run a container with the `-p 8000:8000` flag. To prevent such an exposure, a solution is to run containers using `-p 127.0.0.1:8000:8000`.

For the networking subsystem to function, Docker uses *networking drivers*, as shown in [Table 12-3](#docker_networking_drivers).

Table 12-3\. Docker networking drivers

| Driver | Description | Use case |
| --- | --- | --- |
| Bridge (default) | Connects containers running on the same Docker daemon host. User-defined networks can leverage an embedded DNS server. | Control container communication in isolated Docker networks with a simple setup. |
| Host | Removes the isolation layer between containers and the host system, so any TCP/UDP connections are accessible directly via host network such as the localhost without the need to publish ports. | Simplify access to container from the host network (e.g., localhost) or when a container needs to handle a large range of ports. |
| None | Disables all networking services and isolates running containers within the Docker environment. | Isolate containers from any Docker and non-Docker process for security reasons. Network debugging or simulating outages. Resource isolation and transient containers for short-lived processes. |
| Overlay | Connects containers across multiple hosts/engines or in a *Docker Swarm* cluster.**Note:** Docker engine has *swarm* mode that enables container orchestration via *clusters* of Docker daemons/engines. | Remove the need for OS-level routing when connecting containers across Docker hosts. |
| Macvlan | Assigns mac addresses to containers as if they’re physical devices.Misconfiguration may lead to unintentional degradation of your network due to IP address exhaustion, leading to VLAN spread (large number of mac addresses) or promiscuous mode (overlapping addresses). | Used in legacy systems or applications that monitor network traffic that expect to be directly connected to a physical network. |
| IPVlan | Gives you total control over container IPv4 and IPv6 addressing, providing easy access to external services with no need for port mappings. | Advanced networking setup that bypasses the traditional Linux bridge for isolation, enhanced performance and simplified networking topology. |

To ensure your containers can communicate together, you may need to specify networking settings and drivers. You can select a networking driver that matches your use case based on [Table 12-3](#docker_networking_drivers).

###### Note

Some of these drivers may not be available depending on the platform/host OS you’re running Docker on (Windows, Linux, or macOS host).

The most commonly used network drivers are bridge, host, and none. You likely won’t need to use other drivers (e.g., overlay, Macvlan, IPVlan) unless you need more advanced networking configurations.

[Figure 12-9](#docker_networking_drivers_viz) visualizes the functionality of the bridge, host, none, overlay, Macvlan, and IPVlan drivers.

![bgai 1209](assets/bgai_1209.png)

###### Figure 12-9\. Docker networking drivers

Let’s explore these networking drivers in more detail.

### Bridge network driver

The bridge network driver connects containers by creating a default bridge network `docker0` and associating containers with it and the host’s main network interface, unless otherwise specified. This will allow your containers to access the host network (and the internet) plus allow you to access the containers.

You can view the networks using the `docker network ls` command:

```py
$ docker network ls
NETWORK ID     NAME      DRIVER    SCOPE
72ec0b2e6034   bridge    bridge    local
53ec40b3c639   host      host      local
64368b7baa5f   none      null      local
```

The *network bridge* in Docker is a link layer software device running within the host machine’s kernel, allowing linked containers to communicate while isolating non-connected containers. The bridge driver automatically installs rules in the host machine so that containers on different bridge networks can’t communicate directly.

###### Tip

Bridge networks only apply to containers running on the same Docker engine/daemon host. To connect containers running on other daemon hosts, you can manage routing at the host OS layer or use an *overlay* driver.

In addition to default bridge networks, you can create your own custom networks, which can provide superior isolation and packet routing experience.

#### Configure user-defined bridge networks

If you need more advanced or isolated networking environments for your containers, you can create a separate user-defined network.

User-defined networks are superior to the default bridge networks as they provide better isolation. In addition, containers can resolve each other by name or alias on user-defined bridge networks unlike the default network where they can only communicate via IP addresses.

###### Warning

If you run containers without specifying `--network`, they’ll be attached to the default bridge network. This can be a security issue as unrelated services are then able to communicate and access each other.

To create a network, you can use the `docker network create` command, which will use `--driver bridge` flag by default:

```py
$ docker network create genai-net
```

###### Note

When you create user-defined networks, Docker uses the host OS tools to manage the underlying network infrastructure, such as adding or removing bridge devices and configuring `iptables` rules on Linux.

Once the network is created, you can list the networks using the `docker network ls` command:

```py
$ docker network ls
NETWORK ID     NAME         DRIVER    SCOPE
72ec0b2e6034   bridge       bridge    local
6aa21632e77e   genai-net    bridge    local
```

The network topology will now look like [Figure 12-10](#docker_networking_isolated).

![bgai 1210](assets/bgai_1210.png)

###### Figure 12-10\. Isolated bridge networks

When you run containers, you can now attach them to the created network using the `--network genai-net` flag:

```py
$ docker run --network genai-net genai-service
$ docker run --network genai-net postgresql
```

###### Warning

On Linux, there is a limit of 1,000 containers that can connect to a single bridge network due to the Linux kernel restrictions. Linking more containers to a single bridge network can make it unstable and break inter-container communication.

Both your containers can now access each other on your better isolated `genai-net` user-defined network with automatic *DNS resolution* between containers.

#### Embedded DNS

Docker leverages an embedded DNS server with user-defined networks, as shown in [Figure 12-11](#docker_networking_bridge_dns), to map internal IP addresses so that containers can reach one by name.

![bgai 1211](assets/bgai_1211.png)

###### Figure 12-11\. Embedded DNS

For instance, if you name your application container as `genai-service` and your database container as `db`, then your `genai-service` container can communicate with the database by calling the `db` hostname.

###### Warning

You can’t access the `db` container from outside of the Docker bridge network by its name, as the embedded DNS server is not visible to the host machine.

Instead, you can expose a container port `5432` and access the `db` container using host’s network (e.g., via `localhost:5432`).

Let’s discuss how you can publish container ports to the outside environment such as the host machine next.

#### Publishing ports

When you run containers in a network, they automatically expose ports to each other.

If you need to access containers from the host machine or non-Docker processes on different networks, you’ll need to expose the container ports by publishing them using the `--publish` or `-p` flag:

```py
$ docker run -p 127.0.0.1:8000:8000 myimage
```

This command allows you to create a container with exposed port `8000` mapped to `8000` port on the host machine (e.g., localhost) using the `<host_port>:​<con⁠tainer_port>` syntax.

When you don’t specify a container port, Docker will publish and map port `80` by default.

###### Warning

Always double-check ports you want to expose and avoid publishing container ports that are already in use on your host machine. Otherwise, there’ll be *port conflicts* leading to requests being routed to conflicting services, which will also be time-consuming to troubleshoot.

If using bridge networks and port mappings are causing you a lot of trouble, you can also use the *host* networking driver for connecting your containers, albeit without the same isolation and security benefits of bridge networks.

### Host network driver

A *host* network driver is useful for cases where you want to improve performance, when you want to avoid the container port mapping, or when one of your containers needs to handle a large number of ports.

Running a container with the host driver is as simple as using the `--net=host` flag with the `docker run` command:

```py
$ docker run --net=host genai-service
```

In host networking, containers share the host machine’s network namespace, meaning that containers won’t be isolated from the Docker host. Therefore, containers won’t be allocated their own IP address.

###### Warning

As soon as you enable the host network driver, previously published ports will be discarded, as containers won’t have their own IP address.

The host network driver is more performant because it doesn’t need a *network address translation* (NAT) for mapping IP addresses from one namespace (containers) to another (host machine) and avoids creating a *user-land proxy* (i.e., port forwarding) for each port. However, host networking is only supported with Linux—and not Windows—containers. In addition, containers won’t have access to the network interfaces of the host so can’t bind to host’s IP addresses, leading to added complexity in the network configuration you need.

### None network driver

If you want to completely isolate the networking stack of a container, you can use the `--network none` flag when starting the container. Within the container, only the loopback device is created, a virtual network interface that the container uses to communicate with itself. You can specify the none network driver using the following command:

```py
$ docker run --network none genai-service
```

These are a few cases where isolating containers are useful:

*   Applications handling highly sensitive data or running critical processes

*   Where there’s a higher risk of network-based attacks or malware

*   Performing network debugging and simulating network outages by eliminating external interference

*   Running stand-alone containers without external dependencies can run independently

*   Operating transient containers for short-lived processes to minimize network exposure

Generally, use the none network driver if you need to isolate containers from any Docker and non-Docker processes for security reasons.

## Enabling GPU Driver

If you have an NVIDIA graphics card with the CUDA toolkit and necessary drivers installed, then you can use the `--gpus=all` flag to enable GPU support for your containers in Docker.^([5](ch12.html#id1336))

To test that your system has the necessary drivers and supports GPU in Docker, run the following command to benchmark your GPU:

```py
$ docker run --rm -it \
             --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody \
             -gpu \
             -benchmark

> Windowed mode
> Simulation data stored in video memory
> Single precision floating point simulation
> 1 Devices used for simulation
MapSMtoCores for SM 8.9 is undefined.  Default to use 128 Cores/SM
MapSMtoArchName for SM 8.9 is undefined.  Default to use Ampere
GPU Device 0: "Ampere" with compute capability 8.9

> Compute 8.9 CUDA device: [NVIDIA GeForce RTX 4090]
131072 bodies, total time for 10 iterations: 75.182 ms
= 2285.102 billion interactions per second
= 45702.030 single-precision GFLOP/s at 20 flops per interaction
```

###### Tip

You can also use the NVIDIA system management interface `nvidia-smi` tool to help manage and monitor NVIDIA GPU devices.

Deep learning frameworks such as `tensorflow` or `pytorch` can automatically detect and use the GPU device when running your applications in a GPU-enabled container. This includes Hugging Face libraries such as `transformers` that lets you self-host language models.

If using the `transformers` package, make sure to also install the `accelerate` library:

```py
$ pip install accelerate
```

You can now move the model to GPU before it’s loaded in CPU by using `device_map='cuda'`, as shown in [Example 12-8](#docker_gpu).

##### Example 12-8\. Transferring Hugging Face models to the GPU

```py
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="cuda"
)
```

You should be able to run the predictions on the GPU by passing the `--gpus=all` flag to `docker run`.

## Docker Compose

In multicontainer environments, you can use the *Docker Compose* tool for defining and running application containers for a streamlined development and deployment experience.

Using Docker Compose can help you simplify managing several containers, networks, volumes, variables, and secrets with a single *YAML configuration file*. This simplifies the complex task of orchestrating and coordinating various containers, making it easier to manage and replicate your services across different application environments using environment variables. You can also share the YAML file with others so that they can replicate your container environment. Additionally, it caches configurations to prevent re-creating containers when you restart services.

[Example 12-9](#docker_compose) shows an example YAML configuration file.

##### Example 12-9\. Docker Compose YAML configuration file

```py
# compose.yaml

services: ![1](assets/1.png)
  server:
    build: . ![2](assets/2.png)
    ports:
      - "8000:8000"
    environment:
      SHOW_DOCS_IN_PRODUCTION: $SHOW_DOCS_IN_PRODUCTION
      ALLOWED_CORS_ORIGINS: $ALLOWED_CORS_ORIGINS
    secrets:
       - openai_api_token ![3](assets/3.png)
    volumes:
      - ./src/app:/code/app
    networks:
      - genai-net ![4](assets/4.png)

  db:
    image: postgres:12.2-alpine
    ports:
      - "5433:5432"
    volumes:
      - db-data:/etc/data
    networks:
      - genai-net

volumes:
  db-data:
    name: "my-app-data"

networks:
  genai-net:
    name: "genai-net"
    driver: bridge

secrets:
  openai_api_token:
    environment: OPENAI_API_KEY
```

[![1](assets/1.png)](#co_deployment_of_ai_services_CO4-1)

Create the containers alongside the associated volumes, networks, and secrets.

[![2](assets/2.png)](#co_deployment_of_ai_services_CO4-2)

Use the Dockerfile located at the same directory as the Compose file to build the `server` image.

[![3](assets/3.png)](#co_deployment_of_ai_services_CO4-3)

Use Docker secrets to mask sensitive data like API keys within the container shell environment.

[![4](assets/4.png)](#co_deployment_of_ai_services_CO4-4)

Create a bridge `genai-net` network and attach both `server` and `db` containers to it.

###### Tip

If you have Docker objects like volumes and networks that you’re managing yourself, you can tag them with `external: true` in the compose file so that Docker Compose doesn’t manage them.

Once you have a `compose.yaml` file, you can then use simple compose commands to manage your containers:

```py
# Start services defined in compose.yaml
$ docker compose up

# Stop and remove running services (won't remove created volumes and networks)
$ docker compose down

# Monitor output of running containers
$ docker compose logs

# List all running services with their status
$ docker compose ps
```

You can use these commands to start/stop/restart services and view their logs or container statuses. Additionally, you can edit the Compose file shown in [Example 12-9](#docker_compose) to use `watch` so that your services are automatically updated as you edit and save your code.

[Example 12-10](#docker_compose_watch) shows how to use the `watch` instruction on a given directory.

##### Example 12-10\. Enabling Docker Compose `watch` on a given directory

```py
services:
  server:
    # ...
    develop:
      watch:
        - action: sync
          path: ./src
          target: /code
```

Whenever a file changes in the `./src` folder on your host machine, Compose will sync its content to `/code` and update the running application (server service) without restarting them.

You can then run the `watch` process using `docker compose watch`:

```py
$ docker compose watch

[+] Running 2/2
 ✔ Container project-server-1  Created     0.0s
 ✔ Container project-db-1      Recreated   0.1s
Attaching to db-1, server-1
         ⦿ watch enabled
...
```

Docker Compose `watch` allows for greater granularity than is practical with bind mounts, as shown in [Example 12-9](#docker_compose). For instance, it lets you ignore specific files or entire directories within the watched tree to avoid I/O performance issues.

Besides using Docker Compose `watch`, you can merge and override multiple Compose files to create a composite configuration tailored for specific build environments. Typically, the `compose.yml` file contains the base configurations, which can be overridden by an optional `compose.override.yml` file. For instance, as shown in [Example 12-11](#compose_override), you can inject local environment settings, mount local volumes, and create new a database service.

##### Example 12-11\. Merging and overriding Compose files for environment-specific build configurations

```py
# compose.yml

services: ![1](assets/1.png)
  server:
      ports:
        - 8000:8000
      # ...
      command: uvicorn main:app

# compose.override.yml

services: ![2](assets/2.png)
  server:
    environment:
      - LLM_API_KEY=$LLM_API_KEY
      - DATABASE_URL=$DATABASE_URL
    volumes:
      - ./code:/code
    command: uvicorn main:app --reload

  database:
    image: postgres:latest
    environment:
      - POSTGRES_DB=genaidb
      - POSTGRES_USER=genaiuser
      - POSTGRES_PASSWORD=secretPassword!
    volumes:
      - db_data:/var/lib/postgresql/data

networks:
  app-network:

volumes:
  db_data:
```

[![1](assets/1.png)](#co_deployment_of_ai_services_CO5-1)

The base Compose file contains instructions for running the production version of the application.

[![2](assets/2.png)](#co_deployment_of_ai_services_CO5-2)

Override base instructions by replacing the container start command, inject local variables, and add volume and networking configurations with a local database service.

To use these files, run the following command:

```py
$ docker compose up
```

Docker Compose will automatically merge configurations from both Compose files, applying the environment-specific settings from the override Compose file.

## Enabling GPU Access in Docker Compose

To access GPU devices with services managed by Docker Compose, you’ll need to add the instructions to the composed file (see [Example 12-12](#DockerComposeapp)).

##### Example 12-12\. Adding GPU configurations to the Docker Compose app service

```py
services:
  app:
    # ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1 ![1](assets/1.png)
              capabilities: [gpu]
```

[![1](assets/1.png)](#co_deployment_of_ai_services_CO6-1)

Limit the number of GPU devices accessible by the app service.

These instructions will give you more granular control over how your services should use your GPU resources.

## Optimizing Docker Images

If your Docker images grow in size, they’ll also be slower to run, build, and test in production. You’ll also be spending a lot of development time iterating over the development of the image.

In that case, it’s important to understand image optimization strategies, including how to use Docker’s layering mechanism to keep images lightweight and efficient to run, in particular with GenAI workloads.

These are a few ways to reduce image size and speed up the build process:

*   Using minimal base images

*   Avoiding GPU inference runtimes

*   Externalizing application data

*   Layering ordering and caching

*   Using multi-stage builds

Implementing these optimizations as shown in [Table 12-4](#build_optimization_impact) may reduce typical image sizes from several gigabytes to less than 1 GB. Similarly, build times can reduce from several minutes on average to less than a minute.

Table 12-4\. Impact of build optimization on a typical image^([a](ch12.html#id1343))

| Optimization step | Build time (seconds) | Image size (GB) |
| --- | --- | --- |
| Initial | 352.9 | 1.42 |
| Using minimal base images | 38.5 | 1.38 |
| Use caching | 24.4 | 1.38 |
| Layer ordering | 17.9 | 1.38 |
| Multi-stage builds | 10.3 | 0.034 (34 MB) |
| ^([a](ch12.html#id1343-marker)) Source: [warpbuild.com](https://www.warpbuild.com) |

Let’s review each in more detail with code examples for clarity.

### Use minimal base image

Base images allow you to start from a preconfigured image so you don’t have to install everything from scratch, including the Python interpreter. However, some base images available on the Docker Hub may not be suitable for production deployments. Instead, you’ll want to select the right base image with a minimal OS footprint to work from for faster builds and smaller image sizes, possibly with pre-installed Python dependencies and support for installing its various packages.

Alpine base images use a lightweight Alpine Linux distribution designed to be small and secure, containing only the *base minimum* essential tools to run your application, but this won’t support installing many Python packages. On the other hand, slim base images may use other Linux distributions like Debian or CentOS, containing the *necessary* essential tools for running applications that make them larger than Alpine base images.

###### Tip

Use slim base images if you care about build time and Alpine base images if you care about image size.

You can use the `slim` base images such as `python:3.12-slim` or even Alpine base images like `python:3.12-alpine` that can be as small as 71.4 MB. A bare-bones Alpine image can even go down to 12.1 MB. The following command shows a list of base images pulled from the Docker repository:

```py
$ docker image ls

REPOSITORY  TAG         IMAGE ID       CREATED         SIZE
alpine      3.20        3463e98c969d   4 weeks ago     12.1MB
python      3.12-alpine c6de2e87f545   6 days ago      71.4MB
python      3.12-slim   1ba4bc34383e   6 days ago      186MB
```

###### Tip

Standard-sized images typically contain a full Linux distribution like Ubuntu or Debian containing a variety of pre-installed packages and dependencies, making them suitable for local development but perhaps not production environments.

### Avoid GPU inference runtimes

In AI workloads where you’re serving ML/GenAI models, you may need to install deep learning frameworks, dependencies, and GPU libraries that can suddenly explode the footprint of your images. For instance, to make inferences on a GPU using the `transformers` library, you’ll need to install 3 GB of NVIDIA packages for GPU inference, 1.6 GB for the `torch` to perform the inference.

Unfortunately, you can’t reduce the image size if you need to use a GPU to perform an inference. However, if you can avoid GPU inference and just rely on CPUs, you may be able to reduce the image size by up to 10 times using the Open Neural Network Exchange (ONNX) runtime with model quantization.

As discussed in [Chapter 10](ch10.html#ch10), you can use the INT8 quantization with an ONNX model to benefit from model compression without much loss in output quality.

To switch from the GPU inference runtime to the ONNX runtime for Hugging Face transformer models, you can use the `transformers[onnx]` package:

```py
$ pip install transformers[onnx]
```

You can then export any Hugging Face transformer model checkpoint with default configurations to the ONNX format with `transformers.onnx`:

```py
$ python -m transformers.onnx --model=distilbert/distilbert-base-uncased onnx/
```

This command exports the `distilbert/distilbert-base-uncased` model checkpoint as an ONNX graph stored in `onnx/model.onnx`, which can be run with any Hugging Face model accelerator that supports the ONNX standard, as shown in [Example 12-13](#docker_onnx).

##### Example 12-13\. Model inference using the ONNX runtime with quantization

```py
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
session = InferenceSession("onnx/model.onnx")

inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np") ![1](assets/1.png)
output = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
```

[![1](assets/1.png)](#co_deployment_of_ai_services_CO7-1)

ONNX runtime expects `numpy` arrays as input.

Using a technique such as shown in [Example 12-13](#docker_onnx), you can downsize from image sizes between 5 and 10 GB to around 0.5 GB, which is a massive footprint reduction, significantly more cost-effective and scalable.

### Externalize application data

A core contributor to image size is copying models and application data into the image during build time. This approach increases both the build time and image size.

A better approach is to use volumes during local development and external storage solutions for downloading and loading models at application startup in production. In Kubernetes container orchestration environments, you can also use persistent volumes for model storage.

###### Tip

If your application container takes a long time to download data and model artifacts from an external source, your health checks may fail, and the hosting platform can kill your containers prematurely. In such cases, configure health check probes to wait longer or as a last resort, bake the model into the image.

### Layer ordering and caching

Docker uses a layered filesystem to create layers in an image for each instruction in the Dockerfile. These layers are like a stack, with each layer adding more content on top of the previous layers. Whenever a layer changes, that layer (and further layers) will need to be rebuilt for those changes to appear in the image (i.e., build cache must be invalidated).

A layer (i.e., a filesystem snapshot) is created if the instruction is writing or deleting files into the container’s union filesystem.

###### Tip

Dockerfile instructions that modify the filesystem like `ENV`, `COPY`, `ADD`, and `RUN` will contribute new layers to the build process, effectively increasing the image size. On the other hand, instructions such as `WORKDIR`, `ENTRYPOINT`, `LABEL`, and `CMD` that only update the image metadata don’t create any layers and any build cache.

After creation, each layer is then cached for reusability across image rebuilds if the instruction and files it depends on haven’t changed since the last build. Therefore, ideally, you want to write a Dockerfile that allows you to stop, destroy, rebuild, and replace containers with minimal setup and configuration.

There are a few techniques you can use to minimize and optimize these layers as much as possible.

#### Layer ordering to avoid frequent cache invalidation

Since changes to the earlier layers can invalidate the build cache leading to repeating steps, you should order your Dockerfile from the most stable (e.g., installations) to the most frequently changing or volatile (e.g., application code, configuration files).

Following this ordering, place the most stable yet expensive instructions (e.g., model downloads or heavy dependency installations) at the start of the Dockerfile, and volatile, fast operations (e.g., copying application code) at the bottom.

Imagine your Dockerfile file looks like this:

```py
FROM python:3.12-slim as base
# Changes to the
COPY . .
RUN pip install requirements.txt
```

Here you’re creating a layer by copying your working directory containing the application code into the image before downloading and installing dependencies.

If any one of source files changes, Docker builder will invalidate the cache causing the dependency installation to be repeated, which is expensive and can take several minutes to complete, if not cached by `pip`.

To avoid repeating expensive steps, you can logically order your Dockerfile instructions to optimize the layer caching by reordering instructions like these:

```py
FROM python:3.12-slim as base
COPY requirements.txt requirements.txt
RUN pip install requirements.txt
COPY . .
```

Now any changes to the source files won’t affect the long dependency installation step, drastically speeding up the build process.

#### Minimize layers

To keep image sizes small, you’ll want to minimize image layers as much as possible.

A simple technique to achieve this is to combine multiple `RUN` instructions into one. For instance, instead of writing multiple `RUN apt-get` installations, you can combine them into a single `RUN` command with `&&`:

```py
RUN apt-get update && apt-get install -y
```

This will avoid adding unnecessary layers and prevents caching issues with `apt-get update` using the *cache busting* technique.

Since the builder may potentially skip updating the package index, causing installations to fail or use outdated packages, using the `&&` ensures that the latest packages are installed if the package index is updated.

###### Tip

You can also use the `--no-cache` flag when using `docker build` to avoid cache hits and ensure fresh downloads of base images and dependencies on every build.

#### Keep build context small

The *build context* is the set of files and directories that’ll be sent to the builder to carry out the Dockerfile instruction. A smaller build context reduces the amount of data sent to the builder and lowers the chance of cache invalidation, resulting in faster builds.

When you use the `COPY . .` command in a Dockerfile to copy your working directory into an image, you may also add tool caches, development dependencies, virtual environments, and unused files into the build context. Not only the image size will be increased, but also the Docker builder will cache these unnecessary files. Any changes to these files will then invalidate the build, restarting the whole build process.

To prevent the unnecessary cache invalidation, you can add a *.dockerignore* file next to your Dockerfile, listing all files and directories that your services won’t need in production. As an example, here are items you can include in a *.dockerignore* file:

```py
**/.DS_Store
**/__pycache__
**/.mypy_cache
**/.venv
**/.env
**/.git
```

Docker builder will then ignore these files even when you run the `COPY` command across your entire working directory.

#### Use cache and bind mounts

You can use *bind mounts* to avoid adding unnecessary layers to the image and *cache mounts* to speed up subsequent builds.

Bind mounts temporarily include files in the build context for a single `RUN` instruction and won’t persist as image layers after. Cache mounts specify a persistent cache location that you can read and write data to across multiple builds.

Here is an example where you can download a pretrained model from Hugging Face into a mounted cache to optimize layer caching:

```py
RUN --mount=type=cache,target=/root/.cache/huggingface && \
    pip install transformers && \
    python -c "from transformers import AutoModel; \
 AutoModel.from_pretrained('bert-base-uncased')"
```

This `RUN` instruction creates a cache of the downloaded pretrained model at `/root/.cache/huggingface`, which can be shared across multiple builds. This helps avoid redundant downloads and optimizes the build process by reusing cached layers.

You can also use the `--no-cache-dir` flag when using the `pip` package manager to avoid caching altogether for minimizing image size. However, you’ll have a significantly slower build process as follow-on builds will need to redownload each time.

#### Use external cache

If you’re building and deploying containers using a CI/CD pipeline, you can benefit from an external cache hosted on a remote location. An external cache can drastically speed up the build process in CI/CD pipelines where builders are often ephemeral and build minutes are precious.

To use an external cache, you can specify the `--cache-to` and `--cache-from` options with the `docker buildx build` command:

```py
docker buildx build --cache-from type=registry,ref=user/app:buildcache .
```

Besides layer ordering and cache optimization, you can use multi-stage builds to significantly shrink your image sizes.

### Multi-stage builds

Using *multi-stage builds*, you can reduce the size of your final image by splitting out the Dockerfile instructions into distinct stages. Common stages can be reused to include shared components and serve as a starting point for further stages.

You can also selectively copy artifacts from one stage to another, leaving behind everything you don’t want in the final image. This ensures that only the required outputs are included in the final image from previous stages, avoiding any non-essential artifacts. Furthermore, you can also execute multiple build stages in parallel to speed up the build process of your images.

A common multi-stage build pattern is when you need a testing/development image and a slimmer production one with both starting from a shared first stage image. The development or testing image can include additional layers of tooling (i.e., compilers, build systems, and debugging tools) to support the required workflows.

Imagine you need to serve a bert transformer model from Hugging Face in a FastAPI service. You can write your Dockerfile instructions to use three distinct sequential stages.

The first stage downloads the transformer model into `/root/.cache/huggingface` and creates a Python virtual environment at `/opt/venv`:

```py
# Stage 1: Base
FROM python:3.11.0-slim as base

RUN python -m venv /opt/venv
RUN pip install transformers && \
    python -c "from transformers import AutoModel; \
 AutoModel.from_pretrained('bert-base-uncased')"
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt
```

The second stage then copies the model artifacts and virtual Python environment `/opt/ven` from the `base` stage before copying source files over and creating a production version of the FastAPI service:

```py
# Stage 2: Production
FROM base as production
RUN apt-get update && apt-get install -y
COPY --from=base /opt/venv /opt/venv
COPY --from=base /root/.cache/huggingface /root/.cache/huggingface

WORKDIR /code
COPY . .

EXPOSE 8000

ENV BUILD_ENV=PROD
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The last stage copies the production stage virtual Python environment with installed packages and adds several development tools on top. It then starts the server with hot reload functionality:

```py
# Stage 3: Development
FROM production as development

COPY --from=production /opt/venv /opt/venv
COPY ./requirements_dev.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements_dev.txt

ENV BUILD_ENV=DEV
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

Using a single Dockerfile, we were able to create three distinct stages and use them as we see fit via the `--target development` command when needed.

## docker init

You now have an in-depth understanding of the containerization process with the Docker platform and the relevant best practices.

If you ever need to add Docker to an existing project, you can use the `docker init` command, which will guide you through a wizard to create all the necessary Docker deployment files in your current working directory:

```py
$ docker init
>> Answer a few questions in the terminal...

project/
│
├── .dockerignore
├── compose.yaml
├── Dockerfile
└── README.Docker.md
... # other application files
```

This will provide a great starting point that you can work from to include additional configuration steps, dependencies, or services as required.

###### Tip

I recommend using `docker init` when starting out as every generated file will adhere to best practices including leveraging `dockerignore`, optimizing image layers, using bind and cache mounts for package installation, and switching to nonroot users.

Once you have an optimized image and a set of working containers, you can choose any cloud provider or self-hosting solution for pushing images to registries and deploying your new GenAI services.

# Summary

In this chapter, we reviewed various strategies for deploying your GenAI services—for instance, on virtual machines, as cloud functions, with managed app service platforms, or via containers. As part of this, I covered how virtualization differs from containerization and why you may want to deploy your services as containers.

Next, you learned about the Docker containerization platform and how you can use it to build self-contained images of your applications that can run as containers.

We covered the Docker storage and networking mechanisms that allow you to persist data using the union filesystem in containers and how to connect containers with different networking drivers.

Finally, you were introduced to various optimization techniques for reducing the build time and size of your images to deploy your GenAI services as efficiently as possible.

With services containerized, you can push them to container registries to share, distribute, and run them on any cloud or hosting environment of your choice.

^([1](ch12.html#id1291-marker)) Slim base Python images balance the size and compatibility of the Linux distribution with a wider range of Python packages out of the box compared to Alpine base Python images that minimize size but require extra configurations.

^([2](ch12.html#id1292-marker)) You can use the `-p` or `--publish` flag when running the container to map and enable container access via a port.

^([3](ch12.html#id1297-marker)) Images built on one machine can only run on other machines with the same processor architecture.

^([4](ch12.html#id1317-marker)) You can still run executable files with the `r` permission alone by using the `bash script.sh` command instead of `./script.sh`.

^([5](ch12.html#id1336-marker)) Refer to the NVIDIA documentation on how to install the latest CUDA toolkit and graphics drivers for your system.