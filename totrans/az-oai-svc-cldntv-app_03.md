# Chapter 2\. Designing Cloud Native Architectures for Generative AI

Cloud native architecture is a way of designing and building applications that can take advantage of the cloud’s unique capabilities and constraints. Cloud native applications are typically composed of microservices that run in containers, orchestrated by platforms like Kubernetes, and use DevOps and continuous integration and continuous deployment (CI/CD) practices to enable rapid delivery and scalability. Cloud native architectures are at the core of the generative AI era.

Organizations such as the [Cloud Native Computing Foundation (CNCF)](https://oreil.ly/dUsAO) are great catalysts of cloud native best practices and community development. Their goal is to be *“*the vendor-neutral hub of cloud native computing, to make cloud native universal and sustainable.” CNCF is a great source of information and learning material for these topics. Another great resource is the [twelve-factor app](https://oreil.ly/AFEgd), a public methodology for building cloud native applications.

As part of the cloud native movement, there are several projects and communities oriented to the use of cloud native architecture to enable scalable, reliable, and robust AI systems. They often require large amounts of data, complex algorithms, and specialized hardware to perform tasks such as image recognition, natural language processing, or recommendation systems. This is not always possible with traditional IT architecture patterns (e.g., [monolithic applications](https://oreil.ly/TrFNL)).

The need for cloud native architecture for AI systems arises from the following reasons:

System performance

AI systems need to process large volumes of data and run complex computations in a fast and efficient manner. Cloud native architecture enables AI systems to leverage the cloud’s elastic resources, such as compute, storage, and network, to scale up or down according to demand. It also allows AI systems to use specialized hardware, such as graphics processing units (GPUs) or tensor processing units (TPUs), that are optimized for AI workloads.

Agility

AI systems need to adapt to changing business requirements, user feedback, and data quality. Cloud native architecture enables AI systems to deploy new features, models, or updates quickly and reliably using DevOps and CI/CD practices. It also allows AI systems to experiment with different architectures, algorithms, or parameters using techniques such as A/B testing or canary deployments.

Innovation and integrability

AI systems need to leverage the latest advances in AI research and technology. Cloud native architecture enables AI systems to access the cloud’s rich ecosystem of AI services, tools, and frameworks that offer state-of-the-art functionality and performance. It also allows AI systems to integrate with other cloud services, such as data analytics, Internet of Things, or edge computing, that can enhance the value and intelligence of AI systems.

The most important areas for cloud native are described by CNCF as CI/CD, DevOps, microservices, and containers, as shown in [Figure 2-1](#fig_1__cloud_native_building_blocks_for_generative_ai_a).

![](assets/aoas_0201.png)

###### Figure 2-1\. Cloud native building blocks for generative AI (source: adapted from an image by CNCF)

These four areas are relevant to generative AI applications:

CI/CD

Enables a streamlined and automated process for integrating code changes, building, testing, and deploying AI models and applications, and facilitates faster iterations and reduces time to market for generative AI developments.

DevOps

Combines the principles and practices of DevOps for AI technologies to improve the development, deployment, and operations of AI systems, and facilitates the integration of generative AI into the overall software development lifecycle. It also ensures reliable monitoring, logging, and feedback loops, enabling quick identification and resolution of issues in generative AI systems.

Microservices

Allows complex generative AI systems to be broken down into smaller, independent services, which enables modular development and deployment of different components of the AI system. It also enhances scalability and flexibility, as individual microservices can be developed, deployed, and scaled independently.

Containers

Offers a lightweight and portable way to package and deploy generative AI models and applications, and enables easy scaling, replication, and orchestration of generative AI workloads.

Cloud native architecture is a key enabler for developing advanced, intelligent AI systems that can deliver high performance, agility, and innovation on the cloud platform. In this chapter, we will explore how to prepare a cloud native architecture for an AI-enabled system that leverages Azure OpenAI Service, regardless of the kind of application you are planning to develop. Let’s start by digging into some typical scenarios for AI cloud native development.

# Modernizing Applications for Generative AI

This book focuses on the development of new cloud native applications with Azure OpenAI Service and the rest of the Microsoft Azure stack. However, there may be scenarios in which a company tries to leverage these capabilities for their existing applications. Let’s compare both scenarios and see the approaches:

New cloud native applications

Designed from scratch using [containerization](https://oreil.ly/U0o9G) and a microservices architecture, enabling scalability, resilience, and elasticity. They leverage the four areas previously mentioned, and they make the deployment and maintenance of generative AI applications a bit simpler.

Existing apps

Likely require migration or modernization. This means they’ll either be migrated to the cloud, or modified to align with cloud native principles, such as breaking down a monolithic architecture into microservices or introducing containerization. The modernization process involves step-by-step upgrades, addressing scalability, resilience, and fault tolerance, and adopting DevOps practices gradually.

[*Learning Microsoft Azure* (O’Reilly)](https://oreil.ly/Rqw0P) by Jonah Carrio Andersson lays out some different strategies, and Microsoft’s [modernization guide](https://oreil.ly/5Sm8X) describes the process for migrating and modernizing existing on-prem/monolithic applications to the cloud, with specific cloud native features. [Figure 2-2](#fig_2_cloud_native_modernization_levels_moving_towards_g) illustrates the different levels of cloud modernization.

![](assets/aoas_0202.png)

###### Figure 2-2\. Cloud native modernization levels moving toward generative AI (source: adapted from an image by Microsoft)

Based on the modernization steps, there are different levels of maturity that range from existing on-premises applications to full cloud native ones. This is relevant for implementations with Azure OpenAI Service, as a native cloud-enabled PaaS, because new and existing applications will need some level of cloud readiness before integrating generative AI capabilities. Think of this as the way the rest of the application blocks connect with Azure OpenAI Service in a cloud-enabled way, with native and simple integrations.

The levels of maturity are as follows:

Cloud infrastructure–ready applications

With this migration strategy, you simply transfer or relocate your existing on-site applications to an infrastructure-as-a-service (IaaS) environment. While the structure of your applications remains largely unchanged, they are now hosted on virtual machines in the cloud. This straightforward migration method is commonly referred to as “lift and shift” within the sector, but it only gets a portion of the cloud value you can get from managed PaaS/SaaS services.

Cloud-optimized applications

At this stage, without making major code changes or redesigning, you can tap into the advantages of running your application in the cloud using contemporary technologies like containers and other cloud-managed services. This enhances your application’s flexibility, allowing for quicker releases by optimizing your business’s DevOps practices. This enhancement is made possible by tools like Windows containers, rooted in the Docker Engine. Containers address the challenges posed by application dependencies during multistage deployments. In this maturity framework, you have the option to deploy containers on either IaaS or PaaS, leveraging additional cloud-managed services such as database solutions, caching services, monitoring, and CI/CD workflows.

Cloud native applications

This migration approach typically is driven by business needs and targets modernizing your mission-critical applications. At this level, you use cloud services to move your apps to PaaS computing platforms. You implement cloud native applications and microservices architecture to evolve applications with long-term agility, and to scale to new limits. This type of modernization usually requires architecting specifically for the cloud, and even writing new code (or rewriting it), especially when you move to cloud native application and microservice-based models. This approach can help you gain benefits that are difficult to achieve in your monolithic and on-premises application environment.

The last level is the end goal for optimal generative AI–enabled applications, but any of these levels (especially the last two) would be “good enough” for any application to “connect” to Azure OpenAI Service. The rest of the chapter will focus on new cloud native applications, but if you plan to leverage Azure OpenAI Service for existing applications, please start by evaluating them and analyzing the next migration or modernization steps towards AI adoption.

Now, let’s focus on the key advantages of cloud native, and the key Azure-enabled building blocks that will allow you to build your Azure OpenAI solutions.

# Cloud Native Development with Azure OpenAI Service

Part of the idea behind cloud native architectures is to split code development into different pieces called microservices, so all modules communicate based on a functional flow, without being part of the same technical block. This has a series of advantages, not only for Azure OpenAI–enabled development, but any cloud native implementation. We can imagine several reasons to leverage a microservices architecture:

Modular and granular AI functionality

In AI applications, different tasks such as data preprocessing, feature extraction, model training, inference, and result visualization may be involved. By implementing each of these functionalities as separate microservices, the AI system becomes more modular and granular. This allows developers to focus on building and maintaining individual services, making it easier to understand, develop, test, and deploy specific AI components. This also allows reusability of components as there might be certain cleaning pipelines or even models that could be used for different applications within the same company. Last but not least, it supports team specialization depending on the task (e.g., model output processing tends to be an integration or data engineering task, while model implementation a data science one).

Scalability and performance optimization

AI workloads can vary in intensity, with some tasks requiring more computational resources than others. By breaking down an AI application into microservices, each service can be scaled independently based on its specific resource needs. This scalability ensures efficient resource utilization and improved performance. For example, model training and inference services can be scaled independently to handle varying workloads, providing better response times and overall system performance.

AI algorithm lifecycle management

AI applications often require experimenting with different algorithms, models, or data sources to achieve the desired outcome. With microservices, developers can easily swap out or update individual AI services without affecting the rest of the system. This flexibility enables rapid prototyping, experimentation, and iteration with different AI approaches, facilitating the discovery of the most effective algorithms or models for specific tasks. Also, certain systems might run algorithms in parallel to obtain a better result by selecting the best answers of those algorithms.

Integration with external services

Microservices architecture promotes loose coupling and well-defined APIs, making it easier to integrate AI services with external systems, tools, or services. This allows AI functionality to be leveraged across different applications, domains, or platforms. For instance, an AI service for NLP can be exposed via an API and utilized by multiple applications or integrated into a chatbot or customer support system.

Now, if we think about generative AI–enabled applications with Azure OpenAI Service, the goal is to structure the end-to-end architecture in a way that makes sense and connects the “AI pieces” to both backend elements (code, cloud resources), and frontend interfaces (one or several, depending on the application), as you can see in [Figure 2-3](#fig_3_microservice_enabled_ai_development).

![](assets/aoas_0203.png)

###### Figure 2-3\. Microservice-enabled AI development

All the involved elements need to be interoperable, replaceable, and available. For that purpose, organizing the building blocks in microservices is key. The next two sections look at the containerization and serverless approaches. Let’s discuss their role as cloud native enablers.

## Microservice-Based Apps and Containers

Cloud native development approaches leverage the power of the cloud, by choosing the right way to develop and to deploy applications. They rely on containerization, which often refers to Docker-type containers, and Kubernetes orchestration. As they are both based on international standards (e.g., the [Open Container Initiative [OCI]](https://oreil.ly/JKa4L)), cloud native applications are usually portable and scalable to different public and private cloud providers.

For Microsoft Azure, the key managed containerization services are [Azure Kubernetes Service (AKS)](https://oreil.ly/ymIkj) and [Azure Red Hat OpenShift (ARO)](https://oreil.ly/SXs9T). While both are managed Kubernetes services offered by Microsoft, there are some key differences:

[AKS](https://oreil.ly/YqfZ3)

A managed Kubernetes service provided by Microsoft Azure, utilizing native Kubernetes technology. It offers a fully managed Kubernetes cluster on Azure infrastructure and focuses on providing a streamlined and simplified Kubernetes experience on Azure. It provides essential Kubernetes features, including scaling, load balancing, and deployment management. AKS integrates well with other Azure services and provides native Azure resource management and monitoring capabilities. You can find pricing information [online](https://oreil.ly/OoChO).

[ARO](https://oreil.ly/mM0MD)

A joint offering between Microsoft and Red Hat, built on the [Red Hat OpenShift](https://oreil.ly/AftCs) Container Platform. ARO incorporates Kubernetes technology but provides additional features and integrations from the OpenShift platform. It provides a more comprehensive and enterprise-focused platform with additional security, compliance, and management capabilities.

In summary, they differ in terms of the underlying technology, vendor, and platform features. The choice between AKS and ARO depends on the specific requirements and preferences of your organization, such as the need for additional enterprise features and any existing investments or partnerships with Red Hat. Other related services you may want to explore are [Azure Container Apps](https://oreil.ly/QDzs2) and [Azure Arc for Kubernetes](https://oreil.ly/X5vd_) (for hybrid cloud scenarios).

Now that we have explored the containerization options in Azure, let’s understand the notion of serverless and its relevance for microservice-based implementations.

## Serverless Workflows

An alternative or complementary option is the serverless approach. Serverless computing is a cloud computing model that allows developers to build and run applications without the need to manage underlying infrastructure. It is particularly beneficial for AI workloads, including generative AI, as it provides a scalable and cost-effective solution.

In serverless architecture, developers focus on writing code for specific functions or tasks, known as serverless functions, with [Azure Functions](https://oreil.ly/Gm-h9) being the native Microsoft option. These functions are executed in containers that are managed and scaled automatically by the cloud provider, as you can see in [Figure 2-4](#fig_4_managed_cloud_as_a_service_levels). This eliminates the need for developers to provision and manage servers, making it easier to deploy and maintain AI applications.

![](assets/aoas_0204.png)

###### Figure 2-4\. Managed cloud–as-a-service levels

Much like other cloud native elements, one of the key advantages of serverless for AI workloads is scalability. Generative AI models often require significant computational resources, especially when training large models or generating complex outputs. Serverless platforms automatically scale resources on demand, allowing AI applications to handle fluctuations in workload without manual intervention. This scalability enables efficient resource utilization and cost optimization, as developers pay for only the actual compute resources used during execution.

Another advantage of serverless computing is its event-driven nature. Serverless functions are triggered by specific events, such as HTTP requests or messages from message queues. This event-driven architecture is well-suited for AI workloads that require real-time or asynchronous processing. For example, generative AI applications can be triggered by user interactions or scheduled tasks, allowing them to generate outputs on demand or periodically. Additionally, serverless can be used to perform actions within a generative AI pipeline. For that purpose, [Azure Logic Apps](https://oreil.ly/Qvt6X) can be used to trigger orchestration and workflows, and it has integration with other Microsoft 365 and Azure services, which can be useful in triggering generative AI pipelines or events.

There are some limitations related to serverless platforms, such as execution time limits, memory constraints, and deployment package size limits. However, techniques like function composition, caching, and parallel execution can help improve the efficiency and responsiveness of generative AI applications running on serverless architectures. Fine-tuning resource allocation and optimizing data processing pipelines can also contribute to better overall performance.

In general terms, you will be combining a PaaS such as Azure OpenAI, plus containerized and/or serverless pieces, depending on your implementation approach. We will now explore the web development part of your applications, to get an initial idea of the services that Azure OpenAI leverages to deploy generative AI–enabled web-based applications.

## Azure-Based Web Development and CI/CD

Now, let’s focus on development building blocks that go beyond core AI capabilities. As a cloud native practitioner, you will likely split your application code into several pieces. As you have already seen, those blocks are microservices that could contain backend and frontend modules (mobile applications, websites, intranets, etc.).

The interesting part comes when you discover you can host web-based applications directly via Azure App Service. Azure App Service is a PaaS, a fully managed service that allows adopters to build, deploy, and scale web applications and APIs without the need to manage underlying infrastructure. It supports various programming languages and frameworks and enables web, mobile, and API app development, as well as workflows (Logic Apps), CI/CD, and monitoring, while offering simple integration with the whole Microsoft Azure suite.

Overall, Azure App Service simplifies the process of building, deploying, and scaling web applications and APIs in the Azure cloud. It offers a robust and feature-rich platform that enables developers to focus on application development while benefiting from the scalability, availability, and management capabilities provided by the Azure platform.

You will see in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure) how Azure OpenAI offers simple deployment options that leverage Azure App Service to create chat-based applications with preexisting templates.

###### Note

If you want to dive deeper into any of these topics, please visit the following links:

*   Application hosting: [Azure App Service Overview | Microsoft Learn](https://oreil.ly/moBFz)

*   GitHub for CI/CD: [Deploy to App Service Using GitHub Actions | Microsoft Learn](https://oreil.ly/Z9EBe)

*   YouTube video: [How to Deploy Your Web App Using GitHub Actions | Azure Portal Series](https://oreil.ly/dSe0R)

We will now cover the fundamentals of the Azure portal, mostly for readers with no or low Azure experience, as a way to help you understand how to search, configure, and deploy Azure OpenAI and other related services. If you have already worked with Azure and its portal, you may skip this section.

# Understanding the Azure Portal

The Azure portal is a web-based UI provided by Microsoft Azure that allows users to manage and interact with their Azure resources. It serves as a central hub for accessing and managing various Azure services and functionalities, including Azure OpenAI Service. The portal provides a visually appealing and intuitive interface that simplifies the management and monitoring of Azure resources ([Figure 2-5](#fig_5_azure_portal_main_interface)).

![](assets/aoas_0205.png)

###### Figure 2-5\. Azure portal: main interface

As you can see in [Figure 2-5](#fig_5_azure_portal_main_interface), it includes a customizable dashboard that provides an overview of your Azure resources, recent activities, and personalized tiles for quick access to frequently used services.

The navigation pane on the left side of the portal allows you to access different categories of Azure services, including Compute, Storage, Networking, Security + Identity, AI + Machine Learning, and more. You can see the sequence in [Figure 2-6](#fig_6_azure_portal_left_panel).

![](assets/aoas_0206.png)

###### Figure 2-6\. Azure portal: left panel

Also, clicking on a specific category expands a menu with subcategories and services within that category. You can actually find Azure OpenAI Service within the AI + Machine Learning category ([Figure 2-7](#fig_7_azure_portal_resources_azure_openai_example)).

![](assets/aoas_0207.png)

###### Figure 2-7\. Azure portal: resources (Azure OpenAI Service example)

Alternatively, the Azure portal offers a search bar at the top, allowing you to quickly find services, resources, or documentation. As you can see in [Figure 2-8](#fig_8_azure_portal_search_azure_openai_example), you can search by keywords or use the natural language query to locate specific functionalities or resources within Azure. Basically, you can find Azure OpenAI by just typing it there.

![](assets/aoas_0208.png)

###### Figure 2-8\. Azure portal: search (Azure OpenAI Service example)

Each Azure service has its own dedicated blade, which is essentially a panel that provides detailed information and management options for that service. If you choose Azure OpenAI from either the search engine or the left panel, you will enter your resource details ([Figure 2-9](#fig_9_azure_portal_resource_details_azure_openai_examp)). Basically, you are able to create new resources for Azure OpenAI, or manage those previously deployed. If you choose Create, you can see the required information to deploy a new Azure OpenAI Service.

![](assets/aoas_0209.png)

###### Figure 2-9\. Azure portal: resource details (Azure OpenAI Service example)

You can find details related to your subscription, geographic region preferences, the unique name chosen for your Azure resource, and the pricing tier. (Tiers are the level of pricing based on estimated usage; for now there is only one option for Azure OpenAI, called “Standard S0.” Any update should be available via the [official pricing page](https://oreil.ly/7Gmq6), and the [Azure calculator](https://oreil.ly/2SQ4C).)

In addition to managing individual resources, the Azure portal allows you to create [resource groups](https://oreil.ly/J2LMM) to logically organize and manage related resources together. This is an interesting feature, and a recommended best practice to group the required resources for your generative AI implementations with Azure, including Azure OpenAI Service and others we will need for our projects.

###### Note

If you haven’t created an Azure account before, the first step is to [create a free one](https://oreil.ly/WVIm2). It usually includes credits with a value of USD $200 for initial experimentation. It requires a corporate email for the specific account and payment information.

We will explore the details of generative AI implementation approaches with Microsoft Azure in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure), but the idea behind the Azure portal is to facilitate the deployment, management, and maintenance process of the different resources required to create these architectures, regardless of the type of service. Deploying any Azure services from the Azure portal involves several steps, so remember the high-level process:

1\. Sign in to the Azure portal.

Open a web browser, navigate to the Azure portal, and sign up with your Azure account credentials.

2\. Create a resource.

To deploy an Azure service, you need to create a resource. A resource represents a service or component in Azure, such as a virtual machine, a storage account, or a database. Click on the “Create a resource” button in the Azure portal.

3\. Select a service.

In the resource creation wizard, you’ll see a list of available Azure services. Choose the service you want to deploy by browsing through the categories or using the search bar.

4\. Configure the resource.

Once you’ve selected a service, you’ll be taken to a configuration page where you can specify the settings for the resource. The options available depend on the specific service you’re deploying. Fill in the required information, such as resource name, region, pricing tier, and any other relevant settings.

5\. Review and create.

After configuring the resource, review the settings to ensure they are correct. You can also enable additional features or add-ons if available. Once you’re satisfied, click the “Review + Create” button.

6\. Validation and deployment.

Azure will validate the configuration settings and check for any potential issues. If everything is in order, click the “Create” button to initiate the deployment process.

7\. Monitor the deployment.

Azure will start provisioning the resources based on your configuration. You can monitor the deployment progress in the Azure portal. Depending on the service, the deployment may take a few minutes to complete.

8\. Access and manage the deployed service.

Once the deployment is finished, you can access and manage the deployed service through the Azure portal. You can view its properties, make changes to its configuration, monitor its performance, and perform other administrative tasks as needed.

This is the process for most of the Azure resources, but there are other deployment methods such as [Azure Resource Manager templates](https://oreil.ly/TZXTy), [API-enabled resource orchestration](https://oreil.ly/jezRs), [Azure Bicep](https://oreil.ly/aZOxZ), [Terraform on Azure](https://oreil.ly/Wi9xy), or command-line tools such as [Azure CLI](https://oreil.ly/Mm4N1) or [Azure PowerShell](https://oreil.ly/22BEd), all of them for more advanced admin/technical users. Feel free to explore them if you want to learn more.

For Azure OpenAI Service, you can always visit the [official resource deployment guide](https://oreil.ly/hSPh3), which summarizes the steps we just walked through. Other information you may want to review before deploying the service includes the [main product page](https://oreil.ly/MDBhf), the previously mentioned [pricing guide](https://oreil.ly/7Gmq6), the service [availability by geographic region](https://oreil.ly/tYnCe) (for example, if you deploy the service from the European Union, you may want to use a closer region, such as West Europe in Amsterdam, for better latency, performance, and maybe pricing), and the [general documentation](https://oreil.ly/3oNQU).

Now that you know how to use the Azure portal, and the key information about the Azure OpenAI Service deployment process, let’s analyze some important considerations at the model and general architecture levels. This will be key to creating the end-to-end implementations we will see in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure).

# General Azure OpenAI Service Considerations

Now that we have explored the notion of cloud native development with Azure, and the fundamentals of the Azure portal for Azure OpenAI Service, let’s go deeper into the different AI models that are available and the high-level architectures so you can know how to make sense of the Azure-enabled generative AI offerings.

## Available Azure OpenAI Service Models

Most cloud-enabled PaaS resources from any public cloud, including those from Microsoft Azure, leverage native endpoints and APIs as a way to connect and to consume their models. This is the case for Azure OpenAI Service and the rest of the Azure AI Services we have seen in this chapter.

Also, there are visual elements such as [Azure AI Studio](https://oreil.ly/PCMD3) and [Azure ML Studio](https://oreil.ly/kdZhY) (not to be confused with [Azure OpenAI Studio](https://oreil.ly/LWQO1), which we will explain and leverage in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure)) that provide access to different proprietary and open source AI/foundation models. This includes a model catalog to leverage the curated selection of models, including those from Azure OpenAI, Meta, and Hugging Face (e.g., the [Hugging Face Hub in Azure](https://oreil.ly/96mAx), announced by both Microsoft and Hugging Face during Microsoft Build 2023). This also allows us to test and deploy those models in a very simple way.

As you can see in [Figure 2-10](#fig_10_azure_ai_studio_main_interface), if you visit the [Studio page](https://oreil.ly/kdZhY), you will get access to your existing workspaces, or you will be able to create a new one if it is your first time connecting to the studio.

![](assets/aoas_0210.png)

###### Figure 2-10\. Azure AI Studio: main interface

If you access the workspace, you will see the same kind of visual interface we reviewed earlier in this chapter. In [Figure 2-11](#fig_11_azure_ai_studio_left_panel), the left panel for the workspace menu offers all options related to data, models, endpoints, required resources, etc. For the sake of simplicity, we will focus on two main features: the [model catalog](https://oreil.ly/BYkuc), and later in [Chapter 4](ch04.html#additional_cloud_and_ai_capabilities), the prompt flow functionality.

![](assets/aoas_0211.png)

###### Figure 2-11\. Azure AI Studio: left panel

If you choose the model catalog option and search “Azure OpenAI” or click directly on the tile as shown in [Figure 2-12](#fig_12_azure_ai_studio_model_catalog), you will get access to the updated list of available Azure OpenAI models.

![](assets/aoas_0212.png)

###### Figure 2-12\. Azure AI Studio: model catalog

The models in [Figure 2-13](#fig_13_azure_ai_studio_azure_openai_models) are those available at the time of writing, but depending on when you check the catalog, you will likely find these and/or others. An alternative way to check all the available models at the moment is to use the [List API](https://oreil.ly/bk7Zd).

![](assets/aoas_0213.png)

###### Figure 2-13\. Azure AI Studio: Azure OpenAI Service models

Now, keeping in mind the evolving nature of the availability of Azure OpenAI models, explore the key model families and some examples of specific models that you will leverage for your generative AI projects. This will certainly change over time, but it is a good beginning.

Azure OpenAI Service splits its capabilities into different *model families*. A model family typically associates AI models by their intended task, such as natural language understanding, code generation, or image synthesis. Some of the most popular Azure OpenAI model families are as follows:

Language-related models

Popular language-related models include the following:

GPT-3.5 Turbo and GPT-3.5 Turbo Instruct

Models that improve on previous GPT-3 versions and can understand and generate natural language and code. There are several versions with different context length limits, including those for 4K and 16K tokens, which is the measure of the maximum text input.

GPT-4, GPT-4 Turbo, GPT-4o

Models with better performance (and higher cost) than 3.5 Turbo, which can handle more complex tasks and generate more accurate and diverse outputs. They can also handle bigger text inputs (which we usually define as “context”) than their predecessors.

Speech

There are other options in Azure, but Azure AI Studio includes the speech-to-text [Whisper model](https://oreil.ly/9si-P) from OpenAI (i.e., by typing “whisper” and selecting the model). It is not directly available from Azure OpenAI Studio, but it can be integrated with the rest of GPT models to create voice-to-text scenarios.

Other models

Other popular models include the following:

Codex for programming code

A series of models that can understand and generate code, including translating natural language to code. The reality is that Codex was initially a separate model, but after some time OpenAI added its capabilities to the regular GPT-3.5 Turbo and GPT-4 language models. This means the same models handle both natural language and programming code.

DALL·E for images

A series of models that can generate original images from natural language. This is the model behind tools like [Bing Create](https://oreil.ly/YwDy-) and [Microsoft Designer](https://oreil.ly/oIRon), and it is directly available from Azure OpenAI Studio, as we will see in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure).

It is important to differentiate the different model families and their specific capabilities to understand which ones we will use for our generative AI projects. Also, the trade-off of different Azure OpenAI models depends on the use case and the available budget. Generally speaking, more capable models like GPT-4o can handle more complex tasks and generate more accurate and diverse outputs, but they also consume more resources and incur higher costs. We will explore several scenarios in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure) that can work with all these GPT models. You can also explore the whole set of OpenAI models, including some deprecated ones that are still [traced via OpenAI’s documentation](https://oreil.ly/SG-fe).

Besides all these functionalities, one of the key features for LLM-enabled systems is *embeddings*. This is a general term related to NLP and LLMs. Embeddings are a way of representing data in a multidimensional space. They are often used to capture the semantic meaning of words, images, or other types of data. For example, in [Figure 2-14](#fig_14_embedding_model), an embedding model can map a word to a vector of numbers, such that words with similar meanings have similar vectors. This means we can connect pieces of information that are not directly connected, but that may have a mathematical or linguistic connection (e.g., several knowledge bases from companies of the same sector, internal and external sources, etc.).

![](assets/aoas_0214.png)

###### Figure 2-14\. Embedding model

This example illustrates the typical *generation and search process*:

1.  We collect different data inputs (PDFs, text files, URLs, etc.) to create our knowledge base. This is a simplified view as sources are previously processed to extract the text-based information. We will see options for this such as official accelerators and Azure AI Document Intelligence in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure).

2.  We leverage the [Embeddings API](https://oreil.ly/bhgTY) to generate the embeddings from diverse sources. We can use a basic API call with the text input that returns the generated vectors.

3.  The generated vectors/embeddings are stored in a vector database. We will explore several database options in Azure in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure).

4.  After the generation process, we can assume end users will want to search for specific topics or information that will be included as part of the different data inputs we have collected and vectorized. For that purpose, we will use the same embeddings API to generate the embeddings of the questions itself (note: we need the same embedding model for both knowledge and questions).

5.  The vector database will support search functions. This means we will use the vectorized user questions as input to find information from the vector database that contains our knowledge base.

6.  If there are related topics, the search function will return a Top-k variety of results that we can use to generate the answer (either by directly printing the results or by passing them as input for a chat-based scenario).

The embeddings use cases available in Azure OpenAI Service are as follows:

Text similarity

A set of models that provide embeddings that capture the semantic similarity of pieces of text. These models are useful for many tasks such as clustering, regression, anomaly detection, and visualization.

Text search

A set of models that provide embeddings that enable semantic information retrieval over documents. These models are useful for tasks such as search, context relevance, and information retrieval.

Code search

A set of models that provide embeddings that enable finding relevant code with a query in natural language. These models are useful for tasks such as code search and relevance.

At a technical level, the recommended model option for embeddings with Azure OpenAI Service is called “Ada”; this is an [improved and more cost-effective](https://oreil.ly/6m7SL) model than its predecessors. This is pretty useful to increase the knowledge scope of Azure OpenAI, by consuming information from PDFs, websites, text files, etc.

As previously mentioned, embeddings generation is based on a very simple API call/response dynamic, and the specific details on how to generate embeddings for a given source are available in [the official documentation](https://oreil.ly/2cxWx), as well as the specific [context length limits](https://oreil.ly/SQSGw) (e.g., 8K tokens for Ada version 2). Generating embeddings is as simple as calling the embedding API with the desired text input you want to vectorize. For example, in Python:

```py
import openai
openai.api_type = "azure"
openai.api_key = YOUR_API_KEY
openai.api_base = "https://YOUR_RESOURCE_NAME.openai.azure.com"
openai.api_version = "2023-05-15"

response = openai.Embedding.create(
    input="Your text string goes here",
    engine="YOUR_DEPLOYMENT_NAME"
)
embeddings = response['data'][0]['embedding']
print(embeddings)
```

The output of this would be a numerical representation, where each number in the list corresponds to a dimension in the embedding space. The exact values will depend on the specific model and its training data, but it could look like this:

```py
[0.123, 0.456, 0.789, ..., 0.987]
```

We have completed the review of Azure OpenAI models and their capabilities. While we will cover the details of project examples and architectures in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure), the next section will explore general architectural building blocks for Azure OpenAI–enabled implementations, as well as general cloud infrastructure topics.

## Architectural Elements of Generative AI Systems

Azure-based architectures rely on a series of interconnected services that can communicate with each other for a specific purpose. In this case, Azure OpenAI plays a crucial role to enable interactions between any customer-side application, but we rely on more building blocks to build our generative AI solutions. In [Figure 2-15](#fig_15_high_level_architecture_building_blocks), you can see the main building blocks of an Azure OpenAI–enabled (simplified) architecture.

![](assets/aoas_0215.png)

###### Figure 2-15\. High-level architecture building blocks

Let’s take a look at these pieces in a little more detail:

Application frontend

Any app-side element that leverages generative AI capabilities.

Middleware/orchestration

We will explore this element in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure), but the orchestration piece basically allows us to connect different Azure OpenAI skills with other relevant services. Also, the middleware can include API management and other topics that we will see in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure).

Azure OpenAI Service

For text-based skills, such as explaining the answer to a complex question, for both completion and chat-based scenarios.

Additional knowledge base

This is a combination of the core data sources (databases, blob storage, etc.) and knowledge extraction elements such as embeddings, Azure Cognitive Search, Bing Search, etc. For now, we will define them as “grounding blocks,” but we will see the details in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure).

If you develop an application that leverages Azure OpenAI and other Azure services, and that implementation is part of a bigger data/AI-enabled platform, the end-to-end architecture might start to look something like [Figure 2-16](#fig_16_end_to_end_azure_platform_including_azure_openai).

![](assets/aoas_0216.png)

###### Figure 2-16\. End-to-end Azure platform (including Azure OpenAI Service)

In this case, Azure OpenAI Service is just part of a bigger end-to-end that includes data sources, integration processes, SQL/NoSQL databases, containerization, analytics, etc. The final setup depends on the structure of the platform itself, but this is a good overview to understand where Azure OpenAI sits for any data and AI implementation with Microsoft Azure.

If you want to learn more about Azure-enabled architectures and the details of all these cloud services, please check out [*Learning Microsoft Azure*](https://oreil.ly/G2U08) by Jonah Carrio Andersson. Also, the main reference for architecture is the official [Microsoft Architecture Center](https://oreil.ly/0jzik) for specific [Azure OpenAI scenarios](https://oreil.ly/y-gPD). You may want to bookmark this resource as the Microsoft teams continuously update the content with new visual architectures and explanations, including some examples with Azure OpenAI Service.

Another interesting architecture you can explore is the [Azure OpenAI Landing Zone reference architecture](https://oreil.ly/xLs8X), which includes end-to-end cloud considerations, including core infrastructure topics such as identity and security, monitorization, cost management, user and API management, FinOps, etc. This is a very rich and complete overview of what an enterprise-grade implementation would include, beyond the core generative AI capabilities.

Last but not least, don’t forget to explore the [CNCF Cloud Native AI Whitepaper](https://oreil.ly/qd4lq) from the [AI Working Group](https://oreil.ly/8k0bc), which includes technology building blocks, techniques, and cloud native resources for generative AI topics.

# Conclusion

As you can see, cloud native architectures are valuable for generative AI development, as they seamlessly integrate with Azure OpenAI and other Azure services. We will explore different implementation approaches in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure), but all of them rely on the capabilities and key building blocks discussed here. As an adopter, you may face situations where you will need to optimize existing applications so they can incorporate generative AI capabilities (as we reviewed in the modernization section), but you will also have the opportunity to develop new Azure OpenAI–enabled applications from scratch. In this case, leveraging containerization, serverless, and PaaS pieces will help you design well-architected and scalable architectures and solutions. Depending on your current level of knowledge, it will be important for you to understand the cloud fundamentals behind Microsoft Azure and specific services for development, APIs, and Kubernetes container orchestration.

[Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure) will focus on different alternatives for enhancing your Azure OpenAI applications with specific company knowledge, as well as the main features and interfaces that you will leverage for your next projects. It also includes new terms that we briefly explored in this, such as vector database and orchestration. Let’s continue.