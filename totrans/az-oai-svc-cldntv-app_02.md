# Chapter 1\. Introduction to Generative AI and Azure OpenAI Service

This first chapter covers the fundamentals of artificial intelligence (AI) as a way to contextualize the new developments with generative AI. It includes some technology-agnostic topics that will be useful for any kind of implementation, but it focuses on Azure OpenAI Service as the key building block to enable cloud-native application development with generative AI.

# What Is Artificial Intelligence?

This section focuses on the historical evolution of AI technologies and related use cases as a way to demystify what AI actually looks like, and to connect traditional approaches with new generative AI techniques and capabilities.

Let’s start with its origins. The term “AI” was coined during the 1950s. Concretely, Professor John McCarthy defined artificial intelligence in 1955 as [“the science and engineering of making intelligent machines”](https://oreil.ly/hir0l). It is also fair to say that Professor Alan Turing previously introduced the notion of thinking machines. In 1956, Dartmouth College hosted the Summer Research Project on AI conference, with a group of participants from the most relevant universities and companies. That conference was led by Prof. McCarthy and other renowned researchers, and it was the beginning of the AI area of research. In the time since, there have been multiple cycles of hype, disappointments due to unrealistic expectations (periods often referred as AI winters due to reduced funding and general interest for AI topics), renovated expectations, and finally vast commercialization of AI-enabled solutions such as personal assistant speakers, intelligent autonomous vacuum cleaners, etc.

That said, AI has evolved a lot during the last two decades, but the reality is that initially it was mainly and only adopted by some of the biggest companies, such as Microsoft (no, not necessarily for their famous [Clippy](https://oreil.ly/0iNuM)!), Google, Amazon, Uber, and other technology unicorns. That first wave of adoption created a great baseline so they could offer these same capabilities as managed cloud services to other AI adopters out there, which gave them a clear competitive advantage. This started the stage of data and AI democratization we are currently experiencing, where smaller companies are developing or leveraging AI-enabled services, and those solutions are already part of our day-to-day.

Before going into the details, let’s take a step back and analyze the context of what artificial intelligence is today, and what it means for companies and individuals.

## Current Level of AI Adoption

The term “AI adoption” describes how organizations around the world are either implementing AI systems or leveraging AI-enabled tools from other companies. Each company’s level of adoption really depends on several factors, such as technology maturity, type of organization (big or small companies, public administration, startups, etc.), geography, etc. McKinsey indicates that the level of AI adoption in 2022 (from their [state of AI report](https://oreil.ly/6GDdZ)) was 50% from all its respondents, with an interesting increase at the international level, and an even more significant increase for developing countries. Additionally, they also estimate that [generative AI could add to the global economy](https://oreil.ly/4L3JC) the equivalent of $2.6 trillion to $4.4 trillion annually.

In addition, Boston Consulting Group [defined](https://oreil.ly/mrLCw) the level of success and AI maturity as a combination of internal adoption plus the knowledge of AI within the organization, with only 20% of organizations being actual pioneers in terms of AI adoption. Last but not least, Gartner [predicts](https://oreil.ly/b-x4K) that by 2025, 70% of enterprises will identify the sustainable and ethical use of AI among their top concerns, and 35% of large organizations will have a Chief AI Officer who reports to the CEO or COO.

These figures show that even if the level of global adoption is increasing, there are still differences in how companies are using AI and how successful they are. The next sections will showcase multiple examples of AI-enabled systems, at both the technology and use case levels.

## The Many Technologies of AI

There are different ways to define artificial intelligence, but the reality is that there is not only one single technology under the umbrella of AI. Let’s explore the main AI technologies:

Machine learning (ML)

A type of AI that relies on advanced statistical models that learn from past data to predict future situations. Take a simple use case of classifying fruits based on their existing pictures. To describe an apple to the system, we would say it is somewhat round in shape and that its color is a varied shade of red, green, or yellow. As for oranges, the explanation is similar except for the color. The algorithm then takes these attributes (based on past examples) as guidelines for it to understand what each of the fruits looks like. Upon being exposed to more and more samples, it develops a better capacity to differentiate oranges from apples and gets better at correctly identifying them. There are plenty of ML models depending on the kind of algorithm and type of task, but some relevant examples are decision forests, k-means clustering, regressions, and support vector machines (note: if you want to explore this family of AI models, take a look at the [Microsoft ML Algorithm Cheat Sheet](https://oreil.ly/T7JG9), which explains the type of tasks for different models and their data requirements).

Deep learning (DL)

Deep learning can be defined as a subset of machine learning, with models that rely on algebra and calculus principles. The differentiating character of deep learning is that the algorithm uses a neural network to extract features of the input data and classify it based on patterns to provide output without needing manual input of definitions. The key aspect here is neural networks. The idea of neural networks comes from the fact that they mimic the way the brain functions, as a multilayer system that performs mathematical calculations. Layered with multiple levels of algorithms designed to detect patterns, neural networks interpret data by reviewing and labeling its output. If we consider our fruit example, instead of having to provide the attributes of what each fruit looks like, we have to feed many images of the fruits into the deep learning model. The images will be processed, and the model will create definitions such as the shapes, sizes, and colors.

Natural language processing (NLP)

NLP combines computational linguistics (rule-based modeling of human language) with statistical, machine learning, and deep learning models. These kinds of models were initially available only in English (e.g., BERT from Google AI), but the current trend is to create local versions or multilanguage models to support others like Spanish, Chinese, French, etc. That said, NLP has undergone a tremendous evolution in the last 20 years. NLP algorithms used to be task-specific, but modern architectures have allowed them to better generalize to different tasks and even to gain emerging capabilities that they were not trained for. From a Microsoft Azure perspective, both Azure OpenAI Service and [Azure AI Language](https://oreil.ly/b191_) resources rely on NLP models.

Robotic process automation (RPA)

This is a set of technologies that replicates the manual interactions of human agents with visual interfaces. For example, imagine you are working in HR and you need to do the same task every week, which could be checking some information related to the employees via an internal platform, then filling out some information, and finally sending a customized email. RPA tools are easy to implement, reduce wasted time, and increase internal efficiencies, so employees can focus on added value tasks and avoid monotonous work.

Operations research (OR)

Operational research is a very important area, often included as part of the family of AI technologies, and very related to ML and the previously mentioned reinforced approaches. The University of Montreal [defines operations research](https://oreil.ly/u_H5h) as “a field at the crossroads of computer science, applied mathematics, management, and industrial engineering. Its goal is to provide automated logic-based decision-making systems, generally for control or optimization tasks such as improving efficiency or reducing costs in industry.”

OR usually relies on a set of variables and constraints that guide some sort of simulation that can be used for different kinds of planning activities: managing limited healthcare in hospitals, optimizing service schedules, planning energy use, planning public transit systems, etc.

These are the main categories of AI technologies, but the list can change depending on the interpretation of what AI means. Regardless of the details, it is important to keep in mind these technologies as a set of capabilities to predict, interpret, optimize, etc. based on specific data inputs. Let’s see now how these different AI technologies apply to all sorts of use cases, which are likely to leverage one technology or combine them depending on the implementation approach.

## Typical AI Use Cases

Regardless of the level of technical complexity, there are many different kinds of AI implementations, and their usefulness usually depends on the specific use cases that organizations decide to implement. For example, one organization might say, “we would like to get automatic notifications when there is a specific pattern from our billing figures” and develop some basic anomaly detection model, or even a basic rule-based one, and this could be considered an AI. Others will require more advanced developments (including generative AI), but they will need to have a business justification behind it.

Before we explore the technical and business considerations for an adopter company, here are some examples of AI-enabled applications:

Chatbots

You’re likely very familiar with chatbots—those little friends that are embedded into websites—as well as automated phone bots that allow companies to automate their communication and customer support. They are based on NLP/linguistic capabilities that allow them (with different levels of success) to understand the intent of what a client wants or needs, so they can provide them with an initial answer or hints to find the final answer. They also reduce the burden on support folks to answer initial requests, as chatbots can analyze, filter, and dispatch cases depending on the topic. The main advantage is automation and scalability of business activities (i.e., doing more with less), but there are challenges related to how efficient chatbots are for complex tasks and information. That said, chatbots are exponentially evolving with the arrival of generative AI, going from traditional rule-based engines to dynamic assistants that can adapt to the context of the discussion.

Computer vision systems

Image detection and classification applications that rely on DL technologies to analyze images and videos. For example, personal devices such as laptops and smartphones rely on this kind of technology to unlock them with an image of your face. Computer vision also supports advanced video analytics for a variety of applications.

Fraud detection

Widely used by financial institutions, AI can help detect unusual patterns that may indicate some sort of misuse of financial assets, such as credit cards. This could be a card translation from a remote country, unusual purchases, repetitive attempts to get money from an ATM, etc. These AI-enabled systems rely on different kinds of technologies (NLP, behavioral analysis, etc.) and make the surveillance more scalable, allowing humans to focus only on critical cases.

Voice-enabled personal assistants

Integrated via smartphones, speakers, cars (check out the amazing [case of Mercedes with Azure OpenAI](https://oreil.ly/yR55l)), TVs, and other kinds of devices, these personal assistants enable interaction with human users by simulating conversation capabilities. It is widely used to reduce the accessibility barrier (i.e., it uses voice and does not require visual, writing, and reading capabilities) and allows users to free their hands while activating features such as apps, music players, video games, etc. There are also privacy concerns related to these systems, as they can act purely in a reactive manner, or “listen” continuously to human discussions.

Marketing personalization

The actual rainmaker for big companies such as Google and Meta. The ability to first understand the features related to a user (their age, location, preferences, etc.) and to connect that with the business goals of companies advertising their products and services is the key feature of modern online business. Marketing departments also use AI to segment their customer base and adapt their marketing techniques to these different segments.

In-product recommendations

Companies such as Netflix and Amazon have in-product recommendations based on their understanding of user needs. If someone looks for sports equipment, Amazon can recommend related products. It is the same for TV shows and movies on Netflix and other streaming platforms—they’re able to make recommendations based on what you’ve watched previously. Everything is based on customer data and it relies on relatively complex AI models that we will explore later.

Robots

Examples include the Roomba vacuum cleaner, the incredible creations from [Boston Dynamics](https://oreil.ly/eVmm5) that can even dance and perform complex tasks, the humanoid [Sophia](https://oreil.ly/RjtE9), etc.

Autonomous vehicles

This type of system is equipped with different sets of advanced technologies, but some of them leverage AI techniques that allow cars to understand the physical context and adapt to dynamic situations. For example, these vehicles can autonomously drive with no need for a human driver, and they can make decisions based on different visual signals from the road and other cars. [Tesla’s Autopilot](https://oreil.ly/-3AK_) is a great example of this.

Security systems

This includes both cyber and physical security. As with fraud detection, AI helps security systems spot specific patterns from data and metrics, in order to avoid undesired access to precious resources. For example, [Microsoft Copilot for Security](https://oreil.ly/ISrdc) detects hidden patterns, hardens defenses, and responds to incidents faster with generative AI. Another example would be AI-enabled cameras that can spot specific situations or objects from the video images.

Online search

Systems such as Microsoft Bing, Google Search, Yahoo, etc. leverage massive amounts of data and customized AI models to find the best answers to specific user queries. This is not a new concept, but we have seen how this kind of system has evolved a lot during recent years with the new [Microsoft Copilot](https://oreil.ly/NofXj) and [Google Gemini](https://oreil.ly/Zv5lb) apps. Additionally, we will see some examples for generative AI and web search applications in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure).

Predictive maintenance

A very relevant case for industrial applications, this leverages different types of data to anticipate situations where machinery and industrial equipment may need maintenance before having specific issues. This is a perfect example of understanding past data to generate predictions, and it helps businesses avoid potential problems and approach maintenance activities in a proactive way.

Obviously, these applications can be transversal or specific to different industries (e.g., agriculture, healthcare), but they rely on the same technology pieces. Now that you understand them and their typical applications, let’s focus on how AI models can learn, as this will be relevant for the general generative AI topic of this book.

## Types of AI Learning Approaches

As humans, we start to learn when we are babies, but the way we do it will depend on the process we follow. We can learn by ourselves, based on our own positive or negative experiences. We can also learn from the advice of adult humans, who previously learned from their own experience; this can help us accelerate our own learning process. AI models are very similar, and the way to leverage previous experiences (in this case data and models) depends on the type of AI model learning approach, as you can see in [Figure 1-1](#fig_1_ai_model_learning_categories).

![](assets/aoas_0101.png)

###### Figure 1-1\. AI model learning categories

Let’s walk through each approach in the figure:

Unsupervised learning

This is based on unsupervised techniques that don’t require human data annotation or support for the AI models to learn. This type usually relies on mathematical operations that automatically calculate values between data entries. It doesn’t require any sort of annotation, but it is suitable for only specific types of AI models, including those used for customer segmentation in marketing. The king of unsupervised techniques is what we call “clustering,” which automatically groups data based on specific patterns and model parameters.

Supervised learning

Supervised learning is a very important type of learning for AI implementations. In this case, AI models use not only the input data, but also knowledge from human experts (subject matter experts, or SMEs) who can help AI understand specific situations by labeling input data (e.g., What’s a picture of a dog? What’s a negative pattern?). It usually requires some sort of data annotation, which means adding additional information (e.g., an extra column for a table-based dataset, a tag for a set of pictures). In general, this is a manual process and getting it right will impact the quality of the AI implementation, as this is as important as the quality of the dataset itself.

Reinforced learning

Last but not least, we have reinforced learning (RL) methods. Without getting too into the weeds with technical details, the main principle is the ability to simulate scenarios, and to provide the system with positive or negative rewards based on the attained outcome. This kind of learning pattern is especially important for generative AI, because of the application of *reinforcement learning from human feedback* (RLHF) to Azure OpenAI and other models. Concretely, [RLHF](https://oreil.ly/RCdKb) gets retrained based on rewards from human feedback (i.e., reviewers with specific topic knowledge). We will explore the details in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure), because RLHF is highly relevant to the creation of Azure OpenAI models.

There are different ways in which models learn, depending on the internal architecture, the type of data sources, and expected outcomes. For the purposes of this book, it is important to differentiate and understand the high-level differences, as we will be referring to some of them in the context of generative AI.

Generative AI is here to stay, and Azure OpenAI Service is already a key factor for adoption and democratization. Let’s now explore the fundamentals of generative AI, to understand how it works and what it can do for you and your organization.

# About Generative AI

The term “generative AI” refers to the field of artificial intelligence that focuses on creating models and systems that have the ability to generate new content, such as images, text, music, videos, diagrams, etc.

As you may already know, this term has gained a lot of relevance in recent years, but it is not new. We can talk about probabilistic models in the 1990s, such as latent variable models and graphical models, which aimed to capture and generate data distribution. Also, recent advancements in deep learning, specifically in the form of generative adversarial networks (GANs) and variational autoencoders (VAEs), have significantly contributed to the popularization and advancement of generative AI.

The term “generative AI” has gained momentum as researchers, companies, and practitioners begin to explore the potential of these techniques to generate realistic and creative outputs. The result is now obvious, as AI encompasses a wide range of applications and techniques, including image synthesis, text generation, music generation, etc. Obviously, this is an evolving field and both academia and industry continue to innovate.

As you can see in [Figure 1-2](#fig_2_types_of_ai_capabilities), the generation capability can be seen as an extension of other existing types of AI techniques, which are more oriented to either describe, predict, or prescribe data patterns, or to optimize specific scenarios. Advanced AI techniques, including OR and generative AI, allow adopters to go from “only insights” to automated decision making and actions.

![](assets/aoas_0102.png)

###### Figure 1-2\. Types of AI capabilities

From a technical point of view, these models work in a very particular way. Instead of “just” predicting a certain pattern for a data entry (e.g., forecasting the ideal insurance premium for a specific client), they generate several outcomes to a specific instruction. The interaction with the generative AI model happens in a question-answer fashion, and this includes both direct instructions from humans (based on natural language instructions) and automated actions.

The term “prompt engineering” has emerged more recently in the context of NLP and the development of language models. While there isn’t a specific origin or a definitive moment when the term was coined, it has gained popularity as a way to describe the process of designing and refining prompts to elicit desired responses from language models.

Prompt engineering involves carefully crafting the instructions or input provided to a language model to achieve the desired output. It includes selecting the right wording, structure, and context to guide the model toward generating the desired response or completing a specific task. There are ongoing efforts to develop systematic approaches for designing effective prompts, fine-tuning models for specific tasks, and mitigating biases or undesired behaviors in language generation.

From the previously mentioned question-answer dynamic, as you can see in [Figure 1-3](#fig_3_prompts_and_completions), *prompt* is the question, and the answer is called *completion*. The term “completion” in the context of NLP and language models refers to the generation or prediction of text that completes a given prompt or input, and it became more widely used as larger and more powerful models like OpenAI’s GPT were developed. In summary, the term “completion” in language models emerged from the evolving field of language modeling, reflecting the ability of models to generate or predict text that fills in or completes a given context or prompt.

![](assets/aoas_0103.png)

###### Figure 1-3\. Prompts and completions

Generative AI is a new kind of artificial intelligence, and its main advantage for wide adoption is the ability to enable communication between users and the generative AI models via natural language prompts and completions. That’s a game changer, but let’s see now the main kind of capabilities we can get from these models.

## Primary Capabilities of Generative AI

It is true that language and text-based information are a key aspect of generative AI. However, language-based prompts can serve other purposes. Companies and researchers are working on several streams:

Language

Besides the core ChatGPT-type functionality, with questions and answers between the AI model and the human user, there are other related tasks that rely on linguistics but go a step further. What if you can use language as the creation catalyst for:

Code

Technically, a programming language is just that…a language. LLMs are good at handling English or Spanish, but they are also great at understanding and generating code, and handling Java, Python, or C++ as they do any other spoken language. This may not be intuitive, but it makes sense to treat coding languages as any other language. And that’s what generative AI does.

Melodies

Based on musical notes, LLMs can generate melodies as they generate regular sentences. The potential of generative AI in this area is still unexplored, but it shows promising results for music creation.

Lyrics

Another example of linguistics, lyrics can be built based on specific criteria explained via prompt, in which the users can specify the type of words, inspiration, style, etc.

Image

The principle behind image creation is surprisingly intuitive: writing down the description (with simple natural language) of a potential image, to include it as part of the “prompt,” then waiting for the generative AI engine to return one or several results matching that prompt, based on its own interpretation of previously consumed images. This kind of capability is very interesting for creative and marketing activities, where human pros can leverage image generation tools as a source of inspiration. A good example of this is [Microsoft Designer](https://oreil.ly/oIRon), or the image creator capabilities of Microsoft Copilot.

Audio

Imagine a technology that allows you to record your own voice for a few minutes, and then reproduce and replicate it for whatever purpose you want. Some sort of scalable voice licensing that leverages audio data to detect patterns and then imitate them. There are systems that can even generate music and other sounds (for example, with [Microsoft Copilot integration with Suno’s AI-enabled music creation](https://oreil.ly/5ltEG)).

Video

As with image generation, the input can be a prompt describing specific scenes with different levels of detail, for which the model will deliver a video scene according to these details. A good example would be OpenAI Sora.

Others

Generative capabilities are not limited to only these formats and types of data. Actually, there are generative AI applications to create synthetic data, generate chemical compounds, etc.

These are just some of the capabilities that generative AI offers. They are fairly impressive, but certainly not the last step of the new AI era, as there are very relevant actors making sure that’s the case. Let’s see who the main contenders are next.

## Relevant Industry Actors

While this book focuses on Azure OpenAI Service, which is related to both Microsoft and OpenAI, it is important to understand the competitive landscape for generative AI. As you already know, this field is witnessing significant advancements and competition. Researchers and organizations are actively working to develop innovative models and algorithms to push the boundaries of generative AI capabilities. Here are some examples of relevant actors accelerating the competition:

[OpenAI](https://oreil.ly/xSlss)

Probably the most important actor of the generative AI wave. The company has created both proprietary tools such as ChatGPT, and other open source projects such as [Whisper](https://oreil.ly/9si-P)). OpenAI’s origins can be traced back to December 2015 when it was founded as a nonprofit organization by Elon Musk, Sam Altman, Greg Brockman, Ilya Sutskever, John Schulman, and Wojciech Zaremba. [Their mission](https://oreil.ly/yRGrR) is to ensure that artificial general intelligence (AGI) benefits all of humanity.

OpenAI initially focused on conducting research and publishing papers in the field of artificial intelligence to foster knowledge sharing and collaboration. In 2019, OpenAI created a for-profit subsidiary called OpenAI LP to secure additional funding for its ambitious projects. The company’s objective is to develop and deploy AGI that is safe, beneficial, and aligned with human values. They aim to build cutting-edge AI technology while ensuring it is used responsibly and ethically. They have democratized access to different kind of AI models:

*   *Conversational GPT models*, with their well-known [ChatGPT application](https://oreil.ly/HUWak), which relies on AI language models. It is based on the GPT (generative pre-trained transformer) architecture, which is the foundation of state-of-the-art language models known for their ability to generate human-like text and engage in conversational interactions. ChatGPT is designed to understand and generate natural language responses, making it well-suited for chat-based applications. It has been trained on a vast amount of diverse text data from the internet, allowing it to acquire knowledge and generate coherent and contextually relevant responses.

*   *Generative AI models* for text ([GPT-4o](https://oreil.ly/c2NvZ), [GPT-4](https://oreil.ly/0SY9B), and others), code ([Codex](https://oreil.ly/ZAbMD)), images ([DALL·E 3](https://oreil.ly/C9seS)), and videos ([Sora](https://oreil.ly/ppSjf)). Some of these models are available, as we will see in [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure), via Azure OpenAI Service.

*   *State-of-the-art speech-to-text models*, such as [Whisper](https://oreil.ly/9si-P), available as an open source repository, but also as an OpenAI [paid API](https://oreil.ly/DbmUg). Additionally, Whisper models are [available via Microsoft Azure](https://oreil.ly/qKUO5).

[Microsoft](https://oreil.ly/lYMby)

Along with OpenAI, the other key actor and one of the earliest adopters of generative AI technologies, thanks to the multimillion dollar investment in OpenAI and the [partnership between both companies](https://oreil.ly/hvKP2). Besides Azure OpenAI Service (the main topic of this book, which we will explore deeply in upcoming chapters), Microsoft has adopted LLMs as part of their technology stack to create a series of AI copilots for all their productivity and cloud solutions, including Microsoft Copilot. Also, they have released the [small language models (SML) Phi-2](https://oreil.ly/FW-xG) and [Phi-3](https://oreil.ly/vQTnL), setting a new standard for the industry from a size/performance point of view. We will explore more details in upcoming chapters, but the company strategy has become AI-first, with a lot of focus on generative AI and the continuous delivery of new products, platforms, features, and integrations.

[Hugging Face](https://oreil.ly/CGiU7)

Hugging Face is a technology company specializing in NLP and machine learning. It is known for developing the Transformers library, which provides a powerful and flexible framework for training, fine-tuning, and deploying various NLP models. Hugging Face’s goal is to democratize and simplify access to state-of-the-art NLP models and techniques. It was founded in 2016 by Clément Delangue and Julien Chaumond. Initially, the company started as an open source project aiming to create a community-driven platform for sharing NLP models and resources. Their Hugging Face Hub is a platform for sharing and accessing pre-trained models, datasets, and training pipelines. The hub enables users to easily download and integrate various NLP resources into their own applications, making it a valuable resource for developers and researchers. In addition to their open source contributions, Hugging Face offers commercial products and services. Their models are available via Azure AI thanks to the [corporate partnership between both companies](https://oreil.ly/eR8a0).

[Meta](https://oreil.ly/aHn9W)

Formerly known as TheFacebook and Facebook, Meta is a multinational technology company that focuses on social media, digital communication, and technology platforms. It was originally founded by Mark Zuckerberg, Eduardo Saverin, Andrew McCollum, Dustin Moskovitz, and Chris Hughes in 2004\. In recent years, they have created a very powerful organizational AI structure with relevant AI researchers, and meaningful open source AI contributions. They have released several models, including their most recent LLMs [Llama 3](https://oreil.ly/Mfau4) and [CodeLlama](https://oreil.ly/GBRM1), an interesting data-centric option with good performance (based on industry benchmarks) and lower computing requirements than other existing solutions. The latest models are also [available](https://oreil.ly/hpI6k) via Microsoft Azure, with [new features](https://oreil.ly/zvAg9) to fine-tune and evaluate them via Azure AI Studio, as part of the exclusive Meta-Microsoft partnership that positions Microsoft Azure as the [preferred cloud provider](https://oreil.ly/ehdf3) for Meta’s models.

[Mistral AI](https://oreil.ly/C4PM4)

A French company specializing in artificial intelligence. It was founded in April 2023 by researchers who previously worked at Meta and Google DeepMind. Mistral AI focuses on developing generative language models and stands out for its commitment to open source software, in contrast to proprietary models. Their Mixture of Experts (MoE) models are setting the standard for smaller language models, and are [available via the Azure AI model catalog](https://oreil.ly/t-MkN), including the [Mistral Large](https://oreil.ly/SU0PT) model.

[Databricks](https://oreil.ly/l9hDi)

A data intelligence platform (available as a [native service on Microsoft Azure](https://oreil.ly/7P_jM)) that has released their own LLMs, including an initial open source model called Dolly 2.0, trained by their own employees, and the first open source LLM for commercial purposes. In 2024, they released [new DBRX models](https://oreil.ly/XBH44) (base and instruct versions), also [available via the Azure AI model catalog](https://oreil.ly/WITi2).

[Google](https://oreil.ly/39CFk)

Google is another key competitor and one of the most relevant AI innovators. Its Google Cloud Platform (GCP) introduced new AI-powered features in Google Workspace and G-Suite, and Google Cloud’s Vertex AI platform is used to build and deploy machine learning models and AI applications at scale. Like Microsoft Azure, Google Cloud offers tools that make it easier for developers to build with generative AI and new AI-powered experiences across their cloud, including access to low-code generative AI tools. Finally, Google released [Gemini](https://oreil.ly/FZ8NN) (formerly known as Bard) as their alternative to OpenAI’s ChatGPT and Microsoft Copilot.

[NVIDIA](https://oreil.ly/mwSlg)

A pioneer in generative AI that offers a full-stack platform that enables innovation and creativity for solving complex challenges. Their platform includes accelerated computing, essential AI software, pre-trained models, and AI foundries. From a Microsoft point of view, there is a growing partnership between both companies, including the availability of their [generative AI foundry service on Microsoft Azure](https://oreil.ly/Ge8X7), and the [inclusion of NVIDIA AI models](https://oreil.ly/OPGhp) in the Azure AI model catalog.

[Anthropic](https://oreil.ly/SmS4F)

An AI company founded by former OpenAI employees. They also have their own ChatGPT-style bot called [Claude](https://oreil.ly/AeNhS), which is accessible through a chat interface and API in their developer console. Claude is capable of a wide variety of conversational and text processing tasks while maintaining a good degree of reliability and predictability. Their [Claude models](https://oreil.ly/df896) are available via APIs.

[Amazon Web Services (AWS)](https://oreil.ly/CaxMP)

AWS took some time to release generative AI–related products, but they recently announced their AWS Bedrock platform, a foundational AI service to directly connect to generative AI models. They offer their own models, and others from third parties such as Cohere or Anthropic.

[IBM](https://oreil.ly/z0XDj)

IBM announced their new WatsonX platform, which includes their own model catalog, a lab/playground environment, and API-enabled integrations.

[Cohere](https://oreil.ly/rDW3b)

An LLM-first company, with their own offering of language models, and their Coral productivity chatbot, which works as a knowledge assistant for companies.

You can see in [Figure 1-4](#fig_4_simplified_generative_ai_timeline) the exponential evolution of the generative AI market with a timeline of new models by company, especially after ChatGPT released in 2022, with a 2023 full of model and platform releases.

![](assets/aoas_0104.png)

###### Figure 1-4\. Simplified generative AI timeline

This timeline is a highly simplified version of the advancements and releases from different open source teams and other companies. For more details, both the [State of AI Report](https://oreil.ly/UwMhc) and the [Stanford AI Index Report](https://oreil.ly/GDYK7) contain plenty of details about research and commercial models, as well as other relevant actors we haven’t mentioned here. The list of generative AI innovations will certainly evolve during the coming months and years, and future implementations of existing models like Meta’s Llama 3 and OpenAI’s [GPT-4 and GPT-4o](https://oreil.ly/3GPA5) will likely focus on the efficiency of the models.

Now, let’s see why generative AI is a special kind of artificial intelligence, and explain a new concept called foundation models, which is the key differentiator when comparing to traditional language models.

## The Key Role of Foundation Models

There are several reasons generative AI is a total disruption. The perception of a never-seen level of performance is one of them. The ability to interact using plain language to send our instructions and to interpret the results is another one. However, one of the fundamental aspects for generative AI to deliver the value we see nowadays is the notion of *foundation models*.

Foundation models are base models pre-trained with a vast amount of information (e.g., LLMs) that are able to perform very different tasks. This is something new as traditional AI/NLP models focus on unitary tasks, one specific model per task (e.g., language translation).

For example, Azure OpenAI models such as GPT-4 and GPT-4o can do plenty of things by leveraging one single model. They can perform diverse tasks related to a specific generative capability, such as text/language, and help you analyze, generate, summarize, translate, classify, etc., all with only one model. In addition to that, if the models are able to handle different kinds of inputs at the same time, like text and image, they qualify as *multimodal models* (e.g., [GPT-4V](https://oreil.ly/kDBF5)). You can see the main differences in [Figure 1-5](#fig_5_traditional_ai_versus_foundation_models).

This flexible approach provides multiple options for the development of new use cases, and you will see later (in Chapters [2](ch02.html#designing_cloud_native_architectures_for_generativ) and [3](ch03.html#implementing_cloud_native_generative_ai_with_azure)) how Azure OpenAI facilitates the configuration, testing, and deployment of these foundation models. But what does it represent in terms of AI disruption? Let’s see first one of the fundamental reasons why generative AI and companies such as OpenAI got so much attention in recent years.

![](assets/aoas_0105.png)

###### Figure 1-5\. Traditional AI versus foundation models

## Road to Artificial General Intelligence

Before we dig into the core part of this book, it’s important to contextualize all these innovations within the general state of artificial intelligence, and the current discussions on *artificial general intelligence* (AGI) due to the unexpected capabilities of GPT-4 and other LLMs.

You may remember some cinematographic references to what a lot of people imagine as artificial intelligence—Skynet, Ultron, *I, Robot*, etc. All of them showed some sort of superior intelligence, usually represented by strong and dangerous humanoid robots that evolve over time, and that plan somehow to replace or even destroy the human race. Well, even if it is not the purpose of this book to show a naive vision of what AI and its capabilities are, we will start by demystifying and clarifying the current level of development of artificial intelligence, so everyone can understand where we are and what the realistic expectations of an AI system are. For that purpose, here are three types of AI depending on their scope and level of intelligence:

Narrow AI

The current type of capabilities that AI systems and technologies are offering. Basically, this is an AI that can get a relatively big sample of past data, and then generate predictions based on that, for very specific tasks, for example, detecting objects from new images, recognizing people from audio voices, etc.

General AI (or artificial general intelligence)

The next goal for AI researchers and companies. The idea is to generalize the training process and the knowledge it generates for AI and leverage that within other domains. For example, how can we make an AI-enabled personal assistant aware of the changing context? And then adapt previous learnings to new situations? This is not 100% feasible today, but likely to happen at some point.

Super AI

The kind of artificial intelligence that movies and books are continuously showing. Its capabilities (cognitive, physical, etc.) are far superior to humans, and it can in theory surpass them. However, this kind of super intelligence is currently a futuristic vision of what an artificial intelligence could be. It’s still not feasible and probably will not happen in the upcoming years or even decades (this opinion will be different depending on who you ask).

Bringing this back to the topic of generative AI, current discussions focus on the current stage or type of artificial intelligence. But the real question is, are we still talking about narrow AI? Are we getting closer to general AI? It is a fair question given the new level of performance and flexibility of foundation models to perform a variety of tasks. Regardless of the answer (which can go from the technical to the philosophical), reality is that generative AI in general, and Azure OpenAI Service in particular, are delivering capabilities we never dreamt about before.

There was an [early analysis of the GPT-4 model](https://oreil.ly/rhEVW) capabilities from the Microsoft team that explored this relation between the foundation models, and talks about “close to human-level performance” and an “early version of an AGI system.” Also, companies like OpenAI have declared the [pursuit of AGI](https://oreil.ly/vTgyJ) as one of their main goals.

We have covered all the fundamentals related to generative AI topics, including evolution from traditional AI, recent developments, and ongoing discussions around performance and the impact of generative AI. Let’s now explore the details of Azure OpenAI Service, with special focus on the story behind it and the core capabilities.

# Microsoft, OpenAI, and Azure OpenAI Service

Microsoft, one of the main technology incumbents, and OpenAI, a relatively young AI company, have collaborated and worked together in recent years to create impressive technologies, including AI supercomputers and LLMs. One of the main aspects of this partnership is the creation of [Azure OpenAI Service](https://oreil.ly/oZT0k), the primary reason for this book, and a PaaS cognitive service that offers an enterprise-grade version of the existing OpenAI services and APIs, with additional cloud native security, identity management, moderation, and responsible AI features.

The collaboration between companies became more famous in 2023, but reality is that it had several stages with very important milestones at both the technical and business level:

*   It started in 2019, when Microsoft announced a $1 billion investment in OpenAI to help advance their AI research activities and to create new technologies.

*   In 2021, they announced another level of partnership to build large-scale AI models using Azure’s supercomputers.

*   In January 2023, they announced the third phase of their long-term partnership through a multiyear, multibillion dollar investment to accelerate AI breakthroughs to ensure these benefits are broadly shared with the world.

Obviously, every step of this partnership has deepened the level of collaboration and the implications for both companies. The main areas of work are as follows:

Generative AI infrastructure

Building new Azure AI supercomputing technologies to support scalable applications for both OpenAI and Microsoft generative AI applications, and porting existing OpenAI services to run on Microsoft Azure.

Managed generative AI models

Making Microsoft Azure the preferred cloud partner for commercializing new OpenAI models via Azure OpenAI Service, which for you as an adopter means that any OpenAI model is available via Microsoft Azure, as a native enterprise-grade service in the cloud, in addition to the existing [OpenAI APIs](https://oreil.ly/v-HE1).

Microsoft Copilot products

As we will see in the following pages, Microsoft has infused AI into their product suite by creating AI-enabled copilots that help users perform complex tasks.

Also, Azure OpenAI Service is not the only Microsoft AI service, and it is part of the Azure AI Suite (shown in [Figure 1-6](#fig_6_azure_openai_and_other_azure_ai_services)), which includes other PaaS options for a series of advanced capabilities that can colive and interact to create new AI-enabled solutions.

![](assets/aoas_0106.png)

###### Figure 1-6\. Azure OpenAI Service and other Azure AI services

We will refer to some of these building blocks in Chapters [3](ch03.html#implementing_cloud_native_generative_ai_with_azure) and [4](ch04.html#additional_cloud_and_ai_capabilities), as most of these services interact seamlessly with Azure OpenAI Service, depending on the envisioned solution architecture. But this is a highly evolving field, and [Figure 1-7](#fig_7_azure_openai_service_timeline) shows the timeline of key [Azure OpenAI breakthroughs](https://oreil.ly/5zqx_) in recent months and years.

![](assets/aoas_0107.png)

###### Figure 1-7\. Azure OpenAI Service timeline

If you want to understand more about the origins of the partnership and the initial developments, this [podcast episode](https://oreil.ly/uGSH_) with Microsoft’s CTO Kevin Scott and cofounder (and former CEO) Bill Gates is very interesting and explains how everything started.

## The Rise of AI Copilots

As part of its AI-enabled offerings, Microsoft is promoting the concept of *AI copilots*. They are personal assistants equipped with Microsoft’s AI, OpenAI’s GPT models, and other generative AI technologies, designed to assist users in their tasks and goals, but not to replace humans and their jobs. Copilots work alongside users, providing suggestions, insights, and actions based on AI. Users always have control and the choice to accept, modify, or reject the copilot’s output. From a visual point of view, copilots are usually on the right side of the screen, and Microsoft has included them in several solutions:

GitHub Copilot

An [AI-powered pair programmer](https://oreil.ly/KanxU) that helps developers write better code faster. It suggests whole lines or entire functions right inside the editor, based on the context of the code and comments. GitHub Copilot is powered by GPT-4 (previously enabled by [OpenAI Codex](https://oreil.ly/ZAbMD), now deprecated), a system that can generate natural language and computer code. GitHub Copilot is the original case and the first copilot of the Microsoft suite.

Bing Chat/Microsoft Copilot

A [conversational AI service](https://oreil.ly/gI9u1) that helps users find information, get answers, and complete tasks on the web. It uses GPT models that can produce natural language responses based on user input. Users can chat with Bing Chat using text or voice on browsers or the Bing app. This is the first search engine to incorporate generative AI features for chat-based discussion, now rebranded as Microsoft Copilot.

Microsoft 365 Copilot

An [AI-powered copilot](https://oreil.ly/EQmLe) for work that helps users unleash their creativity, improve their productivity, and elevate their skills. It integrates with Microsoft 365 applications such as Word, Excel, PowerPoint, Outlook, Teams, and Business Chat. It also leverages LLMs such as Azure OpenAI GPT-4 to generate content, insights, and actions based on natural language commands.

Windows Copilot

An [upgraded AI assistant for Windows 11](https://oreil.ly/uJSZC) that helps users easily take action and get things done. It integrates with Bing Chat, as well as with Windows features and third-party applications. Users can interact with Windows Copilot using natural language commands.

Fabric and Power BI Copilot

A [generative AI interface for Microsoft Fabric](https://oreil.ly/Ipshc), the lakehouse platform, and Power BI, for automated reporting.

Security Copilot

An [AI-enabled security solution](https://oreil.ly/aKBXf) that helps users protect their devices and data from cyber threats. It uses AI to detect and prevent malware, phishing, ransomware, and other attacks. It also provides users with security tips and recommendations based on their behavior and preferences.

Clarity Copilot

A feature that incorporates [generative AI into Microsoft Clarity](https://oreil.ly/5MgnN), an analytics tool that helps users understand user behavior on their websites. It allows users to query their Clarity and Google Analytics data using natural language and get concise summaries. It also generates key takeaways from session replays using AI.

Dynamics 365 Copilot

A [feature that brings next-generation AI](https://oreil.ly/AVM6q) to traditional customer relationship management (CRM) and enterprise resource planning (ERP) solutions. It helps users optimize their business processes, improve customer engagement, and increase revenue. It leverages LLMs such as OpenAI’s GPT-4 to generate insights, recommendations, and actions based on natural language commands.

Others

Power Platform Copilot, Microsoft Designer (software as a service [Saas] for visual design with a generative AI prompt interface), and the new [Copilot Studio](https://oreil.ly/iq_fU) for low-code gen AI implementations.

Summarizing, Microsoft has released a series of AI copilots for their product suite, and the reality is that Azure OpenAI Service is the key piece to *creating your own copilots*. We will analyze different building blocks of an AI copilot for cloud native applications (e.g., new terms such as plug-ins and orchestrators), but you can see in [Figure 1-8](#fig_8_the_modern_ai_copilot_technology_stack_adapted_fr) an adapted version of the “AI Copilot” [layered architecture](https://oreil.ly/jGrqE) that Microsoft presented during Microsoft Build 2023.

![](assets/aoas_0108.png)

###### Figure 1-8\. The modern AI copilot technology stack (source: adapted from an image by Microsoft)

As you can see in the figure, the AI infrastructure and foundation models are just part of the equation. Both a cloud native architecture and specific generative AI pieces are required to develop AI copilots for your existing and new applications, and that’s exactly what we will cover in Chapters [2](ch02.html#designing_cloud_native_architectures_for_generativ), [3](ch03.html#implementing_cloud_native_generative_ai_with_azure), and [4](ch04.html#additional_cloud_and_ai_capabilities). But before that, let’s explore the high-level capabilities and typical use cases of Azure OpenAI.

## Azure OpenAI Service Capabilities and Use Cases

We will focus now on the core capabilities and potential use cases of Azure OpenAI–enabled systems, before going into architectural and technical considerations. Keeping in mind the flexible nature of foundation models, it is easy to imagine the multiple applications of Azure OpenAI models. Let’s explore the main capabilities in [Table 1-1](#table-1-1) (there are more, but you can use this as a baseline for your initial use ideation), aligned with those we have previously seen in this chapter.

Table 1-1\. Main Azure OpenAI Service capabilities and use cases

| Type | Capability and illustrative example |
| --- | --- |
| **Language** | Content generation/ analysis | Text generation | Automatic creation of SMS with dynamic formats and content |
| Topic classification | Detect book topics based on their content, for automatic labeling |
| Sentiment analysis | Detect sentiment from social media reviews to detect pain points |
| Entity extraction | Find key topics from specific information |
| Call to APIs | Generate an API call and integrate it with other systems |
| Subject matter expert documents | Creation of role-based documentation based on books or repositories |
| Machine translations | On-demand website translation |
| Technical reports | Generation of reports based on databases and other information |
| Agent assistance | Step-by-step, dynamic blueprints for customer agents |
| Summarization | Book summaries | Summarization of long documents (e.g., books) with specific format and sections |
| Competitive analysis | Extraction of key factors from two companies for competitive analysis |
| Social media trends analysis | Summarization of keyword trends and connection with online news |
| Reading comprehension | Reformulation of key topics with a simpler language |
| Search | Internet results | Semantic search for internet topics |
| Social reviews search | Detailed search of specific topics from social reviews on the internet |
| Knowledge mining | Extraction of knowledge from different sources, from the same topic |
| Document analysis | Search of key topics and other related terms for a specific document |
| Automation | Claim management | Automatic structuration of text-based information to send it as a JSON file |
| Financial reporting | Quarterly reporting based on social media summarization, figures from databases, and automation of the final report and its distribution |
| Automatic answers to clients | Automatic voice-enabled answers, or chatbot discussions for Level 1 support |
| **Coding** | Natural language to coding language | Generating a Java loop, based on natural language instructions |
| Coding recommendations | Live coding recommendations from the development tool |
| Automatic comments | Automatic comment generation based on written code |
| Refactoring | Automated code improvements |
| Code translation | Translation from one programming language to another |
| SQL queries in natural language | Database queries in natural language |
| Code review | AI-enabled pair review |
| Pull request info | Automated pull request comments |
| Text JSON-ization | Conversion of plain text into JSON file with specific parameters |
| **Image** | Creative ideation | Random image generation related to a specific topic |
| Podcast and music playlist images | Image generation based on podcast transcript or music lyrics |
| Content syndication | Material for partner-enabled marketing |
| Hyper-personalization | Visual customization based on user context |
| Marketing campaign personalization | Visuals for marketing campaigns, based on user segment, topic, etc. |

These are just a few examples of how to use the multiple capabilities of Azure OpenAI Service models. They can be combined with other services, and the models may also evolve, so don’t discard scenarios for audio or video generation.

Regardless of the type of capability and use case, Azure OpenAI Service can provide support to different kinds of scenarios:

Completion

Completions are used to generate content that finishes a given prompt. You can think of it as a way to predict or continue a piece of text. Completions are often useful for tasks like content generation, coding assistance, story writing, etc.

Chat

Chat scenarios are designed to simulate a conversation, allowing back-and-forth exchanges with the model. Instead of giving a single prompt and getting a continuation, users provide a series of messages, and the model responds to them in kind. Chat scenarios (like those powering ChatGPT) are useful for interactive tasks, including but not limited to tutoring, customer support, and of course, casual chatting.

Embeddings

We will explore the notion of embeddings by the end of [Chapter 2](ch02.html#designing_cloud_native_architectures_for_generativ), but they basically allow us to consume specific knowledge from documents and other sources. We will leverage this sort of capability in several [Chapter 3](ch03.html#implementing_cloud_native_generative_ai_with_azure) scenarios.

The dynamic behind all these examples is the same. Azure OpenAI is a PaaS that works based on cloud consumption. Unlike other cloud services or APIs that bill their capabilities based on a number of interactions, Azure OpenAI (and other commercial LLM platforms) measure service usage based on a new concept called “tokens.” Let’s see what this is about.

## LLM Tokens as the New Unit of Measure

In general terms, cloud and SaaS providers use very diverse ways to bill their services, from fixed monthly fees and usage tiers with volume discounts to very granular units of measure such as characters, words, or API calls.

In this case, generative AI has adopted the notion of *tokens*, which is a [set of words or characters](https://oreil.ly/Vy9ny) in which we split the text-based information. The tokens unit is used for two purposes:

*   For *consumption*, to calculate the cost of the configuration and interactions with the Azure OpenAI models. Any API call, prompt (text request) sent to the model, and completion (answer) delivered by Azure OpenAI follows this unit. The [service pricing](https://oreil.ly/7Gmq6) is based on cost per 1,000 tokens, and it depends on the model type (GPT-3.5 Turbo, GPT-4, GPT-4o, DALL·E 3, etc.).

*   For *capacity*, at both the model and service levels:

    *   *Token limit*, which is the maximum input we can pass to any Azure OpenAI model (and generative AI models in general). For example, GPT-3.5 Turbo offers two options with a 4K and 16K token limit, and GPT-4, GPT-4 Turbo, and GPT-4o reach 128K. This is likely to evolve in the coming months and years. For updated information, visit the [model availability](https://oreil.ly/BI5Ue) page and check the “Max Request (Tokens)” column.

    *   *Service quotas*, which means the maximum capacity at a certain resource, configuration, and usage level for any Azure OpenAI model. This is also evolving information, and it is available via [official documentation](https://oreil.ly/wwpp8) and the [Quota section](https://oreil.ly/ONn5Q) from Azure OpenAI Studio. These limits are important for any deployment plan, depending on the type of application (e.g., if we are planning to deploy a service for massive business-to-consumer [B2C] applications). Also, there are recommended [best practices](https://oreil.ly/Dv9qf) to handle these limitations.

The specific amount of tokens depends on the number of words (other providers calculate tokens based on characters, instead of words), but also on their length and language. The general rule is 1,000 tokens is approximately 750 words for the English language, but OpenAI explains the [specific way](https://oreil.ly/tMYF_) to calculate tokens depending on the case. Additionally, you can always use Azure OpenAI Playground or OpenAI’s [tokenizer](https://oreil.ly/DDQHG) to calculate a specific token estimate based on the input text.

# Conclusion

This first chapter was a mix of intro-level information related to AI and generative AI and a preliminary introduction to Azure OpenAI topics, including recent developments, primary capabilities, typical use cases, and its value as an AI copilot enabler for your own generative AI developments.

Depending on your background, this information may be just a 101 introduction, but the concepts behind the Azure OpenAI Service, even if they are new and include some new terms, can be as simple as it looks—a managed PaaS that will allow you to deploy your own cloud native, generative AI solutions.

In [Chapter 2](ch02.html#designing_cloud_native_architectures_for_generativ), we will analyze the potential scenarios for cloud native development, their connection with Azure OpenAI, and the architectural requirements that will help you prepare everything, before even implementing your Azure OpenAI–enabled solutions. As with this chapter, if you already have some preliminary knowledge of cloud native and Azure architectures, you may read it as a way to connect the dots and understand the specifics of these topics adapted to generative AI. If you are totally new to the topic, feel free to read the content and explore any external resource that may support your upskilling journey. We’re just getting started!