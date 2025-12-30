# Chapter 1\. Introduction

By the end of this chapter, you should be able to identify the role of GenAI within the roadmap of your own applications and its associated challenges.

# What Is Generative AI?

*Generative AI* is a subset of machine learning that focuses on creating new content using a model trained on a dataset. The *trained model*, which is a mathematical model representing patterns and distributions in the training data, can produce new data that is similar to the training dataset.

To illustrate these concepts, imagine training a model on a dataset containing images of butterflies. The model learns the complex relationships between pixels in images of butterflies. Once trained, you can sample from the model to create novel images of butterflies that didn’t exist in the original dataset. These images will contain similarities with the original images of butterflies yet remain different.

###### Note

Using a trained generative model to create new content based on patterns learned from the training data is known as inference.

[Figure 1-1](#generative_ai) shows the full process.

![bgai 0101](assets/bgai_0101.png)

###### Figure 1-1\. A generative model trained to create new photos of butterflies

Since we don’t want to generate the same outputs as the training dataset, we add some random noise during the sampling process to create variations in the outputs. This random component that affects the generated samples makes the generative model *probabilistic*. It distinguishes a generative model from a fixed calculation function that as an example averages pixels of multiple images to create new ones.

When working on GenAI solutions, you may come across six families of generative models, including:^([1](ch01.html#id448))

Variational autoencoders (VAEs)

VAEs learn to encode data into a low-dimensional mathematical space (called a *latent space*—see [Figure 1-2](#latent_space)) to decode back to the original space when generating new data.

Generative adversarial networks (GANs)

GANs are a pair of neural networks (a discriminator and a generator) battling each other to learn patterns in the data during training. Once trained, you can use the generator to create new data.

Autoregressive models

These models learn to predict the next value in a sequence based on previous values.

Normalizing flow models

These models transform simple probability distributions (patterns in data) to more complex ones for generating new data.

Energy-based models (EBMs)

EBMs are based on statistical mechanics. They define an energy function that assigns lower energy to observed data and higher energy to other configurations, and they are trained to differentiate between these configurations.

Diffusion models

Diffusers learn to add noise to training data to create a pure noisy distribution. Then learn to incrementally remove noise from sampled points (from the pure noisy distribution) to generate new data.

Transformers

Transformers can model large sequential data like corpora of texts with extremely efficient parallelism. These models use a *self-attention* mechanism to capture the context and relationships between elements in a sequence. Given a new sequence, they can use the learned patterns to generate new sequences of data. Transformers are commonly used as *language models* to process and generate textual data as they better handle long-range relationships in text. OpenAI’s ChatGPT is a language model referred to as a *generative pretrained transformer* (GPT).

One fact to note is that these generative models often can process only certain types of data or *modalities* such as text, images, audio, video, point clouds, or even 3D meshes. Some are even *multimodal,* like OpenAI’s GPT-4o, that can natively process multiple modalities like text, audio, and images.

To explain these GenAI concepts, I will use an image generation use case as an example. Other use cases include language models being used as chatbots or document parsers, audio models used for music generation or speech synthesis, and video generators for creating AI avatars and deepfakes. You may have witnessed more than a handful of other use cases with hundreds more yet to be uncovered.

One fact remains for certain. GenAI services will power future applications. Let’s see why.

# Why Generative AI Services Will Power Future Applications

We use computers to automate solutions to everyday problems.

In the past, automating a process required manually coding business rules, which could be time-consuming and tedious, especially for complex problems such as spam detection when relying on handwritten rules. Nowadays, you can train a model to understand the nuances of the business process. Once trained, this model can outperform handwritten rules implemented as application code, resulting in replacing such rules.

This shift toward model-based automation has given rise to AI-powered applications in the market, solving a range of problems including price optimization, product recommendation, or weather forecasting. As part of this wave, generative models emerged that differed from other types of AI in their ability to produce multimedia content (text, code, images, audio, video, etc.), whereas traditional AI is more about prediction and classification.

As a software engineer, I believe these models have certain capabilities that will influence the development roadmap of future applications. They can:

*   Facilitate the creative process

*   Suggest contextually relevant solutions

*   Personalize user experience

*   Minimize delay in resolving customer queries

*   Act as an interface to complex systems

*   Automate manual administrative tasks

*   Scale and democratize content generation

Let’s look at each capability in more detail.

## Facilitating the Creative Process

Mastering skills and acquiring knowledge are cognitively demanding. You can spend a long time studying and practicing something before you form your own original ideas for producing novel and creative content like an essay or a design.

During the creative process, you may have the writer’s block—difficulties with imagining and visualizing scenes, navigating ideas, creating narratives, constructing arguments, and understanding relationships between concepts. The creative process requires a deep understanding of the purpose behind your creation and a clear awareness of sources for inspiration and ideas you intend to draw upon. Often, when you sit down to make something new like an original essay, you may find it difficult to start from a blank screen or piece of paper. You will need to have done extensive research on a topic to have formed your own opinions and the narrative you want to write about.

The creative process also applies to design, not just writing. For instance, when designing a user interface, you may need a few hours of design research by browsing design websites for ideas on the color palette, layout, and composition before you originate a design. Creating something truly original by looking at a blank canvas can feel like climbing a wall bare-handed. You will need inspiration and have to follow a creative process.

Producing original content requires creativity. Therefore, it is an extraordinary feat for humans to produce original ideas from scratch. New ideas and creations are often based on inspirations, connecting ideas and adaptations of other works. Creativity involves complex, nonlinear thinking and emotional intelligence that makes it challenging to replicate or automate with rules and algorithms. Yet it is now possible to mimic creativity with generative AI.

GenAI tools can help you to streamline the process by bridging various ideas and concepts from an extensive repository of human knowledge. Using these tools, you can stumble upon novel ideas that required understanding a large body of interconnected knowledge and the comprehension of the interactions between several concepts. Additionally, these tools can assist you with imagining hard-to-visualize scenes or concepts. To illustrate the point, try imagining the scene described in [Example 1-1](#image_prompt).

##### Example 1-1\. Description of a scene that is hard to visualize for humans

```py
An endless biomechanical forest where trees have metallic roots and glowing neon
leaves, each tree trunk embedded with rotating gears and digital screens
displaying alien glyphs; the ground is a mix of crystalline soil and pulsating
organic veins, while a surreal sky shifts between a digital glitch matrix and a
shimmering aurora made of liquid light.
```

This can be quite difficult to imagine unless you’ve been accustomed to imagining such concepts. However, with the help of a generative model, anyone can now visualize and communicate challenging concepts to others.

Providing the scene description in [Example 1-1](#image_prompt) to an image generation tool such as [DALL-E 3 (OpenAI)](https://oreil.ly/Z80Qm) produces an output shown in [Figure 1-3](#dalle_image).

![bgai 0103](assets/bgai_0103.png)

###### Figure 1-3\. An image produced by [DALL-E 3](https://oreil.ly/Z80Qm)

It is fascinating to see how these GenAI tools can help you visualize and communicate challenging concepts. These tools allow you to expand your imagination and nudge your creativity. When you feel stuck or find it difficult to communicate or imagine novel ideas, you can turn to these tools for help.

In the future, I can see applications including similar features to help users in their creative process. If your applications give several suggestions for the user to build upon, it can help them get onboarded and build momentum.

## Suggesting Contextually Relevant Solutions

Often you will find yourself facing niche problems that don’t have a previously established solution. Solutions to these problems aren’t obvious and require a lot of research, trial and error, consultation with other experts, and reading. Developers are familiar with this situation, as finding relevant solutions to programming problems can be tricky and not straightforward.

This is because developers must solve problems with a certain context in mind. A problem can’t be defined without a thorough description of “circumstances,” and the “situation” arises in the *context*.

Essentially, *context* narrows down the potential solutions to a problem.

With search engines, you look for sources of information with a few keywords that may or may not contain a relevant solution. When developers search for solutions, they paste error logs into Google and are directed to programming Q&A websites like [Stack Overflow](https://stackoverflow.com). Developers must then hope to find someone who has encountered the same problem in the same context and that a solution has been provided. This method of finding solutions to programming problems is not very efficient. You may not always find the solution you’re looking for as a developer on these websites.

Developers are now turning to generative AI to solve programming issues. By providing a prompt that describes the context of a problem, the AI can generate potential solutions. Even better, code editor integrations go a long way in providing this context to the language models, something not possible when searching in Google or Stack Overflow. These AI models can then generate solutions that are contextually relevant and based on a learned knowledge base sourced from online forums and several Q&A websites. With the proposed solution(s), you can then decide if any is appropriate.

Because of this, using GenAI coding tools is often quicker than searching for solutions on online forums and websites. Even the [programming Q&A site, Stack Overflow](https://oreil.ly/nOX_K), has attributed an above-average traffic decrease (~14%) to developers trialing the GPT-4 language model and code generator after its release. This figure may be higher as several site power users have commented on the company’s blog post that they feel user activity on the site has reduced dramatically. In fact, there has been a [reported ~60% decline in questions asked and upvote activity on the site](https://oreil.ly/P6Kur), at the time of writing, when compared to 2018.

In any case, Stack Overflow still expects traffic to rise and fall in the future with the introduction of GenAI coding tools democratizing coding, expanding the developer community, and creating new programming challenges. The power of Q&A sites lies not just in finding the answer but also in understanding the surrounding discussions and the importance of referencing sources.^([2](ch01.html#id470)) As a result, these sites will remain an invaluable resource to developers due to their communities of experts and human-curated content upholding trust in the correctness and quality of answers or solutions.^([3](ch01.html#id472))

## Personalizing the User Experience

Customers and users of modern software expect a certain level of personalization and interactivity when they use modern applications.

By integrating generative models such as a language model into existing applications, you can innovate how users interact with the system. Instead of the traditional UI interaction where you have to click into several screens, you can converse in natural text with a chatbot to ask for the information you seek or an action to be performed on your behalf. For example, when browsing a travel planning site, you can describe your ideal holiday and have the chatbot prepare an itinerary for you based on the platform’s access to airlines, accommodation providers, and the database of package holiday deals. Or, if you’ve already booked a holiday, you can ask for sightseeing recommendations based on itinerary details from your account data. The chatbot can then describe the results back to you and ask for your feedback.

These language models can act as a personal assistant by asking relevant questions until they map your preferences and unique needs to a product catalog for generating personalized recommendations. These virtual assistant can understand your intent and suggest choices relevant to your situation. If you don’t like the suggestions, you can provide some feedback to refine any suggestions to your liking.

In education, these GenAI models can be used to describe or visualize challenging concepts tailored to each student’s learning preferences and abilities.

In gaming and virtual reality (VR), GenAI can be used to construct dynamic environments and worlds based on user interactions with the application. For instance, in role-playing games (RPGs), you can produce narratives and character stories on the fly, based on users’ decisions and dialogue choices in real time using a baked-in large language model. This process creates a unique experience for the gamer and users of these applications.

## Minimizing Delay in Resolving Customer Queries

Aside from personalized user assistants, businesses often need support in handling a large volume of customer service queries. Due to this volume, customers often have to wait in long queues or several business days before they hear back from businesses. Furthermore, as businesses grow in operational complexity and customers, resolving customer queries in a timely manner can become more expensive and require extensive staff training.

GenAI can streamline customer service processes for both the customers and the businesses. Customers can now chat or go on a call with a language model capable of accessing databases and relevant sources to resolve queries in a matter of minutes, not days. As customers describe their issues, the model can address these queries in accordance with business policies and can direct customers to relevant resources when necessary.

While traditional chatbots often relied on a set of handcrafted rules and predefined scripts, GenAI-powered chatbots can better:

*   Understand conversation context

*   Consider user preferences

*   Form dynamic and personalized responses

*   Accept and adjust to user feedback

*   Handle unexpected queries, in particular over historical or larger conversations

These factors enable GenAI chatbots to have a more natural and varied interaction with the customer. These bots will be the first point of contact for customers who want their queries swiftly answered before cases are escalated to human agents. As a customer, you may also prefer talking to one of these GenAI chatbots first if it means avoiding long queues and achieving a quick resolution.

These examples only scratch the surface of all possible features that can be integrated into existing applications. This flexibility and agility of generative models opens up many possibilities for novel applications in the future.

## Acting as an Interface to Complex Systems

Many people these days still face problems when interacting with complex systems such as databases or developer tools. Nondevelopers may need to access information or perform tasks without having the necessary skills to execute commands on these complex systems. LLMs and GenAI models can act as the interface between these systems and their users.

Users can provide a prompt in natural language, and GenAI models can write and execute queries on complex systems. For instance, an investment manager can ask a GenAI bot to aggregate the performance of the portfolio in the company’s database without having to submit requests for reports to be produced by specialists. Another example is Photoshop’s new generative fill tool that generates image layers and performs context-informed edits for users who have not mastered Photoshop’s various tools.

Already, several AI startups have developed GenAI applications in which users interact with a language model in natural language to perform tool actions. Using language models, these startups are replacing complex workflows and clicking around in multiple UI screens.

While GenAI models can act as an interface to complex systems like databases or APIs, developers will still need to implement guardrails and security measures, as you will learn in [Chapter 9](ch09.html#ch09) on AI security. These integrations will need to be handled carefully to avoid malicious queries and attack vectors via generative models on these systems.

## Automating Manual Administrative Tasks

Across many large and long-standing companies, there are often several teams performing manual administrative tasks that are less visible to the front-house teams and their customers.

A typical administrative task involves manually processing documents with complex layouts, like invoices, purchase orders, and remittance slips. Until recently, these tasks have remained mostly manual since each document’s layout and information arrangement can be visually unique, requiring human validation or sign-off. On top of this, any developed software to automate these processes could be fragile and held to a high level of accuracy and correctness, even in edge cases.

Now, language and other generative models can enable some parts of these manual processes to be further automated and enhanced for higher accuracy. If the existing automations fail to perform due to edge cases or changes in the process, language models can step in to check the outputs against some criteria and fill in the gaps or flag items for manual review.

## Scaling and Democratizing Content Generation

People love new content and are always on the lookout for new ideas to explore. Writers can now research and ideate when writing a blog post with the help of GenAI tools. By conversing with a model, they can brainstorm ideas and generate outlines.

The productivity boost is enormous for content generation. You no longer have to perform low-level cognitive tasks of summarizing research or rewording sentences yourself. The time it takes to produce a quality blog post is slashed from days to hours. Instead of starting from scratch, you can focus on the outline, flow, and structure of the content before using GenAI to fill in the gaps. When you struggle with sequencing the right words for clarity and brevity, GenAI tools can shine. However, what makes a piece of writing interesting often isn’t the content, but the style of writing and flow.

Many businesses have already started using these tools to explore ideas and draft documents, proposals, social media, and blog posts.

Overall, these are several reasons why I believe more developers will be integrating GenAI features into their applications in the future. The technology is still in its infancy, and there are still many challenges to overcome before GenAI can be widely adopted.

# How to Build a Generative AI Service

Generative models need access to rich contextual information to provide more accurate and relevant responses. In some cases, they may also need access to tools to perform actions on the user’s behalf—for instance, to place an order by running a custom function. As a result, you may need to build APIs around generative models (as wrappers) to take care of integrations with external data sources (i.e., databases, APIs, etc.) and controlling user access to the model.

To build these API wrappers, you can place generative models behind an HTTP web server and implement the required integrations, controls, and routers as shown in [Figure 1-4](#web_server).

![bgai 0104](assets/bgai_0104.png)

###### Figure 1-4\. FastAPI web server with data source integrations that serve a generative model

The web server controls access to the data sources and the model. Under the hood, your server can query the database and external services to enrich user prompts with relevant information for generating more relevant outputs. Once outputs are generated, the control layer can then sanity-check while the routers return final responses to the user.

###### Tip

You can even go one step further by configuring a language model to construct an instruction for another system and pass it off to another component to execute those commands such as to interact with a database or make an API call.

In summary, the web server acts as a crucial intermediary that manages data access, enriches user prompts, and quality controls the generated outputs before routing them to users. Alongside serving generative models to users, this layered approach enhances the relevance and reliability of responses from generative models.

# Why Build Generative AI Services with FastAPI?

Generative AI services require performant web frameworks as backend engines powering event-driven services and applications. FastAPI, one of the most popular web frameworks in Python, [can compete](https://oreil.ly/LmEg7) in performance with other popular web frameworks such as *gin (Golang)* or *express (Node.js)*, while holding onto the richness of Python’s deep learning ecosystem. Non-Pythonic frameworks lack this direct integration required for working with a generative AI model within one service.

Within the Python ecosystem, there are several core web frameworks for building API services. The most popular options include:

Django

A full-stack framework that comes with batteries included. It is a mature framework with a large community and a lot of support.

Flask

A micro web framework that is lightweight and extensible.

FastAPI

A modern web framework built for speed and performance. It is a full-stack framework that comes with batteries included.

FastAPI, despite its recent entry into the Python web framework space, has gained traction and popularity. As of this writing, FastAPI is the fastest growing Python web framework in terms of package downloads and the second most popular web framework on GitHub. It is on the trajectory to become more popular than Django based on its growing count of [GitHub Stars](https://oreil.ly/8fRO2) (around 80,000 at the time of writing).

Among the frameworks mentioned, Flask leads in number of package downloads due to its reputation, community support, and extensibility. However, as a micro web framework, it ships with a limited number of default features, such as out-of-the-box support for schema validation.

Django is also popular for building APIs (via Django Rest Framework) and monolith applications following the model-view-controller (MVC) design pattern. But it has less mature support for asynchronous APIs with potential performance limitations, plus can add complexity and overhead to building lightweight APIs.

Compared to other web frameworks, FastAPI provides several features out of the box such as data validation, type safety, automatic documentation, and a built-in web server. Because of this, developers familiar with Python may be switching from opinionated and older frameworks like Django to FastAPI. I assume the exceptional developer experience, development freedom, excellent performance, and recent AI model-serving support via lifecycle events may be contributing to this.

This book covers the implementation details of developing generative AI services that can autonomously perform actions and interact with external services, all powered by the FastAPI web framework.

To learn the relevant concepts, I will be guiding you through a capstone project that you can work on as you read through the book. Let’s take a look.

# What Prevents the Adoption of Generative AI Services

Organizations face several challenges when adopting generative AI services. There are issues related to the inaccuracy, relevance, quality, and consistency of GenAI outputs. In addition, there are concerns about data privacy, cybersecurity, and potential abuse and misuse of the models if used in production. As a result, companies don’t want to give full autonomy to these models yet. There is a hesitation in connecting them directly with sensitive systems like internal databases or payment systems.

Integrating the AI service with existing systems, such as internal databases, web interfaces, and external APIs, can pose a challenge. This integration can be difficult due to compatibility issues, the need for technical expertise, potential disruption of existing processes, malicious attempts on these systems, and similar concerns about data security and privacy.

Companies that want to use the service for customer-facing applications would want consistency and relevance in the model’s responses and to ensure outputs are not offensive or inappropriate.

There are also limitations to producing original and high-quality content with these generative models. As covered before, these GenAI tools effectively bridge various ideas and concepts together within certain domains. But they can’t produce totally unseen or novel ideas; rather, they recombine and rephrase existing information in a way that appears novel. Furthermore, they follow common patterns during generation, which can be generic, repetitive, and uninspiring to use out of the box. Finally, they may produce plausible-sounding outputs that are entirely incorrect and made up, not based on facts or reality.

###### Note

Cases where GenAI models produce made-up facts and incorrect information are referred to as *hallucinations*.

The tendency of these models to hallucinate prevents their adoption in sensitive use cases that require highly accurate outputs such as medical diagnosis, legal advisory, and automated examinations.

Some challenges—such as data privacy and security issues—can be solved with software engineering best practices, which you will read more about in this book. Solutions to other challenges require optimizing inputs to the models or fine-tuning these models (adjusting their parameters through new examples in a particular use case) to improve the relevance, quality, coherence, and consistency of the outputs.

# Overview of the Capstone Project

In this book, I will lead you through building a generative AI service using FastAPI as the underlying web framework.

The service will:

*   Integrate with multiple models including a language model for text generation and chat, an audio model for text to speech, and a Stable Diffusion model for image generation

*   Generate real-time responses to user queries as text, audio, or image

*   Use the RAG technique to “talk” to uploaded documents using a vector database

*   Scrape the web and communicate with internal databases, external systems, and APIs to gather sufficient information when responding to queries

*   Record conversation histories in a relational database

*   Authenticate users via token-based credentials and GitHub identity login

*   Restrict responses based on user permission via authorization guards

*   Provide sufficient protections against misuse and abuse using guardrails

As the focus of this book is on building API services, you will learn to use the Python Streamlit package and simple HTML for developing UIs. In real-world applications, you may interface your generative AI services with custom UIs built with libraries such as React or frameworks like Next.js for modularity, extensibility, and scalability.

# Summary

In this chapter, you learned about the concept of generative AI and how it can create data across various modalities like text, audio, video, etc., using learned patterns in its training data. You also saw several practical examples and use cases of this technology and why most future applications will be powered by GenAI capabilities.

You also learned how GenAI can facilitate the creative process, eliminate intermediaries, personalize user experiences, and democratize access to complex systems and content generation. Further, you were introduced to several challenges preventing widespread adoption of GenAI alongside several solutions. Finally, you learned more about the GenAI API service that you will build with the FastAPI web framework as you follow the code examples in this book.

In the next chapter, you will learn about FastAPI, which will enable you to implement your own GenAI services.

^([1](ch01.html#id448-marker)) You can learn more about these models in [*Generative Deep Learning*](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/) by David Foster (O’Reilly, 2024).

^([2](ch01.html#id470-marker)) Recent GenAI tools can now provide source references alongside the solution (e.g., [phind.com](https://phind.com)).

^([3](ch01.html#id472-marker)) Stack Overflow’s [2024 Developer Survey](https://oreil.ly/odPkB) of 65,000 coders found that 72% of developers are favorable toward AI tools, but only 43% trust the accuracy of those tools.