# Preface

I cannot hide my excitement. The 2022–2024 period has been one of the most amazing moments of the modern technology era. Some call it the “iPhone moment” of artificial intelligence, and a lot of people are now discovering the actual potential of AI. But I don’t think that’s all it is. I believe we are entering an exponential phase where all technological advancements move so fast that it is difficult to keep track of them. But that is wonderful. Several years of progress and industry competition in just a few months. What was often thought of as impossible (or even magic) is now a reality…and it’s just getting started.

That sense of innovation and complete disruption is how I felt the first time I tried the Azure OpenAI Studio. I got early access as an AI Specialist at Microsoft. It was a very early version, and definitively not the Studio and related features and models we have today, but it was very promising. Little did we know that this cloud-enabled service was about to become the superstar of the generative AI era. And it was a reality, not a concept or a future product. It was something we could use to create our very own GPT-style implementations, with different models and cost/performance trade-offs, but also with relatively low implementation and deployment complexity.

After a few months of testing and tracking new functionalities, OpenAI released ChatGPT. Boom. I have never seen such a viral moment related to AI technologies. Even at Microsoft, the feeling of being witnesses to something extraordinary was there, every day and evening, during countless “nerd” discussions with my colleagues. The key moment was the announcement of the “chat” functionality in Azure OpenAI Service, which allowed any company to test and deploy a ChatGPT-ish kind of instance for their own purposes for the first time. Then came Bing Chat (which would evolve into what we today call Microsoft Copilot). Boom ×2\. That was the first time we saw the combination of classic search engines with a GPT chat experience, on the same screen…and it worked! People could get a direct answer to their precise needs instead of searching for information using keywords and having to determine the right answer themselves, and they were doing it with plain language. Not keywords, not complex combinations of words. Just asking for information and waiting for an answer.

Months passed and we started to deploy the first proof-of-concepts with Azure OpenAI. I’m part of a field team, so I was very close to the reality of the adopters—their understanding of what generative AI is, their envisioned use cases, their concerns, etc. I was also part of the AI community at Microsoft, which had plenty of energy and creativity to explore new approaches, discover new architectures, and learn about the most recent techniques and accelerators. Trust me, I wasn’t the only one feeling lucky in those moments. This was pure energy.

At one point, and I assume this was due to my academic background as a university professor, I felt like the vast amount of information—while very useful for any learner or adopter—was also a bit overwhelming for any company or individual trying to get started with generative AI and the Azure OpenAI Service. There was a lot of demand from companies around the world, and this technology deserved to be massively adopted, in a safe, responsible manner.

It was then that I started drafting the main concepts of a technical guide for application development with Azure OpenAI Service. Initially, it was just a way to keep track of all the URLs and pieces of information I was continuously collecting. Then, I continued adding my notes, based on my own implementation experiences. Finally, I kept changing or adding content based on the recurrent questions and discussions I was getting from clients, friends, and even family!

This was a great baseline, and I knew it could become an official technical guide, or even a full book. I decided to talk to my O’Reilly colleagues and present the topic. These conversations took only a matter of weeks. The potential was clear, but the challenge was huge: creating high-quality O’Reilly-level content, in a timely manner (as soon as possible) so all generative AI adopters could start reading and learning.

This has been one of my most challenging, but still rewarding, experiences. I feel really honored to write this book. So many Microsoft folks around the world could have done it, and for this reason, I took the opportunity very seriously. My main goal was to create something that would include all critical elements for Azure OpenAI learning, keeping in mind the (constantly) evolving context—showing the best features and implementation approaches, but knowing that there will be others soon and a continuously changing mix of generative availability and new features in preview. But that’s part of the charm, and the reason why I like this book and the creative process behind it so much.

One of my favorite things (and I hope you like it too) is the combination of the typical static content of a book, with the interactivity of online repositories, references to evolving documentation…and the incredible power of the guest interviews. Having such an amazing amount of talent and knowledge from a roster of AI pros is an authentic luxury, for you as readers and avid learners, but also for me as an AI professional.

Now, I hope that if you have decided to start reading this book, you are ready to explore each piece, from the core technical aspects to the other relevant business and ethical aspects that will help you during your first generative AI projects with Azure OpenAI Service.

Truly yours,

Adrián

# How This Book Is Organized

The content of this book is organized in a way that follows the typical adoption workstreams for new technologies: initial understanding of their potential, exploration of technical implementations, considerations for operationalization, and business requirements. Depending on the company and its level of maturity, the sequence of things may change. For example, experienced AI teams will have a clearer understanding of the business aspects and future operationalization, then consolidate the technical part. Regardless of you and your company’s context, the seven book chapters (plus an appendix) should cover what you need to leverage Azure OpenAI Service for your generative AI implementations:

[Chapter 1, “Introduction to Generative AI and Azure OpenAI Service”](ch01.html#introduction_to_generative_ai_and_azure_openai_ser)

A 101 overview of AI, generative AI, and the role of Azure OpenAI Service for enterprise-grade implementations. Ideal if you are starting your generative AI journey from scratch.

[Chapter 2, “Designing Cloud Native Architectures for Generative AI”](ch02.html#designing_cloud_native_architectures_for_generativ)

A top-down approach for architecting generative AI applications based on cloud native principles, with the most relevant building blocks, including those from the Microsoft Azure cloud. The key preliminary step before you start exploring Azure OpenAI Service.

[Chapter 3, “Implementing Cloud Native Generative AI with Azure OpenAI Service”](ch03.html#implementing_cloud_native_generative_ai_with_azure)

The core chapter for you to explore the different Azure OpenAI interfaces, including visual playgrounds and APIs, as well as the main implementation approaches and patterns.

[Chapter 4, “Additional Cloud and AI Capabilities”](ch04.html#additional_cloud_and_ai_capabilities)

The perfect complement to the third chapter. The place to go if you want to learn about all related “pieces,” such as vector databases, orchestration engines, and other Azure-related services.

[Chapter 5, “Operationalizing Generative AI Implementations”](ch05.html#operationalizing_generative_ai_implementations)

The most important chapter, from my point of view, if you want to understand what a “real life” generative AI implementation means. You can design a wonderful architecture and make the most of Azure OpenAI and other services, but it is important to implement all required measures to secure, scale, protect, and optimize your deployments. A must if you are creating generative AI applications for a company.

[Chapter 6, “Elaborating Generative AI Business Cases”](ch06.html#elaborating_generative_ai_business_cases)

Even if you master every single technical aspect related to your generative AI apps with Azure OpenAI, you still need to make it work from a business point of view. This means creating sustainable business cases, supported by realistic cost estimations and project roadmaps. By the end of the day, no AI system will be adopted if these topics are not discussed up front.

[Chapter 7, “Exploring the Big Picture”](ch07.html#exploring_the_big_picture)

An overview of the future state of generative AI systems with Microsoft technology, along with interviews with some of the industry’s top talent to give you key insights from the people on the ground.

[Appendix A, “Other Learning Resources”](app01.html#appendix_other_learning_resources)

A collection of resources for you to expand your learning experience.

My goal is for this collection of chapters to give you a 360-degree view of what generative AI implementations mean today, which will enable you to start your new projects with all the required knowledge.

# Conventions Used in This Book

The following typographical conventions are used in this book:

*Italic*

Indicates new terms, URLs, email addresses, filenames, and file extensions.

`Constant width`

Used for program listings, as well as within paragraphs to refer to program elements such as variable or function names, databases, data types, environment variables, statements, and keywords.

**`Constant width bold`**

Used to call attention to snippets of interest in code blocks.

###### Note

This element signifies a general note.

###### Warning

This element indicates a warning or caution.

# Using Code Examples

Supplemental material (code examples, exercises, etc.) is available for download at [*https://oreil.ly/azure-openai-service-code*](https://oreil.ly/azure-openai-service-code).

If you have a technical question or a problem using the code examples, please send email to [*support@oreilly.com*](mailto:support@oreilly.com).

This book is here to help you get your job done. In general, if example code is offered with this book, you may use it in your programs and documentation. You do not need to contact us for permission unless you’re reproducing a significant portion of the code. For example, writing a program that uses several chunks of code from this book does not require permission. Selling or distributing examples from O’Reilly books does require permission. Answering a question by citing this book and quoting example code does not require permission. Incorporating a significant amount of example code from this book into your product’s documentation does require permission.

We appreciate, but generally do not require, attribution. An attribution usually includes the title, author, publisher, and ISBN. For example: “*Azure OpenAI Service for Cloud Native Applications* by Adrián González Sánchez (O’Reilly). Copyright 2024 Adrián González Sánchez, 978-1-098-15499-8.”

If you feel your use of code examples falls outside fair use or the permission given above, feel free to contact us at [*permissions@oreilly.com*](mailto:permissions@oreilly.com).

# O’Reilly Online Learning

###### Note

For more than 40 years, [*O’Reilly Media*](https://oreilly.com) has provided technology and business training, knowledge, and insight to help companies succeed.

Our unique network of experts and innovators share their knowledge and expertise through books, articles, and our online learning platform. O’Reilly’s online learning platform gives you on-demand access to live training courses, in-depth learning paths, interactive coding environments, and a vast collection of text and video from O’Reilly and 200+ other publishers. For more information, visit [*https://oreilly.com*](https://oreilly.com).

# How to Contact Us

Please address comments and questions concerning this book to the publisher:

*   O’Reilly Media, Inc.
*   1005 Gravenstein Highway North
*   Sebastopol, CA 95472
*   800-889-8969 (in the United States or Canada)
*   707-827-7019 (international or local)
*   707-829-0104 (fax)
*   [*support@oreilly.com*](mailto:support@oreilly.com)
*   [*https://www.oreilly.com/about/contact.html*](https://www.oreilly.com/about/contact.html)

We have a web page for this book, where we list errata, examples, and any additional information. You can access this page at [*https://oreil.ly/azure-openai*](https://oreil.ly/azure-openai).

For news and information about our books and courses, visit [*https://oreilly.com*](https://oreilly.com).

Find us on LinkedIn: [*https://linkedin.com/company/oreilly-media*](https://linkedin.com/company/oreilly-media)

Watch us on YouTube: [*https://youtube.com/oreillymedia*](https://youtube.com/oreillymedia)

# Acknowledgments

Thanks to all involved “stakeholders” for helping me to make this happen. My wife Malini, the amazing O’Reilly team for their methodology (and their unlimited doses of patience and support, especially Melissa), my Microsoft colleagues for being a continuous source of inspiration (including my boss, Agustin, and his unwavering support), all technical reviewers and interviewees for their wealth of knowledge (many thanks to Jonah Anderson, Sergio Gonzalez, and Jorge Garcia Ximenez), and so many learners around the world showing their interest in this amazing topic. This book is for all of you. Please enjoy it.