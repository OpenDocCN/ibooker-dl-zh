# Preface

In the past few years, progress in the field of artificial intelligence has been occurring at breakneck speeds, spearheaded by advances in LLMs. It was not too long ago that LLMs were a nascent technology that struggled to generate a coherent paragraph; today they are able to solve complex mathematical problems, write convincing essays, and conduct long engaging conversations with humans.

As AI advances from strength to strength, it is rapidly being woven into the fabric of society, touching so many facets of our lives. Learning how to use AI models like LLMs effectively might be one of the most useful skills to learn this decade. LLMs are revolutionizing the world of software, and have made possible the development of applications previously considered impossible.

With all the promise that LLMs bring, the reality is that they are still not a mature technology and have many limitations like deficiencies in reasoning, lack of adherence to factuality, “hallucinations”, difficulties in steering them toward our goals, bias and fairness issues, and so on. Despite the existence of these limitations, we can still harness LLMs for good use and build a variety of helpful applications provided we effectively address their shortcomings.

Plenty of software frameworks have emerged that enable rapid prototype development of LLM applications. However, advancing from prototypes to production-grade applications is a road much less traveled, and is still a very challenging task. This is where this book comes in—a holistic overview of the LLM landscape that provides you with the intuition and tools to build complex LLM applications.

With this book, my goal is to provide you with an intuitive understanding of how LLMs work, the tools you have at your disposal to harness them, and the various application paradigms they can be built with. Unique to this book are the exercises; more than 80 exercises are sprinkled throughout to help you solidify your intuitions and sharpen your understanding of what is happening underneath the hood. While preparing the content of the book, I read over 800 research papers, with many of them referenced and linked at appropriate locations in the book, providing you with a jumping off point for further exploration. All in all, I am confident that you will come out of the book an LLM expert if you read the book in its entirety, complete all the exercises, and explore the recommended references.

# Who This Book Is For

This book is intended for a broad audience, including software engineers transitioning to AI application development, machine learning practitioners and scientists, and product managers. Much of the content in this book is borne from my own experiments with LLMs, so even if you are an experienced scientist, I expect you will find value in it. Similarly, even if you have very limited exposure to the world of AI, I expect you will still find the book useful for understanding the fundamentals of this technology.

The only prerequisites for this book are knowledge of Python coding and an understanding of basic machine learning and deep learning principles. Where required, I provide links to external resources that you can use to sharpen or develop your prerequisites.

# How This Book Is Structured

The book is divided into 3 parts with a total of 13 chapters. The first part deals with understanding the ingredients of a language model. I strongly feel that even though you may never train a language model from scratch yourself, knowing what goes into making it is crucial. The second part discusses various ways to harness language models, be it by directly prompting the model, or by fine-tuning it in various ways. It also addresses limitations such as hallucinations and reasoning constraints, along with methods to mitigate these issues. Finally, the third part of the book deals with application paradigms like retrieval augmented generation (RAG) and agents, positioning LLMs within the broader context of an entire software system.

For an extended table of contents, see my [Substack blog post](https://oreil.ly/-2zkH).

# What This Book Is Not About

To keep the book at a reasonable length, certain topics were deemed out of scope. I have taken care to not cover topics that I am not confident will stand the test of time. This field is very fast moving, so writing a book that maintains its relevance over time is extremely challenging.

This book focuses only on English-language LLMs and leaves out discussion on multilingual models for the most part. I also disagree with the notion of mushing all the non-English languages of the world under the “multilingual” banner. Every language has its own nuances and deserves its own book.

This book also doesn’t cover multimodal models. New models are increasingly multimodal, i.e., a single model supports multiple modalities like text, image, video, speech, etc. However, text remains the most important modality and is the binding substrate in these models. Thus, reading this book will still help you prepare for the multimodal future.

This book does not focus on theory or go too deep into math. There are plenty of other books that cover that, and I have generously linked to them where needed. This book contains minimal math equations and instead focuses on building intuitions.

This book contains only a rudimentary introduction to reasoning models, the latest LLM paradigm. At the time of the book’s writing, reasoning models are still in their infancy, and the jury is still out on which techniques will prove to be most effective.

# How to Read the Book

The best way to consume this book is to read it sequentially, while working on the exercises and exploring the reference links. That said, there are a few alternative paths, depending on your interests:

*   If your interest lies in understanding the LLM landscape and not necessarily in building applications with them, you can focus on Chapters [1](ch01.html#chapter_llm-introduction), [2](ch02.html#ch02), [3](ch03.html#chapter-LLM-tokenization), [4](ch04.html#chapter_transformer-architecture), [5](ch05.html#chapter_utilizing_llms), [10](ch10.html#ch10), and [11](ch11.html#chapter_llm_interfaces).

*   If you are a product manager seeking to understand the scope of possibilities for LLM applications, Chapters [1](ch01.html#chapter_llm-introduction), [2](ch02.html#ch02), [3](ch03.html#chapter-LLM-tokenization), [5](ch05.html#chapter_utilizing_llms), [8](ch08.html#ch8), [10](ch10.html#ch10), [11](ch11.html#chapter_llm_interfaces), [12](ch12.html#ch12), and [13](ch13.html#ch13) are a good bet.

*   If you are an ML scientist, then Chapters [7](ch07.html#ch07), [8](ch08.html#ch8), [9](ch09.html#ch09), [10](ch10.html#ch10), [11](ch11.html#chapter_llm_interfaces), and [12](ch12.html#ch12) will be sure to give you food-for-thought and new research challenges.

*   If you want to train your own LLM from scratch, Chapters [2](ch02.html#ch02), [3](ch03.html#chapter-LLM-tokenization), [4](ch04.html#chapter_transformer-architecture), [5](ch05.html#chapter_utilizing_llms), and [7](ch07.html#ch07) will provide you with the foundational principles.

# Conventions Used in This Book

The following typographical conventions are used in this book:

*Italic*

Indicates new terms, URLs, email addresses, filenames, and file extensions.

`Constant width`

Used for program listings, as well as within paragraphs to refer to program elements such as variable or function names, databases, data types, environment variables, statements, and keywords.

**`Constant width bold`**

Shows commands or other text that should be typed literally by the user.

*`Constant width italic`*

Shows text that should be replaced with user-supplied values or by values determined by context.

###### Tip

This element signifies a tip or suggestion.

###### Note

This element signifies a general note.

###### Warning

This element indicates a warning or caution.

# Using Code Examples

Supplemental material (code examples, exercises, etc.) is available for download at [*https://oreil.ly/llm-playbooks*](https://oreil.ly/llm-playbooks).

If you have a technical question or a problem using the code examples, please send email to [*support@oreilly.com*](mailto:support@oreilly.com).

This book is here to help you get your job done. In general, if example code is offered with this book, you may use it in your programs and documentation. You do not need to contact us for permission unless you’re reproducing a significant portion of the code. For example, writing a program that uses several chunks of code from this book does not require permission. Selling or distributing examples from O’Reilly books does require permission. Answering a question by citing this book and quoting example code does not require permission. Incorporating a significant amount of example code from this book into your product’s documentation does require permission.

We appreciate, but generally do not require, attribution. An attribution usually includes the title, author, publisher, and ISBN. For example: “*Designing Large Language Model Applications* by Suhas Pai (O’Reilly). Copyright 2025 Suhas Pai, 978-1-098-15050-1.”

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
*   [*https://oreilly.com/about/contact.html*](https://oreilly.com/about/contact.html)

We have a web page for this book, where we list errata, examples, and any additional information. You can access this page at [*https://oreil.ly/designing-llm-applications-1e*](https://oreil.ly/designing-llm-applications-1e).

For news and information about our books and courses, visit [*https://oreilly.com*](https://oreilly.com).

Find us on LinkedIn: [*https://linkedin.com/company/oreilly-media*](https://linkedin.com/company/oreilly-media).

Watch us on YouTube: [*https://youtube.com/oreillymedia*](https://youtube.com/oreillymedia).

# Acknowledgments

They say it takes a village to raise a child; I now realize it takes a metropolis to write a book.

Firstly, I would like to thank the O’Reilly team for the meticulous professionalism and finesse with which they worked with me throughout the development and launch of the book. No wonder they are the world’s top technical book publishers. I would particularly like to thank Nicole Butterfield for signing me up as an author and Michele Cronin, the world’s best editor, whose frequent reviews ensured that the book developed a coherent structure. I will miss our regular check-ins! Thanks to Ashley Stussy, Kristen Brown, and the rest of the production team for their diligent work in getting the book to production.

I am deeply thankful to my friend Amber Teng, who helped me with drawing the book illustrations and setting up the book’s Github repository. I am also immensely indebted to my technical reviewers Serena McDonnell, Yenson Lau, Susan Shu Chang, Gordon Gibson, and Nour Fahmy for the dozens of hours each of them spent in writing extremely detailed and thoughtful technical reviews. The book is so much better for it.

I am thankful to the Toronto AI ecosystem, especially the Aggregate Intellect, TMLS (Toronto Machine Learning Summit), and SharpestMinds communities for providing me with the space to engage with the community and ensure that I always had a finger on the pulse of the industry. Special thanks go to my friends Madhav Singhal, Jay Alammar, and Megan Risdal (who helped me coin the phrase “token etymology”) for our regular intellectually stimulating conversations on LLMs and for being the first readers of the book. I also want to give a shout out to my open-source collaborator Huu Nguyen, who I worked with on various open-source LLM projects, for the dozens of late night discussions on the most audacious ideas in LLM research.

Writing a book while also being the cofounder of an AI startup was possible only due to the unwavering support of my partner in business and crime, Kris Bennatti (who also convinced me to remove the word “orifice” from the book). I will forever be in gratitude to the entire Hudson Labs team for their steadfast and consistent backing throughout, with a special shout out to Xiao Quan, whose steady hands ensured that I found the time to focus on the book. Additionally, I would like to thank my friends Kaaveh Shoamanesh, Abdullah Al-hayali, Zach Nguyen, Samarth Bhasin, Sadegh Raeisi, and Ian Yu for their moral support throughout and regularly checking that I was getting the right amount of sleep.

Finally, I would like to dedicate this book to my mom, Kusuma Pai, whom I simply refer to as “The Legend” for her lifelong sacrifices to ensure that I grew up and was in a position to write the book. Any success of this book should be predominantly credited to my mother for molding me into the person I am today.