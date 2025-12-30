# Preface

On November 30, 2022, San Francisco–based firm OpenAI [publicly released ChatGPT](https://oreil.ly/uAnsr)—the viral AI chatbot that can generate content, answer questions, and solve problems like a human. Within two months of its launch, ChatGPT attracted over [100 million monthly active users](https://oreil.ly/ATsLe), the fastest adoption rate of a new consumer technology application (so far). ChatGPT is a chatbot experience powered by an instruction and dialogue-tuned version of OpenAI’s GPT-3.5 family of large language models (LLMs). We’ll get to definitions of these concepts very shortly.

###### Note

Building LLM applications with or without LangChain requires the use of an LLM. In this book we will be making use of the OpenAI API as the LLM provider we use in the code examples (pricing is listed on its [platform](https://oreil.ly/-YYoR)). One of the benefits of working with LangChain is that you can follow along with all of these examples using either OpenAI or alternative commercial or open source LLM providers.

Three months later, OpenAI [released the ChatGPT API](https://oreil.ly/DwU7R), giving developers access to the chat and speech-to-text capabilities. This kickstarted an uncountable number of new applications and technical developments under the loose umbrella term of *generative AI*.

Before we define generative AI and LLMs, let’s touch on the concept of *machine learning* (ML). Some computer *algorithms* (imagine a repeatable recipe for achievement of some predefined task, such as sorting a deck of cards) are directly written by a software engineer. Other computer algorithms are instead *learned* from vast amounts of training examples—the job of the software engineer shifts from writing the algorithm itself to writing the training logic that creates the algorithm. A lot of attention in the ML field went into developing algorithms for predicting any number of things, from tomorrow’s weather to the most efficient delivery route for an Amazon driver.

With the advent of LLMs and other generative models (such as diffusion models for generating images, which we don’t cover in this book), those same ML techniques are now applied to the problem of generating new content, such as a new paragraph of text or drawing, that is at the same time unique and informed by examples in the training data. LLMs in particular are generative models dedicated to generating text.

LLMs have two other differences from previous ML algorithms:

*   They are trained on much larger amounts of data; training one of these models from scratch would be very costly.

*   They are more versatile.

The same text generation model can be used for summarization, translation, classification, and so forth, whereas previous ML models were usually trained and used for a specific task.

These two differences conspire to make the job of the software engineer shift once more, with increasing amounts of time dedicated to working out how to get an LLM to work for their use case. And that’s what LangChain is all about.

By the end of 2023, competing LLMs emerged, including Anthropic’s Claude and Google’s Bard (later renamed Gemini), providing even wider access to these new capabilities. And subsequently, thousands of successful startups and major enterprises have incorporated generative AI APIs to build applications for various use cases, ranging from customer support chatbots to writing and debugging code.

On October 22, 2022, Harrison Chase [published the first commit](https://oreil.ly/mCdYZ) on GitHub for the LangChain open source library. LangChain started from the realization that the most interesting LLM applications needed to use LLMs together with [“other sources of computation or knowledge”](https://oreil.ly/uXiPi). For instance, you can try to get an LLM to generate the answer to this question:

```py
How many balls are left after splitting 1,234 balls evenly among 123 people?
```

You’ll likely be disappointed by its math prowess. However, if you pair it up with a calculator function, you can instead instruct the LLM to reword the question into an input that a calculator could handle:

```py
1,234 % 123
```

Then you can pass that to a calculator function and get an accurate answer to your original question. LangChain was the first (and, at the time of writing, the largest) library to provide such building blocks and the tooling to reliably combine them into larger applications. Before discussing what it takes to build compelling applications with these new tools, let’s get more familiar with LLMs and LangChain.

# Brief Primer on LLMs

In layman’s terms, LLMs are trained algorithms that receive text input and predict and generate humanlike text output. Essentially, they behave like the familiar autocomplete feature found on many smartphones, but taken to an extreme.

Let’s break down the term *large language model*:

*   *Large* refers to the size of these models in terms of training data and parameters used during the learning process. For example, OpenAI’s GPT-3 model contains 175 billion *parameters*, which were learned from training on 45 terabytes of text data.^([1](preface01.html#id259)) *Parameters* in a neural network model are made up of the numbers that control the output of each *neuron* and the relative weight of its connections with its neighboring neurons. (Exactly which neurons are connected to which other neurons varies for each neural network architecture and is beyond the scope of this book.)

*   *Language model* refers to a computer algorithm trained to receive written text (in English or other languages) and produce output also as written text (in the same language or a different one). These are *neural networks*, a type of ML model which resembles a stylized conception of the human brain, with the final output resulting from the combination of the individual outputs of many simple mathematical functions, called *neurons*, and their interconnections. If many of these neurons are organized in specific ways, with the right training process and the right training data, this produces a model that is capable of interpreting the meaning of individual words and sentences, which makes it possible to use them for generating plausible, readable, written text.

Because of the prevalence of English in the training data, most models are better at English than they are at other languages with a smaller number of speakers. By “better” we mean it is easier to get them to produce desired outputs in English. There are LLMs designed for multilingual output, such as [BLOOM](https://oreil.ly/Nq7w0), that use a larger proportion of training data in other languages. Curiously, the difference in performance between languages isn’t as large as might be expected, even in LLMs trained on a predominantly English training corpus. Researchers have found that LLMs are able to transfer some of their semantic understanding to other languages.^([2](preface01.html#id267))

Put together, *large language models* are instances of big, general-purpose language models that are trained on vast amounts of text. In other words, these models have learned from patterns in large datasets of text—books, articles, forums, and other publicly available sources—to perform general text-related tasks. These tasks include text generation, summarization, translation, classification, and more.

Let’s say we instruct an LLM to complete the following sentence:

```py
The capital of England is _______.
```

The LLM will take that input text and predict the correct output answer as `London`. This looks like magic, but it’s not. Under the hood, the LLM estimates the probability of a sequence of word(s) given a previous sequence of words.

###### Tip

Technically speaking, the model makes predictions based on tokens, not words. A *token* represents an atomic unit of text. Tokens can represent individual characters, words, subwords, or even larger linguistic units, depending on the specific tokenization approach used. For example, using GPT-3.5’s tokenizer (called `cl100k`), the phrase *good morning dearest friend* would consist of [five tokens](https://oreil.ly/dU83b) (using `_` to show the space character):

`Good`

With token ID `19045`

`_morning`

With token ID `6693`

`_de`

With token ID `409`

`arest`

With token ID `15795`

`_friend`

With token ID `4333`

Usually tokenizers are trained with the objective of having the most common words encoded into a single token, for example, the word *morning* is encoded as the token `6693`. Less common words, or words in other languages (usually tokenizers are trained on English text), require several tokens to encode them. For example, the word *dearest* is encoded as tokens `409, 15795`. One token spans on average four characters of text for common English text, or roughly three quarters of a word.

The driving engine behind LLMs’ predictive power is known as the *transformer neural network architecture*.^([3](preface01.html#id271)) The transformer architecture enables models to handle sequences of data, such as sentences or lines of code, and make predictions about the likeliest next word(s) in the sequence. Transformers are designed to understand the context of each word in a sentence by considering it in relation to every other word. This allows the model to build a comprehensive understanding of the meaning of a sentence, paragraph, and so on (in other words, a sequence of words) as the joint meaning of its parts in relation to each other.

So, when the model sees the sequence of words *the capital of England is*, it makes a prediction based on similar examples it saw during its training. In the model’s training corpus the word *England* (or the token(s) that represent it) would have often shown up in sentences in similar places to words like *France*, *United States*, *China*. The word *capital* would figure in the training data in many sentences also containing words like *England*, *France*, and *US*, and words like *London*, *Paris*, *Washington*. This repetition during the model’s training resulted in the capacity to correctly predict that the next word in the sequence should be *London*.

The instructions and input text you provide to the model is called a *prompt*. Prompting can have a significant impact on the quality of output from the LLM. There are several best practices for *prompt design* or *prompt engineering*, including providing clear and concise instructions with contextual examples, which we discuss later in this book. Before we go further into prompting, let’s look at some different types of LLMs available for you to use.

The base type, from which all the others derive, is commonly known as a *pretrained LLM*: it has been trained on very large amounts of text (found on the internet and in books, newspapers, code, video transcripts, and so forth) in a self-supervised fashion. This means that—unlike in supervised ML, where prior to training the researcher needs to assemble a dataset of pairs of *input* to *expected output*—for LLMs those pairs are inferred from the training data. In fact, the only feasible way to use datasets that are so large is to assemble those pairs from the training data automatically. Two techniques to do this involve having the model do the following:

Predict the next word

Remove the last word from each sentence in the training data, and that yields a pair of *input* and *expected output*, such as *The capital of England is ___* and *London*.

Predict a missing word

Similarly, if you take each sentence and omit a word from the middle, you now have other pairs of input and expected output, such as *The ___ of England is London* and *capital*.

These models are quite difficult to use as is, they require you to prime the response with a suitable prefix. For instance, if you want to know the capital of England, you might get a response by prompting the model with *The capital of England is*, but not with the more natural *What is the capital of England?*

## Instruction-Tuned LLMs

[Researchers](https://oreil.ly/lP6hr) have made pretrained LLMs easier to use by further training (additional training applied on top of the long and costly training described in the previous section), also known as *fine-tuning* them on the following:

Task-specific datasets

These are datasets of pairs of questions/answers manually assembled by researchers, providing examples of desirable responses to common questions that end users might prompt the model with. For example, the dataset might contain the following pair: *Q: What is the capital of England? A: The capital of England is London.* Unlike the pretraining datasets, these are manually assembled, so they are by necessity much smaller:

Reinforcement learning from human feedback (RLHF)

Through the use of [RLHF methods](https://oreil.ly/lrlAK), those manually assembled datasets are augmented with user feedback received on output produced by the model. For example, user A preferred *The capital of England is London* to *London is the capital of England* as an answer to the earlier question.

Instruction-tuning has been key to broadening the number of people who can build applications with LLMs, as they can now be prompted with *instructions*, often in the form of questions such as, *What is the capital of England?*, as opposed to *The capital of England is*.

## Dialogue-Tuned LLMs

Models tailored for dialogue or chat purposes are a [further enhancement](https://oreil.ly/1DxW6) of instruction-tuned LLMs. Different providers of LLMs use different techniques, so this is not necessarily true of all *chat models*, but usually this is done via the following:

Dialogue datasets

The manually assembled *fine-tuning* datasets are extended to include more examples of multiturn dialogue interactions, that is, sequences of prompt-reply pairs.

Chat format

The input and output formats of the model are given a layer of structure over freeform text, which divides text into parts associated with a role (and optionally other metadata like a name). Usually, the roles available are *system* (for instructions and framing of the task), *user* (the actual task or question), and *assistant* (for the outputs of the model). This method evolved from early [prompt engineering techniques](https://oreil.ly/dINx0) and makes it easier to tailor the model’s output while making it harder for models to confuse user input with instructions. Confusing user input with prior instructions is also known as *jailbreaking*, which can, for instance, lead to carefully crafted prompts, possibly including trade secrets, being exposed to end users.

## Fine-Tuned LLMs

Fine-tuned LLMs are created by taking base LLMs and further training them on a proprietary dataset for a specific task. Technically, instruction-tuned and dialogue-tuned LLMs are fine-tuned LLMs, but the term “fine-tuned LLM” is usually used to describe LLMs that are tuned by the developer for their specific task. For example, a model can be fine-tuned to accurately extract the sentiment, risk factors, and key financial figures from a public company’s annual report. Usually, fine-tuned models have improved performance on the chosen task at the expense of a loss of generality. That is, they become less capable of answering queries on unrelated tasks.

Throughout the rest of this book, when we use the term *LLM*, we mean instruction-tuned LLMs, and for *chat model* we mean dialogue-instructed LLMs, as defined earlier in this section. These should be your workhorses when using LLMs—the first tools you reach for when starting a new LLM application.

Now let’s quickly discuss some common LLM prompting techniques before diving into LangChain.

# Brief Primer on Prompting

As we touched on earlier, the main task of the software engineer working with LLMs is not to train an LLM, or even to fine-tune one (usually), but rather to take an existing LLM and work out how to get it to accomplish the task you need for your application. There are commercial providers of LLMs, like OpenAI, Anthropic, and Google, as well as open source LLMs ([Llama](https://oreil.ly/ld3Fu), [Gemma](https://oreil.ly/RGKfi), and others), released free-of-charge for others to build upon. Adapting an existing LLM for your task is called *prompt engineering*.

Many prompting techniques have been developed in the past two years, and in a broad sense, this is a book about how to do prompt engineering with LangChain—how to use LangChain to get LLMs to do what you have in mind. But before we get into LangChain proper, it helps to go over some of these techniques first (and we apologize in advance if your favorite [prompting technique](https://oreil.ly/8uGK_) isn’t listed here; there are too many to cover).

To follow along with this section we recommend copying these prompts to the OpenAI Playground to try them yourself:

1.  Create an account for the OpenAI API at [*http://platform.openai.com*](http://platform.openai.com), which will let you use OpenAI LLMs programmatically, that is, using the API from your Python or JavaScript code. It will also give you access to the OpenAI Playground, where you can experiment with prompts from your web browser.

2.  If necessary, add payment details for your new OpenAI account. OpenAI is a commercial provider of LLMs and charges a fee for each time you use their models through OpenAI’s API or through Playground. You can find the latest pricing on their [website](https://oreil.ly/MiKRD). Over the past two years, the price for using OpenAI’s models has come down significantly as new capabilities and optimizations are introduced.

3.  Head on over to the [OpenAI Playground](https://oreil.ly/rxiAG) and you’re ready to try out the following prompts for yourself. We’ll make use of the OpenAI API throughout this book.

4.  Once you’ve navigated to the Playground, you will see a panel of presets on the right side of the screen, including your model of choice. If you look further down the panel, you will see Temperature under the “Model configuration” title. Move the Temperature toggle from middle to left until the number shows 0.00\. Essentially, temperature controls the randomness of LLM output. The lower the temperature, the more deterministic the model output.

Now on to the prompts!

## Zero-Shot Prompting

The first and most straightforward prompting technique consists of simply instructing the LLM to perform the desired task:

```py
How old was the 30th president of the United States when his wife's mother died?
```

This is typically what you should try first, and it will usually work for simple questions, especially when the answer to it was likely present in some of the training data. If we prompt OpenAI’s `gpt-3.5-turbo` with the preceding prompt, the following is returned:

```py
The 30th president of the United States, Calvin Coolidge, was 48 years old when
his wife's mother passed away in 1926.
```

###### Note

You may get a different result from what we get. There is an element of randomness to how LLMs generate responses, and OpenAI may have updated the model by the time you try it.

While the model did correctly identify the 30th president, the answer isn’t quite right. Often you’ll have to iterate on prompts and responses to get a reliable system. The next sections expand on how.

## Chain-of-Thought

A very useful iteration is to further instruct the model to [take the time to *think*](https://oreil.ly/8xWcs). This technique has been found to increase performance on a variety of tasks. Curiously, a more recent paper^([4](preface01.html#id304)) suggests that this technique can also reduce LLM performance on tasks where thinking reduces human performance. Called chain-of-thought (CoT) prompting, this is usually done by prepending the prompt with instructions for the LLM to describe how it could arrive at the answer:

```py
Think step by step.

How old was the 30th president of the United States when his wife's mother died?
```

And now let’s see what the same model produces for this new prompt:

```py
1\. First, find out who the 30th president of the United States was. This can be
done by researching a list of US presidents or by searching for "30th president 
of the United States" online. You will find that the 30th president was Calvin 
Coolidge.

2\. Next, find out the year when Calvin Coolidge was the president. This can also 
be found by researching online or in a history book. You will find that he was 
the president from 1923-1929.

3\. Now, you need to determine the year when Calvin Coolidge's wife's mother died. 
This can be done by researching the life and death of his wife, Grace Goodhue 
Coolidge. You will find that she lived from 1879-1957.

4\. Calculate the age of Calvin Coolidge's wife's mother when she died. She was 
78 years old.

5\. Determine the age of Calvin Coolidge when his wife's mother died. Subtract 
the year of his birth, 1872, from the year his mother-in-law died, 1957\. This 
means that Calvin Coolidge was 85 years old when his wife's mother passed away.
```

Notice how the only change in the prompt was the addition of `Think step by step`, and how much of a difference that made in the model’s output. Now the model outputs a sequence of steps you could go through when trying to answer this question. Unfortunately, while the steps are very reasonable, the final output is still incorrect, given it got some facts wrong, namely it listed the wrong birth and death years for President Coolidge’s wife’s mother. Let’s see how to improve on this.

## Retrieval-Augmented Generation

*Retrieval-augmented generation* (RAG) consists of finding relevant pieces of text, also known as *context*, such as facts you’d find in an encyclopedia and including that context in the prompt. The RAG technique can (and in real applications should) be combined with CoT, but for simplicity we’ll use these techniques one at a time here. Here’s the prompt including RAG:

```py
Context:

- Calvin Coolidge (born John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; July 4, 1872 – 
January 5, 1933) was an American attorney and politician who served as the 
30th president of the United States from 1923 to 1929.

- Grace Anna Coolidge (née Goodhue; January 3, 1879 – July 8, 1957) was the
wife of the 30th president of the United States, Calvin Coolidge.

- Grace Anna Goodhue was born on January 3, 1879, in Burlington, Vermont, the 
only child of Andrew Issachar Goodhue and Lemira Barrett Goodhue.

- Lemira A. Goodhue (Barrett) ; Birthdate: April 26, 1849 ; Birthplace: 
Burlington, Chittenden County, VT, United States ; Death: October 24, 1929.

How old was the 30th president of the United States when his wife's mother died?
```

And the output from the model:

```py
The 30th president of the United States, Calvin Coolidge, was 54 years old when 
his wife's mother, Lemira A. Goodhue, died on October 24, 1929.
```

Now we’re a lot closer to the correct answer, but as we touched on earlier, LLMs aren’t great at out-of-the-box math. In this case, the final result of 54 years old is off by 3\. Let’s see how we can improve on this.

## Tool Calling

The *tool calling* technique consists of prepending the prompt with a list of external functions the LLM can make use of, along with descriptions of what each is good for and instructions on how to signal in the output that it *wants* to use one (or more) of these functions. Finally, you—the developer of the application—should parse the output and call the appropriate functions. Here’s one way to do this:

```py
Tools:

- calculator: This tool accepts math expressions and returns their result.

- search: This tool accepts search engine queries and returns the first search 
result.

If you want to use tools to arrive at the answer, output the list of tools and
inputs in CSV format, with this header row `tool,input`.

How old was the 30th president of the United States when his wife's mother died?
```

And this is the output you might get:

```py
tool,input

calculator,2023-1892

search,"What age was Calvin Coolidge when his mother-in-law died?"
```

While the LLM correctly followed the output format instructions, the tools and inputs selected aren’t the most appropriate for this question. This gets at one of the most important things to keep in mind when prompting LLMs: *each prompting technique is most useful when used in combination with (some of) the others*. For instance, here we could improve on this by combining tool calling, chain-of-thought, and RAG into a prompt that uses all three. Let’s see what that looks like:

```py
Context:

- Calvin Coolidge (born John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; July 4, 1872 –
January 5, 1933) was an American attorney and politician who served as the 30th
president of the United States from 1923 to 1929.

- Grace Anna Coolidge (née Goodhue; January 3, 1879 – July 8, 1957) was the wife 
of the 30th president of the United States, Calvin Coolidge.

- Grace Anna Goodhue was born on January 3, 1879, in Burlington, Vermont, the 
only child of Andrew Issachar Goodhue and Lemira Barrett Goodhue.

- Lemira A. Goodhue (Barrett) ; Birthdate: April 26, 1849 ; Birthplace: 
Burlington, Chittenden County, VT, United States ; Death: October 24, 1929.

Tools:

- calculator: This tool accepts math expressions and returns their result.

If you want to use tools to arrive at the answer, output the list of tools and 
inputs in CSV format, with this header row `tool,input`.

Think step by step.

How old was the 30th president of the United States when his wife's mother died?
```

And with this prompt, maybe after a few tries, we might get this output:

```py
tool,input

calculator,1929 - 1872
```

If we parse that CSV output, and have a calculator function execute the operation `1929 - 1827`, we finally get the right answer: 57 years.

As per the previous example, by combining RAG with chain-of-thought and tool calling, you can retrieve the most relevant data to ground your model’s output, then guide it step by step to ensure it uses that context effectively.

## Few-Shot Prompting

Finally, we come to another very useful prompting technique: *few-shot prompting*. This consists of providing the LLM with examples of other questions and the correct answers, which enables the LLM to *learn* how to perform a new task without going through additional training or fine-tuning. When compared to fine-tuning, few-shot prompting is more flexible—you can do it on the fly at query time—but less powerful, and you might achieve better performance with fine-tuning. That said, you should usually always try few-shot prompting before fine-tuning:

Static few-shot prompting

The most basic version of few-shot prompting is to assemble a predetermined list of a small number of examples that you include in the prompt.

Dynamic few-shot prompting

If you assemble a dataset of many examples, you can instead pick the most relevant examples for each new query.

The next section covers using LangChain to build applications using LLMs and these prompting techniques.

# LangChain and Why It’s Important

LangChain was one of the earliest open source libraries to provide LLM and prompting building blocks and the tooling to reliably combine them into larger applications. As of writing, LangChain has amassed over [28 million monthly downloads](https://oreil.ly/8OKbf), [99,000 GitHub stars](https://oreil.ly/bF5pc), and the largest developer community in generative AI ([72,000+ strong](https://oreil.ly/PNWL3)). It has enabled software engineers who don’t have an ML background to utilize the power of LLMs to build a variety of apps, ranging from AI chatbots to AI agents that can reason and take action responsibly.

LangChain builds on the idea stressed in the preceding section: that prompting techniques are most useful when used together. To make that easier, LangChain provides simple *abstractions* for each major prompting technique. By abstraction we mean Python and JavaScript functions and classes that encapsulate the ideas of those techniques into easy-to-use wrappers. These abstractions are designed to play well together and to be combined into a larger LLM application.

First of all, LangChain provides integrations with the major LLM providers, both commercial ([OpenAI](https://oreil.ly/TTLXA), [Anthropic](https://oreil.ly/O4UXw), [Google](https://oreil.ly/12g3Z), and more) and open source ([Llama](https://oreil.ly/5WAVi), [Gemma](https://oreil.ly/-40Ne), and others). These integrations share a common interface, making it very easy to try out new LLMs as they’re announced and letting you avoid being locked-in to a single provider. We’ll use these in [Chapter 1](ch01.html#ch01_llm_fundamentals_with_langchain_1736545659776004).

LangChain also provides *prompt template* abstractions, which enable you to reuse prompts more than once, separating static text in the prompt from placeholders that will be different for each time you send it to the LLM to get a completion generated. We’ll talk more about these also in [Chapter 1](ch01.html#ch01_llm_fundamentals_with_langchain_1736545659776004). LangChain prompts can also be stored in the LangChain Hub for sharing with teammates.

LangChain contains many integrations with third-party services (such as Google Sheets, Wolfram Alpha, Zapier, just to name a few) exposed as *tools*, which is a standard interface for functions to be used in the tool-calling technique.

For RAG, LangChain provides integrations with the major *embedding models* (language models designed to output a numeric representation, the *embedding*, of the meaning of a sentence, paragraph, and so on), *vector stores* (databases dedicated to storing embeddings), and *vector indexes* (regular databases with vector-storing capabilities). You’ll learn a lot more about these in Chapters [2](ch02.html#ch02_rag_part_i_indexing_your_data_1736545662500927) and [3](ch03.html#ch03_rag_part_ii_chatting_with_your_data_1736545666793580).

For CoT, LangChain (through the LangGraph library) provides *agent* abstractions that combine chain-of-thought reasoning and tool calling, first popularized by the [ReAct paper](https://oreil.ly/27BIC). This enables building LLM applications that do the following:

1.  Reason about the steps to take.

2.  Translate those steps into external tool calls.

3.  Receive the output of those tool calls.

4.  Repeat until the task is accomplished.

We cover these in Chapters [5](ch05.html#ch05_cognitive_architectures_with_langgraph_1736545670030774) through [8](ch08.html#ch08_patterns_to_make_the_most_of_llms_1736545674143600).

For chatbot use cases, it becomes useful to keep track of previous interactions and use them when generating the response to a future interaction. This is called *memory*, and [Chapter 4](ch04.html#ch04_using_langgraph_to_add_memory_to_your_chatbot_1736545668266431) discusses using it in LangChain.

Finally, LangChain provides the tools to compose these building blocks into cohesive applications. Chapters [1](ch01.html#ch01_llm_fundamentals_with_langchain_1736545659776004) through [6](ch06.html#ch06_agent_architecture_1736545671750341) talk more about this.

In addition to this library, LangChain provides [LangSmith](https://oreil.ly/geRgx)—a platform to help debug, test, deploy, and monitor AI workflows—and LangGraph Platform—a platform for deploying and scaling LangGraph agents. We cover these in Chapters [9](ch09.html#ch09_deployment_launching_your_ai_application_into_pro_1736545675509604) and [10](ch10.html#ch10_testing_evaluation_monitoring_and_continuous_im_1736545678108525).

# What to Expect from This Book

With this book, we hope to convey the excitement and possibility of adding LLMs to your software engineering toolbelt.

We got into programming because we like building things, getting to the end of a project, looking at the final product and realizing there’s something new out there, and we built it. Programming with LLMs is so exciting to us because it expands the set of things we can build, it makes previously hard things easy (for example, extracting relevant numbers from a long text) and previously impossible things possible—try building an automated assistant a year ago and you end up with the *phone tree hell* we all know and love from calling up customer support numbers.

Now with LLMs and LangChain, you can actually build pleasant assistants (or myriad other applications) that chat with you and understand your intent to a very reasonable degree. The difference is night and day! If that sounds exciting to you (as it does to us) then you’ve come to the right place.

In this Preface, we’ve given you a refresher on what makes LLMs tick and why exactly that gives you “thing-building” superpowers. Having these very large ML models that understand language and can output answers written in conversational English (or some other language) gives you a *programmable* (through prompt engineering), versatile language-generation tool. By the end of the book, we hope you’ll see just how powerful that can be.

We’ll begin with an AI chatbot customized by, for the most part, plain English instructions. That alone should be an eye-opener: you can now “program” part of the behavior of your application without code.

Then comes the next capability: giving your chatbot access to your own documents, which takes it from a generic assistant to one that’s knowledgeable about any area of human knowledge for which you can find a library of written text. This will allow you to have the chatbot answer questions or summarize documents you wrote, for instance.

After that, we’ll make the chatbot remember your previous conversations. This will improve it in two ways: It will feel a lot more natural to have a conversation with a chatbot that remembers what you have previously chatted about, and over time the chatbot can be personalized to the preferences of each of its users individually.

Next, we’ll use chain-of-thought and tool-calling techniques to give the chatbot the ability to plan and act on those plans, iteratively. This will enable it to work toward more complicated requests, such as writing a research report about a subject of your choice.

As you use your chatbot for more complicated tasks, you’ll feel the need to give it the tools to collaborate with you. This encompasses both giving you the ability to interrupt or authorize actions before they are taken, as well as providing the chatbot with the ability to ask for more information or clarification before acting.

Finally, we’ll show you how to deploy your chatbot to production and discuss what you need to consider before and after taking that step, including latency, reliability, and security. Then we’ll show you how to monitor your chatbot in production and continue to improve it as it is used.

Along the way, we’ll teach you the ins and outs of each of these techniques, so that when you finish the book, you will have truly added a new tool (or two) to your software engineering toolbelt.

# Conventions Used in This Book

The following typographical conventions are used in this book:

*Italic*

Indicates new terms, URLs, email addresses, filenames, and file extensions.

`Constant width`

Used for program listings, as well as within paragraphs to refer to program elements such as variable or function names, databases, data types, environment variables, statements, and keywords.

###### Tip

This element signifies a tip or suggestion.

###### Note

This element signifies a general note.

# Using Code Examples

Supplemental material (code examples, exercises, etc.) is available for download at [*https://oreil.ly/supp-LearningLangChain*](https://oreil.ly/supp-LearningLangChain).

If you have a technical question or a problem using the code examples, please send email to [*support@oreilly.com*](mailto:support@oreilly.com).

This book is here to help you get your job done. In general, if example code is offered with this book, you may use it in your programs and documentation. You do not need to contact us for permission unless you’re reproducing a significant portion of the code. For example, writing a program that uses several chunks of code from this book does not require permission. Selling or distributing examples from O’Reilly books does require permission. Answering a question by citing this book and quoting example code does not require permission. Incorporating a significant amount of example code from this book into your product’s documentation does require permission.

We appreciate, but generally do not require, attribution. An attribution usually includes the title, author, publisher, and ISBN. For example: “*Learning LangChain* by Mayo Oshin and Nuno Campos (O’Reilly). Copyright 2025 Olumayowa “Mayo” Olufemi Oshin, 978-1-098-16728-8.”

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

We have a web page for this book, where we list errata, examples, and any additional information. You can access this page at [*https://oreil.ly/learning-langchain*](https://oreil.ly/learning-langchain).

For news and information about our books and courses, visit [*https://oreilly.com*](https://oreilly.com).

Find us on LinkedIn: [*https://linkedin.com/company/oreilly-media*](https://linkedin.com/company/oreilly-media).

Watch us on YouTube: [*https://youtube.com/oreillymedia*](https://youtube.com/oreillymedia).

# Acknowledgments

We would like to express our gratitude and appreciation to the reviewers—Rajat Kant Goel, Douglas Bailley, Tom Taulli, Gourav Bais, and Jacob Lee—for providing valuable technical feedback on improving this book.

^([1](preface01.html#id259-marker)) Tom B. Brown et al., [“Language Models Are Few-Shot Learners”](https://oreil.ly/1qoM6), arXiv, July 22, 2020.

^([2](preface01.html#id267-marker)) Xiang Zhang et al., [“Don’t Trust ChatGPT When Your Question Is Not in English: A Study of Multilingual Abilities and Types of LLMs”](https://oreil.ly/u5Cy1), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, December 6–10, 2023\.

^([3](preface01.html#id271-marker)) For more information, see Ashish Vaswani et al., [“Attention Is All You Need "](https://oreil.ly/Frtul), arXiv, June 12, 2017.

^([4](preface01.html#id304-marker)) Ryan Liu et al. [“Mind Your Step (by Step): Chain-of-Thought Can Reduce Performance on Tasks Where Thinking Makes Humans Worse”](https://oreil.ly/UHFp9), arXiv, November 8, 2024\.