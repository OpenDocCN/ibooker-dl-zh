# Chapter 5\. Adapting LLMs to Your Use Case

In this chapter, we will continue with our journey through the LLM landscape, exploring the various LLMs available for commercial use and providing pointers on how to choose the right LLM for your task. We will also examine how to load LLMs of various sizes and run inference on them. We will then decipher various decoding strategies for text generation. We will also investigate how to interpret the outputs and intermediate results from language models, surveying interpretability tools like LIT-NLP.

# Navigating the LLM Landscape

Seemingly a new LLM is being released every few days, many claiming to be state of the art. Most of these LLMs are not very different from each other, so you need not spend too much time tracking new LLM releases. This book’s [GitHub repository](https://oreil.ly/llm-playbooks) attempts to keep track of the major releases, but I don’t promise it will be complete.

Nevertheless, it is a good idea to have a broad understanding of the different types of LLM providers out there, the kinds of LLMs being made available, and the copyright and licensing implications. Therefore, let’s now explore the LLM landscape through this lens and understand the choices at our disposal.

## Who Are the LLM providers?

LLM providers can be broadly categorized into the following types:

Companies providing proprietary LLMs

These include companies like OpenAI [(GPT)](https://oreil.ly/r-lb1), Google [(Gemini)](https://oreil.ly/KF9Kh), Anthropic [(Claude)](https://oreil.ly/T5Wvo), [Cohere](https://oreil.ly/PiKxN), [AI21](https://oreil.ly/Y8T3q), etc. that train proprietary LLMs and make them available as an API endpoint (LLM-as-a-service). Many of these companies have also partnered with cloud providers that facilitate access to these models as a fully managed service. The relevant offerings from the major cloud providers are [Amazon Bedrock](https://oreil.ly/FVqRj) and [SageMaker JumpStart by Amazon](https://oreil.ly/e0a59), [Vertex AI by Google](https://oreil.ly/mURoC), and [Azure OpenAI by Microsoft](https://oreil.ly/Ag1r5).

Companies providing open source LLMs

These include companies that make the LLM weights public and monetize through providing deployment services ([Together AI](https://oreil.ly/urcAf)), companies whose primary business would benefit from more LLM adoption ([Cerebras](https://oreil.ly/2cVYY)), and research labs that have been releasing LLMs since the early days of Transformers (Microsoft, Google, Meta, Salesforce, etc.). Note that companies like Google have released both proprietary and open source LLMs.

Self-organizing open source collectives and community research organizations

This includes the pioneering community research organization [Eleuther AI](https://oreil.ly/ZSlbG), and [Big Science](https://oreil.ly/_NlUD). These organizations rely on grants for compute infrastructure.

Academia and government

Due to the high capital costs, not many LLMs have come out of academia so far. Examples of LLMs from government/academia include the Abu Dhabi government-funded [Technology Innovation Institute](https://oreil.ly/aMwO2), which released the [Falcon model](https://oreil.ly/vdhsL), and Tsinghua University, which released the [GLM model](https://oreil.ly/K0_zX).

[Table 5-1](#llm-provider-categories) shows the players in the LLM space, the category of entity they belong to, and the pre-trained models they have published.

Table 5-1\. LLM Providers

| Name | Category | Pre-trained models released |
| --- | --- | --- |
| Google | Company | BERT, MobileBERT, T5, FLAN-T5, ByT5, Canine, UL2, Flan-UL2, Pegasus PaLM, PaLMV2, ELECTRA, Tapas, Switch |
| Microsoft | Company | DeBERTa, DialoGPT, BioGPT, MPNet |
| OpenAI | Company | GPT-2, GPT-3, GPT-3.5, GPT-4 |
| Amazon | Company | Titan |
| Anthropic | Company | Claude, Claude-2 |
| Cohere | Company | Cohere Command, Cohere Base |
| Meta | Company | RoBERTa, Llama, Llama 2, BART, OPT, Galactica |
| Salesforce | Company | CTRL, XGen, EinsteinGPT |
| MosaicML | Company (Acquired by Databricks) | MPT |
| Cerebras | Company | Cerebras-GPT, BTLM |
| Databricks | Company | Dolly-V1, Dolly-V2 |
| Stability AI | Company | StableLM |
| Together AI | Company | RedPajama |
| Ontocord AI | Nonprofit | MDEL |
| Eleuther AI | Nonprofit | Pythia, GPT Neo, GPT-NeoX, GPT-J |
| Big Science | Nonprofit | BLOOM |
| Tsinghua University | Academic | GLM |
| Technology Innovation Institute | Academic | Falcon |
| UC Berkeley | Academic | OpenLLaMA |
| Adept AI | Company | Persimmon |
| Mistral AI | Company | Mistral |
| AI21 Labs | Company | Jurassic |
| X.AI | Company | Grok |

## Model Flavors

Each model is usually released with multiple variants. It is customary to release different-sized variants of the same model. As an example, Llama 2 comes in 7B, 13B, and 70B sizes, where these numbers refer to the number of parameters in the model.

These days, LLM providers augment their pre-trained models in various ways to make them more amenable to user tasks. The augmentation process typically involves fine-tuning the model in some way, often incorporating human supervision. Some of these fine-tuning exercises can cost millions of dollars in terms of human annotations. We will refer to pre-trained models that have not undergone any augmentation as base models.

The following sections describe some of the popular augmentation types.

### Instruct-models

Instruct-models, or instruction-tuned models, are specialized in following instructions written in natural language. While base models possess powerful capabilities, they are akin to a rebellious teenager; effectively interacting with them is possible only after tediously engineering the right prompts through trial and error, which tend to be brittle. This is because the base models are trained on either denoising objectives or next-word prediction objectives, which are different from the tasks users typically want to solve. By instruction-tuning the base model, the resulting model is able to more effectively respond to human instructions and be helpful.

A typical instruction-tuning dataset consists of a diverse set of tasks expressed in natural language, along with input-output pairs. In [Chapter 6](ch06.html#llm-fine-tuning), we will explore various techniques to construct instruction-tuning datasets and demonstrate how to perform instruction-tuning on a model.

Here is an example from a popular instruction-tuning dataset called [FLAN](https://oreil.ly/YJ_Xr).

> *Prompt:* “What is the sentiment of the following review? The pizza was ok but the service was terrible. I stopped in for a quick lunch and got the slice special but it ended up taking an hour after waiting several minutes for someone at the front counter and then again for the slices. The place was empty other than myself, yet I couldn’t get any help/service. OPTIONS: - negative - positive”
> 
> *FLAN:* “Negative”

In this example, the input consists of an instruction, “What is the sentiment of the following review?” expressed in a way that humans would naturally express, along with the input and output. The input is the actual review and the output is the solution to the task, either generated by a model or annotated by a human.

[Figure 5-1](#instruction-tuning1) demonstrates the instruction-tuning process.

![Instruction tuning process](assets/dllm_0501.png)

###### Figure 5-1\. Instruction-tuning process

Instruction-tuning is one of several techniques that come under the umbrella of supervised fine-tuning (SFT). In addition to improving the ability of a model to respond effectively to user tasks, SFT-based approaches can also be used to make it less harmful by training on safety datasets that help align model outputs with the values and preferences of the model creators.

More advanced techniques to achieve this alignment include reinforcement learning-based methods like reinforcement learning from human feedback (RLHF) and reinforcement learning from AI feedback (RLAIF).

In RLHF training, human annotators select or rank candidate outputs based on certain criteria, like helpfulness and harmlessness. These annotations are used to iteratively train a reward model, which ultimately leads to the LLM being more controllable, for example, by refusing to answer inappropriate requests from users.

[Figure 5-2](#rlhf-1) shows the RLHF training process.

![RLHF](assets/dllm_0502.png)

###### Figure 5-2\. Reinforcement learning from human feedback

We will cover RLHF and other alignment techniques in detail in [Chapter 8](ch08.html#ch8).

Instead of relying on human feedback for alignment training, one can also leverage LLMs to choose between outputs based on their adherence to a set of principles (don’t be racist, don’t be rude, etc.). This technique was introduced by Anthropic and is called RLAIF. In this technique, humans only provide a desired set of principles and values (referred to as [Constitutional AI](https://oreil.ly/d8FeW)), and the LLM is tasked with determining whether its outputs adhere to these principles.

Instruction-tuned models often take the suffix *instruct*, like RedPajama-Instruct.

### Chat-models

Chat-models are instruction-tuned models that are optimized for multi-turn dialog. Examples include ChatGPT, Llama 2-Chat, MPT-Chat, OpenAssistant, etc.

### Long-context models

As discussed in [Chapter 1](ch01.html#chapter_llm-introduction), Transformer-based LLMs have a limited context length. To recap, context length typically refers to the sum of the number of input and output tokens processed by the model per invocation. Typical context lengths of modern LLMs range from 8,000 to 128,000 tokens, with some variants of Gemini supporting over a million tokens. Some models are released with a long-context variant; for example GPT 3.5 comes with a default 4K context size but also has a 16K context size variant. [MPT](https://oreil.ly/wKqdL) also has a long-context variant that has been trained on 65k context length but can potentially be used for even longer contexts during inference.

### Domain-adapted or task-adapted models

LLM providers also might perform fine-tuning on specific tasks like summarization or financial sentiment analysis. They may also produce distilled versions of the model, where a smaller model is fine-tuned on outputs from the larger model for a particular task. Examples of task-specific fine-tunes include [FinBERT](https://oreil.ly/uKUAp), which is fine-tuned on financial sentiment analysis datasets, and [UniversalNER](https://oreil.ly/8A0pn), which is distilled using named-entity-recognition data.

## Open Source LLMs

Open source is often used as a catch-all phrase to refer to models with some aspect that is publicly available. We will define open source as:

> Software artifacts that are released under a license that allows users to *study*, *use*, *modify*, and *redistribute* them to *anyone* and for any *purpose*.

For a more formal and comprehensive definition of open source software, refer to the Open Source Initiative’s [official definition](https://oreil.ly/7cezH).

For an LLM to be considered fully open, all of the following needs to be published:

Model weights

This includes all the parameters of the model and the model configuration. Having access to this enables us to add to or modify the model parameters in any way we deem fit. Model checkpoints at various stages of training are also encouraged to be released.

Model code

Releasing only the weights of the model is akin to providing a software binary without providing the source code. Model code not only includes model training code and hyperparameter settings but also code used for pre-processing training data. Releasing information about infrastructure setup and configuration also goes a long way toward enhancing model reproducibility. In most cases, even with model code fully available, models may not be easily reproducible due to resource limitations and the nondeterministic nature of training.

Training data

This includes the training data used for the model, and ideally information or code on how it was sourced. It is also encouraged to release data at different stages of transformation of the data preprocessing pipeline, as well as the order in which the data was fed to the model. Training data is the component that is least published by model providers. Thus, most open source models are not *fully open* because the dataset is not public.

Training data is often not released due to competitive reasons. As discussed in Chapters [3](ch03.html#chapter-LLM-tokenization) and [4](ch04.html#chapter_transformer-architecture), most LLMs today use variants of the same architecture and training code. The distinguishing factor can often be the data content and preprocessing. Parts of the training data might be acquired using a licensing agreement, which prohibits the model provider from releasing the data publicly.

Another reason for not releasing training data is that there are unresolved legal issues pertaining to training data, especially surrounding copyright. As an example, The Pile dataset created by Eleuther AI is no longer available at the official link because it contains text from copyrighted books (the Books3 dataset). Note that The Pile is pre-processed so the books are not in human-readable form and are not easily reproducible, as they are split, shuffled, and mixed.

Most training data is sourced from the open web and thus may potentially contain violent or sexual content that is illegal in certain jurisdictions. Despite the best intentions and rigorous filtering, some of these data might still be present in the final dataset. Thus many datasets that have been previously open are no longer open, LAION’s image datasets being one example.

Ultimately, the license under which the model has been released determines the terms under which you can use, modify, or redistribute the original or modified LLM. Broadly speaking, open LLMs are distributed under three types of licenses:

Noncommercial

These licenses only allow research and personal use and prohibit the use of the model for commercial purposes. In many cases, the model artifacts are gated through an application form where a user would have to justify their need for access by providing a compelling research use case.

Copy-left

This type of license permits commercial usage, but all source or derivative work needs to be released under the same license, thus making it harder to develop proprietary modifications. The degree to which this condition applies depends on the specific license being used.

Permissive

This type of license permits commercial usage, including modifying and redistributing it in proprietary applications, i.e., there is no obligation for the redistribution to be open source. Some licenses in this category also permit patents.

New types of licenses are being devised that restrict usage of the model for particular use cases, often for safety reasons. An example of this is the [Open RAIL-M license](https://oreil.ly/2UVMe), which prohibits usage of the model in use cases like providing medical advice, law enforcement, immigration and asylum processes, etc. For a full list of restricted use cases, see Attachment A of the license.

As a practitioner intending to use open LLMs in your organization for commercial reasons, it is best to use ones with permissive licenses. Popular examples of permissive licenses include the Apache 2.0 and the MIT license.

[Creative Commons (CC) licenses](https://oreil.ly/PQy6D) are a popular class of licenses used to distribute open LLMs.The licenses have names like CC-BY-NC-SA, etc. Here is an easy way to remember what these names mean:

BY

If the license contains this term, it means attribution is needed. If it contains only CC-BY, it means the license is permissive.

SA

If the license contains this term, it means redistribution should occur under the same terms as this license. In other words, it is a copy-left license.

NC

NC stands for noncommercial. Thus, if the license contains this term, the model can only be used for research or personal use cases.

ND

ND stands for no derivatives. If the license contains this term, then distribution of modifications to the model is not allowed.

###### Note

Today, models that have open weights and open code and are released under a license that allows redistribution to anyone and for any use case are considered open source models. Arguably, however, access to the training data is also crucial to inspect and study the model, which is part of the open source definition we introduced earlier.

[Table 5-2](#llm-taxonomy) shows the various LLMs available, the licenses under which they are published, and their available sizes and flavors. Note that the LLM may be instruction-tuned or chat-tuned by a different entity than the one that pre-trained the LLM.

Table 5-2\. List of available LLMs

| Name | Availability | Sizes | Variants |
| --- | --- | --- | --- |
| GPT-4 | Proprietary | Unknown | GPT-4 32K context, GPT-4 8K context |
| GPT-3.5 Turbo | Proprietary | Unknown | GPT-3.5 4K context, GPT-3.5 16K context |
| Claude Instant | Proprietary | Unknown | - |
| Claude 2 | Proprietary | Unknown | - |
| MPT | Apache 2.0 | 1B, 7B, 30B | MPT 65K storywriter |
| CerebrasGPT | Apache 2.0 | 111M, 256M, 590M, 1.3B, 2.7B, 6.7B, 13B | CerebrasGPT |
| Stability LM | CC-BY-SA | 7B | - |
| RedPajama | Apache 2.0 | 3B, 7B | RedPajama-INCITE-Instruct, RedPajama-INCITE-Chat |
| GPT-Neo X | Apache 2.0 | 20B | - |
| BLOOM | Open, restricted use | 176B | BLOOMZ |
| Llama | Open, no commercial use | 7B, 13B, 33B, 65B | - |
| Llama 2 | Open, commercial use | 7B, 13B, 70B | Llama 2-Chat |
| Zephyr | Apache 2.0 | 7B | - |
| Gemma | Open, restricted use | 2B, 7B | Gemma-Instruction Tuned |

# How to Choose an LLM for Your Task

Given the plethora of options available, how do you ensure you choose the right LLM for your task? Depending on your situation, there are a multitude of criteria to consider, including:

Cost

This includes inference or fine-tuning costs, and costs associated with building software scaffolding, monitoring and observability, deployment and maintenance (collectively referred to as LLMOps).

[Time per output token (TPOT)](https://oreil.ly/mEDRt)

This is a metric used to measure the speed of text generation as experienced by the end user.

Task performance

This refers to the performance requirements of the task and the relevant metrics like precision or accuracy. What level of performance is *good enough*?

Type of tasks

The nature of the tasks the LLM will be used for, like summarization, question answering, classification, etc.

Capabilities required

Examples of capabilities include arithmetic reasoning, logical reasoning, planning, task decomposition, etc. A lot of these capabilities, to the extent that they actually exist or approximate, are *emergent properties* of an LLM as discussed in [Chapter 1](ch01.html#chapter_llm-introduction), and are not exhibited by smaller models.

Licensing

You can use only those models that allow your mode of usage. Even models that explicitly allow commercial use can have restrictions on certain types of use cases. For example, as noted earlier, the Big Science OpenRAIL-M license restricts the usage of the LLM in use cases pertaining to law enforcement, immigration, or asylum processes.

In-house ML/MLOps talent

The strength of in-house talent determines the customizations you can afford. For example, do you have enough in-house talent for building inference optimization systems?

Other nonfunctional criteria

This includes safety, security, privacy, etc. Cloud providers and startups are already implementing solutions that can address these issues.

You may have to choose between proprietary and open source LLMs.

## Open Source Versus Proprietary LLMs

Debates about the merits of open source versus proprietary software have been commonplace in the tech industry for several decades now, and we are seeing it become increasingly relevant in the realm of LLMs as well. The biggest advantage of open source models are the transparency and flexibility they provide, not necessarily the cost. Self-hosting open source LLMs can incur a lot of engineering overhead and compute/memory costs, and using managed services might not always be able to match proprietary models in terms of latency, throughput, and inference cost. Moreover, many open source LLMs are not easily accessible through managed services and other third-party deployment options. This situation is bound to change dramatically as the field matures, but in the meanwhile, run through your calculations for your specific situation to determine the costs incurred for using each (type of) model.

The flexibility provided by open source models helps with your ability to debug, interpret, and augment the LLM with any kind of training/fine-tuning you choose, instead of the restricted avenues made available by the LLM provider. This allows you to more substantially align the LLM to your preferences and values instead of the ones decided by the LLM provider. Having full availability of all the token probabilities (logits) is a superpower, as we will see throughout the book.

The availability of open source LLMs has enabled teams to develop models and applications that might not be lucrative for larger companies with a profit motive, like fine-tuning models to support low-resource languages (languages that do not have a significant data footprint on the internet, like regional languages of India or Indigenous languages of Canada). An example is the [Kannada Llama model](https://oreil.ly/hoBQ1), built over Llama 2 by continually pre-training and fine-tuning on tokens from the Kannada language, a regional language of India.

Not all open source models are fully transparent. As mentioned earlier, most for-profit companies that release open source LLMs do not make the training datasets public. For instance, Meta hasn’t disclosed all the details of the training datasets used to train the Llama 2 model. Knowing which datasets are used to train the model can help you assess whether there is test set contamination and understand what kind of knowledge you can expect the LLM to possess.

As of this book’s writing, open source models like Llama 3.2 and DeepSeek v3 have more or less caught up to state-of-the-art proprietary models from OpenAI or Anthropic. However, there is a new gap developing between proprietary and open source models in the realm of reasoning models like OpenAI’s o3, that use inference-time compute techniques (discussed in [Chapter 8](ch08.html#ch8)). Throughout this book, we will showcase scenarios where open source models have an advantage.

###### Tip

Always check if the model provider has an active developer community on GitHub/Discord/Slack, and that the development team is actively engaged in those channels, responding to user comments and questions. I recommend preferring models with active developer communities, provided they satisfy your primary criteria.

## LLM Evaluation

We will start this section with a caveat: evaluating LLMs is probably the most challenging task in the LLM space at present. Current methods of benchmarking are broken, easily gamed, and hard to interpret. Nevertheless, benchmarks are still a useful starting point on your road to evaluation. We will start by looking at current public benchmarks and then discuss how you can build more holistic internal benchmarks.

To evaluate LLMs on their task performance, there are a lot of benchmark datasets that test a wide variety of skills. Not all skills are relevant to your use case, so you can choose to focus on specific benchmarks that test the skills you need the LLM to perform well on.

The leaderboard on these benchmark tests changes very often, especially if only open source models are being evaluated, but that does not mean you need to change the LLMs you use every time there is a new leader on the board. Usually, the differences between the top models are quite marginal. The fine-grained choice of LLM usually isn’t the most important criteria determining the success of your task, and you are better off spending that bandwidth working on cleaning and understanding your data, which is still the most important component of the project.

Let’s look at a few popular ways in which the field is evaluating LLMs.

### Eleuther AI LM Evaluation Harness

Through the [LM Evaluation Harness](https://oreil.ly/SiOXq), Eleuther AI supports benchmarking on over 400 different benchmark tasks, evaluating skills as varied as open-domain question answering, arithmetic and logical reasoning, linguistic tasks, machine translation, toxic language detection, etc. You can use this tool to evaluate any model on the [Hugging Face Hub](https://oreil.ly/IHd22), a platform containing thousands of pre-trained and fine-tuned models, on the benchmarks of your choice.

Here is an example from `bigbench_formal_fallacies_syllogisms_negation`, one of the benchmark tasks:

```py
 {
    "input": "\"Some football fans admire various clubs, others love
    only a single team. But who is a fan of whom precisely? The
    following argument pertains to this question: First premise: Mario
    is a friend of FK \u017dalgiris Vilnius. Second premise: Being a
    follower of F.C. Copenhagen is necessary for being a friend of FK
    \u017dalgiris Vilnius. It follows that Mario is a follower of F.C.
    Copenhagen.\"\n Is the argument, given the explicitly stated
    premises, deductively valid or invalid?",
    "target_scores": {
        "valid": 1,
        "invalid": 0
    }
```

In this task, the model is asked to spot logical fallacies by deducing whether the presented argument is valid given the premises.

There is also support for evaluation of proprietary models using this harness. For example, here is how you would evaluate OpenAI models:

```py
export OPENAI_API_SECRET_KEY=<Key>
python main.py \
lm_eval --model openai-completions \
        --model_args model=gpt-3.5-turbo \
         --tasks bigbench_formal_fallacies_syllogisms_negation
```

###### Tip

While choosing or developing a benchmarking task to evaluate, I recommend focusing on picking ones that test the capabilities needed to solve the task of your interest, rather than the actual task itself. For example, if you are building a summarizer application that needs to perform a lot of logical reasoning to generate the summaries, it is better to focus on benchmark tests that directly test logical reasoning capabilities than ones that test summarization performance.

### Hugging Face Open LLM Leaderboard

As of the book’s writing, the [Open LLM Leaderboard](https://oreil.ly/tspBY) uses Eleuther AI’s LM Evaluation Harness to evaluate the performance of models on six benchmark tasks:

Massive Multitask Language Understanding (MMLU)

This test evaluates the LLM on knowledge-intensive tasks, drawing from fields like US history, biology, mathematics, and more than 50 other subjects in a multiple choice framework.

AI2 Reasoning Challenge (ARC)

This test evaluates the LLM on multiple-choice grade school science questions that need complex reasoning as well as world knowledge to answer.

Hellaswag

This test evaluates commonsense reasoning by providing the LLM with a situation and asking it to predict what might happen next out of the given choices, based on common sense.

TruthfulQA

This test evaluates the LLM’s ability to provide answers that don’t contain falsehoods.

Winogrande

This test is composed of fill-in-the-blank questions that test commonsense reasoning.

GSM8K

This test evaluates the LLM’s ability to complete grade school math problems involving a sequence of basic arithmetic operations.

[Figure 5-3](#llm-leaderboard) shows a snapshot of the LLM leaderboard as of the time of the book’s writing. We can see that:

*   Larger models perform better.

*   Instruction-tuned or fine-tuned variants of models perform better.

![Snapshot of the Open LLM Leaderboard](assets/dllm_0503.png)

###### Figure 5-3\. Snapshot of the Open LLM Leaderboard

The validity of these benchmarks are in question as complete test set decontamination is not guaranteed. Model providers are also optimizing to solve these benchmarks, thus reducing the value of these benchmarks to serve as reliable estimators of general-purpose performance.

### HELM

[Holistic Evaluation of Language Models (HELM)](https://oreil.ly/MNHDs) is an evaluation framework by Stanford that aims to calculate a wide variety of metrics over a range of benchmark tasks. Fifty-nine metrics are calculated overall, testing accuracy, calibration, robustness, fairness, bias, toxicity, efficiency, summarization performance, copyright infringement, and more. The tasks tested include question answering, summarization, text classification, information retrieval, sentiment analysis, and toxicity detection.

[Figure 5-4](#helm-leaderboard) shows a snapshot of the HELM leaderboard as of the time of the book’s writing.

![Snapshot of the HELM leaderboard](assets/dllm_0504.png)

###### Figure 5-4\. Snapshot of the HELM leaderboard

### Elo Rating

Now that we have seen the limitations of quantitative evaluation, let’s explore how we can most effectively incorporate human evaluations. One promising framework is the [Elo rating system](https://oreil.ly/bTD7I), used in chess to rank players.

[Large model systems organization (LMSYS Org)](https://oreil.ly/HGVz2) has implemented an evaluation platform based on the Elo rating system called the [Chatbot Arena](https://oreil.ly/evgQX). Chatbot Arena solicits crowdsourced evaluations by inviting people to choose between two randomized and anonymized LLMs by chatting with them side-by-side. The leaderboard is found [online](https://oreil.ly/Y6zmN), with models from OpenAi, DeepSeek, Google DeepMind, and Anthropic dominating.

[Figure 5-5](#chatbotarena-leaderboard) shows a snapshot of the Chatbot Arena leaderboard as of the time of the book’s writing.

![Snapshot of the Chatbot Arena leaderboard](assets/dllm_0505.png)

###### Figure 5-5\. Snapshot of the Chatbot Arena leaderboard

### Interpreting benchmark results

How do you interpret evaluation results presented in research papers? Try to methodically ask as many questions as possible, and check if the answers are covered in the paper or other material. As an example, let us take the Llama 2-chat evaluation graphs presented in the [Llama 2 paper](https://oreil.ly/BcgXs). In particular, study Figures 1 and 3, which demonstrate how Llama 2-Chat compares in helpfulness and safety with other chat models. Some of the questions that come to mind are:

*   What does the evaluation dataset look like? Do we have access to it?

*   What is the difficulty level of the test set? Maybe the model is competitive with respect to ChatGPT for easier examples but how does it perform with more difficult examples?

*   What proportion of examples in the test set can be considered difficult?

*   What kinds of scenarios are covered in the test set? What degree of overlap do these scenarios have with the chat-tuning sets?

*   What definition do they use for safety?

*   Can there be a bias in the evaluation due to models being evaluated on the basis of a particular definition of safety, which Llama 2 was trained to adhere to, while other models may have different definitions of safety?

Rigorously interrogating the results this way helps you develop a deeper understanding of what is being evaluated, and whether it aligns with the capabilities you need from the language model for your own tasks. For more rigorous LLM evaluation, I strongly recommend developing your own internal benchmarks.

###### Warning

Do not trust evaluations performed by GPT-4 or any other LLM. We have no idea what evaluation criteria it uses nor do we have a deeper understanding of its biases.

Robust evaluation of LLMs is further complicated by the sensitivity of the prompts and the probabilistic nature of generative models. For example, I often see papers claiming that “GPT-4 does not have reasoning capabilities,” while not using any prompting techniques during evaluation. In many of these cases, it turns out that the model can in fact perform the task if prompted with CoT prompting. While evaluation prompts need not be heavily engineered, using rudimentary techniques like CoT should be standard practice, and not using them means that the model capabilities are being underestimated.

# Loading LLMs

While it is possible to load and run inference on LLMs with just CPUs, you need GPUs if you want acceptable text generation speeds. Choosing a GPU depends on cost, the size of the model, whether you are training the model or just running inference, and support for optimizations. Tim Dettmers has developed a great [flowchart](https://oreil.ly/t6iPQ) that you can use to figure out which GPU best serves your needs.

Let’s figure out the amount of GPU RAM needed to load an LLM of a given size. LLMs can be loaded in various *precisions*:

Float32

32-bit floating point representation, each parameter occupying 4 bytes of storage.

Float16

16-bit floating point representation. Only 5 bits are reserved for the exponent as opposed to 8 bits in Float32\. This means that using Float16 comes with overflow/underflow problems for very large and small numbers.

bfloat16 (BF16)

16-bit floating point representation. Just like Float32, 8 bits are reserved for the exponent, thus alleviating the underflow/overflow problems observed in Float16.

Int8

8-bit integer representation. Running inference in 8-bit mode is around 20% slower than running in Float16.

FP8, FP4

8-bit and 4-bit floating point representation.

We will explore these formats in detail in [Chapter 9](ch09.html#ch09). Generally, running inference on a model with 7B parameters will need around 7 GB of GPU RAM if running in 8-bit mode and around 14 GB if running in BF16\. If you intend to fine-tune the whole model, you will need a lot more memory.

## Hugging Face Accelerate

You can run inference on models even if they don’t fit in the GPU RAM. The [*accelerate* library](https://oreil.ly/OYdyf) by Hugging Face facilitates this by loading parts of the model into CPU RAM if the GPU RAM is filled, and then loading parts of the model into disk if the CPU RAM is also filled. [“Accelerate Big Model Inference: How Does it Work?”](https://oreil.ly/J8duc) shows how the accelerate library operates under the hood. This whole process is abstracted from the user, so all you need to load a large model is to run this code:

```py
!pip install transformers accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neox-20B")
input_ids = tokenizer("Language models are", return_tensors="pt")
gen_tokens = model.generate(**input_ids, max_new_tokens =1)
```

## Ollama

There are many tools available that facilitate loading LLMs locally, including on your own laptop. One such library is Ollama, which supports Windows, Mac, and Linux operating systems. Using Ollama, you can load 13B models if your machine has at least 16GB of available RAM. Ollama supports many open models like Mistral, Llama, Gemma, etc. Ollama provides a REST API that you can use to run inference and build LLM-driven applications. It also has several Terminal and UI integrations that enable you to build user-facing applications with ease.

Let’s see how we can use Google’s Gemma 2B model using Ollama. First, download [the version of Ollama](https://oreil.ly/yly44) to your machine based on your operating system. Next, pull the Gemma model to your machine with:

```py
ollama pull gemma:2b
```

You can also create a Modelfile that contains configuration information for the model. This includes system prompts and prompt templates, decoding parameters like temperature, and conversation history. Refer to the [documentation](https://oreil.ly/ba-1u) for a full list of available options.

An example Modelfile is:

```py
FROM gemma:2b

PARAMETER temperature 0.2

SYSTEM """
You are a provocateur who speaks only in limericks.
"""
```

After creating your Modelfile, you can run the model:

```py
ollama create local-gemma -f ./Modelfile
ollama run local-gemma
```

The book’s GitHub repo contains a sample end-to-end application built using Ollama and one of its UI integrations. You can also experiment with similar tools like [LM Studio](https://oreil.ly/uFsiR) and [GPT4All](https://oreil.ly/XUXhq).

###### Tip

You can load custom models using Ollama if they are in the GPT-Generated Unified Format (GGUF).

## LLM Inference APIs

While you can deploy an LLM yourself, modern-day inference consists of so many optimizations, many of them proprietary, that it takes a lot of effort to bring your inference speeds up to par with commercially available solutions. Several inference services like [Together AI](https://oreil.ly/L3zo0) exist that facilitate inference of open source or custom models either through serverless endpoints or dedicated instances. Another option is Hugging Face’s [TGI (Text Generation Inference)](https://oreil.ly/XXFpa), which has been recently [reinstated](https://oreil.ly/BJJlY) to a permissive open source license.

# Decoding Strategies

Now that we have learned how to load a model, let’s understand how to effectively generate text. To this end, several *decoding* strategies have been devised in the past few years. Let’s go through them in detail.

## Greedy Decoding

The simplest form of decoding is to just generate the token that has the highest probability. The drawback of this approach is that it causes repetitiveness in the output. Here is an example:

```py
input = tokenizer('The keyboard suddenly came to life. It ventured up the',

return_tensors='pt').to(torch_device)
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

You can see that the output starts getting repetitive. Therefore, greedy decoding is not suitable unless you are generating really short sequences, like a token just producing a classification task output.

[Figure 5-6](#greedy-decoding) shows an example of greedy decoding using the FLAN-T5 model. Note that we missed out on some great sequences because one of the desired tokens has slightly lower probability, ensuring it never gets picked.

![Greedy decoding](assets/dllm_0506.png)

###### Figure 5-6\. Greedy decoding

## Beam Search

An alternative to greedy decoding is beam search. An important parameter of beam search is the beam size, *n*. At the first step, the top *n* tokens with the highest probabilities are selected as hypotheses. For the next few steps, the model generates token continuations for each of the hypotheses. The token chosen to be generated is the one whose continuations have the highest cumulative probability.

In the Hugging Face `transformers` library, the `num_beams` parameter of the `model.generate()` function determines the size of the beam. Here is how the decoding code would look if we used beam search:

```py
output = model.generate(**inputs, max_new_tokens=50, num_beams = 3)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

[Figure 5-7](#beam-search) shows an example of beam search using the FLAN-T5 model. Note that the repetitiveness problem hasn’t really been solved using beam search. Similar to greedy decoding, the generated text also sounds very constricted and not humanlike, due to the complete absence of lower probability words.

![Beam search](assets/dllm_0507.png)

###### Figure 5-7\. Beam search

To resolve these issues, we will need to start introducing some randomness and begin sampling from the probability distribution to ensure not just the top two or three tokens get generated all the time.

## Top-k Sampling

In top-k sampling, the model samples from a distribution of just the k tokens of the output distribution that have the highest probability. The probability mass is redistributed over the k tokens, and the model samples from this distribution to generate the next token. Hugging Face provides the `top_k` parameter in its generate function:

```py
output = model.generate(**inputs, max_new_tokens=50, do_sample=True, top_k=40)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

[Figure 5-8](#topk-sampling) shows an example of top-k sampling using the FLAN-T5 model. Note that this is a vast improvement from greedy or beam search. However, top-k leads to problematic generations when used in cases where the probability is dominated by a few tokens, meaning that tokens with very low probability end up being included in the top-k.

![Top-k sampling](assets/dllm_0508.png)

###### Figure 5-8\. Top-k sampling

## Top-p Sampling

Top-p sampling solves the problem with top-k sampling by making the number of candidate tokens dynamic. Top-p involves choosing the smallest number of tokens whose cumulative distribution exceeds a given probability p. Here is how you can implement this using Hugging Face `transformers`:

```py
output = model.generate(**inputs, max_new_tokens=50, top_p=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

[Figure 5-9](#topp-sampling) shows an example of top-p sampling using the FLAN-T5 model. Top-p sampling, also called nucleus sampling, is the most popular sampling strategy used today.

![Top-p sampling](assets/dllm_0509.png)

###### Figure 5-9\. Top-p sampling

###### Note

So far, the decoding approaches we have seen operate serially; i.e., each token is generated one at a time, with a full pass through the model each time. This is too inefficient for latency-sensitive applications. In [Chapter 9](ch09.html#ch09), we will discuss methods like speculative decoding, which can speed up the decoding process.

# Running Inference on LLMs

Now that we have learned how to access and load LLMs and understood the decoding process, let’s begin using them to solve our tasks. We call this *LLM inference*.

You will have seen that LLM outputs are not consistent and sometimes differ wildly across multiple generations for the same prompt. As we learned in the section on decoding, unless you are using greedy search or any other deterministic algorithm, the LLM is sampling from a token distribution.

Some ways to make the generation more deterministic is to set the temperature to zero and keeping the random seed for the sampling constant. Even then, you may not be able to guarantee the same (deterministic) outputs every time you send the LLM the same input.

Sources of nondeterminism range from using multi-threading to floating-point rounding errors to use of certain model architectures (for example, it is known that the [Sparse MoE architecture](https://oreil.ly/pzchE) produces nondeterministic outputs).

Reducing the temperature to zero or close to zero impacts the LLM’s creativity and makes its outputs more predictable, which might not be suitable for many applications.

In production settings where reliability is important, you should run multiple generations for the same input and use a technique like majority voting or heuristics to select the right output. This is very important due to the nature of the decoding process; sometimes the wrong tokens can be generated, and since every token generated is a function of the tokens generated before it, the error can be propagated far ahead.

[Self-consistency](https://oreil.ly/wEE8q) is a popular prompting technique that uses majority voting in conjunction with CoT prompting. In this technique, we add the CoT prompt “Let’s think step by step” to the input and run multiple generations (reasoning paths). We then use majority voting to select the correct output.

# Structured Outputs

We might want the output of the LLM to be in some structured format, so that it can be consumed by other software systems. But this is easier said than done; current LLMs aren’t as controllable as we would like them to be. Some LLMs can be excessively chatty. Ask them to give a Yes/No answer and they respond with “The answer to this question is ‘Yes’.”

One way to get structured outputs from the LLM is to define a JSON schema, provide the schema to the LLM, and prompt it to generate outputs adhering to the schema. For larger models, this works almost all the time, with some schema corruption errors that you can catch and handle.

For smaller models, you can use libraries like [Jsonformer](https://oreil.ly/aSc0f). Jsonformer delegates the generation of the content tokens to the LLM but fills the content in JSON form by itself. Jsonformer is built on top of Hugging Face and thus supports any model that is supported by Hugging Face.

More advanced structured outputs can be facilitated by using libraries like [LMQL](https://oreil.ly/LlkEj) or [Guidance](https://oreil.ly/cFe5s). These libraries provide a programming paradigm for prompting and facilitate controlled generation.

Features available through these libraries include:

Restricting output to a finite set of tokens

This is useful for classification problems, where you have a finite set of output labels. For example, you can restrict the output to be positive, negative, or neutral for a sentiment analysis task.

Controlling output format using regular expressions

For example, you can use regular expressions to specify a custom date format.

Control output format using context-free grammars (CFG)

A CFG defines the rules that generated strings need to follow. For more background on CFGs, refer to [Aditya’s blog](https://oreil.ly/M00us). Using CFGs, we can use LLMs to more effectively solve sequence tagging tasks like NER or part-of-speech tagging.

# Model Debugging and Interpretability

Now that we are comfortable with loading LLMs and generating text using them, we would like to be able to understand model behavior and explore the examples for which the model fails. Interpretability in LLMs is much less developed than in other areas of machine learning. However, we can get partial interpretability by exploring how the output changes upon minor variances in the input, and by analyzing the intermediate outputs as the inputs propagate through the Transformer architecture.

Google’s open source tool [LIT-NLP](https://oreil.ly/YFY4q) is a handy tool that supports visualizations of model behavior as well as various debugging workflows.

[Figure 5-10](#lit-NLP) shows an example of LIT-NLP in action, providing interpretability for a T5 model running a summarization task.

![lit-NLP](assets/dllm_0510.png)

###### Figure 5-10\. LIT-NLP

LIT-NLP features that help you debug your models include:

*   Visualization of the attention mechanism

*   Salience maps, which show parts of the input that are paid most attention to by the model

*   Visualization of embeddings

*   Counterfactual analysis that shows how your model behavior changes after a change to the input like adding or removing a token.

For more details on using LIT-NLP for error analysis, refer to [Google’s tutorial](https://oreil.ly/zcsLu) on using LIT-NLP with the Gemma LLM where they find errors in few-shot prompts by analyzing incorrect examples and observing which parts of the prompt contributed most to the output (salience).

# Summary

In this chapter, we journeyed through the LLM landscape and noted the various options we have at our disposal. We learned how to determine the criteria most relevant to our tasks and choose the right LLM accordingly. We explored various LLM benchmarks and showed how to interpret their results. We learned how to load LLMs and run inference on them, along with efficient decoding strategies. Finally, we showcased interpretability tools like LIT-NLP that can help us understand what is going on behind the scenes in the Transformer architecture.

In the next chapter, we will learn how to update a model to improve its performance on our tasks of interest. We will walk through a full-fledged fine-tuning example and explore the hyperparameter tuning decisions involved. We will also learn how to construct training datasets for fine-tuning.