# Chapter 7\. Advanced Fine-Tuning Techniques

In the previous chapter, we presented the canonical way to fine-tune a typical LLM. In the real world, there are a wide variety of motivations for updating an LLM, and similarly there are multiple ways to update it. In this chapter, we will describe several advanced fine-tuning techniques and highlight the scenarios in which each technique would be suitable.

Why would you want to update the parameters of an LLM? We touched upon this in previous chapters but let’s go through it in more detail now:

Domain adaptation

The data that we work with belongs to a specialized domain that the LLM might not have been familiarized with during pre-training. In this case, we would like to update the model by training it on domain-specific data.

Task adaptation

We care about LLM performance on specific downstream tasks. To improve the LLM’s performance on these tasks, we can train it on task-specific data. This can be supervised or unsupervised.

Knowledge updating

We would like to keep the LLM’s knowledge up-to-date by continually training it on new data.

Controllability/steerability

We would like to control the behavior of the LLM, including making it more likely to follow user requests written in natural language, reject certain types of requests, and so on. Techniques to achieve this are collectively called alignment training. We will defer discussion of alignment training to [Chapter 8](ch08.html#ch8).

In this chapter, we will learn techniques that can be used to update the LLM for the aforementioned reasons. To this end, the chapter is divided into three sections:

Continual pre-training

Primarily used for domain adaptation and keeping the knowledge of the LLM up-to-date (the latter is also called lifelong-learning).

Parameter-Efficient Fine-Tuning (PEFT)

A set of fine-tuning techniques that make the fine-tuning process more efficient by updating only a small number of model parameters, thus needing less memory and compute.

Model merging/model fusion

An exciting new subfield of LLMs that explores combining the parameters of two or more models. I call this the “dark arts” of NLP, as it is poorly understood but uncannily effective if done the right way.

Let’s begin with my personal favorite: continual pre-training!

# Continual Pre-Training

The premise of continual pre-training is simple. Take a pre-trained model checkpoint and continue pre-training it with your own data. But why would you want to do that? Here are some scenarios where continual pre-training can help.

*   You work in a specialized domain like law, finance, or biomedical. In each of these cases, text belonging to these domains differs linguistically and structurally from naturally occurring English text. For example, legal text is characterized by long sentences written in a formal tone, containing jargon specific to the legal domain. Financial text is interspersed with a lot of numbers. Both legal and financial text contain a significant proportion of boilerplate text. Biomedical text contains a lot of scientific terms that are not part of the standard English vocabulary. In all these cases, you would like to pre-train your LLM on domain-specific data so that the LLM is exposed to the nuances and characteristics of domain-specific text. This is called *domain-adaptive pre-training (DAPT)*.

*   Taking DAPT one step further, you can also continue pre-training your model not just on general text from your domain of interest but also on domain text specifically related to your downstream tasks. This is called *task-adaptive pre-training (TAPT)*.

*   Your LLM is a reservoir of knowledge. But this knowledge can become obsolete over time. To keep its knowledge up-to-date, you continue pre-training the model at regular time periods or when new data is available. This is called *life-long learning*.

###### Note

You might be thinking, “If I want a domain-specific LLM, why don’t I just take my domain-specific data and train an LLM from scratch?” Well, you can, but your LLM just won’t be as performant, and the exercise will cost a whole lot more than continual pre-training. LLMs learn a wide variety of linguistic capabilities that might not be able to be learned from domain-specific text alone. Therefore, it is better to take an already pre-trained LLM that was trained on general text and then continue pre-training it with domain-specific text.

In practice, continual pre-training is a challenging exercise. This is due to the phenomenon of catastrophic forgetting, where the LLM *forgets* its previously learned capabilities and knowledge when it continues to be trained on new and different data. We will soon explore various techniques to combat the catastrophic forgetting problem.

How does continual pre-training differ from fine-tuning? The differences are mostly cosmetic and terminology-related. Just like pre-training, continual pre-training is self-supervised, while we typically use the term fine-tuning when we use supervised datasets. Continual pre-training uses the same (but not necessarily) learning objective as the one used in the original pre-training setup. Finally, continual pre-training datasets are usually orders of magnitude larger than typical fine-tuning datasets.

[Figure 7-1](#continual-pt) depicts the general continual pre-training process.

![continual-pt](assets/dllm_0701.png)

###### Figure 7-1\. Illustration of the continual pre-training process

This book’s [GitHub repo](https://oreil.ly/llm-playbooks) contains a tutorial for continual pre-training. This setup is no different than fine-tuning, except that the dataset is not labeled (self-supervised training), and the dataset is orders of magnitude larger than typical fine-tuning datasets.

As mentioned earlier, naive continual pre-training leads to catastrophic forgetting of capabilities and knowledge learned previously. Several techniques exist to alleviate this issue:

Replay (memory)

Uses training examples from the original pre-training and mixes them with the new training data.

Distillation

Takes an older checkpoint of the model and during training compares the KL-divergence between the older and the current representations and penalizes it.

Regularization

Penalizes large changes to the parameters during continual training.

Parameter expansion

Adds more parameters to the model as continual pre-training is performed. This can be done by increasing either the width or the depth of the model.

For a more comprehensive set of continual learning techniques, check out [Jin et al.’s paper](https://oreil.ly/yNa-H). In this chapter, we will dive deeper into replay and parameter expansion methods.

## Replay (Memory)

Replay-based techniques are one of the simplest techniques to alleviate catastrophic forgetting. In this approach, we store pre-training examples from the original dataset and interleave them with the continual training dataset. Thus, the data drift is not so pronounced.

The following formula has worked very well for me: sample from different subsets of the original pre-training datasets and mix them with the continual training dataset. At the start of training, let the proportion of new data be around 25%. Over training steps, this can be slowly increased up to a maximum proportion, like 80%.

If the original pre-training dataset is a monolith and not made up of several smaller datasets, you might need to identify domains yourself so that all domains in the original pre-training set are included.

## Parameter Expansion

An alternative to the replay approach is to use parameter expansion techniques. The naive way would be to just add a new layer or two on top of the model and train only those parameters during continual pre-training. You can also insert and train domain-specific parameter modules (called adapters) within existing layers. We will discuss adapter-based approaches in [“Parameter-Efficient Fine-Tuning”](#parameter-efficient-fine-tuning).

As mentioned earlier, continual pre-training can also be used to facilitate life-long learning, with the model continually being updated with new facts and knowledge. However, currently this may not be the most effective paradigm for new knowledge learning. You are probably better off using RAG for that. We will explore RAG in more detail in [Chapter 12](ch12.html#ch12).

###### Tip

[Task-adaptive pre-training (TAPT)](https://oreil.ly/H38wF) is a useful supplement to domain-adaptive pre-training. TAPT involves continual pre-training of the LLM on a much smaller but more task-specific unsupervised dataset. To prevent catastrophic forgetting, you should perform DAPT first before TAPT, and then subsequently perform any supervised fine-tuning on your downstream tasks. Unsupervised data for TAPT can be selected using similar methods as that used for DAPT: by constructing embeddings of data and selecting data that is clustered with gold-truth sentences.

In summary, continual pre-training can be very effective in cases where you have a large body of domain-specific text and the domain is very distinctly characterized by a specialized linguistic structure or vocabulary. Continual pre-training can also be used to help adapt the LLM to a new language.

###### Tip

Domain-specific text can contain jargon specific to that domain. One strategy that has worked well for me is to add extra tokens to represent domain-specific jargon.

Continual pre-training can take a lot of computational resources. Fine-tuning on smaller datasets takes substantially less resources. However, in the era of large language models, it is imperative to do all we can to reduce compute and memory requirements. Therefore, let’s next discuss some parameter-efficient fine-tuning techniques that make the fine-tuning process more accessible in resource-constrained environments.

# Parameter-Efficient Fine-Tuning

In PEFT, instead of updating all the parameters of the model, we update only a small number of parameters. This can vastly bring down compute and storage requirements.

We can categorize current PEFT techniques into three types:

Adding new parameters

This involves adding some extra parameters to the LLM and training only them.

Selecting a subset of parameters

This involves choosing to update only a small subset of parameters of the LLM, either by selecting the subset apriori or by learning the appropriate subset.

Low-rank methods

This involves using methods that reduce the number of parameters to train by finding a smaller matrix containing almost the same information as a larger matrix.

Let’s now go through each of these in detail.

## Adding New Parameters

Perhaps your work needs you to fine-tune models for a large number of tasks. Or maybe you need to drive personalization by fine-tuning a model for each user. It will be cumbersome to maintain and deploy so many full copies of fine-tuned models.

One way to avoid updating all the parameters of the model is to add a few extra parameters to the model and train only them. Instead of storing and deploying full copies of each fine-tuned model, you store only the newly added parameters.

Common ways of adding new parameters for fine-tuning include:

Bottleneck adapters

These are lightweight modules added to the Transformer layers.

Prefix tuning

These are task-specific vectors that are trained and prefixed to the input.

Prompt tuning (soft prompts)

This is similar to prefix tuning but with a simplified training approach.

Let’s discuss each of these techniques in detail.

### Bottleneck adapters

Adapters are parameter modules attached to the LLM architecture. Adapters can be integrated into the LLM architecture in a variety of ways, but in Transformers, the common way is to insert them at each layer of the Transformer. To reduce the number of parameters, the width of the adapter module should be much less than the width of the underlying Transformer model. This constitutes a *down-projection*, also called a bottleneck.

Therefore, a bottleneck adapter sublayer consists of a down-projection matrix, an up-projection matrix at the end to project back to the original dimensions, and parameters that can be configured in a variety of ways in the middle. During fine-tuning, only the adapter modules are updated. The original pre-trained model is not updated. Adapters are initialized with a near-identity initialization to ensure smooth training.

[Figure 7-2](#adapters) shows where in the Transformer architecture the bottleneck adapters typically are inserted. Note that this is just one possible configuration.

![adapters](assets/dllm_0702.png)

###### Figure 7-2\. Adapter modules in the Transformer

How does this all work in practice? The [*adapters* library](https://oreil.ly/z05rI) comes in handy to facilitate fine-tuning LLMs using these advanced techniques.

Here is how you can start using bottleneck adapters using the adapters library:

```py
from adapters import DoubleSeqBnConfig
adapter_config = DoubleSeqBnConfig()
model.add_adapter("bottleneck_adapter", config=adapter_config)
```

`DoubleSeqBnConfig` refers to a config natively supported by the library, corresponding to the adapter architecture shown in [Figure 7-2](#adapters). But as I mentioned before, you can change the size and shape of the adapters as you wish. To do that, we need to use `BnConfig`:

```py
from adapters import BnConfig
adapter_config = BnConfig(mh_adapter=True, output_adapter=True,

reduction_factor=32, non_linearity="gelu")
```

Here is what these arguments stand for:

`mh_adapter`

Refers to the adapter modules added right after the multi-head attention sublayer of the Transformer.

`output_adapter`

Refers to the adapter modules added right after the feedforward network sublayer of the Transformer.

`reduction_factor`

Refers to the down-projection factor: by how much should the adapter width be scaled down compared to the Transformer layer width?

`non_linearity`

Refers to the activation function being used, like RELU or GELU.

Refer to the adapters library [documentation](https://oreil.ly/n1Pga) for more configuration options. There are so many configuration options available!

While using bottleneck adapters leads to a vast decrease in fine-tuning time and complexity, adding parameters across all layers of the Transformer increases inference latency by a small amount. Typically, the inference time using commonly used adapter configurations is expected to increase by 6%–8%.

###### Tip

It is possible to reduce the inference latency by dropping some adapter layers during inference. [Rücklé et al. propose AdapterDrop](https://oreil.ly/GM_1X), a set of methods for dropping adapter modules during training and inference. They propose dropping adapters from the first few layers of the Transformer during inference or pruning the adapters from each layer that is the least activated.

### Prefix-tuning

One drawback of using adapter-based fine-tuning techniques is that during inference, each batch can support only a single adapter instance, i.e., an adapter fine-tuned for a particular task. Prefix-tuning, in contrast, enables multiple tasks to be run in the same batch.

In prefix-tuning, we add and train task-specific vectors to the prefix of the input. This vastly reduces the number of parameters we need to fine-tune. Recall that the prompt contains the instruction, the input, and optionally some few-shot examples. The text generated by the LLM is conditioned on the output generated so far, and the prompt. To this, we add additional context that the LLM can attend to, in the form of these prefix vectors. The new tokens prefixed to the input are called **virtual tokens** or **soft prompts**.

[Figure 7-3](#prefix-tuning) shows how prefix-tuning occurs in the Transformer.

![prefix-tuning](assets/dllm_0703.png)

###### Figure 7-3\. Prefix-tuning

As the figure shows, prefix parameters are added at each layer.

Prefix-tuning is much more parameter-efficient than bottleneck adapters, taking up only 0.1% or less of a model’s parameters, as compared to adapters where it is usually 2% or more. However, prefix-tuning is harder to train effectively than adapters. Prefix-tuning also reduces the sequence length of the model in order to accommodate the virtual tokens.

Similar to adapters, initialization is very important for prefix-tuning. The virtual tokens can be initialized by choosing words that are related to the task the model is being fine-tuned for.

Using the adapters library, we can implement prefix-tuning:

```py
from adapters import PrefixTuningConfig
adapter_config = PrefixTuningConfig()
model.add_adapter("prefix_tuning", config=adapter_config)
```

### Prompt tuning

Prompt tuning is a simplified version of prefix-tuning. Unlike prefix tuning, there are no prefix parameters at each layer.

[Figure 7-4](#prompt-tuning) shows how prompt-tuning occurs in the Transformer.

![prompt-tuning](assets/dllm_0704.png)

###### Figure 7-4\. Prompt-tuning

The adapters library provides a built-in configuration for prompt tuning:

```py
from adapters import PromptTuningConfig
adapter_config = PromptTuningConfig()
model.add_adapter("prompt_tuning", config=adapter_config)
```

Some relevant configuration parameters for prompt tuning include:

`prompt_length`

The length of the prompt tokens; 10–30 is a good start.

`prompt_init`

The method for initializing these tokens. They can be initialized either through the embedding of a string or by a random uniform initialization.

`prompt_init_text`

If the soft prompt is initialized by string, the text that is used to initialize it. This can be a descriptor of the task at hand.

[Lester et al.](https://oreil.ly/BPpRu), who introduced prompt-tuning, also leverage it to perform soft prompt ensembling. For soft prompt ensembling, you train several soft prompts for each task. Then, for a given input, you use each of them as a prefix separately and generate the output. You can then use majority voting to select the correct output among the generated ones.

So far, we have seen techniques where new parameters are added to the model for fine-tuning. However, we can implement PEFT by fine-tuning only a small subset of parameters of the model without having to add new parameters. Let’s explore these methods next.

## Subset Methods

A naive way of choosing a subset of parameters to fine-tune on would be to fine-tune only the upper layers of the Transformer and keep everything else frozen. The lower layers of the Transformer are known to be specialized in more fundamental aspects of language like syntax, which we want the LLM to preserve.

Another way is to fine-tune only the bias terms (discussed in [Chapter 2](ch02.html#ch02)) of the Transformer. This was proposed by [Zaken et al.](https://oreil.ly/SaWoe), who show that you can gain almost the same level of performance as that of fully fine-tuning a model by just fine-tuning on the bias terms. The authors observed that this technique is mostly effective when your training data is limited.

Ultimately, as we have seen here, there are tradeoffs involved in selecting each of these fine-tuning approaches. The ML community is working on developing best practices around this area. In the meanwhile, experimentation is key!

Next, let’s look at another way to update the parameters of an LLM: by merging it with the parameters of another LLM.

# Combining Multiple Models

If you have access to multiple LLMs, each of them overlapping in terms of capabilities yet possessing certain unique characteristics, you want to leverage the capabilities of all the models in your downstream tasks in some way. This can be done by a variety of means, including model ensembling and model fusion or merging. This area of LLMs is in its infancy, and more work remains to be done to reap its full benefits. I call it the dark arts of NLP because the theoretical underpinnings of these techniques remain poorly understood. However, I do believe that even with these caveats it merits inclusion in this book, because the practical benefits are already visible. Let’s explore a few of these methods.

## Model Ensembling

Different LLMs may possess different but complementary capabilities, a byproduct of the difference in their training regimens, training hyperparameters, etc. This is especially true when it comes to open source LLMs, where we have a plethora of models, most of them being trained on largely overlapping datasets, performing very closely to each other in benchmark evaluation metrics. Thus, an ensembling approach might bring forth benefits by allowing complementary capabilities from multiple models to be leveraged to generate better outputs.

In [Chapter 5](ch05.html#chapter_utilizing_llms), we discussed how, for generative tasks, it is useful to generate multiple outputs for the same input and select the best one using heuristics. We can extend this principle to multiple models. Each input is passed through *n* models. Optionally, an initial step can choose the top k models with the most high-quality or relevant outputs. The outputs from these models can be combined and fed through a model (which can be an LLM) to generate the final output.

[Jiang et al.](https://oreil.ly/Sipzu) present a framework called LLM-Blender for enabling LLM ensembling. The framework consists of two components:

*   PairRanker scores the output from two models, thus choosing a winner.

*   GenFuser takes as input the output from k different models to generate the final output.

[Figure 7-5](#LLMBlender) shows the workings of the LLM-Blender framework.

![LLMBlender](assets/dllm_0705.png)

###### Figure 7-5\. LLM-Blender

Let’s dig deeper into each of these modules.

### PairRanker

Consider you have access to *n* different models. For a given input, you feed the input to each of these models to generate the outputs. Now, for each pair of outputs, you can combine them with the input and feed them to the PairRanker module. The PairRanker module is trained to provide scores for each of the outputs. If you end up feeding all the pairs of outputs to the PairRanker module, you will then find the output (model) with the highest score. This output can then be taken as the final output.

However, this just selects the best output and doesn’t necessarily combine the capabilities of the different models. For that, the LLM-Blender framework consists of a module called GenFuser.

### GenFuser

For GenFuser, we take the top k results from the PairRanker scores. We then feed them together to the GenFuser, which generates the final output. The GenFuser in practice is just a fine-tuned LLM that is tuned to accept several candidate inputs and generate an output that combines the characteristics of the different candidates.

Let’s see how this works in practice. We can use the [LLM-Blender library](https://oreil.ly/F2IcX):

```py
import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks

ensemble = llm_blender.Blender()
ensemble.loadranker("llm-blender/PairRM")
ensemnle.loadfuser("llm-blender/gen_fuser_3b")

rank_list = blender.rank(input, candidate_outputs)
top_k = get_topk_candidates_from_ranks(rank_list, candidate_outputs, top_k=4)
final_output = ensemble.fuse(input, top_k)
```

Given an input and a list of `candidate_outputs` from *n* different language models, we rank the outputs using the PairRanker and then select the top-k ranked outputs and fuse them to generate the final output.

While ensembling methods can be effective, there is a lot of recent interest in model fusion techniques.

## Model Fusion

In this approach, we combine the parameters of multiple models in some way. The idea is that by combining the parameters of multiple models, we might be able to benefit from all the complementary capabilities possessed by each of the individual models, within a single model.

Some of the common methods used in model fusion are:

Averaging

The simplest way to combine multiple models is to average their parameters. Simple averaging has been shown to be quite effective.

Weighted averaging

During averaging, certain models or even certain layers in models can be weighted more.

Interpolation

Each model can be weighted by a factor w1, w2,…wn, with:

```py
w1 + w2 + w3 +...wn = 1
w1p1 + w2p2 + w3p3 +...wnpn
```

where p1, p2, p3…​pn are the parameters of models m1, m2, m3…mn.

One of the benefits in merging multiple models is model reuse. Say you have a base LLM at your organization. It is used by people all across the organization, who take the model and fine-tune it on their own tasks. They then upload the fine-tuned models back. You can then merge the weights of all the models, resulting in a stronger pre-trained model. This model can then be used as a new version of the base model. This process has been coined Collaborative Descent (ColD) Fusion by [Don-Yehiya et al.](https://oreil.ly/LTcdf)

Why would we want to do this? The idea is that if we want to fine-tune an LLM on a dataset, it would be nice to have a good starting point such that the training is optimal. The hypothesis is that if we already fine-tuned the LLM on another task, the fine-tuned LLM is a better starting point than the base LLM. This is called intertraining. This too is a fairly new concept, so proceed with caution.

Instead of merging all the parameters of the model, you can merge only a small portion of them. In fact, we could just merge the adapter modules.

## Adapter Merging

Earlier in the chapter, we learned about adapters, which can be used for a variety of purposes including domain-adaptive pre-training. While you can train different adapters for different domains, the question remains on how you would treat new domains seen at inference time. One solution would be to average the adapters related to the closest domains and use that for novel domains. This has been shown to work well, by [Chronopoulou et al.’s AdapterSoup framework](https://oreil.ly/mKoZ1).

Another way to combine adapter parameters is in the context of an MoE framework, introduced in [Chapter 4](ch04.html#chapter_transformer-architecture). Recall that in a mixture-of-experts model, the routing function determines which expert(s) will handle the input. [Wang et al.’s AdaMix framework](https://oreil.ly/pc7Js) extends this to adapter modules. Instead of learning only one adapter module per layer, we learn multiple expert modules. During inference, all the adaptation layers are merged.

Model merging is a fascinating subarea of LLMs. Even if you are not using it in your applications, I highly recommend experimenting with it because it doubles as a really neat tool to understand the working of LLMs.

# Summary

In this chapter, we learned a plethora of advanced fine-tuning techniques, including continual pre-training strategies like experience replay and parameter expansion; parameter-efficient fine-tuning techniques like bottleneck adapters, prefix tuning, prompt tuning, and subset selection; and various types of model merging and ensembling. We also learned the various motivations for updating model weights and the suitability of different methods for each of those situations.

As discussed in the previous and current chapter, fine-tuning is not a panacea and cannot learn new capabilities or necessarily digest new knowledge. In the next chapter, we will discuss limitations of LLMs like poor steerability, hallucinations, and reasoning issues, along with techniques for mitigating them.