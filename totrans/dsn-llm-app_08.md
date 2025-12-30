# Chapter 6\. Fine-Tuning

In the previous chapter, we discussed the various factors that need to be taken into account while choosing the right LLM for your specific needs, including pointers on how to evaluate LLMs to be able to make an informed choice. Next, let us utilize these LLMs to solve our tasks.

In this chapter, we will explore the process of adapting an LLM to solve your task of interest, using fine-tuning. We will go through a full example of fine-tuning, covering all the important decisions one needs to make. We will also discuss the art and science of creating fine-tuning datasets.

# The Need for Fine-Tuning

Why do we need to fine-tune LLMs? Why doesn’t a pre-trained LLM with few-shot prompts suffice for our needs? Let us look at a couple of examples to drive the point home:

Use Case 1

Consider you are working on the rather whimsical task of detecting all sentences written in the past tense within a body of text and transforming them to future tense. To solve this task, you might provide a few examples of past tense sentences and input-output pairs representing past tense and their corresponding future tense sentences. However, the LLM doesn’t seem to be able to tackle this task to your satisfaction, making mistakes in both the identification and transformation steps. In response, you elaborate on your instructions, adding grammar rules and exceptions in the English language into your prompt. You notice an increase in performance. But with each new rule added, your prompt balloons, slowly turning into a grammar mini-book.

As we saw in [Chapter 5](ch05.html#chapter_utilizing_llms), the LLM can adhere to only a finite set of instructions in the prompt, and its effective context window is much smaller than the advertised context window. We have hit an impasse.

Use Case 2

Consider a task that deals with answering questions from content in financial text. LLMs are not financial experts and have difficulty dealing with financial jargon. To address this, you add the definitions of key financial terms in the prompt. While you notice a small improvement in performance, it is not long before you realize you need to stuff the entire curriculum of the CPA exam into your measly context window to achieve the desired gains.

This is where fine-tuning comes in. By providing a dataset of input-output pairs, such that the model learns the input-output mapping by updating its weights, you can accomplish tasks that cannot be performed by in-context learning alone. For both the tasks mentioned above, fine-tuning the model massively improves performance.

When should fine-tuning not be used? If your primary goal is to impart new or updated facts or knowledge to the language model, this is better served with techniques like RAG, which we will explore in Chapters [10](ch10.html#ch10) and [12](ch12.html#ch12). Fine-tuning is best suited for situations where you need the model to learn a particular input-output mapping, be familiarized to a new textual domain, or exhibit more complex capabilities and behavior.

###### Warning

Recall from [Chapter 5](ch05.html#chapter_utilizing_llms) that updating a language model’s parameters can cause the base model capabilities to regress! Fine-tuning a model on one task can inadvertently cause the base model to perform worse on other tasks. Handle with care.

# Fine-Tuning: A Full Example

Let’s walk through a practical fine-tuning example from start to finish. We would like to train a *political promises detector*, which can be used to identify promises made by representatives of the ruling party in campaign speeches or parliamentary proceedings. We define a political promise as something that is tangible, specific, and an action that the government has the agency to make.

An example of such a sentence is: “We will build 10,000 kilometres of subway lines in the next ten years.”

However, not all future tense or forward-looking statements are promises. The following sentences are not promises, per our definition:

*   “We expect the Japanese to increase tariffs next year.” (expectation, and not something the government can control)
*   “We will work toward making Canada a better place.” (no specifics provided)
*   “AI will cause the loss of a million jobs next year.” (prediction, not promise)

Our base LLM, Llama2-7B, finds it difficult to accurately identify such promises in an in-context learning setup. Therefore, we will fine-tune it for this specific task. We can then use the resulting model to detect political promises, and then match those promises against structured datasets or budgetary text to track whether these promises have been fulfilled over a period of time.

To this end, I have constructed a synthetic fine-tuning dataset containing examples of both promises and mere statements. Later in this chapter, we will go through the process of creating such a dataset.

Fortunately, fine-tuning today is easier due to the existence of several libraries that streamline the fine-tuning process. The most important of these libraries are [Transformers](https://oreil.ly/BTi76), [Accelerate](https://oreil.ly/W8oLi), [PEFT](https://oreil.ly/QbQoq), [TRL](https://oreil.ly/Ya9Xj), and [bitsandbytes](https://oreil.ly/ruVEX). The first four are from Hugging Face. You have encountered many of these libraries in prior chapters already. Being familiar with the inner workings of these libraries is a very useful skill.

###### Tip

Given that these libraries are relatively new and are part of a fast-moving field, they frequently undergo substantial updates. I recommend keeping in touch with major updates of these libraries, as they continue to introduce enhancements that will simplify your workflow.

Let’s begin by loading the dataset. The custom dataset can be downloaded from this book’s [GitHub repo](https://oreil.ly/llm-playbooks):

```py
from datasets import load_dataset
tune_data = load_dataset("csv", data_files='/path/to/finetune_data.csv'
```

###### Tip

I highly recommend using the [*datasets* library](https://oreil.ly/3LX5X) for loading your training and fine-tuning datasets, as it is an excellent abstraction for efficiently loading large datasets, abstracting away memory management details.

Next, let us set some relevant hyperparameters in the `Transformers` library through the `TrainingArguments` class:

```py
# Make sure you have installed the correct version
!pip install transformers==4.35.0

from transformers import TrainingArguments
```

There are more than a hundred arguments available; we will go through the important ones. The arguments relate to the learning algorithms used, memory and space optimizations, quantization, regularization, and distributed training. Let’s explore these in detail.

## Learning Algorithms Parameters

Let’s explore optimization algorithms used for training the network and learn how to choose the right one for our purposes.

### Optimizers

AdamW and Adafactor are currently the most used optimizers. Other popular optimization algorithms include stochastic gradient descent (SGD), RMSProp, Adagrad, Lion, and their variants. For more background on optimization algorithms, refer to Florian June’s [blog post](https://oreil.ly/VTiDa).

Adafactor and SGD use four bytes of memory per parameter, while AdamW uses eight bytes per parameter. This means that a 7B model undergoing full fine-tuning with the AdamW optimizer requires 7 * 8 = ~56GB of memory to store the optimizer states alone. Even more memory is needed to store the parameters, gradients, and the forward activations.

More recently, [8-bit optimizers](https://oreil.ly/4Z14D) have been introduced that perform quantization of the optimizer state. A 7B model undergoing full fine-tuning with the AdamW 8-bit version requires only ~14GB of memory for the optimizer state.

These 8-bit optimizers are available through the bitsnbytes library and are also supported by Hugging Face. For using the 8-bit AdamW version, you can set in the `TrainingArguments`:

```py
optim = 'adamw_bnb_8bit'
```

For all the optimizer options directly available through Hugging Face, refer to the [OptimizerNames class](https://oreil.ly/7kdSO).

###### Tip

In his benchmarking experiments, Stas Bekman [shows](https://oreil.ly/0_0lt) that surprisingly, the 8-bit AdamW optimizer is actually faster than the standard AdamW optimizer. His experiments also show that Adafactor is slightly slower than AdamW overall.

The default optimizer provided in the Hugging Face `TrainingArguments` class is AdamW. For most cases, the default optimizer works just fine. However, if it doesn’t, you can try Adafactor and Lion. For reinforcement learning, SGD seems to work well.

If you are especially memory constrained, 8-bit AdamW is a compelling choice. If available, the paged version of these optimizers will further mitigate your memory requirements.

### Learning rates

For each optimizer, certain learning rates have been shown to be very effective. A recommended learning rate for AdamW is 1e-4 with a weight decay of 0.01\. Weight decay is a regularization technique that helps reduce overfitting. Similarly, the default values for minor optimizer parameters like *adam_beta1*, *adam_beta2*, and *adam_epsilon* are good enough and need not be changed.

### Learning schedules

Toward the end of the training process, it is a good idea to lower the learning rate because you do not want to overshoot when you are so close to convergence. In a similar vein, you would like to prevent your model from learning too much from the first few batches of examples. In either case, we would like to be able to automatically adjust the learning rate as training progresses. To facilitate this, we can use a learning schedule.

Hugging Face supports several different types of learning schedulers. Here are a few important ones:

Constant

This is the vanilla training schedule where the learning rate remains constant throughout the course of the training.

Constant with warmup

In this setting, the learning rate starts from zero and is increased linearly toward the specified learning rate during a warmup phase. After the warmup phase is completed, the learning rate remains constant.

[Figure 6-1](#constant-lr) shows how the learning rate changes over time while using the constant with warmup scheduler.

![constant-lr](assets/dllm_0601.png)

###### Figure 6-1\. Learning rate with a constant schedule with warmup

Cosine

In this setting, called *cosine annealing*, the learning rate has a warmup phase after which it slowly declines to zero, as per the cosine function.

[Figure 6-2](#cosine-warmup) shows how the learning rate changes over time while using the cosine scheduler.

![cosine-warmup](assets/dllm_0602.png)

###### Figure 6-2\. Learning rate with a cosine schedule

Cosine with restarts

In this setting, called *cosine annealing with warm restart*, after a warmup phase, the learning rate decreases to zero following the cosine function, but undergoes several hard restarts, where the learning rate shoots back to the specified learning rate after it reaches zero. For more details on why this is effective, check out Loshcilov and Hutter’s [paper](https://oreil.ly/Q4c3o) that introduced this concept.

[Figure 6-3](#cosine-restart) shows how the learning rate changes across time while using the cosine with restarts scheduler.

![cosine-restart](assets/dllm_0603.png)

###### Figure 6-3\. Learning rate with a cosine with restarts schedule

Linear

This is very similar to the cosine setting, except that the learning rate decreases to zero linearly instead of following the cosine function.

[Figure 6-4](#linear-lr) shows how the learning rate changes over time while using the linear scheduler.

![linear-lr](assets/dllm_0604.png)

###### Figure 6-4\. Learning rate with a linear scheduler

If you are using AdamW, schedulers with a warmup phase are even more important to prevent getting trapped in a bad minima. Empirically, it has been found that cosine annealing outperforms linear decay.

For our political promises detector fine-tuning, let’s use the paged variant of AdamW, a learning rate of 3e-4, a weight decay of 0.01, and the cosine learning schedule:

```py
optim = "paged_adamw_32bit"
learning_rate = 3e-4
weight_decay = 0.01
lr_scheduler_type = 'cosine'
warmup_ratio = 0.03  #The proportion of training steps to be used as warmup
```

## Memory Optimization Parameters

After we have set the parameters related to the optimizers, let’s explore memory and compute optimization parameters. Two prevalent techniques in this area include gradient checkpointing and gradient accumulation.

### Gradient checkpointing

Gradient checkpointing helps save memory at the cost of more compute. During the forward pass of the backpropagation algorithm, activations are computed and saved in memory so that they can be used in the backward pass. What if we did not save all of the activations? The missing activations could be recalculated on the fly during the backward pass. This does cost us more compute, but we could save a lot of memory. We could even train models where a batch size of only one does not fit in our GPU memory. For more technical details on gradient checkpointing, check out Yaroslav Bulatov’s [blog](https://oreil.ly/i-R4I).

### Gradient accumulation

Let’s say we have a desired batch size but we do not have the required memory to support that batch size. We can simulate the desired batch size using a technique called gradient accumulation. In this technique, the gradient updates are not done at every batch, but are accumulated over several batches and then summed or averaged.

###### Note

Gradient accumulation can make training slower, since there are fewer updates being made. Gradient accumulation does not reduce the computation required.

### Quantization

A very effective form of saving memory is through quantization, as introduced in [Chapter 5](ch05.html#chapter_utilizing_llms). We will go through quantization techniques in more detail in [Chapter 9](ch09.html#ch09). For our use case, we will use bf16 as it represents a sound tradeoff between memory savings and performance.

For our political promises detector fine-tuning, we’ll set the following parameters for memory optimization, given that we are trying to train it on a relatively memory constrained 16 GB RAM GPU:

```py
gradient_accumulation_steps = 4
bf16 = True
gradient_checkpointing = True
```

## Regularization Parameters

Next, let’s look at various techniques available for tackling model overfitting.

### Label smoothing

Label smoothing is a technique that not only helps with combatting overfitting but also aids in model calibration.

Calibration is an underappreciated topic in deep learning. A model is said to be well-calibrated if there is a correlation between its output probability values and task accuracy.

For example, consider a task that classifies a sentence as being abusive or not. If the model is well-calibrated, then among all examples for which the model produces an output probability of 0.9, 90% of them would be expected to be correctly classified. Similarly, for an output probability of 0.6, there should be a lower (~60%) likelihood of the classification being correct. Simply put, the output probability should accurately reflect the confidence in the classification decision.

A model being well-calibrated implies that it is not overconfident. This helps us in nuanced handling of examples that have low output probabilities (using a bigger model to handle those examples, for instance).

###### Note

Larger models are less calibrated compared to models like BERT, according to a study by [Li et al.](https://oreil.ly/ij7mS) Larger models tend to be more confident in general about their predictions. The inability to calculate reasonably accurate uncertainty estimates for large language models could be an argument to use smaller ones instead!

One of the techniques for calibrating models is label smoothing. The usual training process involves training against hard target labels (0 or 1 for a binary classification task). When using cross-entropy as the loss function, this amounts to pushing the model logits closer to 0 or 1, thus making the model highly confident. Label smoothing involves using a regularization term that is subtracted or divided from the hard target label.

Label smoothing is especially useful when the input dataset is noisy, i.e., contains some inaccurate labels. Regularization prevents the model from learning too much from incorrect examples.

For the political promises detector, we will use label smoothing, given that some examples could be subjective or open to interpretation.

### Noise Embeddings

The datasets we use for fine-tuning typically consist of a small number of examples (< 50,000). We would like our model to not overfit to the stylistic characteristics of the dataset, like the formatting, wording, and length of the text. One way to address this is by adding noise to the input embeddings.

[Jain et al.](https://oreil.ly/ouESL) observe that adding noise embeddings reduces the tendency of the model to overfit to wording and formatting of the fine-tuning datasets. An interesting side effect of noise embeddings is that the models generate longer, verbose texts. By measuring token diversity of the outputs, they confirmed that the longer texts actually include more information and are not just repetitive.

Hugging Face supports [Noisy Embedding Instruction Fine-Tuning (NEFTune)](https://oreil.ly/dSaem), a noise addition technique. In NEFTune, a noise vector is added to each embedding vector. The elements in the noise vector are generated by sampling independent and identically distributed (iid) from [-1,1]. The resulting vector is scaled using a scaling factor before being added to the embedding vector.

Noise embeddings have been empirically found to be very effective in reducing overfitting. Therefore, we will use it for our political promises detector fine-tuning. Note that the noise embeddings are added only during training and not during inference.

###### Warning

The impact of noise embeddings is not yet well understood. Improvements in the fine-tuning task could come at the cost of other model capabilities. Make sure you test the model for regressions!

For our political promises detector fine-tuning task, let’s activate both label smoothing and noise embeddings:

```py
# Label 0 will be transformed to label_smoothing_factor/num_labels
# Label 1 will be transformed to 1 - label_smoothing_factor +
#label_smoothing_factor/num_labels

label_smoothing_factor = 0.1
neftune_noise_alpha = 5
```

## Batch Size

Along with the learning rate, the batch size is one of the most important hyperparameters we need to set. A larger batch size means training will proceed faster. However, larger batch sizes also require more memory. Larger batch sizes can also lead the model to land in a sharp local minima, which can be a sign of overfitting. Therefore, there are trade offs involving memory, compute, and performance.

For the political promises detector, we will use a batch size of 8, given our memory limitations. Of course during inference, the maximum possible batch size is the ideal one. Note that it is recommended that the batch size be always a number that is a power of two, to reduce GPU I/O overhead.

The `TrainingArguments` class by Hugging Face supports *auto_find_batch_size*, which when set, selects the maximum possible batch size supported by the memory. To use this feature, you need to install the `accelerate` library:

```py
per_device_train_batch_size = 8
per_device_eval_batch_size = 8
```

###### Tip

You can reduce your maximum sequence length to support a larger batch size.

Finally, let’s set some miscellaneous parameters:

`max_grad_norm`

This is used for gradient clipping, which is a solution for the exploding gradients issue that is sometimes encountered during training. The `max_grad_norm` value is the threshold for gradient clipping. If the L2 gradient norm is above the threshold, then it will be rescaled to `max_grad_norm`. For more details on gradient clipping, see [“Understanding Gradient Clipping (and How It Can Fix Exploding Gradients Problem)”](https://oreil.ly/gH7L7).

`group_by_length`

This is used to group examples that have similar lengths in the same batch, so that the padding tokens can be optimized.

`max_train_epochs`

Number of passes over the training dataset. This is usually set to less than five to prevent overfitting:

```py
max_grad_norm=2
group_by_length=True
max_train_epochs=3
```

## Parameter-Efficient Fine-Tuning

After filling in the `TrainingArguments`, let’s next fill in parameters of the PEFT library.

The PEFT library by Hugging Face is an impressive facilitator of parameter-efficient fine-tuning. This refers to a set of fine-tuning techniques that update only a small proportion of parameters in the model while keeping the performance closer to what it would have been if all the parameters were updated.

In this example, we will use low-rank adaptation (LoRA) as the fine-tuning technique. Here are some hyperparameters to consider:

`r`

The attention dimension of LoRA.

`lora_alpha`

The alpha parameter in the LoRA technique.

`lora_dropout`

The dropout probability used in the layers being tuned. This helps reduce overfitting.

`layers_to_transform`

This specifies the layers for which the LoRA transformation is to be applied.

Here are some recommended default values:

```py
r = 64
lora_alpha = 8
lora_dropout = 0.1
```

For more background on LoRA, refer to Ogban Ugot’s [blog post](https://oreil.ly/_l91y).

## Working with Reduced Precision

The bitsandbytes library, built by Tim Dettmers, facilitates working with reduced precision formats, which we introduced in [Chapter 5](ch05.html#chapter_utilizing_llms). In this example, we will work with the FP4 format. Note that you need the bitsandbytes version to be >= 0.39.0.

Hugging Face has integrated bitsandbytes support into its ecosystem. The `BitsAndBytesConfig` class allows us to set the parameters. Here are some relevant ones:

`load_in_8bit/load_in_4bit`

This is used to specify if we want to load the model in 4-bit mode or 8-bit mode.

`llm_int8_threshold`

We need to specify a threshold of values beyond which fp16 will be used. This is because int8 quantization works well only for values lesser than 5–6.

`llm_int8_skip_modules`

This is used to specify the exceptions for which we do not want int8 quantization.

`llm_int8_enable_fp32_cpu_offload`

If we want parts of the model to be run in int8 on GPU and the rest in FP32 on CPU, this parameter facilitates it. This is used in cases where the model is too large to fit on our GPU.

`bnb_4bit_compute_dtype`

This sets the computational type, regardless of the input type.

`bnb_4bit_quant_type`

The options here are FP4 or NF4\. This is used to set the quantization type in the 4-bit layers.

Here are some recommended default values:

```py
use_4bit = True
bnb_4bit_compute_dtype = 'float16'
bnb_4bit_quant_type = 'nf4'
use_nested_quant = False
```

Finally, we use the Transformer Reinforcement Learning (TRL) library that, in addition to reinforcement learning, provides support for supervised fine-tuning.

Here are some recommended default values:

```py
max_seq_length = 128
# Packing is used to place multiple instructions in the same input sequence

packing = True
```

## Putting It All Together

Now that we have set up all the requisite parameters, here is the full code for the fine-tuning process:

```py
# Ensure that the specified versions of these libraries are installed.
!pip install transformers==4.35.0 accelerate==0.24.0 peft==0.6.0
bitsandbytes==0.41.0  trl==0.7.4

from datasets import load_dataset
from transformers import TrainingArguments, BitsAndBytesConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, LoraConfig
from trl import SFTTrainer

train_params = TrainingArguments(
    optim = "paged_adamw_32bit",
    learning_rate = 3e-4,
    weight_decay = 0.01,
    lr_scheduler_type = 'cosine',
    warmup_ratio = 0.03,
    gradient_accumulation_steps = 4,
    bf16 = True,
    gradient_checkpointing = True,
    label_smoothing_factor = 0.1,
    neftune_noise_alpha = 5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    max_grad_norm=2,
    group_by_length=True,
    max_train_epochs=3,
    output_dir = '/model_outputs',
    save_steps = 50,
    logging_steps = 10
    )

quantize_params = BitsAndBytesConfig (

    use_4bit = True,
    bnb_4bit_compute_dtype = 'float16',
    bnb_4bit_quant_type = 'nf4',
    use_nested_quant = False
    )

lora_params = LoraConfig (
    r = 64,
    lora_alpha = 8,
    lora_dropout = 0.1
    )

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path = 'meta-llama/Llama-2-7b',
    quantization_config=quantize_params,
    device_map='auto'
    )

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b')

tune_data = load_dataset("csv", data_files='/path/to/finetune_data.csv')

sft = SFTTrainer (
    model = model,
    args = train_params,
    train_dataset = tune_data,
    tokenizer = tokenizer
    peft_config = lora_params,
    max_seq_length = 128,
    dataset_text_field = 'text',
    packing = True
    )

sft.train()
sft.model.save_pretrained('/path/to/llama-2-it.csv')
```

The relationship between the hyperparameters is very complex, and you might find surprising results. It will take several iterations before you hit the sweet spot. However, do not spend too much time squeezing out the last bit of performance from your fine-tuning, as that time is better spent developing better training data. In the next section, we will learn how to create effective training datasets.

The exact memory you need to fine-tune an LLM depends on several factors: the optimizer used, whether gradient accumulation and gradient checkpointing are activated, the type of quantization used, etc.

# Fine-Tuning Datasets

In our fine-tuning example, we directly loaded a preconstructed dataset, focusing primarily on the fine-tuning process. Now, let’s shift our attention to the dataset, to understand the various techniques for creating datasets.

First, let’s look into the dataset we used in our fine-tuning example:

```py
from datasets import load_dataset
tune_data = load_dataset("csv", data_files='/path/to/finetune_data.csv')
print(tune_data[:2])
```

Output:

```py
Input: We will support women and children and give every child the best
possible start with $10 a day child care.
Identify if the above sentence represents a political promise. A political
promise is a promise that is tangible, specific, and an action that the
government has the agency to make. Reply 'True' if the sentence represents a
political promise, 'False' if not.
Output: True
Input: It is time for leadership that never seeks to divide Canadians, but
takes every single opportunity to bring us together, including in Parliament.
Identify if the above sentence represents a political promise. A political
promise is a promise that is tangible, specific, and an action that the
government has the agency to make. Reply 'True' if the sentence represents a
political promise, 'False' if not.
Output: False
```

As we can see, this is not a traditional dataset with just (input, output) pairs but one that also contains the task description in natural language. A typical example in this type of fine-tuning dataset consists of :

*   The instruction, which describes the task and specifies the desired output format. Optionally, the instruction contains positive and/or negative examples of the task. It can also contain constraints and exceptions to be followed.

*   An optional input, which in our example is the sentence or paragraph for the model to evaluate.

*   The output, which is the correct answer to the task in the format specified in the instruction.

###### Note

Fine-tuning datasets can be either multi-task or single-task. Multi-task datasets are used for instruction-tuning. In general, instruction-tuning can be treated as an intermediate step before single-task fine-tuning. For example, you can take a T5 language model, instruction-tune it with FLAN to create FLAN-T5, and then further fine-tune it with your task-specific dataset. This approach is [shown](https://oreil.ly/e-MVh) to yield better results than directly fine-tuning on T5 alone.

Later in this chapter, we will learn how to create task-specific datasets. First, let’s look at how we can create instruction-tuning datasets.

There are plenty of instruction-tuned LLMs available, both open source and proprietary. Why do we want to instruction-tune the LLM ourselves? Public datasets are too general, lack diversity, and are primarily geared to general usage. Leveraging your domain expertise and knowledge of intended use cases to construct the dataset can be highly effective. In fact, at my company, which specializes in the financial domain, this technique delivered the single largest boost in performance.

Approaches to creating instruction-tuning datasets include:

*   Utilizing publicly available instruction-tuning datasets

*   Transforming traditional fine-tuning datasets into instruction-tuning datasets

*   Starting with manually crafted seed examples, followed by optionally augmenting the dataset by utilizing an LLM to generate similar examples

Next, let’s examine these methods more closely.

## Utilizing Publicly Available Instruction-Tuning Datasets

If your use case is sufficiently general or popular, you may be able to use publicly available datasets for instruction-tuning. The following table lists some popular instruction-tuning datasets, along with information on their creators, sizes, and creation process.

Table 6-1\. Popular instruction-tuning datasets

| Name | Size | Created by | Created using |
| --- | --- | --- | --- |
| OIG | 43M | [Ontocord](https://www.ontocord.ai/) | Rule-based  |
| FLAN | 4.4M | Google | Templates |
| P3 (Public Pool of Prompts) | 12M | Big Science | Templates |
| Natural Instruction | 193K | Allen AI | Templates |
| Unnatural Instructions | 240K | [Honovich et al.](https://github.com/orhonovich/unnatural-instructions), Meta | LLMs |
| LIMA (Less Is More for Alignment) | 1K | [Zhou et al.](https://arxiv.org/abs/2305.11206), Meta | Templates |
| Self-Instruct | 52K | [Wang et al.](https://github.com/yizhongw/self-instruct) | LLMs |
| Evol-Instruct | 52K | [Xu et al.](https://arxiv.org/abs/2304.12244) | LLMs |
| InstructWild v2 | 110K | [Ni et al.](https://github.com/XueFuzhao/InstructionWild) | LLMs |
| Alpaca | 52K | Stanford | LLMs |
| Guanaco | 534K | [Dettmers et al.](https://arxiv.org/abs/2305.14314) | LLMs |
| Vicuna | 70K | LMSYS | Human conversations |
| OpenAssistant | 161K | Open Assistant | Human conversations |

Let’s go through fine-tuned language net (FLAN), one of the most popular instruction-tuning datasets in detail. Understanding how it was constructed will provide you with roadmaps to create your own instruction-tuning datasets. Most publicly available instruction-tuning datasets are meant to augment an LLM that will be used for open-ended tasks, as opposed to domain-specific use cases.

FLAN is actually a collection of several datasets. The [FLAN collection](https://oreil.ly/SrXV-), published in 2022, is composed of five components:

*   FLAN 2021

*   T0

*   Super-natural Instructions

*   Chain-of-Thought

*   Dialog

The original FLAN 2021 datasets were one of the pioneering instruction-tuning datasets, which were used to train FLAN-T5\. The FLAN 2021 datasets were constructed by taking existing academic NLP datasets and converting them to the instruction format using instruction templates. The templates were manually constructed, with ten templates created for each task. The templates are available [here](https://oreil.ly/DNKCv).

Here is how a template list for a task looks, as drawn from the [templates.py](https://oreil.ly/DNKCv) file in the FLAN GitHub repo. Our example task is text summarization on the CNN/DailyMail news dataset:

```py
"cnn_dailymail": [
  ("Write highlights for this article:\n\n{text}", "{highlights}"),
  ("Write some highlights for the following article:\n\n{text}", "{highlights}"),
  ("{text}\n\nWrite highlights for this article.", "{highlights}"),
  ("{text}\n\nWhat are highlight points for this article?", "{highlights}"),
  ("{text}\nSummarize the highlights of this article.", "{highlights}"),
  ("{text}\nWhat are the important parts of this article?", "{highlights}"),
  ("{text}\nHere is a summary of the highlights for this article:",
    "{highlights}"),
  ("Write an article using the following points:\n\n{highlights}", "{text}"),
  ("Use the following highlights to write an article:\n\n{highlights}",
    "{text}"),
  ("{highlights}\n\nWrite an article based on these highlights.", "{text}"),
],
```

Note that the last three instructions represent an inverted version of the task, where given a summary, the model is encouraged to write the entire article. This has been done to increase the diversity of the instructions at scale.

Rather than painstakingly constructing these templates by hand, can we automate their generation using LLMs? Yes, this is possible. We can leverage LLMs to generate more diverse templates. When I asked my favorite LLM to generate similar instructions to a news summarization task template provided in the prompt, it came up with:

```py
"cnn_dailymail": [
  ("Distill the essence of this article:\n\n{text}", "{highlights}"),
  ("Give a quick rundown of this article's key points:\n\n{text}",
    "{highlights}"),
  ("Summarize the main elements of this text:\n\n{text}", "{highlights}"),
  ("Highlight the primary takeaways from the following:\n\n{text}",
    "{highlights}"),
  ("Extract and summarize the top points of this article:\n\n{text}",
    "{highlights}"),
  ("Condense this article into its most important aspects:\n\n{text}",
    "{highlights}"),
  ("What are the key insights of this article?\n\n{text}", "{highlights}"),
      ],
```

As you can see, the generated templates reflect various ways of expressing the summarization task.

For classification tasks, it is recommended to append the instruction with an *Options* clause. This introduces the LLM to the output space and can thus concentrate the probability mass over the defined label space. Without this guidance, the LLM would distribute its probability across several different tokens that express the same concept, for example there are several different ways of expressing the *True* label in a binary classification task. An example prompt is: “Identify the tone of this text. OPTIONS: happy, sad, neutral.”

Constructing these prompts manually can be a tedious exercise. The [*promptsource* tool](https://oreil.ly/WIyOq) enables you to create, access, and apply prompts through a graphical user interface tool or through the promptsource Python library. Here is an example from the Public Pool of Prompts (P3) collection for the paraphrasing task, constructed by Big Science, which is available through the promptsource tool. P3 prompts consist of an Input template, a Target template, and an Answer Choices template:

```py
Input Template:
I want to know whether the following two sentences mean the same thing.
{{sentence1}}
{{sentence2}}
Do they?

Target Template:
{{ answer_choices[label] }}

Answer Choices Template:
no ||| yes
```

Another key component of the FLAN collection is the [Super-NaturalInstructions dataset](https://oreil.ly/D_rv_). This dataset contains very rich descriptions of instructions that contain not just task definitions, but also positive and negative examples, constraints, and things to watch out for. The answers are enriched with explanations on why the answer was chosen. The effectiveness of adding explanations to the answer is not yet determined.

Here is an example of such a task from the Super-NaturalInstructions dataset:

```py
Definition
In this task, we ask you convert a data table of restaurant descriptions into
fluent natural-sounding English sentences.
The input is a string of key-value pairs; the output should be a natural and
grammatical English sentence containing all the information from the input.

Positive Example

Input: name[Aromi], eatType[restaurant], food[English], area[city centre]

Output: Aromi is an English restaurant in the city centre.
Explanation: The output sentence faithfully converts the data in the input
into a natural-sounding sentence.

Negative Example
Input: name[Blue Spice], eatType[coffee shop], priceRange[more than 00a330],
customer rating[5 out of 5], ˘
area[riverside], familyFriendly[yes], near[Avalon]
Output: Blue Spice is a Colombian coffee shop located by the riverside, near
Avalon in Boston. Its prices are over
00a330. Its customer ratings are 5 out of 5. ˘

Explanation: While the output contains most of the information from the input,
it hallucinates by adding ungrounded
information such as "Colombian" and "Boston".

Instance Input: name[The Mill], eatType[restaurant], area[riverside], near[The
Rice Boat]

Valid Output: ["A restaurant called The Mill, can be found near the riverside
next to The Rice Boat."]
```

Let’s now look at datasets that are constructed with the help of LLMs.

## LLM-Generated Instruction-Tuning Datasets

As seen earlier, hand-constructing these datasets can be painstaking, and paraphrasing/synthetic data generation is where LLMs shine. Therefore, we can leverage LLMs to generate our instruction-tuning datasets. The [Self-Instruct](https://oreil.ly/HVBfK) and [Unnatural Instructions papers](https://oreil.ly/1wV_G) are the first attempts in this regard. Both start from a seed set of high-quality hand-generated examples, and then in a few-shot setting, ask the LLM to generate similar examples with more diverse linguistic expressions.

Given an instruction, a combination of input-first and output-first is shown to be beneficial for generating input-output pairs. Typically, you would generate input-output pairs using an input-first approach, where the LLM is asked to generate an input instance for the given instruction and subsequently asked to generate the output label for that input. However, this approach might lead to label imbalance as shown in [Wang et al.](https://oreil.ly/hYFYH), with certain labels being overrepresented. Therefore, it is a good approach to mix output-first generation, where you ask the LLM to generate the output label first and then ask it to generate an input text that satisfies the label.

###### Warning

It is against OpenAI’s policies to use its outputs to generate data that can be used to train a competing model. While there are several public instruction-tuning datasets that have been synthetically generated using GPT-4, they are technically violating OpenAI’s terms of service. I recommend using open source LLMs for synthetic data generation instead.

Simply asking an LLM to generate similar examples to your seed set may not give you the desired results. You want a diverse but relevant set of examples, and it is easy for your LLM to drift into territory that ends up generating spurious examples outside of your desired distribution.

###### Note

How large should your instruction-tuning dataset be? The [“LIMA: Less Is More for Alignment” paper](https://oreil.ly/z0BWh) shows that you need only a few thousand high-quality examples to effectively fine-tune a model.

Xu et al. propose [Evol-Instruct](https://oreil.ly/9nw3G), a structured way to generate these synthetic instructions by making controlled edits to the seed examples. The process consists of three steps:

1.  Instruction evolution: The seed examples are evolved using in-depth and in-breadth strategies. In-depth evolution increases the complexity and difficulty of the original instruction through five types of prompts:

    *   Adding constraints

    *   Increasing reasoning steps

    *   Asking deeper questions

    *   Asking more specific questions

    *   Increasing the complexity of the input

        In-breadth evolution increases topic coverage by generating a completely new instruction from the same domain as the original instruction.

2.  Response generation: The response for the evolved instruction is generated, either using humans or LLMs.

3.  Candidate filtering: Candidate instances that do not meet quality criteria are filtered out. You could use either heuristics or LLMs for candidate filtering.

###### Note

Why not pre-train on instruction-tuning datasets? If instruction-tuning is a necessary step after pre-training a model, why don’t we just pre-train the model using an instruction-tuning dataset? It is indeed possible, but these datasets are hard to construct at scale without incurring a significant drop in quality.

We need not wait until someone releases a massive dataset to reap the benefits of instruction-tuning during the pre-training phase. It has been [shown](https://oreil.ly/tfO4a) that mixing instruction-tuning data during pre-training is beneficial.

# Summary

In this chapter, we underscored the inevitability of needing to fine-tune models to solve more complex tasks. We performed a deep dive of the fine-tuning process and highlighted the tradeoffs involved in selecting hyperparameters. We also showed the uncanny effectiveness of instruction-tuning along with pointers on how to create your own instruction-tuning datasets.

In the next chapter, we will discuss more advanced techniques for updating an LLM’s parameters, including continual pre-training, parameter efficient fine-tuning, and model merging.