# Chapter 4\. Architectures and Learning Objectives

In Chapters [2](ch02.html#ch02) and [3](ch03.html#chapter-LLM-tokenization), we discussed some of the key ingredients that go into making a language model: the training datasets, and the vocabulary and tokenizer. Next, let’s complete the puzzle by learning about the models themselves, the architectures underpinning them, and their learning objectives.

In this chapter, we will learn the composition of language models and their structure. Modern-day language models are predominantly based on the Transformer architecture, and hence we will devote most of our focus to understanding it, by going through each component of the architecture in detail. Over the last few years, several variants and alternatives to the original Transformer architecture have been proposed. We will go through the promising ones, including Mixture of Experts (MoE) models. We will also examine commonly used learning objectives the language models are trained over, including next-token prediction. Finally, we will bring together the concepts of the last three chapters in practice by learning how to pre-train a language model from scratch.

# Preliminaries

Just about every contemporary language model is based on neural networks, composed of processing units called *neurons*. While modern neural networks do not resemble the workings of the human brain at all, many of the ideas behind neural networks and the terminology used is inspired by the field of neuroscience.

The neurons in a neural network are connected to each other according to some configuration. Each connection between a pair of neurons is associated with a weight (also called *parameter*), indicating the strength of the connection. The role these neurons play and the way they are connected to each other constitutes the *architecture* of the model.

The early 2010s saw the proliferation of multi-layer architectures, with layers of neurons stacked on top of each other, each layer extracting progressively more complex features of the input. This paradigm is called *deep learning.*

[Figure 4-1](#MLP-00) depicts a simple multi-layer neural network, also called the multi-layer perceptron.

![Transformer](assets/dllm_0401.png)

###### Figure 4-1\. Multi-layer perceptron

###### Tip

For a more comprehensive treatment of neural networks, refer to [Goldberg’s book](https://oreil.ly/oDc6x) on neural network–based natural language processing.

As discussed in [Chapter 1](ch01.html#chapter_llm-introduction), language models are primarily pre-trained using self-supervised learning. Input text from the training dataset is tokenized and converted to vector form. The input is then propagated through the neural network, affected by its weights and *activation functions*, the latter introducing nonlinearity to the model. The output of the model is compared to the expected output, called the gold truth. The weights of the output are adapted such that next time for the same input, the output can be closer to the gold truth.

In practice, this adaptation process is implemented through a *loss function*. The goal of the model is to minimize the loss, which is the difference between the model output and the gold truth. To minimize the loss, the weights are updated using a gradient-descent based method, called backpropagation. I strongly recommend developing an intuitive understanding of this algorithm before diving into model training.

# Representing Meaning

While describing neural network–based architectures in the previous section, we glossed over the fact that the input text is converted into vectors and then propagated through the network. What are these vectors composed of and what do they represent? Ideally, after the model is trained, these vectors should accurately represent some aspect of the meaning of the underlying text, including its social connotations. Developing the right representations for modalities like text or images is a very active field of research, called *representation learning*.

###### Note

When training a language model from scratch, these vectors initially mean nothing, as they are randomly generated. In practice, there are initialization algorithms used like Glorot, He, etc. Refer to [this report](https://oreil.ly/A8Iro) for a primer on neural network initialization.

How can a list of numbers represent meaning? It is hard for humans to describe the meaning of a word or sentence, let alone represent it in numerical form that can be processed by a computer. The *form* of a word, i.e., the letters that comprise it, usually do not give any information whatsoever about the meaning it represents. For example, the sequence of letters in the word *umbrella* contains no hints about its meaning, even if you are already exposed to thousands of other words in the English language.

The prominent way of representing meaning in numerical form is through the *distributional hypothesis* framework. The distributional hypothesis states that words that have similar meaning occur in similar contexts. The implication of this hypothesis is best represented by the adage:

> You shall know a word by the company it keeps.
> 
> John Rupert Firth, 1957

This is one of the primary ways in which we pick up the meaning of words we haven’t encountered previously, without needing to look them up in a dictionary. A large number of words we know weren’t learned from the dictionary or by explicitly learning the meaning of a word but by estimating meaning based on the contexts words appear in.

Let’s investigate how the distributional hypothesis works in practice. The Natural Language Toolkit (NLTK) library provides a feature called *concordance view*, which presents you with the surrounding contexts that a given word appears in a corpus.

For example, let’s see the contexts in which the word “nervous” occurs in the Jane Austen classic *Emma*:

```py
from nltk.corpus import gutenberg
from nltk.text import Text
corpus = gutenberg.words('austen-emma.txt')
text = Text(corpus)
text.concordance("nervous")
```

The output looks like this:

```py
Displaying 11 of 11 matches:
...spirits required support . He was a nervous man , easily depressed...
...sitting for his picture made him so nervous , that I could only take...
...assure you , excepting those little nervous headaches and palpitations...
...My visit was of use to the nervous part of her complaint , I hope...
...much at ease on the subject as his nervous constitution allowed...
...Her father was growing nervous , and could not understand her....
...
```

# The Transformer Architecture

Now that we have developed an intuition on how text is represented in vector form, let’s further explore the canonical architecture used for training language models today, the Transformer.

In the mid 2010s, the predominant architectures used for NLP tasks were recurrent neural networks, specifically a variant called long short-term memory (LSTM). While knowledge of recurrent neural networks is not a prerequisite for this book, I recommend [*Neural Network Methods for Natural Language Processing*](https://oreil.ly/CHCTd) for more details.

Recurrent neural networks were sequence models, which means they processed text one token at a time, sequentially. A single vector was used to represent the state of the entire sequence, so as the sequence grew longer, more and more information needed to be captured in the single state vector. Because of the sequential nature of processing, long-range dependencies were harder to capture, as the content from the beginning of the sequence would be harder to retain.

This issue was candidly articulated by Ray Mooney, a senior computer scientist who remarked at the Association for Computational Linguistics (ACL) 2014 conference:

> You can’t cram the meaning of a whole %&!$# sentence into a single $&!#* vector!
> 
> Ray Mooney, 2014

Thus, there was a need for an architecture that solved for the deficiencies of LSTM: the limitations in representing long-range dependencies, the dependence on a single vector for representing the state of the entire sequence, and more. The Transformer architecture was designed to address these issues.

[Figure 4-2](#Transformer0) depicts the original Transformer architecture, developed in 2017 by [Vaswani et al.](https://oreil.ly/tIvGZ) As we can see in the figure, a Transformer model is typically composed of Transformer blocks stacked on top of each other, called *layers*. The key components of each block are:

*   Self-attention

*   Positional encoding

*   Feedforward networks

*   Normalization blocks

![Transformer](assets/dllm_0402.png)

###### Figure 4-2\. The Transformer architecture

At the beginning of the first block is a special layer called the *embedding* layer. This is where the tokens in the input text are mapped to their corresponding vector. The embedding layer is a matrix whose size is:

```py
Number of tokens in the vocabulary * The vector dimension size
```

On Hugging Face, we can inspect the embedding layer as such, using the `transformers` library:

```py
import torch
from transformers import LlamaTokenizer, LlamaModel

tokenizer = LlamaTokenizer.from_pretrained('llama3-base')
model = LlamaModel.from_pretrained('llama3-base')

sentence = "He ate it all"

inputs = tokenizer(sentence, return_tensors="pt")
input_ids = inputs['input_ids']
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

with torch.no_grad():
    embeddings = model.embeddings(input_ids)

for token, embedding in zip(tokens, embeddings[0]):
    print(f"Token: {token}\n
    print(f"Embedding: {embedding}\n")
```

The embedding vectors are the inputs that are then propagated through the rest of the network.

Next, let’s go through each of the components in a Transformer block in detail and explore their role in the modeling process.

## Self-Attention

The self-attention mechanism draws on the same principle as the distributional hypothesis introduced in [“Representing Meaning”](#representing-meaning), emphasizing the role of context in shaping the meaning of a token. This operation generates representations for each token in a text sequence, capturing various aspects of language like syntax, semantics, and even pragmatics.

In the standard self-attention implementation, the representation of each token is a function of the representation of all other tokens in the sequence. Given a token for which we are calculating its representation, tokens in the sequence that contribute more to the meaning of the token are given more weight.

For example, consider the sequence:

```py
'Mark told Sam that he was planning to resign.'
```

[Figure 4-3](#Attention-map) depicts how the representation for the token *he* is heavily weighted by the representation of the token *Mark*. In this case, the token *he* is a pronoun used to describe Mark in shorthand. In NLP, mapping a pronoun to its referent is called *co-reference resolution*.

![Attention-map](assets/dllm_0403.png)

###### Figure 4-3\. Attention map

In practice, self-attention in the Transformer is calculated using three sets of weight matrices called queries, keys, and values. Let’s go through them in detail. [Figure 4-4](#kqv) shows how the query, key, and value matrices are used in the self-attention calculation.

Each token is represented by its embedding vector. This vector is multiplied with the query, key, and value weight matrices to generate three input vectors. Self-attention for each token is then calculated like this:

1.  For each token, the dot products of its query vector with the key vectors of all the tokens (including itself) are taken. The resulting values are called attention scores.

2.  The scores are scaled down by dividing them by the square root of the dimension of the key vectors.

3.  The scores are then passed through a [*softmax function*](https://oreil.ly/b6gHV) to turn them into a probability distribution that sums to 1\. The softmax activation function tends to amplify larger values, hence the reason for scaling down the attention scores in the previous step.

4.  The normalized attention scores are then multiplied by the value vector for the corresponding token. The normalized attention score can be interpreted as the proportion that each token contributes to the representation of a given token.

5.  In practice, there are multiple sets of query, key, and value vectors, calculating parallel representations. This is called multi-headed attention. The idea behind using multiple heads is that the model gets sufficient capacity to model various aspects of the input. The more the number of heads, the more chances that the *right* aspects of the input are being represented.

![kqv](assets/dllm_0404.png)

###### Figure 4-4\. Self-attention calculation

This is how we implement self-attention in code:

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

q = wQ(input_embeddings)
k = WK(input_embeddings)
v = WV(input_embeddings)
dim_k = k.size(-1)

attn_scores = torch.matmul(q, k.transpose(-2, -1))
scaled_attn_scores = attn_scores/torch.sqrt(torch.tensor(dim_k,
  dtype=torch.float32))

normalized_attn_scores = F.softmax(scaled_attn_scores, dim=-1)

output = torch.matmul(normalized_attn_scores, v)
```

###### Note

In some Transformer variants, self-attention is calculated only on a subset of tokens in the sequence; thus the vector representation of a token is a function of the representations of only some and not all the tokens in the sequence.

## Positional Encoding

As discussed earlier, pre-Transformer architectures like LSTM were sequence models, with tokens being processed one after the other. Thus the positional information about the tokens, i.e., the relative positions of the tokens in a sequence, was implicitly baked into the model. However, for Transformers all calculations are done in parallel, and positional information should be fed to the model explicitly. Several methods have been proposed to add positional information, and this is still a very active field of research. Some of the common methods used in LLMs today include:

Absolute positional embeddings

These were used in the original Transformer implementation by [Vaswani et al.](https://oreil.ly/CDq60); examples of models using absolute positional embeddings include earlier models like BERT and RoBERTa.

Attention with Linear Biases (ALiBi)

In this technique, the attention scores are [penalized](https://arxiv.org/abs/2108.12409) with a bias term proportional to the distance between the query token and the key token. This technique also enables modeling sequences of longer length during inference than what was encountered in the training process.

Rotary Position Embedding (RoPE)

Just like ALiBi, this [technique](https://arxiv.org/abs/2104.09864) has the property of relative decay; there is a decay in the attention scores as the distance between the query token and the key token increases.

No Positional Encoding (NoPE)

A contrarian [technique](https://oreil.ly/QM9dW) argues that positional embeddings in fact are not required and that Transformers implicitly capture positional information.

Models these days are mostly using ALiBi or RoPE, although this is one aspect of the Transformer architecture that is still actively improving.

## Feedforward Networks

The output from a self-attention block is fed through a [*feedforward network*](https://oreil.ly/Bdphg). Each token representation is independently fed through the network. The feedforward network incorporates a nonlinear activation function like [Rectified Linear Unit (ReLU)](https://oreil.ly/KUqtP) or [Gaussian Error Linear Units (GELU)](https://oreil.ly/MSDKE), thus enabling the model to learn more complex features from the data. For more details on these activation functions, refer to this [blog post from v7](https://oreil.ly/NfOb0).

The feedforward layers are implemented in code in this way:

```py
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, input_dim)
        self.selu = nn.SeLU()

    def forward(self, x):
        x = self.selu(self.l1(x))
        x = self.l2(x)
        return x

feed_forward = FeedForward(input_dim, hidden_dim)
outputs = feed_forward(inputs)
```

## Layer Normalization

Layer normalization is performed to ensure training stability and faster training convergence. While the original Transformer architecture performed normalization at the beginning of the block, modern implementations do it at the end of the block. The normalization is performed as follows:

1.  Given an input of batch size `b`, sequence length `n`, and vector dimension `d`, calculate the mean and variance across each vector dimension.

2.  Normalize the input by subtracting the mean and dividing it by the square root of the variance. A small epsilon value is added to the denominator for numerical stability.

3.  Multiply by a scale parameter and add a shift parameter to the resulting values. These parameters are learned during the training process.

This is how it is represented in code:

```py
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dimension, gamma=None, beta=None, epsilon=1e-5):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.gamma = gamma if gamma is not None else
        nn.Parameter(torch.ones(dimension))
        self.beta = beta if beta is not None else
        nn.Parameter(torch.zeros(dimension))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        return self.gamma * x_normalized + self.beta

layer_norm = LayerNorm(embedding_dim)
outputs = layer_norm(inputs)
```

# Loss Functions

So far, we have discussed all the components of each Transformer block. For the next token-prediction learning objective, the input is propagated through the Transformer layers to generate the final output, which is a probability distribution across all tokens. During training, the loss is calculated by comparing the output distribution with the gold truth. The gold truth distribution assigns a 1 to the gold truth token and 0 to all other tokens.

There are many possible ways to quantify the difference between the output and the gold truth. The most popular one is cross-entropy, which is calculated by the formula:

```py
Cross-Entropy= −∑(gold truth probability)×log(output probability)
```

For example, consider the sequence:

```py
'His pizza tasted ______'
```

Let’s say the gold truth token is *good*, and the output probability distribution is (*terrible*: 0.65, *bad*:0.12, *good*:011,…​)

The cross-entropy is calculated as:

```py
−(0×log(0.65)+0×log(0.12)+1×log(0.11)+...)= −log(0.11)
```

Since the gold truth distribution values are 0 for all but the correct token, the equation can be simplified to:

```py
Cross-Entropy = -log(output probability of gold truth token)
```

Once the loss is calculated, the gradient of the loss with respect to the parameters of the model is calculated and the weights are updated, using the backpropagation algorithm.

# Intrinsic Model Evaluation

How do we know if the backpropagation algorithm is actually working and that the model is getting better over time? We can use either intrinsic model evaluation or extrinsic model evaluation.

Extrinsic model evaluation involves testing the model’s performance on real-world downstream tasks. These tasks directly test the performance of the model but only on a narrow range of the model’s capabilities. In contrast, intrinsic model evaluation involves a more general evaluation of the model’s ability to model language, but with no guarantee that its performance in the intrinsic evaluation metric is directly proportional to its performance across all possible downstream tasks.

The most common intrinsic evaluation metric is *perplexity*. Perplexity measures the ability of a language model to accurately predict the next token in a sequence. A model that can always correctly predict the next token has a perplexity of 1\. The higher the perplexity, the worse the language model. In the worst case, if the model is predicting at random, with probability of predicting each token in a vocabulary of size V being 1/V, then the perplexity is V.

Perplexity is related to cross-entropy by this formula:

```py
Perplexity = 2^Cross-Entropy
```

# Transformer Backbones

So far, we described the components of the canonical version of the Transformer. In practice, three major types of architecture backbones are used to implement the Transformer:

*   Encoder-only

*   Encoder-decoder

*   Decoder-only

Let’s look at each of these in detail.

[Figure 4-5](#enc-dec) depicts encoder-only, encoder-decoder, and decoder-only architectures.

![enc-dec](assets/dllm_0405.png)

###### Figure 4-5\. Visualization of various Transformer backbones

## Encoder-Only Architectures

Encoder-only architectures were all the rage when Transformer-based language models first burst on the scene. Iconic language models from yesteryear (circa 2018) that use encoder-only architectures include BERT, RoBERTa, etc.

There haven’t been many encoder-only LLMs trained since 2021 for a few reasons, including:

*   They are relatively harder to train.

*   The masked language modeling objective typically used to train them provides a learning signal in only a small percentage of tokens (the masking rate), thus needing a lot more data to reach the same level of performance as decoder-only models.

*   For every downstream task, you need to train a separate task-specific head, making usage inefficient.

However, the release of ModernBERT seems to have reinvigorated this space.

The creators of the UL2 language model claim that encoder-only models should be considered obsolete. I personally wouldn’t go that far; encoder-only models are still great choices for classification tasks. Moreover, if you already have a satisfactory pipeline for your use case built around encoder-only models, I would say if it ain’t broke, why fix it?

Here are some guidelines for adopting encoder-only models:

*   RoBERTa performs better than BERT most of the time, since it is trained a lot longer on more data, and it has adopted best practices learned after the release of BERT.

*   DeBERTa and ModernBERT are currently regarded as the best-performing encoder-only models.

*   The distilled versions of encoder-only models like DistilBERT, etc., are not too far off from the original models in terms of performance, and they should be considered if you are operating under resource constraints.

Several embedding models are built from encoder-only models. For example, one of the most important libraries in the field of NLP, the Swiss Army knife of NLP tools, *sentence transformers*, provides encoder-only embedding models that are very widely used. all-mpnet-base-v2, based on an encoder-only model called MPNet, and fine-tuned on several task datasets, is still competitive with much larger embedding models.

## Encoder-Decoder Architectures

This is the original architecture of the Transformer, as it was first proposed. The T5 series of models uses this architecture type.

In encoder-decoder models, the input is text and the output is also text. A standardized interface ensures that the same model and training procedure can be used for multiple tasks. The inputs are handled by an encoder, and the outputs by the decoder.

## Decoder-Only Architectures

A majority of LLMs trained today use decoder-only models. Decoder-only models came into fashion starting from the original GPT model from OpenAI. Decoder-only models excel at zero-shot and few-shot learning.

Decoder models can be causal and noncausal. Noncausal models have bidirectionality over the input sequence, while the output is still autoregressive (you cannot look ahead).

###### Tip

While the field is still evolving, there has been some [compelling evidence](https://oreil.ly/Sb7JS) for the following results:

*   Decoder-only models are the best choice for zero-shot and few-shot generalization.

*   Encoder-decoder models are the best choice for multi-task fine tuning.

The best of both worlds is to combine the two: start with auto-regressive training, and then in an adaptation step, pre-train further with a noncausal setup using a span corruption objective.

In this section, we discussed how architectural backbones can be classified according to how they use the architecture’s encoder and decoder. Another architectural backbone type that is making inroads in the past year is the Mixture of Experts (MoE) paradigm. Let’s explore that in detail.

## Mixture of Experts

Remarkably, in the seven years since the invention of the Transformer architecture, the Transformer implementation used in current language models isn’t too different from the original version, despite hundreds of papers proposing modifications to it. The original architecture has proven to be surprisingly robust, with most proposed variants barely moving the needle in terms of performance. However, some components of the Transformer have seen changes, like positional encodings as discussed earlier in the chapter.

MoE models have been seeing a lot of success in the past couple of years. Examples include OpenAI’s GPT-4 (unconfirmed), Google’s Switch, DeepSeek’s DeepSeek V3, and Mistral’s Mixtral. In this section, we will learn the motivations behind developing this architecture and how it works in practice.

As shown in [Chapter 1](ch01.html#chapter_llm-introduction), the scaling laws dictate that performance of the language model increases as you increase the size of the model and its training data. However, increasing the model capacity implies more compute is needed for both training and inference. This is undesirable, especially at inference time, when latency requirements can be stringent. Can we increase the capacity of a model without increasing the required compute?

One way to achieve this is using conditional computation; each input (either a token or the entire sequence) sees a different subset of the model, interacting with only the parameters that are best suited to process it. This is achieved by composing the architecture to be made up of several components called experts, with only a subset of experts being activated for each input.

[Figure 4-6](#MoE-0) depicts a canonical MoE model.

![MoE](assets/dllm_0406.png)

###### Figure 4-6\. Mixture of Experts

A key component of the MoE architecture is the *gating function*. The gating function helps decide which expert is more suited to process a given input. The gating function is implemented as a weight applied to each expert.

The experts are typically added to the feedforward component of the Transformer. Therefore, if there are eight experts, then there will be eight feedforward networks instead of one. Based on the routing strategy used, only a small subset of these networks will be activated for a given input.

The routing strategy determines the number and type of experts activated. Two types of popular routing strategies exist:

*   Tokens choose

*   Experts choose

In the tokens choose strategy, each token chooses k experts. k is typically a small number (~2). The disadvantage of using this strategy is the need for load balancing. If in a given input batch, most of the tokens end up using the same experts, then additional time is needed to finish the computation as we cannot benefit from the parallelization afforded by multiple experts.

In the experts choose strategy, each expert picks the tokens that it is most equipped to handle. This solves the load balancing problem as we can specify that each expert choose the same number of tokens. However, this also leads to inefficient token-expert matching, as each expert is limited to picking only a finite number of tokens in a batch.

# Learning Objectives

Now that we have discussed the architecture of language models, let’s turn our focus to understanding the tasks they are trained on during the pre-training process.

As mentioned earlier in the chapter, language models are pre-trained in a self-supervised manner. The scale of data we need to train them makes it prohibitively expensive to perform supervised learning, where (input, output) examples need to come from humans. Instead, we use a form of training called self-supervision, where the data itself contains the target labels. The goal of self-supervised learning is to learn a task which acts as a proxy for learning the syntax and semantics of a language, as well as skills like reasoning, arithmetic and logical manipulation, and other cognitive tasks, and (hopefully) eventually leading up to general human intelligence. How does this work?

For example, let’s take the canonical language modeling task: predicting the next word that comes in a sequence. Consider the sequence:

```py
'Tammy jumped over the'
```

and the language model is asked to predict the next token. The total number of possible answers is the size of the vocabulary. There are many valid continuations to this sequence, like (hedge, fence, barbecue, sandcastle, etc.), but many continuations to this sequence would violate English grammar rules like (is, of, the). During the training process, after seeing billions of sequences, the model will know that it is highly improbable for the word “the” to be followed by the word “is” or “of,” regardless of the surrounding context. Thus, you can see how just predicting the next token is such a powerful tool: in order to correctly predict the next token you can eventually learn more and more complex functions that you can encode in your model connections. However, whether this paradigm is all we need to develop general intelligence is an open question.

Self-supervised learning objectives used for pre-training LLMs can be broadly classified (nonexhaustively) into three types:

*   Full language modeling (FLM)

*   Masked language modeling (MLM)

*   Prefix language modeling (PrefixLM)

Let’s explore these in detail.

## Full Language Modeling

[Figure 4-7](#full-language-modeling) shows the canonical FLM objective at work.

![Full Language Modeling](assets/dllm_0407.png)

###### Figure 4-7\. Full language modeling

This is the canonical language modeling objective of learning to predict the next token in a sequence and currently the simplest and most common training objective, used by GPT-4 and a vast number of open source models. The loss is computed for every token the model sees, i.e., every single token in the training set that is being asked to be predicted by the language model provides a learning signal for the model, making it very efficient.

Let’s explore an example, using the GPT Neo model.

Suppose we continue pre-training the GPT Neo model from its publicly available checkpoint, using the full language modeling objective. Let’s say the current training sequence is:

```py
'Language models are ubiquitous'
```

You can run this code:

```py
import torch
from transformers import AutoTokenizer, GPTNeoForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

input_ids = tokenizer("Language models are", return_tensors="pt")
gen_tokens = model.generate(**input_ids, max_new_tokens =1,

output_scores=True, return_dict_in_generate=True)
output_scores = gen_tokens["scores"]
scores_tensor = output_scores[0]
sorted_indices = torch.argsort(scores_tensor[0], descending=True)[:20]

for index in sorted_indices:
    token_id = index
    token_name = tokenizer.decode([token_id.item()])
    token_score = scores_tensor[0][index].item()
    print(f"Token: {token_name}, Score: {token_score}")
```

This code tokenizes the input text `Language models are` and feeds it to the model by invoking the `generate()` function. The function predicts the continuation, given the sequence “Language models are.” It outputs only one token and stops generating because `max_new_tokens` is set to 1\. The rest of the code enables it to output the top 20 list of tokens with the highest score, prior to applying the softmax at the last layer.

The top 20 tokens with the highest prediction score are:

```py
Output: Token:  a, Score: -1.102203369140625
Token:  used, Score: -1.4315788745880127
Token:  the, Score: -1.7675716876983643
Token:  often, Score: -1.8415470123291016
Token:  an, Score: -2.4652323722839355
Token:  widely, Score: -2.657834053039551
Token:  not, Score: -2.6726579666137695
Token:  increasingly, Score: -2.7568516731262207
Token:  ubiquitous, Score: -2.8688106536865234
Token:  important, Score: -2.902832508087158
Token:  one, Score: -2.9083480834960938
Token:  defined, Score: -3.0815649032592773
Token:  being, Score: -3.2117576599121094
Token:  commonly, Score: -3.3110013008117676
Token:  very, Score: -3.317342758178711
Token:  typically, Score: -3.4478530883789062
Token:  complex, Score: -3.521362781524658
Token:  powerful, Score: -3.5338563919067383
Token:  language, Score: -3.550961971282959
Token:  pervasive, Score: -3.563507080078125
```

Every word in the top 20 seems to be a valid continuation of the sequence. The ground truth is the token `ubiquitous`, which we can use to calculate the loss and initiate the backpropagation process for learning.

As another example, consider the text sequence:

```py
'I had 25 eggs. I gave away 12\. I now have 13'
```

Run the same code as previously, except for this change:

```py
input_ids = tokenizer("'I had 25 eggs. I gave away 12\. I now have",
  return_tensors="pt")
```

The top 20 output tokens are:

```py
Token:  12, Score: -2.3242850303649902
Token:  25, Score: -2.5023117065429688
Token:  only, Score: -2.5456185340881348
Token:  a, Score: -2.5726099014282227
Token:  2, Score: -2.6731367111206055
Token:  15, Score: -2.6967623233795166
Token:  4, Score: -2.8040688037872314
Token:  3, Score: -2.839219570159912
Token:  14, Score: -2.847306728363037
Token:  11, Score: -2.8585362434387207
Token:  1, Score: -2.877161979675293
Token:  10, Score: -2.9321107864379883
Token:  6, Score: -2.982785224914551
Token:  18, Score: -3.0570476055145264
Token:  20, Score: -3.079172134399414
Token:  5, Score: -3.111320972442627
Token:  13, Score: -3.117424726486206
Token:  9, Score: -3.125835657119751
Token:  16, Score: -3.1476120948791504
Token:  7, Score: -3.1622045040130615
```

The correct answer has the 17th highest score. A lot of numbers appear in the top 10, showing that the model is more or less randomly guessing the answer, which is not surprising for a smaller model like GPT Neo.

The OpenAI API provides the `logprobs` parameter that allows you to specify the number of tokens along with their log probabilities that need to be returned. As of the book’s writing, only the `logprobs` of the 20 most probable tokens are available. The tokens returned are in order of their log probabilities:

```py
import openai
openai.api_key = <Insert your OpenAI key>

openai.Completion.create(
  model="gpt-4o",
  prompt="I had 25 eggs. I gave away 12\. I now have ",
  max_tokens=1,
  temperature=0,
  logprobs = 10
)
```

This code calls the older gpt-4o model, asking it to generate a maximum of one token. The output is:

```py
"top_logprobs": [
          {
            "\n": -0.08367541,
            " 13": -2.8566456,
            "____": -4.579212,
            "_____": -4.978668,
            "________": -6.220278
            …
          }
```

gpt-4o is pretty confident that the answer is 13, and rightfully so. The rest of the top probability tokens are all related to output formatting.

###### Tip

During inference, we don’t necessarily need to generate the token with the highest score. Several *decoding strategies* allow you to generate more diverse text. We will discuss these strategies in [Chapter 5](ch05.html#chapter_utilizing_llms).

## Prefix Language Modeling

Prefix LM is similar to the FLM setting. The difference is that FLM is fully causal, i.e., in a left-to-right writing system like English, tokens do not attend to tokens to the right (future). In the prefix LM setting, a part of the text sequence, called the prefix, is allowed to attend to future tokens in the prefix. The prefix part is thus noncausal. For training prefix LMs, a random prefix length is sampled, and the loss is calculated over only the tokens in the suffix.

## Masked Language Modeling

[Figure 4-8](#masked-language-modeling-bert) shows the canonical MLM objective at work.

![Masked Language Modeling in BERT](assets/dllm_0408.png)

###### Figure 4-8\. Masked Language Modeling in BERT

In the MLM setting, rather than predict the next token in a sequence, we ask the model to predict masked tokens within the sequence. In the most basic form of MLM implemented in the BERT model, 15% of tokens are randomly chosen to be masked and are replaced with a special mask token, and the language model is asked to predict the original tokens.

The T5 model creators used a modification of the original MLM objective. In this variant, 15% of tokens are randomly chosen to be removed from a sequence. Consecutive dropped-out tokens are replaced by a single unique special token called the *sentinel token*. The model is then asked to predict and generate the dropped tokens, delineated by the sentinel tokens.

As an example, consider this sequence:

> Tempura has always been a source of conflict in the family due to unexplained reasons

Let’s say we drop the tokens “has,” “always,” “of,” and “conflict.” The sequence is now:

> Tempura <S1> been a source <S2> in the family due to unexplained reasons

with S1, S2 being the sentinel tokens. The model is expected to output:

> <S1> has always <S2> of conflict <E>

The output sequence is terminated by a special token indicating the end of the sequence.

Generating only the dropped tokens and not the entire sequence is computationally more efficient and saves training time. Note that unlike in Full Language Modeling, the loss is calculated over only a small proportion of tokens (the masked tokens) in the input sequence.

Let’s explore this on Hugging Face:

```py
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-3b")
model = T5ForConditionalGeneration.from_pretrained("t5-3b")

input_ids = tokenizer("Tempura <extra_id_0>  been a source <extra_id_1> in the
family due to unexplained reasons", return_tensors="pt").input_ids
targets = tokenizer("<extra_id_0> has always <extra_id_1> of conflict

<extra_id_2>", return_tensors="pt").input_ids
loss = model(input_ids=input_ids, labels=labels).loss
```

The targets can be prepared using a simple templating function.

More generally, MLM can be interpreted as a *denoising autoencoder*. You corrupt your input by adding noise (masking, dropping tokens), and then you train a model to regenerate the original input. BART takes this to the next level by using five different types of span corruptions:

Random token masking

[Figure 4-9](#bart-enoiser-objectives1) depicts the corruption and denoising steps.

![BART Denoiser Objectives1](assets/dllm_0409.png)

###### Figure 4-9\. Random token masking in BART

Random token deletion

The model needs to predict the positions in the text where tokens have been deleted. [Figure 4-10](#bart-enoiser-objectives2) depicts the corruption and denoising steps.

![BART Denoiser Objectives2](assets/dllm_0410.png)

###### Figure 4-10\. Random token deletion in BART

Span masking

Text spans are sampled from text, with span lengths coming from a Poisson distribution. This means zero-length spans are possible. The spans are deleted from the text and replaced with a single mask token. Therefore, the model now has to also predict the number of tokens deleted. [Figure 4-11](#bart-enoiser-objectives3) depicts the corruption and denoising steps.

![BART Denoiser Objectives3](assets/dllm_0411.png)

###### Figure 4-11\. Span masking in BART

Document shuffling

Sentences in the input document are shuffled. The model is taught to arrange them in the right order. [Figure 4-12](#bart-enoiser-objectives4) depicts the corruption and denoising steps.

![BART Denoiser Objectives4](assets/dllm_0412.png)

###### Figure 4-12\. Document shuffling objective in BART

Document rotation

The document is rotated so that it starts from an arbitrary token. The model is trained to detect the correct start of the document. [Figure 4-13](#bart-enoiser-objectives5) depicts the corruption and denoising steps.

![BART Denoiser Objectives5](assets/dllm_0413.png)

###### Figure 4-13\. Document rotation objective in BART

## Which Learning Objectives Are Better?

It has been shown that models trained with FLM are better at generation, and models trained with MLM are better at classification tasks. However, it is inefficient to use different language models for different use cases. The consolidation effect continues to take hold, with the introduction of [UL2](https://oreil.ly/xJc3U), a paradigm that combines the best of different learning objective types in a single model.

UL2 mimics the effect of PLMs, MLMs, and PrefixLMs in a single paradigm called *Mixture of Denoisers*.

The denoisers used are as follows:

R-Denoiser

This is similar to the T5 span corruption task. Spans between length 2–5 tokens are replaced by a single mask token. [Figure 4-14](#ul2-mixture-denoisers1) depicts the workings of the R-denoiser.

![UL2's Mixture of Denoisers1](assets/dllm_0414.png)

###### Figure 4-14\. UL2’s R-Denoiser

S-Denoiser

Similar to prefix LM, the text is divided into a prefix and a suffix. The suffix is masked, while the prefix has access to bidirectional context. [Figure 4-15](#ul2-mixture-denoisers2) depicts the workings of the S-Denoiser.

![UL2's Mixture of Denoisers2](assets/dllm_0415.png)

###### Figure 4-15\. UL2’s S-Denoiser

X-Denoiser

This stands for extreme denoising, where a large proportion of text is masked (often over 50%). [Figure 4-16](#ul2-mixture-denoisers3) depicts the workings of the X-Denoiser.

![UL2's Mixture of Denoisers3](assets/dllm_0416.png)

###### Figure 4-16\. UL2’s X-Denoiser

# Pre-Training Models

Now that we have learned about the ingredients that go into a language model in detail, let’s learn how to pre-train one from scratch.

The language models of today are learning to model two types of concepts with one model:

*   Language, the vehicle used to communicate facts, opinions, and feelings.

*   The underlying phenomena that led to the construction of text in the language.

For many application areas, we are far more interested in learning to model the latter than the former. While a language model that is fluent in the language is welcome, we would prefer to see it get better at domains like science or law and skills like reasoning and arithmetic.

These concepts and skills are expressed in languages like English, which primarily serve a social function. Human languages are inherently ambiguous, contain lots of redundancies, and in general are inefficient vehicles to transmit underlying concepts.

This brings us to the question: are human languages even the best vehicle for language models to learn underlying skills and concepts? Can we separate the process of modeling the language from modeling the underlying concepts expressed through language?

Let’s put this theory to the test using an example. Consider training an LLM from scratch to learn to play the game of chess.

Recall the ingredients of a language model from [Chapter 2](ch02.html#ch02). We need:

*   A pre-training dataset

*   A vocabulary and tokenization scheme

*   A model architecture

*   A learning objective

For training the chess language model, we can choose the Transformer architecture with the next-token prediction learning objective, which is the de facto paradigm used today.

For the pre-training dataset, we can use the chess games dataset from [Lichess](https://oreil.ly/XmWvv), containing billions of games. We select a subset of 20 million chess games for our training.

This dataset is in the Portable Game Notation (PGN) format, which is used to represent the sequence of chess moves in a concise notation.

Finally, we have to choose the vocabulary of the model. Since the only purpose of this model is to learn chess, we don’t need to support an extensive English vocabulary. In fact, we can take advantage of the PGN notation to assign tokens to specific chess concepts.

Here is an example of a chess game in PGN format, taken from [pgnmentor.com](https://oreil.ly/H3yOs):

```py
1\. e4 c5 2\. Nf3 a6 3\. d3 g6 4\. g3 Bg7 5\. Bg2 b5 6\. O-O Bb7 7\. c3 e5 8\. a3 Ne7
9\. b4 d6 10\. Nbd2 O-O 11\. Nb3 Nd7 12\. Be3 Rc8 13\. Rc1 h6 14\. Nfd2 f5 15\. f4
Kh7 16\. Qe2 cxb4 17\. axb4 exf4 18\. Bxf4 Rxc3 19\. Rxc3 Bxc3 20\. Bxd6 Qb6+ 21.
Bc5 Nxc5 22\. bxc5 Qe6 23\. d4 Rd8 24\. Qd3 Bxd2 25\. Nxd2 fxe4 26\. Nxe4 Nf5 27.
d5 Qe5 28\. g4 Ne7 29\. Rf7+ Kg8 30\. Qf1 Nxd5 31\. Rxb7 Qd4+ 32\. Kh1 Rf8 33\. Qg1
Ne3 34\. Re7 a5 35\. c6 a4 36\. Qxe3 Qxe3 37\. Nf6+ Rxf6 38\. Rxe3 Rd6 39\. h4 Rd1+
40\. Kh2 b4 41\. c7 1-0
```

The rows of the board are assigned letters a–h and the columns are assigned numbers 1–8\. Except for pawns, each piece type is assigned a capital letter, with N for knight, R for rook, B for bishop, Q for queen, and K for king. A + appended to a move indicates a check, a % appended to the move indicates a checkmate, and 0-0 is used to indicate castling. If you are unfamiliar with the rules of chess, refer to [this piece for a primer](https://oreil.ly/EbcfQ).

Based on this notation, the vocabulary can consist of:

*   A separate token for each square on the board, with 64 total (a1, a2, a3…​h6, h7, h8)

*   A separate token for each piece type (N, B, R, K, Q)

*   Tokens for move numbers (1., 2., 3., etc.)

*   Tokens for special moves (+ for check, x for capture, etc.)

Now, let’s train a language model from scratch on this chess dataset using our special domain-specific vocabulary. The model is directly learning from the PGN notation with no human language text present in the dataset. The book’s [GitHub repo](https://oreil.ly/llm-playbooks) contains the code and setup for training this model.

After training the model for three epochs, let’s test the model’s ability to play chess. We can see that the model seems to have learned the rules of the game without having to be provided the rules explicitly in natural language. In fact, the model can even beat human players some of the time and can execute moves like castling.

Note that this model was able to learn the concepts (chess) using a domain-specific language (PGN). How will we fare if the concepts were taught in natural language?

Let’s explore this in another experiment. Take the same dataset used to pre-train the chess language model and run it through an LLM to convert each move in PGN to a sentence in English. An example game would look like:

*White moves pawn to e4*

*Black moves bishop to g7*

and so on. Train a new language model on the same number of games as the previous one, but this time with the English-language dataset. Let the vocabulary of this model be the standard English vocabulary generated by training the tokenizer over the training set.

How does this compare to the chess LM trained on the PGN dataset? The model trained on English descriptions of chess moves performs worse and doesn’t seem to have understood the rules of the game yet, despite being trained on the same number of games as the other model.

This shows that natural language is not necessarily the most efficient vehicle for a model to learn skills and concepts, and domain-specific languages and notations perform better.

Thus, language design is an important skill to acquire, enabling you to create domain-specific languages for learning concepts and skills. For your application areas, you could use existing domain-specific languages or create a new one yourself.

# Summary

In this chapter, we discussed the various components of the Transformer architecture in detail, including self-attention, feedforward networks, position encodings, and layer normalization. We also discussed several variants and configurations such as encoder-only, encoder-decoder, decoder-only, and MoE models. Finally, we learned how to put our knowledge of language models together to train our own model from scratch and how to design domain-specific languages for more efficient learning.