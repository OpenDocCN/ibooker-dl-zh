# Chapter 15\. Transformers and transformers

With the paper [“Attention Is All You Need” by Ashish Vaswani et al.](https://oreil.ly/R7og7) in 2017, the field of AI was changed forever. While the abstract of the paper indicates something lightweight and simple—an evolution of the architecture of convolutions and recurrence (see Chapters [4](ch04.html#ch04_using_data_with_pytorch_1748548966496246) through [9](ch09.html#ch09_understanding_sequence_and_time_series_data_1748549698134578) of this book)—the impact of the work was, if you’ll forgive the pun, transformative. It utterly revolutionized AI, beginning with NLP. Despite the authors’ claim of the simplicity of the approach, implementing it in code is and was inherently complex. At its core was a new approach to ML architecture: *Transformers* (which we capitalize to indicate that we’re referring to them as a concept).

In this chapter we’ll explore the ideas behind Transformers at a high level, demonstrating the three main architectures: encoder, decoder and encoder-decoder. Please note that we will just be exploring at a very high level, giving an overview of how these architectures work. To go deep into these would require several books, not just a single chapter!

We’ll then explore *transformers*, which we lowercase to indicate that they are the APIs and libraries from Hugging Face that are designed to make using Transformer-based models easy to use. Before transformers, you had to read the papers and figure out how to implement the details for yourself for the most part. So, the Hugging Face transformers library has widened access to models created using the Transformer architecture and has become the de facto standard for using the many models that have been created using the transformer-based architecture.

###### Note

Just to clarify, for the rest of this chapter, I’ll refer to the architecture, models, and concepts as Transformers (with a capital *T*) and the Hugging Face libraries as transformers (with a lowercase *t*) to prevent confusion.

# Understanding Transformers

Since the publication of the original paper mentioned in the introduction to this chapter, the field of Transformers has evolved and grown, but its underlying basis has remained pretty much the same. In this section, we’ll explore this.

When working with LLMs anywhere (not just with Hugging Face), you’ll hear of the terms *encoder*, *decoder*, and *encoder-decoder*. Therefore, I think it’s a good idea for you to get a high-level understanding of them. Each of these architectures represents a different approach to text management—be it processing, classification, or generation. They have specific strengths for particular scenarios, and to optimize for your scenario, it’s good to understand them so you can choose the appropriate ones.

## Encoder Architectures

Encoder-only architectures (e.g., BERT, RoBERTa) generally excel at *understanding* text because of how rigorous they are in processing it. They’re bidirectional in nature, being able to “see” the entire input sequence at once. With that nature of understanding, they’re particularly effective for tasks that require a deep understanding and comprehension of the text and its semantics. So, they’re particularly suited for tasks such as classification, named-entity recognition, and extraction of meaning for things like question answering. Their strength is transforming text into rich, contextual representations, but they’re not designed to *generate* new text.

You can see the encoder-based architecture in [Figure 15-1](#ch15_figure_1_1748549808951859).

![](assets/aiml_1501.png)

###### Figure 15-1\. Encoder-based architecture

Let’s explore this architecture in a little more detail. It begins with the tokenized inputs, which are then passed to the self-attention layer.

### The self-attention layer

*Self-attention* is the core mechanism that allows tokens to “pay attention” to other tokens in the input sequence. So, for example, consider the sentence “I went to high school in Ireland, so I had to study how to speak Gaelic.” The last word in this sentence is *Gaelic*, and it’s effectively triggered by the word *Ireland* earlier in the sentence. If a model pays attention to the entire sentence, it can predict the word *Gaelic* to be the next word. On the other hand, if the model didn’t pay attention to the entire sentence, then it might interpret from the sentence something more appropriate to “how to speak,” such as *politely* or another adjective.

However, the self-attention mechanism—by considering the entire sentence—can understand context like that more granularly. It works by having each token in the sentence get three vectors associated with it. These are the query (Q) vector (aka “What am I looking for that’s relevant to this token?”), the key (K) vector (aka “What tokens might reference me?”), and the value (V) vector (aka “What type of information do I carry?”). The representations in these vectors are learned over time, in much the same process as we saw in earlier chapters of this book. An attention score is then calculated as a function of these, and using Softmax, the embeddings for the words will be updated with the attention details, bending word embeddings closer to one another when there are similarities learned between them.

Do note that self-attention is generally bidirectional, so the order of the words doesn’t matter.

The self-attention mechanism usually also has the context of *heads*, which are effectively multiple, parallel instances of the three vectors (Q, K, and V) that we saw previously, which can learn different representations and effectively specialize in different aspects of the input. A high-level representation of these heads is shown in [Figure 15-2](#ch15_figure_2_1748549808951907).

Thus, each head has its own set of learned weights for Q, K, and V vectors. The processing and learning for these vectors is done in parallel, with their results concatenated, and their final output projection is then a combination of information from each of the heads. As models have grown larger over time, one of the factors for this growth is the number of heads. For example, BERT-base has 12 heads and BERT-large has 16 heads. Architectures that also use a decoder, such as GPT, have grown similarly. GPT-2 had 12 attention heads, whereas GPT-3 grew to 96!

Returning to [Figure 15-1](#ch15_figure_1_1748549808951859), the self-attention instance (in which [Figure 15-2](#ch15_figure_2_1748549808951907) can be encapsulated into the self-attention box from [Figure 15-1](#ch15_figure_1_1748549808951859)) then outputs to a *feedforward network* (FFN), which is often more accurately referred to as a *position-wise feedforward network*.

![](assets/aiml_1502.png)

###### Figure 15-2\. Multihead self-attention

### The feedforward network layer

The FFN layer is vital in supporting the model’s capacity to learn complex patterns in the text. It does this by introducing nonlinearity into the model.

Why is that important? First of all, let’s understand the difference between linearity and nonlinearity. A *linear equation* is one for which the value is relatively easy to predict. For example, consider an equation that determines a house price. A linear version of this might be the cost of the land plus a particular dollar amount per square foot, and every house would follow the same formula. But as we know, house prices are far more complex than this—they don’t (unfortunately) follow a simple linear equation.

Understanding sentiment can be the same. So, for example, if you were to assign coordinates on a graph (like we did when explaining embeddings in [Chapter 7](ch07.html#ch07_recurrent_neural_networks_for_natural_language_pro_1748549654891648)) to the words *good* and *not*, where *good* might be +1 and *not* might be –1, then a linear relationship between these for *not good* would give us 0, which is neutral, whereas *not good* is clearly negative. So, we need equations that are more nuanced (i.e., nonlinear) when capturing sentiment and effectively understanding our text.

That’s the job of the FFN. It achieves this nonlinearity by expanding the dimensions of its input vector, applying a transformation to that, and using ReLU to “remove” the negative values (and thus remove the linearity) before restoring the vector back to its original dimensions. You can see this in [Figure 15-3](#ch15_figure_3_1748549808951933).

![](assets/aiml_1503.png)

###### Figure 15-3\. A feedforward network

The underlying math and logic behind how it works are a little beyond the scope of this book, but let’s explore them with a simple example. Consider this code, which simulates what’s happening in [Figure 15-3](#ch15_figure_3_1748549808951933):

```py
import torch
import torch.nn as nn

# Simplified example with small dimensions
d_model = 2  # Input/output dimension
d_ff = 4     # Hidden dimension

# Create some sample input
x = torch.tensor([[–1.0, 2.0]])

# First linear layer (2 → 4)
W1 = torch.tensor([
    [1.0, –1.0],
    [–1.0, 1.0],
    [0.5, 0.5],
    [–0.5, –0.5]
])
b1 = torch.tensor([0.0, 0.0, 0.0, 0.0])
layer1_out = torch.matmul(x, W1.t()) + b1
print("After first linear layer:", layer1_out)

# Apply ReLU
relu_out = torch.relu(layer1_out)
print("After ReLU:", relu_out)
# Notice how negative values became 0; this is the nonlinear operation!

# Second linear layer (4 → 2)
W2 = torch.tensor([
    [1.0, –1.0, 0.5, –0.5],
    [–1.0, 1.0, 0.5, –0.5]
])
b2 = torch.tensor([0.0, 0.0])
final_out = torch.matmul(relu_out, W2.t()) + b2
print("Final output:", final_out)

```

We start with a simple 2D tensor: `[–1.0, 2.0].`

The first linear layer has the following weights and biases:

```py
W1 = torch.tensor([
    [1.0, –1.0],
    [–1.0, 1.0],
    [0.5, 0.5],
    [–0.5, –0.5]
])
b1 = torch.tensor([0.0, 0.0, 0.0, 0.0])

```

When we pass our 2D tensor through this layer to get `layer1_out`, the matrix multiplication gives us a 4D output that looks like this:

```py
After first layer: [–3.0, 3.0, 0.5, –0.5]
```

There are two negative values in this layer (–3 and –0.5), so when we pass it through the ReLU, they are set to zero and our matrix becomes this:

```py
Output: [0.0, 3.0, 0.5, 0.0]
```

This process is called *bending*. By taking these values out, we’re now introducing nonlinearity into the equation. The relationships between the values have become much more complex, so a process that attempts to learn the parameters of those relationships will have to deal with that complexity, and if it succeeds in doing so, it will avoid the linearity trap.

Next, we’ll convert back to a 2D tensor by going through another layer, with weights and biases like this:

```py
W2 = torch.tensor([
    [1.0, –1.0, 0.5, –0.5],
    [–1.0, 1.0, 0.5, –0.5]
])
b2 = torch.tensor([0.0, 0.0])
```

The output of the layer will then look like this:

```py
Final output: [–2.75, 3.25]
```

So, the effect of the FFN is to take in a vector and output a vector of the same dimension, with linearity removed.

We can explore this with our simple code. Consider what happens when we take our input and apply simple, linear changes to it. So, if we take [–1.0, 2.0] and double it to [–2.0, 4.0], the nonlinearity introduced by the FFN will mean that the output won’t be a simple doubling. And similarly, if we negate it, the output again won’t be a simple negation:

```py
Input: [–1.0, 2.0] → Output: [–2.75, 3.25]
Input: [–2.0, 4.0] → Output: [–5.5, 6.5]    # Not a simple doubling!
Input: [1.0, –2.0] → Output: [2.75, –3.25]  # Not a simple negation!

```

Over time, the parameters that are learned for the weights and biases should maintain the relationships between the tokens and allow the network to learn more nuanced, nonlinear equations that define the overall relationships between them.

### Layer normalization

Referring back to [Figure 15-1](#ch15_figure_1_1748549808951859), the next step in the process is called *layer normalization*. At this point, the goal is to stabilize the data flowing through the neural network by removing outliers and high variance. Layer normalization does this by calculating the mean and variance of the input features, which it then normalizes and scales/shifts before outputting (see [Figure 15-4](#ch15_figure_4_1748549808951956)).

![](assets/aiml_1504.png)

###### Figure 15-4\. Layer normalization

From a statistical perspective, the idea of removing the outliers from using mean and variance and then normalizing them is quite straightforward. I won’t go into details on the statistics here, but that’s generally the goal of doing these types of calculations.

The *Scale and Shift* box then becomes a mystery. Why would you want to do this? Well, if you dig a little bit into the logic, the idea here is that the process of *normalization* will drive the mean of a set of values to 0 and the standard deviation to 1\. The process itself can destroy distinctiveness in our input features by making them too alike. So, if there’s a process that we can use to return some level of variance to them with parameters that are learned, we can clean up the data without destroying it—meaning we won’t throw the baby out with the bathwater!

Therefore, multiplying our outputs by values with offsets can change this. These values are typically called *gamma* and *beta* values, and they act a little like weights and biases. It’s probably easiest to show this in code.

So, consider this example, in which we’ll take an input feature containing some values and then normalize them. We’ll see that the normalized values will have a mean of 0 and a standard deviation of 1:

```py
import torch

# Create sample feature values
features = torch.tensor([5.0, 1.0, 0.1])
print("\nOriginal features:", features)
print("Original mean:", features.mean().item())
print("Original std:", features.std().item())

# Standard normalization
mean = features.mean()
std = features.std()
normalized = (features - mean) / std
print("\nJust normalized:", normalized)
print("Normalized mean:", normalized.mean().item())
print("Normalized std:", normalized.std().item())

# With learnable scale and shift
gamma = torch.tensor([2.0, 0.5, 1.0])  # Learned parameters
beta = torch.tensor([1.0, 0.0, —1.0])  # Learned parameters
scaled_shifted = gamma * normalized + beta
print("\nAfter scale and shift:", scaled_shifted)
print("Final mean:", scaled_shifted.mean().item())
print("Final std:", scaled_shifted.std().item())
```

But when we move the values through the scale and shift by using gamma and beta values, we get a new set of parameters that maintain a closer relationship to the originals but with massive variance (aka noise) removed. The output of this code should look like this:

```py
Original features: tensor([5.0000, 1.0000, 0.1000])
Original mean: 2.0333333015441895
Original std: 2.6083199977874756

Just normalized: tensor([ 1.1374, —0.3962, —0.7412])
Normalized mean: 0.0
Normalized std: 1.0

After scale and shift: tensor([ 3.2748, —0.1981, —1.7412])
Final mean: 0.44515666365623474
Final std: 2.5691161155700684
```

Like the FFN, this is effectively destroying and then reconstructing the features in a clever way. In this case, it’s designed to do it to remove variance, which has the effect of amplifying or dampening features as needed by shifting the overall distribution to better ranges for activation functions.

I like to think of this as what you do with your TV to get a better image—sometimes, you adjust the contrast or the brightness. By finding the optimal values of these settings, you can see the important details of a particular image better. Think of the contrast as the scale and the brightness as the shift. If a network can learn these for your input features, it will improve its ability to understand them!

### Repeated encoder layers

Referring back to [Figure 15-1](#ch15_figure_1_1748549808951859), you’ll see that the self-attention, feedforward, and layer normalization layers can be repeated N times. Typically, smaller models will have 12 instances and larger ones will have 24\. The deeper the model, the more computational resources are required. More layers provide more capacity to learn complex patterns, but of course, this means longer training time, more GPU memory overhead, and potentially greater risks of overfitting. Additionally, larger models can impact the corresponding data requirements, leading to a possibly negative knock-on effect for complexity.

For the most part, there’s a trade-off between the depth of the model (the number of layers) and the width (the size of each layer, including the number of heads). In some cases, models can reuse the same layer multiple times to reduce the overall parameter count—and ALBERT is an example of this.

## The Decoder Architecture

While the encoder architecture specializes in understanding text by having attention across the entire context of the input sequence, the decoder architecture serves as the generative powerhouse. It is designed to produce sequential outputs one element at a time. While the encoder processes all inputs simultaneously (or as many of them as it can, based on the parallelization of the system), the decoder operates autoregressively. It generates each output token while considering both the encoded input representations *and* the previously generated outputs. The goal is to maintain coherence and contextual relevance through the process.

You can see a diagram of the encoder architecture in [Figure 15-5](#ch15_figure_5_1748549808951977).

![](assets/aiml_1505.png)

###### Figure 15-5\. The decoder architecture

Let’s explain this from the top down. The first box is the previously generated tokens. In a pure decoder architecture, this is the set of tokens that have already been generated or provided. When they’re provided, they’re typically called the prompt. So, say you provide the following tokens:

```py
[“If”, “you”, “are”, “happy”, “and”, “you”, “know”, “it”]
```

after one is run through the decoder, the token for “clap” will be generated. You will now have these:

```py
[“If”, “you”, “are”, “happy”, “and”, “you”, “know”, “it”, “clap”]
```

These tokens will flow into the box for token embedding + positional encoding.

### Understanding token and positional encoding

This transforms each token into a vector representation called an *embedding* (as explained in [Chapter 5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759)), which clusters words of similar semantic meaning in a similar vector space. Remember that these embeddings are learned over time.

So, for example, if we think about the words *it* and *cla*p at the end of the aforementioned token list, they may have token embeddings that look like this:

```py
"it" -> [0.2, –0.5, 0.7] (position 7)
"clap" -> [–0.3, 0.4, 0.1] (position 8)

```

I have simplified the embedding to just three dimensions for readability.

The next step is to perform the *positional* encoding, which is a huge innovation in Transformers. In addition to an encoding in a vector space for the semantics and meaning of the word, an innovative method using sine and cosine waves is performed to encode the word’s position and the impact of this position on neighbors.

So, for example, given that we have 3D vectors for our encodings, we’ll create a 3D vector using sine waves for the odd-numbered indices and cosine waves for the even-numbered ones, like this:

```py
Position 7 -> [sin(7), cos(7), sin(7)] = [.122, .992, .122]
Position 8 -> [sin(8), cos(8), sin(8)] = [.139, .990, .139]
```

We then add these together to get this:

```py
“It” -> [0.2+.122, —0.5+.992, 0.7+.122] = [0.322, 0.492, 0.822]
“Clap” -> [—0.3+.139, 0.4+.990, 0.1+.139] = [—0.161, 1.390, 0.239]
```

While this may seem arbitrary, the positional encodings actually come from a specific formula. These are shown here:

```py
# For Even-numbered dimensions
PE(pos, d) = sin(pos / 10000^(d/d_model))

# For Odd-numbered dimensions
PE(pos, d) = cos(pos / 10000^(d-1/d_model))
```

If you plot these values as a table, they’ll look like [Table 15-1](#ch15_table_1_1748549808960051).

Table 15-1\. Positional encodings

| Position | Dimension 0 | Dimension 1 | Dimension 2 | Dimension 3 | Dimension 4 | Dimension 5 | Dimension 6 | Dimension 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| **1** | 0.841 | 0.540 | 0.100 | 0.995 | 0.010 | 1.000 | 0.001 | 1.000 |
| **2** | 0.909 | –0.416 | 0.199 | 0.980 | 0.020 | 1.000 | 0.002 | 1.000 |
| **3** | 0.141 | –0.990 | 0.296 | 0.955 | 0.030 | 1.000 | 0.003 | 1.000 |

The ultimate goal here is to have a relationship between each position on the list and every other position that, in some dimensions, has tokens that are typically far apart being a little closer and those that are typically closer being a little further apart!

So, imagine you have an input sequence of four tokens in positions 0 through 3, as charted in the table. The token in position 3 is as far away as possible from the token in position 0, so they are at extreme ends of the sequence.

With a positional encoding like this, we are given the possibility that in some dimensions, they are closer together. You can see in the first column that the values for dimension 0 places position 3 closer to position 0 than either of the others, whereas in the column for dimension 2, they are further apart. By using these positional encodings, we’re opening up the possibility that words can be clustered, even if they’re far apart in a sentence.

Think back to the sentence “I went to high school in Ireland, so I had to study how to speak Gaelic.” In that case, the final token “Gaelic” was most accurate because it described a language in “Ireland,” which was earlier in the sentence. Without positional encoding, this would have been missed!

Also, the positional encodings are *added* to the token embeddings, so they provide a sort of pressure to keep together the words that might be semantically related in different parts of the sentence, but they don’t completely override the embeddings.

This is then fed into the multihead masked attention. We’ll look at that next.

### Understanding multihead masked attention

Earlier in this chapter, in the section on attention, we saw how the Q, K, and V vectors for each token are learned and used to update the embeddings of the word with attention to the other words. The idea behind *masked attention* updates this to ignore words that we shouldn’t be paying attention to. In other words (sic), the goal is that we should only pay attention to *previous* positions in the sequence.

So, imagine you have a sequence of eight words and you want to predict the ninth. When you’re processing the third word in the sentence, it should only pay attention to the second and first word but nothing after them. You can achieve this with a triangular matrix like this:

```py
1 0 0
1 1 0
1 1 1
```

So, imagine this for a set of words like *the*, *cat*, and *sat* (see [Table 15-2](#ch15_table_2_1748549808960104)).

Table 15-2\. Simple masked attention

|   | the | cat | sat |
| --- | --- | --- | --- |
| the | 1 | 0 | 0 |
| cat | 1 | 1 | 0 |
| sat | 1 | 1 | 1 |

When processing *the*, using this method means we can only pay attention to *The* itself. When processing *cat*, we can pay attention to both *the* and *cat*. When processing sat, we can pay attention to *the*, *cat*, and *sat*.

So, recalling that the K, Q, and V vectors will amend the embeddings for the word in a way that bends them closer together for instances where the words may not have close syntactical meaning but are impacting one another through attention (like *Ireland* and *Gaelic* in the earlier example), the goal of the masked attention layer will only do this bending for words that we’re allowed to pay attention to.

When this is performed multiple times, in parallel, across multiple heads, and aggregated together, we get an attention adjustment of the embeddings that’s very similar to the one we did with the encoder—except that the masking prevents any amendment of the token from words *after* it in the sequence, particularly those that are generated.

### Adding and normalizing

Next, we take the attention output from the masked attention layer and add it to the original input. This is called making a *residual connection*.

So, for example, the process might look like this:

```py
Original input (word "cat" embedding):
[0.5, —0.3, 0.7, 0.1]  # Contains basic info about "cat"

Attention output (learned changes):
[0.2, 0.1, —0.1, 0.3]  # Contains contextual updates based on other words

After adding (final result):
[0.7, —0.2, 0.6, 0.4]  # Original meaning of "cat" PLUS contextual information
```

As we think about this, we’ll see that we don’t *replace* the original information with the attention mechanism but instead we *add* the new learned embeddings from the attention mechanism, thus preserving the original information. Over time, this will have the effect of helping the network learn. If some attention updates aren’t useful, then the network will just make them close to zero through the backpropagation of gradients.

This is beyond the scope of this book, and it’s generally found in the papers behind the creation of these models. But ultimately, the goal here is to get rid of a problem called the *vanishing gradient problem*—in which, if the original input was *not* maintained, then the gradients of the attention layer can get smaller and smaller with successive layers, thus limiting the number of layers you can use. But if you always add the gradients to the original input, then there will be a floor—such as the [0.5, –0.3, 0.7, 0.1] for the cat gradient previously mentioned—so the small changes from the attention gradients won’t push these values close to zero and cause the overall gradients to vanish.

This is then pushed through a layer normalization, as described in the encoder chapter, to remove outliers while keeping the knowledge of the sequences intact.

### The feedforward layer

The feedforward layer operates in exactly the same way as those layers used in encoders (see earlier in this chapter), with the goal of reducing any linear dependencies in the token sequence. The output from this is again added to the original data and then normalized, the logic being that the process of removing the outliers with the FFN should also prevent the gradients from vanishing and thus preserve important information. The normalization also keeps the values in a stable range, as repeatedly adding as we’re doing here might push some values far above 1.0 or below –1.0, and normalized values in these ranges tend to be better for matrix calculation.

We can repeat this process of masked attention -> add and normalize -> feedforward -> add and normalize multiple times before we get to the next layer, where we’ll use the learned values to predict the next token.

### The linear and Softmax layers

The linear and Softmax layers are responsible for turning the decoder’s representations into probabilities for the next token.

The linear layer will learn representations for each of the words in the dictionary with the transposed size of the decoder’s representations. This is a bit of a mouthful, so let’s explore it with an example.

Say our decoder output, having flowed through all the layers, is a 4D representation, like in [Table 15-3](#ch15_table_3_1748549808960134).

Table 15-3\. Simulated decoder output

|   |   |   |   |
| --- | --- | --- | --- |
| 0.2 | –0.5 | 0.8 | –0.3 |

We now have a weights matrix for each word in our vocabulary that is learned during training, and that matrix might look like the one in [Table 15-4](#ch15_table_4_1748549808960160).

Table 15-4\. Weights matrix for words in our vocab

| cat | dog | sat | mat | the |
| --- | --- | --- | --- | --- |
| 1.0 | 0.5 | 2.0 | 0.3 | 0.7 |
| -0.3 | 0.8 | 1.5 | 0.4 | 0.2 |
| 2.0 | 0.3 | 2.1 | 0.5 | 0.8 |
| 0.4 | 0.6 | 0.9 | 0.2 | 0.5 |

Note that the decoder representation is 1 × 4 and that each matrix for each word is 4 × 1. That’s the transposition, and multiplying them out is now easy.

So, for *cat*, our final score will be this:

(0.2 × 1.0) + (–.5 × –.3) + (0.8 × 2.0) + (–0.3 × 0.4) = 1.8

We can then get final scores for each word, as in [Table 15-5](#ch15_table_5_1748549808960183):

Table 15-5\. Final scores for each word

| cat | dog | sat | mat | the |
| --- | --- | --- | --- | --- |
| 1.8 | -0.2 | 1.1 | 0.2 | 0.5 |

Using the Softmax function, these are then turned into probabilities, as in [Table 15-6](#ch15_table_6_1748549808960204).

Table 15-6\. Probabilities from Softmax function

| cat | dog | sat | mat | the |
| --- | --- | --- | --- | --- |
| 47.5% | 6.4% | 23.6% | 9.6% | 12.9% |

And then we can take the highest-probability word as the next token, in a process called *greedy decoding*. Alternatively, we can take 1 from *k* possible top values, in a process called *top-k decoding*, in which we pick, for example, the top three probabilities and choose one value from there.

This is then fed back into the top box as the new token list so that the process can continue to predict the next token.

And that’s pretty much how decoders work, at least from a high level. In the next section, we’ll look at how they can be combined in the encoder-decoder architecture.

# The Encoder-Decoder Architecture

The *encoder-decoder architecture*, also known as *sequence-to-sequence*, combines the two aforementioned architecture types. It does this to tackle tasks that require transformation between input and output sequences of varying lengths. It’s proven to be very effective for machine translation in particular, but it also can be used in models for text summarization and question answering.

As you can see in [Figure 15-6](#ch15_figure_6_1748549808951997), it’s very similar to the decoder architecture, for the most part. The difference is the addition of a cross-attention layer that takes in the output from the encoder and injects it into the middle of the workflow.

The encoder will process the entire input sequence, creating a rich contextual representation that captures the full meaning of the input. The decoder layer can then query this, combining it with its representations to allow the decoder to focus additionally on relevant parts of the input when generating each new token.

![](assets/aiml_1506.png)

###### Figure 15-6\. The encoder-decoder architecture

Now, you might wonder at this point why the encoder-decoder architecture needs the encoder’s output, which it merges with cross-attention. Why can’t it just unmask in its own self-attention block? The fundamental reason boils down to the fact that this is more powerful for the following reasons:

Separation of concerns and parameter focus

If the decoder self-attention block were to be unmasked, it would have to handle the tasks of both understanding the input *and* generating the output simultaneously. That could lead to issues with learning because there’s a poor target. But if we separate them, each can focus on its own specialized role.

Quality

If we separate concerns, each role can build up a rich representation that’s suitable for its task. In particular with the encoder, we have a well-known, battle-tested architecture for artificial understanding that we know works for that task.

The major innovation here is the *cross-attention block*. We can demonstrate the intuition behind this with the analogy of a human language translator. When a person translates a sentence from French to English, they don’t just memorize the entire French corpus and then write English. Instead, while writing the English words, they actively look at different parts of the French sentence, focusing on the most relevant parts of the sentence for the word they are currently writing.

In French, the sentence “Le chat noir” translates to “The black cat,” but the noun and adjective are reversed. A straight translation would be “The cat black.” The human translator, when paying attention, would know this and would focus on other words in the French sentence. Cross-attention does the same thing. As the decoder generates each new word, it needs to refer to the source material to figure out what word to generate next. The words that have gone before in its output may not be enough.

You can see this mechanism in action in [Figure 15-7](#ch15_figure_7_1748549808952017).

Ultimately, an encoder creates a rich, contextual representation of the input because it artificially understands the source sentence. Cross-attention is a selective spotlight that highlights what the model believes to be the most relevant parts of this understanding for each word it generates, and this highlighting makes the model more effective at generating the correct words. You can see how this is very effective for machine translation, but it’s not much of a stretch to understand how it might be useful for other tasks!

The mechanism for cross-attention works with the K, Q, and V vectors as before, but the innovation here is that the Q vector will code from the decoder, while the K and V vectors will come from the encoder.

This concludes a very high-level look at Transformers and how they work. There’s lots more detail—in particular, about *how* they learn things like the Q, K, and V vectors—that’s beyond the scope of this chapter, and I’d recommend reading the original paper to learn more.

Now, let’s switch gears to the transformers (with a lowercase *t*) API from Hugging Face, which makes it really easy for you to use Transformer-based models in code.

![](assets/aiml_1507.png)

###### Figure 15-7\. Cross-attention

# The transformers API

At their core, transformers provide an API for working with pretrained models that use Python and PyTorch. The library’s success stems from three key innovations: a simple interface for using pretrained models, an extensive collection of pretrained models (as we explored in [Chapter 14](ch14.html#ch14_using_third_party_models_and_hubs_1748549787242797)), and a vibrant community that frequently contributes improvements and new models.

The library has evolved beyond its NLP roots to support multiple types of models, including those that support computer vision, audio processing, and multimodal tasks. Originally, the goal of the transformer-based architecture was to be effective in learning how a sequence of tokens is followed by another sequence of tokens. Then, innovative models built on this idea to allow concepts such as sound to be expressed as a sequence of tokens, and as a result, transformers could then learn sound patterns.

For developers, transformers offer multiple abstraction levels. The high-level pipeline API, which we explored in [Chapter 14](ch14.html#ch14_using_third_party_models_and_hubs_1748549787242797), enables immediate use of models for common tasks, while lower-level interfaces provide fine-grained control for custom implementations. Also, the library’s modular design allows developers to mix and match components like tokenizers, models, and optimizers.

Perhaps most powerfully, transformers emphasize transparency and reproducibility. All model implementations are open source, well documented, and accompanied by model cards describing their capabilities, limitations, and ethical considerations. It’s a wonderful learning process to crack open the transformers library on GitHub and explore the source code for common models like GPT and Gemma.

This commitment to openness has made the transformers API an invaluable tool for you, and it’s something that’s well worth investing your time to learn!

# Getting Started with transformers

In [Chapter 14](ch14.html#ch14_using_third_party_models_and_hubs_1748549787242797), we explored how to access a model from the Hugging Face Hub, and we saw how easy it was to instantiate one and then use it with the various pipelines. We’ll go a little deeper into that in this chapter, but let’s begin by installing the requisite libraries:

```py
pip install transformers
pip install datasets  # for working with HuggingFace datasets
pip install tokenizers  # for fast tokenization
```

Many of the models in the Hugging Face Hub, which are accessible via transformers, need you to have an authentication token on Hugging Face, along with permission to use that model. In [Chapter 14](ch14.html#ch14_using_third_party_models_and_hubs_1748549787242797), we showed you how to get an access token. After that, depending on the model you choose to use, you may need to ask permission on its landing page if you want access to it. You also learned how to use the transformers library in Google Colab, but there are, of course, many other ways to use transformers other than in Colab!

Once you have a token, you can use it in your Python code like this:

```py
from huggingface_hub import login
login(token="your_token_here")
```

Or, if you prefer, you can set an environment variable:

```py
export HUGGINGFACE_TOKEN="your_token_here"
```

With that, your development environment can now support development with transformers. Next, we’ll look at some core concepts within the library.

# Core Concepts

There are a number of important core concepts of using transformers that you can take advantage of. The simplicity of the transformers design hides the core concepts from you until you need them, but let’s take a look at some of them here in a little more detail.

We’ll start with the pipelines that you learned about in [Chapter 14](ch14.html#ch14_using_third_party_models_and_hubs_1748549787242797).

## Pipelines

The `pipeline` class implements transformers’ most useful abstraction, with a goal of making complex transformer operations accessible with minimal code. You can get the core default functionality of the model by using the appropriate pipeline with its defaults, and you can also override the defaults to create custom functionality.

Pipelines encapsulate all of ML processing—from data preprocessing to model inference to result formatting—in a single method.

To see the power of pipelines in action, let’s look at an example:

```py
from transformers import pipeline

# Basic sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love working with transformers!")
```

In this case, we didn’t specify a model but rather a scenario (`sentiment-analysis`), and the pipeline configuration chose the default model for that scenario and did all of the initialization for us. As you saw in Chapters [5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759) and [6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888), when you’re dealing with text, you need to tokenize and sequence it. You also need to know which tokenizer was used by a particular model so you can ensure that your text is encoded correctly and then converted to tensors in the correct, normalized format. You also need to parse and potentially detokenize the output, yet none of that code is present here. Instead, you simply say that you want sentiment analysis and then point the pipeline toward the text you want to classify.

We’re not just limited to text sentiment analysis, of course. We can also do text classification, generation, summarization, and translation. Other scenarios include entity recognition and question answering. As more multimodal transformer-based models come online, there’s also image classification and segmentation, object detection, image generation, and audio scenarios including speech recognition and generation.

While there are default models for a scenario, like we saw just now, there’s also the ability to override the defaults and send custom parameters to a model.

So, for example, with text generation, we can use the default experience like this:

```py
generator = pipeline("text-generation")
text = generator("The future of AI is")
print(text)
```

Or we can customize it further by passing parameters for things like the number of tokens to generate (`max_length`), the temperature (how creative it will be), and the number of tokens to evaluate when outputting a new one (`top_k`):

```py
# Text generation with specific parameters
generator = pipeline(
    "text-generation",
    model="gpt2",
    max_length=50,
    temperature=0.7,
    top_k=50
)
text = generator("The future of AI is")
```

This gives you the flexibility you need to have fine control of any particular model, meaning it lets you set your desired parameters to override the default behavior. Importantly, the pipeline’s abstraction handles the crucial steps of using the model.

To reiterate: when you’re using a pipeline, you’re getting more than just the model download. Depending on your scenario, you’ll typically get the following steps:

Model loading

When you specify a model name, the pipeline API automatically downloads and caches the model and any associated tools, such as the tokenizer.

Preprocessing

This takes your raw inputs in their typical data format (strings, bitmaps, wave files, etc.) and turns them into model-compatible formats, even when multiple steps are needed. For example, it can tokenize a string and then turn the resulting tokens into embeddings or tensors.

Tokenization

As mentioned previously, this helps you with the specific tokenization strategy that a particular model needs. One tokenization scheme does not fit all!

Batching

This abstracts away the need for you to calculate optimal batch sizes. It will efficiently handle batching for you while respecting memory constraints.

Post-processing

Models output tensors of probabilities, and the pipeline will turn them into a human-readable format that you or your code can work with.

Now that we’ve had a quick look at pipelines and what they are (and you’ll be using them a lot in this book), let’s continue our tour of the core concepts and switch to tokenizers.

## Tokenizers

We’ve spoken about tokenizers a lot, and we’ve even built our own in [Chapter 5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759) and [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888). But prebuilt tokenizers can be very powerful and useful tools as you create your own apps. Hugging Face transformers give you the `AutoTokenizer` class, and they make your life a lot easier when you’re dealing with tokenizers by handling many of the complex scenarios for you. Even if you are creating your own models and training them from scratch, using an AutoTokenizer is probably a much smarter way of handling this task, rather than rolling your own.

To go a little deeper, the tokenizer is fundamental to how transformer models will process text. It’s the first step in converting your raw text into a format that the model can work with. It’s also often overlooked in discussions of building models, and that’s a big mistake. The tokenization strategy is vital in the design of any system, and a badly designed one can negatively impact your overall model performance. Therefore, it’s important to have a well-designed tokenizer for the task at hand.

At its core, the tokenizer’s job is to break down text into smaller, numeric units called tokens. They can be words, parts of words (aka *subwords*), or even individual characters. To choose the right tokenization strategy, you’ll need to make trade-offs among vocabulary size, the length of the sequences of tokens you will use in the model architecture, and your desire and requirements to handle words that are uncommon.

The transformers library supports multiple approaches, with subword tokenization being the most common one. It’s a nice balance between character- and word-level tokenization that allows for less frequent words to still be in the corpus, because they’re made of more common subwords. For example, the word *antidisestablishmentarianism* isn’t a frequently used term, but it is made up of the letter combinations *anti*, *di*s, *est*, *ab*, *lish*, *ment*, *ari*, *an,* and *ism*, which are! It’s also a fun word to use with AI models that interpret your speech, to see if it can complete the word before you do!

As you can see, this can give you terrific *vocabulary efficiency*, which is the ability to maintain a manageable vocabulary size while still being able to capture meaningful semantics. It can also handle *out-of-vocabulary* effectively, which means being effective and handling unseen words by breaking them down into known subwords.

Here’s a really interesting example of this. In the very early days of transformers (pre-GPT), I worked on a project in which I created a transformer that I trained with the scripts of the TV show *Stargate,* and then I worked with the producers and actors on the show to do a table read of an AI-generated script. Given that the TV show is science fiction, with a lot of made-up words (aka technobabble), I used a subtoken tokenizer, following the logic that it could make up new words too. However, it ended up getting a little too creative! You can see the actors struggling with the new words in [this video of the table read](https://oreil.ly/A42Ko).

So now, let’s explore some common tokenizers in the ecosystem and see how they work. Note that tokenizers are associated with their model type, so when you explore models in the Hugging Face Hub (see [Chapter 14](ch14.html#ch14_using_third_party_models_and_hubs_1748549787242797)), you’ll be able to find their tokenizer. You must use the correct tokenizer with each model, or it won’t be able to understand your input. Depending on the licenses associated with the tokenizers, you could also use them to train or tune your own models, instead of rolling your own as we did in Chapters [4](ch04.html#ch04_using_data_with_pytorch_1748548966496246) and [5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759).

### The WordPiece tokenizer

The WordPiece tokenizer, which is associated with the BERT model, is a common tokenizer that’s highly efficient at managing subwords. It starts with a basic vocabulary and then iteratively adds the most frequent combinations it sees. The subwords are denoted by *##*. While generally created for English, it also works well for similar languages that have clear word boundaries denoted by spaces and other punctuation.

So now, let’s consider an example sentence that contains complex words:

```py
              # Let's use a sentence with some interesting words to tokenize
text = "The ultramarathoner prequalified for the 
        immunohistochemistry conference in neuroscience."
```

To load the tokenizer, you instantiate an instance of AutoTokenizer and call the `from_pretrained` method, passing the name of the tokenizer. For Bert’s WordPiece, you can use `bert-base-uncased` as the tokenizer name:

```py
from transformers import AutoTokenizer

# Load BERT tokenizer which uses WordPiece
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
```

Then, to tokenize the text, all you need to do is call the `tokenize` method and pass it your string. This will output the list of tokens:

```py
# Tokenize the text
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
```

This will then output the list of tokens from the sentence, which looks like this:

```py
Tokens: ['the', 'ultra', '##mara', '##th', '##one', '##r', 'pre', 
         '##qual', '##ified', 'for', 'the', 'im', '##mun', '##oh', '##isto', 
         '##chemist', '##ry', 'conference', 'in', 'neuroscience', '.']
```

Long words that aren’t commonly used (like *marathoner* and *immunohistochemistry*) are broken into subwords, whereas others (like *conference* and *neuroscience*) are kept as whole words. This is based on the training corpus used in BERT, and the decisions about which ones are common enough and which ones are not (for example, I would have expected *qualified* to be quite common) were made by the original researcher.

If you want to see the IDs for these tokens, you can get them with the encode method, like this:

```py
# Get the token IDs
token_ids = tokenizer.encode(text)
print("\nToken IDs:", token_ids)
```

The list of IDs is shown here:

```py
Token IDs: [101, 1996, 14357, 2108, 2339, 3840, 2837, 13462, 2005, 1996, 19763, 
            2172, 3075, 7903, 5273, 13172, 1027, 2005, 23021, 1012, 102]

```

Note that the first and last tokens are `101` and `102`, which are special tokens for the start and end of the sentence that the tokenizer inserted and which are expected by the model.

Now, say that you decode the list of token IDs back into a string, like this:

```py
# Decode back to show special tokens
decoded = tokenizer.decode(token_ids)
print("\nDecoded with special tokens:", decoded)
```

Then, you’ll see how the sentence has these special tokens inserted:

```py
Decoded with special tokens: [CLS] the ultramarathoner prequalified for the 
                             immunohistochemistry conference in 
                             neuroscience. [SEP]
```

I would recommend that you continue to experiment with the tokenizer to understand how it manages text by turning it into tokens and special characters. Having this knowledge is often important when you’re debugging model behavior or if you’re doing some kind of fine-tuning to manage how your vocabulary will be used.

### Byte-pair encoding

The GPT family uses a format called *byte-pair encoding* (BPE), which is a data compression algorithm as well as a tokenization one. It starts with the vocabulary of individual characters, which progressively learns common byte or character pairs to merge into new tokens.

The algorithm initially splits the training corpus into characters, assigning tokens to each. It then iteratively merges the most frequent pairs into new tokens, adding them to the vocabulary. The process continues for a predetermined number of merges. So, for example, over time, common patterns in words become their own tokens. This tends to veer toward the beginnings of the words with common prefixes (like *inter*) or the ends of words with common suffixes (like *er*) ending up having their own token. Instead of using the `##` string to determine the beginning of a subword, BPE uses a special character (usually *Ġ*).

Here’s the code you can use to tokenize the same sentence. You’ll use the `gpt2` AutoTokenizer from OpenAI:

```py
from transformers import AutoTokenizer

# Load GPT-2 tokenizer which uses BPE
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Same sentence as before
text = "The ultramarathoner prequalified for the immunohistochemistry 
        conference in neuroscience."

# Tokenize the text
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
```

The output is shown here:

```py
Tokens: ['The', 'Ġult', 'ram', 'ar', 'athon', 'er', 'Ġpre', 'qualified', 'Ġfor', 
         'Ġthe', 'Ġimmun', 'oh', 'ist', 'ochemistry', 'Ġconference', 
         'Ġin', 'Ġneuroscience', '.']
```

Over time, you’ll see that the splits will be slightly different, which reflects the difference between the training sets. BERT was trained on Wikipedia and the Toronto BookCorpus, while GPT-2 was trained on web text.

### SentencePiece

SentencePiece, which is used by the T5 model, is a unique tokenizer. It treats all input text as a raw sequence of Unicode characters, which gives it strong support for non-English languages. As part of this, it treats whitespaces like any other characters. That makes it effective for languages like Japanese and Chinese that don’t always have clear word boundaries, and it also removes the need for language-specific preprocessing. In fact, while it was being built, this tokenizer learned its subword units directly from raw sentences in multiple languages.

Here’s how to use it:

```py
from transformers import AutoTokenizer

# Load T5 tokenizer which uses SentencePiece
tokenizer = AutoTokenizer.from_pretrained('t5-base')

# Same sentence as before
text = "The ultramarathoner prequalified for the immunohistochemistry 
        conference in neuroscience."

# Tokenize the text
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
```

This produces the following set of tokens:

```py
Tokens: ['▁The', '▁ultra', 'marathon', 'er', '▁pre', 'qualified', '▁for', 
         '▁the', '▁immun', 'oh', 'ist', 'ochemistry', '▁conference', '▁in',
         '▁neuroscience', '.']

```

As mentioned, where it’s particularly powerful is with non-English languages. So, for example, consider this code:

```py
# Let's also try a multilingual example with mixed scripts
text2 = "Tokyo 東京 is beautiful! Preprocessing in 2024 costs $123.45"
tokens2 = tokenizer.tokenize(text2)
print("\nMultilingual example tokens:", tokens2)
```

It will output the following:

```py
Multilingual example tokens: ['▁Tokyo', '▁東', '京', '▁is', '▁beautiful', '!', 
                              '▁Pre', '-', 'processing', '▁in', '▁2024', 
                              '▁costs', '▁$', '123', '.', '45']
```

Note how the Japanese characters for Tokyo were split into multiple tokens and the numbers were kept whole (i.e., `123` and `45`).

Given that transformers were initially designed to improve machine translation, you can see that SentencePiece, which predates generative AI like GPT, was designed with internationalization in mind!

# Summary

In this chapter, we looked at Transformers, the architecture that underpins modern LLMs, and transformers, the library from Hugging Face that makes Transformers easy to use.

We explored how the original Transformer architecture revolutionized AI through its use of attention mechanisms, in which context vectors for words were amended based on learned details of where the sequence might be paying appropriate attention to other parts of the sequence. We also looked at encoders that excel at artificial understanding of text, decoders that can intelligently generate text, and encoder-decoders that bring the best of both for sequence-to-sequence models. We also double-clicked into their architecture to understand how the mechanisms such as attention, feedforward, normalization, and many other parts of the architecture work.

We then looked into transformers, which form the library from Hugging Face that makes downloading and instantiation of Transformer-based models (including the entire inference pipeline) very easy. There’s a whole lot more still in there, and hopefully, this gave you a good head start.

In the next chapter, you’re going to go a little further and explore how to adapt LLM models to your specific needs, taking custom data and using it to fine-tune or prompt tune models to make them work for your specific use cases. Get ready to turn theory into practice!