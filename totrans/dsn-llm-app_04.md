# Chapter 3\. Vocabulary and Tokenization

In [Chapter 2](ch02.html#ch02), we dug deep into the datasets that are used to train the language models of today, including the process of creating them. Hopefully this foray has underscored how influential pre-training data is to the resulting model. In this chapter, we will discuss another fundamental ingredient of a language model: its vocabulary.

# Vocabulary

What do you do first when you start learning a new language? You start acquiring its vocabulary, expanding it as you gain more proficiency in the language. Let’s define vocabulary here as:

> All the words in a language that are understood by a specific person.

The average native English speaker has a vocabulary of [20,000–35,000 words](https://oreil.ly/bkc2C). Similarly, every language model has its own vocabulary, with most vocabulary sizes ranging anywhere between 5,000 and 500,000 *tokens*.

As an example, let us explore the vocabulary of the GPT-NeoX-20B model. Open the file [*tokenizer.json*](https://oreil.ly/Kages) and Ctrl+F for “vocab,” a dictionary containing the vocabulary of the model. You can see that the words comprising the language model vocabulary don’t entirely look like English language words that appear in a dictionary. These word-like units are called “types,” and the instantiation of a type (when it appears in a sequence of text) is called a token.

###### Note

Recently, and especially in industry, I seldom hear anyone use the term “type” except in older NLP textbooks. The term “token” is broadly used to refer to both the vocabulary units and when they appear in a text sequence. We will henceforth use the word “token” to describe both concepts, even though I personally am not the biggest fan of this usage.

In the vocabulary file, we see that next to each token is a number, which is called the *input ID* or the *token index*. The vocabulary size of GPT-NeoX is just above 50,000.

Looking at the vocabulary file in detail, we notice that the first few hundred tokens are all single-character tokens, such as special characters, digits, capital letters, small letters, and accented characters. Longer words appear later on in the vocabulary. A lot of tokens correspond to just a part of a word, called a *subword*, like “impl,” “inated,” and so on.

Let’s Ctrl+F for “office.” We get nine results:

```py
"Ġoffice": 3906
"Ġofficer": 5908
"Ġofficers": 6251
"ĠOffice": 7454
"ĠOfficer": 12743
"Ġoffices": 14145
"office": 30496
"Office": 33577
"ĠOfficers": 37209
```

The Ġ character refers to a space before the word. For instance, in the sentence, “He stopped going to the office,” the space before the letter “o” in the word “office” is considered part of the token. You can see that the tokens are case-sensitive: there are separate tokens for “office” and “Office.” Most models these days have case-sensitive vocabularies. Back in the day, the BERT model was released with both a cased and an uncased version.

###### Note

Language models learn vector representations called embeddings for each of these tokens that reflect their syntactic and semantic meaning. We will go through the learning process in [Chapter 4](ch04.html#chapter_transformer-architecture), and dive deeper into embeddings in [Chapter 11](ch11.html#chapter_llm_interfaces).

Cased vocabularies are almost always better, especially when you are training on such a huge body of text such that most tokens are seen by the model enough times to learn distinctive representations for them. For instance, there is a definite semantic difference between “web” and “Web,” and it is good to have separate tokens for them.

Let’s search for some numbers. Ctrl+F for “93.” There are only three results:

```py
"93": 4590
"937": 47508
"930": 48180
```

It seems like not all numbers get their own tokens! Where is the token for 934? It is impractical to give every number its own token, especially if you want to limit your vocabulary size to say, just 50,000\. Later in this chapter, we will discuss how vocabulary sizes are determined. Popular names and places get their own token. There is a token representing Boston, Toronto, and Amsterdam, but none representing Mesa or Chennai. There is a token representing Ahmed and Donald, but none for Suhas or Maryam.

You might have noticed that tokens like:

```py
"]);": 9259
```

exist, indicating that GPT-NeoX is also primed to process programming languages.

How are vocabularies determined? Surely, there was no executive committee holding emergency meetings burning midnight oil, with members making impassioned pleas to include the number 937 in the vocabulary at the expense of 934.

Let us revisit the definition of a vocabulary:

> All the words in a language that are understood by a specific person.

Since we want our language model to be an expert at English, we can just include all words in the English dictionary as part of its vocabulary. Problem solved?

Not nearly. What do you do when you communicate with the language model using a word that it has never seen? This happens a lot more often than you think. New words get invented all the time, words have multiple forms (“understand,” “understanding,” “understandable”), multiple words can be combined into a single word, and so on. Moreover, there are millions of domain-specific words (biomedical, chemistry, and so on).

###### Note

The account [@NYT_first_said](https://oreil.ly/FzfI9) on the social media platform X posts words except proper nouns when they appear in the *New York Times* for the first time. Each day, an average of five new words appear in the US paper of record for the first time ever. On the day I wrote this section, the words were “unflippant,” “dumbeyed,” “dewdrenched,” “faceflat,” “saporous,” and “dronescape.” Many of these words might never get added to a dictionary.

A token that doesn’t exist in the vocabulary is called an out-of-vocabulary (OOV) token. Traditionally, OOV tokens were represented using a special <UNK> token. The <UNK> token is a placeholder for all tokens that don’t exist in the vocabulary. All OOV tokens share the same embedding (and encode the same meaning), which is undesirable. Moreover, the <UNK> token cannot be used in generative models. You don’t want your model to output something like:

```py
'As a language model, I am trained to <UNK> sequences, and output <UNK> text'.
```

To solve the OOV problem, one possible solution could be to represent tokens in terms of characters instead of words. Each character has its own embedding, and as long as all valid characters are included in the vocabulary, there will never be a chance of encountering an OOV token. However, there are many downsides to this. The number of tokens needed to represent the average sentence becomes much larger. For example, the sentence, “The number of tokens needed to represent the average sentence becomes much larger,” contains 13 tokens when you treat each word as a token, but 81 tokens when you treat each character as a token. This reduces the amount of content you can represent within a fixed sequence length, which makes both model training and inference slower, as we will show further in [Chapter 4](ch04.html#chapter_transformer-architecture). Models support a limited sequence length, so this also reduces the amount of content you can fit in a single prompt. Later in this chapter, we will discuss models like CANINE, ByT5, and Charformer that attempt to use character-based tokens.

So, the middle ground and the best of both worlds (or the worst of both worlds—the field hasn’t come to a consensus yet) is using subwords. Subwords are the predominant mode of representing vocabulary units in the language model space today. The GPT-NeoX vocabulary we explored earlier uses subword tokens. [Figure 3-1](#subword-tokens) shows the OpenAI tokenizer playground that demonstrates how words are split into their constituent subwords by OpenAI models.

![Subword tokens](assets/dllm_0301.png)

###### Figure 3-1\. Subword tokens

# Tokenizers

Next, let’s dive into tokenizers, the software that serves as a text-processing interface between humans and models.

A tokenizer has two responsibilities:

1.  In the tokenizer pre-training stage, the tokenizer is run over a body of text to generate a vocabulary.

2.  While processing input during both model training and inference, free-form raw text is run through the tokenizer algorithm to break the text into sequences of valid tokens. [Figure 3-2](#tokenizer-workflow) depicts the roles played by a tokenizer.

![Tokenizer Workflows](assets/dllm_0302.png)

###### Figure 3-2\. Tokenizer workflow

When we feed raw text to the tokenizer, it breaks the text into tokens that are part of the vocabulary and maps the tokens to their token indices. The sequence of token indices (input IDs) are then fed to the language model, where they are mapped to their corresponding embeddings. Let us explore this process in detail.

This time, let’s experiment with the FLAN-T5 model. You need a Google Colab Pro or equivalent system to be able to run it:

```py
!pip install transformers accelerate sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-largel")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large",
    device_map="auto")

input_text = "what is 937 + 934?"
encoded_text = tokenizer.encode(input_text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text)
print(encoded_text)
print(tokens)
```

The output is:

```py
[125, 19, 668, 4118, 1768, 668, 3710, 58, 1]
['▁what', '▁is', '▁9', '37', '▁+', '▁9', '34', '?', '</s>']
```

The `encode()` function tokenizes the input text and returns the corresponding token indices. The token indices are mapped to the tokens they represent using the `convert_ids_to_tokens()` function.

As you can see, the FLAN-T5 tokenizer doesn’t have dedicated tokens for the numbers 937 or 934\. Therefore, it splits the numbers into “9” and “37.” The `</s>` token is a special token indicating the end of the string. The `_` means that the token is preceded by a space.

Let’s try another example:

```py
input_text = "Insuffienct adoption of corduroy pants is the reason this

economy is in the dumps!!!"
encoded_text = tokenizer.encode(input_text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text)
print(tokens)
```

The output is:

```py
['▁In', 's', 'uff', 'i', 'en', 'c', 't', '▁adoption', '▁of', '▁cord', 'u',
'roy', '▁pants', '▁is', '▁the', '▁reason', '▁this', '▁economy', '▁is', '▁in',
'▁the', '▁dump', 's', '!!!', '</s>']
```

I made a deliberate typo with the word “Insufficient.” Note that subword tokenization is rather brittle with typos. But at least the OOV problem has been dealt with by breaking the words into subwords. The vocabulary also doesn’t seem to have an entry for the word “corduroy,” thus confirming its poor sense of fashion. Meanwhile, note that there is a distinct token for three contiguous exclamation points, which is different from the token that represents a single exclamation point. Semantically, they do convey slightly different meanings.

###### Note

Very large models trained on a massive body of text are more robust to misspellings. A lot of misspellings already occur in the training set. For example, even the rare misspelling “Insuffienct” occurs 14 times in the C4 pre-training dataset. The more common misspelling “insufficent” occurs over 1,100 times. Larger models can also infer the misspelled word from its context. Smaller models like BERT are quite sensitive to misspellings.

If you are using models from OpenAI, you can explore their tokenization scheme using the [tiktoken library](https://oreil.ly/2QByi) (no relation to the social media website).

Using tiktoken, let’s see the different vocabularies available in the OpenAI ecosystem:

```py
!pip install tiktoken

import tiktoken
tiktoken.list_encoding_names()
```

The output is:

```py
['gpt2', 'r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base', 'o200k_base']
```

The numbers like 50K/100K are presumed to be the vocabulary size. OpenAI hasn’t revealed much information about these vocabularies. Their documentation does state that o200k_base is used by GPT-4o, while cl100k_base is used by GPT-4:

```py
encoding = tiktoken.encoding_for_model("gpt-4")
input_ids = encoding.encode("Insuffienct adoption of corduroy pants is the

reason this economy is in the dumps!!!")
tokens = [encoding.decode_single_token_bytes(token) for token in input_ids]
```

The output is:

```py
[b'Ins', b'uff', b'ien', b'ct', b' adoption', b' of', b' cord', b'uro', b'y',
b' pants', b' is', b' the', b' reason', b' this', b' economy', b' is', b' in',
b' the', b' dumps', b'!!!']
```

As you can see there is not much difference between the tokenization used by GPT-4 and FLAN-T5.

###### Tip

For a given task, if you observe strange behavior from LLMs on only a subset of your inputs, it is worthwhile to check how they have been tokenized. While you cannot definitively diagnose your problem just by analyzing the tokenization, it is often helpful in analysis. In my experience, a non-negligible number of LLM failures can be attributed to the way the text was tokenized. This is especially true if your target domain is different from the pre-training domain.

# Tokenization Pipeline

[Figure 3-3](#huggingface-tokenizers-pipeline) depicts the sequence of steps performed by a tokenizer.

![Hugging Face Tokenizers pipeline](assets/dllm_0303.png)

###### Figure 3-3\. Hugging Face tokenizers pipeline

If you are using the `tokenizers` library from Hugging Face, your input text is run through a [multistage tokenization pipeline](https://oreil.ly/CcOKV). This pipeline is composed of four components:

*   Normalization

*   Pre-tokenization

*   Tokenization

*   Postprocessing

Note that different models will execute different steps within these four components.

## Normalization

Different types of normalization applied include:

*   Converting text to lowercase (if you are using an uncased model)

*   Stripping off accents from characters, like from the word Peña

*   Unicode normalization

Let’s see what kind of normalization is applied on the uncased version of BERT:

```py
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.backend_tokenizer.normalizer.normalize_str(
    'Pédrò pôntificated at üs:-)')
```

The output is:

```py
pedro pontificated at us:-)
```

As we can see, the accents have been removed and the text has been converted to lowercase.

There isn’t much normalization done in tokenizers for more recent models.

## Pre-Tokenization

Before we run the tokenizer on the text, we can optionally perform a pre-tokenization step. As mentioned earlier, most tokenizers today employ subword tokenization. A common step is to first perform word tokenization and then feed the output of it to the subword tokenization algorithm. This step is called pre-tokenization.

Pre-tokenization is a relatively easy step in English compared to many other languages, since you can start with a very strong baseline just by splitting text on whitespace. There are outlier decisions to be made, such as how to deal with punctuation, multiple spaces, numbers, etc. In Hugging Face the regular expression:

```py
\w+|[^\w\s]+
```

is used to split on whitespace.

Let’s run the pre-tokenization step of the T5 tokenizer:

```py
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("I'm starting to

suspect - I am 55 years old!   Time to vist New York?")
```

The output is:

```py
[("▁I'm", (0, 3)),
 ('▁starting', (3, 12)),
 ('▁to', (12, 15)),
 ('▁suspect', (15, 23)),
 ('▁-', (23, 25)),
 ('▁I', (25, 27)),
 ('▁am', (27, 30)),
 ('▁55', (30, 33)),
 ('▁years', (33, 39)),
 ('▁old!', (39, 44)),
 ('▁', (44, 45)),
 ('▁', (45, 46)),
 ('▁Time', (46, 51)),
 ('▁to', (51, 54)),
 ('▁vist', (54, 59)),
 ('▁New', (59, 63)),
 ('▁York?', (63, 69))]
```

Along with the pre-tokens (or word tokens), the character offsets are returned.

The T5 pre-tokenizer splits only on whitespace, doesn’t collapse multiple spaces into one, and doesn’t split on punctuation or numbers. The behavior can be vastly different for other tokenizers.

## Tokenization

After the optional pre-tokenization step, the actual tokenization step is performed. Some of the important algorithms in this space are byte pair encoding (BPE), byte-level BPE, WordPiece, and Unigram LM. The tokenizer comprises a set of rules that is learned during a pre-training phase over a pre-training dataset. Now let’s go through these algorithms in detail.

## Byte Pair Encoding

This algorithm is the simplest and most widely used tokenization algorithm.

### Training stage

We take a training dataset, run it through the normalization and pre-tokenization steps discussed earlier, and record the unique tokens in the resulting output and their frequencies. We then construct an initial vocabulary consisting of the unique characters that make up these tokens. Starting from this initial vocabulary, we continue adding new tokens using *merge* rules. The merge rule is simple; we create a new token using the most frequent consecutive pairs of tokens. The merges continue until we reach the desired vocabulary size.

Let’s explore this with an example. Imagine our training dataset is composed of six words, each appearing just once:

```py
'bat', 'cat', 'cap', 'sap', 'map', 'fan'
```

The initial vocabulary is then made up of:

```py
'b', 'a', 't', 'c', 'p', 's', 'm', 'f', 'n'
```

The frequencies of contiguous token pairs are:

```py
'ba' - 1, 'at' - 2, 'ca' - 2, 'ap' - 3, 'sa' - 1, 'ma' - 1, 'fa' - 1, 'an' - 1
```

The most frequent pair is “ap,” so the first merge rule is to merge “a” and “p.” The vocabulary now is:

```py
'b', 'a', 't', 'c', 'p', 's', 'm', 'f', 'n', 'ap'
```

The new frequencies are:

```py
'ba' - 1, 'at' - 2, 'cap' - 1, 'sap' - 1, 'map' - 1, 'fa' - 1, 'an' - 1
```

Now, the most frequent pair is “at,” so the next merge rule is to merge “a” and “t.” This process continues until we reach the vocabulary size.

### Inference stage

After the tokenizer has been trained, it can be used to divide the text into appropriate subword tokens and feed the text into the model. This happens in a similar fashion as the training step. After normalization and pre-tokenization of the input text, the resulting tokens are broken into individual characters, and all the merge rules are applied in order. The tokens standing after all merge rules have been applied are the final tokens, which are then fed to the model.

You can open the [vocabulary file](https://oreil.ly/7JAyY) for GPT-NeoX again, and Ctrl+F “merges” to see the merge rules. As expected, the initial merge rules join single characters with each other. At the end of the merge list, you can see larger subwords like “out” and “comes” being merged into a single token.

###### Note

Since all unique individual characters in the tokenizer training set will get their own token, it is guaranteed that there will be no OOV tokens as long as all tokens seen during inference in the future are made up of characters that were present in the training set. But Unicode consists of over a million code points and around 150,000 valid characters, which would not fit in a vocabulary of size 30,000\. This means that if your input text contained a character that wasn’t in the training set, that character would be assigned an <UNK> token. To resolve this, a variant of BPE called byte-level BPE is used. Byte-level BPE starts with 256 tokens, representing all the characters that can be represented by a byte. This ensures that every Unicode character can be encoded just by the concatenation of the constituent byte tokens. Hence, it also ensures that we will never encounter an <UNK> token. The GPT family of models use this tokenizer.

## WordPiece

WordPiece is similar to BPE, so we will highlight only the differences.

Instead of the frequency approach used by BPE, WordPiece uses the maximum likelihood approach. The frequency of the token pairs in the dataset is normalized by the product of the frequency of the individual tokens. The pairs with the resulting highest score are then merged:

```py
score = freq(a,b)/(freq(a) * freq(b))
```

This means that if a token pair is made up of tokens that individually have low frequency, they will be merged first.

[Figure 3-4](#WordPiece) shows the merge priority and how the normalization by individual frequencies affects the order of merging.

![WordPiece tokenization](assets/dllm_0304.png)

###### Figure 3-4\. WordPiece tokenization

During inference, merge rules are not used. Instead, for each pre-tokenized token in the input text, the tokenizer finds the longest subword from the vocabulary in the token and splits on it. For example, if the token is “understanding” and the longest subword in the dictionary within this token is “understand,” then it will be split into “understand” and “ing.”

### Postprocessing

Now that we have looked at a couple of tokenizer algorithms, let’s move on to the next stage of the pipeline, the postprocessing stage. This is where model-specific special tokens are added. Common tokens include [CLS], the classification token used in many language models, and [SEP], a separator token used to separate parts of the input.

## Special Tokens

Depending on the model, a few special tokens are added to the vocabulary to facilitate processing. These tokens can include:

<PAD>

To indicate padding, in case the size of the input is less than the maximum sequence length.

<EOS>

To indicate the end of the sequence. Generative models stop generating after outputting this token.

<UNK>

To indicate an OOV term.

<TOOL_CALL>, </TOOL_CALL>

Content between these tokens is used as input to an external tool, like an API call or a query to a database.

<TOOL_RESULT>, </TOOL_RESULT>

Content between these tokens is used to represent the results from calling the aforementioned tools.

As we have seen, if our data is domain-specific like healthcare, scientific literature, etc., tokenization from a general-purpose tokenizer will be unsatisfactory. GALACTICA by Meta introduced several domain-specific tokens in their model and special tokenization rules:

*   [START_REF] and [END_REF] for wrapping citations.

*   <WORK> to wrap tokens that make up an internal working memory, used for reasoning and code generation.

*   Numbers are handled by assigning each digit in the number its own token.

*   [START_SMILES], [START_DNA], [START_AMINO], [END_SMILES], [END_DNA], [END_AMINO] for protein sequences, DNA sequences, and amino acid sequences, respectively.

If you are using a model on domain-specific data like healthcare, finance, law, biomedical, etc., with a tokenizer that was trained on general-purpose data, the compression ratio will be relatively lower because domain-specific words do not have their own tokens and will be split into multiple tokens. One way to adapt models to specialized domains is for models to learn good vector representations for domain-specific terms.

To this end, we can add new tokens to existing tokenizers and continue pre-training the model on domain-specific data so that those new domain-specific tokens learn effective representations. We will learn more about continued pre-training in [Chapter 7](ch07.html#ch07).

For now, let’s see how we can add new tokens to a vocabulary using Hugging Face.

Consider the sentence, “The addition of CAR-T cells and antisense oligonucleotides drove down incidence rates.” The FLAN-T5 tokenizer splits this text as follows:

*   ['▁The', '▁addition', '▁of', '▁C', ' AR', '-', ' T', '▁cells', '▁and', '▁anti', ' s', ' ense', '▁', ' oli', ' gon', ' u', ' cle', ' o', ' t', ' ides', '▁drove', '▁down', '▁incidence', '▁rates', ' .', '</s>']

Let’s add the domain-specific terms to the vocabulary:

```py
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large",
    device_map="auto")

tokenizer.add_tokens(["CAR-T", "antisense", "oligonucleotides"])
model.resize_token_embeddings(len(tokenizer))
```

Now, tokenizing the string again gives the following tokens, with the domain-specific tokens being added:

*   ['▁The', '▁addition', '▁of', ' CAR-T', '▁cells', '▁and', ' antisense', ' oligonucleotides', '▁drove', '▁down', '▁incidence', '▁rates', ' .', '</s>']

We are only halfway done here. The embedding vectors corresponding to these new tokens do not contain any information about these tokens. We will need to learn the right representations for these tokens, which we can do using fine-tuning or continued pre-training, which we will discuss in [Chapter 7](ch07.html#ch07).

# Summary

In this chapter, we focused on a key ingredient of language models: their vocabulary. We discussed how vocabularies are defined and constructed in the realm of language models. We introduced the concept of tokenization and presented tokenization algorithms like BPE and WordPiece that are used to construct vocabularies and break down raw input text into a sequence of tokens that can be consumed by the language model. We also explored the vocabularies of popular language models and noted how tokens can differ from human conceptions of a word.

In the next chapter, we will continue exploring the remaining ingredients of a language model, including its architecture and the learning objectives on which models are trained.