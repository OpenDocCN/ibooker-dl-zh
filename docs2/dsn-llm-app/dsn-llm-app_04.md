# 第三章\. 词汇和分词

在第二章中，我们深入探讨了用于训练当今语言模型的语料库，包括创建它们的过程。希望这次探索强调了预训练数据对最终模型的影响。在本章中，我们将讨论语言模型的另一个基本组成部分：其词汇。

# 词汇

当你开始学习一门新语言时，你首先做什么？你开始积累它的词汇，随着你在语言上的熟练程度提高，不断扩大词汇量。在这里，我们将词汇定义为：

> 一个特定的人所能理解的语言中的所有单词。

平均的母语英语说话者的词汇量在[20,000–35,000]个单词之间（[`oreil.ly/bkc2C`](https://oreil.ly/bkc2C)）。同样，每个语言模型都有自己的词汇表，其中大多数词汇量的大小在 5,000 到 500,000 个*分词*之间。

例如，让我们探索 GPT-NeoX-20B 模型的词汇。打开文件[*tokenizer.json*](https://oreil.ly/Kages)并使用 Ctrl+F 查找“vocab”，这是一个包含模型词汇的字典。你可以看到构成语言模型词汇的单词并不完全像字典中出现的英语单词。这些类似单词的单位被称为“类型”，而类型的实例化（它在文本序列中出现时）被称为分词。

###### 注意

最近，尤其是在工业界，我很少听到有人使用“类型”这个词，除了在较老的自然语言处理教科书中。术语“分词”广泛用于指代词汇单元以及它们在文本序列中的出现。因此，我们将使用“分词”这个词来描述这两个概念，尽管我个人并不是特别喜欢这种用法。

在词汇文件中，我们可以看到每个分词旁边都有一个数字，这被称为*输入 ID*或*分词索引*。GPT-NeoX 的词汇量刚刚超过 50,000。

详细查看词汇文件，我们会注意到前几百个分词都是单字符分词，例如特殊字符、数字、大写字母、小写字母和带重音的字符。较长的单词出现在词汇表的后面。许多分词对应于单词的一部分，称为*子词*，如“impl”、“inated”等等。

让我们使用 Ctrl+F 查找“office”。我们得到了九个结果：

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

Ġ字符表示单词前的空格。例如，在句子“他停止去办公室”中，“office”单词中“o”字母前的空格被认为是分词的一部分。你可以看到分词是区分大小写的：有“office”和“Office”两个不同的分词。如今，大多数模型都有区分大小写的词汇表。在早期，BERT 模型发布时，既有带大小写的版本，也有不带大小写的版本。

###### 注意

语言模型为每个这些标记学习向量表示，称为嵌入，这些嵌入反映了它们的句法和语义意义。我们将在第四章中介绍学习过程，并在第十一章中更深入地探讨嵌入。

有标记的词汇几乎总是更好的，尤其是当你在一个如此庞大的文本体上进行训练时，大多数标记都被模型足够多次地看到，以便学习它们的独特表示。例如，“web”和“Web”之间有明显的语义差异，为它们分别保留单独的标记是好的。

让我们搜索一些数字。使用 Ctrl+F 搜索“93。”只有三个结果：

```py
"93": 4590
"937": 47508
"930": 48180
```

看起来并非所有数字都有自己的标记！934 的标记在哪里？给每个数字分配一个标记是不切实际的，尤其是如果你想将词汇量限制在比如说 50,000 个单词以内。在本章的后面，我们将讨论词汇量是如何确定的。流行的人名和地名有自己的标记。有一个标记代表波士顿、多伦多和阿姆斯特丹，但没有代表梅萨或钦奈的标记。有一个标记代表艾哈迈德和唐纳德，但没有代表苏哈斯或玛丽亚姆的标记。

你可能已经注意到像这样的标记：

```py
"]);": 9259
```

存在，这表明 GPT-NeoX 也准备好处理编程语言。

词汇是如何确定的？当然，没有执行委员会在紧急会议上熬夜，成员们为了将数字 937 纳入词汇而牺牲 934 而做出激动的请求。

让我们重新审视词汇的定义：

> 一个特定的人能理解的语言中的所有单词。

由于我们希望我们的语言模型精通英语，我们只需将其英语词典中的所有单词作为其词汇的一部分即可。问题解决了吗？

远远不够。当你使用语言模型从未见过的单词进行交流时，你会怎么做？这种情况比你想象的要频繁得多。新词不断被创造出来，单词有多种形式（“understand”、“understanding”、“understandable”），多个单词可以组合成一个单词，等等。此外，还有数百万个特定领域的单词（生物医学、化学等等）。

###### 注意

社交媒体平台 X 上的账户[@NYT_first_said](https://oreil.ly/FzfI9)在首次出现在《纽约时报》时，除了专有名词外，会发布单词。每天，平均有五个新单词首次出现在美国记录报纸上。在我撰写这一部分的时候，这些单词是“unflippant”、“dumbeyed”、“dewdrenched”、“faceflat”、“saporous”和“dronescape”。其中许多单词可能永远不会被添加到词典中。

不存在于词汇表中的标记被称为词汇外（OOV）标记。传统上，OOV 标记使用特殊的 <UNK> 标记表示。<UNK> 标记是所有不在词汇表中存在的标记的占位符。所有 OOV 标记共享相同的嵌入（并编码相同的意义），这是不理想的。此外，<UNK> 标记不能用于生成模型。你不想你的模型输出类似的内容：

```py
'As a language model, I am trained to <UNK> sequences, and output <UNK> text'.
```

为了解决 OOV 问题，一个可能的解决方案可能是用字符而不是单词来表示标记。每个字符都有自己的嵌入，只要所有有效字符都包含在词汇表中，就永远不会遇到 OOV 标记。然而，这种方法有很多缺点。表示平均句子所需的标记数量变得很大。例如，句子“表示平均句子所需的标记数量变得很大”当每个单词被视为一个标记时包含 13 个标记，但当每个字符被视为一个标记时包含 81 个标记。这减少了在固定序列长度内可以表示的内容量，这会使模型训练和推理速度变慢，我们将在第四章中进一步展示。模型支持有限的序列长度，这也减少了可以在单个提示中放入的内容量。在本章的后面部分，我们将讨论像 CANINE、ByT5 和 Charformer 这样的模型，它们试图使用基于字符的标记。

因此，折中方案和两者之最佳（或者两者之最差——该领域尚未达成共识）是使用子词。子词是目前在语言模型空间中表示词汇单位的主要方式。我们之前探索的 GPT-NeoX 词汇表使用子词标记。图 3-1 展示了 OpenAI 分词器游乐场，该游乐场演示了 OpenAI 模型如何将单词分解为其构成子词。

![子词标记](img/dllm_0301.png)

###### 图 3-1\. 子词标记

# 分词器

接下来，让我们深入了解分词器，它是介于人类和模型之间的文本处理接口的软件。

分词器有两个职责：

1.  在分词器预训练阶段，分词器在一段文本上运行以生成词汇表。

1.  在模型训练和推理过程中处理输入时，自由形式的原始文本通过分词器算法进行处理，将其分解为有效的标记序列。图 3-2 描述了分词器所扮演的角色。

![分词器工作流程](img/dllm_0302.png)

###### 图 3-2\. 分词器工作流程

当我们将原始文本输入到分词器中时，它会将文本分解为词汇表中的标记，并将标记映射到它们的标记索引。然后，标记索引的序列（输入 ID）被输入到语言模型中，在那里它们被映射到相应的嵌入。让我们详细探讨这个过程。

这次，让我们实验一下 FLAN-T5 模型。你需要一个 Google Colab Pro 或等效的系统才能运行它：

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

输出结果为：

```py
[125, 19, 668, 4118, 1768, 668, 3710, 58, 1]
['▁what', '▁is', '▁9', '37', '▁+', '▁9', '34', '?', '</s>']
```

`encode()` 函数将输入文本进行分词，并返回相应的标记索引。使用 `convert_ids_to_tokens()` 函数将标记索引映射到它们所代表的标记。

如你所见，FLAN-T5 分词器没有为数字 937 或 934 设置专门的标记。因此，它将数字拆分为“9”和“37”。`</s>` 标记是一个特殊标记，表示字符串的结束。`_` 表示该标记前面有一个空格。

让我们再试一个例子：

```py
input_text = "Insuffienct adoption of corduroy pants is the reason this `economy` `is` `in` `the` `dumps``!!!``"` ``` `encoded_text` `=` `tokenizer``.``encode``(``input_text``)` `tokens` `=` `tokenizer``.``convert_ids_to_tokens``(``encoded_text``)` `print``(``tokens``)` ```py
```

```py```` ```py``` 输出结果为：    ``` ['▁In', 's', 'uff', 'i', 'en', 'c', 't', '▁adoption', '▁of', '▁cord', 'u', 'roy', '▁pants', '▁is', '▁the', '▁reason', '▁this', '▁economy', '▁is', '▁in', '▁the', '▁dump', 's', '!!!', '</s>'] ```py    我故意在单词“Insufficient”上犯了一个拼写错误。请注意，子词分词对拼写错误相当敏感。但至少通过将单词分解为子词，已经解决了 OOV 问题。词汇表似乎也没有“corduroy”这个词的条目，从而证实了它糟糕的时尚感。同时，请注意，有三个连续感叹号有一个独特的标记，这与表示单个感叹号的标记不同。从语义上看，它们确实传达了略微不同的含义。    ###### 注意    在大量文本上训练的非常大的模型对拼写错误更加鲁棒。训练集中已经存在许多拼写错误。例如，即使是罕见的拼写错误“Insuffienct”在 C4 预训练数据集中也出现了 14 次。更常见的拼写错误“insufficent”出现了超过 1,100 次。更大的模型还可以从上下文中推断出拼写错误的单词。较小的模型，如 BERT，对拼写错误非常敏感。    如果你正在使用 OpenAI 的模型，你可以通过[tiktoken 库](https://oreil.ly/2QByi)（与社交媒体网站无关）来探索它们的分词方案。    使用 tiktoken，让我们看看 OpenAI 生态系统中可用的不同词汇表：    ``` !pip install tiktoken  import tiktoken tiktoken.list_encoding_names() ```py    输出结果为：    ``` ['gpt2', 'r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base', 'o200k_base'] ```py    数字如 50K/100K 被认为是词汇表的大小。OpenAI 没有透露太多关于这些词汇表的信息。他们的文档确实指出，o200k_base 被 GPT-4o 使用，而 cl100k_base 被 GPT-4 使用：    ``` encoding = tiktoken.encoding_for_model("gpt-4") input_ids = encoding.encode("Insuffienct adoption of corduroy pants is the `reason` `this` `economy` `is` `in` `the` `dumps``!!!``")` ```py `tokens` `=` `[``encoding``.``decode_single_token_bytes``(``token``)` `for` `token` `in` `input_ids``]` ``` ```py   ````` ```py`The output is:    ``` [b'Ins', b'uff', b'ien', b'ct', b' adoption', b' of', b' cord', b'uro', b'y', b' pants', b' is', b' the', b' reason', b' this', b' economy', b' is', b' in', b' the', b' dumps', b'!!!'] ```py    As you can see there is not much difference between the tokenization used by GPT-4 and FLAN-T5.    ###### Tip    For a given task, if you observe strange behavior from LLMs on only a subset of your inputs, it is worthwhile to check how they have been tokenized. While you cannot definitively diagnose your problem just by analyzing the tokenization, it is often helpful in analysis. In my experience, a non-negligible number of LLM failures can be attributed to the way the text was tokenized. This is especially true if your target domain is different from the pre-training domain.```` ```py`` ``````py ``````py`  ``````py`` ``````py` ``````py # 分词流程    图 3-3 展示了分词器执行的步骤序列。  ![Hugging Face 分词器流程](img/dllm_0303.png)  ###### 图 3-3\. Hugging Face 分词器流程    如果你正在使用 Hugging Face 的`tokenizers`库，你的输入文本将通过一个[多阶段分词流程](https://oreil.ly/CcOKV)。这个流程由四个组件组成：    *   正规化           *   预分词           *   分词           *   后处理              注意，不同的模型将在这四个组件中执行不同的步骤。    ## 正规化    应用了不同类型的正规化包括：    *   将文本转换为小写（如果你使用的是无大小写模型）           *   从字符中去除重音，例如从单词 Peña 中去除           *   Unicode 正规化              让我们看看 BERT 的无大小写版本应用了什么样的正规化：    ``` tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") print(tokenizer.backend_tokenizer.normalizer.normalize_str(     'Pédrò pôntificated at üs:-)') ```py    输出结果为：    ``` pedro pontificated at us:-) ```py    如我们所见，重音已被去除，文本已被转换为小写。    对于更近期的模型，分词器中进行的正规化并不多。    ## 预分词    在我们对文本运行分词器之前，我们可以选择执行预分词步骤。如前所述，大多数分词器今天都采用子词分词。一个常见的步骤是首先执行单词分词，然后将输出传递给子词分词算法。这一步骤称为预分词。    相比于许多其他语言，预分词在英语中是一个相对简单的步骤，因为你可以通过在空白处分割文本来获得一个非常强大的基线。有一些异常决策需要做出，例如如何处理标点符号、多个空格、数字等。在 Hugging Face 中，正则表达式：    ``` \w+|[^\w\s]+ ```py    用于在空白处分割。    让我们运行 T5 分词器的预分词步骤：    ``` tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl") tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("I'm starting to `suspect` `-` `I` `am` `55` `years` `old``!`   `Time` `to` `vist` `New` `York``?``")` ```py   ````` ```py`The output is:    ``` [("▁I'm", (0, 3)),  ('▁starting', (3, 12)),  ('▁to', (12, 15)),  ('▁suspect', (15, 23)),  ('▁-', (23, 25)),  ('▁I', (25, 27)),  ('▁am', (27, 30)),  ('▁55', (30, 33)),  ('▁years', (33, 39)),  ('▁old!', (39, 44)),  ('▁', (44, 45)),  ('▁', (45, 46)),  ('▁Time', (46, 51)),  ('▁to', (51, 54)),  ('▁vist', (54, 59)),  ('▁New', (59, 63)),  ('▁York?', (63, 69))] ```py    Along with the pre-tokens (or word tokens), the character offsets are returned.    The T5 pre-tokenizer splits only on whitespace, doesn’t collapse multiple spaces into one, and doesn’t split on punctuation or numbers. The behavior can be vastly different for other tokenizers.```` ```py``  ````` ```py`## Tokenization    After the optional pre-tokenization step, the actual tokenization step is performed. Some of the important algorithms in this space are byte pair encoding (BPE), byte-level BPE, WordPiece, and Unigram LM. The tokenizer comprises a set of rules that is learned during a pre-training phase over a pre-training dataset. Now let’s go through these algorithms in detail.    ## Byte Pair Encoding    This algorithm is the simplest and most widely used tokenization algorithm.    ### Training stage    We take a training dataset, run it through the normalization and pre-tokenization steps discussed earlier, and record the unique tokens in the resulting output and their frequencies. We then construct an initial vocabulary consisting of the unique characters that make up these tokens. Starting from this initial vocabulary, we continue adding new tokens using *merge* rules. The merge rule is simple; we create a new token using the most frequent consecutive pairs of tokens. The merges continue until we reach the desired vocabulary size.    Let’s explore this with an example. Imagine our training dataset is composed of six words, each appearing just once:    ``` 'bat', 'cat', 'cap', 'sap', 'map', 'fan' ```py    The initial vocabulary is then made up of:    ``` 'b', 'a', 't', 'c', 'p', 's', 'm', 'f', 'n' ```py    The frequencies of contiguous token pairs are:    ``` 'ba' - 1, 'at' - 2, 'ca' - 2, 'ap' - 3, 'sa' - 1, 'ma' - 1, 'fa' - 1, 'an' - 1 ```py    The most frequent pair is “ap,” so the first merge rule is to merge “a” and “p.” The vocabulary now is:    ``` 'b', 'a', 't', 'c', 'p', 's', 'm', 'f', 'n', 'ap' ```py    The new frequencies are:    ``` 'ba' - 1, 'at' - 2, 'cap' - 1, 'sap' - 1, 'map' - 1, 'fa' - 1, 'an' - 1 ```py    Now, the most frequent pair is “at,” so the next merge rule is to merge “a” and “t.” This process continues until we reach the vocabulary size.    ### Inference stage    After the tokenizer has been trained, it can be used to divide the text into appropriate subword tokens and feed the text into the model. This happens in a similar fashion as the training step. After normalization and pre-tokenization of the input text, the resulting tokens are broken into individual characters, and all the merge rules are applied in order. The tokens standing after all merge rules have been applied are the final tokens, which are then fed to the model.    You can open the [vocabulary file](https://oreil.ly/7JAyY) for GPT-NeoX again, and Ctrl+F “merges” to see the merge rules. As expected, the initial merge rules join single characters with each other. At the end of the merge list, you can see larger subwords like “out” and “comes” being merged into a single token.    ###### Note    Since all unique individual characters in the tokenizer training set will get their own token, it is guaranteed that there will be no OOV tokens as long as all tokens seen during inference in the future are made up of characters that were present in the training set. But Unicode consists of over a million code points and around 150,000 valid characters, which would not fit in a vocabulary of size 30,000\. This means that if your input text contained a character that wasn’t in the training set, that character would be assigned an <UNK> token. To resolve this, a variant of BPE called byte-level BPE is used. Byte-level BPE starts with 256 tokens, representing all the characters that can be represented by a byte. This ensures that every Unicode character can be encoded just by the concatenation of the constituent byte tokens. Hence, it also ensures that we will never encounter an <UNK> token. The GPT family of models use this tokenizer.    ## WordPiece    WordPiece is similar to BPE, so we will highlight only the differences.    Instead of the frequency approach used by BPE, WordPiece uses the maximum likelihood approach. The frequency of the token pairs in the dataset is normalized by the product of the frequency of the individual tokens. The pairs with the resulting highest score are then merged:    ``` 分数 = freq(a,b)/(freq(a) * freq(b)) ```py    This means that if a token pair is made up of tokens that individually have low frequency, they will be merged first.    Figure 3-4 shows the merge priority and how the normalization by individual frequencies affects the order of merging.  ![WordPiece tokenization](img/dllm_0304.png)  ###### Figure 3-4\. WordPiece tokenization    During inference, merge rules are not used. Instead, for each pre-tokenized token in the input text, the tokenizer finds the longest subword from the vocabulary in the token and splits on it. For example, if the token is “understanding” and the longest subword in the dictionary within this token is “understand,” then it will be split into “understand” and “ing.”    ### Postprocessing    Now that we have looked at a couple of tokenizer algorithms, let’s move on to the next stage of the pipeline, the postprocessing stage. This is where model-specific special tokens are added. Common tokens include [CLS], the classification token used in many language models, and [SEP], a separator token used to separate parts of the input.    ## Special Tokens    Depending on the model, a few special tokens are added to the vocabulary to facilitate processing. These tokens can include:    <PAD>      To indicate padding, in case the size of the input is less than the maximum sequence length.      <EOS>      To indicate the end of the sequence. Generative models stop generating after outputting this token.      <UNK>      To indicate an OOV term.      <TOOL_CALL>, </TOOL_CALL>      Content between these tokens is used as input to an external tool, like an API call or a query to a database.      <TOOL_RESULT>, </TOOL_RESULT>      Content between these tokens is used to represent the results from calling the aforementioned tools.      As we have seen, if our data is domain-specific like healthcare, scientific literature, etc., tokenization from a general-purpose tokenizer will be unsatisfactory. GALACTICA by Meta introduced several domain-specific tokens in their model and special tokenization rules:    *   [START_REF] and [END_REF] for wrapping citations.           *   <WORK> to wrap tokens that make up an internal working memory, used for reasoning and code generation.           *   Numbers are handled by assigning each digit in the number its own token.           *   [START_SMILES], [START_DNA], [START_AMINO], [END_SMILES], [END_DNA], [END_AMINO] for protein sequences, DNA sequences, and amino acid sequences, respectively.              If you are using a model on domain-specific data like healthcare, finance, law, biomedical, etc., with a tokenizer that was trained on general-purpose data, the compression ratio will be relatively lower because domain-specific words do not have their own tokens and will be split into multiple tokens. One way to adapt models to specialized domains is for models to learn good vector representations for domain-specific terms.    To this end, we can add new tokens to existing tokenizers and continue pre-training the model on domain-specific data so that those new domain-specific tokens learn effective representations. We will learn more about continued pre-training in Chapter 7.    For now, let’s see how we can add new tokens to a vocabulary using Hugging Face.    Consider the sentence, “The addition of CAR-T cells and antisense oligonucleotides drove down incidence rates.” The FLAN-T5 tokenizer splits this text as follows:    *   ['▁The', '▁addition', '▁of', '▁C', ' AR', '-', ' T', '▁cells', '▁and', '▁anti', ' s', ' ense', '▁', ' oli', ' gon', ' u', ' cle', ' o', ' t', ' ides', '▁drove', '▁down', '▁incidence', '▁rates', ' .', '</s>']    Let’s add the domain-specific terms to the vocabulary:    ``` from transformers import T5Tokenizer, T5ForConditionalGeneration   tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large") model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large",     device_map="auto")  tokenizer.add_tokens(["CAR-T", "antisense", "oligonucleotides"]) model.resize_token_embeddings(len(tokenizer)) ```py    Now, tokenizing the string again gives the following tokens, with the domain-specific tokens being added:    *   ['▁The', '▁addition', '▁of', ' CAR-T', '▁cells', '▁and', ' antisense', ' oligonucleotides', '▁drove', '▁down', '▁incidence', '▁rates', ' .', '</s>']    We are only halfway done here. The embedding vectors corresponding to these new tokens do not contain any information about these tokens. We will need to learn the right representations for these tokens, which we can do using fine-tuning or continued pre-training, which we will discuss in Chapter 7.```` ```py``  `` `# Summary    In this chapter, we focused on a key ingredient of language models: their vocabulary. We discussed how vocabularies are defined and constructed in the realm of language models. We introduced the concept of tokenization and presented tokenization algorithms like BPE and WordPiece that are used to construct vocabularies and break down raw input text into a sequence of tokens that can be consumed by the language model. We also explored the vocabularies of popular language models and noted how tokens can differ from human conceptions of a word.    In the next chapter, we will continue exploring the remaining ingredients of a language model, including its architecture and the learning objectives on which models are trained.` `` ``````py ``````py` ``````
