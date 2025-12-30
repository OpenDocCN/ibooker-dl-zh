# Chapter 5\. Introduction to Natural Language Processing

Natural language processing (NLP) is a technique in AI that deals with the understanding of language. It involves programming techniques to create models that can understand language, classify content, and even generate and create new compositions in language. It’s also the underlying foundation to large language models (LLMs) such as GPT, Gemini, and Claude. We’ll explore LLMs in later chapters, but first, we’ll look at more basic NLP over the next few chapters to equip you for what’s to come.

There are also lots of services that use NLP to create applications such as chatbots, but that’s not in the scope of this book—instead, we’ll be looking at the foundations of NLP and how to model language so that you can train neural networks to understand and classify text. In later chapters, you’ll also learn how to use the predictive elements of an ML model to write some poetry. This isn’t just for fun—it’s also a precursor to learning how to use the transformer-based models that underpin generative AI!

We’ll start this chapter by looking at how you can decompose language into numbers and how you can then use those numbers in neural networks.

# Encoding Language into Numbers

Ultimately, computers deal in numbers, so to handle language, you need to convert it into numerics in a process called *encoding.*

You can encode language into numbers in many ways. The most common is to encode by letters, as is done naturally when strings are stored in your program. In memory, however, you don’t store the letter *a* but an encoding of it—perhaps an ASCII or Unicode value or something else. For example, consider the word *listen*. You can encode it with ASCII into the numbers 76, 73, 83, 84, 69, and 78\. This is good in that you can now use numerics to represent the word. But then consider the word *silent*, which is an anagram of *listen*. The same numbers represent that word, albeit in a different order, which might make building a model to understand the text much more difficult.

A better alternative might be to use numbers to encode entire words instead of the letters within them. In that case, *silent* could be the number *x* and *listen* could be the number *y*, and they wouldn’t overlap with each other.

Using this technique, consider a sentence like “I love my dog.” You could encode that with the numbers [1, 2, 3, 4]. If you then wanted to encode “I love my cat,” you could do it with [1, 2, 3, 5]. By now, you’ve probably gotten to the point where you can tell that the sentences have a similar meaning because they’re similar numerically—in other words, [1, 2, 3, 4] looks a lot like [1, 2, 3, 5].

The numbers representing words are also called *tokens*, and as a result, this process is called *tokenization*. You’ll explore how to do that in code next.

## Getting Started with Tokenization

The PyTorch ecosystem contains many libraries for tokenization, which takes words and turns them into tokens. A common tokenizer you might see in code samples is `torchtext,` but this has been deprecated since 2023\. So, be careful when using it, especially because PyTorch versions advance but it doesn’t. So, some alternatives are to use a custom tokenizer, a pretrained one from elsewhere, or (surprisingly) those from the Keras ecosystem.

### Using a custom tokenizer

To give you a simple example, here’s some code I used to create a custom tokenizer to turn the words of a small corpus (two sentences) into tokens:

```py
import torch

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

# Tokenization function
def tokenize(text):
    return text.lower().split()

# Build the vocabulary
def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        tokens = tokenize(sentence)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1 
    return vocab

# Create the vocabulary index
vocab = build_vocab(sentences)

print("Vocabulary Index:", vocab)
```

###### Note

The word *corpus* is commonly used to denote a set of text items that you will use for training. It’s literally the *body* of text that you’ll use to train the model and create tokenizers for.

The output of this is as follows:

```py
Vocabulary Index: {'today': 1, 'is': 2, 'a': 3, 'sunny': 4, 'day': 5, 'rainy': 6}
```

As you can see, the tokenizer did a really simple job of creating a list with my vocabulary, and every time it hit a unique word, it added it to the list. So the first sentence, “Today is a sunny day,” yielded five tokens for the five words: “today,” “is,” “a,” “sunny,” and “day.” The second sentence had *most* of these words in common, with “rainy” being the exception, so that became the sixth token.

On the other hand, you can imagine that for a very large corpus, this process would be very slow.

### Using a pretrained tokenizer from Hugging Face

With that in mind, I’m going to use Hugging Face’s transformers library and pre-built tokenizers from within it. In this case, because the transformers library supports many language models and these language models need tokenizers to work with their corpus of text, the tokenizer, which is trained on many millions of words, is freely available for you to use. It has bigger coverage than one you might create, and it’s free and easy to use!

If you don’t have this library already, you can install it with this:

```py
!pip install transformers
```

Now, let’s see it in action with a simple example:

```py
from transformers import BertTokenizerFast

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, 
                           return_tensors='pt')

# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids) 
           for ids in encoded_inputs["input_ids"]]

# To get the word index similar to Keras' tokenizer
word_index = tokenizer.get_vocab()

print("Tokens:", tokens)
print("Token IDs:", encoded_inputs['input_ids'])
print("Word Index:", dict(list(word_index.items())[:10]))  
# show only the first 10 for brevity

```

The output from it looks like this:

```py
Tokens: [['[CLS]', 'today', 'is', 'a', 'sunny', 'day', '[SEP]'], 
         ['[CLS]', 'today', 'is', 'a', 'rainy', 'day', '[SEP]']]

Token IDs: tensor([
        [  101,  2651,  2003,  1037, 11559,  2154,   102],
        [  101,  2651,  2003,  1037, 16373,  2154,   102]])

Word Index: {'protestant': 8330, 'initial': 3988, '##pt': 13876, 
             'charters': 23010, '243': 22884, 'ref': 25416, '##dies': 18389, 
             '##uchi': 15217, 'sainte': 16947, 'annette': 22521}
```

Now, let’s break this down. We start by importing the `BertTokenizerFast` from the transformers library. This can be initialized with a number of pretrained tokenizers, and we choose the `'bert-base-uncased'` one. You might be wondering what on earth that is! Well, the idea here is that I wanted to take a pretrained tokenizer, and they are usually partnered with the model they were trained on. BERT (which stands for bidirectional encoder representations from transformers) is a model trained by Google on a large corpus, with a vocabulary of 30,000 words. You can find models like this in the Hugging Face model repository, and when you dig down into a model, you’ll often see the transformer’s code to get its tokenizer. For example, see [this page that I used](https://oreil.ly/Ok7L9)—and while I’m not using the model, I can still get its tokenizer instead of creating a custom one.

In this case, we create a `tokenizer` object and specify the number of words that it can tokenize. This will be the maximum number of tokens to generate from the corpus of words. We also have a very small corpus here, containing only six unique words, so we’ll be well under the maximum of one hundred specified.

Once I have the tokenizer, I can then just pass the text to it:

```py
# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, 
                           return_tensors='pt')

```

We’ll explore padding and truncation a little later in this chapter, but for now, you should note the `return_tensors='pt'` parameter. This is a nice convenience for us PyTorch developers because the return values will be `torch.Tensor` objects, which are easy for us to handle.

The BERT model uses a number of overlays on the original tokenization, such as `attention_masking`, which means it works with `IDs` for each word instead of raw tokens. This is beyond the scope of this chapter, but where it impacts you right now is if you don’t need all that. If you just want the tokens, you have to extract the tokens in the following way, noting that your sentences were encoded as `input_Ids` within the BERT tokenizer:

```py
# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids) 
           for ids in encoded_inputs["input_ids"]]
```

Once you’ve done that, you can easily print out the following `Tokens` collection:

```py
Tokens: [['[CLS]', 'today', 'is', 'a', 'sunny', 'day', '[SEP]'], 
         ['[CLS]', 'today', 'is', 'a', 'rainy', 'day', '[SEP]']]
```

Now, you may be wondering what `[CLS]` and `[SEP]` are—and how the BERT model has been trained to expect sentences to begin with `[CLS]` (for *classifier*) and end with or be separated by `[SEP]` (for *separator*). These two expressions are tokenized to values 101 and 102, respectively, so when you print out the token values for your sentences, you’ll see this:

```py
Token IDs: tensor([
        [  101,  2651,  2003,  1037, 11559,  2154,   102],
        [  101,  2651,  2003,  1037, 16373,  2154,   102]])
```

From this, you can derive that *today* is token 2651 in BERT, *is* is token 2003, etc.

So, it really depends on you how you want to approach this. For learning with small datasets, the custom tokenizer is probably OK. But once you start getting into larger datasets, you may want to opt for a pretrained tokenizer. In that case, you may have to deal with some overhead—so for the rest of this chapter, I’m going to use custom code to tokenize and preprocess the text without the overhead of something like the BERT tokenizer.

Either way, once you have the words in your sentences tokenized, the next step is to convert your sentences into lists of numbers, with the number being the value where the word is the key. This process is called *sequencing*.

## Turning Sentences into Sequences

Now that you’ve seen how to take words and tokenize them into numbers, the next step is to encode the sentences into sequences of numbers, which you can do as follows:

```py
def text_to_sequence(text, vocab):
    return [vocab.get(token, 0) for token in tokenize(text)]  
# 0 for unknown words

```

Then, you’ll be given the sequences representing the three sentences. Remember that the word index is this:

```py
Vocabulary Index: {'today': 1, 'is': 2, 'a': 3, 'sunny': 4, 'day': 5, 'rainy': 6}
```

And the output will look like this:

```py
[1, 2, 3, 4, 5]
[1, 2, 3, 6, 5]
```

You can then substitute the words for the numbers, and you’ll see that the sentences make sense.

Now, consider what happens if you are training a neural network on a set of data. The typical pattern is that you have a set of data that’s used for training but that you know won’t cover 100% of your needs, but you hope it covers as much as possible. In the case of NLP, you might have many thousands of words in your training data that are used in many different contexts, but you can’t have every possible word in every possible context. So when you show your neural network some new, previously unseen text that contains previously unseen words, what might happen? You guessed it—the network will get confused because it simply has no context for those words, and as a result, any prediction it gives will be negatively affected.

### Using out-of-vocabulary tokens

One tool you can use to handle these situations is an *out-of-vocabulary* (OOV) *token*, which can help your neural network understand the context of the data containing previously unseen text. For example, given the previous small example corpus, suppose you want to process sentences like these:

```py
test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]
```

Remember that you’re not adding this input to the corpus of existing text (which you can think of as your training data) but you’re considering how a pretrained network might view this text. Say you tokenize it with the words that you’ve already used and your existing tokenizer, like this:

```py
for test_sentence in test_data:
  test_seq = text_to_sequence(test_sentence, vocab)
  print(test_seq)
```

Then, your results will look like this:

```py
[1, 2, 3, 0, 5]
[0, 0, 0, 6, 0]
```

So, the new sentences, swapping back tokens for words, would be “today is a <UNK> day” and “<UNK> <UNK> <UNK> rainy <UNK>.”

Here I’m using the tag <UNK> (which stands for *unknown*) for token 0\. If you check out the `text_to_sequence` code I showed previously, it uses `0` for words that aren’t in its dictionary. You can, of course, use any value you like.

### Understanding padding

When training neural networks, you typically need all your data to be in the same shape. Recall from earlier chapters that when you were training with images, you reformatted the images to be the same width and height. With text, you face the same issue—once you’ve tokenized your words and converted your sentences into sequences, they can all be different lengths. But to get them to be the same size and shape, you can use *padding*.

All the sentences we have used so far are composed of five words, so you can see that our sequences are five tokens. But what would happen if you had some sentences that were longer. Say a few had 5 words, some had 8 words, and some had 10 words. To have a neural network handle them all, they would need to be of the same length! You could convert everything to 10 words by lengthening the sentences that are shorter, convert everything to 5 words by chopping off bits of the longer ones, or follow some other strategy!

To explore padding, let’s add another, much longer, sentence to the corpus:

```py
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]
```

When you sequence that, you’ll see that your lists of numbers have different lengths. Also note that if you haven’t retokenized to build the new vocabulary, the latter two sentences will be full of zeros:

```py
[1, 2, 3, 4, 5]
[1, 2, 3, 6, 5]
[2, 0, 4, 0]
[0, 0, 0, 0, 0, 0, 0, 1]
```

So, don’t forget to call:

```py
vocab = build_vocab(sentences)
```

And then you’ll have new tokens for the new words in your tokenizer, so the output will look like this:

```py
[1, 2, 3, 4, 5]
[1, 2, 3, 6, 5]
[2, 7, 4, 8]
[9, 10, 11, 12, 13, 14, 15, 1]
```

Remember that when you were training neural networks in earlier chapters, your input layers of the neural network required images to have consistent sizes and shapes. It’s the same with NLP, for the most part. (There’s an exception for something called *ragged tensors*, but that’s beyond the scope of this chapter.) So, we need a way to make our sentences the same length.

Here’s a simple padding function:

```py
def pad_sequences(sequences, maxlen):
    return [seq + [0] * (maxlen - len(seq)) if len(seq) < maxlen 
            else seq[:maxlen] for seq in sequences]
```

This function will reshape every array in the sequence to be the same length as the maximum-length one. So, say we take our sentences and pad them after sequencing them with code like this:

```py
for sentence in sentences:
  seq = text_to_sequence(sentence, vocab)
  padded_seq = pad_sequences([seq], maxlen=10)  # Example maxlen
  print(padded_seq)
```

Then, the output will look like this:

```py
[[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]]
[[1, 2, 3, 6, 5, 0, 0, 0, 0, 0]]
[[2, 7, 4, 8, 0, 0, 0, 0, 0, 0]]
[[9, 10, 11, 12, 13, 14, 15, 1, 0, 0]]
```

Now, each of the sequences has a length of 10 because of the `maxlen` parameter. It’s a pretty simple implementation that you would likely want to build on if you’re using this in a more serious way. For example, you might want to consider what would happen if you had a sequence that was longer than the maximum length. Right now, it would cut off everything *after* the maximum, but you might want it to exhibit different behavior!

Also note that if you’re using off-the-shelf tokenizers like the BERT one we showed you earlier, much of this functionality may already be available to you, so be sure to experiment.

# Removing Stopwords and Cleaning Text

In the next section you’ll look at some real-world datasets, and you’ll find that there’s often text that you *don’t* want in your dataset. You may also want to filter out so-called *stopwords*—like “the,” “and,” and “but”—that are too common and don’t add any meaning. You may also encounter a lot of HTML tags in your text, and it would be good to have a clean way to remove them. Other things you may want to filter out include rude words, punctuation, or names. Later, we’ll explore a dataset of tweets that often have somebody’s user ID in them, and we’ll want to filter those out.

While every task is different based on your corpus of text, there are three main things that you can do to clean up your text programmatically.

## Stripping Out HTML Tags

The first thing you can do is strip out HTML tags, and fortunately, there’s a library called BeautifulSoup that makes this straightforward. For example, if your sentences contain HTML tags such as `<br>`, then you can remove them by using this code:

```py
from bs4 import BeautifulSoup
soup = BeautifulSoup(sentence)
sentence = soup.get_text()
```

## Stripping Out Stopwords

The second thing to do is strip out stopwords, and a common way to do it is to have a stopwords list and preprocess your sentences by removing instances of stopwords. Here’s an abbreviated example:

```py
stopwords = ["a", "about", "above", ... "yours", "yourself", "yourselves"]
```

You can find a full stopwords list in some of the [online examples for this chapter](https://github.com/lmoroney/PyTorch-Book-FIles).

Then, as you’re iterating through your sentences, you can use code like this to remove the stopwords from your sentences:

```py
words = sentence.split()
filtered_sentence = ""
for word in words:
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + " "
sentences.append(filtered_sentence)
```

## Stripping Out Punctuation

The third thing you can do is strip out punctuation, and you’ll want to do it because punctuation can fool a stopword remover. The one we just showed you looks for words surrounded by spaces, so it won’t spot a stopword that’s immediately followed by a period or a comma.

Fixing this problem is easy with the translation functions provided by the Python string library. But do be careful with this approach, as there are scenarios where it might impact NLP analysis, particularly when detecting sentiment.

The library also comes with a constant called `string.punctuation` that contains a list of common punctuation marks, so to remove them from a word, you can do the following:

```py
import string
table = str.maketrans('', '', string.punctuation)
words = sentence.split()
filtered_sentence = ""
for word in words:
    word = word.translate(table)
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + " "
sentences.append(filtered_sentence)
```

Here, before filtering for stopwords, the constant removes punctuation from each word in the sentence. So, if splitting a sentence gives you the word *it*, the word will be converted to *it* and then stripped out as a stopword. Note, however, that when doing this, you may have to update your stopwords list. It’s also common for these lists to have abbreviated words and contractions like *you’ll* in them, and the translator will change *you’ll* to *youll*. So if you want to have those words filtered out, you’ll need to update your stopwords list to include them.

Following these three steps will give you a much cleaner set of text to use. But of course, every dataset will have its idiosyncrasies that you’ll need to work with.

# Working with Real Data Sources

Now that you’ve seen the basics of getting sentences, encoding them with a word index, and sequencing the results, you can take it to the next level by taking some well-known public datasets and using the tools Python provides to get them into a format where you can easily sequence them. We’ll start with a dataset in which a lot of the work has already been done for you: the IMDb dataset. After that, we’ll get a bit more hands-on by processing a JSON-based dataset and a couple of comma-separated values (CSV) datasets with emotion data in them.

## Getting Text Datasets

We explored some datasets in [Chapter 4](ch04.html#ch04_using_data_with_pytorch_1748548966496246), so if you get stuck on any of the concepts in this section, you can get a quick review there. However, at the time of writing, accessing *text*-based datasets is a little unusual. Given that the torchtext library has been deprecated, it’s not clear what will happen with its built-in datasets, so we’ll get hands-on in dealing with raw data in this section.

We’ll start by exploring the IMDb reviews, which is a dataset of 50,000 labeled movie reviews from the Internet Movie Database (IMDb), each of which is determined to be positive or negative in sentiment.

This code will download the raw dataset and unzip it into folders where training and test splits are already pre-made for us. These will then be stored in subdirectories, and there are further subdirectories called `pos` and `neg` in each that determine the labels of the text files they contain:

```py
import os
import urllib.request
import tarfile

def download_and_extract(url, destination):
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)
    file_path = os.path.join(destination, "aclImdb_v1.tar.gz")

    if not os.path.exists(file_path):
        print("Downloading the dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")

    if "aclImdb" not in os.listdir(destination):
        print("Extracting the dataset...")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=destination)
        print("Extraction complete.")

# URL for the dataset
dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
download_and_extract(dataset_url, "./data")
```

The file structure will look like the one in [Figure 5-1](#ch05_figure_1_1748549080734915).

![](assets/aiml_0501.png)

###### Figure 5-1\. Exploring the IMDb dataset structure

In this figure you can see the *test/pos* directory and the first couple of files in it. Note that these are text files, so to create a tokenizer and vocabulary, we’re going to have to read files instead of in-memory strings like in the earlier example.

Let’s take a look at the code for a custom tokenizer for this:

```py
from collections import Counter
import os

# Simple tokenizer
def tokenize(text):
    return text.lower().split()

# Build vocabulary
def build_vocab(path):
    counter = Counter()
    for folder in ["pos", "neg"]:
        folder_path = os.path.join(path, folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', 
                                   encoding='utf-8') as file:
                counter.update(tokenize(file.read()))
    return {word: i+1 for i, word in enumerate(counter)} # Starting index from 1

vocab = build_vocab("./data/aclImdb/train/")
```

It’s pretty straightforward code that just reads through each file and adds new words it discovers to the vocabulary, giving each word a new token value. Generally, you’ll only want to do this for the training data, with the understanding that there will be words in the test data that aren’t in the training data and that they would be tokenized with an OOV or unknown token.

The output should look like this (truncated):

```py
{'a': 1, 'year': 2, 'or': 3, 'so': 4, 'ago,': 5, 'i': 6, 'was': 7…
```

This is a naive tokenizer in that the first word it sees gets the first token, the second gets the second, etc. For performance reasons, it’s often better for the more frequent words in the corpus to get the earlier tokens and the less frequent ones to get the later tokens. We’ll explore that in a moment.

You can then do sequencing and padding as you did earlier:

```py
def text_to_sequence(text, vocab):
    return [vocab.get(token, 0) for token in tokenize(text)]  # 0 for unknown

def pad_sequences(sequences, maxlen):
    return [seq + [0] * (maxlen - len(seq)) 
           if len(seq) < maxlen else seq[:maxlen] for seq in sequences]

# Example use
text = "This is an example."
seq = text_to_sequence(text, vocab)
padded_seq = pad_sequences([seq], maxlen=256)  # Example maxlen
print(seq)
```

So, for example, our sentence `This is an example` will output as `[30, 56, 144, 16040]` because those are the tokens assigned to those words. The padded sequence would have a tensor of 256 values, with these tokens as the first 4 and the next 252 being zeros!

Now, let’s update the tokenizer to do the words in order of frequency. This update changes the tokenizer so that we load all of the files into memory and count the instance of each word to get a frequency table:

```py
# Build vocabulary
def build_vocab(path):
    counter = Counter()
    for folder in ["pos", "neg"]:
        folder_path = os.path.join(path, folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', 
                                   encoding='utf-8') as file:
                counter.update(tokenize(file.read()))

    # Sort words by frequency in descending order
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # Create vocabulary with indices starting from 1
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words)}
    vocab['<pad>'] = 0  # Add padding token with index 0
    return vocab
```

We can then output the vocabulary as this frequency table. The vocabulary is too large to show the entire index, but here are the top 20 words. Note that the tokenizer lists them in order of frequency in the dataset, so common words like *the*, *and*, and *a* are indexed:

```py
{'the': 1, 'a': 2, 'and': 3, 'of': 4, 'to': 5, 'is': 6, 'in': 7, 'i': 8, 
'this': 9, 'that': 10, 'it': 11, '/><br': 12, 'was': 13, 'as': 14,
'for': 15, 'with': 16, 'but': 17, 'on': 18, 'movie': 19, 'his': 20,
```

These are stopwords, as described in the previous section. Having these present can impact your training accuracy because they’re the most common words and they’re nondistinct (i.e., they’re likely present in both positive and negative reviews), so they add noise to our training.

Also note that *br* is included in this list because it’s commonly used in this corpus as the `<br>` HTML tag.

You can also update the code to use `BeautifulSoup` to remove the HTML tags, and you can remove stopwords from the given list as follows. To do this, you can update the tokenizer to remove the HTML tags by using `BeautifulSoup`:

```py
# Simple tokenizer
from bs4 import BeautifulSoup

# Note that the list of stopwords is defined in the source code. 
# It’s an array of words. You can define your own or just get the one from 
# the book’s github.

def tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()  # Extract text from HTML
    return [word.lower() for word in cleaned_text.split() if word.lower() 
            not in stopwords]
```

Now, when you print out your word index, you’ll see this:

```py
{'movie': 1, 'not': 2, 'film': 3, 'one': 4, 'like': 5, 'just': 6, "it's": 7, 
 'even': 8, 'good': 9, 'no': 10, 'really': 11, 'can': 12, 'see': 13, '-': 14, 
 'get': 15, 'will': 16, 'much': 17, 'story': 18, 'also': 19, 'first': 20
```

You can see that this is much cleaner than before. There’s always room to improve, however, and one thing I noted when looking at the full index was that some of the less common words toward the end were nonsensical. Often, reviewers would combine words, for example with a dash (as in *annoying-conclusion*) or a slash (as in *him/her*), and the stripping of punctuation would incorrectly turn these combined words into a single word. Or, as you can see in the preceding code, the dash (*-*) character was common enough to be tokenized. You can strip that out by adding it as a stopword.

Now that you have a tokenizer for the corpus, you can encode your sentences. For example, the simple sentences we were looking at earlier in the chapter will come out like this:

```py
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

[[1094, 6112, 246, 0, 0, 0, 0, 0]]
[[1094, 6730, 246, 0, 0, 0, 0, 0]]
[[6112, 25065, 0, 0, 0, 0, 0, 0]]

```

If you decode these, you’ll see that the stopwords are dropped and you get the sentences encoded as `today sunny day`, `today rainy day`, and `sunny today`.

If you want to do this in code, you can create a new `dict` with the reversed keys and values (i.e., for a key/value pair in the word index, you can make the value the key and the key the value) and do the lookup from that. Here’s the code:

```py
reverse_word_index = dict(
    [(value, key) for (key, value) in vocab.items()])

decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in seq])

print(decoded_review)

```

This will give the following result:

```py
today sunny day
```

A common way to store labeled text data is in the comma-separated value (CSV) format. We’ll discuss that next.

## Getting Text from CSV Files

NLP data is also commonly available in CSV file format. Over the next couple of chapters, you’ll use a CSV of data that I adapted from the open source [Sentiment Analysis in Text dataset](https://oreil.ly/7ZKEU). The creator of this dataset sourced it from Twitter (now called X). You will use two different datasets, one where the emotions have been reduced to “positive” or “negative” for binary classification and one where the full range of emotion labels is used. Both datasets use the same structure, so I’ll just show the binary version here.

While the name *CSV* seems to suggest a standard file format in which values are comma separated, there’s actually a wide diversity of formats that can be considered CSV, and there’s very little adherence to any particular standard. To solve this, the Python csv library makes handling CSV files straightforward. In this case, the data is stored with two values per line. The first value is a number (0 or 1) denoting whether the sentiment is negative or positive, and the second value is a string containing the text.

The following code snippet will read the CSV and do preprocessing that’s similar to what we saw in the previous section. For the full code, please check the repo for this book. The code adds spaces around the punctuation in compound words, uses `BeautifulSoup` to strip HTML content, and then removes all punctuation characters:

```py
import csv
sentences=[]
labels=[]
with open('/tmp/binary-emotion.csv', encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        labels.append(int(row[0]))
        sentence = row[1].lower()
        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")
        soup = BeautifulSoup(sentence)
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence = ""
        for word in words:
            word = word.translate(table)
            if word not in stopwords:
                filtered_sentence = filtered_sentence + word + " "
        sentences.append(filtered_sentence)
```

This will give you a list of 35,327 sentences.

Note that this code is specific to this data. It’s intended to help you understand the types of tasks you may have to take on in order to make stuff work, and it’s not intended to be an exhaustive list of things that you’ll have to do for every task—so your mileage may vary.

### Creating training and test subsets

Now that the text corpus has been read into a list of sentences, you’ll need to split it into training and test subsets for training a model. For example, if you want to use 28,000 sentences for training with the rest held back for testing, you can use code like this:

```py
training_size = 28000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
```

Now that you have a training set, you can edit the tokenizer and vocabulary builder to create the word index from this corpus. As the corpus is an in-memory array of strings (`training_sentences`), the process is a lot simpler:

```py
from collections import Counter

# Assuming the tokenize function is defined elsewhere
def tokenize(text):
    # Tokenization logic, removing HTML and stopwords as discussed earlier
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    tokens = cleaned_text.lower().split()
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens

def build_vocab(sentences):
    counter = Counter()
    for text in sentences:
        counter.update(tokenize(text))

    # Sort words by frequency in descending order
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # Create vocabulary with indices starting from 1
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words)}
    vocab['<pad>'] = 0  # Add padding token with index 0
    return vocab

vocab = build_vocab(training_sentences)
print(vocab)
```

You can use the same helper functions to turn the text into a sequence and then padding, like this:

```py
print(testing_sentences[1])
seq = text_to_sequence(testing_sentences[1], vocab)
print(seq)
```

The results will be as follows:

```py
made many new friends twitter around usa another bike across usa trip amazing 
 see people
[146, 259, 30, 110, 53, 198, 2161, 111, 752, 970, 2161, 407, 217, 26, 73]
```

Another common format for structured data, particularly in response to web calls, is JavaScript Object Notation (JSON). We’ll explore how to read JSON data next.

## Getting Text from JSON Files

JSON is an open standard file format that’s used often for data interchange, particularly with web applications. It’s human readable and designed to use name/value pairs, and as such, it’s particularly well suited for labeled text. A quick search of Kaggle datasets for JSON yields over 2,500 results. For example, popular datasets such as the Stanford Question Answering Dataset (SQuAD) are stored in JSON.

JSON has very simple syntax in which objects are contained within braces as name/value pairs, each of which is separated by a comma. For example, a JSON object representing my name would be as follows:

```py
{"firstName" : "Laurence",
 "lastName" : "Moroney"}
```

JSON also supports arrays, which are a lot like Python lists and are denoted by the square bracket syntax. Here’s an example:

```py
[
 {"firstName" : "Laurence",
 "lastName" : "Moroney"},
 {"firstName" : "Sharon",
 "lastName" : "Agathon"}
]
```

Objects can also contain arrays, so this is perfectly valid JSON:

```py
[
 {"firstName" : "Laurence",
 "lastName" : "Moroney",
 "emails": ["lmoroney@gmail.com", "lmoroney@galactica.net"]
 },
 {"firstName" : "Sharon",
 "lastName" : "Agathon",
 "emails": ["sharon@galactica.net", "boomer@cylon.org"]
 }
]
```

A smaller dataset that’s stored in JSON and a lot of fun to work with is the “News Headlines Dataset for Sarcasm Detection” by [Rishabh Misra](https://oreil.ly/wZ3oD), which is available on [Kaggle](https://oreil.ly/_AScB). This dataset collects news headlines from two sources: *The Onion* for funny or sarcastic ones and the *HuffPost* for normal headlines.

The file structure in the sarcasm dataset is very simple:

```py
{"is_sarcastic": 1 or 0, 
 "headline": String containing headline, 
 "article_link": String Containing link}
```

The dataset consists of about 26,000 items, one per line. To make it more readable in Python, I’ve created a version that encloses these items in an array so the dataset can be read as a single list, which is used in the source code for this chapter.

### Reading JSON files

Python’s json library makes reading JSON files simple. Given that JSON uses name/value pairs, you can index the content based on the name. So, for example, for the sarcasm dataset, you can create a file handle to the JSON file, open it with the `json` library, and have an iterable go through, read each field line by line, and get the data item by using the name of the field.

Here’s the code:

```py
import json
with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)
    for item in datastore:
        sentence = item['headline'].lower()
        label= item['is_sarcastic']
        link = item['article_link']
```

This makes it simple for you to create lists of sentences and labels, as you’ve done throughout this chapter, and then tokenize the sentences. You can also do preprocessing on the fly as you read a sentence, removing stopwords, HTML tags, punctuation, and more.

Here’s the complete code to create lists of sentences, labels, and URLs while having the sentences cleaned of unwanted words and characters:

```py
with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = [] 
labels = []
urls = []
for item in datastore:
    sentence = item['headline'].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    sentences.append(filtered_sentence)
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
```

As before, you can split these into training and test sets. If you want to use 23,000 of the 26,000 items in the dataset for training, you can do the following:

```py
training_size = 23000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
```

Now that you have them as `in_memory` string arrays, tokenizing them and sequencing them will work exactly the same way as tokenizing and sequencing the sarcasm dataset.

Hopefully, the similar-looking code will help you see the pattern that you can follow when preparing text for neural networks to classify or generate. In the next chapter, you’ll learn how to build a classifier for text using embeddings, and in [Chapter 7](ch07.html#ch07_recurrent_neural_networks_for_natural_language_pro_1748549654891648) you’ll take that a step further by exploring recurrent neural networks. Then, in [Chapter 8](ch08.html#ch08_using_ml_to_create_text_1748549671852453), you’ll learn how to further enhance the sequence data to create a neural network that can generate new text!

###### Tip

Regular expressions (aka Regex) are terrific tools for sorting, filtering, and cleaning text. They have a syntax that’s often hard to understand and difficult to learn, but I have found that generative AI tools like Gemini, Claude, and ChatGPT are really useful here.

# Summary

In earlier chapters, you used images to build a classifier. Images, by definition, have well-defined dimensions—you know their width, height, and format. Text, on the other hand, can be far more difficult to work with. It is often unstructured, can contain undesirable content such as formatting instructions, doesn’t always contain what you want, and often has to be filtered to remove nonsensical or irrelevant content.

In this chapter, you saw how to take text and convert it to numbers using word tokenization, and you then explored how to read and filter text in a variety of formats. With these skills in hand, you’re now ready to take the next step and learn how *meaning* can be inferred from words—which is the first step in understanding natural language.