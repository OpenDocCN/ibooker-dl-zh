# Chapter 6\. Making Sentiment Programmable by Using Embeddings

In [Chapter 5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759), you saw how to take words and encode them into tokens. Then, you saw how to encode sentences full of words into sequences full of tokens, padding or truncating them as appropriate to end up with a well-shaped set of data that you can use to train a neural network. However, in none of that was there any type of modeling of the *meaning* of a word. And while it’s true that there’s no absolute numeric encoding that could encapsulate meaning, there are relative ones.

In this chapter, you’ll learn about techniques to encapsulate meaning, and in particular, the concept of *embeddings*, in which vectors in high-dimensional space are created to represent words. The directions of these vectors can be learned over time, based on the use of the words in the corpus. Then, when you’re given a sentence, you can investigate the directions of the word vectors, sum them up, and from the overall direction of the summation, establish the sentiment of the sentence as a product of its words. Also, related to this, as the model scans the sentences, the positioning of the words in the sentence can also help train an appropriate embedding.

In this chapter, we’ll also explore how that works. Using the News Headlines Dataset for Sarcasm Detection dataset from [Chapter 5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759), you’ll build embeddings to help a model detect sarcasm in a sentence. You’ll also work with some cool visualization tools that help you understand how words in a corpus get mapped to vectors so you can see which words determine the overall classification.

# Establishing Meaning from Words

Before we get into the higher-dimensional vectors for embeddings, let’s use some simple examples to try to visualize how meaning can be derived from numerics. Consider this: using the sarcasm dataset from [Chapter 5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759), what would happen if you encoded all of the words that make up sarcastic headlines with positive numbers and those that make up realistic headlines with negative numbers?

## A Simple Example: Positives and Negatives

Take, for example, this sarcastic headline from the dataset:

```py
christian bale given neutered male statuette named oscar
```

Assuming that all words in our vocabulary start with a value of 0, we could add 1 to the value for each of the words in this sentence, and we would end up with this:

```py
{ "christian" : 1, "bale" : 1, "given" : 1, "neutered": 1, "male" : 1, 
  "statuette": 1, "named" : 1, "oscar": 1}
```

###### Note

This isn’t the same as the *tokenization* of words that you did in the last chapter. You could consider replacing each word (e.g., *christian*) with the token representing it that is encoded from the corpus, but I’ll leave the words in for now to make the code easier to read.

Then, in the next step, consider an ordinary headline (not a sarcastic one), like this:

```py
gareth bale scores wonder goal against germany
```

Because this is a different sentiment, we could instead subtract 1 from the current value of each word, so our value set would look like this:

```py
{ "christian" : 1, "bale" : 0, "given" : 1, "neutered": 1, "male" : 1,
  "statuette": 1, "named" : 1, "oscar": 1, "gareth" : –1, "scores": –1,
  "wonder" : –1, "goal" : –1, "against" : –1, "germany" : –1}
```

Note that the sarcastic `bale` (from `christian bale`) has been offset by the nonsarcastic `bale` (from `gareth bale`), so its score ends up as 0\. Repeat this process thousands of times and you’ll end up with a huge list of words from your corpus that are scored based on their usage.

Now, imagine we want to establish the sentiment of this sentence:

```py
neutered male named against germany, wins statuette!
```

Using our existing value set, we could look at the scores of each word and add them up. We would get a score of 2, indicating (because it’s a positive number) that this is a sarcastic sentence.

###### Note

For what it’s worth, the word *bale* is used five times in the Sarcasm dataset, twice in a normal headline and three times in a sarcastic one. So, in a model like this, the word *bale* would be scored –1 across the whole dataset.

## Going a Little Deeper: Vectors

Hopefully, the previous example has helped you understand the mental model of establishing some form of *relative* meaning for a word, through its association with other words in the same “direction.” In our case, while the computer doesn’t understand the meanings of individual words, it can move labeled words from a known sarcastic headline in one direction (by adding 1) and move labeled words from a known normal headline in another direction (by subtracting 1). This gives us a basic understanding of the meaning of the words, but it does lose some nuance.

But what if we increased the dimensionality of the direction to try to capture some more information? For example, suppose we were to look at characters from the Jane Austen novel *Pride and Prejudice*, considering the dimensions of gender and nobility. We could plot the former on the *x*-axis and the latter on the *y*-axis, with the length of the vector denoting each character’s wealth (see [Figure 6-1](#ch06_clean_figure_1_1748752380714921)).

![](assets/aiml_0601.png)

###### Figure 6-1\. Characters in Pride and Prejudice as vectors

From an inspection of the graph, you can derive a fair amount of information about each character. Three of them are male. Mr. Darcy is extremely wealthy, but his nobility isn’t clear (he’s called “Mister,” unlike the less wealthy but apparently more noble Sir William Lucas). The other “Mister,” Mr. Bennet, is clearly not nobility and is struggling financially. Elizabeth Bennet, his daughter, is similar to him but female. Lady Catherine, the other female character in our example, is noble and incredibly wealthy. The romance between Mr. Darcy and Elizabeth causes tension—with *prejudice* coming from the noble side of the vectors toward the less-noble.

As this example shows, by considering multiple dimensions, we can begin to see real meaning in the words (which are character names here). Again, we’re not talking about concrete definitions but more about a *relative* meaning based on the axes and the relationship between the vector for one word and the other vectors.

This leads us to the concept of an *embedding*, which is simply a vector representation of a word that is learned while training a neural network. We’ll explore that next.

# Embeddings in PyTorch

Much like you’ve seen with `Linear` and `Conv2D`, PyTorch implements embeddings by using a layer. This creates a lookup table that maps from an integer to an embedding table, the contents of which are the coefficients of the vector representing the word identified by that integer. So, in the *Pride and Prejudice* example from the previous section, the *x* and *y* coordinates would give us the embeddings for a particular character from the book. Of course, in a real NLP problem, we’ll use far more than two dimensions. Thus, the direction of a vector in the vector space could be seen as encoding the “meaning” of a word, and words with similar vectors (i.e., pointing in roughly the same direction) could be considered related to that word.

The embedding layer will be initialized randomly—that is, the coordinates of the vectors will be completely random to start with and will be learned during training by using backpropagation. When training is complete, the embeddings will roughly encode similarities between words, allowing us to identify words that are somewhat similar based on the direction of the vectors for those words.

This is all quite abstract, so I think the best way to understand how to use embeddings is to roll up your sleeves and give them a try. Let’s start with a sarcasm detector using the Sarcasm dataset from [Chapter 5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759).

## Building a Sarcasm Detector by Using Embeddings

In [Chapter 5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759), you loaded and did some preprocessing on a JSON dataset called the News Headlines Dataset for Sarcasm Detection (the sarcasm dataset, for short). By the time you were done, you had lists of training and testing data and labels:

```py
training_size = 28000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
```

For the training data, you created a `build_vocab` helper function to create a dictionary of the frequency of each word, sorted in order of the most frequent. The size of this dictionary is the `vocab_size`.

To get an embedding layer in PyTorch, you can use the `nn.Embedding` layer type, like this, by specifying the desired vocab size and the number of embedding dimensions:

```py
nn.Embedding(vocab_size, embedding_dim)
```

This will initialize a vector with `embedding_dim` axes for each word. So, for example, if `embedding_dim` is `16`, then every word in the vocabulary will be assigned a 16-dimensional vector.

Over time, the attributes for each token (encoded as values for the vector in each of its dimensions) will be learned through backpropagation as the network learns by matching the training data to its labels.

An important next step is feeding the output of the embedding layer into a dense layer. The easiest way to do this, similar to how you would when using a convolutional neural network, is to use pooling. In this instance, the dimensions of the embeddings are averaged out to produce a fixed-length output vector, and `Adaptive​A⁠ve​Pool1d(1)` reduces the input along the length of the sequence to a fixed vector size of 1.

As an example, consider this model architecture:

```py
self.embedding = nn.Embedding(vocab_size, embedding_dim)
self.global_pool = nn.AdaptiveAvgPool1d(1)
self.fc1 = nn.Linear(embedding_dim, 24)
self.fc2 = nn.Linear(24, 1)
self.relu = nn.ReLU()
self.sigmoid = nn.Sigmoid()
```

Here, an embedding layer is defined, and it’s given the vocab size and an embedding dimension. Let’s take a look at the number of trainable parameters in the network, using `torchinfo.summary`:

```py
==========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================
TextClassificationModel                  [32, 1]                   --
├─Embedding: 1-1                         [32, 100, 100]            2,429,200
├─AdaptiveAvgPool1d: 1-2                 [32, 100, 1]              --
├─Linear: 1-3                            [32, 24]                  2,424
├─ReLU: 1-4                              [32, 24]                  --
├─Linear: 1-5                            [32, 1]                   25
├─Sigmoid: 1-6                           [32, 1]                   --
==========================================================================
Total params: 2,431,649
Trainable params: 2,431,649
Non-trainable params: 0
Total mult-adds (M): 77.81
==========================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 2.57
Params size (MB): 9.73
Estimated Total Size (MB): 12.32
==========================================================================
```

The vocabulary size is 24,292 words, and as the embedding has 100 dimensions, the total number of trainable parameters in the embedding layer will be 2,429,200\. The first linear layer has 100 values in with 24 values out, so that’s a total of 2,400 weights, but each of the neurons also has a bias, so add 24 to get to 24, 24\.

Similarly, the last linear has 24 values in, with just a single neuron out. For a total of 24 parameters, plus one for the bias, this equals 25\. The entire network has 2,431,649 parameters to learn. Note that the average pooling layer has 0 trainable parameters, as it’s just averaging the parameters in the embedding layer before it to get a single 16-value vector.

If we train this model, we’ll get a pretty decent training accuracy of 99%+ after 30 epochs—but our validation accuracy will be below 80% (see [Figure 6-2](#ch06_clean_figure_2_1748752380714952)).

![](assets/aiml_0602.png)

###### Figure 6-2\. Training accuracy versus validation accuracy

That might seem to be a reasonable curve, given that the validation data likely contains many words that aren’t present in the training data. However, if you examine the loss curves for training versus validation over one hundred epochs, you’ll see a problem. Although you would expect to see that the training accuracy is higher than the validation accuracy, a clear indicator of overfitting is that while the validation accuracy is dropping a little over time (as shown in [Figure 6-2](#ch06_clean_figure_2_1748752380714952)), its loss is increasing sharply, as shown in [Figure 6-3](#ch06_clean_figure_3_1748752380714977).

![](assets/aiml_0603.png)

###### Figure 6-3\. Training loss versus validation loss

Overfitting like this is common with NLP models, due to the somewhat unpredictable nature of language. In the next sections, we’ll look at how to reduce this effect by using a number of techniques.

## Reducing Overfitting in Language Models

Overfitting happens when the network becomes overspecialized to the training data, and one part of this involves the network becoming very good at matching patterns in “noisy” data in the training set that doesn’t exist anywhere else. Because this particular noise isn’t present in the validation set, the better the network gets at matching it, the worse the loss of the validation set will be. This can result in the escalating loss that you saw in [Figure 6-3](#ch06_clean_figure_3_1748752380714977).

In this section, we’ll explore several ways to generalize the model and reduce overfitting.

### Adjusting the learning rate

A hyperparameter of the optimizer is the learning rate (LR). The details of this parameter are beyond the scope of this chapter, but consider it to be a value that if too high will cause the network to potentially learn too quickly and miss nuance. The flipside is also true—if you set it too low, your network may not learn effectively.

Perhaps the biggest factor that can lead to overfitting is whether the LR of your optimizer is too high. If it is, then your network learns *too quickly*. For this example, the code to define the optimizer was as follows:

```py
optimizer = optim.Adam(model.parameters(), lr=0.001, 
                       betas=(0.9, 0.999), amsgrad=False)
```

These are the defaults for the `Adam` optimizer. One thing to experiment with is the `learning rate` parameter (`lr`), and in the following code, you’ll see the results of an instance when I reduced by an order of 10 to 0.0001, like this:

```py
optimizer = optim.Adam(model.parameters(), `lr``=``0.0001`, 
                       betas=(0.9, 0.999), amsgrad=False)
```

The `betas` values stay at their defaults, as does `amsgrad`. Also note that both `beta` values must be between 0 and 1, and typically, both are close to 1\. Amsgrad is an alternative implementation of the Adam optimizer that was introduced in the paper [“On the Convergence of Adam and Beyond” by Sashank Reddi, Satyen Kale, and Sanjiv Kumar](https://oreil.ly/FhTDi).

This much lower LR has a profound impact on the network. [Figure 6-4](#ch06_clean_figure_4_1748752380714997) shows the accuracy of the network over one hundred epochs. The lower LR can be seen in the first 10 epochs or so, where it appears that the network isn’t learning, before it “breaks out” and starts to learn quickly.

Exploring the loss (as illustrated in [Figure 6-5](#ch06_clean_figure_5_1748752380715014)), we can see that even while the accuracy wasn’t going up for the first few epochs, the loss was going down. You could therefore be confident that the network would eventually start to learn, if you were watching it epoch by epoch.

![](assets/aiml_0604.png)

###### Figure 6-4\. Accuracy with a lower LR

![](assets/aiml_0605.png)

###### Figure 6-5\. Loss with a lower LR

And while the loss does start to show the same curve of overfitting that you saw in [Figure 6-3](#ch06_clean_figure_3_1748752380714977), note that it happens much later and at a much lower rate. By epoch 30, the loss is at about 0.49, whereas with the higher LR in [Figure 6-3](#ch06_clean_figure_3_1748752380714977), it was more than double that amount. And while it takes the network longer to get to a good accuracy rate, it does so with less loss, so you can be more confident in the results. With these hyperparameters, the loss on the validation set started to increase at about epoch 60, at which point, the training set had 90% accuracy and the validation set had about 81% accuracy, showing that we have quite an effective network.

Of course, it’s easy to just tweak the optimizer and then declare victory, but there are a number of other methods you can use to improve your model. You’ll learn about those in the next few sections, and for them, I’ve reverted back to using the default Adam optimizer so the effects of tweaking the LR won’t hide the benefits offered by these other techniques.

### Exploring vocabulary size

The sarcasm dataset deals with words, so if you explore the words in the dataset and in particular their frequency, you might get a clue that helps fix the overfitting issue.

I’ve provided a `word_frequency` helper function that lets you explore the frequency of words in the vocabulary. It looks like this:

```py
def word_frequency(sentences, word_dict):
    frequency = {word: 0 for word in word_dict}

    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            if word in frequency:
                frequency[word] += 1

    return frequency
```

You can run it with code like this:

```py
word_freq = word_frequency(training_sentences, word_index)
print(word_freq)
```

You’ll then see results like this: a dictionary containing the frequency of each word, starting with the most frequently used one, and moving on from there. Here are the first few words:

```py
{'new': 1318, 'trump': 1117, 'man': 1075, 'not': 634, 'just': 501, 
 'will': 484, 'one': 469, 'year': 440, …
```

If you want to plot this, you can iterate through each item in the list and make the *x* value the ordinal of where you are (1 for the first item, 2 for the second item, etc.). The *y* value will then be a `newlist[item]`, which you can plot with `matplotlib`. Here’s the code:

```py
import matplotlib.pyplot as plt
from collections import OrderedDict
newlist = (OrderedDict(sorted(word_freq.items(), key=lambda t: t[1], 
                       reverse=True)))

xs=[]
ys=[]
curr_x = 1
for item in newlist:
  xs.append(curr_x)
  curr_x=curr_x+1
  ys.append(newlist[item])

print(ys)
plt.plot(xs,ys)

```

The result is shown in [Figure 6-6](#ch06_clean_figure_6_1748752380715030).

![](assets/aiml_0606.png)

###### Figure 6-6\. Exploring the frequency of words

This “hockey stick” curve shows us that very few words are used many times, whereas most words are used very few times. But every word is effectively weighted equally because every word has an “entry” in the embedding. Given that we have a relatively large training set in comparison with the validation set, we’re ending up in a situation where there are many words present in the training set that aren’t present in the validation set.

You can zoom in on the data by changing the axis of the plot just before calling `plt.show`. For example, to look at the volume of words from 300 to 10,000 on the *x*-axis with the scale from 0 to 100 on the *y*-axis, you can use this code:

```py
plt.plot(xs,ys)
plt.axis([300,10000,0,100])
plt.show()
```

The result is in [Figure 6-7](#ch06_clean_figure_7_1748752380715045).

![Frequency of words 300–10,000](assets/aiml_0607.png)

###### Figure 6-7\. Frequency of words from 300 to 10,000

There are almost 25,000 words in the corpus, and the code is set up to only train for all of them! But if we look at the words in positions 2,000 onward, which is over 90% of our vocabulary, we’ll see that they’re each used fewer than 20 times in the entire corpus!

This could explain the overfitting, so the logical next step is to see if we can reduce the vocabulary we are training for. Within the `build_vocab` helper function, we can add a parameter for the maximum vocab size we’re interested in, like this:

```py
def build_vocab(sentences, max_vocab_size=10000):
    counter = Counter()
    for text in sentences:
        counter.update(tokenize(text))

# Take only the top max_vocab_size-1 most frequent words 
# (leave room for special tokens)
    most_common = counter.most_common(max_vocab_size – 2)  
    # -2 for <pad> and <unk>

    # Create vocabulary with indices starting from 2
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab['<pad>'] = 0  # Add padding token
    vocab['<unk>'] = 1  # Add unknown token
    return vocab
```

Then, when building our `word_index`, we can specify a maximum vocab size that we’re interested in exploring:

```py
vocab_size = 2000
word_index = build_vocab(training_sentences, max_vocab_size=vocab_size)
```

The embedding layer was already initialized with the vocab size, so the model architecture doesn’t need to change. Indeed, with the reduced vocab size, the number of learned parameters drops sharply, giving us a simpler network that learns faster:

```py

==========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================
TextClassificationModel                  [32, 1]                   --
├─Embedding: 1-1                         [32, 100, 100]            200,100
├─AdaptiveAvgPool1d: 1-2                 [32, 100, 1]              --
├─Linear: 1-3                            [32, 24]                  2,424
├─ReLU: 1-4                              [32, 24]                  --
├─Linear: 1-5                            [32, 1]                   25
├─Sigmoid: 1-6                           [32, 1]                   --
==========================================================================
Total params: 202,549
Trainable params: 202,549
Non-trainable params: 0
Total mult-adds (M): 6.48
==========================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 2.57
Params size (MB): 0.81
Estimated Total Size (MB): 3.40
==========================================================================

```

The model has shrunk from 2.4 million parameters to only 202,549\.

After retraining and exploring the smaller model, we can see that the results have changed.

[Figure 6-8](#ch06_clean_figure_8_1748752380715056) shows the accuracy metrics. Now, the training set accuracy is about 82% and the validation accuracy is about 76%. They’re closer to each other and not diverging, which is a good sign that we’ve gotten rid of most of the overfitting.

![](assets/aiml_0608.png)

###### Figure 6-8\. Accuracy with a two thousand–word vocabulary

This is somewhat reinforced by the loss plot in [Figure 6-9](#ch06_clean_figure_9_1748752380715067). The loss on the validation set is rising but much slower than before, so reducing the size of the vocabulary to prevent the training set from overfitting on low-frequency words that were possibly only present in the training set appears to have worked.

It’s worth experimenting with different vocab sizes, but remember that you can also have too small of a vocab size and overfit to that. You’ll need to find a balance. In this case, my choice of taking words that appear 20 times or more was purely arbitrary.

![](assets/aiml_0609.png)

###### Figure 6-9\. Loss with a two thousand–word vocabulary

### Exploring embedding dimensions

For this example, I arbitrarily chose an embedding dimension of 16\. In this instance, words are encoded as vectors in 16-dimensional space, with their directions indicating their overall meaning. But is 16 a good number? With only two thousand words in our vocabulary, it might be on the high side, leading to a high degree of sparseness of direction.

###### Note

I believe that the best way to think about sparseness is to project into three dimensions. Think of it like the earth, with one thousand vectors pointing from the core to a place on the surface. The vectors are in three dimensions, *x*, *y*, and *z*. There’s a lot of surface area for them to cover, but if many of them are missing *x* and *y*, meaning they’re just zero, a lot of them will be pointing to (0, 0, *z*) and a whole lot of the earth’s surface will be untouched! Thus, there will be a total lack of distinctiveness.

Research has shown that a best practice for embedding size is to have it be the fourth root of the vocabulary size. The fourth root of 2,000 is 6.687, so let’s explore what happens if we round this up and change the embedding dimension to 7.

You can see the result of training for one hundred epochs in [Figure 6-10](#ch06_clean_figure_10_1748752380715076). The training set’s accuracy stabilized at about 83%, and the validation set’s accuracy stabilized at about 77%. Despite some jitters, the lines are pretty flat, showing that the model has converged. This isn’t much different from the results in [Figure 6-8](#ch06_clean_figure_8_1748752380715056), but reducing the embedding dimensionality allows the model to train significantly faster.

![](assets/aiml_0610.png)

###### Figure 6-10\. Training versus validation accuracy for seven dimensions

[Figure 6-11](#ch06_clean_figure_11_1748752380715085) shows the loss in training and validation. While it initially appeared that the loss was climbing at about epoch 20, it soon flattened out. Again, a good sign!

![](assets/aiml_0611.png)

###### Figure 6-11\. Training versus validation loss for seven dimensions

Now that the dimensionality has been reduced, we can do a bit more tweaking of the model architecture.

### Exploring the model architecture

After the optimizations in the previous sections, the model architecture looks like this:

```py
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=24):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

```

One thing that comes to mind is the dimensionality—the `GlobalAveragePooling1D` layer now emits just 7 dimensions, but they’re being fed into a hidden layer of 24 neurons, which is overkill. Let’s explore what happens when this is reduced to 8 neurons and trained for 100 epochs.

You can see the training versus validation accuracy in [Figure 6-12](#ch06_clean_figure_12_1748752380715093). When compared to [Figure 6-7](#ch06_clean_figure_7_1748752380715045), where 24 neurons were used, the overall result is quite similar, but the model was somewhat faster to train.

![](assets/aiml_0612.png)

###### Figure 6-12\. Reduced dense-architecture accuracy results

The loss curves in [Figure 6-13](#ch06_clean_figure_13_1748752380715102) show similar results.

By following these exercises, we were able to reduce the model architecture significantly, reducing the number of parameters while improving the quality and mitigating overfitting. But there are a few more things we can do—starting with dropout.

![](assets/aiml_0613.png)

###### Figure 6-13\. Reduced dense architecture loss results

### Using dropout

A common technique for reducing overfitting is to add dropout to a dense neural network. We explored this for convolutional neural networks back in [Chapter 3](ch03.html#ch03_going_beyond_the_basics_detecting_features_in_ima_1748570891074912), so it’s tempting to go straight to it here to see its effects on overfitting. But in this case, I want to wait until the vocabulary size, embedding size, and architecture complexity have been addressed. Those changes can often have a much larger impact than using dropout, and we’ve already seen some nice results.

Now that our architecture has been simplified to have only eight neurons in the middle dense layer, the effect of dropout may be minimized—but let’s explore it anyway. Here’s the updated code for the model architecture to add a dropout of 0.25 (which equates to two of our eight neurons):

```py
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=8, 
                       `dropout_rate``=``0.25``)``:`
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)  # Add dropout layer
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # Change for pooling layer
        x = self.global_pool(x).squeeze(2)
        `x` `=` `self``.``dropout`(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

```

[Figure 6-14](#ch06_clean_figure_14_1748752380715110) shows the accuracy results when trained for one hundred epochs. This time, we see that the training accuracy and validation accuracy are converging, with the training accuracy now lower than before. Similarly, the loss curves in [Figure 6-15](#fig-6-15) show convergence, so while dropout is making our network a little *less* accurate, it appears to generalize it better.

But do exercise caution before declaring victory! A close examination of the curves shows that the losses have nicely converged but that they *are* higher than previously. The training loss is above 0.5 with dropout but was around 0.3 without. It is also trending downward, so it’s worth experimenting to see whether longer training will produce a better result.

![](assets/aiml_0614.png)

###### Figure 6-14\. Accuracy with added dropout

![](assets/aiml_0615.png)

###### Figure 6-15\. Loss with added dropout

You can also see that the model is heading back to its previous pattern of increasing validation loss over time. It’s not nearly as bad as before, but it’s heading in the wrong direction.

In this case, when there were very few neurons, introducing dropout probably wasn’t the right idea. It’s still good to have this tool in your arsenal, though, so be sure to keep it in mind for more sophisticated architectures than this one.

### Using regularization

*Regularization* is a technique that helps prevent overfitting by reducing the polarization of weights. If the weights on some of the neurons are too heavy, regularization effectively punishes them. Broadly speaking, there are two types of regularization:

L1 regularization

This is often called *least absolute shrinkage* and *selection operator* (*lasso*) regularization. It effectively helps us ignore the zero or close-to-zero weights when calculating a result in a layer.

L2 regularization

This is often called *ridge* regression because it pushes values apart by taking their squares. This tends to amplify the differences between nonzero values and zero or close-to-zero ones, creating a ridge effect.

The two approaches can also be combined into what is sometimes called *elastic* regularization.

For NLP problems like the one we’re considering, L2 is most commonly used. It can be added as the `weight_decay` attribute to the `optimizer.` Here’s an example:

```py
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), 
                       amsgrad=False, `weight_decay``=``0.01`)
```

This will apply the `weight_decay` of `0.01.` (Usually, you’ll have a value between 0.01 and 0.001 here). Alternatively, a neat trick you can do with PyTorch is to define different weight decays for different layers by specifying them within the `Adam` declaration call, like this:

```py
# Different weight decay for different layers
optimizer = torch.optim.Adam([
# L2 reg on fc1
        {'params': model.fc1.parameters(), 'weight_decay': 0.01},      
    # No L2 reg on other layers
{'params': [p for name, p in model.named_parameters() 
            if 'fc1' not in name]}  
], lr=0.0001)
```

The impact of adding regularization in a simple model like this isn’t particularly large, but it does smooth out our training loss and validation loss somewhat. It might be overkill for this scenario, but as with dropout, it’s a good idea to understand how to use regularization to prevent your model from getting overspecialized.

### Other optimization considerations

While the modifications we’ve made have given us a much-improved model with less overfitting, there are other hyperparameters that you can experiment with. For example, we chose to make the maximum sentence length one hundred words, but that was purely arbitrary and probably not optimal. It’s a good idea to explore the corpus and see what a better sentence length might be. Here’s a snippet of code that looks at the sentences and plots the lengths of each one, sorted from low to high:

```py
xs=[]
ys=[]
current_item=1
for item in sentences:
  xs.append(current_item)
  current_item=current_item+1
  ys.append(len(item))
newys = sorted(ys)

import matplotlib.pyplot as plt
plt.plot(xs,newys)
plt.show()
```

See [Figure 6-16](#ch06_clean_figure_15_1748752380715124) for the results of this.

![](assets/aiml_0616.png)

###### Figure 6-16\. Exploring sentence length

Less than 200 sentences in the total corpus of 26,000+ have a length of 100 words or greater, so by choosing this as the maximum length, we’re introducing a lot of padding that isn’t necessary and thus affecting the model’s performance. Reducing the maximum to 85 words would still keep 26,000 of the sentences (99%+) with greatly reduced padding.

## Putting It All Together

Taking all of the preceding optimizations into effect and retraining the model for three hundred epochs gives you the results in [Figure 6-17](#ch06_clean_figure_16_1748752380715140) for training and validation accuracies. Given that their curves are roughly matched, it shows that we’ve taken huge steps toward avoiding overfitting and that we have a network that’s learning effectively.

Similarly, the training and validation loss curves over three hundred epochs are showing remarkable similarity, as depicted in [Figure 6-18](#ch06_clean_figure_17_1748752380715155), which indicates that the optimizations are a step in the right direction to prevent overfitting for this model.

![](assets/aiml_0617.png)

###### Figure 6-17\. Optimized training and validation accuracy

![](assets/aiml_0618.png)

###### Figure 6-18\. Optimized training and validation loss

## Using the Model to Classify a Sentence

Now that you’ve created the model, trained it, and optimized it to remove a lot of the problems that caused the overfitting, the next step is to run the model and inspect its results. To do this, you’ll create an array of new sentences. Consider, for example:

```py
test_sentences = [
             "granny starting to fear spiders in the garden might be real", 
             "game of thrones season finale showing this sunday night", 
             "PyTorch book will be a best seller"]
```

You can then encode these by using the same tokenizer that you used when creating the vocabulary for training:

```py
print(texts_to_sequences(test_sentences, word_index))

```

It’s important to use this tokenizer because it has the tokens for the words that the network was trained on!

The output of the print statement will be the sequences for the preceding sentences:

```py
[
[1, 803, 753, 1, 1, 312, 97], 
[123, 1183, 160, 1, 1, 1543, 152], 
[1, 235, 7, 47, 1]
]
```

There are a lot of `1` tokens here (“<OOV>”), because words like *granny* and *spiders* don’t appear in the dictionary. The sequences are also shorter because the stopwords have been removed.

Next, before you can pass the sequences to the model, you’ll need to put them in the shape that the model expects—that is, the desired length. You can do this with `pad_sequences` in the same way you did when training the model:

```py
padded = pad_sequences(sequences, max_len)
```

This will output the sentences as sequences of length `85`, so the output for the first sequence will be as follows:

```py
[1, 803, 753, 1, 1, 312, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0]
```

It was a very short sentence, so it’s padded up to 85 characters with a lot of zeros!

Now that you’ve padded and tokenized the sentences to fit the model’s expectations for the input dimensions, it’s time to pass them to the model and get predictions back.

This involves multiple steps. First, convert the padded sequence into an input tensor:

```py
# Convert to tensor
input_ids = torch.tensor(padded, dtype=torch.long).to(device)
```

Next, put the model into evaluation mode to get the predictions, and then simply pass the `input_ids` to it to get the outputs:

```py
# Get predictions
model.eval()
with torch.no_grad():
    outputs = model(input_ids)
```

The results will be passed back as a list and printed, with high values indicating likely sarcasm. Here are the results for our sample sentences:

```py
tensor([[0.5516],
        [0.0765],
        [0.0987]], device='cuda:0')
```

The high score for the first sentence (“granny starting to fear spiders in the garden might be real”), despite it having a lot of stopwords and being padded with a lot of zeros, indicates that there is a level of sarcasm there. The other two sentences scored much lower, indicating a lower likelihood of sarcasm in them.

To get the probabilities, you can call the `squeeze()` method to retrieve the tensor values. And if you want to make a comparison to a threshold to get your prediction—for example, above 0.5 indicates sarcasm and below 0.5 indicates no sarcasm—then you can use code like this:

```py
probabilities = outputs.squeeze().cpu().numpy()
predictions = (probabilities >= threshold).astype(int)
```

Based on your network tuning, you could also establish what you think the appropriate threshold should be. Running this with a 0.5 threshold gives us the following:

```py
Text: granny starting to fear spiders in the garden might be real
Probability: 0.5516
Classification: `Sarcastic`
Confidence: 0.5516
--------------------------------------------------------------------------

Text: game of thrones season finale showing this sunday night
Probability: 0.0765
Classification: Not Sarcastic
Confidence: 0.9235
--------------------------------------------------------------------------

Text: PyTorch book will be a best seller
Probability: 0.0987
Classification: `Not` `Sarcastic`
Confidence: 0.9013
--------------------------------------------------------------------------
```

So, with these test sentences, we’re beginning to get a good indication that our network is performing as desired. You should test it with other data to see if you can break it, and if you break it consistently, then it’ll be time to try a different model architecture, use transfer learning from an existing working network, or explore using pretrained embeddings.

We’ll learn about this in the next section, but before that, I’d like to show you how you can visualize the custom embeddings that this network learned.

# Visualizing the Embeddings

To visualize embeddings, you can use an online tool called the [Embedding Projector](http://projector.tensorflow.org). It comes preloaded with many existing datasets, but in this section, you’ll see how to take the data from the model you’ve just trained and visualize it by using this tool.

But first, you’ll need a function to reverse the word index. It currently has the word as the token and the key as the value, but you need to invert it so you’ll have word values to plot on the projector. Here’s the code to do this:

```py
reverse_word_index = dict([(value, key)
for (key, value) in word_index.items()])

```

You’ll also need to extract the weights of the vectors in the embeddings:

```py
embedding_weights = model.embedding.weight.data.cpu().numpy()
print(embedding_weights.shape)

```

If you’ve followed the optimizations in this chapter, the output of this will be `(2000,7)` because we used a 2,000 word vocabulary and 7 dimensions for the embedding. If you want to explore a word and its vector details, you can do so with code like this:

```py
print(reverse_word_index[2])
print(embedding_weights[2])

```

This will produce the following output:

```py
new
[–0.27116913 –1.3026129   1.6390767   0.4922502  –0.6025921   1.4584142
  0.05054485]

```

So, the word *new* is represented by a vector with those seven coefficients on its axes.

The Embedding Projector uses two tab-separated values (TSV) files, one for the vector dimensions and one for metadata. This code will generate them for you:

```py
import io
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = embedding_weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

```

Alternatively, if you are using Google Colab, you can download the TSV files with the following code or from the Files pane:

```py
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

```

Once you have the files, you can press the Load button on the projector to visualize the embeddings (see [Figure 6-19](#ch06_clean_figure_18_1748752380715170)).

![](assets/aiml_0619.png)

###### Figure 6-19\. Using the Embeddings Projector

You can also use the vectors and meta TSV files where recommended in the resulting dialog and then click Sphereize Data on the projector. This will cause the words to be clustered in a sphere and will give you a clear visualization of the binary nature of this classifier. It’s only been trained on sarcastic and nonsarcastic sentences, so words tend to cluster toward one label or another (see [Figure 6-20](#ch06_clean_figure_19_1748752380715184)).

![](assets/aiml_0620.png)

###### Figure 6-20\. Visualizing the sarcasm embeddings

Screenshots don’t do all of this justice—you should try it for yourself! You can rotate the center sphere and explore the words on each “pole” to see the impact they have on the overall classification, and you can also select words and show related ones in the righthand pane. Have a play and experiment!

# Using Pretrained Embeddings

An alternative to training your own embeddings is to use ones that have been pretrained by others on your behalf. There are many sources where you can find these, including Kaggle and Hugging Face. You can even find pretrained embeddings posted alongside research results. One such set of pretrained embeddings is the [Stanford GloVe embeddings](https://oreil.ly/s1YWw), and we’ll explore those here.

Note, however, that when using embeddings that have been pretrained, you should also consider updating and changing your tokenizer to match any rules used with the pretrained embeddings.

For example, with the GloVE pretrained embeddings—which simply comprise a large text file of words with their pretrained embedding in a number of dimensions from 50 to 300—the rules used to tokenize words are a little different from those for the handmade tokenizer we’ve been using for raw data. So, for GloVe, you should consider rules such as all of the words being lowercase or numbers being normalized to 0.

Once you’ve done this (I’ve provided code for GloVe in the downloads, and I discuss it in a little more detail in the next chapter), then it’s simply a matter of loading the weights of the pretrained embeddings to your model definition like this:

```py
# Initialize embedding layer
self.embedding = nn.Embedding(vocab_size, embedding_dim)

# Load pretrained embeddings if provided
if pretrained_embeddings is not None:
    self.embedding.weight.data.copy_(pretrained_embeddings)
    if freeze_embeddings:
        self.embedding.weight.requires_grad = False
```

If you don’t want to learn from these embeddings and you want to just use them, then you should set `freeze_embeddings` to `True`. Otherwise, the network will fine-tune by using the pre-loaded embedding weights as a starting point.

This model will rapidly reach peak accuracy in training, and it will not overfit as much as we saw previously. The accuracy over three hundred epochs shows that training and validation are very much in step with each other (see [Figure 6-22](#ch06_clean_figure_21_1748752380715231)). The loss values are also in step, which shows that we are fitting very nicely over the first couple of hundred epochs. However, they also begin to diverge (see [Figure 6-22](#ch06_clean_figure_21_1748752380715231)).

On the other hand, it is worth noting that the overall accuracy (at about 70%) is quite low, considering that a coin flip would have a 50% chance of getting it right! So, while using pretrained embeddings can make for much faster training with less overfitting, you should also understand what it is that they’re useful for and that they may not always be best for your scenario. You may therefore need to explore optimization methods or alternatives where appropriate.

![](assets/aiml_0621.png)

###### Figure 6-21\. Accuracy metrics using GloVe embeddings

![](assets/aiml_0622.png)

###### Figure 6-22\. Loss metrics using GloVe embeddings

# Summary

In this chapter, you built your first model that can understand sentiment in text. It did this by taking the tokenized text from [Chapter 5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759) and mapping it to vectors. Then, using backpropagation, it learned the appropriate “direction” for each vector based on the label for the sentence containing it. Finally, it was able to use all of the vectors for a collection of words to build up an idea of the sentiment within the sentence.

You also explored ways to optimize your model to avoid overfitting, and you saw a neat visualization of the final vectors representing your words. But while this was a nice way to classify sentences, it simply treated each sentence as a bunch of words. There was no inherent sequence involved, and the order of appearance of words is very important in determining the real meaning of a sentence.

Therefore, it’s a good idea to see if we can improve our models by taking sequence into account. We’ll explore that in the next chapter with the introduction of a new layer type: a *recurrent* layer, which is the foundation of recurrent neural networks.