# Chapter 7\. Recurrent Neural Networks for Natural Language Processing

In [Chapter 5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759), you saw how to tokenize and sequence text, turning sentences into tensors of numbers that could then be fed into a neural network. You then extended that in [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888) by looking at embeddings, which constitute a way to have words with similar meanings cluster together to enable the calculation of sentiment. This worked really well, as you saw by building a sarcasm classifier. But there’s a limitation to that: namely, sentences aren’t just collections of words—and often, the *order* in which the words appear will dictate their overall meaning. Also, adjectives can add to or change the meaning of the nouns they appear beside. For example, the word *blue* might be meaningless from a sentiment perspective, as might *sky*, but when you put them together to get *blue sky*, it indicates a clear sentiment that’s usually positive. Finally, some nouns may qualify others, such as in *rain cloud*, *writing desk*, and *coffee mug*.

To take sequences like this into account, you need to take an additional approach: you need to factor *recurrence* into the model architecture. In this chapter, you’ll look at different ways of doing this. We’ll explore how sequence information can be learned and how you can use this information to create a type of model that is better able to understand text: the *recurrent neural network* (RNN).

# The Basis of Recurrence

To understand how recurrence might work, let’s first consider the limitations of the models used thus far in the book. Ultimately, creating a model looks a little bit like [Figure 7-1](#ch07_figure_1_1748549654877957). You provide data and labels and define a model architecture, and the model learns the rules that fit the data to the labels. Those rules then become available to you as an application programming interface (API) that will give you back predicted labels for future data.

![](assets/aiml_0701.png)

###### Figure 7-1\. High-level view of model creation

But, as you can see, the data is lumped in wholesale. There’s no granularity involved and no effort to understand the sequence in which that data occurs. This means the words *blue* and *sky* have no different meaning in sentences such as, “Today I am blue, because the sky is gray,” and “Today I am happy, and there’s a beautiful blue sky.” To us, the difference in the use of these words is obvious, but to a model, with the architecture shown here, there really is no difference.

So, how do we fix this? Let’s first explore the nature of recurrence, and from there, you’ll be able to see how a basic RNN can work.

Consider the famous Fibonacci sequence of numbers. In case you aren’t familiar with it, I’ve put some of them into [Figure 7-2](#ch07_figure_2_1748549654878009).

![](assets/aiml_0702.png)

###### Figure 7-2\. The first few numbers in the Fibonacci sequence

The idea behind this sequence is that every number is the sum of the two numbers preceding it. So if we start with 1 and 2, the next number is 1 + 2, which is 3\. The one after that is 2 + 3, which is 5, and then there’s 3 + 5, which is 8, and so on.

We can place this in a computational graph to get [Figure 7-3](#ch07_figure_3_1748549654878036).

![](assets/aiml_0703.png)

###### Figure 7-3\. A computational graph representation of the Fibonacci sequence

Here, you can see that we feed 1 and 2 into the function and get 3 as the output. We then carry the second parameter (2) over to the next step and feed it into the function along with the output from the previous step (3). The output of this is 5, and it gets fed into the function with the second parameter from the previous step (3) to produce an output of 8\. This process continues indefinitely, with every operation depending on those before it. The 1 at the top left sort of “survives” through the process—it’s an element of the 3 that gets fed into the second operation, it’s an element of the 5 that gets fed into the third operation, and so on. Thus, some of the essence of the 1 is preserved throughout the sequence, though its impact on the overall value is diminished.

This is analogous to how a recurrent neuron is architected. You can see the typical representation of a recurrent neuron in [Figure 7-4](#ch07_figure_4_1748549654878060).

![](assets/aiml_0704.png)

###### Figure 7-4\. A recurrent neuron

A value *x* is fed into the function *F* at a time step, so it’s typically labeled *x*[*t*]. This produces an output *y* at that time step, which is typically labeled *y*[*t*]. It also produces a value that is fed forward to the next step, which is indicated by the arrow from *F* to itself.

This is made a little clearer if you look at how recurrent neurons work beside one another across time steps, which you can see in [Figure 7-5](#ch07_figure_5_1748549654878082).

Here, *x*[0] is operated on to get *y*[0] and a value that’s passed forward. The next step gets that value and *x*[1] and produces *y*[1] and a value that’s passed forward. The next one gets that value and *x*[2] and produces *y*[2] and a passed-forward value, and so on down the line. This is similar to what we saw with the Fibonacci sequence, and I always find it to be a handy mnemonic when trying to remember how an RNN works.

![](assets/aiml_0705.png)

###### Figure 7-5\. Recurrent neurons in time steps

# Extending Recurrence for Language

In the previous section, you saw how an RNN operating over several time steps can help maintain context across a sequence. Indeed, we’ll use RNNs for sequence modeling later in this book—but there’s a nuance when it comes to language that you can miss when using a simple RNN like those shown in [Figure 7-4](#ch07_figure_4_1748549654878060) and [Figure 7-5](#ch07_figure_5_1748549654878082). As in the Fibonacci sequence example mentioned earlier, the amount of context that’s carried over will diminish over time. The effect of the output of the neuron at step 1 is huge at step 2, smaller at step 3, smaller still at step 4, and so on. So, if we have a sentence like “Today has a beautiful blue <something>,” the word *blue* will have a strong impact on what the next word could be: we can guess that it’s likely to be *sky*. But what about context that comes from earlier in a sentence? For example, consider the sentence “I lived in Ireland, so in high school, I had to learn how to speak and write <something>.”

That <something> is *Gaelic*, but the word that really gives us that context is *Ireland*, which is much earlier in the sentence. Thus, for us to be able to recognize what <something> should be, we need a way to preserve context across a longer distance. The short-term memory of an RNN needs to get longer, and in recognition of this, an enhancement to the architecture called *long short-term memory* (LSTM) was invented.

While I won’t go into detail on the underlying architecture of how LSTMs work, the high-level diagram shown in [Figure 7-6](#ch07_figure_6_1748549654878103) gets the main point across. To learn more about the internal operations of LSTM, check out Christopher Olah’s excellent [blog post on the subject](https://oreil.ly/6KcFA).

The LSTM architecture enhances the basic RNN by adding a “cell state” that enables context to be maintained not just from step to step but across the entire sequence of steps. Remembering that these are neurons that learn in the way neurons do, you can see that this enhancement ensures that the context that is important will be learned over time.

![](assets/aiml_0706.png)

###### Figure 7-6\. High-level view of LSTM architecture

An important part of an LSTM is that it can be *bidirectional*—the time steps can be iterated both forward and backward so that context can be learned in both directions. Often, context for a word can come *after* it in the sentence and not just before.

See [Figure 7-7](#ch07_figure_7_1748549654878124) for a high-level view of this.

![](assets/aiml_0707.png)

###### Figure 7-7\. High-level view of LSTM bidirectional architecture

This is how evaluation in the direction from 0 to `number_of_steps` is done, and it’s also how evaluation from `number_of_steps` to 0 is done. At each step, the *y* result is an aggregation of the “forward” pass and the “backward” pass. You can see this in [Figure 7-8](#ch07_figure_8_1748549654878143).

![](assets/aiml_0708.png)

###### Figure 7-8\. Bidirectional LSTM

It’s easy to confuse the bidirectional nature of the LSTM with the terms *forward* and *backward* when it comes to the training of the network, but they’re very different. When I refer to the forward and backward pass, I’m referring to the setting of the parameters of the neurons and their updating from the learning process, respectively. Don’t confuse this with the values that the LSTM is paying attention to as being the next or previous tokens in the sequence.

Also, consider each neuron at each time step to be F0, F1, F2, etc. The direction of the time step is shown, so the calculation at F1 in the forward direction is F1(->), and in the reverse direction, it’s (<-)F1\. The values of these are aggregated to give the *y* value for that time step. Additionally, the cell state is bidirectional, and this can be really useful for managing context in sentences. Again, considering the sentence “I lived in Ireland, so in high school, I had to learn how to speak and write <something>,” you can see how the <something> was qualified to be *Gaelic* by the context word *Ireland*. But what if it were the other way around: “I lived in <this country>, so in high school, I had to learn how to speak and write Gaelic”? You can see that by going *backward* through the sentence, we can learn about what <this country> should be. Thus, using bidirectional LSTMs can be very powerful for understanding sentiment in text. (And as you’ll see in [Chapter 8](ch08.html#ch08_using_ml_to_create_text_1748549671852453), they’re really powerful for generating text, too!)

Of course, there’s a lot going on with LSTMs, in particular bidirectional ones, so expect training to be slow. Here’s where it’s worth investing in a GPU or at the very least using a hosted one in Google Colab if you can.

# Creating a Text Classifier with RNNs

In [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888), you experimented with creating a classifier for the Sarcasm dataset by using embeddings. In that case, you turn words into vectors before aggregating them and then feeding them into dense layers for classification. But when you’re using an RNN layer such as an LSTM, you don’t do the aggregation, and you can feed the output of the embedding layer directly into the recurrent layer. When it comes to the dimensionality of the recurrent layer, a rule of thumb you’ll often see is that it’s the same size as the embedding dimension. This isn’t necessary, but it can be a good starting point. Also note that while in [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888) I mentioned that the embedding dimension is often the fourth root of the size of the vocabulary, when using RNNs, you’ll often see that that rule may be ignored because it would make the size of the recurrent layer too small.

For this example, I have used the number of neurons in the hidden layer as a starting point, and you can experiment from there.

So, for example, you could update the simple model architecture for the sarcasm classifier you developed in [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888) to the following to use a bidirectional LSTM:

```py
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, 
                 hidden_dim=24, lstm_layers=1):
        super(TextClassificationModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length)

        # Get embeddings
        embedded = self.embedding(x)  
        # Shape: (batch_size, sequence_length, embedding_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)  
        # Shape: (batch_size, sequence_length, hidden_dim)

        # Transpose for global pooling 
        # (expecting: batch, channels, sequence_length)
        lstm_out = lstm_out.transpose(1, 2)  
        # Shape: (batch_size, hidden_dim, sequence_length)

        # Apply global pooling
        pooled = self.global_pool(lstm_out)  
        # Shape: (batch_size, hidden_dim, 1)
        pooled = pooled.squeeze(–1)  # Shape: (batch_size, hidden_dim)

        # Pass through fully connected layers
        x = self.relu(self.fc1(pooled))
        x = self.sigmoid(self.fc2(x))

        return x
```

You can then set the loss function and classifier to this. (Note that the LR is 0.001, or 1e–3.):

```py
# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, 
                       betas=(0.9, 0.999), amsgrad=False)
```

When you print out the model architecture summary, you’ll see something like the following:

```py
==========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================
TextClassificationModel                  [32, 1]                   --
├─Embedding: 1-1                         [32, 85, 7]               14,000
├─LSTM: 1-2                              [32, 85, 48]              6,336
├─AdaptiveAvgPool1d: 1-3                 [32, 48, 1]               --
├─Linear: 1-4                            [32, 24]                  1,176
├─ReLU: 1-5                              [32, 24]                  --
├─Linear: 1-6                            [32, 1]                   25
├─Sigmoid: 1-7                           [32, 1]                   --
==========================================================================
Total params: 21,537
Trainable params: 21,537
Non-trainable params: 0
Total mult-adds (M): 17.72
==========================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 1.20
Params size (MB): 0.09
Estimated Total Size (MB): 1.31
==========================================================================
```

Note that the vocab size is 2,000 and the embedding dimension is 7\. This gives 14,000 parameters in the embedding layer, and the bidirectional layer will have 48 neurons (24 out, 24 back) with a sequence length of 85 characters

[Figure 7-9](#ch07_figure_9_1748549654878164) shows the results of training with this over three hundred epochs.

This gives us a network with only 21,537 parameters. As you can see, the accuracy of the network on training data rapidly climbs toward 85%, but the validation data plateaus at around 75%. This is similar to the figures we got earlier, but inspecting the loss chart in [Figure 7-10](#ch07_figure_10_1748549654878185) shows that while the loss for the test set diverged after 15 epochs, the validation loss turned to increase, indicating we have overfitting.

![](assets/aiml_0709.png)

###### Figure 7-9\. Accuracy for LSTM over 30 epochs

![](assets/aiml_0710.png)

###### Figure 7-10\. Loss with LSTM over 30 epochs

However, this was just using a single LSTM layer with a hidden layer of 24 neurons. In the next section, you’ll see how to use stacked LSTMs and explore the impact on the accuracy of classifying this dataset.

## Stacking LSTMs

In the previous section, you saw how to use an LSTM layer after the embedding layer to help classify the contents of the sarcasm dataset. But LSTMs can be stacked on top of one another, and this approach is used in many state-of-the-art NLP models.

Stacking LSTMs with PyTorch is pretty straightforward. You add them as extra layers just like you would with any other layer, but you will need to be careful in specifying the dimensions. So, for example, if the first LSTM has *x* number of hidden layers, then the next LSTM will have *x* number of inputs. If the LST is bidirectional, then the next will need to double the size. Here’s an example:

```py
# First LSTM layer
self.lstm1 = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    num_layers=lstm_layers,
    batch_first=True,
    bidirectional=True
)

# Second LSTM layer
# Note: Input size is hidden_dim*2 because first LSTM is bidirectional.
self.lstm2 = nn.LSTM(
    input_size=hidden_dim * 2,
    hidden_size=hidden_dim,
    num_layers=lstm_layers,
    batch_first=True,
    bidirectional=True
)

```

Note that the `input_size` for the first layer is the embedding dimension because it’s preceded by the embedding layer. The second LSTM then has its input size as (`hidden_dim * 2`) because the output from the first LSTM is that size, given that it’s bidirectional.

The model architecture will look like this:

```py
==========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================
TextClassificationModel                  [32, 1]                   --
├─Embedding: 1-1                         [32, 85, 7]               14,000
├─LSTM: 1-2                              [32, 85, 48]              6,336
├─LSTM: 1-3                              [32, 85, 48]              14,208
├─AdaptiveAvgPool1d: 1-4                 [32, 48, 1]               --
├─Linear: 1-5                            [32, 24]                  1,176
├─ReLU: 1-6                              [32, 24]                  --
├─Linear: 1-7                            [32, 1]                   25
├─Sigmoid: 1-8                           [32, 1]                   --
==========================================================================
Total params: 35,745
Trainable params: 35,745
Non-trainable params: 0
Total mult-adds (M): 56.37
==========================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 2.25
Params size (MB): 0.14
Estimated Total Size (MB): 2.41
==========================================================================

```

Adding the extra layer will give us roughly 14,000 extra parameters that need to be learned, which is an increase of about 75%. So, it might slow the network down, but the cost is relatively low if there’s a reasonable benefit.

After training for three hundred epochs, the result looks like [Figure 7-11](#ch07_figure_11_1748549654878205). While the accuracy on the validation set is flat, examining the loss (shown in [Figure 7-12](#ch07_figure_12_1748549654878225)) tells a different story. As you can see in [Figure 7-12](#ch07_figure_12_1748549654878225), while the accuracy for both training and validation looked good, the validation loss quickly took off upward, which is a clear sign of overfitting.

![](assets/aiml_0711.png)

###### Figure 7-11\. Accuracy for stacked LSTM architecture

This overfitting (which is indicated by the training accuracy climbing toward 100% as the loss falls smoothly while the validation accuracy is relatively steady and the loss increases drastically) is a result of the model getting overspecialized for the training set. As with the examples in [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888), this shows that it’s easy to be lulled into a false sense of security if you just look at the accuracy metrics without examining the loss.

![](assets/aiml_0712.png)

###### Figure 7-12\. Loss for stacked LSTM architecture

### Optimizing stacked LSTMs

In [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888), you saw that a very effective method of reducing overfitting was to reduce the LR. It’s worth exploring here whether that will have a positive effect on an RNN, too.

For example, the following code reduces the LR by 50%, from 0.0001 to 0.00005:

```py
# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), `lr``=``0.00005`, 
                       betas=(0.9, 0.999), amsgrad=False)

```

[Figure 7-13](#ch07_figure_13_1748549654878247) demonstrates the impact of this on training. As you can see, there’s a small difference in the validation accuracy, indicating that we’re overfitting a bit less.

![](assets/aiml_0713.png)

###### Figure 7-13\. Impact of reduced LR on accuracy with stacked LSTMs

While an initial look at [Figure 7-14](#ch07_figure_14_1748549654878268) similarly suggests a decent impact on loss due to the reduced LR, with the curve not moving up so sharply, it’s worth looking a little closer. We see that the loss on the training set is actually a little higher (~0.35 versus ~0.27) than the previous example, while the loss on the validation set is lower (~0.5 versus 0.6).

Adjusting the LR hyperparameter certainly seems worth investigation.

Indeed, further experimentation with the LR showed a marked improvement in getting training and validation curves to converge, indicating that while the network was less accurate after training, we could tell that it was generalizing better. Figures [7-15](#ch07_figure_15_1748549654878288) and [7-16](#ch07_figure_16_1748549654878310) show the impact of using a lower LR (.0003 rather than .0005).

![](assets/aiml_0714.png)

###### Figure 7-14\. Impact of reduced LR on loss with stacked LSTMs

![](assets/aiml_0715.png)

###### Figure 7-15\. Accuracy with further-reduced LR with stacked LSTM

![](assets/aiml_0716.png)

###### Figure 7-16\. Loss with further-reduced LR and stacked LSTM

Indeed, reducing the LR even further, to .00001, gave potentially even better results, as shown in Figures [7-17](#ch07_figure_17_1748549654878331) and [7-18](#ch07_figure_18_1748549654878358). As with the previous diagrams, while the overall accuracy isn’t as good and the loss is higher, that’s an indication that we’re getting closer to a “realistic” result for this network architecture and not being led into having a false sense of security by overfitting on the training data.

In addition to changing the LR parameter, you should also consider using dropout in the LSTM layers. It works exactly the same as for dense layers, as discussed in [Chapter 3](ch03.html#ch03_going_beyond_the_basics_detecting_features_in_ima_1748570891074912), where random neurons are dropped to prevent a proximity bias from impacting the learning. That being said, you should be careful about setting it *too* low, because when you start tweaking with different architectures, you might freeze the ability of the network to learn.

![](assets/aiml_0717.png)

###### Figure 7-17\. Accuracy with lower LR

**![](assets/aiml_0718.png)

###### Figure 7-18\. Loss with lower LR**  **### Using dropout

In addition to changing the LR parameter, you should also consider using dropout in the LSTM layers. It works exactly the same as for dense layers, as discussed in [Chapter 3](ch03.html#ch03_going_beyond_the_basics_detecting_features_in_ima_1748570891074912), where random neurons are dropped to prevent a proximity bias from impacting the learning.

You can implement dropout by using `nn.Dropout`. Here’s an example:

```py
self.embedding_dropout = nn.Dropout(dropout_rate)
self.lstm_dropout = nn.Dropout(dropout_rate)
self.final_dropout = nn.Dropout(dropout_rate)
```

Then, in your forward pass, you can apply the dropouts at the appropriate levels, like this:

```py
def forward(self, x):
    # Get embeddings
    embedded = self.embedding(x)  

    # Apply first dropout after embedding layer
    embedded = self.embedding_dropout(embedded)

    lstm1_out, _ = self.lstm1(embedded)

    # Apply dropout between LSTM layers
    lstm1_out = self.lstm_dropout(lstm1_out)

    lstm2_out, _ = self.lstm2(lstm1_out)

    # Apply final dropout
    lstm2_out = self.final_dropout(lstm2_out)

    lstm_out = lstm2_out.transpose(1, 2)

    pooled = self.global_pool(lstm_out)  
    pooled = pooled.squeeze(–1) 

    x = self.relu(self.fc1(pooled))
    x = self.sigmoid(self.fc2(x))

    return x
```

When I ran this with the lowest LR I had tested prior to dropout, the network didn’t learn. So, I moved the LR back up to 0.0003 and ran for 300 epochs using this dropout (note that the dropout rate is 0.2, so about 20% of neurons are dropped at random). The accuracy results can be seen in [Figure 7-19](#ch07_figure_19_1748549654878378). The curves for training and validation are still close to each other, but they’re hitting greater than 75% accuracy, whereas without dropout, it was hard to get above 70%.

![](assets/aiml_0719.png)

###### Figure 7-19\. Accuracy of stacked LSTMs using dropout

As you can see, using dropout can have a positive impact on the accuracy of the network, which is good! There’s always a worry that losing neurons will make your model perform worse, but as we can see here, that’s not the case. But do be careful when using dropout because it can lead to underfitting or overfitting if not used appropriately.

There’s also a positive impact on loss, as you can see in [Figure 7-20](#ch07_figure_20_1748549654878397). While the curves are clearly diverging, they are closer than they were previously, and the validation set is flattening out at a loss of about 0.45, which also demonstrates an improvement! As this example shows, dropout is another handy technique that you can use to improve the performance of LSTM-based RNNs.

It’s worth exploring these techniques for avoiding overfitting in your data, and it’s also worth exploring the techniques for preprocessing your data that we covered in [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888). But there’s one thing that we haven’t yet tried: a form of transfer learning in which you can use pre-learned embeddings for words instead of trying to learn your own. We’ll explore that next.

![](assets/aiml_0720.png)

###### Figure 7-20\. Loss curves for dropout-enabled LSTMs**  **# Using Pretrained Embeddings with RNNs

In all the previous examples, you gathered the full set of words to be used in the training set and then trained embeddings with them. You initially aggregated them before feeding them into a dense network, and in this chapter, you explored how to improve the results using an RNN. While doing this, you were restricted to the words in your dataset and how their embeddings could be learned by using the labels from that dataset.

Now, think back to [Chapter 4](ch04.html#ch04_using_data_with_pytorch_1748548966496246), where we discussed transfer learning. What if instead of learning the embeddings for yourself, you could use pre-learned embeddings, where researchers have already done the hard work of turning words into vectors and those vectors are proven? One example of this, as we saw in [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888), is the [GloVe (Global Vectors for Word Representation) model](https://oreil.ly/4ENdQ) developed by Jeffrey Pennington, Richard Socher, and Christopher Manning at Stanford.

In this case, the researchers have shared their pretrained word vectors for a variety of datasets:

*   A 6-billion-token, 400,000-word vocabulary set in 50, 100, 200, and 300 dimensions with words taken from Wikipedia and Gigaword

*   A 42-billion-token, 1.9-million-word vocabulary in 300 dimensions from a common crawl

*   An 840-billion-token, 2.2-million-word vocabulary in 300 dimensions from a common crawl

*   A 27-billion-token, 1.2-million-word vocabulary in 25, 50, 100, and 200 dimensions from a Twitter crawl of 2 billion tweets

Given that the vectors are already pretrained, it’s simple for you to reuse them in your PyTorch code, instead of learning them from scratch. First, you’ll have to download the GloVe data. I’ve opted to use the 6-billion-word version, in 50 dimensions, using this code to download and unzip it:

```py
import urllib.request
import zipfile

# Download GloVe embeddings
url = "https://nlp.stanford.edu/data/glove.6B.zip"
urllib.request.urlretrieve(url, "glove.6B.zip")

# Unzip
with zipfile.ZipFile("glove.6B.zip", 'r') as zip_ref:
    zip_ref.extractall()

# You can use glove.6B.50d.txt (50 dimensions)
# or glove.6B.100d.txt (100 dimensions)
```

Each entry in the file is a word, followed by the dimensional coefficients that were learned for it. The easiest way to use this is to create a dictionary where the key is the word and the values are the embeddings. You can set up this dictionary like this:

```py
import numpy as np
glove_embeddings = dict()
f = open('glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs
f.close()
```

At this point, you’ll be able to look up the set of coefficients for any word simply by using it as the key. So, for example, to see the embeddings for the word *frog*, you could use this:

```py
glove_embeddings['frog']
```

With these pretrained embeddings in hand, you can now load them into the embeddings layer in your neural architecture and use them as pretrained embeddings instead of learning them from scratch. See the following model architecture definition. If the `pretrained_embeddings` value is not null, then the weights for the embedding layer will be loaded from that. If `freeze_embeddings` is `True`, then they’ll be frozen; otherwise, they’ll be used as the starting point for learning (i.e., you’ll fine-tune the embeddings based on your corpus):

```py
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=16, 
                 dropout_rate=0.25, pretrained_embeddings=None, 
                 freeze_embeddings=True, lstm_layers=2):
        super(TextClassificationModel, self).__init__()

        # Initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
```

This model shows a total of 406.817 parameters of which only 6,817 are trainable, so training will be fast!

```py
==========================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================
TextClassificationModel                  [32, 1]                   --
├─Embedding: 1-1                         [32, 60, 50]              (400,000)
├─Dropout: 1-2                           [32, 60, 50]              --
├─LSTM: 1-3                              [32, 60, 16]              6,528
├─Dropout: 1-4                           [32, 60, 16]              --
├─AdaptiveAvgPool1d: 1-5                 [32, 16, 1]               --
├─Linear: 1-6                            [32, 16]                  272
├─ReLU: 1-7                              [32, 16]                  --
├─Dropout: 1-8                           [32, 16]                  --
├─Linear: 1-9                            [32, 1]                   17
├─Sigmoid: 1-10                          [32, 1]                   --
==========================================================================
Total params: 406,817
Trainable params: 6,817
Non-trainable params: 400,000
Total mult-adds (M): 25.34
==========================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 1.02
Params size (MB): 1.63
Estimated Total Size (MB): 2.66
==========================================================================

```

You can now train as before, and you can see how this architecture, with the pretrained embeddings and stacked LSTMs, reduces overfitting really nicely! [Figure 7-21](#ch07_figure_21_1748549654878419) shows the Training versus Validation accuracy on the sarcasm dataset using LSTMs and pretrained GloVe embeddings, while [Figure 7-22](#ch07_figure_22_1748549654878440) shows the loss on training versus validation, where the closeness of the curves demonstrates that we’re not overfitting.

![](assets/aiml_0721.png)

###### Figure 7-21\. Training versus validation accuracy on the sarcasm dataset with LSTMs and GloVe

![](assets/aiml_0722.png)

###### Figure 7-22\. Training and validation loss on the sarcasm dataset with LSTMs and GloVe

For further analysis, you’ll want to consider your vocab size. One of the optimizations you did in the previous chapter to avoid overfitting was intended to prevent the embeddings becoming overburdened with learning low-frequency words: you avoided overfitting by using a smaller vocabulary of frequently used words. In this case, as the word embeddings have already been learned for you with GloVe, you could expand the vocabulary—but by how much?

The first thing to explore is how many of the words in your corpus are actually in the GloVe set. It has 1.2 million words, but there’s no guarantee it has *all* of your words.

When building the `word_index`, you can call `build_vocab_glove` with a *really* large number and it will ignore any words over the total amount. So, for example, say you call this:

```py
word_index = build_vocab_glove(training_sentences, max_vocab_size=100,000)
```

With the sarcasm dataset, you’ll get a vocab_size of 22,457 returned. If you like, you can then explore the GloVe embeddings to see just how many of these words are present in GloVE. Start by creating a dictionary for the embeddings and reading the GloVE file into it:

```py
embeddings_dict = {}
embedding_dim = 50
glove_file = f'glove.6B.{embedding_dim}d.txt'

# Read GloVe embeddings
print(f"Loading GloVe embeddings from {glove_file}...")
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = vector
```

Then, you can compare this with your `word_index` that you created from the entire corpus with the preceding line:

```py
found_words = 0
for word, idx in word_index.items():
    if word in embeddings_dict:
        found_words += 1
print(found_words)
```

In the case of sarcasm, 21,291 of the words were found in GloVE, which is the vast majority, so the principles you used in [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888) to choose how many you should train on (i.e., picking those with sufficient frequency to have a signal) will still apply!

Using this method, I chose to use a vocabulary size of 8,000 (instead of the 2,000 that was previously used to avoid overfitting) to get the results you saw just now. I then tested it with headlines from *The Onion*, the source of the sarcastic headlines in the sarcasm dataset, against other sentences, as shown here:

```py
test_sentences = ["It Was, For, Uh, Medical Reasons, Says Doctor To 
                   Boris Johnson, Explaining Why They Had To Give Him Haircut",
                  "It's a beautiful sunny day",
                  "I lived in Ireland, so in high school they made me 
                   learn to speak and write in Gaelic",
                  "Census Foot Soldiers Swarm Neighborhoods, Kick Down 
                   Doors To Tally Household Sizes"]

```

The results for these headlines are as follows—remember that values close to 50% (0.5) are considered neutral, those close to 0 are considered nonsarcastic, and those close to 1 are considered sarcastic:

```py
tensor([[0.9316],
        [0.1603],
        [0.6959],
        [0.9594]], device='cuda:0')

Text: It Was, For, Uh, Medical Reasons, Says Doctor To Boris Johnson, 
      Explaining Why They Had To Give Him Haircut
Probability: 0.9316
Classification: Sarcastic
--------------------------------------------------------------------------

Text: It's a beautiful sunny day
Probability: 0.1603
Classification: Not Sarcastic
--------------------------------------------------------------------------

Text: I lived in Ireland, so in high school they made me learn to speak 
      and write in Gaelic
Probability: 0.6959
Classification: Sarcastic
--------------------------------------------------------------------------

Text: Census Foot Soldiers Swarm Neighborhoods, Kick Down Doors To Tally 
      Household Sizes
Probability: 0.9594
Classification: Sarcastic
--------------------------------------------------------------------------
```

The first and fourth sentences, which are taken from *The Onion*, showed 93%+ likelihood of sarcasm. The statement about the weather was strongly nonsarcastic (16%), and the sentence about going to high school in Ireland was deemed to be potentially sarcastic but not with high confidence (69%).

# Summary

This chapter introduced you to recurrent neural networks, which use sequence-oriented logic in their design and can help you understand the sentiment in sentences based not only on the words they contain but also on the order in which they appear. You saw how a basic RNN works, as well as how an LSTM can build on this to enable context to be preserved over the long term. These models are the precursors to the popular and famous “transformers” models used to underpin generative AI.

You also used LSTMs to improve the sentiment analysis model you’ve been working on, and you then looked into overfitting issues with RNNs and techniques to improve them, including by using transfer learning from pretrained embeddings.

In [Chapter 8](ch08.html#ch08_using_ml_to_create_text_1748549671852453), you’ll use what you’ve learned so far to explore how to predict words, and from there, you’ll be able to create a model that creates text and writes poetry for you!**