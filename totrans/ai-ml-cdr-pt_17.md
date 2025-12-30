# Chapter 16\. Using LLMs with Custom Data

In [Chapter 15](ch15.html#ch15_transformers_and_transformers_1748549808974580), we looked at Transformers and how their encoder, decoder, and encoder-decoder architectures work. The results of their revolutionizing NLP can’t be disputed! Then, we looked at transformers, which form the Python library from Hugging Face that’s designed to make it easier to use Transformers.

Large Transformer-based models, which are trained on vast amounts of text, are very powerful, but they aren’t always ideal for specific tasks or domains. In this chapter, we’ll look at how you can use transformers and other APIs to adapt these models to your specific needs.

Fine-tuning allows you to customize pretrained models with your specific data. You could use this approach to create a chatbot, improve classification accuracy, or develop text generation for a more specific domain.

There are several techniques for doing this, including traditional fine-tuning and parameter-efficient tuning with methods like LoRA and parameter-efficient fine-tuning (PEFT). You can also get more out of your LLMs with retrieval-augmented generation (RAG), which we’ll explore in [Chapter 18](ch18.html#ch18_introduction_to_rag_1748550073472936).

In this chapter, we’ll explore some hands-on examples, starting with traditional fine-tuning.

# Fine-Tuning an LLM

Let’s take a look, step by step, at how to fine-tune an LLM like BERT. We’ll take the IMDb database and fine-tune the model on it to be better at detecting sentiment in movie reviews. There are a number of steps involved in doing this, so we’ll look at each one in detail.

## Setup and Dependencies

We’ll start by setting up everything that we need to do fine-tuning with PyTorch. In addition to the basics, there are three new things that you’ll need to include:

Datasets

We covered datasets in [Chapter 4](ch04.html#ch04_using_data_with_pytorch_1748548966496246). We’re going to use these to load the IMDb dataset and the built-in splits for training and testing.

Evaluate

This library provides metrics for measuring load performance.

transformers

As we covered in Chapters [14](ch14.html#ch14_using_third_party_models_and_hubs_1748549787242797) and [15](ch15.html#ch15_transformers_and_transformers_1748549808974580), the transformers Hugging Face library is designed to make using LLMs much easier.

We’ll use some classes from the Hugging Face transformers library for this chapter’s fine-tuning exercise. These include the following:

AutoModelForSequenceClassification

This class loads pretrained models for classification tasks and adds a classification head to the top of the base model. This classification head is then optimized for the specific classification scenario you are fine-tuning for, instead of being a generic model. If we specify the checkpoint name, it will automatically handle the model architecture for us. So, to use the BERT model with a linear classifier layer, we’ll use `bert-base-uncased`.

AutoTokenizer

This class automatically initializes the appropriate tokenizer. This converts text to the appropriate tokens and adds the appropriate special tokens, padding, truncation, etc.

TrainingArguments

This class lets us configure the training settings and all the hyperparameters, as well as setting up things like the device to use.

Trainer

This class manages the training loop on your behalf, handling batching, optimization, loss, backpropagation, and everything you need to retrain the model.

DataCollatorWithPadding

The number of records in the dataset doesn’t always line up with the batch size. This class therefore efficiently batches examples to the appropriate batch sizes while also handling details like attention masks and other model-specific inputs.

We can see this in code here:

```py
# 1\. Setup and Dependencies
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
import numpy as np
```

Now that the dependencies are in place, we’ll load the data.

## Loading and Examining the Data

Next up, let’s load our data by using the datasets API. We’ll also explore the test and training dataset sizes. You can use the following code:

```py
# 2\. Load and Examine Data
dataset = load_dataset("imdb")  # Movie reviews for sentiment analysis
print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")
```

That will output the following:

```py
Train size: 25000
Test size: 25000
```

The next step is to initialize the model and the tokenizer.

## Initializing the Model and Tokenizer

We’ll use the `bert-base-uncased` model in this example, so we need to initialize it by using `AutoModelForSequenceClassification` and getting its associated tokenizer:

```py
# 3\. Initialize Model and Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

Note the `AutoModelForSequenceClassification` needs to be initialized with the number of labels that we want to classify for. This defines the new classification head with two labels. The IMDb database that we’ll be using has two labels for positive and negative sentiment, so we’ll retrain for that.

At this point, it’s also a good idea to specify the device that the model will run on. Training with this model is computationally intensive, and if you’re using Colab, you’ll likely need a high-RAM GPU like an A100\. Training with that will take a couple of minutes, but it can take many hours on a CPU!

## Preprocessing the Data

Once we have the data, we want to preprocess it just to get what we need to train. The first step in this, of course, will be to tokenize the text, and the `preprocess` function here handles that, giving a sequence length of 512 characters with padding:

```py
# 4\. Preprocess Data
def preprocess_function(examples):
   result = tokenizer(
       examples["text"],
       truncation=True,
       max_length=512,
       padding=True
   )
   # Trainer expects a column called labels, so copy over from label
   result["labels"] = examples["label"]
   return result

tokenized_dataset = dataset.map(
   preprocess_function,
   batched=True,
   remove_columns=dataset["train"].column_names
)
```

One important note here is that the original data came with columns for `text` denoting the review and `label` being 0 or 1 for negative or positive sentiment. However, we don’t *need* the `text` column to train the data, and the Hugging Face Trainer (which we will see in a moment) expects the column containing the label to be called `labels` (plural). Therefore, you’ll see that we remove all of the columns in the original dataset, and the tokenized dataset will have the tokenized data and a column called `labels` instead of `label`, with the original values copied over.

## Collating the Data

When we’re dealing with passing sequenced, tokenized data into a model in batches, there can be differences in batch or sequence size that need processing. In our case, we shouldn’t have to worry about the sequence size because we used a tokenizer that forces the length to be 512 (in the previous set). However, as part of the transformers library, the collator classes are still equipped to deal with it, and we’ll be using them to ensure consistent batch sizing.

So ultimately, the role of the `DataCollatorWithPadding` class is to take multiple examples of different lengths, provide padding if and when necessary, convert the inputs into tensors, and create attention masks if necessary.

In our case, we’re really only getting the conversion to tensors for input to the model, but it’s still good practice to use `DataCollatorWithPadding` if we want to change anything in the tokenization process later.

Here’s the code:

```py
# 5\. Create Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

## Defining Metrics

Now, let’s define some metrics that we want to capture as we’re training the model. We’ll just do accuracy, where we compare the predicted value to the actual value. Here’s some simple code to achieve that:

```py
# 6\. Define Metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
```

It’s using `evaluate.load` from Hugging Face’s evaluate library, which provides a simple standardized interface that’s specifically designed for tasks like this one. It can handle the heavy lifting for us, instead of requiring us to roll our own metrics, and for an evaluate task, we simply pass it the set of predictions and the set of labels and have it do the computation. The evaluate library is prebuilt to handle a number of metrics, including f1, BLEU, and many others.

## Configuring Training

Next up, we can configure *how* the model will retrain by using the `TrainingArguments` object. This offers a large variety of hyperparameters you can set—including those for the learning rate, weight decay, etc., as used by the `optimizer` and `loss` function. It’s designed to give you granular control over the learning process while abstracting away the complexity.

Here’s the set that I used for fine-tuning with IMDb:

```py
# 7\. Configure Training
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    report_to="none",
    fp16=True
)
```

It’s important to note and tweak hyperparameters for different results. In addition to the aforementioned ones for the optimizer, you’ll want to consider the batch sizes. You can set different parameters for training or evaluation.

One very useful parameter—in particular for training sessions that are longer than the three epochs here—is `load_best_model_at_end`. Instead of always using the final checkpoint, it will keep track of the best checkpoint according to the specified metric (in this case, accuracy) and will load that one when it’s done. And because I set the `evaluation` and `save` strategies to `epoch`, it will only do this at the end of an epoch.

Note also the `report_to` parameter: the training uses `weights and biases` as the backend for reporting by default. I set `report_to` to `none` to turn off this reporting. If you want to keep it, you’ll need a Weights and Biases API key. You can get this very easily from the status window or by going to the [Weights and Biases website](https://oreil.ly/yMX1A). As you train, you’ll be asked to paste in this API key. Be sure to do so before you walk away, particularly if you are paying for compute units on Colab.

There’s a wealth of parameters to experiment with, and being able to parameterize easily like this also allows you easily to do a neural architecture search with tools like [Ray Tune](https://oreil.ly/fDAhG).

## Initializing the Trainer

As with the training parameters, transformers give you a trainer class that you can use alongside them to encapsulate a full training cycle.

You initialize it with the model, the training arguments, the data, the collator, and the metrics strategy that you’ve previously initialized. All the previous steps build up to this. Here’s the code you’ll need:

```py
# 8\. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```

## Training and Evaluation

With everything now set up, it becomes as simple as calling the `train()` method on the trainer to do the training and the `evaluate()` method to do the evaluation.

Here’s the code:

```py
# 9\. Train and Evaluate
train_results = trainer.train()
print(f"\nTraining results: {train_results}")

eval_results = trainer.evaluate()
print(f"\nEvaluation results: {eval_results}")
```

As an example, while you could train this model with the free tiers in Google Colab, your experience in timing might vary. With the CPU alone, it can take many hours. I trained this model with the T4 High Ram GPU, which costs 1.6 compute units per hour. The entire training process was about 50 minutes, but I’ll round that up to an hour to include all the downloading and setup. At the time of writing, a pro Colab subscription gets one hundred compute Units with the US$9.99 per month subscription. You could also choose the A100 GPU, which is much faster (training took me about 12 minutes with it) but also more expensive, at about 6.8 compute units per hour.

After training, the results looked like this:

```py
Training results:
TrainOutput(global_step=585,
            training_loss=0.18643947177463108,
            metrics={'train_runtime': 597.9931,
            'train_samples_per_second': 125.42,
            'train_steps_per_second': 0.978,
            'total_flos': 1.968912649469952e+16,
            'train_loss': 0.18643947177463108,
            'epoch': 2.9923273657289})
Evaluation results:
            {'eval_loss': 0.18489666283130646,
            'eval_accuracy': 0.93596,
            'eval_runtime': 63.8406,
            'eval_samples_per_second': 391.601,
            'eval_steps_per_second': 48.95,
            'epoch': 2.9923273657289}
```

We can see quite high accuracy on the evaluation dataset (about 94%) after only three epochs, which is a good sign—but of course, there may be overfitting going on that would require a separate evaluation. But after about 12 minutes of work fine-tuning an LLM, we’re clearly moving in the right direction!

## Saving and Testing the Model

Once we’ve trained the model, it’s a good idea to save it out for future use, and the `trainer` object makes this easy:

```py
# 10\. Save Model
trainer.save_model("./final_model")

```

Once we’ve saved the model, we can start using it. To that end, let’s create a helper function that takes in the input text, tokenizes it, and then turns those tokens into a set of input vectors of keys and values (k, v):

```py
# 11\. Example Usage
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

```

We can then use PyTorch in inference mode to get the outputs from those inputs and turn them into a set of predictions:

```py
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

```

The returned predictions will be a tensor with two dimensions. Neuron 0 is the probability that the prediction is negative, and neuron 1 is the probability that the prediction is positive. Therefore, we can look at the positive probability and return a sentiment and confidence with its values. We could also have done the same with the negative one; it’s purely arbitrary:

```py
    positive_prob = predictions[0][1].item()
    return {
        'sentiment': 'positive' if positive_prob > 0.5 else 'negative',
        'confidence': positive_prob if positive_prob > 0.5 else 1 – positive_prob
    }
```

We can now test the prediction with code like this:

```py
# Test prediction
test_text = "This movie was absolutely fantastic! The acting was superb."
result = predict_sentiment(test_text)
print(f"\nTest prediction for '{test_text}':")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

And the output would look something like this:

```py
Test prediction for 'This movie was absolutely fantastic! 
                     The acting was superb.':
Sentiment: positive
Confidence: 99.16%
```

We can see that this statement is positive, with high confidence!

In this process, you can see how, step by step, you can fine-tune an existing LLM on new data to turn it into a classification engine! In many circumstances, this may be overkill (and training your own model instead of fine-tuning an LLM may be quicker and cheaper), but it’s certainly worth evaluating this process. Sometimes, even untuned LLMs will work well for classification! In my experience, using the general artificial-understanding nature of LLMs will lead to the creation of far more effective classifiers with stronger results.

# Prompt-Tuning an LLM

A lightweight alternative to fine-tuning is *prompt tuning,* in which you can adapt a model to specific tasks. With prompt tuning, you do this by prepending trainable *soft prompts* to each input instead of modifying the model weights. These soft prompts will then be optimized during training.

These soft prompts are like learned instructions that guide the model’s behavior. Unlike discrete text prompts (such as `Classify the sentiment`), the idea of soft prompts is that they exist in the model’s embedding space as continuous vectors. So, for example, when processing “This movie was great,” the model would see “[V1][V2]…[V20]This movie was great.” In this case, [V1][V2]...[V20] are vectors that will help steer the model toward the desired classification.

Ultimately, the advantage here is efficiency. So instead of fine-tuning a model, amending its weights for each task, and saving the entire model for reuse, you only need to save the soft prompt vectors. These are much smaller, and they can help you have a suite of fine-tunes that you can easily use to guide the model to a specific task without needing to manage multiple models.

Prompt tuning like this can actually match or exceed the performance of full fine-tuning, particularly with larger models, and it’s significantly more efficient.

Now, let’s explore how to prompt-tune the BART LLM with the IMDb dataset in direct comparison to the fine-tuning earlier in this chapter.

## Preparing the Data

Let’s start by preparing our data, loading it from the IMDb dataset, and setting up the virtual tokens. Here’s the code you’ll need:

```py
# Data preparation
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_length = 512
num_virtual_tokens = 20

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length - num_virtual_tokens
    )
```

This will also tokenize our incoming examples so you should note that the maximum length of any example will now be reduced by the number of virtual tokens. So, for example, with BERT, we have a maximum length of 512, but if we’re going to have 20 virtual tokens, then the sequence maximum length will now be 492.

We’ll now load a subset of the data and try with 5,000 examples, instead of 25,000\. You can experiment with this number and trade off smaller amounts for faster training against larger amounts for better accuracy.

First, we’ll create the indices that we want to take from the dataset for training, and we’ll test them. Think of these as pointers to the records we’re interested in. We’re randomly sampling here:

```py
# Use only 5000 examples for training
train_size = 5000
np.random.seed(42)
train_indices = np.random.choice(len(dataset["train"]), train_size, 
                                                        replace=False)
test_indices = np.random.choice(len(dataset["test"]), train_size, replace=False)

tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)
```

Then, the last two lines define the mapping function, which simply takes the values from the dataset we’re interested in and tokenizes them. We’ll see that in the next step.

## Creating the Data Loaders

Now that we have sets of tokenized training and test data, we want to turn them into data loaders. We’ll do this by first selecting the raw examples from the underlying data that match the content in our indices:

```py
# Create subset for training
tokenized_train = tokenized_train.select(train_indices)
tokenized_test = tokenized_test.select(test_indices)

```

Then, we’ll set the format of the data that we’re interested in. There may be many columns in a dataset, but you won’t use them all for training. In this case, we’ll want the `input_ids`, which are the tokenized versions of our input content; the `attention_mask`, which is a set of vectors that tells us which tokens in the `input_ids` we should be interested in (this has the effect of filtering out padding or other nonsemantic tokens); and the label:

```py
tokenized_train.set_format(type="torch", columns=["input_ids", 
                                                  "attention_mask", "label"])
tokenized_test.set_format(type="torch", columns=["input_ids", 
                                                 "attention_mask", "label"])

```

Now, we can specify the DataLoader that takes these training and test sets. I have a large batch size here because I was testing on a 40Gb GRAM GPU in Colab. In your environment, you may need to adjust these:

```py
train_dataloader = DataLoader(tokenized_train, batch_size=64, shuffle=True)
eval_dataloader = DataLoader(tokenized_test, batch_size=128)
```

Now that the data is processed and loaded into DataLoaders, we can go to the next step: defining the model.

## Defining the Model

First, let’s see how to instantiate the model, and then we can go back to the raw definition. Typically, in our code, once we’ve set up our DataLoaders, we’ll want to create an instance of the model. We’ll use code like this:

```py
# Define the model
model = PromptTuningBERT(num_virtual_tokens=num_virtual_tokens, 
                         max_length=max_length)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
```

This keeps it nice and simple, and we’ll encapsulate the underlying BERT in an override for a prompt-tuning version. Now, as nice as it would be for transformers to have one, they don’t, so we need to create this class for ourselves.

As we would with any PyTorch class that defines a model, we’ll create it with an `__init__` method to set it up and a forward method that PyTorch’s training loop will call during the forward pass. So, let’s start with the `__init__` method and the class definition:

```py
class PromptTuningBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", 
                       num_virtual_tokens=50, 
                       max_length=512):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
                        model_name, 
                        num_labels=2)
        self.bert.requires_grad_(False)

        self.n_tokens = num_virtual_tokens
        self.max_length = max_length - num_virtual_tokens

        vocab_size = self.bert.config.vocab_size
        token_ids = torch.randint(0, vocab_size, (num_virtual_tokens,))
        word_embeddings = self.bert.bert.embeddings.word_embeddings
        prompt_embeddings = word_embeddings(token_ids).unsqueeze(0)
        self.prompt_embeddings = nn.Parameter(prompt_embeddings)
```

There’s a lot going on here, so let’s break it down little by little. First of all, I set the defaults for the `num_virtual_tokens` to 50 and the `max_length` default to 512\. If you don’t specify your own defaults when you instantiate the class, you’ll get these values. In this case, the calling code sets them to 20 and 512, respectively, but you’re free to experiment.

Next, the code sets up the transformers `AutoModelForSequenceClassification` class to get BERT:

```py
        self.bert = AutoModelForSequenceClassification.from_pretrained(
                        model_name, 
                        num_labels=2)
```

As with fine-tuning for IMDb, we’re interested in training the model to recognize two labels, so they’re set up here. However, one difference from fine-tuning is that we’re not going to change any of the weights within the BERT model itself, so we set that we don’t want gradients and freeze it like this:

```py
self.bert.requires_grad_(False)
```

The secret sauce in generating the soft prompts that we’re going to use comes at the end of the init. We’ll create a vector to contain our number of virtual tokens, and I just initialized it with random tokens from the vocabulary. There are smarter things that we might do here to make training more efficient over time, but for the sake of simplicity, let’s go with this:

```py
token_ids = torch.randint(0, vocab_size, (num_virtual_tokens,))
```

The pretrained BERT model in transformers comes with embeddings, so we can use them to turn our list of random tokens into embeddings:

```py
word_embeddings = self.bert.bert.embeddings.word_embeddings
prompt_embeddings = word_embeddings(token_ids).unsqueeze(0)
```

Importantly, we should now specify that the `prompt_embeddings` are parameters of the neural network. This will be important later, when we define the optimizer. We recently specified that all of the BERT parameters were frozen, but *these* parameters are not part of that and thus are not frozen, so they will be tweaked by the optimizer during training:

```py
self.prompt_embeddings = nn.Parameter(prompt_embeddings)
```

We have now initialized a subclassed version of the tunable BERT, specified that we don’t want to amend its gradients, and created a set of soft prompts that we will append to the examples as we’re training—and we’ll tweak only those soft prompts to soft-tune the two output neurons.

Now, let’s look at the `forward` function that will be called during the forward pass at training time. Given that we’ve set up everything, this is pretty straightforward:

```py
def forward(self, input_ids, attention_mask, labels=None):
    batch_size = input_ids.shape[0]
    input_ids = input_ids[:, :self.max_length]
    attention_mask = attention_mask[:, :self.max_length]

    embeddings = self.bert.bert.embeddings.word_embeddings(input_ids)
    prompt_embeddings = self.prompt_embeddings.expand(batch_size, –1, –1)
    inputs_embeds = torch.cat([prompt_embeddings, embeddings], dim=1)

    prompt_attention_mask = torch.ones(batch_size, self.n_tokens, 
                                       device=attention_mask.device)
    attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

    return self.bert(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True
    )
```

Let’s look at it step-by-step. During the forward pass in training, this function will be passed batches of data. Therefore, we need to understand what the size of this batch is and then extract the `input_ids` (the tokens for the values read from the dataset) and the attention mask for that particular ID:

```py
    batch_size = input_ids.shape[0]
    input_ids = input_ids[:, :self.max_length]
    attention_mask = attention_mask[:, :self.max_length]

```

We’ll also need to convert the `input_ids` into embeddings:

```py
embeddings = self.bert.bert.embeddings.word_embeddings(input_ids)
```

Our soft prompts are also tokenized sentences. Originally, they were initialized to random words, and we’ll see over time that they’ll adjust appropriately. But for this step, these tokens need to be converted to embeddings:

```py
prompt_embeddings = self.prompt_embeddings.expand(batch_size, –1, –1)

```

The `expand` method just adds the batch size to the prompt embeddings. When we defined the class, we didn’t know how large each batch coming in would be (and the code is written to let you tweak that based on the size of your available memory), so using `expand(batch_size, –1, –1)` turns the vector of prompt embeddings, which was of shape `[1, num_prompt_tokens, embedding_dimensions]`, into `[batch_size, num_prompt_tokens, embedding_dimensions]`.

Our soft prompt tuning involved prepending the soft embeddings to the embeddings for the actual input data, so we do that with this:

```py
inputs_embeds = torch.cat([prompt_embeddings, embeddings], dim=1)
```

BERT uses an `attention_mask` to filter out the tokens we don’t want to worry about at training or inference time, which are usually the padding tokens. But we want BERT to pay attention to all of the soft prompt tokens, so we’ll set the attention mask for them to be all 1s and then append that to the incoming attention mask(s) for the training data. Here’s the code:

```py
prompt_attention_mask = torch.ones(batch_size, self.n_tokens, 
                                   device=attention_mask.device)

attention_mask = torch.cat([prompt_attention_mask, attention_mask], 
                            dim=1)

```

Now that we’ve done all our tuning, we need to pass the data to the model to have it optimize and calculate the loss:

```py
    return self.bert(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True
    )

```

We will see how this data is used in the training loop, next.

## Training the Model

The key to this training is that we’re going to do a full, normal training loop but in a special circumstance. In this case, we previously froze *everything* in the BERT model, *except* for the soft prompts that we defined as model parameters. Therefore, say we define the optimizer like this:

```py
optimizer = AdamW(model.parameters(), lr=1e-2)
```

In this case, we’re using standard code, telling it to tweak the model’s parameters. But the only ones that are available to tune are the soft prompts, so this should be quick!

Note that the value for the learning rate is quite large. This helps the system learn quickly, but in a real system, you’d likely want the value to be smaller—or at least adjustable, starting large and then shrinking in later epochs.

So now, let’s get into training. First, we’ll set up the training loop:

```py
num_epochs = 3

# Perform the training
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
```

### Managing data batches

For each batch, we’ll get the columns (`input_ids` and `attention_masks`) as well as the labels and pass them to the model:

```py
for batch in tqdm(train_dataloader, 
                  desc=f'Training Epoch {epoch + 1}'):
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch.pop('label')
    outputs = model(**batch, labels=labels)
```

This looks a little different from ones earlier in this book, but it’s pretty much doing the same thing. The `tqdm` code just gives us a status bar because we’re training. We read the data batch by batch, but we want the data to be on the same device as the model. So, for example, if the model is running on a GPU, we want it to access data in the GPU’s memory. Therefore, this line will iterate through each column, reading the key and passing the value to the device:

```py
batch = {k: v.to(device) for k, v in batch.items()}
```

It redefines the batch that was read in to ensure that the data is on the same device as the model. But we don’t want the labels to be in the batch because the model expects them to be fed in separately, so we remove them from the batch with the `pop()` method:

```py
labels = batch.pop('label')
```

Now, we can use the shorthand of `**batch` to pass the set of input values (in this case, the `input_ids` and the `attention_mask`) to the forward method of the model and unpack the dictionary along with the labels, like this:

```py
outputs = model(**batch, labels=labels)
```

### Handling the loss

The forward pass sends the data to the model and gets the loss back. We use this to update our overall loss, and we can then backward pass:

```py
loss = outputs.loss
total_train_loss += loss.item()

loss.backward()
```

With the gradients flowing back, the optimizer can now do its job.

### Optimizing for loss

Remembering that the `model.parameters()` will only manage the *trainable unfrozen* parameters, we can now call the optimizer. I added something called *gradient clipping* here to make the training a little more efficient, but the rest is just calling the optimizer’s next step and then zeroing out the gradients so we can use them next time:

```py
clip_grad_norm_(model.parameters(), max_grad_norm)  # Add here
optimizer.step()
optimizer.zero_grad()
```

###### Note

The idea behind *gradient clipping* is that sometimes, during backpropagation, the gradients can be too large and the optimizer might take very large steps. This can lead to a problem called *exploding gradients*, in which the value changes hide the nuances of what might be learned. But clipping scales the gradients down if their values grow too large, and in a situation like this one, they may not even be necessary.

## Evaluation During Training

We also have a set of test data, so we can evaluate how the model performs during the training cycle. In each epoch, once the forward and backward passes are done and the model parameters are reset, we can switch the model into evaluation mode and then start passing all of the test data through it to get inference. We’ll also compare the results of the inference against the actual labels to calculate accuracy:

```py
model.eval()
val_accuracy = []
total_val_loss = 0
```

Then, we’ll have similar code—but this time, it will be to read the eval batches, turn them into outputs with labels, and get predictions and loss values from the model:

```py
with torch.no_grad():
    for batch in tqdm(eval_dataloader, desc='Validating'):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('label')

        outputs = model(**batch, labels=labels)
        total_val_loss += outputs.loss.item()

        predictions = torch.argmax(outputs.logits, dim=-1)
        val_accuracy.extend((predictions == labels).cpu().numpy())
```

Once we’ve calculated these values, then at the end of each epoch, we can report on them and on training loss.

## Reporting Training Metrics

During training in each epoch, we calculated the training loss, so we can now get the average across all records. We can do the same thing with the validation loss and (of course) with the validation accuracy and then report on them:

```py
avg_train_loss = total_train_loss / len(train_dataloader)
avg_val_loss = total_val_loss / len(eval_dataloader)
val_accuracy = np.mean(val_accuracy)

print(f"\nEpoch {epoch + 1}:")
print(f"Average training loss: {avg_train_loss:.4f}")
print(f"Average validation loss: {avg_val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")
```

Running this training for three epochs gives us this:

```py
Training Epoch 1: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s]
Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]

Epoch 1:
Average training loss: 0.6559
Average validation loss: 0.6037
Validation accuracy: 0.8036
Training Epoch 2: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s]
Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]

Epoch 2:
Average training loss: 0.6112
Average validation loss: 0.5854
Validation accuracy: 0.8386
Training Epoch 3: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s]
Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]

Epoch 3:
Average training loss: 0.5799
Average validation loss: 0.5270
Validation accuracy: 0.8736

```

This was done on an A100 in Colab with 40 Gb of GRAM, and as you can see, each epoch only took about 1 minute to train and 30 seconds to evaluate.

By the end, the average training loss had dropped from about 0.65 to 0.58\. The accuracy was 0.8736\. So, it’s likely overfitting because we only trained for three epochs.

## Saving the Prompt Embeddings

What’s really nice about this approach is that you can simply save out the prompt embeddings when you’re done. You can also load them back in for inference later, as you’ll see in the next section:

```py
torch.save(model.prompt_embeddings, "imdb_prompt_embeddings.pt")
```

What I find really cool about this is that this file is relatively small (61 K), and it doesn’t require you to amend the underlying model in any way. Thus, in an application, you could potentially have a number of these prompt-tuning files and hot-swap and replace them as needed so that you can have multiple models that you can orchestrate, which is the basis for an agentic solution.

## Performing Inference with the Model

To perform inference with a prompt-tuned model, you’ll simply define the model with the soft prompts and then, instead of training them, load the pretrained soft prompts back from disk. We’ll explore that in this section. If you don’t want to train your own model, then [in the download](https://github.com/lmoroney/PyTorch-Book-FIles), I’ve provided soft prompts from a version of the model that was trained for 30 epochs instead of 3.

For tidier encapsulation, I created a class that is similar to the one we used for training but that is just for inference. I call it `PromptTunedBERTInference`, and here’s its initializer:

```py
class PromptTunedBERTInference:
    def __init__(self, model_name="bert-base-uncased", 
                       prompt_path="imdb_prompt_embeddings.pt"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model =  
                  AutoModelForSequenceClassification.from_pretrained(
                                            model_name, num_labels=2)
        self.model.eval()
        self.prompt_embeddings = torch.load(prompt_path)
        self.device = 
           torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
```

It’s very similar to the initializer for the trainable one, except for a couple of important points. The first is that because we’re *only* using it for inference, I’ve set it into eval mode:

```py
self.model.eval()
```

The second is that we don’t need to train the embeddings and do all the associated plumbing—instead, we just load them from the specified path:

```py
self.prompt_embeddings = torch.load(prompt_path)
```

And that’s it! As you can see, it’s quite lightweight and pretty straightforward. It won’t have a `forward` function because we’re not training it, but let’s add a `predict` function that encapsulates doing inference with it.

### The predict function

The job of the `predict` function is to take in the string(s) that we want to perform inference with, tokenize it (them), and then pass it to the model with the soft tokens prepended. Let’s take a look at the code, piece by piece.

First, let’s define it and have it accept text that it will then tokenize:

```py
def predict(self, text):
    inputs = self.tokenizer(text, padding=True, 
                        truncation=True,
                        max_length=512-self.prompt_embeddings.shape[1],
                        return_tensors="pt")
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
```

The text will be tokenized up to the maximum length, less the size of the soft prompt, and then each of the items in the input will be loaded into a dictionary. Note that the tokenizer will return multiple columns for the text—usually, the tokens and the attention mask—so we’ll follow this approach to turn them into a set of key-value pairs that are easy for us to work with later.

Now that we have our inputs, it’s time to pass them to the model. We’ll start by putting `torch` into `no_grad()` mode because we’re not interested in training gradients. We’ll then get the embeddings for each of our tokens:

```py
with torch.no_grad():
    embeddings = self.model.bert.embeddings.word_embeddings(
                                        inputs['input_ids'])

    batch_size = embeddings.shape[0]

    prompt_embeds = self.prompt_embeddings.expand(
                         batch_size, –1, –1).to(self.device)

    inputs_embeds = torch.cat([prompt_embeds, embeddings], dim=1)
```

We have an attention mask for the input that’s generated by the tokenizer, but we don’t have one for the soft prompt. So, let’s create one and then append it to the input attention mask:

```py
attention_mask = inputs['attention_mask']
prompt_attention = torch.ones(batch_size, self.prompt_embeddings.shape[1],
                            device=self.device)
attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
```

Now that we have everything in place, we can pass our data to the model to get our inferences back:

```py
outputs = self.model(inputs_embeds=inputs_embeds,
                   attention_mask=attention_mask)
```

The outputs will be the logits from the two neurons, one representing positive sentiment and the other negative. We can then Softmax these to get the prediction:

```py
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
return {"prediction": outputs.logits.argmax(-1).item(),
       "confidence": probs.max(-1).values.item()}
```

### Usage example

Using this class for a prediction is then pretty straightforward. We create an instance of the class and pass a string to it to get results. The results will contain a prediction and a confidence value, which we can then output:

```py
# Usage example
if __name__ == "__main__":
    model = PromptTunedBERTInference()
    result = model.predict("This movie was great!")
    print(f"Prediction: {'Positive' 
                          if result['prediction'] == 1 else 'Negative'}")
    print(f"Confidence: {result['confidence']:.2f}")

```

One note you might see with prompt tuning is low confidence values that can lead to mis-predictions, especially with binary classifiers like this one. It’s good to explore your inference to make sure that it’s working well, and there are also techniques that you could explore to ensure that the logits are giving the values you want. These include setting the temperature of the Softmax, using more prompt tokens to give the model more capacity, and initializing the prompt tokens with sentiment-related words (instead of random tokens, like we did here).

# Summary

In this chapter, we explored different methods for customizing LLMs with our own data. We looked at two main approaches: traditional fine-tuning and prompt tuning.

Using the IMDb dataset, you saw how to fine-tune BERT for sentiment analysis and walked through all the steps—from data preparation, to model configuration, to training and evaluation. The model achieved an impressive 95% accuracy in sentiment classification in just a few epochs.

However, fine-tuning may not be appropriate in all cases, and to that end, you explored a lightweight alternative called prompt tuning. Instead of modifying model weights, the idea here was to prepend trainable soft prompts to inputs, which are optimized during training. This approach provides significant advantages in that it can be much faster and it doesn’t change the underlying model. In this case, the tuned prompts could be saved (and they were only a few Kb) and then reloaded to program the model to perform the desired task. You then went through a full implementation, showing you how to create, train, and save these soft prompts, plus load them back to perform inference.

In the next chapter, we’ll explore how you can serve LLMs, including customized ones. I’ll explain how to do this in your own data center by using Ollama, which is a powerful tool for handling the serving and management of LLMs. You’ll learn how to take models and turn them into services, and we’ll also explore how to set up Ollama and use it over HTTP to talk with models in your data center.