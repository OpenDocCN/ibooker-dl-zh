# Chapter 12\. Concepts of Inference

In the previous chapters of this book, you focused on *training* models using PyTorch and on how to create models that manage images (aka Computer Vision), text content (aka NLP), and sequence modelling. For the rest of this book, you’ll cover a lot of content around *using trained models* to make predictions from new data (aka *inference*) and in particular using large generative models for text-to-text and text-to-image generative AI.

But before you jump into that, it’s important for you to understand the underlying data transfer technology. We’ve touched on it a little in the training chapters, but as you go deeper into ML—in either training or inference—it’s important for you to be able to understand the underlying concepts of tensors.

Ultimately, no matter what data type you have, you’ll convert it into tensors to pass it *into* the model. Similarly, no matter the data type in which you want to present answers from the model to your users, you’ll get them back as tensors as well!

In many cases, you’ll have helper functions, such as the *transformers* that you’ll see in [Chapter 15](ch15.html#ch15_transformers_and_transformers_1748549808974580) (which covers LLMs) and the *diffusers* that you’ll see in [Chapter 19](ch19.html#ch19_using_generative_models_with_hugging_face_diffuser_1748573005765373) (which handles image generation). And while you won’t be touching tensors with them, you’ll still be using them under the hood.

# Tensors

A *tenso*r is an array that can have any number of dimensions. Tensors are typically used to represent numerical data for deep-learning algorithms; they’re containers that can hold numbers in multiple dimensions.

Tensors can be simple scalar values (in zero dimensions), vectors (in one dimension), matrices (in two dimensions), and beyond (in three dimensions or more). In PyTorch, they’re the fundamental data structure for all computation.

Tensors are also the source of the name *TensorFlow* for the alternative deep-learning framework from Google.

Here are some examples of tensors in PyTorch, which were created using `torch.​ten⁠sor`:

```py
import torch

# Scalar (0D tensor)
scalar = torch.tensor(42)  # Single number

# Vector (1D tensor)
vector = torch.tensor([1, 2, 3, 4])  # Array of numbers

# Matrix (2D tensor)
matrix = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])  # 2x3 grid of numbers

# 3D tensor
cube = torch.tensor([[[1, 2], [3, 4]],
                    [[5, 6], [7, 8]]])  # 2x2x2 cube of numbers
```

What makes tensors so useful for ML is this flexibility to hold different value types. Numbers can be 0D tensors, the embedding vectors representing text can be 1D, and images can be 3D, with dimensions for height, width, and pixel value. Plus, all of these can have an additional dimension added for batches. So, for example, a single image can be a 3D matrix, but 100 images instead of 100 3D matrices could be a single 4D tensor, with the fourth dimension being the index of the image!

When you’re using `torch.tensor`, keep in mind that a lot of work and investment has been put into optimizing them to run on GPUs, which makes them extremely efficient for deep learning computation.

# Image Data

Images are typically stored in formats like JPEG or PNG, which are optimized for *human* viewing as well as storage efficiency. Each dot (or pixel) in the image is usually composed of a number of values, with each value being the intensity of a color channel. Typically, an image will have 24 bits of data, with 8 bits assigned each to red, green, and blue channels. If you see 32 bits, then the additional 8 are for an alpha, or transparency, channel.

So, for example, a green pixel might have values 0 on the red, blue, and alpha channels and 255 on the green channel. If it’s semi-transparent, it might still have a value of 0 on red and blue, 128 on alpha, and 255 on green.

An image file is typically *compressed*, which means that mathematical transforms have been applied to it to avoid wasted data and make the image smaller. An image can also contain metadata, file headers, and more. Then, once the image is loaded into memory, it is usually uncompressed to the 32-bits-per-pixel previously described.

ML models typically use values between –1 and 1, and not 0 to 255, as the native values of the image are stored. If we want to learn the details of an image, it’s good for us to standardize the values by focusing on the meaningful *variations* between the pixel intensities as opposed to just their values. So, one method is to standardize by figuring out how far the pixel value is from the mean and standard deviations. This gives a broader spectrum of values that can lead to smoother loss curves and more effective learning.

In PyTorch, you achieve this with code like this:

```py
import torch
from torchvision import transforms
from PIL import Image

def prepare_image(image_path):
    # Load the image using PIL
    raw_image = Image.open(image_path)

    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply transformations
    input_tensor = preprocess(raw_image)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)
    return input_batch
```

This code uses a very popular Python library called Pillow, or just PIL.

In this case, the `PIL Image.open` will read the image, decompress it into pixels, and then apply a set of transforms to those pixels. The transforms normalize the channels into ML-friendly values, as described earlier, and then convert them into tensors.

The `input_tensor` is then a 3D matrix/tensor, but if we want to have a number of them in a single batch, we can *unsqueeze* this tensor to add a new dimension, making it a 4D tensor.

You can see a single image in [Figure 12-1](#ch12_figure_1_1748549754439352) as a 3D tensor, with one dimension for each color depth.

![](assets/aiml_1201.png)

###### Figure 12-1\. Tensor representing a full-color image

And then, if there’s a batch of images, you can see it as a fourth dimension in [Figure 12-2](#ch12_figure_2_1748549754439405).

![](assets/aiml_1202.png)

###### Figure 12-2\. Tensor representing a batch of colored images

# Text Data

Typically, text is stored in a string, like “The cat sat on the mat,” but training a model or passing text like this to a pretrained model is unfeasible. Models, as you saw in earlier chapters, are trained on numeric data—and the best way to do that is to either *tokenize* the text, by turning words or subwords into numbers, or to *calculate embeddings* for the text, by turning them into vectors. And if you choose to calculate embeddings for the text, you can also encode sentiment about the text into the direction of the vector (see [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888)).

So, as a simple example, let’s consider the following sentences:

```py
texts = [
    "I love my dog",
    "The manatee became a doctor"
]
```

You can *tokenize* this text into a series of numbers by using a tokenizer. You can create your own tokenized series of numbers from the corpus, as you did in [Chapter 5](ch05.html#ch05_introduction_to_natural_language_processing_1748549080743759), or you can use an existing one, like this:

```py
import torch
from transformers import BertTokenizer, BertModel

def text_to_embeddings(texts):
    # Load pretrained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Set to evaluation mode

    # Tokenize the input texts
    encoded = tokenizer(
        texts,
        padding=True,      # Pad shorter sequences to match longest
        truncation=True,   # Truncate sequences that are too long
        return_tensors='pt'  # Return PyTorch tensors
    )

```

You’ll learn a lot more about transformers, including how to use them for BERT, starting in [Chapter 13](ch13.html#ch13_hosting_pytorch_models_for_serving_1748549772563124).

The important line here is the last one, in which we ask the tokenizer to return tensors in PyTorch format. You can see the result of this in the output of the encodings, like this:

```py
Encodings: 
tensor([[  101,  1045,  2293,  2026,  3899,   102,     0,     0],
        [  101,  1996, 24951, 17389,  2150,  1037,  3460,   102]])
```

If you explore this closely, you’ll see that the two sentences have been turned into a series of numbers. The first sentence, which has four words, has six numbers, and the second, which has six words, has eight numbers. Each has the number 101 at the front and 102 at the end, which gives us the two extra tokens. These are special tokens the encoder has added to indicate the start and end of the sentence.

A string can be represented as a 1D vector, but we have multiple strings here, so we can add a dimension to 1D to get 2D, and the second dimension gives us each string. So, value 0 in the second dimension is the first string, value 1 is the second string, etc.

The BERT model can also create embeddings from the sentence by getting an embedding for each word in the sentence and summing them all up to get an overall set of values. In the base model version of BERT, each embedding vector is a 1D matrix with 768 values. The embedding for a sentence is the same. Multiple sentences, just like the previous encodings, will have a second dimension.

Here’s the code:

```py
# Generate embeddings
with torch.no_grad():  # No need to calculate gradients
    outputs = model(**encoded)
    embeddings = outputs.last_hidden_state

```

Each embedding has 768 values, so I’m not going to print them all out, but for the embedding that represents the first sentence, you can see values like this:

```py
First word embedding (first 5 values): 
        tensor([ 0.0401,  0.3046,  0.0669, –0.1975, –0.0103])
```

Don’t worry if you don’t fully understand this yet—we’ll be going into it in more detail starting in [Chapter 13](ch13.html#ch13_hosting_pytorch_models_for_serving_1748549772563124). The important point here is that the idea of a *tensor* is that the very flexible matrix, which can have any number of dimensions, gives you a consistent input into a model. You don’t need to train models on different data types—they’ll always be tensor in, tensor out.

# Tensors Out of a Model

As noted earlier, the power of tensors is in their consistency—regardless of what type of data you pass *into* a model, when they’re tensors, you can be consistent in your coding interface. The same applies for tensors *out* of a model.

So, for example, consider a dataset like `ImageNet` that contains 15 million images in over 21,000 classes. When you design a model to recognize images in this dataset, you’ll need over 21,000 output neurons, each of which gives you a percentage likelihood that the image is of the representative class. So, for example, if neuron 0 represents “goldfish,” the value coming out of it when you do inference will be the probability that the image contains a goldfish!

So, instead of outputting the classification, the model will expose the values of *each* of its output neurons. These values are often called *logits*.

This list of values is a 1D vector of values—so of course, a tensor is the appropriate data type.

Here’s a simulated example, with a set of representative outputs from multiple images passed into the model and the list of class names:

```py
# ImageNet class labels (simplified - just a few examples)
class_names = [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark',
    'electric ray', 'stingray', 'rooster', 'hen', 'ostrich', 'brambling',
    'goldfinch', 'house finch', 'junco', 'indigo bunting', 'robin', 'bulbul',
    'jay', 'magpie', 'chickadee'
]

# Simulate model output for demonstration
# This would normally come from model(input_tensor)
example_output = torch.tensor([
    [ 1.2,  4.5, –0.8,  2.1,  0.3,  # First image predictions
     –1.5,  0.9,  3.2, –0.4,  1.1,
      0.5, –0.2,  1.8,  0.7, –1.0,
      2.8,  1.6, –0.6,  0.4,  1.3],
    [–0.5,  5.2,  0.3,  1.4, –0.8,  # Second image predictions
      0.9,  1.2,  2.8,  0.6,  1.5,
     –1.1,  0.4,  2.1,  0.2, –0.7,
      1.9,  0.8, –0.3,  1.6,  0.5]
])

```

Note that because the output is *tensors*, we can use many of the functions built into PyTorch that are optimized for tensors to work with them.

When dealing with the output values, we want to find the best ones and maybe limit them to a range. Softmax is perfect for that, when it converts the raw output into probabilities—and TopK is used to pick the best *k* values. Here’s an example in which we can use Softmax and TopK functions that manage tensors, regardless of their dimensionality:

```py
def interpret_output(output_tensor, top_k=5):
    # Apply softmax to convert logits to probabilities
    probabilities = torch.nn.functional.softmax(output_tensor, dim=1)

    # Get top k probabilities and class indices
    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    # Convert to numpy for easier handling
    top_probs = top_probs.numpy()
    top_indices = top_indices.numpy()

    return top_probs, top_indices
```

The values can then be converted to NumPy so that other code—like printing out the values—will work more easily.

We can see the output of this here, where Softmax and TopK were used to interpret the data and then print it out:

```py
Image 1 Predictions:
------------------------
Raw logits (first 5): [1.2000000476837158, 4.5, –0.800000011920929, 
                       2.0999999046325684, 0.30000001192092896]

Top 5 Predictions:
1. goldfish: 52.3%
2. rooster: 14.2%
3. robin: 9.5%
4. tiger shark: 4.7%
5. house finch: 3.5%
```

The full code for this can be found in the [GitHub repository](https://github.com/lmoroney/PyTorch-Book-FIles) for this book.

# Summary

In this chapter, you took a brief look at tensors and the underlying idea behind them—that they are a flexible data structure that can be used to represent the best way to put data *into* an ML model, regardless of what it represents, and even batch it. It also provides a consistent way to manage outputs from a model—in which values are typically emitted via neurons that are arranged in the output layer as a list. Thus, by being able to handle tensors, you can build a foundation for how data flows in *and* out of a model.

With that, we’re now going to switch gears from model training to inference, in particular with generative AI, starting with getting models from registries and hubs.