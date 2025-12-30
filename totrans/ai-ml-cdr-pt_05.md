# Chapter 4\. Using Data with PyTorch

In the first three chapters of this book, you trained models using a variety of data, from the Fashion MNIST dataset that was conveniently bundled via an API to the image-based “Horses or Humans” and “Dogs vs. Cats” datasets, which were available as ZIP files that you had to download and preprocess. So by now, you’ve probably realized that there are lots of different ways of getting the data with which to train a model.

However, many public datasets require you to learn lots of different domain-specific skills before you begin to consider your model architecture. The goal behind PyTorch domains and the tools available at the `torch.utils.data.Datasets` namespace is to expose datasets in a way that’s easy to consume, where all the preprocessing steps of acquiring the data and getting it into PyTorch-friendly APIs are done for you.

You’ve already seen a little of this idea in how PyTorch handled Fashion MNIST back in [Chapter 2](ch02.html#ch02_introduction_to_computer_vision_1748548889076080). As a recap, all you had to do to get the data was this:

```py
train_dataset = datasets.FashionMNIST(root='./data', train=True,
                             download=True, transform=transform)

```

In the case of this dataset, we also did an import from the torchvision library to get the datasets object that contained the reference to Fashion MNIST:

```py
from torchvision import datasets
```

Given that it’s a computer vision–oriented dataset, it makes sense that it would be in the torchvision library.

PyTorch has many other datasets of different data types that can be loaded in the same way. These include the following:

Vision

Fashion MNIST is in the aforementioned torchvision library. It’s one of the “Image Classification” built-in datasets, but there are many more for other scenarios like Image Detection, Segmentation, Optical Flow, Stereo Matching, Image Pairing, Image Captioning, Video Classification, Video Predictions, and more.

Text

Common text datasets are available in the torchtext library. There are far too many to list here, but there are ones for Text Classification, Language Modeling, Machine Translation, Sequence Tagging, Question and Answer, and Unsupervised Learning. You can find more details on these in the [PyTorch documentation](https://oreil.ly/aFamN). Note that this library isn’t limited to the datasets; it also has many helper functions that you will use when processing text.

*Audio*

The torchaudio library contains many datasets that can be used in machine learning scenarios for sound or speech. Details can be found in the [PyTorch documentation](https://oreil.ly/tvDe4).

All datasets are subclasses of `torch.utils.data.Dataset,` so it’s important to take a look at this library and understand it well. That will help you not only consume existing datasets but also create your own to share with others.

# Getting Started with Datasets

The `torch.utils.data.Dataset` is an abstract class that represents a dataset. To create a custom dataset, you just need to subclass it and implement these methods:

```py
__len__(self) 
```

This should return the total number of items in your dataset:

```py
__getitem__(self, index) 
```

This should return a single item from your dataset at the specified index. This item will be transformed before sending it to the model.

Here’s an example:

```py
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample
```

That’s pretty much it at a very low level. The data itself is in an array called `data[]`. Now, imagine we want to create a dataset with a linear relationship between an *x* value and a *y* value like we had in [Chapter 1](ch01.html#ch01_introduction_to_pytorch_1748548870019566). How would we use it?

Let’s say we start with some simple synthetic data, like this:

```py
# Generate synthetic data
torch.manual_seed(0)  # For reproducibility
x = torch.arange(0, 100, dtype=torch.float32)
y = 2 * x – 1
```

Then, we could turn it into a dataset like this:

```py
class CustomDataset(Dataset):
    def __init__(self, x, y):
        """
 Initialize the dataset with x and y values.
 Arguments:
 x (torch.Tensor): The input features.
 y (torch.Tensor): The output labels.
 """
        self.x = x
        self.y = y

    def __len__(self):
        """
 Return the total number of samples in the dataset.
 """
        return len(self.x)

    def __getitem__(self, idx):
        """
 Fetch the sample at index `idx` from the dataset.
 Arguments:
 idx (int): The index of the sample to retrieve.
 """
        return self.x[idx], self.y[idx]
```

Then, to use the dataset, we simply create an instance of the class, initialize it with our *x* and *y* values, pass it to a `DataLoader`, and enumerate that:

```py
# Create an instance of CustomDataset
dataset = CustomDataset(x, y)

# Use DataLoader to handle batching and shuffling
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Iterate over the DataLoader
for batch_idx, (inputs, labels) in enumerate(data_loader):
    print(f"Batch {batch_idx+1}")
    print("Inputs:", inputs)
    print("Labels:", labels)
    # Break after the first batch for demonstration
    if batch_idx == 0:
        break
```

With this foundation, you can now explore the dataset classes that have been made available in the various libraries we’ve mentioned since the beginning of this chapter. Given that they will build on or extend this class, the APIs should look familiar.

# Exploring the FashionMNIST Class

Earlier in the book, we saw the `FashionMNIST` class—which provides access to the Fashion-MNIST dataset, which is a training set of 60,000 examples of 10 classes of clothing—and an accompanying test set of 10,000 examples. Each of these examples is a 28 × 28 grayscale image.

In the case of this dataset, you use the *same* class whether you’re using training data or test/validation data, and the data that you receive is based on the `train` parameter that you pass to it. Here’s an example:

```py
# Create the FashionMNIST dataset
fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, 
                                   download=True, transform=transform)
```

When you set `train=True,` the code that overrides the class init method will take the 60,000 records and return them to the caller. Other parameters are there, like specifying the root for where the data should go and even whether or not to download the data. Finally, as you’ll commonly see when downloading data, there is the `transform=` parameter. As you saw in the preceding base class, this parameter will be available as an optional parameter for all datasets, and it will apply a transform when set.

# Generic Dataset Classes

You may need to use some data that isn’t available in the dataset classes, like `FashionMNIST`, but you’ll also want to take advantage of everything in the data ecosystem—such as the ability to transform your data, things like splitting, and all the good stuff in the `DataLoader` class you’ll see later in this chapter. To that end, `torch.utils.data` provides a number of generic dataset classes you could use.

## ImageFolder

In [Chapter 3](ch03.html#ch03_going_beyond_the_basics_detecting_features_in_ima_1748570891074912), we used the “Horses or Humans,” “Rock, Paper, Scissors,” and “Cats vs. Dogs” datasets, which are not available in a class directly but instead as a ZIP file containing the images. When we downloaded them and saved them into subdirectories for the different image types (e.g., one folder for “Horses” and another for “Humans”), the generic `ImageFolder` dataset class could act as a dataset for us.

In that case, the images were streamed (via a `DataLoader`) from the directory according to the batch size and other rules on the `DataLoader`. The labels were derived from the directory names, and the associated class indices were the labels in alphabetical order. So, “Horses” would be class 0, and “Humans” would be class 1\. Please watch out for that when building and debugging because you might miss out on this ordering!

For example, we tend to say, “Rock, Paper, Scissors” in that order, and we would therefore expect them to be classes 0, 1, and 2, respectively. But in *alphabetical* order, Paper would be class 0, Rock would be class 1, and Scissors would be class 2!

###### Tip

One tool to use for this is to create a custom index like the one below:

```py
custom_class_to_idx = {'rabbit': 0, 'dog': 1, 'cat': 2}
dataset = ImageFolder(
  root='data/animals',
  target_transform=
    lambda x: custom_class_to_idx[dataset.classes[x]]
)
dataset.class_to_idx = custom_class_to_idx
print(dataset.class_to_idx)
```

## DatasetFolder

`ImageFolder` is actually a subclass of the more generic `DatasetFolder` class, one that’s customized for images. The `DatasetFolder` class isn’t limited to image data, and you can use it for anything. It also allows you to use directories for labels. So, for example, say you have text files that contain text of different classes with a directory structure like this:

```py
root/sarcasm/document1.txt
root/sarcasm/document2.txt
root/sarcasm/document3.txt
root/factual/factdoc1.rtf
root/factual/factdoc2.doc
```

You could then use a `DatasetFolder` to stream the documents according to the correct labels. Also, because this class is document based, you could apply a transform to extract from the file!

## FakeData

`FakeData` is a useful generic dataset that, as its name suggests, provides you with fake data. At the time of writing, it only supports creating fake image data. It’s also very useful if you don’t have data on hand but want to experiment with different architectures, or if you want to benchmark your system.

You can use `FakeData` in the same way you’d use any of the datasets you’ve seen so far in this book. So, for example, if you wanted to create a set of `FakeData` for the MobileNet model, which uses 224 × 224 color images, you’d do it with code like this:

```py
import torch
from torchvision.datasets import FakeData
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define transformations (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

# Create FakeData
fake_dataset = FakeData(size=100, image_size=(3, 224, 224), 
                        num_classes=10, transform=transform)

# DataLoader
data_loader = DataLoader(fake_dataset, batch_size=10, shuffle=True)
```

This would create 100 images (containing just noise) of the desired size and span them across 10 classes. You could then use this data in a `DataLoader` in the same way you’d use any other dataset.

While `FakeData` only gives image types, you could relatively easily create your own CustomData (as we looked at earlier) to provide fake data in other formats, such as numeric or sequence data.

# Using Custom Splits

Up to this point, all of the data you’ve been using to build models has been pre-split for you into training and test sets. For example, with Fashion MNIST, you had 60,000 and 10,000 records, respectively. But what if you don’t want to use those splits? What if you want to split the data yourself, according to your own needs?

Thankfully, when using datasets, you can generally do this with an easy and intuitive API.

So, for example, when you loaded the `FashionMNIST` class previously, you specified the `train` parameter to get it to give you the training data (60,000 records) or the test data (10,000 records).

To override this, you simply ignore it, and you’ll get all of the data:

```py
# Load the entire Fashion-MNIST dataset
dataset = datasets.FashionMNIST(root='./data', 
                                download=True, transform=transform)
```

To create your own split, you can use the `torch.utils.data` namespace, which contains a function called `random_split`. So, for example, if you want to have a validation set that `FashionMNIST` doesn’t provide, you can divide the dataset into three datasets with `random_split`. Here’s the code that will assign 70% of the data to a training set, 15% to a testing set, and 15% to a validation set:

```py
from torch.utils.data import random_split

total_count = len(dataset)
train_count = int(0.7 * total_count)
val_count = int(0.15 * total_count)

# Ensures all data is used
test_count = total_count – train_count – val_count  

train_dataset, val_dataset, test_dataset = 
     random_split(dataset, [train_count, val_count, test_count])
```

As you can see, this process is pretty straightforward. We get the number of records in the dataset as `total_count` and then calculate 70% of them (0.7 times the total count) to be the training count and 15% of them to be the validation count. When making calculations like this, you may end up with rounding errors that leave some records out—so instead of using 15% for the test count, you can just set the quotient for training to be the total minus the training and validation records. This will ensure all the data is used and none is wasted.

What’s really nice about this approach is that it gives you a really simple way to get new and different slices of your dataset. As you train models, it gives you a new way to evaluate them for accuracy.

For example, one slice of the dataset may train at high accuracy while another does so at low accuracy, indicating that there are likely issues in your model architecture that make it overfit on one dataset. On the other hand, if you try multiple different splits of your data and the model training and validation results are consistent, then you’ll have a signal that your architecture is sound.

I would definitely encourage you to use custom splits when training models, as it can really help you get over some gotchas!

One more thing to consider when using custom splits is that the name `random` doesn’t mean that this approach *shuffles* or randomizes your dataset. It merely slices the dataset at random points to give you different slices each time. Should you want to also shuffle the dataset, you can do it in the DataLoader, which we’ll explore in the next section.

# The ETL Process for Managing Data in Machine Learning

*Extract, Transfer, Load* (ETL) is the core pattern for training ML models, regardless of scale. We’ve been exploring small-scale, single-computer model building in this book, but we can use the same technology for large-scale training across multiple machines with massive datasets.

The Extract, Transfer, Load process consists of the three phases that are in the process’s name:

Extract phase

This occurs when the raw data is loaded from wherever it is stored and prepared in a way that can be transformed.

Transform phase

This occurs when the data is manipulated in a way that makes it suitable or improved for training. For example, batching, image augmentation, mapping to feature columns, and other such logic applied to the data can be considered part of this phase.

Load phase

This occurs when the data is loaded into the neural network for training.

Consider the code from [Chapter 3](ch03.html#ch03_going_beyond_the_basics_detecting_features_in_ima_1748570891074912) that we used to train the “Horses or Humans” classifier. At the top of the code, you saw a chunk like this:

```py
# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(
        degrees=0,  # No rotation
        translate=(0.2, 0.2),  # Translate up to 20% x and y
        scale=(0.8, 1.2),  # Zoom in or out by 20%
        shear=20,  # Shear by up to 20 degrees
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load the datasets
train_dataset = datasets.ImageFolder(root=training_dir, 
                                     transform=train_transform)
val_dataset = datasets.ImageFolder(root=validation_dir, 
                                   transform=train_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
```

This is the ETL pattern embodied in code!

The code begins by defining the `transform` (the “T”), but the active code doesn’t begin until the lines under the `Load the datasets` comment. Look carefully here and you’ll see that the `ImageFolder` is being used to *extract* the data from its location at rest on disk.

Then, as the data is extracted, the `transform` that we defined is applied.

Then, under the `Data loaders` comment, we perform the *load* of the data using the `train_loader` and `val_loader` we’ve defined. Strictly speaking, the actual loading doesn’t take place until we execute the training loop to pull the data out of the loaders.

It’s important to know that using this process can make your data pipelines less susceptible to changes in the data and the underlying schema. When you use this approach to extract data, the same underlying structure is used regardless of whether the data is small enough to fit in memory or large enough that it cannot be contained even on a simple machine. The APIs for applying the transformation are also consistent, so you can use similar ones regardless of the underlying data source. And of course, once the data is transformed, the process of loading the data is also consistent, regardless of your training backend.

However, how you load the data can have a huge impact on your training speed. Let’s take a look at that next.

# Optimizing the Load Phase

Let’s take a closer look at the ETL process when you’re training a model. We can consider the extraction and transformation of the data to be possible on any processor, including a CPU. In fact, the code you use in these phases to perform tasks like downloading data, unzipping it, and going through it record by record and processing those records is not what GPUs and TPUs are built for, so the code will likely execute on the CPU anyway. When it comes to training, however, you can get great benefits from a GPU or TPU, so it makes sense for you to use one for this phase if possible. Thus, in a situation where a GPU or TPU is available to you, you should ideally split the workload between the CPU and the GPU/TPU, with Extract and Transform taking place on the CPU and Load taking place on the GPU/TPU.

If you explore the code that you’ve been using in this book, you’ll notice that we’ve used the .`to(device)` methodology. Whenever you’re dealing with training or inference and you want the data or model to be on the accelerator, you’ll see something like .`to(“cuda”)`, but for the extraction and transform, you won’t see it because it would be a waste of the GPU’s resources.

Suppose you’re working with a large dataset. Assuming it’s so large that you have to prepare the data (i.e., do the extraction and transformation) in batches, you’ll end up with a situation like that shown in [Figure 4-1](#ch04_figure_1_1748548966489345). While the first batch is being prepared, the GPU or TPU is idle. Then, when that batch is ready, you can send it to the GPU/TPU for training, but then the CPU will be idle until the training is done and it can start preparing the second batch. There’s a lot of idle time here, so we can see that there’s room for optimization.

![Training on a CPU/GPU](assets/aiml_0401.png)

###### Figure 4-1\. Training on a CPU or GPU/TPU

The logical solution is to do the work in parallel, preparing and training side by side. This process is called *pipelining* and is illustrated in [Figure 4-2](#ch04_figure_2_1748548966489381).

![](assets/aiml_0402.png)

###### Figure 4-2\. Pipelining

In this case, while the CPU is preparing the first batch, the GPU/TPU again has nothing to work on, so it’s idle. When the first batch is done, the GPU/TPU can start training—but in parallel with this, the CPU will prepare the second batch. Of course, the time it takes to train batch *n* – 1 and prepare batch *n* won’t always be the same, and if the training time is shorter, you’ll have periods of idle time on the GPU/TPU, and if the training time is longer, you’ll have periods of idle time on the CPU. Choosing the correct batch size can help you optimize here—and as GPU/TPU time is likely more expensive, you’ll probably want to reduce its idle time as much as possible.

This is one of the reasons why we use batching, even for the simple examples like MNIST: the pipelining model is in place so that regardless of how large your dataset is, you’ll continue to use a consistent pattern for ETL on it.

# Using the DataLoader Class

We’ve seen the `DataLoader` class many times already, but it’s good to take a slightly deeper look at it now to help you get the most out of it in your ML workflows. It provides the following features that you can use.

## Batching

Intuitively, you might think that the forward pass works one data item at a time. You *could* do that, but some optimizers, like stochastic gradient descent, do much better when inputs are passed in batches so that they can calculate more accurately. Batching can also speed up your training in larger scenarios, where you are using GPUs with fixed memory sizes. It’s most efficient to maximize the use of that memory by having a batch of data that fits it fully. If you’re using a `DataLoader`, batching is simply a matter of setting a parameter.

## Shuffling

Shuffling the data is very important, particularly when you do batching. Consider the following scenario with something like Fashion MNIST.

You have 60,000 samples each for 10 classes, and they are not shuffled. You batch one thousand records at a time, so your first batch of one thousand is all for class 0, the second batch is all for class 1, and so on. In this scenario, the model may not effectively learn because each batch is biased toward a particular label—but the ability of your model to generalize will improve if the batches are shuffled, meaning your first thousand items will have varied labels, etc.

## Parallel Data Loading

Often, and in particular with complex data, loading data into the model for the forward pass can be time-consuming. But the `DataLoader` class offers parallelism through Python’s multiprocessing model, which can significantly speed this up.

As you saw previously, you should consider your data loading/transformation and your model learning to be two separate processes. You want to avoid scenarios where the model training has no data to work with and is sitting idle, waiting for data to be loaded, and you also want to avoid scenarios where you have tons of data piled up in memory but the model can’t get to it. Parallel data loading, when well tuned, can be helpful here, and there’s a skill you can learn to ensure that you’re getting the most out of your training by running this at peak efficiency. You’ll learn how to do this in the next section.

## Custom Data Sampling

In addition to shuffling for random data sampling, you can create custom data sampling, in which you can specify how the data will be loaded. The `torch.utils.data.Sampler` class provides a base class that you can build a custom sampler on. This process is beyond the scope of this book, but there are many excellent examples of it online.

# Parallelizing ETL to Improve Training Performance

If you’re using the `DataLoader` class, you can easily perform parallelization by using the `num_workers` parameter. So, for example, say you want to train a model on the `CIFAR10` dataset, and you want to use parallel training. Let’s take a look at how to do this, step-by-step.

First, we’ll explore the Extract and Transform steps:

```py
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 dataset
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
```

Then, we’ll configure and create the DataLoader to the Load step:

```py
from torch.utils.data import DataLoader

# DataLoader with multiple workers
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, 
                         `num_workers``=``4`)
```

Note the `num_workers=4` parameter, which will create four subprocesses to load the data in parallel simultaneously. Based on the hardware you have available and the number of cores, the speed of your CPU, etc., you can experiment with this number to reduce the overall bottlenecks.

What’s really nice about this approach is that the ETL process is neatly encapsulated in it, so your model training loop doesn’t have to change in any way, even though you’re loading the data by using parallelism! Here’s the code for the simple `CIFAR` model that uses this data:

```py
import torch

# Dummy model and optimizer setup
model = torch.nn.Sequential(
    torch.nn.Linear(3 * 32 * 32, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train(model, data_loader):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Reshape inputs to match the model's expected input
        inputs = inputs.view(inputs.size(0), –1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {batch_idx} Loss: {loss.item()}")

train(model, data_loader)
```

Parallelizing is another tool that’s available to you when you’re training your models. There’s no one-size-fits-all approach, but it’s a good tool to use when you experience training slowdowns. It’s easy to assume that training operates slowly just because of the network architecture, but you may be surprised at how much of that time is wasted by the forward pass waiting for new data! By adding this type of parallelism, you have the potential to greatly speed up training.

# Summary

This chapter covered the data ecosystem in PyTorch and introduced you to the `dataset` and `DataLoader` classes. You saw how they use a common API and a common format to help reduce the amount of code you have to write to get access to data, and you also saw how to use the ETL process, which is at the heart of the common design patterns in training models with PyTorch. In particular, we explored parallelizing the extraction, transformation, and loading of data to improve training performance.

So now that you’ve had a chance to look at the process, see if you can create your own dataset! Maybe do it from some photos in your albums, or some test, or just random noise like we did here.

In the next chapter, you’ll take what you’ve learned so far and start applying it to natural language processing problems.