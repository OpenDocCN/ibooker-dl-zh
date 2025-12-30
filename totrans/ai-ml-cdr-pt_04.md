# Chapter 3\. Going Beyond the Basics: Detecting Features in Images

In [Chapter 2](ch02.html#ch02_introduction_to_computer_vision_1748548889076080), you learned how to get started with computer vision by creating a simple neural network that matched the input pixels of the Fashion MNIST dataset to 10 labels, each of which represented a type (or class) of clothing. And while you created a network that was pretty good at detecting clothing types, there was a clear drawback. Your neural network was trained on small monochrome images, each of which contained only a single item of clothing, and each item was centered within the image.

To take the model to the next level, you need it to be able to detect *features* in images. So, for example, instead of looking merely at the raw pixels in the image, what if we could filter the images down to constituent elements? Matching those elements, instead of raw pixels, would help the model detect the contents of images more effectively. For example, consider the Fashion MNIST dataset that we used in the last chapter. When detecting a shoe, the neural network may have been activated by lots of dark pixels clustered at the bottom of the image, which it would see as the sole of the shoe. But if the shoe were not centered and filling the frame, this logic wouldn’t hold.

One method of detecting features comes from photography and image processing methodologies that you may already be familiar with. If you’ve ever used a tool like Photoshop or GIMP to sharpen an image, you’ve used a mathematical filter that works on the pixels of the image. Another word for what these filters do is *convolution*, and by using such filters in a neural network, you’ll create a *convolutional neural network* (CNN).

In this chapter, you’ll start by learning about how to use convolutions to detect features in an image. Then, you’ll dig deeper into classifying images based on the features within. We’ll also explore augmentation of images to get more features and transfer learning to take preexisting features that were learned by others, and then we’ll look briefly into optimizing your models by using dropouts.

# Convolutions

A *convolution* is simply a filter of weights that are used to multiply a pixel by its neighbors to get a new value for the pixel. For example, consider the ankle boot image from Fashion MNIST and the pixel values for it (see [Figure 3-1](#ch03_figure_1_1748570891059985)).

![](assets/aiml_0301.png)

###### Figure 3-1\. Ankle boot with convolution

If we look at the pixel in the middle of the selection, we can see that it has the value 192\. (Recall that Fashion MNIST uses monochrome images with pixel values from 0 to 255.) The pixel above and to the left has the value 0, the one immediately above has the value 64, etc.

If we then define a filter in the same 3 × 3 grid, as shown below the original values, we can transform that pixel by calculating a new value for it. We do this by multiplying the current value of each pixel in the grid by the value in the same position in the filter grid and then summing up the total amount. This total will be the new value for the current pixel, and we then repeat this calculation for all pixels in the image.

So, in this case, while the current value of the pixel in the center of the selection is 192, we calculate the new value after applying the filter as follows:

```py
new_val = (–1 * 0) + (0 * 64) + (–2 * 128) + 
     (.5 * 48) + (4.5 * 192) + (–1.5 * 144) + 
     (1.5 * 142) + (2 * 226) + (–3 * 168)
```

The result equals 577, which will be the new value for the pixel. Repeating this process for every pixel in the image will give us a filtered image.

Now, let’s consider the impact of applying a filter on a more complicated image: specifically, the [ascent image](https://oreil.ly/wP8TE) that’s built into SciPy for easy testing. This is a 512 × 512 grayscale image that shows two people climbing a staircase.

Using a filter with negative values on the left, positive values on the right, and zeros in the middle will end up removing most of the information from the image except for vertical lines (see [Figure 3-2](#ch03_figure_2_1748570891060023)).

![](assets/aiml_0302.png)

###### Figure 3-2\. Using a filter to derive vertical lines

Similarly, a small change to the filter can emphasize the horizontal lines (see [Figure 3-3](#ch03_figure_3_1748570891060045)).

![](assets/aiml_0303.png)

###### Figure 3-3\. Using a filter to derive horizontal lines

These examples also show that the amount of information in the image is reduced. Therefore, we can potentially *learn* a set of filters that reduce the image to features, and those features can be matched to labels as before. Previously, we learned parameters that were used in neurons to match inputs to outputs, and similarly, we can learn the best filters to match inputs to outputs over time.

When we combine convolution with pooling, we can reduce the amount of information in the image while maintaining the features. We’ll explore that next.

# Pooling

*Pooling* is the process of eliminating pixels in your image while maintaining the semantics of the content within the image. It’s best explained visually. [Figure 3-4](#ch03_figure_4_1748570891060063) depicts the concept of *max pooling*.

![](assets/aiml_0304.png)

###### Figure 3-4\. An example of max pooling

In this case, consider the box on the left to be the pixels in a monochrome image. We group them into 2 × 2 arrays, so in this case, the 16 pixels are grouped into four 2 × 2 arrays. These arrays are called *pools*.

Then, we select the *maximum* value in each of the groups and reassemble them into a new image. Thus, the pixels on the left are reduced by 75% (from 16 to 4), with the maximum value from each pool making up the new image. [Figure 3-5](#ch03_figure_5_1748570891060080) shows the version of ascent from [Figure 3-2](#ch03_figure_2_1748570891060023), with the vertical lines enhanced, after max pooling has been applied.

![](assets/aiml_0305.png)

###### Figure 3-5\. Ascent after applying vertical filter and max pooling

Note how the filtered features have not just been maintained but have been further enhanced. Also, the image size has changed from 512 × 512 to 256 × 256—making it a quarter of the original size.

###### Note

There are other approaches to pooling. These include *min pooling*, which takes the smallest pixel value from the pool, and *average pooling*, which takes the overall average value from the pool.

# Implementing Convolutional Neural Networks

In [Chapter 2](ch02.html#ch02_introduction_to_computer_vision_1748548889076080) you created a neural network that recognized fashion images. For convenience, here’s the code to define the model:

```py
# Define the model
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = FashionMNISTModel()

# Define the loss function and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  
                    [{current:>5d}/{size:>5d}]")

# Training process
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_function, optimizer)
print("Done!")
```

To convert this to a CNN, you simply use convolutional layers in our model definition on top of the current linear ones. You’ll also add pooling layers.

To implement a convolutional layer, you’ll use the `nn.Conv2D` type. It accepts as parameters the number of convolutions to use in the layer, the size of the convolutions, the activation function, etc.

For example, here’s a convolutional layer that uses this type:

```py
nn.Conv2d(1, 64, kernel_size=3, padding=1)
```

In this case, we want the layer to learn `64` convolutions. It will randomly initialize them, and over time, it will learn the filter values that work best to match the input values to their labels. The `kernel_size = 3` indicates the size of the filter. Earlier, we showed you 3 × 3 filters, and that’s what we’re specifying here. The 3 × 3 filter is the most common size of filter. You can change it as you see fit, but you’ll typically see an odd number of axes like 5 × 5 or 7 × 7 because of how filters remove pixels from the borders of the image, as you’ll see later.

Here’s how to use a pooling layer in the neural network. You’ll typically do this immediately after the convolutional layer:

```py
nn.MaxPool2d(kernel_size=2, stride=2)
```

In the example back in [Figure 3-4](#ch03_figure_4_1748570891060063), we split the image into 2 × 2 pools and picked the maximum value in each. However, we could have used the parameters that you see here to define the pool size. The `kernel_size=2` parameter indicates that our pools are 2 × 2, and the `stride=2` parameter means that the filter will jump over two pixels to get the next pool.

Now, let’s explore the full code to define a model for Fashion MNIST with a CNN:

```py
# Define the CNN model
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))  # Output: 64 x 6 x 6

        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), –1)  # Flatten the output
        out = self.fc1(out)
        out = self.fc2(out)
        return out
```

Here, we see that the class has two functions, one for initialization and one that will be called during the forward pass in each epoch during training.

The `init` simply defines what each of the layers in our neural network will look like. The first layer (`self.layer1`) will take in the one-dimensional input, have `64` convolutions, a `kernel_size` of `3`, and `padding` of `1`. It will then ReLU the output before max pooling it.

The next layer (`self.layer2`) will take the 64 convolutions of output from the previous layer and then output `64` of its own before ReLUing them and max pooling them. Its output will now be `64 × 6 × 6` because the `MaxPool` halves the size of the image.

The data is then fed to the next layer (`self.fc1`, where `fc` stands for *fully connected*), with the input being the shape of the output of the previous layer. The output is 128, which is the same number of neurons we used in [Chapter 2](ch02.html#ch02_introduction_to_computer_vision_1748548889076080) for the deep neural network (DNN).

Finally, these 128 are fed into the final layer (`self.fc1`) with 10 outputs—that represent the 10 classes.

###### Note

In the DNN, we ran the input through a `Flatten` layer prior to feeding it into the first `Dense` layer. We’ve lost that in the input layer here—instead, we’ve just specified the 1-D input shape. Note that prior to the first `Linear` layer, after convolutions and pooling, the data will be flattened.

Then, we stack these layers in the `forward` function. We can see that we get the data `x` and pass it through `layer1` to get `out`, which is passed to `layer2` to get a new `out`. At this point, we have the convolutions that we’ve learned, but we need to flatten them before loading them into the `Linear` layers `fc1` and `fc2`. The `out = out.view(out.size(0), -1)` achieves this.

If we train this network on the same data for the same 50 epochs as we used when training the network shown in [Chapter 2](ch02.html#ch02_introduction_to_computer_vision_1748548889076080), we will see that it works nicely. We can get to 91% accuracy on the test set quite easily:

```py
Train Epoch: 44 -- Loss: 0.091689
Train Epoch: 45 -- Loss: 0.066864
Train Epoch: 46 -- Loss: 0.061322
Train Epoch: 47 -- Loss: 0.056557
Train Epoch: 48 -- Loss: 0.039695
Train Epoch: 49 -- Loss: 0.056213
Accuracy of the network on the 10000 test images: 91.31%
```

So, we can see that adding convolutions to the neural network definitely increases its ability to classify images. Next, let’s take a look at the journey an image takes through the network so we can get a little bit more of an understanding of why this process works.

###### Note

If you are using the accompanying code from my GitHub, you’ll notice that I’m using model.to(device) a lot. In PyTorch, if an accelerator is available, you can request that the model and/or its data use the accelerator with this command.

# Exploring the Convolutional Network

With the torchsummary library, you can inspect your model. When you run it on the Fashion MNIST convolutional network we’ve been working on, you’ll see something like this:

```py
from torchsummary import summary
model = FashionCNN().to(device) 
summary(model, input_size=(1, 28, 28))  # (Channels, Height, Width)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [–1, 64, 28, 28]             640
              ReLU-2           [–1, 64, 28, 28]               0
         MaxPool2d-3           [–1, 64, 14, 14]               0
            Conv2d-4           [–1, 64, 12, 12]          36,928
              ReLU-5           [–1, 64, 12, 12]               0
         MaxPool2d-6             [–1, 64, 6, 6]               0
            Linear-7                  [–1, 128]         295,040
            Linear-8                   [–1, 10]           1,290
================================================================
Total params: 333,898
Trainable params: 333,898
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.02
Params size (MB): 1.27
Estimated Total Size (MB): 2.30
```

Let’s first take a look at the `Output Shape` column to get an understanding of what’s going on here. Our first layer will have 28 × 28 images and apply 64 filters to them. But because our filter is 3 × 3, a one-pixel border around the image would typically be lost, reducing our overall information to 26 × 26 pixels. However, because we used the `padding=1` parameter, the image was artificially inflated to 30 × 30, meaning that its output would be the correct 28 × 28 and no information would be lost.

If you don’t pad the image, you’ll end up with a result like the one in [Figure 3-6](#ch03_figure_6_1748570891060097). If we take each of the boxes as a pixel in the image, the first possible filter we can use starts in the second row and the second column. The same would happen on the right side and at the bottom of the diagram.

![](assets/aiml_0306.png)

###### Figure 3-6\. Losing pixels when running a filter

Thus, an image that is *a* × *b* pixels in shape when run through a 3 × 3 filter will become (*a* – 2) × (*b* – 2) pixels in shape. Similarly, a 5 × 5 filter would make it (*a* – 4) × (*b* – 4), and so on. As we’re using a 28 × 28 image and a 3 × 3 filter, our output would now be 26 × 26\. But because we padded the image up to 30 × 30 (again, to prevent loss of information), the output is now 28 × 28.

After that, the pooling layer will be 2 × 2, so the size of the image will halve on each axis, and it will then become 14 × 14\. The next convolutional layer does *not* use padding, so it will reduce this further to 12 × 12, and the next pooling will output 6 x 6.

So, by the time the image has gone through two convolutional layers, the result will be many 6 × 6 images. How many? We can see that in the `Param #` (number of parameters) column.

Each convolution is a 3 × 3 filter, plus a bias. Remember earlier, with our dense layers, when each layer was *y* = *wx* + *b*, where *w* was our parameter (aka weight) and *b* was our bias? This case is very similar, except that because the filter is 3 × 3, there are 9 parameters to learn. Given that we have 64 convolutions defined, we’ll have 640 overall parameters. (Each convolution has 9 parameters plus a bias, for a total of 10, and there are 64 of them.)

The `ReLU and MaxPooling` layers don’t learn anything; they just reduce the image, so there are no learned parameters there—hence, 0 are reported.

The next convolutional layer has 64 filters, but each of them is multiplied across the *previous* 64 filters, each of which has 9 parameters. We have a bias on each of the new 64 filters, so our number of parameters should be (64 × (64 × 9)) + 64, which gives us 36,928 parameters the network needs to learn.

If this is confusing, try changing the number of convolutions in the first layer to something else—for example, 10\. You’ll see that the number of parameters in the second layer becomes 5,824, which is (64 × (10 × 9)) + 64).

By the time we get through the second convolution, our images are 6 × 6, and we have 64 of them. If we multiply this out, we’ll have 1,600 values, which we’ll feed into a dense layer of 128 neurons. Each neuron has a weight and a bias, and we’ll have 128 of them, so the number of parameters the network will learn is ((6 × 6 × 64) × 128) + 128, giving us 295,040 parameters.

Then, our final dense layer of 10 neurons will take in the output of the previous 128, so the number of parameters learned will be (128 × 10) + 10, which is 1,290.

The total number of parameters will be the sum of all of these: 333,898.

Training this network requires us to learn the best set of these 333,898 parameters to match the input images to their labels. It’s a slower process because there are more parameters, but as we can see from the results, it also builds a more accurate model!

Of course, with this dataset, we still have the limitation that the images are 28 × 28, monochrome, and centered. So next we’ll take a look at using convolutions to explore a more complex dataset comprising color pictures of horses and humans, and we’ll try to make the model determine whether an image contains one or the other. In this case, the subject won’t always be centered in the image like with Fashion MNIST, so we’ll have to rely on convolutions to spot distinguishing features.

# Building a CNN to Distinguish Between Horses and Humans

In this section, we’ll explore a more complex scenario than the Fashion MNIST classifier. We’ll extend what we’ve learned about convolutions and CNNs to try to classify the contents of images in which the location of a feature isn’t always in the same place. I’ve created the “Horses or Humans” dataset for this purpose.

## The “Horses or Humans” Dataset

[The dataset for this section](https://oreil.ly/8VXwy) contains over a thousand 300 × 300–pixel images. Approximately half the images are of horses, and the other half are of humans—and all are rendered in different poses. You can see some examples in [Figure 3-7](#ch03_figure_7_1748570891060112).

![](assets/aiml_0307.png)

###### Figure 3-7\. Horses and humans

As you can see, the subjects have different orientations and poses, and the image composition varies. Consider the two horses, for example—their heads are oriented differently, and one image is zoomed out (showing the complete animal), while the other is zoomed in (showing just the head and part of the body). Similarly, the humans are lit differently, have different skin tones, and are posed differently. The man has his hands on his hips, while the woman has hers outstretched. The images also contain backgrounds such as trees and beaches, so a classifier will have to determine which parts of the image are the important features that determine what makes a horse a horse and a human a human, without being affected by the background.

While the previous examples of predicting *y* = 2*x* – 1 or classifying small monochrome images of clothing *might* have been possible with traditional coding, it’s clear that this example is far more difficult and that you are crossing the line into where ML is essential to solve a problem.

An interesting side note is that these images are all computer generated. The theory is that features spotted in a CGI image of a horse should apply to a real image, and you’ll see how well this works later in this chapter.

## Handling the Data

The Fashion MNIST dataset that you’ve been using up to this point comes with labels, and every image file has an associated file with the label details. Many image-based datasets do not have this, and “Horses or Humans” is no exception. Instead of labels, the images are sorted into subdirectories of each type, and with the DataLoader in PyTorch, you can use this structure to *automatically* assign labels to images.

First, you simply need to ensure that your directory structure has a set of named subdirectories, with each subdirectory being a label. For example, the “Horses or Humans” dataset is available as a set of ZIP files, one of which contains the training data (1,000+ images) and another of which contains the validation data (256 images). When you download and unpack them into a local directory for training and validation, you need to ensure that they are in a file structure like the one in [Figure 3-8](#ch03_figure_8_1748570891060128).

Here’s the code to get the training data and extract it into the appropriately named subdirectories, as shown in [Figure 3-8](#ch03_figure_8_1748570891060128):

```py
import urllib.request
import zipfile

url = "https://storage.googleapis.com/learning-datasets/
                                            horse-or-human.zip"
file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'
urllib.request.urlretrieve(url, file_name)

zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()
```

![](assets/aiml_0308.png)

###### Figure 3-8\. Ensuring that images are in named subdirectories

This code simply downloads the ZIP of the training data and unzips it into a directory at *horse-or-human/training*. (We’ll deal with downloading the validation data shortly.) This is the parent directory that will contain subdirectories for the image types.

Now, to use the `DataLoader`, we simply use the following code:

```py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the datasets
train_dataset = datasets.ImageFolder(root=training_dir, 
                                     transform=transform)
val_dataset = datasets.ImageFolder(root=validation_dir, 
                                   transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

First, we create an instance of a `transforms` object that we’ll call `transform`. This will determine the rules for how we modify the images. It resizes the image to 150 × 150 and then normalizes it into a tensor. Note that the raw images are actually 300 × 300, but to make training quicker for the purposes of learning, I’ve resized them to 150 × 150.

Then, we specify the `dataset` objects to be `datasets.ImageFolder` types and point them to the required directory, and that will generate images for the training process by flowing them from that directory while applying the transform. The directory for training is `training_dir`, and the directory for validation is `validation_dir`, as specified earlier.

## CNN Architecture for “Horses or Humans”

There are several major differences between this dataset and the Fashion MNIST one, and you have to take them into account when designing an architecture for classifying the images. First, the images are much larger—150 × 150 pixels—so more layers may be needed. Second, the images are in full color, not grayscale, so each image will have three channels instead of one. Third, there are only two image types, so we can actually classify them with only *one* output neuron. To do this, we’ll drive the value of that neuron toward 0 for one of the labels and toward 1 for the other. The `sigmoid` function is ideal for this process of driving the value to one of these extremes. You can see this at the bottom of the `forward` function:

```py
class HorsesHumansCNN(nn.Module):
    def __init__(self):
        super(HorsesHumansCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 18 * 18, 512)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 1)  
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(–1, 64 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)  # Use sigmoid to output probabilities
        return x
```

There are a number of things to note here. First of all, take a look at the very first layer. We’re defining 16 filters, each of which has a `kernel_size` of 3, but the input shape is 3\. Remember that this is because our input image is in color: there are three channels, instead of just one for the monochrome Fashion MNIST dataset we were using earlier.

At the other end, notice that there’s only one neuron in the output layer. This is because we’re using a binary classifier, and we can get a binary classification with just a single neuron if we activate it with a sigmoid function. The purpose of the sigmoid function is to drive one set of values toward 0 and the other toward 1, which is perfect for binary classification.

Next, notice how we stack several more convolutional layers. We do this because our image source is quite large and we want, over time, to have many smaller images, each with features highlighted. If we take a look at the results of a `summary`, we’ll see this in action:

```py
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [–1, 16, 150, 150]             448
         MaxPool2d-2           [–1, 16, 75, 75]               0
            Conv2d-3           [–1, 32, 75, 75]           4,640
         MaxPool2d-4           [–1, 32, 37, 37]               0
            Conv2d-5           [–1, 64, 37, 37]          18,496
         MaxPool2d-6           [–1, 64, 18, 18]               0
            Linear-7                  [–1, 512]      10,617,344
           Dropout-8                  [–1, 512]               0
            Linear-9                    [–1, 1]             513
================================================================
Total params: 10,641,441
Trainable params: 10,641,441
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.26
Forward/backward pass size (MB): 5.98
Params size (MB): 40.59
Estimated Total Size (MB): 46.83
----------------------------------------------------------------
```

Note that by the time the data has gone through all the convolutional and pooling layers, it ends up as 18 × 18 items. The theory is that these will be activated feature maps that are relatively simple because they will contain just 324 pixels. We can then pass these feature maps to the dense neural network to match them to the appropriate labels.

This, of course, leads this network to have many more parameters than the previous network, so it will be slower to train. With this architecture, we’re going to learn over 10 million parameters.

###### Tip

The code in this section, as well as in many other places in this book, may require you to import Python libraries. To find the correct imports, you can check out [the book’s repository](https://github.com/lmoroney/PyTorch-Book-FIles).

To train the network, we’ll have to compile it with a loss function and an optimizer. In this case, the loss function can be the `BCELoss,` where `BCE` stands for *binary cross entropy*. As the name suggests, because there are only two classes in this scenario, this is a loss function that is designed for it. For the `optimizer`, we can continue using the same `Adam` that we used earlier. Here’s the code:

```py
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

We then train in the usual way:

```py
def train_model(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()  
            optimizer.zero_grad()
            outputs = model(images).view(–1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
```

One thing to note is that the labels are converted to `floats` because of the binary cross entropy, where the value of the final output node will be a float value.

Over just 15 epochs, this architecture gives us a very impressive 95%+ accuracy on the training set. Of course, this is just with the training data, and this performance isn’t an indication of the network’s potential performance on data that it hasn’t previously seen.

Next, we’ll look at adding the validation set and measuring its performance to give us a good indication of how this model might perform in real life.

## Adding Validation to the “Horses or Humans” Dataset

To add validation, you’ll need a validation dataset that’s separate from the training one. In some cases, you’ll get a master dataset that you have to split yourself, but in the case of “Horses or Humans,” there’s a separate validation set that you can download. In the preceding code snippet, you’ve already downloaded the training and validation datasets, put them in directories, and set up data loaders for each of them. However, for training, you only used one of these datasets—the one that was set up to load the training data. So next, we’ll switch the model into evaluation mode and explore how well it did with the validation data.

###### Note

You may be wondering why we’re talking about a validation dataset here, rather than a test dataset, and whether the two are the same thing. For simple models like the ones developed in the previous chapters, it’s often sufficient to split the dataset into two parts: one for training and one for testing. But for more complex models like the one we’re building here, you’ll want to create separate validation and test sets.

What’s the difference? *Training data* is the data that is used to teach the network how the data and labels fit together, while *validation data* is used to see how the network is doing with previously unseen data *while* you are training (i.e., it isn’t used to fit data to labels but to inspect how well the fitting is going). Also, *test data* is used after training to evaluate how the network does with data it has never previously seen. Some datasets come with a three-way split, and in other cases, you’ll want to separate the test set into two parts for validation and testing. Here, you’ll download some additional images for testing the model.

To download the validation set and unzip it into a different directory, you can use code that’s very similar to that used for the training images.

Then, to perform the validation, you simply update your `train_model` method to perform a validation at the end of each training loop (or epoch) and report on the results. For example, you can do this:

```py
def train_model(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()  
            optimizer.zero_grad()
            outputs = model(images).view(–1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / 
                len(train_loader)}')

    # Evaluate on training set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), 
                                 labels.to(device).float()
                outputs = model(images).view(–1)
                predicted = outputs > 0.5  # Threshold predictions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Test Set Accuracy: {100 * correct / total}%')

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to(device), 
                                 labels.to(device).float()
                outputs = model(images).view(–1)
                predicted = outputs > 0.5  # Threshold predictions
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Validation Set Accuracy: {100 * correct / total}%')    

train_model(50)
```

I added code here to do *both* the training and the validation and report on accuracy. Note that this is really just for learning purposes, so you can compare. In a real-world scenario, checking the accuracy of training data is a waste of processing time!

After training for 10 epochs, you should see that your model is 99%+ accurate on the training set but only about 88% on the validation set:

```py
Epoch 7, Loss: 0.0016404045829699512
Training Set Accuracy: 100.0%
Validation Set Accuracy: 88.28125%
Epoch 8, Loss: 0.0010613293736610378
Training Set Accuracy: 100.0%
Validation Set Accuracy: 89.0625%
Epoch 9, Loss: 0.0008372313717332979
Training Set Accuracy: 100.0%
Validation Set Accuracy: 86.328125%
Epoch 10, Loss: 0.0006578459407812646
Training Set Accuracy: 100.0%
Validation Set Accuracy: 87.5%
```

This is an indication that the model is overfitting, which is something we also saw in the previous chapter.It’s easy to be lulled into a false sense of security by the 100% accuracy, but the other figure is more representative of how your model will behave in the real world.

Still, the performance isn’t bad, considering how few images it was trained on and how diverse those images were. You’re beginning to hit a wall caused by lack of data, but there are some techniques that you can use to improve your model’s performance. We’ll explore them later in this chapter, but before that, let’s take a look at how to *use* this model.

## Testing “Horses or Humans” Images

It’s all very well and good to be able to build a model, but of course, you want to try it out. A major frustration of mine when I was starting my AI journey was that I could find lots of code that showed me how to build models and charts of how those models were performing, but very rarely was there code to help me kick the tires of the model myself to try it out. I’ll try to help you avoid that problem in this book!

Testing the model is perhaps easiest using Colab. I’ve provided a “Horses or Humans” notebook on GitHub that you can open directly in [Colab](http://bit.ly/horsehuman).

Once you’ve trained the model, you’ll see a section called “Running the Model.” Before running it, you should find a few pictures of horses or humans online and download them to your computer. I recommend you go to [Pixabay.com](http://pixabay.com), which is a really good site to check out for royalty-free images. It’s also a good idea to get your test images together first, because the node can time out while you’re searching.

[Figure 3-9](#ch03_figure_9_1748570891060144) shows a few pictures of horses and humans that I downloaded from Pixabay to test the model.

![](assets/aiml_0309.png)

###### Figure 3-9\. Test images

When they were uploaded, as you can see in [Figure 3-10](#ch03_figure_10_1748570891060159), the model correctly classified one image as a human and another as a horse—but despite the fact that the third image was obviously of a human, the model incorrectly classified it as a horse!

You can also upload multiple images simultaneously and have the model make predictions for all of them. You may also notice that it tends to overfit toward horses. If the human isn’t fully posed (i.e., if you can’t see their full body), the model can skew toward horses. That’s what happened in this case. The first human model is fully posed, and the image resembles many of the poses in the dataset, so the model was able to classify her correctly. On the other hand, the second human model is facing the camera, but only her upper half is in the image. There was no training data that looked like that, so the model couldn’t correctly identify her.

![](assets/aiml_0310.png)

###### Figure 3-10\. Executing the model

Let’s now explore the code to see what it’s doing. Perhaps the most important part is this chunk:

```py
def load_image(image_path, transform):
    # Load image
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    # Apply transformations
    image = transform(image)
    # Add batch dimension, as the model expects batches
    image = image.unsqueeze(0)
    return image
```

Here, we are loading the image from the path that Colab wrote it to. Note that we specify a `transform` to apply to the image. The images being uploaded can be any shape, but if we are going to feed them into the model, they *must* be the same size that the model was trained on. So, if we use the same `transform` that we defined when performing the training, we’ll know it’s in the same dimensions.

At the end is this strange command: `image = image.unsqueeze(0)`.

When you look back at how the model was trained, the DataLoader objects batched the images going into it. If you think of an image as a 2D array of pixels, then the batch is an array of 2D arrays, which of course is then a 3D array.

But when we’re using this code with one image at a time, there’s no batch, so to make this a 3D array (which is technically a batch with one item in it), we can just unsqueeze the image along axis 0 to simulate this.

With our image in the right format, it’s easy to do the classification:

```py
with torch.no_grad():
    output = model(image)
```

The model then returns an array containing the classifications for the batch. Because there’s only one classification in this case, it’s effectively an array containing an array. You can see this back in [Figure 3-10](#ch03_figure_10_1748570891060159), where for the first human model, the array looks like `tensor([[2.1368e-05]], device='cuda:0').`

So now, it’s simply a matter of inspecting the value of the first element in that array. If it’s greater than 0.5, we’re looking at a human:

```py
class_name = "Human" if prediction.item() == 1 else "Horse"
```

There are a few important points to consider here. First, even though the network was trained on synthetic, computer-generated imagery, it performs quite well at spotting horses and humans and differentiating them in real photographs. This is a potential boon in that you may not need thousands of photographs to train a model, and you can do it relatively cheaply with CGI.

But this dataset also demonstrates a fundamental issue you will face. Your training set cannot hope to represent *every* possible scenario your model might face in the wild, and thus, the model will always have some level of overspecialization toward the training set. We saw a clear and simple example of this earlier in this section, when the model mischaracterized the human in the center of [Figure 3-9](#ch03_figure_9_1748570891060144). The training set didn’t include a human in that pose, and thus, the model didn’t “learn” that a human could look like that. As a result, there was every chance it might see the figure as a horse, and in this case, it did.

What’s the solution? The obvious one is to add more training data, with humans in that particular pose and others that weren’t initially represented. That isn’t always possible, though. Fortunately, there’s a neat trick in PyTorch that you can use to virtually extend your dataset—it’s called *image augmentation*, and we’ll explore that next.

# Image Augmentation

In the previous section, you built a horse-or-human classifier model that was trained on a relatively small dataset. As a result, you soon began to hit problems classifying some previously unseen images, such as the miscategorization of a woman as a horse because the training set didn’t include any images of people in that pose.

One way to deal with such problems is with *image augmentation*. The idea behind this technique is that as PyTorch is loading your data, it can create additional new data by amending what it has using a number of transforms. For example, take a look at [Figure 3-11](#fig-3-11). While there is nothing in the dataset that looks like the woman on the right, the image on the left is somewhat similar.

![](assets/aiml_0311.png)

###### Figure 3-11\. Dataset similarities

So, if you could, for example, zoom into the image on the left as you are training, as shown in [Figure 3-12](#fig-3-12), you would increase the chances of the model being able to correctly classify the image on the right as a person.

![](assets/aiml_0312.png)

###### Figure 3-12\. Zooming in on the training set data

In a similar way, you can broaden the training set with a variety of other transformations, including the following:

*   Rotation (turning the image)

*   Shifting horizontally (moving the pixels horizontally with wrapping)

*   Shifting vertically (moving the pixels vertically with wrapping)

*   Shearing (moving the pixels either horizontally or vertically but offsetting so that the image would look like parallelogram)

*   Zooming (magnifying a particular region)

*   Flipping (vertically or horizontally)

Because you’ve been using the `datasets.ImageFolder` and a `DataLoader` to load the images, you’ve seen the model do a transform already—when it normalized the images like this:

```py
# Define transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

Many other transforms are easily available within the torchvision.transforms library, so, for example, you could do something like this:

```py
# Transforms for the training data
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(150),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

# Transforms for the validation data
val_transforms = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

Here, in addition to rescaling the image to normalize it, you’re doing the following:

*   Randomly flipping horizontally

*   Randomly rotating up to 20 degrees left or right

*   Randomly cropping a 150 × 150 window instead of resizing

In addition, the transforms.RandomAffine library gives you the facility to do all of these things, as well as adding stuff like scaling the image (zooming in or out), shearing the image, etc. Here’s an example:

```py
transforms.RandomAffine(
    degrees=0,  # No rotation
    translate=(0.2, 0.2),  # Translate up to 20% vert and horizontally
    scale=(0.8, 1.2),  # Zoom in or out by 20%
    shear=20,  # Shear by up to 20 degrees
),
```

When you retrain with these parameters, one of the first things you’ll notice is that training takes longer because of all the image processing. Also, your model’s accuracy may not be as high as it was, because previously it was overfitting to a largely uniform set of data.

In my case, when I was training with these augmentations, my accuracy went down from 99% to 94% after 15 epochs, with validation much lower at 64%. This likely indicates overfitting in the model, but it warrants investigation by training with more epochs! One other thing to note is that random cropping might also be an issue—the CGI images generally center the subject, so random cropping will give partial subjects.

But what about the image from [Figure 3-9](#ch03_figure_9_1748570891060144) that the model misclassified earlier? This time, the model gets it right. Thanks to the image augmentations, the training set now has sufficient coverage for the model to understand that this particular image is a human too (see [Figure 3-13](#ch03_figure_11_1748570891060175)). This is just a single data point, and it may not be representative of the results for real data, but it’s a small step in the right direction.

![](assets/aiml_0313.png)

###### Figure 3-13\. The woman is now correctly classified

As you can see, even with a relatively small dataset like “Horses or Humans,” you can start to build a pretty decent classifier. With larger datasets, you could take this further. Another way you can improve the model is by using features that the model has already learned elsewhere. Many researchers with massive resources (millions of images) and huge models that have been trained on thousands of classes have shared their models, and by using a concept called *transfer learning*, you can use the features those models learned and apply them to your data. We’ll explore that next!

# Transfer Learning

As we’ve already seen in this chapter, the use of convolutions to extract features can be a powerful tool for identifying the contents of an image. If we use this tool, we can then feed the resulting feature maps into the dense layers of a neural network to match them to the labels and give us a more accurate way of determining the contents of an image. Using this approach with a simple fast-to-train neural network and some image augmentation techniques, we built a model that was 80–90% accurate at distinguishing between a horse and a human when it was trained on a very small dataset.

However, we can improve our model even further by using a method called *transfer learning*. The idea behind it is simple: instead of having our model learn a set of filters from scratch for our dataset, why not have it use a set of filters that were learned on a much larger dataset, with many more features than we can “afford” to build from scratch? We can place these filters in our network and then train a model with our data using the pre-learned filters. For example, while our “Horses or Humans” dataset has only two classes, we can use an existing model that has been pretrained for one thousand classes—but at some point, we’ll have to throw away some of the preexisting network and add the layers that will let us have a classifier for two classes.

[Figure 3-14](#ch03_figure_12_1748570891060190) shows what a CNN architecture for a classification task like ours might look like. We have a series of convolutional layers that lead to a dense layer, which in turn leads to an output layer.

![](assets/aiml_0314.png)

###### Figure 3-14\. A CNN architecture

We’ve seen that we can build a pretty good classifier using this architecture. But what if we could use transfer learning to take the pre-learned layers from another model, freeze or lock them so that they aren’t trainable, and then put them on top of our model, like in [Figure 3-15](#ch03_figure_13_1748570891060207)?

![](assets/aiml_0315.png)

###### Figure 3-15\. Taking and locking layers from another architecture via transfer learning

When we consider that once they’ve been trained, all these layers are just a set of numbers indicating the filter values, weights, and biases along with a known architecture (the number of filters per layer, the size of the filter, etc.), the idea of reusing them is pretty straightforward.

Let’s look at how this would appear in code. There are several pretrained models already available from a variety of sources, so we’ll use version 3 of the popular Inception model from Google, which is trained on more than a million images from a database called ImageNet. Inception has dozens of layers, and it can classify images into one thousand categories.

The torchvision.models library contains a number of models, including Inception V3, so we can easily get access to the pretrained model:

```py
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import RMSprop

# Load the pretrained Inception V3 model
pre_trained_model = models.inception_v3(pretrained=True, aux_logits=True)
```

Now, we have a full Inception model that’s pretrained. If you want to inspect its architecture, you can do so with this code:

```py
def print_model_summary(model):
    for name, module in model.named_modules():
        print(f"{name} : {module.__class__.__name__}")

# Example of how to use the function with your pretrained model
print_model_summary(pre_trained_model)
```

Be warned—this model is huge! Still, you should take a look through it to see the layers and their names. I like to use the one called `Mixed7_c` because its output is nice and small—it consists of 8 × 8 images—but you should feel free to experiment with others.

Next, we’ll freeze the entire network from retraining and then set a variable to point to `mixed7`’s output as where we want to crop the network. We can do that with this code:

```py
# Freeze all layers up to and including the 'Mixed_7c'
for name, parameter in pre_trained_model.named_parameters():
    parameter.requires_grad = False
    if 'Mixed_7c' in name:
        break
```

You’ll notice that we’re printing the output shape of the last layer, and you’ll also see that we’re getting 8 × 8 images at this point. This indicates that by the time the images have been fed through to `Mixed_7c`, the output images from the filters are 8 × 8 in size, so they’re pretty easy to manage. Again, you don’t have to choose that specific layer; you’re welcome to experiment with others.

Now, let’s see how to modify the model for transfer learning. It’s pretty straightforward—if you go back to the output from the custom `print_model_summary` from a moment ago, you’ll see that the *last* layer in the model is called `fc`. As you might expect, *fc* stands for *fully connected*, which is effectively a Linear layer with our densely connected neurons.

So now, it becomes as simple as replacing that layer with a new layer called `fc`. We don’t need to *know* the input shape for it ahead of time—we can inspect its `in_features` property to find that. So now, to create a new layer of 1,024 neurons that outputs to another layer of two neurons and replace the `fc` from Inception, all we have to do is this:

```py
# Modify the existing fully connected layer
num_ftrs = pre_trained_model.fc.in_features
pre_trained_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),  # New fully connected layer 
    nn.ReLU(),                # Activation layer
    nn.Linear(1024, 2)         # Final layer for binary classification
)
```

It’s as simple as creating a new set of Linear layers from the last output, because we’ll be feeding the results into a dense layer. So, we then add a Linear layer of 1,024 neurons and a dense layer with two neurons for our output. Also, you’ve probably noticed that in the previous model, we did it with one neuron and used sigmoid activation for the two classes—so you’re probably wondering why we’re going to two neurons in the output layer now. This was primarily a stylistic choice. Inception was designed for *n* neurons to output for *n* classes, and I wanted to keep that approach.

Training the model on this architecture over only three epochs gave us an accuracy of 99%+, with a validation accuracy of 95%+. Clearly, that’s a vast improvement. Also, remember that Inception learned a massive set of features that it could use to classify the many classes it was trained on. It turns out that that feature set is also incredibly useful for learning how to classify any other images—not least, those from “Horses or Humans.”

The results we got from this model are much better than those we got from our previous model, but you can continue to tweak and improve it. You can also explore how the model will work with a much larger dataset, like the famous “[Dogs vs. Cats”](https://oreil.ly/UhWMk) from Kaggle. It’s an extremely varied dataset consisting of 25,000 images of cats and dogs, often with the subjects somewhat obscured—for example, if they are held by a human.

Using the same algorithm and model design as before, you can train a “Dogs vs. Cats” classifier on Colab, using a GPU at about 3 minutes per epoch.

When I tested with very complex pictures like those in [Figure 3-16](#ch03_figure_14_1748570891060222), this classifier got them all correct. I chose one picture of a dog with catlike ears and one with its back turned. Both pictures of cats were nontypical.

![](assets/aiml_0316.png)

###### Figure 3-16\. Unusual dogs and cats that the model classified correctly

To parse the results, you can use code like this:

```py
     def load_image(image_path, transform):
    # Load image
    image = Image.open(image_path).convert('RGB')  # Convert to RGB 
    # Apply transformations
    image = transform(image)
    # Add batch dimension, as the model expects batches
    image = image.unsqueeze(0)
    return image

    # Prediction function
def predict(image_path, model, device, transform):
    model.eval()
    image = load_image(image_path, transform)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        print(output)
        prediction = torch.max(output, 1)
        print(prediction)
```

Note the lines where I’m printing the output of the image, calculating the prediction from that, and printing that.

When you upload some images to Colab, you can see how they predict in [Figure 3-17](#ch03_figure_15_1748570891060236).

![](assets/aiml_0317.png)

###### Figure 3-17\. Classifying the cat washing its paw

The first image uploaded was “labrador,” which, as its name suggests, is of a dog. The tensor returned from the model contained [–14.9642, 18.3943], meaning a very low number for the first label and a very high one for the second. Given that we used an image directory when training, the labels ended up being in alphabetical order, so it was low for cat and high for dog. Then, when we called `torch.max`, it gave us [1]. That indicates that neuron 1 is the one for this classification—thus, the image is a dog.

The second image had [5.3486, –4.8260], with the first neuron being higher. Thus, it detected a cat. The size of these numbers indicates the strength of the prediction. For example, it was much surer that the first image is a dog than it was that the second image is a cat.

You can find the complete code for the “Horses or Humans” and “Dogs vs. Cats” classifiers in the book’s [GitHub repository](https://github.com/lmoroney/tfbook).

# Multiclass Classification

In all of the examples so far, you’ve been building *binary* classifiers—ones that choose between two options (horses or humans, cats or dogs). On the other hand, when you’re building *multiclass classifiers*, the models are almost the same but there are a few important differences. Instead of a single neuron that is sigmoid activated or two neurons that are binary activated, your output layer will now require *n* neurons, where *n* is the number of classes you want to classify. You’ll also have to change your loss function to an appropriate one for multiple categories.

A neat feature of the `nn.CrossEntropyLoss` loss function in PyTorch is that it can handle multiple categories, so the “Cats vs. Dogs” and “Horses or Humans” transfer learning classifiers you’ve built thus far in this chapter can use it without modification. But the “Horses or Humans” classifier that you built at the beginning with a *single* output neuron will not be able to because it can’t handle more than two classes. This is always something to look out for, and it’s a common bug when you start writing code for classification.

To go beyond two-class classification, consider, for example, the game Rock, Paper, Scissors. If you wanted to train a dataset to recognize the different hand gestures used in this game, you’d need to handle three categories. Fortunately, there’s a [simple dataset](https://oreil.ly/VHhmS) you can use for this.

There are two downloads: a training set of many diverse hands, with different sizes, shapes, colors, and details such as nail polish; and a testing set of equally diverse hands, none of which are in the training set. You can see some examples in [Figure 3-18](#ch03_figure_16_1748570891060251).

![Examples of Rock/Paper/Scissors gestures](assets/aiml_0318.png)

###### Figure 3-18\. Examples of Rock, Paper, Scissors gestures

Using the dataset is simple. You can download and unzip it—the sorted subdirectories are already present in the ZIP file—and then use it to initialize an `ImageFolder`:

```py
!wget --no-check-certificate \
 https://storage.googleapis.com/learning-datasets/rps.zip \
 -O /tmp/rps.zip
local_zip = '/tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()
training_dir = "/tmp/rps/"

train_dataset = ImageFolder(root=training_dir, transform=transform)
```

Be sure to use a `transform` that fits the input shape of your model. In the last few examples, we were using Inception, and it’s 299 × 299.

You can use the ImageFolder for your DataLoader in the usual way:

```py
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

Earlier, when we tweaked the Inception model for “Horses or Humans” or “Cats vs. Dogs,” there were only *two* classes and thus *two* output neurons. Given that this data has *three* classes, we need to be sure that we change the new fully connected layer at the bottom accordingly:

```py
# Modify the existing fully connected layer
num_ftrs = pre_trained_model.fc.in_features
pre_trained_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),  # New fully connected layer 
    nn.ReLU(),                  # Activation layer
    nn.Linear(1024, 3)         # Final layer for binary classification
)
```

Now, training the model works as before: you specify the loss function and optimizer, and you call the `train_model()` function. For good repetition, this function is the same as the one used in the “Horses or Humans” and “Cats vs. Dogs” examples:

```py
# Only optimize parameters that are set to be trainable
optimizer = RMSprop(filter(lambda p: p.requires_grad, 
                    pre_trained_model.parameters()), lr=0.001)

criterion = nn.CrossEntropyLoss()

# Train the model
train_model(pre_trained_model, criterion, optimizer, train_loader, num_epochs=3)
```

Your code for testing predictions will also need to change somewhat. There are now three output neurons, and they will output a high value for the predicted class and lower values for the other classes.

Note also that when you’re using the `ImageFolder`, the classes are loaded in alphabetical order—so while you might expect the output neurons to be in the order of the name of the game, the order will in fact be Paper, Rock, Scissors.

Code that you can use to try out predictions in a Colab notebook will look like the following. It’s very similar to what you saw earlier:

```py
def load_image(image_path, transform):
    # Load image
    image = Image.open(image_path).convert('RGB')  # Convert to RGB 
    # Apply transformations
    image = transform(image)
    # Add batch dimension, as the model expects batches
    image = image.unsqueeze(0)
    return image

    # Prediction function
def predict(image_path, model, device, transform):
    model.eval()
    image = load_image(image_path, transform)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        print(output)
        prediction = torch.max(output, 1)
        print(prediction)
```

Note that it doesn’t parse the output; it just prints the classes. [Figure 3-19](#ch03_figure_17_1748570891060264) shows what it looks like in actual use.

![](assets/aiml_0319.png)

###### Figure 3-19\. Testing the Rock, Paper, Scissors classifier

You can see from the filenames what the images were.

If you explore this a little deeper, you can see that the file named *scissors4.png* had an output of –2.5582, –1.7362, 3.8465]. The largest number is the third one, and if you think alphabetically, you can see that the third neuron represents scissors, so it was classified correctly. Similar results were achieved for the other files.

Some images that you can use to test the dataset [are available to download](https://oreil.ly/dEUpx). Alternatively, of course, you can try your own. Note that the training images are all done against a plain white background, though, so there may be some confusion if there is a lot of detail in the background of the photos you take.

# Dropout Regularization

Earlier in this chapter, we discussed overfitting, in which a network may become too specialized in a particular type of input data and thus fare poorly on others. One technique to help overcome this is use of *dropout regularization*.

When a neural network is being trained, each individual neuron will have an effect on neurons in subsequent layers. Over time, particularly in larger networks, some neurons can become overspecialized—and that feeds downstream, potentially causing the network as a whole to become overspecialized and thus leading to overfitting. Additionally, neighboring neurons can end up with similar weights and biases, and if not monitored, this condition can lead the overall model to become overspecialized on the features activated by those neurons.

For example, consider the neural network in [Figure 3-20](#ch03_figure_18_1748570891060278), in which there are layers of 2, 6, 6, and 2 neurons. The neurons in the middle layers might end up with very similar weights and biases.

![](assets/aiml_0320.png)

###### Figure 3-20\. A simple neural network

While training, if you remove a random number of neurons and ignore them, then their contribution to the neurons in the next layer is temporarily blocked (see [Figure 3-21](#ch03_figure_19_1748570891060292)). They are effectively dropped out, leading to the term *dropout regularization*.

![](assets/aiml_0321.png)

###### Figure 3-21\. A neural network with dropouts

This reduces the chances of the neurons becoming overspecialized. The network will still learn the same number of parameters, but it should be better at generalization—that is, it should be more resilient to different inputs.

###### Note

The concept of dropouts was proposed by Nitish Srivastava et al. in their 2014 paper “[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://oreil.ly/673CJ).”

To implement dropouts in PyTorch, you can just use a simple layer like this:

```py
nn.Dropout(0.5)
```

This will drop out, at random, the specified percentage of neurons (here, 50%) in the specified layer. Note that it may take some experimentation to find the correct percentage for your network.

For a simple example that demonstrates this, consider the new fully connected layers we added to the bottom of Inception with the transfer learning example in this chapter.

Here it is for Rock, Paper, Scissors with three output neurons:

```py
num_ftrs = pre_trained_model.fc.in_features
pre_trained_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),  # New fully connected layer 
    nn.ReLU(),                # Activation layer
    nn.Linear(1024, 3)         # Final layer for RPS
)
```

With dropout added, it would look like this:

```py
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Adding dropout before the final FC layer
    nn.Linear(num_ftrs, 1024),  # Reduce dimensionality to 1024
    nn.ReLU(),
    nn.Dropout(0.5),  # Adding another dropout layer after ReLU activation
    nn.Linear(1024, 3)  # Final layer for RPS
)
```

The examples that we used in this chapter for transfer learning are already learning really well without the use of dropouts. However, I’d recommend that you always consider dropouts when building your models because they can greatly reduce waste in the ML process—letting your model learn just as well but much faster!

Additionally, as you design your neural networks, keep in mind that getting great results on your training set is not always a good thing because it could be a sign of overfitting. Introducing dropouts can help you remove that problem so that you can optimize your network in other areas without that false sense of security.

# Summary

This chapter introduced you to a more advanced way of achieving computer vision by using convolutional neural networks. You saw how to use convolutions to apply filters that can extract features from images, and you designed your first neural networks to deal with more complex vision scenarios than those you encountered with the MNIST and Fashion MNIST datasets. You also explored techniques to improve your network’s accuracy and avoid overfitting, such as the use of image augmentation and dropouts.

Before we explore further scenarios, in [Chapter 4](ch04.html#ch04_using_data_with_pytorch_1748548966496246), you’ll get an introduction to PyTorch data, which is a technology that makes it much easier for you to get access to data for training and testing your networks. In this chapter, you downloaded ZIP files and extracted images, but that’s not always going to be possible. With PyTorch datasets, you’ll be able to access lots of datasets with a standard API.