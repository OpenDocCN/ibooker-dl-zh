# Chapter 11\. Using Convolutional and Recurrent Methods for Sequence Models

The last few chapters introduced you to sequence data. You saw how to predict it, first by using statistical methods and then by using basic ML methods with a deep neural network. You also explored how to tune the model’s hyperparameters for better performance.

In this chapter, you’ll look at additional techniques that may further enhance your ability to predict sequence data by using convolutional neural networks as well as recurrent neural networks.

# Convolutions for Sequence Data

In [Chapter 3](ch03.html#ch03_going_beyond_the_basics_detecting_features_in_ima_1748570891074912), you were introduced to convolutions in which a two-dimensional (2D) filter was passed over an image to modify it and potentially extract features. Over time, the neural network learned which filter values were effective at matching the modifications that had been made to the pixels to their labels, thus effectively extracting features from the image. The same technique can be applied to numeric time series data, but with one modification: the convolution will be one dimensional (1D) instead of two dimensional.

Consider, for example, the series of numbers in [Figure 11-1](#ch11_figure_1_1748549734749083).

![](assets/aiml_1101.png)

###### Figure 11-1\. A sequence of numbers

A 1D convolution could operate on these as follows. Consider the convolution to be a 1 × 3 filter with filter values of –0.5, 1, and –0.5, respectively. In this case, the first value in the sequence will be lost and the second value will be transformed from 8 to –1.5 (see [Figure 11-2](#ch11_figure_2_1748549734749133)).

![](assets/aiml_1102.png)

###### Figure 11-2\. Using a convolution with the number sequence

The filter will then stride across the values, calculating new ones as it goes. So, for example, in the next stride, 15 will be transformed into 3 (see [Figure 11-3](#ch11_figure_3_1748549734749161)).

![](assets/aiml_1103.png)

###### Figure 11-3\. An additional stride in the 1D convolution

Using this method, it’s possible to extract the patterns between values and learn the filters that extract them successfully, in much the same way that convolutions on the pixels in images can extract features. In this instance, there are no labels, but the convolutions that minimize overall loss can be learned.

## Coding Convolutions

Before coding convolutions, you’ll need to use the *sliding windows* technique to create a dataset, as shown in [Chapter 10](ch10.html#ch10_creating_ml_models_to_predict_sequences_1748549713795870). The code is available [on this book’s GitHub page](https://oreil.ly/pytorch_ch11).

Once you have that dataset, you can add a convolutional layer before the dense layers that you had previously. Here’s the code, which we’ll look at line by line:

```py
class CNN1D(nn.Module):
    def __init__(self, input_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,
                              out_channels=128,
                              kernel_size=3,
                              padding=1)

        conv_output_size = input_size  # Same padding maintains input size

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128 * conv_output_size, 28)
        self.dense2 = nn.Linear(28, 10)
        self.dense3 = nn.Linear(10, 1)

    def forward(self, x):
        # Transpose input from [batch_size, sequence_length] 
        # to [batch_size, 1, sequence_length]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)

        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.dense3(x)
        return x
```

First, notice the new line that defines a 1D convolutional layer:

```py
        self.conv1 = nn.Conv1d(in_channels=1,
                              out_channels=128,
                              kernel_size=3,
                              padding=1)
```

The `in_channels` parameter defines the dimensionality of the input data. As we have a single sequence of numbers with a single value per data point, this is `1`. If we were using multiple features per time step, such as perhaps an RGB color value, this would be `3`.

The `out_channels` parameter is the number of filters (aka convolutions) that the network will learn.

The `kernel_size` parameter determines the size of the convolution (i.e., the number of data points on the line that a convolution will filter). Refer back to [Figure 11-2](#ch11_figure_2_1748549734749133) and [Figure 11-3](#ch11_figure_3_1748549734749161) and you’ll see a convolution there with a kernel size of 3.

The `padding` parameter adds elements to the beginning and end of your list of data. So, for example, the list of numbers in [Figure 11-1](#ch11_figure_1_1748549734749083) is [4 8 15 16 23 42 51 64 99 –1]. When the filter of kernel size 3 looks at this list, it begins with [4 8 15], and *4* never gets to be the “middle” number. The filter effectively ignores the numbers at the beginning and end of the list. With padding, a 0 will be added at the front and back of the list to make it [0 4 8 15 16 23 42 51 64 99 –1 0], and you can now see that the filter will look first at [0 4 8].

Next, we see a line that looks like this:

```py
conv_output_size = input_size  # Same padding maintains input size
```

This helps us to know the size of the output from the convolutional layer to inform the “next” layer in the sequence.

Why is it the input size, you might wonder. This is the idea of “same padding” that comes about from setting the `padding=1` parameter.

If you consider what would happen if you slid a kernel of size 3 across a list of values as in Figures [11-2](#ch11_figure_2_1748549734749133) and [11-3](#ch11_figure_3_1748549734749161), you’d see an odd effect. Because the kernel starts with its left side aligned with the first value and its center at the second value, and because it slides across to the end of the list where the center of the kernel will be aligned to the second-to-last value, the result of the calculations against the values in the list will give us n – 2 answers, where *n* is the length of the list. But if we pad the list with `padding=1`, then the kernel sliding across the list will give us *n* answers, so the output size from the layer will be the *same* as the input size.

So now, after ReLUing and flattening the results, we can see the next line:

```py
        self.dense1 = nn.Linear(128 * conv_output_size, 28)

```

The input to this will be a number of values: the size of the list, multiplied by 128, where 128 is the number of kernels. It will then output 28 values, which will be fed into the next linear layer.

Now, when you get to the forward function, it begins with this line:

```py
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
```

This looks quite unusual, but it’s necessary for dealing with the convolutions. First of all, consider what the input to a convolution should look like. There’ll be batches of them being fed in, each batch will have 1 dimension, and each item in the batch will have a number of items in it. If we are learning from sequences of 20 items, for example, and if we batch them 32 at a time, then the dimensionality of data being fed into the neural network will be [32, 1, 20].

But if our dataset isn’t giving us that—and if, for example, there are only two dimensions [32, 20]—then we want to use unsqueeze to slip in another dimension. When we pass a `1` into it, it will be put at position `1`, so we’ll get [32, 1, 20] as desired.

The other case might be if we haven’t put our dimension in correctly and added it on the end, like in [32, 20, 1], so the `x = x.transpose(1, 2)` will flip these around and make the dimension [32, 1, 20] again.

Now, these are two specific cases I hardcoded for. You may encounter others, so watch out for issues with your data when feeding it into the neural network. This is likely a place where you can fix them.

The rest of the forward pass is pretty straightforward; it’s just passing the data through the different layers.

The loss function and optimizer are going to be pretty straightforward, too, using a mean-squared-error loss and an Adam optimizer:

```py
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

Training with this will give you a model as before, and to get predictions from the model, you can just use the loader in the same way as you did for training the model. So, for example, you can do this:

```py
# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

And here’s a helper function that can predict an entire series, batch by batch:

```py
def predict(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            batch_predictions = model(inputs)
            predictions.append(batch_predictions.cpu().numpy())

    return np.concatenate(predictions)

```

You can then get the full set of predictions like this:

```py
# Make predictions
train_predictions = predict(model, train_loader)
val_predictions = predict(model, val_loader)
```

Similarly, if you want to plot them, you could extend on this a little to pass in a loader and get back arrays of the predictions and the targets as well as an analytic, such as the MAE:

```py
def evaluate_predictions(model, loader):
    """Generate predictions and calculate metrics"""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)

    return predictions, targets, mae
```

And then you’d call this to get back the multiple responses like this:

```py
# Generate predictions
val_predictions, val_targets, val_mae 
             = evaluate_predictions(model, val_loader)
```

This is now nice and easy to plot:

```py
def plot_predictions(val_pred, val_true):
    """Plot the predictions against actual values"""
    plt.figure(figsize=(15, 6))
    # Plot validation data
    offset = len(val_true)
    plt.plot(range(offset, offset + len(val_true)), 
                   val_true, 'b-', label='Validation Actual')
    plt.plot(range(offset, offset + len(val_pred)), 
                   val_pred, 'r-', label='Validation Predicted')
    plt.title('Time Series Prediction vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
```

A plot of the results against the series is in [Figure 11-4](#ch11_figure_4_1748549734749177).

The MAE in this case is 5.33, which is slightly worse than for the previous prediction. This could be because we haven’t tuned the convolutional layer appropriately, or it could be that convolutions simply don’t help. This is the type of experimentation you’ll need to do with your data.

Do note that this data has a random element in it, so values will change across sessions. If you’re using code from [Chapter 10](ch10.html#ch10_creating_ml_models_to_predict_sequences_1748549713795870) and then running this code separately, you will, of course, have random fluctuations affecting your data and thus your MAE.

![](assets/aiml_1104.png)

###### Figure 11-4\. Convolutional neural network with time sequence data prediction versus actual

But when using convolutions, questions always come up. Why choose the parameters that we chose? Why 128 filters? Why size 3 × 1? The good news is that you can experiment with these things easily to explore different results.

## Experimenting with the Conv1D Hyperparameters

In the previous section, you saw a 1D convolution that was hardcoded with parameters for things like filter number, kernel size, number of strides, etc. When you were training the neural network with it, it appeared that the MAE went up slightly, so you got no benefit from using the `Conv1D`. This may not always be the case, depending on your data, but it could be because of suboptimal hyperparameters. So, in this section, you’ll see how you can do a neural architecture search to find the best results.

One of the nice things about how verbose PyTorch is, in particular for defining the neural network and the forward pass, is that it becomes pretty straightforward to change up the parameters that you use. The idea with a *neural architecture search* is to come up with sets of different parameters to try and then explore the impact that they have on the results by training for a short time and finding those that give the best results.

So, for example, here, we used a single `Conv1D` layer. But what if there were more? Similarly, we hardcoded a number of channels and a kernel size, and we also hardcoded the size of the dense layers and the LR for the optimizer. But what if, instead of hardcoding them, we created a set of options like this?

```py
# Define the search space
num_conv_layers_options = [1, 2]  # Reduced for initial testing
conv_channels_options = [
    [32],
    [64],
    [32, 16],
    [64, 32],
]
kernel_sizes = [3, 5]
dense_sizes_options = [
    [16],
    [32, 16],
    [64, 32],
]
learning_rates = [0.001, 0.0001]
```

With 4 options for the `conv` layers, 2 for the kernel sizes, 3 for the dense dimensions, and 2 for the LR, we have 4 × 2 × 3 × 2 options total, which is 48 combinations. This is called the *search space*.

Note that in this case, you might think it would be 96 because there are 2 layers options and 4 channels options. But in the code to define the search space, which you’ll see in a moment, I only allowed `conv` channel options that match the number of layers, so there will just be 4 options in total for the `conv` layers.

These options will be loaded into a configurations array, with name-value pairs set up for the parameters, like this:

```py
# Generate valid configurations
configurations = []
for num_conv_layers in num_conv_layers_options:
    for channels in conv_channels_options:
        # Only use channel configs that match layer count
        if len(channels) == num_conv_layers:  
            for kernel_size in kernel_sizes:
                for dense_sizes in dense_sizes_options:
                    for lr in learning_rates:
                        configurations.append({
                            'num_conv_layers': num_conv_layers,
                            'conv_channels': channels,
                            'kernel_size': kernel_size,
                            'dense_sizes': dense_sizes,
                            'learning_rate': lr
                        })
```

So now, we can loop through these configurations and set up our `CNN1D` model by using them like this. Note the use of the `config[]` array:

```py
for idx, config in enumerate(configurations):
    print(f"\nTrying configuration {idx + 1}/{len(configurations)}:")
    print(config)

    try:
        model = CNN1D(
            input_size=input_size,
            num_conv_layers=config['num_conv_layers'],
            conv_channels=config['conv_channels'],
            kernel_size=config['kernel_size'],
            dense_sizes=config['dense_sizes']
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=config['learning_rate'])
```

It would be really time-consuming to train each of the 48 combinations for the full set of epochs (say, 100), so we’ve introduced the idea of *early stopping*. First, let’s train the model with the parameters we loaded from the configuration and a new parameter: `early stopping patience`:

```py
trained_model, val_loss = train_model(
    model, train_loader, val_loader, criterion, optimizer,
    epochs=100, device=device, early_stopping_patience=10
)
```

Then, within the training loop, we can implement an early stopping like this:

```py
# Early stopping check
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model = deepcopy(model)
    patience_counter = 0
else:
    patience_counter += 1

if patience_counter >= early_stopping_patience:
    print(f'Early stopping triggered after {epoch} epochs')
    break
```

This keeps track of the loss for the best model and compares the current model with the best one. If the current model “loses” more times than our patience parameter (in this case, 10), then we’ll throw it out and move to the next one. If it “wins,” then we’ll keep the current model as the best one.

Starting from this code, you can try to experiment with the hyperparameters for the number of filters, the size of the kernel, and the size of the stride, keeping the other parameters static.

After some experimentation, I discovered that 2 convolutional layers, with 64 and 32 filters (respectively), a kernel size of 5, two dense layers of 64 and 32, and an LR of .0001 gave the best MAE on the validation set, giving me a final result of 4.4439 MAE.

After training with this, the model had improved accuracy compared with both the naive CNN created earlier *and* the original DNN, giving the results shown in [Figure 11-5](#ch11_figure_5_1748549734749203).

![](assets/aiml_1105.png)

###### Figure 11-5\. Optimized CNN time series predictions versus actual

Further experimentation with the CNN hyperparameters may improve this further.

Beyond convolutions, the techniques we explored in the chapters on NLP with RNNs, including LSTMs, may be powerful when working with sequence data. By their very nature, RNNs are designed for maintaining context, so previous values can have an effect on later ones. You’ll explore using RNNs for sequence modeling next. But first, let’s move on from a synthetic dataset and start looking at real data. In this case, we’ll consider weather data.

# Using NASA Weather Data

One great resource for time series weather data is the [NASA Goddard Institute for Space Studies (GISS) Surface Temperature Analysis](https://oreil.ly/6IixP). If you follow the [Station Data link](https://oreil.ly/F9Hmw), on the right side of the page, you can pick a weather station to get data from. For example, I chose the Seattle Tacoma (SeaTac) airport and was taken to the page shown in [Figure 11-6](#ch11_figure_6_1748549734749227).

![](assets/aiml_1106.png)

###### Figure 11-6\. Surface temperature data from GISS

You can also see a link to download monthly data as CSV at the bottom of this page. If you select this link, a file called *station.csv* will be downloaded to your device, and if you open it, you’ll see that it’s a grid of data with a year in each row and a month in each column (see [Figure 11-7](#ch11_figure_7_1748549734749249)).

![](assets/aiml_1107.png)

###### Figure 11-7\. Exploring the data

As this is CSV data, it’s pretty easy to process in Python, but as with any dataset, do note the format. When reading CSV, you tend to read it line by line, and often, each line has one data point that you’re interested in. In this case, there are at least 12 data points of interest per line, so you’ll have to consider this when reading the data.

## Reading GISS Data in Python

The code to read the GISS data is shown here:

```py
def get_data():
    data_file = "station.csv"
    f = open(data_file)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    temperatures=[]
    for line in lines:
        if line:
            linedata = line.split(',')
            linedata = linedata[1:13]
            for item in linedata:
                if item:
                    temperatures.append(float(item))

    series = np.asarray(temperatures)
    time = np.arange(len(temperatures), dtype="float32")
    return time, series
```

This will open the file at the indicated path (yours will, of course, differ) and read in the entire file as a set of lines, where the line split is the new line character (`\n`). It will then loop through each line, ignoring the first line, and split them on the comma character into a new array called `linedata`. The items from 1 through 13 in this array will indicate the values for the months January through February as strings, and these values will then be converted into floats and added to the array called `temperatures`. Once it’s completed, it will be turned into a NumPy array called `series`, and another NumPy array called `time` will be created that’s the same size as `series`. As it is created using `np.arange`, the first element will be 1, the second will be 2, etc. Thus, this function will return `time` in steps from 1 to the number of data points and will return `series` as the data for that time.

I have noticed that often, there will be “unfilled” data in some of the columns, and these are represented by the value 999.9\. This will, of course, skew any predictive results you want to create. But fortunately, 999.9 values are usually at the *end* of the dataset, so they can easily be cropped. Here’s a helper function to normalize the series while cropping out the 999.9 values:

```py
import numpy as np

def normalize_series(data, missing_value=999.9):
    # Convert to numpy array if not already
    data = np.array(data, dtype=np.float64)

    # Create mask for valid values (not NaN and not missing_value)
    valid_mask = (data != missing_value) & (~np.isnan(data))

    # Keep only valid values
    clean_data = data[valid_mask]

    # Normalize using only valid values
    mean = np.mean(clean_data)
    std = np.std(clean_data)
    normalized = (clean_data - mean) / std

    return normalized

time, series = get_data()
series_normalized = normalize_series(series)
```

You can now load this into a `torch.tensor` and turn it into a set of sliding windows with a target value as before. We discussed the helper function in [Chapter 10](ch10.html#ch10_creating_ml_models_to_predict_sequences_1748549713795870):

```py
series_tensor = torch.tensor(series_normalized, dtype=torch.float32)
window_size = 48
features, targets = create_sliding_windows_with_target(series_tensor, 
                    window_size=window_size, shift=1)
```

Once we have that, we can turn it into a `TensorDataset` and split it into subsets for training and validation:

```py
split_location = 800
# Create the full dataset
full_dataset = TensorDataset(features, targets)

# Calculate split indices
# Note: Since we're using windows, we need to account for the overlap
train_size = 800 - window_size + 1  # Adjust for window overlap
total_windows = len(full_dataset)
train_indices = list(range(train_size))
val_indices = list(range(train_size, total_windows))

# Create training and validation datasets using Subset
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
```

Now that we have the splits as datasets, we can create loaders for them that the neural network will use:

```py
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

And now, we’re ready to train with this data. We can inspect the split by charting the data, and we can see this with the training/validation split in [Figure 11-8](#ch11_figure_8_1748549734749270).

![](assets/aiml_1108.png)

###### Figure 11-8\. The time series train/validation split

In the next section, we’ll explore creating a simple RNN-based neural network to see if we can predict the next values in the sequence.

# Using RNNs for Sequence Modeling

Now that you have the data from the NASA CSV in a windowed dataset, it’s relatively easy to create a model to train a predictor for it. (However, it’s a bit more difficult to train a *good* one!) Let’s start with a simple, naive model using RNNs. Here’s the code:

```py
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, 
                       output_size=1, dropout_rate=0.3):
        super(SimpleRNNModel, self).__init__()

        self.rnn1 = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          dropout=dropout_rate)  # Add dropout to RNN

        self.rnn2 = nn.RNN(input_size=hidden_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          dropout=dropout_rate)  # Add dropout to RNN

        self.dropout = nn.Dropout(dropout_rate)  # Additional dropout layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out1, _ = self.rnn1(x)
        out2, _ = self.rnn2(out1)
        last_out = out2[:, –1, :]
        last_out = self.dropout(last_out)  # Add dropout before final layer
        output = self.linear(last_out)
        return output
```

In this case, as you can see, we use a basic RNN. RNNs are a class of neural networks that are powerful for exploring sequence models, and you first saw them in [Chapter 7](ch07.html#ch07_recurrent_neural_networks_for_natural_language_pro_1748549654891648), when you were looking at NLP. I won’t go into detail on how they work here, but if you’re interested and you skipped that chapter, take a look back at it now. Notably, an RNN has an internal loop that iterates over the time steps of a sequence while maintaining an internal state of the time steps it has seen so far.

While training, you can use a loss function and optimizer like this:

```py
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

The full code is available in this book’s [GitHub repository](https://oreil.ly/pytorch_ch11). Even one hundred epochs of training is enough to get an idea of how the model can predict values. [Figure 11-9](#ch11_figure_9_1748549734749285) shows the results.

![](assets/aiml_1109.png)

###### Figure 11-9\. Results of the SimpleRNN time series prediction versus actual

As you can see, the results were pretty good. It may be a little off in the peaks and when the pattern changes unexpectedly (like at time steps 160–170), but on the whole, it’s not bad. Now, let’s see what happens if we train it for 1,500 epochs (see [Figure 11-10](#ch11_figure_10_1748549734749297)).

![](assets/aiml_1110.png)

###### Figure 11-10\. Time series prediction versus actual for RNN trained over 1,500 epochs

There’s not much of a difference, except that some of the peaks are smoothed out. If you look at the history of loss on both the validation set and the training set, it looks like [Figure 11-11](#ch11_figure_11_1748549734749310).

![](assets/aiml_1111.png)

###### Figure 11-11\. Training and validation model loss over time for the SimpleRNN

As you can see, there’s a healthy match between the training loss and the validation loss, but as the epochs increase, the model begins to overfit on the training set. Perhaps a better number of epochs would be around five hundred.

One reason for this could be the fact that the data, being monthly weather data, is highly seasonal. Another is that there is a very large training set and a relatively small validation set.

Next, we’ll explore using a larger climate dataset.

## Exploring a Larger Dataset

The [KNMI Climate Explorer](https://oreil.ly/J8CP0) allows you to explore granular climate data from many locations around the world. [I downloaded a dataset](https://oreil.ly/Ci9DI) consisting of daily temperature readings from the center of England from the years 1772 to 2020\. This data is structured differently from the GISS data, with the date as a string, followed by a number of spaces, followed by the reading. Go back to [Chapter 4](ch04.html#ch04_using_data_with_pytorch_1748548966496246) to check the details on handling and managing large datasets.

I’ve prepared the data, stripping the headers and removing the extraneous spaces, so that there’s only one space between the date and the reading. That way, it’s easy to read with code like this:

```py
import numpy as np
def get_data():
    data_file = "tdaily_cet.dat.txt"
    f = open(data_file)
    data = f.read()
    f.close()
    lines = data.split('\n')
    temperatures=[]
    for line in lines:
        if line:
            linedata = line.split(' ')
            temperatures.append(float(linedata[1]))

    series = np.asarray(temperatures)
    time = np.arange(len(temperatures), dtype="float32")
    return time, series

```

This dataset has 91,502 data points in it, so before training your model, be sure to split it appropriately. I used a split time of 80,000, leaving 10,663 records for validation:

```py
split_location = 80000

features = features.unsqueeze(1)
# Create the full dataset
full_dataset = TensorDataset(features, targets)

# Calculate split indices
# Note: Since we're using windows, we need to account for the overlap
train_size = split_location - window_size + 1  # Adjust for window overlap
total_windows = len(full_dataset)
train_indices = list(range(train_size))
val_indices = list(range(train_size, total_windows))

# Create training and validation datasets using Subset
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

```

Everything else can remain the same. As you can see in [Figure 11-12](#ch11_figure_12_1748549734749322), after training for one hundred epochs, the plot of the predictions against the validation set looks pretty good.

![](assets/aiml_1112.png)

###### Figure 11-12\. Plot of predictions against real data

There’s a lot of data here, so let’s zoom in to the last hundred days’ worth (see [Figure 11-13](#ch11_figure_13_1748549734749335)).

![](assets/aiml_1113.png)

###### Figure 11-13\. Results of time series prediction versus actual for one hundred days’ worth of data

While the chart generally follows the curve of the data and is getting the trends roughly correct, it is pretty far off, particularly at the extreme ends, so there’s room for improvement.

It’s also important to remember that we normalized the data, so while our loss and MAE may look low, that’s because they are based on the loss and MAE of normalized values that have a much lower variance than the real ones. As [Figure 11-14](#ch11_figure_14_1748549734749347) shows, a tiny amount of loss might lead you into having a false sense of security.

![](assets/aiml_1114.png)

###### Figure 11-14\. Training and validation model loss over time for large dataset

To denormalize the data, you can do the inverse of normalization: first, multiply by the standard deviation, and then add back the mean. At that point, if you wish, you can calculate the real MAE for the prediction set as you’ve done previously.

# Using Other Recurrent Methods

In addition to the `RNN`, PyTorch has other recurrent layer types, such as gated recurrent units (GRUs) and long short-term memory layers (LSTMs), which we discussed in [Chapter 7](ch07.html#ch07_recurrent_neural_networks_for_natural_language_pro_1748549654891648). It is relatively simple to just drop in these RNN types if you want to experiment.

So, for example, if you consider the simple, naive RNN that you created earlier, replacing it with a GRU becomes as easy as using `nn.GRU`:

```py
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, 
                       output_size=1, dropout_rate=0.3):
        super(SimpleRNNModel, self).__init__()

        self.rnn1 = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          dropout=dropout_rate)  

        self.rnn2 = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          dropout=dropout_rate)  

        self.dropout = nn.Dropout(dropout_rate)  
        self.linear = nn.Linear(hidden_size, output_size)

```

With an LSTM, it’s similar:

```py
# LSTM Optional Architecture
import torch.nn as nn

class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, 
                       output_size=1, dropout_rate=0.3):
        super(SimpleLSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            dropout=dropout_rate)  # Add dropout to LSTM

        self.lstm2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            dropout=dropout_rate)  # Add dropout to LSTM

        # Add more layers before final output
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out1, _ = self.lstm1(x)  # LSTM returns (output, (h_n, c_n))
        out2, _ = self.lstm2(out1) # We ignore both hidden and cell states with _
        last_out = out2[:, –1, :]
        output = self.linear(last_out)
        return output

```

It’s worth experimenting with these layer types as well as with different hyperparameters, loss functions, and optimizers. There’s no one-size-fits-all solution, so what works best for you in any given situation will depend on your data and your requirements for prediction with that data.

# Using Dropout

If you encounter overfitting in your models, where the MAE or loss for the training data is much better than with the validation data, you can use dropout. As discussed in earlier chapters, with dropout, neighboring neurons are randomly dropped out (ignored) during training to avoid a familiarity bias. When you’re using RNNs, there’s also a *recurrent dropout* parameter that you can use.

What’s the difference? Recall that when using RNNs, you typically have an input value and the neuron calculates an output value and a value that gets passed to the next time step. Dropout will randomly drop out the input values, and recurrent dropout will randomly drop out the recurrent values that get passed to the next step.

For example, consider the basic RNN architecture shown in [Figure 11-15](#ch11_figure_15_1748549734749377).

![](assets/aiml_1115.png)

###### Figure 11-15\. A recurrent neural network

Here, you can see the inputs into the layers at different time steps (*x*). The current time is *t*, and the steps shown are *t* – 2 through *t* + 1\. The relevant outputs at the same time steps (*y*) are also shown, and the recurrent values passed between time steps are indicated by the dotted lines and labeled as *r*.

Using *dropout* will randomly drop out the *x* inputs, while using *recurrent dropout* will randomly drop out the *r* recurrent values.

You can learn more about how recurrent dropout works from a deeper mathematical perspective in the paper [“A Theoretically Grounded Application of Dropout in Recurrent Neural Networks” by Yarin Gal and Zoubin Ghahramani](https://oreil.ly/MqqRR). One other thing to consider when using recurrent dropout is discussed by Gal in his research around [uncertainty in deep learning](https://oreil.ly/3v8IB), in which he demonstrates that the same pattern of dropout units should be applied at every time step and that a similar constant dropout mask should also be applied at every time step.

To add dropout and recurrent dropout, you use the relevant parameters on your layers. For example, adding them to the basic GRU from earlier was as simple as using a parameter in the recurrent layers and adding another layer between the RNNs and the linears:

```py
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, 
                       output_size=1, dropout_rate=0.1):
        super(SimpleRNNModel, self).__init__()

        self.rnn1 = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          `dropout``=``dropout_rate``)`  

        self.rnn2 = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          `dropout``=``dropout_rate``)`  

        self.dropout = nn.Dropout(dropout_rate)  
        self.linear = nn.Linear(hidden_size, output_size)
```

Each parameter takes a value between 0 and 1 that indicates the proportion of values to drop out. For example, a value of 0.1 will drop out 10% of the requisite values.

Training a model with dropout like this shows a much steeper learning curve, which is still trending downward at 100 epochs. The validation is quite flat, indicating that a larger validation set may be necessary. It’s also quite noisy, and you’ll often see noise like this in the loss when using dropout. It’s an indication that you may want to tweak the amount of dropout as well as the parameters of the loss function and optimizer, such as the LR. You can see this in [Figure 11-16](#ch11_figure_16_1748549734749392).

![](assets/aiml_1116.png)

###### Figure 11-16\. Training and validation loss over time using a GRU with dropout

As you’ve seen in this chapter, predicting time sequence data using neural networks is a difficult proposition, but tweaking their hyperparameters can be a powerful way to improve your model and its subsequent predictions.

# Using Bidirectional RNNs

Another technique to consider when classifying sequences is to use bidirectional training. This may seem counterintuitive at first, as you might wonder how future values could impact past ones. But recall that time series values can contain seasonality, where values repeat over time, and when using a neural network to make predictions, all we’re doing is sophisticated pattern matching. Given that data repeats, a signal for how data can repeat might be found in future values—and when using bidirectional training, we can train the network to try to spot patterns going from time *t* to time *t* + *x*, as well as going from time *t* + *x* to time *t*.

Fortunately, coding this is simple. For example, consider the GRU from the previous section. To make this bidirectional, you simply add a `bidirectional` parameter. This will effectively train twice on each step—once with the sequence data in the original order and once with it in reverse order. The results are then merged before proceeding to the next step.

Here’s an example:

```py
class BidirectionalGRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, 
                       output_size=1, dropout_rate=0.1):
        super(BidirectionalGRUModel, self).__init__()

        self.gru1 = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          dropout=dropout_rate,
                          bidirectional=True)

        self.gru2 = nn.GRU(input_size=hidden_size * 2,
                          hidden_size=hidden_size,
                          batch_first=True,
                          dropout=dropout_rate,
                          bidirectional=True)

        # Additional layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out1, _ = self.gru1(x)
        out2, _ = self.gru2(out1)
        last_out = out2[:, –1, :]

        # Additional processing
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.linear(x)
        return output

```

A plot of the results of training with a bidirectional GRU with dropout on the time series is shown in [Figure 11-17](#ch11_figure_17_1748549734749407). While the MAE has improved slightly, the bigger impact is that the predicted curve has lost the “lag” compared with the single direction version.

Additionally, tweaking the training parameters—particularly `window_size`, to get multiple seasons—can have a pretty big impact.

![](assets/aiml_1117.png)

###### Figure 11-17\. Time series prediction training with a bidirectional GRU

As you can see, you can experiment with different network architectures and different hyperparameters to improve your overall predictions. The ideal choices are very much dependent on the data, so the skills you’ve learned in this chapter will help you with your specific datasets!

# Summary

In this chapter, you explored different network types for building models to predict time series data. You built on the simple DNN from [Chapter 10](ch10.html#ch10_creating_ml_models_to_predict_sequences_1748549713795870), adding convolutions, and you experimented with recurrent network types such as simple RNNs, GRUs, and LSTMs. You also learned how to tweak hyperparameters and the network architecture to improve your model’s accuracy, and you practiced working with some real-world datasets, including one massive dataset with hundreds of years’ worth of temperature readings.

Now, you’re ready to get started building networks for a variety of datasets, and you have a good understanding of what you need to know to optimize them!