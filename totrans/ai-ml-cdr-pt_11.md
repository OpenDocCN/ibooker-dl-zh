# Chapter 10\. Creating ML Models to Predict Sequences

[Chapter 9](ch09.html#ch09_understanding_sequence_and_time_series_data_1748549698134578) introduced sequence data and the attributes of a time series, including seasonality, trend, autocorrelation, and noise. You created a synthetic series to use for predictions, and you explored how to do basic statistical forecasting.

Over the next couple of chapters, you’ll learn how to use ML for forecasting. But before you start creating models, you need to understand how to structure the time series data for training predictive models by creating what we’ll call a *windowed dataset.*

To understand why you need to do this, consider the time series you created in [Chapter 9](ch09.html#ch09_understanding_sequence_and_time_series_data_1748549698134578). You can see a plot of it in [Figure 10-1](#ch10_figure_1_1748549713788310).

If at any point, you want to predict a value at time *t*, you’ll want to predict it as a function of the values preceding time *t*. For example, say you want to predict the value of the time series at time step 1,200 as a function of the 30 values preceding it. In this case, the values from time steps 1,170 to 1,199 would determine the value at time step 1,200 (see [Figure 10-2](#ch10_figure_2_1748549713788360)).

![](assets/aiml_1001.png)

###### Figure 10-1\. Synthetic time series

![](assets/aiml_1002.png)

###### Figure 10-2\. Previous values impacting prediction

Now, this begins to look familiar: you can consider the values from 1,170 to 1,199 to be your *features* and the value at 1,200 to be your *label*. If you can get your dataset into a condition where you have a certain number of values as features and the following value as the label, and if you do this for every known value in the dataset, then you’ll end up with a pretty decent set of features and labels that you can use to train a model.

Before doing this for the time series dataset from [Chapter 9](ch09.html#ch09_understanding_sequence_and_time_series_data_1748549698134578), let’s create a very simple dataset that has all the same attributes but a much smaller amount of data.

# Creating a Windowed Dataset

PyTorch has a lot of APIs that are useful for manipulating data. For example, you can use `torch.arange(10)` to create a basic dataset containing the numbers 0–9, thus emulating a time series. You can then turn that dataset into the beginnings of a windowed dataset. Here’s the code:

```py
import torch

def create_sliding_windows(data, window_size, shift=1):
    # Convert input to tensor if it isn't already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    # Calculate number of valid windows
    n = len(data)
    num_windows = max(0, (n – window_size) // shift + 1)

    # Create strided view of data
    windows = data.unfold(0, window_size, shift)

    return windows

# Example usage:
data = torch.arange(10)
windows = create_sliding_windows(data, window_size=5, shift=1)

# Print each window
for window in windows:
    print(window.numpy())

```

First, it creates the dataset by using a range, which simply makes the dataset contain the values 0 to *n* – 1, where *n* is, in this case, 10.

Next, calling `create_sliding_windows` and passing a parameter of `5` specifies that the network should split the dataset into windows of five items. Specifying `shift=1` causes each window to then be shifted one spot from the previous one: the first window will contain the five items beginning at 0, the next window will contain the five items beginning at 1, etc.

Running this code will give you the following result:

```py
[0 1 2 3 4]
[1 2 3 4 5]
[2 3 4 5 6]
[3 4 5 6 7]
[4 5 6 7 8]
[5 6 7 8 9]

```

Earlier, you saw that we want to make training data out of this, where there are *n* values defining a feature and there is a subsequent value giving a label. You can do this with some simple Python list slicing that splits each window into two things: everything before the last value and the last value only.

This uses the `unfold` technique on tensor data, which creates a sliding window over your data to turn it into sets like those outlined previously.

It also takes three parameters:

Dimension (0 in this case)

This is the dimension along which to unfold.

window_size

This is the size of each sliding window.

shift

This is the stride/step size between windows.

This process gives us an `*x*` and a `*y*` dataset, as shown here:

```py
import torch

def create_sliding_windows_with_target(data, window_size, shift=1):
    # Convert input to tensor if it isn't already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    # Create windows using unfold
    windows = data.unfold(0, window_size, shift)

    # Split each window into features
    features = windows[:, :–1]  # All elements except the last
    targets = windows[:, –1:]   # Just the last element

    return features, targets

# Example usage:
data = torch.arange(10)
features, targets = create_sliding_windows_with_target(data, window_size=5, 
                                                             shift=1)

# Print each window's features and target
for x, y in zip(features, targets):
    print(f"Features: {x.numpy()}, Target: {y.numpy()}")

```

The results are now in line with what you’d expect. The first four values in the window can be thought of as the features, with the subsequent value being the label:

```py
[0 1 2 3] [4]
[1 2 3 4] [5]
[2 3 4 5] [6]
[3 4 5 6] [7]
[4 5 6 7] [8]
[5 6 7 8] [9]
```

Now, with the PyTorch TensorDataset type, you can turn this into a dataset and do things like shuffling and batching natively.

Note that when shuffling data, it’s good practice to ensure that the validation and test datasets are separated first. In time series data, shuffling before the split can cause information from the training set to bleed into the test set, which would compromise the evaluation process.

Here, it’s been shuffled and batched with a batch size of 2:

```py
from torch.utils.data import TensorDataset, DataLoader

# Create dataset
data = torch.arange(10)
features, targets = create_sliding_windows_with_target(data, window_size=5, 
                                                             shift=1)

# Combine features and targets into a dataset
dataset = TensorDataset(features, targets)

# Create DataLoader with shuffling and batching
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example iteration
for batch_features, batch_targets in dataloader:
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Features:\n{batch_features}")
    print(f"Targets:\n{batch_targets}\n")

```

The results show that the first batch has two sets of `*x*` (starting at 5 and 0, respectively) with their labels, the second batch has two sets of `*x*` (starting at 1 and 3, respectively) with their labels, and so on:

```py
tensor([[5, 6, 7, 8],
        [0, 1, 2, 3]])
Targets:
tensor([[9],
        [4]])

Features:
tensor([[1, 2, 3, 4],
        [3, 4, 5, 6]])
Targets:
tensor([[5],
        [7]])

Features:
tensor([[4, 5, 6, 7],
        [2, 3, 4, 5]])
Targets:
tensor([[8],
        [6]])

```

With this technique, you can now turn any time series dataset into a set of training data for a neural network. In the next section, you’ll explore how to take the synthetic data from [Chapter 9](ch09.html#ch09_understanding_sequence_and_time_series_data_1748549698134578) and create a training set from it. From there, you’ll move on to creating a simple DNN that is trained on this data and can be used to predict future values.

## Creating a Windowed Version of the Time Series Dataset

As a recap, here’s the code we used in the previous chapter to create a synthetic time series dataset:

```py
import numpy as np
def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
series = trend(time, 0.1)
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5

series = baseline + trend(time, slope)
series += seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)

```

This will create a time series that looks like the one in [Figure 10-3](#ch10_figure_3_1748549713788388). If you want to change it, feel free to tweak the values of the various constants.

![](assets/aiml_1003.png)

###### Figure 10-3\. Plotting the time series with trend, seasonality, and noise

Once you have the series, you can turn it into a windowed dataset with code similar to that in the previous section:

```py
import torch
from torch.utils.data import TensorDataset, DataLoader

# Convert the numpy series to a PyTorch tensor
series_tensor = torch.tensor(series, dtype=torch.float32)

# Create windowed dataset with 30-day windows (predicting next day)
window_size = 30
features, targets = create_sliding_windows_with_target(
    series_tensor, window_size=window_size, shift=1)

# Create PyTorch Dataset and DataLoader
dataset = TensorDataset(features, targets)
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Print some information about the dataset
print(f"Series length: {len(series)}")
print(f"Number of windows: {len(features)}")
print(f"Feature shape: {features.shape}")  # Should be (num_windows, 
                                                        window_size-1)
print(f"Target shape: {targets.shape}")    # Should be (num_windows, 1)

# Show a few examples
print("\nFirst few windows:")
for i in range(3):
    print(f"\nWindow {i+1}:")
    print(f"Features (previous {window_size-1} days): {features[i].numpy()}")
    print(f"Target (next day): {targets[i].item():.2f}")
```

You can see the output here:

```py
Series length: 1461
Number of windows: 1432
Feature shape: torch.Size([1432, 29])
Target shape: torch.Size([1432, 1])

First few windows:

Window 1:
Features (previous 29 days): [32.48357  29.395714 33.40659  37.858486 
 29.14184  29.20528  38.32948  34.322147 28.183279 33.283253 28.287313 
 28.303862 31.864614 21.104889 22.057411 27.875519 25.622026 32.25094  
 26.127428 23.588236 37.95459  29.468477 30.900469 23.39905  27.755371 
 30.980967 24.615065 32.186863 27.23822 ]
Target (next day): 28.71

Window 2:
Features (previous 29 days): [29.395714 33.40659  37.858486 29.14184  
 29.20528  38.32948  34.322147 28.183279 33.283253 28.287313 28.303862 
 31.864614 21.104889 22.057411 27.875519 25.622026 32.25094  26.127428 
 23.588236 37.95459  29.468477 30.900469 23.39905  27.755371 30.980967 
 24.615065 32.186863 27.23822  28.710733]
Target (next day): 27.08

Window 3:
Features (previous 29 days): [33.40659  37.858486 29.14184  29.20528  
 38.32948  34.322147 28.183279 33.283253 28.287313 28.303862 31.864614 
 21.104889 22.057411 27.875519 25.622026 32.25094  26.127428 23.588236 
 37.95459  29.468477 30.900469 23.39905  27.755371 30.980967 24.615065 
 32.186863 27.23822  28.710733 27.083256]
Target (next day): 39.27
```

To train a model with this data, you’ll split the series into training and validation datasets. In this case, we’ll train on 1,000 records by splitting the list and turning it into `train_dataset` and `val_dataset` subsets. This uses the `Subset` class from `torch.utils.data`.

You can then load these with a `DataLoader`, as we saw in [Chapter 3](ch03.html#ch03_going_beyond_the_basics_detecting_features_in_ima_1748570891074912):

```py
train_size = 1000 
total_windows = len(full_dataset)
train_indices = list(range(train_size))
val_indices = list(range(train_size, total_windows))

# Create training and validation datasets using Subset
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

The important thing to remember now is that your data is a dataset, so you can easily use it in model training without further coding.

If you want to inspect what the data looks like, you can do so with code like this:

```py
features, target = train_dataset[0]
print("First window:")
print(f"Features shape: {features.shape}")
print(f"Features: {features.numpy()}")
print(f"Target: {target.item()}\n")

```

Here, the `batch_size` is set to `1`, just to make the results more readable. You’ll therefore end up with output like this, in which a single set of data is in the batch:

```py
First window:
Features shape: torch.Size([29])
Features: [32.48357  29.395714 33.40659  37.858486 29.14184  29.20528  38.32948
 34.322147 28.183279 33.283253 28.287313 28.303862 31.864614 21.104889
 22.057411 27.875519 25.622026 32.25094  26.127428 23.588236 37.95459
 29.468477 30.900469 23.39905  27.755371 30.980967 24.615065 32.186863
 27.23822 ]
Target: 28.71073341369629

```

The first batch of numbers are the features. We’ve set the window size to 30, so it’s a 1 × 30 tensor. The second number is the label (28.710 in this case), which the model will try to fit the features to. You’ll see how that works in the next section.

# Creating and Training a DNN to Fit the Sequence Data

Now that you have the data, creating a neural network model becomes very straightforward. Let’s first explore a simple DNN. Here’s the model definition:

```py
# Define the model
class TimeSeriesModel(nn.Module):
    def __init__(self, window_size):
        super(TimeSeriesModel, self).__init__()
        # window_size-1 because our features are window_size-1
        self.network = nn.Sequential(
            nn.Linear(window_size-1, 10),  
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.network(x)

```

It’s a super-simple model with three linear layers, the first of which accepts the input shape of `window_size` and then a hidden layer of 10 neurons, before an output layer that will contain the predicted value.

The model is initialized with a loss function and optimizer, as before:

```py
# Initialize model, loss function, and optimizer
model = TimeSeriesModel(window_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
```

In this case, the loss function is specified as `MSELoss`, which stands for “mean squared error” and is commonly used in regression problems (which is what this ultimately boils down to). For the optimizer, `Adam` is a good fit. I won’t go into detail on these types of functions in this book, but any good resource on ML will teach you about them—Andrew Ng’s seminal “[Deep Learning Specialization](https://oreil.ly/A8QzN)” on Coursera is a great place to start.

Training is then pretty standard. It’s composed of loading the batches from the training loader and performing a forward pass with them, followed by a backward pass with optimization:

```py
for batch_features, batch_targets in train_loader:
    batch_features = batch_features.to(device)
    batch_targets = batch_targets.to(device)

    # Forward pass
    outputs = model(batch_features)
    loss = criterion(outputs, batch_targets)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
```

Given that we also have a validation dataset, we can also perform a validation pass for each epoch:

```py
# Validation phase
model.eval()
val_loss = 0
with torch.no_grad():
    for batch_features, batch_targets in val_loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        outputs = model(batch_features)
        val_loss += criterion(outputs, batch_targets).item()

# Calculate average losses
train_loss /= len(train_loader)
val_loss /= len(val_loader)

train_losses.append(train_loss)
val_losses.append(val_loss)
```

As you train, you’ll see the loss function report a number that will start high but decline steadily. [Figure 10-4](#ch10_figure_4_1748549713788413) shows the loss over 100 epochs.

![](assets/aiml_1004.png)

###### Figure 10-4\. The DNN predictor of model loss over time for the time series data

# Evaluating the Results of the DNN

Once you have a trained DNN, you can start predicting with it. But remember, you have a windowed dataset, so the prediction for a given point is based on the values of a certain number of time steps before it.

Also, given that the dataset is batched, we can easily use the loaders to access batches and explore what it looks like to predict on them.

Here’s the code. We iterate through each batch in the loader and get the features and targets, and then we can get the predictions for the batch by sending the batch to the model:

```py
for batch_features, batch_targets in val_loader:
    batch_features = batch_features.to(device)
    predictions = model(batch_features)
    val_predictions.extend(predictions.cpu().numpy())
    val_targets.extend(batch_targets.numpy())
```

The predictions are then converted from PyTorch tensors into NumPy arrays using `.numpy()`, and then the batches are turned into a single list with the `extend()` call.

You might have also noticed the `.cpu()` in this code. PyTorch allows you to designate *where* your code runs, and if you have a GPU or other accelerator available, you can push the intense calculations of ML to it. You can also use a CPU to do other things like processing and preprocessing data to save accelerator time. This code allows you to explicitly express that.

So, you can then compare your predictions with the actual values quite easily. Here’s the code you use to plot the predicted values against the actual ones using matplotlib:

```py
# Make predictions
model.eval()
with torch.no_grad():
    # Get predictions for validation set
    val_predictions = []
    val_targets = []
    for batch_features, batch_targets in val_loader:
        batch_features = batch_features.to(device)
        outputs = model(batch_features)
        val_predictions.extend(outputs.cpu().numpy())
        val_targets.extend(batch_targets.numpy())

# Plot predictions vs actual for validation set
plt.figure(figsize=(15, 6))
plt.plot(val_targets, label='Actual', color="lightgrey")
plt.plot(val_predictions, label='Predicted', color="red")
plt.title('Predictions vs Actual Values (Validation Set)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
```

You can see the results of this in [Figure 10-5](#ch10_figure_5_1748549713788437). The line for the predicted values (in red) closely matches the overall pattern of the original data, but it’s less noisy, with a lot less variance.

![](assets/aiml_1005.png)

###### Figure 10-5\. Predicted versus actual values in the validation set

From a quick visual inspection, you can see that the prediction isn’t bad because it’s generally following the curve of the original data. When there are rapid changes in the data, the prediction takes a little time to catch up, but on the whole, it isn’t bad.

However, it’s hard to be precise when eyeballing the curve. It’s best to have a good metric, and in [Chapter 9](ch09.html#ch09_understanding_sequence_and_time_series_data_1748549698134578) you learned about one—the MAE. Now that you have the valid data and the results, you can measure the MAE with this code by using `torch.mean` and `torch.abs`:

```py
val_predictions_tensor = torch.tensor(val_predictions)
val_targets_tensor = torch.tensor(val_targets)
mae_torch = torch.mean(torch.abs(
                       val_predictions_tensor – val_targets_tensor))
print(f"Validation MAE (PyTorch): {mae_torch:.4f}")

```

Randomness has been introduced into the data, so your results may vary, but when I tried it, I got a value of 4.57 as the MAE.

You could also argue that at this point, the process of getting the predictions as accurate has become the process of minimizing that MAE. There are some techniques that you can use to do this, including the obvious changing of the window size. I’ll leave you to experiment with that, but in the next section, you’ll do some basic hyperparameter tuning on the optimizer to improve how your neural network learns, and see what impact that will have on the MAE.

# Tuning the Learning Rate

In the previous example, you might recall that you compiled the model with an optimizer that looked like this:

```py
optimizer = optim.Adam(model.parameters())
```

In that case, you didn’t specify an LR, so the model used the default LR of 1 × 10^–³. But that seemed to be a really arbitrary number. What if you changed it, and how should you go about changing it? It would take a lot of experimentation to find the best rate.

One way to experiment with this is by using a `torch.optim.lr_scheduler,` which can change the LR on the fly, epoch by epoch, as the model trains.

A good practice is to start with a higher LR and gradually reduce it as the network learns. Here’s an example:

```py
optimizer = optim.Adam(model.parameters(), lr=0.01) 
```

In this case, you’re going to start the LR at 1e – 2, which is really high. However, you can set a scheduler to multiply the LR rate by a “gamma” amount every *n* epochs. So, for example, the following code will change it every 30 epochs. We’ll start at 0.01, and then, after the thirtieth epoch, it will multiply the LR by .1 to get 0.001\. After 60, it will be 0.0001, etc.:

```py
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

When we run the scheduler like this, the performance improves a little—giving MAE of 4.36\.

You can continue to explore like this by tweaking the LR and also the size of the window—30 days of data to predict 1 day may not be enough, so you might want to try a window of 40 days. Also, try training for more epochs. With a bit of experimentation, you could get an MAE of close to 4, which isn’t bad.

# Summary

In this chapter, you took the statistical analysis of the time series from [Chapter 9](ch09.html#ch09_understanding_sequence_and_time_series_data_1748549698134578) and applied ML to try to do a better job of prediction. ML really is all about pattern matching, and, as expected, you were able to quickly create a deep neural network to spot the patterns with low error before exploring some hyperparameter tuning to improve the accuracy further.

In [Chapter 11](ch11.html#ch11_using_convolutional_and_recurrent_methods_for_sequ_1748549734762226), you’ll go beyond a simple DNN and examine the implications of using an RNN to predict sequential values.