# Chapter 9\. Understanding Sequence and Time Series Data

Time series are everywhere. You’ve probably seen them in things like weather forecasts, stock prices, and historic trends like Moore’s law. If you’re not familiar with *Moore’s law*, it predicts that the number of transistors on a microchip will roughly double every two years—and for almost 50 years, it has proven to be an accurate predictor of the future of computing power and cost (see [Figure 9-1](#ch09_figure_1_1748549698128120)).

![Moore’s law](assets/aiml_0901.png)

###### Figure 9-1\. Moore’s law

###### Note

The gaps in [Figure 9-1](#ch09_figure_1_1748549698128120) are missing data for that period of time, but the general trend still holds.

*Time series data* is a set of values that are spaced over time, usually in a particular order or denoting values of a thing at a timestamped point in time. When a time series is plotted on a graph, the *x*-axis is usually temporal in nature. Often, there are a number of values plotted on the time axis, such as in the example shown in [Figure 9-1](#ch09_figure_1_1748549698128120), where the number of transistors is one plot and the predicted value from Moore’s law is the other. This is called a *multivariate* time series. If there’s just a single value—for example, the volume of rainfall over time—then it’s called a *univariate* time series.

With Moore’s law, predictions are simple because there’s a fixed and simple rule that allows us to roughly predict the future—a rule that has held for about 50 years.

But what about a time series like the one shown in [Figure 9-2](#ch09_figure_2_1748549698128170)?

![](assets/aiml_0902.png)

###### Figure 9-2\. A real-world time series

While this time series was artificially created (and you’ll see how to do that later in this chapter), it has all the attributes of a complex real-world time series, like a stock chart or a chart depicting seasonal rainfall. Despite their seeming randomness, time series have some common attributes that are helpful in designing ML models that can predict them, as described in the next section.

# Common Attributes of Time Series

While time series might appear random and noisy, they often have common attributes that are predictable. In this section, we’ll explore some of them.

## Trend

Time series typically move in a specific direction. In the case of Moore’s law, it’s easy to see that over time, the values on the *y*-axis increase and there’s an upward trend. There’s also an upward trend in the time series in [Figure 9-2](#ch09_figure_2_1748549698128170). Of course, this won’t always be the case: some time series may be roughly level over time, despite seasonal changes, and others may have a downward trend. For example, this is the case in the inverse version of Moore’s law that predicts the price per transistor.

## Seasonality

Many time series have a repeating pattern over time, with the repeats happening at regular intervals called *seasons*. Consider, for example, temperature in weather. We typically have four seasons per year, with the temperature being highest in summer. So, if you plotted weather over several years, you’d see peaks happening every four seasons, giving us the concept of seasonality. But this phenomenon isn’t limited to weather—consider, for example, [Figure 9-3](#ch09_figure_3_1748549698128196), which is a plot of traffic to a website.

![](assets/aiml_0903.png)

###### Figure 9-3\. Website traffic

It’s plotted week by week, and you can see regular dips. Can you guess what they are? The site in this case is one that provides information for software developers, and as you would expect, it gets less traffic on weekends! Thus, the time series has a seasonality of five high days and two low days. The data is plotted over several months, with the Christmas and New Year’s holidays roughly in the middle, so you can see an additional seasonality there. If I had plotted it over some years, you’d clearly see the additional end-of-year dip.

There are many ways that seasonality can manifest in a time series. Traffic to a retail website, for instance, might peak on the weekends.

## Autocorrelation

Another feature that you may see in time series is predictable behavior after an event. You can see this in [Figure 9-4](#ch09_figure_4_1748549698128218), in which there are clear spikes, but after each spike, there’s a deterministic decay. This is called *autocorrelation*.

In this case, we can see a particular set of behavior that is repeated. Autocorrelations may be hidden in a time series pattern, but they have inherent predictability, so a time series containing many autocorrelations may be predictable.

![](assets/aiml_0904.png)

###### Figure 9-4\. Autocorrelation

## Noise

As its name suggests, *noise* is a set of seemingly random perturbations in a time series. These perturbations lead to a high level of unpredictability and can mask trends, seasonal behavior, and autocorrelation. For example, [Figure 9-5](#ch09_figure_5_1748549698128240) shows the same autocorrelation from [Figure 9-4](#ch09_figure_4_1748549698128218) but with a little noise added. Suddenly, it’s much harder to see the autocorrelation and predict values.

![](assets/aiml_0905.png)

###### Figure 9-5\. Autocorrelated series with added noise

Given all of these factors, let’s explore how you can make predictions on time series that contain these attributes.

# Techniques for Predicting Time Series

Before we get into ML-based prediction—which is the topic of the next few chapters—we’ll explore some more naive prediction methods. These will enable you to establish a baseline that you can use to measure the accuracy of your ML predictions.

## Naive Prediction to Create a Baseline

The most basic method to predict a time series is to say that the predicted value at time *t* + 1 is the same as the value from time *t*, effectively shifting the time series by a single period.

Let’s begin by creating a time series that has trend, seasonality, and noise:

```py
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, .05)  
baseline = 10
amplitude = 15
slope = 0.09
noise_level = 6

# Create the series
series = baseline + trend(time, slope) 
                  + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)
```

After plotting this, you’ll see something like [Figure 9-6](#ch09_figure_6_1748549698128263).

![A time series showing trend, seasonality, and noise](assets/aiml_0906.png)

###### Figure 9-6\. A time series showing trend, seasonality, and noise

Now that you have the data, you can split it like any data source into a training set, a validation set, and a test set. When there’s some seasonality in the data, as you can see in this case, it’s a good idea when splitting the series to ensure that there are whole seasons in each split. So, for example, if you wanted to split the data in [Figure 9-6](#ch09_figure_6_1748549698128263) into training and validation sets, a good place to do this might be at time step 1,000, which would give you training data up to step 1,000 and validation data after step 1,000.

However, you don’t actually need to do the split here because you’re just doing a naive forecast where each value *t* is simply the value at step *t* – 1, but for the purposes of illustration in the next few figures, we’ll zoom in on the data from time step 1,000 onward.

To predict the series from a split time period onward, where the period that you want to split from is in the variable `split_time`, you can use code like this:

```py
naive_forecast = series[split_time – 1:–1]

```

[Figure 9-7](#ch09_figure_7_1748549698128284) shows the validation set (from time step 1,000 onward, which you get by setting `split_time` to `1000`) with the naive prediction overlaid.

![](assets/aiml_0907.png)

###### Figure 9-7\. Naive forecast on time series

It looks pretty good—there is a relationship between the values—and, when charted over time, the predictions appear to closely match the original values. But how would you measure the accuracy?

## Measuring Prediction Accuracy

There are a number of ways to measure prediction accuracy, but we’ll concentrate on two of them: the *mean squared error* (MSE) and *mean absolute error* (MAE).

With MSE, you simply take the difference between the predicted value and the actual value at time *t*, square it (to remove negatives), and then find the average of all of them.

With MAE, you calculate the difference between the predicted value and the actual value at time *t*, take its absolute value to remove negatives (instead of squaring), and find the average of all of them.

For the naive forecast you just created based on our synthetic time series, you can get the MSE and MAE like this:

```py
import torch
import torch.nn.functional as F

# Mean Squared Error
mse = F.mse_loss(torch.tensor(x_valid), torch.tensor(naive_forecast)).item()
print(mse)

# Mean Absolute Error
mae = F.l1_loss(torch.tensor(x_valid), torch.tensor(naive_forecast)).item()
print(mae)

```

I got an MSE of 76.47 and an MAE of 6.89\. As with any prediction, if you can reduce the error, you can increase the accuracy of your predictions. We’ll look at how to do that next.

## Less Naive Predictions: Using a Moving Average for Prediction

The previous naive prediction took the value at time *t* – 1 to be the forecasted value at time *t*. Using a *moving average* is similar, but instead of just taking the value from *t* – 1, it takes a group of values (say, 30), averages them out, and sets that average value to be the predicted value at time *t*. Here’s the code:

```py
def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
 If window_size=1, then this is equivalent to naive forecast"""
  forecast = []
  for time in range(len(series) – window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)

```

[Figure 9-8](#ch09_figure_8_1748549698128303) shows the plot of the moving average against the data.

![](assets/aiml_0908.png)

###### Figure 9-8\. Plotting the moving average

When I plotted this time series, I got an MSE and MAE of 49 and 5.5, respectively, so it definitely improved the prediction a little. But this approach doesn’t take into account the trend or the seasonality, so we may be able to improve it further with a little analysis.

## Improving the Moving-Average Analysis

Given that the seasonality in this time series is 365 days, you can smooth out the trend and seasonality by using a technique called *differencing*, which just subtracts the value at *t* – 365 from the value at *t*. This will flatten out the diagram. Here’s the code:

```py
diff_series = (series[365:] – series[:-365])
diff_time = time[365:]

```

You can now calculate a moving average of *these* values and add back in the past values:

```py
diff_moving_avg = 
    moving_average_forecast(diff_series, 50)[split_time – 365 – 50:]

diff_moving_avg_plus_smooth_past = 
    moving_average_forecast(series[split_time – 370:–360], 10) + 
    diff_moving_avg

```

When you plot this (see [Figure 9-9](#ch09_figure_9_1748549698128324)), you can already see an improvement in the predicted values: the trend line is very close to the actual values, albeit with the noise smoothed out. Seasonality seems to work, as does the trend.

![](assets/aiml_0909.png)

###### Figure 9-9\. Improved moving average

This impression is confirmed when you calculate the MSE and MAE—in this case, I got 40.9 and 5.13, respectively, showing a clear improvement in the predictions.

# Summary

This chapter introduced time series data and some of the common attributes of time series. You created a synthetic time series and saw how you can start making naive predictions on it, and from these predictions, you established baseline measurements using MSE and MAE. Synthetic data is also a really cool area for exploration—and hopefully, some of the techniques you explored in this chapter will be useful on your learning journey.

This chapter was a nice break from PyTorch and ML, but in the next chapter, you’ll go back to using ML to see if you can improve on your predictions!