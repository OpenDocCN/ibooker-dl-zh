# 第三章。描述性统计和数据分析

*描述性统计*是描述数据并尽可能多地从中提取信息的领域。基本上，描述性统计可以像数据的代表一样，因为它概括了数据的倾向、行为和趋势。

交易和分析从这一领域的指标中借鉴了很多。在本章中，您将看到需要掌握数据分析的主要概念。我发现最好的教育工具是实际示例，因此我将使用一个经济数据集的示例来呈现本章。

让我们以来自美国的通胀数据为例。消费者价格指数（CPI）衡量城市消费者每月支付的一系列产品和服务的价格（这意味着每个月都会向公众发布一个新的观察值，从而形成一个连续的时间序列）。任何两个时间段之间的通胀率是通过价格指数的百分比变化来衡量的。例如，如果去年面包的价格是 1.00 美元，今天的价格是 1.01 美元，那么通胀率为 1.00%。

您可以使用的获取 CPI 数据的代码类似于您在第一章中用来获取 VIX 数据的代码。

```py
`# Importing the required library`
import pandas_datareader as pdr

`# Setting the beginning and end of the historical data`
start_date = '1950-01-01'
end_date   = '2023-01-23'

`# Creating a dataframe and downloading the CPI data using its code name and its source`
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

`# Printing the latest five observations of the dataframe`
print(cpi.tail())

`# Importing the required library`
import pandas as pd

`# Checking if there are NaN values in the CPI dataframe previously defined`
count_nan = cpi['CPIAUCSL'].isnull().sum()

`# Printing the result`
print('Number of NaN values in the CPI dataframe: ' + str(count_nan))

`# Dropping the NaN values from the rows`
cpi = cpi.dropna()

`# Transforming the CPI into a year-on-year measure` cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()

```

到目前为止，您应该有一个包含 CPI 年度变化的数据框架。年度变化是 CPI 上最常见的转换，因为它清晰简单地衡量了总体价格水平的变化，足够长的时间来考虑短期波动和季节性影响（回想一下面包的例子）。

因此，CPI 的年度变化作为通胀总体趋势的一个标尺。它也简单易懂，可以与其他国家和历史时期进行比较，因此在政策制定者和经济学家中很受欢迎（尽管在不同国家之间的篮子中元素权重存在缺陷）。让我们从统计角度分析数据集。

# 中心趋势度量

*中心趋势*指的是将数据集总结为可以代表它们的值的计算。第一个和最为人熟知的中心趋势度量是平均值（平均数）。*平均值*就是值的总和除以它们的数量。它是数据集的中心点，很可能是最能代表它的值。平均值的数学公式如下：

<math alttext="x overbar equals StartFraction 1 Over n EndFraction sigma-summation Underscript n Overscript i equals 1 Endscripts x Subscript i Baseline equals StartFraction 1 Over n EndFraction left-parenthesis x 1 plus period period period plus x Subscript n Baseline right-parenthesis"><mrow><mover accent="true"><mi>x</mi> <mo>¯</mo></mover> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mi>n</mi></mfrac></mstyle> <msubsup><mo>∑</mo> <mrow><mi>n</mi></mrow> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow></msubsup> <msub><mi>x</mi> <mi>i</mi></msub> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mi>n</mi></mfrac></mstyle> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <mo>.</mo> <mo>.</mo> <mo>.</mo> <mo>+</mo> <msub><mi>x</mi> <mi>n</mi></msub> <mo>)</mo></mrow></mrow></math>

让我们以两个数据集的简单示例为例。假设您想要计算数据集 A 和数据集 B 的平均值。您会如何做？

+   数据集 A = [1, 2, 3, 4, 5]

+   数据集 B = [1, 1, 1, 1]

数据集 A 包含 5 个值（数量），总和为 15。这意味着平均值等于 3。数据集 B 包含 4 个值，总和为 4。这意味着平均值等于 1。

###### 注意

当数据集中的所有值相同时，平均值与这些值相同。

图 3-1 显示了美国 CPI 过去二十年的年度值。较高的虚线是过去二十年计算的月度平均值。较低的虚线代表零，低于此线的是通货紧缩期。

![](img/dlf_0220.png)

###### 图 3-1。美国 CPI 过去二十年的年度变化

您可以使用以下代码创建图 3-1：

```py
`# Calculating the mean of the CPI over the last 10 years`
cpi_last_ten_years = cpi.iloc[-240:]
mean = cpi_last_ten_years["CPIAUCSL"].mean()

`# Printing the result`
print('The mean of the dataset: ' + str(mean), '%')

`# Importing the required library`
import matplotlib.pyplot as plt

`# Plotting the latest observations in black with a label`
plt.plot(cpi_last_ten_years[:], color = 'black', linewidth = 1.5, label = 'Change in CPI Year-on-Year')

`# Plotting horizontal lines that represent the mean and the zero threshold`
plt.axhline(y = mean, color = 'red', linestyle = 'dashed', label = '10-Year Mean')
plt.axhline(y = 0, color = 'blue', linestyle = 'dashed', linewidth = 1)

`# Calling a grid to facilitate the visual component`
plt.grid()

`# Calling the legend function so it appears with the chart`
plt.legend()

```

平均值的输出应该如下。

```py
The mean of the dataset: 2.4794 %
```

这意味着 CPI 每年变化的平均观察值约为 2.50%。尽管美联储没有明确的通胀目标，但普遍认为存在一致意见，即维持通胀年度变化在 2.00%左右，因此与历史观察值相差不远。由于 2021 年以来政治和经济动荡导致的最近高通胀数字，有必要回归到平均水平以稳定当前局势。这个例子为过去 10 年所谓的正常水平（~2.50%）提供了一个数值价值。

显然，在 2023 年初高通胀率（约 6.00%）的情况下，情况与正常有些偏离，但是偏离多少？这个问题将在下一节中讨论变异性度量时得到回答。现在，让我们继续讨论中心趋势。

下一个度量是*中位数*，简单来说就是将数据集分成两个相等的部分的值。换句话说，如果您按升序排列数据集，中间的值就是中位数。当数据分布中有很多离群值或偏斜时（可能会使均值偏差较大并且不够代表性），中位数就会被使用。

通常与计算中位数相关的有两个主题，第一个与包含偶数个值的数据集有关（例如，24 行），第二个与包含奇数个值的数据集有关（例如，47 行）：

计算偶数数据集的中位数

如果排列好的数据集中有偶数个值，中位数就是两个中间值的平均值。

计算奇数数据集的中位数

如果排列好的数据集中有一个奇数个值，中位数就是中间的值。

让我们以两个数据集的简单示例为例。假设您想要在数据集 A 和数据集 B 上计算中位数。您会如何做呢？

+   数据集 A = [1, 2, 3, 4, 5]

+   数据集 B = [1, 2, 3, 4]

数据集 A 包含五个值，是奇数个。这意味着中间的值就是中位数。在这种情况下，中位数是 3（注意它也是数据集的均值）。数据集 B 包含四个值，是偶数个。这意味着两个中间值的平均值就是中位数。在这种情况下，中位数是 2.5，即 2 和 3 的平均值。

图 3-2 显示了过去二十年美国 CPI 同比值。较高的虚线是过去二十年计算出的月中位数。较低的虚线代表零。基本上，这就像图 3-1，但是用中位数而不是均值绘制。

![](img/dlf_0225.png)

###### 图 3-2. 过去二十年美国 CPI 同比变化与中位数

您可以使用以下代码创建图 3-2：

```py
`# Calculating the median of the dataset`
median = cpi_last_ten_years["CPIAUCSL"].median() 

`# Printing the result`
print('The median of the dataset: ' + str(median), '%')

`# Plotting the latest observations in black with a label`
plt.plot(cpi_last_ten_years[:], color = 'black', linewidth = 1.5, label = 'Change in CPI Year-on-Year')

plt.axhline(y = median, color = 'red', linestyle = 'dashed', label = '10-Year Median')
plt.axhline(y = 0, color = 'blue', linestyle = 'dashed', linewidth = 1)

`# Calling a grid to facilitate the visual component`
plt.grid()

`# Calling the legend function so it appears with the chart`
plt.legend()

```

中位数的输出应该如下：

```py
The median of the dataset: 2.1143 %
```

显然，中位数受最近来自异常环境的离群值的影响较小。中位数约为 2.10%，更符合隐含目标 2.00%。

###### 注意

请记住，第六章将为您提供有关本章中看到的 Python 代码片段的所有必要信息，因此如果您对编码概念感到困惑，也不必担心。

本节中最后一个中心趋势度量是众数。*众数*是最常见的值（但在数据分析中使用最少）。

让我们以两个数据集的简单示例为例。假设您想要在以下数据集上计算众数。您会如何做呢？

+   数据集 A = [1, 2, 2, 4, 5]

+   数据集 B = [1, 2, 3, 4]

+   数据集 C = [1, 1, 2, 2, 3]

数据集 A 包含两次值为 2，这使其成为众数。数据集 B 没有众数，因为每个值只出现一次。数据集 C 是多峰的，因为它包含多个众数（即 1 和 2）。

###### 注意

众数在分类变量（如信用评级）中很有用，而不是连续变量（如价格和收益时间序列）

在分析时间序列时，您不太可能使用众数，因为均值和中位数更有用。以下是一些在金融分析中使用均值和中位数的示例：

+   计算价格数据的移动均值（平均值）以检测潜在趋势。

+   计算价格衍生指标的滚动中位数以了解其中性区域。

+   使用历史均值计算证券的预期收益。

+   通过比较均值和中位数来检查收益分布的正态性。

对于中心趋势指标的讨论非常重要，特别是均值和中位数不仅作为独立指标被广泛使用，而且作为更复杂的测量方法的组成部分。

###### 注意

本节的主要要点如下：

+   主要有三种中心趋势测量：均值、中位数和众数。

+   均值是总和除以数量，而中位数是将数据一分为二的值。众数是数据集中最频繁出现的值。

## 变异性测量

*变异性测量*描述了数据集中值相对于中心趋势测量的分散程度。最早也是最为人熟知的变异性测量是方差。*方差*描述了一组数字相对于它们的均值的变异性。方差公式背后的思想是确定每个数据点离均值有多远，然后平方这些偏差以确保所有数字都是正数（这是因为距离不能为负）。

找到方差的公式如下：

<math alttext="sigma squared equals StartFraction 1 Over n EndFraction sigma-summation Underscript n Overscript i equals 1 Endscripts left-parenthesis x Subscript i Baseline minus x overbar right-parenthesis squared"><mrow><msup><mi>σ</mi> <mn>2</mn></msup> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mi>n</mi></mfrac></mstyle> <msubsup><mo>∑</mo> <mrow><mi>n</mi></mrow> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow></msubsup> <msup><mrow><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>

这个公式背后的直觉是计算每个数据点与均值之间的平方偏差的总和，从而给出不同距离的观察结果，然后计算这些距离观察结果的均值。

让我们以两个数据集的简单示例为例。假设您想要计算数据集 A 和数据集 B 的方差。您会如何做？

+   数据集 A = [1, 2, 3, 4, 5]

+   数据集 B = [5, 5, 5, 5]

第一步是计算数据集的均值，因为这是您将计算数据离散度的基准。数据集 A 的均值为 3。下一步是逐步使用方差公式如下：

<math alttext="left-parenthesis x 1 minus x overbar right-parenthesis squared equals left-parenthesis 1 minus 3 right-parenthesis squared equals 4"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>1</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>1</mn><mo>-</mo><mn>3</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>4</mn></mrow></math>

<math alttext="left-parenthesis x 2 minus x overbar right-parenthesis squared equals left-parenthesis 2 minus 3 right-parenthesis squared equals 1"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>2</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>2</mn><mo>-</mo><mn>3</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>1</mn></mrow></math>

<math alttext="left-parenthesis x 3 minus x overbar right-parenthesis squared equals left-parenthesis 3 minus 3 right-parenthesis squared equals 0"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>3</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>3</mn><mo>-</mo><mn>3</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>0</mn></mrow></math>

<math alttext="left-parenthesis x 4 minus x overbar right-parenthesis squared equals left-parenthesis 4 minus 3 right-parenthesis squared equals 1"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>4</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>4</mn><mo>-</mo><mn>3</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>1</mn></mrow></math>

<math alttext="left-parenthesis x 5 minus x overbar right-parenthesis squared equals left-parenthesis 5 minus 3 right-parenthesis squared equals 4"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>5</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>5</mn><mo>-</mo><mn>3</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>4</mn></mrow></math>

前面的结果总结如下：

<math alttext="4 plus 1 plus 0 plus 1 plus 4 equals 10"><mrow><mn>4</mn> <mo>+</mo> <mn>1</mn> <mo>+</mo> <mn>0</mn> <mo>+</mo> <mn>1</mn> <mo>+</mo> <mn>4</mn> <mo>=</mo> <mn>10</mn></mrow></math>

最后，结果被观察数量除以以找到方差：

<math alttext="sigma squared equals StartFraction 10 Over 5 EndFraction equals 2"><mrow><msup><mi>σ</mi> <mn>2</mn></msup> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>10</mn> <mn>5</mn></mfrac></mstyle> <mo>=</mo> <mn>2</mn></mrow></math>

至于数据集 B，您应该直观地考虑一下。如果观察结果都相等，它们都代表数据集，这也意味着它们是自己的均值。在这种情况下，您对数据的方差会有什么看法，考虑到所有值都等于均值？

如果您的回答是方差为零，那么您是正确的。从数学上讲，您可以按照以下方式计算：

<math alttext="left-parenthesis x 1 minus x overbar right-parenthesis squared equals left-parenthesis 5 minus 5 right-parenthesis squared equals 0"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>1</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>5</mn><mo>-</mo><mn>5</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>0</mn></mrow></math>

<math alttext="left-parenthesis x 2 minus x overbar right-parenthesis squared equals left-parenthesis 5 minus 5 right-parenthesis squared equals 0"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>2</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>5</mn><mo>-</mo><mn>5</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>0</mn></mrow></math>

<math alttext="left-parenthesis x 3 minus x overbar right-parenthesis squared equals left-parenthesis 5 minus 5 right-parenthesis squared equals 0"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>3</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>5</mn><mo>-</mo><mn>5</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>0</mn></mrow></math>

<math alttext="left-parenthesis x 4 minus x overbar right-parenthesis squared equals left-parenthesis 5 minus 5 right-parenthesis squared equals 0"><mrow><msup><mrow><mo>(</mo><msub><mi>x</mi> <mn>4</mn></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <msup><mrow><mo>(</mo><mn>5</mn><mo>-</mo><mn>5</mn><mo>)</mo></mrow> <mn>2</mn></msup> <mo>=</mo> <mn>0</mn></mrow></math>

前面的结果总结为零，如果您将零除以 4（数据集的数量），您将得到零。直觉上，没有方差，因为所有值都是恒定的，它们不偏离其均值。

<math alttext="sigma squared equals StartFraction 0 Over 5 EndFraction equals 0"><mrow><msup><mi>σ</mi> <mn>2</mn></msup> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>0</mn> <mn>5</mn></mfrac></mstyle> <mo>=</mo> <mn>0</mn></mrow></math>

您可以使用以下代码在 Python 中计算方差：

```py
`# Calculating the variance of the dataset`
variance = cpi_last_ten_years["CPIAUCSL"].var() 

`# Printing the result`
print('The variance of the dataset: ' + str(variance), '%')
```

方差的输出应该如下：

```py
The variance of the dataset: 3.6248 %
```

然而，存在一个缺陷，即方差代表平方值，与均值不可比较，因为它们使用不同的单位。通过对方差取平方根来轻松解决这个问题。这样做带来了下一个变异性测量，*标准差*。它是方差的平方根，是值与均值的平均偏差。

低标准差表示值倾向于接近均值（低波动性），而高标准差表示值相对于均值分布在更广范围内（高波动性）。

###### 注意

标准差和波动性这两个词是可以互换使用的。它们指的是同一件事。

找到标准差的公式如下：

<math alttext="sigma equals StartRoot StartFraction 1 Over n EndFraction sigma-summation Underscript n Overscript i equals 1 Endscripts left-parenthesis x Subscript i Baseline minus x overbar right-parenthesis squared EndRoot"><mrow><mi>σ</mi> <mo>=</mo> <msqrt><mrow><mstyle scriptlevel="0" displaystyle="false"><mfrac><mn>1</mn> <mi>n</mi></mfrac></mstyle> <msubsup><mo>∑</mo> <mrow><mi>n</mi></mrow> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow></msubsup> <msup><mrow><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup></mrow></msqrt></mrow></math>

如果您考虑了前面关于方差的例子，那么标准差可以如下找到：

<math alttext="sigma Subscript upper D a t a s e t upper A Baseline equals StartRoot 2 EndRoot equals 1.41"><mrow><msub><mi>σ</mi> <mrow><mi>D</mi><mi>a</mi><mi>t</mi><mi>a</mi><mi>s</mi><mi>e</mi><mi>t</mi><mi>A</mi></mrow></msub> <mo>=</mo> <msqrt><mn>2</mn></msqrt> <mo>=</mo> <mn>1</mn> <mo>.</mo> <mn>41</mn></mrow></math>

<math alttext="sigma Subscript upper D a t a s e t upper B Baseline equals StartRoot 0 EndRoot equals 0"><mrow><msub><mi>σ</mi> <mrow><mi>D</mi><mi>a</mi><mi>t</mi><mi>a</mi><mi>s</mi><mi>e</mi><mi>t</mi><mi>B</mi></mrow></msub> <mo>=</mo> <msqrt><mn>0</mn></msqrt> <mo>=</mo> <mn>0</mn></mrow></math>

标准差通常与均值一起使用，因为它们使用相同的单位。当我讨论正态分布函数时，您很快就会理解这个统计量的重要性，这是描述性统计中的一个关键概念。

您可以使用以下代码在 Python 中计算标准差：

```py
`# Calculating the standard deviation of the dataset`
standard_deviation = cpi_last_ten_years["CPIAUCSL"].std() 

`# Printing the result`
print('The standard deviation of the dataset: ' + str(standard_deviation), '%')
```

标准差的输出应该如下：

```py
The standard deviation of the dataset: 1.9039 %
```

你应该如何解释标准偏差？平均而言，CPI 同比值往往与同一时期的平均值相差±1.90%，该平均值为 2.48%。

在接下来的部分中，您将看到如何更好地利用标准偏差数字。本节中的最后一个变异性度量是范围。*范围*是一个非常简单的统计量，显示数据集中最大值和最小值之间的距离。这让你快速了解两个历史极端值。范围在您将在后面章节中看到的归一化公式中使用。查找范围的公式如下：

<math alttext="upper R a n g e equals m a x left-parenthesis x right-parenthesis minus m i n left-parenthesis x right-parenthesis"><mrow><mi>R</mi> <mi>a</mi> <mi>n</mi> <mi>g</mi> <mi>e</mi> <mo>=</mo> <mi>m</mi> <mi>a</mi> <mi>x</mi> <mo>(</mo> <mi>x</mi> <mo>)</mo> <mo>-</mo> <mi>m</mi> <mi>i</mi> <mi>n</mi> <mo>(</mo> <mi>x</mi> <mo>)</mo></mrow></math>

让我们以相同的例子计算范围。在 Python 中，您可以轻松地做到这一点，因为有内置函数可以显示给定数据集的最大值和最小值：

```py
`# Calculating the range of the dataset`
range_metric = max(cpi["CPIAUCSL"]) - min(cpi["CPIAUCSL"])

`# Printing the result`
print('The range of the dataset: ' + str(range_metric), '%')

```

以下代码的输出应该如下所示：

```py
The range of the dataset: 16.5510 %
```

图 3-3 显示了自 1950 年以来的 CPI 值。对角虚线代表范围。

![](img/dlf_0224.png)

###### 图 3-3\. 自 1950 年以来的美国 CPI 同比变化，带有代表范围的对角虚线

CPI 的范围显示了通货膨胀测量值在 30 年内从一个时期到另一个时期的变化大小。不同国家的通货膨胀数字的年度变化各不相同。一般来说，发达国家如法国和美国在稳定时期具有稳定的变化，而阿根廷和土耳其等新兴和前沿世界国家的通货膨胀数字更加波动和极端。

###### 注意

在继续下一节时，请牢记以下要点：

+   您应该了解的三个关键变异性度量是方差、标准偏差和范围。

+   标准偏差是方差的平方根。这样做是为了使其可与均值进行比较。

+   范围是数据集中最高值和最低值之间的差异。它是对观察的整体波动性的快速快照。

## 形状度量

形状度量描述数据集中围绕中心趋势度量的值的分布。

均值和标准偏差是描述正态分布的两个因素。标准偏差描述了数据的传播或分散，而均值反映了分布的中心。

*概率分布*是描述随机实验中不同结果或事件发生可能性的数学函数。换句话说，它给出了随机变量所有可能值的概率。

有许多类型的概率分布，包括离散和连续分布。*离散分布*只取有限数量的值。最知名的离散分布是伯努利分布、二项分布和泊松分布。

*连续分布*用于可以在给定范围内取任意值的随机变量（如股票价格）。最知名的连续分布是正态分布。

正态分布（也称为高斯分布）是一种连续概率分布，围绕均值对称，并具有钟形。它是统计分析中最广泛使用的分布之一，通常用于描述自然现象，如年龄、体重和考试成绩。图 3-4 显示了正态分布的形状。

![](img/dlf_0280.png)

###### 图 3-4\. 均值=0，标准偏差=1 的正态分布图

您可以使用以下代码块生成图 3-4：

```py
`# Importing libraries`
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

`# Generate data for the plot`
data = np.linspace(-3, 3, num = 1000)

`# Define the mean and standard deviation of the normal distribution`
mean = 0
std = 1

`# Generate the function of the normal distribution`
pdf = stats.norm.pdf(data, mean, std)

`# Plot the normal distribution plot`
plt.plot(data, pdf, '-', color = 'black', lw = 2)
plt.axvline(mean, color = 'black', linestyle = '--')

`# Calling a grid to facilitate the visual component`
plt.grid()

`# Show the plot`
plt.show()
```

###### 注意

由于正态分布变量很常见，大多数统计测试和模型假设分析的数据是正态的。对于金融回报，即使它们经历了一种偏斜和峰度形式，也被假定为正态的，这是本节讨论的形状的两个度量。

在正态分布中，数据围绕平均值对称分布，这也意味着平均值等于中位数和众数。此外，约 68%的数据落在平均值的一个标准差范围内，约 95%落在两个标准差范围内，约 99.7%落在三个标准差范围内。这个属性使正态分布成为进行推断的有用工具。

总结一下，从正态分布中你应该记住以下内容：

+   平均值和标准差描述了分布。

+   平均值将分布一分为二，使其等于中位数。由于对称性属性，众数也等于平均值和中位数。

现在，让我们讨论形状的测量。形状的第一个测量是偏斜度。*偏斜度*描述了分布的不对称性。它分析了分布偏离对称的程度。

正如您可能已经了解的那样，正态分布的偏斜度等于零。这意味着分布在其平均值周围完全对称，平均值两侧的数据点数量相等。

*正偏斜*表示分布向右有一个长尾，这意味着平均值大于中位数，因为平均值对异常值敏感，这将使其向上推（因此，在 x 轴的右侧）。同样，代表最频繁观察的众数将是三个中心趋势测量值中的最小值。图 3-5 显示了一个正偏斜。 

![](img/dlf_11.png)

###### 图 3-5。一个正偏斜分布的示例

*负偏斜*表示分布向左有一个长尾，这意味着平均值低于中位数，原因在于正偏斜时提到的原因。同样，众数将是三个中心趋势测量值中的最大值。图 3-6 显示了一个负偏斜。

![](img/dlf_10.png)

###### 图 3-6。一个负偏斜分布的示例

###### 注意

在金融市场中如何解释偏斜？如果分布是正偏斜的，这意味着高于平均值的回报比低于平均值的回报更多（分布的尾部在正侧更长）。

另一方面，如果分布是负偏斜的，这意味着低于平均值的回报比高于平均值的回报更多（分布的尾部在负侧更长）。

回报系列的偏斜度可以提供有关投资的风险和回报的信息。例如，正偏斜的回报系列可能表明投资具有更高的高回报潜力，但也具有更大的大额损失风险。相反，负偏斜的回报系列可能表明投资具有较低的高回报潜力，但也具有较低的大额损失风险。

找到偏斜度的公式如下：

<math alttext="mu overTilde Subscript 3 Baseline equals StartFraction sigma-summation Underscript n equals 1 Overscript i Endscripts left-parenthesis x Subscript i Baseline minus x overbar right-parenthesis cubed Over upper N sigma cubed EndFraction"><mrow><msub><mover accent="true"><mi>μ</mi> <mo>˜</mo></mover> <mn>3</mn></msub> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mrow><msubsup><mo>∑</mo> <mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow> <mi>i</mi></msubsup> <msup><mrow><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>3</mn></msup></mrow> <mrow><mi>N</mi><msup><mi>σ</mi> <mn>3</mn></msup></mrow></mfrac></mstyle></mrow></math>

该公式将第三中心矩除以标准差的三次方。让我们检查美国 CPI 年度数据的偏斜度：

```py
`# Calculating the skew of the dataset`
skew = cpi["CPIAUCSL"].skew() 

`# Printing the result`
print('The skew of the dataset: ' + str(skew))

```

以下代码的输出应该如下所示：

```py
The skew of the dataset: 1.4639
```

数据的偏斜度为 1.46，但这意味着什么？让我们绘制数据的分布，以便解释变得更容易。您可以使用以下代码片段来做到这一点：

```py
`# Plotting the histogram of the data`
fig, ax = plt.subplots()
ax.hist(cpi['CPIAUCSL'], bins = 30, edgecolor = 'black', color = 'white')

`# Add vertical lines for better interpretation`
ax.axvline(mean, color='black', linestyle='--', label='Mean', linewidth = 2)
ax.axvline(median, color='grey', linestyle='-.', label='Median', linewidth = 2)

`# Calling the grid function for better interpretability`
plt.grid()

`# Calling the legend function to show the labels`
plt.legend()

`# Showing the plot`
plt.show()

```

图 3-7 显示了前一个代码片段的结果。数据明显是正偏斜的，因为平均值大于中位数，偏斜度为正（大于零）。

![](img/dlf_8.png)

###### 图 3-7。美国 CPI 年度数据的数据分布

记住，*偏斜度是概率分布不对称性的度量*。因此，它衡量分布偏离正态的程度。解释偏斜度的经验法则如下：

+   如果偏斜度在-0.5 和 0.5 之间，则数据被认为是对称的。

+   如果偏度在-1.0 和-0.5 之间或在 0.5 和 1.0 之间，数据被认为是轻微偏斜的。

+   如果偏度小于-1.0 或大于 1.0，则数据被认为是高度偏斜的

什么是正偏态？1.17 是高度偏斜的数据（在正面），这与一项偏向通货膨胀的货币政策相一致，随着经济增长（伴随着一些引起偏斜的通货膨胀性波动）。

###### 注意

有趣的是，对于偏斜分布，中位数是首选的度量标准，因为均值往往会被离群值拉动，从而扭曲其价值。

下一个形状度量是*峰度*，它是相对于正态分布的分布的尖峰或扁平程度的度量。峰度描述了分布的尾部，特别是尾部是否比正态分布的尾部更厚或更薄。数学上，峰度是第四中心矩除以标准差的第四次幂。

正态分布的峰度为 3，这意味着它是一个中峰分布。如果一个分布的峰度大于 3，则被称为尖峰分布，意味着它比正态分布有更高的峰值和更厚的尾部。如果一个分布的峰度小于 3，则被称为扁峰分布，意味着它比正态分布有更平的峰值和更薄的尾部。

计算峰度的公式如下：

<math alttext="k equals StartFraction sigma-summation Underscript n equals 1 Overscript i Endscripts left-parenthesis x Subscript i Baseline minus x overbar right-parenthesis Superscript 4 Baseline Over upper N sigma Superscript 4 Baseline EndFraction"><mrow><mi>k</mi> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mrow><msubsup><mo>∑</mo> <mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow> <mi>i</mi></msubsup> <msup><mrow><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>4</mn></msup></mrow> <mrow><mi>N</mi><msup><mi>σ</mi> <mn>4</mn></msup></mrow></mfrac></mstyle></mrow></math>

有时，峰度被测量为超额峰度，以使其起始值为零（对于正态分布）。这意味着将峰度测量值减去 3，以计算超额峰度。让我们计算美国 CPI 年度数据的超额峰度：

```py
`# Calculating the excess kurtosis of the dataset`
excess_kurtosis = cpi["CPIAUCSL"].kurtosis() 

`# Printing the result`
print('The excess kurtosis of the dataset: ' + str(excess_kurtosis))

```

以下代码的输出应该如下：

```py
The excess kurtosis of the dataset: 2.2338
```

从`pandas`获得的超额峰度在正态分布的情况下应为零。在美国 CPI 年度数据的情况下，它为 2.23，这更符合尖峰（峰值更高，尾部更厚）分布。正值表示分布比正常更尖，负峰度表示形状比正常更扁。

###### 注意

独立于统计学，了解你正在分析的术语是很有趣的。*通货膨胀*是经济主体（如家庭）购买力下降。购买力下降意味着经济主体随着时间的推移用同样的金额购买力购买力减少，也被称为一般价格上涨。经济意义上的通货膨胀有以下形式：

+   *通货膨胀*：受控通货膨胀与稳定的经济增长和扩张相关。这是一个增长经济体的理想属性。监管机构监测通货膨胀并试图稳定它，以防止社会和经济问题。

+   *通货紧缩*：每当通货膨胀处于负值领域时，就被称为通货紧缩。通货紧缩对经济非常危险，尽管对于看到价格下降的消费者来说可能很诱人，但通货紧缩是一种增长杀手，可能导致长期的经济过剩，导致失业和熊市股市。

+   *滞涨*：当通货膨胀要么很高要么上升，而经济增长放缓时发生。同时，失业率保持高位。这是最糟糕的情况之一。

+   *通货紧缩*：这是通货膨胀下降但仍处于正值领域。例如，如果今年的通货膨胀率为 2%，而去年的通货膨胀率为 3%，你可以说每年都存在通货紧缩情况。

+   *超级通货膨胀*：这是通货膨胀失控并经历百分比变化如百万年度变化的噩梦般情景（著名案例包括津巴布韦、南斯拉夫和希腊）。

最后，在描述性统计部门中看到的最后一个指标是分位数。*分位数*是形状和变异性的度量，因为它们提供有关值的分布（形状）的信息，并提供有关这些值的离散程度（变异性）的信息。最常用的分位数类型称为四分位数。

*四分位数*将数据集分成四个相等的部分。这是通过对数据进行排序然后执行拆分来完成的。以表 3-1 为例：

| 值 |
| --- |
| 1 |
| 2 |
| 4 |
| 5 |
| 7 |
| 8 |
| 9 |

四分位数如下：

+   下四分位数（Q1）是第一四分之一，在这种情况下是 2。

+   中位数（Q2）也是中位数，在这种情况下是 5。

+   在这种情况下，上四分位数（Q3）为 8。

从数学上讲，您可以使用以下公式计算 Q1 和 Q3：

<math alttext="upper Q 1 equals left-parenthesis StartFraction n plus 1 Over 4 EndFraction right-parenthesis"><mrow><msub><mi>Q</mi> <mn>1</mn></msub> <mo>=</mo> <mrow><mo>(</mo> <mfrac><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow> <mn>4</mn></mfrac> <mo>)</mo></mrow></mrow></math>

<math alttext="upper Q 3 equals 3 left-parenthesis StartFraction n plus 1 Over 4 EndFraction right-parenthesis"><mrow><msub><mi>Q</mi> <mn>3</mn></msub> <mo>=</mo> <mn>3</mn> <mrow><mo>(</mo> <mfrac><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow> <mn>4</mn></mfrac> <mo>)</mo></mrow></mrow></math>

请记住，公式的结果给出的是值的排名，而不是值本身：

<math alttext="upper Q 1 equals left-parenthesis StartFraction 7 plus 1 Over 4 EndFraction right-parenthesis equals 2 Superscript n d Baseline t e r m equals 2"><mrow><msub><mi>Q</mi> <mn>1</mn></msub> <mo>=</mo> <mrow><mo>(</mo> <mfrac><mrow><mn>7</mn><mo>+</mo><mn>1</mn></mrow> <mn>4</mn></mfrac> <mo>)</mo></mrow> <mo>=</mo> <msup><mn>2</mn> <mrow><mi>n</mi><mi>d</mi></mrow></msup> <mi>t</mi> <mi>e</mi> <mi>r</mi> <mi>m</mi> <mo>=</mo> <mn>2</mn></mrow></math>

<math alttext="upper Q 3 equals 3 left-parenthesis StartFraction 7 plus 1 Over 4 EndFraction right-parenthesis equals 6 Superscript t h Baseline t e r m equals 8"><mrow><msub><mi>Q</mi> <mn>3</mn></msub> <mo>=</mo> <mn>3</mn> <mrow><mo>(</mo> <mfrac><mrow><mn>7</mn><mo>+</mo><mn>1</mn></mrow> <mn>4</mn></mfrac> <mo>)</mo></mrow> <mo>=</mo> <msup><mn>6</mn> <mrow><mi>t</mi><mi>h</mi></mrow></msup> <mi>t</mi> <mi>e</mi> <mi>r</mi> <mi>m</mi> <mo>=</mo> <mn>8</mn></mrow></math>

*四分位距*（IQR）是 Q3 和 Q1 之间的差异，并提供了数据集中间 50%值的传播度量。IQR 对异常值具有鲁棒性（因为它依赖于中间值），并提供了对大部分值传播的简要摘要。根据以下公式，表 3-1 中的数据的 IQR 为 6：

<math alttext="upper I upper Q upper R equals upper Q 3 minus upper Q 1"><mrow><mi>I</mi> <mi>Q</mi> <mi>R</mi> <mo>=</mo> <msub><mi>Q</mi> <mn>3</mn></msub> <mo>-</mo> <msub><mi>Q</mi> <mn>1</mn></msub></mrow></math>

<math alttext="upper I upper Q upper R equals 8 minus 2 equals 6"><mrow><mi>I</mi> <mi>Q</mi> <mi>R</mi> <mo>=</mo> <mn>8</mn> <mo>-</mo> <mn>2</mn> <mo>=</mo> <mn>6</mn></mrow></math>

IQR 是一个有价值的指标，可以用作许多不同模型中的输入或风险度量。它还可以用于检测数据中的异常值，因为它不受其影响。此外，IQR 可以帮助评估所分析资产的当前波动性，进而可以与其他方法一起使用以创建更强大的模型。正如所理解的那样，IQR 在有用性和解释性方面优于范围度量，因为前者容易受到异常值的影响。

在计算四分位数时要小心，因为有许多方法使用不同的计算来处理相同的数据集。最重要的是在整个分析过程中始终使用一致的方法。在表 3-1 中用于计算四分位数的方法称为*图基的铰链*方法。默认情况下，当您想使用`pandas`计算四分位数时，默认方法是*线性插值*方法，这将给出不同的结果。

这些方法之间的主要区别在于有些方法可能更适合较小的数据集或具有不同分布特征的正常大小数据集。

###### 注意

本节的关键要点如下：

+   正态分布是具有钟形曲线的连续概率分布。大多数数据聚集在平均值周围。正态分布曲线的均值、中位数和众数都相等。

+   偏度度量概率分布的不对称性。

+   峰度度量概率分布的尖峰度。过度峰度通常用于描述当前概率分布。

+   分位数将排列的数据集分成相等的部分。最著名的分位数是将数据分成四个相等部分的四分位数。

+   IQR 是第三四分位数和第一四分位数之间的差异。它不受异常值的影响，因此在数据分析中非常有帮助。

## 可视化数据

如果您还记得前一章，我提出了数据科学中的六个阶段过程。第四阶段涉及数据可视化。本节将向您展示几种以清晰的视觉方式呈现数据的方法，使您能够解释数据。

有许多常用于可视化数据的统计图类型，例如散点图和折线图。让我们讨论这些图并使用相同的通货膨胀数据创建它们。

第一种数据可视化方法是*散点图*，用于通过对应于变量交点的点来绘制两个变量之间的关系。让我们使用以下代码创建和可视化一个散点图：

```py
`# Importing the required library`
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt

`# Setting the beginning and end of the historical data`
start_date = '1950-01-01'
end_date   = '2022-12-01'

`# Creating a dataframe and downloading the CPI data using its code name and its source`
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

`# Transforming the CPI into a year-on-year measure`
cpi = cpi.pct_change(periods = 12, axis = 0) * 100

`# Dropping the NaN values`
cpi = cpi.dropna()

`# Resetting the index`
cpi = cpi.reset_index()

`# Creating the chart`
fig, ax = plt.subplots()
ax.scatter(cpi['DATE'], cpi['CPIAUCSL'], color = 'black', s = 8,  label = 'Change in CPI Year-on-Year')

`# Calling the grid function for better interpretability`
plt.grid()

`# Calling the legend function to show the labels`
plt.legend()

`# Showing the plot`
plt.show()

```

图 3-8 显示了时间散点图的结果。这意味着您将 CPI 数据作为第一个变量（y 轴），时间作为第二个变量（x 轴）。然而，散点图更常用于比较变量，因此去除时间变量可以提供更多见解。

![](img/dlf_0228.png)

###### 图 3-8。美国 CPI 与时间轴的散点图

如果您拿英国 CPI 年同比变化与美国 CPI 年同比变化进行比较，您将得到图 3-9。请注意两者之间的正相关性，因为一个变量的较高值与另一个变量的较高值相关联。相关性是一个关键指标，您将在下一节中详细了解。绘制图 3-9 的代码如下：

```py
`# Setting the beginning and end of the historical data`
start_date = '1995-01-01'
end_date   = '2022-12-01'

`# Creating a dataframe and downloading the CPI data using its code name and its source`
cpi_us = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi_uk = pdr.DataReader('GBRCPIALLMINMEI', 'fred', start_date, end_date)

`# Dropping the NaN values from the rows`
cpi_us = cpi_us.dropna()
cpi_uk = cpi_uk.dropna()

`# Transforming the CPI into a year-on-year measure`
cpi_us = cpi_us.pct_change(periods = 12, axis = 0) * 100
cpi_us = cpi_us.dropna()

cpi_uk = cpi_uk.pct_change(periods = 12, axis = 0) * 100
cpi_uk = cpi_uk.dropna()

`# Creating the chart`
fig, ax = plt.subplots()
ax.scatter(cpi_us['CPIAUCSL'], cpi_uk['GBRCPIALLMINMEI'], color = 'black', s = 8, label = 'Change in CPI Year-on-Year')

`# Adding a few aesthetic elements to the chart`
ax.set_xlabel('US CPI')
ax.set_ylabel('UK CPI')
ax.axvline(x = 0, color='black', linestyle = 'dashed', linewidth = 1)  # vertical line
ax.axhline(y = 0, color='black', linestyle = 'dashed', linewidth = 1)  # horizontal line
ax.set_ylim(-2,)

`# Calling the grid function for better interpretability`
plt.grid()

`# Calling the legend function to show the labels`
plt.legend()

`# Showing the plot`
plt.show()

```

![](img/dlf_0229.png)

###### 图 3-9。英国 CPI 与美国 CPI 的散点图

散点图在可视化数据之间的相关性时很有用。它们也易于绘制和解释。通常，当点散布在这样一种方式时，即可以画出一个向上倾斜的对角线来代表它们，就假定相关性是正的，因为每当 x 轴上的变量增加时，y 轴上的变量也会增加。

另一方面，当可以画出向下倾斜的对角线来代表不同变量时，可能存在负相关性。负相关意味着每当 x 轴上的变量移动时，y 轴上的变量可能会以相反的方式移动。图 3-10 绘制了两个通货膨胀数据之间的最佳拟合线（通过代码生成）。请注意它是向上倾斜的：

![](img/dlf_4.png)

###### 图 3-10。英国 CPI 与美国 CPI 的散点图及最佳拟合线

现在让我们转向另一种图表方法。*线图*基本上是散点图，它们被连接在一起，大多数情况下绘制在时间轴（x 轴）上。您已经在之前的图表中看到过线图，比如图 3-1 和图 3-2，因为它是绘图的最基本形式。

线图的优点在于其简单性和易于实现。它们还显示了系列随时间的演变，有助于检测趋势和模式。在第五章中，您将看到一个更复杂的金融时间序列绘图版本，称为*蜡烛图*。图 3-11 显示了自 1950 年以来美国 CPI 的基本线图：

![](img/dlf_1.png)

###### 图 3-11。美国 CPI 与时间轴的线图

要创建图 3-11，您可以使用以下代码片段：

```py
`# Setting the beginning and end of the historical data`
start_date = '1950-01-01'
end_date   = '2022-12-01'

`# Creating a dataframe and downloading the CPI data using its code name and its source`
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

`# Transforming the CPI into a year-on-year measure`
cpi = cpi.pct_change(periods = 12, axis = 0) * 100

`# Dropping the NaN values`
cpi = cpi.dropna()

`# Resetting the index`
cpi = cpi.reset_index()

`# Creating the chart`
plt.plot(cpi['DATE'], cpi['CPIAUCSL'], color = 'black', label = 'Change in CPI Year-on-Year')

`# Calling the grid function for better interpretability`
plt.grid()

`# Calling the legend function to show the labels`
plt.legend()

`# Showing the plot`
plt.show()

```

接下来是*条形图*，用于显示变量（通常是分类变量）的分布。图 3-12 显示了自 2022 年初以来美国 CPI 的条形图：

![](img/dlf_0778.png)

###### 图 3-12。美国 CPI 与时间轴的条形图

要创建图 3-12，您可以使用以下代码片段：

```py
`# Taking the values of the previous twelve months`
cpi_one_year = cpi.iloc[-12:]

`# Creating the chart`
plt.bar(cpi_one_year['DATE'], cpi_one_year['CPIAUCSL'], color = 'black', label = 'Change in CPI Year-on-Year', width = 7)

`# Calling the grid function for better interpretability`
plt.grid()

`# Calling the legend function to show the labels`
plt.legend()

`# Showing the plot`
plt.show()

```

条形图易于实现，功能多样。然而，对于绘制美国 CPI 或股票价格等连续数据，它们可能存在局限性。当比例失调时，它们也可能具有误导性。由于它们会占用空间，因此不建议在大型数据集上使用条形图。基于后一原因，直方图更适合。

*直方图*是一种特定类型的条形图，用于通过使用条形来表示统计信息来显示连续数据的频率分布。它指示落入值类别或区间的观测数量。直方图的一个示例是图 3-13（以及上一节的图 3-7）：

![](img/dlf_2.png)

###### 图 3-13。美国 CPI 的直方图

```py
# Creating the chart
fig, ax = plt.subplots()
ax.hist(cpi['CPIAUCSL'], bins = 30, edgecolor = 'black', color = 'white', label = 'Change in CPI Year-on-Year',)

# Add vertical lines for better interpretation
ax.axvline(0, color='black')

# Calling the grid function for better interpretability
plt.grid()

# Calling the legend function to show the labels
plt.legend()

# Showing the plot
plt.show()

```

请注意，条形图是根据时间轴绘制的，而直方图没有时间范围，因为它是一组值，旨在显示整体分布点。从视觉上，您可以看到分布的正偏度。

###### 注意

*分类变量*的一个示例是性别，而*连续变量*的一个示例是商品价格。

统计学中另一种经典绘图技术是著名的*箱线图*。它用于可视化连续变量的分布，同时包括中位数和四分位数，以及异常值。理解箱线图的方法如下：

+   箱子代表 IQR。箱子在第一四分位数和第三四分位数之间绘制。箱子的高度表示该范围内数据的传播。

+   箱子内部的线代表中位数。

+   须在箱子的顶部和底部延伸到仍在 1.5 倍 IQR 内的最高和最低数据点。这些数据点称为*异常值*，并在图中表示为单独的点。

图 3-14 显示了自 1950 年以来美国 CPI 的箱线图：

![](img/dlf_080.png)

###### 图 3-14。美国 CPI 的箱线图

您还可以绘制没有异常值的图（任何值距离箱子的任一端超过箱子长度的一倍半）。要创建图 3-14，您可以使用以下代码片段：

```py
`# Taking the values of the last twenty years`
cpi_last_ten_years = cpi.iloc[-240:]

`# Creating the chart`
fig, ax = plt.subplots()
ax.boxplot(cpi_last_ten_years['CPIAUCSL'])

`# Calling the grid function for better interpretability`
plt.grid()

`# Calling the legend function to show the labels`
plt.legend()

`# Showing the plot`
plt.show()

```

要从图中去除异常值，您只需使用以下调整：

```py
`# Replace the corresponding code line with the following`
ax.boxplot(cpi_last_ten_years['CPIAUCSL'], showfliers = False)
```

这将给您图 3-15：

![](img/dlf_0.81.png)

###### 图 3-15。美国 CPI 的箱线图，没有异常值

存在许多其他数据可视化技术，例如*热图*（通常与相关数据和温度映射一起使用）和*饼图*（通常用于预算和分割）。这完全取决于您需要理解的内容以及哪种更适合您的需求。例如，线图更适合只有一个特征的时间序列（例如，只有某种证券的收盘价可用）。直方图图更适合概率分布数据。

###### 注意

让我们总结一下您需要记住的一切：

+   数据可视化取决于您想要进行的分析和解释的类型。某些图适合某些类型的数据。

+   在数值上确认之前，数据可视化有助于对数据进行初步解释。

+   处理金融时间序列时，您更有可能使用线图和蜡烛图。

## 相关性

*相关性*是用于计算两个变量之间线性关系程度的度量。它是一个介于-1.0 和 1.0 之间的数字，-1.0 表示变量之间有强烈的负相关关系，而 1.0 表示有强烈的正相关关系。

零值表示变量之间没有线性关联。然而，相关性并不意味着因果关系。如果两个变量朝着同一个方向移动，则它们被认为是相关的，但这并不意味着一个导致另一个移动，或者它们是由相同事件的结果移动的。

大多数人都同意一些资产具有自然的相关性。例如，由于它们都属于同一行业，并受到相同趋势和事件的影响，苹果和微软的股票是正相关的（这意味着它们的总体趋势是相同的）。图 3-16 显示了两只股票之间的图表。注意它们如何一起移动。

![](img/dlf_0569.png)

###### 图 3-16。自 2021 年以来的苹果和微软股价​

两只股票的顶部和底部几乎同时出现。同样，由于美国和英国具有相似的经济驱动因素和影响，它们也很可能有正相关的通货膨胀数字。

通过视觉解释和数学公式来检查相关性。在看一个例子之前，让我们深入了解计算相关性的根源，这样您就知道它来自哪里以及它的局限性是什么。

###### 注意

简而言之，要计算相关性，您需要测量两个变量散点图中的点与一条直线的接近程度。它们看起来越像一条直线，它们就越正相关，因此有了术语“线性相关”。

有两种主要的计算相关性的方法，即使用 Spearman 方法或 Pearson 方法。

皮尔逊相关系数是从两个变量的标准差和协方差计算出来的线性关联度量。但是，什么是协方差？

*协方差*计算两个变量均值之间的差异的平均值。如果两个变量有一起移动的倾向，协方差是正的，如果两个变量通常朝相反方向移动，协方差是负的。它的范围在无穷大和负无穷大之间，接近零的值表示没有线性相关。

计算变量*x*和*y*之间的协方差的公式如下：

<math alttext="c o v Subscript x y Baseline equals StartFraction sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis x Subscript i Baseline minus x overbar right-parenthesis left-parenthesis y Subscript i Baseline minus y overbar right-parenthesis Over n EndFraction"><mrow><mi>c</mi> <mi>o</mi> <msub><mi>v</mi> <mrow><mi>x</mi><mi>y</mi></mrow></msub> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mrow><msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></msubsup> <mrow><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow><mrow><mo>(</mo><msub><mi>y</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>y</mi> <mo>¯</mo></mover><mo>)</mo></mrow></mrow> <mi>n</mi></mfrac></mstyle></mrow></math>

因此，协方差是变量之间的平均偏差乘以它们各自的均值的总和（衡量它们之间的关联程度）。取平均值以规范化此计算。皮尔逊相关系数的计算如下：

<math alttext="r Subscript x y Baseline equals StartFraction sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis x Subscript i Baseline minus x overbar right-parenthesis left-parenthesis y Subscript i Baseline minus y overbar right-parenthesis Over StartRoot sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis x Subscript i Baseline minus x overbar right-parenthesis squared EndRoot StartRoot sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis y Subscript i Baseline minus y overbar right-parenthesis squared EndRoot EndFraction"><mrow><msub><mi>r</mi> <mrow><mi>x</mi><mi>y</mi></mrow></msub> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mrow><msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></msubsup> <mrow><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow><mrow><mo>(</mo><msub><mi>y</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>y</mi> <mo>¯</mo></mover><mo>)</mo></mrow></mrow> <mrow><msqrt><mrow><msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></msubsup> <msup><mrow><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup></mrow></msqrt> <msqrt><mrow><msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></msubsup> <msup><mrow><mo>(</mo><msub><mi>y</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>y</mi> <mo>¯</mo></mover><mo>)</mo></mrow> <mn>2</mn></msup></mrow></msqrt></mrow></mfrac></mstyle></mrow></math>

简化前面的相关性公式得到以下结果：

<math alttext="r Subscript x y Baseline equals StartFraction c o v Subscript x y Baseline Over sigma Subscript x Baseline sigma Subscript y Baseline EndFraction"><mrow><msub><mi>r</mi> <mrow><mi>x</mi><mi>y</mi></mrow></msub> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="false"><mfrac><mrow><mi>c</mi><mi>o</mi><msub><mi>v</mi> <mrow><mi>x</mi><mi>y</mi></mrow></msub></mrow> <mrow><msub><mi>σ</mi> <mi>x</mi></msub> <msub><mi>σ</mi> <mi>y</mi></msub></mrow></mfrac></mstyle></mrow></math>

因此，皮尔逊相关系数简单地是两个变量之间的协方差除以它们标准差的乘积。让我们计算美国 CPI 年度变化和英国 CPI 年度变化之间的相关性。直觉告诉我们，由于经济上，英国和美国有关联，所以相关性大于零。以下代码块计算了这两个时间序列的皮尔逊相关系数：

```py
`# Importing the required libraries`
import pandas_datareader as pdr
import pandas as pd

`# Setting the beginning and end of the historical data`
start_date = '1995-01-01'
end_date   = '2022-12-01'

`# Creating a dataframe and downloading the CPI data using its code name and its source`
cpi_us = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi_uk = pdr.DataReader('GBRCPIALLMINMEI', 'fred', start_date, end_date)

`# Dropping the NaN values from the rows`
cpi_us = cpi_us.dropna()
cpi_uk = cpi_uk.dropna()

`# Transforming the US CPI into a year-on-year measure`
cpi_us = cpi_us.pct_change(periods = 12, axis = 0) * 100
cpi_us = cpi_us.dropna()

`# Transforming the UK CPI into a year-on-year measure`
cpi_uk = cpi_uk.pct_change(periods = 12, axis = 0) * 100
cpi_uk = cpi_uk.dropna()

`# Joining both CPI data into one dataframe`
combined_cpi_data = pd.concat([cpi_us['CPIAUCSL'], cpi_uk['GBRCPIALLMINMEI']], axis = 1)

`# Using pandas' correlation function to calculate the measure`
combined_cpi_data.corr(method = 'pearson')

```

输出如下：

```py
                 CPIAUCSL  GBRCPIALLMINMEI
CPIAUCSL         1.000000         0.732164
GBRCPIALLMINMEI  0.732164         1.000000
```

两者之间的相关性高达 0.73。这符合预期。皮尔逊相关通常用于具有成比例变化且正态分布的变量。这可能是一个问题，因为金融数据不是正态分布的。因此，讨论 Spearman 相关性是有趣的。

*Spearman 相关性*是一种非参数秩相关，用于衡量变量之间关系的强度。适用于不遵循正态分布的变量。

###### 注意

请记住，金融回报不是正态分布的，但有时为简单起见会这样处理。

与皮尔逊相关不同，Spearman 秩相关考虑的是值的顺序，而不是实际值。要计算 Spearman 相关性，请按照以下步骤进行：

1.  对每个变量的值进行排名。这是通过将最小变量替换为 1，将数据集的长度替换为最大数字来完成的。

1.  计算排名之间的差异。在数学上，排名之间的差异用数学公式中的字母*d*表示。然后，计算它们的平方差。

1.  将从步骤 2 中计算的平方差相加。

1.  使用以下公式计算 Spearman 相关性。

<math alttext="rho equals 1 minus StartFraction 6 sigma-summation Underscript i equals 1 Overscript n Endscripts d Subscript i Superscript 2 Baseline Over n cubed minus n EndFraction"><mrow><mi>ρ</mi> <mo>=</mo> <mn>1</mn> <mo>-</mo> <mfrac><mrow><mn>6</mn><msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></msubsup> <msubsup><mi>d</mi> <mrow><mi>i</mi></mrow> <mn>2</mn></msubsup></mrow> <mrow><msup><mi>n</mi> <mn>3</mn></msup> <mo>-</mo><mi>n</mi></mrow></mfrac></mrow></math>

与皮尔逊相关类似，Spearman 相关性也在-1.00 到 1.00 之间，并具有相同的解释。

###### 注意

强正相关通常大于 0.70，而强负相关通常小于-0.70。

以下代码块计算了这两个时间序列的 Spearman 秩相关系数：

```py
`# Importing the required libraries`
import pandas_datareader as pdr
import pandas as pd

`# Setting the beginning and end of the historical data`
start_date = '1995-01-01'
end_date   = '2022-12-01'

`# Creating a dataframe and downloading the CPI data using its code name and its source` cpi_us = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi_uk = pdr.DataReader('GBRCPIALLMINMEI', 'fred', start_date, end_date)

`# Dropping the NaN values from the rows`
cpi_us = cpi_us.dropna()
cpi_uk = cpi_uk.dropna()

`# Transforming the US CPI into a year-on-year measure`
cpi_us = cpi_us.pct_change(periods = 12, axis = 0) * 100
cpi_us = cpi_us.dropna()

`# Transforming the UK CPI into a year-on-year measure`
cpi_uk = cpi_uk.pct_change(periods = 12, axis = 0) * 100
cpi_uk = cpi_uk.dropna()

`# Joining both CPI data into one dataframe`
combined_cpi_data = pd.concat([cpi_us['CPIAUCSL'], cpi_uk['GBRCPIALLMINMEI']], axis = 1)

`# Using pandas' correlation function to calculate the measure`
combined_cpi_data.corr(method = 'spearman')

```

输出如下：

```py
                 CPIAUCSL  GBRCPIALLMINMEI
CPIAUCSL         1.000000         0.472526
GBRCPIALLMINMEI  0.472526         1.000000
```

在得到这种结果差异后，让我们回答一个非常重要的问题。为什么这两种度量结果如此不同？

首先要记住的是它们测量的内容。皮尔逊相关度量变量之间的线性关系（趋势），而 Spearman 秩相关度量单调趋势。单调一词指的是朝着同一方向移动，但不完全以相同的速率或幅度。此外，Spearman 相关性将数据转换为序数类型（通过排名），而不是使用实际值的皮尔逊相关性。

*自相关*（也称为串行相关）是一种统计方法，用于研究给定时间序列与滞后版本之间的关系。它通常用于通过数据中的模式（如季节性或趋势）来预测未来值。因此，自相关是值与先前值之间的关系。例如，将每天的微软股价与前一天进行比较，看看是否存在可辨识的相关性。从算法的角度来看，这可以在表 3-2 中表示：

表 3-1。滞后值表

|     t |     t-1 |
| --- | --- |
|  $     1.25 |  $     1.65 |
|  $     1.77 |  $     1.25 |
|  $     1.78 |  $     1.77 |
|  $     1.25 |  $     1.78 |
|  $     1.90 |  $     1.25 |

每行代表一个时间段。列*t*是当前价格，列*t-1*是放在代表当前时间的行上的先前价格。这在创建机器学习模型时是为了了解当前值与以往值之间在每个时间步骤（行）的关系。

正自相关经常出现在趋势资产中，并与持续性（趋势跟随）的概念相关联。另一方面，波动市场表现出负自相关，与反持续性（均值回归）的概念相关联。

###### 注意

短期相关性的度量通常使用价格回报而不是实际价格来计算。然而，可以直接利用价格来识别长期趋势。

以下代码块计算了美国 CPI 年同比的自相关性：

```py
`# Importing the required libraries`
import pandas_datareader as pdr
import pandas as pd

`# Setting the beginning and end of the historical data`
start_date = '1950-01-01'
end_date   = '2022-12-01'

`# Creating a dataframe and downloading the CPI data using its code name and its source` cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

`# Dropping the NaN values from the rows`
cpi = cpi.dropna()

`# Transforming the US CPI into a year-on-year measure`
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()

`# Transforming the data frame to a series structure`
cpi = cpi.iloc[:,0]

`# Calculating autocorrelation with a lag of 1`
print('Correlation with a lag of 1 = ', round(cpi.autocorr(lag = 1), 2))

`# Calculating autocorrelation with a lag of 6`
print('Correlation with a lag of 6 = ', round(cpi.autocorr(lag = 6), 2))

`# Calculating autocorrelation with a lag of 12`
print('Correlation with a lag of 12 = ', round(cpi.autocorr(lag = 12), 2))

```

滞后 12 意味着每个数据与十二个周期前的数据进行比较，然后计算一个计算度量。代码的输出如下：

```py
Correlation with a lag of 1 =  0.99
Correlation with a lag of 6 =  0.89
Correlation with a lag of 12 =  0.73
```

现在，在继续下一节之前，让我们回到信息论并讨论一个能够捕捉非线性关系的有趣相关系数。其中一种方式是最大信息系数（MIC）。

*最大信息系数*（MIC）是两个变量之间关联的非参数测量，旨在处理大型和复杂数据。它通常被视为传统相关性测量的更稳健的替代方法，如皮尔逊相关性和斯皮尔曼秩相关性。由*Reshef 等人*引入，MIC 使用了信息论中的概念，这些概念在第二章中已经介绍过。

MIC 通过计算列联表中的单元格数量来衡量两个变量之间的关联强度，这些单元格对于变量之间的关系是最具信息量的。MIC 值的范围从 0 到 1，数值越高表示关联越强。它可以处理高维数据，并且可以识别变量之间的非线性关系。然而，它是非方向性的，这意味着接近 1 的值仅表明两个变量之间存在强相关性，但并不说明相关性是正向还是负向。

###### 注意

换句话说，在将每个变量的范围划分为一组箱后，计算每个箱内两个变量之间的互信息。

然后，通过跨所有箱子中的最大互信息值来估计两个变量之间的关联强度。

让我们看一个展示 MIC 在检测非线性关系方面的强大性的实际示例。以下示例模拟了正弦和余弦时间序列。直观地看，从图 3-17 可以看出，两者之间存在滞后-领先关系。

![](img/dlf_0230.png)

###### 图 3-17。显示一种非线性关系形式的两个波浪序列的强度

以下 Python 代码片段创建了两个时间序列并绘制了图 3-17：

```py
`# Importing the required libraries`
import numpy as np
import matplotlib.pyplot as plt

`# Setting the range of the data`
data_range = np.arange(0, 30, 0.1)

`# Creating the sine and the cosine waves`
sine = np.sin(data_range)
cosine = np.cos(data_range)

`# Plotting`
plt.plot(sine, color = 'black', label = 'Sine Function')
plt.plot(cosine, color = 'grey', linestyle = 'dashed', label = 'Cosine Function')
plt.grid()
plt.legend()

```

现在，任务是计算三个相关性度量并分析它们的结果。可以使用以下代码完成这项工作：

```py
`# Importing the libraries`
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from minepy import MINE

`# Calculating the linear correlation measures`
print('Correlation | Pearson: ', round(pearsonr(sine, cosine)[0], 3))
print('Correlation | Spearman: ', round(spearmanr(sine, cosine)[0], 3))

`# Calculating the MIC`
mine = MINE(alpha = 0.6, c = 15)
mine.compute_score(sine,cosine)
MIC = mine.mic()
print('Correlation | MIC: ', round(MIC, 3))

```

请注意，由于代码创建了一个数组（而不是数据框），在计算测量之前必须导入所需的库。这将在下一章中明确说明。以下是代码的输出：

```py
Correlation | Pearson:  0.035
Correlation | Spearman:  0.027
Correlation | MIC: 0.602
```

让我们解释一下结果：

+   *Pearson 相关性*：由于缺少非线性关联，这里没有任何类型的相关性。

+   *Spearman 相关性*：在这里也适用于极弱的相关性。

+   *MIC*：该测量返回了两者之间 0.60 的强关系，更接近现实。看起来 MIC 表明这两个波浪之间有强关系，尽管是非线性的。

###### 注意

您可能需要更新 Microsoft Visual C++（至少版本 14.0 或更高版本）以避免在尝试运行`minepy`库时出现任何错误。

如果正确使用，MIC 在经济分析、金融分析甚至寻找交易信号中非常有用。在这些复杂领域中非线性关系很常见，能够检测到这些关系可能会带来可观的优势。

###### 注意

相关性部分的主要要点如下：

+   相关性是用于计算两个变量之间线性关系程度的度量。它是一个介于-1.0 和 1.0 之间的数字，-1.0 表示变量之间强烈的负相关关系，1.0 表示变量之间强烈的正相关关系。

+   有两种主要类型的相关性测量，Spearman 和 Pearson。它们都有各自的优点和局限性。

+   自相关是变量与其自身滞后值的相关性。例如，如果 Nvidia 的股票回报的自相关是正的，则表示趋势配置。

+   当您使用正确的工具时，相关性测量也可以指非线性关系，例如 MIC。

## 平稳性概念

平稳性是统计分析和机器学习中的关键概念之一。*平稳性*是指时间序列的统计特征（均值、方差等）随时间保持恒定。换句话说，当沿时间绘制数据时，没有可辨认的趋势。

不同的学习模型依赖于数据的平稳性，因为这是统计建模的基础之一，主要是为了简化。在金融领域，价格时间序列不是平稳的，因为它们显示具有不同方差（波动性）的趋势。看一下图 3-18，看看您能否检测到趋势。您会说这个时间序列是平稳的吗？

![](img/dlf_0211.png)

###### 图 3-18。模拟数据随时间变化的均值

自然地，答案是否定的，因为一个上升趋势明显正在进行中。这种状态对于统计分析和机器学习是不可取的。幸运的是，您可以对时间序列应用转换使其平稳。但首先，让我们看看如何以数学方式检查平稳性，因为视觉方式并不能证明任何事情。处理数据平稳性问题的正确过程是遵循以下步骤：

1.  使用本节中将看到的不同统计测试来检查平稳性。

1.  如果测试显示数据平稳性，您就可以准备使用数据进行不同的学习算法。如果测试显示数据不平稳，您必须继续进行下一步。

1.  应用您将在本节中看到的价格转换技术。

1.  使用相同测试在新转换的数据上重新检查平稳性。

1.  如果测试显示数据平稳性，那么您已成功转换了数据。否则，请重新进行转换并再次检查，直到您获得平稳数据。

###### 注意

升序或降序时间序列随时间变化的均值和方差，因此很可能是非平稳的。当然也有例外情况，稍后您将看到原因。

请记住，平稳性的目标是随时间稳定和恒定的均值和方差。因此，当您看图 3-19 时，您能说些什么？

![](img/dlf_0210.png)

###### 图 3-19。模拟数据在时间上均值围绕零

从视觉上看，数据似乎没有趋势，看起来围绕着一个稳定的均值波动，具有稳定的方差。第一印象是数据是平稳的。当然，这必须通过统计测试来证明。

第一个和最基本的测试是*增广迪基-富勒*（ADF）测试。ADF 测试使用假设检验来检验平稳性。

ADF 测试在数据中寻找单位根。*单位根*是非平稳数据的一个特性，在时间序列分析的背景下，指的是随机过程的一个特征，其中系列的根等于 1。简单来说，这意味着其统计特性，如均值和方差，会随时间变化。以下是你需要知道的内容：

+   零假设假定存在单位根。这意味着如果你试图证明数据是平稳的，你要拒绝零假设（如第二章的假设检验部分所示）。

+   因此，备择假设是不存在单位根，数据是平稳的。

+   从测试中获得的 p 值必须小于所选的显著性水平（在大多数情况下，为 5%）。

让我们拿美国 CPI 的年度数据进行平稳性测试。以下代码片段使用 ADF 测试检查平稳性：

```py
`# Importing the required libraries`
from statsmodels.tsa.stattools import adfuller
import pandas_datareader as pdr

`# Setting the beginning and end of the historical data`
start_date = '1950-01-01'
end_date   = '2022-12-01'

`# Creating a dataframe and downloading the CPI data using its code name and its source` cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

`# Dropping the NaN values from the rows`
cpi = cpi.dropna()

`# Transforming the US CPI into a year-on-year measure`
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()

`# Applying the ADF test on the CPI data`
adfuller(cpi) 
print('p-value: %f' % adfuller(cpi)[1])

```

代码的输出如下：

```py
p-value: 0.0152
```

假设 5%的显著性水平，似乎可以接受年度数据是平稳的（然而，如果你想更严格一些，使用 1%的显著性水平，那么 p 值表明数据是非平稳的）。无论如何，即使看图表也会让你感到困惑。请记住，在图 3-11 中，美国 CPI 的年度变化似乎是稳定的，但并不像是平稳数据。这就是为什么要使用数值和统计测试的原因。

现在，让我们做同样的事情，但省略年度变化。换句话说，对美国 CPI 原始数据应用代码，不考虑年度变化。以下是代码：

```py
`# Importing the required libraries`
from statsmodels.tsa.stattools import adfuller
import pandas_datareader as pdr

`# Setting the beginning and end of the historical data`
start_date = '1950-01-01'
end_date   = '2022-12-01'

`# Creating a dataframe and downloading the CPI data using its code name and its source` cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

`# Dropping the NaN values from the rows`
cpi = cpi.dropna()

`# Applying the ADF test on the CPI data`
adfuller(cpi) 
print('p-value: %f' % adfuller(cpi)[1])

```

代码的输出如下：

```py
p-value: 0.999

```

显然，p 值大于所有显著性水平，这意味着时间序列是非平稳的。让我们总结一下这些结果：

+   似乎在美国 CPI 的年度变化方面，可以使用 5%的显著性水平拒绝零假设。数据集被假定为平稳。

+   似乎在美国 CPI 的原始数值方面，无法使用 5%的显著性水平拒绝零假设。数据集被假定为非平稳。

当你绘制美国 CPI 的原始数值时，这一点变得明显，如图 3-20 所示。

![](img/dlf_5.png)

###### 图 3-20。美国 CPI 的绝对数值显示出明显的趋势性

你必须了解的另一个测试被称为*Kwiatkowski-Phillips-Schmidt-Shin*（KPSS），这也是一个旨在确定时间序列是平稳还是非平稳的统计测试。然而，KPSS 测试可以检测趋势时间序列中的平稳性，使其成为一个强大的工具。

趋势时间序列实际上可以是平稳的，条件是它们具有稳定的均值。

# 警告

ADF 测试有一个假设检验，主张非平稳性，另一个假设主张平稳性。另一方面，KPSS 测试有一个假设主张平稳性，另一个假设主张非平稳性。

在分析通货膨胀数据之前，让我们看看趋势时间序列如何可以是平稳的。记住，平稳性指的是稳定的均值和标准差，所以如果你有一个逐渐上升或下降的时间序列，具有稳定的统计特性，它可能是平稳的。下一个代码片段模拟了一个正弦波，然后加入了一丝趋势性：

```py
`# Importing the required libraries`
import numpy as np
import matplotlib.pyplot as plt

`# Creating the first time series using sine waves`
length = np.pi * 2 * 5
sinewave = np.sin(np.arange(0, length, length / 1000))

`# Creating the second time series using trending sine waves`
sinewave_ascending = np.sin(np.arange(0, length, length / 1000))

`# Defining the trend variable`
a = 0.01

`# Looping to add a trend factor`
for i in range(len(sinewave_ascending)):

    sinewave_ascending[i] = a + sinewave_ascending[i]

    a = 0.01 + a

```

如图 3-21 所示绘制两个系列的图表显示了趋势正弦波似乎是稳定的。但让我们通过统计检验来证明这一点。

![](img/dlf_6.png)

###### 图 3-21. 具有趋势正弦波的正常正弦波模拟系列

图 3-21 是使用以下代码生成的（确保您已经使用前面的代码块定义了系列）：

```py
`# Plotting the series`
plt.plot(sinewave, label = 'Sine Wave', color = 'black')
plt.plot(sinewave_ascending, label = 'Ascending Sine Wave', color = 'grey')

`# Calling the grid function for better interpretability`
plt.grid()

`# Calling the legend function to show the labels`
plt.legend()

`# Showing the plot`
plt.show()

```

让我们对两个系列进行 ADF 测试，看看结果如何：

```py
`# ADF testing | Normal sine wave`
adfuller(sinewave) 
print('p-value: %f' % adfuller(sinewave)[1])

`# ADF testing | Ascending sine wave`
adfuller(sinewave_ascending) 
print('p-value: %f' % adfuller(sinewave_ascending)[1])

```

输出如下：

```py
p-value: 0.000000 `# For the sine wave`
p-value: 0.898635 `# For the ascending sine wave`

```

显然，ADF 测试与趋势市场不能是平稳的观念一致。但 KPSS 测试呢？以下代码使用 KPSS 在相同数据上检查平稳性：

```py
`# Importing the KPSS library`
from statsmodels.tsa.stattools import kpss

`# KPSS testing | Normal sine wave`
kpss(sinewave) 
print('p-value: %f' % kpss(sinewave)[1])

`# KPSS testing | Ascending sine wave`
kpss(sinewave_ascending) 
print('p-value: %f' % kpss(sinewave_ascending)[1])

`# KPSS testing while taking into account the trend | Ascending sine wave`
kpss(sinewave_ascending, regression = 'ct') 
print('p-value: %f' % kpss(sinewave_ascending, regression = 'ct')[1])

`'''
The 'ct' argument is used to check if the dataset is stationary 
around a trend. By default, the argument is 'c' which is is used
to check if the data is stationary around a constant.
'''`

```

输出如下：

```py
p-value: 0.10 `# For the sine wave`
p-value: 0.01 `# For the ascending sine wave without trend consideration`
p-value: 0.10 `# For the ascending sine wave with trend consideration`

```

请记住，KPSS 测试的零假设是数据是平稳的，因此如果 p 值大于显著性水平，则数据被认为是平稳的，因为无法拒绝零假设。

考虑趋势时，KPSS 统计量表明上升的正弦波是一个平稳的时间序列。这是一个基本示例，展示了如何在趋势时间序列中找到平稳数据。

让我们拿美国 CPI 年度数据来检验其平稳性。以下代码片段使用 KPSS 测试检查平稳性：

```py
`# Importing the required libraries`
from statsmodels.tsa.stattools import kpss
import pandas_datareader as pdr

`# Setting the beginning and end of the historical data`
start_date = '1950-01-01'
end_date   = '2022-12-01'

`# Creating a dataframe and downloading the CPI data using its code name and its source` cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Dropping the NaN values from the rows
cpi = cpi.dropna()

`# Transforming the US CPI into a year-on-year measure`
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()

`# Applying the KPSS (no trend consideration) test on the CPI data`
kpss(cpi) 
print('p-value: %f' % kpss(cpi)[1])

`# Applying the KPSS (with trend consideration) test on the CPI data`
kpss(cpi, regression = 'ct') 
print('p-value: %f' % kpss(cpi, regression = 'ct')[1])

```

代码的输出如下：

```py
p-value: 0.036323 `# without trend consideration`
p-value: 0.010000 `# with trend consideration`

```

看起来 KPSS 测试的结果与 ADF 测试的结果相矛盾。这种情况可能会时有发生，差分可能会解决问题（请记住，年度数据已经是绝对 CPI 值的差分时间序列，但有些时间序列可能需要多次差分才能变得平稳，这也取决于差分的周期）。在矛盾情况下，最安全的解决方案是再次转换数据。

在结束关于平稳性的本节之前，让我们讨论一个复杂的主题，您将在第七章中看到其实际应用。转换数据可能会导致一个不寻常的问题，即*内存丢失*。在他的书中，*Marco Lopez de Prado*提出了一种称为分数差分的技术，旨在使数据保持平稳同时保留一些记忆。

当对非平稳时间序列进行差分以使其平稳时，会发生内存丢失，这另一种说法是值之间的自相关大大减少，从而消除了趋势成分和基础资产的 DNA。差分的程度和原始系列中自相关结构的持久性决定了有多少内存丢失。

###### 注意

本节介绍了许多复杂的概念。您应该记住以下内容：

+   平稳性是指随着时间稳定的均值和方差的概念。大多数机器学习模型都依赖于这一特性。

+   金融价格时间序列很可能是非平稳的，需要进行一阶转换才能变得平稳并准备好进行统计建模。有些甚至可能需要进行二阶转换才能变得平稳。

+   ADF 和 KPSS 测试检查数据中的平稳性，后者能够检查趋势数据中的平稳性，因此更加彻底。

+   趋势数据可能是平稳的。尽管这种特性很少见，但 KPSS 能够检测到平稳性，与 ADF 测试相反。

## 回归分析和统计推断

*推断*，如牛津语言所定义，是基于证据和推理得出的结论。因此，与描述性统计相反，推断统计使用数据或数据样本进行推断（预测）。统计推断中的主要工具是线性回归。

*线性回归*是一种基本的机器学习算法，在本书中您将在第七章与其他机器学习算法中看到。因此，让我们在本节中介绍回归分析的直觉。

线性回归方程的最基本形式如下：

<math alttext="y equals alpha plus beta x plus epsilon"><mrow><mi>y</mi> <mo>=</mo> <mi>α</mi> <mo>+</mo> <mi>β</mi> <mi>x</mi> <mo>+</mo> <mi>ϵ</mi></mrow></math>

<math alttext="y i s t h e d e p e n d e n t v a r i a b l e comma i t i s w h a t y o u w a n t t o f o r e c a s t"><mrow><mi>y</mi> <mi>i</mi> <mi>s</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>d</mi> <mi>e</mi> <mi>p</mi> <mi>e</mi> <mi>n</mi> <mi>d</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>v</mi> <mi>a</mi> <mi>r</mi> <mi>i</mi> <mi>a</mi> <mi>b</mi> <mi>l</mi> <mi>e</mi> <mo>,</mo> <mi>i</mi> <mi>t</mi> <mi>i</mi> <mi>s</mi> <mi>w</mi> <mi>h</mi> <mi>a</mi> <mi>t</mi> <mi>y</mi> <mi>o</mi> <mi>u</mi> <mi>w</mi> <mi>a</mi> <mi>n</mi> <mi>t</mi> <mi>t</mi> <mi>o</mi> <mi>f</mi> <mi>o</mi> <mi>r</mi> <mi>e</mi> <mi>c</mi> <mi>a</mi> <mi>s</mi> <mi>t</mi></mrow></math>

<math alttext="x i s t h e i n d e p e n d e n t v a r i a b l e comma i t i s w h a t y o u u s e a s a n i n p u t t o f o r e c a s t y"><mrow><mi>x</mi> <mi>i</mi> <mi>s</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>i</mi> <mi>n</mi> <mi>d</mi> <mi>e</mi> <mi>p</mi> <mi>e</mi> <mi>n</mi> <mi>d</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>v</mi> <mi>a</mi> <mi>r</mi> <mi>i</mi> <mi>a</mi> <mi>b</mi> <mi>l</mi> <mi>e</mi> <mo>,</mo> <mi>i</mi> <mi>t</mi> <mi>i</mi> <mi>s</mi> <mi>w</mi> <mi>h</mi> <mi>a</mi> <mi>t</mi> <mi>y</mi> <mi>o</mi> <mi>u</mi> <mi>u</mi> <mi>s</mi> <mi>e</mi> <mi>a</mi> <mi>s</mi> <mi>a</mi> <mi>n</mi> <mi>i</mi> <mi>n</mi> <mi>p</mi> <mi>u</mi> <mi>t</mi> <mi>t</mi> <mi>o</mi> <mi>f</mi> <mi>o</mi> <mi>r</mi> <mi>e</mi> <mi>c</mi> <mi>a</mi> <mi>s</mi> <mi>t</mi> <mi>y</mi></mrow></math>

<math alttext="alpha i s t h e e x p e c t e d v a l u e o f t h e d e p e n d e n t v a r i a b l e w h e n t h e i n d e p e n d e n t v a r i a b l e s a r e e q u a l t o z e r o"><mrow><mi>α</mi> <mi>i</mi> <mi>s</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>e</mi> <mi>x</mi> <mi>p</mi> <mi>e</mi> <mi>c</mi> <mi>t</mi> <mi>e</mi> <mi>d</mi> <mi>v</mi> <mi>a</mi> <mi>l</mi> <mi>u</mi> <mi>e</mi> <mi>o</mi> <mi>f</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>d</mi> <mi>e</mi> <mi>p</mi> <mi>e</mi> <mi>n</mi> <mi>d</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>v</mi> <mi>a</mi> <mi>r</mi> <mi>i</mi> <mi>a</mi> <mi>b</mi> <mi>l</mi> <mi>e</mi> <mi>w</mi> <mi>h</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>i</mi> <mi>n</mi> <mi>d</mi> <mi>e</mi> <mi>p</mi> <mi>e</mi> <mi>n</mi> <mi>d</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>v</mi> <mi>a</mi> <mi>r</mi> <mi>i</mi> <mi>a</mi> <mi>b</mi> <mi>l</mi> <mi>e</mi> <mi>s</mi> <mi>a</mi> <mi>r</mi> <mi>e</mi> <mi>e</mi> <mi>q</mi> <mi>u</mi> <mi>a</mi> <mi>l</mi> <mi>t</mi> <mi>o</mi> <mi>z</mi> <mi>e</mi> <mi>r</mi> <mi>o</mi></mrow></math>

<math alttext="beta r e p r e s e n t s t h e c h a n g e i n t h e d e p e n d e n t v a r i a b l e p e r u n i t c h a n g e i n t h e i n d e p e n d e n t v a r i a b l e"><mrow><mi>β</mi> <mi>r</mi> <mi>e</mi> <mi>p</mi> <mi>r</mi> <mi>e</mi> <mi>s</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>s</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>c</mi> <mi>h</mi> <mi>a</mi> <mi>n</mi> <mi>g</mi> <mi>e</mi> <mi>i</mi> <mi>n</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>d</mi> <mi>e</mi> <mi>p</mi> <mi>e</mi> <mi>n</mi> <mi>d</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>v</mi> <mi>a</mi> <mi>r</mi> <mi>i</mi> <mi>a</mi> <mi>b</mi> <mi>l</mi> <mi>e</mi> <mi>p</mi> <mi>e</mi> <mi>r</mi> <mi>u</mi> <mi>n</mi> <mi>i</mi> <mi>t</mi> <mi>c</mi> <mi>h</mi> <mi>a</mi> <mi>n</mi> <mi>g</mi> <mi>e</mi> <mi>i</mi> <mi>n</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>i</mi> <mi>n</mi> <mi>d</mi> <mi>e</mi> <mi>p</mi> <mi>e</mi> <mi>n</mi> <mi>d</mi> <mi>e</mi> <mi>n</mi> <mi>t</mi> <mi>v</mi> <mi>a</mi> <mi>r</mi> <mi>i</mi> <mi>a</mi> <mi>b</mi> <mi>l</mi> <mi>e</mi></mrow></math>

<math alttext="epsilon i s t h e r e s i d u a l o r t h e u n e x p l a i n e d v a r i a t i o n"><mrow><mi>ϵ</mi> <mi>i</mi> <mi>s</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>r</mi> <mi>e</mi> <mi>s</mi> <mi>i</mi> <mi>d</mi> <mi>u</mi> <mi>a</mi> <mi>l</mi> <mi>o</mi> <mi>r</mi> <mi>t</mi> <mi>h</mi> <mi>e</mi> <mi>u</mi> <mi>n</mi> <mi>e</mi> <mi>x</mi> <mi>p</mi> <mi>l</mi> <mi>a</mi> <mi>i</mi> <mi>n</mi> <mi>e</mi> <mi>d</mi> <mi>v</mi> <mi>a</mi> <mi>r</mi> <mi>i</mi> <mi>a</mi> <mi>t</mi> <mi>i</mi> <mi>o</mi> <mi>n</mi></mrow></math>

基本线性回归方程说明，一个因变量（您想要预测的内容）由一个常数、一个敏感度调整的变量和一个残差（用于解释未解释变化的误差项）来解释。考虑表 3-3：

表 3-2. 预测表

| y | x |
| --- | --- |
| 100 | 49 |
| 200 | 99 |
| 300 | 149 |
| 400 | 199 |
| ? | 249 |

预测*y*给定*x*的线性方程如下：

<math alttext="y Subscript i Baseline equals 2 plus 2 x Subscript i"><mrow><msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo> <mn>2</mn> <mo>+</mo> <mn>2</mn> <msub><mi>x</mi> <mi>i</mi></msub></mrow></math>

因此，给定*x* = 249 时，最新的*y*应该是 500：

<math alttext="y Subscript i Baseline equals 2 plus 2 x Subscript i Baseline equals 2 plus left-parenthesis 2 times 249 right-parenthesis equals 500"><mrow><msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo> <mn>2</mn> <mo>+</mo> <mn>2</mn> <msub><mi>x</mi> <mi>i</mi></msub> <mo>=</mo> <mn>2</mn> <mo>+</mo> <mrow><mo>(</mo> <mn>2</mn> <mo>×</mo> <mn>249</mn> <mo>)</mo></mrow> <mo>=</mo> <mn>500</mn></mrow></math>

注意线性回归如何完美捕捉两个变量之间的线性关系，因为没有残差（未解释的变化）。当线性回归完美捕捉两个变量之间的关系时，这意味着它们的坐标点完美对齐在沿 x 轴的一条线上。

多元线性回归可以采用以下形式：

<math alttext="y Subscript i Baseline equals alpha plus beta 1 x 1 plus period period period plus beta Subscript n Baseline x Subscript n Baseline plus epsilon Subscript i Baseline"><mrow><msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo> <mi>α</mi> <mo>+</mo> <msub><mi>β</mi> <mn>1</mn></msub> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <mo>.</mo> <mo>.</mo> <mo>.</mo> <mo>+</mo> <msub><mi>β</mi> <mi>n</mi></msub> <msub><mi>x</mi> <mi>n</mi></msub> <mo>+</mo> <msub><mi>ϵ</mi> <mi>i</mi></msub></mrow></math>

这基本上意味着因变量*y*可能受到多个变量的影响。例如，如果您想要估计房价，您可能需要考虑房间数量、面积、社区和可能影响价格的任何其他变量。同样，如果您想要预测商品价格，您可能需要考虑不同的宏观经济因素、货币价值和任何其他替代数据。

重要的是要理解每个变量所指的内容。确保记住上述公式。线性回归有一些假设：

线性关系

因变量和自变量之间的关系应该是线性的，意味着平面上的一条直线可以描述这种关系。在处理复杂的金融变量时，这在现实生活中是罕见的。

变量的独立性

观察结果应该彼此独立，意味着一个观察结果的值不会影响另一个观察结果的值。

同方差性

残差的方差（因变量的预测值与实际值之间的差异）应该在自变量的所有水平上保持恒定。

残差的正态性

残差应该呈正态分布，意味着大多数残差接近零，分布是对称的。

在多元线性回归的情况下，您可以添加一个新的假设，即*多重共线性*的缺失。自变量之间不应高度相关，否则可能难以确定每个自变量对因变量的独特影响。换句话说，这可以防止冗余。在第七章中，您将看到更详细的线性回归示例，本节仅将其作为统计学领域的一部分进行介绍。

###### 注意

现在，您应该对统计学的关键概念有扎实的理解。让我们总结一下您需要记住的一切：

+   线性回归是推断统计学领域的一部分，它是描述变量之间关系的线性方程。

+   线性回归根据您训练过去数据并期望关系在未来保持的方程来解释和预测数据。

## 总结

能够进行数据分析是部署正确算法以预测时间序列未来值的关键。通过来自统计学领域的各种工具来理解数据。确保您了解什么是平稳性和相关性，因为它们在建模中提供极其有价值的见解。

^(1) 其他方式也有，但这两种方式是最流行的表示形式。
