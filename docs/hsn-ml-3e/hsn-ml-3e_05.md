# 第四章。训练模型

到目前为止，我们大多将机器学习模型及其训练算法视为黑匣子。如果您在之前章节的一些练习中有所了解，您可能会对不知道底层原理的情况下能做多少事情感到惊讶：您优化了一个回归系统，改进了一个数字图像分类器，甚至从头开始构建了一个垃圾邮件分类器，所有这些都是在不知道它们实际如何工作的情况下完成的。实际上，在许多情况下，您并不真正需要知道实现细节。

然而，对事物如何运作有一个良好的理解可以帮助您快速找到适当的模型、正确的训练算法以及适合您任务的一组良好的超参数。了解底层原理还将帮助您更有效地调试问题并执行错误分析。最后，本章讨论的大多数主题将对理解、构建和训练神经网络（本书的[第二部分](part02.html#neural_nets_part)中讨论）至关重要。

在本章中，我们将首先看一下线性回归模型，这是最简单的模型之一。我们将讨论两种非常不同的训练方法：

+   使用一个“封闭形式”方程⁠^([1](ch04.html#idm45720217568672))直接计算最适合训练集的模型参数（即最小化训练集上成本函数的模型参数）。

+   使用一种称为梯度下降（GD）的迭代优化方法，逐渐调整模型参数以最小化训练集上的成本函数，最终收敛到与第一种方法相同的参数集。我们将看一下几种梯度下降的变体，当我们研究神经网络时会一再使用：批量GD、小批量GD和随机GD。

接下来我们将看一下多项式回归，这是一个可以拟合非线性数据集的更复杂模型。由于这个模型比线性回归有更多的参数，所以更容易过拟合训练数据。我们将探讨如何通过学习曲线检测是否存在这种情况，然后我们将看一下几种正则化技术，可以减少过拟合训练集的风险。

最后，我们将研究另外两种常用于分类任务的模型：逻辑回归和softmax回归。

###### 警告

本章将包含相当多的数学方程，使用线性代数和微积分的基本概念。要理解这些方程，您需要知道向量和矩阵是什么；如何转置、相乘和求逆；以及什么是偏导数。如果您对这些概念不熟悉，请查看[在线补充材料](https://github.com/ageron/handson-ml3)中作为Jupyter笔记本提供的线性代数和微积分入门教程。对于那些真正对数学过敏的人，您仍然应该阅读本章，并简单跳过方程；希望文本足以帮助您理解大部分概念。

# 线性回归

在[第一章](ch01.html#landscape_chapter)中，我们看了一个关于生活满意度的简单回归模型：

*life_satisfaction* = *θ*[0] + *θ*[1] × *GDP_per_capita*

该模型只是输入特征`GDP_per_capita`的线性函数。*θ*[0]和*θ*[1]是模型的参数。

更一般地，线性模型通过简单地计算输入特征的加权和加上一个称为*偏置项*（也称为*截距项*）的常数来进行预测，如[方程4-1](#equation_four_one)所示。

##### 方程4-1。线性回归模型预测

<math display="block"><mrow><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mo>=</mo> <msub><mi>θ</mi> <mn>0</mn></msub> <mo>+</mo> <msub><mi>θ</mi> <mn>1</mn></msub> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>θ</mi> <mn>2</mn></msub> <msub><mi>x</mi> <mn>2</mn></msub> <mo>+</mo> <mo>⋯</mo> <mo>+</mo> <msub><mi>θ</mi> <mi>n</mi></msub> <msub><mi>x</mi> <mi>n</mi></msub></mrow></math>

在这个方程中：

+   *ŷ*是预测值。

+   *n*是特征数量。

+   *x*[*i*]是第*i*个特征值。

+   *θ*[*j*]是第*j*个模型参数，包括偏置项*θ*[0]和特征权重*θ*[1]、*θ*[2]、⋯、*θ*[*n*]。

这可以用矢量化形式更简洁地表示，如[方程4-2](#linear_regression_prediction_vectorized_equation)所示。

##### 方程4-2\. 线性回归模型预测（矢量化形式）

<math><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo>=</mo><msub><mi>h</mi><mi mathvariant="bold">θ</mi></msub><mo>(</mo><mi mathvariant="bold">x</mi><mo>)</mo><mo>=</mo><mi mathvariant="bold">θ</mi><mo>·</mo><mi mathvariant="bold">x</mi></math>

在这个方程中：

+   *h*[**θ**]是假设函数，使用模型参数**θ**。

+   **θ**是模型的*参数向量*，包括偏置项*θ*[0]和特征权重*θ*[1]到*θ*[*n*]。

+   **x**是实例的*特征向量*，包含*x*[0]到*x*[*n*]，其中*x*[0]始终等于1。

+   **θ** · **x**是向量**θ**和**x**的点积，等于*θ*[0]*x*[0] + *θ*[1]*x*[1] + *θ*[2]*x*[2] + ... + *θ*[*n*]*x*[*n*]。

###### 注意

在机器学习中，向量通常表示为*列向量*，这是具有单列的二维数组。如果**θ**和**x**是列向量，那么预测值为<math><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo>=</mo><msup><mi mathvariant="bold">θ</mi><mo>⊺</mo></msup><mi mathvariant="bold">x</mi></math>，其中<math><msup><mi mathvariant="bold">θ</mi><mo>⊺</mo></msup></math>是**θ**的*转置*（行向量而不是列向量），<math><msup><mi mathvariant="bold">θ</mi><mo>⊺</mo></msup><mi mathvariant="bold">x</mi></math>是<math><msup><mi mathvariant="bold">θ</mi><mo>⊺</mo></msup></math>和**x**的矩阵乘法。当然，这是相同的预测，只是现在表示为单元格矩阵而不是标量值。在本书中，我将使用这种表示法，以避免在点积和矩阵乘法之间切换。

好的，这就是线性回归模型，但我们如何训练它呢？嗯，回想一下，训练模型意味着设置其参数，使模型最好地适应训练集。为此，我们首先需要一个衡量模型与训练数据拟合程度的指标。在[第2章](ch02.html#project_chapter)中，我们看到回归模型最常见的性能指标是均方根误差（[方程2-1](ch02.html#rmse_equation)）。因此，要训练线性回归模型，我们需要找到最小化RMSE的**θ**的值。在实践中，最小化均方误差（MSE）比最小化RMSE更简单，并且会导致相同的结果（因为最小化正函数的值也会最小化其平方根）。

###### 警告

在训练期间，学习算法通常会优化不同的损失函数，而不是用于评估最终模型的性能指标。这通常是因为该函数更容易优化和/或因为在训练期间仅需要额外的项（例如，用于正则化）。一个好的性能指标应尽可能接近最终的业务目标。一个好的训练损失易于优化，并且与指标强相关。例如，分类器通常使用成本函数进行训练，如对数损失（稍后在本章中将看到），但使用精度/召回率进行评估。对数损失易于最小化，这样做通常会提高精度/召回率。

线性回归假设*h*[**θ**]在训练集**X**上的MSE是使用[方程4-3](#mse_cost_function)计算的。

##### 方程4-3. 线性回归模型的MSE成本函数

<math display="block"><mrow><mtext>MSE</mtext> <mrow><mo>(</mo> <mi mathvariant="bold">X</mi> <mo>,</mo> <msub><mi>h</mi> <mi mathvariant="bold">θ</mi></msub> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>1</mn> <mi>m</mi></mfrac></mstyle> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>m</mi></munderover> <msup><mrow><mo>(</mo><msup><mi mathvariant="bold">θ</mi> <mo>⊺</mo></msup> <msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>-</mo><msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>

大多数这些符号在[第2章](ch02.html#project_chapter)中已经介绍过（参见[“符号”](ch02.html#notations)）。唯一的区别是我们写*h*[**θ**]而不是只写*h*，以明确模型是由向量**θ**参数化的。为了简化符号，我们将只写MSE(**θ**)而不是MSE(**X**, *h*[**θ**])。

## 正规方程

为了找到最小化MSE的**θ**的值，存在一个*闭式解*——换句话说，一个直接给出结果的数学方程。这被称为*正规方程*（[方程4-4](#equation_four_four)）。

##### 方程4-4. 正规方程

<math display="block"><mrow><mover accent="true"><mi mathvariant="bold">θ</mi> <mo>^</mo></mover> <mo>=</mo> <msup><mrow><mo>(</mo><msup><mi mathvariant="bold">X</mi> <mo>⊺</mo></msup> <mi mathvariant="bold">X</mi><mo>)</mo></mrow> <mrow><mo>-</mo><mn>1</mn></mrow></msup>  <msup><mi mathvariant="bold">X</mi> <mo>⊺</mo></msup>  <mi mathvariant="bold">y</mi></mrow></math>

在这个方程中：

+   <math><mover accent="true"><mi mathvariant="bold">θ</mi><mo>^</mo></mover></math>是最小化成本函数的**θ**的值。

+   **y**是包含*y*^((1))到*y*^((*m*))的目标值向量。

让我们生成一些看起来线性的数据来测试这个方程（[图4-1](#generated_data_plot)）：

```py
import numpy as np

np.random.seed(42)  # to make this code example reproducible
m = 100  # number of instances
X = 2 * np.random.rand(m, 1)  # column vector
y = 4 + 3 * X + np.random.randn(m, 1)  # column vector
```

![mls3 0401](assets/mls3_0401.png)

###### 图4-1. 随机生成的线性数据集

现在让我们使用正规方程计算<math><mover accent="true"><mi mathvariant="bold">θ</mi><mo>^</mo></mover></math>。我们将使用NumPy的线性代数模块（`np.linalg`）中的`inv()`函数计算矩阵的逆，以及矩阵乘法的`dot()`方法：

```py
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

###### 注意

`@`运算符执行矩阵乘法。如果`A`和`B`是NumPy数组，则`A @ B`等同于`np.matmul(A, B)`。许多其他库，如TensorFlow、PyTorch和JAX，也支持`@`运算符。但是，不能在纯Python数组（即列表的列表）上使用`@`。

我们用来生成数据的函数是*y* = 4 + 3*x*[1] + 高斯噪声。让我们看看方程找到了什么：

```py
>>> theta_best
array([[4.21509616],
 [2.77011339]])
```

我们希望*θ*[0] = 4和*θ*[1] = 3，而不是*θ*[0] = 4.215和*θ*[1] = 2.770。足够接近，但噪声使得无法恢复原始函数的确切参数。数据集越小且噪声越大，问题就越困难。

现在我们可以使用<math><mover accent="true"><mi mathvariant="bold">θ</mi><mo>^</mo></mover></math>进行预测：

```py
>>> X_new = np.array([[0], [2]])
>>> X_new_b = add_dummy_feature(X_new)  # add x0 = 1 to each instance
>>> y_predict = X_new_b @ theta_best
>>> y_predict
array([[4.21509616],
 [9.75532293]])
```

让我们绘制这个模型的预测（[图4-2](#linear_model_predictions_plot)）：

```py
import matplotlib.pyplot as plt

plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.")
[...]  # beautify the figure: add labels, axis, grid, and legend
plt.show()
```

![mls3 0402](assets/mls3_0402.png)

###### 图4-2. 线性回归模型预测

使用Scikit-Learn执行线性回归相对简单：

```py
>>> from sklearn.linear_model import LinearRegression
>>> lin_reg = LinearRegression()
>>> lin_reg.fit(X, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([4.21509616]), array([[2.77011339]]))
>>> lin_reg.predict(X_new)
array([[4.21509616],
 [9.75532293]])
```

请注意，Scikit-Learn将偏置项（`intercept_`）与特征权重（`coef_`）分开。`LinearRegression`类基于`scipy.linalg.lstsq()`函数（名称代表“最小二乘法”），您可以直接调用该函数：

```py
>>> theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
>>> theta_best_svd
array([[4.21509616],
 [2.77011339]])
```

这个函数计算<math><mover accent="true"><mi mathvariant="bold">θ</mi><mo>^</mo></mover><mo>=</mo><msup><mi mathvariant="bold">X</mi><mo>+</mo></msup><mi mathvariant="bold">y</mi></math>，其中<math><msup><mi mathvariant="bold">X</mi><mo>+</mo></msup></math>是**X**的*伪逆*（具体来说，是Moore-Penrose逆）。您可以使用`np.linalg.pinv()`直接计算伪逆：

```py
>>> np.linalg.pinv(X_b) @ y
array([[4.21509616],
 [2.77011339]])
```

伪逆本身是使用称为*奇异值分解*（SVD）的标准矩阵分解技术计算的，可以将训练集矩阵**X**分解为三个矩阵**U** **Σ** **V**^⊺的矩阵乘法（参见`numpy.linalg.svd()`）。伪逆计算为<math><msup><mi mathvariant="bold">X</mi><mo>+</mo></msup><mo>=</mo><mi mathvariant="bold">V</mi><msup><mi mathvariant="bold">Σ</mi><mo>+</mo></msup><msup><mi mathvariant="bold">U</mi><mo>⊺</mo></msup></math>。为了计算矩阵<math><msup><mi mathvariant="bold">Σ</mi><mo>+</mo></msup></math>，算法取**Σ**并将小于一个微小阈值的所有值设为零，然后用它们的倒数替换所有非零值，最后转置结果矩阵。这种方法比计算正规方程更有效，而且可以很好地处理边缘情况：实际上，如果矩阵**X**^⊺**X**不可逆（即奇异），例如如果*m*<*n*或者某些特征是冗余的，那么正规方程可能无法工作，但伪逆总是被定义的。

## 计算复杂度

正规方程计算**X**^⊺**X**的逆，这是一个（*n*+1）×（*n*+1）矩阵（其中*n*是特征数）。求解这样一个矩阵的*计算复杂度*通常约为*O*(*n*^(2.4))到*O*(*n*³)，取决于实现。换句话说，如果特征数翻倍，计算时间大约会乘以2^(2.4)=5.3到2³=8。

Scikit-Learn的`LinearRegression`类使用的SVD方法大约是*O*(*n*²)。如果特征数量翻倍，计算时间大约会乘以4。

###### 警告

当特征数量增多时（例如100,000），正规方程和SVD方法都变得非常慢。积极的一面是，它们都与训练集中实例数量线性相关（它们是*O*(*m*)），因此它们可以有效地处理大型训练集，只要它们可以放入内存。

此外，一旦训练好线性回归模型（使用正规方程或任何其他算法），预测速度非常快：计算复杂度与您要进行预测的实例数量和特征数量成正比。换句话说，对两倍实例（或两倍特征）进行预测将花费大约两倍的时间。

现在我们将看一种非常不同的训练线性回归模型的方法，这种方法更适用于特征数量较多或训练实例太多无法放入内存的情况。

# 梯度下降

*梯度下降*是一种通用的优化算法，能够找到各种问题的最优解。梯度下降的一般思想是迭代地调整参数，以最小化成本函数。

假设你在浓雾中的山中迷失了方向，只能感受到脚下的坡度。快速到达山谷底部的一个好策略是沿着最陡的坡度方向下坡。这正是梯度下降所做的：它测量了关于参数向量**θ**的误差函数的局部梯度，并沿着下降梯度的方向前进。一旦梯度为零，你就到达了一个最小值！

在实践中，您首先用随机值填充**θ**（这称为*随机初始化*）。然后逐渐改进它，每次尝试减少成本函数（例如MSE）一点点，直到算法*收敛*到最小值（参见[图4-3](#gradient_descent_diagram)）。

![mls3 0403](assets/mls3_0403.png)

###### 图4-3。在这个梯度下降的描述中，模型参数被随机初始化，并不断调整以最小化成本函数；学习步长大小与成本函数的斜率成比例，因此随着成本接近最小值，步长逐渐变小

梯度下降中的一个重要参数是步长的大小，由*学习率*超参数确定。如果学习率太小，那么算法将需要经过许多迭代才能收敛，这将花费很长时间（参见[图4-4](#small_learning_rate_diagram)）。

![mls3 0404](assets/mls3_0404.png)

###### 图4-4。学习率太小

另一方面，如果学习率太高，您可能会跳过山谷，最终停在另一侧，甚至可能比之前更高。这可能导致算法发散，产生越来越大的值，无法找到一个好的解决方案（参见[图4-5](#large_learning_rate_diagram)）。

![mls3 0405](assets/mls3_0405.png)

###### 图4-5。学习率太高

此外，并非所有成本函数都像漂亮的、规则的碗一样。可能会有洞、脊、高原和各种不规则的地形，使得收敛到最小值变得困难。[图4-6](#gradient_descent_pitfalls_diagram)展示了梯度下降的两个主要挑战。如果随机初始化将算法开始于左侧，则它将收敛到*局部最小值*，这不如*全局最小值*好。如果它从右侧开始，则穿过高原将需要很长时间。如果您停得太早，您将永远无法达到全局最小值。

![mls3 0406](assets/mls3_0406.png)

###### 图4-6。梯度下降的陷阱

幸运的是，线性回归模型的MSE成本函数恰好是一个*凸函数*，这意味着如果您选择曲线上的任意两点，连接它们的线段永远不会低于曲线。这意味着没有局部最小值，只有一个全局最小值。它还是一个连续函数，斜率永远不会突然改变。这两个事实有一个重要的结果：梯度下降保证可以无限接近全局最小值（如果等待足够长的时间且学习率不太高）。

虽然成本函数的形状像一个碗，但如果特征具有非常不同的比例，它可能是一个延长的碗。[图4-7](#elongated_bowl_diagram)展示了在特征1和2具有相同比例的训练集上的梯度下降（左侧），以及在特征1的值远小于特征2的训练集上的梯度下降（右侧）。

![mls3 0407](assets/mls3_0407.png)

###### 图4-7。特征缩放的梯度下降（左）和不缩放的梯度下降（右）

正如您所看到的，左侧的梯度下降算法直接朝向最小值，因此快速到达，而右侧首先朝向几乎与全局最小值方向正交的方向，最终沿着几乎平坦的山谷长途跋涉。它最终会到达最小值，但需要很长时间。

###### 警告

在使用梯度下降时，您应确保所有特征具有相似的比例（例如，使用Scikit-Learn的`StandardScaler`类），否则收敛所需的时间将更长。

这个图表还说明了训练模型意味着寻找一组模型参数的组合，使得成本函数（在训练集上）最小化。这是在模型的*参数空间*中进行的搜索。模型的参数越多，空间的维度就越多，搜索就越困难：在一个300维的草堆中搜索一根针比在3维空间中要困难得多。幸运的是，由于线性回归的情况下成本函数是凸的，所以这根针就在碗底。

## 批量梯度下降

要实现梯度下降，您需要计算成本函数相对于每个模型参数*θ*[*j*]的梯度。换句话说，您需要计算如果您稍微改变*θ*[*j*]，成本函数将如何变化。这被称为*偏导数*。这就像问，“如果我面向东，脚下的山坡有多陡？”然后面向北问同样的问题（如果您可以想象一个超过三维的宇宙，那么其他维度也是如此）。[方程4-5](#mse_partial_derivatives)计算了关于参数*θ*[*j*]的MSE的偏导数，表示为∂ MSE(**θ**) / ∂θ[*j*]。

##### 方程4-5\. 成本函数的偏导数

<math display="block"><mrow><mstyle scriptlevel="0" displaystyle="true"><mfrac><mi>∂</mi> <mrow><mi>∂</mi><msub><mi>θ</mi> <mi>j</mi></msub></mrow></mfrac></mstyle> <mtext>MSE</mtext> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>2</mn> <mi>m</mi></mfrac></mstyle> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>m</mi></munderover> <mrow><mo>(</mo> <msup><mi mathvariant="bold">θ</mi> <mo>⊺</mo></msup> <msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>-</mo> <msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>)</mo></mrow> <msubsup><mi>x</mi> <mi>j</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow></math>

与单独计算这些偏导数不同，您可以使用[方程4-6](#mse_gradient_vector)一次性计算它们。梯度向量，表示为∇[**θ**]MSE(**θ**)，包含成本函数的所有偏导数（每个模型参数一个）。

##### 方程4-6\. 成本函数的梯度向量

<math display="block"><mrow><msub><mi>∇</mi> <mi mathvariant="bold">θ</mi></msub> <mtext>MSE</mtext> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow> <mo>=</mo> <mfenced open="(" close=")"><mtable><mtr><mtd><mrow><mfrac><mi>∂</mi> <mrow><mi>∂</mi><msub><mi>θ</mi> <mn>0</mn></msub></mrow></mfrac> <mtext>MSE</mtext> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mi>∂</mi> <mrow><mi>∂</mi><msub><mi>θ</mi> <mn>1</mn></msub></mrow></mfrac> <mtext>MSE</mtext> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mo>⋮</mo></mtd></mtr> <mtr><mtd><mrow><mfrac><mi>∂</mi> <mrow><mi>∂</mi><msub><mi>θ</mi> <mi>n</mi></msub></mrow></mfrac> <mtext>MSE</mtext> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow></mrow></mtd></mtr></mtable></mfenced> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>2</mn> <mi>m</mi></mfrac></mstyle> <msup><mi mathvariant="bold">X</mi> <mo>⊺</mo></msup> <mrow><mo>(</mo> <mi mathvariant="bold">X</mi> <mi mathvariant="bold">θ</mi> <mo>-</mo> <mi mathvariant="bold">y</mi> <mo>)</mo></mrow></mrow></math>

###### 警告

请注意，这个公式涉及对整个训练集**X**进行计算，每次梯度下降步骤都要进行！这就是为什么该算法被称为*批量梯度下降*：它在每一步使用整个批量的训练数据（实际上，*全梯度下降*可能是一个更好的名称）。因此，在非常大的训练集上，它非常慢（我们很快将看到一些更快的梯度下降算法）。然而，梯度下降随着特征数量的增加而扩展得很好；当特征数量达到数十万时，使用梯度下降训练线性回归模型比使用正规方程或SVD分解要快得多。

一旦有了指向上坡的梯度向量，只需朝相反方向前进以下坡。这意味着从**θ**中减去∇[**θ**]MSE(**θ**)。这就是学习率*η*发挥作用的地方：⁠^([4](ch04.html#idm45720216763152))将梯度向量乘以*η*来确定下坡步长的大小（[方程4-7](#gradient_descent_step)）。

##### 方程4-7\. 梯度下降步骤

<math><msup><mi mathvariant="bold">θ</mi><mrow><mo>(</mo><mtext>下一步</mtext><mo>)</mo></mrow></msup><mo>=</mo><mi mathvariant="bold">θ</mi><mo>-</mo><mi>η</mi><msub><mo>∇</mo><mi mathvariant="bold">θ</mi></msub><mtext>MSE</mtext><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></math>

让我们快速实现这个算法：

```py
eta = 0.1  # learning rate
n_epochs = 1000
m = len(X_b)  # number of instances

np.random.seed(42)
theta = np.random.randn(2, 1)  # randomly initialized model parameters

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
```

这并不难！每次对训练集的迭代称为*epoch*。让我们看看得到的`theta`：

```py
>>> theta
array([[4.21509616],
 [2.77011339]])
```

嘿，这正是正规方程找到的！梯度下降完美地工作了。但是如果您使用了不同的学习率（`eta`）会怎样呢？[图4-8](#gradient_descent_plot)显示了使用三种不同学习率的梯度下降的前20步。每个图中底部的线代表随机起始点，然后每个迭代由越来越深的线表示。

![mls3 0408](assets/mls3_0408.png)

###### 图4-8\. 不同学习率的梯度下降

在左侧，学习率太低：算法最终会达到解，但需要很长时间。在中间，学习率看起来相当不错：在几个迭代中，它已经收敛到解。在右侧，学习率太高：算法发散，跳来跳去，实际上每一步都离解越来越远。

要找到一个好的学习率，可以使用网格搜索（参见[第2章](ch02.html#project_chapter)）。然而，您可能希望限制迭代次数，以便网格搜索可以消除收敛时间过长的模型。

您可能想知道如何设置迭代次数。如果太低，当算法停止时，您仍然离最优解很远；但如果太高，您将浪费时间，因为模型参数不再改变。一个简单的解决方案是设置一个非常大的迭代次数，但在梯度向量变得微小时中断算法——也就是说，当其范数小于一个微小数*ϵ*（称为*容差*）时——因为这表示梯度下降已经（几乎）达到了最小值。

## 随机梯度下降

批量梯度下降的主要问题在于，它在每一步使用整个训练集来计算梯度，这使得在训练集很大时非常缓慢。相反，*随机梯度下降* 在每一步选择训练集中的一个随机实例，并仅基于该单个实例计算梯度。显然，一次只处理一个实例使得算法更快，因为每次迭代时需要操作的数据量很少。这也使得在庞大的训练集上进行训练成为可能，因为每次迭代只需要一个实例在内存中（随机梯度下降可以作为一种离线算法实现；参见[第1章](ch01.html#landscape_chapter)）。

另一方面，由于其随机（即随机）性质，这种算法比批量梯度下降不规则得多：成本函数不会温和地减少直到达到最小值，而是会上下波动，仅平均减少。随着时间的推移，它最终会非常接近最小值，但一旦到达那里，它将继续上下波动，永远不会稳定下来（参见[图4-9](#sgd_random_walk_diagram)）。一旦算法停止，最终的参数值将是不错的，但不是最优的。

![mls3 0409](assets/mls3_0409.png)

###### 图4-9。使用随机梯度下降，每个训练步骤比使用批量梯度下降快得多，但也更不规则。

当成本函数非常不规则时（如[图4-6](#gradient_descent_pitfalls_diagram)中所示），这实际上可以帮助算法跳出局部最小值，因此随机梯度下降比批量梯度下降更有可能找到全局最小值。

因此，随机性有助于摆脱局部最优解，但也不好，因为这意味着算法永远无法稳定在最小值处。解决这一困境的一个方法是逐渐降低学习率。步骤开始很大（有助于快速取得进展并摆脱局部最小值），然后变得越来越小，允许算法在全局最小值处稳定下来。这个过程类似于*模拟退火*，这是一种受金属冶炼过程启发的算法，其中熔化的金属被慢慢冷却。确定每次迭代学习率的函数称为*学习计划*。如果学习率降低得太快，您可能会陷入局部最小值，甚至最终冻结在最小值的一半。如果学习率降低得太慢，您可能会在最小值周围跳来跳去很长时间，并且如果您在训练过早停止，最终会得到一个次优解。

此代码使用简单的学习计划实现随机梯度下降：

```py
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)  # for SGD, do not divide by m
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients
```

按照惯例，我们按照*m*次迭代的轮次进行迭代；每一轮称为*epoch*，如前所述。虽然批量梯度下降代码通过整个训练集迭代了1,000次，但这段代码只通过训练集迭代了50次，并达到了一个相当不错的解决方案：

```py
>>> theta
array([[4.21076011],
 [2.74856079]])
```

[图4-10](#sgd_plot)显示了训练的前20步（请注意步骤的不规则性）。

请注意，由于实例是随机选择的，一些实例可能在每个epoch中被多次选择，而其他实例可能根本不被选择。如果您想确保算法在每个epoch中通过每个实例，另一种方法是对训练集进行洗牌（确保同时洗牌输入特征和标签），然后逐个实例地进行，然后再次洗牌，依此类推。然而，这种方法更复杂，通常不会改善结果。

![mls3 0410](assets/mls3_0410.png)

###### 图4-10。随机梯度下降的前20步

###### 警告

在使用随机梯度下降时，训练实例必须是独立同分布的（IID），以确保参数平均被拉向全局最优解。确保这一点的一个简单方法是在训练期间对实例进行洗牌（例如，随机选择每个实例，或在每个epoch开始时对训练集进行洗牌）。如果不对实例进行洗牌，例如，如果实例按标签排序，则SGD将从优化一个标签开始，然后是下一个标签，依此类推，并且不会接近全局最小值。

要使用Scikit-Learn进行随机梯度下降线性回归，您可以使用`SGDRegressor`类，默认情况下优化MSE成本函数。以下代码最多运行1,000个时代（`max_iter`）或在100个时代内损失下降不到10^(–5)（`tol`）时停止（`n_iter_no_change`）。它以学习率0.01（`eta0`）开始，使用默认学习计划（与我们使用的不同）。最后，它不使用任何正则化（`penalty=None`；稍后会详细介绍）：

```py
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets
```

再次，您会发现解决方案与正规方程返回的解非常接近：

```py
>>> sgd_reg.intercept_, sgd_reg.coef_
(array([4.21278812]), array([2.77270267]))
```

###### 提示

所有Scikit-Learn估计器都可以使用`fit()`方法进行训练，但有些估计器还有一个`partial_fit()`方法，您可以调用它来对一个或多个实例运行一轮训练（它会忽略`max_iter`或`tol`等超参数）。反复调用`partial_fit()`会逐渐训练模型。当您需要更多控制训练过程时，这是很有用的。其他模型则有一个`warm_start`超参数（有些模型两者都有）：如果您设置`warm_start=True`，在已训练的模型上调用`fit()`方法不会重置模型；它将继续训练在哪里停止，遵守`max_iter`和`tol`等超参数。请注意，`fit()`会重置学习计划使用的迭代计数器，而`partial_fit()`不会。

## 小批量梯度下降

我们将要看的最后一个梯度下降算法称为*小批量梯度下降*。一旦您了解了批量梯度下降和随机梯度下降，这就很简单了：在每一步中，小批量梯度下降不是基于完整训练集（批量梯度下降）或仅基于一个实例（随机梯度下降）计算梯度，而是在称为*小批量*的小随机实例集上计算梯度。小批量梯度下降相对于随机梯度下降的主要优势在于，您可以通过硬件优化矩阵运算获得性能提升，尤其是在使用GPU时。

该算法在参数空间中的进展比随机梯度下降更加稳定，尤其是在使用相当大的小批量时。因此，小批量梯度下降最终会比随机梯度下降更接近最小值，但它可能更难逃离局部最小值（在存在局部最小值的问题中，不同于具有MSE成本函数的线性回归）。[图4-11](#gradient_descent_paths_plot)显示了训练过程中三种梯度下降算法在参数空间中的路径。它们最终都接近最小值，但批量梯度下降的路径实际上停在最小值处，而随机梯度下降和小批量梯度下降则继续移动。但是，请不要忘记，批量梯度下降需要很长时间才能完成每一步，如果您使用良好的学习计划，随机梯度下降和小批量梯度下降也会达到最小值。

![mls3 0411](assets/mls3_0411.png)

###### 图4-11\. 参数空间中的梯度下降路径

[表4-1](#linear_regression_algorithm_comparison)比较了迄今为止我们讨论过的线性回归算法（请回忆*m*是训练实例的数量，*n*是特征的数量）。

表4-1\. 线性回归算法比较

| 算法 | 大 *m* | 支持离线 | 大 *n* | 超参数 | 需要缩放 | Scikit-Learn |
| --- | --- | --- | --- | --- | --- | --- |
| 正规方程 | 快 | 否 | 慢 | 0 | 否 | N/A |
| SVD | 快 | 否 | 慢 | 0 | 否 | `LinearRegression` |
| 批量梯度下降 | 慢 | 否 | 快 | 2 | 是 | N/A |
| 随机梯度下降 | 快 | 是 | 快 | ≥2 | 是 | `SGDRegressor` |
| 小批量梯度下降 | 快 | 是 | 快 | ≥2 | 是 | N/A |

训练后几乎没有区别：所有这些算法最终得到非常相似的模型，并以完全相同的方式进行预测。

# 多项式回归

如果你的数据比一条直线更复杂怎么办？令人惊讶的是，你可以使用线性模型来拟合非线性数据。一个简单的方法是将每个特征的幂作为新特征添加，然后在这个扩展的特征集上训练线性模型。这种技术称为*多项式回归*。

让我们看一个例子。首先，我们将生成一些非线性数据（参见[图4-12](#quadratic_data_plot)），基于一个简单的*二次方程*——即形式为*y* = *ax*² + *bx* + *c*的方程——再加上一些噪声：

```py
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
```

![mls3 0412](assets/mls3_0412.png)

###### 图4-12。生成的非线性和嘈杂数据集

显然，一条直线永远无法正确拟合这些数据。因此，让我们使用Scikit-Learn的`PolynomialFeatures`类来转换我们的训练数据，将训练集中每个特征的平方（二次多项式）作为新特征添加到训练数据中（在这种情况下只有一个特征）：

```py
>>> from sklearn.preprocessing import PolynomialFeatures
>>> poly_features = PolynomialFeatures(degree=2, include_bias=False)
>>> X_poly = poly_features.fit_transform(X)
>>> X[0]
array([-0.75275929])
>>> X_poly[0]
array([-0.75275929,  0.56664654])
```

`X_poly`现在包含了`X`的原始特征以及该特征的平方。现在我们可以将`LinearRegression`模型拟合到这个扩展的训练数据上（[图4-13](#quadratic_predictions_plot)）：

```py
>>> lin_reg = LinearRegression()
>>> lin_reg.fit(X_poly, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([1.78134581]), array([[0.93366893, 0.56456263]]))
```

![mls3 0413](assets/mls3_0413.png)

###### 图4-13。多项式回归模型预测

不错：模型估计<math><mrow><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mo>=</mo> <mn>0.56</mn> <msup><mrow><msub><mi>x</mi> <mn>1</mn></msub></mrow> <mn>2</mn></msup> <mo>+</mo> <mn>0.93</mn> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <mn>1.78</mn></mrow></math>，而实际上原始函数是<math><mrow><mi>y</mi> <mo>=</mo> <mn>0.5</mn> <msup><mrow><msub><mi>x</mi> <mn>1</mn></msub></mrow> <mn>2</mn></msup> <mo>+</mo> <mn>1.0</mn> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <mn>2.0</mn> <mo>+</mo> <mtext>高斯噪声</mtext></mrow></math>。

请注意，当存在多个特征时，多项式回归能够找到特征之间的关系，这是普通线性回归模型无法做到的。这是因为`PolynomialFeatures`还会添加给定次数的所有特征组合。例如，如果有两个特征*a*和*b*，`PolynomialFeatures`的`degree=3`不仅会添加特征*a*²、*a*³、*b*²和*b*³，还会添加组合*ab*、*a*²*b*和*ab*²。

###### 警告

`PolynomialFeatures(degree=*d*)`将包含*n*个特征的数组转换为包含(*n* + *d*)! / *d*!*n*!个特征的数组，其中*n*!是*n*的*阶乘*，等于1 × 2 × 3 × ⋯ × *n*。注意特征数量的组合爆炸！

# 学习曲线

如果进行高次多项式回归，你很可能会比普通线性回归更好地拟合训练数据。例如，[图4-14](#high_degree_polynomials_plot)将一个300次多项式模型应用于前面的训练数据，并将结果与纯线性模型和二次模型（二次多项式）进行比较。请注意，300次多项式模型在训练实例周围摆动以尽可能接近训练实例。

![mls3 0414](assets/mls3_0414.png)

###### 图4-14。高次多项式回归

这个高次多项式回归模型严重过拟合了训练数据，而线性模型则欠拟合了。在这种情况下，最能泛化的模型是二次模型，这是有道理的，因为数据是使用二次模型生成的。但通常你不会知道是什么函数生成了数据，那么你如何决定模型应该有多复杂呢？你如何判断你的模型是过拟合还是欠拟合了数据？

在[第2章](ch02.html#project_chapter)中，您使用交叉验证来估计模型的泛化性能。如果模型在训练数据上表现良好，但根据交叉验证指标泛化能力差，那么您的模型是过拟合的。如果两者表现都不好，那么它是拟合不足的。这是判断模型过于简单或过于复杂的一种方法。

另一种方法是查看*学习曲线*，这是模型的训练误差和验证误差作为训练迭代的函数的图表：只需在训练集和验证集上定期评估模型，并绘制结果。如果模型无法进行增量训练（即，如果它不支持`partial_fit()`或`warm_start`），那么您必须在逐渐扩大的训练集子集上多次训练它。

Scikit-Learn有一个有用的`learning_curve()`函数来帮助解决这个问题：它使用交叉验证来训练和评估模型。默认情况下，它会在不断增长的训练集子集上重新训练模型，但如果模型支持增量学习，您可以在调用`learning_curve()`时设置`exploit_incremental_learning=True`，它将逐步训练模型。该函数返回评估模型的训练集大小，以及每个大小和每个交叉验证折叠的训练和验证分数。让我们使用这个函数来查看普通线性回归模型的学习曲线（参见[图4-15](#underfitting_learning_curves_plot)）：

```py
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
[...]  # beautify the figure: add labels, axis, grid, and legend
plt.show()
```

![mls3 0415](assets/mls3_0415.png)

###### 图4-15\. 学习曲线

这个模型拟合不足。为了了解原因，首先让我们看看训练误差。当训练集中只有一个或两个实例时，模型可以完美拟合它们，这就是曲线从零开始的原因。但随着新实例被添加到训练集中，模型无法完美拟合训练数据，因为数据存在噪声，而且根本不是线性的。因此，训练数据的误差会上升，直到达到一个平台，在这一点上，向训练集添加新实例不会使平均误差变得更好或更糟。现在让我们看看验证误差。当模型在非常少的训练实例上训练时，它无法正确泛化，这就是为什么验证误差最初相当大的原因。然后，随着模型展示更多的训练示例，它学习，因此验证误差慢慢下降。然而，再次，一条直线无法很好地对数据建模，因此误差最终会达到一个接近另一条曲线的平台。

这些学习曲线是典型的拟合不足模型。两条曲线都达到了一个平台；它们接近且相当高。

###### 提示

如果您的模型对训练数据拟合不足，增加更多的训练样本将无济于事。您需要使用更好的模型或提出更好的特征。

现在让我们看看相同数据上10次多项式模型的学习曲线（参见[图4-16](#learning_curves_plot)）：

```py
from sklearn.pipeline import make_pipeline

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    LinearRegression())

train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
[...]  # same as earlier
```

![mls3 0416](assets/mls3_0416.png)

###### 图4-16\. 10次多项式模型的学习曲线

这些学习曲线看起来有点像之前的曲线，但有两个非常重要的区别：

+   训练数据上的误差比以前低得多。

+   曲线之间存在差距。这意味着模型在训练数据上的表现明显优于验证数据，这是过拟合模型的标志。然而，如果您使用更大的训练集，这两条曲线将继续接近。

###### 提示

改进过拟合模型的一种方法是提供更多的训练数据，直到验证误差达到训练误差。

# 正则化线性模型

正如您在[第1章](ch01.html#landscape_chapter)和[第2章](ch02.html#project_chapter)中看到的，减少过拟合的一个好方法是对模型进行正则化（即，约束它）：它的自由度越少，过拟合数据的难度就越大。对多项式模型进行正则化的一种简单方法是减少多项式次数。

对于线性模型，通常通过约束模型的权重来实现正则化。我们现在将看一下岭回归、套索回归和弹性网络回归，它们实现了三种不同的约束权重的方式。

## 岭回归

*岭回归*（也称为*Tikhonov正则化*）是线性回归的正则化版本：一个等于<math><mfrac><mi>α</mi><mi>m</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msup><msub><mi>θ</mi><mi>i</mi></msub><mn>2</mn></msup></math>的*正则化项*被添加到MSE中。这迫使学习算法不仅拟合数据，还要尽量保持模型权重尽可能小。请注意，正则化项应该只在训练期间添加到成本函数中。一旦模型训练完成，您希望使用未经正则化的MSE（或RMSE）来评估模型的性能。

超参数*α*控制着您希望对模型进行多少正则化。如果*α*=0，则岭回归就是线性回归。如果*α*非常大，则所有权重最终都非常接近零，结果是一条通过数据均值的平坦线。[方程4-8](#ridge_cost_function)呈现了岭回归成本函数。⁠^([7](ch04.html#idm45720215617520))

##### 方程4-8。岭回归成本函数

<math><mrow><mi>J</mi><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>=</mo><mrow><mtext>MSE</mtext><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>+</mo><mfrac><mi>α</mi><mi>m</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msup><msub><mi>θ</mi><mi>i</mi></msub><mn>2</mn></msup></math>

请注意，偏置项*θ*[0]不被正则化（总和从*i*=1开始，而不是0）。如果我们将**w**定义为特征权重的向量（*θ*[1]到*θ*[*n*]），则正则化项等于*α*(∥ **w** ∥[2])² / *m*，其中∥ **w** ∥[2]表示权重向量的ℓ[2]范数。⁠^([8](ch04.html#idm45720215599696)) 对于批量梯度下降，只需将2*α***w** / *m*添加到对应于特征权重的MSE梯度向量的部分，而不要将任何内容添加到偏置项的梯度（参见[方程4-6](#mse_gradient_vector)）。

###### 警告

在执行岭回归之前，重要的是对数据进行缩放（例如，使用`StandardScaler`），因为它对输入特征的规模敏感。这对大多数正则化模型都是正确的。

[图4-17](#ridge_regression_plot)显示了在一些非常嘈杂的线性数据上使用不同*α*值训练的几个岭模型。在左侧，使用普通的岭模型，导致线性预测。在右侧，首先使用`PolynomialFeatures(degree=10)`扩展数据，然后使用`StandardScaler`进行缩放，最后将岭模型应用于生成的特征：这是带有岭正则化的多项式回归。请注意，增加*α*会导致更平缓（即，更不极端，更合理）的预测，从而减少模型的方差但增加其偏差。

![mls3 0417](assets/mls3_0417.png)

###### 图4-17。线性（左）和多项式（右）模型，都具有不同级别的岭正则化

与线性回归一样，我们可以通过计算闭式方程或执行梯度下降来执行岭回归。优缺点是相同的。[方程4-9](#ridge_regression_solution)显示了闭式解，其中**A**是(*n* + 1) × (*n* + 1) *单位矩阵*，⁠^([9](ch04.html#idm45720215579520))除了左上角的单元格为0，对应于偏置项。

##### 方程4-9. 岭回归闭式解

<math display="block"><mrow><mover accent="true"><mi mathvariant="bold">θ</mi> <mo>^</mo></mover> <mo>=</mo> <msup><mrow><mo>(</mo><msup><mi mathvariant="bold">X</mi> <mo>⊺</mo></msup> <mi mathvariant="bold">X</mi><mo>+</mo><mi>α</mi><mi mathvariant="bold">A</mi><mo>)</mo></mrow> <mrow><mo>-</mo><mn>1</mn></mrow></msup>  <msup><mi mathvariant="bold">X</mi> <mo>⊺</mo></msup>  <mi mathvariant="bold">y</mi></mrow></math>

以下是如何使用Scikit-Learn执行岭回归的闭式解（一种[方程4-9](#ridge_regression_solution)的变体，使用André-Louis Cholesky的矩阵分解技术）：

```py
>>> from sklearn.linear_model import Ridge
>>> ridge_reg = Ridge(alpha=0.1, solver="cholesky")
>>> ridge_reg.fit(X, y)
>>> ridge_reg.predict([[1.5]])
array([[1.55325833]])
```

使用随机梯度下降：⁠^([10](ch04.html#idm45720215546880))

```py
>>> sgd_reg = SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None,
...                        max_iter=1000, eta0=0.01, random_state=42)
...
>>> sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets
>>> sgd_reg.predict([[1.5]])
array([1.55302613])
```

`penalty`超参数设置要使用的正则化项的类型。指定`"l2"`表示您希望SGD将正则化项添加到MSE成本函数中，等于`alpha`乘以权重向量的ℓ[2]范数的平方。这就像岭回归一样，只是在这种情况下没有除以*m*；这就是为什么我们传递`alpha=0.1 / m`，以获得与`Ridge(alpha=0.1)`相同的结果。

###### 提示

`RidgeCV`类也执行岭回归，但它会自动使用交叉验证调整超参数。它大致相当于使用`GridSearchCV`，但它针对岭回归进行了优化，并且运行*快得多*。其他几个估计器（主要是线性的）也有高效的CV变体，如`LassoCV`和`ElasticNetCV`。

## Lasso回归

*最小绝对值收缩和选择算子回归*（通常简称为*Lasso回归*）是线性回归的另一个正则化版本：就像岭回归一样，它向成本函数添加一个正则化项，但是它使用权重向量的ℓ[1]范数，而不是ℓ[2]范数的平方（参见[方程4-10](#lasso_cost_function)）。请注意，ℓ[1]范数乘以2*α*，而ℓ[2]范数在岭回归中乘以*α* / *m*。选择这些因子是为了确保最佳*α*值与训练集大小无关：不同的范数导致不同的因子（有关更多细节，请参阅[Scikit-Learn问题＃15657](https://github.com/scikit-learn/scikit-learn/issues/15657)）。

##### 方程4-10. Lasso回归成本函数

<math><mrow><mi>J</mi><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>=</mo><mrow><mtext>MSE</mtext><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>+</mo><mrow><mn>2</mn><mi>α</mi><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mfenced open="|" close="|"><msub><mi>θ</mi><mi>i</mi></msub></mfenced></mrow></math>

[图4-18](#lasso_regression_plot)显示了与[图4-17](#ridge_regression_plot)相同的内容，但用Lasso模型替换了岭模型，并使用不同的*α*值。

![mls3 0418](assets/mls3_0418.png)

###### 图4-18. 线性（左）和多项式（右）模型，都使用不同级别的Lasso正则化

Lasso回归的一个重要特征是它倾向于消除最不重要特征的权重（即将它们设置为零）。例如，[图4-18](#lasso_regression_plot)中右侧图中的虚线看起来大致是立方形：高次多项式特征的所有权重都等于零。换句话说，Lasso回归自动执行特征选择，并输出具有少量非零特征权重的*稀疏模型*。

你可以通过查看[图4-19](#lasso_vs_ridge_plot)来了解这种情况：坐标轴代表两个模型参数，背景轮廓代表不同的损失函数。在左上角的图中，轮廓代表ℓ[1]损失（|θ[1]| + |θ[2]|），随着你靠近任何轴，损失会线性下降。例如，如果你将模型参数初始化为θ[1] = 2和θ[2] = 0.5，运行梯度下降将等量减少两个参数（如虚线黄线所示）；因此θ[2]会先达到0（因为它最初更接近0）。之后，梯度下降将沿着槽滚动，直到达到θ[1] = 0（稍微反弹一下，因为ℓ[1]的梯度从不接近0：对于每个参数，它们要么是-1要么是1）。在右上角的图中，轮廓代表套索回归的成本函数（即，MSE成本函数加上ℓ[1]损失）。小白色圆圈显示了梯度下降优化某些模型参数的路径，这些参数最初设定为θ[1] = 0.25和θ[2] = -1：再次注意路径如何迅速到达θ[2] = 0，然后沿着槽滚动并最终在全局最优解周围反弹（由红色方块表示）。如果增加α，全局最优解将沿着虚线黄线向左移动，而如果减小α，全局最优解将向右移动（在这个例子中，未正则化MSE的最佳参数为θ[1] = 2和θ[2] = 0.5）。

![mls3 0419](assets/mls3_0419.png)

###### 图4-19。套索与岭正则化

两个底部图表展示了相同的情况，但使用了ℓ[2]惩罚。在左下角的图中，你可以看到随着我们靠近原点，ℓ[2]损失减少，因此梯度下降直接朝着那个点前进。在右下角的图中，轮廓代表岭回归的成本函数（即，MSE成本函数加上ℓ[2]损失）。正如你所看到的，随着参数接近全局最优解，梯度变小，因此梯度下降自然减慢。这限制了反弹，有助于岭回归比套索收敛更快。还要注意，当增加α时，最佳参数（由红色方块表示）越来越接近原点，但它们永远不会完全消失。

###### 提示

为了防止在使用套索回归时梯度下降在最后反弹到最优解周围，你需要在训练过程中逐渐减小学习率。它仍然会在最优解周围反弹，但步长会变得越来越小，因此会收敛。

套索成本函数在θ[i] = 0（对于i = 1, 2, ⋯, n）处不可微，但如果在任何θ[i] = 0时使用*子梯度向量* **g**⁠^([11](ch04.html#idm45720215341648))，梯度下降仍然有效。[方程4-11](#lasso_subgradient_vector)展示了一个你可以用于套索成本函数的梯度下降的子梯度向量方程。

##### 方程4-11。套索回归子梯度向量

<math><mrow><mi>g</mi><mo>(</mo><mi mathvariant="bold">θ</mi><mo>,</mo><mi>J</mi><mo>)</mo></mrow><mo>=</mo><mrow><msub><mo>∇</mo><mi mathvariant="bold">θ</mi></msub><mtext>MSE</mtext><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>+</mo><mrow><mn>2</mn><mi>α</mi><mfenced><mtable><mtr><mtd><mo>sign</mo><mo>(</mo><msub><mi>θ</mi><mn>1</mn></msub><mo>)</mo></mtd></mtr><mtr><mtd><mo>sign</mo><mo>(</mo><msub><mi>θ</mi><mn>2</mn></msub><mo>)</mo></mtd></mtr><mtr><mtd><mo>⋮</mo></mtd></mtr><mtr><mtd><mo>sign</mo><mo>(</mo><msub><mi>θ</mi><mi>n</mi></msub><mo>)</mo></mtd></mtr></mtable></mfenced></mrow><mtext>where </mtext><mrow><mo>sign</mo><mo>(</mo><msub><mi>θ</mi><mi>i</mi></msub><mo>)</mo></mrow><mo>=</mo><mrow><mfenced open="{" close=""><mtable><mtr><mtd><mo>-</mo><mn>1</mn></mtd><mtd><mtext>if </mtext><msub><mi>θ</mi><mi>i</mi></msub><mo><</mo><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mtext>if </mtext><msub><mi>θ</mi><mi>i</mi></msub><mo>=</mo><mn>0</mn></mtd></mtr><mtr><mtd><mo>+</mo><mn>1</mn></mtd><mtd><mtext>if </mtext><msub><mi>θ</mi><mi>i</mi></msub><mo>></mo><mn>0</mn></mtd></mtr></mtable></mfenced></mrow></math>

这里有一个使用`Lasso`类的小型Scikit-Learn示例：

```py
>>> from sklearn.linear_model import Lasso
>>> lasso_reg = Lasso(alpha=0.1)
>>> lasso_reg.fit(X, y)
>>> lasso_reg.predict([[1.5]])
array([1.53788174])
```

请注意，您也可以使用`SGDRegressor(penalty="l1", alpha=0.1)`。

## 弹性网回归

*弹性网回归*是岭回归和套索回归之间的中间地带。正则化项是岭回归和套索回归正则化项的加权和，您可以控制混合比例*r*。当*r*=0时，弹性网等同于岭回归，当*r*=1时，它等同于套索回归（[方程4-12](#elastic_net_cost_function)）。

##### 方程4-12。弹性网成本函数

<math><mrow><mi>J</mi><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>=</mo><mrow><mtext>MSE</mtext><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>+</mo><mi>r</mi><mfenced><mrow><mn>2</mn><mi>α</mi><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mfenced open="|" close="|"><msub><mi>θ</mi><mi>i</mi></msub></mfenced></mrow></mfenced><mo>+</mo><mo>(</mo><mn>1</mn><mo>-</mo><mi>r</mi><mo>)</mo><mfenced><mrow><mfrac><mi>α</mi><mi>m</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>θ</mi><mi>i</mi><mn>2</mn></msubsup></mrow></mfenced></math>

那么何时使用弹性网回归，或者岭回归、套索回归，或者普通线性回归（即没有任何正则化）？通常最好至少有一点点正则化，因此通常应避免普通线性回归。岭回归是一个很好的默认选择，但如果您怀疑只有少数特征是有用的，您应该更喜欢套索或弹性网，因为它们倾向于将无用特征的权重降至零，正如前面讨论的那样。总的来说，相对于套索，弹性网更受青睐，因为当特征数量大于训练实例数量或者多个特征强相关时，套索可能表现不稳定。

这里有一个使用Scikit-Learn的`ElasticNet`的简短示例（`l1_ratio`对应混合比例*r*）：

```py
>>> from sklearn.linear_model import ElasticNet
>>> elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
>>> elastic_net.fit(X, y)
>>> elastic_net.predict([[1.5]])
array([1.54333232])
```

## 早停

一种非常不同的正则化迭代学习算法（如梯度下降）的方法是在验证错误达到最小值时停止训练。这被称为*早停止*。[图4-20](#early_stopping_plot)显示了一个复杂模型（在本例中，是一个高次多项式回归模型）在我们之前使用的二次数据集上使用批量梯度下降进行训练。随着时代的变迁，算法学习，其在训练集上的预测误差（RMSE）下降，以及在验证集上的预测误差也下降。然而，一段时间后，验证错误停止下降并开始上升。这表明模型已经开始过拟合训练数据。通过早停止，您只需在验证错误达到最小值时停止训练。这是一种简单而高效的正则化技术，Geoffrey Hinton称之为“美丽的免费午餐”。

![mls3 0420](assets/mls3_0420.png)

###### 图4-20。早停止正则化

###### 提示

对于随机梯度下降和小批量梯度下降，曲线不那么平滑，可能很难知道是否已经达到最小值。一个解决方案是只有在验证错误超过最小值一段时间后（当您确信模型不会再有更好的表现时），然后将模型参数回滚到验证错误最小值的点。

这是早停止的基本实现：

```py
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

X_train, y_train, X_valid, y_valid = [...]  # split the quadratic dataset

preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                              StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train)
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = mean_squared_error(y_valid, y_valid_predict, squared=False)
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)
```

这段代码首先添加多项式特征并缩放所有输入特征，对于训练集和验证集都是如此（代码假定您已将原始训练集分成较小的训练集和验证集）。然后它创建一个没有正则化和较小学习率的`SGDRegressor`模型。在训练循环中，它调用`partial_fit()`而不是`fit()`，以执行增量学习。在每个时代，它测量验证集上的RMSE。如果低于迄今为止看到的最低RMSE，则将模型的副本保存在`best_model`变量中。这个实现实际上并没有停止训练，但它允许您在训练后返回到最佳模型。请注意，使用`copy.deepcopy()`复制模型，因为它同时复制了模型的超参数和学习参数。相比之下，`sklearn.base.clone()`只复制模型的超参数。

# 逻辑回归

正如在[第1章](ch01.html#landscape_chapter)中讨论的那样，一些回归算法可以用于分类（反之亦然）。*逻辑回归*（也称为*logit回归*）通常用于估计一个实例属于特定类别的概率（例如，这封电子邮件是垃圾邮件的概率是多少？）。如果估计的概率大于给定阈值（通常为50%），则模型预测该实例属于该类别（称为*正类*，标记为“1”），否则预测它不属于该类别（即属于*负类*，标记为“0”）。这使其成为一个二元分类器。

## 估计概率

那么逻辑回归是如何工作的呢？就像线性回归模型一样，逻辑回归模型计算输入特征的加权和（加上偏置项），但是不像线性回归模型直接输出结果，它输出这个结果的*逻辑*（参见[方程4-13](#logisticregression_model_estimated_probability_vectorized_form)）。

##### 方程4-13。逻辑回归模型估计概率（向量化形式）

<math display="block"><mrow><mover accent="true"><mi>p</mi> <mo>^</mo></mover> <mo>=</mo> <msub><mi>h</mi> <mi mathvariant="bold">θ</mi></msub> <mrow><mo>(</mo> <mi mathvariant="bold">x</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>σ</mi> <mrow><mo>(</mo> <msup><mi mathvariant="bold">θ</mi> <mo>⊺</mo></msup> <mi mathvariant="bold">x</mi> <mo>)</mo></mrow></mrow></math>

逻辑函数 *σ*(·) 是一个 *S* 形函数，输出介于 0 和 1 之间的数字。它的定义如 [方程式 4-14](#equation_four_fourteen) 和 [图 4-21](#logistic_function_plot) 所示。

##### 方程式 4-14\. 逻辑函数

<math display="block"><mrow><mi>σ</mi> <mrow><mo>(</mo> <mi>t</mi> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>1</mn> <mrow><mn>1</mn><mo>+</mo><mo form="prefix">exp</mo><mo>(</mo><mo>-</mo><mi>t</mi><mo>)</mo></mrow></mfrac></mstyle></mrow></math>![mls3 0421](assets/mls3_0421.png)

###### 图 4-21\. 逻辑函数

逻辑回归模型一旦估计出概率 <math><mover><mi>p</mi><mo>^</mo></mover></math> = *h*[**θ**](**x**)，即实例 **x** 属于正类的概率，它可以轻松地进行预测 *ŷ*（见 [方程式 4-15](#equation_four_fifteen)）。

##### 方程式 4-15\. 使用 50% 阈值概率的逻辑回归模型预测

<math display="block"><mrow><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mo>=</mo> <mfenced separators="" open="{" close=""><mtable><mtr><mtd columnalign="left"><mn>0</mn></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mover accent="true"><mi>p</mi> <mo>^</mo></mover> <mo><</mo> <mn>0.5</mn></mrow></mtd></mtr> <mtr><mtd columnalign="left"><mn>1</mn></mtd> <mtd columnalign="left"><mrow><mtext>if</mtext> <mover accent="true"><mi>p</mi> <mo>^</mo></mover> <mo>≥</mo> <mn>0.5</mn></mrow></mtd></mtr></mtable></mfenced></mrow></math>

注意到当 *t* < 0 时，*σ*(*t*) < 0.5，当 *t* ≥ 0 时，*σ*(*t*) ≥ 0.5，因此使用默认的 50% 概率阈值的逻辑回归模型会在 **θ**^⊺ **x** 为正时预测为 1，为负时预测为 0。

###### 注意

得分 *t* 通常被称为 *对数几率*。这个名字来自于对数几率函数的定义，即 logit(*p*) = log(*p* / (1 – *p*))，它是逻辑函数的反函数。实际上，如果计算估计概率 *p* 的对数几率，你会发现结果是 *t*。对数几率也被称为 *对数几率比*，因为它是正类估计概率与负类估计概率之间的比值的对数。

## 训练和成本函数

现在你知道逻辑回归模型如何估计概率并进行预测了。但是它是如何训练的呢？训练的目标是设置参数向量 **θ**，使模型为正实例（*y* = 1）估计出高概率，为负实例（*y* = 0）估计出低概率。这个想法被 [方程式 4-16](#cost_function_of_a_single_training_instance) 中的成本函数所捕捉，针对单个训练实例 **x**。

##### 方程式 4-16\. 单个训练实例的成本函数

<math><mrow><mi>c</mi><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>=</mo><mrow><mfenced open="{" close=""><mtable><mtr><mtd><mo>-</mo><mi>log</mi><mo>(</mo><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo>)</mo></mtd><mtd><mtext>if </mtext><mi>y</mi><mo>=</mo><mn>1</mn></mtd></mtr><mtr><mtd><mo>-</mo><mi>log</mi><mo>(</mo><mn>1</mn><mo>-</mo><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo>)</mo></mtd><mtd><mtext>if </mtext><mi>y</mi><mo>=</mo><mn>0</mn></mtd></mtr></mtable></mfenced></mrow></math>

这个成本函数是有意义的，因为当 *t* 接近 0 时，–log(*t*) 会变得非常大，所以如果模型为正实例估计出接近 0 的概率，成本会很大，如果模型为负实例估计出接近 1 的概率，成本也会很大。另一方面，当 *t* 接近 1 时，–log(*t*) 接近 0，所以如果负实例的估计概率接近 0，或者正实例的估计概率接近 1，成本会接近 0，这正是我们想要的。

整个训练集上的成本函数是所有训练实例的平均成本。它可以用一个称为*对数损失*的单个表达式来表示，如[方程4-17](#logistic_regression_cost_function)所示。

##### 方程4-17。逻辑回归成本函数（对数损失）

<math><mrow><mi>J</mi><mo>(</mo><mi mathvariant="bold">θ</mi><mo>)</mo></mrow><mo>=</mo><mrow><mo>-</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mfenced open="[" close="]"><mrow><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mrow></msup><mi>l</mi><mi>o</mi><mi>g</mi><mfenced><msup><mover accent="true"><mi>p</mi><mo>^</mo></mover><mrow><mo>(</mo><mi>i</mi><mo>)</mrow></msup></mfenced><mo>+</mo><mo>(</mo><mn>1</mn><mo>-</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mrow></msup><mo>)</mo><mi>l</mi><mi>o</mi><mi>g</mi><mfenced><mrow><mn>1</mn><mo>-</mo><msup><mover accent="true"><mi>p</mi><mo>^</mo></mover><mrow><mo>(</mo><mi>i</mi><mo>)</mrow></msup></mrow></mfenced></mrow></mfenced></mrow></math>

###### 警告

对数损失不是凭空想出来的。可以用数学方法（使用贝叶斯推断）证明，最小化这种损失将导致具有最大可能性的模型是最优的，假设实例围绕其类的平均值遵循高斯分布。当您使用对数损失时，这是您所做的隐含假设。这种假设错误越大，模型就会越有偏见。同样，当我们使用MSE来训练线性回归模型时，我们隐含地假设数据是纯线性的，再加上一些高斯噪声。因此，如果数据不是线性的（例如，如果是二次的），或者噪声不是高斯的（例如，如果异常值不是指数稀有的），那么模型就会有偏见。

坏消息是，没有已知的闭式方程可以计算最小化这个成本函数的**θ**的值（没有等价于正规方程）。但好消息是，这个成本函数是凸的，因此梯度下降（或任何其他优化算法）保证会找到全局最小值（如果学习率不是太大，并且等待足够长的时间）。成本函数对于*j*^(th)模型参数*θ*[*j*]的偏导数由[方程4-18](#logistic_cost_function_partial_derivatives)给出。

##### 方程4-18。逻辑成本函数偏导数

数学显示="block"><mrow><mstyle scriptlevel="0" displaystyle="true"><mfrac><mi>∂</mi> <mrow><mi>∂</mi><msub><mi>θ</mi> <mi>j</mi></msub></mrow></mfrac></mstyle> <mtext>J</mtext> <mrow><mo>(</mo> <mi mathvariant="bold">θ</mi> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>1</mn> <mi>m</mi></mfrac></mstyle> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>m</mi></munderover> <mfenced separators="" open="(" close=")"><mi>σ</mi> <mrow><mo>(</mo> <msup><mi mathvariant="bold">θ</mi> <mo>⊺</mo></msup> <msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup> <mo>)</mo></mrow> <mo>-</mo> <msup><mi>y</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mrow></msup></mfenced> <msubsup><mi>x</mi> <mi>j</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow></math>

这个方程看起来非常像[方程4-5](#mse_partial_derivatives)：对于每个实例，它计算预测误差并将其乘以*j*^(th)特征值，然后计算所有训练实例的平均值。一旦有包含所有偏导数的梯度向量，您就可以在批量梯度下降算法中使用它。就是这样：您现在知道如何训练逻辑回归模型了。对于随机梯度下降，您将一次处理一个实例，对于小批量梯度下降，您将一次处理一个小批量。

## 决策边界

我们可以使用鸢尾花数据集来说明逻辑回归。这是一个包含150朵三种不同物种鸢尾花（*Iris setosa*、*Iris versicolor*和*Iris virginica*）的萼片和花瓣长度和宽度的著名数据集（参见[图4-22](#iris_dataset_diagram)）。

![mls3 0422](assets/mls3_0422.png)

###### 图4-22。三种鸢尾植物物种的花朵⁠^([12](ch04.html#idm45720214766432))

让我们尝试构建一个基于花瓣宽度特征的分类器来检测*Iris virginica*类型。第一步是加载数据并快速查看：

```py
>>> from sklearn.datasets import load_iris
>>> iris = load_iris(as_frame=True)
>>> list(iris)
['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names',
 'filename', 'data_module']
>>> iris.data.head(3)
 sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
>>> iris.target.head(3)  # note that the instances are not shuffled
0    0
1    0
2    0
Name: target, dtype: int64
>>> iris.target_names
array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```

接下来我们将拆分数据并在训练集上训练逻辑回归模型：

```py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
```

让我们看看模型对花朵的估计概率，这些花朵的花瓣宽度从0厘米到3厘米不等（参见[图4-23](#logistic_regression_plot)）：⁠^([13](ch04.html#idm45720214606864))

```py
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # reshape to get a column vector
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]

plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2,
         label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2,
         label="Decision boundary")
[...] # beautify the figure: add grid, labels, axis, legend, arrows, and samples
plt.show()
```

![mls3 0423](assets/mls3_0423.png)

###### 图4-23。估计的概率和决策边界

*Iris virginica*花朵的花瓣宽度（表示为三角形）范围从1.4厘米到2.5厘米，而其他鸢尾花（用方块表示）通常具有较小的花瓣宽度，范围从0.1厘米到1.8厘米。请注意，存在一些重叠。大约在2厘米以上，分类器非常确信花朵是*Iris virginica*（输出该类的高概率），而在1厘米以下，它非常确信它不是*Iris virginica*（“非Iris virginica”类的高概率）。在这两个极端之间，分类器不确定。但是，如果要求它预测类别（使用`predict()`方法而不是`predict_proba()`方法），它将返回最有可能的类别。因此，在大约1.6厘米处有一个*决策边界*，两个概率都等于50%：如果花瓣宽度大于1.6厘米，分类器将预测花朵是*Iris virginica*，否则它将预测它不是（即使它不太自信）：

```py
>>> decision_boundary
1.6516516516516517
>>> log_reg.predict([[1.7], [1.5]])
array([ True, False])
```

[图4-24](#logistic_regression_contour_plot)显示了相同的数据集，但这次显示了两个特征：花瓣宽度和长度。一旦训练完成，逻辑回归分类器可以根据这两个特征估计新花朵是*Iris virginica*的概率。虚线代表模型估计50%概率的点：这是模型的决策边界。请注意，这是一个线性边界。⁠^([14](ch04.html#idm45720214369424)) 每条平行线代表模型输出特定概率的点，从15%（左下角）到90%（右上角）。所有超过右上线的花朵根据模型有超过90%的概率是*Iris virginica*。

![mls3 0424](assets/mls3_0424.png)

###### 图4-24。线性决策边界

###### 注意

控制Scikit-Learn `LogisticRegression`模型正则化强度的超参数不是`alpha`（像其他线性模型一样），而是它的倒数：`C`。`C`值越高，模型的正则化就越*少*。

与其他线性模型一样，逻辑回归模型可以使用ℓ[1]或ℓ[2]惩罚进行正则化。Scikit-Learn实际上默认添加了ℓ[2]惩罚。

## Softmax回归

逻辑回归模型可以直接泛化为支持多类别，而无需训练和组合多个二元分类器（如[第3章](ch03.html#classification_chapter)中讨论的）。这称为*softmax回归*或*多项式逻辑回归*。

这个想法很简单：给定一个实例**x**，Softmax回归模型首先为每个类别*k*计算一个分数*s*[*k*](**x**)，然后通过应用*softmax函数*（也称为*归一化指数函数*）来估计每个类别的概率。计算*s*[*k*](**x**)的方程应该看起来很熟悉，因为它就像线性回归预测的方程（参见[方程4-19](#softmax_score_for_class_k)）。

##### 方程4-19。类别k的Softmax分数

<math display="block"><mrow><msub><mi>s</mi> <mi>k</mi></msub> <mrow><mo>(</mo> <mi mathvariant="bold">x</mi> <mo>)</mo></mrow> <mo>=</mo> <msup><mrow><mo>(</mo><msup><mi mathvariant="bold">θ</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msup> <mo>)</mo></mrow> <mo>⊺</mo></msup> <mi mathvariant="bold">x</mi></mrow></math>

注意每个类别都有自己专用的参数向量**θ**^((*k*))。所有这些向量通常被存储为*参数矩阵* **Θ** 的行。

一旦你计算出每个类别对于实例**x**的得分，你可以通过将得分通过softmax函数（[方程4-20](#softmax_function)）来估计实例属于类别*k*的概率<math><msub><mover><mi>p</mi><mo>^</mo></mover><mi>k</mi></msub></math>。该函数计算每个得分的指数，然后对它们进行归一化（除以所有指数的和）。这些得分通常被称为对数几率或对数几率（尽管它们实际上是未归一化的对数几率）。

##### 方程4-20\. Softmax函数

<math display="block"><mrow><msub><mover accent="true"><mi>p</mi> <mo>^</mo></mover> <mi>k</mi></msub> <mo>=</mo> <mi>σ</mi> <msub><mfenced separators="" open="(" close=")"><mi mathvariant="bold">s</mi><mo>(</mo><mi mathvariant="bold">x</mi><mo>)</mo></mfenced> <mi>k</mi></msub> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mrow><mo form="prefix">exp</mo><mfenced separators="" open="(" close=")"><msub><mi>s</mi> <mi>k</mi></msub> <mrow><mo>(</mo><mi mathvariant="bold">x</mi><mo>)</mo></mrow></mfenced></mrow> <mrow><munderover><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>K</mi></munderover> <mrow><mo form="prefix">exp</mo><mfenced separators="" open="(" close=")"><msub><mi>s</mi> <mi>j</mi></msub> <mrow><mo>(</mo><mi mathvariant="bold">x</mi><mo>)</mo></mrow></mfenced></mrow></mrow></mfrac></mstyle></mrow></math>

在这个方程中：

+   *K* 是类别的数量。

+   **s**(**x**)是包含实例**x**每个类别得分的向量。

+   *σ*(**s**(**x**))[*k*]是实例**x**属于类别*k*的估计概率，给定该实例每个类别的得分。

就像逻辑回归分类器一样，默认情况下，softmax回归分类器预测具有最高估计概率的类别（即具有最高得分的类别），如[方程4-21](#softmax_regression_classifier_prediction)所示。

##### 方程4-21\. Softmax回归分类器预测

<math display="block"><mrow><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mo>=</mo> <munder><mo form="prefix">argmax</mo> <mi>k</mi></munder> <mi>σ</mi> <msub><mfenced separators="" open="(" close=")"><mi mathvariant="bold">s</mi><mo>(</mo><mi mathvariant="bold">x</mi><mo>)</mo></mfenced> <mi>k</mi></msub> <mo>=</mo> <munder><mo form="prefix">argmax</mo> <mi>k</mi></munder> <msub><mi>s</mi> <mi k</mi></msub> <mrow><mo>(</mo> <mi mathvariant="bold">x</mi> <mo>)</mo></mrow> <mo>=</mo> <munder><mo form="prefix">argmax</mo> <mi>k</mi></munder> <mfenced separators="" open="(" close=")"><msup><mrow><mo>(</mo><msup><mi mathvariant="bold">θ</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msup> <mo>)</mo></mrow> <mo>⊺</mo></msup> <mi mathvariant="bold">x</mi></mfenced></mrow></math>

*argmax*运算符返回最大化函数的变量值。在这个方程中，它返回最大化估计概率*σ*(**s**(**x**))[*k*]的*k*值。

###### 提示

softmax回归分类器一次只预测一个类别（即它是多类别的，而不是多输出的），因此它只能用于具有互斥类别的情况，例如不同种类的植物。你不能用它来识别一张图片中的多个人。

现在你知道模型如何估计概率并进行预测了，让我们来看看训练。目标是让模型估计目标类的概率很高（因此其他类的概率很低）。最小化[方程4-22](#cross_entropy_cost_function)中显示的成本函数，称为*交叉熵*，应该能够实现这个目标，因为当模型估计目标类的概率很低时，它会受到惩罚。交叉熵经常用来衡量一组估计的类别概率与目标类别的匹配程度。

##### 方程4-22. 交叉熵成本函数

<math><mrow><mi>J</mi><mo>(</mo><mi mathvariant="bold">Θ</mi><mo>)</mo></mrow><mo>=</mo><mrow><mo>-</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><munderover><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><msubsup><mi>y</mi><mi>k</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mi>log</mi><mfenced><msubsup><mover accent="true"><mi>p</mi><mo>^</mo></mover><mi>k</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mfenced></mrow></math>

在这个方程中，<math><msubsup><mi>y</mi><mi>k</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></math>是第*i*个实例属于第*k*类的目标概率。一般来说，它要么等于1，要么等于0，取决于实例是否属于该类。

注意，当只有两类（*K* = 2）时，这个成本函数等同于逻辑回归成本函数（对数损失；参见[方程4-17](#logistic_regression_cost_function)）。

这个成本函数关于**θ**^((*k*))的梯度向量由[方程4-23](#cross_entropy_gradient_vector_for_class_k)给出。

##### 方程4-23. 类别k的交叉熵梯度向量

<math display="block"><mrow><msub><mi>∇</mi> <msup><mi mathvariant="bold">θ</mi> <mrow><mo>(</mo><mi>k</mi><mo>)</mo></mrow></msup></msub> <mi>J</mi> <mrow><mo>(</mo> <mi mathvariant="bold">Θ</mi> <mo>)</mo></mrow> <mo>=</mo> <mstyle scriptlevel="0" displaystyle="true"><mfrac><mn>1</mn> <mi>m</mi></mfrac></mstyle> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>m</mi></munderover> <mrow><mfenced separators="" open="(" close=")"><msubsup><mover accent="true"><mi>p</mi> <mo>^</mo></mover> <mi>k</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup> <mo>-</mo> <msubsup><mi>y</mi> <mi>k</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mfenced> <msup><mi mathvariant="bold">x</mi> <mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mrow></mrow></math>

现在你可以计算每个类别的梯度向量，然后使用梯度下降（或任何其他优化算法）来找到最小化成本函数的参数矩阵**Θ**。

让我们使用softmax回归将鸢尾花分类为所有三类。当你在多于两类上训练Scikit-Learn的`LogisticRegression`分类器时，它会自动使用softmax回归（假设你使用`solver="lbfgs"`，这是默认值）。它还默认应用ℓ[2]正则化，你可以使用之前提到的超参数`C`来控制：

```py
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)
```

所以下次当你发现一朵花瓣长5厘米，宽2厘米的鸢尾花时，你可以让你的模型告诉你它是什么类型的鸢尾花，它会以96%的概率回答*Iris virginica*（第2类）（或以4%的概率回答*Iris versicolor*）:

```py
>>> softmax_reg.predict([[5, 2]])
array([2])
>>> softmax_reg.predict_proba([[5, 2]]).round(2)
array([[0\.  , 0.04, 0.96]])
```

[图 4-25](#softmax_regression_contour_plot) 显示了由背景颜色表示的决策边界。请注意，任意两个类之间的决策边界是线性的。图中还显示了*Iris versicolor*类的概率，由曲线表示（例如，标有 0.30 的线表示 30% 概率边界）。请注意，模型可以预测估计概率低于 50% 的类。例如，在所有决策边界相交的点，所有类的估计概率均为 33%。

![mls3 0425](assets/mls3_0425.png)

###### 图 4-25\. Softmax 回归决策边界

在本章中，你学习了训练线性模型的各种方法，包括回归和分类。你使用闭式方程解决线性回归问题，以及梯度下降，并学习了在训练过程中如何向成本函数添加各种惩罚以对模型进行正则化。在此过程中，你还学习了如何绘制学习曲线并分析它们，以及如何实现早期停止。最后，你学习了逻辑回归和 softmax 回归的工作原理。我们已经打开了第一个机器学习黑匣子！在接下来的章节中，我们将打开更多黑匣子，从支持向量机开始。

# 练习

1.  如果你有一个拥有数百万个特征的训练集，你可以使用哪种线性回归训练算法？

1.  假设你的训练集中的特征具有非常不同的尺度。哪些算法可能会受到影响，以及如何受影响？你可以采取什么措施？

1.  在训练逻辑回归模型时，梯度下降是否会陷入局部最小值？

1.  如果让所有梯度下降算法运行足够长的时间，它们会导致相同的模型吗？

1.  假设你使用批量梯度下降，并在每个时期绘制验证误差。如果你注意到验证误差持续上升，可能出现了什么问题？如何解决？

1.  当验证误差上升时，立即停止小批量梯度下降是一个好主意吗？

1.  在我们讨论的梯度下降算法中，哪种算法会最快接近最优解？哪种实际上会收敛？如何使其他算法也收敛？

1.  假设你正在使用多项式回归。你绘制学习曲线并注意到训练误差和验证误差之间存在很大差距。发生了什么？有哪三种方法可以解决这个问题？

1.  假设你正在使用岭回归，并且注意到训练误差和验证误差几乎相等且相当高。你会说模型存在高偏差还是高方差？你应该增加正则化超参数*α*还是减少它？

1.  为什么要使用：

    1.  是否可以使用岭回归代替普通线性回归（即，没有任何正则化）？

    1.  是否可以使用 Lasso 代替岭回归？

    1.  是否可以使用弹性网络代替 Lasso 回归？

1.  假设你想要将图片分类为室内/室外和白天/黑夜。你应该实现两个逻辑回归分类器还是一个 softmax 回归分类器？

1.  使用 NumPy 实现批量梯度下降并进行早期停止以进行 softmax 回归，而不使用 Scikit-Learn。将其应用于鸢尾花数据集等分类任务。

这些练习的解决方案可在本章笔记本的末尾找到，网址为[*https://homl.info/colab3*](https://homl.info/colab3)。

^([1](ch04.html#idm45720217568672-marker)) 闭式方程仅由有限数量的常数、变量和标准操作组成：例如，*a* = sin(*b* – *c*)。没有无限求和、极限、积分等。

^([2](ch04.html#idm45720216856720-marker)) 从技术上讲，它的导数是*Lipschitz连续*的。

^([3](ch04.html#idm45720216853456-marker)) 由于特征 1 较小，改变*θ*[1]以影响成本函数需要更大的变化，这就是为什么碗沿着*θ*[1]轴被拉长的原因。

^([4](ch04.html#idm45720216763152-marker)) Eta（*η*）是希腊字母表的第七个字母。

^([5](ch04.html#idm45720216235312-marker)) 而正规方程只能执行线性回归，梯度下降算法可以用来训练许多其他模型，您将会看到。

^([6](ch04.html#idm45720215648240-marker)) 这种偏差的概念不应与线性模型的偏差项混淆。

^([7](ch04.html#idm45720215617520-marker)) 通常使用符号*J*(**θ**)表示没有简短名称的代价函数；在本书的其余部分中，我经常会使用这种符号。上下文将清楚地表明正在讨论哪个代价函数。

^([8](ch04.html#idm45720215599696-marker)) 范数在[第2章](ch02.html#project_chapter)中讨论。

^([9](ch04.html#idm45720215579520-marker)) 一个全是0的方阵，除了主对角线（从左上到右下）上的1。

^([10](ch04.html#idm45720215546880-marker)) 或者，您可以使用`Ridge`类与`"sag"`求解器。随机平均梯度下降是随机梯度下降的一种变体。有关更多详细信息，请参阅由不列颠哥伦比亚大学的Mark Schmidt等人提出的演示[“使用随机平均梯度算法最小化有限和”](https://homl.info/12)。

^([11](ch04.html#idm45720215341648-marker)) 您可以将非可微点处的次梯度向量视为该点周围梯度向量之间的中间向量。

^([12](ch04.html#idm45720214766432-marker)) 照片来源于相应的维基百科页面。*Iris virginica*照片由Frank Mayfield拍摄（[知识共享署名-相同方式共享 2.0](https://creativecommons.org/licenses/by-sa/2.0)），*Iris versicolor*照片由D. Gordon E. Robertson拍摄（[知识共享署名-相同方式共享 3.0](https://creativecommons.org/licenses/by-sa/3.0)），*Iris setosa*照片为公共领域。

^([13](ch04.html#idm45720214606864-marker)) NumPy的`reshape()`函数允许一个维度为-1，表示“自动”：该值是从数组的长度和剩余维度推断出来的。

^([14](ch04.html#idm45720214369424-marker)) 它是一组点**x**，使得*θ*[0] + *θ*[1]*x*[1] + *θ*[2]*x*[2] = 0，这定义了一条直线。
