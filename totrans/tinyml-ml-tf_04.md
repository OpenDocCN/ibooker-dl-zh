# 第四章：TinyML 的“Hello World”：构建和训练模型

在第三章中，我们学习了机器学习的基本概念以及机器学习项目遵循的一般工作流程。在本章和下一章中，我们将开始将我们的知识付诸实践。我们将从头开始构建和训练一个模型，然后将其集成到一个简单的微控制器程序中。

在这个过程中，您将通过一些强大的开发者工具亲自动手，这些工具每天都被尖端机器学习从业者使用。您还将学习如何将机器学习模型集成到 C++程序中，并将其部署到微控制器以控制电路中的电流。这可能是您第一次尝试混合硬件和机器学习，应该很有趣！

您可以在 Mac、Linux 或 Windows 机器上测试我们在这些章节中编写的代码，但要获得完整的体验，您需要其中一个嵌入式设备，如“需要哪些硬件？”中提到的。

+   [Arduino Nano 33 BLE Sense](https://oreil.ly/6qlMD)

+   [SparkFun Edge](https://oreil.ly/-hoL-)

+   [ST Microelectronics STM32F746G Discovery kit](https://oreil.ly/cvm4J)

创建我们的机器学习模型，我们将使用 Python、TensorFlow 和 Google 的 Colaboratory，这是一个基于云的交互式笔记本，用于尝试 Python 代码。这些是真实世界中机器学习工程师最重要的工具之一，而且它们都是免费使用的。

###### 注意

想知道本章标题的含义吗？在编程中，引入新技术通常会附带演示如何做一些非常简单的事情的示例代码。通常，这个简单的任务是使程序输出“Hello, world.”这些词。在机器学习中没有明确的等价物，但我们使用术语“hello world”来指代一个简单、易于阅读的端到端 TinyML 应用程序的示例。

在本章的过程中，我们将执行以下操作：

1.  获取一个简单的数据集。

1.  训练一个深度学习模型。

1.  评估模型的性能。

1.  将模型转换为在设备上运行。

1.  编写代码执行设备推断。

1.  将代码构建成二进制文件。

1.  将二进制部署到微控制器。

我们将使用的所有代码都可以在[TensorFlow 的 GitHub 存储库](https://oreil.ly/TQ4CC)中找到。

我们建议您逐步阅读本章的每个部分，然后尝试运行代码。沿途会有如何操作的说明。但在我们开始之前，让我们讨论一下我们要构建的内容。

# 我们正在构建什么

在第三章中，我们讨论了深度学习网络如何学习模拟其训练数据中的模式，以便进行预测。现在我们将训练一个网络来模拟一些非常简单的数据。您可能听说过正弦函数。它在三角学中用于帮助描述直角三角形的性质。我们将使用的数据是正弦波，这是通过绘制随时间变化的正弦函数的结果得到的图形（请参见图 4-1）。

我们的目标是训练一个模型，可以接受一个值`x`，并预测其正弦值`y`。在实际应用中，如果您需要`x`的正弦值，您可以直接计算它。然而，通过训练一个模型来近似结果，我们可以演示机器学习的基础知识。

我们项目的第二部分将是在硬件设备上运行这个模型。从视觉上看，正弦波是一个愉悦的曲线，从-1 平稳地运行到 1，然后返回。这使得它非常适合控制一个视觉上令人愉悦的灯光秀！我们将使用我们模型的输出来控制一些闪烁的 LED 或图形动画的时间，具体取决于设备的功能。

![随时间变化的正弦函数图](img/timl_0401.png)

###### 图 4-1\. 正弦波

在线，您可以看到这段代码闪烁 SparkFun Edge 的 LED 的[动画 GIF](https://oreil.ly/XhqG9)。图 4-2 是来自此动画的静止图像，显示了设备的几个 LED 灯亮起。这可能不是机器学习的特别有用的应用，但在“hello world”示例的精神中，它简单，有趣，并将有助于演示您需要了解的基本原则。

在我们的基本代码运行后，我们将部署到三种不同的设备：SparkFun Edge，Arduino Nano 33 BLE Sense 和 ST Microelectronics STM32F746G Discovery 套件。

###### 注意

由于 TensorFlow 是一个不断发展的积极开发的开源项目，您可能会注意到此处打印的代码与在线托管的代码之间存在一些细微差异。不用担心，即使有几行代码发生变化，基本原则仍然保持不变。

![显示 SparkFun Edge 带有两个 LED 灯亮起的视频静止图像](img/timl_0402.png)

###### 图 4-2。在 SparkFun Edge 上运行的代码

# 我们的机器学习工具链

为了构建这个项目的机器学习部分，我们正在使用真实世界机器学习从业者使用的相同工具。本节向您介绍这些工具。

## Python 和 Jupyter 笔记本

Python 是机器学习科学家和工程师最喜欢的编程语言。它易于学习，适用于许多不同的应用程序，并且有大量用于涉及数据和数学的有用任务的库。绝大多数深度学习研究都是使用 Python 进行的，研究人员经常发布他们创建的模型的 Python 源代码。

Python 与一种称为[*Jupyter 笔记本*](https://jupyter.org/)结合使用时特别好。这是一种特殊的文档格式，允许您混合编写、图形和代码，可以在点击按钮时运行。Jupyter 笔记本被广泛用作描述、解释和探索机器学习代码和问题的一种方式。

我们将在 Jupyter 笔记本中创建我们的模型，这使我们能够在开发过程中对我们的数据进行可视化。这包括显示显示我们模型准确性和收敛性的图形。

如果您有一些编程经验，Python 易于阅读和学习。您应该能够在没有任何困难的情况下跟随本教程。

## 谷歌 Colaboratory

为了运行我们的笔记本，我们将使用一个名为[Colaboratory](https://oreil.ly/ZV7NK)的工具，简称为*Colab*。Colab 由谷歌制作，它提供了一个在线环境来运行 Jupyter 笔记本。它作为一个免费工具提供，以鼓励机器学习中的研究和开发。

传统上，您需要在自己的计算机上创建一个笔记本。这需要安装许多依赖项，如 Python 库，这可能会让人头疼。与其他人分享结果笔记本也很困难，因为他们可能有不同版本的依赖项，这意味着笔记本可能无法按预期运行。此外，机器学习可能需要大量计算，因此在开发计算机上训练模型可能会很慢。

Colab 允许您在谷歌强大的硬件上免费运行笔记本。您可以从任何网络浏览器编辑和查看您的笔记本，并与其他人分享，他们在运行时保证获得相同的结果。您甚至可以配置 Colab 在专门加速的硬件上运行您的代码，这样可以比普通计算机更快地进行训练。

## TensorFlow 和 Keras

[TensorFlow](https://tensorflow.org)是一套用于构建、训练、评估和部署机器学习模型的工具。最初由谷歌开发，TensorFlow 现在是一个由全球数千名贡献者构建和维护的开源项目。它是最受欢迎和广泛使用的机器学习框架。大多数开发人员通过其 Python 库与 TensorFlow 进行交互。

TensorFlow 可以做很多不同的事情。在本章中，我们将使用[Keras](https://oreil.ly/JgNtS)，这是 TensorFlow 的高级 API，使构建和训练深度学习网络变得容易。我们还将使用[TensorFlow Lite](https://oreil.ly/LbDBK)，这是一组用于在移动和嵌入式设备上部署 TensorFlow 模型的工具，以在设备上运行我们的模型。

第十三章将更详细地介绍 TensorFlow。现在，只需知道它是一个非常强大和行业标准的工具，将在您从初学者到深度学习专家的过程中继续满足您的需求。

# 构建我们的模型

现在我们将逐步介绍构建、训练和转换模型的过程。我们在本章中包含了所有的代码，但您也可以在 Colab 中跟着进行并运行代码。

首先，[加载笔记本](https://oreil.ly/NN6Mj)。页面加载后，在顶部，单击“在 Google Colab 中运行”按钮，如图 4-3 所示。这将把笔记本从 GitHub 复制到 Colab，允许您运行它并进行编辑。

![“在 Google Colab 中运行”按钮](img/timl_0403.png)

###### 图 4-3\. “在 Google Colab 中运行”按钮

默认情况下，除了代码外，笔记本还包含您在运行代码时应该看到的输出样本。由于我们将在本章中运行代码，让我们清除这些输出，使笔记本处于原始状态。要做到这一点，在 Colab 的菜单中，单击“编辑”，然后选择“清除所有输出”，如图 4-4 所示。

![“清除所有输出”选项](img/timl_0404.png)

###### 图 4-4\. “清除所有输出”选项

干得好。我们的笔记本现在已经准备好了！

###### 提示

如果您已经熟悉机器学习、TensorFlow 和 Keras，您可能想直接跳到我们将模型转换为 TensorFlow Lite 使用的部分。在书中，跳到“将模型转换为 TensorFlow Lite”。在 Colab 中，滚动到“转换为 TensorFlow Lite”标题下。

## 导入依赖项

我们的第一个任务是导入我们需要的依赖项。在 Jupyter 笔记本中，代码和文本被安排在*单元格*中。有*代码*单元格，其中包含可执行的 Python 代码，以及*文本*单元格，其中包含格式化的文本。

我们的第一个代码单元格位于“导入依赖项”下面。它设置了我们需要训练和转换模型的所有库。以下是代码：

```py
# TensorFlow is an open source machine learning library
!pip install tensorflow==2.0
import tensorflow as tf
# NumPy is a math library
import numpy as np
# Matplotlib is a graphing library
import matplotlib.pyplot as plt
# math is Python's math library
import math
```

在 Python 中，`import`语句加载一个库，以便我们的代码可以使用它。您可以从代码和注释中看到，这个单元格执行以下操作：

+   使用`pip`安装 TensorFlow 2.0 库，`pip`是 Python 的软件包管理器

+   导入 TensorFlow、NumPy、Matplotlib 和 Python 的`math`库

当我们导入一个库时，我们可以给它一个别名，以便以后容易引用。例如，在前面的代码中，我们使用`import numpy as np`导入 NumPy，并给它别名`np`。当我们在代码中使用它时，可以将其称为`np`。

代码单元格中的代码可以通过单击出现在左上角的按钮来运行，当单元格被选中时会出现该按钮。在“导入依赖项”部分，单击第一个代码单元格的任何位置，使其被选中。图 4-5 显示了选定单元格的外观。

![“导入依赖项”单元格处于选定状态](img/timl_0405.png)

###### 图 4-5\. “导入依赖项”单元格处于选定状态

要运行代码，请单击左上角出现的按钮。当代码正在运行时，按钮将以圆圈的形式显示动画，如图 4-6 所示。

依赖项将开始安装，并会看到一些输出。最终您应该看到以下行，表示库已成功安装：

```py
Successfully installed tensorboard-2.0.0 tensorflow-2.0.0 tensorflow-estimator-2.0.0
```

![“导入依赖项”单元格处于运行状态](img/timl_0406.png)

###### 图 4-6\. “导入依赖项”单元格处于运行状态

在 Colab 中运行一个单元格后，当它不再被选中时，您会看到左上角显示一个`1`，如图 4-7 所示。这个数字是一个计数器，每次运行单元格时都会递增。

![左上角的单元格运行计数器](img/timl_0407.png)

###### 图 4-7\. 左上角的单元格运行计数器

您可以使用这个来了解哪些单元格已经运行过，以及运行了多少次。

## 生成数据

深度学习网络学习对底层数据的模式进行建模。正如我们之前提到的，我们将训练一个网络来模拟由正弦函数生成的数据。这将导致一个模型，可以接受一个值`x`，并预测它的正弦值`y`。

在继续之前，我们需要一些数据。在现实世界的情况下，我们可能会从传感器和生产日志中收集数据。然而，在这个例子中，我们使用一些简单的代码来生成数据集。

接下来的单元格就是这样的。我们的计划是生成 1,000 个代表正弦波上随机点的值。让我们看一下图 4-8 来提醒自己正弦波是什么样子的。

波的每个完整周期称为它的*周期*。从图中，我们可以看到每隔大约六个单位在`x`轴上完成一个完整周期。事实上，正弦波的周期是 2 × π，或 2π。

为了训练完整的正弦波数据，我们的代码将生成从 0 到 2π的随机`x`值。然后将计算每个这些值的正弦值。

![随时间变化的正弦函数图表](img/timl_0401.png)

###### 图 4-8\. 一个正弦波

这是这个单元格的完整代码，它使用 NumPy（我们之前导入的`np`）生成随机数并计算它们的正弦值：

```py
# We'll generate this many sample datapoints
SAMPLES = 1000

# Set a "seed" value, so we get the same random numbers each time we run this
# notebook. Any number can be used here.
SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Generate a uniformly distributed set of random numbers in the range from
# 0 to 2π, which covers a complete sine wave oscillation
x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)

# Shuffle the values to guarantee they're not in order
np.random.shuffle(x_values)

# Calculate the corresponding sine values
y_values = np.sin(x_values)

# Plot our data. The 'b.' argument tells the library to print blue dots.
plt.plot(x_values, y_values, 'b.')
plt.show()
```

除了我们之前讨论的内容，这段代码中还有一些值得指出的地方。首先，您会看到我们使用`np.random.uniform()`来生成我们的`x`值。这个方法返回指定范围内的随机数数组。NumPy 包含许多有用的方法，可以操作整个值数组，这在处理数据时非常方便。

其次，在生成数据后，我们对数据进行了洗牌。这很重要，因为深度学习中使用的训练过程取决于以真正随机的顺序提供数据。如果数据是有序的，那么生成的模型将不够准确。

接下来，请注意我们使用 NumPy 的`sin()`方法来计算正弦值。NumPy 可以一次为所有`x`值执行此操作，返回一个数组。NumPy 太棒了！

最后，您会看到一些神秘的代码调用`plt`，这是我们对 Matplotlib 的别名：

```py
# Plot our data. The 'b.' argument tells the library to print blue dots.
plt.plot(x_values, y_values, 'b.')
plt.show()
```

这段代码是做什么的？它绘制了我们数据的图表。Jupyter 笔记本的一个最好的地方是它们能够显示代码运行输出的图形。Matplotlib 是一个从数据创建图表的优秀工具。由于可视化数据是机器学习工作流程的重要部分，这将在我们训练模型时非常有帮助。

要生成数据并将其呈现为图表，请运行单元格中的代码。代码单元格运行完成后，您应该会看到一个漂亮的图表出现在下面，就像图 4-9 中显示的那样。

![我们生成数据的图表](img/timl_0409.png)

###### 图 4-9\. 我们生成数据的图表

这就是我们的数据！这是沿着一个漂亮、平滑的正弦曲线的随机点的选择。我们可以使用这个来训练我们的模型。然而，这样做太容易了。深度学习网络的一个令人兴奋的地方是它们能够从噪音中提取模式。这使它们能够在训练混乱的真实世界数据时进行预测。为了展示这一点，让我们向我们的数据点添加一些随机噪音并绘制另一个图表：

```py
# Add a small random number to each y value
y_values += 0.1 * np.random.randn(*y_values.shape)

# Plot our data
plt.plot(x_values, y_values, 'b.')
plt.show()
```

运行这个单元格，看看结果，如图 4-10 所示。

更好了！我们的点现在已经随机化，因此它们代表了围绕正弦波的分布，而不是平滑的完美曲线。这更加反映了现实世界的情况，其中数据通常相当混乱。

![我们的数据添加了噪声的图](img/timl_0410.png)

###### 图 4-10。我们的数据添加了噪声

## 分割数据

从上一章，您可能记得数据集通常分为三部分：*训练*、*验证*和*测试*。为了评估我们训练的模型的准确性，我们需要将其预测与真实数据进行比较，并检查它们的匹配程度。

这种评估发生在训练期间（称为验证）和训练之后（称为测试）。在每种情况下，使用的数据都必须是新鲜的，不能已经用于训练模型。

为了确保我们有数据用于评估，我们将在开始训练之前留出一些数据。让我们将我们的数据的 20%保留用于验证，另外 20%用于测试。我们将使用剩下的 60%来训练模型。这是训练模型时常用的典型分割。

以下代码分割我们的数据，然后将每个集合绘制为不同的颜色：

```py
# We'll use 60% of our data for training and 20% for testing. The remaining 20%
# will be used for validation. Calculate the indices of each section.
TRAIN_SPLIT =  int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

# Use np.split to chop our data into three parts.
# The second argument to np.split is an array of indices where the data will be
# split. We provide two indices, so the data will be divided into three chunks.
x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

# Double check that our splits add up correctly
assert (x_train.size + x_validate.size + x_test.size) ==  SAMPLES

# Plot the data in each partition in different colors:
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_validate, y_validate, 'y.', label="Validate")
plt.plot(x_test, y_test, 'r.', label="Test")
plt.legend()
plt.show()
```

为了分割我们的数据，我们使用另一个方便的 NumPy 方法：`split()`。这个方法接受一个数据数组和一个索引数组，然后在提供的索引处将数据分割成部分。

运行此单元格以查看我们分割的结果。每种类型的数据将由不同的颜色表示（或者如果您正在阅读本书的打印版本，则为不同的阴影），如图 4-11 所示。

![我们的数据分为训练、验证和测试集的图](img/timl_0411.png)

###### 图 4-11。我们的数据分为训练、验证和测试集

## 定义基本模型

现在我们有了数据，是时候创建我们将训练以适应它的模型了。

我们将构建一个模型，该模型将接受一个输入值（在本例中为`x`）并使用它来预测一个数值输出值（`x`的正弦）。这种类型的问题称为*回归*。我们可以使用回归模型来处理各种需要数值输出的任务。例如，回归模型可以尝试根据来自加速度计的数据预测一个人的每小时英里数。

为了创建我们的模型，我们将设计一个简单的神经网络。它使用神经元层来尝试学习训练数据中的任何模式，以便进行预测。

实际上，执行此操作的代码非常简单。它使用*Keras*，TensorFlow 的用于创建深度学习网络的高级 API：

```py
# We'll use Keras to create a simple model architecture
from tf.keras import layers
model_1 = tf.keras.Sequential()

# First layer takes a scalar input and feeds it through 16 "neurons." The
# neurons decide whether to activate based on the 'relu' activation function.
model_1.add(layers.Dense(16, activation='relu', input_shape=(1,)))

# Final layer is a single neuron, since we want to output a single value
model_1.add(layers.Dense(1))

# Compile the model using a standard optimizer and loss function for regression
model_1.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Print a summary of the model's architecture
model_1.summary()
```

首先，我们使用 Keras 创建一个`Sequential`模型，这意味着每个神经元层都堆叠在下一个层上，就像我们在图 3-1 中看到的那样。然后我们定义两个层。这是第一层的定义：

```py
model_1.add(layers.Dense(16, activation='relu', input_shape=(1,)))
```

第一层有一个单一的输入—我们的`x`值—和 16 个神经元。这是一个`Dense`层（也称为*全连接*层），意味着在推断时，当我们进行预测时，输入将被馈送到每一个神经元中。然后每个神经元将以一定程度被*激活*。每个神经元的激活程度基于其在训练期间学习到的*权重*和*偏差*值，以及其*激活函数*。神经元的激活作为一个数字输出。

激活是通过一个简单的公式计算的，用 Python 显示。我们永远不需要自己编写这个代码，因为它由 Keras 和 TensorFlow 处理，但随着我们深入学习，了解这个公式将会很有帮助：

```py
activation = activation_function((input * weight) + bias)
```

为了计算神经元的激活，它的输入被权重相乘，偏差被加到结果中。计算出的值被传递到激活函数中。得到的数字是神经元的激活。

激活函数是用于塑造神经元输出的数学函数。在我们的网络中，我们使用了一个称为*修正线性单元*或*ReLU*的激活函数。这在 Keras 中由参数`activation=relu`指定。

ReLU 是一个简单的函数，在 Python 中显示如下：

```py
def relu(input):
    return max(0.0, input)
```

ReLU 返回较大的值：它的输入或零。如果输入值为负，则 ReLU 返回零。如果输入值大于零，则 ReLU 返回不变。

图 4-12 显示了一系列输入值的 ReLU 输出。

![从-10 到 10 的输入的 ReLU 图](img/timl_0412.png)

###### 图 4-12。从-10 到 10 的输入的 ReLU 图

没有激活函数，神经元的输出将始终是其输入的线性函数。这意味着网络只能模拟`x`和`y`之间的比率在整个值范围内保持不变的线性关系。这将阻止网络对我们的正弦波进行建模，因为正弦波是非线性的。

由于 ReLU 是非线性的，它允许多层神经元联合起来模拟复杂的非线性关系，其中`y`值并不是每个`x`增量都增加相同的量。

###### 注意

还有其他激活函数，但 ReLU 是最常用的。您可以在[Wikipedia 关于激活函数的文章](https://oreil.ly/Yxe-N)中看到其他选项。每个激活函数都有不同的权衡，机器学习工程师会进行实验，找出哪些选项对于给定的架构最有效。

来自我们第一层的激活数字将作为输入传递给我们的第二层，该层在以下行中定义：

```py
model_1.add(layers.Dense(1))
```

因为这一层是一个单个神经元，它将接收 16 个输入，每个输入对应前一层中的一个神经元。它的目的是将前一层的所有激活组合成一个单一的输出值。由于这是我们的输出层，我们不指定激活函数，我们只想要原始结果。

因为这个神经元有多个输入，所以它有对应的每个输入的权重值。神经元的输出是通过以下公式计算的，如 Python 中所示：

```py
# Here, `inputs` and `weights` are both NumPy arrays with 16 elements each
output = sum((inputs * weights)) + bias
```

输出值是通过将每个输入与其对应的权重相乘，对结果求和，然后加上神经元的偏差来获得的。

网络的权重和偏差在训练期间学习。在本章前面显示的代码中的`compile()`步骤配置了一些在训练过程中使用的重要参数，并准备好模型进行训练：

```py
model_1.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
```

`optimizer`参数指定了在训练期间调整网络以模拟其输入的算法。有几种选择，找到最佳选择通常归结为实验。您可以在[Keras 文档](https://oreil.ly/oT-pU)中了解选项。

`loss`参数指定了在训练期间使用的方法，用于计算网络预测与现实之间的距离。这种方法称为*损失函数*。在这里，我们使用`mse`，或*均方误差*。这种损失函数用于回归问题，我们试图预测一个数字。Keras 中有各种损失函数可用。您可以在[Keras 文档](https://keras.io/losses)中看到一些选项。

`metrics`参数允许我们指定一些额外的函数，用于评估我们模型的性能。我们指定`mae`，或*平均绝对误差*，这是一个有用的函数，用于衡量回归模型的性能。这个度量将在训练期间进行测量，我们将在训练结束后获得结果。

在编译模型后，我们可以使用以下行打印关于其架构的一些摘要信息：

```py
# Print a summary of the model's architecture
model_1.summary()
```

在 Colab 中运行单元格以定义模型。您将看到以下输出打印：

```py
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 16)                32
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
```

这个表格显示了网络的层、它们的输出形状以及它们的*参数*数量。网络的大小——它占用的内存量——主要取决于它的参数数量，即其总权重和偏差的数量。在讨论模型大小和复杂性时，这可能是一个有用的指标。

对于像我们这样简单的模型，权重的数量可以通过计算模型中神经元之间的连接数来确定，假设每个连接都有一个权重。

我们刚刚设计的网络由两层组成。我们的第一层有 16 个连接——一个连接到每个神经元的输入。我们的第二层有一个神经元，也有 16 个连接——一个连接到第一层的每个神经元。这使得连接的总数为 32。

由于每个神经元都有一个偏差，网络有 17 个偏差，这意味着它总共有 32 + 17 = 49 个参数。

我们现在已经走完了定义我们模型的代码。接下来，我们将开始训练过程。

# 训练我们的模型

定义了我们的模型之后，就是训练它，然后评估其性能，看看它的工作效果如何。当我们看到指标时，我们可以决定是否足够好，或者是否应该对设计进行更改并重新训练。

在 Keras 中训练模型，我们只需调用其`fit()`方法，传递所有数据和一些其他重要参数。下一个单元格中的代码显示了如何：

```py
history_1 = model_1.fit(x_train, y_train, epochs=1000, batch_size=16,
                     validation_data=(x_validate, y_validate))
```

运行单元格中的代码开始训练。您将看到一些日志开始出现：

```py
Train on 600 samples, validate on 200 samples
Epoch 1/1000
600/600 [==============================] - 1s 1ms/sample - loss: 0.7887 - mae: 0.7848 - val_loss: 0.5824 - val_mae: 0.6867
Epoch 2/1000
600/600 [==============================] - 0s 155us/sample - loss: 0.4883 - mae: 0.6194 - val_loss: 0.4742 - val_mae: 0.6056
```

我们的模型现在正在训练。这将需要一些时间，所以在等待时，让我们详细了解我们对`fit()`的调用：

```py
history_1 = model_1.fit(x_train, y_train, epochs=1000, batch_size=16,
                     validation_data=(x_validate, y_validate))
```

首先，您会注意到我们将`fit()`调用的返回值分配给一个名为`history_1`的变量。这个变量包含了关于我们训练运行的大量信息，我们稍后将使用它来调查事情的进展。

接下来，让我们看一下`fit()`函数的参数：

`x_train`，`y_train`

`fit()`的前两个参数是我们训练数据的`x`和`y`值。请记住，我们的数据的部分被保留用于验证和测试，因此只有训练集用于训练网络。

`epochs`

下一个参数指定在训练期间整个训练集将通过网络运行多少次。时期越多，训练就越多。您可能会认为训练次数越多，网络就会越好。然而，一些网络在一定数量的时期后会开始过拟合其训练数据，因此我们可能希望限制我们进行的训练量。

此外，即使没有过拟合，网络在一定数量的训练后也会停止改进。由于训练需要时间和计算资源，最好不要在网络没有变得更好的情况下进行训练！

我们开始使用 1,000 个时期进行训练。训练完成后，我们可以深入研究我们的指标，以发现这是否是正确的数量。

`batch_size`

`batch_size`参数指定在测量准确性并更新权重和偏差之前要向网络提供多少训练数据。如果需要，我们可以指定`batch_size`为`1`，这意味着我们将在单个数据点上运行推断，测量网络预测的损失，更新权重和偏差以使下次预测更准确，然后继续这个循环直到处理完所有数据。

因为我们有 600 个数据点，每个时期会导致网络更新 600 次。这是很多计算量，所以我们的训练会花费很长时间！另一种选择可能是选择并对多个数据点运行推断，测量总体损失，然后相应地更新网络。

如果将`batch_size`设置为`600`，每个批次将包括所有训练数据。现在，我们每个时代只需要对网络进行一次更新，速度更快。问题是，这会导致模型的准确性降低。研究表明，使用大批量大小训练的模型对新数据的泛化能力较差，更容易过拟合。

妥协的方法是使用一个介于中间的批量大小。在我们的训练代码中，我们使用批量大小为 16。这意味着我们会随机选择 16 个数据点，对它们进行推断，计算总体损失，并每批次更新一次网络。如果我们有 600 个训练数据点，网络将在每个时代更新大约 38 次，这比 600 次要好得多。

在选择批量大小时，我们在训练效率和模型准确性之间做出妥协。理想的批量大小会因模型而异。最好从批量大小为 16 或 32 开始，并进行实验以找出最佳工作方式。

`验证数据`

这是我们指定验证数据集的地方。来自该数据集的数据将在整个训练过程中通过网络运行，并且网络的预测将与预期值进行比较。我们将在日志中看到验证结果，并作为`history_1`对象的一部分。

## 训练指标

希望到目前为止，培训已经结束。如果没有，请等待一段时间以完成培训。

我们现在将检查各种指标，以查看我们的网络学习情况如何。首先，让我们查看训练期间编写的日志。这将显示网络如何从其随机初始状态改进。

这是我们第一个和最后一个时代的日志：

```py
Epoch 1/1000
600/600 [==============================] - 1s 1ms/sample - loss: 0.7887 - mae: 0.7848 - val_loss: 0.5824 - val_mae: 0.6867
```

```py
Epoch 1000/1000
600/600 [==============================] - 0s 124us/sample - loss: 0.1524 - mae: 0.3039 - val_loss: 0.1737 - val_mae: 0.3249
```

`损失`，`mae`，`val_loss`和`val_mae`告诉我们各种事情：

`损失`

这是我们损失函数的输出。我们使用均方误差，它表示为正数。通常，损失值越小，越好，因此在评估网络时观察这一点是一个好方法。

比较第一个和最后一个时代，网络在训练过程中显然有所改进，从约 0.7 的损失到更小的约 0.15。让我们看看其他数字，以确定这种改进是否足够！

`mae`

这是我们训练数据的平均绝对误差。它显示了网络预测值与训练数据中预期`y`值之间的平均差异。

可以预期我们的初始误差会非常糟糕，因为它基于未经训练的网络。这当然是事实：网络的预测平均偏差约为 0.78，这是一个很大的数字，当可接受值的范围仅为-1 到 1 时！

然而，即使在训练之后，我们的平均绝对误差仍然约为 0.30。这意味着我们的预测平均偏差约为 0.30，这仍然相当糟糕。

`val_loss`

这是我们验证数据上损失函数的输出。在我们的最后一个时代中，训练损失（约 0.15）略低于验证损失（约 0.17）。这暗示我们的网络可能存在过拟合问题，因为它在未见过的数据上表现更差。

`val_mae`

这是我们验证数据的平均绝对误差。值为约 0.32，比我们训练集上的平均绝对误差更糟糕，这是网络可能存在过拟合的另一个迹象。

## 绘制历史数据

到目前为止，很明显我们的模型并没有做出准确的预测。我们现在的任务是找出原因。为此，让我们利用我们`history_1`对象中收集的数据。

下一个单元格从历史对象中提取训练和验证损失数据，并将其绘制在图表上：

```py
loss = history_1.history['loss']
val_loss = history_1.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

`history_1`对象包含一个名为`history_1.history`的属性，这是一个记录训练和验证期间指标值的字典。我们使用这个来收集我们要绘制的数据。对于我们的 x 轴，我们使用时期数，通过查看损失数据点的数量来确定。运行单元格，您将在图 4-13 中看到图形。

![训练和验证损失的图形](img/timl_0413.png)

###### 图 4-13。训练和验证损失的图形

正如您所看到的，损失量在前 50 个时期内迅速减少，然后趋于稳定。这意味着模型正在改进并产生更准确的预测。

我们的目标是在模型不再改进或训练损失小于验证损失时停止训练，这意味着模型已经学会如此好地预测训练数据，以至于无法推广到新数据。

损失在最初几个时期急剧下降，这使得其余的图表非常难以阅读。让我们通过运行下一个单元格来跳过前 100 个时期：

```py
# Exclude the first few epochs so the graph is easier to read
SKIP = 100

plt.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='Training loss')
plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

图 4-14 展示了此单元格生成的图形。

![跳过前 100 个时期的训练和验证损失图](img/timl_0414.png)

###### 图 4-14。跳过前 100 个时期的训练和验证损失图

现在我们已经放大了，您可以看到损失继续减少直到大约 600 个时期，此时它基本稳定。这意味着可能没有必要训练我们的网络那么长时间。

但是，您还可以看到最低的损失值仍然约为 0.15。这似乎相对较高。此外，验证损失值始终更高。

为了更深入地了解我们模型的性能，我们可以绘制更多数据。这次，让我们绘制平均绝对误差。运行下一个单元格来执行：

```py
# Draw a graph of mean absolute error, which is another way of
# measuring the amount of error in the prediction.
mae = history_1.history['mae']
val_mae = history_1.history['val_mae']

plt.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='Training MAE')
plt.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
plt.title('Training and validation mean absolute error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
```

图 4-15 显示了结果图形。

![训练和验证期间的平均绝对误差图](img/timl_0415.png)

###### 图 4-15。训练和验证期间的平均绝对误差图

这个平均绝对误差图给了我们一些进一步的线索。我们可以看到，平均而言，训练数据显示的误差比验证数据低，这意味着网络可能已经过拟合，或者学习了训练数据，以至于无法对新数据做出有效预测。

此外，平均绝对误差值相当高，约为 0.31 左右，这意味着模型的一些预测至少有 0.31 的错误。由于我们的预期值的范围仅为-1 到+1，0.31 的误差意味着我们离准确建模正弦波还有很大距离。

为了更深入了解发生了什么，我们可以将网络对训练数据的预测与预期值绘制在一起。

这发生在以下单元格中：

```py
# Use the model to make predictions from our validation data
predictions = model_1.predict(x_train)

# Plot the predictions along with the test data
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(x_test, y_test, 'b.', label='Actual')
plt.plot(x_train, predictions, 'r.', label='Predicted')
plt.legend()
plt.show()
```

通过调用`model_1.predict(x_train)`，我们对训练数据中的所有`x`值进行推断。该方法返回一个预测数组。让我们将这个绘制在图上，与我们训练集中的实际`y`值一起。运行单元格，您将在图 4-16 中看到图形。

![我们训练数据的预测与实际值的图形](img/timl_0416.png)

###### 图 4-16。我们训练数据的预测与实际值的图形

哦，亲爱的！图表清楚地表明我们的网络已经学会以非常有限的方式逼近正弦函数。预测非常线性，只是非常粗略地拟合数据。

这种拟合的刚性表明模型没有足够的容量来学习正弦波函数的全部复杂性，因此它只能以过于简单的方式逼近它。通过使我们的模型更大，我们应该能够提高其性能。

## 改进我们的模型

凭借我们的原始模型太小无法学习数据的复杂性的知识，我们可以尝试改进它。这是机器学习工作流程的正常部分：设计模型，评估其性能，并进行更改，希望看到改进。

扩大网络的简单方法是添加另一层神经元。每一层神经元代表输入的转换，希望能使其更接近预期的输出。网络有更多层神经元，这些转换就可以更复杂。

运行以下单元格以重新定义我们的模型，方式与之前相同，但在中间增加了 16 个神经元的额外层：

```py
model_2 = tf.keras.Sequential()

# First layer takes a scalar input and feeds it through 16 "neurons." The
# neurons decide whether to activate based on the 'relu' activation function.
model_2.add(layers.Dense(16, activation='relu', input_shape=(1,)))

# The new second layer may help the network learn more complex representations
model_2.add(layers.Dense(16, activation='relu'))

# Final layer is a single neuron, since we want to output a single value
model_2.add(layers.Dense(1))

# Compile the model using a standard optimizer and loss function for regression
model_2.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Show a summary of the model
model_2.summary()
```

正如您所看到的，代码基本上与我们第一个模型相同，但增加了一个`Dense`层。让我们运行这个单元格来查看`summary()`结果：

```py
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_2 (Dense)              (None, 16)                32
_________________________________________________________________
dense_3 (Dense)              (None, 16)                272
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 17
=================================================================
Total params: 321
Trainable params: 321
Non-trainable params: 0
_________________________________________________________________
```

有了 16 个神经元的两层，我们的新模型要大得多。它有(1 * 16) + (16 * 16) + (16 * 1) = 288 个权重，加上 16 + 16 + 1 = 33 个偏差，总共是 288 + 33 = 321 个参数。我们的原始模型只有 49 个总参数，因此模型大小增加了 555%。希望这种额外的容量将有助于表示数据的复杂性。

接下来的单元格将训练我们的新模型。由于我们的第一个模型改进得太快，这次让我们训练更少的时代——只有 600 个。运行这个单元格开始训练：

```py
history_2 = model_2.fit(x_train, y_train, epochs=600, batch_size=16,
                     validation_data=(x_validate, y_validate))
```

训练完成后，我们可以查看最终日志，快速了解事情是否有所改善：

```py
Epoch 600/600
600/600 [==============================] - 0s 150us/sample - loss: 0.0115 - mae: 0.0859 - val_loss: 0.0104 - val_mae: 0.0806
```

哇！您可以看到我们已经取得了巨大的进步——验证损失从 0.17 降至 0.01，验证平均绝对误差从 0.32 降至 0.08。这看起来非常有希望。

为了了解情况如何，让我们运行下一个单元格。它设置为生成我们上次使用的相同图表。首先，我们绘制损失的图表：

```py
# Draw a graph of the loss, which is the distance between
# the predicted and actual values during training and validation.
loss = history_2.history['loss']
val_loss = history_2.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

图 4-17 显示了结果。

接下来，我们绘制相同的损失图，但跳过前 100 个时代，以便更好地看到细节：

```py
# Exclude the first few epochs so the graph is easier to read
SKIP = 100

plt.clf()

plt.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='Training loss')
plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

![训练和验证损失的图表](img/timl_0417.png)

###### 图 4-17。训练和验证损失的图表

图 4-18 展示了输出。

最后，我们绘制相同一组时代的平均绝对误差：

```py
plt.clf()

# Draw a graph of mean absolute error, which is another way of
# measuring the amount of error in the prediction.
mae = history_2.history['mae']
val_mae = history_2.history['val_mae']

plt.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='Training MAE')
plt.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
plt.title('Training and validation mean absolute error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
```

![训练和验证损失的图表，跳过前 100 个时代](img/timl_0418.png)

###### 图 4-18。训练和验证损失的图表，跳过前 100 个时代

图 4-19 描述了图表。

![训练和验证期间的平均绝对误差图](img/timl_0419.png)

###### 图 4-19。训练和验证期间的平均绝对误差图

很棒的结果！从这些图表中，我们可以看到两个令人兴奋的事情：

+   验证的指标比训练的要好，这意味着网络没有过拟合。

+   总体损失和平均绝对误差比我们之前的网络要好得多。

您可能想知道为什么验证的指标比训练的好，而不仅仅是相同的。原因是验证指标是在每个时代结束时计算的，而训练指标是在训练时代仍在进行时计算的。这意味着验证是在一个训练时间稍长的模型上进行的。

根据我们的验证数据，我们的模型似乎表现很好。然而，为了确保这一点，我们需要进行最后一次测试。

## 测试

之前，我们留出了 20%的数据用于测试。正如我们讨论过的，拥有单独的验证和测试数据非常重要。由于我们根据验证性能微调我们的网络，存在一个风险，即我们可能会意外地调整模型以过度拟合其验证集，并且可能无法推广到新数据。通过保留一些新鲜数据并将其用于对模型的最终测试，我们可以确保这种情况没有发生。

在使用了我们的测试数据之后，我们需要抵制进一步调整模型的冲动。如果我们为了提高测试性能而进行更改，可能会导致过拟合测试集。如果这样做了，我们将无法知道，因为我们没有剩余的新数据来进行测试。

这意味着如果我们的模型在测试数据上表现不佳，那么是时候重新考虑了。我们需要停止优化当前模型，并提出全新的架构。

考虑到这一点，接下来的单元将评估我们的模型与测试数据的表现：

```py
# Calculate and print the loss on our test dataset
loss = model_2.evaluate(x_test, y_test)

# Make predictions based on our test dataset
predictions = model_2.predict(x_test)

# Graph the predictions against the actual values
plt.clf()
plt.title('Comparison of predictions and actual values')
plt.plot(x_test, y_test, 'b.', label='Actual')
plt.plot(x_test, predictions, 'r.', label='Predicted')
plt.legend()
plt.show()
```

首先，我们使用测试数据调用模型的`evaluate()`方法。这将计算并打印损失和平均绝对误差指标，告诉我们模型的预测与实际值的偏差有多大。接下来，我们进行一组预测，并将其与实际值一起绘制在图表上。

现在我们可以运行单元，了解我们的模型表现如何！首先，让我们看看`evaluate()`的结果：

```py
200/200 [==============================] - 0s 71us/sample - loss: 0.0103 - mae: 0.0718
```

这显示有 200 个数据点被评估，这是我们整个测试集。模型每次预测需要 71 微秒。损失指标为 0.0103，非常出色，并且非常接近我们的验证损失 0.0104。我们的平均绝对误差为 0.0718，也非常小，与验证中的 0.0806 相当接近。

这意味着我们的模型运行良好，没有过拟合！如果模型过拟合了验证数据，我们可以预期测试集上的指标会明显比验证结果差。

我们的预测与实际值的图表，显示在图 4-20 中，清楚地展示了我们的模型表现如何。

![我们的测试数据的预测与实际值的图表](img/timl_0420.png)

###### 图 4-20。我们的测试数据的预测与实际值的图表

你可以看到，大部分情况下，代表*预测*值的点形成了一个平滑的曲线，沿着*实际*值的分布中心。我们的网络已经学会了近似正弦曲线，即使数据集很嘈杂！

然而，仔细观察，你会发现一些不完美之处。我们预测的正弦波的峰值和谷值并不完全平滑，像真正的正弦波那样。我们模型学习了训练数据的变化，这些数据是随机分布的。这是过拟合的轻微情况：我们的模型没有学习到平滑的正弦函数，而是学会了复制数据的确切形状。

对于我们的目的，这种过拟合并不是一个主要问题。我们的目标是让这个模型轻轻地控制 LED 的亮度，不需要完全平滑才能实现这一目标。如果我们认为过拟合的程度有问题，我们可以尝试通过正则化技术或获取更多的训练数据来解决。

现在我们对模型满意了，让我们准备在设备上部署它！

# 将模型转换为 TensorFlow Lite

在本章的开头，我们简要提到了 TensorFlow Lite，这是一组用于在“边缘设备”上运行 TensorFlow 模型的工具。

第十三章详细介绍了用于微控制器的 TensorFlow Lite。目前，我们可以将其视为具有两个主要组件：

TensorFlow Lite 转换器

这将 TensorFlow 模型转换为一种特殊的、节省空间的格式，以便在内存受限设备上使用，并且可以应用优化，进一步减小模型大小，并使其在小型设备上运行更快。

TensorFlow Lite 解释器

这将使用给定设备的最有效操作来运行适当转换为 TensorFlow Lite 模型。

在使用 TensorFlow Lite 之前，我们需要将模型转换。我们使用 TensorFlow Lite 转换器的 Python API 来完成这个任务。它将我们的 Keras 模型写入磁盘，以*FlatBuffer*的形式，这是一种专门设计的节省空间的文件格式。由于我们要部署到内存有限的设备，这将非常有用！我们将在第十二章中更详细地了解 FlatBuffers。

除了创建 FlatBuffer 外，TensorFlow Lite 转换器还可以对模型应用优化。这些优化通常会减小模型的大小、运行时间，或者两者兼而有之。这可能会导致准确度降低，但降低通常是小到足以值得的。您可以在第十三章中了解更多关于优化的信息。

最有用的优化之一是*量化*。默认情况下，模型中的权重和偏置以 32 位浮点数存储，以便在训练期间进行高精度计算。量化允许您减少这些数字的精度，使其适合于 8 位整数——大小减小四倍。更好的是，因为 CPU 更容易使用整数而不是浮点数进行数学运算，量化模型将运行得更快。

量化最酷的一点是，它通常会导致准确度的最小损失。这意味着在部署到低内存设备时，几乎总是值得的。

在下一个单元格中，我们使用转换器创建并保存我们模型的两个新版本。第一个转换为 TensorFlow Lite FlatBuffer 格式，但没有任何优化。第二个是量化的。

运行单元格将模型转换为这两种变体：

```py
# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
tflite_model = converter.convert()

# Save the model to disk
open("sine_model.tflite," "wb").write(tflite_model)

# Convert the model to the TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
# Indicate that we want to perform the default optimizations,
# which include quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Define a generator function that provides our test data's x values
# as a representative dataset, and tell the converter to use it
def representative_dataset_generator():
  for value in x_test:
    # Each scalar value must be inside of a 2D array that is wrapped in a list
    yield [np.array(value, dtype=np.float32, ndmin=2)]
converter.representative_dataset = representative_dataset_generator
# Convert the model
tflite_model = converter.convert()

# Save the model to disk
open("sine_model_quantized.tflite," "wb").write(tflite_model)
```

为了创建一个尽可能高效运行的量化模型，我们需要提供一个*代表性数据集*——一组数字，代表了模型训练时数据集的全部输入值范围。

在前面的单元格中，我们可以使用测试数据集的`x`值作为代表性数据集。我们定义一个函数`representative_dataset_generator()`，使用`yield`操作符逐个返回这些值。

为了证明这些模型在转换和量化后仍然准确，我们使用它们进行预测，并将结果与我们的测试结果进行比较。鉴于这些是 TensorFlow Lite 模型，我们需要使用 TensorFlow Lite 解释器来执行此操作。

由于 TensorFlow Lite 解释器主要设计用于效率，因此使用起来比 Keras API 稍微复杂一些。要使用我们的 Keras 模型进行预测，我们只需调用`predict()`方法，传递一个输入数组即可。而对于 TensorFlow Lite，我们需要执行以下操作：

1.  实例化一个`Interpreter`对象。

1.  调用一些为模型分配内存的方法。

1.  将输入写入输入张量。

1.  调用模型。

1.  从输出张量中读取输出。

这听起来很多，但现在不要太担心；我们将在第五章中详细介绍。现在，运行以下单元格，使用两个模型进行预测，并将它们与原始未转换的模型的结果一起绘制在图表上：

```py
# Instantiate an interpreter for each model
sine_model = tf.lite.Interpreter('sine_model.tflite')
sine_model_quantized = tf.lite.Interpreter('sine_model_quantized.tflite')

# Allocate memory for each model
sine_model.allocate_tensors()
sine_model_quantized.allocate_tensors()

# Get indexes of the input and output tensors
sine_model_input_index = sine_model.get_input_details()[0]["index"]
sine_model_output_index = sine_model.get_output_details()[0]["index"]
sine_model_quantized_input_index = sine_model_quantized.get_input_details()[0]["index"]
sine_model_quantized_output_index = \
  sine_model_quantized.get_output_details()[0]["index"]

# Create arrays to store the results
sine_model_predictions = []
sine_model_quantized_predictions = []

# Run each model's interpreter for each value and store the results in arrays
for x_value in x_test:
  # Create a 2D tensor wrapping the current x value
  x_value_tensor = tf.convert_to_tensor([[x_value]], dtype=np.float32)
  # Write the value to the input tensor
  sine_model.set_tensor(sine_model_input_index, x_value_tensor)
  # Run inference
  sine_model.invoke()
  # Read the prediction from the output tensor
  sine_model_predictions.append(
      sine_model.get_tensor(sine_model_output_index)[0])
  # Do the same for the quantized model
  sine_model_quantized.set_tensor\
  (sine_model_quantized_input_index, x_value_tensor)
  sine_model_quantized.invoke()
  sine_model_quantized_predictions.append(
      sine_model_quantized.get_tensor(sine_model_quantized_output_index)[0])

# See how they line up with the data
plt.clf()
plt.title('Comparison of various models against actual values')
plt.plot(x_test, y_test, 'bo', label='Actual')
plt.plot(x_test, predictions, 'ro', label='Original predictions')
plt.plot(x_test, sine_model_predictions, 'bx', label='Lite predictions')
plt.plot(x_test, sine_model_quantized_predictions, 'gx', \
  label='Lite quantized predictions')
plt.legend()
plt.show()
```

运行此单元格将产生图 4-21 中的图表。

![比较模型预测与实际值的图表](img/timl_0421.png)

###### 图 4-21。比较模型预测与实际值的图表

从图表中我们可以看到，原始模型、转换模型和量化模型的预测都非常接近，几乎无法区分。情况看起来很不错！

由于量化使模型变小，让我们比较两个转换后的模型，看看大小上的差异。运行以下单元格计算它们的大小并进行比较：

```py
import os
basic_model_size = os.path.getsize("sine_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size = os.path.getsize("sine_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
difference = basic_model_size - quantized_model_size
print("Difference is %d bytes" % difference)
```

您应该看到以下输出：

```py
Basic model is 2736 bytes
Quantized model is 2512 bytes
Difference is 224 bytes
```

我们的量化模型比原始版本小 224 字节，这很好，但大小只有轻微减小。在约 2.4 KB 左右，这个模型已经非常小，权重和偏差只占整体大小的一小部分。除了权重，模型还包含构成我们深度学习网络架构的所有逻辑，称为*计算图*。对于真正微小的模型，这可能比模型的权重占用更多的空间，这意味着量化几乎没有效果。

更复杂的模型有更多的权重，这意味着量化带来的空间节省将更高。对于大多数复杂模型，可以预期接近四倍。

无论其确切大小如何，我们的量化模型执行起来都比原始版本快，这对于微小微控制器非常重要。

## 转换为 C 文件

为了让我们的模型能够与 TensorFlow Lite for Microcontrollers 一起使用的最后一步是将其转换为一个可以包含在我们应用程序中的 C 源文件。

在本章中，我们一直在使用 TensorFlow Lite 的 Python API。这意味着我们可以使用`Interpreter`构造函数从磁盘加载我们的模型文件。

然而，大多数微控制器没有文件系统，即使有，从磁盘加载模型所需的额外代码也会在有限的空间下是浪费的。相反，作为一个优雅的解决方案，我们提供了一个可以包含在我们的二进制文件中并直接加载到内存中的 C 源文件中的模型。

在文件中，模型被定义为一个字节数组。幸运的是，有一个方便的 Unix 工具名为`xxd`，能够将给定文件转换为所需的格式。

以下单元格在我们的量化模型上运行`xxd`，将输出写入名为*sine_model_quantized.cc*的文件，并将其打印到屏幕上：

```py
# Install xxd if it is not available
!apt-get -qq install xxd
# Save the file as a C source file
!xxd -i sine_model_quantized.tflite > sine_model_quantized.cc
# Print the source file
!cat sine_model_quantized.cc
```

输出非常长，所以我们不会在这里全部复制，但这里有一个片段，包括开头和结尾：

```py
unsigned char sine_model_quantized_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x12, 0x00,
  0x1c, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00,
  // ...
  0x00, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
  0x04, 0x00, 0x00, 0x00
};
unsigned int sine_model_quantized_tflite_len = 2512;
```

要在项目中使用这个模型，您可以复制粘贴源代码，或者从笔记本中下载文件。

# 总结

有了这个，我们构建我们的模型就完成了。我们已经训练、评估并转换了一个 TensorFlow 深度学习网络，可以接收 0 到 2π之间的数字，并输出其正弦的良好近似值。

这是我们第一次使用 Keras 训练微小模型。在未来的项目中，我们将训练仍然微小但*远远*更复杂的模型。

现在，让我们继续第五章，在那里我们将编写代码在微控制器上运行我们的模型。
