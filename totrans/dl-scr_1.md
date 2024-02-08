# 第1章。基础

> 不要死记这些公式。如果你理解了概念，你可以发明自己的符号。
> 
> 约翰·科克兰，《投资笔记》2006

本章的目的是解释一些对理解神经网络工作至关重要的基础心智模型。具体来说，我们将涵盖*嵌套数学函数及其导数*。我们将从最简单的基本构建块开始，逐步展示我们可以构建由“链”组成的复杂函数，即使其中一个函数是接受多个输入的矩阵乘法，也可以计算函数输出相对于输入的导数。理解这个过程如何运作将是理解神经网络的关键，而我们实际上直到[第2章](ch02.html#fundamentals)才会开始涵盖神经网络。

当我们围绕神经网络的这些基础构建块来找到方向时，我们将系统地从三个视角描述我们引入的每个概念：

+   数学，以方程或方程组的形式

+   代码，尽可能少的额外语法（使Python成为理想选择）

+   一个解释正在发生的事情的图表，就像你在编程面试中在白板上画的那种

正如前言中提到的，理解神经网络的一个挑战是需要多种心智模型。在本章中，我们将感受到这一点：这三种视角中的每一种都排除了我们将要涵盖的概念的某些基本特征，只有当它们一起被考虑时，才能提供关于嵌套数学函数工作方式的完整图景。事实上，我坚定地认为，任何试图解释神经网络构建块的尝试，如果排除了这三种视角中的任何一种，都是不完整的。

现在，我们要迈出第一步了。我们将从一些极其简单的基本构建块开始，以说明我们如何可以从这三个视角理解不同的概念。我们的第一个基本构建块将是一个简单但至关重要的概念：函数。

# 函数

什么是函数，我们如何描述它？与神经网络一样，有几种方法来描述函数，其中没有一种能够完整地描绘出整个图景。与其试图给出一个简洁的一句话描述，不如我们简单地逐个走过这三种心智模型，扮演感受大象不同部分的盲人的角色。

## 数学

这里有两个函数的例子，用数学符号描述：

+   *f*[1](*x*) = *x*²

+   *f*[2](*x*) = *max*(*x*, 0)

这个符号表示函数，我们任意地称为*f*[1]和*f*[2]，将一个数字*x*作为输入，并将其转换为*x*²（在第一种情况下）或*max*(*x*, 0)（在第二种情况下）。

## 图表

描述函数的一种方式是：

1.  画一个*x-y*平面（其中*x*指水平轴，*y*指垂直轴）。

1.  绘制一堆点，其中点的x坐标是函数在某个范围内的（通常是均匀间隔的）输入，y坐标是该范围内函数的输出。

1.  连接这些绘制的点。

这最初是由法国哲学家勒内·笛卡尔完成的，它在许多数学领域中非常有用，特别是微积分。[图1-1](#fig_01-01)显示了这两个函数的图表。

![两个连续的、大部分可微的函数](assets/dlfs_0101.png)

###### 图1-1。两个连续的、大部分可微的函数

然而，还有另一种描述函数的方式，在学习微积分时并不那么有用，但在思考深度学习模型时将非常有用。我们可以将函数看作是接受数字输入并产生数字输出的盒子，就像具有其自身内部规则的小工厂，用于处理输入。[图1-2](#fig_01-02)展示了这两个函数被描述为通用规则以及它们如何在特定输入上运行。

![另一种看待函数的方式](assets/dlfs_0102.png)

###### 图1-2. 另一种看待这些函数的方式

## 代码

最后，我们可以使用代码描述这些函数。在此之前，我们应该简要介绍一下我们将在其上编写函数的Python库：NumPy。

### 代码注意事项＃1：NumPy

NumPy是一个广泛使用的Python库，用于快速数值计算，其内部大部分是用C编写的。简而言之：我们在神经网络中处理的数据将始终保存在一个几乎总是一维、二维、三维或四维的*多维数组*中，尤其是二维或三维。来自NumPy库的`ndarray`类允许我们以既直观又快速的方式操作这些数组。举个最简单的例子：如果我们将数据存储在Python列表（或列表的列表）中，使用正常语法逐元素添加或乘以列表是行不通的，而对于`ndarray`却是行得通的：

```py
print("Python list operations:")
a = [1,2,3]
b = [4,5,6]
print("a+b:", a+b)
try:
    print(a*b)
except TypeError:
    print("a*b has no meaning for Python lists")
print()
print("numpy array operations:")
a = np.array([1,2,3])
b = np.array([4,5,6])
print("a+b:", a+b)
print("a*b:", a*b)
```

```py
Python list operations:
a+b: [1, 2, 3, 4, 5, 6]
a*b has no meaning for Python lists

numpy array operations:
a+b: [5 7 9]
a*b: [ 4 10 18]
```

`ndarray`还具有您从`n`维数组中期望的几个特性；每个`ndarray`都有`n`个轴，从0开始索引，因此第一个轴是`0`，第二个是`1`，依此类推。特别是，由于我们经常处理2D `ndarray`，我们可以将`axis = 0`看作行，`axis = 1`看作列——参见[图1-3](#fig_01-03)。

![简单的NumPy数组示例](assets/dlfs_0103.png)

###### 图1-3. 一个2D NumPy数组，其中axis = 0表示行，axis = 1表示列

NumPy的`ndarray`还支持沿着这些轴以直观方式应用函数。例如，沿着轴0（2D数组的*行*）求和基本上会沿着该轴“折叠数组”，返回一个比原始数组少一个维度的数组；对于2D数组，这相当于对每列求和：

```py
print('a:')
print(a)
print('a.sum(axis=0):', a.sum(axis=0))
print('a.sum(axis=1):', a.sum(axis=1))
```

```py
a:
[[1 2]
 [3 4]]
a.sum(axis=0): [4 6]
a.sum(axis=1): [3 7]
```

最后，NumPy的`ndarray`支持将1D数组添加到最后一个轴；对于具有`R`行和`C`列的2D数组`a`，这意味着我们可以添加长度为`C`的1D数组`b`，NumPy将以直观的方式进行加法运算，将元素添加到`a`的每一行：^([1](ch01.html#idm45732632700344))

```py
a = np.array([[1,2,3],
              [4,5,6]])

b = np.array([10,20,30])

print("a+b:\n", a+b)
```

```py
a+b:
[[11 22 33]
 [14 25 36]]
```

### 代码注意事项＃2：类型检查的函数

正如我提到的，我们在本书中编写的代码的主要目标是使我解释的概念变得精确和清晰。随着书的进行，这将变得更具挑战性，因为我们将编写具有许多参数的函数作为复杂类的一部分。为了应对这一挑战，我们将在整个过程中使用带有类型签名的函数；例如，在[第3章](ch03.html#deep_learning_from_scratch)中，我们将初始化我们的神经网络如下：

```py
def __init__(self,
             layers: List[Layer],
             loss: Loss,
             learning_rate: float = 0.01) -> None:
```

仅凭这个类型签名，您就可以对该类的用途有一些了解。相比之下，考虑以下类型签名，我们*可以*用来定义一个操作：

```py
def operation(x1, x2):
```

仅凭这个类型签名，您无法了解正在发生什么；只有通过打印出每个对象的类型，查看在每个对象上执行的操作，或根据名称`x1`和`x2`猜测，我们才能理解这个函数中正在发生的事情。相反，我可以定义一个带有以下类型签名的函数：

```py
def operation(x1: ndarray, x2: ndarray) -> ndarray:
```

您立即知道这是一个接受两个`ndarray`的函数，可能以某种方式将它们组合在一起，并输出该组合的结果。由于它们提供的更清晰性，我们将在本书中始终使用带有类型检查的函数。

### NumPy中的基本函数

在了解了这些基础知识之后，让我们在NumPy中编写我们之前定义的函数：

```py
def square(x: ndarray) -> ndarray:
    '''
 Square each element in the input ndarray.
 '''
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    '''
 Apply "Leaky ReLU" function to each element in ndarray.
 '''
    return np.maximum(0.2 * x, x)
```

###### 注意

NumPy的一个特点是许多函数可以通过写`np.*function_name*(ndarray)`或写`ndarray.*function_name*`来应用于`ndarray`。例如，前面的`relu`函数可以写成：`x.clip(min=0)`。我们将尽量保持一致，使用`np.*function_name*(ndarray)`的约定——特别是，我们将避免诸如`*ndarray*.T`用于转置二维`ndarray`的技巧，而是写成`np.transpose(*ndarray*, (1, 0))`。

如果你能理解数学、图表和代码是表示同一基本概念的三种不同方式，那么你就已经在正确理解深度学习所需的灵活思维方面迈出了重要一步。

# 导数

导数，就像函数一样，是理解深度学习的一个非常重要的概念，你们中的许多人可能已经熟悉了。和函数一样，它们可以用多种方式表示。我们首先简单地说一下，函数在某一点的导数是函数输出相对于该点的输入的“变化率”。现在让我们通过相同的三个导数视角来更好地理解导数的工作原理。

## 数学

首先，我们将数学上精确描述：我们可以将这个数字描述为一个极限，即在特定值*a*的输入处改变其输入时，*f*的输出会发生多少变化：

<math display="block"><mrow><mfrac><mrow><mi>d</mi><mi>f</mi></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>=</mo> <munder><mo form="prefix" movablelimits="true">lim</mo> <mrow><mi>Δ</mi><mo>→</mo><mn>0</mn></mrow></munder> <mfrac><mrow><mi>f</mi><mfenced close=")" open="(" separators=""><mrow><mi>a</mi><mo>+</mo><mi>Δ</mi></mrow></mfenced><mo>-</mo><mi>f</mi><mfenced close=")" open="(" separators=""><mi>a</mi><mo>-</mo><mi>Δ</mi></mfenced></mrow> <mrow><mn>2</mn><mo>×</mo><mi>Δ</mi></mrow></mfrac></mrow></math>

这个极限可以通过设置一个非常小的*Δ*值（例如0.001）来进行数值近似，因此我们可以计算导数为：

<math display="block"><mrow><mfrac><mrow><mi>d</mi><mi>f</mi></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>f</mi><mo>(</mo><mi>a</mi><mo>+</mo><mn>0.001</mn><mo>)</mo><mo>-</mo><mi>f</mi><mo>(</mo><mi>a</mi><mo>-</mo><mn>0.001</mn><mo>)</mo></mrow> <mrow><mn>0.002</mn></mrow></mfrac></mrow></math>

虽然准确，但这只是导数完整心智模型的一部分。让我们从另一个角度来看待它们：一个图表。

## 图表

首先，熟悉的方式：如果我们简单地在函数*f*的笛卡尔表示上画一条切线，*f*在点*a*的导数就是这条线在*a*点的斜率。与前一小节中的数学描述一样，我们实际上可以计算这条线的斜率的两种方法。第一种是使用微积分来计算极限。第二种是只需取连接*f*在*a* - 0.001和*a* + 0.001的线的斜率。后一种方法在[图1-4](#fig_01-04)中有所描述，对于学过微积分的人应该很熟悉。

![dlfs 0104](assets/dlfs_0104.png)

###### 图1-4\. 导数作为斜率

正如我们在前一节中看到的，另一种思考函数的方式是将其视为小工厂。现在想象一下，这些工厂的输入通过一根绳子连接到输出。导数等于这个问题的答案：如果我们向上拉动函数的输入*a*一点点，或者为了考虑到函数在*a*可能是不对称的情况，向下拉动*a*一点点，根据工厂的内部运作，输出将以这个小量的多少倍改变？这在[图1-5](#fig_01-05)中有所描述。

![dlfs 0105](assets/dlfs_0105.png)

###### 图1-5\. 另一种可视化导数的方式

这第二种表示将比第一种更重要，以理解深度学习。

## 代码

最后，我们可以编写我们之前看到的导数近似的代码：

```py
from typing import Callable

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    '''
 Evaluates the derivative of a function "func" at every element in the
 "input_" array.
 '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)
```

###### 注意

当我们说“某物是另一物的函数”时——例如，*P*是*E*的函数（故意随机选择的字母），我们的意思是存在某个函数*f*，使得*f*(*E*) = *P*——或者等价地，存在一个函数*f*，接受*E*对象并产生*P*对象。我们也可以将其理解为*P*“被定义为”将函数*f*应用于*E*时产生的结果：

![另一种可视化函数的方式](assets/dlfs_01in01.png)

我们可以将其编码为：

```py
def f(input_: ndarray) -> ndarray:
    # Some transformation(s)
    return output

P = f(E)
```

# 嵌套函数

现在我们将介绍一个对理解神经网络至关重要的概念：函数可以“嵌套”形成“复合”函数。我所说的“嵌套”到底是什么意思呢？我指的是如果我们有两个函数，按照数学约定我们称为 *f*[1] 和 *f*[2]，其中一个函数的输出成为下一个函数的输入，这样我们就可以“串联”它们。

## 图表

表示嵌套函数最自然的方式是使用“迷你工厂”或“盒子”表示法（来自 [“函数”](#functions-section-01) 的第二种表示法）。

如 [图1-6](#fig_01-07) 所示，一个输入进入第一个函数，被转换，然后出来；然后它进入第二个函数，再次被转换，我们得到最终输出。

![f1 and f2 as a chain](assets/dlfs_0106.png)

###### 图1-6\. 嵌套函数，自然地

## 数学

我们还应该包括不太直观的数学表示：

<math><mrow><msub><mi>f</mi> <mn>2</mn></msub> <mrow><mo>(</mo> <msub><mi>f</mi> <mn>1</mn></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mi>y</mi></mrow></math>

这是不太直观的，因为嵌套函数的怪癖是从“外到内”阅读，但实际上操作是“从内到外”执行的。例如，尽管 <math><mrow><msub><mi>f</mi> <mn>2</mn></msub> <mrow><mo>(</mo> <msub><mi>f</mi> <mn>1</mn></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mi>y</mi></mrow></math> 读作“f 2 of f 1 of x”，但它实际上意味着“首先将 *f*[1] 应用于 *x*，然后将 *f*[2] 应用于将 *f*[1] 应用于 *x* 的结果”。

## 代码

最后，为了遵守我承诺的从三个角度解释每个概念，我们将对此进行编码。首先，我们将为嵌套函数定义一个数据类型：

```py
from typing import List

# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]
```

然后我们将定义数据如何通过长度为2的链传递：

```py
def chain_length_2(chain: Chain,
                   a: ndarray) -> ndarray:
    '''
 Evaluates two functions in a row, in a "Chain".
 '''
    assert len(chain) == 2, \
    "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))
```

## 另一个图表

使用盒子表示法描绘嵌套函数，我们可以看到这个复合函数实际上只是一个单一函数。因此，我们可以简单地表示这个函数为 *f*[1] *f*[2]，如 [图1-7](#fig_01-08) 所示。

![f1f2 nested](assets/dlfs_0107.png)

###### 图1-7\. 另一种思考嵌套函数的方式

此外，微积分中的一个定理告诉我们，由“大部分可微”的函数组成的复合函数本身也是大部分可微的！因此，我们可以将 *f*[1]*f*[2] 视为另一个我们可以计算导数的函数，计算复合函数的导数将对训练深度学习模型至关重要。

然而，我们需要一个公式来计算这个复合函数的导数，以其组成函数的导数表示。这将是我们接下来要讨论的内容。

# 链式法则

链式法则是一个数学定理，让我们能够计算复合函数的导数。深度学习模型在数学上是复合函数，推理它们的导数对于训练它们是至关重要的，我们将在接下来的几章中看到。

## 数学

从数学上讲，定理陈述了一个相当不直观的形式，即对于给定的值 `x`，

<math display="block"><mrow><mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>2</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>2</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>f</mi> <mn>1</mn></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>1</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>

其中 *u* 只是一个代表函数输入的虚拟变量。

###### 注意

当描述具有一个输入和输出的函数 *f* 的导数时，我们可以将表示该函数导数的 *函数* 表示为 <math><mfrac><mrow><mi>d</mi><mi>f</mi></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac></math>。我们可以用另一个虚拟变量代替 *u* —— 这无关紧要，就像 *f*(*x*) = *x*² 和 *f*(*y*) = *y*² 意思相同。

另一方面，稍后我们将处理接受*多个*输入的函数，比如，*x*和*y*。一旦到达那里，编写<math><mfrac><mrow><mi>d</mi><mi>f</mi></mrow> <mrow><mi>d</mi><mi>x</mi></mrow></mfrac></math>并且让它意味着与<math><mfrac><mrow><mi>d</mi><mi>f</mi></mrow> <mrow><mi>d</mi><mi>y</mi></mrow></mfrac></math>不同的东西就会有意义。

这就是为什么在前面的公式中，我们用底部的*u*表示*所有*的导数：*f*[1]和*f*[2]都是接受一个输入并产生一个输出的函数，在这种情况下（具有一个输入和一个输出的函数），我们将在导数符号中使用*u*。

### 图表

前面的公式并没有给出链式法则的太多直觉。对于这一点，框表示法更有帮助。让我们推理一下在简单情况下*f*[1] *f*[2]的导数“应该”是什么。

![f1f2 nested](assets/dlfs_0108.png)

###### 图1-8。链式法则的示例

直觉上，使用[图1-8](#fig_01-09)中的图表，复合函数的导数*应该*是其组成函数的导数的一种乘积。假设我们将值5输入到第一个函数中，再假设在*u*=5处计算第一个函数的*导数*得到一个值为3，即，<math><mrow><mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>1</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mn>5</mn> <mo>)</mo></mrow> <mo>=</mo> <mn>3</mn></mrow></math>。

假设我们然后取出第一个框中的函数的*值*，假设它是1，所以*f*[1](5) = 1，并计算在这个值处第二个函数*f*[2]的导数：即，<math><mrow><mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>2</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mn>1</mn> <mo>)</mo></mrow></mrow></math>。我们发现这个值是-2。

如果我们将这些函数想象成字面上串在一起，那么如果将第二个框的输入改变1个单位会导致第二个框的输出变化-2个单位，那么将第二个框的输入改变3个单位应该会导致第二个框的输出变化-2×3 = -6个单位。这就是为什么在链式法则的公式中，最终结果最终是一个乘积：<math><mrow><mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>2</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>f</mi> <mn>1</mn></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math> *乘以* <math><mrow><mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>1</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>。

因此，通过考虑图表和数学，我们可以通过链式法则推理出嵌套函数输出的导数与其输入应该是什么，代码指令可能是什么样的？

## 代码

让我们编写代码并展示以这种方式计算导数实际上会产生“看起来正确”的结果。我们将使用来自[“NumPy中的基本函数”](#basic-NumPy)的`square`函数，以及`sigmoid`，另一个在深度学习中变得重要的函数：

```py
def sigmoid(x: ndarray) -> ndarray:
    '''
 Apply the sigmoid function to each element in the input ndarray.
 '''
    return 1 / (1 + np.exp(-x))
```

现在我们编写链式法则的代码：

```py
def chain_deriv_2(chain: Chain,
                  input_range: ndarray) -> ndarray:
    '''
 Uses the chain rule to compute the derivative of two nested functions:
 (f2(f1(x))' = f2'(f1(x)) * f1'(x)
 '''

    assert len(chain) == 2, \
    "This function requires 'Chain' objects of length 2"

    assert input_range.ndim == 1, \
    "Function requires a 1 dimensional ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]

    # df1/dx
    f1_of_x = f1(input_range)

    # df1/du
    df1dx = deriv(f1, input_range)

    # df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))

    # Multiplying these quantities together at each point
    return df1dx * df2du
```

[图1-9](#fig_01-10)绘制了结果，并显示链式法则有效：

```py
PLOT_RANGE = np.arange(-3, 3, 0.01)

chain_1 = [square, sigmoid]
chain_2 = [sigmoid, square]

plot_chain(chain_1, PLOT_RANGE)
plot_chain_deriv(chain_1, PLOT_RANGE)

plot_chain(chain_2, PLOT_RANGE)
plot_chain_deriv(chain_2, PLOT_RANGE)
```

![链式法则示例](assets/dlfs_0109.png)

###### 图1-9。链式法则有效，第1部分

链式法则似乎有效。当函数是向上倾斜时，导数是正的；当函数是平的时，导数是零；当函数是向下倾斜时，导数是负的。

因此，我们实际上可以计算嵌套或“复合”函数的导数，如*f*[1] *f*[2]，只要这些单独的函数本身大部分是可微的。

事实证明，深度学习模型在数学上是这些大部分可微函数的长链；花时间详细地手动通过一个稍微更长的例子将有助于建立您对正在发生的事情以及如何将其推广到更复杂模型的直觉。

# 稍微长一点的例子

让我们仔细研究一个稍微更长的链条：如果我们有三个大部分可微的函数—*f*[1]、*f*[2]和*f*[3]—我们将如何计算*f*[1] *f*[2] *f*[3]的导数？我们“应该”能够做到，因为根据之前提到的微积分定理，我们知道“大部分可微”函数的复合是可微的。

## 数学

数学上，结果是以下表达式：

<math display="block"><mrow><mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>3</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>3</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>f</mi> <mn>2</mn></msub> <mrow><mo>(</mo> <msub><mi>f</mi> <mn>1</mn></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>2</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>f</mi> <mn>1</mn></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>1</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>

为什么这个公式适用于长度为2的链条的基本逻辑，<math><mrow><mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>2</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>2</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>f</mi> <mn>1</mn></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>1</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>，在这里也适用——看公式本身缺乏直觉！

## 图表

要（字面上）看到这个公式为什么是有意义的，最好的方法是通过另一个盒子图表，如[图1-10](#fig_01-11)所示。

![dlfs 0110](assets/dlfs_0110.png)

###### 图1-10。计算三个嵌套函数导数的“盒子模型”

使用类似的推理来自前一节：如果我们想象*f*[1] *f*[2] *f*[3]的输入（称为*a*）通过一根绳子连接到输出（称为*b*），那么将*a*改变一个小量*Δ*将导致*f*[1](*a*)的变化为<math><mrow><mi>d</mi><msub><mi>f</mi> <mn>1</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mrow></math>乘以*Δ*，这将导致<math><mrow><msub><mi>f</mi> <mn>2</mn></msub> <mrow><mo>(</mo> <msub><mi>f</mi> <mn>1</mn></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>（链中的下一步）的变化为<math><mrow><mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>2</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>f</mi> <mn>1</mn></msub> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>d</mi><msub><mi>f</mi> <mn>1</mn></msub></mrow> <mrow><mi>d</mi><mi>u</mi></mrow></mfrac> <mrow><mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>乘以*Δ*，以此类推到第三步，当我们到达最终变化时，等于前述链式法则的完整公式乘以*Δ*。花一点时间阅读这个解释和之前的图表，但不要花太多时间，因为当我们编写代码时，我们将对此有更多的直觉。

## 代码

我们如何将这样的公式转化为代码指令，以计算导数，考虑到组成函数？有趣的是，在这个简单的例子中，我们已经看到了神经网络前向和后向传递的开端：

```py
def chain_deriv_3(chain: Chain,
                  input_range: ndarray) -> ndarray:
    '''
 Uses the chain rule to compute the derivative of three nested functions:
 (f3(f2(f1)))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
 '''

    assert len(chain) == 3, \
    "This function requires 'Chain' objects to have length 3"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # f1(x)
    f1_of_x = f1(input_range)

    # f2(f1(x))
    f2_of_x = f2(f1_of_x)

    # df3du
    df3du = deriv(f3, f2_of_x)

    # df2du
    df2du = deriv(f2, f1_of_x)

    # df1dx
    df1dx = deriv(f1, input_range)

    # Multiplying these quantities together at each point
    return df1dx * df2du * df3du
```

这里发生了一些有趣的事情——为了计算这个嵌套函数的链式法则，我们进行了两次“遍历”：

1.  首先，我们“向前走”通过它，沿途计算量`f1_of_x`和`f2_of_x`。我们可以称之为（并将其视为）“前向传递”。

1.  然后，我们“向后走”，使用我们在前向传递中计算的量来计算组成导数的量。

最后，我们将这三个量相乘以得到我们的导数。

现在，让我们展示这是如何工作的，使用我们迄今为止定义的三个简单函数：`sigmoid`、`square` 和 `leaky_relu`。

```py
PLOT_RANGE = np.range(-3, 3, 0.01)
plot_chain([leaky_relu, sigmoid, square], PLOT_RANGE)
plot_chain_deriv([leaky_relu, sigmoid, square], PLOT_RANGE)
```

[图 1-11](#fig_01-12) 显示了结果。

![dlfs 0111](assets/dlfs_0111.png)

###### 图 1-11\. 链式法则有效，即使是三重嵌套函数

再次比较导数的图与原始函数的斜率，我们看到链式法则确实正确计算了导数。

现在让我们将我们的理解应用于具有多个输入的复合函数，这是一类遵循我们已经建立的相同原则并且最终更适用于深度学习的函数。

# 具有多个输入的函数

到目前为止，我们对如何将函数串联起来形成复合函数有了概念上的理解。我们也知道如何将这些函数表示为一系列输入和输出的方框。最后，我们已经了解了如何计算这些函数的导数，以便我们既从数学上又从“前向”和“后向”组件计算的过程中理解这些导数的数量。

在深度学习中，我们处理的函数通常不只有一个输入。相反，它们有几个输入，在某些步骤中被相加、相乘或以其他方式组合。正如我们将看到的，计算这些函数的输出对其输入的导数仍然不是问题：让我们考虑一个非常简单的具有多个输入的场景，其中两个输入被相加，然后通过另一个函数进行馈送。

## 数学

对于这个例子，实际上从数学上看是有用的。如果我们的输入是 *x* 和 *y*，那么我们可以将函数看作是分两步进行的。在第一步中，*x* 和 *y* 被馈送到一个将它们相加的函数中。我们将这个函数表示为 *α*（我们将使用希腊字母来引用函数名称），函数的输出为 *a*。形式上，这简单地表示为：

<math display="block"><mrow><mi>a</mi> <mo>=</mo> <mi>α</mi> <mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo> <mo>=</mo> <mi>x</mi> <mo>+</mo> <mi>y</mi></mrow></math>

第二步是将 *a* 馈送到某个函数 *σ* 中（*σ* 可以是任何连续函数，如 `sigmoid`，或 `square` 函数，甚至一个名称不以 *s* 开头的函数）。我们将这个函数的输出表示为 *s*：

<math display="block"><mrow><mi>s</mi> <mo>=</mo> <mi>σ</mi> <mo>(</mo> <mi>a</mi> <mo>)</mo></mrow></math>

我们可以等价地将整个函数表示为 *f* 并写成：

<math display="block"><mrow><mi>f</mi> <mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo> <mo>=</mo> <mi>σ</mi> <mo>(</mo> <mi>x</mi> <mo>+</mo> <mi>y</mi> <mo>)</mo></mrow></math>

这更加数学上简洁，但它掩盖了这实际上是两个操作按顺序发生的事实。为了说明这一点，我们需要下一节中的图表。

## 图表

现在我们正在检查具有多个输入的函数，让我们暂停一下来定义一个我们一直在围绕的概念：用圆圈和连接它们的箭头表示数学“运算顺序”的图表可以被视为 *计算图*。例如，[图 1-12](#fig_01-13) 显示了我们刚刚描述的函数 *f* 的计算图。

![dlfs 0112](assets/dlfs_0112.png)

###### 图 1-12\. 具有多个输入的函数

这里我们看到两个输入进入 *α*，作为 *a* 出来，然后通过 *σ* 进行馈送。

## 代码

编写这个代码非常简单；但是请注意，我们必须添加一个额外的断言：

```py
def multiple_inputs_add(x: ndarray,
                        y: ndarray,
                        sigma: Array_Function) -> float:
    '''
 Function with multiple inputs and addition, forward pass.
 '''
    assert x.shape == y.shape

    a = x + y
    return sigma(a)
```

与本章前面看到的函数不同，这个函数不仅仅是在其输入 `ndarray` 的每个元素上“逐元素”操作。每当我们处理一个需要多个 `ndarray` 作为输入的操作时，我们必须检查它们的形状，以确保它们满足该操作所需的任何条件。在这里，对于一个简单的加法操作，我们只需要检查形状是否相同，以便可以逐元素进行加法。

# 具有多个输入的函数的导数

我们不应感到惊讶，我们可以计算这样一个函数的输出对其两个输入的导数。

## 图表

从概念上讲，我们只需做与具有一个输入的函数相同的事情：通过计算图“向后”计算每个组成函数的导数，然后将结果相乘以获得总导数。如[图1-13](#fig_01-14)所示。

![dlfs 0113](assets/dlfs_0113.png)

###### 图1-13。通过具有多个输入的函数的计算图向后传递

## 数学

链式法则适用于这些函数，就像适用于前几节中的函数一样。由于这是一个嵌套函数，<math><mrow><mi>f</mi> <mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo> <mo>=</mo> <mi>σ</mi> <mo>(</mo> <mi>α</mi> <mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo> <mo>)</mo></mrow></math>，我们有：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>f</mi></mrow> <mrow><mi>∂</mi><mi>x</mi></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>α</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>∂</mi><mi>α</mi></mrow> <mrow><mi>∂</mi><mi>x</mi></mrow></mfrac> <mrow><mo>(</mo> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>x</mi> <mo>+</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>∂</mi><mi>α</mi></mrow> <mrow><mi>∂</mi><mi>x</mi></mrow></mfrac> <mrow><mo>(</mo> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>

当然，<math><mfrac><mrow><mi>∂</mi><mi>f</mi></mrow> <mrow><mi>∂</mi><mi>y</mi></mrow></mfrac></math>也将是相同的。

现在注意：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>α</mi></mrow> <mrow><mi>∂</mi><mi>x</mi></mrow></mfrac> <mrow><mo>(</mo> <mrow><mo>(</mo> <mi>x</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mn>1</mn></mrow></math>

因为对于*x*的每个单位增加，*a*都会增加一个单位，无论*x*的值如何（*y*的情况也是如此）。

有了这个，我们可以编写如何计算这种函数的导数。

## 代码

```py
def multiple_inputs_add_backward(x: ndarray,
                                 y: ndarray,
                                 sigma: Array_Function) -> float:
    '''
 Computes the derivative of this simple function with respect to
 both inputs.
 '''
    # Compute "forward pass"
    a = x + y

    # Compute derivatives
    dsda = deriv(sigma, a)

    dadx, dady = 1, 1

    return dsda * dadx, dsda * dady
```

读者的一个简单练习是修改这个例子，使得`x`和`y`相乘而不是相加。

接下来，我们将研究一个更复杂的例子，更接近深度学习中发生的情况：与前一个例子类似的函数，但有两个*向量*输入。

# 具有多个向量输入的函数

在深度学习中，我们处理的函数的输入是*向量*或*矩阵*。这些对象不仅可以相加、相乘等，还可以通过点积或矩阵乘法进行组合。在本章的其余部分，我将展示如何使用正向和反向传递计算这些函数的导数的数学和逻辑仍然适用。

这些技术最终将成为理解为什么深度学习有效的核心。在深度学习中，我们的目标是将模型拟合到一些数据中。更准确地说，这意味着我们希望找到一个数学函数，将数据中的*观察*（将是函数的输入）映射到数据中的一些期望的*预测*（将是函数的输出），并以尽可能优化的方式。原来这些观察结果将被编码在矩阵中，通常每行作为一个观察，每列作为该观察的一个数值特征。我们将在下一章中更详细地介绍这一点；目前，能够推理复杂函数的导数，包括点积和矩阵乘法，将是至关重要的。

让我们首先准确地定义我所说的意思。

## 数学

在神经网络中表示单个数据点或“观察”的典型方式是将其表示为具有*n*个特征的行，其中每个特征只是一个数字*x*[1]，*x*[2]，等等，直到*x*[*n*]：

<math display="block"><mrow><mi>X</mi> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><msub><mi>x</mi> <mn>1</mn></msub></mtd> <mtd><msub><mi>x</mi> <mn>2</mn></msub></mtd> <mtd><mo>...</mo></mtd> <mtd><msub><mi>x</mi> <mi>n</mi></msub></mtd></mtr></mtable></mfenced></mrow></math>

在这里要牢记的一个典型例子是预测房价，我们将在下一章中从头开始构建一个神经网络来实现这一点；在这个例子中，*x*[1]，*x*[2]等等是房屋的数值特征，比如房屋的面积或其与学校的距离。

# 从现有特征创建新特征

神经网络中可能最常见的操作之一是形成这些特征的“加权和”，其中加权和可以强调某些特征并减弱其他特征，因此可以被视为一个新特征，它本身只是旧特征的组合。数学上简洁表达这一点的方法是将这个观察结果与与特征相同长度的一组“权重”进行*点积*，*w*[1]，*w*[2]，等等，直到*w*[*n*]。让我们从我们在本章迄今使用的三个角度来探讨这个概念。

## 数学

要在数学上准确，如果：

<math display="block"><mrow><mi>W</mi> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><msub><mi>w</mi> <mn>1</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>w</mi> <mn>2</mn></msub></mtd></mtr> <mtr><mtd><mo>⋮</mo></mtd></mtr> <mtr><mtd><msub><mi>w</mi> <mi>n</mi></msub></mtd></mtr></mtable></mfenced></mrow></math>

然后我们可以定义这个操作的输出为：

<math display="block"><mrow><mi>N</mi> <mo>=</mo> <mi>ν</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>X</mi> <mo>×</mo> <mi>W</mi> <mo>=</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <mo>...</mo> <mo>+</mo> <msub><mi>x</mi> <mi>n</mi></msub> <mo>×</mo> <msub><mi>w</mi> <mi>n</mi></msub></mrow></math>

请注意，这个操作是*矩阵乘法*的特例，只是碰巧是点积，因为*X*有一行，*W*只有一列。

接下来，让我们看一下我们可以用图示来描述这个操作的几种方式。

## 图

一种简单的描述这个操作的方式如[图1-14](#fig_01-15)所示。

![dlfs 0114](assets/dlfs_0114.png)

###### 图1-14。矢量点积的图示

这个图示描述了一个接受两个输入的操作，这两个输入都可以是`ndarray`，并产生一个输出`ndarray`。

但这实际上是对许多操作进行了大量简写，这些操作发生在许多输入上。我们可以选择突出显示各个操作和输入，如图[1-15](#fig_01-16)和[1-16](#fig_01-17)所示。

![dlfs 0115](assets/dlfs_0115.png)

###### 图1-15。矩阵乘法的另一个图示

![dlfs 0116](assets/dlfs_0116.png)

###### 图1-16。矩阵乘法的第三个图示

关键点是点积（或矩阵乘法）是表示许多个体操作的简洁方式；此外，正如我们将在下一节中开始看到的，使用这个操作也使我们在反向传播中的导数计算变得极其简洁。

## 代码

最后，在代码中，这个操作只是：

```py
def matmul_forward(X: ndarray,
                   W: ndarray) -> ndarray:
    '''
 Computes the forward pass of a matrix multiplication.
 '''

    assert X.shape[1] == W.shape[0], \
    '''
 For matrix multiplication, the number of columns in the first array should
 match the number of rows in the second; instead the number of columns in the
 first array is {0} and the number of rows in the second array is {1}.
 '''.format(X.shape[1], W.shape[0])

    # matrix multiplication
    N = np.dot(X, W)

    return N
```

我们有一个新的断言，确保矩阵乘法能够进行。（这是必要的，因为这是我们的第一个不仅仅处理大小相同的`ndarray`并对元素进行操作的操作——我们的输出现在实际上与我们的输入大小不同。）

# 具有多个矢量输入的函数的导数

对于简单将一个数字作为输入并产生一个输出的函数，如*f*(*x*) = *x*²或*f*(*x*) = sigmoid(*x*)，计算导数是直接的：我们只需应用微积分规则。对于矢量函数，导数并不是立即明显的：如果我们将点积写成<math><mrow><mi>ν</mi> <mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo> <mo>=</mo> <mi>N</mi></mrow></math>，如前一节所示，自然会产生一个问题——<math><mfrac><mrow><mi>∂</mi><mi>N</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac></math>和<math><mfrac><mrow><mi>∂</mi><mi>N</mi></mrow> <mrow><mi>∂</mi><mi>W</mi></mrow></mfrac></math>会是什么？

## 图

从概念上讲，我们只是想做类似于[图1-17](#fig_01-18)的事情。

![dlfs 0117](assets/dlfs_0117.png)

###### 图1-17。矩阵乘法的反向传播，概念上

当我们只处理加法和乘法时，计算这些导数是很容易的，就像前面的例子一样。但是如何用矩阵乘法做类似的事情呢？要准确定义这一点，我们将不得不求助于数学。

## 数学

首先，我们如何定义“关于矩阵的导数”？回想一下，矩阵语法只是一堆数字以特定形式排列的简写，“关于矩阵的导数”实际上意味着“关于矩阵的每个元素的导数”。由于*X*是一行，自然的定义方式是：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><msub><mi>x</mi> <mn>1</mn></msub></mrow></mfrac></mtd> <mtd><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><msub><mi>x</mi> <mn>2</mn></msub></mrow></mfrac></mtd> <mtd><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><msub><mi>x</mi> <mn>3</mn></msub></mrow></mfrac></mtd></mtr></mtable></mfenced></mrow></math>

然而，*ν*的输出只是一个数字：<math><mrow><mi>N</mi> <mo>=</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>3</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub></mrow></math>。观察这一点，我们可以看到，例如，如果<math><msub><mi>x</mi> <mn>1</mn></msub></math>变化了*ϵ*单位，那么*N*将变化<math><mrow><msub><mi>w</mi> <mn>1</mn></msub> <mo>×</mo> <mi>ϵ</mi></mrow></math>单位——同样的逻辑也适用于其他*x*[*i*]元素。因此：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><msub><mi>x</mi> <mn>1</mn></msub></mrow></mfrac> <mo>=</mo> <msub><mi>w</mi> <mn>1</mn></msub></mrow></math><math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><msub><mi>x</mi> <mn>2</mn></msub></mrow></mfrac> <mo>=</mo> <msub><mi>w</mi> <mn>2</mn></msub></mrow></math><math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><msub><mi>x</mi> <mn>3</mn></msub></mrow></mfrac> <mo>=</mo> <msub><mi>w</mi> <mn>3</mn></msub></mrow></math>

因此：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><msub><mi>w</mi> <mn>1</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>2</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>3</mn></msub></mtd></mtr></mtable></mfenced> <mo>=</mo> <msup><mi>W</mi> <mi>T</mi></msup></mrow></math>

这是一个令人惊讶且优雅的结果，事实证明这是理解为什么深度学习有效以及如何实现得如此干净的关键部分。

通过类似的推理，我们可以看到：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><mi>W</mi></mrow></mfrac> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><msub><mi>x</mi> <mn>1</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>x</mi> <mn>2</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>x</mi> <mn>3</mn></msub></mtd></mtr></mtable></mfenced> <mo>=</mo> <msup><mi>X</mi> <mi>T</mi></msup></mrow></math>

## 代码

在这里，对答案“应该”是什么进行数学推理是困难的部分。简单的部分是编写结果的代码：

```py
def matmul_backward_first(X: ndarray,
                          W: ndarray) -> ndarray:
    '''
 Computes the backward pass of a matrix multiplication with respect to the
 first argument.
 '''

    # backward pass
    dNdX = np.transpose(W, (1, 0))

    return dNdX
```

这里计算的`dNdX`数量代表了*X*的每个元素相对于输出*N*的和的偏导数。我们在整本书中将使用一个特殊的名称来称呼这个数量：我们将其称为*X*相对于*X*的*梯度*。这个想法是对于*X*的每个单独元素——比如，*x*[3]——向量点积*N*的输出相对于*x*[3]的偏导数对应的元素在`dNdx`中（具体来说是`dNdX[2]`）。在本书中，我们将使用术语“梯度”来指代偏导数的多维模拟；具体来说，它是一个函数的输出相对于该函数输入的每个元素的偏导数数组。

# 向量函数及其导数：再进一步

当然，深度学习模型涉及多个操作：它们包括一系列操作，其中一些是像上一节中介绍的向量函数，一些只是将函数逐个元素应用于它们作为输入接收的`ndarray`。因此，我们现在将看看如何计算包含*两种*函数的复合函数的导数。假设我们的函数接受向量*X*和*W*，执行在前一节中描述的点积——我们将其表示为<math><mrow><mi>ν</mi> <mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow></math>，然后将向量通过一个函数*σ*。我们将用新的语言表达与之前相同的目标：我们想计算这个新函数的输出相对于*X*和*W*的梯度。再次强调，从下一章开始，我们将详细了解这与神经网络的关系，但现在我们只是想建立这样一个概念：我们可以计算任意复杂计算图的梯度。

## 图表

这个函数的图表，显示在[图1-18](#fig_01-19)中，与[图1-17](#fig_01-18)中的相同，只是在末尾简单地添加了*σ*函数。

![dlfs 0118](assets/dlfs_0118.png)

###### 图1-18\. 与之前相同的图表，但在末尾添加了另一个函数

## 数学

从数学上讲，这同样很简单：

<math display="block"><mrow><mi>s</mi> <mo>=</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>ν</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mi>σ</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>3</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub> <mo>)</mo></mrow></mrow></math>

## 代码

最后，我们可以将这个函数编写成：

```py
def matrix_forward_extra(X: ndarray,
                         W: ndarray,
                         sigma: Array_Function) -> ndarray:
    '''
 Computes the forward pass of a function involving matrix multiplication,
 one extra function.
 '''
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    return S
```

## 向量函数及其导数：反向传播

反向传播同样只是先前示例的直接扩展。

### 数学

由于*f*(*X, W*)是一个嵌套函数——具体来说，*f*(*X, W*) = *σ*(*ν*(*X, W*))——它对于例如*X*的导数在概念上应该是：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>f</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>ν</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow></mrow></math>

但这部分简单地是：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>ν</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>3</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub> <mo>)</mo></mrow></mrow></math>

这是很明确的，因为*σ*只是一个连续函数，我们可以在任何点评估它的导数，这里我们只是在<math><mrow><msub><mi>x</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>3</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub></mrow></math>处评估它。

此外，我们在先前的示例中推理，<math><mrow><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>=</mo> <msup><mi>W</mi> <mi>T</mi></msup></mrow></math>。因此：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>f</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>ν</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>3</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub> <mo>)</mo></mrow> <mo>×</mo> <msup><mi>W</mi> <mi>T</mi></msup></mrow></math>

与先前示例中一样，由于最终答案是一个数字，<math><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>3</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub> <mo>)</mo></mrow></mrow></math>，乘以*W*^(*T*)中与*X*相同形状的向量。

### 图表

这个函数的反向传播图，如[图1-19](#fig_01-20)所示，与先前示例的类似，甚至比数学更高级；我们只需要根据在矩阵乘法结果处评估的*σ*函数的导数再添加一个乘法。

![dlfs 0119](assets/dlfs_0119.png)

###### 图1-19\. 具有矩阵乘法的图：反向传播

### 代码

最后，编写反向传播也同样简单：

```py
def matrix_function_backward_1(X: ndarray,
                               W: ndarray,
                               sigma: Array_Function) -> ndarray:
    '''
 Computes the derivative of our matrix function with respect to
 the first element.
 '''
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # backward calculation
    dSdN = deriv(sigma, N)

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    # multiply them together; since dNdX is 1x1 here, order doesn't matter
    return np.dot(dSdN, dNdX)
```

请注意，我们在这里看到了与之前的三个嵌套函数示例中相同的动态：我们在正向传播中计算数量（这里只是`N`），然后在反向传播中使用它们。

### 这样对吗？

我们如何知道我们计算的这些导数是否正确？一个简单的测试是稍微扰动输入，观察输出的变化。例如，在这种情况下，*X*是：

```py
print(X)
```

```py
[[ 0.4723  0.6151 -1.7262]]
```

如果我们将*x*[3]从*-1.726*增加到*-1.716*，我们应该看到正向函数产生的值增加了*关于x[3]的梯度 × 0.01*。[图1-20](#fig_01-21)展示了这一点。

![dlfs 0120](assets/dlfs_0120.png)

###### 图1-20\. 梯度检查：一个示例

使用`matrix_function_backward_1`函数，我们可以看到梯度是`-0.1121`：

```py
print(matrix_function_backward_1(X, W, sigmoid))
```

```py
[[ 0.0852 -0.0557 -0.1121]]
```

为了测试这个梯度是否正确，我们应该看到，在将*x*[3]增加0.01后，函数的*output*大约减少`0.01 × -0.1121 = -0.001121`；如果我们看到减少的数量多或少于这个量，或者出现增加，那么我们就知道我们对链式法则的推理是错误的。然而，当我们进行这个计算时，^([2](ch01.html#idm45732630322664))，我们看到增加*x*[3]一点点确实会减少函数输出的值`0.01 × -0.1121`——这意味着我们计算的导数是正确的！

在本章结束时，我们将介绍一个建立在我们迄今为止所做的一切基础上，并直接应用于我们将在下一章中构建的模型的示例：一个计算图，从将一对二维矩阵相乘开始。

# 带有两个2D矩阵输入的计算图

在深度学习中，以及更普遍地在机器学习中，我们处理的操作以两个二维数组作为输入，其中一个代表数据批次*X*，另一个代表权重*W*。在下一章中，我们将深入探讨为什么在建模上这是有意义的，但在本章中我们将专注于这个操作背后的机制和数学。具体来说，我们将详细介绍一个简单的例子，并展示即使涉及2D矩阵的乘法，而不仅仅是1D向量的点积，我们在本章中一直使用的推理仍然在数学上是有意义的，并且实际上非常容易编码。

和以前一样，推导这些结果所需的数学并不困难，但有些混乱。尽管如此，结果是相当干净的。当然，我们将一步一步地分解它，并始终将其与代码和图表联系起来。

## 数学

假设：

<math display="block"><mrow><mi>X</mi> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><msub><mi>x</mi> <mn>11</mn></msub></mtd> <mtd><msub><mi>x</mi> <mn>12</mn></msub></mtd> <mtd><msub><mi>x</mi> <mn>13</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>x</mi> <mn>21</mn></msub></mtd> <mtd><msub><mi>x</mi> <mn>22</mn></msub></mtd> <mtd><msub><mi>x</mi> <mn>23</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>x</mi> <mn>31</mn></msub></mtd> <mtd><msub><mi>x</mi> <mn>32</mn></msub></mtd> <mtd><msub><mi>x</mi> <mn>33</mn></msub></mtd></mtr></mtable></mfenced></mrow></math>

和：

<math display="block"><mrow><mi>W</mi> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><msub><mi>w</mi> <mn>11</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>12</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>w</mi> <mn>21</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>22</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>w</mi> <mn>31</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>32</mn></msub></mtd></mtr></mtable></mfenced></mrow></math>

这可能对应于一个数据集，其中每个观测具有三个特征，三行可能对应于我们想要进行预测的三个不同观测。

现在我们将对这些矩阵定义以下简单的操作：

1.  将这些矩阵相乘。和以前一样，我们将把执行这个操作的函数表示为*ν*(*X*, *W*)，输出为*N*，所以*N* = *ν*(*X*, *W*)。

1.  通过一些可微函数*σ*将<math><mi>N</mi></math>结果传递，定义(*S* = *σ*(*N*)。

和以前一样，现在的问题是：输出*S*对*X*和*W*的梯度是多少？我们能否再次简单地使用链式法则？为什么或为什么不？

如果你稍微思考一下，你可能会意识到与我们之前看过的例子有所不同：*S现在是一个矩阵*，不再是一个简单的数字。毕竟，一个矩阵对另一个矩阵的梯度意味着什么呢？

这引出了一个微妙但重要的想法：我们可以对多维数组执行任何系列的操作，但为了定义对某个输出的“梯度”是有意义的，我们需要*求和*（或以其他方式聚合成一个数字）序列中的最终数组，以便“改变*X*的每个元素将如何影响输出”的概念甚至有意义。

因此，我们将在最后添加一个第三个函数*Lambda*，它只是取*S*的元素并将它们求和。

让我们在数学上具体化。首先，让我们将*X*和*W*相乘：

<math display="block"><mrow><mi>X</mi> <mo>×</mo> <mi>W</mi> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><msub><mi>x</mi> <mn>11</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>11</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>12</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>21</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>13</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>31</mn></msub></mrow></mtd> <mtd><mrow><msub><mi>x</mi> <mn>11</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>12</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>12</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>22</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>13</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>32</mn></msub></mrow></mtd></mtr> <mtr><mtd><mrow><msub><mi>x</mi> <mn>21</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>11</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>22</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>21</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>23</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>31</mn></msub></mrow></mtd> <mtd><mrow><msub><mi>x</mi> <mn>21</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>12</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>22</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>22</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>23</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>32</mn></msub></mrow></mtd></mtr> <mtr><mtd><mrow><msub><mi>x</mi> <mn>31</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>11</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>32</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>21</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>33</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>31</mn></msub></mrow></mtd> <mtd><mrow><msub><mi>x</mi> <mn>31</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>12</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>32</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>22</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>33</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>32</mn></msub></mrow></mtd></mtr></mtable></mfenced> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub></mrow></mtd> <mtd><mrow><mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub></mrow></mtd></mtr> <mtr><mtd><mrow><mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub></mrow></mtd> <mtd><mrow><mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub></mrow></mtd></mtr> <mtr><mtd><mrow><mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub></mrow></mtd> <mtd><mrow><mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub></mrow></mtd></mtr></mtable></mfenced></mrow></math>

其中我们为方便起见将结果矩阵中的第*i*行第*j*列表示为<math><mrow><mi>X</mi> <msub><mi>W</mi> <mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow></math>。

接下来，我们将通过*σ*将这个结果馈送，这只是意味着将*σ*应用于矩阵<math><mrow><mi>X</mi> <mo>×</mo> <mi>W</mi></mrow></math>的每个元素：

<math display="block"><mrow><mi>σ</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>×</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <msub><mi>x</mi> <mn>11</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>11</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>12</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>21</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>13</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>31</mn></msub> <mo>)</mo></mrow></mtd> <mtd><mrow><mi>σ</mi> <mo>(</mo> <msub><mi>x</mi> <mn>11</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>12</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>12</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>22</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>13</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mtd></mtr> <mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <msub><mi>x</mi> <mn>21</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>11</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>22</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>21</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>23</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>31</mn></msub> <mo>)</mo></mrow></mtd> <mtd><mrow><mi>σ</mi> <mo>(</mo> <msub><mi>x</mi> <mn>21</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>12</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>22</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>22</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>23</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mtd></mtr> <mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <msub><mi>x</mi> <mn>31</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>11</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>32</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>21</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>33</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>31</mn></msub> <mo>)</mo></mrow></mtd> <mtd><mrow><mi>σ</mi> <mo>(</mo> <msub><mi>x</mi> <mn>31</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>12</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>32</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>22</mn></msub> <mo>+</mo> <msub><mi>x</mi> <mn>33</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mtd></mtr></mtable></mfenced> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow></mtd> <mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow></mtd></mtr> <mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow></mtd> <mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow></mtd></mtr> <mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow></mtd> <mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mtd></mtr></mtable></mfenced></mrow></math>

最后，我们可以简单地总结这些元素：

<math display="block"><mrow><mi>L</mi> <mo>=</mo> <mi>Λ</mi> <mrow><mo>(</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>X</mi> <mo>×</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mi>Λ</mi> <mrow><mo>(</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow></mtd> <mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow></mtd></mtr> <mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow></mtd> <mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow></mtd></mtr> <mtr><mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow></mtd> <mtd><mrow><mi>σ</mi> <mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mtd></mtr></mtable></mfenced> <mo>)</mo></mrow> <mo>=</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow> <mo>+</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow> <mo>+</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow> <mo>+</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow> <mo>+</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow> <mo>+</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mrow></math>

现在我们回到了一个纯粹的微积分环境：我们有一个数字*L*，我们想要计算*L*对*X*和*W*的梯度；也就是说，我们想知道改变这些输入矩阵的*每个元素*（*x*[11]，*w*[21]等）会如何改变*L*。我们可以写成：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>)</mo></mrow> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>11</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>12</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>13</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>21</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>22</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>23</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>31</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>33</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr></mtable></mfenced></mrow></math>

现在我们从数学上理解了我们面临的问题。让我们暂停一下数学，跟上我们的图表和代码。

## 图表

从概念上讲，我们在这里所做的与我们在以前的例子中使用多个输入的计算图所做的类似；因此，[图1-21](#fig_01-22)应该看起来很熟悉。

![dlfs 0121](assets/dlfs_0121.png)

###### 图1-21\. 具有复杂前向传递的函数的图表

我们只是像以前一样将输入向前发送。我们声称即使在这种更复杂的情况下，我们也应该能够使用链式法则计算我们需要的梯度。

## 代码

我们可以编写如下代码：

```py
def matrix_function_forward_sum(X: ndarray,
                                W: ndarray,
                                sigma: Array_Function) -> float:
    '''
 Computing the result of the forward pass of this function with
 input ndarrays X and W and function sigma.
 '''
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)

    return L
```

# 有趣的部分：反向传播

现在我们想要为这个函数“执行反向传播”，展示即使涉及矩阵乘法，我们也可以计算出对输入`ndarray`的每个元素的`N`梯度。有了这一最后一步，开始在[第2章](ch02.html#fundamentals)中训练真实的机器学习模型将变得简单。首先，让我们在概念上提醒自己我们正在做什么。

## 图表

再次，我们所做的与本章中之前的例子类似；[图1-22](#fig_01-23)应该和[图1-21](#fig_01-22)一样熟悉。

![dlfs 0122](assets/dlfs_0122.png)

###### 图1-22\. 通过我们复杂的函数的反向传播

我们只需要计算每个组成函数的偏导数，并在其输入处评估它，将结果相乘以得到最终的导数。让我们依次考虑这些偏导数；唯一的方法就是通过数学。

## 数学

首先要注意的是我们可以直接计算这个。值*L*确实是*x*[11]、*x*[12]等等的函数，一直到*x*[33]。

然而，这似乎很复杂。链式法则的整个意义不就是我们可以将复杂函数的导数分解为简单的部分，计算每个部分，然后将结果相乘吗？确实，这个事实使得编写这些代码变得如此容易：我们只需逐步进行前向传播，保存结果，然后使用这些结果来评估反向传播所需的所有必要导数。

我将展示当涉及矩阵时，这种方法只“部分”有效。让我们深入探讨。

我们可以将*L*写成<math><mrow><mi>Λ</mi> <mo>(</mo> <mi>σ</mi> <mo>(</mo> <mi>ν</mi> <mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo> <mo>)</mo> <mo>)</mo></mrow></math>。如果这是一个常规函数，我们只需写出链式法则：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><mi>X</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>N</mi> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>S</mi> <mo>)</mo></mrow></mrow></math>

然后我们依次计算这三个偏导数。这正是我们之前在三个嵌套函数的函数中所做的，我们使用链式法则计算导数，[图1-22](#fig_01-23)表明这种方法对这个函数也应该适用。

第一个导数是最直接的，因此是最好的热身。我们想知道*L*（*Λ*的输出）每个元素增加时*L*会增加多少。由于*L*是*S*的所有元素的和，这个导数就是：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>S</mi> <mo>)</mo></mrow> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd> <mtd><mn>1</mn></mtd></mtr></mtable></mfenced></mrow></math>

因为增加*S*的任何元素，比如说，0.46个单位，会使*Λ*增加0.46个单位。

接下来，我们有<math><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>N</mi> <mo>)</mo></mrow></mrow></math>。这只是*σ*是什么函数的导数，在*N*中的元素处进行评估。在我们之前使用的“*XW*”语法中，这也很容易计算：

<math display="block"><mfenced close="]" open="["><mtable><mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr></mtable></mfenced></math>

请注意，此时我们可以确定我们可以将这两个导数*逐元素*相乘并计算<math><mrow><mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>N</mi> <mo>)</mo></mrow></mrow></math>：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>N</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>S</mi> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>N</mi> <mo>)</mo></mrow> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr></mtable></mfenced> <mo>×</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd> <mtd><mn>1</mn></mtd></mtr></mtable></mfenced> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr></mtable></mfenced></mrow></math>

然而，现在我们卡住了。根据图表和应用链式法则，我们想要的下一步是<math><mrow><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>)</mo></mrow></mrow></math>。然而，请记住，*N*，*ν*的输出，只是*X*与*W*的矩阵乘法的结果。因此，我们想知道增加*X*的每个元素（一个3×3矩阵）将如何增加*N*的每个元素（一个3×2矩阵）。如果你对这种概念感到困惑，那就是关键所在——我们并不清楚如何定义这个概念，或者如果我们这样做是否有用。

为什么现在成为问题了？之前，我们很幸运地*X*和*W*在形状上是彼此的转置。在这种情况下，我们可以证明<math><mrow><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>)</mo></mrow> <mo>=</mo> <msup><mi>W</mi> <mi>T</mi></msup></mrow></math>和<math><mrow><mfrac><mrow><mi>∂</mi><mi>ν</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>=</mo> <msup><mi>X</mi> <mi>T</mi></msup></mrow></math>。这里是否有类似的说法？

### “？”

更具体地说，这就是我们卡住的地方。我们需要弄清楚“？”中应该填什么：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>σ</mi> <mrow><mo>(</mo> <mi>N</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>×</mo> <mo>?</mo> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>11</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>12</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>21</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>22</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr> <mtr><mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>31</mn></msub> <mo>)</mo></mrow></mrow></mtd> <mtd><mrow><mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <msub><mi>W</mi> <mn>32</mn></msub> <mo>)</mo></mrow></mrow></mtd></mtr></mtable></mfenced> <mo>×</mo> <mo>?</mo></mrow></math>

### 答案

事实证明，由于乘法的工作方式，填充“?”的内容只是*W*^(*T*)，就像我们刚刚看到的向量点积的简单示例一样！验证这一点的方法是直接计算*L*对*X*的每个元素的偏导数；当我们这样做时，得到的矩阵确实（令人惊讶地）分解为：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>S</mi> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>N</mi> <mo>)</mo></mrow> <mo>×</mo> <msup><mi>W</mi> <mi>T</mi></msup></mrow></math>

第一个乘法是逐元素的，第二个是矩阵乘法。

这意味着*即使我们计算图中的操作涉及乘以具有多行和列的矩阵，即使这些操作的输出形状与输入的形状不同，我们仍然可以将这些操作包括在我们的计算图中，并使用“链式法则”逻辑进行反向传播。这是一个关键的结果，没有它，训练深度学习模型将会更加繁琐，你将在下一章中进一步体会到。

## 代码

让我们用代码封装我们刚刚推导出的内容，并希望在这个过程中巩固我们的理解：

```py
def matrix_function_backward_sum_1(X: ndarray,
                                   W: ndarray,
                                   sigma: Array_Function) -> ndarray:
    '''
 Compute derivative of matrix function with a sum with respect to the
 first matrix input.
 '''
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)

    # note: I'll refer to the derivatives by their quantities here,
    # unlike the math, where we referred to their function names

    # dLdS - just 1s
    dLdS = np.ones_like(S)

    # dSdN
    dSdN = deriv(sigma, N)

    # dLdN
    dLdN = dLdS * dSdN

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    # dLdX
    dLdX = np.dot(dSdN, dNdX)

    return dLdX
```

现在让我们验证一切是否正常：

```py
np.random.seed(190204)
X = np.random.randn(3, 3)
W = np.random.randn(3, 2)

print("X:")
print(X)

print("L:")
print(round(matrix_function_forward_sum(X, W, sigmoid), 4))
print()
print("dLdX:")
print(matrix_function_backward_sum_1(X, W , sigmoid))
```

```py
X:
[[-1.5775 -0.6664  0.6391]
 [-0.5615  0.7373 -1.4231]
 [-1.4435 -0.3913  0.1539]]
L:
2.3755

dLdX:
[[ 0.2489 -0.3748  0.0112]
 [ 0.126  -0.2781 -0.1395]
 [ 0.2299 -0.3662 -0.0225]]
```

与前面的例子一样，由于`dLdX`表示*X*相对于*L*的梯度，这意味着，例如，左上角的元素表示<math><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><msub><mi>x</mi> <mn>11</mn></msub></mrow></mfrac> <mrow><mo>(</mo> <mi>X</mi> <mo>,</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>=</mo> <mn>0.2489</mn></mrow></math>。因此，如果这个例子的矩阵运算是正确的，那么将*x*[11]增加0.001应该使*L*增加`0.01 × 0.2489`。实际上，我们看到这就是发生的：

```py
X1 = X.copy()
X1[0, 0] += 0.001

print(round(
        (matrix_function_forward_sum(X1, W, sigmoid) - \
         matrix_function_forward_sum(X, W, sigmoid)) / 0.001, 4))
```

```py
0.2489
```

看起来梯度计算正确！

### 描述这些梯度的可视化

回到本章开头我们注意到的内容，我们将问题中的元素*x*[11]通过一个包含许多操作的函数：矩阵乘法——实际上是将矩阵*X*中的九个输入与矩阵*W*中的六个输入组合在一起，得到六个输出——`sigmoid`函数，然后是求和。然而，我们也可以将这看作是一个名为“<math><mrow><mi>W</mi> <mi>N</mi> <mi>S</mi> <mi>L</mi></mrow></math>”的单一函数，如[图1-23](#fig_01-24)所示。

![dlfs 0123](assets/dlfs_0123.png)

###### 图1-23. 描述嵌套函数的另一种方式：作为一个函数，“WNSL”

由于每个函数都是可微的，整个过程只是一个可微函数，以*x*[11]为输入；因此，梯度就是回答问题“<math><mfrac><mrow><mi>d</mi><mi>L</mi></mrow> <mrow><mi>d</mi><msub><mi>x</mi> <mn>11</mn></msub></mrow></mfrac></math>”的答案。为了可视化这一点，我们可以简单地绘制*L*随着*x*[11]的变化而变化的情况。看一下*x*[11]的初始值，我们看到它是`-1.5775`：

```py
print("X:")
print(X)
```

```py
X:
[[-1.5775 -0.6664  0.6391]
 [-0.5615  0.7373 -1.4231]
 [-1.4435 -0.3913  0.1539]]
```

如果我们绘制从先前定义的计算图中将*X*和*W*输入的结果得到的*L*的值，或者换一种表示方法，从将`X`和`W`输入到前面代码中调用的函数中得到的结果，除了*x*[11]（或`X[0, 0]`）的值之外不改变，得到的图像看起来像[图1-24](#x11_vs_L_function_matrix_backward)所示。

![dlfs 0124](assets/dlfs_0124.png)

###### 图1-24. L与*x*[11]的关系，保持X和W的其他值不变

实际上，在*x*[11]的情况下，通过直观观察，这个函数沿着*L*轴增加的距离大约是0.5（从略高于2.1到略高于2.6），我们知道我们正在展示*x*[11]-轴上的变化为2，这将使斜率大约为<math><mrow><mfrac><mrow><mn>0.5</mn></mrow> <mn>2</mn></mfrac> <mo>=</mo> <mn>0.25</mn></mrow></math> —这正是我们刚刚计算的！

因此，我们复杂的矩阵数学实际上似乎已经使我们正确计算了相对于*X*的每个元素的*L*的偏导数。此外，相对于*W*的*L*的梯度也可以类似地计算。

###### 注意

相对于*W*的*L*的梯度表达式将是*X*^(*T*)。然而，由于*X*^(*T*)表达式从*L*的导数中因子出现的顺序，*X*^(*T*)将位于相对于*W*的*L*的梯度表达式的*左侧*：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>W</mi> <mo>)</mo></mrow> <mo>=</mo> <msup><mi>X</mi> <mi>T</mi></msup> <mo>×</mo> <mfrac><mrow><mi>∂</mi><mi>Λ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>S</mi> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mrow><mi>∂</mi><mi>σ</mi></mrow> <mrow><mi>∂</mi><mi>u</mi></mrow></mfrac> <mrow><mo>(</mo> <mi>N</mi> <mo>)</mo></mrow></mrow></math>

因此，在代码中，虽然我们会有`dNdW = np.transpose(X, (1, 0))`，但下一步将是：

```py
dLdW = np.dot(dNdW, dSdN)
```

而不是之前的`dLdX = np.dot(dSdN, dNdX`。

# 结论

在本章之后，您应该有信心能够理解复杂的嵌套数学函数，并通过将它们概念化为一系列箱子，每个代表一个单一的组成函数，通过连接的字符串来推理出它们的工作原理。具体来说，您可以编写代码来计算这些函数的输出相对于任何输入的导数，即使涉及到包含二维`ndarray`的矩阵乘法，也能理解这些导数计算背后的数学原理。这些基础概念正是我们在下一章开始构建和训练神经网络所需要的，以及在之后的章节中从头开始构建和训练深度学习模型所需要的。继续前进！

^([1](ch01.html#idm45732632700344-marker)) 这将使我们能够轻松地在矩阵乘法中添加偏差。

^([2](ch01.html#idm45732630322664-marker)) 在整个过程中，我将提供指向GitHub存储库的相关补充材料的链接，该存储库包含本书的代码，包括[本章](https://oreil.ly/2ZUwKOZ)的代码。

^([3](ch01.html#idm45732628068456-marker)) 在接下来的部分中，我们将专注于计算`N`相对于`X`的梯度，但相对于`W`的梯度也可以通过类似的方式推理。

^([4](ch01.html#idm45732627740296-marker)) 我们在[“矩阵链规则”](app01.html#matrix-chain-rule)中进行了这样的操作。

^([5](ch01.html#idm45732627323432-marker)) 完整的函数可以在[书的网站](https://oreil.ly/deep-learning-github)找到；它只是前一页显示的`matrix function backward sum`函数的一个子集。
