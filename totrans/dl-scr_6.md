# 第6章。循环神经网络

在这一章中，我们将介绍循环神经网络（RNNs），这是一类用于处理数据序列的神经网络架构。到目前为止，我们看到的神经网络将它们接收到的每一批数据视为一组独立的观察结果；在我们在[第4章](ch04.html#extensions)中看到的全连接神经网络或[第5章](ch05.html#convolution)中看到的卷积神经网络中，没有某些MNIST数字在其他数字之前或之后到达的概念。然而，许多种类的数据在本质上是有序的，无论是时间序列数据，在工业或金融背景下可能会处理的数据，还是语言数据，其中字符、单词、句子等是有序的。循环神经网络旨在学习如何接收*这些数据序列*并返回一个正确的预测作为输出，无论这个正确的预测是关于第二天金融资产价格的还是关于句子中下一个单词的。

处理有序数据将需要对我们在前几章中看到的全连接神经网络进行三种改变。首先，它将涉及“向我们馈送神经网络的`ndarray`添加一个新维度”。以前，我们馈送给神经网络的数据在本质上是二维的——每个`ndarray`有一个维度表示观察数量，另一个维度表示特征数量；另一种思考方式是*每个观察*是一个一维向量。使用循环神经网络，每个输入仍然会有一个维度表示观察数量，但每个观察将被表示为一个二维`ndarray`：一个维度将表示数据序列的长度，第二个维度将表示每个序列元素上存在的特征数量。因此，RNN的整体输入将是一个形状为`[batch_size, sequence_length, num_features]`的三维`ndarray`——一批序列。

第二，当然，为了处理这种新的三维输入，我们将不得不使用一种新的神经网络架构，这将是本章的主要焦点。然而，第三个改变是我们将在本章开始讨论的地方：我们将不得不使用完全不同的框架和不同的抽象来处理这种新形式的数据。为什么？在全连接和卷积神经网络的情况下，即使每个“操作”实际上代表了*许多*个单独的加法和乘法（如矩阵乘法或卷积的情况），它都可以被描述为一个单一的“小工厂”，在前向和后向传递中都将一个`ndarray`作为输入并产生一个`ndarray`作为输出（可能在这些计算中使用另一个表示操作参数的`ndarray`）。事实证明，循环神经网络无法以这种方式实现。在继续阅读以了解原因之前，花些时间思考一下：神经网络架构的哪些特征会导致我们迄今为止构建的框架崩溃？虽然答案很有启发性，但完整的解决方案涉及到深入实现细节的概念，超出了本书的范围。为了开始解开这个问题，让我们揭示我们迄今为止使用的框架的一个关键限制。

# 关键限制：处理分支

事实证明，我们的框架无法训练具有像[图6-1](#fig_06_01)所示的计算图的模型。

![长度为一的输出](assets/dlfs_0601.png)

###### 图6-1。导致我们的操作框架失败的计算图：在前向传递过程中同一数量被多次重复，这意味着我们不能像以前那样在后向传递过程中按顺序发送梯度

这有什么问题吗？将前向传递转换为代码似乎没问题（请注意，我们这里仅出于说明目的编写了`Add`和`Multiply`操作）：

```py
a1 = torch.randn(3,3)
w1 = torch.randn(3,3)

a2 = torch.randn(3,3)
w2 = torch.randn(3,3)

w3 = torch.randn(3,3)

# operations
wm1 = WeightMultiply(w1)
wm2 = WeightMultiply(w2)
add2 = Add(2, 1)
mult3 = Multiply(2, 1)

b1 = wm1.forward(a1)
b2 = wm2.forward(a2)
c1 = add2.forward((b1, b2))
L = mult3.forward((c1, b2))
```

问题开始于我们开始后向传递时。假设我们想要使用我们通常的链式法则逻辑来计算`L`相对于`w1`的导数。以前，我们只需按相反顺序在每个操作上调用`backward`。在这里，由于在前向传递中*重复使用`b2`*，这种方法不起作用。例如，如果我们从`mult3`开始调用`backward`，我们将得到每个输入`c1`和`b2`的梯度。然而，如果我们接着在`add2`上调用`backward`，我们不能只传入`c1`的梯度：我们还必须以某种方式传入`b2`的梯度，因为这也会影响损失`L`。因此，为了正确执行此图的后向传递，我们不能只按照完全相反的顺序移动操作；我们必须手动编写类似以下内容的内容：

```py
c1_grad, b2_grad_1 = mult3.backward(L_grad)

b1_grad, b2_grad_2 = add2.backward(c1_grad)

# combine these gradients to reflect the fact that b2 is used twice on the
# forward pass
b2_grad = b2_grad_1 + b2_grad_2

a2_grad = wm2.backward(b2_grad)

a1_grad = wm1.backward(b1_grad)
```

在这一点上，我们可能完全可以跳过使用`Operation`；我们可以简单地保存在前向传递中计算的所有量，并在后向传递中重复使用它们，就像我们在[第2章](ch02.html#fundamentals)中所做的那样！我们可以通过手动定义网络前向和后向传递中要执行的各个计算来始终编写任意复杂的神经网络，就像我们在[第2章](ch02.html#fundamentals)中写出了两层神经网络后向传递中涉及的17个单独操作一样（事实上，我们稍后在本章中的“RNN单元”中会做类似的事情）。我们尝试使用`Operation`构建一个灵活的框架，让我们以高层次的术语描述神经网络，并让所有低级别的计算“自动工作”。虽然这个框架展示了许多关于神经网络的关键概念，但现在我们看到了它的局限性。

有一个优雅的解决方案：自动微分，这是一种完全不同的实现神经网络的方式。我们将在这里涵盖这个概念的足够部分，以便让您了解它的工作原理，但不会进一步构建一个完整功能的自动微分框架将需要几章的篇幅。此外，当我们涵盖PyTorch时，我们将看到如何*使用*一个高性能的自动微分框架。尽管如此，自动微分是一个重要的概念，需要从第一原则理解，在我们深入研究RNN之前，我们将为其设计一个基本框架，并展示它如何解决在前面示例中描述的前向传递中重复使用对象的问题。

# 自动微分

正如我们所看到的，有一些神经网络架构，对于我们迄今使用的`Operation`框架来说，很难轻松地计算输出相对于输入的梯度，而我们必须这样做才能训练我们的模型。自动微分允许我们通过完全不同的路径计算这些梯度：而不是`Operation`是构成网络的原子单位，我们定义一个包装在数据周围的类，允许数据跟踪在其上执行的操作，以便数据可以在参与不同操作时不断累积梯度。为了更好地理解这种“梯度累积”是如何工作的，让我们开始编码吧。

## 编写梯度累积

为了自动跟踪梯度，我们必须重写执行数据基本操作的Python方法。在Python中，使用诸如`+`或`-`之类的运算符实际上调用诸如`__add__`和`__sub__`之类的底层隐藏方法。例如，这是`+`的工作原理：

```py
a = array([3,3])
print("Addition using '__add__':", a.__add__(4))
print("Addition using '+':", a + 4)
```

```py
Addition using '__add__': [7 7]
Addition using '+': [7 7]
```

我们可以利用这一点编写一个类，该类包装了典型的Python“数字”（`float`或`int`）并覆盖了`add`和`mul`方法：

```py
Numberable = Union[float, int]

def ensure_number(num: Numberable) -> NumberWithGrad:
    if isinstance(num, NumberWithGrad):
        return num
    else:
        return NumberWithGrad(num)

class NumberWithGrad(object):

    def __init__(self,
                 num: Numberable,
                 depends_on: List[Numberable] = None,
                 creation_op: str = ''):
        self.num = num
        self.grad = None
        self.depends_on = depends_on or []
        self.creation_op = creation_op

    def __add__(self,
                other: Numberable) -> NumberWithGrad:
        return NumberWithGrad(self.num + ensure_number(other).num,
                              depends_on = [self, ensure_number(other)],
                              creation_op = 'add')

    def __mul__(self,
                other: Numberable = None) -> NumberWithGrad:

        return NumberWithGrad(self.num * ensure_number(other).num,
                              depends_on = [self, ensure_number(other)],
                              creation_op = 'mul')

    def backward(self, backward_grad: Numberable = None) -> None:
        if backward_grad is None: # first time calling backward
            self.grad = 1
        else:
            # These lines allow gradients to accumulate.
            # If the gradient doesn't exist yet, simply set it equal
            # to backward_grad
            if self.grad is None:
                self.grad = backward_grad
            # Otherwise, simply add backward_grad to the existing gradient
            else:
                self.grad += backward_grad

        if self.creation_op == "add":
            # Simply send backward self.grad, since increasing either of these
            # elements will increase the output by that same amount
            self.depends_on[0].backward(self.grad)
            self.depends_on[1].backward(self.grad)

        if self.creation_op == "mul":

            # Calculate the derivative with respect to the first element
            new = self.depends_on[1] * self.grad
            # Send backward the derivative with respect to that element
            self.depends_on[0].backward(new.num)

            # Calculate the derivative with respect to the second element
            new = self.depends_on[0] * self.grad
            # Send backward the derivative with respect to that element
            self.depends_on[1].backward(new.num)
```

这里有很多事情要做，让我们解开这个`NumberWithGrad`类并看看它是如何工作的。请记住，这样一个类的目标是能够编写简单的操作并自动计算梯度；例如，假设我们写：

```py
a = NumberWithGrad(3)

b = a * 4
c = b + 5
```

在这一点上，通过增加*ϵ*，`a`将增加多少会增加`c`的值？很明显，它将增加`c`<math><mrow><mn>4</mn> <mo>×</mo> <mi>ϵ</mi></mrow></math>。确实，使用前面的类，如果我们首先写：

```py
c.backward()
```

然后，不需要编写`for`循环来迭代`Operation`，我们可以写：

```py
print(a.grad)
```

```py
4
```

这是如何工作的？前一个类中融入的基本见解是，每当在`NumberWithGrad`上执行`+`或`*`操作时，都会创建一个新的`NumberWithGrad`，第一个`NumberWithGrad`作为依赖项。然后，当在`NumberWithGrad`上调用`backward`时，就像之前在`c`上调用的那样，用于创建`c`的所有`NumberWithGrad`的所有梯度都会自动计算。因此，确实，不仅计算了`a`的梯度，还计算了`b`的梯度：

```py
print(b.grad)
```

```py
1
```

然而，这个框架的真正美妙之处在于它允许`NumberWithGrad`*累积*梯度，从而在一系列计算中多次重复使用，并且我们最终得到正确的梯度。我们将用相同的一系列操作来说明这一点，这些操作在之前让我们困惑，使用`NumberWithGrad`多次进行一系列计算，然后详细解释它是如何工作的。

### 自动微分示例

这是一系列计算，其中`a`被多次重复使用：

```py
a = NumberWithGrad(3)

b = a * 4
c = b + 3
d = c * (a + 2)
```

我们可以计算出，如果我们进行这些操作，*d* = 75，但正如我们所知道的，真正的问题是：增加`a`的值将如何增加`d`的值？我们可以首先通过数学方法解决这个问题。我们有：

<math display="block"><mrow><mi>d</mi> <mo>=</mo> <mrow><mo>(</mo> <mn>4</mn> <mi>a</mi> <mo>+</mo> <mn>3</mn> <mo>)</mo></mrow> <mo>×</mo> <mrow><mo>(</mo> <mi>a</mi> <mo>+</mo> <mn>2</mn> <mo>)</mo></mrow> <mo>=</mo> <mn>4</mn> <msup><mi>a</mi> <mn>2</mn></msup> <mo>+</mo> <mn>11</mn> <mi>a</mi> <mo>+</mo> <mn>6</mn></mrow></math>

因此，使用微积分中的幂规则：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>d</mi></mrow> <mrow><mi>∂</mi><mi>a</mi></mrow></mfrac> <mo>=</mo> <mn>8</mn> <mi>a</mi> <mo>+</mo> <mn>11</mn></mrow></math>

对于*a* = 3，因此，这个导数的值应该是<math><mrow><mn>8</mn> <mo>×</mo> <mn>3</mn> <mo>+</mo> <mn>11</mn> <mo>=</mo> <mn>35</mn></mrow></math>。通过数值确认：

```py
def forward(num: int):
    b = num * 4
    c = b + 3
    return c * (num + 2)

print(round(forward(3.01) - forward(2.99)) / 0.02), 3)
```

```py
35.0
```

现在，观察到当我们使用自动微分框架计算梯度时，我们得到相同的结果：

```py
a = NumberWithGrad(3)

b = a * 4
c = b + 3
d = (a + 2)
e = c * d
e.backward()

print(a.grad)
```

```py
35
```

### 解释发生了什么

正如我们所看到的，自动微分的目标是使*数据对象本身*——数字、`ndarray`、`张量`等——成为分析的基本单位，而不是以前的`Operation`。

所有自动微分技术都有以下共同点：

+   每种技术都包括一个包装实际计算数据的类。在这里，我们将`NumberWithGrad`包装在`float`和`int`周围；例如，在PyTorch中，类似的类称为`Tensor`。

+   重新定义常见操作，如加法、乘法和矩阵乘法，以便它们始终返回该类的成员；在前面的情况下，我们确保*要么*是`NumberWithGrad`和`NumberWithGrad`的加法*要么*是`NumberWithGrad`和`float`或`int`的加法。

+   `NumberWithGrad`类必须包含有关如何计算梯度的信息，考虑到前向传播时发生了什么。以前，我们通过在类中包含一个`creation_op`参数来实现这一点，该参数简单地记录了`NumberWithGrad`是如何创建的。

+   在反向传播过程中，梯度是使用底层数据类型而不是包装器向后传递的。这意味着梯度的类型是`float`和`int`，而不是`NumberWithGrad`。

+   正如在本节开头提到的，自动微分允许我们在前向传播期间重复使用计算的量——在前面的示例中，我们两次使用`a`而没有问题。允许这样做的关键是这些行：

    ```py
    if self.grad is None:
        self.grad = backward_grad
    else:
        self.grad += backward_grad
    ```

    这些行表明，在接收到新的梯度`backward_grad`时，`NumberWithGrad`应该将`NumberWithGrad`的梯度初始化为这个值，或者简单地将该值添加到`NumberWithGrad`的现有梯度中。这就是允许`NumberWithGrad`在模型中重复使用相关对象时累积梯度的原因。

这就是我们将涵盖的自动微分的全部内容。现在让我们转向激发这一偏离的模型结构，因为在前向传播过程中需要重复使用某些量来进行预测。

# 循环神经网络的动机

正如我们在本章开头讨论的那样，循环神经网络旨在处理以序列形式出现的数据：每个观察结果不再是具有`n`个特征的向量，而是一个二维数组，维度为`n`个特征乘以`t`个时间步。这在[图6-2](#fig_06_02)中有所描述。

![长度为一的输出](assets/dlfs_0602.png)

###### 图6-2\. 顺序数据：在每个时间步中我们有n个特征

在接下来的几节中，我们将解释RNN如何适应这种形式的数据，但首先让我们尝试理解为什么我们需要它们。仅仅使用普通的前馈神经网络来处理这种类型的数据会有什么限制？一种方法是将每个时间步表示为一个独立的特征集。例如，一个观察结果可以具有来自时间`t = 1`的特征和来自时间`t = 2`的目标值，下一个观察结果可以具有来自时间`t = 2`的特征和来自时间`t = 3`的目标值，依此类推。如果我们想要使用*多个*时间步的数据来进行每次预测，而不仅仅是来自一个时间步的数据，我们可以使用`t = 1`和`t = 2`的特征来预测`t = 3`的目标，使用`t = 2`和`t = 3`的特征来预测`t = 4`的目标，依此类推。

然而，将每个时间步视为独立的方式忽略了数据是按顺序排列的事实。我们如何理想地利用数据的顺序性来做出更好的预测？解决方案看起来会像这样：

1.  使用时间步`t = 1`的特征来预测对应时间`t = 1`的目标。

1.  使用时间步`t = 2`的特征以及从`t = 1`包括`t = 1`的目标值的信息来预测`t = 2`。

1.  使用时间步`t = 3`的特征以及从`t = 1`和`t = 2`累积的信息来预测`t = 3`时的结果。

1.  然后，每一步都使用所有先前时间步的信息来进行预测。

为了做到这一点，似乎我们希望逐个序列元素地通过神经网络传递我们的数据，首先传递第一个时间步的数据，然后传递下一个时间步的数据，依此类推。此外，我们希望我们的神经网络在通过新的序列元素时“累积信息”关于它之前所看到的内容。我们将在本章的其余部分详细讨论循环神经网络如何做到这一点。正如我们将看到的，虽然有几种循环神经网络的变体，但它们在处理数据时都共享一个共同的基本结构；我们将大部分时间讨论这种结构，并在最后讨论这些变体的不同之处。

# 循环神经网络简介

让我们通过高层次的方式开始讨论RNN，看看数据是如何通过“前馈”神经网络传递的。在这种类型的网络中，数据通过一系列*层*向前传递。对于单个观察结果，每一层的输出是该观察结果在该层的神经网络“表示”。在第一层之后，该表示由原始特征的组合组成；在下一层之后，它由这些表示的组合或原始特征的“特征的特征”组成，依此类推，直到网络中的后续层。因此，在每次向前传递之后，网络在每个层的输出中包含许多原始观察结果的表示。这在[图6-3](#fig_06_03)中有所体现。

![长度为一的输出](assets/dlfs_0603.png)

###### 图6-3。一个常规的神经网络将观察结果向前传递，并在每一层之后将其转换为不同的表示

然而，当下一组观察结果通过网络传递时，这些表示将被丢弃；循环神经网络及其所有变体的关键创新是*将这些表示传递回网络*，以及下一组观察结果。这个过程看起来是这样的：

1.  在第一个时间步，`t = 1`，我们将通过第一个时间步的观察结果（可能还有随机初始化的表示）进行传递。我们将输出`t = 1`的预测，以及每一层的表示。

1.  在下一个时间步中，我们将通过第二个时间步的观察结果`t = 2`，以及在第一个时间步计算的表示（再次，这些只是神经网络层的输出），并以某种方式将它们结合起来（正是在这个结合步骤中，我们将学习到的RNN的变体有所不同）。我们将使用这两个信息来输出`t = 2`的预测，以及每一层的*更新*表示，这些表示现在是输入在`t = 1`和`t = 2`时传递的函数。

1.  在第三个时间步中，我们将通过来自`t = 3`的观察结果，以及现在包含来自`t = 1`和`t = 2`信息的表示，利用这些信息对`t = 3`进行预测，以及每一层的额外更新的表示，现在包含时间步1-3的信息。

这个过程在[图6-4](#fig_06_04)中描述。

![长度为一的输出](assets/dlfs_0604.png)

###### 图6-4。循环神经网络将每一层的表示向前传递到下一个时间步

我们看到每一层都有一个“持久”的表示，随着时间的推移而更新，因为新的观察结果被传递。事实上，这就是为什么RNN不适用于我们为之前章节编写的`Operation`框架的原因：每一层的表示的`ndarray`会不断更新和重复使用，以便使用RNN对一系列数据进行一次预测。因为我们无法使用之前章节的框架，我们将不得不从头开始考虑如何构建处理RNN的类。

## RNN的第一个类：RNNLayer

根据我们希望RNN工作的描述，我们至少知道我们需要一个`RNNLayer`类，该类将逐个序列元素向前传递数据序列。现在让我们深入了解这样一个类应该如何工作的细节。正如我们在本章中提到的，RNN将处理每个观察都是二维的数据，维度为`(sequence_length, num_features)`；由于在计算上总是更有效率地批量传递数据，`RNNLayer`将需要接收三维的`ndarray`，大小为`(batch_size, sequence_length, num_features)`。然而，我在前一节中解释过，我们希望逐个序列元素通过我们的`RNNLayer`传递数据；如果我们的输入`data`是`(batch_size, sequence_length, num_features)`，我们如何做到这一点呢？这样做：

1.  从第二轴选择一个二维数组，从`data[:, 0, :]`开始。这个`ndarray`的形状将是`(batch_size, num_features)`。

1.  为`RNNLayer`初始化一个“隐藏状态”，该状态将随着传入的每个序列元素而不断更新，这次的形状为`(batch_size, hidden_size)`。这个`ndarray`将代表层对已经在先前时间步传入的数据的“累积信息”。

1.  将这两个`ndarray`通过该层的第一个时间步向前传递。我们将设计`RNNLayer`以输出与输入不同维度的`ndarray`，就像常规的`Dense`层一样，因此输出将是形状为`(batch_size, num_outputs)`。此外，更新神经网络对每个观察的表示：在每个时间步，我们的`RNNLayer`还应该输出一个形状为`(batch_size, hidden_size)`的`ndarray`。

1.  从`data`中选择下一个二维数组：`data[:, 1, :]`。

1.  将这些数据以及RNN在第一个时间步输出的表示值传递到该层的第二个时间步，以获得另一个形状为`(batch_size, num_outputs)`的输出，以及形状为`(batch_size, hidden_size)`的更新表示。

1.  一直持续到所有`sequence_length`时间步都通过该层。然后将所有结果连接在一起，以获得该层的输出形状为`(batch_size, sequence_length, num_outputs)`。

这给了我们一个关于我们的`RNNLayer`应该如何工作的想法——当我们编写代码时，我们将巩固这种理解——但它也暗示我们需要另一个类来处理接收数据并更新每个时间步的层隐藏状态。为此，我们将使用`RNNNode`，这是我们将要介绍的下一个类。

## RNN的第二个类：RNNNode

根据前一节的描述，`RNNNode`应该有一个`forward`方法，具有以下输入和输出：

+   两个`ndarray`作为输入：

    +   一个用于网络的数据输入，形状为`[batch_size, num_features]`

    +   一个用于该时间步观察的表示的形状为`[batch_size, hidden_size]`

+   两个`ndarray`作为输出：

    +   一个用于网络在该时间步的输出的数组，形状为`[batch_size, num_outputs]`

    +   一个用于该时间步观察的*更新*表示，形状为：`[batch_size, hidden_size]`

接下来，我们将展示`RNNNode`和`RNNLayer`这两个类如何配合。

## 将这两个类结合起来

`RNNLayer`类将包装在一个`RNNNode`列表周围，并且（至少）包含一个具有以下输入和输出的`forward`方法：

+   输入：形状为`[batch_size, sequence_length, num_features]`的一批观察序列

+   输出：这些序列的神经网络输出的形状为`[batch_size, sequence_length, num_outputs]`

[图6-5](#fig_06_05)展示了数据如何通过具有每个五个`RNNNode`的两个`RNNLayer`的RNN向前传播的顺序。在每个时间步，初始维度为`feature_size`的输入依次通过每个`RNNLayer`中的第一个`RNNNode`向前传递，网络最终在该时间步输出维度为`output_size`的预测。此外，每个`RNNNode`将“隐藏状态”向前传递到每层内的下一个`RNNNode`。一旦每个五个时间步的数据都通过所有层向前传递，我们将得到一个形状为`(5, output_size)`的最终预测集，其中`output_size`应该与目标的维度相同。然后，这些预测将与目标进行比较，并计算损失梯度，启动反向传播。[图6-5](#fig_06_05)总结了这一点，展示了数据如何从第一个（1）到最后一个（10）依次通过5×2个`RNNNode`的顺序流动。

![长度为一的输出](assets/dlfs_0605.png)

###### 图6-5。设计用于处理长度为5的序列的具有两层的RNN中数据将如何流动的顺序

或者，数据可以按照[图6-6](#fig_06_06)中显示的顺序在RNN中流动。无论顺序如何，以下步骤必须发生：

+   每个层都需要在给定时间步处理其数据，然后才能处理下一层 - 例如，在[图6-5](#fig_06_05)中，2不能在1之前发生，4不能在3之前发生。

+   同样，每个层都必须按顺序处理其所有时间步 - 例如，在[图6-5](#fig_06_05)中，4不能在2之前发生，3不能在1之前发生。

+   最后一层必须为每个观测输出维度`feature_size`。

![长度为一的输出](assets/dlfs_0606.png)

###### 图6-6。数据在同一RNN的前向传播过程中可能流动的另一种顺序

这涵盖了RNN前向传播的工作原理。那么反向传播呢？

## 反向传播

通过递归神经网络的反向传播通常被描述为一个称为“递归神经网络的反向传播”算法。虽然这确实描述了反向传播过程中发生的事情，但这让事情听起来比实际复杂得多。牢记数据如何通过RNN向前流动的解释，我们可以这样描述反向传播过程：我们通过将梯度反向通过网络传递，以与在前向传播过程中向前传递输入的顺序相反的顺序将数据向后传递，这与我们在常规前馈网络中所做的事情是一样的。

观察图[6-5](#fig_06_05)和[6-6](#fig_06_06)中的图表，在前向传播过程中：

1.  我们从形状为`(feature_size, sequence_length)`的一批观测开始。

1.  这些输入被分解为单个`sequence_length`元素，并逐个传入网络。

1.  每个元素都通过所有层，最终被转换为大小为`output_size`的输出。

1.  同时，层将隐藏状态向前传递到下一个时间步的层的计算中。

1.  这将持续进行所有`sequence_length`时间步，最终产生大小为`(output_size, sequence_length)`的总输出。

反向传播只是以相反的方式工作：

1.  我们从形状为`[output_size, sequence_length]`的*梯度*开始，表示输出的每个元素（也是形状为`[output_size, sequence_length]`的）最终对该批观测的损失产生了多大影响。

1.  这些梯度被分解为单个`sequence_length`元素，并*以相反顺序*通过层*向后*传递。

1.  单个元素的梯度通过所有层向后传递。

1.  同时，各层将“与该时间步的隐藏状态相关的损失的梯度”向后传递到先前时间步的层的计算中。

1.  这将持续进行所有`sequence_length`时间步，直到梯度已经向网络中的每一层传递，从而使我们能够计算出损失相对于每个权重的梯度，就像在常规前馈网络的情况下一样。

这种前向传递和后向传递之间的并行性在[图6-7](#fig_06_07)中得到了突出，该图显示了数据在RNN在后向传递过程中的流动方式。当然，您会注意到，它与[图6-5](#fig_06_05)相同，但箭头方向相反，数字也有所改变。

![长度为一的输出](assets/dlfs_0607.png)

###### 图6-7。在后向传递中，RNNs将数据传递的方向与前向传递相反

这突显了，在高层次上，`RNNLayer`的前向和后向传递与普通神经网络中的层非常相似：它们都接收特定形状的`ndarray`作为输入，输出另一种形状的`ndarray`，在后向传递中接收与其输出相同形状的输出梯度，并产生与其输入相同形状的输入梯度。然而，在`RNNLayer`中处理权重梯度的方式与其他层有关键差异，因此在我们开始编码之前，我们将简要介绍一下这一点。

### 在RNN中累积权重的梯度

在循环神经网络中，就像在常规神经网络中一样，每一层都会有*一组权重*。这意味着相同的权重集将影响所有`sequence_length`时间步的层输出；因此，在反向传播过程中，相同的权重集将接收`sequence_length`不同的梯度。例如，在[图6-7](#fig_06_07)中显示的反向传播中标记为“1”的圆圈中，第二层将接收最后一个时间步的梯度，而在标记为“3”的圆圈中，该层将接收倒数第二个时间步的梯度；这两者都将由相同的权重驱动。因此，在反向传播过程中，我们将不得不*累积权重的梯度*，这意味着无论我们选择如何存储权重，我们都将不得不使用类似以下的方法更新它们的梯度：

```py
weight_grad += grad_from_time_step
```

这与`Dense`和`Conv2D`层不同，在这些层中，我们只是将参数存储在`param_grad`参数中。

我们已经阐明了RNN的工作原理以及我们想要构建的类来实现它们；现在让我们开始研究细节。

# RNN：代码

让我们从几种实现RNN与我们在本书中介绍的其他神经网络类似的方式开始：

1.  RNN仍然通过一系列层向前传递数据，这些层在前向传递时将输出向前传递，在后向传递时将梯度向后传递。因此，例如，无论我们的`NeuralNetwork`类的等价物最终是什么，它仍将具有`layers`属性作为`RNNLayer`的列表，并且前向传递将包含如下代码：

    ```py
    def forward(self, x_batch: ndarray) -> ndarray:

        assert_dim(ndarray, 3)

        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out
    ```

1.  RNN的`Loss`与以前相同：最后一个`Layer`生成一个`ndarray` `output`，与`y_batch`进行比较，计算出一个单一值，并返回相对于`Loss`输入的该值的梯度，形状与`output`相同。我们将不得不修改softmax函数，以便与形状为`[batch_size, sequence_length, feature_size]`的`ndarray`适当地配合，但我们可以处理这个问题。

1.  `Trainer`大部分都是相同的：我们循环遍历我们的训练数据，选择输入数据和输出数据的批次，并不断将它们通过我们的模型，产生损失值，告诉我们我们的模型是否在学习，并在每个批次通过后更新权重。说到这里...

1.  我们的`Optimizer`保持不变。正如我们将看到的，我们将不得不更新如何在每个时间步提取`params`和`param_grads`，但“更新规则”（我们在类中的`_update_rule`函数中捕获）保持不变。

`Layer`本身是有趣的地方。

## RNNLayer类

之前，我们给`Layer`提供了一组`Operation`，用于向前传递数据并向后发送梯度。`RNNLayer`将完全不同；它们现在必须保持一个“隐藏状态”，随着新数据被馈送并在每个时间步骤以某种方式与数据“组合”，该状态将不断更新。这应该如何工作？我们可以使用图[6-5](#fig_06_05)和[6-6](#fig_06_06)作为指导：它们建议每个`RNNLayer`应该有一个`RNNNode`列表作为属性，然后该层的`input`中的每个序列元素应该逐个通过每个`RNNNode`传递。每个`RNNNode`将接收此序列元素，以及该层的“隐藏状态”，并在该时间步骤为该层产生一个输出，同时更新该层的隐藏状态。

为了澄清所有这些，让我们深入研究并开始编码：我们将按顺序介绍`RNNLayer`的初始化方式，它在前向传递期间如何发送数据，以及在反向传递期间如何发送数据。

### 初始化

每个`RNNLayer`将以以下方式开始：

+   一个`int` `hidden_size`

+   一个`int` `output_size`

+   一个形状为`(1, hidden_size)`的`ndarray` `start_H`，表示该层的隐藏状态

此外，就像在常规神经网络中一样，当我们初始化层时，我们将设置`self.first = True`；第一次将数据传递到`forward`方法时，我们将将接收到的`ndarray`传递到一个`_init_params`方法中，初始化参数，并设置`self.first = False`。

有了初始化的层，我们准备描述如何将数据发送到前面。

### 前向方法

`forward`方法的大部分将包括接收形状为`(batch_size, sequence_length, feature_size)`的`ndarray` `x_seq_in`，并按顺序通过该层的`RNNNode`。在以下代码中，`self.nodes`是该层的`RNNNode`，`H_in`是该层的隐藏状态：

```py
sequence_length = x_seq_in.shape[1]

x_seq_out = np.zeros((batch_size, sequence_length, self.output_size))

for t in range(sequence_length):

    x_in = x_seq_in[:, t, :]

    y_out, H_in = self.nodes[t].forward(x_in, H_in, self.params)

    x_seq_out[:, t, :] = y_out
```

关于隐藏状态`H_in`的一点说明：`RNNLayer`的隐藏状态通常表示为一个向量，但每个`RNNNode`中的操作要求隐藏状态为大小为`(batch_size, hidden_size)`的`ndarray`。因此，在每次前向传递的开始时，我们简单地“重复”隐藏状态：

```py
batch_size = x_seq_in.shape[0]

H_in = np.copy(self.start_H)

H_in = np.repeat(H_in, batch_size, axis=0)
```

在前向传递之后，我们取得构成批次的观测值的平均值，以获得该层的更新隐藏状态：

```py
self.start_H = H_in.mean(axis=0, keepdims=True)
```

此外，我们可以从这段代码中看到，`RNNNode`将必须具有一个接收两个形状数组的`forward`方法：

+   `(batch_size, feature_size)`

+   `(batch_size, hidden_size)`

并返回两个形状的数组：

+   `(batch_size, output_size)`

+   `(batch_size, hidden_size)`

我们将在下一节中介绍`RNNNode`（及其变体）。但首先让我们介绍`RNNLayer`类的`backward`方法。

### 反向方法

由于`forward`方法输出了`x_seq_out`，`backward`方法将接收与`x_seq_out`形状相同的梯度，称为`x_seq_out_grad`。与`forward`方法相反，我们通过`RNNNode`将此梯度*向后*传递，最终返回整个层的形状为`(batch_size, sequence_length, self.feature_size)`的`x_seq_in_grad`作为梯度：

```py
h_in_grad = np.zeros((batch_size, self.hidden_size))

sequence_length = x_seq_out_grad.shape[1]

x_seq_in_grad = np.zeros((batch_size, sequence_length, self.feature_size))

for t in reversed(range(sequence_length)):

    x_out_grad = x_seq_out_grad[:, t, :]

    grad_out, h_in_grad = \
        self.nodes[t].backward(x_out_grad, h_in_grad, self.params)

    x_seq_in_grad[:, t, :] = grad_out
```

从中我们看到，`RNNNode`应该有一个`backward`方法，遵循模式，与`forward`方法相反，接收两个形状的数组：

+   `(batch_size, output_size)`

+   `(batch_size, hidden_size)`

并返回两个形状的数组：

+   `(batch_size, feature_size)`

+   `(batch_size, hidden_size)`

这就是`RNNLayer`的工作原理。现在似乎唯一剩下的就是描述递归神经网络的核心：`RNNNode`，在这里实际计算发生。在我们继续之前，让我们澄清`RNNNode`及其在整个RNN中的变体的作用。

## RNNNodes的基本要素

在大多数关于RNN的讨论中，首先讨论的是我们在这里称为`RNNNode`的工作原理。然而，我们最后讨论这些，因为关于RNN最重要的概念是我们在本章中迄今为止描述的那些：数据的结构方式以及数据和隐藏状态在层之间和通过时间如何路由。事实证明，我们可以实现`RNNNode`的多种方式，即给定时间步骤的数据的实际处理和层的隐藏状态的更新。一种方式产生了通常被认为是“常规”递归神经网络的东西，我们在这里将其称为另一个常见术语：香草RNN。然而，还有其他更复杂的方式可以产生不同的RNN变体；例如，一种带有称为GRUs的`RNNNode`的变体，GRUs代表“门控循环单元”。通常，GRUs和其他RNN变体被描述为与香草RNN有显著不同；然而，重要的是要理解*所有* RNN变体都共享我们迄今为止看到的层的结构——例如，它们都以相同的方式向前传递数据，更新它们的隐藏状态(s)在每个时间步。它们之间唯一的区别在于这些“节点”的内部工作方式。

为了强调这一点：如果我们实现了一个`GRULayer`而不是一个`RNNLayer`，代码将完全相同！以下代码仍然构成前向传递的核心：

```py
sequence_length = x_seq_in.shape[1]

x_seq_out = np.zeros((batch_size, sequence_length, self.output_size))

for t in range(sequence_length):

    x_in = x_seq_in[:, t, :]

    y_out, H_in = self.nodes[t].forward(x_in, H_in, self.params)

    x_seq_out[:, t, :] = y_out
```

唯一的区别是`self.nodes`中的每个“节点”将是一个`GRUNode`而不是`RNNNode`。类似地，`backward`方法也将是相同的。

这对于香草RNN的最知名变体——LSTMs或“长短期记忆”单元——也几乎完全正确。唯一的区别在于，这些`LSTMLayer`需要在通过时间向前传递序列元素时“记住”*两个*量，并更新：除了“隐藏状态”外，层中还存储着“细胞状态”，使其能够更好地建模长期依赖关系。这导致了我们在实现`LSTMLayer`与`RNNLayer`时会有一些细微差异；例如，`LSTMLayer`将有两个`ndarray`来存储层在整个时间步中的状态：

+   一个形状为`(1, hidden_size)`的`ndarray` `start_H`，表示层的隐藏状态

+   一个形状为`(1, cell_size)`的`ndarray` `start_C`，表示层的细胞状态

因此，每个`LSTMNode`都应该接收输入，以及隐藏状态和细胞状态。在前向传递中，这将如下所示：

```py
y_out, H_in, C_in = self.nodes[t].forward(x_in, H_in, C_in self.params)
```

以及：

```py
grad_out, h_in_grad, c_in_grad = \
    self.nodes[t].backward(x_out_grad, h_in_grad, c_in_grad, self.params)
```

在`backward`方法中。

这里提到的三种变体远不止这些，其中一些，比如带有“窥视孔连接”的LSTMs，除了隐藏状态外还有一个细胞状态，而另一些只保留隐藏状态。尽管如此，由`LSTMPeepholeConnectionNode`组成的层将与我们迄今为止看到的变体一样适用于`RNNLayer`，因此具有相同的`forward`和`backward`方法。RNN的基本结构——数据如何通过层向前路由，以及如何通过时间步向前路由，然后在反向传递期间沿相反方向路由——这就是使递归神经网络独特的地方。例如，香草RNN和基于LSTM的RNN之间的实际结构差异相对较小，尽管它们的性能可能有显著不同。

有了这些，让我们看看`RNNNode`的实现。

## “香草”RNNNodes

RNN每次接收一个序列元素的数据；例如，如果我们正在预测石油价格，在每个时间步，RNN将接收关于我们用于预测该时间步价格的特征的信息。此外，RNN将在其“隐藏状态”中具有一个编码，表示有关先前时间步发生的事情的累积信息。我们希望将这两个数据片段——时间步的特征和所有先前时间步的累积信息——组合成该时间步的预测以及更新的隐藏状态。

要理解RNN应该如何实现这一点，回想一下在常规神经网络中发生的情况。在前馈神经网络中，每一层接收来自前一层的一组“学习到的特征”，每个特征都是网络“学习到”有用的原始特征的组合。然后该层将这些特征乘以一个权重矩阵，使该层能够学习到作为输入接收的特征的组合特征。为了水平设置和规范化输出，我们向这些新特征添加一个“偏置”，并通过激活函数传递它们。

在递归神经网络中，我们希望我们更新的隐藏状态是输入和旧隐藏状态的组合。因此，类似于常规神经网络中发生的情况：

1.  我们首先将输入和隐藏状态连接起来。然后我们将这个值乘以一个权重矩阵，加上一个偏置，并通过`Tanh`激活函数传递结果。这就是我们更新的隐藏状态。

1.  接下来，我们将这个新的隐藏状态乘以一个权重矩阵，将隐藏状态转换为我们想要的维度的输出。例如，如果我们使用这个RNN来预测每个时间步的单个连续值，我们将把隐藏状态乘以一个大小为`(hidden_size, 1)`的权重矩阵。

因此，我们更新的隐藏状态将是在该时间步接收到的输入以及先前隐藏状态的函数，输出将是通过全连接层的操作将此更新的隐藏状态馈送的结果。

让我们编写代码。

### RNNNode：代码

以下代码实现了刚才描述的步骤。请注意，正如我们稍后将对GRUs和LSTMs做的一样（以及我们在[第1章](ch01.html#foundations)中展示的简单数学函数），我们将在`Node`中存储所有在前向传播中计算的量作为属性，以便我们可以使用它们来计算反向传播：

```py
def forward(self,
            x_in: ndarray,
            H_in: ndarray,
            params_dict: Dict[str, Dict[str, ndarray]]
            ) -> Tuple[ndarray]:
    '''
 param x: numpy array of shape (batch_size, vocab_size)
 param H_prev: numpy array of shape (batch_size, hidden_size)
 return self.x_out: numpy array of shape (batch_size, vocab_size)
 return self.H: numpy array of shape (batch_size, hidden_size)
 '''
    self.X_in = x_in
    self.H_in = H_in

    self.Z = np.column_stack((x_in, H_in))

    self.H_int = np.dot(self.Z, params_dict['W_f']['value']) \
                                + params_dict['B_f']['value']

    self.H_out = tanh(self.H_int)

    self.X_out = np.dot(self.H_out, params_dict['W_v']['value']) \
                                    + params_dict['B_v']['value']

    return self.X_out, self.H_out
```

另一个注意事项：由于我们这里没有使用`ParamOperation`，我们需要以不同的方式存储参数。我们将把它们存储在一个名为`params_dict`的字典中，通过名称引用参数。此外，每个参数将有两个键：`value`和`deriv`，分别存储实际参数值和它们关联的梯度。在前向传播中，我们只使用`value`键。

### RNNNodes：反向传播

通过`RNNNode`的反向传播简单地计算损失相对于`RNNNode`输入的梯度值，给定损失相对于`RNNNode`输出的梯度。我们可以使用类似于我们在[第1章](ch01.html#foundations)和[第2章](ch02.html#fundamentals)中解决的逻辑来做到这一点：由于我们可以将`RNNNode`表示为一系列操作，我们可以简单地计算每个操作在其输入处的导数，并将这些导数与之前的导数逐个相乘（注意正确处理矩阵乘法），最终得到表示损失相对于每个输入的梯度的`ndarray`。以下代码实现了这一点：

```py
def forward(self,
            x_in: ndarray,
            H_in: ndarray,
            params_dict: Dict[str, Dict[str, ndarray]]
            ) -> Tuple[ndarray]:
    '''
 param x: numpy array of shape (batch_size, vocab_size)
 param H_prev: numpy array of shape (batch_size, hidden_size)
 return self.x_out: numpy array of shape (batch_size, vocab_size)
 return self.H: numpy array of shape (batch_size, hidden_size)
 '''
    self.X_in = x_in
    self.H_in = H_in

    self.Z = np.column_stack((x_in, H_in))

    self.H_int = np.dot(self.Z, params_dict['W_f']['value']) \
                                + params_dict['B_f']['value']

    self.H_out = tanh(self.H_int)

    self.X_out = np.dot(self.H_out, params_dict['W_v']['value']) \
                                    + params_dict['B_v']['value']

    return self.X_out, self.H_out
```

请注意，就像我们之前的`Operation`s一样，`backward`函数的输入形状必须与`forward`函数的输出形状匹配，`backward`函数的输出形状必须与`forward`函数的输入形状匹配。

## “Vanilla” RNNNodes的局限性

记住：RNNs的目的是对数据序列中的依赖关系进行建模。以模拟石油价格为例，这意味着我们应该能够揭示我们在过去几个时间步中看到的特征序列与明天石油价格会发生什么之间的关系。但“几个”应该是多长时间呢？对于石油价格，我们可能会想象，昨天发生的事情——前一时间步——对于预测明天的石油价格最为重要，前一天的重要性较小，而随着时间的倒退，重要性通常会逐渐减弱。

虽然这对许多现实世界的问题是正确的，但有些领域我们希望应用RNNs，我们希望学习极端长期的依赖关系。*语言建模*是一个典型的例子，即构建一个模型，可以预测下一个字符、单词或单词部分，给定一个理论上极长的过去单词或字符序列（因为这是一个特别普遍的应用，我们将在本章后面讨论一些与语言建模相关的细节）。对于这一点，vanilla RNNs通常是不够的。现在我们已经看到了它们的细节，我们可以理解为什么：在每个时间步，隐藏状态都会被同一组权重矩阵乘以*所有层中的所有时间步。考虑当我们一遍又一遍地将一个数字乘以一个值`x`时会发生什么：如果`x < 1`，数字会指数级地减少到0，如果`x > 1`，数字会指数级地增加到无穷大。循环神经网络也有同样的问题：在长时间跨度上，因为在每个时间步中隐藏状态都会被相同的权重集乘以，这些权重的梯度往往会变得极小或极大。前者被称为*消失梯度问题*，后者被称为*爆炸梯度问题*。这两个问题使得训练RNNs来模拟非常长期的依赖关系（50-100个时间步）变得困难。我们接下来将介绍的两种常用的修改vanilla RNN架构的方法都显著缓解了这个问题。

## 一种解决方案：GRUNodes

Vanilla RNNs可以被描述为将输入和隐藏状态结合在一起，并使用矩阵乘法来确定如何“加权”隐藏状态中包含的信息与新输入中的信息，以预测输出。激发更高级RNN变体的洞察力是，为了模拟长期依赖关系，比如语言中存在的依赖关系，*有时我们会收到告诉我们需要“忘记”或“重置”隐藏状态的信息*。一个简单的例子是句号“。”或冒号“:”——如果语言模型收到其中一个，它就知道应该忘记之前的字符，并开始对字符序列中的新模式进行建模。

第一个简单的变体是GRUs或门控循环单元，利用了这一洞察力，因为输入和先前的隐藏状态通过一系列“门”传递。

1.  第一个门类似于在vanilla RNNs中发生的操作：输入和隐藏状态被连接在一起，乘以一个权重矩阵，然后通过`sigmoid`操作传递。我们可以将其输出视为“更新”门。

1.  第二个门被解释为“重置”门：输入和隐藏状态被连接在一起，乘以一个权重矩阵，通过`sigmoid`操作，*然后乘以先前的隐藏状态*。这使得网络能够“学会忘记”隐藏状态中的内容，给定传入的特定输入。

1.  然后，第二个门的输出乘以另一个矩阵，并通过`Tanh`函数传递，输出为新隐藏状态的“候选”。

1.  最后，隐藏状态更新为更新门乘以新隐藏状态的“候选”，再加上旧隐藏状态乘以1减去更新门。

###### 注意

在本章中，我们将介绍普通RNN的两个高级变体：GRUs和LSTMs。LSTMs更受欢迎，比GRUs早发明很久。尽管如此，GRUs是LSTMs的一个更简单的版本，并更直接地说明了“门”的概念如何使RNN能够“学会重置”其隐藏状态，这就是为什么我们首先介绍它们。

### GRUNodes：一个图表

[图6-8](#fig_06_08)将`GRUNode`描述为一系列门。每个门包含一个`Dense`层的操作：乘以一个权重矩阵，加上一个偏置，并通过激活函数传递结果。使用的激活函数要么是`sigmoid`，在这种情况下，结果的范围在0到1之间，要么是`Tanh`，在这种情况下，范围在-1到1之间；下一个产生的每个中间`ndarray`的范围在数组的名称下显示。

![长度为一的输出](assets/dlfs_0608.png)

###### 图6-8。数据通过GRUNode向前流动，通过门并产生X_out和H_out

在[图6-8](#fig_06_08)中，以及图[6-9](#fig_06_09)和[6-10](#fig_06_10)中，节点的输入为绿色，计算的中间量为蓝色，输出为红色。所有权重（未直接显示）都包含在门中。

请注意，要通过这个过程反向传播，我们必须将其表示为一系列`Operation`，计算每个`Operation`相对于其输入的导数，并将结果相乘。我们在这里没有明确展示这一点，而是将门（实际上是三个操作的组合）显示为一个单独的块。但是，到目前为止，我们知道如何通过组成每个门的`Operation`进行反向传播，而“门”的概念在循环神经网络及其变体的描述中被广泛使用，因此我们将在这里坚持使用这种表示方式。

实际上，[图6-9](#fig_06_09)显示了一个使用门的普通`RNNNode`的表示。

![长度为一的输出](assets/dlfs_0609.png)

###### 图6-9。数据通过RNNNode向前流动，通过两个门并产生X_out和H_out

因此，另一种思考我们之前描述的作为一个普通`RNNNode`组成部分的`Operation`的方式是将输入和隐藏状态通过两个门传递。

### GRUNodes：代码

以下代码实现了先前描述的`GRUNode`的前向传递：

```py
def forward(self,
            X_in: ndarray,
            H_in: ndarray,
            params_dict: Dict[str, Dict[str, ndarray]]) -> Tuple[ndarray]:
    '''
 param X_in: numpy array of shape (batch_size, vocab_size)
 param H_in: numpy array of shape (batch_size, hidden_size)
 return self.X_out: numpy array of shape (batch_size, vocab_size)
 return self.H_out: numpy array of shape (batch_size, hidden_size)
 '''
    self.X_in = X_in
    self.H_in = H_in

    # reset gate
    self.X_r = np.dot(X_in, params_dict['W_xr']['value'])
    self.H_r = np.dot(H_in, params_dict['W_hr']['value'])

    # update gate
    self.X_u = np.dot(X_in, params_dict['W_xu']['value'])
    self.H_u = np.dot(H_in, params_dict['W_hu']['value'])

    # gates
    self.r_int = self.X_r + self.H_r + params_dict['B_r']['value']
    self.r = sigmoid(self.r_int)

    self.u_int = self.X_r + self.H_r + params_dict['B_u']['value']
    self.u = sigmoid(self.u_int)

    # new state
    self.h_reset = self.r * H_in
    self.X_h = np.dot(X_in, params_dict['W_xh']['value'])
    self.H_h = np.dot(self.h_reset, params_dict['W_hh']['value'])
    self.h_bar_int = self.X_h + self.H_h + params_dict['B_h']['value']
    self.h_bar = np.tanh(self.h_bar_int)

    self.H_out = self.u * self.H_in + (1 - self.u) * self.h_bar

    self.X_out = (
  np.dot(self.H_out, params_dict['W_v']['value']) \
  + params_dict['B_v']['value']
  )

    return self.X_out, self.H_out
```

请注意，我们没有明确连接`X_in`和`H_in`，因为—与`RNNNode`不同，在`GRUNode`中我们独立使用它们；具体来说，在`self.h_reset = self.r * H_in`这一行中，我们独立使用`H_in`而不是`X_in`。

`backward`方法可以在[书的网站](https://oreil.ly/2P0lG1G)上找到；它只是通过组成`GRUNode`的操作向后步进，计算每个操作相对于其输入的导数，并将结果相乘。

## LSTMNodes

长短期记忆单元，或LSTMs，是香草RNN单元最受欢迎的变体。部分原因是它们是在深度学习的早期阶段，即1997年发明的^([6](ch06.html#idm45732612415128))，而对于LSTM替代方案如GRUs的调查在过去几年中才加速进行（例如，GRUs是在2014年提出的）。

与GRUs一样，LSTMs的动机是为了让RNN能够在接收新输入时“重置”或“忘记”其隐藏状态。在GRUs中，通过将输入和隐藏状态通过一系列门传递，以及使用这些门计算“建议”的新隐藏状态—`self.h_bar`，使用门`self.r`计算—然后使用建议的新隐藏状态和旧隐藏状态的加权平均值计算最终隐藏状态，由更新门控制：

```py
self.H_out = self.u * self.H_in + (1 - self.u) * self.h_bar
```

相比之下，LSTMs*使用一个单独的“状态”向量，“单元状态”，来确定是否“忘记”隐藏状态中的内容*。然后，它们使用另外两个门来控制它们应该重置或更新*单元状态*中的内容的程度，以及第四个门来确定基于最终单元状态的情况下隐藏状态的更新程度。^([7](ch06.html#idm45732612392904))

### LSTMNodes: Diagram

[图6-10](#fig_06_10)显示了一个`LSTMNode`的图示，其中操作表示为门。

![](assets/dlfs_0610.png)

###### 图6-10\. 数据通过LSTMNode向前流动的流程，通过一系列门传递，并输出更新的单元状态和隐藏状态C_out和H_out，以及实际输出X_out

### LSTMs: The code

与`GRUNode`一样，`LSTMNode`的完整代码，包括`backward`方法和一个示例，展示了这些节点如何适应`LSTMLayer`，都包含在[书的网站](https://oreil.ly/2P0lG1G)上。在这里，我们只展示`forward`方法：

```py
def forward(self,
  X_in: ndarray,
  H_in: ndarray,
  C_in: ndarray,
  params_dict: Dict[str, Dict[str, ndarray]]):
  '''
 param X_in: numpy array of shape (batch_size, vocab_size)
 param H_in: numpy array of shape (batch_size, hidden_size)
 param C_in: numpy array of shape (batch_size, hidden_size)
 return self.X_out: numpy array of shape (batch_size, output_size)
 return self.H: numpy array of shape (batch_size, hidden_size)
 return self.C: numpy array of shape (batch_size, hidden_size)
 '''

  self.X_in = X_in
  self.C_in = C_in

  self.Z = np.column_stack((X_in, H_in))
  self.f_int = (
    np.dot(self.Z, params_dict['W_f']['value']) \
    + params_dict['B_f']['value']
    )
  self.f = sigmoid(self.f_int)

  self.i_int = (
    np.dot(self.Z, params_dict['W_i']['value']) \
    + params_dict['B_i']['value']
    )
  self.i = sigmoid(self.i_int)

  self.C_bar_int = (
    np.dot(self.Z, params_dict['W_c']['value']) \
    + params_dict['B_c']['value']
    )
  self.C_bar = tanh(self.C_bar_int)
  self.C_out = self.f * C_in + self.i * self.C_bar

  self.o_int = (
    np.dot(self.Z, params_dict['W_o']['value']) \
    + params_dict['B_o']['value']
    )
  self.o = sigmoid(self.o_int)
  self.H_out = self.o * tanh(self.C_out)

  self.X_out = (
    np.dot(self.H_out, params_dict['W_v']['value']) \
    + params_dict['B_v']['value']
    )

  return self.X_out, self.H_out, self.C_out
```

这是我们需要开始训练模型的RNN框架的最后一个元素！我们应该涵盖的另一个主题是：如何以一种形式表示文本数据，以便我们可以将其馈送到我们的RNN中。

## 用于基于字符级RNN的语言模型的数据表示

语言建模是RNN最常用的任务之一。我们如何将字符序列重塑为训练数据集，以便RNN可以训练以预测下一个字符？最简单的方法是使用*one-hot编码*。具体操作如下：首先，每个字母都表示为一个维度等于*词汇表的大小*或文本总体语料库中字母数量的向量（这是在网络中预先计算并硬编码为一个超参数）。然后，每个字母都表示为一个向量，其中该字母所在位置为1，其他位置为0。最后，每个字母的向量简单地连接在一起，以获得字母序列的整体表示。

这是一个简单的示例，展示了具有四个字母`a`、`b`、`c`和`d`的词汇表，我们任意地将`a`称为第一个字母，`b`称为第二个字母，依此类推：

<math display="block"><mrow><mi>a</mi> <mi>b</mi> <mi>c</mi> <mi>d</mi> <mi>b</mi> <mo>→</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mtd> <mtd><mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mtd> <mtd><mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mtd> <mtd><mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd></mtr></mtable></mfenced></mtd> <mtd><mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mtd></mtr></mtable></mfenced> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mrow></math>

这个二维数组将代替一个形状为`(sequence_length, num_features) = (5, 4)`的观察值在一个序列批次中。因此，如果我们的文本是“abcdba”—长度为6—并且我们想要将长度为5的序列馈送到我们的数组中，第一个序列将被转换为前述矩阵，第二个序列将是：

<math display="block"><mrow><mi>b</mi> <mi>c</mi> <mi>d</mi> <mi>b</mi> <mi>a</mi> <mo>→</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mtd> <mtd><mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mtd> <mtd><mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd></mtr></mtable></mfenced></mtd> <mtd><mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mtd> <mtd><mfenced close="]" open="["><mtable><mtr><mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mtd></mtr></mtable></mfenced> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced></mrow></math>

然后将它们连接在一起，以创建一个形状为`(batch_size, sequence_length, vocab_size) = (2, 5, 4)`的RNN输入。继续这样，我们可以将原始文本转换为一批序列，以馈送到RNN中。

在书的GitHub存储库中的[第6章笔记本](https://oreil.ly/2P0lG1G)中，我们将其编码为`RNNTrainer`类的一部分，该类可以接受原始文本，使用这里描述的技术对其进行预处理，并将其批量馈送到RNN中。

## 其他语言建模任务

我们在本章中没有强调这一点，但正如您从前面的代码中看到的，所有`RNNNode`变体都允许`RNNLayer`输出与其接收的特征数量不同的特征。所有三个节点的最后一步是将网络的最终隐藏状态乘以我们通过`params_dict[*W_v*]`访问的权重矩阵；这个权重矩阵的第二维将决定`Layer`的输出维度。这使我们可以通过在每个`Layer`中更改一个`output_size`参数来简单地为不同的语言建模任务使用相同的架构。

例如，到目前为止，我们只考虑通过“下一个字符预测”构建语言模型；在这种情况下，我们的输出大小将等于词汇表的大小：`output_size = vocab_size`。然而，对于情感分析之类的任务，我们传入的序列可能只有一个标签“0”或“1”——积极或消极。在这种情况下，我们不仅将有`output_size = 1`，而且只有在传入整个序列后才将输出与目标进行比较。这将看起来像[图6-11](#fig_06_11)中所示。

![长度为一的输出](assets/dlfs_0611.png)

###### 图6-11。对于情感分析，RNN将将其预测与实际值进行比较，并仅为最后一个序列元素的输出产生梯度；然后反向传播将继续进行，每个不是最后一个节点的节点将简单地接收一个全零的“X_grad_out”数组

因此，这个框架可以轻松适应不同的语言建模任务；实际上，它可以适应任何数据是顺序的并且可以逐个序列元素馈送到网络中的建模任务。

在结束之前，我们将讨论RNN中很少讨论的一个方面：这些不同类型的层——`GRULayer`、`LSTMLayer`和其他变体——可以混合和匹配。

## 组合RNNLayer变体

堆叠不同类型的`RNNLayer`非常简单：每个RNN输出一个形状为`(batch_size, sequence_length, output_size)`的`ndarray`，可以被馈送到下一层。就像在`Dense`层中一样，我们不需要指定`input_shape`；我们只需根据层接收的第一个`ndarray`设置权重，使其具有适当的形状。一个`RNNModel`可以具有一个`self.layers`属性：

```py
[RNNLayer(hidden_size=256, output_size=128),
RNNLayer(hidden_size=256, output_size=62)]
```

与我们的全连接神经网络一样，我们只需要确保最后一层产生所需维度的输出；在这里，如果我们处理的词汇量为62并进行下一个字符预测，我们的最后一层必须具有62的`output_size`，就像我们处理MNIST问题的全连接神经网络中的最后一层必须具有维度10一样。

在阅读本章后应该清楚的一点，但在处理RNN时通常不经常涉及的是，因为我们看到的每种类型的层都具有相同的基础结构，接收维度为`feature_size`的序列并输出维度为`output_size`的序列，我们可以轻松地堆叠不同类型的层。例如，在[书籍网站](https://oreil.ly/2P0lG1G)上，我们训练一个具有`self.layers`属性的`RNNModel`：

```py
[GRULayer(hidden_size=256, output_size=128),
LSTMLayer(hidden_size=256, output_size=62)]
```

换句话说，第一层通过使用`GRUNode`将其输入向前传递一段时间，然后将形状为`(batch_size, sequence_length, 128)`的`ndarray`传递到下一层，随后通过其`LSTMNode`将它们传递。

## 将所有内容整合在一起

一个经典的练习是训练RNN以特定风格写文本；[在书的网站上](https://oreil.ly/2P0lG1G)，我们有一个端到端的代码示例，使用本章描述的抽象定义模型，学习以莎士比亚风格写文本。我们还没有展示的唯一组件是一个`RNNTrainer`类，它通过训练数据进行迭代，对其进行预处理，并将其馈送到模型中。这与我们之前看到的`Trainer`的主要区别是，对于RNN，一旦我们选择要馈送的数据批次——每个批次元素仅为一个字符串——我们必须首先对其进行预处理，对每个字母进行独热编码，并将生成的向量连接成一个序列，将长度为`sequence_length`的每个字符串转换为形状为`(sequence_length, vocab_size)`的`ndarray`。为了形成将馈送到我们的RNN中的批次，*这些* `ndarray`将被连接在一起，形成大小为`(sequence_length, vocab_size, batch_size)`的批次。

但是一旦数据经过预处理并且模型定义好了，RNN的训练方式与我们之前看到的其他神经网络相同：批次被迭代地馈送，模型的预测与目标进行比较以生成损失，损失通过构成模型的操作进行反向传播以更新权重。

# 结论

在本章中，您了解了递归神经网络，这是一种专门设计用于处理数据序列而不是单个操作的神经网络架构。您了解了RNN由在时间上向前传递数据的层组成，随着时间的推移更新它们的隐藏状态（以及在LSTMs的情况下更新它们的单元状态）。您看到了高级RNN变体GRUs和LSTMs的细节，以及它们如何通过每个时间步的一系列“门”向前传递数据；然而，您了解到这些高级变体基本上以相同的方式处理数据序列，因此它们的层结构相同，在每个时间步应用的特定操作不同。

希望这个多方面的主题现在不再是一个黑匣子。在[第7章](ch07.html#pytorch)中，我将通过转向深度学习的实践方面来结束本书，展示如何使用PyTorch框架实现我们迄今所讨论的一切，PyTorch是一个高性能、基于自动微分的框架，用于构建和训练深度学习模型。继续前进！

^([1](ch06.html#idm45732615342296-marker)) 我们碰巧发现将观察结果排列在行上，将特征排列在列上很方便，但我们不一定要以这种方式排列数据。然而，数据必须是二维的。

^([2](ch06.html#idm45732615335848-marker)) 或者至少是这本书的这个版本。

^([3](ch06.html#idm45732614959096-marker)) 我想提到作者Daniel Sabinasz在他的博客[*deep ideas*](http://www.deepideas.net)上分享的另一种解决这个问题的方法：他将操作表示为一个图，然后使用广度优先搜索来计算反向传播中的梯度，以正确的顺序构建一个模仿TensorFlow的框架。他关于如何做到这一点的博客文章非常清晰和结构良好。

^([4](ch06.html#idm45732615077368-marker)) 深入了解如何实现自动微分，请参阅Andrew Trask的*Grokking Deep Learning*（Manning）。

^([5](ch06.html#idm45732613324856-marker)) 请查看LSTMs的维基百科页面，了解更多[LSTM变体](https://oreil.ly/2TysrXj)的例子。

^([6](ch06.html#idm45732612415128-marker)) 参见Hochreiter等人的原始LSTM论文[“长短期记忆”](https://oreil.ly/2YYZvwT) (1997)。

至少是标准变体的LSTMs；正如提到的，还有其他变体，比如“带有窥视孔连接的LSTMs”，其门的排列方式不同。
