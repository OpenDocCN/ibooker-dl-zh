# 第3章。从头开始的深度学习

您可能没有意识到，但现在您已经具备回答本书开头提出的关于深度学习模型的关键问题的所有数学和概念基础：您了解*神经网络*是如何工作的——涉及矩阵乘法、损失和相对于该损失的偏导数的计算，以及这些计算为什么有效（即微积分中的链式法则）。通过从第一原理构建神经网络，将它们表示为一系列“构建块”，我们实现了这种理解。在本章中，您将学习将这些构建块本身表示为抽象的Python类，然后使用这些类构建深度学习模型；到本章结束时，您确实将完成“从头开始的深度学习”！

我们还将描述神经网络的描述与您之前可能听过的深度学习模型的更传统描述相匹配。例如，在本章结束时，您将了解深度学习模型具有“多个隐藏层”是什么意思。这实际上是理解一个概念的本质：能够在高级描述和实际发生的低级细节之间进行翻译。让我们开始朝着这个翻译的方向构建。到目前为止，我们只是根据低级别发生的操作来描述模型。在本章的第一部分中，我们将将模型的这种描述映射到常见的更高级概念，例如“层”，最终使我们能够更轻松地描述更复杂的模型。

# 深度学习定义：第一次尝试

“深度学习”模型是什么？在前一章中，我们将模型定义为由计算图表示的数学函数。这样的模型的目的是尝试将输入映射到输出，每个输入都来自具有共同特征的数据集（例如，表示房屋不同特征的单独输入），输出来自相关分布（例如，这些房屋的价格）。我们发现，如果我们将模型定义为一个包括*参数*作为某些操作的输入的函数，我们可以通过以下过程“拟合”它以最佳地描述数据：

1.  反复通过模型传递观察数据，跟踪在这个“前向传递”过程中计算的量。

1.  计算代表我们模型预测与期望输出或目标之间差距有多大的*损失*。

1.  使用在“前向传递”中计算的量和在[第1章](ch01.html#foundations)中推导出的链式法则，计算每个输入*参数*最终对这个损失产生了多大影响。

1.  更新参数的值，以便在下一组观察通过模型时，损失有望减少。

我们最初使用的模型只包含一系列线性操作，将我们的特征转换为目标（结果等同于传统的线性回归模型）。这带来了一个预期的限制，即即使在“最佳拟合”时，模型仍然只能表示特征和目标之间的线性关系。

然后我们定义了一个函数结构，首先应用这些线性操作，然后是一个*非*线性操作（`sigmoid`函数），最后是一组线性操作。我们展示了通过这种修改，我们的模型*可以*学习更接近输入和输出之间真实的非线性关系，同时还具有额外的好处，即它可以学习输入特征和目标之间的*组合*关系。

这些模型与深度学习模型之间的联系是什么？我们将从一个有些笨拙的定义开始：深度学习模型由一系列操作表示，其中至少涉及两个非连续的非线性函数。

我将很快展示这个定义的来源，但首先要注意的是，由于深度学习模型只是一系列操作，训练它们的过程实际上与我们已经看到的简单模型的训练过程是*相同*的。毕竟，使得这个训练过程能够工作的是模型相对于其输入的可微性；正如在[第1章](ch01.html#foundations)中提到的，可微函数的组合是可微的，因此只要组成函数的各个操作是可微的，整个函数就是可微的，我们就能够使用刚刚描述的相同的四步训练过程来训练它。

然而，到目前为止，我们实际上训练这些模型的方法是通过手动编写前向和后向传递来计算这些导数，然后将适当的量相乘以获得导数。对于[第2章](ch02.html#fundamentals)中的简单神经网络模型，这需要17个步骤。由于我们在如此低的层次上描述模型，不清楚如何向这个模型添加更多复杂性（或者这到底意味着什么），甚至进行简单的更改，比如将另一个非线性函数替换为sigmoid函数。为了能够构建任意“深度”和其他“复杂”的深度学习模型，我们必须考虑在这17个步骤中哪里可以创建可重用的组件，比单个操作的层次更高，可以替换并构建不同的模型。为了指导我们创建哪些抽象，我们将尝试将我们一直在使用的操作映射到传统的神经网络描述，即由“层”、“神经元”等组成。

作为第一步，我们将不再重复编写相同的矩阵乘法和偏置添加，而是创建一个抽象来表示我们目前所使用的各个操作。

# 神经网络的构建模块：操作

`Operation`类将代表我们神经网络中的一个组成函数。根据我们在模型中使用这些函数的方式，从高层次来看，它应该有`forward`和`backward`方法，每个方法接收一个`ndarray`作为输入并输出一个`ndarray`。一些操作，比如矩阵乘法，似乎有*另一种*特殊类型的输入，也是一个`ndarray`：参数。在我们的`Operation`类中——或者可能是在另一个继承自它的类中——我们应该允许`params`作为另一个实例变量。

另一个观点是，似乎有两种类型的`Operation`：一些，比如矩阵乘法，返回一个与其输入不同形状的`ndarray`作为输出；相比之下，一些`Operation`，比如`sigmoid`函数，只是将某个函数应用于输入`ndarray`的每个元素。那么，关于在我们的操作之间传递的`ndarray`的形状，有什么“一般规则”呢？让我们考虑通过我们的`Operation`传递的`ndarray`：每个`Operation`将在前向传递中向前发送输出，并在后向传递中接收一个“输出梯度”，这将代表`Operation`输出的每个元素相对于损失的偏导数（由组成网络的其他`Operation`计算）。此外，在后向传递中，每个`Operation`将向后发送一个“输入梯度”，表示相对于输入的每个元素的损失的偏导数。

这些事实对我们的`Operation`的工作方式施加了一些重要的限制，这将帮助我们确保我们正确计算梯度：

+   *输出梯度* `ndarray`的形状必须与*输出*的形状匹配。

+   `Operation`在反向传递期间发送的*输入梯度*的形状必须与`Operation`的*输入*的形状匹配。

一旦您在图表中看到这一切，一切都会更清晰；让我们接着看。

## 图表

这一切都总结在[图3-1](#fig_03-01)中，对于一个操作`O`，它从操作`N`接收输入，然后将输出传递给另一个操作`P`。

![神经网络图](assets/dlfs_0301.png)

###### 图3-1\. 一个带有输入和输出的Operation

[图3-2](#fig_03-02)涵盖了带有参数的`Operation`的情况。

![神经网络图](assets/dlfs_0302.png)

###### 图3-2\. 一个带有输入、输出和参数的ParamOperation

## 代码

有了这一切，我们可以将我们的神经网络的基本构建模块，即`Operation`，写成：

```py
class Operation(object):
    '''
 Base class for an "operation" in a neural network.
 '''
    def __init__(self):
        pass

    def forward(self, input_: ndarray):
        '''
 Stores input in the self._input instance variable
 Calls the self._output() function.
 '''
        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
 Calls the self._input_grad() function.
 Checks that the appropriate shapes match.
 '''
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> ndarray:
        '''
 The _output method must be defined for each Operation.
 '''
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
 The _input_grad method must be defined for each Operation.
 '''
        raise NotImplementedError()
```

对于我们定义的任何单个`Operation`，我们将不得不实现`_output`和`_input_grad`函数，这两个函数的名称是因为它们计算的数量。

###### 注意

我们主要为了教学目的而定义这样的基类：重要的是要有这样的心智模型，即*所有*您在深度学习中遇到的`Operation`都符合这种向前发送输入和向后发送梯度的蓝图，前向传递时接收的形状与反向传递时发送的形状匹配，反之亦然。

我们将在本章后面定义迄今为止使用的特定`Operation`，如矩阵乘法等。首先，我们将定义另一个从`Operation`继承的类，专门用于涉及参数的`Operation`：

```py
class ParamOperation(Operation):
    '''
 An Operation with parameters.
 '''

    def __init__(self, param: ndarray) -> ndarray:
        '''
 The ParamOperation method
 '''
        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
 Calls self._input_grad and self._param_grad.
 Checks appropriate shapes.
 '''

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
 Every subclass of ParamOperation must implement _param_grad.
 '''
        raise NotImplementedError()
```

与基本`Operation`类似，一个单独的`ParamOperation`还必须定义`_param_grad`函数，除了`_output`和`_input_grad`函数。

我们现在已经正式化了迄今为止我们在模型中使用的神经网络构建模块。我们可以直接根据这些`Operation`定义神经网络，但是有一个我们已经绕了一个半章的中间类，我们将首先定义它：`Layer`。

# 神经网络的构建模块：层

就`Operation`而言，层是一系列线性操作后跟一个非线性操作。例如，我们上一章的神经网络可以说有五个总操作：两个线性操作——权重乘法和添加偏置项——跟随`sigmoid`函数，然后又两个线性操作。在这种情况下，我们会说前三个操作，包括非线性操作在内，构成第一层，而最后两个操作构成第二层。此外，我们说输入本身代表一种特殊类型的层，称为*输入*层（在编号层次上，这一层不计算，因此我们可以将其视为“零”层）。同样地，最后一层称为*输出*层。中间层——根据我们的编号，“第一个”——也有一个重要的名称：它被称为*隐藏*层，因为它是唯一一个在训练过程中我们通常不明确看到其值的层。

输出层是对层的这一定义的一个重要例外，因为它不一定*必须*对其应用非线性操作；这仅仅是因为我们通常希望这一层的输出值在负无穷到正无穷之间（或至少在0到正无穷之间），而非线性函数通常会将其输入“压缩”到与我们尝试解决的特定问题相关的该范围的某个子集（例如，`sigmoid`函数将其输入压缩到0到1之间）。

## 图表

为了明确连接，[图3-3](#fig_03-03)显示了前一章中的神经网络的图表，其中将单独的操作分组到层中。

![神经网络图表](assets/dlfs_0303.png)

###### 图3-3。前一章中的神经网络，操作分组成层

您可以看到输入表示“输入”层，接下来的三个操作（以`sigmoid`函数结束）表示下一层，最后两个操作表示最后一层。

当然，这相当繁琐。这就是问题所在：将神经网络表示为一系列单独的操作，同时清楚地显示神经网络的工作原理以及如何训练它们，对于比两层神经网络更复杂的任何东西来说都太“低级”。这就是为什么更常见的表示神经网络的方式是以层为单位，如[图3-4](#fig_03-04)所示。

![神经网络图表](assets/dlfs_0304.png)

###### 图3-4。以层为单位的前一章中的神经网络

### 与大脑的连接

最后，让我们将我们迄今所见的内容与您可能之前听过的概念之间建立最后一个连接：每个层可以说具有等于*表示该层输出中每个观察的向量的维度*的*神经元*数量。因此，前一个示例中的神经网络可以被认为在输入层有13个神经元，然后在隐藏层中有13个神经元（再次），在输出层中有一个神经元。

大脑中的神经元具有这样的特性，它们可以从许多其他神经元接收输入，只有当它们累积接收到的信号达到一定的“激活能量”时，它们才会“发射”并向前发送信号。神经网络的神经元具有类似的属性：它们确实根据其输入向前发送信号，但是输入仅通过非线性函数转换为输出。因此，这个非线性函数被称为*激活函数*，从中出来的值被称为该层的*激活*。^([1](ch03.html#idm45732624417528))

现在我们已经定义了层，我们可以陈述更传统的深度学习定义：*深度学习模型是具有多个隐藏层的神经网络。*

我们可以看到，这等同于早期纯粹基于“操作”定义的定义，因为层只是一系列具有非线性操作的“操作”，最后是一个非线性操作。

现在我们已经为我们的“操作”定义了一个基类，让我们展示它如何可以作为我们在前一章中看到的模型的基本构建模块。

# 构建模块上的构建模块

我们需要为前一章中的模型实现哪些特定的“操作”？根据我们逐步实现神经网络的经验，我们知道有三种：

+   输入与参数矩阵的矩阵乘法

+   添加偏置项

+   `sigmoid`激活函数

让我们从`WeightMultiply`“操作”开始：

```py
class WeightMultiply(ParamOperation):
    '''
 Weight multiplication operation for a neural network.
 '''

    def __init__(self, W: ndarray):
        '''
 Initialize Operation with self.param = W.
 '''
        super().__init__(W)

    def _output(self) -> ndarray:
        '''
 Compute output.
 '''
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
 Compute input gradient.
 '''
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray)  -> ndarray:
        '''
 Compute parameter gradient.
 '''
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)
```

在前向传递中，我们简单地编码矩阵乘法，以及在反向传递中“向输入和参数发送梯度”的规则（使用我们在[第1章](ch01.html#foundations)末尾推理出的规则）。很快您将看到，我们现在可以将其用作我们可以简单插入到我们的“层”中的*构建模块*。

接下来是加法操作，我们将其称为`BiasAdd`：

```py
class BiasAdd(ParamOperation):
    '''
 Compute bias addition.
 '''

    def __init__(self,
                 B: ndarray):
        '''
 Initialize Operation with self.param = B.
 Check appropriate shape.
 '''
        assert B.shape[0] == 1

        super().__init__(B)

    def _output(self) -> ndarray:
        '''
 Compute output.
 '''
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
 Compute input gradient.
 '''
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
 Compute parameter gradient.
 '''
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
```

最后，让我们做`sigmoid`：

```py
class Sigmoid(Operation):
    '''
 Sigmoid activation function.
 '''

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> ndarray:
        '''
 Compute output.
 '''
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
 Compute input gradient.
 '''
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad
```

这只是实现了前一章描述的数学。

###### 注意

对于`sigmoid`和`ParamOperation`，在反向传播期间计算的步骤是：

```py
input_grad = <something> * output_grad
```

是我们应用链规则的步骤，以及`WeightMultiply`的相应规则：

```py
np.dot(output_grad, np.transpose(self.param, (1, 0)))
```

正如我在[第1章](ch01.html#foundations)中所说的，当涉及的函数是矩阵乘法时，这相当于链规则的类比。

现在我们已经准确定义了这些`Operation`，我们可以将它们用作定义`Layer`的构建块。

## 层蓝图

由于我们编写了`Operation`的方式，编写`Layer`类很容易：

+   `forward`和`backward`方法只涉及将输入依次通过一系列`Operation`向前传递 - 就像我们一直在图表中所做的那样！这是关于`Layer`工作最重要的事实；代码的其余部分是围绕这一点的包装，并且主要涉及簿记：

    +   在`_setup_layer`函数中定义正确的`Operation`系列，并在这些`Operation`中初始化和存储参数（这也将在`_setup_layer`函数中进行）

    +   在`forward`方法中存储正确的值在`self.input_`和`self.output`中

    +   在`backward`方法中执行正确的断言检查

+   最后，`_params`和`_param_grads`函数只是从层内的`ParamOperation`中提取参数及其梯度（相对于损失）。

所有这些看起来是这样的：

```py
class Layer(object):
    '''
 A "layer" of neurons in a neural network.
 '''

 def __init__(self,
                 neurons: int):
        '''
 The number of "neurons" roughly corresponds to the "breadth" of the
 layer
 '''
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int) -> None:
        '''
 The _setup_layer function must be implemented for each layer.
 '''
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:
        '''
 Passes input forward through a series of operations.
 '''
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:

            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
 Passes output_grad backward through a series of operations.
 Checks appropriate shapes.
 '''

        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self) -> ndarray:
        '''
 Extracts the _param_grads from a layer's operations.
 '''

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> ndarray:
        '''
 Extracts the _params from a layer's operations.
 '''

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)
```

就像我们从抽象定义的`Operation`转向实现神经网络所需的特定`Operation`一样，让我们现在也实现该网络中的`Layer`。

## 密集层

我们称我们一直在处理的`Operation`为`WeightMultiply`，`BiasAdd`等等。到目前为止我们一直在使用的层应该叫什么？`LinearNonLinear`层？

这一层的一个定义特征是*每个输出神经元都是所有输入神经元的函数*。这就是矩阵乘法真正做的事情：如果矩阵是<math><msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></math>行乘以<math><msub><mi>n</mi> <mrow><mi>o</mi><mi>u</mi><mi>t</mi></mrow></msub></math>列，那么乘法本身计算的是<math><msub><mi>n</mi> <mrow><mi>o</mi><mi>u</mi><mi>t</mi></mrow></msub></math>个新特征，每个特征都是*所有*<math><msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></math>个输入特征的加权线性组合。^([2](ch03.html#idm45732623512888)) 因此，这些层通常被称为*全连接*层；最近，在流行的`Keras`库中，它们也经常被称为`Dense`层，这是一个更简洁的术语，传达了相同的概念。

既然我们知道该如何称呼它以及为什么，让我们根据我们已经定义的操作来定义`Dense`层 - 正如您将看到的，由于我们如何定义了我们的`Layer`基类，我们所需要做的就是在`_setup_layer`函数中将前一节中定义的`Operation`作为列表放入其中。

```py
class Dense(Layer):
    '''
 A fully connected layer that inherits from "Layer."
 '''
    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid()) -> None:
        '''
 Requires an activation function upon initialization.
 '''
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray) -> None:
        '''
 Defines the operations of a fully connected layer.
 '''
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None
```

请注意，我们将默认激活函数设置为`Linear`激活函数，这实际上意味着我们不应用激活函数，只是将恒等函数应用于层的输出。

在`Operation`和`Layer`之上，我们现在应该添加哪些构建块？为了训练我们的模型，我们知道我们将需要一个`NeuralNetwork`类来包装`Layer`，就像`Layer`包装`Operation`一样。不明显需要哪些其他类，所以我们将直接着手构建`NeuralNetwork`，并在进行过程中找出我们需要的其他类。

# 神经网络类，也许还有其他类

我们的`NeuralNetwork`类应该能做什么？在高层次上，它应该能够*从数据中学习*：更准确地说，它应该能够接收代表“观察”（`X`）和“正确答案”（`y`）的数据批次，并学习`X`和`y`之间的关系，这意味着学习一个能够将`X`转换为非常接近`y`的预测`p`的函数。

鉴于刚刚定义的`Layer`和`Operation`类，这种学习将如何进行？回顾上一章的模型是如何工作的，我们将实现以下内容：

1.  神经网络应该接受`X`并将其逐步通过每个`Layer`（实际上是一个方便的包装器，用于通过许多`Operation`进行馈送），此时结果将代表`prediction`。

1.  接下来，应该将`prediction`与值`y`进行比较，计算损失并生成“损失梯度”，这是与网络中最后一个层（即生成`prediction`的层）中的每个元素相关的损失的偏导数。

1.  最后，我们将通过每个层将这个损失梯度逐步向后发送，同时计算“参数梯度”——损失对每个参数的偏导数，并将它们存储在相应的`Operation`中。

## 图

[图3-5](#backpropagation_now_in_terms)以`Layer`的术语捕捉了神经网络的描述。

![神经网络图](assets/dlfs_0305.png)

###### 图3-5。反向传播，现在以Layer而不是Operation的术语

## 代码

我们应该如何实现这一点？首先，我们希望我们的神经网络最终处理`Layer`的方式与我们的`Layer`处理`Operation`的方式相同。例如，我们希望`forward`方法接收`X`作为输入，然后简单地执行类似以下的操作：

```py
for layer in self.layers:
    X = layer.forward(X)

return X
```

同样，我们希望我们的`backward`方法接收一个参数——我们最初称之为`grad`——然后执行类似以下的操作：

```py
for layer in reversed(self.layers):
    grad = layer.backward(grad)
```

`grad`将从哪里来？它必须来自*损失*，一个特殊的函数，它接收`prediction`以及`y`，然后：

+   计算代表网络进行该`prediction`的“惩罚”的单个数字。

+   针对每个`prediction`中的元素，发送一个梯度与损失相关的反向梯度。这个梯度是网络中最后一个`Layer`将作为其`backward`函数输入接收的内容。

在前一章的示例中，损失函数是`prediction`和目标之间的平方差，相应地计算了`prediction`相对于损失的梯度。

我们应该如何实现这一点？这个概念似乎很重要，值得拥有自己的类。此外，这个类可以类似于`Layer`类实现，只是`forward`方法将产生一个实际数字（一个`float`）作为损失，而不是一个`ndarray`被发送到下一个`Layer`。让我们正式化这一点。

## 损失类

`Loss`基类将类似于`Layer`——`forward`和`backward`方法将检查适当的`ndarray`的形状是否相同，并定义两个方法，`_output`和`_input_grad`，任何`Loss`子类都必须定义：

```py
class Loss(object):
    '''
 The "loss" of a neural network.
 '''

    def __init__(self):
        '''Pass'''
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        '''
 Computes the actual loss value.
 '''
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:
        '''
 Computes gradient of the loss value with respect to the input to the
 loss function.
 '''
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        '''
 Every subclass of "Loss" must implement the _output function.
 '''
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        '''
 Every subclass of "Loss" must implement the _input_grad function.
 '''
        raise NotImplementedError()
```

与`Operation`类一样，我们检查损失向后发送的梯度与从网络的最后一层接收的`prediction`的形状是否相同：

```py
class MeanSquaredError(Loss):

    def __init__(self)
        '''Pass'''
        super().__init__()

    def _output(self) -> float:
        '''
 Computes the per-observation squared error loss.
 '''
        loss =
            np.sum(np.power(self.prediction - self.target, 2)) /
            self.prediction.shape[0]

        return loss

    def _input_grad(self) -> ndarray:
        '''
 Computes the loss gradient with respect to the input for MSE loss.
 '''

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
```

在这里，我们简单地编写均方误差损失公式的前向和反向规则。

这是我们需要从头开始构建深度学习的最后一个关键构建块。让我们回顾一下这些部分如何组合在一起，然后继续构建模型！

# 从零开始的深度学习

我们最终希望构建一个`NeuralNetwork`类，使用[图3-5](#backpropagation_now_in_terms)作为指南，我们可以用来定义和训练深度学习模型。在我们深入编码之前，让我们准确描述一下这样一个类会是什么样的，以及它将如何与我们刚刚定义的`Operation`、`Layer`和`Loss`类进行交互：

1.  `NeuralNetwork`将具有`Layer`列表作为属性。`Layer`将如先前定义的那样，具有`forward`和`backward`方法。这些方法接受`ndarray`对象并返回`ndarray`对象。

1.  每个`Layer`在`_setup_layer`函数期间的`operations`属性中保存了一个`Operation`列表。

1.  这些`Operation`，就像`Layer`本身一样，有`forward`和`backward`方法，接受`ndarray`对象作为参数并返回`ndarray`对象作为输出。

1.  在每个操作中，`backward`方法中接收的`output_grad`的形状必须与`Layer`的`output`属性的形状相同。在`backward`方法期间向后传递的`input_grad`的形状和`input_`属性的形状也是如此。

1.  一些操作具有参数（存储在`param`属性中）；这些操作继承自`ParamOperation`类。`Layer`及其`forward`和`backward`方法的输入和输出形状的约束也适用于它们——它们接收`ndarray`对象并输出`ndarray`对象，`input`和`output`属性及其相应梯度的形状必须匹配。

1.  `NeuralNetwork`还将有一个`Loss`。这个类将获取`NeuralNetwork`最后一个操作的输出和目标，检查它们的形状是否相同，并计算损失值（一个数字）和一个`ndarray` `loss_grad`，该`loss_grad`将被馈送到输出层，开始反向传播。

## 实现批量训练

我们已经多次介绍了逐批次训练模型的高级步骤。这些步骤很重要，值得重复：

1.  通过模型函数（“前向传递”）将输入馈送。

1.  计算代表损失的数字。

1.  使用链式法则和在前向传递期间计算的量，计算损失相对于参数的梯度。

1.  使用这些梯度更新参数。

然后我们将通过一批新数据并重复这些步骤。

将这些步骤转换为刚刚描述的`NeuralNetwork`框架是直接的：

1.  接收`X`和`y`作为输入，都是`ndarray`。

1.  逐个将`X`通过每个`Layer`向前传递。

1.  使用`Loss`生成损失值和损失梯度以进行反向传播。

1.  使用损失梯度作为网络`backward`方法的输入，该方法将计算网络中每一层的`param_grads`。

1.  在每一层上调用`update_params`函数，该函数将使用`NeuralNetwork`的整体学习率以及新计算的`param_grads`。

我们最终有了一个完整的神经网络定义，可以进行批量训练。现在让我们编写代码。

## 神经网络：代码

编写所有这些代码非常简单：

```py
class NeuralNetwork(object):
    '''
 The class for a neural network.
 '''
    def __init__(self, layers: List[Layer],
                 loss: Loss,
                 seed: float = 1)
        '''
 Neural networks need layers, and a loss.
 '''
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: ndarray) -> ndarray:
        '''
 Passes data forward through a series of layers.
 '''
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        '''
 Passes data backward through a series of layers.
 '''

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self,
                    x_batch: ndarray,
                    y_batch: ndarray) -> float:
        '''
 Passes data forward through the layers.
 Computes the loss.
 Passes data backward through the layers.
 '''

        predictions = self.forward(x_batch)

        loss = self.loss.forward(predictions, y_batch)

        self.backward(self.loss.backward())

        return loss

    def params(self):
        '''
 Gets the parameters for the network.
 '''
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        '''
 Gets the gradient of the loss with respect to the parameters for the
 network.
 '''
        for layer in self.layers:
            yield from layer.param_grads
```

有了这个`NeuralNetwork`类，我们可以以更模块化、灵活的方式实现上一章中的模型，并定义其他模型来表示输入和输出之间的复杂非线性关系。例如，这里是如何轻松实例化我们在上一章中介绍的两个模型——线性回归和神经网络：^([3](ch03.html#idm45732622822120))

```py
linear_regression = NeuralNetwork(
    layers=[Dense(neurons = 1)],
            loss = MeanSquaredError(),
            learning_rate = 0.01
            )

neural_network = NeuralNetwork(
    layers=[Dense(neurons=13,
                  activation=Sigmoid()),
            Dense(neurons=1,
                  activation=Linear())],
            loss = MeanSquaredError(),
            learning_rate = 0.01
            )
```

基本上我们已经完成了；现在我们只需反复将数据通过网络以便学习。然而，为了使这个过程更清晰、更容易扩展到下一章中将看到的更复杂的深度学习场景，定义另一个类来执行训练以及另一个类来执行“学习”，即根据反向传播计算的梯度实际更新`NeuralNetwork`参数。让我们快速定义这两个类。

# 训练器和优化器

首先，让我们注意这些类与我们在[第2章](ch02.html#fundamentals)中用于训练网络的代码之间的相似之处。在那里，我们使用以下代码来实现用于训练模型的前述四个步骤：

```py
# pass X_batch forward and compute the loss
forward_info, loss = forward_loss(X_batch, y_batch, weights)

# compute the gradient of the loss with respect to each of the weights
loss_grads = loss_gradients(forward_info, weights)

# update the weights
for key in weights.keys():
    weights[key] -= learning_rate * loss_grads[key]
```

这段代码位于一个`for`循环中，该循环反复将数据通过定义和更新我们的网络的函数。

有了我们现在的类，我们最终将在`Trainer`类中的`fit`函数内部执行这些操作，该函数将主要是对前一章中使用的`train`函数的包装。 （完整的代码在本章的[Jupyter Notebook](https://oreil.ly/2MV0aZI)中的书的GitHub页面上。）主要区别是，在这个新函数内部，前面代码块中的前两行将被替换为这一行：

```py
neural_network.train_batch(X_batch, y_batch)
```

更新参数将在以下两行中进行，这将在一个单独的`Optimizer`类中进行。最后，之前包围所有这些内容的`for`循环将在包围`NeuralNetwork`和`Optimizer`的`Trainer`类中进行。

接下来，让我们讨论为什么需要一个`Optimizer`类以及它应该是什么样子。

## 优化器

在上一章描述的模型中，每个`Layer`包含一个简单的规则，根据参数和它们的梯度来更新权重。正如我们将在下一章中提到的，我们可以使用许多其他更新规则，例如涉及梯度更新*历史*而不仅仅是在该迭代中传入的特定批次的梯度更新。创建一个单独的`Optimizer`类将使我们能够灵活地将一个更新规则替换为另一个，这是我们将在下一章中更详细地探讨的内容。

### 描述和代码

基本的`Optimizer`类将接受一个`NeuralNetwork`，每次调用`step`函数时，将根据它们当前的值、梯度和`Optimizer`中存储的任何其他信息来更新网络的参数：

```py
class Optimizer(object):
    '''
 Base class for a neural network optimizer.
 '''
    def __init__(self,
                 lr: float = 0.01):
        '''
 Every optimizer must have an initial learning rate.
 '''
        self.lr = lr

    def step(self) -> None:
        '''
 Every optimizer must implement the "step" function.
 '''
        pass
```

以下是我们迄今为止看到的简单更新规则的实际情况，即*随机梯度下降*：

```py
class SGD(Optimizer):
    '''
 Stochastic gradient descent optimizer.
 '''
    def __init__(self,
                 lr: float = 0.01) -> None:
        '''Pass'''
        super().__init__(lr)

    def step(self):
        '''
 For each parameter, adjust in the appropriate direction, with the
 magnitude of the adjustment based on the learning rate.
 '''
        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):

            param -= self.lr * param_grad
```

###### 注意

请注意，虽然我们的`NeuralNetwork`类没有`_update_params`方法，但我们依赖于`params()`和`param_grads()`方法来提取正确的`ndarray`以进行优化。

这就是基本的`Optimizer`类；接下来让我们来看一下`Trainer`类。

## 训练器

除了按照前面描述的方式训练模型外，`Trainer`类还将`NeuralNetwork`与`Optimizer`连接在一起，确保后者正确训练前者。您可能已经注意到，在前一节中，我们在初始化`Optimizer`时没有传入`NeuralNetwork`；相反，我们将在不久后初始化`Trainer`类时将`NeuralNetwork`分配为`Optimizer`的属性，使用以下代码行：

```py
setattr(self.optim, 'net', self.net)
```

在下一小节中，我展示了一个简化但有效的`Trainer`类的工作版本，目前只包含`fit`方法。该方法为我们的模型训练了一定数量的*epochs*，并在每个一定数量的epochs后打印出损失值。在每个epoch中，我们：

1.  在epoch开始时对数据进行洗牌

1.  通过网络以批次方式传递数据，每传递完一个批次后更新参数

当我们通过`Trainer`将整个训练集传递完时，该epoch结束。

### 训练器代码

下面是一个简单版本的`Trainer`类的代码，我们隐藏了在`fit`函数中使用的两个不言自明的辅助方法：`generate_batches`，它从`X_train`和`y_train`生成用于训练的数据批次，以及`permute_data`，它在每个epoch开始时对`X_train`和`y_train`进行洗牌。在`train`函数中还包括一个`restart`参数：如果为`True`（默认值），则在调用`train`函数时会重新初始化模型的参数为随机值：

```py
class Trainer(object):
    '''
 Trains a neural network.
 '''
    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer)
        '''
 Requires a neural network and an optimizer in order for training to
 occur. Assign the neural network as an instance variable to the optimizer.
 '''
        self.net = net
        setattr(self.optim, 'net', self.net)

    def fit(self, X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            restart: bool = True) -> None:
        '''
 Fits the neural network on the training data for a certain number of
 epochs. Every "eval_every" epochs, it evaluates the neural network on
 the testing data.
 '''
        np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True

        for e in range(epochs):

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train,
                                                    batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

            if (e+1) % eval_every == 0:

                test_preds = self.net.forward(X_test)

                loss = self.net.loss.forward(test_preds, y_test)

                print(f"Validation loss after {e+1} epochs is {loss:.3f}")
```

在书的[GitHub存储库](https://oreil.ly/2MV0aZI)中的这个函数的完整版本中，我们还实现了*提前停止*，它执行以下操作：

1.  它每`eval_every`个epoch保存一次损失值。

1.  它检查验证损失是否低于上次计算时的值。

1.  如果验证损失*不*更低，则使用`eval_every`个epoch之前的模型。

最后，我们已经准备好训练这些模型了！

# 将所有内容整合在一起

这是使用所有`Trainer`和`Optimizer`类以及之前定义的两个模型——`linear_regression`和`neural_network`来训练我们的网络的完整代码。我们将学习率设置为`0.01`，最大迭代次数设置为`50`，并且每`10`次迭代评估我们的模型：

```py
optimizer = SGD(lr=0.01)
trainer = Trainer(linear_regression, optimizer)

trainer.fit(X_train, y_train, X_test, y_test,
       epochs = 50,
       eval_every = 10,
       seed=20190501);
```

```py
Validation loss after 10 epochs is 30.295
Validation loss after 20 epochs is 28.462
Validation loss after 30 epochs is 26.299
Validation loss after 40 epochs is 25.548
Validation loss after 50 epochs is 25.092
```

使用来自[第2章](ch02.html#fundamentals)的相同模型评分函数，并将它们包装在一个`eval_regression_model`函数中，我们得到以下结果：

```py
eval_regression_model(linear_regression, X_test, y_test)
```

```py
Mean absolute error: 3.52

Root mean squared error 5.01
```

这些结果与我们在上一章中运行的线性回归的结果类似，证实了我们的框架正在工作。

使用具有13个神经元的隐藏层的`neural_network`模型运行相同的代码，我们得到以下结果：

```py
Validation loss after 10 epochs is 27.434
Validation loss after 20 epochs is 21.834
Validation loss after 30 epochs is 18.915
Validation loss after 40 epochs is 17.193
Validation loss after 50 epochs is 16.214
```

```py
eval_regression_model(neural_network, X_test, y_test)
```

```py
Mean absolute error: 2.60

Root mean squared error 4.03
```

同样，这些结果与我们在上一章中看到的结果类似，它们比我们直接的线性回归要好得多。

## 我们的第一个深度学习模型（从头开始）

既然所有的设置都已经完成，定义我们的第一个深度学习模型就变得微不足道了：

```py
deep_neural_network = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=13,
                   activation=Sigmoid()),
            Dense(neurons=1,
                   activation=LinearAct())],
    loss=MeanSquaredError(),
    learning_rate=0.01
)
```

我们甚至不会试图在这方面变得聪明（尚未）。我们只会添加一个与第一层具有相同维度的隐藏层，这样我们的网络现在有两个隐藏层，每个隐藏层有13个神经元。

使用与之前模型相同的学习率和评估计划进行训练会产生以下结果：

```py
Validation loss after 10 epochs is 44.134
Validation loss after 20 epochs is 25.271
Validation loss after 30 epochs is 22.341
Validation loss after 40 epochs is 16.464
Validation loss after 50 epochs is 14.604
```

```py
eval_regression_model(deep_neural_network, X_test, y_test)
```

```py
Mean absolute error: 2.45

Root mean squared error 3.82
```

我们最终从头开始进行了深度学习，事实上，在这个真实世界的问题上，没有使用任何技巧（只是稍微调整学习率），我们的深度学习模型的表现略好于只有一个隐藏层的神经网络。

更重要的是，我们通过构建一个易于扩展的框架来实现这一点。我们可以很容易地实现其他类型的`Operation`，将它们包装在新的`Layer`中，并将它们直接放入其中，假设它们已经定义了`_output`和`_input_grad`方法，并且它们的输入、输出和参数的维度与它们各自的梯度相匹配。同样地，我们可以很容易地将不同的激活函数放入我们现有的层中，看看是否会降低我们的错误指标；我鼓励你克隆本书的[GitHub存储库](https://oreil.ly/deep-learning-github)并尝试一下！

# 结论和下一步

在下一章中，我将介绍几种技巧，这些技巧对于让我们的模型在面对比这个简单问题更具挑战性的问题时能够正确训练是至关重要的^([4](ch03.html#idm45732621371848))——特别是定义其他`Loss`和`Optimizer`。我还将介绍调整学习率和在整个训练过程中修改学习率的其他技巧，并展示如何将这些技巧融入`Optimizer`和`Trainer`类中。最后，我们将看到Dropout，这是一种新型的`Operation`，已被证明对增加深度学习模型的训练稳定性至关重要。继续前进！

^([1](ch03.html#idm45732624417528-marker)) 在所有激活函数中，`sigmoid`函数最接近大脑中神经元的实际激活，它将输入映射到0到1之间，但一般来说，激活函数可以是任何单调的非线性函数。

^([2](ch03.html#idm45732623512888-marker)) 正如我们将在[第5章](ch05.html#convolution)中看到的，这并不适用于所有层：例如，在*卷积*层中，每个输出特征是输入特征的*一个小子集*的组合。

^([3](ch03.html#idm45732622822120-marker)) 学习率0.01并不特殊；我们只是在写前一章时在实验过程中发现它是最佳的。

^([4](ch03.html#idm45732621371848-marker)) 即使在这个简单的问题上，稍微改变超参数可能会导致深度学习模型无法击败两层神经网络。克隆[GitHub存储库](https://oreil.ly/deep-learning-github)并尝试一下吧！
