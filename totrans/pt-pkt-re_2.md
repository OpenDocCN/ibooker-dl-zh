# 第2章. 张量

在深入了解PyTorch开发世界之前，熟悉PyTorch中的基本数据结构`torch.Tensor`是很重要的。通过理解张量，您将了解PyTorch如何处理和存储数据，由于深度学习基本上是浮点数的收集和操作，理解张量将帮助您了解PyTorch如何为深度学习实现更高级的功能。此外，在预处理输入数据或在模型开发过程中操作输出数据时，您可能经常使用张量操作。

本章作为理解张量和在代码中实现张量函数的快速参考。我将从描述张量是什么开始，并向您展示如何使用函数来创建、操作和加速GPU上的张量操作的一些简单示例。接下来，我们将更广泛地查看创建张量和执行数学操作的API，以便您可以快速查阅一份全面的张量功能列表。在每个部分中，我们将探讨一些更重要的函数，识别常见的陷阱，并检查它们的使用中的关键点。

# 张量是什么？

在PyTorch中，张量是一种用于存储和操作数据的数据结构。与NumPy数组类似，张量是一个包含单一数据类型元素的多维数组。张量可以用来表示标量、向量、矩阵和*n*维数组，并且是从`torch.Tensor`类派生的。然而，张量不仅仅是数字数组。从`torch.Tensor`类创建或实例化张量对象使我们可以访问一组内置的类属性和操作或类方法，提供了一套强大的内置功能。本章详细描述了这些属性和操作。

张量还包括一些附加优势，使它们比NumPy数组更适合用于深度学习计算。首先，使用GPU加速可以显著加快张量操作的速度。其次，可以使用分布式处理在多个CPU和GPU上以及跨多个服务器上存储和操作张量。第三，张量跟踪它们的图计算，正如我们将在[“自动微分（Autograd）”](#section_autograd)中看到的，这在实现深度学习库中非常重要。

为了进一步解释张量实际上是什么以及如何使用它，我将从一个简单示例开始，创建一些张量并执行一个张量操作。

## 简单CPU示例

这里有一个简单的示例，创建一个张量，执行一个张量操作，并在张量本身上使用一个内置方法。默认情况下，张量数据类型将从输入数据类型派生，并且张量将分配到CPU设备。首先，我们导入PyTorch库，然后我们从二维列表创建两个张量`x`和`y`。接下来，我们将这两个张量相加，并将结果存储在`z`中。我们可以在这里使用`+`运算符，因为`torch.Tensor`类支持运算符重载。最后，我们打印新的张量`z`，我们可以看到它是`x`和`y`的矩阵和，并打印`z`的大小。注意，`z`本身是一个张量对象，`size()`方法用于返回其矩阵维度，即2×3：

```py
import torch

x = torch.tensor([[1,2,3],[4,5,6]])
y = torch.tensor([[7,8,9],[10,11,12]])
z = x + y
print(z)
# out:
#  tensor([[ 8, 10, 12],
#          [14, 16, 18]])

print(z.size())
# out: torch.Size([2, 3])
```

###### 注意

在旧代码中可能会看到使用`torch.Tensor()`（大写T）构造函数。这是`torch.FloatTensor`默认张量类型的别名。您应该改用`torch.tensor()`来创建您的张量。

## 简单GPU示例

由于在GPU上加速张量操作是张量优于NumPy数组的主要优势，我将向您展示一个简单的示例。这是上一节中的相同示例，但在这里，如果有GPU设备，我们将将张量移动到GPU设备。请注意，输出张量也分配给了GPU。您可以使用设备属性（例如`z.device`）来双重检查张量所在的位置。

在第一行中，`torch.cuda.is_available()`函数将在您的机器支持GPU时返回`True`。这是一种方便的编写更健壮代码的方式，可以在存在GPU时加速运行，但在没有GPU时也可以在CPU上运行。在输出中，`device='cuda:0'`表示正在使用第一个GPU。如果您的机器包含多个GPU，您还可以控制使用哪个GPU：

```py
device = "cuda" if torch.cuda.is_available()
  else "cpu"
x = torch.tensor([[1,2,3],[4,5,6]],
                 device=device)
y = torch.tensor([[7,8,9],[10,11,12]],
                 device=device)
z = x + y
print(z)
# out:
#   tensor([[ 8, 10, 12],
#          [14, 16, 18]], device='cuda:0')

print(z.size())
# out: torch.Size([2, 3])

print(z.device)
# out: cuda:0
```

## 在CPU和GPU之间移动张量

前面的代码使用`torch.tensor()`在特定设备上创建张量；然而，更常见的是将现有张量移动到设备上，即如果有GPU设备的话，通常是GPU。您可以使用`torch.to()`方法来实现。当通过张量操作创建新张量时，PyTorch会在相同设备上创建新张量。在下面的代码中，`z`位于GPU上，因为`x`和`y`位于GPU上。张量`z`通过`torch.to("cpu")`移回CPU进行进一步处理。还请注意，操作中的所有张量必须在同一设备上。如果`x`在GPU上，而`y`在CPU上，我们将会收到错误：

```py
device = "cuda" if torch.cuda.is_available()
  else "cpu"
x = x.to(device)
y = y.to(device)
z = x + y
z = z.to("cpu")
# out:
# tensor([[ 8, 10, 12],
#         [14, 16, 18]])
```

###### 注意

您可以直接使用字符串作为设备参数，而不是设备对象。以下都是等效的：

+   `device="cuda"`

+   `device=torch.device("cuda")`

+   `device="cuda:0"`

+   `device=torch.device("cuda:0")`

# 创建张量

前一节展示了创建张量的简单方法；然而，还有许多其他方法可以实现。您可以从现有的数字数据中创建张量，也可以创建随机抽样。张量可以从存储在类似数组结构中的现有数据（如列表、元组、标量或序列化数据文件）以及NumPy数组中创建。

以下代码说明了创建张量的一些常见方法。首先，它展示了如何使用`torch.tensor()`从列表创建张量。此方法也可用于从其他数据结构（如元组、集合或NumPy数组）创建张量：

```py
importnumpy# Created from preexisting arraysw=torch.tensor([1,2,3])![1](Images/1.png)w=torch.tensor((1,2,3))![2](Images/2.png)w=torch.tensor(numpy.array([1,2,3]))![3](Images/3.png)# Initialized by sizew=torch.empty(100,200)![4](Images/4.png)w=torch.zeros(100,200)![5](Images/5.png)w=torch.ones(100,200)![6](Images/6.png)
```

[![1](Images/1.png)](#co_tensors_CO1-1)

从列表中

[![2](Images/2.png)](#co_tensors_CO1-2)

从元组中

[![3](Images/3.png)](#co_tensors_CO1-3)

从NumPy数组中

[![4](Images/4.png)](#co_tensors_CO1-4)

未初始化；元素值不可预测

[![5](Images/5.png)](#co_tensors_CO1-5)

所有元素初始化为0.0

[![6](Images/6.png)](#co_tensors_CO1-6)

所有元素初始化为1.0

如前面的代码示例所示，您还可以使用`torch.empty()`、`torch.ones()`和`torch.zeros()`等函数创建和初始化张量，并指定所需的大小。

如果您想要使用随机值初始化张量，PyTorch支持一组强大的函数，例如`torch.rand()`、`torch.randn()`和`torch.randint()`，如下面的代码所示：

```py
# Initialized by size with random valuesw=torch.rand(100,200)![1](Images/1.png)w=torch.randn(100,200)![2](Images/2.png)w=torch.randint(5,10,(100,200))![3](Images/3.png)# Initialized with specified data type or devicew=torch.empty((100,200),dtype=torch.float64,device="cuda")# Initialized to have the same size, data type,#   and device as another tensorx=torch.empty_like(w)
```

[![1](Images/1.png)](#co_tensors_CO2-1)

创建一个100×200的张量，元素来自区间[0,1)上的均匀分布。

[![2](Images/2.png)](#co_tensors_CO2-2)

元素是均值为0、方差为1的正态分布随机数。

[![3](Images/3.png)](#co_tensors_CO2-3)

元素是介于5和10之间的随机整数。

在初始化时，您可以像前面的代码示例中所示指定数据类型和设备（即CPU或GPU）。此外，示例展示了如何使用PyTorch创建具有与其他张量相同属性但使用不同数据初始化的张量。带有`_like`后缀的函数，如`torch.empty_like()`和`torch.ones_like()`，返回具有与另一个张量相同大小、数据类型和设备的张量，但初始化方式不同（参见[“从随机样本创建张量”](#creating-tensors-section)）。

###### 注意

一些旧函数，如`from_numpy()`和`as_tensor()`，在实践中已被`torch.tensor()`构造函数取代，该构造函数可用于处理所有情况。

[表2-1](#table_creation_ops)列出了用于创建张量的PyTorch函数。您应该使用`torch`命名空间下的每个函数，例如`torch.empty()`。您可以通过访问[PyTorch张量文档](https://pytorch.tips/torch)获取更多详细信息。

表2-1. 张量创建函数

| 函数 | 描述 |
| --- | --- |
| `torch.**tensor**(*data, dtype=None, device=None, requires_grad=False, pin_memory=False*)` | 从现有数据结构创建张量 |
| `torch.**empty**(**size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 根据内存中值的随机状态创建未初始化元素的张量 |
| `torch.**zeros**(**size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 创建一个所有元素初始化为0.0的张量 |
| `torch.**ones**(**size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 创建一个所有元素初始化为1.0的张量 |
| `torch.**arange**(*start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 使用公共步长值在范围内创建值的一维张量 |
| `torch.**linspace**(*start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 在`start`和`end`之间创建线性间隔点的一维张量 |
| `torch.**logspace**(*start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 在`start`和`end`之间创建对数间隔点的一维张量 |
| `torch.**eye**(*n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 创建一个对角线为1，其他位置为0的二维张量 |
| `torch.**full**(*size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 创建一个填充了`fill_value`的张量 |
| `torch.**load**(*f*)` | 从序列化的pickle文件中加载张量 |
| `torch.**save**(*f*)` | 将张量保存到序列化的pickle文件中 |

PyTorch文档包含了创建张量的完整函数列表，以及如何使用它们的更详细解释。在创建张量时，请记住一些常见的陷阱和额外的见解：

+   大多数创建函数都接受可选的`dtype`和`device`参数，因此您可以在创建时设置这些参数。

+   使用`torch.arange()`而不是已弃用的`torch.range()`函数。当步长已知时，请使用`torch.arange()`。当元素数量已知时，请使用`torch.linspace()`。

+   您可以使用`torch.tensor()`从类似数组的结构（如列表、NumPy数组、元组和集合）创建张量。要将现有张量转换为NumPy数组和列表，分别使用`torch.numpy()`和`torch.tolist()`函数。

## 张量属性

PyTorch受欢迎的一个特点是它非常符合Python风格且面向对象。由于张量是自己的数据类型，因此可以读取张量对象本身的属性。现在您可以创建张量了，通过访问它们的属性，可以快速查找有关它们的信息是很有用的。假设`x`是一个张量，您可以按如下方式访问`x`的几个属性：

`x.dtype`

指示张量的数据类型（请参见[表2-2](#table_tensor_dtype)列出的PyTorch数据类型列表）

`x.device`

指示张量的设备位置（例如，CPU或GPU内存）

`x.shape`

显示张量的维度

`x.ndim`

标识张量的维数或秩

`x.requires_grad`

一个布尔属性，指示张量是否跟踪图计算（参见[“自动微分（Autograd）”](#section_autograd)）

`x.grad`

如果`requires_grad`为`True`，则包含实际的梯度

`x.grad_fn`

如果`requires_grad`为`True`，则存储使用的图计算函数

`x.s_cuda`，`x.is_sparse`，`x.is_quantized`，`x.is_leaf`，`x.is_mkldnn`

指示张量是否满足某些条件的布尔属性

`x.layout`

指示张量在内存中的布局方式

请记但，当访问对象属性时，不要像调用类方法那样包括括号（`()`）（例如，使用`x.shape`，而不是`x.shape()`）。

## 数据类型

在深度学习开发中，了解数据及其计算所使用的数据类型非常重要。因此，在创建张量时，应该控制所使用的数据类型。如前所述，所有张量元素具有相同的数据类型。您可以在创建张量时使用`dtype`参数指定数据类型，或者可以使用适当的转换方法或`to()`方法将张量转换为新的`dtype`，如下面的代码所示：

```py
# Specify the data type at creation using dtypew=torch.tensor([1,2,3],dtype=torch.float32)# Use the casting method to cast to a new data typew.int()# w remains a float32 after the castw=w.int()# w changes to an int32 after the cast# Use the to() method to cast to a new typew=w.to(torch.float64)![1](Images/1.png)w=w.to(dtype=torch.float64)![2](Images/2.png)# Python automatically converts data types during# operationsx=torch.tensor([1,2,3],dtype=torch.int32)y=torch.tensor([1,2,3],dtype=torch.float32)z=x+y![3](Images/3.png)print(z.dtype)# out: torch.float32
```

[![1](Images/1.png)](#co_tensors_CO3-1)

传入数据类型。

[![2](Images/2.png)](#co_tensors_CO3-2)

直接使用`dtype`定义数据类型。

[![3](Images/3.png)](#co_tensors_CO3-3)

Python会自动将`x`转换为`float32`，并将`z`返回为`float32`。

请注意，转换和`to()`方法不会改变张量的数据类型，除非重新分配张量。此外，在执行混合数据类型的操作时，PyTorch会自动将张量转换为适当的类型。

大多数张量创建函数允许您在创建时使用`dtype`参数指定数据类型。在设置`dtype`或转换张量时，请记住使用`torch`命名空间（例如，使用`torch.int64`，而不仅仅是`int64`）。

[表2-2](#table_tensor_dtype)列出了PyTorch中所有可用的数据类型。每种数据类型都会导致不同的张量类，具体取决于张量的设备。相应的张量类分别显示在CPU和GPU的最右两列中。

表2-2. 张量数据类型

| 数据类型 | dtype | CPU张量 | GPU张量 |
| --- | --- | --- | --- |
| 32位浮点数（默认） | `torch.float32`或`torch.float` | `torch.FloatTensor` | `torch.cuda.FloatTensor` |
| 64位浮点数 | `torch.float64`或`torch.double` | `torch.DoubleTensor` | `torch.cuda.DoubleTensor` |
| 16位浮点数 | `torch.float16`或`torch.half` | `torch.HalfTensor` | `torch.cuda.HalfTensor` |
| 8位整数（无符号） | `torch.uint8` | `torch.ByteTensor` | `torch.cuda.ByteTensor` |
| 8位整数（有符号） | `torch.int8` | `torch.CharTensor` | `torch.cuda.CharTensor` |
| 16位整数（有符号） | `torch.int16`或`torch.short` | `torch.ShortTensor` | `torch.cuda.ShortTensor` |
| 32位整数（有符号） | `torch.int32`或`torch.int` | `torch.IntTensor` | `torch.cuda.IntTensor` |
| 64位整数（有符号） | `torch.int64`或`torch.long` | `torch.LongTensor` | `torch.cuda.LongTensor` |
| 布尔值 | `torch.bool` | `torch.BoolTensor` | `torch.cuda.BoolTensor` |

###### 注意

为了减少空间复杂度，有时您可能希望重用内存并使用*就地操作*覆盖张量值。要执行就地操作，请在函数名称后附加下划线(_)后缀。例如，函数`y.add_(x)`将`x`添加到`y`，但结果将存储在`y`中。

## 从随机样本创建张量

在深度学习开发过程中经常需要创建随机数据。有时您需要将权重初始化为随机值或创建具有指定分布的随机输入。PyTorch支持一组非常强大的函数，您可以使用这些函数从随机数据创建张量。

与其他创建函数一样，您可以在创建张量时指定dtype和device。[表2-3](#table_random_ops)列出了一些随机抽样函数的示例。

表2-3. 随机抽样函数

| 函数 | 描述 |
| --- | --- |
| `torch.`**`rand`**`(**size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 从区间[0到1]上的均匀分布中选择随机值 |
| `torch.`**`randn`**`(**size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 从均值为零方差为单位的标准正态分布中选择随机值 |
| `torch.`**`normal`**`(*mean, std, *, generator=None, out=None*)` | 从具有指定均值和方差的正态分布中选择随机数 |
| `torch.`**`randint`**`(*low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False*)` | 在指定的低值和高值之间生成均匀分布的随机整数 |
| `torch.`**`randperm`**`(*n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False*)` | 创建从0到*n*-1的整数的随机排列 |
| `torch.`**`bernoulli`**`(*input, *, generator=None, out=None*)` | 从伯努利分布中绘制二进制随机数（0或1） |
| `torch.`**`multinomial`**`(*input, num_samples, replacement=False, *, generator=None, out=None*)` | 根据多项分布中的权重从列表中选择一个随机数 |

###### 提示

您还可以创建从更高级分布（如柯西分布、指数分布、几何分布和对数正态分布）中抽样的值张量。为此，使用`torch.empty()`创建张量，并对分布（例如柯西分布）应用就地函数。请记住，就地方法使用下划线后缀。例如，`x = torch.empty([10,5]).cauchy_()`创建一个从柯西分布中抽取的随机数张量。

## 像其他张量一样创建张量

您可能希望创建并初始化一个具有与另一个张量相似属性的张量，包括`dtype`、`device`和`layout`属性，以便进行计算。许多张量创建操作都有一个相似性函数，允许您轻松地执行此操作。相似性函数将具有后缀`_like`。例如，`torch.empty_like(tensor_a)`将创建一个具有`tensor_a`的`dtype`、`device`和`layout`属性的空张量。一些相似性函数的示例包括`empty_like()`、`zeros_like()`、`ones_like()`、`full_like()`、`rand_like()`、`randn_like()`和`rand_int_like()`。

# 张量操作

现在您了解如何创建张量，让我们探索您可以对其执行的操作。PyTorch支持一组强大的张量操作，允许您访问和转换张量数据。

首先我将描述如何访问数据的部分，操作它们的元素，并组合张量以形成新的张量。然后我将向您展示如何执行简单的计算以及高级的数学计算，通常在恒定时间内。PyTorch提供了许多内置函数。在创建自己的函数之前检查可用的函数是很有用的。

## 索引、切片、组合和拆分张量

创建张量后，您可能希望访问数据的部分并组合或拆分张量以形成新张量。以下代码演示了如何执行这些类型的操作。您可以像切片和索引NumPy数组一样切片和索引张量，如以下代码的前几行所示。请注意，即使数组只有一个元素，索引和切片也会返回张量。在传递给`print()`等其他函数时，您需要使用`item()`函数将单个元素张量转换为Python值：

```py
x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
print(x)
# out:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

# Indexing, returns a tensor
print(x[1,1])
# out: tensor(4)

# Indexing, returns a value as a Python number
print(x[1,1].item())
# out: 4
```

在下面的代码中，我们可以看到我们可以使用与用于切片Python列表和NumPy数组相同的`[*start*:*end*:*step*]`格式执行切片。我们还可以使用布尔索引来提取满足某些条件的数据部分，如下所示：

```py
# Slicing
print(x[:2,1])
# out: tensor([2, 4])

# Boolean indexing
# Only keep elements less than 5
print(x[x<5])
# out: tensor([1, 2, 3, 4])
```

PyTorch还支持转置和重塑数组，如下面的代码所示：

```py
# Transpose array; x.t() or x.T can be used
print(x.t())
# tensor([[1, 3, 5, 7],
#         [2, 4, 6, 8]])

# Change shape; usually view() is preferred over
# reshape()
print(x.view((2,4)))
# tensor([[1, 3, 5, 7],
#         [2, 4, 6, 8]])
```

您还可以使用`torch.stack()`和`torch.unbind()`等函数组合或拆分张量，如下面的代码所示：

```py
# Combining tensors
y = torch.stack((x, x))
print(y)
# out:
# tensor([[[1, 2],
#          [3, 4],
#          [5, 6],
#          [7, 8]],

#         [[1, 2],
#          [3, 4],
#          [5, 6],
#          [7, 8]]])

# Splitting tensors
a,b = x.unbind(dim=1)
print(a,b)
# out:
#  tensor([1, 3, 5, 7]); tensor([2, 4, 6, 8])
```

PyTorch提供了一组强大的内置函数，可用于以不同方式访问、拆分和组合张量。[表2-4](#table_indexing_ops)列出了一些常用的用于操作张量元素的函数。

表2-4。索引、切片、组合和拆分操作

| 函数 | 描述 |
| --- | --- |
| `torch.`**`cat`**`()` | 在给定维度中连接给定序列的张量。 |
| `torch.`**`chunk`**`()` | 将张量分成特定数量的块。每个块都是输入张量的视图。 |
| `torch.`**`gather`**`()` | 沿着由维度指定的轴收集值。 |
| `torch.`**`index_select`**`()` | 使用索引中的条目沿着维度索引输入张量的新张量，索引是`LongTensor`。 |
| `torch.`**`masked_select`**`()` | 根据布尔掩码（`BoolTensor`）索引输入张量的新1D张量。 |
| `torch.`**`narrow`**`()` | 返回输入张量的窄版本的张量。 |
| `torch.`**`nonzero`**`()` | 返回非零元素的索引。 |
| `torch.`**`reshape`**`()` | 返回一个与输入张量具有相同数据和元素数量但形状不同的张量。使用`view()`而不是确保张量不被复制。 |
| `torch.`**`split`**`()` | 将张量分成块。每个块都是原始张量的视图或子分区。 |
| `torch.`**`squeeze`**`()` | 返回一个去除输入张量所有尺寸为1的维度的张量。 |
| `torch.`**`stack`**`()` | 沿新维度连接一系列张量。 |
| `torch.`**`t`**`()` | 期望输入为2D张量并转置维度0和1。 |
| `torch.`**`take`**`()` | 在切片不连续时返回指定索引处的张量。 |
| `torch.`**`transpose`**`()` | 仅转置指定的维度。 |
| `torch.`**`unbind`**`()` | 通过返回已删除维度的元组来移除张量维度。 |
| `torch.`**`unsqueeze`**`()` | 返回一个在指定位置插入大小为1的维度的新张量。 |
| `torch.`**`where`**`()` | 根据指定条件从两个张量中的一个返回所选元素的张量。 |

其中一些函数可能看起来多余。然而，重要的是要记住以下关键区别和最佳实践：

+   `item()`是一个重要且常用的函数，用于从包含单个值的张量返回Python数字。

+   在大多数情况下，用`view()`代替`reshape()`来重新塑造张量。使用`reshape()`可能会导致张量被复制，这取决于其在内存中的布局。`view()`确保不会被复制。

+   使用`x.T`或`x.t()`是转置1D或2D张量的简单方法。处理多维张量时，请使用`transpose()`。

+   `torch.squeeze()`函数在深度学习中经常用于去除未使用的维度。例如，使用`squeeze()`可以将包含单个图像的图像批次从4D减少到3D。

+   `torch.unsqueeze()`函数在深度学习中经常用于添加大小为1的维度。由于大多数PyTorch模型期望批量数据作为输入，当您只有一个数据样本时，可以应用`unsqueeze()`。例如，您可以将一个3D图像传递给`torch.unsqueeze()`以创建一个图像批次。

###### 注意

PyTorch在本质上非常符合Python的特性。与大多数Python类一样，一些PyTorch函数可以直接在张量上使用内置方法，例如`x.size()`。

其他函数直接使用`torch`命名空间调用。这些函数以张量作为输入，就像在`torch.save(x, 'tensor.pt')`中的`x`一样。

## 数学张量操作

深度学习开发在很大程度上基于数学计算，因此PyTorch支持非常强大的内置数学函数集。无论您是创建新的数据转换、自定义损失函数还是构建自己的优化算法，您都可以通过PyTorch提供的数学函数加快研究和开发速度。

本节的目的是快速概述PyTorch中许多可用的数学函数，以便您可以快速了解当前存在的内容，并在需要时找到适当的函数。

PyTorch支持许多不同类型的数学函数，包括逐点操作、缩减函数、比较计算以及线性代数操作，以及频谱和其他数学计算。我们将首先看一下有用的数学操作的第一类是*逐点操作*。逐点操作在张量中的每个点上执行操作，并返回一个新的张量。

它们对于舍入和截断以及三角和逻辑操作非常有用。默认情况下，这些函数将创建一个新的张量或使用由`out`参数传递的张量。如果要执行原地操作，请记得在函数名称后附加下划线。

[表2-5](#table_pointwise_ops)列出了一些常用的逐点操作。

表2-5. 逐点操作

| 操作类型 | 示例函数 |
| --- | --- |
| 基本数学 | `add()`, `div()`, `mul()`, `neg()`, `reciprocal()`, `true_divide()` |
| 截断 | `ceil()`, `clamp()`, `floor()`, `floor_divide()`, `fmod()`, `frac()`, `lerp()`, `remainder()`, `round()`, `sigmoid()`, `trunc()` |
| 复数 | `abs()`, `angle()`, `conj()`, `imag()`, `real()` |
| 三角函数 | `acos()`, `asin()`, `atan()`, `cos()`, `cosh()`, `deg2rad()`, `rad2deg()`, `sin()`, `sinh()`, `tan()`, `tanh()` |
| 指数和对数 | `exp()`, `expm1()`, `log()`, `log10()`, `log1p()`, `log2()`, `logaddexp()`, `pow()`, `rsqrt()`, `sqrt()`, `square()` |
| 逻辑 | `logical_and()`, `logical_not()`, `logical_or()`, `logical_xor()` |
| 累积数学 | `addcdiv()`, `addcmul()` |
| 位运算符 | `bitwise_not()`, `bitwise_and()`, `bitwise_or()`, `bitwise_xor()` |
| 错误函数 | `erf()`, `erfc()`, `erfinv()` |
| 伽玛函数 | `digamma()`, `lgamma()`, `mvlgamma()`, `polygamma()` |

使用Python提示或参考PyTorch文档以获取有关函数使用的详细信息。请注意，`true_divide()`首先将张量数据转换为浮点数，应在将整数除以以获得真实除法结果时使用。

###### 注意

大多数张量操作可以使用三种不同的语法。张量支持运算符重载，因此您可以直接使用运算符，例如`z = x + y`。虽然您也可以使用PyTorch函数如`torch.add()`来执行相同的操作，但这较少见。最后，您可以使用下划线(_)后缀执行原地操作。函数`y.add_(x)`可以实现相同的结果，但它们将存储在`y`中。

第二类数学函数是 *缩减操作*。 缩减操作将一堆数字减少到一个数字或一组较小的数字。 也就是说，它们减少了张量的 *维度* 或 *秩*。 缩减操作包括查找最大值或最小值以及许多统计计算的函数，例如查找平均值或标准差。

这些操作在深度学习中经常使用。 例如，深度学习分类通常使用 `argmax()` 函数将 softmax 输出缩减为主导类。

[表2-6](#table_reduction_ops) 列出了一些常用的缩减操作。

表2-6\. 缩减操作

| 函数 | 描述 |
| --- | --- |
| `torch.`**`argmax`**`(`*`input, dim, keepdim=False, out=None`*`)` | 返回所有元素中最大值的索引，或者如果指定了维度，则只返回一个维度上的索引 |
| `torch.`**`argmin`**`(`*`input, dim, keepdim=False, out=None`*`)` | 返回所有元素中最小值的索引，或者如果指定了维度，则只返回一个维度上的索引 |
| `torch.`**`dist`**`(`*`input, dim, keepdim=False, out=None`*`)` | 计算两个张量的 *p*-范数 |
| `torch.`**`logsumexp`**`(`*`input, dim, keepdim=False, out=None`*`)` | 计算给定维度中输入张量的每行的指数和的对数 |
| `torch.`**`mean`**`(`*`input, dim, keepdim=False, out=None`*`)` | 计算所有元素的平均值，或者如果指定了维度，则只计算一个维度上的平均值 |
| `torch.`**`median`**`(`*`input, dim, keepdim=False, out=None`*`)` | 计算所有元素的中位数或中间值，或者如果指定了维度，则只计算一个维度上的中位数 |
| `torch.`**`mode`**`(`*`input, dim, keepdim=False, out=None`*`)` | 计算所有元素的众数或最频繁出现的值，或者如果指定了维度，则只计算一个维度上的值 |
| `torch.`**`norm`**`(`*`input, p='fro', dim=None,`* *`keepdim=False,`* *`out=None, dtype=None`*`)` | 计算所有元素的矩阵或向量范数，或者如果指定了维度，则只计算一个维度上的范数 |
| `torch.`**`prod`**`(`*`input, dim, keepdim=False, dtype=None`*`)` | 计算所有元素的乘积，或者如果指定了维度，则只计算输入张量的每行的乘积 |
| `torch.`**`std`**`(`*`input, dim, keepdim=False, out=None`*`)` | 计算所有元素的标准差，或者如果指定了维度，则只计算一个维度上的标准差 |
| `torch.`**`std_mean`**`(`*`input, unbiased=True`*`)` | 计算所有元素的标准差和平均值，或者如果指定了维度，则只计算一个维度上的标准差和平均值 |
| `torch.`**`sum`**`(`*`input, dim, keepdim=False, out=None`*`)` | 计算所有元素的和，或者如果指定了维度，则只计算一个维度上的和 |
| `torch.`**`unique`**`(`*`input, dim, keepdim=False, out=None`*`)` | 在整个张量中删除重复项，或者如果指定了维度，则只删除一个维度上的重复项 |
| `torch.`**`unique_​con⁠⁠secu⁠⁠tive`**`(`*`input, dim, keepdim=False, out=None`*`)` | 类似于 `torch.unique()`，但仅删除连续的重复项 |
| `torch.`**`var`**`(`*`input, dim, keepdim=False, out=None`*`)` | 计算所有元素的方差，或者如果指定了维度，则只计算一个维度上的方差 |
| `torch.`**`var_mean`**`(`*`input, dim, keepdim=False, out=None`*`)` | 计算所有元素的平均值和方差，或者如果指定了维度，则只计算一个维度上的平均值和方差 |

请注意，许多这些函数接受 `dim` 参数，该参数指定多维张量的缩减维度。 这类似于 NumPy 中的 `axis` 参数。 默认情况下，当未指定 `dim` 时，缩减会跨所有维度进行。 指定 `dim = 1` 将在每行上计算操作。 例如，`torch.mean(x,1)` 将计算张量 `x` 中每行的平均值。

###### 提示

将方法链接在一起是常见的。 例如，`torch.rand(2,2).max().item()` 创建一个 2 × 2 的随机浮点数张量，找到最大值，并从结果张量中返回值本身。

接下来，我们将看一下PyTorch的*比较函数*。比较函数通常比较张量中的所有值，或将一个张量的值与另一个张量的值进行比较。它们可以根据每个元素的值返回一个充满布尔值的张量，例如`torch.eq()`或`torch.is_boolean()`。还有一些函数可以找到最大或最小值，对张量值进行排序，返回张量元素的顶部子集等。

| `torch.`**`svd`**`()` | 执行奇异值分解 |

表2-7\. 比较操作

| 操作类型 | 示例函数 |
| --- | --- |
| 将张量与其他张量进行比较 | `eq()`, `ge()`, `gt()`, `le()`, `lt()`, `ne()` 或 `==`, `>`, `>=`, `<`, `<=`, `!=`, 分别 |
| 测试张量状态或条件 | `isclose()`, `isfinite()`, `isinf()`, `isnan()` |
| 返回整个张量的单个布尔值 | `allclose()`, `equal()` |
| 查找整个张量或沿给定维度的值 | `argsort()`, `kthvalue()`, `max()`, `min()`, `sort()`, `topk()` |

比较函数似乎很简单；然而，有一些关键点需要记住。常见的陷阱包括以下内容：

+   `torch.eq()`函数或`==`返回一个相同大小的张量，每个元素都有一个布尔结果。`torch.equal()`函数测试张量是否具有相同的大小，如果张量中的所有元素都相等，则返回一个单个布尔值。

+   函数`torch.allclose()`也会返回一个单个布尔值，如果所有元素都接近指定值。

接下来我们将看一下*线性代数函数*。线性代数函数促进矩阵运算，对于深度学习计算非常重要。

许多计算，包括梯度下降和优化算法，使用线性代数来实现它们的计算。PyTorch支持一组强大的内置线性代数操作，其中许多基于基本线性代数子程序（BLAS）和线性代数包（LAPACK）标准化库。

[表 2-8](#table_linalg_ops) 列出了一些常用的线性代数操作。

表2-8\. 线性代数操作

| 函数 | 描述 |
| --- | --- |
| `torch.`**`matmul`**`()` | 计算两个张量的矩阵乘积；支持广播 |
| `torch.`**`chain_matmul`**`()` | 计算*N*个张量的矩阵乘积 |
| `torch.`**`mm`**`()` | 计算两个张量的矩阵乘积（如果需要广播，请使用`matmul()`） |
| `torch.`**`addmm`**`()` | 计算两个张量的矩阵乘积并将其添加到输入中 |
| `torch.`**`bmm`**`()` | 计算一批矩阵乘积 |
| `torch.`**`addbmm`**`()` | 计算一批矩阵乘积并将其添加到输入中 |
| `torch.`**`baddbmm`**`()` | 计算一批矩阵乘积并将其添加到输入批次 |
| `torch.`**`mv`**`()` | 计算矩阵和向量的乘积 |
| `torch.`**`addmv`**`()` | 计算矩阵和向量的乘积并将其添加到输入中 |
| `torch.`**`matrix_power`** | 返回张量的*n*次幂（对于方阵） |
| `torch.`**`eig`**`()` | 找到实方阵的特征值和特征向量 |
| `torch.`**`inverse`**`()` | 计算方阵的逆 |
| `torch.`**`det`**`()` | 计算矩阵或一批矩阵的行列式 |
| `torch.`**`logdet`**`()` | 计算矩阵或一批矩阵的对数行列式 |
| `torch.`**`dot`**`()` | 计算两个张量的内积 |
| `torch.`**`addr`**`()` | 计算两个张量的外积并将其添加到输入中 |
| `torch.`**`solve`**`()` | 返回线性方程组的解 |
| [表 2-7](#table_comparison_ops) 列出了一些常用的比较函数供参考。 |
| `torch.`**`pca_lowrank`**`()` | 执行线性主成分分析 |
| `torch.`**`cholesky`**`()` | 计算Cholesky分解 |
| `torch.`**`cholesky_inverse`**`()` | 计算对称正定矩阵的逆并返回Cholesky因子 |
| `torch.`**`cholesky_solve`**`()` | 使用Cholesky因子解线性方程组 |

[表2-8](#table_linalg_ops)中的函数范围从矩阵乘法和批量计算函数到求解器。重要的是要指出，矩阵乘法与`torch.mul()`或*运算符的逐点乘法不同。

本书不涵盖完整的线性代数研究，但在进行特征降维或开发自定义深度学习算法时，您可能会发现访问一些线性代数函数很有用。请参阅[PyTorch线性代数文档](https://pytorch.tips/linear-algebra)以获取可用函数的完整列表以及如何使用它们的更多详细信息。

我们将考虑的最后一类数学运算是*光谱和其他数学运算*。根据感兴趣的领域，这些函数可能对数据转换或分析有用。例如，光谱运算如快速傅里叶变换（FFT）在计算机视觉或数字信号处理应用中可能起重要作用。

[表2-9](#table_other_ops)列出了一些用于频谱分析和其他数学运算的内置操作。

表2-9. 光谱和其他数学运算

| 操作类型 | 示例函数 |
| --- | --- |
| 快速、逆、短时傅里叶变换 | `fft()`, `ifft()`, `stft()` |
| 实到复FFT和复到实逆FFT（IFFT） | `rfft()`, `irfft()` |
| 窗口算法 | `bartlett_window()`, `blackman_window()`, `hamming_window()`, `hann_window()` |
| 直方图和箱计数 | `histc()`, `bincount()` |
| 累积操作 | `cummax()`, `cummin()`, `cumprod()`, `cumsum()`, `trace()`（对角线之和），`einsum()`（使用爱因斯坦求和的乘积之和） |
| 标准化函数 | `cdist()`, `renorm()` |
| 叉积、点积和笛卡尔积 | `cross()`, `tensordot()`, `cartesian_prod()` |
| 创建对角张量的函数，其元素为输入张量的元素 | `diag()`, `diag_embed()`, `diag_flat()`, `diagonal()` |
| 爱因斯坦求和 | `einsum()` |
| 矩阵降维和重构函数 | `flatten()`, `flip()`, `rot90()`, `repeat_interleave()`, `meshgrid()`, `roll()`, `combinations()` |
| 返回下三角形或上三角形及其索引的函数 | `tril()`, `tril_indices`, `triu()`, `triu_indices()` |

## 自动微分（Autograd）

一个函数，`backward()`，值得在自己的子节中调用，因为它是PyTorch在深度学习开发中如此强大的原因。`backward()`函数使用PyTorch的自动微分包`torch.autograd`，根据链式法则对张量进行微分和计算梯度。

这是自动微分的一个简单示例。我们定义一个函数，*f* = sum(*x*²)，其中x是一个变量矩阵。如果我们想要找到矩阵中每个变量的 *df* / *dx*，我们需要为张量*x*设置`requires_grad = True`标志，如下面的代码所示：

```py
x = torch.tensor([[1,2,3],[4,5,6]],
         dtype=torch.float, requires_grad=True)
print(x)
# out:
# tensor([[1., 2., 3.],
#         [4., 5., 6.]], requires_grad=True)

f = x.pow(2).sum()
print(f)
# tensor(91., grad_fn=<SumBackward0>)

f.backward()
print(x.grad) # df/dx = 2x
# tensor([[ 2.,  4.,  6.],
#         [ 8., 10., 12.]])
```

`f.backward()`函数对*f*进行微分，并将*df* / *dx*存储在`x.grad`属性中。对微积分微分方程的快速回顾将告诉我们*f*相对于*x*的导数，*df* / *dx* = 2*x*。对*x*的值评估*df* / *dx*的结果显示为输出。

###### 注意

只有浮点`dtype`的张量可以需要梯度。

训练神经网络需要我们在反向传播中计算权重梯度。随着我们的神经网络变得更深更复杂，这个功能可以自动化复杂的计算。有关autograd工作原理的更多信息，请参阅[Autograd教程](https://pytorch.tips/autograd-explained)。

本章提供了一个快速参考，用于创建张量和执行操作。现在您已经对张量有了良好的基础，我们将重点讨论如何使用张量和PyTorch来进行深度学习研究。在下一章中，我们将回顾深度学习开发过程，然后开始编写代码。
