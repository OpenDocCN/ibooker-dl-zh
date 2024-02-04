# 附录D。TensorFlow图

在本附录中，我们将探索由TF函数生成的图形（请参阅[第12章](ch12.html#tensorflow_chapter)）。

# TF函数和具体函数

TF函数是多态的，意味着它们支持不同类型（和形状）的输入。例如，考虑以下`tf_cube()`函数：

```py
@tf.function
def tf_cube(x):
    return x ** 3
```

每次您调用一个TF函数并使用新的输入类型或形状组合时，它会生成一个新的*具体函数*，具有为这种特定组合专门优化的图形。这样的参数类型和形状组合被称为*输入签名*。如果您使用它之前已经见过的输入签名调用TF函数，它将重用之前生成的具体函数。例如，如果您调用`tf_cube(tf.constant(3.0))`，TF函数将重用用于`tf_cube(tf.constant(2.0))`（对于float32标量张量）的相同具体函数。但是，如果您调用`tf_cube(tf.constant([2.0]))`或`tf_cube(tf.constant([3.0]))`（对于形状为[1]的float32张量），它将生成一个新的具体函数，对于`tf_cube(tf.constant([[1.0, 2.0], [3.0, 4.0]]))`（对于形状为[2, 2]的float32张量），它将生成另一个新的具体函数。您可以通过调用TF函数的`get_concrete_function()`方法来获取特定输入组合的具体函数。然后可以像普通函数一样调用它，但它只支持一个输入签名（在此示例中为float32标量张量）：

```py
>>> concrete_function = tf_cube.get_concrete_function(tf.constant(2.0))
>>> concrete_function
<ConcreteFunction tf_cube(x) at 0x7F84411F4250>
>>> concrete_function(tf.constant(2.0))
<tf.Tensor: shape=(), dtype=float32, numpy=8.0>
```

[图D-1](#tf_function_diagram)显示了`tf_cube()` TF函数，在我们调用`tf_cube(2)`和`tf_cube(tf.constant(2.0))`之后：生成了两个具体函数，每个签名一个，每个具有自己优化的*函数图*（`FuncGraph`）和自己的*函数定义*（`FunctionDef`）。函数定义指向与函数的输入和输出对应的图的部分。在每个`FuncGraph`中，节点（椭圆形）表示操作（例如，幂运算，常量，或用于参数的占位符如`x`），而边（操作之间的实箭头）表示将在图中流动的张量。左侧的具体函数专门用于`x=2`，因此TensorFlow成功将其简化为始终输出8（请注意，函数定义甚至没有输入）。右侧的具体函数专门用于float32标量张量，无法简化。如果我们调用`tf_cube(tf.constant(5.0))`，将调用第二个具体函数，`x`的占位符操作将输出5.0，然后幂运算将计算`5.0 ** 3`，因此输出将为125.0。

![mls3 ad01](assets/mls3_ad01.png)

###### 图D-1。`tf_cube()` TF函数，及其`ConcreteFunction`和它们的`FuncGraph`

这些图中的张量是*符号张量*，意味着它们没有实际值，只有数据类型、形状和名称。它们代表将在实际值被馈送到占位符`x`并执行图形后流经图形的未来张量。符号张量使得可以预先指定如何连接操作，并且还允许TensorFlow递归推断所有张量的数据类型和形状，鉴于它们的输入的数据类型和形状。

现在让我们继续窥探底层，并看看如何访问函数定义和函数图，以及如何探索图的操作和张量。

# 探索函数定义和图形

您可以使用`graph`属性访问具体函数的计算图，并通过调用图的`get_operations()`方法获取其操作列表：

```py
>>> concrete_function.graph
<tensorflow.python.framework.func_graph.FuncGraph at 0x7f84411f4790>
>>> ops = concrete_function.graph.get_operations()
>>> ops
[<tf.Operation 'x' type=Placeholder>,
 <tf.Operation 'pow/y' type=Const>,
 <tf.Operation 'pow' type=Pow>,
 <tf.Operation 'Identity' type=Identity>]
```

在这个例子中，第一个操作代表输入参数 `x`（它被称为 *占位符*），第二个“操作”代表常数 `3`，第三个操作代表幂运算（`**`），最后一个操作代表这个函数的输出（它是一个恒等操作，意味着它不会做任何比幂运算输出的更多的事情⁠^([1](app04.html#idm45720155973312)））。每个操作都有一个输入和输出张量的列表，您可以通过操作的 `inputs` 和 `outputs` 属性轻松访问。例如，让我们获取幂运算的输入和输出列表：

```py
>>> pow_op = ops[2]
>>> list(pow_op.inputs)
[<tf.Tensor 'x:0' shape=() dtype=float32>,
 <tf.Tensor 'pow/y:0' shape=() dtype=float32>]
>>> pow_op.outputs
[<tf.Tensor 'pow:0' shape=() dtype=float32>]
```

这个计算图在 [图 D-2](#computation_graph_diagram) 中表示。

![mls3 ad02](assets/mls3_ad02.png)

###### 图 D-2\. 计算图示例

请注意每个操作都有一个名称。它默认为操作的名称（例如，`"pow"`），但当调用操作时您可以手动定义它（例如，`tf.pow(x, 3, name="other_name")`）。如果名称已经存在，TensorFlow 会自动添加一个唯一的索引（例如，`"pow_1"`，`"pow_2"` 等）。每个张量也有一个唯一的名称：它总是输出该张量的操作的名称，如果它是操作的第一个输出，则为 `:0`，如果它是第二个输出，则为 `:1`，依此类推。您可以使用图的 `get_operation_by_name()` 或 `get_tensor_by_name()` 方法按名称获取操作或张量：

```py
>>> concrete_function.graph.get_operation_by_name('x')
<tf.Operation 'x' type=Placeholder>
>>> concrete_function.graph.get_tensor_by_name('Identity:0')
<tf.Tensor 'Identity:0' shape=() dtype=float32>
```

具体函数还包含函数定义（表示为协议缓冲区⁠^([2](app04.html#idm45720155848416)）），其中包括函数的签名。这个签名允许具体函数知道要用输入值填充哪些占位符，以及要返回哪些张量：

```py
>>> concrete_function.function_def.signature
name: "__inference_tf_cube_3515903"
input_arg {
 name: "x"
 type: DT_FLOAT
}
output_arg {
 name: "identity"
 type: DT_FLOAT
}
```

现在让我们更仔细地看一下跟踪。

# 更仔细地看一下跟踪

让我们调整 `tf_cube()` 函数以打印其输入：

```py
@tf.function
def tf_cube(x):
    print(f"x = {x}")
    return x ** 3
```

现在让我们调用它：

```py
>>> result = tf_cube(tf.constant(2.0))
x = Tensor("x:0", shape=(), dtype=float32)
>>> result
<tf.Tensor: shape=(), dtype=float32, numpy=8.0>
```

`result` 看起来不错，但看看打印出来的内容：`x` 是一个符号张量！它有一个形状和数据类型，但没有值。而且它有一个名称（`"x:0"`）。这是因为 `print()` 函数不是一个 TensorFlow 操作，所以它只会在 Python 函数被跟踪时运行，这发生在图模式下，参数被替换为符号张量（相同类型和形状，但没有值）。由于 `print()` 函数没有被捕获到图中，所以下一次我们用 float32 标量张量调用 `tf_cube()` 时，什么也不会被打印：

```py
>>> result = tf_cube(tf.constant(3.0))
>>> result = tf_cube(tf.constant(4.0))
```

但是，如果我们用不同类型或形状的张量，或者用一个新的 Python 值调用 `tf_cube()`，函数将再次被跟踪，因此 `print()` 函数将被调用：

```py
>>> result = tf_cube(2)  # new Python value: trace!
x = 2
>>> result = tf_cube(3)  # new Python value: trace!
x = 3
>>> result = tf_cube(tf.constant([[1., 2.]]))  # new shape: trace!
x = Tensor("x:0", shape=(1, 2), dtype=float32)
>>> result = tf_cube(tf.constant([[3., 4.], [5., 6.]]))  # new shape: trace!
x = Tensor("x:0", shape=(None, 2), dtype=float32)
>>> result = tf_cube(tf.constant([[7., 8.], [9., 10.]]))  # same shape: no trace
```

###### 警告

如果您的函数具有 Python 副作用（例如，将一些日志保存到磁盘），请注意此代码只会在函数被跟踪时运行（即每次用新的输入签名调用 TF 函数时）。最好假设函数可能在调用 TF 函数时随时被跟踪（或不被跟踪）。

在某些情况下，您可能希望将 TF 函数限制为特定的输入签名。例如，假设您知道您只会用 28 × 28 像素图像的批次调用 TF 函数，但是批次的大小会有很大的不同。您可能不希望 TensorFlow 为每个批次大小生成不同的具体函数，或者依赖它自行决定何时使用 `None`。在这种情况下，您可以像这样指定输入签名：

```py
@tf.function(input_signature=[tf.TensorSpec([None, 28, 28], tf.float32)])
def shrink(images):
    return images[:, ::2, ::2]  # drop half the rows and columns
```

这个 TF 函数将接受任何形状为 [*, 28, 28] 的 float32 张量，并且每次都会重用相同的具体函数：

```py
img_batch_1 = tf.random.uniform(shape=[100, 28, 28])
img_batch_2 = tf.random.uniform(shape=[50, 28, 28])
preprocessed_images = shrink(img_batch_1)  # works fine, traces the function
preprocessed_images = shrink(img_batch_2)  # works fine, same concrete function
```

然而，如果您尝试用 Python 值调用这个 TF 函数，或者用意外的数据类型或形状的张量调用它，您将会得到一个异常：

```py
img_batch_3 = tf.random.uniform(shape=[2, 2, 2])
preprocessed_images = shrink(img_batch_3)  # ValueError! Incompatible inputs
```

# 使用 AutoGraph 捕获控制流

如果您的函数包含一个简单的 `for` 循环，您期望会发生什么？例如，让我们编写一个函数，通过连续添加 1 来将 10 添加到其输入中：

```py
@tf.function
def add_10(x):
    for i in range(10):
        x += 1
    return x
```

它运行正常，但当我们查看它的图时，我们发现它不包含循环：它只包含 10 个加法操作！

```py
>>> add_10(tf.constant(0))
<tf.Tensor: shape=(), dtype=int32, numpy=15>
>>> add_10.get_concrete_function(tf.constant(0)).graph.get_operations()
[<tf.Operation 'x' type=Placeholder>, [...],
 <tf.Operation 'add' type=AddV2>, [...],
 <tf.Operation 'add_1' type=AddV2>, [...],
 <tf.Operation 'add_2' type=AddV2>, [...],
 [...]
 <tf.Operation 'add_9' type=AddV2>, [...],
 <tf.Operation 'Identity' type=Identity>]
```

实际上这是有道理的：当函数被跟踪时，循环运行了10次，因此`x += 1`操作运行了10次，并且由于它处于图模式下，它在图中记录了这个操作10次。您可以将这个`for`循环看作是一个在创建图表时被展开的“静态”循环。

如果您希望图表包含一个“动态”循环（即在执行图表时运行的循环），您可以手动使用`tf.while_loop()`操作创建一个，但这并不直观（请参见第12章笔记本的“使用AutoGraph捕获控制流”部分以获取示例）。相反，使用TensorFlow的*AutoGraph*功能要简单得多，详见[第12章](ch12.html#tensorflow_chapter)。AutoGraph实际上是默认激活的（如果您需要关闭它，可以在`tf.function()`中传递`autograph=False`）。因此，如果它是开启的，为什么它没有捕获`add_10()`函数中的`for`循环呢？它只捕获对`tf.data.Dataset`对象的张量进行迭代的`for`循环，因此您应该使用`tf.range()`而不是`range()`。这是为了给您选择：

+   如果使用`range()`，`for`循环将是静态的，这意味着仅在跟踪函数时才会执行。循环将被“展开”为每次迭代的一组操作，正如我们所见。

+   如果使用`tf.range()`，循环将是动态的，这意味着它将包含在图表本身中（但在跟踪期间不会运行）。

让我们看看如果在`add_10()`函数中将`range()`替换为`tf.range()`时生成的图表：

```py
>>> add_10.get_concrete_function(tf.constant(0)).graph.get_operations()
[<tf.Operation 'x' type=Placeholder>, [...],
 <tf.Operation 'while' type=StatelessWhile>, [...]]
```

如您所见，图现在包含一个`While`循环操作，就好像我们调用了`tf.while_loop()`函数一样。

# 在TF函数中处理变量和其他资源

在TensorFlow中，变量和其他有状态对象，如队列或数据集，被称为*资源*。TF函数对它们进行特殊处理：任何读取或更新资源的操作都被视为有状态的，并且TF函数确保有状态的操作按照它们出现的顺序执行（与无状态操作相反，后者可能并行运行，因此它们的执行顺序不被保证）。此外，当您将资源作为参数传递给TF函数时，它会通过引用传递，因此函数可能会对其进行修改。例如：

```py
counter = tf.Variable(0)

@tf.function
def increment(counter, c=1):
    return counter.assign_add(c)

increment(counter)  # counter is now equal to 1
increment(counter)  # counter is now equal to 2
```

如果查看函数定义，第一个参数被标记为资源：

```py
>>> function_def = increment.get_concrete_function(counter).function_def
>>> function_def.signature.input_arg[0]
name: "counter"
type: DT_RESOURCE
```

还可以在函数外部使用定义的`tf.Variable`，而无需显式将其作为参数传递：

```py
counter = tf.Variable(0)

@tf.function
def increment(c=1):
    return counter.assign_add(c)
```

TF函数将将其视为隐式的第一个参数，因此实际上最终会具有相同的签名（除了参数的名称）。但是，使用全局变量可能会很快变得混乱，因此通常应该将变量（和其他资源）封装在类中。好消息是`@tf.function`也可以很好地与方法一起使用：

```py
class Counter:
    def __init__(self):
        self.counter = tf.Variable(0)

    @tf.function
    def increment(self, c=1):
        return self.counter.assign_add(c)
```

###### 警告

不要使用`=`、`+=`、`-=`或任何其他Python赋值运算符与TF变量。相反，您必须使用`assign()`、`assign_add()`或`assign_sub()`方法。如果尝试使用Python赋值运算符，当调用该方法时将会出现异常。

这种面向对象的方法的一个很好的例子当然是Keras。让我们看看如何在Keras中使用TF函数。

# 使用TF函数与Keras（或不使用）

默认情况下，您在Keras中使用的任何自定义函数、层或模型都将自动转换为TF函数；您无需做任何事情！但是，在某些情况下，您可能希望停用此自动转换——例如，如果您的自定义代码无法转换为TF函数，或者如果您只想调试代码（在急切模式下更容易）。为此，您只需在创建模型或其任何层时传递`dynamic=True`：

```py
model = MyModel(dynamic=True)
```

如果您的自定义模型或层将始终是动态的，可以使用`dynamic=True`调用基类的构造函数：

```py
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(dynamic=True, **kwargs)
        [...]
```

或者，在调用`compile()`方法时传递`run_eagerly=True`：

```py
model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae],
              run_eagerly=True)
```

现在你知道了 TF 函数如何处理多态性（具有多个具体函数），如何使用 AutoGraph 和追踪自动生成图形，图形的样子，如何探索它们的符号操作和张量，如何处理变量和资源，以及如何在 Keras 中使用 TF 函数。

^([1](app04.html#idm45720155973312-marker)) 你可以安全地忽略它 - 它只是为了技术原因而在这里，以确保 TF 函数不会泄漏内部结构。

^([2](app04.html#idm45720155848416-marker)) 在[第13章](ch13.html#data_chapter)中讨论的一种流行的二进制格式。
