# 附录 C：特殊数据结构

在本附录中，我们将快速查看 TensorFlow 支持的数据结构，超出了常规的浮点或整数张量。这包括字符串、不规则张量、稀疏张量、张量数组、集合和队列。

# 字符串

张量可以保存字节字符串，这在自然语言处理中特别有用（请参阅第十六章）：

```py
>>> tf.constant(b"hello world")
<tf.Tensor: shape=(), dtype=string, numpy=b'hello world'>
```

如果尝试构建一个包含 Unicode 字符串的张量，TensorFlow 会自动将其编码为 UTF-8：

```py
>>> tf.constant("café")
<tf.Tensor: shape=(), dtype=string, numpy=b'caf\xc3\xa9'>
```

还可以创建表示 Unicode 字符串的张量。只需创建一个 32 位整数数组，每个整数代表一个单个 Unicode 码点：⁠¹

```py
>>> u = tf.constant([ord(c) for c in "café"])
>>> u
<tf.Tensor: shape=(4,), [...], numpy=array([ 99,  97, 102, 233], dtype=int32)>
```

###### 注意

在类型为`tf.string`的张量中，字符串长度不是张量形状的一部分。换句话说，字符串被视为原子值。但是，在 Unicode 字符串张量（即 int32 张量）中，字符串的长度*是*张量形状的一部分。

`tf.strings`包含几个函数来操作字符串张量，例如`length()`用于计算字节字符串中的字节数（或者如果设置`unit="UTF8_CHAR"`，则计算代码点的数量），`unicode_encode()`用于将 Unicode 字符串张量（即 int32 张量）转换为字节字符串张量，`unicode_decode()`用于执行相反操作：

```py
>>> b = tf.strings.unicode_encode(u, "UTF-8")
>>> b
<tf.Tensor: shape=(), dtype=string, numpy=b'caf\xc3\xa9'>
>>> tf.strings.length(b, unit="UTF8_CHAR")
<tf.Tensor: shape=(), dtype=int32, numpy=4>
>>> tf.strings.unicode_decode(b, "UTF-8")
<tf.Tensor: shape=(4,), [...], numpy=array([ 99,  97, 102, 233], dtype=int32)>
```

您还可以操作包含多个字符串的张量：

```py
>>> p = tf.constant(["Café", "Coffee", "caffè", "咖啡"])
>>> tf.strings.length(p, unit="UTF8_CHAR")
<tf.Tensor: shape=(4,), dtype=int32, numpy=array([4, 6, 5, 2], dtype=int32)>
>>> r = tf.strings.unicode_decode(p, "UTF8")
>>> r
<tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [99, 97,
102, 102, 232], [21654, 21857]]>
```

请注意，解码的字符串存储在`RaggedTensor`中。那是什么？

# 不规则张量

*不规则张量*是一种特殊类型的张量，表示不同大小数组的列表。更一般地说，它是一个具有一个或多个*不规则维度*的张量，意味着切片可能具有不同长度的维度。在不规则张量`r`中，第二个维度是一个不规则维度。在所有不规则张量中，第一个维度始终是一个常规维度（也称为*均匀维度*）。

不规则张量`r`的所有元素都是常规张量。例如，让我们看看不规则张量的第二个元素：

```py
>>> r[1]
<tf.Tensor: [...], numpy=array([ 67, 111, 102, 102, 101, 101], dtype=int32)>
```

`tf.ragged`包含几个函数来创建和操作不规则张量。让我们使用`tf.ragged.constant()`创建第二个不规则张量，并沿着轴 0 连接它与第一个不规则张量：

```py
>>> r2 = tf.ragged.constant([[65, 66], [], [67]])
>>> tf.concat([r, r2], axis=0)
<tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [99, 97,
102, 102, 232], [21654, 21857], [65, 66], [], [67]]>
```

结果并不太令人惊讶：`r2`中的张量是沿着轴 0 在`r`中的张量之后附加的。但是如果我们沿着轴 1 连接`r`和另一个不规则张量呢？

```py
>>> r3 = tf.ragged.constant([[68, 69, 70], [71], [], [72, 73]])
>>> print(tf.concat([r, r3], axis=1))
<tf.RaggedTensor [[67, 97, 102, 233, 68, 69, 70], [67, 111, 102, 102, 101, 101,
71], [99, 97, 102, 102, 232], [21654, 21857, 72, 73]]>
```

这次，请注意`r`中的第*i*个张量和`r3`中的第*i*个张量被连接。现在这更不寻常，因为所有这些张量都可以具有不同的长度。

如果调用`to_tensor()`方法，不规则张量将转换为常规张量，用零填充较短的张量以获得相等长度的张量（您可以通过设置`default_value`参数更改默认值）：

```py
>>> r.to_tensor()
<tf.Tensor: shape=(4, 6), dtype=int32, numpy=
array([[   67,    97,   102,   233,     0,     0],
 [   67,   111,   102,   102,   101,   101],
 [   99,    97,   102,   102,   232,     0],
 [21654, 21857,     0,     0,     0,     0]], dtype=int32)>
```

许多 TF 操作支持不规则张量。有关完整列表，请参阅`tf.RaggedTensor`类的文档。

# 稀疏张量

TensorFlow 还可以高效地表示*稀疏张量*（即包含大多数零的张量）。只需创建一个`tf.SparseTensor`，指定非零元素的索引和值以及张量的形状。索引必须按“读取顺序”（从左到右，从上到下）列出。如果不确定，只需使用`tf.sparse.reorder()`。您可以使用`tf.sparse.to_dense()`将稀疏张量转换为密集张量（即常规张量）：

```py
>>> s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
...                     values=[1., 2., 3.],
...                     dense_shape=[3, 4])
...
>>> tf.sparse.to_dense(s)
<tf.Tensor: shape=(3, 4), dtype=float32, numpy=
array([[0., 1., 0., 0.],
 [2., 0., 0., 0.],
 [0., 0., 0., 3.]], dtype=float32)>
```

请注意，稀疏张量不支持与密集张量一样多的操作。例如，您可以将稀疏张量乘以任何标量值，得到一个新的稀疏张量，但是您不能将标量值添加到稀疏张量中，因为这不会返回一个稀疏张量：

```py
>>> s * 42.0
<tensorflow.python.framework.sparse_tensor.SparseTensor at 0x7f84a6749f10>
>>> s + 42.0
[...] TypeError: unsupported operand type(s) for +: 'SparseTensor' and 'float'
```

# 张量数组

`tf.TensorArray`表示一个张量列表。这在包含循环的动态模型中可能很方便，用于累积结果并稍后计算一些统计数据。您可以在数组中的任何位置读取或写入张量：

```py
array = tf.TensorArray(dtype=tf.float32, size=3)
array = array.write(0, tf.constant([1., 2.]))
array = array.write(1, tf.constant([3., 10.]))
array = array.write(2, tf.constant([5., 7.]))
tensor1 = array.read(1)  # => returns (and zeros out!) tf.constant([3., 10.])
```

默认情况下，读取一个项目也会用相同形状但全是零的张量替换它。如果不想要这样，可以将`clear_after_read`设置为`False`。

###### 警告

当您向数组写入时，必须将输出分配回数组，就像这个代码示例中所示。如果不这样做，尽管您的代码在急切模式下可以正常工作，但在图模式下会出错（这些模式在第十二章中讨论）。

默认情况下，`TensorArray`具有在创建时设置的固定大小。或者，您可以设置`size=0`和`dynamic_size=True`，以便在需要时自动增长数组。但是，这会影响性能，因此如果您事先知道`size`，最好使用固定大小数组。您还必须指定`dtype`，并且所有元素必须与写入数组的第一个元素具有相同的形状。

您可以通过调用`stack()`方法将所有项目堆叠到常规张量中：

```py
>>> array.stack()
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[1., 2.],
 [0., 0.],
 [5., 7.]], dtype=float32)>
```

# 集合

TensorFlow 支持整数或字符串的集合（但不支持浮点数）。它使用常规张量表示集合。例如，集合`{1, 5, 9}`只是表示为张量`[[1, 5, 9]]`。请注意，张量必须至少有两个维度，并且集合必须在最后一个维度中。例如，`[[1, 5, 9], [2, 5, 11]]`是一个包含两个独立集合的张量：`{1, 5, 9}`和`{2, 5, 11}`。

`tf.sets`包含几个用于操作集合的函数。例如，让我们创建两个集合并计算它们的并集（结果是一个稀疏张量，因此我们调用`to_dense()`来显示它）：

```py
>>> a = tf.constant([[1, 5, 9]])
>>> b = tf.constant([[5, 6, 9, 11]])
>>> u = tf.sets.union(a, b)
>>> u
<tensorflow.python.framework.sparse_tensor.SparseTensor at 0x132b60d30>
>>> tf.sparse.to_dense(u)
<tf.Tensor: [...], numpy=array([[ 1,  5,  6,  9, 11]], dtype=int32)>
```

还可以同时计算多对集合的并集。如果某些集合比其他集合短，必须用填充值（例如 0）填充它们：

```py
>>> a = tf.constant([[1, 5, 9], [10, 0, 0]])
>>> b = tf.constant([[5, 6, 9, 11], [13, 0, 0, 0]])
>>> u = tf.sets.union(a, b)
>>> tf.sparse.to_dense(u)
<tf.Tensor: [...] numpy=array([[ 1,  5,  6,  9, 11],
 [ 0, 10, 13,  0,  0]], dtype=int32)>
```

如果您想使用不同的填充值，比如-1，那么在调用`to_dense()`时必须设置`default_value=-1`（或您喜欢的值）。

###### 警告

默认的`default_value`是 0，所以在处理字符串集合时，必须设置这个参数（例如，设置为空字符串）。

`tf.sets`中还有其他可用的函数，包括`difference()`、`intersection()`和`size()`，它们都是不言自明的。如果要检查一个集合是否包含某些给定值，可以计算该集合和值的交集。如果要向集合添加一些值，可以计算集合和值的并集。

# 队列

队列是一种数据结构，您可以将数据记录推送到其中，然后再将它们取出。TensorFlow 在`tf.queue`包中实现了几种类型的队列。在实现高效的数据加载和预处理流水线时，它们曾经非常重要，但是 tf.data API 基本上使它们变得无用（也许在一些罕见情况下除外），因为使用起来更简单，并提供了构建高效流水线所需的所有工具。为了完整起见，让我们快速看一下它们。

最简单的队列是先进先出（FIFO）队列。要构建它，您需要指定它可以包含的记录的最大数量。此外，每个记录都是张量的元组，因此您必须指定每个张量的类型，以及可选的形状。例如，以下代码示例创建了一个最多包含三条记录的 FIFO 队列，每条记录包含一个 32 位整数和一个字符串的元组。然后将两条记录推送到队列中，查看大小（此时为 2），并取出一条记录：

```py
>>> q = tf.queue.FIFOQueue(3, [tf.int32, tf.string], shapes=[(), ()])
>>> q.enqueue([10, b"windy"])
>>> q.enqueue([15, b"sunny"])
>>> q.size()
<tf.Tensor: shape=(), dtype=int32, numpy=2>
>>> q.dequeue()
[<tf.Tensor: shape=(), dtype=int32, numpy=10>,
 <tf.Tensor: shape=(), dtype=string, numpy=b'windy'>]
```

还可以使用`enqueue_many()`和`dequeue_many()`一次入队和出队多个记录（要使用`dequeue_many()`，必须在创建队列时指定`shapes`参数，就像我们之前做的那样）：

```py
>>> q.enqueue_many([[13, 16], [b'cloudy', b'rainy']])
>>> q.dequeue_many(3)
[<tf.Tensor: [...], numpy=array([15, 13, 16], dtype=int32)>,
 <tf.Tensor: [...], numpy=array([b'sunny', b'cloudy', b'rainy'], dtype=object)>]
```

其他队列类型包括：

`PaddingFIFOQueue`

与`FIFOQueue`相同，但其`dequeue_many()`方法支持出队不同形状的多个记录。它会自动填充最短的记录，以确保批次中的所有记录具有相同的形状。

`PriorityQueue`

一个按优先级顺序出队记录的队列。优先级必须作为每个记录的第一个元素包含在其中，是一个 64 位整数。令人惊讶的是，优先级较低的记录将首先出队。具有相同优先级的记录将按照 FIFO 顺序出队。

`RandomShuffleQueue`

一个记录以随机顺序出队的队列。在 tf.data 出现之前，这对实现洗牌缓冲区很有用。

如果队列已满并且您尝试入队另一个记录，则`enqueue*()`方法将冻结，直到另一个线程出队一条记录。同样，如果队列为空并且您尝试出队一条记录，则`dequeue*()`方法将冻结，直到另一个线程将记录推送到队列中。

如果您不熟悉 Unicode 代码点，请查看[*https://homl.info/unicode*](https://homl.info/unicode)。
