# 第8章。队列、线程和读取数据

在本章中，我们介绍了在TensorFlow中使用队列和线程的方法，主要目的是简化读取输入数据的过程。我们展示了如何编写和读取TFRecords，这是高效的TensorFlow文件格式。然后我们演示了队列、线程和相关功能，并在一个完整的工作示例中连接所有要点，展示了一个包括预处理、批处理和训练的图像数据的多线程输入管道。

# 输入管道

当处理可以存储在内存中的小数据集时，比如MNIST图像，将所有数据加载到内存中，然后使用feeding将数据推送到TensorFlow图中是合理的。然而，对于更大的数据集，这可能变得难以管理。处理这种情况的一个自然范式是将数据保留在磁盘上，并根据需要加载其中的块（比如用于训练的小批量），这样唯一的限制就是硬盘的大小。

此外，在实践中，一个典型的数据管道通常包括诸如读取具有不同格式的输入文件、更改输入的形状或结构、归一化或进行其他形式的预处理、对输入进行洗牌等步骤，甚至在训练开始之前。

这个过程的很多部分可以轻松地解耦并分解为模块化组件。例如，预处理不涉及训练，因此可以一次性对输入进行预处理，然后将其馈送到训练中。由于我们的训练无论如何都是批量处理示例，原则上我们可以在运行时处理输入批次，从磁盘中读取它们，应用预处理，然后将它们馈送到计算图中进行训练。

然而，这种方法可能是浪费的。因为预处理与训练无关，等待每个批次进行预处理会导致严重的I/O延迟，迫使每个训练步骤（急切地）等待加载和处理数据的小批量。更具可扩展性的做法是预取数据，并使用独立的线程进行加载和处理以及训练。但是，当需要重复读取和洗牌许多保存在磁盘上的文件时，这种做法可能变得混乱，并且需要大量的簿记和技术性来无缝运行。

值得注意的是，即使不考虑预处理，使用在前几章中看到的标准馈送机制（使用`feed_dict`）本身也是浪费的。`feed_dict`会将数据从Python运行时单线程复制到TensorFlow运行时，导致进一步的延迟和减速。我们希望通过某种方式直接将数据读取到本机TensorFlow中，避免这种情况。

为了让我们的生活更轻松（和更快），TensorFlow提供了一套工具来简化这个输入管道过程。主要的构建模块是标准的TensorFlow文件格式，用于编码和解码这种格式的实用工具，数据队列和多线程。

我们将逐一讨论这些关键组件，探索它们的工作原理，并构建一个端到端的多线程输入管道。我们首先介绍TFRecords，这是TensorFlow推荐的文件格式，以后会派上用场。

# TFRecords

数据集当然可以采用许多格式，有时甚至是混合的（比如图像和音频文件）。将输入文件转换为一个统一的格式，无论其原始格式如何，通常是方便和有用的。TensorFlow的默认标准数据格式是TFRecord。TFRecord文件只是一个包含序列化输入数据的二进制文件。序列化基于协议缓冲区（protobufs），简单地说，它通过使用描述数据结构的模式将数据转换为存储，独立于所使用的平台或语言（就像XML一样）。

在我们的设置中，使用 TFRecords（以及 protobufs/二进制文件）相比仅使用原始数据文件有许多优势。这种统一格式允许整洁地组织输入数据，所有相关属性都保持在一起，避免了许多目录和子目录的需求。TFRecord 文件实现了非常快速的处理。所有数据都保存在一个内存块中，而不是分别存储每个输入文件，从而减少了从内存读取数据所需的时间。还值得注意的是，TensorFlow 自带了许多针对 TFRecords 进行优化的实现和工具，使其非常适合作为多线程输入管道的一部分使用。

## 使用 TFRecordWriter 进行写入

我们首先将输入文件写入 TFRecord 格式，以便我们可以处理它们（在其他情况下，我们可能已经将数据存储在这种格式中）。在这个例子中，我们将 MNIST 图像转换为这种格式，但是相同的思想也适用于其他类型的数据。

首先，我们将 MNIST 数据下载到 `save_dir`，使用来自 `tensorflow.contrib.learn` 的实用函数：

```py
from__future__importprint_functionimportosimporttensorflowastffromtensorflow.contrib.learn.python.learn.datasetsimportmnistsave_dir="*`path/to/mnist`*"# Download data to save_dirdata_sets=mnist.read_data_sets(save_dir,dtype=tf.uint8,reshape=False,validation_size=1000)
```

我们下载的数据包括训练、测试和验证图像，每个都在一个单独的 *拆分* 中。我们遍历每个拆分，将示例放入适当的格式，并使用 `TFRecordWriter()` 写入磁盘：

```py
data_splits = ["train","test","validation"]
for d in range(len(data_splits)):
 print("saving " + data_splits[d])
 data_set = data_sets[d]

 filename = os.path.join(save_dir, data_splits[d] + '.tfrecords')
 writer = tf.python_io.TFRecordWriter(filename)
 for index in range(data_set.images.shape[0]):
  image = data_set.images[index].tostring()
  example = tf.train.Example(features=tf.train.Features(feature={
  'height': tf.train.Feature(int64_list=
                 tf.train.Int64List(value=
                 [data_set.images.shape[1]])),
  'width': tf.train.Feature(int64_list=
                 tf.train.Int64List(value =
                 [data_set.images.shape[2]])),
  'depth': tf.train.Feature(int64_list=
                 tf.train.Int64List(value =
                 [data_set.images.shape[3]])),
  'label': tf.train.Feature(int64_list=
                 tf.train.Int64List(value =
                 [int(data_set.labels[index])])),
  'image_raw': tf.train.Feature(bytes_list=
                 tf.train.BytesList(value =
                          [image]))}))
  writer.write(example.SerializeToString())
 writer.close()

```

让我们分解这段代码，以理解不同的组件。

我们首先实例化一个 `TFRecordWriter` 对象，给它一个对应数据拆分的路径：

```py
filename = os.path.join(save_dir, data_splits[d] + '.tfrecords')
writer = tf.python_io.TFRecordWriter(filename)
```

然后我们遍历每个图像，将其从 NumPy 数组转换为字节字符串：

```py
image = data_set.images[index].tostring()
```

接下来，我们将图像转换为它们的 protobuf 格式。`tf.train.Example` 是用于存储我们的数据的结构。`Example` 对象包含一个 `Features` 对象，它又包含一个从属性名称到 `Feature` 的映射。`Feature` 可以包含一个 `Int64List`、一个 `BytesList` 或一个 `FloatList`（这里没有使用）。例如，在这里我们对图像的标签进行编码：

```py
tf.train.Feature(int64_list=tf.train.Int64List(value =
                 [int(data_set.labels[index])]))
```

这里是实际原始图像的编码：

```py
tf.train.Feature(bytes_list=tf.train.BytesList(value =[image]))
```

让我们看看我们保存的数据是什么样子。我们使用 `tf.python_io.tf_record_iterator` 来实现这一点，这是一个从 TFRecords 文件中读取记录的迭代器：

```py
filename = os.path.join(save_dir, 'train.tfrecords')
record_iterator = tf.python_io.tf_record_iterator(filename)
seralized_img_example= next(record_iterator)
```

`serialized_img` 是一个字节字符串。为了恢复保存图像到 TFRecord 时使用的结构，我们解析这个字节字符串，使我们能够访问我们之前存储的所有属性：

```py
example = tf.train.Example()
example.ParseFromString(seralized_img_example)
image = example.features.feature['image_raw'].bytes_list.value
label = example.features.feature['label'].int64_list.value[0]
width = example.features.feature['width'].int64_list.value[0]
height = example.features.feature['height'].int64_list.value[0]
```

我们的图像也保存为字节字符串，因此我们将其转换回 NumPy 数组，并将其重新整形为形状为 (28,28,1) 的张量：

```py
img_flat = np.fromstring(image[0], dtype=np.uint8)
img_reshaped = img_flat.reshape((height, width, -1))
```

这个基本示例应该让您了解 TFRecords 以及如何写入和读取它们。在实践中，我们通常希望将 TFRecords 读入一个预取数据队列作为多线程过程的一部分。在下一节中，我们首先介绍 TensorFlow 队列，然后展示如何将它们与 TFRecords 一起使用。

# 队列

TensorFlow 队列类似于普通队列，允许我们入队新项目，出队现有项目等。与普通队列的重要区别在于，就像 TensorFlow 中的任何其他内容一样，队列是计算图的一部分。它的操作像往常一样是符号化的，图中的其他节点可以改变其状态（就像变量一样）。这一点一开始可能会有点困惑，所以让我们通过一些示例来了解基本队列功能。

## 入队和出队

在这里，我们创建一个字符串的 *先进先出*（FIFO）队列，最多可以存储 10 个元素。由于队列是计算图的一部分，它们在会话中运行。在这个例子中，我们使用了一个 `tf.InteractiveSession()`：

```py
import tensorflow as tf

sess= tf.InteractiveSession()
queue1 = tf.FIFOQueue(capacity=10,dtypes=[tf.string])

```

在幕后，TensorFlow 为存储这 10 个项目创建了一个内存缓冲区。

就像 TensorFlow 中的任何其他操作一样，要向队列添加项目，我们创建一个操作：

```py
enque_op = queue1.enqueue(["F"])

```

由于您现在已经熟悉了TensorFlow中计算图的概念，因此定义`enque_op`并不会向队列中添加任何内容——我们需要运行该操作。因此，如果我们在运行操作之前查看`queue1`的大小，我们会得到这个结果：

```py
sess.run(queue1.size())

Out:
0

```

运行操作后，我们的队列现在有一个项目在其中：

```py
enque_op.run()
sess.run(queue1.size())
Out:
1
```

让我们向`queue1`添加更多项目，并再次查看其大小：

```py
enque_op = queue1.enqueue(["I"])
enque_op.run()
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["O"])
enque_op.run()

sess.run(queue1.size())

Out: 
4

```

接下来，我们出队项目。出队也是一个操作，其输出评估为对应于出队项目的张量：

```py
x = queue1.dequeue()
x.eval()

Out: b'F'
x.eval()

Out: b'I'
x.eval()

Out: b'F'
x.eval()

Out: b'O'

```

请注意，如果我们再次对空队列运行`x.eval()`，我们的主线程将永远挂起。正如我们将在本章后面看到的，实际上我们使用的代码知道何时停止出队并避免挂起。

另一种出队的方法是一次检索多个项目，使用`dequeue_many()`操作。此操作要求我们提前指定项目的形状：

```py
queue1 = tf.FIFOQueue(capacity=10,dtypes=[tf.string],shapes=[()])

```

在这里，我们像以前一样填充队列，然后一次出队四个项目：

```py
inputs = queue1.dequeue_many(4)
inputs.eval()

```

```py
Out: 
array([b'F', b'I', b'F', b'O'], dtype=object)
```

## 多线程

TensorFlow会话是多线程的——多个线程可以使用同一个会话并行运行操作。单个操作具有并行实现，默认情况下使用多个CPU核心或GPU线程。然而，如果单个对`sess.run()`的调用没有充分利用可用资源，可以通过进行多个并行调用来提高吞吐量。例如，在典型情况下，我们可能有多个线程对图像进行预处理并将其推送到队列中，而另一个线程则从队列中拉取预处理后的图像进行训练（在下一章中，我们将讨论分布式训练，这在概念上是相关的，但有重要的区别）。

让我们通过一些简单的示例来介绍在TensorFlow中引入线程以及与队列的自然互动，然后在MNIST图像的完整示例中将所有内容连接起来。

我们首先创建一个容量为100个项目的FIFO队列，其中每个项目是使用`tf.random_normal()`生成的随机浮点数：

```py
from __future__ import print_function
import threading
import time

gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100,dtypes=[tf.float32],shapes=())
enque = queue.enqueue(gen_random_normal)

def add():
  for i in range(10):
    sess.run(enque)
```

再次注意，`enque`操作实际上并没有将随机数添加到队列中（它们尚未生成）在图执行之前。项目将使用我们创建的`add()`函数进行入队，该函数通过多次调用`sess.run()`向队列中添加10个项目。

接下来，我们创建10个线程，每个线程并行运行`add()`，因此每个线程异步地向队列中添加10个项目。我们可以（暂时）将这些随机数视为添加到队列中的训练数据：

```py
threads = [threading.Thread(target=add, args=()) for i in range(10)]

threads
Out:
[<Thread(Thread-77, initial)>,
<Thread(Thread-78, initial)>,
<Thread(Thread-79, initial)>,
<Thread(Thread-80, initial)>,
<Thread(Thread-81, initial)>,
<Thread(Thread-82, initial)>,
<Thread(Thread-83, initial)>,
<Thread(Thread-84, initial)>,
<Thread(Thread-85, initial)>,
<Thread(Thread-86, initial)>]

```

我们已经创建了一个线程列表，现在我们执行它们，以短间隔打印队列的大小，从0增长到100：

```py
for t in threads:
  t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))

```

```py
Out:
10
84
100
```

最后，我们使用`dequeue_many()`一次出队10个项目，并检查结果：

```py
x = queue.dequeue_many(10)
print(x.eval())
sess.run(queue.size())

```

```py
Out:
[ 0.05863889 0.61680967 1.05087686 -0.29185265 -0.44238046 0.53796548
-0.24784896 0.40672767 -0.88107938 0.24592835]
90

```

## 协调器和QueueRunner

在现实场景中（正如我们将在本章后面看到的），有效地运行多个线程可能会更加复杂。线程应该能够正确停止（例如，避免“僵尸”线程，或者在一个线程失败时一起关闭所有线程），在停止后需要关闭队列，并且还有其他需要解决的技术但重要的问题。

TensorFlow配备了一些工具来帮助我们进行这个过程。其中最重要的是`tf.train.Coordinator`，用于协调一组线程的终止，以及`tf.train.QueueRunner`，它简化了让多个线程与无缝协作地将数据入队的过程。

### tf.train.Coordinator

我们首先演示如何在一个简单的玩具示例中使用`tf.train.Coordinator`。在下一节中，我们将看到如何将其作为真实输入管道的一部分使用。

我们使用与上一节类似的代码，修改`add()`函数并添加一个协调器：

```py
gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100,dtypes=[tf.float32],shapes=())
enque = queue.enqueue(gen_random_normal)

def add(coord,i):
  while not coord.should_stop():
    sess.run(enque)
    if i == 11:
      coord.request_stop()

coord = tf.train.Coordinator()
threads = [threading.Thread(target=add, args=(coord,i)) for i in range(10)]
coord.join(threads)

for t in threads:
  t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))

10
100
100

```

任何线程都可以调用`coord.request_stop()`来让所有其他线程停止。线程通常运行循环来检查是否停止，使用`coord.should_stop()`。在这里，我们将线程索引`i`传递给`add()`，并使用一个永远不满足的条件（`i==11`）来请求停止。因此，我们的线程完成了它们的工作，将全部100个项目添加到队列中。但是，如果我们将`add()`修改如下：

```py
def add(coord,i):
  while not coord.should_stop():
    sess.run(enque)
    if i == 1:
      coord.request_stop()
```

然后线程`i=1`将使用协调器请求所有线程停止，提前停止所有入队操作：

```py
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))

```

```py
Out:
10
17
17
```

### tf.train.QueueRunner和tf.RandomShuffleQueue

虽然我们可以创建多个重复运行入队操作的线程，但最好使用内置的`tf.train.QueueRunner`，它正是这样做的，同时在异常发生时关闭队列。

在这里，我们创建一个队列运行器，将并行运行四个线程以入队项目：

```py
gen_random_normal = tf.random_normal(shape=())
queue = tf.RandomShuffleQueue(capacity=100,dtypes=[tf.float32],
               min_after_dequeue=1)
enqueue_op = queue.enqueue(gen_random_normal)

qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
coord.request_stop()
coord.join(enqueue_threads)

```

请注意，`qr.create_threads()`将我们的会话作为参数，以及我们的协调器。

在这个例子中，我们使用了`tf.RandomShuffleQueue`而不是FIFO队列。`RandomShuffleQueue`只是一个带有以随机顺序弹出项目的出队操作的队列。这在训练使用随机梯度下降优化的深度神经网络时非常有用，这需要对数据进行洗牌。`min_after_dequeue`参数指定在调用出队操作后队列中将保留的最小项目数，更大的数字意味着更好的混合（随机抽样），但需要更多的内存。

# 一个完整的多线程输入管道

现在，我们将所有部分组合在一起，使用MNIST图像的工作示例，从将数据写入TensorFlow的高效文件格式，通过数据加载和预处理，到训练模型。我们通过在之前演示的排队和多线程功能的基础上构建，并在此过程中介绍一些更有用的组件来读取和处理TensorFlow中的数据。

首先，我们将MNIST数据写入TFRecords，使用与本章开头使用的相同代码：

```py
from__future__importprint_functionimportosimporttensorflowastffromtensorflow.contrib.learn.python.learn.datasetsimportmnistimportnumpyasnpsave_dir="*`path/to/mnist`*"# Download data to save_dirdata_sets=mnist.read_data_sets(save_dir,dtype=tf.uint8,reshape=False,validation_size=1000)data_splits=["train","test","validation"]fordinrange(len(data_splits)):print("saving "+data_splits[d])data_set=data_sets[d]filename=os.path.join(save_dir,data_splits[d]+'.tfrecords')writer=tf.python_io.TFRecordWriter(filename)forindexinrange(data_set.images.shape[0]):image=data_set.images[index].tostring()example=tf.train.Example(features=tf.train.Features(feature={'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[1]])),'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[2]])),'depth':tf.train.Feature(int64_list=tf.train.Int64List(value=[data_set.images.shape[3]])),'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[int(data_set.labels[index])])),'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))}))writer.write(example.SerializeToString())writer.close()
```

## tf.train.string_input_producer()和tf.TFRecordReader()

`tf.train.string_input_producer()`只是在幕后创建一个`QueueRunner`，将文件名字符串输出到我们的输入管道的队列中。这个文件名队列将在多个线程之间共享：

```py
filename = os.path.join(save_dir ,"train.tfrecords")
filename_queue = tf.train.string_input_producer(
  [filename], num_epochs=10)

```

`num_epochs`参数告诉`string_input_producer()`将每个文件名字符串生成`num_epochs`次。

接下来，我们使用`TFRecordReader()`从这个队列中读取文件，该函数接受一个文件名队列并从`filename_queue`中逐个出队文件名。在内部，`TFRecordReader()`使用图的状态来跟踪正在读取的TFRecord的位置，因为它从磁盘加载输入数据的“块之后的块”：

```py
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
  serialized_example,
  features={
    'image_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
  })

```

## tf.train.shuffle_batch()

我们解码原始字节字符串数据，进行（非常）基本的预处理将像素值转换为浮点数，然后使用`tf.train.shuffle_batch()`将图像实例洗牌并收集到`batch_size`批次中，该函数内部使用`RandomShuffleQueue`并累积示例，直到包含`batch_size` + `min_after_dequeue`个元素：

```py
image = tf.decode_raw(features['image_raw'], tf.uint8)
image.set_shape([784]) 
image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
label = tf.cast(features['label'], tf.int32)
# Randomly collect instances into batches 
images_batch, labels_batch = tf.train.shuffle_batch(
  [image, label], batch_size=128,
  capacity=2000,
  min_after_dequeue=1000)

```

`capacity`和`min_after_dequeue`参数的使用方式与之前讨论的相同。由`shuffle_batch()`返回的小批次是在内部创建的`RandomShuffleQueue`上调用`dequeue_many()`的结果。

## tf.train.start_queue_runners()和总结

我们将简单的softmax分类模型定义如下：

```py
W = tf.get_variable("W", [28*28, 10])
y_pred = tf.matmul(images_batch, W)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, 
                                                      labels=labels_batch)

loss_mean = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
init = tf.local_variables_initializer()
sess.run(init)

```

最后，我们通过调用`tf.train.start_queue_runners()`创建线程将数据入队到队列中。与其他调用不同，这个调用不是符号化的，实际上创建了线程（因此需要在初始化之后完成）：

```py
from __future__ import print_function

# Coordinator 
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

```

让我们看一下创建的线程列表：

```py
threads

Out: 
[<Thread(Thread-483, stopped daemon 13696)>,
 <Thread(Thread-484, started daemon 16376)>,
 <Thread(Thread-485, started daemon 4320)>,
 <Thread(Thread-486, started daemon 13052)>,
 <Thread(Thread-487, started daemon 7216)>,
 <Thread(Thread-488, started daemon 4332)>,
 <Thread(Thread-489, started daemon 16820)>]
```

一切就绪后，我们现在准备运行多线程过程，从读取和预处理批次到将其放入队列再到训练模型。重要的是要注意，我们不再使用熟悉的`feed_dict`参数——这样可以避免数据复制并提供加速，正如本章前面讨论的那样：

```py
try:
 step = 0
 while not coord.should_stop(): 
   step += 1
   sess.run([train_op])
   if step%500==0:
     loss_mean_val = sess.run([loss_mean])
     print(step)
     print(loss_mean_val)
except tf.errors.OutOfRangeError: 
  print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
finally:
  # When done, ask the threads to stop
  coord.request_stop()

# Wait for threads to finish
coord.join(threads)
sess.close()

```

直到抛出`tf.errors.OutOfRangeError`错误，表示队列为空，我们已经完成训练：

```py
Out:
Done training for 10 epochs, 2299500 steps.

```

# 未来的输入管道

在2017年中期，TensorFlow开发团队宣布了Dataset API，这是一个新的初步输入管道抽象，提供了一些简化和加速。本章介绍的概念，如TFRecords和队列，是TensorFlow及其输入管道过程的基础，仍然处于核心地位。TensorFlow仍然在不断发展中，自然会不时发生令人兴奋和重要的变化。请参阅[问题跟踪器](https://github.com/tensorflow/tensorflow/issues/7951)进行持续讨论。

# 总结

在本章中，我们看到了如何在TensorFlow中使用队列和线程，以及如何创建一个多线程输入管道。这个过程可以帮助增加吞吐量和资源利用率。在下一章中，我们将进一步展示如何在分布式环境中使用TensorFlow，在多个设备和机器之间进行工作。
