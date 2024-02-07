# 第9章。分布式TensorFlow

在本章中，我们讨论了使用TensorFlow进行分布式计算。我们首先简要调查了在机器学习中分布模型训练的不同方法，特别是深度学习。然后介绍了为支持分布式计算而设计的TensorFlow元素，最后通过一个端到端的示例将所有内容整合在一起。

# 分布式计算

*分布式* *计算*，在最一般的术语中，意味着利用多个组件来执行所需的计算或实现目标。在我们的情况下，这意味着使用多台机器来加快深度学习模型的训练。

这背后的基本思想是通过使用更多的计算能力，我们应该能够更快地训练相同的模型。尽管通常情况下确实如此，但实际上更快多少取决于许多因素（即，如果您期望使用10倍资源并获得10倍加速，您很可能会感到失望！）。

在机器学习环境中有许多分布计算的方式。您可能希望利用多个设备，无论是在同一台机器上还是跨集群。在训练单个模型时，您可能希望在集群上计算梯度以加快训练速度，无论是同步还是异步。集群也可以用于同时训练多个模型，或者为单个模型搜索最佳参数。

在接下来的小节中，我们将详细介绍并行化的许多方面。

## 并行化发生在哪里？

在并行化类型的分类中，第一个分割是位置。我们是在单台机器上使用多个计算设备还是跨集群？

在一台机器上拥有强大的硬件与多个设备变得越来越普遍。云服务提供商（如亚马逊网络服务）现在提供这种类型的平台设置并准备就绪。

无论是在云端还是本地，集群配置在设计和演进方面提供了更多的灵活性，设置可以扩展到目前在同一板上使用多个设备所不可行的程度（基本上，您可以使用任意大小的集群）。

另一方面，虽然同一板上的几个设备可以使用共享内存，但集群方法引入了节点之间通信的时间成本。当需要共享的信息量很大且通信相对缓慢时，这可能成为一个限制因素。

## 并行化的目标是什么？

第二个分割是实际目标。我们是想使用更多硬件使相同的过程更快，还是为了并行化多个模型的训练？

在开发阶段经常需要训练多个模型，需要在模型或超参数之间做出选择。在这种情况下，通常会运行几个选项并选择表现最佳的一个。这样做是很自然的。

另一方面，当训练单个（通常是大型）模型时，可以使用集群来加快训练速度。在最常见的方法中，称为*数据并行*，每个计算设备上都存在相同的模型结构，每个副本上运行的数据是并行化的。

例如，当使用梯度下降训练深度学习模型时，该过程由以下步骤组成：

1.  计算一批训练样本的梯度。

1.  对梯度求和。

1.  相应地对模型参数应用更新。

很明显，这个模式的第1步适合并行化。简单地使用多个设备计算梯度（针对不同的训练样本），然后在第2步中聚合结果并求和，就像常规情况下一样。

# **同步与异步数据并行**

在刚才描述的过程中，来自不同训练示例的梯度被聚合在一起，以对模型参数进行单次更新。这就是所谓的*同步*训练，因为求和步骤定义了一个流必须等待所有节点完成梯度计算的点。

有一种情况可能更好地避免这种情况，即当异构计算资源一起使用时，因为同步选项意味着等待节点中最慢的节点。

*异步*选项是在每个节点完成为其分配的训练示例的梯度计算后独立应用更新步骤。

# TensorFlow元素

在本节中，我们将介绍在并行计算中使用的TensorFlow元素和概念。这不是完整的概述，主要作为本章结束的并行示例的介绍。

## tf.app.flags

我们从一个与并行计算完全无关但对本章末尾的示例至关重要的机制开始。实际上，在TensorFlow示例中广泛使用`flags`机制，值得讨论。

实质上，`tf.app.flags`是Python `argparse`模块的包装器，通常用于处理命令行参数，具有一些额外和特定的功能。

例如，考虑一个具有典型命令行参数的Python命令行程序：

```py
'python distribute.py --job_name="ps" --task_index=0'

```

程序*distribute.py*传递以下内容：

```py
	job_name="ps"
	task_index=0

```

然后在Python脚本中提取这些信息，使用：

```py
tf.app.flags.DEFINE_string("job_name", "", "name of job")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")

```

参数（字符串和整数）由命令行中的名称、默认值和参数描述定义。

`flags`机制允许以下类型的参数：

+   `tf.app.flags.DEFINE_string`定义一个字符串值。

+   `tf.app.flags.DEFINE_boolean`定义一个布尔值。

+   `tf.app.flags.DEFINE_float`定义一个浮点值。

+   `tf.app.flags.DEFINE_integer`定义一个整数值。

最后，`tf.app.flags.FLAGS`是一个结构，包含从命令行输入解析的所有参数的值。参数可以通过`FLAGS.arg`访问，或者在必要时通过字典`FLAGS.__flags`访问（然而，强烈建议使用第一种选项——它设计的方式）。

## 集群和服务器

一个TensorFlow集群只是参与计算图并行处理的节点（也称为任务）的集合。每个任务由其可以访问的网络地址定义。例如：

```py
parameter_servers = ["localhost:2222"]
workers = ["localhost:2223",
"localhost:2224",
"localhost:2225"]
cluster = tf.train.ClusterSpec({"parameter_server": parameter_servers,
"worker": workers})
```

在这里，我们定义了四个本地任务（请注意，`localhost:*XXXX*`指向当前机器上端口*XXXX*，在多台计算机设置中，`localhost`将被IP地址替换）。任务分为一个*参数服务器*和三个*工作节点*。参数服务器/工作节点分配被称为*作业*。我们稍后在本章中进一步描述这些在训练期间的作用。

每个任务必须运行一个TensorFlow服务器，以便既使用本地资源进行实际计算，又与集群中的其他任务通信，以促进并行化。

基于集群定义，第一个工作节点上的服务器（即`localhost:2223`）将通过以下方式启动：

```py
server = tf.train.Server(cluster,
job_name="worker",
task_index=0)

```

由`Server()`接收的参数让它知道自己的身份，以及集群中其他成员的身份和地址。

一旦我们有了集群和服务器，我们就构建计算图，这将使我们能够继续进行并行计算。

## 在设备之间复制计算图

如前所述，有多种方法可以进行并行训练。在[“设备放置”](#device_Placement)中，我们简要讨论如何直接将操作放置在集群中特定任务上。在本节的其余部分，我们将介绍对于图间复制所必需的内容。

*图间* *复制* 指的是常见的并行化模式，其中在每个worker任务上构建一个单独但相同的计算图。在训练期间，每个worker计算梯度，并由参数服务器组合，参数服务器还跟踪参数的当前版本，以及可能是训练的其他全局元素（如全局步骤计数器等）。

我们使用`tf.train.replica_device_setter()`来在每个任务上复制模型（计算图）。`worker_device`参数应该指向集群中当前任务。例如，在第一个worker上我们运行这个：

```py
with tf.device(tf.train.replica_device_setter(
worker_device="/job:worker/task:%d" % 0,
cluster=cluster)):

# Build model...
```

例外是参数服务器，我们不在其上构建计算图。为了使进程不终止，我们使用：

```py
server.join()

```

这将在并行计算的过程中保持参数服务器的运行。

## 管理的会话

在这一部分，我们将介绍我们将在模型的并行训练中使用的机制。首先，我们定义一个`Supervisor`：

```py
sv = tf.train.Supervisor(is_chief=...,
logdir=...,
global_step=...,
init_op=...)

```

正如其名称所示，`Supervisor`用于监督训练，在并行设置中提供一些必要的实用程序。

传递了四个参数：

`is_chief`（布尔值）

必须有一个单一的*chief*，负责初始化等任务。

`logdir`（字符串）

存储日志的位置。

`global_step`

一个TensorFlow变量，将在训练期间保存当前的全局步骤。

`init_op`

一个用于初始化模型的TensorFlow操作，比如`tf.global_variables_initializer()`。

然后启动实际会话：

```py
with sv.managed_session(server.target) as sess:

# Train ... 
```

在这一点上，chief将初始化变量，而所有其他任务等待这个过程完成。

## 设备放置

在本节中我们讨论的最终TensorFlow机制是*设备放置*。虽然这个主题的全部内容超出了本章的范围，但概述中没有提到这种能力是不完整的，这在工程高级系统时非常有用。

在具有多个计算设备（CPU、GPU或这些组合）的环境中操作时，控制计算图中每个操作将发生的位置可能是有用的。这可能是为了更好地利用并行性，利用不同设备的不同能力，并克服某些设备的内存限制等限制。

即使您没有明确选择设备放置，TensorFlow也会在需要时输出所使用的放置。这是在构建会话时启用的：

```py
tf.Session(config=tf.ConfigProto(log_device_placement=True))

```

为了明确选择一个设备，我们使用：

```py
with tf.device('/gpu:0'):
  op = ...  

```

`'/gpu:0'`指向系统上的第一个GPU；同样，我们可以使用`'/cpu:0'`将操作放置在CPU上，或者在具有多个GPU设备的系统上使用`'/gpu:X'`，其中`X`是我们想要使用的GPU的索引。

最后，跨集群的放置是通过指向特定任务来完成的。例如：

```py
with tf.device("/job:worker/task:2"): 
  op = ...

```

这将分配给集群规范中定义的第二个`worker`任务。

# 跨CPU的放置

默认情况下，TensorFlow使用系统上所有可用的CPU，并在内部处理线程。因此，设备放置`'/cpu:0'`是完整的CPU功率，`'/cpu:1'`默认情况下不存在，即使在多CPU环境中也是如此。

为了手动分配到特定的CPU（除非您有非常充分的理由这样做，否则让TensorFlow处理），必须使用指令定义一个会话来分离CPU：

```py
config = tf.ConfigProto(device_count={"CPU": 8},
inter_op_parallelism_threads=8,
intra_op_parallelism_threads=1)
sess = tf.Session(config=config)

```

在这里，我们定义了两个参数：

+   `inter_op_parallelism_threads=8`，意味着我们允许八个线程用于不同的操作

+   `intra_op_parallelism_threads=1`，表示每个操作都有一个线程

这些设置对于一个8-CPU系统是有意义的。

# 分布式示例

在本节中，我们将所有内容整合在一起，以端到端的方式展示了我们在[第4章](ch04.html#convolutional_neural_networks)中看到的MNIST CNN模型的分布式训练示例。我们将使用一个参数服务器和三个工作任务。为了使其易于重现，我们将假设所有任务都在单台机器上本地运行（通过将`localhost`替换为IP地址，如前所述，可以轻松适应多机设置）。像往常一样，我们首先呈现完整的代码，然后将其分解为元素并加以解释：

```py
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 50
TRAINING_STEPS = 5000
PRINT_EVERY = 100
LOG_DIR = "/tmp/log"

parameter_servers = ["localhost:2222"]
workers = ["localhost:2223",
"localhost:2224",
"localhost:2225"]

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")
FLAGS = tf.app.flags.FLAGS

server = tf.train.Server(cluster,
job_name=FLAGS.job_name,
task_index=FLAGS.task_index)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def net(x):
x_image = tf.reshape(x, [-1, 28, 28, 1])
net = slim.layers.conv2d(x_image, 32, [5, 5], scope='conv1')
net = slim.layers.max_pool2d(net, [2, 2], scope='pool1')
net = slim.layers.conv2d(net, 64, [5, 5], scope='conv2')
net = slim.layers.max_pool2d(net, [2, 2], scope='pool2')
net = slim.layers.flatten(net, scope='flatten')
net = slim.layers.fully_connected(net, 500, scope='fully_connected')
net = slim.layers.fully_connected(net, 10, activation_fn=None,
                                  scope='pred')
return net

if FLAGS.job_name == "ps":
server.join()

elif FLAGS.job_name == "worker":

with tf.device(tf.train.replica_device_setter(
worker_device="/job:worker/task:%d" % FLAGS.task_index,
cluster=cluster)):

global_step = tf.get_variable('global_step', [],
initializer=tf.constant_initializer(0),
trainable=False)

x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
y = net(x)

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.AdamOptimizer(1e-4)\
        .minimize(cross_entropy, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
logdir=LOG_DIR,
global_step=global_step,
init_op=init_op)

with sv.managed_session(server.target) as sess:
step = 0

while not sv.should_stop() and step <= TRAINING_STEPS:

batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)

_, acc, step = sess.run([train_step, accuracy, global_step],
feed_dict={x: batch_x, y_: batch_y})

if step % PRINT_EVERY == 0:
print "Worker : {}, Step: {}, Accuracy (batch): {}".\
format(FLAGS.task_index, step, acc)

test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, 
                                         y_: mnist.test.labels})
print "Test-Accuracy: {}".format(test_acc)

sv.stop()

```

为了运行这个分布式示例，我们从四个不同的终端执行四个命令来分派每个任务（我们将很快解释这是如何发生的）：

```py
python distribute.py --job_name="ps" --task_index=0
python distribute.py --job_name="worker" --task_index=0
python distribute.py --job_name="worker" --task_index=1
python distribute.py --job_name="worker" --task_index=2

```

或者，以下将自动分派四个任务（取决于您使用的系统，输出可能全部发送到单个终端或四个单独的终端）：

```py
import subprocess
subprocess.Popen('python distribute.py --job_name="ps" --task_index=0', 
                 shell=True)
subprocess.Popen('python distribute.py --job_name="worker" --task_index=0', 
                 shell=True)
subprocess.Popen('python distribute.py --job_name="worker" --task_index=1', 
                 shell=True)
subprocess.Popen('python distribute.py --job_name="worker" --task_index=2', 
                 shell=True)

```

接下来，我们将检查前面示例中的代码，并突出显示这与我们迄今在书中看到的示例有何不同。

第一个块处理导入和常量：

```py
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 50
TRAINING_STEPS = 5000
PRINT_EVERY = 100
LOG_DIR = "/tmp/log"
```

在这里我们定义：

`BATCH_SIZE`

在每个小批次训练中要使用的示例数。

`TRAINING_STEPS`

我们将在训练中使用的小批次总数。

`PRINT_EVERY`

打印诊断信息的频率。由于在我们使用的分布式训练中，所有任务都有一个当前步骤的计数器，因此在某个步骤上的`print`只会从一个任务中发生。

`LOG_DIR`

训练监督员将把日志和临时信息保存到此位置。在程序运行之间应该清空，因为旧信息可能导致下一个会话崩溃。

接下来，我们定义集群，如本章前面讨论的：

```py
parameter_servers = ["localhost:2222"]
workers = ["localhost:2223",
           "localhost:2224",
           "localhost:2225"]

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

```

我们在本地运行所有任务。为了使用多台计算机，将`localhost`替换为正确的IP地址。端口2222-2225也是任意的，当然（但在使用单台机器时必须是不同的）：在分布式设置中，您可能会在所有机器上使用相同的端口。

在接下来的内容中，我们使用`tf.app.flags`机制来定义两个参数，我们将通过命令行在每个任务调用程序时提供：

```py
tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")
FLAGS = tf.app.flags.FLAGS

```

参数如下：

`job_name`

这将是`'ps'`表示单参数服务器，或者对于每个工作任务将是`'worker'`。

`task_index`

每种类型工作中任务的索引。因此，参数服务器将使用`task_index = 0`，而对于工作任务，我们将有`0`，`1`和`2`。

现在我们准备使用我们在本章中定义的集群中当前任务的身份来定义此当前任务的服务器。请注意，这将在我们运行的四个任务中的每一个上发生。这四个任务中的每一个都知道自己的身份（`job_name`，`task_index`），以及集群中其他每个人的身份（由第一个参数提供）：

```py
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)
```

在开始实际训练之前，我们定义我们的网络并加载要使用的数据。这类似于我们在以前的示例中所做的，所以我们不会在这里再次详细说明。为了简洁起见，我们使用TF-Slim：

```py
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def net(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    net = slim.layers.conv2d(x_image, 32, [5, 5], scope='conv1')
    net = slim.layers.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.layers.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.layers.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.layers.flatten(net, scope='flatten')
    net = slim.layers.fully_connected(net, 500, scope='fully_connected')
    net = slim.layers.fully_connected(net, 10, activation_fn=None, scope='pred')
    return net
```

在训练期间要执行的实际处理取决于任务的类型。对于参数服务器，我们希望机制主要是为参数提供服务。这包括等待请求并处理它们。要实现这一点，只需要这样做：

```py
if FLAGS.job_name == "ps":
server.join()

```

服务器的`.join()`方法即使在所有其他任务终止时也不会终止，因此一旦不再需要，必须在外部终止此进程。

在每个工作任务中，我们定义相同的计算图：

```py
with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
    y = net(x)

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.AdamOptimizer(1e-4)\
            .minimize(cross_entropy, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.global_variables_initializer()
```

我们使用`tf.train.replica_device_setter()`来指定这一点，这意味着TensorFlow变量将通过参数服务器进行同步（这是允许我们进行分布式计算的机制）。

`global_step`变量将保存跨任务训练期间的总步数（每个步骤索引只会出现在一个任务上）。这样可以创建一个时间线，以便我们始终知道我们在整个计划中的位置，从每个任务分开。

其余的代码是我们在整本书中已经看过的许多示例中看到的标准设置。

接下来，我们设置一个`Supervisor`和一个`managed_session`：

```py
sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
logdir=LOG_DIR,
global_step=global_step,
init_op=init_op)

with sv.managed_session(server.target) as sess:
```

这类似于我们在整个过程中使用的常规会话，只是它能够处理分布式的一些方面。变量的初始化将仅在一个任务中完成（通过`is_chief`参数指定的首席任务；在我们的情况下，这将是第一个工作任务）。所有其他任务将等待这个任务完成，然后继续。

会话开启后，我们开始训练：

```py
while not sv.should_stop() and step <= TRAINING_STEPS:

    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)

    _, acc, step = sess.run([train_step, accuracy, global_step],
                            feed_dict={x: batch_x, y_: batch_y})

    if step % PRINT_EVERY == 0:
        print "Worker : {}, Step: {}, Accuracy (batch): {}".\
            format(FLAGS.task_index, step, acc)

```

每隔`PRINT_EVERY`步，我们打印当前小批量的当前准确率。这将很快达到100%。例如，前两行可能是：

```py
Worker : 1, Step: 0.0, Accuracy (batch): 0.140000000596
Worker : 0, Step: 100.0, Accuracy (batch): 0.860000014305

```

最后，我们运行测试准确率：

```py
test_acc = sess.run(accuracy,
                    feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print "Test-Accuracy: {}".format(test_acc)
```

请注意，这将在每个工作任务上执行，因此相同的输出将出现三次。为了节省计算资源，我们可以只在一个任务中运行这个（例如，只在第一个工作任务中）。

# 总结

在本章中，我们涵盖了关于深度学习和机器学习中并行化的主要概念，并以一个关于数据并行化集群上分布式训练的端到端示例结束。

分布式训练是一个非常重要的工具，既可以加快训练速度，也可以训练那些否则不可行的模型。在下一章中，我们将介绍TensorFlow的Serving功能，允许训练好的模型在生产环境中被利用。
