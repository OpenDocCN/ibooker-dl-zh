# 第五章。文本 I：处理文本和序列，以及 TensorBoard 可视化

在本章中，我们将展示如何在 TensorFlow 中处理序列，特别是文本。我们首先介绍循环神经网络（RNN），这是一类强大的深度学习算法，特别适用于自然语言处理（NLP）。我们展示如何从头开始实现 RNN 模型，介绍一些重要的 TensorFlow 功能，并使用交互式 TensorBoard 可视化模型。然后，我们探讨如何在监督文本分类问题中使用 RNN 进行词嵌入训练。最后，我们展示如何构建一个更高级的 RNN 模型，使用长短期记忆（LSTM）网络，并如何处理可变长度的序列。

# 序列数据的重要性

我们在前一章中看到，利用图像的空间结构可以导致具有出色结果的先进模型。正如在那一章中讨论的那样，利用结构是成功的关键。正如我们将很快看到的，一种极其重要和有用的结构类型是顺序结构。从数据科学的角度来看，这种基本结构出现在许多数据集中，跨越所有领域。在计算机视觉中，视频是随时间演变的一系列视觉内容。在语音中，我们有音频信号，在基因组学中有基因序列；在医疗保健中有纵向医疗记录，在股票市场中有金融数据，等等（见图 5-1）。

![](img/letf_0501.png)

###### 图 5-1。序列数据的普遍性。

一种特别重要的具有强烈顺序结构的数据类型是自然语言——文本数据。利用文本中固有的顺序结构（字符、单词、句子、段落、文档）的深度学习方法处于自然语言理解（NLU）系统的前沿，通常将传统方法远远甩在后面。有许多类型的 NLU 任务需要解决，从文档分类到构建强大的语言模型，从自动回答问题到生成人类级别的对话代理。这些任务非常困难，吸引了整个学术界和工业界 AI 社区的努力和关注。

在本章中，我们专注于基本构建模块和任务，并展示如何在 TensorFlow 中处理序列，主要是文本。我们深入研究了 TensorFlow 中序列模型的核心元素，从头开始实现其中一些，以获得深入的理解。在下一章中，我们将展示更高级的文本建模技术，使用 TensorFlow，而在第七章中，我们将使用提供更简单、高级实现方式的抽象库来实现我们的模型。

我们从最重要和流行的用于序列（特别是文本）的深度学习模型类开始：循环神经网络。

# 循环神经网络简介

循环神经网络是一类强大且广泛使用的神经网络架构，用于建模序列数据。RNN 模型背后的基本思想是序列中的每个新元素都会提供一些新信息，从而更新模型的当前状态。

在前一章中，我们探讨了使用 CNN 模型进行计算机视觉的内容，讨论了这些架构是如何受到当前科学对人类大脑处理视觉信息方式的启发。这些科学观念通常与我们日常生活中对顺序信息处理方式的常识直觉非常接近。

当我们接收新信息时，显然我们的“历史”和“记忆”并没有被抹去，而是“更新”。当我们阅读文本中的句子时，随着每个新单词，我们当前的信息状态会被更新，这不仅取决于新观察到的单词，还取决于前面的单词。

在统计学和概率论中的一个基本数学构造，通常被用作通过机器学习建模顺序模式的基本构件是马尔可夫链模型。比喻地说，我们可以将我们的数据序列视为“链”，链中的每个节点在某种程度上依赖于前一个节点，因此“历史”不会被抹去，而是被延续。

RNN 模型也基于这种链式结构的概念，并且在如何确切地维护和更新信息方面有所不同。正如它们的名称所示，循环神经网络应用某种形式的“循环”。如图 5-2 所示，在某个时间点*t*，网络观察到一个输入*x[t]*（句子中的一个单词），并将其“状态向量”从上一个向量*h[t-1]*更新为*h[t]*。当我们处理新的输入（下一个单词）时，它将以某种依赖于*h[t]*的方式进行，因此依赖于序列的历史（我们之前看到的单词影响我们对当前单词的理解）。如图所示，这种循环结构可以简单地被视为一个长长的展开链，链中的每个节点执行相同类型的处理“步骤”，基于它从前一个节点的输出获得的“消息”。当然，这与先前讨论的马尔可夫链模型及其隐马尔可夫模型（HMM）扩展密切相关，这些内容在本书中没有讨论。

![](img/letf_0502.png)

###### 图 5-2。随时间更新的循环神经网络。

## 基础 RNN 实现

在本节中，我们从头开始实现一个基本的 RNN，探索其内部工作原理，并了解 TensorFlow 如何处理序列。我们介绍了一些强大的、相当低级的工具，TensorFlow 提供了这些工具用于处理序列数据，您可以使用这些工具来实现自己的系统。

在接下来的部分中，我们将展示如何使用更高级别的 TensorFlow RNN 模块。

我们从数学上定义我们的基本模型开始。这主要包括定义循环结构 - RNN 更新步骤。

我们简单的基础 vanilla RNN 的更新步骤是

*h[t]* = *tanh*(*W[x]**x[t]* + *W[h]h[t-1]* + *b*)

其中*W[h]*，*W[x]*和*b*是我们学习的权重和偏置变量，*tanh*(·)是双曲正切函数，其范围在[-1,1]之间，并且与前几章中使用的 sigmoid 函数密切相关，*x[t]*和*h[t]*是之前定义的输入和状态向量。最后，隐藏状态向量乘以另一组权重，产生出现在图 5-2 中的输出。

### MNIST 图像作为序列

为了初尝序列模型的强大和普适性，在本节中，我们实现我们的第一个 RNN 来解决您现在熟悉的 MNIST 图像分类任务。在本章的后面，我们将专注于文本序列，并看看神经序列模型如何强大地操纵它们并提取信息以解决 NLU 任务。

但是，你可能会问，图像与序列有什么关系？

正如我们在上一章中看到的，卷积神经网络的架构利用了图像的空间结构。虽然自然图像的结构非常适合 CNN 模型，但从不同角度查看图像的结构是有启发性的。在前沿深度学习研究的趋势中，先进模型尝试利用图像中各种顺序结构，试图以某种方式捕捉创造每个图像的“生成过程”。直观地说，这一切归结为图像中相邻区域在某种程度上相关，并试图对这种结构建模。

在这里，为了介绍基本的 RNN 以及如何处理序列，我们将图像简单地视为序列：我们将数据中的每个图像看作是一系列的行（或列）。在我们的 MNIST 数据中，这意味着每个 28×28 像素的图像可以被视为长度为 28 的序列，序列中的每个元素是一个包含 28 个像素的向量（参见图 5-3）。然后，RNN 中的时间依赖关系可以被想象成一个扫描头，从上到下（行）或从左到右（列）扫描图像。

![](img/letf_0503.png)

###### 图 5-3。图像作为像素列的序列。

我们首先加载数据，定义一些参数，并为我们的数据创建占位符：

```py
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Define some parameters
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# Where to save TensorBoard model summaries
LOG_DIR = "logs/RNN_with_summaries"

# Create placeholders for inputs, labels
_inputs = tf.placeholder(tf.float32,shape=[None, time_steps,
                        element_size],
                                              name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes],
                                              name='labels')

```

`element_size`是我们序列中每个向量的维度，在我们的情况下是 28 个像素的行/列。`time_steps`是序列中这样的元素的数量。

正如我们在之前的章节中看到的，当我们使用内置的 MNIST 数据加载器加载数据时，它以展开的形式呈现，即一个包含 784 个像素的向量。在训练期间加载数据批次时（我们稍后将在本节中介绍），我们只需将每个展开的向量重塑为[`batch_size`, `time_steps`, `element_size`]：

```py
batch_x, batch_y = mnist.train.next_batch(batch_size)
# Reshape data to get 28 sequences of 28 pixels
batch_x = batch_x.reshape((batch_size, time_steps, element_size))
```

我们将`hidden_layer_size`设置为`128`（任意值），控制之前讨论的隐藏 RNN 状态向量的大小。

`LOG_DIR`是我们保存模型摘要以供 TensorBoard 可视化的目录。随着我们的学习，您将了解这意味着什么。

# TensorBoard 可视化

在本章中，我们还将简要介绍 TensorBoard 可视化。TensorBoard 允许您监视和探索模型结构、权重和训练过程，并需要对代码进行一些非常简单的添加。更多细节将在本章和本书后续部分提供。

最后，我们创建了适当维度的输入和标签占位符。

### RNN 步骤

让我们实现 RNN 步骤的数学模型。

我们首先创建一个用于记录摘要的函数，稍后我们将在 TensorBoard 中使用它来可视化我们的模型和训练过程（在这个阶段理解其技术细节并不重要）：

```py
# This helper function, taken from the official TensorFlow documentation,
# simply adds some ops that take care of logging summaries
def variable_summaries(var):
  with tf.name_scope('summaries'):
   mean = tf.reduce_mean(var)
   tf.summary.scalar('mean', mean)
   with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
   tf.summary.scalar('stddev', stddev)
   tf.summary.scalar('max', tf.reduce_max(var))
   tf.summary.scalar('min', tf.reduce_min(var))
   tf.summary.histogram('histogram', var)
```

接下来，我们创建在 RNN 步骤中使用的权重和偏置变量：

```py
# Weights and bias for input and hidden layer
with tf.name_scope('rnn_weights'):
  with tf.name_scope("W_x"):
    Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
    variable_summaries(Wx)
  with tf.name_scope("W_h"):
    Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
    variable_summaries(Wh)
  with tf.name_scope("Bias"):
    b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
    variable_summaries(b_rnn)

```

### 使用 tf.scan()应用 RNN 步骤

现在，我们创建一个函数，实现了我们在前一节中看到的基本 RNN 步骤，使用我们创建的变量。现在应该很容易理解这里使用的 TensorFlow 代码：

```py
def rnn_step(previous_hidden_state,x):

    current_hidden_state = tf.tanh(
      tf.matmul(previous_hidden_state, Wh) +
      tf.matmul(x, Wx) + b_rnn)

    return current_hidden_state

```

接下来，我们将这个函数应用到所有的 28 个时间步上：

```py
# Processing inputs to work with scan function
# Current input shape: (batch_size, time_steps, element_size)
processed_input = tf.transpose(_inputs, perm=[1, 0, 2])
# Current input shape now: (time_steps, batch_size, element_size)

initial_hidden = tf.zeros([batch_size,hidden_layer_size])
# Getting all state vectors across time
all_hidden_states = tf.scan(rnn_step,
              processed_input,
              initializer=initial_hidden,
              name='states')

```

在这个小的代码块中，有一些重要的元素需要理解。首先，我们将输入从`[batch_size, time_steps, element_size]`重塑为`[time_steps, batch_size, element_size]`。`tf.transpose()`的`perm`参数告诉 TensorFlow 我们想要交换的轴。现在，我们的输入张量中的第一个轴代表时间轴，我们可以通过使用内置的`tf.scan()`函数在所有时间步上进行迭代，该函数重复地将一个可调用（函数）应用于序列中的元素，如下面的说明所述。

# tf.scan()

这个重要的函数被添加到 TensorFlow 中，允许我们在计算图中引入循环，而不仅仅是通过添加更多和更多的操作复制来“展开”循环。更技术上地说，它是一个类似于 reduce 操作符的高阶函数，但它返回随时间的所有中间累加器值。这种方法有几个优点，其中最重要的是能够具有动态数量的迭代而不是固定的，以及用于图构建的计算速度提升和优化。

为了演示这个函数的使用，考虑以下简单示例（这与本节中的整体 RNN 代码是分开的）：

```py
import numpy as np
import tensorflow as tf

elems = np.array(["T","e","n","s","o","r", " ", "F","l","o","w"])
scan_sum = tf.scan(lambda a, x: a + x, elems)

sess=tf.InteractiveSession()
sess.run(scan_sum)

```

让我们看看我们得到了什么：

```py
array([b'T', b'Te', b'Ten', b'Tens', b'Tenso', b'Tensor', b'Tensor ',
       b'Tensor F', b'Tensor Fl', b'Tensor Flo', b'Tensor Flow'],
       dtype=object)

```

在这种情况下，我们使用`tf.scan()`将字符顺序连接到一个字符串中，类似于算术累积和。

### 顺序输出

正如我们之前看到的，在 RNN 中，我们为每个时间步获得一个状态向量，将其乘以一些权重，然后获得一个输出向量——我们数据的新表示。让我们实现这个：

```py
# Weights for output layers
with tf.name_scope('linear_layer_weights') as scope:
  with tf.name_scope("W_linear"):
    Wl = tf.Variable(tf.truncated_normal([hidden_layer_size,
                       num_classes],
                       mean=0,stddev=.01))
    variable_summaries(Wl)
  with tf.name_scope("Bias_linear"):
    bl = tf.Variable(tf.truncated_normal([num_classes],
                      mean=0,stddev=.01))
    variable_summaries(bl)

# Apply linear layer to state vector    
def get_linear_layer(hidden_state):

  return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
  # Iterate across time, apply linear layer to all RNN outputs
  all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
  # Get last output
  output = all_outputs[-1]
  tf.summary.histogram('outputs', output)

```

我们的 RNN 的输入是顺序的，输出也是如此。在这个序列分类示例中，我们取最后一个状态向量，并通过一个全连接的线性层将其传递，以提取一个输出向量（稍后将通过 softmax 激活函数传递以生成预测）。这在基本序列分类中是常见的做法，我们假设最后一个状态向量已经“积累”了代表整个序列的信息。

为了实现这一点，我们首先定义线性层的权重和偏置项变量，并为该层创建一个工厂函数。然后我们使用`tf.map_fn()`将此层应用于所有输出，这与典型的 map 函数几乎相同，该函数以元素方式将函数应用于序列/可迭代对象，本例中是在我们序列的每个元素上。

最后，我们提取批次中每个实例的最后输出，使用负索引（类似于普通 Python）。稍后我们将看到一些更多的方法来做这个，并深入研究输出和状态。

### RNN 分类

现在我们准备训练一个分类器，方式与前几章相同。我们定义损失函数计算、优化和预测的操作，为 TensorBoard 添加一些更多摘要，并将所有这些摘要合并为一个操作：

```py
with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
  tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  # Using RMSPropOptimizer
  train_step = tf.train.RMSPropOptimizer(0.001, 0.9)\
                                  .minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(
                               tf.argmax(y,1), tf.argmax(output,1))

  accuracy = (tf.reduce_mean(
                      tf.cast(correct_prediction, tf.float32)))*100
  tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries
merged = tf.summary.merge_all()

```

到目前为止，您应该熟悉用于定义损失函数和优化的大多数组件。在这里，我们使用`RMSPropOptimizer`，实现了一个众所周知且强大的梯度下降算法，带有一些标准的超参数。当然，我们可以使用任何其他优化器（并在本书中一直这样做！）。

我们创建一个包含未见过的 MNIST 图像的小测试集，并添加一些用于记录摘要的技术操作和命令，这些将在 TensorBoard 中使用。

让我们运行模型并查看结果：

```py
# Get a small test set  
test_data = mnist.test.images[:batch_size].reshape((-1, time_steps,
                          element_size))
test_label = mnist.test.labels[:batch_size]

with tf.Session() as sess:
  # Write summaries to LOG_DIR -- used by TensorBoard
  train_writer = tf.summary.FileWriter(LOG_DIR + '/train',
                    graph=tf.get_default_graph())
  test_writer = tf.summary.FileWriter(LOG_DIR + '/test',
                    graph=tf.get_default_graph())

  sess.run(tf.global_variables_initializer())

  for i in range(10000):

      batch_x, batch_y = mnist.train.next_batch(batch_size)
      # Reshape data to get 28 sequences of 28 pixels
      batch_x = batch_x.reshape((batch_size, time_steps,
                   element_size))
      summary,_ = sess.run([merged,train_step],
                feed_dict={_inputs:batch_x, y:batch_y})
      # Add to summaries
      train_writer.add_summary(summary, i)

      if i % 1000 == 0:
        acc,loss, = sess.run([accuracy,cross_entropy],
                  feed_dict={_inputs: batch_x,
                        y: batch_y})
        print ("Iter " + str(i) + ", Minibatch Loss= " + \
           "{:.6f}".format(loss) + ", Training Accuracy= " + \
           "{:.5f}".format(acc)) 
      if i % 10:
        # Calculate accuracy for 128 MNIST test images and
        # add to summaries
        summary, acc = sess.run([merged, accuracy],
                    feed_dict={_inputs: test_data,
                         y: test_label})
        test_writer.add_summary(summary, i)

  test_acc = sess.run(accuracy, feed_dict={_inputs: test_data,
                      y: test_label})
  print ("Test Accuracy:", test_acc)

```

最后，我们打印一些训练和测试准确率的结果：

```py
Iter 0, Minibatch Loss= 2.303386, Training Accuracy= 7.03125
Iter 1000, Minibatch Loss= 1.238117, Training Accuracy= 52.34375
Iter 2000, Minibatch Loss= 0.614925, Training Accuracy= 85.15625
Iter 3000, Minibatch Loss= 0.439684, Training Accuracy= 82.81250
Iter 4000, Minibatch Loss= 0.077756, Training Accuracy= 98.43750
Iter 5000, Minibatch Loss= 0.220726, Training Accuracy= 89.84375
Iter 6000, Minibatch Loss= 0.015013, Training Accuracy= 100.00000
Iter 7000, Minibatch Loss= 0.017689, Training Accuracy= 100.00000
Iter 8000, Minibatch Loss= 0.065443, Training Accuracy= 99.21875
Iter 9000, Minibatch Loss= 0.071438, Training Accuracy= 98.43750
Testing Accuracy: 97.6563

```

总结这一部分，我们从原始的 MNIST 像素开始，并将它们视为顺序数据——每列（或行）的 28 个像素作为一个时间步。然后，我们应用了 vanilla RNN 来提取对应于每个时间步的输出，并使用最后的输出来执行整个序列（图像）的分类。

### 使用 TensorBoard 可视化模型

TensorBoard 是一个交互式基于浏览器的工具，允许我们可视化学习过程，以及探索我们训练的模型。

运行 TensorBoard，转到命令终端并告诉 TensorBoard 相关摘要的位置：

```py
tensorboard--logdir=*`LOG_DIR`*
```

在这里，`*LOG_DIR*`应替换为您的日志目录。如果您在 Windows 上并且这不起作用，请确保您从存储日志数据的相同驱动器运行终端，并按以下方式向日志目录添加名称，以绕过 TensorBoard 解析路径的错误：

```py
tensorboard--logdir=rnn_demo:*`LOG_DIR`*
```

TensorBoard 允许我们为单独的日志目录分配名称，方法是在名称和路径之间放置一个冒号，当使用多个日志目录时可能会有用。在这种情况下，我们将传递一个逗号分隔的日志目录列表，如下所示：

```py
tensorboard--logdir=rnn_demo1:*`LOG_DIR1`*,rnn_demo2:*`LOG_DIR2`*
```

在我们的示例中（有一个日志目录），一旦您运行了`tensorboard`命令，您应该会得到类似以下内容的信息，告诉您在浏览器中导航到哪里：

```py
Starting TensorBoard b'39' on port 6006
(You can navigate to http://10.100.102.4:6006)
```

如果地址无法使用，请转到*localhost:6006*，这个地址应该始终有效。

TensorBoard 递归地遍历以`*LOG_DIR*`为根的目录树，寻找包含 tfevents 日志数据的子目录。如果您多次运行此示例，请确保在每次运行后要么删除您创建的`*LOG_DIR*`文件夹，要么将日志写入`*LOG_DIR*`内的单独子目录，例如`*LOG_DIR*`*/run1/train*，`*LOG_DIR*`*/run2/train*等，以避免覆盖日志文件，这可能会导致一些“奇怪”的图形。

让我们看一些我们可以获得的可视化效果。在下一节中，我们将探讨如何使用 TensorBoard 对高维数据进行交互式可视化-现在，我们专注于绘制训练过程摘要和训练权重。

首先，在浏览器中，转到标量选项卡。在这里，TensorBoard 向我们显示所有标量的摘要，包括通常最有趣的训练和测试准确性，以及我们记录的有关变量的一些摘要统计信息（请参见图 5-4）。将鼠标悬停在图表上，我们可以看到一些数字。

！[](assets/letf_0504.png)

###### 图 5-4。TensorBoard 标量摘要。

在图形选项卡中，我们可以通过放大来获得计算图的交互式可视化，从高级视图到基本操作（请参见图 5-5）。

！[](assets/letf_0505.png)

###### 图 5-5。放大计算图。

最后，在直方图选项卡中，我们可以看到在训练过程中权重的直方图（请参见图 5-6）。当然，我们必须明确将这些直方图添加到我们的日志记录中才能查看它们，使用`tf.summary.histogram()`。

！[](assets/letf_0506.png)

###### 图 5-6。学习过程中权重的直方图。

## TensorFlow 内置的 RNN 函数

前面的示例教会了我们一些使用序列的基本和强大的方法，通过几乎从头开始实现我们的图。在实践中，当然最好使用内置的高级模块和函数。这不仅使代码更短，更容易编写，而且利用了 TensorFlow 实现提供的许多低级优化。

在本节中，我们首先以完整的新代码的新版本呈现。由于大部分整体细节没有改变，我们将重点放在主要的新元素`tf.contrib.rnn.BasicRNNCell`和`tf.nn.dynamic_rnn()`上：

```py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

element_size = 28;time_steps = 28;num_classes = 10
batch_size = 128;hidden_layer_size = 128

_inputs = tf.placeholder(tf.float32,shape=[None, time_steps,
                     element_size],
                     name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes],name='inputs')

# TensorFlow built-in functions
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                  mean=0,stddev=.01))
bl = tf.Variable(tf.truncated_normal([num_classes],mean=0,stddev=.01))

def get_linear_layer(vector):
  return tf.matmul(vector, Wl) + bl

last_rnn_output = outputs[:,-1,:]
final_output = get_linear_layer(last_rnn_output)

softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,
                         labels=y)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

test_data = mnist.test.images[:batch_size].reshape((-1,
                      time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

for i in range(3001):

   batch_x, batch_y = mnist.train.next_batch(batch_size)
   batch_x = batch_x.reshape((batch_size, time_steps, element_size))
   sess.run(train_step,feed_dict={_inputs:batch_x,
                   y:batch_y})
   if i % 1000 == 0:
      acc = sess.run(accuracy, feed_dict={_inputs: batch_x,
                        y: batch_y})
      loss = sess.run(cross_entropy,feed_dict={_inputs:batch_x,
                          y:batch_y})
      print ("Iter " + str(i) + ", Minibatch Loss= " + \
         "{:.6f}".format(loss) + ", Training Accuracy= " + \
         "{:.5f}".format(acc)) 

print ("Testing Accuracy:", 
  sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label}))

```

### tf.contrib.rnn.BasicRNNCell 和 tf.nn.dynamic_rnn()

TensorFlow 的 RNN 单元是表示每个循环“单元”执行的基本操作（请参见本章开头的图 5-2 进行说明），以及其关联状态的抽象。它们通常是`rnn_step()`函数及其所需的相关变量的“替代”。当然，有许多变体和类型的单元，每个单元都有许多方法和属性。我们将在本章末尾和本书后面看到一些更高级的单元。

一旦我们创建了`rnn_cell`，我们将其输入到`tf.nn.dynamic_rnn()`中。此函数替换了我们基本实现中的`tf.scan()`，并创建了由`rnn_cell`指定的 RNN。

截至本文撰写时，即 2017 年初，TensorFlow 包括用于创建 RNN 的静态和动态函数。这是什么意思？静态版本创建一个固定长度的展开图（如图 5-2）。动态版本使用`tf.While`循环在执行时动态构建图，从而加快图的创建速度，这可能是显著的。这种动态构建在其他方面也非常有用，其中一些我们将在本章末尾讨论变长序列时提及。

请注意，*contrib*指的是这个库中的代码是由贡献者贡献的，并且仍需要测试。我们将在第七章中更详细地讨论`contrib`库。`BasicRNNCell`在 TensorFlow 1.0 中被移动到`contrib`作为持续开发的一部分。在 1.2 版本中，许多 RNN 函数和类被移回核心命名空间，并在`contrib`中保留别名以实现向后兼容性，这意味着在撰写本文时，前面的代码适用于所有 1.X 版本。

# 文本序列的 RNN

我们在本章开始时学习了如何在 TensorFlow 中实现 RNN 模型。为了便于说明，我们展示了如何在由 MNIST 图像中的像素组成的序列上实现和使用 RNN。接下来我们将展示如何在文本序列上使用这些序列模型。

文本数据具有与图像数据明显不同的一些属性，我们将在这里和本书的后面进行讨论。这些属性可能使得一开始处理文本数据有些困难，而文本数据总是需要至少一些基本的预处理步骤才能让我们能够处理它。为了介绍在 TensorFlow 中处理文本，我们将专注于核心组件并创建一个最小的、人为的文本数据集，这将让我们直接开始行动。在第七章中，我们将应用 RNN 模型进行电影评论情感分类。

让我们开始吧，展示我们的示例数据并在进行的过程中讨论文本数据集的一些关键属性。

## 文本序列

在之前看到的 MNIST RNN 示例中，每个序列的大小是固定的——图像的宽度（或高度）。序列中的每个元素都是一个由 28 个像素组成的密集向量。在 NLP 任务和数据集中，我们有一种不同类型的“图片”。

我们的序列可以是由单词组成的句子，由句子组成的段落，甚至由字符组成的单词或段落组成的整个文档。

考虑以下句子：“我们公司为农场提供智能农业解决方案，具有先进的人工智能、深度学习。”假设我们从在线新闻博客中获取这个句子，并希望将其作为我们机器学习系统的一部分进行处理。

这个句子中的每个单词都将用一个 ID 表示——一个整数，通常在 NLP 中被称为令牌 ID。因此，例如单词“agriculture”可以映射到整数 3452，单词“farm”到 12，“AI”到 150，“deep-learning”到 0。这种以整数标识符表示的表示形式与图像数据中的像素向量在多个方面都非常不同。我们将在讨论词嵌入时很快详细阐述这一重要观点，并在第六章中进行讨论。

为了使事情更具体，让我们从创建我们简化的文本数据开始。

我们的模拟数据由两类非常短的“句子”组成，一类由奇数组成，另一类由偶数组成（数字用英文书写）。我们生成由表示偶数和奇数的单词构建的句子。我们的目标是在监督文本分类任务中学习将每个句子分类为奇数或偶数。

当然，对于这个简单的任务，我们实际上并不需要任何机器学习——我们只是为了说明目的而使用这个人为的例子。

首先，我们定义一些常量，随着我们的进行将会解释：

```py
import numpy as np
import tensorflow as tf

batch_size = 128;embedding_dimension = 64;num_classes = 2
hidden_layer_size = 32;times_steps = 6;element_size = 1
```

接下来，我们创建句子。我们随机抽取数字并将其映射到相应的“单词”（例如，1 映射到“One”，7 映射到“Seven”等）。

文本序列通常具有可变长度，这当然也适用于所有真实的自然语言数据（例如在本页上出现的句子）。

为了使我们模拟的句子具有不同的长度，我们为每个句子抽取一个介于 3 和 6 之间的随机长度，使用`np.random.choice(range(3, 7))`——下限包括，上限不包括。

现在，为了将所有输入句子放入一个张量中（每个数据实例的批次），我们需要它们以某种方式具有相同的大小—因此，我们用零（或*PAD*符号）填充长度短于 6 的句子，使所有句子大小相等（人为地）。这个预处理步骤称为*零填充*。以下代码完成了所有这些：

```py
digit_to_word_map = {1:"One",2:"Two", 3:"Three", 4:"Four", 5:"Five",
          6:"Six",7:"Seven",8:"Eight",9:"Nine"}
digit_to_word_map[0]="PAD"

even_sentences = []
odd_sentences = []
seqlens = []
for i in range(10000):
  rand_seq_len = np.random.choice(range(3,7))
  seqlens.append(rand_seq_len)
  rand_odd_ints = np.random.choice(range(1,10,2),
                  rand_seq_len)
  rand_even_ints = np.random.choice(range(2,10,2),
                   rand_seq_len)

    # Padding
  if rand_seq_len<6:
    rand_odd_ints = np.append(rand_odd_ints,
                 [0]*(6-rand_seq_len))
    rand_even_ints = np.append(rand_even_ints,
                 [0]*(6-rand_seq_len))

  even_sentences.append(" ".join([digit_to_word_map[r] for
               r in rand_odd_ints]))
  odd_sentences.append(" ".join([digit_to_word_map[r] for
               r in rand_even_ints])) 

data = even_sentences+odd_sentences
# Same seq lengths for even, odd sentences
seqlens*=2

```

让我们看一下我们的句子，每个都填充到长度 6：

```py
even_sentences[0:6]

Out:
['Four Four Two Four Two PAD',
'Eight Six Four PAD PAD PAD',
'Eight Two Six Two PAD PAD',
'Eight Four Four Eight PAD PAD',
'Eight Eight Four PAD PAD PAD',
'Two Two Eight Six Eight Four']

```

```py
odd_sentences[0:6]

Out:
['One Seven Nine Three One PAD',
'Three Nine One PAD PAD PAD',
'Seven Five Three Three PAD PAD',
'Five Five Three One PAD PAD',
'Three Three Five PAD PAD PAD',
'Nine Three Nine Five Five Three']

```

请注意，我们向我们的数据和`digit_to_word_map`字典中添加了*PAD*单词（标记），并分别存储偶数和奇数句子及其原始长度（填充之前）。

让我们看一下我们打印的句子的原始序列长度：

```py
seqlens[0:6]

Out: 
[5, 3, 4, 4, 3, 6]
```

为什么保留原始句子长度？通过零填充，我们解决了一个技术问题，但又创建了另一个问题：如果我们简单地将这些填充的句子通过我们的 RNN 模型，它将处理无用的`PAD`符号。这将通过处理“噪音”损害模型的正确性，并增加计算时间。我们通过首先将原始长度存储在`seqlens`数组中，然后告诉 TensorFlow 的`tf.nn.dynamic_rnn()`每个句子的结束位置来解决这个问题。

在本章中，我们的数据是模拟的——由我们生成。在实际应用中，我们将首先获得一系列文档（例如，一句话的推文），然后将每个单词映射到一个整数 ID。

因此，我们现在将单词映射到索引—单词*标识符*—通过简单地创建一个以单词为键、索引为值的字典。我们还创建了反向映射。请注意，单词 ID 和每个单词代表的数字之间没有对应关系—ID 没有语义含义，就像在任何具有真实数据的 NLP 应用中一样：

```py
# Map from words to indices
word2index_map ={}
index=0
for sent in data:
  for word in sent.lower().split():
    if word not in word2index_map:
      word2index_map[word] = index
      index+=1
# Inverse map    
index2word_map = {index: word for word, index in word2index_map.items()}      
vocabulary_size = len(index2word_map)

```

这是一个监督分类任务—我们需要一个以 one-hot 格式的标签数组，训练和测试集，一个生成实例批次的函数和占位符，和通常一样。

首先，我们创建标签并将数据分为训练集和测试集：

```py
labels = [1]*10000 + [0]*10000
for i in range(len(labels)):
  label = labels[i]
  one_hot_encoding = [0]*2
  one_hot_encoding[label] = 1
  labels[i] = one_hot_encoding

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]

labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]
train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]

test_x = data[10000:]
test_y = labels[10000:]
test_seqlens = seqlens[10000:]

```

接下来，我们创建一个生成句子批次的函数。每个批次中的句子只是一个对应于单词的整数 ID 列表：

```py
def get_sentence_batch(batch_size,data_x,
           data_y,data_seqlens):
  instance_indices = list(range(len(data_x)))
  np.random.shuffle(instance_indices)
  batch = instance_indices[:batch_size]
  x = [[word2index_map[word] for word in data_x[i].lower().split()]
    for i in batch]
  y = [data_y[i] for i in batch]
  seqlens = [data_seqlens[i] for i in batch]
  return x,y,seqlens

```

最后，我们为数据创建占位符：

```py
_inputs = tf.placeholder(tf.int32, shape=[batch_size,times_steps])
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])

# seqlens for dynamic calculation
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])
```

请注意，我们已经为原始序列长度创建了占位符。我们将很快看到如何在我们的 RNN 中使用这些。

## 监督词嵌入

我们的文本数据现在被编码为单词 ID 列表—每个句子是一个对应于单词的整数序列。这种*原子*表示，其中每个单词用一个 ID 表示，对于训练具有大词汇量的深度学习模型来说是不可扩展的，这在实际问题中经常出现。我们可能会得到数百万这样的单词 ID，每个以 one-hot（二进制）分类形式编码，导致数据稀疏和计算问题。我们将在第六章中更深入地讨论这个问题。

解决这个问题的一个强大方法是使用词嵌入。嵌入本质上只是将编码单词的高维度 one-hot 向量映射到较低维度稠密向量。因此，例如，如果我们的词汇量大小为 100,000，那么每个单词在 one-hot 表示中的大小将相同。相应的单词向量或*词嵌入*大小为 300。因此，高维度的 one-hot 向量被“嵌入”到具有更低维度的连续向量空间中。

在第六章中，我们深入探讨了词嵌入，探索了一种流行的无监督训练方法，即 word2vec。

在这里，我们的最终目标是解决文本分类问题，并且我们将在监督框架中训练词向量，调整嵌入的词向量以解决下游分类任务。

将单词嵌入视为基本的哈希表或查找表是有帮助的，将单词映射到它们的密集向量值。这些向量是作为训练过程的一部分进行优化的。以前，我们给每个单词一个整数索引，然后句子表示为这些索引的序列。现在，为了获得一个单词的向量，我们使用内置的`tf.nn.embedding_lookup()`函数，它有效地检索给定单词索引序列中每个单词的向量：

```py
with tf.name_scope("embeddings"):
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size,
             embedding_dimension],
             -1.0, 1.0),name='embedding')
  embed = tf.nn.embedding_lookup(embeddings, _inputs)

```

我们很快将看到单词向量表示的示例和可视化。

## LSTM 和使用序列长度

在我们开始的介绍性 RNN 示例中，我们实现并使用了基本的 vanilla RNN 模型。在实践中，我们经常使用略微更高级的 RNN 模型，其主要区别在于它们如何更新其隐藏状态并通过时间传播信息。一个非常流行的循环网络是长短期记忆（LSTM）网络。它通过具有一些特殊的*记忆机制*与 vanilla RNN 不同，这些机制使得循环单元能够更好地存储信息长时间，从而使它们能够比普通 RNN 更好地捕获长期依赖关系。

这些记忆机制并没有什么神秘之处；它们只是由一些添加到每个循环单元的更多参数组成，使得 RNN 能够克服优化问题并传播信息。这些可训练参数充当过滤器，选择哪些信息值得“记住”和传递，哪些值得“遗忘”。它们的训练方式与网络中的任何其他参数完全相同，使用梯度下降算法和反向传播。我们在这里不深入讨论更多技术数学公式，但有很多很好的资源深入探讨细节。

我们使用`tf.contrib.rnn.BasicLSTMCell()`创建一个 LSTM 单元，并将其提供给`tf.nn.dynamic_rnn()`，就像我们在本章开始时所做的那样。我们还使用我们之前创建的`_seqlens`占位符给`dynamic_rnn()`提供每个示例批次中每个序列的长度。TensorFlow 使用这个长度来停止超出最后一个真实序列元素的所有 RNN 步骤。它还返回所有随时间的输出向量（在`outputs`张量中），这些向量在真实序列的真实结尾之后都是零填充的。因此，例如，如果我们原始序列的长度为 5，并且我们将其零填充为长度为 15 的序列，则超过 5 的所有时间步的输出将为零：

```py
with tf.variable_scope("lstm"):

  lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,
                      forget_bias=1.0)
  outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed,
                    sequence_length = _seqlens,
                    dtype=tf.float32)

weights = {
  'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size,
                          num_classes],
                          mean=0,stddev=.01))
}
biases = {
  'linear_layer':tf.Variable(tf.truncated_normal([num_classes],
                         mean=0,stddev=.01))
}

# Extract the last relevant output and use in a linear layer
final_output = tf.matmul(states[1],
            weights["linear_layer"]) + biases["linear_layer"]
softmax = tf.nn.softmax_cross_entropy_with_logits(logits = final_output,
                         labels = _labels)            
cross_entropy = tf.reduce_mean(softmax)

```

我们取最后一个有效的输出向量——在这种情况下，方便地在`dynamic_rnn()`返回的`states`张量中可用，并通过一个线性层（和 softmax 函数）传递它，将其用作我们的最终预测。在下一节中，当我们查看`dynamic_rnn()`为我们的示例句子生成的一些输出时，我们将进一步探讨最后相关输出和零填充的概念。

## 训练嵌入和 LSTM 分类器

我们已经有了拼图的所有部分。让我们把它们放在一起，完成端到端的单词向量和分类模型的训练：

```py
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(_labels,1),
               tf.argmax(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction,
                 tf.float32)))*100

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for step in range(1000):
    x_batch, y_batch,seqlen_batch = get_sentence_batch(batch_size,
                             train_x,train_y,
                             train_seqlens)
    sess.run(train_step,feed_dict={_inputs:x_batch, _labels:y_batch,
                   _seqlens:seqlen_batch})

    if step % 100 == 0:
      acc = sess.run(accuracy,feed_dict={_inputs:x_batch,
                       _labels:y_batch,
                       _seqlens:seqlen_batch})
      print("Accuracy at %d: %.5f" % (step, acc))

  for test_batch in range(5):
    x_test, y_test,seqlen_test = get_sentence_batch(batch_size,
                            test_x,test_y,
                            test_seqlens)
    batch_pred,batch_acc = sess.run([tf.argmax(final_output,1),
                    accuracy],
                    feed_dict={_inputs:x_test,
                         _labels:y_test,
                         _seqlens:seqlen_test})
    print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))

   output_example = sess.run([outputs],feed_dict={_inputs:x_test,
                         _labels:y_test,
                         _seqlens:seqlen_test})
  states_example = sess.run([states[1]],feed_dict={_inputs:x_test,
                         _labels:y_test,
                         _seqlens:seqlen_test})

```

正如我们所看到的，这是一个非常简单的玩具文本分类问题：

```py
Accuracy at 0: 32.81250
Accuracy at 100: 100.00000
Accuracy at 200: 100.00000
Accuracy at 300: 100.00000
Accuracy at 400: 100.00000
Accuracy at 500: 100.00000
Accuracy at 600: 100.00000
Accuracy at 700: 100.00000
Accuracy at 800: 100.00000
Accuracy at 900: 100.00000
Test batch accuracy 0: 100.00000
Test batch accuracy 1: 100.00000
Test batch accuracy 2: 100.00000
Test batch accuracy 3: 100.00000
Test batch accuracy 4: 100.00000
```

我们还计算了由`dynamic_rnn()`生成的一个示例批次的输出，以进一步说明在前一节中讨论的零填充和最后相关输出的概念。

让我们看一个这些输出的例子，对于一个被零填充的句子（在您的随机数据批次中，您可能会看到不同的输出，当然要寻找一个`seqlen`小于最大 6 的句子）：

```py
seqlen_test[1]

Out:
4
```

```py
output_example[0][1].shape

Out: 
(6, 32)

```

这个输出有如预期的六个时间步，每个大小为 32 的向量。让我们看一眼它的值（只打印前几个维度以避免混乱）：

```py
output_example[0][1][:6,0:3]

Out:
array([[-0.44493711, -0.51363373, -0.49310589],
   [-0.72036862, -0.68590945, -0.73340571],
   [-0.83176643, -0.78206956, -0.87831545],
   [-0.87982416, -0.82784462, -0.91132098],
   [ 0.    , 0.    , 0.    ],
   [ 0.    , 0.    , 0.    ]], dtype=float32)
```

我们看到，对于这个句子，原始长度为 4，最后两个时间步由于填充而具有零向量。

最后，我们看一下`dynamic_rnn()`返回的状态向量：

```py
states_example[0][1][0:3]

Out:
array([-0.87982416, -0.82784462, -0.91132098], dtype=float32)
```

我们可以看到它方便地为我们存储了最后一个相关输出向量——其值与零填充之前的最后一个相关输出向量匹配。

此时，您可能想知道如何访问和操作单词向量，并探索训练后的表示。我们将展示如何做到这一点，包括交互式嵌入可视化，在下一章中。

### 堆叠多个 LSTMs

之前，我们专注于一个单层 LSTM 网络以便更容易解释。添加更多层很简单，使用`MultiRNNCell()`包装器将多个 RNN 单元组合成一个多层单元。

举个例子，假设我们想在前面的例子中堆叠两个 LSTM 层。我们可以这样做：

```py
num_LSTM_layers = 2
with tf.variable_scope("lstm"):

  lstm_cell_list = 
  [tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,forget_bias=1.0) 
    for ii in range(num_LSTM_layers)]
  cell = tf.contrib.rnn.MultiRNNCell(cells=lstm_cell_list, 
    state_is_tuple=True)

  outputs, states = tf.nn.dynamic_rnn(cell, embed,
                    sequence_length = _seqlens,
                    dtype=tf.float32)

```

我们首先像以前一样定义一个 LSTM 单元，然后将其馈送到`tf.contrib.rnn.MultiRNNCell()`包装器中。

现在我们的网络有两层 LSTM，当尝试提取最终状态向量时会出现一些形状问题。为了获得第二层的最终状态，我们只需稍微调整我们的索引：

```py
# Extract the final state and use in a linear layer
final_output = tf.matmul(states[num_LSTM_layers-1][1],
            weights["linear_layer"]) + biases["linear_layer"]

```

# 总结

在这一章中，我们介绍了在 TensorFlow 中的序列模型。我们看到如何通过使用`tf.scan()`和内置模块来实现基本的 RNN 模型，以及更高级的 LSTM 网络，用于文本和图像数据。最后，我们训练了一个端到端的文本分类 RNN 模型，使用了词嵌入，并展示了如何处理可变长度的序列。在下一章中，我们将深入探讨词嵌入和 word2vec。在第七章中，我们将看到一些很酷的 TensorFlow 抽象层，以及它们如何用于训练高级文本分类 RNN 模型，而且付出的努力要少得多。
