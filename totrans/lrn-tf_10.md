# 第十章。使用 TensorFlow 导出和提供模型

在本章中，我们将学习如何使用简单和高级的生产就绪方法保存和导出模型。对于后者，我们介绍了 TensorFlow Serving，这是 TensorFlow 中最实用的用于创建生产环境的工具之一。我们将从快速概述两种简单的保存模型和变量的方法开始：首先是通过手动保存权重并重新分配它们，然后是使用`Saver`类创建训练检查点以及导出我们的模型。最后，我们将转向更高级的应用程序，通过使用 TensorFlow Serving 在服务器上部署我们的模型。

# 保存和导出我们的模型

到目前为止，我们已经学习了如何使用 TensorFlow 创建、训练和跟踪模型。现在我们将学习如何保存训练好的模型。保存当前权重状态对于明显的实际原因至关重要——我们不想每次都从头开始重新训练模型，我们也希望有一种方便的方式与他人分享我们模型的状态（就像我们在第七章中看到的预训练模型一样）。

在这一部分，我们将讨论保存和导出的基础知识。我们首先介绍了一种简单的保存和加载权重到文件的方法。然后我们将看到如何使用 TensorFlow 的`Saver`对象来保持序列化模型检查点，其中包含有关权重状态和构建图的信息。

## 分配加载的权重

在训练后重复使用权重的一个天真但实用的方法是将它们保存到文件中，稍后可以加载它们并重新分配给模型。

让我们看一些例子。假设我们希望保存用于 MNIST 数据的基本 softmax 模型的权重，我们从会话中获取它们后，将权重表示为 NumPy 数组，并以我们选择的某种格式保存它们：

```py
import numpy as np
weights = sess.run(W)
np.savez(os.path.join(path, 'weight_storage'), weights)
```

鉴于我们构建了完全相同的图，我们可以加载文件并使用会话中的`.assign()`方法将加载的权重值分配给相应的变量：

```py
loaded_w = np.load(path + 'weight_storage.npz')
loaded_w = loaded_w.items()[0][1]

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, 
                                                    labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5)\
                                            .minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # Assigning loaded weights
  sess.run(W.assign(loaded_w))
  acc = sess.run(accuracy, feed_dict={x: data.test.images, 
                                       y_true: data.test.labels})

print("Accuracy: {}".format(acc))

Out: 
Accuracy: 0.9199

```

接下来，我们将执行相同的过程，但这次是针对第四章中用于 MNIST 数据的 CNN 模型。在这里，我们有八组不同的权重：两个卷积层 1 和 2 的滤波器权重及其对应的偏置，以及两组全连接层的权重和偏置。我们将模型封装在一个类中，以便方便地保持这八个参数的更新列表。

我们还为要加载的权重添加了可选参数：

```py
if weights is not None and sess is not None:
  self.load_weights(weights, sess)
```

以及在传递权重时分配其值的函数：

```py
def load_weights(self, weights, sess):
  for i,w in enumerate(weights):
    print("Weight index: {}".format(i), 
                           "Weight shape: {}".format(w.shape))
    sess.run(self.parameters[i].assign(w))
```

在整个过程中：

```py
class simple_cnn:
  def __init__(self, x_image,keep_prob, weights=None, sess=None):

    self.parameters = []
    self.x_image = x_image

    conv1 = self.conv_layer(x_image, shape=[5, 5, 1, 32])
    conv1_pool = self.max_pool_2x2(conv1)

    conv2 = self.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = self.max_pool_2x2(conv2)

    conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
    full_1 = tf.nn.relu(self.full_layer(conv2_flat, 1024))

    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    self.y_conv = self.full_layer(full1_drop, 10)

    if weights is not None and sess is not None:
      self.load_weights(weights, sess)

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name='weights')

  def bias_variable(self,shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name='biases')

  def conv2d(self,x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], 
                                      padding='SAME')

  def max_pool_2x2(self,x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
               strides=[1, 2, 2, 1], padding='SAME')

  def conv_layer(self,input, shape):
    W = self.weight_variable(shape)
    b = self.bias_variable([shape[3]])
    self.parameters += [W, b]

    return tf.nn.relu(self.conv2d(input, W) + b)

  def full_layer(self,input, size):
    in_size = int(input.get_shape()[1])
    W = self.weight_variable([in_size, size])
    b = self.bias_variable([size])
    self.parameters += [W, b]
    return tf.matmul(input, W) + b

  def load_weights(self, weights, sess):
    for i,w in enumerate(weights):
      print("Weight index: {}".format(i), 
                              "Weight shape: {}".format(w.shape))
      sess.run(self.parameters[i].assign(w))

```

在这个例子中，模型已经训练好，并且权重已保存为`cnn_weights`。我们加载权重并将它们传递给我们的 CNN 对象。当我们在测试数据上运行模型时，它将使用预训练的权重：

```py
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()

weights = np.load(path + 'cnn_weight_storage.npz')
weights = weights.items()[0][1]
cnn = simple_cnn(x_image, keep_prob, weights, sess)

cross_entropy = tf.reduce_mean(
           tf.nn.softmax_cross_entropy_with_logits(
                                                logits=cnn.y_conv,
                                                 labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(cnn.y_conv, 1), 
                                     tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

X = data.test.images.reshape(10, 1000, 784)
Y = data.test.labels.reshape(10, 1000, 10)
test_accuracy = np.mean([sess.run(accuracy,
            feed_dict={x:X[i], y_:Y[i],keep_prob:1.0})
            for i in range(10)])

sess.close()

print("test accuracy: {}".format(test_accuracy))

Out: 
Weight index: 0 Weight shape: (5, 5, 1, 32)
Weight index: 1 Weight shape: (32,)
Weight index: 2 Weight shape: (5, 5, 32, 64)
Weight index: 3 Weight shape: (64,)
Weight index: 4 Weight shape: (3136, 1024)
Weight index: 5 Weight shape: (1024,)
Weight index: 6 Weight shape: (1024, 10)
Weight index: 7 Weight shape: (10,)

test accuracy: 0.990100026131

```

我们可以获得高准确度，而无需重新训练。

## Saver 类

TensorFlow 还有一个内置的类，我们可以用于与前面的示例相同的目的，提供额外有用的功能，我们很快就会看到。这个类被称为`Saver`类（在第五章中已经简要介绍过）。

`Saver`添加了操作，允许我们通过使用称为*检查点文件*的二进制文件保存和恢复模型的参数，将张量值映射到变量的名称。与前一节中使用的方法不同，这里我们不必跟踪我们的参数——`Saver`会自动为我们完成。

使用`Saver`非常简单。我们首先通过`tf.train.Saver()`创建一个 saver 实例，指示我们希望保留多少最近的变量检查点，以及可选的保留它们的时间间隔。

例如，在下面的代码中，我们要求只保留最近的七个检查点，并且另外指定每半小时保留一个检查点（这对于性能和进展评估分析可能很有用）：

```py
saver = tf.train.Saver(max_to_keep=7, 
                      keep_checkpoint_every_n_hours=0.5)

```

如果没有给出输入，那么默认情况下会保留最后五个检查点，并且`every_n_hours`功能会被有效地禁用（默认设置为`10000`）。

接下来，我们使用`saver`实例的`.save()`方法保存检查点文件，传递会话参数、文件保存路径以及步数（`global_step`），它会自动连接到每个检查点文件的名称中，表示迭代次数。在训练模型时，这会创建不同步骤的多个检查点。

在这个代码示例中，每 50 个训练迭代将在指定目录中保存一个文件：

```py
DIR="*`path/to/model`*"withtf.Session()assess:forstepinrange(1,NUM_STEPS+1):batch_xs,batch_ys=data.train.next_batch(MINIBATCH_SIZE)sess.run(gd_step,feed_dict={x:batch_xs,y_true:batch_ys})ifstep%50==0:saver.save(sess,os.path.join(DIR,"model"),global_step=step)
```

另一个保存的文件名为*checkpoint*包含保存的检查点列表，以及最近检查点的路径：

```py
model_checkpoint_path: "model_ckpt-1000"

all_model_checkpoint_paths: "model_ckpt-700"

all_model_checkpoint_paths: "model_ckpt-750"

all_model_checkpoint_paths: "model_ckpt-800"

all_model_checkpoint_paths: "model_ckpt-850"

all_model_checkpoint_paths: "model_ckpt-900"

all_model_checkpoint_paths: "model_ckpt-950"

all_model_checkpoint_paths: "model_ckpt-1000"
```

在下面的代码中，我们使用`Saver`保存权重的状态：

```py
fromtensorflow.examples.tutorials.mnistimportinput_dataDATA_DIR='/tmp/data'data=input_data.read_data_sets(DATA_DIR,one_hot=True)NUM_STEPS=1000MINIBATCH_SIZE=100DIR="*`path/to/model`*"x=tf.placeholder(tf.float32,[None,784],name='x')W=tf.Variable(tf.zeros([784,10]),name='W')y_true=tf.placeholder(tf.float32,[None,10])y_pred=tf.matmul(x,W)cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_true))gd_step=tf.train.GradientDescentOptimizer(0.5)\ .minimize(cross_entropy)correct_mask=tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))accuracy=tf.reduce_mean(tf.cast(correct_mask,tf.float32))saver=tf.train.Saver(max_to_keep=7,keep_checkpoint_every_n_hours=1)withtf.Session()assess:sess.run(tf.global_variables_initializer())forstepinrange(1,NUM_STEPS+1):batch_xs,batch_ys=data.train.next_batch(MINIBATCH_SIZE)sess.run(gd_step,feed_dict={x:batch_xs,y_true:batch_ys})ifstep%50==0:saver.save(sess,os.path.join(DIR,"model_ckpt"),global_step=step)ans=sess.run(accuracy,feed_dict={x:data.test.images,y_true:data.test.labels})print("Accuracy: {:.4}%".format(ans*100))Out:Accuracy:90.87%
```

现在我们只需使用`saver.restore()`为相同的图模型恢复我们想要的检查点，权重会自动分配给模型：

```py
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 784],name='x')
W = tf.Variable(tf.zeros([784, 10]),name='W')
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, 
                                                    labels=y_true))
gd_step = tf.train.GradientDescentOptimizer(0.5)\
                                            .minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:

  saver.restore(sess, os.path.join(DIR,"model_ckpt-1000"))
  ans = sess.run(accuracy, feed_dict={x: data.test.images, 
                                       y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))

Out:
Accuracy: 90.87% 

```

# 在恢复之前重置图

加载的变量需要与当前图中的变量配对，因此应该具有匹配的名称。如果由于某种原因名称不匹配，那么可能会出现类似于这样的错误：

```py
NotFoundError: Key W_1 not found in checkpoint
	 [[Node: save/RestoreV2_2 = RestoreV2[
   dtypes=[DT_FLOAT], _device="/job:localhost/replica:0
   /task:0/cpu:0"](_recv_save/Const_1_0, save/RestoreV2_2
   /tensor_names, save/RestoreV2_2/shape_and_slices)]]
```

如果名称被一些旧的、无关紧要的图使用，就会发生这种情况。通过使用`tf.reset_default_graph()`命令重置图，您可以解决这个问题。

到目前为止，在这两种方法中，我们需要重新创建图以重新分配恢复的参数。然而，`Saver`还允许我们恢复图而无需重建它，通过生成包含有关图的所有必要信息的*.meta*检查点文件。

关于图的信息以及如何将保存的权重合并到其中（元信息）被称为`MetaGraphDef`。这些信息被序列化——转换为一个字符串——使用协议缓冲区（参见“序列化和协议缓冲区”），它包括几个部分。网络架构的信息保存在`graph_def`中。

这里是图信息的文本序列化的一个小样本（更多关于序列化的内容将在后面介绍）：

```py
meta_info_def {
  stripped_op_list {
    op {
      name: "ApplyGradientDescent"
      input_arg {
        name: "var"
        type_attr: "T"
        is_ref: true
      }
      input_arg {
        name: "alpha"
        type_attr: "T"
      }...

graph_def {
  node {
    name: "Placeholder"
    op: "Placeholder"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: -1
            }
            dim {
              size: 784
            }
          }
        }
      }
    }...

```

为了加载保存的图，我们使用`tf.train.import_meta_graph()`，传递我们想要的检查点文件的名称（带有*.meta*扩展名）。TensorFlow 已经知道如何处理恢复的权重，因为这些信息也被保存了：

```py
tf.reset_default_graph()DIR="*`path/to/model`*"withtf.Session()assess:saver=tf.train.import_meta_graph(os.path.join(DIR,"model_ckpt-1000.meta"))saver.restore(sess,os.path.join(DIR,"model_ckpt-1000"))ans=sess.run(accuracy,feed_dict={x:data.test.images,y_true:data.test.labels})print("Accuracy: {:.4}%".format(ans*100))
```

然而，仅仅导入图并恢复权重是不够的，会导致错误。原因是导入模型并恢复权重并不会给我们额外访问在运行会话时使用的变量（`fetches`和`feed_dict`的键）——模型不知道输入和输出是什么，我们希望计算什么度量等等。

解决这个问题的一种方法是将它们保存在一个集合中。集合是一个类似于字典的 TensorFlow 对象，我们可以以有序、可访问的方式保存我们的图组件。

在这个例子中，我们希望访问度量`accuracy`（我们希望获取）和 feed 键`x`和`y_true`。我们在将模型保存为`train_var`的名称之前将它们添加到一个集合中：

```py
train_var = [x,y_true,accuracy]
tf.add_to_collection('train_var', train_var[0])
tf.add_to_collection('train_var', train_var[1])
tf.add_to_collection('train_var', train_var[2])

```

如所示，`saver.save()`方法会自动保存图结构以及权重的检查点。我们还可以使用`saver.export_meta.graph()`显式保存图，然后添加一个集合（作为第二个参数传递）：

```py
train_var = [x,y_true,accuracy]
tf.add_to_collection('train_var', train_var[0])
tf.add_to_collection('train_var', train_var[1])
tf.add_to_collection('train_var', train_var[2]) 

saver = tf.train.Saver(max_to_keep=7,
           keep_checkpoint_every_n_hours=1)
saver.export_meta_graph(os.path.join(DIR,"model_ckpt.meta")
            ,collection_list=['train_var'])

```

现在我们从集合中检索图，从中可以提取所需的变量：

```py
tf.reset_default_graph()DIR="*`path/to/model`*"withtf.Session()assess:sess.run(tf.global_variables_initializer())saver=tf.train.import_meta_graph(os.path.join(DIR,"model_ckpt.meta")saver.restore(sess,os.path.join(DIR,"model_ckpt-1000"))x=tf.get_collection('train_var')[0]y_true=tf.get_collection('train_var')[1]accuracy=tf.get_collection('train_var')[2]ans=sess.run(accuracy,feed_dict={x:data.test.images,y_true:data.test.labels})print("Accuracy: {:.4}%".format(ans*100))Out:Accuracy:91.4%
```

在定义图形时，请考虑一旦图形已保存和恢复，您想要检索哪些变量/操作，例如前面示例中的准确性操作。在下一节中，当我们谈论 Serving 时，我们将看到它具有内置功能，可以引导导出的模型，而无需像我们在这里做的那样保存变量。

# TensorFlow Serving 简介

TensorFlow Serving 是用 C++编写的高性能服务框架，我们可以在生产环境中部署我们的模型。通过使客户端软件能够访问它并通过 Serving 的 API 传递输入，使我们的模型可以用于生产（图 10-1）。当然，TensorFlow Serving 旨在与 TensorFlow 模型无缝集成。Serving 具有许多优化功能，可减少延迟并增加预测的吞吐量，适用于实时、大规模应用。这不仅仅是关于预测的可访问性和高效服务，还涉及灵活性——通常希望出于各种原因保持模型更新，例如获得额外的训练数据以改进模型，对网络架构进行更改等。

![](img/letf_1001.png)

###### 图 10-1。Serving 将我们训练好的模型链接到外部应用程序，使客户端软件可以轻松访问。

## 概述

假设我们运行一个语音识别服务，并且我们希望使用 TensorFlow Serving 部署我们的模型。除了优化服务外，对我们来说定期更新模型也很重要，因为我们获取更多数据或尝试新的网络架构。稍微更技术化一点，我们希望能够加载新模型并提供其输出，卸载旧模型，同时简化模型生命周期管理和版本策略。

一般来说，我们可以通过以下方式实现 Serving。在 Python 中，我们定义模型并准备将其序列化，以便可以被负责加载、提供和管理版本的不同模块解析。Serving 的核心“引擎”位于一个 C++模块中，只有在我们希望控制 Serving 行为的特定调整和定制时才需要访问它。

简而言之，这就是 Serving 架构的工作方式（图 10-2）：

+   一个名为`Source`的模块通过监视插入的文件系统来识别需要加载的新模型，这些文件系统包含我们在创建时导出的模型及其相关信息。`Source`包括子模块，定期检查文件系统并确定最新相关的模型版本。

+   当它识别到新的模型版本时，*source*会创建一个*loader*。加载器将其*servables*（客户端用于执行计算的对象，如预测）传递给*manager*。根据版本策略（渐进式发布、回滚版本等），管理器处理可服务内容的完整生命周期（加载、卸载和提供）。

+   最后，管理器提供了一个接口，供客户端访问可服务的内容。

![](img/letf_1002.png)

###### 图 10-2。Serving 架构概述。

Serving 的设计特别之处在于它具有灵活和可扩展的特性。它支持构建各种插件来定制系统行为，同时使用其他核心组件的通用构建。

在下一节中，我们将使用 Serving 构建和部署一个 TensorFlow 模型，展示一些其关键功能和内部工作原理。在高级应用中，我们可能需要控制不同类型的优化和定制；例如，控制版本策略等。在本章中，我们将向您展示如何开始并理解 Serving 的基础知识，为生产就绪的部署奠定基础。

## 安装

Serving 需要安装一些组件，包括一些第三方组件。安装可以从源代码或使用 Docker 进行，我们在这里使用 Docker 来让您快速开始。Docker 容器将软件应用程序与运行所需的一切（例如代码、文件等）捆绑在一起。我们还使用 Bazel，谷歌自己的构建工具，用于构建客户端和服务器软件。在本章中，我们只简要介绍了 Bazel 和 Docker 等工具背后的技术细节。更全面的描述出现在书末的附录中。

### 安装 Serving

Docker 安装说明可以在[ Docker 网站](https://docs.docker.com/engine/installation/)上找到。

在这里，我们演示使用[Ubuntu](https://docs.docker.com/engine/installation/linux/ubuntu/)进行 Docker 设置。

Docker 容器是从本地 Docker 镜像创建的，该镜像是从 dockerfile 构建的，并封装了我们需要的一切（依赖安装、项目代码等）。一旦我们安装了 Docker，我们需要[下载 TensorFlow Serving 的 dockerfile](http://bit.ly/2t7ewMb)。

这个 dockerfile 包含了构建 TensorFlow Serving 所需的所有依赖项。

首先，我们生成镜像，然后可以运行容器（这可能需要一些时间）：

```py
docker build --pull -t $USER/tensorflow-serving-devel -f 
                                                  Dockerfile.devel .

```

现在我们在本地机器上创建了镜像，我们可以使用以下命令创建和运行容器：

```py
docker run -v $HOME/docker_files:/host_files 
                         -p 80:80 -it $USER/tensorflow-serving-devel
```

`docker run -it $USER/tensorflow-serving-devel`命令足以创建和运行容器，但我们对此命令进行了两次添加。

首先，我们添加*-v $HOME/home_dir:/docker_dir*，其中`-v`（卷）表示请求共享文件系统，这样我们就可以方便地在 Docker 容器和主机之间传输文件。在这里，我们在主机上创建了共享文件夹*docker_files*，在我们的 Docker 容器上创建了*host_files*。另一种传输文件的方法是简单地使用命令`docker cp foo.txt *mycontainer*:/foo.txt`。第二个添加是`-p <*host port*>:<*container port*>`，这使得容器中的服务可以通过指定的端口暴露在任何地方。

一旦我们输入我们的`run`命令，一个容器将被创建和启动，并且一个终端将被打开。我们可以使用命令`docker ps -a`（在 Docker 终端之外）查看我们容器的状态。请注意，每次使用`docker run`命令时，我们都会创建另一个容器；要进入现有容器的终端，我们需要使用`docker exec -it <*container id*> bash`。

最后，在打开的终端中，我们克隆并配置 TensorFlow Serving：

```py
git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving/tensorflow
./configure
```

就是这样，我们准备好了！

## 构建和导出

现在 Serving 已经克隆并运行，我们可以开始探索其功能和如何使用它。克隆的 TensorFlow Serving 库是按照 Bazel 架构组织的。Bazel 构建的源代码组织在一个工作区目录中，里面有一系列分组相关源文件的包。每个包都有一个*BUILD*文件，指定从该包内的文件构建的输出。

我们克隆库中的工作区位于*/serving*文件夹中，包含*WORKSPACE*文本文件和*/tensorflow_serving*包，稍后我们将返回到这里。

现在我们转向查看处理训练和导出模型的 Python 脚本，并看看如何以一种适合进行 Serving 的方式导出我们的模型。

### 导出我们的模型

与我们使用`Saver`类时一样，我们训练的模型将被序列化并导出到两个文件中：一个包含有关变量的信息，另一个包含有关图形和其他元数据的信息。正如我们很快将看到的，Serving 需要特定的序列化格式和元数据，因此我们不能简单地使用`Saver`类，就像我们在本章开头看到的那样。

我们要采取的步骤如下：

1.  像前几章一样定义我们的模型。

1.  创建一个模型构建器实例。

1.  在构建器中定义我们的元数据（模型、方法、输入和输出等）以序列化格式（称为 `SignatureDef`）。

1.  使用构建器保存我们的模型。

首先，我们通过使用 Serving 的 `SavedModelBuilder` 模块创建一个构建器实例，传递我们希望将文件导出到的位置（如果目录不存在，则将创建）。`SavedModelBuilder` 导出表示我们的模型的序列化文件，格式如下所需：

```py
builder = saved_model_builder.SavedModelBuilder(export_path)

```

我们需要的序列化模型文件将包含在一个目录中，该目录的名称将指定模型及其版本：

```py
export_path_base = sys.argv[-1]
export_path = os.path.join(
 compat.as_bytes(export_path_base),
 compat.as_bytes(str(FLAGS.model_version)))
```

这样，每个版本将被导出到一个具有相应路径的不同子目录中。

请注意，`export_path_base` 是从命令行输入的，使用 `sys.argv` 获取，版本作为标志保留（在上一章中介绍）。标志解析由 `tf.app.run()` 处理，我们很快就会看到。

接下来，我们要定义输入（图的输入张量的形状）和输出（预测张量）签名。在本章的第一部分中，我们使用 TensorFlow 集合对象来指定输入和输出数据之间的关系及其相应的占位符，以及用于计算预测和准确性的操作。在这里，签名起到了类似的作用。

我们使用创建的构建器实例添加变量和元图信息，使用 `SavedModelBuilder.add_meta_graph_and_variables()` 方法：

```py
builder.add_meta_graph_and_variables(
 sess, [tag_constants.SERVING],
 signature_def_map={
   'predict_images':
     prediction_signature,
   signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
     classification_signature,
 },
 legacy_init_op=legacy_init_op)
```

我们需要传递四个参数：会话、标签（用于“服务”或“训练”）、签名映射和一些初始化。

我们传递一个包含预测和分类签名的字典。我们从预测签名开始，可以将其视为在 TensorFlow 集合中指定和保存预测操作，就像我们之前看到的那样：

```py
prediction_signature = signature_def_utils.build_signature_def(
 inputs={'images': tensor_info_x},
 outputs={'scores': tensor_info_y},
 method_name=signature_constants.PREDICT_METHOD_NAME)
```

这里的 `images` 和 `scores` 是我们稍后将用来引用我们的 `x` 和 `y` 张量的任意名称。通过以下命令将图像和分数编码为所需格式：

```py
tensor_info_x = utils.build_tensor_info(x)
tensor_info_y = utils.build_tensor_info(y_conv)

```

与预测签名类似，我们有分类签名，其中我们输入关于分数（前 `k` 个类的概率值）和相应类的信息：

```py
# Build the signature_def_map
classification_inputs = utils.build_tensor_info(
                                           serialized_tf_example)
classification_outputs_classes = utils.build_tensor_info(
                                           prediction_classes)
classification_outputs_scores = utils.build_tensor_info(values)

```

```py
classification_signature = signature_def_utils.build_signature_def(
 inputs={signature_constants.CLASSIFY_INPUTS: 
                             classification_inputs},
 outputs={
   signature_constants.CLASSIFY_OUTPUT_CLASSES:
     classification_outputs_classes,
   signature_constants.CLASSIFY_OUTPUT_SCORES:
     classification_outputs_scores
 },
 method_name=signature_constants.CLASSIFY_METHOD_NAME)
```

最后，我们使用 `save()` 命令保存我们的模型：

```py
builder.save()

```

简而言之，将所有部分整合在一起，以准备在脚本执行时序列化和导出，我们将立即看到。

以下是我们主要的 Python 模型脚本的最终代码，包括我们的模型（来自第四章的 CNN 模型）：

```py
import os
import sys
import tensorflow as tf
from tensorflow.python.saved_model import builder 
                                          as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow_serving.example import mnist_input_data

tf.app.flags.DEFINE_integer('training_iteration', 10,
              'number of training iterations.')
tf.app.flags.DEFINE_integer(
                'model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def weight_variable(shape):
 initial = tf.truncated_normal(shape, stddev=0.1)
 return tf.Variable(initial,dtype='float')

def bias_variable(shape):
 initial = tf.constant(0.1, shape=shape)
 return tf.Variable(initial,dtype='float')

def conv2d(x, W):
 return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
 return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')

def main(_):
  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: mnist_export.py [--training_iteration=x] '
       '[--model_version=y] export_dir')
    sys.exit(-1)
  if FLAGS.training_iteration <= 0:
    print('Please specify a positive 
                                value for training iteration.')
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print ('Please specify a positive 
                                    value for version number.')
    sys.exit(-1)

  print('Training...')
  mnist = mnist_input_data.read_data_sets(
                                 FLAGS.work_dir, one_hot=True)
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(
                                 tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], 
                                           dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, 
                                               feature_configs)

  x = tf.identity(tf_example['x'], name='x') 
  y_ = tf.placeholder('float', shape=[None, 10])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1,28,28,1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  y = tf.nn.softmax(y_conv, name='y')
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
  train_step = tf.train.AdamOptimizer(1e-4)\
                                         .minimize(cross_entropy)

  values, indices = tf.nn.top_k(y_conv, 10)
  prediction_classes = tf.contrib.lookup.index_to_string(
   tf.to_int64(indices), 
     mapping=tf.constant([str(i) for i in xrange(10)]))

  sess.run(tf.global_variables_initializer())

  for _ in range(FLAGS.training_iteration):
    batch = mnist.train.next_batch(50)

    train_step.run(feed_dict={x: batch[0], 
                                 y_: batch[1], keep_prob: 0.5})
    print(_)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), 
                                        tf.argmax(y_,1))

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
           y_: mnist.test.labels})

  print('training accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images, 
       y_: mnist.test.labels, keep_prob: 1.0}))

  print('training is finished!')

  export_path_base = sys.argv[-1]
  export_path = os.path.join(
   compat.as_bytes(export_path_base),
   compat.as_bytes(str(FLAGS.model_version)))
  print 'Exporting trained model to', export_path
  builder = saved_model_builder.SavedModelBuilder(export_path)

  classification_inputs = utils.build_tensor_info(
                                            serialized_tf_example)
  classification_outputs_classes = utils.build_tensor_info(
                                            prediction_classes)
  classification_outputs_scores = utils.build_tensor_info(values)

  classification_signature = signature_def_utils.build_signature_def(
   inputs={signature_constants.CLASSIFY_INPUTS: 
                          classification_inputs},
   outputs={
     signature_constants.CLASSIFY_OUTPUT_CLASSES:
       classification_outputs_classes,
     signature_constants.CLASSIFY_OUTPUT_SCORES:
       classification_outputs_scores
   },
   method_name=signature_constants.CLASSIFY_METHOD_NAME)

  tensor_info_x = utils.build_tensor_info(x)
  tensor_info_y = utils.build_tensor_info(y_conv)

  prediction_signature = signature_def_utils.build_signature_def(
   inputs={'images': tensor_info_x},
   outputs={'scores': tensor_info_y},
   method_name=signature_constants.PREDICT_METHOD_NAME)

  legacy_init_op = tf.group(tf.initialize_all_tables(), 
                                  name='legacy_init_op')
  builder.add_meta_graph_and_variables(
   sess, [tag_constants.SERVING],
   signature_def_map={
     'predict_images':
       prediction_signature,
     signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
       classification_signature,
   },
   legacy_init_op=legacy_init_op)

  builder.save()

  print('new model exported!')

if __name__ == '__main__':
  tf.app.run()

```

`tf.app.run()` 命令为我们提供了一个很好的包装器，用于处理解析命令行参数。

在我们介绍 Serving 的最后部分中，我们使用 Bazel 实际导出和部署我们的模型。

大多数 Bazel *BUILD* 文件仅包含构建规则的声明，指定输入和输出之间的关系，以及构建输出的步骤。

例如，在这个 *BUILD* 文件中，我们有一个 Python 规则 `py_binary` 用于构建可执行程序。这里有三个属性，`name` 用于规则的名称，`srcs` 用于处理以创建目标（我们的 Python 脚本）的文件列表，`deps` 用于链接到二进制目标中的其他库的列表：

```py
py_binary(
    name = "serving_model_ch4",
    srcs = [
        "serving_model_ch4.py",
    ],
    deps = [
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/saved_model:builder",
        "@org_tensorflow//tensorflow/python/saved_model:constants",
        "@org_tensorflow//tensorflow/python/saved_model:loader",
        "@org_tensorflow//tensorflow/python/saved_model:
                                              signature_constants",
        "@org_tensorflow//tensorflow/python/saved_model:
                                              signature_def_utils",
        "@org_tensorflow//tensorflow/python/saved_model:
                                              tag_constants",
        "@org_tensorflow//tensorflow/python/saved_model:utils",
    ],
)
```

接下来，我们使用 Bazel 运行和导出模型，进行 1,000 次迭代训练并导出模型的第一个版本：

```py
bazel build //tensorflow_serving/example:serving_model_ch4
bazel-bin/tensorflow_serving/example/serving_model_ch4 
        --training_iteration=1000 --model_version=1 /tmp/mnist_model

```

要训练模型的第二个版本，我们只需使用：

```py
--model_version=2
```

在指定的子目录中，我们将找到两个文件，*saved_model.pb* 和 *variables*，它们包含有关我们的图（包括元数据）和其变量的序列化信息。在接下来的行中，我们使用标准的 TensorFlow 模型服务器加载导出的模型：

```py
bazel build //tensorflow_serving/model_servers:
                                           tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server 
                  --port=8000 --model_name=mnist 
                  --model_base_path=/tmp/mnist_model/ --logtostderr
```

最后，我们的模型现在已经被提供并准备在 `localhost:8000` 上运行。我们可以使用一个简单的客户端实用程序 `mnist_client` 来测试服务器：

```py
bazel build //tensorflow_serving/example:mnist_client
bazel-bin/tensorflow_serving/example/mnist_client 
                          --num_tests=1000 --server=localhost:8000

```

# 总结

本章讨论了如何保存、导出和提供模型，从简单保存和重新分配权重使用内置的`Saver`实用程序到用于生产的高级模型部署机制。本章的最后部分涉及 TensorFlow Serving，这是一个非常好的工具，可以通过动态版本控制使我们的模型商业化准备就绪。Serving 是一个功能丰富的实用程序，具有许多功能，我们强烈建议对掌握它感兴趣的读者在网上寻找更深入的技术资料。
