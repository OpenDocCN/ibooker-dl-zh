# 第6章。模型创建风格

正如您可能想象的那样，构建深度学习模型有多种方法。在前几章中，您了解了`tf.keras.Sequential`，被称为*符号API*，通常是教授模型创建的起点。您可能遇到的另一种API风格是*命令式API*。符号API和命令式API都能够构建深度学习模型。

总的来说，选择哪种API取决于风格。根据您的编程经验和背景，其中一种可能对您来说更自然。在本章中，您将学习如何使用两种API构建相同的模型。具体来说，您将学习如何使用[CIFAR-10图像数据集](https://oreil.ly/W81qK)构建图像分类模型。该数据集包含10种常见的*类别*或图像类别。与之前使用的花卉图像一样，CIFAR-10图像作为TensorFlow分发的一部分提供。然而，花卉图像是JPEG格式，而CIFAR-10图像是NumPy数组。为了将它们流式传输到训练过程中，您将使用`from_tensor_slices`方法，而不是像在[第5章](ch05.xhtml#data_pipelines_for_streaming_ingestion)中所做的`flow_from_directory`方法。

通过使用`from_tensor_slices`建立数据流程后，您将首先使用符号API构建和训练图像分类模型，然后使用命令式API。您将看到，无论如何构建模型架构，结果都非常相似。

# 使用符号API

您已经在本书的示例中看到了符号API`tf.keras.Sequential`的工作原理。在`tf.keras.Sequential`中有一堆层，每个层对输入数据执行特定操作。由于模型是逐层构建的，这是一种直观的方式来设想这个过程。在大多数情况下，您只有一个输入源（在本例中是一系列图像），输出是输入图像的类别。在[“使用TensorFlow Hub实现模型”](ch04.xhtml#model_implementation_with_tensorflow_hub)中，您学习了如何使用TensorFlow Hub构建模型。模型架构是使用顺序API定义的，如[图6-1](#sequential_api_pattern_and_data_flow)所示。

![顺序API模式和数据流](Images/t2pr_0601.png)

###### 图6-1。顺序API模式和数据流

在本节中，您将学习如何使用此API构建和训练一个使用CIFAR-10图像的图像分类模型。

## 加载CIFAR-10图像

CIFAR-10图像数据集包含10个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。所有图像大小为32×32像素，带有三个通道（RGB）。

首先导入必要的库：

```py
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pylab as plt

(train_images, train_labels), (test_images, test_labels) = 
datasets.cifar10.load_data()
```

此代码将CIFAR-10图像下载到您的Python运行时中，分为训练集和测试集。您可以使用`type`语句验证格式：

```py
print(type(train_images))
```

输出将是一个数据类型：

```py
<class 'numpy.ndarray'>
```

还重要的是要知道数组的形状，您可以使用以下命令找到：

```py
print(train_images.shape, train_labels.shape)
```

以下是图像和标签的数组形状： 

```py
(50000, 32, 32, 3) (50000, 1)
```

您可以对测试数据执行相同的操作：

```py
print(test_images.shape, test_labels.shape)
```

您应该得到以下输出：

```py
(10000, 32, 32, 3) (10000, 1)
```

从输出中可以看出，CIFAR-10数据集包含50,000个训练图像，每个图像大小为32×32×3像素。伴随的50,000个标签是一个一维数组，表示图像类别的索引。同样，还有10,000个测试图像和相应的标签。标签索引对应以下名称：

```py
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

因此，索引0表示标签“飞机”，而索引9表示“卡车”。

## 检查标签分布

现在是时候找出这些类别的分布并查看一些图像了。通过查看训练标签的分布，可以了解每个类别有多少样本，使用NumPy的`unique`函数：

```py
unique, counts = np.unique(train_labels, return_counts=True)
```

这将返回每个标签的样本计数。要显示它：

```py
print(np.asarray((unique, counts)))
```

它将显示以下内容：

```py
[[ 0 1 2 3 4 5 6 7 8 9]
 [5000 5000 5000 5000 5000 5000 5000 5000 5000 5000]]
```

这意味着每个标签（类）中有5,000张图片。训练数据在所有标签之间均匀分布。

同样，您可以验证测试数据的分布：

```py
unique, counts = np.unique(test_labels, return_counts=True)
print(np.asarray((unique, counts)))
```

输出确认了每个标签有1,000张图片：

```py
[[ 0 1 2 3 4 5 6 7 8 9]
 [1000 1000 1000 1000 1000 1000 1000 1000 1000 1000]]
```

## 检查图像

让我们看一些图像，以确保它们的数据质量。在这个练习中，您将随机抽样并显示训练数据集中的50,000张图像中的25张。

TensorFlow如何进行这种随机选择？图像从0到49,999进行索引。要从此范围中随机选择有限数量的索引，使用Python的`random`库，该库以Python列表作为输入，并从中随机选择有限数量的样本：

```py
selected_elements = random.sample(a_list, 25)
```

此代码从`a_list`中随机选择25个元素，并将结果存储在`selected_elements`中。如果`a_list`对应于图像索引，则`selected_elements`将包含从`a_list`中随机抽取的25个索引。您将使用`selected_elements`来访问和显示这25张训练图像。

现在您需要创建`train_idx`，该列表保存训练图像的索引。您将使用Python的`range`函数创建一个包含0到49,999之间整数的对象：

```py
range(len(train_labels))
```

前面的代码创建了一个`range`对象，其中包含从0开始到`len(train_labels)`或列表`training_labels`的长度的整数。

现在，将`range`对象转换为Python列表：

```py
list(range(len(train_labels)))
```

这个列表现在已准备好作为Python`random.sample`函数的输入。现在您可以开始编写代码了。

首先，创建`train_idx`，这是一个从0到49,999的索引列表：

```py
train_idx = list(range(len(train_labels)))
```

然后使用`random`库生成随机选择：

```py
import random
random.seed(2)
random_sel = random.sample(train_idx, 25)
```

第二行中的种子操作确保您的选择是可重现的，这对于调试很有帮助。您可以为`seed`使用任何整数。

`random_sel`列表将保存25个随机选择的索引，看起来像这样：

```py
[3706,
 6002,
 5562,
 23662,
 11081,
 48232,
…
```

现在您可以根据这些索引绘制图像并显示它们的标签：

```py
plt.figure(figsize=(10,10))
for i in range(len(random_sel)):
 plt.subplot(5,5,i+1)
 plt.xticks([])
 plt.yticks([])
 plt.grid(False)
 plt.imshow(train_images[random_sel[i]], cmap=plt.cm.binary)
 plt.xlabel(CLASS_NAMES[train_labels[random_sel[i]][0]])
plt.show()
```

此代码片段显示了一个包含25张图像及其标签的面板，如[图6-2](Images/#twenty-five_images_from_the_cifar-ten_da)所示。（由于这是一个随机样本，您的结果会有所不同。）

![从CIFAR-10数据集中随机选择的25张图像](Images/t2pr_0602.png)

###### 图6-2。从CIFAR-10数据集中随机选择的25张图像

## 构建数据管道

在本节中，您将使用`from_tensor_slices`构建数据摄入管道。由于只有两个分区，训练和测试，您需要在训练过程中将测试分区的一半保留为交叉验证。选择前500个作为交叉验证数据，剩下的500个作为测试数据：

```py
validation_dataset = tf.data.Dataset.from_tensor_slices(
(test_images[:500], test_labels[:500]))

test_dataset = tf.data.Dataset.from_tensor_slices(
(test_images[500:], test_labels[500:]))
```

此代码基于图像索引创建了两个数据集对象，`validation_dataset`和`test_dataset`，每个集合中有500个样本。

现在为训练数据创建一个类似的数据集对象：

```py
train_dataset = tf.data.Dataset.from_tensor_slices(
(train_images, train_labels))
```

这里使用了所有的训练图像。您可以通过计算`train_dataset`中的样本数量来确认：

```py
train_dataset_size = len(list(train_dataset.as_numpy_iterator()))
print('Training data sample size: ', train_dataset_size)
```

这是预期结果：

```py
Training data sample size: 50000
```

## 为训练批处理数据

要完成用于训练的数据摄入管道的设置，您需要将训练数据划分为批次。批次的大小，或训练样本的数量，是模型训练过程中更新模型权重和偏差所需的数量，然后沿着一步减少误差梯度。

使用以下代码对训练数据进行批处理，首先对训练数据集进行洗牌，然后创建多个包含200个样本的批次：

```py
TRAIN_BATCH_SIZE = 200
train_dataset = train_dataset.shuffle(50000).batch(
TRAIN_BATCH_SIZE)
```

同样，您将对交叉验证和测试数据执行相同的操作：

```py
validation_dataset = validation_dataset.batch(500)
test_dataset = test_dataset.batch(500)

STEPS_PER_EPOCH = train_dataset_size // TRAIN_BATCH_SIZE
VALIDATION_STEPS = 1 #validation data // validation batch size
```

交叉验证和测试数据集各包含一个500样本批次。代码设置参数以通知训练过程应该期望多少批次的训练和验证数据。训练数据的参数是`STEPS_PER_EPOCH`。交叉验证数据的参数是`VALIDATION_STEPS`，设置为1，因为数据大小和批次大小都是500。请注意，双斜杠（//）表示*地板除法*（即向下取整到最接近的整数）。

现在您的训练和验证数据集已经准备好了，下一步是使用符号API构建模型。

## 构建模型

现在您已经准备好构建模型了。以下是一个使用`tf.keras.Sequential`类包装的一堆层构建的深度学习图像分类模型的示例代码：

```py
model = tf.keras.Sequential([
 tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
 kernel_initializer='glorot_uniform', padding='same', 
 input_shape = (32,32,3)),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
 tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
 kernel_initializer='glorot_uniform', 
 padding='same'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(256, activation='relu', 
 kernel_initializer='glorot_uniform'),
 tf.keras.layers.Dense(10, activation='softmax', 
 name = 'custom_class')
])
model.build([None, 32, 32, 3])
```

接下来，使用为分类任务指定的损失函数编译模型：

```py
model.compile(
 loss='sparse_categorical_crossentropy',
 optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, 
 momentum=0.9),
 metrics=['accuracy'])
```

为了想象模型如何通过不同层处理和转换数据，您可能希望绘制模型架构，包括它期望的张量的输入和输出形状。您可以使用以下命令：

```py
tf.keras.utils.plot_model(model, show_shapes=True)
```

在运行此命令之前，您可能需要安装`pydot`和`graphviz`库：

```py
pip install pydot
pip install graphviz
```

[图6-3](#image_classification_model_architecture)展示了模型架构。问号表示表示样本大小的维度，在执行期间才知道。这是因为模型设计为适用于任何大小的训练样本。处理样本大小所需的内存是无关紧要的，不需要在模型架构级别指定。相反，所需的内存将在训练执行期间定义。

接下来，开始训练过程：

```py
hist = model.fit(
 train_dataset,
 epochs=5, steps_per_epoch=STEPS_PER_EPOCH,
 validation_data=validation_dataset,
 validation_steps=VALIDATION_STEPS).history
```

您的结果应该与[图6-4](#model_training_results)中的结果类似。

这是如何利用`tf.keras.Sequential`来构建和训练深度学习模型的。正如您所看到的，只要您指定输入和输出形状与图像和标签一致，您可以堆叠任意多的层。训练过程也非常常规；它不偏离您在[第5章](ch05.xhtml#data_pipelines_for_streaming_ingestion)中看到的内容。

![图像分类模型架构](Images/t2pr_0603.png)

###### 图6-3. 图像分类模型架构

![模型训练结果](Images/t2pr_0604.png)

###### 图6-4. 模型训练结果

在我们查看命令式API之前，我们将进行一个快速的绕道：您需要了解Python中的类继承的一些知识才能理解命令式API。

# 理解继承

*继承*是面向对象编程中使用的一种技术。它使用*类*的概念来封装与特定类型对象相关的属性和方法。它还处理不同类型对象之间的关系。继承是允许特定类使用另一个类中的方法的手段。

通过一个简单的例子更容易理解这个工作原理。想象我们有一个名为`vehicle`的基类（或父类）。我们还有另一个类`truck`，它是`vehicle`的*子类*：这也被称为*派生类*或*继承类*。我们可以定义`vehicle`类如下：

```py
class vehicle():
 def __init__(self, make, model, horsepower, weight):
 self.make = make
 self.model = model
 self.horsepower = horsepower
 self.weight = weight

 def horsepower_to_weight_ratio(self, horsepower, weight):
 hp_2_weight_ratio = horsepower / weight
 return hp_2_weight_ratio
```

这段代码展示了定义类的常见模式。它有一个构造函数`__init__`，用于初始化类的属性，比如制造商、型号、马力和重量。然后有一个名为`horsepower_to_weight_ratio`的函数，正如您可能猜到的，它计算车辆的马力重量比（我们将其称为HW比）。这个函数也可以被`vehicle`类的任何子类访问。

现在让我们创建`truck`，作为`vehicle`的子类：

```py
class truck(vehicle):
 def __init__(self, make, model, horsepower, weight, payload):
 super().__init__(make, model, horsepower, weight)
 self.payload = payload

 def __call__(self, horsepower, payload):
 hp_2_payload_ratio = horsepower / payload
 return hp_2_payload_ratio
```

在这个定义中，`class truck(vehicle)`表示`truck`是`vehicle`的子类。

在构造函数`__init__`中，`super`返回父类`vehicle`的临时对象给`truck`类。然后这个对象调用父类的`__init__`，这使得`truck`类能够重用父类中定义的相同属性：制造商、型号、马力和重量。然而，卡车还有一个独特的属性：有效载荷。这个属性*不是*从基类继承的；相反，它是在`truck`类中定义的。您可以用`self.payload = payload`定义有效载荷。这里，`self`关键字指的是这个类的实例。在这种情况下，它是一个`truck`实例，而`payload`是您为这个属性定义的任意名称。

接下来是一个`__call__`函数。这个函数使`truck`类“可调用”。在我们探讨`__call__`做什么或类可调用意味着什么之前，让我们定义一些参数并创建一个`truck`实例：

```py
MAKE = 'Tesla'
MODEL = 'Cybertruck'
HORSEPOWER = 800 #HP
WEIGHT = 3000 #kg
PAYLOAD = 1600 #kg

MyTruck = truck(MAKE, MODEL, HORSEPOWER, WEIGHT, PAYLOAD)
```

为了确保这样做得当，请打印这些属性：

```py
print('Make: ', MyTruck.make,
 '\nModel: ', MyTruck.model,
 '\nHorsepower (HP): ', MyTruck.horsepower,
 '\nWeight (kg): ', MyTruck.weight,
 '\nPayload (kg): ', MyTruck.payload)
```

这应该产生以下输出：

```py
Make: Tesla
Model: Cybertruck
Horsepower (HP): 800
Weight (kg): 3000
Payload (kg): 1600
```

让一个Python类变得*可调用*意味着什么？假设您是一名砌砖工，需要在卡车上运送重物。对您来说，卡车最重要的属性是其马力与有效载荷比（HP比率）。幸运的是，您可以创建一个`truck`对象的实例，并立即计算比率：

```py
MyTruck(HORSEPOWER, PAYLOAD)
```

输出将是0.5。

这意味着`MyTruck`实例实际上有一个与之关联的值。这个值被定义为马力与有效载荷比。这个计算是由`truck`类的`__call__`函数完成的，这是Python类的内置函数。当这个函数被显式定义为执行某种逻辑时，它几乎像一个函数调用。再看一下这行代码：

```py
MyTruck(HORSEPOWER, PAYLOAD)
```

如果您只看到这一行，您可能会认为`MyTruck`是一个函数，而`HORSEPOWER`和`PAYLOAD`是输入。

通过显式定义`__call__`方法来计算HP比率，您使`truck`类可调用；换句话说，您使其表现得像一个函数。现在它可以像Python函数一样被调用。

接下来我们想要找到我们的对象`MyTruck`的HW比率。您可能会注意到`truck`类中没有为此定义任何方法。然而，由于父类`vehicle`中确实有这样一个方法，`horsepower_to_weight_ratio`，`MyTruck`可以使用这个方法进行计算。这是*类继承*的演示，子类可以使用父类直接定义的方法。要做到这一点，您可以使用：

```py
MyTruck.horsepower_to_weight_ratio(HORSEPOWER, WEIGHT)
```

输出是0.26666666666666666。

# 使用命令式API

看过Python的类继承如何工作后，您现在可以学习如何使用命令式API构建模型。命令式API也被称为*模型子类API*，因为您构建的任何模型实际上都是从一个“Model”类继承的。如果您熟悉面向对象编程语言，如C#、C++或Java，那么命令式风格应该感觉很熟悉。

## 将模型定义为一个类

在前面的部分中，您如何定义您构建的模型为一个类？让我们看看代码：

```py
class myModel(tf.keras.Model):
 def __init__(self, input_dim):
 super(myModel, self).__init__()
 self.conv2d_initial = tf.keras.layers.Conv2D(32, 
 kernel_size=(3, 3),
 activation='relu',
 kernel_initializer='glorot_uniform',
 padding='same',
 input_shape = (input_dim,input_dim,3))
 self.cov2d_mid = tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
 activation='relu',
 kernel_initializer='glorot_uniform',
 padding='same')
 self.maxpool2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
 self.flatten = tf.keras.layers.Flatten()
 self.dense = tf.keras.layers.Dense(256, activation='relu',
 kernel_initializer='glorot_uniform')
 self.fc = tf.keras.layers.Dense(10, activation='softmax',
 name = 'custom_class')

 def call(self, input_dim):
 x = self.conv2d_initial(input_dim)
 x = self.maxpool2d(x)
 x = self.cov2d_mid(x)
 x = self.maxpool2d(x)
 x = self.flatten(x)
 x = self.dense(x)
 x = self.fc(x)

 return x
```

正如前面的代码所示，`myModel`类从父类`tf.keras.Model`继承，就像我们的`truck`类从父类`vehicle`继承一样。

模型中的层被视为`myModel`类中的属性。这些属性在构造函数`__init__`中定义。（回想一下，属性是参数，如马力、制造商和型号，而层是通过语法定义的，如`tf.keras.layers.Conv2D`。）对于模型中的第一层，代码是：

```py
self.conv2d_initial = tf.keras.layers.Conv2D(32, 
 kernel_size=(3, 3),
 activation='relu',
 kernel_initializer='glorot_uniform',
 padding='same',
 input_shape = (input_dim,input_dim,3))
```

正如您所看到的，分配层只需要一个名为`conv2d_initial`的对象。在这个定义中的另一个重要元素是，您可以将用户定义的参数传递给属性。在这里，构造函数`__init__`期望用户提供一个参数`input_dim`，它将传递给`input_shape`参数。

这种风格的好处在于，如果您想要为其他类型的图像尺寸重用此模型架构，您无需创建新模型；只需将图像尺寸作为用户参数传递给此类，您将获得一个可以处理您选择的图像尺寸的类的实例。实际上，您可以向构造函数的输入添加更多用户参数，并将它们传递到对象的不同部分，比如`kernel_size`。这是面向对象编程风格促进代码重用的一种方式。

让我们再看一下另一个层的定义：

```py
self.maxpool2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
```

这个层将在模型架构中多次使用，但您只需要定义一次。但是，如果您需要不同的超参数值，比如不同的`pool_size`，那么您需要创建另一个属性：

```py
self.maxpool2d_2 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5))
```

在这里，没有必要这样做，因为我们的模型架构重用了`maxpool2d`。

现在让我们看一下`call`函数。回想一下，通过处理Python内置的`__call__`函数中的某些类型的逻辑或计算，您可以使一个类可调用。在类似的精神中，TensorFlow创建了一个内置的`call`函数，使模型类可调用。在这个函数内部，您可以看到层的顺序与顺序API中的顺序相同（如您在[“构建模型”](#building_the_model)中看到的）。唯一的区别是这些层现在由类属性表示，而不是硬编码的层定义。

此外，请注意，在以下输入中，用户参数`input_dim`被传递给属性：

```py
def call(self, input_dim)
```

这可以根据您的图像尺寸要求为您的模型提供灵活性和可重用性。

在`call`函数中，对象`x`被用来迭代表示模型层。在声明最终层`self.fc(x)`之后，它将`x`作为模型返回。

要创建一个处理32×32像素CIFAR-10图像尺寸的模型实例，请将实例定义为：

```py
mdl = myModel(32)
```

此代码创建了一个`myModel`实例，并用CIFAR-10数据集的图像尺寸进行初始化。这个模型表示为`mdl`对象。接下来，就像您在[“构建模型”](#building_the_model)中所做的那样，您必须使用相同的语法指定损失函数和优化器选择：

```py
mdl.compile(loss='sparse_categorical_crossentropy',
 optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, 
 momentum=0.9),
 metrics=['accuracy'])
```

现在您可以启动训练例程：

```py
mdl_hist = mdl.fit(
 train_dataset,
 epochs=5, steps_per_epoch=STEPS_PER_EPOCH,
 validation_data=validation_dataset,
 validation_steps=VALIDATION_STEPS).history
```

您可以期望与[图6-5](#imperative_api_model_training_results)中的训练结果类似的训练结果。

![命令式API模型训练结果](Images/t2pr_0605.png)

###### 图6-5。命令式API模型训练结果

使用符号API和命令式API训练的模型应该产生类似的训练结果。

# 选择API

您已经看到符号API和命令式API可以用来构建具有相同架构的模型。在大多数情况下，您选择API的依据将基于您喜欢的风格和对语法的熟悉程度。然而，值得注意的是有一些值得注意的权衡。

符号API的最大优势是其代码可读性，这使得维护更容易。可以直观地看到模型架构，并且可以看到输入数据通过不同层的张量流动，就像一个图一样。使用符号API构建的模型还可以利用`tf.keras.utils.plot_model`来显示模型架构。通常，这是我们设计深度学习模型时的起点。

当涉及到实现模型架构时，命令式API绝对不像符号API那样直接。正如您所了解的，这种风格源自类继承的面向对象编程技术。如果您更喜欢将模型视为一个对象而不是一堆操作层，您可能会发现这种风格更直观，如[图6-6](#the_tensorflow_modelapostrophes_imperati)所示。

![TensorFlow模型的命令式API（也称为模型子类化）](Images/t2pr_0606.png)

###### 图6-6。TensorFlow模型的命令式API（也称为模型子类化）

实质上，您构建的任何模型都是基本模型`tf.keras.Model`的*扩展*或继承类。因此，当您构建一个模型时，实际上只是创建了一个继承了基本模型所有属性和函数的类的实例。要适应不同维度的图像模型，您只需使用不同的超参数实例化它。如果重用相同的模型架构是您的工作流程的一部分，那么命令式API是保持代码清洁简洁的明智选择。

# 使用内置训练循环

到目前为止，您已经看到启动模型训练过程所需的只是`fit`函数。这个函数为您包装了许多复杂的操作，如[图6-7](#elements_in_a_built-in_training_loop)所示。

![内置训练循环中的要素](Images/t2pr_0607.png)

###### 图6-7。内置训练循环中的要素

模型对象包含有关架构、损失函数、优化器和模型指标的信息。在`fit`中，您提供训练和验证数据，要训练的时期数，以及多久更新模型参数并使用验证数据进行测试。

这就是您需要做的全部。内置训练循环知道当一个训练时期完成时，是时候使用批处理验证数据执行交叉验证了。这很方便清晰，使您的代码非常易于维护。输出在每个时期结束时产生，如图[6-4](#model_training_results)和[6-5](#imperative_api_model_training_results)所示。

如果您需要查看训练过程的详细信息，例如在时期结束之前每个增量改进步骤中的模型准确性，或者如果您想要创建自己的训练指标，那么您需要构建自己的训练循环。接下来，我们将看看这是如何工作的。

# 创建和使用自定义训练循环

使用自定义训练循环，您失去了`fit`函数的便利性；相反，您需要编写代码来编排训练过程。假设您想要在每个步骤中监视模型参数在一个时期内的准确性。您可以从[“构建模型”](#building_the_model)中重用模型对象（`model`）。

## 创建循环的要素

首先，创建优化器和损失函数对象：

```py
optimizer = tf.keras.optimizers.SGD(
learning_rate=0.1, 
 momentum=0.9)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
from_logits=True)
```

然后创建代表模型指标的对象：

```py
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
```

这段代码为模型准确性创建了两个对象：一个用于训练数据，一个用于验证数据。使用`SparseCategoricalAccuracy`函数是因为输出是一个计算预测与标签匹配频率的指标。

接下来，对于训练，您需要创建一个函数：

```py
@tf.function
def train_step(train_data, train_label):
    with tf.GradientTape() as tape:
    logits = model(train_data, training=True)
    loss_value = loss_fn(train_label, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(train_label, logits)
    return loss_value
```

在前面的代码中，`@tf.function`是一个Python装饰器，它将一个以张量作为输入的函数转换为一个可以加速函数执行的形式。这个函数还包括一个新对象`tf.GradientTape`。在这个范围内，TensorFlow为您执行梯度下降算法；它通过不同iating每个节点中的训练权重相对于损失函数的梯度来自动计算梯度。

以下行指示`GradientTape`对象的范围：

```py
with tf.GradientTape() as tape
```

接下来的代码行表示您调用`model`将训练数据映射到一个输出（`logits`）：

```py
logits = model(train_data, training=True)
```

现在计算损失函数的输出，将模型输出与真实标签`train_label`进行比较：

```py
loss_value = loss_fn(train_label, logits)
```

然后使用模型的参数（`trainable_weights`）和损失函数的值（`loss_value`）来计算梯度并更新模型的准确性。

您需要对验证数据执行相同的操作：

```py
@tf.function
def test_step(validation_data, validation_label):
 val_logits = model(validation_data, training=False)
 val_acc_metric.update_state(validation_label, val_logits)
```

## 将要素组合在一起形成自定义训练循环

现在您已经拥有所有的要素，可以开始创建自定义训练循环了。以下是一般的步骤：

1.  使用`for`循环来迭代每个时期。

1.  在每个时期内，使用另一个`for`循环来迭代数据集中的每个批次。

1.  在每个批次中，打开一个`GradientTape`对象范围。

1.  在范围内，计算损失函数。

1.  在范围外，检索模型权重的梯度。

1.  使用优化器根据梯度值更新模型权重。

以下是自定义训练循环的代码片段：

```py
import time

epochs = 2
for epoch in range(epochs):
 print("\nStarting epoch %d" % (epoch,))
 start_time = time.time()

 # Iterate dataset batches
 for step, (x_batch_train, y_batch_train) in 
 enumerate(train_dataset):
 loss_value = train_step(x_batch_train, y_batch_train)

    # In every 100 batches, log results.
    if step % 100 == 0:
         print(
         "Training loss (for one batch) at step %d: %.4f"
         % (step, float(loss_value))
         )
 print("Sample processed so far: %d samples" % 
 ((step + 1) * TRAIN_BATCH_SIZE))

 # Show accuracy metrics after each epoch is completed
 train_accuracy = train_acc_metric.result()
 print("Training accuracy over epoch: %.4f" % 
 (float(train_accuracy),))

 # Reset training metrics before next epoch starts
 train_acc_metric.reset_states()

 # Test with validation data at end of each epoch
 for x_batch_val, y_batch_val in validation_dataset:
 test_step(x_batch_val, y_batch_val)

 val_accuracy = val_acc_metric.result()
 val_acc_metric.reset_states()
 print("Validation accuracy: %.4f" % (float(val_accuracy),))
 print("Time taken: %.2fs" % (time.time() - start_time))
```

[图6-8](#output_from_executing_the_custom_trainin)显示了执行自定义训练循环的典型输出。

![执行自定义训练循环的输出](Images/t2pr_0608.png)

###### 图6-8。执行自定义训练循环的输出

正如您所看到的，每个200个样本批次结束时，训练循环会计算并显示损失函数的值，让您可以查看训练过程内部发生的情况。如果您需要这种可见性，构建自己的自定义训练循环将提供它。只需知道，这比`fit`函数的便捷内置训练循环需要更多的努力。

# 总结

在本章中，您学习了如何使用符号和命令式API在TensorFlow中构建深度学习模型。通常情况下，两者都能够实现相同的架构，特别是当数据从输入到输出以直线流动时（意味着没有反馈或多个输入）。您可能会看到使用命令式API的复杂架构和定制实现的模型。选择适合您情况、方便和可读性的API。

无论您选择哪种方式，您都将使用内置的`fit`函数以相同的方式训练模型。`fit`函数执行内置的训练循环，并让您不必担心如何实际编排训练过程。诸如计算损失函数、将模型输出与真实标签进行比较以及使用梯度值更新模型参数等细节都在幕后为您处理。您将看到的是每个时代结束时的结果：模型相对于训练数据和交叉验证数据的准确性。

如果您需要查看每个批次训练数据中模型的准确性等时代内部发生的情况，那么您需要编写自己的训练循环，这是一个相当费力的过程。

在下一章中，您将看到模型训练过程中提供的其他选项，这些选项提供了更多的灵活性，而无需进行自定义训练循环的复杂编码过程。
