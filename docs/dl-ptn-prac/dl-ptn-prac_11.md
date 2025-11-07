# 9 自编码器

本章涵盖了

+   理解深度神经网络（DNN）和卷积神经网络（CNN）自编码器的设计原则和模式

+   使用过程设计模式编码这些模型

+   训练自编码器时的正则化

+   使用自编码器进行压缩、去噪和超分辨率

+   使用自编码器进行预训练以提高模型泛化能力

到目前为止，我们只讨论了监督学习模型。*自编码器模型*属于无监督学习的范畴。提醒一下，在监督学习中，我们的数据由特征（例如，图像数据）和标签（例如，类别）组成，我们训练模型学习从特征预测标签。在无监督学习中，我们可能没有标签或者不使用它们，我们训练模型在数据中找到相关模式。你可能会问，没有标签我们能做什么？我们可以做很多事情，自编码器就是可以从未标记数据中学习的一种模型架构。

自编码器是无监督学习的根本深度学习模型。即使没有人工标记，自编码器也可以学习图像压缩、表示学习、图像去噪、超分辨率和预训练任务——我们将在本章中介绍每个这些内容。

那么，无监督学习是如何与自编码器一起工作的呢？尽管我们没有图像数据的标签，我们可以操作图像使其同时成为输入数据和输出标签，并训练模型来预测输出标签。例如，输出标签可以是简单的输入图像——在这里，模型将学习恒等函数。或者，我们可以复制图像并向其添加噪声，然后使用噪声版本作为输入，原始图像作为输出标签——这就是我们的模型学习去噪图像的方式。在本章中，我们将介绍这些以及其他几种将输入图像转换为输出标签的技术。

## 9.1 深度神经网络自编码器

我们将从这个章节开始介绍自编码器的经典深度神经网络版本。虽然你可以仅使用 DNN 学习到有趣的东西，但它不适用于图像数据，所以接下来的几节我们将转向使用 CNN 自编码器。

### 9.1.1 自编码器架构

DNN 自编码器如何有用的一个例子是在图像重建方面。我最喜欢的重建之一，通常用作预训练任务，是拼图。在这种情况下，输入图像被分成九个拼块，然后随机打乱。重建任务就是预测拼块被打乱的顺序。由于这个任务本质上是一个多值回归器输出，它非常适合传统的 CNN，其中多类分类器被多值回归器所取代。

自动编码器由两个基本组件组成：编码器和解码器。对于图像重建，*编码器*学习一个最优（或几乎最优）的方法来逐步将图像数据池化到潜在空间，而*解码器*学习一个最优（或几乎最优）的方法来逐步反池化潜在空间以进行图像重建。重建任务决定了表示学习和转换学习的类型。例如，在恒等函数中，重建任务是重建输入图像。但你也可以重建一个无噪声的图像（通过降噪）或更高分辨率的图像（超分辨率）。这些类型的重建与自动编码器工作得很好。

让我们看看编码器和解码器在自动编码器中如何协同工作来完成这些类型的重建。基本的自动编码器架构，如图 9.1 所示，实际上有三个关键组件，编码器和解码器之间有潜在空间。编码器对输入进行表示学习，学习一个函数 *f*(*x*) = *x*'。这个 *x*' 被称为 *潜在空间*，它是从 *x* 学习到的低维表示。然后解码器从潜在空间进行转换学习，以执行原始图像的某种形式的重建。

![图片](img/CH09_F01_Ferlitsch.png)

图 9.1 自动编码器宏架构中学习图像输入/输出的恒等函数

假设图 9.1 中的自动编码器学习恒等函数 *f*(*x*) = *x*。由于潜在空间 *x*' 的维度更低，我们通常将这种形式的自动编码器描述为学习在数据集中压缩图像的最优方式（编码器）然后解压缩图像（解码器）。我们也可以将这描述为函数序列：编码器(*x*) = *x*', 解码器(*x*') = *x*。

换句话说，数据集代表了一种分布，对于这种分布，自动编码器学习最优的方法来压缩图像到更低的维度，并学习最优的解压缩方法来重建图像。让我们更详细地看看编码器和解码器，然后看看我们如何训练这种模型。

### 9.1.2 编码器

学习恒等函数的基本自动编码器形式使用密集层（隐藏单元）。池化是通过编码器中的每一层逐渐减少节点（隐藏单元）的数量来实现的，而反池化是通过每一层逐渐增加节点数量来学习的。最终反池化密集层中的节点数与输入像素数相同。

对于恒等函数，图像本身是标签。你不需要知道图像描绘的是什么，无论是猫、狗、马、飞机还是其他什么。当模型训练时，图像既是自变量（特征）也是因变量（标签）。

以下代码是自动编码器学习恒等函数的编码器的一个示例实现。它遵循图 9.1 中描述的过程，通过`layers`参数逐步池化节点（隐藏单元）的数量。编码器的输出是潜在空间。

我们首先将图像输入展平成一个一维向量。参数`layers`是一个列表；元素的数量是隐藏层的数量，元素值是该层的单元数。由于我们是逐步池化，每个后续元素的值逐渐减小。与用于分类的 CNN 相比，编码器在层上通常较浅，我们添加批量归一化以增强其正则化效果：

```
def encoder(x, layers):
    ''' Construct the Encoder 
        x     : input to the encoder
        layers: number of nodes per layer
    '''
    x = Flatten()(x)                ❶

    for layer in layers:            ❷
        n_nodes = layer['n_nodes']
        x = Dense(n_nodes)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    return x                        ❸
```

❶ 输入图像的展平

❷ 逐步单元池化（降维）

❸ 编码（潜在空间）

### 9.1.3 解码器

现在，让我们看看自动编码器解码器的一个示例实现。同样，遵循图 9.1 中描述的过程，我们通过`layers`参数逐步反池化节点（隐藏单元）的数量。解码器的输出是重构的图像。为了与编码器对称，我们以相反的方向遍历`layers`参数。最终`Dense`层的激活函数是`sigmoid`。为什么？每个节点代表一个重构的像素。由于我们已经将图像数据归一化到 0 到 1 之间，我们希望将输出挤压到相同的 0 到 1 范围内。

最后，为了重构图像，我们对来自最终`Dense`层的 1D 向量进行`Reshape`操作，将其重塑为图像格式（*H* × *W* × *C*）：

```
def decoder(x, layers, input_shape):
    ''' Construct the Decoder
        x     : input to the decoder (encoding)
        layers: nodes per layer
   input_shape: input shape for reconstruction
    '''
    for _ in range(len(layers)-1, 0, -1):                            ❶
        n_nodes = layers[_]['n_nodes']
        x = Dense(n_nodes)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        units = input_shape[0] * input_shape[1] * input_shape[2]     ❷
        x = Dense(units, activation='sigmoid')(x)

        outputs = Reshape(input_shape)(x)                            ❸

        return outputs                                               ❹
```

❶ 逐步单元反池化（维度扩展）

❷ 最后一次反池化

❸ 重塑回图像输入形状

❹ 解码后的图像

### 9.1.4 训练

自动编码器想要学习一个低维度的表示（我们称之为*潜在空间*），然后学习一个根据预定义任务重构图像的变换；在这种情况下，恒等函数。

以下代码示例将训练前面的自动编码器，以学习 MNIST 数据集的恒等函数。该示例创建了一个具有隐藏单元 256、128、64（潜在空间）、128、256 和 784（用于像素重构）的自动编码器。

通常，一个深度神经网络自动编码器在编码器和解码器组件中都会包含三个或有时四个层。由于 DNNs 的有效性有限，增加更多的容量通常不会提高学习恒等函数的效果。

对于 DNN 自编码器，你在这里看到的另一个约定是，编码器中的每一层将节点数量减半，相反，解码器将节点数量加倍，除了最后一层。最后一层重建图像，因此节点数量与输入向量的像素数量相同；在这种情况下，784。在示例中选择从 256 个节点开始是有些任意的；除了从一个大尺寸开始会增加容量外，它对提高学习恒等函数的能力帮助很小，或者根本不起作用。

对于数据集，我们将图像形状从（28，28）扩展到（28，28，1），因为 TF.Keras 模型期望显式指定通道数——即使只有一个通道。最后，我们使用`fit()`方法训练自编码器，并将`x_train`作为训练数据和相应的标签（恒等函数）。同样，在评估时，我们将`x_test`作为测试数据和相应的标签。图 9.2 显示了自编码器学习恒等函数。

![图片](img/CH09_F02_Ferlitsch.png)

图 9.2 自编码器学习两个函数：编码器学习将高维表示转换为低维表示，然后解码器学习将输入转换回高维表示，即输入的翻译。

以下代码演示了如图 9.2 所示的自动编码器的构建和训练，其中训练数据是 MNIST 数据集：

```
layers = [ {'n_nodes': 256 }, { 'n_nodes': 128 }, { 'n_nodes': 64 } ]      ❶

inputs = Input((28, 28, 1))                                                ❷
encoding = encoder(inputs, layers)
outputs = decoder(encoding, layers, (28, 28, 1))
ae = Model(inputs, outputs)

from tensorflow.keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test  / 255.0).astype(np.float32)
x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test, axis=-1)

ae.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
ae.fit(x_train, x_train, epochs=10, batch_size=32, validation_split=0.1, 
       verbose=1)                                                          ❸
ae.evaluate(x_test, x_test)
```

❶ 每层的过滤器数量元参数

❷ 构建自动编码器

❸ 无监督训练，其中输入和标签相同

让我们总结一下。自编码器想要学习一个低维度的表示（潜在空间），然后学习一个变换来根据预定义的任务（如恒等函数）重建图像。

使用 Idiomatic procedure reuse 设计模式为 DNN 自编码器提供的完整代码版本可在 GitHub 上找到（[`mng.bz/JvaK`](https://shortener.manning.com/JvaK)）。接下来，我们将描述如何使用卷积层代替密集层来设计和编写一个自编码器。

## 9.2 卷积自编码器

在 MNIST 或 CIFAR-10 数据集中的小图像中，DNN 自编码器运行良好。但是，当我们处理较大图像时，使用节点（即隐藏单元）进行（反）池化的自编码器在计算上很昂贵。对于较大图像，深度卷积（DC）自编码器更有效。它们不是学习（反）池化节点，而是学习（反）池化特征图。为此，它们在编码器中使用卷积，在解码器中使用*反卷积*，也称为*转置卷积*。

当步长卷积（进行特征池化）学习下采样分布的最佳方法时，步长反卷积（特征反池化）则做相反的操作，并学习上采样分布的可行方法。特征池化和反池化都在图 9.3 中展示。

让我们使用与 MNIST 的 DNN 自动编码器相同的上下文来描述这个过程。在那个例子中，编码器和解码器各有三层，编码器从 256 个特征图开始。对于 CNN 自动编码器，相应的等效结构是一个编码器，具有 256、128 和 64 个过滤器的三个卷积层，以及一个具有 128、256 和 C 个过滤器的解码器，其中 C 是输入的通道数。

![](img/CH09_F03_Ferlitsch.png)

图 9.3 对比特征池化与特征反池化

### 9.2.1 架构

深度卷积自动编码器（DC 自动编码器）的宏观架构可以分解如下：

+   *Stem*—进行粗粒度特征提取

+   *Learner*—代表性和转换性学习

+   *Task* (*重建*)—进行投影和重建

图 9.4 显示了 DC 自动编码器的宏观架构。

![](img/CH09_F04_Ferlitsch.png)

图 9.4 DC 自动编码器的宏观架构区分了表示学习和转换学习。

### 9.2.2 编码器

深度卷积自动编码器（如图 9.5 所示）中的*编码器*通过使用步长卷积逐步减少特征图的数量（通过特征减少）和特征图的大小（通过特征池化）。

![](img/CH09_F05_Ferlitsch.png)

图 9.5 CNN 编码器中输出特征图的数量和尺寸的逐步减少

如你所见，编码器逐步减少过滤器的数量，也称为*通道*，以及相应的尺寸。编码器的输出是潜在空间。

现在让我们看看一个编码器的示例代码实现。参数`layers`是一个列表，其中元素的数量是卷积层的数量，元素值是每个卷积的过滤器数量。由于我们是逐步池化，每个后续元素的值都是逐步变小的。此外，每个卷积层通过使用步长为 2 来减少特征图的大小，进一步对特征图进行池化。

在这个实现中，对于卷积，我们使用 Conv-BN-RE 约定。你可能想尝试使用 BN-RE-Conv 来查看是否能得到更好的结果。

```
def encoder(inputs, layers):
    """ Construct the Encoder
        inputs : the input vector
        layers : number of filters per layer
    """
    outputs = inputs

    for n_filters in layers:                                                ❶
        outputs = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same')
                        (outputs)
        outputs = BatchNormalization()(outputs)
        outputs = ReLU()(outputs)

    return outputs                                                          ❷
```

❶ 逐步特征池化（降维）

❷ 编码（潜在空间）

### 9.2.3 解码器

对于解码器，如图 9.6 所示。解码器通过使用步长反卷积（转置卷积）逐步增加特征图的数量（通过特征扩展）和特征图的大小（通过特征反池化）。最后一个反池化层根据重建任务将特征图投影。对于恒等函数示例，该层将特征图投影到编码器输入图像的形状。

![](img/CH09_F06_Ferlitsch.png)

图 9.6 CNN 解码器中输出特征图数量和尺寸的渐进扩展

这里是一个实现恒等函数解码器的示例。在这个例子中，输出是一个 RGB 图像；因此，在最后一个转置卷积层上有三个过滤器，每个过滤器对应一个 RGB 通道：

```
def decoder(inputs, layers):
    """ Construct the Decoder
      inputs : input to decoder
      layers : the number of filters per layer (in encoder)
    """
    outputs = inputs
    for _ in range(len(layers)-1, 0, -1):            ❶
        n_filters = layers[_]
        outputs = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), 
                                  padding='same')(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = ReLU()(outputs)

    outputs = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same')
                             (outputs)               ❷
    outputs = BatchNormalization()(outputs)
    outputs = Activation('sigmoid')(outputs)

    return outputs                                   ❸
```

❶ 渐进特征反池化（维度扩展）

❷ 最后的反池化和恢复到图像输入形状

❸ 解码后的图像

现在让我们将编码器与解码器组装起来。

在这个例子中，卷积层将逐步从 64 个、32 个到 16 个过滤器进行特征池化，而反卷积层将逐步从 32 个、64 个到 3 个过滤器进行特征反池化，以重建图像。对于 CIFAR，图像大小非常小（32 × 32 × 3），因此如果我们添加更多层，潜在空间将太小，无法进行重建；如果我们通过更多过滤器加宽层，我们可能会因为额外的参数容量而面临过拟合（欠拟合）的风险。

```
layers = [64, 32, 16]                   ❶

inputs = Input(shape=(32, 32, 3))

encoding = encoder(inputs, layers)      ❷
                                        ❷
outputs = decoder(encoding, layers)     ❷
                                        ❷
model = Model(inputs, outputs)          ❷
```

❶ 编码器每层的过滤器数量元参数

❷ 构建自编码器

使用 Idiomatic 程序重用设计模式为 CNN 自编码器编写的一个完整代码示例在 GitHub 上（[`mng.bz/JvaK`](https://shortener.manning.com/JvaK)）。

## 9.3 稀疏自编码器

潜在空间的大小是一个权衡。如果我们做得太大，模型可能会过度拟合训练数据的表示空间，而无法泛化。如果我们做得太小，它可能会欠拟合，以至于我们无法执行指定的任务（例如，恒等函数）的转换和重建。

我们希望在这两者之间找到一个“甜蜜点”。为了增加自编码器不过度拟合或欠拟合的可能性，一种方法是添加一个*稀疏性约束*。稀疏性约束的概念是限制瓶颈层输出潜在空间的神经元激活。这既是一个压缩函数，也是一个正则化器，有助于自编码器泛化潜在空间表示。

稀疏性约束通常描述为仅激活具有大激活值的单元，并使其余单元输出为零。换句话说，接近零的激活被设置为零（稀疏性）。

从数学上讲，我们可以这样表述：我们希望任何单元（σ[i]）的激活被限制在平均激活值（σ[µ]）的附近：

σ[i] `≈` σ[µ]

为了实现这一点，我们添加了一个惩罚项，该惩罚项惩罚当激活 σ[i] 显著偏离 σ[µ] 时。

在 TF.Keras 中，我们通过在编码器的最后一层添加 `activity_regularizer` 参数来添加稀疏性约束。该值指定了激活值在 +/– 零附近的阈值，将其更改为零。一个典型的值是 1e-4。

下面是使用稀疏性约束实现的 DC-自编码器的实现。参数 `layers` 是一个列表，表示逐步池化特征图的数量。我们首先从列表的末尾弹出，这是编码器的最后一层。然后我们继续构建剩余的层。然后我们使用弹出（最后一层）的特征图数量来构建最后一层，其中我们添加稀疏性约束。这个最后的卷积层是潜在空间：

```
from tensorflow.keras.regulaziers import l1

def encoder(inputs, layers):
    """ Construct the Encoder
        inputs : the input vector
        layers : number of filters per layer
    """
    outputs = inputs

    last_filters = layers.pop()                                             ❶

    for n_filters in layers:                                                ❷
        outputs = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same')
                        (outputs)
        outputs = BatchNormalization()(outputs)
        outputs = ReLU()(outputs)

    outputs = Conv2D(last_filters, (3, 3), strides=(2, 2), padding='same',  ❸
               activity_regularizer=l1(1e-4))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = ReLU()(outputs)

    return outputs
```

❶ 保留最后一层

❷ 特征池化

❸ 在编码器的最后一层添加稀疏性约束

## 9.4 去噪自编码器

使用自编码器的另一种方式是将其训练为图像去噪器。我们输入一个噪声图像，然后输出图像的去噪版本。将这个过程视为学习带有一些噪声的恒等函数。如果我们用方程表示这个过程，假设 *x* 是图像，*e* 是噪声。该函数学习返回 *x*：

*f*(*x* + *e*) = *x*

我们不需要为此目的更改自编码器架构；相反，我们更改我们的训练数据。更改训练数据需要三个基本步骤：

1.  构建一个随机生成器，它将输出一个具有你想要添加到训练（和测试）图像中的噪声值范围的随机分布。

1.  在训练时，向训练数据中添加噪声。

1.  对于标签，使用原始图像。

下面是训练用于去噪的自编码器的代码。我们将噪声设置为在以 0.5 为中心的正态分布内，标准差为 0.5。然后我们将随机噪声分布添加到训练数据的副本（`x_train_noisy`）中。我们使用 `fit()` 方法来训练去噪器，其中噪声训练数据是训练数据，原始（去噪）训练数据是对应的标签：

```
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)         ❶
x_train_noisy = x_train + noise                                          ❷

model.fit(x_train_noisy, x_train, epochs=epochs, batch_size=batch_size, 
          verbose=1)                                                     ❸
```

❶ 生成噪声为以 0.5 为中心，标准差为 0.5 的正态分布

❷ 将噪声添加到图像训练数据的副本中

❸ 通过将噪声图像作为训练数据，原始图像作为标签来训练编码器

## 9.5 超分辨率

自编码器也被用来开发用于 *超分辨率* (*SR*) 的模型。这个过程将低分辨率（LR）图像上采样以提高细节，以获得高分辨率（HR）图像。与压缩中学习恒等函数或去噪中学习噪声恒等函数不同，我们想要学习低分辨率图像和高分辨率图像之间的表示映射。让我们用一个函数来表示我们想要学习的这个映射：

*f*(*x*[lr]) = *x*[hr]

在这个方程中，*f()*代表模型正在学习的变换函数。术语*x*[lr]代表函数输入的低分辨率图像，而术语*x*[hr]是函数从高分辨率预测输出的变换。

尽管现在非常先进的模型可以进行超分辨率处理，但早期版本（约 2015 年）使用自动编码器的变体来学习从低分辨率表示到高分辨率表示的映射。一个例子是 Chao Dong 等人提出的超分辨率卷积神经网络（SRCNN）模型，该模型在“使用深度卷积网络进行图像超分辨率”一文中被介绍（[`arxiv.org/pdf/1501.00092.pdf`](https://arxiv.org/pdf/1501.00092.pdf)）。在这种方法中，模型学习在多维空间中对低分辨率图像的表示（潜在空间）。然后它学习从低分辨率图像的高维空间到高分辨率图像的映射，以重建高分辨率图像。注意，这与典型的自动编码器相反，自动编码器在低维空间中学习表示。

### 9.5.1 预上采样 SR

SRCNN 模型的创造者引入了全卷积神经网络在图像超分辨率中的应用。这种方法被称为*预上采样 SR 方法*，如图 9.7 所示。我们可以将模型分解为四个组件：低分辨率特征提取、高维表示、编码到低维表示，以及用于重建的卷积层。

![](img/CH09_F07_Ferlitsch.png)

图 9.7 预上采样超分辨率模型学习从低分辨率图像重建高分辨率图像。

让我们深入了解。与自动编码器不同，在低分辨率特征提取组件中没有特征池化（或下采样）。相反，特征图的大小与低输入图像中的通道大小相同。例如，如果输入形状是（16，16，3），则特征图的*H* × *W*将保持 16 × 16。

在主干卷积中，特征图的数量从输入的通道数（3）显著增加到，这为我们提供了低分辨率图像的高维表示。然后编码器将高维表示降低到低维表示。最后的卷积将图像重建为高分辨率图像。

通常，您会通过使用现有的图像数据集来训练这种方法，该数据集成为 HR 图像。然后您复制训练数据，其中每个图像都已被调整大小为更小，然后调整回原始大小。为了进行这两次调整大小，您使用静态算法，如双三次插值。LR 图像将与 HR 图像具有相同的大小，但由于调整大小操作期间所做的近似，LR 图像的质量将低于原始图像。

究竟什么是插值，更具体地说，*双三次插值*？可以这样想：如果我们有 4 个像素，用 2 个像素替换它们，或者反过来，你需要一种数学方法来对替换表示进行良好的估计——这就是插值。*三次插值*是用于向量的特定方法（1D），而 *双三次* 是用于矩阵（2D）的变体。对于图像缩小，双三次插值通常比其他插值算法给出更好的估计。

这里有一个代码示例，用于展示使用 CIFAR-10 数据集进行此训练数据准备的过程。在这个例子中，NumPy 数组 `x_train` 包含了训练数据图像。然后我们通过依次将 `x_train` 中的每个图像调整大小到一半的 *H* × *W*（16, 16），然后将图像调整回原始的 *H* × *W*（32, 32），并在 `x_train_lr` 中放置相同的索引位置，来创建一个低分辨率配对列表 `x_train_lr`。最后，我们对两组图像中的像素数据进行归一化：

```
from tensorflow.keras.datasets import cifar10
import numpy as np
import cv2

(x_train, y_train), (x_test, y_test) = cifar10.load_data()               ❶

x_train_lr = []                                                          ❷
for image in x_train:                                                    ❷
    image = cv2.resize(image, (16, 16), interpolation=cv2.INTER_CUBIC)   ❷
    x_train_lr.append(cv2.resize(image, (32, 32),                        ❷
                      interpolation=cv2.INTER_CUBIC))                    ❷
x_train_lr = np.asarray(x_train_lr)                                      ❷

x_train = (x_train / 255.0).astype(np.float32)                           ❸
x_train_lr = (x_train_lr / 255.0).astype(np.float32)                     ❸
```

❶ 将 CIFAR-10 数据集下载到内存中作为高分辨率图像

❷ 创建训练图像的低分辨率配对

❸ 对训练中的像素数据进行归一化

现在，让我们看看用于在小型图像（如 CIFAR-10）上实现高分辨率重建质量的预上采样 SR 模型的代码。为了训练它，我们将原始 CIFAR-10 32 × 32 图像（`x_train`）视为高分辨率图像，将镜像配对图像（`x_train_lr`）视为低分辨率图像。对于训练，低分辨率图像是输入，配对的 HR 图像是相应的标签。

这个例子在 CIFAR-10 上仅用 20 个周期就得到了相当好的重建结果，重建准确率为 88%。如代码所示，`stem()` 组件使用粗略的 9 × 9 滤波器进行低分辨率特征提取，并为高维表示输出 64 个特征图。`encoder()` 由一个卷积组成，使用 1 × 1 瓶颈卷积将低分辨率表示从高维度降低到低维度，并将特征图的数量减少到 32。最后，使用粗略的 5 × 5 滤波器学习从低分辨率表示到高分辨率的映射以进行重建：

```
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.ketas.layers import ReLU, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam

def stem(inputs):                                      ❶

    x = Conv2D(64, (9, 9), padding='same')(inputs)     ❷
    x = BatchNormalization()(x)                        ❷
    x = ReLU()(x)                                      ❷
    return x

def encoder(x):
    x = Conv2D(32, (1, 1), padding='same')(x)          ❸
    x = BatchNormalization()(x)                        ❸
    x = ReLU()(x)                                      ❸

    x = Conv2D(3, (5, 5), padding='same')(x)           ❹
    x = BatchNormalization()(x)                        ❹
    outputs = Activation('sigmoid')(x)                 ❹
    return outputs

inputs = Input((32, 32, 3))
x = stem(inputs)
outputs = encoder(x)

model = Model(inputs, outputs)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), 
              metrics=['accuracy'])

model.fit(x_train_lr, x_train, epochs=25, batch_size=32, verbose=1, 
          validation_split=0.1)
```

❶ 低分辨率特征提取

❷ 高维表示

❸ 作为编码器的 1 × 1 瓶颈卷积

❹ 用于将重建为高分辨率图像的 5 × 5 卷积

现在让我们看看一些实际的图像。图 9.8 展示了 CIFAR-10 训练数据集中同一只孔雀的一组图像。前两个图像是用于训练的低分辨率和高分辨率图像对，第三个是模型训练后对同一孔雀图像的超分辨率重建。请注意，低分辨率图像比高分辨率图像有更多的伪影——即边缘周围的区域是方形的，颜色过渡不平滑。重建的超分辨率图像在边缘周围的色彩过渡更平滑，类似于高分辨率图像。

![](img/CH09_F08_Ferlitsch.png)

图 9.8 预上采样超分辨率中 LR、HR 配对和重建 SR 图像的比较

### 9.5.2 后上采样超分辨率

另一个 SRCNN 风格模型的例子是后上采样超分辨率模型，如图 9.9 所示。我们可以将这个模型分解为三个部分：低分辨率特征提取、高维表示和重建的解码器。

![](img/CH09_F09_Ferlitsch.png)

图 9.9 后上采样超分辨率模型

让我们更深入地探讨。与自动编码器不同，在低分辨率特征提取组件中没有特征池化（或下采样）。相反，特征图的大小与低输入图像中的通道大小相同。例如，如果输入形状是 (16, 16, 3)，特征图的 *H* × *W* 将保持 16 × 16。

在卷积过程中，我们逐步增加特征图的数量——这就是我们得到高维空间的原因。例如，我们可能从三通道输入到 16，然后到 32，再到 64 个特征图。所以你可能想知道为什么维度更高？我们希望丰富的不同低分辨率特征提取表示有助于我们学习从它们到高分辨率的映射，这样我们就可以使用反卷积进行重建。但是，如果我们有太多的特征图，我们可能会使模型暴露在训练数据中的映射记忆中。

通常，我们使用现有的图像数据集来训练超分辨率模型，这些数据集将成为高分辨率图像，然后复制训练数据，其中每个图像都被调整大小以生成低分辨率图像对。

以下代码示例展示了使用 CIFAR-10 数据集进行此训练数据准备的过程。在这个例子中，NumPy 数组 `x_train` 包含训练数据图像。然后我们通过逐个调整 `x_train` 中每个图像的大小，并将其放置在 `x_train_lr` 中的相同索引位置，创建了一个低分辨率图像对列表 `x_train_lr`。最后，我们对两组图像中的像素数据进行归一化。

在后上采样的情况下，低分辨率图像保持为 16 × 16，而不是像预上采样那样调整回 32 × 32，这是因为在调整回 32 × 32 时，通过静态插值丢失了像素信息。

```
from tensorflow.keras.datasets import cifar10
import numpy as np
import cv2

(x_train, y_train), (x_test, y_test) = cifar10.load_data()     ❶

x_train_lr = []                                                ❷
for image in x_train:                                          ❷
    x_train_lr.append(cv2.resize(image, (16, 16),              ❷
                       interpolation=cv2.INTER_CUBIC))         ❷
x_train_lr = np.asarray(x_train_lr)                            ❷

x_train = (x_train / 255.0).astype(np.float32)                 ❸
x_train_lr = (x_train_lr / 255.0).astype(np.float32)           ❸
```

❶ 将 CIFAR-10 数据集作为高分辨率图像下载到内存中

❷ 对训练图像进行低分辨率配对

❸ 对训练的像素数据进行归一化

下面的代码实现了一个后上采样 SR 模型，它在 CIFAR-10 等小图像上获得了良好的 HR 重建质量。我们专门为 CIFAR-10 编写了这个实现。为了训练它，我们将原始 CIFAR-10 32 × 32 图像 (`x_train`) 作为 HR 图像，将镜像配对图像 (`x_train_lr`) 作为 LR 图像。对于训练，LR 图像是输入，配对的 HR 图像是相应的标签。

这个示例在 CIFAR-10 上仅用 20 个 epoch 就获得了相当好的重建结果，重建准确率达到 90%。在这个示例中，`stem()` 和 `learner()` 组件执行低分辨率特征提取，并逐步扩展特征图维度从 16、32 到 64 个特征图。64 个特征图的最后一个卷积的输出是高维表示。`decoder()` 由一个反卷积组成，用于学习从低分辨率表示到高分辨率的映射以进行重建：

```
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam

def stem(inputs):                                     ❶
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def learner(x):                                       ❶
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)         ❷
    x = BatchNormalization()(x)                       ❷
    x = ReLU()(x)                                     ❷
    return x

def decoder(x):                                       ❸
    x = Conv2DTranspose(3, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    return x

inputs = Input((16, 16, 3))
x = stem(inputs)
x = learner(x)
outputs = decoder(x)

model = Model(inputs, outputs)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), 
              metrics=['accuracy'])
model.fit(x_train_lr, x_train, epochs=25, batch_size=32, verbose=1, 
          validation_split=0.1)
```

❶ 低分辨率特征提取

❷ 高维表示

❸ 低到高分辨率重建

让我们回到之前看过的那些孔雀图像。在图 9.10 中，前两个图像是用于训练的低分辨率和高分辨率配对，第三个是模型训练后对同一孔雀图像的超分辨率重建。与之前的预上采样 SR 模型一样，后上采样 SR 模型产生的重建 SR 图像比低分辨率图像的伪影更少。

![图片](img/CH09_F10_Ferlitsch.png)

图 9.10 LR、HR 配对和后上采样 SR 重建图像的比较

在 GitHub 上提供了使用 Idiomatic 程序重用设计模式对 SRCNN 进行完整代码实现的示例 ([`mng.bz/w0a2`](https://shortener.manning.com/w0a2)).

## 9.6 预训练任务

正如我们讨论的，自动编码器可以在没有标签的情况下进行训练，以学习关键特征的特征提取，这些特征我们可以重新用于迄今为止给出的示例之外：压缩和去噪。

我们所说的“关键特征”是什么意思？对于成像，我们希望我们的模型学习数据的本质特征，而不是数据本身。这使得模型不仅能够泛化到同一分布中的未见数据，而且还能在模型部署后，当输入分布发生偏移时，更好地预测其正确性。

例如，假设我们有一个训练好的模型用于识别飞机，训练时使用的图像包括各种场景，如停机坪、滑向航站楼和在空中，但没有一个是停机库中的。如果在部署模型后，它现在看到了停机库中的飞机，那么输入分布发生了变化；这被称为*数据漂移*。而当飞机图像出现在停机库中时，我们得到的准确度会降低。

在这个示例案例中，我们可能会尝试通过重新训练模型并添加包含背景中飞机的额外图像来改进模型。很好，现在部署时它工作了。但假设新模型看到了它没有训练过的其他背景中的飞机，比如在水面上的飞机（水上飞机）、在飞机坟场上的沙地上的飞机、在工厂中部分组装的飞机。好吧，在现实世界中，总有你预料不到的事情！

正因如此，学习数据集中的基本特征而不是数据本身非常重要。对于自动编码器来说，它们必须学习像素之间的相关性——即表示学习。相关性越强，关系越有可能在潜在空间表示中显现出来，相关性越弱，则不太可能显现。

我们不会在这里详细讨论使用前缀任务进行预训练，但我们将简要地在此处提及它，特别是在自动编码器的上下文中。就我们的目的而言，我们希望使用自动编码器方法来训练主干卷积组，以便在数据集上训练模型之前学习提取基本粗略级特征。以下是步骤：

1.  在目标模型上进行预热（监督学习）训练，以实现数值稳定（将在第十四章中进一步讨论）。

1.  构建一个自动编码器，其中模型的主干组作为编码器，反转的主干组作为解码器。

1.  将目标模型中的数值稳定权重转移到自动编码器的编码器中。

1.  在前缀任务（例如，压缩、去噪）上训练（无监督学习）自动编码器。

1.  将前缀任务训练的权重从自动编码器的编码器转移到目标模型。

1.  训练（监督学习）目标模型。

图 9.11 描述了这些步骤。

![图片](img/CH09_F11_Ferlitsch.png)

图 9.11 使用自动编码器预训练主干组，以改善在模型完全使用标记数据训练后对未见数据的泛化。

让我们再讨论一下这种前缀任务的一部分。你可能已经想到，来自主干卷积组的输出将大于输入。当我们对通道进行静态或特征池化时，我们增加了总通道数。例如，我们可能使用池化将通道大小减少到 25%甚至仅为 6%，但我们将通道数从三个（RGB）增加到 64 个左右。

因此，潜在空间现在比输入更大，更容易过拟合。为此特定目的，我们构建了一个稀疏自动编码器来抵消过拟合的潜在可能性。

以下是一个示例实现。虽然我们尚未讨论`UpSampling2D`层，但它是对步长`MaxPooling2D`的逆操作。它不是使用静态算法将高度和宽度减半，而是使用静态算法将高度和宽度增加 2：

```
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.regularizers import l1

def stem(inputs):
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
               activity_regularizer=l1(1e-4))(inputs)                   ❶
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)                         ❷
    return x

def inverted_stem(inputs):
    x = UpSampling2D((2, 2))(inputs)                                    ❸
    x = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(x)   ❹
    return x

inputs = Input((128, 128, 3))
_encoder = stem(inputs)
_decoder = inverted_stem(_encoder)
model = Model(inputs, _decoder)
```

❶ 使用 5 × 5 滤波器进行粗略特征提取并使用特征池化

❷ 使用最大池化将特征图减少到图像大小的 6%

❸ 反转最大池化

❹ 反转特征池化并重建图像

以下是从该自动编码器的`summary()`方法输出的内容。请注意，输入大小等于输出大小：

```
Layer (type)                 Output Shape              Param #   
=================================================================
input_4 (InputLayer)         [(None, 128, 128, 3)]     0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 64)        4864      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 64)        0         
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 128, 128, 3)       4803      
=================================================================
Total params: 9,667
Trainable params: 9,667
Non-trainable params: 0
```

## 9.7 超越计算机视觉：序列到序列

让我们简要地看看一种基本的自然语言处理模型架构，称为*序列到序列*（*Seq2Seq*）。此类模型结合了自然语言理解（NLU）——理解文本，和自然语言生成（NLG）——生成新文本。对于 NLG，Seq2Seq 模型可以执行诸如语言翻译、摘要和问答等操作。例如，聊天机器人是执行问答的 Seq2Seq 模型。

在第五章的结尾，我们介绍了 NLU 模型架构，并看到了组件设计如何与计算机视觉相媲美。我们还研究了注意力机制，它与残差网络中的身份链接相当。我们没有涵盖的是在 2017 年引入的 Transformer 模型架构，它引入了注意力机制。这一创新将 NLU 从基于时间序列的解决方案，使用 RNN，转变为空间问题。在 RNN 中，模型一次只能查看文本输入的片段并保持顺序。此外，对于每个片段，模型必须保留重要特征的记忆。这增加了模型设计的复杂性，因为您需要在图中实现循环以保留先前看到的特征。有了 Transformer 和注意力机制，模型可以一次性查看文本。

图 9.12 展示了 Transformer 模型架构，该架构实现了一个 Seq2Seq 模型。

![](img/CH09_F12_Ferlitsch.png)

图 9.12 Transformer 架构包括用于 NLU 的编码器和用于 NLG 的解码器

如您所见，学习组件包括用于 NLU 的编码器和用于 NLG 的解码器。您通过使用文本对、句子、段落等来训练模型。例如，如果您正在训练一个问答聊天机器人，输入将是问题，标签是答案。对于摘要，输入将是文本，标签是摘要。

在转换器模型中，编码器按顺序学习输入上下文的降维，这与计算机视觉自动编码器中编码器的表征学习相当。编码器的输出被称为*中间表示*，与计算机视觉自动编码器中的潜在空间相当。

解码器按顺序学习将中间表示扩展到变换上下文的维度扩展，这与计算机视觉自动编码器中解码器的变换学习相当。

解码器的输出传递给任务组件，该组件学习文本生成。文本生成任务与计算机视觉自动编码器中的重建任务相当。

## 摘要

+   自动编码器学习输入到低维表示的最佳映射，然后学习映射回高维表示，以便可以进行图像的变换重建。

+   自动编码器可以学习的变换函数示例包括恒等函数（压缩）、去噪图像和构建图像的高分辨率版本。

+   在卷积神经网络自动编码器中，池化操作通过步长卷积完成，而反池化操作通过步长反卷积完成。

+   在无监督学习中使用自动编码器可以训练模型学习数据集分布的基本特征，而无需标签。

+   使用编码器作为无监督学习预训练任务的前缀可以辅助后续的监督学习，以学习更好的泛化所需的基本特征。

+   NLU 的 Seq2Seq 模型模式使用一个编码器和解码器，与自动编码器相当。
