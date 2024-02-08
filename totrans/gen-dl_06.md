# 第三章. 变分自动编码器

2013 年，Diederik P. Kingma 和 Max Welling 发表了一篇论文，奠定了一种称为*变分自动编码器*（VAE）的神经网络类型的基础。^(1) 这现在是最基本和最知名的深度学习架构之一，用于生成建模，也是我们进入生成式深度学习旅程的绝佳起点。

在本章中，我们将首先构建一个标准的自动编码器，然后看看如何扩展这个框架以开发一个变分自动编码器。在这个过程中，我们将分析这两种模型，以了解它们在细粒度级别上的工作原理。通过本章的结束，您应该完全了解如何构建和操作基于自动编码器的模型，特别是如何从头开始构建一个变分自动编码器，以根据您自己的数据集生成图像。

# 介绍

让我们从一个简单的故事开始，这将有助于解释自动编码器试图解决的基本问题。

现在让我们探讨这个故事如何与构建自动编码器相关。

# 自动编码器

故事描述的过程的图表显示在图 3-2 中。您扮演*编码器*的角色，将每件服装移动到衣柜中的一个位置。这个过程称为*编码*。Brian 扮演*解码器*的角色，接受衣柜中的一个位置，并尝试重新创建该项目。这个过程称为*解码*。

![](img/gdl2_0302.png)

###### 图 3-2. 无限衣柜中的服装项目-每个黑点代表一个服装项目

衣柜中的每个位置由两个数字表示（即一个 2D 向量）。例如，图 3-2 中的裤子被编码为点[6.3，-0.9]。这个向量也被称为*嵌入*，因为编码器试图将尽可能多的信息嵌入其中，以便解码器可以产生准确的重建。

*自动编码器*只是一个经过训练的神经网络，用于执行编码和解码项目的任务，使得这个过程的输出尽可能接近原始项目。关键是，它可以用作生成模型，因为我们可以解码 2D 空间中的任何点（特别是那些不是原始项目的嵌入）以生成新的服装项目。

现在让我们看看如何使用 Keras 构建自动编码器并将其应用于真实数据集！

# 运行此示例的代码

此示例的代码可以在位于书籍存储库中的 Jupyter 笔记本*notebooks/03_vae/01_autoencoder/autoencoder.ipynb*中找到。

## Fashion-MNIST 数据集

在这个例子中，我们将使用[Fashion-MNIST 数据集](https://oreil.ly/DS4-4)-一个由 28×28 像素大小的服装项目的灰度图像组成的集合。数据集中的一些示例图像显示在图 3-3 中。

![](img/gdl2_0303.png)

###### 图 3-3. Fashion-MNIST 数据集中的图像示例

数据集已经预先打包到 TensorFlow 中，因此可以按照示例 3-1 中所示进行下载。

##### 示例 3-1. 加载 Fashion-MNIST 数据集

```py
from tensorflow.keras import datasets
(x_train,y_train), (x_test,y_test) = datasets.fashion_mnist.load_data()
```

这些是 28×28 的灰度图像（像素值在 0 到 255 之间），我们需要对其进行预处理，以确保像素值在 0 到 1 之间。我们还将每个图像填充到 32×32，以便更容易地处理通过网络的张量形状，如示例 3-2 中所示。

##### 示例 3-2. 数据预处理

```py
def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)
```

接下来，我们需要了解自动编码器的整体结构，以便我们可以使用 TensorFlow 和 Keras 对其进行编码。

## 自动编码器架构

*自动编码器*是由两部分组成的神经网络：

+   一个*编码器*网络，将高维输入数据（如图像）压缩成较低维度的嵌入向量

+   一个*解码器*网络，将给定的嵌入向量解压缩回原始域（例如，回到图像）

网络架构图显示在图 3-4 中。输入图像被编码为潜在嵌入向量<math alttext="z"><mi>z</mi></math>，然后解码回原始像素空间。

![](img/gdl2_0304.png)

###### 图 3-4。自动编码器架构图

自动编码器经过编码器和解码器后被训练重建图像。这一开始可能看起来很奇怪——为什么要重建一组已经可用的图像？然而，正如我们将看到的，自动编码器中有趣的部分是嵌入空间（也称为*潜在空间*），因为从这个空间中取样将允许我们生成新的图像。

首先定义一下我们所说的嵌入。嵌入（<math alttext="z"><mi>z</mi></math>）是原始图像压缩到较低维度潜在空间中。这个想法是通过选择潜在空间中的任意点，我们可以通过解码器生成新颖的图像，因为解码器已经学会如何将潜在空间中的点转换为可行的图像。

在我们的示例中，我们将图像嵌入到一个二维潜在空间中。这将帮助我们可视化潜在空间，因为我们可以轻松在 2D 中绘制点。实际上，自动编码器的潜在空间通常会有超过两个维度，以便更自由地捕获图像中更多的细微差别。

# 自动编码器作为去噪模型

自动编码器可以用于清理嘈杂的图像，因为编码器学习到捕获潜在空间中随机噪声的位置并不有用，以便重建原始图像。对于这样的任务，一个二维潜在空间可能太小，无法从输入中编码足够的相关信息。然而，正如我们将看到的，如果我们想将自动编码器用作生成模型，增加潜在空间的维度会很快导致问题。

现在让我们看看如何构建编码器和解码器。

## 编码器

在自动编码器中，编码器的任务是将输入图像映射到潜在空间中的嵌入向量。我们将构建的编码器的架构显示在表 3-1 中。

表 3-1。编码器的模型摘要

| 层（类型） | 输出形状 | 参数 # |
| --- | --- | --- |
| InputLayer | （None，32，32，1） | 0 |
| Conv2D | （None，16，16，32） | 320 |
| Conv2D | （None，8，8，64） | 18,496 |
| Conv2D | （None，4，4，128） | 73,856 |
| Flatten | （None，2048） | 0 |
| Dense | （None，2） | 4,098 |
| 总参数 | 96,770 |
| 可训练参数 | 96,770 |
| 不可训练参数 | 0 |

为了实现这一点，我们首先为图像创建一个“输入”层，并依次通过三个`Conv2D`层，每个层捕获越来越高级的特征。我们使用步幅为 2，以减半每个层的输出大小，同时增加通道数。最后一个卷积层被展平，并连接到大小为 2 的`Dense`层，代表我们的二维潜在空间。

示例 3-3 展示了如何在 Keras 中构建这个模型。

##### 示例 3-3。编码器

```py
encoder_input = layers.Input(
    shape=(32, 32, 1), name = "encoder_input"
) ![1](img/1.png)
x = layers.Conv2D(32, (3, 3), strides = 2, activation = 'relu', padding="same")(
    encoder_input
) ![2](img/2.png)
x = layers.Conv2D(64, (3, 3), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides = 2, activation = 'relu', padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x) ![3](img/3.png)
encoder_output = layers.Dense(2, name="encoder_output")(x) ![4](img/4.png)

encoder = models.Model(encoder_input, encoder_output) ![5](img/5.png)
```

![1](img/#co_variational_autoencoders_CO1-1)

定义编码器（图像）的“输入”层。

![2](img/#co_variational_autoencoders_CO1-2)

顺序堆叠`Conv2D`层。

![3](img/#co_variational_autoencoders_CO1-3)

将最后一个卷积层展平为一个向量。

![4](img/#co_variational_autoencoders_CO1-4)

将这个向量连接到 2D 嵌入中的`Dense`层。

![5](img/#co_variational_autoencoders_CO1-5)

定义编码器的 Keras`Model`——一个将输入图像并将其编码为 2D 嵌入的模型。

###### 提示

我强烈建议您尝试不同数量的卷积层和滤波器，以了解架构如何影响模型参数的总数、模型性能和模型运行时间。

## 解码器

解码器是编码器的镜像——我们使用*卷积转置*层，而不是卷积层，如表 3-2 所示。

表 3-2. 解码器的模型摘要

| 层（类型） | 输出形状 | 参数 # |
| --- | --- | --- |
| InputLayer | (None, 2) | 0 |
| Dense | (None, 2048) | 6,144 |
| 重塑 | (None, 4, 4, 128) | 0 |
| Conv2DTranspose | (None, 8, 8, 128) | 147,584 |
| Conv2DTranspose | (None, 16, 16, 64) | 73,792 |
| Conv2DTranspose | (None, 32, 32, 32) | 18,464 |
| Conv2D | (None, 32, 32, 1) | 289 |
| 总参数 | 246,273 |
| 可训练参数 | 246,273 |
| 不可训练参数 | 0 |

示例 3-4 展示了我们如何在 Keras 中构建解码器。

##### 示例 3-4. 解码器

```py
decoder_input = layers.Input(shape=(2,), name="decoder_input") ![1](img/1.png)
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input) ![2](img/2.png)
x = layers.Reshape(shape_before_flattening)(x) ![3](img/3.png)
x = layers.Conv2DTranspose(
    128, (3, 3), strides=2, activation = 'relu', padding="same"
)(x) ![4](img/4.png)
x = layers.Conv2DTranspose(
    64, (3, 3), strides=2, activation = 'relu', padding="same"
)(x)
x = layers.Conv2DTranspose(
    32, (3, 3), strides=2, activation = 'relu', padding="same"
)(x)
decoder_output = layers.Conv2D(
    1,
    (3, 3),
    strides = 1,
    activation="sigmoid",
    padding="same",
    name="decoder_output"
)(x)

decoder = models.Model(decoder_input, decoder_output) ![5](img/5.png)
```

![1](img/#co_variational_autoencoders_CO2-1)

定义解码器（嵌入）的`Input`层。

![2](img/#co_variational_autoencoders_CO2-2)

将输入连接到`Dense`层。

![3](img/#co_variational_autoencoders_CO2-3)

将这个向量重塑成一个张量，可以作为输入传递到第一个`Conv2DTranspose`层。

![4](img/#co_variational_autoencoders_CO2-4)

将`Conv2DTranspose`层堆叠在一起。

![5](img/#co_variational_autoencoders_CO2-5)

定义解码器的 Keras `Model`——一个模型，将潜在空间中的嵌入解码为原始图像域。

## 将编码器连接到解码器

为了同时训练编码器和解码器，我们需要定义一个模型，表示图像通过编码器流动并通过解码器返回。幸运的是，Keras 使这变得非常容易，如示例 3-5 所示。请注意我们如何指定自编码器的输出只是经过解码器传递后的编码器的输出的方式。

##### 示例 3-5. 完整自编码器

```py
autoencoder = Model(encoder_input, decoder(encoder_output)) ![1](img/1.png)
```

![1](img/#co_variational_autoencoders_CO3-1)

定义完整自编码器的 Keras `Model`——一个模型，将图像通过编码器传递并通过解码器返回，生成原始图像的重建。

现在我们已经定义了我们的模型，我们只需要使用损失函数和优化器对其进行编译，如示例 3-6 所示。损失函数通常选择为原始图像的每个像素之间的均方根误差（RMSE）或二进制交叉熵。

##### 示例 3-6. 编译自编码器

```py
# Compile the autoencoder
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
```

现在我们可以通过将输入图像作为输入和输出来训练自编码器，如示例 3-7 所示。

##### 示例 3-7. 训练自编码器

```py
autoencoder.fit(
    x_train,
    x_train,
    epochs=5,
    batch_size=100,
    shuffle=True,
    validation_data=(x_test, x_test),
)
```

现在我们的自编码器已经训练好了，我们需要检查的第一件事是它是否能够准确重建输入图像。

## 重建图像

我们可以通过将测试集中的图像通过自编码器并将输出与原始图像进行比较来测试重建图像的能力。这个代码在示例 3-8 中展示。

##### 示例 3-8. 使用自编码器重建图像

```py
example_images = x_test[:5000]
predictions = autoencoder.predict(example_images)
```

在图 3-6 中，您可以看到一些原始图像的示例（顶部行），编码后的 2D 向量，以及解码后的重建物品（底部行）。

![](img/gdl2_0306.png)

###### 图 3-6. 服装项目的编码和解码示例

注意重建并不完美——解码过程中仍然有一些原始图像的细节没有被捕捉到，比如标志。这是因为将每个图像减少到只有两个数字，自然会丢失一些信息。

现在让我们来研究编码器如何在潜在空间中表示图像。

## 可视化潜在空间

我们可以通过将测试集通过编码器并绘制结果嵌入来可视化图像如何嵌入到潜在空间中，如示例 3-9 所示。

##### 示例 3-9。使用编码器嵌入图像

```py
embeddings = encoder.predict(example_images)

plt.figure(figsize=(8, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=3)
plt.show()
```

结果绘图是图 3-2 中显示的散点图-每个黑点代表一个被嵌入到潜在空间中的图像。

为了更好地理解这个潜在空间的结构，我们可以利用 Fashion-MNIST 数据集中附带的标签，描述每个图像中的物品类型。总共有 10 组，如表 3-3 所示。

表 3-3。时尚 MNIST 标签

| ID | 服装标签 |
| --- | --- |
| 0 | T 恤/上衣 |
| 1 | 裤子 |
| 2 | 套头衫 |
| 3 | 连衣裙 |
| 4 | 外套 |
| 5 | 凉鞋 |
| 6 | 衬衫 |
| 7 | 运动鞋 |
| 8 | 包 |
| 9 | 短靴 |

我们可以根据相应图像的标签对每个点进行着色，以生成图 3-7 中的绘图。现在结构变得非常清晰！尽管在训练期间模型从未展示过服装标签，但自动编码器自然地将外观相似的项目分组到潜在空间的相同部分。例如，潜在空间右下角的深蓝色点云都是不同的裤子图像，中心附近的红色点云都是短靴。

![](img/gdl2_0307.png)

###### 图 3-7。潜在空间的绘图，按服装标签着色

## 生成新图像

我们可以通过在潜在空间中抽样一些点并使用解码器将其转换回像素空间来生成新图像，如示例 3-10 所示。

##### 示例 3-10。使用解码器生成新图像

```py
mins, maxs = np.min(embeddings, axis=0), np.max(embeddings, axis=0)
sample = np.random.uniform(mins, maxs, size=(18, 2))
reconstructions = decoder.predict(sample)
```

一些生成的图像示例显示在图 3-8 中，以及它们在潜在空间中的嵌入。

![](img/gdl2_0308.png)

###### 图 3-8。生成的服装项目

每个蓝点映射到图表右侧显示的图像之一，下面显示嵌入向量。注意一些生成的项目比其他项目更真实。为什么呢？

为了回答这个问题，让我们首先观察一下潜在空间中点的整体分布，参考图 3-7：

+   一些服装项目在一个非常小的区域内表示，而其他服装项目在一个更大的区域内表示。

+   分布关于点(0, 0)不对称，也不是有界的。例如，具有正 y 轴值的点比负值更多，甚至有些点甚至延伸到 y 轴值> 8。

+   颜色之间有很大的间隙，包含很少的点。

这些观察实际上使得从潜在空间中抽样变得非常具有挑战性。如果我们将解码点的图像叠加在网格上的潜在空间上，如图 3-9 所示，我们可以开始理解为什么解码器可能不总是生成令人满意的图像。

![](img/gdl2_0309.png)

###### 图 3-9。解码嵌入的网格，与数据集中原始图像的嵌入叠加，按项目类型着色

首先，我们可以看到，如果我们在我们定义的有界空间中均匀选择点，我们更有可能抽样出看起来像包（ID 8）而不是短靴（ID 9）的东西，因为为包（橙色）划定的潜在空间部分比短靴区域（红色）更大。

其次，我们应该如何选择潜在空间中的*随机*点并不明显，因为这些点的分布是未定义的。从技术上讲，我们可以选择 2D 平面上的任意点！甚至不能保证点会围绕(0, 0)中心。这使得从我们的潜在空间中抽样成为问题。

最后，我们可以看到潜在空间中存在一些空洞，原始图像没有被编码。例如，在域的边缘有大片白色空间——自动编码器没有理由确保这些点被解码为可识别的服装项目，因为训练集中很少有图像被编码到这里。

即使中心点也可能无法解码为形式良好的图像。这是因为自动编码器没有被迫确保空间是连续的。例如，即使点（-1，-1）可能被解码为一个令人满意的凉鞋图像，但没有机制来确保点（-1.1，-1.1）也产生一个令人满意的凉鞋图像。

在二维中，这个问题是微妙的；自动编码器只有少量维度可用，因此自然地必须将服装组合在一起，导致服装组之间的空间相对较小。然而，当我们开始在潜在空间中使用更多维度来生成更复杂的图像，如面孔时，这个问题变得更加明显。如果我们让自动编码器自由使用潜在空间来编码图像，那么相似点之间将会有巨大的间隙，而没有动机使之间的空间生成形式良好的图像。

为了解决这三个问题，我们需要将我们的自动编码器转换为*变分自动编码器*。

# 变分自动编码器

为了解释，让我们重新审视无限衣柜并做一些改变...​

现在让我们尝试理解如何将我们的自动编码器模型转换为变分自动编码器，从而使其成为一个更复杂的生成模型。

我们需要更改的两个部分是编码器和损失函数。

## 编码器

在自动编码器中，每个图像直接映射到潜在空间中的一个点。在变分自动编码器中，每个图像实际上被映射到潜在空间中某一点周围的多变量正态分布，如图 3-10 所示。

![](img/gdl2_0310.png)

###### 图 3-10。自动编码器和变分自动编码器中编码器的区别

编码器只需要将每个输入映射到一个均值向量和一个方差向量，不需要担心潜在空间维度之间的协方差。变分自动编码器假设潜在空间中的维度之间没有相关性。

方差值始终为正，因此我们实际上选择将其映射到*方差的对数*，因为这可以取任何实数范围内的值（ <math alttext="负无穷"><mrow><mo>-</mo> <mi>∞</mi></mrow></math> , <math alttext="正无穷"><mi>∞</mi></math> ）。这样我们可以使用神经网络作为编码器，将输入图像映射到均值和对数方差向量。

总之，编码器将每个输入图像编码为两个向量，这两个向量一起定义了潜在空间中的多变量正态分布：

`z_mean`

分布的均值点

`z_log_var`

每个维度的方差的对数

我们可以使用以下方程从这些值定义的分布中采样一个点`z`：

```py
z = z_mean + z_sigma * epsilon
```

其中：

```py
z_sigma = exp(z_log_var * 0.5)
epsilon ~ N(0,I)
```

###### 提示

`z_sigma`（ <math alttext="sigma"><mi>σ</mi></math> ）和`z_log_var`（ <math alttext="对数左括号 sigma 平方右括号"><mrow><mo form="prefix">log</mo> <mo>(</mo> <msup><mi>σ</mi> <mn>2</mn></msup> <mo>)</mo></mrow></math> ）之间的关系推导如下：

<math alttext="sigma equals exp left-parenthesis log left-parenthesis sigma right-parenthesis right-parenthesis equals exp left-parenthesis 2 log left-parenthesis sigma right-parenthesis slash 2 right-parenthesis equals exp left-parenthesis log left-parenthesis sigma squared right-parenthesis slash 2 right-parenthesis" display="block"><mrow><mi>σ</mi> <mo>=</mo> <mo form="prefix">exp</mo> <mrow><mo>(</mo> <mo form="prefix">log</mo> <mrow><mo>(</mo> <mi>σ</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>=</mo> <mo form="prefix">exp</mo> <mrow><mo>(</mo> <mn>2</mn> <mo form="prefix">log</mo> <mrow><mo>(</mo> <mi>σ</mi> <mo>)</mo></mrow> <mo>/</mo> <mn>2</mn> <mo>)</mo></mrow> <mo>=</mo> <mo form="prefix">exp</mo> <mrow><mo>(</mo> <mo form="prefix">log</mo> <mrow><mo>(</mo> <msup><mi>σ</mi> <mn>2</mn></msup> <mo>)</mo></mrow> <mo>/</mo> <mn>2</mn> <mo>)</mo></mrow></mrow></math>

变分自动编码器的解码器与普通自动编码器的解码器相同，给出了图 3-12 所示的整体架构。

![](img/gdl2_0312.png)

###### 图 3-12。VAE 架构图

为什么对编码器进行这种小改变有帮助？

先前，我们看到潜在空间不需要连续——即使点（-2, 2）解码为一个良好形成的凉鞋图像，也没有要求（-2.1, 2.1）看起来相似。现在，由于我们从`z_mean`周围的区域对随机点进行采样，解码器必须确保同一邻域中的所有点在解码时产生非常相似的图像，以便重构损失保持较小。这是一个非常好的特性，确保即使我们选择一个解码器从未见过的潜在空间中的点，它也可能解码为一个良好形成的图像。

### 构建 VAE 编码器

现在让我们看看如何在 Keras 中构建这个编码器的新版本。

# 运行此示例的代码

此示例的代码可以在位于书籍存储库中的*notebooks/03_vae/02_vae_fashion/vae_fashion.ipynb*的 Jupyter 笔记本中找到。

该代码已经改编自由 Francois Chollet 创建的优秀[VAE 教程](https://oreil.ly/A7yqJ)，可在 Keras 网站上找到。

首先，我们需要创建一个新类型的`Sampling`层，这将允许我们从由`z_mean`和`z_log_var`定义的分布中进行采样，如示例 3-11 所示。

##### 示例 3-11. `Sampling`层

```py
class Sampling(layers.Layer): ![1](img/1.png)
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon ![2](img/2.png)
```

![1](img/#co_variational_autoencoders_CO4-1)

我们通过对 Keras 基础`Layer`类进行子类化来创建一个新层（请参阅“子类化 Layer 类”侧边栏）。

![2](img/#co_variational_autoencoders_CO4-2)

我们使用重参数化技巧（请参阅“重参数化技巧”侧边栏）来构建由`z_mean`和`z_log_var`参数化的正态分布的样本。

包括新的`Sampling`层在内的编码器的完整代码显示在示例 3-12 中。

##### 示例 3-12. 编码器

```py
encoder_input = layers.Input(
    shape=(32, 32, 1), name="encoder_input"
)
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(
    encoder_input
)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x)
z_mean = layers.Dense(2, name="z_mean")(x) ![1](img/1.png)
z_log_var = layers.Dense(2, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var]) ![2](img/2.png)

encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder") ![3](img/3.png)
```

![1](img/#co_variational_autoencoders_CO5-1)

我们将`Flatten`层直接连接到 2D 潜在空间，而不是直接连接到`z_mean`和`z_log_var`层。

![2](img/#co_variational_autoencoders_CO5-2)

`Sampling`层从由参数`z_mean`和`z_log_var`定义的正态分布中对潜在空间中的点`z`进行采样。

![3](img/#co_variational_autoencoders_CO5-3)

定义编码器的 Keras `Model`——一个接受输入图像并输出`z_mean`、`z_log_var`和由这些参数定义的正态分布中的采样点`z`的模型。

编码器的摘要显示在表 3-4 中。

表 3-4. VAE 编码器的模型摘要

| Layer (type) | 输出形状 | 参数 # | 连接到 |
| --- | --- | --- | --- |
| InputLayer (input) | (None, 32, 32, 1) | 0 | [] |
| Conv2D (conv2d_1) | (None, 16, 16, 32) | 320 | [input] |
| Conv2D (conv2d_2) | (None, 8, 8, 64) | 18,496 | [conv2d_1] |
| Conv2D (conv2d_3) | (None, 4, 4, 128) | 73,856 | [conv2d_2] |
| Flatten (flatten) | (None, 2048) | 0 | [conv2d_3] |
| Dense (z_mean) | (None, 2) | 4,098 | [flatten] |
| Dense (z_log_var) | (None, 2) | 4,098 | [flatten] |
| Sampling (z) | (None, 2) | 0 | [z_mean, z_log_var] |
| 总参数 | 100,868 |
| 可训练参数 | 100,868 |
| 不可训练参数 | 0 |

我们需要更改的原始自动编码器的唯一其他部分是损失函数。

## 损失函数

先前，我们的损失函数仅包括图像与通过编码器和解码器传递后的尝试副本之间的*重构损失*。重构损失也出现在变分自动编码器中，但现在我们需要一个额外的组件：*Kullback-Leibler（KL）散度*项。

KL 散度是衡量一个概率分布与另一个之间差异的一种方式。在 VAE 中，我们想要衡量我们的具有参数`z_mean`和`z_log_var`的正态分布与标准正态分布之间的差异。在这种特殊情况下，可以证明 KL 散度具有以下封闭形式：

```py
kl_loss = -0.5 * sum(1 + z_log_var - z_mean ^ 2 - exp(z_log_var))
```

或者用数学符号表示：

<math alttext="upper D Subscript upper K upper L Baseline left-bracket upper N left-parenthesis mu comma sigma parallel-to upper N left-parenthesis 0 comma 1 right-parenthesis right-bracket equals minus one-half sigma-summation left-parenthesis 1 plus l o g left-parenthesis sigma squared right-parenthesis minus mu squared minus sigma squared right-parenthesis" display="block"><mstyle scriptlevel="0" displaystyle="true"><mrow><msub><mi>D</mi> <mrow><mi>K</mi><mi>L</mi></mrow></msub> <mrow><mo>[</mo> <mi>N</mi> <mrow><mo>(</mo> <mi>μ</mi> <mo>,</mo> <mi>σ</mi> <mo>∥</mo> <mi>N</mi> <mrow><mo>(</mo> <mn>0</mn> <mo>,</mo> <mn>1</mn> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>=</mo> <mo>-</mo></mrow> <mfrac><mn>1</mn> <mn>2</mn></mfrac> <mo>∑</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>+</mo> <mi>l</mi> <mi>o</mi> <mi>g</mi> <mrow><mo>(</mo> <msup><mi>σ</mi> <mn>2</mn></msup> <mo>)</mo></mrow> <mo>-</mo> <msup><mi>μ</mi> <mn>2</mn></msup> <mo>-</mo> <msup><mi>σ</mi> <mn>2</mn></msup> <mo>)</mo></mrow></mrow></mstyle></math>

总和取自潜在空间中的所有维度。当所有维度上的`z_mean = 0`和`z_log_var = 0`时，`kl_loss`被最小化为 0。当这两个项开始与 0 不同，`kl_loss`增加。

总的来说，KL 散度项惩罚网络将观测编码为与标准正态分布的参数明显不同的`z_mean`和`z_log_var`变量，即`z_mean = 0`和`z_log_var = 0`。

为什么将这个损失函数的添加有助于什么？

首先，我们现在有一个明确定义的分布，可以用于选择潜在空间中的点——标准正态分布。其次，由于这个项试图将所有编码分布推向标准正态分布，因此大型间隙形成的机会较小。相反，编码器将尝试对称且高效地使用原点周围的空间。

在原始 VAE 论文中，VAE 的损失函数简单地是重建损失和 KL 散度损失项的加法。这个变体（β-VAE）包括一个因子，用于对 KL 散度进行加权，以确保它与重建损失平衡良好。如果我们过分权重重建损失，KL 损失将不会产生所需的调节效果，我们将看到与普通自动编码器相同的问题。如果 KL 散度项权重过大，KL 散度损失将占主导地位，重建图像将很差。在训练 VAE 时，这个权重项是需要调整的参数之一。

## 训练变分自动编码器

示例 3-13 展示了我们如何将整体 VAE 模型构建为抽象 Keras `Model`类的子类。这使我们能够在自定义的`train_step`方法中包含损失函数的 KL 散度项的计算。

##### 示例 3-13。训练 VAE

```py
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs): ![1](img/1.png)
        z_mean, z_log_var, z = encoder(inputs)
        reconstruction = decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data): ![2](img/2.png)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                500
                * losses.binary_crossentropy(
                    data, reconstruction, axis=(1, 2, 3)
                )
            ) ![3](img/3.png)
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5
                    * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis = 1,
                )
            )
            total_loss = reconstruction_loss + kl_loss ![4](img/4.png)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

vae = VAE(encoder, decoder)
vae.compile(optimizer="adam")
vae.fit(
    train,
    epochs=5,
    batch_size=100
)
```

![1](img/#co_variational_autoencoders_CO6-1)

这个函数描述了我们希望在特定输入图像上返回的内容，我们称之为 VAE。

![2](img/#co_variational_autoencoders_CO6-2)

这个函数描述了 VAE 的一个训练步骤，包括损失函数的计算。

![3](img/#co_variational_autoencoders_CO6-3)

重建损失中使用了一个 beta 值为 500。

![4](img/#co_variational_autoencoders_CO6-4)

总损失是重建损失和 KL 散度损失的总和。

# 梯度带

TensorFlow 的*梯度带*是一种机制，允许在模型前向传递期间计算操作的梯度。要使用它，您需要将执行您想要区分的操作的代码包装在`tf.GradientTape()`上下文中。一旦记录了操作，您可以通过调用`tape.gradient()`计算损失函数相对于某些变量的梯度。然后可以使用这些梯度来更新变量与优化器。

这个机制对于计算自定义损失函数的梯度（就像我们在这里所做的那样）以及创建自定义训练循环非常有用，正如我们将在第四章中看到的。

## 变分自动编码器的分析

现在我们已经训练了我们的 VAE，我们可以使用编码器对测试集中的图像进行编码，并在潜在空间中绘制`z_mean`值。我们还可以从标准正态分布中进行采样，生成潜在空间中的点，并使用解码器将这些点解码回像素空间，以查看 VAE 的性能如何。

图 3-13 展示了新潜在空间的结构，以及一些采样点和它们的解码图像。我们可以立即看到潜在空间的组织方式发生了几处变化。

![](img/gdl2_0313.png)

###### 图 3-13。新的潜在空间：黑点显示每个编码图像的`z_mean`值，而蓝点显示潜在空间中的一些采样点（其解码图像显示在右侧）

首先，KL 散度损失项确保编码图像的`z_mean`和`z_log_var`值永远不会偏离标准正态分布太远。其次，由于编码器现在是随机的而不是确定性的，因此现在的潜在空间更加连续，因此没有那么多形状不佳的图像。

最后，通过按服装类型对潜在空间中的点进行着色（图 3-14），我们可以看到没有任何一种类型受到优待。右侧的图显示了空间转换为*p*值——我们可以看到每种颜色大致上都有相同的表示。再次强调，重要的是要记住在训练过程中根本没有使用标签；VAE 已经自己学会了各种服装形式，以帮助最小化重构损失。

![](img/gdl2_0314.png)

###### 图 3-14。VAE 的潜在空间按服装类型着色

# 探索潜在空间

到目前为止，我们对自动编码器和变分自动编码器的所有工作都局限于具有两个维度的潜在空间。这帮助我们在页面上可视化 VAE 的内部工作原理，并理解我们对自动编码器架构所做的小调整是如何将其转变为一种更强大的网络类别，可用于生成建模。

现在让我们将注意力转向更复杂的数据集，并看看当我们增加潜在空间的维度时，变分自动编码器可以实现的惊人成就。

# 运行此示例的代码

此示例的代码可以在书籍存储库中的 Jupyter 笔记本中找到，位置为*notebooks/03_vae/03_faces/vae_faces.ipynb*。

## CelebA 数据集

我们将使用[CelebFaces Attributes (CelebA)数据集](https://oreil.ly/tEUnh)来训练我们的下一个变分自动编码器。这是一个包含超过 200,000 张名人面孔彩色图像的集合，每张图像都带有各种标签（例如，*戴帽子*，*微笑*等）。一些示例显示在图 3-15 中。

![](img/gdl2_0315.png)

###### 图 3-15。CelebA 数据集的一些示例（来源：[Liu 等，2015](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))^(3)

当我们开始探索这些特征如何在多维潜在空间中被捕获时，我们当然不需要标签来训练 VAE，但这些标签以后会很有用。一旦我们的 VAE 训练完成，我们就可以从潜在空间中进行采样，生成名人面孔的新示例。

CelebA 数据集也可以通过 Kaggle 获得，因此您可以通过在书籍存储库中运行 Kaggle 数据集下载脚本来下载数据集，如示例 3-14 所示。这将把图像和相关元数据保存到*/data*文件夹中。

##### 示例 3-14。下载 CelebA 数据集

```py
bash scripts/download_kaggle_data.sh jessicali9530 celeba-dataset
```

我们使用 Keras 函数`image_dataset_from_directory`来创建一个指向存储图像的目录的 TensorFlow 数据集，如示例 3-15 所示。这使我们能够在需要时（例如在训练期间）将图像批量读入内存，以便我们可以处理大型数据集，而不必担心将整个数据集装入内存。它还将图像调整大小为 64×64，在像素值之间进行插值。

##### 示例 3-15。处理 CelebA 数据集

```py
train_data = utils.image_dataset_from_directory(
    "/app/data/celeba-dataset/img_align_celeba/img_align_celeba",
    labels=None,
    color_mode="rgb",
    image_size=(64, 64),
    batch_size=128,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)
```

原始数据在范围[0, 255]内进行缩放以表示像素强度，我们将其重新缩放到范围[0, 1]，如示例 3-16 所示。

##### 示例 3-16。处理 CelebA 数据集

## 训练变分自动编码器

面部模型的网络架构与 Fashion-MNIST 示例类似，有一些细微差异：

+   我们的数据现在有三个输入通道（RGB），而不是一个（灰度）。这意味着我们需要将解码器的最后一个卷积转置层中的通道数更改为 3。

+   我们将使用一个具有 200 维而不是 2 维的潜在空间。由于面孔比时尚 MNIST 图像复杂得多，我们增加了潜在空间的维度，以便网络可以从图像中编码出令人满意的细节量。

+   每个卷积层后都有批量归一化层以稳定训练。尽管每个批次运行时间较长，但达到相同损失所需的批次数量大大减少。

+   我们将 KL 散度的β因子增加到 2,000。这是一个需要调整的参数；对于这个数据集和架构，这个值被发现可以产生良好的结果。

编码器和解码器的完整架构分别显示在表 3-5 和表 3-6 中。 

表 3-5。VAE 面部编码器的模型摘要

| 层（类型） | 输出形状 | 参数 # | 连接到 |
| --- | --- | --- | --- |
| 输入层（input） | (None, 32, 32, 3) | 0 | [] |
| 卷积层（conv2d_1） | (None, 16, 16, 128) | 3,584 | [input] |
| 批量归一化（bn_1） | (None, 16, 16, 128) | 512 | [conv2d_1] |
| LeakyReLU（lr_1） | (None, 16, 16, 128) | 0 | [bn_1] |
| 卷积层（conv2d_2） | (None, 8, 8, 128) | 147,584 | [lr_1] |
| 批量归一化（bn_2） | (None, 8, 8, 128) | 512 | [conv2d_2] |
| LeakyReLU（lr_2） | (None, 8, 8, 128) | 0 | [bn_2] |
| 卷积层（conv2d_3） | (None, 4, 4, 128) | 147,584 | [lr_2] |
| 批量归一化（bn_3） | (None, 4, 4, 128) | 512 | [conv2d_3] |
| LeakyReLU（lr_3） | (None, 4, 4, 128) | 0 | [bn_3] |
| 卷积层（conv2d_4） | (None, 2, 2, 128) | 147,584 | [lr_3] |
| 批量归一化（bn_4） | (None, 2, 2, 128) | 512 | [conv2d_4] |
| LeakyReLU（lr_4） | (None, 2, 2, 128) | 0 | [bn_4] |
| 展平（flatten） | (None, 512) | 0 | [lr_4] |
| 密集层（z_mean） | (None, 200) | 102,600 | [flatten] |
| 密集层（z_log_var） | (None, 200) | 102,600 | [flatten] |
| 采样（z） | (None, 200) | 0 | [z_mean, z_log_var] |
| 总参数 | 653,584 |
| 可训练参数 | 652,560 |
| 不可训练参数 | 1,024 |

表 3-6。VAE 面部解码器的模型摘要

| 层（类型） | 输出形状 | 参数 # |
| --- | --- | --- |
| 输入层 | (None, 200) | 0 |
| 密集层 | (None, 512) | 102,912 |
| 批量归一化 | (None, 512) | 2,048 |
| LeakyReLU | (None, 512) | 0 |
| 重塑 | (None, 2, 2, 128) | 0 |
| 转置卷积层 | (None, 4, 4, 128) | 147,584 |
| 批量归一化 | (None, 4, 4, 128) | 512 |
| LeakyReLU | (None, 4, 4, 128) | 0 |
| 转置卷积层 | (None, 8, 8, 128) | 147,584 |
| 批量归一化 | (None, 8, 8, 128) | 512 |
| LeakyReLU | (None, 8, 8, 128) | 0 |
| 转置卷积层 | (None, 16, 16, 128) | 147,584 |
| 批量归一化 | (None, 16, 16, 128) | 512 |
| LeakyReLU | (None, 16, 16, 128) | 0 |
| 转置卷积层 | (None, 32, 32, 128) | 147,584 |
| 批量归一化 | (None, 32, 32, 128) | 512 |
| LeakyReLU | (None, 32, 32, 128) | 0 |
| 转置卷积层 | (None, 32, 32, 3) | 3,459 |
| 总参数 | 700,803 |
| 可训练参数 | 698,755 |
| 不可训练参数 | 2,048 |

在训练大约五个时期后，我们的 VAE 应该能够生成名人面孔的新颖图像！

## 变分自动编码器的分析

首先，让我们看一下重建面孔的样本。图 3-16 中的顶行显示原始图像，底行显示它们通过编码器和解码器后的重建图像。

![](img/gdl2_0316.png)

###### 图 3-16。通过编码器和解码器传递后重建的面孔

我们可以看到 VAE 成功地捕捉了每张脸的关键特征-头部角度、发型、表情等。一些细节缺失，但重要的是要记住，构建变分自动编码器的目的不是为了实现完美的重构损失。我们的最终目标是从潜在空间中取样，以生成新的面孔。

为了实现这一点，我们必须检查潜在空间中的点的分布是否大致类似于多元标准正态分布。如果我们看到任何维度与标准正态分布明显不同，那么我们可能应该减少重构损失因子，因为 KL 散度项没有足够的影响。

我们潜在空间中的前 50 个维度显示在图 3-17 中。没有任何分布突出为与标准正态有显著不同，所以我们可以继续生成一些面孔！

![](img/gdl2_0317.png)

###### 图 3-17。潜在空间中前 50 个维度的点的分布

## 生成新面孔

要生成新的面孔，我们可以使用示例 3-17 中的代码。

##### 示例 3-17。从潜在空间生成新面孔

```py
grid_width, grid_height = (10,3)
z_sample = np.random.normal(size=(grid_width * grid_height, 200)) ![1](img/1.png)

reconstructions = decoder.predict(z_sample) ![2](img/2.png)

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.imshow(reconstructions[i, :, :]) ![3](img/3.png)
```

![1](img/#co_variational_autoencoders_CO7-1)

从具有 200 维度的标准多元正态分布中采样 30 个点。

![2](img/#co_variational_autoencoders_CO7-2)

解码采样点。

![3](img/#co_variational_autoencoders_CO7-3)

绘制图像！

输出显示在图 3-18 中。

![](img/gdl2_0318.png)

###### 图 3-18。新生成的面孔

令人惊讶的是，VAE 能够将我们从标准正态分布中采样的一组点转换为令人信服的人脸图像。这是我们第一次看到生成模型真正力量的一瞥！

接下来，让我们看看是否可以开始使用潜在空间对生成的图像执行一些有趣的操作。

## 潜在空间算术

将图像映射到较低维度的潜在空间的一个好处是，我们可以对这个潜在空间中的向量进行算术运算，当解码回原始图像域时具有视觉类比。

例如，假设我们想要拍摄一个看起来很伤心的人的图像，并给他们一个微笑。为此，我们首先需要找到潜在空间中指向增加微笑方向的向量。将这个向量添加到潜在空间中原始图像的编码中，将给我们一个新点，解码后应该给我们一个更加微笑的原始图像版本。

那么我们如何找到*微笑*向量呢？CelebA 数据集中的每个图像都带有属性标签，其中之一是`Smiling`。如果我们在具有属性`Smiling`的编码图像的潜在空间中取平均位置，并减去没有属性`Smiling`的编码图像的平均位置，我们将获得指向`Smiling`方向的向量，这正是我们需要的。

在潜在空间中，我们进行以下向量算术，其中`alpha`是一个确定添加或减去多少特征向量的因子：

```py
z_new = z + alpha * feature_vector
```

让我们看看这个过程。图 3-19 显示了几幅已编码到潜在空间中的图像。然后，我们添加或减去某个向量的倍数（例如，`Smiling`、`Black_Hair`、`Eyeglasses`、`Young`、`Male`、`Blond_Hair`）以获得图像的不同版本，只改变相关特征。

![](img/gdl2_0319.png)

###### 图 3-19。向面孔添加和减去特征

令人惊奇的是，即使我们在潜在空间中移动点的距离相当大，核心图像仍然大致相同，除了我们想要操作的一个特征。这展示了变分自动编码器在捕捉和调整图像中的高级特征方面的能力。

## 在面孔之间变形

我们可以使用类似的想法在两个面孔之间变形。想象一下潜在空间中表示两个图像的两个点 A 和 B。如果您从点 A 开始沿直线走向点 B，解码沿途的每个点，您将看到从起始面孔到结束面孔的逐渐过渡。

数学上，我们正在遍历一条直线，可以用以下方程描述：

```py
z_new = z_A * (1- alpha) + z_B * alpha
```

在这里，`alpha`是一个介于 0 和 1 之间的数字，确定我们离点 A 有多远。

图 3-20 展示了这一过程的实际操作。我们取两个图像，将它们编码到潜在空间中，然后在它们之间的直线上以固定间隔解码点。

![](img/gdl2_0320.png)

###### 图 3-20。在两个面孔之间变形

值得注意的是过渡的平滑性——即使同时有多个要同时更改的特征（例如，去除眼镜、头发颜色、性别），VAE 也能够流畅地实现这一点，显示出 VAE 的潜在空间确实是一个可以遍历和探索以生成多种不同人脸的连续空间。`  `# 摘要

在本章中，我们看到变分自动编码器是生成建模工具箱中的一个强大工具。我们首先探讨了如何使用普通自动编码器将高维图像映射到低维潜在空间，以便从单独无信息的像素中提取高级特征。然而，我们很快发现使用普通自动编码器作为生成模型存在一些缺点——例如，从学习的潜在空间中进行采样是有问题的。

变分自动编码器通过在模型中引入随机性并约束潜在空间中的点如何分布来解决这些问题。我们看到，通过一些微小的调整，我们可以将我们的自动编码器转变为变分自动编码器，从而赋予它成为真正生成模型的能力。

最后，我们将我们的新技术应用于面部生成问题，并看到我们如何简单地解码标准正态分布中的点以生成新的面孔。此外，通过在潜在空间内执行向量算术，我们可以实现一些惊人的效果，如面部变形和特征操作。

在下一章中，我们将探索一种不同类型的模型，这种模型仍然是生成图像建模的一种流行选择：生成对抗网络。

^(1) Diederik P. Kingma 和 Max Welling，“自动编码变分贝叶斯”，2013 年 12 月 20 日，[*https://arxiv.org/abs/1312.6114*](https://arxiv.org/abs/1312.6114)。

^(2) Vincent Dumoulin 和 Francesco Visin，“深度学习卷积算术指南”，2018 年 1 月 12 日，[*https://arxiv.org/abs/1603.07285*](https://arxiv.org/abs/1603.07285)。

^(3) Ziwei Liu 等，“大规模 CelebFaces 属性（CelebA）数据集”，2015 年，[*http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html*](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。
