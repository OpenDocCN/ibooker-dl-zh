# 第四章。生成对抗网络

2014 年，Ian Goodfellow 等人在蒙特利尔的神经信息处理系统会议（NeurIPS）上发表了一篇名为“生成对抗网络”的论文^(1)。生成对抗网络（或者更常见的称为 GAN）的引入现在被认为是生成建模历史上的一个关键转折点，因为这篇论文中提出的核心思想衍生出了一些最成功和令人印象深刻的生成模型。

本章将首先阐述 GAN 的理论基础，然后我们将看到如何使用 Keras 构建我们自己的 GAN。

# 介绍

让我们从一个简短的故事开始，以阐明 GAN 训练过程中使用的一些基本概念。

Brickki 砖块和伪造者的故事描述了生成对抗网络的训练过程。

GAN 是生成器和鉴别器之间的一场战斗。生成器试图将随机噪声转换为看起来像是从原始数据集中抽样的观察结果，而鉴别器试图预测一个观察结果是来自原始数据集还是生成器的伪造品之一。两个网络的输入和输出示例显示在图 4-2 中。

![](img/gdl2_0402.png)

###### 图 4-2\. GAN 中两个网络的输入和输出

在过程开始时，生成器输出嘈杂的图像，鉴别器随机预测。GAN 的关键在于我们如何交替训练这两个网络，使得随着生成器变得更擅长欺骗鉴别器，鉴别器必须适应以保持其正确识别哪些观察结果是伪造的能力。这驱使生成器找到欺骗鉴别器的新方法，循环继续。

# 深度卷积生成对抗网络（DCGAN）

为了看到这个过程，让我们开始在 Keras 中构建我们的第一个 GAN，以生成砖块的图片。

我们将密切关注 GAN 的第一篇重要论文之一，“使用深度卷积生成对抗网络进行无监督表示学习”^(2)。在这篇 2015 年的论文中，作者展示了如何构建一个深度卷积 GAN 来从各种数据集中生成逼真的图像。他们还引入了一些显著改进生成图像质量的变化。

# 运行此示例的代码

这个例子的代码可以在位于书籍存储库中的 Jupyter 笔记本中找到，路径为*notebooks/04_gan/01_dcgan/dcgan.ipynb*。

## 砖块数据集

首先，您需要下载训练数据。我们将使用通过 Kaggle 提供的[乐高砖块图像数据集](https://oreil.ly/3vp9f)。这是一个包含 40,000 张来自多个角度拍摄的 50 种不同玩具砖块的照片的计算机渲染集合。一些 Brickki 产品的示例图像显示在图 4-3 中。

![](img/gdl2_0403.png)

###### 图 4-3\. 砖块数据集中的图像示例

您可以通过在书籍存储库中运行 Kaggle 数据集下载脚本来下载数据集，如示例 4-1 所示。这将把图像和相关元数据保存到*/data*文件夹中。

##### 示例 4-1\. 下载砖块数据集

```py
bash scripts/download_kaggle_data.sh joosthazelzet lego-brick-images
```

我们使用 Keras 函数`image_dataset_from_directory`创建一个指向存储图像的目录的 TensorFlow 数据集，如示例 4-2 所示。这使我们能够在需要时（例如在训练期间）将图像批量读入内存，以便我们可以处理大型数据集而不必担心必须将整个数据集装入内存。它还将图像调整为 64×64 大小，插值像素值之间的差值。

##### 示例 4-2\. 从目录中的图像文件创建 TensorFlow 数据集

```py
train_data = utils.image_dataset_from_directory(
    "/app/data/lego-brick-images/dataset/",
    labels=None,
    color_mode="grayscale",
    image_size=(64, 64),
    batch_size=128,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)
```

原始数据在范围[0, 255]内缩放以表示像素强度。在训练 GAN 时，我们将数据重新缩放到范围[-1, 1]，以便我们可以在生成器的最后一层使用 tanh 激活函数，该函数提供比 sigmoid 函数更强的梯度（示例 4-3）。

##### 示例 4-3。预处理砖块数据集

```py
def preprocess(img):
    img = (tf.cast(img, "float32") - 127.5) / 127.5
    return img

train = train_data.map(lambda x: preprocess(x))
```

现在让我们看看如何构建鉴别器。## 鉴别器

鉴别器的目标是预测图像是真实的还是伪造的。这是一个监督图像分类问题，因此我们可以使用与我们在第二章中使用的类似架构：堆叠的卷积层，带有单个输出节点。

我们将构建的鉴别器的完整架构显示在表 4-1 中。

表 4-1。鉴别器的模型摘要

| 层（类型） | 输出形状 | 参数 # |
| --- | --- | --- |
| InputLayer | (None, 64, 64, 1) | 0 |
| Conv2D | (None, 32, 32, 64) | 1,024 |
| LeakyReLU | (None, 32, 32, 64) | 0 |
| Dropout | (None, 32, 32, 64) | 0 |
| Conv2D | (None, 16, 16, 128) | 131,072 |
| BatchNormalization | (None, 16, 16, 128) | 512 |
| LeakyReLU | (None, 16, 16, 128) | 0 |
| Dropout | (None, 16, 16, 128) | 0 |
| Conv2D | (None, 8, 8, 256) | 524,288 |
| BatchNormalization | (None, 8, 8, 256) | 1,024 |
| LeakyReLU | (None, 8, 8, 256) | 0 |
| Dropout | (None, 8, 8, 256) | 0 |
| Conv2D | (None, 4, 4, 512) | 2,097,152 |
| BatchNormalization | (None, 4, 4, 512) | 2,048 |
| LeakyReLU | (None, 4, 4, 512) | 0 |
| Dropout | (None, 4, 4, 512) | 0 |
| Conv2D | (None, 1, 1, 1) | 8,192 |
| Flatten | (None, 1) | 0 |
| 总参数 | 2,765,312 |
| 可训练参数 | 2,763,520 |
| 不可训练参数 | 1,792 |

提供构建鉴别器的 Keras 代码在示例 4-4 中。

##### 示例 4-4。鉴别器

```py
discriminator_input = layers.Input(shape=(64, 64, 1)) ![1](img/1.png)
x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias = False)(
    discriminator_input
) ![2](img/2.png)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    128, kernel_size=4, strides=2, padding="same", use_bias = False
)(x)
x = layers.BatchNormalization(momentum = 0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    256, kernel_size=4, strides=2, padding="same", use_bias = False
)(x)
x = layers.BatchNormalization(momentum = 0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    512, kernel_size=4, strides=2, padding="same", use_bias = False
)(x)
x = layers.BatchNormalization(momentum = 0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(
    1,
    kernel_size=4,
    strides=1,
    padding="valid",
    use_bias = False,
    activation = 'sigmoid'
)(x)
discriminator_output = layers.Flatten()(x) ![3](img/3.png)

discriminator = models.Model(discriminator_input, discriminator_output) ![4](img/4.png)
```

![1](img/#co_generative_adversarial_networks_CO1-1)

定义鉴别器的`Input`层（图像）。

![2](img/#co_generative_adversarial_networks_CO1-2)

将`Conv2D`层堆叠在一起，中间夹有`BatchNormalization`、`LeakyReLU`激活和`Dropout`层。

![3](img/#co_generative_adversarial_networks_CO1-3)

将最后一个卷积层展平-到这一点，张量的形状为 1×1×1，因此不需要最终的`Dense`层。

![4](img/#co_generative_adversarial_networks_CO1-4)

定义鉴别器的 Keras 模型-一个接受输入图像并输出介于 0 和 1 之间的单个数字的模型。

请注意，我们在一些`Conv2D`层中使用步幅为 2，以减少通过网络时张量的空间形状（原始图像中为 64，然后 32、16、8、4，最后为 1），同时增加通道数（灰度输入图像中为 1，然后 64、128、256，最后为 512），最终折叠为单个预测。

我们在最后一个`Conv2D`层上使用 sigmoid 激活函数，输出一个介于 0 和 1 之间的数字。

## 生成器

现在让我们构建生成器。生成器的输入将是从多元标准正态分布中抽取的向量。输出是与原始训练数据中的图像大小相同的图像。

这个描述可能让你想起变分自动编码器中的解码器。事实上，GAN 的生成器与 VAE 的解码器完全履行相同的目的：将潜在空间中的向量转换为图像。在生成建模中，从潜在空间映射回原始域的概念非常常见，因为它使我们能够操纵潜在空间中的向量以改变原始域中图像的高级特征。

我们将构建的生成器的架构显示在表 4-2 中。

表 4-2。生成器的模型摘要

| 层（类型） | 输出形状 | 参数 # |
| --- | --- | --- |
| InputLayer（无，100）0 |
| Reshape（无，1，1，100）0 |
| Conv2DTranspose（无，4，4，512）819,200 |
| BatchNormalization（无，4，4，512）2,048 |
| ReLU（无，4，4，512）0 |
| Conv2DTranspose（无，8，8，256）2,097,152 |
| BatchNormalization（无，8，8，256）1,024 |
| ReLU（无，8，8，256）0 |
| Conv2DTranspose（无，16，16，128）524,288 |
| BatchNormalization（无，16，16，128）512 |
| ReLU（无，16，16，128）0 |
| Conv2DTranspose（无，32，32，64）131,072 |
| BatchNormalization（无，32，32，64）256 |
| ReLU（无，32，32，64）0 |
| Conv2DTranspose（无，64，64，1）1,024 |
| 总参数 3,576,576 |
| 可训练参数 3,574,656 |
| 不可训练参数 1,920 |

构建生成器的代码在示例 4-5 中给出。

##### 示例 4-5。生成器

```py
generator_input = layers.Input(shape=(100,)) ![1](img/1.png)
x = layers.Reshape((1, 1, 100))(generator_input) ![2](img/2.png)
x = layers.Conv2DTranspose(
    512, kernel_size=4, strides=1, padding="valid", use_bias = False
)(x) ![3](img/3.png)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
    256, kernel_size=4, strides=2, padding="same", use_bias = False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
    128, kernel_size=4, strides=2, padding="same", use_bias = False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(
    64, kernel_size=4, strides=2, padding="same", use_bias = False
)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
generator_output = layers.Conv2DTranspose(
    1,
    kernel_size=4,
    strides=2,
    padding="same",
    use_bias = False,
    activation = 'tanh'
)(x) ![4](img/4.png)
generator = models.Model(generator_input, generator_output) ![5](img/5.png)
```

![1](img/#co_generative_adversarial_networks_CO2-1)

定义生成器的`Input`层-长度为 100 的向量。

![2](img/#co_generative_adversarial_networks_CO2-2)

我们使用一个`Reshape`层来给出一个 1×1×100 的张量，这样我们就可以开始应用卷积转置操作。

![3](img/#co_generative_adversarial_networks_CO2-3)

我们通过四个`Conv2DTranspose`层传递这些数据，其中夹在中间的是`BatchNormalization`和`LeakyReLU`层。

![4](img/#co_generative_adversarial_networks_CO2-4)

最终的`Conv2DTranspose`层使用 tanh 激活函数将输出转换为范围[-1,1]，以匹配原始图像域。

![5](img/#co_generative_adversarial_networks_CO2-5)

定义生成器的 Keras 模型-接受长度为 100 的向量并输出形状为`[64，64，1]`的张量。

请注意，我们在一些`Conv2DTranspose`层中使用步幅为 2，以增加通过网络传递时张量的空间形状（原始向量中为 1，然后为 4，8，16，32，最终为 64），同时减少通道数（512，然后为 256，128，64，最终为 1 以匹配灰度输出）。

## 训练 DCGAN

正如我们所看到的，在 DCGAN 中生成器和鉴别器的架构非常简单，并且与我们在第三章中看到的 VAE 模型并没有太大不同。理解 GAN 的关键在于理解生成器和鉴别器的训练过程。

我们可以通过创建一个训练集来训练鉴别器，其中一些图像是来自训练集的*真实*观察结果，一些是来自生成器的*假*输出。然后我们将其视为一个监督学习问题，其中真实图像的标签为 1，假图像的标签为 0，损失函数为二元交叉熵。

我们应该如何训练生成器？我们需要找到一种评分每个生成的图像的方法，以便它可以优化到高分图像。幸运的是，我们有一个鉴别器正是这样做的！我们可以生成一批图像并将其通过鉴别器以获得每个图像的分数。然后生成器的损失函数就是这些概率与一个全为 1 的向量之间的二元交叉熵，因为我们希望训练生成器生成鉴别器认为是真实的图像。

至关重要的是，我们必须交替训练这两个网络，确保我们一次只更新一个网络的权重。例如，在生成器训练过程中，只有生成器的权重会被更新。如果我们允许鉴别器的权重也发生变化，那么鉴别器将只是调整自己，以便更有可能预测生成的图像是真实的，这不是期望的结果。我们希望生成的图像被预测接近 1（真实），因为生成器强大，而不是因为鉴别器弱。

鉴别器和生成器的训练过程的图示如图 4-5 所示。

![](img/gdl2_0405.png)

###### 图 4-5。训练 DCGAN-灰色框表示在训练过程中权重被冻结

Keras 提供了创建自定义`train_step`函数来实现这一逻辑的能力。示例 4-7 展示了完整的`DCGAN`模型类。

##### 示例 4-7。编译 DCGAN

```py
class DCGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super(DCGAN, self).compile()
        self.loss_fn = losses.BinaryCrossentropy() ![1](img/1.png)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        ) ![2](img/2.png)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(
                random_latent_vectors, training = True
            ) ![3](img/3.png)
            real_predictions = self.discriminator(real_images, training = True) ![4](img/4.png)
            fake_predictions = self.discriminator(
                generated_images, training = True
            ) ![5](img/5.png)

            real_labels = tf.ones_like(real_predictions)
            real_noisy_labels = real_labels + 0.1 * tf.random.uniform(
                tf.shape(real_predictions)
            )
            fake_labels = tf.zeros_like(fake_predictions)
            fake_noisy_labels = fake_labels - 0.1 * tf.random.uniform(
                tf.shape(fake_predictions)
            )

            d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)
            d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)
            d_loss = (d_real_loss + d_fake_loss) / 2.0 ![6](img/6.png)

            g_loss = self.loss_fn(real_labels, fake_predictions) ![7](img/7.png)

        gradients_of_discriminator = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        gradients_of_generator = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )

        self.d_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        ) ![8](img/8.png)
        self.g_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

dcgan = DCGAN(
    discriminator=discriminator, generator=generator, latent_dim=100
)

dcgan.compile(
    d_optimizer=optimizers.Adam(
        learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.999
    ),
    g_optimizer=optimizers.Adam(
        learning_rate=0.0002, beta_1 = 0.5, beta_2 = 0.999
    ),
)

dcgan.fit(train, epochs=300)
```

![1](img/#co_generative_adversarial_networks_CO3-1)

生成器和鉴别器的损失函数是`BinaryCrossentropy`。

![2](img/#co_generative_adversarial_networks_CO3-2)

为了训练网络，首先从多元标准正态分布中抽取一批向量。

![3](img/#co_generative_adversarial_networks_CO3-3)

接下来，通过生成器生成一批生成的图像。

![4](img/#co_generative_adversarial_networks_CO3-4)

现在让鉴别器预测一批真实图像的真实性...​

![5](img/#co_generative_adversarial_networks_CO3-5)

...​和一批生成的图像。

![6](img/#co_generative_adversarial_networks_CO3-6)

鉴别器损失是真实图像（标签为 1）和假图像（标签为 0）之间的平均二元交叉熵。

![7](img/#co_generative_adversarial_networks_CO3-7)

生成器损失是鉴别器对生成图像的预测与标签 1 之间的二元交叉熵。

![8](img/#co_generative_adversarial_networks_CO3-8)

分别更新鉴别器和生成器的权重。

鉴别器和生成器不断争夺主导地位，这可能使 DCGAN 训练过程不稳定。理想情况下，训练过程将找到一个平衡点，使生成器能够从鉴别器那里学习有意义的信息，图像的质量将开始提高。经过足够的 epochs，鉴别器往往最终占据主导地位，如图 4-6 所示，但这可能不是问题，因为生成器可能已经学会在这一点上生成足够高质量的图像。

![](img/gdl2_0406.png)

###### 图 4-6。训练过程中鉴别器和生成器的损失和准确率

# 向标签添加噪声

在训练 GAN 时的一个有用技巧是向训练标签添加少量随机噪声。这有助于改善训练过程的稳定性并增强生成的图像。这种*标签平滑*作为一种驯服鉴别器的方式，使其面临更具挑战性的任务，不会压倒生成器。

## DCGAN 的分析

通过观察训练过程中特定时期生成器生成的图像（图 4-7），可以清楚地看到生成器越来越擅长生成可能来自训练集的图像。

![](img/gdl2_0407.png)

###### 图 4-7。训练过程中特定时期生成器的输出

神奇的是神经网络能够将随机噪声转换为有意义的东西。值得记住的是，我们除了原始像素之外没有提供模型任何额外的特征，因此它必须自行解决如何绘制阴影、立方体和圆等高级概念。

成功生成模型的另一个要求是它不仅仅是复制训练集中的图像。为了测试这一点，我们可以找到训练集中与特定生成示例最接近的图像。一个好的距离度量是*L1 距离*，定义为：

```py
def compare_images(img1, img2):
    return np.mean(np.abs(img1 - img2))
```

图 4-8 显示了一些生成图像在训练集中最接近的观察结果。我们可以看到，虽然生成图像与训练集之间存在一定程度的相似性，但它们并不完全相同。这表明生成器已经理解了这些高级特征，并且能够生成与已经看到的图像不同的示例。

![](img/gdl2_0408.png)

###### 图 4-8\. 从训练集中生成图像的最接近匹配

## GAN 训练：技巧和窍门

虽然 GAN 是生成建模的重大突破，但训练起来也非常困难。在本节中，我们将探讨训练 GAN 时遇到的一些最常见问题和挑战，以及潜在的解决方案。在下一节中，我们将看一些更基本的调整 GAN 框架的方法，以解决许多这些问题。

### 鉴别器压倒生成器

如果鉴别器变得过于强大，损失函数的信号变得太弱，无法驱动生成器中的任何有意义的改进。在最坏的情况下，鉴别器完全学会区分真实图像和假图像，梯度完全消失，导致没有任何训练，如图 4-9 所示。

![](img/gdl2_0409.png)

###### 图 4-9\. 当鉴别器压倒生成器时的示例输出

如果发现鉴别器损失函数坍缩，需要找到削弱鉴别器的方法。尝试以下建议：

+   增加鉴别器中`Dropout`层的`rate`参数，以减少通过网络的信息量。

+   降低鉴别器的学习率。

+   减少鉴别器中的卷积滤波器数量。

+   在训练鉴别器时向标签添加噪音。

+   在训练鉴别器时，随机翻转一些图像的标签。

### 生成器压倒鉴别器

如果鉴别器不够强大，生成器将找到一种方法轻松欺骗鉴别器，只需少量几乎相同的图像样本。这被称为*模式坍塌*。

例如，假设我们在不更新鉴别器的情况下训练生成器多个批次。生成器会倾向于找到一个始终欺骗鉴别器的单个观察（也称为*模式*），并开始将潜在输入空间中的每个点映射到这个图像。此外，损失函数的梯度会坍缩到接近 0，因此无法从这种状态中恢复。

即使我们尝试重新训练鉴别器以阻止它被这一点欺骗，生成器也会简单地找到另一个欺骗鉴别器的模式，因为它已经对其输入麻木，因此没有多样化其输出的动机。

模式坍塌的效果可以在图 4-10 中看到。

![](img/gdl2_0410.png)

###### 图 4-10\. 当生成器压倒鉴别器时模式坍塌的示例

如果发现生成器遭受模式坍塌，可以尝试使用与前一节中列出的相反建议来加强鉴别器。此外，可以尝试降低两个网络的学习率并增加批量大小。

### 无信息损失

由于深度学习模型被编译为最小化损失函数，自然会认为生成器的损失函数越小，生成的图像质量就越好。然而，由于生成器只针对当前鉴别器进行评分，而鉴别器不断改进，我们无法比较在训练过程中不同点评估的损失函数。实际上，在图 4-6 中，生成器的损失函数随时间增加，尽管图像质量明显提高。生成器损失与图像质量之间的缺乏相关性有时使得 GAN 训练难以监控。

### 超参数

正如我们所看到的，即使是简单的 GAN，也有大量的超参数需要调整。除了鉴别器和生成器的整体架构外，还有控制批量归一化、丢弃、学习率、激活层、卷积滤波器、内核大小、步幅、批量大小和潜在空间大小的参数需要考虑。GAN 对所有这些参数的微小变化非常敏感，找到一组有效的参数通常是经过有教养的试错过程，而不是遵循一套已建立的指导方针。

这就是为什么重要理解 GAN 的内部工作原理并知道如何解释损失函数——这样你就可以识别出可能改善模型稳定性的超参数的合理调整。

### 解决 GAN 的挑战

近年来，一些关键进展大大提高了 GAN 模型的整体稳定性，并减少了一些早期列出的问题的可能性，比如模式崩溃。

在本章的其余部分，我们将研究带有梯度惩罚的 Wasserstein GAN（WGAN-GP），该模型对我们迄今为止探索的 GAN 框架进行了几个关键调整，以改善图像生成过程的稳定性和质量。` `#带有梯度惩罚的 Wasserstein GAN（WGAN-GP）

在本节中，我们将构建一个 WGAN-GP 来从我们在第三章中使用的 CelebA 数据集中生成人脸。

# 运行此示例的代码

这个示例的代码可以在书库中的*notebooks/04_gan/02_wgan_gp/wgan_gp.ipynb*中找到。

这段代码是从由 Aakash Kumar Nain 创建的优秀的[WGAN-GP 教程](https://oreil.ly/dHYbC)中改编而来，该教程可在 Keras 网站上找到。

Wasserstein GAN（WGAN）是由 Arjovsky 等人在 2017 年的一篇论文中引入的，是稳定 GAN 训练的第一步。通过一些改变，作者们能够展示如何训练具有以下两个特性的 GAN（引用自论文）：

+   一个与生成器的收敛和样本质量相关的有意义的损失度量

+   优化过程的稳定性提高

具体来说，该论文为鉴别器和生成器引入了*Wasserstein 损失函数*。使用这个损失函数而不是二元交叉熵会导致 GAN 更稳定地收敛。

在本节中，我们将定义 Wasserstein 损失函数，然后看看我们需要对模型架构和训练过程做哪些其他更改以整合我们的新损失函数。

您可以在书库中的*chapter05/wgan-gp/faces/train.ipynb*中找到完整的模型类。

## Wasserstein 损失

让我们首先回顾一下二元交叉熵损失的定义——我们目前用来训练 GAN 的函数（方程 4-1）。

##### 方程 4-1. 二元交叉熵损失

<math alttext="minus StartFraction 1 Over n EndFraction sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis y Subscript i Baseline log left-parenthesis p Subscript i Baseline right-parenthesis plus left-parenthesis 1 minus y Subscript i Baseline right-parenthesis log left-parenthesis 1 minus p Subscript i Baseline right-parenthesis right-parenthesis" display="block"><mrow><mo>-</mo> <mfrac><mn>1</mn> <mi>n</mi></mfrac> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <mrow><mo>(</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo form="prefix">log</mo> <mrow><mo>(</mo> <msub><mi>p</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>+</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo form="prefix">log</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <msub><mi>p</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>

为了训练 GAN 鉴别器 D，我们计算了对比真实图像 x_i 的预测 p_i 与响应 y_i=1 以及对比生成图像 G(z_i)的预测 p_i 与响应 y_i=0 的损失。因此，对于 GAN 鉴别器，最小化损失函数可以写成方程 4-2 所示。

##### 方程 4-2. GAN 鉴别器损失最小化

<math alttext="min Underscript upper D Endscripts minus left-parenthesis double-struck upper E Subscript x tilde p Sub Subscript upper X Subscript Baseline left-bracket log upper D left-parenthesis x right-parenthesis right-bracket plus double-struck upper E Subscript z tilde p Sub Subscript upper Z Subscript Baseline left-bracket log left-parenthesis 1 minus upper D left-parenthesis upper G left-parenthesis z right-parenthesis right-parenthesis right-parenthesis right-bracket right-parenthesis" display="block"><mrow><munder><mo movablelimits="true" form="prefix">min</mo> <mi>D</mi></munder> <mo>-</mo> <mrow><mo>(</mo> <msub><mi>𝔼</mi> <mrow><mi>x</mi><mo>∼</mo><msub><mi>p</mi> <mi>X</mi></msub></mrow></msub> <mrow><mo>[</mo> <mo form="prefix">log</mo> <mrow><mi>D</mi> <mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>+</mo> <msub><mi>𝔼</mi> <mrow><mi>z</mi><mo>∼</mo><msub><mi>p</mi> <mi>Z</mi></msub></mrow></msub> <mrow><mo>[</mo> <mo form="prefix">log</mo> <mrow><mo>(</mo> <mn>1</mn> <mo>-</mo> <mi>D</mi> <mo>(</mo> <mi>G</mi> <mo>(</mo> <mi>z</mi> <mo>)</mo> <mo>)</mo> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>)</mo></mrow></mrow></math>

为了训练 GAN 生成器 G，我们计算了对比生成图像 G(z_i)的预测 p_i 与响应 y_i=1 的损失。因此，对于 GAN 生成器，最小化损失函数可以写成方程 4-3 所示。

##### 方程 4-3. GAN 生成器损失最小化

<math alttext="min Underscript upper G Endscripts minus left-parenthesis double-struck upper E Subscript z tilde p Sub Subscript upper Z Subscript Baseline left-bracket log upper D left-parenthesis upper G left-parenthesis z right-parenthesis right-parenthesis right-bracket right-parenthesis" display="block"><mrow><munder><mo movablelimits="true" form="prefix">min</mo> <mi>G</mi></munder> <mo>-</mo> <mrow><mo>(</mo> <msub><mi>𝔼</mi> <mrow><mi>z</mi><mo>∼</mo><msub><mi>p</mi> <mi>Z</mi></msub></mrow></msub> <mrow><mo>[</mo> <mo form="prefix">log</mo> <mrow><mi>D</mi> <mo>(</mo> <mi>G</mi> <mo>(</mo> <mi>z</mi> <mo>)</mo> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>)</mo></mrow></mrow></math>

现在让我们将其与 Wasserstein 损失函数进行比较。

首先，Wasserstein 损失要求我们使用 y_i=1 和 y_i=-1 作为标签，而不是 1 和 0。我们还从鉴别器的最后一层中移除了 sigmoid 激活，使得预测 p_i 不再受限于[0, 1]范围，而是可以是任何范围内的任意数字（负无穷，正无穷）。因此，在 WGAN 中，鉴别器通常被称为*评论家*，输出*分数*而不是概率。

Wasserstein 损失函数定义如下：

<math alttext="minus StartFraction 1 Over n EndFraction sigma-summation Underscript i equals 1 Overscript n Endscripts left-parenthesis y Subscript i Baseline p Subscript i Baseline right-parenthesis" display="block"><mrow><mo>-</mo> <mfrac><mn>1</mn> <mi>n</mi></mfrac> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <mrow><mo>(</mo> <msub><mi>y</mi> <mi>i</mi></msub> <msub><mi>p</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></math>

为了训练 WGAN 评论家<math alttext="upper D"><mi>D</mi></math>，我们计算真实图像的预测与响应之间的损失<math alttext="p Subscript i Baseline equals upper D left-parenthesis x Subscript i Baseline right-parenthesis"><mrow><msub><mi>p</mi> <mi>i</mi></msub> <mo>=</mo> <mi>D</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>)</mo></mrow></mrow></math>与响应<math alttext="y Subscript i Baseline equals 1"><mrow><msub><mi>y</mi> <mi>i</msub> <mo>=</mo> <mn>1</mn></mrow></math>，以及生成图像的预测与响应之间的损失<math alttext="p Subscript i Baseline equals upper D left-parenthesis upper G left-parenthesis z Subscript i Baseline right-parenthesis right-parenthesis"><mrow><msub><mi>p</mi> <mi>i</msub> <mo>=</mo> <mi>D</mi> <mrow><mo>(</mo> <mi>G</mi> <mrow><mo>(</mo> <msub><mi>z</mi> <mi>i</msub> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>与响应<math alttext="y Subscript i"><msub><mi>y</mi> <mi>i</msub></math> = -1。因此，对于 WGAN 评论家，最小化损失函数可以写成如下形式：

<math alttext="min Underscript upper D Endscripts minus left-parenthesis double-struck upper E Subscript x tilde p Sub Subscript upper X Subscript Baseline left-bracket upper D left-parenthesis x right-parenthesis right-bracket minus double-struck upper E Subscript z tilde p Sub Subscript upper Z Subscript Baseline left-bracket upper D left-parenthesis upper G left-parenthesis z right-parenthesis right-parenthesis right-bracket right-parenthesis" display="block"><mrow><munder><mo movablelimits="true" form="prefix">min</mo> <mi>D</mi></munder> <mo>-</mo> <mrow><mo>(</mo> <msub><mi>𝔼</mi> <mrow><mi>x</mi><mo>∼</mo><msub><mi>p</mi> <mi>X</mi></msub></mrow></msub> <mrow><mo>[</mo> <mi>D</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>-</mo> <msub><mi>𝔼</mi> <mrow><mi>z</mi><mo>∼</mo><msub><mi>p</mi> <mi>Z</mi></msub></mrow></msub> <mrow><mo>[</mo> <mi>D</mi> <mrow><mo>(</mo> <mi>G</mi> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>)</mo></mrow></mrow></math>

换句话说，WGAN 评论家试图最大化其对真实图像和生成图像的预测之间的差异。

为了训练 WGAN 生成器，我们计算生成图像的预测与响应之间的损失<math alttext="p Subscript i Baseline equals upper D left-parenthesis upper G left-parenthesis z Subscript i Baseline right-parenthesis right-parenthesis"><mrow><msub><mi>p</mi> <mi>i</mi></msub> <mo>=</mo> <mi>D</mi> <mrow><mo>(</mo> <mi>G</mi> <mrow><mo>(</mo> <msub><mi>z</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>)</mo></mrow></mrow></math>与响应<math alttext="y Subscript i Baseline equals 1"><mrow><msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo> <mn>1</mn></mrow></math>。因此，对于 WGAN 生成器，最小化损失函数可以写成如下形式：

<math alttext="min Underscript upper G Endscripts minus left-parenthesis double-struck upper E Subscript z tilde p Sub Subscript upper Z Subscript Baseline left-bracket upper D left-parenthesis upper G left-parenthesis z right-parenthesis right-parenthesis right-bracket right-parenthesis" display="block"><mrow><munder><mo movablelimits="true" form="prefix">min</mo> <mi>G</mi></munder> <mo>-</mo> <mrow><mo>(</mo> <msub><mi>𝔼</mi> <mrow><mi>z</mi><mo>∼</mo><msub><mi>p</mi> <mi>Z</mi></msub></mrow></msub> <mrow><mo>[</mo> <mi>D</mi> <mrow><mo>(</mo> <mi>G</mi> <mrow><mo>(</mo> <mi>z</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>)</mo></mrow></mrow></math>

换句话说，WGAN 生成器试图生成评论家尽可能高分的图像（即，评论家被欺骗以为它们是真实的）。

## 利普希茨约束

也许让你惊讶的是，我们现在允许评论家输出范围内的任何数字（<math alttext="negative normal infinity"><mrow><mo>-</mo> <mi>∞</mi></mrow></math>，<math alttext="normal infinity"><mi>∞</mi></math>），而不是应用 Sigmoid 函数将输出限制在通常的[0,1]范围内。因此，Wasserstein 损失可能非常大，这令人不安——通常情况下，神经网络中的大数值应该避免！

事实上，WGAN 论文的作者表明，为了使 Wasserstein 损失函数起作用，我们还需要对评论家施加额外的约束。具体来说，评论家必须是*1-Lipschitz 连续函数*。让我们详细解释一下这意味着什么。

评论家是一个将图像转换为预测的函数<math alttext="upper D"><mi>D</mi></math>。如果对于任意两个输入图像<math alttext="x 1"><msub><mi>x</mi> <mn>1</mn></msub></math>和<math alttext="x 2"><msub><mi>x</mi> <mn>2</mn></msub></math>，该函数满足以下不等式，则我们称该函数为 1-Lipschitz：

<math alttext="StartFraction StartAbsoluteValue upper D left-parenthesis x 1 right-parenthesis minus upper D left-parenthesis x 2 right-parenthesis EndAbsoluteValue Over StartAbsoluteValue x 1 minus x 2 EndAbsoluteValue EndFraction less-than-or-equal-to 1" display="block"><mrow><mfrac><mrow><mrow><mo>|</mo><mi>D</mi></mrow><mrow><mo>(</mo><msub><mi>x</mi> <mn>1</mn></msub> <mo>)</mo></mrow><mo>-</mo><mi>D</mi><mrow><mo>(</mo><msub><mi>x</mi> <mn>2</mn></msub> <mo>)</mo></mrow><mrow><mo>|</mo></mrow></mrow> <mrow><mrow><mo>|</mo></mrow><msub><mi>x</mi> <mn>1</mn></msub> <mo>-</mo><msub><mi>x</mi> <mn>2</mn></msub> <mrow><mo>|</mo></mrow></mrow></mfrac> <mo>≤</mo> <mn>1</mn></mrow></math>

在这里，<math alttext="StartAbsoluteValue x 1 minus x 2 EndAbsoluteValue"><mrow><mrow><mo>|</mo></mrow> <msub><mi>x</mi> <mn>1</mn></msub> <mo>-</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mrow><mo>|</mo></mrow></mrow></math>是两个图像之间的平均像素绝对差异，<math alttext="StartAbsoluteValue upper D left-parenthesis x 1 right-parenthesis minus upper D left-parenthesis x 2 right-parenthesis EndAbsoluteValue"><mrow><mrow><mo>|</mo> <mi>D</mi></mrow> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>)</mo></mrow> <mo>-</mo> <mi>D</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mo>)</mo></mrow> <mrow><mo>|</mo></mrow></mrow></math>是评论家预测之间的绝对差异。基本上，我们要求评论家的预测在两个图像之间变化的速率有限（即，梯度的绝对值必须在任何地方最多为 1）。我们可以看到这应用于利普希茨连续的一维函数中，如图 4-11 所示——在任何位置放置锥体时，线都不会进入锥体。换句话说，线在任何点上升或下降的速率都有限制。

![](img/gdl2_0411.png)

###### 图 4-11。利普希茨连续函数（来源：[维基百科](https://oreil.ly/Ki7ds)）

###### 提示

对于那些想深入了解为什么只有在强制执行这个约束时，Wasserstein 损失才有效的数学原理的人，Jonathan Hui 提供了[一个优秀的解释](https://oreil.ly/devy5)。

## 强制利普希茨约束

在原始的 WGAN 论文中，作者展示了通过在每个训练批次后将评论家的权重剪辑到一个小范围[-0.01, 0.01]内来强制执行利普希茨约束的可能性。

这种方法的批评之一是，评论家学习的能力大大降低，因为我们正在剪辑它的权重。事实上，即使在原始的 WGAN 论文中，作者们也写道，“权重剪辑显然是一种强制利普希茨约束的可怕方式。”强大的评论家对于 WGAN 的成功至关重要，因为没有准确的梯度，生成器无法学习如何调整其权重以生成更好的样本。

因此，其他研究人员寻找了其他方法来强制执行利普希茨约束，并提高 WGAN 学习复杂特征的能力。其中一种方法是带有梯度惩罚的 Wasserstein GAN。

在引入这种变体的论文中，作者展示了如何通过在评论家的损失函数中直接包含*梯度惩罚*项来强制执行利普希茨约束，如果梯度范数偏离 1，模型将受到惩罚。这导致了一个更加稳定的训练过程。

在下一节中，我们将看到如何将这个额外项构建到我们评论家的损失函数中。

## 梯度惩罚损失

图 4-12 是 WGAN-GP 评论家的训练过程的图表。如果我们将其与图 4-5 中原始鉴别器训练过程进行比较，我们可以看到的关键添加是梯度惩罚损失作为整体损失函数的一部分，与真实和虚假图像的 Wasserstein 损失一起。

![](img/gdl2_0412.png)

###### 图 4-12。WGAN-GP 评论家训练过程

梯度惩罚损失衡量了预测梯度的范数与输入图像之间的平方差异和 1 之间的差异。模型自然倾向于找到确保梯度惩罚项最小化的权重，从而鼓励模型符合利普希茨约束。

在训练过程中无法计算每个地方的梯度，因此 WGAN-GP 只在少数几个点处评估梯度。为了确保平衡混合，我们使用一组插值图像，这些图像位于连接真实图像批次和虚假图像批次的线上随机选择的点上，如图 4-13 所示。

![](img/gdl2_0413.png)

###### 图 4-13。图像之间的插值

在示例 4-8 中，我们展示了如何在代码中计算梯度惩罚。

##### 示例 4-8。梯度惩罚损失函数

```py
def gradient_penalty(self, batch_size, real_images, fake_images):
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0) ![1](img/1.png)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff ![2](img/2.png)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = self.critic(interpolated, training=True) ![3](img/3.png)

    grads = gp_tape.gradient(pred, [interpolated])[0] ![4](img/4.png)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])) ![5](img/5.png)
    gp = tf.reduce_mean((norm - 1.0) ** 2) ![6](img/6.png)
    return gp
```

![1](img/#co_generative_adversarial_networks_CO4-1)

批次中的每个图像都会得到一个介于 0 和 1 之间的随机数，存储为向量`alpha`。

![2](img/#co_generative_adversarial_networks_CO4-2)

计算一组插值图像。

![3](img/#co_generative_adversarial_networks_CO4-3)

批评家被要求对这些插值图像进行评分。

![4](img/#co_generative_adversarial_networks_CO4-4)

根据输入图像计算预测的梯度。

![5](img/#co_generative_adversarial_networks_CO4-5)

计算这个向量的 L2 范数。

![6](img/#co_generative_adversarial_networks_CO4-6)

该函数返回 L2 范数与 1 之间的平均平方距离。

## 训练 WGAN-GP

使用 Wasserstein 损失函数的一个关键优势是我们不再需要担心平衡批评家和生成器的训练——事实上，当使用 Wasserstein 损失时，必须在更新生成器之前将批评家训练到收敛，以确保生成器更新的梯度准确。这与标准 GAN 相反，标准 GAN 中重要的是不要让鉴别器变得太强。

因此，使用 Wasserstein GAN，我们可以简单地在生成器更新之间多次训练批评家，以确保它接近收敛。通常使用的比例是每次生成器更新三到五次批评家更新。

我们现在介绍了 WGAN-GP 背后的两个关键概念——Wasserstein 损失和包含在批评家损失函数中的梯度惩罚项。包含所有这些想法的 WGAN 模型的训练步骤显示在示例 4-9 中。

##### 示例 4-9。训练 WGAN-GP

```py
def train_step(self, real_images):
    batch_size = tf.shape(real_images)[0]

    for i in range(3): ![1](img/1.png)
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        with tf.GradientTape() as tape:
            fake_images = self.generator(
                random_latent_vectors, training = True
            )
            fake_predictions = self.critic(fake_images, training = True)
            real_predictions = self.critic(real_images, training = True)

            c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                real_predictions
            ) ![2](img/2.png)
            c_gp = self.gradient_penalty(
                batch_size, real_images, fake_images
            ) ![3](img/3.png)
            c_loss = c_wass_loss + c_gp * self.gp_weight ![4](img/4.png)

        c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
        self.c_optimizer.apply_gradients(
            zip(c_gradient, self.critic.trainable_variables)
        ) ![5](img/5.png)

    random_latent_vectors = tf.random.normal(
        shape=(batch_size, self.latent_dim)
    )
    with tf.GradientTape() as tape:
        fake_images = self.generator(random_latent_vectors, training=True)
        fake_predictions = self.critic(fake_images, training=True)
        g_loss = -tf.reduce_mean(fake_predictions) ![6](img/6.png)

    gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
    self.g_optimizer.apply_gradients(
        zip(gen_gradient, self.generator.trainable_variables)
    ) ![7](img/7.png)

    self.c_loss_metric.update_state(c_loss)
    self.c_wass_loss_metric.update_state(c_wass_loss)
    self.c_gp_metric.update_state(c_gp)
    self.g_loss_metric.update_state(g_loss)

    return {m.name: m.result() for m in self.metrics}
```

![1](img/#co_generative_adversarial_networks_CO5-1)

执行三次批评家更新。

![2](img/#co_generative_adversarial_networks_CO5-2)

为批评家计算 Wasserstein 损失——虚假图像和真实图像的平均预测之间的差异。

![3](img/#co_generative_adversarial_networks_CO5-3)

计算梯度惩罚项（参见示例 4-8）。

![4](img/#co_generative_adversarial_networks_CO5-4)

批评家损失函数是 Wasserstein 损失和梯度惩罚的加权和。

![5](img/#co_generative_adversarial_networks_CO5-5)

更新批评家的权重。

![6](img/#co_generative_adversarial_networks_CO5-6)

为生成器计算 Wasserstein 损失。

![7](img/#co_generative_adversarial_networks_CO5-7)

更新生成器的权重。

# WGAN-GP 中的批归一化

在训练 WGAN-GP 之前我们应该注意的最后一个考虑是批归一化不应该在批评家中使用。这是因为批归一化会在同一批次中的图像之间创建相关性，使得梯度惩罚损失效果不佳。实验证明，即使在批评家中没有批归一化，WGAN-GP 仍然可以产生出色的结果。

我们现在已经涵盖了标准 GAN 和 WGAN-GP 之间的所有关键区别。回顾一下：

+   WGAN-GP 使用 Wasserstein 损失。

+   WGAN-GP 使用标签 1 表示真实和-1 表示虚假进行训练。

+   在批评家的最后一层没有 Sigmoid 激活。

+   在评论者的损失函数中包含一个梯度惩罚项。

+   对生成器进行多次更新之前多次训练评论者。

+   评论中没有批量归一化层。

## WGAN-GP 的分析

让我们看一下生成器在训练 25 个时期后的一些示例输出（图 4-14）。

![](img/gdl2_0414.png)

###### 图 4-14\. WGAN-GP 面部示例

模型已经学会了面部的重要高级属性，没有出现模式坍塌的迹象。

我们还可以看到模型的损失函数随时间的演变（图 4-15）—评论者和生成器的损失函数都非常稳定和收敛。

如果我们将 WGAN-GP 的输出与上一章的 VAE 输出进行比较，我们可以看到 GAN 图像通常更清晰—特别是头发和背景之间的定义。这在一般情况下是正确的；VAE 倾向于产生模糊颜色边界的柔和图像，而众所周知，GAN 倾向于产生更清晰、更明确定义的图像。

![](img/gdl2_0415.png)

###### 图 4-15\. WGAN-GP 损失曲线：评论者损失（`epoch_c_loss`）分解为 Wasserstein 损失（`epoch_c_wass`）和梯度惩罚损失（`epoch_c_gp`）

同样，GAN 通常比 VAE 更难训练，并需要更长的时间达到令人满意的质量。然而，今天许多最先进的生成模型都是基于 GAN 的，因为在 GPU 上训练大规模 GAN 并花费更长时间的回报是显著的。

# 条件 GAN（CGAN）

到目前为止，在本章中，我们已经构建了能够从给定的训练集生成逼真图像的 GAN。然而，我们无法控制我们想要生成的图像类型—例如，男性或女性的面孔，或大砖或小砖。我们可以从潜在空间中随机采样一个点，但我们无法轻松地了解在选择潜在变量的情况下将产生什么样的图像。

在本章的最后部分，我们将把注意力转向构建一个能够控制输出的 GAN—所谓的*条件 GAN*。这个想法最早是在 2014 年由 Mirza 和 Osindero 在“条件生成对抗网络”中首次提出的，是对 GAN 架构的一个相对简单的扩展。

# 运行此示例的代码

此示例的代码可以在位于书籍存储库中的*notebooks/04_gan/03_cgan/cgan.ipynb*的 Jupyter 笔记本中找到。

代码已经从由 Sayak Paul 创建的优秀[CGAN 教程](https://oreil.ly/Ey11I)中调整，该教程可在 Keras 网站上找到。

## CGAN 架构

在这个例子中，我们将在面部数据集的*金发*属性上对我们的 CGAN 进行条件化。也就是说，我们可以明确指定我们是否想要生成一张有金发的图像。这个标签作为 CelebA 数据集的一部分提供。

高级 CGAN 架构如图 4-16 所示。

![](img/gdl2_0416.png)

###### 图 4-16\. CGAN 中生成器和评论者的输入和输出

标准 GAN 和 CGAN 之间的关键区别在于，在 CGAN 中，我们向生成器和评论者传递与标签相关的额外信息。在生成器中，这只是作为一个独热编码向潜在空间样本附加。在评论者中，我们将标签信息作为额外通道添加到 RGB 图像中。我们通过重复独热编码向量来填充与输入图像相同形状的方式来实现这一点。

CGAN 之所以有效是因为评论者现在可以访问有关图像内容的额外信息，因此生成器必须确保其输出与提供的标签一致，以继续愚弄评论者。如果生成器生成了与图像标签不符的完美图像，评论者将能够简单地判断它们是假的，因为图像和标签不匹配。

###### 提示

在我们的示例中，我们的独热编码标签长度为 2，因为有两个类别（金发和非金发）。但是，您可以有任意数量的标签——例如，您可以在 Fashion-MNIST 数据集上训练一个 CGAN，以输出 10 种不同的时尚物品之一，通过将长度为 10 的独热编码标签向量合并到生成器的输入中，并将 10 个额外的独热编码标签通道合并到评论家的输入中。

我们需要对架构进行的唯一更改是将标签信息连接到生成器和评论家的现有输入中，如示例 4-10 所示。

##### 示例 4-10。CGAN 中的输入层

```py
critic_input = layers.Input(shape=(64, 64, 3)) ![1](img/1.png)
label_input = layers.Input(shape=(64, 64, 2))
x = layers.Concatenate(axis = -1)([critic_input, label_input])
...
generator_input = layers.Input(shape=(32,)) ![2](img/2.png)
label_input = layers.Input(shape=(2,))
x = layers.Concatenate(axis = -1)([generator_input, label_input])
x = layers.Reshape((1,1, 34))(x)
...
```

![1](img/#co_generative_adversarial_networks_CO6-1)

图像通道和标签通道分别传递给评论家并连接。

![2](img/#co_generative_adversarial_networks_CO6-2)

潜在向量和标签类别分别传递给生成器，并在重塑之前连接。

## 训练 CGAN

我们还需要对 CGAN 的`train_step`进行一些更改，以匹配生成器和评论家的新输入格式，如示例 4-11 所示。

##### 示例 4-11。CGAN 的`train_step`

```py
def train_step(self, data):
    real_images, one_hot_labels = data ![1](img/1.png)

    image_one_hot_labels = one_hot_labels[:, None, None, :] ![2](img/2.png)
    image_one_hot_labels = tf.repeat(
        image_one_hot_labels, repeats=64, axis = 1
    )
    image_one_hot_labels = tf.repeat(
        image_one_hot_labels, repeats=64, axis = 2
    )

    batch_size = tf.shape(real_images)[0]

    for i in range(self.critic_steps):
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        with tf.GradientTape() as tape:
            fake_images = self.generator(
                [random_latent_vectors, one_hot_labels], training = True
            ) ![3](img/3.png)

            fake_predictions = self.critic(
                [fake_images, image_one_hot_labels], training = True
            ) ![4](img/4.png)
            real_predictions = self.critic(
                [real_images, image_one_hot_labels], training = True
            )

            c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                real_predictions
            )
            c_gp = self.gradient_penalty(
                batch_size, real_images, fake_images, image_one_hot_labels
            ) ![5](img/5.png)
            c_loss = c_wass_loss + c_gp * self.gp_weight

        c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
        self.c_optimizer.apply_gradients(
            zip(c_gradient, self.critic.trainable_variables)
        )

    random_latent_vectors = tf.random.normal(
        shape=(batch_size, self.latent_dim)
    )

    with tf.GradientTape() as tape:
        fake_images = self.generator(
            [random_latent_vectors, one_hot_labels], training=True
        ) ![6](img/6.png)
        fake_predictions = self.critic(
            [fake_images, image_one_hot_labels], training=True
        )
        g_loss = -tf.reduce_mean(fake_predictions)

    gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
    self.g_optimizer.apply_gradients(
        zip(gen_gradient, self.generator.trainable_variables)
    )
```

![1](img/#co_generative_adversarial_networks_CO7-1)

图像和标签从输入数据中解压缩。

![2](img/#co_generative_adversarial_networks_CO7-2)

独热编码向量被扩展为具有与输入图像相同空间大小（64×64）的独热编码图像。

![3](img/#co_generative_adversarial_networks_CO7-3)

现在，生成器被提供了两个输入的列表——随机潜在向量和独热编码标签向量。

![4](img/#co_generative_adversarial_networks_CO7-4)

评论家现在被提供了两个输入的列表——假/真实图像和独热编码标签通道。

![5](img/#co_generative_adversarial_networks_CO7-5)

梯度惩罚函数还需要将独热编码标签通道传递给评论家，因为它使用评论家。

![6](img/#co_generative_adversarial_networks_CO7-6)

对评论家培训步骤所做的更改也适用于生成器训练步骤。

## CGAN 的分析

我们可以通过将特定的独热编码标签传递到生成器的输入来控制 CGAN 的输出。例如，要生成一个头发不是金色的脸，我们传入向量`[1, 0]`。要生成一个金发的脸，我们传入向量`[0, 1]`。

可以在图 4-17 中看到 CGAN 的输出。在这里，我们保持示例中的随机潜在向量相同，只改变条件标签向量。很明显，CGAN 已经学会使用标签向量来控制图像的头发颜色属性。令人印象深刻的是，图像的其余部分几乎没有改变——这证明了 GAN 能够以这样一种方式组织潜在空间中的点，以便将各个特征解耦。

![](img/gdl2_0417.png)

###### 图 4-17。当*Blond*和*Not Blond*向量附加到潜在样本时，CGAN 的输出

###### 提示

如果您的数据集中有标签，通常最好将它们包含在 GAN 的输入中，即使您不一定需要将生成的输出条件化为标签，因为它们往往会提高生成的图像质量。您可以将标签视为像素输入的高度信息性扩展。

# 总结

在本章中，我们探讨了三种不同的生成对抗网络（GAN）模型：深度卷积 GAN（DCGAN）、更复杂的带有梯度惩罚的 Wasserstein GAN（WGAN-GP）和条件 GAN（CGAN）。

所有 GAN 都以生成器与鉴别器（或评论家）架构为特征，鉴别器试图“发现”真假图像之间的差异，生成器旨在欺骗鉴别器。通过平衡这两个对手的训练方式，GAN 生成器可以逐渐学习如何产生与训练集中的观察结果相似的图像。

我们首先看到如何训练 DCGAN 生成玩具积木的图像。它能够学习如何以图像形式真实地表示 3D 物体，包括阴影、形状和纹理的准确表示。我们还探讨了 GAN 训练可能失败的不同方式，包括模式坍塌和梯度消失。

然后，我们探讨了 Wasserstein 损失函数如何纠正了许多问题，并使 GAN 训练更加可预测和可靠。WGAN-GP 通过在损失函数中包含一个术语来将 1-Lipschitz 要求置于训练过程的核心，以将梯度范数拉向 1。

我们将 WGAN-GP 应用于人脸生成问题，并看到通过简单地从标准正态分布中选择点，我们可以生成新的人脸。这个采样过程与 VAE 非常相似，尽管 GAN 生成的人脸通常更加清晰，图像的不同部分之间的区别更大。

最后，我们构建了一个 CGAN，使我们能够控制生成的图像类型。这通过将标签作为输入传递给评论家和生成器来实现，从而为网络提供了所需的额外信息，以便根据给定的标签对生成的输出进行条件化。

总的来说，我们已经看到 GAN 框架非常灵活，能够适应许多有趣的问题领域。特别是，GAN 已经在图像生成领域取得了显著进展，有许多有趣的扩展到基础框架中，我们将在第十章中看到。

在下一章中，我们将探讨一种适合建模序列数据的不同生成模型家族——自回归模型。

^(1) Ian J. Goodfellow 等人，“生成对抗网络”，2014 年 6 月 10 日，[*https://arxiv.org/abs/1406.2661*](https://arxiv.org/abs/1406.2661)

^(2) Alec Radford 等人，“使用深度卷积生成对抗网络进行无监督表示学习”，2016 年 1 月 7 日，[*https://arxiv.org/abs/1511.06434*](https://arxiv.org/abs/1511.06434)。

^(3) Augustus Odena 等人，“反卷积和棋盘伪影”，2016 年 10 月 17 日，[*https://distill.pub/2016/deconv-checkerboard*](https://distill.pub/2016/deconv-checkerboard)。

^(4) Martin Arjovsky 等人，“Wasserstein GAN”，2017 年 1 月 26 日，[*https://arxiv.org/abs/1701.07875*](https://arxiv.org/abs/1701.07875)。

^(5) Ishaan Gulrajani 等人，“改进的 Wasserstein GANs 训练”，2017 年 3 月 31 日，[*https://arxiv.org/abs/1704.00028*](https://arxiv.org/abs/1704.00028)。

^(6) Mehdi Mirza 和 Simon Osindero，“条件生成对抗网络”，2014 年 11 月 6 日，[*https://arxiv.org/abs/1411.1784*](https://arxiv.org/abs/1411.1784)。
