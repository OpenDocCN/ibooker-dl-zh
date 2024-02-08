# 第十三章。使用自动编码器生成图像

在第五章中，我们探讨了如何生成文本，以某个现有语料库的风格为基础，无论是莎士比亚的作品还是 Python 标准库中的代码，而在第十二章中，我们研究了通过优化预训练网络中通道的激活来生成图像。在本章中，我们将结合这些技术并在其基础上生成基于示例的图像。

基于示例生成图像是一个活跃研究领域，新的想法和突破性进展每月都有报道。然而，目前最先进的算法在模型复杂性、训练时间和所需数据方面超出了本书的范围。相反，我们将在一个相对受限的领域进行工作：手绘草图。

我们将从查看谷歌的 Quick Draw 数据集开始。这是一个在线绘图游戏的结果，包含许多手绘图片。这些绘画以矢量格式存储，因此我们将把它们转换为位图。我们将选择带有一个标签的草图：猫。

基于这些猫的草图，我们将构建一个自动编码器模型，能够学习*猫的特征*——它可以将猫的绘图转换为内部表示，然后从该内部表示生成类似的东西。我们将首先查看这个网络在我们的猫上的性能可视化。

然后我们将切换到手绘数字的数据集，然后转向*变分自动编码器*。这些网络生成密集空间，是它们输入的抽象表示，我们可以从中进行采样。每个样本将产生一个看起来逼真的图像。我们甚至可以在点之间进行插值，看看图像是如何逐渐变化的。

最后，我们将看看*条件变分自动编码器*，在训练时考虑标签，因此可以以随机方式再现某一类别的图像。

与本章相关的代码可以在以下笔记本中找到：

```py
13.1 Quick Draw Cat Autoencoder
13.2 Variational Autoencoder
```

# 13.1 从 Google Quick Draw 导入绘图

## 问题

你在哪里可以获得一组日常手绘图像？

## 解决方案

使用谷歌 Quick Draw 的数据集。

[Google Quick Draw](https://quickdraw.withgoogle.com/)是一个在线游戏，用户被挑战绘制某物，并查看 AI 是否能猜出他们试图创建的内容。这个游戏很有趣，作为一个副产品，产生了一个大量带标签的绘画数据库。谷歌已经让任何想玩机器学习的人都可以访问这个数据集。

数据可在[多种格式](https://github.com/googlecreativelab/quickdraw-dataset)中获得。我们将使用简化的矢量绘图的二进制编码版本。让我们开始获取所有的猫：

```py
BASE_PATH = 'https://storage.googleapis.com/quickdraw_dataset/full/binary/
path = get_file('cat', BASE_PATH + 'cat.bin')
```

我们将逐个解压这些图像。它们以二进制矢量格式存储，我们将在一个空位图上绘制。这些绘画以 15 字节的头部开始，因此我们只需继续处理，直到我们的文件不再至少有 15 字节为止：

```py
x = []
with open(path, 'rb') as f:
    while True:
        img = PIL.Image.new('L', (32, 32), 'white')
        draw = ImageDraw.Draw(img)
        header = f.read(15)
        if len(header) != 15:
            break
```

一幅图是一系列笔画的列表，每个笔画由一系列*x*和*y*坐标组成。*x*和*y*坐标分开存储，因此我们需要将它们压缩成一个列表，以便输入到我们刚刚创建的`ImageDraw`对象中：

```py
            strokes, = unpack('H', f.read(2))
            for i in range(strokes):
                n_points, = unpack('H', f.read(2))
                fmt = str(n_points) + 'B'
                read_scaled = lambda: (p // 8 for
                                       p in unpack(fmt, f.read(n_points)))
                points = [*zip(read_scaled(), read_scaled())]
                draw.line(points, fill=0, width=2)
            img = img_to_array(img)
            x.append(img)
```

超过十万幅猫的绘画属于你。

## 讨论

通过游戏收集用户生成的数据是建立机器学习数据集的一种有趣方式。这不是谷歌第一次使用这种技术——几年前，它运行了[Google Image Labeler 游戏](http://bit.ly/wiki-gil)，两个不相识的玩家会为图像打标签，并根据匹配的标签获得积分。然而，该游戏的结果从未向公众公开。

数据集中有 345 个类别。在本章中，我们只使用了猫，但您可以尝试其余的类别来构建图像分类器。数据集存在缺点，其中最主要的是并非所有的绘画都是完成的；当 AI 识别出绘画时，游戏就结束了，对于一幅骆驼画来说，画两个驼峰可能就足够了。

###### 注意

在这个示例中，我们自己对图像进行了光栅化处理。Google 确实提供了一个`numpy`数组版本的数据，其中图像已经被预先光栅化为 28×28 像素。

# 13.2 创建图像的自动编码器

## 问题

即使没有标签，是否可以自动将图像表示为固定大小的向量？

## 解决方案

使用自动编码器。

在第九章中，我们看到我们可以使用卷积网络通过连续的层从像素到局部特征再到更多结构特征最终到图像的抽象表示来对图像进行分类，然后我们可以使用这个抽象表示来预测图像的内容。在第十章中，我们将图像的抽象表示解释为高维语义空间中的向量，并利用接近的向量表示相似的图像作为构建反向图像搜索引擎的一种方式。最后，在第十二章中，我们看到我们可以可视化卷积网络中不同层级的各种神经元的激活意味着什么。

为了做到这一点，我们需要对图像进行标记。只有因为网络看到了大量的狗、猫和许多其他东西，它才能在这个高维空间中学习这些东西的抽象表示。如果我们的图像没有标签怎么办？或者标签不足以让网络形成对事物的直觉？在这些情况下，自动编码器可以提供帮助。

自动编码器背后的想法是强制网络将图像表示为具有特定大小的向量，并且基于网络能够从该表示中准确复制输入图像的损失函数。输入和期望输出是相同的，这意味着我们不需要标记的图像。任何一组图像都可以。

网络的结构与我们之前看到的非常相似；我们取原始图像并使用一系列卷积层和池化层来减小大小并增加深度，直到我们有一个一维向量，这是该图像的抽象表示。但是，我们不会就此罢手并使用该向量来预测图像是什么，而是跟进并通过一组*上采样*层从图像的抽象表示开始，进行相反操作，直到我们再次得到一个图像。作为我们的损失函数，我们然后取输入和输出图像之间的差异：

```py
def create_autoencoder():
    input_img = Input(shape=(32, 32, 1))

    channels = 2
    x = input_img
    for i in range(4):
        channels *= 2
        left = Conv2D(channels, (3, 3),
                      activation='relu', padding='same')(x)
        right = Conv2D(channels, (2, 2),
                       activation='relu', padding='same')(x)
        conc = Concatenate()([left, right])
        x = MaxPooling2D((2, 2), padding='same')(conc)

    x = Dense(channels)(x)

    for i in range(4):
        x = Conv2D(channels, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        channels //= 2
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder

autoencoder = create_autoencoder()
autoencoder.summary()
```

我们可以将网络架构想象成一个沙漏。顶部和底部层代表图像。网络中最小的点位于中间，并经常被称为*潜在表示*。我们在这里有一个具有 128 个条目的潜在空间，这意味着我们强制网络使用 128 个浮点数来表示每个 32×32 像素的图像。网络能够最小化输入和输出图像之间的差异的唯一方法是尽可能多地将信息压缩到潜在表示中。

我们可以像以前一样训练网络：

```py
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                validation_data=(x_test, x_test))
```

这应该相当快地收敛。

## 讨论

自动编码器是一种有趣的神经网络类型，因为它们能够在没有任何监督的情况下学习其输入的紧凑、有损表示。在这个示例中，我们已经将它们用于图像，但它们也成功地被部署来处理文本或其他形式的时间序列数据。

自动编码器思想有许多有趣的扩展。其中之一是*去噪*自动编码器。这里的想法是要求网络不是从自身预测目标图像，而是从自身的损坏版本预测目标图像。例如，我们可以向输入图像添加一些随机噪声。损失函数仍然会将网络的输出与原始（非加噪）输入进行比较，因此网络将有效地学习如何从图片中去除噪声。在其他实验中，这种技术在恢复黑白图片的颜色方面被证明是有用的。

我们在第十章中使用图像的抽象表示来创建一个反向图像搜索引擎，但我们需要标签。使用自动编码器，我们不需要这些标签；我们可以在模型仅训练了一组图像之后测量图像之间的距离。事实证明，如果我们使用去噪自动编码器，我们的图像相似性算法的性能会提高。这里的直觉是噪声告诉网络不要注意的内容，类似于数据增强的工作方式（参见“图像的预处理”）。

# 13.3 可视化自动编码器结果

## 问题

您想要了解您的自动编码器的工作情况如何。

## 解决方案

从输入中随机抽取几张猫的图片，并让模型预测这些图片；然后将输入和输出呈现为两行。

让我们预测一些猫：

```py
cols = 25
idx = np.random.randint(x_test.shape[0], size=cols)
sample = x_test[idx]
decoded_imgs = autoencoder.predict(sample)
```

并在我们的笔记本中展示它们：

```py
def decode_img(tile):
    tile = tile.reshape(tile.shape[:-1])
    tile = np.clip(tile * 400, 0, 255)
    return PIL.Image.fromarray(tile)

overview = PIL.Image.new('RGB', (cols * 32, 64 + 20), (128, 128, 128))
for idx in range(cols):
    overview.paste(decode_img(sample[idx]), (idx * 32, 5))
    overview.paste(decode_img(decoded_imgs[idx]), (idx * 32, 42))
f = BytesIO()
overview.save(f, 'png')
display(Image(data=f.getvalue()))
```

![一排猫](img/dlcb_13in01.png)

正如您所看到的，网络确实捕捉到了基本形状，但似乎对自己不太确定，这导致模糊的图标绘制，几乎像阴影一样。

在下一个步骤中，我们将看看是否可以做得更好。

## 讨论

由于自动编码器的输入和输出*应该*是相似的，检查网络性能的最佳方法就是从验证集中随机选择一些图标，并要求网络对其进行重建。使用 PIL 创建一个显示两行图像并在 Jupyter 笔记本中显示的图像是我们之前见过的。

这里的一个问题是我们使用的损失函数会导致网络模糊其输出。输入图纸包含细线，但我们模型的输出却没有。我们的模型没有动力预测清晰的线条，因为它不确定线条的确切位置，所以它宁愿分散其赌注并绘制模糊的线条。这样至少有一些像素会被命中的机会很高。为了改进这一点，我们可以尝试设计一个损失函数，强制网络限制它绘制的像素数量，或者对清晰的线条加以奖励。

# 13.4 从正确分布中抽样图像

## 问题

如何确保向量中的每个点都代表一个合理的图像？

## 解决方案

使用*变分*自动编码器。

自动编码器作为一种将图像表示为比图像本身小得多的向量的方式非常有趣。但是，这些向量的空间并不是*密集*的；也就是说，每个图像在该空间中都有一个向量，但并非该空间中的每个向量都代表一个合理的图像。自动编码器的解码器部分当然会根据任何向量创建一个图像，但其中大多数图像都不会被识别。变分自动编码器确实具有这种属性。

在本章和接下来的配方中，我们将使用手写数字的 MNIST 数据集，包括 60000 个训练样本和 10000 个测试样本。这里描述的方法适用于图标，但会使模型复杂化，为了良好的性能，我们需要比现有的图标更多。如果您感兴趣，笔记本目录中有一个可用的模型。让我们从加载数据开始：

```py
def prepare(images, labels):
    images = images.astype('float32') / 255
    n, w, h = images.shape
    return images.reshape((n, w * h)), to_categorical(labels)

train, test = mnist.load_data()
x_train, y_train = prepare(*train)
x_test, y_test = prepare(*test)
img_width, img_height = train[0].shape[1:]
```

变分自动编码器背后的关键思想是在损失函数中添加一个项，表示图像和抽象表示之间的统计分布差异。为此，我们将使用 Kullback-Leibler 散度。我们可以将其视为概率分布空间的距离度量，尽管从技术上讲它不是距离度量。[维基百科文章](http://bit.ly/k-l-d)中有详细信息，供那些想要了解数学知识的人阅读。

我们模型的基本轮廓与上一个示例类似。我们从代表我们像素的输入开始，将其通过一些隐藏层，然后将其采样到一个非常小的表示。然后我们再次逐步提升，直到我们恢复我们的像素：

```py
pixels = Input(shape=(num_pixels,))
encoder_hidden = Dense(512, activation='relu')(pixels)
z_mean = Dense(latent_space_depth,
               activation='linear')(encoder_hidden)
z_log_var = Dense(latent_space_depth,
                  activation='linear')(encoder_hidden)
z = Lambda(sample_z, output_shape=(latent_space_depth,))(
        [z_mean, z_log_var])
decoder_hidden = Dense(512, activation='relu')
reconstruct_pixels = Dense(num_pixels, activation='sigmoid')
hidden = decoder_hidden(z)
outputs = reconstruct_pixels(hidden)
auto_encoder = Model(pixels, outputs)
```

这里有趣的部分是`z`张量和它被分配给的`Lambda`。这个张量将保存我们图像的潜在表示，而`Lambda`使用`sample_z`方法进行采样：

```py
def sample_z(args):
    z_mean, z_log_var = args
    eps = K.random_normal(shape=(batch_size, latent_space_depth),
                         mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * eps
```

这是我们使用两个变量`z_mean`和`z_log_var`从正态分布中随机采样点的地方。

现在让我们来看看我们的损失函数。第一个组件是重构损失，它衡量了输入像素和输出像素之间的差异：

```py
def reconstruction_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
```

我们需要的第二件事是在我们的损失函数中添加一个使用 Kullback-Leibler 散度来引导分布走向正确方向的组件：

```py
def KL_loss(y_true, y_pred):
    return 0.5 * K.sum(K.exp(z_log_var) +
                       K.square(z_mean) - 1 - z_log_var,
                       axis=1)
```

然后我们简单地将其相加：

```py
def total_loss(y_true, y_pred):
    return (KL_loss(y_true, y_pred) +
            reconstruction_loss(y_true, y_pred))
```

我们可以用以下方式编译我们的模型：

```py
auto_encoder.compile(optimizer=Adam(lr=0.001),
                     loss=total_loss,
                     metrics=[KL_loss, reconstruction_loss])
```

这也会方便地跟踪训练过程中损失的各个组件。

由于额外的损失函数和对`sample_z`的带外调用，这个模型稍微复杂；要查看详细信息，最好在相应的笔记本中查看。现在我们可以像以前一样训练模型：

```py
cvae.fit(x_train, x_train, verbose = 1, batch_size=batch_size, epochs=50,
         validation_data = (x_test, x_test))
```

一旦训练完成，我们希望通过在潜在空间中提供一个随机点并查看出现的图像来使用结果。我们可以通过创建一个第二个模型，其输入是我们`auto_encoder`模型的中间层，输出是我们的目标图像来实现这一点：

```py
decoder_in = Input(shape=(latent_space_depth,))
decoder_hidden = decoder_hidden(decoder_in)
decoder_out = reconstruct_pixels(decoder_hidden)
decoder = Model(decoder_in, decoder_out)
```

现在我们可以生成一个随机输入，然后将其转换为一幅图片：

```py
random_number = np.asarray([[np.random.normal()
                            for _ in range(latent_space_depth)]])
def decode_img(a):
    a = np.clip(a * 256, 0, 255).astype('uint8')
    return PIL.Image.fromarray(a)

decode_img(decoder.predict(random_number)
               .reshape(img_width, img_height)).resize((56, 56))
```

![随机生成的数字](img/dlcb_13in02.png)

## 讨论

当涉及生成图像而不仅仅是复制图像时，变分自动编码器为自动编码器添加了一个重要组件；通过确保图像的抽象表示来自一个“密集”空间，其中靠近原点的点映射到可能的图像，我们现在可以生成具有与我们输入相同的可能性分布的图像。

基础数学知识超出了本书的范围。这里的直觉是一些图像更“正常”，一些更意外。潜在空间具有相同的特征，因此从原点附近绘制的点对应于“正常”的图像，而更极端的点则对应于更不太可能的图像。从正态分布中采样将导致生成具有与模型训练期间看到的预期和意外图像混合的图像。

拥有密集空间很好。它允许我们在点之间进行插值，并仍然获得有效的结果。例如，如果我们知道潜在空间中的一个点映射到 6，另一个点映射到 8，我们期望在两者之间的点会产生从 6 到 8 的图像。如果我们找到相同的图像但是以不同的风格，我们可以寻找混合风格的中间图像。或者我们甚至可以朝着另一个方向前进，并期望找到更极端的风格。

在第三章中，我们看过了词嵌入，其中每个单词都有一个将其投影到语义空间的向量，以及我们可以对其进行的计算。尽管这很有趣，但由于空间不是密集的，我们通常不会期望在两个单词之间找到某种折中的东西——在 *驴* 和 *马* 之间没有 *骡子*。同样，我们可以使用预训练的图像识别网络为一张猫的图片找到一个向量，但其周围的向量并不都代表猫的变化。

# 13.5 可视化变分自动编码器空间

## 问题

如何可视化您可以从潜在空间生成的图像的多样性？

## 解决方案

使用潜在空间中的两个维度创建一个生成图像的网格。

从我们的潜在空间中可视化两个维度是直接的。对于更高的维度，我们可以首先尝试 t-SNE 将其降至两个维度。幸运的是，在前一个示例中我们只使用了两个维度，所以我们可以通过一个平面并将每个 (*x*, *y*) 位置映射到潜在空间中的一个点。由于我们使用正态分布，我们期望合理的图像出现在 [-1.5, 1.5] 范围内：

```py
num_cells = 10
overview = PIL.Image.new('RGB',
                         (num_cells * (img_width + 4) + 8,
                          num_cells * (img_height + 4) + 8),
                         (128, 128, 128))
vec = np.zeros((1, latent_space_depth))
for x in range(num_cells):
    vec[:, 0] = (x * 3) / (num_cells - 1) - 1.5
    for y in range(num_cells):
        vec[:, 1] = (y * 3) / (num_cells - 1) - 1.5
        decoded = decoder.predict(vec)
        img = decode_img(decoded.reshape(img_width, img_height))
        overview.paste(img, (x * (img_width + 4) + 6,
                             y * (img_height + 4) + 6))
overview
```

这将为我们提供网络学习的不同数字的漂亮图像：

![随机生成的网格](img/dlcb_13in03.png)

## 讨论

通过将 (*x*, *y*) 映射到我们的潜在空间并将结果解码为图像，我们可以很好地了解我们的空间包含的内容。正如我们所看到的，空间确实相当密集。并非所有点都会产生数字；一些点，如预期的那样，代表中间形式。但模型确实找到了一种在网格上以自然方式分布数字的方法。

这里要注意的另一件事是，我们的变分自动编码器在压缩图像方面表现出色。每个输入图像仅由 2 个浮点数在潜在空间中表示，而它们的像素表示使用 28 × 28 = 784 个浮点数。这是接近 400 的压缩比，远远超过了 JPEG。当然，这种压缩是有损的——一个手写的 5 在编码和解码后仍然看起来像一个手写的 5，仍然是同一风格，但在像素级别上没有真正的对应。此外，这种形式的压缩是极其领域特定的。它只适用于手写数字，而 JPEG 可用于压缩各种图像和照片。

# 13.6 条件变分自动编码器

## 问题

如何生成特定类型的图像而不是完全随机的图像？

## 解决方案

使用条件变分自动编码器。

前两个示例中的自动编码器很好地生成了随机数字，还能够接收一个数字并将其编码为一个漂亮、密集的潜在空间。但它无法区分 5 和 3，所以我们唯一能让它生成一个随机的 3 的方法是首先找到潜在空间中的所有 3，然后从该子空间中进行采样。条件变分自动编码器通过将标签作为输入并将标签连接到模型中的潜在空间向量 `z` 来帮助这里。

这样做有两个作用。首先，它让模型在学习编码时考虑实际标签。其次，由于它将标签添加到潜在空间中，我们的解码器现在将同时接收潜在空间中的一个点和一个标签，这使我们能够明确要求生成特定的数字。模型现在看起来像这样：

```py
pixels = Input(shape=(num_pixels,))
label = Input(shape=(num_labels,), name='label')
inputs = concat([pixels, label], name='inputs')

encoder_hidden = Dense(512, activation='relu',
                       name='encoder_hidden')(inputs)
z_mean = Dense(latent_space_depth,
               activation='linear')(encoder_hidden)
z_log_var = Dense(latent_space_depth,
                  activation='linear')(encoder_hidden)
z = Lambda(sample_z,
           output_shape=(latent_space_depth, ))([z_mean, z_log_var])
zc = concat([z, label])

decoder_hidden = Dense(512, activation='relu')
reconstruct_pixels = Dense(num_pixels, activation='sigmoid')
decoder_in = Input(shape=(latent_space_depth + num_labels,))
hidden = decoder_hidden(decoder_in)
decoder_out = reconstruct_pixels(hidden)
decoder = Model(decoder_in, decoder_out)

hidden = decoder_hidden(zc)
outputs = reconstruct_pixels(hidden)
cond_auto_encoder = Model([pixels, label], outputs)
```

我们通过向模型提供图像和标签来训练模型：

```py
cond_auto_encoder.fit([x_train, y_train], x_train, verbose=1,
                      batch_size=batch_size, epochs=50,
                      validation_data = ([x_test, y_test], x_test))
```

现在我们可以生成一个明确的数字 4：

```py
number_4 = np.zeros((1, latent_space_depth + y_train.shape[1]))
number_4[:, 4 + latent_space_depth] = 1
decode_img(cond_decoder.predict(number_4).reshape(img_width, img_height))
```

![数字四](img/dlcb_13in04.png)

由于我们指定了要生成的数字的一位有效编码，我们也可以要求在两个数字之间生成某些东西：

```py
number_8_3 = np.zeros((1, latent_space_depth + y_train.shape[1]))
number_8_3[:, 8 + latent_space_depth] = 0.5
number_8_3[:, 3 + latent_space_depth] = 0.5
decode_img(cond_decoder.predict(number_8_3).reshape(
    img_width, img_height))
```

这确实产生了介于两者之间的东西：

![数字八或三](img/dlcb_13in05.png)

另一个有趣的尝试是将数字放在 *y* 轴上，使用 *x* 轴来选择我们的潜在维度之一的值：

```py
num_cells = 10
overview = PIL.Image.new('RGB',
                         (num_cells * (img_width + 4) + 8,
                          num_cells * (img_height + 4) + 8),
                         (128, 128, 128))
img_it = 0
vec = np.zeros((1, latent_space_depth + y_train.shape[1]))
for x in range(num_cells):
    vec = np.zeros((1, latent_space_depth + y_train.shape[1]))
    vec[:, x + latent_space_depth] = 1
    for y in range(num_cells):
        vec[:, 1] = 3 * y / (num_cells - 1) - 1.5
        decoded = cond_decoder.predict(vec)
        img = decode_img(decoded.reshape(img_width, img_height))
        overview.paste(img, (x * (img_width + 4) + 6,
                             y * (img_height + 4) + 6))
overview
```

![风格和数字](img/dlcb_13in06.png)

正如您所看到的，潜在空间表达了数字的风格，而这种风格在数字之间是一致的。在这种情况下，它似乎控制了数字倾斜的程度。

## 讨论

条件变分自动编码器标志着我们穿越各种自动编码器的旅程的最终站。这种类型的网络使我们能够将我们的数字映射到一个稠密的潜在空间，该空间也带有标签，使我们能够在指定图像类型的同时对随机图像进行采样。

向网络提供标签的副作用是，它现在不再需要学习数字，而是可以专注于数字的风格。
