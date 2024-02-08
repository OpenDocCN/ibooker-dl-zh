# 第十四章。使用深度网络生成图标

在上一章中，我们看了一下从 Quick Draw 项目生成手绘草图和从 MNIST 数据集生成数字。在本章中，我们将尝试三种类型的网络来完成一个稍微具有挑战性的任务：生成图标。

在我们进行任何生成之前，我们需要获取一组图标。在线搜索“免费图标”会得到很多结果。这些都不是“言论自由”的，大多数在“免费啤酒”方面都有困难。此外，您不能自由重复使用这些图标，通常网站强烈建议您最终付费。因此，我们将从如何下载、提取和处理图标开始，将它们转换为我们可以在本章的其余部分中使用的标准格式。

我们将尝试的第一件事是在我们的图标集上训练一个条件变分自动编码器。我们将使用上一章中得到的网络作为基础，但我们将添加一些卷积层以使其表现更好，因为图标空间比手绘数字的空间复杂得多。

我们将尝试的第二种类型的网络是生成对抗网络。在这里，我们将训练两个网络，一个用于生成图标，另一个用于区分生成的图标和真实的图标。两者之间的竞争会带来更好的结果。

我们将尝试的第三种也是最后一种类型的网络是 RNN。在第五章中，我们使用它来生成特定风格的文本。通过将图标重新解释为一组绘图指令，我们可以使用相同的技术来生成图像。

本章相关的代码可以在以下笔记本中找到：

```py
14.1 Importing Icons
14.2 Icon Autoencoding
14.3 Icon GAN
14.4 Icon RNN
```

# 14.1 获取训练图标

## 问题

如何获取标准格式的大量图标？

## 解决方案

从 Mac 应用程序*Icons8*中提取它们。

Icons8 分发了一个庞大的图标集——超过 63,000 个。这部分是因为不同格式的图标被计为两倍，但仍然是一个不错的集合。不幸的是，这些图标分布在 Mac 和 Windows 的应用程序中。好消息是 Mac 的*.dmg*存档实际上只是一个包含应用程序的 p7zip 存档，而应用程序本身也是一个 p7zip 存档。让我们从下载应用程序开始。转到[*https://icons8.com/app*](https://icons8.com/app)并确保下载 Mac 版本（即使您在 Linux 或 Windows 上也是如此）。现在为您喜欢的操作系统安装 p7zip 的命令行版本，并将*.dmg*文件的内容提取到自己的文件夹中：

```py
7z x Icons8App_for_Mac_OS.dmg
```

*.dmg*包含一些元信息和 Mac 应用程序。让我们也解压缩应用程序：

```py
cd Icons8\ v5.6.3
7z x Icons8.app
```

就像洋葱一样，这个东西有很多层。现在您应该看到一个也需要解压缩的*.tar*文件：

```py
tar xvf icons.tar
```

这给我们一个名为*icons*的目录，其中包含一个*.ldb*文件，这表明该目录代表一个 LevelDB 数据库。切换到 Python，我们可以查看其中的内容：

```py
# Adjust to your local path:
path = '/some/path/Downloads/Icons8 v5.6.3/icons'
db = plyvel.DB(path)

for key, value in db:
    print(key)
    print(value[:400])
    break
```

```py
> b'icon_1'
b'TSAF\x03\x00\x02\x00\x07\x00\x00\x00\x00\x00\x00\x00$\x00\x00\x00
\x18\x00\x00\x00\r\x00\x00\x00-\x08id\x00\x08Messaging\x00\x08categ
ory\x00\x19\x00\x03\x00\x00\x00\x08Business\x00\x05\x01\x08User
Interface\x00\x08categories\x00\x18\x00\x00\x00\x03\x00\x00\x00\x08
Basic Elements\x00\x05\x04\x01\x05\x01\x08Business
Communication\x00\x05\x03\x08subcategories\x00\x19\x00\r\x00\x00\x00
\x08contacts\x00\x08phone book\x00\x08contacts
book\x00\x08directory\x00\x08mail\x00\x08profile\x00\x08online\x00
\x08email\x00\x08records\x00\x08alphabetical\x00\x08sim\x00\x08phone
numbers\x00\x08categorization\x00\x08tags\x00\x0f9\x08popularity\x00
\x18\x00\x00\x02\x00\x00\x00\x1c\x00\x00\x00\xe8\x0f\x00\x00<?xml
version="1.0" encoding="utf-8"?>\n<!-- Generato'
```

太棒了。我们已经找到了我们的图标，它们似乎是使用*.svg*矢量格式编码的。看起来它们包含在另一种格式中，带有`TSAF`头。在线阅读，似乎是一些与 IBM 相关的格式，但是很难找到一个 Python 库来从中提取数据。再说一遍，这个简单的转储表明我们正在处理由`\x00`分隔的键/值对，键和值由`\x08`分隔。它并不完全奏效，但足以构建一个 hacky 解析器：

```py
splitter = re.compile(b'[\x00-\x09]')

def parse_value(value):
    res = {}
    prev = ''
    for elem in splitter.split(value):
        if not elem:
            continue
        try:
            elem = elem.decode('utf8')
        except UnicodeDecodeError:
            continue
        if elem in ('category', 'name', 'platform',
                    'canonical_name', 'svg', 'svg.simplified'):
            res[elem] = prev
        prev = elem
    return res
```

这会提取 SVG 文件和一些可能在以后有用的基本属性。各个平台包含更多或更少相同的图标，因此我们需要选择一个平台。iOS 似乎拥有最多的图标，所以让我们选择它：

```py
icons = {}
for _, value in db:
    res = parse_value(value)
    if res.get('platform') == 'ios':
        name = res.get('name')
        if not name:
            name = res.get('canonical_name')
            if not name:
                continue
            name = name.lower().replace(' ', '_')
        icons[name] = res
```

现在让我们将所有内容写入磁盘以供以后处理。我们将保留 SVG 文件，同时也将位图写出为 PNG 文件：

```py
saved = []
for icon in icons.values():
    icon = dict(icon)
    if not 'svg' in icon:
        continue
    svg = icon.pop('svg')
    try:
        drawing = svg2rlg(BytesIO(svg.encode('utf8')))
    except ValueError:
        continue
    except AttributeError:
        continue
    open('icons/svg/%s.svg' % icon['name'], 'w').write(svg)
    p = renderPM.drawToPIL(drawing)
    for size in SIZES:
        resized = p.resize((size, size), Image.ANTIALIAS)
        resized.save('icons/png%s/%s.png' % (size, icon['name']))
    saved.append(icon)
json.dump(saved, open('icons/index.json', 'w'), indent=2)
```

## 讨论

尽管有许多在线广告免费图标的网站，但实际上获得一个好的训练集是相当复杂的。在这种情况下，我们在一个神秘的`TSAF`商店内找到了 SVG 格式的图标，这些图标存储在一个 LevelDB 数据库内的 Mac 应用程序内，而这个 Mac 应用程序存储在我们下载的*.dmg*文件中。一方面，这似乎比应该的要复杂。另一方面，这表明通过一点侦探工作，我们可以发现一些非常有趣的数据集。

# 14.2 将图标转换为张量表示

## 问题

如何将保存的图标转换为适合训练网络的格式？

## 解决方案

连接它们并对它们进行归一化。

这与我们为预训练网络处理图像的方式类似，只是现在我们将训练自己的网络。我们知道所有图像都将是 32×32 像素，我们将跟踪均值和标准差，以便正确地对图像进行归一化和反归一化。我们还将数据分成训练集和测试集：

```py
def load_icons(train_size=0.85):
    icon_index = json.load(open('icons/index.json'))
    x = []
    img_rows, img_cols = 32, 32
    for icon in icon_index:
        if icon['name'].endswith('_filled'):
            continue
        img_path = 'icons/png32/%s.png' % icon['name']
        img = load_img(img_path, grayscale=True,
                       target_size=(img_rows, img_cols))
        img = img_to_array(img)
        x.append(img)
    x = np.asarray(x) / 255
    x_train, x_val = train_test_split(x, train_size=train_size)
    return x_train, x_val
```

## 讨论

处理过程相当标准。我们读取图像，将它们全部附加到一个数组中，对数组进行归一化，然后将结果集拆分为训练集和测试集。我们通过将灰度像素除以 255 来进行归一化。我们稍后将使用的激活函数是 sigmoid，它只会产生正数，因此不需要减去均值。

# 14.3 使用变分自动编码器生成图标

## 问题

您想以某种风格生成图标。

## 解决方案

在第十三章的 MNIST 解决方案中添加卷积层。

我们用于生成数字的变分自动编码器只有两个维度的潜在空间。我们可以使用这样一个小空间，因为手写数字之间的变化并不那么大。从本质上讲，只有 10 种看起来相似的不同数字。此外，我们使用全连接层来进入和离开潜在空间。我们的图标更加多样化，因此我们将使用一些卷积层来减小图像的大小，然后应用一个全连接层，最终得到我们的潜在状态：

```py
input_img = Input(shape=(32, 32, 1))
channels = 4
x = input_img
for i in range(5):
    left = Conv2D(channels, (3, 3),
                  activation='relu', padding='same')(x)
    right = Conv2D(channels, (2, 2),
                  activation='relu', padding='same')(x)
    conc = Concatenate()([left, right])
    x = MaxPooling2D((2, 2), padding='same')(conc)
    channels *= 2

 x = Dense(channels)(x)
 encoder_hidden = Flatten()(x)
```

我们像以前一样处理损失函数和分布。`KL_loss`的权重很重要。如果设置得太低，结果空间将不会很密集。如果设置得太高，网络将很快学会预测空位图会得到一个体面的`reconstruction_loss`和一个很好的`KL_loss`：

```py
z_mean = Dense(latent_space_depth,
               activation='linear')(encoder_hidden)
z_log_var = Dense(latent_space_depth,
                  activation='linear')(encoder_hidden)

def KL_loss(y_true, y_pred):
    return (0.001 * K.sum(K.exp(z_log_var)
            + K.square(z_mean) - 1 - z_log_var, axis=1))

def reconstruction_loss(y_true, y_pred):
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    return binary_crossentropy(y_true, y_pred)

    def total_loss(y_true, y_pred):
        return (reconstruction_loss(y_true, y_pred)
                + KL_loss(y_true, y_pred))
```

现在我们将潜在状态放大回图标。与以前一样，我们为编码器和自动编码器并行执行此操作：

```py
z = Lambda(sample_z,
           output_shape=(latent_space_depth, ))([z_mean, z_log_var])
decoder_in = Input(shape=(latent_space_depth,))

d_x = Reshape((1, 1, latent_space_depth))(decoder_in)
e_x = Reshape((1, 1, latent_space_depth))(z)
for i in range(5):
    conv = Conv2D(channels, (3, 3), activation='relu', padding='same')
    upsampling = UpSampling2D((2, 2))
    d_x = conv(d_x)
    d_x = upsampling(d_x)
    e_x = conv(e_x)
    e_x = upsampling(e_x)
    channels //= 2

final_conv = Conv2D(1, (3, 3), activation='sigmoid', padding='same')
auto_decoded = final_conv(e_x)
decoder_out = final_conv(d_x)
```

为了训练网络，我们需要确保训练集和测试集的大小可以被`batch_size`整除，否则`KL_loss`函数将失败：

```py
def truncate_to_batch(x):
    l = x.shape[0]
    return x[:l - l % batch_size, :, :, :]

x_train_trunc = truncate_to_batch(x_train)
x_test_trunc = truncate_to_batch(x_test)
x_train_trunc.shape, x_test_trunc.shape
```

我们可以像以前一样从空间中抽样一些随机图标：

![自动编码器生成的图标图像](img/dlcb_14in01.png)

正如您所看到的，网络确实学到了一些关于图标的东西。它们往往有一种填充在某种程度上的盒子，通常不会触及 32×32 容器的外部。但仍然相当模糊！

## 讨论

为了将我们在上一章中开发的变分自动编码器应用于图标更异构的空间，我们需要使用卷积层逐步减少位图的维度并增加抽象级别，直到我们进入潜在空间。这与图像识别网络的功能非常相似。一旦我们的图标投影到一个 128 维空间中，我们就可以为生成器和自动编码器使用上采样层。

结果比一次扣篮更有趣。问题的一部分是，图标，就像上一章中的猫一样，包含许多线条绘图，这使得网络很难完全正确地识别它们。当存在疑问时，网络会选择模糊的线条。更糟糕的是，图标通常包含像棋盘一样抖动的区域。这些模式肯定是可以学习的，但一个像素错误就会导致整个答案完全错误！

我们的网络性能相对较差的另一个原因是我们的图标相对较少。下一个配方展示了一个绕过这个问题的技巧。

# 14.4 使用数据增强来提高自动编码器的性能

## 问题

如何在不增加更多数据的情况下提高网络的性能？

## 解决方案

使用数据增强。

在上一个配方中，我们的自动编码器学习了图标集的模糊轮廓，但仅限于此。结果表明它正在捕捉某些东西，但不足以做出出色的工作。增加更多数据可能有所帮助，但这将要求我们找到更多图标，并且这些图标必须与我们的原始集合足够相似才能帮助。相反，我们将生成更多数据。

数据增强的背后思想，如第一章中讨论的，是生成输入数据的变化，这些变化对网络不重要。在这种情况下，我们希望通过输入图标让我们的网络学习*图标特性*的概念。但如果我们翻转或旋转我们的图标，这会使它们不那么*图标化*吗？实际上并不是。这样做将使我们的输入增加 16 倍。我们的网络将从这些新的训练示例中学习，旋转和翻转并不重要，希望能够表现更好。增强会是这样的：

```py
def augment(icons):
    aug_icons = []
    for icon in icons:
        for flip in range(4):
            for rotation in range(4):
                aug_icons.append(icon)
                icon = np.rot90(icon)
            icon = np.fliplr(icon)
    return np.asarray(aug_icons)
```

让我们将这应用到我们的训练和测试数据中：

```py
x_train_aug = augment(x_train)
x_test_aug = augment(x_test)
```

现在训练网络显然需要更长一点时间。但结果也更好：

![数据增强后的自动编码器图标](img/dlcb_14in02.png)

## 讨论

数据增强是在计算机图像方面广泛使用的技术。旋转和翻转是这样做的一种明显方式，但考虑到我们实际上是从图标的*.svg*表示开始的，我们还可以做很多其他事情。SVG 是一种矢量格式，因此我们可以轻松地创建具有轻微旋转或放大的图标，而不会出现我们如果基线数据仅包含位图时会出现的那种伪影。

我们得到的图标空间比上一个配方的要好，似乎捕捉到了某种图标特性。

# 14.5 构建生成对抗网络

## 问题

您想构建一个可以生成图像的网络，另一个可以学习区分生成图像和原始图像的网络。

## 解决方案

创建一个可以一起工作的图像生成器和图像鉴别器。

生成对抗网络背后的关键见解是，如果你有两个网络，一个生成图像，一个评判生成的图像，并且同时训练它们，它们在学习过程中会互相刺激。让我们从一个生成器网络开始。这与自动编码器的解码器部分类似：

```py
inp = Input(shape=(latent_size,))
x = Reshape((1, 1, latent_size))(inp)

channels = latent_size
padding = 'valid'
strides = 1
for i in range(4):
    x = Conv2DTranspose(channels, kernel_size=4,
                        strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(.2)(x)

    channels //= 2
    padding = 'same'
    strides = 2

x = Conv2DTranspose(1, kernel_size=4, strides=1, padding='same')(x)
image_out = Activation('tanh')(x)

model = Model(inputs=inp, outputs=image_out)
```

另一个网络，鉴别器，将接收一幅图像并输出它认为是生成的还是原始图像之一。在这个意义上，它看起来像一个具有二进制输出的经典卷积网络：

```py
inp = Input(shape=(32, 32, 1))
x = inp

channels = 16

for i in range(4):
    layers = []
    conv = Conv2D(channels, 3, strides=2, padding='same')(x)
    if i:
        conv = BatchNormalization()(conv)
    conv = LeakyReLU(.2)(conv)
    layers.append(conv)
    bv = Lambda(lambda x: K.mean(K.abs(x[:] - K.mean(x, axis=0)),
                          axis=-1,
                          keepdims=True))(conv)
    layers.append(bv)
    channels *= 2
    x = Concatenate()(layers)

x = Conv2D(128, 2, padding='valid')(x)
x = Flatten(name='flatten')(x)

fake = Dense(1, activation='sigmoid', name='generation')(x)

m = Model(inputs=inp, outputs=fake)
```

在下一个配方中，我们将看看如何训练这两个网络。

## 讨论

生成对抗网络或 GAN 是一种相对较新的用于生成图像的创新。看待它们的一种方式是将两个组件网络，生成器和鉴别器，视为一起学习，通过竞争变得更好。

另一种看待它们的方式是将鉴别器视为生成器的动态损失函数。当网络学习区分猫和狗时，直接的损失函数效果很好；某物是猫，或者不是，我们可以使用答案与真相之间的差异作为损失函数。

在生成图像时，这更加棘手。如何比较两幅图像？在本章早些时候，当我们使用自动编码器生成图像时，我们遇到了这个问题。在那里，我们只是逐像素比较图像；当查看两幅图像是否相同时，这种方法有效，但对于相似性来说效果不佳。两个完全相同但偏移一个像素的图标不一定会有许多像素处于相同位置。因此，自动编码器通常选择生成模糊的图像。

让第二个网络进行评判允许整个系统发展出更流畅的图像相似性感知。此外，随着图像变得更好，它可以变得更加严格，而使用自动编码器，如果我们一开始过于强调密集空间，网络将永远无法学习。

# 14.6 训练生成对抗网络

## 问题

如何训练 GAN 的两个组件？

## 解决方案

回退到底层 TensorFlow 框架来同时运行两个网络。

通常，当涉及与底层 TensorFlow 框架通信时，我们只让 Keras 承担繁重的工作。但是，我们直接使用 Keras 所能做的最好的事情是在训练生成器和鉴别器网络之间交替，这是次优的。秦永亮写了一篇[博客文章](http://bit.ly/2ILx7Te)描述了如何解决这个问题。

我们将从生成一些噪音开始，并将其输入生成器以获得生成的图像，然后将真实图像和生成的图像输入鉴别器：

```py
noise = Input(shape=g.input_shape[1:])
real_data = Input(shape=d.input_shape[1:])

generated = g(noise)
gscore = d(generated)
rscore = d(real_data)
```

现在我们可以构建两个损失函数。生成器的得分取决于鉴别器认为图像是真实的可能性。鉴别器的得分取决于它在假和真实图像上的表现：

```py
dloss = (- K.mean(K.log((1 - gscore) + .1 * K.log((1 - rscore)
         + .9 * K.log((rscore)))
gloss = - K.mean(K.log((gscore))
```

现在我们将计算梯度，以优化这两个损失函数对两个网络的可训练权重：

```py
optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.2)
grad_loss_wd = optimizer.compute_gradients(dloss, d.trainable_weights)
update_wd = optimizer.apply_gradients(grad_loss_wd)
grad_loss_wg = optimizer.compute_gradients(gloss, g.trainable_weights)
update_wg = optimizer.apply_gradients(grad_loss_wg)
```

我们收集各种步骤和张量：

```py
other_parameter_updates = [get_internal_updates(m) for m in [d, g]]
train_step = [update_wd, update_wg, other_parameter_updates]
losses = [dloss, gloss]
learning_phase = K.learning_phase()
```

现在我们准备设置训练器。Keras 需要设置 `learning_phase`：

```py
 def gan_feed(sess,batch_image, z_input):
       feed_dict = {
           noise: z_input,
           real_data: batch_image,
           learning_phase: True,
       }
       loss_values, = sess.run([losses], feed_dict=feed_dict)
```

我们可以通过生成自己的批次提供的变量：

```py
sess = K.get_session()
l = x_train.shape[0]
l -= l % BATCH_SIZE
for i in range(epochs):
    np.random.shuffle(x_train)
    for batch_start in range(0, l, BATCH_SIZE):
        batch = x_train[batch_start: batch_start + BATCH_SIZE]
        z_input = np.random.normal(loc=0.,
                                   scale=1.,
                                   size=(BATCH_SIZE, LATENT_SIZE))
        losses = gan_feed(sess, batch, z_input)
```

## 讨论

同时更新两个网络的权重使我们降到了 TensorFlow 本身的级别。虽然这有点复杂，但也很好地了解底层系统，而不总是依赖 Keras 提供的“魔法”。

###### 注意

网络上有许多实现采用简单的方式，只是逐步运行两个网络，但不是同时。

# 14.7 展示 GAN 生成的图标

## 问题

在学习过程中如何展示 GAN 的进展？

## 解决方案

在每个时代后添加一个图标渲染器。

由于我们正在运行自己的批处理，我们可以利用这一点，并在每个时代结束时使用中间结果更新笔记本。让我们从使用生成器渲染一组图标开始：

```py
def generate_images(count):
    noise = np.random.normal(loc=0.,
                             scale=1.,
                             size=(count, LATENT_SIZE))
    for tile in gm.predict([noise]).reshape((count, 32, 32)):
        tile = (tile * 300).clip(0, 255).astype('uint8')
        yield PIL.Image.fromarray(tile)
```

接下来，让我们将它们放在海报概览上：

```py
def poster(w_count, h_count):
    overview = PIL.Image.new('RGB',
                             (w_count * 34 + 2, h_count * 34 + 2),
                             (128, 128, 128))
    for idx, img in enumerate(generate_images(w_count * h_count)):
        x = idx % w_count
        y = idx // w_count
        overview.paste(img, (x * 34 + 2, y * 34 + 2))
    return overview
```

现在我们可以将以下代码添加到我们的时代循环中：

```py
        clear_output(wait=True)
        f = BytesIO()
        poster(8, 5).save(f, 'png')
        display(Image(data=f.getvalue()))
```

经过一个时代后，一些模糊的图标已经开始出现：

![第一个 GAN 生成的图像](img/dlcb_14in03.png)

再经过 25 个时代，我们真的开始看到一些图标的出现：

![经过 25 个时代后](img/dlcb_14in04.png)

## 讨论

使用 GAN 生成图标的最终结果比我们从自动编码器中得到的要好。大多数绘图更加清晰，这可以归因于鉴别器网络决定一个图标是否好，而不是逐像素比较图标。

###### 注意

GAN 及其衍生应用已经爆炸性增长，范围从从图片重建 3D 模型到为旧图片上色和超分辨率，网络可以增加小图像的分辨率而不使其看起来模糊或块状。

# 14.8 将图标编码为绘图指令

## 问题

您想将图标转换为适合训练 RNN 的格式。

## 解决方案

将图标编码为绘图指令。

RNN 可以学习序列，就像我们在第五章中看到的那样。但是如果我们想使用 RNN 生成图标怎么办？我们可以简单地将每个图标编码为像素序列。一种方法是将图标视为一系列已“打开”的像素。有 32 * 32 = 1,024 个不同的像素，因此这将是我们的词汇表。这样做是有效的，但是通过使用实际的绘图指令，我们可以做得更好。

如果我们将图标视为一系列扫描线，我们只需要 32 个不同的令牌来表示扫描线中的像素。添加一个令牌以移动到下一个扫描线，再添加一个最终令牌来标记图标的结束，我们就有了一个很好的顺序表示。或者，在代码中：

```py
def encode_icon(img, icon_size):
    size_last_x = 0
    encoded = []
    for y in range(icon_size):
        for x in range(icon_size):
            p = img.getpixel((x, y))
            if img.getpixel((x, y)) < 192:
                encoded.append(x)
                size_last_x = len(encoded)
        encoded.append(icon_size)
    return encoded[:size_last_x]
```

然后，我们可以通过遍历像素来解码图像：

```py
def decode_icon(encoded, icon_size):
    y = 0
    for idx in encoded:
        if idx == icon_size:
            y += 1
        elif idx == icon_size + 1:
            break
        else:
            x = idx
            yield x, y

    icon = PIL.Image.new('L', (32, 32), 'white')
    for x, y in decode_icon(sofar, 32):
        if y < 32:
            icon.putpixel((x, y), 0)
```

## 讨论

将图标编码为一组绘图指令只是预处理数据的另一种方式，使网络更容易学习我们想要学习的内容，类似于我们在第一章中看到的其他方法。通过具体的绘图指令，我们确保网络不会学习绘制模糊线条，就像我们的自动编码器容易做的那样——它将无法做到。

# 14.9 训练 RNN 绘制图标

## 问题

您想训练一个 RNN 来生成图标。

## 解决方案

基于绘图指令训练网络。

现在我们可以将单个图标编码为绘图指令，下一步是编码整个集合。由于我们将向 RNN 馈送块，并要求它预测下一个指令，因此我们实际上构建了一个大“文档”：

```py
def make_array(icons):
    res = []
    for icon in icons:
        res.extend(icon)
        res.append(33)
    return np.asarray(res)

def load_icons(train_size=0.90):
    icon_index = json.load(open('icons/index.json'))
    x = []
    img_rows, img_cols = 32, 32
    for icon in icon_index:
        if icon['name'].endswith('_filled'):
            continue
        img_path = 'icons/png32/%s.png' % icon['name']
        x.append(encode_icon(PIL.Image.open(img_path), 32))
    x_train, x_val = train_test_split(x, train_size=train_size)
    x_train = make_array(x_train)
    x_val = make_array(x_val)
    return x_train, x_val

x_train, x_test = load_icons()
```

我们将使用帮助我们生成莎士比亚文本的相同模型运行：

```py
def icon_rnn_model(num_chars, num_layers, num_nodes=512, dropout=0.1):
    input = Input(shape=(None, num_chars), name='input')
    prev = input
    for i in range(num_layers):
        lstm = LSTM(num_nodes, return_sequences=True,
                    name='lstm_layer_%d' % (i + 1))(prev)
        if dropout:
            prev = Dropout(dropout)(lstm)
        else:
            prev = lstm
    dense = TimeDistributed(Dense(num_chars,
                                  name='dense',
                                  activation='softmax'))(prev)
    model = Model(inputs=[input], outputs=[dense])
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

model = icon_rnn_model(34, num_layers=2, num_nodes=256, dropout=0)
```

## 讨论

要更详细地了解我们在这里使用的网络是如何训练的以及数据是如何生成的，最好回顾一下第五章。

您可以尝试不同层数和节点数，或尝试不同的 dropout 值。不同的 RNN 层也会产生影响。该模型有点脆弱；很容易陷入不学习任何内容的状态，或者在学习时陷入局部最大值。

# 14.10 使用 RNN 生成图标

## 问题

您已经训练了网络；现在如何让它生成图标？

## 解决方案

将一些测试集的随机位馈送到网络中，并将预测解释为绘图指令。

这里的基本方法与我们生成莎士比亚文本或 Python 代码时相同；唯一的区别是我们需要将预测输入到图标解码器中以获取图标。让我们首先运行一些预测：

```py
def generate_icons(model, num=2, diversity=1.0):
    start_index = random.randint(0, len(x_test) - CHUNK_SIZE - 1)
    generated = x_test[start_index: start_index + CHUNK_SIZE]
    while num > 0:
        x = np.zeros((1, len(generated), 34))
        for t, char in enumerate(generated):
            x[0, t, char] = 1.
        preds = model.predict(x, verbose=0)[0]
        preds = np.asarray(preds[len(generated) - 1]).astype('float64')
        exp_preds = np.exp(np.log(preds) / diversity)
```

`diversity`参数控制预测与确定性之间的距离（如果`diversity`为`0`，模型将将其转换为确定性）。我们需要这个来生成多样化的图标，但也要避免陷入循环。

我们将每个预测收集到一个变量`so_far`中，每次遇到值`33`（图标结束）时我们都会清空它。我们还检查`y`值是否在范围内——模型更多或更少地学习了图标的大小，但有时会尝试在线条外部着色：

```py
            if next_index == 33:
                icon = PIL.Image.new('L', (32, 32), 'white')
                for x, y in decode_icon(sofar, 32):
                    if y < 32:
                        icon.putpixel((x, y), 0)
                yield icon
                num -= 1
            else:
                sofar.append(next_index)
```

有了这个，我们现在可以绘制一个图标“海报”：

```py
cols = 10
rows = 10
overview = PIL.Image.new('RGB',
                         (cols * 36 + 4, rows * 36 + 4),
                         (128, 128, 128))
for idx, icon in enumerate(generate_icons(model, num=cols * rows)):
    x = idx % cols
    y = idx // cols
    overview.paste(icon, (x * 36 + 4, y * 36 + 4))
overview
```

![RNN 生成的图标](img/dlcb_14in05.png)

## 讨论

使用 RNN 生成的图标是我们在本章中进行的三次尝试中最大胆的，可以说最好地捕捉了图标的本质。该模型学习了图标中的对称性和基本形状，甚至偶尔进行抖动以获得半色调的概念。

我们可以尝试结合本章中的不同方法。例如，我们可以有一个 RNN，它不再尝试预测下一个绘图指令，而是接受绘图指令，捕捉该点的潜在状态，然后有一个基于该状态的第二个 RNN 重构绘图指令。这样我们就会有一个基于 RNN 的自动编码器。在文本世界中，在这个领域已经取得了一些成功。

RNNs 也可以与 GANs 结合。我们不再使用一个生成器网络，它接受一个潜变量并将其放大成一个图标，而是使用 RNN 生成绘图指令，然后让鉴别器网络决定这些是真实的还是假的。
