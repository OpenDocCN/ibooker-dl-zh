# 第十二章 图像风格

在本章中，我们将探讨一些技术，以可视化卷积网络在分类图像时看到的内容。我们将通过反向运行网络来实现这一点——而不是给网络一个图像并询问它是什么，我们告诉网络要看到什么，并要求它以一种使检测到的物品更夸张的方式修改图像。

我们将首先为单个神经元执行此操作。这将向我们展示该神经元对哪种模式做出反应。然后，我们将引入八度概念，当我们优化图像以获得更多细节时，我们会放大。最后，我们将看看将这种技术应用到现有图像，并可视化网络在图像中“几乎”看到的内容，这种技术被称为深度梦想。

然后，我们将转变方向，看看网络的“较低”层的组合如何决定图像的艺术风格，以及我们如何仅可视化图像的风格。这使用了格拉姆矩阵的概念，以及它们如何代表一幅画的风格。

接下来，我们将看看如何将这种概念与稳定图像的方法相结合，这样我们就可以生成一幅只复制图像风格的图像。然后，我们将应用这种技术到现有图像，这样就可以以文森特·梵高的《星夜》风格呈现最近的照片。最后，我们将使用两种风格图像，并在两种风格之间的同一图片上呈现不同版本。

以下笔记本包含本章的代码：

```py
12.1 Activation Optimization
12.2 Neural Style
```

# 12.1 可视化 CNN 激活

## 问题

您想看看图像识别网络内部实际发生了什么。

## 解决方案

最大化神经元的激活，看看它对哪些像素反应最强烈。

在前一章中，我们看到卷积神经网络在图像识别方面是首选网络。最低层直接处理图像的像素，随着层叠的增加，我们推测识别特征的抽象级别也会提高。最终层能够实际识别图像中的事物。

这是直观的。这些网络的设计方式类似于我们认为人类视觉皮层是如何工作的。让我们看看个别神经元在做什么，看看这是否属实。我们将像之前一样加载网络。我们在这里使用 VGG16，因为它的架构更简单：

```py
model = vgg16.VGG16(weights='imagenet', include_top=False)
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
```

现在我们将反向运行网络。也就是说，我们将定义一个损失函数，优化一个特定神经元的激活，并要求网络计算改变图像的方向，以优化该神经元的值。在这种情况下，我们随机选择`block3_conv`层和索引为 1 的神经元：

```py
input_img = model.input
neuron_index  = 1
layer_output = layer_dict['block3_conv1'].output
loss = K.mean(layer_output[:, neuron_index, :, :])
```

要反向运行网络，我们需要定义一个名为`iterate`的 Keras 函数。它将接受一个图像，并返回损失和梯度（我们需要对网络进行的更改）。我们还需要对梯度进行归一化：

```py
grads = K.gradients(loss, input_img)[0]
grads = normalize(grads)
iterate = K.function([input_img], [loss, grads])
```

我们将从一个随机噪音图像开始，并将其重复输入到我们刚刚定义的`iterate`函数中，然后将返回的梯度添加到我们的图像中。这样逐步改变图像，使其朝着我们选择的神经元和层具有最大激活的方向变化——20 步应该可以解决问题：

```py
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step
```

在我们能够显示生成的图像之前，需要对数值进行归一化和剪裁，使其在通常的 RGB 范围内：

```py
def visstd(a, s=0.1):
    a = (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5
    return np.uint8(np.clip(a, 0, 1) * 255)
```

完成后，我们可以显示图像：

![一个激活的神经元](img/dlcb_12in01.png)

这很酷。这让我们一窥网络在这个特定层次上的运作方式。尽管整个网络有数百万个神经元；逐个检查它们并不是一个很可扩展的策略，以便深入了解正在发生的事情。

一个很好的方法是选择一些逐渐抽象的层：

```py
layers = ['block%d_conv%d' % (i, (i + 1) // 2) for i in range(1, 6)]
```

对于每一层，我们将找到八个代表性神经元，并将它们添加到一个网格中：

```py
grid = []
layers = [layer_dict['block%d_conv%d' % (i, (i + 1) // 2)]
          for i in range(1, 6)]
for layer in layers:
    row = []
    neurons = random.sample(range(max(x or 0
                            for x in layers[0].output_shape)
    for neuron in tqdm(neurons), sample_size), desc=layer.name):
        loss = K.mean(layer.output[:, neuron, :, :])
        grads = normalize(K.gradients(loss, input_img)[0])
        iterate = K.function([input_img], [loss, grads])
        img_data = np.random.uniform(size=(1, 3, 128, 128, 3)) + 128.
        for i in range(20):
            loss_value, grads_value = iterate([img_data])
            img_data += grads_value
        row.append((loss_value, img_data[0]))
    grid.append([cell[1] for cell in
                islice(sorted(row, key=lambda t: -t[0]), 10)])
```

将网格转换并在笔记本中显示与我们在 Recipe 3.3 中所做的类似：

```py
img_grid = PIL.Image.new('RGB',
                         (8 * 100 + 4, len(layers) * 100 + 4), (180, 180, 180))
for y in range(len(layers)):
    for x in range(8):
        sub = PIL.Image.fromarray(
                 visstd(grid[y][x])).crop((16, 16, 112, 112))
        img_grid.paste(sub,
                       (x * 100 + 4, (y * 100) + 4))
display(img_grid)
```

![激活的神经元网格](img/dlcb_12in02.png)

## 讨论

最大化神经网络中神经元的激活是可视化该神经元在网络整体任务中功能的好方法。通过从不同层中抽样神经元，我们甚至可以可视化随着我们在堆栈中上升，神经元检测的特征的复杂性的增加。

我们看到的结果主要包含小图案。我们更新像素的方式使得更大的对象难以出现，因为一组像素必须协同移动，它们都针对其局部内容进行了优化。这意味着更抽象的层次更难“得到它们想要的”，因为它们识别的模式具有更大的尺寸。我们可以在我们生成的网格图像中看到这一点。在下一个配方中，我们将探索一种技术来帮助解决这个问题。

你可能会想为什么我们只尝试激活低层和中间层的神经元。为什么不尝试激活最终预测层？我们可以找到“猫”的预测，并告诉网络激活它，我们期望最终得到一张猫的图片。

遗憾的是，这并不起作用。事实证明，网络将分类为“猫”的所有图像的宇宙是惊人地庞大的，但只有极少数的图像对我们来说是可识别的猫。因此，生成的图像几乎总是对我们来说像噪音，但网络认为它是一只猫。

在第十三章中，我们将探讨一些生成更真实图像的技术。

# 12.2 Octaves 和 Scaling

## 问题

如何可视化激活神经元的较大结构？

## 解决方案

在优化图像以最大化神经元激活的同时进行缩放。

在前一步中，我们看到我们可以创建最大化神经元激活的图像，但模式仍然相当局部。一个有趣的方法来解决这个问题是从一个小图像开始，然后通过一系列步骤来优化它，使用前一个配方的算法，然后对图像进行放大。这允许激活步骤首先勾勒出图像的整体结构，然后再填充细节。从一个 64×64 的图像开始：

```py
img_data = np.random.uniform(size=(1, 3, size, size)) + 128.
```

现在我们可以进行 20 次缩放/优化操作：

```py
for octave in range(20):
    if octave>0:
        size = int(size * 1.1)
        img_data = resize_img(img_data, (size, size))
    for i in range(10):
        loss_value, grads_value = iterate([img_data])
        img_data += grads_value
    clear_output()
    showarray(visstd(img_data[0]))
```

使用`block5_conv1`层和神经元 4 会得到一个看起来很有机的结果：

![Octave 激活的神经元](img/dlcb_12in03.png)

## 讨论

Octaves 和 scaling 是让网络生成某种程度上代表它所看到的东西的图像的好方法。

这里有很多可以探索的地方。在解决方案中的代码中，我们只优化一个神经元的激活，但我们可以同时优化多个神经元，以获得更混合的图片。我们可以为它们分配不同的权重，甚至为其中一些分配负权重，迫使网络远离某些激活。

当前算法有时会产生太多高频率，特别是在第一个 octave 中。我们可以通过对第一个 octave 应用高斯模糊来抵消这一点，以产生一个不那么锐利的结果。

当图像达到我们的目标大小时，为什么要停止调整大小呢？相反，我们可以继续调整大小，但同时裁剪图像以保持相同的大小。这将创建一个视频序列，我们在不断缩放的同时，新的图案不断展开。

一旦我们开始制作电影，我们还可以改变我们激活的神经元集合，并通过这种方式探索网络。*movie_dream.py*脚本结合了其中一些想法，并生成了令人着迷的电影，你可以在[YouTube](https://youtu.be/rubLdCdfDSk)上找到一个示例。

# 12.3 可视化神经网络几乎看到了什么

## 问题

你能夸大网络检测到的东西，以更好地了解它看到了什么吗？

## 解决方案

扩展前一个配方的代码以操作现有图像。

有两件事情我们需要改变才能使现有的算法工作。首先，对现有图像进行放大会使其变得相当块状。其次，我们希望保持与原始图像的某种相似性，否则我们可能会从一个随机图像开始。修复这两个问题会重现谷歌著名的 DeepDream 实验，其中出现了怪异的图片，如天空和山脉景观。

我们可以通过跟踪由于放大而丢失的细节来实现这两个目标，并将丢失的细节注入生成的图像中；这样我们就可以消除缩放造成的伪影，同时在每个八度将图像“引导”回原始状态。在以下代码中，我们获取所有想要经历的形状，然后逐步放大图像，优化图像以适应我们的损失函数，然后通过比较放大和缩小之间丢失的内容来添加丢失的细节：

```py
successive_shapes = [tuple(int(dim / (octave_scale ** i))
                     for dim in original_shape)
                     for i in range(num_octave - 1, -1, -1)]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    for i in range(20):
        loss_value, grads_value = iterate([img])
        img += grads_value
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
```

这给出了一个相当不错的结果：

![Deep Dream 一个神经元](img/dlcb_12in04.png)

原始的谷歌 Deep Dream 算法略有不同。我们刚刚告诉网络优化图像以最大化特定神经元的激活。而谷歌所做的是让网络夸大它已经看到的东西。

事实证明，我们可以通过调整我们先前定义的损失函数来优化图像，以增加当前激活。我们不再考虑一个神经元，而是要使用整个层。为了使其工作，我们必须修改我们的损失函数，使其最大化已经高的激活。我们通过取激活的平方和来实现这一点。

让我们首先指定我们要优化的三个层及其相应的权重：

```py
settings = {
        'block3_pool': 0.1,
        'block4_pool': 1.2,
        'block5_pool': 1.5,
}
```

现在我们将损失定义为这些的总和，通过仅涉及损失中的非边界像素来避免边界伪影：

```py
loss = K.variable(0.)
for layer_name, coeff in settings.items():
    x = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if K.image_data_format() == 'channels_first':
        loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
    else:
        loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling
```

`iterate`函数保持不变，生成图像的函数也是如此。我们唯一做出的改变是，在将梯度添加到图像时，通过将`grad_value`乘以 0.1 来减慢速度：

```py
    for i in range(20):
        loss_value, grads_value = iterate([img])
        img += grads_value * 0.10
```

运行此代码，我们看到眼睛和类似动物脸的东西出现在风景中：

![使用整个图像的 Deep Dream](img/dlcb_12in05.png)

您可以尝试调整层、它们的权重和速度因子以获得不同的图像。

## 讨论

Deep Dream 似乎是一种生成致幻图像的有趣方式，它确实允许无休止地探索和实验。但它也是一种了解神经网络在图像中看到的内容的方式。最终，这反映了网络训练的图像：一个训练有关猫和狗的网络将在云的图像中“看到”猫和狗。

我们可以利用来自第九章的技术。如果我们有一组大量的图像用于重新训练现有网络，但我们只将该网络的一个层设置为可训练，那么网络必须将其所有“偏见”放入该层中。然后，当我们以该层作为优化层运行深度梦想步骤时，这些“偏见”应该被很好地可视化。

总是很诱人地将神经网络的功能与人类大脑的工作方式进行类比。由于我们对后者了解不多，这当然是相当推测性的。然而，在这种情况下，某些神经元的激活似乎接近于大脑实验，研究人员通过在其中插入电极来人为激活人脑的一部分，受试者会体验到某种图像、气味或记忆。

同样，人类有能力在云的形状中识别出脸部和动物。一些改变思维的物质增加了这种能力。也许这些物质在我们的大脑中人为增加了神经层的激活？

# 12.4 捕捉图像的风格

## 问题

如何捕捉图像的风格？

## 解决方案

计算图像的卷积层的格拉姆矩阵。

在前一个教程中，我们看到了如何通过要求网络优化图像以最大化特定神经元的激活来可视化网络学到的内容。一层的格拉姆矩阵捕捉了该层的风格，因此如果我们从一个充满随机噪声的图像开始优化它，使其格拉姆矩阵与目标图像的格拉姆矩阵匹配，我们期望它开始模仿目标图像的风格。

###### 注意

格拉姆矩阵是激活的扁平化版本，乘以自身的转置。

然后，我们可以通过从每个激活集中减去格拉姆矩阵，平方结果，然后将所有结果相加来定义两组激活之间的损失函数：

```py
def gram_matrix(x):
    if K.image_data_format() != 'channels_first':
        x = K.permute_dimensions(x, (2, 0, 1))
    features = K.batch_flatten(x)
    return K.dot(features, K.transpose(features))

def style_loss(layer_1, layer_2):
    gr1 = gram_matrix(layer_1)
    gr2 = gram_matrix(layer_1)
    return K.sum(K.square(gr1 - gr2))
```

与以前一样，我们希望一个预训练网络来完成工作。我们将在两个图像上使用它，一个是我们生成的图像，另一个是我们想要从中捕捉风格的图像——在这种情况下是克劳德·莫奈 1912 年的《睡莲》。因此，我们将创建一个包含两者的输入张量，并加载一个没有最终层的网络，该网络以此张量作为输入。我们将使用`VGG16`，因为它很简单，但任何预训练网络都可以：

```py
style_image = K.variable(preprocess_image(style_image_path,
                                          target_size=(1024, 768)))
result_image = K.placeholder(style_image.shape)
input_tensor = K.concatenate([result_image,
                              style_image], axis=0)

model = vgg16.VGG16(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
```

一旦我们加载了模型，我们就可以定义我们的损失变量。我们将遍历模型的所有层，并对其中名称中包含`_conv`的层（卷积层）收集`style_image`和`result_image`之间的`style_loss`：

```py
loss = K.variable(0.)
for layer in model.layers:
    if '_conv' in layer.name:
        output = layer.output
        loss += style_loss(output[0, :, :, :], output[1, :, :, :])
```

现在我们有了一个损失，我们可以开始优化。我们将使用`scipy`的`fmin_l_bfgs_b`优化器。该方法需要一个梯度和一个损失值来完成其工作。我们可以通过一次调用获得它们，因此我们需要缓存这些值。我们使用一个方便的辅助类`Evaluator`来做到这一点，它接受一个损失和一个图像：

```py
class Evaluator(object):
    def __init__(self, loss_total, result_image):
        grads = K.gradients(loss_total, result_image)
        outputs = [loss_total] + grads
        self.iterate = K.function([result_image], outputs)
        self.shape = result_image.shape

        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        outs = self.iterate([x.reshape(self.shape)])
        self.loss_value = outs[0]
        self.grad_values = outs[-1].flatten().astype('float64')
        return self.loss_value

    def grads(self, x):
        return np.copy(self.grad_values)
```

现在我们可以通过重复调用来优化图像：

```py
image, min_val, _ = fmin_l_bfgs_b(evaluator.loss, image.flatten(),
                                  fprime=evaluator.grads, maxfun=20)
```

经过大约 50 步后，生成的图像开始看起来相当合理。

## 讨论

在这个教程中，我们看到了格拉姆矩阵有效地捕捉了图像的风格。天真地，我们可能会认为匹配图像风格的最佳方法是直接匹配所有层的激活。但这种方法太过直接了。

也许不明显格拉姆矩阵方法会更好。其背后的直觉是，通过将给定层的每个激活与其他每个激活相乘，我们捕捉了神经元之间的相关性。这些相关性编码了风格，因为它是激活分布的度量，而不是绝对激活。

考虑到这一点，我们可以尝试一些事情。一个要考虑的事情是零值。将一个向量与其自身转置的点积将在任一乘数为零时产生零。这使得无法发现与零的相关性。由于零值经常出现，这是不太理想的。一个简单的修复方法是在进行点操作之前向特征添加一个增量。值为`-1`效果很好：

```py
return K.dot(features - 1, K.transpose(features - 1))
```

我们还可以尝试向表达式添加一个常数因子。这可以使结果更加平滑或夸张。同样，`-1`效果很好。

最后要考虑的是我们正在获取所有激活的格拉姆矩阵。这可能看起来有点奇怪——难道我们不应该只为每个像素的通道执行此操作吗？实际上正在发生的是，我们为每个像素的通道计算格拉姆矩阵，然后查看它们在整个图像上的相关性。这允许一种快捷方式：我们可以计算平均通道并将其用作格拉姆矩阵。这给我们一个捕捉平均风格的图像，因此更加规则。它也运行得更快：

```py
def gram_matrix_mean(x):
    x = K.mean(x, axis=1)
    x = K.mean(x, axis=1)
    features = K.batch_flatten(x)
    return K.dot(features - 1,
                 K.transpose(features - 1)) / x.shape[0].value
```

在这个教程中添加的总变差损失告诉网络保持相邻像素之间的差异。如果没有这个，结果将更加像素化和跳跃。在某种程度上，这种方法与我们用来控制网络层的权重或输出的正则化非常相似。总体效果类似于在输出像素上应用轻微模糊滤镜。

# 12.5 改进损失函数以增加图像的连贯性

## 问题

如何使从捕捉到的风格生成的图像不那么像素化？

## 解决方案

添加一个损失组件来控制图像的局部连贯性。

前一个配方中的图像看起来已经相当合理。然而，如果我们仔细看，似乎有些像素化。我们可以通过添加一个损失函数来引导算法，确保图像在局部上是连贯的。我们将每个像素与其左侧和下方的邻居进行比较。通过试图最小化这种差异，我们引入了一种对图像进行模糊处理的方法：

```py
def total_variation_loss(x, exp=1.25):
    _, d1, d2, d3 = x.shape
    a = K.square(x[:, :d1 - 1, :d2 - 1, :] - x[:, 1:, :d2 - 1, :])
    b = K.square(x[:, :d1 - 1, :d2 - 1, :] - x[:, :d1 - 1, 1:, :])
    return K.sum(K.pow(a + b, exp))
```

1.25 指数确定了我们惩罚异常值的程度。将这个添加到我们的损失中得到：

```py
loss_variation = total_variation_loss(result_image, h, w) / 5000
loss_with_variation = loss_variation + loss_style
evaluator_with_variation = Evaluator(loss_with_variation, result_image)
```

如果我们运行这个评估器 100 步，我们会得到一个非常令人信服的图片：

![没有图像的神经风格](img/dlcb_12in06.png)

## 讨论

在这个配方中，我们向我们的损失函数添加了最终组件，使图片在全局上看起来像内容图像。实际上，我们在这里所做的是优化生成的图像，使得上层的激活对应于内容图像，而下层的激活对应于风格图像。由于下层对应于风格，上层对应于内容，我们可以通过这种方式实现风格转移。

结果可能非常引人注目，以至于新手可能认为计算机现在可以创作艺术。但仍然需要调整，因为有些风格比其他风格更狂野，我们将在下一个配方中看到。

# 12.6 将风格转移到不同的图像

## 问题

如何将从一幅图像捕捉到的风格应用到另一幅图像上？

## 解决方案

使用一个损失函数，平衡一幅图像的内容和另一幅图像的风格。

在现有图像上运行前一个配方中的代码会很容易，而不是在噪声图像上运行，但结果并不那么好。起初似乎是将风格应用到现有图像上，但随着每一步，原始图像都会逐渐消失一点。如果我们继续应用算法，最终结果将基本相同，不管起始图像如何。

我们可以通过向我们的损失函数添加第三个组件来修复这个问题，考虑生成的图像与我们参考图像之间的差异：

```py
def content_loss(base, combination):
    return K.sum(K.square(combination - base))
```

现在我们需要将参考图像添加到我们的输入张量中：

```py
w, h = load_img(base_image_path).size
base_image = K.variable(preprocess_image(base_image_path))
style_image = K.variable(preprocess_image(style2_image_path, target_size=(h, w)))
combination_image = K.placeholder(style_image.shape)
input_tensor = K.concatenate([base_image,
                              style_image,
                              combination_image], axis=0)
```

我们像以前一样加载网络，并在网络的最后一层定义我们的内容损失。最后一层包含了网络所看到的最佳近似，所以这确实是我们想要保持不变的：

```py
loss_content = content_loss(feature_outputs[-1][0, :, :, :],
                            feature_outputs[-1][2, :, :, :])
```

我们将稍微改变风格损失，考虑网络中的层的位置。我们希望较低的层承载更多的权重，因为较低的层捕捉更多的纹理/风格，而较高的层更多地涉及图像的内容。这使得算法更容易平衡图像的内容（使用最后一层）和风格（主要使用较低的层）：

```py
loss_style = K.variable(0.)
for idx, layer_features in enumerate(feature_outputs):
    loss_style += style_loss(layer_features[1, :, :, :],
                             layer_features[2, :, :, :]) * (0.5 ** idx)
```

最后，我们平衡这三个组件：

```py
loss_content /= 40
loss_variation /= 10000
loss_total = loss_content + loss_variation + loss_style
```

在阿姆斯特丹的 Oude Kerk（古教堂）的图片上运行这个算法，以梵高的《星夜》作为风格输入，我们得到：

![梵高的 Oude Kerk](img/dlcb_12in07.png)

# 12.7 风格插值

## 问题

您已经捕捉到两幅图像的风格，并希望将这两者之间的风格应用到另一幅图像中。如何混合它们？

## 解决方案

使用一个损失函数，带有额外的浮点数，指示应用每种风格的百分比。

我们可以轻松地扩展我们的输入张量，以接受两种风格图像，比如夏季和冬季各一种。在我们像以前一样加载模型之后，我们现在可以为每种风格创建一个损失：

```py
loss_style_summer = K.variable(0.)
loss_style_winter = K.variable(0.)
for idx, layer_features in enumerate(feature_outputs):
    loss_style_summer += style_loss(layer_features[1, :, :, :],
                                    layer_features[-1, :, :, :]) * (0.5 ** idx)
    loss_style_winter += style_loss(layer_features[2, :, :, :],
                                    layer_features[-1, :, :, :]) * (0.5 ** idx)
```

然后我们引入一个占位符`summerness`，我们可以输入以获得所需的`summerness`损失：

```py
summerness = K.placeholder()
loss_total = (loss_content + loss_variation +
              loss_style_summer * summerness +
              loss_style_winter * (1 - summerness))
```

我们的`Evaluator`类没有一种传递`summerness`的方法。我们可以创建一个新类或者子类现有类，但在这种情况下，我们可以通过“猴子补丁”来解决：

```py
combined_evaluator = Evaluator(loss_total, combination_image,
                               loss_content=loss_content,
                               loss_variation=loss_variation,
                               loss_style=loss_style)
iterate = K.function([combination_image, summerness],
                     combined_evaluator.iterate.outputs)
combined_evaluator.iterate = lambda inputs: iterate(inputs + [0.5])
```

这将创建一个图像，其中有 50%的夏季风格，但我们可以指定任何值。

## 讨论

在损失变量中添加另一个组件允许我们指定两种不同风格之间的权重。当然，没有任何阻止我们添加更多风格图像并改变它们的权重。值得探索的是改变风格图像的相对权重；梵高的*星空*图像非常鲜明，其风格很容易压倒更加微妙的绘画风格。
