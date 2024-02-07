# 第4章。迁移学习和其他技巧

在查看了上一章的架构之后，您可能会想知道是否可以下载一个已经训练好的模型，然后进一步训练它。答案是肯定的！这是深度学习领域中一种非常强大的技术，称为*迁移学习*，即将一个任务（例如ImageNet）训练的网络适应到另一个任务（鱼与猫）。

为什么要这样做呢？事实证明，一个在ImageNet上训练过的架构已经对图像有了很多了解，特别是对于是否是猫或鱼（或狗或鲸鱼）有相当多的了解。因为您不再从一个基本上空白的神经网络开始，使用迁移学习，您可能会花费更少的时间进行训练，*而且*您可以通过一个远远较小的训练数据集来完成。传统的深度学习方法需要大量数据才能产生良好的结果。使用迁移学习，您可以用几百张图像构建人类级别的分类器。

# 使用ResNet进行迁移学习

现在，显而易见的事情是创建一个ResNet模型，就像我们在[第3章](ch03.html#convolutional-neural-networks)中所做的那样，并将其插入到我们现有的训练循环中。您可以这样做！ResNet模型中没有什么神奇的东西；它是由您已经看到的相同构建块构建而成。然而，这是一个庞大的模型，尽管您将看到一些改进，但您需要大量数据来确保训练*信号*到达架构的所有部分，并显著训练它们以适应您的新分类任务。我们试图避免在这种方法中使用大量数据。

然而，这里有一点需要注意：我们不再处理一个使用随机参数初始化的架构，就像我们过去所做的那样。我们的预训练的ResNet模型已经编码了大量信息，用于图像识别和分类需求，那么为什么要尝试重新训练它呢？相反，我们*微调*网络。我们稍微改变架构，以在末尾包含一个新的网络块，替换通常执行ImageNet分类的标准1,000个类别的线性层。然后，我们*冻结*所有现有的ResNet层，当我们训练时，我们只更新我们新层中的参数，但仍然从我们冻结的层中获取激活。这样可以快速训练我们的新层，同时保留预训练层已经包含的信息。

首先，让我们创建一个预训练的ResNet-50模型：

```py
from torchvision import models
transfer_model = models.ResNet50(pretrained=True)
```

接下来，我们需要冻结层。我们这样做的方法很简单：通过使用`requires_grad()`来阻止它们累积梯度。我们需要为网络中的每个参数执行此操作，但幸运的是，PyTorch提供了一个`parameters()`方法，使这变得相当容易：

```py
for name, param in transfer_model.named_parameters():
    param.requires_grad = False
```

###### 提示

您可能不想冻结模型中的`BatchNorm`层，因为它们将被训练来逼近模型最初训练的数据集的均值和标准差，而不是您想要微调的数据集。由于`BatchNorm`会*校正*您的输入，您的数据中的一些*信号*可能会丢失。您可以查看模型结构，并仅冻结不是`BatchNorm`的层，就像这样：

```py
for name, param in transfer_model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False
```

然后，我们需要用一个新的分类块替换最终的分类块，用于检测猫或鱼。在这个例子中，我们用几个`Linear`层、一个`ReLU`和`Dropout`来替换它，但您也可以在这里添加额外的CNN层。令人高兴的是，PyTorch对ResNet的实现定义了最终分类器块作为一个实例变量`fc`，所以我们只需要用我们的新结构替换它（PyTorch提供的其他模型使用`fc`或`classifier`，所以如果您尝试使用不同的模型类型，您可能需要检查源代码中的定义）：

```py
transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500),
nn.ReLU(),
nn.Dropout(), nn.Linear(500,2))
```

在上面的代码中，我们利用了`in_features`变量，它允许我们获取传入层的激活数量（在本例中为2,048）。你也可以使用`out_features`来发现传出的激活数量。当你像搭积木一样组合网络时，这些都是很方便的函数；如果一层的传入特征与前一层的传出特征不匹配，你会在运行时得到一个错误。

最后，我们回到我们的训练循环，然后像往常一样训练模型。你应该在几个epochs内看到一些准确度的大幅提升。

迁移学习是提高深度学习应用准确性的关键技术，但我们可以采用一堆其他技巧来提升我们模型的性能。让我们看看其中一些。

# 找到那个学习率

你可能还记得我在[第2章](ch02.html#image-classification-with-pytorch)中介绍了训练神经网络的*学习率*的概念，提到它是你可以改变的最重要的超参数之一，然后又提到了你应该使用什么值，建议使用一个相对较小的数字，让你尝试不同的值。不过...坏消息是，很多人确实是这样发现他们架构的最佳学习率的，通常使用一种称为*网格搜索*的技术，通过穷举搜索一部分学习率值，将结果与验证数据集进行比较。这是非常耗时的，尽管有人这样做，但许多其他人更倾向于从实践者的传统中获得经验。例如，一个已经被观察到与Adam优化器一起工作的学习率值是3e-4。这被称为Karpathy的常数，以安德烈·卡帕西（目前是特斯拉AI主管）在2016年[发推文](https://oreil.ly/WLw3q)后得名。不幸的是，更少的人读到了他的下一条推文：“我只是想确保人们明白这是一个笑话。”有趣的是，3e-4往往是一个可以提供良好结果的值，所以这是一个带有现实意味的笑话。

一方面，你可以进行缓慢而繁琐的搜索，另一方面，通过在无数架构上工作直到对一个好的学习率有了*感觉*来获得的晦涩和神秘的知识——甚至可以说是手工制作的神经网络。除了这两个极端，还有更好的方法吗？

幸运的是，答案是肯定的，尽管你会对有多少人没有使用这种更好的方法感到惊讶。美国海军研究实验室的研究科学家莱斯利·史密斯撰写的一篇有些晦涩的论文包含了一种寻找适当学习率的方法。但直到杰里米·霍华德在他的fast.ai课程中将这种技术推广开来，深度学习社区才开始关注。这个想法非常简单：在一个epoch的过程中，从一个小的学习率开始，逐渐增加到一个更高的学习率，每个小批次结束时都会有一个较高的学习率。计算每个速率的损失，然后查看绘图，选择使下降最大的学习率。例如，查看[图4-1](#learning-rate-loss-graph)中的图表。

![学习率损失图](assets/ppdl_0401.png)

###### 图4-1。学习率与损失

在这种情况下，我们应该考虑使用大约1e-2的学习率（在圆圈内标记），因为这大致是梯度下降最陡峭的点。

###### 注意

请注意，你不是在寻找曲线的底部，这可能是更直观的地方；你要找的是最快到达底部的点。

以下是fast.ai库在幕后执行的简化版本：

```py
import math
def find_lr(model, loss_fn, optimizer, init_value=1e-8, final_value=10.0):
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        inputs, labels = data
        inputs, labels = inputs, labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Crash out if loss explodes

        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs[10:-5], losses[10:-5]

        # Record the best loss

        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values

        losses.append(loss)
        log_lrs.append(math.log10(lr))

        # Do the backward pass and optimize

        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    return log_lrs[10:-5], losses[10:-5]
```

这里发生的情况是，我们遍历批次，几乎像往常一样训练；我们通过模型传递我们的输入，然后从该批次获取损失。我们记录到目前为止的`best_loss`是多少，并将新的损失与其进行比较。如果我们的新损失是`best_loss`的四倍以上，我们就会退出函数，返回到目前为止的内容（因为损失可能趋向无穷大）。否则，我们会继续附加当前学习率的损失和日志，并在循环结束时更新学习率到最大速率的下一步。然后可以使用`matplotlib`的`plt`函数显示绘图：

```py
logs,losses = find_lr()
plt.plot(logs,losses)
found_lr = 1e-2
```

请注意，我们返回`lr`日志和损失的切片。我们这样做只是因为训练的最初部分和最后几部分（特别是如果学习率变得非常快地变大）往往不会告诉我们太多信息。

fast.ai库中的实现还包括加权平滑，因此您在绘图中会得到平滑的线条，而此代码段会产生尖锐的输出。最后，请记住，因为这个函数实际上确实训练了模型并干扰了优化器的学习率设置，所以您应该在调用`find_lr()`之前保存和重新加载您的模型，以恢复到调用该函数之前的状态，并重新初始化您选择的优化器，您现在可以这样做，传入您从图表中确定的学习率！

这为我们提供了一个良好的学习率值，但我们可以通过*差异学习率*做得更好。

# 差异学习率

到目前为止，我们对整个模型应用了一个学习率。从头开始训练模型时，这可能是有道理的，但是在迁移学习时，如果我们尝试一些不同的东西，通常可以获得更好的准确性：以不同速率训练不同组层。在本章的前面，我们冻结了模型中的所有预训练层，并只训练了我们的新分类器，但是我们可能想要微调我们正在使用的ResNet模型的一些层。也许给我们的分类器之前的层添加一些训练会使我们的模型更准确一点。但是由于这些前面的层已经在ImageNet数据集上进行了训练，也许与我们的新层相比，它们只需要一点点训练？PyTorch提供了一种简单的方法来实现这一点。让我们修改ResNet-50模型的优化器：

```py
optimizer = optimizer.Adam([
{ 'params': transfer_model.layer4.parameters(), 'lr': found_lr /3},
{ 'params': transfer_model.layer3.parameters(), 'lr': found_lr /9},
], lr=found_lr)
```

这将把`layer4`（就在我们的分类器之前）的学习率设置为*找到的*学习率的三分之一，`layer3`的学习率的九分之一。这种组合在我的工作中经验上表现得非常好，但显然您可以随意尝试。不过还有一件事。正如您可能还记得本章开头所说的，我们*冻结*了所有这些预训练层。给它们一个不同的学习率是很好的，但是目前，模型训练不会触及它们，因为它们不会累积梯度。让我们改变这一点：

```py
unfreeze_layers = [transfer_model.layer3, transfer_model.layer4]
for layer in unfreeze_layers:
    for param in layer.parameters():
        param.requires_grad = True
```

现在这些层的参数再次接受梯度，当您微调模型时将应用差异学习率。请注意，您可以随意冻结和解冻模型的部分，并对每个层进行进一步的微调，如果您愿意的话！

现在我们已经看过学习率了，让我们来研究训练模型的另一个方面：我们输入的数据。

# 数据增强

数据科学中令人恐惧的短语之一是，“哦，不，我的模型在数据上过拟合了！”正如我在[第2章](ch02.html#image-classification-with-pytorch)中提到的，*过拟合*发生在模型决定反映训练集中呈现的数据而不是产生一个泛化解决方案时。你经常会听到人们谈论特定模型*记住了数据集*，意味着模型学习了答案，然后在生产数据上表现不佳。

传统的防范方法是积累大量数据。通过观察更多数据，模型对它试图解决的问题有一个更一般的概念。如果你把这种情况看作是一个压缩问题，那么如果你阻止模型简单地能够存储所有答案（通过用大量数据压倒性地超出其存储容量），它被迫*压缩*输入，因此产生一个不能简单地在自身内部存储答案的解决方案。这是可以的，而且效果很好，但是假设我们只有一千张图片，我们正在进行迁移学习。我们能做什么呢？

我们可以使用的一种方法是*数据增强*。如果我们有一张图像，我们可以对该图像做一些事情，应该可以防止过拟合，并使模型更加通用。考虑图4-2(#normal-cat-in-box)和图4-3(#flipped-cat-in-box)中的Helvetica猫的图像。

![盒子里的猫](assets/ppdl_0402.png)

###### 图4-2\. 我们的原始图像

![翻转的盒子里的猫](assets/ppdl_0403.png)

###### 图4-3\. 翻转的Helvetica

显然对我们来说，它们是相同的图像。第二个只是第一个的镜像副本。张量表示将会不同，因为RGB值将在3D图像中的不同位置。但它仍然是一只猫，所以训练在这张图像上的模型希望能够学会识别左侧或右侧帧上的猫形状，而不仅仅是将整个图像与*猫*关联起来。在PyTorch中做到这一点很简单。你可能还记得这段代码片段来自[第2章](ch02.html#image-classification-with-pytorch)：

```py
transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225] )
        ])
```

这形成了一个转换管道，所有图像在进入模型进行训练时都会经过。但是`torchivision.transforms`库包含许多其他可以用于增强训练数据的转换函数。让我们看一下一些更有用的转换，并查看Helvetica在一些不太明显的转换中会发生什么。

## Torchvision转换

`torchvision`包含了一个大量的潜在转换集合，可以用于数据增强，以及构建新转换的两种方式。在本节中，我们将看一下提供的最有用的转换，以及一些你可以在自己的应用中使用的自定义转换。

```py
torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
```

`ColorJitter`会随机改变图像的亮度、对比度、饱和度和色调。对于亮度、对比度和饱和度，你可以提供一个浮点数或一个浮点数元组，所有非负数在0到1的范围内，随机性将在0和提供的浮点数之间，或者它将使用元组生成在提供的一对浮点数之间的随机性。对于色调，需要一个在-0.5到0.5之间的浮点数或浮点数元组，它将在[-*hue*,*hue*]或[*min*, *max*]之间生成随机色调调整。参见[图4-4](#colorjitter)作为示例。

![ColorJitter应用于所有参数为0.5](assets/ppdl_0404.png)

###### 图4-4\. ColorJitter应用于所有参数为0.5

如果你想翻转你的图像，这两个转换会随机地在水平或垂直轴上反射图像：

```py
torchvision.transforms.RandomHorizontalFlip(p=0.5)
torchvision.transforms.RandomVerticalFlip(p=0.5)
```

要么提供一个从0到1的概率来发生反射，要么接受默认的50%反射几率。在[图4-5](#randomverticalflip)中展示了一个垂直翻转的猫。

![RandomVerticalFlip](assets/ppdl_0405.png)

###### 图4-5\. 垂直翻转

`RandomGrayscale`是一种类似的转换类型，不同之处在于它会随机将图像变为灰度，取决于参数*p*（默认为10%）：

```py
torchvision.transforms.RandomGrayscale(p=0.1)
```

`RandomCrop`和`RandomResizeCrop`，正如你所期望的那样，在图像上执行随机裁剪，`size`可以是一个整数，表示高度和宽度，或包含不同高度和宽度的元组。[图4-6](#randomcrop)展示了`RandomCrop`的示例。

```py
torchvision.transforms.RandomCrop(size, padding=None,
pad_if_needed=False, fill=0, padding_mode='constant')
torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0),
ratio=(0.75, 1.3333333333333333), interpolation=2)
```

现在在这里需要小心一点，因为如果您的裁剪区域太小，就有可能剪掉图像的重要部分，使模型训练错误的内容。例如，如果图像中有一只猫在桌子上玩耍，而裁剪掉了猫，只留下部分桌子被分类为*猫*，那就不太好。虽然`RandomResizeCrop`会调整裁剪大小以填充给定大小，但`RandomCrop`可能会取得靠近边缘并进入图像之外黑暗区域的裁剪。

###### 注意

`RandomResizeCrop`使用双线性插值，但您也可以通过更改`interpolation`参数选择最近邻或双三次插值。有关更多详细信息，请参阅[PIL滤镜页面](https://oreil.ly/rNOtN)。

正如您在[第3章](ch03.html#convolutional-neural-networks)中看到的，我们可以添加填充以保持图像所需的大小。默认情况下，这是`constant`填充，它用`fill`中给定的值填充图像之外的空白像素。然而，我建议您改用`reflect`填充，因为经验上它似乎比只是填充空白常数空间要好一点。

![尺寸为100的RandomCrop](assets/ppdl_0406.png)

###### 图4-6。尺寸为100的RandomCrop

如果您想要随机旋转图像，`RandomRotation`将在`[-degrees, degrees]`之间变化，如果`degrees`是一个单个浮点数或整数，或者在元组中是`(min,max)`：

```py
torchvision.transforms.RandomRotation(degrees, resample=False,expand=False, center=None)
```

如果`expand`设置为`True`，此函数将扩展输出图像，以便包含整个旋转；默认情况下，它设置为在输入尺寸内裁剪。您可以指定PIL重采样滤镜，并可选择提供一个`(x,y)`元组作为旋转中心；否则，变换将围绕图像中心旋转。[图4-7](#randomrotation)是一个`RandomRotation`变换，其中`degrees`设置为45。

![旋转角度为45度的RandomRotation](assets/ppdl_0407.png)

###### 图4-7。旋转角度为45度的RandomRotation

`Pad`是一个通用的填充变换，它在图像的边界上添加填充（额外的高度和宽度）：

```py
torchvision.transforms.Pad(padding, fill=0, padding_mode=constant)
```

`padding`中的单个值将在所有方向上应用填充。两元组`padding`将在长度为（左/右，上/下）的方向上产生填充，四元组将在（左，上，右，下）的方向上产生填充。默认情况下，填充设置为`constant`模式，它将`fill`的值复制到填充槽中。其他选择是`edge`，它将图像边缘的最后值填充到填充长度；`reflect`，它将图像的值（除边缘外）反射到边界；以及`symmetric`，它是`reflection`，但包括图像边缘的最后值。[图4-8](#padding)展示了`padding`设置为25和`padding_mode`设置为`reflect`。看看盒子如何在边缘重复。

![填充为25和填充模式为reflect的Pad](assets/ppdl_0408.png)

###### 图4-8。使用填充为25和填充模式为reflect的填充

`RandomAffine`允许您指定图像的随机仿射变换（缩放、旋转、平移和/或剪切，或任何组合）。[图4-9](#randomaffine)展示了仿射变换的一个示例。

```py
torchvision.transforms.RandomAffine(degrees, translate=None, scale=None,
shear=None, resample=False, fillcolor=0)
```

![旋转角度为10度和剪切为50的RandomAffine](assets/ppdl_0409.png)

###### 图4-9。旋转角度为10度，剪切为50的RandomAffine

`degrees`参数可以是单个浮点数或整数，也可以是一个元组。以单个形式，它会产生在（–`*degrees*`，`*degrees*`）之间的随机旋转。使用元组时，它会产生在（`*min*`，`*max*`）之间的随机旋转。必须明确设置`degrees`以防止旋转发生——没有默认设置。`translate`是一个包含两个乘数（`*horizontal_multipler*`，`*vertical_multiplier*`）的元组。在变换时，水平偏移`dx`在范围内取样，即（–`*image_width × horizontal_multiplier < dx < img_width × horizontal_width*`），垂直偏移也以相同的方式相对于图像高度和垂直乘数进行取样。

缩放由另一个元组（`*min*`，`*max*`）处理，从中随机抽取一个均匀缩放因子。剪切可以是单个浮点数/整数或一个元组，并以与`degrees`参数相同的方式随机取样。最后，`resample`允许您可选地提供一个PIL重采样滤波器，`fillcolor`是一个可选的整数，指定最终图像中位于最终变换之外的区域的填充颜色。

至于在数据增强流水线中应该使用哪些转换，我强烈建议开始使用各种随机翻转、颜色抖动、旋转和裁剪。

`torchvision`中还提供其他转换；查看[文档](https://oreil.ly/b0Q0A)以获取更多详细信息。但当然，您可能会发现自己想要创建一个特定于您的数据领域的转换，而这并不是默认包含的，因此PyTorch提供了各种定义自定义转换的方式，接下来您将看到。

## 颜色空间和Lambda转换

即使提到这似乎有点奇怪，但到目前为止，我们所有的图像工作都是在相当标准的24位RGB颜色空间中进行的，其中每个像素都有一个8位的红色、绿色和蓝色值来指示该像素的颜色。然而，其他颜色空间也是可用的！

HSV是一种受欢迎的替代方案，它具有三个8位值，分别用于*色调*、*饱和度*和*值*。一些人认为这种系统比传统的RGB颜色空间更准确地模拟了人类视觉。但为什么这很重要呢？在RGB中的一座山在HSV中也是一座山，对吧？

最近在着色方面的深度学习工作中有一些证据表明，其他颜色空间可能比RGB产生稍微更高的准确性。一座山可能是一座山，但在每个空间的表示中形成的张量将是不同的，一个空间可能比另一个更好地捕捉到您的数据的某些特征。

与集成结合使用时，您可以轻松地创建一系列模型，将RGB、HSV、YUV和LAB颜色空间的训练结果结合起来，从而从您的预测流水线中挤出更多的准确性百分点。

一个小问题是PyTorch没有提供可以执行此操作的转换。但它提供了一些工具，我们可以使用这些工具将标准RGB图像随机转换为HSV（或其他颜色空间）。首先，如果我们查看PIL文档，我们会发现可以使用`Image.convert()`将PIL图像从一种颜色空间转换为另一种。我们可以编写一个自定义的`transform`类来执行这种转换，但PyTorch添加了一个`transforms.Lambda`类，以便我们可以轻松地包装任何函数并使其可用于转换流水线。这是我们的自定义函数：

```py
def _random_colour_space(x):
    output = x.convert("HSV")
    return output
```

然后将其包装在`transforms.Lambda`类中，并可以在任何标准转换流水线中使用，就像我们以前看到的那样：

```py
colour_transform = transforms.Lambda(lambda x: _random_colour_space(x))
```

如果我们想将*每张*图像都转换为HSV，那倒没什么问题，但实际上我们并不想这样。我们希望它在每个批次中随机更改图像，因此很可能图像在不同的时期以不同的颜色空间呈现。我们可以更新我们的原始函数以生成一个随机数，并使用该随机数生成更改图像的随机概率，但相反，我们更懒惰，使用`RandomApply`：

```py
random_colour_transform = torchvision.transforms.RandomApply([colour_transform])
```

默认情况下，`RandomApply`会用值`0.5`填充参数`p`，所以转换被应用的概率是50/50。尝试添加更多的颜色空间和应用转换的概率，看看它对我们的猫和鱼问题有什么影响。

让我们看看另一个稍微复杂一些的自定义转换。

## 自定义转换类

有时一个简单的lambda不够；也许我们有一些初始化或状态要跟踪，例如。在这些情况下，我们可以创建一个自定义转换，它可以操作PIL图像数据或张量。这样的类必须实现两个方法：`__call__`，转换管道在转换过程中将调用该方法；和`__repr__`，它应该返回一个字符串表示转换，以及可能对诊断目的有用的任何状态。

在下面的代码中，我们实现了一个转换类，它向张量添加随机高斯噪声。当类被初始化时，我们传入所需噪声的均值和标准分布，在`__call__`方法中，我们从这个分布中采样并将其添加到传入的张量中：

```py
class Noise():
    """Adds gaussian noise to a tensor.

 >>> transforms.Compose([
 >>>     transforms.ToTensor(),
 >>>     Noise(0.1, 0.05)),
 >>> ])

 """
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor):
        noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)
        return tensor.add_(noise)

    def __repr__(self):
        repr = f"{self.__class__.__name__  }(mean={self.mean},
               stddev={self.stddev})"
        return repr
```

如果我们将这个添加到管道中，我们可以看到`__repr__`方法被调用的结果：

```py
transforms.Compose([Noise(0.1, 0.05))])
>> Compose(
    Noise(mean=0.1,sttdev=0.05)
)
```

因为转换没有任何限制，只是继承自基本的Python对象类，你可以做任何事情。想在运行时完全用来自Google图像搜索的东西替换图像？通过完全不同的神经网络运行图像并将结果传递到管道中？应用一系列图像转换，将图像变成其以前的疯狂反射阴影？所有这些都是可能的，尽管不完全推荐。尽管看到Photoshop的*Twirl*变换效果会使准确性变得更糟还是更好会很有趣！为什么不试试呢？

除了转换，还有一些其他方法可以尽可能地从模型中挤出更多性能。让我们看更多例子。

## 从小开始，变得更大！

这里有一个看起来奇怪但确实能获得真实结果的提示：从小开始，变得更大。我的意思是，如果你在256×256图像上训练，创建几个更多的数据集，其中图像已经缩放到64×64和128×128。使用64×64数据集创建你的模型，像平常一样微调，然后使用完全相同的模型在128×128数据集上训练。不是从头开始，而是使用已经训练过的参数。一旦看起来你已经从128×128数据中挤出了最大的价值，转移到目标256×256数据。你可能会发现准确性提高了一个或两个百分点。

虽然我们不知道为什么这样做有效，但工作理论是通过在较低分辨率训练，模型学习图像的整体结构，并随着传入图像的扩展来完善这些知识。但这只是一个理论。然而，这并不能阻止它成为一个很好的小技巧，当你需要从模型中挤出每一点性能时。

如果你不想在存储中留下数据集的多个副本，你可以使用`torchvision`转换来使用`Resize`函数实时进行操作：

```py
resize = transforms.Compose([ transforms.Resize(64),
 …_other augmentation transforms_…
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

你要付出的代价是你最终会花更多时间在训练上，因为PyTorch必须每次应用调整大小。如果你事先调整了所有图像，你可能会得到更快的训练运行，但这会填满你的硬盘。但这种权衡难道不是一直存在的吗？

从小开始然后变得更大的概念也适用于架构。使用像ResNet-18或ResNet-34这样的ResNet架构来测试转换方法并了解训练的工作方式，比起一开始就使用ResNet-101或ResNet-152模型，提供了一个更紧密的反馈循环。从小开始，逐步建立，你可以在预测时通过将它们添加到集成模型中来潜在地重复使用较小的模型运行。

# 集成

有什么比一个模型做出预测更好？那么，多个模型怎么样？*集成*是一种在传统机器学习方法中相当常见的技术，在深度学习中也非常有效。其思想是从一系列模型中获得预测，并将这些预测组合起来产生最终答案。由于不同模型在不同领域具有不同的优势，希望所有预测的组合将产生比单个模型更准确的结果。

有很多集成方法，我们不会在这里详细介绍所有方法。相反，这里有一种简单的集成方法，可以在我的经验中再增加1%的准确率；简单地平均预测结果：

```py
# Assuming you have a list of models in models, and input is your input tensor

predictions = [m[i].fit(input) for i in models]
avg_prediction = torch.stack(b).mean(0).argmax()
```

`stack`方法将张量数组连接在一起，因此，如果我们在猫/鱼问题上工作，并且在我们的集成中有四个模型，我们将得到一个由四个1×2张量构成的4×2张量。`mean`执行您所期望的操作，取平均值，尽管我们必须传入维度0以确保它在第一维上取平均值，而不仅仅是将所有张量元素相加并产生标量输出。最后，`argmax`选择具有最高元素的张量索引，就像您以前看到的那样。

很容易想象更复杂的方法。也许可以为每个单独模型的预测结果添加权重，并且如果模型回答正确或错误，则调整这些权重。您应该使用哪些模型？我发现ResNets（例如34、50、101）的组合效果非常好，没有什么能阻止您定期保存模型并在集成中使用模型的不同快照！

# 结论

当我们结束[第4章](#transfer-learning-and-other-tricks)时，我们将离开图像，转向文本。希望您不仅了解卷积神经网络在图像上的工作原理，还掌握了一系列技巧，包括迁移学习、学习率查找、数据增强和集成，这些技巧可以应用于您特定的应用领域。

# 进一步阅读

如果您对图像领域的更多信息感兴趣，请查看Jeremy Howard，Rachel Thomas和Sylvain Gugger的[fast.ai](https://fast.ai)课程。正如我所提到的，本章的学习率查找器是他们使用的简化版本，但该课程详细介绍了本章中许多技术。建立在PyTorch上的fast.ai库使您可以轻松将它们应用于您的图像（和文本！）领域。

+   [Torchvision文档](https://oreil.ly/vNnST)

+   [PIL/Pillow文档](https://oreil.ly/Jlisb)

+   Leslie N. Smith（2015）的[“用于训练神经网络的循环学习率”](https://arxiv.org/abs/1506.01186)

+   Shreyank N. Gowda和Chun Yuan（2019）的[“ColorNet：研究颜色空间对图像分类的重要性”](https://arxiv.org/abs/1902.00267)

^([1](ch04.html#idm45762364996360-marker)) 请参阅Leslie Smith（2015）的[“用于训练神经网络的循环学习率”](https://arxiv.org/abs/1506.01186)。
