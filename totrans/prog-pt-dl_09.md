# 第九章。PyTorch 在实践中

在我们的最后一章中，我们将看看 PyTorch 如何被其他人和公司使用。您还将学习一些新技术，包括调整图片大小、生成文本和创建可以欺骗神经网络的图像。与之前章节略有不同的是，我们将集中在如何使用现有库快速上手，而不是从头开始使用 PyTorch。我希望这将成为进一步探索的跳板。

让我们从检查一些最新的方法开始，以充分利用您的数据。

# 数据增强：混合和平滑

回到第四章，我们看了各种增强数据的方法，以帮助减少模型在训练数据集上的过拟合。在深度学习研究中，能够用更少的数据做更多事情自然是一个活跃的领域，在本节中，我们将看到两种越来越受欢迎的方法，以从您的数据中挤出最后一滴信号。这两种方法也将使我们改变如何计算我们的损失函数，因此这将是对我们刚刚创建的更灵活的训练循环的一个很好的测试。

## mixup

*mixup*是一种有趣的增强技术，它源于对我们希望模型做什么的侧面看法。我们对模型的正常理解是，我们向其发送一张像图 9-1 中的图像，并希望模型返回一个结果，即该图像是一只狐狸。

![一只狐狸](img/ppdl_0901.png)

###### 图 9-1。一只狐狸

但是，正如您所知，我们不仅从模型中得到这些；我们得到一个包含所有可能类别的张量，希望该张量中具有最高值的元素是*狐狸*类。实际上，在理想情况下，我们将有一个张量，除了狐狸类中的 1 之外，其他都是 0。

除了神经网络很难做到这一点！总会有不确定性，我们的激活函数如`softmax`使得张量很难达到 1 或 0。mixup 利用这一点提出了一个问题：图 9-2 的类是什么？

![一只猫和一只狐狸的混合](img/ppdl_0902.png)

###### 图 9-2。一只猫和一只狐狸的混合

对我们来说，这可能有点混乱，但是它是 60%的猫和 40%的狐狸。如果我们不试图让我们的模型做出明确的猜测，而是让它针对两个类别呢？这意味着我们的输出张量在训练中不会遇到接近但永远无法达到 1 的问题，我们可以通过不同的分数改变每个*混合*图像，提高我们模型的泛化能力。

但是我们如何计算这个混合图像的损失函数呢？如果*p*是混合图像中第一幅图像的百分比，那么我们有以下简单的线性组合：

```py
p * loss(image1) + (1-p) * loss(image2)
```

它必须预测这些图像，对吧？我们需要根据这些图像在最终混合图像中的比例来缩放，因此这种新的损失函数似乎是合理的。要选择*p*，我们可以像在许多其他情况下那样，使用从正态分布或均匀分布中抽取的随机数。然而，mixup 论文的作者确定，从*beta*分布中抽取的样本在实践中效果要好得多。不知道 beta 分布是什么样子？嗯，我在看到这篇论文之前也不知道！图 9-3 展示了在给定论文中描述的特征时它的样子。

![Beta 分布，其中⍺ = β](img/ppdl_0903.png)

###### 图 9-3。Beta 分布，其中⍺ = β

U 形状很有趣，因为它告诉我们，大部分时间，我们混合的图像主要是一张图像或另一张图像。再次，这是直观的，因为我们可以想象网络在工作中会更难以解决 50/50 混合比例而不是 90/10 的情况。

这是一个修改后的训练循环，它接受一个新的额外数据加载器`mix_loader`，并将批次混合在一起：

```py
def train(model, optimizer, loss_fn, train_loader, val_loader,
epochs=20, device, mix_loader):
  for epoch in range(epochs):
    model.train()
    for batch in zip(train_loader,mix_loader):
      ((inputs, targets),(inputs_mix, targets_mix)) = batch
      optimizer.zero_grad()
      inputs = inputs.to(device)
      targets = targets.to(device)
      inputs_mix = inputs_mix.to(device)
      target_mix = targets_mix.to(device)

      distribution = torch.distributions.beta.Beta(0.5,0.5)
      beta = distribution.expand(torch.zeros(batch_size).shape).sample().to(device)

      # We need to transform the shape of beta
      # to be in the same dimensions as our input tensor
      # [batch_size, channels, height, width]

      mixup = beta[:, None, None, None]

      inputs_mixed = (mixup * inputs) + (1-mixup * inputs_mix)

      # Targets are mixed using beta as they have the same shape

      targets_mixed = (beta * targets) + (1-beta * inputs_mix)

      output_mixed = model(inputs_mixed)

      # Multiply losses by beta and 1-beta,
      # sum and get average of the two mixed losses

      loss = (loss_fn(output, targets) * beta
             + loss_fn(output, targets_mixed)
             * (1-beta)).mean()

      # Training method is as normal from herein on

      loss.backward()
      optimizer.step()
      …
```

这里发生的是在获取两个批次后，我们使用`torch.distribution.Beta`生成一系列混合参数，使用`expand`方法生成一个`[1, batch_size]`的张量。我们可以遍历批次并逐个生成参数，但这样更整洁，记住，GPU 喜欢矩阵乘法，所以一次跨批次进行所有计算会更快（这在第七章中展示了，当修复我们的`BadRandom`转换时，记住！）。我们将整个批次乘以这个张量，然后使用广播将要混合的批次乘以`1 - mix_factor_tensor`。

然后我们计算两个图像的预测与目标之间的损失，最终的损失是这些损失之和的平均值。发生了什么？如果你查看`CrossEntropyLoss`的源代码，你会看到注释`损失在每个 minibatch 的观察中进行平均。`还有一个`reduction`参数，默认设置为`mean`（到目前为止我们使用了默认值，所以你之前没有看到它！）。我们需要保持这个条件，所以我们取我们合并的损失的平均值。

现在，拥有两个数据加载器并不会带来太多麻烦，但它确实使代码变得更加复杂。如果你运行这段代码，可能会出错，因为最终批次从加载器中出来时不平衡，这意味着你将不得不编写额外的代码来处理这种情况。mixup 论文的作者建议你可以用随机洗牌来替换混合数据加载器。我们可以使用`torch.randperm()`来实现这一点：

```py
shuffle = torch.randperm(inputs.size(0))
inputs_mix = inputs[shuffle]
targets_mix = targets[shuffle]
```

在这种方式下使用 mixup 时，要注意更有可能出现*碰撞*，即最终将相同的参数应用于相同的图像集，可能会降低训练的准确性。例如，你可能有 cat1 与 fish1 混合，然后抽取一个 beta 参数为 0.3。然后在同一批次中的后续步骤中，你再次抽取 fish1 并将其与 cat1 混合，参数为 0.7—这样就得到了相同的混合！一些 mixup 的实现—特别是 fast.ai 的实现—通过用以下内容替换我们的混合参数来解决这个问题：

```py
mix_parameters = torch.max(mix_parameters, 1 - mix_parameters)
```

这确保了非混洗的批次在与混合批次合并时始终具有最高的分量，从而消除了潜在的问题。

哦，还有一件事：我们在图像转换流程之后执行了 mixup 转换。此时，我们的批次只是我们相加在一起的张量。这意味着 mixup 训练不应该仅限于图像。我们可以对任何转换为张量的数据使用它，无论是文本、图像、音频还是其他任何类型的数据。

我们仍然可以做更多工作让我们的标签更加有效。现在进入另一种方法，这种方法现在是最先进模型的主要特点：*标签平滑*。

## 标签平滑

与 mixup 类似，*标签平滑*有助于通过使模型对其预测不那么确定来提高模型性能。我们不再试图强迫它预测预测类别为`1`（这在前一节中讨论的所有问题中都有问题），而是将其改为预测 1 减去一个小值，*epsilon*。我们可以创建一个新的损失函数实现，将我们现有的`CrossEntropyLoss`函数与这个功能包装在一起。事实证明，编写自定义损失函数只是`nn.Module`的另一个子类：

```py
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        num_classes = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = (-log_preds.sum(dim=-1)).mean()
        nll = F.nll_loss(log_preds, target)
        final_loss = self.epsilon * loss / num_classes +
                     (1-self.epsilon) * nll
        return final_loss
```

在计算损失函数时，我们按照`CrossEntropyLoss`的实现计算交叉熵损失。我们的`final_loss`由负对数似然乘以 1 减 epsilon（我们的*平滑*标签）加上损失乘以 epsilon 除以类别数构成。这是因为我们不仅将预测类别的标签平滑为 1 减 epsilon，还将其他标签平滑为不是被迫为零，而是在零和 epsilon 之间的值。

这个新的自定义损失函数可以替代书中任何地方使用的`CrossEntropyLoss`进行训练，并与 mixup 结合使用，是从输入数据中获得更多的一种非常有效的方法。

现在我们将从数据增强转向另一个当前深度学习趋势中的热门话题：生成对抗网络。

# 计算机，增强！

深度学习能力不断增强的一个奇怪后果是，几十年来，我们计算机人一直在嘲笑那些电视犯罪节目，其中侦探点击按钮，使模糊的摄像头图像突然变得清晰、聚焦。我们曾经嘲笑和嘲弄 CSI 等节目做这种事情。但现在我们实际上可以做到这一点，至少在一定程度上。这里有一个巫术的例子，将一个较小的 256×256 图像缩放到 512×512，见图 9-4 和 9-5。

![256x256 分辨率下的邮箱](img/ppdl_0904.png)

###### 图 9-4. 256×256 分辨率下的邮箱

![512x512 分辨率下的 ESRGAN 增强邮箱](img/ppdl_0905.png)

###### 图 9-5. 512×512 分辨率下的 ESRGAN 增强邮箱

神经网络学习如何*幻想*新的细节来填补不存在的部分，效果可能令人印象深刻。但这是如何工作的呢？

## 超分辨率简介

这是一个非常简单的超分辨率模型的第一部分。起初，它几乎与你迄今为止看到的任何模型完全相同：

```py
class OurFirstSRNet(nn.Module):

  def __init__(self):
      super(OurFirstSRNet, self).__init__()
      self.features = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=8, stride=4, padding=2),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 192, kernel_size=2, padding=2),
          nn.ReLU(inplace=True),
          nn.Conv2d(192, 256, kernel_size=2, padding=2),
          nn.ReLU(inplace=True)
      )

  def forward(self, x):
      x = self.features(x)
      return x
```

如果我们通过网络传递一个随机张量，我们最终得到一个形状为`[1, 256, 62, 62]`的张量；图像表示已经被压缩为一个更小的向量。现在让我们引入一个新的层类型，`torch.nn.ConvTranspose2d`。你可以将其视为一个反转标准`Conv2d`变换的层（具有自己的可学习参数）。我们将添加一个新的`nn.Sequential`层，`upsample`，并放入一系列这些新层和`ReLU`激活函数。在`forward()`方法中，我们将输入通过其他层后通过这个整合层：

```py
class OurFirstSRNet(nn.Module):
  def __init__(self):
      super(OurFirstSRNet, self).__init__()
      self.features = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=8, stride=4, padding=2),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 192, kernel_size=2, padding=2),
          nn.ReLU(inplace=True),
          nn.Conv2d(192, 256, kernel_size=2, padding=2),
          nn.ReLU(inplace=True)

      )
      self.upsample = nn.Sequential(
          nn.ConvTranspose2d(256,192,kernel_size=2, padding=2),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(192,64,kernel_size=2, padding=2),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(64,3, kernel_size=8, stride=4,padding=2),
          nn.ReLU(inplace=True)
      )

  def forward(self, x):
      x = self.features(x)
      x = self.upsample(x)
      return x
```

如果现在用一个随机张量测试模型，你将得到一个与输入完全相同大小的张量！我们构建的是一个*自动编码器*，一种网络类型，通常在将其压缩为更小维度后重新构建其输入。这就是我们在这里做的；`features`顺序层是一个*编码器*，将图像转换为大小为`[1, 256, 62, 62]`的张量，`upsample`层是我们的*解码器*，将其转换回原始形状。

用于训练图像的标签当然是我们的输入图像，但这意味着我们不能使用像我们相当标准的`CrossEntropyLoss`这样的损失函数，因为，嗯，我们没有类别！我们想要的是一个告诉我们输出图像与输入图像有多大不同的损失函数，为此，计算图像像素之间的均方损失或均绝对损失是一种常见方法。

###### 注意

尽管以像素为单位计算损失非常合理，但事实证明，许多最成功的超分辨率网络使用增强损失函数，试图捕捉生成图像与原始图像的相似程度，容忍像素损失以获得更好的纹理和内容损失性能。一些列在“进一步阅读”中的论文深入讨论了这一点。

现在，这使我们回到了与输入相同大小的输入，但如果我们在其中添加另一个转置卷积会怎样呢？

```py
self.upsample = nn.Sequential(...
nn.ConvTranspose2d(3,3, kernel_size=2, stride=2)
nn.ReLU(inplace=True))
```

试试吧！您会发现输出张量是输入的两倍大。如果我们有一组与该大小相同的地面真实图像作为标签，我们可以训练网络以接收大小为*x*的图像并为大小为*2x*的图像生成图像。在实践中，我们倾向于通过扩大两倍所需的量，然后添加一个标准的卷积层来执行这种上采样，如下所示：

```py
self.upsample = nn.Sequential(......
nn.ConvTranspose2d(3,3, kernel_size=2, stride=2),
nn.ReLU(inplace=True),
nn.Conv2d(3,3, kernel_size=2, stride=2),
nn.ReLU(inplace=True))
```

我们这样做是因为转置卷积有添加锯齿和 moire 图案的倾向，因为它扩展图像。通过扩展两次，然后缩小到我们需要的大小，我们希望为网络提供足够的信息来平滑这些图案，并使输出看起来更真实。

这些是超分辨率背后的基础。目前大多数性能优越的超分辨率网络都是使用一种称为生成对抗网络的技术进行训练的，这种技术在过去几年中席卷了深度学习世界。

## GANs 简介

深度学习（或任何机器学习应用）中的一个普遍问题是产生标记数据的成本。在本书中，我们大多数情况下通过使用精心标记的样本数据集来避免这个问题（甚至一些预先打包的易于训练/验证/测试集！）。但在现实世界中，产生大量标记数据。确实，到目前为止，您学到的技术，如迁移学习，都是关于如何用更少的资源做更多的事情。但有时您需要更多，*生成对抗网络*（GANs）有办法帮助。

GANs 是由 Ian Goodfellow 在 2014 年的一篇论文中提出的，是一种提供更多数据以帮助训练神经网络的新颖方法。而这种方法主要是“我们知道你喜欢神经网络，所以我们添加了另一个。”

## 伪造者和评论家

GAN 的设置如下。两个神经网络一起训练。第一个是*生成器*，它从输入张量的向量空间中获取随机噪声，并产生虚假数据作为输出。第二个网络是*鉴别器*，它在生成的虚假数据和真实数据之间交替。它的工作是查看传入的输入并决定它们是真实的还是虚假的。GAN 的简单概念图如图 9-6 所示。

![一个简单的 GAN 设置](img/ppdl_0906.png)

###### 图 9-6。一个简单的 GAN 设置

GANs 的伟大之处在于，尽管细节最终变得有些复杂，但总体思想很容易传达：这两个网络相互对立，在训练过程中，它们尽力击败对方。在过程结束时，*生成器*应该生成与真实输入数据的*分布*匹配的数据，以迷惑*鉴别器*。一旦达到这一点，您可以使用生成器为所有需求生成更多数据，而鉴别器可能会退休到神经网络酒吧淹没忧愁。

## 训练 GAN

训练 GAN 比训练传统网络稍微复杂一些。在训练循环中，我们首先需要使用真实数据开始训练鉴别器。我们计算鉴别器的损失（使用 BCE，因为我们只有两类：真实或虚假），然后进行反向传播以更新鉴别器的参数，就像往常一样。但这一次，我们*不*调用优化器来更新。相反，我们从生成器生成一批数据并通过模型传递。我们计算损失并进行*另一次*反向传播，因此此时训练循环已计算了两次通过模型的损失。现在，我们根据这些*累积*梯度调用优化器进行更新。

在训练的后半段，我们转向生成器。我们让生成器访问鉴别器，然后生成一批新数据（生成器坚持说这些都是真实的！）并将其与鉴别器进行测试。我们根据这些输出数据形成一个损失，鉴别器说是假的每个数据点都被视为*错误*答案——因为我们试图欺骗它——然后进行标准的反向/优化传递。

这是 PyTorch 中的一个通用实现。请注意，生成器和鉴别器只是标准的神经网络，因此从理论上讲，它们可以生成图像、文本、音频或任何类型的数据，并且可以由迄今为止看到的任何类型的网络构建：

```py
generator = Generator()
discriminator = Discriminator()

# Set up separate optimizers for each network
generator_optimizer = ...
discriminator_optimizer = ...

def gan_train():
  for epoch in num_epochs:
    for batch in real_train_loader:
      discriminator.train()
      generator.eval()
      discriminator.zero_grad()

      preds = discriminator(batch)
      real_loss = criterion(preds, torch.ones_like(preds))
      discriminator.backward()

      fake_batch = generator(torch.rand(batch.shape))
      fake_preds = discriminator(fake_batch)
      fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))
      discriminator.backward()

      discriminator_optimizer.step()

      discriminator.eval()
      generator.train()
      generator.zero_grad()

      forged_batch = generator(torch.rand(batch.shape))
      forged_preds = discriminator(forged_batch)
      forged_loss = criterion(forged_preds, torch.ones_like(forged_preds))

      generator.backward()
      generator_optimizer.step()
```

请注意，PyTorch 的灵活性在这里非常有帮助。没有专门为更标准的训练而设计的训练循环，构建一个新的训练循环是我们习惯的事情，我们知道需要包含的所有步骤。在其他一些框架中，训练 GAN 有点更加繁琐。这很重要，因为训练 GAN 本身就是一个困难的任务，如果框架阻碍了这一过程，那就更加困难了。

## 模式崩溃的危险

在理想的情况下，训练过程中发生的是，鉴别器一开始会擅长检测假数据，因为它是在真实数据上训练的，而生成器只允许访问鉴别器而不是真实数据本身。最终，生成器将学会如何欺骗鉴别器，然后它将迅速改进以匹配数据分布，以便反复产生能够欺骗评论者的伪造品。

但是困扰许多 GAN 架构的一件事是*模式崩溃*。如果我们的真实数据有三种类型的数据，那么也许我们的生成器会开始生成第一种类型，也许它开始变得相当擅长。鉴别器可能会决定任何看起来像第一种类型的东西实际上是假的，甚至是真实的例子本身，然后生成器开始生成看起来像第三种类型的东西。鉴别器开始拒绝所有第三种类型的样本，生成器选择另一个真实例子来生成。这个循环无休止地继续下去；生成器永远无法进入一个可以从整个分布中生成样本的阶段。

减少模式崩溃是使用 GAN 的关键性能问题，也是一个正在进行研究的领域。一些方法包括向生成的数据添加相似性分数，以便可以检测和避免潜在的崩溃，保持一个生成图像的重放缓冲区，以便鉴别器不会过度拟合到最新批次的生成图像，允许从真实数据集中添加实际标签到生成器网络等等。

接下来，我们通过检查一个执行超分辨率的 GAN 应用程序来结束本节。

## ESRGAN

*增强超分辨率生成对抗网络*（ESRGAN）是一种在 2018 年开发的网络，可以产生令人印象深刻的超分辨率结果。生成器是一系列卷积网络块，其中包含残差和稠密层连接的组合（因此是 ResNet 和 DenseNet 的混合），移除了`BatchNorm`层，因为它们似乎会在上采样图像中产生伪影。对于鉴别器，它不是简单地产生一个结果，说*这是真实的*或*这是假的*，而是预测一个真实图像相对更真实的概率比一个假图像更真实，这有助于使模型产生更自然的结果。

### 运行 ESRGAN

为了展示 ESRGAN，我们将从[GitHub 存储库](https://github.com/xinntao/ESRGAN)下载代码。使用**`git`**克隆：

```py
git clone https://github.com/xinntao/ESRGAN
```

然后我们需要下载权重，这样我们就可以在不训练的情况下使用模型。使用自述文件中的 Google Drive 链接，下载*RRDB_ESRGAN_x4.pth*文件并将其放在*./models*中。我们将对 Helvetica 的缩小版本进行上采样，但可以随意将任何图像放入*./LR*目录。运行提供的*test.py*脚本，您将看到生成的上采样图像并保存在*results*目录中。

这就是超分辨率的全部内容，但我们还没有完成图像处理。

# 图像检测的进一步探索

我们在第二章到第四章的图像分类都有一个共同点：我们确定图像属于一个类别，猫或鱼。显然，在实际应用中，这将扩展到一个更大的类别集。但我们也希望图像可能包含猫和鱼（这对鱼可能是个坏消息），或者我们正在寻找的任何类别。场景中可能有两个人、一辆车和一艘船，我们不仅希望确定它们是否出现在图像中，还希望确定它们在图像中的*位置*。有两种主要方法可以实现这一点：*目标检测*和*分割*。我们将看看这两种方法，然后转向 Facebook 的 PyTorch 实现的 Faster R-CNN 和 Mask R-CNN，以查看具体示例。

## 目标检测

让我们看看我们的盒子里的猫。我们真正想要的是让网络将猫放在另一个盒子里！特别是，我们希望有一个*边界框*，包围模型认为是*猫*的图像中的所有内容，如图 9-7 所示。

![盒子里的猫在一个边界框中](img/ppdl_0907.png)

###### 图 9-7\. 盒子里的猫在一个边界框中

但我们如何让我们的网络解决这个问题呢？请记住，这些网络可以预测您想要的任何内容。如果在我们的 CATFISH 模型中，我们除了预测一个类别之外，还产生四个额外的输出怎么样？我们将有一个输出大小为`6`的`Linear`层，而不是`2`。额外的四个输出将使用*x[1]、x[2]、y[1]、y[2]*坐标定义一个矩形。我们不仅要提供图像作为训练数据，还必须用边界框增强它们，以便模型有东西可以训练，当然。我们的损失函数现在将是类别预测的交叉熵损失和边界框的均方损失的组合损失。

这里没有魔法！我们只需设计模型以满足我们的需求，提供具有足够信息的数据进行训练，并包含一个告诉网络它的表现如何的损失函数。

与边界框的泛滥相比，*分割* 是一种替代方法。我们的网络不是生成框，而是输出与输入相同大小的图像掩模；掩模中的像素根据它们所属的类别着色。例如，草可能是绿色的，道路可能是紫色的，汽车可能是红色的，等等。

由于我们正在输出图像，您可能会认为我们最终会使用与超分辨率部分类似的架构。这两个主题之间存在很多交叉，近年来变得流行的一种模型类型是*U-Net*架构，如图 9-8 所示。³

![简化的 U-Net 架构](img/ppdl_0908.png)

###### 图 9-8\. 简化的 U-Net 架构

正如您所看到的，经典的 U-Net 架构是一组卷积块，将图像缩小，另一系列卷积将其缩放回目标图像。然而，U-Net 的关键在于从左侧块到右侧对应块的横跨线，这些线与输出张量连接在一起，当图像被缩放回来时，这些连接允许来自更高级别卷积块的信息传递，保留可能在卷积块减少输入图像时被移除的细节。

您会发现基于 U-Net 的架构在 Kaggle 分割竞赛中随处可见，从某种程度上证明了这种结构对于分割是一个不错的选择。已经应用到基本设置的另一种技术是我们的老朋友迁移学习。在这种方法中，U 的第一部分取自预训练模型，如 ResNet 或 Inception，U 的另一侧加上跳跃连接，添加到训练网络的顶部，并像往常一样进行微调。

让我们看看一些现有的预训练模型，可以直接从 Facebook 获得最先进的目标检测和分割。

## Faster R-CNN 和 Mask R-CNN

Facebook Research 开发了*maskrcnn-benchmark*库，其中包含目标检测和分割算法的参考实现。我们将安装该库并添加代码来生成预测。在撰写本文时，构建模型的最简单方法是使用 Docker（当 PyTorch 1.2 发布时可能会更改）。从[*https://github.com/facebookresearch/maskrcnn-benchmark*](https://github.com/facebookresearch/maskrcnn-benchmark)克隆存储库，并将此脚本*predict.py*添加到*demo*目录中，以设置使用 ResNet-101 骨干的预测管道：

```py
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import sys
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2.yaml"

cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=500,
    confidence_threshold=0.7,
)

pil_image = Image.open(sys.argv[1])
image = np.array(pil_image)[:, :, [2, 1, 0]]
predictions = coco_demo.run_on_opencv_image(image)
predictions = predictions[:,:,::-1]

plt.imsave(sys.argv[2], predictions)
```

在这个简短的脚本中，我们首先设置了`COCODemo`预测器，确保我们传入的配置设置了 Faster R-CNN 而不是 Mask R-CNN（后者会产生分割输出）。然后我们打开一个在命令行上设置的图像文件，但是我们必须将其转换为`BGR`格式而不是`RGB`格式，因为预测器是在 OpenCV 图像上训练的，而不是我们迄今为止使用的 PIL 图像。最后，我们使用`imsave`将`predictions`数组（原始图像加上边界框）写入一个新文件，也在命令行上指定。将一个测试图像文件复制到这个*demo*目录中，然后我们可以构建 Docker 镜像：

```py
docker build docker/
```

我们从 Docker 容器内运行脚本，并生成类似于图 9-7 的输出（我实际上使用了该库来生成该图像）。尝试尝试不同的`confidence_threshold`值和不同的图片。您还可以切换到`e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml`配置，尝试 Mask R-CNN 并生成分割蒙版。

要在这些模型上训练您自己的数据，您需要提供一个数据集，为每个图像提供边界框标签。该库提供了一个名为`BoxList`的辅助函数。以下是一个数据集的骨架实现，您可以将其用作起点：

```py
from maskrcnn_benchmark.structures.bounding_box import BoxList

class MyDataset(object):
    def __init__(self, path, transforms=None):
        self.images = # set up image list
        self.boxes = # read in boxes
        self.labels = # read in labels

    def __getitem__(self, idx):
        image = # Get PIL image from self.images
        boxes = # Create a list of arrays, one per box in x1, y1, x2, y2 format
        labels = # labels that correspond to the boxes

        boxlist = BoxList(boxes, image.size, mode="xyxy")
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        return image, boxlist, idx

    def get_img_info(self, idx):
        return {"height": img_height, "width": img_width
```

然后，您需要将新创建的数据集添加到*maskrcnn_benchmark/data/datasets/*init*.py*和*maskrcnn_benchmark/config/paths_catalog.py*中。然后可以使用存储库中提供的*train_net.py*脚本进行训练。请注意，您可能需要减少批量大小以在单个 GPU 上训练这些网络中的任何一个。

这就是目标检测和分割的全部内容，但是请参阅“进一步阅读”以获取更多想法，包括标题为 You Only Look Once（YOLO）架构的内容。与此同时，我们将看看如何恶意破坏模型。

# 对抗样本

你可能在网上看到过关于图像如何阻止图像识别正常工作的文章。如果一个人将图像举到相机前，神经网络会认为它看到了熊猫或类似的东西。这些被称为*对抗样本*，它们是发现架构限制以及如何最好地防御的有趣方式。

创建对抗样本并不太困难，特别是如果你可以访问模型。这里有一个简单的神经网络，用于对来自流行的 CIFAR-10 数据集的图像进行分类。这个模型没有什么特别之处，所以可以随意将其替换为 AlexNet、ResNet 或本书中迄今为止介绍的任何其他网络：

```py
class ModelToBreak(nn.Module):
    def __init__(self):
        super(ModelToBreak, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

一旦网络在 CIFAR-10 上训练完成，我们可以为图 9-9 中的图像获得预测。希望训练已经足够好，可以报告这是一只青蛙（如果没有，可能需要再多训练一会儿！）。我们要做的是稍微改变我们的青蛙图片，让神经网络感到困惑，认为它是其他东西，尽管我们仍然可以清楚地认出它是一只青蛙。

![我们的青蛙示例](img/ppdl_0909.png)

###### 图 9-9。我们的青蛙示例

为此，我们将使用一种名为*快速梯度符号方法*的攻击方法。⁴ 这个想法是拿我们想要误分类的图像并像往常一样通过模型运行它，这给我们一个输出张量。通常情况下，对于预测，我们会查看张量中哪个值最高，并将其用作我们类的索引，使用`argmax()`。但这一次，我们将假装再次训练网络，并将结果反向传播回模型，给出模型相对于原始输入（在这种情况下，我们的青蛙图片）的梯度变化。

完成后，我们创建一个新的张量，查看这些梯度并用+1 替换一个条目，如果梯度为正则用-1。这给我们这个图像推动模型决策边界的方向。然后我们乘以一个小标量（在论文中称为*epsilon*）来生成我们的恶意掩码，然后将其添加到原始图像中，创建一个对抗样本。

这里有一个简单的 PyTorch 方法，当提供批次的标签、模型和用于评估模型的损失函数时，返回输入批次的快速梯度符号张量：

```py
def fgsm(input_tensor, labels, epsilon=0.02, loss_function, model):
    outputs = model(input_tensor)
    loss = loss_function(outputs, labels)
    loss.backward(retain_graph=True)
    fsgm = torch.sign(inputs.grad) * epsilon
    return fgsm
```

通过实验通常可以找到 Epsilon。通过尝试各种图像，我发现`0.02`对这个模型效果很好，但你也可以使用类似网格或随机搜索的方法来找到将青蛙变成船的值！

在我们的青蛙和模型上运行这个函数，我们得到一个掩码，然后我们可以将其添加到我们的原始图像中生成我们的对抗样本。看看图 9-10 看看它是什么样子！

```py
model_to_break = # load our model to break here
adversarial_mask = fgsm(frog_image.unsqueeze(-1),
                        batch_labels,
                        loss_function,
                        model_to_break)
adversarial_image = adversarial_mask.squeeze(0) + frog_image
```

![我们的对抗性青蛙](img/ppdl_0910.png)

###### 图 9-10。我们的对抗性青蛙

显然，我们创建的图像对我们的人眼来说仍然是一只青蛙。（如果对你来说看起来不像青蛙，那么你可能是一个神经网络。立即报告自己进行 Voight-Kampff 测试。）但如果我们从模型对这个新图像的预测中得到一个结果会发生什么？

```py
model_to_break(adversarial_image.unsqueeze(-1))
# look up in labels via argmax()
>> 'cat'
```

我们打败了模型。但这是否像最初看起来的那样成为问题呢？

## 黑盒攻击

你可能已经注意到，要生成愚弄分类器的图像，我们需要了解使用的模型的很多信息。我们面前有整个模型的结构以及在训练模型时使用的损失函数，我们需要在模型中进行前向和后向传递以获得我们的梯度。这是计算机安全领域中所知的*白盒攻击*的一个典型例子，我们可以窥视代码的任何部分来弄清楚发生了什么并利用我们能找到的任何东西。

那么这有关系吗？毕竟，大多数你在网上遇到的模型都不会让你窥视内部。*黑盒攻击*，即只有输入和输出的攻击，实际上可能吗？很遗憾，是的。考虑我们有一组输入和一组要与之匹配的输出。输出是*标签*，可以使用模型的有针对性查询来训练一个新模型，你可以将其用作本地代理并以白盒方式进行攻击。就像你在迁移学习中看到的那样，对代理模型的攻击可以有效地作用于实际模型。我们注定要失败吗？

## 防御对抗性攻击

我们如何防御这些攻击？对于像将图像分类为猫或鱼这样的任务，这可能不是世界末日，但对于自动驾驶系统、癌症检测应用等，这可能真的意味着生与死的区别。成功地防御各种类型的对抗性攻击仍然是一个研究领域，但迄今为止的重点包括提炼和验证。

通过使用模型来训练*另一个*模型来*提炼*似乎有所帮助。使用本章前面概述的新模型的标签平滑也似乎有所帮助。使模型对其决策不那么确定似乎可以在一定程度上平滑梯度，使我们在本章中概述的基于梯度的攻击不那么有效。

更强大的方法是回到早期计算机视觉时代的一些部分。如果我们对传入数据进行输入验证，可能可以防止对抗性图像首先到达模型。在前面的例子中，生成的攻击图像有一些像素与我们看到青蛙时期望的非常不匹配。根据领域的不同，我们可以设置一个过滤器，只允许通过一些过滤测试的图像。你理论上也可以制作一个神经网络来做这个，因为攻击者必须尝试用相同的图像破坏两个不同的模型！

现在我们真的已经结束了关于图像的讨论。但让我们看看过去几年发生的文本网络方面的一些发展。

# 眼见不一定为实：变压器架构

在过去的十年中，迁移学习一直是使基于图像的网络变得如此有效和普遍的一个重要特征，但文本一直是一个更难解决的问题。然而，在过去几年中，已经迈出了一些重要的步骤，开始揭示了在文本中使用迁移学习的潜力，用于各种任务，如生成、分类和回答问题。我们还看到了一种新型架构开始占据主导地位：*变压器网络*。这些网络并不来自赛博特隆，但这种技术是我们看到的最强大的基于文本的网络背后的技术，OpenAI 于 2019 年发布的 GPT-2 模型展示了其生成文本的惊人质量，以至于 OpenAI 最初推迟了模型的更大版本，以防止其被用于不良目的。我们将研究变压器的一般理论，然后深入探讨如何使用 Hugging Face 的 GPT-2 和 BERT 实现。

## 专注

通往变压器架构的途中的初始步骤是*注意力*机制，最初引入到 RNN 中，以帮助序列到序列的应用，如翻译。⁵

*注意力*机制试图解决的问题是翻译句子如“猫坐在垫子上，她发出了咕噜声。”我们知道该句中的*她*指的是猫，但让标准的 RNN 理解这个概念很困难。它可能有我们在第五章中讨论过的隐藏状态，但当我们到达*她*时，我们已经有了很多时间步和每个步骤的隐藏状态！

那么*attention*的作用是为每个时间步添加一组额外的可学习权重，将网络聚焦在句子的特定部分。这些权重通常通过`softmax`层传递，生成每个步骤的概率，然后将注意力权重的点积与先前的隐藏状态进行计算。图 9-11 展示了关于我们句子的这个简化版本。

![指向“cat”的注意向量](img/ppdl_0911.png)

###### 图 9-11. 指向“cat”的注意向量

这些权重确保当隐藏状态与当前状态组合时，“cat”将成为决定“she”时间步输出向量的主要部分，这将为将其翻译成法语提供有用的上下文！

我们不会详细介绍*attention*在具体实现中如何工作，但知道这个概念足够强大，足以在 2010 年代中期推动了谷歌翻译的显著增长和准确性。但更多的东西即将到来。

## 注意力机制就是你需要的一切

在开创性的论文“注意力就是你需要的一切”中，谷歌研究人员指出，我们花了很多时间将注意力添加到已经相对较慢的基于 RNN 的网络上（与 CNN 或线性单元相比）。如果我们根本不需要 RNN 呢？该论文表明，通过堆叠基于注意力的编码器和解码器，您可以创建一个完全不依赖于 RNN 隐藏状态的模型，为今天主导文本深度学习的更大更快的 Transformer 铺平了道路。

关键思想是使用作者称之为*多头注意力*，它通过使用一组`Linear`层在所有输入上并行化*attention*步骤。借助这些技巧，并从 ResNet 借鉴一些残差连接技巧，Transformer 迅速开始取代 RNN 用于许多基于文本的应用。两个重要的 Transformer 版本，BERT 和 GPT-2，代表了当前的最先进技术，本书付印时。

幸运的是，Hugging Face 有一个库在 PyTorch 中实现了这两个模型。它可以使用`pip`或`conda`进行安装，您还应该`git clone`该存储库本身，因为我们稍后将使用一些实用脚本！

```py
pip install pytorch-transformers
conda install pytorch-transformers
```

首先，我们将看一下 BERT。

## BERT

谷歌 2018 年的*双向编码器表示转换器*（BERT）模型是将强大模型的迁移学习成功应用的首批案例之一。BERT 本身是一个庞大的基于 Transformer 的模型（在其最小版本中有 1.1 亿个参数），在维基百科和 BookCorpus 数据集上进行了预训练。传统上，Transformer 和卷积网络在处理文本时存在的问题是，因为它们一次看到所有数据，这些网络很难学习语言的时间结构。BERT 通过在预训练阶段随机屏蔽文本输入的 15%，并强制模型预测已被屏蔽的部分来解决这个问题。尽管在概念上很简单，但最大模型中 3.4 亿个参数的庞大规模与 Transformer 架构的结合，为一系列与文本相关的基准测试带来了新的最先进结果。

当然，尽管 BERT 是由 Google 与 TensorFlow 创建的，但也有适用于 PyTorch 的 BERT 实现。现在让我们快速看一下其中一个。

## FastBERT

在您自己的分类应用程序中开始使用 BERT 模型的一种简单方法是使用*FastBERT*库，该库将 Hugging Face 的存储库与 fast.ai API 混合在一起（稍后我们将在 ULMFiT 部分更详细地看到）。它可以通过常规方式使用`pip`进行安装：

```py
pip install fast-bert
```

以下是一个可以用来在我们在第五章中使用的 Sentiment140 Twitter 数据集上微调 BERT 的脚本：

```py
import torch
import logger

from pytorch_transformers.tokenization import BertTokenizer
from fast_bert.data import BertDataBunch
from fast_bert.learner import BertLearner
from fast_bert.metrics import accuracy

device = torch.device('cuda')
logger = logging.getLogger()
metrics = [{'name': 'accuracy', 'function': accuracy}]

tokenizer = BertTokenizer.from_pretrained
                ('bert-base-uncased',
                  do_lower_case=True)

databunch = BertDataBunch([PATH_TO_DATA],
                          [PATH_TO_LABELS],
                          tokenizer,
                          train_file=[TRAIN_CSV],
                          val_file=[VAL_CSV],
                          test_data=[TEST_CSV],
                          text_col=[TEST_FEATURE_COL], label_col=[0],
                          bs=64,
                          maxlen=140,
                          multi_gpu=False,
                          multi_label=False)

learner = BertLearner.from_pretrained_model(databunch,
                      'bert-base-uncased',
                      metrics,
                      device,
                      logger,
                      is_fp16=False,
                      multi_gpu=False,
                      multi_label=False)

learner.fit(3, lr='1e-2')
```

在导入之后，我们设置了`device`、`logger`和`metrics`对象，这些对象是`BertLearner`对象所需的。然后我们创建了一个`BERTTokenizer`来对我们的输入数据进行标记化，在这个基础上我们将使用`bert-base-uncased`模型（具有 12 层和 1.1 亿参数）。接下来，我们需要一个包含训练、验证和测试数据集路径的`BertDataBunch`对象，以及标签列的位置、批处理大小和我们输入数据的最大长度，对于我们的情况来说很简单，因为它只能是推文的长度，那时是 140 个字符。做完这些之后，我们将通过使用`BertLearner.from_pretrained_model`方法来设置 BERT 模型。这个方法传入了我们的输入数据、BERT 模型类型、我们在脚本开始时设置的`metric`、`device`和`logger`对象，最后一些标志来关闭我们不需要但方法签名中没有默认值的训练选项。

最后，`fit()`方法负责在我们的输入数据上微调 BERT 模型，运行自己的内部训练循环。在这个例子中，我们使用学习率为`1e-2`进行三个 epochs 的训练。训练后的 PyTorch 模型可以通过`learner.model`进行访问。

这就是如何开始使用 BERT。现在，进入比赛。

## GPT-2

现在，当谷歌悄悄地研究 BERT 时，OpenAI 正在研究自己版本的基于 Transformer 的文本模型。该模型不使用掩码来强制模型学习语言结构，而是将架构内的注意机制限制在简单地预测序列中的下一个单词，类似于第五章中的 RNN 的风格。结果，GPT 在 BERT 的出色性能下有些落后，但在 2019 年，OpenAI 推出了*GPT-2*，这是该模型的新版本，重新定义了文本生成的标准。

GPT-2 背后的魔力在于规模：该模型训练于 800 万个网站的文本，最大变体的 GPT-2 拥有 15 亿个参数。虽然它仍然无法在特定基准上击败 BERT，比如问答或其他 NLP 任务，但它能够从基本提示中创建出极为逼真的文本，这导致 OpenAI 将全尺寸模型锁在了闭门之后，以防止被武器化。然而，他们发布了模型的较小版本，其中 117 和 340 亿个参数。

这里是 GPT-2 可以生成的输出示例。所有斜体部分都是由 GPT-2 的 340M 模型编写的：

> 杰克和吉尔骑着自行车上山。天空是灰白色的，风在吹，导致大雪纷飞。下山真的很困难，我不得不向前倾斜一点点才能继续前行。但接着有一个我永远不会忘记的自由时刻：自行车完全停在山坡上，我就在其中间。我没有时间说一句话，但我向前倾斜，触碰了刹车，自行车开始前进。

除了从*杰克和吉尔*切换到*I*，这是一个令人印象深刻的文本生成。对于短文本，它有时几乎无法与人类创作的文本区分开。随着生成文本的继续，它揭示了幕后的机器，但这是一个令人印象深刻的成就，它现在可以写推文和 Reddit 评论。让我们看看如何在 PyTorch 中实现这一点。

## 使用 GPT-2 生成文本

与 BERT 一样，OpenAI 发布的官方 GPT-2 版本是一个 TensorFlow 模型。与 BERT 一样，Hugging Face 发布了一个 PyTorch 版本，该版本包含在同一个库（`pytorch-transformers`）中。然而，围绕原始 TensorFlow 模型构建了一个蓬勃发展的生态系统，而目前在 PyTorch 版本周围并不存在。因此，这一次，我们将作弊：我们将使用一些基于 TensorFlow 的库来微调 GPT-2 模型，然后导出权重并将其导入模型的 PyTorch 版本。为了节省我们太多的设置，我们还在 Colab 笔记本中执行所有 TensorFlow 操作！让我们开始吧。

打开一个新的 Google Colab 笔记本，并安装我们正在使用的库，Max Woolf 的*gpt-2-simple*，它将 GPT-2 微调封装在一个单一软件包中。通过将此添加到单元格中进行安装：

```py
!pip3 install gpt-2-simple
```

接下来，您需要一些文本。在此示例中，我使用了 PG Wodehouse 的*My Man Jeeves*的公共领域文本。我还不打算在从 Project Gutenberg 网站使用`wget`下载文本后对文本进行任何进一步处理：

```py
!wget http://www.gutenberg.org/cache/epub/8164/pg8164.txt
```

现在我们可以使用库进行训练。首先确保您的笔记本连接到 GPU（在 Runtime→Change Runtime Type 中查看），然后在单元格中运行此代码：

```py
import gpt_2_simple as gpt2

gpt2.download_gpt2(model_name="117M")

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              "pg8164.txt",model_name="117M",
              steps=1000)
```

用您正在使用的文本文件替换文本文件。当模型训练时，它将每一百步输出一个样本。在我的情况下，看到它从模糊的莎士比亚剧本变成接近伍德豪斯散文的东西很有趣。这可能需要一个小时或两个小时来训练 1,000 个时代，所以在云端的 GPU 忙碌时，去做一些更有趣的事情吧。

完成后，我们需要将权重从 Colab 中取出并放入您的 Google Drive 帐户，以便您可以将它们下载到运行 PyTorch 的任何地方：

```py
gpt2.copy_checkpoint_to_gdrive()
```

这将指引您打开一个新的网页，将认证代码复制到笔记本中。完成后，权重将被打包并保存到您的 Google Drive 中，文件名为*run1.tar.gz*。

现在，在运行 PyTorch 的实例或笔记本上，下载该 tar 文件并解压缩。我们需要重命名一些文件，使这些权重与 GPT-2 的 Hugging Face 重新实现兼容：

```py
mv encoder.json vocab.json
mv vocab.bpe merges.txt
```

现在我们需要将保存的 TensorFlow 权重转换为与 PyTorch 兼容的权重。方便的是，`pytorch-transformers`存储库附带了一个脚本来执行此操作：

```py
 python [REPO_DIR]/pytorch_transformers/convert_gpt2_checkpoint_to_pytorch.py
 --gpt2_checkpoint_path [SAVED_TENSORFLOW_MODEL_DIR]
 --pytorch_dump_folder_path [SAVED_TENSORFLOW_MODEL_DIR]
```

然后可以在代码中创建一个新的 GPT-2 模型实例：

```py
from pytorch_transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained([SAVED_TENSORFLOW_MODEL_DIR])
```

或者，只是为了玩弄模型，您可以使用*run_gpt2.py*脚本获得一个提示，输入文本并从基于 PyTorch 的模型获取生成的样本：

```py
python [REPO_DIR]/pytorch-transformers/examples/run_gpt2.py
--model_name_or_path [SAVED_TENSORFLOW_MODEL_DIR]
```

随着 Hugging Face 在其存储库中整合所有模型的一致 API，训练 GPT-2 可能会变得更加容易，但目前使用 TensorFlow 方法是最容易入门的。

BERT 和 GPT-2 目前是文本学习中最流行的名称，但在我们结束之前，我们将介绍当前最先进模型中的黑马：ULMFiT。

## ULMFiT

与 BERT 和 GPT-2 这两个庞然大物相比，*ULMFiT*基于一个老式的 RNN。看不到 Transformer，只有 AWD-LSTM，这是由 Stephen Merity 最初创建的架构。在 WikiText-103 数据集上训练，它已被证明适合迁移学习，尽管是*老式*的架构，但在分类领域已被证明与 BERT 和 GPT-2 具有竞争力。

虽然 ULMFiT 本质上只是另一个可以像其他模型一样在 PyTorch 中加载和使用的模型，但它的自然家园是 fast.ai 库，该库位于 PyTorch 之上，并为快速掌握深度学习并快速提高生产力提供了许多有用的抽象。为此，我们将看看如何在 Twitter 数据集上使用 fast.ai 库中的 ULMFiT，该数据集在第五章中使用过。

我们首先使用 fast.ai 的 Data Block API 为微调 LSTM 准备数据：

```py
data_lm = (TextList
           .from_csv("./twitter-data/",
           'train-processed.csv', cols=5,
           vocab=data_lm.vocab)
           .split_by_rand_pct()
           .label_from_df(cols=0)
           .databunch())
```

这与第五章中的`torchtext`助手非常相似，只是产生了 fast.ai 称为`databunch`的东西，从中其模型和训练例程可以轻松获取数据。接下来，我们创建模型，但在 fast.ai 中，这种情况有些不同。我们创建一个`learner`，与之交互以训练模型，而不是模型本身，尽管我们将其作为参数传递。我们还提供了一个 dropout 值（我们使用了 fast.ai 培训材料中建议的值）：

```py
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
```

一旦我们有了`learner`对象，我们可以找到最佳学习率。这与我们在第四章中实现的类似，只是它内置在库中，并使用指数移动平均值来平滑图表，我们的实现中相当尖锐：

```py
learn.lr_find()
learn.recorder.plot()
```

从图 9-12 中的图表来看，`1e-2`是我们开始出现急剧下降的地方，因此我们将选择它作为我们的学习率。Fast.ai 使用一种称为`fit_one_cycle`的方法，它使用 1cycle 学习调度器（有关 1cycle 的更多详细信息，请参见“进一步阅读”），并使用非常高的学习率在数量级更少的时代内训练模型。

![ULMFiT 学习率图](img/ppdl_0912.png)

###### 图 9-12. ULMFiT 学习率图

在这里，我们只训练一个周期，并保存网络的微调头部（*编码器*）：

```py
learn.fit_one_cycle(1, 1e-2)
learn.save_encoder('twitter_encoder')
```

随着语言模型的微调完成（您可能希望在训练中尝试更多周期），我们为实际分类问题构建了一个新的`databunch`：

```py
twitter_classifier_bunch = TextList
           .from_csv("./twitter-data/",
           'train-processed.csv', cols=5,
           vocab=data_lm.vocab)
           .split_by_rand_pct()
           .label_from_df(cols=0)
           .databunch())
```

这里唯一的真正区别是，我们通过使用`label_from_df`提供实际标签，并且从之前执行的语言模型训练中传入一个`vocab`对象，以确保它们使用相同的单词到数字的映射，然后我们准备创建一个新的`text_classifier_learner`，其中库在幕后为您创建所有模型。我们将微调的编码器加载到这个新模型上，并开始再次进行训练过程：

```py
learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
```

通过少量代码，我们得到了一个报告准确率为 76%的分类器。我们可以通过训练语言模型更多周期，添加不同的学习率并在训练时冻结部分模型来轻松改进，所有这些都是 fast.ai 支持的，定义在`learner`上的方法。

## 使用什么？

在深度学习文本模型的当前前沿进行了一番快速的介绍后，您可能心中有一个问题：“这一切都很棒，但我应该实际*使用*哪一个？”一般来说，如果您正在处理分类问题，我建议您从 ULMFiT 开始。BERT 令人印象深刻，但在准确性方面，ULMFiT 与 BERT 竞争，并且它还有一个额外的好处，即您不需要购买大量的 TPU 积分来充分利用它。对于大多数人来说，单个 GPU 微调 ULMFiT 可能已经足够了。

至于 GPT-2，如果您想要生成的文本，那么是的，它更适合，但对于分类目的，接近 ULMFiT 或 BERT 的性能会更难。我认为可能有趣的一件事是让 GPT-2 在数据增强上自由发挥；如果您有一个类似 Sentiment140 的数据集，我们在整本书中一直在使用它，为什么不在该输入上微调一个 GPT-2 模型并使用它生成更多数据呢？

# 结论

本章介绍了 PyTorch 的更广泛世界，包括可以导入到自己项目中的现有模型的库，一些可应用于任何领域的尖端数据增强方法，以及可能破坏模型的对抗样本以及如何防御它们。希望当我们结束这段旅程时，你能理解神经网络是如何组装的，以及如何让图像、文本和音频作为张量流经它们。你应该能够训练它们，增强数据，尝试不同的学习率，并在模型出现问题时进行调试。一旦所有这些都完成了，你就知道如何将它们打包到 Docker 中，并让它们为更广泛的世界提供服务。

接下来我们去哪里？考虑查看 PyTorch 论坛和网站上的其他文档。我强烈推荐访问 fast.ai 社区，即使你最终不使用该库；这是一个充满活力的社区，充满了好主意和尝试新方法的人，同时也对新手友好！

跟上深度学习的前沿变得越来越困难。大多数论文都发表在[arXiv](https://arxiv.org)，但论文的发表速度似乎以近乎指数级增长；当我写这个结论时，[XLNet](https://arxiv.org/abs/1906.08237)刚刚发布，据说在各种任务上击败了 BERT。永无止境！为了帮助解决这个问题，我在这里列出了一些 Twitter 账号，人们经常推荐有趣的论文。我建议关注它们，以了解当前和有趣的工作，然后你可以使用工具如[arXiv Sanity Preserver](http://arxiv-sanity.com)来更轻松地深入研究。

最后，我在这本书上训练了一个 GPT-2 模型，它想说几句话：

> 深度学习*是我们如何处理当今深度学习应用的关键驱动力，预计深度学习将继续扩展到新领域，如基于图像的分类，在 2016 年，NVIDIA 推出了 CUDA LSTM 架构。随着 LSTM 变得越来越流行，LSTM 也成为了一种更便宜、更易于生产的用于研究目的的构建方法，而 CUDA 已经证明在深度学习市场上是一种非常有竞争力的架构。*

幸运的是，你可以看到在我们作者失业之前还有很长的路要走。但也许你可以帮助改变这一点！

# 进一步阅读

+   一份[当前超分辨率技术的调查](https://arxiv.org/pdf/1902.06068.pdf)

+   Ian Goodfellow 的[GAN 讲座](https://www.youtube.com/watch?v=Z6rxFNMGdn0)

+   [You Only Look Once (YOLO)](https://pjreddie.com/darknet/yolo) —— 一系列快速目标检测模型，具有非常易读的论文

+   [CleverHans](https://github.com/tensorflow/cleverhans) —— 一个为 TensorFlow 和 PyTorch 提供对抗生成技术的库

+   [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer) —— 通过 Transformer 架构进行深入探索

一些推荐关注的 Twitter 账号：

+   *@jeremyphoward* —— fast.ai 的联合创始人

+   *@miles_brundage* —— OpenAI 的研究科学家（政策）

+   *@BrundageBot* —— 一个每天生成有趣论文摘要的 Twitter 机器人（警告：通常每天推出 50 篇论文！）

+   *@pytorch* —— 官方 PyTorch 账号

¹ 请查看张宏毅等人（2017 年）的论文“混合：超越经验风险最小化”。

² 请查看 Ian J. Goodfellow 等人（2014 年）的论文“生成对抗网络”。

³ 请查看 Olaf Ronneberger 等人（2015 年）的论文“U-Net：用于生物医学图像分割的卷积网络”。

⁴ 请参阅 Ian Goodfellow 等人撰写的“解释和利用对抗样本”（2014 年）。

⁵ 请参阅 Dzmitry Bahdanau 等人撰写的“通过联合学习对齐和翻译进行神经机器翻译”（2014 年）。

⁶ 请参阅 Ashish Vaswani 等人撰写的“注意力机制就是一切”（2017 年）。
