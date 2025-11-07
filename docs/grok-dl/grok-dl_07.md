## 第八章。学习信号和忽略噪声：正则化和批处理简介

**本章内容**

+   过拟合

+   Dropout

+   批量梯度下降

> “用四个参数我可以画一头大象，用五个参数我甚至可以让它扭动它的鼻子。”
> 
> *约翰·冯·诺伊曼，数学家、物理学家、计算机科学家和通才*

### MNIST 上的三层网络

#### 让我们回到 MNIST 数据集，并尝试使用新的网络进行分类

在前面的几章中，你已经了解到神经网络建模相关性。隐藏层（三层网络中的中间层）甚至可以创建中间相关性来帮助解决问题（似乎是凭空而来）。你怎么知道网络正在创建好的相关性呢？

当我们讨论具有多个输入的随机梯度下降时，我们进行了一个实验，其中我们冻结了一个权重，然后要求网络继续训练。在训练过程中，点就像找到了碗底。你看到权重被调整以最小化错误。

当我们将权重冻结时，冻结的权重仍然找到了碗底。由于某种原因，碗移动了，使得冻结的权重值变得最优。此外，如果我们解冻权重进行更多训练，它就不会学习。为什么？嗯，错误已经下降到 0。对于网络来说，再也没有什么可以学习的了。

这就引出了一个问题，如果冻结的权重的输入对预测现实世界中的棒球胜利很重要怎么办？如果网络已经找到了一种准确预测训练数据集中比赛的方法（因为这就是网络所做的事情：最小化错误），但它却忘记了包含一个有价值的输入怎么办？

不幸的是，这种现象——过拟合——在神经网络中非常普遍。我们可以说它是神经网络的宿敌；神经网络的表达能力越强（更多层和权重），就越容易过拟合。在研究中，人们不断发现需要更强大层级的任务，但随后又必须进行大量问题解决以确保网络不过拟合。

在本章中，我们将研究正则化的基础知识，这是对抗神经网络过拟合的关键。为此，我们将从最强大的神经网络（具有`relu`隐藏层的三层网络）开始，在最具挑战性的任务（MNIST 数字分类）上进行。

首先，按照下面的步骤训练网络。你应该看到与列出的相同的结果。唉，网络学会了完美地预测训练数据。我们应该庆祝吗？

```
import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = (x_train[0:1000].reshape(1000,28*28) \
                                        255, y_train[0:1000])
one_hot_labels = np.zeros((len(labels),10))

for i,l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test),28*28) / 255
test_labels = np.zeros((len(y_test),10))
for i,l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)
relu = lambda x:(x>=0) * x        *1*
relu2deriv = lambda x: x>=0       *2*
alpha, iterations, hidden_size, pixels_per_image, num_labels = \
                                              (0.005, 350, 40, 784, 10)
weights_0_1 = 0.2*np.random.random((pixels_per_image,hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size,num_labels)) - 0.1

for j in range(iterations):
    error, correct_cnt = (0.0, 0)

    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        layer_2 = np.dot(layer_1,weights_1_2)
        error += np.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == \
                                        np.argmax(labels[i:i+1]))
        layer_2_delta = (labels[i:i+1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)\
                                    * relu2deriv(layer_1)
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    sys.stdout.write("\r"+ \
                     " I:"+str(j)+ \
                     " Error:" + str(error/float(len(images)))[0:5] +\
                     " Correct:" + str(correct_cnt/float(len(images))))
```

+   ***1* 当 x 大于 0 时返回 x；否则返回 0**

+   ***2* 当输入大于 0 时返回 1；否则返回 0**

```
....
I:349 Error:0.108 Correct:1.0
```

### 嗯，这很简单

#### 神经网络完美地学会了预测所有 1000 张图像

在某些方面，这是一个真正的胜利。神经网络能够从 1,000 张图像的数据集中学习，将每个输入图像与正确的标签相关联。

它是如何做到这一点的？它逐个遍历每张图像，做出预测，然后略微更新每个权重，以便下一次预测更好。在所有图像上这样做足够长时间后，网络最终达到了能够正确预测所有图像的状态。

这里有一个不那么明显的问题：神经网络在它之前从未见过的图像上的表现会怎样？换句话说，它在它训练的 1,000 张图像之外的图像上的表现会怎样？MNIST 数据集包含的图像比训练的 1,000 张图像多得多；让我们试试。

在之前的代码笔记本中有两个变量：`test_images` 和 `test_labels`。如果你执行以下代码，它将在这些图像上运行神经网络并评估网络对它们的分类效果：

```
if(j % 10 == 0 or j == iterations-1):
  error, correct_cnt = (0.0, 0)

  for i in range(len(test_images)):

        layer_0 = test_images[i:i+1]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        layer_2 = np.dot(layer_1,weights_1_2)

        error += np.sum((test_labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == \
                                        np.argmax(test_labels[i:i+1]))
  sys.stdout.write(" Test-Err:" + str(error/float(len(test_images)))[0:5] +\
             " Test-Acc:" + str(correct_cnt/float(len(test_images))))
  print()
```

```
Error:0.653 Correct:0.7073
```

网络的表现非常糟糕！它的准确率只有 70.7%。为什么它在这些新的测试图像上的表现如此糟糕，尽管它在训练数据上学会了以 100%的准确率进行预测？真奇怪。

这个 70.7%的数字被称为*测试准确率*。这是神经网络在它没有训练过的数据上的准确率。这个数字很重要，因为它模拟了如果你尝试在现实世界中使用神经网络（这给网络只有它之前没有见过的图像）时，神经网络的表现会怎样。这是重要的分数。

### 记忆与泛化

#### 记忆 1,000 张图像比泛化到所有图像要容易

让我们再次考虑神经网络是如何学习的。它调整每个矩阵中的每个权重，以便网络能够更好地处理*特定输入*并做出*特定预测*。也许一个更好的问题可能是，“如果我们用 1,000 张图像来训练它，它学会了完美预测，为什么它还要在其他图像上工作呢？”

如你所预期，当完全训练好的神经网络应用于新图像时，它只有在新的图像几乎与训练数据中的图像完全相同的情况下才能保证表现良好。为什么？因为神经网络学会了将输入数据转换为输出数据，只针对*非常特定的输入配置*。如果你给它一些看起来不熟悉的东西，它将随机预测。

这使得神经网络变得有点没有意义。一个只在你训练它的数据上工作的神经网络的目的是什么？你已经知道这些数据点的正确分类。神经网络只有在处理你不知道答案的数据时才有用。

事实上，有一种方法可以对抗这种情况。在这里，我已经打印出了神经网络在训练过程中（每 10 次迭代）的*训练*和*测试*准确率。你注意到什么有趣的东西了吗？你应该看到更好的网络的线索：

```
I:0 Train-Err:0.722 Train-Acc:0.537 Test-Err:0.601 Test-Acc:0.6488
I:10 Train-Err:0.312 Train-Acc:0.901 Test-Err:0.420 Test-Acc:0.8114
I:20 Train-Err:0.260 Train-Acc:0.93 Test-Err:0.414 Test-Acc:0.8111
I:30 Train-Err:0.232 Train-Acc:0.946 Test-Err:0.417 Test-Acc:0.8066
I:40 Train-Err:0.215 Train-Acc:0.956 Test-Err:0.426 Test-Acc:0.8019
I:50 Train-Err:0.204 Train-Acc:0.966 Test-Err:0.437 Test-Acc:0.7982
I:60 Train-Err:0.194 Train-Acc:0.967 Test-Err:0.448 Test-Acc:0.7921
I:70 Train-Err:0.186 Train-Acc:0.975 Test-Err:0.458 Test-Acc:0.7864
I:80 Train-Err:0.179 Train-Acc:0.979 Test-Err:0.466 Test-Acc:0.7817
I:90 Train-Err:0.172 Train-Acc:0.981 Test-Err:0.474 Test-Acc:0.7758
I:100 Train-Err:0.166 Train-Acc:0.984 Test-Err:0.482 Test-Acc:0.7706
I:110 Train-Err:0.161 Train-Acc:0.984 Test-Err:0.489 Test-Acc:0.7686
I:120 Train-Err:0.157 Train-Acc:0.986 Test-Err:0.496 Test-Acc:0.766
I:130 Train-Err:0.153 Train-Acc:0.99 Test-Err:0.502 Test-Acc:0.7622
I:140 Train-Err:0.149 Train-Acc:0.991 Test-Err:0.508 Test-Acc:0.758
                                ....
I:210 Train-Err:0.127 Train-Acc:0.998 Test-Err:0.544 Test-Acc:0.7446
I:220 Train-Err:0.125 Train-Acc:0.998 Test-Err:0.552 Test-Acc:0.7416
I:230 Train-Err:0.123 Train-Acc:0.998 Test-Err:0.560 Test-Acc:0.7372
I:240 Train-Err:0.121 Train-Acc:0.998 Test-Err:0.569 Test-Acc:0.7344
I:250 Train-Err:0.120 Train-Acc:0.999 Test-Err:0.577 Test-Acc:0.7316
I:260 Train-Err:0.118 Train-Acc:0.999 Test-Err:0.585 Test-Acc:0.729
I:270 Train-Err:0.117 Train-Acc:0.999 Test-Err:0.593 Test-Acc:0.7259
I:280 Train-Err:0.115 Train-Acc:0.999 Test-Err:0.600 Test-Acc:0.723
I:290 Train-Err:0.114 Train-Acc:0.999 Test-Err:0.607 Test-Acc:0.7196
I:300 Train-Err:0.113 Train-Acc:0.999 Test-Err:0.614 Test-Acc:0.7183
I:310 Train-Err:0.112 Train-Acc:0.999 Test-Err:0.622 Test-Acc:0.7165
I:320 Train-Err:0.111 Train-Acc:0.999 Test-Err:0.629 Test-Acc:0.7133
I:330 Train-Err:0.110 Train-Acc:0.999 Test-Err:0.637 Test-Acc:0.7125
I:340 Train-Err:0.109 Train-Acc:1.0 Test-Err:0.645 Test-Acc:0.71
I:349 Train-Err:0.108 Train-Acc:1.0 Test-Err:0.653 Test-Acc:0.7073
```

### 神经网络中的过拟合

#### 如果你过度训练神经网络，它可能会变得更差！

由于某种原因，*测试*准确率在前 20 次迭代中上升，然后在网络训练得越来越多时（在此期间*训练*准确率仍在提高）缓慢下降。这在神经网络中很常见。让我通过一个类比来解释这个现象。

想象你正在为一个常见的餐叉制作模具，但不是用它来制作其他叉子，而是想用它来识别某个特定的餐具是否是叉子。如果一个物体能放入模具，你就会得出结论说这个物体是叉子，如果不能，你就会得出结论说它*不是*叉子。

假设你开始制作这个模具，你从一个湿的粘土块和一个装满三叉叉子、勺子和刀的大桶开始。然后你反复将所有叉子压入模具的同一位置以创建一个轮廓，这有点像一团糊状的叉子。你反复将所有叉子放入粘土中，数百次。当你让粘土干燥后，你会发现没有勺子或刀能放入这个模具，但所有叉子都能。太棒了！你做到了。你正确地制作了一个只能适合叉子形状的模具。

但如果有人给你一个四叉叉子会发生什么？你看看你的模具，注意到粘土中有一个特定的三细叉轮廓。四叉叉子不合适。为什么？它仍然是一个叉子。

这是因为粘土没有在四叉叉子上塑形。它只塑形了三叉叉子。这样，粘土就*过度拟合*了，只能识别它“训练”过的形状类型。

这正是你刚才在神经网络中看到的相同现象。这甚至比你想象的更接近。一种看待神经网络权重的方法是将其视为一个高维形状。随着训练的进行，这个形状*塑造*着数据的形状，学习区分不同的模式。不幸的是，测试数据集中的图像与训练数据集中的模式*略有*不同。这导致网络在许多测试示例上失败。

一个更正式的过度拟合神经网络的定义是：一个神经网络学会了数据集中的*噪声*，而不是仅基于*真实信号*做出决策。

### 过度拟合的来源

#### 什么导致了神经网络过度拟合？

让我们稍微改变一下这个场景。再次想象那块新鲜的粘土（未成型的）。如果你只把一个叉子压进去会怎样？假设粘土非常厚，它不会有之前模具（印制多次）那么多的细节。因此，它只会是一个*非常一般的叉子形状*。这个形状可能与三叉和四叉的叉子都兼容，因为它仍然是一个模糊的印痕。

假设这个信息，随着你印制更多的叉子，模具在测试数据集上变得更差，因为它学到了更多关于训练数据集的详细信息。这导致它拒绝那些与它在训练数据中反复看到的哪怕是一点点不同的图像。

这些图像中的*详细信息*是什么，与测试数据不兼容？在分叉的类比中，它就是叉子上的叉数。在图像中，这通常被称为*噪声*。在现实中，它要复杂一些。考虑这两张狗的照片。

![](img/f0151-01.jpg)

任何使这些照片在捕捉“狗”的本质之外变得独特的东西都包含在*噪声*这个术语中。在左边的图片中，枕头和背景都是噪声。在右边的图片中，狗中间的空黑部分也是一种噪声。实际上，是边缘告诉你这是一只狗；中间的黑色区域并没有告诉你任何东西。在左边的图片中，狗的中间部分有狗的毛茸茸的质感和颜色，这有助于分类器正确地识别它。

你如何让神经网络只训练在*信号*（狗的本质）上，而忽略噪声（与分类无关的其他东西）？一种方法就是*提前停止*。结果证明，大量的噪声都存在于图像的细粒度细节中，而大部分的信号（对于物体）都存在于图像的一般形状和可能的颜色中。

### 最简单的正则化：提前停止

#### 当网络开始变差时停止训练

你如何让神经网络忽略细粒度细节，只捕捉数据中存在的通用信息（如狗的一般形状或 MNIST 数字的一般形状）？你不让网络训练足够长的时间来学习它。

在分叉模具的例子中，需要多次印制许多叉子才能创造出三叉叉的完美轮廓。最初几次的印制通常只能捕捉到叉子的浅轮廓。对于神经网络来说也是如此。因此，*提前停止*是成本最低的正则化形式，如果你处于困境中，它可能非常有效。

这把我们带到了本章的主题：*正则化*。正则化是模型对新数据点*泛化*的方法的一个子领域（而不是仅仅记住训练数据）。它是帮助神经网络学习信号并忽略噪声的方法的一个子集。在这种情况下，它是一套你可以使用的工具，以创建具有这些特性的神经网络。


**正则化**

正则化是用于鼓励学习模型泛化的方法的一个子集，通常通过增加模型学习训练数据细粒度细节的难度来实现。


接下来的问题可能是，你如何知道何时停止？唯一真正知道的方法是将模型运行在不在训练数据集中的数据上。这通常是通过使用第二个测试数据集，称为*验证集*来完成的。在某些情况下，如果你使用测试集来决定何时停止，你可能会对测试集*过拟合*。一般来说，你不使用它来控制训练。相反，你使用验证集。

### 行业标准正则化：Dropout

#### 方法：在训练过程中随机关闭神经元（将它们设置为 0）

这种正则化技术听起来很简单。在训练过程中，你随机将网络中的神经元设置为 0（通常是在反向传播期间同一节点的 delta，但技术上你不必这样做）。这导致神经网络仅使用神经网络的*随机子集*进行训练。

信不信由你，这种正则化技术通常被广泛接受为大多数网络的首选、最先进的正则化技术。其方法简单且成本低廉，尽管它背后的*为什么*它有效的原因要复杂一些。

| |
| --- |

**为什么 Dropout 有效（可能过于简化**）

Dropout 通过随机训练网络的小部分子集，每次只训练一小部分，使得大网络表现得像一个小网络，而小网络不会过拟合。

| |
| --- |

结果表明，神经网络越小，它越不容易过拟合。为什么？好吧，小神经网络没有太多的表达能力。它们不能抓住更细粒度的细节（噪声），这些细节往往是过拟合的来源。它们只能捕捉到大的、明显的、高级特征。

这个关于*空间*或*容量*的概念非常重要，需要牢记在心。可以这样想。还记得粘土的比喻吗？想象一下，如果粘土是由像一角硬币大小的粘土石头制成的，那么这种粘土能够做出一个好的叉子印吗？当然不能。这些石头就像权重。它们围绕着数据形成，捕捉你感兴趣的模式。如果你只有几个较大的石头，它们就不能捕捉细微的细节。每个石头基本上是由叉子的大部分推动，或多或少地*平均*形状（忽略细小的褶皱和角落）。

现在，想象一下由非常细小的沙子制成的粘土。它由数百万个小石头组成，可以填充叉子的每一个角落。这就是大神经网络通常用来对数据集过拟合的表达能力。

如何获得大神经网络的强大能力，同时具有小神经网络的抗过拟合能力？将大神经网络中的节点随机关闭。当你将一个大神经网络只使用其中的一小部分时，它会表现得像一个小神经网络。但是，当你随机地在数百万个不同的子网络中这样做时，整个网络的总体表达能力仍然保持不变。这不是很酷吗？

### Dropout 为什么有效：集成工作

#### Dropout 是一种训练多个网络并取平均的方法

需要记住的是：神经网络总是从随机状态开始。这有什么关系呢？因为神经网络通过试错来学习，这最终意味着每个神经网络的学习可能同样有效，但没有任何两个神经网络是完全相同的（除非它们由于某种随机或故意的原因从完全相同的状态开始）。

这有一个有趣的特点。当你过拟合两个神经网络时，没有两个神经网络会以完全相同的方式过拟合。过拟合只会持续到每个训练图像都可以被完美预测，此时错误率等于 0，网络停止学习（即使你继续迭代）。但由于每个神经网络都是从随机预测开始，然后调整其权重以做出更好的预测，因此每个网络不可避免地会犯不同的错误，导致不同的更新。这最终导致一个核心概念：


虽然大型、未正则化的神经网络可能会过拟合噪声，但它们不太可能过拟合到**相同的**噪声。


为什么它们不会过拟合到相同的噪声呢？因为它们是从随机状态开始的，一旦它们学会了足够多的噪声来区分训练集中的所有图像，就会停止训练。MNIST 网络只需要找到几个随机像素，这些像素恰好与输出标签相关联，以实现过拟合。但与此相对的是，也许一个更加重要的概念：


神经网络，尽管它们是随机生成的，但仍然首先学习最大的、最广泛的特点，然后再学习关于噪声的更多内容。


吸取的教训是：如果你训练 100 个神经网络（所有初始化都是随机的），它们各自倾向于锁定不同的噪声，但具有相似的广泛**信号**。因此，当它们犯错时，它们通常会犯**不同的**错误。如果你允许它们平等投票，它们的噪声往往会相互抵消，只揭示它们共同学习的内容：**信号**。

### Dropout 在代码中的应用

#### 这里是如何在现实世界中使用 dropout 的

在 MNIST 分类模型中，让我们在隐藏层中添加 dropout，这样在训练过程中将有 50%的节点被随机关闭。你可能惊讶地发现这只是在代码中做了三行改动。以下是来自之前神经网络逻辑的一个熟悉的片段，其中添加了 dropout 掩码：

```
i = 0
layer_0 = images[i:i+1]
dropout_mask = np.random.randint(2,size=layer_1.shape)

layer_1 *= dropout_mask * 2
layer_2 = np.dot(layer_1, weights_1_2)

error += np.sum((labels[i:i+1] - layer_2) ** 2)

correct_cnt += int(np.argmax(layer_2) == \
             np.argmax(labels[i+i+1]))

layer_2_delta = (labels[i:i+1] - layer_2)
layer_1_delta = layer_2_delta.dot(weights_1_2.T)\
             * relu2deriv(layer_1)

layer_1_delta *= dropout_mask

weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
```

要在层（在这种情况下，`layer_1`）上实现 dropout，将`layer_1`的值乘以一个由 1s 和 0s 组成的随机矩阵。这会随机关闭`layer_1`中的节点，将它们设置为等于 0。请注意，`dropout_mask`使用所谓的**50%伯努利分布**，这意味着 50%的时间，`dropout_mask`中的每个值都是 1，而（1 - 50% = 50%）的时间，它是 0。

接下来是可能显得有些奇特的事情。你将`layer_1`乘以 2。你为什么要这样做？记住，`layer_2`将对`layer_1`执行加权求和。尽管它是加权的，但它仍然是对`layer_1`的值的求和。如果你关闭`layer_1`中一半的节点，这个和将减半。因此，`layer_2`将增加对`layer_1`的敏感性，有点像当音量太低而听不清楚时，一个人会靠近收音机。但在测试时间，当你不再使用 dropout 时，音量会恢复到正常。这会干扰`layer_2`监听`layer_1`的能力。你需要通过将`layer_1`乘以（1 / 打开的节点百分比）来对此进行对抗。在这种情况下，那就是 1/0.5，等于 2。这样，`layer_1`在训练和测试时的音量是相同的，尽管有 dropout。

```
import numpy, sys
np.random.seed(1)
def relu(x):
   return (x >= 0) * x      *1*

def relu2deriv(output):
   return output >= 0       *2*

alpha, iterations, hidden_size = (0.005, 300, 100)
pixels_per_image, num_labels = (784, 10)

weights_0_1 = 0.2*np.random.random((pixels_per_image,hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size,num_labels)) - 0.1

for j in range(iterations):
   error, correct_cnt = (0.0,0)
   for i in range(len(images)):
      layer_0 = images[i:i+1]
      layer_1 = relu(np.dot(layer_0,weights_0_1))
      dropout_mask = np.random.randint(2, size=layer_1.shape)
      layer_1 *= dropout_mask * 2
      layer_2 = np.dot(layer_1,weights_1_2)

      error += np.sum((labels[i:i+1] - layer_2) ** 2)
      correct_cnt += int(np.argmax(layer_2) == \
                                      np.argmax(labels[i:i+1]))
      layer_2_delta = (labels[i:i+1] - layer_2)
      layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
      layer_1_delta *= dropout_mask

      weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
      weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

   if(j%10 == 0):
      test_error = 0.0
      test_correct_cnt = 0

      for i in range(len(test_images)):
           layer_0 = test_images[i:i+1]
           layer_1 = relu(np.dot(layer_0,weights_0_1))
           layer_2 = np.dot(layer_1, weights_1_2)

           test_error += np.sum((test_labels[i:i+1] - layer_2) ** 2)
           test_correct_cnt += int(np.argmax(layer_2) == \
                                     np.argmax(test_labels[i:i+1]))

      sys.stdout.write("\n" + \
           "I:" + str(j) + \
           " Test-Err:" + str(test_error/ float(len(test_images)))[0:5] +\
           " Test-Acc:" + str(test_correct_cnt/ float(len(test_images)))+\
           " Train-Err:" + str(error/ float(len(images)))[0:5] +\
           " Train-Acc:" + str(correct_cnt/ float(len(images))))
```

+   ***1* 如果 x 大于 0 则返回 x；否则返回 0**

+   ***2* 对于输入大于 0 的情况返回 1**

### 在 MNIST 上评估 Dropout

如果你记得之前的内容，没有 dropout 的神经网络之前在测试准确率达到 81.14%后下降到 70.73%的准确率完成训练。当你添加 dropout 时，神经网络反而表现出这种行为：

```
I:0 Test-Err:0.641 Test-Acc:0.6333 Train-Err:0.891 Train-Acc:0.413
I:10 Test-Err:0.458 Test-Acc:0.787 Train-Err:0.472 Train-Acc:0.764
I:20 Test-Err:0.415 Test-Acc:0.8133 Train-Err:0.430 Train-Acc:0.809
I:30 Test-Err:0.421 Test-Acc:0.8114 Train-Err:0.415 Train-Acc:0.811
I:40 Test-Err:0.419 Test-Acc:0.8112 Train-Err:0.413 Train-Acc:0.827
I:50 Test-Err:0.409 Test-Acc:0.8133 Train-Err:0.392 Train-Acc:0.836
I:60 Test-Err:0.412 Test-Acc:0.8236 Train-Err:0.402 Train-Acc:0.836
I:70 Test-Err:0.412 Test-Acc:0.8033 Train-Err:0.383 Train-Acc:0.857
I:80 Test-Err:0.410 Test-Acc:0.8054 Train-Err:0.386 Train-Acc:0.854
I:90 Test-Err:0.411 Test-Acc:0.8144 Train-Err:0.376 Train-Acc:0.868
I:100 Test-Err:0.411 Test-Acc:0.7903 Train-Err:0.369 Train-Acc:0.864
I:110 Test-Err:0.411 Test-Acc:0.8003 Train-Err:0.371 Train-Acc:0.868
I:120 Test-Err:0.402 Test-Acc:0.8046 Train-Err:0.353 Train-Acc:0.857
I:130 Test-Err:0.408 Test-Acc:0.8091 Train-Err:0.352 Train-Acc:0.867
I:140 Test-Err:0.405 Test-Acc:0.8083 Train-Err:0.355 Train-Acc:0.885
I:150 Test-Err:0.404 Test-Acc:0.8107 Train-Err:0.342 Train-Acc:0.883
I:160 Test-Err:0.399 Test-Acc:0.8146 Train-Err:0.361 Train-Acc:0.876
I:170 Test-Err:0.404 Test-Acc:0.8074 Train-Err:0.344 Train-Acc:0.889
I:180 Test-Err:0.399 Test-Acc:0.807 Train-Err:0.333 Train-Acc:0.892
I:190 Test-Err:0.407 Test-Acc:0.8066 Train-Err:0.335 Train-Acc:0.898
I:200 Test-Err:0.405 Test-Acc:0.8036 Train-Err:0.347 Train-Acc:0.893
I:210 Test-Err:0.405 Test-Acc:0.8034 Train-Err:0.336 Train-Acc:0.894
I:220 Test-Err:0.402 Test-Acc:0.8067 Train-Err:0.325 Train-Acc:0.896
I:230 Test-Err:0.404 Test-Acc:0.8091 Train-Err:0.321 Train-Acc:0.894
I:240 Test-Err:0.415 Test-Acc:0.8091 Train-Err:0.332 Train-Acc:0.898
I:250 Test-Err:0.395 Test-Acc:0.8182 Train-Err:0.320 Train-Acc:0.899
I:260 Test-Err:0.390 Test-Acc:0.8204 Train-Err:0.321 Train-Acc:0.899
I:270 Test-Err:0.382 Test-Acc:0.8194 Train-Err:0.312 Train-Acc:0.906
I:280 Test-Err:0.396 Test-Acc:0.8208 Train-Err:0.317 Train-Acc:0.9
I:290 Test-Err:0.399 Test-Acc:0.8181 Train-Err:0.301 Train-Acc:0.908
```

不仅网络在得分 82.36%时达到峰值，它也没有那么严重地过拟合，最终以 81.81%的测试准确率完成训练。注意，dropout 也减缓了`Training-Acc`的上升速度，之前它直接升到 100%并保持在那里。

这应该指向 dropout 真正是什么：它是噪声。它使网络在训练数据上训练变得更加困难。这就像在腿上绑着重物跑马拉松。训练起来更难，但当你为一场难度更大的比赛脱下重物时，你最终会跑得相当快，因为你训练的是一件更困难的事情。

### 批量梯度下降

#### 这里有一种提高训练速度和收敛率的方法

在本章的背景下，我想简要应用几章前引入的一个概念：小批量随机梯度下降。我不会过多地详细介绍，因为这主要是神经网络训练中被默认接受的东西。此外，它是一个简单的概念，即使是最先进的神经网络也不会变得更加复杂。

之前我们一次训练一个训练示例，在每个示例之后更新权重。现在，让我们一次训练 100 个训练示例，对所有 100 个示例的平均权重更新进行平均。接下来显示训练/测试输出，然后是训练逻辑的代码。

```
I:0 Test-Err:0.815 Test-Acc:0.3832 Train-Err:1.284 Train-Acc:0.165
I:10 Test-Err:0.568 Test-Acc:0.7173 Train-Err:0.591 Train-Acc:0.672
I:20 Test-Err:0.510 Test-Acc:0.7571 Train-Err:0.532 Train-Acc:0.729
I:30 Test-Err:0.485 Test-Acc:0.7793 Train-Err:0.498 Train-Acc:0.754
I:40 Test-Err:0.468 Test-Acc:0.7877 Train-Err:0.489 Train-Acc:0.749
I:50 Test-Err:0.458 Test-Acc:0.793 Train-Err:0.468 Train-Acc:0.775
I:60 Test-Err:0.452 Test-Acc:0.7995 Train-Err:0.452 Train-Acc:0.799
I:70 Test-Err:0.446 Test-Acc:0.803 Train-Err:0.453 Train-Acc:0.792
I:80 Test-Err:0.451 Test-Acc:0.7968 Train-Err:0.457 Train-Acc:0.786
I:90 Test-Err:0.447 Test-Acc:0.795 Train-Err:0.454 Train-Acc:0.799
I:100 Test-Err:0.448 Test-Acc:0.793 Train-Err:0.447 Train-Acc:0.796
I:110 Test-Err:0.441 Test-Acc:0.7943 Train-Err:0.426 Train-Acc:0.816
I:120 Test-Err:0.442 Test-Acc:0.7966 Train-Err:0.431 Train-Acc:0.813
I:130 Test-Err:0.441 Test-Acc:0.7906 Train-Err:0.434 Train-Acc:0.816
I:140 Test-Err:0.447 Test-Acc:0.7874 Train-Err:0.437 Train-Acc:0.822
I:150 Test-Err:0.443 Test-Acc:0.7899 Train-Err:0.414 Train-Acc:0.823
I:160 Test-Err:0.438 Test-Acc:0.797 Train-Err:0.427 Train-Acc:0.811
I:170 Test-Err:0.440 Test-Acc:0.7884 Train-Err:0.418 Train-Acc:0.828
I:180 Test-Err:0.436 Test-Acc:0.7935 Train-Err:0.407 Train-Acc:0.834
I:190 Test-Err:0.434 Test-Acc:0.7935 Train-Err:0.410 Train-Acc:0.831
I:200 Test-Err:0.435 Test-Acc:0.7972 Train-Err:0.416 Train-Acc:0.829
I:210 Test-Err:0.434 Test-Acc:0.7923 Train-Err:0.409 Train-Acc:0.83
I:220 Test-Err:0.433 Test-Acc:0.8032 Train-Err:0.396 Train-Acc:0.832
I:230 Test-Err:0.431 Test-Acc:0.8036 Train-Err:0.393 Train-Acc:0.853
I:240 Test-Err:0.430 Test-Acc:0.8047 Train-Err:0.397 Train-Acc:0.844
I:250 Test-Err:0.429 Test-Acc:0.8028 Train-Err:0.386 Train-Acc:0.843
I:260 Test-Err:0.431 Test-Acc:0.8038 Train-Err:0.394 Train-Acc:0.843
I:270 Test-Err:0.428 Test-Acc:0.8014 Train-Err:0.384 Train-Acc:0.845
I:280 Test-Err:0.430 Test-Acc:0.8067 Train-Err:0.401 Train-Acc:0.846
I:290 Test-Err:0.428 Test-Acc:0.7975 Train-Err:0.383 Train-Acc:0.851
```

注意，训练准确率比之前更加平滑。持续进行平均权重更新会在训练过程中产生这种现象。实际上，单个训练示例在生成的权重更新方面非常嘈杂。因此，平均它们会使学习过程更加平滑。

```
import numpy as np
np.random.seed(1)

def relu(x):
    return (x >= 0) * x       *1*

def relu2deriv(output):
    return output >= 0        *2*

batch_size = 100
alpha, iterations = (0.001, 300)
pixels_per_image, num_labels, hidden_size = (784, 10, 100)

weights_0_1 = 0.2*np.random.random((pixels_per_image,hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size,num_labels)) - 0.1

for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size),((i+1)*batch_size))

        layer_0 = images[batch_start:batch_end]
        layer_1 = relu(np.dot(layer_0,weights_0_1))
        dropout_mask = np.random.randint(2,size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = np.dot(layer_1,weights_1_2)

        error += np.sum((labels[batch_start:batch_end] - layer_2) ** 2)
        for k in range(batch_size):
            correct_cnt += int(np.argmax(layer_2[k:k+1]) == \
                    np.argmax(labels[batch_start+k:batch_start+k+1]))

            layer_2_delta = (labels[batch_start:batch_end]-layer_2) \
                                                            /batch_size
            layer_1_delta = layer_2_delta.dot(weights_1_2.T)* \
                                                     relu2deriv(layer_1)
            layer_1_delta *= dropout_mask

            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if(j%10 == 0):
        test_error = 0.0
        test_correct_cnt = 0

        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1]
            layer_1 = relu(np.dot(layer_0,weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)
```

+   ***1* 如果 x 大于 0 则返回 x**

+   ***2* 当输入大于 0 时返回 1**

当你运行这段代码时，你首先会注意到它运行得更快。这是因为每个`np.dot`函数现在一次执行 100 个向量点积。CPU 架构以这种方式批量执行点积要快得多。

然而，这里还有更多的事情发生。请注意，`alpha`现在比之前大 20 倍。你可以出于一个有趣的原因增加它。想象一下，如果你试图使用一个非常不稳定的指南针找到一个城市。如果你低头看了一下，得到了一个航向，然后跑了 2 英里，你很可能会偏离航线。但如果你低头看了 100 次航向，然后取平均值，跑 2 英里可能会让你大致朝正确的方向前进。

因为这个例子取了一个噪声信号的平均值（100 个训练示例中的平均权重变化），它可以采取更大的步长。你通常会看到批量大小从 8 到高达 256。通常，研究人员会随机选择数字，直到他们找到一个似乎工作良好的`batch_size`/`alpha`对。

### 摘要

本章讨论了两种几乎适用于任何神经网络架构以提高准确性和训练速度的最常用方法。在接下来的章节中，我们将从适用于几乎所有神经网络的通用工具集转向针对特定类型现象建模的有利架构。
