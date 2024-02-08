# 第5章。卷积神经网络

在本章中，我们将介绍卷积神经网络（CNNs）。CNNs是用于预测的标准神经网络架构，当输入观察数据是图像时，这在各种神经网络应用中都是适用的情况。到目前为止，在本书中，我们专注于全连接神经网络，我们将其实现为一系列`Dense`层。因此，我们将通过回顾这些网络的一些关键元素来开始本章，并用此来激发我们为什么可能想要为图像使用不同的架构。然后，我们将以与本书中介绍其他概念类似的方式介绍CNNs：我们将首先讨论它们在高层次上的工作原理，然后转向在低层次上讨论它们，最后通过编写从头开始的卷积操作的代码详细展示它们的工作方式。到本章结束时，您将对CNNs的工作原理有足够深入的理解，能够使用它们来解决问题，并自行学习高级CNN变体，如ResNets、DenseNets和Octave Convolutions。

# 神经网络和表示学习

神经网络最初接收关于观察数据的信息，每个观察数据由一些特征的数量*n*表示。到目前为止，我们在两个非常不同的领域中看到了两个例子：第一个是房价数据集，每个观察数据由13个特征组成，每个特征代表房屋的某个数值特征。第二个是手写数字的MNIST数据集；由于图像用784个像素表示（28像素宽，28像素高），每个观察数据由784个数值表示，指示每个像素的明暗程度。

在每种情况下，经过适当缩放数据后，我们能够构建一个模型，以高准确度预测该数据集的适当结果。在每种情况下，一个具有一个隐藏层的简单神经网络模型表现比没有隐藏层的模型更好。为什么呢？一个原因，正如我在房价数据的案例中所展示的，是神经网络可以学习输入和输出之间的*非线性*关系。然而，一个更一般的原因是，在机器学习中，我们通常需要我们原始特征的*线性组合*来有效地预测我们的目标。假设MNIST数字的像素值为*x*[1]到*x*[784]。例如，可能情况是，*x*[1]高于平均值，*x*[139]低于平均值，*并且* *x*[237]也低于平均值的组合强烈预测图像将是数字9。可能还有许多其他这样的组合，所有这些组合都对图像是特定数字的概率产生积极或消极的影响。神经网络可以通过训练过程自动*发现*原始特征的重要组合。这个学习过程从通过乘以一个随机权重矩阵创建最初的随机原始特征组合开始；通过训练，神经网络学会了改进有用的组合并丢弃无用的组合。学习哪些特征组合是重要的这个过程被称为*表示学习*，这也是神经网络在不同领域成功的主要原因。这在[图5-1](#fig_05_01)中总结。

![神经网络图示](assets/dlfs_0501.png)

###### 图5-1。到目前为止我们看到的神经网络从<math><mi>n</mi></math>个特征开始，然后学习介于<math><msqrt><mi>n</mi></msqrt></math>和<math><mi>n</mi></math>之间的“组合”来进行预测

是否有理由修改这个过程以适应图像数据？提示答案是“是”的基本见解是，在图像中，有趣的“特征组合”（像素）往往来自图像中彼此靠近的像素。在图像中，有趣的特征不太可能来自整个图像中随机选择的9个像素的组合，而更可能来自相邻像素的3×3补丁。我们想要利用关于图像数据的这一基本事实：特征的顺序很重要，因为它告诉我们哪些像素在空间上彼此靠近，而在房价数据中，特征的顺序并不重要。但是我们该如何做呢？

## 图像数据的不同架构

在高层次上，解决方案将是创建特征的组合，就像以前一样，但是数量级更多，并且每个特征只是输入图像中一个小矩形补丁的像素的组合。图5-2描述了这一点。

![神经网络图](assets/dlfs_0502.png)

###### 图5-2。对于图像数据，我们可以将每个学习到的特征定义为数据的一个小补丁的函数，因此可以定义介于n和n²之间的输出神经元

让我们的神经网络学习*所有*输入特征的组合——也就是说，输入图像中*所有*像素的组合——事实证明是非常低效的，因为它忽略了前一节描述的见解：图像中有趣的特征组合大多出现在这些小补丁中。尽管如此，以前至少非常容易计算新特征，这些特征是所有输入特征的组合：如果我们有*f*个输入特征，并且想要计算*n*个新特征，我们只需将包含我们输入特征的`ndarray`乘以一个`f`×`n`矩阵。我们可以使用什么操作来计算输入图像的局部补丁中像素的许多组合？答案是卷积操作。

## 卷积操作

在描述卷积操作之前，让我们明确一下我们所说的“来自图像局部补丁的像素组合的特征”是什么意思。假设我们有一个5×5的输入图像*I*：

<math display="block"><mrow><mi>I</mi> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><msub><mi>i</mi> <mn>11</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>12</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>13</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>14</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>15</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>i</mi> <mn>21</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>22</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>23</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>24</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>25</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>i</mi> <mn>31</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>32</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>33</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>34</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>35</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>i</mi> <mn>41</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>42</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>43</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>44</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>45</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>i</mi> <mn>51</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>52</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>53</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>54</mn></msub></mtd> <mtd><msub><mi>i</mi> <mn>55</mn></msub></mtd></mtr></mtable></mfenced></mrow></math>

假设我们想计算一个新特征，它是中间3×3像素补丁的函数。就像我们迄今所见的神经网络中将新特征定义为旧特征的线性组合一样，我们将定义一个新特征，它是这个3×3补丁的函数，我们将通过定义一个3×3的权重*W*来实现：

<math display="block"><mrow><mi>w</mi> <mo>=</mo> <mfenced close="]" open="["><mtable><mtr><mtd><msub><mi>w</mi> <mn>11</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>12</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>13</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>w</mi> <mn>21</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>22</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>23</mn></msub></mtd></mtr> <mtr><mtd><msub><mi>w</mi> <mn>31</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>32</mn></msub></mtd> <mtd><msub><mi>w</mi> <mn>33</mn></msub></mtd></mtr></mtable></mfenced></mrow></math>

然后，我们将简单地将*W*与*I*中相关补丁的点积，以获取输出中特征的值，由于涉及的输入图像部分位于(3,3)处，我们将表示为*o*[33]（*o*代表“输出”）：

<math display="block"><mrow><msub><mi>o</mi> <mn>33</mn></msub> <mo>=</mo> <msub><mi>w</mi> <mn>11</mn></msub> <mo>×</mo> <msub><mi>i</mi> <mn>22</mn></msub> <mo>+</mo> <msub><mi>w</mi> <mn>12</mn></msub> <mo>×</mo> <msub><mi>i</mi> <mn>23</mn></msub> <mo>+</mo> <msub><mi>w</mi> <mn>13</mn></msub> <mo>×</mo> <msub><mi>i</mi> <mn>24</mn></msub> <mo>+</mo> <msub><mi>w</mi> <mn>21</mn></msub> <mo>×</mo> <msub><mi>i</mi> <mn>32</mn></msub> <mo>+</mo> <msub><mi>w</mi> <mn>22</mn></msub> <mo>×</mo> <msub><mi>i</mi> <mn>33</mn></msub> <mo>+</mo> <msub><mi>w</mi> <mn>23</mn></msub> <mo>×</mo> <msub><mi>i</mi> <mn>34</mn></msub> <mo>+</mo> <msub><mi>w</mi> <mn>31</mn></msub> <mo>×</mo> <msub><mi>i</mi> <mn>42</mn></msub> <mo>+</mo> <msub><mi>w</mi> <mn>32</mn></msub> <mo>×</mo> <msub><mi>i</mi> <mn>43</mn></msub> <mo>+</mo> <msub><mi>w</mi> <mn>33</mn></msub> <mo>×</mo> <msub><mi>i</mi> <mn>44</mn></msub></mrow></math>

然后，这个值将被视为我们在神经网络中看到的其他计算特征：可能会添加一个偏差，然后可能会通过激活函数，然后它将代表一个“神经元”或“学习到的特征”，将被传递到网络的后续层。因此，我们可以定义特征，这些特征是输入图像的小补丁的函数。

我们应该如何解释这些特征？事实证明，以这种方式计算的特征有一个特殊的解释：它们表示权重定义的*视觉模式*是否存在于图像的该位置。当将3×3或5×5的数字数组与图像的每个位置的像素值进行点积时，它们可以表示“模式检测器”，这在计算机视觉领域已经很久了。例如，将以下3×3数字数组进行点积：

<math display="block"><mfenced close="]" open="["><mtable><mtr><mtd><mn>0</mn></mtd> <mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd></mtr> <mtr><mtd><mn>1</mn></mtd> <mtd><mrow><mo>-</mo> <mn>4</mn></mrow></mtd> <mtd><mn>1</mn></mtd></mtr> <mtr><mtd><mn>0</mn></mtd> <mtd><mn>1</mn></mtd> <mtd><mn>0</mn></mtd></mtr></mtable></mfenced></math>

对于输入图像的给定部分，检测该图像位置是否存在边缘。已知有类似的矩阵可以检测角是否存在，垂直或水平线是否存在，等等。

现在假设我们使用*相同的权重集* *W* 来检测输入图像的每个位置是否存在由*W*定义的视觉模式。我们可以想象“在输入图像上滑动*W*”，将*W*与图像每个位置的像素进行点积，最终得到一个几乎与原始图像大小相同的新图像*O*（可能略有不同，取决于我们如何处理边缘）。这个图像*O*将是一种“特征图”，显示了输入图像中*W*定义的模式存在的位置。这实际上就是卷积神经网络中发生的操作；它被称为*卷积*，其输出确实被称为*特征图*。

这个操作是CNN如何工作的核心。在我们可以将其整合到我们在前几章中看到的那种完整的`Operation`之前，我们必须为其添加另一个维度-字面上。

## 多通道卷积操作

回顾一下：卷积神经网络与常规神经网络的不同之处在于它们创建了数量级更多的特征，并且每个特征仅是来自输入图像的一个小块的函数。现在我们可以更具体：从*n*个输入像素开始，刚刚描述的卷积操作将为输入图像中的每个位置创建*n*个输出特征。在神经网络的卷积`Layer`中实际发生的事情更进一步：在那里，我们将创建*f*组*n*个特征，*每个*都有一个对应的（最初是随机的）权重集，定义了在输入图像的每个位置检测到的视觉模式，这将在特征图中捕获。这*f*个特征图将通过*f*个卷积操作创建。这在[图5-3](#fig_05_03)中有所体现。

![神经网络图示](assets/dlfs_0503.png)

###### 图5-3。比以前更具体，对于具有n个像素的输入图像，我们定义一个输出，其中包含f个特征图，每个特征图的大小与原始图像大致相同，总共有n×f个输出神经元用于图像，每个神经元仅是原始图像的一个小块的函数

现在我们已经介绍了一堆概念，让我们为了清晰起见对它们进行定义。在卷积`Layer`的上下文中，每个由特定权重集检测到的“特征集”称为特征图，特征图的数量在卷积`Layer`中被称为`Layer`的*通道数*，这就是为什么与`Layer`相关的操作被称为多通道卷积。此外，*f*组权重*W*[*i*]被称为卷积*滤波器*。

# 卷积层

现在我们了解了多通道卷积操作，我们可以考虑如何将这个操作整合到神经网络层中。以前，我们的神经网络层相对简单：它们接收二维的`ndarray`作为输入，并产生二维的`ndarray`作为输出。然而，根据前一节的描述，卷积层将会为*单个图像*产生一个三维的`ndarray`作为输出，维度为*通道数*（与“特征图”相同）×*图像高度*×*图像宽度*。

这引发了一个问题：我们如何将这个`ndarray`前向传递到另一个卷积层中，以创建一个“深度卷积”神经网络？我们已经看到如何在具有单个通道和我们的滤波器的图像上执行卷积操作；当两个卷积层串联时，我们如何在*具有多个*通道的*输入*上执行多通道卷积？理解这一点是理解*深度*卷积神经网络的关键。

考虑在具有全连接层的神经网络中会发生什么：在第一个隐藏层中，我们有，假设，*h*[1]个特征，这些特征是来自输入层的所有原始特征的组合。在接下来的层中，特征是来自前一层的所有特征的组合，因此我们可能有*h*[2]个原始特征的“特征的特征”。为了创建这一层的*h*[2]个特征，我们使用<math><mrow><msub><mi>h</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>h</mi> <mn>2</mn></msub></mrow></math>个权重来表示*h*[2]个特征中的每一个都是前一层中的*h*[1]个特征的函数。

如前一节所述，在卷积神经网络的第一层中会发生类似的过程：我们首先使用*m*[1]个卷积*滤波器*将输入图像转换为*m*[1]个*特征图*。我们应该将这一层的输出看作是表示权重的*m*[1]个滤波器在输入图像的每个位置是否存在的不同视觉模式。就像全连接神经网络的不同层可以包含不同数量的神经元一样，卷积神经网络的下一层可能包含*m*[2]个滤波器。为了使网络学习复杂的模式，每个滤波器的解释应该是在图像的每个位置是否存在前一层中*m*[1]个视觉模式的*组合*或更高阶视觉特征。这意味着如果卷积层的输出是一个形状为*m*[2]个通道×图像高度×图像宽度的3D `ndarray`，那么图像中一个给定位置上的*m*[2]个特征图中的一个是在前一层对应的*m*[1]个特征图的每个相同位置上卷积*m*[1]个不同的滤波器的线性组合。这将使得*m*[2]个滤波器图中的每个位置都表示前一卷积层中已学习的*m*[1]个视觉特征的*组合*。

## 实现影响

了解两个多通道卷积层如何连接告诉我们如何实现这个操作：正如我们需要<math><mrow><msub><mi>h</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>h</mi> <mn>2</mn></msub></mrow></math>个权重来连接一个具有*h*[1]个神经元的全连接层和一个具有<math><msub><mi>h</mi> <mn>2</mn></msub></math>个神经元的全连接层，我们需要<math><mrow><msub><mi>m</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>m</mi> <mn>2</mn></msub></mrow></math>个*卷积滤波器*来连接一个具有*m*[1]个通道的卷积层和一个具有*m*[2]个通道的卷积层。有了这个最后的细节，我们现在可以指定构成完整的多通道卷积操作的`ndarray`的维度，包括输入、输出和参数：

1.  输入的形状将是：

    +   批量大小

    +   输入通道

    +   图像高度

    +   图像宽度

1.  输出的形状将是：

    +   批量大小

    +   输出通道

    +   图像高度

    +   图像宽度

1.  卷积滤波器本身的形状将是：

    +   输入通道

    +   输出通道

    +   滤波器高度

    +   滤波器宽度

###### 注意

维度的顺序可能因库而异，但这四个维度始终存在。

我们将在本章后面实现这个卷积操作时牢记所有这些。

## 卷积层和全连接层之间的差异

在本章的开头，我们讨论了卷积层和全连接层在高层次上的区别；[图5-4](#fig_05_04)重新审视了这种比较，现在我们已经更详细地描述了卷积层。

![神经网络图示](assets/dlfs_0504.png)

###### 图5-4。卷积层和全连接层之间的比较

此外，这两种层之间的最后一个区别是个体神经元本身的解释方式：

+   全连接层中每个神经元的解释是，它检测先前层学习的*特定特征组合*是否存在于当前观察中。

+   卷积层中的神经元的解释是，它检测先前层学习的*特定视觉模式组合*是否存在于输入图像的*给定位置*。

在我们将这样的层合并到神经网络之前，我们需要解决另一个问题：如何使用输出的多维数组来进行预测。

## 使用卷积层进行预测：Flatten层

我们已经讨论了卷积层如何学习代表图像中是否存在视觉模式的特征，并将这些特征存储在特征图层中；我们如何使用这些特征图层来进行预测呢？在上一章中使用全连接神经网络预测图像属于10个类别中的哪一个时，我们只需要确保最后一层的维度为10；然后我们可以将这10个数字输入到softmax交叉熵损失函数中，以确保它们被解释为概率。现在我们需要弄清楚在卷积层的情况下我们可以做什么，其中每个观察都有一个三维的形状为*m*通道数 × 图像高度 × 图像宽度的`ndarray`。

要找到答案，回想一下，每个神经元只是表示图像中是否存在特定视觉特征组合（如果这是一个深度卷积神经网络，则可能是特征的特征或特征的特征的特征）在图像的给定位置。这与如果我们将全连接神经网络应用于此图像时学习的特征没有区别：第一个全连接层将表示单个像素的特征，第二个将表示这些特征的特征，依此类推。在全连接架构中，我们只需将网络学习的每个“特征的特征”视为单个神经元，用作预测图像属于哪个类别的输入。

事实证明，我们可以用卷积神经网络做同样的事情——我们将*m*个特征图视为<math><mrow><mi>m</mi> <mo>×</mo> <mi>i</mi> <mi>m</mi> <mi>a</mi> <mi>g</mi> <msub><mi>e</mi> <mrow><mi>h</mi><mi>e</mi><mi>i</mi><mi>g</mi><mi>h</mi><mi>t</mi></mrow></msub> <mo>×</mo> <mi>i</mi> <mi>m</mi> <mi>a</mi> <mi>g</mi> <msub><mi>e</mi> <mrow><mi>w</mi><mi>i</mi><mi>d</mi><mi>t</mi><mi>h</mi></mrow></msub></mrow></math>个神经元，并使用`Flatten`操作将这三个维度（通道数、图像高度和图像宽度）压缩成一个一维向量，然后我们可以使用简单的矩阵乘法进行最终预测。这样做的直觉是，每个单独的神经元*基本上代表与全连接层中的神经元相同的“类型”*——具体来说，表示在图像的给定位置是否存在给定的视觉特征（或特征组合）——因此我们可以在神经网络的最后一层中以相同的方式处理它们。^([4](ch05.html#idm45732618457096))

我们将在本章后面看到如何实现`Flatten`层。但在我们深入实现之前，让我们讨论另一种在许多CNN架构中很重要的层，尽管本书不会详细介绍它。

## 池化层

*池化*层是卷积神经网络中常用的另一种类型的层。它们简单地对由卷积操作创建的每个特征图进行*下采样*；对于最常用的池化大小为2，这涉及将每个特征图的每个2×2部分映射到该部分的最大值（最大池化）或该部分的平均值（平均池化）。因此，对于一个n×n的图像，整个图像将被映射到一个<math><mfrac><mi>n</mi> <mn>2</mn></mfrac></math> × <math><mfrac><mi>n</mi> <mn>2</mn></mfrac></math>的大小。图5-5说明了这一点。

![神经网络图示](assets/dlfs_0505.png)

###### 图5-5。一个4×4输入的最大池化和平均池化示例；每个2×2的块被映射到该块的平均值或最大值

池化的主要优势在于计算：通过将图像下采样为前一层的四分之一像素数，池化将网络训练所需的权重数量和计算数量减少了四分之一；如果网络中使用了多个池化层，这种减少可以进一步叠加，就像在CNN的早期架构中使用的许多架构中一样。当然，池化的缺点是，从下采样的图像中只能提取四分之一的信息。然而，尽管池化通过降低图像的分辨率导致网络“丢失”了关于图像的信息，但尽管如此，这种权衡在增加计算速度方面是值得的，因为架构在图像识别基准测试中表现非常出色。然而，许多人认为池化只是一个偶然起作用的技巧，应该被淘汰；正如Geoffrey Hinton在2014年的Reddit AMA中写道：“卷积神经网络中使用的池化操作是一个大错误，它能够如此成功地运行是一场灾难。”事实上，大多数最近的CNN架构（如残差网络或“ResNets”）最小化或根本不使用池化。因此，在本书中，我们不会实现池化层，但考虑到它们在著名架构（如AlexNet）中的使用对“推动CNN发展”至关重要，我们在这里提及它们以保持完整性。

### 将CNN应用于图像之外

到目前为止，我们所描述的一切在使用神经网络处理图像方面都是非常标准的：图像通常被表示为一组*m*[1]通道的像素，其中*m*[1]=1表示黑白图像，*m*[1]=3表示彩色图像—然后对每个通道应用一定数量的卷积操作（使用之前解释过的<math><mrow><msub><mi>m</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>m</mi> <mn>2</mn></msub></mrow></math>滤波器映射），这种模式会持续几层。这些内容在其他卷积神经网络的处理中已经涵盖过；不太常见的是，将数据组织成“通道”，然后使用CNN处理数据的想法不仅仅适用于图像。例如，这种数据表示是DeepMind的AlphaGo系列程序的关键，展示了神经网络可以学会下围棋。引用论文中的话：

> 神经网络的输入是一个 19 × 19 × 17 的图像堆栈，包括 17 个二进制特征平面。8 个特征平面 *X*[t] 包含二进制值，指示当前玩家的棋子的存在（如果时间步 t 的交叉点 *i* 包含玩家颜色的棋子，则为 <math><mrow><msub><mi>X</mi> <msub><mi>t</mi> <mi>i</mi></msub></msub> <mo>=</mo> <mn>1</mn></mrow></math>；如果交叉点为空，包含对手的棋子，或者 t < 0，则为 0）。另外 8 个特征平面 *Y*[t] 表示对手棋子的相应特征。最后一个特征平面 C 表示要下的颜色，其常量值为 1（黑色下棋）或 0（白色下棋）。这些平面被连接在一起以给出输入特征 *s*[t] = *X*[t], *Y*[t], *X*[t – 1], *Y*[t – 1], …, *X*[t – 7], *Y*[t – 7], *C*。历史特征 *X*[t], *Y*[t] 是必要的，因为围棋不仅仅通过当前的棋子就能完全观察到，重复是被禁止的；同样，颜色特征 C 是必要的，因为贴目是不可观察的。

换句话说，他们基本上将棋盘表示为一个 19 × 19 像素的“图像”，有 17 个通道！他们使用其中的 16 个通道来编码每个玩家之前 8 步所发生的情况；这是必要的，以便他们可以编码防止重复之前步骤的规则。第 17 个通道实际上是一个 19 × 19 的网格，要么全是 1，要么全是 0，取决于轮到谁走。^([7](ch05.html#idm45732618397624)) CNN 和它们的多通道卷积操作主要应用于图像，但更一般的是，用多个“通道”表示沿某些空间维度排列的数据的想法即使超出图像也是适用的。

然而，为了真正理解多通道卷积操作，您必须从头开始实现它，接下来的几节将详细描述这个过程。

# 实现多通道卷积操作

事实证明，如果我们首先检查一维情况，即涉及四维输入 `ndarray` 和四维参数 `ndarray` 的实现将更清晰。从那个起点逐步构建到完整操作将主要是添加一堆 `for` 循环的问题。在整个过程中，我们将采取与[第一章](ch01.html#foundations)相同的方法，交替使用图表、数学和工作的 Python 代码。

## 前向传播

一维卷积在概念上与二维卷积相同：我们将一维输入和一维卷积滤波器作为输入，然后通过沿着输入滑动滤波器来创建输出。

假设我们的输入长度为 5：

![卷积](assets/dlfs_05in01.png)

假设我们要检测的“模式”的大小为 3：

![长度为三的滤波器](assets/dlfs_05in02.png)

### 图表和数学

输出的第一个元素将通过将输入的第一个元素与滤波器进行卷积来创建：

![输出元素 1](assets/dlfs_05in03.png)

输出的第二个元素将通过将滤波器向右滑动一个单位并将其与系列的下一组值进行卷积来创建：

![长度为一的输出](assets/dlfs_05in04.png)

好吧。然而，当我们计算下一个输出值时，我们意识到我们已经没有空间了：

![长度为一的输出](assets/dlfs_05in05.png)

我们已经到达了输入的末尾，结果输出只有三个元素，而我们开始时有五个！我们该如何解决这个问题？

### 填充

为了避免由于卷积操作导致输出缩小，我们将引入一种在卷积神经网络中广泛使用的技巧：我们在边缘周围“填充”输入与零，以使输出保持与输入大小相同。否则，每次我们在输入上卷积一个滤波器时，我们最终得到的输出会略小于输入，就像之前看到的那样。

正如您可以从前面的卷积示例推理出的：对于大小为3的滤波器，应该在边缘周围添加一个单位的填充，以保持输出与输入大小相同。更一般地，由于我们几乎总是使用奇数大小的滤波器，我们添加填充等于滤波器大小除以2并向下舍入到最接近的整数。

假设我们添加了这种填充，这样，输入不再是从*i*[1]到*i*[5]，而是从*i*[0]到*i*[6]，其中*i*[0]和*i*[6]都是0。然后我们可以计算卷积的输出为：

<math display="block"><mrow><msub><mi>o</mi> <mn>1</mn></msub> <mo>=</mo> <msub><mi>i</mi> <mn>0</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>i</mi> <mn>1</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mi>i</mi> <mn>2</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub></mrow></math>

依此类推，直到：

<math display="block"><mrow><msub><mi>o</mi> <mn>5</mn></msub> <mo>=</mo> <msub><mi>i</mi> <mn>4</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>i</mi> <mn>5</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mi>i</mi> <mn>6</mn></msub> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub></mrow></math>

现在输出与输入大小相同了。我们如何编写代码呢？

### 代码

编写这部分代码实际上非常简单。在我们开始之前，让我们总结一下我们刚刚讨论的步骤：

1.  我们最终希望生成一个与输入大小相同的输出。

1.  为了在不“缩小”输出的情况下执行此操作，我们首先需要填充输入。

1.  然后我们将不得不编写一些循环，通过输入并将其每个位置与滤波器进行卷积。

我们将从我们的输入和滤波器开始：

```py
input_1d = np.array([1,2,3,4,5])
param_1d = np.array([1,1,1])
```

这里有一个辅助函数，可以在一维输入的两端填充：

```py
def _pad_1d(inp: ndarray,
            num: int) -> ndarray:
    z = np.array([0])
    z = np.repeat(z, num)
    return np.concatenate([z, inp, z])

_pad_1d(input_1d, 1)
```

```py
array([0., 1., 2., 3., 4., 5., 0.])
```

卷积本身呢？观察到，对于我们想要生成的每个输出元素，我们在“填充”输入中有一个对应的元素，我们在那里“开始”卷积操作；一旦我们弄清楚从哪里开始，我们只需循环遍历滤波器中的所有元素，在每个元素上进行乘法并将结果添加到总和中。

我们如何找到这个“对应的元素”？注意，简单地说，输出中第一个元素的值从填充输入的第一个元素开始！这使得`for`循环非常容易编写：

```py
def conv_1d(inp: ndarray,
            param: ndarray) -> ndarray:

    # assert correct dimensions
    assert_dim(inp, 1)
    assert_dim(param, 1)

    # pad the input
    param_len = param.shape[0]
    param_mid = param_len // 2
    input_pad = _pad_1d(inp, param_mid)

    # initialize the output
    out = np.zeros(inp.shape)

    # perform the 1d convolution
    for o in range(out.shape[0]):
        for p in range(param_len):
            out[o] += param[p] * input_pad[o+p]

    # ensure shapes didn't change
    assert_same_shape(inp, out)

    return out

conv_1d_sum(input_1d, param_1d)
```

```py
array([ 3.,  6.,  9., 12.,  9.])
```

这已经足够简单了。在我们继续进行此操作的反向传递之前——棘手的部分——让我们简要讨论一下我们正在忽略的卷积的一个超参数：步幅。

### 关于步幅的说明

我们之前注意到，池化操作是从特征图中对图像进行下采样的一种方法。在许多早期的卷积架构中，这确实显著减少了所需的计算量，而且没有对准确性造成重大影响；然而，它们已经不再受欢迎，因为它们的缺点：它们有效地对图像进行下采样，使得分辨率减半的图像传递到下一层。

一个更广泛接受的方法是修改卷积操作的*步幅*。步幅是滤波器在图像上逐步滑动的量——在先前的情况下，我们使用步幅为1，因此每个滤波器与输入的每个元素进行卷积，这就是为什么输出的大小与输入的大小相同。使用步幅为2，滤波器将与输入图像的*每隔一个*元素进行卷积，因此输出的大小将是输入的一半；使用步幅为3，滤波器将与输入图像的*每隔两个*元素进行卷积，依此类推。这意味着，例如，使用步幅为2将导致相同的输出大小，因此与使用大小为2的池化相比，计算减少了很多，但没有太多的*信息损失*：使用大小为2的池化，只有输入中四分之一的元素对输出产生*任何*影响，而使用步幅为2，*每个*输入元素对输出都有*一些*影响。因此，即使在今天最先进的CNN架构中，使用大于1的步幅进行下采样的情况比池化更为普遍。

然而，在这本书中，我只会展示步幅为1的示例，将这些操作修改为允许大于1的步幅是留给读者的练习。使用步幅等于1也使得编写反向传播更容易。

## 卷积：反向传播

反向传播是卷积变得有点棘手的地方。让我们回顾一下我们要做的事情：之前，我们使用输入和参数生成了卷积操作的输出。现在我们想要计算：

+   损失相对于卷积操作的*输入*的每个元素的偏导数——之前是`inp`

+   损失相对于卷积操作的*滤波器*的每个元素的偏导数——之前是`param_1d`

想想我们在[第4章](ch04.html#extensions)中看到的`ParamOperation`是如何工作的：在`backward`方法中，它们接收一个表示每个输出元素最终影响损失程度的输出梯度，然后使用这个输出梯度来计算输入和参数的梯度。因此，我们需要编写一个函数，该函数接受与输入形状相同的`output_grad`，并产生一个`input_grad`和一个`param_grad`。

我们如何测试计算出的梯度是否正确？我们将从第一章中带回一个想法：我们知道对于任何一个输入，对于和的偏导数是1（如果和*s* = *a* + *b* + *c*，那么<math><mrow><mfrac><mrow><mi>∂</mi><mi>s</mi></mrow> <mrow><mi>∂</mi><mi>a</mi></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>s</mi></mrow> <mrow><mi>∂</mi><mi>b</mi></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>s</mi></mrow> <mrow><mi>∂</mi><mi>c</mi></mrow></mfrac> <mo>=</mo> <mn>1</mn></mrow></math>）。因此，我们可以使用我们的`_input_grad`和`_param_grad`函数（我们将很快推理和编写）以及一个全部为1的`output_grad`来计算`input_grad`和`param_grad`量。然后，我们将通过改变输入的元素一些数量*α*，并查看结果的总和是否通过梯度乘以*α*而改变来检查这些梯度是否正确。

### 梯度“应该”是多少？

使用刚才描述的逻辑，让我们计算输入向量的一个元素的梯度*应该*是多少：

```py
def conv_1d_sum(inp: ndarray,
                param: ndarray) -> ndarray:
    out = conv_1d(inp, param)
    return np.sum(out)
```

```py
# randomly choose to increase 5th element by 1
input_1d_2 = np.array([1,2,3,4,6])
param_1d = np.array([1,1,1])

print(conv_1d_sum(input_1d, param_1d))
print(conv_1d_sum(input_1d_2, param_1d))
```

```py
39.0
41.0
```

因此，输入的第五个元素的梯度*应该*是41 - 39 = 2。

现在让我们尝试推理如何计算这样的梯度，而不仅仅是计算这两个总和之间的差异。这就是事情变得有趣的地方。

### 计算1D卷积的梯度

我们看到增加输入的这个元素使输出增加了2。仔细观察输出，可以清楚地看到它是如何做到这一点的：

![完整输出](assets/dlfs_05in06.png)

输入的特定元素被表示为*t*[5]。它在输出中出现在两个地方：

+   作为*o*[4]的一部分，它与*w*[3]相乘。

+   作为*o*[5]的一部分，它与*w*[2]相乘。

为了帮助看到输入如何映射到输出总和的一般模式，请注意，如果存在*o*[6]，*t*[5]也将通过与*w*[1]相乘而对输出产生影响。

因此，<math><msub><mi>t</mi> <mn>5</mn></msub></math>最终影响损失的数量，我们可以表示为<math><mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><msub><mi>t</mi> <mn>5</mn></msub></mrow></mfrac></math>，将是：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><msub><mi>t</mi> <mn>5</mn></msub></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><msub><mi>o</mi> <mn>4</mn></msub></mrow></mfrac> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub> <mo>+</mo> <mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><msub><mi>o</mi> <mn>5</mn></msub></mrow></mfrac> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><msub><mi>o</mi> <mn>6</mn></msub></mrow></mfrac> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub></mrow></math>

当然，在这个简单的例子中，当损失只是总和时，对于所有输出元素，<math><mrow><mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><msub><mi>o</mi> <mi>i</mi></msub></mrow></mfrac> <mo>=</mo> <mn>1</mn></mrow></math>（对于“填充”元素除外，该数量为0）。这个总和非常容易计算：它只是*w*[2] + *w*[3]，确实是2，因为*w*[2] = *w*[3] = 1。

### 一般模式是什么？

现在让我们寻找通用输入元素的一般模式。这实际上是一个跟踪索引的练习。由于我们在这里将数学转换为代码，让我们使用<math><msubsup><mi>o</mi> <mi>i</mi> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup></math>来表示输出梯度的第*i*个元素（因为我们最终将通过`output_grad[i]`访问它）。然后：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><msub><mi>t</mi> <mn>5</mn></msub></mrow></mfrac> <mo>=</mo> <msubsup><mi>o</mi> <mn>4</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub> <mo>+</mo> <msubsup><mi>o</mi> <mn>5</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msubsup><mi>o</mi> <mn>6</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub></mrow></math>

仔细观察这个输出，我们可以类似地推理：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><msub><mi>t</mi> <mn>3</mn></msub></mrow></mfrac> <mo>=</mo> <msubsup><mi>o</mi> <mn>2</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub> <mo>+</mo> <msubsup><mi>o</mi> <mn>3</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msubsup><mi>o</mi> <mn>4</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub></mrow></math>

和：

<math display="block"><mrow><mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><msub><mi>t</mi> <mn>4</mn></msub></mrow></mfrac> <mo>=</mo> <msubsup><mi>o</mi> <mn>3</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>×</mo> <msub><mi>w</mi> <mn>3</mn></msub> <mo>+</mo> <msubsup><mi>o</mi> <mn>4</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>×</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>+</mo> <msubsup><mi>o</mi> <mn>5</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>×</mo> <msub><mi>w</mi> <mn>1</mn></msub></mrow></math>

这里显然有一个模式，将其转换为代码有点棘手，特别是因为输出上的索引增加的同时权重上的索引减少。然而，表达这一点的方式是通过以下双重`for`循环：

```py
# param: in our case an ndarray of shape (1,3)
# param_len: the integer 3
# inp: in our case an ndarray of shape (1,5)
# input_grad: always an ndarray the same shape as "inp"
# output_pad: in our case an ndarray of shape (1,7)
for o in range(inp.shape[0]):
    for p in range(param.shape[0]):
        input_grad[o] += output_pad[o+param_len-p-1] * param[p]
```

这样做适当地增加了权重的索引，同时减少了输出上的权重。

尽管现在可能不明显，但通过推理并得到它是计算卷积操作的梯度中最棘手的部分。增加更多复杂性，例如批量大小、具有二维输入的卷积或具有多个通道的输入，只是在前面的几行中添加更多的`for`循环，我们将在接下来的几节中看到。

### 计算参数梯度

我们可以类似地推理，关于如何增加滤波器的一个元素应该增加输出。首先，让我们增加（任意地）滤波器的第一个元素一个单位，并观察对总和的影响：

```py
input_1d = np.array([1,2,3,4,5])
# randomly choose to increase first element by 1
param_1d_2 = np.array([2,1,1])

print(conv_1d_sum(input_1d, param_1d))
print(conv_1d_sum(input_1d, param_1d_2))
```

```py
39.0
49.0
```

所以我们应该发现<math><mrow><mfrac><mrow><mi>∂</mi><mi>L</mi></mrow> <mrow><mi>∂</mi><msub><mi>w</mi> <mn>1</mn></msub></mrow></mfrac> <mo>=</mo> <mn>10</mn></mrow></math>。

就像我们为输入所做的那样，通过仔细检查输出并看到哪些滤波器元素影响它，以及填充输入以更清楚地看到模式，我们看到：

<math display="block"><mrow><msubsup><mi>w</mi> <mn>1</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>=</mo> <msub><mi>t</mi> <mn>0</mn></msub> <mo>×</mo> <msubsup><mi>o</mi> <mn>1</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>+</mo> <msub><mi>t</mi> <mn>1</mn></msub> <mo>×</mo> <msubsup><mi>o</mi> <mn>2</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>+</mo> <msub><mi>t</mi> <mn>2</mn></msub> <mo>×</mo> <msubsup><mi>o</mi> <mn>3</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>+</mo> <msub><mi>t</mi> <mn>3</mn></msub> <mo>×</mo> <msubsup><mi>o</mi> <mn>4</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>+</mo> <msub><mi>t</mi> <mn>4</mn></msub> <mo>×</mo> <msubsup><mi>o</mi> <mn>5</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup></mrow></math>

由于对于总和，所有的<math><msubsup><mi>o</mi> <mi>i</mi> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup></math>元素都是1，而*t*[0]是0，我们有：

<math display="block"><mrow><msubsup><mi>w</mi> <mn>1</mn> <mrow><mi>g</mi><mi>r</mi><mi>a</mi><mi>d</mi></mrow></msubsup> <mo>=</mo> <msub><mi>t</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mi>t</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mi>t</mi> <mn>3</mn></msub> <mo>+</mo> <msub><mi>t</mi> <mn>4</mn></msub> <mo>=</mo> <mn>1</mn> <mo>+</mo> <mn>2</mn> <mo>+</mo> <mn>3</mn> <mo>+</mo> <mn>4</mn> <mo>=</mo> <mn>10</mn></mrow></math>

这证实了之前的计算。

### 编码这个

编码这个比编写输入梯度的代码更容易，因为这次“索引是朝着同一个方向移动的。”在同一个嵌套的`for`循环中，代码是：

```py
# param: in our case an ndarray of shape (1,3)
# param_grad: an ndarray the same shape as param
# inp: in our case an ndarray of shape (1,5)
# input_pad: an ndarray the same shape as (1,7)
# output_grad: in our case an ndarray of shape (1,5)
for o in range(inp.shape[0]):
    for p in range(param.shape[0]):
        param_grad[p] += input_pad[o+p] * output_grad[o]
```

最后，我们可以结合这两个计算，并编写一个函数来计算输入梯度和滤波器梯度，具体步骤如下：

1.  将输入和滤波器作为参数。

1.  计算输出。

1.  填充输入和输出梯度（例如，得到`input_pad`和`output_pad`）。

1.  如前所示，使用填充的输出梯度和滤波器来计算梯度。

1.  类似地，使用输出梯度（未填充）和填充输入来计算滤波器梯度。

我展示了包装在书的[GitHub存储库](https://oreil.ly/2H99xkJ)中的前述代码块的完整函数。

这就结束了我们关于如何在1D中实现卷积的解释！正如我们将在接下来的几节中看到的，将这种推理扩展到在二维输入、二维输入批次，甚至多通道的二维输入批次上工作是（也许令人惊讶地）直接的。

## 批处理、2D卷积和多通道

首先，让我们为这些卷积函数添加能够处理*批量*输入的功能——2D输入的第一个维度表示输入的批量大小，第二个维度表示1D序列的长度：

```py
input_1d_batch = np.array([[0,1,2,3,4,5,6],
                           [1,2,3,4,5,6,7]])
```

我们可以遵循之前定义的相同一般步骤：首先填充输入，使用此输入计算输出，然后填充输出梯度以计算输入和滤波器梯度。

### 带批处理的1D卷积：正向传播

在输入具有表示批处理大小的第二维度时实现正向传播的唯一区别是，我们必须为每个观察值单独填充和计算输出（就像之前一样），然后`stack`结果以获得一批输出。例如，`conv_1d`变为：

```py
def conv_1d_batch(inp: ndarray,
                  param: ndarray) -> ndarray:

    outs = [conv_1d(obs, param) for obs in inp]
    return np.stack(outs)
```

### 带批处理的1D卷积：反向传播

反向传播类似：现在计算输入梯度只需从前一节的计算输入梯度的`for`循环中获取，为每个观察值计算它，并`stack`结果：

```py
# "_input_grad" is the function containing the for loop from earlier:
# it takes in a 1d input, a 1d filter, and a 1d output_gradient and computes
# the input grad
grads = [_input_grad(inp[i], param, out_grad[i])[1] for i in range(batch_size)]
np.stack(grads)
```

处理一批观察时，滤波器的梯度有点不同。这是因为滤波器与输入中的每个观察值进行卷积，因此与输出中的每个观察值相连。因此，为了计算参数梯度，我们必须遍历所有观察值，并在这样做时递增参数梯度的适当值。不过，这只涉及在计算我们之前看到的参数梯度的代码中添加一个外部`for`循环：

```py
# param: in our case an ndarray of shape (1,3)
# param_grad: an ndarray the same shape as param
# inp: in our case an ndarray of shape (1,5)
# input_pad: an ndarray the same shape as (1,7)
# output_grad: in our case an ndarray of shape (1,5)
for i in range(inp.shape[0]): # inp.shape[0] = 2
    for o in range(inp.shape[1]): # inp.shape[0] = 5
        for p in range(param.shape[0]): # param.shape[0] = 3
            param_grad[p] += input_pad[i][o+p] * output_grad[i][o]
```

将这个维度添加到原始1D卷积之上确实很简单；从一维到二维输入的扩展同样是直接的。

## 2D卷积

2D卷积是1D情况的直接扩展，因为从根本上讲，通过每个维度的滤波器将输入连接到输出的方式在2D情况下与1D情况相同。因此，正向传播和反向传播的高级步骤保持不变：

1.  在正向传播中，我们：

    +   适当地填充输入。

    +   使用填充的输入和参数来计算输出。

1.  在反向传播中，为了计算输入梯度，我们：

    +   适当地填充输出梯度。

    +   使用这个填充的输出梯度，以及输入和参数，来计算输入梯度和参数梯度。

1.  同样在反向传播中，为了计算参数梯度，我们：

    +   适当地填充输入。

    +   遍历填充输入的元素，并在进行时适当递增参数梯度。

### 2D卷积：编写正向传播

具体来说，回想一下，对于1D卷积，计算输出的代码在正向传播中如下所示：

```py
# input_pad: a version of the input that has been padded appropriately based on
# the size of param

out = np.zeros_like(inp)

for o in range(out.shape[0]):
    for p in range(param_len):
        out[o] += param[p] * input_pad[o+p]
```

对于2D卷积，我们只需将其修改为：

```py
# input_pad: a version of the input that has been padded appropriately based on
# the size of param

out = np.zeros_like(inp)

for o_w in range(img_size): # loop through the image height
    for o_h in range(img_size): # loop through the image width
        for p_w in range(param_size): # loop through the parameter width
            for p_h in range(param_size): # loop through the parameter height
                out[o_w][o_h] += param[p_w][p_h] * input_pad[o_w+p_w][o_h+p_h]
```

您可以看到我们只是将每个`for`循环“展开”为两个`for`循环。

当我们有一批图像时，扩展到两个维度也类似于1D情况：就像我们在那里所做的那样，我们只需在这里显示的循环外部添加一个`for`循环。

### 2D卷积：编写反向传播

果然，就像在正向传播中一样，我们可以在反向传播中使用与1D情况相同的索引。回想一下，在1D情况下，代码是：

```py
input_grad = np.zeros_like(inp)

for o in range(inp.shape[0]):
    for p in range(param_len):
        input_grad[o] += output_pad[o+param_len-p-1] * param[p]
```

在2D情况下，代码简单地是：

```py
# output_pad: a version of the output that has been padded appropriately based
# on the size of param
input_grad = np.zeros_like(inp)

for i_w in range(img_width):
    for i_h in range(img_height):
        for p_w in range(param_size):
            for p_h in range(param_size):
                input_grad[i_w][i_h] +=
                  output_pad[i_w+param_size-p_w-1][i_h+param_size-p_h-1] \
                  * param[p_w][p_h]
```

请注意，输出的索引与1D情况相同，但是在两个维度上进行；在1D情况下，我们有：

```py
output_pad[i+param_size-p-1] * param[p]
```

在2D情况下，我们有：

```py
output_pad[i_w+param_size-p_w-1][i_h+param_size-p_h-1] * param[p_w][p_h]
```

1D情况下的其他事实也适用：

+   对于一批输入图像，我们只需为每个观察执行前面的操作，然后`stack`结果。

+   对于参数梯度，我们必须循环遍历批次中的所有图像，并将每个图像的组件添加到参数梯度的适当位置中。

```py
# input_pad: a version of the input that has been padded appropriately based on
# the size of param

param_grad = np.zeros_like(param)

for i in range(batch_size): # equal to inp.shape[0]
    for o_w in range(img_size):
        for o_h in range(img_size):
            for p_w in range(param_size):
                for p_h in range(param_size):
                    param_grad[p_w][p_h] += input_pad[i][o_w+p_w][o_h+p_h] \
                    * output_grad[i][o_w][o_h]
```

到目前为止，我们几乎已经编写了完整的多通道卷积操作的代码；目前，我们的代码在二维输入上卷积滤波器并产生二维输出。当然，正如我们之前描述的那样，每个卷积层不仅沿着这两个维度排列神经元，还有一定数量的“通道”，等于该层创建的特征图的数量。解决这个最后的挑战是我们接下来要讨论的内容。

## 最后一个元素：添加“通道”

我们如何修改我们迄今为止所写的内容，以考虑输入和输出都是多通道的情况？答案与之前添加批次时一样简单：我们在已经看到的代码中添加两个外部`for`循环——一个循环用于输入通道，另一个循环用于输出通道。通过循环遍历输入通道和输出通道的所有组合，我们使每个输出特征图成为所有输入特征图的组合，如所需。

为了使其工作，我们将*始终*将我们的图像表示为三维`ndarray`，而不是我们一直使用的二维数组；我们将用一个通道表示黑白图像，用三个通道表示彩色图像（一个通道用于图像中每个位置的红色值，一个用于蓝色值，一个用于绿色值）。然后，无论通道数量如何，操作都会按照前面描述的方式进行，从图像中创建多个特征图，每个特征图都是来自图像中所有通道（或者来自网络中更深层次的层的通道）卷积的组合。

### 前向传播

考虑到这一切，为了计算卷积层的输出，给定输入和参数的四维`ndarray`，完整的代码如下：

```py
def _compute_output_obs(obs: ndarray,
                       param: ndarray) -> ndarray:
    '''
 obs: [channels, img_width, img_height]
 param: [in_channels, out_channels, param_width, param_height]
 '''
    assert_dim(obs, 3)
    assert_dim(param, 4)

    param_size = param.shape[2]
    param_mid = param_size // 2
    obs_pad = _pad_2d_channel(obs, param_mid)

    in_channels = fil.shape[0]
    out_channels = fil.shape[1]
    img_size = obs.shape[1]

    out = np.zeros((out_channels,) + obs.shape[1:])
    for c_in in range(in_channels):
        for c_out in range(out_channels):
            for o_w in range(img_size):
                for o_h in range(img_size):
                    for p_w in range(param_size):
                        for p_h in range(param_size):
                            out[c_out][o_w][o_h] += \
                              param[c_in][c_out][p_w][p_h]
                              * obs_pad[c_in][o_w+p_w][o_h+p_h]
    return out

def _output(inp: ndarray,
            param: ndarray) -> ndarray:
    '''
 obs: [batch_size, channels, img_width, img_height]
 param: [in_channels, out_channels, param_width, param_height]
 '''
    outs = [_compute_output_obs(obs, param) for obs in inp]

    return np.stack(outs)
```

请注意，`_pad_2d_channel`是一个沿着通道维度对输入进行填充的函数。

再次强调，进行计算的实际代码与之前显示的简单2D情况（无通道）中的代码类似，只是现在我们有了例如`fil[c_out][c_in][p_w][p_h]`而不仅仅是`fil[p_w][p_h]`，因为有两个更多的维度和`c_out × c_in`更多的元素在滤波器数组中。

### 反向传播

反向传播与简单的2D情况下的反向传播遵循相同的概念原则：

1.  对于输入梯度，我们分别计算每个观察的梯度——填充输出梯度以进行计算——然后`stack`梯度。

1.  我们还使用填充的输出梯度来计算参数梯度，但我们也循环遍历观察，并使用每个观察中的适当值来更新参数梯度。

这是计算输出梯度的代码：

```py
def _compute_grads_obs(input_obs: ndarray,
                       output_grad_obs: ndarray,
                       param: ndarray) -> ndarray:
    '''
 input_obs: [in_channels, img_width, img_height]
 output_grad_obs: [out_channels, img_width, img_height]
 param: [in_channels, out_channels, img_width, img_height]
 '''
    input_grad = np.zeros_like(input_obs)
    param_size = param.shape[2]
    param_mid = param_size // 2
    img_size = input_obs.shape[1]
    in_channels = input_obs.shape[0]
    out_channels = param.shape[1]
    output_obs_pad = _pad_2d_channel(output_grad_obs, param_mid)

    for c_in in range(in_channels):
        for c_out in range(out_channels):
            for i_w in range(input_obs.shape[1]):
                for i_h in range(input_obs.shape[2]):
                    for p_w in range(param_size):
                        for p_h in range(param_size):
                            input_grad[c_in][i_w][i_h] += \
                            output_obs_pad[c_out][i_w+param_size-p_w-1][i_h+param_size-p_h-1] \
                            * param[c_in][c_out][p_w][p_h]
    return input_grad

def _input_grad(inp: ndarray,
                output_grad: ndarray,
                param: ndarray) -> ndarray:

    grads = [_compute_grads_obs(inp[i], output_grad[i], param) for i in range(output_grad.shape[0])]

    return np.stack(grads)
```

这是参数梯度：

```py
def _param_grad(inp: ndarray,
                output_grad: ndarray,
                param: ndarray) -> ndarray:
    '''
 inp: [in_channels, img_width, img_height]
 output_grad_obs: [out_channels, img_width, img_height]
 param: [in_channels, out_channels, img_width, img_height]
 '''
    param_grad = np.zeros_like(param)
    param_size = param.shape[2]
    param_mid = param_size // 2
    img_size = inp.shape[2]
    in_channels = inp.shape[1]
    out_channels = output_grad.shape[1]

    inp_pad = _pad_conv_input(inp, param_mid)
    img_shape = output_grad.shape[2:]

    for i in range(inp.shape[0]):
        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for o_w in range(img_shape[0]):
                    for o_h in range(img_shape[1]):
                        for p_w in range(param_size):
                            for p_h in range(param_size):
                                param_grad[c_in][c_out][p_w][p_h] += \
                                inp_pad[i][c_in][o_w+p_w][o_h+p_h] \
                                * output_grad[i][c_out][o_w][o_h]
    return param_grad
```

这三个函数——`_output`、`_input_grad`和`_param_grad`——正是我们需要创建一个`Conv2DOperation`所需的，这最终将构成我们在CNN中使用的`Conv2DLayer`的核心！在我们能够在工作的卷积神经网络中使用这个`Operation`之前，还有一些细节需要解决。

# 使用这个操作来训练CNN

在我们能够拥有一个可工作的CNN模型之前，我们需要实现一些额外的部分：

1.  我们必须实现本章前面讨论的`Flatten`操作；这是为了使模型能够进行预测。

1.  我们必须将这个`Operation`以及`Conv2DOpOperation`整合到一个`Conv2D` `Layer`中。

1.  最后，为了使这些可用，我们必须编写一个更快的Conv2D操作的版本。我们将在这里概述这一点，并在[“矩阵链规则”](app01.html#matrix-chain-rule)中分享详细信息。

## Flatten操作

我们需要完成卷积层的另一个“操作”：Flatten操作。卷积操作的输出是每个观察结果的3D ndarray，维度为（通道数，图像高度，图像宽度）。然而，除非我们将这些数据传递到另一个卷积层，否则我们首先需要将其转换为每个观察结果的*向量*。幸运的是，正如之前描述的那样，由于涉及的每个单独神经元都编码了图像中特定视觉特征是否存在的信息，我们可以简单地将这个3D ndarray“展平”为一个1D向量，并且在没有任何问题的情况下将其传递到前面。这里显示的Flatten操作就是这样做的，考虑到在卷积层中，与任何其他层一样，我们的ndarray的第一个维度始终是批量大小：

```py
class Flatten(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> ndarray:
        return self.input.reshape(self.input.shape[0], -1)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad.reshape(self.input.shape)
```

这是我们需要的最后一个“操作”；让我们将这些“操作”封装在一个“层”中。

## 完整的Conv2D层

因此，完整的卷积层看起来会像这样：

```py
class Conv2D(Layer):

    def __init__(self,
                 out_channels: int,
                 param_size: int,
                 activation: Operation = Sigmoid(),
                 flatten: bool = False) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.param_size = param_size
        self.activation = activation
        self.flatten = flatten

def _setup_layer(self, input_: ndarray) -> ndarray:

    self.params = []
    conv_param = np.random.randn(self.out_channels,
                                 input_.shape[1],  # input channels
                                 self.param_size,
                                 self.param_size)
    self.params.append(conv_param)

    self.operations = []
    self.operations.append(Conv2D(conv_param))
    self.operations.append(self.activation)

    if self.flatten:
        self.operations.append(Flatten())

    return None
```

Flatten操作是可选的，取决于我们是否希望将此层的输出传递到另一个卷积层或传递到另一个全连接层进行预测。

### 关于速度和另一种实现的说明

熟悉计算复杂性的人会意识到，这段代码运行速度非常慢：为了计算参数梯度，我们需要编写*七个*嵌套的for循环！这样做没有问题，因为从头开始编写卷积操作的目的是巩固我们对CNN工作原理的理解。然而，完全可以以完全不同的方式编写卷积；与我们在本章中所做的方式不同，我们可以将其分解为以下步骤：

1.  从输入中提取大小为filter_height×filter_width的测试集的image_height×image_width×num_channels个补丁。

1.  对于这些补丁中的每一个，执行与将输入通道连接到输出通道的适当滤波器的点积。

1.  堆叠和重塑所有这些点积的结果以形成输出。

通过一点巧妙，我们几乎可以用一个批量矩阵乘法来表达之前描述的所有操作，使用NumPy的np.matmul函数实现。如何做到这一点的细节在[附录A](app01.html#appendix)中有描述，并且在[本书的网站](https://oreil.ly/2H99xkJ)上实现，但可以说这样可以让我们编写相对较小的卷积神经网络，可以在合理的时间内进行训练。这让我们实际上可以运行实验，看看卷积神经网络的工作效果如何！

## 实验

即使使用通过重塑和matmul函数定义的卷积操作，训练这个只有一个卷积层的模型一个时期大约需要10分钟，因此我们限制自己演示一个只有一个卷积层的模型，具有32个通道（一个相当随意选择的数字）：

```py
model = NeuralNetwork(
    layers=[Conv2D(out_channels=32,
                   param_size=5,
                   dropout=0.8,
                   weight_init="glorot",
                   flatten=True,
                  activation=Tanh()),
            Dense(neurons=10,
                  activation=Linear())],
            loss = SoftmaxCrossEntropy(),
seed=20190402)
```

请注意，这个模型在第一层只有32×5×5=800个参数，但这些参数用于创建32×28×28=25,088个神经元或“学习特征”。相比之下，具有32个隐藏单元的全连接层将有784×32=25,088个*参数*，只有32个神经元。

一些简单的试错——用不同的学习率训练这个模型几百批次，并观察结果验证损失——显示，现在我们的第一层是卷积层，而不是全连接层，学习率为0.01比学习率为0.1效果更好。使用优化器`SGDMomentum(lr = 0.01, momentum=0.9)`训练这个网络一个时期，得到：

```py
Validation accuracy after 100 batches is 79.65%
Validation accuracy after 200 batches is 86.25%
Validation accuracy after 300 batches is 85.47%
Validation accuracy after 400 batches is 87.27%
Validation accuracy after 500 batches is 88.93%
Validation accuracy after 600 batches is 88.25%
Validation accuracy after 700 batches is 89.91%
Validation accuracy after 800 batches is 89.59%
Validation accuracy after 900 batches is 89.96%
Validation loss after 1 epochs is 3.453

Model validation accuracy after 1 epoch is 90.50%
```

这表明我们确实可以从头开始训练一个卷积神经网络，在通过训练集进行一次遍历后，准确率达到90%以上！

# 结论

在本章中，您已经了解了卷积神经网络。您从高层次开始学习它们是什么，以及它们与全连接神经网络的相似之处和不同之处，然后一直到最低层次，看到它们如何在Python中从头开始实现核心多通道卷积操作。

从高层开始，卷积层比我们迄今为止看到的全连接层创建大约一个数量级更多的神经元，每个神经元是前一层中仅有几个特征的组合，而不是每个神经元都是前一层所有特征的组合，就像全连接层中那样。在更低的层次上，我们看到这些神经元实际上被分组成“特征图”，每个特征图表示在图像的特定位置是否存在特定的视觉特征，或者在深度卷积神经网络的情况下，特定的视觉特征组合是否存在。总体上，我们将这些特征图称为卷积`Layer`的“通道”。

尽管与我们在`Dense`层中看到的`Operation`有很多不同，卷积操作与我们看到的其他`ParamOperation`一样适合相同的模板：

+   它有一个`_output`方法，根据其输入和参数计算输出。

+   它有`_input_grad`和`_param_grad`方法，给定与`Operation`的`output`相同形状的`output_grad`，计算与输入和参数相同形状的梯度。

唯一的区别在于现在`_input`、`output`和`param`是四维的`ndarray`，而在全连接层的情况下它们是二维的。

这些知识应该为您未来学习或应用卷积神经网络奠定非常坚实的基础。接下来，我们将介绍另一种常见的高级神经网络架构：递归神经网络，设计用于处理以序列形式出现的数据，而不仅仅是我们在房屋和图像的情况下处理的非顺序批次。继续前进！

我们将编写的代码，虽然清楚地表达了卷积的工作原理，但效率非常低下。在[“关于偏差项的损失梯度”](app01.html#gradient-loss-bias-terms)中，我提供了一个更有效的实现，使用NumPy描述了本章中我们将描述的批量、多通道卷积操作。

查看维基百科页面[“Kernel (image processing)”](https://oreil.ly/2KOwfzs)获取更多示例。

这些也被称为*内核*。

^([4](ch05.html#idm45732618457096-marker)) 这就是为什么重要理解卷积操作的输出，既可以看作是创建一定数量的滤波器映射（比如说，<math><mi>m</mi></math>），也可以看作是创建 <math><mrow><mi>m</mi> <mo>×</mo> <mi>i</mi> <mi>m</mi> <mi>a</mi> <mi>g</mi> <msub><mi>e</mi> <mrow><mi>h</mi><mi>e</mi><mi>i</mi><mi>g</mi><mi>h</mi><mi>t</mi></mrow></msub> <mo>×</mo> <mi>i</mi> <mi>m</mi> <mi>a</mi> <mi>g</mi> <msub><mi>e</mi> <mrow><mi>w</mi><mi>i</mi><mi>d</mi><mi>t</mi><mi>h</mi></mrow></msub></mrow></math> 个独立的神经元。正如在神经网络中一样，同时在脑海中保持多个层次的解释，并看到它们之间的联系是关键。

^([5](ch05.html#idm45732618426472-marker)) 请参阅[原始 ResNet 论文](http://tiny.cc/dlfs_resnet_paper)，作者是 Kaiming He 等人，题目是“用于图像识别的深度残差学习”。

^([6](ch05.html#idm45732618415512-marker)) DeepMind（David Silver 等人），[*无需人类知识掌握围棋*](https://oreil.ly/wUpMW)，2017 年。

^([7](ch05.html#idm45732618397624-marker)) 一年后，DeepMind 发布了使用类似表示法的结果，只是这一次，为了编码更复杂的国际象棋规则，输入有 119 个通道！参见 DeepMind（David Silver 等人），[“一个通用的强化学习算法，通过自我对弈掌握国际象棋、将棋和围棋”](https://oreil.ly/E6ydw)。

^([8](ch05.html#idm45732616694312-marker)) 请在[书籍的网站](https://oreil.ly/2H99xkJ)上查看这些内容的完整实现。

^([9](ch05.html#idm45732615365528-marker)) 完整的代码可以在[书籍的 GitHub 仓库](https://oreil.ly/2H99xkJ)的本章节中找到。
