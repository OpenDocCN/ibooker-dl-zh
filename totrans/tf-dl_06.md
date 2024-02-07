# 第六章。卷积神经网络

卷积神经网络允许深度网络学习结构化空间数据（如图像、视频和文本）上的函数。从数学上讲，卷积网络提供了有效利用数据局部结构的工具。图像满足某些自然的统计特性。让我们假设将图像表示为像素的二维网格。在像素网格中彼此接近的图像部分很可能一起变化（例如，图像中对应桌子的所有像素可能都是棕色的）。卷积网络学会利用这种自然的协方差结构以有效地学习。

卷积网络是一个相对古老的发明。卷积网络的版本早在上世纪 80 年代就在文献中提出过。虽然这些旧卷积网络的设计通常相当合理，但它们需要超过当时可用硬件的资源。因此，卷积网络在研究文献中相对默默无闻。

这一趋势在 2012 年 ILSVRC 挑战赛中戏剧性地逆转，该挑战赛是关于图像中物体检测的，卷积网络 AlexNet 实现的错误率是其最近竞争对手的一半。AlexNet 能够利用 GPU 在大规模数据集上训练旧的卷积架构。这种旧架构与新硬件的结合使得 AlexNet 能够在图像物体检测领域显著超越现有技术。这一趋势仅在继续，卷积神经网络在处理图像方面取得了巨大的提升。几乎可以说，现代几乎所有的图像处理流程现在都由卷积神经网络驱动。

卷积网络设计也经历了复兴，将卷积网络推进到远远超过上世纪 80 年代基本模型的水平。首先，网络变得更加深层，强大的最新网络达到了数百层深度。另一个广泛的趋势是将卷积架构泛化到适用于新数据类型。例如，图卷积架构允许卷积网络应用于分子数据，如我们在前几章中遇到的 Tox21 数据集！卷积架构还在基因组学、文本处理甚至语言翻译中留下了痕迹。

在本章中，我们将介绍卷积网络的基本概念。这些将包括构成卷积架构的基本网络组件，以及指导这些组件如何连接的设计原则的介绍。我们还将提供一个深入的示例，演示如何使用 TensorFlow 训练卷积网络。本章的示例代码改编自 TensorFlow 文档中有关卷积神经网络的教程。如果您对我们所做的更改感兴趣，请访问 TensorFlow 网站上的[原始教程](https://www.tensorflow.org/tutorials/deep_cnn)。与往常一样，我们鼓励您在本书的相关[GitHub 存储库](https://github.com/matroid/dlwithtf)中逐步完成本章的脚本。

# 卷积架构简介

大多数卷积架构由许多基本原语组成。这些原语包括卷积层和池化层等层。还有一组相关的词汇，包括局部感受野大小、步幅大小和滤波器数量。在本节中，我们将简要介绍卷积网络基本词汇和概念的基础。

## 局部感受野

局部感受野的概念源自神经科学，神经元的感受野是影响神经元放电的身体感知部分。神经元在处理大脑看到的感官输入时有一定的“视野”。这个视野传统上被称为局部感受野。这个“视野”可以对应于皮肤的一小块或者一个人的视野的一部分。图 6-1 展示了一个神经元的局部感受野。

![neuron_receptive_field.png](img/tfdl_0601.png)

###### 图 6-1\. 一个神经元的局部感受野的插图。

卷积架构借用了这个概念，计算概念上的“局部感受野”。图 6-2 提供了应用于图像数据的局部感受野概念的图示表示。每个局部感受野对应于图像中的一组像素，并由一个单独的“神经元”处理。这些“神经元”与全连接层中的神经元直接类似。与全连接层一样，对传入数据（源自局部感受图像补丁）应用非线性变换。

![local_receiptive_input.png](img/tfdl_0602.png)

###### 图 6-2\. 卷积网络中“神经元”的局部感受野（RF）。

这样的“卷积神经元”层可以组合成一个卷积层。这一层可以被看作是一个空间区域到另一个空间区域的转换。在图像的情况下，一个批次的图像通过卷积层被转换成另一个。图 6-3 展示了这样的转换。在接下来的部分，我们将向您展示卷积层是如何构建的更多细节。

![conv_receptive.png](img/tfdl_0603.png)

###### 图 6-3\. 一个卷积层执行图像转换。

值得强调的是，局部感受野不一定局限于图像数据。例如，在堆叠的卷积架构中，其中一个卷积层的输出馈送到下一个卷积层的输入，局部感受野将对应于处理过的特征数据的“补丁”。

## 卷积核

在上一节中，我们提到卷积层对其输入中的局部感受野应用非线性函数。这种局部应用的非线性是卷积架构的核心，但不是唯一的部分。卷积的第二部分是所谓的“卷积核”。卷积核只是一个权重矩阵，类似于与全连接层相关联的权重。图 6-4 以图解的方式展示了卷积核如何应用到输入上。

![sliding_kernal.png](img/tfdl_0604.png)

###### 图 6-4\. 一个卷积核被应用到输入上。卷积核的权重与局部感受野中对应的数字逐元素相乘，相乘的数字相加。请注意，这对应于一个没有非线性的卷积层。

卷积网络背后的关键思想是相同的（非线性）转换应用于图像中的每个局部感受野。在视觉上，将局部感受野想象成在图像上拖动的滑动窗口。在每个局部感受野的位置，非线性函数被应用以返回与该图像补丁对应的单个数字。正如图 6-4 所示，这种转换将一个数字网格转换为另一个数字网格。对于图像数据，通常以每个感受野大小的像素数来标记局部感受野的大小。例如，在卷积网络中经常看到 5×5 和 7×7 的局部感受野大小。

如果我们想要指定局部感受野不重叠怎么办？这样做的方法是改变卷积核的*步幅大小*。步幅大小控制感受野在输入上的移动方式。图 6-4 展示了一个一维卷积核，分别具有步幅大小 1 和 2。图 6-5 说明了改变步幅大小如何改变感受野在输入上的移动方式。

![stride_size.png](img/tfdl_0605.png)

###### 图 6-5。步幅大小控制局部感受野在输入上的“滑动”。这在一维输入上最容易可视化。左侧的网络步幅为 1，而右侧的网络步幅为 2。请注意，每个局部感受野计算其输入的最大值。

现在，请注意我们定义的卷积核将一个数字网格转换为另一个数字网格。如果我们想要输出多个数字网格怎么办？这很容易；我们只需要添加更多的卷积核来处理图像。卷积核也称为*滤波器*，因此卷积层中的滤波器数量控制我们获得的转换网格数量。一组卷积核形成一个*卷积层*。

# 多维输入上的卷积核

在本节中，我们主要将卷积核描述为将数字网格转换为其他数字网格。回想一下我们在前几章中使用的张量语言，卷积将矩阵转换为矩阵。

如果您的输入具有更多维度怎么办？例如，RGB 图像通常具有三个颜色通道，因此 RGB 图像在正确的情况下是一个秩为 3 的张量。处理 RGB 数据的最简单方法是规定每个局部感受野包括与该补丁中的像素相关联的所有颜色通道。然后，您可以说局部感受野的大小为 5×5×3，对于一个大小为 5×5 像素且具有三个颜色通道的局部感受野。

一般来说，您可以通过相应地扩展局部感受野的维度来将高维张量推广到更高维度的张量。这可能还需要具有多维步幅，特别是如果要分别处理不同维度。细节很容易解决，我们将探索多维卷积核作为您要进行的练习。

## 池化层

在前一节中，我们介绍了卷积核的概念。这些核将可学习的非线性变换应用于输入的局部补丁。这些变换是可学习的，并且根据通用逼近定理，能够学习局部补丁上任意复杂的输入变换。这种灵活性赋予了卷积核很大的能力。但同时，在深度卷积网络中具有许多可学习权重可能会减慢训练速度。

与使用可学习变换不同，可以使用固定的非线性变换来减少训练卷积网络的计算成本。一种流行的固定非线性是“最大池化”。这样的层选择并输出每个局部感受补丁中激活最大的输入。图 6-6 展示了这个过程。池化层有助于以结构化方式减少输入数据的维度。更具体地说，它们采用局部感受野，并用最大（或最小或平均）函数替换字段的每个部分的非线性激活函数。

![maxpool.jpeg](img/tfdl_0606.png)

###### 图 6-6。最大池化层的示例。请注意，每个彩色区域（每个局部感受野）中的最大值报告在输出中。

随着硬件的改进，池化层变得不那么有用。虽然池化仍然作为一种降维技术很有用，但最近的研究倾向于避免使用池化层，因为它们固有的丢失性（无法从池化数据中推断出输入中的哪个像素产生了报告的激活）。尽管如此，池化出现在许多标准卷积架构中，因此值得理解。

## 构建卷积网络

一个简单的卷积架构将一系列卷积层和池化层应用于其输入，以学习输入图像数据上的复杂函数。在构建这些网络时有很多细节，但在其核心，架构设计只是一种复杂的乐高堆叠形式。图 6-7 展示了一个卷积架构可能是如何由组成块构建起来的。

![cnnimage.png](img/tfdl_0607.png)

###### 图 6-7。一个简单的卷积架构的示例，由堆叠的卷积和池化层构成。

## 膨胀卷积

膨胀卷积或空洞卷积是一种新近流行的卷积层形式。这里的见解是为每个神经元在局部感受野中留下间隙（atrous 意味着*a trous*，即法语中的“带孔”）。这个基本概念在信号处理中是一个古老的概念，最近在卷积文献中找到了一些好的应用。

空洞卷积的核心优势是每个神经元的可见区域增加。让我们考虑一个卷积架构，其第一层是具有 3×3 局部感受野的普通卷积。然后，在架构中更深一层的第二个普通卷积层中的神经元具有 5×5 的感受野深度（第二层中局部感受野中的每个神经元本身在第一层中具有局部感受野）。然后，更深的两层的神经元具有 7×7 的感受视图。一般来说，卷积架构中第*N*层的神经元具有大小为(2*N* + 1) × (2*N* + 1)的感受视图。这种感受视图的线性增长对于较小的图像是可以接受的，但对于大型图像很快就会成为一个负担。

空洞卷积通过在其局部感受野中留下间隙实现了可见感受野的指数增长。一个“1-膨胀”卷积不留下间隙，而一个“2-膨胀”卷积在每个局部感受野元素之间留下一个间隙。堆叠膨胀层会导致局部感受野大小呈指数增长。图 6-8 说明了这种指数增长。

膨胀卷积对于大型图像非常有用。例如，医学图像在每个维度上可以延伸到数千个像素。创建具有全局理解的普通卷积网络可能需要不合理深的网络。使用膨胀卷积可以使网络更好地理解这些图像的全局结构。

![dilated_convolution.png](img/tfdl_0608.png)

###### 图 6-8。一个膨胀（或空洞）卷积。为每个神经元在局部感受野中留下间隙。图（a）描述了一个 1-膨胀的 3×3 卷积。图（b）描述了将一个 2-膨胀的 3×3 卷积应用于（a）。图（c）描述了将一个 4-膨胀的 3×3 卷积应用于（b）。注意，（a）层的感受野宽度为 3，（b）层的感受野宽度为 7，（c）层的感受野宽度为 15。

# 卷积网络的应用

在前一节中，我们介绍了卷积网络的机制，并向您介绍了构成这些网络的许多组件。在本节中，我们描述了一些卷积架构可以实现的应用。

## 目标检测和定位

目标检测是检测照片中存在的对象（或实体）的任务。目标定位是识别图像中对象存在的位置，并在每个出现的位置周围绘制“边界框”的任务。图 6-9 展示了标准图像上检测和定位的样子。

![detection_and_localization.jpg](img/tfdl_0609.png)

###### 图 6-9。在一些示例图像中检测和定位的对象，并用边界框标出。

为什么检测和定位很重要？一个非常有用的定位任务是从自动驾驶汽车拍摄的图像中检测行人。不用说，自动驾驶汽车能够识别所有附近的行人是非常重要的。目标检测的其他应用可能用于在上传到社交网络的照片中找到所有朋友的实例。另一个应用可能是从无人机中识别潜在的碰撞危险。

这些丰富的应用使得检测和定位成为大量研究活动的焦点。本书中多次提到的 ILSVRC 挑战专注于检测和定位在 ImagetNet 集合中找到的对象。

## 图像分割

图像分割是将图像中的每个像素标记为其所属对象的任务。分割与目标定位相关，但要困难得多，因为它需要准确理解图像中对象之间的边界。直到最近，图像分割通常是通过图形模型完成的，这是一种与深度网络不同的机器学习形式，但最近卷积分割已经崭露头角，并使图像分割算法取得了新的准确性和速度记录。图 6-10 显示了应用于自动驾驶汽车图像数据的图像分割的示例。

![nvidia_digits.png](img/tfdl_0610.png)

###### 图 6-10。图像中的对象被“分割”为各种类别。图像分割预计将对自动驾驶汽车和机器人等应用非常有用，因为它将实现对场景的细粒度理解。

## 图卷积

到目前为止，我们向您展示的卷积算法期望其输入为矩形张量。这样的输入可以是图像、视频，甚至句子。是否可能将卷积推广到不规则输入？

卷积层背后的基本思想是局部感受野的概念。每个神经元计算其局部感受野中的输入，这些输入通常构成图像输入中相邻的像素。对于不规则输入，例如图 6-11 中的无向图，这种简单的局部感受野的概念是没有意义的；没有相邻的像素。如果我们可以为无向图定义一个更一般的局部感受野，那么我们应该能够定义接受无向图的卷积层。

![graph_example.png](img/tfdl_0611.png)

###### 图 6-11。由边连接的节点组成的无向图的示例。

如图 6-11 所示，图由一组由边连接的节点组成。一个潜在的局部感受野的定义可能是将其定义为一个节点及其邻居的集合（如果两个节点通过边连接，则被认为是邻居）。使用这种局部感受野的定义，可以定义卷积和池化层的广义概念。这些层可以组装成图卷积架构。

这种图卷积架构可能在哪些地方证明有用？在化学中，分子可以被建模为原子形成节点，化学键形成边缘的无向图。因此，图卷积架构在化学机器学习中特别有用。例如，图 6-12 展示了图卷积架构如何应用于处理分子输入。

![graphconv_graphic_v2.png](img/tfdl_0612.png)

###### 图 6-12。展示了一个图卷积架构处理分子输入的示意图。分子被建模为一个无向图，其中原子形成节点，化学键形成边缘。"图拓扑"是对应于分子的无向图。"原子特征"是向量，每个原子一个，总结了局部化学信息。改编自“一次性学习的低数据药物发现”。

## 使用变分自动编码器生成图像

到目前为止，我们描述的应用都是监督学习问题。有明确定义的输入和输出，任务仍然是（使用卷积网络）学习一个将输入映射到输出的复杂函数。有没有无监督学习问题可以用卷积网络解决？回想一下，无监督学习需要“理解”输入数据点的结构。对于图像建模，理解输入图像结构的一个好的衡量标准是能够“采样”来自输入分布的新图像。

什么是“采样”图像的意思？为了解释，假设我们有一组狗的图像数据集。采样一个新的狗图像需要生成一张*不在训练数据中*的新狗图像！这个想法是，我们希望得到一张狗的图片，这张图片可能已经被包含在训练数据中，但实际上并没有。我们如何用卷积网络解决这个任务？

也许我们可以训练一个模型，输入词标签如“狗”，并预测狗的图像。我们可能能够训练一个监督模型来解决这个预测问题，但问题在于我们的模型只能在输入标签“狗”时生成一张狗的图片。现在假设我们可以给每只狗附加一个随机标签，比如“dog3422”或“dog9879”。那么我们只需要给一只新的狗附加一个新的随机标签，比如“dog2221”，就可以得到一张新的狗的图片。

变分自动编码器形式化了这些直觉。变分自动编码器由两个卷积网络组成：编码器网络和解码器网络。编码器网络用于将图像转换为一个平坦的“嵌入”向量。解码器网络负责将嵌入向量转换为图像。为了确保解码器可以生成不同的图像，会添加噪音。图 6-13 展示了一个变分自动编码器。

![variational_autoencoder.png](img/tfdl_0613.png)

###### 图 6-13。变分自动编码器的示意图。变分自动编码器由两个卷积网络组成，编码器和解码器。

在实际实现中涉及更多细节，但变分自动编码器能够对图像进行采样。然而，朴素的变分编码器似乎生成模糊的图像样本，正如图 6-14 所示。这种模糊可能是因为*L*²损失不会严厉惩罚图像的模糊（回想我们关于*L*²不惩罚小偏差的讨论）。为了生成清晰的图像样本，我们将需要其他架构。

![variational-autoencoder-faces.jpg](img/tfdl_0614.png)

###### 图 6-14。从一个训练有素的人脸数据集上训练的变分自动编码器中采样的图像。请注意，采样的图像非常模糊。

### 对抗模型

L2 损失会严厉惩罚大的局部偏差，但不会严重惩罚许多小的局部偏差，导致模糊。我们如何设计一个替代的损失函数，更严厉地惩罚图像中的模糊？事实证明，编写一个能够解决问题的损失函数是相当具有挑战性的。虽然我们的眼睛可以很快发现模糊，但我们的分析工具并不那么快捕捉到这个问题。

如果我们能够“学习”一个损失函数会怎样？这个想法起初听起来有点荒谬；我们从哪里获取训练数据呢？但事实证明，有一个聪明的想法使这变得可行。

假设我们可以训练一个单独的网络来学习损失。让我们称这个网络为鉴别器。让我们称制作图像的网络为生成器。生成器可以与鉴别器对抗，直到生成器能够产生逼真的图像。这种架构通常被称为生成对抗网络，或 GAN。

由 GAN 生成的面部图像（图 6-15）比朴素变分自动编码器生成的图像要清晰得多（图 6-14）！GAN 已经取得了许多其他有希望的成果。例如，CycleGAN 似乎能够学习复杂的图像转换，例如将马转变为斑马，反之亦然。图 6-16 展示了一些 CycleGAN 图像转换。

![GAN_faces.png](img/tfdl_0615.png)

###### 图 6-15。从一个在面部数据集上训练的生成对抗网络（GAN）中采样的图像。请注意，采样的图像比变分自动编码器生成的图像更清晰。

![CycleGAN.jpg](img/tfdl_0616.png)

###### 图 6-16。CycleGAN 能够执行复杂的图像转换，例如将马的图像转换为斑马的图像（反之亦然）。

不幸的是，生成对抗网络在实践中仍然具有挑战性。使生成器和鉴别器学习合理的函数需要许多技巧。因此，虽然有许多令人兴奋的 GAN 演示，但 GAN 尚未发展到可以广泛部署在工业应用中的阶段。

# 在 TensorFlow 中训练卷积网络

在这一部分，我们考虑了一个用于训练简单卷积神经网络的代码示例。具体来说，我们的代码示例将演示如何使用 TensorFlow 在 MNIST 数据集上训练 LeNet-5 卷积架构。和往常一样，我们建议您通过运行与本书相关的[GitHub 存储库](https://github.com/matroid/dlwithtf)中的完整代码示例来跟随。

## MNIST 数据集

MNIST 数据集包含手写数字的图像。与 MNIST 相关的机器学习挑战包括创建一个在数字训练集上训练并推广到验证集的模型。图 6-17 展示了从 MNIST 数据集中绘制的一些图像。

![minst_images.png](img/tfdl_0617.png)

###### 图 6-17。来自 MNIST 数据集的一些手写数字图像。学习挑战是从图像中预测数字。

对于计算机视觉的机器学习方法的发展，MNIST 是一个非常重要的数据集。该数据集足够具有挑战性，以至于明显的非学习方法往往表现不佳。与此同时，MNIST 数据集足够小，以至于尝试新架构不需要非常大量的计算资源。

然而，MNIST 数据集大多已经过时。最佳模型实现了接近百分之百的测试准确率。请注意，这并不意味着手写数字识别问题已经解决！相反，很可能是人类科学家已经过度拟合了 MNIST 数据集的架构，并利用其特点实现了非常高的预测准确性。因此，不再建议使用 MNIST 来设计新的深度架构。尽管如此，MNIST 仍然是一个非常好的用于教学目的的数据集。

## 加载 MNIST

MNIST 代码库位于[Yann LeCun 的网站](http://yann.lecun.com/exdb/mnist/)上。下载脚本从网站下载原始文件。请注意脚本如何缓存下载，因此重复调用`download()`不会浪费精力。

作为一个更一般的说明，将机器学习数据集存储在云中，并让用户代码在处理之前检索数据，然后输入到学习算法中是非常常见的。我们在第四章中通过 DeepChem 库访问的 Tox21 数据集遵循了相同的设计模式。一般来说，如果您想要托管一个大型数据集进行分析，将其托管在云端并根据需要下载到本地机器进行处理似乎是一个不错的做法。（然而，对于非常大的数据集，网络传输时间变得非常昂贵。）请参见示例 6-1。

##### 示例 6-1。这个函数下载 MNIST 数据集

```py
def download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(WORK_DIRECTORY):
    os.makedirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    size = os.stat(filepath).st_size
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath
```

此下载检查`WORK_DIRECTORY`的存在。如果该目录存在，则假定 MNIST 数据集已经被下载。否则，脚本使用`urllib` Python 库执行下载并打印下载的字节数。

MNIST 数据集以字节编码的原始字符串形式存储像素值。为了方便处理这些数据，我们需要将其转换为 NumPy 数组。函数`np.frombuffer`提供了一个方便的方法，允许将原始字节缓冲区转换为数值数组（示例 6-2）。正如我们在本书的其他地方所指出的，深度网络可能会被占据广泛范围的输入数据破坏。为了稳定的梯度下降，通常需要将输入限制在一个有界范围内。原始的 MNIST 数据集包含从 0 到 255 的像素值。为了稳定性，这个范围需要被移动，使其均值为零，范围为单位（从-0.5 到+0.5）。

##### 示例 6-2。从下载的数据集中提取图像到 NumPy 数组

```py
def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

 Values are rescaled from [0, 255] down to [-0.5, 0.5].
 """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data
```

标签以简单的文件形式存储为字节字符串。有一个包含 8 个字节的标头，其余的数据包含标签（示例 6-3）。

##### 示例 6-3。这个函数将从下载的数据集中提取标签到一个标签数组中

```py
def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels
```

在前面示例中定义的函数的基础上，现在可以下载并处理 MNIST 训练和测试数据集（示例 6-4）。

##### 示例 6-4。使用前面示例中定义的函数，此代码片段下载并处理 MNIST 训练和测试数据集

```py
# Get the data.
train_data_filename = download('train-images-idx3-ubyte.gz')
train_labels_filename = download('train-labels-idx1-ubyte.gz')
test_data_filename = download('t10k-images-idx3-ubyte.gz')
test_labels_filename = download('t10k-labels-idx1-ubyte.gz')

# Extract it into NumPy arrays.
train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)
```

MNIST 数据集并没有明确定义用于超参数调整的验证数据集。因此，我们手动将训练数据集的最后 5,000 个数据点指定为验证数据（示例 6-5）。

##### 示例 6-5。提取训练数据的最后 5,000 个数据集用于超参数验证

```py
VALIDATION_SIZE = 5000  # Size of the validation set.

# Generate a validation set.
validation_data = train_data[:VALIDATION_SIZE, ...]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, ...]
train_labels = train_labels[VALIDATION_SIZE:]
```

# 选择正确的验证集

在示例 6-5 中，我们使用训练数据的最后一部分作为验证集来评估我们学习方法的进展。在这种情况下，这种方法相对无害。测试集中的数据分布在验证集中得到了很好的代表。

然而，在其他情况下，这种简单的验证集选择可能是灾难性的。在分子机器学习中（使用机器学习来预测分子的性质），测试分布几乎总是与训练分布截然不同。科学家最感兴趣的是*前瞻性*预测。也就是说，科学家希望预测从未针对该属性进行测试的分子的性质。在这种情况下，使用最后一部分训练数据进行验证，甚至使用训练数据的随机子样本，都会导致误导性地高准确率。分子机器学习模型在验证时具有 90%的准确率，而在测试时可能只有 60%是非常常见的。

为了纠正这个错误，有必要设计验证集选择方法，这些方法要尽力使验证集与训练集不同。对于分子机器学习，存在各种算法，大多数使用各种数学估计图的不相似性（将分子视为具有原子节点和化学键边的数学图）。

这个问题在许多其他机器学习领域也会出现。在医学机器学习或金融机器学习中，依靠历史数据进行预测可能是灾难性的。对于每个应用程序，重要的是要批判性地思考所选验证集上的性能是否实际上是真实性能的良好代理。

## TensorFlow 卷积原语

我们首先介绍用于构建我们的卷积网络的 TensorFlow 原语（示例 6-6）。

##### 示例 6-6。在 TensorFlow 中定义 2D 卷积

```py
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=None,
    data_format=None,
    name=None
)
```

函数`tf.nn.conv2d`是内置的 TensorFlow 函数，用于定义卷积层。这里，`input`被假定为形状为`(batch, height, width, channels)`的张量，其中`batch`是一个小批量中的图像数量。

请注意，先前定义的转换函数将 MNIST 数据读入此格式。参数`filter`是形状为`(filter_height, filter_width, channels, out_channels)`的张量，指定了在卷积核中学习的非线性变换的可学习权重。`strides`包含滤波器步幅，是长度为 4 的列表（每个输入维度一个）。

`padding`控制输入张量是否被填充（如图 6-18 中的额外零）以确保卷积层的输出与输入具有相同的形状。如果`padding="SAME"`，则填充`input`以确保卷积层输出与原始输入图像张量具有相同形状的图像张量。如果`padding="VALID"`，则不添加额外填充。

![conv_padding.png](img/tfdl_0618.png)

###### 图 6-18。卷积层的填充确保输出图像具有与输入图像相同的形状。

示例 6-7 中的代码定义了 TensorFlow 中的最大池化。

##### 示例 6-7。在 TensorFlow 中定义最大池化

```py
tf.nn.max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
)
```

`tf.nn.max_pool`函数执行最大池化。这里`value`与`tf.nn.conv2d`的`input`具有相同的形状，即`(batch, height, width, channels)`。`ksize`是池化窗口的大小，是长度为 4 的列表。`strides`和`padding`的行为与`tf.nn.conv2d`相同。

## 卷积架构

本节中定义的架构将与 LeNet-5 非常相似，LeNet-5 是最初用于在 MNIST 数据集上训练卷积神经网络的原始架构。在 LeNet-5 架构被发明时，计算成本非常昂贵，需要多周的计算才能完成训练。如今的笔记本电脑幸运地足以训练 LeNet-5 模型。图 6-19 展示了 LeNet-5 架构的结构。

![lenet5.png](img/tfdl_0619.png)

###### 图 6-19。LeNet-5 卷积架构的示意图。

# 更多计算会有什么不同？

LeNet-5 架构已有几十年历史，但实质上是解决数字识别问题的正确架构。然而，它的计算需求使得这种架构在几十年来相对默默无闻。因此，有趣的是，今天有哪些研究问题同样被解决，但仅仅受限于缺乏足够的计算能力？

一个很好的应用是视频处理。卷积模型在处理视频方面非常出色。然而，在大型视频数据集上存储和训练模型是不方便的，因此大多数学术论文不会报告视频数据集的结果。因此，要拼凑出一个良好的视频处理系统并不容易。

随着计算能力的增强，这种情况可能会发生变化，视频处理系统可能会变得更加普遍。然而，今天的硬件改进与过去几十年的硬件改进之间存在一个关键区别。与过去几年不同，摩尔定律的放缓明显。因此，硬件的改进需要更多的比自然晶体管缩小更多的东西，通常需要在架构设计上付出相当大的智慧。我们将在后面的章节中回到这个话题，并讨论深度网络的架构需求。

让我们定义训练 LeNet-5 网络所需的权重。我们首先定义一些用于定义权重张量的基本常量（示例 6-8）。

##### 示例 6-8。为 LeNet-5 模型定义基本常量

```py
NUM_CHANNELS = 1
IMAGE_SIZE = 28
NUM_LABELS = 10
```

我们定义的架构将使用两个卷积层交替使用两个池化层，最后是两个完全连接的层。请记住，池化不需要可学习的权重，因此我们只需要为卷积和完全连接的层创建权重。对于每个`tf.nn.conv2d`，我们需要创建一个与`tf.nn.conv2d`的`filter`参数对应的可学习权重张量。在这种特定的架构中，我们还将添加一个卷积偏置，每个输出通道一个（示例 6-9）。

##### 示例 6-9。为卷积层定义可学习的权重

```py
conv1_weights = tf.Variable(
    tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                        stddev=0.1,
                        seed=SEED, dtype=tf.float32))
conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
conv2_weights = tf.Variable(tf.truncated_normal(
    [5, 5, 32, 64], stddev=0.1,
    seed=SEED, dtype=tf.float32))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
```

请注意，卷积权重是 4 维张量，而偏置是 1 维张量。第一个完全连接的层将卷积层的输出转换为大小为 512 的向量。输入图像从大小`IMAGE_SIZE=28`开始。经过两个池化层（每个将输入减少 2 倍），我们最终得到大小为`IMAGE_SIZE//4`的图像。我们相应地创建完全连接权重的形状。

第二个完全连接的层用于提供 10 路分类输出，因此其权重形状为`(512,10)`，偏置形状为`(10)`，如示例 6-10 所示。

##### 示例 6-10。为完全连接的层定义可学习的权重

```py
fc1_weights = tf.Variable(  # fully connected, depth 512.
    tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                        stddev=0.1,
                        seed=SEED,
                        dtype=tf.float32))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                              stddev=0.1,
                                              seed=SEED,
                                              dtype=tf.float32))
fc2_biases = tf.Variable(tf.constant(
    0.1, shape=[NUM_LABELS], dtype=tf.float32))
```

所有权重定义完成后，我们现在可以自由定义网络的架构。该架构有六层，模式为 conv-pool-conv-pool-full-full（示例 6-11）。

##### 示例 6-11。定义 LeNet-5 架构。调用此示例中定义的函数将实例化架构。

```py
def model(data, train=False):
  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].
  conv = tf.nn.conv2d(data,
                      conv1_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  # Bias and rectified linear non-linearity.
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
  # Max pooling. The kernel size spec {ksize} also follows the layout of
  # the data. Here we have a pooling window of 2, and a stride of 2.
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  conv = tf.nn.conv2d(pool,
                      conv2_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  # Reshape the feature map cuboid into a 2D matrix to feed it to the
  # fully connected layers.
  pool_shape = pool.get_shape().as_list()
  reshape = tf.reshape(
      pool,
      [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  # Fully connected layer. Note that the '+' operation automatically
  # broadcasts the biases.
  hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
  # Add a 50% dropout during training only. Dropout also scales
  # activations such that no rescaling is needed at evaluation time.
  if train:
    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
  return tf.matmul(hidden, fc2_weights) + fc2_biases
```

如前所述，网络的基本架构交替使用`tf.nn.conv2d`、`tf.nn.max_pool`和非线性，以及最后一个完全连接的层。为了正则化，在最后一个完全连接的层之后应用一个 dropout 层，但只在训练期间。请注意，我们将输入作为参数`data`传递给函数`model()`。

网络中仍需定义的唯一部分是占位符（示例 6-12）。我们需要定义两个占位符，用于输入训练图像和训练标签。在这个特定的网络中，我们还定义了一个用于评估的单独占位符，允许我们在评估时输入更大的批次。

##### 示例 6-12。为架构定义占位符

```py
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64

train_data_node = tf.placeholder(
    tf.float32,
    shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
eval_data = tf.placeholder(
    tf.float32,
    shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
```

有了这些定义，我们现在已经处理了数据，指定了输入和权重，并构建了模型。我们现在准备训练网络（示例 6-13）。

##### 示例 6-13。训练 LeNet-5 架构

```py
# Create a local session to run the training.
start_time = time.time()
with tf.Session() as sess:
  # Run all the initializers to prepare the trainable parameters.
  tf.global_variables_initializer().run()
  # Loop through training steps.
  for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    # This dictionary maps the batch data (as a NumPy array) to the
    # node in the graph it should be fed to.
    feed_dict = {train_data_node: batch_data,
                 train_labels_node: batch_labels}
    # Run the optimizer to update weights.
    sess.run(optimizer, feed_dict=feed_dict)
```

这个拟合代码的结构看起来与本书迄今为止看到的其他拟合代码非常相似。在每一步中，我们构建一个 feed 字典，然后运行优化器的一步。请注意，我们仍然使用小批量训练。

## 评估经过训练的模型

我们现在有一个正在训练的模型。我们如何评估训练模型的准确性？一个简单的方法是定义一个错误度量。与前几章一样，我们将使用一个简单的分类度量来衡量准确性（示例 6-14）。

##### 示例 6-14。评估经过训练的架构的错误

```py
def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])
```

我们可以使用这个函数来评估网络在训练过程中的错误。让我们引入一个额外的方便函数，以批处理的方式评估任何给定数据集上的预测（示例 6-15）。这种便利是必要的，因为我们的网络只能处理固定批量大小的输入。

##### 示例 6-15。一次评估一批数据

```py
def eval_in_batches(data, sess):
  """Get predictions for a dataset by running it in small batches."""
  size = data.shape[0]
  if size < EVAL_BATCH_SIZE:
    raise ValueError("batch size for evals larger than dataset: %d"
                     % size)
  predictions = numpy.ndarray(shape=(size, NUM_LABELS),
                              dtype=numpy.float32)
  for begin in xrange(0, size, EVAL_BATCH_SIZE):
    end = begin + EVAL_BATCH_SIZE
    if end <= size:
      predictions[begin:end, :] = sess.run(
          eval_prediction,
          feed_dict={eval_data: data[begin:end, ...]})
    else:
      batch_predictions = sess.run(
          eval_prediction,
          feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
      predictions[begin:, :] = batch_predictions[begin - size:, :]
  return predictions
```

现在我们可以在训练过程中的内部`for`循环中添加一些仪器（instrumentation），定期评估模型在验证集上的准确性。我们可以通过评分测试准确性来结束训练。示例 6-16 展示了添加了仪器的完整拟合代码。

##### 示例 6-16。训练网络的完整代码，添加了仪器

```py
# Create a local session to run the training.
start_time = time.time()
with tf.Session() as sess:
  # Run all the initializers to prepare the trainable parameters.
  tf.global_variables_initializer().run()
  # Loop through training steps.
  for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    # This dictionary maps the batch data (as a NumPy array) to the
    # node in the graph it should be fed to.
    feed_dict = {train_data_node: batch_data,
                 train_labels_node: batch_labels}
    # Run the optimizer to update weights.
    sess.run(optimizer, feed_dict=feed_dict)
    # print some extra information once reach the evaluation frequency
    if step % EVAL_FREQUENCY == 0:
      # fetch some extra nodes' data
      l, lr, predictions = sess.run([loss, learning_rate,
                                     train_prediction],
                                    feed_dict=feed_dict)
      elapsed_time = time.time() - start_time
      start_time = time.time()
      print('Step %d (epoch %.2f), %.1f ms' %
            (step, float(step) * BATCH_SIZE / train_size,
             1000 * elapsed_time / EVAL_FREQUENCY))
      print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
      print('Minibatch error: %.1f%%'
            % error_rate(predictions, batch_labels))
      print('Validation error: %.1f%%' % error_rate(
          eval_in_batches(validation_data, sess), validation_labels))
      sys.stdout.flush()
  # Finally print the result!
  test_error = error_rate(eval_in_batches(test_data, sess),
                          test_labels)
  print('Test error: %.1f%%' % test_error)
```

## 读者的挑战

尝试自己训练网络。您应该能够达到<1%的测试错误！

# 回顾

在这一章中，我们向您展示了卷积网络设计的基本概念。这些概念包括构成卷积网络核心构建模块的卷积和池化层。然后我们讨论了卷积架构的应用，如目标检测、图像分割和图像生成。我们以一个深入的案例研究结束了这一章，向您展示了如何在 MNIST 手写数字数据集上训练卷积架构。

在第七章中，我们将介绍循环神经网络，另一个核心深度学习架构。与为图像处理而设计的卷积网络不同，循环架构非常适合处理顺序数据，如自然语言数据集。
