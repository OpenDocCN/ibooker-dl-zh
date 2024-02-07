# 第六章. 高级模型和 UI

> “做到之前总是看似不可能。”
> 
> —纳尔逊·曼德拉

您已经有了理解模型的基线。您已经消化和利用了模型，甚至在叠加中显示了结果。看起来可能无限制。但是，您已经看到模型往往以各种复杂的方式返回信息。对于井字棋模型，您只想要一个移动，但它仍然返回所有九个可能的框，留下了一些清理工作，然后您才能利用模型的输出。随着模型变得更加复杂，这个问题可能会加剧。在本章中，我们将选择一个广泛和复杂的模型类型进行对象检测，并通过 UI 和概念来全面了解可能遇到的任务。

让我们回顾一下您当前的工作流程。首先，您选择一个模型。确定它是一个 Layers 模型还是 Graph 模型。即使您没有这些信息，您也可以通过尝试以某种方式加载它来弄清楚。

接下来，您需要确定模型的输入和输出，不仅是形状，还有数据实际代表的内容。您批处理数据，对模型调用`predict`，然后输出就可以了，对吗？

不幸的是，您还应该知道一些其他内容。一些最新和最伟大的模型与您所期望的有显著差异。在许多方面，它们要优越得多，但在其他方面，它们更加繁琐。不要担心，因为您已经在上一章中建立了张量和画布叠加的坚实基础。通过一点点指导，您可以处理这个新的高级模型世界。

我们将：

+   深入了解理论如何挑战您的张量技能

+   了解高级模型特性

+   学习许多新的图像和机器学习术语

+   确定绘制多个框以进行对象检测的最佳方法

+   学习如何在画布上为检测绘制标签

当您完成本章时，您将对实现高级 TensorFlow.js 模型的理论要求有深刻的理解。本章作为对您今天可以使用的最强大模型之一的认知演练，伴随着大量的学习。这不会很难，但请做好学习准备，并不要回避复杂性。如果您遵循本章中解释的逻辑，您将对机器学习的核心理论和实践有深刻的理解和掌握。

# 再谈 MobileNet

当您在[TFHub.dev](https://tfhub.dev)上浏览时，您可能已经看到我们的老朋友 MobileNet 以许多不同的风格和版本被提及。一个版本有一个简单的名字，`ssd_mobilenet_v2`，用于图像对象检测（请参见图 6-1 中的突出显示部分）。

多么令人兴奋！看起来您可以从之前的 TensorFlow Hub 示例中获取代码，并将模型更改为查看一组边界框及其相关类，对吗？

![MobileNet SSD on TFHub](img/ltjs_0601.png)

###### 图 6-1. 用于对象检测的 MobileNet

这样做后，您立即收到一个失败消息，要求您使用`model.executeAsync`而不是`model.predict`（请参见图 6-2）。

![MobileNet Predict Error](img/ltjs_0602.png)

###### 图 6-2. 预测不起作用

那么出了什么问题？到目前为止，您可能有一大堆问题。

+   模型想要的`executeAsync`是什么？

+   为什么这个 MobileNet 模型用于对象检测？

+   为什么这个模型规范不关心输入大小？

+   “SSD”在机器学习中的名称中代表什么？

###### 警告

在 Parcel 中，您可能会收到关于`regeneratorRuntime`未定义的错误。这是由于 Babel polyfill 中的弃用。如果您遇到此错误，您可以添加`core-js`和`regenerator-runtime`包并在主文件中导入它们。如果您遇到此问题，请参见本章的相关[GitHub 代码](https://oreil.ly/LKc8v)。

这是一个需要更多信息、理论和历史来理解的高级模型的完美例子。现在也是学习一些我们为方便起见而保留的概念的好时机。通过本章的学习，您将准备好处理一些新术语、最佳实践和复杂模型的特性。

## SSD MobileNet

到目前为止，书中已经提到了两种模型的名称，但没有详细说明。MobileNet 和 Inception 是由谷歌 AI 团队创建的已发布的模型架构。在下一章中，您将设计自己的模型，但可以肯定地说，它们不会像这两个知名模型那样先进。每个模型都有一组特定的优点和缺点。准确性并不总是模型的唯一度量标准。

MobileNet 是一种用于低延迟、低功耗模型的特定架构。这使得它非常适合设备和网络。尽管基于 Inception 的模型更准确，但 MobileNet 的速度和尺寸使其成为边缘设备上分类和对象检测的标准工具。

查看由谷歌发布的[性能和延迟图表，比较设备上的模型版本](https://oreil.ly/dHEKZ)。您可以看到，尽管 Inception v2 的大小是 MobileNetV2 的几倍，需要更多计算才能进行单个预测，但 MobileNetV2 速度更快，虽然准确性不如 Inception，但仍然接近。MobileNetV3 甚至有望在尺寸略微增加的情况下更准确。这些模型的核心研究和进展使它们成为经过良好测试的资源，具有已知的权衡。正是因为这些原因，您会看到相同的模型架构在新问题中反复使用。

前面提到的这两种架构都是由谷歌用数百万张图片进行训练的。MobileNet 和 Inception 可以识别的经典 1,001 类来自一个名为[ImageNet](https://image-net.org/about.php)的知名数据集。因此，在云中的许多计算机上进行长时间训练后，这些模型被调整为立即使用。虽然这些模型是分类模型，但它们也可以被重新用于检测对象。

就像建筑物一样，模型可以稍作修改以处理不同的目标。例如，一个剧院可以从最初用于举办现场表演的目的进行修改，以便支持 3D 特色电影。是的，需要进行一些小的更改，但整体架构是可以重复使用的。对于从分类到对象检测重新用途的模型也是如此。

有几种不同的方法可以进行对象检测。一种方法称为*基于区域的卷积神经网络*（R-CNN）。不要将 R-CNN 与 RNN 混淆，它们是完全不同的，是机器学习中的真实事物。基于区域的卷积神经网络听起来可能像《哈利波特》中的咒语，但实际上只是通过查看图像的补丁来检测对象的一种流行方法，使用滑动窗口（即重复采样图像的较小部分，直到覆盖整个图像）。R-CNN 通常速度较慢，但非常准确。慢速方面与网站和移动设备不兼容。

检测对象的第二种流行方法是使用另一个时髦词汇，“完全卷积”方法（有关卷积的更多信息，请参阅第十章）。这些方法没有深度神经网络，这就是为什么它们避免需要特定的输入尺寸。没错，您不需要为完全卷积方法调整图像大小，而且它们也很快。

这就是 SSD MobileNet 中的“SSD”之所以重要的地方。它代表*单次检测器*。是的，您和我可能一直在想固态驱动器，但命名事物可能很困难，所以我们将数据科学放过。SSD 模型类型被设计为完全卷积模型，一次性识别图像的特征。这种“单次检测”使 SSD 比 R-CNN 快得多。不深入细节，SSD 模型有两个主要组件，一个*骨干模型*，它了解如何识别对象，以及一个*SSD 头部*，用于定位对象。在这种情况下，骨干是快速友好的 MobileNet。

结合 MobileNet 和 SSD 需要一点魔法，称为*控制流*，它允许您在模型中有条件地运行操作。这就是使`predict`方法从简单变得需要异步调用`executeAsync`的原因。当模型实现控制流时，同步的`predict`方法将无法工作。

条件逻辑通常由本地语言处理，但这会显著减慢速度。虽然大多数 TensorFlow.js 可以通过利用 GPU 或 Web Assembly（WASM）后端进行优化，但 JavaScript 中的条件语句需要卸载优化张量并重新加载它们。SSD MobileNet 模型为您隐藏了这个头疼的问题，只需使用控制流操作的低成本。虽然实现控制流超出了本书的范围，但使用这些高级功能的模型并不是。

由于这个模型的现代性，它不是为处理图像批次而设置的。这意味着输入的唯一限制不是图像大小，而是批量大小。但是，它确实期望一个批量为 1，因此一个 1,024×768 的 RGB 图像将以`[1, 768, 1024, 3]`的形式输入到该模型中，其中`1`是批量的堆栈大小，`768`是图像高度，`1024`是图像宽度，`3`是每个像素的 RGB 值。

深入了解您将处理的输入和输出类型非常重要。值得注意的是，模型的输出边界框遵循输入的经典高度和宽度架构，与宠物面部检测器不同。这意味着边界框将是`[y1, x1, y2, x2]`而不是`[x1, y1, x2, y2]`。如果不注意到这些小问题，可能会非常令人沮丧。您的边界框看起来会完全错乱。每当您实现一个新模型时，重要的是您从所有可用的文档中验证规范。

在深入代码之前还有一个注意事项。根据我的经验，生产中的目标检测很少用于识别成千上万种不同的类别，就像您在 MobileNet 和 Inception 中看到的那样。这样做有很多很好的理由，因此目标检测通常在少数类别上进行测试和训练。人们用于目标检测训练的一个常见组标记数据是[Microsoft Common Objects in Context（COCO）](https://cocodataset.org/#home)数据集。这个 SSD MobileNet 使用了该数据集来教会模型看到 80 种不同的类别。虽然 80 种类别比 1,001 种可能的类别要少很多，但仍然是一个令人印象深刻的集合。

现在您对 SSD MobileNet 的了解比大多数使用它的人更多。您知道它是一个使用控制流将 MobileNet 速度与 80 个类别的 SSD 结果联系起来的目标检测模型。这些知识将帮助您以后解释模型的结果。

# 边界框输出

现在您了解了模型，可以获得结果。在这个模型中，`executeAsync`返回的值是两个张量堆栈的普通 JavaScript 数组。第一个张量堆栈是检测到的内容，第二个张量堆栈是每个检测的边界框堆栈，换句话说，分数和它们的框。

## 阅读模型输出

你可以通过几行代码查看图像的结果。以下代码就是这样做的，也可以在[本章的源代码](https://oreil.ly/JLo5C)中找到：

```py
tf.ready().then(()=>{constmodelPath="https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1";![1](img/1.png)tf.tidy(()=>{tf.loadGraphModel(modelPath,{fromTFHub: true}).then((model)=>{constmysteryImage=document.getElementById("mystery");constmyTensor=tf.browser.fromPixels(mysteryImage);// SSD Mobilenet batch of 1
constsingleBatch=tf.expandDims(myTensor,0);![2](img/2.png)model.executeAsync(singleBatch).then((result)=>{console.log("First",result[0].shape);![3](img/3.png)result[0].print();console.log("Second",result[1].shape);![4](img/4.png)result[1].print();});});});});
```

![1](img/#co_advanced_models_and_ui_CO1-1)

这是 JavaScript 模型的 TFHub URL。

![2](img/#co_advanced_models_and_ui_CO1-2)

输入在秩上扩展为一个批次，形状为[1, 高度, 宽度, 3]。

![3](img/#co_advanced_models_and_ui_CO1-3)

得到的张量是[1, 1917, 90]，其中返回了 1,917 个检测结果，每行中的 90 个概率值加起来为 1。

![4](img/#co_advanced_models_and_ui_CO1-4)

张量的形状为[1, 1917, 4]，为 1,917 个检测提供了边界框。

图 6-3 显示了模型的输出。

![控制台中的 SSD MobileNet 输出](img/ltjs_0603.png)

###### 图 6-3。前一段代码的输出

###### 注

你可能会惊讶地看到 90 个值而不是 80 个可能的类别。仍然只有 80 个可能的类别。该模型中的十个结果索引未被使用。

虽然看起来你已经完成了，但还有一些警告信号。正如你可能想象的那样，绘制 1,917 个边界框不会有用或有效，但是尝试一下看看。

## 显示所有输出

是时候编写代码来绘制多个边界框了。直觉告诉我们，1,917 个检测结果太多了。现在是时候编写一些代码来验证一下了。由于代码变得有点过于依赖 promise，现在是时候切换到 async/await 了。这将阻止代码进一步缩进，并提高可读性。如果你不熟悉在 promise 和 async/await 之间切换，请查看 JavaScript 的相关部分。

绘制模型检测的完整代码可以在[书籍源代码文件*too_many.html*](https://oreil.ly/bMPVa)中找到。这段代码使用了与上一章节中对象定位部分描述的相同技术，但参数顺序已调整以适应模型的预期输出。

```py
const results = await model.executeAsync(readyfied);
const boxes = await results[1].squeeze().array();

// Prep Canvas
const detection = document.getElementById("detection");
const ctx = detection.getContext("2d");
const imgWidth = mysteryImage.width;
const imgHeight = mysteryImage.height;
detection.width = imgWidth;
detection.height = imgHeight;

boxes.forEach((box, idx) => {
  ctx.strokeStyle = "#0F0";
  ctx.lineWidth = 1;
  const startY = box[0] * imgHeight;
  const startX = box[1] * imgWidth;
  const height = (box[2] - box[0]) * imgHeight;
  const width = (box[3] - box[1]) * imgWidth;
  ctx.strokeRect(startX, startY, width, height);
});
```

无论模型的置信度如何，绘制每个检测并不困难，但结果输出完全无法使用，如图 6-4 所示。

![检测结果过多](img/ltjs_0604.png)

###### 图 6-4。1,917 个边界框，使图像无用

在图 6-4 中看到的混乱表明有大量的检测结果，但没有清晰度。你能猜到是什么导致了这种噪音吗？导致你看到的噪音有两个因素。

# 检测清理

对结果边界框的第一个批评是没有质量或数量检查。代码没有检查检测值的概率或过滤最有信心的值。你可能不知道，模型可能只有 0.001%的确信度，那么微小的检测就不值得绘制边界框。清理的第一步是设置检测分数的最小阈值和最大边界框数量。

其次，在仔细检查后，绘制的边界框似乎一遍又一遍地检测到相同的对象，只是略有变化。稍后将对此进行验证。最好是当它们识别出相同类别时，它们的重叠应该受到限制。如果两个重叠的框都检测到一个人，只保留检测分数最高的那个。

模型在照片中找到了东西（或没有找到），现在轮到你来进行清理了。

## 质量检查

你需要最高排名的预测。你可以通过抑制低于给定分数的任何边界框来实现这一点。通过一次调用`topk`来识别整个检测系列中的最高分数，如下所示：

```py
const prominentDetection = tf.topk(results[0]);
// Print it to be sure
prominentDetection.indices.print()
prominentDetection.values.print()
```

对所有检测结果调用`topk`将返回一个仅包含最佳结果的数组，因为`k`默认为`1`。每个检测的索引对应于类别，值是检测的置信度。输出看起来会像图 6-5。

![整个批次的 Topk 检测日志](img/ltjs_0605.png)

###### 图 6-5。`topk`调用适用于整个批次

如果显著检测低于给定阈值，您可以拒绝绘制框。然后，您可以限制绘制的框，仅绘制前 N 个预测。我们将把这个练习的代码留给章节挑战，因为它无法解决第二个问题。仅仅进行质量检查会导致在最强预测周围出现一堆框，而不是单个预测。结果框看起来就像您的检测系统喝了太多咖啡（参见图 6-6）。

![框重叠](img/ltjs_0606.png)

###### 图 6-6。绘制前 20 个预测会产生模糊的边框

幸运的是，有一种内置的方法来解决这些模糊的框，并为您的晚餐派对提供一些新术语。

## IoUs 和 NMS

直到现在，您可能认为 IoUs 只是由 Lloyd Christmas 支持的一种获得批准的法定货币，但在目标检测和训练领域，它们代表*交集与并集*。交集与并集是用于识别对象检测器准确性和重叠的评估指标。准确性部分对于训练非常重要，而重叠部分对于清理重叠输出非常重要。

IoU 是用于确定两个框在重叠中共享多少面积的公式。如果框完全重叠，IoU 为 1，而它们的适合程度越低，数字就越接近零。标题“IoU”来自于这个计算公式。框的交集面积除以框的并集面积，如图 6-7 所示。

![IoU 图](img/ltjs_0607.png)

###### 图 6-7。交集与并集

现在您有一个快速的公式来检查边界框的相似性。使用 IoU 公式，您可以执行一种称为*非极大值抑制*（NMS）的算法来消除重复。NMS 会自动获取得分最高的框，并拒绝任何 IoU 超过指定水平的相似框。图 6-8 展示了一个包含三个得分框的简单示例。

![非极大值抑制图](img/ltjs_0608.png)

###### 图 6-8。只有最大值存活；其他得分较低的框被移除

如果将 NMS 的 IoU 设置为 0.5，则任何与得分更高的框共享 50%面积的框将被删除。这对于消除与同一对象重叠的框非常有效。但是，对于彼此重叠并应该有两个边界框的两个对象可能会出现问题。这对于具有不幸角度的真实对象是一个问题，因为它们的边界框将相互抵消，您只会得到两个实际对象的一个检测。对于这种情况，您可以启用一个名为[Soft-NMS](https://arxiv.org/pdf/1704.04503.pdf)的 NMS 的高级版本，它将降低重叠框的分数而不是删除它们。如果它们在被降低后的分数仍然足够高，检测结果将存活并获得自己的边界框，即使 IoU 非常高。图 6-9 使用 Soft-NMS 正确识别了与极高交集的两个对象。

![Soft-NMS 示例](img/ltjs_0609.png)

###### 图 6-9。即使是真实世界中重叠的对象也可以使用 Soft-NMS 进行检测

Soft-NMS 最好的部分是它内置在 TensorFlow.js 中。我建议您为所有目标检测需求使用这个 TensorFlow.js 函数。在这个练习中，您将使用内置的方法，名为`tf.image.nonMaxSuppressionWithScoreAsync`。TensorFlow.js 内置了许多 NMS 算法，但`tf.image.nonMaxSuppressionWithScoreAsync`具有两个优点，使其非常适合使用：

+   `WithScore`提供 Soft-NMS 支持。

+   `Async`可以阻止 GPU 锁定 UI 线程。

在使用非异步高级方法时要小心，因为它们可能会锁定整个 UI。如果出于任何原因想要移除 Soft-NMS 方面，可以将最后一个参数（Soft-NMS Sigma）设置为零，然后您就得到了传统的 NMS。

```py
const nmsDetections = await tf.image.nonMaxSuppressionWithScoreAsync(
  justBoxes, // shape [numBoxes, 4]
  justValues, // shape [numBoxes]
  maxBoxes, // Stop making boxes when this number is hit
  iouThreshold, // Allowed overlap value 0 to 1
  detectionThreshold, // Minimum detection score allowed
  1 // 0 is normal NMS, 1 is max Soft-NMS
);
```

只需几行代码，您就可以将 SSD 结果澄清为几个清晰的检测结果。

结果将是一个具有两个属性的对象。`selectedIndices`属性将是一个张量，其中包含通过筛选的框的索引，`selectedScores`将是它们对应的分数。您可以循环遍历所选结果并绘制边界框。

```py
constchosen=awaitnmsDetections.selectedIndices.data();![1](img/1.png)chosen.forEach((detection)=>{ctx.strokeStyle="#0F0";ctx.lineWidth=4;constdetectedIndex=maxIndices[detection];![2](img/2.png)constdetectedClass=CLASSES[detectedIndex];![3](img/3.png)constdetectedScore=scores[detection];constdBox=boxes[detection];console.log(detectedClass,detectedScore);![4](img/4.png)// No negative values for start positions
conststartY=dBox[0]>0?dBox[0]*imgHeight : 0;![5](img/5.png)conststartX=dBox[1]>0?dBox[1]*imgWidth : 0;constheight=(dBox[2]-dBox[0])*imgHeight;constwidth=(dBox[3]-dBox[1])*imgWidth;ctx.strokeRect(startX,startY,width,height);});
```

![1](img/#co_advanced_models_and_ui_CO2-1)

从结果中高得分的框的索引创建一个普通的 JavaScript 数组。

![2](img/#co_advanced_models_and_ui_CO2-2)

从先前的`topk`调用中获取最高得分的索引。

![3](img/#co_advanced_models_and_ui_CO2-3)

将类别作为数组导入以匹配给定的结果索引。这种结构就像上一章中 Inception 示例中的代码一样。

![4](img/#co_advanced_models_and_ui_CO2-4)

记录在画布中被框定的内容，以便验证结果。

![5](img/#co_advanced_models_and_ui_CO2-5)

禁止负数，以便框至少从帧开始。否则，一些框将从左上角被切断。

返回的检测数量各不相同，但受限于 NMS 中设置的规格。示例代码导致了五个正确的检测结果，如图 6-10 所示。

![非最大抑制结果](img/ltjs_0610.png)

###### 图 6-10。干净的 Soft-NMS 检测结果

循环中的控制台日志打印出五个检测结果分别为三个“人”检测、一个“酒杯”和一个“餐桌”。将图 6-11 中的五个日志与图 6-10 中的五个边界框进行比较。

![非最大抑制结果日志](img/ltjs_0611.png)

###### 图 6-11。结果日志类别和置信水平

UI 已经取得了很大进展。覆盖层应该能够识别检测和它们的置信度百分比是合理的。普通用户不知道要查看控制台以查看日志。

# 添加文本覆盖

您可以以各种花式方式向画布添加文本，并使其识别相关的边界框。在此演示中，我们将回顾最简单的方法，并将更美观的布局留给读者作为任务。

可以使用画布 2D 上下文的`fillText`方法向画布绘制文本。您可以通过重复使用绘制框时使用的`X, Y`坐标将文本定位在每个框的左上角。

有两个绘制文本时需要注意的问题：

+   文本与背景之间很容易出现低对比度。

+   与同时绘制的框相比，同时绘制的文本可能会被后绘制的框覆盖。

幸运的是，这两个问题都很容易解决。

## 解决低对比度

创建可读标签的典型方法是绘制一个背景框，然后放置文本。正如您所知，`strokeRect`创建一个没有填充颜色的框，所以不应该感到意外的是`fillRect`绘制一个带有填充颜色的框。

矩形应该有多大？一个简单的答案是将矩形绘制到检测框的宽度，但不能保证框足够宽，当框非常宽时，这会在结果中创建大的阻挡条。唯一有效的解决方案是测量文本并相应地绘制框。文本高度可以通过利用上下文`font`属性来设置，宽度可以通过`measureText`确定。

最后，您可能需要考虑从绘图位置减去字体高度，以便将文本绘制在框内而不是在框上方，但上下文已经有一个属性可以设置以保持简单。`context.textBaseline`属性有各种选项。图 6-12 显示了每个可能属性选项的起始点。

![文本基线选项](img/ltjs_0612.png)

###### 图 6-12。将`textBaseline`设置为`top`可以保持文本在 X 和 Y 坐标内

现在你知道如何绘制一个填充矩形到适当的大小并将标签放在内部。您可以将这些方法结合在您的`forEach`循环中，您在其中绘制检测结果。标签绘制在每个检测的左上角，如图 6-13 所示。

![显示绘制结果](img/ltjs_0613.png)

###### 图 6-13。标签与每个框一起绘制

重要的是文本在背景框之后绘制，否则框将覆盖文本。对于我们的目的，标签将使用略有不同颜色的绿色绘制，而不是边界框。

```py
// Draw the label background. ctx.fillStyle="#0B0";ctx.font="16px sans-serif";![1](img/1.png)ctx.textBaseline="top";![2](img/2.png)consttextHeight=16;consttextPad=4;![3](img/3.png)constlabel=`${detectedClass}${Math.round(detectedScore*100)}%`;consttextWidth=ctx.measureText(label).width;ctx.fillRect(![4](img/4.png)startX,startY,textWidth+textPad,textHeight+textPad);// Draw the text last to ensure it's on top. ctx.fillStyle="#000000";![5](img/5.png)ctx.fillText(label,startX,startY);![6](img/6.png)
```

![1](img/#co_advanced_models_and_ui_CO3-1)

设置标签使用的字体和大小。

![2](img/#co_advanced_models_and_ui_CO3-2)

设置`textBaseline`如上所述。

![3](img/#co_advanced_models_and_ui_CO3-3)

添加一点水平填充以在`fillRect`渲染中使用。

![4](img/#co_advanced_models_and_ui_CO3-4)

使用相同的`startX`和`startY`绘制矩形，这与绘制边界框时使用的相同。

![5](img/#co_advanced_models_and_ui_CO3-5)

将`fillStyle`更改为黑色以进行文本渲染。

![6](img/#co_advanced_models_and_ui_CO3-6)

最后，绘制文本。这可能也应该略微填充。

现在每个检测都有一个几乎可读的标签。但是，根据您的图像，您可能已经注意到了一些问题，我们现在将解决。

## 解决绘制顺序

尽管标签是绘制在框的上方，但框是在不同的时间绘制的，可以轻松重叠一些现有标签文本，使它们难以阅读甚至不可能阅读。如您在图 6-14 中所见，由于重叠检测，餐桌百分比很难阅读。

![上下文重叠问题](img/ltjs_0614.png)

###### 图 6-14。上下文绘制顺序重叠问题

解决这个问题的一种方法是遍历检测结果并绘制框，然后再进行第二次遍历并绘制文本。这将确保文本最后绘制，但代价是需要在两个连续循环中遍历检测结果。

作为替代方案，您可以使用代码处理这个问题。您可以设置上下文`globalCompositeOperation`来执行各种令人惊奇的操作。一个简单的操作是告诉上下文在现有内容的上方或下方渲染，有效地设置 z 顺序。

`strokeRect`调用可以设置为`globalCompositeOperation`为`destination-over`。这意味着任何存在于目标中的像素将获胜并放置在添加的内容上方。这有效地在任何现有内容下绘制。

然后，在绘制标签时，将`globalCompositionOperation`返回到其默认行为，即`source-over`。这会将新的源像素绘制在任何现有绘图上。如果在这两种操作之间来回切换，您可以确保您的标签是最优先的，并在主循环内处理所有内容。

总的来说，绘制边界框、标签框和标签的单个循环如下所示：

```py
chosen.forEach((detection)=>{ctx.strokeStyle="#0F0";ctx.lineWidth=4;ctx.globalCompositeOperation='destination-over';![1](img/1.png)constdetectedIndex=maxIndices[detection];constdetectedClass=CLASSES[detectedIndex];constdetectedScore=scores[detection];constdBox=boxes[detection];// No negative values for start positions
conststartY=dBox[0]>0?dBox[0]*imgHeight : 0;conststartX=dBox[1]>0?dBox[1]*imgWidth : 0;constheight=(dBox[2]-dBox[0])*imgHeight;constwidth=(dBox[3]-dBox[1])*imgWidth;ctx.strokeRect(startX,startY,width,height);// Draw the label background.
ctx.globalCompositeOperation='source-over';![2](img/2.png)ctx.fillStyle="#0B0";consttextHeight=16;consttextPad=4;constlabel=`${detectedClass}${Math.round(detectedScore*100)}%`;consttextWidth=ctx.measureText(label).width;ctx.fillRect(startX,startY,textWidth+textPad,textHeight+textPad);// Draw the text last to ensure it's on top.
ctx.fillStyle="#000000";ctx.fillText(label,startX,startY);});
```

![1](img/#co_advanced_models_and_ui_CO4-1)

在任何现有内容下绘制。

![2](img/#co_advanced_models_and_ui_CO4-2)

在任何现有内容上绘制。

结果是一个动态的人类可读结果，您可以与您的朋友分享（参见图 6-15）。

![完全工作的目标检测](img/ltjs_0615.png)

###### 图 6-15。使用`destination-over`修复重叠问题

# 连接到网络摄像头

所有这些速度的好处是什么？正如前面提到的，选择 SSD 而不是 R-CNN，选择 MobileNet 而不是 Inception，以及一次绘制画布而不是两次。当你加载页面时，它看起来相当慢。似乎至少需要四秒才能加载和渲染。

是的，把一切都放在适当的位置需要一点时间，但在内存分配完毕并且模型下载完成后，你会看到一些相当显著的速度。是的，足以在你的网络摄像头上运行实时检测。

加快流程的关键是运行设置代码一次，然后继续运行检测循环。这意味着你需要将这节课的庞大代码库分解；否则，你将得到一个无法使用的界面。为简单起见，你可以按照示例 6-1 中所示分解项目。

##### 示例 6-1。分解代码库

```py
asyncfunctiondoStuff() {try{constmodel=awaitloadModel()![1](img/1.png)constmysteryVideo=document.getElementById('mystery')![2](img/2.png)constcamDetails=awaitsetupWebcam(mysteryVideo)![3](img/3.png)performDetections(model,mysteryVideo,camDetails)![4](img/4.png)}catch(e){console.error(e)![5](img/5.png)}}
```

![1](img/#co_advanced_models_and_ui_CO5-1)

加载模型时最长的延迟应该首先发生，且仅发生一次。

![2](img/#co_advanced_models_and_ui_CO5-2)

为了效率，你可以一次捕获视频元素，并将该引用传递到需要的地方。

![3](img/#co_advanced_models_and_ui_CO5-3)

设置网络摄像头应该只发生一次。

![4](img/#co_advanced_models_and_ui_CO5-4)

`performDetections`方法可以在检测网络摄像头中的内容并绘制框时无限循环。

![5](img/#co_advanced_models_and_ui_CO5-5)

不要让所有这些`awaits`吞没错误。

## 从图像到视频的转换

从静态图像转换为视频实际上并不复杂，因为将所见内容转换为张量的困难部分由`tf.fromPixels`处理。`tf.fromPixels`方法可以读取画布、图像，甚至视频元素。因此，复杂性在于将`img`标签更改为`video`标签。

你可以通过更换标签来开始。原始的`img`标签：

```py
<img id="mystery" src="/dinner.jpg" height="100%" />
```

变成以下内容：

```py
<video id="mystery" height="100%" autoplay></video>
```

值得注意的是，视频元素的宽度/高度属性稍微复杂，因为有输入视频的宽度/高度和实际客户端的宽度/高度。因此，所有使用`width`的计算都需要使用`clientWidth`，同样，`height`需要使用`clientHeight`。如果使用错误的属性，框将不对齐，甚至可能根本不显示。

## 激活网络摄像头

为了我们的目的，我们只会设置默认的网络摄像头。这对应于示例 6-1 中的第四点。如果你对`getUserMedia`不熟悉，请花点时间分析视频元素如何连接到网络摄像头。这也是你可以将画布上下文设置移动到适应视频元素的时间。

```py
asyncfunctionsetupWebcam(videoRef){if(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia){constwebcamStream=awaitnavigator.mediaDevices.getUserMedia({![1](img/1.png)audio: false,video:{facingMode:'user',},})if('srcObject'invideoRef){![2](img/2.png)videoRef.srcObject=webcamStream}else{videoRef.src=window.URL.createObjectURL(webcamStream)}returnnewPromise((resolve,_)=>{![3](img/3.png)videoRef.onloadedmetadata=()=>{![4](img/4.png)// Prep Canvas
constdetection=document.getElementById('detection')constctx=detection.getContext('2d')constimgWidth=videoRef.clientWidth![5](img/5.png)constimgHeight=videoRef.clientHeightdetection.width=imgWidthdetection.height=imgHeightctx.font='16px sans-serif'ctx.textBaseline='top'resolve([ctx,imgHeight,imgWidth])![6](img/6.png)}})}else{alert('No webcam - sorry!')}}
```

![1](img/#co_advanced_models_and_ui_CO6-1)

这些是网络摄像头用户媒体配置约束。这里可以应用[几个选项](https://oreil.ly/MkWml)，但为简单起见，保持得很简单。

![2](img/#co_advanced_models_and_ui_CO6-2)

这个条件检查是为了支持不支持新的`srcObject`配置的旧浏览器。根据你的支持需求，这可能会被弃用。

![3](img/#co_advanced_models_and_ui_CO6-3)

在视频加载完成之前无法访问视频，因此该事件被包装在一个 promise 中，以便等待。

![4](img/#co_advanced_models_and_ui_CO6-4)

这是你需要等待的事件，然后才能将视频元素传递给`tf.fromPixels`。

![5](img/#co_advanced_models_and_ui_CO6-5)

在设置画布时，注意使用`clientWidth`而不是`width`。

![6](img/#co_advanced_models_and_ui_CO6-6)

promise 解析后，你将需要将信息传递给检测和绘制循环。

## 绘制检测结果

最后，您执行检测和绘图的方式与对图像执行的方式相同。在每次调用的开始时，您需要删除上一次调用的所有检测；否则，您的画布将慢慢填满旧的检测。清除画布很简单；您可以使用`clearRect`来删除指定坐标的任何内容。传递整个画布的宽度和高度将擦除所有内容。

```py
ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
```

在每次绘制检测结束时，*不要*在清理中处理模型，因为您需要在每次检测中使用它。然而，其他所有内容都可以和应该被处理。

在示例 6-1 中确定的`performDetections`函数应该在无限循环中递归调用自身。该函数的循环速度可能比画布绘制速度更快。为了确保不浪费循环，使用浏览器的`requestAnimationFrame`来限制这一点：

```py
// Loop forever
requestAnimationFrame(() => {
  performDetections(model, videoRef, camDetails)
})
```

就是这样。通过一些逻辑调整，您已经从静态图像转移到了实时速度的视频输入。在我的电脑上，我看到大约每秒 16 帧。在人工智能领域，这已经足够快，可以处理大多数用例。我用它来证明我至少是 97%的人，如图 6-16 所示。

![在浏览器中运行的 SSD MobileNet 的屏幕截图](img/ltjs_0616.png)

###### 图 6-16。具有 SSD MobileNet 的完全功能网络摄像头

# 章节回顾

祝贺您挑战了 TensorFlow Hub 上存在的最有用但也最复杂的模型之一。虽然用 JavaScript 隐藏这个模型的复杂性很简单，但您现在熟悉了一些最令人印象深刻的物体检测和澄清概念。机器学习背负着快速解决问题的概念，然后解决后续代码以将 AI 的壮丽属性附加到给定领域。您可以期待任何显著先进模型和领域都需要大量研究。

## 章节挑战：顶级侦探

NMS 简化了排序和消除检测。假设您想解决识别顶级预测然后将它们从高到低排序的问题，以便您可以创建类似图 6-6 的图形。与其依赖 NMS 来找到您最可行和最高值，您需要自己解决最高值问题。将这个小但类似的分组视为整个检测数据集。想象这个`[1, 6, 5]`的张量检测集合是您的`result[0]`，您只想要具有最高置信度值的前三个检测。您如何解决这个问题？

```py
  const t = tf.tensor([[
    [1, 2, 3, 4, 5],
    [1.1, 2.1, 3.1, 4.1, 5.1],
    [1.2, 2.2, 3.2, 4.2, 5.2],
    [1.2, 12.2, 3.2, 4.2, 5.2],
    [1.3, 2.3, 3.3, 4.3, 5.3],
    [1, 1, 1, 1, 1]
  ]])

  // Get the top-three most confident predictions.
```

您的最终解决方案应该打印`[3, 4, 2]`，因为索引为 3 的张量具有最大值（12.2），其次是索引为 4（包含 5.3），然后是索引为 2（5.2）。

您可以在附录 B 中找到此挑战的答案。

## 复习问题

让我们回顾您在本章编写的代码中学到的知识。花点时间回答以下问题：

1.  在物体检测机器学习领域，SSD 代表什么？

1.  您需要使用哪种方法来预测使用动态控制流操作的模型？

1.  SSD MobileNet 预测多少类别和多少值？

1.  去重相同对象的检测的方法是什么？

1.  使用大型同步 TensorFlow.js 调用的缺点是什么？

1.  您应该使用什么方法来识别标签的宽度？

1.  `globalCompositeOperation`会覆盖画布上现有的内容吗？

这些问题的解决方案可以在附录 A 中找到。
