# 第12章。骰子化：顶点项目

> “每个人都有一个计划，直到他们被打在嘴巴上。”
> 
> —铁拳迈克·泰森

你的所有训练使你通过各种理论和练习。现在，你已经知道足够多，可以提出一个计划，在TensorFlow.js中为机器学习构建新的创意用途。在这一章中，你将开发你的顶点项目。与其用TensorFlow.js学习另一个机器学习方面，不如在这一章开始时接受一个挑战，并利用你现有的技能构建一个可行的解决方案。从构思到完成，这一章将指导你解决问题的执行。无论这是你第一本机器学习书籍还是第十本，这个顶点项目是你展现才华的时刻。

我们将：

+   研究问题

+   创建和增强数据

+   训练一个能解决问题的模型

+   在网站中实施解决方案

当你完成这一章时，你将运用从头到尾的技能来解决一个有趣的机器学习项目。

# 一个具有挑战性的任务

我们将利用你新发现的技能来模糊艺术和科学之间的界限。工程师们多年来一直在利用机器进行令人印象深刻的视觉壮举。最值得注意的是，暗箱相机技术（如[图12-1](#camera_obscura)所示）让疯狂的科学家们可以用镜头和镜子追踪实景。^([1](ch12.html#idm45049236364264))

![人看着黑匣子相机暗箱](assets/ltjs_1201.png)

###### 图12-1。相机暗箱

如今，人们正在用最奇怪的东西制作艺术品。在我的大学，艺术系用便利贴像素创造了一个完整的《超级马里奥兄弟》场景。虽然我们中有些人有艺术的神启，但其他人可以通过发挥他们的其他才能制作类似的作品。

你的挑战，如果你选择接受并从这本书中学到尽可能多的东西，就是教会人工智能如何使用骰子绘画。通过排列六面骰子并选择正确的数字显示，你可以复制任何图像。艺术家们会购买数百个骰子，并利用他们的技能重新创作图像。在这一章中，你将运用你学到的所有技能，教会人工智能如何将图像分解成骰子艺术，如[图12-2](#dicify_tfjs)所示。

![图像转换为骰子版本](assets/ltjs_1202.png)

###### 图12-2。将图形转换为骰子

一旦你的人工智能能够将黑白图像转换为骰子，你可以做很多事情，比如创建一个酷炫的网络摄像头滤镜，制作一个出色的网站，甚至为自己打印一个有趣的手工艺项目的说明。

在继续之前花10分钟，策划如何利用你的技能从零开始构建一个体面的图像到骰子转换器。

# 计划

理想情况下，你想到了与我类似的东西。首先，你需要数据，然后你需要训练一个模型，最后，你需要创建一个利用训练模型的网站。

## 数据

虽然骰子并不是非常复杂，但每个像素块应该是什么并不是一个现有的数据集。你需要生成一个足够好的数据集，将图像的一个像素块映射到最适合的骰子。你将创建像[图12-3](#pixel_to_die)中那样的数据。

![垂直线转换为骰子中的数字三](assets/ltjs_1203.png)

###### 图12-3。教AI如何选择哪个骰子适用

一些骰子可以旋转。数字二、三和六将需要在数据集中重复出现，因此它们对每种配置都是特定的。虽然它们在游戏中是可互换的，但在艺术中不是。[图12-4](#three_ne_three)展示了这些数字如何在视觉上镜像。

![三个骰子和三个旋转](assets/ltjs_1204.png)

###### 图12-4。角度很重要；这两个不相等

这意味着你需要总共九种可能的配置。那就是六个骰子，其中三个旋转了90度。[图12-5](#nine_config)展示了你平均六面游戏骰子的所有可能配置。

![用实际骰子说明的六面骰子的九种可能配置](assets/ltjs_1205.png)

###### 图12-5。九种可能的配置

这些是用一种必须平放的骰子风格重新创建任何图像的可用模式。虽然这对于直接表示图像来说并不完美，但随着数量和距离的增加，分辨率会提高。

## 训练

在设计模型时，会有两个重要问题：

+   是否有什么东西对迁移学习有用？

+   模型应该有卷积层吗？

首先，我从未见过类似的东西。在创建模型时，我们需要确保有一个验证和测试集来验证模型是否训练良好，因为我们将从头开始设计它。

其次，模型应该避免使用卷积。卷积可以帮助您提取复杂的特征，而不考虑它们的位置，但这个模型非常依赖位置。两个像素块可以是一个2或一个旋转的2。对于这个练习，我将不使用卷积层。

直到完成后我们才知道跳过卷积是否是一个好计划。与大多数编程不同，机器学习架构中有一层实验。我们可以随时回去尝试其他架构。

## 网站

一旦模型能够将一小块像素分类为相应的骰子，您将需要激活您的张量技能，将图像分解成小块以进行转换。图像的片段将被堆叠，预测并与骰子的图片重建。

###### 注意

由于本章涵盖的概念是先前解释的概念的应用，本章将讨论高层次的问题，并可能跳过解决这个毕业项目的代码细节。如果您无法跟上，请查看先前章节以获取概念和[相关源代码](https://oreil.ly/PjNLO)的具体信息。*本章不会展示每一行代码*。

# 生成训练数据

本节的目标是创建大量数据以用于训练模型。这更多是一门艺术而不是科学。我们希望有大量的数据。为了生成数百张图像，我们可以轻微修改现有的骰子像素。对于本节，我创建了12 x 12的骰子印刷品，使用简单的二阶张量。可以通过一点耐心创建九种骰子的配置。查看[示例12-1](#dice_array_example)，注意代表骰子黑点的零块。

##### 示例12-1。骰子一和二的数组表示

```py
[
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
],
[
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
],
```

您可以使用`tf.ones`创建一个`[9, 12, 12]`的全为`1`的浮点数，然后手动将位置转换为`0`，以制作每个骰子的黑点。

一旦您拥有所有九种配置，您可以考虑图像增强以创建新数据。标准图像增强库在这里无法使用，但您可以利用您的张量技能编写一个函数，稍微移动每个骰子位置一个像素。这种小变异将一个骰子变成九种变体。然后您的数据集中将有九种骰子的九种变体。

在代码中实现这一点，想象一下增加骰子的大小，然后在周围滑动一个12 x 12的窗口，稍微偏离中心剪切图像的新版本：这是一种*填充和裁剪增强*。

```py
constpixelShift=async(inputTensor,mutations=[])=>{// Add 1px white padding to height and width
constpadded=inputTensor.pad(![1](assets/1.png)[[1,1],[1,1],],1)constcutSize=inputTensor.shapefor(leth=0;h<3;h++){for(letw=0;w<3;w++){![2](assets/2.png)mutations.push(padded.slice([h,w],cutSize))![3](assets/3.png)}}padded.dispose()returnmutations}
```

[![1](assets/1.png)](#co_dicify__capstone_project_CO1-1)

`.pad`为现有张量添加一个值为`1`的白色边框。

[![2](assets/2.png)](#co_dicify__capstone_project_CO1-2)

为了生成九个新的移位值，每次都会移动切片位置的起点。

[![3](assets/3.png)](#co_dicify__capstone_project_CO1-3)

切片的子张量每次都会成为一个新的12 x 12值，起点不同。

`pixelShift`的结果创建了一些小变化，这些变化应该仍然可以用原始骰子解决。[图12-6](#mods)显示了移动像素的视觉表示。

![从一个骰子生成九个新图像](assets/ltjs_1206.png)

###### 图12-6。移动像素创建新的骰子

虽然每个骰子有九个版本比一个好，但数据集仍然非常小。您必须想出一种方法来创建新数据。

您可以通过随机组合这九个移位图像来创建新的变体。有很多方法可以组合这些图像中的任意两个。一种方法是使用`tf.where`，并将两个图像中较小的保留在它们的新组合图像中。这样可以保留任意两个移位骰子的黑色像素。

```py
// Creates combinations take any two from array // (like Python itertools.combinations) constcombos=async(tensorArray)=>{conststartSize=tensorArray.lengthfor(leti=0;i<startSize-1;i++){for(letj=i+1;j<startSize;j++){constoverlay=tf.tidy(()=>{returntf.where(![1](assets/1.png)tf.less(tensorArray[i],tensorArray[j]),![2](assets/2.png)tensorArray[i],![3](assets/3.png)tensorArray[j]![4](assets/4.png))})tensorArray.push(overlay)}}}
```

[![1](assets/1.png)](#co_dicify__capstone_project_CO2-1)

`tf.where`就像在每个元素上运行条件。

[![2](assets/2.png)](#co_dicify__capstone_project_CO2-2)

当第一个参数小于第二个参数时，`tf.less`返回true。

[![3](assets/3.png)](#co_dicify__capstone_project_CO2-3)

如果`where`中的条件为true，则返回`arrCopy[i]`中的值。

[![4](assets/4.png)](#co_dicify__capstone_project_CO2-4)

如果`where`中的条件为false，则返回`arrCopy[j]`中的值。

当您重叠这些骰子时，您会得到看起来像之前骰子的小变异的新张量。骰子上的4 x 4个点被组合在一起，可以创建相当多的新骰子，可以添加到您的数据集中。

甚至可以对变异进行两次。变异的变异仍然可以被人眼区分。当您查看[图12-7](#combos)中生成的四个骰子时，仍然可以明显看出这些骰子是从显示值为一的一面生成的。即使它们是由虚构的第二代变异组合而成，新骰子仍然在视觉上与所有其他骰子组合明显不同。

![通过组合以前的骰子进行变异以制作新骰子](assets/ltjs_1207.png)

###### 图12-7。通过骰子组合的四种变异

正如您可能已经猜到的那样，在创建这些类似俄罗斯方块的形状时，会有一些意外的重复。与其试图避免重复配置，不如通过调用`tf.unique`来删除重复项。

###### 警告

目前GPU不支持`tf.unique`，因此您可能需要将后端设置为CPU来调用`unique`。之后，如果您愿意，可以将后端返回到GPU。

在高层次上，对生成的骰子图像进行移位和变异，从单个骰子生成了两百多个骰子。以下是高层次的总结：

1.  将图像在每个方向上移动一个像素。

1.  将移位后的张量组合成所有可能的组合。

1.  对先前集合执行相同的变异组合。

1.  仅使用唯一结果合并数据。

现在，对于每种九种可能的组合，我们有两百多个张量。考虑到刚才只有九个张量，这还不错。两百张图片足够吗？我们需要测试才能找出答案。

您可以立即开始训练，或者将数据保存到文件中。[本章相关的代码](https://oreil.ly/Vr98u)会写入一个文件。本节的主要功能可以用以下代码概括：

```py
const createDataObject = async () => {
  const inDice = require('./dice.json').data
  const diceData = {}
  // Create new data from each die
  for (let idx = 0; idx < inDice.length; idx++) {
    const die = inDice[idx]
    const imgTensor = tf.tensor(die)
    // Convert this single die into 200+ variations
    const results = await runAugmentation(imgTensor, idx)
    console.log('Unique Results:', idx, results.shape)
    // Store results
    diceData[idx] = results.arraySync()
    // clean
    tf.dispose([results, imgTensor])
  }

  const jsonString = JSON.stringify(diceData)
  fs.writeFile('dice_data.json', jsonString, (err) => {
    if (err) throw err
    console.log('Data written to file')
  })
}
```

# 训练

现在您总共有将近两千张图片，可以尝试训练您的模型。数据应该被堆叠和洗牌：

```py
constdiceImages=[].concat(![1](assets/1.png)diceData['0'],diceData['1'],diceData['2'],diceData['3'],diceData['4'],diceData['5'],diceData['6'],diceData['7'],diceData['8'],)// Now the answers to their corresponding index constanswers=[].concat(newArray(diceData['0'].length).fill(0),![2](assets/2.png)newArray(diceData['1'].length).fill(1),newArray(diceData['2'].length).fill(2),newArray(diceData['3'].length).fill(3),newArray(diceData['4'].length).fill(4),newArray(diceData['5'].length).fill(5),newArray(diceData['6'].length).fill(6),newArray(diceData['7'].length).fill(7),newArray(diceData['8'].length).fill(8),)// Randomize these two sets together tf.util.shuffleCombo(diceImages,answers)![3](assets/3.png)
```

[![1](assets/1.png)](#co_dicify__capstone_project_CO3-1)

通过连接单个数据数组来创建大量数据数组。

[![2](assets/2.png)](#co_dicify__capstone_project_CO3-2)

然后，您创建与每个数据集大小完全相同的答案数组，并使用`Array`的`.fill`来填充它们。

[![3](assets/3.png)](#co_dicify__capstone_project_CO3-3)

然后，您可以将这两个数组一起随机化。

从这里，您可以拆分出一个测试集，也可以不拆分。如果您需要帮助，可以查看相关代码。一旦您按照自己的意愿拆分了数据，然后将这两个JavaScript数组转换为正确的张量：

```py
consttrainX=tf.tensor(diceImages).expandDims(3)![1](assets/1.png)consttrainY=tf.oneHot(answers,numOptions)![2](assets/2.png)
```

[![1](assets/1.png)](#co_dicify__capstone_project_CO4-1)

创建堆叠张量，并为简单起见，通过在索引三处扩展维度将其返回为三维图像。

[![2](assets/2.png)](#co_dicify__capstone_project_CO4-2)

然后，将数字答案进行独热编码为张量，以适应softmax模型输出。

该模型采用了简单而小型的设计。您可能会找到更好的结构，但对于这个，我选择了两个隐藏层。随时回来并尝试使用架构进行实验，看看您可以获得什么样的速度和准确性。

```py
const model = tf.sequential()
model.add(tf.layers.flatten({ inputShape }))
model.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
}))
model.add(tf.layers.dense({
    units: 8,
    activation: 'relu',
}))
model.add(tf.layers.dense({
    units: 9,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax',
}))
```

该模型首先通过将图像输入展平以将它们连接到神经网络，然后有一个`64`和一个`8`单元层。最后一层是九种可能的骰子配置。

这个模型在几个时代内就能达到近乎完美的准确率。这对于生成的数据来说是很有希望的，但在下一节中，我们将看到它在实际图像中的表现如何。

# 网站界面

现在您已经有了一个经过训练的模型，是时候用非生成数据进行测试了。肯定会有一些错误，但如果模型表现得不错，这将是相当成功的！

您的网站需要告诉需要使用多少个骰子，然后将输入图像分成相同数量的块。这些块将被调整大小为12 x 12的输入（就像我们的训练数据），然后在图像上运行模型进行预测。在[图12-8](#convert_img)中显示的示例中，一个X的图像被告知要转换为四个骰子。因此，图像被切割成四个象限，然后对每个象限进行预测。它们应该理想地将骰子对齐以绘制X。

![将TensorFlow标志切割成32 x 32块之前和之后](assets/ltjs_1208.png)

###### 图12-8。将TensorFlow标志切割成32 x 32块

一旦您获得了预测结果，您可以重建一个由指定图像张量组成的新张量。

###### 注意

这些图像是在0和1上进行训练的。这意味着，为了期望得到一个体面的结果，您的输入图像也应该由0和1组成。颜色甚至灰度都会产生虚假的结果。

应用程序代码的核心应该看起来像这样：

```py
const dicify = async () => {
  const modelPath = '/dice-model/model.json'
  const dModel = await tf.loadLayersModel(modelPath)

  const grid = await cutData("input")
  const predictions = await predictResults(dModel, grid)
  await displayPredictions(predictions)

  tf.dispose([dModel, predictions])
  tf.dispose(grid)
}
```

结果的预测是您经典的“数据输入，数据输出”模型行为。最复杂的部分将是`cutData`和`displayPredictions`方法。在这里，您的张量技能将大放异彩。

## 切成块

`cutData`方法将利用`tf.split`，它沿着一个轴将张量分割为N个子张量。您可以通过使用`tf.split`沿着每个轴将图像分割成一个补丁或图像网格来进行预测。

```py
constnumDice=32constpreSize=numDice*10constcutData=async(id)=>{constimg=document.getElementById(id)constimgTensor=tf.browser.fromPixels(img,1)![1](assets/1.png)constresized=tf.image.resizeNearestNeighbor(![2](assets/2.png)imgTensor,[preSize,preSize])constcutSize=numDiceconstheightCuts=tf.split(resized,cutSize)![3](assets/3.png)constgrid=heightCuts.map((sliver)=>![4](assets/4.png)tf.split(sliver,cutSize,1))returngrid}
```

[![1](assets/1.png)](#co_dicify__capstone_project_CO5-1)

您只需要将图像的灰度版本从像素转换过来。

[![2](assets/2.png)](#co_dicify__capstone_project_CO5-2)

图像被调整大小，以便可以被所需数量的骰子均匀分割。

[![3](assets/3.png)](#co_dicify__capstone_project_CO5-3)

图像沿着第一个轴（高度）被切割。

[![4](assets/4.png)](#co_dicify__capstone_project_CO5-4)

然后将这些列沿着宽度轴切割，以创建一组张量。

`grid`变量现在包含一个图像数组。在需要时，您可以调整图像大小并堆叠它们进行预测。例如，[图12-9](#tf_grid)是一个切片网格，因为TensorFlow标志的黑白切割将创建许多较小的图像，这些图像将被转换为骰子。

![将TensorFlow标志切割成27x27块](assets/ltjs_1209.png)

###### 图12-9。黑白TensorFlow标志的切片

## 重建图像

一旦您有了预测结果，您将想要重建图像，但您将希望将原始块替换为它们预测的骰子。

从预测答案重建和创建大张量的代码可能如下所示：

```py
constdisplayPredictions=async(answers)=>{tf.tidy(()=>{constdiceTensors=diceData.map(![1](assets/1.png)(dt)=>tf.tensor(dt))const{indices}=tf.topk(answers)constanswerIndices=indices.dataSync()consttColumns=[]for(lety=0;y<numDice;y++){consttRow=[]for(letx=0;x<numDice;x++){constcurIndex=y*numDice+x![2](assets/2.png)tRow.push(diceTensors[answerIndices[curIndex]])}constoneRow=tf.concat(tRow,1)![3](assets/3.png)tColumns.push(oneRow)}constdiceConstruct=tf.concat(tColumns)![4](assets/4.png)// Print the reconstruction to the canvas
constcan=document.getElementById('display')tf.browser.toPixels(diceConstruct,can)![5](assets/5.png)})}
```

[![1](assets/1.png)](#co_dicify__capstone_project_CO6-1)

要绘制的`diceTensors`从`diceData`中加载并转换。

[![2](assets/2.png)](#co_dicify__capstone_project_CO6-2)

要从1D返回到2D，需要为每一行计算索引。

[![3](assets/3.png)](#co_dicify__capstone_project_CO6-3)

行是通过沿着宽度轴进行连接而创建的。

[![4](assets/4.png)](#co_dicify__capstone_project_CO6-4)

列是通过沿着默认（高度）轴连接行来制作的。

[![5](assets/5.png)](#co_dicify__capstone_project_CO6-5)

哒哒！新构建的张量可以显示出来了。

如果你加载了一个黑白图像并处理它，现在是真相的时刻。每个类别生成了大约200张图像是否足够？

我将`numDice`变量设置为27。一个27 x 27的骰子图像是相当低分辨率的，需要在亚马逊上花费大约80美元。让我们看看加上TensorFlow标志会是什么样子。[图12-10](#tf)展示了结果。

![TensorFlow标志转换为27 x 27骰子之前和之后](assets/ltjs_1210.png)

###### 图12-10。TensorFlow标志转换为27 x 27骰子

它有效！一点也不错。你刚刚教会了一个AI如何成为一个艺术家。如果你增加骰子的数量，图像会变得更加明显。

# 章节回顾

使用本章的策略，我训练了一个AI来处理红白骰子。我没有太多耐心，所以我只为一个朋友制作了一个19x19的图像。结果相当令人印象深刻。我花了大约30分钟将所有的骰子放入[图12-11](#ir_dice)中显示的影子盒中。如果没有印刷说明，我想我不会冒这个风险。

![19 x 19成品图像。](assets/ltjs_1211.png)

###### 图12-11。完成的19 x 19红白骰子带背光

你可以走得更远。哪个疯狂的科学家没有自己的肖像？现在你的肖像可以由骰子制成。也许你可以教一个小机器人如何为你摆放骰子，这样你就可以建造满是数百磅骰子的巨大画框（见[图12-12](#dice_wall)）。

![一个人看着一堵用骰子做成的图像的墙](assets/ltjs_1212.png)

###### 图12-12。完美的疯狂科学肖像

你可以继续改进数据并获得更好的结果，你不仅仅局限于普通的黑白骰子。你可以利用你的AI技能用装饰性骰子、便利贴、魔方、乐高积木、硬币、木片、饼干、贴纸或其他任何东西来绘画。

虽然这个实验对于1.0版本来说是成功的，但我们已经确定了无数个实验，可以让你改进你的模型。

## 章节挑战：简单如01、10、11

现在你有了一个强大的新模型，可以成为由黑色`0`和白色`1`像素组成的任何照片的艺术家。不幸的是，大多数图像，即使是灰度图像，也有中间值。如果有一种方法可以高效地将图像转换为黑白就好了。

将图像转换为二进制黑白被称为*二值化*。计算机视觉领域有各种各样的令人印象深刻的算法，可以最好地将图像二值化。让我们专注于最简单的方法。

在这个章节挑战中，使用`tf.where`方法来检查像素是否超过给定的阈值。使用该阈值，你可以将灰度图像的每个像素转换为`1`或`0`。这将为你的骰子模型准备正常的图形输入。

通过几行代码，你可以将成千上万种光的变化转换为黑白像素，如[图12-13](#binarize)所示。

![一个头骨被转换成黑白像素。](assets/ltjs_1213.png)

###### 图12-13。二值化的头骨

你可以在[附录B](app02.html#appendix_b)中找到这个挑战的答案。

## 复习问题

让我们回顾一下你在本章编写的代码中学到的知识。花点时间回答以下问题：

1.  TensorFlow.js的哪个方法允许你将张量分解为一组相等的子张量？

1.  用于创建数据的稍微修改的替代品以扩大数据集的过程的名称是什么？

1.  为什么Gant Laborde如此了不起？

这些练习的解决方案可以在[附录A](app01.html#book_appendix)中找到。

^([1](ch12.html#idm45049236364264-marker))如果你想了解更多关于暗箱的知识，请观看纪录片[*Tim's Vermeer*](https://oreil.ly/IrjNM)。
