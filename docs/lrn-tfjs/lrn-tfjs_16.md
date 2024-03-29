# 附录 B. 章节挑战答案

# 第二章：卡车警报！

MobileNet 模型可以检测各种不同类型的卡车。您可以通过查看可识别卡车的列表来解决这个问题，或者您可以简单地在给定的类名列表中搜索*truck*这个词。为简单起见，提供的答案选择了后者。

包含 HTML 和 JavaScript 的整个解决方案在这里：

```js
<!DOCTYPE html>
<html>
  <head>
    <script
    src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.7.0/dist/tf.min.js">
  </script>
    <script
    src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@1.0.0">
  </script> <!-- ① -->
    <script>
      mobilenet.load().then(model => {
        const img = document.getElementById('myImage'); <!-- ② -->
        // Classify the image
        model.classify(img).then(predictions => {
          console.log('Predictions: ', predictions);
          // Was there a truck?
          let foundATruck
          predictions.forEach(p => {
            foundATruck = foundATruck || p.className.includes("truck") <!-- ③ -->
          })
          // TRUCK ALERT!
          if (foundATruck) alert("TRUCK DETECTED!") <!-- ④ -->
        });
      });
    </script>
  </head>
  <body>
    <h1>Is this a truck?</h1>
    <img id="myImage" src="truck.jpg" width="100%"></img>
  </body>
</html>
```

①

从 CDN 加载 MobileNet 模型。

②

通过 ID 访问 DOM 上的图像。由于等待模型加载，DOM 可能已经加载了一段时间。

③

如果在任何预测中检测到*truck*这个词，将`foundATruck`设置为 true。

④

真相时刻！只有在`foundATruck`为 true 时才会弹出警报。

这个带有卡车图像的章节挑战答案可以在本书的[GitHub](https://github.com/GantMan/learn-tfjs)源代码中找到。

# 第三章：你有什么特别之处？

这个简单的练习是关于查找 TensorFlow.js `tf.unique`方法。一旦找到这个友好的方法，就很容易构建一个解决方案，就像这样：

```js
const callMeMaybe = tf.tensor([8367677, 4209111, 4209111, 8675309, 8367677])
const uniqueTensor = tf.unique(callMeMaybe).values
const result = uniqueTensor.arraySync()
console.log(`There are ${result.length} unique values`, result)
```

不要忘记将此代码包装在`tf.tidy`中以进行自动张量清理！

# 第四章：混乱排序

一种优雅的解决方案是对`randomUniform`创建的张量使用`topk`进行排序。由于`randomUniform`创建的值在`0`和`1`之间，并且`topk`沿着最后一个轴对值进行排序，您可以使用以下代码完成这个练习：

```js
const rando = tf.randomUniform([400, 400]) // ①
const sorted = tf.topk(rando, 400).values // ②
const answer = sorted.reshape([400, 400, 1]) // ③
```

①

创建一个 2D 的 400 x 400 张量，其中包含介于`0`和`1`之间的随机值。

②

使用`topk`对最后一个维度（宽度）进行排序，并返回所有 400 个值。

③

可选：将张量重塑为 3D 值。

先前的解决方案非常冗长，可以压缩为一行代码：

```js
tf.topk(tf.randomUniform([400, 400]), 400).values
```

# 第五章：可爱的脸

现在，第一个模型已经给出了脸部的坐标，一个张量裁剪将提供这些像素。这几乎与`strokeRect`完全相同，因为您提供了一个起始位置和所需的大小。然而，我们之前的所有测量对于这个裁剪都不起作用，因为它们是在图像的调整版本上计算的。您需要在原始张量数据上进行类似的计算，以便提取正确的信息。

###### 提示

如果您不想重新计算位置，可以将张量调整大小以匹配`petImage`的宽度和高度。这将允许您重用相同的`startX`、`startY`、`width`和`height`变量进行裁剪。

以下代码可能引用原始人脸定位代码中创建的一些变量，特别是原始的`fromPixels`张量`myTensor`：

```js
// Same bounding calculations but for the tensor
const tHeight = myTensor.shape[0] // ①
const tWidth = myTensor.shape[1]
const tStartX = box[0] * tWidth
const tStartY = box[1] * tHeight
const cropLength = parseInt((box[2] - box[0]) * tWidth, 0)  // ②
const cropHeight = parseInt((box[3] - box[1]) * tHeight, 0)

const startPos = [tStartY, tStartX, 0]
const cropSize = [cropHeight, cropLength, 3]

const cropped = tf.slice(myTensor, startPos, cropSize)

// Prepare for next model input
const readyFace = tf.image
  .resizeBilinear(cropped, [96, 96], true)
  .reshape([1, 96, 96, 3]);  // ③
```

①

请注意，张量的顺序是高度然后宽度。它们的格式类似于数学矩阵，而不是图像特定的宽度乘以高度的标准。

②

减去比率可能会留下浮点值；您需要将这些值四舍五入到特定的像素索引。在这种情况下，答案是使用`parseInt`来去除任何小数。

③

显然，批处理，然后取消批处理，然后重新批处理是低效的。在可能的情况下，您应该将所有操作保持批处理，直到绝对必要。

现在，您已经成功地准备好将狗脸张量传递到下一个模型中，该模型将返回狗在喘气的可能性百分比。

结果模型的输出从未指定，但您可以确保它将是一个两值的一维张量，索引 0 表示不 panting，索引 1 表示 panting，或者是一个一值的一维张量，表示从零到一的 panting 可能性。这两种情况都很容易处理！

# 第六章：顶级侦探

使用`topk`的问题在于它仅在特定张量的最终维度上起作用。因此，您可以通过两次调用`topk`来找到两个维度上的最大值。第二次您可以将结果限制为前三名。

```js
const { indices, values } = tf.topk(t)
const topvals = values.squeeze()
const sorted = tf.topk(topvals, 3)
// prints [3, 4, 2]
sorted.indices.print()
```

然后，您可以循环遍历结果并从`topvals`变量中访问前几个值。

# 第七章：再见，MNIST

通过向导您可以选择所有所需的设置；您应该已经创建了一些有趣的结果。结果应该如下：

+   100 个二进制文件被生成在一个分组中。

+   最终大小约为 1.5 MB。

+   由于大小为 1.5 MB，如果使用默认值，这可以适合单个 4 MB 分片。

# 第八章：模型架构师

您被要求创建一个符合给定规格的 Layers 模型。该模型的输入形状为五，输出形状为四，中间有几个具有指定激活函数的层。

构建模型的代码应该如下所示：

```js
const model = tf.sequential();

model.add(
  tf.layers.dense({
    inputShape: 5,
    units: 10,
    activation: "sigmoid"
  })
);

model.add(
  tf.layers.dense({
    units: 7,
    activation: "relu"
  })
);

model.add(
  tf.layers.dense({
    units: 4,
    activation: "softmax"
  })
);

model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy"
});
```

可训练参数的数量计算为进入一个层的行数 + 该层中的单元数。您可以使用每个层的计算`layerUnits[i] * layerUnits[i - 1] + layerUnits[i]`来解决这个问题。`model.summary()`的输出将验证您的数学计算。将您的摘要与示例 B-1 进行比较。

##### 示例 B-1\. 模型摘要

```js
_________________________________________________________________
Layer (type)                 Output shape              Param #
=================================================================
dense_Dense33 (Dense)        [null,10]                 60
_________________________________________________________________
dense_Dense34 (Dense)        [null,7]                  77
_________________________________________________________________
dense_Dense35 (Dense)        [null,4]                  32
=================================================================
Total params: 169
Trainable params: 169
Non-trainable params: 0
_________________________________________________________________
```

# 第九章：船出事了

当然，有很多获取这些信息的方法。这只是其中一种方式。

要提取每个名称的敬语，您可以使用`.apply`并通过空格分割。这将让您很快得到大部分答案。但是，一些名称中有“von”之类的内容，这会导致额外的空格并稍微破坏您的代码。为此，一个好的技巧是使用正则表达式。我使用了`/,\s(.*?)\./`，它查找逗号后跟一个空格，然后匹配直到第一个句点。

您可以应用这个方法创建一个新行，按该行分组，然后使用`.mean()`对幸存者的平均值进行表格化。

```js
mega_df['Name'] = mega_df['Name'].apply((x) => x.split(/,\s(.*?)\./)[1])
grp = mega_df.groupby(['Name'])
table(grp.col(['Survived']).mean())
```

`mega_df['Name']`被替换为有用的内容，然后进行分组以进行验证。然后可以轻松地对其进行编码或进行分箱处理以用于您的模型。

图 B-1 显示了在 Dnotebook 中显示的分组代码的结果。

![Dnotebook 解决方案的屏幕截图](img/ltjs_ab01.png)

###### 图 B-1\. 敬语和生存平均值

# 第十章：保存魔法

为了保存最高的验证准确性，而不是最后的验证准确性，您可以在时期结束回调中添加一个条件保存。这可以避免您意外地陷入过拟合时期的困扰。

```js
// initialize best at zero
let best = 0

//...

// In the callback object add the onEpochEnd save condition
onEpochEnd: async (_epoch, logs) => {
  if (logs.val_acc > best) {
    console.log("SAVING")
    model.save(savePath)
    best = logs.val_acc
  }
}
```

还有[`earlyStopping`](https://oreil.ly/BZw2o)预打包回调，用于监视和防止过拟合。将您的回调设置为`callbacks: tf.callbacks.earlyStopping({monitor: 'val_acc'})`将在验证准确性回退时停止训练。

# 第十一章：光速学习

您现在知道很多解决这个问题的方法，但我们将采取快速简单的方式。解决这个问题有四个步骤：

1.  加载新的图像数据

1.  将基础模型削减为特征模型

1.  创建读取特征的新层

1.  训练新层

加载新的图像数据：

```js
const dfy = await dfd.read_csv('labels.csv')
const dfx = await dfd.read_csv('images.csv')

const Y = dfy.tensor
const X = dfx.tensor.reshape([dfx.shape[0], 28, 28, 1])
```

将基础模型削减为特征模型：

```js
const model = await tf.loadLayersModel('sorting_hat/model.json')
const layer = model.getLayer('max_pooling2d_MaxPooling2D3')
const shaved = tf.model({
  inputs: model.inputs,
  outputs: layer.output
})
// Run data through shaved model to get features
const XFEATURES = shaved.predict(X)
```

创建读取特征的新层：

```js
transferModel = tf.sequential({
  layers: [
    tf.layers.flatten({ inputShape: shaved.outputs[0].shape.slice(1) }),
    tf.layers.dense({ units: 128, activation: 'relu' }),
    tf.layers.dense({ units: 3, activation: 'softmax' }),
  ],
})
transferModel.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
})
```

训练新层：

```js
await transferModel.fit(XFEATURES, Y, {
  epochs: 10,
  validationSplit: 0.1,
  callbacks: {
    onEpochEnd: console.log,
  },
})
```

结果在 10 个时期内训练到了很高的准确性，如图 B-2 所示。

![在几个时期内达到完美的验证准确性](img/ltjs_ab02.png)

###### 图 B-2\. 仅从 150 张图像训练

这个挑战的完整答案可以在[本章的相关源代码](https://oreil.ly/lKaUm)中找到，这样你就可以查看代码，甚至与结果进行交互。

# 第十二章：简单如 01, 10, 11

将图像转换为灰度很容易。一旦你这样做了，你可以在图像上使用 `tf.where` 来用白色或黑色像素替换每个像素。

以下代码将具有 `input` ID 的图像转换为一个二值化图像，该图像显示在同一页上名为 `output` 的画布上：

```js
// Simply read from the DOM
const inputImage = document.getElementById('input')
const inTensor = tf.browser.fromPixels(inputImage, 1)

// Binarize
const threshold = 50
const light = tf.onesLike(inTensor).asType('float32')
const dark = tf.zerosLike(inTensor)
const simpleBinarized = tf.where(
  tf.less(inTensor, threshold),
  dark, // False Case: place zero
  light, // True Case: place one
)

// Show results
const myCanvas = document.getElementById('output')
tf.browser.toPixels(simpleBinarized, myCanvas)
```

本章挑战答案的完全运行示例可以在[本章的相关源代码](https://oreil.ly/gMVzA)中找到。

有更高级和更健壮的方法来对图像进行二值化。如果你想处理更多的图像，请查看二值化算法。
