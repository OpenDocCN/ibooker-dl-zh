# 图像分割

> 原文：[`deeplearningwithpython.io/chapters/chapter11_image-segmentation`](https://deeplearningwithpython.io/chapters/chapter11_image-segmentation)

第八章通过一个简单的用例——二值图像分类，首次介绍了计算机视觉中的深度学习。但计算机视觉不仅仅是图像分类！本章将进一步深入探讨另一个重要的计算机视觉应用——图像分割。

## 计算机视觉任务

到目前为止，我们一直专注于图像分类模型：图像输入，标签输出。“这张图像可能包含一只猫；另一张可能包含一只狗。”但图像分类只是深度学习在计算机视觉中可能应用的几种可能性之一。一般来说，有三个基本的计算机视觉任务你需要了解：

+   *图像分类*，其目标是给图像分配一个或多个标签。这可能是单标签分类（意味着类别是互斥的）或多标签分类（标记图像所属的所有类别，如图 11.1 所示）。例如，当你在 Google Photos 应用中搜索关键词时，在幕后你正在查询一个非常大的多标签分类模型——一个拥有超过 20,000 个不同类别，并在数百万张图像上训练的模型。

+   *图像分割*，其目标是“分割”或“划分”图像为不同的区域，每个区域通常代表一个类别（如图 11.1 所示）。例如，当 Zoom 或 Google Meet 在视频通话中显示你背后的自定义背景时，它正在使用图像分割模型以像素级的精度区分你的面部和其后的内容。

+   *目标检测*，其目标是围绕图像中感兴趣的对象绘制矩形（称为*边界框*），并将每个矩形与一个类别关联。例如，自动驾驶汽车可以使用目标检测模型来监控其摄像头视野中的车辆、行人和标志。

![图像](img/2045275007f5b1bc39eac7d6965c23da.png)

![图 11.1](img/#figure-11-1)：三种主要的计算机视觉任务：分类、分割和检测

除了这三个任务之外，计算机视觉的深度学习还包括一些相对较窄的任务，例如图像相似度评分（估计两张图像在视觉上的相似程度）、关键点检测（在图像中定位感兴趣的特征，如面部特征）、姿态估计、3D 网格估计、深度估计等等。但首先，图像分类、图像分割和目标检测构成了每个机器学习工程师都应该熟悉的基石。几乎所有的计算机视觉应用都可以归结为这三个中的某一个。

你在第八章中已经看到了图像分类的实际应用。接下来，让我们深入探讨图像分割。这是一个非常有用且非常通用的技术，你可以直接运用你到目前为止所学到的知识来接近它。然后，在下一章中，你将详细了解目标检测。

### 图像分割类型

使用深度学习进行图像分割是关于使用模型为图像中的每个像素分配一个类别，从而将图像分割成不同的区域（如“背景”和“前景”或“道路”、“汽车”和“人行道”）。这类技术可以用于支持图像和视频编辑、自动驾驶、机器人技术、医学成像等多种有价值的应用。

你应该了解三种不同的图像分割类型：

+   *语义分割*，其中每个像素都被独立地分类到语义类别，如“猫”。如果图像中有两只猫，相应的像素都将映射到相同的通用“猫”类别（见图 11.2）。

+   *实例分割*，旨在解析出单个对象实例。在一个有两只猫的图像中，实例分割将区分属于“猫 1”的像素和属于“猫 2”的像素（见图 11.2）。

+   *全景分割*，通过为图像中的每个像素分配语义标签（如“猫”）和实例标签（如“猫 2”）来结合语义分割和实例分割。这是三种分割类型中最具信息量的。

![图片](img/7fd2d264b2190f8be9fac2be63b59e45.png)

图 11.2：语义分割与实例分割的比较

为了更熟悉分割，让我们从从头开始在您自己的数据上训练一个小型分割模型开始。

## 从头开始训练分割模型

在这个第一个例子中，我们将专注于语义分割。我们将再次查看猫和狗的图像，这次我们将学习区分主要主题及其背景。

### 下载分割数据集

我们将使用牛津-IIIT 宠物数据集（[`www.robots.ox.ac.uk/~vgg/data/pets/`](https://www.robots.ox.ac.uk/~vgg/data/pets/)），该数据集包含 7,390 张各种品种的猫和狗的图片，以及每张图片的前景-背景*分割掩码*。分割掩码是图像分割的等价物：它是一个与输入图像大小相同的图像，具有单个颜色通道，其中每个整数值对应于输入图像中相应像素的类别。在我们的情况下，我们的分割掩码的像素可以取三个整数值之一：

+   1 (前景)

+   2 (背景)

+   3 (轮廓)

首先，让我们通过使用`wget`和`tar`shell 工具下载和解压我们的数据集：

```py
!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
!tar -xf images.tar.gz
!tar -xf annotations.tar.gz 
```

输入图片存储在`images/`文件夹中作为 JPG 文件（例如`images/Abyssinian_1.jpg`），相应的分割掩码存储在`annotations/trimaps/`文件夹中，文件名与图片相同，为 PNG 文件（例如`annotations/trimaps/Abyssinian_1.png`）。

让我们准备输入文件路径的列表，以及相应的掩码文件路径列表：

```py
import pathlib

input_dir = pathlib.Path("images")
target_dir = pathlib.Path("annotations/trimaps")

input_img_paths = sorted(input_dir.glob("*.jpg"))
# Ignores some spurious files in the trimaps directory that start with
# a "."
target_paths = sorted(target_dir.glob("[!.]*.png")) 
```

现在，这些输入及其掩码看起来是什么样子？让我们快速看一下（见图 11.3）。

```py
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array, array_to_img

plt.axis("off")
# Displays input image number 9
plt.imshow(load_img(input_img_paths[9])) 
```

![](img/539e2f264128e5b1ac6b451aa3d2e9a4.png)

图 11.3：一个示例图像

让我们看看它的目标掩码（见图 11.4）：

```py
def display_target(target_array):
    # The original labels are 1, 2, and 3\. We subtract 1 so that the
    # labels range from 0 to 2, and then we multiply by 127 so that the
    # labels become 0 (black), 127 (gray), 254 (near-white).
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])

# We use color_mode='grayscale' so that the image we load is treated as
# having a single color channel.
img = img_to_array(load_img(target_paths[9], color_mode="grayscale"))
display_target(img) 
```

![](img/2686b58935fc0c0e5d28d781497c7142.png)

图 11.4：相应的目标掩码

接下来，让我们将我们的输入和目标加载到两个 NumPy 数组中。由于数据集非常小，我们可以将所有内容加载到内存中：

```py
import numpy as np
import random

# We resize everything to 200 x 200 for this example.
img_size = (200, 200)
# Total number of samples in the data
num_imgs = len(input_img_paths)

# Shuffles the file paths (they were originally sorted by breed). We
# use the same seed (1337) in both statements to ensure that the input
# paths and target paths stay in the same order.
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))

def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode="grayscale")
    )
    # Subtracts 1 so that our labels become 0, 1, and 2
    img = img.astype("uint8") - 1
    return img

# Loads all images in the input_imgs float32 array and their masks in
# the targets uint8 array (same order). The inputs have three channels
# (RGB values), and the targets have a single channel (which contains
# integer labels).
input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i]) 
```

和往常一样，让我们将数组分成训练集和验证集：

```py
# Reserves 1,000 samples for validation
num_val_samples = 1000
# Splits the data into a training and a validation set
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:] 
```

### 构建和训练分割模型

现在，是时候定义我们的模型了：

```py
import keras
from keras.layers import Rescaling, Conv2D, Conv2DTranspose

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    # Don't forget to rescale input images to the [0–1] range.
    x = Rescaling(1.0 / 255)(inputs)

    # We use padding="same" everywhere to avoid the influence of border
    # padding on feature map size.
    x = Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2D(128, 3, activation="relu", padding="same")(x)
    x = Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = Conv2D(256, 3, activation="relu", padding="same")(x)

    x = Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)

    # We end the model with a per-pixel three-way softmax to classify
    # each output pixel into one of our three categories.
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    return keras.Model(inputs, outputs)

model = get_model(img_size=img_size, num_classes=3) 
```

模型的前半部分与您用于图像分类的 ConvNet 非常相似：一系列`Conv2D`层，滤波器大小逐渐增加。我们通过每次减半的因子将图像下采样三次——最终得到大小为`(25, 25, 256)`的激活。这一半的目的是将图像编码成较小的特征图，其中每个空间位置（或“像素”）包含有关原始图像中较大空间块的信息。你可以将其理解为一种压缩。

这个模型的前半部分与您之前看到的分类模型的一个重要区别在于我们进行下采样的方式：在第八章的分类 ConvNets 中，我们使用了`MaxPooling2D`层来下采样特征图。在这里，我们通过在每个卷积层中添加*步长*来下采样（如果你不记得卷积步长的细节，请参阅第八章的 8.1.1 节）。我们这样做是因为，在图像分割的情况下，我们非常关注图像中信息的空间位置，因为我们需要将每个像素的目标掩码作为模型的输出。当你进行 2×2 最大池化时，你完全破坏了每个池化窗口内的位置信息：你为每个窗口返回一个标量值，而对四个位置中的哪一个值来自窗口一无所知。

因此，虽然最大池化层在分类任务中表现良好，但对于分割任务，它们可能会给我们带来相当大的伤害。同时，步长卷积在降采样特征图的同时，更好地保留了位置信息。在这本书的整个过程中，你会发现我们倾向于在关注特征位置的任何模型中使用步长而不是最大池化，例如第十七章中的生成模型。

模型的后半部分是一系列 `Conv2DTranspose` 层。那些是什么？嗯，模型前半部分的输出是一个形状为 `(25, 25, 256)` 的特征图，但我们希望最终的输出为每个像素预测一个类别，匹配原始的空间维度。最终的模型输出将具有形状 `(200, 200, num_classes)`，在这里是 `(200, 200, 3)`。因此，我们需要应用一种 *逆* 变换，即 *上采样* 特征图而不是下采样它们。这就是 `Conv2DTranspose` 层的目的：你可以把它想象成一种 *学习上采样* 的卷积层。如果你有一个形状为 `(100, 100, 64)` 的输入，并且通过 `Conv2D(128, 3, strides=2, padding="same")` 层运行它，你会得到一个形状为 `(50, 50, 128)` 的输出。如果你将这个输出通过 `Conv2DTranspose(64, 3, strides=2, padding="same")` 层运行，你会得到一个形状为 `(100, 100, 64)` 的输出，与原始输入相同。因此，通过一系列 `Conv2D` 层将我们的输入压缩成形状为 `(25, 25, 256)` 的特征图后，我们可以简单地应用相应的 `Conv2DTranspose` 层序列，然后是一个最终的 `Conv2D` 层，以产生形状为 `(200, 200, 3)` 的输出。

为了评估模型，我们将使用一个名为 *交并比* (IoU) 的指标。它是真实分割掩码与预测掩码之间匹配程度的度量。它可以针对每个类别单独计算，也可以在多个类别上平均计算。以下是它是如何工作的：

1.  计算掩码之间的 *交集*，即预测和真实重叠的区域。

1.  计算掩码的 *并集*，即两个掩码共同覆盖的总区域。这是我们感兴趣的全部空间——目标对象以及你的模型可能错误包含的任何额外部分。

1.  将交集区域除以并集区域以获得 IoU。它是一个介于 0 和 1 之间的数字，其中 1 表示完美匹配，0 表示完全未命中。

我们可以直接使用内置的 Keras 指标，而不是自己构建：

```py
foreground_iou = keras.metrics.IoU(
    # Specifies the total number of classes
    num_classes=3,
    # Specifies the class to compute IoU for (0 = foreground)
    target_class_ids=(0,),
    name="foreground_iou",
    # Our targets are sparse (integer class IDs).
    sparse_y_true=True,
    # But our model's predictions are a dense softmax!
    sparse_y_pred=False,
) 
```

现在，我们可以编译和拟合我们的模型：

```py
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[foreground_iou],
)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "oxford_segmentation.keras",
        save_best_only=True,
    ),
]
history = model.fit(
    train_input_imgs,
    train_targets,
    epochs=50,
    callbacks=callbacks,
    batch_size=64,
    validation_data=(val_input_imgs, val_targets),
) 
```

让我们显示我们的训练和验证损失（见图 11.5）：

```py
epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend() 
```

![](img/9f64353f124760b76a5383248fac2a7f.png)

图 11.5：显示训练和验证损失曲线

你可以看到，我们在大约第 25 个 epoch 时开始过拟合。让我们根据验证损失重新加载我们表现最好的模型，并展示如何使用它来预测一个分割掩码（见图 11.6）：

```py
model = keras.models.load_model("oxford_segmentation.keras")

i = 4
test_image = val_input_imgs[i]
plt.axis("off")
plt.imshow(array_to_img(test_image))

mask = model.predict(np.expand_dims(test_image, 0))[0]

# Utility to display a model's prediction
def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)

display_mask(mask) 
```

![](img/5df726de6b0d2524f74f492e8d42c2d1.png)

图 11.6：一个测试图像及其预测的分割掩码

我们的预测掩码中存在一些小的伪影，这是由前景和背景中的几何形状造成的。尽管如此，我们的模型看起来工作得很好。

## 使用预训练的分割模型

在第八章的图像分类示例中，你看到了使用预训练模型如何显著提高你的准确率——尤其是在你只有少量样本进行训练时。图像分割也不例外。

*Segment Anything 模型*，或简称 SAM，是一个强大的预训练分割模型，你可以用它来做几乎所有的事情。它由 Meta AI 开发并于 2023 年 4 月发布。它是在 1100 万张图像及其分割掩码上训练的，覆盖了超过 10 亿个对象实例。如此大量的训练数据为模型提供了对自然图像中几乎任何出现的对象的内置知识。

SAM 的主要创新在于它不仅限于预定义的对象类别集合。你可以通过提供一个你正在寻找的示例来简单地用它来分割新的对象。你甚至不需要先微调模型。让我们看看它是如何工作的。

### 下载 Segment Anything 模型

首先，让我们实例化 SAM 并下载其权重。同样，我们可以使用 KerasHub 包来使用这个预训练模型，而无需从头开始实现它。

记得我们在上一章中使用的`ImageClassifier`任务吗？我们可以使用另一个 KerasHub 任务`ImageSegmenter`来将预训练的图像分割模型包装成一个具有标准输入和输出的高级模型。在这里，我们将使用`sam_huge_sa1b`预训练模型，其中`sam`代表模型，`huge`指的是模型中的参数数量，而`sa1b`代表与模型一起发布的 SA-1B 数据集，包含 10 亿个注释过的掩码。现在让我们下载它：

```py
import keras_hub

model = keras_hub.models.ImageSegmenter.from_preset("sam_huge_sa1b") 
```

我们可以立即注意到的是，我们的模型确实是巨大的：

```py
>>> model.count_params()
641090864
```

在 6410 万个参数的情况下，SAM 是我们在这本书中使用的最大的模型。预训练模型越来越大，使用的数据越来越多这一趋势将在第十六章中更详细地讨论。

### Segment Anything 是如何工作的

在我们尝试使用该模型进行一些分割之前，让我们更多地谈谈 SAM 是如何工作的。模型的大部分能力都来自于预训练数据集的规模。Meta 与模型一起开发了 SA-1B 数据集，其中部分训练的模型被用来辅助数据标注过程。也就是说，数据集和模型以一种反馈循环的方式共同开发。

使用 SA-1B 数据集的目标是创建完全分割的图像，其中图像中的每个对象都分配了一个唯一的分割遮罩。见图 11.7 作为示例。数据集中的每张图像平均有约 100 个遮罩，有些图像有超过 500 个单独遮罩的对象。这是通过一个越来越自动化的数据收集流程完成的。最初，人类专家手动分割了一个小型的图像示例数据集，该数据集用于训练初始模型。该模型被用来帮助推动数据收集的半自动化阶段，在这一阶段，图像首先由 SAM 分割，然后通过人工校正和进一步标注进行改进。

![图片](img/f7fee6b67ad6bf846f8c26fdfbde4e3e.png)

图 11.7：SA-1B 数据集的一个示例图像

该模型在`(图像, 提示, 遮罩)`三元组上进行训练。`图像`和`提示`是模型的输入。图像可以是任何输入图像，而提示可以采取几种形式：

+   遮罩对象内部的一个点

+   围绕遮罩对象的一个框

给定`图像`和`提示`输入，模型预计将产生一个准确的预测遮罩，该遮罩对应于提示中指示的对象，并将其与地面真实`遮罩`标签进行比较。

该模型由几个独立的组件组成。一个类似于我们在前几章中使用的 Xception 模型的图像编码器，将输入图像转换为更小的图像嵌入。这是我们已知如何构建的。

接下来，我们添加一个提示编码器，它负责将之前提到的任何形式的提示映射到一个嵌入向量，以及一个遮罩解码器，它接收图像嵌入和提示嵌入，并输出几个可能的预测遮罩。我们不会在这里详细介绍提示编码器和遮罩解码器的细节，因为它们使用了我们在后面的章节中才会看到的建模技术。我们可以将这些预测遮罩与我们的地面真实遮罩进行比较，就像我们在本章早期部分所做的那样（见图 11.8）。

![图片](img/02a451fa68c4d16b5f76ddfbd8d7e7e4.png)

图 11.8：Segment Anything 高级架构概述

所有这些子组件都是通过形成新的`(图像, 提示, 遮罩)`三元组批次来同时训练的，这些批次是从 SA-1B 图像和遮罩数据中训练的。这里的流程实际上相当简单。对于给定的输入图像，选择输入中的一个随机遮罩。接下来，随机选择是否创建一个框提示或一个点提示。要创建一个点提示，选择遮罩标签内的一个随机像素。要创建一个框提示，围绕遮罩标签内的所有点绘制一个框。我们可以无限重复这个过程，从每个图像输入中采样一定数量的`(图像, 提示, 遮罩)`三元组。

### 准备测试图像

让我们通过尝试模型来使这个例子更具体。我们可以从加载用于分割工作的测试图像开始。我们将使用一个水果碗的图片（见图 11.9）：

```py
# Downloads the image and returns the local file path
path = keras.utils.get_file(
    origin="https://s3.amazonaws.com/keras.io/img/book/fruits.jpg"
)
# Loads the image as a Python Imaging Library (PIL) object
pil_image = keras.utils.load_img(path)
# Turns the PIL object into a NumPy matrix
image_array = keras.utils.img_to_array(pil_image)

# Displays the NumPy matrix
plt.imshow(image_array.astype("uint8"))
plt.axis("off")
plt.show() 
```

![图片](img/cb3ddcfe0d2fd3d54cc924ac0bb4ea5c.png)

图 11.9：我们的测试图像

SAM 预期输入的尺寸为 1024 × 1024。然而，强制将任意图像调整到 1024 × 1024 的大小会扭曲其宽高比——例如，我们的图像不是正方形。更好的做法是首先将图像调整到其最长边为 1,024 像素，然后用填充值（如 0）填充剩余的像素。我们可以通过在 `keras.ops.image.resize()` 操作中使用 `pad_to_aspect_ratio` 参数来实现这一点，如下所示：

```py
from keras import ops

image_size = (1024, 1024)

def resize_and_pad(x):
    return ops.image.resize(x, image_size, pad_to_aspect_ratio=True)

image = resize_and_pad(image_array) 
```

接下来，让我们定义一些在使用模型时将很有用的实用工具。我们需要做的是

+   显示图像。

+   在图像上显示叠加的分割掩码。

+   在图像上突出显示特定的点。

+   在图像上显示叠加的框。

我们的所有工具都接受一个 Matplotlib `axis` 对象（记作 `ax`），这样它们就可以写入同一个图像：

```py
import matplotlib.pyplot as plt
from keras import ops

def show_image(image, ax):
    ax.imshow(ops.convert_to_numpy(image).astype("uint8"))

def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w, _ = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(points, ax):
    x, y = points[:, 0], points[:, 1]
    ax.scatter(x, y, c="green", marker="*", s=375, ec="white", lw=1.25)

def show_box(box, ax):
    box = box.reshape(-1)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, ec="red", fc="none", lw=2)) 
```

### 使用目标点提示模型

要使用 SAM，你需要提示它。这意味着我们需要以下之一：

+   *点提示*——在图像中选择一个点，并让模型分割该点所属的对象。

+   *框提示*——在对象周围画一个大致的框（不需要特别精确），然后让模型在框内分割对象。

让我们从点提示开始。点被标记，其中 1 表示前景（你想要分割的对象），0 表示背景（对象周围的一切）。在模糊的情况下，为了提高你的结果，你可以传递多个标记的点，而不是单个点，以细化你想要包含（标记为 1 的点）和排除（标记为 0 的点）的定义。

我们尝试一个单独的前景点（见图 11.10）。这是一个测试点：

```py
import numpy as np

# Coordinates of our point
input_point = np.array([[580, 450]])
# 1 means foreground, and 0 means background.
input_label = np.array([1])

plt.figure(figsize=(10, 10))
# "gca" means "get current axis" — the current figure.
show_image(image, plt.gca())
show_points(input_point, plt.gca())
plt.show() 
```

![](img/1fb3211965623f1112b05761214c02f1.png)

图 11.10：一个提示点，落在桃子上

让我们用这个图像提示 SAM：

```py
outputs = model.predict(
    {
        "images": ops.expand_dims(image, axis=0),
        "points": ops.expand_dims(input_point, axis=0),
        "labels": ops.expand_dims(input_label, axis=0),
    }
) 
```

返回值 `outputs` 有一个 `"masks"` 字段，它包含四个 256 × 256 的候选掩码，按降低的匹配质量排序。掩码的质量分数作为模型输出的 `"iou_pred"` 字段的一部分提供：

```py
>>> outputs["masks"].shape
(1, 4, 256, 256)
```

让我们在图像上叠加第一个掩码（见图 11.11）：

```py
def get_mask(sam_outputs, index=0):
    mask = sam_outputs["masks"][0][index]
    mask = np.expand_dims(mask, axis=-1)
    mask = resize_and_pad(mask)
    return ops.convert_to_numpy(mask) > 0.0

mask = get_mask(outputs, index=0)

plt.figure(figsize=(10, 10))
show_image(image, plt.gca())
show_mask(mask, plt.gca())
show_points(input_point, plt.gca())
plt.show() 
```

![](img/9a04a54d693d52d5d46a0779a32ef31d.png)

图 11.11：分割的桃子

很不错！

接下来，让我们尝试一个香蕉。我们将用坐标 `(300, 550)` 来提示模型，这个坐标落在从左数第二个香蕉上（见图 11.12）：

```py
input_point = np.array([[300, 550]])
input_label = np.array([1])

outputs = model.predict(
    {
        "images": ops.expand_dims(image, axis=0),
        "points": ops.expand_dims(input_point, axis=0),
        "labels": ops.expand_dims(input_label, axis=0),
    }
)
mask = get_mask(outputs, index=0)

plt.figure(figsize=(10, 10))
show_image(image, plt.gca())
show_mask(mask, plt.gca())
show_points(input_point, plt.gca())
plt.show() 
```

![](img/d57b293d144b43710f13bcb5dc407813.png)

图 11.12：分割的香蕉

现在，关于其他掩码候选者呢？它们对于模糊提示很有用。让我们尝试绘制其他三个掩码（见图 11.13）：

```py
fig, axes = plt.subplots(1, 3, figsize=(20, 60))
masks = outputs["masks"][0][1:]
for i, mask in enumerate(masks):
    show_image(image, axes[i])
    show_points(input_point, axes[i])
    mask = get_mask(outputs, index=i + 1)
    show_mask(mask, axes[i])
    axes[i].set_title(f"Mask {i + 1}", fontsize=16)
    axes[i].axis("off")
plt.show() 
```

![](img/2825d36b86ac06b70d4c188bdeeb0383.png)

图 11.13：香蕉提示的替代分割掩码

如你所见，模型找到的替代分割包括两个香蕉。

### 使用目标框提示模型

除了提供一个或多个目标点之外，您还可以提供近似分割对象位置的框。这些框应通过其左上角和右下角的坐标传递。这里有一个围绕芒果的框（见图 11.14）：

```py
input_box = np.array(
    [
        # Top-left corner
        [520, 180],
        # Bottom-right corner
        [770, 420],
    ]
)

plt.figure(figsize=(10, 10))
show_image(image, plt.gca())
show_box(input_box, plt.gca())
plt.show() 
```

![图片](img/611f443ab648ad48cdc642f1c265fd6e.png)

图 11.14：围绕芒果的框提示

让我们用它来提示 SAM（见图 11.15）：

```py
outputs = model.predict(
    {
        "images": ops.expand_dims(image, axis=0),
        "boxes": ops.expand_dims(input_box, axis=(0, 1)),
    }
)
mask = get_mask(outputs, 0)
plt.figure(figsize=(10, 10))
show_image(image, plt.gca())
show_mask(mask, plt.gca())
show_box(input_box, plt.gca())
plt.show() 
```

![图片](img/afe8dd8fc5b03bf9c605eb1d054321d6.png)

图 11.15：分割的芒果

SAM 可以是一个强大的工具，快速创建带有分割掩码的大图像数据集。

## 摘要

+   图像分割是计算机视觉任务的主要类别之一。它包括计算分割掩码，这些掩码描述了图像在像素级别的内容。

+   要构建自己的分割模型，使用一系列步进`Conv2D`层来“压缩”输入图像到一个较小的特征图，然后使用相应的`Conv2DTranspose`层堆叠来“扩展”特征图，使其大小与输入图像相同的分割掩码。

+   您还可以使用预训练的分割模型。KerasHub 中包含的 Segment Anything 是一个支持图像提示、文本提示、点提示和框提示的强大模型。

### 脚注

1.  Kirillov 等人，“Segment Anything”，在*IEEE/CVF 国际计算机视觉会议论文集*，arXiv (2023)，[`arxiv.org/abs/2304.02643`](https://arxiv.org/abs/2304.02643)。[[↩]](#footnote-link-1)
