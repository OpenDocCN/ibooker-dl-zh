# 物体检测

> 原文：[`deeplearningwithpython.io/chapters/chapter12_object-detection`](https://deeplearningwithpython.io/chapters/chapter12_object-detection)

物体检测主要是围绕图像中感兴趣对象的绘制框（称为 *边界框*）（参见图 12.1）。这使得你不仅知道图像中有什么对象，还能知道它们的位置。其最常见的一些应用包括

+   *计数* — 查找图像中对象的实例数量。

+   *跟踪* — 通过对电影每一帧执行物体检测来跟踪场景中物体随时间移动。

+   *裁剪* — 识别图像中包含感兴趣对象的区域进行裁剪，并将图像块的高分辨率版本发送到分类器或光学字符识别（OCR）模型。

![图片](img/80fcdad34894a13571da65573962efca.png)

图 12.1：物体检测器在图像中绘制边界框并对它们进行标记。

你可能会想，如果我有一个对象实例的分割掩码，我已能计算包含掩码的最小框的坐标。那么我们是否可以一直使用图像分割？我们还需要物体检测模型吗？

事实上，分割是检测的严格超集。它返回检测模型可能返回的所有信息——以及更多。这种信息量的增加带来了显著的计算成本：一个好的物体检测模型通常比图像分割模型运行得快得多。它还有数据标注成本：要训练分割模型，你需要收集像素级精确的掩码，这比物体检测模型所需的边界框生产耗时得多。

因此，如果你不需要像素级信息，你总是想使用物体检测模型——例如，如果你只想在图像中计数对象。

## 单阶段与两阶段物体检测器

物体检测架构主要分为两大类：

+   两阶段检测器，首先提取区域提议，称为基于区域的卷积神经网络（R-CNN）模型

+   单阶段检测器，例如 RetinaNet 或 You Only Look Once 系列模型

这是它们的工作原理。

### 两阶段 R-CNN 检测器

基于区域的卷积神经网络（R-CNN）模型是一个两阶段模型。第一阶段接收一个图像，并在看起来像物体的区域周围生成几千个部分重叠的边界框。这些框被称为 *区域提议*。这一阶段并不十分智能，所以在那个阶段我们还不确定提议的区域是否确实包含对象，以及如果包含，包含哪些对象。

这是第二阶段的工作——一个卷积神经网络，它查看每个区域提议并将其分类为多个预定义的类别，就像你在第九章中看到的模型一样（见图 12.2）。具有低得分的区域提议被丢弃。然后我们剩下的一组箱子，每个箱子都有一个特定类别的较高类别存在分数。最后，围绕每个对象的边界框进一步细化，以消除重复并尽可能使每个边界框尽可能精确。

![图片](img/314b3b5fa4332a8e1c10e75fadb679d7.png)

图 12.2：R-CNN 首先提取区域提议，然后使用卷积神经网络（CNN）对这些提议进行分类。

在早期 R-CNN 版本中，第一阶段是一个名为*选择性搜索*的启发式模型，它使用一些空间一致性的定义来识别类似物体的区域。"启发式"是你在机器学习中会经常听到的一个术语——它仅仅意味着“某人编造的一套硬编码的规则。”它通常用于与学习模型（规则是自动导出的）或理论导出的模型相对立。在 R-CNN 的后期版本中，如 Faster-R-CNN，框生成阶段变成了一个深度学习模型，称为区域提议网络。

R-CNN 的双阶段方法在实践中效果很好，但计算成本相当高，最显著的是因为它要求你为每张处理的图像分类数千个补丁。这使得它不适合大多数实时应用和嵌入式系统。我的观点是，在实际应用中，你通常不需要像 R-CNN 这样的计算密集型目标检测系统，因为如果你在服务器端使用强大的 GPU 进行推理，那么你可能会更愿意使用像我们在上一章中看到的 Segment Anything 模型这样的分割模型。如果你资源有限，那么你将想要使用一个计算效率更高的目标检测架构——单阶段检测器。

### 单阶段检测器

大约在 2015 年，研究人员和从业者开始尝试使用单个深度学习模型来联合预测边界框坐标及其标签，这种架构被称为*单阶段检测器*。单阶段检测器的主要家族包括 RetinaNet、单次多框检测器（SSD）和 YOLO 系列，简称 YOLO。是的，就像那个梗。这是故意的。

单阶段检测器，尤其是最近的 YOLO 迭代版本，与双阶段检测器相比，具有显著更快的速度和更高的效率，尽管在准确性方面存在一些潜在的小型权衡。如今，YOLO 可以说是最受欢迎的目标检测模型，尤其是在实时应用方面。通常每年都会有一个新版本出现——有趣的是，每个新版本往往是由不同的组织开发的。

在下一节中，我们将从头开始构建一个简化的 YOLO 模型。

## 从头开始训练 YOLO 模型

总体来说，构建一个目标检测器可能是一项相当艰巨的任务——并不是说它在理论上有什么复杂之处。只是需要大量的代码来处理边界框和预测输出的操作。为了保持简单，我们将重新创建 2015 年的第一个 YOLO 模型。截至本文写作时，已有 12 个 YOLO 版本，但原始版本更易于操作。

### 下载 COCO 数据集

在我们开始创建模型之前，我们需要用于训练的数据。COCO 数据集^([[1]](#footnote-1))，简称*Common Objects in Context*，是众所周知且最常用的目标检测数据集之一。它由多个不同来源的真实世界照片以及人类创建的注释组成。这包括对象标签、边界框注释和完整的分割掩码。我们将忽略分割掩码，只使用边界框。

让我们下载 2017 版本的 COCO 数据集。虽然按照今天的标准这不是一个大型数据集，但这个 18GB 的数据集将是本书中我们使用的最大数据集。如果您在阅读代码时运行，这是一个休息的好机会。

```py
import keras
import keras_hub

images_path = keras.utils.get_file(
    "coco",
    "http://images.cocodataset.org/zips/train2017.zip",
    extract=True,
)
annotations_path = keras.utils.get_file(
    "annotations",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    extract=True,
) 
```

列表 12.1：下载 2017 年 COCO 数据集

在我们准备好使用这些数据之前，我们需要进行一些输入处理。第一次下载给我们提供了一个所有 COCO 图像的无标签目录。第二次下载包含所有图像元数据，通过一个 JSON 文件。COCO 将每个图像文件与一个 ID 关联，每个边界框都与这些 ID 之一配对。我们需要将所有框和图像数据汇总在一起。

每个边界框都包含`x, y, width, height`像素坐标，从图像的左上角开始。在我们加载数据时，我们可以调整所有边界框坐标，使它们成为 `[0, 1]` 单位正方形中的点。这将使操作这些框变得更加容易，而无需检查每个输入图像的大小。

```py
import json

with open(f"{annotations_path}/annotations/instances_train2017.json", "r") as f:
    annotations = json.load(f)

# Sorts image metadata by ID
images = {image["id"]: image for image in annotations["images"]}

# Converts bounding box to coordinates on a unit square
def scale_box(box, width, height):
    scale = 1.0 / max(width, height)
    x, y, w, h = [v * scale for v in box]
    x += (height - width) * scale / 2 if height > width else 0
    y += (width - height) * scale / 2 if width > height else 0
    return [x, y, w, h]

# Aggregates all bounding box annotations by image ID
metadata = {}
for annotation in annotations["annotations"]:
    id = annotation["image_id"]
    if id not in metadata:
        metadata[id] = {"boxes": [], "labels": []}
    image = images[id]
    box = scale_box(annotation["bbox"], image["width"], image["height"])
    metadata[id]["boxes"].append(box)
    metadata[id]["labels"].append(annotation["category_id"])
    metadata[id]["path"] = images_path + "/train2017/" + image["file_name"]
metadata = list(metadata.values()) 
```

列表 12.2：解析 COCO 数据

让我们看看我们刚刚加载的数据。

```py
>>> len(metadata)
117266
>>> min([len(x["boxes"]) for x in metadata])
1
>>> max([len(x["boxes"]) for x in metadata])
63
>>> max(max(x["labels"]) for x in metadata) + 1
91
>>> metadata[435]
{"boxes": [[0.12, 0.27, 0.57, 0.33],
  [0.0, 0.15, 0.79, 0.69],
  [0.0, 0.12, 1.0, 0.75]],
 "labels": [17, 15, 2],
 "path": "/root/.keras/datasets/coco/train2017/000000171809.jpg"}
>>> [keras_hub.utils.coco_id_to_name(x) for x in metadata[435]["labels"]]
["cat", "bench", "bicycle"]
```

列表 12.3：检查 COCO 数据

我们有 117,266 张图像。每张图像可以有 1 到 63 个与边界框关联的对象。COCO 数据集创建者选择了 91 个可能的标签。

我们可以使用 KerasHub 实用工具`keras_hub.utils.coco_id_to_name(id)`将这些整数标签映射到可读的人名，类似于我们在第八章中用来解码 ImageNet 预测到文本标签的实用工具。

让我们可视化一个示例图像，使这一点更加具体。我们可以定义一个函数来使用 Matplotlib 绘制图像，另一个函数来在这个图像上绘制标记的边界框。我们将在本章中需要这两个函数。我们可以使用 HSV 颜色空间作为一个简单的技巧来为每个新标签生成新的颜色。通过固定颜色的饱和度和亮度，只更新其色调，我们可以生成鲜艳的新颜色，这些颜色可以从我们的图像中清楚地脱颖而出。

```py
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle

color_map = {0: "gray"}

def label_to_color(label):
    # Uses the golden ratio to generate new hues of a bright color with
    # the HSV colorspace
    if label not in color_map:
        h, s, v = (len(color_map) * 0.618) % 1, 0.5, 0.9
        color_map[label] = hsv_to_rgb((h, s, v))
    return color_map[label]

def draw_box(ax, box, text, color):
    x, y, w, h = box
    ax.add_patch(Rectangle((x, y), w, h, lw=2, ec=color, fc="none"))
    textbox = dict(fc=color, pad=1, ec="none")
    ax.text(x, y, text, c="white", size=10, va="bottom", bbox=textbox)

def draw_image(ax, image):
    # Draws the image on a unit cube with (0, 0) at the top left
    ax.set(xlim=(0, 1), ylim=(1, 0), xticks=[], yticks=[], aspect="equal")
    image = plt.imread(image)
    height, width = image.shape[:2]
    # Pads the image so it fits inside the unit cube
    hpad = (1 - height / width) / 2 if width > height else 0
    wpad = (1 - width / height) / 2 if height > width else 0
    extent = [wpad, 1 - wpad, 1 - hpad, hpad]
    ax.imshow(image, extent=extent) 
```

列表 12.4：使用框注释可视化 COCO 图像

让我们使用我们的新可视化来查看我们之前检查的样本图像^([[2]](#footnote-2))（见图 12.3）：

```py
sample = metadata[435]
ig, ax = plt.subplots(dpi=300)
draw_image(ax, sample["path"])
for box, label in zip(sample["boxes"], sample["labels"]):
    label_name = keras_hub.utils.coco_id_to_name(label)
    draw_box(ax, box, label_name, label_to_color(label))
plt.show() 
```

![图片](img/c1af83579d7491efa5129821c5d53854.png)

图 12.3：YOLO 为每个图像区域输出一个边界框预测和类别标签。

虽然在所有 18GB 的输入数据上训练会很刺激，但我们希望这本书中的示例能够在普通的硬件上轻松运行。如果我们只限制使用只有四个或更少框的图像，我们将使我们的训练问题更容易，并且将数据大小减半。让我们这样做，并打乱我们的数据——图像按对象类型分组，这对训练来说会很糟糕：

```py
import random

metadata = list(filter(lambda x: len(x["boxes"]) <= 4, metadata))
random.shuffle(metadata) 
```

数据加载就到这里！让我们开始创建我们的 YOLO 模型。

### 创建 YOLO 模型

如前所述，YOLO 模型是一个单阶段检测器。而不是首先尝试在场景中识别所有候选对象，然后对对象区域进行分类，YOLO 将一次性提出边界框和对象标签。

我们将把图像分割成网格，并在每个网格位置预测两个不同的输出——一个边界框和一个类别标签。在 Redmon 等人原始论文中，模型实际上在每个网格位置预测了多个边界框，但我们保持简单，只在每个网格方块中预测一个边界框。

大多数图像在网格上不会均匀分布对象，为了解决这个问题，模型将输出一个*置信度分数*，与每个框一起，如图 12.4 所示。我们希望当在某个位置检测到对象时，这个置信度很高，而没有对象时为零。大多数网格位置将没有对象，应该报告接近零的置信度。

![图片](img/3fde6b657f5c44d01f695075c1fa3462.png)

图 12.4：YOLO 在第一篇 YOLO 论文中的输出可视化

与计算机视觉中的许多模型一样，YOLO 模型使用 ConvNet *骨干*来获取输入图像的有趣高级特征，这是我们首次在第八章中探讨的概念。在他们的论文中，作者创建了自己的骨干模型，并使用 ImageNet 对其进行预训练以进行分类。我们不必自己这样做，而是可以使用 KerasHub 来加载预训练的骨干。

与本书中迄今为止使用的 Xception 骨干网络不同，我们将切换到 ResNet，这是我们在第九章首次提到的模型系列。结构相当类似，但 ResNet 使用步长而不是池化层来下采样图像。正如我们在第十一章中提到的，当我们关注输入的 *空间位置* 时，步长卷积更好。

让我们加载我们的预训练模型和匹配的预处理（以调整图像大小）。我们将调整图像大小到 448 × 448；图像输入大小对于目标检测任务非常重要。

```py
image_size = 448

backbone = keras_hub.models.Backbone.from_preset(
    "resnet_50_imagenet",
)
preprocessor = keras_hub.layers.ImageConverter.from_preset(
    "resnet_50_imagenet",
    image_size=(image_size, image_size),
) 
```

列表 12.5：加载 ResNet 模型

接下来，我们可以通过添加用于输出框和类别预测的新层，将骨干网络转换为一个检测模型。YOLO 论文中提出的设置相当简单。取卷积网络骨干网络的输出，通过中间带有激活函数的两个密集连接层，然后分割输出。前五个数字将用于边界框预测（四个用于框和一个是框的置信度）。其余的将用于图 12.4 中显示的 *类别概率图* —— 对所有可能的 91 个标签在每个网格位置上的分类预测。

让我们把它写出来。

```py
from keras import layers

grid_size = 6
num_labels = 91

inputs = keras.Input(shape=(image_size, image_size, 3))
x = backbone(inputs)
# Makes our backbone outputs smaller and then flattens the output
# features
x = layers.Conv2D(512, (3, 3), strides=(2, 2))(x)
x = keras.layers.Flatten()(x)
# Passes our flattened feature maps through two densely connected
# layers
x = layers.Dense(2048, activation="relu", kernel_initializer="glorot_normal")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(grid_size * grid_size * (num_labels + 5))(x)
# Reshapes outputs to a 6 × 6 grid
x = layers.Reshape((grid_size, grid_size, num_labels + 5))(x)
# Split box and class predictions
box_predictions = x[..., :5]
class_predictions = layers.Activation("softmax")(x[..., 5:])
outputs = {"box": box_predictions, "class": class_predictions}
model = keras.Model(inputs, outputs) 
```

列表 12.6：附加 YOLO 预测头

我们可以通过查看模型摘要来更好地理解模型：

```py
>>> model.summary()
Model: "functional_3"
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)          ┃ Output Shape      ┃     Param # ┃ Connected to       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ input_layer_7         │ (None, 448, 448,  │           0 │ -                  │
│ (InputLayer)          │ 3)                │             │                    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ res_net_backbone_12   │ (None, 14, 14,    │  23,580,512 │ input_layer_7[0][… │
│ (ResNetBackbone)      │ 2048)             │             │                    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ conv2d_3 (Conv2D)     │ (None, 6, 6, 512) │   9,437,696 │ res_net_backbone_… │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ flatten_3 (Flatten)   │ (None, 18432)     │           0 │ conv2d_3[0][0]     │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ dense_6 (Dense)       │ (None, 2048)      │  37,750,784 │ flatten_3[0][0]    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ dropout_3 (Dropout)   │ (None, 2048)      │           0 │ dense_6[0][0]      │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ dense_7 (Dense)       │ (None, 3456)      │   7,081,344 │ dropout_3[0][0]    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ reshape_3 (Reshape)   │ (None, 6, 6, 96)  │           0 │ dense_7[0][0]      │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ get_item_7 (GetItem)  │ (None, 6, 6, 91)  │           0 │ reshape_3[0][0]    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ get_item_6 (GetItem)  │ (None, 6, 6, 5)   │           0 │ reshape_3[0][0]    │
├───────────────────────┼───────────────────┼─────────────┼────────────────────┤
│ activation_33         │ (None, 6, 6, 91)  │           0 │ get_item_7[0][0]   │
│ (Activation)          │                   │             │                    │
└───────────────────────┴───────────────────┴─────────────┴────────────────────┘
 Total params: 77,850,336 (296.98 MB)
 Trainable params: 77,797,088 (296.77 MB)
 Non-trainable params: 53,248 (208.00 KB)
```

我们的骨干网络输出形状为 `(batch_size, 14, 14, 2048)`。这意味着每张图像有 401,408 个输出浮点数，对于输入到我们的密集层来说有点太多。我们使用步长卷积层将特征图下采样到 `(batch_size, 6, 6, 512)`，每张图像有 18,432 个浮点数，更容易处理。

接下来，我们可以添加我们的两个密集连接层。我们将整个特征图展平，通过一个带有 `relu` 激活的 `Dense` 层，然后通过一个最终带有我们确切数量的输出预测的 `Dense` 层 — 5 个用于边界框和置信度，以及每个网格位置上的每个对象类别的 91 个。

最后，我们将输出重新塑形回 6 × 6 的网格，并分割我们的框和类别预测。对于我们的分类输出，我们通常应用 softmax。框输出需要更多的特殊考虑；我们将在后面讨论这个问题。

看起来不错！请注意，由于我们通过分类层展平整个特征图，每个网格检测器都可以使用整个图像的特征；没有局部性约束。这是有意为之的 — 大型对象不会局限于单个网格单元。

### 准备 COCO 数据以供 YOLO 模型使用

我们的模式相对简单，但我们仍然需要预处理我们的输入，以便与预测网格对齐。每个网格检测器将负责检测任何中心落在网格框内的框。我们的模型将为框 `(x, y, w, h, confidence)` 输出五个浮点数。`x` 和 `y` 将表示对象中心相对于网格单元边界的相对位置（从 0 到 1）。`w` 和 `h` 将表示对象大小相对于图像大小的相对位置。

我们已经在训练数据中有了正确的 `w` 和 `h` 值。然而，我们需要将我们的 `x` 和 `y` 值从网格中转换出来。让我们定义两个实用工具：

```py
def to_grid(box):
    x, y, w, h = box
    cx, cy = (x + w / 2) * grid_size, (y + h / 2) * grid_size
    ix, iy = int(cx), int(cy)
    return (ix, iy), (cx - ix, cy - iy, w, h)

def from_grid(loc, box):
    (xi, yi), (x, y, w, h) = loc, box
    x = (xi + x) / grid_size - w / 2
    y = (yi + y) / grid_size - h / 2
    return (x, y, w, h) 
```

让我们重新整理我们的训练数据，使其符合这个新的网格结构。只要我们的数据集与我们的网格一样长，我们就可以创建两个数组：

+   第一个将包含我们的类别概率图。我们将标记所有与边界框相交的网格单元，并使用正确的标签。为了使我们的代码简单，我们不会担心重叠的框。

+   第二个将包含实际的框。我们将所有框转换到网格中，并用框的坐标为正确的网格单元标记。在我们标记的数据中，实际框的置信度始终为 1，而所有其他位置的置信度将为 0。

```py
import numpy as np
import math

class_array = np.zeros((len(metadata), grid_size, grid_size))
box_array = np.zeros((len(metadata), grid_size, grid_size, 5))

for index, sample in enumerate(metadata):
    boxes, labels = sample["boxes"], sample["labels"]
    for box, label in zip(boxes, labels):
        (x, y, w, h) = box
        # Finds all grid cells whose center falls inside the box
        left, right = math.floor(x * grid_size), math.ceil((x + w) * grid_size)
        bottom, top = math.floor(y * grid_size), math.ceil((y + h) * grid_size)
        class_array[index, bottom:top, left:right] = label

for index, sample in enumerate(metadata):
    boxes, labels = sample["boxes"], sample["labels"]
    for box, label in zip(boxes, labels):
        # Transforms the box to the grid coordinate system
        (xi, yi), (grid_box) = to_grid(box)
        box_array[index, yi, xi] = [*grid_box, 1.0]
        # Makes sure the class label for the box's center location
        # matches the box
        class_array[index, yi, xi] = label 
```

列表 12.7：创建 YOLO 目标

让我们使用我们的框绘制助手可视化我们的 YOLO 训练数据（图 12.5）。我们将在第一个输入图像上绘制整个类别激活图^([[4]](#footnote-4))，并添加框的置信度分数及其标签。

```py
def draw_prediction(image, boxes, classes, cutoff=None):
    fig, ax = plt.subplots(dpi=300)
    draw_image(ax, image)
    # Draws the YOLO output grid and class probability map
    for yi, row in enumerate(classes):
        for xi, label in enumerate(row):
            color = label_to_color(label) if label else "none"
            x, y, w, h = (v / grid_size for v in (xi, yi, 1.0, 1.0))
            r = Rectangle((x, y), w, h, lw=2, ec="black", fc=color, alpha=0.5)
            ax.add_patch(r)
    # Draws all boxes at each grid location above our cutoff
    for yi, row in enumerate(boxes):
        for xi, box in enumerate(row):
            box, confidence = box[:4], box[4]
            if not cutoff or confidence >= cutoff:
                box = from_grid((xi, yi), box)
                label = classes[yi, xi]
                color = label_to_color(label)
                name = keras_hub.utils.coco_id_to_name(label)
                draw_box(ax, box, f"{name} {max(confidence, 0):.2f}", color)
    plt.show()

draw_prediction(metadata[0]["path"], box_array[0], class_array[0], cutoff=1.0) 
```

列表 12.8：可视化 YOLO 目标

![](img/865c1914e45f8307001760f66070ae00.png)

图 12.5：YOLO 为每个图像区域输出一个边界框预测和类别标签。

最后，让我们使用 `tf.data` 加载我们的图像数据。我们将从磁盘加载我们的图像，应用我们的预处理，并将它们分批。我们还应该分割一个验证集来监控训练。

```py
import tensorflow as tf

# Loads and resizes the model with tf.data
def load_image(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_jpeg(x, channels=3)
    return preprocessor(x)

images = tf.data.Dataset.from_tensor_slices([x["path"] for x in metadata])
images = images.map(load_image, num_parallel_calls=8)
labels = {"box": box_array, "class": class_array}
labels = tf.data.Dataset.from_tensor_slices(labels)

# Creates a merged dataset and batches it
dataset = tf.data.Dataset.zip(images, labels).batch(16).prefetch(2)
# Splits off some validation data
val_dataset, train_dataset = dataset.take(500), dataset.skip(500) 
```

列表 12.9：创建用于训练的数据集

有了这些，我们的数据就准备好进行训练了。

### 训练 YOLO 模型

我们已经有了我们的模型和训练数据，但在我们实际运行 `fit()` 之前，我们还需要一个最后的元素：损失函数。我们的模型输出预测框和预测网格标签。在第七章中，我们看到了如何为每个输出定义多个损失——Keras 将在训练期间简单地将损失相加。我们可以像往常一样用 `sparse_categorical_crossentropy` 处理分类损失。

然而，框损失需要一些特别的考虑。YOLO 作者提出的基损失相当简单。他们使用目标框参数与预测参数之间差异的平方和误差。我们只为标记数据中有实际框的网格单元计算这个误差。

损失函数中的难点在于边界框置信度的输出。作者希望置信度输出不仅反映物体的存在，还要反映预测框的好坏。为了创建一个平滑的框预测好坏估计，作者提出使用我们在上一章看到的*交并比*（IoU）度量。如果一个网格单元为空，该位置的预测置信度应该是零。然而，如果一个网格单元包含一个物体，我们可以使用当前框预测与实际框之间的 IoU 分数作为目标置信度值。这样，随着模型在预测框位置方面变得更好，IoU 分数和学习的置信度值将上升。

这需要自定义损失函数。我们可以先定义一个计算目标和预测框的 IoU 分数的实用工具。

```py
from keras import ops

# Unpacks a tensor of boxes
def unpack(box):
    return box[..., 0], box[..., 1], box[..., 2], box[..., 3]

# Computes the intersection area between two box tensors
def intersection(box1, box2):
    cx1, cy1, w1, h1 = unpack(box1)
    cx2, cy2, w2, h2 = unpack(box2)
    left = ops.maximum(cx1 - w1 / 2, cx2 - w2 / 2)
    bottom = ops.maximum(cy1 - h1 / 2, cy2 - h2 / 2)
    right = ops.minimum(cx1 + w1 / 2, cx2 + w2 / 2)
    top = ops.minimum(cy1 + h1 / 2, cy2 + h2 / 2)
    return ops.maximum(0.0, right - left) * ops.maximum(0.0, top - bottom)

# Computes the IoU between two box tensors
def intersection_over_union(box1, box2):
    cx1, cy1, w1, h1 = unpack(box1)
    cx2, cy2, w2, h2 = unpack(box2)
    intersection_area = intersection(box1, box2)
    a1 = ops.maximum(w1, 0.0) * ops.maximum(h1, 0.0)
    a2 = ops.maximum(w2, 0.0) * ops.maximum(h2, 0.0)
    union_area = a1 + a2 - intersection_area
    return ops.divide_no_nan(intersection_area, union_area) 
```

代码列表 12.10：计算两个框的 IoU

让我们使用这个实用工具来定义我们的自定义损失。Redmon 等人提出了一些损失缩放技巧来提高训练质量：

+   他们将框放置损失放大五倍，使其成为整体训练中更重要的一部分。

+   由于大多数网格单元是空的，他们还将空位置的置信度损失缩小两倍。这保持了这些零置信度预测不会压倒损失。

+   他们计算损失之前先取宽度和高度的平方根。这是为了防止大框相对于小框产生不成比例的影响。我们将使用一个保留输入符号的`sqrt`函数，因为我们的模型在训练开始时可能会预测负的宽度和高度。

让我们把它写出来。

```py
def signed_sqrt(x):
    return ops.sign(x) * ops.sqrt(ops.absolute(x) + keras.config.epsilon())

def box_loss(true, pred):
    # Unpacks values
    xy_true, wh_true, conf_true = true[..., :2], true[..., 2:4], true[..., 4:]
    xy_pred, wh_pred, conf_pred = pred[..., :2], pred[..., 2:4], pred[..., 4:]
    # If confidence_true is 0.0, there is no object in this grid cell.
    no_object = conf_true == 0.0
    # Computes box placement errors
    xy_error = ops.square(xy_true - xy_pred)
    wh_error = ops.square(signed_sqrt(wh_true) - signed_sqrt(wh_pred))
    # Computes confidence error
    iou = intersection_over_union(true, pred)
    conf_target = ops.where(no_object, 0.0, ops.expand_dims(iou, -1))
    conf_error = ops.square(conf_target - conf_pred)
    # Concatenates the errors weith scaling hacks
    error = ops.concatenate(
        (
            ops.where(no_object, 0.0, xy_error * 5.0),
            ops.where(no_object, 0.0, wh_error * 5.0),
            ops.where(no_object, conf_error * 0.5, conf_error),
        ),
        axis=-1,
    )
    # Returns one loss value per sample; Keras will sum over the batch.
    return ops.sum(error, axis=(1, 2, 3)) 
```

代码列表 12.11：定义 YOLO 边界框损失

我们终于准备好开始训练我们的 YOLO 模型了。为了使这个例子简短，我们将跳过指标。在实际应用中，你在这里会想要很多指标——例如，模型在不同置信度截止值下的准确性。

```py
model.compile(
    optimizer=keras.optimizers.Adam(2e-4),
    loss={"box": box_loss, "class": "sparse_categorical_crossentropy"},
)
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=4,
) 
```

代码列表 12.12：训练 YOLO 模型

训练在 Colab 免费 GPU 运行时需要超过一小时，而且我们的模型仍然欠训练（验证损失仍在下降！）让我们尝试可视化我们模型的输出（图 12.6）。我们将使用低置信度截止值，因为我们的模型目前不是一个很好的物体检测器。

```py
# Rebatches our dataset to get a single sample instead of 16
x, y = next(iter(val_dataset.rebatch(1)))
preds = model.predict(x)
boxes = preds["box"][0]
# Uses argmax to find the most likely label at each grid location
classes = np.argmax(preds["class"][0], axis=-1)
# Loads the image from disk to view it a full size
path = metadata[0]["path"]
draw_prediction(path, boxes, classes, cutoff=0.1) 
```

代码列表 12.13：训练 YOLO 模型

![](img/1652a078ff7fe8abde2a104c8dd8455b.png)

图 12.6：对我们样本图像的预测

我们可以看到我们的模型开始理解框位置和类别标签，尽管它仍然不够准确。让我们可视化模型预测的每一个框（图 12.7），即使那些置信度为零的框：

```py
draw_prediction(path, boxes, classes, cutoff=None) 
```

![](img/c85676d921d7491707c312d5d985e6c2.png)

图 12.7：YOLO 模型预测的每一个边界框

我们的模型学习到非常低置信度的值，因为它还没有学会在场景中一致地定位物体。为了进一步提高模型，我们应该尝试以下几种方法：

+   训练更多轮次

+   使用整个 COCO 数据集

+   数据增强（例如，平移和旋转输入图像和框）

+   改善重叠框的类别概率图

+   使用更大的输出网格在每个网格位置预测多个框

所有这些都会对模型性能产生积极影响，并让我们更接近原始 YOLO 训练配方。然而，这个例子实际上只是为了让我们对目标检测训练有一个感觉——从头开始训练一个准确的 COCO 检测模型需要大量的计算和时间。相反，为了获得一个性能更好的检测模型的感觉，让我们尝试使用一个名为 RetinaNet 的预训练目标检测模型。

## 使用预训练的 RetinaNet 检测器

RetinaNet 也是一个单阶段目标检测器，其工作原理与 YOLO 模型相同。我们模型与 RetinaNet 之间最大的概念性区别在于，RetinaNet 使用其底层的 ConvNet 的方式不同，以更好地同时处理小和大物体。

在我们的 YOLO 模型中，我们简单地取了 ConvNet 的最终输出，并使用它们来构建我们的目标检测器。这些输出特征映射到输入图像的大区域——因此，它们在寻找场景中的小物体方面不是很有效。

解决这个尺度问题的一个选项是直接使用我们 ConvNet 中早期层的输出。这将提取映射到我们输入图像小局部区域的高分辨率特征。然而，这些早期层的输出并不是非常 *语义上有意义*。它们可能映射到不同类型的简单特征，如边缘和曲线，但只有在 ConvNet 的后期层中，我们才开始构建整个物体的潜在表示。

RetinaNet 使用的解决方案被称为特征金字塔网络。从 ConvNet 基础模型得到的最终特征通过渐进的 `Conv2DTranspose` 层上采样，正如我们在上一章所看到的。但关键的是，我们还包含了 *侧向连接*，其中我们将这些上采样的特征图与原始 ConvNet 中相同大小的特征图相加。这结合了 ConvNet 末尾的语义有趣、低分辨率的特征与 ConvNet 开头的具有高分辨率、小尺度的特征。这种架构的粗略草图如图 12.8 所示。

![图片](img/a8fcc8365e8b204bd67598284de0d65e.png)

图 12.8：特征金字塔网络在不同尺度上创建了具有语义意义的特征图。

特征金字塔网络可以通过为像素足迹大小的小和大物体构建有效特征来显著提升性能。YOLO 的最新版本也使用了相同的设置。

让我们尝试使用在 COCO 数据集上训练的 RetinaNet 模型，为了使这个过程更有趣，让我们尝试一个对于模型来说是分布外的图像，即点彩画《大岛星期日下午》。

我们可以先下载图像并将其转换为 NumPy 数组：

```py
url = "https://s3.us-east-1.amazonaws.com/book.keras.io/3e/seurat.jpg"
path = keras.utils.get_file(origin=url)
image = np.array([keras.utils.load_img(path)]) 
```

接下来，让我们下载模型并进行预测。正如我们在上一章中所做的那样，我们可以使用 KerasHub 中的高级任务 API 创建一个`ObjectDetector`并使用它——包括预处理。

```py
detector = keras_hub.models.ObjectDetector.from_preset(
    "retinanet_resnet50_fpn_v2_coco",
    bounding_box_format="rel_xywh",
)
predictions = detector.predict(image) 
```

代码清单 12.14：创建 ResNet 模型

你会注意到我们传递了一个额外的参数来指定边界框格式。我们可以为大多数支持边界框的 Keras 模型和层这样做。我们传递`"rel_xywh"`以使用与 YOLO 模型相同的格式，这样我们就可以使用相同的框绘制工具。在这里，`rel`代表相对于图像大小（例如，从[0, 1]）。让我们检查我们刚刚做出的预测：

```py
>>> [(k, v.shape) for k, v in predictions.items()]
[("boxes", (1, 100, 4)),
 ("confidence", (1, 100)),
 ("labels", (1, 100)),
 ("num_detections", (1,))]
>>> predictions["boxes"][0][0]
array([0.53, 0.00, 0.81, 0.29], dtype=float32)
```

我们有四种不同的模型输出：边界框、置信度、标签和检测总数。这总体上与我们的 YOLO 模型非常相似。模型可以为每个输入模型预测总共 100 个对象。

让我们尝试使用我们的框绘制工具显示预测结果（图 12.9）。

```py
fig, ax = plt.subplots(dpi=300)
draw_image(ax, path)
num_detections = predictions["num_detections"][0]
for i in range(num_detections):
    box = predictions["boxes"][0][i]
    label = predictions["labels"][0][i]
    label_name = keras_hub.utils.coco_id_to_name(label)
    draw_box(ax, box, label_name, label_to_color(label))
plt.show() 
```

代码清单 12.15：使用 RetinaNet 进行推理

![图片](img/1a2576e827efdef8df4265325d5c885c.png)

![图 12.9](img/#figure-12-9)：RetinaNet 模型在测试图像上的预测

RetinaNet 模型能够轻松地将点彩画泛化到这种风格，尽管没有在这个输入风格上进行训练！这实际上是单阶段目标检测器的一个优点。绘画和照片在像素级别上非常不同，但在高层次上具有相似的结构。与 R-CNNs 这样的两阶段检测器相比，它们被迫独立地对输入图像的小块进行分类，当小块像素看起来与训练数据非常不同时，这会变得更加困难。单阶段检测器可以借鉴整个输入的特征，并且对新颖的测试时间输入更加鲁棒。

有了这些，你已经到达了这本书计算机视觉部分的结尾！我们从零开始训练了图像分类器、分割器和目标检测器。我们对卷积神经网络的工作原理有了很好的直觉，这是深度学习时代的第一次重大成功。我们还没有完全结束图像；你将在第十七章中再次看到它们，当我们开始生成图像输出时。

## 摘要

+   目标检测通过使用边界框在图像中识别和定位对象。这基本上是图像分割的一个较弱版本，但可以运行得更加高效。

+   目标检测主要有两种方法：

    +   基于区域的卷积神经网络（R-CNNs），这是一种两阶段模型，首先提出感兴趣区域，然后使用卷积神经网络对其进行分类。

    +   单阶段检测器（如 RetinaNet 和 YOLO），它们在单步中执行两项任务。单阶段检测器通常更快、更高效，使其适用于实时应用（例如，自动驾驶汽车）。

+   YOLO 在训练期间同时计算两个独立的输出——可能的边界框和类别概率图：

    +   每个候选边界框都与一个置信度分数配对，该分数被训练以针对预测框和真实框的**交并比**。

    +   类别概率图将图像的不同区域分类为属于不同的对象。

+   RetinaNet 通过使用特征金字塔网络（FPN）来构建这一想法，该网络结合了多个 ConvNet 层的特征以创建不同尺度的特征图，使其能够更准确地检测不同大小的对象。

### 脚注

1.  COCO 2017 检测数据集可在 [`cocodataset.org/`](https://cocodataset.org/) 探索。本章中的大多数图像都来自该数据集。[[↩]](#footnote-link-1)

1.  来自 COCO 2017 数据集的图像，[`cocodataset.org/`](https://cocodataset.org/)。图像来自 Flickr，[`farm8.staticflickr.com/7250/7520201840_3e01349e3f_z.jpg`](http://farm8.staticflickr.com/7250/7520201840_3e01349e3f_z.jpg)，CC BY 2.0 [`creativecommons.org/licenses/by/2.0/`](https://creativecommons.org/licenses/by/2.0/)。[[↩]](#footnote-link-2)

1.  Redmon 等人，“你只看一次：统一、实时目标检测”，CoRR (2015)，[`arxiv.org/abs/1506.02640`](https://arxiv.org/abs/1506.02640). [[↩]](#footnote-link-3)

1.  来自 COCO 2017 数据集的图像，[`cocodataset.org/`](https://cocodataset.org/)。图像来自 Flickr，[`farm9.staticflickr.com/8081/8387882360_5b97a233c4_z.jpg`](http://farm9.staticflickr.com/8081/8387882360_5b97a233c4_z.jpg)，CC BY 2.0 [`creativecommons.org/licenses/by/2.0/`](https://creativecommons.org/licenses/by/2.0/)。[[↩]](#footnote-link-4)
