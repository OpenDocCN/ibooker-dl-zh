# 第十一章。检测多个图像

在之前的章节中，我们看到了如何使用预训练分类器来检测图像并学习新的类别。然而，在所有这些实验中，我们总是假设我们的图像中只有一件事情要看。在现实世界中，情况并非总是如此——例如，我们可能有一张既有猫又有狗的图像。

这一章探讨了一些技术来克服这个限制。我们首先建立在一个预训练的分类器上，并修改设置，以便我们得到多个答案。然后我们看一下解决这个问题的最先进的解决方案。

这是一个活跃研究领域，最先进的算法很难在 Python 笔记本上重现，而且还要在 Keras 之上。相反，我们在本章的第二和第三个配方中使用一个开源库来演示可能性。

本章的代码可以在以下笔记本中找到：

```py
11.1 Detecting Multiple Images
```

# 11.1 使用预训练分类器检测多个图像

## 问题

如何在单个图像中找到多个图像类别？

## 解决方案

使用中间层的输出作为特征图，并在其上运行一个滑动窗口。

使用预训练的神经网络进行图像分类并不难，一旦我们设置好一切。如果图像中有多个对象要检测，我们就做得不太好：预训练网络将返回图像代表任何类别的可能性。如果它看到两个不同的对象，它可能会分割返回的分数。如果它看到一个对象但不确定它是两个类别中的一个，它也会分割分数。

一个想法是在图像上运行一个滑动窗口。我们不是将图像下采样到 224×224，而是将其下采样到 448×448，原始尺寸的两倍。然后我们将所有不同的裁剪都输入到较大图像中：

![带有两个裁剪的猫和狗](img/cat_dog_cropped.png)

让我们从较大的图像中创建裁剪：

```py
cat_dog2 = preprocess_image('data/cat_dog.jpg', target_size=(448, 448))
crops = []
for x in range(7):
    for y in range(7):
        crops.append(cat_dog2[0,
                              x * 32: x * 32 + 224,
                              y * 32: y * 32 + 224,
                              :])
crops = np.asarray(crops)
```

分类器在批处理上运行，因此我们可以以相同的方式将`crops`对象馈送到之前加载的分类器中：

```py
preds = base_model.predict(vgg16.preprocess_input(crops))
l = defaultdict(list)
for idx, pred in enumerate(vgg16.decode_predictions(preds, top=1)):
    _, label, weight = pred[0]
    l[label].append((idx, weight))
l.keys()
```

```py
dict_keys(['Norwegian_elkhound', 'Egyptian_cat', 'standard_schnauzer',
           'kuvasz', 'flat-coated_retriever', 'tabby', 'tiger_cat',
           'Labrador_retriever'])
```

分类器似乎大多认为各种瓷砖要么是猫，要么是狗，但并不确定是哪种类型。让我们看看对于给定标签具有最高值的裁剪：

```py
def best_image_for_label(l, label):
    idx = max(l[label], key=lambda t:t[1])[0]
    return deprocess_image(crops[idx], 224, 224)

showarray(best_image_for_label(crop_scores, 'Egyptian_cat'))
```

![埃及猫的最佳裁剪](img/dlcb_11in01.png)

```py
showarray(best_image_for_label(crop_scores, 'Labrador_retriever'))
```

![拉布拉多的最佳裁剪](img/dlcb_11in02.png)

这种方法有效，但相当昂贵。而且我们重复了很多工作。记住 CNN 的工作方式是通过在图像上运行卷积来进行的，这与做所有这些裁剪非常相似。此外，如果我们加载一个没有顶层的预训练网络，它可以在任何大小的图像上运行：

```py
bottom_model = vgg16.VGG16(weights='imagenet', include_top=False)
```

```py
(1, 14, 14, 512)
```

网络的顶层期望输入为 7×7×512。我们可以根据已加载的网络重新创建网络的顶层，并复制权重：

```py
def top_model(base_model):
    inputs = Input(shape=(7, 7, 512), name='input')
    flatten = Flatten(name='flatten')(inputs)
    fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
    fc2 = Dense(4096, activation='relu', name='fc2')(fc1)
    predictions = Dense(1000, activation='softmax',
                        name='predictions')(fc2)
    model = Model(inputs,predictions, name='top_model')
    for layer in model.layers:
        if layer.name != 'input':
            print(layer.name)
            layer.set_weights(
                base_model.get_layer(layer.name).get_weights())
    return model

model = top_model(base_model)
```

现在我们可以根据底部模型的输出进行裁剪，并将其输入到顶部模型中，这意味着我们只对原始图像的 4 倍像素运行底部模型，而不是像之前那样运行 64 倍。首先，让我们通过底部模型运行图像：

```py
bottom_out = bottom_model.predict(cat_dog2)
```

现在，我们将创建输出的裁剪：

```py
vec_crops = []
for x in range(7):
    for y in range(7):
        vec_crops.append(bottom_out[0, x: x + 7, y: y + 7, :])
vec_crops = np.asarray(vec_crops)
```

然后运行顶部分类器：

```py
crop_pred = top_model.predict(vec_crops)
l = defaultdict(list)
for idx, pred in enumerate(vgg16.decode_predictions(crop_pred, top=1)):
    _, label, weight = pred[0]
    l[label].append((idx, weight))
l.keys()
```

这应该给我们带来与以前相同的结果，但速度更快！

## 讨论

在这个配方中，我们利用了神经网络的较低层具有关于网络看到的空间信息的事实，尽管这些信息在预测时被丢弃。这个技巧基于围绕 Faster RCNN（见下一个配方）进行的一些工作，但不需要昂贵的训练步骤。

我们的预训练分类器在固定大小的图像（在本例中为 224×224 像素）上运行，这在这里有些限制。输出区域始终具有相同的大小，我们必须决定将原始图像分成多少个单元格。然而，它确实很好地找到有趣的子图像，并且易于部署。

Faster RNN 本身并没有相同的缺点，但训练成本更高。我们将在下一个示例中看一下这一点。

# 11.2 使用 Faster RCNN 进行目标检测

## 问题

如何在图像中找到多个紧密边界框的对象？

## 解决方案

使用（预训练的）Faster RCNN 网络。

Faster RCNN 是一种神经网络解决方案，用于在图像中找到对象的边界框。不幸的是，该算法过于复杂，无法在 Python 笔记本中轻松复制；相反，我们将依赖于一个开源实现，并将该代码更多地视为黑盒。让我们从 GitHub 克隆它：

```py
git clone https://github.com/yhenon/keras-frcnn.git
```

在我们从*requirements.txt*安装了依赖项之后，我们可以训练网络。我们可以使用我们自己的数据进行训练，也可以使用[Visual Object Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)的标准数据集进行训练。后者包含许多带有边界框和 20 个类别的图像。

在我们下载了 VOC 2007/2012 数据集并解压缩后，我们可以开始训练：

```py
python train_frcnn.py -p <*`downloaded``-``data``-``set`*>

```

###### 注意

这需要相当长的时间——在一台强大的 GPU 上大约需要一天，而在仅使用 CPU 上则需要更长时间。如果您希望跳过此步骤，可以在[*https://storage.googleapis.com/deep-learning-cookbook/model_frcnn.hdf5*](https://storage.googleapis.com/deep-learning-cookbook/model_frcnn.hdf5)上找到一个预训练的网络。

训练脚本会在每次看到改进时保存模型的权重。为了测试目的实例化模型有些复杂：

```py
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(c.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

shared_layers = nn.nn_base(img_input, trainable=True)

num_anchors = len(c.anchor_box_scales) * len(c.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input,
                           roi_input,
                           c.num_rois,
                           nb_classes=len(c.class_mapping),
                           trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)
```

现在我们有两个模型，一个能够建议可能有一些有趣内容的区域，另一个能够告诉我们那是什么。让我们加载模型的权重并编译：

```py
model_rpn.load_weights('data/model_frcnn.hdf5', by_name=True)
model_classifier.load_weights('data/model_frcnn.hdf5', by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')
```

现在让我们将图像输入到区域建议模型中。我们将重塑输出，以便更容易运行下一步。之后，`r2`是一个三维结构，最后一个维度保存了预测：

```py
img_vec, ratio = format_img(cv2.imread('data/cat_dog.jpg'), c)
y1, y2, f = model_rpn.predict(img_vec)
r = keras_frcnn.roi_helpers.rpn_to_roi(y1, y2, c, K.image_dim_ordering(),
                                       overlap_thresh=0.7)
roi_count = R.shape[0] // c.num_rois
r2 = np.zeros((roi_count * c.num_rois, r.shape[1]))
r2 = r[:r2.shape[0],:r2.shape[1]]
r2 = np.reshape(r2, (roi_count, c.num_rois, r.shape[1]))
```

图像分类器在一维批次上运行，因此我们必须逐个输入`r2`的两个维度。`p_cls`将包含检测到的类别，`p_regr`将包含框的微调信息：

```py
p_cls = []
p_regr = []
for i in range(r2.shape[0]):
    pred = model_classifier_only.predict([F, r2[i: i + 1]])
    p_cls.append(pred[0][0])
    p_regr.append(pred[1][0])
```

将三个数组组合在一起以获得实际的框、标签和确定性是通过循环遍历两个维度来实现的：

```py
boxes = []
w, h, _ = r2.shape
for x in range(w):
    for y in range(h):
        cls_idx = np.argmax(p_cls[x][y])
        if cls_idx == len(idx_to_class) - 1:
            continue
        reg = p_regr[x, y, 4 * cls_idx:4 * (cls_idx + 1)]
        params = list(r2[x][y])
        params += list(reg / c.classifier_regr_std)
        box = keras_frcnn.roi_helpers.apply_regr(*params)
        box = list(map(lambda i: i * c.rpn_stride, box))
        boxes.append((idx_to_class[cls_idx], p_cls[x][y][cls_idx], box))
```

现在列表`boxes`中包含了检测到的猫和狗。有许多重叠的矩形，可以解析成彼此。

## 讨论

Faster RCNN 算法是 Fast RCNN 算法的演变，而 Fast RCNN 算法又是原始 RCNN 的改进。所有这些算法工作方式类似；一个区域提议者提出可能包含有趣图像的矩形，然后图像分类器检测那里是否有什么。这种方法与我们在上一个示例中所做的并没有太大不同，那里我们的区域提议者只是生成了图像的 64 个子裁剪。

Jian Sun 提出了 Faster RCNN，他聪明地观察到在前一个示例中使用的产生特征图的 CNN 也可以成为区域提议的良好来源。因此，Faster RCNN 不是将区域提议问题单独处理，而是在相同的特征图上并行训练区域提议，该特征图也用于图像分类。

您可以在 Athelas 博客文章["CNN 在图像分割中的简要历史：从 R-CNN 到 Mask-CNN"](https://bit.ly/2oUCh88)中了解 RCNN 演变为 Faster RCNN 以及这些算法的工作原理。

# 11.3 在我们自己的图像上运行 Faster RCNN

## 问题

您想要训练一个 Faster RCNN 模型，但不想从头开始。

## 解决方案

从预训练模型开始训练。

从头开始训练需要大量标记数据。VOC 数据集包含 20 个类别的 20000 多个标记图像。那么如果我们没有那么多标记数据怎么办？我们可以使用我们在第九章中首次遇到的迁移学习技巧。

如果重新启动训练脚本，它已经加载了权重；我们需要做的是将来自在 VOC 数据集上训练的网络的权重转换为我们自己的权重。在之前的示例中，我们构建了一个双网络并加载了权重。只要我们的新任务类似于 VOC 分类任务，我们只需要改变类别数量，写回权重，然后开始训练。

最简单的方法是让训练脚本运行足够长的时间，以便它写入其配置文件，然后使用该配置文件和先前加载的模型来获取这些权重。对于训练我们自己的数据，最好使用 GitHub 上描述的逗号分隔格式：

```py
filepath,x1,y1,x2,y2,class_name

```

在这里，`filepath`应该是图像的完整路径，`x1`、`y1`、`x2`和`y2`形成了该图像上的像素边界框。我们现在可以用以下方式训练模型：

```py
python train_frcnn.py -o simple -p my_data.txt \
       --config_filename=newconfig.pickle
```

现在，在我们像以前一样加载了预训练模型之后，我们可以加载新的配置文件：

```py
new_config = pickle.load(open('data/config.pickle', 'rb'))
Now construct the model for training and load the weights:

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))
shared_layers = nn.nn_base(img_input, trainable=True)

num_anchors = len(c.anchor_box_scales) * len(c.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, c.num_rois,
                           len(c.class_mapping), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

model_rpn.load_weights('data/model_frcnn.hdf5', by_name=True)
model_classifier.load_weights('data/model_frcnn.hdf5', by_name=True)
```

我们可以看到训练模型只取决于分类器对象的类别数量。因此，我们想要重建分类器对象和任何依赖它的对象，然后保存权重。这样我们就基于旧权重构建了新模型。如果我们查看构建分类器的代码，我们会发现它完全依赖于倒数第三层。因此，让我们复制该代码，但使用`new_config`运行：

```py
new_nb_classes = len(new_config.class_mapping)
out = model_classifier_only.layers[-3].output
new_out_class = TimeDistributed(Dense(new_nb_classes,
                    activation='softmax', kernel_initializer='zero'),
                    name='dense_class_{}'.format(new_nb_classes))(out)
new_out_regr = TimeDistributed(Dense(4 * (new_nb_classes-1),
                    activation='linear', kernel_initializer='zero'),
                    name='dense_regress_{}'.format(new_nb_classes))(out)
new_classifer =  [new_out_class, new_out_regr]
```

有了新的分类器，我们可以像以前一样构建模型并保存权重。这些权重将保留模型之前学到的内容，但对于特定于新训练任务的分类器部分将为零：

```py
new_model_classifier = Model([img_input, roi_input], classifier)
new_model_rpn = Model(img_input, rpn[:2])
new_model_all = Model([img_input, roi_input], rpn[:2] + classifier)
new_model_all.save_weights('data/model_frcnn_new.hdf5')
```

我们现在可以继续训练：

```py
python train_frcnn.py -o simple -p my_data.txt \
       --config_filename=newconfig.pickle \
       --input_weight_path=data/model_frcnn_new.hdf5
```

## 讨论

大多数迁移学习的例子都基于图像识别网络。这部分是因为预训练网络易于获取，而且获取带标签图像的训练集也很简单。在这个示例中，我们看到我们也可以在其他情况下应用这种技术。我们只需要一个预训练网络和对网络构建方式的了解。通过加载网络权重，修改网络以适应新数据集，并再次保存权重，我们可以显著提高学习速度。

即使在没有预训练网络可用的情况下，但有大量公共训练数据可用且我们自己的数据集很小的情况下，首先在公共数据集上进行训练，然后将学习迁移到我们自己的数据集可能是有意义的。对于本示例中讨论的边界框情况，这很容易成为可能。

###### 提示

如果您自己的数据集很小，可能有必要像我们在第 9.7 节中所做的那样，将网络的一部分设置为不可训练。
