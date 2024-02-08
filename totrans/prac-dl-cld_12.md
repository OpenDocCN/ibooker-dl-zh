# 第12章。在iOS上使用Core ML和Create ML的Not Hotdog

“我是富人，”新晋百万富翁简阳在接受彭博社采访时说（[图12-1]（part0014.html＃jian_yang_being_interviewed_by_bloomberg））。他做了什么？他创建了Not Hotdog应用程序（[图12-2]（part0014.html＃the_not_hotdog_app_in_action_left_parent）），让世界“变得更美好”。

！[简阳在Periscope收购他的“Not Hotdog”技术后接受彭博新闻采访（来源：HBO的硅谷）]（../images/00307.jpeg）

###### 图12-1。简阳在Periscope收购他的“Not Hotdog”技术后接受彭博新闻采访（图片来源：来自HBO的硅谷）

对于我们中的少数人可能会感到困惑（包括本书的三分之一作者），我们正在参考HBO的*硅谷*，这是一部节目，其中一个角色被要求制作SeeFood - “食物的Shazam”。它旨在对食物图片进行分类，并提供食谱和营养信息。令人发笑的是，该应用程序最终只能识别热狗。其他任何东西都将被分类为“Not Hotdog”。

我们选择引用这个虚构的应用程序有几个原因。它在流行文化中非常重要，许多人可以轻松地与之联系。它是一个典范：足够简单，但又足够强大，可以在现实世界的应用中看到深度学习的魔力。它也非常容易地可以推广到识别多个类别的物品。

！[Not Hotdog应用程序的操作（图片来源：苹果应用商店中的Not Hotdog应用程序列表）]（../images/00273.jpeg）

###### 图12-2。Not Hotdog应用程序的操作（图片来源：苹果应用商店中的Not Hotdog应用程序）

在本章中，我们通过几种不同的方法来构建一个Not Hotdog克隆。整个端到端过程的一般概述如下：

1.  收集相关数据。

1.  训练模型。

1.  转换为Core ML。

1.  构建iOS应用程序。

[表12-1]（part0014.html＃various_approaches_to_get_a_model_ready）介绍了步骤1到3的不同选项。在本章后面，我们将深入研究每个选项。

表12-1。从头开始为移动部署准备模型的各种方法

| **数据收集** | **训练机制** | **模型转换** |
| --- | --- | --- |

|

+   查找或收集数据集

+   Fatkun Chrome浏览器扩展

+   使用Bing Image Search API的网络爬虫

|

+   基于Web的GUI：CustomVision.ai，IBM Watson，Clarifai，Google AutoML

+   创建ML

+   使用任何选择的框架，如Keras进行微调

|

+   创建ML，CustomVision.ai和其他GUI工具生成.*mlmodel*。

+   对于Keras，请使用Core ML工具。

+   对于TensorFlow训练的模型，请使用`tf-coreml`。

|

让我们开始吧！

# 收集数据

要开始使用深度学习解决任何计算机视觉任务，我们首先需要有一组图像数据集进行训练。在本节中，我们使用三种不同的方法来收集相关类别的图像，所需时间逐渐增加，从几分钟到几天。

## 方法1：查找或收集数据集

解决问题的最快方法是手头上有现有的数据集。有大量公开可用的数据集，其中一个类别或子类别可能与我们的任务相关。例如，ETH Zurich的Food-101（[*https://www.vision.ee.ethz.ch/datasets_extra/food-101/*]（https://oreil.ly/dkS6X））包含一个热狗类别。另外，ImageNet包含1,257张热狗的图像。我们可以使用剩余类别的随机样本作为“Not Hotdog”。

要从特定类别下载图像，可以使用[ImageNet-Utils]（https://oreil.ly/ftyOU）工具：

1.  在ImageNet网站上搜索相关类别；例如，“热狗”。

1.  注意URL中的`wnid`（WordNet ID）：[*http://image-net.org/synset?wnid=n07697537*]（http://image-net.org/synset?wnid=n07697537）。

1.  克隆ImageNet-Utils存储库：

    ```py
    $ git clone --recursive
    	https://github.com/tzutalin/ImageNet_Utils.git
    ```

1.  通过指定`wnid`下载特定类别的图像：

    ```py
    $ ./downloadutils.py --downloadImages --wnid n07697537
    ```

如果我们找不到数据集，我们也可以通过使用智能手机自己拍照来构建自己的数据集。我们拍摄的照片必须代表我们的应用程序在现实世界中的使用方式。另外，通过向朋友、家人和同事提问，可以生成一个多样化的数据集。大公司使用的另一种方法是雇佣承包商负责收集图像。例如，Google Allo发布了一个功能，可以将自拍照片转换为贴纸。为了构建这个功能，他们雇佣了一组艺术家拍摄照片并创建相应的贴纸，以便他们可以对其进行训练。

###### 注意

确保检查数据集中图像发布的许可证。最好使用根据宽松许可证发布的图像，如知识共享许可证。

## 方法2：Fatkun Chrome浏览器扩展

有几个浏览器扩展可以让我们从网站批量下载多个图像。一个例子是[Fatkun批量下载图像](https://oreil.ly/7T4JU)，这是Chrome浏览器上可用的浏览器扩展。

我们可以通过以下简短快速的步骤准备好整个数据集。

1.  将扩展添加到我们的浏览器。

1.  在Google或必应图像搜索中搜索关键字。

1.  选择搜索设置中适合图像许可证的过滤器。

1.  页面重新加载后，多次向下滚动几次，以确保页面上加载更多缩略图。

1.  打开扩展并选择“此标签”选项，如[图12-3](part0014.html#bing_search_results_for_quotation_markho)所示。

    ![“热狗”的必应搜索结果](../images/00312.jpeg)

    ###### 图12-3\. “热狗”的必应搜索结果

1.  请注意，默认情况下选择了所有缩略图。在屏幕顶部，单击切换按钮以取消选择所有缩略图并仅选择我们需要的缩略图。我们可以设置最小宽度和高度为224（大多数预训练模型将224x224作为输入尺寸）。

    ![通过Fatkun扩展选择图像](../images/00195.jpeg)

    ###### 图12-4\. 通过Fatkun扩展选择图像

1.  在右上角，单击“保存图像”以将所有选定的缩略图下载到我们的计算机上。

###### 注意

请注意，屏幕截图中显示的图像是标志性图像（即主要对象直接聚焦在干净的背景上）。只使用这样的图像在我们的模型中可能导致其无法很好地推广到真实世界的图像。例如，在具有干净白色背景的图像中（如在电子商务网站上），神经网络可能会错误地学习到白色背景等于热狗。因此，在进行数据收集时，请确保您的训练图像代表真实世界。

###### 提示

对于“非热狗”的负类别，我们希望收集大量可用的随机图像。此外，收集看起来类似于热狗但实际上不是的物品；例如，潜水艇三明治、面包、盘子、汉堡等等。

热狗的常见共现物品缺失，如盘子上的食物、纸巾、番茄酱瓶或包装袋，可能会误导模型认为那些是真正的热狗。因此，请确保将这些添加到负类别中。

当您安装像Fatkun这样的浏览器扩展时，它将请求权限读取和修改我们访问的所有网站上的数据。当您不使用它下载图像时，最好禁用该扩展。

## 方法3：使用必应图像搜索API的网络爬虫

对于构建更大的数据集，使用Fatkun收集图像可能是一个繁琐的过程。此外，Fatkun浏览器扩展返回的图像是缩略图，而不是原始大小的图像。对于大规模图像集合，我们可以使用搜索图像的API，比如必应图像搜索API，其中我们可以建立一定的约束，如关键字、图像大小和许可证。谷歌曾经有图像搜索API，但在2011年停止了。

必应的搜索API是其基于AI的图像理解和传统信息检索方法（即使用来自“alt-text”、“metadata”和“caption”等字段的标签）的融合。由于这些字段的误导性标签，我们往往会得到一些不相关的图像。因此，我们希望手动解析收集的图像，以确保它们实际上与我们的任务相关。

当我们有一个非常庞大的图像数据集时，手动筛选出所有质量差的训练示例可能是一项艰巨的任务。以迭代的方式逐步改进训练数据集的质量会更容易。以下是高层次的步骤：

1.  通过手动审查少量图像创建训练数据的子集。例如，如果我们的原始数据集中有50k张图像，我们可能希望手动选择大约500个良好的训练示例进行第一次迭代。

1.  对这500张图像进行训练。

1.  在剩余的图像上测试模型，并为每个图像获取置信度值。

1.  在置信度值最低的图像中（即经常错误预测的图像），审查一个子集（比如500个）并丢弃不相关的图像。将此子集中剩余的图像添加到训练集中。

1.  重复步骤1到4几次，直到我们对模型的质量感到满意。

这是一种半监督学习形式。

###### 提示

您可以通过将被丢弃的图像重新用作负训练示例来进一步提高模型的准确性。

###### 注意

对于一组没有标签的大量图像，您可能希望使用其他定义文本的共现作为标签；例如，标签、表情符号、alt文本等。

Facebook利用帖子文本中的标签构建了一个包含35亿张图像的数据集，将它们作为弱标签进行训练，并最终在ImageNet数据集上进行微调。这个模型比最先进的结果提高了2%（85%的前1%准确率）。

现在我们已经收集了图像数据集，让我们最终开始训练它们。

# 训练我们的模型

广义上说，有三种简单的训练方式，我们之前已经讨论过。在这里，我们提供了几种不同方法的简要概述。

## 方法1：使用基于Web UI的工具

如[第8章](part0010.html#9H5K3-13fa565533764549a6f0ab7f11eed62b)中讨论的，有几种工具可以通过提供带标签的图像并使用Web UI进行训练来构建自定义模型。微软的CustomVision.ai、Google AutoML、IBM Watson Visual Recognition、Clarifai和Baidu EZDL是几个例子。这些方法无需编码，许多提供简单的拖放GUI进行训练。

让我们看看如何在不到五分钟的时间内使用CustomVision.ai创建一个适合移动设备的模型：

1.  访问[*http://customvision.ai*](http://customvision.ai)，并创建一个新项目。因为我们想要将训练好的模型导出到手机上，所以选择一个紧凑型模型类型。由于我们的领域与食品相关，选择“食品（紧凑型）”，如[图12-5](part0014.html#define_a_new_project_on_customvisiondota)所示。

    ![在CustomVision.ai上定义一个新项目](../images/00026.jpeg)

    ###### 图12-5\. 在CustomVision.ai上定义一个新项目

1.  上传图像并分配标签（标签），如[图12-6](part0014.html#uploading_images_on_the_customvisiondota)所示。每个标签至少上传30张图像。

    ![在CustomVision.ai仪表板上上传图像。请注意，标签已填充为Hotdog和Not Hotdog](../images/00110.jpeg)

    ###### 图12-6. 在CustomVision.ai仪表板上上传图片。请注意，标签已填充为Hotdog和Not Hotdog

1.  点击训练按钮。一个对话框会打开，如[图12-7](part0014.html#options_for_training_type)所示。快速训练主要训练最后几层，而高级训练可能会调整整个网络，从而获得更高的准确性（显然需要更多时间和金钱）。对于大多数情况，快速训练选项应该足够了。

    ![训练类型选项](../images/00072.jpeg)

    ###### 图12-7. 训练类型选项

1.  不到一分钟，一个屏幕应该出现，显示每个类别新训练模型的精度和召回率，如[图12-8](part0014.html#precisioncomma_recallcomma_and_average_p)所示。（这应该让你想起我们之前在书中讨论过的精度和召回率。）

    ![新训练模型的精度、召回率和平均精度](../images/00031.jpeg)

    ###### 图12-8. 新训练模型的精度、召回率和平均精度

1.  调整概率阈值，看看它如何改变模型的性能。默认的90%阈值可以取得相当不错的结果。阈值越高，模型变得越精确，但召回率会降低。

1.  点击导出按钮，选择iOS平台（[图12-9](part0014.html#the_model_exporter_options_in_customvisi)）。在内部，CustomVision.ai将模型转换为Core ML（如果要导出到Android，则转换为TensorFlow Lite）。

    ![CustomVision.ai中的模型导出选项](../images/00318.jpeg)

    ###### 图12-9. CustomVision.ai中的模型导出选项

我们完成了，而且完全不需要编写一行代码！现在让我们看看更方便的无编码训练方式。

## 方法2：使用Create ML

2018年，苹果推出了Create ML，作为苹果生态系统内开发者训练计算机视觉模型的一种方式。开发者可以打开一个*playground*，写几行Swift代码来训练图像分类器。或者，他们可以使用`CreateMLUI`导入在playground内显示有限的GUI训练体验。这是一个让Swift开发者能够部署Core ML模型而无需太多机器学习经验的好方法。

一年后，在2019年的苹果全球开发者大会（WWDC）上，苹果通过在macOS Catalina（10.15）上宣布独立的Create ML应用，进一步降低了门槛。它提供了一个易于使用的GUI，可以训练神经网络而无需编写任何代码。训练神经网络只需将文件拖放到此UI中。除了支持图像分类器外，他们还宣布支持目标检测器、NLP、声音分类、活动分类（根据来自Apple Watch和iPhone的运动传感器数据对活动进行分类）以及表格数据（包括推荐系统）。

而且，速度很快！模型可以在不到一分钟内训练完成。这是因为它使用迁移学习，所以不需要训练网络中的所有层。它还支持各种数据增强，如旋转、模糊、噪声等，你只需要点击复选框。

在Create ML出现之前，通常认为任何想在合理时间内训练一个严肃的神经网络的人都必须拥有NVIDIA GPU。Create ML利用了内置的Intel和/或Radeon显卡，使MacBook上的训练速度更快，而无需购买额外的硬件。Create ML允许我们同时训练多个模型，来自不同的数据源。它特别受益于强大的硬件，如Mac Pro甚至外部GPU（eGPU）。

使用Create ML的一个主要动机是其输出的模型大小。完整模型可以分解为基础模型（生成特征）和更轻的特定任务分类层。苹果将基础模型内置到其每个操作系统中。因此，Create ML只需要输出特定任务的分类器。这些模型有多小？仅几千字节（与MobileNet模型的15 MB相比，后者已经相当小了）。在越来越多的应用开发人员开始将深度学习整合到其应用中的今天，这一点至关重要。同一神经网络不需要在多个应用程序中不必要地复制，消耗宝贵的存储空间。

简而言之，Create ML易于使用，速度快，体积小。听起来太好了。事实证明，完全垂直集成的反面是开发人员被绑定到苹果生态系统中。Create ML只导出*.mlmodel*文件，这些文件只能在iOS、iPadOS、macOS、tvOS和watchOS等苹果操作系统上使用。遗憾的是，Create ML尚未实现与Android的集成。

在本节中，我们使用Create ML构建Not Hotdog分类器：

1.  打开Create ML应用程序，点击新建文档，从可用的几个选项中选择图像分类器模板（包括声音、活动、文本、表格），如[图12-10](part0014.html#choosing_a_template_for_a_new_project)所示。请注意，这仅适用于Xcode 11（或更高版本），macOS 10.15（或更高版本）。

    ![选择新项目的模板](../images/00277.jpeg)

    ###### 图12-10。选择新项目的模板

1.  在下一个屏幕中，输入项目名称，然后选择完成。

1.  我们需要将数据分类到正确的目录结构中。如[图12-11](part0014.html#train_and_test_data_in_separate_director)所示，我们将图像放在以其标签名称命名的目录中。将训练和测试数据分别放在相应的目录中是有用的。

    ![将训练和测试数据放在不同的目录中](../images/00235.jpeg)

    ###### 图12-11。将训练和测试数据放在不同的目录中

1.  将UI指向训练和测试数据目录，如[图12-12](part0014.html#training_interface_in_create_ml)所示。

    ![Create ML中的训练界面](../images/00200.jpeg)

    ###### 图12-12。Create ML中的训练界面

1.  在选择训练和测试数据目录后，[图12-12](part0014.html#training_interface_in_create_ml)显示了UI。请注意，验证数据是由Create ML自动选择的。此外，请注意可用的增强选项。在这一点上，我们可以点击播放按钮（右向三角形；参见[图12-13](part0014.html#create_ml_screen_that_opens_after_loadin)）开始训练过程。

    ![加载训练和测试数据后打开的Create ML屏幕](../images/00164.jpeg)

    ###### 图12-13。加载训练和测试数据后打开的Create ML屏幕

    ###### 注意

    当您进行实验时，您会很快注意到每个添加的增强都会使训练变慢。为了设定一个快速的基准性能指标，我们应该避免在第一次运行中使用增强。随后，我们可以尝试添加更多增强来评估它们对模型质量的影响。

1.  当训练完成时，我们可以看到模型在训练数据、（自动选择的）验证数据和测试数据上的表现，如[图12-14](part0014.html#the_create_ml_screen_after_training_comp)所示。在屏幕底部，我们还可以看到训练过程花费的时间以及最终模型的大小。在不到两分钟内达到97%的测试准确率。而且输出只有17 KB。相当不错。

    ![训练完成后的Create ML屏幕](../images/00113.jpeg)

    ###### 图12-14。训练完成后的Create ML屏幕

1.  我们现在非常接近了，我们只需要导出最终模型。将输出按钮（在[图12-14](part0014.html#the_create_ml_screen_after_training_comp)中突出显示）拖到桌面上创建*.mlmodel*文件。

1.  我们可以双击新导出的*.mlmodel*文件，检查输入和输出层，以及通过将图像拖放到其中来测试模型，如[图12-15](part0014.html#the_model_inspector_ui_within_xcode)所示。

    ![Xcode中的模型检查器UI](../images/00076.jpeg)

    ###### 图12-15\. Xcode中的模型检查器UI

该模型现在已准备好插入到任何苹果设备的应用程序中。

###### 注意

Create ML使用迁移学习，仅训练最后几层。根据您的用例，苹果提供的底层模型可能不足以进行高质量的预测。这是因为您无法训练模型的早期层，从而限制了模型可以调整的潜力。对于大多数日常问题，这不应该是一个问题。但是，对于非常特定领域的应用，如X射线，或者外观非常相似的对象，细微的细节很重要（如区分货币票据），训练完整的CNN将是一个更好的方法。我们将在下一节中探讨如何做到这一点。

## 方法3：使用Keras进行微调

到目前为止，我们已经成为使用Keras的专家。如果我们愿意进行实验并愿意花更多时间训练模型，这个选项可以让我们获得更高的准确性。我们可以重用[第3章](part0005.html#4OIQ3-13fa565533764549a6f0ab7f11eed62b)中的代码，并修改参数，如目录和文件名，批量大小和图像数量。您可以在书的GitHub网站上找到代码（请参见[*http://PracticalDeepLearning.ai*](http://PracticalDeepLearning.ai)），位于*code/chapter-12/1-keras-custom-classifier-with-transfer-learning.ipynb*。

模型训练应该需要几分钟才能完成，具体取决于硬件，在训练结束时，我们应该在磁盘上准备好一个*NotHotDog.h5*文件。

# 使用Core ML工具进行模型转换

如[第11章](part0013.html#CCNA3-13fa565533764549a6f0ab7f11eed62b)中所讨论的，有几种将我们的模型转换为Core ML格式的方法。

从CustomVision.ai生成的模型直接以Core ML格式可用，因此无需转换。对于在Keras中训练的模型，Core ML工具可以帮助进行转换。请注意，因为我们使用了一个使用名为`relu6`的自定义层的MobileNet模型，我们需要导入`CustomObjectScope`：

```py
from tensorflow.keras.models import load_model
from tensorflow.keras.utils.generic_utils import CustomObjectScope
import tensorflow.keras

with CustomObjectScope({'relu6':
tensorflow.keras.applications.mobilenet.relu6,'DepthwiseConv2D':
tensorflow.keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('NotHotDog-model.h5')

`import` coremltools
coreml_model `=` coremltools`.`converters`.`keras`.`convert(model)
coreml_model`.`save('NotHotDog.mlmodel')
```

现在我们有一个准备好的Core ML模型，我们只需要构建应用程序。

# 构建iOS应用程序

我们可以使用[第11章](part0013.html#CCNA3-13fa565533764549a6f0ab7f11eed62b)中的代码，只需用新生成的模型文件替换*.mlmodel*，如[图12-16](part0014.html#loading_the_dotmlmodel_into_xcode)所示。

![将.mlmodel加载到Xcode中](../images/00035.jpeg)

###### 图12-16\. 将`.mlmodel`加载到Xcode中

现在，编译并运行应用程序，完成了！[图12-17](part0014.html#our_app_identifying_the_hot_dog)展示了令人惊叹的结果。

![我们的应用程序识别热狗](../images/00323.jpeg)

###### 图12-17\. 我们的应用程序识别热狗

# 进一步探索

我们可以使这个应用程序更有趣吗？我们可以通过在下一章中涵盖的Food-101数据集中训练所有类别来构建一个真正的“Shazam for food”。此外，我们可以改进UI，与我们当前应用程序显示的基本百分比相比。为了使其像“Not Hotdog”一样病毒传播，提供一种将分类分享到社交媒体平台的方式。

# 总结

在这一章中，我们通过一个端到端的流程，收集数据、训练和转换模型，并在iOS设备上实际应用中使用它。对于流程的每一步，我们探索了一些不同复杂程度的选项。而且，我们将前几章涵盖的概念放在了一个真实应用的背景下。

现在，就像建阳一样，去赚取你的百万美元吧！
