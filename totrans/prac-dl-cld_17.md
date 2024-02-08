# 第16章。使用Keras进行端到端深度学习模拟自动驾驶汽车

由客座作者Aditya Sharma和Mitchell Spryn贡献

从蝙蝠车到骑士骑手，从机器人出租车到自动送披萨，自动驾驶汽车已经吸引了现代流行文化和主流媒体的想象力。为什么不呢？科幻概念何时会变为现实？更重要的是，自动驾驶汽车承诺解决当今城市社会面临的一些关键挑战：道路事故、污染水平、城市越来越拥挤、在交通中浪费的生产力小时数等等。

毫无疑问，一个完整的自动驾驶系统构建起来相当复杂，不可能在一章中解决。就像一个洋葱一样，任何复杂的问题都包含可以剥离的层。在我们的情况下，我们打算解决一个基本问题：如何转向。更好的是，我们不需要一辆真正的汽车来做到这一点。我们将训练一个神经网络在模拟环境中自主驾驶我们的汽车，使用逼真的物理和视觉效果。

但首先，简短的历史。

![SAE驾驶自动化级别（图片来源）](../images/00220.jpeg)

###### 图16-1。SAE驾驶自动化级别（图片来源）

# 自动驾驶的简史

尽管自动驾驶似乎是一个非常近期的技术发展，但这个领域的研究起源于30多年前。卡内基梅隆大学的科学家们在1984年首次尝试了这个看似不可能的壮举，当时他们开始研究后来被称为Navlab 1的项目。 “Navlab”是卡内基梅隆大学导航实验室的简称，这个名字并不花哨，但它将成为人类未来非常令人印象深刻的第一步。Navlab 1是一辆雪佛兰面包车，车上装有硬件机架，其中包括研究团队的工作站和Sun Microsystems超级计算机等设备。当时它被认为是尖端技术，是一个自包含的移动计算系统。当时远程无线通信几乎不存在，云计算甚至还不是一个概念。把整个研究团队放在一辆高度实验性的自动驾驶面包车后面可能看起来像是一个危险的想法，但Navlab的最高时速只有20英里，只在空旷的道路上行驶。这确保了工作在车上的科学家们的安全。但非常令人印象深刻的是，这辆车使用的技术为今天的自动驾驶汽车所用的技术奠定了基础。例如，Navlab 1使用视频摄像头和测距仪（激光雷达的早期版本）来观察周围环境。为了导航，他们使用了单层神经网络来根据传感器数据预测转向角度。相当不错，不是吗？

![NavLab 1的全貌（图片来源）](../images/00006.jpeg)

###### 图16-2。NavLab 1的全貌（图片来源）

# 深度学习、自动驾驶和数据问题

即使35年前科学家们就知道神经网络将在实现自动驾驶汽车方面发挥关键作用。当时，当然我们没有所需的技术（GPU、云计算、FPGA等）来训练和部署大规模的深度神经网络，以使自动驾驶成为现实。今天的自动驾驶汽车使用深度学习来执行各种任务。表16-1列出了一些示例。

表16-1。深度学习在自动驾驶汽车中的应用示例

| **任务** | **示例** |
| --- | --- |
| **感知** | 检测可驾驶区域、车道标线、交叉口、人行横道等 |
| 检测其他车辆、行人、动物、道路上的物体等 |
| 检测交通标志和信号 |
| **导航** | 根据传感器输入预测和输出转向角、油门等 |
| 进行超车、变道、掉头等操作 |
| 进行并线、出口、穿越环岛等操作 |
| 在隧道、桥梁等地方驾驶 |
| 响应交通违规者、逆行驾驶者、意外环境变化等 |
| 穿梭于交通拥堵中 |

我们知道深度学习需要数据。事实上，一个经验法则是更多的数据可以确保更好的模型性能。在之前的章节中，我们看到训练图像分类器可能需要数百万张图像。想象一下训练一辆汽车自动驾驶需要多少数据。当涉及到验证和测试所需的数据时，自动驾驶汽车的情况也非常不同。汽车不仅必须能够行驶，还必须安全行驶。因此，测试和验证模型性能是基本关键的，需要比传统深度学习问题所需的数据多得多。然而，准确预测所需数据的确切数量是困难的，估计各不相同。[一项2016年的研究](https://oreil.ly/mOUeJ)显示，一辆汽车要想达到与人类驾驶员一样的水平，需要行驶110亿英里。对于一队100辆自动驾驶汽车，每天24小时收集数据，以25英里每小时的平均速度行驶，这将需要超过500年！

当然，通过让车队在现实世界中行驶110亿英里来收集数据是非常不切实际和昂贵的。这就是为什么几乎每个在这个领域工作的人——无论是大型汽车制造商还是初创公司——都使用模拟器来收集数据和验证训练模型。例如，[Waymo](https://waymo.com/tech)（谷歌的自动驾驶汽车团队）截至2018年底在道路上行驶了近1000万英里。尽管这比地球上任何其他公司都多，但它仅占我们110亿英里的不到1%。另一方面，Waymo在模拟中行驶了70亿英里。尽管每个人都使用模拟进行验证，但有些公司构建自己的模拟工具，而其他公司则从Cognata、Applied Intuition和AVSimulation等公司许可。还有一些很棒的开源模拟工具可用：来自微软的[AirSim](https://github.com/Microsoft/AirSim)，来自英特尔的[Carla](http://carla.org/)，以及来自百度的[Apollo模拟](https://oreil.ly/rag7v)。由于有了这些工具，我们不需要连接我们汽车的CAN总线来学习让它自动驾驶的科学。

在本章中，我们使用了一个专为微软的[自动驾驶食谱](https://oreil.ly/uzOGl)定制的AirSim版本。

# 自动驾驶的“Hello, World！”：在模拟环境中操纵方向盘

在这一部分，我们实现了自动驾驶的“Hello, World！”问题。自动驾驶汽车是复杂的，有数十个传感器传输着大量数据，同时在行驶中做出多个决策。就像编程一样，对于自动驾驶汽车的“Hello, World！”，我们将要求简化为基本原理：

+   汽车始终保持在道路上。

+   对于外部传感器输入，汽车使用安装在前引擎盖上的单个摄像头的单个图像帧。不使用其他传感器（激光雷达、雷达等）。

+   基于这个单图像输入，以恒定速度行驶，汽车预测其输出的转向角。不会预测其他可能的输出（刹车、油门、换挡等）。

+   道路上没有其他车辆、行人、动物或其他任何东西，也没有相关的环境。

+   所行驶的道路是单车道，没有标线、交通标志和信号，也没有保持在道路左侧或右侧的规则。

+   道路主要在转弯方面变化（这将需要改变转向角来导航），而不是在高程方面变化（这将需要改变油门，刹车等）。

## 设置和要求

我们将在AirSim的景观地图中实现“Hello, World!”问题（[图16-4](part0019.html#the_landscape_map_in_airsim)）。AirSim是一个开源的逼真模拟平台，是一个用于训练数据收集和验证基于深度学习的自主系统模型开发的研究工具。

![AirSim中的景观地图](../images/00252.jpeg)

###### 图16-4。AirSim中的景观地图

您可以在GitHub上的[*Autonomous Driving Cookbook*](https://oreil.ly/uF5Bz)中找到本章的代码，形式为Jupyter笔记本的[端到端深度学习教程](https://oreil.ly/_IJl9)。在食谱仓库中，转到[*AirSimE2EDeepLearning/*](https://oreil.ly/YVKPp)。我们可以通过运行以下命令来实现：

```py
$ git clone https://github.com/Microsoft/AutonomousDrivingCookbook.git
$ cd AutonomousDrivingCookbook/AirSimE2EDeepLearning/
```

我们将使用Keras这个我们已经熟悉的库来实现我们的神经网络。在这个问题上，除了本书之前章节介绍的内容，我们不需要学习任何新的深度学习概念。

对于本章，我们通过在AirSim中驾驶来创建了一个数据集。数据集的未压缩大小为3.25 GB。这比我们之前使用的数据集稍大一些，但请记住，我们正在实现一个非常复杂问题的“Hello, World!”。相比之下，汽车制造商每天在道路上收集多个PB的数据是相当正常的做法。您可以通过访问[*aka.ms/AirSimTutorialDataset*](https://aka.ms/AirSimTutorialDataset)来下载数据集。

我们可以在Windows或Linux机器上运行提供的代码并训练我们的模型。我们为本章创建了AirSim的独立构建，您可以从[*aka.ms/ADCookbookAirSimPackage*](http://aka.ms/ADCookbookAirSimPackage)下载。下载后，请将模拟器包提取到您选择的位置。请注意路径，因为您以后会需要它。请注意，此构建仅在Windows上运行，但仅需要查看我们最终训练好的模型。如果您使用Linux，可以将模型文件拷贝到提供的模拟器包的Windows机器上运行。

AirSim是一个高度逼真的环境，这意味着它能够生成逼真的环境图片供模型训练。在玩高清视频游戏时，你可能遇到过类似的图形。考虑到提供的模拟器包的数据量和图形质量，GPU肯定是运行代码和模拟器的首选。

最后，运行提供的代码还需要一些额外的工具和Python依赖项。您可以在代码仓库的[README文件中的环境设置](https://oreil.ly/RzZe7)中找到这些详细信息。在高层次上，我们需要Anaconda与Python 3.5（或更高版本）、TensorFlow（作为Keras的后端运行）和h5py（用于存储和读取数据和模型文件）。设置好Anaconda环境后，我们可以通过以root或管理员身份运行*InstallPackages.py*文件来安装所需的Python依赖项。

[表16-2](part0019.html#setup_and_requirements_summary)提供了我们刚刚定义的所有要求的摘要。

表16-2。设置和要求摘要

| **项目** | **要求/链接** | **备注/评论** |
| --- | --- | --- |
| 代码仓库 | [*https://oreil.ly/_IJl9*](https://oreil.ly/_IJl9) | 可在Windows或Linux机器上运行 |
| 使用的Jupyter笔记本 | [*DataExplorationAndPreparation.ipynb*](https://oreil.ly/6yvCq)[*TrainModel.ipynb*](https://oreil.ly/rcR47)[*TestModel.ipynb*](https://oreil.ly/FE-EP) |   |
| 数据集下载 | [*aka.ms/AirSimTutorialDataset*](http://aka.ms/AirSimTutorialDataset) | 大小为3.25 GB；用于训练模型 |
| 模拟器下载 | [*aka.ms/ADCookbookAirSimPackage*](http://aka.ms/ADCookbookAirSimPackage) | 不需要用于训练模型，只用于部署和运行；仅在Windows上运行 |
| GPU | 推荐用于训练，运行模拟器所需 |   |
| 其他工具+Python依赖项 | [存储库中README文件中的环境设置部分](https://oreil.ly/RzZe7) |   |

有了这些准备工作，让我们开始吧！

# 数据探索和准备

当我们深入探索神经网络和深度学习的世界时，您会注意到，机器学习的许多魔力并不是来自神经网络是如何构建和训练的，而是来自数据科学家对问题、数据和领域的理解。自动驾驶也不例外。正如您很快将看到的，深入了解数据和我们试图解决的问题对于教会我们的汽车如何自动驾驶至关重要。

这里的所有步骤也都在Jupyter Notebook [*DataExplorationAndPreparation.ipynb*](https://oreil.ly/6aE7z)中详细说明。我们首先导入这部分练习所需的所有必要模块：

```py
import os
import random

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import Cooking #This module contains helper code. Please do not edit this file.
```

接下来，我们确保我们的程序可以找到我们解压缩的数据集的位置。我们还为其提供一个位置来存储我们处理（或“烹饪”）的数据：

```py
# << Point this to the directory containing the raw data >>
RAW_DATA_DIR = 'data_raw/'

# << Point this to the desired output directory for the cooked (.h5) data >>
COOKED_DATA_DIR = 'data_cooked/'

# The folders to search for data under RAW_DATA_DIR
# For example, the first folder searched will be RAW_DATA_DIR/normal_1
DATA_FOLDERS = ['normal_1', 'normal_2', 'normal_3', 'normal_4', 'normal_5',
	'normal_6', 'swerve_1', 'swerve_2', 'swerve_3']

# The size of the figures and illustrations used
FIGURE_SIZE = (10,10)
```

查看提供的数据集，我们会看到它包含九个文件夹：*normal_1*到*normal_6*和*swerve_1*到*swerve_3*。稍后我们会回到这些文件夹的命名。在每个文件夹中都有图像以及*.tsv*（或*.txt*）文件。让我们看看其中一个图像：

```py
sample_image_path = os.path.join(RAW_DATA_DIR, 'normal_1/images/img_0.png')
sample_image = Image.open(sample_image_path)
plt.title('Sample Image')
plt.imshow(sample_image)
plt.show()
```

我们应该看到以下输出。我们可以看到这张图像是由放置在汽车引擎盖中央的摄像头拍摄的（在[图16-5](part0019.html#plot_showing_the_contents_of_the_file_no)中底部略有可见）。我们还可以从模拟器中看到用于这个问题的景观环境。请注意，道路上没有其他车辆或物体，符合我们的要求。尽管道路看起来是多雪的，但对于我们的实验来说，雪只是视觉上的，不是实际的；也就是说，它不会对我们汽车的物理和运动产生影响。当然，在现实世界中，准确复制雪、雨等的物理特性非常重要，以便我们的模拟器可以生成高度准确的现实版本。这是一个非常复杂的任务，许多公司都投入了大量资源。幸运的是，对于我们的目的，我们可以安全地忽略所有这些，并将雪视为图像中的白色像素。

![显示文件normal_1/images/img_0.png的内容的图](../images/00211.jpeg)

###### 图16-5\. 显示文件normal_1/images/img_0.png的内容的图

让我们通过显示一个*.tsv/.txt*文件的内容来预览其余数据是什么样的：

```py
sample_tsv_path = os.path.join(RAW_DATA_DIR, 'normal_1/airsim_rec.txt')
sample_tsv = pd.read_csv(sample_tsv_path, sep='\t')
sample_tsv.head()
```

我们应该看到以下输出：

![图像](../images/00177.jpeg)

这里有很多内容，让我们来分解一下。首先，我们可以看到表中的每一行对应一个图像帧（这里我们只显示了前五帧）。因为这是我们数据收集驱动的开始，我们可以看到汽车还没有开始移动：速度和油门都是0，档位是空档。因为我们只是尝试预测问题中的转向角，我们不会预测速度、油门、刹车或档位的值。然而，这些值将作为输入数据的一部分（稍后会详细介绍）。

## 识别感兴趣区域

让我们现在回到我们的示例图像。到目前为止，我们所做的所有观察都是从人类的眼睛看到的。让我们再次看一下这幅图像，只是这一次我们换成了一辆刚开始学习自动驾驶并试图理解所看到的东西的汽车的角度（或轮胎）。

如果我是汽车，看着我的摄像头呈现给我的这张图像，我可以将其分为三个不同的部分（[图16-6](part0019.html#the_three_parts_of_the_image_as_seen_by)）。首先是下面的第三部分，看起来更或多少一致。它主要由直线（道路、车道分隔线、道路围栏等）组成，颜色一致，白色、黑色、灰色和棕色。底部还有一个奇怪的黑色弧线（这是汽车的引擎盖）。图像的上面第三部分，与下面的第三部分一样，也是一致的。主要是灰色和白色（天空和云）。最后，中间的第三部分有很多事情发生。有大的棕色和灰色形状（山），它们完全不一致。还有四个高大的绿色图形（树），它们的形状和颜色与我看到的其他任何东西都不同，所以它们一定非常重要。随着我看到更多的图像，上面的第三部分和下面的第三部分并没有太大变化，但中间的第三部分发生了很多变化。因此，我得出结论，这是我应该最关注的部分。

![自动驾驶汽车所看到的图像的三个部分](../images/00091.jpeg)

###### 图16-6. 自动驾驶汽车所看到的图像的三个部分

你看到我们汽车面临的挑战了吗？它无法知道呈现给它的图像的哪一部分是重要的。它可能会尝试将转向角度的变化与景观的变化联系起来，而不是专注于道路的转弯，这与其他事物相比非常微妙。景观的变化不仅令人困惑，而且与我们正在解决的问题毫不相关。尽管天空大部分时间都没有变化，但它也不提供与我们汽车转向相关的任何信息。最后，每张照片中捕捉到的引擎盖部分也是不相关的。

让我们通过仅关注图像的相关部分来解决这个问题。我们通过在图像中创建一个感兴趣区域（ROI）来实现这一点。正如我们可以想象的那样，没有一个硬性规则来规定我们的ROI应该是什么样子的。这真的取决于我们的用例。对于我们来说，因为摄像头处于固定位置，道路上没有高度变化，ROI是一个简单的矩形，专注于道路和道路边界。我们可以通过运行以下代码片段来查看ROI：

```py
sample_image_roi = sample_image.copy()

fillcolor=(255,0,0)
draw = ImageDraw.Draw(sample_image_roi)
points = [(1,76), (1,135), (255,135), (255,76)]
for i in range(0, len(points), 1):
    draw.line([points[i], points[(i+1)%len(points)]], fill=fillcolor, width=3)
del draw

plt.title('Image with sample ROI')
plt.imshow(sample_image_roi)
plt.show()
```

我们应该看到[图16-7](part0019.html#the_roi_for_our_car_to_focus_on_during_t)中显示的输出。

![我们的汽车在训练期间应关注的ROI](../images/00109.jpeg)

###### 图16-7. 我们的汽车在训练期间应关注的ROI

我们的模型现在将只关注道路，不会被环境中其他任何事情所困扰。这也将图像的大小减少了一半，这将使我们的神经网络训练更容易。

## 数据增强

正如前面提到的，在训练深度学习模型时，尽可能获得尽可能多的数据总是更好的。我们已经在之前的章节中看到了数据增强的重要性，以及它如何帮助我们不仅获得更多的训练数据，还可以避免过拟合。然而，先前讨论的大多数用于图像分类的数据增强技术对于我们当前的自动驾驶问题并不适用。让我们以旋转为例。随机将图像旋转20度在训练智能手机相机分类器时非常有帮助，但是我们汽车引擎盖上的摄像头固定在一个位置，永远不会看到旋转的图像（除非我们的汽车在翻转，那时我们有更大的问题）。随机移位也是如此；我们在驾驶时永远不会遇到这些。然而，对于我们当前的问题，我们可以做一些其他事情来增强数据。

仔细观察我们的图像，我们可以看到在y轴上翻转它会产生一个看起来很容易来自不同测试运行的图像（[图16-8](part0019.html#flipping_image_on_the_y-axis)）。如果我们使用图像的翻转版本以及它们的常规版本，我们可以有效地将数据集的大小加倍。以这种方式翻转图像将要求我们同时修改与之相关的转向角。因为我们的新图像是原始图像的反射，所以我们只需改变相应转向角的符号（例如，从0.212变为-0.212）。

![在y轴上翻转图像](../images/00048.jpeg)

###### 图16-8。在y轴上翻转图像

我们可以做另一件事来增加我们的数据。自动驾驶汽车不仅需要准备好应对道路上的变化，还需要准备好应对外部条件的变化，如天气、时间和光照条件。今天大多数可用的模拟器都允许我们合成这些条件。我们所有的数据都是在明亮的光照条件下收集的。我们可以在训练过程中通过调整图像亮度引入随机光照变化，而不是返回模拟器收集更多不同光照条件下的数据。[图16-9](part0019.html#reducing_image_brightness_by_40percent)展示了一个例子。

![将图像亮度降低40%](../images/00075.jpeg)

###### 图16-9。将图像亮度降低40%

## 数据集不平衡和驾驶策略

深度学习模型的好坏取决于它们所训练的数据。与人类不同，今天的人工智能无法自行思考和推理。因此，当面对完全新的情况时，它会回归到以前看到的内容，并基于其训练的数据集进行预测。此外，深度学习的统计特性使其忽略那些不经常发生的实例，将其视为异常值和离群值，即使它们具有重要意义。这被称为*数据集不平衡*，对数据科学家来说是一个常见的困扰。

想象一下，我们正在尝试训练一个分类器，该分类器查看皮肤镜图像以检测病变是良性还是恶性。假设我们的数据集有一百万张图像，其中有1万张包含恶性病变。很可能我们在这些数据上训练的分类器总是预测图像是良性的，从不预测是恶性的。这是因为深度学习算法旨在在训练过程中最小化错误（或最大化准确性）。通过将所有图像分类为良性，我们的模型达到了99%的准确性，然后结束训练，严重失败于其训练的任务：检测癌症病变。这个问题通常通过在更平衡的数据集上训练预测器来解决。

自动驾驶也无法免于数据集不平衡的问题。让我们以我们的转向角预测模型为例。从我们日常驾驶经验中，我们对转向角了解多少？在驾驶过程中，我们大多数时间都是直线行驶。如果我们的模型只是在正常驾驶数据上进行训练，它将永远不会学会如何转弯，因为在训练数据集中相对稀缺。为了解决这个问题，我们的数据集不仅需要包含正常驾驶策略的数据，还需要包含“转弯”驾驶策略的数据，数量上具有统计学意义。为了说明我们的意思，让我们回到我们的*.tsv*/*.txt*文件。在本章的前面，我们指出了数据文件夹的命名。现在应该清楚了，我们的数据集包含了使用正常驾驶策略进行的六次收集运行和使用转弯驾驶策略进行的三次收集运行的数据。

让我们将所有*.tsv*/*.txt*文件中的数据聚合到一个DataFrame中，以便更容易进行分析：

```py
full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in DATA_FOLDERS]

dataframes = []
for folder in full_path_raw_folders:
    current_dataframe = pd.read_csv(os.path.join(folder, 'airsim_rec.txt'), 
                                    sep='\t')
    current_dataframe['Folder'] = folder
    dataframes.append(current_dataframe) 
dataset = pd.concat(dataframes, axis=0)
```

让我们在散点图上绘制来自两种驾驶策略的转向角：

```py
min_index = 100
max_index = 1500
steering_angles_normal_1 = dataset[dataset['Folder'].apply(lambda v: 'normal_1'
in v)]['Steering'][min_index:max_index]
steering_angles_swerve_1 = dataset[dataset['Folder'].apply(lambda v: 'swerve_1' 
in v)]['Steering'][min_index:max_index]

plot_index = [i for i in range(min_index, max_index, 1)]
fig = plt.figure(figsize=FIGURE_SIZE)
ax1 = fig.add_subplot(111)
ax1.scatter(plot_index, steering_angles_normal_1, c='b', marker='o',
label='normal_1')
ax1.scatter(plot_index, steering_angles_swerve_1, c='r', marker='_',
label='swerve_1')
plt.legend(loc='upper left');
plt.title('Steering Angles for normal_1 and swerve_1 runs')
plt.xlabel('Time')
plt.ylabel('Steering Angle')
plt.show()
```

[图16-10](part0019.html#plot_showing_steering_angles_from_the_tw)显示了结果。

![显示两种驾驶策略的转向角的图](../images/00299.jpeg)

###### 图16-10。显示两种驾驶策略的转向角的图

让我们也绘制一下使用每种策略收集的数据点数量：

```py
dataset['Is Swerve'] = dataset.apply(lambda r: 'swerve' in r['Folder'], axis=1)
grouped = dataset.groupby(by=['Is Swerve']).size().reset_index()
grouped.columns = ['Is Swerve', 'Count']

def make_autopct(values):
    def my_autopct(percent):
        total = sum(values)
        val = int(round(percent*total/100.0))
        return '{0:.2f}% ({1:d})'.format(percent,val)
    return my_autopct

pie_labels = ['Normal', 'Swerve']
fig, ax = plt.subplots(figsize=FIGURE_SIZE)
ax.pie(grouped['Count'], labels=pie_labels, autopct =
make_autopct(grouped['Count']), explode=[0.1, 1], textprops={'weight': 'bold'},
colors=['lightblue', 'salmon'])
plt.title('Number of data points per driving strategy')
plt.show()
```

[图16-11](part0019.html#dataset_split_for_the_two_driving_strate)显示了这些结果。

![两种驾驶策略的数据集拆分](../images/00114.jpeg)

###### 图16-11。两种驾驶策略的数据集拆分

查看[图16-10](part0019.html#plot_showing_steering_angles_from_the_tw)，正如我们所预期的，我们看到正常驾驶策略产生的转向角大多数是我们在日常驾驶中观察到的，大部分是直行，偶尔转弯。相比之下，突然转向驾驶策略主要集中在急转弯，因此转向角的值较高。如[图16-11](part0019.html#dataset_split_for_the_two_driving_strate)所示，这两种策略的结合给我们一个不错的、虽然在现实生活中不太现实的分布，以75/25的比例进行训练。这也进一步巩固了模拟对自动驾驶的重要性，因为在现实生活中我们不太可能使用实际汽车进行突然转向策略来收集数据。

在结束关于数据预处理和数据集不平衡的讨论之前，让我们最后再看一下我们两种驾驶策略的转向角分布，通过在直方图上绘制它们（[图16-12](part0019.html#steering_angle_distribution_for_the_two)）： 

```py
bins = np.arange(-1, 1.05, 0.05)
normal_labels = dataset[dataset['Is Swerve'] == False]['Steering']
swerve_labels = dataset[dataset['Is Swerve'] == True]['Steering']

def steering_histogram(hist_labels, title, color):
    plt.figure(figsize=FIGURE_SIZE)
    n, b, p = plt.hist(hist_labels.as_matrix(), bins, normed=1, facecolor=color)
    plt.xlabel('Steering Angle')
    plt.ylabel('Normalized Frequency')
    plt.title(title)
    plt.show()

steering_histogram(normal_labels, 'Normal driving strategy label distribution',
'g')
steering_histogram(swerve_labels, 'Swerve driving strategy label distribution',
'r')
```

![两种驾驶策略的转向角分布](../images/00055.jpeg)

###### 图16-12。两种驾驶策略的转向角分布

正如前面所述，与正常驾驶策略相比，突然转向驾驶策略给我们提供了更广泛的角度范围，如[图16-12](part0019.html#steering_angle_distribution_for_the_two)所示。这些角度将帮助我们的神经网络在汽车偏离道路时做出适当的反应。我们数据集中的不平衡问题在一定程度上得到解决，但并非完全解决。我们仍然有很多零，跨越两种驾驶策略。为了进一步平衡我们的数据集，我们可以在训练过程中忽略其中的一部分。尽管这给我们提供了一个非常平衡的数据集，但大大减少了我们可用的数据点总数。在构建和训练神经网络时，我们需要记住这一点。

在继续之前，让我们确保我们的数据适合训练。让我们从所有文件夹中获取原始数据，将其分割为训练、测试和验证数据集，并将它们压缩成HDF5文件。HDF5格式允许我们以块的方式访问数据，而不需要一次性将整个数据集读入内存。这使其非常适合深度学习问题。它还可以与Keras无缝配合。以下代码将需要一些时间来运行。当运行完成后，我们将得到三个数据集文件：*train.h5*、*eval.h5*和*test.h5*。

```py
train_eval_test_split = [0.7, 0.2, 0.1]
full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in DATA_FOLDERS]
Cooking.cook(full_path_raw_folders, COOKED_DATA_DIR, train_eval_test_split)
```

每个数据集文件有四个部分：

图像

包含图像数据的NumPy数组。

上一个状态

包含汽车最后已知状态的NumPy数组。这是一个（转向、油门、刹车、速度）元组。

标签

包含我们希望预测的转向角的NumPy数组（在范围-1..1上进行了标准化）。

元数据

包含有关文件的元数据的NumPy数组（它们来自哪个文件夹等）。

现在我们已经准备好从数据集中获取观察和经验，并开始训练我们的模型。

# 训练我们的自动驾驶模型

本节中的所有步骤也在Jupyter Notebook [*TrainModel.ipynb*](https://oreil.ly/ESHVS)中详细说明。与之前一样，我们首先导入一些库并定义路径：

```py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda,
Input, concatenate
from tensorflow.keras.optimizers import Adam, SGD, Adamax, Nadam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,
	CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, 
from tensorflow.keras.layers import Lambda, Input, concatenate, BatchNormalization

from keras_tqdm import TQDMNotebookCallback
import json
import os
import numpy as np
import pandas as pd
from Generator import DriveDataGenerator
from Cooking import checkAndCreateDir
import h5py
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt

# << The directory containing the cooked data from the previous step >>
COOKED_DATA_DIR = 'data_cooked/'
# << The directory in which the model output will be placed >>
MODEL_OUTPUT_DIR = 'model'
```

让我们也设置好我们的数据集：

```py
train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5'), 'r')
test_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'test.h5'), 'r')

num_train_examples = train_dataset['image'].shape[0]
num_eval_examples = eval_dataset['image'].shape[0]
num_test_examples = test_dataset['image'].shape[0]

batch_size=32
```

## 驾驶数据生成器

我们在之前的章节中介绍了Keras数据生成器的概念。数据生成器会遍历数据集，并从磁盘中以块的形式读取数据。这使我们能够让CPU和GPU保持繁忙，提高吞吐量。为了实现前一节讨论的想法，我们创建了自己的Keras生成器，称为`DriveDataGenerator`。

让我们回顾一下前一节中的一些观察结果：

+   我们的模型应该只关注每个图像中的ROI。

+   我们可以通过水平翻转图像并反转转向角的符号来增强我们的数据集。

+   我们可以通过在图像中引入随机亮度变化来进一步增强数据。这将模拟不同的光照条件，并使我们的模型更加健壮。

+   我们可以随机删除一定百分比的转向角为零的数据点，以便在训练时模型看到一个平衡的数据集。

+   我们平衡后可用的数据点总数将显著减少。

让我们看看`DriveDataGenerator`如何处理这些前四项。当我们开始设计神经网络时，我们将回到最后一项。

ROI是一个简单的图像裁剪。[76,135,0,255]（在接下来的代码块中显示）是表示ROI的矩形的[x1,x2,y1,y2]值。生成器从每个图像中提取此矩形。我们可以使用参数`roi`来修改ROI。

水平翻转相对简单。在生成批次时，随机图像沿y轴翻转，并且如果参数`horizontal_flip`设置为`True`，它们的转向角值将被反转。

对于随机亮度变化，我们引入参数`brighten_range`。将此参数设置为`0.4`会随机地使任何给定批次中的图像亮度增加或减少最多40%。我们不建议将此值增加到`0.4`之上。为了计算亮度，我们将图像从RGB转换为HSV空间，将“V”坐标缩放上下，然后转换回RGB。

通过删除零来平衡数据集，我们引入了参数`zero_drop_percentage`。将此设置为`0.9`将在任何给定批次中随机删除90%的0标签数据点。

让我们使用这些参数初始化我们的生成器：

```py
data_generator = DriveDataGenerator(rescale=1./255., horizontal_flip=True,
brighten_range=0.4)

train_generator = data_generator.flow\
    (train_dataset['image'], train_dataset['previous_state'], 
    train_dataset['label'], batch_size=batch_size, zero_drop_percentage=0.95, 
    roi=[76,135,0,255])
eval_generator = data_generator.flow\
    (eval_dataset['image'], eval_dataset['previous_state'],
    eval_dataset['label'],
batch_size=batch_size, zero_drop_percentage=0.95, roi=[76,135,0,255]
```

让我们通过在相应图像上绘制标签（转向角）来可视化一些样本数据点：

```py
def draw_image_with_label(img, label, prediction=None):
    theta = label * 0.69 #Steering range for the car is +- 40 degrees -> 0.69
# radians
    line_length = 50
    line_thickness = 3
    label_line_color = (255, 0, 0)
    prediction_line_color = (0, 255, 255)
    pil_image = image.array_to_img(img, K.image_data_format(), scale=True)
    print('Actual Steering Angle = {0}'.format(label))
    draw_image = pil_image.copy()
    image_draw = ImageDraw.Draw(draw_image)
    first_point = (int(img.shape[1]/2), img.shape[0])
    second_point = (int((img.shape[1]/2) + (line_length * math.sin(theta))),
int(img.shape[0] - (line_length * math.cos(theta))))
    image_draw.line([first_point, second_point], fill=label_line_color,
width=line_thickness)

    if (prediction is not None):
        print('Predicted Steering Angle = {0}'.format(prediction))
        print('L1 Error: {0}'.format(abs(prediction-label)))
        theta = prediction * 0.69
        second_point = (int((img.shape[1]/2) + ((line_length/2) *
math.sin(theta))), int(img.shape[0] - ((line_length/2) * math.cos(theta))))
        image_draw.line([first_point, second_point], fill=prediction_line_color,
width=line_thickness * 3)

    del image_draw
    plt.imshow(draw_image)
    plt.show()

[sample_batch_train_data, sample_batch_test_data] = next(train_generator)
for i in range(0, 3, 1):
    draw_image_with_label(sample_batch_train_data[0][i],
	sample_batch_test_data[i])
```

我们应该看到类似于[图16-13](part0019.html#drawing_steering_angles_on_images)的输出。请注意，我们现在在训练模型时只关注ROI，从而忽略原始图像中存在的所有非相关信息。线表示地面真实转向角。这是汽车在摄像机拍摄图像时行驶的角度。

![在图像上绘制转向角](../images/00182.jpeg)

###### 图16-13。在图像上绘制转向角

## 模型定义

现在我们准备定义神经网络的架构。在这里，我们必须考虑到在删除零后，我们的数据集非常有限的问题。因此，我们不能构建太深的网络。由于我们处理的是图像，我们将需要一些卷积/最大池化对来提取特征。

然而，仅仅使用图像可能不足以使我们的模型收敛。仅仅使用图像进行训练也不符合现实世界中做驾驶决策的方式。在驾驶时，我们不仅感知周围环境，还意识到我们的速度、转向程度，以及油门和刹车踏板的状态。相机、激光雷达、雷达等传感器输入到神经网络中只对应驾驶员在做驾驶决策时手头上所有信息的一部分。呈现给神经网络的图像可能是从一辆静止的汽车或以60英里/小时行驶的汽车中拍摄的；网络无法知道是哪种情况。在以5英里/小时行驶时将方向盘向右转两度与以50英里/小时行驶时做同样动作将产生非常不同的结果。简而言之，试图预测转向角度的模型不应仅依赖感官输入。它还需要关于汽车当前状态的信息。幸运的是，我们有这些信息可用。

在上一节的结尾，我们指出我们的数据集有四个部分。对于每个图像，除了转向角标签和元数据外，我们还记录了与图像对应的汽车上次已知状态。这以 (转向、油门、刹车、速度) 元组的形式存储，我们将使用这些信息以及我们的图像作为神经网络的输入。请注意，这不违反我们的“Hello, World!”要求，因为我们仍然将单个摄像头作为唯一的外部传感器。

将我们讨论的所有内容放在一起，您可以在 [图16-14](part0019.html#network_architecture) 中看到我们将用于此问题的神经网络。我们使用了三个卷积层，分别具有16、32、32个滤波器，并且使用了 (3,3) 的卷积窗口。我们将图像特征（从卷积层输出）与提供汽车先前状态的输入层进行合并。然后将组合特征集传递到两个分别具有64和10个隐藏神经元的全连接层中。我们网络中使用的激活函数是 ReLU。请注意，与我们在前几章中处理的分类问题不同，我们网络的最后一层是一个没有激活的单个神经元。这是因为我们要解决的问题是一个回归问题。我们网络的输出是转向角度，一个浮点数，而不是我们之前预测的离散类别。

![网络架构](../images/00141.jpeg)

###### 图16-14\. 网络架构

现在让我们实现我们的网络。我们可以使用 `model.summary()`：

```py
image_input_shape = sample_batch_train_data[0].shape[1:]
state_input_shape = sample_batch_train_data[1].shape[1:]
activation = 'relu'

# Create the convolutional stacks
pic_input = Input(shape=image_input_shape)

img_stack = Conv2D(16, (3, 3), name="convolution0", padding='same',
activation=activation)(pic_input)
img_stack = MaxPooling2D(pool_size=(2,2))(img_stack)
img_stack = Conv2D(32, (3, 3), activation=activation, padding='same',
name='convolution1')(img_stack)
img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
img_stack = Conv2D(32, (3, 3), activation=activation, padding='same',
name='convolution2')(img_stack)
img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
img_stack = Flatten()(img_stack)
img_stack = Dropout(0.2)(img_stack)

# Inject the state input
state_input = Input(shape=state_input_shape)
merged = concatenate([img_stack, state_input])

# Add a few dense layers to finish the model
merged = Dense(64, activation=activation, name='dense0')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(10, activation=activation, name='dense2')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1, name='output')(merged)

adam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model = Model(inputs=[pic_input, state_input], outputs=merged)
model.compile(optimizer=adam, loss='mse')
```

### 回调

Keras 的一个很好的特性是能够声明 *回调函数*。回调函数在每个训练周期结束后执行，帮助我们深入了解训练过程并在一定程度上控制超参数。它们还让我们定义在训练进行时执行某些操作的条件；例如，如果损失停止减少，则提前停止训练。我们将为我们的实验使用一些回调函数：

`ReduceLROnPlateau`

如果模型接近最小值且学习率过高，模型将在该最小值周围循环而无法达到它。当验证损失达到平稳期并停止改善时，此回调将允许模型减少学习率，使我们能够达到最佳点。

`CSVLogger`

这使我们能够将每个周期结束后模型的输出记录到一个 CSV 文件中，这样我们就可以跟踪进展而无需使用控制台。

`ModelCheckpoint`

通常，我们希望使用在验证集上损失最低的模型。此回调将在每次验证损失改善时保存模型。

`EarlyStopping`

当验证损失不再改善时，我们将停止训练。否则，我们会面临过拟合的风险。此选项将检测验证损失停止改善的时候，并在发生这种情况时停止训练过程。

现在让我们继续实现这些回调：

```py
plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
min_lr=0.0001, verbose=1)

checkpoint_filepath = os.path.join(MODEL_OUTPUT_DIR, 'models', '{0}_model.{1}
-{2}.h5'.format('model', '{epoch:02d}', '{val_loss:.7f}'))
checkAndCreateDir(checkpoint_filepath)
checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True,
verbose=1)

csv_callback = CSVLogger(os.path.join(MODEL_OUTPUT_DIR, 'training_log.csv'))

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, 
                                                            verbose=1)

nbcallback = TQDMNotebookCallback()
setattr(nbcallback, 'on_train_batch_begin', `lambda` x,y: `None`)
setattr(nbcallback, 'on_train_batch_end', `lambda` x,y: `None`)
setattr(nbcallback, 'on_test_begin', `lambda` x: `None`)
setattr(nbcallback, 'on_test_end', `lambda` x: `None`)
setattr(nbcallback, 'on_test_batch_begin', `lambda` x,y: `None`)
setattr(nbcallback, 'on_test_batch_end', `lambda` x,y: `None`)

callbacks=[plateau_callback, csv_callback, checkpoint_callback,
early_stopping_callback, nbcallback]
```

我们现在已经准备好开始训练了。模型需要一段时间来训练，所以这将是一个不错的Netflix休息时间。训练过程应该以约0.0003的验证损失终止：

```py
history = model.fit_generator(train_generator,
steps_per_epoch=num_train_examples // batch_size, epochs=500,
	callbacks=callbacks, validation_data=eval_generator, 
	validation_steps=num_eval_examples // batch_size, verbose=2)
```

```py
Epoch 1/500
Epoch 00001: val_loss improved from inf to 0.02338, saving model to
model\models\model_model.01-0.0233783.h5
- 442s - loss: 0.0225 - val_loss: 0.0234
Epoch 2/500
Epoch 00002: val_loss improved from 0.02338 to 0.00859, saving model to
model\models\model_model.02-0.0085879.h5
- 37s - loss: 0.0184 - val_loss: 0.0086
Epoch 3/500
Epoch 00003: val_loss improved from 0.00859 to 0.00188, saving model to
model\models\model_model.03-0.0018831.h5
- 38s - loss: 0.0064 - val_loss: 0.0019
…………………………….
```

我们的模型现在已经训练好并准备就绪。在看到它的表现之前，让我们进行一个快速的合理性检查，并将一些预测绘制在图像上：

```py
[sample_batch_train_data, sample_batch_test_data] = next(train_generator)
predictions = model.predict([sample_batch_train_data[0],
sample_batch_train_data[1]])
for i in range(0, 3, 1):
    draw_image_with_label(sample_batch_train_data[0][i],
                          sample_batch_test_data[i], predictions[i])
```

我们应该看到类似于[图16-15](part0019.html#drawing_actual_and_predicted_steering_an)的输出。在这个图中，粗线是预测输出，细线是标签输出。看起来我们的预测相当准确（我们还可以在图像上方看到实际和预测值）。是时候部署我们的模型并看看它的表现了。

![在图像上绘制实际和预测的转向角](../images/00095.jpeg)

###### 图16-15。在图像上绘制实际和预测的转向角

# 部署我们的自动驾驶模型

本节中的所有步骤也在Jupyter Notebook [*TestModel.ipynb*](https://oreil.ly/5saDl)中有详细说明。现在我们的模型已经训练好了，是时候启动模拟器并使用我们的模型驾驶汽车了。

与之前一样，首先导入一些库并定义路径：

```py
from tensorflow.keras.models import load_model
import sys
import numpy as np
import glob
import os

if ('../../PythonClient/' not in sys.path):
    sys.path.insert(0, '../../PythonClient/')
from AirSimClient import *

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be
# used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('model/models/*.h5') 
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model

print('Using model {0} for testing.'.format(MODEL_PATH))
```

接下来，加载模型并在景观环境中连接到AirSim。要启动模拟器，在Windows机器上，打开一个PowerShell命令窗口，位于我们解压缩模拟器包的位置，并运行以下命令：

```py
.\AD_Cookbook_Start_AirSim.ps1 landscape
```

在Jupyter Notebook中，运行以下命令将模型连接到AirSim客户端。确保模拟器在启动此过程之前已经运行：

```py
model = load_model(MODEL_PATH)

client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
print('Connection established!')
```

连接建立后，让我们现在设置汽车的初始状态以及用于存储模型输出的一些缓冲区：

```py
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

image_buf = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1,4))
```

下一步是设置模型期望从模拟器接收RGB图像作为输入。我们需要为此定义一个辅助函数：

```py
def get_image():
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene,
False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)

    return image_rgba[76:135,0:255,0:3].astype(float)
```

最后，让我们为我们的模型设置一个无限循环，从模拟器中读取图像以及汽车的当前状态，预测转向角，并将其发送回模拟器。因为我们的模型只预测转向角，所以我们需要提供一个控制信号来自行维持速度。让我们设置这样一个控制信号，使汽车尝试以恒定的5m/s速度行驶：

```py
while (True):
    car_state = client.getCarState()

    if (car_state.speed < 5):
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0

    image_buf[0] = get_image()
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle,
car_controls.brake, car_state.speed])
    model_output = model.predict([image_buf, state_buf])
    car_controls.steering = round(0.5 * float(model_output[0][0]), 2)

    print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering,
car_controls.throttle))

    client.setCarControls(car_controls)
```

我们应该看到类似于以下的输出：

```py
Sending steering = 0.03, throttle = 1.0
Sending steering = 0.03, throttle = 1.0
Sending steering = 0.03, throttle = 1.0
Sending steering = 0.03, throttle = 1.0
Sending steering = 0.03, throttle = 1.0
Sending steering = -0.1, throttle = 1.0
Sending steering = -0.12, throttle = 1.0
Sending steering = -0.13, throttle = 1.0
Sending steering = -0.13, throttle = 1.0
Sending steering = -0.13, throttle = 1.0
Sending steering = -0.14, throttle = 1.0
Sending steering = -0.15, throttle = 1.0
```

我们成功了！汽车在道路上很好地行驶，大部分时间保持在右侧，小心地穿过所有急转弯和潜在的偏离道路的地方。恭喜我们训练出了我们的第一个自动驾驶模型！

![训练模型驾驶汽车](../images/00232.jpeg)

###### 图16-16。训练模型驾驶汽车

在我们结束之前，有几件值得注意的事情。首先，请注意汽车的运动不是完全平滑的。这是因为我们正在处理一个回归问题，并为汽车看到的每个图像帧做出一个转向角预测。解决这个问题的一种方法是在一系列连续的图像上平均预测。另一个想法可能是将其转化为一个分类问题。更具体地，我们可以为转向角定义桶（...，-0.1，-0.05，0，0.05，0.1，...），将标签分桶化，并为每个图像预测正确的桶。

如果让模型运行一段时间（略长于五分钟），我们会观察到汽车最终会随机偏离道路并撞车。这发生在一个有陡峭上坡的赛道部分。还记得我们在问题设置中的最后一个要求吗？高度变化需要操作油门和刹车。因为我们的模型只能控制转向角，所以在陡峭的道路上表现不佳。

# 进一步探索

在“Hello, World!”场景中训练过后，我们的模型显然不是一个完美的驾驶员，但不要灰心。请记住，我们只是刚刚触及深度学习和自动驾驶汽车交汇点的可能性表面。我们能够让我们的汽车几乎完美地学会驾驶，只使用了一个非常小的数据集，这是值得骄傲的事情！

以下是一些新的想法，可以在本章学到的基础上进行扩展。您可以使用本章已经准备好的设置来实现所有这些想法。

## 扩展我们的数据集

一般规则是，使用更多的数据有助于提高模型性能。现在我们已经启动并运行了模拟器，通过进行更多的数据收集运行来扩展我们的数据集将是一个有用的练习。我们甚至可以尝试将来自AirSim中各种不同环境的数据结合起来，看看我们在这些数据上训练的模型在不同环境中的表现如何。

在本章中，我们只使用了单个摄像头的RGB数据。AirSim允许我们做更多的事情。例如，我们可以收集深度视图、分割视图、表面法线视图等每个可用摄像头的图像。因此，对于每个实例，我们可能有20个不同的图像（对于四种模式中的所有五个摄像头）。使用所有这些额外数据能帮助我们改进我们刚刚训练的模型吗？

## 在序列数据上训练

我们的模型目前对每个预测使用单个图像和单个车辆状态。然而，这并不是我们在现实生活中驾驶的方式。我们的行动总是考虑到导致给定时刻的最近一系列事件。在我们的数据集中，我们有所有图像的时间戳信息可用，我们可以使用这些信息创建序列。我们可以修改我们的模型，使用前 *N* 个图像和状态进行预测。例如，给定过去的10个图像和过去的10个状态，预测下一个转向角度。（提示：这可能需要使用RNNs。）

## 强化学习

在下一章学习强化学习之后，我们可以回来尝试[*自动驾驶食谱*](https://oreil.ly/bTphH)中的[Distributed Deep Reinforcement Learning for Autonomous Driving](https://oreil.ly/u1hoC)教程。使用AirSim中包含的Neighborhood环境，该环境已经包含在我们为本章下载的软件包中，我们将学习如何通过迁移学习和云来扩展深度强化学习训练作业，并将训练时间从不到一周缩短到不到一个小时。

# 总结

本章让我们一窥深度学习如何推动自动驾驶行业。利用前几章学到的技能，我们使用Keras实现了自动驾驶的“Hello, World!”问题。通过探索手头的原始数据，我们学会了如何预处理数据，使其适合训练高性能模型。而且我们能够用一个非常小的数据集完成这个任务。我们还能够拿出我们训练好的模型，并将其部署到模拟世界中驾驶汽车。您不认为这其中有一些魔力吗？
