# 第7章。实用工具、技巧和窍门

本章包含我们作为作者在专业工作中以及在撰写本书过程中遇到的材料，主要是在实验过程中。这里涵盖的材料不一定适用于任何单独的章节；相反，这些材料是深度学习从业者在日常工作中可能会发现有用的，涵盖了各种任务的实用指南，包括设置环境、训练、模型互操作性、数据收集和标记、代码质量、管理实验、团队协作实践、隐私以及进一步探索主题。

由于人工智能领域变化迅速，本章是书籍Github存储库（请参见[*http://PracticalDeepLearning.ai*](http://PracticalDeepLearning.ai)）中的“活动”文档的一个小子集，该文档位于*code/chapter-9*，正在不断发展。如果您有更多问题，或者更好的答案可以帮助其他读者，请随时在Twitter上发送推文[@PracticalDLBook](https://www.twitter.com/PracticalDLBook)或提交拉取请求。

# 安装

**Q:** *我在GitHub上发现了一个有趣且有用的Jupyter Notebook。要使代码运行，需要克隆存储库、安装软件包、设置环境等多个步骤。有没有一种即时的方法可以交互式运行它？*

只需将Git存储库URL输入到Binder（*[mybinder.org](http://mybinder.org)*），它将把它转换为一组交互式笔记本。在幕后，它会搜索存储库根目录中的*requirements.txt*或*environment.yml*等依赖文件。这将用于构建一个Docker镜像，以帮助在浏览器中交互式运行笔记本。

**Q:** *在新的Ubuntu机器上使用NVIDIA GPU，快速搭建深度学习环境的最快方法是什么？*

如果`pip install tensorflow-gpu`能解决所有问题，那该多好啊。然而，这离现实还很远。在新安装的Ubuntu机器上，列出所有安装步骤至少需要三页以上的时间来跟随，包括安装NVIDIA GPU驱动程序、CUDA、cuDNN、Python、TensorFlow和其他软件包。然后需要仔细检查CUDA、cuDNN和TensorFlow之间的版本互操作性。很多时候，这会导致系统崩溃。可以说是一种痛苦的世界！

如果两行代码就能轻松解决所有问题，那不是很棒吗？有求必应：

```py
$ sudo apt update && sudo ubuntu-drivers autoinstall && sudo reboot 
$ export LAMBDA_REPO=$(mktemp) \
&& wget -O${LAMBDA_REPO} \
https://lambdalabs.com/static/misc/lambda-stack-repo.deb \
&& sudo dpkg -i ${LAMBDA_REPO} && rm -f ${LAMBDA_REPO} \
&& sudo apt-get update && sudo apt-get install -y lambda-stack-cuda \
&& sudo reboot
```

第一行确保所有驱动程序都已更新。第二行是由总部位于旧金山的Lambda Labs提供的。该命令设置Lambda Stack，安装了TensorFlow、Keras、PyTorch、Caffe、Caffe2、Theano、CUDA、cuDNN和NVIDIA GPU驱动程序。因为公司需要在成千上万台机器上安装相同的深度学习软件包，所以它使用了一行命令自动化了这个过程，然后开源了它，以便其他人也可以使用。

**Q:** *在Windows PC上安装TensorFlow的最快方法是什么？*

1.  安装Anaconda Python 3.7。

1.  在命令行上运行`conda install tensorflow-gpu`。

1.  如果您没有GPU，运行`conda install tensorflow`。

基于CPU的Conda安装的一个额外好处是它安装了经过优化的Intel MKL TensorFlow，比使用`pip install tensorflow`得到的版本运行更快。

**Q:** *我有一块AMD GPU。在现有系统上，我能从TensorFlow的GPU加速中受益吗？*

尽管大多数深度学习领域使用NVIDIA GPU，但有越来越多的人群在AMD硬件上运行，并借助ROCm堆栈。使用命令行进行安装很简单：

1.  `sudo apt install rocm-libs miopen-hip cxlactivitylogger`

1.  `sudo apt install wget python3-pip`

1.  `pip3 install --user tensorflow-rocm`

**Q:** *忘记安装，我在哪里可以获得预安装的深度学习容器？*

Docker与设置环境是同义词。Docker帮助运行与工具、库和配置文件捆绑在一起的隔离容器。在选择来自主要云提供商（如AWS、Microsoft Azure、GCP、阿里巴巴等）的虚拟机（VM）时，有几个深度学习Docker容器可用于立即开始工作。NVIDIA还免费提供NVIDIA GPU云容器，这些容器与用于在MLPerf基准测试中打破训练速度记录的高性能容器相同。您甚至可以在桌面机器上运行这些容器。

# 训练

**Q:** *我不喜欢不断盯着屏幕检查我的训练是否已完成。我能否在手机上收到通知警报？*

使用[Knock Knock](https://oreil.ly/uX3qb)，这是一个Python库，正如其名称所示，当您的训练结束（或程序崩溃）时，通过电子邮件、Slack甚至Telegram发送警报通知您！最重要的是，只需向您的训练脚本添加两行代码即可。不再需要打开程序一千次来检查训练是否已完成。

**Q:** *我更喜欢图形和可视化而不是纯文本。我能否获得我的训练过程的实时可视化？*

FastProgress进度条（最初由Sylvain Gugger为fast.ai开发）来拯救。

**Q:** *我进行大量的实验迭代，经常忘记每次实验之间的变化以及变化的影响。如何以更有组织的方式管理我的实验？*

软件开发通过版本控制具有保留历史更改记录的能力。不幸的是，机器学习并没有同样的奢侈。现在通过Weights and Biases和Comet.ml等工具，情况正在改变。它们允许您跟踪多次运行，并记录训练曲线、超参数、输出、模型、注释等，只需向您的Python脚本添加两行代码。最重要的是，通过云的力量，即使您远离机器，也可以方便地跟踪实验并与他人分享结果。

**Q:** *如何检查TensorFlow是否在我的机器上使用GPU？*

使用以下方便的命令：

```py
tf.test.is_gpu_available()
```

**Q:** *我的机器上有多个GPU。我不希望我的训练脚本占用所有GPU。如何限制我的脚本仅在特定GPU上运行？*

*使用* `CUDA_VISIBLE_DEVICES=GPU_ID`。只需在训练脚本命令前加上以下前缀：

```py
$ CUDA_VISIBLE_DEVICES=GPU_ID python train.py
```

或者，在您的培训脚本中的早期写入以下行：

```py
import os
os.environ["CUDA_VISIBLE_DEVICES"]="GPU_ID"
```

`GPU_ID`可以具有值如0、1、2等。您可以使用`nvidia-smi`命令查看这些ID（以及GPU使用情况）。要分配给多个GPU，使用逗号分隔的ID列表。

**Q:** *有时候感觉在训练时有太多旋钮要调整。是否可以自动完成，以获得最佳准确性？*

有许多自动化超参数调整的选项，包括针对Keras的Hyperas和Keras Tuner，以及更通用的框架，如Hyperopt和贝叶斯优化，执行广泛的实验以更智能地最大化我们的目标（即在我们的情况下最大化准确性）比简单的网格搜索更智能。

**Q:** *ResNet和MobileNet对我的用例已经足够好了。是否可能构建一个可以在我的情况下实现更高准确度的模型架构？*

三个词：神经架构搜索（NAS）。让算法为您找到最佳架构。NAS可以通过Auto-Keras和AdaNet等软件包实现。

**Q:** *如何调试我的TensorFlow脚本？*

答案就在问题中：TensorFlow调试器（`tfdbg）。

# 模型

**Q:** *我想快速了解我的模型的输入和输出层，而不必编写代码。我该如何实现？*

使用Netron。它以图形方式显示您的模型，并在点击任何层时提供有关架构的详细信息。

**Q:** 我需要发表一篇研究论文。我应该使用哪个工具来绘制我的有机、自由范围、无麸质模型架构？

当然是MS Paint！不，我们只是开玩笑。我们也喜欢NN-SVG和PlotNeuralNet，用于创建高质量的CNN图表。

**Q:** 是否有一个一站式的所有模型的平台？

确实！探索[PapersWithCode.com](http://PapersWithCode.com)、ModelZoo.co和ModelDepot.io，获取一些灵感。

**Q:** 我已经训练好了我的模型。如何让其他人可以使用它？

你可以开始让模型可以从GitHub下载。然后将其列在前面答案中提到的模型动物园中。为了更广泛地采用，将其上传到TensorFlow Hub（tfhub.dev）。

除了模型之外，您应该发布一个“模型卡片”，它本质上就像模型的简历。这是一个详细介绍作者信息、准确度指标和基准数据集的简短报告。此外，它提供了关于潜在偏见和超出范围用途的指导。

**Q:** 我之前在X框架中训练过一个模型，但我需要在Y框架中使用它。我需要浪费时间在Y框架中重新训练吗？

不需要。你只需要ONNX的力量。对于不在TensorFlow生态系统中的模型，大多数主要的深度学习库都支持将它们保存为ONNX格式，然后可以将其转换为TensorFlow格式。微软的MMdnn可以帮助进行这种转换。

# 数据

**Q:** 我能在几分钟内收集关于一个主题的数百张图像吗？

是的，你可以使用一个名为Fatkun Batch Download Image的Chrome扩展，在三分钟或更短的时间内收集数百张图像。只需在你喜欢的图像搜索引擎中搜索一个关键词，按正确的使用权限（例如，公共领域）过滤图像，然后按下Fatkun扩展下载所有图像。参见第12章，在那里我们使用它构建了一个Not Hotdog应用程序。

额外提示：要从单个网站下载，请搜索一个关键词，后面跟着site:网站地址。例如，“马 site:flickr.com”。

**Q:** 忘掉浏览器。如何使用命令行从Google爬取图像？

```py
$ pip install google_images_download
$ googleimagesdownload -k=horse -l=50 -r=labeled-for-reuse
```

`-k`、`-l`和`-r`分别是`keyword`、`limit`（图像数量）和`usage_rights`的缩写。这是一个功能强大的工具，有许多选项可以控制和过滤从Google搜索中下载的图像。此外，它保存了搜索引擎链接的原始图像，而不仅仅是加载Google Images显示的缩略图。要保存超过100张图像，请安装`selenium`库以及`chromedriver`。

**Q:** 这些工具不足以收集图像。我需要更多控制。还有哪些工具可以帮助我以更自定义的方式下载数据，超越搜索引擎？

带有GUI（无需编程）：

[ScrapeStorm.com](http://ScrapeStorm.com)

易于GUI识别要提取的元素规则

[WebScraper.io](http://WebScraper.io)

基于Chrome的爬虫扩展，特别用于从单个网站提取结构化输出

[80legs.com](http://80legs.com)

基于云的可扩展爬虫，用于并行、大型任务

基于Python的编程工具：

[Scrapy.org](http://Scrapy.org)

对于更可编程的爬取控制，这是最著名的爬虫之一。与构建自己的天真爬虫来探索网站相比，它提供了按域名、代理和IP进行节流速率的功能；可以处理robots.txt；提供了在向Web服务器显示的浏览器标头方面的灵活性；并处理了几种可能的边缘情况。

InstaLooter

一个用于爬取Instagram的基于Python的工具。

**Q:** 我有目标类别的图像，但现在需要负类别（非项目/背景）的图像。有什么快速方法可以构建一个负类别的大数据集？

[ImageN](https://oreil.ly/s4Nyk)提供了1,000张图片——200个ImageNet类别的5张随机图片——您可以将其用作负类。如果需要更多，请从ImageNet中以编程方式下载随机样本。

**Q：** *我如何搜索适合我的需求的预构建数据集？*

尝试Google数据集搜索、*VisualData.io*和*[DatasetList.com](http://DatasetList.com)*。

**Q：** *对于像ImageNet这样的数据集，下载、弄清楚格式，然后加载它们进行训练需要太多时间。有没有一种简单的方法来读取流行的数据集？*

TensorFlow数据集是一个不断增长的数据集集合，可以与TensorFlow一起使用。其中包括ImageNet、COCO（37 GB）和Open Images（565 GB）等。这些数据集被公开为`tf.data.Datasets`，并提供了高性能的代码来将它们输入到您的训练流程中。

**Q：** *在数百万张ImageNet图片上进行训练将需要很长时间。有没有一个更小的代表性数据集，我可以尝试进行训练，以便快速进行实验和迭代？*

尝试[Imagenette](https://oreil.ly/NpYBe)。由fast.ai的Jeremy Howard创建，这个1.4 GB的数据集只包含10个类别，而不是1,000个。

**Q：** *有哪些可用于训练的最大数据集？*

+   腾讯ML图片：1,770万张图片，带有11,000个类别标签

+   Open Images V4（来自Google）：19.7 K类别中的9百万张图片

+   BDD100K（来自加州大学伯克利分校）：来自100,000个驾驶视频的图像，超过1,100小时

+   YFCC100M（来自雅虎）：99.2百万张图片

**Q：** *有哪些现成的大型视频数据集可以使用？*

| **名称** | **详情** |
| --- | --- |
| YouTube-8M | 6.1百万个视频，3,862个类别，26亿个视听特征每个视频3.0个标签1.53 TB的随机抽样视频 |
| Something Something（来自Twenty Billion Neurons） | 174个动作类别中的221,000个视频，例如“将水倒入酒杯但错过了，所以洒在旁边”人们用日常物品执行预定义的动作 |
| Jester（来自Twenty Billion Neurons） | 27个类别中的148,000个视频，例如“用两根手指放大”在网络摄像头前预定义的手势 |

**Q：** *这些是有史以来组装的最大标记数据集吗？*

没有！像Facebook和Google这样的公司会精心策划自己的私人数据集，这些数据集比我们可以使用的公共数据集要大得多：

+   Facebook：35亿张带有嘈杂标签的Instagram图片（首次报道于2018年）

+   谷歌 - JFT-300M：30亿张带有嘈杂标签的图片（首次报道于2017年）

遗憾的是，除非你是这些公司的员工，否则你实际上无法访问这些数据集。我们必须说，这是一个不错的招聘策略。

**Q：** *我如何获得帮助注释数据？*

有几家公司可以帮助标记不同类型的注释。值得一提的几家公司包括SamaSource、Digital Data Divide和iMerit，他们雇用那些机会有限的人，最终通过在贫困社区就业来创造积极的社会经济变革。

**Q：** *是否有用于数据集的版本控制工具，就像Git用于代码一样？*

Qri和Quilt可以帮助版本控制我们的数据集，有助于实验的可重复性。

**Q：** *如果我没有一个大型数据集来解决我的独特问题怎么办？*

尝试为训练开发一个合成数据集！例如，找到感兴趣对象的逼真3D模型，并使用Unity等3D框架将其放置在逼真的环境中。调整光照和相机位置、缩放和旋转，从多个角度拍摄这个对象的快照，生成无限的训练数据。此外，像AI.Reverie、CVEDIA、Neuromation、Cognata、Mostly.ai和DataGen Tech这样的公司提供了用于训练需求的逼真模拟。合成训练数据的一个重要好处是标记过程内置于合成过程中。毕竟，您会知道自己在创造什么。与手动标记相比，这种自动标记可以节省大量金钱和精力。

# 隐私

**Q:** *如何开发一个更注重隐私的模型而不深入研究密码学？*

TensorFlow Encrypted可能是您正在寻找的解决方案。它可以使用加密数据进行开发，这在云端尤为重要。内部，大量的安全多方计算和同态加密实现了隐私保护的机器学习。

**Q:** *我可以防止别人窥探我的模型吗？*

嗯，除非你在云端，权重是可见的并且可以被反向工程。在智能手机上部署时，使用Fritz库来保护您模型的知识产权。

# 教育和探索

**Q:** *我想成为人工智能专家。除了这本书，我应该在哪里投入更多时间学习？*

在互联网上有几个资源可以深入学习深度学习。我们强烈推荐这些来自一些最好的老师的视频讲座，涵盖了从计算机视觉到自然语言处理等各种应用领域。

+   Fast.ai（由Jeremy Howard和Rachel Thomas创建）提供了一个免费的14集视频讲座系列，采用更多的PyTorch实践学习方法。除了课程，还有一整套工具和一个活跃的社区，已经导致了许多研究论文和可直接使用的代码的突破（比如使用fast.ai库训练最先进网络只需三行代码）。

+   Deeplearning.ai（由Andrew Ng创建）提供了一个包含五门课程的“深度学习专项课程”。它是免费的（尽管您可以支付一小笔费用以获得证书），将进一步巩固您的理论基础。Ng博士在Coursera上的第一门机器学习课程已经教授了超过两百万名学生，这个系列延续了深受初学者和专家喜爱的高度易懂的内容传统。

+   如果我们没有在这个列表中鼓励您注意[O'Reilly的在线学习](http://oreilly.com)平台，我们将感到遗憾。这个平台帮助超过两百万用户提升他们的职业，包含数百本书籍、视频、在线培训和由O'Reilly的人工智能和数据会议上的领先思想家和实践者发表的主题演讲。

**Q:** *我在哪里可以找到有趣的笔记本进行学习？*

Google Seedbank是一个互动机器学习示例集合。基于Google Colaboratory构建，这些Jupyter笔记本可以立即运行，无需任何安装。一些有趣的示例包括：

+   使用GANs生成音频

+   视频动作识别

+   生成莎士比亚风格的文本

+   音频风格转移

**Q:** *我在哪里可以了解特定主题的最新技术？*

考虑到人工智能领域技术更新迅速，SOTAWHAT是一个方便的命令行工具，可以搜索最新模型、数据集、任务等研究论文。例如，要查找ImageNet的最新结果，请在命令行上使用`sotawhat imagenet`。此外，[*paperswithcode.com/sota*](http://paperswithcode.com/sota)还提供了论文、源代码和发布模型的存储库，以及一个交互式的基准时间轴。

**Q:** *我正在阅读一篇Arxiv上的论文，我非常喜欢。我需要从头开始编写代码吗？*

一点也不！ResearchCode Chrome扩展程序使得在浏览*[arxiv.org](http://arxiv.org)*或Google Scholar时轻松找到代码。只需按一下扩展按钮即可。您也可以在*[ResearchCode.com](http://ResearchCode.com)*网站上查找代码而无需安装扩展程序。

**Q:** *我不想写任何代码，但我仍然想使用我的摄像头进行交互式实验模型。我该怎么做？

[Runway ML](https://runwayml.com)是一个易于使用但功能强大的GUI工具，允许您下载模型（来自互联网或您自己的模型）并使用网络摄像头或其他输入，如视频文件，以交互方式查看输出。这使得进一步组合和混合模型的输出以创建新作品成为可能。所有这些只需点击几下鼠标就能完成；因此，它吸引了大量的艺术家社区！

**Q:** *如果我可以在没有代码的情况下进行测试，那么我也可以在没有代码的情况下进行训练吗？*

我们在[第8章](part0010.html#9H5K3-13fa565533764549a6f0ab7f11eed62b)（基于网络）和[第12章](part0014.html#DB7S3-13fa565533764549a6f0ab7f11eed62b)（基于桌面）中详细讨论了这个问题。简而言之，诸如微软的[CustomVision.ai](http://CustomVision.ai)、谷歌的Cloud AutoML Vision、Clarifai、百度EZDL和苹果的Create ML等工具提供了拖放式训练功能。其中一些工具只需几秒钟就能完成训练。

# 最后一个问题

**Q:** *告诉我一个伟大的深度学习恶作剧？*

打印并挂在[*keras4kindergartners.com*](http://keras4kindergartners.com)上显示的海报，靠近饮水机，看着人们的反应。

![来自keras4kindergartners.com的关于AI状态的讽刺海报](../images/00056.jpeg)

###### 图7-1\. 来自[keras4kindergartners.com](http://keras4kindergartners.com)的关于AI状态的讽刺海报
