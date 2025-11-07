# 附录 C. 人工智能应用的数据源

第七章概述了在构建人工智能应用时良好数据收集和准备的重要性。本附录列出了你可能利用的一些数据源，以确保你有适合人工智能成功的正确数据。

## C.1 公共数据集

1.  AWS 上的开放数据注册处（[`registry.opendata.aws`](https://registry.opendata.aws)）包括其他数据集，如 PB 级的 Common Crawl 数据（[`commoncrawl.org/the-data`](http://commoncrawl.org/the-data)）。

1.  公共 API，如 Twitter API，提供了大量数据。我们在第六章中看到了社交媒体帖子如何用于执行分类和情感分析。

1.  Google 有一个公共数据集的搜索引擎（[`toolbox.google.com/datasetsearch`](https://toolbox.google.com/datasetsearch)）和公共数据集列表（[`ai.google/tools/datasets/`](https://ai.google/tools/datasets/)）。

1.  Kaggle 有一个包含数千个数据集的目录（[`www.kaggle.com/datasets`](https://www.kaggle.com/datasets)）。

1.  许多政府数据源现在都可以使用。一个例子是美国开放政府数据在[`data.gov`](https://data.gov)。

1.  如果你是一个觉得第二章中猫的内容太多的狗爱好者，你将会被斯坦福狗数据集中可用的 20,000 张狗图片所安慰（[`vision.stanford.edu/aditya86/ImageNetDogs/`](http://vision.stanford.edu/aditya86/ImageNetDogs/))！

提示：许多公共数据集都受到许可的限制。做你的作业，了解在你的工作中使用数据集的法律影响。

## C.2 软件分析和日志

除了公共、预包装的数据之外，还有许多收集机器学习应用数据的方法。现有的软件系统具有分析和日志数据，这些数据可以准备和优化以用于机器学习算法：

+   收集来自 Web 和移动应用程序的最终用户交互数据的分析平台是用户行为和交互的原始数据的宝贵来源。Google Analytics 就是这样一个例子。

+   网络服务器和后端应用程序日志或审计日志也可能是系统内外交互的全面来源。

## C.3 人类数据收集

当数据不易获得且需要大规模收集或转换时，有几种方式可以众包这项工作：

+   数据收集公司提供收集（通过调查或其他方式）或转换数据的服务。

+   存在着由 API 驱动的众包服务。Amazon Mechanical Turk (MTurk) 是一个著名的例子（[`www.mturk.com/`](https://www.mturk.com/)）。

+   我们中的许多人已经进行了无数次的 Captcha 检查，作为验证我们不是机器人的手段！这项服务实际上提供了两个好处。像 reCAPTCHA 这样的服务也充当了为图像识别算法收集标记训练数据的一种手段。1

## C.4 设备数据

根据您的应用，您可能可以从现有系统中收集遥测数据，无论是使用软件监控工具还是硬件传感器：

+   传感器不再仅限于工业自动化设备。物联网（IoT）设备正在许多环境中变得普遍，并生成可能庞大的数据集。

+   静态图像或视频摄像头可以用来收集用于训练和分析的图像数据。例如，想想谷歌街景所需的图像捕捉规模，以及 reCAPTCHA 如何作为大规模标注这些图像的手段。

* * *

1. James O’Malley, “Captcha if you can: how you’ve been training AI for years without realising it.” TechRadar 12 January 2018, [`www.techradar.com/news/captcha-if-you-can-how-youve-been-training-ai-for-years-without-realising-it`](https://www.techradar.com/news/captcha-if-you-can-how-youve-been-training-ai-for-years-without-realising-it).
