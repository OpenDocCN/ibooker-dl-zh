# 前言

# 机器学习海啸

2006 年，Geoffrey Hinton 等人发表了一篇论文，展示了如何训练一个能够以最先进的精度（>98%）识别手写数字的深度神经网络。他们将这种技术称为“深度学习”。深度神经网络是我们大脑皮层的（非常）简化模型，由一系列人工神经元层组成。在当时，训练深度神经网络被普遍认为是不可能的，大多数研究人员在 1990 年代末放弃了这个想法。这篇论文重新激起了科学界的兴趣，不久之后，许多新论文证明了深度学习不仅是可能的，而且能够实现令人惊叹的成就，其他任何机器学习（ML）技术都无法匹敌（在巨大的计算能力和大量数据的帮助下）。这种热情很快扩展到许多其他机器学习领域。

十年后，机器学习已经征服了行业，如今它是许多高科技产品中许多神奇功能的核心，比如排名您的网络搜索结果，为您的智能手机提供语音识别，推荐视频，甚至可能驾驶您的汽车。

# 在您的项目中应用机器学习

因此，您对机器学习感到兴奋，并希望加入这个派对！

也许您想让您自制的机器人拥有自己的大脑？让它识别人脸？或学会四处走动？

或者您的公司有大量数据（用户日志、财务数据、生产数据、机器传感器数据、热线统计、人力资源报告等），很可能如果您知道在哪里寻找，您可以发现一些隐藏的宝藏。通过机器学习，您可以实现以下目标[以及更多](https://homl.info/usecases)：

+   分析客户并找到每个群体的最佳营销策略。

+   根据类似客户购买的产品，为每个客户推荐产品。

+   检测哪些交易可能是欺诈性的。

+   预测明年的收入。

无论出于何种原因，您已经决定学习机器学习并在您的项目中实施它。好主意！

# 目标和方法

本书假设您对机器学习几乎一无所知。其目标是为您提供实现能够*从数据中学习*的程序所需的概念、工具和直觉。

我们将涵盖大量技术，从最简单和最常用的（如线性回归）到一些经常赢得比赛的深度学习技术。为此，我们将使用生产就绪的 Python 框架：

+   Scikit-Learn 非常易于使用，同时高效实现了许多机器学习算法，因此它是学习机器学习的绝佳入门点。它由 David Cournapeau 于 2007 年创建，现在由法国国家计算机与自动化研究所（Inria）的一组研究人员领导。

+   TensorFlow 是一个更复杂的分布式数值计算库。它通过在数百个多 GPU（图形处理单元）服务器上分布计算，使得训练和运行非常大的神经网络变得高效。TensorFlow（TF）由 Google 创建，并支持许多其大规模机器学习应用。它于 2015 年 11 月开源，2.0 版本于 2019 年 9 月发布。

+   Keras 是一个高级深度学习 API，使训练和运行神经网络变得非常简单。Keras 与 TensorFlow 捆绑在一起，并依赖于 TensorFlow 进行所有密集计算。

本书倾向于实践方法，通过具体的工作示例和一点点理论来培养对机器学习的直观理解。

###### 提示

虽然您可以不用拿起笔记本阅读本书，但我强烈建议您尝试一下代码示例。

# 代码示例

本书中的所有代码示例都是开源的，可以在[*https://github.com/ageron/handson-ml3*](https://github.com/ageron/handson-ml3)上在线获取，作为 Jupyter 笔记本。这些是交互式文档，包含文本、图片和可执行的代码片段（在我们的案例中是 Python）。开始的最简单最快的方法是使用 Google Colab 运行这些笔记本：这是一个免费服务，允许您直接在线运行任何 Jupyter 笔记本，无需在您的机器上安装任何东西。您只需要一个网络浏览器和一个 Google 账号。

###### 注意

在本书中，我假设您正在使用 Google Colab，但我也在其他在线平台上测试了这些笔记本，如 Kaggle 和 Binder，所以如果您愿意，也可以使用这些平台。或者，您可以安装所需的库和工具（或本书的 Docker 镜像），并在自己的机器上直接运行这些笔记本。请参阅[*https://homl.info/install*](https://homl.info/install)上的说明。

本书旨在帮助您完成工作。如果您希望使用代码示例以外的其他内容，并且该使用超出了公平使用准则的范围（例如出售或分发 O'Reilly 图书的内容，或将本书的大量材料整合到产品文档中），请通过*permissions@oreilly.com*联系我们以获取许可。

我们感谢，但不要求署名。署名通常包括标题、作者、出版商和 ISBN。例如：“*使用 Scikit-Learn、Keras 和 TensorFlow 进行实践机器学习* by Aurélien Géron. 版权所有 2023 Aurélien Géron, 978-1-098-12597-4.”

# 先决条件

本书假定您具有一些 Python 编程经验。如果您还不了解 Python，[*https://learnpython.org*](https://learnpython.org)是一个很好的开始。[Python.org](https://docs.python.org/3/tutorial)上的官方教程也非常不错。

本书还假定您熟悉 Python 的主要科学库，特别是[NumPy](https://numpy.org)、[Pandas](https://pandas.pydata.org)和[Matplotlib](https://matplotlib.org)。如果您从未使用过这些库，不用担心；它们很容易学习，我为每个库创建了一个教程。您可以在[*https://homl.info/tutorials*](https://homl.info/tutorials)上在线访问它们。

此外，如果您想完全理解机器学习算法的工作原理（不仅仅是如何使用它们），那么您应该至少对一些数学概念有基本的了解，尤其是线性代数。具体来说，您应该知道什么是向量和矩阵，以及如何执行一些简单的操作，比如添加向量，或转置和相乘矩阵。如果您需要快速了解线性代数（这真的不是什么难事！），我在[*https://homl.info/tutorials*](https://homl.info/tutorials)提供了一个教程。您还会找到一个关于微分计算的教程，这可能有助于理解神经网络是如何训练的，但并非完全必要掌握重要概念。本书偶尔还使用其他数学概念，如指数和对数，一些概率论，以及一些基本的统计概念，但没有太高级的内容。如果您需要帮助，请查看[*https://khanacademy.org*](https://khanacademy.org)，该网站提供许多优秀且免费的数学课程。

# 路线图

本书分为两部分。第一部分，“机器学习基础”，涵盖以下主题：

+   机器学习是什么，它试图解决什么问题，以及其系统的主要类别和基本概念

+   典型机器学习项目中的步骤

+   通过将模型拟合到数据来学习

+   优化成本函数

+   处理，清洗和准备数据

+   选择和工程特征

+   使用交叉验证选择模型和调整超参数

+   机器学习的挑战，特别是欠拟合和过拟合（偏差/方差权衡）

+   最常见的学习算法：线性和多项式回归，逻辑回归，k-最近邻，支持向量机，决策树，随机森林和集成方法

+   减少训练数据的维度以对抗“维度灾难”

+   其他无监督学习技术，包括聚类，密度估计和异常检测

第二部分，“神经网络和深度学习”，涵盖以下主题：

+   神经网络是什么以及它们适用于什么

+   使用 TensorFlow 和 Keras 构建和训练神经网络

+   最重要的神经网络架构：用于表格数据的前馈神经网络，用于计算机视觉的卷积网络，用于序列处理的循环网络和长短期记忆（LSTM）网络，用于自然语言处理的编码器-解码器和Transformer（以及更多！），自编码器，生成对抗网络（GANs）和扩散模型用于生成学习

+   训练深度神经网络的技术

+   如何构建一个代理（例如游戏中的机器人），通过试错学习良好策略，使用强化学习

+   高效加载和预处理大量数据

+   规模化训练和部署 TensorFlow 模型

第一部分主要基于 Scikit-Learn，而第二部分使用 TensorFlow 和 Keras。

###### 注意

不要过于仓促地跳入深水：尽管深度学习无疑是机器学习中最令人兴奋的领域之一，但您应该先掌握基础知识。此外，大多数问题可以使用更简单的技术（如随机森林和集成方法）很好地解决（在第一部分讨论）。深度学习最适合复杂问题，如图像识别，语音识别或自然语言处理，它需要大量数据，计算能力和耐心（除非您可以利用预训练的神经网络，如您将看到的那样）。

# 第一版和第二版之间的变化

如果您已经阅读了第一版，以下是第一版和第二版之间的主要变化：

+   所有代码都已从 TensorFlow 1.x 迁移到 TensorFlow 2.x，并且我用更简单的 Keras 代码替换了大部分低级 TensorFlow 代码（图形，会话，特征列等）。

+   第二版引入了用于加载和预处理大型数据集的 Data API，用于规模训练和部署 TF 模型的分布策略 API，用于将模型投入生产的 TF Serving 和 Google Cloud AI Platform，以及（简要介绍）TF Transform，TFLite，TF Addons/Seq2Seq，TensorFlow.js 和 TF Agents。

+   它还引入了许多其他 ML 主题，包括一个新的无监督学习章节，用于目标检测和语义分割的计算机视觉技术，使用卷积神经网络（CNN）处理序列，使用循环神经网络（RNN）、CNN 和Transformer进行自然语言处理（NLP），GANs 等。

有关更多详细信息，请参阅[*https://homl.info/changes2*](https://homl.info/changes2)。

# 第二版和第三版之间的变化

如果您阅读了第二版，以下是第二版和第三版之间的主要变化：

+   所有代码都已更新到最新的库版本。特别是，这第三版引入了许多新的 Scikit-Learn 补充（例如，特征名称跟踪，基于直方图的梯度提升，标签传播等）。它还引入了用于超参数调整的 Keras Tuner 库，用于自然语言处理的 Hugging Face 的 Transformers 库，以及 Keras 的新预处理和数据增强层。

+   添加了几个视觉模型（ResNeXt、DenseNet、MobileNet、CSPNet 和 EfficientNet），以及选择正确模型的指南。

+   第十五章现在分析芝加哥公共汽车和轨道乘客数据，而不是生成的时间序列，并介绍了 ARMA 模型及其变体。

+   第十六章关于自然语言处理现在构建了一个英语到西班牙语的翻译模型，首先使用编码器-解码器 RNN，然后使用Transformer模型。该章还涵盖了语言模型，如 Switch Transformers、DistilBERT、T5 和 PaLM（带有思维链提示）。此外，它介绍了视觉Transformer（ViTs）并概述了一些基于Transformer的视觉模型，如数据高效图像Transformer（DeiTs）、Perceiver 和 DINO，以及一些大型多模态模型的简要概述，包括 CLIP、DALL·E、Flamingo 和 GATO。

+   第十七章关于生成学习现在引入了扩散模型，并展示了如何从头开始实现去噪扩散概率模型（DDPM）。

+   第十九章从 Google Cloud AI 平台迁移到 Google Vertex AI，并使用分布式 Keras Tuner 进行大规模超参数搜索。该章现在包括您可以在线尝试的 TensorFlow.js 代码。它还介绍了其他分布式训练技术，包括 PipeDream 和 Pathways。

+   为了容纳所有新内容，一些部分被移至在线，包括安装说明、核主成分分析（PCA）、贝叶斯高斯混合的数学细节、TF Agents，以及以前的附录 A（练习解决方案）、C（支持向量机数学）和 E（额外的神经网络架构）。

更多详情请查看[*https://homl.info/changes3*](https://homl.info/changes3)。

# 其他资源

有许多优秀的资源可供学习机器学习。例如，Andrew Ng 在 Coursera 上的 ML 课程令人惊叹，尽管需要投入大量时间。

还有许多关于机器学习的有趣网站，包括 Scikit-Learn 的出色[用户指南](https://homl.info/skdoc)。您可能还会喜欢[Dataquest](https://dataquest.io)，它提供非常好的交互式教程，以及像[Quora](https://homl.info/1)上列出的 ML 博客。

还有许多关于机器学习的入门书籍。特别是：

+   Joel Grus 的《从零开始的数据科学》，第二版（O'Reilly），介绍了机器学习的基础知识，并使用纯 Python 实现了一些主要算法（从头开始，正如名称所示）。

+   Stephen Marsland 的《机器学习：算法视角》，第二版（Chapman＆Hall），是机器学习的一个很好的入门，深入涵盖了各种主题，使用 Python 中的代码示例（也是从头开始，但使用 NumPy）。

+   Sebastian Raschka 的《Python 机器学习》，第三版（Packt Publishing），也是机器学习的一个很好的入门，利用了 Python 开源库（Pylearn 2 和 Theano）。

+   François Chollet 的《Python 深度学习》，第二版（Manning），是一本非常实用的书，以清晰简洁的方式涵盖了广泛的主题，正如你可能从优秀的 Keras 库的作者所期望的那样。它更偏向于代码示例而不是数学理论。

+   Andriy Burkov 的《百页机器学习书》（自出版）非常简短，但涵盖了令人印象深刻的一系列主题，以平易近人的术语介绍，而不回避数学方程式。

+   Yaser S. Abu-Mostafa，Malik Magdon-Ismail 和 Hsuan-Tien Lin 的《从数据中学习》（AMLBook）是一个相当理论化的 ML 方法，提供了深刻的见解，特别是关于偏差/方差权衡（参见第四章）。

+   Stuart Russell 和 Peter Norvig 的《人工智能：现代方法》，第 4 版（Pearson），是一本涵盖大量主题的伟大（而庞大）的书籍，包括机器学习。它有助于将 ML 置于透视中。

+   Jeremy Howard 和 Sylvain Gugger 的《使用 fastai 和 PyTorch 进行编码的深度学习》（O'Reilly）提供了一个清晰实用的深度学习介绍，使用了 fastai 和 PyTorch 库。

最后，加入 ML 竞赛网站，如[Kaggle.com](https://kaggle.com/)，将使您能够在实际问题上练习技能，并获得来自一些最优秀的 ML 专业人士的帮助和见解。

# 本书使用的约定

本书使用以下排版约定：

*斜体*

指示新术语、URL、电子邮件地址、文件名和文件扩展名。

`等宽`

用于程序清单，以及在段落中引用程序元素，如变量或函数名、数据库、数据类型、环境变量、语句和关键字。

**`等宽粗体`**

显示用户应按照字面意义输入的命令或其他文本。

*`等宽斜体`*

显示应替换为用户提供的值或由上下文确定的值的文本。

标点

为避免混淆，本书中引号外的标点符号。对纯粹主义者表示歉意。

###### 提示

此元素表示提示或建议。

###### 注意

此元素表示一般说明。

###### 警告

此元素表示警告或注意事项。

# 致谢

在我最疯狂的梦想中，我从未想象过这本书的第一版和第二版会获得如此庞大的读者群。我收到了许多读者的留言，许多人提出问题，一些人友善地指出勘误，大多数人给我寄来鼓励的话语。我无法表达我对所有这些读者的巨大支持的感激之情。非常感谢你们所有人！如果您在代码示例中发现错误（或者只是想提问），请毫不犹豫地在 GitHub 上[提交问题](https://homl.info/issues3)，或者如果您在文本中发现错误，请提交[勘误](https://homl.info/errata3)。一些读者还分享了这本书如何帮助他们找到第一份工作，或者如何帮助他们解决了他们正在处理的具体问题。我发现这样的反馈极具激励性。如果您觉得这本书有帮助，我会很高兴如果您能与我分享您的故事，无论是私下（例如通过[LinkedIn](https://linkedin.com/in/aurelien-geron)）还是公开（例如在 Twitter 上@ aureliengeron 发推文或撰写[亚马逊评论](https://homl.info/amazon3)）。

同样非常感谢所有那些慷慨提供时间和专业知识来审阅这第三版，纠正错误并提出无数建议的出色人士。多亏了他们，这版书变得更好了：Olzhas Akpambetov、George Bonner、François Chollet、Siddha Ganju、Sam Goodman、Matt Harrison、Sasha Sobran、Lewis Tunstall、Leandro von Werra 和我亲爱的弟弟 Sylvain。你们都太棒了！

我也非常感激许多人在我前进的道路上支持我，回答我的问题，提出改进建议，并在 GitHub 上贡献代码：特别是 Yannick Assogba、Ian Beauregard、Ulf Bissbort、Rick Chao、Peretz Cohen、Kyle Gallatin、Hannes Hapke、Victor Khaustov、Soonson Kwon、Eric Lebigot、Jason Mayes、Laurence Moroney、Sara Robinson、Joaquín Ruales 和 Yuefeng Zhou。

没有 O'Reilly 出色的员工，特别是 Nicole Taché，这本书就不会存在，她给了我深刻的反馈，总是充满活力、鼓励和帮助：我无法想象有比她更好的编辑。非常感谢 Michele Cronin，她在最后几章中给予我鼓励，并帮助我完成了最后的工作。感谢整个制作团队，特别是 Elizabeth Kelly 和 Kristen Brown。同样感谢 Kim Cofer 进行了彻底的编辑工作，以及 Johnny O'Toole，他管理了与亚马逊的关系，并回答了我许多问题。感谢 Kate Dullea 大大改进了我的插图。感谢 Marie Beaugureau、Ben Lorica、Mike Loukides 和 Laurel Ruma 相信这个项目，并帮助我定义其范围。感谢 Matt Hacker 和整个 Atlas 团队回答了我关于格式、AsciiDoc、MathML 和 LaTeX 的所有技术问题，感谢 Nick Adams、Rebecca Demarest、Rachel Head、Judith McConville、Helen Monroe、Karen Montgomery、Rachel Roumeliotis 以及 O'Reilly 的所有其他贡献者。

我永远不会忘记所有在这本书的第一版和第二版中帮助过我的美好人们：朋友、同事、专家，包括 TensorFlow 团队的许多成员。名单很长：Olzhas Akpambetov，Karmel Allison，Martin Andrews，David Andrzejewski，Paige Bailey，Lukas Biewald，Eugene Brevdo，William Chargin，François Chollet，Clément Courbet，Robert Crowe，Mark Daoust，Daniel “Wolff” Dobson，Julien Dubois，Mathias Kende，Daniel Kitachewsky，Nick Felt，Bruce Fontaine，Justin Francis，Goldie Gadde，Irene Giannoumis，Ingrid von Glehn，Vincent Guilbeau，Sandeep Gupta，Priya Gupta，Kevin Haas，Eddy Hung，Konstantinos Katsiapis，Viacheslav Kovalevskyi，Jon Krohn，Allen Lavoie，Karim Matrah，Grégoire Mesnil，Clemens Mewald，Dan Moldovan，Dominic Monn，Sean Morgan，Tom O’Malley，James Pack，Alexander Pak，Haesun Park，Alexandre Passos，Ankur Patel，Josh Patterson，André Susano Pinto，Anthony Platanios，Anosh Raj，Oscar Ramirez，Anna Revinskaya，Saurabh Saxena，Salim Sémaoune，Ryan Sepassi，Vitor Sessak，Jiri Simsa，Iain Smears，Xiaodan Song，Christina Sorokin，Michel Tessier，Wiktor Tomczak，Dustin Tran，Todd Wang，Pete Warden，Rich Washington，Martin Wicke，Edd Wilder-James，Sam Witteveen，Jason Zaman，Yuefeng Zhou，以及我的兄弟 Sylvain。

最后，我要无限感谢我亲爱的妻子 Emmanuelle 和我们三个美妙的孩子 Alexandre、Rémi 和 Gabrielle，他们鼓励我努力完成这本书。他们无尽的好奇心是无价的：向妻子和孩子们解释这本书中一些最困难的概念帮助我澄清了思路，直接改进了许多部分。此外，他们还不断给我送来饼干和咖啡，还能要求什么呢？

¹ Geoffrey E. Hinton 等人，“深度信念网络的快速学习算法”，《神经计算》18 (2006): 1527–1554。

² 尽管 Yann LeCun 的深度卷积神经网络自上世纪 90 年代以来在图像识别方面表现良好，但它们并不是通用的。
