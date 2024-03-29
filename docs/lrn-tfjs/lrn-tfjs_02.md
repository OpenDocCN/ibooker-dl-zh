# 第一章：AI 是魔法

> “任何足够先进的技术都是不可区分的魔法。”
> 
> —亚瑟·克拉克

好吧，AI 并不是真正的魔法。事实上，AI 是超越其他技术的一步，以至于它*感觉*像魔法。解释一个有效的排序算法是很容易的，但深入研究智能本身会触及第三导轨，将我们全部带入一个全新的技术力量水平。这种指数级的能力提升是通过 TensorFlow.js 的智慧和才能实现的。

科学家和工程师们在 AI 领域已经重新创造了比喻性的轮子，并调整了控制它的机制。当我们在本书中深入研究 AI 时，我们将用 TensorFlow.js 灵活而坚固的框架掌握这些概念，并借此将我们的想法实现在 JavaScript 不断扩展的领域中。是的，JavaScript，世界上最流行的编程语言。

这里提供的概念和定义将为您提供技术射线视野。您将能够穿透首字母缩略词和流行词汇，看到并理解我们周围各个领域中不断出现的 AI 基础设施。AI 和机器学习概念将变得清晰，本章的定义可以作为识别核心原则的参考，这些原则将推动我们在 TensorFlow.js 的学术发展中启程。

我们将：

+   澄清 AI 和智能的领域

+   讨论机器学习的类型

+   回顾和定义常见术语

+   通过 TensorFlow.js 的视角审视概念

让我们开始吧！

###### 提示

如果您已经熟悉 TensorFlow.js 和机器学习术语、哲学和基本应用，您可能希望直接跳转到第二章。

# JavaScript 中的 AI 之路

TensorFlow.js，简单来说，是一个处理 JavaScript 中特定 AI 概念的框架。就是这样。幸运的是，您在正确的书籍中，处于历史上正确的时刻。AI 工业革命才刚刚开始。

当计算机出现时，一个拥有计算机的人可以在显著的规模上执行几乎不可能的任务。他们可以破解密码，从大量数据中瞬间提取信息，甚至像与另一个人玩游戏一样。一个人无法做到的事情不仅变得可能，而且变得司空见惯。自数字发明诞生以来，我们的目标一直是赋予计算机更多的能力。作为独立的人类，我们能够做任何事情，但不是所有事情。计算机以一种扩展我们限制的方式使我们获得了新的力量。我们中的许多人花费一生来磨练一些技能，而在这些技能中，甚至更少的人成为我们的专长。我们都在建立一生的成就，我们中的一些人在某一领域成为世界上最优秀的人，这种技能只能通过运气、信息和成千上万天的努力获得...直到现在。

AI 使我们能够跳到队伍的最前面；大胆地建造以前从未建造过的东西。每天我们都看到公司和研究人员一次又一次地迈出计算的飞跃。我们站在一个新行业的入口处，邀请我们参与世界将如何改变的过程。

您掌握着方向盘，驶向下一个大事件，而这本书就是您的方向盘。我们学习 AI 魔法的旅程只会受到您想象力的限制。借助启用 JavaScript 的机器学习，您可以利用摄像头、麦克风、即时更新、位置和其他物理传感器、服务和设备！

我相信你会问，“为什么以前 AI 没有做到这一点？为什么现在这很重要？”要理解这一点，你需要踏上人类对再现智能的探索之旅。

# 什么是智能？

关于思想的概念，尤其是通向机器智能的道路，可以写成一本又一本的书。就像所有哲学努力一样，关于智能的每一个具体陈述都可以在途中争论。你不需要对一切都有确定的了解，但我们需要了解 AI 的领域，这样我们才能理解我们是如何来到 TensorFlow.js 这本书中的。

诗人和数学家们几个世纪以来一直在说，人类的思想不过是预先存在的概念的组合。生命的出现被认为是一种类似神的设计；我们都只是由元素“制造”而成。希腊神话中有发明之神赫淮斯托斯创造了能行走和行动如士兵般的自动铜制机器人。基本上，这些就是第一批机器人。机器人和智能的概念已经根植于我们的基础，作为古代传说中的终极和神圣的工艺。巨大的活力战士[塔洛斯](https://oreil.ly/L8Ng2)被著名地编程来守护克里特岛。虽然没有实际的铜制机器人，但这个故事为机械的志向提供了动力。数百年来，机械古代一直被认为是通往看似人类“智能”的途径，几个世纪后，我们开始看到生活在模仿艺术。小时候，我记得去我当地的 Chuck E. Cheese，这是美国一家受欢迎的餐厅，为儿童提供动画音乐表演。我记得曾经相信，就在那一刻，每天播放的木偶驱动的电子音乐会是真实的。我被激发了，这种激发驱使科学家追逐智能。这种火花一直存在，通过故事、娱乐，现在又通过科学传递。

随着能够自主工作和智能地工作的机器的概念在历史上不断发展，我们努力定义这些概念实体。学者们继续研究推理和学习，并出版作品，同时保持他们的术语在“机器”和“机器人”的领域内。机械智能的模仿总是受限于速度和电力的缺乏。

智能的概念在数百年来一直固定在人类思想中，远远超出了机械结构的范围，直到最终机器的诞生，计算机。计算机的诞生与大多数机器一样，都是为整个设备的单一目的而生。随着计算机的出现，一个新术语出现了，用来说明智能的不断增长，这在很大程度上反映了人类的智力。术语 AI 代表人工智能，直到 20 世纪 50 年代才被创造出来。随着计算机变得通用，哲学和学科开始结合。模仿智能的概念从神话中跃入科学研究领域。每一种为人类而设计的电子测量设备都成为计算机的新感官器官，也是电子和智能科学的一个令人兴奋的机会。

在相对较短的时间内，我们已经有了与人类交互并模拟人类行为的计算机。对人类活动的模仿提供了一种我们愿意称之为“智能”的形式。*人工智能*是这些战略行动的总称，无论其复杂程度或技术水平如何。一台可以下井字棋的计算机不必赢才能被归类为 AI。AI 是一个低门槛，不应与人类的一般智能混淆。最简单的代码片段也可以是合法的 AI，而好莱坞电影中的有感情机器的末日也是 AI。

当使用术语 AI 时，它是指来自一种惰性和通常非生物的设备的智能的总称。无论术语的最低门槛是什么，人类，拥有一门研究和一个不断增长的实际用途，都有一个统一的术语和明确的目标。所有被衡量的都被管理，因此人类开始衡量、改进并竞相追求更高级的 AI。

# AI 的历史

AI 框架最初非常具体，但今天不再是这样。正如您可能知道或不知道的那样，TensorFlow.js 作为一个框架的概念可以应用于音乐、视频、图像、统计数据以及我们可以收集的任何数据。但并非总是这样。AI 的实现最初是缺乏任何动态能力的特定领域代码。

互联网上有一些笑话流传，称 AI 只是一堆 IF/THEN 语句的集合，而在我看来，它们并不完全错误。正如我们已经提到的，AI 是对自然智能各种模仿的总称。即使是初学者程序员也是通过解决简单的 AI 问题来学习编程，比如[Ruby Warrior](https://oreil.ly/ze9mi)。这些编程练习教授算法的基础知识，并且需要相对较少的代码。这种简单性的代价是，虽然它仍然是 AI，但它被困在模仿程序员智能的境地。

很长一段时间以来，实施 AI 的主要方法依赖于一个编写 AI 程序的人的技能和哲学，这些技能和哲学直接转化为代码，以便计算机执行指令。数字逻辑正在执行那些传达程序的人的人类逻辑。当然，这是创建 AI 的最大延迟。您需要一个知道如何制造懂得的机器的人，我们受到他们的理解和翻译能力的限制。硬编码的 AI 无法推断超出其指令。这很可能是将任何人类智慧转化为人工智能的最大障碍。如果您想教会机器下棋，您如何教它下棋理论？如果您想告诉程序区分猫和狗，对于幼儿来说是微不足道的事情，您甚至知道从哪里开始编写算法吗？

在 50 年代末和 60 年代初，教师的概念从人类转变为能够阅读原始数据的算法。阿瑟·塞缪尔在一个事件中创造了术语*机器学习*（ML），这个事件使 AI 摆脱了创作者的实际限制。一个程序可以根据数据增长并掌握程序员无法将其转化为代码或自己从未理解的概念。

利用数据来训练程序的应用或功能是一个令人兴奋的抱负。但在计算机需要整个房间而数据远非数字化的时代，这也是一个不可逾越的要求。几十年过去了，计算机才达到了模拟人类信息和架构能力的关键转折点。

在 2000 年代，机器学习研究人员开始使用图形处理单元（GPU）来绕过“CPU 和内存之间的孤立通道”，即*冯·诺伊曼瓶颈*。2006 年，杰弗里·辛顿等人利用数据和神经网络（我们将在下一节中介绍的概念）来理解模式，并让计算机读取手写数字。这是一个以前对于普通计算来说太不稳定和不完美的壮举。*深度学习*能够读取和适应手写的随机性，以正确识别字符的最先进水平超过 98%。在这些发表的论文中，数据作为训练代理的概念从学术界的发表作品跃升到现实。

当 Hinton 陷入制作一个从头开始的学术证明神经网络有效的困境时，“这是什么数字？”问题成为机器学习实践者的一个支柱。这个问题已经成为机器学习框架的关键简单示例之一。TensorFlow.js 有一个[演示](https://oreil.ly/vsANx)，可以在不到两分钟的时间内直接在您的浏览器中解决这个问题。借助 TensorFlow.js 的优势，我们可以轻松构建在网站、服务器和设备上无缝运行的高级学习算法。但这些框架实际上在做什么呢？

AI 的最大目标一直是接近甚至超越人类的能力，98%的手写识别准确率正是如此。Hinton 的研究引发了对这些高效机器学习方法的关注，并创造了行业术语如*深度神经网络*。我们将在下一节详细说明原因，但这是应用机器学习的开始，它开始蓬勃发展，并最终进入了像 TensorFlow.js 这样的机器学习框架。虽然新的基于机器的学习算法不断被创造出来，但一个灵感和术语的来源变得非常清晰。我们可以模拟我们内部的生物系统来创造出一些先进的东西。从历史上看，我们使用自己和我们大脑的皮层（我们大脑的一层）作为结构化训练和智能的灵感来源。

# 神经网络

深度神经网络的概念始终受到我们人类身体的启发。数字节点（有时称为*感知器网络*）模拟我们大脑中的神经元，并像我们自己的突触一样激活，以创建一种平衡的思维机制。这就是为什么神经网络被称为*神经*，因为它模拟了我们大脑的生物化学结构。许多数据科学家厌恶与人脑的类比，但它通常是合适的。通过连接数百万个节点，我们可以构建优雅的深度神经网络，这些网络是用于做出决策的优雅数字机器。

通过增加更多层的神经通路，我们到达了*深度学习*这个术语。深度学习是一个庞大的层层连接的隐藏节点。您会听到这些节点被称为*神经元*、*人工神经元*、*单元*，甚至*感知器*。术语的多样性证明了为机器学习做出贡献的广泛科学家的广泛性。

这整个学习领域只是 AI 的一部分。如果您一直在关注，人工智能有一个称为机器学习的子集或分支，在这个集合内，我们有深度学习的概念。深度学习主要是一类推动机器学习的算法，但不是唯一的一种。请参见图 1-1 以获得这些主要术语的视觉表示。

![AI 水平](img/ltjs_0101.png)

###### 图 1-1\. AI 子领域

就像人类一样，通过示例和数据来适当平衡和构建神经元的迭代教学或“训练”被用来。起初，这些神经网络通常是错误和随机的，但随着他们看到一个又一个数据示例，他们的预测能力“学习”。

但我们的大脑并不直接感知世界。就像计算机一样，我们依赖于被组织成连贯数据发送到我们大脑的电信号。对于计算机来说，这些电信号类似于*张量*，但我们将在第三章中更详细地介绍这一点。TensorFlow.js 体现了研究和科学家们已经确认的所有这些进步。所有这些帮助人体执行的技术都可以包装在一个优化的框架中，这样我们就可以利用几十年来受人体启发的研究。

例如，我们的视觉系统从视网膜开始，使用神经节将光感信息传递到大脑，以激活这些神经元。正如一些人从儿童生物学中记得的那样，我们的视觉中有一些盲点，技术上我们看到的一切都是颠倒的。信号并不是“原样”发送到我们的大脑。这个视觉系统内置了技术，我们在今天的软件中利用了这些技术。

虽然我们都很兴奋地迎接我们的 8K 分辨率电视，你可能认为我们的大脑和视觉仍然超出了现代计算能力，但情况并非总是如此。连接我们眼睛到大脑的视觉信号的神经通道只有大约 10 Mb 的带宽。这相当于上世纪 80 年代初的局域网连接。即使是流媒体宽带连接也需要比这更多的带宽。但我们却能瞬间和迅速地感知一切，对吧？那么这是怎么做到的？我们是如何在这种过时的硬件上获得优越的信号的？答案是我们的视网膜在将数据发送到我们深度连接的神经网络之前对数据进行了压缩和“特征化”。这就是我们开始在计算机上做的事情。

卷积神经网络（CNN）在视觉数据上的工作方式与我们的眼睛和大脑一起压缩和激活我们的神经通路的方式相同。您将在第十章中进一步了解并编写自己的 CNN。我们每天都在了解我们的工作方式，并将这数百万年的进化直接应用于我们的软件。虽然了解这些 CNN 如何工作对您很有好处，但自己编写它们太学术化了。TensorFlow.js 带有您需要处理图像的卷积层。这是利用机器学习框架的基本好处。

你可以花费数年时间阅读和研究使计算机视觉、神经网络和人类有效运作的所有独特技巧和黑客技术。但我们生活在一个这些根源已经有时间生长、分支并最终结果的时代：这些先进的概念是可访问的，并内置在我们周围的服务和设备中。

# 今天的人工智能

今天我们使用这些最佳实践与人工智能相结合，以增强机器学习。卷积用于边缘检测，对某些区域的关注超过其他区域，甚至为一个单一项目提供多摄像头输入，为我们提供了一个在云机器训练 AI 的光纤服务器农场中的预先处理的数据宝库。

2015 年，AI 算法开始在某些视觉任务中胜过人类。正如你可能在新闻中听说的那样，AI 已经在[癌症检测](https://oreil.ly/ZCz0B)方面超越了人类，甚至在识别[法律缺陷](https://oreil.ly/9dW3S)方面超过了美国顶级律师。正如数字信息一样，AI 在几秒钟内完成了这些任务，而不是几小时。AI 的“魔力”令人惊叹。

人们一直在寻找将人工智能应用于他们的项目的新颖有趣的方法，甚至创造全新的行业。

AI 已被应用于：

+   生成写作、音乐和视觉方面的新内容

+   推荐有用的内容

+   取代简单的统计模型

+   从数据中推断法则

+   可视化分类器和识别器

所有这些突破都是深度学习的方面。今天我们拥有必要的硬件、软件和数据，可以通过深度机器学习网络实现突破性的变革。每天，社区和《财富》500 强公司都在人工智能领域发布新的数据集、服务和架构突破。

凭借您手头的工具和本书中的知识，您可以轻松地创造以前从未见过的东西，并将其带到网络上。无论是为了娱乐、科学还是财富，您都可以为任何现实世界问题或业务创建一个可扩展的智能解决方案。

如果说目前机器学习的问题是它是一种新的超级能力，而世界是广阔的。我们没有足够的例子来理解在 JavaScript 中拥有人工智能的全部好处。当电池寿命有了显著改善时，它们为更强大的手机和相机等设备带来了一个全新的世界，这些设备可以在单次充电的情况下持续数月。这一突破带来了数年内市场上无数新产品。机器学习不断取得突破，带来了新技术的飞速发展，我们甚至无法澄清或认识到这种洪流的加速。本书将重点介绍具体和抽象的例子，以便您可以使用 TensorFlow.js 应用实用解决方案。

# 为什么选择 TensorFlow.js？

您有选择。您可以从头开始编写自己的机器学习模型，也可以选择各种编程语言中的任何现有框架。即使在 JavaScript 领域，已经存在竞争框架、示例和选项。是什么使 TensorFlow.js 能够处理和承载今天的人工智能？

## 重要支持

TensorFlow.js 由 Google 创建和维护。我们将在第二章中更多地介绍这一点，但值得注意的是，世界上一些最优秀的开发人员已经齐心协力使 TensorFlow.js 成为可能。这也意味着，即使没有社区的努力，TensorFlow.js 也能够与最新和最伟大的突破性发展一起工作。

与其他基于 JavaScript 的机器学习库和框架实现不同，TensorFlow.js 支持经过优化和测试的 GPU 加速代码。这种优化传递给您和您的机器学习项目。

## 在线就绪

大多数机器学习解决方案都限制在一个非常定制的机器上。如果您想要创建一个网站来分享您的突破性技术，人工智能通常被锁在 API 后面。虽然在 Node.js 中运行 TensorFlow.js 是完全可行的，但在浏览器中直接运行 TensorFlow.js 也是可行的。这种无需安装的体验在机器学习领域是罕见的，这使您能够无障碍地分享您的创作。您可以版本化并访问丰富的互动世界。

## 离线就绪

JavaScript 的另一个好处是它可以在任何地方运行。代码可以保存到用户的设备上，如渐进式 Web 应用程序（PWA）、Electron 或 React Native 应用程序，然后可以在没有任何互联网连接的情况下始终如一地运行。不言而喻的是，与托管的 AI 解决方案相比，这也提供了显著的速度和成本提升。在本书中，您将发现许多完全存在于浏览器中的例子，这些例子可以使您和您的用户免受延迟和托管成本的困扰。

## 隐私

人工智能可以帮助用户识别疾病、税收异常和其他个人信息。通过互联网发送敏感数据可能存在危险。设备上的结果保留在设备上。甚至可以训练一个人工智能并将结果存储在用户的设备上，而没有任何信息离开浏览器的安全性。

## 多样性

应用 TensorFlow.js 在机器学习领域和平台上具有强大而广泛的影响。TensorFlow.js 可以利用 Web Assembly 在 CPU 或 GPU 上运行，适用于性能更强大的机器。如今的机器学习 AI 领域的光谱对于新手来说是一个重要而庞大的新术语和复杂性的世界。拥有一个可以处理各种数据的框架是有用的，因为它保持了您的选择。

精通 TensorFlow.js 使您能够将您的技能应用于支持 JavaScript 的各种平台（请参阅图 1-2）。

![TFJS 的多平台演示。](img/ltjs_0102.png)

###### 图 1-2\. TensorFlow.js 平台

使用 TensorFlow.js，您可以自由选择、原型设计和部署您的技能到各种领域。为了充分利用您的机器学习自由度，您需要了解一些术语，这些术语可以帮助您进入机器学习领域。

# 机器学习类型

许多人将机器学习分为三类，但我认为我们需要将所有机器学习视为四个重要元素：

+   监督

+   无监督

+   半监督

+   强化

每个元素都值得写成一本书。接下来的简短定义只是为了让您熟悉这个领域中会听到的术语。

## 快速定义：监督学习

在这本书中，我们将重点放在最常见的机器学习类别上，即*监督机器学习*（有时简称为*监督学习*或简称为*监督*）。监督机器学习简单地意味着我们对用于训练机器的每个问题都有一个答案。换句话说，我们的数据是有标签的。因此，如果我们试图教会机器区分一张照片是否包含鸟类，我们可以立即根据 AI 的答案是对还是错来评分。就像使用 Scantron 一样，我们有答案。但与 Scantron 不同的是，由于这是概率数学，我们还可以确定答案有多错。

如果 AI 对一张鸟类照片有 90%的确定性，虽然它回答正确，但还可以提高 10%。这突显了 AI 的“训练”方面，即通过即时数据驱动的满足感。

###### 提示

如果你没有成百上千个现成的标记问题和答案，也不用担心。在这本书中，我们要么为您提供标记数据，要么向您展示如何自己生成数据。

## 快速定义：无监督学习

无监督学习不需要我们有答案。我们只需要问题。无监督机器学习是理想的，因为世界上大多数信息都没有标签。这类机器学习侧重于机器可以从未标记数据中学习和报告的内容。虽然这个主题可能有点令人困惑，但人类每天都在执行它！例如，如果我给你一张我的花园照片，并问你我拥有多少种不同类型的植物，你可以告诉我答案，而不必知道每株植物的属种。这有点类似于我们如何理解自己的世界。很多无监督学习都集中在对大量数据进行分类上。

## 快速定义：半监督学习

大多数时候，我们并不是在处理 100%未标记的数据。回到之前的花园例子，你可能不知道每株植物的属种，但也不完全无法对植物进行分类为 A 和 B。你可能告诉我，我有十株植物，其中三株是花，七株是草本植物。拥有少量已知标签可以帮助很多，当今的研究在半监督突破方面取得了巨大进展！

您可能听说过术语*生成网络*或*生成对抗网络*（GANs）。这些流行的 AI 构造在许多 AI 新闻文章中被提及，并源自半监督学习策略。生成网络是根据我们希望网络创建的示例进行训练的，通过半监督方法，可以构建新的示例。生成网络非常擅长从少量标记数据中创建新内容。流行的 GAN 的例子通常有自己的网站，比如[*https://thispersondoesnotexist.com*](https://thispersondoesnotexist.com)，越来越受欢迎，创意人员正在享受半监督输出带来的乐趣。

###### 注意

GAN 在生成新内容方面发挥了重要作用。虽然流行的 GAN 是半监督的，但 GAN 的更深层概念并不局限于半监督网络。人们已经将 GAN 调整为适用于我们定义的每种学习类型。

## 快速定义：强化学习

解释强化学习最简单的方法是展示它在处理更真实的活动时是必需的，而不是之前的假设构造。

例如，如果我们在下棋时，我开始游戏时移动一个兵，那是一个好动作还是坏动作？或者如果我想让一个机器人把球踢进篮筐，它开始迈步，这是好还是坏？就像对待人类一样，答案取决于结果。这是为了最大奖励而采取的一系列动作，不总是有一个动作会产生一个结果。训练机器人首先迈步或首先看很重要，但可能不如在其他关键时刻所做的事情重要。而这些关键时刻都是由奖励作为强化来驱动的。

如果我要教一个 AI 玩《超级马里奥兄弟》，我想要高分还是快速胜利？奖励教会 AI 什么组合动作是最优化的以实现目标。强化学习（RL）是一个不断发展的领域，经常与其他形式的人工智能结合以培养最大的结果。

## 信息过载

对于刚提到的机器学习的许多应用，感到惊讶是可以的。在某种程度上，这就是为什么我们需要像 TensorFlow.js 这样的框架。我们甚至无法理解所有这些奇妙系统的用途及其未来几十年的影响！在我们理解这一切的同时，人工智能和机器学习的时代已经到来，我们将成为其中的一部分。监督学习是进入人工智能所有好处的一个很好的第一步。

我们将一起探讨一些最令人兴奋但实用的机器学习用途。在某些方面，我们只会触及表面，而在其他方面，我们将深入探讨它们的工作原理。以下是我们将涵盖的一些广泛类别。这些都是监督学习概念：

+   图像分类

+   自然语言处理（NLP）

+   图像分割

本书的主要目标之一是，虽然你可以理解这些类别的概念，但不会受到限制。我们将倾向于实验和实践科学。有些问题可以通过过度工程解决，而有些问题可以通过数据工程解决。AI 和机器学习的思维是看到、识别和创建新工具的关键。

# 人工智能无处不在

我们正在进入一个人工智能渗透到一切的世界。我们的手机现在已经加速到深度学习硬件。摄像头正在应用实时人工智能检测，而在撰写本文时，一些汽车正在街上行驶，没有人类驾驶员。

在过去的一年中，我甚至注意到我的电子邮件已经开始自动写作，提供了一个“按 Tab 键完成”选项来完成我的句子。这个功能，比我最初写的任何东西更清晰更简洁，这是一个显著的可见成就，超过了多年来一直在同一个收件箱中保护我们免受垃圾邮件的被遗忘的机器学习 AI。

随着每个机器学习计划的展开，新的平台变得需求量大。我们正在将模型推向边缘设备，如手机、浏览器和硬件。寻找新语言来传承是合理的。很快搜索就会以 JavaScript 为明显选择。

# 框架提供的一次导览

机器学习是什么样子？以下是一个准确的描述，会让博士生因其简洁而感到不安。

在正常的代码中，人类直接编写代码，计算机读取和解释该代码，或者读取其某种派生形式。现在我们处于一个人类不编写算法的世界，那么实际发生了什么？算法从哪里来？

这只是一个额外的步骤。一个人编写算法的训练器。在框架的帮助下，甚至从头开始，一个人在代码中概述了问题的参数、所需的结构和要学习的数据的位置。现在，机器运行这个程序训练程序，不断地编写一个不断改进的算法作为解决问题的解决方案。在某个时候，您停止这个程序，取出最新的算法结果并使用它。

*就是这样！*

算法比用于创建它的数据要小得多。数千兆字节的电影和图像可以用来训练一个机器学习解决方案数周，所有这些只是为了创建几兆字节的数据来解决一个非常具体的问题。

最终的算法本质上是一组数字，这些数字平衡了人类程序员所确定的结构。这组数字及其相关的神经图通常被称为*模型*。

您可能已经在技术文章中看到这些图表，它们被绘制为从左到右的一组节点，就像图 1-3。

![神经网络](img/ltjs_0103.png)

###### 图 1-3. 密集连接神经网络的示例

我们的框架 TensorFlow.js 处理了指定模型结构或架构的 API，加载数据，通过我们的机器学习过程传递数据，并最终调整机器以更好地预测下次给定输入的答案。这就是 TensorFlow.js 真正的好处所在。我们只需要担心正确调整框架以解决问题，并保存生成的模型。

## 什么是模型？

当您在 TensorFlow.js 中创建一个神经网络时，它是所需神经网络的代码表示。该框架为每个神经元智能选择随机值生成一个图。此时，模型的文件大小通常是固定的，但内容会发展。当通过将数据传递给一个未经训练的具有随机值的网络进行预测时，我们得到的答案通常与正确答案相去甚远，就像纯随机机会一样。我们的模型没有经过任何数据训练，所以在工作中表现糟糕。因此，作为开发人员，我们编写的代码是完整的，但未经训练的结果很差。

一旦训练迭代发生了相当长的一段时间，神经网络的权重就会被评估然后调整。速度，通常被称为*学习率*，会影响结果。在以学习率为步长进行数千次这样的小步骤后，我们开始看到一个不断改进的机器，我们正在设计一个成功概率远远超出原始机器的模型。我们已经摆脱了随机性，收敛于使神经网络工作的数字！分配给给定结构中的神经元的那些数字就是训练好的模型。

TensorFlow.js 知道如何跟踪所有这些数字和计算图，所以我们不必，它还知道如何以适当和可消费的格式存储这些信息。

一旦您获得了这些数字，我们的神经网络模型就可以停止训练，只用来进行预测。在编程术语中，这已经变成了一个简单的函数。数据进去，数据出来。

通过神经网络传递数据看起来很像机会，如图 1-4 所示，但在计算世界中，这是一个平衡概率和排序的精密机器，具有一致和可重复的结果。数据被输入到机器中，然后得到一个概率性的结果。

![神经网络分类](img/ltjs_0104.png)

###### 图 1-4. 一个平衡的网络隐喻

在下一章中，我们将尝试导入并使用一个完全训练好的模型进行预测。我们将利用数小时的训练来在微秒内得到智能分析。

# 在本书中

本书的结构使您可以将其收起放在度假中，一旦找到您的小天堂，就可以跟着书本阅读，学习概念，并审查答案。图像和截图应该足以解释 TensorFlow.js 的深层原理。

然而，要真正掌握这些概念，您需要超越简单地阅读本书。随着每个概念的展开，您应该编写代码，进行实验，并在实际计算机上测试 TensorFlow.js 的边界。对于那些作为机器学习领域的新手的人来说，重要的是您巩固您第一次看到的术语和工作流程。花时间逐步学习本书中的概念和代码。

## 相关代码

在整本书中，有可运行的源代码来说明 TensorFlow.js 的课程和功能。在某些情况下提供整个源代码，但在大多数情况下，打印的代码将仅限于重要部分。建议您立即下载与本书相匹配的源代码。即使您计划在示例旁边从头编写代码，您可能会遇到的小配置问题已经在相关代码中得到解决并可引用。

您可以在[*https://github.com/GantMan/learn-tfjs*](https://github.com/GantMan/learn-tfjs)看到 GitHub 源页面。

如果您对 GitHub 和 Git 不熟悉，可以简单地下载最新的项目源代码的单个 ZIP 文件并引用它。

您可以从[*https://github.com/GantMan/learn-tfjs/archive/master.zip*](https://github.com/GantMan/learn-tfjs/archive/master.zip)下载源 ZIP 文件。

源代码结构化以匹配每个章节。您应该能够在具有相同名称的文件夹中找到所有章节资源。在每个章节文件夹中，您将找到最多四个包含课程信息的文件夹。当您运行第一个 TensorFlow.js 代码时，将在第二章中进行审查。现在，请熟悉每个文件夹的目的，以便选择最适合您学习需求的示例代码。

### 额外文件夹

这个文件夹包含章节中引用的任何额外材料，包括文档或其他参考资料。这些部分的材料是每章的有用文件。

### 节点文件夹

这个文件夹包含了章节代码的 Node.js 特定实现，用于基于服务器的解决方案。该文件夹可能包含其中几个特定项目。Node.js 项目将安装一些额外的软件包，以简化实验过程。本书的示例项目使用以下内容：

`nodemon`

Nodemon 是一个实用程序，将监视源中的任何更改并自动重新启动服务器。这样可以保存文件并立即查看其相关更新。

`ts-node`

TypeScript 有很多选项，最明显的是强类型。然而，为了易于理解，本书专注于 JavaScript 而不是 TypeScript。`ts-node`模块用于支持 ECMAScript。您可以在这些节点示例中编写现代 JavaScript 语法，通过 TypeScript，代码将正常工作。

这些依赖项在*package.json*文件中标识。Node.js 示例用于说明使用 TensorFlow.js 的服务器解决方案，通常不需要在浏览器中打开。

要运行这些示例，请使用 Yarn 或 Node Package Manager（NPM）安装依赖项，然后执行启动脚本：

```js
# Install dependencies with NPM
$ npm i
# Run the start script to start the server
$ npm run start
# OR use yarn to install dependencies
$ yarn
# Run the start script to start the server
$ yarn start
```

启动服务器后，您将在终端中看到任何控制台日志的结果。查看结果后，您可以使用 Ctrl+C 退出服务器。

### 简单文件夹

这个文件夹将包含没有使用 NPM 的解决方案。所有资源都简单地放在独立的 HTML 文件中进行服务。这绝对是最简单的解决方案，也是最常用的。这个文件夹很可能包含最多的结果。

### web 文件夹

如果您熟悉基于客户端的 NPM Web 应用程序，您将会对`web`文件夹感到舒适。这个文件夹很可能包含其中的几个具体项目。`web`文件夹示例是使用[Parcel.js](https://parceljs.org)打包的。这是一个用于 Web 项目的快速多核打包工具。Parcel 提供热模块替换（HMR），因此您可以保存文件并立即看到页面反映您的代码更改，同时还提供友好的错误日志记录和访问 ECMAScript。

要运行这些示例，请使用 Yarn 或 NPM 安装依赖项，然后执行启动脚本：

```js
# Install dependencies with NPM
$ npm i
# Run the start script to start the server
$ npm run start
# OR use yarn to install dependencies
$ yarn
# Run the start script to start the server
$ yarn start
```

运行打包程序后，将打开一个网页，使用您的默认浏览器访问该项目的本地 URL。

###### 提示

如果项目使用像照片这样的资源，那么该项目的根文件夹中将存在一个*credit.txt*文件，以正确归功于摄影师和来源。

## 章节部分

每一章都从确定章节目标开始，然后立即深入讨论。在每一章的末尾，您将看到一个章节挑战，这是一个资源，让您立即应用您刚学到的知识。每个挑战的答案可以在附录 B 中找到。

最后，每一章都以一组发人深省的问题结束，以验证您是否已内化了本章的信息。建议您尽可能通过代码验证答案，但答案也会在附录 A 中提供给您。

## 常见的 AI/ML 术语

您可能会想，“为什么模型不只是称为函数？模型在编程中已经有了意义，不需要另一个！”事实是这源自机器学习起源的问题。原始数据问题根植于统计学。统计模型将模式识别为样本数据的统计假设，因此我们从这些示例的数学运算中得到的产品是一个机器学习模型。机器学习术语通常会大量反映发明它的科学家的领域和文化。

数据科学伴随着大量的数学术语。我们将在整本书中看到这一主题，并且我们将为每个术语找出原因。有些术语立即就有意义，有些与现有的 JavaScript 和框架术语冲突，有些新术语与其他新术语冲突！命名事物是困难的。我们将尽力以易记的方式解释一些关键术语，并在途中详细说明词源。TensorFlow 和 TensorFlow.js 文档为开发人员提供了大量新词汇。阅读以下机器学习术语，看看您是否能掌握这些基本术语。如果不能，没关系。随着我们的进展，您可以随时回到本章并参考这些定义。

### 训练

训练是通过让机器学习算法审查数据并改进其数学结构以使其在未来做出更好的预测的过程。

TensorFlow.js 提供了几种方法来训练和监控训练模型，无论是在机器上还是在客户端浏览器上。

> 例如，“请不要触碰我的电脑，它已经在我的最新的空气弯曲算法上训练了三天。”

### 训练集

有时被称为“训练数据”，这是您将向算法展示的数据，让它从中学习。你可能会想，“这不就是我们拥有的所有数据吗？”答案是“不是”。

通常，大多数机器学习模型可以从它们以前见过的示例中学习，但测试并不能保证我们的模型可以推广到识别它以前从未见过的数据。重要的是，我们用来训练人工智能的数据要与验证和核实分开。

> 例如，“我的模型一直将热狗识别为三明治，所以我需要向我的训练集中添加更多照片。”

### 测试集

为了测试我们的模型是否能够对从未见过的数据进行处理，我们必须保留一些数据进行测试，并且永远不让我们的模型从中学习。这通常被称为“测试集”或“测试数据”。这个集合帮助我们测试我们是否已经创建了一个可以推广到现实世界新问题的东西。测试集通常比训练集要小得多。

> 例如，“我确保测试集是我们试图训练模型解决的问题的一个很好的代表。”

### 验证集

即使您还没有达到需要它的水平，这个术语也是很重要的。正如您经常会听到的，训练有时可能需要几小时、几天甚至几周。启动一个长时间运行的过程，只是回来发现您构建了错误的结构，必须重新开始，这有点令人担忧！虽然在本书中我们可能不会遇到任何这些大规模训练的需求，但这些情况可能需要一组数据进行更快的测试。当这与您的训练数据分开时，它是用于验证的“留出法”。基本上，这是一种实践，在让您的模型在昂贵的基础设施上训练或花费更长时间之前，将一小部分训练数据保留下来进行验证测试。这种调整和验证是您的验证集。

有很多方法可以选择、切片、分层甚至折叠您的验证集。这涉及到一种超出本书范围的科学，但当您讨论、阅读和提升自己的大型数据集时，了解这些知识是很有用的。

TensorFlow.js 在训练过程中有整个训练参数，用于识别和绘制验证结果。

> 例如，“我已经划分了一个小的验证集，在构建模型架构时使用。”

### 张量

我们将在第三章中详细介绍张量，但值得注意的是，张量是优化的数据结构，允许 GPU 和 Web Assembly 加速巨大的人工智能/机器学习计算集。张量是数据的数值持有者。

> 例如，“我已经将您的照片转换为灰度张量，以查看我们可以获得什么样的速度提升。”

### 归一化

归一化是将输入值缩放到更简单领域的操作。当一切都变成数字时，数字的稀疏性和数量的差异可能会导致意想不到的问题。

例如，房屋的大小和房屋中的浴室数量都会影响价格，但它们通常以完全不同的数字单位进行测量。并非所有事物都以相同的度量标准来衡量，虽然人工智能可以适应这些模式中的波动，但一个常见的技巧是简单地将数据缩放到相同的小领域。这样可以让模型更快地训练并更容易地找到模式。

> 例如，“我已经对房价和浴室数量进行了一些归一化，这样我们的模型可以更快地找到两者之间的模式。”

### 数据增强

在照片编辑软件中，我们可以拍摄图像并操纵它们，使其看起来像完全不同环境中的同一物体。这种方法有效地创建了一张全新的照片。也许您想要将您的标志放在建筑物的一侧或者压印在名片上。如果我们试图检测您的标志，原始照片和一些编辑过的版本将有助于我们的机器学习训练数据。

通常情况下，我们可以从原始数据中创建符合我们模型目标的新数据。例如，如果我们的模型将被训练来检测人脸，一个人的照片和一个镜像的人的照片都是有效的，而且明显不同的照片！

TensorFlow.js 有专门用于数据增强的库。我们将在本书的后面看到增强的数据。

> 例如，“我们通过镜像所有南瓜进行了一些数据增强，以扩大我们的训练集。”

### 特征和特征化

我们之前提到过特征化，当我们谈到眼睛如何将最重要的信息发送到大脑时。我们在机器学习中也是这样做的。如果我们试图制作一个猜测房子价值的 AI，那么我们必须确定哪些输入是有用的，哪些输入是噪音。

房子上的数据不缺乏，从砖块的数量到装饰线。如果你经常看家装电视节目，你会知道识别房子的大小、年龄、浴室数量、厨房最后一次更新的日期和社区是明智的。这些通常是识别房价的关键特征，你会更关心提供给模型这些信息，而不是一些琐碎的东西。特征化是从所有可能的数据中选择这些特征作为输入的过程。

如果我们决定把所有可能的数据都放进去，我们就给了我们的模型找到新模式的机会，但代价是时间和精力。没有理由选择像草叶的数量、房子的气味或正午的自然光线这样的特征，即使我们有这些信息或者我们觉得这对我们很重要。

即使我们选择了我们的特征，仍然会有错误和异常值会减慢实用机器学习模型的训练。有些数据只会让预测模型更成功的指针移动，选择明智的特征会使快速训练的智能 AI。

> 例如，“我相当确定计算感叹号的数量是检测这些营销邮件的关键特征。”

# 章节回顾

在这一章中，我们已经掌握了总称 AI 的术语和概念。我们也触及了我们将在本书中涵盖的关键原则。理想情况下，你现在对机器学习中必不可少的术语和结构更有信心了。

## 复习问题

让我们花点时间确保你完全掌握了我们提到的概念。花点时间回答以下问题：

1.  你能给出机器学习的充分定义吗？

1.  如果一个人想到了一个机器学习项目的想法，但是他们没有标记的数据，你会推荐什么？

1.  什么样的机器学习对打败你最喜欢的视频游戏有用？

1.  机器学习是唯一的 AI 形式吗？

1.  一个模型是否保存了用于使其工作的所有训练示例数据？

1.  机器学习数据是如何分解的？

这些练习的解决方案可以在附录 A 中找到。

¹ 编程语言统计：[*https://octoverse.github.com*](https://octoverse.github.com)

² *人工智能*是由约翰·麦卡锡在 1956 年首次学术会议上创造的。
