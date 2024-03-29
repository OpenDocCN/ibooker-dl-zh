# 第十四章：设计你自己的 TinyML 应用程序

到目前为止，我们已经探讨了重要领域如音频、图像和手势识别的现有参考应用。如果你的问题类似于其中一个示例，你应该能够调整训练和部署过程，但如果不明显如何修改我们的示例以适应呢？在本章和接下来的章节中，我们将介绍为一个没有易于起点的问题构建嵌入式机器学习解决方案的过程。你对示例的经验将为创建自己的系统奠定良好基础，但你还需要了解更多关于设计、训练和部署新模型的知识。由于我们平台的约束非常严格，我们还花了很多时间讨论如何进行正确的优化，以适应存储和计算预算，同时不会错过准确性目标。你肯定会花费大量时间尝试理解为什么事情不起作用，因此我们涵盖了各种调试技术。最后，我们探讨了如何为用户的隐私和安全建立保障措施。

# 设计过程

训练模型可能需要几天或几周的时间，引入新的嵌入式硬件平台也可能非常耗时——因此，任何嵌入式机器学习项目最大的风险之一是在你有可用的东西之前时间耗尽。减少这种风险的最有效方法是尽早回答尽可能多的未解决问题，通过规划、研究和实验。对训练数据或架构的每次更改很容易涉及一周的编码和重新训练，部署硬件更改会在整个软件堆栈中产生连锁反应，需要大量重写先前有效的代码。你可以在开始时做的任何事情，以减少后续开发过程中所需更改的数量，可以节省你本来会花在这些更改上的时间。本章重点介绍了我们建议用于在编写最终应用程序之前回答重要问题的一些技术。

# 你需要微控制器，还是更大的设备可以工作？

你真正需要回答的第一个问题是，你是否需要嵌入式系统的优势，或者至少对电池寿命、成本和尺寸的要求可以放松，至少对于一个初始原型来说。在具有完整现代操作系统（如 Linux）的系统上编程比在嵌入式世界中开发要容易得多（也更快）。你可以以低于 25 美元的价格获得像树莓派这样的完整桌面级系统，以及许多外围设备，如摄像头和其他传感器。如果你需要运行计算密集型的神经网络，NVIDIA 的 Jetson 系列板卡起价为 99 美元，并带来了强大的软件堆栈，体积小。这些设备的最大缺点是它们会消耗几瓦的电力，使它们的电池寿命在几小时或几天左右，具体取决于能量存储的物理尺寸。只要延迟不是一个硬性约束，你甚至可以启动尽可能多的强大云服务器来处理神经网络工作负载，让客户端设备处理界面和网络通信。

我们坚信能够在任何地方部署的力量，但如果你试图确定一个想法是否有效，我们强烈建议尝试使用一个易于快速实验的设备进行原型设计。开发嵌入式系统非常痛苦，所以在深入之前尽可能梳理出应用程序的真正需求，你成功的机会就越大。

举个实际的例子，想象一下你想要建造一个设备来帮助监测羊的健康。最终产品需要能够在没有良好连接的环境中运行数周甚至数月，因此必须是嵌入式系统。然而，在开始时，你不想使用这种难以编程的设备，因为你还不知道关键细节，比如你想要运行哪些模型，需要哪些传感器，或者根据你收集的数据需要采取什么行动，而且你还没有任何训练数据。为了启动你的工作，你可能会想找一个友好的农民，他们有一小群在易于接近的地方放牧的羊。你可以组装一个树莓派平台，每晚从每只被监测的羊身上取下来自己充电，然后建立一个覆盖放牧区域范围的室外 WiFi 网络，这样设备就可以轻松地与网络通信。显然，你不能指望真正的客户去做这种麻烦的事情，但通过这种设置，你将能够回答许多关于你需要构建什么的问题，尝试新的模型、传感器和形态因素将比嵌入式版本快得多。

微控制器很有用，因为它们可以按照其他硬件无法做到的方式进行扩展。它们便宜、小巧，几乎不需要能量，但这些优势只有在实际需要扩展时才会发挥作用。如果可以的话，推迟处理扩展的问题，直到绝对必要，这样你就可以确信你正在扩展正确的东西。

# 理解可能性

很难知道深度学习能够解决什么问题。我们发现一个非常有用的经验法则是，神经网络模型擅长处理人们可以“眨眼间”解决的任务。我们直觉上似乎能够瞬间识别物体、声音、单词和朋友，而这些正是神经网络可以执行的任务。同样，DeepMind 的围棋解决算法依赖于一个卷积神经网络，它能够查看棋盘并返回每个玩家处于多强势位置的估计。然后，该系统的长期规划部分是基于这些基础组件构建的。

这是一个有用的区分，因为它在不同种类的“智能”之间划定了界限。神经网络并不自动具备规划或像定理证明这样的高级任务的能力。它们更擅长接收大量嘈杂和混乱的数据，并稳健地发现模式。例如，神经网络可能不是指导牧羊犬如何引导羊群穿过大门的好解决方案，但它很可能是利用各种传感器数据（如体温、脉搏和加速度计读数）来预测羊羊是否感到不适的最佳方法。我们几乎无意识地执行的判断更有可能被深度学习覆盖，而不是需要明确思考的问题。这并不意味着那些更抽象的问题不能通过神经网络得到帮助，只是它们通常只是一个更大系统的组成部分，该系统使用它们的“本能”预测作为输入。

# 跟随他人的脚步

在研究领域，“文献回顾”是一个相对宏大的名字，用于指阅读与你感兴趣的问题相关的研究论文和其他出版物。即使你不是研究人员，当涉及深度学习时，这也是一个有用的过程，因为有很多有用的关于尝试将神经网络模型应用于各种挑战的账户，如果你能从他人的工作中获得一些启示，你将节省很多时间。理解研究论文可能是具有挑战性的，但最有用的是了解人们在类似问题上使用了什么样的模型，以及是否有任何现有的数据集可以使用，考虑到收集数据是机器学习过程中最困难的部分之一。

例如，如果你对机械轴承的预测性维护感兴趣，你可以在 arxiv.org 上搜索[“深度学习预测性维护轴承”](https://oreil.ly/xljQN)，这是机器学习研究论文最受欢迎的在线主机。截至本文撰写时，排名第一的结果是来自 2019 年的一篇综述论文，[“用于轴承故障诊断的机器学习和深度学习算法：综合回顾”](https://oreil.ly/-dqy7)由 Shen Zhang 等人撰写。从中，你将了解到有一个名为[Case Western Reserve University 轴承数据集](https://oreil.ly/q2_79)的标记轴承传感器数据的标准公共数据集。拥有现有的数据集非常有帮助，因为它将帮助你在甚至还没有从自己的设置中收集读数之前就进行实验。还有对已经用于该问题的不同模型架构的很好概述，以及对它们的优势、成本和整体结果的讨论。

# 寻找一些类似的模型进行训练

在你对模型架构和训练数据有了一些想法之后，值得花一些时间在训练环境中进行实验，看看在没有资源限制的情况下你能够取得什么样的结果。本书专注于 TensorFlow，因此我们建议你找到一个示例 TensorFlow 教程或脚本（取决于你的经验水平），将其运行起来，然后开始适应你的问题。如果可能的话，可以参考本书中的训练示例，因为它们还包括部署到嵌入式平台所需的所有步骤。

一个思考哪种模型可能有效的好方法是查看传感器数据的特征，并尝试将其与教程中的类似内容进行匹配。例如，如果你有来自轮轴承的单通道振动数据，那将是一个相对高频的时间序列，与麦克风的音频数据有很多共同之处。作为一个起点，你可以尝试将所有轴承数据转换为*.wav*格式，然后将其输入到[语音训练过程](https://oreil.ly/dG9gQ)中，而不是标准的语音命令数据集，带有适当的标签。然后你可能需要更多地定制这个过程，但希望至少能得到一个有些预测性的模型，并将其用作进一步实验的基准。类似的过程也适用于将手势教程适应到任何基于加速度计的分类问题，或者为不同的机器视觉应用重新训练人员检测器。如果在本书中没有明显的示例可供参考，那么搜索展示如何使用 Keras 构建你感兴趣的模型架构的教程是一个很好的开始。

# 查看数据

大部分机器学习研究的重点是设计新的架构；对于训练数据集的覆盖并不多。这是因为在学术界，通常会给你一个固定的预生成训练数据集，你的竞争重点是你的模型在这个数据集上的得分如何与其他人相比。在研究之外，我们通常没有现成的数据集来解决问题，我们关心的是我们为最终用户提供的体验，而不是在一个固定数据集上的得分，因此我们的优先事项变得非常不同。

其中一位作者写了一篇[博客文章](https://oreil.ly/ghEbc)更详细地介绍了这一点，但总结是你应该期望花费更多的时间收集、探索、标记和改进你的数据，而不是在模型架构上。你投入的时间回报会更高。

在处理数据时，我们发现一些常见的技术非常有用。其中一个听起来非常明显但我们经常忘记的技巧是：查看你的数据！如果你有图像，将它们下载到按标签排列的文件夹中，并在本地机器上浏览它们。如果你在处理音频文件，也是同样的操作，并且听一些音频文件的选段。你会很快发现各种你没有预料到的奇怪和错误，比如标记为美洲豹的汽车被标记为美洲豹猫，或者录音中声音太微弱或被裁剪导致部分单词被切断。即使你只有数值数据，查看逗号分隔值（CSV）文本文件中的数字也会非常有帮助。过去我们发现了一些问题，比如许多数值达到传感器的饱和限制并达到最大值，甚至超出，或者灵敏度太低导致大部分数据被挤压到一个过小的数值范围内。你可以在数据分析中更加深入，你会发现像 TensorBoard 这样的工具对于聚类和其他数据集中发生的可视化非常有帮助。

另一个要注意的问题是训练集不平衡。如果你正在对类别进行分类，不同类别在训练输入中出现的频率将影响最终的预测概率。一个容易陷入的陷阱是认为网络的结果代表真实概率——例如，“是”得分为 0.5 意味着网络预测说话的单词是“是”的概率为 50%。事实上，这种关系更加复杂，因为训练数据中每个类别的比例将控制输出值，但应用程序真实输入分布中每个类别的先验概率是需要了解真实概率的。举个例子，想象一下在 10 种不同物种的鸟类图像分类器上进行训练。如果你将其部署在南极，看到一个指示你看到了鹦鹉的结果会让你非常怀疑；如果你在亚马逊看视频，看到企鹅同样会让你感到惊讶。将这种领域知识融入训练过程可能是具有挑战性的，因为你通常希望每个类别的样本数量大致相等，这样网络才能平等“关注”每个类别。相反，通常在模型推断运行后会进行一个校准过程，根据先验知识对结果进行加权。在南极的例子中，你可能需要一个非常高的阈值才能报告一只鹦鹉，但对企鹅的阈值可能要低得多。

# 奥兹巫师

我们最喜欢的机器学习设计技术之一实际上并不涉及太多技术。工程中最困难的问题是确定需求是什么，很容易花费大量时间和资源在实际上对于一个问题并不起作用的东西上，特别是因为开发一个机器学习模型的过程需要很长时间。为了澄清需求，我们强烈推荐[绿野仙踪方法](https://oreil.ly/Omr6N)。在这种情况下，你创建一个系统的模拟，但是不是让软件做决策，而是让一个人作为“幕后之人”。这让你在经历耗时的开发周期之前测试你的假设，以确保在将它们融入设计之前对规格进行了充分测试。

这在实践中是如何运作的呢？想象一下，你正在设计一个传感器，用于检测会议室内是否有人，如果没有人在房间里，它会调暗灯光。与构建和部署运行人员检测模型的无线微控制器不同，采用绿野仙踪方法，你会创建一个原型，只需将实时视频传送给一个坐在附近房间里的人，他手里有一个控制灯光的开关，并有指示在没有人可见时将其调暗。你很快会发现可用性问题，比如如果摄像头没有覆盖整个房间，灯光会在有人仍然在场时不断关闭，或者当有人进入房间时打开灯光存在不可接受的延迟。你可以将这种方法应用于几乎任何问题，它将为你提供关于产品的假设的宝贵验证，而不需要你花费时间和精力在基于错误基础的机器学习模型上。更好的是，你可以设置这个过程，以便从中生成用于训练集的标记数据，因为你将拥有输入数据以及你的绿野仙踪根据这些输入所做的决定。

# 首先在桌面上让它运行起来

绿野仙踪方法是尽快让原型运行起来的一种方式，但即使在进行模型训练之后，你也应该考虑如何尽快进行实验和迭代。将模型导出并使其在嵌入式平台上运行足够快可能需要很长时间，因此一个很好的捷径是从环境中的传感器向附近的桌面或云计算机传输数据进行处理。这可能会消耗太多能量，无法成为生产中可部署的解决方案，但只要你能确保延迟不会影响整体体验，这是一个很好的方式来获取关于你的机器学习解决方案在整个产品设计背景下运行情况的反馈。

另一个重要的好处是，你可以录制一次传感器数据流，然后一遍又一遍地用于对模型进行非正式评估。如果模型在过去曾经犯过特别严重的错误，而这些错误可能无法在正常指标中得到充分体现，这将尤其有用。如果你的照片分类器将一个婴儿标记为狗，即使你整体准确率为 95%，你可能也会特别想避免这种情况，因为这会让用户感到不安。

在桌面上运行模型有很多选择。开始的最简单方法是使用像树莓派这样具有良好传感器支持的平台收集示例数据，然后将数据批量复制到您的桌面机器（或者如果您喜欢，可以复制到云实例）。然后，您可以使用标准的 Python TensorFlow 以离线方式训练和评估潜在模型，没有交互性。当您有一个看起来很有前途的模型时，您可以采取增量步骤，例如将您的 TensorFlow 模型转换为 TensorFlow Lite，但继续在 PC 上针对批处理数据进行评估。在这之后，您可以尝试将桌面 TensorFlow Lite 应用程序放在一个简单的 Web API 后面，并从具有您所瞄准的外形因素的设备上调用它，以了解它在真实环境中的工作方式。
