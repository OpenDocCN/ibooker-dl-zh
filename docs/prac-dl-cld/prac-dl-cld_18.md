# 第十七章：在不到一个小时内构建自动驾驶汽车：使用 AWS DeepRacer 进行强化学习

由客座作者 Sunil Mallya 撰写

如果你关注科技新闻，你可能已经看到了关于计算机何时将接管世界的辩论再次兴起。尽管这是一个有趣的思考练习，但是是什么引发了这些辩论的再次兴起呢？这种辩论再次兴起的很大一部分原因归功于计算机在决策任务中击败人类的消息——在国际象棋中获胜，在视频游戏中取得高分，如《Atari》（2013），在复杂的围棋比赛中击败人类（2016），最后，在 2017 年击败人类团队在《Defense of the Ancients》（Dota）2 中。这些成功最令人惊讶的事情是，“机器人”通过相互对抗并强化他们发现的成功策略来学习这些游戏。

如果我们更广泛地思考这个概念，这与人类教导他们的宠物没有什么不同。为了训练一只狗，每一种好行为都会通过奖励狗狗一块零食和许多拥抱来加强，而每一种不良行为都会通过断言“坏狗狗”来加以阻止。强化好行为和阻止不良行为的概念基本上构成了*强化学习*的核心。

计算机游戏，或者说一般的游戏，需要做出一系列决策，因此传统的监督方法并不适用，因为它们通常专注于做出单一决策（例如，这是一张猫还是狗的图片？）。强化学习社区内部的一个笑话是我们整天都在玩视频游戏（剧透：这是真的！）。目前，强化学习正在被应用于各行各业，以优化股票交易，管理大型建筑和数据中心的供暖和制冷，进行实时广告竞价，优化视频流质量，甚至优化实验室中的化学反应。鉴于这些生产系统的例子，我们强烈建议在顺序决策制定和优化问题中使用强化学习。在本章中，我们专注于学习这种机器学习范式，并将其应用于一个真实世界问题：在不到一个小时内构建一个 1/18 比例的自动驾驶汽车。

# 强化学习简介

与深度学习类似，强化学习在过去几年中经历了复兴，自从以前保持的人类视频游戏记录被打破以来。强化学习理论在上世纪 90 年代达到鼎盛时期，但由于计算要求和训练这些系统的困难，它没有进入大规模生产系统。传统上，强化学习被认为是计算密集型的；相比之下，神经网络是数据密集型的。但是深度神经网络的进步也使强化学习受益。神经网络现在被用来表示强化学习模型，从而诞生了深度强化学习。在本章中，我们将强化学习和深度强化学习这两个术语互换使用，但在几乎所有情况下，如果没有另有说明，当我们提到强化学习时，我们指的是深度强化学习。

尽管最近有所进展，但强化学习的领域并不友好。用于训练深度学习模型的界面逐渐变得简单，但在强化学习社区中还没有跟上。强化学习的另一个具有挑战性的方面是显著的计算要求和模型收敛所需的时间（学习完成）——要创建一个收敛的模型实际上需要几天，甚至几周的时间。现在，假设我们有耐心、神经网络知识和金钱带宽，关于强化学习的教育资源是少之又少的。大多数资源都针对高级数据科学家，有时对开发人员来说难以触及。我们之前提到的那个 1/18 比例的自动驾驶汽车？那就是 AWS 的 DeepRacer。AWS DeepRacer 背后最大的动机之一是让强化学习对开发人员更加可访问。它由亚马逊 SageMaker 强化学习提供支持，这是一个通用的强化学习平台。而且，让我们真实一点：谁不喜欢自动驾驶汽车呢？

![AWS DeepRacer 的 1/18 比例自主汽车](img/00305.jpeg) 

###### 图 17-1。AWS DeepRacer 的 1/18 比例自主汽车

# 为什么要通过自动驾驶汽车学习强化学习？

近年来，自动驾驶技术得到了重大投资和成功。DIY 自动驾驶和无线电控制（RC）汽车竞速社区因此变得流行。开发人员在真实硬件上构建按比例缩小的自主汽车，并在真实场景中进行测试的热情空前，这促使我们使用“车辆”（字面上）来教育开发人员学习强化学习。尽管存在其他算法来构建自动驾驶汽车，如传统计算机视觉或监督学习（行为克隆），但我们认为强化学习比这些算法更具优势。

表 17-1 总结了一些供开发人员使用的热门自动驾驶套件，以及支持它们的技术。强化学习的一个关键优势是模型可以在模拟器中进行专门训练。但强化学习系统也带来了一系列挑战，其中最大的挑战之一是从模拟到真实（sim2real）问题。在将完全在模拟中训练的模型部署到真实环境中时，总是存在挑战。DeepRacer 通过一些简单而有效的解决方案来解决这个问题，我们稍后在本章中讨论。在 2018 年 11 月推出 DeepRacer 后的前六个月内，近 9000 名开发人员在模拟器中训练了他们的模型，并成功在真实赛道上进行了测试。

表 17-1。自主自动驾驶技术的概况

|  | **硬件** | **组装** | **技术** | **成本** |  |
| --- | --- | --- | --- | --- | --- |
| AWS DeepRacer | Intel Atom with 100 GFLOPS GPU | 预装 | 强化学习 | $399 | ![](img/00263.jpeg) |
| OpenMV | OpenMV H7 | DIY (两小时) | 传统计算机视觉 | $90 | ![](img/00203.jpeg) |
| Duckietown | Raspberry Pi | 预装 | 强化学习，行为克隆 | $279–$350 | ![](img/00079.jpeg) |
| DonkeyCar | Raspberry Pi | DIY (两到三小时) | 行为克隆 | $250 | ![](img/00266.jpeg) |
| NVIDIA JetRacer | Jetson Nano | DIY (三到五小时) | 监督学习 | 约$400 | ![](img/00225.jpeg) |

# 使用 DeepRacer 进行实用的深度强化学习

现在是本章最令人兴奋的部分：构建我们的第一个基于强化学习的自主赛车模型。在我们踏上这个旅程之前，让我们建立一个快速的术语备忘单，帮助您熟悉重要的强化学习术语：

目标

完成绕过赛道一圈而不偏离赛道。

输入

在人类驾驶的汽车中，人类通过可视化环境并利用驾驶知识做出决策并驾驶车辆。DeepRacer 也是一个视觉驱动系统，因此我们将单个摄像头图像作为系统的输入。具体来说，我们使用灰度 120x160 图像作为输入。

输出（动作）

在现实世界中，我们通过油门（油门）、刹车和方向盘来驾驶汽车。建立在遥控车上的 DeepRacer 有两个控制信号：油门和方向盘，两者都由传统的脉冲宽度调制（PWM）信号控制。将驾驶映射到 PWM 信号可能不直观，因此我们将汽车可以采取的驾驶动作离散化。还记得我们在电脑上玩的那些老式赛车游戏吗？其中最简单的使用箭头键——左、右和上——来驾驶汽车。同样，我们可以定义汽车可以采取的一组固定动作，但对油门和方向盘有更精细的控制。我们的强化学习模型在训练后将决定采取哪种动作，以便成功地驾驶赛道。在创建模型时，我们将有灵活性定义这些动作。

###### 注意

遥控爱好车中的舵机通常由 PWM 信号控制，这是一系列脉冲信号，脉冲宽度不同。舵机需要到达的位置是通过发送特定宽度的脉冲信号来实现的。脉冲的参数是最小脉冲宽度、最大脉冲宽度和重复率。

代理

学习并做出决策的系统。在我们的情况下，是汽车学习如何驾驶环境（赛道）。

环境

代理通过与动作的交互学习。在 DeepRacer 中，环境包含一个定义代理可以前往和停留的赛道。代理探索环境以收集数据，以训练基础的深度强化学习神经网络。

状态（s）

代理在环境中的位置表示。这是代理的一个瞬时快照。对于 DeepRacer，我们使用图像作为状态。

动作（a）

代理可以做出的决策集。

步骤

从一个状态离散过渡到下一个状态。

情节

这指的是汽车为实现目标而尝试的努力；即，在赛道上完成一圈。因此，一个情节是一系列步骤或经验。不同的情节可能有不同的长度。

奖励（r）

给定输入状态，代理采取的动作的值。

策略（π）

决策策略或函数；从状态到动作的映射。

价值函数（V）

状态到值的映射，其中值代表给定状态下对动作的预期奖励。

重播或经验缓冲区

临时存储缓冲区，存储经验，这是一个元组（s，a，r，s`'`），其中“s”代表摄像头捕获的观察（或状态），“a”代表车辆采取的动作，“r”代表该动作产生的预期奖励，“s`'`”代表采取动作后的新观察（或新状态）。

奖励函数

任何强化学习系统都需要一个指导，告诉模型在学习过程中在特定情况下什么是好的或坏的动作。奖励函数充当这个指导，评估汽车采取的动作并给予奖励（标量值），指示该动作在该情况下的可取性。例如，在左转时，采取“左转”动作将被认为是最佳的（例如，奖励=1；在 0-1 范围内），但采取“右转”动作将是不好的（奖励=0）。强化学习系统最终根据奖励函数收集这些指导并训练模型。这是训练汽车的最关键部分，也是我们将重点关注的部分。

最后，当我们组装系统时，原理图流程如下：

```py
Input (120x160 grayscale Image) → (reinforcement learning Model) → 
Output (left, right, straight)

```

在 AWS DeepRacer 中，奖励函数是模型构建过程中的重要部分。在训练 AWS DeepRacer 模型时，我们必须提供它。

在一个 episode 中，代理与赛道互动，学习最优动作集，以最大化预期累积奖励。但是，单个 episode 并不产生足够的数据来训练代理。因此，我们最终会收集许多 episode 的数据。定期，在每个第 n 个 episode 结束时，我们启动一个训练过程，生成一个强化学习模型的迭代。我们运行许多迭代来生成我们能够的最佳模型。这个过程在下一节中详细解释。训练结束后，代理通过在模型上运行推理来执行自主驾驶，以根据图像输入采取最佳动作。模型的评估可以在模拟环境中使用虚拟代理或在物理 AWS DeepRacer 汽车的真实环境中进行。

终于是时候创建我们的第一个模型了。因为汽车的输入是固定的，即来自摄像头的单个图像，所以我们只需要关注输出（动作）和奖励函数。我们可以按照以下步骤开始训练模型。

## 构建我们的第一个强化学习

要进行这个练习，您需要一个 AWS 账户。使用您的账户凭据登录 AWS 控制台，如图 17-2 所示。

![AWS 登录控制台](img/00171.jpeg)

###### 图 17-2。AWS 登录控制台

首先，让我们确保我们在北弗吉尼亚地区，因为该服务仅在该地区提供，并转到 DeepRacer 控制台页面：[*https://console.aws.amazon.com/deepracer/home?region=us-east-1#getStarted*](https://console.aws.amazon.com/deepracer/home?region=us-east-1#getStarted)。

在选择“强化学习”后，模型页面会打开。该页面显示了所有已创建模型的列表以及每个模型的状态。要创建模型，请从这里开始该过程。

![训练 AWS DeepRacer 模型的工作流程](img/00189.jpeg)

###### 图 17-3。训练 AWS DeepRacer 模型的工作流程

## 步骤 1：创建模型

我们将创建一个模型，供 AWS DeepRacer 汽车在赛道上自主驾驶（采取动作）。我们需要选择特定的赛道，提供我们的模型可以选择的动作，提供一个奖励函数，用于激励我们期望的驾驶行为，并配置训练期间使用的超参数。

在 AWS DeepRacer 控制台上创建模型

###### 图 17-4。在 AWS DeepRacer 控制台上创建模型

## 步骤 2：配置训练

在这一步中，我们选择我们的训练环境，配置动作空间，编写奖励函数，并在启动训练作业之前调整其他与训练相关的设置。

### 配置模拟环境

我们的强化学习模型训练发生在模拟赛道上，我们可以选择赛道来训练我们的模型。我们将使用 AWS RoboMaker，这是一个使构建机器人应用程序变得简单的云服务，来启动模拟环境。

在训练模型时，我们选择与我们打算在赛道上比赛的最终赛道最相似的赛道。截至 2019 年 7 月，AWS DeepRacer 提供了七个可以进行训练的赛道。虽然配置这样一个辅助环境并不是必需的，也不能保证一个好的模型，但它将最大化我们的模型在赛道上表现最好的可能性。此外，如果我们在一条直线赛道上训练，那么我们的模型很可能不会学会如何转弯。就像在监督学习的情况下，模型不太可能学会不属于训练数据的内容一样，在强化学习中，代理人不太可能从训练环境中学到超出范围的内容。对于我们的第一个练习，选择 re:Invent 2018 赛道，如图 17-5 所示。

要训练一个强化学习模型，我们必须选择一个学习算法。目前，AWS DeepRacer 控制台仅支持近端策略优化（PPO）算法。团队最终将支持更多的算法，但选择 PPO 是为了更快的训练时间和更优越的收敛性能。训练一个强化学习模型是一个迭代的过程。首先，定义一个奖励函数来覆盖代理在环境中的所有重要行为是一个挑战。其次，通常需要调整超参数以确保令人满意的训练性能。这两者都需要实验。一个谨慎的方法是从一个简单的奖励函数开始，这将是本章的方法，然后逐步增强它。AWS DeepRacer 通过允许我们克隆一个训练好的模型来促进这个迭代过程，在这个模型中，我们可以增强奖励函数以处理之前被忽略的变量，或者我们可以系统地调整超参数直到结果收敛。检测这种收敛的最简单方法是查看日志，看看汽车是否超过了终点线；换句话说，进展是否达到了 100%。或者，我们可以直观地观察汽车的行为，并确认它是否超过了终点线。

![在 AWS DeepRacer 控制台上选择赛道](img/00085.jpeg)

###### 图 17-5。在 AWS DeepRacer 控制台上选择赛道

### 配置动作空间

接下来，我们配置我们的模型在训练期间和训练后选择的动作空间。一个动作（输出）是速度和转向角的组合。目前在 AWS DeepRacer 中，我们使用离散动作空间（固定的动作集）而不是连续动作空间（以*x*速度转动*x*度，其中*x*和*y*取实值）。这是因为更容易映射到物理汽车上的值，我们稍后会深入探讨这一点在“驾驶 AWS DeepRacer 汽车”。为了构建这个离散动作空间，我们指定了最大速度、速度级别、最大转向角和转向级别，如图 17-6 所示。

![在 AWS DeepRacer 控制台上定义动作空间](img/00041.jpeg)

###### 图 17-6。在 AWS DeepRacer 控制台上定义动作空间

以下是动作空间的配置参数：

最大转向角度

这是汽车前轮可以向左和向右转动的最大角度。轮子可以转动的角度是有限的，因此最大转向角度为 30 度。

转向角度粒度

指的是最大转向角两侧的转向间隔数。因此，如果我们的最大转向角为 30 度，+30 度是向左，-30 度是向右。具有 5 个转向粒度时，从左到右的转向角如图 17-6 所示，将在行动空间中：30 度，15 度，0 度，-15 度和-30 度。转向角始终围绕 0 度对称。

最大速度

指的是模拟器中车辆将以米/秒（m/s）为单位测量的最大速度驾驶的速度。

速度级别

指的是从最大速度（包括）到零（不包括）的速度级别数。因此，如果我们的最大速度是 3m/s，速度粒度为 3，那么我们的行动空间将包含 1m/s、2m/s 和 3m/s 的速度设置。简单来说，3m/s 除以 3 等于 1m/s，所以从 0m/s 到 3m/s 以 1m/s 的增量进行。0m/s 不包括在行动空间中。

根据前面的例子，最终的行动空间将包括 15 个离散行动（三种速度 x 五种转向角），这些应该在 AWS DeepRacer 服务中列出。随意尝试其他选项，只需记住较大的行动空间可能需要更长时间进行训练。

###### 提示

根据我们的经验，以下是一些建议如何配置行动空间：

+   我们的实验表明，具有更快最大速度的模型收敛所需的时间比具有较慢最大速度的模型更长。在某些情况下（奖励函数和赛道相关），5m/s 模型收敛可能需要超过 12 小时。

+   我们的模型不会执行不在行动空间中的行动。同样，如果模型在从未需要使用此行动的赛道上进行训练，例如，在直道上不会激励转弯，那么模型将不知道如何使用此行动，因为它不会被激励转弯。在开始考虑构建强大模型时，请确保记住行动空间和训练赛道。

+   指定快速速度或大转向角是很好的，但我们仍然需要考虑我们的奖励函数，以及是否有意义全速驶入转弯，或在赛道的直线段上展示之字形行为。

+   我们还需要牢记物理学。如果我们尝试以超过 5m/s 的速度训练模型，我们可能会看到我们的车在拐弯时打滑，这可能会增加模型收敛的时间。

### 配置奖励函数

正如我们之前解释的那样，奖励函数评估了在给定情况下行动结果的质量，并相应地奖励该行动。在实践中，奖励是在每次行动后进行训练时计算的，并且构成了用于训练模型的经验的关键部分。然后我们将元组（状态，行动，下一个状态，奖励）存储在内存缓冲区中。我们可以使用模拟器提供的多个变量来构建奖励函数逻辑。这些变量代表了车辆的测量，如转向角和速度；车辆与赛道的关系，如（x，y）坐标；以及赛道，如路标（赛道上的里程碑标记）。我们可以使用这些测量值来在 Python 3 语法中构建我们的奖励函数逻辑。

所有参数都作为字典提供给奖励函数。它们的键，数据类型和描述在图 17-7 中有文档记录，一些更微妙的参数在图 17-8 中有进一步说明。

奖励函数参数（这些参数的更深入审查可在文档中找到）

###### 图 17-7. 奖励函数参数（这些参数的更深入审查可在文档中找到）

图解释了一些奖励函数参数

###### 图 17-8. 奖励函数参数的可视化解释

为了构建我们的第一个模型，让我们选择一个示例奖励函数并训练我们的模型。让我们使用默认模板，在其中汽车试图跟随中心虚线，如图 17-9 所示。这个奖励函数背后的直觉是沿着赛道采取最安全的导航路径，因为保持在中心位置可以使汽车远离赛道外。奖励函数的作用是：在赛道周围创建三个层次，使用三个标记，然后为在第二层次驾驶的汽车提供更多奖励，而不是在中心或最后一层次驾驶。还要注意奖励的大小差异。我们为保持在狭窄的中心层次提供 1 的奖励，为保持在第二（偏离中心）层次提供 0.5 的奖励，为保持在最后一层次提供 0.1 的奖励。如果我们减少中心层次的奖励，或增加第二层次的奖励，实质上我们在激励汽车使用更大的赛道表面。记得你考驾照的时候吗？考官可能也是这样做的，当你靠近路缘或车道标志时扣分。这可能会很有用，特别是在有急转弯的情况下。

![一个示例奖励函数](img/00249.jpeg)

###### 图 17-9. 一个示例奖励函数

以下是设置这一切的代码：

```py
def reward_function(params):
    '''
 Example of rewarding the agent to follow center line
 '''

    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3 # likely crashed/ close to off track

    return float(reward)
```

因为这是第一次训练运行，让我们专注于理解创建和评估基本模型的过程，然后专注于优化它。在这种情况下，我们跳过算法设置和超参数部分，使用默认设置。

###### 注意

*常见问题：奖励应该在某个范围内吗，我可以给负奖励吗？*

没有真正的约束来决定我们可以奖励和不奖励什么，但作为一个良好的实践，更容易理解奖励的方式是将它们放在 0-1 或 0-100 的范围内。更重要的是，我们的奖励尺度应该适当地为行为提供相对奖励。例如，在右转时，我们应该用高奖励奖励正确的行为，用接近 0 的奖励奖励左转行为，也许用介于两者之间的奖励或高于左转行为的奖励奖励直行行为，因为这可能不是完全错误的行为。

### 配置停止条件

这是我们开始训练之前的最后一节。在这里，我们指定模型将训练的最长时间。这是一个方便的机制，让我们可以终止训练，因为我们将根据训练时间计费。

指定 60 分钟，然后选择“开始训练”。如果出现错误，我们将被带到错误位置。开始训练后，可能需要长达六分钟的时间来启动所需的服务（如 Amazon SageMaker、AWS Robomaker、AWS Lambda、AWS Step Function）来开始训练。请记住，如果我们确定模型已经收敛（如下一节所述），我们随时可以通过点击“停止”按钮提前停止训练。

## 第三步：模型训练

当我们的模型开始训练后，我们可以从 DeepRacer 控制台上列出的模型中选择它。然后，我们可以通过查看随时间变化的总奖励图以及从模拟器中汽车的第一人称视角来定量地了解训练的进展情况（参见图 17-10）。

起初，我们的车无法在直路上行驶，但随着它学习到更好的驾驶行为，我们应该看到它的表现提高，奖励图增加。此外，当我们的车驶离赛道时，它将被重置在赛道上。我们可能会观察到奖励图呈波动状态。

###### 注意

*常见问题：为什么奖励图呈波动状态？*

代理从高探索开始，并逐渐开始利用训练模型。因为代理始终对其决策的一部分采取随机动作，所以可能会有时候完全做出错误决定并最终偏离轨道。这通常在训练开始时很高，但随着模型开始学习，这种尖锐性应该会减少。

日志始终是关于我们模型训练更细粒度信息的良好来源。在本章后面，我们将探讨如何以编程方式使用日志来更深入地了解我们模型的训练。与此同时，我们可以查看 Amazon SageMaker 和 AWS RoboMaker 的日志文件。日志输出到 Amazon CloudWatch。要查看日志，请将鼠标悬停在奖励图上，并选择刷新按钮下方出现的三个点。然后，选择“查看日志”。因为模型训练需要一个小时，这可能是一个好时机跳到下一节，了解更多关于强化学习的知识。

![AWS DeepRacer 控制台上的训练图和模拟视频流](img/00209.jpeg)

###### 图 17-10。AWS DeepRacer 控制台上的训练图和模拟视频流

## 第 4 步：评估模型的性能

在强化学习中，评估模型能力的最佳方法是运行它，使其仅利用——也就是说，它不会采取随机动作。在我们的情况下，首先在类似于训练的赛道上测试它，看看它是否能复制训练行为。接下来，尝试在不同的赛道上测试泛化能力。在我们的模型训练完成后，我们可以开始模型评估。从我们观察到训练的模型详细信息页面中，选择“开始评估”。现在我们可以选择要评估我们模型性能的赛道以及圈数。选择“re:Invent 2018”赛道和 5 圈，然后选择开始。完成后，我们应该看到与图 17-11 中显示的类似的页面，总结了我们模型尝试绕过赛道并完成圈数的结果。

![AWS DeepRacer 控制台上的模型评估页面](img/00176.jpeg)

###### 图 17-11。AWS DeepRacer 控制台上的模型评估页面

干得好！我们已经成功构建了我们的第一个强化学习启用的自动驾驶汽车。

###### 注意

*常见问题：当我运行评估时，有时只看到 100%的完成率？*

当模型在模拟器中运行推理并在赛道周围导航时，由于模拟器的保真度，汽车可能会以稍微不同的位置结束相同的动作——换句话说，15 度左转动作可能只会导致 14.9 度。实际上，我们在模拟器中观察到的只是非常小的偏差，但这些小偏差会随着时间的推移而累积。训练良好的模型能够从接近越野位置恢复，但训练不足的模型可能无法从接近事故的位置恢复。

现在我们的模型已经成功评估，我们继续改进它，并学习如何实现更好的圈速。但在此之前，您需要更多地了解强化学习背后的理论，并深入了解学习过程。

###### 注意

我们可以开始使用 AWS DeepRacer 控制台服务创建我们的第一个模型，对其进行训练（最多六个小时），评估它，并免费提交给 AWS DeepRacer 联赛。

# 行动中的强化学习

让我们更仔细地看看强化学习的实际应用。在这里，我们讨论一些理论、内部工作原理以及与我们自动驾驶汽车项目相关的许多实用见解。

## 强化学习系统是如何学习的？

首先，我们必须理解*探索*与*利用*。类似于孩子学习的方式，强化学习系统通过*探索*和发现什么是好的和坏的来学习。在孩子的情况下，父母指导或告诉孩子他的错误，评价所做的决定是好还是坏，或者好坏程度如何。孩子记住了他在特定情况下所做的决定，并试图在合适的时机重播，或者*利用*那些决定。实质上，孩子试图最大化父母的认可或欢呼。在孩子的早期生活中，它更加关注父母的建议，从而创造学习的机会。而在后来的生活中，成年人很少听从父母的建议，而是利用他们学到的概念来做决定。

如图 17-12 所示，在每个时间步“t”，DeepRacer 汽车（代理）接收到它的当前观察，状态（S[t]），并基于此选择一个行动（A[t]）。作为代理所采取的行动的结果，它获得奖励 R[t+1]并移动到状态 S[t+1]，这个过程在整个 episode 中持续进行。

深度强化学习理论基础要点

###### 图 17-12\. 深度强化学习理论基础要点

在 DeepRacer 的背景下，代理通过采取随机行动来进行探索，奖励函数，本质上就像父母一样，告诉汽车它在给定状态下所采取的行动是好还是不好。通常，对于给定状态的行动的“好坏”被表示为一个数字，较高的数字意味着它接近最优，较低的数字意味着它不是。系统为每一步记录所有这些信息，具体包括：*当前状态，采取的行动，行动的奖励和下一个状态*（s,a,r,s`'`），这就是我们所谓的经验回放缓冲区，本质上是一个临时内存缓冲区。整个想法在于汽车可以从哪些决定是好的以及其他决定是坏的中学习。关键点是我们*从高度探索开始，并逐渐增加利用*。

在 DeepRacer 模拟器中，我们以每秒 15 帧的速度采样输入图像状态。在每一步或捕获的图像帧中，汽车从一个状态转移到另一个状态。每个图像代表汽车所处的状态，最终在强化学习模型训练后，它将试图通过推断采取哪种行动来进行利用。为了做出决策，我们可以随机采取行动，也可以使用我们的模型进行推荐。随着模型的训练，训练过程平衡了探索和利用。首先，我们更多地进行探索，因为我们的模型不太可能很好，但随着它通过经验学习，我们更多地向利用方向转变，让模型更多地控制。图 17-13 描述了这个流程。这种转变可以是线性衰减、指数衰减或任何类似的策略，通常根据我们假设的学习程度进行调整。通常，在实际的强化学习训练中使用指数衰减。

![DeepRacer 训练流程](img/00089.jpeg)

###### 图 17-13\. DeepRacer 训练流程

在通过随机选择或模型预测采取行动后，汽车转移到一个新状态。使用奖励函数，计算奖励并分配给结果。这个过程将继续，对于每个状态，直到达到终端状态为止；也就是说，汽车偏离赛道或完成一圈，此时汽车将被重置，然后重复。一步是从一个状态到另一个状态的转换，每一步都会记录一个（状态，行动，新状态，奖励）元组。从重置点到终端状态的所有步骤称为*episode*。

为了说明一个情节，让我们看一下图 17-14 中一个微型赛道的例子，奖励函数鼓励沿着中心线行驶，因为这是从起点到终点最短、最快的路径。这个情节包括四个步骤：在第 1 和第 2 步，汽车沿着中心线行驶，然后在第 3 步左转 45 度，继续沿着这个方向，最终在第 4 步发生碰撞。

![Illustration of an agent exploring during an episode](img/00046.jpeg)

###### 图 17-14\. 一个代理在一个情节中探索的插图

我们可以将这些情节视为经验，或者是我们模型的训练数据。定期，强化学习系统会从内存缓冲区中随机选择一小部分这些记录，并训练一个 DNN（我们的深度强化学习模型），以便根据我们的奖励函数“指导”来产生一个最能够在环境中导航的模型；换句话说，获得最大的累积奖励。随着时间的推移或更多的情节，我们会看到一个随着时间变得更好的模型。当然，一个极其重要的警告是奖励函数必须被明确定义，并引导代理朝着目标前进。如果我们的奖励函数不好，我们的模型就无法学习正确的行为。

让我们稍微偏离一下，以了解一个糟糕的奖励函数。当我们最初设计系统时，汽车可以自由选择任何方向（左、右、前、后）。我们最简单的奖励函数没有区分方向，最终模型学会了通过来回摇摆来累积奖励。后来通过激励汽车向前行驶来克服了这个问题。幸运的是，现在的开发者们，DeepRacer 团队通过让汽车只能向前移动，使这一切变得更简单，因此我们甚至不需要考虑这种行为。

现在回到强化学习。*我们如何知道模型是否变得更好？* 让我们回到探索和利用的概念。记住，我们从高探索、低利用开始，然后逐渐增加利用率。这意味着在训练过程的任何时刻，强化学习系统会探索一定百分比的时间；也就是说，采取随机行动。随着经验缓冲区的增长，对于任何给定状态，它存储了各种决策（行动），包括导致最高奖励和最低奖励的行动。模型基本上利用这些信息来学习和预测在给定状态下采取的最佳行动。假设模型最初对状态 S 采取了“直行”行动，并获得了 0.5 的奖励，但下一次到达状态 S 时，它采取了“右转”行动，并获得了 1 的奖励。如果内存缓冲区中有几个样本的奖励为“1”，对于相同的状态 S 和行动“右转”，模型最终会学习到“右转”是该状态的最佳行动。随着时间的推移，模型会探索的时间越来越少，利用的时间越来越多，这个百分比分配会线性或指数地改变，以便实现最佳学习。

关键在于，如果模型正在采取最佳行动，系统会学会继续选择这些行动；如果没有选择最佳行动，系统将继续尝试学习给定状态的最佳行动。从实际角度来看，可能存在许多到达目标的路径，但对于图 17-13 中的小型赛道示例来说，最快的路径是沿着赛道中间直线前进，因为实际上任何转弯都会减慢汽车速度。在训练过程中，在模型开始学习的早期阶段，我们观察到它可能到达终点线（目标），但可能没有选择最佳路径，如图 17-15（左）所示，汽车来回穿梭，因此需要更长时间到达终点线，累积奖励仅为 9。但因为系统仍然继续探索一部分时间，代理给自己机会找到更好的路径。随着经验的积累，代理学会找到最佳路径并收敛到一个最佳策略，以达到总累积奖励 18，如图 17-15（右）所示。

![达到目标的不同路径示例](img/00009.jpeg)

###### 图 17-15. 达到目标的不同路径示例

从数量上来说，对于每一集，你应该看到奖励逐渐增加的趋势。如果模型做出了最佳决策，汽车必须在赛道上并且在课程中导航，从而累积奖励。然而，你可能会看到即使在高奖励的集数之后，图表也会下降，这可能是因为汽车仍然具有较高程度的探索，正如前面提到的那样。

## 强化学习理论

现在我们了解了强化学习系统是如何学习和工作的，特别是在 AWS DeepRacer 的背景下，让我们来看一些正式的定义和一般的强化学习理论。当我们使用强化学习解决其他问题时，这些背景知识将会很有用。

### 马尔可夫决策过程

马尔可夫决策过程（MDP）是一个用于建模控制过程中决策制定的离散随机状态转移过程框架。马尔可夫性质定义了每个状态仅仅依赖于前一个状态。这是一个方便的性质，因为这意味着要进行状态转移，所有必要的信息必须在当前状态中可用。强化学习中的理论结果依赖于问题被制定为一个 MDP，因此重要的是要理解如何将问题建模为一个 MDP，以便使用强化学习来解决。

### 无模型与基于模型

在这个背景下，模型指的是对环境的学习表示。这是有用的，因为我们可以潜在地学习环境中的动态并使用模型训练我们的代理，而不必每次都使用真实环境。然而，在现实中，学习环境并不容易，通常更容易在模拟中拥有真实世界的表示，然后联合学习感知和动态作为代理导航的一部分，而不是按顺序一个接一个地学习。在本章中，我们只关注无模型的强化学习。

### 基于价值

对于代理采取的每个动作，奖励函数都会分配相应的奖励。对于任何给定的状态-动作对，了解其价值（奖励）是有帮助的。如果这样的函数存在，我们可以计算在任何状态下可以实现的最大奖励，并简单地选择相应的动作来导航环境。例如，在一个 3x3 的井字棋游戏中，游戏情况的数量是有限的，因此我们可以建立一个查找表，以便在给定情况下给出最佳移动。但是在国际象棋游戏中，考虑到棋盘的大小和游戏的复杂性，这样的查找表将是计算昂贵的，并且存储空间将会很大。因此，在复杂的环境中，很难列出状态-动作值对或定义一个可以将状态-动作对映射到值的函数。因此，我们尝试使用神经网络通过对价值函数进行参数化，并使用神经网络来近似给定状态观察下每个动作的价值。一个基于值的算法的例子是深度 Q 学习。

### 基于策略

策略是代理学习如何在环境中导航的一组规则。简单来说，策略函数告诉代理从当前状态中采取哪个动作是最佳动作。基于策略的强化学习算法，如 REINFORCE 和策略梯度，找到最佳策略，无需将值映射到状态。在强化学习中，我们对策略进行参数化。换句话说，我们允许神经网络学习什么是最佳策略函数。

### 基于策略还是基于值——为什么不两者兼而有之？

一直存在关于使用基于策略还是基于值的强化学习的争论。新的架构尝试同时学习价值函数和策略函数，而不是保持其中一个固定。这种在强化学习中的方法被称为*演员评论家*。

您可以将演员与策略关联起来，将评论家与价值函数关联起来。演员负责采取行动，评论家负责估计这些行动的“好坏”或价值。演员将状态映射到动作，评论家将状态-动作对映射到值。在演员评论家范式中，这两个网络（演员和评论家）使用梯度上升分别进行训练，以更新我们深度神经网络的权重。（请记住，我们的目标是最大化累积奖励；因此，我们需要找到全局最大值。）随着情节的推移，演员变得更擅长采取导致更高奖励状态的行动，评论家也变得更擅长估计这些行动的价值。演员和评论家学习的信号纯粹来自奖励函数。

给定状态-动作对的价值称为*Q 值*，表示为 Q(s,a)。我们可以将 Q 值分解为两部分：估计值和衡量动作优于其他动作的因素的量化度量。这个度量被称为*优势*函数。我们可以将优势视为给定状态-动作对的实际奖励与该状态的预期奖励之间的差异。差异越大，我们离选择最佳动作就越远。

考虑到估计状态的价值可能会成为一个困难的问题，我们可以专注于学习优势函数。这使我们能够评估动作不仅基于其有多好，还基于它可能有多好。这使我们比其他简单的基于策略梯度的方法更容易收敛到最佳策略，因为通常策略网络具有很高的方差。

### 延迟奖励和折扣因子（γ）

根据采取的动作，每个状态转换都会获得奖励。但这些奖励的影响可能是非线性的。对于某些问题，即时奖励可能更重要，在某些情况下，未来奖励可能更重要。例如，如果我们要构建一个股票交易算法，未来奖励可能具有更高的不确定性；因此，我们需要适当地进行折现。折现因子（γ）是介于[0,1]之间的乘法因子，接近零表示即时未来奖励更重要。对于接近 1 的高值，代理将专注于采取最大化未来奖励的行动。

## AWS DeepRacer 中的强化学习算法

首先，让我们看一个最简单的策略优化强化学习算法的例子：香草策略梯度。

![香草策略梯度算法的训练过程](img/00296.jpeg)

###### 图 17-16. 香草策略梯度算法的训练过程

我们可以将深度强化学习模型看作由两部分组成：输入嵌入器和策略网络。输入嵌入器将从图像输入中提取特征并将其传递给策略网络，策略网络做出决策；例如，预测对于给定输入状态哪个动作是最佳的。鉴于我们的输入是图像，我们使用卷积层（CNNs）来提取特征。因为策略是我们想要学习的内容，我们对策略函数进行参数化，最简单的方法是使用全连接层进行学习。输入 CNN 层接收图像，然后策略网络使用图像特征作为输入并输出一个动作。因此，将状态映射到动作。随着模型的训练，我们变得更擅长映射输入空间和提取相关特征，同时优化策略以获得每个状态的最佳动作。我们的目标是收集最大的累积奖励。为了实现这一目标，我们更新模型权重以最大化累积未来奖励，通过这样做，我们给导致更高累积未来奖励的动作赋予更高的概率。在以前训练神经网络时，我们使用随机梯度下降或其变体；在训练强化学习系统时，我们寻求最大化累积奖励；因此，我们不是最小化，而是最大化。因此，我们使用梯度上升来将权重移动到最陡峭奖励信号的方向。

DeepRacer 使用一种高级的策略优化变体，称为 Proximal Policy Optimization（PPO），在图 17-17 中进行了总结。

![使用 PPO 算法进行训练](img/00043.jpeg)

###### 图 17-17. 使用 PPO 算法进行训练

在图 17-17 的左侧，我们的模拟器使用最新的策略（模型）获取新的经验（s，a，r，s'）。经验被馈送到经验重放缓冲区中，在我们完成一定数量的周期后，将经验馈送给我们的 PPO 算法。

在图 17-17 的右侧，我们使用 PPO 更新我们的模型。尽管 PPO 是一种策略优化方法，但它使用了我们之前描述的优势演员-评论家方法。我们计算 PPO 梯度并将策略移动到我们获得最高奖励的方向。盲目地朝这个方向迈大步可能会导致训练中的变化过大；如果我们迈小步，训练可能会持续很长时间。PPO 通过限制每个训练步骤中策略可以更新的程度来改善策略（演员）的稳定性。这是通过使用剪切的替代目标函数来实现的，它防止策略更新过多，从而解决了策略优化方法中常见的大方差问题。通常情况下，对于 PPO，我们保持新旧策略的比率在[0.8, 1.2]。评论家告诉演员采取的行动有多好，以及演员应该如何调整其网络。在策略更新后，新模型被发送到模拟器以获取更多经验。

## 以 DeepRacer 为例的深度强化学习总结

要使用强化学习解决任何问题，我们需要按照以下步骤进行：

1.  定义目标。

1.  选择输入状态。

1.  定义动作空间。

1.  构建奖励函数。

1.  定义 DNN 架构。

1.  选择强化学习优化算法（DQN、PPO 等）。

训练强化学习模型的基本方式在构建自动驾驶汽车或构建机器人手臂抓取物体时并没有改变。这是该范式的一个巨大优势，因为它允许我们专注于更高层次的抽象。要使用强化学习解决问题，首要任务是将问题定义为 MDP，然后定义输入状态和代理在给定环境中可以采取的一组动作，以及奖励函数。实际上，奖励函数可能是最难定义的部分之一，通常也是最重要的，因为这会影响我们的代理学习的策略。在定义了与环境相关的因素之后，我们可以专注于深度神经网络架构应该如何将输入映射到动作，然后选择强化学习算法（基于价值、基于策略、演员-评论家）进行学习。选择算法后，我们可以专注于控制算法行为的高级旋钮。当我们驾驶汽车时，我们倾向于关注控制，对内燃机的理解并不会太大程度上影响我们的驾驶方式。同样，只要我们了解每个算法暴露的旋钮，我们就可以训练强化学习模型。

现在是时候结束了。让我们制定 DeepRacer 赛车问题：

1.  目标：在最短时间内绕过赛道完成一圈

1.  输入：灰度 120x160 图像

1.  动作：具有组合速度和转向角值的离散动作

1.  奖励：奖励汽车在赛道上行驶，鼓励更快行驶，并防止进行大量校正或曲线行为

1.  DNN 架构：三层 CNN + 全连接层（输入 → CNN → CNN → CNN → FC → 输出）

1.  优化算法：PPO

## 步骤 5：改进强化学习模型

我们现在可以着手改进我们的模型，并了解我们模型训练的见解。首先，我们专注于在控制台中进行训练改进。我们可以改变强化学习算法设置和神经网络超参数。

### 算法设置

这一部分指定了在训练过程中强化学习算法将使用的超参数。超参数用于提高训练性能。

### 神经网络的超参数

表 17-2 介绍了可调整神经网络的超参数。尽管默认值在实践中被证明是不错的，但从实际角度来看，开发人员应该专注于批量大小、时代数量和学习率，因为它们被发现对生成高质量模型最有影响；也就是说，充分利用我们的奖励函数。

表 17-2. 深度神经网络可调超参数的描述和指导

| **参数** | **描述** | **提示** |
| --- | --- | --- |
| 批量大小 | 从经验缓冲区中随机抽取的最近车辆经验数量，用于更新底层深度学习神经网络权重。如果我们在缓冲区中有 5,120 个经验，并指定批量大小为 512，那么忽略随机抽样，我们将获得 10 个经验批次。每个批次将依次用于在训练过程中更新我们的神经网络权重。 | 使用更大的批量大小可以促进神经网络权重的更稳定和平滑的更新，但要注意训练可能会变慢。 |
| 时代数量 | 一个时代代表对所有批次的一次遍历，神经网络权重在处理每个批次后更新，然后继续下一个批次。十个时代意味着我们逐个更新神经网络权重，使用所有批次，但重复这个过程 10 次。 | 使用更多的时代数量可以促进更稳定的更新，但预计训练会变慢。当批量大小较小时，可以使用较少的时代数量。 |
| 学习率 | 学习率控制神经网络权重的更新幅度。简单来说，当我们需要改变策略的权重以获得最大累积奖励时，我们应该如何调整我们的策略。 | 更大的学习率会导致更快的训练，但可能难以收敛。较小的学习率会导致稳定的收敛，但训练时间可能较长。 |
| 探索 | 这指的是确定探索和利用之间的权衡方法。换句话说，我们应该使用什么方法来确定何时停止探索（随机选择动作）以及何时利用我们积累的经验。 | 由于我们将使用离散动作空间，我们应该始终选择“CategoricalParameters”。 |
| 熵 | 添加到动作空间的概率分布中的不确定性或随机性程度。这有助于促进选择随机动作，以更广泛地探索状态/动作空间。 |   |
| 折扣因子 | 一个指定未来奖励对预期累积奖励的贡献程度的因子。折扣因子越大，模型查看未来奖励以确定预期累积奖励的距离越远，训练速度越慢。使用折扣因子为 0.9 时，车辆包括来自 10 个未来步骤的奖励以进行移动。使用折扣因子为 0.999 时，车辆考虑来自 1,000 个未来步骤的奖励以进行移动。 | 推荐的折扣因子值为 0.99、0.999 和 0.9999。 |
| 损失类型 | 损失类型指定用于更新网络权重的目标函数（成本函数）的类型。对于小的更新，Huber 和均方误差损失类型的行为类似。但随着更新变大，Huber 损失相对于均方误差损失采取较小的增量。 | 当出现收敛问题时，使用 Huber 损失类型。当收敛良好且希望训练更快时，使用均方误差损失类型。 |
| 每次训练之间的剧集数量 | 此参数控制汽车在每次模型训练迭代之间应获取多少经验。对于具有更多局部最大值的更复杂问题，需要更大的经验缓冲区，以提供更多不相关的数据点。在这种情况下，训练会更慢但更稳定。 | 推荐值为 10、20 和 40。 |

### 模型训练见解

在模型训练完成后，从宏观角度来看，随时间变化的奖励图表，就像图 17-10 中的图表，让我们了解了训练的进展以及模型开始收敛的点。但它并没有给我们一个收敛策略的指示，也没有让我们了解我们的奖励函数的行为，或者汽车速度可以改进的地方。为了获得更多见解，我们开发了一个[Jupyter Notebook](https://oreil.ly/OWw_E)，分析训练日志，并提供建议。在本节中，我们将看一些更有用的可视化工具，可以用来深入了解我们模型的训练。

日志文件记录了汽车所采取的每一步。在每一步中，它记录了汽车的 x、y 位置，偏航（旋转），转向角，油门，从起点开始的进度，采取的行动，奖励，最近的航路点等等。

### 热图可视化

对于复杂的奖励函数，我们可能想要了解赛道上的奖励分布；也就是说，奖励函数在赛道上给汽车奖励的位置以及大小。为了可视化这一点，我们可以生成一个热图，如图 17-18 所示。鉴于我们使用的奖励函数在赛道中心附近给出最大奖励，我们看到该区域很亮，中心线两侧的一个小带是红色的，表示奖励较少。最后，赛道的其余部分是黑暗的，表示当汽车在这些位置时没有奖励或接近 0 的奖励。我们可以按照[代码](https://oreil.ly/n-w7G)生成自己的热图，并调查我们的奖励分布。

![示例中心线奖励函数的热图可视化](img/00214.jpeg)

###### 图 17-18. 示例中心线奖励函数的热图可视化

### 改进我们模型的速度

在我们运行评估之后，我们得到了圈速的结果。此时，我们可能会对汽车所采取的路径或失败的地方感到好奇，或者它减速的地方。为了了解所有这些，我们可以使用这个[笔记本](https://oreil.ly/tLOtk)中的代码来绘制一张赛道热图。在接下来的示例中，我们可以观察汽车在赛道周围导航时所采取的路径，并可视化它通过赛道上各个点的速度。这让我们了解了我们可以优化的部分。快速查看图 17-19（左）表明汽车在赛道的直线部分并没有真正快；这为我们提供了一个机会。我们可以通过给予更多奖励来激励模型在赛道的这一部分更快地行驶。

![评估运行的速度热图；（左）使用基本示例奖励函数的评估圈，（右）使用修改后的奖励函数更快的圈](img/00180.jpeg)

###### 图 17-19. 评估运行的速度热图；（左）使用基本示例奖励函数的评估圈，（右）使用修改后的奖励函数更快的圈

在图 17-19（左）中，有时汽车似乎有一个小的曲折模式，因此这里的一个改进可能是当汽车转弯过多时对其进行惩罚。在接下来的代码示例中，如果汽车转向超过阈值，我们将奖励乘以 0.8 的因子。我们还通过给予汽车 20%的额外奖励来激励汽车更快地行驶，如果汽车以 2 米/秒或更快的速度行驶。当使用这个新的奖励函数进行训练时，我们可以看到汽车比以前的奖励函数更快。图 17-19（右）显示更加稳定；汽车几乎完美地沿着中心线行驶，并且完成一圈比我们第一次尝试快大约两秒。这只是改进模型的一个简短介绍。我们可以继续迭代我们的模型，并使用这些工具记录更好的圈速。所有建议都包含在这里显示的奖励函数示例中：

```py
def reward_function(params):
    '''
 Example of penalize steering, which helps mitigate zigzag behaviors and
 speed incentive
 '''

    # Read input parameters
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    steering = abs(params['steering_angle']) # Only need absolute steering angle
    speed = params['speed'] # in meter/sec

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the agent is closer to the center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3 # likely crashed/ close to off track

    # Steering penalty threshold, change the number based on your action space 
      setting
    ABS_STEERING_THRESHOLD = 15

    # Penalize reward if the agent is steering too much
    if steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8

    # Incentivize going faster
    if speed >= 2:
        reward *= 1.2
```

# 比赛 AWS DeepRacer 汽车

现在是时候将我们从虚拟世界学到的知识带到现实世界中，与一辆真正的自动驾驶汽车比赛。当然是玩具大小的！

如果您拥有 AWS DeepRacer 汽车，请按照提供的说明在实际汽车上测试您的模型。对于有兴趣购买汽车的人，AWS DeepRacer 可以在亚马逊上购买。

## 建造赛道

现在我们有了一个训练过的模型，我们可以在真实赛道上评估这个模型，并使用实际的 AWS DeepRacer 汽车。首先，让我们在家里建造一个临时赛道来比赛我们的模型。为简单起见，我们只会建造部分赛道，但提供了如何建造整个赛道的说明[在这里](https://oreil.ly/2xZQA)。

要建造一条赛道，您需要以下材料：

对于赛道边界：

我们可以使用约两英寸宽的白色或米色胶带在深色赛道表面上创建一条赛道。在虚拟环境中，赛道标记的厚度设置为两英寸。对于深色表面，使用白色或米色胶带。例如，[1.88 英寸宽的珍珠白色胶带](https://oreil.ly/2x6dl)或[1.88 英寸（粘性较小）的粘贴胶带](https://oreil.ly/Hn0AD)。

对于赛道表面：

我们可以在深色硬地板上创建一条赛道，例如硬木、地毯、混凝土或[沥青毡](https://oreil.ly/7Q1ae)。后者模仿了现实世界的道路表面，反射很少。

## AWS DeepRacer 单圈赛道模板

这个基本的赛道模板由两个直线赛道段连接的弯曲赛道段组成，如图 17-20 所示。使用这个赛道训练的模型应该使我们的 AWS DeepRacer 车辆直线行驶或向一个方向转弯。*指定的转弯角度仅供参考；在铺设赛道时，我们可以使用近似的测量值。*

![测试赛道布局](img/00137.jpeg)

###### 图 17-20。测试赛道布局

## 在 AWS DeepRacer 上运行模型

要启动 AWS DeepRacer 车辆的自动驾驶，我们必须将至少一个 AWS DeepRacer 模型上传到我们的 AWS DeepRacer 车辆。

要上传模型，请从 AWS DeepRacer 控制台中选择我们训练过的模型，然后将模型工件从其 Amazon S3 存储下载到可以从计算机访问的（本地或网络）驱动器上。在模型页面上提供了一个方便的下载模型按钮。

要将训练过的模型上传到车辆上，请执行以下操作：

1.  从设备控制台的主导航窗格中，选择模型，如图 17-21 所示。

    ![AWS DeepRacer 汽车网络控制台上的模型上传菜单](img/00101.jpeg)

    ###### 图 17-21。AWS DeepRacer 汽车网络控制台上的模型上传菜单

1.  在模型页面上，选择上传，位于模型列表上方。

1.  从文件选择器中，导航到您下载模型工件的驱动器或共享位置，并选择要上传的模型。

1.  成功上传模型后，它将被添加到模型列表中，并可以加载到车辆的推理引擎中。

## 让 AWS DeepRacer 车辆自主驾驶

要开始自动驾驶，请将车辆放在物理赛道上，并执行以下操作：

1.  按照[说明](https://oreil.ly/OwzSz)登录到车辆的设备控制台，然后执行以下操作进行自动驾驶：

    1.  在“控制车辆”页面的控制部分中，选择“自动驾驶”，如图 17-22 所示。

        ![AWS DeepRacer 车辆 Web 控制台上的驾驶模式选择菜单](img/00053.jpeg)

        ###### 图 17-22\. AWS DeepRacer 车辆 Web 控制台上的驾驶模式选择菜单

1.  在“选择模型”下拉列表中（图 17-23），选择一个已上传的模型，然后选择“加载模型”。这将开始将模型加载到推理引擎中。该过程大约需要 10 秒钟才能完成。

1.  调整车辆的“最大速度”设置为训练模型中使用的最大速度的百分比。（诸如真实赛道的表面摩擦等因素可能会降低车辆的最大速度，使其低于训练中使用的最大速度。您需要进行实验以找到最佳设置。）

    ![AWS DeepRacer 车辆 Web 控制台上的模型选择菜单](img/00027.jpeg)

    ###### 图 17-23\. AWS DeepRacer 车辆 Web 控制台上的模型选择菜单

1.  选择“启动车辆”以使车辆自主驾驶。

1.  观看车辆在物理赛道上行驶，或在设备控制台上的流媒体视频播放器上观看。

1.  要停止车辆，请选择“停止车辆”。

### Sim2Real 转移

模拟器可能比真实世界更方便进行训练。在模拟器中可以轻松创建某些场景；例如，汽车与人或其他汽车碰撞——我们不希望在现实世界中这样做 :）。然而，在大多数情况下，模拟器不会具有我们在现实世界中的视觉保真度。此外，它可能无法捕捉真实世界的物理特性。这些因素可能会影响模拟器中环境的建模，因此，即使在模拟中取得了巨大成功，当代理在现实世界中运行时，我们可能会遇到失败。以下是处理模拟器限制的一些常见方法：

系统识别

建立一个数学模型来模拟真实环境，并校准物理系统，使其尽可能真实。

领域适应

将模拟域映射到真实环境，或反之亦然，使用正则化、GAN 或迁移学习等技术。

领域随机化

创建各种具有随机属性的模拟环境，并在所有这些环境的数据上训练模型。

在 DeepRacer 的背景下，模拟保真度是对真实世界的近似表示，物理引擎可能需要一些改进来充分模拟汽车的所有可能的物理特性。但深度强化学习的美妙之处在于我们不需要一切都完美。为了减轻影响汽车的大幅感知变化，我们做了两件重要的事情：a）我们不使用 RGB 图像，而是将图像转换为灰度，使模拟器和真实世界之间的感知差异变窄，b）我们有意使用浅层特征嵌入器；例如，我们只使用了几个 CNN 层；这有助于网络不完全学习模拟环境。相反，它迫使网络只学习重要特征。例如，在赛道上，汽车学会专注于使用白色赛道边缘标记进行导航。查看图 17-24，它使用一种称为 GradCAM 的技术生成图像中最具影响力部分的热图，以了解汽车正在寻找导航的位置。

![AWS DeepRacer 导航的 GradCAM 热图](img/00302.jpeg)

###### 图 17-24。AWS DeepRacer 导航的 GradCAM 热图

# 进一步探索

要继续冒险，您可以参与各种虚拟和实体赛车联赛。以下是一些探索的选项。

## DeepRacer 联赛

AWS DeepRacer 在 AWS 峰会和每月虚拟联赛中有一个实体联赛。要在当前虚拟赛道上比赛并赢取奖品，请访问联赛页面：[*https://console.aws.amazon.com/deepracer/home?region=us-east-1#leaderboards*](https://console.aws.amazon.com/deepracer/home?region=us-east-1#leaderboards)。

## 高级 AWS DeepRacer

我们了解到一些高级开发人员可能希望对模拟环境有更多控制，并且还具有定义不同神经网络架构的能力。为了实现这一体验，我们提供了一个基于[Jupyter Notebook](https://oreil.ly/Xto3S)的设置，您可以使用它来提供训练自定义 AWS DeepRacer 模型所需的组件。

## AI 驾驶奥林匹克

在 2018 年的 NeurIPS 上，AI 驾驶奥林匹克（AI-DO）以自动驾驶汽车的人工智能为重点推出。在第一届比赛中，这项全球竞赛在 Duckietown 平台上展示了车道跟随和车队管理挑战。这个平台有两个组成部分，Duckiebots（微型自动出租车）和 Duckietowns（包含道路、标志的微型城市环境）。Duckiebots 的工作是运送 Duckietown 的市民（小鸭子）。搭载了一个微型摄像头并在树莓派上运行计算，Duckiebots 配备了易于使用的软件，帮助从高中生到大学研究人员相对快速地运行他们的代码。自第一届 AI-DO 以来，这项比赛已经扩展到其他顶级人工智能学术会议，并现在包括涵盖 Duckietown 和 DeepRacer 平台的挑战。

![AI 驾驶奥林匹克的 Duckietown](img/00259.jpeg)

###### 图 17-25。AI 驾驶奥林匹克的 Duckietown

## DIY Robocars

[DIY Robocars Meetup](https://oreil.ly/SJOfT)最初在加利福尼亚州奥克兰开始，现在已经扩展到全球 50 多个 Meetup 小组。这些是有趣而引人入胜的社区，可以尝试与其他自动驾驶和自主无人车领域的爱好者合作。许多人每月举办比赛，是一个很好的场所来进行实体赛车比赛。

## Roborace

现在是发挥我们内心的迈克尔·舒马赫的时候了。赛车运动通常被认为是马力的竞争，现在 Roborace 正在将其转变为智能的竞争。Roborace 组织了全电动、自动驾驶赛车之间的比赛，如由丹尼尔·西蒙设计的时尚外观的 Robocar（以未来设计而闻名，如《创：战纪》中的 Tron Legacy Light Cycle），如图 17-26 所示。我们不再谈论微缩比例的汽车了。这些是全尺寸的、重达 1,350 公斤、长 4.8 米，能够达到 200 英里/小时（320 公里/小时）。最好的部分？我们不需要复杂的硬件知识来比赛。

这里真正的明星是人工智能开发人员。每个团队都会得到一辆相同的汽车，因此获胜的关键在于竞争对手编写的自主人工智能软件。车载传感器的输出，如摄像头、雷达、激光雷达、声纳和超声波传感器，都是可用的。为了进行高吞吐量的计算，汽车还装载了强大的 NVIDIA DRIVE 平台，能够每秒处理数万亿次浮点运算。我们需要做的就是构建算法，让汽车保持在赛道上，避免发生事故，并当然尽可能快地领先。

为了开始，Roborace 提供了一个赛车模拟环境，其中提供了真实汽车的精确虚拟模型。随之而来的是虚拟资格赛，获胜者有机会参加包括在电动方程式赛道上的真实比赛。到 2019 年，顶级赛车队已经将差距缩小到最佳人类表现的 5-10%之内。很快，开发人员将站在领奖台上，击败专业车手。最终，这样的比赛会带来创新，希望这些经验可以转化回自动驾驶汽车行业，使它们更安全、更可靠，同时性能更出色。

![Roborace 的 Robocar 由 Daniel Simon 设计](img/00221.jpeg)

###### 图 17-26。Roborace 的 Robocar 由 Daniel Simon 设计（图片由 Roborace 提供）

# 摘要

在前一章中，我们看了如何通过在模拟器内手动驾驶来训练自动驾驶车辆的模型。在本章中，我们探讨了与强化学习相关的概念，并学习了如何制定每个人都在关注的最终问题：如何让自动驾驶汽车学会驾驶。我们利用强化学习的魔力，将人类从循环中移除，教车辆在模拟器中独立驾驶。但为什么要限制在虚拟世界呢？我们将这些经验带入现实世界，并驾驶了一辆真实汽车。而这一切只需要一个小时！

深度强化学习是一个相对较新但令人兴奋的领域，值得进一步探索。最近对强化学习的扩展正在开辟新的问题领域，其应用可以实现大规模自动化。例如，分层强化学习使我们能够建模细粒度的决策制定。元强化学习使我们能够在不同环境中建模广义决策制定。这些框架让我们更接近模仿类似人类行为。毫不奇怪，许多机器学习研究人员认为强化学习有潜力让我们更接近*人工通用智能*，并开辟了以前被认为是科幻的道路。
