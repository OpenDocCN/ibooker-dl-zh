# 第13章。深度强化学习

[Nicholas Locascio](http://nicklocascio.com)

在本章中，我们将讨论强化学习，这是机器学习的一个分支，涉及通过互动和反馈学习。强化学习对于构建不仅可以感知和解释世界，还可以采取行动并与之互动的代理至关重要。我们将讨论如何将深度神经网络纳入强化学习框架，并讨论这一领域的最新进展和改进。

# 深度强化学习掌握Atari游戏

2014年，伦敦初创公司DeepMind在强化学习中应用深度神经网络取得了重大突破，令机器学习社区大为震惊，他们揭示了一个可以学会以超人类技能玩Atari游戏的深度神经网络。这个网络被称为*深度Q网络*（DQN），是强化学习与深度神经网络的首次大规模成功应用。DQN之所以如此引人注目，是因为相同的架构，没有任何改变，能够学会玩49种不同的Atari游戏，尽管每个游戏都有不同的规则、目标和游戏结构。为了实现这一壮举，DeepMind汇集了许多传统的强化学习思想，同时也开发了一些关键的新技术，这些技术对DQN的成功至关重要。在本章的后面，我们将实现DQN，如《自然》杂志上描述的那样，“通过深度强化学习实现人类水平控制”。^([1](ch13.xhtml#idm45934163590304)) 但首先，让我们深入了解强化学习（[图13-1](#fig0801)）。

![](Images/fdl2_1301.png)

###### 图13-1。一个深度强化学习代理在玩Breakout游戏

# 什么是强化学习？

强化学习在本质上是通过与环境互动学习。这个学习过程涉及到一个*代理*，一个*环境*和一个*奖励信号*。代理选择在环境中采取行动，根据行动获得奖励。演员选择行动的方式被称为*策略*。代理希望增加它接收到的奖励，因此必须学习与环境互动的最佳策略（[图13-2](#fig0802)）。

强化学习与我们迄今为止涵盖的其他学习类型不同。在传统的监督学习中，我们被给定数据和标签，并被要求根据数据预测标签。在无监督学习中，我们只给定数据，并被要求发现数据中的潜在结构。在强化学习中，我们既没有数据也没有标签。我们的学习信号来自环境给予代理的奖励。

![](Images/fdl2_1302.png)

###### 图13-2。强化学习设置

强化学习对AI社区的许多人来说是令人兴奋的，因为它是创建智能代理的通用框架。给定一个环境和一些奖励，代理学会与该环境互动以最大化其总奖励。这种学习更符合人类的发展方式。是的，我们可以通过在成千上万张图像上进行训练，构建一个非常准确的模型来区分狗和猫。但你不会发现这种方法在任何小学中使用。人类通过与环境互动来学习世界的表示，以便做出决策。

此外，强化学习应用处于许多尖端技术的前沿，包括自动驾驶汽车、机器人电机控制、游戏玩法、空调控制、广告放置优化和股票交易策略。

作为一个说明性练习，我们将解决一个称为平衡杆的简单强化学习和控制问题。在这个问题中，有一个连接在铰链上的杆的小车，因此杆可以围绕小车摆动。有一个代理可以控制小车，将其向左或向右移动。有一个环境，当杆指向上方时，奖励代理，当杆倒下时，惩罚代理（[图13-3](#fig0803)）。

![](Images/fdl2_1303.png)

###### 图13-3。一个简单的强化学习代理：平衡杆^([3](ch13.xhtml#idm45934163560160))

# 马尔可夫决策过程

我们的平衡杆示例有一些重要的元素，我们将其形式化为*马尔可夫决策过程*（MDP）。这些元素包括：

状态

小车在x平面上有一系列可能的位置。同样，杆有一系列可能的角度。

动作

代理可以通过将小车向左或向右移动来采取行动。

状态转移

当代理行动时，环境会发生变化：小车移动，杆的角度和速度也会改变。

奖励

如果代理平衡杆得当，它会获得正面奖励。如果杆倒下，代理会受到负面奖励。

MDP定义如下：

+   <math alttext="upper S"><mi>S</mi></math>，一组可能状态的有限集

+   <math alttext="upper A"><mi>A</mi></math>，一组有限的动作

+   <math alttext="upper P left-parenthesis r comma s prime vertical-bar s comma a right-parenthesis"><mrow><mi>P</mi> <mo>(</mo> <mi>r</mi> <mo>,</mo> <msup><mi>s</mi> <mo>'</mo></msup> <mo>|</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow></math>，状态转移函数

+   <math alttext="upper R"><mi>R</mi></math>，奖励函数

MDP为在给定环境中建模决策制定提供了一个数学框架。[图13-4](#fig0804)显示了一个示例，其中圆圈代表环境的状态，菱形代表可以采取的动作，从菱形到圆圈的边表示从一个状态转移到下一个状态的转换。沿着这些边的数字表示采取某个动作的概率，箭头末端的数字表示给予代理的奖励，用于进行给定转换。

![](Images/fdl2_1304.png)

###### 图13-4。MDP示例

当代理在MDP框架中采取行动时，它形成一个*情节*。一个情节由一系列状态、动作和奖励的元组组成。情节运行直到环境达到终止状态，例如Atari游戏中的“游戏结束”屏幕，或者在我们的杆-小车示例中，杆触地。以下方程式显示了情节中的变量：

<math alttext="left-parenthesis s 0 comma a 0 comma r 0 right-parenthesis comma left-parenthesis s 1 comma a 1 comma r 1 right-parenthesis comma ellipsis left-parenthesis s Subscript n Baseline comma a Subscript n Baseline comma r Subscript n Baseline right-parenthesis"><mrow><mrow><mo>(</mo> <msub><mi>s</mi> <mn>0</mn></msub> <mo>,</mo> <msub><mi>a</mi> <mn>0</mn></msub> <mo>,</mo> <msub><mi>r</mi> <mn>0</mn></msub> <mo>)</mo></mrow> <mo>,</mo> <mrow><mo>(</mo> <msub><mi>s</mi> <mn>1</mn></msub> <mo>,</mo> <msub><mi>a</mi> <mn>1</mn></msub> <mo>,</mo> <msub><mi>r</mi> <mn>1</mn></msub> <mo>)</mo></mrow> <mo>,</mo> <mo>...</mo> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>n</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>n</mi></msub> <mo>,</mo> <msub><mi>r</mi> <mi>n</mi></msub> <mo>)</mo></mrow></mrow></math>

在杆-小车中，我们的环境状态可以是小车的位置和杆的角度的元组，如下所示：（<math alttext="x Subscript c a r t"><msub><mi>x</mi> <mrow><mi>c</mi><mi>a</mi><mi>r</mi><mi>t</mi></mrow></msub></math>，<math alttext="theta Subscript p o l e"><msub><mi>θ</mi> <mrow><mi>p</mi><mi>o</mi><mi>l</mi><mi>e</mi></mrow></msub></math>）。

## 策略

MDP的目标是为我们的代理找到一个最优策略。*策略*是我们的代理根据其当前状态采取行动的方式。形式上，策略可以表示为一个函数<math alttext="pi"><mi>π</mi></math>，选择代理在状态<math alttext="s"><mi>s</mi></math>中将采取的动作<math alttext="a"><mi>a</mi></math>。

我们MDP的目标是找到一个策略，以最大化预期的未来回报：

<math alttext="max Underscript pi Endscripts upper E left-bracket upper R 0 plus upper R 1 plus ellipsis upper R Subscript t Baseline vertical-bar pi right-bracket"><mrow><msub><mo movablelimits="true" form="prefix">max</mo> <mi>π</mi></msub> <mi>E</mi> <mrow><mo>[</mo> <msub><mi>R</mi> <mn>0</mn></msub> <mo>+</mo> <msub><mi>R</mi> <mn>1</mn></msub> <mo>+</mo> <mo>...</mo> <msub><mi>R</mi> <mi>t</mi></msub> <mo>|</mo> <mi>π</mi> <mo>]</mo></mrow></mrow></math>

在这个目标中，R代表每个情节的*未来回报*。让我们准确定义未来回报的含义。

## 未来回报

未来回报是我们如何考虑未来奖励的方式。选择最佳行动需要考虑该行动的即时效果以及长期后果。有时，最佳行动实际上具有负面的即时效果，但具有更好的长期结果。例如，一个根据海拔高度奖励的登山代理实际上可能必须下山才能到达山顶的更好路径。

因此，我们希望我们的代理人优化*未来回报*。为了做到这一点，代理人必须考虑其行动的未来后果。例如，在乒乓球比赛中，当球传入对手的球门时，代理人会获得奖励。然而，导致这种奖励的行动（使球拍定位以打出得分击球的输入）发生在获得奖励之前的许多时间步之前。每个行动的奖励都是延迟的。

我们可以通过构建每个时间步的*回报*来将延迟奖励纳入我们的整体奖励信号中，该回报考虑了未来奖励以及即时奖励。计算一个时间步的*未来回报*的一个天真的方法可能是一个简单的总和，如下所示：

<math alttext="upper R Subscript t Baseline equals sigma-summation Underscript k equals 0 Overscript upper T Endscripts r Subscript t plus k"><mrow><msub><mi>R</mi> <mi>t</mi></msub> <mo>=</mo> <mrow><munderover><mo>∑</mo> <mrow><mi>k</mi><mo>=</mo><mn>0</mn></mrow> <mi>T</mi></munderover> <msub><mi>r</mi> <mrow><mi>t</mi><mo>+</mo><mi>k</mi></mrow></msub></mrow></mrow></math>

我们可以计算所有回报R，其中R={R0，R1，...，Ri，...，Rn}，使用以下代码：

```py
def calculate_naive_returns(rewards):
""" Calculates a list of naive returns given a 
    list of rewards."""
    total_returns = np.zeros(len(rewards))
    total_return = 0.0
    for t in range(len(rewards), 0):
        total_return = total_return + reward
        total_returns[t] = total_return
    return total_returns
```

这种天真的方法成功地将未来奖励纳入，以便代理人可以学习一个最优的全局策略。这种方法将未来奖励与即时奖励同等看待。然而，对所有奖励的平等考虑是有问题的。在无限时间步长下，这个表达式可能会发散到无穷大，因此我们必须找到一种方法来限制它。此外，在每个时间步长上平等考虑，代理人可以优化未来奖励，我们将学习到一个缺乏紧迫感或时间敏感性的策略。

相反，我们应该稍微低估未来奖励，以便迫使我们的代理人学会快速获得奖励。我们通过一种称为*折现未来回报*的策略来实现这一点。

## 折现未来回报

为了实现折现未来回报，我们将当前状态的奖励按照折现因子γ的当前时间步幂进行缩放。通过这种方式，我们惩罚那些在获得正面奖励之前采取许多行动的代理人。折现奖励使我们的代理人倾向于更喜欢在即时未来获得奖励，这有利于学习一个良好的策略。我们可以将奖励表达如下：

<math alttext="upper R Subscript t Baseline equals sigma-summation Underscript k equals 0 Overscript upper T Endscripts gamma Superscript t Baseline r Subscript t plus k plus 1"><mrow><msub><mi>R</mi> <mi>t</mi></msub> <mo>=</mo> <mrow><munderover><mo>∑</mo> <mrow><mi>k</mi><mo>=</mo><mn>0</mn></mrow> <mi>T</mi></munderover> <mrow><msup><mi>γ</mi> <mi>t</mi></msup> <msub><mi>r</mi> <mrow><mi>t</mi><mo>+</mo><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow></mrow></mrow></math>

折现因子，γ，代表我们希望实现的折现水平，可以介于0和1之间。高γ意味着很少折现，低γ提供了很多折现。典型的γ超参数设置在0.99和0.97之间。

我们可以这样实现折现回报：

```py
def discount_rewards(rewards, gamma=0.98):
    discounted_returns = [0 for _ in rewards]
    discounted_returns[-1] = rewards[-1]
    for t in range(len(rewards)-2, -1, -1): # iterate backwards
        discounted_returns[t] = rewards[t] + 
          discounted_returns[t+1]*gamma
    return discounted_returns

```

# 探索与利用

强化学习基本上是一个反复试验的过程。在这样的框架中，一个害怕犯错误的代理人可能会变得非常问题。考虑以下情景。一个老鼠被放置在迷宫中，我们的代理人必须控制老鼠以最大化奖励。如果老鼠喝到水，它会获得+1的奖励；如果老鼠到达毒物容器（红色），它会获得-10的奖励；如果老鼠得到奶酪，它会获得+100的奖励。一旦获得奖励，该情节就结束了。最佳策略涉及老鼠成功地导航到奶酪并吃掉它。

![](Images/fdl2_1305.png)

###### 图13-5。许多老鼠发现自己陷入的困境。

在第一集中，老鼠选择了左侧路线，踩到了陷阱，获得了-10的奖励。在第二集中，老鼠避开了左侧路径，因为它导致了如此负面的奖励，并立即喝了右侧的水，获得了+1的奖励。经过两集，老鼠似乎找到了一个好的策略。它继续在后续集中遵循其学到的策略，并可靠地获得适度的+1奖励。由于我们的代理采用了贪婪策略——始终选择模型的最佳动作——它被困在了一个*局部最大值*的策略中。

为了避免这种情况，代理可能有必要偏离模型的建议，采取次优动作以便更多地*探索*环境。因此，我们的代理可能选择向左转而不是立即向右转以*利用*环境获取水和可靠的+1奖励，而是冒险进入更危险的区域寻找更优的策略。过多的探索会导致我们的代理无法优化任何奖励。探索不足可能导致我们的代理陷入局部最小值。*探索与利用*的这种平衡对于学习成功的策略至关重要。

## ε-Greedy

平衡探索和利用困境的一种策略被称为*ε-greedy*。ε-greedy是一种简单的策略，每一步都要做出选择，要么采取代理的顶级推荐动作，要么采取随机动作。代理采取随机动作的概率是称为ε的值。

我们可以这样实现ε-greedy：

```py
def epsilon_greedy_action(action_distribution,
                          epsilon=1e-1):
    action_distribution = action_distribution.detach().numpy()
    if random.random() < epsilon:
        return np.argmax(np.random.random(
           action_distribution.shape))
    else:
        return np.argmax(action_distribution)

```

## 退火ε-Greedy

在训练强化学习模型时，通常我们希望在开始时进行更多的探索，因为我们的模型对世界知之甚少。后来，一旦我们的模型看到了环境的大部分并学到了一个好的策略，我们希望我们的代理更加信任自己以进一步优化其策略。为了实现这一点，我们放弃了固定的ε的概念，而是随着时间逐渐降低它，使其从低值开始，并在每次训练集之后增加一个因子。典型的退火ε-greedy场景设置包括在10,000个场景中从0.99退火到0.1。我们可以这样实现退火：

```py
def epsilon_greedy_action_annealed(action_distribution,
                                   percentage,
                                   epsilon_start=1.0,
                                   epsilon_end=1e-2):
    action_distribution = action_distribution.detach().numpy()
    annealed_epsilon = (epsilon_start*(1.0-percentage) +
                        epsilon_end*percentage)
    if random.random() < annealed_epsilon:
        return np.argmax(np.random.random(
          action_distribution.shape))
    else:
        return np.argmax(action_distribution)

```

# 策略学习与值学习

到目前为止，我们已经定义了强化学习的设置，讨论了折现未来回报，并看了探索与利用的权衡。我们还没有讨论的是我们实际上将如何教导代理最大化其奖励。对此的方法分为两大类：*策略学习*和*值学习*。在策略学习中，我们直接学习最大化奖励的策略。在值学习中，我们学习每个状态+动作对的值。如果你试图学会骑自行车，策略学习方法是考虑在你向左倾斜时如何踩右脚踏板来纠正你的方向。如果你试图用值学习方法学会骑自行车，你会为不同的自行车方向和你可以在这些位置采取的行动分配一个分数。我们将在本章中涵盖这两种方法，所以让我们从策略学习开始。

在典型的监督学习中，我们可以使用随机梯度下降来更新我们的参数，以最小化从网络输出和真实标签计算出的损失。我们正在优化表达式：

<math alttext="arg min Underscript theta Endscripts sigma-summation Underscript i Endscripts log p left-parenthesis y Subscript i Baseline bar x Subscript i Baseline semicolon theta right-parenthesis"><mrow><mo form="prefix">arg</mo> <msub><mo movablelimits="true" form="prefix">min</mo> <mi>θ</mi></msub> <msub><mo>∑</mo> <mi>i</mi></msub> <mo form="prefix">log</mo> <mi>p</mi> <mrow><mo>(</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>∣</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>;</mo> <mi>θ</mi> <mo>)</mo></mrow></mrow></math>

在强化学习中，我们没有真正的标签，只有奖励信号。然而，我们仍然可以使用SGD来优化我们的权重，使用一种称为*策略梯度*的方法。我们可以使用代理所采取的动作以及与这些动作相关的回报，来鼓励我们的模型权重采取导致高奖励的良好动作，并避免导致低奖励的不良动作。我们优化的表达式是：

<math alttext="arg min Underscript theta Endscripts minus sigma-summation Underscript i Endscripts upper R Subscript i Baseline log p left-parenthesis y Subscript i Baseline bar x Subscript i Baseline semicolon theta right-parenthesis"><mrow><mo form="prefix">arg</mo> <msub><mo movablelimits="true" form="prefix">min</mo> <mi>θ</mi></msub> <mo>-</mo> <msub><mo>∑</mo> <mi>i</mi></msub> <msub><mi>R</mi> <mi>i</mi></msub> <mo form="prefix">log</mo> <mi>p</mi> <mrow><mo>(</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>∣</mo> <msub><mi>x</mi> <mi>i</mi></msub> <mo>;</mo> <mi>θ</mi> <mo>)</mo></mrow></mrow></math>

其中<math alttext="y Subscript i"><msub><mi>y</mi> <mi>i</mi></msub></math>是代理在时间步<math alttext="t"><mi>t</mi></math>采取的动作，<math alttext="upper R Subscript i"><msub><mi>R</mi> <mi>i</mi></msub></math>是我们的折现未来回报。通过这种方式，我们通过我们的回报值来缩放我们的损失，因此，如果模型选择导致负回报的动作，这将导致更大的损失。此外，如果模型对该错误决策有信心，它将受到更严厉的惩罚，因为我们考虑了模型选择该动作的对数概率。有了我们定义的损失函数，我们可以应用SGD来最小化我们的损失并学习一个良好的策略。

# 使用策略梯度的Pole-Cart

我们将实现一个策略梯度代理来解决pole-cart，这是一个经典的强化学习问题。我们将使用OpenAI Gym为此任务专门创建的环境。

## OpenAI Gym

OpenAI Gym是一个用于开发强化学习代理的Python工具包。OpenAI Gym提供了一个易于使用的接口，用于与各种环境进行交互。它包含了一百多个常见强化学习环境的开源实现。OpenAI Gym通过处理环境模拟方面的一切来加速强化学习代理的开发，使研究人员可以专注于他们的代理和学习算法。OpenAI Gym的另一个好处是，研究人员可以公平地比较和评估他们的结果，因为他们都可以使用相同的标准化环境来执行任务。我们将使用OpenAI Gym中的pole-cart环境来创建一个可以轻松与该环境交互的代理。

## 创建代理

创建一个可以与OpenAI环境交互的代理，我们将定义一个名为`PGAgent`的类，其中包含我们的模型架构、模型权重和超参数：

```py
from torch import optim
class PGAgent(object):
    def __init__(self, state_size, num_actions,
                 hidden_size,
                 learning_rate=1e-3,
                 explore_exploit_setting= \
                 'epsilon_greedy_annealed_1.0->0.001'):
        self.state_size = state_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.explore_exploit_setting = \
                        explore_exploit_setting
        self.build_model()

    def build_model(self):
      self.model = torch.nn.Sequential(
        nn.Linear(self.state_size, self.hidden_size),
        nn.Linear(self.hidden_size, self.hidden_size),
        nn.Linear(self.hidden_size, self.num_actions),
        nn.Softmax(dim=0))

    def train(self, state, action_input, reward_input):
        state = torch.tensor(state).float()
        action_input = torch.tensor(action_input).long()
        reward_input = torch.tensor(reward_input).float()
        self.output = self.model(state)
        # Select the logits related to the action taken
        logits_for_actions = self.output.gather(1,
                                           action_input.view(-1,1))

        self.loss = -torch.mean(
            torch.log(logits_for_actions) * reward_input)
        self.loss.backward()
        self.optimizer = optim.Adam(self.model.parameters())
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self.loss.item()

    def sample_action_from_distribution(self,
                                        action_distribution,
                                        epsilon_percentage):
        # Choose an action based on the action probability
        # distribution and an explore vs exploit
        if self.explore_exploit_setting == 'greedy':
              action = epsilon_greedy_action(action_distribution,
                                             0.00)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.05':
              action = epsilon_greedy_action(action_distribution,
                                             0.05)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.25':
              action = epsilon_greedy_action(action_distribution,
                                             0.25)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.50':
              action = epsilon_greedy_action(action_distribution,
                                             0.50)
        elif self.explore_exploit_setting == 'epsilon_greedy_0.90':
              action = epsilon_greedy_action(action_distribution,
                                             0.90)
        elif self.explore_exploit_setting == \
          'epsilon_greedy_annealed_1.0->0.001':
              action = epsilon_greedy_action_annealed(
                  action_distribution,
                  epsilon_percentage, 1.0,0.001)
        elif self.explore_exploit_setting == \
          'epsilon_greedy_annealed_0.5->0.001':
              action = epsilon_greedy_action_annealed(
                  action_distribution,
                  epsilon_percentage, 0.5, 0.001)
        elif self.explore_exploit_setting == \
          'epsilon_greedy_annealed_0.25->0.001':
              action = epsilon_greedy_action_annealed(
                  action_distribution,
                  epsilon_percentage, 0.25, 0.001)
        return action

    def predict_action(self, state, epsilon_percentage):
        action_distribution = self.model(
                                 torch.from_numpy(state).float())
        action = self.sample_action_from_distribution(
            action_distribution, epsilon_percentage)
        return action

```

## 构建模型和优化器

让我们分解一些重要的函数。在`build_model()`中，我们将我们的模型架构定义为一个三层神经网络。模型返回一个包含三个节点的层，每个节点代表模型的动作概率分布。在`build_training()`中，我们实现了我们的策略梯度优化器。我们表达了我们的目标损失，如我们所讨论的，通过将模型对动作的预测概率与采取该动作获得的回报进行缩放，并将所有这些相加形成一个小批量。有了我们定义的目标，我们可以使用`torch.optim.AdamOptimizer`，它将根据梯度调整我们的权重以最小化我们的损失。

## 采样动作

我们定义了`predict_action`函数，根据模型的动作概率分布输出对动作进行采样。我们支持我们讨论过的各种采样策略，以平衡探索与利用，包括贪婪、ϵ贪婪和ϵ贪婪退火。

## 跟踪历史

我们将从多个情节运行中聚合我们的梯度，因此跟踪状态、动作和奖励元组将非常有用。为此，我们实现了一个情节历史和记忆：

```py
class EpisodeHistory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []

    def add_to_history(self, state, action, reward,
      state_prime):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_primes.append(state_prime)

class Memory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []

    def reset_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []

    def add_episode(self, episode):
        self.states += episode.states
        self.actions += episode.actions
        self.rewards += episode.rewards
        self.discounted_returns += episode.discounted_returns

```

## 策略梯度主要函数

让我们在我们的主函数中将所有这些放在一起，该函数将为CartPole创建一个OpenAI Gym环境，创建我们的代理的一个实例，并让我们的代理与CartPole环境进行交互和训练：

```py
# Configure Settings
#total_episodes = 5000
total_episodes = 16
total_steps_max = 10000
epsilon_stop = 3000
train_frequency = 8
max_episode_length = 500
render_start = -1
should_render = False

explore_exploit_setting = 'epsilon_greedy_annealed_1.0->0.001'

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]  # 4 for
                                              # CartPole-v0
num_actions = env.action_space.n  # 2 for CartPole-v0

solved = False
agent = PGAgent(state_size=state_size,
                num_actions=num_actions,
                hidden_size=16,
                explore_exploit_setting= \
                  explore_exploit_setting)

episode_rewards = []
batch_losses = []

global_memory = Memory()
steps = 0
for i in range(total_episodes):
  state = env.reset()
  episode_reward = 0.0
  episode_history = EpisodeHistory()
  epsilon_percentage = float(min(i/float(epsilon_stop), 1.0))

  for j in range(max_episode_length):
      action = agent.predict_action(state, epsilon_percentage)
      state_prime, reward, terminal, _ = env.step(action)

      episode_history.add_to_history(
          state, action, reward, state_prime)
      state = state_prime
      episode_reward += reward
      steps += 1

      if j == (max_episode_length - 1):
            terminal = True

      if terminal:
          episode_history.discounted_returns = \
            discount_rewards(episode_history.rewards)
          global_memory.add_episode(episode_history)

          # every 8th episode train the NN
          # train on all actions from episodes in memory, 
          # then reset memory
          if np.mod(i, train_frequency) == 0:
            reward_input = global_memory.discounted_returns
            action_input = global_memory.actions
            state = global_memory.states

            # train step
            batch_loss = agent.train(state, action_input, 
                                     reward_input)
              # print(f'Batch loss: {batch_loss}')
              # batch_losses.append(batch_loss)
            global_memory.reset_memory()

          episode_rewards.append(episode_reward)

          if i % 10 == 0:
              mean_rewards = torch.mean(torch.tensor(
                                         episode_rewards[:-10]))
              if mean_rewards > 10.0:
                  solved = True
              else:
                  solved = False
              print(f'Solved: {solved} Mean Reward: {mean_rewards}')
          break # stop playing if terminal

  print(f'Episode[{i}]: {len(episode_history.actions)} \
          actions {episode_reward} reward')

```

这段代码将训练一个CartPole代理成功并持续地平衡杆。

## PGAgent在Pole-Cart上的表现

[图13-6](#explore_exploit_configurations)是我们代理在每个训练步骤中的平均奖励的图表。我们尝试了8种不同的采样方法，并通过从1.0到0.001的ϵ贪婪退火获得了最佳结果。

![](Images/fdl2_1306.png)

###### 图13-6。探索-利用配置影响学习的速度和效果如何

请注意，从整体上看，标准的ϵ贪婪表现非常糟糕。让我们讨论一下可能的原因。当ϵ设置为0.9时，我们90%的时间会随机采取行动。即使模型学会执行完美的动作，我们仍然只会使用这些动作的10%。另一方面，当ϵ为0.05时，我们大部分时间会采取模型认为是最佳动作。这种表现稍微好一些，但会陷入局部奖励最小值，因为它缺乏探索其他策略的能力。因此，ϵ为0.05或0.9都不能给我们带来很好的结果。前者过于强调探索，后者则太少。这就是为什么ϵ退火是如此强大的采样策略。它允许模型早期探索，晚期利用，这对学习良好策略至关重要。

# 信任区域策略优化

*信任区域策略优化*，简称*TRPO*，是一个框架，确保在每次训练步骤中防止策略发生过大变化的同时实现策略改进。经验表明，TRPO在许多政策梯度和政策迭代方法中表现优异，使研究人员能够有效地学习复杂的、非线性的策略（通常由大型神经网络参数化），这是以前通过梯度方法无法实现的。在本节中，我们将介绍TRPO并更详细地描述其目标。

在每次训练步骤中防止策略发生过大变化的想法并不新颖——大多数正则化优化程序通过惩罚参数的范数间接实现这一点，例如，全局确保参数的范数不会过高。当然，在正则化优化也可以被表述为约束优化的情况下（其中参数向量的范数有明确的界限，例如，L2正则化线性回归），我们直接等价于在每次训练步骤中防止策略发生过大变化的想法。参数范数的每步变化受到约束范围的限制，因为所有可能的参数值必须落在这个范围内。对于那些感兴趣的人，我建议进一步研究线性回归中Tikhonov和Ivanov正则化之间的等价性。

在每次训练步骤中防止策略发生过大变化具有正则化优化的标准效果：它促进训练的稳定性，这对于防止过拟合到新数据是理想的。我们如何定义策略的变化？策略只是给定状态的动作空间上的离散概率分布，<math alttext="pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis"><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>，我们可以使用在[第2章](ch02.xhtml#fundamentals-of-proba)中介绍的不相似性概念。原始的TRPO论文介绍了当前策略和新策略之间的平均KL散度（在所有可能的状态上）的界限。

现在我们已经介绍了TRPO约束优化的约束部分，我们将激励并定义目标函数。

让我们回顾并介绍一些术语：

<math alttext="eta 左括号 pi 右括号 等于"><mrow><mi>η</mi> <mo>(</mo> <mi>π</mi> <mo>)</mo> <mo>=</mo></mrow></math>   <math alttext="双划线上标 E 下标 s 0 逗号 a 0 逗号 s 1 逗号 a 1 逗号 省略 基线左方括号 sigma-求和下标 t 等于 0 上标 正无穷大 上标 gamma 上标 t 基线 r 左括号 s 下标 t 右括号 右方括号"><mrow><msub><mi>𝔼</mi> <mrow><msub><mi>s</mi> <mn>0</mn></msub> <mo>,</mo><msub><mi>a</mi> <mn>0</mn></msub> <mo>,</mo><msub><mi>s</mi> <mn>1</mn></msub> <mo>,</mo><msub><mi>a</mi> <mn>1</mn></msub> <mo>,</mo><mo>...</mo></mrow></msub> <mrow><mo>[</mo> <msubsup><mo>∑</mo> <mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow> <mi>∞</mi></msubsup> <msup><mi>γ</mi> <mi>t</mi></msup> <mi>r</mi> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

<math alttext="上标 Q 下标 π 基线左括号 s 下标 t 逗号 a 下标 t 右括号 等于"><mrow><msub><mi>Q</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>)</mo></mrow> <mo>=</mo></mrow></math>   <math alttext="双划线上标 E 下标 s 上标 t 加 1 下标 逗号 a 上标 t 加 1 下标 逗号 省略 基线左方括号 sigma-求和下标 l 等于 0 上标 正无穷大 上标 gamma 上标 l 基线 r 左括号 s 下标 t 加 l 基线 右括号 右方括号"><mrow><msub><mi>𝔼</mi> <mrow><msub><mi>s</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>,</mo><msub><mi>a</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>,</mo><mo>...</mo></mrow></msub> <mrow><mo>[</mo> <msubsup><mo>∑</mo> <mrow><mi>l</mi><mo>=</mo><mn>0</mn></mrow> <mi>∞</mi></msubsup> <msup><mi>γ</mi> <mi>l</mi></msup> <mi>r</mi> <mrow><mo>(</mo> <msub><mi>s</mi> <mrow><mi>t</mi><mo>+</mo><mi>l</mi></mrow></msub> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

<math alttext="上标 V 下标 π 基线左括号 s 下标 t 右括号"><mrow><msub><mi>V</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>)</mo></mrow></mrow></math>   <math alttext="等于双划线上标 E 下标 a 上标 t 下标 逗号 s 下标 t 加 1 下标 逗号 a 下标 t 加 1 省略 基线左方括号 sigma-求和下标 l 等于 0 上标 正无穷大 上标 gamma 上标 l 基线 r 左括号 s 下标 t 加 l 基线 右括号 右方括号"><mrow><mo>=</mo> <msub><mi>𝔼</mi> <mrow><msub><mi>a</mi> <mi>t</mi></msub> <mo>,</mo><msub><mi>s</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>,</mo><msub><mi>a</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>...</mo></mrow></msub> <mrow><mo>[</mo> <msubsup><mo>∑</mo> <mrow><mi>l</mi><mo>=</mo><mn>0</mn></mrow> <mi>∞</mi></msubsup> <msup><mi>γ</mi> <mi>l</mi></msup> <mi>r</mi> <mrow><mo>(</mo> <msub><mi>s</mi> <mrow><mi>t</mi><mo>+</mo><mi>l</mi></mrow></msub> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

<math alttext="上标 A 下标 π 基线左括号 s 逗号 a 右括号 等于"><mrow><msub><mi>A</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>=</mo></mrow></math>   <math alttext="上标 Q 下标 π 基线左括号 s 逗号 a 右括号 减上标 V 下标 π 基线左括号 s 右括号"><mrow><msub><mi>Q</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>-</mo> <msub><mi>V</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="rho Subscript pi Baseline left-parenthesis s right-parenthesis equals sigma-summation Underscript i equals 0 Overscript normal infinity Endscripts gamma Superscript i Baseline upper P left-parenthesis s Subscript t Baseline equals s right-parenthesis"><mrow><msub><mi>ρ</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow> <mo>=</mo> <msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>0</mn></mrow> <mi>∞</mi></msubsup> <msup><mi>γ</mi> <mi>i</mi></msup> <mi>P</mi> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>=</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>

第一个术语是表示*预期折扣奖励*的<math alttext="eta left-parenthesis pi right-parenthesis"><mrow><mi>η</mi> <mo>(</mo> <mi>π</mi> <mo>)</mo></mrow></math>，我们在之前讨论未来折扣奖励时已经看到了该术语的有限时间视角版本。在这里，我们不再只看一个单一轨迹，而是通过我们的策略<math alttext="pi"><mi>π</mi></math>定义的所有可能轨迹上的期望。通常情况下，我们可以通过使用<math alttext="pi"><mi>π</mi></math>采样轨迹来估计这个期望。第二个术语是*Q函数* <math alttext="upper Q Subscript pi Baseline left-parenthesis s Subscript t Baseline comma a Subscript t Baseline right-parenthesis"><mrow><msub><mi>Q</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>)</mo></mrow></mrow></math>，将在[“Q学习和深度Q网络”](#q-learning-sect)中更详细地讨论，它看起来与前一个术语非常相似，但实际上被定义为在时间*t*时的预期折扣回报，假设我们处于某个状态<math alttext="s Subscript t"><msub><mi>s</mi> <mi>t</mi></msub></math>并在该状态执行一个定义好的动作<math alttext="a Subscript t"><msub><mi>a</mi> <mi>t</mi></msub></math>。我们再次使用我们的策略<math alttext="pi"><mi>π</mi></math>来计算期望。请注意，时间*t*实际上并不太重要，因为我们只考虑无限时间视角，并且从*t*时刻的预期折扣回报而不是从轨迹开始处计算。

第三项是<math alttext="upper V Subscript pi Baseline left-parenthesis s Subscript t Baseline right-parenthesis"><mrow><msub><mi>V</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>)</mo></mrow></mrow></math>，或者在特定时间*t*的特定状态的*值函数*。值函数实际上可以更简洁地写为<math alttext="upper V Subscript pi Baseline left-parenthesis s Subscript t Baseline right-parenthesis equals double-struck upper E Subscript a Sub Subscript t Baseline left-bracket upper Q Subscript pi Baseline left-parenthesis s Subscript t Baseline comma a Subscript t Baseline right-parenthesis right-bracket"><mrow><msub><mi>V</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>)</mo></mrow> <mo>=</mo> <msub><mi>𝔼</mi> <msub><mi>a</mi> <mi>t</mi></msub></msub> <mrow><mo>[</mo> <msub><mi>Q</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>，或者关于<math alttext="pi left-parenthesis a Subscript t Baseline vertical-bar s Subscript t Baseline right-parenthesis"><mrow><mi>π</mi> <mo>(</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>|</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>)</mo></mrow></math>的Q函数的期望。实质上，Q函数假设我们在状态<math alttext="s Subscript t"><msub><mi>s</mi> <mi>t</mi></msub></math>中采取定义的动作<math alttext="a Subscript t"><msub><mi>a</mi> <mi>t</mi></msub></math>，而值函数将<math alttext="a Subscript t"><msub><mi>a</mi> <mi>t</mi></msub></math>作为变量。因此，要得到值函数，我们只需要对关于<math alttext="a Subscript t"><msub><mi>a</mi> <mi>t</msub></math>的分布的期望，知道当前状态<math alttext="s Subscript t"><msub><mi>s</mi> <mi>t</msub></math>。结果是Q函数的加权平均值，其中权重为<math alttext="pi left-parenthesis a Subscript t Baseline vertical-bar s Subscript t Baseline right-parenthesis"><mrow><mi>π</mi> <mo>(</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>|</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>)</mo></mrow></math>。实质上，这一项捕捉了我们期望在某个状态<math alttext="s Subscript t"><msub><mi>s</mi> <mi>t</msub></math>开始看到的平均未来折现回报。

第四项是<math alttext="upper A Subscript pi Baseline left-parenthesis s comma a right-parenthesis"><mrow><msub><mi>A</mi> <mi>π</mi></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow></mrow></math>，或者*优势函数*。请注意，出于前面提到的原因，我们现在已经放弃了时间*t*。优势函数的直觉是，在固定策略<math alttext="pi"><mi>π</mi></math>下，量化在当前状态*s*中采取特定动作<math alttext="a"><mi>a</mi></math>后让轨迹继续进行的好处，相对于完全不受限制地让轨迹从当前状态*s*继续进行。更简洁地说，它定义了在长期内相对于平均值，最初在状态*s*中采取动作*a*的好坏程度。

最后一项，或者*未归一化的折现访问频率*，重新引入了时间项*t*。这一项是关于从开始到无穷大的每个时间*t*处于状态*s*的概率的函数。这一项将在我们定义目标函数时很重要。原始的TRPO论文选择通过最大化这个目标函数来优化模型参数：

<math alttext="upper L Subscript theta Sub Subscript o l d Baseline left-parenthesis theta right-parenthesis equals sigma-summation Underscript s Endscripts rho Subscript theta Sub Subscript o l d Baseline left-parenthesis s right-parenthesis sigma-summation Underscript a Endscripts pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis upper A Subscript theta Sub Subscript o l d Baseline left-parenthesis s comma a right-parenthesis"><mrow><msub><mi>L</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>θ</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mo>∑</mo> <mi>s</mi></msub> <msub><mi>ρ</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow> <msub><mo>∑</mo> <mi>a</mi></msub> <msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow> <msub><mi>A</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="theta Subscript n e w Baseline equals argmax Subscript theta Baseline upper L Subscript theta Sub Subscript o l d Baseline left-parenthesis theta right-parenthesis"><mrow><msub><mi>θ</mi> <mrow><mi>n</mi><mi>e</mi><mi>w</mi></mrow></msub> <mo>=</mo> <msub><mtext>argmax</mtext> <mi>θ</mi></msub> <msub><mi>L</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>θ</mi> <mo>)</mo></mrow></mrow></math>

尽管我们不会完全展示这个目标背后的推导，因为它在数学上相当复杂，超出了本文的范围，但我们提供一些直觉。让我们首先检查这个术语：<math alttext="sigma-summation Underscript a Endscripts pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis upper A Subscript theta Sub Subscript o l d Baseline left-parenthesis s comma a right-parenthesis"><mrow><msub><mo>∑</mo> <mi>a</mi></msub> <msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow> <msub><mi>A</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow></mrow></math>，假设一个固定的状态 *s*。为了论证，让我们用我们建议策略的参数 <math alttext="theta Subscript o l d"><msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></math> 替换 <math alttext="theta"><mi>θ</mi></math>，它也代表我们当前策略的参数：

<math alttext="sigma-summation Underscript a Endscripts pi Subscript theta Sub Subscript o l d Baseline left-parenthesis a vertical-bar s right-parenthesis upper A Subscript pi Sub Subscript theta Sub Sub Subscript o l d Baseline left-parenthesis s comma a right-parenthesis equals double-struck upper E Subscript a tilde pi Sub Subscript theta Sub Sub Subscript o l d Subscript left-parenthesis a vertical-bar s right-parenthesis Baseline left-bracket upper A Subscript pi Sub Subscript theta Sub Sub Subscript o l d Sub Subscript Subscript Baseline left-parenthesis s comma a right-parenthesis right-bracket"><mrow><msub><mo>∑</mo> <mi>a</mi></msub> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow> <msub><mi>A</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>a</mi><mo>∼</mo><msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow></msub> <mrow><mo>[</mo> <msub><mi>A</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

<math alttext="equals double-struck upper E Subscript a tilde pi Sub Subscript theta Sub Sub Subscript o l d Subscript left-parenthesis a vertical-bar s right-parenthesis Baseline left-bracket upper Q Subscript pi Sub Subscript theta Sub Sub Subscript o l d Sub Subscript Subscript Baseline left-parenthesis s comma a right-parenthesis right-bracket minus double-struck upper E Subscript a tilde pi Sub Subscript theta Sub Sub Subscript o l d Subscript left-parenthesis a vertical-bar s right-parenthesis Baseline left-bracket upper V Subscript pi Sub Subscript theta Sub Sub Subscript o l d Sub Subscript Subscript Baseline left-parenthesis s right-parenthesis right-bracket"><mrow><mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>a</mi><mo>∼</mo><msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow></msub> <mrow><mo>[</mo> <msub><mi>Q</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>-</mo> <msub><mi>𝔼</mi> <mrow><mi>a</mi><mo>∼</mo><msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow></msub> <mrow><mo>[</mo> <msub><mi>V</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

<math alttext="equals double-struck upper E Subscript a tilde pi Sub Subscript theta Sub Sub Subscript o l d Subscript left-parenthesis a vertical-bar s right-parenthesis Baseline left-bracket upper Q Subscript pi Sub Subscript theta Sub Sub Subscript o l d Sub Subscript Subscript Baseline left-parenthesis s comma a right-parenthesis right-bracket minus upper V Subscript pi Sub Subscript theta Sub Sub Subscript o l d Baseline left-parenthesis s right-parenthesis"><mrow><mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>a</mi><mo>∼</mo><msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow></msub> <mrow><mo>[</mo> <msub><mi>Q</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>-</mo> <msub><mi>V</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="equals upper V Subscript pi Sub Subscript theta Sub Sub Subscript o l d Baseline left-parenthesis s right-parenthesis minus upper V Subscript pi Sub Subscript theta Sub Sub Subscript o l d Baseline left-parenthesis s right-parenthesis"><mrow><mo>=</mo> <msub><mi>V</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow> <mo>-</mo> <msub><mi>V</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="equals 0"><mrow><mo>=</mo> <mn>0</mn></mrow></math>

我们在这里展示了什么？早些时候，我们谈到了 <math alttext="upper A Subscript pi Sub Subscript theta Sub Sub Subscript o l d Baseline left-parenthesis s comma a right-parenthesis"><mrow><msub><mi>A</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow></mrow></math> 定义了在当前策略下，采取动作 *a* 在状态 *s* 中相对于我们期望从状态 *s* 开始看到的情况是更好还是更差。在这里，我们展示了如果我们按照当前策略的分布加权平均每个这些优势，我们将得到相对于当前策略的零平均优势——这在直觉上是很有意义的，因为建议策略和当前策略是完全相同的。我们不希望通过用当前策略替换自身来看到任何性能提升。

现在，如果我们用不同的建议策略参数 <math alttext="theta Subscript a l t"><msub><mi>θ</mi> <mrow><mi>a</mi><mi>l</mi><mi>t</mi></mrow></msub></math> 替换 <math alttext="theta"><mi>θ</mi></math>，上述推导将导致：

<math alttext="double-struck upper E Subscript a tilde pi Sub Subscript theta Sub Sub Subscript a l t Subscript left-parenthesis a vertical-bar s right-parenthesis Baseline left-bracket upper Q Subscript pi Sub Subscript theta Sub Sub Subscript o l d Sub Subscript Subscript Baseline left-parenthesis s comma a right-parenthesis right-bracket minus upper V Subscript pi Sub Subscript theta Sub Sub Subscript o l d Baseline left-parenthesis s right-parenthesis"><mrow><msub><mi>𝔼</mi> <mrow><mi>a</mi><mo>∼</mo><msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>a</mi><mi>l</mi><mi>t</mi></mrow></msub></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow></msub> <mrow><mo>[</mo> <msub><mi>Q</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>]</mo></mrow> <mo>-</mo> <msub><mi>V</mi> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>

这是简化的极限，因为第一项中的操作不再按照当前策略分布，我们无法进行导致倒数第二步的简化。如果我们评估这个表达式并得到一个正结果，我们可以将结果解释为相对于遵循当前策略而言，遵循建议策略会带来正的平均优势，直接转化为通过用建议策略替换当前策略来提高这个特定状态 *s* 的性能。

###### 注意

请注意，我们只考虑了一个特定状态 *s*。但即使我们看到某个状态的性能提升，也可能是这个状态很少出现的情况。这导致我们包含了术语 <math alttext="sigma-summation Underscript s Endscripts rho Subscript theta Sub Subscript o l d Baseline left-parenthesis s right-parenthesis"><mrow><msub><mo>∑</mo> <mi>s</mi></msub> <msub><mi>ρ</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>，它量化了我们看到给定状态的频率。实际上，我们可以将这个重写为期望，尽管这是一个未归一化的分布——我们只需要因子出归一化常数，这也是从 <math alttext="theta"><mi>θ</mi></math> 的角度来看的一个常数，因为归一化常数仅仅是 <math alttext="theta Subscript o l d"><msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></math> 的函数。

请记住<math alttext="sigma-summation Underscript s Endscripts rho Subscript theta Sub Subscript o l d Baseline left-parenthesis s right-parenthesis"><mrow><msub><mo>∑</mo> <mi>s</mi></msub> <msub><mi>ρ</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>是使用当前策略而不是提议策略进行评估的；这是因为，正如论文中所指出的，当优化替代目标（使用<math alttext="sigma-summation Underscript s Endscripts rho Subscript theta Baseline left-parenthesis s right-parenthesis"><mrow><msub><mo>∑</mo> <mi>s</mi></msub> <msub><mi>ρ</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>）时，对于<math alttext="theta"><mi>θ</mi></math>的复杂依赖性使得优化过程困难。此外，论文证明了一阶梯度与替代目标的梯度匹配，允许我们进行这种替换而不引入有偏的梯度估计。然而，我们不会在这里展示这一点，因为这超出了文本的范围。

将所有内容放在一起，我们有以下受限制的优化目标：

<math alttext="theta Superscript asterisk Baseline equals argmax Subscript theta Baseline sigma-summation Underscript s Endscripts rho Subscript theta Sub Subscript o l d Baseline left-parenthesis s right-parenthesis sigma-summation Underscript a Endscripts pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis upper A Subscript theta Sub Subscript o l d Baseline left-parenthesis s comma a right-parenthesis"><mrow><msup><mi>θ</mi> <mo>*</mo></msup> <mo>=</mo> <msub><mtext>argmax</mtext> <mi>θ</mi></msub> <msub><mo>∑</mo> <mi>s</mi></msub> <msub><mi>ρ</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow> <msub><mo>∑</mo> <mi>a</mi></msub> <msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow> <msub><mi>A</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="s period t period Avg period KL left-parenthesis theta Subscript o l d Baseline comma theta right-parenthesis less-than-or-equal-to delta"><mrow><mi>s</mi> <mo>.</mo> <mi>t</mi> <mo>.</mo> <mtext>Avg.</mtext> <mtext>KL</mtext> <mo>(</mo> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub> <mo>,</mo> <mi>θ</mi> <mo>)</mo> <mo>≤</mo> <mi>δ</mi></mrow></math>

其中平均KL散度表示所有状态下策略之间的期望KL散度。这就是我们所谓的*信任区域*，它代表了与当前参数设置足够接近的参数设置，有助于减轻训练不稳定性和减轻过拟合。我们如何优化这个目标？内部求和看起来像是关于<math alttext="pi Subscript theta Baseline left-parenthesis a comma s right-parenthesis"><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>,</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>的期望，但我们只有当前参数值的设置<math alttext="theta Subscript o l d"><msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></math>。在标准设置或*on-policy*设置中，我们从我们正在优化的相同策略中进行抽样，因此我们可以使用经典的策略梯度优化。然而，TRPO也可以修改为在*off-policy*设置中工作，其中我们从不同于我们正在优化的策略中进行抽样。一般来说，这种区别的原因是我们可能有一个行为策略，即我们从中进行抽样的策略可能更具探索性质，而我们学习的目标策略则是要优化的。在off-policy设置中，由于我们从不同分布*q(a|s)*（行为策略）而不是从<math alttext="pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis"><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow></mrow></math>（目标策略）中进行抽样，因此我们使用以下受限制的优化目标：

<math alttext="theta Superscript asterisk Baseline equals argmax Subscript theta Baseline sigma-summation Underscript s Endscripts rho Subscript theta Sub Subscript o l d Baseline left-parenthesis s right-parenthesis sigma-summation Underscript a Endscripts StartFraction pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis Over q left-parenthesis a vertical-bar s right-parenthesis EndFraction upper A Subscript theta Sub Subscript o l d Baseline left-parenthesis s comma a right-parenthesis"><mrow><msup><mi>θ</mi> <mo>*</mo></msup> <mo>=</mo> <msub><mtext>argmax</mtext> <mi>θ</mi></msub> <msub><mo>∑</mo> <mi>s</mi></msub> <msub><mi>ρ</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>)</mo></mrow> <msub><mo>∑</mo> <mi>a</mi></msub> <mfrac><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow> <mrow><mi>q</mi><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mfrac> <msub><mi>A</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="s period t period Avg period KL left-parenthesis theta Subscript o l d Baseline comma theta right-parenthesis less-than-or-equal-to delta"><mrow><mi>s</mi> <mo>.</mo> <mi>t</mi> <mo>.</mo> <mtext>Avg.</mtext> <mtext>KL</mtext> <mo>(</mo> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub> <mo>,</mo> <mi>θ</mi> <mo>)</mo> <mo>≤</mo> <mi>δ</mi></mrow></math>

添加*q(a|s)*考虑到我们从一个单独的行为策略中进行抽样的事实。我们可以更具体地从期望的角度来思考：

<math alttext="sigma-summation Underscript a Endscripts pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis upper A Subscript theta Sub Subscript o l d Baseline left-parenthesis s comma a right-parenthesis equals sigma-summation Underscript a Endscripts StartFraction q left-parenthesis a vertical-bar s right-parenthesis Over q left-parenthesis a vertical-bar s right-parenthesis EndFraction pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis upper A Subscript theta Sub Subscript o l d Baseline left-parenthesis s comma a right-parenthesis"><mrow><msub><mo>∑</mo> <mi>a</mi></msub> <msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow> <msub><mi>A</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>=</mo> <msub><mo>∑</mo> <mi>a</mi></msub> <mfrac><mrow><mi>q</mi><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow> <mrow><mi>q</mi><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mfrac> <msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow> <msub><mi>A</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="equals sigma-summation Underscript a Endscripts q left-parenthesis a vertical-bar s right-parenthesis StartFraction pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis Over q left-parenthesis a vertical-bar s right-parenthesis EndFraction upper A Subscript theta Sub Subscript o l d Baseline left-parenthesis s comma a right-parenthesis"><mrow><mo>=</mo> <msub><mo>∑</mo> <mi>a</mi></msub> <mi>q</mi> <mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow> <mfrac><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow> <mrow><mi>q</mi><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mfrac> <msub><mi>A</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow></mrow></math>

<math alttext="equals double-struck upper E Subscript q left-parenthesis a vertical-bar s right-parenthesis Baseline left-bracket StartFraction pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis Over q left-parenthesis a vertical-bar s right-parenthesis EndFraction upper A Subscript theta Sub Subscript o l d Subscript Baseline left-parenthesis s comma a right-parenthesis right-bracket"><mrow><mo>=</mo> <msub><mi>𝔼</mi> <mrow><mi>q</mi><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></msub> <mrow><mo>[</mo> <mfrac><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow> <mrow><mi>q</mi><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mfrac> <msub><mi>A</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

请注意，第一个等式的左侧可以写成相对于目标策略的优势的期望。通过一些代数操作，我们能够将原始目标转换为等效目标，但期望是相对于行为策略进行的。这是理想的，因为我们从行为策略中抽样，因此可以使用标准的小批量梯度下降技术来优化这个目标（添加KL散度的约束使得这比普通梯度下降复杂一些）。最后，我们已经看到了从未标准化的概率分布rho Subscript old Baseline（s）中抽样的方法，这些方法存在于学术文献中。

# 近端策略优化

TRPO的一个问题是，由于包含平均KL散度项，其优化相对复杂，并涉及二阶优化。近端策略优化，简称为PPO，是一种试图保留TRPO优点而避免复杂优化的算法。PPO提出了以下目标：

<math alttext="upper J left-parenthesis theta right-parenthesis equals double-struck upper E left-bracket min left-parenthesis StartFraction pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis Over pi Subscript theta Sub Subscript o l d Subscript Baseline left-parenthesis a vertical-bar s right-parenthesis EndFraction upper A Subscript theta Sub Subscript o l d Subscript Baseline left-parenthesis s comma a right-parenthesis comma clip left-parenthesis StartFraction pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis Over pi Subscript theta Sub Subscript o l d Subscript Baseline left-parenthesis a vertical-bar s right-parenthesis EndFraction comma 1 minus epsilon comma 1 plus epsilon right-parenthesis upper A Subscript theta Sub Subscript o l d Subscript Baseline left-parenthesis s comma a right-parenthesis right-parenthesis right-bracket"><mrow><mi>J</mi> <mrow><mo>(</mo> <mi>θ</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>𝔼</mi> <mo>[</mo> <mtext>min</mtext> <mrow><mo>(</mo> <mfrac><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow> <mrow><msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow></mfrac> <msub><mi>A</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>,</mo> <mtext>clip</mtext> <mrow><mo>(</mo> <mfrac><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow> <mrow><msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow></mfrac> <mo>,</mo> <mn>1</mn> <mo>-</mo> <mi>ϵ</mi> <mo>,</mo> <mn>1</mn> <mo>+</mo> <mi>ϵ</mi> <mo>)</mo></mrow> <msub><mi>A</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <mo>]</mo></mrow></math>

<math alttext="theta Superscript asterisk Baseline equals argmax Subscript theta Baseline upper J left-parenthesis theta right-parenthesis"><mrow><msup><mi>θ</mi> <mo>*</mo></msup> <mo>=</mo> <msub><mtext>argmax</mtext> <mi>θ</mi></msub> <mi>J</mi> <mrow><mo>(</mo> <mi>θ</mi> <mo>)</mo></mrow></mrow></math>

请注意，我们不再有复杂的约束，而是在优化目标中加入了额外的项。剪切函数表示目标策略和行为策略之间比率的上限和下限，任何超出这些限制的比率都将设置为相应的限制。注意原始值和剪切值之间的最小值的包含，这可以防止我们进行极端更新，并防止过拟合。

正如PPO介绍论文中所述，重要的是注意TRPO和PPO在θ等于θ Subscript old时具有相同的梯度。至少在在线策略设置中是这样，在这种设置中，我们从中抽样和优化的是单个策略（即行为策略和目标策略之间没有区别）。让我们更仔细地看看为什么会出现这种情况。为此，我们首先需要将TRPO的约束优化目标重新表述为等效的正则化优化目标（回想一下前一节早期的内容），根据理论，我们可以这样做。目标看起来像：

<math alttext="upper J Superscript upper T upper R upper P upper O Baseline left-parenthesis theta right-parenthesis equals double-struck upper E left-bracket StartFraction pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis Over pi Subscript theta Sub Subscript o l d Subscript Baseline left-parenthesis a vertical-bar s right-parenthesis EndFraction upper A left-parenthesis s comma a right-parenthesis minus beta asterisk KL left-parenthesis pi Subscript theta Sub Subscript o l d Subscript Baseline left-parenthesis a vertical-bar s right-parenthesis StartAbsoluteValue EndAbsoluteValue pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis right-parenthesis right-bracket"><mrow><msup><mi>J</mi> <mrow><mi>T</mi><mi>R</mi><mi>P</mi><mi>O</mi></mrow></msup> <mrow><mrow><mo>(</mo> <mi>θ</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>𝔼</mi> <mo>[</mo></mrow> <mfrac><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow> <mrow><msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow></mfrac> <mrow><mi>A</mi> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>-</mo> <mi>β</mi> <mo>*</mo> <mtext>KL</mtext> <mo>(</mo></mrow> <msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow> <mo>|</mo> <mo>|</mo></mrow> <msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo> <mi>a</mi> <mo>|</mo> <mi>s</mi> <mo>)</mo></mrow> <mrow><mo>)</mo> <mo>]</mo></mrow></mrow></math>

注意，由于期望的线性性，我们可以将期望中的表达式分开为期望的差异。如果我们首先考虑第二个期望，或者KL项，我们会注意到这个项在θ等于θ Subscript old时被最小化，因为参考分布是使用θ Subscript old参数化的。因此，在这种设置下，梯度为零，因为我们已经达到全局最小值。我们只剩下第一个期望的梯度：

<math alttext="normal nabla Subscript theta Baseline double-struck upper E left-bracket StartFraction pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis Over pi Subscript theta Sub Subscript o l d Subscript Baseline left-parenthesis a vertical-bar s right-parenthesis EndFraction upper A left-parenthesis s comma a right-parenthesis right-bracket"><mrow><msub><mi>∇</mi> <mi>θ</mi></msub> <mi>𝔼</mi> <mrow><mo>[</mo> <mfrac><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow> <mrow><msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow></mfrac> <mi>A</mi> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

针对PPO的目标，我们注意到在θ等于θ Subscript old时，两个策略之间的比率为1，消除了剪切项的需要。因此，我们只剩下两个等价项的最小值，简化为单个项的期望。梯度正好与TRPO目标中看到的相同：

<math alttext="normal nabla Subscript theta Baseline double-struck upper E left-bracket StartFraction pi Subscript theta Baseline left-parenthesis a vertical-bar s right-parenthesis Over pi Subscript theta Sub Subscript o l d Subscript Baseline left-parenthesis a vertical-bar s right-parenthesis EndFraction upper A left-parenthesis s comma a right-parenthesis right-bracket"><mrow><msub><mi>∇</mi> <mi>θ</mi></msub> <mi>𝔼</mi> <mrow><mo>[</mo> <mfrac><mrow><msub><mi>π</mi> <mi>θ</mi></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow> <mrow><msub><mi>π</mi> <msub><mi>θ</mi> <mrow><mi>o</mi><mi>l</mi><mi>d</mi></mrow></msub></msub> <mrow><mo>(</mo><mi>a</mi><mo>|</mo><mi>s</mi><mo>)</mo></mrow></mrow></mfrac> <mi>A</mi> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

我们已经证明了在选择在线策略设置中，PPO与TRPO具有相同的梯度，并且在实践中更容易优化。PPO在各种任务上也表现出强大的经验结果，并且已经广泛应用于深度强化学习领域。

# Q学习和深度Q网络

Q学习属于强化学习中的值学习类别。我们不是直接学习一个策略，而是学习状态和动作的值。Q学习涉及学习一个函数，一个*Q函数*，它表示状态、动作对的质量。定义为*Q(s, a)*的Q函数是一个计算在状态*s*中执行动作*a*时的最大折扣未来回报的函数。

Q值代表我们在一个状态下采取一个动作，然后完美地采取每一个后续动作（以最大化预期未来奖励）时的预期长期奖励。这可以正式表达为：

<math alttext="upper Q Superscript asterisk Baseline left-parenthesis s Subscript t Baseline comma a Subscript t Baseline right-parenthesis equals m a x Subscript pi Baseline upper E left-bracket sigma-summation Underscript i equals t Overscript upper T Endscripts gamma Superscript i Baseline r Superscript i Baseline right-bracket"><mrow><msup><mi>Q</mi> <mo>*</mo></msup> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>)</mo></mrow> <mo>=</mo> <mi>m</mi> <mi>a</mi> <msub><mi>x</mi> <mi>π</mi></msub> <mi>E</mi> <mrow><mo>[</mo> <msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mi>t</mi></mrow> <mi>T</mi></msubsup> <mrow><msup><mi>γ</mi> <mi>i</mi></msup> <msup><mi>r</mi> <mi>i</mi></msup></mrow> <mo>]</mo></mrow></mrow></math>

你可能会问，我们怎么知道Q值呢？即使对于人类来说，知道一个动作有多好也是困难的，因为你需要知道未来你将如何行动。我们的预期未来回报取决于我们的长期策略将是什么。这似乎是一个鸡生蛋的问题。为了评估一个状态、动作对，你需要知道所有完美的后续动作。为了知道最佳动作，你需要准确的状态和动作值。

## 贝尔曼方程

我们通过将我们的Q值定义为未来Q值的函数来解决这个困境。这种关系被称为*贝尔曼方程*，它表明采取行动的最大未来奖励是当前奖励加上下一步采取下一个动作*a'*的*最大*未来奖励：

<math alttext="upper Q Superscript asterisk Baseline left-parenthesis s Subscript t Baseline comma a Subscript t Baseline right-parenthesis equals upper E left-bracket r Subscript t Baseline plus gamma max Underscript a prime Endscripts upper Q Superscript asterisk Baseline left-parenthesis s Subscript t plus 1 Baseline comma a Superscript prime Baseline right-parenthesis right-bracket"><mrow><msup><mi>Q</mi> <mo>*</mo></msup> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>)</mo></mrow> <mo>=</mo> <mi>E</mi> <mrow><mo>[</mo> <msub><mi>r</mi> <mi>t</mi></msub> <mo>+</mo> <mi>γ</mi> <msub><mo movablelimits="true" form="prefix">max</mo> <msup><mi>a</mi> <mo>'</mo></msup></msub> <msup><mi>Q</mi> <mo>*</mo></msup> <mrow><mo>(</mo> <msub><mi>s</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>,</mo> <msup><mi>a</mi> <msup><mo>'</mo></msup></msup> <mo>)</mo></mrow> <mo>]</mo></mrow></mrow></math>

这个递归定义使我们能够关联Q值之间的关系。

现在我们可以关联过去和未来的Q值，这个方程方便地定义了一个更新规则。也就是说，我们可以根据未来的Q值更新过去的Q值。这是强大的，因为存在一个我们知道是正确的Q值：在剧集结束之前的最后一个动作的Q值。对于这个最后的状态，我们确切地知道下一个动作导致了下一个奖励，所以我们可以完美地设置该状态的Q值。然后，我们可以使用更新规则将该Q值传播到之前的时间步：

<math alttext="ModifyingAbove upper Q Subscript j Baseline With caret right-arrow ModifyingAbove upper Q Subscript j plus 1 Baseline With caret right-arrow ModifyingAbove upper Q Subscript j plus 2 Baseline With caret right-arrow ellipsis right-arrow upper Q Superscript asterisk"><mrow><mover accent="true"><msub><mi>Q</mi> <mi>j</mi></msub> <mo>^</mo></mover> <mo>→</mo> <mover accent="true"><msub><mi>Q</mi> <mrow><mi>j</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>^</mo></mover> <mo>→</mo> <mover accent="true"><msub><mi>Q</mi> <mrow><mi>j</mi><mo>+</mo><mn>2</mn></mrow></msub> <mo>^</mo></mover> <mo>→</mo> <mo>...</mo> <mo>→</mo> <msup><mi>Q</mi> <mo>*</mo></msup></mrow></math>

这种Q值的更新被称为*值迭代*。

我们的第一个Q值完全错误，但这是完全可以接受的。每次迭代，我们可以通过未来的正确Q值更新我们的Q值。经过一次迭代，最后一个Q值是准确的，因为它只是在剧集终止之前的最后一个状态和动作的奖励。然后我们执行我们的Q值更新，设置倒数第二个Q值。在下一次迭代中，我们可以保证最后两个Q值是正确的，依此类推。通过值迭代，我们将保证收敛到最终最优的Q值。

## 值迭代的问题

值迭代产生了状态和动作对之间对应的Q值的映射，我们正在构建这些映射的表，或者*Q表*。让我们简要谈谈这个Q表的大小。值迭代是一个耗尽的过程，需要完全遍历整个状态、动作对空间。在像打砖块这样的游戏中，有100块砖可能存在或不存在，球拍有50个位置可在，球有250个位置可在，还有3个动作，我们已经构建了一个远远大于人类所有计算能力之和的空间。此外，在随机环境中，我们的Q表空间会更大，可能是无限的。在这样一个巨大的空间中，我们将无法找到每个状态、动作对的所有Q值。显然，这种方法行不通。我们还能怎么做Q学习呢？

## 逼近Q函数

我们的Q表的大小使得朴素方法在任何非玩具问题上都难以处理。然而，如果我们放宽对最优Q函数的要求呢？如果我们学习Q函数的近似值，我们可以使用一个模型来估计我们的Q函数。我们不必体验每个状态、动作对来更新我们的Q表，我们可以学习一个近似这个表的函数，甚至可以在自己的经验之外进行泛化。这意味着我们不必对所有可能的Q值进行详尽搜索来学习Q函数。

## 深度Q网络

这是DeepMind在深度Q网络（DQN）上工作的主要动机。DQN使用一个深度神经网络，将一个图像（状态）输入，估计所有可能动作的Q值。

## 训练DQN

我们希望训练我们的网络来逼近Q函数。我们将这个Q函数逼近表示为我们模型参数的函数，如下所示：

<math alttext="ModifyingAbove upper Q Subscript theta Baseline With caret left-parenthesis s comma a bar theta right-parenthesis tilde upper Q Superscript asterisk Baseline left-parenthesis s comma a right-parenthesis"><mrow><mover accent="true"><msub><mi>Q</mi> <mi>θ</mi></msub> <mo>^</mo></mover> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>∣</mo> <mi>θ</mi> <mo>)</mo></mrow> <mo>∼</mo> <msup><mi>Q</mi> <mo>*</mo></msup> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <mi>a</mi> <mo>)</mo></mrow></mrow></math>

记住，Q学习是一种值学习算法。我们不是直接学习策略，而是学习每个状态、动作对的值，无论它们是好是坏。我们已经将我们模型的Q函数逼近表示为Qtheta，我们希望这个值接近未来的预期奖励。使用之前的贝尔曼方程，我们可以将这个未来的预期奖励表示为：

<math><mrow><msubsup><mi>R</mi> <mi>t</mi> <mo>*</mo></msubsup> <mo>=</mo> <mfenced close=")" open="(" separators=""><msub><mi>r</mi> <mi>t</mi></msub> <mo>+</mo> <mi>γ</mi> <msub><mo form="prefix" movablelimits="true">max</mo> <msup><mi>a</mi> <mo>'</mo></msup></msub> <mover accent="true"><mi>Q</mi> <mo>^</mo></mover> <mrow><mo>(</mo><msub><mi>s</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>,</mo><msup><mi>a</mi> <mo>'</mo></msup> <mo>|</mo><mi>θ</mi><mo>)</mo></mrow></mfenced></mrow></math>

我们的目标是最小化我们的Q的逼近值与下一个Q值之间的差异：

<math alttext="min Underscript theta Endscripts sigma-summation Underscript e element-of upper E Endscripts sigma-summation Underscript t equals 0 Overscript upper T Endscripts ModifyingAbove upper Q With caret left-parenthesis s Subscript t Baseline comma a Subscript t Baseline vertical-bar theta right-parenthesis minus upper R Subscript t Superscript asterisk"><mrow><msub><mo movablelimits="true" form="prefix">min</mo> <mi>θ</mi></msub> <msub><mo>∑</mo> <mrow><mi>e</mi><mo>∈</mo><mi>E</mi></mrow></msub> <msubsup><mo>∑</mo> <mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow> <mi>T</mi></msubsup> <mover accent="true"><mi>Q</mi> <mo>^</mo></mover> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>|</mo> <mi>θ</mi> <mo>)</mo></mrow> <mo>-</mo> <msubsup><mi>R</mi> <mi>t</mi> <mo>*</mo></msubsup></mrow></math>

扩展这个表达式给我们完整的目标：

<math><mrow><msub><mo form="prefix" movablelimits="true">min</mo> <mi>θ</mi></msub> <msub><mo>∑</mo> <mrow><mi>e</mi><mo>∈</mo><mi>E</mi></mrow></msub> <msubsup><mo>∑</mo> <mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow> <mi>T</mi></msubsup> <mover accent="true"><mi>Q</mi> <mo>^</mo></mover> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>|</mo> <mi>θ</mi> <mo>)</mo></mrow> <mo>-</mo> <mfenced close=")" open="(" separators=""><msub><mi>r</mi> <mi>t</mi></msub> <mo>+</mo> <mi>γ</mi> <msub><mo form="prefix" movablelimits="true">max</mo> <msup><mi>a</mi> <mo>'</mo></msup></msub> <mover accent="true"><mi>Q</mi> <mo>^</mo></mover> <mrow><mo>(</mo><msub><mi>s</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>,</mo><msup><mi>a</mi> <mo>'</mo></msup> <mo>|</mo><mi>θ</mi><mo>)</mo></mrow></mfenced></mrow></math>

这个目标完全可微，作为我们模型参数的函数，我们可以找到梯度用于随机梯度下降来最小化这个损失。

## 学习稳定性

你可能已经注意到一个问题，那就是我们基于模型预测的这一步的Q值和下一步的预测Q值之间的差异来定义我们的损失函数。这样，我们的损失双重依赖于我们的模型参数。随着每次参数更新，Q值不断变化，我们使用变化的Q值进行进一步更新。这种更新的高相关性可能导致反馈循环和学习不稳定，其中我们的参数可能会振荡并使损失发散。

我们可以采用一些简单的工程技巧来解决这个相关性问题：即目标Q网络和经验重放。

## 目标Q网络

我们可以通过引入第二个网络，称为“目标网络”，减少频繁更新单个网络与自身的相互依赖。我们的损失函数包含Q函数的两个实例，<math alttext="ModifyingAbove upper Q With caret left-parenthesis s Subscript t Baseline comma a Subscript t Baseline vertical-bar theta right-parenthesis"><mrow><mover accent="true"><mi>Q</mi> <mo>^</mo></mover> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>|</mo> <mi>θ</mi> <mo>)</mo></mrow></mrow></math> 和 <math><mrow><mover accent="true"><mi>Q</mi> <mo>^</mo></mover> <mrow><mo>(</mo> <msub><mi>s</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>,</mo> <msup><mi>a</mi> <mo>'</mo></msup> <mo>|</mo> <mi>θ</mi> <mo>)</mo></mrow></mrow></math> 。我们将第一个Q表示为我们的预测网络，第二个Q将由目标Q网络生成。目标Q网络是我们预测网络的一个副本，它在参数更新方面滞后。我们仅在几个批次之后将目标Q网络更新为等于预测网络。这为我们的Q值提供了非常需要的稳定性，现在我们可以正确地学习一个好的Q函数。

## 经验重放

我们的学习中还有另一个让人不安的不稳定因素：最近经验的高相关性。如果我们用最近的经验来训练我们的DQN，这些动作、状态对都会彼此相关。这是有害的，因为我们希望我们的批量梯度能够代表整个梯度，如果我们的数据不代表数据分布，我们的批量梯度就不会准确估计真实梯度。

因此，我们必须打破我们批量数据的相关性。我们可以使用一种称为“经验重放”的方法来做到这一点。在经验重放中，我们将所有代理的经验存储为一个表，为了构建一个批次，我们从这些经验中随机抽样。我们将这些经验存储在一个表中，形式为<math alttext="left-parenthesis s Subscript i Baseline comma a Subscript i Baseline comma r Subscript i Baseline comma s Subscript i plus 1 Baseline right-parenthesis"><mrow><mo>(</mo> <msub><mi>s</mi> <mi>i</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>i</mi></msub> <mo>,</mo> <msub><mi>r</mi> <mi>i</mi></msub> <mo>,</mo> <msub><mi>s</mi> <mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>)</mo></mrow></math> 四元组。通过这四个值，我们可以计算我们的损失函数，从而优化我们的网络。

这个经验重放表更像是一个队列而不是一个表。代理在训练早期看到的经验可能不代表训练后代理发现自己处于的经验，因此有必要从我们的表中删除旧的经验。

## 从Q函数到策略

Q学习是一种价值学习范式，而不是一种策略学习算法。这意味着我们不是直接学习在环境中行动的策略。但是我们不能根据我们的Q函数告诉我们的内容构建一个策略吗？如果我们学习了一个好的Q函数近似，这意味着我们知道每个状态的每个动作的价值。然后我们可以轻松地按照以下方式构建一个最优策略：查看我们当前状态中所有动作的Q函数，选择具有最大Q值的动作，进入新状态，然后重复。如果我们的Q函数是最优的，那么从中派生出来的策略也将是最优的。考虑到这一点，我们可以将最优策略表达如下：

<math><mrow><mi>π</mi> <mrow><mo>(</mo> <mi>s</mi> <mo>;</mo> <mi>θ</mi> <mo>)</mo></mrow> <mo>=</mo> <mo form="prefix">arg</mo> <msub><mo form="prefix" movablelimits="true">max</mo> <msup><mi>a</mi> <mo>'</mo></msup></msub> <mover accent="true"><msup><mi>Q</mi> <mo>*</mo></msup> <mo>^</mo></mover> <mrow><mo>(</mo> <mi>s</mi> <mo>,</mo> <msup><mi>a</mi> <mo>'</mo></msup> <mo>;</mo> <mi>θ</mi> <mo>)</mo></mrow></mrow></math>

我们还可以使用之前讨论过的采样技术来制定一个随机策略，有时会偏离Q函数的建议，以改变我们的代理程序进行探索的程度。

## DQN和马尔可夫假设

DQN仍然是一个依赖于*马尔可夫假设*的马尔可夫决策过程，该假设假定下一个状态<math alttext="s Subscript i plus 1"><msub><mi>s</mi> <mrow><mi>i</mi><mo>+</mo><mn>1</mn></mrow></msub></math>仅取决于当前状态<math alttext="s Subscript i"><msub><mi>s</mi> <mi>i</mi></msub></math>和动作<math alttext="a Subscript i"><msub><mi>a</mi> <mi>i</mi></msub></math>，而不取决于任何先前的状态或动作。这个假设对于许多环境并不成立，其中游戏状态无法在单个帧中总结。例如，在乒乓球中，球的速度（成功游戏的重要因素）在任何单个游戏帧中都没有被捕捉到。马尔可夫假设使得建模决策过程变得更简单和可靠，但通常会损失建模能力。

## DQN对马尔可夫假设的解决方案

DQN通过利用*状态历史*来解决这个问题。DQN不是将一个游戏帧作为游戏状态，而是将过去四个游戏帧视为游戏的当前状态。这使得DQN能够利用时间相关信息。这有点工程上的技巧，我们将在本章末尾讨论处理状态序列的更好方法。

## 使用DQN玩Breakout

让我们将我们学到的所有内容整合在一起，实际上开始实施DQN来玩Breakout。我们首先定义我们的`DQNAgent`：

```py
# DQNAgent

class DQNAgent(object):

    def __init__(self, num_actions,
                 learning_rate=1e-3, history_length=4,
                 screen_height=84, screen_width=84,
                 gamma=0.99):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.history_length = history_length
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.gamma = gamma

        self.build_prediction_network()
        self.build_target_network()
        #self.build_training()

    def build_prediction_network(self):
        self.model_predict = nn.Sequential(
          nn.Conv2d(4, 32, kernel_size=8 , stride=4),
          nn.Conv2d(32, 64, kernel_size=4, stride=2),
          nn.Conv2d(64, 64, kernel_size=3, stride=1),
          nn.Flatten(),
          nn.Linear(3136, 512),
          nn.Linear(512, self.num_actions)
          )

    def build_target_network(self):
        self.model_target = nn.Sequential(
          nn.Conv2d(4, 32, kernel_size=8 , stride=4),
          nn.Conv2d(32, 64, kernel_size=4, stride=2),
          nn.Conv2d(64, 64, kernel_size=3, stride=1),
          nn.Flatten(),
          nn.Linear(3136, 512),
          nn.Linear(512, self.num_actions)
          )

    def sample_and_train_pred(self, replay_table, batch_size):

        s_t, action, reward, s_t_plus_1, terminal = \
                   replay_table.sample_batch(batch_size)

        # given state_t, find q_t (predict_model) and 
        #  q_t+1 (target_model)
        # do it in batches
        # Find q_t_plus_1
        input_t = torch.from_numpy(s_t_plus_1).float()
        model_t = self.model_target.float()
        q_t_plus_1 = model_t(input_t)

        terminal = torch.tensor(terminal).float()
        max_q_t_plus_1, _ = torch.max(q_t_plus_1, dim=1)
        reward = torch.from_numpy(reward).float()
        target_q_t = (1\. - terminal) * self.gamma * \
                     max_q_t_plus_1 + reward

        # Find q_t, and q_of_action
        input_p = torch.from_numpy(s_t).float()
        model_p = self.model_predict.float()
        q_t = model_p(input_p)
        action = torch.from_numpy(action)
        action_one_hot = nn.functional.one_hot(action,
                                               self.num_actions)
        q_of_action = torch.sum(q_t * action_one_hot)

        # Compute loss
        self.delta = (target_q_t - q_of_action)
        self.loss = torch.mean(self.delta)

        # Update predict_model gradients (only)
        self.optimizer = optim.Adam(self.model_predict.parameters(),
                                    lr = self.learning_rate)
        self.loss.backward()
        self.optimizer.step()

        return q_t

    def predict_action(self, state, epsilon_percentage):
        input_p = torch.from_numpy(state).float().unsqueeze(dim=0)
        model_p = self.model_predict.float()
        action_distribution = model_p(input_p)
        # sample from action distribution
        action = epsilon_greedy_action_annealed(
                                       action_distribution.detach(),
                                       epsilon_percentage)
        return action

    def process_state_into_stacked_frames(self,
                                          frame,
                                          past_frames,
                                          past_state=None):
        full_state = np.zeros((self.history_length,
                              self.screen_width,
                              self.screen_height))

        if past_state is not None:
            for i in range(len(past_state)-1):
                full_state[i, :, :] = past_state[i+1, :, :]
            full_state[-1, :, :] = self.preprocess_frame(frame,
                                                 (self.screen_width,
                                                  self.screen_height)
                                                 )
        else:
            all_frames = past_frames + [frame]
            for i, frame_f in enumerate(all_frames):
                full_state[i, :, :] = self.preprocess_frame(frame_f,
                                                 (self.screen_width,
                                                  self.screen_height)
                                                  )
        return full_state

    def to_grayscale(self, x):
        return np.dot(x[...,:3], [0.299, 0.587, 0.114])

    def preprocess_frame(self, im, shape):
        cropped = im[16:201,:] # (185, 160, 3)
        grayscaled = self.to_grayscale(cropped) # (185, 160)
        # resize to (84,84)
        resized = np.array(Image.fromarray(grayscaled).resize(shape))
        mean, std = 40.45, 64.15
        frame = (resized-mean)/std
        return frame

```

这个类中有很多内容，让我们在以下部分中逐一解释。

## 构建我们的架构

我们构建两个Q网络：预测网络和目标Q网络。请注意它们具有相同的架构定义，因为它们是相同的网络，只是目标Q具有延迟的参数更新。由于我们正在学习从纯像素输入中玩Breakout，我们的游戏状态是一个像素数组。我们将这个图像通过三个卷积层，然后两个全连接层，以产生我们每个潜在动作的Q值。

## 堆叠帧

您可能注意到我们的状态输入实际上是大小为`[None, self.history_length, self.screen_height, self.screen_width]`。记住，为了建模和捕捉像速度这样的时间相关状态变量，DQN不仅使用一个图像，而是一组连续的图像，也称为*历史*。这些连续的图像中的每一个被视为一个单独的通道。我们使用辅助函数`process_state_into_stacked_frames(self, frame, past_frames, past_state=None)`构建这些堆叠帧。

## 设置训练操作

我们的损失函数源自本章前面的目标表达式：

<math><mrow><msub><mo form="prefix" movablelimits="true">min</mo> <mi>θ</mi></msub> <msub><mo>∑</mo> <mrow><mi>e</mi><mo>∈</mo><mi>E</mi></mrow></msub> <msubsup><mo>∑</mo> <mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow> <mi>T</mi></msubsup> <mover accent="true"><mi>Q</mi> <mo>^</mo></mover> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>|</mo> <mi>θ</mi> <mo>)</mo></mrow> <mo>-</mo> <mfenced close=")" open="(" separators=""><msub><mi>r</mi> <mi>t</mi></msub> <mo>+</mo> <mi>γ</mi> <msub><mo form="prefix" movablelimits="true">max</mo> <msup><mi>a</mi> <mo>'</mo></msup></msub> <mover accent="true"><mi>Q</mi> <mo>^</mo></mover> <mrow><mo>(</mo><msub><mi>s</mi> <mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub> <mo>,</mo><msup><mi>a</mi> <mo>'</mo></msup> <mo>|</mo><mi>θ</mi><mo>)</mo></mrow></mfenced></mrow></math>

我们希望我们的预测网络等于我们的目标网络，再加上当前时间步的回报。我们可以用纯PyTorch代码表示这一点，即我们的预测网络的输出与目标网络的输出之间的差异。我们使用这个梯度来更新和训练我们的预测网络，使用`AdamOptimizer`。

## 更新我们的目标Q网络

为了确保稳定的学习环境，我们只在每四个批次中更新一次目标Q网络。我们的目标Q网络的更新规则非常简单：我们只需将其权重设置为预测网络的权重。我们在函数`update_target_q_network(self)`中执行这个操作。`optimizer_predict.step()`函数将目标Q网络的权重设置为预测网络的权重。

## 实施经验重放

我们已经讨论了经验重放如何帮助去相关我们的梯度批次更新，以提高我们的Q学习和随后派生的策略的质量。让我们简单实现一下经验重放。我们暴露一个方法`add_episode(self, episode)`，它接受整个剧集（一个`EpisodeHistory`对象）并将其添加到ExperienceReplayTable中。然后检查表是否已满，并从表中删除最旧的经验。

当需要从这个表中抽样时，我们可以调用`sample_batch(self, batch_size)`来随机构建一个批次从我们的经验表中：

```py
class ExperienceReplayTable(object):

    def __init__(self, table_size=50000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.terminals = []

        self.table_size = table_size

    def add_episode(self, episode):
        self.states += episode.states
        self.actions += episode.actions
        self.rewards += episode.rewards
        self.state_primes += episode.state_primes
        self.terminals += episode.terminals

        self.purge_old_experiences()

    def purge_old_experiences(self):
        while len(self.states) > self.table_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.state_primes.pop(0)

    def sample_batch(self, batch_size):
        s_t, action, reward, s_t_plus_1, terminal = [], [], [], 
                                                    [], []
        rands = np.arange(len(self.states))
        np.random.shuffle(rands)
        rands = rands[:batch_size]

        for r_i in rands:
            s_t.append(self.states[r_i])
            action.append(self.actions[r_i])
            reward.append(self.rewards[r_i])
            s_t_plus_1.append(self.state_primes[r_i])
            terminal.append(self.terminals[r_i])
        return (np.array(s_t), np.array(action), np.array(reward),
                np.array(s_t_plus_1), np.array(terminal))

```

## DQN主循环

让我们在我们的主函数中将所有这些放在一起，这将为Breakout创建一个OpenAI Gym环境，创建我们的`DQNAgent`的实例，并让我们的代理与成功玩Breakout进行交互和训练：

```py
learn_start = 4
total_episodes = 32
epsilon_stop = 32
train_frequency = 2
target_frequency = 4
batch_size = 4
max_episode_length = 1000
env = gym.make('Breakout-v4')
num_actions = env.action_space.n
solved = False

agent = DQNAgent(num_actions=num_actions,
                 learning_rate=1e-4,
                 history_length=4,
                 gamma=0.98)

episode_rewards = []
q_t_list = []
batch_losses = []
past_frames_last_time = None

replay_table = ExperienceReplayTable()
global_step_counter = 0

for i in range(total_episodes):
    # Get initial frame -> state
    frame = env.reset() # np.array of shape (210, 160, 3)
    # past_frames is a list of past 3 frames (np.arrays)
    past_frames = [copy.deepcopy(frame) for _ in range(
                                           agent.history_length-1)]
    state = agent.process_state_into_stacked_frames(
        frame, past_frames, past_state=None) # state is (4,84,84)

    # initialize episode history (s_t, a, r, s_t+1, terminal)
    episode_reward = 0.0
    episode_history = EpisodeHistory()
    epsilon_percentage = float(min(i/float(epsilon_stop), 1.0))

    for j in range(max_episode_length):
        # predict action or choose random action at first
        if global_step_counter < learn_start:
          action = np.argmax(np.random.random((agent.num_actions)))
        else:
          action = agent.predict_action(state, epsilon_percentage)

        # take action, get next frame (-> next state), reward, 
        # and terminal
        reward = 0
        frame_prime, reward, terminal, _ = env.step(action)
        if terminal == True:
          reward -= 1

        # get next state from next frame and past frames
        state_prime = agent.process_state_into_stacked_frames(
                                                   frame_prime,
                                                   past_frames,
                                                   past_state=state)
        # Update past_frames with frame_prime for next time
        past_frames.append(frame_prime)
        past_frames = past_frames[len(past_frames)- \
                            agent.history_length:]
        past_frames_last_time = past_frames

        # Add to episode history (state, action, reward, 
        #  state_prime, terminal)
        episode_history.add_to_history(
                    state, action, reward, state_prime, terminal)
        state = state_prime
        episode_reward += reward
        global_step_counter += 1

        #  Do not train predict_model until we have enough
        #   episodes in episode history
        if global_step_counter > learn_start:
          if global_step_counter % train_frequency == 0:
              if(len(replay_table.actions) != 0):
                q_t = agent.sample_and_train_pred(replay_table, 
                                                  batch_size)
                q_t_list.append(q_t)

                if global_step_counter % target_frequency == 0:
                    agent.model_target.load_state_dict(
                        agent.model_predict.state_dict())

        # If terminal or max episodes reached,
        #   add episode_history to replay table
        if j == (max_episode_length - 1):
            terminal = True

        if terminal:
            replay_table.add_episode(episode_history)
            episode_rewards.append(episode_reward)
            break
    print(f'Episode[{i}]: {len(episode_history.actions)} \
              actions {episode_reward} reward')

```

## Breakout上的DQNAgent结果

我们训练我们的`DQNAgent`一千个周期以查看学习曲线。要在Atari上获得超人类的结果，典型的训练时间长达数天。然而，我们可以很快看到奖励的总体上升趋势，如[图13-7](#fig0807)所示。

![](Images/fdl2_1307.png)

###### 图13-7。我们的`DQNAgent`在训练过程中在Breakout上变得越来越好，因为它学会了一个良好的值函数，并且由于<math alttext="epsilon"><mi>ϵ</mi></math>-贪心退火，行动变得更少随机

# 改进和超越DQN

DQN在2013年解决Atari任务方面做得相当不错，但也存在一些严重缺陷。DQN的许多弱点包括训练时间非常长，在某些类型的游戏上表现不佳，并且需要为每个新游戏重新训练。过去几年的深度强化学习研究大部分是在解决这些不同的弱点。

## 深度递归Q网络

还记得马尔可夫假设吗？它说下一个状态仅依赖于前一个状态和代理所采取的行动？DQN对马尔可夫假设问题的解决方案是将四个连续帧堆叠为单独的通道，从而避开了这个问题，有点像临时工程技巧。为什么是4帧而不是10帧？这个强加的帧历史超参数限制了模型的普适性。我们如何处理任意相关数据序列？没错：我们可以使用我们在第8章中学到的关于RNN的知识来模拟具有*深度递归Q网络*（DRQNs）的序列。

DRQN使用一个循环层将状态的潜在知识从一个时间步传输到下一个。通过这种方式，模型本身可以学习包含多少帧对其状态有信息量，并且甚至可以学会丢弃无信息的帧或者记住很久以前的事情。

DRQN甚至已经扩展到包括神经注意机制，正如Sorokin等人在2015年的论文“Deep Attention Recurrent Q-Network”（DAQRN）中所示。由于DRQN处理数据序列，它可以关注序列的某些部分。这种关注图像某些部分的能力既提高了性能，又通过为采取的行动提供理由来提供模型可解释性。

DRQN在玩第一人称射击游戏（FPS）时表现比DQN更好，比如[DOOM](https://oreil.ly/KKZC7)，同时也提高了在具有长时间依赖性的某些Atari游戏上的表现，比如[Seaquest](https://oreil.ly/uevTS)。

## 异步优势演员-评论家代理

*异步优势演员-评论家*（A3C）是一种新的深度强化学习方法，介绍于2016年DeepMind的论文“深度强化学习的异步方法”中。让我们讨论一下它是什么以及为什么它改进了DQN。

A3C是*异步*的，这意味着我们可以在许多线程中并行化我们的代理，从而通过加速环境模拟来实现数量级上的更快训练。A3C同时运行许多环境以收集经验。除了速度增加外，这种方法还具有另一个重要优势，即进一步使我们批次中的经验解耦，因为批次中填充了同时在不同场景中的众多代理的经验。

A3C使用*演员-评论家*方法。演员-评论家方法涉及学习价值函数<math alttext="V(s_t)"><mrow><mi>V</mi> <mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>)</mo></mrow></math>（评论家）以及策略<math alttext="π(s_t)"><mrow><mi>π</mi> <mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>)</mo></mrow></math>（演员）。在本章的早期，我们划分了强化学习的两种不同方法：价值学习和策略学习。A3C结合了每种方法的优势，使用评论家的价值函数来改进演员的策略。

A3C使用*优势*函数而不是纯折扣未来回报。在进行策略学习时，我们希望在代理选择导致不良奖励的行动时对其进行惩罚。A3C旨在实现相同的目标，但使用优势而不是奖励作为其标准。优势表示模型对所采取行动的质量的预测与实际所采取行动的质量之间的差异。我们可以将优势表示为：

<math alttext="上标t基线的A等于上标星号基线的Q左括号s基线，a基线右括号减去上标V左括号s基线右括号"><mrow><msub><mi>A</mi> <mi>t</mi></msub> <mo>=</mo> <msup><mi>Q</mi> <mo>*</mo></msup> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>,</mo> <msub><mi>a</mi> <mi>t</mi></msub> <mo>)</mo></mrow> <mo>-</mo> <mi>V</mi> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>)</mo></mrow></mrow></math>。

A3C有一个价值函数V(t)，但它不表达Q函数。相反，A3C通过使用折扣未来回报来估计优势作为Q函数的近似：

<math alttext="upper A Subscript t Baseline equals upper R Subscript t Baseline minus upper V left-parenthesis s Subscript t Baseline right-parenthesis"><mrow><msub><mi>A</mi> <mi>t</mi></msub> <mo>=</mo> <msub><mi>R</mi> <mi>t</mi></msub> <mo>-</mo> <mi>V</mi> <mrow><mo>(</mo> <msub><mi>s</mi> <mi>t</mi></msub> <mo>)</mo></mrow></mrow></math>

这三种技术被证明是A3C在大多数深度强化学习基准中取得成功的关键。A3C代理可以在不到12小时内学会玩Atari Breakout，而DQN代理可能需要3到4天。

## 无监督强化学习和辅助学习

*UNREAL*是A3C的改进，由Jaderberg等人在“带无监督辅助任务的强化学习”中介绍，他们来自DeepMind。

UNREAL解决了奖励稀疏性的问题。强化学习非常困难，因为我们的代理只接收奖励，很难确定奖励增加或减少的确切原因，这使得学习变得困难。此外，在强化学习中，我们必须学习世界的良好表示以及实现奖励的良好策略。使用稀疏奖励这样的弱学习信号来完成所有这些工作是相当困难的。

UNREAL提出了一个问题，我们可以从世界中学到什么，而不需要奖励？它旨在以无监督的方式学习一个有用的世界表示。具体来说，UNREAL在其整体目标中添加了一些额外的无监督辅助任务。

第一个任务涉及UNREAL代理学习其行动如何影响环境。代理被要求通过采取行动来控制屏幕上的像素值。为了在下一帧产生一组像素值，代理必须在这一帧中采取特定的行动。通过这种方式，代理学习其行动如何影响周围的世界，使其能够学习一个考虑自己行动的世界表示。

第二个任务涉及UNREAL代理学习*奖励预测*。给定一系列状态，代理被要求预测下一个接收到的奖励的值。这背后的直觉是，如果一个代理能够预测下一个奖励，那么它可能对环境未来状态有一个相当好的模型，这在构建策略时将会很有用。

通过这些无监督的辅助任务，UNREAL能够在Labyrynth游戏环境中比A3C学习速度快约10倍。UNREAL强调了学习良好的世界表示的重要性，以及无监督学习如何帮助弱学习信号或低资源学习问题，如强化学习。

# 总结

在本章中，我们涵盖了强化学习的基础知识，包括MDPs、最大折扣未来奖励以及探索与利用。我们还涵盖了深度强化学习的各种方法，包括策略梯度和深度Q网络，并涉及了一些关于DQN的最新改进和深度强化学习的新发展。

强化学习对于构建能够不仅感知和解释世界，还能够采取行动并与之互动的代理至关重要。深度强化学习在这方面取得了重大进展，成功地产生了能够掌握Atari游戏、安全驾驶汽车、盈利交易股票、控制机器人等的代理。

^([1](ch13.xhtml#idm45934163590304-marker)) Mnih, Volodymyr, et al. “Human-Level Control Through Deep Reinforcement Learning.” *Nature* 518.7540 (2015): 529-533.

^([2](ch13.xhtml#idm45934163585712-marker)) 这幅图片来自我们在本章中构建的OpenAI Gym DQN代理：Brockman, Greg, et al. “OpenAI Gym.” *arXiv preprint arXiv*:1606.01540 (2016). *https://gym.openai.com*

^([3](ch13.xhtml#idm45934163560160-marker)) 这幅图片来自我们在本章中构建的OpenAI Gym策略梯度代理。

^([4](ch13.xhtml#idm45934166621248-marker)) Sutton, Richard S., et al. “Policy Gradient Methods for Reinforcement Learning with Function Approximation.” NIPS. Vol. 99\. 1999.

^([5](ch13.xhtml#idm45934165544416-marker)) Sorokin, Ivan, et al. “Deep Attention Recurrent Q-Network.” *arXiv preprint arXiv*:1512.01693 (2015).

^([6](ch13.xhtml#idm45934165531792-marker)) Mnih, Volodymyr, et al. “Asynchronous Methods for Deep Reinforcement Learning.” *International Conference on Machine Learning*. 2016.

^([7](ch13.xhtml#idm45934165526704-marker)) Konda, Vijay R., and John N. Tsitsiklis. “Actor-Critic Algorithms.” *NIPS*. Vol. 13\. 1999.

^([8](ch13.xhtml#idm45934165511120-marker)) Jaderberg, Max, et al. “Reinforcement Learning with Unsupervised Auxiliary Tasks.” *arXiv preprint arXiv*:1611.05397 (2016).
