# 第二章：大型语言模型：深入语言建模

### 本章涵盖

+   理解意义和解释的语言背景

+   语言建模技术的比较研究

+   注意力和 transformer 架构

+   大型语言模型如何融入并建立在这些历史之上

> 如果你了解敌人，也了解自己，你就无需害怕百战百胜的结果。——孙子

本章深入探讨与 LLMs 发展相关的语言学，探索符号学的基础、语言特征以及塑造自然语言处理（NLP）领域的语言建模技术的进展。我们将从研究语言学的基础及其与 LLMs 的相关性开始，强调语法、语义和语用学等关键概念，这些概念构成了自然语言的基础，并在 LLMs 的功能中发挥着至关重要的作用。我们将深入研究符号学，即对符号和符号的研究，并探讨其原则如何指导 LLMs 的设计和解释。

然后，我们将追溯语言建模技术的演变，概述早期方法，包括 N-gram、朴素贝叶斯分类器和基于神经网络的多种方法，如多层感知器（MLPs）、循环神经网络（RNNs）和长短期记忆（LSTM）网络。我们还将讨论基于 transformer 架构的突破性转变，这为大型语言模型（LLMs）的出现奠定了基础，而 LLMs 实际上只是大型基于 transformer 的模型。最后，我们将介绍 LLMs 及其独特特征，讨论它们如何建立在并超越早期的语言建模技术，从而彻底改变自然语言处理（NLP）领域。

本书关于生产中的 LLMs。我们坚信，如果你想将 LLM 转化为实际产品，更好地理解技术将提高你的成果并帮助你避免代价高昂且耗时耗力的错误。任何工程师都可以弄清楚如何将大型模型拖入生产并投入大量资源使其运行，但这种蛮力策略完全忽略了人们之前尝试做同样事情时已经学到的教训，这正是我们最初尝试使用 LLMs 的原因。掌握这些基础知识将更好地为你准备那些棘手的部分、陷阱和边缘情况，你将在与 LLMs 合作时遇到这些情况。通过理解 LLMs 出现的背景，我们可以欣赏它们对 NLP 的变革性影响以及如何使它们能够创造无数的应用。

## 2.1 语言建模

如果不首先解决语言问题，就深入探讨 LLMs，那将是一种极大的失职。为此，我们将从对语言建模的简要但全面的概述开始，重点关注有助于我们理解现代 LLMs 的教训。让我们首先讨论抽象层次，因为这将帮助我们理解语言建模。

语言作为一个概念，是我们头脑中产生的情感和思想的抽象。在生成语言的过程中，情感首先出现，但这并不是我们在这里想要强调的唯一事物。我们还在观察语言作为无法捕捉我们能够感受到的全部范围的能力，这就是为什么我们称之为抽象。它远离了原始材料并丢失了信息。数学是语言的抽象，专注于逻辑和可证明性，但正如任何数学家都会告诉你的，它是一种用于以有组织和逻辑的方式描述和定义的语言的子集。从数学中又产生了另一个抽象，二进制语言，这是一个由开或关组成的二进制数值表示系统。

这并不是对有用性的评论，因为二进制和数学与语言的低级方面一样有用，也不是在评论顺序，因为我们之前已经说过。在数学和二进制中，顺序偶然地与抽象层相吻合。计算机不能独立做任何事情，需要接收命令才能变得有用。不幸的是，二进制对于人类来说太耗时了，以至于无法在其中传达重要的事情，因此二进制也被抽象为汇编，这是一种更易于人类理解的语言，用于与计算机通信。这进一步抽象为高级汇编语言 C，它又被进一步抽象为面向对象的语言，如 Python 或 Java（哪一个不重要——我们只是在衡量与二进制的距离）。我们刚才讨论的流程在图 2.1 中有所概述。

![figure](img/2-1.png)

##### 图 2.1 我们将认知抽象层与编程抽象层进行比较，直到逻辑二进制抽象。Python 不是来自 C，也不是编译成 C。然而，Python 是另一个远离二进制的抽象层。语言遵循相似的路径。每一层抽象都创造了一个潜在的错误点。在创建模型的过程中也有几个抽象层，每一层在看到从我们的感受到一个工作模型的全路径时都很重要。

这显然是一种简化；然而，了解你头脑中的感受与计算机实际读取的语言——即大多数人用来编程的语言——在抽象层面上距离二进制相同，是有用的。有些人可能会争论 Python 和二进制之间有更多的步骤，比如编译器或使用汇编来支持 C 语言，这是真的，但在语言方面也有更多的步骤，比如形态学、句法、逻辑、对话和一致性。

这种简化可以帮助我们理解我们想要被 LLM 理解的过程实际上有多么困难，甚至有助于我们更好地理解语言建模技术。在这里我们关注二进制，以说明从你拥有的想法或我们的代码示例到工作模型之间有相似数量的抽象层。就像孩子们玩电话游戏，参与者互相耳语，每个抽象层都创造了一个断开点或障碍，错误可能在这里发生。

图 2.1 不仅旨在说明创建可靠代码和语言输入的难度，而且还强调中间抽象步骤，如分词和嵌入，对模型本身的重要性。即使你拥有完全可靠的代码和完美表达的思想，这些过程之一也可能在它到达 LLM 之前就弄错了意义。

在本章中，我们将尝试帮助你了解你可以在语言、编码或建模方面做些什么来降低这些失败点的风险。不幸的是，在给你太多不立即与当前任务相关的语言学知识，以及给你太多虽然有用但不会帮助你培养对语言建模实践直觉的技术知识之间取得平衡是有点棘手的。考虑到这一点，你应该知道语言学可以追溯到我们历史数千年前，并且有很多东西可以从中学习。我们在附录 A 中简要概述了语言建模随时间的发展，并鼓励你查看。

让我们从构成语言本身的基本构建块开始。我们希望我们的读者至少尝试过语言建模，并且听说过像 PyTorch 和 TensorFlow 这样的库，但我们并不期望大多数读者在之前考虑过语言方面的问题。通过理解构成语言的本质特征，我们可以更好地欣赏创建有效语言模型所涉及的复杂性，以及这些特征如何相互作用，形成连接我们所有人的复杂沟通网络。在下一节中，我们将检查语言的各个组成部分，例如语音学、语用学、形态学、句法和语义学，以及它们在塑造我们对世界各地语言的理解和使用中所起的作用。让我们花点时间来探讨我们目前对语言的理解，以及我们面临的挑战，这些挑战正是 LLMs 旨在解决的。

### 2.1.1 语言特征

我们对语言的理解是，语言至少由五个部分组成：语音学、句法、语义、语用和形态学。这些部分中的每一个都对任何对话中听众所吸收的整体体验和意义有显著的贡献。并不是所有的交流都使用所有这些形式；例如，你现在正在阅读的这本书没有语音学，这也是许多人认为短信不适合更严肃或复杂对话的原因之一。让我们逐一探讨这五个部分，以了解如何将它们呈现给语言模型，以实现全面的沟通能力。

#### 语音学

对于语言模型来说，语音学可能是最容易吸收的，因为它涉及到语言的实际声音。这就是口音显现和处理语音声音生产与感知的地方，语音学专注于特定语言系统中声音的组织方式。与计算机视觉类似，虽然处理整个声音可能并不容易，但在解析、矢量化或标记实际声音波方面没有歧义。每个部分（如每个频率周期的峰值、谷值和斜率）都附有数值。与文本相比，语音学在计算机标记和处理方面要容易得多，尽管它并不简单。

声音本身也包含比文本更多的编码意义。例如，想象有人对你说“是啊，没错”。这可能是一种讽刺，也可能是一种祝贺，这取决于语气——而英语甚至不是音调语言！不幸的是，语音学没有与它通常相关的大型数据集，对语音数据进行数据采集和清洗，尤其是在训练大型语言模型所需的规模上，是极其困难的。在一个音频数据比文本数据更普遍且占用更小内存的世界里，基于语音或对语音敏感的大型语言模型将会更加复杂，创造这样一个世界是一个值得努力的目标。

预见到这个语音问题，1888 年创建了一个名为国际音标（IPA）的系统。它在 20 世纪和 21 世纪都进行了修订，以使其更加简洁、一致、清晰，并且可以作为一种将语音意识插入文本数据的方法。IPA 作为每个语言声音轮廓的国际标准化版本。声音轮廓是语言使用的声音集合；例如，在英语中，我们永远不会把/ʃ/（she, shirt, sh）和/v/声音放在一起。IPA 用于书写声音，而不是像大多数语言那样书写字母或象形文字。例如，你可以使用这些符号来描述如何发音单词“cat”：/k/，/æ/，和/t/。当然，这是一个非常简化的版本，但对于模型来说，并不需要这样。你还可以描述音调和送气。这可能是文本和语音之间的一个折中方案，捕捉一些语音信息。想想短语“What’s up？”你的发音和音调可以极大地改变你对这个短语的理解，有时听起来像友好的“Wazuuuuup？”有时则像几乎威胁的“‘Sup？”而 IPA 可以完全捕捉到这一点。尽管如此，IPA 并不是一个完美的解决方案；例如，它并不能很好地解决复制音调的问题。

声音学在这里被列为首位，因为相较于所有其他特征，声音学在大型语言模型（LLM）中的应用最少，因此具有最大的改进空间。即使是现代的文本到语音（TTS）和声音克隆模型，大部分情况下，最终都是将声音转换成频谱图并分析该图像，而不是结合任何类型的语音语言模型。在未来的几个月和几年里，改善 LLM 中的语音数据和表示将是研究的一个方向。

#### 语法

语法是当前 LLM 表现最出色的领域，无论是从用户那里解析语法还是生成自己的语法。语法通常是我们所认为的语法和词序；它是研究单词如何组合成短语、从句和句子的学科。语法也是语言学习程序开始帮助人们学习新语言的第一步，尤其是基于他们本来的语言背景。例如，对于学习土耳其语的英语母语者来说，了解语法完全不同是很重要的，在土耳其语中，你可以构建整个句子，而英语中我们永远不会把主语和动词放在一起成为一个单词。

语法在语言中很大程度上与意义是分开的，正如著名的句子来自诺姆·乔姆斯基，被称为语法的“之父”所展示的那样：“无色的绿色想法疯狂地睡觉。”这个句子的每一部分在语法上都是正确的，在语义上也是可以理解的。问题不在于它没有意义；而在于它有意义，这些词的编码意义是冲突的。这是一种简化；然而，你可以把 LLMs 给出无意义答案的所有时间都看作是这个现象的表现。不幸的是，语法也是歧义最常见的地方。考虑这个句子，“我看到一个老人和一个女人。”现在回答这个问题：这个女人也是老人吗？这是句法歧义，我们不确定修饰语“老人”是适用于后面短语中的所有人，还是仅仅适用于它直接前面的那个人。这比语义和语用歧义也出现在句法中的事实要轻微得多。考虑这个句子：“我看到一个在山上的男人，他手里拿着望远镜。”回答这些问题：说话者在哪里，他们在做什么？说话者是在山上用望远镜把一个男人切成两半吗？很可能，你在读这个句子的时候甚至没有考虑这个选项，因为当我们解释句法时，我们所有的解释至少在语义和语用上都是有所依据的。我们从生活经验中知道那种解释根本不可能，所以我们立即将其排除，通常甚至没有花时间去处理我们正在从可能的含义池中排除它。单模态 LLMs 将始终存在这个问题，而多模态 LLMs（到目前为止）只能趋近于解决方案。

理解为什么 LLMs 需要具备语法意识才能表现出色，不应该需要任何逻辑上的跳跃。那些没有正确处理词序或生成无意义输出的 LLMs 通常不会被描述为“好”。LLMs 对语法的依赖甚至促使乔姆斯基将 LLMs 称为“随机的鹦鹉”。在我们看来，2018 年的 GPT-2 是语言模型将语法作为一个完全独立于意义的演示解决的时候，我们很高兴看到最近尝试将 GPT-2 输出的语法与编码和蕴涵的意义相结合的尝试，我们现在将深入探讨这一点。

#### 语义

语义是话语中词语的直译意义，它以惊人的速度在波浪中变化。人们会自动优化语义意义，只使用他们认为在当前语言时代有意义的词语。如果你曾经创建或使用过语言模型（word2vec、ELMo、BERT、MUSE [E 代表嵌入]，等等）的嵌入，你就已经使用了语义近似。词语经常经历语义变化，虽然我们不会完全涵盖这个主题或深入探讨，但这里有一些你可能已经熟悉的常见变化：缩小，从更广泛的意义到更具体的一个；扩大，与缩小相反，从具体意义到广泛意义；以及重新解释，通过整体或部分转换。这些变化并没有某种伟大的逻辑基础。它们甚至不需要与现实相关联，语言的使用者通常在变化发生时也不会有意识地思考这些变化。这并不能阻止变化的发生，在语言建模的背景下，这也不能阻止我们必须跟上这种变化。

让我们来看一些例子。缩小包括“鹿”，在古英语和中古英语中这个词仅仅指任何野生动物，甚至包括熊或美洲狮，而现在只指一种森林动物。对于扩大，我们有“狗”，它曾经只指来自英格兰的一种犬种，而现在可以用来指任何家养犬。关于狗的扩大有一个有趣的小插曲，在 FromSoft 游戏《艾尔登法环》中，由于玩家之间有限的消息系统，"狗"这个词会被用来指从乌龟到巨型蜘蛛以及介于两者之间的所有事物。对于重新解释，我们可以考虑“漂亮”，它曾经意味着聪明或工艺精湛，而不是视觉上吸引人。另一个很好的例子是“比基尼”，它从指一个特定的环礁，到指在访问那个环礁时可能穿的衣服，再到人们好像“bi-”指的是衣服的两件式结构，从而暗示了坦克尼和单件比基尼。基于专家研究和数十年的研究，我们可以认为语言是不断被母语使用者比较和重新评估的，从中产生了常见的模式。这些模式在语言社会学的传播被密切研究，但很大程度上超出了当前目的的范围，但当地化（l10n）或国际化（i18n）作为 LLMs 项目需求出现时，可以迅速进入范围。像声望这样的社会语言学现象可以帮助设计出对每个人都适用的系统。

在 LLMs（大型语言模型）的背景下，所谓的语义嵌入是文本的向量化版本，试图模仿语义含义。目前，最流行的方法是通过分词或将字典中的任意数字分配给每个话语中的子词（例如前缀、后缀和一般词素），应用连续语言模型来增加向量中每个标记的维度，以便有一个更大的向量代表每个标记向量的索引，然后对每个这些向量应用位置编码以捕捉词序。每个子词最终都会根据其使用方式与其他字典中的单词进行比较。我们稍后会展示一个例子。在思考词嵌入时，需要考虑的是，它们难以捕捉那些标记的深层编码含义，而简单地增加嵌入的维度并没有显示出显著的改进。嵌入与人类工作方式相似的证据是，你可以应用距离函数到相关单词上，并看到它们比无关单词更接近。如何更完整地捕捉和表示意义是未来几年和几个月内有望出现突破性研究的另一个领域。

#### 话语学

话语学有时被省略在语言学中，因为它的参照物是所有影响听者解释和说话者决定以某种方式表达的非语言环境。话语学在很大程度上指的是文化、地区、社会经济阶层和共同生活经历中遵循的教条，这些教条在对话中通过蕴涵来走捷径。

如果我们说，“一位受欢迎的明星刚刚被送进了 ICU”，根据你的生活经验，你的语用解释可能是假设一个深受爱戴的人受到了严重伤害，现在正在一家设备齐全的医院接受治疗。你可能想知道这是哪位明星，他们是否需要支付医疗费用，或者伤害是否是自残的，这也基于你的生活经验。这些都不能直接从文本及其编码的意义中推断出来。你需要知道 ICU 代表一组更大的词汇以及这些词汇是什么。你需要知道医院是什么，以及为什么有人需要被送到那里而不是自己去。如果这些感觉很明显，很好。你生活在一个社会中，你对这个社会的语用知识很好地与提供的例子重叠。如果我们分享一个来自人口较少社会的例子，“Janka 昨天受到了严厉的惩罚；她明天会得到 Peter 的惩罚”，你可能会感到困惑。如果你感到困惑，意识到这可能就是许多文本数据对 LLMs（承认拟人化）看起来像的样子。对于那些想知道的人，这个句子来自斯洛伐克的复活节传统。如果你不习惯这些特定的传统，那么这里很多含义都会被错过，无法解释。这位作者个人很享受尝试向外国同事解释复活节兔子和它对鸡蛋的迷恋，并享受着看起来像是脱离了现实的感觉。

在 LLMs（大型语言模型）的背景下，我们可以有效地将所有非文本上下文归入语用学。这意味着 LLMs 在开始时对现实世界没有任何知识，并且在训练过程中也不会获得这些知识。它们只获得对人类如何对特定的语用刺激做出反应的知识。LLMs 不理解社会阶层、种族、性别或总统候选人，或者任何可能基于你的生活经历激发某种情感的其他事物。我们并不期望语用学能够在任何时刻直接被纳入模型中，因为模型无法生活在社会中。然而，我们已经看到了通过数据工程和整理、RAG（阅读-询问-生成）提示机制以及指令数据集上的监督微调间接纳入语用学的益处。在未来，我们预计在将语用学纳入 LLMs 方面将取得重大进步，但我们强调这是一个渐近解，因为语言最终仍然是一种抽象。

无论是你有意还是无意，一旦你获取了将要训练的数据，实用结构就会被添加进去。你可以将这种类型的实用结构视为偏见，它本身并不一定是好是坏，但却是无法摆脱的。在后续过程中，你可以通过标准化和整理，增强特定代表性不足的点，以及削减过度代表或噪声示例，来选择你希望数据保持的偏见类型。指令数据集展示了你如何在训练数据中利用实用结构来创建极其有用的偏见，例如，当被要求回答问题时，让模型偏向于回答问题而不是尝试对问题的情感进行分类。

语境和上下文都围绕着蕴涵。蕴涵是数据中的实用标记，与数据集中包含的文本内容相对。例如，假设你有一个模型试图处理一个输入，如“为我写一篇关于青蛙吃湿袜子且不押韵的演讲稿，每行的第一个字母拼写出两栖动物”，并真正遵循这个指令。你可以立即判断这个输入要求很多。作为数据工程师的你，需要确保输入要求的所有内容都在你的数据中得到明确考虑。你需要演讲的例子，青蛙和袜子的例子以及它们的行为，以及首字母诗的例子。如果你没有这些例子，模型可能只能从数据集中存在的蕴涵中理解，但这很不确定。如果你更进一步，跟踪数据集中蕴涵与显性信息以及任务，以及数据分布，你将会有例子来回答“垃圾输入导致垃圾输出是什么？”

大型语言模型（LLMs）在捕捉语用学方面比人更困难，但它们确实能捕捉到普通人平均标准差范围内的东西。它们甚至可以复制超出那个标准差的人的回答，但如果没有精确的刺激，这种复制通常是不一致的。这意味着，如果没有在训练期间和提示中提供正确的偏见和蕴涵，模型很难在没有提供正确偏见和蕴涵的情况下给出一个普通人不知道的问题的专家答案。例如，在图像生成提示的开头包含“杰作”一词将引发不同且通常质量更高的生成，但只有当这种区别存在于训练集中，并且只有当你要求的是一幅“杰作”是赞美之词的图像时。基于指令的数据集试图在训练期间通过提问和给出包含代表性回答的指令来制造这些刺激。在训练中考虑到每一种可能的情况是不可能的，并且你可能会在试图考虑到一切时无意中从你的最终用户那里创造出新的回答类型。训练后，你可以通过提示从你的模型中诱导出特定的信息，这有一个基于你的数据最初包含的内容的技能上限。

#### 形态学

形态学是研究单词结构和它们是如何由称为词素的小单元构成的学科。词素是意义的最小单位，比如“re-”在“redo”或“relearn”中。然而，并不是单词的所有部分都是词素，例如“ra-”在“ration”或“na-”在“nation”中，有些可能是出乎意料的，比如“helico-”在“helicoid”和“-pter”在“pterodactyl”中。

理解单词是如何构成的有助于创建更好的语言模型和解析算法，这对于像分词这样的任务至关重要。分词是自然语言处理中使用的最基本单位；它们可以是单词、子词、字符或整个话语，并且不必与现有的词素相对应。人们并不自觉地决定他们的意义单位是什么，因此它们往往是逻辑不合理的。语言模型的有效性可能取决于它理解和处理这些分词的能力有多强。例如，在分词过程中，模型需要存储一组字典来在单词及其对应的索引之间进行转换。其中这些分词之一通常是一个`/<UNK/>`分词，它代表模型不认识的任何单词。如果这个分词使用得太频繁，可能会阻碍模型的表现，要么是因为模型词汇量太小，要么是因为分词器没有使用正确的算法来完成这项任务。

考虑这样一个场景，你想要构建一个代码补全模型，但你使用的是一个只能识别由空格分隔的单词的标记器，比如 NLTK 的`punkt`标记器。当它遇到字符串`def` `add_two_numbers_together(x,` `y):`时，它将把`[def,` `[UNK],` `y]`传递给模型。这导致模型丢失了有价值的信息，不仅因为它不识别标点符号，而且还因为函数目的的重要部分被标记器形态算法替换成了未知标记。为了提高模型的表现，需要更好地理解词的结构和适当的解析算法。

### 2.1.2 符号学

在探索语言的基本特征并考察它们在 LLMs（大型语言模型）背景下的重要性之后，考虑人类沟通中意义构建和解释的更广泛视角是至关重要的。符号学，即对符号和象征的研究，提供了一个有价值的视角，通过它我们可以更好地理解人们如何解释和加工语言。我们将深入探讨符号学，考察符号、符号指示物和抽象之间的关系，以及 LLMs 如何利用这些元素来生成有意义的输出。这次讨论将提供对 LLMs 如何模仿人类对语言的理解的复杂过程的更深入理解，同时也会揭示它们在这一努力中所面临的挑战和限制。我们并不一定认为模仿人类行为是 LLM 改进的正确答案，只是认为模仿是迄今为止该领域自我评估的方式。

为了介绍符号学，让我们考虑图 2.2，一个改编的皮尔士符号三角形。这些三角形被用来将基本思想组织成第一性、第二性和第三性的序列，其中第一性位于左上角，第二性位于底部，第三性位于右上角。如果你以前见过符号三角形，你可能会对角的数量和方向感到惊讶。为了解释，我们将它们倒置，使其稍微容易阅读一些。此外，因为系统是递归的，我们展示了系统如何同时模拟整个过程和每个部分。虽然这些想法的整体概念非常酷，但深入探讨哲学超出了本书的范围。相反，我们可以专注于那些词（第一、第二、第三）的基本部分，以展示意义处理的过程。

![figure](img/2-2.png)

##### 图 2.2 一个递归的皮尔士符号三角形是一个组织从任何事物中提取意义的过程的系统——在我们的例子中，是从语言中提取。三角形上的每一个点都说明了在系统被用来描述的任何事物中合成意义所需的最低限度的部分，因此每个点都是语言意义的最小单位。第一性、第二性和第三性不是三角形上的点；相反，它们更像是符号学家用来在这个图表中定位自己的标记。

我们还可以观察三角形的每一个交点，以了解为什么事物以这种顺序呈现。情感可以在它们能够附加到单词和表格之前就附加到图像和编码上。仪式和常见的脚本为解释性行动提供了一个空间，这种行动是第二本能的，不需要思考，就像大多数短语只是从单词中组合起来，母语使用者不需要对每个单词进行元认知一样。所有这些最终都会导致一种解释或一份文档（一系列话语）；在我们的例子中，这种解释应该由 LLM 得出。这就是为什么，例如，提示工程可以提高模型的有效性。在数百万个仪式脚本示例上训练的基础 LLM，当你明确告诉模型在提示中需要遵循哪个脚本时，可以显著更好地复制脚本类型。试着要求模型给出逐步解释——也许在你的生成前加上“让我们一步步来考虑这个问题。”模型将根据它之前看到的脚本生成逐步脚本。

对于感兴趣的人来说，有特定的方法来阅读这些图表，以及一个完整的符号学领域需要考虑；然而，并不能保证你通过理解这一切就能创造出最佳的 LLMs。我们不会深入探讨这一点，而是考虑最基本的可以帮助你构建最佳模型、用户体验和用户界面的要素。例如，创建意义的过程中的一个方面是递归性。当有人和你交谈，他们说的话对你来说没有意义（对你来说是“无意义的”），你会怎么做？通常，人们会提出一个或多个澄清问题来弄清楚意义，这个过程会一次又一次地重复，直到意义对你来说变得清晰。目前市场上最先进的模型并不这样做，但通过非常目的性的提示，它们可以被训练去做这件事。许多人甚至不知道该怎么做，除非有人指出。换句话说，这只是一个关于符号学的简要介绍。你不需要在阅读本节结束时能够向符号学领域的专家提供深入和准确的坐标特定解释。我们试图说明的是，这是一个展示你需要创建一个完整意义图景的最小要素的组织系统。我们在训练模型时并没有提供相同数量和类型的相同信息，但如果我们这样做，将会显著改善模型的行为。

图 2.2 和图 2.3 旨在表示一个最小化的组织模型，其中每个部分都是至关重要的。让我们考虑图 2.3，它通过一个使用符号三角形的示例来展示。考虑图像、图片和记忆，想想在没有眼睛处理图像和没有正字法（一种书写系统）来抽象知识的情况下，尝试吸收这本书中的知识会是什么样子。看看项目符号等，如果没有章节、字母之间的空白和项目符号来显示顺序和结构，你将如何阅读这本书？看看语义和字面编码的意义，想象一下没有图表或没有字典定义的单词的书。中间的表格可能是一本书，没有任何表格或比较信息组织者，包括这些图表。在没有文化或社会习惯和教条作为我们解释的透镜的情况下，阅读这本书会是什么样子？所有这些点构成了我们解读信息的能力，以及我们通过透镜传递信息以识别模式的能力。

![图表](img/2-3.png)

##### 图 2.3 从左上角开始，按照箭头指示的顺序，查看我们构建解释和从我们与之互动的事物中提取意义的一般顺序。在这里，我们用每个点的例子替换了描述性词汇。试着想象在没有任何文字、例子、箭头，甚至知道这本书中的图例应该用于什么目的的实用语境下解释这个图。

因此，这些问题很重要：LLMs 能够访问多少这些事物以返回有意义的解释？LLMs 能够访问情感或社会仪式吗？目前，它们还不能，但随着我们通过传统的和更新的 NLP 推理技术进行探索，想想不同模型能够访问什么。

### 2.1.3 多语言 NLP

在我们评估之前的 NLP 技术和当前一代 LLMs 之前，我们需要触及的最后一个挑战是语言学的基础，以及 LLMs 甚至存在的理由。自从最早的人类文明开始接触以来，人们就想要理解或利用彼此。这些案例导致了翻译的需求，随着全球经济的发展和繁荣，这种需求呈指数级增长。

对于企业来说，这也很简单。你知道孟加拉语母语者的人数几乎和英语母语者一样多吗？如果你第一次听说孟加拉语，这应该有助于你认识到多语言模型有一个有价值的市场。世界上有数十亿人，但只有大约三分之一的人母语是英语。如果你的模型以英语为中心，就像大多数模型一样，你正在错过世界上 95%的人作为客户和用户。西班牙语和普通话在这个领域很容易取得成功，但大多数人甚至没有走那么远。

在这本书的范围之外，有许多政治上充满争议的例子，包括不同语言中称呼同一事物的情况。这些情况通常是因为外部因素，如政府干预。牢记这两点——一个以英语为中心的单语系统并没有许多企业所声称的覆盖面或盈利潜力，而语言和方言之间的界限在最坏的情况下是有害的——应该会凸显出意见的险恶沼泽。许多企业和研究科学家在设计产品或系统时甚至不愿假装想要用 50 英尺长的杆子去触碰这个沼泽。

目前，没有简单的解决方案。然而，考虑到这些因素可以帮助你作为一个科学家或工程师（以及希望是一个有道德的人）设计 LLMs，至少它们不会加剧现有问题并产生负面影响。这个过程的第一步是在项目开始时就决定一个方向目标，要么是本地化（l10n），要么是国际化（i18n）。本地化是一个由 Mozilla 所体现的方法，它在超过 90 种语言中通过众包 l10n 提供了其浏览器的不同版本，并且没有迹象表明会停止这项努力。国际化与此类似，但方向相反；例如，宜家试图在其说明书上使用尽可能少的文字，而是选择使用国际上公认的符号和图片来帮助客户导航 DIY 项目。在项目开始时做出决定可以减少扩展到任何解决方案所需的努力。它足够大，足以将翻译和格式化的感知从成本转变为投资。在 LLMs 及其在公众意识中的快速扩张的背景下，尽早考虑这一点变得更加重要。带着一种自动禁止世界上大多数人与之互动的世界改变技术进入市场，贬低了那些声音。需要等待则危及企业的经济前景。

在继续之前，让我们花一点时间回顾一下到目前为止我们所讨论的内容。我们在语言学方面触及了重要观点，为我们提供了需要考虑的概念，例如理解语言的结构与其意义是分开的。我们已经展示了一个旅程，每个人作为个人以及作为社会，都在朝着拥有元认知来以连贯的方式理解和表示语言给计算机使用前进。随着我们加深对认知领域的了解并解决我们遇到的语用特征，这种理解将得到改善。跟随图 2.1，我们现在将展示我们所遵循的语言建模的计算路径，并探讨它如何以及如何没有解决那些语用特征或努力创造意义。让我们进入评估表示语言的各种算法技术。

## 2.2 语言建模技术

在深入研究语言的基本特征、符号学原理以及 LLMs 如何解释和处理语言信息之后，我们现在过渡到一个更实际的领域。我们将探讨为创建这些强大的语言模型而开发和应用的多种 NLP 技术。通过检查每种方法的优缺点，我们将获得宝贵的见解，了解这些技术在捕捉人类语言和交流本质方面的有效性。这种知识不仅将帮助我们欣赏 NLP 领域取得的进步，而且使我们能够更好地理解这些模型的当前局限性以及未来研究和开发面临的挑战。

让我们花点时间回顾一下在所有语言模型中都会通用的数据处理方法。首先，我们需要决定如何将我们传递给模型的单词和符号进行拆分，实际上就是决定在我们模型中一个标记将是什么。我们需要一种方法将这些标记转换为数值，然后再转换回来。然后，我们需要选择我们的模型将如何处理这些标记化的输入。以下的所有技术至少以一种方式建立在之前的技术之上。

这些技术中的第一个被称为词袋（BoW）模型，它简单地由文本中出现的单词计数组成。你可以从 sklearn 导入 CountVectorizer 类来使用它，但如果我们用一个小的代码片段来展示，可能会更有教育意义。这可以通过一个扫描文本的字典非常容易地完成，为每个新单词创建一个新的词汇条目作为键，并从 1 开始递增的值：

```py
sentence = "What is a bag of words and what does it do for me when " \
    "processing words?"
clean_text = sentence.lower().split(" ")
bow = {word:clean_text.count(word) for word in clean_text}
print(bow)
# {'what': 2, 'is': 1, 'a': 1, 'bag': 1, 'of': 1, 'words': 1, 'and': 1, 
# 'does': 1, 'it': 1, 'do': 1, 'for': 1, 'me': 1, 'when': 1, 'processing': 1,
# 'words?': 1}
```

考虑到其简单性，即使是完全基于频率的这种模型，在试图洞察说话者的意图或至少他们的独特性时，也可以相当强大。例如，你可以在美国总统的就职演说上运行一个简单的词袋模型（BoW model），搜索“freedom”、“economy”和“enemy”这些词，从而对哪些总统在和平时期、战争时期和货币紧张时期上任有一个相当好的了解，这仅仅基于每个词被提及的次数。然而，BoW 模型有很多弱点，因为模型不提供图像、语义、语用、短语或情感。在我们的例子中，有两个“words”的实例，但由于我们的标记化策略仅仅是空白字符，它没有增加模型中的键。它没有任何机制来评估上下文或语音学，并且因为它默认在空白字符处分割单词（你显然可以按任何方式标记化，但尝试在子词上标记化并看看这个模型会发生什么——剧透：很糟糕），它也没有考虑形态学。总的来说，它应该被认为是一个弱的语言表示模型，但是一个强大的基准，可以用来评估其他模型。为了解决 BoW 模型无法捕捉任何序列数据的问题，人们提出了 N-gram 模型。

### 2.2.1 N-gram 和语料库技术

##### 列表 2.1 一个生成 N-gram 语言模型的实现

```py
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.util import everygrams
from nltk.lm.preprocessing import (
    pad_both_ends,
    flatten,
    padded_everygram_pipeline,
)
from nltk.lm import MLE

my_corpus = PlaintextCorpusReader("./", ".*\.txt")    #1

for sent in my_corpus.sents(fileids="hamlet.txt"):
    print(sent)

padded_trigrams = list(
    pad_both_ends(my_corpus.sents(fileids="hamlet.txt")[1104], n=2)
)                                                  #2
list(everygrams(padded_trigrams, max_len=3))

list(
    flatten(
        pad_both_ends(sent, n=2)
        for sent in my_corpus.sents(fileids="hamlet.txt")
    )
)

train, vocab = padded_everygram_pipeline(
    3, my_corpus.sents(fileids="hamlet.txt")
)                                              #3

lm = MLE(3)           #4
len(lm.vocab)           #5

lm.fit(train, vocab)
print(lm.vocab)
len(lm.vocab)

lm.generate(6, ["to", "be"])   #6
```

#1 从任意数量的普通 .txt 文件创建语料库

#3 允许 everygrams 从数据中创建一个训练集和一个词汇对象

#4 实例化和训练我们将用于 N-grams 的最大似然估计器（MLE）的模型

#5 此模型将采用 everygrams 词汇表，包括用于未知词汇的<UNK>标记。

#6 可以使用此模型生成语言，并使用 n-1 个标记进行条件化。

这段代码就是您需要创建生成性 N-gram 模型的所有内容。对于那些有兴趣进一步评估该模型的人来说，我们包括了以下代码，以便您可以获取概率和对数分数，或者分析特定短语的熵和困惑度。因为这一切都是基于频率的，尽管它在数学上很重要，但它仍然无法很好地描述现实世界语言的实际困惑度或频率：

```py
print(lm.counts)
Lm.counts[["to"]]["be"]      #1

print(lm.score("be"))           #2
print(lm.score("be", ["to"]))
print(lm.score("be", ["not", "to"]))

print(lm.logscore("be"))             #3
print(lm.logscore("be", ["to"]))
print(lm.logscore("be", ["not", "to"]))

test = [("to", "be"), ("or", "not"), ("to", "be")]   #4
print(lm.entropy(test))
print(lm.perplexity(test))
```

#1 可以轻松计数长度不超过 n 的任何一组标记，以确定频率。

#2 任何标记都可以赋予一个发生概率，并可以增加最多 n-1 个标记来先于它。

#3 这也可以作为一个对数分数来完成，以避免非常大的和非常小的数字。

#4 可以测试标记集的熵和困惑度。

尽管这个代码示例说明了创建三元语言模型，但不幸的是，并非所有需要捕获的短语都只有三个标记长。例如，从《哈姆雷特》中，“To be or not to be”由一个包含两个单词和一个包含四个单词的短语组成。请注意，尽管 N-grams 通常是较小的语言模型，但通过将 N 设置为 10 亿或更高，可以制作一个 N-gram LLM，但不要期望从中得到任何实际用途。仅仅因为我们做得很大，并不意味着它会更好，或者意味着它将具有任何实际应用：99.9%的所有文本和 100%的有意义文本中，出现次数超过一次的标记少于 10 亿，而且这种计算能力可以更好地用于其他地方。

N-grams 仅使用静态信号（空白字符、正字法）和单词来提取意义（图 2.2）。它们试图手动测量短语，假设所有短语都将具有相同的长度。话虽如此，N-grams 可以用于创建文本分析的强大基线。此外，如果分析师已经知道话语的语用背景，N-grams 可以快速准确地洞察现实世界场景。尽管如此，这种短语建模无法捕捉到单个单词可能具有的任何语义编码。为了解决这个问题，将贝叶斯统计应用于语言建模。

### 2.2.2 贝叶斯技术

贝叶斯定理是描述你的输出在输入空间中发生的一种最数学上合理且简单的理论。本质上，它基于先验知识计算事件发生的概率。该定理提出，给定证据的假设为真的概率——例如，一个句子具有积极情感的概率——等于给定假设为真的证据发生的概率乘以假设发生的概率，所有这些除以证据为真的概率。它可以表示为数学公式

*P*(*hypothesis* | *evidence*) = (*P*(*evidence* | *hypothesis*) × *P*(*hypothesis*)) / *P*(*evidence*)

或者

*P*(*A*|*B*) × *P*(*B*) = *P*(*B*|*A*) × *P*(*A*)

由于这不是一本数学书，我们将深入探讨贝叶斯定理，就像我们深入研究其他语言学概念一样，并相信感兴趣的读者会去寻找更多。

不幸的是，尽管这个定理以数学上合理的方式表示数据，但它没有考虑到任何随机性或单词的多种含义。你可以向贝叶斯模型投掷的一个总是能使其混淆的词是“它”。任何指示代词最终都会以与所有其他单词相同的`LogPrior`和`LogLikelihood`方式分配值，并且它得到一个静态值，这与这些词的使用是相反的。例如，如果你正在尝试对一个话语进行情感分析，将所有代词分配一个空值会比让它们通过贝叶斯训练更好。请注意，贝叶斯技术不会像其他技术那样创建生成语言模型。由于贝叶斯定理验证假设的性质，这些模型适用于分类，并且可以为生成语言模型带来强大的增强。

列表 2.2 展示了如何创建一个朴素贝叶斯分类语言模型，或者是一个基于先前学习内部语言模型对文本进行分类的系统。我们选择编写我们正在做的事情，而不是使用像 sklearn 这样的包或使编写代码变得容易的东西，所以它会更长，但它应该提供更多关于它是如何工作的信息。我们使用的是朴素贝叶斯模型的最简单版本。我们没有将其改为多项式或添加任何花哨的东西；显然，如果你选择升级它以解决任何问题，它会工作得更好。我们强烈建议你这样做。

注意：为了使代码更容易理解并帮助突出我们想要关注的部分，我们通过提取部分代码到实用辅助函数中，简化了一些我们的代码列表。如果你看到导入错误，这就是原因。这些辅助方法可以在本书附带的代码库中找到：[`github.com/IMJONEZZ/LLMs-in-Production/`](https://github.com/IMJONEZZ/LLMs-in-Production/)

##### 列表 2.2 分类朴素贝叶斯语言模型实现

```py
from utils import process_utt, lookup
from nltk.corpus.reader import PlaintextCorpusReader
import numpy as np

my_corpus = PlaintextCorpusReader("./", ".*\.txt")

sents = my_corpus.sents(fileids="hamlet.txt")

def count_utts(result, utts, ys):
    """
    Input:
        result: a dictionary that is used to map each pair to its frequency
        utts: a list of utts
        ys: a list of the sentiment of each utt (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    """

    for y, utt in zip(ys, utts):
        for word in process_utt(utt):
            pair = (word, y)          #1

            if pair in result:
            result[pair] += 1     #2

            else:
            result[pair] = 1      #3

    return result

result = {}
utts = [" ".join(sent) for sent in sents]
ys = [sent.count("be") > 0 for sent in sents]
count_utts(result, utts, ys)

freqs = count_utts({}, utts, ys)
lookup(freqs, "be", True)
for k, v in freqs.items():
    if "be" in k:
        print(f"{k}:{v}")

def train_naive_bayes(freqs, train_x, train_y):
    """
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of utts
        train_y: a list of labels correponding to the utts (0,1)
    Output:
        logprior: the log prior.
        loglikelihood: the log likelihood of you Naive bayes equation.
    """
    loglikelihood = {}
    logprior = 0

    vocab = set([pair[0] for pair in freqs.keys()])      #4
    V = len(vocab)

    N_pos = N_neg = 0                  #5
    for pair in freqs.keys():
        if pair[1] > 0:                   #6
            N_pos += lookup(freqs, pair[0], True)     #7

        else:                                         #8
            N_neg += lookup(freqs, pair[0], False)     #9

    D = len(train_y)     #10

    D_pos = sum(train_y)      #11

    D_neg = D - D_pos     #12

    logprior = np.log(D_pos) - np.log(D_neg)      #13

    for word in vocab:                        #14
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        p_w_pos = (freq_pos + 1) / (N_pos + V)    #15
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        loglikelihood[word] = np.log(p_w_pos / p_w_neg)     #16

    return logprior, loglikelihood

def naive_bayes_predict(utt, logprior, loglikelihood):
    """
    Input:
        utt: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods + logprior
    """
    word_l = process_utt(utt)     #17

    p = 0          #18

    p += logprior      #19

    for word in word_l:
        if word in loglikelihood:       #20
            p += loglikelihood[word]      #21

    return p

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Input:
        test_x: A list of utts
        test_y: the corresponding labels for the list of utts
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of utts classified correctly)/(total # of utts)
    """
    accuracy = 0       #22

    y_hats = []
    for utt in test_x:
        if naive_bayes_predict(utt, logprior, loglikelihood) > 0:    #23
            y_hat_i = 1    #24
        else:
            y_hat_i = 0    #25

        y_hats.append(y_hat_i)      #26

    error = sum(
        [abs(y_hat - test) for y_hat, test in zip(y_hats, test_y)]
    ) / len(y_hats)                 #27

    accuracy = 1 - error    #28

    return accuracy

if __name__ == "__main__":
    logprior, loglikelihood = train_naive_bayes(freqs, utts, ys)
    print(logprior)
    print(len(loglikelihood))

    my_utt = "To be or not to be, that is the question."
    p = naive_bayes_predict(my_utt, logprior, loglikelihood)
    print("The expected output is", p)

    print(
        f"Naive Bayes accuracy = {test_naive_bayes(utts, ys, logprior, loglikelihood):0.4f}
    )
```

#1 Δ 定义了键，即单词和标签元组

#2 如果键存在于字典中，则增加计数

#3 如果键是新的，则将其添加到字典中并将计数设置为 1

#4 计算 V，词汇表中的唯一单词数量

#5 计算 N_pos 和 N_neg

#6 如果标签是正面的（大于零） . . .

#7 . . . 增加正面单词的数量（单词，标签）

#8 否则，标签是负面的。

#9 增加负面单词的数量（单词，标签）

#10 计算 Δ，文档数量

#11 计算正面文档的数量

#12 计算负面文档的数量

#13 计算对数先验概率

#14 对于词汇表中的每个单词 . . .

#15 . . . 计算每个单词是正面还是负面的概率

#16 计算单词的对数似然值

#17 处理语音以获取单词列表

#18 将概率初始化为零

#19 添加对数先验概率

#20 检查单词是否存在于对数似然字典中

#21 将该单词的对数似然值添加到概率中

#22 正确返回

#23 如果预测值大于 0 . . .

#24 . . . 预测的类别是 1。

#25 否则，预测的类别是 0。

#26 将预测的类别添加到列表 y_hats 中

#27 误差 = y_hats 和 test_y 之间差的绝对值的平均值。

#28 准确率是 1 减去误差。

这个定理并没有创建同类型的语言模型，而是一个与一个假设相关联的概率列表。因此，贝叶斯语言模型不能有效地用于生成语言，但它们在分类任务中可以非常强大地实现。然而，我们认为，贝叶斯模型对于这项任务通常被过度炒作。一位作者职业生涯的辉煌成就之一就是从生产中替换并移除了一个贝叶斯模型。

在贝叶斯模型中，一个主要问题就是所有序列都是完全未连接的，就像词袋模型一样，将我们推向序列建模和语言生成的对立面，从 N-gram 开始。就像摆锤一样，语言建模又回到了序列建模和语言生成，使用马尔可夫链。

### 2.2.3 马尔可夫链

通常被称为隐马尔可夫模型（HMMs），马尔可夫链本质上是在 N-gram 模型中添加状态，使用隐藏状态存储概率。它们常用于帮助解析文本数据，用于更大的模型，如进行词性标注（PoS；标记单词的词性）和命名实体识别（NER；标记具有其指代和通常类型的识别词；例如，LA – 洛杉矶 – 城市）等。建立在之前的贝叶斯模型之上，马尔可夫模型完全依赖于遇到的标记中的随机性（可预测的随机性）。任何发生的事情的概率完全取决于现在的状态，这一观点与贝叶斯定理一样，在数学上是合理的。因此，我们不是仅仅基于单词的历史出现来建模单词，并从中抽取概率，而是根据当前发生的情况来建模它们的未来和过去搭配。因此，“快乐”发生的概率如果“快乐”刚刚输出就会下降到几乎为零，但如果“am”刚刚出现，概率就会显著上升。马尔可夫链如此直观，以至于它们被纳入了贝叶斯统计学的后续迭代中，并且至今仍在生产系统中使用。

在列表 2.3 中，我们训练了一个马尔可夫链生成语言模型。这是我们第一次使用特定的标记器，在这种情况下，它将基于单词之间的空白进行标记。这也是我们第二次将一组旨在一起查看的陈述视为文档。当您玩弄这个模型时，请密切关注，并自己做一些比较，看看 HMM 的生成效果是否比大型 N-gram 模型好。

##### 列表 2.3 生成隐马尔可夫语言模型实现

```py
import re
import random
from nltk.tokenize import word_tokenize
from collections import defaultdict, deque

class MarkovChain:
    def __init__(self):
        self.lookup_dict = defaultdict(list)
        self._seeded = False
        self.__seed_me()

    def __seed_me(self, rand_seed=None):
        if self._seeded is not True:
            try:
                if rand_seed is not None:
                    random.seed(rand_seed)
                else:
                    random.seed()
                self._seeded = True
            except NotImplementedError:
                self._seeded = False

    def add_document(self, str):
        preprocessed_list = self._preprocess(str)
        pairs = self.__generate_tuple_keys(preprocessed_list)
        for pair in pairs:
            self.lookup_dict[pair[0]].append(pair[1])

    def _preprocess(self, str):
        cleaned = re.sub(r"\W+", " ", str).lower()
        tokenized = word_tokenize(cleaned)
        return tokenized

    def __generate_tuple_keys(self, data):
        if len(data) < 1:
            return

        for i in range(len(data) - 1):
            yield [data[i], data[i + 1]]

    def generate_text(self, max_length=50):
        context = deque()
        output = []
        if len(self.lookup_dict) > 0:
            self.__seed_me(rand_seed=len(self.lookup_dict))
            chain_head = [list(self.lookup_dict)[0]]
            context.extend(chain_head)

            while len(output) < (max_length - 1):
                next_choices = self.lookup_dict[context[-1]]
                if len(next_choices) > 0:
                    next_word = random.choice(next_choices)
                    context.append(next_word)
                    output.append(context.popleft())
                else:
                    break
            output.extend(list(context))
        return " ".join(output)

if __name__ == "__main__":
    with open("hamlet.txt", "r", encoding="utf-8") as f:
        text = f.read()
    HMM = MarkovChain()
    HMM.add_document(text)

    print(HMM.generate_text(max_length=25))
```

这段代码展示了生成马尔可夫模型的基本实现，我们鼓励您对其进行实验。用您最喜欢的音乐家的歌曲或您最喜欢的作者的书中的文本进行尝试，看看输出的内容是否像他们。隐马尔可夫模型（HMMs）非常快速，常用于预测文本或预测搜索应用。马尔可夫模型代表了从描述性语言学的角度对语言进行建模的第一个全面尝试，而不是规范性建模。这种视角很有趣，因为马尔可夫最初并没有打算使用语言建模，只是想在一个关于连续独立状态的争论中获胜。后来，马尔可夫使用马尔可夫链来模拟普希金小说中的元音分布，所以他至少意识到了可能的用途。

描述性语言学和规范性语言学的区别在于，后者关注事物应该如何，而前者关注事物实际上是如何的。从语言建模的角度来看，从语料库或马尔可夫视角描述语言所做的事情，比试图规定语言应该如何表现要有效得多。不幸的是，当前状态本身不能用来提供超越现在的语境，因此历史或社会语境在马尔可夫模型中无法有效表示。单词的语义编码也变得有问题，如代码示例所示：马尔可夫链会输出语法上正确但语义上无意义的单词序列，类似于“无色的绿色想法疯狂地睡觉。”为了解决这个问题，开发了“连续”模型，以允许对标记进行“语义嵌入”表示。

### 2.2.4 连续语言建模

连续词袋模型（CBoW）——就像其同名的词袋模型一样——是一种基于频率分析语言的方法，这意味着它根据单词出现的频率来建模单词。人类话语中的下一个单词从未根据概率或频率来确定。因此，我们提供了一个使用 CBoW 创建词嵌入的例子，以便其他模型可以摄取或比较。我们将使用神经网络为您提供一种良好的方法。

这是我们将看到的第一种语言建模技术，它本质上是在给定的话语上滑动一个上下文窗口（上下文窗口是一个 N-gram 模型），并试图根据窗口周围的单词来猜测中间的单词。例如，假设你的窗口长度为 5，你的句子是“学习语言学让我很高兴。”你会给 CBoW `['learning',` `'about',` `'makes',` `'me']`，试图让模型根据模型之前在类似位置看到该单词出现的次数来猜测“linguistics”。这个例子说明了为什么对于像这样训练的模型来说生成是困难的。比如说你给模型输入 `['makes',` `'me',` `'</s>

在列表 2.4 中，我们创建了我们的第一个连续模型。在我们的情况下，为了尽可能保持简单，我们使用 BoW 模型进行语言处理，并使用一个层和两个参数的神经网络进行嵌入估计，尽管两者都可以替换为任何其他模型。例如，你可以用 N-gram 替换 BoW，用朴素贝叶斯模型替换神经网络，以获得连续的朴素 N-gram 模型。重点是，在这个技术中实际使用的模型有点任意；重要的是连续技术。为了进一步说明这一点，我们除了`numpy`之外不使用任何其他包来为神经网络进行数学运算，尽管它是本节中第一个出现的。 

仔细注意以下步骤——初始化模型权重、修正线性单元（ReLU）激活函数、最终的 softmax 层以及前向和反向传播——以及它们如何在`gradient_descent`函数中相互配合。这些是拼图中反复出现的部分，无论编程语言或框架如何。你将需要在 TensorFlow、PyTorch、Hugging Face 中初始化模型、选择激活函数、选择最终层以及定义前向和反向传播，如果你开始创建自己的模型而不是使用别人的。

##### 列表 2.4 生成式 CBoW 语言模型实现

```py
import nltk
import numpy as np
from utils import get_batches, compute_pca, get_dict
import re
from matplotlib import pyplot

with open("hamlet.txt", "r", encoding="utf-8") as f:
    data = f.read()                              #1

data = re.sub(r"[,!?;-]", ".", data)         #2
data = nltk.word_tokenize(data)
data = [ch.lower() for ch in data if ch.isalpha() or ch == "."]
print("Number of tokens:", len(data), "\n", data[500:515])

fdist = nltk.FreqDist(word for word in data)     #3
print("Size of vocabulary:", len(fdist))
print("Most Frequent Tokens:", fdist.most_common(20))

word2Ind, Ind2word = get_dict(data)     #4
V = len(word2Ind)
print("Size of vocabulary:", V)

print("Index of the word 'king':", word2Ind["king"])
print("Word which has index 2743:", Ind2word[2743])

def initialize_model(N, V, random_seed=1):     #5
    """
    Inputs:
        N: dimension of hidden vector
        V: dimension of vocabulary
        random_seed: seed for consistent results in tests
    Outputs:
        W1, W2, b1, b2: initialized weights and biases
    """
    np.random.seed(random_seed)

    W1 = np.random.rand(N, V)
    W2 = np.random.rand(V, N)
    b1 = np.random.rand(N, 1)
    b2 = np.random.rand(V, 1)

    return W1, W2, b1, b2

def softmax(z):      #6
    """
    Inputs:
        z: output scores from the hidden layer
    Outputs:
        yhat: prediction (estimate of y)
    """
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0)
    return yhat

def forward_prop(x, W1, W2, b1, b2):      #7
    """
    Inputs:
        x: average one-hot vector for the context
        W1,W2,b1,b2: weights and biases to be learned
    Outputs:
        z: output score vector
    """
    h = W1 @ x + b1
    h = np.maximum(0, h)
    z = W2 @ h + b2
    return z, h

def compute_cost(y, yhat, batch_size):     #8
    logprobs = np.multiply(np.log(yhat), y) + np.multiply(
        np.log(1 - yhat), 1 - y
    )
    cost = -1 / batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):    #9
    """
    Inputs:
        x:  average one hot vector for the context
        yhat: prediction (estimate of y)
        y:  target vector
        h:  hidden vector (see eq. 1)
        W1, W2, b1, b2:  weights and biases
        batch_size: batch size
    Outputs:
        grad_W1, grad_W2, grad_b1, grad_b2:  gradients of weights and biases
    """
    l1 = np.dot(W2.T, yhat - y)
    l1 = np.maximum(0, l1)
    grad_W1 = np.dot(l1, x.T) / batch_size
    grad_W2 = np.dot(yhat - y, h.T) / batch_size
    grad_b1 = np.sum(l1, axis=1, keepdims=True) / batch_size
    grad_b2 = np.sum(yhat - y, axis=1, keepdims=True) / batch_size

    return grad_W1, grad_W2, grad_b1, grad_b2

def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):     #10
    """
    This is the gradient_descent function
        Inputs:
            data:      text
            word2Ind:  words to Indices
            N:         dimension of hidden vector
            V:         dimension of vocabulary
            num_iters: number of iterations
        Outputs:
            W1, W2, b1, b2:  updated matrices and biases

        """
        W1, W2, b1, b2 = initialize_model(N, V, random_seed=8855)
        batch_size = 128
        iters = 0
        C = 2
        for x, y in get_batches(data, word2Ind, V, C, batch_size):
            z, h = forward_prop(x, W1, W2, b1, b2)
            yhat = softmax(z)
            cost = compute_cost(y, yhat, batch_size)
            if (iters + 1) % 10 == 0:
                print(f"iters: {iters+1} cost: {cost:.6f}")
            grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(
                x, yhat, y, h, W1, W2, b1, b2, batch_size
            )
            W1 = W1 - alpha * grad_W1
            W2 = W2 - alpha * grad_W2
            b1 = b1 - alpha * grad_b1
            b2 = b2 - alpha * grad_b2
            iters += 1
            if iters == num_iters:
                break
            if iters % 100 == 0:
                alpha *= 0.66

        return W1, W2, b1, b2

C = 2      #11
N = 50
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
num_iters = 150
print("Call gradient_descent")
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)
# Call gradient descent
# Iters: 10 loss: 0.525015
# Iters: 20 loss: 0.092373
# Iters: 30 loss: 0.050474
# Iters: 40 loss: 0.034724
# Iters: 50 loss: 0.026468
# Iters: 60 loss: 0.021385
# Iters: 70 loss: 0.017941
# Iters: 80 loss: 0.015453
# Iters: 90 loss: 0.012099
# Iters: 100 loss: 0.012099
# Iters: 110 loss: 0.011253
# Iters: 120 loss: 0.010551
# Iters: 130 loss: 0.009932
# Iters: 140 loss: 0.009382
# Iters: 150 loss: 0.008889
```

#1 创建我们的训练语料库

#2 通过删除标点符号、按单词标记化并转换为小写字母字符来稍微清理数据

#3 获取我们的词袋，以及一个分布

#4 创建两个字典以加快转换时间和跟踪词汇

#5 在这里，我们使用一个层和两个参数创建我们的神经网络。

#6 创建我们的最终分类层，使得所有可能性加起来等于 1

#7 定义了通过我们的模型向前移动的行为，以及一个激活函数

#8 定义了如何确定真实值和模型预测之间的距离

#9 定义了如何通过模型向后移动并收集梯度

#10 将所有内容组合在一起并进行训练

#11 训练模型

CBoW 示例是我们第一个展示机器学习中完整且有效训练循环的代码示例。在这个过程中，请特别注意训练循环中的步骤，特别是激活函数 ReLU。由于我们预期你对各种 ML 范式至少是熟悉的，包括不同的激活函数，我们在这里不会解释 ReLU。我们将讨论何时应该使用它以及何时不应该使用它。ReLU 虽然解决了梯度消失问题，但并没有解决梯度爆炸问题，并且会破坏模型中所有的负比较。更好的情境变体包括指数线性单元（ELU），它允许负数归一化到 alpha，以及广义高斯线性单元（GEGLU）/Swish 门控线性单元（SWIGLU），它们在越来越复杂的场景中表现良好，如语言。然而，人们经常使用 ReLU，并不是因为它们在某个情境下是最好的，而是因为它们易于理解、编码和直观，甚至比它们所取代的激活函数（如 sigmoid 或 tanh）更直观。

许多内容最终都会通过包和其他类似工具进行抽象化，但了解底层发生的事情对于将 LLMs 投入生产的你来说将非常有帮助。你应该能够以某种确定性预测不同模型在不同情况下的行为。下一节将深入探讨这些抽象化之一——在这种情况下，是连续建模技术创建的抽象化。

### 2.2.5 嵌入

回顾到我们关于语言的特征，应该很容易理解为什么连续风格的文本建模是一个突破。嵌入将我们创建的、不包含任何意义的标记化向量，尝试根据对文本的观察插入意义，例如词序和出现在相似上下文中的子词。尽管主要的意义模式是搭配（共定位，相邻出现的词），但它们证明是有用的，甚至显示出与人类编码的词义的一些相似性。

Word2Vec 中的一个经典示例，也是最早的预训练向量嵌入之一，就是取“king”（国王）的向量，减去“man”（男人）的向量，加上“woman”（女人）的向量，然后找到与这个和最近的邻居向量，就是“queen”（王后）这个词的向量。这在我们看来是有道理的，因为它模仿了人类的语义。其中一个已经被提到几次的主要区别是：语用学。人类使用语用上下文来告知语义意义，理解到你说“我需要食物”并不意味着你实际上没有食物就会处于身体危险之中。嵌入没有受到任何纯使用之外的影响，这感觉就像人类学习的方式一样，而且在这里有很好的论据。唯一确定的是，如果我们能以某种方式给模型提供更具代表性的数据，这可能为更有效的嵌入打开大门，但这是一个鸡生蛋、蛋生鸡的问题，因为更有效的嵌入会带来更好的模型性能。

在列出 2.5 时，我们深入探讨了如何使用`pyplot`可视化嵌入。我们将在后面的章节中更深入地探讨嵌入。这对于模型可解释性和在预训练步骤中的验证都很有帮助。如果你看到你的语义相似的嵌入在图上相对较近，你很可能是在正确的方向上。

##### 列出 2.5：嵌入可视化

```py
words = [          #1
    "King",
    "Queen",
    "Lord",
    "Man",
    "Woman",
    "Prince",
    "Ophelia",
    "Rich",
    "Happy",
]              
embs = (W1.T + W2) / 2.0
idx = [word2Ind[word] for word in words]
X = embs[idx, :]
print(X.shape, idx)

result = compute_pca(X, 2)
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
```

#1 列出 2.4 完成后，并执行了梯度下降

如图 2.4 所示，这是我们从 CBoW 模型中训练出的一个成功但非常稀疏的嵌入表示。使这些语义表示（嵌入）更密集是我们在这个领域可以看到的主要改进之处，尽管已经进行了许多实验，其中更密集的语义意义被通过指导和不同的思维链技术用更大的语用上下文所取代。我们将在后面讨论思维链（CoT）和其他技术。现在，让我们转向讨论为什么我们的连续嵌入技术甚至可以成功，鉴于基于频率的模型通常很难与现实相关联。所有这一切都始于半个多世纪前的 MLP。

![figure](img/2-4.png)

##### 图 2.4：词嵌入的可视化技术。可视化嵌入对于模型可解释性很重要。

### 2.2.6  多层感知器

MLPs 是“机器擅长做一件事，所以希望我们能够只用一些擅长这件事的机器来做出擅长很多事情的机器”这一观点的体现。MLP 神经网络中的每一个权重和偏差都擅长做一件事，这可能是检测一个或多个特征。因此，我们将它们绑定在一起以检测更大、更复杂的特征。MLPs 是大多数神经网络架构中的主要构建块。架构之间的关键区别，如卷积神经网络和循环神经网络，主要源于数据加载方法以及处理在模型层中流动的标记化和嵌入数据，而不是单个层的功能，尤其是全连接层。

列表 2.6 提供了一个更动态的神经网络类，它可以拥有你任务所需的所有层和参数。我们使用 PyTorch 提供了一个更定义明确和显式的类，以便你能够以任何你想要的方式实现 MLP，无论是从头开始还是在一个流行的框架中。

##### 列表 2.6 多层感知器 PyTorch 类实现

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=2,
        output_size=3,
        num_hidden_layers=1,
        hidden_activation=nn.Sigmoid,
    ):
        """Initialize weights.
        Args:
            input_size (int): size of the input
            hidden_size (int): size of the hidden layers
            output_size (int): size of the output
            num_hidden_layers (int): number of hidden layers
            hidden_activation (torch.nn.*): the activation class
        """
        super(MultiLayerPerceptron, self).__init__()
        self.module_list = nn.ModuleList()
        interim_input_size = input_size
        interim_output_size = hidden_size
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for _ in range(num_hidden_layers):
            self.module_list.append(
                nn.Linear(interim_input_size, interim_output_size)
            )
            self.module_list.append(hidden_activation())
            interim_input_size = interim_output_size

        self.fc_final = nn.Linear(interim_input_size, output_size)

        self.last_forward_cache = []

    def forward(self, x, apply_softmax=False):
        """The forward pass of the MLP

        Args:
            x_in (torch.Tensor): an input data tensor.
            x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        for module in self.module_list:
            x = module(x)

        output = self.fc_final(x)

        if apply_softmax:
            output = F.softmax(output, dim=1)

        return output
```

从这段代码中，我们可以看出，与具有两个静态层的 CBoW 实现不同，这个 MLP 在实例化之前并不是静态的尺寸。如果你想给这个模型一百万层，你必须在实例化类时将`num_hidden_layers=1000000`。然而，仅仅因为给模型那么多参数并不意味着它会立即变得更好。LLMs 不仅仅是很多层。就像 RNNs 和 CNNs 一样，LLMs 的魔力在于数据如何进入模型并在其中移动。为了说明这一点，让我们看看 RNN 及其变体。

### 2.2.7 循环神经网络和长短期记忆网络

RNNs 是一类神经网络，旨在根据先前语言建模技术的弱点来分析序列。可以将序列视为一个有序数组，其中整个数组的总和在任何一个部分被移动时都会改变值。逻辑是这样的：如果语言以序列的形式呈现，那么也许它应该以序列的形式进行处理，而不是一次处理一个标记。RNNs 通过使用我们在 MLPs 和马尔可夫链中看到过的逻辑来实现这一点，即当处理新的输入时，会引用内部状态或记忆，并在检测到节点之间的连接有用时创建循环。

在完全循环网络中，如列表 2.7 中的那种，所有节点最初都连接到所有后续节点，但这些连接可以设置为零，以模拟它们在不适用时断开。这解决了早期模型所面临的最大问题之一，即静态输入大小，并使 RNN 及其变体能够处理可变长度的输入。不幸的是，较长的序列会带来新的问题。因为网络中的每个神经元都连接到后续的神经元，较长的序列会导致整体总和的变化更小，使得梯度更小，直到最终消失，即使对于重要的单词；这被称为梯度消失。还存在其他问题，如梯度爆炸和梯度减小。

例如，让我们考虑以下用于情感分析的句子：“我昨晚非常喜欢这部电影”和“我昨晚去看的电影是我曾经期望看到的最棒的一部。”这些句子在语义上可以被认为是相似的，即使它们并不完全相同。当通过 RNN 移动时，第一句中的每个单词都更有价值，结果是第一句的正面评分比第二句高，仅仅因为第一句更短。反之亦然：梯度爆炸是这种序列处理的结果，这使得训练深层 RNN 变得困难。

为了解决这个问题，LSTMs（一种 RNN），通过使用记忆单元和门控机制来处理可变长度的序列，但避免了理解长序列和短序列不同的难题。考虑到多语言场景，并理解人们不会只从单一方向思考语言，LSTMs 还可以通过连接两个 RNN 的输出（一个从左到右读取序列，另一个从右到左读取）来双向处理序列。这种双向性提高了结果，即使在成千上万的标记之后，信息也能被看到并记住。

在列表 2.7 中，我们给出了 RNN 和 LSTM 的类。在与此书相关的代码库（[`github.com/IMJONEZZ/LLMs-in-Production`](https://github.com/IMJONEZZ/LLMs-in-Production)）中，你可以看到训练 RNN 和 LSTM 的结果。关键点是，LSTM 在训练和验证集上都能在更少的 epoch（25 个与 RNN 的 50 个相比）内达到更好的准确率。值得注意的一项创新是，打包嵌入利用填充将所有可变长度的序列扩展到最大长度。因此，只要输入长度小于最大长度，LSTMs 就可以处理任何长度的输入。为了有效地设置 LSTM，我们将在数据集（一个 Twitter 情感分析数据集）上做一些经典的 NLP 工作。该工作流程将使用自然语言工具包正则表达式进行标记化。它只查找单词，其他什么也不找，然后传递给 spacy 词形还原器，以获得只包含单词基本非屈折形式的列表。

##### 列表 2.7 RNN 和 LSTM PyTorch 类实现。

```py
import torch
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import nltk
import spacy

    tokenizer = nltk.tokenize.RegexpTokenizer("\w+'?\w+|\w+'")
    tokenizer.tokenize("This is a test")
    stop_words = nltk.corpus.stopwords.words("english")
    nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger", "ner"])

dataset = pd.read_csv("./data/twitter.csv")     #1

text_data = list(
    map(lambda x: tokenizer.tokenize(x.lower()), dataset["text"])
)
text_data = [
    [token.lemma_ for word in text for token in nlp(word)]
    for text in text_data
]
label_data = list(map(lambda x: x, dataset["feeling"]))
assert len(text_data) == len(
    label_data
), f"{len(text_data)} does not equal {len(label_data)}"

EMBEDDING_DIM = 100
model = Word2Vec(
    text_data, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4
)
word_vectors = model.wv
print(f"Vocabulary Length: {len(model.wv)}")
del model

padding_value = len(word_vectors.index_to_key)
    embedding_weights = torch.Tensor(word_vectors.vectors)    #2

class RNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        embedding_weights,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embedding_weights
    )
    self.rnn = torch.nn.RNN(embedding_dim, hidden_dim)
    self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, text_lengths):
        embedded = self.embedding(x)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths
        )
        packed_output, hidden = self.rnn(packed_embedded)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output
        )
        return self.fc(hidden.squeeze(0))

INPUT_DIM = padding_value
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

rnn_model = RNN(
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, embedding_weights
)

rnn_optimizer = torch.optim.SGD(rnn_model.parameters(), lr=1e-3)
rnn_criterion = torch.nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(torch.nn.Module):
    def __init__(
    self,
    input_dim,
    embedding_dim,
    hidden_dim,
    output_dim,
    n_layers,
    bidirectional,
        dropout,
        embedding_weights,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embedding_weights
        )
        self.rnn = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, text_lengths):
        embedded = self.embedding(x)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths
        )
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        )
        return self.fc(hidden.squeeze(0))

INPUT_DIM = padding_value
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

lstm_model = LSTM(
    INPUT_DIM,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    N_LAYERS,
    BIDIRECTIONAL,
    DROPOUT,
    embedding_weights,
)

lstm_optimizer = torch.optim.Adam(lstm_model.parameters())
lstm_criterion = torch.nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch["text"], batch["length"]).squeeze(1)
        loss = criterion(predictions, batch["label"])
        acc = binary_accuracy(predictions, batch["label"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch["text"], batch["length"]).squeeze(1)
            loss = criterion(predictions, batch["label"])
            acc = binary_accuracy(predictions, batch["label"])

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

batch_size = 2     #3

def iterator(X, y):
    size = len(X)
    permutation = np.random.permutation(size)
    iterate = []
    for i in range(0, size, batch_size):
    indices = permutation[i : i + batch_size]
        batch = {}
        batch["text"] = [X[i] for i in indices]
        batch["label"] = [y[i] for i in indices]

        batch["text"], batch["label"] = zip(
            *sorted(
                zip(batch["text"], batch["label"]),
                key=lambda x: len(x[0]),
                reverse=True,
            )
        )
        batch["length"] = [len(utt) for utt in batch["text"]]
        batch["length"] = torch.IntTensor(batch["length"])
        batch["text"] = torch.nn.utils.rnn.pad_sequence(
            batch["text"], batch_first=True
        ).t()
        batch["label"] = torch.Tensor(batch["label"])

        batch["label"] = batch["label"].to(device)
        batch["length"] = batch["length"].to(device)
        batch["text"] = batch["text"].to(device)

        iterate.append(batch)

    return iterate

index_utt = [
    torch.tensor([word_vectors.key_to_index.get(word, 0) for word in text])
    for text in text_data
]

X_train, X_test, y_train, y_test = train_test_split(
    index_utt, label_data, test_size=0.2
)                                                    #4
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2
)

train_iterator = iterator(X_train, y_train)
validate_iterator = iterator(X_val, y_val)
test_iterator = iterator(X_test, y_test)

print(len(train_iterator), len(validate_iterator), len(test_iterator))

N_EPOCHS = 25

for model in [rnn_model, lstm_model]:
    print(
    "|-----------------------------------------------------------------------------------------|"
    )
    print(f"Training with {model.__class__.__name__}")
    if "RNN" in model.__class__.__name__:
        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train(
                rnn_model, train_iterator, rnn_optimizer, rnn_criterion
            )
            valid_loss, valid_acc = evaluate(
                rnn_model, validate_iterator, rnn_criterion
            )

            print(
                f"| Epoch: {epoch+1:02} | Train Loss: {train_loss: .3f} | 
                ↪ Train Acc: {train_acc*100: .2f}% | Validation Loss: 
                ↪ {valid_loss: .3f} | Validation Acc: {valid_acc*100: .2f}% |"
            )
    else:
        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train(
                lstm_model, train_iterator, lstm_optimizer, lstm_criterion
            )
            valid_loss, valid_acc = evaluate(
                lstm_model, validate_iterator, lstm_criterion
            )

            print(
                f"| Epoch: {epoch+1:02} | Train Loss: {train_loss: .3f} | 
                ↪ Train Acc: {train_acc*100: .2f}% | Validation Loss: 
                ↪ {valid_loss: .3f} | Validation Acc: {valid_acc*100: .2f}% |"
            )
# Training on our dataset
# | Epoch: 01 | Train Loss:  0.560 | Train Acc:  70.63% | Validation Loss:
# 0.574 | Validation Acc:  70.88% |
# | Epoch: 05 | Train Loss:  0.391 | Train Acc:  82.81% | Validation Loss:
# 0.368 | Validation Acc:  83.08% |
# | Epoch: 10 | Train Loss:  0.270 | Train Acc:  89.11% | Validation Loss:
# 0.315 | Validation Acc:  86.22% |
# | Epoch: 15 | Train Loss:  0.186 | Train Acc:  92.95% | Validation Loss:
# 0.381 | Validation Acc:  87.49% |
# | Epoch: 20 | Train Loss:  0.121 | Train Acc:  95.93% | Validation Loss:
# 0.444 | Validation Acc:  86.29% |
# | Epoch: 25 | Train Loss:  0.100 | Train Acc:  96.28% | Validation Loss:
# 0.451 | Validation Acc:  86.83% |
```

#1 创建我们的训练语料库并执行一些经典的 NLP 预处理。

#2 需要嵌入来为 LSTM 的输入赋予语义价值。

#3 通常应该是 2 的幂，因为这是计算机内存中最容易处理的形式。

#4 你必须为你在训练的东西确定一些标签。

看看我们的类和实例化，你应该会看到 LSTM 与 RNN 并没有太大区别。唯一的区别是`init`输入变量是`n_layers`（为了方便，你也可以用 RNN 指定它），`bidirectional`和`dropout`。`bidirectional`允许 LSTM 在序列中向前看，以帮助理解意义和上下文。它还在多语言场景中帮助很大，因为像英语这样的从左到右的语言并不是正文的唯一格式。`dropout`是另一个巨大的创新，它改变了过拟合的模式，从数据依赖转变为帮助模型不过拟合，通过在训练过程中逐层关闭随机节点来强制所有节点不相互关联，并防止复杂的共适应。模型外的参数唯一的不同之处在于，用于 RNN 的优化器是随机梯度下降（SGD），就像我们的 CBoW 一样；LSTM 使用 Adam（尽管两者都可以使用任何，取决于性能，包括 AdamW）。接下来，我们定义我们的训练循环并训练 LSTM。将这个训练循环与`gradient_descent`函数中定义的列表 2.4 进行比较。

在这里展示的代码中，令人惊叹的一点是，与之前的模型迭代相比，LSTM 可以学得更快，这得益于`bidirectionality`和`dropout`。尽管之前的模型训练速度比 LSTM 快，但它们需要数百个 epoch 才能达到 LSTM 在 25 个 epoch 内达到的性能。正如其名所示，验证集上的性能增加了架构的有效性，在它未训练的示例上进行推理，并保持准确度与训练集相当。

这些模型的问题并不那么明显，主要表现为资源消耗极其巨大，尤其是在应用于像医疗和法律这样更长、更注重细节的问题时。尽管`dropout`和`bidirectional`处理具有令人难以置信的优势，但它们至少将训练所需的处理能力翻倍。因此，虽然推理最终只比相同大小的 MLP 贵 2 到 3 倍，但训练成本却高达 10 到 12 倍。也就是说，`dropout`和`bidirectional`很好地解决了梯度爆炸问题，但同时也增加了训练所需的计算量。为了解决这个问题，设计并实施了一种捷径，允许任何模型，包括 LSTM，找出序列中哪些部分是最有影响力的，哪些部分可以安全忽略，这被称为*注意力*。

### 2.2.8 注意力

注意力是一种数学捷径，它通过一个突现的数学公式告诉模型考虑输入的哪些部分以及多少，从而为模型提供一个机制，以更快的速度解决更大的上下文窗口。注意力基于字典的升级版本，其中不仅包含键值对，还添加了上下文查询。简单来说，以下代码是较老的自然语言处理技术与更现代技术之间的主要区别。

注意力解决了训练 LSTMs 的缓慢问题，同时保持了在少量 epoch 上的高性能。还有多种类型的注意力。点积注意力方法捕捉了查询中的每个词（或嵌入）与键中的每个词之间的关系。当查询和键是同一句子的部分时，这被称为*双向自注意力*。然而，在某些情况下，只关注当前词之前的内容可能更合适。这种类型的注意力，尤其是当查询和键来自同一句子时，被称为*因果注意力*。通过掩码序列的部分并迫使模型猜测掩码后面的内容，语言模型进一步得到改进。以下列表中的函数展示了点积注意力和掩码注意力。

##### 列表 2.8 多头注意力实现

```py
import numpy as np
from scipy.special import softmax

x = np.array([[1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
            [1.0, 1.0, 1.0, 1.0]])     #1

w_query = np.array([1,0,1],
             [1,0,0],
             [0,0,1],
             [0,1,1]])          #2
w_key = np.array([[0,0,1],
             [1,1,0],
             [0,1,0],
             [1,1,0]])         #2
w_value = np.array([[0,2,0], 
             [0,3,0],
             [1,0,3],
             [1,1,0]])         #2

Q = np.matmul(x,w_query)     #3
K = np.matmul(x,w_key)       #3
V = np.matmul(x,w_value)     #3

k_d = 1                                        #4
attention_scores = (Q @ K.transpose())/k_d     #4

attention_scores[0] = softmax(attention_scores[0])     #5
attention_scores[1] = softmax(attention_scores[1])     #5
attention_scores[2] = softmax(attention_scores[2])     #5

attention1 = attention_scores[0].reshape(-1,1)     #6
attention1 = attention_scores[0][0]*V[0]           #6
attention2 = attention_scores[0][1]*V[1]           #6
attention3 = attention_scores[0][2]*V[2]           #6

attention_input1 = attention1 + attention2 + attention3    #7

attention_head1 = np.random.random((3,64))      #8

z0h1 = np.random.random((3,64))     #9
z1h2 = np.random.random((3,64))     #9
z2h3 = np.random.random((3,64))     #9
z3h4 = np.random.random((3,64))     #9
z4h5 = np.random.random((3,64))     #9
z5h6 = np.random.random((3,64))     #9
z6h7 = np.random.random((3,64))     #9
z7h8 = np.random.random((3,64))     #9

Output_attention = np.hstack((z0h1,z1h2,z2h3,z3h4,z4h5,z5h6,z6h7,z7h8))    #10

def dot_product_attention(query, key, value, mask, scale=True):    #11
    assert query.shape[-1] == key.shape[-1] == value.shape[-1], "q,k,v have different dimensions!"
    if scale:
        depth = query.shape[-1]
    else:
        depth = 1
    dots = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(depth)
    if mask is not None:
        dots = np.where(mask, dots, np.full_like(dots, -1e9))
    logsumexp = scipy.special.logsumexp(dots, axis=-1, keepdims=True)
    dots = np.exp(dots - logsumexp)
    attention = np.matmul(dots, value)
    return attention
def masked_dot_product_self_attention(q,k,v,scale=True):     #12
    mask_size = q.shape[-2]
    mask = np.tril(np.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)
    return DotProductAttention(q,k,v,mask,scale=scale)
```

#1 第 1 步：输入：三个输入，d_model=4

#2 第 2 步：权重三个维度 x d_model=4

#3 第 3 步：矩阵乘法以获得 Q，K，V；查询：x * w_query；键：x * w_key；值：x * w_value

#4 第 4 步：缩放后的注意力分数；维度的平方根

#5 第 5 步：每个向量的缩放 softmax 注意力分数

#6 第 6 步：通过 score1/k_d * V 获得的注意力值

#7 第 7 步：将结果相加以创建输出矩阵的第一行

#8 第 8 步：对输入 1 到 3 执行步骤 1 到 7；因为这只是个演示，我们将使用正确维度的随机矩阵。

#9 第 9 步：我们使用步骤 1 到 7 训练注意力子层的所有八个头。

#10 第 10 步：将头 1 到 8 连接起来，以获得模型原始的 8 × 64 输出维度。

#11 此函数执行所有这些步骤。

#12 此函数执行前面的步骤，但增加了掩码中的因果关系。

在注意力的完整实现中，你可能注意到了一些你熟悉的术语——即`键`和`值`，但你可能之前没有接触过`查询`。`键`和`值`对之所以熟悉，是因为字典和查找表，其中我们将一组键映射到一个值数组。`查询`应该感觉直观，就像是一种检索的搜索。`查询`与从正常操作中检索`值`的`键`进行比较。

在注意力机制中，`Query`和`Key`通过点积相似度比较来获得注意力分数，随后这个分数会被乘以`Value`以得到模型应该关注序列该部分的最终分数。这可能会根据你的模型架构变得更加复杂，因为必须考虑到编码器和解码器的序列长度，但在此我们可以简单地说，在这个空间中建模的最有效方法是投影所有输入源到一个公共空间，并使用点积进行高效比较。

这段代码解释比之前的例子数学性更强，但这是为了说明概念。注意力的数学原理真正具有创新性，并推动了该领域的发展。不幸的是，即使注意力为序列建模过程带来了优势，在 LSTMs 和 RNNs 中，仍然存在速度和内存大小的问题。你可能从代码和数学中注意到取了平方根，这意味着我们使用的注意力是二次的。包括像 Hyena 和循环记忆转换器（RMT，基本上是一个结合了转换器的 RNN）这样的亚二次技术在内的各种技术已经被开发出来以解决这些问题，我们将在稍后更详细地介绍。现在，让我们继续探讨注意力的最终应用：转换器。

## 2.3 注意力即一切

在开创性的论文“Attention Is All You Need”中¹，Vaswani 等人将数学捷径推进了几步，提出为了性能，根本不需要任何循环（RNN 中的“R”）或任何卷积。

备注：我们不讨论卷积，因为它们对 NLP 来说并不好，但在计算机视觉中却很受欢迎。

相反，Vaswani 等人选择只使用注意力，并更加仔细地指定 Q、K 和 V 的来源。我们将在稍后深入探讨这一点。在我们对这一系列多样化的 NLP 技术的回顾中，我们观察到了它们随时间的发展以及每种方法如何寻求改进其前辈。从基于规则的方法到统计模型和神经网络，该领域一直在努力寻找更高效、更准确处理和理解自然语言的方法。

现在，我们将注意力转向一项具有革命性的创新，它彻底改变了 NLP 领域：转换器架构。在下一节中，我们将探讨支撑转换器的关键概念和机制，以及它们如何使得开发超越先前技术的最先进语言模型成为可能。我们还将讨论转换器对更广泛的 NLP 领域的影响，并考虑在这一激动人心的研究领域中进一步进步的潜力。

### 2.3.1 编码器

编码器是完整变压器模型的前半部分，在分类和特征工程领域表现出色。Vaswani 等人发现，在编码器内部的嵌入层之后，对张量进行的任何额外转换都可能损害它们进行“语义”比较的能力，而这正是嵌入层的目的。这些模型严重依赖于自注意力和巧妙的位置编码来操纵这些向量，而不会显著降低所表达的相似度。

再次强调，嵌入的一个关键特性是它们是数据的向量表示——在我们的案例中，是标记。标记是你选择的代表语言的任何东西。我们建议按照一般规则使用子词，但你将逐渐了解哪些位置和哪些类型的标记效果良好。考虑以下句子：“The cat in the hat rapidly leapt above the red fox and the brown unmotivated dog。” “Red” 和 “brown” 在语义上是相似的，并且在嵌入层之后有类似的表示。然而，它们在句子中的位置分别是第 10 和第 14 位，假设我们是按单词进行分词。因此，位置编码在它们之间引入了距离，同时也增加了区分句子中不同位置相同标记的能力。然而，一旦应用正弦和余弦函数，它们的意义就只比编码后稍微远一点，并且这种编码机制在递归和更多数据的情况下表现出色。为了说明这一点，让我们假设在嵌入后 [red] 和 [brown] 之间有 99% 的余弦相似度。编码将大大将其降低到大约 85% 到 86% 的相似度。按照描述应用正弦和余弦方法将它们的相似度恢复到大约 96%。

BERT 是在 Vaswani 等人原始论文之后出现的首批架构之一，并且是仅使用编码器的变压器的一个例子。鉴于其体积之小，BERT 是一个极其强大的模型架构，至今仍在生产系统中使用。BERT 是第一个仅使用编码器的变压器，其受欢迎程度激增，展示了使用变压器进行连续或顺序（它们是相同的）建模，其嵌入效果比 Word2Vec 更好。我们可以看到，这些嵌入效果更好，因为它们可以非常快速地应用于新任务和数据，并且只需要最少的训练，就能得到比 Word2Vec 嵌入更符合人类偏好的结果。在一段时间内，大多数人使用基于 BERT 的模型在较小的数据集上进行少样本学习任务。BERT 使大多数研究人员和企业能够以最小的努力获得最先进的性能。

![figure](img/2-5.png)

##### 图 2.5 一个编码器的可视化。编码器是完整传输器架构的前半部分，在自然语言理解任务（如分类或命名实体识别）中表现出色。编码器模型通过不需要先验或递归，并使用巧妙的位置编码和多头注意力来创建每个标记的向量嵌入，从而改进了先前设计。

编码器的优势（如图 2.5 所示）包括以下内容：

+   展示理解的分类和层次任务

+   考虑到长距离依赖建模，速度极快

+   建立在已知模型之上，如嵌入中的 CBoW，前馈中的 MLP 等。

+   并行

编码器的弱点包括以下内容：

+   如建议所示，需要大量数据（尽管少于 RNNs）才能有效

+   更加复杂的架构

### 2.3.2 解码器

解码器模型，如图 2.6 所示，是编码器的大版本，其基本形式包含两个多头注意力块和三个求和归一化层。它们是编码器之后的传输器的一半。解码器在掩码语言建模和学习以及快速应用语法方面非常出色，几乎立即产生了只需要解码器模型才能实现人工通用智能的想法。编码器与解码器任务的一个有用简化是，编码器在自然语言理解（NLU）任务中表现出色，而解码器在自然语言生成（NLG）任务中表现出色。解码器仅传输器架构的例子是生成预训练传输器（GPT）模型系列。这些模型遵循转换生成语法的逻辑，完全基于语法，允许无限生成一种语言中所有可能的句子（见附录 A）。

![figure](img/2-6.png)

##### 图 2.6 一个解码器的可视化。解码器是完整传输器的后半部分，在 NLG 任务（如聊天机器人和讲故事）中表现出色。解码器与编码器一样改进了先前架构，但它们将输出向右移动一个空格以帮助下一词生成，从而利用多头自注意力的优势。

解码器的优势包括以下内容：

+   在序列中生成下一个标记（向右移动表示考虑已生成的标记）

+   建立在已知模型和编码器之上

+   生成过程中可以流式传输，提供良好的用户体验

它们的弱点包括以下内容：

+   仅语法模型往往难以插入预期的或意图的含义（参见所有“我强迫一个 AI 观看 1000 小时的 x 并生成”的 2018-至今的 meme）。

+   幻觉。

### 2.3.3 传输器

完整的 Transformer 架构同时利用了编码器和解码器，在输出之前，将编码器的理解传递给解码器的第二个多头注意力块。由于 Transformer 的每一部分在理解或生成方面都有其专长，因此整个产品在需要生成之前有一定理解条件的条件生成任务，如翻译或摘要，上表现得最好，这一点应该是直观的。编码器侧重于在高级别处理输入，而解码器则更专注于生成连贯的输出。完整的 Transformer 架构能够成功理解数据，然后根据这种理解生成输出，如图 2.7 所示。文本到文本迁移 Transformer（T5）模型系列是 Transformer 的一个例子。

![figure](img/2-7.png)

##### 图 2.7 一个完整的 Transformer 可视化。完整的 Transformer 结合了编码器和解码器，在所有任务上都做得很好，包括摘要和翻译等条件生成任务。由于 Transformer 比其每个部分都要庞大和缓慢，研究人员和企业通常选择使用这些部分而不是整个 Transformer。

备注：Transformer 模型的优势在于它们围绕输入的并行化构建，这增加了 LSTM 目前无法复制的速度。如果 LSTM 达到可以像 Transformer 一样快速运行的程度，它们可能在最先进的领域中具有竞争力。

Transformer 的优势如下：

+   包含编码器和解码器，因此它在它们擅长的所有事情上都做得很好

+   高度并行化以提高速度和效率

弱点包括以下：

+   记忆密集型，但仍然小于相同大小的 LSTM

+   训练需要大量的数据和 VRAM

如您可能已经注意到的，我们讨论的大多数模型都不是语言导向的，它们高度关注语法，即使它们试图模拟真实语言。模型，即使是最先进的 Transformer，也只有语义近似——没有语用学，没有语音学——并且在标记化过程中，只有在没有上下文的情况下才真正利用形态学的数学模型。这并不意味着模型不能学习这些，也不意味着例如，Transformer 不能将音频作为输入；这仅仅意味着平均使用情况不是这样。考虑到这一点，它们能像现在这样工作，简直是个奇迹，它们真正应该得到赞赏。

到目前为止，我们试图强调模型当前的局限性，并在本书的剩余部分深入探讨如何改进它们。其中一条途径已经取得并且仍在取得巨大成功的探索：迁移学习和微调大型基础模型。这种技术是在 BERT 最初发布后不久出现的。研究人员发现，尽管 BERT 在大量任务上表现良好，但如果他们希望它在特定任务或数据领域上表现更好，他们只需在代表该任务或领域的但不是从头开始的数据上重新训练模型即可。鉴于 BERT 在创建语义近似嵌入时在更大的数据集上学习到的所有预训练权重，所需的训练数据量显著减少，就可以在所需的部分上获得最先进的性能。我们已经在 BERT 和 GPT 系列模型中看到了这一点，现在我们再次看到这一点来解决我们讨论的 exactly the challenges：语义近似覆盖、领域专业知识和数据可用性。

## 2.4 非常大的转换器

进入大型语言模型（LLMs）。自从它们被引入以来，基于转换器的模型一直在不断增大，不仅是在规模和参数数量上，还包括它们的训练数据集和训练周期的大小和长度。如果你在 2010 年代学习机器学习或深度学习，你很可能听说过这样的说法：“增加更多层并不会使模型变得更好。”LLMs 既证明了这一点是错误的，也证明了这一点是正确的——错误是因为它们的性能无与伦比，通常与经过精心微调的较小模型相匹配，这些模型在特定领域和数据集上进行了训练，甚至是在专有数据上训练的，正确是因为训练和部署 LLMs 所带来的挑战。

LLMs 与语言模型之间的一大区别在于迁移学习和微调。与之前的语言模型一样，LLMs 在庞大的文本语料库上进行预训练，使它们能够学习通用的语言特征和表示，这些特征和表示可以针对特定任务进行微调。由于 LLMs 规模庞大，其训练数据集也很大，因此它们能够在更少的标记数据的情况下实现更好的性能，这是早期语言模型的一个重大限制。通常，你只需用十几个示例就可以微调一个 LLM 来执行高度专业化的任务。

然而，使大型语言模型（LLMs）如此强大并为其在商业领域的广泛应用打开了大门的是它们能够通过简单的提示执行专门任务，而无需任何微调的能力。只需在查询中给出你想要的一些示例，LLM 就能生成结果。在较小的标记数据集上训练 LLM 被称为少样本提示。当只给出一个示例时，被称为单样本提示，而当任务是全新的时，则称为零样本提示。LLMs，尤其是那些通过人类反馈和提示工程方法进行强化学习训练的 LLMs，能够进行少样本学习，这意味着它们能够通过仅几个示例进行泛化和解决任务，达到全新的水平。这种能力是相对于早期模型的一个重大进步，早期模型需要针对每个特定任务进行大量的微调或大量的标记数据。

之前，语言模型在少样本和零样本学习领域已经显示出潜力，而 LLMs 已经证明这种潜力是真实的。随着模型规模的增大，我们发现它们能够完成小模型无法完成的任务。我们称之为*涌现行为*。² 图 2.8 展示了八个先前语言模型在随机情况下表现不佳的任务，然后一旦模型足够大，它们就能完成这些任务。

![图](img/2-8.png)

##### 图 2.8 LLMs 在模型规模达到一定大小后，执行少样本提示任务时展现出的涌现行为示例

由于 LLMs 具有庞大的参数规模，它们在零样本能力方面也表现出显著的优势，这是它们在商业世界中受欢迎和可行的最主要原因。LLMs 也由于规模大和容量大，在处理歧义方面表现出改进。它们在区分具有多个含义的单词和理解语言细微差别方面做得更好，从而产生更准确的预测和响应。这种改进并不是因为它们的能力或架构更好，因为它们与较小的 Transformer 共享相同的架构，而是因为它们拥有大量关于人们通常如何消除歧义的事例。因此，LLMs 以与数据集中通常表示的相同方式消除歧义。多亏了 LLMs 训练所依据的文本数据的多样性，它们在处理各种输入风格、噪声文本和语法错误方面表现出更高的鲁棒性。

LLMs（大型语言模型）与语言模型之间的另一个关键区别是输入空间。更大的输入空间非常重要，因为它使得少样本提示任务变得更加可行。许多 LLMs 的最大输入大小为 8,000+个 token（最初为 32K，GPT-4 自 2023 年 11 月以来已达到 128K），尽管所有之前讨论的模型也可以有如此高的输入空间，但它们通常并不具备。我们最近在这个领域看到了爆炸式增长，例如循环记忆转换器（RMT）技术允许 1M+个 token 的上下文空间，这进一步推动了 LLMs 向证明更大的模型总是更好的方向迈进。LLMs 被设计用来捕捉文本中的长距离依赖关系，这使得它们比前辈们更有效地理解上下文。这种改进的理解能力使得 LLMs 在机器翻译、摘要和对话式 AI 等任务中能够生成更连贯和上下文相关的响应。

LLMs 通过提供早期语言模型难以解决的问题的强大解决方案，已经彻底改变了 NLP（自然语言处理）。它们在上下文理解、迁移学习和少样本学习方面带来了显著的改进。随着 NLP 领域的持续发展，研究人员正在积极努力最大化 LLMs 的益处，同时减轻所有潜在的风险。因为还没有找到更好的近似语义的方法，所以他们做出了更大和更多维度的近似。因为还没有找到存储语用上下文的好方法，LLMs 通常允许直接将上下文插入到提示中，或者插入到为上下文保留的输入部分，甚至通过在推理时与 LLM 共享数据库。这种能力不会在模型内创建语用或语用系统，就像嵌入不会创建语义一样，但它允许模型正确生成模仿人类对那些语用和语义刺激做出反应的语法。语音学是 LLMs 可能取得巨大进步的地方，无论是作为完全无文本的模型，还是作为文本-语音混合模型，也许会利用国际音标（IPA）作为文本的补充或替代。现在我们正在观察这个领域的可能发展，这令人兴奋。

到目前为止，你应该已经对 LLMs（大型语言模型）有了相当好的理解，以及一些在将 LLMs 投入生产时将非常有用的语言学关键原则。现在，你应该能够开始推理哪些产品更容易或更难构建。考虑图 2.9：左下角的任务，如写作助手和聊天机器人，是 LLMs 的拿手好戏。基于提示中少量上下文的文本生成是一个严格基于语法的难题；只要模型足够大，训练数据足够多，我们就可以相对容易地做到这一点。购物助手与此类似，并且也相对容易构建；我们只是缺少了语用学。助手需要了解更多关于世界的信息，比如产品、商店和价格。通过一点工程，我们可以将这些信息添加到数据库中，并通过提示将这些上下文提供给模型。

![figure](img/2-9.png)

##### 图 2.9：某些任务对 LLMs 的难易程度以及解决这些任务的方法

在另一端，考虑一下棋类机器人。LLMs*可以*下棋，但它们并不擅长。它们在棋局上接受过训练，并理解 E4 是一个常见的第一步，但它们的理解完全是基于语法的。LLMs 只理解它们生成的文本应该包含 A 到 H 之间的字母和 1 到 8 之间的数字。就像购物助手一样，它们缺少语用学，并且没有清晰的棋局模型。此外，它们还缺少语义学。编码器可能有助于我们理解“国王”和“王后”这两个词是相似的，但它们不能帮助我们理解 E4 对于某个玩家来说是一步好棋，而对于另一个玩家来说却是非常糟糕的一步。LLMs 在棋类上还缺乏基于语音学和形态学的知识，尽管在这个案例中它们并不那么重要。无论如何，我们希望这个练习能更好地让你和你的团队了解你们下一个项目。

LLMs 具有惊人的好处，但所有这些能力也带来了一些限制。基础 LLMs 需要大量的计算资源进行训练，这使得它们对个人研究人员和小型组织来说不太容易获得。这个问题正在通过本书中将要讨论的技术得到解决，比如量化、文本嵌入、低秩自适应、参数高效微调和图优化。然而，基础模型目前仍然超出了普通个人有效训练的能力范围。除此之外，还有关于与训练 LLMs 相关的能源消耗可能对环境产生重大影响并引起可持续性问题担忧。这些问题很复杂，很大程度上超出了本书的范围，但我们不能不提及它们。

最后但同样重要的是，由于 LLM 是在包含真实世界文本的大规模数据集上训练的，它们可能会学习和延续数据中存在的偏见，这引发了道德上的担忧，因为现实世界的人不会自我审查以提供最佳的无偏见数据。此外，了解你正在训练的数据的很多信息并不是一种普遍的做法。例如，如果你要求一个文本到图像的扩散 LLM 生成 1000 张“领导者”的图像，99%的图像都是男性，95%的图像都是白人。这里的担忧并不是说男性和白人不应该被描绘为领导者，而是模型没有准确地代表世界，并且正在展示这一点。

有时，更细微的偏见会被揭示出来。例如，在图 2.10 中的 Midjourney 示例中，模型在没有被提示的情况下（唯一的提示是单词“领导者”），将流行的女权主义图标罗斯·拉弗特变成了男性。模型并没有考虑这种变化；它只是在采样步骤中确定，提示“领导者”在训练集中有更多看起来像男性的描绘。许多人会争论在这个背景下“好”和“坏”的含义，而不会追求道德上的义务，我们将讨论准确性的含义。LLM 是在大量数据上训练的，目的是返回尽可能准确的表现。当它们无法返回准确的表现时，特别是当它们具有高度的能力来消除歧义时，我们可以将其视为损害模型实现其目的的偏见。稍后，我们将讨论对抗有害偏见的技术，以便您作为 LLM 的创建者，能够得到您期望的确切输出，并最大限度地减少不期望的输出数量。

![图](img/2-10.png)

##### 图 2.10 中的 Midjourney 5，在撰写本文时，是市场上最受欢迎的文本到图像模型。当只提示一个标记“领导者”（左）时，将一个著名的流行女权主义图标，罗斯·拉弗特，变成了男性的描绘。ChatGPT（右）编写了一个函数，根据种族、性别和年龄为你安排工作。这些都是意外输出的例子。

好的，我们整个章节都在为这一刻做准备。现在我们就来运行我们的第一个大型语言模型（LLM）吧！在列表 2.9 中，我们下载了 Bloom 模型，这是最早创建的开源 LLM 之一，并生成文本！我们使用的是 Hugging Face 的 Transformers 库，它为我们处理了所有繁重的工作。这真是太激动人心了！

##### 列表 2.9 运行我们的第一个 LLM

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "bigscience/bloom"    

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = "Hello world! This is my first time running an LLM!"

input_tokens = tokenizer.encode(prompt, return_tensors="pt", padding=True)
generated_tokens = model.generate(input_tokens, max_new_tokens=20)
generated_text = tokenizer.batch_decode(
    generated_tokens, skip_special_tokens=True
)
print(generated_text)
```

你尝试运行它了吗？！？如果你尝试了，你可能刚刚让你的笔记本电脑崩溃了。哎呀！请原谅我这个小无害的 MLOps 折磨，但了解这些模型有多大以及它们有多难运行是一段有用的经验。在下一章中，我们将更多地讨论运行 LLMs 的困难以及你需要运行此代码的一些工具。如果你不想等待，并希望运行一个类似但规模小得多的 LLM，将模型名称更改为`"bigscience/bloom-3b"`，然后再次运行。这次在大多数硬件上应该可以正常运行。

总的来说，LLMs 是一项惊人的技术，它让我们的想象力在可能性中自由驰骋，这是理所当然的。考虑 LLM 而不是较小的语言模型的首要用例是当模型将帮助的人需要少样本能力时，例如，当 CEO 在筹集资金或软件工程师在编写代码时。LLMs 具有这些能力正是因为它们的规模。LLMs 中参数数量的增加直接使它们能够在更大维度上的较小空间中进行泛化。在本章中，我们触及了 LLMs 不太为人所知的方面，即语言学和语言建模方面。在下一章中，我们将涵盖另一半，即 MLOps 方面，我们将深入了解大量参数大小如何影响模型以及设计来支持该模型和使其对目标客户或员工可访问的系统。

## 摘要

+   语言学五大组成部分是语音学、句法、语义、语用学和形态学：

    +   语音学可以通过处理音频文件的多模态模型来添加，这可能会在未来提高大型语言模型（LLMs），但当前的语料库数据集太小。

    +   句法是当前模型擅长的。

    +   语义是通过嵌入层添加的。

    +   语用学可以通过工程努力来添加。

    +   形态学是在标记化层中添加的。

+   语言不一定与现实相关。了解人们如何在外部现实之外创造意义的过程对于训练对人们有意义的模型是有用的。

+   正确的标记化可能是一个主要障碍，因为存在过多的 `<UNK>` 标记，尤其是在处理像代码或数学这样的专业问题时。

+   多语言处理始终优于单语言处理，甚至在没有模型的单语言任务中也如此。

+   按顺序展示的语言模型类型显示了 LLM 领域的自然和有机增长，因为越来越多的语言学概念被添加，使模型变得更好。

+   语言建模在效率上经历了指数级增长，这与建模的语言学重点相关。

+   注意力是解决更大上下文窗口的数学捷径，是现代架构（编码器、解码器和转换器）的支柱：

    +   编码器提高了嵌入中的语义近似。

    +   解码器在文本生成方面表现最佳。

    +   转换器结合了这两者。

+   更大的模型表现出出现行为，突然能够完成之前无法完成的任务。

[[1]](#footnote-source-1) Vaswani 等人，2017 年，《Attention Is All You Need》，[`arxiv.org/abs/1706.03762`](https://arxiv.org/abs/1706.03762)。

[[2]](#footnote-source-2) J. Wei 等人，“大型语言模型的出现能力”，机器学习研究交易，2022 年 8 月，[`openreview.net/forum?id=yzkSU5zdwD`](https://openreview.net/forum?id=yzkSU5zdwD)。
