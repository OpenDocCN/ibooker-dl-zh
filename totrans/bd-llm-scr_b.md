# 附录 B. 参考文献和进一步阅读

## B.1 第 1 章

定制构建的大型语言模型能够超越通用大型语言模型，正如 Bloomberg 的团队通过一种从头开始预训练的金融数据版本的 GPT 所展示的那样。该定制大型语言模型在金融任务上优于 ChatGPT，同时在通用大型语言模型基准上表现良好：

+   *BloombergGPT：一个大型金融语言模型*（2023）作者为 Wu *等*，[https://arxiv.org/abs/2303.17564](abs.html)

现有的大型语言模型也可以适应并微调，以超越通用大型语言模型，这一点由 Google Research 和 Google DeepMind 的团队在医疗环境中展示：

+   *Towards Expert-Level Medical Question Answering with Large Language Models*（2023）作者为 Singhal *等*，[https://arxiv.org/abs/2305.09617](abs.html)

提出了原始变换器架构的论文：

+   *Attention Is All You Need*（2017）作者为 Vaswani *等*，[https://arxiv.org/abs/1706.03762](abs.html)

原始编码器风格的变换器，称为 BERT：

+   *BERT：用于语言理解的深度双向变换器预训练*（2018）作者为 Devlin *等*，[https://arxiv.org/abs/1810.04805](abs.html)

描述解码器风格的 GPT-3 模型的论文，该模型启发了现代大型语言模型，并将在本书中用作从头实现大型语言模型的模板：

+   *语言模型是少样本学习者*（2020）作者为 Brown *等*，[https://arxiv.org/abs/2005.14165](abs.html)

用于图像分类的原始视觉变换器，说明变换器架构不仅限于文本输入：

+   *一幅图像价值 16x16 个单词：用于大规模图像识别的变换器*（2020）作者为 Dosovitskiy *等*，[https://arxiv.org/abs/2010.11929](abs.html)

两个实验性（但不太流行）的大型语言模型架构作为示例，表明并非所有大型语言模型都需基于变换器架构：

+   *RWKV：为变换器时代重新定义 RNN*（2023）作者为 Peng *等*，[https://arxiv.org/abs/2305.13048](abs.html)

+   *Hyena Hierarchy: Towards Larger Convolutional Language Models (2023)*作者为 Poli *等*，[https://arxiv.org/abs/2302.10866](abs.html)

+   *Mamba：具有选择性状态空间的线性时间序列建模*（2023）作者为 Gu 和 Dao，[https://arxiv.org/abs/2312.00752](abs.html)

Meta AI 的模型是一个受欢迎的 GPT 类模型的实现，与 GPT-3 和 ChatGPT 相比是公开可用的：

+   *Llama 2：开放基础和微调聊天模型*（2023）作者为 Touvron *等*，[https://arxiv.org/abs/2307.09288](abs.html)[1](abs.html)

对于对第 1.5 节数据集参考的额外细节感兴趣的读者，本文描述了由 Eleuther AI 策划的公开可用数据集 *The Pile*：

+   *The Pile：用于语言建模的 800GB 多样文本数据集*（2020）作者为 Gao *等*，[https://arxiv.org/abs/2101.00027](abs.html)。

以下论文提供了用于微调 GPT-3 的 InstructGPT 的参考，该内容在第 1.6 节中提到，并将在第 7 章中更详细讨论：

+   *通过人类反馈训练语言模型以遵循指令*（2022）由*Ouyang等*著，[https://arxiv.org/abs/2203.02155](abs.html)

## B.2 第二章

有兴趣讨论和比较嵌入空间与潜在空间以及向量表示一般概念的读者，可以在我的书《机器学习 Q 和 AI》的第一章中找到更多信息：

+   *机器学习 Q 和 AI*（2023）由Sebastian Raschka著，[https://leanpub.com/machine-learning-q-and-ai](machine-learning-q-and-ai.html)

以下论文提供了更深入的讨论，说明字节对编码如何作为一种标记化方法使用：

+   稀有词汇的神经机器翻译与子词单元（2015）由Sennrich等著，[https://arxiv.org/abs/1508.07909](abs.html)

用于训练GPT-2的字节对编码标记器的代码已由OpenAI开源：

+   [https://github.com/openai/gpt-2/blob/master/src/encoder.py](src.html)

OpenAI提供了一个交互式网络UI，说明GPT模型中的字节对标记器是如何工作的：

+   [https://platform.openai.com/tokenizer](platform.openai.com.html)

对于有兴趣从头开始编码和训练BPE标记器的读者，Andrej Karpathy的GitHub库`minbpe`提供了一个最小且易读的实现：

+   BPE标记器的最小实现，[https://github.com/karpathy/minbpe](karpathy.html)

有兴趣研究一些其他流行大型语言模型使用的替代标记化方案的读者，可以在SentencePiece和WordPiece论文中找到更多信息：

+   SentencePiece：一种简单且与语言无关的子词标记器和去标记器，用于神经文本处理（2018）由Kudo和Richardson著，[https://aclanthology.org/D18-2012/](D18-2012.html)

+   快速WordPiece标记化（2020）由Song等著，[https://arxiv.org/abs/2012.15524](abs.html)

## B.3 第三章

有兴趣了解Bahdanau注意力在RNN和语言翻译中应用的读者，可以在以下论文中找到详细的见解：

+   *通过联合学习对齐和翻译的神经机器翻译*（2014）由Bahdanau、Cho和Bengio著，[https://arxiv.org/abs/1409.0473](abs.html)

自注意力的概念作为缩放点积注意力在原始变换器论文中被介绍：

+   *注意力机制就是你所需要的一切*（2017）由Vaswani等著，[https://arxiv.org/abs/1706.03762](abs.html)

*FlashAttention*是自注意力机制的高效实现，通过优化内存访问模式加速计算过程。FlashAttention在数学上与标准自注意力机制相同，但优化了计算过程以提高效率：

+   *FlashAttention: 快速且内存高效的确切注意力与IO感知*（2022）由Dao *等*著，[https://arxiv.org/abs/2205.14135](abs.html)

+   *FlashAttention-2: 更快的注意力与更好的并行性和工作分配*（2023）由Dao著，[https://arxiv.org/abs/2307.08691](abs.html)

PyTorch实现了一个支持FlashAttention以提高效率的自注意力和因果注意力功能。该功能为测试版，可能会有所更改：

+   `scaled_dot_product_attention`文档：[https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html](generated.html)

PyTorch还实现了基于`scaled_dot_product`函数的高效`MultiHeadAttention`类：

+   `MultiHeadAttention`文档：[https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html](generated.html)

Dropout是一种正则化技术，用于在神经网络中防止过拟合，通过在训练期间随机丢弃单元（连同它们的连接）来实现：

+   *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*（2014）由Srivastava *等*，[https://jmlr.org/papers/v15/srivastava14a.html](v15.html)

尽管基于缩放点积注意力的多头注意力在实践中仍然是自注意力最常见的变体，作者发现也可以在没有值权重矩阵和投影层的情况下实现良好的性能：

+   *Simplifying Transformer Blocks*（2023）由He和Hofmann，[https://arxiv.org/abs/2311.01906](abs.html)

## B.4 第四章

标题为“层归一化”的论文介绍了一种通过归一化隐藏层内神经元的输入总和来稳定神经网络的隐藏状态动态的技术，相比于之前发布的方法显著减少了训练时间：

+   *Layer Normalization*（2016）由Ba、Kiros和Hinton，[https://arxiv.org/abs/1607.06450](abs.html)

在原始Transformer模型中使用的Post-LayerNorm在自注意力和前馈网络之后应用层归一化。相比之下，像GPT-2和更新的LLM中采用的Pre-LayerNorm在这些组件之前应用层归一化，这可能导致更稳定的训练动态，并且在某些情况下已被证明可以提高性能，如以下论文中所讨论：

+   *On Layer Normalization in the Transformer Architecture*（2020）由Xiong *等*，[https://arxiv.org/abs/2002.04745](abs.html)

+   *ResiDual: Transformer with Dual Residual Connections*（2023）由Tie *等*， [https://arxiv.org/abs/2304.14802](abs.html)

现代LLM中使用的流行层归一化变体是RMSNorm，因为它提高了计算效率。该变体通过仅使用输入的均方根来归一化输入，从而简化了归一化过程，而无需在平方前减去均值。这意味着在计算比例之前不对数据进行中心化。RMSNorm在以下论文中有更详细的描述：

+   *Root Mean Square Layer Normalization*（2019）由Zhang和Sennrich，[https://arxiv.org/abs/1910.07467](abs.html)

GELU（高斯误差线性单元）激活函数结合了经典ReLU激活函数和正态分布累积分布函数的特性，以建模层输出，从而在深度学习模型中允许随机正则化和非线性，如以下论文所介绍：

+   *高斯误差线性单元（GELUs）*（2016）由Hendricks和Gimpel撰写，[https://arxiv.org/abs/1606.08415](abs.html)

GPT-2论文介绍了一系列基于变压器的LLM，参数大小分别为124M、355M、774M和1.5B：

+   *语言模型是无监督的多任务学习者*（2019）由Radford *等人*撰写，[https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](better-language-models.html)

OpenAI的GPT-3使用的架构基本上与GPT-2相同，但最大版本（1750亿参数）比最大的GPT-2模型大100倍，并且训练数据量也更多。感兴趣的读者可以参考OpenAI的官方GPT-3论文和Lambda Labs的技术概述，该概述计算在单个RTX 8000消费者GPU上训练GPT-3需要665年：

+   语言模型是少样本学习者（2023）由Brown等人撰写，[https://arxiv.org/abs/2005.14165](abs.html)

+   OpenAI的GPT-3语言模型：技术概述，[https://lambdalabs.com/blog/demystifying-gpt-3](blog.html)

NanoGPT是一个代码库，提供了一个简约而高效的GPT-2模型实现，类似于本书中实现的模型。虽然本书中的代码与nanoGPT不同，但该代码库启发了将大型GPT Python父类实现重组为更小的子模块：

+   NanoGPT，一个用于训练中等大小GPT的代码库，[https://github.com/karpathy/nanoGPT](karpathy.html)

一篇信息丰富的博客文章显示，当上下文大小小于32,000个标记时，LLM中的大部分计算花费在前馈层而非注意力层上：

+   “在长（上下文）运行中”由Harm de Vries撰写，[https://www.harmdevries.com/post/context-length/](context-length.html)

## B.5 第五章

作者的一场视频讲座详细说明了损失函数，并应用对数变换以便于数学优化处理：

+   L8.2 逻辑回归损失函数，[https://www.youtube.com/watch?v=GxJe0DZvydM](www.youtube.com.html)

以下两篇论文详细说明了用于预训练LLM的数据集、超参数和架构细节：

+   Pythia：分析大型语言模型的训练和扩展套件（2023）由Biderman *等人*撰写，[https://arxiv.org/abs/2304.01373](abs.html)

+   OLMo：加速语言模型科学（2024）由Groeneveld *等人*撰写，[https://arxiv.org/abs/2402.00838](abs.html)

本书附带的以下补充代码包含为LLM训练准备来自古腾堡计划的60,000本公共领域书籍的说明：

+   在古腾堡项目数据集上预训练GPT，[https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg](ch05.html)

第五章讨论了LLM的预训练，附录D涵盖了更高级的训练功能，如线性预热和余弦退火。以下论文发现，类似的技术可以成功应用于继续预训练已经预训练的LLM，并提供额外的建议和见解：

+   持续预训练大型语言模型的简单且可扩展的策略（2024），作者*Ibrahim*等，[https://arxiv.org/abs/2403.08763](abs.html)

BloombergGPT是一个领域特定的大型语言模型（LLM）的例子，通过对一般和领域特定文本语料库的训练而创建，特别是在金融领域：

+   BloombergGPT：用于金融的大型语言模型（2023），作者*Wu*等，[https://arxiv.org/abs/2303.17564](abs.html)

GaLore是一个近期的研究项目，旨在提高LLM预训练的效率。所需的代码更改归结为仅将训练函数中PyTorch的`AdamW`优化器替换为由`galore-torch` Python包提供的`GaLoreAdamW`优化器。

+   GaLore：通过梯度低秩投影进行内存高效的LLM训练（2024），作者*Zhao*等，[https://arxiv.org/abs/2403.03507](abs.html)

+   GaLore代码库，[https://github.com/jiaweizzhao/GaLore](jiaweizzhao.html)

以下论文和资源公开分享了用于LLM的大规模预训练数据集，这些数据集包含数百GB到TB级别的文本数据：

+   Dolma：一个用于LLM预训练研究的三万亿标记开放语料库，作者*Soldaini*等（2024），[https://arxiv.org/abs/2402.00159](abs.html)

+   The Pile：一个包含800GB多样化文本的数据集，用于语言建模，作者Gao等（2020），[https://arxiv.org/abs/2101.00027](abs.html)

+   Falcon LLM的RefinedWeb数据集：通过网络数据超越精心策划的语料库，并仅使用网络数据，作者*Penedo*等（2023），[https://arxiv.org/abs/2306.01116](abs.html)

+   RedPajama，由Together AI提供，[https://github.com/togethercomputer/RedPajama-Data](togethercomputer.html)

最早引入top-k采样的论文：

+   分层神经故事生成，作者*Fan*等（2018），[https://arxiv.org/abs/1805.04833](abs.html)

束搜索（在第五章中未涵盖）是一种替代的解码算法，通过在每一步仅保留得分最高的部分序列来生成输出序列，以平衡效率和质量：

+   多样化的束搜索：解码来自神经序列模型的多样化解决方案，作者*Vijayakumar*等（2016），[https://arxiv.org/abs/1610.02424](abs.html)
