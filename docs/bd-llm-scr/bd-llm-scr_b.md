# 附录 B. 参考文献和进一步阅读

## B.1 第一章

定制构建的大型语言模型能够超越通用大型语言模型，正如 Bloomberg 的团队通过一种从头开始预训练的金融数据版本的 GPT 所展示的那样。该定制大型语言模型在金融任务上优于 ChatGPT，同时在通用大型语言模型基准上表现良好：

+   *BloombergGPT：一个大型金融语言模型*（2023）作者为 Wu *等*，`arxiv.org/abs/2303.17564`

现有的大型语言模型也可以适应并微调，以超越通用大型语言模型，这一点由 Google Research 和 Google DeepMind 的团队在医疗环境中展示：

+   *Towards Expert-Level Medical Question Answering with Large Language Models*（2023）作者为 Singhal *等*，`arxiv.org/abs/2305.09617`

提出了原始变换器架构的论文：

+   *Attention Is All You Need*（2017）作者为 Vaswani *等*，`arxiv.org/abs/1706.03762`

原始编码器风格的变换器，称为 BERT：

+   *BERT：用于语言理解的深度双向变换器预训练*（2018）作者为 Devlin *等*，`arxiv.org/abs/1810.04805`

描述解码器风格的 GPT-3 模型的论文，该模型启发了现代大型语言模型，并将在本书中用作从头实现大型语言模型的模板：

+   *语言模型是少样本学习者*（2020）作者为 Brown *等*，`arxiv.org/abs/2005.14165`

用于图像分类的原始视觉变换器，说明变换器架构不仅限于文本输入：

+   *一幅图像价值 16x16 个单词：用于大规模图像识别的变换器*（2020）作者为 Dosovitskiy *等*，`arxiv.org/abs/2010.11929`

两个实验性（但不太流行）的大型语言模型架构作为示例，表明并非所有大型语言模型都需基于变换器架构：

+   *RWKV：为变换器时代重新定义 RNN*（2023）作者为 Peng *等*，`arxiv.org/abs/2305.13048`

+   *Hyena Hierarchy: Towards Larger Convolutional Language Models (2023)*作者为 Poli *等*，`arxiv.org/abs/2302.10866`

+   *Mamba：具有选择性状态空间的线性时间序列建模*（2023）作者为 Gu 和 Dao，`arxiv.org/abs/2312.00752`

Meta AI 的模型是一个受欢迎的 GPT 类模型的实现，与 GPT-3 和 ChatGPT 相比是公开可用的：

+   *Llama 2：开放基础和微调聊天模型*（2023）作者为 Touvron *等*，`arxiv.org/abs/2307.09288`1

对于对第 1.5 节数据集参考的额外细节感兴趣的读者，本文描述了由 Eleuther AI 策划的公开可用数据集 *The Pile*：

+   *The Pile：用于语言建模的 800GB 多样文本数据集*（2020）作者为 Gao *等*，`arxiv.org/abs/2101.00027`。

以下论文提供了用于微调 GPT-3 的 InstructGPT 的参考，该内容在第 1.6 节中提到，并将在第七章中更详细讨论：

+   *通过人类反馈训练语言模型以遵循指令*（2022）由*Ouyang 等*著，`arxiv.org/abs/2203.02155`

## B.2 第二章

有兴趣讨论和比较嵌入空间与潜在空间以及向量表示一般概念的读者，可以在我的书《机器学习 Q 和 AI》的第一章中找到更多信息：

+   *机器学习 Q 和 AI*（2023）由 Sebastian Raschka 著，`leanpub.com/machine-learning-q-and-ai`

以下论文提供了更深入的讨论，说明字节对编码如何作为一种标记化方法使用：

+   稀有词汇的神经机器翻译与子词单元（2015）由 Sennrich 等著，`arxiv.org/abs/1508.07909`

用于训练 GPT-2 的字节对编码标记器的代码已由 OpenAI 开源：

+   `github.com/openai/gpt-2/blob/master/src/encoder.py`

OpenAI 提供了一个交互式网络 UI，说明 GPT 模型中的字节对标记器是如何工作的：

+   `platform.openai.com/tokenizer`

对于有兴趣从头开始编码和训练 BPE 标记器的读者，Andrej Karpathy 的 GitHub 库`minbpe`提供了一个最小且易读的实现：

+   BPE 标记器的最小实现，`github.com/karpathy/minbpe`

有兴趣研究一些其他流行大型语言模型使用的替代标记化方案的读者，可以在 SentencePiece 和 WordPiece 论文中找到更多信息：

+   SentencePiece：一种简单且与语言无关的子词标记器和去标记器，用于神经文本处理（2018）由 Kudo 和 Richardson 著，`aclanthology.org/D18-2012/`

+   快速 WordPiece 标记化（2020）由 Song 等著，`arxiv.org/abs/2012.15524`

## B.3 第三章

有兴趣了解 Bahdanau 注意力在 RNN 和语言翻译中应用的读者，可以在以下论文中找到详细的见解：

+   *通过联合学习对齐和翻译的神经机器翻译*（2014）由 Bahdanau、Cho 和 Bengio 著，`arxiv.org/abs/1409.0473`

自注意力的概念作为缩放点积注意力在原始变换器论文中被介绍：

+   *注意力机制就是你所需要的一切*（2017）由 Vaswani 等著，`arxiv.org/abs/1706.03762`

*FlashAttention*是自注意力机制的高效实现，通过优化内存访问模式加速计算过程。FlashAttention 在数学上与标准自注意力机制相同，但优化了计算过程以提高效率：

+   *FlashAttention: 快速且内存高效的确切注意力与 IO 感知*（2022）由 Dao *等*著，`arxiv.org/abs/2205.14135`

+   *FlashAttention-2: 更快的注意力与更好的并行性和工作分配*（2023）由 Dao 著，`arxiv.org/abs/2307.08691`

PyTorch 实现了一个支持 FlashAttention 以提高效率的自注意力和因果注意力功能。该功能为测试版，可能会有所更改：

+   `scaled_dot_product_attention`文档：`pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html`

PyTorch 还实现了基于`scaled_dot_product`函数的高效`MultiHeadAttention`类：

+   `MultiHeadAttention`文档：`pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`

Dropout 是一种正则化技术，用于在神经网络中防止过拟合，通过在训练期间随机丢弃单元（连同它们的连接）来实现：

+   *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*（2014）由 Srivastava *等*，`jmlr.org/papers/v15/srivastava14a.html`

尽管基于缩放点积注意力的多头注意力在实践中仍然是自注意力最常见的变体，作者发现也可以在没有值权重矩阵和投影层的情况下实现良好的性能：

+   *Simplifying Transformer Blocks*（2023）由 He 和 Hofmann，`arxiv.org/abs/2311.01906`

## B.4 第四章

标题为“层归一化”的论文介绍了一种通过归一化隐藏层内神经元的输入总和来稳定神经网络的隐藏状态动态的技术，相比于之前发布的方法显著减少了训练时间：

+   *Layer Normalization*（2016）由 Ba、Kiros 和 Hinton，`arxiv.org/abs/1607.06450`

在原始 Transformer 模型中使用的 Post-LayerNorm 在自注意力和前馈网络之后应用层归一化。相比之下，像 GPT-2 和更新的 LLM 中采用的 Pre-LayerNorm 在这些组件之前应用层归一化，这可能导致更稳定的训练动态，并且在某些情况下已被证明可以提高性能，如以下论文中所讨论：

+   *On Layer Normalization in the Transformer Architecture*（2020）由 Xiong *等*，`arxiv.org/abs/2002.04745`

+   *ResiDual: Transformer with Dual Residual Connections*（2023）由 Tie *等*， `arxiv.org/abs/2304.14802`

现代 LLM 中使用的流行层归一化变体是 RMSNorm，因为它提高了计算效率。该变体通过仅使用输入的均方根来归一化输入，从而简化了归一化过程，而无需在平方前减去均值。这意味着在计算比例之前不对数据进行中心化。RMSNorm 在以下论文中有更详细的描述：

+   *Root Mean Square Layer Normalization*（2019）由 Zhang 和 Sennrich，`arxiv.org/abs/1910.07467`

GELU（高斯误差线性单元）激活函数结合了经典 ReLU 激活函数和正态分布累积分布函数的特性，以建模层输出，从而在深度学习模型中允许随机正则化和非线性，如以下论文所介绍：

+   *高斯误差线性单元（GELUs）*（2016）由 Hendricks 和 Gimpel 撰写，`arxiv.org/abs/1606.08415`

GPT-2 论文介绍了一系列基于变压器的 LLM，参数大小分别为 124M、355M、774M 和 1.5B：

+   *语言模型是无监督的多任务学习者*（2019）由 Radford *等人*撰写，`d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf`

OpenAI 的 GPT-3 使用的架构基本上与 GPT-2 相同，但最大版本（1750 亿参数）比最大的 GPT-2 模型大 100 倍，并且训练数据量也更多。感兴趣的读者可以参考 OpenAI 的官方 GPT-3 论文和 Lambda Labs 的技术概述，该概述计算在单个 RTX 8000 消费者 GPU 上训练 GPT-3 需要 665 年：

+   语言模型是少样本学习者（2023）由 Brown 等人撰写，`arxiv.org/abs/2005.14165`

+   OpenAI 的 GPT-3 语言模型：技术概述，`lambdalabs.com/blog/demystifying-gpt-3`

NanoGPT 是一个代码库，提供了一个简约而高效的 GPT-2 模型实现，类似于本书中实现的模型。虽然本书中的代码与 nanoGPT 不同，但该代码库启发了将大型 GPT Python 父类实现重组为更小的子模块：

+   NanoGPT，一个用于训练中等大小 GPT 的代码库，`github.com/karpathy/nanoGPT`

一篇信息丰富的博客文章显示，当上下文大小小于 32,000 个标记时，LLM 中的大部分计算花费在前馈层而非注意力层上：

+   “在长（上下文）运行中”由 Harm de Vries 撰写，`www.harmdevries.com/post/context-length/`

## B.5 第五章

作者的一场视频讲座详细说明了损失函数，并应用对数变换以便于数学优化处理：

+   L8.2 逻辑回归损失函数，`www.youtube.com/watch?v=GxJe0DZvydM`

以下两篇论文详细说明了用于预训练 LLM 的数据集、超参数和架构细节：

+   Pythia：分析大型语言模型的训练和扩展套件（2023）由 Biderman *等人*撰写，`arxiv.org/abs/2304.01373`

+   OLMo：加速语言模型科学（2024）由 Groeneveld *等人*撰写，`arxiv.org/abs/2402.00838`

本书附带的以下补充代码包含为 LLM 训练准备来自古腾堡计划的 60,000 本公共领域书籍的说明：

+   在古腾堡项目数据集上预训练 GPT，`github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg`

第五章讨论了 LLM 的预训练，附录 D 涵盖了更高级的训练功能，如线性预热和余弦退火。以下论文发现，类似的技术可以成功应用于继续预训练已经预训练的 LLM，并提供额外的建议和见解：

+   持续预训练大型语言模型的简单且可扩展的策略（2024），作者*Ibrahim*等，`arxiv.org/abs/2403.08763`

BloombergGPT 是一个领域特定的大型语言模型（LLM）的例子，通过对一般和领域特定文本语料库的训练而创建，特别是在金融领域：

+   BloombergGPT：用于金融的大型语言模型（2023），作者*Wu*等，`arxiv.org/abs/2303.17564`

GaLore 是一个近期的研究项目，旨在提高 LLM 预训练的效率。所需的代码更改归结为仅将训练函数中 PyTorch 的`AdamW`优化器替换为由`galore-torch` Python 包提供的`GaLoreAdamW`优化器。

+   GaLore：通过梯度低秩投影进行内存高效的 LLM 训练（2024），作者*Zhao*等，`arxiv.org/abs/2403.03507`

+   GaLore 代码库，`github.com/jiaweizzhao/GaLore`

以下论文和资源公开分享了用于 LLM 的大规模预训练数据集，这些数据集包含数百 GB 到 TB 级别的文本数据：

+   Dolma：一个用于 LLM 预训练研究的三万亿标记开放语料库，作者*Soldaini*等（2024），`arxiv.org/abs/2402.00159`

+   The Pile：一个包含 800GB 多样化文本的数据集，用于语言建模，作者 Gao 等（2020），`arxiv.org/abs/2101.00027`

+   Falcon LLM 的 RefinedWeb 数据集：通过网络数据超越精心策划的语料库，并仅使用网络数据，作者*Penedo*等（2023），`arxiv.org/abs/2306.01116`

+   RedPajama，由 Together AI 提供，`github.com/togethercomputer/RedPajama-Data`

最早引入 top-k 采样的论文：

+   分层神经故事生成，作者*Fan*等（2018），`arxiv.org/abs/1805.04833`

束搜索（在第五章中未涵盖）是一种替代的解码算法，通过在每一步仅保留得分最高的部分序列来生成输出序列，以平衡效率和质量：

+   多样化的束搜索：解码来自神经序列模型的多样化解决方案，作者*Vijayakumar*等（2016），`arxiv.org/abs/1610.02424`
