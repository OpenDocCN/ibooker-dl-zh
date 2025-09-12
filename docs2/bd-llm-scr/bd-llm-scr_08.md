# 附录 B 参考文献及进一步阅读

## 第一章

如彭博团队通过从零开始预训练金融数据的 GPT 版本所展示，定制构建的 LLM 能够超越通用 LLM，同时在通用 LLM 基准测试中保持良好的性能：

+   Wu 等人撰写的《“BloombergGPT：用于金融的大型语言模型”》（2023），[`arxiv.org/abs/2303.17564`](https://arxiv.org/abs/2303.17564)

现有的 LLM 可以进行适配和微调，以超越通用 LLM，谷歌研究和谷歌 DeepMind 团队在医疗环境中展示了这一点：

+   Singhal 等人撰写的《“使用大型语言模型实现专家级医学问答”》（2023），[`arxiv.org/abs/2305.09617`](https://arxiv.org/abs/2305.09617)

以下论文提出了原始的 Transformer 架构：

+   Vaswani 等人撰写的《“注意力即一切”》（2017），[`arxiv.org/abs/1706.03762`](https://arxiv.org/abs/1706.03762)

关于原始编码器风格的 Transformer，即 BERT，请参阅

+   Devlin 等人撰写的《“BERT：用于语言理解的深度双向 Transformer 的预训练”》（2018），[`arxiv.org/abs/1810.04805`](https://arxiv.org/abs/1810.04805)

描述解码器风格的 GPT-3 模型的论文，该模型启发了现代 LLM，并将作为本书中从头实现 LLM 的模板，该论文是

+   Brown 等人撰写的《“语言模型是少样本学习者”》（2020），[`arxiv.org/abs/2005.14165`](https://arxiv.org/abs/2005.14165)

以下涵盖了用于图像分类的原版视觉 Transformer，这表明 Transformer 架构不仅限于文本输入：

+   Dosovitskiy 等人撰写的《“一张图片等于 16x16 个单词：大规模图像识别的 Transformer”》（2020），[`arxiv.org/abs/2010.11929`](https://arxiv.org/abs/2010.11929)

以下实验（但不太流行）的 LLM 架构作为例子，说明并非所有 LLM 都必须基于 Transformer 架构：

+   Peng 等人撰写的《“RWKV：为 Transformer 时代重新发明 RNN”》（2023），[`arxiv.org/abs/2305.13048`](https://arxiv.org/abs/2305.13048)

+   Poli 等人撰写的《“鬣狗等级：向更大型的卷积语言模型迈进”》（2023），[`arxiv.org/abs/2302.10866`](https://arxiv.org/abs/2302.10866)

+   Gu 和 Dao 撰写的《“Mamba：具有选择性状态空间的线性时间序列建模”》（2023），[`arxiv.org/abs/2312.00752`](https://arxiv.org/abs/2312.00752)

Meta AI 的模型是 GPT 类模型的一种流行实现，与 GPT-3 和 ChatGPT 相比，它是公开可用的：

+   Touvron 等人撰写的《“Llama 2：开放基础和微调的聊天模型”》（2023），[`arxiv.org/abs/2307.092881`](https://arxiv.org/abs/2307.092881)

对于对第 1.5 节中数据集引用的详细信息感兴趣的读者，本文描述了由 Eleuther AI 精心整理的公开可用 *The Pile* 数据集：

+   “The Pile：用于语言建模的 800GB 多样化文本数据集”（2020）由 Gao 等人所著，[`arxiv.org/abs/2101.00027`](https://arxiv.org/abs/2101.00027)

以下论文提供了 InstructGPT 微调 GPT-3 的参考，这在第 1.6 节中提到，将在第七章中更详细地讨论：

+   “通过人类反馈训练语言模型以遵循指令”（2022）由 Ouyang 等人所著，[`arxiv.org/abs/2203.02155`](https://arxiv.org/abs/2203.02155)

## 第二章

对于那些对讨论和比较嵌入空间与潜在空间以及向量表示的通用概念感兴趣的读者，可以在我的书的第一章中找到更多信息：

+   机器学习 Q 和 AI（2023）由 Sebastian Raschka 所著，[`leanpub.com/machine-learning-q-and-ai`](https://leanpub.com/machine-learning-q-and-ai)

以下论文更深入地讨论了字节对编码作为分词方法的使用：

+   “使用子词单元进行罕见词的神经机器翻译”（2015）由 Sennrich 等人所著，[`arxiv.org/abs/1508.07909`](https://arxiv.org/abs/1508.07909)

OpenAI 开源了用于训练 GPT-2 的字节对编码分词器的代码：

+   [`github.com/openai/gpt-2/blob/master/src/encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py)

OpenAI 提供了一个交互式的 Web 用户界面，以展示 GPT 模型中的字节对分词器是如何工作的：

+   [`platform.openai.com/tokenizer`](https://platform.openai.com/tokenizer)

对于那些对从零开始编码和训练 BPE 分词器感兴趣的读者，Andrej Karpathy 的 GitHub 仓库`minbpe`提供了一个最小化和可读的实现：

+   “一个 BPE 分词器的最小实现”，[`github.com/karpathy/minbpe`](https://github.com/karpathy/minbpe)

对于那些对研究某些其他流行 LLM 使用的替代分词方案感兴趣的读者，可以在 SentencePiece 和 WordPiece 论文中找到更多信息：

+   “SentencePiece：用于神经文本处理的一个简单且语言无关的子词分词器和反分词器”（2018）由 Kudo 和 Richardson 所著，[`aclanthology.org/D18-2012/`](https://aclanthology.org/D18-2012/)

+   “快速 WordPiece 分词”（2020）由 Song 等人所著，[`arxiv.org/abs/2012.15524`](https://arxiv.org/abs/2012.15524)

## 第三章

对于那些想要了解更多关于 RNN 和语言翻译中 Bahdanau 注意力机制的读者，可以在以下论文中找到详细见解：

+   “一个 BPE 分词器的最小实现”，[`github.com/karpathy/minbpe`](https://github.com/karpathy/minbpe)

自注意力作为缩放点积注意力在原始的 Transformer 论文中被引入：

+   “Attention Is All You Need”（2017）由 Vaswani 等人所著，[`arxiv.org/abs/1706.03762`](https://arxiv.org/abs/1706.03762)

FlashAttention 是自注意力机制的高效实现，通过优化内存访问模式加速计算过程。从数学上讲，FlashAttention 与标准自注意力机制相同，但优化了计算过程以提高效率：

+   “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness” (2022) by Dao et al., [`arxiv.org/abs/2205.14135`](https://arxiv.org/abs/2205.14135)

+   “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning” (2023) by Dao, [`arxiv.org/abs/2307.08691`](https://arxiv.org/abs/2307.08691)

PyTorch 实现了一个支持 FlashAttention 以提高效率的自注意力和因果注意力的函数。此函数处于测试阶段，可能会发生变化：

+   `scaled_dot_product_attention`文档：[`mng.bz/NRJd`](https://mng.bz/NRJd)

PyTorch 也实现了一个基于`scaled_` `dot_product`函数的高效的`MultiHeadAttention`类：

+   `MultiHeadAttention`文档：[`mng.bz/DdJV`](https://mng.bz/DdJV)

Dropout 是一种在训练过程中随机从神经网络中丢弃单元（及其连接）的正则化技术，以防止过拟合：

+   “Dropout: A Simple Way to Prevent Neural Networks from Overfitting” (2014) by Srivastava et al., [`jmlr.org/papers/v15/srivastava14a.html`](https://jmlr.org/papers/v15/srivastava14a.html)

虽然在实践中最常见的自注意力变体是基于缩放点积注意力的多头注意力，但作者发现，即使没有值权重矩阵和投影层，也可以实现良好的性能：

+   “Simplifying Transformer Blocks” (2023) by He and Hofmann, [`arxiv.org/abs/2311.01906`](https://arxiv.org/abs/2311.01906)

## 第四章

以下论文介绍了一种通过标准化隐藏层中神经元的总和输入来稳定隐藏状态动态的神经网络的技术，与之前发表的方法相比，显著减少了训练时间：

+   “Layer Normalization” (2016) by Ba, Kiros, and Hinton, [`arxiv.org/abs/1607.06450`](https://arxiv.org/abs/1607.06450)

Post-LayerNorm，在原始 Transformer 模型中使用，在自注意力和前馈网络之后应用层归一化。相比之下，Pre-LayerNorm，如 GPT-2 和更新的 LLMs 中采用，在这些组件之前应用层归一化，这可能导致更稳定的训练动态，并且已经证明在某些情况下可以提高性能，如以下论文所述：

+   “On Layer Normalization in the Transformer Architecture” (2020) by Xiong et al., [`arxiv.org/abs/2002.04745`](https://arxiv.org/abs/2002.04745)

+   “ResiDual: Transformer with Dual Residual Connections” (2023) by Tie et al., [`arxiv.org/abs/2304.14802`](https://arxiv.org/abs/2304.14802)

在现代 LLM 中使用的 LayerNorm 的一个流行变体是 RMSNorm，因为它提高了计算效率。这种变体通过仅使用输入的均方根来归一化输入，而不在平方之前减去均值，从而简化了归一化过程。这意味着在计算尺度之前不会对数据进行中心化。RMSNorm 在以下内容中描述得更详细：

+   “均方根层归一化”（2019）由 Zhang 和 Sennrich 撰写，[`arxiv.org/abs/1910.07467`](https://arxiv.org/abs/1910.07467)

高斯误差线性单元（GELU）激活函数结合了经典 ReLU 激活函数和正态分布的累积分布函数的特性，以建模层输出，允许在深度学习模型中进行随机正则化和非线性：

+   “高斯误差线性单元（GELUs）”（2016）由 Hendricks 和 Gimpel 撰写，[`arxiv.org/abs/1606.08415`](https://arxiv.org/abs/1606.08415)

GPT-2 论文介绍了一系列基于 transformer 的、大小各异的 LLM——1.24 亿、3.55 亿、7.74 亿和 15 亿个参数：

+   “语言模型是无监督多任务学习者”（2019）由 Radford 等人撰写，[`mng.bz/DMv0`](http://mng.bz/DMv0)

OpenAI 的 GPT-3 基本上使用了与 GPT-2 相同的架构，除了最大的版本（1750 亿）比最大的 GPT-2 模型大 100 倍，并且使用了更多的数据进行训练。感兴趣的读者可以参考 OpenAI 的官方 GPT-3 论文以及 Lambda Labs 的技术概述，该概述计算在单个 RTX 8000 消费级 GPU 上训练 GPT-3 需要 665 年：

+   “语言模型是少样本学习者”（2023）由 Brown 等人撰写，[`arxiv.org/abs/2005.14165`](https://arxiv.org/abs/2005.14165)

+   “OpenAI 的 GPT-3 语言模型：技术概述”，[`lambdalabs.com/blog/demystifying-gpt-3`](https://lambdalabs.com/blog/demystifying-gpt-3)

NanoGPT 是一个代码仓库，它以简约而高效的方式实现了 GPT-2 模型，类似于本书中实现的模型。虽然本书中的代码与 nanoGPT 不同，但这个仓库激发了将大型 GPT Python 父类实现重组为较小子模块的灵感：

+   “NanoGPT，一个用于训练中等大小 GPT 的仓库，[`github.com/karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT)

一篇信息丰富的博客文章展示了，当上下文大小小于 32,000 个标记时，大多数 LLM 的计算都是在前馈层而不是注意力层中进行的：

+   “在长（上下文）运行中”由 Harm de Vries 撰写，[`www.harmdevries.com/post/context-length/`](https://www.harmdevries.com/post/context-length/)

## 第五章

关于详细说明损失函数和应用对数变换以使其更容易进行数学优化的信息，请参阅我的讲座视频：

+   L8.2 逻辑回归损失函数，[`www.youtube.com/watch?v=GxJe0DZvydM`](https://www.youtube.com/watch?v=GxJe0DZvydM)

作者提供的以下讲座和代码示例解释了 PyTorch 的交叉熵函数在底层是如何工作的：

+   L8.7.1 OneHot 编码和多类别交叉熵，[`www.youtube.com/watch?v=4n71-tZ94yk`](https://www.youtube.com/watch?v=4n71-tZ94yk)

+   理解 PyTorch 中的 Onehot 编码和交叉熵，[`mng.bz/o05v`](https://mng.bz/o05v)

以下两篇论文详细介绍了用于预训练 LLM 的数据集、超参数和架构细节：

+   “Pythia：用于分析大型语言模型训练和扩展的套件”（2023）由 Biderman 等人撰写，[`arxiv.org/abs/2304.01373`](https://arxiv.org/abs/2304.01373)

+   “OLMo：加速语言模型科学”（2024）由 Groeneveld 等人撰写，[`arxiv.org/abs/2402.00838`](https://arxiv.org/abs/2402.00838)

以下为本书提供的补充代码，包含从 Project Gutenberg 项目准备 60,000 本公共领域书籍用于 LLM 训练的说明：

+   在 Project Gutenberg 数据集上预训练 GPT，[`mng.bz/Bdw2`](https://mng.bz/Bdw2)

第五章讨论了 LLM 的预训练，附录 D 涵盖了更高级的训练函数，例如线性预热和余弦退火。以下论文发现，类似的技巧可以成功应用于继续预训练已经预训练的 LLM，并附带额外的技巧和见解：

+   “简单且可扩展的策略以持续预训练大型语言模型”（2024）由 Ibrahim 等人撰写，[`arxiv.org/abs/2403.08763`](https://arxiv.org/abs/2403.08763)

BloombergGPT 是通过对通用和特定领域文本语料库进行训练创建的特定领域 LLM 的示例，特别是在金融领域：

+   “BloombergGPT：用于金融的 LLM”（2023）由 Wu 等人撰写，[`arxiv.org/abs/2303.17564`](https://arxiv.org/abs/2303.17564)

GaLore 是一个旨在使 LLM 预训练更高效的研究项目。所需的代码更改仅限于将训练函数中的 PyTorch 的`AdamW`优化器替换为`galore-torch`Python 包提供的`GaLoreAdamW`优化器：

+   “GaLore：通过梯度低秩投影提高 LLM 训练效率”（2024）由 Zhao 等人撰写，[`arxiv.org/abs/2403.03507`](https://arxiv.org/abs/2403.03507)

+   GaLore 代码仓库，[`github.com/jiaweizzhao/GaLore`](https://github.com/jiaweizzhao/GaLore)

以下论文和资源公开分享了适用于 LLM 的大规模预训练数据集，这些数据集包含数百 GB 到 TB 的文本数据：

+   “Dolma：用于 LLM 预训练研究的 3000 亿标记开放语料库”（2024）由 Soldaini 等人撰写，[`arxiv.org/abs/2402.00159`](https://arxiv.org/abs/2402.00159)

+   “The Pile：用于语言建模的 800GB 多样化文本数据集”（2020）由 Gao 等人撰写，[`arxiv.org/abs/2101.00027`](https://arxiv.org/abs/2101.00027)

+   “Falcon LLM 的 RefinedWeb 数据集：仅使用网页数据超越精选语料库，”（2023）由 Penedo 等人撰写，[`arxiv.org/abs/2306.01116`](https://arxiv.org/abs/2306.01116)

+   “RedPajama”，由 Together AI 编写，[`mng.bz/d6nw`](https://mng.bz/d6nw)

+   FineWeb 数据集，包括来自 CommonCrawl 的超过 1500 万亿个清洗和去重的英文网页数据，[`mng.bz/rVzy`](https://mng.bz/rVzy)

原始介绍 top-k 采样的论文是

+   “分层神经故事生成”（2018）由 Fan 等人撰写，[`arxiv.org/abs/1805.04833`](https://arxiv.org/abs/1805.04833)

top-k 采样的替代方法是 top-p 采样（第五章未涉及），它从累积概率超过阈值*p*的最小 top tokens 集中选择，而 top-k 采样则按概率从 top *k* tokens 中选择：

+   Top-p 采样，[`en.wikipedia.org/wiki/Top-p_sampling`](https://en.wikipedia.org/wiki/Top-p_sampling)

束搜索（第五章未涉及）是一种替代解码算法，通过在每个步骤中仅保留得分最高的部分序列来生成输出序列，以平衡效率和质量：

+   “多样化的束搜索：从神经序列模型解码多样化的解决方案”（2016）由 Vijayakumar 等人撰写，[`arxiv.org/abs/1610.02424`](https://arxiv.org/abs/1610.02424)

## 第六章

讨论不同类型微调的额外资源包括

+   “使用和微调预训练的 Transformer，”[`mng.bz/VxJG`](https://mng.bz/VxJG)

+   “微调大型语言模型”，[`mng.bz/x28X`](https://mng.bz/x28X)

包括比较微调第一个输出标记与最后一个输出标记的额外实验，可以在 GitHub 上的补充代码材料中找到：

+   额外的垃圾邮件分类实验，[`mng.bz/AdJx`](https://mng.bz/AdJx)

对于像垃圾邮件分类这样的二分类任务，技术上可以使用单个输出节点而不是两个输出节点，正如我在以下文章中讨论的那样：

+   “损失学习——在 PyTorch 中优化负对数似然和交叉熵”，[`mng.bz/ZEJA`](https://mng.bz/ZEJA)

你可以在以下文章中找到关于微调 LLM 不同层的额外实验，该文章表明除了输出层外，微调最后一个 Transformer 块可以显著提高预测性能：

+   “微调大型语言模型”，[`mng.bz/RZJv`](https://mng.bz/RZJv)

读者可以在 imbalanced-learn 文档中找到处理不平衡分类数据集的额外资源和信息：

+   “Imbalanced-Learn 用户指南”，[`mng.bz/2KNa`](https://mng.bz/2KNa)

对于那些对分类垃圾邮件而非垃圾短信感兴趣的读者，以下资源提供了一个方便的 CSV 格式的大电子邮件垃圾邮件分类数据集，类似于第六章中使用的数据集格式：

+   电子邮件垃圾邮件分类数据集，[`mng.bz/1GEq`](https://mng.bz/1GEq)

GPT-2 是基于 transformer 架构的解码器模块的模型，其主要目的是生成新的文本。作为替代，基于编码器的模型，如 BERT 和 RoBERTa，在分类任务中可能非常有效：

+   “BERT：用于语言理解的深度双向变换器预训练”（2018）由 Devlin 等人撰写，[`arxiv.org/abs/1810.04805`](https://arxiv.org/abs/1810.04805)

+   刘等人撰写的“RoBERTa：一种鲁棒优化的 BERT 预训练方法”（2019），[`arxiv.org/abs/1907.11692`](https://arxiv.org/abs/1907.11692)

+   “对 50k IMDB 电影评论进行情感分类的额外实验”，[`mng.bz/PZJR`](https://mng.bz/PZJR)

近期论文显示，通过在分类微调期间移除因果掩码以及进行其他修改，可以进一步提高分类性能：

+   “标签监督的 LLaMA 微调”（2023）由 Li 等人撰写，[`arxiv.org/abs/2310.01208`](https://arxiv.org/abs/2310.01208)

+   “LLM2Vec：大型语言模型是秘密强大的文本编码器”（2024）由 BehnamGhader 等人撰写，[`arxiv.org/abs/2404.05961`](https://arxiv.org/abs/2404.05961)

## 第七章

用于指令微调的 Alpaca 数据集包含 52,000 个指令-响应对，是第一个也是最流行的公开可用的指令微调数据集之一：

+   “斯坦福 Alpaca：一个遵循指令的 Llama 模型”，[`github.com/tatsu-lab/stanford_alpaca`](https://github.com/tatsu-lab/stanford_alpaca)

适用于指令微调的额外公开可访问数据集包括

+   LIMA，[`huggingface.co/datasets/GAIR/lima`](https://huggingface.co/datasets/GAIR/lima)

    +   更多信息，请参阅周等人撰写的“LIMA：对齐的‘少即是多’”，[`arxiv.org/abs/2305.11206`](https://arxiv.org/abs/2305.11206)

+   UltraChat，[`huggingface.co/datasets/openchat/ultrachat-sharegpt`](https://huggingface.co/datasets/openchat/ultrachat-sharegpt)

    +   一个包含 805,000 个指令-响应对的庞大数据集；更多信息，请参阅 Ding 等人撰写的“通过扩展高质量指令对话来增强聊天语言模型”，[`arxiv.org/abs/2305.14233`](https://arxiv.org/abs/2305.14233)

+   Alpaca GPT4，[`mng.bz/Aa0p`](https://mng.bz/Aa0p)

+   一个类似 Alpaca 的数据集，包含 52,000 个指令-响应对，使用 GPT-4 而不是 GPT-3.5 生成

Phi-3 是一个拥有 38 亿参数的模型，其指令微调变体据称与许多更大的专有模型相当，例如 GPT-3.5：

+   “Phi-3 技术报告：在您的手机上本地运行的高性能语言模型”（2024）由 Abdin 等人撰写，[`arxiv.org/abs/2404.14219`](https://arxiv.org/abs/2404.14219)

研究人员提出了一种合成指令数据生成方法，从指令微调的 Llama-3 模型中生成 300,000 个高质量的指令-响应对。在这些指令示例上微调的预训练 Llama 3 基础模型的表现与原始指令微调的 Llama-3 模型相当：

+   “Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing” (2024) by Xu et al., [`arxiv.org/abs/2406.08464`](https://arxiv.org/abs/2406.08464)

研究表明，在指令微调中不屏蔽指令和输入可以有效提高各种 NLP 任务和开放式生成基准测试的性能，尤其是在训练数据集包含长指令和简短输出或使用少量训练示例时：

+   “Instruction Tuning with Loss Over Instructions” (2024) by Shi, [`arxiv.org/abs/2405.14394`](https://arxiv.org/abs/2405.14394)

Prometheus 和 PHUDGE 是公开可用的 LLM，它们在评估长文本响应时可以与 GPT-4 相媲美，并支持自定义标准。我们之所以没有使用这些模型，是因为在撰写本文时，它们没有得到 Ollama 的支持，因此无法在笔记本电脑上高效执行：

+   “Prometheus: Inducing Finegrained Evaluation Capability in Language Models” (2023) by Kim et al., [`arxiv.org/abs/2310.08491`](https://arxiv.org/abs/2310.08491)

+   “PHUDGE: Phi-3 as Scalable Judge” (2024) by Deshwal and Chawla, “[`arxiv.org/abs/2405.08029`](https://arxiv.org/abs/2405.08029)

+   “Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models” (2024), by Kim et al., [`arxiv.org/abs/2405.01535`](https://arxiv.org/abs/2405.01535)

以下报告中的结果支持了这样一个观点：大型语言模型在预训练期间主要获取事实知识，而微调主要增强了它们使用这种知识的能力。此外，这项研究探讨了使用新事实信息微调大型语言模型如何影响它们使用现有知识的能力，揭示了模型学习新事实的速度较慢，并且在微调期间引入新事实增加了模型生成错误信息的倾向：

+   “Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?” (2024) by Gekhman, [`arxiv.org/abs/2405.05904`](https://arxiv.org/abs/2405.05904)

预设偏好微调是在指令微调之后的一个可选步骤，目的是使 LLM 更接近人类的偏好。作者以下文章提供了更多关于此过程的信息：

+   “LLM Training: RLHF and Its Alternatives,” [`mng.bz/ZVPm`](https://mng.bz/ZVPm)

+   “Tips for LLM Pretraining and Evaluating Reward Models,” [`mng.bz/RNXj`](https://mng.bz/RNXj)

## 附录 A

虽然附录 A 应该足以让您跟上进度，但如果您正在寻找更全面的深度学习介绍，我推荐以下书籍：

+   《使用 PyTorch 和 Scikit-Learn 进行机器学习》（2022）由 Sebastian Raschka、Hayden Liu 和 Vahid Mirjalili 著。ISBN 978-1801819312

+   《使用 PyTorch 进行深度学习》（2021）由 Eli Stevens、Luca Antiga 和 Thomas Viehmann 著。ISBN 978-1617295263

对于更深入的张量概念介绍，读者可以找到我录制的一个 15 分钟的视频教程：

+   “第 4.1 节：深度学习中的张量，” [`www.youtube.com/watch?v=JXfDlgrfOBY`](https://www.youtube.com/watch?v=JXfDlgrfOBY)

如果你想了解更多关于机器学习中模型评估的内容，我推荐我的文章

+   “机器学习中的模型评估、模型选择和算法选择”（2018）由 Sebastian Raschka 著，[`arxiv.org/abs/1811.12808`](https://arxiv.org/abs/1811.12808)

对于对微积分复习或温和介绍感兴趣的读者，我在我的网站上写了一章关于微积分的内容，免费提供：

+   “微积分导论，”由 Sebastian Raschka 著，[`mng.bz/WEyW`](https://mng.bz/WEyW)

为什么 PyTorch 不在后台自动为我们调用 `optimizer.zero_grad()`？在某些情况下，可能希望累积梯度，PyTorch 将将其作为选项留给我们。如果您想了解更多关于梯度累积的信息，请参阅以下文章：

+   “使用梯度累积在单个 GPU 上微调大型语言模型”由 Sebastian Raschka 著，[`mng.bz/8wPD`](https://mng.bz/8wPD)

本附录涵盖了 DDP，这是一种在多个 GPU 上训练深度学习模型的流行方法。对于单个模型无法适应 GPU 的更高级用例，您还可以考虑 PyTorch 的完全分片数据并行（FSDP）方法，该方法执行分布式数据并行并在不同的 GPU 上分配大型层。更多信息，请参阅以下概述，其中包含进一步链接到 API 文档：

+   “介绍 PyTorch 完全分片数据并行（FSDP）API，” [`mng.bz/EZJR`](https://mng.bz/EZJR)
