# 第六章\. 微调

在上一章中，我们讨论了在选择适合你特定需求的 LLM 时需要考虑的各种因素，包括如何评估 LLMs 以便做出明智选择的一些提示。接下来，让我们利用这些 LLMs 来解决我们的任务。

在本章中，我们将探讨使用微调使 LLM 适应解决你感兴趣的任务的过程。我们将通过一个完整的微调示例来展示，涵盖一个人需要做出的所有重要决策。我们还将讨论创建微调数据集的艺术和科学。

# 微调的必要性

我们为什么需要微调 LLMs？为什么一个经过预训练且带有少量提示的 LLM 不足以满足我们的需求？让我们通过几个例子来说明这一点：

用例 1

假设你正在处理一个相当古怪的任务，即检测文本中所有用过去时态写的句子并将它们转换为将来时态。为了解决这个问题，你可能会提供一些过去时态句子的例子以及表示过去时态及其对应将来时态句子的输入-输出对。然而，LLM 似乎无法满足你的要求，在识别和转换步骤中都犯了错误。作为回应，你详细说明了你的指令，将英语语法规则和例外情况添加到你的提示中。你注意到性能有所提高。但随着每条新规则的添加，你的提示逐渐膨胀，慢慢地变成了一本语法迷你书。

如我们在第五章中看到的，LLM 只能遵循提示中的有限指令集，其有效上下文窗口远小于宣传的上下文窗口。我们已经陷入了僵局。

用例 2

考虑一个处理从金融文本中回答问题的任务。大型语言模型（LLMs）并非金融专家，难以处理金融术语。为了解决这个问题，你在提示中添加了关键金融术语的定义。虽然你注意到性能有轻微的提升，但很快你就会意识到，为了实现预期的收益，你需要将整个注册会计师（CPA）考试的整个课程塞进你那微不足道的上下文窗口中。

这就是微调发挥作用的地方。通过提供一个输入-输出对的数据集，模型通过更新其权重来学习输入-输出映射，你可以完成仅凭上下文学习无法完成的任务。对于上述提到的两个任务，微调模型大大提高了性能。

何时不应使用微调？如果你的主要目标是向语言模型传授新的或更新的事实或知识，那么像 RAG 这样的技术会更好，我们将在第十章和 12 章中探讨这些技术。微调最适合需要模型学习特定的输入-输出映射、熟悉新的文本领域或展示更复杂的能力和行为的情况。

###### 警告

回想一下第五章，更新语言模型的参数可能会使基础模型的能力下降！在一个特定任务上微调模型可能会无意中导致基础模型在其他任务上的表现变差。请谨慎处理。

# 微调：一个完整示例

让我们从头到尾走一遍一个实际的微调示例。我们希望训练一个*政治承诺检测器*，它可以用来识别执政党代表在竞选演讲或议会程序中做出的承诺。我们定义政治承诺为具体、明确，并且政府有权力采取的行动。

这样的句子例子是：“我们将在未来十年内建设 10,000 公里的地铁线路。”

然而，并非所有未来时态或前瞻性陈述都是承诺。以下句子根据我们的定义不是承诺：

+   “我们预计日本明年将提高关税。”（预期，而不是政府可以控制的事情）

+   “我们将努力使加拿大成为一个更好的地方。”（没有提供具体信息）

+   “AI 将在明年导致一百万个工作岗位的丧失。”（预测，不是承诺）

我们的基础 LLM，Llama2-7B，在上下文学习设置中难以准确识别此类承诺。因此，我们将针对这个特定任务对其进行微调。然后我们可以使用生成的模型来检测政治承诺，并将这些承诺与结构化数据集或预算文本进行匹配，以跟踪这些承诺在一段时间内是否得到履行。

为了这个目的，我已经构建了一个包含承诺和陈述示例的合成微调数据集。在本章的后面部分，我们将介绍创建此类数据集的过程。

幸运的是，由于存在几个简化微调过程的库，今天的微调变得更加容易。其中最重要的库包括[Transformers](https://oreil.ly/BTi76)、[Accelerate](https://oreil.ly/W8oLi)、[PEFT](https://oreil.ly/QbQoq)、[TRL](https://oreil.ly/Ya9Xj)和[bitsandbytes](https://oreil.ly/ruVEX)。前四个来自 Hugging Face。你已经在之前的章节中遇到了许多这些库。熟悉这些库的内部工作原理是一项非常有用的技能。

###### 小贴士

由于这些库相对较新，并且属于快速发展的领域，它们经常经历重大更新。我建议关注这些库的主要更新，因为它们继续引入将简化你工作流程的增强功能。

让我们从加载数据集开始。自定义数据集可以从本书的 [GitHub 仓库](https://oreil.ly/llm-playbooks) 下载：

```py
from datasets import load_dataset
tune_data = load_dataset("csv", data_files='/path/to/finetune_data.csv'
```

###### 小贴士

我强烈推荐使用 [*datasets* 库](https://oreil.ly/3LX5X) 来加载你的训练和微调数据集，因为它是一个高效加载大型数据集的绝佳抽象，可以抽象出内存管理细节。

接下来，让我们通过 `TrainingArguments` 类在 `Transformers` 库中设置一些相关的超参数：

```py
# Make sure you have installed the correct version
!pip install transformers==4.35.0

from transformers import TrainingArguments
```

可用超过一百个参数；我们将讨论其中重要的参数。这些参数与使用的学习算法、内存和空间优化、量化、正则化和分布式训练相关。让我们详细探讨这些内容。

## 学习算法参数

让我们探索用于训练网络的优化算法，并学习如何为我们的目的选择正确的优化器。

### 优化器

AdamW 和 Adafactor 目前是最常用的优化器。其他流行的优化算法包括随机梯度下降（SGD）、RMSProp、Adagrad、Lion 以及它们的变体。有关优化算法的更多背景信息，请参阅 Florian June 的 [博客文章](https://oreil.ly/VTiDa)。

Adafactor 和 SGD 每个参数使用 4 个字节的内存，而 AdamW 每个参数使用 8 个字节的内存。这意味着使用 AdamW 优化器进行完全微调的 7B 模型需要 7 * 8 = ~56GB 的内存来存储优化器状态。存储参数、梯度和前向激活还需要更多的内存。

最近，[8 位优化器](https://oreil.ly/4Z14D) 已被引入，它们可以对优化器状态进行量化。使用 AdamW 8 位版本进行完全微调的 7B 模型只需要大约 ~14GB 的内存来存储优化器状态。

这些 8 位优化器通过 bitsnbytes 库提供，并且也得到 Hugging Face 的支持。要使用 8 位 AdamW 版本，你可以在 `TrainingArguments` 中设置：

```py
optim = 'adamw_bnb_8bit'
```

对于所有通过 Hugging Face 直接可用的优化器选项，请参阅 [OptimizerNames 类](https://oreil.ly/7kdSO)。

###### 小贴士

在他的基准测试实验中，Stas Bekman [显示](https://oreil.ly/0_0lt)令人惊讶的是，8 位 AdamW 优化器实际上比标准 AdamW 优化器更快。他的实验还显示，Adafactor 整体上略慢于 AdamW。

Hugging Face `TrainingArguments` 类中提供的默认优化器是 AdamW。对于大多数情况，默认优化器工作得很好。然而，如果它不起作用，你可以尝试 Adafactor 和 Lion。对于强化学习，SGD 似乎效果不错。

如果你特别内存受限，8 位 AdamW 是一个不错的选择。如果可用，这些优化器的分页版本将进一步减轻你的内存需求。

### 学习率

对于每个优化器，某些学习率已被证明非常有效。AdamW 推荐的学习率是 1e-4，权重衰减为 0.01。权重衰减是一种正则化技术，有助于减少过拟合。同样，对于像*adam_beta1*、*adam_beta2*和*adam_epsilon*这样的次要优化器参数，默认值已经足够好，无需更改。

### 学习计划

在训练过程的后期，降低学习率是一个好主意，因为你不想在接近收敛时 overshoot。在类似的情况下，你希望防止你的模型从第一批示例中学习太多。在任何情况下，我们都希望能够在训练过程中自动调整学习率。为了实现这一点，我们可以使用学习计划。

Hugging Face 支持多种不同类型的调度器。以下是一些重要的调度器：

常数

这是一种纯训练计划，其中学习率在整个训练过程中保持不变。

带有预热的常数

在这种设置中，学习率从零开始，在预热阶段线性增加到指定的学习率。预热阶段完成后，学习率保持不变。

图 6-1 展示了在使用带有预热常数的调度器时，学习率随时间的变化情况。

![constant-lr](img/dllm_0601.png)

###### 图 6-1。带有预热常数的调度器的学习率

余弦

在这种称为*余弦退火*的设置中，学习率在预热阶段之后会按照余弦函数缓慢下降到零。

图 6-2 展示了在使用余弦调度器时，学习率随时间的变化情况。

![cosine-warmup](img/dllm_0602.png)

###### 图 6-2。带有余弦调度器的学习率

带有重启的余弦

在这种称为*余弦退火带预热重启*的设置中，在预热阶段之后，学习率按照余弦函数下降到零，但会经历几次硬重启，学习率在达到零后迅速回到指定的学习率。关于为什么这种方法有效的更多细节，请参阅介绍了这一概念的 Loshcilov 和 Hutter 的[论文](https://oreil.ly/Q4c3o)。

图 6-3 展示了在使用带有重启的余弦调度器时，学习率随时间的变化情况。

![cosine-restart](img/dllm_0603.png)

###### 图 6-3。带有带有重启余弦调度器的学习率

线性

这与余弦设置非常相似，不同之处在于学习率是线性下降到零，而不是遵循余弦函数。

图 6-4 展示了在使用线性调度器时，学习率随时间的变化情况。

![线性学习率](img/dllm_0604.png)

###### 图 6-4. 线性调度器的学习率

如果你使用 AdamW，具有预热阶段的调度器甚至更加重要，以防止陷入局部最小值。经验上发现，余弦退火优于线性衰减。

对于我们的政治承诺检测器微调，让我们使用 AdamW 的分页变体，学习率为 3e-4，权重衰减为 0.01，并采用余弦学习计划：

```py
optim = "paged_adamw_32bit"
learning_rate = 3e-4
weight_decay = 0.01
lr_scheduler_type = 'cosine'
warmup_ratio = 0.03  #The proportion of training steps to be used as warmup
```

## 内存优化参数

在我们设置了与优化器相关的参数之后，让我们探索内存和计算优化参数。这个领域中的两种流行技术包括梯度检查点和梯度累积。

### 梯度检查点

梯度检查点技术以增加计算成本为代价来帮助节省内存。在反向传播算法的前向传递过程中，激活值被计算并保存在内存中，以便在反向传递中使用。如果我们没有保存所有的激活值怎么办？缺失的激活值可以在反向传递过程中即时重新计算。这确实会增加我们的计算成本，但我们可以节省大量的内存。我们甚至可以训练那些批次大小仅为一个无法适应我们 GPU 内存的模型。有关梯度检查点的更多技术细节，请参阅 Yaroslav Bulatov 的[博客](https://oreil.ly/i-R4I)。

### 梯度累积

假设我们有一个期望的批次大小，但我们没有足够的内存来支持该批次大小。我们可以使用一种称为梯度累积的技术来模拟期望的批次大小。在这种技术中，梯度更新不是在每个批次都进行，而是在几个批次中累积，然后求和或平均。

###### 注意

梯度累积可能会使训练变慢，因为更新的次数较少。梯度累积不会减少所需的计算量。

### 量化

通过量化来节省内存是一种非常有效的方法，如第五章（ch05.html#chapter_utilizing_llms）中介绍的那样。我们将在第九章（ch09.html#ch09）中更详细地介绍量化技术。对于我们的用例，我们将使用 bf16，因为它在内存节省和性能之间提供了一个合理的权衡。

对于我们的政治承诺检测器微调，鉴于我们试图在一个相对内存受限的 16 GB RAM GPU 上训练它，我们将设置以下参数以进行内存优化：

```py
gradient_accumulation_steps = 4
bf16 = True
gradient_checkpointing = True
```

## 正则化参数

接下来，让我们看看可用于解决模型过拟合的各种技术。

### 标签平滑

标签平滑是一种技术，它不仅有助于对抗过拟合，还有助于模型校准。

校准是深度学习中一个被低估的话题。如果一个模型其输出概率值与任务准确性之间存在相关性，则称该模型校准良好。

例如，考虑一个将句子分类为侮辱性或非侮辱性的任务。如果模型校准良好，那么在模型输出概率为 0.9 的所有示例中，预期有 90%会被正确分类。同样，对于输出概率为 0.6 的情况，分类正确的可能性应该更低（约 60%）。简单来说，输出概率应该准确地反映对分类决策的信心。

一个模型校准良好意味着它不是过度自信的。这有助于我们细致地处理输出概率较低的示例（例如，使用更大的模型来处理这些示例）。

###### 注意

根据 Li 等人的一项研究，与 BERT 等模型相比，更大的模型校准度较低。更大的模型通常对其预测更有信心。无法为大型语言模型计算合理的准确不确定性估计可能是使用较小模型的一个论据！

校准模型的技术之一是标签平滑。通常的训练过程涉及针对硬目标标签（二分类任务中的 0 或 1）进行训练。当使用交叉熵作为损失函数时，这相当于将模型的对数几率推向 0 或 1，从而使模型高度自信。标签平滑涉及使用一个正则化项，该项从硬目标标签中减去或除以。

标签平滑在输入数据集噪声较大时特别有用，即包含一些不准确标签。正则化可以防止模型从错误示例中学习过多。

对于政治承诺检测器，鉴于一些示例可能是主观的或可解释的，我们将使用标签平滑。

### 噪声嵌入

我们用于微调的数据集通常包含少量示例（< 50,000）。我们希望我们的模型不要过度拟合数据集的风格特征，如格式、措辞和文本长度。解决这一问题的方法之一是在输入嵌入中添加噪声。

[Jain 等人](https://oreil.ly/ouESL)观察到，添加噪声嵌入可以减少模型过度拟合微调数据集的措辞和格式。噪声嵌入的一个有趣副作用是模型生成的文本更长，更冗长。通过测量输出标记的多样性，他们证实了较长的文本实际上包含更多信息，而不仅仅是重复。

Hugging Face 支持[噪声嵌入指令微调（NEFTune）](https://oreil.ly/dSaem），这是一种噪声添加技术。在 NEFTune 中，每个嵌入向量都添加一个噪声向量。噪声向量中的元素是通过从[-1,1]中独立同分布（iid）采样生成的。在将其添加到嵌入向量之前，使用缩放因子对结果向量进行缩放。

噪声嵌入在经验上已被证明在减少过拟合方面非常有效。因此，我们将使用它来对我们的政治承诺检测器微调。请注意，噪声嵌入仅在训练期间添加，而不是在推理期间。

###### 警告

噪声嵌入的影响尚未得到充分理解。微调任务的改进可能会以牺牲其他模型能力为代价。确保测试模型以检查回归！

对于我们的政治承诺检测器微调任务，让我们激活标签平滑和噪声嵌入：

```py
# Label 0 will be transformed to label_smoothing_factor/num_labels
# Label 1 will be transformed to 1 - label_smoothing_factor +
#label_smoothing_factor/num_labels

label_smoothing_factor = 0.1
neftune_noise_alpha = 5
```

## 批处理大小

除了学习率外，批处理大小是我们需要设置的最重要超参数之一。较大的批处理大小意味着训练将更快进行。然而，较大的批处理大小也需要更多的内存。较大的批处理大小也可能导致模型陷入尖锐的局部最小值，这可能是过拟合的迹象。因此，涉及内存、计算和性能的权衡。

对于政治承诺检测器，鉴于我们的内存限制，我们将使用 8 个批处理大小。当然，在推理期间，最大的可能批处理大小是理想的。请注意，建议批处理大小始终是 2 的幂，以减少 GPU I/O 开销。

Hugging Face 的`TrainingArguments`类支持`*auto_find_batch_size*`，当设置时，选择由内存支持的最大的可能批处理大小。要使用此功能，您需要安装`accelerate`库：

```py
per_device_train_batch_size = 8
per_device_eval_batch_size = 8
```

###### 小贴士

您可以将最大序列长度减少以支持更大的批处理大小。

最后，让我们设置一些其他参数：

`max_grad_norm`

这用于梯度裁剪，它是解决训练过程中有时遇到的梯度爆炸问题的解决方案。`max_grad_norm`的值是梯度裁剪的阈值。如果 L2 梯度范数高于阈值，则将其重新缩放到`max_grad_norm`。有关梯度裁剪的更多详细信息，请参阅[“理解梯度裁剪（以及它是如何修复梯度爆炸问题的）”](https://oreil.ly/gH7L7)。

`group_by_length`

这用于将具有相似长度的示例分组到同一个批处理中，以便优化填充标记。

`max_train_epochs`

在训练数据集上的遍历次数。这通常设置为少于五次，以防止过拟合：

```py
max_grad_norm=2
group_by_length=True
max_train_epochs=3
```

## 参数高效微调

在填写完`TrainingArguments`之后，让我们接下来填写 PEFT 库的参数。

Hugging Face 的 PEFT 库是一个令人印象深刻的参数高效微调的促进者。这指的是一组微调技术，这些技术只更新模型中一小部分参数，同时保持性能接近如果更新所有参数将会有的性能。

在本例中，我们将使用低秩适应（LoRA）作为微调技术。以下是一些需要考虑的超参数：

`r`

LoRA 的注意力维度。

`lora_alpha`

LoRA 技术中的 alpha 参数。

`lora_dropout`

在正在调整的层中使用的丢弃概率。这有助于减少过拟合。

`layers_to_transform`

这指定了要应用 LoRA 变换的层。

这里有一些推荐的默认值：

```py
r = 64
lora_alpha = 8
lora_dropout = 0.1
```

关于 LoRA 的更多背景信息，请参阅 Ogban Ugot 的[博客文章](https://oreil.ly/_l91y)。

## 使用降低精度

由 Tim Dettmers 构建的 bitsandbytes 库，简化了使用降低精度格式的操作，我们在第五章中介绍了这些格式。在这个例子中，我们将使用 FP4 格式。请注意，你需要 bitsandbytes 版本 >= 0.39.0。

Hugging Face 已将其生态系统中的 bitsandbytes 支持集成。`BitsAndBytesConfig`类允许我们设置参数。以下是一些相关的参数：

`load_in_8bit/load_in_4bit`

这用于指定我们是否要以 4 位模式或 8 位模式加载模型。

`llm_int8_threshold`

我们需要指定一个阈值，超过该阈值将使用 fp16。这是因为 int8 量化仅适用于小于 5-6 的值。

`llm_int8_skip_modules`

这用于指定我们不想进行 int8 量化的异常。

`llm_int8_enable_fp32_cpu_offload`

如果我们想在 GPU 上以 int8 运行模型的一部分，而在 CPU 上以 FP32 运行其余部分，此参数可以简化操作。这在模型太大而无法适应我们的 GPU 时使用。

`bnb_4bit_compute_dtype`

这设置计算类型，而不管输入类型如何。

`bnb_4bit_quant_type`

这里的选项是 FP4 或 NF4。这用于在 4 位层中设置量化类型。

这里有一些推荐的默认值：

```py
use_4bit = True
bnb_4bit_compute_dtype = 'float16'
bnb_4bit_quant_type = 'nf4'
use_nested_quant = False
```

最后，我们使用 Transformer Reinforcement Learning (TRL)库，除了强化学习之外，还提供了监督微调的支持。

这里有一些推荐的默认值：

```py
max_seq_length = 128
# Packing is used to place multiple instructions in the same input sequence

packing = True
```

## 整合所有内容

现在我们已经设置了所有必要的参数，以下是微调过程的完整代码：

```py
# Ensure that the specified versions of these libraries are installed.
!pip install transformers==4.35.0 accelerate==0.24.0 peft==0.6.0
bitsandbytes==0.41.0  trl==0.7.4

from datasets import load_dataset
from transformers import TrainingArguments, BitsAndBytesConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, LoraConfig
from trl import SFTTrainer

train_params = TrainingArguments(
    optim = "paged_adamw_32bit",
    learning_rate = 3e-4,
    weight_decay = 0.01,
    lr_scheduler_type = 'cosine',
    warmup_ratio = 0.03,
    gradient_accumulation_steps = 4,
    bf16 = True,
    gradient_checkpointing = True,
    label_smoothing_factor = 0.1,
    neftune_noise_alpha = 5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    max_grad_norm=2,
    group_by_length=True,
    max_train_epochs=3,
    output_dir = '/model_outputs',
    save_steps = 50,
    logging_steps = 10
    )

quantize_params = BitsAndBytesConfig (

    use_4bit = True,
    bnb_4bit_compute_dtype = 'float16',
    bnb_4bit_quant_type = 'nf4',
    use_nested_quant = False
    )

lora_params = LoraConfig (
    r = 64,
    lora_alpha = 8,
    lora_dropout = 0.1
    )

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path = 'meta-llama/Llama-2-7b',
    quantization_config=quantize_params,
    device_map='auto'
    )

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b')

tune_data = load_dataset("csv", data_files='/path/to/finetune_data.csv')

sft = SFTTrainer (
    model = model,
    args = train_params,
    train_dataset = tune_data,
    tokenizer = tokenizer
    peft_config = lora_params,
    max_seq_length = 128,
    dataset_text_field = 'text',
    packing = True
    )

sft.train()
sft.model.save_pretrained('/path/to/llama-2-it.csv')
```

超参数之间的关系非常复杂，你可能会发现令人惊讶的结果。在找到最佳点之前可能需要多次迭代。然而，不要花太多时间从微调中挤出最后一点性能，因为这段时间最好用于开发更好的训练数据。在下一节中，我们将学习如何创建有效的训练数据集。

微调 LLM 所需的精确内存取决于多个因素：使用的优化器、是否激活梯度累积和梯度检查点、使用的量化类型等。

# 微调数据集

在我们的微调示例中，我们直接加载了一个预构建的数据集，主要关注微调过程。现在，让我们将注意力转向数据集，了解创建数据集的各种技术。

首先，让我们看看我们在微调示例中使用的数据集：

```py
from datasets import load_dataset
tune_data = load_dataset("csv", data_files='/path/to/finetune_data.csv')
print(tune_data[:2])
```

输出：

```py
Input: We will support women and children and give every child the best
possible start with $10 a day child care.
Identify if the above sentence represents a political promise. A political
promise is a promise that is tangible, specific, and an action that the
government has the agency to make. Reply 'True' if the sentence represents a
political promise, 'False' if not.
Output: True
Input: It is time for leadership that never seeks to divide Canadians, but
takes every single opportunity to bring us together, including in Parliament.
Identify if the above sentence represents a political promise. A political
promise is a promise that is tangible, specific, and an action that the
government has the agency to make. Reply 'True' if the sentence represents a
political promise, 'False' if not.
Output: False
```

如我们所见，这不仅仅是一个只包含（输入，输出）对的常规数据集，而是一个还包含自然语言中任务描述的数据集。这类微调数据集的一个典型例子包括：

+   指令，它描述了任务并指定了所需的输出格式。可选地，指令包含任务的正面和/或负面示例。它还可以包含需要遵循的约束和例外。

+   可选的输入，在我们的例子中是模型要评估的句子或段落。

+   输出，即任务在指令中指定格式的正确答案。

###### 注意

微调数据集可以是多任务或单任务的。多任务数据集用于指令微调。通常，指令微调可以被视为单任务微调之前的一个中间步骤。例如，你可以使用 T5 语言模型，用 FLAN 对其进行指令微调以创建 FLAN-T5，然后进一步使用你的特定任务数据集对其进行微调。这种方法[显示](https://oreil.ly/e-MVh)比仅在 T5 上直接微调有更好的结果。

在本章的后面部分，我们将学习如何创建特定任务的数据集。首先，让我们看看我们如何创建指令微调数据集。

可用的指令微调 LLM 有很多，既有开源的也有专有的。我们为什么想自己指令微调 LLM 呢？公共数据集过于通用，缺乏多样性，并且主要针对通用用途。利用你的领域专业知识和对预期用例的了解来构建数据集可以非常有效。事实上，在我公司，我们专注于金融领域，这项技术带来了单次最大的性能提升。

创建指令微调数据集的方法包括：

+   利用公开可用的指令微调数据集

+   将传统微调数据集转换为指令微调数据集

+   从手动制作的种子示例开始，然后可选地通过利用 LLM 生成类似示例来扩充数据集

接下来，让我们更详细地考察这些方法。

## 利用公开可用的指令微调数据集

如果你的用例足够通用或流行，你可能能够使用公开可用的数据集进行指令微调。以下表格列出了一些流行的指令微调数据集，以及它们创建者、大小和创建过程的信息。

表 6-1. 流行的指令微调数据集

| 名称 | 大小 | 创建者 | 创建方式 |
| --- | --- | --- | --- |
| OIG | 43M | [Ontocord](https://www.ontocord.ai/) | 基于规则 |
| FLAN | 4.4M | Google | 模板 |
| P3（公共提示池） | 12M | 大科学 | 模板 |
| 自然指令 | 193K | Allen AI | 模板 |
| 不自然的指令 | 240K | [Honovich 等人](https://github.com/orhonovich/unnatural-instructions), Meta | LLMs |
| LIMA (Less Is More for Alignment) | 1K | [周等人](https://arxiv.org/abs/2305.11206), Meta | 模板 (Templates) |
| Self-Instruct | 52K | [王等人](https://github.com/yizhongw/self-instruct) | 大型语言模型 (LLMs) |
| Evol-Instruct | 52K | [徐等人](https://arxiv.org/abs/2304.12244) | 大型语言模型 (LLMs) |
| InstructWild v2 | 110K | [Ni 等人](https://github.com/XueFuzhao/InstructionWild) | 大型语言模型 (LLMs) |
| Alpaca | 52K | 斯坦福 | 大型语言模型 (LLMs) |
| Guanaco | 534K | [Dettmers 等人](https://arxiv.org/abs/2305.14314) | 大型语言模型 (LLMs) |
| Vicuna | 70K | LMSYS | 人际对话 |
| OpenAssistant | 161K | 开放助手 | 人际对话 |

让我们详细了解一下微调语言网络 (FLAN)，这是最受欢迎的指令微调数据集之一。了解它是如何构建的，将为你提供创建自己指令微调数据集的路线图。大多数公开可用的指令微调数据集旨在增强用于开放性任务的 LLM，而不是特定领域的用例。

FLAN 实际上是一组几个数据集的集合。2022 年发布的[FLAN 集合](https://oreil.ly/SrXV-)由五个组件组成：

+   FLAN 2021

+   T0

+   超自然指令

+   思维链

+   对话

原始 FLAN 2021 数据集是早期指令微调数据集之一，用于训练 FLAN-T5。FLAN 2021 数据集是通过将现有的学术 NLP 数据集转换为使用指令模板的指令格式来构建的。这些模板是手动构建的，每个任务创建了十个模板。模板可在[这里](https://oreil.ly/DNKCv)找到。

下面是一个任务模板列表的示例，它来自 FLAN GitHub 仓库中的[templates.py](https://oreil.ly/DNKCv)文件。我们的示例任务是 CNN/DailyMail 新闻数据集上的文本摘要：

```py
"cnn_dailymail": [
  ("Write highlights for this article:\n\n{text}", "{highlights}"),
  ("Write some highlights for the following article:\n\n{text}", "{highlights}"),
  ("{text}\n\nWrite highlights for this article.", "{highlights}"),
  ("{text}\n\nWhat are highlight points for this article?", "{highlights}"),
  ("{text}\nSummarize the highlights of this article.", "{highlights}"),
  ("{text}\nWhat are the important parts of this article?", "{highlights}"),
  ("{text}\nHere is a summary of the highlights for this article:",
    "{highlights}"),
  ("Write an article using the following points:\n\n{highlights}", "{text}"),
  ("Use the following highlights to write an article:\n\n{highlights}",
    "{text}"),
  ("{highlights}\n\nWrite an article based on these highlights.", "{text}"),
],
```

注意，最后三个指令代表任务的倒置版本，即给定一个摘要，模型被鼓励写出整篇文章。这样做是为了增加大规模指令的多样性。

我们能否通过使用 LLMs 来自动化这些模板的构建过程呢？是的，这是可能的。我们可以利用 LLMs 生成更多样化的模板。当我要求我最喜欢的 LLM 生成与提示中提供的新闻摘要任务模板类似的指令时，它提出了以下内容：

```py
"cnn_dailymail": [
  ("Distill the essence of this article:\n\n{text}", "{highlights}"),
  ("Give a quick rundown of this article's key points:\n\n{text}",
    "{highlights}"),
  ("Summarize the main elements of this text:\n\n{text}", "{highlights}"),
  ("Highlight the primary takeaways from the following:\n\n{text}",
    "{highlights}"),
  ("Extract and summarize the top points of this article:\n\n{text}",
    "{highlights}"),
  ("Condense this article into its most important aspects:\n\n{text}",
    "{highlights}"),
  ("What are the key insights of this article?\n\n{text}", "{highlights}"),
      ],
```

如您所见，生成的模板反映了表达摘要任务的各种方式。

对于分类任务，建议在指令后附加一个 *Options* 子句。这向 LLM 介绍了输出空间，因此可以集中概率质量在定义的标签空间上。没有这种指导，LLM 将将其概率分布到表示相同概念的几个不同标记上，例如在二元分类任务中，有几种不同的方式来表达 *True* 标签。一个示例提示是：“确定这段文字的语气。选项：快乐、悲伤、中性。”

手动构建这些提示可能是一项繁琐的工作。[*promptsource* 工具](https://oreil.ly/WIyOq) 允许您通过图形用户界面工具或通过 promptsource Python 库创建、访问和应用提示。以下是从公共提示池（P3）收集的用于释义任务的示例，由 Big Science 构建，可通过 promptsource 工具获取。P3 提示由输入模板、目标模板和答案选项模板组成：

```py
Input Template:
I want to know whether the following two sentences mean the same thing.
{{sentence1}}
{{sentence2}}
Do they?

Target Template:
{{ answer_choices[label] }}

Answer Choices Template:
no ||| yes
```

FLAN 收集的关键组成部分之一是 [Super-NaturalInstructions 数据集](https://oreil.ly/D_rv_)。这个数据集包含了非常丰富的指令描述，不仅包括任务定义，还包括正例、反例、约束条件和需要注意的事项。答案被丰富了为什么选择该答案的解释。添加解释到答案的有效性尚未确定。

下面是从 Super-NaturalInstructions 数据集中这样一个任务的例子：

```py
Definition
In this task, we ask you convert a data table of restaurant descriptions into
fluent natural-sounding English sentences.
The input is a string of key-value pairs; the output should be a natural and
grammatical English sentence containing all the information from the input.

Positive Example

Input: name[Aromi], eatType[restaurant], food[English], area[city centre]

Output: Aromi is an English restaurant in the city centre.
Explanation: The output sentence faithfully converts the data in the input
into a natural-sounding sentence.

Negative Example
Input: name[Blue Spice], eatType[coffee shop], priceRange[more than 00a330],
customer rating[5 out of 5], ˘
area[riverside], familyFriendly[yes], near[Avalon]
Output: Blue Spice is a Colombian coffee shop located by the riverside, near
Avalon in Boston. Its prices are over
00a330. Its customer ratings are 5 out of 5. ˘

Explanation: While the output contains most of the information from the input,
it hallucinates by adding ungrounded
information such as "Colombian" and "Boston".

Instance Input: name[The Mill], eatType[restaurant], area[riverside], near[The
Rice Boat]

Valid Output: ["A restaurant called The Mill, can be found near the riverside
next to The Rice Boat."]
```

现在我们来看看在 LLM 的帮助下构建的数据集。

## LLM 生成的指令微调数据集

如前所述，手动构建这些数据集可能非常耗时，而释义/合成数据生成正是 LLM 发挥优势的地方。因此，我们可以利用 LLM 来生成我们的指令微调数据集。[Self-Instruct](https://oreil.ly/HVBfK) 和 [Unnatural Instructions 论文](https://oreil.ly/1wV_G) 是这方面的首次尝试。两者都是从一组高质量的手动生成示例开始的，然后在几样本设置中，要求 LLM 生成具有更多样化语言表达的类似示例。

给定一个指令，输入优先和输出优先的组合被证明对生成输入-输出对有益。通常，你会使用输入优先的方法来生成输入-输出对，即要求 LLM 为给定的指令生成一个输入实例，然后要求它为该输入生成输出标签。然而，这种方法可能会导致标签不平衡，如 [Wang 等人](https://oreil.ly/hYFYH) 所示，某些标签被过度表示。因此，混合输出优先的生成方法是一个好方法，即先要求 LLM 生成输出标签，然后要求它生成满足该标签的输入文本。

###### 警告

使用 OpenAI 的输出生成可用于训练竞争模型的训练数据违反了 OpenAI 的政策。虽然有几个使用 GPT-4 合成的公共指令微调数据集，但它们在技术上违反了 OpenAI 的服务条款。我建议使用开源 LLM 来生成合成数据。

简单地要求 LLM 生成与你的种子集相似的示例可能不会给你带来期望的结果。你想要一个多样化但相关的示例集，而且你的 LLM 很容易偏离你的期望分布，最终生成虚假的示例。

###### 注意

你的指令微调数据集应该有多大？[“LIMA: Less Is More for Alignment”论文](https://oreil.ly/z0BWh)显示，你只需要几千个高质量示例就能有效地微调模型。

Xu 等人提出了[Evol-Instruct](https://oreil.ly/9nw3G)，这是一种通过在种子示例上进行有控制的编辑来生成这些合成指令的结构化方法。该过程包括三个步骤：

1.  指令进化：通过深度和广度策略进化种子示例。深度进化通过五种类型的提示增加了原始指令的复杂性和难度：

    +   添加约束

    +   增加推理步骤

    +   提出更深层次的问题

    +   提出更具体的问题

    +   增加输入的复杂性

        在广度上进化通过从与原始指令相同领域的完全新的指令生成来增加主题覆盖范围。

1.  响应生成：对进化指令的响应是通过使用人类或 LLM 生成的。

1.  候选过滤：不符合质量标准的候选实例被过滤掉。你可以使用启发式方法或 LLM 进行候选过滤。

###### 注意

为什么不在指令微调数据集上预训练？如果指令微调是模型预训练后的一个必要步骤，为什么我们不直接使用指令微调数据集来预训练模型？这确实可能，但这些数据集在规模上难以构建，而不会导致质量显著下降。

我们不需要等到有人发布一个大规模数据集，才能在预训练阶段获得指令微调的好处。已经[证明](https://oreil.ly/tfO4a)，在预训练期间混合指令微调数据是有益的。

# 摘要

在本章中，我们强调了需要微调模型以解决更复杂任务的必然性。我们对微调过程进行了深入研究，并突出了在选择超参数时涉及的权衡。我们还展示了指令微调的非凡有效性，并提供了如何创建自己的指令微调数据集的指导。

在下一章中，我们将讨论更新 LLM 参数的更高级技术，包括持续预训练、参数高效的微调和模型合并。
