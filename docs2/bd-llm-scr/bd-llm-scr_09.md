# 附录 C 练习解答

练习答案的完整代码示例可以在补充的 GitHub 仓库[`github.com/rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch)中找到。

## 第二章

### 练习 2.1

您可以通过每次提示编码器一个字符串来获取单个标记 ID：

```py
print(tokenizer.encode("Ak"))
print(tokenizer.encode("w"))
# ...
```

这将打印

```py
[33901]
[86]
# ...
```

您可以使用以下代码组装原始字符串：

```py
print(tokenizer.decode([33901, 86, 343, 86, 220, 959]))
```

这将返回

```py
'Akwirw ier'
```

### 练习 2.2

代码示例，数据加载器`max_length=2`和`stride=2`：

```py
dataloader = create_dataloader(
    raw_text, batch_size=4, max_length=2, stride=2
)
```

它产生以下格式的批次：

```py
tensor([[  40,  367],
        [2885, 1464],
        [1807, 3619],
        [ 402,  271]])
```

第二个数据加载器的代码，其中`max_length=8`和`stride=2`：

```py
dataloader = create_dataloader(
    raw_text, batch_size=4, max_length=8, stride=2
)
```

一个示例批次看起来像

```py
tensor([[   40,   367,  2885,  1464,  1807,  3619,   402,   271],
        [ 2885,  1464,  1807,  3619,   402,   271, 10899,  2138],
        [ 1807,  3619,   402,   271, 10899,  2138,   257,  7026],
        [  402,   271, 10899,  2138,   257,  7026, 15632,   438]])
```

## 第三章

### 练习 3.1

正确的权重分配是

```py
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
```

### 练习 3.2

为了达到 2 的输出维度，类似于我们在单头注意力中拥有的，我们需要将投影维度`d_out`更改为 1。

```py
d_out = 1
mha = MultiHeadAttentionWrapper(d_in, d_out, block_size, 0.0, num_heads=2)
```

### 练习 3.3

最小 GPT-2 模型的初始化如下：

```py
block_size = 1024
d_in, d_out = 768, 768
num_heads = 12
mha = MultiHeadAttention(d_in, d_out, block_size, 0.0, num_heads)
```

## 第四章

### 练习 4.1

我们可以如下计算前馈和注意力模块中的参数数量：

```py
block = TransformerBlock(GPT_CONFIG_124M)

total_params = sum(p.numel() for p in block.ff.parameters())
print(f"Total number of parameters in feed forward module: {total_params:,}")

total_params = sum(p.numel() for p in block.att.parameters())
print(f"Total number of parameters in attention module: {total_params:,}")
```

如我们所见，前馈模块包含大约是注意力模块两倍多的参数：

```py
Total number of parameters in feed forward module: 4,722,432
Total number of parameters in attention module: 2,360,064
```

### 练习 4.2

要实例化其他 GPT 模型大小，我们可以修改配置字典，如下所示（此处以 GPT-2 XL 为例）：

```py
GPT_CONFIG = GPT_CONFIG_124M.copy()
GPT_CONFIG["emb_dim"] = 1600
GPT_CONFIG["n_layers"] = 48
GPT_CONFIG["n_heads"] = 25
model = GPTModel(GPT_CONFIG)
```

然后，重新使用第 4.6 节中的代码来计算参数数量和 RAM 需求，我们发现

```py
gpt2-xl:
Total number of parameters: 1,637,792,000
Number of trainable parameters considering weight tying: 1,557,380,800
Total size of the model: 6247.68 MB
```

### 练习 4.3

在第四章中，我们使用 dropout 层的三个不同位置：嵌入层、快捷层和多头注意力模块。我们可以通过在配置文件中单独编码每个层的 dropout 率，然后相应地修改代码实现来控制每个层的 dropout 率。

修改后的配置如下：

```py
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_attn": 0.1,      #1
    "drop_rate_shortcut": 0.1,      #2
    "drop_rate_emb": 0.1,      #3
    "qkv_bias": False
}
```

#1 多头注意力中的 Dropout

#2 快捷连接中的 Dropout

#3 嵌入层中的 Dropout

修改后的`TransformerBlock`和`GPTModel`看起来像

```py
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate_attn"],      #1
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(        #2
            cfg["drop_rate_shortcut"]           #2
        )                                       #2

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"]
        )
        self.pos_emb = nn.Embedding(
            cfg["context_length"], cfg["emb_dim"]
        )
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"])    #3

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logitss
```

#1 多头注意力中的 Dropout

#2 快捷连接中的 Dropout

#3 嵌入层中的 Dropout

## 第五章

### 练习 5.1

我们可以使用在本节中定义的`print_sampled_tokens`函数打印标记（或单词）“pizza”被采样的次数。让我们从第 5.3.1 节中定义的代码开始。

当温度为 0 或 0.1 时，“pizza”标记被采样 0x，当温度提升到 5 时，它被采样 32×。估计概率为 32/1000 × 100% = 3.2%。

实际概率为 4.3%，包含在重新缩放的 softmax 概率张量（`scaled_probas[2][6]`）中。

### 练习 5.2

Top-k 采样和温度缩放是根据 LLM 和期望的输出多样性和随机程度进行调整的设置。

当使用相对较小的 top-k 值（例如，小于 10）并且温度设置在 1 以下时，模型的输出变得更加不随机和确定。这种设置在需要生成的文本更加可预测、连贯，并且基于训练数据更接近最可能的结果时很有用。

这种低 k 和温度设置的应用包括生成正式文件或报告，其中清晰度和准确性最为重要。其他应用示例包括技术分析或代码生成任务，其中精度至关重要。此外，问答和教育内容需要准确的答案，其中温度低于 1 有助于提高准确性。

另一方面，较大的 top-k 值（例如，20 到 40 范围内的值）以及超过 1 的温度值，在用 LLM 进行头脑风暴或生成创意内容（如小说）时很有用。

### 练习 5.3

有多种方法可以通过 `generate` 函数强制实现确定性行为：

1.  设置为 `top_k=None` 并不应用温度缩放

1.  设置 `top_k=1`

### 练习 5.4

实质上，我们必须加载我们在主要章节中保存的模型和优化器：

```py
checkpoint = torch.load("model_and_optimizer.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

然后，使用 `num_epochs=1` 调用 `train_simple_function` 以训练模型另一个周期。

### 练习 5.5

我们可以使用以下代码来计算 GPT 模型的训练和验证集损失：

```py
train_loss = calc_loss_loader(train_loader, gpt, device)
val_loss = calc_loss_loader(val_loader, gpt, device)
```

1240 万参数的结果损失如下：

```py
Training loss: 3.754748503367106
Validation loss: 3.559617757797241
```

主要观察结果是训练和验证集的性能在同一水平。这可能有多个解释：

1.  当 OpenAI 训练 GPT-2 时，“The Verdict” 并不是预训练数据集的一部分。因此，模型并没有明确地过度拟合训练集，在 “The Verdict” 的训练和验证集部分表现相似。（验证集损失略低于训练集损失，这在深度学习中是不常见的。然而，这很可能是由于随机噪声，因为数据集相对较小。在实践中，如果没有过度拟合，训练和验证集的性能预计将大致相同）。

1.  “The Verdict” 是 GPT-2 训练数据集的一部分。在这种情况下，我们无法判断模型是否过度拟合了训练数据，因为验证集也已被用于训练。为了评估过度拟合的程度，我们需要一个在 OpenAI 完成训练 GPT-2 后生成的新的数据集，以确保它不可能成为预训练的一部分。

### 练习 5.6

在主要章节中，我们尝试了最小的 GPT-2 模型，它只有 1240 万个参数。这样做的原因是为了尽可能降低资源需求。然而，你可以通过最小的代码更改轻松地尝试更大的模型。例如，在第五章中，我们不是加载 1240 万个参数的模型权重，而是只需要更改以下两行代码：

```py
hparams, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
model_name = "gpt2-small (124M)"
```

更新后的代码如下

```py
hparams, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2")
model_name = "gpt2-xl (1558M)"
```

## 第六章

### 练习 6.1

我们可以通过将最大长度设置为`max_length` `=` `1024`来初始化数据集，从而将输入填充到模型支持的标记数最大值：

```py
train_dataset = SpamDataset(..., max_length=1024, ...)
val_dataset = SpamDataset(..., max_length=1024, ...)
test_dataset = SpamDataset(..., max_length=1024, ...)
```

然而，额外的填充导致测试准确率显著下降至 78.33%（与主章节中的 95.67%相比）。

### 练习 6.2

我们不仅可以微调最终的 transformer 块，还可以通过从代码中删除以下行来微调整个模型：

```py
for param in model.parameters():
    param.requires_grad = False
```

这种修改使测试准确率提高了 1%，达到 96.67%（与主章节中的 95.67%相比）。

### 练习 6.3

我们可以不是微调最后一个输出标记，而是通过将代码中的`model(input_batch)[:, -1, :]`更改为`model(input_batch)[:, 0, :]`来微调第一个输出标记。

如预期的那样，由于第一个标记包含的信息少于最后一个标记，这种变化导致测试准确率显著下降至 75.00%（与主章节中的 95.67%相比）。

## 第七章

### 练习 7.1

Phi-3 提示格式，如图 7.4 所示，对于给定的示例输入如下所示：

```py
<user>
Identify the correct spelling of the following word: 'Occasion'

<assistant>
The correct spelling is 'Occasion'.
```

要使用此模板，我们可以按如下方式修改`format_input`函数：

```py
def format_input(entry):
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )
    input_text = f"\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
```

最后，当我们收集测试集响应时，我们还需要更新提取生成响应的方式：

```py
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    tokenizer=tokenizer
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (                       #1
        generated_text[len(input_text):]
        .replace("<|assistant|>:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text
```

#1 新：调整###Response 到<|assistant|>

使用 Phi-3 模板微调模型大约快 17%，因为它导致模型输入更短。分数接近 50，这与我们之前使用 Alpaca 风格提示获得的分数在同一水平。

### 练习 7.2

为了屏蔽如图 7.13 所示的指令，我们需要对`InstructionDataset`类和`custom_collate_fn`函数进行轻微修改。我们可以修改`InstructionDataset`类以收集指令长度，我们将在 collate 函数中使用这些长度来定位目标中的指令内容位置，如下所示：

```py
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.instruction_lengths = []     #1
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\\n### Response:\\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
            instruction_length = ( 
                len(tokenizer.encode(instruction_plus_input)
            )
            self.instruction_lengths.append(instruction_length)      #2

    def __getitem__(self, index):    #3
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
```

#1 分离的指令长度列表

#2 收集指令长度

#3 分别返回指令长度和文本

接下来，我们更新了`custom_collate_fn`，由于`InstructionDataset`数据集的变化，现在每个`batch`都是一个包含`(instruction_length, item)`的元组，而不是仅仅`item`。此外，我们现在在目标 ID 列表中屏蔽相应的指令标记：

```py
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):

    batch_max_length = max(len(item)+1 for instruction_length, item in batch)
    inputs_lst, targets_lst = [], []          #1

    for instruction_length, item in batch:   
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item)
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        targets[:instruction_length-1] = -100       #2

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor
```

#1 现在 batch 是一个元组。

#2 在目标中屏蔽所有输入和指令标记

当使用这种指令屏蔽方法微调模型进行评估时，其表现略差（使用第七章的 Ollama Llama 3 方法，大约低 4 分）。这与“带有指令损失的指令调整”论文中的观察结果一致([`arxiv.org/abs/2405.14394`](https://arxiv.org/abs/2405.14394))。

### 练习 7.3

要在原始斯坦福 Alpaca 数据集([`github.com/tatsu-lab/stanford_alpaca`](https://github.com/tatsu-lab/stanford_alpaca))上微调模型，我们只需更改文件 URL：

```py
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
```

到

```py
url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
```

注意，数据集包含 52,000 条条目（比第七章多 50 倍），条目长度也比我们在第七章中处理的条目长。

因此，强烈建议在 GPU 上运行训练。

如果你遇到内存不足的错误，考虑将批大小从 8 减少到 4、2 或 1。除了降低批大小外，你还可能希望考虑将 `allowed_max_length` 从 1024 降低到 512 或 256。

下面是来自 Alpaca 数据集的一些示例，包括生成的模型响应：

### 练习 7.4

要使用 LoRA 指令微调模型，请使用附录 E 中的相关类和函数：

```py
from appendix_E import LoRALayer, LinearWithLoRA, replace_linear_with_lora
```

接下来，在 7.5 节中的模型加载代码下方添加以下几行代码：

```py
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters before: {total_params:,}")

for param in model.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after: {total_params:,}")
replace_linear_with_lora(model, rank=16, alpha=16)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")
model.to(device)
```

注意，在 Nvidia L4 GPU 上，使用 LoRA 进行微调在 L4 上运行需要 1.30 分钟。在相同的 GPU 上，原始代码运行需要 1.80 分钟。因此，在这种情况下，LoRA 大约快了 28%。使用第七章的 Ollama Llama 3 方法评估的分数大约为 50，与原始模型相当。

## 附录 A

### 练习 A.1

可选的 Python 设置提示文档([`github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences`](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences)) 包含了额外的建议和提示，如果你需要额外的帮助来设置你的 Python 环境。

### 练习 A.2

可选的 "安装本书中使用的库" 文档([`github.com/rasbt/LLMs-from-scratch/tree/main/setup/02_installing-python-libraries`](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/02_installing-python-libraries)) 包含了检查你的环境是否正确设置的实用工具。

### 练习 A.3

网络有两个输入和两个输出。此外，还有 2 个隐藏层，分别有 30 和 20 个节点。从编程的角度来看，我们可以这样计算参数数量：

```py
model = NeuralNetwork(2, 2)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)
```

这将返回

```py
752
```

我们也可以手动计算如下：

+   *第一隐藏层:* 2 个输入乘以 30 个隐藏单元加上 30 个偏置单元

+   *第二隐藏层:* 30 个输入单元乘以 20 个节点加上 20 个偏置单元

+   *输出层:* 20 个输入节点乘以 2 个输出节点加上 2 个偏置单元

然后，将每层的所有参数相加，结果为 2×30+30 + 30×20+20 + 20×2+2 = 752。

### 练习 A.4

确切的运行时间结果将取决于用于此实验的硬件。在我的实验中，即使对于小的矩阵乘法，使用连接到 V100 GPU 的 Google Colab 实例也能观察到显著的加速：

```py
a = torch.rand(100, 200)
b = torch.rand(200, 300)
%timeit a@b
```

在 CPU 上这导致了

```py
63.8 µs ± 8.7 µs per loop
```

在 GPU 上执行时

```py
a, b = a.to("cuda"), b.to("cuda")
%timeit a @ b
```

结果如下

```py
13.8 µs ± 425 ns per loop
```

在这种情况下，在 V100 上，计算速度大约快了四倍。
