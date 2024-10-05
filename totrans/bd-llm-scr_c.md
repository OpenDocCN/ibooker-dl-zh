# 附录C. 练习解答

练习答案的完整代码示例可以在补充的GitHub库中找到，网址为[https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。

## C.1 第2章

#### 练习2.1

你可以通过逐次提示编码器输入一个字符串来获得各个标记的ID：

```py
print(tokenizer.encode("Ak"))
print(tokenizer.encode("w"))
# ...
```

这将打印：

```py
[33901]
[86]
# ...
```

然后，你可以使用以下代码组装原始字符串：

```py
print(tokenizer.decode([33901, 86, 343, 86, 220, 959]))
```

这将返回：

```py
'Akwirw ier'
```

#### 练习2.2

数据加载器的代码为`max_length=2 and stride=2`：

```py
dataloader = create_dataloader(raw_text, batch_size=4, max_length=2, stride=2)
```

它生成以下格式的批次：

```py
tensor([[  40,  367],
        [2885, 1464],
        [1807, 3619],
        [ 402,  271]])
```

第二个数据加载器的代码为`max_length=8 and stride=2`：

```py
dataloader = create_dataloader(raw_text, batch_size=4, max_length=8, stride=2)
```

一个示例批次如下所示：

```py
tensor([[   40,   367,  2885,  1464,  1807,  3619,   402,   271],
        [ 2885,  1464,  1807,  3619,   402,   271, 10899,  2138],
        [ 1807,  3619,   402,   271, 10899,  2138,   257,  7026],
        [  402,   271, 10899,  2138,   257,  7026, 15632,   438]])
```

## C.2 第3章

#### 练习3.1

正确的权重分配如下：

```py
sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
```

#### 练习3.2

要实现与单头注意力相似的输出维度为2，我们需要将投影维度`d_out`更改为1。

```py
d_out = 1
mha = MultiHeadAttentionWrapper(d_in, d_out, block_size, 0.0, num_heads=2)
```

#### 练习3.3

最小GPT-2模型的初始化如下：

```py
block_size = 1024
d_in, d_out = 768, 768
num_heads = 12
mha = MultiHeadAttention(d_in, d_out, block_size, 0.0, num_heads)

```

## C.3 第4章

#### 练习4.1

我们可以按如下方式计算前馈和注意力模块中的参数数量：

```py
block = TransformerBlock(GPT_CONFIG_124M)

total_params = sum(p.numel() for p in block.ff.parameters())
print(f"Total number of parameters in feed forward module: {total_params:,}")

total_params = sum(p.numel() for p in block.att.parameters())
print(f"Total number of parameters in attention module: {total_params:,}")
```

正如我们所看到的，前馈模块的参数大约是注意力模块的两倍：

```py
Total number of parameters in feed forward module: 4,722,432
Total number of parameters in attention module: 2,360,064
```

#### 练习4.2

要实例化其他GPT模型的尺寸，我们可以按如下方式修改配置字典（此处显示的是GPT-2 XL）：

```py
GPT_CONFIG = GPT_CONFIG_124M.copy()
GPT_CONFIG["emb_dim"] = 1600
GPT_CONFIG["n_layers"] = 48
GPT_CONFIG["n_heads"] = 25
model = GPTModel(GPT_CONFIG)
```

然后，重用第4.6节的代码来计算参数数量和RAM要求，我们发现如下：

```py
gpt2-xl:
Total number of parameters: 1,637,792,000
Number of trainable parameters considering weight tying: 1,557,380,800
Total size of the model: 6247.68 MB
```

## C.4 第5章

#### 练习5.1

我们可以使用在本节定义的`print_sampled_tokens`函数打印“pizza”标记（或单词）的采样次数。让我们从第5.3.1节中定义的代码开始。

如果温度为0或0.1，则“pizza”标记的采样次数为0x，如果温度提高到5，则采样32×。估计的概率为32/1000 × 100% = 3.2%。

实际概率为4.3%，包含在重新缩放的softmax概率张量中（`scaled_probas[2][6]`）。

#### 练习5.2

Top-k采样和温度缩放是需要根据LLM和输出中期望的多样性及随机性调整的设置。

当使用相对较小的top-k值（例如，小于10）并且温度设置低于1时，模型的输出变得不那么随机，更具确定性。这种设置在我们需要生成的文本更可预测、连贯，并且更接近基于训练数据的最可能结果时非常有用。

这种低k值和低温度设置的应用包括生成正式文档或报告，其中清晰度和准确性至关重要。其他应用示例包括技术分析或代码生成任务，在这些任务中，精确性至关重要。此外，问答和教育内容也需要准确的答案，在这种情况下，温度低于1是有帮助的。

另一方面，当使用 LLM 进行头脑风暴或生成创意内容（如小说）时，较大的 top-k 值（例如 20 到 40 的范围内的值）和温度值高于 1 是有用的。

#### 练习 5.3

有多种方法可以强制 `generate` 函数表现出确定性：

1.  设置为 `top_k=None` 并不进行温度缩放；

1.  设置 `top_k=1`。

#### 练习 5.4

本质上，我们需要加载在主要章节中保存的模型和优化器：

```py
checkpoint = torch.load("model_and_optimizer.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

然后，调用 `train_simple_function`，设置 `num_epochs=1` 来训练模型一个新的周期。

#### 练习 5.5

我们可以使用以下代码计算 GPT 模型的训练集和验证集损失：

```py
train_loss = calc_loss_loader(train_loader, gpt, device)
val_loss = calc_loss_loader(val_loader, gpt, device)
```

124M 参数的损失结果如下：

```py
Training loss: 3.754748503367106
Validation loss: 3.559617757797241
```

主要观察是训练集和验证集的性能在同一水平上。这可能有多种解释。

1.  当 OpenAI 训练 GPT-2 时，《判决》并不是预训练数据集的一部分。因此，模型并没有明显地过拟合训练集，并且在《判决》的训练和验证集部分表现相似。（验证集损失略低于训练集损失，这在深度学习中并不常见。然而，这可能是由于随机噪声，因为数据集相对较小。在实践中，如果没有过拟合，训练集和验证集的性能预计是大致相同的）。

1.  《判决》是 GPT-2 训练数据集的一部分。在这种情况下，我们无法判断模型是否在训练数据上过拟合，因为验证集也会被用于训练。要评估过拟合的程度，我们需要在 OpenAI 完成 GPT-2 训练后生成的新数据集，以确保它不是预训练的一部分。

#### 练习 5.6

在主要章节中，我们实验了最小的 GPT-2 模型，该模型仅有 124M 参数。这样做的原因是尽量降低资源需求。然而，你可以通过最小的代码更改轻松尝试更大的模型。例如，在第 5 章中加载 1558M 而不是 124M 模型时，我们只需更改以下 2 行代码：

```py
hparams, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
model_name = "gpt2-small (124M)"
```

更新后的代码如下：

```py
hparams, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2")
model_name = "gpt2-xl (1558M)"
```
