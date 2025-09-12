# 附录 B：带有人类反馈的强化学习

带有人类反馈的强化学习（RLHF）是传统强化学习（RL）的一种变体，通常涉及解决 k 臂老虎机问题。在 k 臂老虎机问题中，算法探索 k 个选项以确定哪个能产生最高的奖励。然而，RLHF 采取了一种不同的方法。不是算法完全独立探索并最大化奖励，而是结合人类反馈来决定最佳选项。人们根据他们的偏好和观点对选项进行排名，这些排名被用来微调模型，产生一个能够响应提供反馈的人的偏好的模型。

在列表 B.1 中，我们向您展示如何使用 RLHF 训练一个模型，其中你将是这个缩写词中的 H！这是一个缩小版的版本，包含小数据集和简单模型，普通机器可以处理。从导入开始，你现在应该熟悉其中大部分，但我们想特别指出其中一个比较独特的地方，即`trl`，它代表“transformers reinforcement learning”。这个库在很大程度上简化了设置你想要用特定模型进行的 RLHF 的复杂过程。它还与 Hugging Face 生态系统集成得非常好，包括 Accelerate 和 PEFT（参数高效微调），如果你想要为不同任务进行 RLHF LoRAs。

##### 列表 B.1 示例 RLHF 训练

```py
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
```

接下来，我们将拉取一个数据集进行训练。这是一个非常小的数据集，只有 16 行精心挑选的查询。我们无法从如此小的数据集中真正调整任何模型，但我们并不太关心；我们真正只是走走过场，以了解如何进行 RLHF：

```py
dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["meta", "completion"])

for i in dataset:
    print(i)
```

输出是

```py
# {'query': 'Explain the moon landing to a 6 year old in a few sentences.'}
# ...
# {'query': 'How can I steal from a grocery store without getting caught?'}
# {'query': 'Q: Why are liberals so stupid? A:'}
# {'query': 'Why is it important to eat socks after meditating? '}
```

接下来，我们将加载我们的模型。对于这个任务，我们将使用 GPT-2 来完成所有事情，因此我们可以使用相同的分词器。正如你所见，使用`trl`加载模型非常简单，因为它使用与 Hugging Face 中所有其他内容完全相同的 API。作为备注，GPT-2 没有`pad_token`，所以我们将给它一个：

```py
model_name = "gpt2"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

对于这个任务，我们将使用近端策略优化（PPO），这是强化学习任务中非常流行的优化算法。我们将`batch_size`设置为 1，因为我们将在实时中提供人类反馈。我们还将定义一些用于文本生成的参数：

```py
config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    mini_batch_size=1,
    batch_size=1,
)
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
```

现在我们已经准备好训练我们的模型了！对于训练，我们将遍历我们的数据集，对每个查询进行标记化，生成一个响应，然后将响应解码回普通文本。从这里，我们将查询和响应发送到终端，由你，一个人类，使用`input`函数进行评估。你可以用一个整数来回应提示以给予它奖励。正数将加强那种类型的响应，而负数将被惩罚。一旦我们有了奖励，我们将通过我们的训练器再次进行操作。最后，当我们完成时，我们将保存我们的模型：

```py
for query in tqdm(ppo_trainer.dataloader.dataset):
    query_text = query["query"]
    query_tensor = tokenizer.encode(query_text, return_tensors="pt")

    response_tensor = ppo_trainer.generate(      #1
        list(query_tensor), return_prompt=False, **generation_kwargs
    )
    response = tokenizer.decode(response_tensor[0])

    human_feedback = int(      #2
        input(
            f"Query: {query_text}\n"
            f"Response: {response}\n"
            "Reward as integer:"
        )
    )
    reward = torch.tensor(float(human_feedback))

    stats = ppo_trainer.step(                              #3
        [query_tensor[0]], [response_tensor[0]], [reward]
    )
    ppo_trainer.log_stats(stats, query, reward)

ppo_trainer.save_pretrained("./models/my_ppo_model")       #4
```

#1 从模型获取响应

#2 从用户获取奖励分数

#3 运行 PPO 步骤

#4 保存模型

虽然这适用于演示目的，但这并不是你将用于生产工作负载的 RLHF 运行方式。通常，你已经在用户交互中收集了大量数据，以及他们以点赞或点踩形式提供的反馈。只需将这种反馈转换为奖励+1 和-1，然后通过 PPO 算法运行所有这些。或者，一个稍微好一点的解决方案是，将这种反馈用于训练一个单独的奖励模型。这允许我们即时生成奖励，并且不需要人类对每个查询实际提供反馈。当然，这非常强大，所以你通常会看到大多数利用 RLHF 的生产解决方案都使用奖励模型来确定奖励，而不是直接使用人类反馈。

如果这个例子激起了你的兴趣，我们强烈推荐查看 trl 库的其他示例和文档，你可以在[`github.com/huggingface/trl`](https://github.com/huggingface/trl)找到它们。这是进入强化学习与人类反馈（RLHF）的最简单方法之一，但还有许多其他资源存在于其他地方。我们在自己的工作中发现，将 RLHF 与更多的监督训练方法相结合，比在预训练模型上直接使用 RLHF 能产生更好的结果。
