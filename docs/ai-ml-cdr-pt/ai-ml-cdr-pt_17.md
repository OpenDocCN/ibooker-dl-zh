# 第十六章\. 使用自定义数据与 LLM 交互

在第十五章中，我们探讨了 Transformers 以及它们的编码器、解码器和编码器-解码器架构是如何工作的。它们在革命性 NLP 方面的成果不容置疑！然后，我们探讨了 Hugging Face 的 Python 库中的 transformers，该库旨在使使用 Transformers 更加容易。

基于大型 Transformer 的模型，在大量文本上训练，非常强大，但它们并不总是适合特定的任务或领域。在本章中，我们将探讨如何使用 transformers 和其他 API 来适应这些模型以满足您的特定需求。

微调允许您使用您特定的数据来定制预训练模型。您可以使用这种方法来创建聊天机器人，提高分类准确率，或者为更具体的领域开发文本生成。

有几种方法可以做到这一点，包括传统的微调以及使用 LoRA 和参数高效微调（PEFT）等方法的参数高效微调。您还可以通过检索增强生成（RAG）从您的 LLM 中获得更多收益，我们将在第十八章中探讨。

在本章中，我们将探索一些实际示例，从传统的微调开始。

# 微调 LLM

让我们一步一步地看看如何微调一个像 BERT 这样的 LLM。我们将使用 IMDb 数据库，并在其上微调模型以更好地检测电影评论中的情感。这个过程涉及多个步骤，所以我们将详细查看每一个。

## 设置和依赖项

我们将首先设置所有我们需要用 PyTorch 进行微调的东西。除了基础知识外，您还需要包括以下三个新内容：

数据集

我们在第四章中介绍了数据集。我们将使用这些数据集来加载 IMDb 数据集和内置的训练和测试分割。

评估

这个库提供了用于衡量加载性能的指标。

transformers

正如我们在第十四章和第十五章中所述，Hugging Face 的 transformers 库旨在使使用 LLM 变得更加容易。

我们将在这个章节的微调练习中使用 Hugging Face transformers 库的一些类。这些包括以下内容：

AutoModelForSequenceClassification

这个类加载用于分类任务的预训练模型，并在基础模型顶部添加一个分类头。这个分类头随后将针对您正在微调的特定分类场景进行优化，而不是成为一个通用模型。如果我们指定了检查点名称，它将自动为我们处理模型架构。因此，要使用 BERT 模型和线性分类器层，我们将使用`bert-base-uncased`。

AutoTokenizer

这个类会自动初始化适当的分词器。这会将文本转换为适当的标记，并添加适当的特殊标记、填充、截断等。

TrainingArguments

这个类让我们可以配置训练设置和所有超参数，以及设置诸如要使用的设备等事项。

Trainer

这个类代表你管理训练循环，处理批处理、优化、损失、反向传播以及你需要重新训练模型的所有内容。

DataCollatorWithPadding

数据集中的记录数量并不总是与批大小相匹配。因此，这个类有效地将示例分批到适当的批大小，同时处理诸如注意力掩码和其他模型特定输入等细节。

我们可以在代码中看到这一点：

```py
# 1\. Setup and Dependencies
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
import numpy as np
```

现在所有依赖项都已就绪，我们将加载数据。

## 加载数据并检查

接下来，让我们使用 datasets API 加载数据。我们还将探索测试和训练数据集的大小。你可以使用以下代码：

```py
# 2\. Load and Examine Data
dataset = load_dataset("imdb")  # Movie reviews for sentiment analysis
print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")
```

这将输出以下内容：

```py
Train size: 25000
Test size: 25000
```

下一步是初始化模型和分词器。

## 初始化模型和分词器

在这个例子中，我们将使用`bert-base-uncased`模型，因此我们需要使用`AutoModelForSequenceClassification`初始化它并获取其关联的分词器：

```py
# 3\. Initialize Model and Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

注意`AutoModelForSequenceClassification`需要用我们想要分类的标签数量进行初始化。这定义了一个具有两个标签的新分类头。我们将使用的 IMDb 数据库有两个标签，分别表示正面和负面情感，因此我们将针对这一点进行重新训练。

在这一点上，指定模型将运行的设备也是一个好主意。使用这个模型进行训练是计算密集型的，如果你使用 Colab，你可能需要一个高 RAM 的 GPU，比如 A100。使用它将需要几分钟，但在 CPU 上可能需要几个小时！

## 数据预处理

一旦我们有了数据，我们想要对其进行预处理，只是为了得到我们训练所需的内容。当然，这一步的第一步将是分词文本，这里的`preprocess`函数处理这一点，给出一个带有填充的 512 个字符的序列长度：

```py
# 4\. Preprocess Data
def preprocess_function(examples):
   result = tokenizer(
       examples["text"],
       truncation=True,
       max_length=512,
       padding=True
   )
   # Trainer expects a column called labels, so copy over from label
   result["labels"] = examples["label"]
   return result

tokenized_dataset = dataset.map(
   preprocess_function,
   batched=True,
   remove_columns=dataset["train"].column_names
)
```

这里的一个重要注意事项是，原始数据带有表示评论的`text`列和表示负面或正面情感的`label`列（0 或 1）。然而，我们**不需要**`text`列来训练数据，并且 Hugging Face Trainer（我们将在稍后看到）期望包含标签的列被命名为`labels`（复数）。因此，你会看到我们移除了原始数据集中的所有列，并且标记化数据集将包含标记化数据和名为`labels`的列，而不是`label`，原始值被复制过来。

## 数据收集

当我们批量处理序列化、分词数据输入模型时，可能会存在批或序列大小差异需要处理。在我们的案例中，我们不需要担心序列大小，因为我们使用了一个将长度强制设置为 512（在之前的设置中）的分词器。然而，作为 transformers 库的一部分，collator 类仍然能够处理它，我们将使用它们来确保批大小的一致性。

因此，`DataCollatorWithPadding`类的最终作用是接受不同长度的多个示例，在必要时提供填充，将输入转换为张量，并在必要时创建注意力掩码。

在我们的案例中，我们实际上只是在将数据转换为张量以供模型输入，但如果以后想要在分词过程中进行任何更改，使用`DataCollatorWithPadding`仍然是一个好的实践。

下面是代码：

```py
# 5\. Create Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

## 定义指标

现在，让我们定义一些我们希望在训练模型时捕获的指标。我们将只做准确度，即比较预测值和实际值。以下是一些简单的代码来实现这一点：

```py
# 6\. Define Metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
```

它使用了 Hugging Face 的 evaluate 库中的`evaluate.load`，该库提供了一个简单标准化的接口，专门为这类任务设计。它可以为我们处理繁重的工作，而不是要求我们自行创建指标，对于评估任务，我们只需传递预测集和标签集，然后由它进行计算。evaluate 库预先构建，可以处理包括 f1、BLEU 在内的多种指标。

## 配置训练

接下来，我们可以通过使用`TrainingArguments`对象来配置模型如何重新训练。这提供了一系列你可以设置的超参数，包括用于学习率、权重衰减等，这些参数由`optimizer`和`loss`函数使用。它旨在让你对学习过程有更细致的控制，同时简化复杂性。

下面是我用于与 IMDb 进行微调的设置：

```py
# 7\. Configure Training
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    report_to="none",
    fp16=True
)
```

注意并调整超参数以获得不同的结果非常重要。除了上述用于优化器的参数外，你还需要考虑批大小。你可以为训练或评估设置不同的参数。

一个非常有用的参数——特别是对于超过这里三个 epoch 的训练会话——是`load_best_model_at_end`。它不会总是使用最后的检查点，而是会根据指定的指标（在这种情况下是准确度）跟踪最佳的检查点，并在完成后加载它。由于我将`evaluation`和`save`策略设置为`epoch`，它只会在 epoch 结束时这样做。

注意 `report_to` 参数：默认情况下，训练使用 `weights and biases` 作为报告的后端。我将 `report_to` 设置为 `none` 以关闭此报告。如果你想保留它，你需要一个 Weights and Biases API 密钥。你可以很容易地从状态窗口或通过访问 [Weights and Biases 网站](https://oreil.ly/yMX1A) 获取这个密钥。在训练过程中，你会被要求粘贴这个 API 密钥。确保在离开之前完成此操作，尤其是如果你在 Colab 上支付计算单元费用的话。

有许多参数可以进行实验，并且能够这样轻松地进行参数化，也允许你使用像 [Ray Tune](https://oreil.ly/fDAhG) 这样的工具轻松地进行神经架构搜索。

## 初始化 Trainer

与训练参数一样，transformers 提供了一个 trainer 类，你可以与它们一起使用来封装一个完整的训练周期。

你可以通过模型、训练参数、数据、collator 以及你之前初始化的度量策略来初始化它。所有之前的步骤都是为了这一步做准备。以下是你需要用到的代码：

```py
# 8\. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```

## 训练和评估

现在一切都已经设置好了，调用 trainer 上的 `train()` 方法进行训练，以及 `evaluate()` 方法进行评估，变得非常简单。

这是代码：

```py
# 9\. Train and Evaluate
train_results = trainer.train()
print(f"\nTraining results: {train_results}")

eval_results = trainer.evaluate()
print(f"\nEvaluation results: {eval_results}")
```

例如，虽然你可以在 Google Colab 的免费层中训练这个模型，但你的计时体验可能会有所不同。仅使用 CPU，可能需要好几个小时。我用 T4 高内存 GPU 训练了这个模型，每小时需要 1.6 个计算单元。整个训练过程大约需要 50 分钟，但我将其四舍五入到一小时，包括所有下载和设置。在撰写本文时，专业 Colab 订阅每月 9.99 美元可以获得一百个计算单元。你也可以选择 A100 GPU，它要快得多（使用它训练我大约只需要 12 分钟），但价格也更贵，每小时大约 6.8 个计算单元。

训练后，结果看起来是这样的：

```py
Training results:
TrainOutput(global_step=585,
            training_loss=0.18643947177463108,
            metrics={'train_runtime': 597.9931,
            'train_samples_per_second': 125.42,
            'train_steps_per_second': 0.978,
            'total_flos': 1.968912649469952e+16,
            'train_loss': 0.18643947177463108,
            'epoch': 2.9923273657289})
Evaluation results:
            {'eval_loss': 0.18489666283130646,
            'eval_accuracy': 0.93596,
            'eval_runtime': 63.8406,
            'eval_samples_per_second': 391.601,
            'eval_steps_per_second': 48.95,
            'epoch': 2.9923273657289}
```

在评估数据集上，我们只经过三个周期就看到了相当高的准确率（大约 94%），这是一个好兆头——但当然，可能存在过拟合，这需要单独的评估。但在大约 12 分钟的 LLM 微调工作后，我们显然正在朝着正确的方向前进！

## 保存和测试模型

一旦我们训练了模型，保存它以供将来使用是个好主意，而 `trainer` 对象使得这一点变得简单：

```py
# 10\. Save Model
trainer.save_model("./final_model")

```

保存模型后，我们可以开始使用它。为此，让我们创建一个辅助函数，该函数接受输入文本，对其进行标记化，然后将这些标记转换为键值（k, v）的一组输入向量：

```py
# 11\. Example Usage
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

```

然后，我们可以使用 PyTorch 的推理模式从这些输入中获取输出，并将它们转换为一组预测：

```py
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

```

返回的预测将是一个二维张量。神经元 0 是预测为负的概率，神经元 1 是预测为正的概率。因此，我们可以查看正概率，并返回一个带有其值的情感和置信度。我们也可以对负概率做同样的处理；这完全是任意的：

```py
    positive_prob = predictions[0][1].item()
    return {
        'sentiment': 'positive' if positive_prob > 0.5 else 'negative',
        'confidence': positive_prob if positive_prob > 0.5 else 1 – positive_prob
    }
```

我们现在可以使用如下代码进行预测测试：

```py
# Test prediction
test_text = "This movie was absolutely fantastic! The acting was superb."
result = predict_sentiment(test_text)
print(f"\nTest prediction for '{test_text}':")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

输出将类似于以下内容：

```py
Test prediction for 'This movie was absolutely fantastic! 
                     The acting was superb.':
Sentiment: positive
Confidence: 99.16%
```

我们可以看到，这个声明是积极的，并且信心很高！

在这个过程中，您可以逐步看到如何对现有 LLM 进行微调，以在新数据上将其转变为分类引擎！在许多情况下，这可能有些过度（而且训练自己的模型而不是微调 LLM 可能更快、更便宜），但评估这个过程肯定是有价值的。有时，甚至未经调整的 LLM 在分类方面也能很好地工作！根据我的经验，利用 LLM 的一般人工理解性质将导致创建出更有效的分类器，并产生更强的结果。

# 提示调整 LLM

与微调相比，*提示调整*是一种轻量级的替代方案，您可以通过向每个输入前缀可训练的*软提示*来调整模型以适应特定任务。在提示调整中，您通过修改模型权重而不是修改模型权重来完成此操作。这些软提示将在训练过程中进行优化。

这些软提示就像学习到的指令，可以引导模型的行为。与离散文本提示（如`Classify the sentiment`）不同，软提示的想法是它们存在于模型的嵌入空间中作为连续向量。例如，当处理“这部电影太棒了”时，模型会看到“[V1][V2]…[V20]这部电影太棒了”。在这种情况下，[V1][V2]...[V20]是帮助将模型引导到所需分类的向量。

最终，这里的优势在于效率。所以，您不需要微调模型，修改每个任务的权重，并保存整个模型以供重用，您只需要保存软提示向量。这些向量要小得多，并且可以帮助您拥有一系列微调，您可以使用它们轻松地引导模型执行特定任务，而无需管理多个模型。

这样的提示调整实际上可以匹配或超过完整微调的性能，尤其是在较大的模型中，并且效率显著更高。

现在，让我们探讨如何使用 IMDb 数据集直接比较本章早些时候的微调来提示调整 BART LLM。

## 准备数据

让我们从准备我们的数据开始，从 IMDb 数据集中加载数据，并设置虚拟标记。以下是您需要的代码：

```py
# Data preparation
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_length = 512
num_virtual_tokens = 20

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length - num_virtual_tokens
    )
```

这也将对输入的示例进行分词，所以你应该注意，任何示例的最大长度现在将减少虚拟标记的数量。例如，对于 BERT，我们有一个最大长度为 512，但如果我们有 20 个虚拟标记，那么序列最大长度现在将是 492。

我们现在将加载数据的一个子集，并尝试使用 5,000 个示例，而不是 25,000 个。你可以对这个数字进行实验，以较小的样本量换取更快的训练速度，或者以较大的样本量换取更高的准确性。

首先，我们将创建从数据集中提取用于训练的索引，并对它们进行测试。将这些视为指向我们感兴趣记录的指针。在这里我们进行随机采样：

```py
# Use only 5000 examples for training
train_size = 5000
np.random.seed(42)
train_indices = np.random.choice(len(dataset["train"]), train_size, 
                                                        replace=False)
test_indices = np.random.choice(len(dataset["test"]), train_size, replace=False)

tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)
```

然后，最后两行定义了映射函数，它简单地从我们感兴趣的数据库中提取值并对它们进行分词。我们将在下一步中看到这一点。

## 创建数据加载器

现在我们有了分词的训练和测试数据集，我们希望将它们转换为数据加载器。我们将通过首先从底层数据中选择与我们的索引内容匹配的原始示例来完成此操作：

```py
# Create subset for training
tokenized_train = tokenized_train.select(train_indices)
tokenized_test = tokenized_test.select(test_indices)

```

然后，我们将设置我们感兴趣的数据格式。数据集中可能有多个列，但你在训练中不会使用它们全部。在这种情况下，我们想要`input_ids`，这是我们输入内容的分词版本；`attention_mask`，它是一组向量，告诉我们应该关注`input_ids`中的哪些标记（这具有过滤掉填充或其他非语义标记的效果）；以及标签：

```py
tokenized_train.set_format(type="torch", columns=["input_ids", 
                                                  "attention_mask", "label"])
tokenized_test.set_format(type="torch", columns=["input_ids", 
                                                 "attention_mask", "label"])

```

现在，我们可以指定用于这些训练和测试集的数据加载器。在这里我有一个较大的批量大小，因为我是在 Colab 上的 40Gb GRAM GPU 上进行测试。在你的环境中，你可能需要调整这些：

```py
train_dataloader = DataLoader(tokenized_train, batch_size=64, shuffle=True)
eval_dataloader = DataLoader(tokenized_test, batch_size=128)
```

现在数据已经处理并加载到数据加载器中，我们可以进行下一步：定义模型。

## 定义模型

首先，让我们看看如何实例化模型，然后我们可以回到原始定义。通常，在我们的代码中，一旦我们设置了数据加载器，我们就会想要创建一个模型的实例。我们将使用如下代码：

```py
# Define the model
model = PromptTuningBERT(num_virtual_tokens=num_virtual_tokens, 
                         max_length=max_length)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
```

这使得代码既简洁又清晰，我们将把底层的 BERT 封装在一个提示调整版本的覆盖中。现在，虽然对于转换器来说有一个是很好的，但它们并没有，所以我们需要为自己创建这个类。

就像我们会用任何定义模型的 PyTorch 类一样，我们将通过一个`__init__`方法来创建它以进行设置，以及一个 PyTorch 训练循环在正向传递期间会调用的正向方法。所以，让我们从`__init__`方法和类定义开始：

```py
class PromptTuningBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", 
                       num_virtual_tokens=50, 
                       max_length=512):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
                        model_name, 
                        num_labels=2)
        self.bert.requires_grad_(False)

        self.n_tokens = num_virtual_tokens
        self.max_length = max_length - num_virtual_tokens

        vocab_size = self.bert.config.vocab_size
        token_ids = torch.randint(0, vocab_size, (num_virtual_tokens,))
        word_embeddings = self.bert.bert.embeddings.word_embeddings
        prompt_embeddings = word_embeddings(token_ids).unsqueeze(0)
        self.prompt_embeddings = nn.Parameter(prompt_embeddings)
```

这里有很多事情在进行中，所以让我们一点一点地分解它。首先，我将`num_virtual_tokens`的默认值设置为 50，将`max_length`的默认值设置为 512。如果你在实例化类时没有指定自己的默认值，你会得到这些值。在这种情况下，调用代码将它们分别设置为 20 和 512，但你完全可以根据自己的需要进行实验。

接下来，代码设置了 transformers 的`AutoModelForSequenceClassification`类以获取 BERT：

```py
        self.bert = AutoModelForSequenceClassification.from_pretrained(
                        model_name, 
                        num_labels=2)
```

与为 IMDb 进行的微调一样，我们感兴趣的是训练模型以识别两个标签，所以它们在这里被设置。然而，与微调的一个不同之处在于，我们不会改变 BERT 模型内部的任何权重，所以我们设置我们不希望梯度并像这样冻结它：

```py
self.bert.requires_grad_(False)
```

我们将要使用的软提示的秘密配方在初始化的末尾。我们将创建一个向量来包含我们的虚拟标记数量，我刚刚用词汇表中的随机标记初始化了它。我们可能在这里做更聪明的事情来使训练在长时间内更有效率，但为了简单起见，让我们就这样做：

```py
token_ids = torch.randint(0, vocab_size, (num_virtual_tokens,))
```

在 transformers 库中的预训练 BERT 模型自带嵌入层，因此我们可以使用它们将我们的随机标记列表转换为嵌入层：

```py
word_embeddings = self.bert.bert.embeddings.word_embeddings
prompt_embeddings = word_embeddings(token_ids).unsqueeze(0)
```

重要的是，我们现在应该指定`prompt_embeddings`是神经网络的参数。这将在我们定义优化器时变得很重要。我们最近指定了所有的 BERT 参数都被冻结，但*这些*参数不包含在其中，因此它们不会被冻结，所以它们将在训练过程中被优化器调整：

```py
self.prompt_embeddings = nn.Parameter(prompt_embeddings)
```

我们现在初始化了一个可调整的 BERT 的子类版本，指定我们不希望修改其梯度，并创建了一组软提示，我们将在训练过程中将其附加到示例上——并且我们只调整这些软提示以软调整两个输出神经元。

现在，让我们看看在训练时前向传递过程中将被调用的`forward`函数。鉴于我们已经设置了一切，这个过程相当直接：

```py
def forward(self, input_ids, attention_mask, labels=None):
    batch_size = input_ids.shape[0]
    input_ids = input_ids[:, :self.max_length]
    attention_mask = attention_mask[:, :self.max_length]

    embeddings = self.bert.bert.embeddings.word_embeddings(input_ids)
    prompt_embeddings = self.prompt_embeddings.expand(batch_size, –1, –1)
    inputs_embeds = torch.cat([prompt_embeddings, embeddings], dim=1)

    prompt_attention_mask = torch.ones(batch_size, self.n_tokens, 
                                       device=attention_mask.device)
    attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

    return self.bert(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True
    )
```

让我们一步一步地看看。在训练过程中的前向传递中，这个函数将传递数据批次。因此，我们需要了解这个批次的大小，然后提取`input_ids`（从数据集中读取的值的标记）和特定 ID 的注意力掩码：

```py
    batch_size = input_ids.shape[0]
    input_ids = input_ids[:, :self.max_length]
    attention_mask = attention_mask[:, :self.max_length]

```

我们还需要将`input_ids`转换为嵌入层：

```py
embeddings = self.bert.bert.embeddings.word_embeddings(input_ids)
```

我们的软提示也是标记化的句子。最初，它们被初始化为随机单词，随着时间的推移，我们会看到它们会相应地调整。但在这个步骤中，这些标记需要被转换为嵌入层：

```py
prompt_embeddings = self.prompt_embeddings.expand(batch_size, –1, –1)

```

`expand` 方法只是将批大小添加到提示嵌入中。当我们定义类时，我们不知道每个进入的批大小会有多大（代码编写是为了让你根据可用内存的大小进行调整），所以使用 `expand(batch_size, –1, –1)` 将提示嵌入的向量，其形状为 `[1, num_prompt_tokens, embedding_dimensions]`，转换为 `[batch_size, num_prompt_tokens, embedding_dimensions]`。

我们的软提示调整涉及将软嵌入添加到实际输入数据的嵌入之前，所以我们这样做：

```py
inputs_embeds = torch.cat([prompt_embeddings, embeddings], dim=1)
```

BERT 使用 `attention_mask` 来过滤掉在训练或推理时间我们不关心的标记，通常是填充标记。但我们要让 BERT 关注所有的软提示标记，所以我们将它们的注意力掩码设置为全 1，然后将它们附加到训练数据的传入注意力掩码（s）中。以下是代码：

```py
prompt_attention_mask = torch.ones(batch_size, self.n_tokens, 
                                   device=attention_mask.device)

attention_mask = torch.cat([prompt_attention_mask, attention_mask], 
                            dim=1)

```

现在我们已经完成了所有的调整，我们需要将数据传递给模型以进行优化和计算损失：

```py
    return self.bert(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True
    )

```

接下来，我们将看到这些数据如何在训练循环中使用。

## 训练模型

这次训练的关键是我们将进行完整的、正常的训练循环，但在特殊情况下。在这种情况下，我们之前冻结了 BERT 模型中的所有内容，*除了*我们定义的作为模型参数的软提示。因此，例如，我们定义优化器如下：

```py
optimizer = AdamW(model.parameters(), lr=1e-2)
```

在这种情况下，我们使用标准代码，告诉它调整模型的参数。但唯一可调整的是软提示，所以这应该很快！

注意，学习率的值相当大。这有助于系统快速学习，但在实际系统中，你可能希望该值更小——或者至少是可调整的，从大的值开始，然后在后续的周期中减小。

现在，让我们进入训练阶段。首先，我们将设置训练循环：

```py
num_epochs = 3

# Perform the training
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
```

### 管理数据批次

对于每个批次，我们将获取列（`input_ids` 和 `attention_masks`）以及标签，并将它们传递给模型：

```py
for batch in tqdm(train_dataloader, 
                  desc=f'Training Epoch {epoch + 1}'):
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch.pop('label')
    outputs = model(**batch, labels=labels)
```

这看起来与本书中较早的部分略有不同，但它基本上在做同样的事情。`tqdm` 代码只是提供了一个状态栏，因为我们正在训练。我们按批次读取数据，但我们希望数据与模型位于同一设备上。例如，如果模型在 GPU 上运行，我们希望它能够访问 GPU 的内存。因此，此行将遍历每一列，读取键并将值传递到设备：

```py
batch = {k: v.to(device) for k, v in batch.items()}
```

它重新定义了读入的批次，以确保数据与模型位于同一设备上。但我们不希望标签在批次中，因为模型期望它们单独输入，所以我们使用 `pop()` 方法将它们从批次中移除：

```py
labels = batch.pop('label')
```

现在，我们可以使用`**batch`的缩写来将输入值的集合（在这种情况下，是`input_ids`和`attention_mask`）传递给模型的 forward 方法，并像这样与标签一起解包字典：

```py
outputs = model(**batch, labels=labels)
```

### 处理损失

前向传递将数据发送到模型并返回损失。我们使用这个来更新我们的总损失，然后我们可以进行反向传递：

```py
loss = outputs.loss
total_train_loss += loss.item()

loss.backward()
```

随着梯度的反向流动，优化器现在可以开始工作。

### 优化损失

记住`model.parameters()`只会管理*可训练未冻结*的参数，我们现在可以调用优化器。我在这里添加了*梯度裁剪*来使训练更有效率，但其余的只是调用优化器的下一步，然后清零梯度以便我们下次可以使用：

```py
clip_grad_norm_(model.parameters(), max_grad_norm)  # Add here
optimizer.step()
optimizer.zero_grad()
```

###### 注意

*梯度裁剪*背后的想法是，在反向传播过程中，梯度有时可能太大，优化器可能会采取非常大的步骤。这可能导致一个称为*梯度爆炸*的问题，其中值的改变隐藏了可能学习到的细微差别。但是，如果梯度值变得太大，裁剪会将其缩小，在这种情况下，它们甚至可能不是必需的。

## 训练期间的评估

我们还有一组测试数据，因此我们可以评估模型在整个训练周期中的表现。在每个 epoch 中，一旦完成前向和反向传递并且模型参数被重置，我们可以将模型切换到评估模式，然后开始将所有测试数据通过它以获取推理。我们还将比较推理结果与实际标签以计算准确率：

```py
model.eval()
val_accuracy = []
total_val_loss = 0
```

然后，我们将有类似的代码——但这次，它将是读取评估批次，将它们转换为带有标签的输出，并从模型中获取预测和损失值：

```py
with torch.no_grad():
    for batch in tqdm(eval_dataloader, desc='Validating'):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('label')

        outputs = model(**batch, labels=labels)
        total_val_loss += outputs.loss.item()

        predictions = torch.argmax(outputs.logits, dim=-1)
        val_accuracy.extend((predictions == labels).cpu().numpy())
```

一旦我们计算了这些值，那么在每个 epoch 结束时，我们可以报告它们和训练损失。

## 报告训练指标

在每个 epoch 的训练中，我们计算了训练损失，因此我们现在可以获取所有记录的平均值。我们可以用验证损失做同样的事情，当然还有验证准确率，然后报告它们：

```py
avg_train_loss = total_train_loss / len(train_dataloader)
avg_val_loss = total_val_loss / len(eval_dataloader)
val_accuracy = np.mean(val_accuracy)

print(f"\nEpoch {epoch + 1}:")
print(f"Average training loss: {avg_train_loss:.4f}")
print(f"Average validation loss: {avg_val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")
```

进行三个 epoch 的训练给出以下结果：

```py
Training Epoch 1: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s]
Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]

Epoch 1:
Average training loss: 0.6559
Average validation loss: 0.6037
Validation accuracy: 0.8036
Training Epoch 2: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s]
Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]

Epoch 2:
Average training loss: 0.6112
Average validation loss: 0.5854
Validation accuracy: 0.8386
Training Epoch 3: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s]
Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]

Epoch 3:
Average training loss: 0.5799
Average validation loss: 0.5270
Validation accuracy: 0.8736

```

这是在 Colab 上的 A100 上完成的，有 40 Gb 的 GRAM，正如你所看到的，每个 epoch 的训练只需大约 1 分钟，评估只需 30 秒。

到最后，平均训练损失从大约 0.65 下降到 0.58。准确率为 0.8736。因此，很可能是过拟合，因为我们只训练了三个 epoch。

## 保存提示嵌入

这种方法的真正好处是，当你完成时，你可以简单地保存提示嵌入。你还可以在稍后加载它们以进行推理，正如你将在下一节中看到的：

```py
torch.save(model.prompt_embeddings, "imdb_prompt_embeddings.pt")
```

我觉得这真的很酷的是，这个文件相对较小（61 K），并且它不需要你以任何方式修改底层模型。因此，在应用中，你可能会有一系列这样的提示调整文件，并且可以根据需要热插拔和替换它们，这样你就可以拥有多个可以协调的模型，这是代理解决方案的基础。

## 使用模型进行推理

要使用提示调整模型进行推理，你只需用软提示定义模型，然后，而不是训练它们，从磁盘加载预训练的软提示。我们将在本节中探讨这一点。如果你不想训练自己的模型，那么[在下载](https://github.com/lmoroney/PyTorch-Book-FIles)中，我提供了一种经过 30 个 epoch 训练的模型版本的软提示，而不是 3 个 epoch。

为了更整洁的封装，我创建了一个类，它类似于我们用于训练的类，但仅用于推理。我称它为`PromptTunedBERTInference`，以下是它的初始化器：

```py
class PromptTunedBERTInference:
    def __init__(self, model_name="bert-base-uncased", 
                       prompt_path="imdb_prompt_embeddings.pt"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model =  
                  AutoModelForSequenceClassification.from_pretrained(
                                            model_name, num_labels=2)
        self.model.eval()
        self.prompt_embeddings = torch.load(prompt_path)
        self.device = 
           torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
```

它与可训练的初始化器非常相似，除了几个重要的点。首先是因为我们*仅*使用它进行推理，我已经将其设置为评估模式：

```py
self.model.eval()
```

第二点是我们不需要训练嵌入并执行所有相关操作——相反，我们只需从指定的路径加载它们：

```py
self.prompt_embeddings = torch.load(prompt_path)
```

就这样！如你所见，它相当轻量级且非常直接。它不会有`forward`函数，因为我们没有在训练它，但让我们添加一个`predict`函数来封装使用它的推理。

### `predict`函数

`predict`函数的职责是接收我们想要进行推理的字符串（字符串），对其进行标记化（它们），然后将它们与前置的软标记一起传递给模型。让我们逐块查看代码。

首先，让我们定义它，并让它接受它将随后进行标记化的文本：

```py
def predict(self, text):
    inputs = self.tokenizer(text, padding=True, 
                        truncation=True,
                        max_length=512-self.prompt_embeddings.shape[1],
                        return_tensors="pt")
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
```

文本将被标记化到最大长度，减去软提示的大小，然后输入中的每个项目将被加载到一个字典中。请注意，标记器将为文本返回多个列——通常是标记和注意力掩码——因此我们将遵循这种方法将它们转换成一组易于我们以后处理的键值对。

现在我们有了输入，是时候将它们传递给模型以获取我们的推理结果了。我们首先将`torch`置于`no_grad()`模式，因为我们不感兴趣于训练梯度。然后我们将为我们的每个标记获取嵌入：

```py
with torch.no_grad():
    embeddings = self.model.bert.embeddings.word_embeddings(
                                        inputs['input_ids'])

    batch_size = embeddings.shape[0]

    prompt_embeds = self.prompt_embeddings.expand(
                         batch_size, –1, –1).to(self.device)

    inputs_embeds = torch.cat([prompt_embeds, embeddings], dim=1)
```

我们有一个由标记器生成的输入注意力掩码，但没有软提示的注意力掩码。所以，让我们创建一个，并将其附加到输入注意力掩码上：

```py
attention_mask = inputs['attention_mask']
prompt_attention = torch.ones(batch_size, self.prompt_embeddings.shape[1],
                            device=self.device)
attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)
```

现在我们已经准备好了所有东西，我们可以将我们的数据传递给模型以获取我们的推理结果：

```py
outputs = self.model(inputs_embeds=inputs_embeds,
                   attention_mask=attention_mask)
```

输出将是两个神经元的 logits，一个表示积极情绪，另一个表示消极。然后我们可以对它们进行 Softmax 以获得预测：

```py
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
return {"prediction": outputs.logits.argmax(-1).item(),
       "confidence": probs.max(-1).values.item()}
```

### 使用示例

使用这个类进行预测是非常直接的。我们创建类的实例，并传递一个字符串给它以获取结果。结果将包含一个预测和一个置信度值，然后我们可以将其输出：

```py
# Usage example
if __name__ == "__main__":
    model = PromptTunedBERTInference()
    result = model.predict("This movie was great!")
    print(f"Prediction: {'Positive' 
                          if result['prediction'] == 1 else 'Negative'}")
    print(f"Confidence: {result['confidence']:.2f}")

```

你可能会在提示微调中看到一些低置信度值，这可能导致错误预测，尤其是在像这样的二元分类器中。探索你的推理以确保其正常工作是个好主意，同时也有一些技术你可以探索来确保 logits 给出你想要的价值。这包括设置 Softmax 的温度、使用更多的提示标记来给模型更多的容量，以及用与情感相关的词初始化提示标记（而不是像我们这里这样使用随机标记）。

# 摘要

在本章中，我们探讨了使用我们自己的数据定制 LLMs 的不同方法。我们研究了两种主要方法：传统的微调和提示微调。

使用 IMDb 数据集，你看到了如何微调 BERT 进行情感分析，并走过了所有步骤——从数据准备到模型配置，再到训练和评估。该模型在仅几个 epoch 内就实现了令人印象深刻的 95%的情感分类准确率。

然而，微调可能并不适用于所有情况，因此你探索了一个轻量级的替代方案，称为提示微调。在这里，不是修改模型权重，而是将可训练的软提示预置于输入之前，这些提示在训练过程中被优化。这种方法提供了显著的优势，它可以更快，并且不会改变底层模型。在这种情况下，调整后的提示可以被保存（并且它们只有几 Kb），然后重新加载以编程模型执行所需的任务。然后你进行了一次完整的实现，展示了如何创建、训练和保存这些软提示，以及如何将它们加载回来以执行推理。

在下一章中，我们将探讨如何提供 LLMs，包括定制的 LLMs。我会解释如何通过使用 Ollama 来实现这一点，Ollama 是一个用于处理 LLMs 的托管和管理功能强大的工具。你将学习如何将模型转换为服务，我们还将探讨如何设置 Ollama 并使用它通过 HTTP 与数据中心中的模型进行通信。
