# 第十六章\. 使用自定义数据与 LLM 结合

在第十五章（ch15.html#ch15_transformers_and_transformers_1748549808974580）中，我们探讨了 Transformer 以及它们的编码器、解码器和编码器-解码器架构的工作原理。它们在革命性 NLP 方面的成果无可争议！然后，我们探讨了 Hugging Face 的 transformers 库，这是一个设计用来简化 Transformer 使用的 Python 库。

基于大量文本训练的大型 Transformer 模型非常强大，但它们并不总是适用于特定任务或领域。在本章中，我们将探讨如何使用 Transformer 和其他 API 来适应这些模型以满足您的特定需求。

微调允许您使用特定数据自定义预训练模型。您可以使用这种方法创建聊天机器人、提高分类准确度或为特定领域开发文本生成。

有几种方法可以实现这一点，包括传统的微调以及使用 LoRA 和参数高效微调（PEFT）等方法的参数高效微调。您还可以通过检索增强生成（RAG）从您的 LLM 中获得更多收益，我们将在第十八章（ch18.html#ch18_introduction_to_rag_1748550073472936）中探讨这一方法。

在本章中，我们将探索一些实际示例，从传统的微调开始。

# 微调 LLM

让我们一步一步地看看如何微调一个像 BERT 这样的 LLM。我们将使用 IMDb 数据库，并在其上微调模型以更好地检测电影评论中的情感。这个过程涉及多个步骤，因此我们将逐一详细探讨。

## 设置和依赖项

我们将首先设置所有使用 PyTorch 进行微调所需的内容。除了基础知识外，您还需要包括以下三个新内容：

数据集

我们在第四章（ch04.html#ch04_using_data_with_pytorch_1748548966496246）中介绍了数据集。我们将使用这些数据集来加载 IMDb 数据集和内置的训练和测试分割。

评估

这个库提供了用于衡量加载性能的指标。

transformers

正如我们在第十四章（ch14.html#ch14_using_third_party_models_and_hubs_1748549787242797）和第十五章（ch15.html#ch15_transformers_and_transformers_1748549808974580）中所述，Hugging Face 的 transformers 库旨在使使用 LLM 变得更加容易。

我们将使用 Hugging Face transformers 库的一些类来完成本章的微调练习。以下是一些类：

AutoModelForSequenceClassification

这个类加载用于分类任务的预训练模型，并在基础模型顶部添加一个分类头。然后，这个分类头将针对您正在微调的特定分类场景进行优化，而不是一个通用模型。如果我们指定了检查点名称，它将自动为我们处理模型架构。因此，要使用带有线性分类器层的 BERT 模型，我们将使用 `bert-base-uncased`。

AutoTokenizer

这个类会自动初始化适当的分词器。它将文本转换为适当的标记，并添加适当的特殊标记、填充、截断等。

TrainingArguments

这个类让我们配置训练设置和所有超参数，以及设置使用设备等。

Trainer

这个类代表你管理训练循环，处理批处理、优化、损失、反向传播以及你需要重新训练模型的所有事情。

DataCollatorWithPadding

数据集中的记录数并不总是与批处理大小相匹配。因此，这个类有效地将示例批处理到适当的批处理大小，同时处理诸如注意力掩码和其他模型特定输入等细节。

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

现在依赖项已经就绪，我们将加载数据。

## 加载和检查数据

接下来，让我们使用数据集 API 加载数据。我们还将探索测试和训练数据集的大小。你可以使用以下代码：

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

在这个例子中，我们将使用`bert-base-uncased`模型，因此我们需要通过使用`AutoModelForSequenceClassification`并获取其相关的分词器来初始化它：

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

注意`AutoModelForSequenceClassification`需要用我们想要分类的标签数量进行初始化。这定义了具有两个标签的新分类头。我们将使用的 IMDb 数据库有两个标签，分别表示正面和负面情感，因此我们将为此进行重新训练。

在这一点上，指定模型将运行的设备也是一个好主意。使用此模型进行训练计算量很大，如果你使用 Colab，你可能会需要一个高 RAM 的 GPU，比如 A100。使用它将需要几分钟，但在 CPU 上可能需要几个小时！

## 数据预处理

一旦我们有了数据，我们想要对其进行预处理，以便仅获取我们需要的用于训练的数据。这一步的第一步当然是分词文本，这里的`preprocess`函数处理这一点，给出 512 个字符的序列长度，并进行填充：

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

这里有一个重要的注意事项，即原始数据包含表示评论的`text`列和表示情感为 0 或 1 的`label`列。然而，我们不需要`text`列来训练数据，并且 Hugging Face Trainer（我们稍后将看到）期望包含标签的列被命名为`labels`（复数）。因此，你会看到我们移除了原始数据集中的所有列，并且分词后的数据集将包含分词后的数据和名为`labels`的列，而不是`label`列，原始值将被复制过来。

## 数据收集

当我们将经过序列化和分词的数据批量输入模型时，可能会存在批量和序列大小差异，这些差异需要处理。在我们的案例中，我们不需要担心序列大小，因为我们使用了一个将长度强制设置为 512（在之前的集合中）的分词器。然而，作为 transformers 库的一部分，collator 类仍然能够处理这种情况，我们将使用它们来确保批量大小的统一。

因此，`DataCollatorWithPadding`类的最终作用是接受不同长度的多个示例，在必要时提供填充，将输入转换为张量，并在必要时创建注意力掩码。

在我们的案例中，我们实际上只得到了模型输入的转换为张量的过程，但如果我们希望在以后更改分词过程，使用`DataCollatorWithPadding`仍然是一个好的实践。

下面是代码示例：

```py
# 5\. Create Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

## 定义指标

现在，让我们定义一些我们希望在训练模型时捕捉的指标。我们将只做准确性，即比较预测值和实际值。以下是一些简单的代码来实现这一点：

```py
# 6\. Define Metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
```

它使用的是 Hugging Face 的 evaluate 库中的`evaluate.load`，该库提供了一个简单标准化的接口，专门为这类任务设计。它可以为我们处理繁重的工作，而不是要求我们自行创建指标，对于评估任务，我们只需传递一组预测和一组标签，然后由它进行计算。evaluate 库预先构建，可以处理包括 f1、BLEU 在内的多种指标。

## 配置训练

接下来，我们可以通过使用`TrainingArguments`对象来配置模型如何重新训练。它提供了大量可以设置的超参数——包括那些用于`optimizer`和`loss`函数的学习率、权重衰减等——旨在在抽象复杂性的同时，给您提供对学习过程的精细控制。

下面是我用于在 IMDb 上微调时使用的集合：

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

注意并调整超参数以获得不同的结果非常重要。除了上述用于优化器的参数外，您还希望考虑批大小。您可以针对训练或评估设置不同的参数。

一个非常有用的参数——特别是对于超过这里三个 epoch 的训练会话——是`load_best_model_at_end`。它不会总是使用最后的检查点，而是会根据指定的指标（在这种情况下，准确性）跟踪最佳的检查点，并在完成后加载它。由于我将`evaluation`和`save`策略设置为`epoch`，它只会在 epoch 结束时这样做。

注意还有`report_to`参数：默认情况下，训练使用`weights and biases`作为报告的后端。我将`report_to`设置为`none`以关闭此报告。如果您想保留它，您将需要一个 Weights and Biases API 密钥。您可以从状态窗口或通过访问[Weights and Biases 网站](https://oreil.ly/yMX1A)非常容易地获得这个密钥。在训练过程中，您将被要求粘贴此 API 密钥。确保在离开之前这样做，尤其是如果您在 Colab 上支付计算单元费用。

有许多参数可以实验，并且能够像这样轻松参数化也允许您使用像[Ray Tune](https://oreil.ly/fDAhG)这样的工具轻松地进行神经架构搜索。

## 初始化训练器

与训练参数一样，transformers 提供了一个 trainer 类，您可以使用它来封装完整的训练周期。

您需要用模型、训练参数、数据、collator 以及您之前初始化的度量策略来初始化它。所有之前的步骤都是为了达到这一步。以下是您需要的代码：

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

现在一切准备就绪，调用 trainer 上的`train()`方法进行训练，以及`evaluate()`方法进行评估就变得非常简单了。

下面是代码：

```py
# 9\. Train and Evaluate
train_results = trainer.train()
print(f"\nTraining results: {train_results}")

eval_results = trainer.evaluate()
print(f"\nEvaluation results: {eval_results}")
```

例如，虽然您可以在 Google Colab 的免费层上训练此模型，但您的计时体验可能会有所不同。仅使用 CPU，可能需要好几个小时。我用 T4 高 RAM GPU 训练了这个模型，每小时需要 1.6 个计算单元。整个训练过程大约需要 50 分钟，但我将其四舍五入到一小时，包括所有下载和设置。在撰写本文时，专业 Colab 订阅每月 9.99 美元可以获得 100 个计算单元。您还可以选择 A100 GPU，它要快得多（使用它训练我大约需要 12 分钟），但价格也更贵，每小时大约 6.8 个计算单元。

训练完成后，结果看起来是这样的：

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

我们可以看到，在仅经过三个 epoch 后，评估数据集上的准确率相当高（大约 94%），这是一个好兆头——但当然，可能存在过拟合，这需要单独的评估。但在大约 12 分钟的 LLM 微调工作后，我们显然正在朝着正确的方向前进！

## 保存和测试模型

一旦我们训练了模型，保存它以供将来使用是个好主意，`trainer`对象使这变得很容易：

```py
# 10\. Save Model
trainer.save_model("./final_model")

```

保存模型后，我们就可以开始使用它了。为此，让我们创建一个辅助函数，该函数接受输入文本，对其进行标记化，然后将这些标记转换为键值（k, v）的一组输入向量：

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

然后，我们可以使用 PyTorch 的推理模式从这些输入中获取输出，并将它们转换成一系列预测：

```py
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

```

返回的预测结果将是一个具有两个维度的张量。神经元 0 是预测为负面的概率，神经元 1 是预测为正面的概率。因此，我们可以查看正面的概率，并返回一个带有其值的情感和置信度。我们也可以用同样的方式处理负面的预测；这完全是任意的：

```py
    positive_prob = predictions[0][1].item()
    return {
        'sentiment': 'positive' if positive_prob > 0.5 else 'negative',
        'confidence': positive_prob if positive_prob > 0.5 else 1 – positive_prob
    }
```

我们现在可以用如下代码测试预测：

```py
# Test prediction
test_text = "This movie was absolutely fantastic! The acting was superb."
result = predict_sentiment(test_text)
print(f"\nTest prediction for '{test_text}':")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

输出结果可能看起来像这样：

```py
Test prediction for 'This movie was absolutely fantastic! 
                     `The` `acting` `was` `superb``.``':` ``` `情感`：`积极` `置信度`：`99.16%` ```py
```

`` `我们可以看到这个陈述是积极的，非常有信心！在这个过程中，你可以看到如何一步一步地，你可以对新数据进行微调，将其变成一个分类引擎！在许多情况下，这可能有些过度（而且训练自己的模型而不是微调 LLM 可能更快更便宜），但评估这个过程肯定值得。有时，即使是未经调整的 LLM 在分类方面也可能表现良好！根据我的经验，利用 LLM 的一般人工智能理解能力将导致创建出更多有效且结果更强的分类器。` ``  ```py`` ````# Prompt-Tuning an LLM    一种轻量级的微调替代方案是 *prompt tuning*，其中你可以将模型适应特定任务。使用 prompt tuning，你通过在每个输入前添加可训练的 *软提示* 来这样做，而不是修改模型权重。这些软提示将在训练过程中进行优化。    这些软提示就像学习到的指令，可以引导模型的行为。与离散文本提示（如 `Classify the sentiment`）不同，软提示的想法是它们存在于模型的嵌入空间中作为连续向量。所以，例如，当处理“这部电影太棒了”时，模型会看到“[V1][V2]…[V20]这部电影太棒了。”在这种情况下，[V1][V2]...[V20]是帮助模型指向所需分类的向量。    最终，这里的优势是效率。所以，你不需要微调模型，为每个任务修改其权重，并保存整个模型以供重用，你只需要保存软提示向量。这些要小得多，并且可以帮助你拥有一系列微调，你可以轻松地使用它们来引导模型执行特定任务，而无需管理多个模型。    这样的 prompt tuning 实际上可以匹配或超过完整微调的性能，尤其是在大型模型中，并且效率显著更高。    现在，让我们探索如何使用 IMDb 数据集直接与本章前面提到的微调进行比较来 prompt-tune BART LLM。    ## 准备数据    让我们首先准备我们的数据，从 IMDb 数据集中加载数据，并设置虚拟标记。以下是你需要的代码：    ```py # Data preparation dataset = load_dataset("imdb") tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") max_length = 512 num_virtual_tokens = 20   def tokenize_function(examples):     return tokenizer(         examples["text"],         padding="max_length",         truncation=True,         max_length=max_length - num_virtual_tokens     ) ```    这也将对我们的输入示例进行标记化，所以你应该注意，任何示例的最大长度现在将减少虚拟标记的数量。例如，对于 BERT，我们有一个最大长度为 512，但如果我们有 20 个虚拟标记，那么序列最大长度现在将是 492。    现在，我们将加载数据的一个子集，并尝试使用 5,000 个示例，而不是 25,000 个。你可以尝试这个数字，以较小的数量换取更快的训练，或者以较大的数量换取更好的准确性。    首先，我们将创建从数据集中提取的索引，并进行测试。将这些视为指向我们感兴趣的记录的指针。我们在这里进行随机采样：    ```py # Use only 5000 examples for training train_size = 5000 np.random.seed(42) train_indices = np.random.choice(len(dataset["train"]), train_size,                                                          replace=False) test_indices = np.random.choice(len(dataset["test"]), train_size, replace=False)   tokenized_train = dataset["train"].map(tokenize_function, batched=True) tokenized_test = dataset["test"].map(tokenize_function, batched=True) ```    然后，最后两行定义了映射函数，它只是从我们感兴趣的数据集中提取值并将它们标记化。我们将在下一步中看到。    ## 创建数据加载器    现在我们有了标记化的训练和测试数据集，我们希望将它们转换为数据加载器。我们将通过首先选择与我们的索引内容匹配的底层数据中的原始示例来完成此操作：    ```py # Create subset for training tokenized_train = tokenized_train.select(train_indices) tokenized_test = tokenized_test.select(test_indices)   ```    然后，我们将设置我们感兴趣的数据格式。数据集中可能有许多列，但你不会使用它们的所有列进行训练。在这种情况下，我们想要`input_ids`，这是我们输入内容的标记化版本；`attention_mask`，它是一组向量，告诉我们应该关注`input_ids`中的哪些标记（这具有过滤掉填充或其他非语义标记的效果）；以及标签：    ```py tokenized_train.set_format(type="torch", columns=["input_ids",                                                    "attention_mask", "label"]) tokenized_test.set_format(type="torch", columns=["input_ids",                                                   "attention_mask", "label"])   ```    现在，我们可以指定 DataLoader，它接受这些训练和测试集。我在这里有一个很大的批量大小，因为我正在 Colab 上测试一个 40Gb 的 GRAM GPU。在你的环境中，你可能需要调整这些：    ```py train_dataloader = DataLoader(tokenized_train, batch_size=64, shuffle=True) eval_dataloader = DataLoader(tokenized_test, batch_size=128) ```    现在数据已处理并加载到 DataLoaders 中，我们可以进行下一步：定义模型。    ## 定义模型    首先，让我们看看如何实例化模型，然后我们可以回到原始定义。通常，在我们的代码中，一旦我们设置了 DataLoaders，我们就会想要创建一个模型的实例。我们将使用如下代码：    ```py # Define the model model = PromptTuningBERT(num_virtual_tokens=num_virtual_tokens,                           max_length=max_length)   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   model.to(device) ```    这使它既简单又清晰，我们将把底层的 BERT 封装在一个 prompt-tuning 版本的重写中。现在，虽然很希望 transformers 有一个这样的，但它们没有，所以我们需要为自己创建这个类。    就像我们会对任何定义模型的 PyTorch 类一样，我们将使用`__init__`方法来设置它，并在 PyTorch 的训练循环中调用`forward`方法。所以，让我们从`__init__`方法和类定义开始：    ```py class PromptTuningBERT(nn.Module):     def __init__(self, model_name="bert-base-uncased",                         num_virtual_tokens=50,                         max_length=512):         super().__init__()         self.bert = AutoModelForSequenceClassification.from_pretrained(                         model_name,                          num_labels=2)         self.bert.requires_grad_(False)           self.n_tokens = num_virtual_tokens         self.max_length = max_length - num_virtual_tokens           vocab_size = self.bert.config.vocab_size         token_ids = torch.randint(0, vocab_size, (num_virtual_tokens,))         word_embeddings = self.bert.bert.embeddings.word_embeddings         prompt_embeddings = word_embeddings(token_ids).unsqueeze(0)         self.prompt_embeddings = nn.Parameter(prompt_embeddings) ```    这里有很多事情要做，所以让我们一点一点地分解。首先，我将`num_virtual_tokens`的默认值设置为 50，将`max_length`的默认值设置为 512。如果你在实例化类时没有指定自己的默认值，你会得到这些值。在这种情况下，调用代码将它们分别设置为 20 和 512，但你可以自由实验。    接下来，代码设置 transformers 的`AutoModelForSequenceClassification`类以获取 BERT：    ```py         self.bert = AutoModelForSequenceClassification.from_pretrained(                         model_name,                          num_labels=2) ```    就像为 IMDb 进行微调一样，我们感兴趣的是训练模型以识别两个标签，所以它们在这里设置。然而，与微调的一个不同之处在于，我们不会更改 BERT 模型内部的任何权重，所以我们将其设置为不想要梯度并冻结，如下所示：    ```py self.bert.requires_grad_(False) ```    我们将要使用的软提示的“秘密配方”在初始化的末尾。我们将创建一个向量来包含我们的虚拟标记数量，我刚刚用词汇表中的随机标记初始化了它。我们可能在这里做更聪明的事情来使训练随着时间的推移更有效率，但为了简单起见，让我们就这样做：    ```py token_ids = torch.randint(0, vocab_size, (num_virtual_tokens,)) ```    transformers 中的预训练 BERT 模型包含嵌入，所以我们可以使用它们将我们的随机标记列表转换为嵌入：    ```py word_embeddings = self.bert.bert.embeddings.word_embeddings prompt_embeddings = word_embeddings(token_ids).unsqueeze(0) ```    重要的是，我们现在应该指定`prompt_embeddings`是神经网络的参数。这将在我们定义优化器时很重要。我们最近指定了所有 BERT 参数都被冻结，但 *这些* 参数不包含在其中，因此它们不会被冻结，所以它们将在训练过程中被优化器调整：    ```py self.prompt_embeddings = nn.Parameter(prompt_embeddings) ```    我们现在初始化了一个可调整 BERT 的子类版本，指定我们不希望修改其梯度，并创建了一组软提示，我们将在训练过程中将其附加到示例上——我们只调整这些软提示以软调整两个输出神经元。    现在，让我们看看在训练时前向传递期间将被调用的`forward`函数。鉴于我们已经设置了一切，这很简单：    ```py def forward(self, input_ids, attention_mask, labels=None):     batch_size = input_ids.shape[0]     input_ids = input_ids[:, :self.max_length]     attention_mask = attention_mask[:, :self.max_length]       embeddings = self.bert.bert.embeddings.word_embeddings(input_ids)     prompt_embeddings = self.prompt_embeddings.expand(batch_size, –1, –1)     inputs_embeds = torch.cat([prompt_embeddings, embeddings], dim=1)       prompt_attention_mask = torch.ones(batch_size, self.n_tokens,                                         device=attention_mask.device)     attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)       return self.bert(         inputs_embeds=inputs_embeds,         attention_mask=attention_mask,         labels=labels,         return_dict=True     ) ```    让我们一步一步地看看。在训练时的前向传递中，这个函数将传递数据批次。因此，我们需要了解这个批次的大小，然后提取特定 ID 的`input_ids`（从数据集中读取的值）和该 ID 的注意力掩码：    ```py     batch_size = input_ids.shape[0]     input_ids = input_ids[:, :self.max_length]     attention_mask = attention_mask[:, :self.max_length]   ```    我们还需要将`input_ids`转换为嵌入：    ```py embeddings = self.bert.bert.embeddings.word_embeddings(input_ids) ```    我们的软提示也是标记化句子。最初，它们被初始化为随机单词，随着时间的推移，我们会看到它们会相应地调整。但在这个步骤中，这些标记需要被转换为嵌入：    ```py prompt_embeddings = self.prompt_embeddings.expand(batch_size, –1, –1)   ```    `expand`方法只是将批次大小添加到提示嵌入中。当我们定义类时，我们不知道每个传入的批次的大小（代码编写得可以根据你可用内存的大小进行调整），所以使用`expand(batch_size, –1, –1)`将提示嵌入的向量从形状为`[1, num_prompt_tokens, embedding_dimensions]`转换为`[batch_size, num_prompt_tokens, embedding_dimensions]`。    我们的软提示微调涉及将软嵌入添加到实际输入数据的嵌入之前，所以我们这样做：    ```py inputs_embeds = torch.cat([prompt_embeddings, embeddings], dim=1) ```    BERT 使用`attention_mask`来过滤掉在训练或推理时间我们不关心的标记，这通常是填充标记。但我们要 BERT 关注所有的软提示标记，所以我们将它们的注意力掩码设置为全 1，然后将其附加到传入的训练数据的注意力掩码上。以下是代码：    ```py prompt_attention_mask = torch.ones(batch_size, self.n_tokens,                                     device=attention_mask.device)   attention_mask = torch.cat([prompt_attention_mask, attention_mask],                              dim=1)   ```    现在我们已经完成了所有的调整，我们需要将数据传递给模型以进行优化和计算损失：    ```py     return self.bert(         inputs_embeds=inputs_embeds,         attention_mask=attention_mask,         labels=labels,         return_dict=True     )   ```    我们将在下一个训练循环中看到这些数据是如何被使用的。    ## 训练模型    这个训练的关键是，我们将进行完整的、正常的训练循环，但在特殊情况下。在这种情况下，我们之前冻结了 BERT 模型中的 *一切*，*除了* 我们定义的软提示，这些软提示被视为模型参数。因此，假设我们定义优化器如下：    ```py optimizer = AdamW(model.parameters(), lr=1e-2) ```    在这种情况下，我们使用标准的代码，告诉它调整模型的参数。但唯一可调整的是软提示，所以这应该很快！    注意，学习率的值相当大。这有助于系统快速学习，但在实际系统中，你可能会想要这个值更小——或者至少是可调整的，开始时很大，然后在后续的周期中缩小。    所以现在，让我们进入训练。首先，我们将设置训练循环：    ```py num_epochs = 3   # Perform the training for epoch in range(num_epochs):     model.train()     total_train_loss = 0 ```    ### 管理数据批次    对于每个批次，我们将获取列（`input_ids`和`attention_masks`）以及标签，并将它们传递给模型：    ```py for batch in tqdm(train_dataloader,                    desc=f'Training Epoch {epoch + 1}'):     batch = {k: v.to(device) for k, v in batch.items()}     labels = batch.pop('label')     outputs = model(**batch, labels=labels) ```    这看起来与本书前面的代码略有不同，但它基本上在做同样的事情。`tqdm`代码只是给我们一个状态栏，因为我们正在训练。我们按批次读取数据，但我们希望数据在模型相同的设备上。例如，如果模型在 GPU 上运行，我们希望它访问 GPU 的内存。因此，此行将遍历每个列，读取键并将值传递到设备：    ```py batch = {k: v.to(device) for k, v in batch.items()} ```    它重新定义了读取的批次，以确保数据与模型在相同的设备上。但我们不希望标签在批次中，因为模型期望它们单独提供，所以我们使用`pop()`方法从批次中删除它们：    ```py labels = batch.pop('label') ```    现在，我们可以使用`**batch`的简写来将输入值（在这种情况下，`input_ids`和`attention_mask`）传递给模型的`forward`方法，并像这样展开字典以及标签：    ```py outputs = model(**batch, labels=labels) ```    ### 处理损失    前向传递将数据发送到模型，并返回损失。我们使用它来更新我们的总损失，然后进行反向传递：    ```py loss = outputs.loss total_train_loss += loss.item()   loss.backward() ```    随着梯度的反向流动，优化器现在可以开始工作了。    ### 优化损失    记住`model.parameters()`只会管理 *可训练未冻结* 的参数，我们现在可以调用优化器。我在这里添加了一个名为 *梯度裁剪* 的东西，以使训练更有效率，但其余的只是调用优化器的下一步，然后清空梯度，以便我们下次可以使用它们：    ```py clip_grad_norm_(model.parameters(), max_grad_norm)  # Add here optimizer.step() optimizer.zero_grad() ```    ###### 注意    *梯度裁剪* 的想法是，在反向传播过程中，梯度有时可能太大，优化器可能会采取非常大的步骤。这可能导致一个称为 *梯度爆炸* 的问题，其中值的变化隐藏了可能学到的细微差别。但是，如果梯度值变得太大，裁剪会将其缩小，在这种情况下，它们可能甚至不是必要的。    ## 训练过程中的评估    我们还有一个测试数据集，所以我们可以评估模型在训练周期中的表现。在每个周期中，一旦前向和反向传递完成，并且模型参数被重置，我们可以将模型切换到评估模式，然后开始将所有测试数据传递给它以进行推理。我们还将比较推理结果与实际标签，以计算准确率：    ```py model.eval() val_accuracy = [] total_val_loss = 0 ```    然后，我们将有类似的代码——但这次，它将用于读取评估批次，将它们转换为带有标签的输出，并从模型获取预测和损失值：    ```py with torch.no_grad():     for batch in tqdm(eval_dataloader, desc='Validating'):         batch = {k: v.to(device) for k, v in batch.items()}         labels = batch.pop('label')           outputs = model(**batch, labels=labels)         total_val_loss += outputs.loss.item()           predictions = torch.argmax(outputs.logits, dim=-1)         val_accuracy.extend((predictions == labels).cpu().numpy()) ```    一旦我们计算了这些值，那么在每个周期的末尾，我们可以报告它们以及训练损失。    ## 报告训练指标    在每个周期中训练时，我们计算了训练损失，所以我们现在可以获取所有记录的平均值。我们可以用同样的方法来计算验证损失，以及（当然）验证准确率，然后报告它们：    ```py avg_train_loss = total_train_loss / len(train_dataloader) avg_val_loss = total_val_loss / len(eval_dataloader) val_accuracy = np.mean(val_accuracy)   print(f"\nEpoch {epoch + 1}:") print(f"Average training loss: {avg_train_loss:.4f}") print(f"Average validation loss: {avg_val_loss:.4f}") print(f"Validation accuracy: {val_accuracy:.4f}") ```    在 Colab 的 A100 上运行三个周期，我们得到这个：    ```py Training Epoch 1: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s] Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]  Epoch 1: Average training loss: 0.6559 Average validation loss: 0.6037 Validation accuracy: 0.8036 Training Epoch 2: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s] Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]   Epoch 2: Average training loss: 0.6112 Average validation loss: 0.5854 Validation accuracy: 0.8386 Training Epoch 3: 100%|██████████| 79/79 [01:01<00:00, 1.28it/s] Validating: 100%|██████████| 40/40 [00:27<00:00, 1.44it/s]   Epoch 3: Average training loss: 0.5799 Average validation loss: 0.5270 Validation accuracy: 0.8736  ```    这是在 Colab 的 A100 上使用 40 Gb 的 GRAM 完成的，正如你所看到的，每个周期只花了大约 1 分钟来训练和 30 秒来评估。    到最后，平均训练损失从大约 0.65 下降到 0.58。准确率为 0.8736。所以，它很可能是过拟合，因为我们只训练了三个周期。    ## 保存提示嵌入    这种方法的真正好处是，你可以在完成后简单地保存提示嵌入。你还可以在稍后将其加载回来以进行推理，正如你将在下一节中看到的：    ```py torch.save(model.prompt_embeddings, "imdb_prompt_embeddings.pt") ```    我觉得这很酷的是，这个文件相对较小（61 K），并且它不需要你以任何方式修改底层模型。因此，在实际应用中，你可能会有一系列这样的 prompt-tuning 文件，并根据需要热插拔和替换它们，这样你就可以拥有多个模型，你可以对它们进行编排，这是代理解决方案的基础。    ## 使用模型进行推理    要使用提示调整后的模型进行推理，你只需定义具有软提示的模型，然后，而不是训练它们，从磁盘加载预训练的软提示。我们将在本节中探讨这一点。如果你不想训练自己的模型，那么 [在下载](https://github.com/lmoroney/PyTorch-Book-FIles) 中，我提供了一种经过 30 个周期训练而不是 3 个周期的模型的软提示。    为了更整洁的封装，我创建了一个类似于我们用于训练的类，但它只是用于推理。我称之为`PromptTunedBERTInference`，以下是它的初始化器：    ```py class PromptTunedBERTInference:     def __init__(self, model_name="bert-base-uncased",                         prompt_path="imdb_prompt_embeddings.pt"):         self.tokenizer = AutoTokenizer.from_pretrained(model_name)         self.model =                     AutoModelForSequenceClassification.from_pretrained(                                             model_name, num_labels=2)         self.model.eval()         self.prompt_embeddings = torch.load(prompt_path)         self.device =             torch.device('cuda' if torch.cuda.is_available() else 'cpu')         self.model.to(self.device) ```    它与可训练的初始化器非常相似，但有两个重要点。第一个是，因为我们 *只* 使用它进行推理，所以我将其设置为评估模式：    ```py self.model.eval() ```    第二个是，我们不需要训练嵌入并执行所有相关的管道——我们只需从指定的路径加载它们：    ```py self.prompt_embeddings = torch.load(prompt_path) ```    就这样！正如你所看到的，它非常轻量级，非常简单。它不会有`forward`函数，因为我们不训练它，但让我们添加一个`predict`函数，它封装了使用它进行推理的过程。    ### 预测函数    `predict`函数的职责是接受我们想要进行推理的字符串（s），对其进行标记化（它们），然后将其与软标记一起传递给模型。让我们逐个分析代码。    首先，让我们定义它，并让它接受它将对其进行标记化的文本：    ```py def predict(self, text):     inputs = self.tokenizer(text, padding=True,                          truncation=True,                         max_length=512-self.prompt_embeddings.shape[1],                         return_tensors="pt")     inputs = {k: v.to(self.device) for k, v in inputs.items()} ```    文本将被标记化到最大长度，减去软提示的大小，然后输入中的每个项都将被加载到字典中。请注意，标记化器将为文本返回多个列——通常是标记和注意力掩码——所以我们将遵循这种方法将它们转换为易于我们稍后使用的键值对集合。    现在我们有了输入，是时候将它们传递给模型了。我们将首先将`torch`设置为`no_grad()`模式，因为我们不感兴趣于训练梯度。然后，我们将获取每个标记的嵌入：    ```py with torch.no_grad():     embeddings = self.model.bert.embeddings.word_embeddings(                                         inputs['input_ids'])       batch_size = embeddings.shape[0]       prompt_embeds = self.prompt_embeddings.expand(                          batch_size, –1, –1).to(self.device)       inputs_embeds = torch.cat([prompt_embeds, embeddings], dim=1) ```    我们有一个由标记化器生成的输入的注意力掩码，但没有软提示的注意力掩码。所以，让我们创建一个并将其附加到输入的注意力掩码上：    ```py attention_mask = inputs['attention_mask'] prompt_attention = torch.ones(batch_size, self.prompt_embeddings.shape[1],                             device=self.device) attention_mask = torch.cat([prompt_attention, attention_mask], dim=1) ```    现在我们已经准备好了所有东西，我们可以将我们的数据传递给模型以获取我们的推理结果：    ```py outputs = self.model(inputs_embeds=inputs_embeds,                    attention_mask=attention_mask) ```    输出将是两个神经元的 logits，一个表示积极情绪，另一个表示消极情绪。然后我们可以对它们进行 Softmax 以获得预测：    ```py probs = torch.nn.functional.softmax(outputs.logits, dim=-1) return {"prediction": outputs.logits.argmax(-1).item(),        "confidence": probs.max(-1).values.item()} ```    ### 使用示例    使用这个类进行预测非常简单。我们创建这个类的实例，并将一个字符串传递给它以获取结果。结果将包含一个预测和一个置信度值，然后我们可以输出它们：    ```py # Usage example if __name__ == "__main__":     model = PromptTunedBERTInference()     result = model.predict("This movie was great!")     print(f"Prediction: {'Positive'                            if result['prediction'] == 1 else 'Negative'}")     print(f"Confidence: {result['confidence']:.2f}")   ```    可能会看到的一个关于 prompt tuning 的注意事项是低置信度值，这可能导致错误预测，尤其是在像这种二进制分类器中。确保你的推理工作良好是很好的，还有其他一些技术你可以探索以确保 logits 给出你想要的值。这包括设置 Softmax 的温度、使用更多的提示标记以给模型更多的容量，以及用与情感相关的单词初始化提示标记（而不是像我们这里这样使用随机标记）。    # 摘要    在本章中，我们探讨了使用我们自己的数据定制 LLMs 的不同方法。我们研究了两种主要方法：传统的微调和 prompt tuning。    使用 IMDb 数据集，你看到了如何微调 BERT 进行情感分析，并详细介绍了所有步骤——从数据准备到模型配置，再到训练和评估。该模型在几个周期内实现了令人印象深刻的 95%的情感分类准确率。    然而，微调可能并不适用于所有情况，为此，你探索了一种轻
