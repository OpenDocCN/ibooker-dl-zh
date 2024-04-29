# 第四章：GPT-4 和 ChatGPT 的高级技术

现在你已经熟悉了 LLM 的基础知识和 OpenAI API，是时候将你的技能提升到下一个水平了。本章涵盖了强大的策略，将使你能够充分利用 ChatGPT 和 GPT-4 的潜力。从提示工程、零-shot 学习和少-shot 学习到为特定任务微调模型，本章将为你提供创建任何你能想象的应用所需的所有知识。

# 提示工程

在深入研究提示工程之前，让我们简要回顾一下聊天模型的`completion`函数，因为本节将广泛使用它。为了使代码更紧凑，我们将定义该函数如下：

```py
def chat_completion(prompt, model="gpt-4", temperature=0):
    res = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    print(res["choices"][0]["message"]["content"])
```

该函数接收一个提示，并在终端显示完成的结果。模型和温度是两个可选的特性，默认分别设置为 GPT-4 和 0。

为了演示提示工程，我们将回到示例文本“正如笛卡尔所说，我思故我在”。如果将此输入传递给 GPT-4，模型自然会通过迭代添加最有可能的标记来完成句子：

```py
chat_completion("As Descartes said, I think therefore")
```

因此，我们得到以下输出消息：

```py
I am. This famous philosophical statement, also known as "Cogito, ergo sum," 
emphasizes the existence of the self through the act of thinking or doubting.
Descartes used this statement as a foundational principle in his philosophy,
arguing that one's own existence is the most certain and indubitable fact
that can be known.
```

*提示工程*是一门新兴的学科，专注于制定构建 LLM 的最佳输入的最佳实践，以便尽可能以编程方式产生理想的输出。作为一名 AI 工程师，你必须知道如何与 AI 交互，以获取可利用的结果，如何提出正确的问题，以及如何编写高质量的提示；这些都是我们将在本节中涵盖的所有主题。

应该注意，提示工程可能会影响使用 OpenAI API 的成本。你将支付的金额与你发送和接收的令牌数量成正比。如在第二章中提到的，强烈建议使用`max_token`参数，以避免账单上的不愉快的惊喜。

还要注意，你应该考虑在`openai`方法中可以使用的不同参数，因为如果你使用`temperature`、`top_p`和`max_token`等参数，即使使用相同的提示，也可能得到显著不同的结果。

## 设计有效的提示

通过提示可以执行许多任务。它们包括摘要、文本分类、情感分析和问题回答。在所有这些任务中，通常在提示中定义三个要素：角色、上下文和任务，如图 4-1 所示。

![](img/dev-app-gpt4-cgpt-dagc_0401.png)

###### 图 4-1。一个有效的提示

这三个要素并非总是必要的，它们的顺序也可以改变，但如果你的提示构造得当，要素定义得当，你应该能够得到良好的结果。请注意，即使在使用这三个要素时，对于复杂的任务，你可能需要使用更高级的技术，如零-shot 学习、少-shot 学习和微调。这些高级技术将在本章后面讨论。

### 上下文

提示中的第一个基本要素是上下文。当你为 LLM 编写输入文本时，你必须尽可能详细地描述上下文。为了说明这一点，假设你想使用 GPT-4 创建一个应用程序，为午餐建议主菜。接下来，我们将比较在两种不同上下文下获得的结果。第一个上下文将有很少的细节，第二个将有更多的细节。

对于一个简短的输入消息，比如：

```py
chat_completion("Give me a suggestion for the main course for today's lunch.")
```

我们得到以下输出消息：

```py
Grilled chicken with roasted vegetables and quinoa.
```

现在，关于输入消息的上下文的更多细节：

```py
prompt = """
Context: I do 2 hours of sport a day. I am vegetarian, and I don't like green 
vegetables. I am conscientious about eating healthily. 
Task: Give me a suggestion for a main course for today's lunch."""
chat_completion(prompt)
```

我们得到以下输出消息：

```py

For today's lunch, you can try a Chickpea and Sweet Potato Curry served with
Quinoa. This dish is packed with protein, fiber, and essential nutrients,
while also being vegetarian and not relying on green vegetables. The curry
can be made with chickpeas, sweet potatoes, tomatoes, onions, garlic, and a
blend of spices like turmeric, cumin, and coriander. Cook the quinoa
separately and serve it alongside the curry for a delicious and healthy meal.
```

在第二个例子中，句子更加详细，因为模型有更多的上下文：它建议了一个富含蛋白质的健康素食菜肴。

上下文指导 GPT 的“思考”，为其回应增加价值。构建具有良好上下文的提示是一个迭代的过程，通常需要试错步骤。有时确定上下文必须包含的要点是有些棘手的。为了确定可能的改进，您可以向 GPT-4 寻求建议。在以下示例中，我们改变了提示：不是要求模型回答我们的问题，而是指示它在努力实现更好的完成时询问上下文的问题。

使用以下输入消息：

```py
prompt = """
Context: I do 2 hours of sport a day. I am vegetarian and I don't like green 
vegetables. I am very careful to eat healthily.
Task: Give me a suggestion for a main course for today's lunch?
Do not perform the requested task! Instead, can you ask me questions about the 
context so that when I answer, you can perform the requested task more
efficiently?
"""
chat_completion(prompt)
```

我们从 GPT-4 模型中得到以下问题：

```py
Sure, I have a few questions to help me suggest a suitable main course for you:
1\. What type of sport do you do, and do you have any specific nutritional
  requirements related to it?
2\. Are there any other vegetables or ingredients that you particularly enjoy
  or dislike?
3\. Do you prefer a hot or cold meal for lunch?
4\. Are you open to trying new recipes or do you prefer sticking to
  familiar dishes?
```

模型提出的问题通常是相关的。根据这些问题，您可以更新提示中的上下文。因此，我们建议您在构建提示时经常使用这个技巧。

### 任务

任务定义了您对 GPT-4 模型的使用，并且应该明确定义和具体。您应该为任务提供足够的信息，并在提示中使用适当的短语来引导模型达到期望的结果。

在前面的部分中，任务是让模型建议午餐的主菜。我们可以添加细节来澄清我们想要什么。在这里，我们要求模型添加一个表格，其中包含每种食材和所需数量（以克为单位）：

```py
prompt = """
Context: I do 2 hours of sport a day. I am vegetarian, and I don't like green 
vegetables. I am conscientious about eating healthily.
Task: Give me a suggestion for a main course for today's lunch.
With this suggestion, I also want a table with two columns where each row 
contains an ingredient from the main course.
The first column in the table is the name of the ingredient.
The second column of the table is the number of grams of that ingredient needed 
for one person. Do not give the recipe for preparing the main course.
"""
chat_completion(prompt)
```

该模型给出了以下输出：

```py
Suggestion for today's lunch: Stuffed Bell Peppers with Quinoa and Black Beans
| Ingredient          | Grams per person |
|---------------------|------------------|
| Bell peppers        | 200g             |
| Quinoa              | 60g              |
| Black beans         | 75g              |
| Onion               | 50g              |
| Garlic              | 5g               |
| Olive oil           | 10g              |
| Tomato sauce        | 50g              |
| Cumin               | 1g               |
| Chili powder        | 1g               |
| Salt                | 1g               |
| Pepper              | 1g               |
| Shredded cheese     | 30g              |
```

在[OpenAI API 示例页面](https://platform.openai.com/examples)上，有一个包含 48 个 GPT 模型可以执行的任务示例列表，始终与相关提示和演示一起。虽然这些示例使用了 GPT-3 模型和完成端点，但对于聊天端点原则是相同的，这些示例很好地说明了如何向 OpenAI 模型提供任务。我们不会在这里逐个讨论它们，但以下是其中的一些：

语法纠正

纠正句子到标准英语。

提示：

```py
Correct this to standard English: She no went to the market.
```

为二年级学生总结

将复杂的文本转化为更简单的概念。

提示：

```py
Summarize this for a second-grade student: Jupiter is the fifth planet [...]
```

TL;DR 总结

TL;DR 代表“太长了；没读”。已经观察到，可以通过简单地在末尾添加`T``l;dr`来总结一段文本。

提示：

```py
A neutron star [...] atomic nuclei. Tl;dr
```

Python 转自然语言

用人们能理解的语言解释一段 Python 代码。

提示：

```py
# Python 3 
def hello(x): 
print('hello '+str(x)) 
# Explanation of what the code does
```

计算时间复杂度

找到函数的时间复杂度。

提示：

```py
# Python 3 
def hello(x, n):
     for i in range(n):
        print('hello '+str(x))
# The time complexity of this function is 
```

Python 错误修复程序

修复包含错误的代码。

提示：

```py
### Buggy Python
def hello(x, n):
     for i in rang(n):
        print('hello '+str(x))
### Fixed Python
```

SQL 请求

简单的 SQL 查询构建。

提示：

```py
Create a SQL request to find all users who live in California and have
over 1000 credits.
```

类比制造者

可以对两个单词进行类比。

提示：

```py
Create an analogy for this phrase. Questions are arrows in that:
```

总结笔记

总结会议记录。

提示：

```py
Convert my shorthand into a first-hand account of the meeting:
Tom: Profits up 50%
Jane: New servers are online
Kjel: Need more time to fix software
```

### 角色

在撰写提示时影响模型的最后一种方法是给它一个角色。当然，角色和上下文可以独立使用，但同时使用可以增加您对模型输出的控制。在以下示例中，我们让模型扮演体育营养专家的角色：

```py
prompt = """
Role: You are a nutritionist designing healthy diets for high-performance 
athletes. You take into account the nutrition needed for a good recovery.
Context: I do 2 hours of sport a day. I am vegetarian, and I don't like green 
vegetables. I am conscientious about eating healthily.
Task: Based on your expertise defined in your role, give me a suggestion for a 
main course for today's lunch. With this suggestion, I also want a table with
two columns where each row in the table contains an ingredient from the main 
course. The first column in the table is the name of the ingredient.
The second column of the table is the number of grams of that ingredient needed
for one person. Do not give the recipe for preparing the main course.
"""
chat_completion(prompt)
```

然后我们得到以下结果：

```py
Main Course Suggestion: Quinoa and Chickpea Stuffed Bell Peppers
Ingredient Table:
| Ingredient          | Grams per Person |
|---------------------|------------------|
| Quinoa              | 60               |
| Chickpeas           | 100              |
| Bell Peppers        | 200              |
| Olive Oil           | 10               |
| Onion               | 50               |
| Garlic              | 5                |
| Tomato              | 100              |
| Feta Cheese         | 30               |
| Kalamata Olives     | 20               |
| Fresh Basil         | 5                |
| Salt                | 2                |
| Black Pepper        | 1                |
```

正如您所见，提示可以用来调整 GPT 模型等 LLM 的概率分布集。它们可以被视为指导模型产生特定类型结果的指南。虽然提示设计没有确定的结构，但一个有用的框架是上下文、角色和任务的组合。

重要的是要理解，这只是一种方法，提示可以在不明确定义这些元素的情况下创建。有些提示可能会受益于不同的结构，或者根据您的应用程序的特定需求需要更有创意的方法。因此，这个上下文-角色-任务框架不应限制您的思维，而应该是帮助您在适当时有效地设计提示的工具。

## 逐步思考

众所周知，GPT-4 不擅长计算。它无法计算 369 × 1,235：

```py
prompt = "How much is 369 * 1235?"
chat_completion(prompt)
```

我们得到以下答案：`454965`

正确答案是 455,715。GPT-4 不能解决复杂的数学问题吗？请记住，模型通过预测答案中的每个标记来逐个顺序生成答案，从左边开始。这意味着 GPT-4 首先生成最左边的数字，然后将其作为上下文的一部分来生成下一个数字，依此类推，直到形成完整的答案。这里的挑战是每个数字都是独立于最终正确值的预测。GPT-4 将数字视为标记；没有数学逻辑。

###### 注意

在第五章中，我们将探讨 OpenAI 如何通过插件丰富了 GPT-4。一个例子是用于提供准确数学解决方案的计算器插件。

有一个技巧可以增加语言模型的推理能力。例如，当要求解 369×1235 时，我们可以看到模型试图直接在一次尝试中回答。考虑到你可能也无法在没有铅笔和一张纸的帮助下解决这个乘法。可以通过提示来鼓励模型进行中间推理。就像你用铅笔和纸一样，如果给模型足够的时间来推理，它就可以解决更复杂的问题。

在提示的末尾添加“让我们一步一步地思考”已经被实验证明能够使模型解决更复杂的推理问题。这种技术被称为*零射击 CoT 策略*，它是由 Kojima 等人在 2022 年发表的科学论文[“大型语言模型是零射击推理者”](https://oreil.ly/2tHHy)中介绍的。

*CoT*代表*思维链*；它在这里指的是使用鼓励模型模仿逐步推理的提示技术。术语*零射击*意味着模型不依赖于特定任务的示例来执行这种推理；它准备好根据其一般训练来处理新任务。与需要模型看到任务示例的少量学习等技术不同（我们将很快讨论），零射击学习试图在不需要特定任务示例的情况下进行泛化。

正如我们将看到的，通过将这个句子添加到提示中，模型开始通过将问题分解为子问题来进行推理。因此，它可能需要时间来推理，从而能够找到以前只能在一次尝试中无法解决的问题的解决方案。让我们看看如果我们改变我们的输入提示会发生什么：

```py
prompt = "How much is 369 * 1235 ? Let's think step by step."
chat_completion(prompt)
```

我们得到了以下输出：

```py
Step 1: Multiply 369 by 5 (the last digit of 1235).
369 * 5 = 1845
Step 2: Multiply 369 by 30 (the second last digit of 1235 is 3, so multiply by
   3 and add a 0 at the end).
369 * 30 = 11070
Step 3: Multiply 369 by 200 (the third digit of 1235 is 2, so multiply by 2 and
   add two 0s at the end).
369 * 200 = 73800
Step 4: Multiply 369 by 1000 (the first digit of 1235 is 1, so multiply by 1 and 
   add three 0s at the end).
369 * 1000 = 369000
Step 5: Add the results from steps 1 to 4.
1845 + 11070 + 73800 + 369000 = 455715
So, 369 * 1235 = 455,715.
```

模型的方法发生了显著的转变。它现在将主要问题分解为更小、更易管理的步骤，而不是试图直接解决问题。

###### 警告

尽管提示模型“一步一步地思考”是至关重要的，但您仍然需要仔细评估其响应，因为 GPT-4 并非万无一失。对于像 3695×123548 这样更复杂的计算，即使使用这个技巧，LLM 也无法找到正确的解决方案。

当然，很难从一个例子中判断这个技巧是否普遍有效，或者我们只是幸运。在各种数学问题的基准测试中，实证实验证明这个技巧显著提高了 GPT 模型的准确性。尽管这个技巧对大多数数学问题都有效，但并不适用于所有情况。《大型语言模型是零射击推理者》的作者发现，它对多步算术问题、涉及符号推理的问题、涉及策略的问题以及其他涉及推理的问题最有益。它并不适用于常识问题。

## 实施少量学习

*少样本学习*，由 Brown 等人在《语言模型是少样本学习者》中介绍，指的是 LLM 仅凭少量示例就能概括和产生有价值的结果的能力。通过少样本学习，您可以给出您希望模型执行的任务的几个示例，如图 4-2 所示。这些示例指导模型处理所需的输出格式。

![](img/dev-app-gpt4-cgpt-dagc_0402.png)

###### 图 4-2。包含几个示例的提示

在这个例子中，我们要求 LLM 将特定单词转换为表情符号。很难想象要在提示中放入什么指令来执行这个任务。但是通过少样本学习，这很容易。给它一些示例，模型将自动尝试复制它们。

```py
prompt="""I go home -->  go my dog is sad --> my  is I run fast -->  run I love my wife -->  my wifethe girl plays with the ball --> the  with the The boy writes a letter to a girl --> """chat_completion(prompt)
```

从前面的例子中，我们得到以下消息作为输出：

```py
The   a  to a 
```

少样本学习技术提供了具有期望输出的输入示例。然后，在最后一行，我们提供了我们想要完成的提示。这个提示与之前的示例的形式相同。自然地，语言模型将根据给定示例的模式执行完成操作。

我们可以看到，仅凭几个示例，模型就可以复制指令。通过利用 LLM 在训练阶段获得的广泛知识，它们可以快速适应并根据仅有的几个示例生成准确的答案。

###### 注意

少样本学习是 LLM 的一个强大方面，因为它使它们能够高度灵活和适应，只需要有限的额外信息就能执行各种任务。

当您在提示中提供示例时，确保上下文清晰和相关是至关重要的。清晰的示例可以提高模型匹配所需输出格式并执行解决问题的能力。相反，不充分或含糊的示例可能导致意外或不正确的结果。因此，仔细编写示例并确保它们传达正确的信息可以显著影响模型执行任务的准确性。

引导 LLM 的另一种方法是*一样本学习*。顾名思义，在这种情况下，您只提供一个示例来帮助模型执行任务。尽管这种方法提供的指导比少样本学习少，但对于更简单的任务或者 LLM 已经对主题有相当多的背景知识的情况下，它可能是有效的。一样本学习的优势在于简单、更快的提示生成以及更低的计算成本和因此更低的 API 成本。然而，对于复杂的任务或需要更深入理解期望结果的情况，少样本学习可能是更适合的方法，以确保准确的结果。

###### 提示

提示工程已成为一个热门话题，您会发现许多在线资源来深入研究这个主题。例如，这个[GitHub 存储库](https://github.com/f/awesome-chatgpt-prompts)包含了由 70 多个不同用户贡献的有效提示列表。

虽然本节探讨了各种可以单独使用的提示工程技术，请注意您可以结合这些技术以获得更好的结果。作为开发人员，您的工作是找到特定问题的最有效提示。请记住，提示工程是一个反复试验的迭代过程。

## 提高提示效果

我们已经看到了几种提示工程技术，可以影响 GPT 模型的行为，以获得满足我们需求的更好结果。我们将以一些在编写 GPT 模型提示时可以在不同情况下使用的技巧和窍门结束本节。

### 指导模型提出更多问题

以询问模型是否理解问题并指示模型提出更多问题来结束提示是一种有效的技术，如果你正在构建基于聊天机器人的解决方案。你可以在提示的末尾添加这样的文本：

```py
Did you understand my request clearly? If you do not fully understand my request,
ask me questions about the context so that when I answer, you can
perform the requested task more efficiently.
```

### 格式化输出

有时候你会希望在更长的过程中使用 LLM 输出：在这种情况下，输出格式很重要。例如，如果你想要 JSON 输出，模型往往会在 JSON 块之前和之后写入输出。如果你在提示中添加“输出必须被 json.loads 接受”，那么它往往会工作得更好。这种技巧可以在许多情况下使用。

例如，使用这个脚本：

```py
prompt = """
Give a JSON output with 5 names of animals. The output must be accepted 
by json.loads.
"""
chat_completion(prompt, model='gpt-4')
```

我们得到了以下的 JSON 代码块：

```py
{
  "animals": [
    "lion",
    "tiger",
    "elephant",
    "giraffe",
    "zebra"
  ]
}
```

### 重复指令

经验表明，重复指令会产生良好的结果，特别是当提示很长时。思路是多次在提示中添加相同的指令，但每次都用不同的方式表达。

这也可以通过负面提示来实现。

### 使用负面提示

在文本生成的背景下使用负面提示是一种指导模型的方式，指定你不希望在输出中看到的内容。它们作为约束或指导，用于过滤出某些类型的响应。当任务复杂时，这种技术特别有用：当任务以不同方式多次重复时，模型往往更精确地遵循指令。

继续上一个例子，我们可以通过添加“不要在 json 之前或之后添加任何内容。”来坚持输出格式的负面提示。

在第三章中的第三个项目中，我们使用了负面提示：

```py
Extract the keywords from the following question: {user_question}. Do not answer
anything else, only the keywords.
```

如果没有这个提示的补充，模型往往不会遵循指令。

### 添加长度约束

长度约束通常是一个好主意：如果你只期望得到一个单词的答案或 10 个句子，就把它添加到你的提示中。这就是我们在第三章中在第一个项目中所做的：我们指定了“长度：100 个单词”来生成一篇合适的新闻文章。在第四个项目中，我们的提示也有一个长度指令：“如果你可以回答问题：ANSWER，如果你需要更多信息：MORE，如果你无法回答：OTHER。只回答一个”“单词”。如果没有最后一句，模型会倾向于形成句子，而不是遵循指令。

# 微调

OpenAI 提供了许多现成的 GPT 模型。虽然这些模型在各种任务上表现出色，但对特定任务或情境进行微调可以进一步提高它们的性能。

## 入门

假设你想为你的公司创建一个电子邮件回复生成器。由于你的公司在特定行业中使用特定词汇，你希望生成的电子邮件回复保留你当前的写作风格。有两种策略可以做到这一点：要么你可以使用之前介绍的提示工程技术来强制模型输出你想要的文本，要么你可以对现有模型进行微调。本节探讨了第二种技术。

对于这个例子，你必须收集大量包含有关你特定业务领域的数据的电子邮件，客户的询问以及对这些询问的回复。然后，你可以使用这些数据对现有模型进行微调，以学习你公司特定的语言模式和词汇。微调后的模型本质上是从 OpenAI 提供的原始模型中构建的新模型，其中模型的内部权重被调整以适应你的特定问题，使得新模型在类似于它在微调数据集中看到的示例的任务上提高了准确性。通过微调现有的 LLM，可以创建一个高度定制和专门针对你特定业务中使用的语言模式和词汇的电子邮件回复生成器。

图 4-3 说明了微调过程，其中使用特定领域的数据集来更新现有 GPT 模型的内部权重。目标是使新的微调模型在特定领域比原始 GPT 模型做出更好的预测。应该强调的是这是一个*新模型*。这个新模型在 OpenAI 服务器上：与以前一样，您必须使用 OpenAI 的 API 来使用它，因为它无法在本地访问。

![](img/dev-app-gpt4-cgpt-dagc_0403.png)

###### 图 4-3。微调过程

###### 注意

即使您使用自己的特定数据对 LLM 进行了微调，新模型仍然保留在 OpenAI 的服务器上。您将通过 OpenAI 的 API 与其进行交互，而不是在本地。

### 为特定领域的需求调整 GPT 基础模型

目前，微调仅适用于`davinci`、`curie`、`babbage`和`ada`基础模型。每个模型在准确性和所需资源之间都有一个权衡。作为开发者，您可以为您的应用程序选择最合适的模型：较小的模型，如`ada`和`babbage`，可能对于简单任务或资源有限的应用程序来说更快速和更具成本效益，而较大的模型`curie`和`davinci`提供了更先进的语言处理和生成能力，使它们成为更复杂任务的理想选择，其中更高的准确性至关重要。

这些是不属于 InstructGPT 模型系列的原始模型。例如，它们没有从人类在循环中进行强化学习阶段中受益。通过微调这些基础模型，例如根据自定义数据集调整其内部权重，您可以将它们定制为特定任务或领域。尽管它们没有 InstructGPT 系列的处理和推理能力，但它们通过利用其预训练的语言处理和生成能力为构建专门应用程序提供了坚实的基础。

###### 注意

对于微调，您必须使用基础模型；不可能使用指导模型。

### 微调与少样本学习

微调是一种*重新训练*现有模型的过程，以改善其性能并使其答案更准确。在微调中，您更新模型的内部参数。正如我们之前所看到的，少样本学习通过其输入提示向模型提供有限数量的良好示例，从而引导模型基于这些少量示例产生期望的结果。通过少样本学习，模型的内部参数不会被修改。

微调和少样本学习都可以用来增强 GPT 模型。微调产生了一个高度专业化的模型，可以为特定任务提供更准确和上下文相关的结果。这使其成为在大量数据可用的情况下的理想选择。这种定制确保生成的内容更符合目标领域的特定语言模式、词汇和语气。

少样本学习是一种更灵活和数据高效的方法，因为它不需要重新训练模型。当只有有限的示例可用或需要快速适应不同任务时，这种技术是有益的。少样本学习允许开发者快速原型设计和尝试各种任务，使其成为许多用例的多功能和实用选择。选择两种方法之间的另一个重要标准是，使用和训练使用微调的模型更昂贵。

微调方法通常需要大量数据。可用示例的缺乏经常限制了这种类型技术的使用。为了让您了解微调所需的数据量，您可以假设对于相对简单的任务或仅需要进行轻微调整时，您可以通过几百个输入提示及其相应的期望完成示例来获得良好的微调结果。这种方法适用于预训练的 GPT 模型在任务上表现良好，但需要轻微调整以更好地与目标领域对齐的情况。然而，对于更复杂的任务或在您的应用程序需要更多定制的情况下，您的模型可能需要使用成千上万个示例进行训练。例如，这可以对应我们之前提出的用例，即自动回复符合您写作风格的电子邮件。您还可以针对非常专业的任务进行微调，这种情况下，您的模型可能需要数十万甚至数百万个示例。这种微调规模可以带来显著的性能改进，并更好地适应特定领域。

###### 注意

*迁移学习*将从一个领域学到的知识应用到不同但相关的环境中。因此，您有时可能会听到与微调相关的*迁移学习*术语。

## 使用 OpenAI API 进行微调

本节将指导您如何使用 OpenAI API 调整 LLM 的过程。我们将解释如何准备您的数据，上传数据集，并使用 API 创建一个经过微调的模型。

### 准备您的数据

要更新 LLM 模型，需要提供一个包含示例的数据集。数据集应该是一个 JSONL 文件，其中每一行对应一个提示和完成的配对：

```py
{"prompt": "<prompt text>", "completion": "<completion text>"}
{"prompt": "<prompt text>", "completion": "<completion text>"}
{"prompt": "<prompt text>", "completion": "<completion text>"}
…
```

JSONL 文件是一个文本文件，每行代表一个单独的 JSON 对象。您可以使用它来高效地存储大量数据。OpenAI 提供了一个工具，帮助您生成这个训练文件。该工具可以接受各种文件格式作为输入（CSV、TSV、XLSX、JSON 或 JSONL），只要它们包含提示和完成列/键，并且输出一个准备好发送进行微调过程的训练 JSONL 文件。该工具还会验证并提供建议，以改善您的数据质量。

在您的终端中使用以下代码行运行此工具：

```py
$ openai tools fine_tunes.prepare_data -f <LOCAL_FILE>
```

该应用程序将提出一系列建议，以改善最终文件的结果；您可以接受或拒绝这些建议。您还可以指定选项`-q`，自动接受所有建议。

###### 注意

当您执行`pip install openai`时，此`openai`工具已安装并在您的终端中可用。

如果您有足够的数据，该工具将询问是否有必要将数据分成训练集和验证集。这是一种推荐的做法。算法将使用训练数据在微调过程中修改模型的参数。验证集可以衡量模型在未用于更新参数的数据集上的性能。

对 LLM 进行微调有赖于使用高质量的示例，最好由专家审查。在使用预先存在的数据集进行微调时，确保数据经过筛查，排除冒犯性或不准确的内容，或者如果数据集太大无法手动审核所有条目，则检查随机样本。

### 使您的数据可用

一旦您准备好带有训练示例的数据集，您需要将其上传到 OpenAI 服务器。OpenAI API 提供了不同的功能来操作文件。以下是最重要的功能：

上传文件：

```py
openai.File.create(
    file=open("out_openai_completion_prepared.jsonl", "rb"),
    purpose='fine-tune'
)
```

+   有两个必填参数：`file`和`purpose`。将`purpose`设置为`fine-tune`。这将验证用于微调的下载文件格式。此函数的输出是一个字典，您可以从中检索`id`字段中的`file_id`。目前，总文件大小可达 1GB。如需更多，请联系 OpenAI。

删除文件：

```py
openai.File.delete("file-z5mGg(...)")
```

+   一个参数是必需的：`file_id`。

列出所有上传的文件：

```py
openai.File.list()
```

+   例如，在开始微调过程时，检索文件的 ID 可能会有所帮助。

### 创建一个经过精细调整的模型

对上传的文件进行微调是一个简单的过程。端点`openai.FineTune.create()`在 OpenAI 服务器上创建一个作业，以从给定数据集中细化指定的模型。此函数的响应包含排队作业的详细信息，包括作业的状态、`fine_tune_id`和过程结束时模型的名称。

主要输入参数在表 4-1 中描述。

表 4-1。`openai.FineTune.create()`的参数

| 字段名称 | 类型|描述|
| --- | ---|---|
| `training_file` | 字符串 | 这是包含上传文件的`file_id`的唯一必填参数。您的数据集必须格式化为 JSONL 文件。每个训练示例都是一个具有`prompt`和`completion`键的 JSON 对象。 |
| `model` | 字符串 | 这指定了用于微调的基础模型。您可以选择`ada`、`babbage`、`curie`、`davinci`或先前调整的模型。默认的基础模型是`curie`。 |
| `validation_file` | 字符串 | 这包含具有验证数据的上传文件的`file_id`。如果提供此文件，数据将用于在微调过程中定期生成验证指标。 |
| `suffix` | 字符串 | 这是一个最多 40 个字符的字符串，添加到您的自定义模型名称中。 |

### 列出微调作业

可以通过以下函数在 OpenAI 服务器上获取所有微调作业的列表：

```py
openai.FineTune.list()
```

结果是一个包含所有精细调整模型信息的字典。

### 取消微调作业

可以通过以下函数立即中断在 OpenAI 服务器上运行的作业：

```py
openai.FineTune.cancel()
```

此功能只有一个必填参数：`fine_tune_id`。`fine_tune_id`参数是一个以`ft-`开头的字符串；例如，`ft-Re12otqdRaJ(...) `。它是在使用`openai.FineTune.​cre⁠ate()`函数创建作业后获得的。如果您丢失了`fine_tune_id`，可以使用`openai.FineTune.list()`来检索它。

## 微调应用

微调提供了一种强大的方式来增强各种应用程序中模型的性能。本节将介绍几种已经有效部署微调的用例。从这些例子中获得灵感！也许您在您的用例中有相同类型的问题。再次提醒，微调比基于提示工程的其他技术更昂贵，因此在大多数情况下并不是必需的。但是当需要时，这种技术可以显著改善您的结果。

### 法律文件分析

在这种情况下，LLM 用于处理法律文件并提取有价值的信息。这些文件通常使用特定行话编写，这使得非专业人士难以理解这些类型的文本。我们已经在第一章中看到，当在统一律师考试上进行测试时，GPT-4 的得分位于 90 分位数。在这种情况下，微调可以使模型专门针对特定领域和/或允许其协助非专业人士参与法律流程。通过在特定主题的法律语料库上对 LLM 进行微调，或者针对特定类型的最终用户，模型可以更好地处理法律语言的复杂性，并更擅长执行与特定类型最终用户相关的任务。

通过使用大量数据对 LLM 进行微调，以分析法律文件，可以显著提高模型在这些任务中的性能，使其能够更好地处理通常超出提示工程技术能力范围的法律语言的细微差别。

### 自动代码审查

在这种用例中，微调可以帮助模型分析并提出开发人员编写的代码的改进建议。这需要在包含代码片段和注释的数据集上训练模型，使模型能够处理特定语言的语法、语义和最佳实践。请注意，这个用例类似于 GitHub 的 Copilot 工具，该工具旨在通过实时提供代码和整个函数的建议来帮助开发人员编写代码。您可以使用微调来训练模型执行特定任务，如代码审查，并为自己的代码构建一个符合特定标准或专门用于特定编程框架的项目。

作为这种用例的输入文件的一个例子，您的 JSONL 文件可以包含代码和相应的审查评论对。这将帮助模型学习如何提供准确和相关的代码质量反馈。JSONL 文件的一行可能是这样的：

```py
{"prompt": "Review the following Python code: 
    'def sum(a, b): return a + b\nresult = sum(5, '5')'",
"completion": "Type error: The 'sum' function is adding an integer and a string. 
    Consider converting the string to an integer using int() before passing it
    to the function."}
```

### 财务文件摘要

在财务文件合成中，经过微调的 LLM 可以生成财务文件的简洁准确摘要，如收益报告、财务报表和分析报告。通过在与财务记录明确相关的数据集上微调语言模型，生成的模型可以更准确地理解这些文件的术语和上下文。例如，该模型可以将央行利率上调的详细报告压缩成简洁而富有信息的摘要。

要调整现有模型以获得更好的财务文件摘要，您需要已经有大量样本摘要。然后，您需要向模型提供一个包含财务文件及其理想摘要对的 JSONL 文件。例如：

```py
{"prompt": "The central bank announced a 0.25% increase in the benchmark 
    interest rate, citing concerns about inflationary pressures. This move is
    expected to affect borrowing costs for consumers and businesses.",
"completion": "Bank increases rate 0.25% for inflation worry. May affect people,
    business borrowing costs."}
```

### 技术文件翻译

使用微调来翻译技术文件可以显著提高基础模型的性能，与少量样本学习相比。主要原因是技术文件通常包含专业词汇和复杂的句子结构，少量样本学习无法有效处理。基础模型是未经调整的 GPT-3 模型，如 InstructGPT-3 模型，使用强化学习和人类反馈技术。要使用它们，需要进行微调。要调整现有的基础模型，您需要准备一个包含训练数据的 JSONL 文件。对于技术文件翻译的用例，该文件的内容将包括将技术文本翻译成目标语言的内容。

### 针对利基主题的新闻文章生成

在这种用例中，经过微调的模型可以为高度专业化的主题生成高质量、引人入胜和与上下文相关的新闻文章，这些主题基础模型可能没有足够的数据来准确生成。与所有其他用例一样，您需要创建一个训练数据集，以使您的模型专门用于撰写文章。为此，您需要准备许多关于特定利基主题的文章。这些数据将用于创建包含提示-完成对的 JSONL 文件。以下是一个小例子：

```py
{"prompt": "Write an introductory article about a new environmentally friendly
    cryptocurrency: 'EcoCoin: The Green Cryptocurrency Taking
    the Market by Storm'",
"completion": "As concerns over the environmental impact of cryptocurrency
    mining (...) mining process and commitment to sustainability."}
```

## 为电子邮件营销活动生成和微调合成数据

在这个例子中，我们将为电子邮件营销机构制作一个文本生成工具，利用定向内容为企业创建个性化的电子邮件营销活动。这些电子邮件旨在吸引受众并推广产品或服务。

假设我们的机构有一个支付处理行业的客户，他们要求帮助他们运行直接电子邮件营销活动，为电子商务提供新的支付服务。电子邮件营销机构决定为这个项目使用微调技术。我们的电子邮件营销机构将需要大量数据来进行微调。

在我们的情况下，我们需要为演示目的合成生成数据，正如您将在下一小节中看到的。通常，最好的结果是使用人类专家的数据，但在某些情况下，合成数据生成可能是一个有用的解决方案。

### 创建合成数据集

在以下示例中，我们从 GPT-3.5 Turbo 创建人工数据。为此，我们将在提示中指定要将促销句子发送给特定商家以销售电子商务服务。商家的特征是活动领域、商店所在城市和商店的大小。我们通过将提示发送到之前定义的`chat_completion`函数中的 GPT-3.5 Turbo 来获得促销句子。

我们通过定义三个列表来开始我们的脚本，分别对应于商店类型、商店所在的城市和商店的大小：

```py
l_sector = ['Grocery Stores', 'Restaurants', 'Fast Food Restaurants',
              'Pharmacies', 'Service Stations (Fuel)', 'Electronics Stores']
l_city = ['Brussels', 'Paris', 'Berlin']
l_size = ['small', 'medium', 'large'] 
```

然后我们在一个字符串中定义第一个提示。在此提示中，角色、上下文和任务都很明确，因为它们是使用本章前面描述的提示工程技术构建的。在此字符串中，大括号中的三个值将在代码中稍后替换为相应的值。这个第一个提示用于生成合成数据：

```py
f_prompt = """ 
Role: You are an expert content writer with extensive direct marketing 
experience. You have strong writing skills, creativity, adaptability to 
different tones and styles, and a deep understanding of audience needs and
preferences for effective direct campaigns.
Context: You have to write a short message in no more than 2 sentences for a
direct marketing campaign to sell a new e-commerce payment service to stores. 
The target stores have the following three characteristics:
- The sector of activity: {sector} `- The city where the stores are located:` `{city}`
`- The size of the stores:` `{size}` ``Task: Write a short message for the direct marketing campaign. Use the skills`
`defined in your role to write this message! It is important that the message`
`you create takes into account the product you are selling and the`
`characteristics of the store you are writing to.`
`"""``
```

以下提示仅包含三个变量的值，用逗号分隔。它不用于创建合成数据；仅用于微调：

```py
f_sub_prompt = "{sector}, {city}, {size}"
```

然后是代码的主要部分，它迭代我们之前定义的三个值列表。我们可以看到循环中的代码块很简单。我们用两个提示的大括号中的值替换为适当的值。变量`prompt`与函数`chat_completion`一起使用，以生成保存在`response_txt`中的广告。然后将`sub_prompt`和`response_txt`变量添加到*out_openai_completion.csv*文件中，这是我们微调的训练集。

```py
df = pd.DataFrame()
for sector in l_sector:
    for city in l_city:
        for size in l_size:
            for i in range(3):  ## 3 times each
                prompt = f_prompt.format(sector=sector, city=city, size=size)
                sub_prompt = f_sub_prompt.format(
                    sector=sector, city=city, size=size
                )
                response_txt = chat_completion(
                    prompt, model="gpt-3.5-turbo", temperature=1
                )
                new_row = {"prompt": sub_prompt, "completion": response_txt}
                new_row = pd.DataFrame([new_row])
                df = pd.concat([df, new_row], axis=0, ignore_index=True)
df.to_csv("out_openai_completion.csv",  index=False)
```

请注意，对于每种特征组合，我们生成三个示例。为了最大化模型的创造力，我们将温度设置为`1`。在此脚本结束时，我们有一个存储在*out_openai_completion.csv*文件中的 Pandas 表。它包含 162 个观察结果，其中有两列包含提示和相应的完成。这个文件的前两行如下：

```py
"Grocery Stores, Brussels, small",Introducing our new e-commerce payment service - 
the perfect solution for small Brussels-based grocery stores to easily and 
securely process online transactions. "Grocery Stores, Brussels, small",
Looking for a hassle-free payment solution for your small grocery store in
Brussels? Our new e-commerce payment service is here to simplify your
transactions and increase your revenue. Try it now!
```

现在我们可以调用工具从*out_openai_completion.csv*生成训练文件，如下所示：

```py
$ openai tools fine_tunes.prepare_data -f out_openai_completion.csv
```

正如您在以下代码行中所看到的，这个工具提出了改进我们提示-完成对的建议。在文本的结尾，它甚至提供了如何继续微调过程以及在微调过程完成后如何使用模型进行预测的建议。

```py
Analyzing...
- Based on your file extension, your file is formatted as a CSV file
- Your file contains 162 prompt-completion pairs
- Your data does not contain a common separator at the end of your prompts. 
Having a separator string appended to the end of the prompt makes it clearer
to the fine-tuned model where the completion should begin. See
https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
for more detail and examples. If you intend to do open-ended generation, 
then you should leave the prompts empty
- Your data does not contain a common ending at the end of your completions. 
Having a common ending string appended to the end of the completion makes it
clearer to the fine-tuned model where the completion should end. See
https://oreil.ly/MOff7 for more detail and examples.
- The completion should start with a whitespace character (` `). This tends to
produce better results due to the tokenization we use. See 
https://oreil.ly/MOff7 for more details
Based on the analysis we will perform the following actions:
- [Necessary] Your format `CSV` will be converted to `JSONL`
- [Recommended] Add a suffix separator ` ->` to all prompts [Y/n]: Y
- [Recommended] Add a suffix ending `\n` to all completions [Y/n]: Y
- [Recommended] Add a whitespace character to the beginning of the completion
[Y/n]: Y
Your data will be written to a new JSONL file. Proceed [Y/n]: Y
Wrote modified file to `out_openai_completion_prepared.jsonl`
Feel free to take a look!
Now use that file when fine-tuning:
> openai api fine_tunes.create -t "out_openai_completion_prepared.jsonl"
After you’ve fine-tuned a model, remember that your prompt has to end with the 
indicator string ` ->` for the model to start generating completions, rather
than continuing with the prompt. Make sure to include `stop=["\n"]` so that the
generated texts ends at the expected place.
Once your model starts training, it'll approximately take 4.67 minutes to train
a `curie` model, and less for `ada` and `babbage`. Queue will approximately
take half an hour per job ahead of you.
```

在此过程结束时，将会有一个名为*out_openai_completion_prepared.jsonl*的新文件可供使用，并准备好发送到 OpenAI 服务器以运行微调过程。

请注意，如函数消息中所解释的，提示已被修改，末尾添加了字符串`->`，并且所有完成都添加了以`\n`结尾的后缀。### 使用合成数据集微调模型

以下代码上传文件并进行微调。在此示例中，我们将使用`davinci`作为基础模型，生成的模型名称将以`direct_marketing`作为后缀：

```py
ft_file = openai.File.create(
    file=open("out_openai_completion_prepared.jsonl", "rb"), purpose="fine-tune"
)
openai.FineTune.create(
    training_file=ft_file["id"], model="davinci", suffix="direct_marketing"
)
```

这将启动`davinci`模型的更新过程，使用我们的数据。这个微调过程可能需要一些时间，但完成后，您将拥有一个适合您任务的新模型。微调所需的时间主要取决于数据集中的示例数量、示例中的标记数量以及您选择的基础模型。为了让您了解微调所需的时间，我们的示例中不到五分钟就完成了。但是，我们也看到一些情况下微调需要超过 30 分钟：

```py
$ openai api fine_tunes.create -t out_openai_completion_prepared.jsonl \ 
                -m davinci --suffix "direct_marketing"
```

```py

Upload progress: 100%|| 40.8k/40.8k [00:00<00:00, 65.5Mit/s]
Uploaded file from out_openai_completion_prepared.jsonl: file-z5mGg(...)
Created fine-tune: ft-mMsm(...)
Streaming events until fine-tuning is complete...
(Ctrl-C will interrupt the stream, but not cancel the fine-tune)
[] Created fine-tune: ft-mMsm(...)
[] Fine-tune costs $0.84
[] Fine-tune enqueued. Queue number: 0
[] Fine-tune started
[] Completed epoch 1/4
[] Completed epoch 2/4
[] Completed epoch 3/4
[] Completed epoch 4/4
```

###### 警告

正如终端中的消息所解释的那样，您可以通过在命令行中键入 Ctrl+C 来断开与 OpenAI 服务器的连接，但这不会中断微调过程。

要重新连接到服务器并获取正在运行的微调作业的状态，可以使用以下命令`fine_tunes.follow`，其中`fine_tune_id`是微调作业的 ID：

```py
$ openai api fine_tunes.follow -i ***fine_tune_id***
```

当您创建工作时，会得到此 ID。在我们之前的示例中，我们的`fine_tune_id`是`ft-mMsm(...) `。如果您丢失了`fine_tune_id`，可以通过以下方式显示所有模型：

```py
$ openai api fine_tunes.list
```

要立即取消微调作业，请使用此命令：

```py
$ openai api fine_tunes.cancel -i ***fine_tune_id***
```

要删除微调作业，请使用此命令：

```py
$ openai api fine_tunes.delete -i ***fine_tune_id***
```

### 使用微调模型进行文本完成

构建新模型后，可以通过不同的方式访问它以进行新的完成。测试它的最简单方法可能是通过游乐场。要在此工具中访问您的模型，可以在游乐场界面右侧的下拉菜单中搜索它们（请参见图 4-4）。所有您微调的模型都在此列表的底部。选择模型后，可以使用它进行预测。

![](img/dev-app-gpt4-cgpt-dagc_0404.png)

###### 图 4-4。在游乐场中使用微调模型

我们在以下示例中使用了微调的 LLM，输入提示为`Hotel, New York, small ->`。没有进一步的说明，模型自动生成了一则广告，以出售纽约的小型酒店的电子商务支付服务。

我们已经使用了一个仅包含 162 个示例的小数据集获得了出色的结果。对于微调任务，通常建议有几百个实例，最好是几千个。此外，我们的训练集是通过合成生成的，理想情况下应该由营销专家编写。

要将其与 OpenAI API 一起使用，我们按照以前的方式进行，使用`openai.Completion.​cre⁠ate()`，只是需要使用我们的新模型的名称作为输入参数。不要忘记以`->`结束所有提示，并将`\n`设置为停用词：

```py
openai.Completion.create(
  model="davinci:ft-book:direct-marketing-2023-05-01-15-20-35",
  prompt="Hotel, New York, small ->",
  max_tokens=100,
  temperature=0,
  stop="\n"
)
```

我们得到了以下答案：

```py
<OpenAIObject text_completion id=cmpl-7BTkrdo(...) at 0x7f2(4ca5c220> JSON: {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": " \"Upgrade your hotel's payment system with our new e-commerce \ 
service, designed for small businesses.
    }
  ],
  "created": 1682970309,
  "id": "cmpl-7BTkrdo(...)",
  "model": "davinci:ft-book:direct-marketing-2023-05-01-15-20-35",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 37,
    "prompt_tokens": 8,
    "total_tokens": 45
  }
}
```

正如我们所展示的，微调可以使 Python 开发人员根据其独特的业务需求定制 LLM，特别是在我们的电子邮件营销示例等动态领域。这是一种定制语言模型的强大方法，可以帮助您更好地为客户服务并推动业务增长。``  ``##微调成本

使用微调模型是昂贵的。首先，您必须为培训付费，一旦模型准备就绪，每次预测都会比使用 OpenAI 提供的基本模型多一点。

价格可能会有所变化，但在撰写本文时，看起来像是表 4-2。

表 4-2。撰写本书时微调模型的定价

| 模型 | 培训 | 用法 |
| --- | --- | --- |
| `ada` | 每 1,000 个标记 0.0004 美元|每 1,000 个标记 0.0016 美元 |
| `babbage` | 每 1,000 个标记 0.0006 美元|每 1,000 个标记 0.0024 美元 |
| `curie` | 每 1,000 个标记 0.0030 美元|每 1,000 个标记 0.0120 美元 |
| `davinci` | 每 1,000 个标记 0.0300 美元|每 1,000 个标记 0.1200 美元 |

作为比较，`gpt-3.5-turbo`模型的价格为每 1,000 个标记 0.002 美元。如前所述，`gpt-3.5-turbo`具有最佳的性价比。

要获取最新价格，请访问[OpenAI 定价页面](https://openai.com/pricing)。``  ``#摘要

本章讨论了解锁 GPT-4 和 ChatGPT 的全部潜力的高级技术，并提供了关键的可操作的要点，以改进使用 LLM 开发应用程序。

开发人员可以通过了解 prompt 工程、零-shot 学习、few-shot 学习和微调来创建更有效和有针对性的应用程序。我们探讨了如何通过考虑上下文、任务和角色来创建有效的提示，从而实现与模型更精确的交互。通过逐步推理，开发人员可以鼓励模型更有效地推理和处理复杂任务。此外，我们讨论了 few-shot 学习提供的灵活性和适应性，突出了其数据高效的特性和快速适应不同任务的能力。

表 4-3 提供了所有这些技术的快速摘要，何时使用它们以及它们的比较。

表 4-3。不同技术的比较

|  | 零-shot 学习 | few-shot 学习 | prompt 工程技巧 | 微调 |
| --- | --- | --- | --- | --- |
| 定义 | 预测没有先前示例的未见任务 | 提示包括输入和期望的输出示例 | 可包括上下文、角色和任务的详细提示，或者“逐步思考”等技巧 | 模型在更小、更具体的数据集上进一步训练；使用的提示很简单 |
| 用例 | 简单任务 | 定义明确但复杂的任务，通常具有特定的输出格式 | 创造性、复杂的任务 | 高度复杂的任务 |
| 数据 | 不需要额外的示例数据 | 需要一些示例 | 数据量取决于 prompt 工程技术 | 需要大型训练数据集 |
| 定价 | 使用：每个令牌（输入+输出）的定价 | 使用：每个令牌（输入+输出）的定价；可能导致长提示 | 使用：每个令牌（输入+输出）的定价，可能导致长提示 | 训练：使用：每个令牌（输入+输出）的定价，与 GPT-3.5 Turbo 相比，fine-tuned `davinci`大约贵 80 倍。这意味着如果其他技术导致提示长度增加 80 倍，经济上更倾向于进行微调。 |
| 结论 | 默认使用 | 如果零-shot 学习不起作用，因为输出需要特定的话，使用 few-shot 学习。 | 如果零-shot 学习不起作用，因为任务太复杂，尝试 prompt 工程。 | 如果您有一个非常具体和大型的数据集，其他解决方案效果不够好，这应该作为最后的手段。 |

为了确保构建 LLM 应用程序的成功，开发人员应该尝试其他技术，并评估模型的响应是否准确和相关。此外，开发人员应该意识到 LLM 的计算限制，并相应地调整他们的提示以获得更好的结果。通过整合这些先进技术并不断完善他们的方法，开发人员可以创建功能强大和创新的应用程序，释放 GPT-4 和 ChatGPT 的真正潜力。

在下一章中，您将发现将 LLM 功能集成到您的应用程序中的另外两种方法：插件和 LangChain 框架。这些工具使开发人员能够创建创新的应用程序，访问最新信息，并简化集成 LLM 的应用程序的开发。我们还将提供关于 LLM 未来及其对应用程序开发的影响的见解。


