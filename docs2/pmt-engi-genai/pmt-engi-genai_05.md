# 第五章\. 使用 FAISS 和 Pinecone 的向量数据库

本章介绍了嵌入和向量数据库的概念，讨论了它们如何用于在提示中提供相关上下文。

*向量数据库* 是一种最常用于以支持基于相似性或语义意义查询的方式存储文本数据的工具。这项技术通过引用模型未训练过的数据来减少幻觉（即 AI 模型编造内容），显著提高了 LLM（大型语言模型）响应的准确性和质量。向量数据库的应用案例还包括阅读文档、推荐类似产品或记住过去的对话。

*向量* 是代表文本（或图像）的数字列表，你可以将其视为一个位置的坐标。使用 OpenAI 的 text-embedding-ada-002 模型对单词 *mouse* 的向量是一个包含 1,536 个数字的列表，每个数字代表嵌入模型在训练过程中学习到的特征值：

```py
[-0.011904156766831875,
 -0.0323905423283577,
 0.001950666424818337,
...]
```

当这些模型进行训练时，训练数据中一起出现的文本在数值上会被推得更近，而无关的文本会被推得更远。想象一下，我们训练了一个只有两个参数 `Cartoon` 和 `Hygiene` 的简单模型，这个模型必须描述整个世界，但只能用这两个变量来描述。从单词 *mouse* 开始，增加 `Cartoon` 参数的值，我们会走向最著名的卡通老鼠 `mickey mouse`，如图 图 5-1 所示。减少 `Hygiene` 参数的值会带我们走向 `rat`，因为老鼠是和老鼠相似啮齿动物，但与瘟疫和疾病（即不卫生）有关。

![pega 0501](img/pega_0501.png)

###### 图 5-1\. 2 维向量距离

图表上的每个位置都可以通过 x 轴和 y 轴上的两个数字找到，这些数字代表模型 `Cartoon` 和 `Hygiene` 的特征。实际上，向量可以拥有数千个参数，因为更多的参数允许模型捕捉更广泛的相似性和差异性。卫生并不是老鼠和老鼠之间唯一的区别，米老鼠也不只是一个卡通老鼠。这些特征是从数据中学习到的，使得它们对人类来说难以解释，我们需要一个拥有数千个轴的图表来显示 *潜在空间*（由模型参数形成的抽象的多维空间）中的位置。通常，没有人类可以理解的关于特征含义的解释。然而，我们可以创建一个简化的二维向量距离投影，如图 图 5-2 所示。

进行向量搜索时，你首先获取你想要查找的内容的向量（或位置），然后在数据库中找到最近的 `k` 个记录。在这种情况下，单词 *mouse* 与 `mickey mouse`、`cheese` 和 `trap` 最接近，其中 `k=3`（返回三个最近的记录）。如果 `k=3`，则排除单词 *rat*，但如果 `k=4`，则将其包括在内，因为它是最接近的下一个向量。在这个例子中，单词 *airplane* 很远，因为它在训练数据中很少与单词 *mouse* 相关。单词 *ship* 仍然与其他交通方式一起定位，但它比 `mouse` 和 `rat` 更接近，因为根据训练数据，它们经常在船上。

![pega 0502](img/pega_0502.png)

###### 图 5-2\. 多维向量距离

向量数据库以文本记录的向量表示作为键存储文本记录。这与其他类型的数据库不同，在其他类型的数据库中，你可能根据 ID、关系或文本中包含的字符串来查找记录。例如，如果你根据 图 5-2 中的文本查询关系型数据库以找到包含 `mouse` 的记录，你会返回记录 `mickey mouse` 但没有其他记录，因为没有其他记录包含该确切短语。使用向量搜索，你也可以返回记录 `cheese` 和 `trap`，因为它们密切相关，即使它们不是查询的精确匹配。

基于相似性进行查询的能力非常实用，向量搜索为许多 AI 功能提供了动力。例如：

文档阅读

找到相关的文本部分进行阅读，以便提供更准确的答案。

推荐系统

发现类似的产品或项目，以便向用户推荐。

长期记忆

查找相关的对话历史片段，以便聊天机器人记住过去的交互。

AI 模型能够处理这些任务，只要你的文档、产品列表或对话记忆适合你使用的模型的代币限制。然而，在规模上，你很快就会遇到代币限制和由于每个提示中传递过多代币而产生的额外成本。OpenAI 的 `gpt-4-1106-preview` 在 2023 年 11 月发布，拥有巨大的 128,000 个代词上下文窗口，但每个代词的成本是 `gpt-3.5-turbo` 的 10 倍，后者有 88% 更少的代词，并且一年前发布。更有效的方法是在运行时仅查找最相关的记录以传递到提示中，以便提供最相关的上下文来形成回答。这种做法通常被称为 RAG。

# 检索增强生成 (RAG)

向量数据库是 RAG 的关键组成部分，通常涉及通过查询的相似性进行搜索，检索最相关的文档，并将它们作为上下文插入到提示中。这让你保持在当前上下文窗口内，同时避免通过在上下文中插入无关文本文档而浪费代币。

检索也可以使用传统的数据库搜索或网络浏览来完成，在许多情况下，使用语义相似性的向量搜索并不是必要的。RAG 通常用于解决开放场景中的幻觉问题，例如用户与一个在询问不在其训练数据中的内容时倾向于编造东西的聊天机器人交谈。向量搜索可以将与用户查询语义相似的文档插入到提示中，大大降低聊天机器人产生幻觉的可能性。

例如，如果您的作者 Mike 告诉聊天机器人“我的名字是 Mike”，然后在接下来的三条消息中询问“我的名字是什么？”它可以轻松地回忆起正确的答案。包含 Mike 名字的消息仍然在聊天上下文中。然而，如果这些消息是在 3,000 条消息之前，那么这些消息的文本可能太大，无法适应上下文窗口。没有这个重要的上下文，它可能会产生一个名字或者因为信息不足而拒绝回答。关键词搜索可能会有所帮助，但可能会返回太多不相关的文档，或者无法回忆起过去捕获信息时的正确上下文。Mike 可能多次在不同格式和不同原因下提到“名字”这个词。通过将问题传递给向量数据库，它可以返回与用户询问最相似的聊天中的前三条消息：

```py
## Context
Most relevant previous user messages:
1\. "My name is Mike".
2\. "My dog's name is Hercules".
3\. "My coworker's name is James".

## Instructions
Please answer the user message using the context above.
User message: What is my name?
AI message:
```

对于大多数模型来说，将所有 3,000 条过去的消息传递到提示中是不可能的，对于传统搜索，AI 模型必须制定正确的搜索查询，这可能是不可靠的。使用 RAG 模式，您会将当前用户的消息传递给向量搜索函数，并返回最相关的三条记录作为上下文，然后聊天机器人可以使用这些上下文来正确回答。

# 指明方向

向量搜索允许您将最相关的知识动态地插入到提示中，而不是将静态知识插入到提示中。

这是使用 RAG 的生产应用流程：

1.  将文档分成文本块。

1.  在向量数据库中索引文本块。

1.  通过向量搜索相似记录。

1.  将记录作为上下文插入到提示中。

在这种情况下，文档将是所有 3,000 条过去的用户消息，作为聊天机器人的记忆，但它也可以是上传到聊天机器人以使其能够阅读的 PDF 文档的部分，或者是一份所有相关产品的列表，以使聊天机器人能够做出推荐。我们向量搜索找到最相似文本的能力完全取决于用于生成向量的 AI 模型，当处理语义或上下文信息时，这些向量被称为“嵌入”。

# 引入嵌入

术语*嵌入*通常指的是从预训练的 AI 模型返回的文本的向量表示。在撰写本文时，生成嵌入的标准模型是 OpenAI 的 text-embedding-ada-002，尽管嵌入模型在生成式 AI 出现之前就已经存在。

虽然将向量空间可视化为二维图表很有帮助，如图 5-2 所示，但现实中 text-embedding-ada-002 返回的嵌入是在 1,536 个维度上，这在图形上很难表示。更多的维度允许模型捕捉更深层次的语义意义和关系。例如，一个二维空间可能能够将猫和狗分开，而一个 300 维的空间可以捕捉关于品种、大小、颜色和其他复杂细节的信息。以下代码展示了如何从 OpenAI API 检索嵌入。以下示例的代码包含在此书的[GitHub 仓库](https://oreil.ly/6RzTy)中。

输入：

```py
from openai import OpenAI
client = OpenAI()

# Function to get the vector embedding for a given text
def get_vector_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = [r.embedding for r in response.data]
    return embeddings[0]

get_vector_embeddings("Your text string goes here")
```

输出：

```py
[
-0.006929283495992422,
-0.005336422007530928,
...
-4.547132266452536e-05,
-0.024047505110502243
]
```

此代码使用 OpenAI API 创建给定输入文本的嵌入，使用特定的嵌入模型：

1.  `from openai import OpenAI` 导入 OpenAI 库，`client = OpenAI()` 设置客户端。它从环境变量`OPENAI_API_KEY`中检索您的 OpenAI API 密钥，以便将嵌入的成本记到您的账户上。您需要在您的环境中设置此变量（通常在*.env*文件中），这可以通过创建账户并访问[*https://oreil.ly/apikeys*](https://oreil.ly/apikeys)来获取。

1.  `response = client.embeddings.create(...)`：这一行调用 OpenAI 库中`Embedding`类的`create`方法。该方法接受两个参数：

    +   `input`：这是您提供要生成嵌入的文本字符串的地方。

    +   `model`：这指定了您想要使用的嵌入模型。在这种情况下，它是`text-embedding-ada-002`，这是 OpenAI API 中的一个模型。

1.  `embeddings = [r.embedding for r in response.data]`：在 API 调用之后，`response`对象包含以 JSON 格式生成的嵌入。这一行通过遍历`response.data`中的嵌入列表，从响应中提取实际的数值嵌入。

执行此代码后，`embeddings`变量将包含输入文本的数值表示（嵌入），然后可以用于各种 NLP 任务或机器学习模型。检索或生成嵌入的过程有时被称为*文档加载*。

在此上下文中，术语 *loading* 指的是从模型中计算或检索文本的数值（向量）表示并将其存储在变量中以供以后使用的行为。这与 *chunking* 的概念不同，通常指的是将文本分解成更小、更易于管理的片段或块，以促进处理。这两种技术通常一起使用，因为通常将大型文档拆分成页面或段落以促进更准确的匹配，并且只将最相关的标记传递到提示中。

从 OpenAI 获取嵌入存在成本，但根据写作时的价格，每 1,000 个标记的费用相对较低，为 $0.0004。例如，包含大约 800,000 个单词或约 4,000,000 个标记的《圣经》钦定版，检索整个文档的所有嵌入将花费大约 $1.60。

为 OpenAI 的嵌入付费并不是你的唯一选择。还有你可以使用的开源模型，例如 Hugging Face 提供的 [Sentence Transformers 库](https://oreil.ly/8OV3c)，它具有 384 维度。

输入：

```py
import requests
import os

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = os.getenv("HF_TOKEN")

api_url = "https://api-inference.huggingface.co/"
api_url += f"pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts):
    response = requests.post(api_url, headers=headers,
    json={"inputs": texts,
    "options":{"wait_for_model":True}})
    return response.json()

texts = ["mickey mouse",
        "cheese",
        "trap",
        "rat",
        "ratatouille"
        "bus",
        "airplane",
        "ship"]

output = query(texts)
output
```

输出：

```py
[[-0.03875632584095001, 0.04480459913611412,
0.016051070764660835, -0.01789097487926483,
-0.03518553078174591, -0.013002964667975903,
0.14877274632453918, 0.048807501792907715,
0.011848390102386475, -0.044042471796274185,
...
-0.026688814163208008, -0.0359361357986927,
-0.03237859532237053, 0.008156519383192062,
-0.10299170762300491, 0.0790356695652008,
-0.008071334101259708, 0.11919838190078735,
0.0005506130401045084, -0.03497892618179321]]
```

此代码使用 Hugging Face API 通过预训练模型为文本输入列表获取嵌入。这里使用的模型是 `sentence-transformers/all-MiniLM-L6-v2`，这是 BERT 的一个较小版本，BERT 是 Google 在 2017 年（基于 Transformer 模型）引入的开源 NLP 模型，它针对句子级任务进行了优化。以下是它的工作步骤：

1.  `model_id` 被分配了预训练模型的标识符，`sentence-transformers/all-MiniLM-L6-v2`。

1.  `hf_token = os.getenv("HF_TOKEN")` 从你的环境中检索 Hugging Face API 令牌。你需要使用自己的令牌设置此环境，该令牌可以通过创建账户并访问[*https://hf.co/settings/tokens*](https://hf.co/settings/tokens)来获取。

1.  导入了 `requests` 库以向 API 发送 HTTP 请求。

1.  `api_url` 被分配为包含模型 ID 的 Hugging Face API URL。

1.  `headers` 是一个包含你的 Hugging Face API 令牌的授权头的字典。

1.  `query()` 函数被定义，它接受一个文本输入列表，并向 Hugging Face API 发送带有适当头和包含输入以及等待模型可用的选项的 JSON 有效负载的 `POST` 请求。然后该函数返回 API 的 JSON 响应。

1.  `texts` 是来自你数据库的字符串列表。

1.  `output` 被分配为调用 `query()` 函数并使用 `texts` 列表的结果。

1.  `output` 变量将被打印出来，这将显示输入文本的特征嵌入。

当你运行此代码时，脚本将向 Hugging Face API 发送文本，API 将为发送的每条文本字符串返回嵌入。

如果你将相同的文本输入到嵌入模型中，你将每次都得到相同的向量。然而，由于训练的不同，向量通常不可比（或模型的版本）之间。从 OpenAI 获得的嵌入与从 BERT 或 spaCy（一个自然语言处理库）获得的嵌入不同。

与现代转换器模型生成的嵌入相比，主要区别在于向量是上下文相关的而不是静态的，这意味着单词“bank”在“riverbank”和“financial bank”的上下文中会有不同的嵌入。从 OpenAI Ada 002 和 HuggingFace Sentence Transformers 获得的嵌入是密集向量的例子，其中数组中的每个数字几乎总是非零的（即，它们包含语义信息）。还有[稀疏向量](https://oreil.ly/d1cmb)，通常具有大量的维度（例如，100,000+），其中许多维度具有零值。这允许捕捉特定的关键特征（每个特征可以有自己的维度），这对于基于关键字的搜索应用中的性能通常很重要。大多数 AI 应用使用密集向量进行检索，尽管混合搜索（密集和稀疏向量）越来越受欢迎，因为相似性和关键字搜索可以结合使用。

向量的准确性完全依赖于你用来生成嵌入的模型的准确性。底层模型中存在的任何偏差或知识差距也将成为向量搜索的问题。例如，`text-embedding-ada-002`模型目前仅训练到 2020 年 8 月，因此对该截止日期之后形成的新词或新的文化关联一无所知。这可能会对需要更多近期上下文或训练数据中不可用的利基领域知识的用例造成问题，这可能需要训练自定义模型。

在某些情况下，训练自己的嵌入模型可能是有意义的。例如，如果你使用的文本具有特定领域的词汇，其中某些词的含义与通常接受的含义不同，你可能就会这样做。一个例子可能是追踪社交媒体上如 Q-Anon 等有毒群体使用的语言，这些群体会不断演变他们在帖子中使用的语言以绕过审查措施。

使用工具如 word2vec 训练自己的嵌入模型是可行的，word2vec 是一种在向量空间中表示单词的方法，使你能够捕捉单词的语义含义。可以使用更先进的模型，如 GloVe（用于单词表示的全局向量），这是 spaCy 用于其嵌入的模型，该模型在 Common Crawl 数据集上训练，Common Crawl 是互联网的开源快照。Gensim 库提供了一个简单的流程，使用[开源算法](https://oreil.ly/RmXVR) word2vec 来训练你自己的自定义嵌入。

输入：

```py
from gensim.models import Word2Vec

# Sample data: list of sentences, where each sentence is
# a list of words.
# In a real-world scenario, you'd load and preprocess your
# own corpus.
sentences = [
    ["the", "cake", "is", "a", "lie"],
    ["if", "you", "hear", "a", "turret", "sing", "you're",
    "probably", "too", "close"],
    ["why", "search", "for", "the", "end", "of", "a",
    "rainbow", "when", "the", "cake", "is", "a", "lie?"],
    # ...
    ["there's", "no", "cake", "in", "space,", "just", "ask",
    "wheatley"],
    ["completing", "tests", "for", "cake", "is", "the",
    "sweetest", "lie"],
    ["I", "swapped", "the", "cake", "recipe", "with", "a",
    "neurotoxin", "formula,", "hope", "that's", "fine"],
] + [
    ["the", "cake", "is", "a", "lie"],
    ["the", "cake", "is", "definitely", "a", "lie"],
    ["everyone", "knows", "that", "cake", "equals", "lie"],
    # ...
] * 10  # repeat several times to emphasize

# Train the word2vec model
model =  Word2Vec(sentences, vector_size=100, window=5,
min_count=1, workers=4, seed=36)

# Save the model
model.save("custom_word2vec_model.model")

# To load the model later
# loaded_model = word2vec.load(
# "custom_word2vec_model.model")

# Get vector for a word
vector = model.wv['cake']

# Find most similar words
similar_words = model.wv.most_similar("cake", topn=5)
print("Top five most similar words to 'cake': ", similar_words)

# Directly query the similarity between "cake" and "lie"
cake_lie_similarity = model.wv.similarity("cake", "lie")
print("Similarity between 'cake' and 'lie': ",
cake_lie_similarity)
```

输出：

```py
Top 5 most similar words to 'cake':  [('lie',
0.23420444130897522), ('test', 0.23205122351646423),
('tests', 0.17178669571876526), ('GLaDOS',
0.1536172330379486), ('got', 0.14605288207530975)]
Similarity between 'cake' and 'lie':  0.23420444
```

这段代码使用 Gensim 库创建了一个 word2vec 模型，然后使用该模型确定与给定单词相似的单词。让我们将其分解：

1.  变量 `sentences` 包含一个句子列表，其中每个句子都是一个单词列表。这是 Word2Vec 模型将要训练的数据。在实际应用中，你通常会加载一个大型文本语料库并对其进行预处理，以获得这样的标记句子列表。

1.  创建了一个 `word2vec` 类的实例来表示模型。在初始化这个实例时，提供了几个参数：

    +   `sentences`：这是训练数据。

    +   `vector_size=100`：这定义了单词向量的尺寸。因此，每个单词都将表示为一个 100 维的向量。

    +   `window=5`：这表示句子中当前单词和预测单词之间的最大距离。

    +   `min_count=1`：这确保了即使数据集中只出现一次的单词也会为其创建向量。

    +   `workers=4`：在训练期间使用的 CPU 核心数。它可以在多核机器上加速训练。

    +   `seed=36`：这是为了可重复性而设置的，以确保训练中的随机过程每次都能产生相同的结果（在多个工作者的情况下不保证）。

1.  训练完成后，使用 `save` 方法将模型保存到名为 `custom_word2vec_model.model` 的文件中。这允许你在以后重用训练好的模型，而无需再次进行训练。

1.  有一个注释掉的行显示了如何从保存的文件中加载模型。这在你想在不同的脚本或会话中加载预训练模型时很有用。

1.  变量 `vector` 被分配了单词 *cake* 的向量表示。这个向量可以用于各种目的，如相似度计算、算术运算等。

1.  使用 `most_similar` 方法来查找与提供的向量（在这种情况下，是 *cake* 的向量）最相似的单词。该方法返回最相似的五个单词（`topn=5`）。

1.  `similarity` 方法查询了 *cake* 和 *lie* 方向上的相似度，显示了一个小的正值。

数据集很小且高度重复，这可能无法提供多样化的上下文来正确学习单词之间的关系。通常，word2vec 从更大、更多样化的语料库中受益，并且通常直到你拥有数千万个单词时才会得到良好的结果。在示例中，我们设置了一个种子值以挑选出一个实例，其中 *lie* 出现在前五个结果中，但如果你移除该种子，你会发现它很少能够成功地发现这种关联。

对于较小的文档大小，建议使用更简单的技术 *TF-IDF*（词频-逆文档频率），这是一种用于评估一个词在文档中相对于一组文档的重要性程度的统计度量。TF-IDF 值与词在文档中出现的次数成正比，但被词在更广泛语料库中的频率所抵消，这有助于调整某些词比其他词更普遍的事实。

要使用 TF-IDF 计算 *cake* 和 *lie* 之间的相似度，你可以使用开源 [科学库](https://oreil.ly/gHb3F) scikit-learn 并计算 *余弦相似度*（两个向量之间距离的度量）。在句子中经常共现的词将具有高余弦相似度（接近 1），而出现频率低的词将显示低值（或 0，如果根本不共现）。这种方法对即使是像我们的玩具示例那样的小文档也具有鲁棒性。

输入：

```py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert sentences to a list of strings for TfidfVectorizer
document_list = [' '.join(s) for s in sentences]

# Compute TF-IDF representation
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(document_list)

# Extract the position of the words "cake" and "lie" in
# the feature matrix
cake_idx = vectorizer.vocabulary_['cake']
lie_idx = vectorizer.vocabulary_['lie']

# Extract and reshape the vector for 'cake'
cakevec = tfidf_matrix[:, cake_idx].toarray().reshape(1, -1)

# Compute the cosine similarities
similar_words = cosine_similarity(cakevec, tfidf_matrix.T).flatten()

# Get the indices of the top 6 most similar words
# (including 'cake')
top_indices = np.argsort(similar_words)[-6:-1][::-1]

# Retrieve and print the top 5 most similar words to
# 'cake' (excluding 'cake' itself)
names = []
for idx in top_indices:
    names.append(vectorizer.get_feature_names_out()[idx])
print("Top five most similar words to 'cake': ", names)

# Compute cosine similarity between "cake" and "lie"
similarity = cosine_similarity(np.asarray(tfidf_matrix[:,
    cake_idx].todense()), np.asarray(tfidf_matrix[:, lie_idx].todense()))
# The result will be a matrix; we can take the average or
# max similarity value
avg_similarity = similarity.mean()
print("Similarity between 'cake' and 'lie'", avg_similarity)

# Show the similarity between "cake" and "elephant"
elephant_idx = vectorizer.vocabulary_['sing']
similarity = cosine_similarity(np.asarray(tfidf_matrix[:,
    cake_idx].todense()), np.asarray(tfidf_matrix[:,
    elephant_idx].todense()))
avg_similarity = similarity.mean()
print("Similarity between 'cake' and 'sing'",
    avg_similarity)
```

输出：

```py
Top 5 most similar words to 'cake':  ['lie', 'the', 'is',
'you', 'definitely']
Similarity between 'cake' and 'lie' 0.8926458157227388
Similarity between 'cake' and 'sing' 0.010626735901461177
```

让我们逐步分解这段代码：

1.  `sentences` 变量从上一个示例中重用。代码使用列表推导将这些单词列表转换为完整的句子（字符串），从而得到 `document_list`。

1.  从上一个示例中重用 `sentences` 变量。代码使用列表推导将这些单词列表转换为完整的句子（字符串），从而得到 `document_list`。

1.  代码使用向量化器的 `vocabulary_` 属性提取词 *cake* 和 *lie* 在特征矩阵中的位置（或索引）。

1.  从矩阵中提取与词 *cake* 对应的 TF-IDF 向量并将其重塑。

1.  计算 TF-IDF 矩阵中所有其他向量与向量 *cake* 之间的余弦相似度。这产生了一个相似度分数列表。

    +   确定最相似的前六个词（包括 *cake*）的索引。

    +   使用这些索引，检索并打印与 *cake* 最相似的前五个词（不包括 *cake*）。

1.  计算词 *cake* 和 *lie* 的 TF-IDF 向量的余弦相似度。由于结果是矩阵，代码计算矩阵中所有值的平均相似度值，然后打印平均值。

1.  现在我们计算 *cake* 和 *sing* 之间的相似度。计算并打印平均相似度值以显示这两个词不常见共现（接近零）。

除了使用的嵌入模型外，嵌入内容的策略也很重要，因为存在语境和相似度之间的权衡。如果你嵌入一个大块文本，比如整本书，你得到的向量将是构成全文的标记位置的均值。随着块大小的增加，它将回归到均值，接近所有向量的均值，并且不再包含很多语义信息。

较小的文本块在向量空间中的位置将更加具体，因此当需要接近的相似性时可能更有用。例如，从小说中隔离较小的文本部分可能更好地将故事中的喜剧时刻和悲剧时刻分开，而嵌入整个页面或章节可能会将两者混合在一起。然而，如果文本块太小，也可能导致它们在句子或段落的中间被切断而失去意义。与向量数据库一起工作的很大一部分艺术在于你如何加载文档并将其分割成块。

# 文档加载

人工智能的一个常见用途是能够根据与用户查询文本的相似性在文档中进行搜索。例如，你可能有一系列代表你的员工手册的 PDF 文件，你希望从这些 PDF 文件中返回与员工问题相关的正确文本片段。你将文档加载到向量数据库中的方式将由你的文档结构、你希望从每个查询中返回多少示例以及你可以在每个提示中承担的标记数量决定。

例如，`gpt-4-0613`有[8,192 个标记限制](https://oreil.ly/wbx1f)，这需要在提示模板、插入到提示中的示例以及模型提供的响应之间共享。为提示和响应预留大约 2,000 个单词或大约 3,000 个标记，你可以将五个最相似的、每个 1,000 个标记的文本块作为上下文拉入提示中。然而，如果你天真地将文档分割成 1,000 个标记的块，你将遇到问题。每次分割的任意位置可能位于段落或句子的中间，这样你可能会丢失所传达的意义。LangChain 有一系列[文本分割器](https://oreil.ly/qsG7J)，包括常用的递归字符文本分割器。它试图在行中断和空格之间分割，直到块足够小。这尽可能地将所有段落（然后是句子，然后是单词）保持在一起，以保留文本结构中固有的语义分组。

输入：

```py
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, # 100 tokens
    chunk_overlap=20, # 20 tokens of overlap
    )

text = """
Welcome to the "Unicorn Enterprises: Where Magic Happens"
Employee Handbook! We're thrilled to have you join our team
of dreamers, doers, and unicorn enthusiasts. At Unicorn
Enterprises, we believe that work should be as enchanting as
it is productive. This handbook is your ticket to the
magical world of our company, where we'll outline the
principles, policies, and practices that guide us on this
extraordinary journey. So, fasten your seatbelts and get
ready to embark on an adventure like no other!

...

As we conclude this handbook, remember that at Unicorn
Enterprises, the pursuit of excellence is a never-ending
quest. Our company's success depends on your passion,
creativity, and commitment to making the impossible
possible. We encourage you to always embrace the magic
within and outside of work, and to share your ideas and
innovations to keep our enchanted journey going. Thank you
for being a part of our mystical family, and together, we'll
continue to create a world where magic and business thrive
hand in hand!
"""

chunks = text_splitter.split_text(text=text)
print(chunks[0:3])
```

输出：

```py
['Welcome to the "Unicorn Enterprises: Where Magic Happens"
Employee Handbook! We\'re thrilled to have you join our team
of dreamers, doers, and unicorn enthusiasts.',
"We're thrilled to have you join our team of dreamers,
doers, and unicorn enthusiasts. At Unicorn Enterprises, we
believe that work should be as enchanting as it is
productive.",
 ...
"Our company's success depends on your passion, creativity,
and commitment to making the impossible possible. We
encourage you to always embrace the magic within and outside
of work, and to share your ideas and innovations to keep our
enchanted journey going.",
"We encourage you to always embrace the magic within and
outside of work, and to share your ideas and innovations to
keep our enchanted journey going. Thank you for being a part
of our mystical family, and together, we'll continue to
create a world where magic and business thrive hand in
hand!"]
```

下面是这段代码逐步工作的方式：

1.  *创建文本分割器实例*：使用`from_tiktoken_encoder`方法创建`RecursiveCharacterTextSplitter`的实例。此方法专门设计用于根据标记计数来分割文本。

    `chunk_size`参数设置为 100，确保每个文本块将包含大约 100 个标记。这是一种控制每个文本段大小的方法。

    `chunk_overlap`参数设置为 20，指定连续块之间将有 20 个标记的重叠。这种重叠确保了上下文在块之间不会丢失，这对于准确理解和处理文本至关重要。

1.  *准备文本：* 变量 `text` 包含一个多段落字符串，表示要分割成块的文本内容。

1.  *分割文本：* 使用 `text_splitter` 实例的 `split_text` 方法根据先前定义的 `chunk_size` 和 `chunk_overlap` 来分割文本成块。此方法处理文本并返回一个文本块列表。

1.  *输出块：* 代码打印出分割文本的前三个块，以展示文本是如何被分割的。这种输出有助于验证文本是否按照预期的分割方式，并遵循指定的块大小和重叠。

# 指定格式

提供的文本块与提示的相关性将很大程度上取决于你的分割策略。没有重叠的短文本块可能不包含正确的答案，而重叠过多的长文本块可能会返回太多不相关的结果，并混淆 LLM 或花费你太多的令牌。

# 使用 FAISS 进行内存检索

现在你已经将文档处理成块，你需要将它们存储在向量数据库中。通常的做法是将向量存储在数据库中，这样你就不需要重新计算它们，因为这样做通常会有一些成本和延迟。如果你不更改你的嵌入模型，向量就不会改变，所以一旦存储后通常不需要更新它们。你可以使用一个名为 FAISS 的开源库来存储和查询你的向量，这是一个由 [Facebook AI](https://oreil.ly/gIcTI) 开发的库，它提供了密集向量的高效相似性搜索和聚类。首先在终端中使用 `pip install faiss-cpu` 安装 FAISS。本例的代码包含在本书的 [GitHub 仓库](https://oreil.ly/4wR7o) 中。

输入：

```py
import numpy as np
import faiss

#  The get_vector_embeddings function is defined in a preceding example
emb = [get_vector_embeddings(chunk) for chunk in chunks]
vectors = np.array(emb)

# Create a FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Function to perform a vector search
def vector_search(query_text, k=1):
    query_vector = get_vector_embeddings(query_text)
    distances, indices = index.search(
        np.array([query_vector]), k)
    return [(chunks[i], float(dist)) for dist,
        i in zip(distances[0], indices[0])]

# Example search
user_query = "do we get free unicorn rides?"
search_results = vector_search(user_query)
print(f"Search results for {user_query}:", search_results)
```

输出：

```py
Search results for do we get free unicorn rides?: [("You'll
enjoy a treasure chest of perks, including unlimited unicorn
rides, a bottomless cauldron of coffee and potions, and
access to our company library filled with spellbinding
books. We also offer competitive health and dental plans,
ensuring your physical well-being is as robust as your
magical spirit.\n\n**5: Continuous Learning and
Growth**\n\nAt Unicorn Enterprises, we believe in continuous
learning and growth.", 0.3289167582988739)]
```

下面是对前面代码的解释：

1.  使用 `import faiss` 导入 Facebook AI 相似性搜索（FAISS）库。

1.  `vectors = np.array([get_vector_embeddings(chunk) for chunk in chunks])` 将 `get_vector_embeddings` 函数应用于 `chunks` 中的每个元素，这会返回每个元素的向量表示（嵌入）。然后，这些向量被用来创建一个 numpy 数组，该数组存储在变量 `vectors` 中。

1.  `index = faiss.IndexFlatL2(vectors.shape[1])` 这行代码创建了一个用于高效相似性搜索的 FAISS 索引。参数 `vectors.shape[1]` 是将要添加到索引中的向量的维度。这种索引（`IndexFlatL2`）执行穷举 L2 距离搜索，通过测量它们之间的直线距离来寻找集合中与特定项目最接近的项目，逐个检查集合中的每个项目。

1.  然后使用 `index.add(vectors)` 将向量数组添加到创建的 FAISS 索引中。

1.  `def vector_search(query_text, k=1)` 定义了一个名为 `vector_search` 的新函数，该函数接受两个参数：`query_text` 和 `k`（默认值为 1）。该函数将检索 `query_text` 的嵌入，然后使用该嵌入在索引中搜索 `k` 个最近的向量。

1.  在 `vector_search` 函数内部，`query_vector = get_vector_embeddings(query_text)` 使用 `get_vector_embeddings` 函数生成查询文本的向量嵌入。

1.  `distances, indices = index.search(np.array([query_vector]), k)` 这一行在 FAISS 索引中执行搜索。它寻找与 `query_vector` 最接近的 `k` 个向量。该方法返回两个数组：`distances`（到查询向量的平方 L2 距离）和 `indices`（索引）。

1.  `return [(chunks[i], float(dist)) for dist, i in zip(distances[0], indices[0])]` 返回一个元组列表。每个元组包含一个块（使用在搜索中找到的索引检索到的）和查询向量对应的距离。请注意，在返回之前，距离被转换为浮点数。

1.  最后，你执行对包含用户查询的字符串的向量搜索：`search_results = vector_search(user_query)`。结果（最近的块及其距离）存储在变量 `search_results` 中。

一旦向量搜索完成，可以将结果注入提示中，以提供有用的上下文。同时，也很重要设置系统消息，以便模型专注于根据提供的上下文回答，而不是随意作出回答。这里展示的 RAG 技术在 AI 中被广泛使用，以帮助防止幻觉。

输入：

```py
# Function to perform a vector search and then ask # GPT-3.5-turbo a question
def search_and_chat(user_query, k=1):
  # Perform the vector search
  search_results = vector_search(user_query, k)
  print(f"Search results: {search_results}\n\n")

  prompt_with_context = f"""Context:{search_results}\
 Answer the question: {user_query}"""

  # Create a list of messages for the chat
  messages = [
      {"role": "system", "content": """Please answer the
 questions provided by the user. Use only the context
 provided to you to respond to the user, if you don't
 know the answer say \"I don't know\"."""},
      {"role": "user", "content": prompt_with_context},
  ]

  # Get the model's response
  response = client.chat.completions.create(
    model="gpt-3.5-turbo", messages=messages)

  # Print the assistant's reply
  print(f"""Response:
  {response.choices[0].message.content}""")

# Example search and chat
search_and_chat("What is Unicorn Enterprises' mission?")
```

输出：

```py
Search results: [("""As we conclude this handbook, remember that at
Unicorn Enterprises, the pursuit of excellence is a never-ending
quest. Our company's success depends on your passion,
creativity, and commitment to making the impossible
possible. We encourage you to always embrace the magic
within and outside of work, and to share your ideas and
innovations to keep our enchanted journey going. Thank you",
0.26446571946144104)]

Response:
Unicorn Enterprises' mission is to pursue excellence in their
work by encouraging their employees to embrace the magic within
and outside of work, share their ideas and innovations, and make
the impossible possible.
```

这里是对函数所做步骤的逐步解释：

1.  使用名为 `vector_search` 的函数，程序使用 `user_query` 作为搜索字符串和 `k` 作为要返回的搜索结果数量执行向量搜索。结果存储在 `search_results` 中。

1.  然后将搜索结果打印到控制台。

1.  通过连接 `search_results` 和 `user_query` 创建了一个 `prompt_with_context`。目的是为模型提供搜索结果中的上下文和一个要回答的问题。

1.  创建了一个消息列表。第一条消息是一个系统消息，指示模型仅使用给定上下文来回答用户提出的问题。如果模型不知道答案，建议回答“我不知道”。第二条消息是一个包含 `prompt_with_context` 的用户消息。

1.  `openai.ChatCompletion.create()` 函数用于获取模型的响应。它提供了模型名称（`gpt-3.5-turbo`）和消息列表。

1.  在代码的末尾，使用问题作为 `user_query` 调用了 `search_and_chat()` 函数。

# 提供示例

没有测试写作风格，很难猜测哪种提示策略会获胜。现在你可以有信心这是正确的方法。

尽管我们的代码现在从头到尾都能正常工作，但每次查询都收集嵌入并创建向量数据库是没有意义的。即使你使用开源模型进行嵌入，也会在计算和延迟方面产生成本。你可以使用 `faiss.write_index` 函数将 FAISS 索引保存到文件：

```py
# Save the index to a file
faiss.write_index(index, "data/my_index_file.index")
```

这将在你的当前目录中创建一个名为 *my_index_file.index* 的文件，其中包含序列化的索引。你可以稍后使用 `faiss.read_index` 将此索引加载回内存：

```py
# Load the index from a file
index = faiss.read_index("data/my_index_file.index")
```

这样，你可以在不同的会话之间持久化你的索引，甚至在不同机器或环境中共享它。只需确保小心处理这些文件，因为对于大型索引，它们可能相当大。

如果你有一个以上的已保存向量数据库，合并它们也是可能的。这在序列化文档加载或对记录进行批量更新时可能很有用。

你可以使用 `faiss.IndexFlatL2` 索引的 `add` 方法合并两个 FAISS 索引：

```py
# Assuming index1 and index2 are two IndexFlatL2 indices
index1.add(index2.reconstruct_n(0, index2.ntotal))
```

在此代码中，`reconstruct_n(0, index2.ntotal)` 用于从 `index2` 中获取所有向量，然后使用 `index1.add()` 将这些向量添加到 `index1` 中，从而有效地合并两个索引。

这应该会工作，因为 `faiss.IndexFlatL2` 支持使用 `reconstruct` 方法检索向量。但是请注意，此过程不会将 `index2` 中与向量关联的任何 ID 移动到 `index1`。合并后，`index2` 中的向量将在 `index1` 中具有新的 ID。

如果你需要保留向量 ID，你需要通过保留从向量 ID 到你的数据项的单独映射来外部管理。然后，在合并索引时，你也会合并这些映射。

###### 提示

注意，此方法可能不适用于所有类型的索引，特别是对于不支持 `reconstruct` 方法的索引，如 `IndexIVFFlat` 或如果两个索引具有不同的配置。在这些情况下，最好保留构建每个索引所使用的原始向量，然后合并并重新构建索引。

# 基于 LangChain 的 RAG

作为最受欢迎的 AI 工程框架之一，LangChain 涵盖了广泛的 RAG 技术。其他框架如 [LlamaIndex](https://www.llamaindex.ai) 专注于 RAG，对于复杂用例值得探索。由于你熟悉 LangChain（见第四章），我们将继续在这个框架中展示本章节的示例。在手动根据所需上下文执行 RAG 之后，让我们使用 LCEL 在四个小型文本文档上创建一个类似的示例，使用 FAISS：

```py
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1\. Create the documents:
documents = [
    "James Phoenix worked at JustUnderstandingData.",
    "James Phoenix currently is 31 years old.",
    """Data engineering is the designing and building systems for collecting,
 storing, and analyzing data at scale.""",
]

# 2\. Create a vectorstore:
vectorstore = FAISS.from_texts(texts=documents, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 3\. Create a prompt:
template = """Answer the question based only on the following context:
---
Context: {context} `---`
`Question:` `{question}` ``` `"""` `prompt` `=` `ChatPromptTemplate``.``from_template``(``template``)`  `# 4\. 创建聊天模型:` `model` `=` `ChatOpenAI``()` ```py
```

```py`` ````代码首先从 LangChain 库中导入必要的模块，并定义了一个要处理的文本文档列表。它使用 `FAISS`，一个用于高效相似性搜索的库，从文本文档中创建一个向量存储。这涉及到使用 OpenAI 的嵌入模型将文本转换为向量嵌入。初始化一个用于处理问题和生成响应的 `ChatOpenAI` 模型。此外，提示模板强制 LLM 只使用检索器提供的上下文进行回复。您需要创建一个包含 `"context"` 和 `"question"` 键的 LCEL 链：```py chain = (     {"context": retriever, "question": RunnablePassthrough()}     | prompt     | model     | StrOutputParser() ) ```    通过将检索器添加到 `"context"`，它将自动获取四个转换为字符串值的文档。结合 `"question"` 键，然后用于格式化提示。LLM 生成一个响应，然后由 `StrOutputParser()` 解析为字符串值。您将调用链并传入您的问题，该问题被分配给 `"question"`，并手动测试三个不同的查询：```py chain.invoke("What is data engineering?") # 'Data engineering is the process of designing and building systems for # collecting, storing, and analyzing data at scale.'  chain.invoke("Who is James Phoenix?") # 'Based on the given context, James Phoenix is a 31-year-old individual who # worked at JustUnderstandingData.'  chain.invoke("What is the president of the US?") # I don't know ```    注意 LLM 只适当地回答了前两个查询，因为它在向量数据库中没有包含任何相关上下文来回答第三个查询！LangChain 的实现代码量显著减少，易于阅读，并允许您快速实现检索增强生成。```py` `````  ```py`` ````# 基于 Pinecone 的托管向量数据库    有许多托管向量数据库提供商出现，以支持 AI 用例，包括 [Chroma](https://www.trychroma.com)，[Weaviate](https://weaviate.io)，和 [Pinecone](https://www.pinecone.io)。其他类型的数据库托管商也在提供向量搜索功能，例如 [Supabase](https://supabase.com) 与 [pgvector 扩展](https://oreil.ly/pgvector)。本书中的示例使用 Pinecone，因为它是当时的市场领导者，但使用模式在提供商之间相对一致，概念应该是可转移的。托管向量数据库相对于开源本地向量存储具有几个优势：维护      使用托管向量数据库，您无需担心自己设置、管理和维护数据库。这可以节省大量时间和资源，尤其是对于可能没有专门的 DevOps 或数据库管理团队的企业来说。可伸缩性      托管向量数据库旨在与您的需求一起扩展。随着您的数据增长，数据库可以自动扩展以处理增加的负载，确保您的应用程序继续高效运行。可靠性      管理服务通常提供高可用性以及服务级别协议，以及自动备份和灾难恢复功能。这可以提供安心，并让您免受潜在数据丢失的风险。性能      托管向量数据库通常具有优化基础设施和算法，可以提供比自管理的开源解决方案更好的性能。这对于依赖于实时或近实时向量搜索功能的应用程序尤为重要。支持      使用托管服务，您通常可以访问提供服务的公司的支持。如果您遇到问题或需要帮助优化数据库的使用，这将非常有帮助。安全性      管理服务通常实施强大的安全措施来保护您的数据，包括加密、访问控制和监控等。主要托管提供商更有可能拥有必要的合规证书，并符合欧盟等地区的隐私法规。当然，这种额外功能是有成本的，以及过度支出的风险。就像使用 Amazon Web Services、Microsoft Azure 或 Google Cloud 一样，开发者因配置错误或代码错误而意外花费数千美元的故事屡见不鲜。还存在一些供应商锁定风险，因为尽管每个供应商都有类似的功能，但在某些方面存在差异，因此迁移它们并不完全直接。另一个主要考虑因素是隐私，因为与第三方共享数据会带来安全风险和潜在的法律后果。    与您设置开源 FAISS 向量存储时的工作步骤相同，与托管向量数据库一起工作。首先，您将文档分块并检索向量；然后，您在向量数据库中索引文档块，以便检索与查询相似的记录，以便将其作为上下文插入。首先，让我们在流行的商业向量数据库提供商 [Pinecone](https://www.pinecone.io) 中创建一个索引。然后登录 Pinecone 并检索您的 API 密钥（访问侧菜单中的 API Keys 并单击“创建 API 密钥”）。此示例的代码在本书的 [GitHub 仓库](https://oreil.ly/Q0rIw) 中提供。    输入：```py from pinecone import Pinecone, ServerlessSpec import os  # Initialize connection (get API key at app.pinecone.io): os.environ["PINECONE_API_KEY"] = "insert-your-api-key-here"  index_name = "employee-handbook" environment = "us-west-2" pc = Pinecone()  # This reads the PINECONE_API_KEY env var  # Check if index already exists: # (it shouldn't if this is first time) if index_name not in pc.list_indexes().names():     # if does not exist, create index     pc.create_index(         index_name,         # Using the same vector dimensions as text-embedding-ada-002         dimension=1536,         metric="cosine",         spec=ServerlessSpec(cloud="aws", region=environment),     )  # Connect to index: index = pc.Index(index_name)  # View index stats: index.describe_index_stats() ```    输出：```py {'dimension': 1536,  'index_fullness': 0.0,  'namespaces': {},  'total_vector_count': 0} ```    让我们逐步分析此代码：    1.  导入必要的库。脚本从导入必要的模块开始。`from pinecone import Pinecone, ServerlessSpec,` `import os` 用于访问和设置环境变量。      2.  设置 Pinecone API 密钥。至关重要的 Pinecone API 密钥通过使用 `os.environ["PINECONE_API_KEY"] = "insert-your-api-key-here"` 设置为环境变量。重要的是用您实际的 Pinecone API 密钥替换 `"insert-your-api-key-here"`。      3.  定义索引名称和环境。变量 `index_name` 和 `environment` 被设置。`index_name` 被赋予值 `"employee-handbook"`，这是要在 Pinecone 数据库中创建或访问的索引的名称。`environment` 变量被分配为 `"us-west-2"`，表示服务器的位置。      4.  初始化 Pinecone 连接。使用 `Pinecone()` 构造函数初始化与 Pinecone 的连接。此构造函数会自动从环境变量中读取 `PINECONE_API_KEY`。      5.  检查现有索引。脚本检查名为 `index_name` 的索引是否已存在于 Pinecone 数据库中。这是通过 `pc.list_indexes().names()` 函数完成的，该函数返回所有现有索引名称的列表。      6.  创建索引。如果索引不存在，则使用 `pc.create_index()` 函数创建它。此函数使用配置新索引的几个参数调用：    *   `index_name`：指定索引的名称           *   `dimension=1536`：将存储在索引中的向量的维度设置为 1536           *   `metric='cosine'`：确定将使用余弦相似度度量来比较向量                7.  连接到索引。在验证或创建索引后，脚本使用 `pc.Index(index_name)` 连接到它。此连接对于后续操作（如插入或查询数据）是必要的。      8.  索引统计信息。脚本以调用 `index.describe_index_stats()` 结束，该函数检索并显示有关索引的各种统计信息，例如其维度和存储的向量的总数。      接下来，您需要将向量存储在新建的索引中，通过遍历所有文本块和向量并将它们作为记录上载到 Pinecone。数据库操作 `upsert` 是 `update` 和 `insert` 的组合，它将更新现有记录或如果记录不存在则插入新记录（有关 `chunks` 变量的更多信息，请参阅 [此 Jupyter Notebook](https://oreil.ly/YC-nV)）。    输入：```py from tqdm import tqdm # For printing a progress bar from time import sleep  # How many embeddings you create and insert at once batch_size = 10 retry_limit = 5  # maximum number of retries  for i in tqdm(range(0, len(chunks), batch_size)):     # Find end of batch     i_end = min(len(chunks), i+batch_size)     meta_batch = chunks[i:i_end]     # Get ids     ids_batch = [str(j) for j in range(i, i_end)]     # Get texts to encode     texts = [x for x in meta_batch]     # Create embeddings     # (try-except added to avoid RateLimitError)     done = False     try:         # Retrieve embeddings for the whole batch at once         embeds = []         for text in texts:             embedding = get_vector_embeddings(text)             embeds.append(embedding)         done = True     except:         retry_count = 0         while not done and retry_count < retry_limit:             try:                 for text in texts:                     embedding = get_vector_embeddings(text)                     embeds.append(embedding)                 done = True             except:                 sleep(5)                 retry_count += 1      if not done:         print(f"""Failed to get embeddings after         {retry_limit} retries.""")         continue      # Cleanup metadata     meta_batch = [{         'batch': i,         'text': x     } for x in meta_batch]     to_upsert = list(zip(ids_batch, embeds, meta_batch))      # Upsert to Pinecone     index.upsert(vectors=to_upsert) ```    输出：```py 100% 13/13 [00:53<00:00, 3.34s/it] ```    让我们逐步分析此代码：    1.  导入必要的库 `tqdm` 和 `time`。库 `tqdm` 显示进度条，`time` 提供了 `sleep()` 函数，该函数在此脚本中用于重试逻辑。           2.  将变量 `batch_size` 设置为 10（通常设置为 100 用于实际工作负载），表示在即将到来的循环中一次将处理多少项。同时设置 `retry_limit` 以确保我们在尝试五次后停止。           3.  `tqdm(range(0, len(chunks), batch_size))` 部分是一个循环，它将从 0 运行到 `chunks`（之前定义）的长度，步长为 `batch_size`。"chunks" 是要处理的文本列表。在这里使用 `tqdm` 显示此循环的进度条。           4.  `i_end` 变量被计算为 `chunks` 的长度和 `i + batch_size` 中的较小值。这用于防止 `i + batch_size` 超过 `chunks` 的长度时出现索引错误。           5.  `meta_batch` 是 `chunks` 的当前批次的子集。这是通过从 `chunks` 列表中的 `i` 到 `i_end` 切片创建的。           6.  `ids_batch` 是 `i` 到 `i_end` 范围的字符串表示形式的列表。这些是用于识别 `meta_batch` 中每个项目的 ID。           7.  `texts` 列表只是 `meta_batch` 中的文本，准备好进行嵌入处理。           8.  尝试通过调用 `get_vector_embeddings()` 并将 `texts` 作为参数传递来获取嵌入。结果存储在变量 `embeds` 中。这是在 `try-except` 块中完成的，以处理此函数可能引发的任何异常，例如速率限制错误。           9.  如果引发异常，脚本将进入一个 `while` 循环，其中它将睡眠五秒钟，然后再次尝试检索嵌入。它将继续这样做，直到成功或达到重试次数，此时它将 `done = True` 设置为退出 `while` 循环。           10.  将 `meta_batch` 修改为字典列表。每个字典有两个键：`batch`，设置为当前批次号 `i`，和 `text`，设置为 `meta_batch` 中相应的项目。这是您可以添加附加元数据以供以后过滤查询的地方，例如页面、标题或章节。           11.  使用 `zip` 函数将 `ids_batch`、`embeds` 和 `meta_batch` 组合为元组，然后将其转换为列表来创建 `to_upsert` 列表。每个元组包含每个批次的 ID、相应的嵌入和相应的元数据。           12.  循环的最后一行在 `index` 上调用一个名为 `upsert` 的方法，这是一个 Pinecone（向量数据库服务）索引的方法。`vectors=to_upsert` 参数将 `to_upsert` 列表作为要插入或更新到索引中的数据传递。如果索引中已存在具有给定 ID 的向量，则将其更新；如果不存在，则将其插入。              一旦记录存储在 Pinecone 中，您就可以像使用 FAISS 在本地保存您的向量一样查询它们。只要您使用相同的嵌入模型来检索查询的向量，嵌入就保持不变，因此除非您有额外的记录或元数据要添加，否则您无需更新数据库。    输入：```py # Retrieve from Pinecone user_query = "do we get free unicorn rides?"  def pinecone_vector_search(user_query, k):     xq = get_vector_embeddings(user_query)     res = index.query(vector=xq, top_k=k, include_metadata=True)     return res  pinecone_vector_search(user_query, k=1) ```    输出：```py {'matches':     [{'id': '15',     'metadata': {'batch': 10.0,     'text': "You'll enjoy a treasure chest of perks, "             'including unlimited unicorn rides, a '             'bottomless cauldron of coffee and potions, '             'and access to our company library filled '             'with spellbinding books. We also offer '             'competitive health and dental plans, '             'ensuring your physical well-being is as '             'robust as your magical spirit.\n'             '\n'             '**5: Continuous Learning and Growth**\n'             '\n'             'At Unicorn Enterprises, we believe in '             'continuous learning and growth.'},     'score': 0.835591,     'values': []},],  'namespace': ''} ```    此脚本使用 Pinecone 的 API 执行最近邻搜索，以在多维空间中识别与给定输入向量最相似的向量。以下是逐步分析：    1.  定义了一个名为 `pinecone_vector_search` 的函数，该函数有两个参数：`user_query` 和 `k`。"user_query" 是用户输入的文本，准备好转换为向量，而 `k` 指定了您想要检索的最近向量数量。           2.  在函数内部，通过调用另一个函数 `get_vector_embeddings(user_query)` 定义了 `xq`。此函数（之前定义）负责将 `user_query` 转换为向量表示。           3.  下一行在名为 `index` 的对象上执行查询，该对象是一个 Pinecone 索引对象，使用 `query` 方法。`query` 方法接受三个参数：                    第一个参数是 `vector=xq`，我们的 `user_query` 的向量表示。                    第二个参数 `top_k=k` 指定您只想从 Pinecone 索引中返回 `k` 个最近的向量。                    第三个参数 `include_metadata=True` 指定您想要包含返回结果中的元数据（例如 ID 或其他相关数据）。如果您想[通过元数据过滤结果](https://oreil.ly/BBYD4)，例如指定批次（或任何其他您已上载的元数据），您可以在此处添加第四个参数：`filter={"batch": 1}`。           4.  `query` 方法的输出分配给 `res`，然后通过函数返回。           5.  最后，使用 `user_query` 和 `k` 作为参数调用函数 `pinecone_vector_search`，返回 Pinecone 的响应。              您现在已经有效地模拟了 FAISS 的工作，通过向量相似性搜索返回手册中的相关记录。如果您将 `vector_search(user_query, k)` 替换为 `pinecone_vector_search(user_query, k)` 在 `search_and_chat` 函数（来自前面的示例）中，聊天机器人将运行相同的代码，除了向量将存储在托管的 Pinecone 数据库中而不是本地使用 FAISS。    当您将记录上载到 Pinecone 时，您将批次号作为元数据传递。Pinecone 支持以下元数据格式：    *   字符串           *   数字（整数或浮点数，转换为 64 位浮点数）           *   布尔值（true，false）           *   字符串列表              存储记录的元数据策略与分块策略一样重要，因为您可以使用元数据来过滤查询。例如，如果您只想在特定批次号中搜索相似性，您可以在 `index.query` 中添加一个过滤器：    ```py res = index.query(xq, filter={         "batch": {"$eq": 1}     }, top_k=1, include_metadata=True) ```    这可以限制您搜索相似性的范围。例如，它
