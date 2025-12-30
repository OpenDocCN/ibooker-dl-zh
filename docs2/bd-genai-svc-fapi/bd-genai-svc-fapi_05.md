# 第三章。人工智能集成与模型服务

这项工作是用 AI 翻译的。我们很高兴收到您的反馈和评论：translation-feedback@oreilly.com

在本章中，你将学习各种 GenAI 模型的机制以及如何在 FastAPI 应用程序中提供服务。此外，使用[Streamlit UI 包](https://oreil.ly/9BXmn)，你将创建一个简单的浏览器客户端来与提供模型的端点进行交互。我们将探讨不同的模型服务策略，例如如何预加载模型以提高效率，以及如何使用 FastAPI 的功能进行服务监控。

为了巩固本章所学内容，我们将逐步构建一个 FastAPI 服务，使用开源的 GenAI 模型生成文本、图像、音频和 3D 几何形状，一切从头开始。在接下来的章节中，你将构建你的 GenAI 服务的文档和内容分析功能，以便能够使用语言模型与他们进行对话。

###### 注意

在上一章中，你看到了如何配置 Python 中的新 FastAPI 项目。在阅读本章的其余部分之前，请确保你有一个新的安装。或者，你可以克隆或下载本书的[GitHub 仓库](https://github.com/Ali-Parandeh/building-generative-ai-services)。一旦克隆，切换到`ch03-start`分支，准备执行后续步骤。

到了这一章的结尾，你将拥有一个 FastAPI 服务，它可以服务于各种开源的 GenAI 模型，你可以在 Streamlit 用户界面内对其进行测试。此外，你的服务将能够使用中间件在磁盘上记录使用数据。

# 为客户服务的*生成性*模型

在你的应用程序中使用预训练的生成性模型之前，了解这些模型是如何训练和生成数据的很有价值。有了这些知识，你可以根据需要定制应用程序的内部结构，以改善提供给用户的结果。

在本章中，我将向你展示如何以不同的模式提供服务，包括：

+   基于转换器神经网络架构的*语言*模型

+   在 text-to-speech 和 text-to-audio 服务中基于激进变换架构的*音频*模型

+   基于稳定扩散和视觉变换架构的*视觉*模型，用于 text-to-image 和 text-to-video 服务

+   基于条件隐式函数编码器和解码器架构的 text-to-3D 服务的*3D*模型

这个列表并不全面，只涵盖了一小部分 GenAI 模型。要探索其他模型，请访问[Hugging Face 模型仓库](https://oreil.ly/-4wlQ)^(1)

## 语言模型

在本节中，我们将讨论语言模型，包括转换器和循环神经网络（RNN）。

### 转换器与循环神经网络（RNN）

人工智能（AI）领域因重要文章“Attention Is All You Need”的发表而受到震动。2 在这篇文章中，作者提出了一种完全不同的自然语言处理（NLP）和序列建模方法，与现有的循环神经网络（RNN）架构不同。

图 3-1 展示了文章原始提案中提出的转换器架构的简化版本。

![bgai 0301](img/bgai_0301.png)

###### 图 3-1\. 转换器架构

从历史上看，文本生成活动利用循环神经网络（RNN）模型来学习序列数据（如自由文本）中的模型。为了处理文本，这些模型将其分割成小块，如单词或字符，称为*标记（token*），这些标记可以按顺序进行处理。

循环神经网络（RNN）维护一个称为*状态向量（vector of state*）的记忆档案，它将信息从一个标记传递到另一个标记，贯穿整个文本序列，直到序列的末尾。这意味着当到达文本序列的末尾时，最初标记对状态向量的影响远小于较近的标记。

理想情况下，每个标记（token）在文本中应该具有与其他标记相同的重要性。然而，由于循环神经网络（RNN）只能通过观察前面的元素来预测序列中的下一个元素，因此它们无法捕捉到长距离依赖关系，也无法在大量文本中建模模型。因此，它们无法记住或理解大型文档中的关键信息或上下文。

随着转换器的发明，循环或卷积建模可以被更有效的方法所取代。由于转换器不维护隐藏状态的记忆，并利用一种称为*自我注意力（auto-attention*）的新能力，因此它们能够建模单词之间的关系，而不管它们在句子中的距离有多远。这个自我注意力组件允许模型“关注”句子中上下文中相关的单词。

当循环神经网络（RNN）建模句子中相邻单词之间的关系时，转换器将文本中每个单词之间的成对关系映射到关系上。

图 3-2 展示了循环神经网络（RNN）与转换器在处理句子方面的差异。

![bgai 0302](img/bgai_0302.png)

###### 图 3-2\. RNN 与转换器在句子处理中的比较

自我注意力系统由称为*注意力头（heads of attention*）的特殊块提供动力，这些块捕获单词之间的成对模式，如*注意力图（attention maps*）。

图 3-3 展示了注意力头（即注意力单元）的注意力图。3）连接可以是双向的，而厚度则代表句子中词语之间关系的强度。

![bgai 0303](img/bgai_0303.png)

###### 图 3-3. 注意力头内部的注意力图

一个变压器模型包含多个注意力头，这些头分布在网络神经层的不同层中。每个注意力头独立计算自己的注意力图，以捕捉输入中特定模式下的词语关系。通过使用多个注意力头，模型可以从不同的角度和上下文中同时分析输入，以理解数据中的复杂模式和依赖关系。

图 3-4 显示了模型每个层中每个注意力头（即一组独立的注意力权重）的注意力图。

![bgai 0304](img/bgai_0304.png)

###### 图 3-4. 模型内部的注意力图

RNN 还需要大量的计算能力来训练，因为其训练过程不能在多个 GPU 上并行化，因为其训练算法具有序列性质。相反，变压器以非序列方式处理词语，因此可以在 GPU 上并行执行注意力机制。

变压器架构的效率意味着当有更多数据、计算能力和内存时，这些模型更具可扩展性。可以使用涵盖人类生产的书籍库的语料库构建语言模型。这一切所需的就是强大的计算能力和数据来训练一个大型语言模型（LLM）。这正是 OpenAI 所做的事情，它是著名应用 ChatGPT 背后的公司，该应用由多个自有的 LLM 提供支持，包括 GPT-4o。

到我们撰写本文时，OpenAI 的 LLM 实现细节仍被视为商业机密。尽管许多研究人员对 OpenAI 的方法有一般了解，但并不意味着他们有资源来复制这些方法。然而，从那时起，已经发布了多个开源替代方案，用于研究和商业用途，包括 Llama（Facebook）、Gemma（Google）、Mistral 和 Falcon 等，仅举几个例子。4）到我们撰写本文时，模型的规模从 0.05B 到 480B 个参数（即模型的权重和偏差）不等，以适应您的需求。

LLM 服务仍然是一个挑战，因为对内存的高要求，如果需要在你的数据集上训练和调整，这些要求会加倍。这是因为训练过程需要在多个训练批次之间缓存和重用模型参数。因此，大多数组织可以依赖轻量级模型（高达 3B）或 OpenAI、Anthropic、Cohere、Mistral 等 LLM 提供商的 API。

随着 LLM 的普及，了解它们如何训练和如何处理数据变得尤为重要，因此我们来谈谈背后的机制。

### 分词和嵌入

神经网络不能直接处理单词，因为它们是处理数字的大型统计模型。为了弥合语言和数字之间的差距，需要依赖 *分词*。通过分词，将文本分解成模型可以处理的更小的片段。

任何文本都必须首先分为代表单词、音节、符号和标点的 *标记* 列表。然后，这些标记被映射为唯一的数字，以便可以数值化地建模。

向训练好的转换器提供输入的标记向量，网络可以预测生成文本的最佳下一个标记，一次一个单词。

图 3-5 展示了 OpenAI 分词器如何将文本转换为一系列标记，并为每个标记分配唯一的标识符。

![bgai 0305](img/bgai_0305.png)

###### 图 3-5\. OpenAI 分词器（来源：[OpenAI](https://oreil.ly/S-a9M))

那么，在分词文本之后可以做什么？这些标记在语言模型可以处理之前需要进一步处理。

分词之后，需要使用一个 *嵌入器*^(5) 将这些标记转换为实数密集向量，称为 *嵌入*，这些嵌入在连续向量空间中捕获语义信息（即每个标记的含义）。图 3-6 展示了这些嵌入。

![bgai 0306](img/bgai_0306.png)

###### 图 3-6\. 在嵌入过程中为每个标记分配一个嵌入向量

###### 建议

这些嵌入向量使用小的 *浮点数*（非整数）来捕捉标记之间微妙的关系，具有更大的灵活性和精确度。此外，它们通常呈正态分布，因此语言模型的训练和推理可以更加稳定和一致。

在嵌入过程之后，每个标记都被分配了一个由 *n* 个数字组成的嵌入向量。嵌入向量中的每个数字都专注于表示标记含义的一个特定维度。

### 训练转换器

一旦获得一系列嵌入向量，你可以在你的文档上训练一个模型来更新每个嵌入中的值。在模型训练过程中，训练算法会更新嵌入层的参数，使得嵌入向量尽可能描述输入文本中每个标记的意义。

理解嵌入向量的工作原理可能很困难，因此我们尝试一种可视化方法。

假设你使用的是二维嵌入向量，即只包含两个数字的向量。如果你在模型训练前后追踪这些向量，你会观察到类似于图 3-7 中的图表。具有相似意义的标记或单词向量将彼此更接近。

![bgai 0307](img/bgai_0307.png)

###### 图 3-7. 使用嵌入向量训练的变换器网络潜在空间

为了确定两个单词之间的相似度，你可以使用一种称为*余弦相似度*的计算方法。较小的角度意味着更大的相似度，这表示语境和意义相似。在训练后，计算具有相似意义的嵌入向量之间的余弦相似度将验证这些向量彼此之间是接近的。

图 3-8 展示了整个分词、嵌入和形成的全过程。

![bgai 0308](img/bgai_0308.png)

###### 图 3-8. 将序列数据（如文本）作为标记向量和标记嵌入进行加工

一旦获得训练好的嵌入层，你可以用它来将任何新的输入文本嵌入到图 3-1 中所示的变换器模型中。

### 位置编码

在将嵌入向量传递到变换器网络的注意力层之前，最后一个阶段是实施*位置编码*。位置编码过程产生位置嵌入向量，然后这些向量被加到标记嵌入向量上。

由于变换器是同时而不是按顺序处理单词的，因此需要位置嵌入来记录序列数据（如句子）中的单词顺序和上下文。结果向量嵌入捕捉了句子中单词的意义和位置信息，在传递给变换器的注意力机制之前。这个过程确保注意力头拥有所有必要的信息来有效地学习模型。

图 3-9 展示了位置编码过程，其中位置嵌入被加到标记嵌入上。

![bgai 0309](img/bgai_0309.png)

###### 图 3-9\. 位置编码

### 自回归预测

转换器是一个自回归模型（即序列模型），因为未来的预测基于过去的价值，如图 3-10 所示。#autoregressive_prediction3。

![bgai 0310](img/bgai_0310.png)

###### 图 3-10\. 自回归预测

模型接收输入标记，然后将其嵌入并通过网络进行预测以生成下一个标记的最佳预测。这个过程会一直重复，直到生成一个`<stop>`或句子结束标记`<eos>`。^(6)

然而，模型在其内存中可以存储的标记数量有一个上限，用于生成下一个标记。这个标记上限被称为模型的*上下文窗口*，在选择用于你的 GenAI 服务的模型时是一个重要的考虑因素。

如果达到上下文窗口的极限，模型会简单地丢弃最近使用的标记，这意味着它可能会*忘记*在文档或对话消息中最近使用的句子。

###### 注意

在我们撰写本文时，成本较低的 OpenAI `gpt-4o-mini`模型的上下文约为 128,000 个标记，相当于超过 300 页的文本。

2025 年 3 月最大的上下文窗口属于[Magic.Dev LTM-2-mini](https://oreil.ly/10Mj1)，拥有 1 亿个标记。这相当于约 750 部小说的约 1000 万行代码。

其他模型的上下文窗口大约在数十万个标记左右。

短窗口会导致信息丢失，难以维持对话，以及与用户查询的连贯性降低。

另一方面，较长的上下文窗口需要更高的内存要求，并且当有成千上万的用户同时使用你的服务时，可能会引起性能问题或服务延迟。此外，你还需要考虑使用具有较宽上下文窗口的模型的成本，因为它们通常更昂贵，因为它们对计算和内存的要求更高。正确的选择将取决于你的预算和你的用例中用户的需求。

### 在你的应用程序中集成语言模型

你可以用几行代码在你的应用程序中下载和使用语言模型。在 Nel Esempio 3-1 中，你将下载一个拥有 1.1 亿参数、在 3000 亿标记上预训练的 TinyLlama 模型。

```py`##### Esempio 3-1\. Scaricare e caricare un modello linguistico dal repository di Hugging Face    ``` # models.py  import torch from transformers import Pipeline, pipeline  prompt = "How to set up a FastAPI project?" system_prompt = """ Your name is FastAPI bot and you are a helpful chatbot responsible for teaching FastAPI to your users. Always respond in markdown. """  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ![1](img/1.png)  def load_text_model():     pipe = pipeline(         "text-generation",         model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", ![2](img/2.png)         torch_dtype=torch.bfloat16,         device=device ![3](img/3.png)     )     return pipe   def generate_text(pipe: Pipeline, prompt: str, temperature: float = 0.7) -> str:     messages = [         {"role": "system", "content": system_prompt},         {"role": "user", "content": prompt},     ] ![4](img/4.png)     prompt = pipe.tokenizer.apply_chat_template(         messages, tokenize=False, add_generation_prompt=True     ) ![5](img/5.png)     predictions = pipe(         prompt,         temperature=temperature,         max_new_tokens=256,         do_sample=True,         top_k=50,         top_p=0.95,     ) ![6](img/6.png)     output = predictions[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1] ![7](img/7.png)     return output ```py    ![1](img/#co_ai_integration_and_model_serving_CO1-1)      Controlla se è disponibile una GPU NVIDIA e, in caso affermativo, imposta `device` sulla GPU corrente abilitata a CUDA. Altrimenti, continua a usare la CPU.      ![2](img/#co_ai_integration_and_model_serving_CO1-2)      Scarica e carica in memoria il modello TinyLlama con un tipo di dati di precisione tensoriale `float16`.^(9)      ![3](img/#co_ai_integration_and_model_serving_CO1-3)      Sposta l'intera pipeline sulla GPU al primo caricamento.      ![4](img/#co_ai_integration_and_model_serving_CO1-4)      Prepara l'elenco dei messaggi, che consiste in dizionari con coppie chiave-valore di ruolo e contenuto. L'ordine dei dizionari detta l'ordine dei messaggi da quelli più vecchi a quelli più recenti in una conversazione. Il primo messaggio è spesso un prompt del sistema per guidare l'output del modello in una conversazione.      ![5](img/#co_ai_integration_and_model_serving_CO1-5)      Convertire l'elenco dei messaggi di chat in un elenco di token interi per il modello. Al modello viene quindi chiesto di generare un output in formato testuale, non in token interi `tokenize=False`. Viene anche aggiunto un prompt di generazione alla fine dei messaggi di chat (`add_generation_prompt=True`) in modo che il modello sia incoraggiato a generare una risposta basata sulla cronologia della chat.      ![6](img/#co_ai_integration_and_model_serving_CO1-6)      Il prompt preparato viene passato al modello con diversi parametri di inferenza per ottimizzare le prestazioni di generazione del testo. Alcuni di questi parametri di inferenza chiave includono:    *   `max_new_tokens`: Specifica il numero massimo di nuovi token da generare nell'output.           *   `do_sample`: Determina, quando produce l'output, se scegliere un token in modo casuale da un elenco di token adatti (`True`) o se scegliere semplicemente il token più probabile a ogni passo (`False`).           *   `temperature`: I valori più bassi rendono i risultati del modello più precisi, mentre quelli più alti consentono risposte più creative.           *   `top_k`: Limita le previsioni dei token del modello alle prime K opzioni.`top_k=50` significa creare un elenco dei 50 token più adatti da scegliere nella fase di previsione dei token corrente.           *   `top_p`: Implementa il *campionamento dei nuclei* quando si crea un elenco dei token più adatti.`top_p=0.95` significa creare un elenco dei token migliori fino a quando non si è soddisfatti che l'elenco contenga il 95% dei token più adatti da cui scegliere, per la fase di previsione dei token corrente.                ![7](img/#co_ai_integration_and_model_serving_CO1-7)      L'output finale è ottenuto dall'oggetto `predictions`. Il testo generato da TinyLlama include l'intera cronologia della conversazione, con la risposta generata aggiunta alla fine. Il token di stop `</s>` seguito dai token `\n<|assistant|>\n` sono utilizzati per selezionare il contenuto dell'ultimo messaggio della conversazione, che è la risposta del modello.      L'esempio 3-1 è un buon punto di partenza; puoi caricare questo modello sulla tua CPU e ottenere risposte in tempi ragionevoli. Tuttavia, TinyLlama potrebbe non avere le stesse prestazioni delle sue controparti più grandi. Per i carichi di lavoro di produzione, vorrai utilizzare modelli più grandi per ottenere una migliore qualità e prestazioni.    A questo punto puoi utilizzare le funzioni `load_model` e `predict` all'interno di una funzione del controller^(10) e poi aggiungere un decoratore di gestione delle rotte per servire il modello tramite un endpoint, come mostrato nell'Esempio 3-2.    ##### Esempio 3-2\. Servire un modello linguistico tramite un endpoint FastAPI    ``` # main.py  from fastapi import FastAPI from models import load_text_model, generate_text  app = FastAPI()  @app.get("/generate/text") ![1](img/1.png) def serve_language_model_controller(prompt: str) -> str: ![2](img/2.png)     pipe = load_text_model() ![3](img/3.png)     output = generate_text(pipe, prompt) ![4](img/4.png)     return output ![5](img/5.png) ```py    ![1](img/#co_ai_integration_and_model_serving_CO2-1)      Crea un server FastAPI e aggiungi un gestore di rotte `/generate` per servire il modello.      ![2](img/#co_ai_integration_and_model_serving_CO2-2)      Il sito `serve_language_model_controller` è responsabile di prendere il prompt dai parametri della query di richiesta.      ![3](img/#co_ai_integration_and_model_serving_CO2-3)      Il modello viene caricato in memoria.      ![4](img/#co_ai_integration_and_model_serving_CO2-4)      Il controllore passa la query al modello per eseguire la previsione.      ![5](img/#co_ai_integration_and_model_serving_CO2-5)      Il server FastAPI invia l'output come risposta HTTP al client.      Una volta che il servizio FastAPI è attivo e funzionante, puoi visitare la pagina di documentazione Swagger all'indirizzo `http://localhost:8000/docs`per testare il tuo nuovo endpoint:    ``` http://localhost:8000/generate/text?prompt="What is FastAPI?" ```py    Se stai eseguendo gli esempi di codice su una CPU, ci vorrà circa un minuto per ricevere una risposta dal modello, come mostrato nella Figura 3-11.  ![bgai 0311](img/bgai_0311.png)  ###### Figura 3-11\. Risposta di TinyLlama    Non è una cattiva risposta per un piccolo modello di linguaggio (SLM) che gira su una CPU del tuo computer, se non fosse che TinyLlama ha avuto *l'allucinazione* di credere che FastAPI utilizzi Flask. Si tratta di un'affermazione errata: FastAPI utilizza Starlette come framework web sottostante, non Flask.    Le*allucinazioni* si riferiscono a risultati che non sono basati sui dati di addestramento o sulla realtà. Anche se SLM open source come TinyLlama sono stati addestrati su un numero impressionante di token (3 trilioni), un numero ridotto di parametri del modello può aver limitato la loro capacità di apprendere la verità di base nei dati.Inoltre, potrebbero essere stati utilizzati anche dati di addestramento non filtrati, che possono contribuire ad aumentare i casi di allucinazioni.    ###### Avvertenze    Quando utilizzi i modelli linguistici, informa sempre i tuoi utenti di controllare i risultati con fonti esterne, perché i modelli linguistici potrebbero avere delle *allucinazioni* e produrre affermazioni errate.    Ora puoi utilizzare un client per browser web in Python per testare visivamente il tuo servizio con maggiore interattività rispetto all'utilizzo di un client a riga di comando.    Un ottimo pacchetto Python per sviluppare rapidamente un'interfaccia utente è [Streamlit](https://oreil.ly/9BXmn), che ti permette di creare UI belle e personalizzabili per i tuoi servizi di AI con poco sforzo.````  ```py`` ### Connettere FastAPI con il generatore di UI Streamlit    Streamlit ti permette di creare facilmente un'interfaccia utente di chat per il test e la prototipazione di modelli. Puoi installare il pacchetto `streamlit` utilizzando `pip`:    ``` $ pip install streamlit ```py   ```` L'esempio 3-3 mostra come sviluppare una semplice interfaccia utente per connettersi al servizio.    ##### Esempio 3-3\. L'interfaccia utente della chat di Streamlit che utilizza l'endpoint FastAPI /`generate`    ```py # client.py  import requests import streamlit as st  st.title("FastAPI ChatBot") ![1](img/1.png)  if "messages" not in st.session_state:     st.session_state.messages = [] ![2](img/2.png)  for message in st.session_state.messages:     with st.chat_message(message["role"]):         st.markdown(message["content"]) ![3](img/3.png)  if prompt := st.chat_input("Write your prompt in this input field"): ![4](img/4.png)     st.session_state.messages.append({"role": "user", "content": prompt}) ![5](img/5.png)      with st.chat_message("user"):         st.text(prompt) ![6](img/6.png)      response = requests.get(         f"http://localhost:8000/generate/text", params={"prompt": prompt}     ) ![7](img/7.png)     response.raise_for_status() ![8](img/8.png)      with st.chat_message("assistant"):         st.markdown(response.text) ![9](img/9.png) ```    ![1](img/#co_ai_integration_and_model_serving_CO3-1)      Aggiungi un titolo alla tua applicazione che sarà reso all'interfaccia utente.      ![2](img/#co_ai_integration_and_model_serving_CO3-2)      Inizializza la chat e tiene traccia della cronologia della chat.      ![3](img/#co_ai_integration_and_model_serving_CO3-3)      Visualizza i messaggi della cronologia delle chat al riavvio dell'applicazione.      ![4](img/#co_ai_integration_and_model_serving_CO3-4)      Attendi che l'utente invii un prompt tramite il campo di inserimento della chat.      ![5](img/#co_ai_integration_and_model_serving_CO3-5)      Aggiungi i messaggi dell'utente o dell'assistente alla cronologia delle chat.      ![6](img/#co_ai_integration_and_model_serving_CO3-6)      Visualizza il messaggio dell'utente nel contenitore dei messaggi della chat.      ![7](img/#co_ai_integration_and_model_serving_CO3-7)      Invia una richiesta `GET` con il prompt come parametro di query al tuo endpoint FastAPI per generare una risposta da TinyLlama.      ![8](img/#co_ai_integration_and_model_serving_CO3-8)      Convalida che la risposta sia OK.      ![9](img/#co_ai_integration_and_model_serving_CO3-9)      Visualizza il messaggio dell'assistente nel contenitore dei messaggi della chat.      Ora puoi avviare l'applicazione client Streamlit:^(11)    ```py $ streamlit run client.py ```   `Ora dovresti essere in grado di interagire con TinyLlama all'interno di Streamlit, come mostrato nella [Figura 3-12](#streamlit_ui_text_results
