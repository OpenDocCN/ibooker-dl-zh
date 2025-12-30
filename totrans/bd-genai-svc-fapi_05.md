# Chapter 3\. AI Integration and Model Serving

In this chapter, you will learn the mechanisms of various GenAI models and how to serve them in a FastAPI application. Additionally, using the [Streamlit UI package](https://oreil.ly/9BXmn), you will create a simple browser client for interacting with the model-serving endpoints. We will explore differing model-serving strategies, how to preload models for efficiency, and how to use FastAPI features for service monitoring.

To solidify your learning in this chapter, we will progressively build a FastAPI service using open source GenAI models that generate text, images, audio, and 3D geometries, all from scratch. In later chapters, you’ll build the functionality to parse documents and web content for your GenAI service so you can talk to them using a language model.

###### Note

In the previous chapter, you saw how to set up a fresh FastAPI project in Python. Make sure you have fresh installation ready before you read the rest of this chapter. Alternatively, you can clone or download the book’s [GitHub repository](https://github.com/Ali-Parandeh/building-generative-ai-services). Then once cloned, switch to the `ch03-start` branch, ready for the steps to follow.

By the end of this chapter, you will have a FastAPI service that serves various open source GenAI models that you can test inside the Streamlit UI. Additionally, your service will be capable of logging usage data to disk using middleware.

# Serving Generative Models

Before you serve pretrained generative models in your application, it is worth learning how these models are trained and generate data. With this understanding, you can customize the internals of your application to enhance the outputs that you provide to the user.

In this chapter, I will show you how to serve models across a variety of modalities including:

*   *Language* models based on the transformer neural network architecture

*   *Audio* models in text-to-speech and text-to-audio services based on the aggressive transformer architecture

*   *Vision* models for text-to-image and text-to-video services based on the Stable Diffusion and vision transformer architectures

*   *3D* models for text-to-3D services based on the conditional implicit function encoder and diffusion decoder architecture

This list is not exhaustive and covers a handful of GenAI models. To explore other models, please visit the [Hugging Face model repository](https://oreil.ly/-4wlQ).^([1](ch03.html#id630))

## Language Models

In this section, we talk about language models, including transformers and recurrent neural networks (RNNs).

### Transformers versus recurrent neural networks

The world of AI was shaken with the release of the landmark paper “Attention Is All You Need.”^([2](ch03.html#id631)) In this paper, the authors proposed a completely different approach to natural language processing (NLP) and sequence modeling that differed from the existing RNN architectures.

[Figure 3-1](#transformer_architecture) shows a simplified version of the proposed transformer architecture from the original paper.

![bgai 0301](assets/bgai_0301.png)

###### Figure 3-1\. Transformer architecture

Historically, text generation tasks leveraged RNN models to learn patterns in sequential data such as free text. To process text, these models chunk text into small pieces such as a word or character called a *token* that can be sequentially processed.

RNNs maintain a memory store called a *state vector*, which carries information from one token to the next throughout the full text sequence, until the end. This means that by the time you get to the end of the text sequence, the impact of early tokens on the state vector is a lot smaller compared to the most recent tokens.

Ideally, every token should be as important as the other tokens in any text. However, as RNNs can only predict the next item in a sequence by looking at the items that came before, they struggle with this ideal in capturing long-range dependencies and modeling patterns in large chunks of texts. As a result, they effectively fail to remember or comprehend essential information or context in large documents.

With the invention of transformers, recurrent or convolutional modeling could now be replaced with a more efficient approach. Since transformers don’t maintain a hidden state memory and leverage a new capability termed *self-attention*, they’re capable of modeling relationships between words, no matter how far apart they appeared in a sentence. This self-attention component allows the model to “place attention” on contextually relevant words within a sentence.

While RNNs model relationships between neighboring words in a sentence, transformers map pairwise relationships between every word in the text.

[Figure 3-2](#rnn_vs_transformer) shows how RNNs process sentences in comparison to transformers.

![bgai 0302](assets/bgai_0302.png)

###### Figure 3-2\. RNNs versus transformers in processing sentences

What powers the self-attention system are specialized blocks called *attention heads* that capture pairwise patterns between words as *attention maps*.

[Figure 3-3](#head_attention_map) visualizes the attention map of an attention head.^([3](ch03.html#id635)) Connections can be bidirectional with the thickness representing the strength of the relationship between words in the sentence.

![bgai 0303](assets/bgai_0303.png)

###### Figure 3-3\. View of an attention map inside an attention head

A transformer model contains several attention heads distributed across its neural network layers. Each head computes its own attention map independently to capture relationships between words focusing on certain patterns in the inputs. Using multiple attention heads, the model can simultaneously analyze the inputs from various angles and contexts to understand complex patterns and dependencies within the data.

[Figure 3-4](#model_attention_map) shows the attention maps for each head (i.e., independent set of attention weights) within each layer of the model.

![bgai 0304](assets/bgai_0304.png)

###### Figure 3-4\. View of the attention maps within the model

RNNs also required extensive compute power to train, as the training process couldn’t be parallelized on multiple GPU due to the sequential nature of their training algorithms. Transformers, on the other hand, process words nonsequentially, so they can run attention mechanisms in parallel on GPUs.

The efficiency of the transformer architecture means that these models are more scalable as long as there is more data, compute power, and memory. You can build language models with a corpus that spans libraries of books produced by humanity. All you would need is ample compute power and data to train an LLM. And, that is exactly what OpenAI did, the company behind the famous ChatGPT application that was powered by several of their proprietary LLMs including GPT-4o.

At the time of this writing, the implementation details behind OpenAI’s LLMs remain a trade secret. While many researchers have a general understanding of OpenAI’s methods, they may not necessarily have the resources to replicate them. However, several open source alternatives for research and commercial use have been released since, including Llama (Facebook), Gemma (Google), Mistral, and Falcon to name a few.^([4](ch03.html#id637)) At the time of this writing, the model sizes vary between 0.05B and 480B parameters (i.e., model weights and biases) to suit your needs.

Serving LLMs still remains a challenge due to high memory requirements with requirements doubling if you need to train and fine-tune them on your own dataset. This is because the training process will require caching and reusing model parameters across training batches. As a result, most organizations may rely on lightweight (up to 3B) models or on APIs of LLM providers such as OpenAI, Anthropic, Cohere, Mistral, etc.

As LLMs grow in popularity, it becomes even more important to understand how they’re trained and how they process data, so let’s discuss underlying mechanisms next.

### Tokenization and embedding

Neural networks can’t process words directly as they’re big statistical models that function on numbers. To bridge that gap between language and numbers, you need to use *tokenization*. With tokenization, you break down text into smaller pieces that a model can process.

Any piece of text must be first sliced into a list of *tokens* that represent words, syllables, symbols, and punctuations. These tokens are then mapped to unique numbers so that patterns can be numerically modeled.

By providing a vector of input tokens to a trained transformer, the network can then predict the next best token to generate text, one word at a time.

[Figure 3-5](#openai_tokenizer) shows how the OpenAI tokenizer converts text into a sequence of tokens, assigning unique token identifiers to each.

![bgai 0305](assets/bgai_0305.png)

###### Figure 3-5\. OpenAI tokenizer (Source: [OpenAI](https://oreil.ly/S-a9M))

So what can you do after you tokenize some text? These tokens need to be processed further before a language model can process them.

After tokenization, you need to use an *embedder*^([5](ch03.html#id647)) to convert these tokens into dense vectors of real numbers called *embeddings*, capturing semantic information (i.e., meaning of each token) in a continuous vector space. [Figure 3-6](#embeddings) demonstrates these embeddings.

![bgai 0306](assets/bgai_0306.png)

###### Figure 3-6\. Assigning an embedding vector of size n to each token during the embedding process

###### Tip

These embedding vectors use small *floating-point numbers* (not integers) to capture nuanced relationships between tokens with more flexibility and precision. They also tend to be *normally distributed*, so language model training and inference can be more stable and consistent.

After the embedding process, each token is assigned an embedding vector filled with *n* numbers. Each number in the embedding vector focuses on a dimension that represents a specific aspect of the token’s meaning.

### Training transformers

Once you have a set of embedding vectors, you can train a model on your documents to update the values inside each embedding. During model training, the training algorithm updates the parameters of the embedding layers so that the embedding vectors describe the meaning of each token as close as possible within the input text.

Understanding how embedding vectors work can be challenging, so let’s try a visualization approach.

Imagine you used a two-dimensional embedding vectors, meaning the vectors contained only two numbers. Then, if you plot these vectors, before and after model training, you will observe plots similar to [Figure 3-7](#untrained_to_trained_transformer). The embedding vectors of tokens, or words, with similar meanings will be closer to each other.

![bgai 0307](assets/bgai_0307.png)

###### Figure 3-7\. Training latent space of transformer network using embedding vectors

To determine the similarity between two words, you can compute the angle between vectors using a calculation known as *cosine similarity*. Smaller angles imply higher similarity, representing similar context and meaning. After training, the cosine similarity calculation of two embedding vectors with similar meanings will validate that those vectors are close to each other.

[Figure 3-8](#embedding_vectors) illustrates the full tokenization, embedding, and training process.

![bgai 0308](assets/bgai_0308.png)

###### Figure 3-8\. Processing sequential data like a piece of text into a vector of tokens and token embeddings

Once you have a trained embedding layer, you can now use it to embed any new input text to the transformer model shown in [Figure 3-1](#transformer_architecture).

### Positional encoding

A final step before forwarding the embedding vectors to the attention layers in the transformer network is to implement *positional encoding*. The positional encoding process produces the positional embedding vectors that then are summed with the token embedding vectors.

Since transformers process words simultaneously rather than sequentially, positional embeddings are needed to record the word order and context within the sequential data, like sentences. The resultant embedding vectors capture both meaning and positional information of words in the sentences before they’re passed to the attention mechanisms of the transformer. This process ensures attention heads have all the information they need to learn patterns effectively.

[Figure 3-9](#positional_encoding) shows the positional encoding process where the positional embeddings are summed with token embeddings.

![bgai 0309](assets/bgai_0309.png)

###### Figure 3-9\. Positional encoding

### Autoregressive prediction

The transformer is an autoregressive (i.e., sequential) model as future predictions are based on the past values, as shown in [Figure 3-10](#autoregressive_prediction3).

![bgai 0310](assets/bgai_0310.png)

###### Figure 3-10\. Autoregressive prediction

The model receives input tokens that are then embedded and passed through the network to make the next best token prediction. This process repeats until a `<stop>` or end of sentence `<eos>` token is generated.^([6](ch03.html#id658))

However, there is a limit to the number of tokens that the model can store in its memory to generate the next token. This token limit is referred to as the model’s *context window*, which is an important factor to consider during the model selection stage for your GenAI services.

If the context window limit is reached, the model simply discards the least recently used tokens. This means it can *forget* the least recently used sentences in documents or messages in a conversation.

###### Note

At the time of writing, the context of the least expensive OpenAI `gpt-4o-mini` model is around ~128,000 tokens, equivalent to more than 300 pages of text.

The largest context window as of March 2025 belongs to [Magic.Dev LTM-2-mini](https://oreil.ly/10Mj1) with 100 million tokens. This equals ~10 million lines of code of ~750 novels.

The context window of other models falls in the range of hundreds of thousands of tokens.

Short windows will lead to loss of information, difficulty maintaining conversations, and reduced coherence with the user query.

On the other hand, long context windows have larger memory requirements and can lead to performance issues or slow services when scaling to thousands of concurrent users who are using your service. In addition, you will need to consider the costs of relying on models with larger context windows as they tend to be more expensive due to increased compute and memory requirements. The correct choice will depend on your budget and user needs in your use case.

### Integrating a language model into your application

You can download and use a language model within your application with a few lines of code. In [Example 3-1](#language_model_usage_example), you will download a TinyLlama model that has 1.1 billion parameters and is pretrained on 3 trillion tokens.

##### Example 3-1\. Download and load a language model from the Hugging Face repository

```py
# models.py

import torch
from transformers import Pipeline, pipeline

prompt = "How to set up a FastAPI project?"
system_prompt = """
Your name is FastAPI bot and you are a helpful
chatbot responsible for teaching FastAPI to your users.
Always respond in markdown.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ![1](assets/1.png)

def load_text_model():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", ![2](assets/2.png)
        torch_dtype=torch.bfloat16,
        device=device ![3](assets/3.png)
    )
    return pipe

def generate_text(pipe: Pipeline, prompt: str, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ] ![4](assets/4.png)
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) ![5](assets/5.png)
    predictions = pipe(
        prompt,
        temperature=temperature,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    ) ![6](assets/6.png)
    output = predictions[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1] ![7](assets/7.png)
    return output
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO1-1)

Check if an NVIDIA GPU is available, and if so, set `device` to the current CUDA-enabled GPU. Otherwise, continue using the CPU.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO1-2)

Download and load the TinyLlama model into memory with a `float16` tensor precision data type.^([9](ch03.html#id667))

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO1-3)

Move the whole pipeline to GPU on the first load.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO1-4)

Prepare the message list, which consists of dictionaries that have role and content key-value pairs. The order of the dictionaries dictates the order of messages from older to newer in a conversation. The first message is often a system prompt to guide the model’s output in a conversation.

[![5](assets/5.png)](#co_ai_integration_and_model_serving_CO1-5)

Convert the list of chat messages into a list of integer tokens for the model. The model is then asked to generate output in textual format, not integer tokens `tokenize=False`. A generation prompt is also added to the end of chat messages (`add_generation_prompt=True`) so that the model is encouraged to generate a response based on the chat history.

[![6](assets/6.png)](#co_ai_integration_and_model_serving_CO1-6)

The prepared prompt is passed to the model with several inference parameters to optimize the text generation performance. A few of these key inference parameters include:

*   `max_new_tokens`: Specifies the maximum number of new tokens to generate in the output.

*   `do_sample`: Determines, when producing output, whether to pick a token randomly from a list of suitable tokens (`True`) or to simply choose the most likely token at each step (`False`).

*   `temperature`: Controls the randomness of the output generation. Lower values make the model’s outputs more precise, while higher values allow for more creative responses.

*   `top_k`: Restricts the model’s token predictions to the top K options. `top_k=50` means create a list of top 50 most suitable tokens to pick from in the current token prediction step.

*   `top_p`: Implements *nucleus sampling* when creating a list of most suitable tokens. `top_p=0.95` means create a list of the top tokens until you’re satisfied that your list has 95% of the most suitable tokens to pick from, for the current token prediction step.

[![7](assets/7.png)](#co_ai_integration_and_model_serving_CO1-7)

The final output is obtained from the `predictions` object. The generated text from TinyLlama includes the full conversation history, with the generated response appended to the end. The `</s>` stop token followed by `\n<|assistant|>\n` tokens are used to pick the content of the last message in the conversation, which is the model’s response.

[Example 3-1](#language_model_usage_example) is a good starting point; you can still load this model on your CPU and get responses within a reasonable time. However, TinyLlama may also not perform as well as its larger counterparts. For production workloads, you will want to use bigger models for better output quality and performance.

You can now use the `load_model` and `predict` functions inside a controller function^([10](ch03.html#id668)) and then add a route handling decorator to serve the model via an endpoint, as shown in [Example 3-2](#text_endpoint).

##### Example 3-2\. Serving a language model via a FastAPI endpoint

```py
# main.py

from fastapi import FastAPI
from models import load_text_model, generate_text

app = FastAPI()

@app.get("/generate/text") ![1](assets/1.png)
def serve_language_model_controller(prompt: str) -> str: ![2](assets/2.png)
    pipe = load_text_model() ![3](assets/3.png)
    output = generate_text(pipe, prompt) ![4](assets/4.png)
    return output ![5](assets/5.png)
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO2-1)

Create a FastAPI server and add a `/generate` route handler for serving the model.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO2-2)

The `serve_language_model_controller` is responsible for taking the prompt from the request query parameters.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO2-3)

The model is loaded into memory.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO2-4)

The controller passes the query to the model to perform the prediction.

[![5](assets/5.png)](#co_ai_integration_and_model_serving_CO2-5)

The FastAPI server sends the output as an HTTP response to the client.

Once the FastAPI service is up and running, you can visit the Swagger documentation page located at `http://localhost:8000/docs` to test your new endpoint:

```py
http://localhost:8000/generate/text?prompt="What is FastAPI?"
```

If you’re running the code samples on a CPU, it will take around a minute to receive a response from the model, as shown in [Figure 3-11](#text_gen_response).

![bgai 0311](assets/bgai_0311.png)

###### Figure 3-11\. Response from TinyLlama

Not a bad response for a small language model (SLM) that runs on a CPU in your own computer, except that TinyLlama has *hallucinated* that FastAPI uses Flask. That is an incorrect statement; FastAPI uses Starlette as the underlying web framework, not Flask.

*Hallucinations* refer to outputs that aren’t grounded in the training data or reality. Even though open source SLMs such as TinyLlama have been trained on impressive number of tokens (3 trillion), a small number of model parameters may have restricted their ability to learn the ground truth in data. Additionally, some unfiltered training data may also have been used, both of which can contribute to more instances of hallucinations.

###### Warning

When serving language models, always let your users know to fact-check the outputs with external sources as language models may *hallucinate* and produce incorrect statements.

You can now use a web browser client in Python to visually test your service with more interactivity compared to using a command-line client.

A great Python package to quickly develop a user interface is [Streamlit](https://oreil.ly/9BXmn), which enables you to create beautiful and customizable UIs for your AI services with little effort.

### Connecting FastAPI with Streamlit UI generator

Streamlit allows you to easily create a chat user interface for testing and prototyping with models. You can install the `streamlit` package using `pip`:

```py
$ pip install streamlit
```

[Example 3-3](#streamlit_chat_ui) shows how to develop a simple UI to connect with your service.

##### Example 3-3\. Streamlit chat UI consuming the FastAPI /`generate` endpoint

```py
# client.py

import requests
import streamlit as st

st.title("FastAPI ChatBot") ![1](assets/1.png)

if "messages" not in st.session_state:
    st.session_state.messages = [] ![2](assets/2.png)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) ![3](assets/3.png)

if prompt := st.chat_input("Write your prompt in this input field"): ![4](assets/4.png)
    st.session_state.messages.append({"role": "user", "content": prompt}) ![5](assets/5.png)

    with st.chat_message("user"):
        st.text(prompt) ![6](assets/6.png)

    response = requests.get(
        f"http://localhost:8000/generate/text", params={"prompt": prompt}
    ) ![7](assets/7.png)
    response.raise_for_status() ![8](assets/8.png)

    with st.chat_message("assistant"):
        st.markdown(response.text) ![9](assets/9.png)
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO3-1)

Add a title to your application that will be rendered to the UI.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO3-2)

Initialize the chat and keep track of the chat history.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO3-3)

Display the chat messages from the chat history on app rerun.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO3-4)

Wait until the user has submitted a prompt via the chat input field.

[![5](assets/5.png)](#co_ai_integration_and_model_serving_CO3-5)

Add the user or assistant messages to the chat history.

[![6](assets/6.png)](#co_ai_integration_and_model_serving_CO3-6)

Display the user message in the chat message container.

[![7](assets/7.png)](#co_ai_integration_and_model_serving_CO3-7)

Send a `GET` request with the prompt as a query parameter to your FastAPI endpoint to generate a response from TinyLlama.

[![8](assets/8.png)](#co_ai_integration_and_model_serving_CO3-8)

Validate the response is OK.

[![9](assets/9.png)](#co_ai_integration_and_model_serving_CO3-9)

Display the assistant message in the chat message container.

You can now start your Streamlit client application:^([11](ch03.html#id674))

```py
$ streamlit run client.py
```

You should now be able to interact with TinyLlama inside Streamlit, as shown in [Figure 3-12](#streamlit_ui_text_results). All of this was possible with a few short Python scripts.

![bgai 0312](assets/bgai_0312.png)

###### Figure 3-12\. Streamlit client

[Figure 3-13](#tiny_llama_fastapi_architecture) shows the overall system architecture of the solution we’ve developed so far.

![bgai 0313](assets/bgai_0313.png)

###### Figure 3-13\. FastAPI service system architecture

###### Warning

While the solution in [Example 3-3](#streamlit_chat_ui) is great for prototyping and testing models, it is not suitable for production workloads where several users would need simultaneous access to the model. This is because with the current setup, the model is loaded and unloaded onto memory every time a request is processed. Having to load/unload a large model to and from memory is slow and I/O blocking.

The TinyLlama service you’ve just built used a *decoder* transformer, optimized for conversational and chat use cases. However, the [original paper on transformers](https://oreil.ly/RqztC) introduced an architecture that consisted of both an encoder and a decoder.

You should now feel more confident in the inner workings of language models and how to package them in a FastAPI web server.

Language models represent just a fraction of all generative models. The upcoming sections will expand your knowledge to include the function and serving of models that generate audio, images, and videos.

We can start working with audio models first.

## Audio Models

In GenAI services, audio models are important for creating interactive and realistic sounds. Unlike text models that you’re now familiar with, which focus on processing and generating text, audio models can handle audio signals. With them, you can synthesize speech, generate music, and even create sound effects for applications like virtual assistants, automated dubbing, game development, and immersive audio environments.

One of the most capable text-to-speech and text-to-audio models is the Bark model created by Suno AI. This transformer-based model can generate realistic multilingual speech and audio including music, background noise, and sound effects.

The Bark model consists of four models chained together as a pipeline to synthesize audio waveforms from textual prompts, as shown in [Figure 3-15](#bark_pipeline).

![bgai 0315](assets/bgai_0315.png)

###### Figure 3-15\. Bark synthesis pipeline

1. Semantic text model

A causal (sequential) autoregressive transformer model accepts tokenized input text and captures the meaning via semantic tokens. Autoregressive models predict future values in a sequence by reusing their own previous outputs.

2. Coarse acoustics model

A causal autoregressive transformer receives the semantic model’s outputs and generates the initial audio features, which lack finer details. Each prediction is based on past and present information in the semantic token sequence.

3. Fine acoustics model

A noncausal auto-encoder transformer refines the audio representation by generating the remaining audio features. As the coarse acoustics model has generated the entire audio sequence, the fine model doesn’t need to be casual.

4. Encodec audio codec model

The model decodes the output audio array from all previously generated audio codes.

Bark synthesizes the audio waveform by decoding the refined audio features into the final audio output in the form of spoken words, music, or simple audio effects.

[Example 3-4](#small_bark) shows how to use the small Bark model.

##### Example 3-4\. Download and load the small Bark model from the Hugging Face repository

```py
# schemas.py

from typing import Literal

VoicePresets = Literal["v2/en_speaker_1", "v2/en_speaker_9"] ![1](assets/1.png)

# models.py
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel, BarkProcessor, BarkModel
from schemas import VoicePresets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio_model() -> tuple[BarkProcessor, BarkModel]:
    processor = AutoProcessor.from_pretrained("suno/bark-small", device=device) ![2](assets/2.png)
    model = AutoModel.from_pretrained("suno/bark-small", device=device) ![3](assets/3.png)
    return processor, model

def generate_audio(
    processor: BarkProcessor,
    model: BarkModel,
    prompt: str,
    preset: VoicePresets,
) -> tuple[np.array, int]:
    inputs = processor(text=[prompt], return_tensors="pt",voice_preset=preset) ![4](assets/4.png)
    output = model.generate(**inputs, do_sample=True).cpu().numpy().squeeze() ![5](assets/5.png)
    sample_rate = model.generation_config.sample_rate ![6](assets/6.png)
    return output, sample_rate
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO4-1)

Specify supported voice preset options using a `Literal` type.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO4-2)

Download the small Bark processor, which prepares the input text prompt for the core model.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO4-3)

Download the Bark model, which will be used to generate the output audio. Both objects will be needed for audio generation later.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO4-4)

Preprocess the text prompt with a speaker voice preset embedding and return a Pytorch tensor array of tokenized inputs using `return_tensors="pt"`.

[![5](assets/5.png)](#co_ai_integration_and_model_serving_CO4-5)

Generate an audio array that contains amplitude values of the synthesized audio signal over time.

[![6](assets/6.png)](#co_ai_integration_and_model_serving_CO4-6)

Get the sampling rate from model generating configurations, which can be used to produce the audio.

When you generate audio using a model, the output is a sequence of floating-point numbers that represent the *amplitude* (or strength) of the audio signal at each point in time.

To play back this audio, it needs to be converted to a digital format that can be sent to the speakers. This involves sampling the audio signal at a fixed rate and quantizing the amplitude values to a fixed number of bits. The `soundfile` library can help you here by generating the audio file using a *sampling rate*. The higher the sampling rate, the more samples that are taken, which enhances the audio quality but also increases the file size.

You can install the `soundfile` audio library for writing audio files using `pip`:

```py
$ pip install soundfile
```

[Example 3-5](#audio_endpoint) shows how you can stream the audio content to the client.

##### Example 3-5\. FastAPI endpoint for returning generated audio

```py
# utils.py

from io import BytesIO
import soundfile
import numpy as np

def audio_array_to_buffer(audio_array: np.array, sample_rate: int) -> BytesIO:
    buffer = BytesIO()
    soundfile.write(buffer, audio_array, sample_rate, format="wav") ![1](assets/1.png)
    buffer.seek(0)
    return buffer ![2](assets/2.png)

# main.py

from fastapi import FastAPI, status
from fastapi.responses import StreamingResponse

from models import load_audio_model, generate_audio
from schemas import VoicePresets
from utils import audio_array_to_buffer

@app.get(
    "/generate/audio",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
) ![3](assets/3.png)
def serve_text_to_audio_model_controller(
    prompt: str,
    preset: VoicePresets = "v2/en_speaker_1",
):
    processor, model = load_audio_model()
    output, sample_rate = generate_audio(processor, model, prompt, preset)
    return StreamingResponse(
        audio_array_to_buffer(output, sample_rate), media_type="audio/wav"
    ) ![4](assets/4.png)
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO5-1)

Install the `soundfile` library to write the audio array to memory buffer using its sampling rate.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO5-2)

Reset the buffer cursor to the start of the buffer and return the iterable buffer.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO5-3)

Create a new audio endpoint that returns the `audio/wav` content type as `StreamingResponse`. `StreamingResponse` is typically used when you want to stream the response data, such as when returning large files or when generating the response data. It allows you to return a generator function that yields chunks of data to be sent to the client.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO5-4)

Convert the generated audio array to an iterable buffer that can be passed to streaming response.

In [Example 3-5](#audio_endpoint), you generated an audio array using the small Bark model and streamed the memory buffer of the audio content. Streaming is more efficient for larger files as the client can consume the content as it is being served. In previous examples, we didn’t use streaming responses, as generated images or text can be fairly small compared to audio or video content.

###### Tip

Streaming audio content directly from a memory buffer is faster and more efficient than writing the audio array to a file and streaming the content from the hard drive.

If you need the memory available for other tasks, you can write the audio array to a file first and then stream from it using a file reader generator. You will be trading off latency for memory.

Now that you have an audio generation endpoint, you can update your Streamlit UI client code to render audio messages. Update your Streamlit client code as shown in [Example 3-6](#barksmall_streamlit_ui).

##### Example 3-6\. Streamlit audio UI consuming the FastAPI `/audio` generation endpoint

```py
# client.py

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, bytes):
            st.audio(content)
        else:
            st.markdown(content)

if prompt := st.chat_input("Write your prompt in this input field"):
    response = requests.get(
        f"http://localhost:8000/generate/audio", params={"prompt": prompt}
    )
    response.raise_for_status()
    with st.chat_message("assistant"):
        st.text("Here is your generated audio")
        st.audio(response.content) ![1](assets/1.png)
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO6-1)

Update the Streamlit client code to render audio content.

With Streamlit, you can swap components to render any type of content including images, audio, and video.

You should now be able to generate highly realistic speech audio in your updated Streamlit UI, as shown in [Figure 3-16](#streamlit_bark_ui).

![bgai 0316](assets/bgai_0316.png)

###### Figure 3-16\. Rendering audio responses in the Streamlit UI

Bear in mind that you’re using the compressed version of the Bark model, but with the light version, you can generate speech and music audio fairly quickly even on a single CPU. This is in exchange for some audio generation quality.

You should now feel more comfortable serving larger content to your users via streaming responses and working with audio models.

So far, you’ve been building conversational and text-to-speech services. Now let’s see how to interact with a vision model to build an image generator service.

## Vision Models

Using vision models, you can generate, enhance, and understand visual information from prompts.

Since these models can produce very realistic outputs faster than any human and can understand and manipulate existing visual content, they’re extremely useful for applications like image generators and editors, object detection, image classification and captioning, and augmented reality.

One of the most popular architectures used to train image models is called *Stable Diffusion* (SD).

SD models are trained to encode input images into a latent space. This latent space is the mathematical representation of patterns in the training data that the model has learned. If you try to visualize an encoded image, all you would see is a white noise image, similar to the black and white dots you would see on your TV screen when it loses signal.

[Figure 3-17](#stable_diffusion) shows the full process for training and inference and visualizes how images are encoded and decoded via the forward and reverse diffusion processes. A text encoder using text, images, and semantic maps assists in controlling the output via the reverse diffusion.

![bgai 0317](assets/bgai_0317.png)

###### Figure 3-17\. Stable Diffusion training and inference

What makes these models magical is their ability to decode noisy images back into original input images. Effectively, the SD models also learn to remove white noise from an encoded image to reproduce the original image. The model performs this denoising process over several iterations.

However, you don’t want to re-create images you already have. You will want the model to create new, never-before-seen images. But how can an SD model achieve this for you? The answer lies in the latent space where the encoded noisy images live. You can change the noise in these images so that when the model denoises them and decodes them back, you get a whole new image that the model has never seen before.

A challenge remains: how can you control the image generation process so that the model doesn’t produce random images? The solution is to also encode image descriptions alongside the image. The patterns in the latent space are then mapped to textual image descriptions of what is seen in each input image. Now, you use textual prompts to sample the noisy latent space such that the produced output image after the denoising process is what you want.

This is how SD models can generate new images that they’ve never seen before in their training data. In essence, these models navigate a latent space that contains encoded representations of various patterns and meanings.^([12](ch03.html#id686)) The model iteratively refines this noise through a denoising process to produce a novel image not present in its training dataset.

To download an SD model, you will need to have the Hugging Face `diffusers` library installed:

```py
$ pip install diffusers
```

[Example 3-7](#sd_model_usage_example) shows how to load an SD model into memory.

##### Example 3-7\. Download and load an SD model from the Hugging Face repository

```py
# models.py

import torch
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipelineLegacy
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image_model() -> StableDiffusionInpaintPipelineLegacy:
    pipe = DiffusionPipeline.from_pretrained(
        "segmind/tiny-sd", torch_dtype=torch.float32,
        device=device
    ) ![1](assets/1.png)
    return pipe

def generate_image(
    pipe: StableDiffusionInpaintPipelineLegacy, prompt: str
) -> Image.Image:
    output = pipe(prompt, num_inference_steps=10).images[0] ![2](assets/2.png) ![3](assets/3.png)
    return output ![4](assets/4.png)
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO7-1)

Download and load the TinySD model into memory with the less memory efficient `float32` tensor type. Using `float16`, which has limited precision for large and complex models, leads to numerical instability and loss of accuracy. Additionally, hardware support for `float16` is limited, so trying to run an SD model on your CPU with the `float16` tensor type may not be possible. Source: [Hugging Face](https://oreil.ly/rzw8P).

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO7-2)

Pass the text prompt to the model to generate a list of images and pick the first one. Some models allow you to generate multiple images in a single inference step.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO7-3)

The `num_inference_steps=10` specifies the number of diffusion steps to perform during inference. In each diffusion step, a stronger noisy image is produced from previous diffusion steps. The model generates multiple noisy images by undertaking multiple diffusion steps. With these images, the model can better understand the patterns of noise that are present in the input data and learn to remove them more effectively. The more inference steps, the better results you will get, but at the cost of computing power needed and longer processing times.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO7-4)

The generated image will be a Python Pillow image type, so you have access to a variety of Pillow’s image methods for post-processing and storage. For instance, you can call the `image.save()` method to store the image in your filesystem.

###### Note

Vision models are extremely resource hungry. To load and use a small vision model such as TinySD on CPU, you will need around 5 GB of disk space and RAM. However, you can install `accelerate` using `pip install accelerate` to optimize resources required so that the model pipeline uses lower CPU memory usage.

When serving video models, you will need to use a GPU. Later in this chapter, I will show you how to leverage GPUs for video models.

You can now package this model into another endpoint as similar to [Example 3-2](#text_endpoint), with the difference being that the returned response will be an image binary (not text). Refer to [Example 3-8](#image_endpoint).

##### Example 3-8\. FastAPI endpoint for returning a generated image

```py
# utils.py

from typing import Literal
from PIL import Image
from io import BytesIO

def img_to_bytes(
    image: Image.Image, img_format: Literal["PNG", "JPEG"] = "PNG"
) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=img_format)
    return buffer.getvalue() ![1](assets/1.png)

# main.py

from fastapi import FastAPI, Response, status
from models import load_image_model, generate_image
from utils import img_to_bytes

...

@app.get("/generate/image",
         responses={status.HTTP_200_OK: {"content": {"image/png": {}}}}, ![2](assets/2.png)
         response_class=Response) ![3](assets/3.png)
def serve_text_to_image_model_controller(prompt: str):
    pipe = load_image_model()
    output = generate_image(pipe, prompt) ![4](assets/4.png)
    return Response(content=img_to_bytes(output), media_type="image/png") ![5](assets/5.png)
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO8-1)

Create an in-memory buffer, save the image to this buffer in a given format, and then return the raw byte data from the buffer.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO8-2)

Specify the media content type and status codes for the auto-generated Swagger UI documentation page.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO8-3)

Specify the response class to prevent FastAPI from adding `application/json` as an additional acceptable response media type.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO8-4)

The response returned from the model will be Pillow image format.

[![5](assets/5.png)](#co_ai_integration_and_model_serving_CO8-5)

We will need to use the FastAPI `Response` class to send a special response carrying image bytes with a PNG media type.

[Figure 3-18](#tinysd_swagger_docs) shows the results of testing the new `/generate/image` endpoint via FastAPI Swagger docs with the text prompt `A cosy living room with trees in it`.

![bgai 0318](assets/bgai_0318.png)

###### Figure 3-18\. TinySD FastAPI service

Now, connect your endpoint to a Streamlit UI for prototyping, as shown in [Example 3-9](#tinysd_streamlit_code).

##### Example 3-9\. Streamlit Vision UI consuming the FastAPI `*/image*` generation endpoint

```py
# client.py

...

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.image(message["content"]) ![1](assets/1.png)
...

if prompt := st.chat_input("Write your prompt in this input field"):
    ...
    response = requests.get(
        f"http://localhost:8000/generate/image", params={"prompt": prompt}
    ) ![2](assets/2.png)
    response.raise_for_status()
    with st.chat_message("assistant"):
        st.text("Here is your generated image")
        st.image(response.content)

    ...
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO9-1)

Images transferred over the HTTP protocol will be in binary format. Therefore, we update the display function to render binary image content. You can use the `st.image` method to display images to the UI.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO9-2)

Update the `GET` request to hit the `/generate/image` endpoint. Then, render a textual and image message to the user.

[Figure 3-19](#tinysd_streamlitui) shows the final results of the user experience with the model.

![bgai 0319](assets/bgai_0319.png)

###### Figure 3-19\. Rendering image messages in the Streamlit UI

We saw how even with a tiny SD model, you can generate reasonable looking images. The XL versions can produce even more realistic images but still have their own limitations.

At the time of writing, the current open source SD models do have certain limitations:

Coherency

The models can’t produce every detail described in the prompts and complex compositions.

Output size

The output images can only be predefined sizes such as 512 × 512 or 1024 × 1024 pixels.

Composability

You can’t fully control the generated image and define composition in the image.

Photorealism

The generated outputs do show details that give away they’ve been generated by AI.

Legible text

Some models cannot generate legible texts.

The `tinysd` model you worked with is an early phase model that has undergone the *distillation* process (i.e., compression) from the larger V1.5 SD model. As a result, the generated outputs may not meet production standards or be entirely cohesive and could fail to incorporate all the concepts mentioned in the text prompts. However, the distilled models may perform well if you [*fine-tune* them using *Low-Rank Adaptation* (LoRA)](https://oreil.ly/Nqtkm) on specific concepts/styles.

You can now build both text- and image-based GenAI services. However, you may be wondering how to build text-to-video services based on video models. Let’s learn more about video models, how they work, and how to build an image animator service with them next.

## Video Models

Video models are some of the most resource-hungry generative models and often require a GPU to produce a short snippet of good quality. These models have to generate several tens of frames to produce a single second of video, even without any audio content.

Stability AI has released several open source video models based on the SD architecture on Hugging Face. We will work with the compressed version of their image-to-video model for a faster image animation service.

To get started, let’s get a small image-to-video model running using [Example 3-10](#video_model_loading).

###### Note

To run [Example 3-10](#video_model_loading), you may need access to a CUDA-capable NVIDIA GPU.

Also, for commercial use of the `stable-video-diffusion-img2vid` model, please refer to its [model card](https://oreil.ly/DM-0p).

##### Example 3-10\. Download and load the Stability AI’s *img2vid* model from the Hugging Face repository

```py
# models.py

import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_video_model() -> StableVideoDiffusionPipeline:
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
        variant="fp16",
        device=device,
    )
    return pipe

def generate_video(
    pipe: StableVideoDiffusionPipeline, image: Image.Image, num_frames: int = 25
) -> list[Image.Image]:
    image = image.resize((1024, 576)) ![1](assets/1.png)
    generator = torch.manual_seed(42) ![2](assets/2.png)
    frames = pipe(
        image, decode_chunk_size=8, generator=generator, num_frames=num_frames
    ).frames[0] ![3](assets/3.png)
    return frames
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO10-1)

Resize the input image to a standard size expected by model input. Resizing will also protect against large inputs.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO10-2)

Create a random tensor generator with the seed set to 42 for reproducible video frame generation.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO10-3)

Run the frame generation pipeline to produce all video frames at once. Grab the first batch of generated frames. This step requires significant video memory. `num_frames` specifies the number of frames to generate, while `decode_chunk_size` specifies how many frames to generate at once.

With the model loading functions in place, you can now build the video-serving endpoint.

However, before you proceed with declaring the route handler, you do need a utility function to process the video model outputs from frames into a streamable video using an I/O buffer.

To export a sequence of frames to videos, you need to encode them into a video container using a video library such as `av`, which implements Python bindings to the popular `ffmpeg` video processing library.

You can install the `av` library via:

```py
$ pip install av
```

Now you can use [Example 3-11](#frames_to_videos) to create streamable video buffers.

##### Example 3-11\. Exporting video model output from frames to a streamable video buffer using the `av` library

```py
# utils.py

from io import BytesIO
from PIL import Image
import av

def export_to_video_buffer(images: list[Image.Image]) -> BytesIO:
    buffer = BytesIO()
    output = av.open(buffer, "w", format="mp4") ![1](assets/1.png)
    stream = output.add_stream("h264", 30) ![2](assets/2.png)
    stream.width = images[0].width
    stream.height = images[0].height
    stream.pix_fmt = "yuv444p" ![3](assets/3.png)
    stream.options = {"crf": "17"} ![4](assets/4.png)
    for image in images:
        frame = av.VideoFrame.from_image(image)
        packet = stream.encode(frame)   ![5](assets/5.png)
        output.mux(packet) ![6](assets/6.png)
    packet = stream.encode(None)
    output.mux(packet)
    return buffer ![7](assets/7.png)
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO11-1)

Open a buffer for writing an MP4 file and then configure a video stream with AV’s video multiplexer.^([13](ch03.html#id697))

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO11-2)

Set the video encoding to `h264` at 30 frames per second and make sure the frame dimensions match the frames provided to the function.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO11-3)

Set the pixel format of the video stream to `yuv444p` so that each pixel has the full resolution for the `y` (luminance or brightness) and both `u` and `v` (chrominance or color) components.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO11-4)

Configure the stream’s constant rate factor (CRF) to control the video quality and compression. Set the CRF to 17 to output a lossless high-quality video with minimal compression.

[![5](assets/5.png)](#co_ai_integration_and_model_serving_CO11-5)

Encode the input frames into encoded packets with the configured stream video multiplexer.

[![6](assets/6.png)](#co_ai_integration_and_model_serving_CO11-6)

Add the encoded frames into the opened video container buffer.

[![7](assets/7.png)](#co_ai_integration_and_model_serving_CO11-7)

Flush any remaining frames in the encoder and combine the resulting packet into the output file before returning the buffer containing the encoded video.

To use image prompts with the service as file uploads, you must install the `python-multipart` library:^([14](ch03.html#id698))

```py
$ pip install python-multipart
```

Once installed, you can set up the new endpoint using [Example 3-12](#video_endpoint).

##### Example 3-12\. Serving generated videos from the image-to-video model

```py
# main.py

from fastapi import status, FastAPI, File
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image

from models import load_video_model, generate_video
from utils import export_to_video_buffer

...

@app.post(
    "/generate/video",
    responses={status.HTTP_200_OK: {"content": {"video/mp4": {}}}},
    response_class=StreamingResponse,
)
def serve_image_to_video_model_controller(
    image: bytes = File(...), num_frames: int = 25 ![1](assets/1.png)
):
    image = Image.open(BytesIO(image)) ![2](assets/2.png)
    model = load_video_model()
    frames = generate_video(model, image, num_frames)
    return StreamingResponse(
        export_to_video_buffer(frames), media_type="video/mp4" ![3](assets/3.png)
    )
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO12-1)

Use the `File` object to specify `image` as a form file upload.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO12-2)

Create a Pillow `Image` object by passing the image bytes transferred to the service. The model pipeline expects a Pillow image format as input.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO12-3)

Export the generated frames as a MP4 video and stream it to the client using an iterable video buffer.

With the video endpoint set up, you can now upload images to your FastAPI service to animate them as videos.

There are other video models available on the hub that allow you to generate GIFs and animations. For additional practice, you can try building a GenAI service with them. While open source video models can produce videos at ample quality, OpenAI’s announcement of a new large vision model (LVM) called Sora has shaken the video generation industry.

### OpenAI Sora

Text-to-video models are limited in their generation capabilities. Apart from the immense computational power needed to sequentially generate coherent video frames, training these models can be challenging due to:

*   *Maintaining temporal and spatial consistency across frames* to achieve realistic undistorted video outputs.

*   *Lack of training data* with high-quality caption and metadata needed to train video models.

*   *Captioning challenges* when captioning the content of videos clearly and descriptively is time-consuming and moves beyond drafting short pieces of text. Captioning must describe the narrative and scenes for each sequence for the model to learn and map the rich patterns contained in the video to text.

Because of these reasons, there has not been a breakthrough with video generation models until the announcement of OpenAI’s Sora model.

Sora is a generalist large vision diffusion transformer model capable of generating videos and images spanning diverse durations, aspect ratios, and resolutions, up to a full minute of high-definition video. Its architecture is based on the transformers commonly used in LLMs and the diffusion process. Whereas LLMs use text tokens, Sora uses visual patches.

###### Tip

The Sora model combines elements and principles of the transformer and SD architectures, while in [Example 3-10](#video_model_loading), you used Stability AI’s SD model to generate videos.

So what makes Sora different?

Transformers have demonstrated remarkable scalability across language models, computer vision, and image generation, so it made sense for Sora architecture to be based on transformers to handle diverse inputs like text, images, or video frames. Also, since transformers can understand complex patterns and long-range dependencies in sequential data, Sora as a vision transformer can also capture fine-grained temporal and spatial relationships between video frames to generate coherent frames with smooth transitions between them (i.e., exhibiting temporal consistency).

Furthermore, Sora borrows capabilities of the SD models to generate high-quality and visually coherent video frames with precise controls using the iterative noise reduction process. Using the diffusion process lets Sora generate images with fine detail and desirable properties.

By combining both sequential reasoning of transformers with iterative refinement of SD, Sora can generate high-resolution, coherent, and smooth videos from multimodal inputs like text and images that contain abstract concepts.

Sora’s network architecture is also designed to reduce dimensionality through a U-shape network where high-dimensional visual data is compressed and encoded into a latent noisy space. Sora can then generate patches from the latent space through the denoising diffusion process.

The diffusion process is similar to image-based SD models. Instead of having a 2D U-Net normally used for images, OpenAI has trained a 3D U-Net where the third dimension is a sequence of frames across time (making a video), as shown in [Figure 3-20](#images_to_videos).

![bgai 0320](assets/bgai_0320.png)

###### Figure 3-20\. A sequence of images forms a video

OpenAI has demonstrated that by compressing videos into patches, as shown in [Figure 3-21](#videos_to_patches), the model can achieve scalability of learning high-dimensional representations when training on diverse types of videos and images varying in resolution, durations, and aspect ratios.

![bgai 0321](assets/bgai_0321.png)

###### Figure 3-21\. Video compression into space-time patches

Through the diffusion process, Sora crunches input noisy patches to generate clean videos and images in any aspect ratio, size, and resolution for devices directly in their native screen sizes.

While a text transformer is predicting the next token in a text sequence, Sora’s vision transformer is predicting the next patch to generate an image or a video, as shown in [Figure 3-22](#vision_transformer_sequence).

![bgai 0322](assets/bgai_0322.png)

###### Figure 3-22\. Token prediction by the vision transformer

Through training on various datasets, OpenAI overcame the previously mentioned challenges of training vision models such as lack of quality captions, high dimensionality of video data, etc., to name a few.

What is fascinating about Sora and potentially other LVMs is the emerging capabilities they exhibit:

3D consistency

Objects in the generated scenes remain consistent and adjust to perspective even when the camera moves and rotates around the scene.

Object permanence and large range coherence

Objects and people that are occluded or leave a frame at a location will persist when they reappear in the field of view. In some cases, the model effectively remembers how to keep them consistent in the environment. This is also referred to as *temporal consistency* that most video models struggle with.

World interaction

Actions simulated in generated videos realistically affect the environment. For instance, Sora understands the action of eating a burger should leave a bite mark on it.

Simulating environments

Sora can also simulate worlds—real or fictional environments like in games—while adhering to the rules of interactions in those environments, such as playing a character in a *Minecraft* level. In other words, Sora has learned to be a data-driven physics engine.

[Figure 3-23](#sora_emerging_capabilities) illustrates these capabilities.

![bgai 0323](assets/bgai_0323.png)

###### Figure 3-23\. Sora’s emergent capabilities

At the time of this writing, Sora has not yet been released as an API, but open source alternatives have already emerged. A promising large vision model called “Latte” allows you to fine-tune the LVM on your own visual data.

###### Caution

You can’t yet commercialize some open source models, including Latte, at the time of writing. Always check the model card and the license to ensure any commercial use is allowed.

Combining transformers with diffusers to create LVMs is a promising area of research for generating complex outputs like videos. However, I imagine the same process can be applied for generating other types of high-dimensional data that can be represented as multidimensional arrays.

You should now feel more comfortable building services with text, audio, vision, and video models. Next, let’s take a look at another set of models capable of generating complex data such as 3D geometries by building a 3D asset generator service.

## 3D Models

You now understand how previously mentioned models use transformers and diffusers to generate any form of textual, audio, or visual data. Producing 3D geometries requires a different approach than image, audio, and text generation because you must account for spatial relationships, depth information, and geometric consistency, which add layers of complexity not present in other data types.

For 3D geometries, *meshes* are used to define the shape of an object. Software packages like Autodesk 3ds Max, Maya, and SolidWorks can be used to produce, edit, and render these meshes.

Meshes are effectively a collection of *vertices*, *edges*, and *faces* that reside in a 3D virtual space. Vertices are points in space that connect to form edges. Edges form faces (polygons) when they enclose on a flat surface, often in the shape of triangles or quadrilaterals. [Figure 3-24](#vertices_edges_faces) shows the differences between vertices, edges, and faces.

![bgai 0324](assets/bgai_0324.png)

###### Figure 3-24\. Vertices, edges, and faces

You can define vertices by their coordinates in a 3D space, usually determined by a Cartesian coordinate system (x, y, z). Essentially, the arrangement and connection of vertices form surfaces of a 3D mesh that define a geometry.

[Figure 3-25](#mesh) shows how these features combine to define a mesh of a 3D geometry such as a monkey’s head.

![bgai 0325](assets/bgai_0325.png)

###### Figure 3-25\. Mesh for 3D geometry of a monkey head using both triangular and quadrilateral polygons (shown in Blender, open source 3D modeling software)

You can train and use a transformer model to predict the next token in a sequence where the sequence is coordinates of vertices on a 3D mesh surface. Such a generative model can produce 3D geometries by predicting the next set of vertices and faces within a 3D space that form the desired geometry. However, the geometry would require thousands of vertices and faces to achieve a smooth surface.

This means for each 3D object, you need to wait for a long time for the generation to complete, and the results may still remain low fidelity. Because of this, the most capable models (i.e., OpenAI’s Shap-E) in producing 3D geometry train functions (with many parameters) to implicitly define surfaces and volumes in a 3D space.

Implicit functions are useful for creating smooth surfaces or handling intricate details that are challenging for discrete representations like meshes. A trained model can consist of an encoder that maps patterns to an implicit function. Instead of explicitly generating sequences of vertices and faces for a mesh, *conditional* 3D models can evaluate the trained implicit functions across a continuous 3D space. As a result, the generation process has a high degree of freedom, control, and flexibility in producing high-fidelity outputs, becoming suitable for applications that require detailed and intricate 3D geometries.

Once the model’s encoder is trained to produce implicit functions, it leverages the *neural radiance fields* (NeRF) rendering technique, as part of the decoder, to construct 3D scenes. NeRF maps a pair of inputs—a 3D spatial coordinate and a 3D viewing direction—to an output consisting of an object density and RGB color via the implicit functions. To synthesize new views in a 3D scene, the NeRF method considers the viewport as a matrix of rays. Each pixel corresponding to a ray, originates from the camera position, and then extends in the viewing direction. The color of each ray and associated pixel is computed by evaluating the implicit function along the ray and integrating the results to calculate the RGB color.

Once the 3D scene is computed, *signed distance functions* (SDFs) are used to generate meshes, or wireframes of 3D objects by calculating the distance and color of any point to the nearest surface of the 3D object. Think of SDFs as a way to describe a 3D object by telling you how far away every point in space is from the object’s surface. This function gives a number for each point: if the point is inside the object, the number is negative; if it’s on the surface, the number is zero; and if it’s outside, the number is positive. The surface of the object is where all points have the number zero. SDFs help to turn this information into a 3D mesh.

Despite the use of implicit functions, the quality of outputs is still inferior to human-created 3D assets and may feel cartoonish. However, with 3D GenAI models, you can generate the initial 3D geometries to iterate over concepts and refine 3D assets quickly.

### OpenAI Shap-E

*Shap-E* (developed by OpenAI) is an open source model “conditioned” on input 3D data (descriptions, parameters, partial geometries, colors, etc.) to generate specific 3D shapes. You can use Shap-E to create an image or text-to-3D services.

As usual, you start by downloading and loading the model from Hugging Face, as shown in [Example 3-13](#loading_shap-e).

##### Example 3-13\. Downloading and loading OpenAI’s Shap-E model

```py
# models.py
import torch
from diffusers import ShapEPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_3d_model() -> ShapEPipeline:
    pipe = ShapEPipeline.from_pretrained("openai/shap-e", device=device)
    return pipe

def generate_3d_geometry(
    pipe: ShapEPipeline, prompt: str, num_inference_steps: int
):
    images = pipe(
        prompt, ![1](assets/1.png)
        guidance_scale=15.0, ![2](assets/2.png)
        num_inference_steps=num_inference_steps, ![3](assets/3.png)
        output_type="mesh", ![4](assets/4.png)
    ).images[0] ![5](assets/5.png)
    return images
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO13-1)

This specific Shap-E pipeline accepts textual prompts, but if you want to pass image prompts, you need to load a different pipeline.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO13-2)

Use the `guidance_scale` parameter to fine-tune the generation process to better match the prompt.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO13-3)

Use the `num_inference_steps` parameter to control the output resolution in exchange for additional computation. Requesting a higher number of inference steps or increasing the guidance scale can elongate the rendering time in exchange for higher-quality outputs that better follow the user’s request.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO13-4)

Set the `output_type` parameter to produce `mesh` tensors as output.

[![5](assets/5.png)](#co_ai_integration_and_model_serving_CO13-5)

By default, the Shap-E pipeline will produce a sequence of images that can be combined to generate a rotating GIF animation of the object. You can export this output to either GIFs, videos, or OBJ files that can be loaded in 3D modeling tools such as Blender.

Now that you have a model loading and 3D mesh generation functions, let’s export the mesh into a buffer using [Example 3-14](#mesh_to_buffer).

###### Tip

`open3d` is an open source library for processing 3D data such as point clouds, meshes, and color images with depth information (i.e., RGB-D images). You will need to install `open3d` to run [Example 3-14](#mesh_to_buffer):

```py
$ pip install open3d
```

##### Example 3-14\. Exporting a 3D tensor mesh to a Wavefront OBJ buffer

```py
# utils.py

import os
import tempfile
from io import BytesIO
from pathlib import Path
import open3d as o3d
import torch
from diffusers.pipelines.shap_e.renderer import MeshDecoderOutput

def mesh_to_obj_buffer(mesh: MeshDecoderOutput) -> BytesIO:
    mesh_o3d = o3d.geometry.TriangleMesh() ![1](assets/1.png)
    mesh_o3d.vertices = o3d.utility.Vector3dVector(
        mesh.verts.cpu().detach().numpy() ![2](assets/2.png)
    )
    mesh_o3d.triangles = o3d.utility.Vector3iVector(
        mesh.faces.cpu().detach().numpy() ![2](assets/2.png)
    )

    if len(mesh.vertex_channels) == 3:  # You have color channels
        vert_color = torch.stack(
            [mesh.vertex_channels[channel] for channel in "RGB"], dim=1
        ) ![3](assets/3.png)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
            vert_color.cpu().detach().numpy()
        ) ![4](assets/4.png)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp:
        o3d.io.write_triangle_mesh(tmp.name, mesh_o3d, write_ascii=True)
        with open(tmp.name, "rb") as f:
            buffer = BytesIO(f.read()) ![5](assets/5.png)
        os.remove(tmp.name) ![6](assets/6.png)

    return buffer
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO14-1)

Create an Open3D triangle mesh object.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO14-2)

Convert the generated mesh from the model into an Open3D triangle mesh object. To do so, grab vertices and triangles from the generated 3D mesh by moving the mesh vertices and faces tensors to the CPU and converting them to `numpy` arrays.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO14-4)

Check if the mesh has three vertex color channels (indicating RGB color data) and stack these channels into a tensor.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO14-5)

Convert mesh color tensor to a format compatible with Open3D for setting the vertex colors of the mesh.

[![5](assets/5.png)](#co_ai_integration_and_model_serving_CO14-6)

Use a temporary file to create and return a data buffer.

[![6](assets/6.png)](#co_ai_integration_and_model_serving_CO14-7)

Windows doesn’t support `NameTemporaryFile`’s `delete=True` option. Instead, manually remove the created temporary file just before returning the in-memory buffer.

Finally, you can build the endpoints, as shown in [Example 3-15](#shap-e_endpoint).

##### Example 3-15\. Creating the 3D model-serving endpoint

```py
# main.py

from fastapi import FastAPI, status
from fastapi.responses import StreamingResponse
from models import load_3d_model, generate_3d_geometry
from utils import mesh_to_obj_buffer

...

@app.get(
    "/generate/3d",
    responses={status.HTTP_200_OK: {"content": {"model/obj": {}}}}, ![1](assets/1.png)
    response_class=StreamingResponse,
)
def serve_text_to_3d_model_controller(
    prompt: str, num_inference_steps: int = 25
):
    model = load_3d_model()
    mesh = generate_3d_geometry(model, prompt, num_inference_steps)
    response = StreamingResponse(
        mesh_to_obj_buffer(mesh), media_type="model/obj"
    )
    response.headers["Content-Disposition"] = (
        f"attachment; filename={prompt}.obj"
    ) ![2](assets/2.png)
    return response
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO15-1)

Specify the OpenAPI specification for a successful response to include `model/obj` as the media content type.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO15-2)

Indicate to clients that the content of streaming response should be treated as an attachment.

If you send a request to the `/generate/3d` endpoint, the download of the 3D object as a Wavefront OBJ file should start as soon as generation is complete.

You can import the OBJ file into any 3D modeling software such as Blender to view the 3D geometry. Using prompts such as `apple`, `car`, `phone`, and `donut` you can generate the 3D geometries shown in [Figure 3-26](#shape_e_blender).

![bgai 0326](assets/bgai_0326.png)

###### Figure 3-26\. 3D geometries of a car, apple, phone, and donut imported into Blender

If you isolate an object like the apple and enable the wireframe view, you can see all the vertices and edges that make up the apple’s mesh, represented as triangular polygons, as shown in [Figure 3-27](#shape_e_apple_wireframe).

![bgai 0327](assets/bgai_0327.png)

###### Figure 3-27\. Zooming in on the generated 3D mesh to view triangular polygons; inset: viewing the generated apple geometry mesh (including vertices and edges)

Shap-E supersedes another older model called *Point-E* that generates *point clouds* of 3D objects. This is because Shap-E, compared to Point-E, converges faster and reaches comparable or better generation shape quality despite modeling a higher-dimensional, multirepresentation output space.

Point clouds (often used in the construction industry) are a large collection of point coordinates that closely represent a 3D object (such as a building structure) in a real-world space. Environment scanning devices including LiDAR laser scanners produce point clouds to represent objects within a 3D space at approximate measurements close to the real-world environment.

As 3D models improve, it may be possible to generate objects that closely represent their real counterparts.

# Strategies for Serving Generative AI Models

You now should feel more confident building your own endpoints that serve a variety of models from the Hugging Face model repository. We touched upon a few different models, including those that generate text, image, video, audio, and 3D shapes.

The models you used were small, so they could be loaded and used on a CPU with reasonable outputs. However, in a production scenarios, you may want to use larger models to produce higher-quality results that may run only on GPUs and require a significant amount of video random access memory (VRAM).

In addition to leveraging GPUs, you will need to pick a model-serving strategy from several options:

Be model agnostic

Load models and generate outputs on every request (useful for model swapping).

Be compute efficient

Use the FastAPI lifespan to preload models that can be reused for every request.

Be lean

Serve models externally without frameworks or work with third-party model APIs and interact with them via FastAPI.

Let’s take a look at each strategy in detail.

## Be Model Agnostic: Swap Models on Every Request

In the previous code examples, you defined the model loading and generation functions and then used them in route handler controllers. Using this serving strategy, FastAPI loads a model into RAM (or VRAM if using a GPU) and runs a generation process. Once FastAPI returns the results, the model is then unloaded from RAM. The process repeats for the next request.

As the model is unloaded after use, the memory is released to be used by another process or model. With this approach, you dynamically swap various models in a single request if processing time isn’t a concern. This means other concurrent requests must wait before the server responds to them.

When serving requests, FastAPI will queue incoming requests and process them in a first in first out (FIFO) order. This behavior will lead to long waiting times as a model needs to be loaded and unloaded every time. In most cases, this strategy is not recommended, but if you need to swap between multiple large models and you don’t have sufficient RAM, then you can adopt this strategy for prototyping. However, in production scenarios, you should never use this strategy for obvious reasons—your users will want to avoid the long wait times.

[Figure 3-28](#model_loading_on_request) shows this model service strategy.

![bgai 0329](assets/bgai_0329.png)

###### Figure 3-28\. Loading and using models on every request

If you need to use different models in each request and have limited memory, this method can work well for quickly trying things on a less powerful machine with just a few users. The trade-off is significantly slower processing time due to model swapping. However, in production scenarios, it is better to get larger RAM and use the model preloading strategy with FastAPI application lifespan.

## Be Compute Efficient: Preload Models with the FastAPI Lifespan

The most compute-efficient strategy for loading models in FastAPI is to use the application lifespan. With this approach, you load models on application startup and unload them on shutdown. During shutdown, you can also undertake any cleanup steps required, such as filesystem cleanup or logging.

The main benefit of this strategy compared to the first one mentioned is that you avoid reloading heavy models on each request. You can load a heavy model once and then make generations on every request coming using a preloaded model. As a result, you will save several minutes in processing time in exchange for a significant chunk of your RAM (or VRAM if using GPU). However, your application user experience will improve considerably due to shorter response times.

[Figure 3-29](#model_loading_lifespan) shows the model-serving strategy that uses application lifespan.

![bgai 0330](assets/bgai_0330.png)

###### Figure 3-29\. Using the FastAPI application lifespan to preload models

You can implement model preloading using the application lifespan, as shown in [Example 3-16](#model_preloading_lifespan).

##### Example 3-16\. Model preloading with application lifespan

```py
# main.py

from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, Response, status
from models import load_image_model, generate_image
from utils import img_to_bytes

models = {} ![1](assets/1.png)

@asynccontextmanager ![2](assets/2.png)
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    models["text2image"] = load_image_model() ![3](assets/3.png)

    yield ![4](assets/4.png)

    ... # Run cleanup code here

    models.clear() ![5](assets/5.png)

app = FastAPI(lifespan=lifespan) ![6](assets/6.png)

@app.get(
    "/generate/image",
    responses={status.HTTP_200_OK: {"content": {"image/png": {}}}},
    response_class=Response,
)
def serve_text_to_image_model_controller(prompt: str):
    output = generate_image(models["text2image"], prompt) ![7](assets/7.png)
    return Response(content=img_to_bytes(output), media_type="image/png")
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO16-1)

Initialize an empty mutable dictionary at the *global* application scope to hold one or multiple models.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO16-2)

Use the `asynccontextmanager` decorator to handle startup and shutdown events as part of an async context manager:

*   The context manager will run code before and after the `yield` keyword.

*   The `yield` keyword in the decorated `lifespan` function separates the startup and shutdown phases.

*   Code prior to the `yield` keyword runs at application startup before any requests are handled.

*   When you want to terminate the application, FastAPI will run the code after the `yield` keyword as part of the shutdown phase.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO16-3)

Preload the model on startup onto the `models` dictionary.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO16-4)

Start handling requests as the startup phase is now finished.

[![5](assets/5.png)](#co_ai_integration_and_model_serving_CO16-5)

Clear the model on application shutdown.

[![6](assets/6.png)](#co_ai_integration_and_model_serving_CO16-6)

Create the FastAPI server and pass it the lifespan function to use.

[![7](assets/7.png)](#co_ai_integration_and_model_serving_CO16-7)

Pass the global preloaded model instance to the generation function.

If you start the application now, you should immediately see model pipelines being loaded onto memory. Before you applied these changes, the model pipelines used to load only when you made your first request.

###### Warning

You can preload more than one model into memory using the lifespan model-serving strategy, but this isn’t practical with large GenAI models. Generative models can be resource hungry, and in most cases you’ll need GPUs to speed up the generation process. The most powerful consumer GPUs ship with only 24 GB of VRAM. Some models require 18 GB of memory to perform inference, so try to deploy models on separate application instances and GPUs instead.

## Be Lean: Serve Models Externally

Another strategy to serve GenAI models is to package them as external services via other tools. You can then use your FastAPI application as the logical layer between your client and the external model server. In this logical layer, you can handle coordination between models, communication with APIs, management of users, security measures, monitoring activities, content filtering, enhancing prompts, or any other required logic.

### Cloud providers

Cloud providers are constantly innovating serverless and dedicated compute solutions that you can use to serve your models externally. For instance, Azure Machine Learning Studio now provides a PromptFlow tool that you can use to deploy and customize OpenAI or open source language models. Upon deployment, you will receive a model endpoint run on your Azure compute ready for usage. However, there is a steep learning curve in using PromptFlow or similar tools as they may require particular dependencies and nontraditional steps to be followed.

### BentoML

Another great contender for serving models external to FastAPI is BentoML. BentoML is inspired by FastAPI but implements a different serving strategy, purpose built for AI models.

A huge improvement over FastAPI for handling concurrent model requests is BentoML’s ability to run different requests on different worker processes. It can parallelize CPU-bound requests without you having to directly deal with Python multiprocessing. On top of this, BentoML can also batch model inferences such that the generation process for multiple users can be done with a single model call.

I covered BentoML in detail in [Chapter 2](ch02.html#ch02).

###### Tip

To run BentoML, you will need to install a few dependencies first:

```py
$ pip install bentoml
```

You can see how to start a BentoML server in [Example 3-18](#bentoml_usage).

##### Example 3-18\. Serving an image model with BentoML

```py
# bento.py
import bentoml
from models import load_image_model

@bentoml.service(
    resources={"cpu": "4"}, traffic={"timeout": 120}, http={"port": 5000}
) ![1](assets/1.png)
class Generate:
    def __init__(self) -> None:
        self.pipe = load_image_model()

    @bentoml.api(route="/generate/image") ![2](assets/2.png)
    def generate(self, prompt: str) -> str:
        output = self.pipe(prompt, num_inference_steps=10).images[0]
        return output
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO17-1)

Declare a BentoML service with four allocated CPUs. The service should time out in 120 seconds if the model doesn’t generate in time and should run from port `5000`.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO17-2)

Declare an API controller for undertaking the core model generation process. This controller will hook to BentoML’s API route handler.

You can then run the BentoML service locally:

```py
$ bentoml serve service:Generate
```

Your FastAPI server can now become a client with the model being served externally. You can now make HTTP `POST` requests from within FastAPI to get a response, as shown in [Example 3-19](#fastapi_bentoml_usage).

##### Example 3-19\. BentoML endpoints via FastAPI

```py
# main.py

import httpx
from fastapi import FastAPI, Response

app = FastAPI()

@app.get(
    "/generate/bentoml/image",
    responses={status.HTTP_200_OK: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def serve_bentoml_text_to_image_controller(prompt: str):
    async with httpx.AsyncClient() as client: ![1](assets/1.png)
        response = await client.post(
            "http://localhost:5000/generate", json={"prompt": prompt}
        ) ![2](assets/2.png)
    return Response(content=response.content, media_type="image/png")
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO18-1)

Create an asynchronous HTTP client using the `httpx` library.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO18-2)

Send a `POST` request to the BentoML image generation model endpoint.

### Model providers

Aside from BentoML and cloud providers, you can also use external model service providers such as OpenAI. In this case, your FastAPI application becomes a service wrapper over OpenAI’s API.

Luckily, integrating with model provider APIs such as OpenAI is quite straightforward, as shown in [Example 3-20](#openai_usage).

###### Tip

To run [Example 3-20](#openai_usage), you must get an API key and set the `OPENAI_API_KEY` environment variable to this key, as recommended by OpenAI.

##### Example 3-20\. Integrating with OpenAI service

```py
# main.py

from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
openai_client = OpenAI()
system_prompt = "You are a helpful assistant."

@app.get("/generate/openai/text")
def serve_openai_language_model_controller(prompt: str) -> str | None:
    response = openai_client.chat.completions.create( ![1](assets/1.png)
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO19-1)

Use the `gpt-4o` model to chat with the model via the OpenAI API.

And now you should be able to get outputs via external calls to the OpenAI service.

When using external services, be mindful that data will be shared with third-party service providers. In this case, you may prefer self-hosted solutions if you value data privacy and security. With self-hosting, the trade-off will be an increased complexity in deploying and managing your own model servers.

If you really want to avoid serving large models yourself, cloud providers can provide managed solutions where your data is never shared with third parties. An example is Azure OpenAI, which at the time of writing provides snapshots of OpenAI’s best LLMs and image generator.

You now have a few options for model serving. One final system to implement before we wrap up this chapter is logging and monitoring of the service.

# The Role of Middleware in Service Monitoring

You can implement a simple monitoring tool where prompts and responses can be logged alongside their request and response token usage. To implement the logging system, you can write a few logging functions inside your model-serving controller. However, if you have multiple models and endpoints, you may benefit from leveraging the FastAPI middleware mechanism.

Middleware is an essential block of code that runs before and after a request is processed by any of your controllers. You can define custom middleware that you then attach to any API route handlers. Once the requests reach the route handlers, the middleware acts as an intermediary, processing the requests and responses between the client and server controller.

Excellent uses cases for middleware include logging and monitoring, rate limiting, content filtering, and cross-origin resource sharing (CORS) implementations.

[Example 3-22](#middleware_monitoring_example) shows how you can monitor your model-serving handlers.

# Usage logging via custom middleware in production

Don’t use [Example 3-22](#middleware_monitoring_example) in production as the monitoring logs can disappear if you run the application from a Docker container or a host machine that can be deleted or restarted without a mounted persistent volume or logging to a database.

In [Chapter 7](ch07.html#ch07), you will integrate the monitoring system with a database to persist logs outside the application environment.

##### Example 3-22\. Using middleware mechanisms to capture service usage logs

```py
# main.py

import csv
import time
from datetime import datetime, timezone
from uuid import uuid4
from typing import Awaitable, Callable
from fastapi import FastAPI, Request, Response

# preload model with a lifespan
...

app = FastAPI(lifespan=lifespan)

csv_header = [
    "Request ID", "Datetime", "Endpoint Triggered", "Client IP Address",
    "Response Time", "Status Code", "Successful"
]

@app.middleware("http") ![1](assets/1.png)
async def monitor_service(
    req: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response: ![2](assets/2.png)
    request_id = uuid4().hex ![3](assets/3.png)
    request_datetime = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    response: Response = await call_next(req)
    response_time = round(time.perf_counter() - start_time, 4) ![4](assets/4.png)
    response.headers["X-Response-Time"] = str(response_time)
    response.headers["X-API-Request-ID"] = request_id ![5](assets/5.png)
    with open("usage.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(csv_header)
        writer.writerow( ![6](assets/6.png)
            [
                request_id,
                request_datetime,
                req.url,
                req.client.host,
                response_time,
                response.status_code,
                response.status_code < 400,
            ]
        )
    return response

# Usage Log Example

""""
Request ID: 3d15d3d9b7124cc9be7eb690fc4c9bd5
Datetime: 2024-03-07T16:41:58.895091
Endpoint triggered: http://localhost:8000/generate/text
Client IP Address: 127.0.0.1
Processing time: 26.7210 seconds
Status Code: 200
Successful: True
"""

# model-serving handlers
...
```

[![1](assets/1.png)](#co_ai_integration_and_model_serving_CO21-1)

Declare a function decorated by the FastAPI HTTP middleware mechanism. The function must receive the `Request` object and `call_next` callback function to be considered valid `http` middleware.

[![2](assets/2.png)](#co_ai_integration_and_model_serving_CO21-2)

Pass the request to the route handler to process the response.

[![3](assets/3.png)](#co_ai_integration_and_model_serving_CO21-3)

Generate a request ID for tracking all incoming requests even if an error is raised in `call_next` during request processing.

[![4](assets/4.png)](#co_ai_integration_and_model_serving_CO21-4)

Calculate the response duration to four decimal places.

[![5](assets/5.png)](#co_ai_integration_and_model_serving_CO21-5)

Set custom response headers for the processing time and request ID.

[![6](assets/6.png)](#co_ai_integration_and_model_serving_CO21-6)

Log the URL of the endpoint triggered, request datetime and ID, client IP address, response processing time, and status code into a CSV file on disk in `append` mode.

In this section, you captured information about endpoint usage including processing time, status code, endpoint path, and client IP.

Middleware is a powerful system for executing blocks of code before requests are passed to the route handlers and before responses are sent to the user. You saw an example of how middleware can be used to log model usage for any model-serving endpoint.

# Accessing request and response bodies in middleware

If you need to track interactions with your models, including prompts and the content they generate, using middleware for logging is more efficient than adding individual loggers to each handler. However, you should take into account data privacy and performance concerns when logging request and response bodies as the user could submit sensitive or large data to your service, which will require careful handling.

# Summary

We covered a lot of concepts in this chapter, so let’s quickly review everything we’ve discussed.

You saw how you can download, integrate, and serve a variety of open source GenAI models from the Hugging Face repository in a simple UI using the Streamlit package, within a few lines of code. You also reviewed several types of models and how to serve them via FastAPI endpoints. The models you experimented with were text, image, audio, video, and 3D-based, and you saw how they process data. You also learned the model architectures and the underlying mechanisms powering these models.

Then, you reviewed several different model-serving strategies including model swapping on request, model preloading, and finally model serving outside the FastAPI application using other frameworks such as BentoML or using third-party APIs.

Next, you noticed that the larger models could take some time to generate responses. Finally, you implemented a service monitoring mechanism for your models that leverage the FastAPI middleware system for every model-serving endpoint. You then wrote the logs to disk for future analysis.

You should now feel more confident building your own GenAI services powered by a variety of open source models.

In the next chapter, you will learn more about type safety and its role in eliminating application bugs and reducing uncertainty when working with external APIs and services. You will also see how to validate requests and response schemas to make your services even more reliable.

# Additional References

*   [“Bark”](https://oreil.ly/HKT8O), in “Transformers” documentation, *Hugging Face*, accessed on 26 March 2024.

*   Borsos, Z., et al. (2022). [“AudioLM: A Language Modeling Approach to Audio Generation”](https://oreil.ly/8YZBr). arXiv preprint arXiv:2209.03143.

*   Brooks, T., et al. (2024). [“Video Generation Models as World Simulators”](https://oreil.ly/52duF). OpenAI.

*   Défossez, A., et al. (2022). [“High-Fidelity Neural Audio Compression”](https://oreil.ly/p4_-5). arXiv preprint arXiv:2210.13438.

*   Jun, H. & Nichol, A. (2023). [“Shap-E: Generating Conditional 3D Implicit Functions”](https://oreil.ly/LzLy0). arXiv preprint arXiv:2305.02463.

*   Kim, B.-K., et al. (2023). [“BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion”](https://oreil.ly/uErOQ). arXiv preprint arXiv:2305.15798.

*   Liu, Y., et al. (2024). [“Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models”](https://oreil.ly/Zr6bJ). arXiv preprint arXiv:2402.17177.

*   Mildenhall, B., et al. (2020). [“NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis”](https://oreil.ly/hBiBV). arXiv preprint arXiv:2003.08934.

*   Nichol, A., et al. (2022). [“Point-E: A System for Generating 3D Point Clouds from Complex Prompts”](https://oreil.ly/FW-wT). arXiv preprint arXiv:2212.08751.

*   Vaswani, A., et al. (2017). [“Attention Is All You Need”](https://oreil.ly/N4MkH). arXiv preprint arXiv:1706.03762.

*   Wang, C., et al. (2023). [“Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers”](https://oreil.ly/h1D0e). arXiv preprint arXiv:2301.02111.

*   Zhang, P., et al. (2024). [“TinyLlama: An Open-Source Small Language Model”](https://oreil.ly/Idi1B). arXiv preprint arXiv:2401.02385.

^([1](ch03.html#id630-marker)) Hugging Face provides access to a wide range of pretrained machine learning models, datasets, and applications.

^([2](ch03.html#id631-marker)) A. Vaswani et al. (2017), [“Attention Is All You Need”](https://oreil.ly/sO33r), arXiv preprint arXiv:1706.03762.

^([3](ch03.html#id635-marker)) A great tool for visualizing attention maps is [BertViz](https://oreil.ly/e2Q7X).

^([4](ch03.html#id637-marker)) You can find the up-to-date list of open source LLMs on the [Open LLM GitHub repository](https://oreil.ly/GZaEr).

^([5](ch03.html#id647-marker)) An embedding model or an embedding layer such as in a transformer

^([6](ch03.html#id658-marker)) This sequential token generation process can also limit scalability for long sequences, as each token relies on the previous one.

^([7](ch03.html#id665-marker)) The [Hugging Face model repository](https://huggingface.co) is a resource for AI developers to publish and share their pretrained models.

^([8](ch03.html#id666-marker)) See the [Pytorch documentation](https://pytorch.org) for installation instructions.

^([9](ch03.html#id667-marker)) `float16` tensor precision is more memory efficient in memory constraint environments. The computations can be faster but precision is lower compared to `float32` tensor types. See the [TinyLlama model card](https://oreil.ly/rsmoB) for more information.

^([10](ch03.html#id668-marker)) As we saw in [Chapter 2](ch02.html#ch02), controllers are functions that handle an API route’s incoming requests and return responses to the client via a logical execution of services or providers.

^([11](ch03.html#id674-marker)) Streamlit collects usage statistics by default, but you can turn this off using a [configuration file](https://oreil.ly/m_Jix).

^([12](ch03.html#id686-marker)) The latent space of a trained model when visualized may look like white noise but will contain structured representations that the model has learned to encode and decode.

^([13](ch03.html#id697-marker)) *Multiplexing* is the process of combining multiple streams (such as audio, video, and subtitles) into a single file or stream in a synchronized manner.

^([14](ch03.html#id698-marker)) The `python-multipart` library is used for parsing `multipart/form-data`, which is commonly used encoding in file upload form submissions.