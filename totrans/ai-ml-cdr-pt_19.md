# Chapter 18\. Introduction to RAG

Remember that first time you chatted with an LLM like ChatGPT—and how it was extremely insightful about things you didn’t expect it to know? I had worked with LLMs before the release of ChatGPT and on projects that highlighted LLM abilities, and I *still* was surprised by what they could do. Remember the famous on-stage demonstration by Google, where the CEO had a conversation with the planet Pluto? It was one of those fundamental mind shifts in the possibilities of AI that we’re *still* exploring as it continues to evolve.

But, despite all that brilliance, there were still limitations, and the more I and others worked with LLMs, the more we encountered them. The transformer-based architecture that we discussed in [Chapter 15](ch15.html#ch15_transformers_and_transformers_1748549808974580) was brilliant at snarfing up text data, creating QKV mappings from it, and learning how to artificially understand the semantics of the text as a result. But despite the volume of text used to build those mappings, there was—and always is—one blind spot: private data. In particular, if there is data that you want to work with that the model was not trained on, you’re at a major risk of hallucination!

Gaining skills to help mitigate this blind spot could potentially be the *most* valuable thing you can do as a software developer.

For this chapter, I want you to think about AI models and in particular large generative models like LLMs *differently*. Stop seeing them as intelligent and knowledgeable and start seeing them as *utilities* to help you parse your data better. Think of everything they have learned not as a knowledge base in and of itself but as a way that they have generalized understanding of language by being extensively well read.

I call this *artificial understanding,* as a complementary technology to AI.

Then, once you treat your favorite LLM as an engine for artificial understanding, you can start having it understand your private text—stuff that wasn’t in its training set—and through that understanding, process your text in new and interesting ways.

Let’s explore this with a scenario. Imagine you’re discussing your favorite sci-fi novel with an AI model. You want to ask about characters, plot, theme, and stuff like that, but the model struggles with the specifics, offering only general responses—or worse, hallucinating them. For example, take a look at [Figure 18-1](#ch18_figure_1_1748550073457538), which shows the results I got when I was chatting with ChatGPT about a character from a novel called *Space Cadets.*

![](assets/aiml_1801.png)

###### Figure 18-1\. Chatting with GPT about a character

This is all very interesting—except that it’s wrong. First of all, the character is from *North* Korea, not *South* Korea.

GPT is being confidently incorrect. Why? Because this novel isn’t in the training set! I wrote it in 2014, and it was published by a small press that folded just a few months afterward. As such, it’s relatively obscure and the perfect fodder for us to use to explore RAG. By the end of this chapter, you’ll have used your PyTorch skills to create an application that is much smarter at understanding this novel and, indeed, the character in question. And yes, you’ll have the full novel to work with!

A small aside: when I first used an LLM for tasks like this, my mind was blown. Its ability to *artificially understand* the contents and context of my own writing was like having a partner beside me to critique my work and to help me dig deep into the characters and themes. The book ends on a cliffhanger, and I never came back to write any sequels. Having conversations with an LLM about the character arcs, etc., gave me a whole new fount of wisdom about where it could go.

And of course, you aren’t limited to works of fiction. Almost every business has a trove of internal intelligence that’s locked up in documents that would take a human too much time to read, index, cross-correlate, and understand to be able to answer queries—so the ability of an LLM to artificially understand them to help you mine the text for knowledge is second-to-none.

That’s why I’m excited about RAG. And I hope you will be, too, after you finish this chapter.

# What Is RAG?

The acronym *RAG* stands for *retrieval augmented generation*, which works to bridge the knowledge gap between what an LLM has been trained on and private data you own that it doesn’t have mappings for. At query time, as well as with a prompt like “Tell me about the character…,” we’ll also feed it information snippets from the local datastore. So, for example, if we’re querying about a character from a novel, local data might include things like her hometown, her favorite food, her values, and how she speaks. When we pass *that* data along with the query, a lot of it *is* in the training set for the LLM, and as such, the LLM can have a much more informed opinion about her. Not least, the mistake the LLM made in [Figure 18-1](#ch18_figure_1_1748550073457538) can be mitigated—when the LLM is given her hometown, it can at least get the country right!

[Figure 18-2](#ch18_figure_2_1748550073457600) shows the flow of a typical query to an LLM. It’s quite basic: you pass in a prompt and the transformers do their magic by going through the knowledge that the LLM learned to produce QKV values to generate a response.

![](assets/aiml_1802.png)

###### Figure 18-2\. Typical flow of a query to an LLM

As we’ve demonstrated , if the LLM doesn’t have much knowledge of the specifics, it will fill in the gaps—and it does a pretty good job. For example, even though it got her nationality wrong in the example shown in [Figure 18-1](#ch18_figure_1_1748550073457538), it was at least able to infer that her name is Korean!

With RAG, we change this flow to augment the query with extra information that we bundle in (see [Figure 18-3](#ch18_figure_3_1748550073457629)). We do this by having a local database of the content of the book, and then we search that for things that are *similar* to the query. You’ll see the details of how that works shortly.

![](assets/aiml_1803.png)

###### Figure 18-3\. Typical flow of a RAG query with an LLM

The goal here is to enhance the initial prompt with a lot of additional context. So, scenes in the book might have her mention her hometown, her family history, favorite foods, why she likes people or things, etc. When that is passed to the LLM along with the query, the LLM has a lot more to work with—including things that it *has* learned about, so its interpretation of the character becomes a lot more intelligent. It therefore *artificially understands* the content better.

The key to all of this, of course, is in being able to retrieve the best information to bundle with the prompt to make the most of the LLM. You can achieve this by storing content from the source material (in this case, the book) in a way that lets you do searches for things that are semantically relevant. To that end, you’ll use a vector store. We’ll explore that next.

# Getting Started with RAG

To get started, let’s first explore how to create a vector database. To do this, you’ll use a database engine that supports vectors and similarity search.

These work with the idea of storing text as vectors that represent it by using embeddings. We saw these in action in [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888). For simplicity, you’ll start by using a pre-built, pre-learned set of embeddings from OpenAI with an API provided by LangChain. These will be combined with a vector store database called Chroma that is free and open source.

Let’s include the following imports:

```py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
```

The `PyPDFLoader`, as its name suggests, is used for managing PDF files in Python. I’m providing the book as a PDF, so we’ll need this.

The `RecursiveCharacterTextSplitter` is a really useful class for slicing the book up into text chunks. It provides flexibility on the size of the chunk and the overlap between chunks. We’ll explore that in detail a little later.

The `OpenAIEmbeddings` class gives us access to the embeddings learned by Open AI while training GPT, and it’s a nice shortcut to make things quicker for us. We don’t need to learn our own embeddings for this application—as long as our text is encoded in a set of embeddings and our prompt uses the same ones, we can use them for similarity search. There are lots of options for this, and Hugging Face is a great repository where you can look for the latest and greatest.

Finally, the `Chroma` database provides us with the ability to store and search text based on similarity.

## Understanding Similarity

We’ve mentioned similarity a few times now, and it’s important for you to understand where it can be useful for you. Recall that in [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888), we discussed how embeddings can be used to turn words into vectors. A simple representation of this is shown in [Figure 18-4](#ch18_figure_4_1748550073457653).

![](assets/aiml_1804.png)

###### Figure 18-4\. Words as vectors

Here, we plot the words *Awesome*, *Great*, and *Terrible* based on their learned vectors. It’s an oversimplification in two dimensions, but hopefully it’s enough to demonstrate the concept. In this case, we can visualize that *Awesome* and *Great* are similar because they’re close to each other, but we can quantify that by looking at the angle of the vectors between them. Taking a function of that angle, like its *cosine,* can give us a great indication of how close the vectors are to each other. Similarly, if we look at the word *Terrible*, the angle between *Awesome* and *Terrible* is very large, indicating that the two words aren’t similar.

This process is called *cosine similarity*, and we’ll be using it as we create our RAG. We’ll split the book into chunks, calculate the embedding for those chunks, and store them in the database. Then, by using a store (ChromaDB, in this case) that provides a search based on cosine similarity, we’ll have the key to our RAG.

There are many different ways to calculate similarity, with cosine similarity being one of them. It’s worth looking into these other ways to fine-tune your RAG solution, but for the rest of this chapter, I’ll use cosine similarity because of its simplicity.

## Creating the Database

To create the vector store, we’ll go through the process of loading the PDF file, splitting it into chunks, calculating the chunks’ embeddings, and then storing them. Let’s look at this step-by-step.

First, we’ll load the PDF file by using `PyPDFLoader`:

```py
# Load the PDF file
loader = PyPDFLoader(pdf_path)
documents = loader.load()
```

Next, we’ll set up a text splitter that reads what we’ll use to chunk the text. An important part of your application will be establishing the appropriate sizes of chunks:

```py
# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)
```

In this case, the code will split the text into chunks of one thousand characters. But it uses a recursive strategy to calculate the split, in which it tries to do it on the natural boundaries in the text, rather than making hard cuts at exactly one thousand characters. It tries to split on newlines first, then on sentences, then on punctuation, and then on spaces. As a last resort, it will split in the middle of a word.

The overlap means that the next chunk won’t start at the immediate next character but around two hundred characters back. If we have these overlaps, some text will be included twice in the data—and that’s OK. It means that we won’t lose content by splitting in the middle of a sentence, etc. You should explore the size of the chunk and overlap based on what suits your scenario. Larger chunks like this will be faster to search because there will be fewer chunks than if they were smaller, but it also lowers the likelihood of the chunks being very similar to your prompt if the prompt is shorter than the chunk size.

The splitter provides the ability for you to specify your own length function if you want to measure length differently. In this case, I’m just using Python’s default `len` function. Typically, for a RAG like this, you may not need to override the `len` function, but the idea is that different models and encoders may count tokens in different ways. For example, GPT 3.5 recognizes a phrase like `lol` as a single token, but an emoji can be four tokens.

The `add_start_index` parameter adds metadata to each chunk, indicating where it was located in the original text. This is useful for debugging, in which you can trace back where each chunk came from or provide things like citations.

Once you’ve specified the text, you can use it to split the PDF into multiple texts:

```py
texts = text_splitter.split_documents(documents)
```

Now that you have the texts, you can turn them into embeddings by using the `OpenAIEmbeddings` class, and you can also specify that you want a vector store using Chroma by passing it the documents:

```py
# Initialize OpenAI embeddings
# Make sure to set your OPENAI_API_KEY environment variable
embeddings = OpenAIEmbeddings()

# Create and persist the vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory
)
```

As shown, you then simply pass the texts and embeddings you specified and a directory to store the embeddings. Then save the vector store to disk with this:

```py
# Persist the vector store
vectorstore.persist()
```

###### Note

The `OpenAIEmbeddings` requires an `OPENAI_API_KEY` environment variable. You can get one at the [Open AIPlatform website](https://oreil.ly/41hwI) and then follow the instructions for your operating system by setting one. Make sure you name it exactly as shown.

The underlying database is an SQLite3 one (see [Figure 18-5](#ch18_figure_5_1748550073457676)).

![](assets/aiml_1805.png)

###### Figure 18-5\. The directory containing the ChromaDB content

This gives you the ability to browse and inspect the database by using any tools that work with SQLite. So, for example, you can use the free [DB Browser for SQLite](https://sqlitebrowser.org) to access the data (see [Figure 18-6](#ch18_figure_6_1748550073457698)).

![](assets/aiml_1806.png)

###### Figure 18-6\. Browsing data in the SQLite browser

Now that we have the vector store, let’s explore what happens when we want to search it for similar text.

## Performing a Similarity Search

Once you have the vector store set up, it’s easy to search it.

Here’s a function you can use to perform a similarity search with the vector store:

```py
def search_vectorstore(vectorstore, query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return results
```

As you can see, it’s pretty straightforward! You can override or extend some of the functionality if you like with optional parameters, including the following:

Search_type

This defaults to `similarity` but can also be `mmr` for *maximum marginal relevance* (MMR), which is worth experimenting with as you build out production systems. MMR is particularly useful when you want to avoid redundant results.

Distance_metric

This defaults to `cosine`, as we saw earlier, but it can also be `l2`, which is the *distance*—effectively, the straight-line distance between the two vectors in the embedding space. Alternatively, it can be `ip` for *inner product*, which provides a very fast calculation but at the cost of lower accuracy.

Lambda_mult

This is an optional value between 0 and 1 that you use to control the strictness of the distance measurement. A value of 1.0 will give highly relevant scores, and a value of 0.0 will give much more diverse scores.

As you build systems, I recommend that you try multiple approaches to see which works best for your scenario.

## Putting It All Together

Now, you can use code like the following to take your PDF, slice and store it as vectors in the store, and run a query against it:

```py
# Path to your PDF file
pdf_path = "space-cadets-2020-master.pdf"

# Create the vector store
vectorstore = create_vectorstore(pdf_path)

# Example search
query = "Give me some details about Soo-Kyung Kim. 
         Where is she from, what does she like, tell me all about her?"
results = search_vectorstore(vectorstore, query, 5)
```

When running this, I got detailed results about her character. Here are some snippets:

```py
“I think we are going to be good friends,” said Soo-Kyung. “I like how
you are straightforward. I am too, but that intimidates some people.”
“So where are you from?”
“I am from a small village called Sijungho,” continued Soo -
Kyung. “There’s not much to see there.”
“Sounds Korean,” said Aisha. “You from South Korea?”
“North Korea,” corrected Soo -Kyung. “I’ve never even been to
South Korea.”

```

So, when we’re making a query about the character to an LLM, we have all this extra content. We’ll explore that next.

# Using RAG Content with an LLM

Now that you’ve created a vector store and stored the book in it, let’s explore how you would read snippets back from the store, add them to a prompt, and get data back. We’ll use a local Ollama server to keep things simple. For more on Ollama, see [Chapter 17](ch17.html#ch17_serving_llms_with_ollama_1748550058915113).

First, let’s load the vector store that we created in the previous step:

```py
def load_vectorstore(persist_directory="./chroma_db"):
    embeddings = OpenAIEmbeddings()

    # Load existing vector store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vectorstore
```

You *must* use the same embeddings as those you used when you created the vector store. Otherwise, there will be a mismatch when you try to encode your prompt and search for stuff similar to it.

In this case, I’m using the OpenAIEmbeddings, but it’s entirely up to you how to approach this. There are many embeddings available in open source on Hugging Face, or you could use things like the GLoVE embeddings we explored in [Chapter 6](ch06.html#ch06_clean_making_sentiment_programmable_by_using_embeddings_1748752380728888).

ChromaDB persisted the embeddings in an SQLite database at a specific directory. Make sure you embed that, and then all you have to do is pass this and your embedding function to Chroma to get a reference to your database.

To search the vector store, you’ll use the same code as earlier:

```py
def search_vectorstore(vectorstore, query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return results
```

Next, input a query. For example, input this:

```py
query = "Please tell me all about Soo-Kyung Kim."
```

At this point, you have all the pieces you need to do a RAG query, which you can do like this:

```py
# Example query
query = "Please tell me all about Soo-Kyung Kim."

# Perform RAG query
answer, sources = rag_query(vectorstore, query, num_contexts=10)
```

Here, you create a helper function that will pass the query and the vector store, and you also have a parameter with the number of items to find in the vector store. The app will return the answer (from the LLM) as well as a list of sources from the data that it used to augment the query.

Let’s explore this function in depth:

```py
def rag_query(vectorstore, query, num_contexts=3):
    # Retrieve relevant documents
    relevant_docs = search_vectorstore(vectorstore, query, k=num_contexts)

    # Combine context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Generate response using Ollama
    response = query_ollama(query, context)

    return response, relevant_docs
```

You’ll start by searching the vector store with the code provided earlier. This will give you the decoded chunks from the datastore as strings, and you should call these `relevant_docs`.

You’ll then create the context string by joining the chunks together with some new line characters to separate them. It’s as simple as that.

Now, the query and the context will be used in a call to Ollama. Let’s see how that will work.

Start by defining the function:

```py
def query_ollama(prompt, context, model="llama3.1:latest", temperature=0.7):

    ollama_url = "http://localhost:11434/api/chat"
```

Here, you can set the function to accept the prompt and context. I’ve added a couple of optional parameters that, if they’re not set, will use the defaults. The first is the model. To get a list of available models on your server, you can just use “ollama list” from the command prompt. The `temperature` parameter indicates how deterministic your response will be: the smaller the number, the more deterministic the answer, and the higher the number, the more creative the answer. I set a default of 0.7, which gives some flexibility to the model to make it natural sounding while staying relevant. But when you use smaller models in Ollama (like `llama3.1`, as shown), it does make hallucination more likely.

You’ll also want to specify the `ollama_url` endpoint, as shown in [Chapter 17](ch17.html#ch17_serving_llms_with_ollama_1748550058915113).

Next, you create the messages that will be used to interact with the model.

The structure of conversations with a model typically looks like the one in [Figure 18-7](#ch18_figure_7_1748550073457720). The model will optionally be primed with a system message that gives it instructions on how to behave. It will then have an initial message that it emits to the user, like, “Welcome to the Chat. How can I help?” The user will then respond with a prompt asking the model to do something, to which the model will respond, and so on.

![](assets/aiml_1807.png)

###### Figure 18-7\. Anatomy of a conversation with a model

The *memory* of the conversation will be a JSON document with each of the roles prefixed by a `role` value. The initial message will have the `system` role, the model messages will have the `model` role, and the user messages will have the `user` role.

So, for the simple RAG app we’re creating, we can create an instance of a conversation like this—passing the system message and the user message, which will be composed of the prompt and the context, like this:

```py
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. 
                        Use the provided context to answer questions. 
                        If you cannot find the answer in the context, say so. 
                        Only use information from the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {prompt}"
        }
    ]
```

Depending on how you set up the system role, you’ll get very different behavior. In this case, I used a prompt that gets it to heavily focus on the provided context. You don’t *need* to do this, and by working with this prompt, you might get much better results.

Within the user role, this is just as simple as creating a string with `Context:` and `Question:` content that you paste the context and prompt into.

From this, you can now create a JSON payload to pass to Ollama that contains the desired model, the messages, the temperature, and the stream (which must be set to `False` if you want to get a single answer back):

```py
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature
    }
```

Note also that the desired model must be installed in Ollama or you’ll get an error, so see [Chapter 17](ch17.html#ch17_serving_llms_with_ollama_1748550058915113) for adding models to Ollama.

Then, you simply have to use an HTTP post to the Ollama URL, passing it the payload. When you get the response, you can query the returned message—where there’ll now be new content added by the model. This content will contain your answer!

```py
    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama: {str(e)}"
```

In this case, I used Llama 3.1 and got some excellent answers. Here’s an example:

```py
Based on the provided context, here's what can be gathered about Soo-Kyung Kim:

1. She is from North Korea.
2. She has been trained in various skills, including science, technology, 
martial arts, languages, piloting, and strategy.
3. Her family name "Kim" is significant, as it is the name of the ruling family 
   of the Democratic People's Republic of Korea (North Korea).
4. Soo-Kyung's presence on the space academy may be related to her exceptional 
   abilities, but there is also a suggestion that she was chosen for other 
   reasons.
…
```

Your results will vary, based on the temperature, the slicing size for the chunks, and various other factors.

One thing to note is that you can also use a *really* small model like Gemma2b and still get really good results. However, the context window of a model this small could have issues when you’re retrieving and augmenting your query with lots of information. As you saw earlier in this chapter, we were using one-thousand-character chunks, and we’re retrieving the 10 closest ones to the prompt. This is already in order of 10 k characters, and depending on the tokenization strategy, that could be more than 10 k tokens. Given that the context window for that model is only 2 k tokens, you could hit a problem. Watch out for that!

## Extending to Hosted Models

In the example we just walked through, we used smaller models like Llama and Gemma to perform RAG on a local Ollama server. If you want to use larger, hosted models like GPT, the process is exactly the same. One change I would make, though, is with the system prompt. Given that these models have huge amounts of parameters that have learned a lot, it’s good to unshackle them a bit and not expect them to be limited solely to the context provided!

For example, for GPT, you can import classes that support OpenAI’s GPT models like this:

```py
from langchain_openai import ChatOpenAI
```

You can then instantiate this class like this:

```py
chat = ChatOpenAI(
    model=model,
    temperature=temperature
)
```

The model value is a string containing the name of the model you want to use. For example, you could use `gpt-3.5-turbo` or `gpt-4`. Check the [OpenAI API documentation for model versions](https://oreil.ly/SVBXr) available at the time you’re reading this.

Then, you can create the prompt very simply. First, create a prompt template to hold the system and user prompts:

```py
# Create prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. 
                Use the following context to answer questions. "
               "Please provide as much detail as possible in a comprehensive 
               answer."),
    ("system", "Context:\n{context}"),
    ("user", "{question}")
])
```

Then, you can make the formatted prompt with the details of the context and prompt:

```py
# Format the prompt with the context and question
formatted_prompt = prompt_template.format(
    context=context,
    question=prompt
)

```

Finally, you can invoke the GPT chat with the formatted prompt and get the response:

```py
# Get the response
response = chat.invoke(formatted_prompt)
return response.content
```

Now, as long as you ensure that you have an `OPENAI_API_KEY` environment variable, as discussed earlier, you’re RAGging against GPT! Please pay attention to the pricing on OpenAI for using the available models.

# Summary

In this chapter, you dipped your toes into the RAG waters, where you learned a powerful technique that enhances the capabilities of LLMs by combining their general understanding skills with local, private data. You saw how RAG works by creating a vector database with the contents of a book, and then you searched that database for information that was relevant to your given prompts.

We also explored querying a character from the book to learn more about her—and despite models like Llama and GPT not being trained on content about her, they were able to artificially understand the text and provide great information and analysis.

You also explored tools like ChromaDB (for vector storage) and pretrained embeddings (such as OpenAIs for vector encoding of text allowing similarity searches). You also explored various models that could be enhanced by using RAG, both small and local ones (like Llama and Gemma with Ollama) and large hosted models (like GPT via the OpenAI API). This took you through the process end to end: slicing text, encoding it, storing it, searching it based on similarity, and bundling it with a prompt to a model to perform RAG.

In the next chapter, we’ll shift gears a bit to another exciting aspect of AI: generative image models. We’ll explore a number of different models that provide images from text prompts, and we’ll dig down a little into how they work.