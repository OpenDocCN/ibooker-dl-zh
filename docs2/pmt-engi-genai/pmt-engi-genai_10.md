# 第十章\. 构建 AI 驱动应用程序

在本章中，你将应用提示的五个原则到内容写作的端到端 AI 工作流程中。该服务将根据用户对访谈问题的回答，以用户的写作风格撰写博客文章。这个系统最初在[Saxifrage 博客](https://oreil.ly/saxifrage)上进行了记录。

# AI 博客写作

使用 AI 创建博客写作服务的天真方法是将 ChatGPT 提示为`Write a blog post on {blogPostTopic}`。生成的内容质量可能合理，但不太可能包含关于主题的有价值意见或独特经验。内容也可能很短、很普通，因此不太可能排在 Google 搜索结果的前列。

一种更复杂的方法可能是构建一个更长的提示，并添加更多指令。可以添加关于规定的写作语气、博客文章的结构和要包含的关键词的详细信息。一个常见的博客文章[写作提示](https://oreil.ly/uMfZa)的例子可以在这里看到。

输入：

```py
Create a blog post about “{blogPostTopic}”. Write it in a “{tone}” tone.
Use transition words.
Use active voice. Write over 1000 words.
Use very creative titles for the blog post.
Add a title for each section. Ensure there are a minimum of 9 sections. Each
section should have a minimum of two paragraphs.
Include the following keywords: “{keywords}”.
Create a good slug for this post and a meta description with a maximum of 100
words and add it to the end of the blog post.
```

这个更长、更复杂的提示可能会产生更好的内容质量。然而，让我们回顾一下提示的五个原则作为检查清单：

方向

提供了一些指令，例如语气、使用过渡词和主动语态。然而，内容仍然可能听起来像 AI，而不是像用户。

格式

虽然有一些关于结构的提及，包括指定九个两段的内容，但这些指令很可能会被忽略。ChatGPT 在数学方面表现不佳，通常无法遵循指定多个部分或单词的指令。

示例

没有给出如何执行任务的示例，这可能会损害在多个主题或甚至在同一主题上多次运行此提示的可靠性。即使提供一个示例（一次性提示）也可能极大地提高质量。

评估

这是一个*盲提示*（在提示中添加指令[而不测试它们](https://oreil.ly/r7sXi)）的例子。很可能其中一些指令对质量没有影响（不必要地消耗代币），甚至可能降低质量。

分区

整个任务仅用一个提示尝试，这可能会损害性能。如果不将任务分解为子任务，就很难理解哪个部分的过程是成功的或失败的。

通过本章，你将创建多个 LLM 链组件。每个链都将使用 LangChain 实现，使其更易于维护，并便于进行监控和优化的日志记录。这个系统将帮助你基于用户的独特观点和经验生成*听起来像人*的内容。

首先准备好你的工作空间，配备必要的工具至关重要。因此，让我们将重点转向主题研究，并开始设置你的编程环境。

# 主题研究

您将需要安装几个 Python 包来有效地使用 LangChain 的文档加载器，包括以下这些：

google-searchresults

一个用于抓取和处理 Google 搜索结果的 Python 库。

pandas

这提供了操作数值表和时间序列数据的结构和操作。

html2text

此工具将 HTML 文件或网页转换为 markdown (*.md*) 文件或文本。

pytest-playwright

此包使您可以使用 Playwright 进行端到端测试。

chromadb

ChromaDB 是一个开源的向量数据库。

nest_asyncio

这扩展了 Python 标准的`asyncio`，以便与 Jupyter Notebooks 兼容。

使用此命令可以轻松安装这些包：

```py
pip install google-searchresults pandas html2text pytest-playwright chromadb \
nest_asyncio --quiet
```

此外，您将使用 LangChain 的文档加载器，这些加载器需要 Playwright。

在您的终端上输入此命令：**playwright install**。

此外，您还需要选择一个`TOPIC`并为`SERPAPI_API_KEY`和`STABILITY_API_KEY`设置环境变量。如果您在没有 Jupyter Notebook 的情况下运行脚本，那么您不需要使用任何`nest_asyncio`代码：

```py
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Custom imports:
from content_collection import collect_serp_data_and_extract_text_from_webpages
from custom_summarize_chain import create_all_summaries, DocumentSummary

import nest_asyncio
nest_asyncio.apply()

# Constant variables:
TOPIC = "Neural networks"
os.environ["SERPAPI_API_KEY"] = ""
os.environ["STABILITY_API_KEY"] = ""
```

接下来，您将专注于高效地总结网络内容：

```py
# Extract content from webpages into LangChain documents:
text_documents = await \
collect_serp_data_and_extract_text_from_webpages(TOPIC)

# LLM, text splitter + parser:
llm = ChatOpenAI(temperature=0)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1500, chunk_overlap=400
)
parser = PydanticOutputParser(pydantic_object=DocumentSummary)

summaries = await create_all_summaries(text_documents,
parser,
llm,
text_splitter)
```

首先，导入所需的工具，然后获取与您的`TOPIC`相关的网页内容。在设置好`ChatOpenAI`模型后，您将使用`text_splitter`来管理文本块。分割器确保没有片段过长，同时保持上下文重叠。然后创建`PydanticOutputParser`来处理和结构化摘要。通过将提取的文档通过一个专门的摘要函数，LLM 产生简洁的摘要。

如果您想深入了解`create_all_summaries`函数，请查看[*custom_summarize_chain.py*](https://oreil.ly/KyKjS)。

一些需要强调的关键点是，您可以在 LangChain 中的大多数类中进行子类化。例如，您可以覆盖默认的`ChromiumLoader`以使其异步：

```py
from langchain_community.document_loaders import AsyncHtmlLoader, \
AsyncChromiumLoader

class ChromiumLoader(AsyncChromiumLoader):
    async def load(self):
        raw_text = [await self.ascrape_playwright(url) for url in self.urls]
        # Return the raw documents:
        return [Document(page_content=text) for text in raw_text]

async def get_html_content_from_urls(
    df: pd.DataFrame, number_of_urls: int = 3, url_column: str = "link"
) -> List[Document]:
    # Get the HTML content of the first 3 URLs:
    urls = df[url_column].values[:number_of_urls].tolist()

    # If there is only one URL, convert it to a list:
    if isinstance(urls, str):
        urls = [urls]

    # Check for empty URLs:
    urls = [url for url in urls if url != ""]

    # Check for duplicate URLs:
    urls = list(set(urls))

    # Throw error if no URLs are found:
    if len(urls) == 0:
        raise ValueError("No URLs found!")
    # loader = AsyncHtmlLoader(urls) # Faster but might not always work.
    loader = ChromiumLoader(urls)
    docs = await loader.load()
    return docs

async def create_all_summaries(
    # ... commented out for brevity
) -> List[DocumentSummary]:
    # ... commented out for brevity
```

通过继承`ChromiumLoader`，您可以轻松创建一个自定义实现，使用 Chrome 浏览器从多个 URL 异步抓取内容。`get_html_content_from_urls`从 URL 列表中获取 HTML 内容，确保没有重复并处理潜在的错误。

# 专家访谈

现在，您已经成功从 Google 提取了前三项结果摘要，您将进行一次与 LLM 的访谈，生成相关的问题，以确保您的文章使用`InterviewChain`类具有独特的视角：

```py
from expert_interview_chain import InterviewChain
interview_chain = InterviewChain(topic=TOPIC, document_summaries=summaries)
interview_questions = interview_chain()

for question in interview_questions.questions:
    print(f"Answer the following question: {question.question}\n", flush=True)
    answer = input(f"Answer the following question: {question.question}\n")
    print('------------------------------------------')
    question.answer = answer
```

InterviewChain 实例化

拥有您的话题和获得的摘要后，创建一个`InterviewChain`实例，根据您数据独特的上下文进行定制。

生成问题

通过简单地调用`interview_chain`，您启动了从您的摘要中生成一系列探究性问题的过程。

互动问答环节

进入一个引人入胜的循环，其中每个派生问题都会打印出来，使用 `input()` 提示你回答。然后你的回答会被保存回 Pydantic 对象。

# 给出方向

给一个 LLM 独特的答案提供独特的上下文，这允许 LLM 生成更丰富、更细腻的回复，确保你的文章提供新颖且深入的视角。

`InterviewChain` 的所有代码都在 *[expert_interview_chain.py](https://oreil.ly/0d5Hi)* 文件中。它有两个重要组成部分：

一个定制的 `System` 消息

这个提示包括角色提示、之前生成的摘要、主题和格式说明（用于输出解析器）：

```py
system_message = """You are a content SEO researcher. Previously you have
summarized and extracted key points from SERP results. The insights gained
will be used to do content research and we will compare the key points,
insights and summaries across multiple articles. You are now going to
interview a content expert. You will ask them questions about the following
topic: {topic}.

You must follow the following rules:
 - Return a list of questions that you would ask a content expert about
 the topic.
 - You must ask at least and at most 5 questions.
 - You are looking for information gain and unique insights that are not
 already covered in the {document_summaries} information.
 - You must ask questions that are open-ended and not yes/no questions.
    {format_instructions} `"""`
```

```py`Output parsers      Diving deeper into the class, you encounter the `PydanticOutputParser`. This parser actively structures the LLMs responses into parsable, Pydantic `InterviewQuestions` objects:      ```从`expert_interview_chain`导入`InterviewQuestions`模块 # 设置解析器并将指令注入到提示模板中：`parser = PydanticOutputParser(pydantic_object=InterviewQuestions)` ```py    In essence, you’re orchestrating a conversation with the AI and instructing it to conceive potent questions that amplify content insights, all the while making customization a breeze.````  ```py````` # 生成大纲    包括之前的访谈和研究，你可以使用`BlogOutlineGenerator`为帖子生成大纲。`TOPIC`、`question_answers`和 Google 的`summaries`被传递以提供额外的上下文：    ```py from article_outline_generation import BlogOutlineGenerator  blog_outline_generator = BlogOutlineGenerator(topic=TOPIC, questions_and_answers=[item.dict() for item in interview_questions.questions])  questions_and_answers = blog_outline_generator.questions_and_answers outline_result = blog_outline_generator.generate_outline(summaries) ```    让我们详细探索`BlogOutlineGenerator`类：    ```py from typing import List, Any from pydantic.v1 import BaseModel  class SubHeading(BaseModel):     title: str # Each subheading should have a title.  class BlogOutline(BaseModel):     title: str     sub_headings: List[SubHeading] # An outline has many sub_headings  # Langchain libraries: from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate) from langchain.output_parsers import PydanticOutputParser from langchain_openai.chat_models import ChatOpenAI  # Custom types: from custom_summarize_chain import DocumentSummary  class BlogOutlineGenerator:     def __init__(self, topic: str, questions_and_answers: Any):         self.topic = topic         self.questions_and_answers = questions_and_answers          # Create a prompt         prompt_content = """  Based on my answers and the summary, generate an outline for a blog  article on {topic}.  topic: {topic} `document_summaries:` `{document_summaries}` ````` `---`  `以下是我在访谈中回答的问题:`         `{interview_questions_and_answers}` ```py` `---`  `Output format:` `{format_instructions}` ``` `"""`          `system_message_prompt` `=`         `SystemMessagePromptTemplate``.``from_template``(``prompt_content``)`          `self``.``chat_prompt` `=` `ChatPromptTemplate``.``from_messages``(`         `[``system_message_prompt``])`          `# 创建输出解析器`         `self``.``parser` `=` `PydanticOutputParser``(``pydantic_object``=``BlogOutline``)`          `# 设置链`         `self``.``outline_chain` `=` `self``.``chat_prompt` `|` `ChatOpenAI``()` `|` `self``.``parser`      `def` `generate_outline``(``self``,` `summaries``:` `List``[``DocumentSummary``])` `->` `Any``:`         `print``(``"正在生成大纲...``\n``---"``)`         `result` `=` `self``.``outline_chain``.``invoke``(`             `{``"topic"``:` `self``.``topic``,`             `"document_summaries"``:` `[``s``.``dict``()` `for` `s` `in` `summaries``],`             `"interview_questions_and_answers"``:` `self``.``questions_and_answers``,`             `"format_instructions"``:` `self``.``parser``.``get_format_instructions``(),`             `}`         `)`         `print``(``"大纲生成完成!``\n``---"``)`         `return` `result` ```py ```` ```py`` ```   ```py`` ````创建了一个包含`title`和`sub_headings`键的`BlogOutline` Pydantic 对象。同时，使用 LangChain 表达式语言（LCEL）设置了大纲链，该链将提示传递到聊天模型，然后最终传递到输出解析器：    ```py # Set up the chain: self.outline_chain = self.chat_prompt | ChatOpenAI() | self.parser ```    通过使用 Pydantic 输出解析器，链将返回一个`BlogOutline` Pydantic 对象，该对象将在未来的链中使用。```py` `````  ```py```` ```py``` # 文本生成    在获得摘要、访谈问题和博客文章大纲后，是时候开始生成文本了。`ContentGenerator`类结合了 SEO 专业知识与多种 LLM 技术，包括以下内容：    嵌入和检索      这有效地将原始网页分割并矢量化，存储在 Chroma 数据库中，并在编写每个部分时检索相关网页文本。      定制记忆      在制作每个博客部分时，它使用记忆来避免重复相同的信息，同时如果对话变得过长，还会总结对话。      定制上下文      LLM 包含混合信息，包括你之前的访谈洞察、之前所说的话以及来自 Google 的相关网页文本片段：      ``` from article_generation import ContentGenerator  content_gen = ContentGenerator( topic=TOPIC, outline=outline_result, questions_and_answers=questions_and_answers)  # Vectorize and store the original webpages: content_gen.split_and_vectorize_documents(text_documents) # Create the blog post: blog_post = content_gen.generate_blog_post() ```py    所有源代码都在`*[article_generation.py](https://oreil.ly/0IFyI)*`中，但让我们特别关注这个链中的三个关键组件。    `OnlyStoreAIMemory`类是`ConversationSummaryBufferMemory`的定制子类：    ``` from typing import List, Dict, Any from langchain.memory import ConversationSummaryBufferMemory  from langchain_core.messages import SystemMessage  class OnlyStoreAIMemory(ConversationSummaryBufferMemory):     def save_context(self, inputs: Dict[str, Any],     outputs: Dict[str, str]) -> None:         input_str, output_str = self._get_input_output(inputs, outputs)         self.chat_memory.add_ai_message(output_str) ```py    它专门设计用来确保聊天消息的记忆保持简洁和相关性，通过*仅存储 AI 生成的消息*。    这个明确的选择绕过了存储在生成步骤中使用的检索文档，防止内存膨胀。此外，内存机制确保 AI 始终了解其先前的写作，使其能够在累积的上下文超过设定限制时提供简化的摘要。    `generate_blog_post`函数遍历所有子标题，并尝试检索尽可能多的相关文档，同时适应当前上下文长度：    ``` def generate_blog_post(self) -> List[str]:         blog_post = []         print("Generating the blog post...\n---")         for subheading in self.outline.sub_headings:             k = 5  # Initialize k             while k >= 0:                 try:                     relevant_documents = (self.chroma_db.as_retriever() \                     .invoke(subheading.title,                     k=k))                     section_prompt = f"""  ...prompt_excluded_for_brevity...  Section text:  """                     result = self.blog_post_chain.predict(section_prompt)                     blog_post.append(result)                     break                 except Exception as e:                     print(f"An error occurred: {e}")                     k -= 1                 if k < 0:                     print('''All attempts to fetch relevant documents have  failed. Using an empty string for relevant_documents.  ''')                     relevant_documents = ""         print("Finished generating the blog post!\n---")         return blog_post ```py    这个函数`generate_blog_post`遍历每个子标题。它尝试获取最多五个相关文档。如果获取文档出现问题，它会智能地减少数量并再次尝试。如果所有尝试都失败，它将优雅地默认为没有文档。    最后，生成每个部分的提示非常丰富上下文：    ``` section_prompt = f"""You are currently writing the section: {subheading.title} `---` `Here are the relevant documents for this section:` `{``relevant_documents``}``.` `If the relevant documents are not useful, you can ignore them.` `You must never copy the relevant documents as this is plagiarism.` `---` `Here are the relevant insights that we gathered from our interview questions` `and answers:` `{``self``.``questions_and_answers``}``.` `You must include these insights where possible as they are important and will` `help our content rank better.` `---` `You must follow the following principles:` `- You must write the section:` `{``subheading``.``title``}` `` `- Render the output in .md format` `- Include relevant formats such as bullet points, numbered lists, etc.` `---` `Section text:` `"""` `` ```py   ``` ``The `section_prompt` elegantly sets the stage by announcing the specific section you’re working on, using `{subheading.title}`. But it doesn’t stop there. By feeding the LLM with `{relevant_documents}`, it offers background and depth, while explicitly cautioning against plagiarism. Moreover, by including insights from your interview via `{self.questions_and_answers}`, the prompt ensures that valuable information is front and center. Finally, it sets clear expectations on the format, the inclusion of certain features, and the topic at hand. This makes the LLM not just a tool but an informed coauthor, working diligently alongside you to create content.`` ```py  ````` ```py`# Writing Style    Now that the article is written, we can go a step further in terms of making it sound uniquely human, by rewriting the content in a specific writing style. This will go a long way in making the content less detectable as obviously AI (though ethically you should still declare any AI assistance) and decreasing the amount of time you spend editing the final draft before publishing.    Before rewriting, you need to know what writing style you want to emulate, be it your own or someone else’s. One common approach is to ask ChatGPT to summarize the writing style of someone who is famous, or at least popular enough in your industry to appear in ChatGPT’s training data. Commonly the model will want to respond with the name of the author and examples of writing, so adding instructions not to and ending the prompt with a bullet point (or an `-` character in this case) will give you the format you need.    Input:    ``` Describe in detail the writing style of Harry Dry from MarketingExamples.com Do not mention the writer or source, and respond only with bullet points: - ```py    Output:    ``` - Concise and to the point - Humorous and knowledgeable tone - Relatable and accessible language - Uses storytelling to convey ideas - Relies on examples and case studies - Provides actionable advice and tips - Uses subheadings and bullet points for easy readability - Emphasizes key points or takeaways with bold or italicized text - Addresses the reader directly, creating a conversational tone ```py    Often provided only a single sample of text, ChatGPT can generate a reasonable writing style guide, which can then be used for rewriting. Once the writing style is defined, elements can be mixed and matched to arrive at a more ideal style. The following example takes elements from both Mike Taylor’s writing style and Harry Dry’s writing style from the previous example. This is another example of meme unbundling, as discussed in “Meme Unbundling”:    ``` Rewrite the article in the following style:  - Concise and to the point - Professional and knowledgeable tone - Relatable and accessible language - Uses storytelling to convey ideas - Relies on examples and case studies - Mixes personal anecdotes with industry insights - Provides actionable advice and tips - Uses subheadings and bullet points for easy readability ```py    In practice this part of the AI writing process is the most difficult to get right, and it’s the only one that requires the larger and more expensive GPT-4 model to get passable results. If this part of the process isn’t right, the user can be left doing a lot of manual editing to get the writing in the house style. Given the strategic importance of this prompt, it makes sense to do a round of [prompt optimization](https://oreil.ly/H3VtJ), trying multiple approaches.    When optimizing prompts you can run the same prompt multiple times and check the average performance against an evaluation metric. As an example, here are the results of testing five different prompt approaches against an evaluation metric of embedding distance. The lower the score, the closer the embeddings of the response were to a reference answer (the text as rewritten manually is in the correct style). The prompts tested were as follows:    A      Control—the standard prompt as detailed in the preceding example.      B      One-shot writing sample—we provided one sample of text, and asked GPT-4 to describe the writing style.      C      Three-shot rewriting example—we gave three samples of the input text to GPT-4 and the rewritten version and asked it to describe the writing style.      D      Three-shot writing sample—same as previous, except without the input text, only the final samples of Mike’s writing.      These prompts were [tested in an experiment we ran](https://oreil.ly/vRRYO) against three test cases—memetics, skyscraper technique, and value-based pricing—which were snippets of text that were first generated by ChatGPT on a topic, for example: *explain value-based pricing*. We then manually rewrote the text in the style we desired to make reference texts for comparison. The embedding distance was calculated by getting the embeddings for the reference text (from OpenAI’s `text-embedding-ada-002`) and comparing them to the embeddings for the output from the prompt, using *cosine similarity* (a method for calculating the distance between two sets of numbers), as detailed in [LangChain’s embedding evaluator](https://oreil.ly/400gJ) (Figure 10-1).  ![pega 1001](img/pega_1001.png)  ###### Figure 10-1\. Test results from prompt optimization    As you can see from the results in Figure 10-1, some prompts work better than others, and some cases are easier for the AI to deliver on. It’s important to test across multiple cases, with 10 or more runs per case, to get a realistic result for each prompt. Otherwise, the nondeterministic nature of the responses might mean you’ll think the performance was better or worse than you can actually expect when scaling up usage of a prompt. Here was the final resulting prompt that performed best:    ``` You will be provided with the sample text. Your task is to rewrite the text into a different writing style. The writing style can be described as follows: 1\. Informative and Analytical: The writer presents detailed information about different strategies, especially the main theme of the text, and breaks down its benefits, challenges, and implementation steps. This depth of information shows that the writer has a solid grasp of the topic. 2\. Structured and Organized: The writing follows a logical flow, starting with a brief overview of different approaches, delving into a deep dive on the topic, and concluding with potential challenges and contexts where it might be best applied. 3\. Conversational Tone with Professionalism: While the information is presented in a professional manner, the writer uses a conversational tone ("Here’s how to implement..."), which makes it more relatable and easier for readers to understand. 4\. Practical and Actionable: The writer not only explains the concept but also offers actionable advice ("Here’s how to implement X") with step-by-step guidance based on real world-experience. 5\. Balanced Perspective: The writer doesn’t just present the benefits of the topic but also discusses its challenges, which gives a well-rounded perspective to readers. 6\. Examples and Analogies: To make concepts clearer, the writer uses concrete examples (e.g., how much a company might save per month) and analogies (e.g., making comparisons to popular frames of reference). This helps readers relate to the concepts and understand them better. 7\. Direct and Clear: The writer uses straightforward language without excessive jargon. Concepts are broken down into digestible bits, making it accessible for a broad audience, even if they're not well-versed in business strategies. In essence, this writing style is a blend of professional analysis with practical, actionable advice, written in a clear and conversational tone. ```py    # Evaluate Quality    Without testing the writing style, it would be hard to guess which prompting strategy would win. With a small amount of testing, you can be more confident this is the correct approach. Testing doesn’t have to be highly organized or systematized, and the builders of many successful AI products like [GitHub Copilot](https://oreil.ly/vu0IU) admit their eval process was haphazard and messy (but it got the job done!).    In this project we’ll use this well-tested example, but you may take this opportunity to try to beat this score. The repository with the reference texts and code is [publicly available on GitHub](https://oreil.ly/O6RdB), and please feel free to contribute to the repository if you find a better approach. One potential path to try is fine-tuning, which may get you better results in matching the writing style if you have enough samples ([OpenAI recommends at least 50](https://oreil.ly/OMMKi)). Even if you don’t perform an A/B test (comparing two versions of a prompt to see which one performs better) on this prompt, these results should convince you of the value of testing your prompts in general.    # Title Optimization    You can optimize the content’s title by generating various options, testing them through A/B prompts, and gauging their effectiveness with a thumbs-up/thumbs-down rating system, as shown in Figure 10-2.  ![pega 1002](img/pega_1002.png)  ###### Figure 10-2\. A simple thumbs-up and thumbs-down rating system    After evaluating all the prompts, you’ll be able to see which prompt had the highest average score and the token usage (Figure 10-3).  ![pega 1003](img/pega_1003.png)  ###### Figure 10-3\. Example A/B test results after manually evaluating a prompt    If you still aren’t getting the level of quality you need from this prompt, or the rest of the chain, this is a good time to experiment with a prompt optimization framework like [DSPy](https://oreil.ly/dspy). Upon defining an evaluation metric, DSPy tests different combinations of instructions and few-shot examples in your prompts, selecting the best-performing combination automatically. [See their documentation for examples](https://oreil.ly/vercel).    # AI Blog Images    One thing you can do to make your blog look more professional is to add custom illustrations to your blog posts, with a consistent style. At its maximum this may mean training a Dreambooth model, as covered in Chapter 9, on your brand style guide or a mood board of images with a certain visual consistency or aesthetic quality you value. In many cases, however, training a custom model is not necessary, because a style can be replicated well using simple prompting.    One popular visual style among business-to-business (B2B) companies, [Corporate Memphis](https://oreil.ly/3UHQs), is characterized by its vibrant color palettes, bold and asymmetric shapes, and a mix of both organic and geometric forms. This style arose as a [costly signaling technique](https://oreil.ly/haoTZ), showing that the company could afford to commission custom illustrations from a designer and therefore was serious enough to be trusted. You can replicate this style with AI, saving yourself the cost of custom illustrations, while benefiting from the prior associations formed in consumers’ minds. Figure 10-4 shows an example of Corporate Memphis style generated by Stable Diffusion, via the Stability AI API.    Input:    ``` illustration of websites being linked together. in the style of Corporate Memphis, white background, professional, clean lines, warm pastel colors ```py    Figure 10-4 shows the output.  ![pega 1004](img/pega_1004.png)  ###### Figure 10-4\. Corporate Memphis: “websites being linked together”    # Give Direction    Stable Diffusion is trained on many different styles, including obscure or niche styles like Corporate Memphis. If you know the name of a style, often that’s all that’s needed to guide the model toward the desired image. You can find a variety of art styles within this [visual prompt builder](https://oreil.ly/nxEzu).    In our blog writing project we could ask the user for an idea of what image they want to accompany the blog post, but let’s make it easier for them and automate this step. You can make an API call to ChatGPT and get back an idea for what could go in the image. When you get that response, it can form the basis of your prompt to Stability AI, a technique called *meta-prompting*, where one AI model writes the prompt for another AI model.    Input:    ``` Describe an image that would go well at the top of this article:  {text} ```py    Output:    ``` A seamless collage or mosaic of diverse cultural elements from around the world, including traditional dances, art pieces, landmarks, and people in various traditional attires, symbolizing the interconnectedness of human cultures. ```py    Stability AI hosts Stable Diffusion, including the latest models like Stable Diffusion XL, in their DreamStudio platform. You can also call them [via API](https://oreil.ly/XD_jQ) or via the Stability AI SDK (a library that simplifies the process of making the API call). In the following example, we’ll create a function for calling Stability AI with our prompt.    Input:    ``` import base64 import os import requests import uuid  engine_id = "stable-diffusion-xl-1024-v1-0" api_host = os.getenv('API_HOST', 'https://api.stability.ai') api_key = os.getenv("STABILITY_API_KEY")  def generate_image(prompt):     response = requests.post(         f"{api_host}/v1/generation/{engine_id}/text-to-image",         headers={             "Content-Type": "application/json",             "Accept": "application/json",             "Authorization": f"Bearer {api_key}"         },         json={             "text_prompts": [                 {                     "text":'''an illustration of "+prompt+". in the style of  Corporate Memphis,  white background, professional, clean lines, warm pastel  colors'''                 }             ],             "cfg_scale": 7,             "height": 1024,             "width": 1024,             "samples": 1,             "steps": 30,         },     )      if response.status_code != 200:         raise Exception("Non-200 response: " + str(response.text))      data = response.json()      image_paths = []      for i, image in enumerate(data["artifacts"]):         filename = f"{uuid.uuid4().hex[:7]}.png"         with open(filename, "wb") as f:             f.write(base64.b64decode(image["base64"]))          image_paths.append(filename)      return image_paths  prompt = """A seamless collage or mosaic of diverse cultural elements from around the world, including traditional dances, art pieces, landmarks, and people in various traditional attires, symbolizing the interconnectedness of human cultures."""  generate_image(prompt) ```py    Figure 10-5 shows the output.  ![pega 1005](img/pega_1005.png)  ###### Figure 10-5\. A seamless collage or mosaic of diverse cultural elements from around the world    To encapsulate the whole system for image generation, you can bring the call to ChatGPT and the resulting call to Stability AI together in one function that uses the `outline_result.title`:    ``` from image_generation_chain import create_image image = create_image(outline_result.title) ```py    The `create_image` function in *[image_generation_chain.py](https://oreil.ly/cWpXH)* utilizes Stable Diffusion to create an image based on a generated title from GPT-4:    ``` import base64 from langchain_openai.chat_models import ChatOpenAI from langchain_core.messages import SystemMessage import os import requests import uuid  engine_id = "stable-diffusion-xl-1024-v1-0" api_host = os.getenv("API_HOST", "https://api.stability.ai") api_key = os.getenv("STABILITY_API_KEY", "INSERT_YOUR_IMAGE_API_KEY_HERE")  if api_key == "INSERT_YOUR_IMAGE_API_KEY_HERE":     raise Exception(         '''You need to insert your API key in the  image_generation_chain.py file.'''         "You can get your API key from https://platform.openai.com/"     )   def create_image(title) -> str:     chat = ChatOpenAI()     # 1\. Generate the image prompt:     image_prompt = chat.invoke(         [             SystemMessage(content=f"""Create an image prompt  that will be used for Midjourney for {title}."""             )         ]     ).content       # 2\. Generate the image::     response = requests.post(         f"{api_host}/v1/generation/{engine_id}/text-to-image",         headers={             "Content-Type": "application/json",             "Accept": "application/json",             "Authorization": f"Bearer {api_key}",         },         json={             "text_prompts": [                 {                     "text": f'''an illustration of {image_prompt} in the  style of Corporate Memphis, white background,  professional, clean lines, warm pastel colors'''                 }             ],             "cfg_scale": 7,             "height": 1024,             "width": 1024,             "samples": 1,             "steps": 30,         },     )      if response.status_code != 200:         raise Exception("Non-200 response: " + str(response.text))      data = response.json()     image_paths = []      for i, image in enumerate(data["artifacts"]):         filename = f"{uuid.uuid4().hex[:7]}.png"         with open(filename, "wb") as f:             f.write(base64.b64decode(image["base64"]))         image_paths.append(filename)     return image_paths ```py    Here’s the high-level process:    1.  With the `ChatOpenAI` model, you’ll craft an image prompt for your given `title`.           2.  Using the Stability AI API, you’ll send this prompt to generate an image with precise styling instructions.           3.  Then you’ll decode and save this image locally using a unique filename and return its path.              With these steps, you’re not just prompting the AI to create textual content, but you’re directing it to bring your prompts to life visually.    This system is flexible based on whatever style you decide to use for blog images. Parameters can be adjusted as needed, and perhaps this API call can be replaced in future with a call to a custom fine-tuned Dreambooth model of your own. In the meantime, however, you have a quick and easy way to generate a custom image for each blog post, without requiring any further input from the user, in a consistent visual style.    # User Interface    Now that you have your script working end to end, you probably want to make it a little easier to work with, and maybe even get it into the hands of people who can give you feedback. The frontend of many AI tools in production is typically built using JavaScript, specifically the [NextJS](https://nextjs.org) framework based on React. This is usually paired with a CSS library such as [Tailwind CSS](https://tailwindcss.com), which makes rapid prototyping of design elements easier.    However, most of your AI code is likely in Python at this stage, and switching programming languages and development environments can be a daunting challenge. As well as learning JavaScript, NextJS, and Tailwind, you may also run into a series of issues getting a server running for your Python code, and a database live for your application and user data, and then integrating all of that with a frontend web design.    Instead of spending a lot of time spinning up servers, building databases, and adjusting button colors, it might make sense to create a simple prototype frontend to get early feedback, before investing too much at this stage in an unproven idea. Once you have built and tested a simple interface, you’ll have a better understanding of what to build when you do need to get your app production-ready.    For launching simple user interfaces for AI-based prototypes, there are several popular open source interfaces, including [gradio](https://www.gradio.app) and [Streamlit](https://streamlit.io). Gradio was acquired by HuggingFace and powers the web user interface for many interactive demos of open source AI models, famously including the [AUTOMATIC1111](https://oreil.ly/GlwJT) Stable Diffusion Web UI. You can quickly build a Gradio interface to make it easier to run your code locally, as well as sharing the prototype to get feedback.    We’ve created an interface that allows you to automate the entire process within two steps. You can get access to the [gradio source code here](https://oreil.ly/HNqVX).    Then run the gradio application by going into the [chapter_10 folder](https://oreil.ly/chapter10) within your terminal and running `python3 gradio_code_example.py`. The script will ask you to enter a `SERPAPI_API_KEY` and a `STABILITY_API_KEY` in your terminal.    Then you can access the gradio interface as shown in Figure 10-6.  ![pega 1006](img/pega_1006.png)  ###### Figure 10-6\. Gradio user interface    When you run gradio, you get an inline interface you can use directly or a URL that you can click to open the web interface in your browser. If you run gradio with the parameter `share=True`, for example `demo.launch(share=True)`, you get a publicly accessible link to share with friends, coworkers, or early users to get feedback on your prototype.    After initializing the interface, input a topic by clicking the Summarize and Generate Questions button. This will then collect and summarize the Google results as well as generate interview questions.    You’ll then need to fill in the answers for each question. Finally, click the Generate Blog Post & Image button, which will take all the questions, answers, and summaries and will create an entire blog post and image using GPT-4!    # Evaluate Quality    The most valuable evaluation data in AI is human feedback, as it has been the key to many AI alignment breakthroughs, including those that power ChatGPT. Asking for feedback from users via a user interface, or even building feedback mechanisms into your product, helps you identify and fix edge cases.    If you are building for research purposes or want to contribute to the open source community, consider sharing your gradio demo on Hugging Face Spaces. Hugging Face Spaces allows anyone to host their gradio demos freely, and uploading your project only takes a few minutes. New spaces can be created via the [Hugging Face website](https://oreil.ly/pSrP3), or done programmatically using the Hugging Face API.    # Summary    Congratulations! You’ve journeyed through the comprehensive world of prompt engineering for generative AI. You started with learning the prompt engineering principles and explored the historical context of LLMs, gaining awareness of their capabilities and the privacy concerns they pose.    You learned how to extract structured data, apply best practices of prompt engineering, and familiarize yourself with an LLM package called LangChain. Then you discovered vector databases for storing and querying text based on similarity and ventured into the world of autonomous agents.    Also, you immersed yourself in image generation techniques using diffusion models, learning how to navigate through this latent space. Your journey covered everything from format modifiers and art-style replication to inpainting and outpainting techniques. Moreover, you explored more advanced usage cases such as prompt expansion, meme mapping, and CLIP Interrogator, alongside many others.    Finally, you transitioned toward utilizing prompt engineering for content writing. You learned about creating a blog writing service that generates posts based on user responses, mimicking their writing styles, along with topic research strategies.    Overall, this journey not only enriched your knowledge but also equipped you with practical skills, setting you up to work professionally in the field of prompt engineering.    It’s been our pleasure to guide you through the wide domain of prompt engineering for generative AI. Thank you for staying with us to the end of this book. We trust it will become a useful tool in all your future work with AI.    We would also greatly appreciate hearing your thoughts about the book, as well as any remarkable projects you create using the techniques we’ve discussed.    Please feel free to share your feedback or showcase your work by emailing us at hi@brightpool.dev. Once again, thank you! Your curiosity and perseverance are what shapes the future of this exciting field, and we can’t wait to see what you contribute.    Happy prompting!```` ```py`` ``````py ``````py` ````````
