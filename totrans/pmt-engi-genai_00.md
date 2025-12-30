# Preface

The rapid pace of innovation in generative AI promises to change how we live and work, but it’s getting increasingly difficult to keep up. The number of [AI papers published on arXiv is growing exponentially](https://oreil.ly/EN5ay), [Stable Diffusion](https://oreil.ly/QX-yy) has been among the fastest growing open source projects in history, and AI art tool [Midjourney’s Discord server](https://oreil.ly/ZVZ5o) has tens of millions of members, surpassing even the largest gaming communities. What most captured the public’s imagination was OpenAI’s release of ChatGPT, [which reached 100 million users in two months](https://oreil.ly/FbYWk), making it the fastest-growing consumer app in history. Learning to work with AI has quickly become one of the most in-demand skills.

Everyone using AI professionally quickly learns that the quality of the output depends heavily on what you provide as input. The discipline of *prompt engineering* has arisen as a set of best practices for improving the reliability, efficiency, and accuracy of AI models. “In ten years, half of the world’s jobs will be in prompt engineering,” [claims Robin Li](https://oreil.ly/IdIfO), the cofounder and CEO of Chinese tech giant Baidu. However, we expect prompting to be a skill required of many jobs, akin to proficiency in Microsoft Excel, rather than a popular job title in itself. This new wave of disruption is changing everything we thought we knew about computers. We’re used to writing algorithms that return the same result every time—not so for AI, where the responses are non-deterministic. Cost and latency are real factors again, after decades of Moore’s law making us complacent in expecting real-time computation at negligible cost. The biggest hurdle is the tendency of these models to confidently make things up, dubbed *hallucination*, causing us to rethink the way we evaluate the accuracy of our work.

We’ve been working with generative AI since the GPT-3 beta in 2020, and as we saw the models progress, many early prompting tricks and hacks became no longer necessary. Over time a consistent set of principles emerged that were still useful with the newer models, and worked across both text and image generation. We have written this book based on these timeless principles, helping you learn transferable skills that will continue to be useful no matter what happens with AI over the next five years. The key to working with AI isn’t “figuring out how to hack the prompt by adding one magic word to the end that changes everything else,” as [OpenAI cofounder Sam Altman asserts](https://oreil.ly/oo262), but what will always matter is the “quality of ideas and the understanding of what you want.” While we don’t know if we’ll call it “prompt engineering” in five years, working effectively with generative AI will only become more important.

# Software Requirements for This Book

All of the code in this book is in Python and was designed to be run in a [Jupyter Notebook](https://jupyter.org) or [Google Colab notebook](https://colab.research.google.com). The concepts taught in the book are transferable to JavaScript or any other coding language if preferred, though the primary focus of this book is on prompting techniques rather than traditional coding skills. The code can all be [found on GitHub](https://oreil.ly/BrightPool), and we will link to the relevant notebooks throughout. It’s highly recommended that you utilize the [GitHub repository](https://oreil.ly/BrightPool) and run the provided examples while reading the book.

For non-notebook examples, you can run the script with the format `python content/chapter_x/script.py` in your terminal, where `x` is the chapter number and `script.py` is the name of the script. In some instances, API keys need to be set as environment variables, and we will make that clear. The packages used update frequently, so install our [*requirements.txt*](https://oreil.ly/BPreq) in a virtual environment before running code examples.

The *requirements.txt* file is generated for Python 3.9\. If you want to use a different version of Python, you can generate a new *requirements.txt* from this [*requirements.in*](https://oreil.ly/YRwP7) file found within the GitHub repository, by running these commands:

```py
`pip install pip-tools`
`pip-compile requirements.in`
```

For Mac users:

1.  Open Terminal: You can find the Terminal application in your Applications folder, under Utilities, or use Spotlight to search for it.

2.  Navigate to your project folder: Use the `cd` command to change the directory to your project folder. For example: `cd path/to/your/project`.

3.  Create the virtual environment: Use the following command to create a virtual environment named `venv` (you can name it anything): `python3 -m venv venv`.

4.  Activate the virtual environment: Before you install packages, you need to activate the virtual environment. Do this with the command `source venv/bin/activate`.

5.  Install packages: Now that your virtual environment is active, you can install packages using `pip`. To install packages from the *requirements.txt* file, use `pip install -r requirements.txt`.

6.  Deactivate virtual environment: When you’re done, you can deactivate the virtual environment by typing **`deactivate`**.

For Windows users:

1.  Open Command Prompt: You can search for `cmd` in the Start menu.

2.  Navigate to your project folder: Use the `cd` command to change the directory to your project folder. For example: `cd path\to\your\project`.

3.  Create the virtual environment: Use the following command to create a virtual environment named `venv`: `python -m venv venv`.

4.  Activate the virtual environment: To activate the virtual environment on Windows, use `.\venv\Scripts\activate`.

5.  Install packages: With the virtual environment active, install the required packages: `pip install -r requirements.txt`.

6.  Deactivate the virtual environment: To exit the virtual environment, simply type: `deactivate`.

Here are some additional tips on setup:

*   Always ensure your Python is up-to-date to avoid compatibility issues.

*   Remember to activate your virtual environment whenever you work on the project.

*   The *requirements.txt* file should be in the same directory where you create your virtual environment, or you should specify the path to it when using `pip install -r`.

Access to an OpenAI developer account is assumed, as your `OPENAI_API_KEY` must be set as an environment variable in any examples importing the OpenAI library, for which we use version 1.0\. Quick-start instructions for setting up your development environment can be found in [OpenAI’s documentation](https://oreil.ly/YqbrY) on their website.

You must also ensure that *billing is enabled* on your OpenAI account and that a valid payment method is attached to run some of the code within the book. The examples in the book use GPT-4 where not stated, though we do briefly cover Anthropic’s competing [Claude 3 model](https://oreil.ly/jY8Ai), as well as Meta’s open source [Llama 3](https://oreil.ly/BbXZ3) and [Google Gemini](https://oreil.ly/KYgij).

For image generation we use [Midjourney](https://www.midjourney.com), for which you need a Discord account to sign up, though these principles apply equally to DALL-E 3 (available with a ChatGPT Plus subscription or via the API) or Stable Diffusion (available as an [API](https://oreil.ly/cmTtW) or it can [run locally](https://oreil.ly/Ha0T5) on your computer if it has a GPU). The image generation examples in this book use Midjourney v6, Stable Diffusion v1.5 (as many extensions are still only compatible with this version), or [Stable Diffusion XL](https://oreil.ly/S0P4s), and we specify the differences when this is important.

We provide examples using open source libraries wherever possible, though we do include commercial vendors where appropriate—for example, [Chapter 5](ch05.html#vector_databases_05) on vector databases demonstrates both FAISS (an open source library) and Pinecone (a paid vendor). The examples demonstrated in the book should be easily modifiable for alternative models and vendors, and the skills taught are transferable. [Chapter 4](ch04.html#advanced_text_04) on advanced text generation is focused on the LLM framework LangChain, and [Chapter 9](ch09.html#advanced_image_09) on advanced image generation is built on AUTOMATIC1111’s open source Stable Diffusion Web UI.

# Conventions Used in This Book

The following typographical conventions are used in this book:

*Italic*

Indicates new terms, URLs, email addresses, filenames, and file extensions.

`Constant width`

Used for program listings, as well as within paragraphs to refer to program elements such as variable or function names, databases, data types, environment variables, statements, and keywords.

**`Constant width bold`**

Shows commands or other text that should be typed literally by the user.

*`Constant width italic`*

Shows text that should be replaced with user-supplied values or by values determined by context.

###### Tip

This element signifies a tip or suggestion.

###### Note

This element signifies a general note.

###### Warning

This element indicates a warning or caution.

Throughout the book we reinforce what we call the Five Principles of Prompting, identifying which principle is most applicable to the example at hand. You may want to refer to [Chapter 1](ch01.html#five_principles_01), which describes the principles in detail.

# Principle Name

This will explain how the principle is applied to the current example or section of text.

# Using Code Examples

Supplemental material (code examples, exercises, etc.) is available for download at [*https://oreil.ly/prompt-engineering-for-generative-ai*](https://oreil.ly/prompt-engineering-for-generative-ai).

If you have a technical question or a problem using the code examples, please send email to [*bookquestions@oreilly.com*](mailto:bookquestions@oreilly.com).

This book is here to help you get your job done. In general, if example code is offered with this book, you may use it in your programs and documentation. You do not need to contact us for permission unless you’re reproducing a significant portion of the code. For example, writing a program that uses several chunks of code from this book does not require permission. Selling or distributing examples from O’Reilly books does require permission. Answering a question by citing this book and quoting example code does not require permission. Incorporating a significant amount of example code from this book into your product’s documentation does require permission.

We appreciate, but generally do not require, attribution. An attribution usually includes the title, author, publisher, and ISBN. For example: “*Prompt Engineering for Generative AI* by James Phoenix and Mike Taylor (O’Reilly). Copyright 2024 Saxifrage, LLC and Just Understanding Data LTD, 978-1-098-15343-4.”

If you feel your use of code examples falls outside fair use or the permission given above, feel free to contact us at [*permissions@oreilly.com*](mailto:permissions@oreilly.com).

# O’Reilly Online Learning

###### Note

For more than 40 years, [*O’Reilly Media*](https://oreilly.com) has provided technology and business training, knowledge, and insight to help companies succeed.

Our unique network of experts and innovators share their knowledge and expertise through books, articles, and our online learning platform. O’Reilly’s online learning platform gives you on-demand access to live training courses, in-depth learning paths, interactive coding environments, and a vast collection of text and video from O’Reilly and 200+ other publishers. For more information, visit [*https://oreilly.com*](https://oreilly.com).

# How to Contact Us

Please address comments and questions concerning this book to the publisher:

*   O’Reilly Media, Inc.
*   1005 Gravenstein Highway North
*   Sebastopol, CA 95472
*   800-889-8969 (in the United States or Canada)
*   707-827-7019 (international or local)
*   707-829-0104 (fax)
*   [*support@oreilly.com*](mailto:support@oreilly.com)
*   [*https://www.oreilly.com/about/contact.html*](https://www.oreilly.com/about/contact.html)

We have a web page for this book, where we list errata, examples, and any additional information. You can access this page at [*https://oreil.ly/prompt-engineering-generativeAI*](https://oreil.ly/prompt-engineering-generativeAI).

For news and information about our books and courses, visit [*https://oreilly.com*](https://oreilly.com).

Find us on LinkedIn: [*https://linkedin.com/company/oreilly-media*](https://linkedin.com/company/oreilly-media).

Watch us on YouTube: [*https://youtube.com/oreillymedia*](https://youtube.com/oreillymedia).

# Acknowledgments

We’d like to thank the following people for their contribution in conducting a technical review of the book and their patience in correcting a fast-moving target:

*   Mayo Oshin, early LangChain contributor and founder at [SeinnAI Analytics](https://www.siennaianalytics.com)

*   Ellis Crosby, founder at [Scarlett Panda](https://www.scarlettpanda.com) and AI agency [Incremen.to](https://incremen.to)

*   Dave Pawson, O’Reilly author of [*XSL-FO*](https://oreil.ly/XSL-FO)

*   Mark Phoenix, a senior software engineer

*   Aditya Goel, GenAI consultant

*   Sanyam Kumar, Associate Director, Data Science, Genmab

*   Lakshmanan Sethu, TAM, Gen AI Solutions, Google

*   Janit Anjaria, Staff TLM, Aurora Innovation Inc.

We are also grateful to our families for their patience and understanding and would like to reassure them that we still prefer talking to them over ChatGPT.