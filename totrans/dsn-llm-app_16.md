# Chapter 13\. Design Patterns and System Architecture

Throughout this book, we have explored a variety of techniques to adapt LLMs to solve our tasks, including in-context learning, fine-tuning, RAG, and tool use. While these techniques can potentially be successful in satisfying the performance requirements of your use case, deploying an LLM-based application in production requires adherence to a variety of other criteria like cost, latency, and reliability. To achieve these goals, an LLM application needs a lot of software scaffolding and specialized components.

To this end, in this chapter we will discuss various techniques to compose a production-level LLM system that can power useful applications. We will explore how to leverage multi-LLM architectures to balance cost and performance. Finally, we will look into software frameworks like DSPy that integrate LLM application development into the conventional software programming paradigm.

Treating an LLM-based application as just a standalone LLM component is inadequate if we intend to deploy it as a production-grade system. We need to treat it as a system, made up of several software and model components that support the LLM and make it reliable, fast, and cost-effective. The way these components are composed and connected is referred to as the *system architecture.*

Let’s begin by discussing a specific type: multi-LLM architectures that leverage multiple LLMs to solve your task.

# Multi-LLM Architectures

Throughout this book, we have discussed the tradeoffs involved in choosing the right LLM for a task. Often, it can be beneficial to leverage multiple LLMs to achieve the desired outcome. Multi-LLM architectures can exist in the following two modes (or a combination):

Each LLM is specialized for a different subtask

Different problem subtasks may require different levels of capabilities. To minimize cost and latency, for each task we would like to use the smallest possible LLM that can solve the subtask at the performance threshold we set.

All LLMs solve the same task

In this case, all the LLMs are solving the same task, but for each input, only one or a subset of LLMs may be chosen to solve it.

###### Tip

A given task can be solved by an ensemble of LLMs, and the final outputs can be chosen based on some rules (majority voting, interpolation, etc.). Refer to [Jiang et al.’s ensembling framework](https://oreil.ly/FEikT) called LLM-Blender for an example of thoughtful ensembling.

Let’s walk through some commonly used multi-LLM architectures.

## LLM Cascades

While using the state-of-the-art LLM for processing all our inputs is an option, realistically this might be cost-prohibitive or latency sensitive. To optimize costs while keeping performance standards high, we could leverage multiple LLMs, organized in a cascade architecture.

Let’s illustrate LLM cascades. Consider you have an application using three LLMs: one small, one medium, and one large, as illustrated in [Figure 13-1](#llm-cascades).

![llm-cascades](assets/dllm_1301.png)

###### Figure 13-1\. LLM cascades

The following process is observed during inference:

1.  Each input is fed to the small LLM.

2.  If the small LLM makes an output prediction with a confidence level greater than a threshold, then we accept the output as the final output.

3.  If the small LLM makes an output prediction with a confidence level that doesn’t surpass the threshold, then we pass the input to the medium model.

4.  Similarly, if the medium LLM makes an output prediction with a confidence level greater than a threshold, then we stop and accept this output as the final output.

5.  However, if the medium LLM makes an output prediction with a confidence level that doesn’t surpass the threshold, then we pass the input to the large model.

6.  The large model generates the final output.

This architecture is most beneficial when most user inputs can be processed by the small model.

If you are using encoder-only models like BERT, the output probability scores can be used as the measure of confidence. Thus, a group of well-calibrated models will enable us to efficiently route the input to the most suitable model. (Recall our discussion on model calibration in [Chapter 5](ch05.html#chapter_utilizing_llms).)

For decoder models, a popular method is to use self-consistency as a measure of confidence. (Recall our discussion on self-consistency in [Chapter 1](ch01.html#chapter_llm-introduction).) If we generate multiple times from the model and the outputs are mostly consistent with each other, then we can say that the model is being confident in its predictions. If they are not consistent, then we can move down the cascade and apply the inputs to the next LLM in the cascade.

###### Warning

Some works propose asking the LLM to explicitly state the confidence level of its output. This has not been proven to be effective yet. Beware of asking the LLM to verify its own work in any form!

Another method for assessing confidence is to use margin sampling, as proposed by [Ramirez et al.](https://oreil.ly/5s1rJ) In the margin sampling method, we generate the first token and use the difference in the probability of the most probable token and the second most probable token as the margin. The assumption is that the higher the margin, the more confident the model. If the margin is below a certain threshold, then the input is sent to the next model in the cascade.

An alternative to using cascades is using a router scheme.

## Routers

A router is a program or a model that processes input queries and dispatches them to the appropriate model. The advantage of using the router architecture is that, unlike cascades, the same input need not be run on potentially multiple models. However, the effectiveness of this strategy relies on the router effectively dispatching inputs to the optimal model, which may not always be fulfilled.

A router can perform intent classification, i.e., understand the intention of the user and dispatch the input to a suitable LLM that can solve the task being requested. If all the LLMs in the architecture are intended to solve the same task, then the router assesses the difficulty of the input query and dispatches the input to the smallest model that can adequately solve the task.

[Figure 13-2](#routers) illustrates the role of the router in picking the right model to solve a task.

![router](assets/dllm_1302.png)

###### Figure 13-2\. Router

###### Tip

Routers can also be used in RAG pipelines. The router can assess the input and dispatch it to one of several different types of retrievers.

Assessing the complexity of an input query can be done using either heuristics or a fine-tuned model. Heuristics can be based on certain keywords that appear in the input (with RAG, *When* queries are more easily answered than *How* queries) or the identity of the tasks (for instance, sentiment analysis is an easier task that can be accomplished by a smaller model).

Next, let’s discuss task-specialized LLMs.

## Task-Specialized LLMs

Yet another way of organizing multi-LLM architectures is to deploy a variety of task-specific LLMs, each of them specialized in solving a particular type of task or subtask.

Given a complex user query, a relatively powerful LLM can be used to decompose the query into its constituent subtasks. A router can then assign each of these subtasks to the specialized model most equipped to handle at the subtask. (Recall our discussion on task decomposition in [Chapter 8](ch08.html#ch8).)

Specialized LLMs can be constructed by fine-tuning them on task- and domain-specific datasets.

[Figure 13-3](#task-specific-llms) illustrates how a complex query can be divided into several subtasks, with each subtask being dispatched to the model most likely to solve it in a cost-optimal way.

![task-specific-llms](assets/dllm_1303.png)

###### Figure 13-3\. Task-specific LLMs

Let’s now explore some programming paradigms that facilitate more effective LLM application development.

# Programming Paradigms

As we have seen in this chapter, production-grade LLM systems can be composed of a lot of software components that help make the system robust and reliable. Naturally, we would like to use software design patterns to help us build these systems to be productive and maintainable. The developer community is still maturing in this regard, and it will take more time for tried and tested design patterns to emerge.

At this juncture, there are several proposals for LLM programming paradigms. While many are not yet well-tested, some of these paradigms are mature enough to support production-grade applications. Let’s explore a couple of major ones.

## DSPy

LLM application development is a highly iterative process. You might want to experiment with a few candidate LLMs before selecting the right one. You might start with zero-shot prompting, which involves a lot of iterative prompt manipulation, also called prompt engineering. If zero-shot isn’t sufficient, you might venture into few-shot prompting, which involves iterating over various candidate examples. If few-shot prompting isn’t sufficient, you might want to fine-tune the model, which involves iteratively preparing a dataset and trying various hyperparameters for the model. *D*eclarative *S*elf-improving Language *P*rograms, p*y*thonically (DSPy) is an open source programming framework that seeks to abstract a large part of the iterative process. Programming, not prompting, as their motto goes.

DSPy presents a framework where the application’s control flow is separated from variables that need to be iterated. The variables can be prompts, parameters of LLMs, etc. The programming blocks that manage the control flow of the application are called *modules*, and the blocks that perform the iterative updates of variables are called *optimizers*.

### Modules

A module is a building block of an LLM application. Each module corresponds to an underlying prompt in the prompt chain. Each module type is an abstraction of a different prompting technique, like CoT. A module can be declared using a *signature* that declaratively provides the input-output specification.

Declaring a CoT prompting module with a signature is as simple as:

```py
import dspy
summarizer = dspy.ChainOfThought('document -> summary')
```

`ChainOfThought` is a module that provides an abstraction for the CoT prompting technique. The module is declared with a signature `document → summary` that specifies the input and output types in a declarative form. For instance, if you are building a question-answering application, then the signature could be `question → answer`.

For some applications, you would like to provide more details on the input-output mapping than just a short string. For those instances, signatures can be declared using Python classes. Here’s an example:

```py
class RAGQA(dspy.Signature):
    """Using only information in the provided context,
 answer the question in the text"""

    context = dspy.InputField(desc="context might be irrelevant")
    text = dspy.InputField()
    answer = dspy.OutputField(desc="Answer in at most two sentences.")

context = "Tempura was invented in New Zealand by a retired rugby player. The
word 'tempura' comes from the German opera by Neubig."
text = "Which year was tempura invented in?"
answer = dspy.ChainOfThought(RAGQA)
answer(context=context, text=text)
```

In this example, instructions can be provided in three places:

*   The docstring, with a more detailed description of the task

*   The input field, with details on any input constraints

*   The output field, with details on any output constraints

Refer to the [DSPy documentation](https://oreil.ly/4Vy5c) for a full list of available modules. We can use these modules as building blocks for constructing complex LLM applications. Next let’s look at optimizers that work under the hood to *compile* our modules into an executable program.

### Optimizers

Optimizers are components that update prompts or model parameters. Several optimizers are natively supported by DSPy. An optimizer can be used to update one of the following:

*   The instruction prompt

*   Few-shot training examples

*   Model parameters (fine-tuning)

An optimizer takes as input the modules it needs to be applied to, the metric to evaluate the output of the modules, and fine-tuning or few-shot training data consisting of input-output pairs or just inputs. Optimizers use algorithms to update the prompts or parameters to optimize the desired metric. DSPy supports metrics like *accuracy* or *precision* or *exact match*.

You can implement your own modules and optimizers if the ones provided by default are inadequate to your needs. Thus, DSPy is a powerful framework that separates the control flow of the LLM application from iterative aspects like LLM prompting and fine-tuning, and potentially automates the latter. The downsides of DSPy are that the optimizers might not be effective enough to work in an automated fashion and might need manual intervention to tune them correctly. More often than not, you will find yourself writing your own optimizers.

Let’s now explore another framework called Language Model Query Language (LMQL). We have already been introduced to this framework in [Chapter 5](ch05.html#chapter_utilizing_llms) in the context of structured generation, but here we will look at how the same framework can be used as a programming paradigm for developing LLM applications.

## LMQL

LMQL is a superset of Python that enables specifying prompts, output constraints, and program control flow using declarative Python code. Here is an example:

```py
import lmql

@lmql.query(model="gpt-4")
def jeopardy():
    '''lmql
 """Generate a Jeopardy! question and answer.
 A:[ANSWER]
 Q:[QUESTION]""" where STOPS_AT(ANSWER, "?") and \
 STOPS_AT(QUESTION, "\n")
 '''

jeopardy(model=lmql.model("gpt-4"))
```

In this example we are asking the model to generate a Jeopardy! question. Jeopardy! is a TV show that executes a modified version of a trivia quiz; the host supplies the answers and the contestants provide the question for the given answer.

In LMQL, we achieve this by defining a function called `jeopardy` and supplying the prompt instructions in the doc string. The doc string contains the instruction `Generate a Jeopardy! question and answer`. The `[ANSWER]` and `[QUESTION]` markers refer to templates that the LLM will fill in based on the constraints specified in the `WHERE` clause.

For the answer (which in Jeopardy is the question), we stop generation after generating the `?` symbol. Similarly, for the question (which in Jeopardy is the answer), we stop generation after the newline symbol. The `WHERE` clause can be used to provide complex constraints for generation.

LMQL syntax might take a while to get used to, but overall it provides a robust programmatic foundation for developing LLM programs. Both LMQL and DSPy have a learning curve, so I recommend being patient during your first few iterations.

As LLMs and LLM-driven applications mature, I expect more programming paradigms to emerge and for existing paradigms to vastly evolve. Current paradigms might be too brittle in many cases, so be cautious and verify they are effective before you adopt them in production.

# Summary

In this chapter, we explored the construction of LLM systems and various system architectures. We showcased how we can leverage multi-LLM architectures to optimize for cost and latency. Finally, we introduced LLM programming frameworks for streamlining LLM application development.