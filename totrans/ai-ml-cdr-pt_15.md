# Chapter 14\. Using Third-Party Models and Hubs

The success of the open source PyTorch framework has led to the growth of supplementary ecosystems. In this chapter, we’ll look at the various options of pretrained models and the associated tools and resources used to download, instantiate, and use them for inference.

While the PyTorch framework provides the foundation for deep learning, the community has created numerous repositories and hubs that store models that are ready to use and extend, making it easier for you to use and extend existing work rather than starting from scratch. I like to call this “standing on the shoulders of giants.”

Since the advent of generative AI, these hubs have exploded in popularity, and many scenarios of generative ML models within workflows have grown out of this. As a result, when it comes to using pretrained models, there are many options. You might use them directly for inference, taking advantage of those trained on massive datasets that would be impractical to replicate. Or you might use these models as starting points for fine-tuning, adapting them to specific domains or tasks while retaining their learned features. This can take the form of low-rank adaption (LoRA), as we’ll discuss in [Chapter 20](ch20.html#ch20_tuning_generative_image_models_with_lora_and_diffu_1748550104901965), or *transfer learning*, in which knowledge from one task is applied to another. Transfer learning or other fine-tuning has become a standard practice, especially when working with limited data or computational resources.

The advantages of using pretrained models extend beyond saving computational resources and time. These models often represent state-of-the-art architectures, and they’ve been trained on diverse, high-quality datasets that you may not have direct access to.

Additionally, the providers generally release the model with extensive documentation, performance benchmarks, and community support, giving you a long head start. Given the importance of responsible AI, these models often come with model cards that help you understand any research and work done so you can navigate any potential responsibility issues.

There is no “One Hub to Rule Them All,” so it’s useful to understand each of the major ones and how you can make the most of them. To that end, we’ll look at some of the more popular ones in this chapter.

Hugging Face has become the de facto standard for transformer models, while PyTorch Hub offers officially supported implementations. Platforms like Kaggle provide competition-winning models, and GitHub-based TorchHub enables direct access to research implementations.

I think it’s important for you to understand these resources and how to use them effectively. As the field of deep learning continues to advance, these hubs play an increasingly crucial role in widening access to state-of-the-art models and enabling rapid development of AI applications. And as the role of AI developer matures and grows, I’m personally seeing huge growth in the careers of software developers who don’t train models from scratch and instead use or fine-tune existing ones. To that end, I hope this chapter helps you grow!

# The Hugging Face Hub

In recent years, particularly with the rise of generative AI, the Hugging Face Hub has emerged as a leading platform for discovering and using pretrained ML models, particularly for NLP. Much of its usefulness (and a significant driver of its success) is the open source availability of two things: a transformers library (which makes using pretrained language models very easy to use) and a diffusers library (which does the same for text-to-image generative models like stable diffusion).

As a result, what started as a repository for transformer-based models has evolved into a comprehensive ecosystem supporting computer vision, audio processing, and reinforcement learning models. It has grown into a one-stop shop combining version control for models, documentation, and model cards—and because of the PyTorch-friendly libraries like transformers and diffusers, using these models with your Python and PyTorch skills is relatively easy.

Collaboration has also been one of the keys to the Hub’s success. You can download, use, and fine-tune models with just a few lines of code, and many developers and organizations have shared their models or fine-tunes with the community. There were over 900,000 publicly available models at the time of writing, so there’s plenty to choose from!

## Using Hugging Face Hub

Before rolling up your sleeves to code with Hugging Face Hub, you should get an account and use it to get an API token.

### Getting a Hugging Face token

This section will walk you through the [*HuggingFace.co*](http://huggingface.co) user interface as it existed at the time of writing. It may have changed by the time you’re reading this, but the principles are still the same. Hopefully, they’ll still apply!

Start by visiting [*Huggingface.co*](http://huggingface.co), and if you don’t already have an account, you can use the Sign Up button at the top right to create one (see [Figure 14-1](#ch14_figure_1_1748549787233053)).

![](assets/aiml_1401.png)

###### Figure 14-1\. Signing up for Hugging Face

Once you’ve signed up and gotten an account, you can sign in, and in the top-right-hand corner of the page, you’ll see your avatar icon. Select this and a drop-down menu will appear. On this menu, you’ll see an option to Access Tokens, and you can select it to view your access tokens (see [Figure 14-2](#ch14_figure_2_1748549787233102)).

![](assets/aiml_1402.png)

###### Figure 14-2\. Access tokens

On this page, you’ll see a Create New Token button, which will take you to a screen where you can specify your token details. Select the Read tab and give the token a name. For example, in [Figure 14-3](#ch14_figure_3_1748549787233128), you can see where I created a new Read token called PyTorch Book.

You’ll also see a pop-up asking you to save your access token (see [Figure 14-4](#ch14_figure_4_1748549787233152)). Note that it tells you that you will not be able to see the token again after you close this dialog modal, so be sure to hit the Copy button to have the token ready for the next steps.

![](assets/aiml_1403.png)

###### Figure 14-3\. Creating an access token

![](assets/aiml_1404.png)

###### Figure 14-4\. Saving your access token

If you *do* forget the token, you’ll have to Invalidate and Refresh it on the token list screen. To do this, you select the three dots to the right of the token and then select Invalidate and Refresh from the drop-down menu (see [Figure 14-5](#ch14_figure_5_1748549787233175)).

![](assets/aiml_1405.png)

###### Figure 14-5\. Invalidating and refreshing a token

Then, go back to the dialog from [Figure 14-4](#ch14_figure_4_1748549787233152) with a new token value. Copy it if you want to use it.

Now that you have a token, let’s explore how to configure Colab to use it.

### Getting permission to use models

Many models on Hugging Face will require additional permission to use them. In those cases, you should always check the model page and apply for permission on the link provided. Your permission to use the model will be tracked using the Hugging Face token. If you do *not* have permission, you’ll see an error like this:

```py
GatedRepoError: 401 Client Error.
(Request ID: [...])
Cannot access gated repo for url [...]
Access to model [...] is restricted.
You must have access to it and be authenticated to access it.
Please log in.
```

When this happens, the easiest thing to do is use the model name to find its landing page on the Hugging Face Hub and follow the steps to get permission to use it from there.

### Configuring Colab for a Hugging Face token

If you want to use models from Hugging Face in Google Colab, then you need to configure a Colab secret in which code executing in Colab will read the token value, send it to Hugging Face on your behalf, and grant you access to the object.

It’s pretty easy to do. First, in Colab, you select the key icon on the left of the screen (see [Figure 14-6](#ch14_figure_6_1748549787233195)).

You should see a list of secrets that looks like the one in [Figure 14-7](#ch14_figure_7_1748549787233216). Don’t worry if you don’t have any API keys there yet. At the bottom of the list is a button that says Add new secret, and you’ll select that.

![](assets/aiml_1406.png)

###### Figure 14-6\. Selecting the Colab secrets

![](assets/aiml_1407.png)

###### Figure 14-7\. List of Colab secrets

Use the name “HF_TOKEN” in the Name field, and paste the value of the key into the Value field. Then, flip the switch to give Notebook Access to the secret (see [Figure 14-8](#ch14_figure_8_1748549787233237)).

![](assets/aiml_1408.png)

###### Figure 14-8\. Configuring the HF_TOKEN in Colab

Your code in Colab will now use this token to access Hugging Face.

### Using the Hugging Face token in code

If you just want to use the token directly in your code, whether in Colab or not, you’ll have to log in to the Hugging Face Hub in your code and pass the key to it. It’s pretty straightforward, with the Hugging Face Hub libraries providing the required support.

To start, just import the `login` class like this:

```py
from huggingface_hub import login
```

You can then pass the token to the `login` class and initialize by using it in your Python session like this:

```py
login(token="YOUR_TOKEN_HERE")
```

The Hugging Face classes will then use the token for the remainder of your session.

## Using a Model From Hugging Face Hub

Once you have your token set up, getting and using a model is very simple. For this walk-through, we’ll explore using a language model for text classification and sentiment analysis. This will require you to use the transformers library, so be sure to have it installed with this:

```py
pip install transformers
```

The `transformers` API offers a pipeline class that lets you download and use a model based on its name in the Hugging Face repository. This one was fine-tuned using the SST sentiment analysis dataset from Stanford:

```py
# Load a small sentiment analysis model
classifier = pipeline("sentiment-analysis", 
             model="distilbert-base-uncased-finetuned-sst-2-english")
```

Pipeline offers much more than just downloading. It encapsulates what’s needed to perform common tasks with models. The first parameter, which in this case is `sentiment analysis`, describes the overall pipeline task that you’ll do. Transformers offer a variety of task types, including this, text classification, text generation, and a whole lot more.

When using the `pipeline` class, a number of key steps take place under the hood. These include the following:

Tokenization

In this step, the text is converted into tokens (as we discussed in [Chapter 4](ch04.html#ch04_using_data_with_pytorch_1748548966496246)).

Input processing

In this step, special tokens are added and the text is converted into tensors.

The model forward pass

In this step, the tokenized input is passed through the model’s layers to get a result.

Output processing

In this step, the output is decoded from tensors back to the desired labels.

You can see this workflow in [Figure 14-9](#ch14_figure_9_1748549787233257).

![](assets/aiml_1409.png)

###### Figure 14-9\. NLP pipeline flow

Then, when you want to use it, the burden of coding is removed from you as the developer and you just use the model like this:

```py
# Test the model
text = "I love programming with PyTorch!"
result = classifier(text)
print(result)  # Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

All of the steps required for text classification and sentiment analysis are encapsulated and abstracted away from you. It makes your code much simpler!

Similarly, if you want to use the diffusers library, it comes with a number of pipelines that are often associated with a model type. So, for example, if you want to use the popular Stable Diffusion model for text to image—in which you give a prompt and the model will draw an image based on that prompt—you can do so very easily.

Let’s explore this with an example.

First, from the diffusers library, you can import the pipeline that supports Stable Diffusion like this:

```py
from diffusers import StableDiffusionPipeline
```

With this, you can specify the name of the model in the Hugging Face repository and use it to initialize the pipeline:

```py
import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                               torch_dtype=torch.float16)
pipe = pipe.to(device)
```

Similar to the preceding text example, the pipeline encapsulates and abstracts a number of steps away from you. This means you can write relatively simple code like this:

```py
prompt = "a cute colorful cartoon cat"
image = pipe(prompt).images[0]

image.save("cat.png")
```

But a number of steps have been handled for you. These include the following:

1.  In the text encoding step, Stable Diffusion uses a technology called CLIP to take the text prompts and turn them into embeddings that the model can understand.

2.  An initial image is then constructed from random noise.

3.  The embeddings are then fed into the model, which uses a process of denoising to create pixels and features that match the embeddings.

4.  The final output of the model is then converted from tensors into an RGB image.

The overall process of creating an image from text is beyond the scope of this chapter, but it’s well explained in [this video from Google Research](https://oreil.ly/zNjUT).

The important thing to note here is that because the image-generation process begins by creating random noise, any images you create with the preceding code will be different. So, don’t be alarmed if you’re not getting the same picture consistently! There are ways of guiding this noise by using a seed, which we’ll discuss in later chapters.

# PyTorch Hub

One of the primary reasons for PyTorch’s success—particularly in the research community—is the foresight of the developers in creating a hub where people could share their models. While this functionality has been massively superseded by Hugging Face Hub, as described earlier in this chapter, it’s still worth looking at because many new and innovative models (or updates to existing ones) like YOLO are often shared on the Hub.

###### Note

YOLO is “You Only Look Once,"” a popular and efficient object detection model.

As with Hugging Face Hub, the primary benefit of PyTorch Hub is that it gives you access either to models that you may not have the compute resources to train yourself or to the required data used to train them. At its core, PyTorch Hub functions as a centralized repository where researchers and developers can publish, share, and access models that have been trained on diverse datasets across various domains.

In this section, we’ll explore PyTorch Hub and the APIs that you’ll use to access models within it. Unfortunately, the APIs aren’t as consistent as they could be, and it can sometimes be a little bit of a struggle to understand everything. But hopefully, this chapter will help!

We’ll start with the PyTorch Vision libraries, which are composed of image classifiers, object detectors, and other computer vision models.

## Using PyTorch Vision Models

Before you begin, you’ll need to ensure that you have torchvision installed. Use this:

```py
pip install torchvision
```

When you have installed it, you’ll see the install version. This is really important when using Hub, in particular when you want to list the models to see what’s available. You can also see the versions of these [on GitHub](https://oreil.ly/KIiFD).

So, to list the models that are available, you’ll use code like this:

```py
models = torch.hub.list('pytorch/vision:v0.20.1')
for model in models:
        print(model)
```

Note the version number (which you can get from the GitHub page we just mentioned).

At the time of writing, there were close to a hundred models on this list. Do note that your version of the tag should match your version of torchvision, so if you are having problems, you can use this code to see your current version:

```py
print(torchvision.__version__)
```

You can also choose a model from those available and load it into memory like this:

```py
# Load ResNet-50 from PyTorch Hub
model = torch.hub.load('pytorch/vision:v0.20.1', 'resnet50', 
                        pretrained=True)

# Set the model to evaluation mode
model.eval()
```

The model will be downloaded, cached, and then placed into evaluation mode.

Next up, you’ll need to get your data ready for inference, and that requires you to have some domain knowledge of the model. So, for example, in the code we just cited, we used `resnet50` as the model. This model (ResNet) is a very popular one for image classification that uses CNNs. A great place to go to learn more about this is the [PyTorch Hub site](https://pytorch.org/hub).

From here, you can dig into model details—such as the size of the desired input, the labels that it can classify, etc. Then, with this information in hand, you can write inference code for the model. Here’s an example:

```py

# Load and preprocess the image
image_path = "example.jpg"  # Replace with your image path
image = Image.open(image_path).convert('RGB')

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)
```

You saw similar code in [Chapter 3](ch03.html#ch03_going_beyond_the_basics_detecting_features_in_ima_1748570891074912) and [Chapter 4](ch04.html#ch04_using_data_with_pytorch_1748548966496246), where you explored building your own image classifier. This code resizes the images to 256 × 256 and then crops a 224 × 224 image from the center of that. Images are typically 32 bit RGB, in which each pixel is represented by 8 bits of alpha, 8 bits of red, 8 bits of green, and 8 bits of blue. However, for image classification, the neural network usually expects normalized values (i.e., between 0 and 1), so the transform to normalize the image performs this.

When you’re using models in PyTorch, even though you know the desired dimensions (in this case, 224 × 224), you’ll also need to batch the images for inference, even if you’re just doing a single image. The `input_tensor.unsqueeze(0)` adds this extra dimension to the input tensor to handle this.

Next up, you’ll do the actual inference, which just means you’ll load the model onto the appropriate device—which is cuda if you have a GPU and the CPU otherwise. You’ll then pass the input batch to the model to get an output:

```py
# Perform inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_batch = input_batch.to(device)

with torch.no_grad():
    output = model(input_batch)

_, predicted_idx = torch.max(output, 1)
```

The `predicted_idx` is the output for the class that has the highest probability of matching the input image. You’ll see something numeric, like `tensor([153])`, in the output here. Recall that models’ output layers will be neurons that correspond to the index of the required label. In the case of ResNet, the number 153 is a Maltese dog.

To decode this, you can use code like the following. You can find the URL of the labels file by digging into the model page on PyTorch Hub:

```py
# Get class labels and map to prediction
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/
                                         master/imagenet-simple-labels.json"
class_labels = json.load(urllib.request.urlopen(url))
predicted_label = class_labels[predicted_idx]
print("Predicted Label:", predicted_label)
```

This code will then print out the label for `predicted_idx`. So, in the case of `tensor([153])`, this will output the label for a Maltese dog.

## Natural Language Processing

The PyTorch Hub for NLP ultimately directs to two different repositories: those implemented using [Hugging Face Transformers](https://oreil.ly/j1w3-) as outlined earlier in this chapter and those from [Facebook’s fairseq research team](https://oreil.ly/mmQeY).

If you use the fairseq models, you may encounter a lot of sharp edges. Therefore, I thoroughly recommend setting up an environment with Python 3.11 (no later than that).

Within that, you can set up fairseq2 like this:

```py
pip install fairseq
```

You’ll likely also need other dependencies like hydra-core, OmegaConf, and requests.

Once you have the full system set up, you can use fairseq models like this:

```py
import torch

en2de = torch.hub.load(
     'pytorch/fairseq','transformer.wmt19.en-de.single_model')

en2de.translate('Hello Pytorch', beam=5)
# 'Hallo Pytorch'
```

Do note that the environment will be very picky about which versions of PyTorch, pip, and many other libraries you can use. It can make for a very brittle experience, and unless you really want to use the models from the fairseq repository, I’d recommend just going with the Hugging Face transformer versions.

## Other Models

PyTorch Hub also has repos for a variety of other model types, including audio, reinforcement learning, generative AI, and more. I’ve found the best way to explore them is to browse at the [PyTorch Hub](https://pytorch.org/hub) and use the links on the landing pages to navigate to the requisite GitHub.

# Summary

This chapter explores the ecosystem of pretrained models and model repositories for PyTorch, focusing on two major platforms: Hugging Face Hub and PyTorch Hub. While PyTorch Hub was the granddaddy that started the ball rolling, Hugging Face Hub has rapidly taken over as the go-to resource for pretrained models.

We took a look at how to use the transformers and diffusers libraries from Hugging Face, which encapsulate model loading and instantiation. With these, you have the keys to over 900,000 publicly available models. As a bonus, many of these have comprehensive documentation and model cards to get you up and running quickly and responsibly. You also got hands-on with using them, including setting up your account and getting authentication from Hugging Face using tokens.

Hugging Face APIs offer pipelines that encapsulate many of the common tasks of using models, such as tokenization and sequencing for NLP under the hood, making your coding surface much easier. We explored these with a text sentiment analysis scenario as well as another for image classification.

While PyTorch Hub has a lot less in it and accessing the models can be brittle in comparison, it’s worth looking at because it’s still well used in the research community. We looked at how to access PyTorch Vision models, prepare data for inference, and handle model outputs. The Hub also includes practical examples of using pretrained models like ResNet50 for image classification.

Ultimately, you should consider the advantages of using pretrained models, which have been built by expert researchers who have used expensive hardware and high-quality datasets that you may not otherwise have access to. To that end, you may find that using and fine-tuning existing models rather than training from scratch might be better for your scenario. We’re going to explore that over the next few chapters, starting with [Chapter 15](ch15.html#ch15_transformers_and_transformers_1748549808974580), where we will go deeper into using LLMs with Hugging Face Transformers.