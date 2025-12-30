# Chapter 19\. Using Generative Models with Hugging Face Diffusers

Over the last few chapters, we have been looking at inference on generative models and primarily using LLMs (aka text-to-text models) to explore different scenarios. However, generative AI isn’t limited just to text-based models, and another important innovation is, of course, image generation (aka text-to-image). Most image generation models today are based on a process called *diffusion*, which inspires the name *diffusers* for the Hugging Face APIs used to create images from text prompts. In this chapter, we’ll explore how diffusion models work and how to get up and running with your own apps that can generate images from prompts.

# What Are Diffusion Models?

By now, most of us have seen images that are AI created, and we’ve likely been amazed at how quickly they have grown from abstract, rough representations to near photoreal representations of what we asked for via a prompt. Because the models allow for longer prompts, with more detail, and as their training sets have grown, we’ve seen a near endless stream of improvements to what can be done with AI image generation.

But how does all of this work? It starts with the idea of diffusion.

You can start this process by creating a dataset of images and their associated noise. Consider [Figure 19-1](#ch19_figure_1_1748573005759131).

![](assets/aiml_1901.png)

###### Figure 19-1\. Noising an image

Then, once you have a set of images you’ve made noisy like this, you can train a model that learns how to denoise to get the image back to its original state. Consider the noise to be the data and the original image to be the labels. So, in the case of [Figure 19-1](#ch19_figure_1_1748573005759131), the noise on the right can be the data and the image of the puppy can be the label. At that point, you can train a model that, when it sees noise, can figure out how to turn that noise into an image. The logical extension is that you can then *generate* noise, and the model will figure out how to turn that noise into an image that will look a little bit like one of those in your training set.

But, what if you go back to the step of creating the noisy image and add text to it with a very verbose description? Then, your noisy image will have a text label (represented in embeddings) attached to it (see [Figure 19-2](#fig-19-2))!

![](assets/aiml_1902.png)

###### Figure 19-2\. Adding text encodings to the diffusion process

Now, the noisy image has the embeddings describing it attached to it. In simple terms, the piece of noise is enhanced by embeddings that describe it, so the process of denoising this image back into the original image of the puppy has the extra data to guide it in how it denoises. So, again, if you train a model with the noise plus embeddings as the data and the original images as the labels, then a model can now learn more effectively how to turn noise plus embeddings into a picture.

You probably see where this is going. Once that model is trained, then, in the future, if someone gives it a piece of text in a prompt, the text can be encoded into embeddings, a set of random noise can be generated, and the model can try to figure out how to take that random noise and denoise it, guided by the text, into an image. For all intents and purposes, it will create a whole new image as a result (see [Figure 19-3](#fig-19-3)).

![](assets/aiml_1903.png)

###### Figure 19-3\. Beginning the process of denoising an image

Here, we can start with purely random noise and a prompt. The prompt is something that likely wasn’t in the training set—there are no known images (other than AI-generated ones, of course) of teddy bears eating pizza on the surface of Mars.

So a model can then denoise this over multiple steps. As you can imagine, the very first step will be random noise, the second step will be where the model tries to get the noise to match the prompt, the third step will get it a little closer, and so on.

This is depicted in [Figure 19-4](#ch19_figure_2_1748573005759167), where you can see what the image looks like with the popular *stable diffusion* models.

![](assets/aiml_1904.png)

###### Figure 19-4\. Gradually denoising an image based on a prompt

In this case, I used a diffusion model with the prompt from [Figure 19-3](#fig-19-3) about teddy bears eating pizza. You’ll see the code for this a little later in this chapter.

In Step 0, you can see that we just have pure noise. In Step 1, the model has already started taking some of the stronger characteristics of the prompt—the surface of Mars—and given the image a very red hue. By Step 10, we have teddy bears and pizza, and by Step 40, the teddy bears are actually eating the pizza and the lighting has changed—presumably for dinnertime!

The *size* of the image depends on the model. Many earlier models, or those designed to run on consumer hardware, will generate smaller images that they will then upscale to give the desired output. The images I have shown here were created with Stable Diffusion 3.5, which creates 1024 × 1024 images by default.

###### Note

While this chapter will focus on diffusion models, using them isn’t the *only* way to generate images. There are also *autoregressive models*, which learn the mappings between the tokens for the text in the description of the image and the tokens that represent the visual contents of the image. With lots of examples of these mappings, you can train a model on them. Then, you can give the model a piece of text, and it will be able to predict the tokens for that text and reassemble them into an image.

# Using Hugging Face Diffusers

Just as Hugging Face offers a transformers library (as we explained in [Chapter 15](ch15.html#ch15_transformers_and_transformers_1748549808974580)), it also offers a diffusers library to make it easier for you to use diffusion models. Diffusers abstract the complexities of using various models into an easy-to-use API.

To get started with diffusers, you simply install them like this:

```py
pip install -U diffusers
```

The diffusers library manages the pipelining of model inference in the same way we experienced in earlier chapters with transformers. There are many steps involved in getting a model to render an image based on a prompt: encoding the prompt, making embeddings, passing the embeddings to the model along with any hyperparameters it needs, grabbing the output tensors, and turning them into an image. But diffusers encapsulate this for you into a pipeline, and there are a number of open source pipelines for many different models.

So, for example, in the image of teddy bears on Mars, I used Stable Diffusion 3.5 Medium, which you can find on the [Hugging Face website](https://oreil.ly/liUY-).

This model has limited access, so at the top of the Hugging Face page, you’ll see a form that you need to fill out to get permission. You’ll also need to configure your Hugging Face secret key in Colab (if you’re using Colab), which we demonstrated how to do back in [Chapter 14](ch14.html#ch14_using_third_party_models_and_hubs_1748549787242797).

If you aren’t using Colab, your code will need to be signed in to Hugging Face using their API. You can do this with the following code:

```py
from huggingface_hub import login

login(token="<YOUR TOKEN HERE>")
```

Once you’re signed in (or if you’re using a model that doesn’t require signing in), the process of generating an image is as follows:

1.  Create a Generator object, which allows you to specify the seed.

2.  Create an instance of the appropriate pipeline for the model you require.

3.  Send that pipeline to the appropriate accelerator.

4.  Generate the image with the pipeline, giving it the appropriate parameters.

Let’s look at this step-by-step.

First, you specify the generator using `torch.Generator`, where you will specify the accelerator for the generator and set the seed. You use the seed value to create the initial noise with a level of determinism. If you want to be able to *replicate* the images that are generated, despite the noise being random, you do so by guiding the noise with the seed. In other words, when the noise is generated with a seed value, the *same* noise will be generated subsequent times with the same seed. So effectively, the noise will be pseudo-random, as there will be a deterministic seed at play. On the other hand, if you don’t specify a seed, you’ll get a random value for it. Here’s an example:

```py
# Set your seed value
seed = 123456  # You can use any integer value you want

# Create a generator with the seed
generator = torch.Generator("cuda").manual_seed(seed)
```

Next, you’ll specify the pipeline and instantiate it with a model:

```py
pipe = StableDiffusion3Pipeline.from_pretrained(
           "stabilityai/stable-diffusion-3.5-medium", 
           torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
```

Here, we’re using Stable Diffusion 3.5, which uses the `StableDiffusion3Pipeline` class. The diffusers API is open source, with new pipelines being added all the time. You can inspect them on [GitHub](https://oreil.ly/uYGnJ).

You can also browse the different models on the [Hugging Face website](https://oreil.ly/R4dKT). Often, their landing pages will include source code about which pipeline to use and the address of the model.

Once you have the pipeline, you can use it to create an image by specifying the prompt and some other model-dependent parameters that you’ll find in the model document. So, for example, for stable diffusion, you’ll specify the number of inference steps and the generator that you specified earlier. You should also specify *where* you want the pipe to execute—(in this case, it’s `cuda`, as you can see in the previous code, which uses the GPU accelerator in Colab):

```py
image = pipe(
    "A photo of a group of teddy bears eating pizza on the surface of mars",
    num_inference_steps=40,
    generator=generator  # Add the generator here
).images
image[0].save("teddies.png")
```

I’ve found that the best way to experiment with this is to explore the pipeline’s source code and see the parameters that it supports. For example, with the Stable Diffusion 3 pipeline, there’s a *negative prompt* that dictates things that you do *not* want to see in the image. Often, you can use this to make images better. For example, you may have heard that image generators, particularly early ones, were very bad at drawing hands. You could use the negative prompt to have the image generator avoid this problem by saying “deformed hands” or something similar in that prompt.

You can also specify things you don’t want to see in the image that are more trivial! For example, every instance of the image I drew had the teddy bears eating *pepperoni* pizza. I could remove the pepperoni from this image with this code:

```py
image = pipe(
    "A photo of a group of teddy bears eating pizza on the surface of mars",
    negative_prompt="pepperoni",
    num_inference_steps=40,
    generator=generator  # Add the generator here
).images
```

The resulting image is shown in [Figure 19-5](#ch19_figure_3_1748573005759194).

The teddy on the left doesn’t look thrilled about it, but the others seem more content!

In this case, we used text-to-image to create these images—but diffusion models have become a little more advanced with add-ons for *image-to-image.* With such add-ons, instead of starting with random noise, we can begin with an existing image and then perform *inpainting*, in which we can have the model fill in new details in an existing image. We’ll explore this next.

![](assets/aiml_1905.png)

###### Figure 19-5\. Teddies that don’t like pepperoni

## Image-to-Image with Diffusers

When inspecting the source code for the pipeline, you may have discovered other classes in there, such as this one: `StableDiffusion3Img2ImgPipeline.`

As its name suggests, this class allows you to start with one image to create another. You can initialize it in a way that’s very similar to initializing the text-to-image pipeline:

```py
from diffusers import StableDiffusion3Img2ImgPipeline

# Set your seed value
seed = 123456

# Create a generator with the seed
generator = torch.Generator("cuda").manual_seed(seed)

# Load the model
pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", 
     torch_dtype=torch.bfloat16)

pipe = pipe.to("cuda")
```

Then, you’ll specify an image to use as the source image:

```py
from PIL import Image
# Load and preprocess the initial image
init_image = Image.open("puppy1.jpg").convert("RGB")
```

I’m starting with an image of a puppy (see [Figure 19-6](#ch19_figure_4_1748573005759214)).

![](assets/aiml_1906.png)

###### Figure 19-6\. Source image of a puppy

We’ll use this as the initialization image in an image-to-image pipeline with the following code. Note that the prompt is specifying a highly detailed photograph of a baby *dragon*:

```py
# Generate the image
image = pipe(
    prompt="A highly detailed photograph of a baby dragon",
    image=init_image,
    strength=0.7,
    num_inference_steps=100,
    generator=generator
).images
```

The strength parameter specifies how closely the generated image should follow the input image. At 0.0, the model won’t do anything and the output will be the input image. At 1.0, it will effectively *ignore* the input image and will just act as a text-to-image model.

Under the hood, it does this with the following process.

Given that the code specified a strength of 0.7, the model will add noise to the image until the image has had 70% of its pixels replaced by noise (and thus only 30% of the image is the original values).

The model will then run 70 denoising steps (70% of the 100 specified), which will give an image like [Figure 19-7](#ch19_figure_5_1748573005759233).

Typically, if you use strength 0.2 to 0.4, you’ll get style transfer and other minor modifications. At 0.5 to 0.7, you’ll have basic composition maintained, but major element changes, like puppy to dragon, will be seen. Above 0.8, you’ll see almost complete regeneration, but some slight influence from the original may be retained.

![](assets/aiml_1907.png)

###### Figure 19-7\. Using image-to-image to turn a puppy into a dragon

You can see that the basic pose has been maintained, but the computer has imagined a dragon to replace the puppy as required. There’s also new foreground and background, as we didn’t specify anything about them, but they’re pretty close to the originals.

As an example of a different strength level, [Figure 19-8](#ch19_figure_6_1748573005759251) shows the strength at 0.4\. We can also see that the basic shape of the puppy has been maintained, but it has become more dragon-like, with scaly skin and the beginnings of claws!

![](assets/aiml_1908.png)

###### Figure 19-8\. Strength level of 0.4 for the puppy to dragon image-to-image

This technique can be very useful in helping you create new images by starting from existing ones. I’ve seen it used in scenarios like filmmaking—where one can start with existing video that’s filmed in a basic, cheap locale but then enhanced with image-to-image frame by frame to get a different outcome. It’s a much cheaper way of doing postproduction by adding special effects!

## Inpainting with Diffusers

Another scenario that involves using diffusers that is supported by some models—including stable diffusion models—is the idea of *inpainting*, in which you can take an image and replace parts of it with AI-generated content. So, for example, consider the puppy from [Figure 19-6](#ch19_figure_4_1748573005759214) and how you can change the image so the little pooch is on the moon, as in [Figure 19-9](#ch19_figure_7_1748573005759269).

![](assets/aiml_1909.png)

###### Figure 19-9\. Using inpainting to put our puppy on the moon

You can do this by using a pattern that’s similar to the previous one. First, you’ll set up the pipeline for inpainting:

```py
from diffusers import StableDiffusion3InpaintPipeline

# Load the inpainting pipeline
pipe = StableDiffusion3InpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
```

The parameters to initialize it are the same as earlier. Next, you’ll need the generator:

```py
# Set seed for reproducibility
generator = torch.Generator("cuda").manual_seed(42)
```

Then, you’ll specify the source image, which in this case is the original image of the puppy:

```py
# Load the original image and mask
original_image = Image.open("puppy.jpg").convert("RGB")
```

The complicated step is the next one, in which you specify the *mask* for the image:

```py
mask_image = Image.open("puppymask.png").convert("L")  
```

A *mask* is simply an image that corresponds to the original one, in which pieces to be *replaced* are in white and pieces to be *preserved* are in black. [Figure 19-10](#ch19_figure_8_1748573005759284) shows the mask image used for the puppy. I like to think of this as similar to the green-screen process used in making movies. The white part of the image is the screen, and the black is the stuff that’s in front of it! The model will then replace the white with whatever you prompt it for.

![](assets/aiml_1910.png)

###### Figure 19-10\. Mask for the image

There are many ways to create masks. For this one, I used the Acorn 8 tool for the Mac. This tool gives you the ability to remove the background and paint it all in white, and then, for what’s left, it lets you select the pixels with a magic wand and paint them all in black. Every tool does this differently, so be sure to check the appropriate documentation.

Once you have the image and the mask, you can easily use the pipeline to have the model inpaint the areas that correspond to the white part of the mask. Given that the puppy is already present, I didn’t mention it in the prompt, and I just used “on the surface of the moon” to get the image in [Figure 19-9](#ch19_figure_7_1748573005759269):

```py
# Generate the inpainted image
image = pipe(
    prompt="on the surface of the moon",
    image=original_image,
    mask_image=mask_image,
    num_inference_steps=50,
    generator=generator,
    strength=0.99  # How much to inpaint the masked area
).images[0]
```

The diffusers API, as you can see, gives you a very consistent approach to managing image creation, be it directly from a text prompt, starting from a source image, or inpainting a particular area.

# Summary

In this chapter, you explored how to use generative models for image creation by using the Hugging Face diffusers library. You started by looking at the fundamental underlying concepts, seeing how the idea of denoising to create new content works.

You also looked into practical code-based implementation of image generation by using the diffusers API, and you focused on three main approaches:

1.  You explored text-to-image by converting text prompts directly into images using the Stable Diffusion 3.5 model. You also looked at how you can control this process with parameters like the seed value and the number of inference steps.

2.  You explored image-to-image by starting with an existing image and transforming it by using a prompt. In particular, you saw how the `strength` hyperparameter controls the overall transformation

3.  You explored inpainting by preserving parts of the original image by using a mask, which allows for targeted modifications while preserving some elements.

You also explored hands-on, concrete code examples of each of these approaches, which showed you how to do the pipeline setup, generator initialization, and basic parameter tuning. You also saw how *negative* prompts can help you get images closer to what you really want.

In the next chapter, you’ll look at LoRA (low-ranking adaptation), which lets you fine-tune diffusion models to achieve more controlled and customized images. LoRA is a powerful technique that allows for efficient model adaptation by only fine-tuning a small number of parameters, thus helping you guide the model toward specific styles, subjects, or artistic directions. You’ll explore how to implement LoRA with the diffusers library, and you’ll customize these models to create *specialized* image generators for your needs.