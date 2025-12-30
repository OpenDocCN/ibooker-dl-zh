# Chapter 20\. Tuning Generative Image Models with LoRA and Diffusers

In [Chapter 19](ch19.html#ch19_using_generative_models_with_hugging_face_diffuser_1748573005765373), you explored the idea of diffusers and how models trained with diffusion techniques can generate images based on prompts. Like text-based models (as we explored in [Chapter 16](ch16.html#ch16_using_llms_with_custom_data_1748550037719939)), text-to-image models can be fine-tuned for specific tasks. The architecture of diffusion models and how to fine-tune them is enough for a full book in its own right, so in this chapter, you’ll just explore these concepts at a high level. There are several techniques for doing this, including *DreamBooth, textual inversion,* and the more recent *low-ranking adaptation* (LoRA), which you’ll go through step-by step in this chapter. This last technique allows you to customize models for a specific subject or style with very little data.

As with transformers, the diffusers Hugging Face library is designed to make using diffusers, as well as fine-tuning them, as easy as possible. To that end, it includes pre-built scripts that you can use.

We’ll go through a full sample of creating a dataset of a fictitious digital influencer called Misato, using LoRA and diffusers to fine-tune a text-to-image model called Stable Diffusion 2 for her. Then, we’ll perform text-to-image inference to demonstrate how to create new images of Misato (see [Figure 20-1](#ch20_figure_1_1748550104889464)).

![](assets/aiml_2001.png)

###### Figure 20-1\. LoRA-tuned Stable Diffusion 2 images

# Training a LoRA with Diffusers

To train a LoRA with diffusers, you’ll need to perform the following steps. First, you’ll need to get the source code for diffusers so you can have access to its premade training scripts. Then, you’ll get or create a dataset that you can use to fine-tune Stable Diffusion. After that, you’ll run the training scripts to get a fine-tune for the model, publish the fine-tune to Hugging Face, and run inference against the base model with the LoRA layers applied. Once you’re done, you should be able to create images like those shown in [Figure 20-1](#ch20_figure_1_1748550104889464). Let’s walk through each of these steps.

## Getting Diffusers

To get started with LoRA, I have found the best thing to do is to first clone the source code for diffusers to get the training scripts.

You can do this quite simply by git-cloning it, changing into the directory, and running `pip install` at the current location:

```py
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

If you’re using Colab or another hosted notebook, you’ll use syntax like this:

```py
!git clone https://github.com/huggingface/diffusers
%cd diffusers
!pip install .
```

This will give you a local version of diffusers that you can use. The text-to-image LoRA fine-tuning scripts are in the */diffusers/examples/text_to_image* directory, and you’ll need to install their dependencies like this:

```py
%cd /content/diffusers/examples/text_to_image # or whatever your dir is
!pip install -r requirements.txt
```

These dependencies include the specific versions of tools like accelerate, transformers, and torchvision. It’s good to git-clone from source so that you get the latest versions of the *requirements.txt* to make your life easier!

Finally, you’ll also need the xformers library, which is designed to make transformers more efficient and thus speed up the process for you. You can get it like this:

```py
!pip install xformers
```

Now, you have a diffusers environment that you can use for fine-tuning. In the next step, you’ll get the data.

## Getting Data for Fine-Tuning a LoRA

The two main ways in which you’ll fine-tune a LoRA are for *style* and for *subject.* In the former case, you can get a number of images of the specific style that you want and train the model so that it will output in that style. I would urge caution when doing this because many artists earn their livelihood from their style of creation, and you should respect that. Similarly, you should consider the impact of training models based on commercial styles. Unfortunately, many of the tutorials I have seen online ignore this, and such practices bring down the overall impact of AI and drive the narrative of generative AI away from being *creative* and toward *stealing IP*. So, please be careful with that.

Similarly, when it comes to the subject, I see many tutorials that use examples of doing a Google Image search for a celebrity so you can create a LoRA of them. Again, I would urge you *not* to do this. Please only create a LoRA for someone whose likeness you have permission to use.

So that you can have something you *can* use, I created a dataset for a digital influencer. I call her Misato, after my favorite character in a popular anime. All of the images were rendered by me using the popular Daz 3D rendering software.

You can find this dataset on the [Hugging Facewebsite](https://oreil.ly/Y1qeY).

If you want to create a dataset like this, I would recommend that you use images of the same figure from multiple angles that also focus on specific segments. For example, you can use these:

*   3–4 portrait headshots (passport-style photos)

*   3–4 three-quarters headshots from each side

*   3–4 profile pictures, showing the side of the face

*   3–4 full-length body shots

For each of these images, you also need a prompt that describes the image. You’ll use this in training to give context to the image and how it should be represented.

So, for example, consider [Figure 20-2](#ch20_figure_2_1748550104889522), which is a portrait shot that I generated for Misato.

![](assets/aiml_2002.png)

###### Figure 20-2\. Portrait shot of Misato from the dataset

This image is paired with the following prompt: “Photo of (lora-misato-token), high-quality portrait, clear facial features, neutral expression, front view, natural lighting.”

Note the use of (lora-misato-token), where we indicate the subject of the image. Later, when we create prompts to generate new images, we can use the same token—for example, “(lora-misato-token) in food ad, billboard sign, 90s, anime, Japanese pop, Japanese words, front view, plain background.” This prompt will give us what you can see in [Figure 20-3](#ch20_figure_3_1748550104889557). We have an entirely new composition, with Misato as the model in a fast-food campaign!

Once you have a set of images, you’ll need to create a *metadata.jsonl* file that contains the images associated with their prompts in a standard format that you can use when fine-tuning. It’s JSON with a link to the filename and the prompt for that image. The one for Misato is on the [Hugging Face website](https://oreil.ly/MfmGh).

![](assets/aiml_2003.png)

###### Figure 20-3\. Inference from a LoRA token

A snippet of the *metadata.jsonl* file is here:

```py
{ "file_name": "rightprofile-smile.png", 
               "prompt": "photo of (lora-misato-token), 
               right side profile, high quality, detailed features, 
               smiling, professional photo "}

{ "file_name": "rightprofile-neutral.png", 
               "prompt": "photo of (lora-misato-token), 
               right side profile, high quality, detailed features, 
               professional photo "}
```

That’s pretty much all you need. For training with diffusers, I’ve found it much easier if you publish your dataset on Hugging Face. To do this, when logged in, visit the [Hugging Face website](https://oreil.ly/Ez3Gp). There, you’ll be able to specify the name of the new dataset and whether or not it’s public. Once you’ve done this, you’ll be able to upload the files through the web interface (see [Figure 20-4](#ch20_figure_4_1748550104889591)).

Once you’ve done this, your dataset will be available at [*https://huggingface.co/datasets/*](https://huggingface.co/datasets/)*<yourname>/<datasetname>*. So, for example, my username (see [Figure 20-4](#ch20_figure_4_1748550104889591)) is “lmoroney,” and the dataset name is “misato,” so you can see this dataset at [*https://huggingface.co/datasets/lmoroney/misato*](https://huggingface.co/datasets/lmoroney/misato).

![](assets/aiml_2004.png)

###### Figure 20-4\. Creating a new dataset on Hugging Face

## Fine-Tuning a Model with Diffusers

As mentioned earlier, when you clone the diffusers repo, you get access to a number of example pre-written scripts that give you a head start in various tasks. One of these is training text-to-image LoRAs. But before running the script, it’s a good idea to use `accelerate`, which abstracts underlying accelerator hardware, including distribution across multiple chips. With `accelerate`, you can define a configuration. Find the details on the [Hugging Face website](https://oreil.ly/TnaII).

For the purposes of simplicity, when you’re using Colab, here’s how you can set up a basic `accelerate` profile:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

Then, once you have that, you can use `accelerate launch` to run the training script. Here’s an example:

```py
!accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
  --dataset_name="lmoroney/misato" \
  --caption_column="prompt" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=1000 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="/content/lm-misato-lora"
```

Note that running this is very computationally intensive. With the preceding set of hyperparameters (I’ll explain each one in a moment), using an A100 in Google Colab took me about 2 hours (or 17 compute units) to train. Compute units cost money (at the time of publication, about 10 cents each), so be sure to understand how this all works and that it does cost money!

The script takes the following hyperparameters:

Pretrained_model_name_or_path

This can be a local folder (for example, */content/model/*) or the location on *huggingface.co*—so for example, [*http://huggingface.co/stabilityai/stable-diffusion-2*](http://huggingface.co/stabilityai/stable-diffusion-2) is the location of the model called Stable Diffusion 2\. You can also specify this without the *huggingface.co* part of the URL.

Dataset-name

Similarly, this can be a local directory containing the dataset or the address of it on *huggingface.co*. As you can see, I’m using the Misato dataset here.

Caption_column

This is the column in the *jsonl* file that contains the caption for the images. You can specify the caption here.

Resolution

This is the resolution that we’ll train the images for. In this case, it’s 512 × 512.

Random_Flip

This is image augmentation (as in [Chapter 3](ch03.html#ch03_going_beyond_the_basics_detecting_features_in_ima_1748570891074912)). As the Misato dataset already has multiple angles covered, this probably isn’t needed.

Train_batch_size

This is the number of images per batch. It’s good to start with 1 and then tweak it as you see fit. When I was using the A100 GPU in Colab, I noticed that training was only using about 7 GB of the 40 GB, so this could be safely turned up to speed up training.

Num_training_epochs

This is how many epochs to train for.

Checkpointing_steps

This is how often you should save a checkpoint.

Learning_rate

This is the LR hyperparameter.

LR_scheduler

If you want to use an adjustable learning rate, you can specify the scheduler here. The nice thing with an adjustable LR is that the best LR later in the training cycle isn’t always the same as the best one from earlier in the cycle, so you can adjust it on the fly.

LR_Warmup_steps

This is the number of steps you’ll take to set the initial LR.

Seed

This is a random seed.

Output_dir

This is where you save the checkpoints as training happens.

Then, when training, you’ll see a status that looks something like this:

```py
Resolving data files: 100% 22/22 [00:00<00:00, 74.14it/s]
12/30/2024 19:23:48 - INFO - __main__ - ***** Running training *****
12/30/2024 19:23:48 - INFO - __main__ -   Num examples = 21
12/30/2024 19:23:48 - INFO - __main__ -   Num Epochs = 1000
12/30/2024 19:23:48 - INFO - __main__ -   Instantaneous batch size per device...
12/30/2024 19:23:48 - INFO - __main__ -   Total train batch size (w. parallel...
12/30/2024 19:23:48 - INFO - __main__ -   Gradient Accumulation steps = 1
12/30/2024 19:23:48 - INFO - __main__ -   Total optimization steps = 1000
Steps:  10% 103/1000 [05:03<44:00,  2.94s/it, lr=0.0001, step_loss=0.227]
```

Once the model is trained, in its directory folder, you’ll see a structure like the one depicted in [Figure 20-5](#ch20_figure_5_1748550104889622).

![](assets/aiml_2005.png)

###### Figure 20-5\. A trained directory

The original `model.safetensors` model is highlighted, and you can see that it is 3.47 GB in size! The fine-tuned LoRA, on the other hand, is much smaller at just 3.4 MB.

You can use this in the next step, where you upload the model to the Hugging Face repository to make it very easy for inference to use it.

## Publishing Your Model

The fine-tuned directory that you’ve saved while training contains a lot more information than you need, including clones of the base model. As a result, if you try to publish and upload the model, you’ll end up taking a lot longer because you’ll have to upload lots of unneeded gigabytes!

Therefore, you should edit your directory structure to remove the *model.safetensors* files from the checkpoint directories and keep the rest.

Then, when you’re signed into Hugging Face, you can visit [*huggingface.co/new*](http://huggingface.co/new) to see the “Create New Model Repository” page (see [Figure 20-6](#ch20_figure_6_1748550104889653)).

![](assets/aiml_2006.png)

###### Figure 20-6\. Creating a new repository

Follow the steps, and be sure to select a license. Then, when you’re done, you can upload the files via the web interface in the next step. When you’re done with that, you should see something like the screen depicted in [Figure 20-7](#ch20_figure_7_1748550104889686), where I named the model “finetuned-misato-sd2,” given that the data was “misato” and the model I tuned was Stable Diffusion 2.

You can see this for yourself on the [Hugging Face website](https://oreil.ly/zmlal).

![](assets/aiml_2007.png)

###### Figure 20-7\. The fine-tuned Misato LoRA for Stable Diffusion 2

Now that the dataset and the model are both published on Hugging Face, using diffusers to do an inference with it is super simple. We’ll see that in the next step.

## Generating an Image with the Custom LoRA

To create an image using the custom LoRA, we’ll go through a process that’s similar to the one in [Chapter 19](ch19.html#ch19_using_generative_models_with_hugging_face_diffuser_1748573005765373). You’ll use diffusers to create a pipeline, but you’ll also add a scheduler. In stable diffusion, the role of the scheduler determines how the image evolves from random noise to the final image. Not all schedulers work with LoRA, and you’ll have to ensure that the scheduler you use works with the base model you’re working with.

There are lots of schedulers you can use, and you can find them on the [Hugging Face website](https://oreil.ly/SUlZl).

In this case, you can experiment with using the `EulerAncestralDiscreteScheduler`:

```py
import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,  
)
```

Then, specify our `model_id` and pick the appropriate version of the scheduler for it:

```py
model_id = "stabilityai/stable-diffusion-2"

# Choose your device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1\. Pick your scheduler
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    model_id,
    subfolder="scheduler"
)
```

Once you’ve done that, you can create the pipeline from the `StableDiffusionPipeline` class and load it to the accelerator device:

```py
# 2\. Load the pipeline with the chosen scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16
).to(device)
```

The next step is to assign the new LoRA weights, which are the retrained layers that determine the new behavior of the model:

```py
# 3\. (Optional) Load LoRA weights
pipe.load_lora_weights("lmoroney/finetuned-misato-sd2")
```

Stable diffusion supports both a prompt *and* a negative prompt, where the first prompt defines what you want in the image and the second prompt defines what you *do not* want. Here’s an example:

```py
# 4\. Define prompts and parameters
prompt = "(lora-misato-token) in food ad, billboard sign, 90s, anime, 
         japanese pop, japanese words, front view, plain background"

negative_prompt = (
    "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, 
     wrong anatomy, "
    "extra limb, missing limb, floating limbs, (mutated hands and 
     fingers:1.4), "
    "disconnected limbs, mutation, mutated, ugly, disgusting, blurry, 
     amputation"
)
```

The negative prompt is very useful in helping you avoid some of the issues with AI-generated visuals, such as deformed hands and faces.

Next up is to define the hyperparameters, such as the number of inference steps, the size of the image, and the seed. There’s also a parameter called *guidance scale*, which controls how imaginative your model is. A guidance scale value of less than 5 gives the model more creative freedom, but the model may not follow your prompt closely. A guidance scale value that’s higher than 7 will make the model adhere more strongly to your prompt, but it can also lead to strange artifacts. The guidance scale value in the middle—6—is a nice balance between freedom and adherence. There’s no hard and fast rule, so feel free to experiment:

```py
num_inference_steps = 50
guidance_scale = 6.0
width = 512
height = 512
seed = 1234567
```

Next, you just generate the image as usual:

```py
# 5\. Create a generator for reproducible results
generator = torch.Generator(device=device).manual_seed(seed)

# 6\. Run the pipeline
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]

# 7\. Save the result
image.save("lora-with-negative.png")
```

As an experiment, you can try using a different scheduler with the same hyperparameters to yield similar results (see [Figure 20-8](#ch20_figure_8_1748550104889721)):

```py
# For DPMSolver, use:
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, 
            subfolder="scheduler", algorithm_type="dpmsolver++")
```

![](assets/aiml_2008.png)

###### Figure 20-8\. The same prompt and hyperparameters with different schedulers

Note that the text in the image is entirely made up, but given that the prompt is about advertisements, the tone is similar. In the picture on the left, the characters represent “loneliness” and “no,” while in the image on the right, they suggest “husband split?”

What’s most interesting is the consistency in the character! For example, consider [Figure 20-9](#ch20_figure_9_1748550104889755), in which Misato was painted in the styles of Monet and Picasso. We can see that the features learned by LoRA were consistent enough to (mostly) survive the restyling process.

![](assets/aiml_2009.png)

###### Figure 20-9\. Character consistency across styles

This example used Stable Diffusion 2, which is an older model but one that’s easy to tune with LoRA. As you use more advanced models and tune them, you can get much better results, but the time and costs of tuning will be much higher. I’d recommend starting with a simpler model like this one and working on your craft. From there, you can build up to the more advanced models.

Additionally, Misato’s synthetic nature has triggered different features in the LoRA retraining, leading to the new images that have been created from her having a low-res, highly synthetic look. While the images have been close to photoreal to the human eye, they clearly haven’t been to the model, which learned a LoRA that was very CGI in nature and lower resolution than the ones in the training set!

# Summary

In this chapter, you had a walk-through of how to fine tune a text-to-image model like stable diffusion by using LoRA and the diffusers library. This technique allows you to customize models for a specific subject or style with a small custom file. In this case, you saw how to tune Stable Diffusion 2 for a synthetic character. In this chapter, you also went through all the steps—from cloning diffusers to creating a training environment for them that included a fully custom dataset. You learned how to use the training scripts to create a new LoRA based on the synthetic character and how to publish that to Hugging Face. Finally, you saw how to apply the LoRA to the model at inference time to create novel images using the LoRA for the Misato character!