# 第二十章\. 使用 LoRA 和 Diffusers 调整生成图像模型

在第十九章中，你探讨了 diffusers 的概念以及使用扩散技术训练的模型如何根据提示生成图像。就像我们在第十六章中探讨的基于文本的模型一样，文本到图像模型也可以针对特定任务进行微调。扩散模型的结构以及如何微调它们本身就可以写成一整本书，所以在本章中，你将只对这些概念进行高层次探讨。为此，有几种技术，包括 *DreamBooth、文本反转* 以及更近期的 *低秩适应* (LoRA)，你将在本章中逐步了解。最后一种技术允许你使用非常少的数据来定制特定主题或风格的模型。

与变压器类似，Hugging Face 的 diffusers 库旨在尽可能简化使用 diffusers 以及微调它们的过程。为此，它包括预构建的脚本，您可以直接使用。

我们将使用 LoRA 和 diffusers 创建一个虚构数字影响者 Misato 的数据集的完整示例，并对其中的文本到图像模型 Stable Diffusion 2 进行微调。然后，我们将执行文本到图像推理以展示如何创建 Misato 的新图像（见图 20-1）。

![图片](img/aiml_2001.png)

###### 图 20-1\. LoRA 调整后的 Stable Diffusion 2 图像

# 使用 Diffusers 训练 LoRA

要使用 diffusers 训练 LoRA，你需要执行以下步骤。首先，你需要获取 diffusers 的源代码，以便访问其预制的训练脚本。然后，你需要获取或创建一个数据集，你可以用它来微调 Stable Diffusion。之后，你将运行训练脚本以获取模型的微调版本，将微调版本发布到 Hugging Face，并使用应用了 LoRA 层的基础模型进行推理。一旦完成，你应该能够创建像图 20-1 中显示的图像。让我们逐一了解这些步骤。

## 获取 Diffusers

要开始使用 LoRA，我发现最好的做法是首先克隆 diffusers 的源代码，以获取训练脚本。

你可以通过 git 克隆它，切换到目录，并在当前位置运行 `pip install` 来非常简单地完成这项操作：

```py
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

如果你使用 Colab 或其他托管笔记本，你将使用如下语法：

```py
!git clone https://github.com/huggingface/diffusers
%cd diffusers
!pip install .
```

这将为你提供一个本地版本的 diffusers，你可以使用。文本到图像的 LoRA 微调脚本位于 */diffusers/examples/text_to_image* 目录中，你需要安装它们的依赖项，如下所示：

```py
%cd /content/diffusers/examples/text_to_image # or whatever your dir is
!pip install -r requirements.txt
```

这些依赖关系包括加速、transformers 和 torchvision 等工具的特定版本。从源代码 git-clone 是一个好主意，这样你可以获得最新的*requirements.txt*版本，使你的生活更加轻松！

最后，你还需要 xformers 库，它旨在使 transformers 更高效，从而加快你的过程。你可以这样获取它：

```py
!pip install xformers
```

现在，你已经有一个可以用于微调的 diffusers 环境。在下一步中，你将获取数据。

## 为微调 LoRA 获取数据

你将主要在两种情况下微调 LoRA：*风格*和*主题*。在前一种情况下，你可以获取你想要的特定风格的图片，并训练模型使其能够以该风格输出。我强烈建议你在做这件事时要小心，因为许多艺术家靠他们的创作风格谋生，你应该尊重这一点。同样，你应该考虑基于商业风格训练模型的影响。不幸的是，我看到的许多在线教程都忽略了这一点，这种做法降低了 AI 的整体影响，并将生成 AI 的叙事从*创造性*推向了*窃取知识产权*。所以，请小心行事。

同样，当涉及到主题时，我看到很多教程使用名人谷歌图片搜索的例子来创建 LoRA。再次，我强烈建议你*不要*这样做。请只为那些你拥有使用其肖像权的个人创建 LoRA。

为了让你有东西可以使用，我创建了一个针对数字影响者的数据集。我称她为 Misato，这个名字来源于我喜欢的流行动漫中的角色。所有图片都是我使用流行的 Daz 3D 渲染软件渲染的。

你可以在[Hugging Face 网站](https://oreil.ly/Y1qeY)上找到这个数据集。

如果你想创建这样一个数据集，我建议你使用从多个角度拍摄的同一个人像图片，同时也要关注特定的部分。例如，你可以使用以下这些：

+   3-4 张肖像头像（护照风格照片）

+   每一边的 3-4 张四分之三头像

+   3-4 张侧面照，展示脸部侧面

+   3-4 张全身照

对于这些图片中的每一张，你还需要一个描述图片的提示词。你将在训练中使用这个提示词为图片提供上下文，并说明它应该如何被表示。

例如，考虑图 20-2，这是我为 Misato 生成的肖像照。

![](img/aiml_2002.png)

###### 图 20-2. 数据集中的 Misato 肖像照

这张图片配以下提示词：“(lora-misato-token)的高质量肖像，清晰的五官特征，中性表情，正面视角，自然光照。”

注意(lora-misato-token)的使用，其中我们指出了图像的主题。稍后，当我们创建生成新图像的提示时，我们可以使用相同的标记——例如，“(lora-misato-token)在食品广告，广告牌标志，90 年代，动漫，日本流行，日本单词，正面视角，纯背景。”这个提示将给我们图 20-3 中可以看到的内容。我们有一个全新的组合，Misato 作为快餐广告中的模特！

一旦你有一组图像，你需要创建一个*metadata.jsonl*文件，该文件包含与图像相关的提示，并采用一个标准格式，这样你就可以在微调时使用。这是一个包含文件名链接和图像提示的 JSON 文件。Misato 的文件在[Hugging Face 网站上](https://oreil.ly/MfmGh)。

![图片](img/aiml_2003.png)

###### 图 20-3. 从 LoRA 标记进行推理

*metadata.jsonl*文件的一个片段如下：

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

这基本上就是你需要的一切。对于使用 diffusers 进行训练，我发现如果你在 Hugging Face 上发布你的数据集会容易得多。为此，当你登录后，访问[Hugging Face 网站](https://oreil.ly/Ez3Gp)。在那里，你可以指定新数据集的名称以及它是否公开。完成这些后，你将能够通过网页界面上传文件（见图 20-4）。

完成这些后，你的数据集将在[*https://huggingface.co/datasets/*](https://huggingface.co/datasets/)*<你的用户名>/<数据集名称>*处可用。例如，我的用户名（见图 20-4）是“lmoroney”，数据集名称是“misato”，因此你可以在这个[*https://huggingface.co/datasets/lmoroney/misato*](https://huggingface.co/datasets/lmoroney/misato)看到这个数据集。

![图片](img/aiml_2004.png)

###### 图 20-4. 在 Hugging Face 上创建新的数据集

## 使用 Diffusers 微调模型

如前所述，当你克隆 diffusers 仓库时，你可以访问一些示例预写的脚本，这些脚本可以帮助你在各种任务上取得先机。其中之一是训练文本到图像 LoRAs。但在运行脚本之前，使用`accelerate`是一个好主意，它抽象了底层加速硬件，包括跨多个芯片的分布。使用`accelerate`，你可以定义一个配置。有关详细信息，请参阅[Hugging Face 网站](https://oreil.ly/TnaII)。

为了简化，当你使用 Colab 时，以下是设置基本`accelerate`配置的方法：

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

然后，一旦你有了这些，你可以使用`accelerate launch`来运行训练脚本。以下是一个示例：

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

注意，运行这个过程非常计算密集。使用前面的一组超参数（我稍后会解释每一个），在 Google Colab 中使用 A100 大约花费了我 2 小时（或 17 个计算单元）来训练。计算单元是需要付费的（在出版时，每个大约 10 美分），所以请确保你理解这一切是如何工作的，并且确实需要付费！

脚本使用以下超参数：

Pretrained_model_name_or_path

这可以是本地文件夹（例如，*/content/model/*）或 *huggingface.co* 上的位置——例如，[*http://huggingface.co/stabilityai/stable-diffusion-2*](http://huggingface.co/stabilityai/stable-diffusion-2) 是名为 Stable Diffusion 2 的模型的存储位置。您也可以不包含 *huggingface.co* 部分的 URL 来指定此位置。

Dataset-name

类似地，这可以是包含数据集的本地目录或 *huggingface.co* 上的地址。如您所见，我在这里使用 Misato 数据集。

Caption_column

这是 *jsonl* 文件中包含图像标题的列。您可以在此处指定标题。

Resolution

这是我们将为图像训练的分辨率。在这种情况下，它是 512 × 512。

Random_Flip

这是图像增强（如 第三章 中所述）。由于 Misato 数据集已经涵盖了多个角度，这可能不是必需的。

Train_batch_size

这是每个批次中图像的数量。最好从 1 开始，然后根据您的需要进行调整。当我使用 Colab 中的 A100 GPU 时，我注意到训练只使用了大约 7 GB 的 40 GB，因此可以安全地将其调高以加快训练速度。

Num_training_epochs

这是训练的轮数。

Checkpointing_steps

这是您应该保存检查点的频率。

Learning_rate

这是 LR 超参数。

LR_scheduler

如果您想使用可调整的学习率，您可以在此处指定调度器。可调整 LR 的好处是，训练周期后期最佳 LR 不一定与周期早期最佳 LR 相同，因此您可以在运行时进行调整。

LR_Warmup_steps

这是您将用于设置初始 LR 的步骤数。

Seed

这是一个随机种子。

Output_dir

这是在训练过程中保存检查点的位置。

然后，在训练时，您将看到类似以下的状态：

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

一旦模型训练完成，在其目录文件夹中，您将看到类似于 图 20-5 中描述的结构。

![](img/aiml_2005.png)

###### 图 20-5\. 训练后的目录

原始的 `model.safetensors` 模型被突出显示，您可以看到它的大小为 3.47 GB！另一方面，微调后的 LoRA 仅 3.4 MB。

您可以在下一步中使用此参数，将模型上传到 Hugging Face 仓库，以便推理时使用更加方便。

## 发布您的模型

您在训练过程中保存的微调目录包含比您所需多得多的信息，包括基础模型的副本。因此，如果您尝试发布和上传模型，您将花费更多时间，因为您将不得不上传大量不必要的吉字节！

因此，你应该编辑你的目录结构，从检查点目录中删除 *model.safetensors* 文件，并保留其余部分。

然后，当你登录到 Hugging Face 后，你可以访问[*huggingface.co/new*](http://huggingface.co/new)来查看“创建新模型仓库”页面（见图 20-6）。

![aiml_2006.png](img/aiml_2006.png)

###### 图 20-6\. 创建新仓库

按照步骤操作，并确保选择一个许可证。完成之后，你可以在下一步通过网页界面上传文件。完成这一步后，你应该会看到类似图 20-7 所示的屏幕，其中我给模型命名为“finetuned-misato-sd2”，因为数据是“misato”，而我调整的模型是稳定扩散 2。

你可以在[Hugging Face 网站上](https://oreil.ly/zmlal)亲自查看。

![aiml_2007.png](img/aiml_2007.png)

###### 图 20-7\. 为稳定扩散 2 定制的 Misato LoRA

现在，数据集和模型都已发布在 Hugging Face 上，使用 diffusers 进行推理非常简单。我们将在下一步中看到这一点。

## 使用自定义 LoRA 生成图像

要使用自定义 LoRA 创建图像，我们将通过一个类似于第十九章中的过程。你将使用 diffusers 创建管道，但还会添加一个调度器。在稳定扩散中，调度器的角色决定了图像如何从随机噪声演变到最终图像。并非所有调度器都与 LoRA 兼容，你必须确保你使用的调度器与你的基础模型兼容。

有很多调度器可供使用，你可以在[Hugging Face 网站上](https://oreil.ly/SUlZl)找到它们。

在这种情况下，你可以尝试使用`EulerAncestralDiscreteScheduler`：

```py
import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,  
)
```

然后，指定我们的`model_id`并选择适合它的调度器版本：

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

完成这些后，你可以从`StableDiffusionPipeline`类创建管道并将其加载到加速设备：

```py
# 2\. Load the pipeline with the chosen scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16
).to(device)
```

下一步是分配新的 LoRA 权重，这些是重新训练的层，决定了模型的新行为：

```py
# 3\. (Optional) Load LoRA weights
pipe.load_lora_weights("lmoroney/finetuned-misato-sd2")
```

稳定扩散支持**提示**和**负面提示**，其中第一个提示定义了图像中你想要的内容，第二个提示定义了你**不想要**的内容。以下是一个示例：

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

负面提示非常有用，可以帮助你避免一些与 AI 生成的视觉问题，例如变形的手和脸。

接下来是定义超参数，例如推理步骤的数量、图像的大小和种子。还有一个叫做*指导尺度*的参数，它控制着你的模型有多具想象力。指导尺度值小于 5 会给模型更多的创造性自由，但模型可能不会严格遵循你的提示。指导尺度值高于 7 会使模型更紧密地遵循你的提示，但也可能导致奇怪的伪影。中间的指导尺度值 6 在自由和遵循之间提供了一个很好的平衡。没有固定的规则，所以请随意实验：

```py
num_inference_steps = 50
guidance_scale = 6.0
width = 512
height = 512
seed = 1234567
```

然后，你只需像往常一样生成图像：

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

作为实验，你可以尝试使用具有相同超参数的不同调度器来产生类似的结果（见图 20-8）：

```py
# For DPMSolver, use:
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, 
            subfolder="scheduler", algorithm_type="dpmsolver++")
```

![](img/aiml_2008.png)

###### 图 20-8\. 使用不同调度器的相同提示和超参数

注意，图片中的文字完全是虚构的，但鉴于提示是关于广告的，语气相似。在左边的图片中，人物代表“孤独”和“不”，而在右边的图片中，它们暗示着“丈夫分居？”

最有趣的是角色的连贯性！例如，考虑图 20-9，其中 Misato 被画成了莫奈和毕加索的风格。我们可以看到，LoRA 学习到的特征足够一致，足以（大部分）在重新设计过程中幸存下来。

![](img/aiml_2009.png)

###### 图 20-9\. 不同风格下的角色一致性

这个例子使用了 Stable Diffusion 2，这是一个较老的模型，但使用 LoRA 很容易调整。随着你使用更先进的模型并对其进行调整，你可以得到更好的结果，但调整的时间和成本会更高。我建议从这样一个简单的模型开始，并专注于你的技艺。从那里，你可以逐步过渡到更先进的模型。

此外，Misato 的合成特性在 LoRA 重新训练中触发了不同的特征，导致从她低分辨率、高度合成的外观中创建出了新的图像。虽然这些图像在人类眼中几乎接近照片真实，但它们显然没有达到模型的要求，该模型学习到的 LoRA 本质上是 CGI 风格，并且分辨率低于训练集中的图像！

# 摘要

在本章中，您了解了如何通过使用 LoRA 和 diffusers 库来微调像 stable diffusion 这样的文本到图像模型。这项技术允许您通过一个小型自定义文件来定制特定主题或风格的模型。在这种情况下，您看到了如何调整 Stable Diffusion 2 以适应合成角色。在本章中，您还经历了所有步骤——从克隆 diffusers 到为它们创建一个包含完整自定义数据集的训练环境。您学习了如何使用训练脚本创建基于合成角色的新 LoRA，以及如何将其发布到 Hugging Face。最后，您看到了如何在推理时将 LoRA 应用于模型，以使用 Misato 角色的 LoRA 创建新颖的图像！
