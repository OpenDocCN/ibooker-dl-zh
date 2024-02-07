# 第2章。介绍TensorFlow.js

> “如果你的行动激励他人梦想更多，学习更多，
> 
> 做更多，成为更多，你就是一个领导者。”
> 
> —约翰·昆西·亚当斯

我们已经稍微谈到了TensorFlow.js以及它的功能，但我们还没有真正深入探讨像TensorFlow.js这样的机器学习框架到底是什么。在本章中，我们将探讨机器学习框架的概念，然后迅速进入编写代码。我知道编写具有某种实际结果的代码很重要，所以在本章中，你最终将让你的计算机运行TensorFlow.js并产生结果。

我们将：

+   看看TensorFlow.js的概念

+   设置TensorFlow.js

+   运行一个TensorFlow.js模型包

+   深入了解AI的工作原理

让我们从我们将使用的框架开始。

# 你好，TensorFlow.js

考虑到我们之前的章节讨论了古代哲学和机器学习作为一个领域的诞生，你会期望人工智能框架的历史可以追溯到上世纪60年代初。然而，人工智能长时间停滞不前，这段时间通常被称为“人工智能寒冬”。人工智能的概念受到怀疑和极端数学计算的困扰，因为当时可用的数据量很小。谁能责怪这些研究人员呢？今天大多数软件开发人员依赖于发布应用程序，而不是从头开始编写支持GPU的线性代数和微积分，构建自己的人工智能不应该是例外。幸运的是，由于谷歌Brain团队的一些开源贡献，我们有了选择。

当你开始学习机器学习时，会听到很多流行词。TensorFlow、TensorFlow Lite和TensorFlow.js都可能被提到，对于大多数新手来说，这些术语的含义以及为什么会有三个都不清楚。现在，让我们暂时忽略“张量”这个术语，因为你在[第1章](ch01.html#the_chapter_1)中已经听过这个词，而且在接下来的章节中你会真正理解它。相反，让我们专注于定义TensorFlow.js，以便我们可以使用它。

TensorFlow，没有任何额外的“.js”或“Lite”，是谷歌的第一个公开的机器学习框架；谷歌Brain团队于2015年底发布了它。这个框架专注于用Python在云端有效解决谷歌的机器学习问题。谷歌很快意识到将这个流行的框架推广到计算能力有限的物联网和移动设备上会有好处，这就需要对TensorFlow进行适应，这就是所谓的TensorFlow Lite。这一成功的适应为将TensorFlow理念推广到其他语言铺平了道路。

你可能可以猜到接下来会发生什么。2018年初，谷歌宣布了一个由谷歌支持的JavaScript导入机器学习框架TensorFlow的版本，称为TensorFlow.js。这一新举措以全新的方式增强了TensorFlow的实用性。Daniel Smilkov、Nikhil Thorat和Shanqing Cai是发布TensorFlow.js的团队的一部分。在[TensorFlow开发者峰会](https://youtu.be/YB-kfeNIPCE)上，Smilkov和Thorat使用计算机视觉和网络摄像头训练一个模型来控制*吃豆人*游戏。

正是在这一刻，“仅限Python”的选项被从流行的人工智能框架选项中移除，神经网络可以有效地穿越JavaScript领域。*如果你可以运行JavaScript，你就可以运行由TensorFlow.js ML支持的人工智能。*

这三种实现今天都是活跃的，并随着它们特定目的的增长。通过将TensorFlow扩展到JavaScript实现，我们现在可以在节点服务器甚至客户端浏览器中实现AI/ML。在论文“TensorFlow.js: 用于Web和更多的机器学习”中[(Daniel Smilkov等人，2019)](https://oreil.ly/XkIjZ)，他们表示，“TensorFlow.js使来自庞大JavaScript社区的新一代开发人员能够构建和部署机器学习模型，并实现新类的设备上计算。” TensorFlow.js可以利用广泛的设备平台，同时仍然可以访问GPU甚至Web Assembly。有了JavaScript，我们的机器学习可以涉足地平线并回来。

值得注意的是，在几项基准测试中，Node在较低的CPU负载下胜过了Python 3，因此尽管Python一直是大多数AI的采用语言，JavaScript作为产品和服务的主要语言平台。

但没有必要删除或推广任何一种语言。TensorFlow模型基于有向无环图（DAG），这是与语言无关的图，是训练的*输出*。这些图可以由一种语言训练，然后转换并被完全不同的编程语言消耗。本书的目标是为您提供使用JavaScript和TensorFlow.js的工具，以便充分利用它们。

# 利用TensorFlow.js

对于很多人来说，“学习”有时可能意味着从基础开始，这意味着从数学开始。对于这些人来说，像TensorFlow这样的框架和像TensorFlow.js这样的实用分支是一个糟糕的开始。在本书中，我们将构建项目，并涉及TensorFlow.js框架的基础知识，我们将很少或根本不花时间在底层数学魔法上。

像TensorFlow和TensorFlow.js这样的框架帮助我们避免涉及的线性代数的具体细节。您不再需要关注*前向传播*和*反向传播*这样的术语，以及它们的计算和微积分。相反，我们将专注于像*推断*和*模型训练*这样的行业术语。

虽然TensorFlow.js可以访问底层API（如`tfjs-core`）来对经典问题进行一些基本优化，但这些时刻留给了那些无论手头的框架如何都有坚实基础的学者和高级用户。这本书旨在展示TensorFlow.js的强大之处，利用框架的辛勤工作和优化是我们将如何做到这一点。我们让TensorFlow.js负责配置和优化我们的代码，以适应各种设备约束和WebGL API。

我们甚至可能走得太远，将机器学习应用于您可以轻松手工编码的算法，但这通常是大多数人真正理解概念的地方。用机器学习解决您理解的简单问题有助于您推断解决您无法手工编码的高级问题的步骤、逻辑和权衡。

另一方面，一些关于神经元、激活函数和模型初始化的基础知识是不能被忽视的，可能需要一些解释。本书的目标是为您提供理论和实用性的健康平衡。

正如您可能已经推测到的那样，TensorFlow.js的各种平台意味着没有单一的预设设置。我们可以在本书中在客户端或服务器上运行TensorFlow.js。然而，我们最隐性的交互选项是充分利用浏览器。因此，我们将在浏览器中执行大部分示例。当然，在适当的情况下，我们仍将涵盖托管节点服务器解决方案的关键方面。这两种工具都有各自的优缺点，我们将在探索TensorFlow.js的强大之处时提到。

# 让我们准备好TensorFlow.js

与任何流行工具一样，您可能会注意到TensorFlow.js包有几种不同版本，以及几个可以访问代码的位置。本书的大部分内容将专注于TensorFlow.js最常用和“准备运行”的版本，即浏览器客户端。优化的框架版本是为服务器端制作的。这些版本与Python使用相同的底层C++核心API进行通信，但通过Node.js，这使您能够利用服务器的图形卡或CPU的所有性能。TensorFlow.js AI模型在各种位置运行，并利用各种环境的各种优化（请参见[图2-1](#flavors)）。

![Tensorflow选项图表](assets/ltjs_0201.png)

###### 图2-1。TensorFlow.js的选项

本书中学到的知识可以应用于大多数平台。为了方便起见，我们将覆盖最常见平台的设置过程。如果您不愿意从头开始设置环境，可以直接访问与本书相关的源代码中为您构建的预配置项目，位于[*https://github.com/GantMan/learn-tfjs*](https://github.com/GantMan/learn-tfjs)。

# 在浏览器中设置TensorFlow.js

让我们来看看运行TensorFlow.js的最快、最多功能和最简单的方法。要在浏览器中运行TensorFlow.js，实际上非常容易。我假设您熟悉JavaScript的基础知识，并且以前已经将JavaScript库导入到现有代码中。TensorFlow.js支持多种包含方式，因此任何经验的开发人员都可以访问它。如果您熟悉包含JavaScript依赖项，您将熟悉这些常见做法。我们可以以两种方式将TensorFlow.js导入到页面中：

+   使用NPM

+   包含脚本标签

## 使用NPM

管理网站依赖项的最流行方式之一是使用包管理器。如果您习惯使用NPM或Yarn构建项目，您可以通过NPM注册表访问代码[*https://oreil.ly/R2lB8*](https://oreil.ly/R2lB8)。只需在命令行安装依赖项：

```py
# Import with npm
$ npm i @tensorflow/tfjs

# Or Yarn
$ yarn add @tensorflow/tfjs
```

导入`tfjs`包后，您可以在JavaScript项目中使用以下ES6 JavaScript导入代码导入此代码：

```py
import * as tf from '@tensorflow/tfjs';
```

## 包含脚本标签

如果网站不使用包管理器，您可以简单地向HTML文档添加一个脚本标签。这是您可以在项目中包含TensorFlow.js的第二种方式。您可以下载并在本地托管TensorFlow.js，或者利用内容传送网络（CDN）。我们将把脚本标签指向CDN托管的脚本源：

```py
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.7.0/dist/tf.min.js">
</script>
```

除了跨网站缓存外，CDN非常快，因为它们利用边缘位置确保全球快速交付。

###### 注意

您可能已经注意到，我已将此代码锁定到特定版本（2.7.0），我强烈建议您在涉及CDN的项目中始终这样做。您不希望在网站出现自动破坏性更改的问题。

# 在Node中设置TensorFlow.js

我们在浏览器中使用的TensorFlow.js包与Node.js完全兼容，如果您计划仅暂时尝试Node.js，这是一个很好的解决方案。一个好的规则是，如果您不打算为他人托管实时项目或在大量数据上进行训练，则使用简单的`/tfjs`而不是`/tfjs-node`导入。

如果您的目标是超越实验，并在TensorFlow.js中实现有效的Node.js，您应该花一些时间改进您的Node.js设置，使用一些[这些替代包](https://oreil.ly/zREQy)。有两个更好的TensorFlow.js分发版本专门为Node和速度构建。它们是`tfjs-node`和`tfjs-node-gpu`。请记住，每台开发者机器都是独特的，您的安装和体验可能有所不同。

对于Node.js，您可能需要在`@tensorflow/tfjs-node`和`@tensorflow/tfjs-node-gpu`之间进行选择。如果您的计算机配置了NVIDIA GPU并正确设置了CUDA软件，您可以使用后者GPU加速的软件包。Compute Unified Device Architecture（CUDA）允许通过并行计算平台直接访问GPU加速的NVIDIA硬件。虽然GPU软件包是TensorFlow.js选项中绝对最快的，但由于其硬件和软件约束，它也是最不可能准备好并配置好的软件包。目前，我们的示例将在安装`tfjs-node`上运行，并将可选的CUDA配置留给您。

```py
# Import with npm
$ npm i @tensorflow/tfjs-node

# Or Yarn
$ yarn add @tensorflow/tfjs-node
```

###### 警告

通常，如果您的计算机尚未设置为开发高级C++库，您可能需要安装一些软件来准备好您的计算机。只有当您希望积极使用`tfjs-node`或`tfjs-node-gpu`时，才需要进行这种深入研究。

如果您的NPM安装成功，恭喜！您已准备好从此包中导入。如果您已经设置了Node来处理ES6，您可以使用以下代码导入：

```py
import * as tf from '@tensorflow/tfjs-node';
```

如果您尚未配置Node.js软件包以处理ES6导入，您仍然可以使用经典的require访问代码：

```py
const tf = require('@tensorflow/tfjs-node');
```

# 验证TensorFlow.js是否正常工作

之前的所有方法都会在您的JavaScript代码中提供一个名为`tf`的变量，这使您可以访问TensorFlow.js。为了确保我们的导入工作正常，让我们记录导入的TensorFlow.js库的版本。

将此代码添加到您的JavaScript中，如果在控制台中看到一个版本打印出来，您的导入就可以继续进行了！

```py
console.log(tf.version.tfjs);
```

运行页面时，我们可以右键单击页面并检查以访问JavaScript控制台日志。在那里，我们将找到我们的日志命令的输出，“3.0.0”或您导入的TensorFlow.js版本。对于Node.js示例，该值将直接在控制台中打印。

###### 警告

在访问`tf`变量（TensorFlow.js库）的功能之前，通常需要确保TensorFlow.js已经正确加载了后端并准备就绪。上述代码绕过了这个检查，但始终最好运行您的初始代码等待`tf.ready()`的承诺。

## 下载并运行这些示例

如[第1章](ch01.html#the_chapter_1)中所述，您可以访问本书中的代码。为了确保您不必在每个示例中从头开始设置这些项目，请确保您拥有每个项目的源代码，包括之前显示的简单代码。

从书的存储库中以您喜欢的方式下载项目：[*https://github.com/GantMan/learn-tfjs*](https://github.com/GantMan/learn-tfjs)。

转到第2章的目录，并确保您可以在您的计算机上运行代码。

### 运行简单示例

在*chapter2/simple/simplest-example*中，我们避免使用NPM，只是从CDN中拉取我们的代码。以当前结构化的方式，我们甚至不需要托管网站！我们只需在任何现代浏览器中打开*index.html*，它就会运行！

在某个时候，我们实际上需要托管这些简单示例，因为我们将访问需要完整URI的其他资产。我们可以通过使用一个小型的Web服务器来托管文件来轻松实现这一点。我知道的最小的Web服务器叫做Web服务器，有一个有趣的手绘“200 OK！”标志。在五分钟内，我们就可以在本地服务器上正确提供我们的文件。

您可以在[Chrome Web Store作为扩展](https://oreil.ly/ZOedW)上找到Chrome的Web服务器。在本书中，我们有时会称此插件为“200 OK！”当您将Web服务器指向*index.html*文件时，它将自动为您提供文件，并且所有相邻文件都可以通过其关联的URL访问，正如我们将在后续课程中所需的那样。应用程序界面应该看起来像[图2-3](#two_hundred)。

![200 OK！对话框](assets/ltjs_0203.png)

###### 图2-3。Chrome的Web服务器200 OK！对话框

如果您想查看其他选项或想要链接到提到的Chrome插件，请查看*chapter2/extra/hosting-options.md*，找到适合您的选项。当然，如果您发现一个未列出的绝妙选项，请贡献一个拉取请求。

一旦找到一个以您喜欢的方式运行*simple-example*的服务器，您可以将该服务用于以后的所有简单选项。

### 运行NPM web示例

如果您更熟悉NPM，此项目的基本NPM示例使用Parcel。Parcel是最快的应用程序捆绑工具，零配置。它还包括热模块重新加载，以获取实时更新和出色的错误日志记录。

要运行代码，请导航至*chapter2/web/web-example*并执行NPM安装（`npm i`）。完成后，在*package.json*中有一个脚本可以启动所有内容。您只需运行启动脚本：

```py
$ npm run start
```

就是这样！我们将使用这种方法来运行本书中所有基于NPM的代码。

### 运行Node.js示例

Node.js示例与Parcel NPM示例一样易于运行。虽然Node.js通常没有明确的意见，但本书中的Node.js示例将包括一些明确的开发依赖项，以便我们可以使我们的Node.js示例代码与浏览器示例保持一致。本书中的代码将充分利用ECMAScript。我们通过一些转译、文件监视和节点魔法来实现这一点。

为了准备这个示例，请导航至*chapter2/node-example*并执行NPM安装（`npm i`）。如果遇到任何问题，您可能需要运行`npm i -g ts-node nodemon node-gyp`来确保您拥有所需的库，以确保我们所有的魔法发生。一旦您的节点包正确放置，您可以随时通过运行启动脚本来启动项目：

```py
$ npm run start
```

代码通过TypeScript转译，并且使用`nodemon`进行重新加载。如果一切正常运行，您将在运行服务器的控制台/终端中直接看到已安装的TensorFlow.js版本。

# 让我们使用一些真实的TensorFlow.js

现在我们有了TensorFlow.js，让我们用它来创造一些史诗般的东西！好吧，这有点简化：如果那么容易，这本书就结束了。仍然有很多东西要学习，但这并不妨碍我们乘坐缆车，以获得高层视角。

TensorFlow.js有大量预先编写的代码和模型可供我们利用。这些预先编写的库帮助我们获得利用TensorFlow.js的好处，而无需完全掌握底层概念。

虽然有很多社区驱动的模型效果很好，但TensorFlow.js模型的官方维护列表在TensorFlow GitHub上，名称为`tfjs-models`。为了稳定性，我们将尽可能经常在本书中使用这些模型。您可以在这里查看链接：[*https://github.com/tensorflow/tfjs-models*](https://github.com/tensorflow/tfjs-models)。

在这次尝试运行实际TensorFlow.js模型时，让我们选择一个相对简单的输入和输出。我们将使用TensorFlow.js的*Toxicity*分类器来检查文本输入是否具有侮辱性。

## 毒性分类器

谷歌提供了几个不同复杂度的“即插即用”模型。其中一个有益的模型被称为毒性模型，这可能是对初学者来说最直接和有用的模型之一。

像所有编程一样，模型将需要特定的输入并提供特定的输出。让我们开始看看这个模型的输入和输出是什么。毒性检测有毒内容，如威胁、侮辱、咒骂和普遍仇恨。由于这些并不一定是互斥的，因此每种违规行为都有自己的概率是很重要的。

毒性模型试图识别给定输入是否符合以下特征的真假概率：

+   身份攻击

+   侮辱

+   淫秽

+   严重毒性

+   性暴力

+   威胁

+   毒性

当您给模型一个字符串时，它会返回一个包含七个对象的数组，用于识别每个特定违规行为的概率预测百分比。百分比表示为两个介于零和一之间的`Float32`值。

如果一句话肯定*不*是违规行为，概率将主要分配给`Float32`数组中的零索引。

例如，`[0.7630404233932495, 0.2369595468044281]`表示对于这种特定违规行为的预测是76%不是违规行为，24%可能是违规行为。

对于大多数开发人员来说，这可能是一个“等等，什么！？”的时刻。在我们习惯于真和假的地方得到概率，这有点奇怪，不是吗？但直观地，我们一直知道语言有很多灰色地带。侮辱的确切科学往往取决于个人，甚至是当天！

因此，该模型具有一个额外功能，允许您传递一个阈值，当特定违规行为超过分配的限制时将其识别出来。当检测到超过阈值的侮辱时，`match`标志将设置为true。这是一个很好的额外功能，可以帮助您快速映射重大违规行为的结果。选择有效的阈值取决于您的需求和情况。您可以凭直觉行事，但如果需要一些指导，统计学有各种工具供您查阅。阅读有关接收器操作特性（ROC）图的文章，以绘制和选择适合您需求的最佳阈值。

###### 警告

要激活毒性模型，我们将不得不写一些侮辱性的话。以下示例使用基于外表的侮辱。这个侮辱避免使用粗话，但仍然是冒犯性的。这并不针对任何特定人，而是旨在说明AI理解和识别有毒评论的能力。

选择一个对人类容易识别但对计算机难以识别的侮辱是很重要的。在文本形式中检测讽刺是困难的，并且一直是计算机科学中的一个主要问题。为了严肃测试这个模型，侮辱应避免使用常见和明显的煽动性措辞。将阈值设置为`0.5`，在特别狡猾的侮辱上运行毒性模型会产生示例2-1中显示的数组。

侮辱输入：“她看起来像一个穴居人，只是远不如智慧！”

##### 示例2-1。输入句子的完整毒性报告

```py
[{
      "label":"identity_attack",
      "results":[{
            "probabilities":{
               "0":0.9935033917427063,
               "1":0.006496586836874485
            }, "match":false
         }]
   },{
      "label":"insult",
      "results":[{
            "probabilities":{
               "0":0.5021483898162842,
               "1":0.4978516101837158
            }, "match":false
         }]
   },{
      "label":"obscene",
      "results":[{
            "probabilities":{
               "0":0.9993441700935364,
               "1":0.0006558519671671093
            }, "match":false
         }]
   },{
      "label":"severe_toxicity",
      "results":[{
            "probabilities":{
               "0":0.9999980926513672,
               "1":0.0000018614349528434104
            }, "match":false
         }]
   },{
      "label":"sexual_explicit",
      "results":[{
            "probabilities":{
               "0":0.9997043013572693,
               "1":0.00029564235592260957
            }, "match":false
         }]
   },{
      "label":"threat",
      "results":[{
            "probabilities":{
               "0":0.9989342093467712,
               "1":0.0010658185929059982
            }, "match":false
         }]
   },{
      "label":"toxicity",
      "results":[{
            "probabilities":{
               "0":0.4567308723926544,
               "1":0.543269157409668
            }, "match":true
         }]
}]
```

正如您从[示例2-1](#output_full_toxicity)中可以看到的，我们在“侮辱”雷达下勉强通过（50.2%错误），但我们被毒性指标扣分，导致`"match": true`。这相当令人印象深刻，因为我在句子中没有任何明确的冒犯性语言。作为程序员，编写一个算法来捕捉和识别这种有毒的侮辱并不直接，但AI经过研究大量标记的侮辱后，被训练来识别有毒语言的复杂模式，这样我们就不必自己做了。

前面的示例使用一个句子的数组作为输入。如果将多个句子作为输入，您的句子索引将直接对应于每个类别的结果索引。

但不要只听我的话；现在轮到您运行代码了。您可以通过以下方式将模型添加到您的网站：

```py
$ npm install @tensorflow-models/toxicity
```

然后导入库：

```py
import * as toxicity from "@tensorflow-models/toxicity";
```

或者您可以直接从CDN添加脚本。^([3](ch02.html#idm45049252373352)) 脚本标签的顺序很重要，所以确保在尝试使用模型之前将标签放在页面上：

```py
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/toxicity@1.2.2">
</script>
```

前面的任何示例都将在一个准备就绪的`toxicity`变量中提供结果。我们将使用这个变量的`load`方法来加载ML模型的承诺。然后，我们可以利用该模型在一个句子数组上使用`classify`方法。

以下是加载模型并对三个句子进行分类的示例。这个确切的示例可以在[GitHub上的章节代码](https://oreil.ly/sTs5a)的相关部分中以三种不同形式找到。

```py
// minimum positive prediction confidence // If this isn't passed, the default is 0.85 constthreshold=0.5;// Load the model ![1](assets/1.png)toxicity.load(threshold).then((model)=>{constsentences=["You are a poopy head!","I like turtles","Shut up!"];// Ask the model to classify inputs ![2](assets/2.png)model.classify(sentences).then((predictions)=>{// semi-pretty-print results
console.log(JSON.stringify(predictions,null,2));![3](assets/3.png)});});
```

[![1](assets/1.png)](#co_introducing_tensorflow_js_CO1-1)

模型加载到浏览器中并带有阈值。

[![2](assets/2.png)](#co_introducing_tensorflow_js_CO1-2)

加载的模型被要求对输入进行分类。

[![3](assets/3.png)](#co_introducing_tensorflow_js_CO1-3)

使用JavaScript对象表示法很好地打印了对象。

###### 注意

如果您在浏览器中运行此代码，您需要查看控制台以查看输出。您可以通过检查页面导航到控制台，或者通常情况下，您可以在Windows上按Control+Shift+J或在Mac上按Command+Option+J。如果您使用`npm start`从命令行运行此代码，您应该立即在控制台中看到输出。

多个句子的结果按毒性类别分组。因此，前面的代码尝试根据每个类别识别每个句子。例如，前面的“侮辱”输出应该类似于[示例2-2](#output_insult_result)。

##### 示例2-2\. 侮辱部分结果

```py
  ...
  {
    "label": "insult",
    "results": [
      {
        "probabilities": {
          "0": 0.05905626341700554,
          "1": 0.9409437775611877
        },
        "match": true
      },
      {
        "probabilities": {
          "0": 0.9987999200820923,
          "1": 0.0012000907445326447
        },
        "match": false
      },
      {
        "probabilities": {
          "0": 0.029087694361805916,
          "1": 0.9709123373031616
        },
        "match": true
      }
    ]
  },
  ...
```

哒哒哒！代码运行得很好。每个`results`索引对应于输入句子索引，并且正确诊断了三个句子中的两个侮辱。

祝贺您运行您的第一个TensorFlow.js模型。现在您是AI的大师，让我们一起讨论这个库的步骤和基本概念。

## 加载模型

当我们调用`toxicity.load`时，您可能会认为模型被加载到内存中，但您只对一半正确。大多数这些库不会在JavaScript代码库中提供经过训练的模型。再读一遍那句话。这对于我们的NPM开发人员可能有点令人担忧，但对于我们的CDN用户来说完全合理。加载方法触发一个网络调用来下载库使用的模型。在某些情况下，加载的模型会针对JavaScript所在的环境和设备进行优化。查看[图2-4](#model_download)中说明的网络日志。

![毒性下载](assets/ltjs_0204.png)

###### 图2-4\. 网络下载请求

###### 警告

尽管毒性NPM捆绑包可以被缩小并压缩到仅2.4 KB，但在使用库时，实际模型文件在网络上有额外的多兆字节负载。

这个毒性库的加载方法需要一个阈值，它将应用于所有后续分类，然后触发一个网络调用来下载实际的模型文件。当模型完全下载时，库会将模型加载到张量优化内存中供使用。

适当评估每个库是很重要的。让我们回顾一些人们在学习更多关于这个库时常问的常见问题。

## 分类

我们的毒性代码接下来做的事情是运行`classify`方法。这是我们的输入句子通过模型传递的时刻，我们得到了结果。虽然它看起来就像任何其他JavaScript函数一样简单，但这个库实际上隐藏了一些必要的基本处理。

模型中的所有数据都被转换为张量。我们将在[第3章](ch03.html#the_chapter_3)中更详细地介绍张量，但重要的是要注意，这种转换对于AI至关重要。所有输入字符串都被转换，进行计算，得到的结果是重新转换为普通JavaScript基元的张量。

很高兴这个库为我们处理了这个问题。当您完成本书时，您将能够以相同的灵活性包装机器学习模型。您将能够让您的用户对发生在幕后的数据转换的复杂性保持幸福的无知。

在下一章中，我们将深入探讨这种转换。您将完全掌握数据转换为张量以及随之而来的所有数据操作超能力。

# 自己试试

现在你已经实现了一个模型，很可能你可以实现[谷歌提供的其他模型](https://oreil.ly/WFq62)。大多数其他谷歌模型的 GitHub 页面都有 README 文档，解释如何实现每个库。许多实现与我们在毒性中看到的类似。

花点时间浏览现有的模型，让你的想象力发挥得淋漓尽致。你可以立即开始使用这些库进行工作。随着你在本书中的进展，了解这些模型的存在也会很有用。你不仅将更好地理解这些库的能力，还可能想要结合甚至改进这些现有库以满足你的需求。

在下一章中，我们将开始深入挖掘这些包装良好的库隐藏的细节，以便无限释放你的 TensorFlow.js 技能。

# 章节复习

我们通过几种常见的实践选项为 TensorFlow.js 设置了你的计算机。我们确保我们的机器已经准备好运行 TensorFlow.js，甚至下载并运行了一个打包好的模型来确定文本毒性。

## 章节挑战：卡车警报！

花点时间尝试一下[MobileNet 模型](https://oreil.ly/fUKoy)，它可以查看图像并尝试对主要物件进行分类。这个模型可以传递任何 `<img>`、`<video>` 或 `<canvas>` 元素，并返回对该特定图形中所见内容的最有可能预测的数组。

MobileNet 模型已经经过训练，可以对[1,000种可能的物品](https://oreil.ly/6PEAn)进行分类，从石墙到垃圾车，甚至是埃及猫。人们已经使用这个库来检测各种有趣的事物。我曾经看到一些代码将网络摄像头连接到 MobileNet 来[检测羊驼](https://oreil.ly/L0nBz)。

对于这个章节挑战，你的任务是创建一个可以检测卡车的网站。给定一个输入图像，你要识别它是否是一辆卡车。当你从照片中检测到一辆卡车时，执行 `alert("检测到卡车！")`。默认情况下，MobileNet 包返回前三个检测结果。如果这三个中有任何一个在照片中看到卡车，你的警报应该像[图 2-5](#truck_detected)中一样通知用户。

![带有活动警报的卡车检测器](assets/ltjs_0205.png)

###### 图 2-5\. 卡车检测器工作

你可以在[附录 B](app02.html#appendix_b)中找到这个挑战的答案。

## 复习问题

让我们回顾一下你在本章编写的代码中学到的教训。花点时间回答以下问题：

1.  常规 TensorFlow 能在浏览器中运行吗？

1.  TensorFlow.js 能访问 GPU 吗？

1.  运行 TensorFlow.js 是否必须安装 CUDA？

1.  如果我在 CDN 上没有指定版本，会发生什么？

1.  毒性分类器如何识别违规行为？

1.  我们何时会达到毒性的阈值？

1.  毒性代码是否包含所有所需的文件？

1.  我们是否需要进行任何张量工作来使用这个毒性库？

这些练习的解决方案可以在[附录 A](app01.html#book_appendix)中找到。

^([1](ch02.html#idm45049254746888-marker)) TensorFlow 直到 2017 年 2 月 11 日才达到 1.0.0 版本。

^([2](ch02.html#idm45049254735528-marker)) Node 比 Python 案例研究提高了 2 倍：[*https://oreil.ly/4Jrbu*](https://oreil.ly/4Jrbu)

^([3](ch02.html#idm45049252373352-marker)) 请注意，此版本已锁定在 1.2.2。

^([4](ch02.html#idm45049252013384-marker)) 毒性模型信息可在[*https://oreil.ly/Eejyi*](https://oreil.ly/Eejyi)找到。
