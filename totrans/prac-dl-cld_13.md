# 第13章。食物的Shazam：使用TensorFlow Lite和ML Kit开发Android应用程序

在开发了病毒性的Not Hotdog应用程序（我们在[第12章](part0014.html#DB7S3-13fa565533764549a6f0ab7f11eed62b)中看到）之后，建阳最初应该构建一个能识别所有食物的分类器。事实上，该应用最初应该被称为SeeFood——一款可以“看到”食物并立即识别的应用程序（[图13-1](part0015.html#not_hotdog_app_listing_on_the_apple_app)）。换句话说，这是“食物的Shazam”。然而，该应用太成功了，以至于被Periscope收购。他的投资者Erlich Bachman的最初愿景仍未实现。在本章中，我们的任务是实现这个梦想。

![在苹果应用商店上的Not Hotdog应用列表](../images/00282.jpeg)

###### 图13-1。在苹果应用商店上的Not Hotdog应用列表

这样的功能在哪里会有用呢？对于健康狂热者，它可以查看一道菜并提供营养信息，包括卡路里数。或者，它可以扫描几种成分，并根据它们推荐食谱。或者，它甚至可以查看市场上的产品，并检查它是否含有任何被列入黑名单的成分，如特定过敏原。

这是一个有趣的问题，有几个原因需要解决，因为它代表了几个挑战：

数据收集挑战

世界上有一百多种美食，每种美食都有数百甚至数千种菜肴。

准确性挑战

它应该大部分时间都是正确的。

性能挑战

它应该几乎立即运行。

平台挑战

仅有iPhone应用程序是不够的。许多发展中国家的用户使用性能较弱的智能手机，特别是Android设备。跨平台开发是必须的。

为一个美食分类器应用程序制作一个美食已经够棘手了。想象一下要为存在的每种食物做到这一点——并且在两个平台上都要做到！一个个人或一个小团队将很快遇到尝试解决这个问题的扩展问题。在本章中，我们以此示例作为动力，探索我们在[第11章](part0013.html#CCNA3-13fa565533764549a6f0ab7f11eed62b)中探讨的移动AI开发生命周期的不同部分。

我们在这里探讨的材料不需要局限于智能手机。我们可以将我们的学习应用到移动以外的边缘设备，如Google Coral和Raspberry Pi，这些我们在本书后面讨论。

# 食品分类器应用程序的生命周期

因此，我们希望构建一个全球多美食、多平台的食品分类器。听起来是一项艰巨的任务，但我们可以将其分解为可管理的步骤。就像生活中一样，我们首先需要爬行，然后才能行走，最后才能奔跑。以下是一个可能要考虑的潜在方法：

1.  为单一美食（例如意大利美食）收集一小组初始图像。

1.  用相应的菜品标识符（例如`margherita_pizza`）标记这些图像。

1.  训练分类器模型。

1.  将模型转换为移动框架兼容格式（例如*.tflite*）。

1.  通过将模型与出色的用户体验集成来构建移动应用程序。

1.  招募α用户并与他们分享应用程序。

1.  收集详细的使用指标以及来自活跃用户的反馈，包括相机帧（倾向于反映真实世界的使用）和相应的代理标签（指示分类是正确还是错误）。

1.  使用新收集的图像作为额外的训练数据来改进模型。这个过程需要迭代。

1.  当模型的质量达到最低质量标准时，将应用程序/功能发布给更多/所有用户。继续监控和改进该美食的模型质量。

1.  为每种美食重复这些步骤。

###### 提示

对于第7步，我们可以选择在用户体验中集成反馈。例如，应用程序可以显示一个按概率排名的预测列表，这些预测是给定图片的可能候选项。如果我们的模型表现良好，用户应该大部分时间选择第一个选项。选择排名较低的预测基本上被视为错误的预测。在最坏的情况下，如果没有一个选项是正确的，允许用户手动添加一个新标签。这张照片，连同标签（在所有三种情况下），可以作为训练数据纳入。

我们不需要大量数据就可以开始。尽管上述每个步骤听起来可能有些复杂，但我们可以显著地自动化这个过程。这种方法的酷之处在于，应用程序被使用得越多，它就变得越好，自动地。就好像它有自己的生命一样。我们在本章末尾探讨了这种自我进化的方法。

###### 提示

您的应用/公司的铁杆粉丝将成为一个好的α用户。α用户理想情况下是那些对您的产品成功有着切身利益的人。对于一个食物识别应用程序，潜在的用户群可能是那些关注每一卡路里和食材的健身爱好者。这些用户明白应用程序的质量在一开始可能不尽如人意，但他们也看到了通过持续、建设性的反馈来塑造它的角色。他们自愿同意签署一个自由的数据共享协议，以提供诸如使用度量和日常使用中的图像帧等数据。我们建议您的用户清楚地知道您将收集关于他们的哪些信息，并允许他们选择退出或删除。不要显得可怕！

在本章中，我们探讨了上述生命周期的不同部分以及帮助我们完成每个步骤的工具。最后，我们将全面、端到端地查看整个移动开发生命周期，不仅仅是从本章开始，还包括之前的章节，并结合它们看看我们如何有效地在构建一个生产质量的真实应用程序中使用它们。

我们的旅程始于理解谷歌生态系统中以下工具。

TensorFlow Lite

模型转换和移动推理引擎。

ML Kit

高级软件开发工具包（SDK），具有几个内置API，以及运行自定义TensorFlow Lite模型的能力，以及与Firebase在谷歌云上的集成。

Firebase

一个基于云的框架，为生产质量的移动应用程序提供必要的基础设施，包括分析、崩溃报告、A/B测试、推送通知等。

TensorFlow模型优化工具包

用于优化模型大小和性能的一组工具。

# TensorFlow Lite概述

正如[第11章](part0013.html#CCNA3-13fa565533764549a6f0ab7f11eed62b)中提到的，谷歌发布了一个名为TensorFlow Lite的设备端推理引擎，将TensorFlow生态系统的覆盖范围扩展到了云端和桌面之外。在此之前，TensorFlow生态系统中的选项是将整个TensorFlow库移植到iOS（这样会很沉重和缓慢），后来是其略微简化的版本TensorFlow Mobile（有所改进，但仍然相当庞大）。

TensorFlow Lite从头开始针对移动端进行了优化，具有以下显著特点：

小

TensorFlow Lite附带了一个更轻量级的解释器。即使包含了所有操作符，解释器的大小也不到300KB。在使用常见模型如MobileNet时，我们可以预期该占用空间小于200KB。作为参考，上一代的TensorFlow Mobile曾占用1.5MB。此外，TensorFlow Lite使用了选择性注册——它只打包模型将使用的操作，最小化不必要的开销。

快速

TensorFlow Lite提供了显著的加速，因为它能够利用设备上的硬件加速，如GPU和NPU（如果可用）。在Android生态系统中，它使用Android神经网络API进行加速。类似地，在iPhone上，它使用Metal API。谷歌声称在使用GPU时（相对于CPU），在一系列任务中可以实现两到七倍的加速。

TensorFlow使用Protocol Buffers（Protobufs）进行反序列化/序列化。Protobufs是一种表示数据的强大工具，由于其灵活性和可扩展性，但这会在低功耗设备（如移动设备）上产生性能成本。

*FlatBuffers*证明是解决这个问题的答案。最初为视频游戏开发构建，低开销和高性能是必须的，它们也被证明是移动设备的一个很好的解决方案，显著减少了代码占用空间、内存使用和用于模型序列化和反序列化的CPU周期。这也大大提高了启动时间。

在网络中，有一些层在推断时具有固定的计算；例如，批量归一化层，可以预先计算，因为它们依赖于训练期间获得的值，如均值和标准差。因此，批量归一化层的计算可以与前一层的计算提前融合（即，在模型转换期间），从而减少推断时间，使整个模型更快。这被称为*预融合激活*，TensorFlow Lite支持。

解释器使用静态内存和静态执行计划。这有助于减少模型加载时间。

较少的依赖项

TensorFlow Lite的代码库主要是标准的C/C++，依赖项很少。这使得打包和部署更容易，同时还减小了部署包的大小。

支持自定义操作符

TensorFlow Lite包含量化和浮点核心操作符，其中许多已经针对移动平台进行了调整，可以用于创建和运行自定义模型。如果TensorFlow Lite不支持我们模型中的某个操作，我们也可以编写自定义操作符来使我们的模型运行起来。

在构建我们的初始Android应用程序之前，检查TensorFlow Lite的架构会很有用。

## TensorFlow Lite架构

[图13-2](part0015.html#high-level_architecture_of_the_tensorflo)提供了TensorFlow Lite架构的高级视图。

![TensorFlow Lite生态系统的高级架构](../images/00242.jpeg)

###### 图13-2. TensorFlow Lite生态系统的高级架构

作为应用开发者，我们将在顶层层中与TensorFlow Lite API（或者选择与ML Kit交互，后者又使用TensorFlow Lite）进行交互。TensorFlow Lite API将所有使用较低级API（如Android的神经网络API）时涉及的复杂性抽象化。请记住，这类似于Core ML在Apple生态系统中的工作方式。

从另一个极端来看，计算可以在各种类型的硬件模块上运行。其中最常见的是CPU，仅仅因为其普遍性和灵活性。现代智能手机越来越配备了专门的模块，包括GPU和新的NPU（专门用于神经网络计算，如iPhone X上的）。此外，数字信号处理器（DSP）专门用于单一任务，如面部认证、指纹认证和唤醒词检测（如“嘿Siri”）。

在物联网（IoT）世界中，微控制器（MCU）占主导地位。没有操作系统，没有处理器，内存很少（几KB），这些产品在大量生产中成本低廉，并且易于整合到各种应用程序中。使用TensorFlow Lite for Microcontrollers，开发人员可以在这些裸机设备上运行人工智能，而无需互联网连接。针对MCU的TensorFlow Lite解释器的精简版本（大约20 KB）称为TensorFlow Lite Micro Interpreter。

那么TensorFlow Lite如何与硬件交互呢？通过使用代理，这些代理是平台感知对象，公开一致的平台无关API。换句话说，代理屏蔽了解释器对其运行的具体硬件的任何了解。它们承担了图执行的全部或部分责任，否则将在CPU上运行，而现在则在效率更高的GPU和NPU上运行。在Android上，GPU代理使用OpenGL加速性能，而在iOS上则使用Metal API。

鉴于TensorFlow Lite本身是平台无关的，它需要调用一个实现已知合同的特定平台库。这个合同是TensorFlow Lite Delegate API。在Android中，这个合同由Android神经网络API来实现（适用于运行Android 8.1及以上版本的设备）。神经网络API旨在为更高级别的机器学习框架提供基础功能层。在苹果世界中，相当于神经网络API的是Metal Performance Shaders。

根据我们目前所了解的信息，让我们动手操作。

# 模型转换为TensorFlow Lite

在本书的这一部分，我们应该已经有一个模型（在ImageNet上预训练或在Keras中自定义训练）。在我们可以将该模型插入到Android应用程序之前，我们需要将其转换为TensorFlow Lite格式（一个.*tflite*文件）。

让我们看看如何使用TensorFlow Lite Converter工具转换模型，这个`tflite_convert`命令与我们的TensorFlow安装捆绑在一起：

```py
*# Keras to TensorFlow Lite*
$ tflite_convert \
  --output_file=my_model.tflite \
  --keras_model_file=my_model.h5

*# TensorFlow to TensorFlow Lite*
$ tflite_convert \
  --output_file=my_model.tflite \
  --graph_def_file=my_model/frozen_graph.pb
```

这个命令的输出是新的*my_model.tflite*文件，然后我们可以将其插入到下一节中的Android应用程序中。稍后，我们将看看如何使用`tflite_convert`工具使该模型更具性能。此外，TensorFlow Lite团队已经创建了许多预训练模型，这些模型以TensorFlow Lite格式可用，省去了这一转换步骤。

# 构建实时物体识别应用程序

从TensorFlow存储库中运行示例应用程序是玩转TensorFlow Lite API的简单方法。请注意，我们需要一部Android手机或平板电脑来运行该应用程序。以下是构建和部署应用程序的步骤：

1.  克隆TensorFlow存储库：

    ```py
    git clone https://github.com/tensorflow/tensorflow.git
    ```

1.  从[*https://developer.android.com/studio*](https://developer.android.com/studio)下载并安装Android Studio。

1.  打开Android Studio，然后选择“打开现有的Android Studio项目”（[图13-3](part0015.html#start_screen_of_android_studio)）。

    ![Android Studio的启动界面](../images/00020.jpeg)

    ###### 图13-3. Android Studio的启动界面

1.  转到克隆的TensorFlow存储库的位置，然后进一步导航至*tensorflow/tensorflow/contrib/lite/java/demo/*（[图13-4](part0015.html#android_studio_quotation_markopen_existi)）。选择打开。

    ![TensorFlow存储库中Android Studio“打开现有项目”屏幕](../images/00308.jpeg)

    ###### 图13-4. TensorFlow存储库中Android Studio“打开现有项目”屏幕

1.  在Android设备上，启用开发者选项。（请注意，我们在这里使用的是Pixel设备，它使用原生Android操作系统。对于其他制造商，说明可能会有所不同。）

    1.  转到设置。

    1.  向下滚动到“关于手机”或“关于平板电脑”选项（[图13-5](part0015.html#system_information_screen_on_an_android)），然后选择它。

        ![Android手机上的系统信息屏幕；在此处选择“关于手机”选项](../images/00253.jpeg)

        ###### 图13-5。Android手机上的系统信息屏幕；在此处选择“关于手机”选项

    1.  查找“构建号”行并点击七次。（是的，你没看错——七次！）

    1.  您应该看到一条消息（[图13-6](part0015.html#the_about_phone_screen_on_an_android_dev)），确认开发者模式已启用。

        ![Android设备上的“关于手机”屏幕](../images/00050.jpeg)

        ###### 图13-6。Android设备上的“关于手机”屏幕

    1.  如果您正在使用手机，请点击返回按钮返回到上一个菜单。

    1.  您应该看到一个“开发者选项”按钮，直接位于“关于手机”或“关于平板电脑”选项的上方（[图13-7](part0015.html#the_system_information_screen_showing_qu)）。点击此按钮以显示“开发者选项”菜单（[图13-8](part0015.html#quotation_markdeveloper_optionsquotation)）。

        ![显示“开发者选项”已启用的系统信息屏幕](../images/00241.jpeg)

        ###### 图13-7。显示“开发者选项”已启用的系统信息屏幕

        ![启用USB调试的Android设备上的“开发者选项”屏幕](../images/00150.jpeg)

        ###### 图13-8。启用USB调试的Android设备上的“开发者选项”屏幕

1.  通过USB电缆将Android设备连接到计算机。

1.  Android设备可能会显示一条消息，要求允许USB调试。启用“始终允许此计算机”，然后选择确定（[图13-9](part0015.html#allow_usb_debugging_on_the_displayed_ale)）。

    ![在显示的警报上允许USB调试](../images/00103.jpeg)

    ###### 图13-9。在显示的警报上允许USB调试

1.  在Android Studio中，在调试工具栏上（[图13-10](part0015.html#debug_toolbar_in_android_studio)），点击“运行应用”按钮（右向三角形）。

    ![Android Studio中的调试工具栏](../images/00066.jpeg)

    ###### 图13-10。Android Studio中的调试工具栏

1.  一个窗口打开，显示所有可用的设备和模拟器（[图13-11](part0015.html#select_the_phone_from_the_deployment_tar)）。选择您的设备，然后选择确定。

    ![从部署目标选择屏幕中选择手机](../images/00023.jpeg)

    ###### 图13-11。从部署目标选择屏幕中选择手机

1.  应用程序应该安装并开始在我们的手机上运行。

1.  应用程序将请求您的相机权限；请继续授予权限。

1.  相机的实时视图应该出现，以及实时对象分类预测，以及进行预测所需的秒数，如[图13-12](part0015.html#the_app_up-and-running_appcomma_showing)所示。

    ![应用程序正在运行，显示实时预测](../images/00161.jpeg)

    ###### 图13-12。应用程序正在运行，显示实时预测

就是这样！我们在手机上运行了一个基本的应用程序，它可以拍摄视频帧并对其进行分类。它简单而且运行得相当不错。

除了对象分类之外，TensorFlow Lite存储库还有许多其他AI问题的示例应用程序（iOS和Android），包括以下内容：

+   对象检测

+   姿势估计

+   手势识别

+   语音识别

拥有这些示例应用程序的好处是，只要有基本的说明，没有移动开发背景的人就可以在手机上运行它们。更好的是，如果我们有一个自定义训练的模型，我们可以将其插入应用程序并看到它为我们的自定义任务运行。

这对于刚开始很棒。然而，在现实世界中情况要复杂得多。拥有成千上万甚至数百万用户的严肃现实世界应用程序的开发者需要考虑超越推断的问题，比如更新和分发模型，在用户子集中测试不同版本，保持iOS和Android之间的一致性，最终降低每个工程成本。在内部完成所有这些工作可能会昂贵、耗时，而且实际上是不必要的。自然而然，提供这些功能的平台会很吸引人。这就是ML Kit和Firebase的作用所在。

# ML Kit + Firebase

ML Kit是在2018年Google I/O大会上推出的移动SDK。它为初学者和高级ML开发人员提供了方便的API，用于执行许多常见的ML任务。默认情况下，ML Kit具有视觉和语言智能的通用功能集。[表4-1](part0006.html#top_1percent_accuracy_and_feature_length)列出了我们可以在几行代码中完成的一些常见ML任务。

表13-1\. ML Kit内置功能

| **视觉** | **语言** |
| --- | --- |
| 对象分类对象检测和跟踪流行的地标检测文本识别人脸检测条形码检测 | 语言识别设备上翻译智能回复 |

ML Kit还使我们能够使用自定义训练的TensorFlow Lite模型进行推断。让我们花点时间来欣赏为什么这对开发者如此重要。想象一下，我们正在构建一个名片扫描器。我们可以引入一个自定义的名片检测模型，识别出名片可见时及其边界（以创建一个漂亮的视觉用户界面），运行内置的文本识别，并过滤掉边界之外的文本，以防止多余的字符。或者，考虑一个可以通过指向物体构建的语言学习游戏，运行一个物体分类器，然后使用设备上的翻译API来宣布标签的法语。使用ML Kit完全可以相对快速地构建这些功能。虽然许多这些功能也在Core ML中可用，但ML Kit具有跨平台的额外优势。

然而，ML Kit只是谜题的一部分。它集成到Google的Firebase中，这是Google Cloud的一部分，是移动和Web应用程序开发平台。Firebase提供了一系列功能，这些功能是生产质量应用程序所必需的基础设施，例如以下功能：

+   推送通知

+   认证

+   崩溃报告

+   日志

+   性能监控

+   托管设备测试

+   A/B测试

+   模型管理

最后一点对我们非常重要。Firebase给我们带来的最大好处之一是能够在云端托管我们的自定义模型，并根据需要在应用程序中下载它们。只需将模型复制到Google Cloud上的Firebase，引用应用程序内的ML Kit模型，然后我们就可以开始了。A/B测试功能使我们能够向不同用户展示同一模型的不同版本，并测量不同模型之间的性能。

###### 注意

对于许多内置功能，ML Kit还提供了一个在云端处理图像的功能，其中模型比设备上的模型要大得多。这些更大的模型显然需要更强大的硬件来运行它们，但可以提供一些准确性的改进，可能有更大的分类法（比如成千上万的对象类别而不是数百个）。事实上，一些功能，比如地标识别功能，只能在云端上运行。

云处理选项在我们需要一点额外准确性和/或用户手机处理能力不足以很好地运行设备上模型时特别有用。

## ML Kit中的对象分类

对于我们之前的实时对象分类任务，如果我们使用ML Kit而不是原始的TensorFlow Lite，我们可以简化我们的代码到以下几行（使用Kotlin）：

```py
val image = FirebaseVisionImage.fromBitmap(bitmap)
val detector = FirebaseVision.getInstance().visionLabelDetector
val result = detector.detectInImage(image).addOnSuccessListener { labels ->
    *`// Print labels`*
}
```

## ML Kit中的自定义模型

除了ML Kit提供的预构建模型外，我们还可以运行自己的自定义模型。这些模型必须是TensorFlow Lite格式。以下是一个简单的代码片段，用于加载打包到应用中的自定义模型：

```py
val customModel = FirebaseLocalModelSource.Builder("my_custom_model")
        .setAssetFilePath("my_custom_model.tflite").build()
FirebaseModelManager.getInstance().registerLocalModelSource(customModel)
```

接下来，我们指定模型的输入和输出配置（对于一个接收尺寸为224x224的RGB图像并为1,000个类别名称提供预测的模型）：

```py
val IMAGE_WIDTH = 224
val IMAGE_HEIGHT = 224
val modelConfig = FirebaseModelInputOutputOptions.Builder()
        .setInputFormat(0, FirebaseModelDataType.FLOAT32, intArrayOf(1,
IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        .setOutputFormat(0, FirebaseModelDataType.FLOAT32, intArrayOf(1, 1000))
        .build()
```

接下来，我们创建一个单个图像数组，并将每个像素归一化到范围[-1,1]：

```py
val bitmap = Bitmap.createScaledBitmap(image, IMAGE_WIDTH, IMAGE_HEIGHT, true)
val input = Array(1) {
	Array(IMAGE_WIDTH) { Array(IMAGE_HEIGHT) { FloatArray(3) } }
	}
for (x in 0..IMAGE_WIDTH) {
    for (y in 0..IMAGE_HEIGHT) {
        val pixel = bitmap.getPixel(x, y)
        input[0][x][y][0] = (Color.red(pixel) - 127) / 128.0f
        input[0][x][y][1] = (Color.green(pixel) - 127) / 128.0f
        input[0][x][y][2] = (Color.blue(pixel) - 127) / 128.0f
    }
}
```

现在，我们基于我们的自定义模型设置一个解释器：

```py
val options = FirebaseModelOptions.Builder()
     .setLocalModelName("my_custom_model").build()
val interpreter = FirebaseModelInterpreter.getInstance(options)
```

接下来，我们在解释器上运行我们的输入批处理：

```py
val modelInputs = FirebaseModelInputs.Builder().add(input).build()
interpreter.run(modelInputs, modelConfig).addOnSuccessListener { result ->
    // Print results
}
```

是的，就是这么简单！在这里，我们看到了如何将自定义模型与应用捆绑在一起。有时，我们可能希望应用动态从云端下载模型，原因如下：

+   我们希望在Play商店上保持默认应用大小较小，以免阻止有数据使用限制的用户下载我们的应用。

+   我们希望尝试不同种类的模型，并根据可用的指标选择最佳模型。

+   我们希望用户拥有最新和最好的模型，而无需经历整个应用发布流程。

+   需要模型的功能可能是可选的，我们希望节省用户设备上的空间。

这带我们来到了托管模型。

## 托管模型

ML Kit与Firebase一起，使我们能够在Google Cloud上上传和存储我们的模型，并在需要时从应用中下载。模型下载后，其功能与将模型捆绑到应用中完全相同。此外，它还为我们提供了推送模型更新的能力，而无需对应用进行整体发布。此外，它还让我们可以在真实世界中对我们的模型进行实验，以查看哪些模型在实际中表现最佳。对于托管模型，我们需要看两个方面。

### 访问托管模型

以下行通知Firebase我们想使用名为`my_remote_custom_model`的模型：

```py
val remoteModel = FirebaseCloudModelSource.Builder("my_remote_custom_model")
        .enableModelUpdates(true).build()
FirebaseModelManager.getInstance().registerCloudModelSource(remoteModel)
```

请注意，我们将`enableModelUpdates`设置为使我们能够从云端向设备推送模型更新。我们还可以选择配置模型首次下载的条件与每次下载的条件——设备是否空闲，当前是否正在充电，下载是否仅限于WiFi网络等。

接下来，我们设置一个解释器，就像我们在本地模型中所做的那样：

```py
val options = FirebaseModelOptions.Builder()
    .setCloudModelName("my_remote_custom_model").build()
val interpreter = FirebaseModelInterpreter.getInstance(options)
```

在此之后，执行预测的代码看起来与本地模型的代码完全相同。

接下来，我们讨论托管模型的另一个方面——上传模型。

### 上传托管模型

截至撰写本文时，Firebase仅支持托管在GCP上的模型。在本节中，我们将介绍创建、上传和存储托管模型的简单过程。本小节假设我们已经拥有现有的GCP帐户。

以下列出了我们需要采取的步骤来将模型托管在云端：

1.  前往[*https://console.firebase.google.com*](https://console.firebase.google.com)。选择一个现有项目或添加一个新项目（[图13-13](part0015.html#homepage_of_google_cloud_firebase)）。

    ![Google Cloud Firebase的主页](../images/00279.jpeg)

    ###### 图13-13。Google Cloud Firebase的主页

1.  在项目概述屏幕上，创建一个Android应用（[图13-14](part0015.html#the_project_overview_screen_on_google_cl)）。

    ![Google Cloud Firebase上的项目概述屏幕](../images/00229.jpeg)

    ###### 图13-14。Google Cloud Firebase上的项目概述屏幕

1.  在Android Studio中使用项目的应用ID（[图13-15](part0015.html#app_creation_screen_on_firebase)）。

    ![Firebase上的应用创建屏幕](../images/00260.jpeg)

    ###### 图13-15。Firebase上的应用创建屏幕

1.  点击“注册应用程序”后，下载配置文件。这个配置文件为应用程序提供了访问我们云账户所需的凭据。按照应用程序创建页面上显示的方式将配置文件和Firebase SDK添加到Android应用程序中。

1.  在ML Kit部分，选择开始，然后选择“添加自定义模型”（[图13-16](part0015.html#the_ml_kit_custom_models_tab)）。

    ![ML Kit自定义模型选项卡](../images/00158.jpeg)

    ###### 图13-16。ML Kit自定义模型选项卡

1.  在名称字段中，输入`my_remote_custom_model`以匹配代码中的名称。

1.  从计算机上传模型文件（[图13-17](part0015.html#uploading_a_tensorflow_lite_model_file_t)）。

    ![将TensorFlow Lite模型文件上传到Firebase](../images/00106.jpeg)

    ###### 图13-17。将TensorFlow Lite模型文件上传到Firebase

1.  文件上传完成后，点击“发布”按钮。

就是这样！我们的模型现在已经准备好从应用程序动态访问和使用。接下来，我们将探讨如何使用Firebase在模型之间进行A/B测试。

## A/B测试托管模型

让我们假设我们有一个名为`my_model_v1`的版本1模型，最初部署给我们的用户。在用户使用一段时间后，我们获得了更多数据可以进行训练。这次训练的结果是`my_model_v2`（[图13-18](part0015.html#currently_uploaded_custom_models_to_fire)）。我们想评估这个新版本是否会给我们带来更好的结果。这就是A/B测试的用武之地。

![当前上传的自定义模型到Firebase](../images/00070.jpeg)

###### 图13-18。当前上传的自定义模型到Firebase

A/B测试被行业广泛使用，是一种统计假设检验技术，回答了“B是否比A更好？”的问题。这里的A和B可以是同类的任何东西：网站上的内容、手机应用程序上的设计元素，甚至是深度学习模型。在积极开发模型并发现用户对模型不同迭代的反应时，A/B测试是一个非常有用的功能。

用户已经使用`my_model_v1`一段时间了，我们想看看v2版本是否让我们的用户疯狂。我们想慢慢开始；也许只有10%的用户应该得到v2。为此，我们可以设置一个A/B测试实验如下：

1.  在Firebase中，点击A/B测试部分，然后选择“创建实验”（[图13-19](part0015.html#asolidusb_testing_screen_in_firebase_whe)）。

    ![在Firebase中进行A/B测试的屏幕，我们可以创建一个实验](../images/00029.jpeg)

    ###### 图13-19。在Firebase中进行A/B测试的屏幕，我们可以创建一个实验

1.  选择远程配置选项。

1.  在基础部分中，在“实验名称”框中输入实验名称和可选描述（[图13-20](part0015.html#the_basics_section_of_the_screen_to_crea)），然后点击下一步。

    ![创建远程配置实验屏幕的基础部分](../images/00315.jpeg)

    ###### 图13-20。创建远程配置实验屏幕的基础部分

1.  在打开的定位部分中，从“目标用户”下拉菜单中选择我们的应用程序，并输入目标用户的百分比（[图13-21](part0015.html#the_targeting_section_of_the_remote_conf)）。

    ![远程配置屏幕的定位部分](../images/00274.jpeg)

    ###### 图13-21。远程配置屏幕的定位部分

1.  选择一个有意义的目标指标。我们将在下一节中更详细地讨论这个问题。

1.  在变体部分（[图13-22](part0015.html#the_variants_section_of_the_remote_confi)），创建一个名为`model_name`的新参数，反映了特定用户将使用的模型的名称。对照组使用默认模型，即`my_model_v1`。我们还创建了一个名为`my_model_v2`的额外变体，分配给10%的用户。

    ![远程配置屏幕的变体部分](../images/00234.jpeg)

    ###### 图13-22。远程配置屏幕的变体部分

1.  选择“审查”，然后选择“开始实验”。随着时间的推移，我们可以增加使用变体的用户分布。

哒哒！现在我们的实验已经开始运行。

### 测量实验效果

接下来我们该怎么做？我们希望给我们的实验一些时间来看它的表现。根据实验的类型，我们可能需要给它几天到几周的时间。实验的成功可以通过任意数量的标准来确定。Google提供了一些开箱即用的指标，我们可以使用，如[图13-23](part0015.html#analytics_available_when_setting_up_an_a)所示。

![设置A/B测试实验时可用的分析](../images/00198.jpeg)

###### 图13-23。设置A/B测试实验时可用的分析

假设我们想要最大化我们估计的总收入——毕竟，我们都想像Jian-Yang一样变得富有。我们将衡量那些打开实验的用户与基线（即未打开实验的用户）的收入。如果我们的每位用户的收入相对于基线增加，我们将认为实验是成功的。相反，如果每位用户的收入没有增加/减少，我们将得出相反的结论。对于成功的实验，我们希望逐步将其推广到所有用户。在那一点上，它不再是一个实验，而是“毕业”成为核心产品。

## 在代码中使用实验

现在我们已经设置了一个实验，让我们看看如何在我们的应用程序中使用代码包含它。要在代码中使用适当的模型，我们只需访问远程配置对象（`remoteConfig`）并从中获取模型名称。我们从远程配置对象获取的模型名称将取决于用户是否包含在实验中。以下代码行实现了这一点：

```py
val remoteConfig = FirebaseRemoteConfig.getInstance()
remoteConfig.fetch()
val modelName = remoteConfig.getString("current_best_model")
val remoteModel = FirebaseCloudModelSource.Builder(modelName)
        .enableModelUpdates(true).build()
FirebaseModelManager.getInstance().registerCloudModelSource(remoteModel)
```

执行预测的其余代码与前几节完全相同。我们的应用现在已准备好使用我们实验规定的正确模型。

# iOS上的TensorFlow Lite

前几章展示了在iOS上使用苹果的Core ML有多么容易。我们只需将模型拖放到Xcode中，然后只需几行代码就可以开始进行推理。相比之下，即使是查看TensorFlow Lite存储库中的基本iOS示例，也可以明显看出，需要大量样板代码才能使最基本的应用程序运行。相比之下，与ML Kit一起使用的TensorFlow Lite是一种相当愉快的体验。除了能够使用干净简洁的API外，我们还可以获得本章前面详细介绍的所有功能——远程下载和更新模型、模型A/B测试以及云回退处理。所有这些都不需要做太多额外的工作。编写同时面向iOS和Android的深度学习应用程序的开发人员可能会考虑使用ML Kit作为“一次构建，随处使用”的方式。

# 性能优化

在[第6章](part0008.html#7K4G3-13fa565533764549a6f0ab7f11eed62b)中，我们从理论角度探讨了量化和剪枝。让我们从TensorFlow Lite的角度以及实现它们的工具中近距离地看看它们。

## 使用TensorFlow Lite转换器进行量化

对于iOS，苹果在其Core ML工具包中提供了`quantization_utils`。而对于TensorFlow Lite，等效的是已经内置的`tflite_convert`工具，我们在本章前面使用过。在命令行中，我们可以指定输入文件、模型图、要转换为的数据类型以及输入和输出的名称（可以使用Netron进行检查，如[第11章](part0013.html#CCNA3-13fa565533764549a6f0ab7f11eed62b)所示）。从32位转换为8位整数表示意味着模型大小减小四倍，而准确性损失相对较小。

```py
$ tflite_convert \
  --output_file=quantized-model.tflite \
  --graph_def_file=/tmp/some-graph.pb \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1 \
  --mean_values=128 \
  --std_dev_values=127
```

完成后，此命令应该给我们提供`quantized-model.tflite`模型。

## TensorFlow模型优化工具包

TensorFlow Lite转换器是获得我们的模型量化的最简单方法。值得注意的是，转换器在训练后对模型进行量化。由于表示能力的降低，可能会出现轻微但明显的准确率损失。我们能做得更好吗？*量化感知训练*，顾名思义，在训练期间考虑量化的影响，并试图补偿和减少后续训练中可能发生的损失。

尽管两种量化形式都使模型大小减少了75%，但实验结果显示如下：

+   对于MobileNetV2，与训练后量化导致的准确率下降8个点相比，量化感知训练仅导致了1个点的下降。

+   对于InceptionV3，与训练后量化相比，量化感知训练使延迟减少了惊人的52%，而后者仅减少了25%。

###### 注意

值得注意的是，这些准确率指标是在1,000类ImageNet测试集上的。大多数问题在类别较少时具有较低的复杂性。对于这种更简单的问题，训练后量化应该导致更小的损失。

量化感知训练可以使用 TensorFlow Model Optimization Toolkit 实现。该工具包还提供了一系列用于模型压缩的工具，包括修剪。此外，TensorFlow Lite 模型仓库已经提供了使用这种技术的预量化模型。[表13-2](part0015.html#effects_of_different_quantizatio-id00001)列出了各种量化策略的效果。

表13-2\. 不同量化策略（8位）对模型的影响（来源：TensorFlow Lite模型优化文档）

| 模型 | MobileNet | MobileNetV2 | InceptionV3 |
| --- | --- | --- | --- |
| **Top-1准确率** | **原始** | 0.709 | 0.719 | 0.78 |
| **训练后量化** | 0.657 | 0.637 | 0.772 |
| **量化感知训练** | 0.7 | 0.709 | 0.775 |
| **延迟（毫秒）** | **原始** | 124 | 89 | 1130 |
| **训练后量化** | 112 | 98 | 845 |
| **量化感知训练** | 64 | 54 | 543 |
| **大小（MB）** | **原始** | 16.9 | 14 | 95.7 |
| **优化后** | 4.3 | 3.6 | 23.9 |

# Fritz

正如我们迄今所见，Core ML 和 TensorFlow Lite 的主要目的是为了提供快速的移动推断。有一个模型，将其插入应用程序中，并运行推断。然后出现了ML Kit，除了具有内置的AI功能外，还使得部署模型和监视我们的自定义模型（使用Firebase）变得更加容易。回顾整个流程，我们有训练、转换为移动格式、优化速度、部署到用户、监视性能、跟踪模型版本。这些是分布在许多工具中的几个步骤。很可能没有一个人拥有整个流程。这是因为通常这些工具需要一定程度的熟悉度（例如，将模型部署给用户可能超出数据科学家的舒适区）。总部位于波士顿的初创公司Fritz正试图消除这些障碍，使数据科学家和移动开发人员更加简单地完成从模型开发到部署的整个周期。

Fritz提供了一个移动AI开发的端到端解决方案，包括以下值得注意的功能：

+   能够在训练完成后使用回调函数直接将模型部署到用户设备上。

+   能够在计算机上直接对模型进行基准测试，而无需将其部署到手机上。以下代码演示了这一点：

    ```py
    $ fritz model benchmark <path to keras model.h5>
     ...
     ------------------------
     Fritz Model Grade Report
     ------------------------

     Core ML Compatible:              True
     Predicted Runtime (iPhone X):    31.4 ms (31.9 fps)
     Total MFLOPS:                    686.90
     Total Parameters:                1,258,580
     Fritz Version ID:                <Version UID>
    ```

+   能够加密模型，以便我们的知识产权不被恶意行为者从设备上窃取。

+   能够用几行代码实现许多先进的计算机视觉算法。SDK带有这些算法的移动友好实现，可以以高帧率运行。例如，图像分割、风格转移、对象检测和姿势估计。[图13-24](part0015.html#performance_of_fritz_sdkapostrophes_obje)显示了在各种iOS设备上运行的对象检测的基准。

    ```py
    let poseModel = FritzVisionPoseModel()
    guard let poseResult = try? poseModel.predict(image) else { return }
    let imageWithPose = poseResult.drawPose() *`//` `O``v``e``r``l``a``y``s` `p``o``s``e` `o``n` `i``n``p``u``t``.`*
    ```

    ![Fritz SDK在不同移动设备上的对象检测功能性能，相对于iPhone X](../images/00163.jpeg)

    ###### 图13-24。Fritz SDK在不同移动设备上的对象检测功能性能，相对于iPhone X

+   能够通过重新训练使用他们的Jupyter笔记本自定义预构建模型到自定义数据集。值得注意的是，这可能是一个困难的问题（即使对于专业数据科学家来说），但因为开发人员只需确保数据格式正确，这个问题就得到了很大的简化。

+   能够从命令行管理所有模型版本。

+   能够使用Fritz的开源应用Heartbeat测试我们模型的移动准备性（也可在iOS/Android应用商店中找到）。假设我们已经准备好一个模型，但对移动端不太了解，我们可以克隆该应用，用我们自己的模型替换现有模型，并在手机上运行。

+   一个充满活力的社区，关于移动AI最新信息的博客在*heartbeat.fritz.ai*上。

# 移动AI应用开发周期的整体视角

到目前为止，在本书中，我们已经看过许多技术和技术，使我们能够在移动AI开发周期中执行各种任务。在这里，我们通过探索贯穿整个生命周期的问题类型，将所有内容联系在一起。[图13-25](part0015.html#mobile_ai_app_development_life_cycle)提供了开发周期的广泛概述。

![移动AI应用开发生命周期](../images/00162.jpeg)

###### 图13-25。移动AI应用开发生命周期

## 如何收集初始数据？

我们可以采用几种不同的策略来实现这一点：

+   找到感兴趣的对象，并手动拍摄不同角度、光照、环境、构图等照片。

+   使用浏览器扩展程序（如Fatkun）从互联网上抓取数据（[第12章](part0014.html#DB7S3-13fa565533764549a6f0ab7f11eed62b)）。

+   查找现有数据集（[Google数据集搜索](https://oreil.ly/NawZ6)）。例如，Food-101用于菜肴。

+   合成自己的数据集。

    1.  将对象（前景）放在绿幕（背景）前拍摄照片。将背景替换为随机图像以合成大型数据集，同时进行缩放、裁剪和旋转，以创建数百张图像。

    1.  如果无法使用绿幕，从现有的真实世界图像中分割（即切割）出对象并重复上一步骤，以构建一个强大、多样化的数据集。需要注意的是，在(a)和(b)点中，前景需要有足够的多样性；否则，网络可能会过度学习一个例子，而不是理解对象。

    1.  找到感兴趣对象的真实3D模型，并将其放置在使用Unity等3D框架的真实环境中。调整光照和相机位置、缩放和旋转，从多个角度拍摄这个对象的快照。我们在[第7章](part0009.html#8IL23-13fa565533764549a6f0ab7f11eed62b)中探讨了AI.Reverie和CVEDIA等在这一领域工作的公司。我们使用逼真的模拟器来为自动驾驶章节（[第16章](part0019.html#I3QM3-13fa565533764549a6f0ab7f11eed62b)和[第17章](part0020.html#J2B83-13fa565533764549a6f0ab7f11eed62b)）训练模型。

## 如何为我的数据标记标签？

在前一个答案中的大多数步骤应该已经为您提供了数据的标签。对于未标记的集合，请使用Supervisely、Labelbox和Diffgram等标记工具。对于真正大型的数据集，自己注释数据可能不可行，提供收入机会给弱势群体的社会责任标记服务，如Digital Data Divide、iMerit和Samasource，可能是一个不错的选择。

## 如何训练我的模型？

训练模型有两种广泛的方法：

+   使用代码：使用Keras和TensorFlow。

+   无需代码：使用自定义分类器服务，如Google的Auto ML、Microsoft的CustomVision.ai和Clarifai，或仅适用于Apple生态系统的Create ML。

## 如何将模型转换为移动友好格式？

以下是将模型转换为移动兼容格式的几种不同方法：

+   使用Core ML Tools（仅适用于Apple）。

+   对于iOS和Android，请使用TensorFlow Lite Converter。

+   或者，使用Fritz进行端到端的流程。

## 如何使我的模型性能良好？

以下是一些使模型性能良好的技术：

+   从MobileNet系列等高效模型开始，甚至更好的是EfficientNet。

+   通过量化和修剪模型来减小模型的大小，以改善加载和推断时间，同时保持模型准确性相对完好。预计大小会减小高达75%，准确性损失很小。

+   对于Apple生态系统，请使用Core ML Tools，对于iOS和Android，请使用TensorFlow模型优化工具包。

## 如何为用户构建出色的用户体验？

显然，这将取决于您试图解决的问题的类型。但一个一般的指导原则是在互动性、性能和资源使用（内存、CPU、电池等）之间取得适当的平衡。当然，一个智能的反馈机制可以实现轻松的数据收集和反馈。将这种体验变成游戏化将使其达到一个全新的水平。

在食品分类器应用程序的情况下，用户拍照后，我们的用户界面将显示给定照片的前五个候选预测列表（按置信度降序排列）。例如，如果用户拍摄了一张披萨的照片，屏幕上显示的候选预测可能是“馅饼 - 75%”，“披萨 - 15%”，“烤饼 - 6%”，“面包 - 3%”，“砂锅 - 1%”。在这种情况下，用户将选择第二个预测。对于完美的模型，用户将始终选择第一个预测。由于我们的模型并不完美，用户选择的预测的排名将成为未来改进模型的信号。在最坏的情况下，如果没有一个预测是正确的，用户应该有一种手动标记数据的方法。在手动输入时提供自动建议功能将有助于保持数据集中的干净标签。

一个更好的体验可能是用户永远不需要拍照。相反，预测是实时可用的。

## 如何让模型对用户可用？

以下是将模型部署给用户的一些方法：

+   将模型捆绑到应用程序二进制文件中，并在应用商店发布。

+   或者，将模型托管在云中，并在需要时让应用程序下载模型。

+   使用Fritz或Firebase等模型管理服务（与ML Kit集成）。

## 如何衡量模型的成功？

第一步是确定成功标准。考虑以下示例：

+   我的模型应该在第90百分位数以下的时间内运行推断。

+   “使用这个模型的用户每天都会打开应用程序。”

+   在食品分类器应用程序中，一个成功的指标可能是“80%的用户选择了预测列表中的第一个预测”。

这些成功标准并非一成不变，应该随着时间的推移不断发展。数据驱动非常重要。当你有一个新的模型版本时，在一部分用户中运行A/B测试，并根据该版本评估成功标准，以确定它是否比上一个版本更好。

## 如何改进我的模型？

以下是改进我们模型质量的一些方法：

+   从用户那里收集关于个别预测的反馈：什么是正确的，更重要的是，什么是错误的。将这些图像与相应的标签一起作为下一个模型训练周期的输入。[图13-26](part0015.html#the_feedback_cycle_of_an_incorrect_predi)说明了这一点。

+   对于明确选择加入的用户，在预测置信度低时自动收集帧。手动标记这些帧并将其输入到下一个训练周期中。

![错误预测的反馈周期，生成更多训练数据，导致改进的模型](../images/00073.jpeg)

###### 图13-26. 错误预测的反馈周期，生成更多训练数据，导致改进的模型

## 如何在用户手机上更新模型？

以下是如何在用户手机上更新模型的一些方法：

+   将新模型捆绑到下一个应用程序发布中。这是缓慢且不灵活的。

+   将新模型托管到云端，并强制应用程序在世界范围内下载新模型。最好在用户连接WiFi时执行此操作。

+   使用模型管理系统，如Firebase（与ML Kit一起）或Fritz来自动化涉及的大部分繁重工作。

有了所有这些问题的答案，让我们欣赏这个应用程序如何自我改进的美丽。

# 自我进化模型

对于只需要成熟预训练模型的应用程序，我们的工作已经完成。只需将模型插入应用程序，我们就可以开始了。对于特别依赖稀缺训练数据的定制训练模型，我们可以让用户参与构建一个自我改进、不断发展的模型。

在最基本的层面上，每当用户使用应用程序时，他们都会提供必要的反馈（图像+标签）以进一步改进模型，如[图13-27](part0015.html#the_self-evolving_model_cycle)所示。

![自我进化模型周期](../images/00033.jpeg)

###### 图13-27. 自我进化模型周期

就像大学生必须在步入现实世界之前经历多年的毕业准备一样，模型在到达最终用户之前必须经历开发阶段。以下是软件发布阶段的流行模型，也是发布AI模型的有用指南。

1. Dev

这是开发的初始阶段，在这个阶段，应用程序的开发人员是唯一的用户。数据积累缓慢，体验非常不稳定，模型预测可能相当不可靠。

2. Alpha

在应用程序准备好由开发人员之外的少数用户测试之后，现在被认为处于alpha阶段。用户反馈在这里非常关键，因为它将有助于改善应用体验，并提供更大规模的数据给管道。这里的体验不像以前那样有bug，模型预测也更可靠一些。在一些组织中，这个阶段也被称为*dogfooding*（即员工进行内部应用测试）。

3. Beta

测试版应用的用户数量比α版要多得多。用户反馈在这里也非常重要。数据收集速度更大规模。许多不同的设备在现实世界中收集数据，以帮助快速改进模型。由于α用户的存在，体验更加稳定，模型预测更加可靠。测试版应用通常托管在苹果的TestFlight（iOS）和谷歌的Play Console（Android）上。第三方服务，如HockeyApp和TestFairy，也很受欢迎，用于托管测试版程序。

4. 产品

生产中的应用是稳定的，并且广泛可用。尽管数据以大量速率进入，但在这一点上，模型相当稳定，不会学习太多。然而，随着真实世界的使用不断增长，它能够在可能在前三个阶段之前没有见过的边缘情况上变得更好。当模型足够成熟时，可以对小版本改进进行α/β测试或在生产受众的子集上进行A/B测试。

尽管许多数据科学家在开始开发之前假设数据是可用的，但移动AI领域的人可能需要在一个小数据集上启动，并逐步改进他们的系统。

对于食物Shazam应用的这种自我进化系统，开发人员必须做出的最困难的选择应该是他们去哪些餐馆收集种子数据。

# 案例研究

让我们看一些有趣的例子，展示迄今为止学到的知识如何应用于行业中。

## 放手吧！

我们需要在这里承认。我们在本章一直在构建的应用已经存在。一个名为Fit Now的波士顿公司在一年前已经实现了Erlich Bachman的梦想。Lose It!([图13-28](part0015.html#snap_it_feature_from_lose_itexclamation))声称已经帮助了3000万用户减掉了8500万磅，通过跟踪他们所吃的食物。用户可以用手机的摄像头对准条形码、营养标签和他们即将吃的食物，以跟踪每一顿饭所摄入的卡路里和宏量元素。

公司首先使用基于云的算法实现了其食品扫描系统。由于资源需求和网络延迟，体验无法实时。为了改善体验，Fit Now团队将其模型迁移到TensorFlow Lite以优化移动端，并使用ML Kit无缝部署到数百万用户。通过持续的反馈循环、模型更新和A/B测试，该应用基本上可以自我改进。

![Lose It!中的Snap It功能显示了扫描食物项目的多个建议](../images/00321.jpeg)

###### 图13-28。Lose It!中的Snap It功能显示了扫描食物项目的多个建议

## Pixel 3手机上的人像模式

将专业摄影师与业余爱好者区分开来的一个关键视觉概念是bokeh，或者图像主题背后背景的模糊。相机距离主题越近，背景就越模糊。借助低* f *值的专业镜头（通常价格高达数千美元），可以产生壮观的模糊效果。

但如果你没有那种钱，人工智能可以帮助。谷歌的Pixel 3提供了一个“人像”模式，可以创建虚化效果。基本上，它使用CNN来估计场景中每个像素的深度，并确定哪些像素属于前景和背景。然后将背景像素模糊到不同的强度，以创建虚化效果，如[图13-29](part0015.html#portrait_effect_on_pixel_3_which_achieve)所示。

深度估计是一个计算密集型任务，因此快速运行是至关重要的。具有GPU后端的Tensorflow Lite挺身而出，比CPU后端快大约3.5倍。

Google 通过训练一个专门用于深度估计的神经网络来实现这一点。它使用了一个带有多个手机摄像头的装置（被称为“Frankenphone”），从每个摄像头属于的略有不同的视角拍摄同一场景的照片。然后，它利用每个像素经历的 *视差效应* 精确确定每个像素的深度。然后将这个深度图与图像一起输入到 CNN 中。

![Pixel 3 上的肖像效果，通过模糊实现前景和背景的分离](../images/00003.jpeg)

###### 图 13-29\. Pixel 3 上的肖像效果，通过模糊实现前景和背景的分离

## 阿里巴巴的说话者识别

想象一下对着朋友的手机说“嘿 Siri”或“嘿 Google”，然后接着说“读取最后一条短信”的神奇词语。从隐私的角度来看，如果这个方法奏效，那将是令人恐惧的。显然，如今大多数移动操作系统都要求我们先解锁手机才能继续。这给手机的所有者带来了不太流畅的体验。

阿里巴巴机器智能实验室通过以下方法解决了这个问题。首先，将语音转换为声谱图像（使用梅尔频率倒谱算法提取音频特征），训练一个 CNN（使用迁移学习），最后使用 Tensorflow Lite 部署到设备上。没错：CNN 不仅可以用于计算机视觉！

通过识别说话者，他们能够在家庭中有多个用户的设备上个性化内容（例如，在其类似 Netflix 的电视应用程序中的“与您的电视交谈”功能）。此外，通过识别声音是否为人类声音，它能够将口头命令与背景噪音隔离开来进行转录。为了加快使用 Tensorflow Lite 的处理速度，工程师们保留了 `USE_NEON` 标志以加速基于 ARM 的 CPU 的指令集。团队报告称，通过这种优化，速度提高了四倍。

## ML Kit 中的面部轮廓

想要快速构建类似 Snapchat 面部滤镜的功能，而不需要获得博士学位吗？ML Kit 也提供了实现这一功能所需的工具——一个用于识别面部轮廓的 API，一组点（以 x 和 y 坐标表示），这些点遵循输入图像中面部特征的形状。总共有 133 个点映射到各种面部轮廓，包括代表每只眼睛的 16 个点，而 36 个点映射到脸部周围的椭圆形状，如 [图 13-30](part0015.html#onethreethree_face_contour_points_identi) 所示。

在内部，ML Kit 正在使用 TensorFlow Lite 运行深度学习模型。在使用新的 TensorFlow Lite GPU 后端进行实验时，Google 发现与之前的 CPU 后端相比，Pixel 3 和三星 Galaxy S9 的速度提高了四倍，iPhone 7 的速度提高了六倍。这样做的效果是，我们可以实时精确地在脸上放置帽子或太阳镜。

![ML Kit 识别的 133 个面部轮廓点（图片来源）](../images/00119.jpeg)

###### 图 13-30\. ML Kit 识别的 133 个面部轮廓点（[图片来源](https://oreil.ly/8PMGa)）

## YouTube Stories 中的实时视频分割

绿幕是视频制作行业中任何人的基本设备。通过选择与主题不匹配的颜色，可以在后期制作过程中使用 *色度键* 技术更改背景。显然，这需要昂贵的软件和强大的后期制作设备。许多 YouTuber 有同样的需求，但可能没有足够的预算。现在他们在 YouTube Stories 应用程序中有了解决方案——通过 TensorFlow Lite 实现的实时视频分割选项。

这里的关键要求首先是快速运行语义分割（每秒30帧以上），其次是时间上的一致性；例如，在边缘实现平滑的帧与帧之间的时间连续性。如果我们尝试在多个帧上运行语义分割，我们很快就会注意到许多分割掩模的边缘在跳动。这里采用的关键技巧是将人脸的分割掩模传递给下一帧作为先验。因为我们传统上使用三个通道（RGB），技巧是添加第四个通道，本质上是前一帧的输出，如[图13-31](part0015.html#an_input_image_left_parenthesisleftright)所示。

![输入图像（左）被分解为其三个组件层（R、G、B）。然后将前一帧的输出掩模与这些组件连接在一起（图片来源）](../images/00202.jpeg)

###### 图13-31。输入图像（左）被分解为其三个组件层（R、G、B）。然后将前一帧的输出掩模与这些组件连接在一起（[图片来源](https://oreil.ly/rHNH5)）

通过对基础CNN结构的其他优化，YouTube团队能够在iPhone 7上以每秒100帧以上的速度运行系统，并且在Pixel 2上以每秒40帧以上的速度运行。此外，比较TensorFlow Lite的CPU与GPU后端，选择GPU后端时速度增加了18倍（与仅针对图像的其他语义分割任务通常为2到7倍的加速相比更多）。

# 摘要

在本章中，我们讨论了TensorFlow Lite的架构，以及它与Android的神经网络API的关系。然后我们着手在Android设备上运行一个简单的物体识别应用程序。我们介绍了Google的ML Kit，并讨论了您可能想要使用它的原因。此外，我们还讨论了如何将我们的TensorFlow模型转换为TensorFlow Lite格式，以便它们可以在Android设备中使用。最后，我们讨论了TensorFlow Lite如何被用来解决真实世界问题的一些示例。

在下一章中，我们将探讨如何利用实时深度学习开发交互式、真实世界的应用程序。
