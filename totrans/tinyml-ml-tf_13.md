# 第13章。用于微控制器的TensorFlow Lite

在本章中，我们将介绍我们在本书中所有示例中使用的软件框架：TensorFlow Lite for Microcontrollers。我们将详细介绍，但您不需要理解我们涵盖的所有内容才能在应用程序中使用它。如果您对底层发生的事情不感兴趣，请随意跳过本章；当您有问题时，您可以随时返回。如果您想更好地了解您用于运行机器学习的工具，我们在这里介绍了该库的历史和内部工作原理。

# 什么是用于微控制器的TensorFlow Lite？

您可能会问的第一个问题是该框架实际上是做什么的。要理解这一点，有助于稍微解释一下（相当长的）名称并解释组件。

## TensorFlow

如果您研究过机器学习，很可能已经听说过TensorFlow本身。[TensorFlow](https://tensorflow.org)是谷歌的开源机器学习库，其座右铭是“一个面向所有人的开源机器学习框架”。它是在谷歌内部开发的，并于2015年首次向公众发布。自那时以来，围绕该软件形成了一个庞大的外部社区，贡献者数量超过了谷歌内部。它面向Linux、Windows和macOS桌面和服务器平台，提供了许多工具、示例和优化，用于在云中训练和部署模型。它是谷歌内部用于支持其产品的主要机器学习库，核心代码在内部和发布版本中是相同的。

此外，谷歌和其他来源还提供了大量示例和教程。这些示例可以向您展示如何为从语音识别到数据中心电力管理或视频分析等各种用途训练和使用模型。

当TensorFlow推出时最大的需求是能够在桌面环境中训练模型并运行它们。这影响了很多设计决策，例如为了更低的延迟和更多功能性而交换可执行文件的大小-在云服务器上，即使RAM也是以吉字节计量，存储空间也有几千兆字节，拥有几百兆字节大小的二进制文件并不是问题。另一个例子是，它在推出时的主要接口语言是Python，这是一种在服务器上广泛使用的脚本语言。

然而，这些工程上的权衡对其他平台并不太适用。在Android和iPhone设备上，即使将几兆字节添加到应用程序的大小中也会大大降低下载量和客户满意度。您可以为这些手机平台构建TensorFlow，但默认情况下会将20 MB添加到应用程序大小中，即使经过一些工作，也永远不会缩小到2 MB以下。

## TensorFlow Lite

为了满足移动平台对更小尺寸的要求，谷歌在2017年启动了一个名为TensorFlow Lite的辅助项目，以便在移动设备上高效轻松地运行神经网络模型。为了减少框架的大小和复杂性，它放弃了在这些平台上不太常见的功能。例如，它不支持训练，只能在之前在云平台上训练过的模型上运行推断。它也不支持主要TensorFlow中可用的所有数据类型（如`double`）。此外，一些不常用的操作也不存在，比如`tf.depth_to_space`。您可以在[TensorFlow网站](https://oreil.ly/otEIp)上找到最新的兼容性信息。

作为这些折衷的回报，TensorFlow Lite可以适应几百千字节，使其更容易适应大小受限的应用程序。它还具有针对Arm Cortex-A系列CPU高度优化的库，以及通过OpenGL支持Android的神经网络API的加速器和GPU。另一个关键优势是它对网络的8位量化有很好的支持。因为一个模型可能有数百万个参数，仅从32位浮点数到8位整数的75%大小减少就是值得的，但还有专门的代码路径，使得推断在较小的数据类型上运行得更快。

## 微控制器的TensorFlow Lite

TensorFlow Lite已被移动开发人员广泛采用，但其工程折衷并不符合所有平台的要求。团队注意到有很多谷歌和外部产品可以从在嵌入式平台上构建的机器学习中受益，而现有的TensorFlow Lite库则不适用。再次，最大的限制是二进制大小。对于这些环境来说，即使几百千字节也太大了；他们需要适合在20 KB或更小范围内的东西。许多移动开发人员认为理所当然的依赖项，如C标准库，也不存在，因此不能使用依赖于这些库的代码。然而，许多要求非常相似。推断是主要用例，量化网络对性能很重要，并且具有足够简单以供开发人员探索和修改的代码库是首要任务。

考虑到这些需求，2018年，谷歌团队（包括本书的作者）开始尝试专门针对这些嵌入式平台的TensorFlow Lite的特殊版本。目标是尽可能重用移动项目中的代码、工具和文档，同时满足嵌入式环境的严格要求。为了确保谷歌正在构建实用的东西，团队专注于识别口头“唤醒词”的真实用例，类似于商业语音界面中的“Hey Google”或“Alexa”示例。旨在提供一个端到端的示例来解决这个问题，谷歌努力确保我们设计的系统适用于生产系统。

## 要求

谷歌团队知道在嵌入式环境中运行对代码编写有很多限制，因此确定了库的一些关键要求：

没有操作系统依赖

机器学习模型基本上是一个数学黑匣子，其中输入数字，输出结果也是数字。执行这些操作不需要访问系统的其余部分，因此可以编写一个不调用底层操作系统的机器学习框架。一些目标平台根本没有操作系统，避免在基本代码中引用文件或设备使得可以将其移植到这些芯片上。

在链接时没有标准的C或C++库依赖

这比操作系统要求更微妙一些，但团队的目标是部署在可能只有几十KB内存来存储程序的设备上，因此二进制大小非常重要。即使看似简单的函数如`sprintf()`本身可能就需要20KB的空间，因此团队的目标是避免从包含C和C++标准库实现的库存档案中提取任何内容。这很棘手，因为头文件依赖（如*stdint.h*，其中包含数据类型的大小）和标准库的链接时部分（如许多字符串函数或`sprintf()`）之间没有明确定义的边界。实际上，团队必须运用一些常识来理解，通常情况下，编译时常量和宏是可以接受的，但应避免使用更复杂的内容。唯一的例外是标准C `math`库，它被用于需要链接的三角函数等功能。

不需要浮点硬件

许多嵌入式平台不支持硬件浮点运算，因此代码必须避免对浮点数的性能关键使用。这意味着专注于具有8位整数参数的模型，并在操作中使用8位算术（尽管为了兼容性，该框架还支持浮点运算，如果需要的话）。

不支持动态内存分配

许多使用微控制器的应用程序需要连续运行数月或数年。如果程序的主循环使用`malloc()`/`new`和`free()`/`delete`来分配和释放内存，很难保证堆最终不会处于碎片化状态，导致分配失败和崩溃。大多数嵌入式系统上可用的内存非常有限，因此提前规划这种有限资源比其他平台更为重要，而且没有操作系统可能甚至没有堆和分配例程。这意味着嵌入式应用程序通常完全避免使用动态内存分配。因为该库是为这些应用程序设计的，所以它也需要这样做。实际上，该框架要求调用应用程序在初始化时传入一个小型、固定大小的区域，框架可以在其中进行临时分配（如激活缓冲区）。如果区域太小，库将立即返回错误，客户端需要重新编译以使用更大的区域。否则，进行推理调用时不会有进一步的内存分配，因此可以反复进行，而不会出现堆碎片化或内存错误的风险。

团队还决定不采用嵌入式社区中常见的其他一些约束，因为这将使共享代码和与移动TensorFlow Lite的兼容性维护变得太困难。因此：

它需要C++11

在C中编写嵌入式程序很常见，有些平台根本不支持C++，或者支持的版本比2011年的标准修订版旧。TensorFlow Lite主要是用C++编写的，具有一些纯C API，这使得从其他语言调用它更容易。它不依赖于复杂的模板等高级功能；其风格是“更好的C”，使用类来帮助模块化代码。将框架重写为C将需要大量工作，并且对于移动平台上的用户来说是一种倒退，当我们调查最受欢迎的平台时，我们发现，它们都已经支持C++11，因此团队决定牺牲对旧设备的支持，以使代码更容易在所有版本的TensorFlow Lite之间共享。

它需要32位处理器

嵌入式世界中有大量不同的硬件平台可用，但近年来的趋势是向32位处理器发展，而不是以前常见的16位或8位芯片。在调查了生态系统之后，Google决定将开发重点放在更新的32位设备上，因为这样可以保持假设，例如C `int`数据类型为32位，这样可以使移动和嵌入式版本的框架保持一致。我们已经收到了一些成功移植到一些16位平台的报告，但这些平台依赖于弥补限制的现代工具链，并不是我们的主要重点。

## 为什么要解释模型？

经常出现的一个问题是，为什么我们选择在运行时解释模型，而不是提前从模型生成代码。解释该决定涉及分析涉及的不同方法的一些好处和问题。

代码生成涉及将模型直接转换为C或C++代码，其中所有参数都存储为代码中的数据数组，架构表示为一系列函数调用，这些函数调用将激活从一层传递到下一层。这些代码通常输出到一个单独的大型源文件中，其中包含少量入口点。然后可以直接将该文件包含在IDE或工具链中，并像任何其他代码一样进行编译。以下是代码生成的一些关键优势：

易于构建

用户告诉我们，最大的好处是它有多么容易集成到构建系统中。如果您只有几个C或C++文件，没有外部库依赖项，您可以轻松地将它们拖放到几乎任何IDE中，并构建一个项目，几乎没有出错的机会。

可修改性

当您有少量代码在单个实现文件中时，如果需要，通过代码进行步进和更改会更简单，至少与首先需要确定哪些实现正在使用的大型库相比是如此。

内联数据

模型本身的数据可以存储为实现源代码的一部分，因此不需要额外的文件。它也可以直接存储为内存中的数据结构，因此不需要加载或解析步骤。

代码大小

如果您提前知道要构建的模型和平台，可以避免包含永远不会被调用的代码，因此可以保持程序段的大小最小化。

解释模型是一种不同的方法，依赖于加载定义模型的数据结构。执行的代码是静态的；只有模型数据发生变化，模型中的信息控制执行哪些操作以及从哪里提取参数。这更像是在解释语言（如Python）中运行脚本，而将代码生成视为更接近传统编译语言（如C）。以下是与解释模型数据结构相比，代码生成的一些缺点：

可升级性

如果您在本地修改了生成的代码，但想要升级到整体框架的新版本以获得新功能或优化，会发生什么？您要么需要手动将更改挑选到本地文件中，要么完全重新生成它们，然后尝试将本地更改补丁回去。

多个模型

通过代码生成很难支持多个模型，而不会有大量源代码重复。

替换模型

每个模型都表示为程序中源代码和数据数组的混合，因此很难在不重新编译整个程序的情况下更改模型。

团队意识到的是，可以通过使用我们所谓的*项目生成*来获得代码生成的许多好处，而不会遇到缺点。

## 项目生成

在TensorFlow Lite中，项目生成是一个过程，它创建了构建特定模型所需的源文件副本，而不对其进行任何更改，并且还可以选择设置任何特定于IDE的项目文件，以便可以轻松构建。它保留了大部分代码生成的好处，但它具有一些关键优势：

可升级性

所有的源文件都只是主要TensorFlow Lite代码库中原始文件的副本，并且它们出现在文件夹层次结构中的相同位置，因此如果您进行本地修改，可以轻松地将其移植回原始源，并且可以简单地使用标准合并工具合并库升级。

多个和替换模型

底层代码是一个解释器，因此您可以拥有多个模型或轻松更换数据文件而无需重新编译。

内联数据

如果需要，模型参数本身仍然可以编译到程序中作为C数据数组，并且使用FlatBuffers序列化格式意味着这种表示可以直接在内存中使用，无需解包或解析。

外部依赖

构建项目所需的所有头文件和源文件都复制到与常规TensorFlow代码相邻的文件夹中，因此不需要单独下载或安装任何依赖项。

最大的优势并不是自动获得的代码大小，因为解释器结构使得更难以发现永远不会被调用的代码路径。在TensorFlow Lite中，通过手动使用`OpResolver`机制来注册您在应用程序中期望使用的内核实现，可以单独解决这个问题。

# 构建系统

TensorFlow Lite最初是在Linux环境中开发的，因此我们的许多工具基于传统的Unix工具，如shell脚本、Make和Python。我们知道这对于嵌入式开发人员来说并不常见，因此我们旨在支持其他平台和编译工具链作为一流公民。

我们通过上述项目生成来实现这一点。如果您从GitHub获取TensorFlow源代码，可以使用Linux上的标准Makefile方法为许多平台构建。例如，这个命令行应该编译和测试库的x86版本：

```py
make -f tensorflow/lite/micro/tools/make/Makefile test
```

您可以构建特定目标，比如为SparkFun Edge平台构建语音唤醒示例，使用以下命令：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET="sparkfun_edge" micro_speech_bin
```

如果您在Windows机器上运行或想要使用Keil、Mbed、Arduino或其他专门的构建系统，那么项目生成就派上用场了。您可以通过在Linux上运行以下命令行来生成一个准备在Mbed IDE中使用的文件夹：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET="disco_f746ng" generate_micro_speech_mbed_project
```

现在，您应该在*tensorflow/lite/micro/tools/make/gen/disco_f746ng_x86_64/prj/micro_speech/mbed/*中看到一组源文件，以及在Mbed环境中构建所需的所有依赖项和项目文件。同样的方法适用于Keil和Arduino，还有一个通用版本，只输出源文件的文件夹层次结构，不包括项目元信息（尽管它包括一个定义了一些构建规则的Visual Studio Code文件）。

您可能想知道这种Linux命令行方法如何帮助其他平台上的用户。我们会自动将此项目生成过程作为我们每晚的持续集成工作流的一部分以及每次进行重大发布时运行。每次运行时，它会自动将生成的文件放在公共Web服务器上。这意味着所有平台上的用户应该能够找到适合其首选IDE的版本，并且可以下载该项目作为一个独立的文件夹，而不是通过GitHub。

## 专门化代码

代码生成的一个好处是很容易重写库的部分，使其在特定平台上运行良好，甚至只是针对你知道在你的用例中很常见的一组特定参数进行函数优化。我们不想失去这种修改的便利性，但我们也希望尽可能地使更普遍有用的更改能够轻松地合并回主框架的源代码中。我们还有一个额外的约束条件，即一些构建环境在编译过程中不容易传递自定义的`#define`宏，因此我们不能依赖于在编译时使用宏保护切换到不同的实现。

为了解决这个问题，我们将库拆分为小模块，每个模块都有一个实现其功能的单个C++文件，以及一个定义其他代码可以调用以使用该模块的接口的C++头文件。然后我们采用了一个约定，如果您想编写一个模块的专门版本，您将您的新版本保存为与原始文件同名但在原始文件所在目录的子文件夹中的C++实现文件。这个子文件夹应该有您专门为其进行特化的平台或功能的名称（参见[图13-1](#specialize_screenshot)），并且在为该平台或功能构建时将自动使用Makefile或生成的项目而不是原始实现。这可能听起来很复杂，所以让我们通过几个具体的例子来解释一下。

语音唤醒词示例代码需要从麦克风中获取音频数据，但不幸的是没有跨平台的方法来捕获音频。因为我们至少需要在各种设备上进行编译，所以我们编写了一个默认实现，它只返回一个充满零值的缓冲区，而不使用麦克风。以下是该模块的接口是什么样子的，来自[*audio_provider.h*](https://oreil.ly/J5N0N)：

```py
TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples);
int32_t LatestAudioTimestamp();
```

![专门音频提供者文件的截图](Images/timl_1301.png)

###### 图13-1\. 专门音频提供者文件的截图

第一个函数为给定时间段输出填充有音频数据的缓冲区，如果出现问题则返回错误。第二个函数返回最近捕获到的音频数据的时间戳，因此客户端可以请求正确的时间范围，并知道何时有新数据到达。

因为默认实现不能依赖于麦克风的存在，所以[*audio_provider.cc*](https://oreil.ly/8V1Ll)中的两个函数的实现非常简单：

```py
namespace {
int16_t g_dummy_audio_data[kMaxAudioSampleSize];
int32_t g_latest_audio_timestamp = 0;
}  // namespace

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  for (int i = 0; i < kMaxAudioSampleSize; ++i) {
    g_dummy_audio_data[i] = 0;
  }
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_dummy_audio_data;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() {
  g_latest_audio_timestamp += 100;
  return g_latest_audio_timestamp;
}
```

时间戳在每次调用函数时自动递增，以便客户端表现得好像有新数据进来，但捕获例程每次都返回相同的零数组。这样做的好处是可以让您在系统上的麦克风工作之前就可以对示例代码进行原型设计和实验。`kMaxAudioSampleSize`在模型头文件中定义，是函数将被要求的最大样本数。

在真实设备上，代码需要更复杂，因此我们需要一个新的实现。早些时候，我们为STM32F746NG Discovery kit开发板编译了这个示例，该开发板内置了麦克风，并使用单独的Mbed库来访问它们。代码在[*disco_f746ng/audio_provider.cc*](https://oreil.ly/KrdSO)中。这里没有内联包含它，因为它太大了，但如果您查看该文件，您会看到它实现了与默认*audio_provider.cc*相同的两个公共函数：`GetAudioSamples()`和`LatestAudioTimestamp()`。这些函数的定义要复杂得多，但从客户端的角度来看，它们的行为是相同的。复杂性被隐藏起来，调用代码可以保持不变，尽管平台发生了变化，现在，而不是每次都接收到一个零数组，捕获的音频将显示在返回的缓冲区中。

如果你查看这个专门实现的完整路径，*tensorflow/lite/micro/examples/micro_speech/disco_f746ng/audio_provider.cc*，你会发现它几乎与默认实现的*tensorflow/lite/micro/examples/micro_speech/audio_provider.cc*相同，但它位于与原始*.cc*文件相同级别的*disco_f746ng*子文件夹内。如果你回顾一下用于构建STM32F746NG Mbed项目的命令行，你会看到我们传入了`TARGET=disco_f746ng`来指定我们想要的平台。构建系统总是在目标名称的子文件夹中寻找*.cc*文件，以便可能的专门实现，因此在这种情况下，*disco_f746ng/audio_provider.cc*被用来代替父文件夹中的默认*audio_provider.cc*版本。在为Mbed项目复制源文件时，会忽略父级*.cc*文件，并复制子文件夹中的文件；因此，生成的项目将使用专门版本。

在几乎每个平台上，音频捕获的方式都不同，因此我们有许多不同的专门实现这个模块。甚至还有一个macOS版本，[*osx/audio_provider.cc*](https://oreil.ly/ZaMtF)，如果你在Mac笔记本上本地调试，这将非常有用。

这种机制不仅用于可移植性，还足够灵活以用于优化。实际上，我们在语音唤醒词示例中使用这种方法来加速深度卷积操作。如果你查看[*tensorflow/lite/micro/kernels*](https://oreil.ly/0yHNd)，你会看到TensorFlow Lite for Microcontrollers支持的所有操作的实现。这些默认实现被设计为简短、易于理解，并在任何平台上运行，但是为了达到这些目标，它们通常会错过提高运行速度的机会。优化通常涉及使算法更复杂、更难理解，因此这些参考实现预计会相对较慢。我们的想法是要让开发人员能够以最简单的方式运行代码，并确保他们获得正确的结果，然后逐步更改代码以提高性能。这意味着每个小改变都可以进行测试，以确保它不会破坏正确性，从而使调试变得更加容易。

语音唤醒词示例中使用的模型严重依赖深度卷积操作，该操作在[*tensorflow/lite/micro/kernels/depthwise_conv.cc*](https://oreil.ly/a16dw)中有一个未经优化的实现。核心算法在[*tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h*](https://oreil.ly/2gQ-e)中实现，并被写成一组嵌套循环。以下是代码本身：

```py
   for (int b = 0; b < batches; ++b) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int ic = 0; ic < input_depth; ++ic) {
            for (int m = 0; m < depth_multiplier; m++) {
              const int oc = m + ic * depth_multiplier;
              const int in_x_origin = (out_x * stride_width) - pad_width;
              const int in_y_origin = (out_y * stride_height) - pad_height;
              int32 acc = 0;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  const int in_x =
                      in_x_origin + dilation_width_factor * filter_x;
                  const int in_y =
                      in_y_origin + dilation_height_factor * filter_y;
                  // If the location is outside the bounds of the input image,
                  // use zero as a default value.
                  if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height)) {
                    int32 input_val =
                        input_data[Offset(input_shape, b, in_y, in_x, ic)];
                    int32 filter_val = filter_data[Offset(
                        filter_shape, 0, filter_y, filter_x, oc)];
                    acc += (filter_val + filter_offset) *
                           (input_val + input_offset);
                  }
                }
              }
              if (bias_data) {
                acc += bias_data[oc];
              }
              acc = DepthwiseConvRound<output_rounding>(acc, output_multiplier,
                                                        output_shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                  static_cast<uint8>(acc);
            }
          }
        }
      }
    }
```

你可能会从快速查看中看到许多加速的机会，比如在内部循环中每次计算的所有数组索引都预先计算出来。这些改变会增加代码的复杂性，因此对于这个参考实现，我们避免了它们。然而，语音唤醒词示例需要在微控制器上多次运行，结果发现这种朴素的实现是阻碍SparkFun Edge Cortex-M4处理器实现这一目标的主要速度瓶颈。为了使示例以可用的速度运行，我们需要添加一些优化。

为了提供一个优化的实现，我们在*tensorflow/lite/micro/kernels*内创建了一个名为*portable_optimized*的新子文件夹，并添加了一个名为[*depthwise_conv.cc*](https://oreil.ly/BYRho)的新的C++源文件。这比参考实现复杂得多，并利用了语音模型的特定特性来实现专门的优化。例如，卷积窗口的宽度是8的倍数，因此我们可以将值作为两个32位字从内存中加载，而不是作为8个单独的字节。

您会注意到我们将子文件夹命名为*portable_optimized*，而不是像前面的示例那样特定于平台。这是因为我们所做的更改都不与特定芯片或库绑定；它们是通用优化，预计将有助于各种处理器，例如预先计算数组索引或将多个字节值加载为更大的字。然后，我们通过将`portable_optimized`添加到[`ALL_TAGS`列表](https://oreil.ly/XSWFk)中来指定应在`make`项目文件中使用此实现。由于存在此标签，并且在具有相同名称的子文件夹中存在*depthwise_conv.cc*的实现，因此链接了优化实现，而不是默认的参考版本。

希望这些示例展示了如何利用子文件夹机制来扩展和优化库代码，同时保持核心实现简洁易懂。

## Makefiles

说到易于理解，Makefiles并不是。[Make构建系统](https://oreil.ly/8Ft1J)现在已经有40多年的历史，具有许多令人困惑的特性，比如使用制表符作为有意义的语法或通过声明性规则间接指定构建目标。我们选择使用Make而不是Bazel或Cmake等替代方案，因为它足够灵活，可以实现像项目生成这样的复杂行为，我们希望大多数TensorFlow Lite for Microcontrollers的用户会在更现代的IDE中使用这些生成的项目，而不是直接与Makefiles交互。

如果您对核心库进行更改，可能需要更深入了解Makefiles中的内部情况，因此，本节涵盖了一些您需要熟悉的约定和辅助函数，以便进行修改。

###### 注意

如果您在Linux或macOS上使用bash终端，可以通过键入正常的`make -f tensorflow/lite/micro/tools/make/Makefile`命令，然后按Tab键来查看所有可用的目标（可以构建的内容的名称）。在查找或调试目标时，此自动完成功能非常有用。

如果您只是添加一个模块或操作的专门版本，您根本不需要更新Makefile。有一个名为[`specialize()`](https://oreil.ly/teIF6)的自定义函数，它会自动获取字符串（包含平台名称以及任何自定义标签）的`ALL_TAGS`列表和源文件列表，并返回替换原始版本的正确专门版本的列表。这也使您有灵活性，在命令行上手动指定标签。例如，这样：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET="bluepill" TAGS="portable_optimized foo" test
```

将生成一个看起来像“bluepill portable_optimized foo”的`ALL_TAGS`列表，对于每个源文件，将按顺序搜索子文件夹以查找任何专门的版本来替换。

如果您只是向标准文件夹添加新的C++文件，也不需要修改Makefile，因为大多数情况下这些文件会被通配符规则自动捕捉，比如[`MICROLITE_CC_BASE_SRCS`](https://oreil.ly/QAtDk)的定义。

Makefile依赖于在根级别定义要构建的源文件和头文件列表，然后根据指定的平台和标签进行修改。这些修改发生在从父构建项目包含的子Makefiles中。例如，[*tensorflow/lite/micro/tools/make/targets*](https://oreil.ly/79zOB)文件夹中的所有*.inc*文件都会自动包含。如果您查看其中一个，比如用于Ambiq和SparkFun Edge平台的[*apollo3evb_makefile.inc*](https://oreil.ly/gKKXO)，您会看到它检查了是否已为此构建指定了目标芯片；如果有，它会定义许多标志并修改源列表。以下是包含一些最有趣部分的简化版本：

```py
ifeq ($(TARGET),$(filter $(TARGET),apollo3evb sparkfun_edge))
  export PATH := $(MAKEFILE_DIR)/downloads/gcc_embedded/bin/:$(PATH)
  TARGET_ARCH := cortex-m4
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
...
  $(eval $(call add_third_party_download,$(GCC_EMBEDDED_URL), \
      $(GCC_EMBEDDED_MD5),gcc_embedded,))
  $(eval $(call add_third_party_download,$(CMSIS_URL),$(CMSIS_MD5),cmsis,))
...
  PLATFORM_FLAGS = \
    -DPART_apollo3 \
    -DAM_PACKAGE_BGA \
    -DAM_PART_APOLLO3 \
    -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
...
  LDFLAGS += \
    -mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
    -nostartfiles -static \
    -Wl,--gc-sections -Wl,--entry,Reset_Handler \
...
  MICROLITE_LIBS := \
    $(BOARD_BSP_PATH)/gcc/bin/libam_bsp.a \
    $(APOLLO3_SDK)/mcu/apollo3/hal/gcc/bin/libam_hal.a \
    $(GCC_ARM)/lib/gcc/arm-none-eabi/7.3.1/thumb/v7e-m/fpv4-sp/hard/crtbegin.o \
    -lm
  INCLUDES += \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/Core/Include/ \
    -isystem$(MAKEFILE_DIR)/downloads/cmsis/CMSIS/DSP/Include/ \
    -I$(MAKEFILE_DIR)/downloads/CMSIS_ext/ \
...
  MICROLITE_CC_SRCS += \
    $(APOLLO3_SDK)/boards/apollo3_evb/examples/hello_world/gcc_patched/ \
        startup_gcc.c \
    $(APOLLO3_SDK)/utils/am_util_delay.c \
    $(APOLLO3_SDK)/utils/am_util_faultisr.c \
    $(APOLLO3_SDK)/utils/am_util_id.c \
    $(APOLLO3_SDK)/utils/am_util_stdio.c
```

这是特定平台的所有定制发生的地方。在这段代码中，我们指示构建系统在哪里找到我们想要使用的编译器，并指定要使用的架构。我们指定了一些额外的外部库要下载，如GCC工具链和Arm的CMSIS库。我们为构建设置编译标志，并传递给链接器的参数，包括要链接的额外库归档文件和要查找头文件的包含路径。我们还添加了一些我们需要在Ambiq平台上成功构建的额外C文件。

构建示例时也使用了类似的子Makefile包含。语音唤醒词示例代码在[*micro_speech/Makefile.inc*](https://oreil.ly/XjuJP)中有自己的Makefile，并定义了要编译的源代码文件列表，以及要下载的额外外部依赖项。

您可以使用[`generate_microlite_projects()`](https://oreil.ly/iv94T)函数为不同的IDE生成独立项目。这将接受一组源文件和标志，然后将所需文件复制到一个新文件夹中，以及构建系统所需的任何其他项目文件。对于某些IDE，这非常简单，但例如Arduino需要将所有*.cc*文件重命名为*.cpp*，并且在复制时需要更改源文件中的一些包含路径。

外部库，如用于嵌入式Arm处理器的C++工具链，将作为Makefile构建过程的一部分自动下载。这是因为对每个所需库调用的[`add_third_party_download`](https://oreil.ly/E9tS-)规则，传入一个URL以拉取文件，并传入一个MD5校验和以检查归档文件以确保正确性。这些文件应为ZIP、GZIP、BZ2或TAR文件，根据文件扩展名将调用适当的解压程序。如果构建目标需要这些文件中的头文件或源文件，则应明确包含在Makefile中的文件列表中，以便将其复制到任何生成的项目中，因此每个项目的源树都是自包含的。这很容易被忽略，因为设置包含路径足以使Makefile编译正常工作，而无需明确提及每个包含的文件，但生成的项目将无法构建。您还应确保包含任何许可文件在您的文件列表中，以便外部库的副本保留正确的归属。

## 编写测试

TensorFlow旨在为其所有代码编写单元测试，我们已经在[第5章](ch05.xhtml#chapter_building_an_application)中详细介绍了其中一些测试。这些测试通常安排为与正在测试的模块相同文件夹中的*_test.cc*文件，并具有与原始源文件相同的前缀。例如，深度卷积操作的实现通过[*tensorflow/lite/micro/kernels/depthwise_conv_test.cc*](https://oreil.ly/eIiRO)进行测试。如果要添加新的源文件，如果要将修改提交回主树，则必须添加一个相应的单元测试来测试它。这是因为我们需要支持许多不同的平台和模型，许多人正在我们的代码之上构建复杂系统，因此重要的是我们的核心组件可以检查正确性。

如果您在*tensorflow/tensorflow/lite/experimental/micro*的直接子文件夹中添加文件，您应该能够将其命名为*<something>_test.cc*，并且它将被自动捕获。如果您正在测试示例内的模块，则需要向`microlite_test` Makefile辅助函数添加显式调用，例如[此处](https://oreil.ly/wkYgu)：

```py
# Tests the feature provider module using the mock audio provider.
$(eval $(call microlite_test,feature_provider_mock_test,\
$(FEATURE_PROVIDER_MOCK_TEST_SRCS),$(FEATURE_PROVIDER_MOCK_TEST_HDRS)))
```

测试本身需要在微控制器上运行，因此它们必须遵守围绕动态内存分配、避免OS和外部库依赖的相同约束，这是框架旨在满足的。不幸的是，这意味着像[Google Test](https://oreil.ly/GZWdj)这样的流行单元测试系统是不可接受的。相反，我们编写了自己非常简化的测试框架，定义和实现在[*micro_test.h*](https://oreil.ly/GcIbP)头文件中。

要使用它，创建一个包含头文件的*.cc*文件。在新行上以`TF_LITE_MICRO_TESTS_BEGIN`语句开始，然后定义一系列测试函数，每个函数都有一个`TF_LITE_MICRO_TEST()`宏。在每个测试中，您调用像`TF_LITE_MICRO_EXPECT_EQ()`这样的宏来断言您希望从正在测试的函数中看到的预期结果。在所有测试函数的末尾，您将需要`TF_LITE_MICRO_TESTS_END`。这里是一个基本示例：

```py
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SomeTest) {
  TF_LITE_LOG_EXPECT_EQ(true, true);
}

TF_LITE_MICRO_TESTS_END
```

如果您为您的平台编译此代码，您将获得一个正常的二进制文件，您应该能够运行它。执行它将输出类似于这样的日志信息到`stderr`（或者在您的平台上由`ErrorReporter`写入的任何等效内容）：

```py
----------------------------------------------------------------------------
Testing SomeTest
1/1 tests passed
~~~ALL TESTS PASSED~~~
----------------------------------------------------------------------------
```

这是为了便于人类阅读，因此您可以手动运行测试，但只有在所有测试确实通过时，字符串`~~~ALL TESTS PASSED~~~`才应该出现。这使得可以通过扫描输出日志并查找该魔术值来与自动化测试系统集成。这就是我们能够在微控制器上运行测试的方式。只要有一些调试日志连接回来，主机就可以刷新二进制文件，然后监视输出日志以确保预期的字符串出现以指示测试是否成功。

# 支持新硬件平台

TensorFlow Lite for Microcontrollers项目的主要目标之一是使在许多不同设备、操作系统和架构上运行机器学习模型变得容易。核心代码被设计为尽可能可移植，构建系统编写为使引入新环境变得简单。在本节中，我们提供了一个逐步指南，以在新平台上运行TensorFlow Lite for Microcontrollers。

## 打印到日志

TensorFlow Lite绝对需要的唯一平台依赖是能够将字符串打印到可以从桌面主机机器外部检查的日志中。这样我们就可以看到测试是否成功运行，并通常调试我们正在运行的程序内部发生的情况。由于这是一个困难的要求，您在您的平台上需要做的第一件事是确定可用的日志记录设施类型，然后编写一个小程序来打印一些内容以测试它们。

在Linux和大多数其他桌面操作系统上，这将是许多C培训课程的经典“hello world”示例。它通常看起来像这样：

```py
#include <stdio.h>

int main(int argc, char** argv) {
  fprintf(stderr, "Hello World!\n");
}
```

如果您在Linux、macOS或Windows上编译和构建此代码，然后从命令行运行可执行文件，您应该会在终端上看到“Hello World!”打印出来。如果微控制器正在运行高级操作系统，它可能也会工作，但至少您需要弄清楚文本本身出现在哪里，因为嵌入式系统本身没有显示器或终端。通常，您需要通过USB或其他调试连接连接到桌面机器才能查看任何日志，即使在编译时支持`fprintf()`。

从微控制器的角度来看，这段代码有一些棘手的部分。其中一个问题是，*stdio.h*库需要链接函数，其中一些函数非常庞大，可能会使二进制文件大小超出小型设备可用的资源。该库还假定所有常规的C标准库设施都可用，如动态内存分配和字符串函数。而在嵌入式系统上，`stderr`应该放在哪里并没有自然的定义，因此API不清晰。

相反，大多数平台定义了自己的调试日志接口。这些接口的调用方式通常取决于主机和微控制器之间使用的连接类型，以及嵌入式系统上运行的硬件架构和操作系统（如果有）。例如，Arm Cortex-M微控制器支持[*semihosting*](https://oreil.ly/LmC4k)，这是在开发过程中在主机和目标系统之间通信的标准。如果你正在使用类似[OpenOCD](https://oreil.ly/lSn0n)的连接从主机机器上，从微控制器调用[`SYS_WRITE0`](https://oreil.ly/6IyrK)系统调用将导致寄存器1中的零终止字符串参数显示在OpenOCD终端上。在这种情况下，等效“hello world”程序的代码将如下所示：

```py
void DebugLog(const char* s) {
  asm("mov r0, #0x04\n"  // SYS_WRITE0
      "mov r1, %[str]\n"
      "bkpt #0xAB\n"
      :
      : [ str ] "r"(s)
      : "r0", "r1");
}

int main(int argc, char** argv) {
  DebugLog("Hello World!\n");
}
```

这里需要汇编的原因显示了这个解决方案有多么特定于平台，但它确实避免了完全不引入任何外部库的需要（甚至是标准C库）。

如何做到这一点在不同平台上会有很大差异，但一个常见的方法是使用串行UART连接到主机。这是在Mbed上如何做的：

```py
#include <mbed.h>

// On mbed platforms, we set up a serial port and write to it for debug logging.
void DebugLog(const char* s) {
  static Serial pc(USBTX, USBRX);
  pc.printf("%s", s);
}

int main(int argc, char** argv) {
  DebugLog("Hello World!\n");
}
```

这里有一个稍微复杂一点的Arduino示例：

```py
#include "Arduino.h"

// The Arduino DUE uses a different object for the default serial port shown in
// the monitor than most other models, so make sure we pick the right one. See
// https://github.com/arduino/Arduino/issues/3088#issuecomment-406655244
#if defined(__SAM3X8E__)
#define DEBUG_SERIAL_OBJECT (SerialUSB)
#else
#define DEBUG_SERIAL_OBJECT (Serial)
#endif

// On Arduino platforms, we set up a serial port and write to it for debug
// logging.
void DebugLog(const char* s) {
  static bool is_initialized = false;
  if (!is_initialized) {
    DEBUG_SERIAL_OBJECT.begin(9600);
    // Wait for serial port to connect. Only needed for some models apparently?
    while (!DEBUG_SERIAL_OBJECT) {
    }
    is_initialized = true;
  }
  DEBUG_SERIAL_OBJECT.println(s);
}

int main(int argc, char** argv) {
  DebugLog("Hello World!\n");
}
```

这两个示例都创建了一个串行对象，然后期望用户将串行连接到微控制器上的主机机器上。

移植工作的关键第一步是为你的平台创建一个最小示例，在你想要使用的IDE中运行，以某种方式将一个字符串打印到主机控制台。如果你能让这个工作起来，你使用的代码将成为你将添加到TensorFlow Lite代码中的专门函数的基础。

## 实现DebugLog()

如果你查看[*tensorflow/lite/micro/debug_log.cc*](https://oreil.ly/Lka3T)，你会看到`DebugLog()`函数的实现，看起来与我们展示的第一个“hello world”示例非常相似，使用*stdio.h*和`fprintf()`将字符串输出到控制台。如果你的平台完全支持标准C库，并且不介意额外的二进制文件大小，你可以使用这个默认实现，忽略本节的其余部分。不过，更有可能的是你需要使用不同的方法。

作为第一步，我们将使用已经存在的`DebugLog()`函数的测试。首先，运行以下命令行：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  generate_micro_error_reporter_test_make_project
```

当你查看*tensorflow/lite/micro/tools/make/gen/linux_x86_64/prj/micro_error_reporter_test/make/*（如果你在不同的主机平台上，请将*linux*替换为*osx*或*windows*），你应该会看到一些像*tensorflow*和*third_party*这样的文件夹。这些文件夹包含C++源代码，如果你将它们拖入你的IDE或构建系统并编译所有文件，你应该会得到一个可执行文件，用于测试我们需要创建的错误报告功能。你第一次尝试构建这段代码很可能会失败，因为它仍在使用[*debug_log.cc*](https://oreil.ly/fDkLh)中的默认`DebugLog()`实现，依赖于*stdio.h*和C标准库。为了解决这个问题，修改*debug_log.cc*，删除`#include` `<cstdio>`语句，并用一个什么都不做的实现替换`DebugLog()`：

```py
#include "tensorflow/lite/micro/debug_log.h"

extern "C" void DebugLog(const char* s) {
  // Do nothing for now.
}
```

更改后，尝试成功编译一组源文件。完成后，将生成的二进制文件加载到嵌入式系统上。如果可以的话，检查程序是否可以正常运行，尽管您目前还看不到任何输出。

当程序似乎构建和运行正确时，请查看是否可以使调试日志记录正常工作。将您在上一节“hello world”程序中使用的代码放入*debug_log.cc*中的`DebugLog()`实现中。

实际的测试代码存在于[*tensorflow/lite/micro/micro_error_reporter_test.cc*](https://oreil.ly/0jD00)，看起来是这样的：

```py
int main(int argc, char** argv) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  error_reporter->Report("Number: %d", 42);
  error_reporter->Report("Badly-formed format string %");
  error_reporter->Report("Another % badly-formed %% format string");
  error_reporter->Report("~~~%s~~~", "ALL TESTS PASSED");
}
```

它不直接调用`DebugLog()`，而是通过处理变量数量等内容的`ErrorReporter`接口，但它确实依赖于您刚刚编写的代码作为其基础实现。如果一切正常，您应该在调试控制台中看到类似以下内容：

```py
Number: 42
Badly-formed format string
Another  badly-formed  format string
~~~ALL TESTS PASSED~~~
```

在这方面工作后，您将希望将`DebugLog()`的实现放回主源代码树中。为此，您将使用我们之前讨论过的子文件夹专业化技术。您需要决定一个短名称（不含大写字母、空格或其他特殊字符），用于标识您的平台。例如，我们已经支持的一些平台使用*arduino*、*sparkfun_edge*和*linux*。在本教程中，我们将使用*my_mcu*。首先，在您从GitHub检出的源代码副本中的*tensorflow/lite/micro/*中创建一个名为*my_mcu*的新子文件夹（不是您刚生成或下载的那个）。将带有您实现的*debug_log.cc*文件复制到该*my_mcu*文件夹中，并使用Git进行源代码跟踪。将生成的项目文件复制到备份位置，然后运行以下命令：

```py
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=my_mcu clean
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=my_mcu generate_micro_error_reporter_test_make_project
```

如果您现在查看*tensorflow/lite/micro/tools/make/gen/my_mcu_x86_64/prj/micro_error_reporter_test/make/tensorflow/lite/micro/*，您会发现默认的*debug_log.cc*不再存在，而是在*my_mcu*子文件夹中。如果您将这组源文件拖回到您的IDE或构建系统中，您现在应该看到一个成功构建、运行并输出到调试控制台的程序。

## 运行所有目标

如果成功，恭喜：您现在已启用所有TensorFlow测试和可执行目标！实现调试日志记录是您需要进行的唯一必需的特定于平台的更改；代码库中的其他所有内容应该以足够便携的方式编写，以便在任何支持C++11的工具链上构建和运行，无需标准库链接，只需使用`math`库。要创建所有目标，以便在IDE中尝试它们，您可以从终端运行以下命令：

```py
make -f tensorflow/lite/micro/tools/make/Makefile generate_projects \
  TARGET=my_mcu
```

这将在与生成的错误报告测试类似的位置创建大量文件夹，每个文件夹都会测试库的不同部分。如果您想在您的平台上运行语音唤醒词示例，您可以查看*tensorflow/lite/micro/tools/make/gen/my_mcu_x86_64/prj/micro_speech/make/*。

现在您已经实现了`DebugLog()`，它应该在您的平台上运行，但它不会执行任何有用的操作，因为默认的*audio_provider.cc*实现总是返回全零数组。要使其正常工作，您需要创建一个专门的*audio_provider.cc*模块，返回捕获的声音，使用之前描述的子文件夹专业化方法。如果您不关心一个工作演示，您仍然可以查看使用相同示例代码在您的平台上的神经网络推理延迟等内容，或者其他一些测试。

除了支持传感器和LED等输出设备的硬件支持外，您可能还希望实现更快运行的神经网络运算符版本，通过利用您平台的特殊功能。我们欢迎这种专门优化，并希望子文件夹专用化技术能够很好地将它们整合回主源树中，如果它们被证明是有用的。

## 与Makefile构建集成

到目前为止，我们只讨论了如何使用自己的集成开发环境，因为对许多嵌入式程序员来说，这通常比使用我们的Make系统更简单和更熟悉。如果您希望能够通过我们的持续集成构建来测试您的代码，或者希望在特定集成开发环境之外使用它，您可能希望更全面地将您的更改与我们的Makefiles集成。其中一个关键是找到适用于您平台的可公开下载的工具链，以及任何SDK或其他依赖项的公开下载，这样一个shell脚本就可以自动获取构建所需的一切，而无需担心网站登录或注册。例如，我们从Arm下载macOS和Linux版本的GCC嵌入式工具链，URL在[*tensorflow/lite/micro/tools/make/third_party_downloads.inc*]中。

然后，您需要确定传递给编译器和链接器的正确命令行标志，以及您需要的任何额外源文件，这些文件无法使用子文件夹专用化找到，并将这些信息编码到[*tensorflow/lite/micro/tools/make/targets*]中的一个子Makefile中。如果您想获得额外的学分，您可以尝试使用类似[Renode](https://renode.io/)的工具在x86服务器上模拟您的微控制器，以便我们可以在持续集成期间运行测试，而不仅仅是确认构建。您可以在[*tensorflow/lite/micro/testing/test_bluepill_binary.sh*]中看到我们使用Renode来测试“Bluepill”二进制文件的脚本示例。

如果您已经正确配置了所有构建设置，您将能够运行类似以下的命令来生成可刷写的二进制文件（根据您的平台设置目标）：

```py
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=bluepill micro_error_reporter_test_bin
```

如果您已经正确配置了运行测试的脚本和环境，您可以这样做来运行平台的所有测试：

```py
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=bluepill test
```

# 支持新的集成开发环境或构建系统

TensorFlow Lite for Microcontrollers可以为Arduino、Mbed和Keil工具链创建独立项目，但我们知道许多嵌入式工程师使用许多其他开发环境。如果您需要在新环境中运行框架，我们建议的第一步是查看在生成Make项目时生成的“原始”文件集是否可以导入到您的集成开发环境中。这种项目存档仅包含特定目标所需的源文件，包括任何第三方依赖项，因此在许多情况下，您只需将您的工具链指向根文件夹，并要求它包含一切。

###### 注意

当您只有少量文件时，将它们保留在原始源树的嵌套子文件夹（如*tensorflow/lite/micro/examples/micro_speech*）中，当您将它们导出到生成的项目时，可能会显得有些奇怪。将目录层次结构展平会更有意义吗？

我们选择保持深度嵌套的文件夹结构是为了尽可能简化合并回主源树，即使在处理生成的项目文件时可能不太方便。如果原始代码从GitHub检出并在每个项目中的副本之间的路径始终匹配，跟踪更改和更新就会更容易。

不幸的是，这种方法并不适用于所有IDE。例如，Arduino库要求所有C++源代码文件的后缀为*.cpp*，而不是TensorFlow默认的*.cc*，它们也无法指定包含路径，因此当我们将原始文件复制到Arduino目标时，我们需要在代码中更改路径。为了支持这些更复杂的转换，我们在Makefile构建中有一些规则和脚本，根函数[`generate_microlite_projects()`](https://oreil.ly/YYoHm)调用每个IDE的专门版本，然后依赖于更多的[规则](https://oreil.ly/KHo7G)、[Python脚本](https://oreil.ly/BKLhn)和[模板文件](https://oreil.ly/tDFhh)来创建最终输出。如果你需要为自己的IDE做类似的事情，你需要使用Makefile添加类似的功能，这并不容易实现，因为构建系统相当复杂。

# 在项目和存储库之间集成代码更改

代码生成系统的一个最大缺点是，你最终会得到源代码的多个副本分散在不同的位置，这使得处理代码更新变得非常棘手。为了最小化合并更改的成本，我们采用了一些惯例和推荐的程序，这应该会有所帮助。最常见的用例是，你对本地项目副本中的文件进行了一些修改，然后想要更新到新版本的TensorFlow Lite框架以获得额外的功能或错误修复。以下是我们建议处理该过程的方法：

1.  要么下载一个IDE和目标的项目文件的预构建存档，要么使用你感兴趣的框架版本从Makefile手动生成一个。

1.  将这组新文件解压到一个文件夹中，并确保新文件夹和包含你一直在修改的项目文件的文件夹之间的文件夹结构匹配。例如，两者顶层都应该有*tensorflow*子文件夹。

1.  在两个文件夹之间运行合并工具。你使用的工具将取决于你的操作系统，但[Meld](https://meldmerge.org/)是一个在Linux、Windows和macOS上都能工作的不错选择。合并过程的复杂程度将取决于你在本地更改了多少文件，但预计大部分差异将是在框架方面的更新，所以你通常应该能够选择“接受他们”的等效选项。

如果你只在本地修改了一个或两个文件，可能更容易的方法是从旧版本中复制修改后的代码，然后手动合并到新导出的项目中。

你也可以通过将修改后的代码提交到Git中，将最新的项目文件导入为一个新的分支，然后使用Git内置的合并工具来处理集成。我们还不够高级，无法提供关于这种方法的建议，所以我们自己也没有使用过。

这个过程与使用更传统的代码生成方法做同样的事情的区别在于，代码仍然分成许多逻辑文件，其路径随时间保持不变。典型的代码生成会将所有源代码连接成一个文件，这样合并或跟踪更改就变得非常困难，因为对顺序或布局的微小更改会使历史比较变得不可能。

有时候你可能想要将变更从项目文件合并到主源代码树中。这个主源代码树不需要是[GitHub上的官方仓库](https://oreil.ly/o8Ytb)；它可以是你维护并且不分发的本地分支。我们很乐意接收主仓库的拉取请求，包括修复或升级，但我们知道在专有嵌入式开发中这并不总是可能，所以我们也很乐意帮助保持分支的健康。关键是要注意，你要尽量保持开发文件的单一“真相源”。特别是如果你有多个开发者，很容易在项目存档中的不同本地副本中进行不兼容的更改，这会使更新和调试变得一团糟。无论是仅内部使用还是公开共享，我们强烈建议使用一个源代码控制系统，每个文件只有一个副本，而不是检入多个版本。

为了处理将更改迁移到真相源仓库，你需要跟踪你修改过的文件。如果你没有这些信息，你可以随时回到最初下载或生成的项目文件，并运行diff来查看有什么变化。一旦你知道哪些文件被修改或新增了，只需将它们复制到Git（或其他源代码控制系统）仓库中，路径与项目文件中的路径相同。

唯一的例外是第三方库的文件，因为这些文件在TensorFlow仓库中不存在。提交这些文件的更改超出了本书的范围——这个过程将取决于每个单独仓库的规则——但作为最后手段，如果你的更改没有被接受，你通常可以在GitHub上fork该项目，并将你的平台构建系统指向那个新的URL，而不是原始URL。假设你只是在更改TensorFlow源文件，那么现在你应该有一个包含你的更改的本地修改过的仓库。为了验证这些修改已经成功集成，你需要使用Make运行`generate_projects()`，然后确保你的IDE和目标项目已经应用了你期望的更新。当这一切完成，并且你已经运行了测试以确保没有其他问题，你可以将你的更改提交到你的TensorFlow分支。一旦完成，最后一步是提交一个拉取请求，如果你希望看到你的更改被公开。

# 回馈开源Contributing Back to Open Source

TensorFlow外部的贡献者已经比内部的更多，而微控制器工作更多地依赖于协作。我们非常渴望得到社区的帮助，其中通过拉取请求是最重要的帮助方式之一（虽然还有很多其他方式，比如[Stack Overflow](https://oreil.ly/7btPw)或创建你自己的示例项目）。GitHub有很好的[文档](https://oreil.ly/8rDKL)涵盖了拉取请求的基础知识，但在使用TensorFlow时有一些细节是有帮助的：

+   我们有一个由内外部Google项目维护者运行的代码审查流程。这是通过GitHub的代码审查系统管理的，所以你应该期望在那里看到关于你提交的讨论。

+   不仅仅是修复错误或优化的更改通常需要先有一个设计文档。有一个名为[SIG Micro](https://oreil.ly/JKiwD)的组织，由外部贡献者运营，帮助定义我们的优先事项和路线图，所以这是一个讨论新设计的好地方。这个文档可以只有一页或两页，对于较小的更改来说，了解拉取请求背后的背景和动机是有帮助的。

+   维护一个公共分支可以是在提交到主分支之前获取实验性变更反馈的好方法，因为您可以进行任何繁琐的流程更改而不会拖慢您的速度。

+   有自动化测试针对所有拉取请求运行，包括公开的和一些额外的谷歌内部工具，检查与我们依赖的项目的集成。遗憾的是，这些测试的结果有时很难解释，甚至更糟糕的是，它们偶尔会出现“不稳定”的情况，测试失败的原因与您的更改无关。我们一直在努力改进这个过程，因为我们知道这是一个糟糕的体验，但如果您在理解测试失败方面遇到困难，请在对话线程中联系维护者。

+   我们的目标是实现100%的测试覆盖率，因此如果一个变更没有被现有的测试覆盖到，我们会要求您提供一个新的测试。这些测试可以非常简单；我们只是想确保我们所做的一切都有一定的覆盖范围。

+   为了可读性起见，我们在整个TensorFlow代码库中一致使用Google的C和C++代码格式指南，因此我们要求任何新的或修改过的代码都采用这种风格。您可以使用[`clang-format`](https://oreil.ly/KkRKL)并使用`google`风格参数自动格式化您的代码。

非常感谢您对TensorFlow所做的任何贡献，以及对提交变更所涉及工作的耐心。这并不总是容易的，但您将对全球许多开发人员产生影响！

# 支持新的硬件加速器

TensorFlow Lite for Microcontrollers的一个目标是成为一个参考软件平台，帮助硬件开发人员更快地推进他们的设计。我们观察到，让一个新芯片在机器学习中做一些有用的事情的工作很大一部分在于诸如从训练环境编写导出器之类的任务，特别是涉及到量化和实现机器学习模型所需的“长尾”操作等棘手细节。这些任务所需的时间很少，它们不适合进行硬件优化。

为了解决这些问题，我们希望硬件开发人员将采取的第一步是在其平台上运行TensorFlow Lite for Microcontrollers的未优化参考代码，并产生正确的结果。这将证明除了硬件优化之外的一切都在运行，因此可以将剩下的工作重点放在硬件优化上。一个挑战可能是如果芯片是一个不支持通用C++编译的加速器，因为它只具有专门的功能而不是传统的CPU。对于嵌入式用例，我们发现几乎总是需要一些通用计算能力，即使它很慢（比如一个小型微控制器），因为许多用户的图形操作无法紧凑地表达，除非作为任意的C++实现。我们还做出了设计决策，即TensorFlow Lite for Microcontrollers解释器不支持子图的异步执行，因为这将使代码变得更加复杂，而且在嵌入式领域似乎不常见（不像移动世界，Android的神经网络API很受欢迎）。

这意味着TensorFlow Lite for Microcontrollers支持的架构类型看起来更像是与传统处理器同步运行的协处理器，加速器加速计算密集型函数，否则这些函数将需要很长时间，但将更灵活要求更小的操作推迟到CPU。实际上，我们建议首先通过在内核级别替换单个操作符实现来调用任何专门的硬件。这意味着结果和输入预计将在CPU可寻址的正常内存中，因为您无法保证后续操作将在哪个处理器上运行，并且您将需要等待加速器完成后才能继续，或者使用特定于平台的代码切换到微框架之外的线程。尽管存在这些限制，但至少应该能够进行一些快速的原型设计，并希望能够在始终能够测试每个小修改的正确性的同时进行增量更改。

# 理解文件格式

TensorFlow Lite用于存储其模型的格式具有许多优点，但不幸的是简单性不是其中之一。不过，不要被复杂性吓倒；一旦理解了一些基本原理，就会发现实际上很容易处理。

正如我们在[第3章](ch03.xhtml#chapter_get_up_to_speed)中提到的，神经网络模型是具有输入和输出的操作图。某些操作的输入可能是大型数组，称为权重，而其他输入可能来自先前操作的结果，或者由应用层提供的输入值数组。这些输入可能是图像像素、音频样本数据或加速度计时间序列数据。在运行模型的单次传递结束时，最终操作将在它们的输出中留下值数组，通常表示不同类别的分类预测等内容。

模型通常在台式机上进行训练，因此我们需要一种将其转移到手机或微控制器等其他设备的方法。在TensorFlow世界中，我们使用一个转换器来将从Python中训练的模型导出为TensorFlow Lite文件。这个导出阶段可能会遇到问题，因为很容易在TensorFlow中创建一个依赖于桌面环境特性的模型（比如能够执行Python代码片段或使用高级操作），而这些特性在简单平台上不受支持。还需要将训练中可变的所有值（如权重）转换为常量，删除仅用于梯度反向传播的操作，并执行优化，如融合相邻操作或将昂贵的操作（如批量归一化）折叠为更便宜的形式。更加棘手的是，主线TensorFlow中有800多个操作，而且新的操作一直在增加。这意味着编写自己的转换器来处理一小部分模型是相当简单的，但要可靠地处理用户在TensorFlow中创建的更广泛的网络范围则更加困难。跟上新操作的步伐已经是一项全职工作了。

转换过程中得到的TensorFlow Lite文件不会受到大多数这些问题的影响。我们试图生成一个更简单、更稳定的训练模型表示，具有清晰的输入和输出，将变量“冻结”为权重，并进行常见的图优化，如已应用的融合。这意味着即使您不打算在微控制器上使用TensorFlow Lite，我们也建议使用TensorFlow Lite文件格式作为访问TensorFlow模型进行推断的方式，而不是从Python层编写自己的转换器。

## FlatBuffers

我们使用[FlatBuffers](https://oreil.ly/jfoBx)作为我们的序列化库。它专为性能关键的应用程序设计，因此非常适合嵌入式系统。其中一个好处是，它的运行时内存表示与其序列化形式完全相同，因此模型可以直接嵌入到闪存中并立即访问，无需任何解析或复制。这意味着生成的代码类用于读取属性可能有点难以理解，因为存在几层间接引用，但重要数据（如权重）直接存储为可以像原始C数组一样访问的小端blob。也几乎没有浪费空间，因此使用FlatBuffers不会产生大小惩罚。

FlatBuffers使用*模式*来定义我们要序列化的数据结构，以及一个编译器，将该模式转换为本机C++（或C、Python、Java等）代码，用于读取和写入信息。对于TensorFlow Lite，模式位于[*tensorflow/lite/schema/schema.fbs*](https://oreil.ly/JoDE9)，我们将生成的C++访问器代码缓存到[*tensorflow/lite/schema/schema_generated.h*](https://oreil.ly/LjxOp)。我们可以在每次进行新构建时生成C++代码，而不是将其存储在源代码控制中，但这将要求我们构建的每个平台都包含`flatc`编译器以及其他工具链的其余部分，我们决定牺牲自动生成的便利性以换取易于移植。

如果您想了解字节级别的格式，我们建议查看FlatBuffers C++项目的[内部页面](https://oreil.ly/EBg3-)，或者[C库的等效页面](https://oreil.ly/xXkZg)。我们希望大多数需求都可以通过各种高级语言接口来满足，因此您不需要以那种粒度工作。为了向您介绍格式背后的概念，我们将逐步介绍模式和`MicroInterpreter`中读取模型的代码；希望具体示例将有助于理解。

具有讽刺意味的是，要开始，我们需要滚动到[模式的最末尾](https://oreil.ly/aHYM-)。在这里，我们看到一行声明`root_type`为`Model`：

```py
root_type Model;
```

FlatBuffers需要一个作为文件中包含的其他数据结构树的根的单个容器对象。这个声明告诉我们，这种格式的根将是一个`Model`。要了解这意味着什么，我们再向上滚动几行到`Model`的定义：

```py
table Model {
```

这告诉我们`Model`是FlatBuffers称为`table`的内容。您可以将其视为Python中的`Dict`或C或C++中的`struct`（尽管它比这更灵活）。它定义了对象可以具有的属性，以及它们的名称和类型。FlatBuffers中还有一种不太灵活的类型称为`struct`，对于对象数组更节省内存，但我们目前在TensorFlow Lite中没有使用这种类型。

您可以通过查看[`micro_speech`示例的`main()`函数](https://oreil.ly/StkFf)来了解实际应用中如何使用这个功能：

```py
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(g_tiny_conv_micro_features_model_data);
```

`g_tiny_conv_micro_features_model_data`变量是指向包含序列化TensorFlow Lite模型的内存区域的指针，而对`::tflite::GetModel()`的调用实际上只是一个转换，以获取由该底层内存支持的C++对象。它不需要任何内存分配或遍历数据结构，因此这是一个非常快速和高效的调用。要理解我们如何使用它，请看我们在数据结构上执行的下一个操作：

```py
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }
```

如果您查看[模式中`Model`定义的开始](https://oreil.ly/vPpDw)，您可以看到此代码所引用的`version`属性的定义：

```py
  // Version of the schema.
  version:uint;
```

这告诉我们`version`属性是一个32位无符号整数，因此为`model->version()`生成的C++代码返回该类型的值。在这里，我们只是进行错误检查，以确保版本是我们可以理解的版本，但对于模式中定义的所有属性，都生成了相同类型的访问函数。

要了解文件格式的更复杂部分，值得跟随`MicroInterpreter`类的流程，因为它加载模型并准备执行。构造函数接收一个指向内存中模型的指针，例如前面示例中的`g_tiny_conv_micro_features_model_data`。它访问的第一个属性是[缓冲区](https://oreil.ly/nQjwY)：

```py
  const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers =
      model->buffers();
```

您可能会在类型定义中看到`Vector`名称，并担心我们试图在嵌入式环境中使用类似于标准模板库（STL）类型的对象，而不需要动态内存管理，这将是一个坏主意。然而，幸运的是，FlatBuffers的`Vector`类只是对底层内存的只读包装器，因此就像根`Model`对象一样，创建它不需要解析或内存分配。

要了解`buffers`数组代表的更多信息，值得查看[模式定义](https://oreil.ly/QOTlY)：

```py
// Table of raw data buffers (used for constant tensors). Referenced by tensors
// by index. The generous alignment accommodates mmap-friendly data structures.
table Buffer {
  data:[ubyte] (force_align: 16);
}
```

每个缓冲区都被定义为一个无符号8位值的原始数组，在内存中第一个值是16字节对齐的。这是用于图中所有权重数组（和任何其他常量值）的容器类型。张量的类型和形状是分开保存的；这个数组只是保存了数组内部数据的原始字节。操作通过在顶层向量内部的索引引用这些常量缓冲区。

我们访问的下一个属性是[子图列表](https://oreil.ly/9Fa9V)：

```py
  auto* subgraphs = model->subgraphs();
  if (subgraphs->size() != 1) {
    error_reporter->Report("Only 1 subgraph is currently supported.\n");
    initialization_status_ = kTfLiteError;
    return;
  }
  subgraph_ = (*subgraphs)[0];
```

子图是一组操作符、它们之间的连接以及它们使用的缓冲区、输入和输出。未来可能需要多个子图来支持一些高级模型，例如支持控制流，但目前我们想要在微控制器上支持的所有网络都有一个单独的子图，因此我们可以通过确保当前模型满足该要求来简化后续的代码。要了解子图中的内容，我们可以回顾一下[模式](https://oreil.ly/Z9mLn)：

```py
// The root type, defining a subgraph, which typically represents an entire
// model.
table SubGraph {
  // A list of all tensors used in this subgraph.
  tensors:[Tensor];

  // Indices of the tensors that are inputs into this subgraph. Note this is
  // the list of non-static tensors that feed into the subgraph for inference.
  inputs:[int];

  // Indices of the tensors that are outputs out of this subgraph. Note this is
  // the list of output tensors that are considered the product of the
  // subgraph's inference.
  outputs:[int];

  // All operators, in execution order.
  operators:[Operator];

  // Name of this subgraph (used for debugging).
  name:string;
}
```

每个子图的第一个属性是张量列表，`MicroInterpreter`代码访问它[如此](https://oreil.ly/EsO7M)：

```py
  tensors_ = subgraph_->tensors();
```

正如我们之前提到的，`Buffer`对象只保存权重的原始值，没有关于它们类型或形状的任何元数据。张量是存储常量缓冲区的额外信息的地方。它们还为临时数组（如输入、输出或激活层）保存相同的信息。您可以在它们的定义中看到这些元数据[在模式文件的顶部附近](https://oreil.ly/mH0IL)：

```py
table Tensor {
  // The tensor shape. The meaning of each entry is operator-specific but
  // builtin ops use: [batch size, height, width, number of channels] (That's
  // Tensorflow's NHWC).
  shape:[int];
  type:TensorType;
  // An index that refers to the buffers table at the root of the model. Or,
  // if there is no data buffer associated (i.e. intermediate results), then
  // this is 0 (which refers to an always existent empty buffer).
  //
  // The data_buffer itself is an opaque container, with the assumption that the
  // target device is little-endian. In addition, all builtin operators assume
  // the memory is ordered such that if `shape` is [4, 3, 2], then index
  // [i, j, k] maps to data_buffer[i*3*2 + j*2 + k].
  buffer:uint;
  name:string;  // For debugging and importing back into tensorflow.
  quantization:QuantizationParameters;  // Optional.

  is_variable:bool = false;
}
```

`shape`是一个简单的整数列表，指示张量的维度，而`type`是一个枚举，映射到TensorFlow Lite支持的可能数据类型。`buffer`属性指示根级列表中的哪个`Buffer`具有支持此张量的实际值，如果它是从文件中读取的常量，则为零，如果值是动态计算的（例如激活层），则为零。`name`只是为张量提供一个可读的标签，有助于调试，`quantization`属性定义了如何将低精度值映射到实数。最后，`is_variable`成员用于支持未来的训练和其他高级应用，但在微控制器单元（MCU）上不需要使用。

回到`MicroInterpreter`代码，我们从子图中提取的第二个主要属性是[操作符列表](https://oreil.ly/6Yl8d)：

```py
operators_ = subgraph_->operators();
```

这个列表保存了模型的图结构。要了解这是如何编码的，我们可以回到[`Operator`的模式定义](https://oreil.ly/xTs7j)：

```py
// An operator takes tensors as inputs and outputs. The type of operation being
// performed is determined by an index into the list of valid OperatorCodes,
// while the specifics of each operations is configured using builtin_options
// or custom_options.
table Operator {
  // Index into the operator_codes array. Using an integer here avoids
  // complicate map lookups.
  opcode_index:uint;

  // Optional input and output tensors are indicated by -1.
  inputs:[int];
  outputs:[int];

  builtin_options:BuiltinOptions;
  custom_options:[ubyte];
  custom_options_format:CustomOptionsFormat;

  // A list of booleans indicating the input tensors which are being mutated by
  // this operator.(e.g. used by RNN and LSTM).
  // For example, if the "inputs" array refers to 5 tensors and the second and
  // fifth are mutable variables, then this list will contain
  // [false, true, false, false, true].
  //
  // If the list is empty, no variable is mutated in this operator.
  // The list either has the same length as `inputs`, or is empty.
  mutating_variable_inputs:[bool];
}
```

`opcode_index`成员是`Model`内部`operator_codes`向量中的索引。因为特定类型的操作符，比如`Conv2D`，可能在一个图中出现多次，而且一些操作需要一个字符串来定义它们，所以将所有操作定义保存在一个顶层数组中，并从子图间接引用它们可以节省序列化大小。

`inputs`和`outputs`数组定义了操作符与图中邻居之间的连接。这些是整数列表，指的是父子图中的张量数组，可能指的是从模型中读取的常量缓冲区，应用程序输入的输入，运行其他操作的结果，或者在计算完成后将被应用程序读取的输出目标缓冲区。

关于子图中保存的操作符列表的一个重要事项是，它们总是按照拓扑顺序排列，这样如果你从数组的开头执行它们到结尾，所有依赖于先前操作的给定操作的输入在到达该操作时都已经计算完成。这使得编写解释器变得更简单，因为执行循环不需要在执行之前执行任何图操作，只需按照它们列出的顺序执行操作。这意味着以不同顺序运行相同的子图（例如，使用反向传播进行训练）并不简单，但TensorFlow Lite的重点是推断，因此这是一个值得的权衡。

操作符通常还需要参数，比如`Conv2D`内核的滤波器的形状和步幅。这些参数的表示非常复杂。出于历史原因，TensorFlow Lite支持两种不同的操作族。内置操作首先出现，是移动应用程序中最常用的操作。您可以在[模式中](https://oreil.ly/HjdHn)看到一个列表。截至2019年11月，只有122个，但TensorFlow支持超过800个操作——那么我们该怎么处理剩下的操作呢？自定义操作由字符串名称定义，而不是像内置操作那样的固定枚举，因此可以更容易地添加而不影响模式。

对于内置操作，参数结构在模式中列出。以下是`Conv2D`的示例：

```py
table Conv2DOptions {
  padding:Padding;
  stride_w:int;
  stride_h:int;
  fused_activation_function:ActivationFunctionType;
  dilation_w_factor:int = 1;
  dilation_h_factor:int = 1;
}
```

希望列出的大多数成员看起来都有些熟悉，它们的访问方式与其他FlatBuffers对象相同：通过每个`Operator`对象的`builtin_options`联合体，根据操作符代码选择适当的类型（尽管这样做的代码基于[一个庞大的`switch`语句](https://oreil.ly/SkzaA)）。

如果操作符代码表明是自定义操作符，我们事先不知道参数列表的结构，因此无法生成代码对象。相反，参数信息被打包到[FlexBuffer](https://oreil.ly/qPwo9)中。这是FlatBuffer库提供的一种格式，用于在不事先知道结构的情况下编码任意数据，这意味着实现操作符的代码需要访问生成的数据，指定类型是什么，并且语法比内置操作符更混乱。以下是[一些目标检测代码](https://oreil.ly/xQoTR)的示例：

```py
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  op_data->max_detections = m["max_detections"].AsInt32();
```

在这个示例中被引用的缓冲指针最终来自`Operator`表的`custom_options`成员，展示了如何从这个属性访问参数数据。

`Operator`的最后一个成员是`mutating_variable_inputs`。这是一个实验性功能，用于帮助管理长短期记忆（LSTM）和其他可能希望将其输入视为变量的操作，对于大多数MCU应用程序来说并不相关。

这些是TensorFlow Lite序列化格式的关键部分。还有一些我们没有涵盖的成员（比如`Model`中的`metadata_buffer`），但这些是可选的非必要功能，通常可以忽略。希望这个概述足以让您开始阅读、编写和调试自己的模型文件。

# 将TensorFlow Lite移动操作移植到Micro

在主要的TensorFlow Lite版本中，针对移动设备有一百多个“内置”操作。微控制器的TensorFlow Lite重用了大部分代码，但是因为这些操作的默认实现引入了像pthread、动态内存分配或其他嵌入式系统不支持的功能，因此操作实现（也称为内核）需要一些工作才能在Micro上使用。

最终，我们希望统一两个op实现的分支，但这需要在整个框架中进行一些设计和API更改，因此短期内不会发生。大多数操作应该已经有Micro实现，但如果您发现一个在移动TensorFlow Lite上可用但在嵌入式版本中不可用的操作，本节将指导您完成转换过程。在确定要移植的操作后，有几个阶段。

## 分离参考代码

所有列出的操作应该已经有参考代码，但这些函数可能在[*reference_ops.h*](https://oreil.ly/QmW4H)中。这是一个几乎有5000行长的单片头文件。因为它涵盖了这么多操作，它引入了许多在嵌入式平台上不可用的依赖项。要开始移植过程，您首先需要将所需操作的参考函数提取到单独的头文件中。您可以在[*https://oreil.ly/*](https://oreil.ly/)*vH-6[_conv.h*]和[*pooling.h*](https://oreil.ly/pwP_0)中看到这些较小头文件的示例。参考函数本身应该与它们实现的操作名称匹配，并且通常会有多个不同数据类型的实现，有时使用模板。

一旦文件从较大的头文件中分离出来，您需要从*reference_ops.h*中包含它，以便所有使用该头文件的现有用户仍然看到您移动的函数（尽管我们的Micro代码将单独包含分离的头文件）。您可以查看我们如何为`conv2d` [这里](https://oreil.ly/jtXLU)。您还需要将头文件添加到`kernels/internal/BUILD:reference_base`和`kernels/internal/BUILD:legacy_reference_base`构建规则中。在进行这些更改后，您应该能够运行测试套件并看到所有现有的移动测试都通过了：

```py
bazel test tensorflow/lite/kernels:all
```

这是一个创建初始拉取请求供审查的好时机。您尚未将任何内容移植到`micro`分支，但您已经为更改准备好了现有代码，因此值得尝试在您继续以下步骤时进行审查和提交。

## 创建运算符的微型副本

每个微操作符实现都是移动版本的修改副本，保存在*tensorflow/lite/kernels/*中。例如，微*conv.cc*基于移动*conv.cc*。有一些重要的区别。首先，在嵌入式环境中动态内存分配更加棘手，因此为了在推理期间使用的计算中缓存计算值，OpData结构的创建被移动到一个单独的函数中，以便它可以在`Invoke()`期间调用，而不是从`Prepare()`返回。这对每个`Invoke()`调用需要更多的工作，但通常减少内存开销对于微控制器是有意义的。

其次，在`Prepare()`中的大部分参数检查代码通常会被删除。最好将其封装在`#if defined(DEBUG)`中，而不是完全删除，但删除可以将代码大小保持最小。应从包含和代码中删除对外部框架（`Eigen`，`gemmlowp`，`cpu_backend_support`）的所有引用。在`Eval()`函数中，除了调用`reference_ops::`命名空间中的函数的路径之外，应删除其他内容。

修改后的运算符实现应保存在与移动版本相同名称的文件中（通常是运算符名称的小写版本），保存在*tensorflow/lite/micro/kernels/*文件夹中。

## 将测试移植到微框架

我们无法在嵌入式平台上运行完整的Google Test框架，因此需要使用Micro Test库。这对于GTest的用户应该很熟悉，但它避免了任何需要动态内存分配或C++全局初始化的构造。本书的其他地方有更多文档。

您需要在嵌入式环境中运行与移动端相同的测试，因此您需要使用*tensorflow/lite/kernels/<`your op name`>_test.cc*中的版本作为起点。例如，查看[*tensorflow/lite/kernels/conv_test.cc*](https://oreil.ly/76KXK)和移植版本[*tensorflow/lite/micro/kernels/conv_test.cc*](https://oreil.ly/r1wKh)。以下是主要区别：

+   移动代码依赖于C++ STL类，如`std::map`和`std::vector`，这些类需要动态内存分配。

+   移动代码还使用辅助类，并以涉及分配的方式传递数据对象。

+   微版本在堆栈上分配所有数据，使用`std::initializer_list`传递类似于`std::vectors`的对象，但不需要动态内存分配。

+   运行测试的调用表示为函数调用，而不是对象分配，因为这有助于重用大量代码而不会遇到分配问题。

+   大多数标准错误检查宏都可用，但带有`TF_LITE_MICRO_`后缀。例如，`EXPECT_EQ`变为`TF_LITE_MICRO_EXPECT_EQ`。

所有测试都必须位于一个文件中，并被单个`TF_LITE_MICRO_TESTS_BEGIN/TF_LITE_MICRO_TESTS_END`对包围。在底层，这实际上创建了一个`main()`函数，以便可以将测试作为独立的二进制运行。

我们还尽量确保测试仅依赖于内核代码和API，而不引入其他类，如解释器。测试应直接调用内核实现，使用从`GetRegistration()`返回的C API。这是因为我们希望确保内核可以完全独立使用，而不需要框架的其余部分，因此测试代码也应避免这些依赖关系。

## 构建一个Bazel测试

现在您已经创建了运算符实现和测试文件，您需要检查它们是否有效。您需要使用Bazel开源构建系统来执行此操作。在[*BUILD*文件](https://oreil.ly/CbwMI)中添加一个`tflite_micro_cc_test`规则，然后尝试构建和运行以下命令行（将`conv`替换为您的运算符名称）：

```py
bazel test ttensorflow/lite/micro/kernels:conv_test --test_output=streamed
```

毫无疑问会出现编译错误和测试失败，因此需要花费一些时间来迭代修复这些问题。

## 将您的运算符添加到AllOpsResolver

应用程序可以选择仅拉取某些运算符实现，以减小二进制大小，但有一个运算符解析器会拉取所有可用的运算符，以便轻松入门。您应该在[*all_ops_resolver.cc*](https://oreil.ly/0Nq06)的构造函数中添加一个调用来注册您的运算符实现，并确保实现和头文件也包含在*BUILD*规则中。

## 构建一个Makefile测试

到目前为止，您所做的一切都在TensorFlow Lite的`micro`分支中进行，但您一直在x86上构建和测试。这是开发的最简单方式，最初的任务是创建所有操作的可移植、未优化的实现，因此我们建议您尽可能多地在这个领域进行工作。不过，到了这一点，您应该已经在桌面Linux上完全运行和测试了操作员实现，所以现在是时候开始在嵌入式设备上进行编译和测试了。

Google开源项目的标准构建系统是Bazel，但不幸的是，使用Bazel实现交叉编译和支持嵌入式工具链并不容易，因此我们不得不转向备受尊敬的Make进行部署。Makefile本身在内部非常复杂，但希望您的新操作员应该会根据其实现文件和测试的名称和位置自动选择。唯一的手动步骤应该是将您创建的参考头文件添加到`MICROLITE_CC_HDRS`文件列表中。

要在这种环境中测试您的操作员，请`cd`到文件夹，并运行以下命令（将您自己的操作员名称替换为`conv`）：

```py
make -f tensorflow/lite/micro/tools/make/Makefile test_conv_test
```

希望这次编译和测试能够通过。如果没有通过，请按照正常的调试程序来找出问题所在。

这仍然在您本地的Intel x86桌面机上本地运行，尽管它使用与嵌入式目标相同的构建机制。您现在可以尝试将代码编译并刷写到像SparkFun Edge这样的真实微控制器上（只需在Makefile行中传入`TARGET=sparkfun_edge`即可），但为了让生活更轻松，我们还提供了Cortex-M3设备的软件仿真。您应该能够通过执行以下命令来运行您的测试：

```py
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=bluepill test_conv_test
```

这可能有点不稳定，因为有时仿真器执行时间太长，进程会超时，但希望再试一次会解决问题。如果您已经走到这一步，我们鼓励您尽可能将您的更改贡献回开源构建。开源您的代码的完整过程可能有点复杂，但[ TensorFlow社区指南](https://oreil.ly/YcbFB)是一个很好的起点。

# 总结

完成本章后，您可能感觉自己像是在尝试从消防栓中喝水。我们为您提供了关于TensorFlow Lite for Microcontrollers如何工作的大量信息。如果您不理解全部内容，甚至大部分内容也不用担心，我们只是想给您足够的背景知识，以便在需要深入了解时知道从哪里开始查找。代码都是开源的，是了解框架运作方式的终极指南，但我们希望这些评论能帮助您理解其结构，并理解为什么会做出某些设计决策。

在看完如何运行一些预构建示例并深入了解库的工作原理后，您可能想知道如何将所学到的应用到自己的应用程序中。本书的剩余部分将集中讨论您需要掌握的技能，以便在自己的产品中部署自定义机器学习，涵盖优化、调试和移植模型，以及隐私和安全性。
