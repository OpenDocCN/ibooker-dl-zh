# 第17章。优化模型和二进制大小

无论您选择哪种平台，闪存存储和RAM都可能非常有限。大多数嵌入式系统的闪存只读存储器少于1 MB，许多只有几十KB。内存也是如此：很少有超过512 KB的静态RAM（SRAM）可用，而在低端设备上，这个数字可能只有几个KB。好消息是，TensorFlow Lite for Microcontrollers被设计为可以使用至少20 KB的闪存和4 KB的SRAM，但您需要仔细设计您的应用程序并做出工程权衡以保持占用空间较小。本章介绍了一些方法，您可以使用这些方法来监控和控制内存和存储需求。

# 了解系统的限制

大多数嵌入式系统具有一种架构，其中程序和其他只读数据存储在闪存存储器中，仅在上传新可执行文件时才写入。通常还有可修改的内存可用，通常使用SRAM技术。这是用于较大CPU缓存的相同技术，它提供快速访问和低功耗，但尺寸有限。更先进的微控制器可以提供第二层可修改内存，使用更耗电但可扩展的技术，如动态RAM（DRAM）。

您需要了解潜在平台提供的内容以及权衡。例如，具有大量二级DRAM的芯片可能因其灵活性而具有吸引力，但如果启用额外的内存超出了您的功耗预算，那可能不值得。如果您正在操作本书关注的1 mW及以下功率范围，通常不可能使用超出SRAM的任何东西，因为更大的内存方法将消耗太多能量。这意味着您需要考虑的两个关键指标是可用的闪存只读存储器量和可用的SRAM量。这些数字应列在您查看的任何芯片的描述中。希望您甚至不需要深入挖掘数据表[“硬件选择”](ch16.xhtml#hard_choice)。

# 估算内存使用量

当您了解硬件选项时，您需要了解软件将需要的资源以及您可以做出的权衡来控制这些要求。

## 闪存使用量

通常，通过编译完整的可执行文件，然后查看生成图像的大小，您可以确定在闪存中需要多少空间。这可能会令人困惑，因为链接器生成的第一个工件通常是带有调试符号和部分信息的可执行文件的注释版本，格式类似于ELF（我们在[“测量代码大小”](#measuring_code_size)中更详细地讨论）。您要查看的文件是实际上刷入设备的文件，通常由`objcopy`等工具生成。用于估算所需闪存内存量的最简单方程式是以下因素之和：

操作系统大小

如果您使用任何类型的实时操作系统（RTOS），则需要在可执行文件中留出空间来保存其代码。这通常可以根据您使用的功能进行配置，并且估算占用空间的最简单方法是使用所需功能构建一个示例“hello world”程序。如果查看图像文件大小，这将为您提供OS程序代码有多大的基准。可能占用大量程序空间的典型模块包括USB、WiFi、蓝牙和蜂窝无线电堆栈，因此请确保启用它们，如果您打算使用它们。

TensorFlow Lite for Microcontrollers代码大小

机器学习框架需要空间来加载和执行神经网络模型的程序逻辑，包括运行核心算术的操作实现。本章后面我们将讨论如何配置框架以减小特定应用程序的大小，但首先只需编译一个标准单元测试（比如[`micro_speech`测试](https://oreil.ly/7cafy)），其中包括框架，并查看估计的结果图像大小。

模型数据大小

如果您还没有训练好的模型，可以通过计算其权重来估计它所需的闪存存储空间。例如，全连接层的权重数量等于其输入向量的大小乘以其输出向量的大小。对于卷积层，情况会更复杂一些；您需要将滤波框的宽度和高度乘以输入通道的数量，然后乘以滤波器的数量。您还需要为与每一层相关的任何偏置向量添加存储空间。这很快就会变得复杂，因此最简单的方法可能是在TensorFlow中创建一个候选模型，然后将其导出为TensorFlow Lite文件。该文件将直接映射到闪存中，因此其大小将为您提供占用多少空间的确切数字。您还可以查看[Keras的`model.summary()`方法](https://keras.io/models/about-keras-models)列出的权重数量。

###### 注意

我们在[第4章](ch04.xhtml#chapter_hello_world_training)中介绍了量化，并在[第15章](ch15.xhtml#optimizing_latency)中进一步讨论了它，但在模型大小的背景下进行一个快速的复习是值得的。在训练期间，权重通常以浮点值存储，每个占用4个字节的内存。由于空间对于移动和嵌入式设备来说是一个限制，TensorFlow Lite支持将这些值压缩到一个字节中，这个过程称为*量化*。它通过跟踪存储在浮点数组中的最小值和最大值，然后将所有值线性转换为该范围内均匀间隔的256个值中最接近的一个。这些代码都存储在一个字节中，可以对它们进行算术运算而几乎不损失精度。

应用程序代码大小

您需要编写代码来访问传感器数据，对其进行预处理以准备神经网络，并响应结果。您可能还需要一些其他类型的用户界面和机器学习模块之外的业务逻辑。这可能很难估计，但您至少应该尝试了解是否需要任何外部库（例如用于快速傅立叶变换），并计算它们的代码空间需求。

## RAM使用量

确定所需的可修改内存量可能比理解存储需求更具挑战性，因为程序的RAM使用量会随着程序的生命周期而变化。类似于估计闪存需求的过程，您需要查看软件的不同层以估计整体使用要求：

操作系统大小

大多数[RTOS（如FreeRTOS）](https://www.freertos.org/FAQMem.html)记录了它们不同配置选项所需的RAM量，您应该能够使用这些信息来规划所需的大小。您需要注意可能需要缓冲区的模块，特别是通信堆栈如TCP/IP、WiFi或蓝牙。这些将需要添加到任何核心操作系统要求中。

微控制器的TensorFlow Lite RAM大小

ML框架的核心运行时不需要大量内存，并且其数据结构在SRAM中不应该需要超过几千字节的空间。这些分配为解释器使用的类的一部分，因此您的应用程序代码是将这些创建为全局或局部对象将决定它们是在堆栈上还是在一般内存中。我们通常建议将它们创建为全局或`static`对象，因为空间不足通常会导致链接时错误，而堆栈分配的局部变量可能会导致更难理解的运行时崩溃。

模型内存大小

当神经网络执行时，一个层的结果被馈送到后续操作中，因此必须保留一段时间。这些激活层的寿命因其在图中的位置而异，每个激活层所需的内存大小由层写出的数组的形状控制。这些变化意味着需要随时间计划以将所有这些临时缓冲区尽可能地放入内存的小区域中。目前，这是在解释器首次加载模型时完成的，因此如果竞技场不够大，您将在控制台上看到错误。如果您在错误消息中看到可用内存与所需内存之间的差异，并将竞技场增加该数量，您应该能够解决该错误。

应用程序内存大小

与程序大小一样，应用程序逻辑的内存使用在编写之前可能很难计算。但是，您可以对内存的更大使用者进行一些猜测，例如您将需要用于存储传入样本数据的缓冲区，或者库将需要用于预处理的内存区域。

# 不同问题上模型准确性和大小的大致数字

了解不同类型问题的当前技术水平将有助于您规划您的应用程序可能实现的目标。机器学习并非魔法，了解其局限性将有助于您在构建产品时做出明智的权衡。[第14章](ch14.xhtml#designing_your_own_tinyml_applications)探讨了设计过程，是开始培养直觉的好地方，但您还需要考虑随着模型被迫适应严格资源限制时准确性如何下降。为了帮助您，这里有一些为嵌入式系统设计的架构示例。如果其中一个接近您需要做的事情，可能会帮助您设想在模型创建过程结束时可能实现的结果。显然，您的实际结果将在很大程度上取决于您的具体产品和环境，因此请将这些作为规划的指导，并不要依赖于能够实现完全相同的性能。

## 语音唤醒词模型

我们之前提到的使用400,000次算术运算的小型（18 KB）模型作为代码示例，能够在区分四类声音时达到85%的一级准确性（参见[“建立度量”](ch18.xhtml#establish_a_metric)）。这是训练评估指标，这意味着通过呈现一秒钟的片段并要求模型对其输入进行一次分类来获得结果。在实践中，您通常会在流式音频上使用模型，根据逐渐向前移动的一秒钟窗口重复预测结果，因此在实际应用中的实际准确性低于该数字可能表明的准确性。您通常应该将这种大小的音频模型视为更大处理级联中的第一阶段门卫，以便更复杂的模型可以容忍和处理其错误。

作为一个经验法则，您可能需要一个具有300到400 KB权重和数千万算术操作的模型，才能以足够可接受的准确性检测唤醒词，以在语音界面中使用。不幸的是，您还需要一个商业质量的数据集进行训练，因为目前仍然没有足够的开放标记语音数据库可用，但希望这种限制随着时间的推移会减轻。

## 加速度计预测性维护模型

有各种不同的预测性维护问题，但其中一个较简单的情况是检测电机轴承故障。这通常表现为加速度计数据中可以看到的明显震动模式。一个合理的模型来识别这些模式可能只需要几千个权重，使其大小不到10 KB，并且数十万个算术操作。您可以期望使用这样的模型对这些事件进行分类的准确率超过95％，并且您可以想象从那里增加模型的复杂性来处理更困难的问题（例如检测具有许多移动部件或自行移动的机器上的故障）。当然，参数和操作的数量也会相应增加。

## 人员存在检测

计算机视觉在嵌入式平台上并不是常见的任务，因此我们仍在探索哪些应用是有意义的。我们听到的一个常见请求是能够检测到附近有人时，唤醒用户界面或执行其他更耗电的处理，这是不可能一直运行的。我们试图在[Visual Wake Word Challenge](https://oreil.ly/E8GoU)中正式捕捉这个问题的要求，结果显示，如果使用一个250 KB模型和大约6000万算术操作，您可以期望在一个小（96×96像素）单色图像的二进制分类中获得大约90%的准确性。这是使用缩减版MobileNet v2架构的基线（如本书中早期描述的），因此我们希望随着更多研究人员解决这一特殊需求集，准确性会提高，但它给出了您在微控制器内存占用中可能在视觉问题上表现如何的粗略估计。您可能会想知道这样一个小模型在流行的ImageNet-1000类别问题上会表现如何 - 很难说确切的原因是最终的全连接层对于一千个类别很快就会占用一百多千字节（参数数量是嵌入输入乘以类别计数），但对于大约500 KB的总大小，您可以期望在top-one准确性方面达到大约50%。

# 模型选择

在优化模型和二进制大小方面，我们强烈建议从现有模型开始。正如我们在[第14章](ch14.xhtml#designing_your_own_tinyml_applications)中讨论的那样，投资最有价值的领域是数据收集和改进，而不是调整架构，从已知模型开始将让您尽早专注于数据改进。嵌入式平台上的机器学习软件也仍处于早期阶段，因此使用现有模型增加了其操作在您关心的设备上得到支持和优化的机会。我们希望本书附带的代码示例将成为许多不同应用的良好起点 - 我们选择它们以涵盖尽可能多种不同类型的传感器输入，但如果它们不适合您的用例，您可能可以在线搜索一些替代方案。如果找不到适合的大小优化架构，您可以尝试在TensorFlow的训练环境中从头开始构建自己的架构，但正如[第13章](ch13.xhtml#chapter_tensorflow_lite_for_microcontrollers)和[第19章](ch19.xhtml#ch19)讨论的那样，成功将其移植到微控制器可能是一个复杂的过程。

# 减小可执行文件的大小

您的模型可能是微控制器应用程序中只读内存的最大消耗者之一，但您还必须考虑编译代码占用了多少空间。代码大小的限制是我们在针对嵌入式平台时不能只使用未经修改的TensorFlow Lite的原因：它将占用数百KB的闪存内存。TensorFlow Lite for Microcontrollers可以缩减至至少20 KB，但这可能需要您进行一些更改，以排除您的应用程序不需要的代码部分。

## 测量代码大小

在开始优化代码大小之前，您需要知道它有多大。在嵌入式平台上，这可能有点棘手，因为构建过程的输出通常是一个文件，其中包含调试和其他信息，这些信息不会传输到嵌入式设备上，因此不应计入总大小限制。在Arm和其他现代工具链中，这通常被称为可执行和链接格式（ELF）文件，无论是否具有*.elf*后缀。如果您在Linux或macOS开发机器上，可以运行`file`命令来调查您的工具链的输出；它将向您显示文件是否为ELF。

查看的更好文件通常被称为*bin*：实际上传到嵌入式设备的闪存存储的代码二进制快照。这通常会完全等于将要使用的只读闪存内存的大小，因此您可以使用它来了解实际使用情况。您可以通过在主机上使用`ls -l`或`dir`之类的命令行，甚至在GUI文件查看器中检查它来找出其大小。并非所有工具链都会自动显示这个*bin*文件，它可能没有任何后缀，但它是您通过USB在Mbed上下载并拖放到设备上的文件，并且使用gcc工具链可以通过运行类似`arm-none-eabi-objcopy app.elf app.bin -O binary`来生成它。查看*.o*中间文件或甚至构建过程生成的*.a*库并不有用，因为它们包含了许多元数据，这些元数据不会出现在最终代码占用空间中，并且很多代码可能会被修剪为未使用。

因为我们期望您将模型编译为可执行文件中的C数据数组（因为您不能依赖存在文件系统来加载它），所以包括模型的任何程序的二进制大小将包含模型数据。要了解实际代码占用了多少空间，您需要从二进制文件长度中减去这个模型大小。模型大小通常应在包含C数据数组的文件中定义（比如在[*tiny_conv_micro_features_model_data.cc*](https://oreil.ly/Vknl2)的末尾），因此您可以从二进制文件大小中减去它以了解真实的代码占用空间。

## Tensorflow Lite for Microcontrollers占用了多少空间？

当您了解整个应用程序的代码占用空间大小时，您可能想要调查TensorFlow Lite占用了多少空间。测试这一点的最简单方法是注释掉所有对框架的调用（包括创建`OpResolvers`和解释器等对象），看看二进制文件变小了多少。您应该至少期望减少20到30 KB，因此如果您没有看到类似的情况，您应该再次检查是否捕捉到了所有引用。这应该有效，因为链接器将剥离您从未调用的任何代码，将其从占用空间中删除。这也可以扩展到代码的其他模块，只要确保没有引用，以帮助更好地了解空间的去向。

## OpResolver

TensorFlow Lite支持100多种操作，但在单个模型中不太可能需要所有这些操作。每个操作的单独实现可能只占用几千字节，但随着这么多可用的操作，总量很快就会增加。幸运的是，有一种内置机制可以去除你不需要的操作的代码占用空间。

当TensorFlow Lite加载模型时，它会使用[`OpResolver`接口](https://oreil.ly/dfwOP)来搜索每个包含的操作的实现。这是一个你传递给解释器以加载模型的类，它包含了查找函数指针以获取操作实现的逻辑，给定操作定义。存在这个的原因是为了让你可以控制哪些实现实际上被链接进来。在大多数示例代码中，你会看到我们正在创建并传递一个[`AllOpsResolver`类的实例](https://oreil.ly/tbzg6)。正如我们在[第5章](ch05.xhtml#chapter_building_an_application)中讨论的那样，这实现了`OpResolver`接口，正如其名称所示，它为TensorFlow Lite for Microcontrollers中支持的每个操作都有一个条目。这对于入门很方便，因为这意味着你可以加载任何支持的模型，而不必担心它包含哪些操作。

然而，当你开始担心代码大小时，你会想要重新审视这个类。在你的应用程序主循环中，不要再传递`AllOpsResolver`的实例，而是将*all_ops_resolver.cc*和*.h*文件复制到你的应用程序中，并将它们重命名为*my_app_resolver.cc*和*.h*，类重命名为`MyAppResolver`。在你的类构造函数中，删除所有适用于你模型中不使用的操作的`AddBuiltin()`调用。不幸的是，我们不知道有一种简单的自动方式来创建模型使用的操作列表，但[Netron](https://oreil.ly/MKqF9)模型查看器是一个可以帮助这个过程的好工具。

确保你用`MyAppResolver`替换你传递给解释器的`AllOpsResolver`实例。现在，一旦编译你的应用程序，你应该会看到大小明显缩小。这个改变背后的原因是，大多数链接器会自动尝试删除不能被调用的代码（或*死代码*）。通过删除`AllOpsResolver`中的引用，你允许链接器确定可以排除所有不再列出的操作实现。

如果你只使用了少数操作，你不需要像我们使用大型`AllOpsResolver`那样将注册包装在一个新类中。相反，你可以创建一个`MicroMutableOpResolver`类的实例，并直接添加你需要的操作注册。`MicroMutableOpResolver`实现了`OpResolver`接口，但有额外的方法让你添加操作到列表中（这就是为什么它被命名为`Mutable`）。这是用来实现`AllOpsResolver`的类，也是你自己的解析器类的一个很好的基础，但直接调用它可能更简单。我们在一些示例中使用了这种方法，你可以在这个来自[`micro_speech`示例](https://oreil.ly/gdZts)的片段中看到它是如何工作的：

```py
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_FULLY_CONNECTED,
      tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                       tflite::ops::micro::Register_SOFTMAX());
```

你可能会注意到我们将解析器对象声明为`static`。这是因为解释器可以随时调用它，所以它的生命周期至少需要与我们为解释器创建的对象一样长。

## 理解单个函数的大小

如果你使用GCC工具链，你可以使用像`nm`这样的工具来获取目标（*.o*）中间文件中函数和对象的大小信息。这里有一个构建二进制文件然后检查编译后的*audio_provider.cc*对象文件中项目大小的示例：

```py
nm -S tensorflow/lite/micro/tools/make/gen/ \
  sparkfun_edge_cortex-m4/obj/tensorflow/lite/micro/ \
  examples/micro_speech/sparkfun_edge/audio_provider.o
```

你应该会看到类似以下的结果：

```py
00000140 t $d
00000258 t $d
00000088 t $d
00000008 t $d
00000000 b $d
00000000 b $d
00000000 b $d
00000000 b $d
00000000 b $d
00000000 b $d
00000000 b $d
00000000 b $d
00000000 b $d
00000000 b $d
00000000 b $d
00000000 b $d
00000000 r $d
00000000 r $d
00000000 t $t
00000000 t $t
00000000 t $t
00000000 t $t
00000001 00000178 T am_adc_isr
         U am_hal_adc_configure
         U am_hal_adc_configure_dma
         U am_hal_adc_configure_slot
         U am_hal_adc_enable
         U am_hal_adc_initialize
         U am_hal_adc_interrupt_clear
         U am_hal_adc_interrupt_enable
         U am_hal_adc_interrupt_status
         U am_hal_adc_power_control
         U am_hal_adc_sw_trigger
         U am_hal_burst_mode_enable
         U am_hal_burst_mode_initialize
         U am_hal_cachectrl_config
         U am_hal_cachectrl_defaults
         U am_hal_cachectrl_enable
         U am_hal_clkgen_control
         U am_hal_ctimer_adc_trigger_enable
         U am_hal_ctimer_config_single
         U am_hal_ctimer_int_enable
         U am_hal_ctimer_period_set
         U am_hal_ctimer_start
         U am_hal_gpio_pinconfig
         U am_hal_interrupt_master_enable
         U g_AM_HAL_GPIO_OUTPUT_12
00000001 0000009c T _Z15GetAudioSamplesPN6tflite13ErrorReporterEiiPiPPs
00000001 000002c4 T _Z18InitAudioRecordingPN6tflite13ErrorReporterE
00000001 0000000c T _Z20LatestAudioTimestampv
00000000 00000001 b _ZN12_GLOBAL__N_115g_adc_dma_errorE
00000000 00000400 b _ZN12_GLOBAL__N_121g_audio_output_bufferE
00000000 00007d00 b _ZN12_GLOBAL__N_122g_audio_capture_bufferE
00000000 00000001 b _ZN12_GLOBAL__N_122g_is_audio_initializedE
00000000 00002000 b _ZN12_GLOBAL__N_122g_ui32ADCSampleBuffer0E
00000000 00002000 b _ZN12_GLOBAL__N_122g_ui32ADCSampleBuffer1E
00000000 00000004 b _ZN12_GLOBAL__N_123g_dma_destination_indexE
00000000 00000004 b _ZN12_GLOBAL__N_124g_adc_dma_error_reporterE
00000000 00000004 b _ZN12_GLOBAL__N_124g_latest_audio_timestampE
00000000 00000008 b _ZN12_GLOBAL__N_124g_total_samples_capturedE
00000000 00000004 b _ZN12_GLOBAL__N_128g_audio_capture_buffer_startE
00000000 00000004 b _ZN12_GLOBAL__N_1L12g_adc_handleE
         U _ZN6tflite13ErrorReporter6ReportEPKcz
```

许多这些符号是内部细节或无关紧要的，但最后几个可以识别为我们在*audio_provider.cc*中定义的函数，它们的名称被搅乱以匹配C++链接器约定。第二列显示它们的大小是多少十六进制。您可以看到`InitAudioRecording()`函数的大小为`0x2c4`或708字节，这在小型微控制器上可能相当显著，因此如果空间紧张，值得调查函数内部大小的来源。

我们发现的最佳方法是将源代码与反汇编函数混合在一起。幸运的是，`objdump`工具通过使用`-S`标志让我们可以做到这一点——但与`nm`不同，您不能使用安装在Linux或macOS桌面上的标准版本。相反，您需要使用随您的工具链一起提供的版本。如果您正在使用TensorFlow Lite for Microcontrollers的Makefile构建，通常会自动下载。它通常会存在于类似*tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin*的位置。以下是一个运行以查看*audio_provider.cc*内部函数更多信息的命令：

```py
tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/ \
  arm-none-eabi-objdump -S tensorflow/lite/micro/tools/make/gen/ \
  sparkfun_edge_cortex-m4/obj/tensorflow/lite/micro/examples/ \
  micro_speech/sparkfun_edge/audio_provider.o
```

我们不会展示所有的输出，因为太长了；相反，我们只展示一个简化版本，只显示我们感兴趣的函数：

```py
...
Disassembly of section .text._Z18InitAudioRecordingPN6tflite13ErrorReporterE:

00000000 <_Z18InitAudioRecordingPN6tflite13ErrorReporterE>:

TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
   0:	b570      	push	{r4, r5, r6, lr}
  // Set the clock frequency.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0)) {
   2:	2100      	movs	r1, #0
TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
   4:	b088      	sub	sp, #32
   6:	4604      	mov	r4, r0
      am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0)) {
   8:	4608      	mov	r0, r1
   a:	f7ff fffe 	bl	0 <am_hal_clkgen_control>
  if (AM_HAL_STATUS_SUCCESS !=
   e:	2800      	cmp	r0, #0
  10:	f040 80e1 	bne.w	1d6 <_Z18InitAudioRecordingPN6tflite13ErrorReporterE+0x1d6>
    return kTfLiteError;
  }

  // Set the default cache configuration and enable it.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_cachectrl_config(&am_hal_cachectrl_defaults)) {
  14:	4890      	ldr	r0, [pc, #576]	; (244 <am_hal_cachectrl_config+0x244>)
  16:	f7ff fffe 	bl	0 <am_hal_cachectrl_config>
  if (AM_HAL_STATUS_SUCCESS !=
  1a:	2800      	cmp	r0, #0
  1c:	f040 80d4 	bne.w	1c8 <_Z18InitAudioRecordingPN6tflite13ErrorReporterE+0x1c8>
    error_reporter->Report("Error - configuring the system cache failed.");
    return kTfLiteError;
  }
  if (AM_HAL_STATUS_SUCCESS != am_hal_cachectrl_enable()) {
  20:	f7ff fffe 	bl	0 <am_hal_cachectrl_enable>
  24:	2800      	cmp	r0, #0
  26:	f040 80dd 	bne.w	1e4 <_Z18InitAudioRecordingPN6tflite13Error\
    ReporterE+0x1e4>
...
```

您不需要理解汇编在做什么，但希望您可以看到通过查看函数大小（反汇编行最左边的数字；例如，在`InitAudioRecording()`末尾的十六进制*10*）如何随着每个C++源代码行的增加而增加。如果查看整个函数，您会发现所有的硬件初始化代码都已内联在`InitAudioRecording()`实现中，这解释了为什么它如此庞大。

## 框架常量

在库代码中有一些地方我们使用硬编码的数组大小来避免动态内存分配。如果RAM空间非常紧张，值得尝试看看是否可以减少它们以适应您的应用程序（或者，对于非常复杂的用例，甚至可能需要增加它们）。其中一个数组是[`TFLITE_REGISTRATIONS_MAX`](https://oreil.ly/hYTLi)，它控制可以注册多少不同的操作。默认值为128，这对于大多数应用程序来说可能太多了——特别是考虑到它创建了一个包含128个`TfLiteRegistration`结构的数组，每个结构至少占用32字节，需要4 KB的RAM。您还可以查看像`MicroInterpreter`中的[`kStackDataAllocatorSize`](https://oreil.ly/wIsPm)这样的较小的问题，或者尝试缩小您传递给解释器构造函数的arena的大小。

# 真正微小的模型

本章中的许多建议都与能够承受使用20 KB框架代码占用的嵌入式系统有关，以运行机器学习，并且不试图仅使用不到10 KB的RAM。如果您的设备资源约束非常严格——例如，只有几千字节的RAM或闪存，您将无法使用相同的方法。对于这些环境，您需要编写自定义代码，并非常小心地调整每个细节以减小大小。

我们希望TensorFlow Lite for Microcontrollers在这些情况下仍然有用。我们建议您仍然在TensorFlow中训练一个模型，即使它很小，然后使用导出工作流从中创建一个TensorFlow Lite模型文件。这可以作为提取权重的良好起点，并且您可以使用现有的框架代码来验证您自定义版本的结果。您正在使用的操作的参考实现也应该是您自己操作代码的良好起点；它们应该是可移植的、易于理解的，并且在内存效率方面表现良好，即使它们对延迟不是最佳的。

# 总结

在这一章中，我们看了一些最好的技术，来缩小嵌入式机器学习项目所需的存储量。这很可能是你需要克服的最艰难的限制之一，但当你拥有一个足够小、足够快、并且不消耗太多能量的应用程序时，你就有了一个明确的路径来推出你的产品。剩下的是排除所有不可避免的小精灵，它们会导致你的设备以意想不到的方式行为。调试可能是一个令人沮丧的过程（我们听说过它被描述为一场谋杀案，你是侦探、受害者和凶手），但这是一个必须学会的技能，以便将产品推向市场。[第18章](ch18.xhtml#debugging)介绍了可以帮助你理解机器学习系统发生了什么的基本技术。
